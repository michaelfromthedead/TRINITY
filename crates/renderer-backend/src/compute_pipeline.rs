//! Compute pipeline descriptor and wrapper types.
//!
//! Provides a builder-pattern API for creating wgpu compute pipelines with fluent
//! configuration and sensible defaults.
//!
//! # Architecture
//!
//! ```text
//! ComputePipelineDescriptor
//!     - Builder pattern for all wgpu ComputePipelineDescriptor fields
//!     - Module and entry point (required at construction)
//!     - Layout association (auto or explicit)
//!     - Compilation options (constants, workgroup memory init)
//!     - Label for debugging
//!
//! PipelineLayoutSource
//!     - Auto: Let wgpu derive layout from shader reflection
//!     - Explicit: Provide a pre-built wgpu::PipelineLayout
//!
//! CompilationOptions
//!     - Pipeline constants (specialization constants)
//!     - Workgroup memory zero-initialization flag
//!
//! TrinityComputePipeline
//!     - Wrapper around wgpu::ComputePipeline
//!     - Tracks label and layout ID for cache invalidation
//! ```
//!
//! # wgpu 22-25.x Compatibility
//!
//! This module targets wgpu 22+ and is compatible through 25.x using:
//!
//! ```text
//! device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
//!     label: Some("my_compute"),
//!     layout: Some(&pipeline_layout), // or None for auto
//!     module: &shader_module,
//!     entry_point: Some("main"),
//!     compilation_options: wgpu::PipelineCompilationOptions {
//!         constants: &[("BLOCK_SIZE", 64.0)],
//!         zero_initialize_workgroup_memory: true,
//!     },
//!     cache: None,
//! })
//! ```
//!
//! # Example
//!
//! ```no_run
//! use renderer_backend::compute_pipeline::{
//!     ComputePipelineDescriptor, PipelineLayoutSource,
//! };
//!
//! # fn example(
//! #     device: &wgpu::Device,
//! #     shader_module: &wgpu::ShaderModule,
//! # ) {
//! // Simple compute pipeline with auto layout
//! let pipeline = ComputePipelineDescriptor::new(shader_module, "main")
//!     .label("particle_update")
//!     .layout_auto()
//!     .constant("BLOCK_SIZE", 256.0)
//!     .zero_init_workgroup(true)
//!     .build(device);
//!
//! // Use in compute pass
//! compute_pass.set_pipeline(pipeline.raw());
//! # }
//! ```
//!
//! # Thread Safety
//!
//! All types in this module are `Send + Sync` when their underlying wgpu types are.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use ordered_float::OrderedFloat;
use parking_lot::RwLock;

// ---------------------------------------------------------------------------
// Layout ID generator (for cache invalidation tracking)
// ---------------------------------------------------------------------------

static LAYOUT_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Generate a unique layout ID for cache invalidation tracking.
fn next_layout_id() -> u64 {
    LAYOUT_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// ShaderModuleRef — abstraction over shader module sources
// ---------------------------------------------------------------------------

/// Reference to a shader module for compute pipeline creation.
///
/// Can be either a direct reference to a pre-compiled shader module or a WGSL
/// source string that will be compiled during pipeline creation.
#[derive(Debug, Clone)]
pub enum ShaderModuleRef<'a> {
    /// Direct reference to a pre-compiled shader module.
    Module(&'a wgpu::ShaderModule),
    /// WGSL source string to be compiled.
    Source(Cow<'a, str>),
}

impl<'a> ShaderModuleRef<'a> {
    /// Create from a shader module reference.
    pub fn module(module: &'a wgpu::ShaderModule) -> Self {
        Self::Module(module)
    }

    /// Create from a WGSL source string.
    pub fn source(source: impl Into<Cow<'a, str>>) -> Self {
        Self::Source(source.into())
    }
}

impl<'a> From<&'a wgpu::ShaderModule> for ShaderModuleRef<'a> {
    fn from(module: &'a wgpu::ShaderModule) -> Self {
        Self::Module(module)
    }
}

impl<'a> From<&'a str> for ShaderModuleRef<'a> {
    fn from(source: &'a str) -> Self {
        Self::Source(Cow::Borrowed(source))
    }
}

impl From<String> for ShaderModuleRef<'static> {
    fn from(source: String) -> Self {
        Self::Source(Cow::Owned(source))
    }
}

// ---------------------------------------------------------------------------
// PipelineLayoutSource — layout configuration
// ---------------------------------------------------------------------------

/// Specifies how the pipeline layout should be determined.
///
/// # Auto Layout
///
/// When using `Auto`, wgpu derives the pipeline layout from shader reflection.
/// This is convenient for simple cases but has limitations:
///
/// - Cannot share bind groups across pipelines
/// - May not work with all shader patterns
///
/// # Explicit Layout
///
/// For production use, prefer explicit layouts to:
///
/// - Enable bind group sharing across pipelines
/// - Ensure consistent resource binding
/// - Avoid reflection overhead
#[derive(Debug, Clone)]
pub enum PipelineLayoutSource<'a> {
    /// Let wgpu derive the layout from shader reflection.
    Auto,
    /// Use an explicitly provided pipeline layout.
    Explicit(&'a wgpu::PipelineLayout),
}

impl<'a> PipelineLayoutSource<'a> {
    /// Create an auto layout source.
    #[inline]
    pub fn auto() -> Self {
        Self::Auto
    }

    /// Create an explicit layout source.
    #[inline]
    pub fn explicit(layout: &'a wgpu::PipelineLayout) -> Self {
        Self::Explicit(layout)
    }

    /// Check if this is an auto layout.
    #[inline]
    pub fn is_auto(&self) -> bool {
        matches!(self, Self::Auto)
    }

    /// Get the explicit layout reference, if any.
    #[inline]
    pub fn as_explicit(&self) -> Option<&'a wgpu::PipelineLayout> {
        match self {
            Self::Explicit(layout) => Some(layout),
            Self::Auto => None,
        }
    }
}

impl Default for PipelineLayoutSource<'_> {
    fn default() -> Self {
        Self::Auto
    }
}

// ---------------------------------------------------------------------------
// CompilationOptions — shader compilation configuration
// ---------------------------------------------------------------------------

/// Configuration options for compute shader compilation.
///
/// # Pipeline Constants
///
/// Pipeline constants (also known as specialization constants) allow
/// compile-time customization of shader behavior:
///
/// ```wgsl
/// override BLOCK_SIZE: u32 = 64;
///
/// @compute @workgroup_size(BLOCK_SIZE)
/// fn main(...) { ... }
/// ```
///
/// Set at pipeline creation:
///
/// ```no_run
/// # use renderer_backend::compute_pipeline::CompilationOptions;
/// let options = CompilationOptions::new()
///     .constant("BLOCK_SIZE", 256.0);
/// ```
///
/// # Workgroup Memory Initialization
///
/// When `zero_initialize_workgroup_memory` is true, all workgroup-shared
/// variables are zero-initialized at the start of each workgroup invocation.
/// This has a small performance cost but ensures deterministic behavior.
#[derive(Debug, Clone, Default)]
pub struct CompilationOptions {
    /// Pipeline constants (specialization constants).
    pub constants: HashMap<String, f64>,
    /// Whether to zero-initialize workgroup memory.
    pub zero_initialize_workgroup_memory: bool,
}

impl CompilationOptions {
    /// Create new compilation options with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a pipeline constant.
    pub fn constant(mut self, name: impl Into<String>, value: f64) -> Self {
        self.constants.insert(name.into(), value);
        self
    }

    /// Set multiple pipeline constants at once.
    pub fn constants(mut self, constants: impl IntoIterator<Item = (String, f64)>) -> Self {
        self.constants.extend(constants);
        self
    }

    /// Enable or disable workgroup memory zero-initialization.
    pub fn zero_init_workgroup(mut self, enable: bool) -> Self {
        self.zero_initialize_workgroup_memory = enable;
        self
    }

    /// Convert to wgpu compilation options.
    ///
    /// Note: The returned struct borrows from this `CompilationOptions`, so
    /// keep this struct alive during pipeline creation.
    pub fn to_wgpu(&self) -> wgpu::PipelineCompilationOptions<'_> {
        wgpu::PipelineCompilationOptions {
            constants: &self.constants,
            zero_initialize_workgroup_memory: self.zero_initialize_workgroup_memory,
            vertex_pulling_transform: false,
        }
    }
}

// ---------------------------------------------------------------------------
// TrinityComputePipeline — compiled pipeline wrapper
// ---------------------------------------------------------------------------

/// A compiled compute pipeline wrapper with metadata for cache management.
///
/// Wraps a [`wgpu::ComputePipeline`] with:
/// - Optional debug label for identification
/// - Layout ID for cache invalidation tracking
/// - Owned shader module (if compiled from WGSL source)
///
/// # Thread Safety
///
/// `TrinityComputePipeline` is `Send + Sync` because `wgpu::ComputePipeline` is.
#[derive(Debug)]
pub struct TrinityComputePipeline {
    inner: wgpu::ComputePipeline,
    label: Option<String>,
    layout_id: u64,
    /// Holds the shader module if it was compiled from WGSL source.
    /// This ensures the module is properly dropped when the pipeline is dropped.
    #[allow(dead_code)]
    owned_shader: Option<wgpu::ShaderModule>,
}

impl TrinityComputePipeline {
    /// Create a new `TrinityComputePipeline` from a wgpu pipeline.
    pub(crate) fn new(
        inner: wgpu::ComputePipeline,
        label: Option<String>,
        layout_id: u64,
        owned_shader: Option<wgpu::ShaderModule>,
    ) -> Self {
        Self {
            inner,
            label,
            layout_id,
            owned_shader,
        }
    }

    /// Access the underlying [`wgpu::ComputePipeline`].
    #[inline]
    pub fn raw(&self) -> &wgpu::ComputePipeline {
        &self.inner
    }

    /// Get the debug label, if any.
    #[inline]
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Get the layout ID for cache invalidation tracking.
    ///
    /// When the pipeline layout changes, pipelines with old layout IDs
    /// should be invalidated.
    #[inline]
    pub fn layout_id(&self) -> u64 {
        self.layout_id
    }

    /// Consume and return the inner wgpu pipeline.
    #[inline]
    pub fn into_inner(self) -> wgpu::ComputePipeline {
        self.inner
    }
}

// ---------------------------------------------------------------------------
// ComputePipelineDescriptor — builder API
// ---------------------------------------------------------------------------

/// Builder for creating compute pipelines with all wgpu configuration options.
///
/// # Required Fields
///
/// - `module`: Shader module (required at construction)
/// - `entry_point`: Compute shader entry point name (required at construction)
///
/// # Optional Fields
///
/// - `label`: Debug label for identification
/// - `layout`: Pipeline layout source (default: Auto)
/// - `compilation_options`: Shader compilation options (default: empty)
/// - `cache`: Pipeline cache (default: None)
///
/// # Example
///
/// ```no_run
/// use renderer_backend::compute_pipeline::ComputePipelineDescriptor;
///
/// # fn example(device: &wgpu::Device, module: &wgpu::ShaderModule) {
/// let pipeline = ComputePipelineDescriptor::new(module, "main")
///     .label("reduction_sum")
///     .constant("WORKGROUP_SIZE", 128.0)
///     .zero_init_workgroup(true)
///     .build(device);
/// # }
/// ```
///
/// # Thread Safety
///
/// `ComputePipelineDescriptor` is `Send + Sync` when all referenced data is.
#[derive(Debug)]
pub struct ComputePipelineDescriptor<'a> {
    /// Debug label.
    label: Option<&'a str>,
    /// Pipeline layout source.
    layout: PipelineLayoutSource<'a>,
    /// Layout ID for cache invalidation tracking.
    layout_id: u64,
    /// Shader module reference.
    module: ShaderModuleRef<'a>,
    /// Compute shader entry point name.
    entry_point: Cow<'a, str>,
    /// Compilation options.
    compilation_options: CompilationOptions,
    /// Pipeline cache.
    cache: Option<&'a wgpu::PipelineCache>,
}

impl<'a> ComputePipelineDescriptor<'a> {
    /// Create a new compute pipeline descriptor.
    ///
    /// # Arguments
    ///
    /// * `module` - Shader module containing the compute shader
    /// * `entry_point` - Name of the `@compute` entry point function
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn example(device: &wgpu::Device, module: &wgpu::ShaderModule) {
    /// use renderer_backend::compute_pipeline::ComputePipelineDescriptor;
    ///
    /// let desc = ComputePipelineDescriptor::new(module, "cs_main");
    /// # }
    /// ```
    pub fn new(module: impl Into<ShaderModuleRef<'a>>, entry_point: impl Into<Cow<'a, str>>) -> Self {
        Self {
            label: None,
            layout: PipelineLayoutSource::Auto,
            layout_id: next_layout_id(),
            module: module.into(),
            entry_point: entry_point.into(),
            compilation_options: CompilationOptions::default(),
            cache: None,
        }
    }

    /// Create a descriptor from WGSL source string.
    ///
    /// The shader module will be compiled during pipeline creation.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn example(device: &wgpu::Device) {
    /// use renderer_backend::compute_pipeline::ComputePipelineDescriptor;
    ///
    /// let wgsl = r#"
    ///     @group(0) @binding(0) var<storage, read_write> data: array<f32>;
    ///
    ///     @compute @workgroup_size(64)
    ///     fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    ///         data[id.x] = data[id.x] * 2.0;
    ///     }
    /// "#;
    ///
    /// let desc = ComputePipelineDescriptor::from_wgsl(wgsl, "main");
    /// # }
    /// ```
    pub fn from_wgsl(source: impl Into<Cow<'a, str>>, entry_point: impl Into<Cow<'a, str>>) -> Self {
        Self {
            label: None,
            layout: PipelineLayoutSource::Auto,
            layout_id: next_layout_id(),
            module: ShaderModuleRef::Source(source.into()),
            entry_point: entry_point.into(),
            compilation_options: CompilationOptions::default(),
            cache: None,
        }
    }

    /// Set the debug label for the pipeline.
    ///
    /// Labels appear in GPU debugging tools and validation messages.
    pub fn label(mut self, label: &'a str) -> Self {
        self.label = Some(label);
        self
    }

    /// Set the pipeline layout source.
    pub fn layout(mut self, layout: PipelineLayoutSource<'a>) -> Self {
        self.layout = layout;
        self
    }

    /// Use automatic layout derivation from shader reflection.
    ///
    /// This is the default and is equivalent to `layout(PipelineLayoutSource::Auto)`.
    pub fn layout_auto(mut self) -> Self {
        self.layout = PipelineLayoutSource::Auto;
        self
    }

    /// Use an explicit pipeline layout.
    ///
    /// Prefer explicit layouts for production use to enable bind group sharing
    /// across pipelines.
    pub fn layout_explicit(mut self, layout: &'a wgpu::PipelineLayout) -> Self {
        self.layout = PipelineLayoutSource::Explicit(layout);
        self
    }

    /// Set the layout ID for cache coordination.
    ///
    /// Used when multiple descriptors should share the same layout ID for
    /// cache invalidation purposes.
    pub fn with_layout_id(mut self, layout_id: u64) -> Self {
        self.layout_id = layout_id;
        self
    }

    /// Add a pipeline constant (specialization constant).
    ///
    /// Pipeline constants allow compile-time customization of shader behavior:
    ///
    /// ```wgsl
    /// override BLOCK_SIZE: u32 = 64;
    /// ```
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn example(device: &wgpu::Device, module: &wgpu::ShaderModule) {
    /// use renderer_backend::compute_pipeline::ComputePipelineDescriptor;
    ///
    /// let pipeline = ComputePipelineDescriptor::new(module, "main")
    ///     .constant("BLOCK_SIZE", 256.0)
    ///     .constant("USE_FAST_PATH", 1.0)
    ///     .build(device);
    /// # }
    /// ```
    pub fn constant(mut self, name: impl Into<String>, value: f64) -> Self {
        self.compilation_options.constants.insert(name.into(), value);
        self
    }

    /// Set multiple pipeline constants at once.
    pub fn constants(mut self, constants: impl IntoIterator<Item = (String, f64)>) -> Self {
        self.compilation_options.constants.extend(constants);
        self
    }

    /// Enable or disable workgroup memory zero-initialization.
    ///
    /// When enabled, all workgroup-shared variables are zero-initialized at
    /// the start of each workgroup invocation. This has a small performance
    /// cost but ensures deterministic behavior.
    pub fn zero_init_workgroup(mut self, enable: bool) -> Self {
        self.compilation_options.zero_initialize_workgroup_memory = enable;
        self
    }

    /// Set complete compilation options.
    pub fn compilation_options(mut self, options: CompilationOptions) -> Self {
        self.compilation_options = options;
        self
    }

    /// Set the pipeline cache.
    pub fn cache(mut self, cache: &'a wgpu::PipelineCache) -> Self {
        self.cache = Some(cache);
        self
    }

    /// Convert to a wgpu compute pipeline descriptor.
    ///
    /// This method handles temporary storage for the shader module if created
    /// from WGSL source. The returned descriptor borrows from this builder
    /// and the optional `temp_module` output parameter.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device (needed to compile WGSL sources)
    /// * `temp_module` - Output parameter for a temporary shader module
    ///
    /// Note: For most use cases, prefer `build()` which handles all temporaries.
    pub fn to_wgpu_descriptor<'b>(
        &'b self,
        device: &wgpu::Device,
        temp_module: &'b mut Option<wgpu::ShaderModule>,
    ) -> wgpu::ComputePipelineDescriptor<'b>
    where
        'a: 'b,
    {
        // Resolve shader module
        let module = match &self.module {
            ShaderModuleRef::Module(m) => *m,
            ShaderModuleRef::Source(source) => {
                let shader_label = self.label.map(|l| format!("{}_shader", l));
                let compiled = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: shader_label.as_deref(),
                    source: wgpu::ShaderSource::Wgsl(source.clone()),
                });
                *temp_module = Some(compiled);
                temp_module.as_ref().unwrap()
            }
        };

        wgpu::ComputePipelineDescriptor {
            label: self.label,
            layout: self.layout.as_explicit(),
            module,
            entry_point: &self.entry_point,
            compilation_options: self.compilation_options.to_wgpu(),
            cache: self.cache,
        }
    }

    /// Build the compute pipeline.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn example(device: &wgpu::Device, module: &wgpu::ShaderModule) {
    /// use renderer_backend::compute_pipeline::ComputePipelineDescriptor;
    ///
    /// let pipeline = ComputePipelineDescriptor::new(module, "main")
    ///     .label("my_compute")
    ///     .build(device);
    ///
    /// // Use in compute pass
    /// compute_pass.set_pipeline(pipeline.raw());
    /// # }
    /// ```
    pub fn build(self, device: &wgpu::Device) -> TrinityComputePipeline {
        let label_owned = self.label.map(String::from);
        let layout_id = self.layout_id;

        // Resolve shader module - store owned module to prevent memory leak
        let owned_shader: Option<wgpu::ShaderModule> = match &self.module {
            ShaderModuleRef::Module(_) => None,
            ShaderModuleRef::Source(source) => {
                let compiled = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: self.label,
                    source: wgpu::ShaderSource::Wgsl(source.clone()),
                });
                Some(compiled)
            }
        };

        // Get a reference to the shader module for pipeline creation
        let module_ref: &wgpu::ShaderModule = match &self.module {
            ShaderModuleRef::Module(m) => *m,
            ShaderModuleRef::Source(_) => owned_shader.as_ref().unwrap(),
        };

        let inner = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: self.label,
            layout: self.layout.as_explicit(),
            module: module_ref,
            entry_point: &self.entry_point,
            compilation_options: self.compilation_options.to_wgpu(),
            cache: self.cache,
        });

        TrinityComputePipeline::new(inner, label_owned, layout_id, owned_shader)
    }
}

// ---------------------------------------------------------------------------
// Helper function
// ---------------------------------------------------------------------------

/// Create a compute pipeline from a descriptor.
///
/// Convenience function equivalent to `descriptor.build(device)`.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::compute_pipeline::{
///     create_compute_pipeline, ComputePipelineDescriptor,
/// };
///
/// # fn example(device: &wgpu::Device, module: &wgpu::ShaderModule) {
/// let desc = ComputePipelineDescriptor::new(module, "main")
///     .label("my_compute");
///
/// let pipeline = create_compute_pipeline(device, desc);
/// # }
/// ```
pub fn create_compute_pipeline(
    device: &wgpu::Device,
    descriptor: ComputePipelineDescriptor<'_>,
) -> TrinityComputePipeline {
    descriptor.build(device)
}

// ---------------------------------------------------------------------------
// ComputePipelineCache — T-WGPU-P3.9.2
// ---------------------------------------------------------------------------

/// Key for looking up cached compute pipelines.
///
/// Pipelines are uniquely identified by their shader, entry point, and
/// specialization constants. Two pipelines with the same key are
/// functionally identical.
///
/// # Example
///
/// ```
/// use renderer_backend::compute_pipeline::{ComputePipelineKey, SpecializationKey};
///
/// let key = ComputePipelineKey {
///     shader_id: 42,
///     entry_point: "main".to_string(),
///     specialization: SpecializationKey::default(),
/// };
///
/// // Keys are hashable and comparable
/// let key2 = key.clone();
/// assert_eq!(key, key2);
/// ```
#[derive(Hash, Eq, PartialEq, Clone, Debug)]
pub struct ComputePipelineKey {
    /// Unique identifier for the shader module.
    pub shader_id: u64,
    /// Entry point function name in the shader.
    pub entry_point: String,
    /// Specialization constants and options.
    pub specialization: SpecializationKey,
}

impl ComputePipelineKey {
    /// Create a new pipeline key.
    ///
    /// # Arguments
    ///
    /// * `shader_id` - Unique identifier for the shader module
    /// * `entry_point` - Name of the compute entry point function
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::compute_pipeline::ComputePipelineKey;
    ///
    /// let key = ComputePipelineKey::new(123, "cs_main");
    /// assert_eq!(key.shader_id, 123);
    /// assert_eq!(key.entry_point, "cs_main");
    /// ```
    pub fn new(shader_id: u64, entry_point: impl Into<String>) -> Self {
        Self {
            shader_id,
            entry_point: entry_point.into(),
            specialization: SpecializationKey::default(),
        }
    }

    /// Create a key with specialization constants.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::compute_pipeline::{ComputePipelineKey, SpecializationKey};
    ///
    /// let spec = SpecializationKey::new()
    ///     .constant("BLOCK_SIZE", 256.0)
    ///     .zero_init_workgroup(true);
    ///
    /// let key = ComputePipelineKey::with_specialization(42, "main", spec);
    /// ```
    pub fn with_specialization(
        shader_id: u64,
        entry_point: impl Into<String>,
        specialization: SpecializationKey,
    ) -> Self {
        Self {
            shader_id,
            entry_point: entry_point.into(),
            specialization,
        }
    }
}

/// Specialization key for pipeline compilation options.
///
/// Contains pipeline constants and compilation flags that affect shader
/// compilation. Two pipelines with different specialization keys will
/// be compiled separately even if they use the same shader.
///
/// # Hashing
///
/// Uses [`OrderedFloat`] for f64 values to ensure consistent hashing.
/// NaN values are handled safely by OrderedFloat's Ord implementation.
///
/// # Example
///
/// ```
/// use renderer_backend::compute_pipeline::SpecializationKey;
///
/// let spec = SpecializationKey::new()
///     .constant("WORKGROUP_SIZE", 64.0)
///     .constant("USE_FAST_PATH", 1.0)
///     .zero_init_workgroup(true);
/// ```
#[derive(Hash, Eq, PartialEq, Clone, Debug, Default)]
pub struct SpecializationKey {
    /// Pipeline constants (specialization constants).
    /// Uses BTreeMap for deterministic iteration order.
    constants: BTreeMap<String, OrderedFloat<f64>>,
    /// Whether to zero-initialize workgroup memory.
    zero_init_workgroup: bool,
}

impl SpecializationKey {
    /// Create a new empty specialization key.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a pipeline constant.
    ///
    /// # Arguments
    ///
    /// * `name` - Constant name as declared in the shader
    /// * `value` - Constant value (f64 for wgpu compatibility)
    pub fn constant(mut self, name: impl Into<String>, value: f64) -> Self {
        self.constants.insert(name.into(), OrderedFloat(value));
        self
    }

    /// Add multiple constants at once.
    pub fn constants(mut self, constants: impl IntoIterator<Item = (String, f64)>) -> Self {
        for (name, value) in constants {
            self.constants.insert(name, OrderedFloat(value));
        }
        self
    }

    /// Set the zero-initialize workgroup memory flag.
    pub fn zero_init_workgroup(mut self, enable: bool) -> Self {
        self.zero_init_workgroup = enable;
        self
    }

    /// Get the constants as a HashMap for wgpu.
    pub fn to_constants_map(&self) -> HashMap<String, f64> {
        self.constants
            .iter()
            .map(|(k, v)| (k.clone(), v.into_inner()))
            .collect()
    }

    /// Check if workgroup memory should be zero-initialized.
    pub fn should_zero_init_workgroup(&self) -> bool {
        self.zero_init_workgroup
    }

    /// Check if this key has any specialization.
    pub fn is_empty(&self) -> bool {
        self.constants.is_empty() && !self.zero_init_workgroup
    }

    /// Get the number of constants.
    pub fn num_constants(&self) -> usize {
        self.constants.len()
    }
}

/// Thread-safe cache for compute pipelines.
///
/// Caches compiled compute pipelines by their key (shader ID + entry point +
/// specialization). Pipeline compilation is expensive, so caching is critical
/// for runtime performance.
///
/// # Thread Safety
///
/// Uses [`RwLock`] for concurrent read access with exclusive write access.
/// Pipelines are stored as [`Arc`] for safe sharing across threads.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::compute_pipeline::{
///     ComputePipelineCache, ComputePipelineKey, ComputePipelineDescriptor,
/// };
/// use std::sync::Arc;
///
/// # fn example(device: &wgpu::Device, shader_module: &wgpu::ShaderModule) {
/// let cache = ComputePipelineCache::new();
///
/// let key = ComputePipelineKey::new(42, "main");
///
/// let pipeline = cache.get_or_create(key.clone(), || {
///     ComputePipelineDescriptor::new(shader_module, "main")
///         .label("cached_compute")
///         .build(device)
/// });
///
/// // Second lookup returns cached pipeline
/// let same_pipeline = cache.get(&key).unwrap();
/// # }
/// ```
///
/// # Cache Invalidation
///
/// When shaders are hot-reloaded or modified, invalidate affected pipelines:
///
/// ```no_run
/// # use renderer_backend::compute_pipeline::ComputePipelineCache;
/// let cache = ComputePipelineCache::new();
/// // ... populate cache ...
///
/// // Invalidate all pipelines using shader 42
/// let invalidated = cache.invalidate_by_shader(42);
/// println!("Invalidated {} pipelines", invalidated);
/// ```
pub struct ComputePipelineCache {
    pipelines: RwLock<HashMap<ComputePipelineKey, Arc<TrinityComputePipeline>>>,
}

impl ComputePipelineCache {
    /// Create a new empty pipeline cache.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::compute_pipeline::ComputePipelineCache;
    ///
    /// let cache = ComputePipelineCache::new();
    /// assert!(cache.is_empty());
    /// ```
    pub fn new() -> Self {
        Self {
            pipelines: RwLock::new(HashMap::new()),
        }
    }

    /// Create a cache with pre-allocated capacity.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Expected number of pipelines to cache
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::compute_pipeline::ComputePipelineCache;
    ///
    /// // Pre-allocate for 64 pipelines
    /// let cache = ComputePipelineCache::with_capacity(64);
    /// ```
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            pipelines: RwLock::new(HashMap::with_capacity(capacity)),
        }
    }

    /// Get a pipeline from the cache, or create it if not present.
    ///
    /// This is the primary cache access method. If the pipeline exists,
    /// it returns a clone of the Arc. Otherwise, it calls the create
    /// function to compile a new pipeline and caches it.
    ///
    /// # Arguments
    ///
    /// * `key` - The pipeline key to look up
    /// * `create` - Function to create the pipeline if not cached
    ///
    /// # Thread Safety
    ///
    /// Uses double-checked locking: first attempts a read lock for lookup,
    /// then acquires a write lock only if creation is needed.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::compute_pipeline::{
    ///     ComputePipelineCache, ComputePipelineKey, ComputePipelineDescriptor,
    /// };
    ///
    /// # fn example(device: &wgpu::Device, module: &wgpu::ShaderModule) {
    /// let cache = ComputePipelineCache::new();
    /// let key = ComputePipelineKey::new(1, "main");
    ///
    /// let pipeline = cache.get_or_create(key, || {
    ///     ComputePipelineDescriptor::new(module, "main")
    ///         .build(device)
    /// });
    /// # }
    /// ```
    pub fn get_or_create<F>(&self, key: ComputePipelineKey, create: F) -> Arc<TrinityComputePipeline>
    where
        F: FnOnce() -> TrinityComputePipeline,
    {
        // Fast path: read lock for lookup
        {
            let read_guard = self.pipelines.read();
            if let Some(pipeline) = read_guard.get(&key) {
                return Arc::clone(pipeline);
            }
        }

        // Slow path: write lock for insertion
        let mut write_guard = self.pipelines.write();

        // Double-check after acquiring write lock
        if let Some(pipeline) = write_guard.get(&key) {
            return Arc::clone(pipeline);
        }

        // Create and cache the pipeline
        let pipeline = Arc::new(create());
        write_guard.insert(key, Arc::clone(&pipeline));
        pipeline
    }

    /// Get a pipeline from the cache.
    ///
    /// Returns `None` if the pipeline is not cached.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::compute_pipeline::{ComputePipelineCache, ComputePipelineKey};
    ///
    /// let cache = ComputePipelineCache::new();
    /// let key = ComputePipelineKey::new(1, "main");
    ///
    /// assert!(cache.get(&key).is_none());
    /// ```
    pub fn get(&self, key: &ComputePipelineKey) -> Option<Arc<TrinityComputePipeline>> {
        self.pipelines.read().get(key).cloned()
    }

    /// Check if a pipeline is in the cache.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::compute_pipeline::{ComputePipelineCache, ComputePipelineKey};
    ///
    /// let cache = ComputePipelineCache::new();
    /// let key = ComputePipelineKey::new(1, "main");
    ///
    /// assert!(!cache.contains(&key));
    /// ```
    pub fn contains(&self, key: &ComputePipelineKey) -> bool {
        self.pipelines.read().contains_key(key)
    }

    /// Invalidate (remove) a specific pipeline from the cache.
    ///
    /// Returns `true` if the pipeline was in the cache.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::compute_pipeline::{ComputePipelineCache, ComputePipelineKey};
    ///
    /// let cache = ComputePipelineCache::new();
    /// let key = ComputePipelineKey::new(1, "main");
    ///
    /// // Nothing to invalidate
    /// assert!(!cache.invalidate(&key));
    /// ```
    pub fn invalidate(&self, key: &ComputePipelineKey) -> bool {
        self.pipelines.write().remove(key).is_some()
    }

    /// Invalidate all pipelines using a specific shader.
    ///
    /// Call this when a shader is hot-reloaded or modified.
    /// Returns the number of pipelines invalidated.
    ///
    /// # Arguments
    ///
    /// * `shader_id` - The shader ID to invalidate
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::compute_pipeline::{ComputePipelineCache, ComputePipelineKey};
    ///
    /// let cache = ComputePipelineCache::new();
    ///
    /// // After shader 42 is reloaded, invalidate all pipelines using it
    /// let count = cache.invalidate_by_shader(42);
    /// assert_eq!(count, 0); // No pipelines in empty cache
    /// ```
    pub fn invalidate_by_shader(&self, shader_id: u64) -> usize {
        let mut guard = self.pipelines.write();
        let before = guard.len();
        guard.retain(|key, _| key.shader_id != shader_id);
        before - guard.len()
    }

    /// Invalidate all pipelines in the cache.
    ///
    /// Returns the number of pipelines invalidated.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::compute_pipeline::ComputePipelineCache;
    ///
    /// let cache = ComputePipelineCache::new();
    /// let count = cache.invalidate_all();
    /// assert_eq!(count, 0);
    /// ```
    pub fn invalidate_all(&self) -> usize {
        let mut guard = self.pipelines.write();
        let count = guard.len();
        guard.clear();
        count
    }

    /// Get the number of cached pipelines.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::compute_pipeline::ComputePipelineCache;
    ///
    /// let cache = ComputePipelineCache::new();
    /// assert_eq!(cache.len(), 0);
    /// ```
    pub fn len(&self) -> usize {
        self.pipelines.read().len()
    }

    /// Check if the cache is empty.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::compute_pipeline::ComputePipelineCache;
    ///
    /// let cache = ComputePipelineCache::new();
    /// assert!(cache.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.pipelines.read().is_empty()
    }

    /// Get all cached shader IDs.
    ///
    /// Useful for debugging or cache statistics.
    pub fn cached_shader_ids(&self) -> Vec<u64> {
        let guard = self.pipelines.read();
        let mut ids: Vec<u64> = guard.keys().map(|k| k.shader_id).collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    /// Get cache statistics.
    pub fn stats(&self) -> ComputePipelineCacheStats {
        let guard = self.pipelines.read();
        let total = guard.len();
        let unique_shaders = {
            let mut ids: Vec<_> = guard.keys().map(|k| k.shader_id).collect();
            ids.sort_unstable();
            ids.dedup();
            ids.len()
        };
        let unique_entry_points = {
            let mut eps: Vec<_> = guard.keys().map(|k| k.entry_point.clone()).collect();
            eps.sort();
            eps.dedup();
            eps.len()
        };

        ComputePipelineCacheStats {
            total_pipelines: total,
            unique_shaders,
            unique_entry_points,
        }
    }
}

impl Default for ComputePipelineCache {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for ComputePipelineCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let guard = self.pipelines.read();
        f.debug_struct("ComputePipelineCache")
            .field("count", &guard.len())
            .finish()
    }
}

/// Statistics about the compute pipeline cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComputePipelineCacheStats {
    /// Total number of cached pipelines.
    pub total_pipelines: usize,
    /// Number of unique shader IDs in the cache.
    pub unique_shaders: usize,
    /// Number of unique entry point names in the cache.
    pub unique_entry_points: usize,
}

// Thread safety assertions
static_assertions::assert_impl_all!(ComputePipelineCache: Send, Sync);
static_assertions::assert_impl_all!(ComputePipelineKey: Send, Sync);
static_assertions::assert_impl_all!(SpecializationKey: Send, Sync);

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── Layout ID generation ─────────────────────────────────────────────────

    #[test]
    fn test_layout_id_uniqueness() {
        let id1 = next_layout_id();
        let id2 = next_layout_id();
        let id3 = next_layout_id();

        assert_ne!(id1, id2);
        assert_ne!(id2, id3);
        assert_ne!(id1, id3);
    }

    // ── ShaderModuleRef ──────────────────────────────────────────────────────

    #[test]
    fn test_shader_module_ref_from_str() {
        let source = "fn main() {}";
        let module_ref: ShaderModuleRef<'_> = source.into();
        assert!(matches!(module_ref, ShaderModuleRef::Source(_)));
    }

    #[test]
    fn test_shader_module_ref_from_string() {
        let source = String::from("fn main() {}");
        let module_ref: ShaderModuleRef<'static> = source.into();
        assert!(matches!(module_ref, ShaderModuleRef::Source(_)));
    }

    // ── PipelineLayoutSource ─────────────────────────────────────────────────

    #[test]
    fn test_pipeline_layout_source_default_is_auto() {
        let source = PipelineLayoutSource::default();
        assert!(source.is_auto());
    }

    #[test]
    fn test_pipeline_layout_source_auto() {
        let source = PipelineLayoutSource::auto();
        assert!(source.is_auto());
        assert!(source.as_explicit().is_none());
    }

    // ── CompilationOptions ───────────────────────────────────────────────────

    #[test]
    fn test_compilation_options_default() {
        let options = CompilationOptions::default();
        assert!(options.constants.is_empty());
        assert!(!options.zero_initialize_workgroup_memory);
    }

    #[test]
    fn test_compilation_options_constant() {
        let options = CompilationOptions::new()
            .constant("BLOCK_SIZE", 64.0)
            .constant("USE_FAST", 1.0);

        assert_eq!(options.constants.len(), 2);
        assert_eq!(options.constants.get("BLOCK_SIZE"), Some(&64.0));
        assert_eq!(options.constants.get("USE_FAST"), Some(&1.0));
    }

    #[test]
    fn test_compilation_options_constants_batch() {
        let constants = vec![
            (String::from("A"), 1.0),
            (String::from("B"), 2.0),
            (String::from("C"), 3.0),
        ];

        let options = CompilationOptions::new().constants(constants);
        assert_eq!(options.constants.len(), 3);
    }

    #[test]
    fn test_compilation_options_zero_init() {
        let options = CompilationOptions::new().zero_init_workgroup(true);
        assert!(options.zero_initialize_workgroup_memory);

        let options = options.zero_init_workgroup(false);
        assert!(!options.zero_initialize_workgroup_memory);
    }

    #[test]
    fn test_compilation_options_to_wgpu() {
        let options = CompilationOptions::new()
            .constant("SIZE", 128.0)
            .zero_init_workgroup(true);

        let wgpu_options = options.to_wgpu();
        assert_eq!(wgpu_options.constants.get("SIZE"), Some(&128.0));
        assert!(wgpu_options.zero_initialize_workgroup_memory);
    }

    // ── ComputePipelineDescriptor ────────────────────────────────────────────

    #[test]
    fn test_descriptor_from_wgsl() {
        let wgsl = r#"
            @compute @workgroup_size(64)
            fn main() {}
        "#;

        let desc = ComputePipelineDescriptor::from_wgsl(wgsl, "main");
        assert!(desc.label.is_none());
        assert!(desc.layout.is_auto());
        assert!(matches!(desc.module, ShaderModuleRef::Source(_)));
    }

    #[test]
    fn test_descriptor_builder_chain() {
        let wgsl = "@compute @workgroup_size(64) fn cs() {}";

        let desc = ComputePipelineDescriptor::from_wgsl(wgsl, "cs")
            .label("test_pipeline")
            .layout_auto()
            .constant("SIZE", 64.0)
            .constant("ENABLE", 1.0)
            .zero_init_workgroup(true);

        assert_eq!(desc.label, Some("test_pipeline"));
        assert!(desc.layout.is_auto());
        assert_eq!(desc.compilation_options.constants.len(), 2);
        assert!(desc.compilation_options.zero_initialize_workgroup_memory);
    }

    #[test]
    fn test_descriptor_layout_id_assignment() {
        let wgsl = "@compute @workgroup_size(64) fn main() {}";

        let desc1 = ComputePipelineDescriptor::from_wgsl(wgsl, "main");
        let desc2 = ComputePipelineDescriptor::from_wgsl(wgsl, "main");

        // Each descriptor gets a unique layout ID by default
        assert_ne!(desc1.layout_id, desc2.layout_id);
    }

    #[test]
    fn test_descriptor_custom_layout_id() {
        let wgsl = "@compute @workgroup_size(64) fn main() {}";

        let desc = ComputePipelineDescriptor::from_wgsl(wgsl, "main")
            .with_layout_id(42);

        assert_eq!(desc.layout_id, 42);
    }

    // ── GPU tests (require device) ───────────────────────────────────────────

    /// Helper: obtain a (device, queue) pair, skipping the test if no GPU
    /// is available.
    fn create_test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            },
        ))?;
        Some(
            pollster::block_on(adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("test device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            ))
            .expect("device creation"),
        )
    }

    #[test]
    fn test_build_simple_compute_pipeline() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let wgsl = r#"
            @group(0) @binding(0) var<storage, read_write> data: array<f32>;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                data[id.x] = data[id.x] * 2.0;
            }
        "#;

        let pipeline = ComputePipelineDescriptor::from_wgsl(wgsl, "main")
            .label("test_compute")
            .build(&device);

        assert_eq!(pipeline.label(), Some("test_compute"));
        assert!(pipeline.layout_id() > 0);
    }

    #[test]
    fn test_build_with_shader_module() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let wgsl = r#"
            @group(0) @binding(0) var<storage, read_write> data: array<f32>;

            @compute @workgroup_size(64)
            fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
                data[id.x] = data[id.x] + 1.0;
            }
        "#;

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("test_module"),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });

        let pipeline = ComputePipelineDescriptor::new(&module, "cs_main")
            .label("module_pipeline")
            .build(&device);

        assert_eq!(pipeline.label(), Some("module_pipeline"));
    }

    #[test]
    fn test_build_with_explicit_layout() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        // Create bind group layout
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compute_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        // Create pipeline layout
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("compute_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let wgsl = r#"
            @group(0) @binding(0) var<storage, read_write> data: array<f32>;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                data[id.x] = data[id.x] * 2.0;
            }
        "#;

        let pipeline = ComputePipelineDescriptor::from_wgsl(wgsl, "main")
            .label("explicit_layout_pipeline")
            .layout_explicit(&layout)
            .build(&device);

        assert_eq!(pipeline.label(), Some("explicit_layout_pipeline"));
    }

    #[test]
    fn test_build_with_constants() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        // Note: Pipeline constants require shader support
        // This test validates the API even if constants aren't used in shader
        let wgsl = r#"
            @group(0) @binding(0) var<storage, read_write> data: array<f32>;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                data[id.x] = data[id.x] * 2.0;
            }
        "#;

        // This should compile successfully even with unused constants
        // (wgpu ignores constants not declared in shader)
        let pipeline = ComputePipelineDescriptor::from_wgsl(wgsl, "main")
            .label("constants_pipeline")
            .constant("BLOCK_SIZE", 64.0)
            .zero_init_workgroup(true)
            .build(&device);

        assert_eq!(pipeline.label(), Some("constants_pipeline"));
    }

    #[test]
    fn test_create_compute_pipeline_helper() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let wgsl = r#"
            @group(0) @binding(0) var<storage, read_write> data: array<f32>;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                data[id.x] = data[id.x] * 2.0;
            }
        "#;

        let desc = ComputePipelineDescriptor::from_wgsl(wgsl, "main")
            .label("helper_pipeline");

        let pipeline = create_compute_pipeline(&device, desc);
        assert_eq!(pipeline.label(), Some("helper_pipeline"));
    }

    #[test]
    fn test_pipeline_into_inner() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let wgsl = r#"
            @group(0) @binding(0) var<storage, read_write> data: array<f32>;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                data[id.x] = data[id.x] * 2.0;
            }
        "#;

        let pipeline = ComputePipelineDescriptor::from_wgsl(wgsl, "main")
            .build(&device);

        // Verify we can extract the inner pipeline
        let _inner: wgpu::ComputePipeline = pipeline.into_inner();
    }

    // ── Thread safety assertions ─────────────────────────────────────────────

    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    #[test]
    fn test_compilation_options_is_send_sync() {
        assert_send::<CompilationOptions>();
        assert_sync::<CompilationOptions>();
    }

    #[test]
    fn test_trinity_compute_pipeline_is_send_sync() {
        assert_send::<TrinityComputePipeline>();
        assert_sync::<TrinityComputePipeline>();
    }

    // ══════════════════════════════════════════════════════════════════════════
    // WHITEBOX TESTS — T-WGPU-P3.9.1
    // ══════════════════════════════════════════════════════════════════════════

    // ── ShaderModuleRef comprehensive tests ──────────────────────────────────

    #[test]
    fn test_shader_module_ref_source_constructor() {
        // Test the explicit source() constructor
        let ref1 = ShaderModuleRef::source("fn main() {}");
        assert!(matches!(ref1, ShaderModuleRef::Source(Cow::Borrowed(_))));

        let ref2 = ShaderModuleRef::source(String::from("fn main() {}"));
        assert!(matches!(ref2, ShaderModuleRef::Source(Cow::Owned(_))));
    }

    #[test]
    fn test_shader_module_ref_debug_impl() {
        let source_ref: ShaderModuleRef<'_> = "fn main() {}".into();
        let debug_str = format!("{:?}", source_ref);
        assert!(debug_str.contains("Source"));
        assert!(debug_str.contains("fn main()"));
    }

    #[test]
    fn test_shader_module_ref_clone() {
        let original: ShaderModuleRef<'_> = "fn main() {}".into();
        let cloned = original.clone();

        // Both should be Source variants with same content
        match (&original, &cloned) {
            (ShaderModuleRef::Source(a), ShaderModuleRef::Source(b)) => {
                assert_eq!(a.as_ref(), b.as_ref());
            }
            _ => panic!("Expected both to be Source variants"),
        }
    }

    #[test]
    fn test_shader_module_ref_cow_borrowed_vs_owned() {
        // Borrowed case
        let borrowed_source = "borrowed shader";
        let borrowed_ref = ShaderModuleRef::source(borrowed_source);
        if let ShaderModuleRef::Source(cow) = borrowed_ref {
            assert!(matches!(cow, Cow::Borrowed(_)));
        }

        // Owned case
        let owned_source = String::from("owned shader");
        let owned_ref = ShaderModuleRef::source(owned_source);
        if let ShaderModuleRef::Source(cow) = owned_ref {
            assert!(matches!(cow, Cow::Owned(_)));
        }
    }

    // ── PipelineLayoutSource comprehensive tests ─────────────────────────────

    #[test]
    fn test_pipeline_layout_source_debug_impl() {
        let auto = PipelineLayoutSource::Auto;
        let debug_str = format!("{:?}", auto);
        assert!(debug_str.contains("Auto"));
    }

    #[test]
    fn test_pipeline_layout_source_clone() {
        let auto = PipelineLayoutSource::Auto;
        let cloned = auto.clone();
        assert!(cloned.is_auto());
    }

    #[test]
    fn test_pipeline_layout_source_is_auto_false_for_explicit() {
        // We can't create an Explicit variant without a real layout,
        // but we can test the Auto case thoroughly
        let auto = PipelineLayoutSource::auto();
        assert!(auto.is_auto());
        assert!(auto.as_explicit().is_none());
    }

    // ── CompilationOptions comprehensive tests ───────────────────────────────

    #[test]
    fn test_compilation_options_clone() {
        let original = CompilationOptions::new()
            .constant("A", 1.0)
            .constant("B", 2.0)
            .zero_init_workgroup(true);

        let cloned = original.clone();

        assert_eq!(cloned.constants.len(), 2);
        assert_eq!(cloned.constants.get("A"), Some(&1.0));
        assert_eq!(cloned.constants.get("B"), Some(&2.0));
        assert!(cloned.zero_initialize_workgroup_memory);
    }

    #[test]
    fn test_compilation_options_debug_impl() {
        let options = CompilationOptions::new()
            .constant("DEBUG_CONST", 42.0)
            .zero_init_workgroup(true);

        let debug_str = format!("{:?}", options);
        assert!(debug_str.contains("CompilationOptions"));
        assert!(debug_str.contains("DEBUG_CONST"));
        assert!(debug_str.contains("42"));
        assert!(debug_str.contains("true"));
    }

    #[test]
    fn test_compilation_options_constant_override() {
        // Setting same constant twice should override
        let options = CompilationOptions::new()
            .constant("SIZE", 64.0)
            .constant("SIZE", 128.0);

        assert_eq!(options.constants.len(), 1);
        assert_eq!(options.constants.get("SIZE"), Some(&128.0));
    }

    #[test]
    fn test_compilation_options_many_constants() {
        // Test with many constants
        let mut options = CompilationOptions::new();
        for i in 0..100 {
            options = options.constant(format!("CONST_{}", i), i as f64);
        }

        assert_eq!(options.constants.len(), 100);
        assert_eq!(options.constants.get("CONST_0"), Some(&0.0));
        assert_eq!(options.constants.get("CONST_99"), Some(&99.0));
    }

    #[test]
    fn test_compilation_options_empty_constant_name() {
        // Empty string as constant name (edge case)
        let options = CompilationOptions::new().constant("", 1.0);
        assert_eq!(options.constants.get(""), Some(&1.0));
    }

    #[test]
    fn test_compilation_options_unicode_constant_name() {
        // Unicode constant names
        let options = CompilationOptions::new()
            .constant("常量", 1.0)
            .constant("КОНСТАНТА", 2.0)
            .constant("定数_αβγ", 3.0);

        assert_eq!(options.constants.len(), 3);
        assert_eq!(options.constants.get("常量"), Some(&1.0));
        assert_eq!(options.constants.get("КОНСТАНТА"), Some(&2.0));
        assert_eq!(options.constants.get("定数_αβγ"), Some(&3.0));
    }

    #[test]
    fn test_compilation_options_special_float_values() {
        // Test special float values
        let options = CompilationOptions::new()
            .constant("ZERO", 0.0)
            .constant("NEGATIVE", -1.0)
            .constant("LARGE", 1e38)
            .constant("SMALL", 1e-38)
            .constant("INF", f64::INFINITY)
            .constant("NEG_INF", f64::NEG_INFINITY);

        assert_eq!(options.constants.get("ZERO"), Some(&0.0));
        assert_eq!(options.constants.get("NEGATIVE"), Some(&-1.0));
        assert_eq!(options.constants.get("LARGE"), Some(&1e38));
        assert_eq!(options.constants.get("SMALL"), Some(&1e-38));
        assert_eq!(options.constants.get("INF"), Some(&f64::INFINITY));
        assert_eq!(options.constants.get("NEG_INF"), Some(&f64::NEG_INFINITY));
    }

    #[test]
    fn test_compilation_options_nan_constant() {
        // NaN requires special comparison
        let options = CompilationOptions::new().constant("NAN", f64::NAN);
        let value = options.constants.get("NAN").unwrap();
        assert!(value.is_nan());
    }

    #[test]
    fn test_compilation_options_constants_batch_empty() {
        let options = CompilationOptions::new().constants(std::iter::empty());
        assert!(options.constants.is_empty());
    }

    #[test]
    fn test_compilation_options_constants_batch_merge() {
        // First add some constants, then batch add more
        let options = CompilationOptions::new()
            .constant("A", 1.0)
            .constants(vec![
                (String::from("B"), 2.0),
                (String::from("C"), 3.0),
            ]);

        assert_eq!(options.constants.len(), 3);
        assert_eq!(options.constants.get("A"), Some(&1.0));
        assert_eq!(options.constants.get("B"), Some(&2.0));
        assert_eq!(options.constants.get("C"), Some(&3.0));
    }

    #[test]
    fn test_compilation_options_to_wgpu_empty() {
        let options = CompilationOptions::new();
        let wgpu_opts = options.to_wgpu();

        assert!(wgpu_opts.constants.is_empty());
        assert!(!wgpu_opts.zero_initialize_workgroup_memory);
        assert!(!wgpu_opts.vertex_pulling_transform);
    }

    #[test]
    fn test_compilation_options_chaining_order_independence() {
        // Order of chaining shouldn't matter for final result
        let opts1 = CompilationOptions::new()
            .constant("A", 1.0)
            .zero_init_workgroup(true);

        let opts2 = CompilationOptions::new()
            .zero_init_workgroup(true)
            .constant("A", 1.0);

        assert_eq!(opts1.constants.get("A"), opts2.constants.get("A"));
        assert_eq!(
            opts1.zero_initialize_workgroup_memory,
            opts2.zero_initialize_workgroup_memory
        );
    }

    // ── ComputePipelineDescriptor comprehensive tests ────────────────────────

    #[test]
    fn test_descriptor_debug_impl() {
        let wgsl = "@compute @workgroup_size(64) fn main() {}";
        let desc = ComputePipelineDescriptor::from_wgsl(wgsl, "main")
            .label("debug_test");

        let debug_str = format!("{:?}", desc);
        assert!(debug_str.contains("ComputePipelineDescriptor"));
        assert!(debug_str.contains("debug_test"));
        assert!(debug_str.contains("main"));
    }

    #[test]
    fn test_descriptor_constants_batch() {
        let wgsl = "@compute @workgroup_size(64) fn main() {}";
        let constants = vec![
            (String::from("X"), 1.0),
            (String::from("Y"), 2.0),
            (String::from("Z"), 3.0),
        ];

        let desc = ComputePipelineDescriptor::from_wgsl(wgsl, "main")
            .constants(constants);

        assert_eq!(desc.compilation_options.constants.len(), 3);
    }

    #[test]
    fn test_descriptor_compilation_options_setter() {
        let wgsl = "@compute @workgroup_size(64) fn main() {}";
        let options = CompilationOptions::new()
            .constant("CUSTOM", 99.0)
            .zero_init_workgroup(true);

        let desc = ComputePipelineDescriptor::from_wgsl(wgsl, "main")
            .compilation_options(options);

        assert_eq!(
            desc.compilation_options.constants.get("CUSTOM"),
            Some(&99.0)
        );
        assert!(desc.compilation_options.zero_initialize_workgroup_memory);
    }

    #[test]
    fn test_descriptor_label_none_by_default() {
        let wgsl = "@compute @workgroup_size(64) fn main() {}";
        let desc = ComputePipelineDescriptor::from_wgsl(wgsl, "main");
        assert!(desc.label.is_none());
    }

    #[test]
    fn test_descriptor_empty_entry_point() {
        // Empty entry point (edge case - may fail at build time but descriptor should accept it)
        let wgsl = "@compute @workgroup_size(64) fn main() {}";
        let desc = ComputePipelineDescriptor::from_wgsl(wgsl, "");
        assert_eq!(desc.entry_point.as_ref(), "");
    }

    #[test]
    fn test_descriptor_unicode_label() {
        let wgsl = "@compute @workgroup_size(64) fn main() {}";
        let desc = ComputePipelineDescriptor::from_wgsl(wgsl, "main")
            .label("计算管线_αβγ_パイプライン");

        assert_eq!(desc.label, Some("计算管线_αβγ_パイプライン"));
    }

    #[test]
    fn test_descriptor_unicode_entry_point() {
        let wgsl = "@compute @workgroup_size(64) fn main() {}";
        let desc = ComputePipelineDescriptor::from_wgsl(wgsl, "主函数_メイン");
        assert_eq!(desc.entry_point.as_ref(), "主函数_メイン");
    }

    #[test]
    fn test_descriptor_layout_method_with_auto() {
        let wgsl = "@compute @workgroup_size(64) fn main() {}";
        let desc = ComputePipelineDescriptor::from_wgsl(wgsl, "main")
            .layout(PipelineLayoutSource::Auto);

        assert!(desc.layout.is_auto());
    }

    #[test]
    fn test_descriptor_layout_auto_is_idempotent() {
        let wgsl = "@compute @workgroup_size(64) fn main() {}";
        let desc = ComputePipelineDescriptor::from_wgsl(wgsl, "main")
            .layout_auto()
            .layout_auto()
            .layout_auto();

        assert!(desc.layout.is_auto());
    }

    #[test]
    fn test_descriptor_entry_point_from_cow_borrowed() {
        let wgsl = "@compute @workgroup_size(64) fn main() {}";
        let entry_point: &str = "main";
        let desc = ComputePipelineDescriptor::from_wgsl(wgsl, entry_point);

        assert_eq!(desc.entry_point.as_ref(), "main");
    }

    #[test]
    fn test_descriptor_entry_point_from_cow_owned() {
        let wgsl = "@compute @workgroup_size(64) fn main() {}";
        let entry_point = String::from("main");
        let desc = ComputePipelineDescriptor::from_wgsl(wgsl, entry_point);

        assert_eq!(desc.entry_point.as_ref(), "main");
    }

    #[test]
    fn test_descriptor_cache_none_by_default() {
        let wgsl = "@compute @workgroup_size(64) fn main() {}";
        let desc = ComputePipelineDescriptor::from_wgsl(wgsl, "main");
        assert!(desc.cache.is_none());
    }

    #[test]
    fn test_descriptor_multiple_constant_calls() {
        let wgsl = "@compute @workgroup_size(64) fn main() {}";
        let desc = ComputePipelineDescriptor::from_wgsl(wgsl, "main")
            .constant("A", 1.0)
            .constant("B", 2.0)
            .constant("C", 3.0)
            .constant("D", 4.0)
            .constant("E", 5.0);

        assert_eq!(desc.compilation_options.constants.len(), 5);
    }

    #[test]
    fn test_descriptor_constant_and_constants_batch_combined() {
        let wgsl = "@compute @workgroup_size(64) fn main() {}";
        let desc = ComputePipelineDescriptor::from_wgsl(wgsl, "main")
            .constant("SINGLE", 1.0)
            .constants(vec![
                (String::from("BATCH_A"), 2.0),
                (String::from("BATCH_B"), 3.0),
            ])
            .constant("ANOTHER_SINGLE", 4.0);

        assert_eq!(desc.compilation_options.constants.len(), 4);
        assert_eq!(
            desc.compilation_options.constants.get("SINGLE"),
            Some(&1.0)
        );
        assert_eq!(
            desc.compilation_options.constants.get("BATCH_A"),
            Some(&2.0)
        );
        assert_eq!(
            desc.compilation_options.constants.get("BATCH_B"),
            Some(&3.0)
        );
        assert_eq!(
            desc.compilation_options.constants.get("ANOTHER_SINGLE"),
            Some(&4.0)
        );
    }

    // ── TrinityComputePipeline unit tests (no GPU) ───────────────────────────

    // Note: TrinityComputePipeline requires a wgpu::ComputePipeline which needs
    // a GPU. The following tests verify the API behavior with GPU tests.

    // ── GPU tests for TrinityComputePipeline ─────────────────────────────────

    #[test]
    fn test_trinity_compute_pipeline_raw_accessor() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let wgsl = r#"
            @group(0) @binding(0) var<storage, read_write> data: array<f32>;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                data[id.x] = data[id.x] * 2.0;
            }
        "#;

        let pipeline = ComputePipelineDescriptor::from_wgsl(wgsl, "main")
            .label("raw_test")
            .build(&device);

        // Verify raw() returns a valid reference
        let _raw: &wgpu::ComputePipeline = pipeline.raw();
    }

    #[test]
    fn test_trinity_compute_pipeline_no_label() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let wgsl = r#"
            @group(0) @binding(0) var<storage, read_write> data: array<f32>;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                data[id.x] = data[id.x] * 2.0;
            }
        "#;

        // Build without setting a label
        let pipeline = ComputePipelineDescriptor::from_wgsl(wgsl, "main")
            .build(&device);

        assert!(pipeline.label().is_none());
    }

    #[test]
    fn test_trinity_compute_pipeline_layout_id_preserved() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let wgsl = r#"
            @group(0) @binding(0) var<storage, read_write> data: array<f32>;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                data[id.x] = data[id.x] * 2.0;
            }
        "#;

        let custom_id = 12345u64;
        let pipeline = ComputePipelineDescriptor::from_wgsl(wgsl, "main")
            .with_layout_id(custom_id)
            .build(&device);

        assert_eq!(pipeline.layout_id(), custom_id);
    }

    #[test]
    fn test_trinity_compute_pipeline_debug_impl() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let wgsl = r#"
            @group(0) @binding(0) var<storage, read_write> data: array<f32>;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                data[id.x] = data[id.x] * 2.0;
            }
        "#;

        let pipeline = ComputePipelineDescriptor::from_wgsl(wgsl, "main")
            .label("debug_pipeline")
            .build(&device);

        let debug_str = format!("{:?}", pipeline);
        assert!(debug_str.contains("TrinityComputePipeline"));
        assert!(debug_str.contains("debug_pipeline"));
    }

    #[test]
    fn test_build_with_zero_init_enabled() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let wgsl = r#"
            var<workgroup> shared_data: array<f32, 64>;

            @group(0) @binding(0) var<storage, read_write> data: array<f32>;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                shared_data[id.x % 64] = 0.0;
                data[id.x] = shared_data[id.x % 64];
            }
        "#;

        let pipeline = ComputePipelineDescriptor::from_wgsl(wgsl, "main")
            .label("zero_init_enabled")
            .zero_init_workgroup(true)
            .build(&device);

        assert_eq!(pipeline.label(), Some("zero_init_enabled"));
    }

    #[test]
    fn test_build_with_zero_init_disabled() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let wgsl = r#"
            @group(0) @binding(0) var<storage, read_write> data: array<f32>;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                data[id.x] = data[id.x] * 2.0;
            }
        "#;

        let pipeline = ComputePipelineDescriptor::from_wgsl(wgsl, "main")
            .label("zero_init_disabled")
            .zero_init_workgroup(false)
            .build(&device);

        assert_eq!(pipeline.label(), Some("zero_init_disabled"));
    }

    #[test]
    fn test_build_with_full_compilation_options() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let wgsl = r#"
            @group(0) @binding(0) var<storage, read_write> data: array<f32>;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                data[id.x] = data[id.x] * 2.0;
            }
        "#;

        let options = CompilationOptions::new()
            .constant("A", 1.0)
            .constant("B", 2.0)
            .zero_init_workgroup(true);

        let pipeline = ComputePipelineDescriptor::from_wgsl(wgsl, "main")
            .label("full_options")
            .compilation_options(options)
            .build(&device);

        assert_eq!(pipeline.label(), Some("full_options"));
    }

    // ── Additional thread safety assertions ──────────────────────────────────

    #[test]
    fn test_shader_module_ref_is_send_sync() {
        // ShaderModuleRef<'static> with owned String is Send + Sync
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ShaderModuleRef<'static>>();
    }

    #[test]
    fn test_pipeline_layout_source_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<PipelineLayoutSource<'static>>();
    }

    #[test]
    fn test_compute_pipeline_descriptor_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ComputePipelineDescriptor<'static>>();
    }

    // ── Layout ID counter tests ──────────────────────────────────────────────

    #[test]
    fn test_layout_id_monotonically_increasing() {
        let ids: Vec<u64> = (0..10).map(|_| next_layout_id()).collect();

        for window in ids.windows(2) {
            assert!(window[1] > window[0], "IDs should be monotonically increasing");
        }
    }

    #[test]
    fn test_layout_id_thread_safety() {
        use std::sync::Arc;
        use std::thread;

        let ids = Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut handles = vec![];

        for _ in 0..4 {
            let ids_clone = Arc::clone(&ids);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    let id = next_layout_id();
                    ids_clone.lock().unwrap().push(id);
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let all_ids = ids.lock().unwrap();
        assert_eq!(all_ids.len(), 400);

        // All IDs should be unique
        let mut sorted_ids = all_ids.clone();
        sorted_ids.sort();
        sorted_ids.dedup();
        assert_eq!(sorted_ids.len(), 400, "All layout IDs should be unique");
    }

    // ── Edge case: very long label ───────────────────────────────────────────

    #[test]
    fn test_descriptor_very_long_label() {
        let long_label = "a".repeat(10000);
        let wgsl = "@compute @workgroup_size(64) fn main() {}";

        // Need to keep the label alive for the lifetime of the descriptor
        let desc = ComputePipelineDescriptor::from_wgsl(wgsl, "main")
            .label(long_label.as_str());

        assert_eq!(desc.label.unwrap().len(), 10000);
    }

    #[test]
    fn test_descriptor_very_long_entry_point() {
        let long_entry = "main_".to_string() + &"x".repeat(10000);
        let wgsl = "@compute @workgroup_size(64) fn main() {}";
        let desc = ComputePipelineDescriptor::from_wgsl(wgsl, long_entry.clone());

        assert_eq!(desc.entry_point.len(), 10005);
    }

    // ── Edge case: whitespace handling ───────────────────────────────────────

    #[test]
    fn test_descriptor_whitespace_label() {
        let wgsl = "@compute @workgroup_size(64) fn main() {}";
        let desc = ComputePipelineDescriptor::from_wgsl(wgsl, "main")
            .label("   spaces   ");

        assert_eq!(desc.label, Some("   spaces   "));
    }

    #[test]
    fn test_descriptor_whitespace_entry_point() {
        let wgsl = "@compute @workgroup_size(64) fn main() {}";
        let desc = ComputePipelineDescriptor::from_wgsl(wgsl, "  main  ");

        assert_eq!(desc.entry_point.as_ref(), "  main  ");
    }

    // ── Verify PipelineLayoutSource explicit path with GPU ───────────────────

    #[test]
    fn test_pipeline_layout_source_explicit_is_not_auto() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("test_bgl"),
            entries: &[],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("test_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let source = PipelineLayoutSource::explicit(&layout);
        assert!(!source.is_auto());
        assert!(source.as_explicit().is_some());
    }

    #[test]
    fn test_pipeline_layout_source_from_methods() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        // Test explicit() constructor
        let explicit = PipelineLayoutSource::explicit(&layout);
        assert!(!explicit.is_auto());

        // Test auto() constructor
        let auto = PipelineLayoutSource::auto();
        assert!(auto.is_auto());
    }

    // ── Descriptor new() with pre-compiled module ────────────────────────────

    #[test]
    fn test_descriptor_new_with_module_and_chaining() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let wgsl = r#"
            @group(0) @binding(0) var<storage, read_write> data: array<f32>;

            @compute @workgroup_size(64)
            fn cs_entry(@builtin(global_invocation_id) id: vec3<u32>) {
                data[id.x] = data[id.x] + 1.0;
            }
        "#;

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("precompiled"),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });

        // Test full builder chain with pre-compiled module
        let pipeline = ComputePipelineDescriptor::new(&module, "cs_entry")
            .label("chained_module_pipeline")
            .layout_auto()
            .constant("VAL", 42.0)
            .zero_init_workgroup(false)
            .with_layout_id(9999)
            .build(&device);

        assert_eq!(pipeline.label(), Some("chained_module_pipeline"));
        assert_eq!(pipeline.layout_id(), 9999);
    }

    // ══════════════════════════════════════════════════════════════════════════
    // COMPUTE PIPELINE CACHE TESTS — T-WGPU-P3.9.2
    // ══════════════════════════════════════════════════════════════════════════

    // ── ComputePipelineKey tests ─────────────────────────────────────────────

    #[test]
    fn test_pipeline_key_new() {
        let key = ComputePipelineKey::new(42, "main");
        assert_eq!(key.shader_id, 42);
        assert_eq!(key.entry_point, "main");
        assert!(key.specialization.is_empty());
    }

    #[test]
    fn test_pipeline_key_with_specialization() {
        let spec = SpecializationKey::new()
            .constant("SIZE", 64.0)
            .zero_init_workgroup(true);

        let key = ComputePipelineKey::with_specialization(100, "cs_main", spec);
        assert_eq!(key.shader_id, 100);
        assert_eq!(key.entry_point, "cs_main");
        assert!(!key.specialization.is_empty());
    }

    #[test]
    fn test_pipeline_key_equality() {
        let key1 = ComputePipelineKey::new(1, "main");
        let key2 = ComputePipelineKey::new(1, "main");
        let key3 = ComputePipelineKey::new(1, "other");
        let key4 = ComputePipelineKey::new(2, "main");

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
        assert_ne!(key1, key4);
    }

    #[test]
    fn test_pipeline_key_hash() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let key1 = ComputePipelineKey::new(1, "main");
        let key2 = ComputePipelineKey::new(1, "main");
        let key3 = ComputePipelineKey::new(2, "main");

        let hash = |k: &ComputePipelineKey| {
            let mut h = DefaultHasher::new();
            k.hash(&mut h);
            h.finish()
        };

        assert_eq!(hash(&key1), hash(&key2));
        assert_ne!(hash(&key1), hash(&key3));
    }

    #[test]
    fn test_pipeline_key_clone() {
        let key1 = ComputePipelineKey::with_specialization(
            42,
            "entry",
            SpecializationKey::new().constant("A", 1.0),
        );
        let key2 = key1.clone();

        assert_eq!(key1, key2);
    }

    #[test]
    fn test_pipeline_key_debug() {
        let key = ComputePipelineKey::new(123, "debug_entry");
        let debug_str = format!("{:?}", key);

        assert!(debug_str.contains("ComputePipelineKey"));
        assert!(debug_str.contains("123"));
        assert!(debug_str.contains("debug_entry"));
    }

    // ── SpecializationKey tests ──────────────────────────────────────────────

    #[test]
    fn test_specialization_key_new() {
        let spec = SpecializationKey::new();
        assert!(spec.is_empty());
        assert_eq!(spec.num_constants(), 0);
        assert!(!spec.should_zero_init_workgroup());
    }

    #[test]
    fn test_specialization_key_constant() {
        let spec = SpecializationKey::new()
            .constant("A", 1.0)
            .constant("B", 2.0);

        assert_eq!(spec.num_constants(), 2);
        assert!(!spec.is_empty());
    }

    #[test]
    fn test_specialization_key_constants_batch() {
        let constants = vec![
            (String::from("X"), 10.0),
            (String::from("Y"), 20.0),
            (String::from("Z"), 30.0),
        ];

        let spec = SpecializationKey::new().constants(constants);
        assert_eq!(spec.num_constants(), 3);
    }

    #[test]
    fn test_specialization_key_zero_init() {
        let spec = SpecializationKey::new().zero_init_workgroup(true);
        assert!(spec.should_zero_init_workgroup());
        assert!(!spec.is_empty());

        let spec2 = spec.zero_init_workgroup(false);
        assert!(!spec2.should_zero_init_workgroup());
    }

    #[test]
    fn test_specialization_key_to_constants_map() {
        let spec = SpecializationKey::new()
            .constant("BLOCK", 64.0)
            .constant("FLAG", 1.0);

        let map = spec.to_constants_map();
        assert_eq!(map.len(), 2);
        assert_eq!(map.get("BLOCK"), Some(&64.0));
        assert_eq!(map.get("FLAG"), Some(&1.0));
    }

    #[test]
    fn test_specialization_key_equality() {
        let spec1 = SpecializationKey::new()
            .constant("A", 1.0)
            .constant("B", 2.0);

        let spec2 = SpecializationKey::new()
            .constant("B", 2.0)
            .constant("A", 1.0);

        // BTreeMap ensures deterministic ordering
        assert_eq!(spec1, spec2);
    }

    #[test]
    fn test_specialization_key_inequality_by_constant_value() {
        let spec1 = SpecializationKey::new().constant("A", 1.0);
        let spec2 = SpecializationKey::new().constant("A", 2.0);

        assert_ne!(spec1, spec2);
    }

    #[test]
    fn test_specialization_key_inequality_by_zero_init() {
        let spec1 = SpecializationKey::new().zero_init_workgroup(true);
        let spec2 = SpecializationKey::new().zero_init_workgroup(false);

        assert_ne!(spec1, spec2);
    }

    #[test]
    fn test_specialization_key_hash_consistency() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let spec1 = SpecializationKey::new()
            .constant("A", 1.0)
            .constant("B", 2.0)
            .zero_init_workgroup(true);

        let spec2 = spec1.clone();

        let hash = |s: &SpecializationKey| {
            let mut h = DefaultHasher::new();
            s.hash(&mut h);
            h.finish()
        };

        assert_eq!(hash(&spec1), hash(&spec2));
    }

    #[test]
    fn test_specialization_key_special_floats() {
        // Test that special float values work with OrderedFloat
        let spec = SpecializationKey::new()
            .constant("ZERO", 0.0)
            .constant("NEG", -1.0)
            .constant("INF", f64::INFINITY)
            .constant("NEG_INF", f64::NEG_INFINITY);

        let map = spec.to_constants_map();
        assert_eq!(map.get("INF"), Some(&f64::INFINITY));
        assert_eq!(map.get("NEG_INF"), Some(&f64::NEG_INFINITY));
    }

    #[test]
    fn test_specialization_key_nan_handling() {
        // NaN should work with OrderedFloat (all NaNs are equal)
        let spec1 = SpecializationKey::new().constant("NAN", f64::NAN);
        let spec2 = SpecializationKey::new().constant("NAN", f64::NAN);

        // OrderedFloat treats all NaNs as equal
        assert_eq!(spec1, spec2);
    }

    // ── ComputePipelineCache tests ───────────────────────────────────────────

    #[test]
    fn test_cache_new() {
        let cache = ComputePipelineCache::new();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_cache_with_capacity() {
        let cache = ComputePipelineCache::with_capacity(64);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_default() {
        let cache: ComputePipelineCache = Default::default();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_get_missing() {
        let cache = ComputePipelineCache::new();
        let key = ComputePipelineKey::new(1, "main");
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn test_cache_contains_missing() {
        let cache = ComputePipelineCache::new();
        let key = ComputePipelineKey::new(1, "main");
        assert!(!cache.contains(&key));
    }

    #[test]
    fn test_cache_invalidate_missing() {
        let cache = ComputePipelineCache::new();
        let key = ComputePipelineKey::new(1, "main");
        assert!(!cache.invalidate(&key));
    }

    #[test]
    fn test_cache_invalidate_all_empty() {
        let cache = ComputePipelineCache::new();
        assert_eq!(cache.invalidate_all(), 0);
    }

    #[test]
    fn test_cache_invalidate_by_shader_empty() {
        let cache = ComputePipelineCache::new();
        assert_eq!(cache.invalidate_by_shader(42), 0);
    }

    #[test]
    fn test_cache_debug_impl() {
        let cache = ComputePipelineCache::new();
        let debug_str = format!("{:?}", cache);
        assert!(debug_str.contains("ComputePipelineCache"));
        assert!(debug_str.contains("0"));
    }

    #[test]
    fn test_cache_stats_empty() {
        let cache = ComputePipelineCache::new();
        let stats = cache.stats();

        assert_eq!(stats.total_pipelines, 0);
        assert_eq!(stats.unique_shaders, 0);
        assert_eq!(stats.unique_entry_points, 0);
    }

    #[test]
    fn test_cache_cached_shader_ids_empty() {
        let cache = ComputePipelineCache::new();
        assert!(cache.cached_shader_ids().is_empty());
    }

    // ── ComputePipelineCache GPU tests ───────────────────────────────────────

    #[test]
    fn test_cache_get_or_create_basic() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let cache = ComputePipelineCache::new();
        let key = ComputePipelineKey::new(1, "main");

        let wgsl = r#"
            @group(0) @binding(0) var<storage, read_write> data: array<f32>;
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                data[id.x] = data[id.x] * 2.0;
            }
        "#;

        let pipeline = cache.get_or_create(key.clone(), || {
            ComputePipelineDescriptor::from_wgsl(wgsl, "main")
                .label("cached_pipeline")
                .build(&device)
        });

        assert_eq!(pipeline.label(), Some("cached_pipeline"));
        assert!(cache.contains(&key));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_cache_get_or_create_returns_cached() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let cache = ComputePipelineCache::new();
        let key = ComputePipelineKey::new(1, "main");

        let wgsl = r#"
            @group(0) @binding(0) var<storage, read_write> data: array<f32>;
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                data[id.x] = data[id.x] * 2.0;
            }
        "#;

        let mut create_count = 0;

        let pipeline1 = cache.get_or_create(key.clone(), || {
            create_count += 1;
            ComputePipelineDescriptor::from_wgsl(wgsl, "main")
                .build(&device)
        });

        let pipeline2 = cache.get_or_create(key.clone(), || {
            create_count += 1;
            ComputePipelineDescriptor::from_wgsl(wgsl, "main")
                .build(&device)
        });

        // Should only have created once
        assert_eq!(create_count, 1);

        // Should be same Arc
        assert!(Arc::ptr_eq(&pipeline1, &pipeline2));
    }

    #[test]
    fn test_cache_get_after_insert() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let cache = ComputePipelineCache::new();
        let key = ComputePipelineKey::new(42, "cs_main");

        let wgsl = r#"
            @group(0) @binding(0) var<storage, read_write> data: array<f32>;
            @compute @workgroup_size(64)
            fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
                data[id.x] = data[id.x] + 1.0;
            }
        "#;

        // Insert via get_or_create
        let inserted = cache.get_or_create(key.clone(), || {
            ComputePipelineDescriptor::from_wgsl(wgsl, "cs_main")
                .build(&device)
        });

        // Retrieve via get
        let retrieved = cache.get(&key).expect("should be cached");

        assert!(Arc::ptr_eq(&inserted, &retrieved));
    }

    #[test]
    fn test_cache_invalidate_single() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let cache = ComputePipelineCache::new();
        let key = ComputePipelineKey::new(1, "main");

        let wgsl = r#"
            @group(0) @binding(0) var<storage, read_write> data: array<f32>;
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                data[id.x] = data[id.x] * 2.0;
            }
        "#;

        cache.get_or_create(key.clone(), || {
            ComputePipelineDescriptor::from_wgsl(wgsl, "main")
                .build(&device)
        });

        assert!(cache.contains(&key));
        assert!(cache.invalidate(&key));
        assert!(!cache.contains(&key));
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_invalidate_by_shader_selective() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let cache = ComputePipelineCache::new();

        let wgsl = r#"
            @group(0) @binding(0) var<storage, read_write> data: array<f32>;
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                data[id.x] = data[id.x] * 2.0;
            }
        "#;

        // Add pipelines with different shader IDs
        let key1 = ComputePipelineKey::new(100, "main");
        let key2 = ComputePipelineKey::with_specialization(
            100,
            "main",
            SpecializationKey::new().constant("A", 1.0),
        );
        let key3 = ComputePipelineKey::new(200, "main");

        cache.get_or_create(key1.clone(), || {
            ComputePipelineDescriptor::from_wgsl(wgsl, "main").build(&device)
        });
        cache.get_or_create(key2.clone(), || {
            ComputePipelineDescriptor::from_wgsl(wgsl, "main").build(&device)
        });
        cache.get_or_create(key3.clone(), || {
            ComputePipelineDescriptor::from_wgsl(wgsl, "main").build(&device)
        });

        assert_eq!(cache.len(), 3);

        // Invalidate shader 100
        let invalidated = cache.invalidate_by_shader(100);
        assert_eq!(invalidated, 2);
        assert_eq!(cache.len(), 1);

        // Shader 200 should still be cached
        assert!(cache.contains(&key3));
        assert!(!cache.contains(&key1));
        assert!(!cache.contains(&key2));
    }

    #[test]
    fn test_cache_invalidate_all() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let cache = ComputePipelineCache::new();

        let wgsl = r#"
            @group(0) @binding(0) var<storage, read_write> data: array<f32>;
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                data[id.x] = data[id.x] * 2.0;
            }
        "#;

        for i in 0..5 {
            let key = ComputePipelineKey::new(i, "main");
            cache.get_or_create(key, || {
                ComputePipelineDescriptor::from_wgsl(wgsl, "main").build(&device)
            });
        }

        assert_eq!(cache.len(), 5);

        let invalidated = cache.invalidate_all();
        assert_eq!(invalidated, 5);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_stats_with_content() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let cache = ComputePipelineCache::new();

        let wgsl = r#"
            @group(0) @binding(0) var<storage, read_write> data: array<f32>;
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                data[id.x] = data[id.x] * 2.0;
            }
        "#;

        // Same shader, different entry points
        cache.get_or_create(ComputePipelineKey::new(1, "main"), || {
            ComputePipelineDescriptor::from_wgsl(wgsl, "main").build(&device)
        });
        cache.get_or_create(ComputePipelineKey::new(1, "alt"), || {
            ComputePipelineDescriptor::from_wgsl(wgsl, "main").build(&device)
        });
        // Different shader
        cache.get_or_create(ComputePipelineKey::new(2, "main"), || {
            ComputePipelineDescriptor::from_wgsl(wgsl, "main").build(&device)
        });

        let stats = cache.stats();
        assert_eq!(stats.total_pipelines, 3);
        assert_eq!(stats.unique_shaders, 2);
        assert_eq!(stats.unique_entry_points, 2);
    }

    #[test]
    fn test_cache_cached_shader_ids() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let cache = ComputePipelineCache::new();

        let wgsl = r#"
            @group(0) @binding(0) var<storage, read_write> data: array<f32>;
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                data[id.x] = data[id.x] * 2.0;
            }
        "#;

        cache.get_or_create(ComputePipelineKey::new(10, "main"), || {
            ComputePipelineDescriptor::from_wgsl(wgsl, "main").build(&device)
        });
        cache.get_or_create(ComputePipelineKey::new(20, "main"), || {
            ComputePipelineDescriptor::from_wgsl(wgsl, "main").build(&device)
        });
        cache.get_or_create(ComputePipelineKey::new(10, "alt"), || {
            ComputePipelineDescriptor::from_wgsl(wgsl, "main").build(&device)
        });

        let ids = cache.cached_shader_ids();
        assert_eq!(ids, vec![10, 20]);
    }

    #[test]
    fn test_cache_thread_safety() {
        use std::thread;

        let cache = Arc::new(ComputePipelineCache::new());
        let mut handles = vec![];

        // Spawn multiple threads that read/write concurrently
        for i in 0..4 {
            let cache_clone = Arc::clone(&cache);
            handles.push(thread::spawn(move || {
                for j in 0..10 {
                    let key = ComputePipelineKey::new(i as u64, &format!("entry_{}", j));
                    assert!(!cache_clone.contains(&key));
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_cache_get_or_create_thread_safety() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        use std::thread;

        let cache = Arc::new(ComputePipelineCache::new());
        let device = Arc::new(device);
        let mut handles = vec![];

        let wgsl = r#"
            @group(0) @binding(0) var<storage, read_write> data: array<f32>;
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                data[id.x] = data[id.x] * 2.0;
            }
        "#;

        // Multiple threads try to create the same pipeline
        for _ in 0..4 {
            let cache_clone = Arc::clone(&cache);
            let device_clone = Arc::clone(&device);
            let wgsl_owned = wgsl.to_string();

            handles.push(thread::spawn(move || {
                let key = ComputePipelineKey::new(999, "main");
                cache_clone.get_or_create(key, || {
                    ComputePipelineDescriptor::from_wgsl(wgsl_owned.as_str(), "main")
                        .build(&device_clone)
                });
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Should only have one pipeline despite multiple threads
        assert_eq!(cache.len(), 1);
    }

    // ── Thread safety assertions ─────────────────────────────────────────────

    #[test]
    fn test_cache_types_are_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}

        assert_send_sync::<ComputePipelineCache>();
        assert_send_sync::<ComputePipelineKey>();
        assert_send_sync::<SpecializationKey>();
        assert_send_sync::<ComputePipelineCacheStats>();
    }

    // ══════════════════════════════════════════════════════════════════════════
    // ADDITIONAL WHITEBOX TESTS — T-WGPU-P3.9.2 (Comprehensive Coverage)
    // ══════════════════════════════════════════════════════════════════════════

    // ── ComputePipelineKey edge cases ────────────────────────────────────────

    #[test]
    fn test_pipeline_key_empty_entry_point() {
        let key = ComputePipelineKey::new(1, "");
        assert_eq!(key.entry_point, "");
        assert_eq!(key.shader_id, 1);
    }

    #[test]
    fn test_pipeline_key_unicode_entry_point() {
        let key = ComputePipelineKey::new(42, "计算_main_αβγ");
        assert_eq!(key.entry_point, "计算_main_αβγ");
    }

    #[test]
    fn test_pipeline_key_very_long_entry_point() {
        let long_name = "entry_".to_string() + &"x".repeat(10000);
        let key = ComputePipelineKey::new(1, long_name.clone());
        assert_eq!(key.entry_point.len(), 10006);
    }

    #[test]
    fn test_pipeline_key_max_shader_id() {
        let key = ComputePipelineKey::new(u64::MAX, "main");
        assert_eq!(key.shader_id, u64::MAX);
    }

    #[test]
    fn test_pipeline_key_zero_shader_id() {
        let key = ComputePipelineKey::new(0, "main");
        assert_eq!(key.shader_id, 0);
    }

    #[test]
    fn test_pipeline_key_whitespace_entry_point() {
        let key = ComputePipelineKey::new(1, "  spaced  ");
        assert_eq!(key.entry_point, "  spaced  ");
    }

    #[test]
    fn test_pipeline_key_hash_differs_by_entry_point() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let key1 = ComputePipelineKey::new(1, "main");
        let key2 = ComputePipelineKey::new(1, "other");

        let hash = |k: &ComputePipelineKey| {
            let mut h = DefaultHasher::new();
            k.hash(&mut h);
            h.finish()
        };

        assert_ne!(hash(&key1), hash(&key2));
    }

    #[test]
    fn test_pipeline_key_hash_differs_by_specialization() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let key1 = ComputePipelineKey::new(1, "main");
        let key2 = ComputePipelineKey::with_specialization(
            1,
            "main",
            SpecializationKey::new().constant("A", 1.0),
        );

        let hash = |k: &ComputePipelineKey| {
            let mut h = DefaultHasher::new();
            k.hash(&mut h);
            h.finish()
        };

        assert_ne!(hash(&key1), hash(&key2));
    }

    #[test]
    fn test_pipeline_key_partial_eq_reflexive() {
        let key = ComputePipelineKey::new(42, "main");
        assert_eq!(key, key.clone());
    }

    #[test]
    fn test_pipeline_key_partial_eq_symmetric() {
        let key1 = ComputePipelineKey::new(42, "main");
        let key2 = ComputePipelineKey::new(42, "main");
        assert_eq!(key1, key2);
        assert_eq!(key2, key1);
    }

    // ── SpecializationKey edge cases ─────────────────────────────────────────

    #[test]
    fn test_specialization_key_default() {
        let spec: SpecializationKey = Default::default();
        assert!(spec.is_empty());
        assert_eq!(spec.num_constants(), 0);
    }

    #[test]
    fn test_specialization_key_only_zero_init_not_empty() {
        // Just setting zero_init_workgroup should make it non-empty
        let spec = SpecializationKey::new().zero_init_workgroup(true);
        assert!(!spec.is_empty());
        assert_eq!(spec.num_constants(), 0);
    }

    #[test]
    fn test_specialization_key_ordering_determinism() {
        // BTreeMap should ensure deterministic iteration order
        let spec1 = SpecializationKey::new()
            .constant("Z", 26.0)
            .constant("A", 1.0)
            .constant("M", 13.0);

        let spec2 = SpecializationKey::new()
            .constant("A", 1.0)
            .constant("M", 13.0)
            .constant("Z", 26.0);

        // Order of insertion shouldn't matter for equality
        assert_eq!(spec1, spec2);

        // Hash should also be the same
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let hash = |s: &SpecializationKey| {
            let mut h = DefaultHasher::new();
            s.hash(&mut h);
            h.finish()
        };

        assert_eq!(hash(&spec1), hash(&spec2));
    }

    #[test]
    fn test_specialization_key_many_constants() {
        let mut spec = SpecializationKey::new();
        for i in 0..1000 {
            spec = spec.constant(format!("CONST_{}", i), i as f64);
        }
        assert_eq!(spec.num_constants(), 1000);
        assert!(!spec.is_empty());
    }

    #[test]
    fn test_specialization_key_negative_zero() {
        // Test that -0.0 and 0.0 are handled by OrderedFloat
        let spec1 = SpecializationKey::new().constant("VAL", 0.0);
        let spec2 = SpecializationKey::new().constant("VAL", -0.0);

        // IEEE 754: 0.0 == -0.0, so specs should be equal
        assert_eq!(spec1, spec2);
    }

    #[test]
    fn test_specialization_key_subnormal_floats() {
        // Test subnormal (denormalized) float values
        let tiny = f64::MIN_POSITIVE / 2.0; // Subnormal value
        let spec = SpecializationKey::new().constant("TINY", tiny);

        let map = spec.to_constants_map();
        let retrieved = map.get("TINY").unwrap();
        assert!(*retrieved > 0.0);
        assert!(*retrieved < f64::MIN_POSITIVE);
    }

    #[test]
    fn test_specialization_key_constant_override() {
        // Later constant should override earlier one
        let spec = SpecializationKey::new()
            .constant("VAL", 1.0)
            .constant("VAL", 2.0);

        let map = spec.to_constants_map();
        assert_eq!(map.get("VAL"), Some(&2.0));
        assert_eq!(spec.num_constants(), 1);
    }

    #[test]
    fn test_specialization_key_empty_constant_name() {
        let spec = SpecializationKey::new().constant("", 42.0);
        let map = spec.to_constants_map();
        assert_eq!(map.get(""), Some(&42.0));
    }

    #[test]
    fn test_specialization_key_unicode_constant_names() {
        let spec = SpecializationKey::new()
            .constant("サイズ", 64.0)
            .constant("размер", 128.0)
            .constant("大小", 256.0);

        assert_eq!(spec.num_constants(), 3);
        let map = spec.to_constants_map();
        assert_eq!(map.get("サイズ"), Some(&64.0));
        assert_eq!(map.get("размер"), Some(&128.0));
        assert_eq!(map.get("大小"), Some(&256.0));
    }

    #[test]
    fn test_specialization_key_clone_independence() {
        let spec1 = SpecializationKey::new()
            .constant("A", 1.0)
            .zero_init_workgroup(true);

        let spec2 = spec1.clone();

        // Modifying spec1's builder-style methods creates a new instance anyway,
        // but we verify clone gives us an independent copy
        assert_eq!(spec1, spec2);
        assert_eq!(spec1.num_constants(), spec2.num_constants());
        assert_eq!(
            spec1.should_zero_init_workgroup(),
            spec2.should_zero_init_workgroup()
        );
    }

    #[test]
    fn test_specialization_key_debug_output() {
        let spec = SpecializationKey::new()
            .constant("DEBUG_VAL", 42.0)
            .zero_init_workgroup(true);

        let debug_str = format!("{:?}", spec);
        assert!(debug_str.contains("SpecializationKey"));
        assert!(debug_str.contains("DEBUG_VAL"));
        assert!(debug_str.contains("42"));
    }

    // ── ComputePipelineCache additional tests ────────────────────────────────

    #[test]
    fn test_cache_multiple_different_keys() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let cache = ComputePipelineCache::new();

        let wgsl = r#"
            @group(0) @binding(0) var<storage, read_write> data: array<f32>;
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                data[id.x] = data[id.x] * 2.0;
            }
        "#;

        // Insert multiple pipelines with different keys
        let keys: Vec<ComputePipelineKey> = (0..10)
            .map(|i| ComputePipelineKey::new(i, format!("entry_{}", i)))
            .collect();

        for key in &keys {
            cache.get_or_create(key.clone(), || {
                ComputePipelineDescriptor::from_wgsl(wgsl, "main")
                    .label(&format!("pipeline_{}", key.shader_id))
                    .build(&device)
            });
        }

        assert_eq!(cache.len(), 10);

        // Verify all are cached
        for key in &keys {
            assert!(cache.contains(key));
        }

        // Verify each returns correct pipeline
        for key in &keys {
            let pipeline = cache.get(key).unwrap();
            assert_eq!(
                pipeline.label(),
                Some(format!("pipeline_{}", key.shader_id).as_str())
            );
        }
    }

    #[test]
    fn test_cache_reinsertion_after_invalidation() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let cache = ComputePipelineCache::new();
        let key = ComputePipelineKey::new(1, "main");

        let wgsl = r#"
            @group(0) @binding(0) var<storage, read_write> data: array<f32>;
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                data[id.x] = data[id.x] * 2.0;
            }
        "#;

        // Insert
        let pipeline1 = cache.get_or_create(key.clone(), || {
            ComputePipelineDescriptor::from_wgsl(wgsl, "main")
                .label("first")
                .build(&device)
        });
        assert_eq!(pipeline1.label(), Some("first"));

        // Invalidate
        assert!(cache.invalidate(&key));
        assert!(!cache.contains(&key));

        // Re-insert with different label
        let pipeline2 = cache.get_or_create(key.clone(), || {
            ComputePipelineDescriptor::from_wgsl(wgsl, "main")
                .label("second")
                .build(&device)
        });
        assert_eq!(pipeline2.label(), Some("second"));

        // Should be different Arc (different pipeline)
        assert!(!Arc::ptr_eq(&pipeline1, &pipeline2));
    }

    #[test]
    fn test_cache_stats_after_invalidation() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let cache = ComputePipelineCache::new();

        let wgsl = r#"
            @group(0) @binding(0) var<storage, read_write> data: array<f32>;
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                data[id.x] = data[id.x] * 2.0;
            }
        "#;

        // Add 5 pipelines
        for i in 0..5 {
            cache.get_or_create(ComputePipelineKey::new(i, "main"), || {
                ComputePipelineDescriptor::from_wgsl(wgsl, "main").build(&device)
            });
        }

        let stats_before = cache.stats();
        assert_eq!(stats_before.total_pipelines, 5);
        assert_eq!(stats_before.unique_shaders, 5);

        // Invalidate some
        cache.invalidate_by_shader(2);
        cache.invalidate_by_shader(4);

        let stats_after = cache.stats();
        assert_eq!(stats_after.total_pipelines, 3);
        assert_eq!(stats_after.unique_shaders, 3);
    }

    #[test]
    fn test_cache_with_specialization_variants() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let cache = ComputePipelineCache::new();

        let wgsl = r#"
            @group(0) @binding(0) var<storage, read_write> data: array<f32>;
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                data[id.x] = data[id.x] * 2.0;
            }
        "#;

        // Same shader, same entry point, different specialization
        let key1 = ComputePipelineKey::new(1, "main");
        let key2 = ComputePipelineKey::with_specialization(
            1,
            "main",
            SpecializationKey::new().constant("SIZE", 64.0),
        );
        let key3 = ComputePipelineKey::with_specialization(
            1,
            "main",
            SpecializationKey::new().constant("SIZE", 128.0),
        );
        let key4 = ComputePipelineKey::with_specialization(
            1,
            "main",
            SpecializationKey::new().zero_init_workgroup(true),
        );

        cache.get_or_create(key1.clone(), || {
            ComputePipelineDescriptor::from_wgsl(wgsl, "main")
                .label("no_spec")
                .build(&device)
        });
        cache.get_or_create(key2.clone(), || {
            ComputePipelineDescriptor::from_wgsl(wgsl, "main")
                .label("size_64")
                .build(&device)
        });
        cache.get_or_create(key3.clone(), || {
            ComputePipelineDescriptor::from_wgsl(wgsl, "main")
                .label("size_128")
                .build(&device)
        });
        cache.get_or_create(key4.clone(), || {
            ComputePipelineDescriptor::from_wgsl(wgsl, "main")
                .label("zero_init")
                .build(&device)
        });

        assert_eq!(cache.len(), 4);

        // All keys should be distinct
        assert!(cache.contains(&key1));
        assert!(cache.contains(&key2));
        assert!(cache.contains(&key3));
        assert!(cache.contains(&key4));

        // Verify correct pipelines retrieved
        assert_eq!(cache.get(&key1).unwrap().label(), Some("no_spec"));
        assert_eq!(cache.get(&key2).unwrap().label(), Some("size_64"));
        assert_eq!(cache.get(&key3).unwrap().label(), Some("size_128"));
        assert_eq!(cache.get(&key4).unwrap().label(), Some("zero_init"));

        // Stats should show 1 unique shader
        let stats = cache.stats();
        assert_eq!(stats.unique_shaders, 1);
        assert_eq!(stats.total_pipelines, 4);
    }

    #[test]
    fn test_cache_double_checked_locking() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::thread;

        let cache = Arc::new(ComputePipelineCache::new());
        let device = Arc::new(device);
        let create_count = Arc::new(AtomicUsize::new(0));

        let wgsl = r#"
            @group(0) @binding(0) var<storage, read_write> data: array<f32>;
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                data[id.x] = data[id.x] * 2.0;
            }
        "#;

        let mut handles = vec![];

        // Multiple threads try to create the same pipeline simultaneously
        for _ in 0..8 {
            let cache_clone = Arc::clone(&cache);
            let device_clone = Arc::clone(&device);
            let create_count_clone = Arc::clone(&create_count);
            let wgsl_owned = wgsl.to_string();

            handles.push(thread::spawn(move || {
                let key = ComputePipelineKey::new(42, "main");
                cache_clone.get_or_create(key, || {
                    create_count_clone.fetch_add(1, Ordering::SeqCst);
                    ComputePipelineDescriptor::from_wgsl(wgsl_owned.as_str(), "main")
                        .build(&device_clone)
                });
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Due to double-checked locking, create should only be called once
        // (or possibly a few times in a race, but definitely not 8 times)
        let count = create_count.load(Ordering::SeqCst);
        assert!(count <= 2, "Create was called {} times, expected 1-2", count);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_cache_debug_with_content() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let cache = ComputePipelineCache::new();

        let wgsl = r#"
            @group(0) @binding(0) var<storage, read_write> data: array<f32>;
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                data[id.x] = data[id.x] * 2.0;
            }
        "#;

        cache.get_or_create(ComputePipelineKey::new(1, "main"), || {
            ComputePipelineDescriptor::from_wgsl(wgsl, "main").build(&device)
        });
        cache.get_or_create(ComputePipelineKey::new(2, "main"), || {
            ComputePipelineDescriptor::from_wgsl(wgsl, "main").build(&device)
        });

        let debug_str = format!("{:?}", cache);
        assert!(debug_str.contains("ComputePipelineCache"));
        assert!(debug_str.contains("2")); // count = 2
    }

    // ── ComputePipelineCacheStats tests ──────────────────────────────────────

    #[test]
    fn test_cache_stats_debug() {
        let stats = ComputePipelineCacheStats {
            total_pipelines: 10,
            unique_shaders: 5,
            unique_entry_points: 3,
        };

        let debug_str = format!("{:?}", stats);
        assert!(debug_str.contains("ComputePipelineCacheStats"));
        assert!(debug_str.contains("10"));
        assert!(debug_str.contains("5"));
        assert!(debug_str.contains("3"));
    }

    #[test]
    fn test_cache_stats_clone() {
        let stats1 = ComputePipelineCacheStats {
            total_pipelines: 10,
            unique_shaders: 5,
            unique_entry_points: 3,
        };

        let stats2 = stats1;

        assert_eq!(stats1, stats2);
    }

    #[test]
    fn test_cache_stats_equality() {
        let stats1 = ComputePipelineCacheStats {
            total_pipelines: 10,
            unique_shaders: 5,
            unique_entry_points: 3,
        };

        let stats2 = ComputePipelineCacheStats {
            total_pipelines: 10,
            unique_shaders: 5,
            unique_entry_points: 3,
        };

        let stats3 = ComputePipelineCacheStats {
            total_pipelines: 11,
            unique_shaders: 5,
            unique_entry_points: 3,
        };

        assert_eq!(stats1, stats2);
        assert_ne!(stats1, stats3);
    }

    // ── RwLock behavior tests ────────────────────────────────────────────────

    #[test]
    fn test_cache_concurrent_reads() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        use std::thread;

        let cache = Arc::new(ComputePipelineCache::new());
        let device = Arc::new(device);

        let wgsl = r#"
            @group(0) @binding(0) var<storage, read_write> data: array<f32>;
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                data[id.x] = data[id.x] * 2.0;
            }
        "#;

        // Pre-populate cache
        let key = ComputePipelineKey::new(1, "main");
        cache.get_or_create(key.clone(), || {
            ComputePipelineDescriptor::from_wgsl(wgsl, "main").build(&device)
        });

        let mut handles = vec![];

        // Multiple threads reading concurrently
        for _ in 0..10 {
            let cache_clone = Arc::clone(&cache);
            let key_clone = key.clone();

            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    assert!(cache_clone.contains(&key_clone));
                    let _ = cache_clone.get(&key_clone);
                    let _ = cache_clone.len();
                    let _ = cache_clone.is_empty();
                    let _ = cache_clone.stats();
                    let _ = cache_clone.cached_shader_ids();
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_cache_mixed_read_write_operations() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        use std::thread;

        let cache = Arc::new(ComputePipelineCache::new());
        let device = Arc::new(device);

        let wgsl = r#"
            @group(0) @binding(0) var<storage, read_write> data: array<f32>;
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                data[id.x] = data[id.x] * 2.0;
            }
        "#;

        let mut handles = vec![];

        // Writer threads
        for i in 0..4 {
            let cache_clone = Arc::clone(&cache);
            let device_clone = Arc::clone(&device);
            let wgsl_owned = wgsl.to_string();

            handles.push(thread::spawn(move || {
                for j in 0..10 {
                    let key = ComputePipelineKey::new(i * 10 + j, "main");
                    cache_clone.get_or_create(key, || {
                        ComputePipelineDescriptor::from_wgsl(wgsl_owned.as_str(), "main")
                            .build(&device_clone)
                    });
                }
            }));
        }

        // Reader threads
        for _ in 0..4 {
            let cache_clone = Arc::clone(&cache);

            handles.push(thread::spawn(move || {
                for _ in 0..50 {
                    let _ = cache_clone.len();
                    let _ = cache_clone.stats();
                    for i in 0..40 {
                        let key = ComputePipelineKey::new(i, "main");
                        let _ = cache_clone.get(&key);
                    }
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Should have 40 pipelines (4 writers * 10 each)
        assert_eq!(cache.len(), 40);
    }

    // ── Trait implementations comprehensive tests ────────────────────────────

    #[test]
    fn test_pipeline_key_eq_transitive() {
        let key1 = ComputePipelineKey::new(1, "main");
        let key2 = ComputePipelineKey::new(1, "main");
        let key3 = ComputePipelineKey::new(1, "main");

        // Transitive: if a == b and b == c, then a == c
        assert_eq!(key1, key2);
        assert_eq!(key2, key3);
        assert_eq!(key1, key3);
    }

    #[test]
    fn test_specialization_key_eq_transitive() {
        let spec1 = SpecializationKey::new().constant("A", 1.0);
        let spec2 = SpecializationKey::new().constant("A", 1.0);
        let spec3 = SpecializationKey::new().constant("A", 1.0);

        assert_eq!(spec1, spec2);
        assert_eq!(spec2, spec3);
        assert_eq!(spec1, spec3);
    }

    #[test]
    fn test_pipeline_key_in_hashmap() {
        let mut map = HashMap::new();

        let key1 = ComputePipelineKey::new(1, "main");
        let key2 = ComputePipelineKey::new(2, "main");
        let key3 = ComputePipelineKey::new(1, "main"); // Same as key1

        map.insert(key1.clone(), "first");
        map.insert(key2.clone(), "second");

        assert_eq!(map.get(&key1), Some(&"first"));
        assert_eq!(map.get(&key2), Some(&"second"));
        assert_eq!(map.get(&key3), Some(&"first")); // key3 == key1

        // Insert with key3 should overwrite key1's value
        map.insert(key3, "third");
        assert_eq!(map.get(&key1), Some(&"third"));
    }

    #[test]
    fn test_specialization_key_in_hashset() {
        use std::collections::HashSet;

        let mut set = HashSet::new();

        let spec1 = SpecializationKey::new()
            .constant("A", 1.0)
            .constant("B", 2.0);
        let spec2 = SpecializationKey::new()
            .constant("B", 2.0)
            .constant("A", 1.0); // Same as spec1, different order
        let spec3 = SpecializationKey::new().constant("A", 1.0);

        set.insert(spec1.clone());
        set.insert(spec2.clone()); // Should not increase size (same as spec1)
        set.insert(spec3.clone());

        assert_eq!(set.len(), 2);
        assert!(set.contains(&spec1));
        assert!(set.contains(&spec2));
        assert!(set.contains(&spec3));
    }
}
