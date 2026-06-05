//! Render bundle recording and caching for wgpu 22.x.
//!
//! Render bundles allow pre-recording of draw commands that can be replayed
//! multiple times without re-encoding. This is ideal for static geometry,
//! UI elements, and shadow map rendering where draw commands don't change
//! between frames.
//!
//! # Overview
//!
//! Render bundles provide two main benefits:
//! 1. **Reduced CPU overhead** - Commands are validated once at bundle creation
//! 2. **Consistent command stream** - Same commands guaranteed across replay
//!
//! # Architecture
//!
//! ```text
//! RenderBundleEncoderDescriptor
//!     |-- label: Option<String>
//!     |-- color_formats: Vec<Option<TextureFormat>>
//!     |-- depth_stencil: Option<RenderBundleDepthStencil>
//!     |-- sample_count: u32
//!     `-- multiview: Option<NonZeroU32>
//!
//! RenderBundleRecorder<'a>
//!     |-- encoder: wgpu::RenderBundleEncoder<'a>
//!     |-- Recording Methods (same as RenderPass)
//!     |   |-- set_pipeline()
//!     |   |-- set_bind_group()
//!     |   |-- set_vertex_buffer()
//!     |   |-- set_index_buffer()
//!     |   |-- draw()
//!     |   |-- draw_indexed()
//!     |   |-- draw_indirect()
//!     |   `-- draw_indexed_indirect()
//!     `-- finish() -> wgpu::RenderBundle
//!
//! RenderBundleCache
//!     |-- bundles: HashMap<BundleKey, Arc<wgpu::RenderBundle>>
//!     |-- get_or_create()
//!     |-- invalidate()
//!     `-- invalidate_all()
//! ```
//!
//! # Use Cases
//!
//! | Use Case | Benefit |
//! |----------|---------|
//! | Static Geometry | Record once, replay every frame |
//! | UI Rendering | Pre-encode common UI batches |
//! | Shadow Maps | Cache shadow caster draw commands |
//! | G-Buffer | Record static mesh draw calls |
//!
//! # Compatibility Notes (wgpu 22.x)
//!
//! - `RenderBundleEncoder` is created via `device.create_render_bundle_encoder()`
//! - Bundle must match the render pass format (color, depth, samples)
//! - Bundles can be executed in any render pass with matching format
//! - Push constants are NOT supported in render bundles
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::render_pipeline::render_bundle::{
//!     RenderBundleEncoderDescriptor, RenderBundleRecorder, RenderBundleCache,
//!     BundleKey, execute_bundles,
//! };
//!
//! // Create bundle descriptor matching render pass format
//! let desc = RenderBundleEncoderDescriptor::new()
//!     .label("static_geometry")
//!     .color_format(wgpu::TextureFormat::Bgra8UnormSrgb)
//!     .depth_stencil(wgpu::TextureFormat::Depth32Float, false, false)
//!     .sample_count(4);
//!
//! // Record draw commands
//! let mut recorder = RenderBundleRecorder::new(&device, &desc);
//! recorder.set_pipeline(&pipeline);
//! recorder.set_bind_group(0, &bind_group, &[]);
//! recorder.set_vertex_buffer(0, vertex_buffer.slice(..));
//! recorder.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
//! recorder.draw_indexed(0..36, 0, 0..1);
//!
//! // Finish and cache
//! let bundle = recorder.finish();
//!
//! // Execute in render pass
//! execute_bundles(&mut render_pass, &[&bundle]);
//! ```
//!
//! # Thread Safety
//!
//! - `RenderBundleCache` is thread-safe via `RwLock`
//! - `RenderBundleRecorder` is NOT `Send`/`Sync` (like `RenderPass`)
//! - Finished `RenderBundle` IS `Send + Sync`

use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::num::NonZeroU32;
use std::ops::Range;
use std::sync::{Arc, RwLock};

// ============================================================================
// RenderBundleEncoderDescriptor
// ============================================================================

/// Descriptor for creating a render bundle encoder.
///
/// This specifies the format that the render bundle must be compatible with.
/// The bundle can only be executed in render passes with matching:
/// - Color attachment formats
/// - Depth/stencil format
/// - Sample count
/// - Multiview configuration
///
/// # Example
///
/// ```ignore
/// let desc = RenderBundleEncoderDescriptor::new()
///     .label("my_bundle")
///     .color_format(wgpu::TextureFormat::Bgra8UnormSrgb)
///     .depth_stencil(wgpu::TextureFormat::Depth32Float, true, false)
///     .sample_count(4);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderBundleEncoderDescriptor {
    /// Debug label for the bundle encoder.
    pub label: Option<String>,

    /// Format of color attachments.
    ///
    /// `None` entries represent unused attachment slots.
    /// Order must match render pass color attachments.
    pub color_formats: Vec<Option<wgpu::TextureFormat>>,

    /// Depth/stencil configuration.
    ///
    /// Must match render pass depth attachment format and read-only state.
    pub depth_stencil: Option<wgpu::RenderBundleDepthStencil>,

    /// Sample count for MSAA.
    ///
    /// Must be 1, 2, 4, 8, or 16. Must match render pass sample count.
    pub sample_count: u32,

    /// Multiview array layers.
    ///
    /// For VR/stereo rendering. Must match render pass configuration.
    pub multiview: Option<NonZeroU32>,
}

impl Default for RenderBundleEncoderDescriptor {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderBundleEncoderDescriptor {
    /// Create a new descriptor with default settings.
    ///
    /// Default: no label, no color formats, no depth, 1 sample, no multiview.
    pub fn new() -> Self {
        Self {
            label: None,
            color_formats: Vec::new(),
            depth_stencil: None,
            sample_count: 1,
            multiview: None,
        }
    }

    /// Set the debug label.
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Add a color format to the attachment list.
    ///
    /// Call multiple times for multiple render targets (MRT).
    pub fn color_format(mut self, format: wgpu::TextureFormat) -> Self {
        self.color_formats.push(Some(format));
        self
    }

    /// Add an optional color format (for sparse MRT).
    pub fn color_format_opt(mut self, format: Option<wgpu::TextureFormat>) -> Self {
        self.color_formats.push(format);
        self
    }

    /// Set all color formats at once.
    pub fn color_formats(mut self, formats: Vec<Option<wgpu::TextureFormat>>) -> Self {
        self.color_formats = formats;
        self
    }

    /// Set depth/stencil configuration.
    ///
    /// # Parameters
    /// - `format`: The depth/stencil texture format
    /// - `depth_read_only`: If true, depth will not be written
    /// - `stencil_read_only`: If true, stencil will not be written
    pub fn depth_stencil(
        mut self,
        format: wgpu::TextureFormat,
        depth_read_only: bool,
        stencil_read_only: bool,
    ) -> Self {
        self.depth_stencil = Some(wgpu::RenderBundleDepthStencil {
            format,
            depth_read_only,
            stencil_read_only,
        });
        self
    }

    /// Set depth/stencil from a raw wgpu struct.
    pub fn depth_stencil_raw(mut self, ds: wgpu::RenderBundleDepthStencil) -> Self {
        self.depth_stencil = Some(ds);
        self
    }

    /// Remove depth/stencil attachment.
    pub fn no_depth_stencil(mut self) -> Self {
        self.depth_stencil = None;
        self
    }

    /// Set sample count for MSAA.
    ///
    /// # Panics
    /// In debug builds, panics if count is not 1, 2, 4, 8, or 16.
    pub fn sample_count(mut self, count: u32) -> Self {
        debug_assert!(
            matches!(count, 1 | 2 | 4 | 8 | 16),
            "Sample count must be 1, 2, 4, 8, or 16"
        );
        self.sample_count = count;
        self
    }

    /// Enable multiview rendering with specified array layers.
    pub fn multiview(mut self, array_layers: NonZeroU32) -> Self {
        self.multiview = Some(array_layers);
        self
    }

    /// Disable multiview rendering.
    pub fn no_multiview(mut self) -> Self {
        self.multiview = None;
        self
    }

    /// Convert to wgpu descriptor for encoder creation.
    pub fn to_wgpu(&self) -> wgpu::RenderBundleEncoderDescriptor<'_> {
        wgpu::RenderBundleEncoderDescriptor {
            label: self.label.as_deref(),
            color_formats: &self.color_formats,
            depth_stencil: self.depth_stencil,
            sample_count: self.sample_count,
            multiview: self.multiview,
        }
    }

    /// Validate the descriptor.
    ///
    /// Returns an error if:
    /// - Sample count is invalid
    /// - Too many color attachments (>8)
    pub fn validate(&self) -> Result<(), RenderBundleError> {
        if !matches!(self.sample_count, 1 | 2 | 4 | 8 | 16) {
            return Err(RenderBundleError::InvalidSampleCount(self.sample_count));
        }
        if self.color_formats.len() > 8 {
            return Err(RenderBundleError::TooManyColorAttachments(
                self.color_formats.len(),
            ));
        }
        Ok(())
    }
}

impl fmt::Display for RenderBundleEncoderDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RenderBundleEncoderDescriptor {{ label: {:?}, colors: {}, depth: {}, samples: {}, multiview: {:?} }}",
            self.label,
            self.color_formats.len(),
            self.depth_stencil.is_some(),
            self.sample_count,
            self.multiview
        )
    }
}

// ============================================================================
// RenderBundleRecorder
// ============================================================================

/// A recorder for creating render bundles.
///
/// Wraps `wgpu::RenderBundleEncoder` and provides the same draw command
/// interface as `wgpu::RenderPass`. Once recording is complete, call
/// [`finish()`](Self::finish) to produce the final `RenderBundle`.
///
/// # Supported Commands
///
/// | Command | Description |
/// |---------|-------------|
/// | `set_pipeline` | Set render pipeline |
/// | `set_bind_group` | Bind a bind group |
/// | `set_vertex_buffer` | Set vertex buffer |
/// | `set_index_buffer` | Set index buffer |
/// | `draw` | Draw non-indexed |
/// | `draw_indexed` | Draw indexed |
/// | `draw_indirect` | GPU-driven draw |
/// | `draw_indexed_indirect` | GPU-driven indexed draw |
///
/// # NOT Supported
///
/// - Push constants (not available in bundles)
/// - Viewport/scissor changes (set by render pass)
/// - Blend constants (set by render pass)
/// - Stencil reference (set by render pass)
///
/// # Example
///
/// ```ignore
/// let mut recorder = RenderBundleRecorder::new(&device, &descriptor);
/// recorder.set_pipeline(&pipeline);
/// recorder.set_bind_group(0, &bind_group, &[]);
/// recorder.set_vertex_buffer(0, buffer.slice(..));
/// recorder.draw(0..3, 0..1);
/// let bundle = recorder.finish();
/// ```
pub struct RenderBundleRecorder<'a> {
    encoder: wgpu::RenderBundleEncoder<'a>,
    label: Option<String>,
}

impl<'a> RenderBundleRecorder<'a> {
    /// Create a new bundle recorder.
    ///
    /// # Parameters
    /// - `device`: The wgpu device
    /// - `desc`: Descriptor specifying render target formats
    pub fn new(device: &'a wgpu::Device, desc: &RenderBundleEncoderDescriptor) -> Self {
        let encoder = device.create_render_bundle_encoder(&desc.to_wgpu());
        Self {
            encoder,
            label: desc.label.clone(),
        }
    }

    /// Get the label of this recorder.
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    // -------------------------------------------------------------------------
    // Pipeline State
    // -------------------------------------------------------------------------

    /// Set the render pipeline.
    ///
    /// This determines the shader programs and fixed-function state used
    /// for subsequent draw calls.
    pub fn set_pipeline(&mut self, pipeline: &'a wgpu::RenderPipeline) {
        self.encoder.set_pipeline(pipeline);
    }

    // -------------------------------------------------------------------------
    // Bind Groups
    // -------------------------------------------------------------------------

    /// Set a bind group at the specified index.
    ///
    /// # Parameters
    /// - `index`: Bind group index (0-3)
    /// - `bind_group`: The bind group to bind
    /// - `offsets`: Dynamic buffer offsets (for uniform buffers with dynamic offset)
    pub fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: &'a wgpu::BindGroup,
        offsets: &[wgpu::DynamicOffset],
    ) {
        self.encoder.set_bind_group(index, bind_group, offsets);
    }

    // -------------------------------------------------------------------------
    // Vertex/Index Buffers
    // -------------------------------------------------------------------------

    /// Set a vertex buffer at the specified slot.
    ///
    /// # Parameters
    /// - `slot`: Vertex buffer slot (typically 0-7)
    /// - `buffer_slice`: The buffer data to bind
    pub fn set_vertex_buffer(&mut self, slot: u32, buffer_slice: wgpu::BufferSlice<'a>) {
        self.encoder.set_vertex_buffer(slot, buffer_slice);
    }

    /// Set the index buffer.
    ///
    /// # Parameters
    /// - `buffer_slice`: The index buffer data
    /// - `format`: Index format (Uint16 or Uint32)
    pub fn set_index_buffer(
        &mut self,
        buffer_slice: wgpu::BufferSlice<'a>,
        format: wgpu::IndexFormat,
    ) {
        self.encoder.set_index_buffer(buffer_slice, format);
    }

    // -------------------------------------------------------------------------
    // Basic Draw Commands
    // -------------------------------------------------------------------------

    /// Draw non-indexed geometry.
    ///
    /// # Parameters
    /// - `vertices`: Range of vertices to draw
    /// - `instances`: Range of instances to draw
    pub fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        self.encoder.draw(vertices, instances);
    }

    /// Draw indexed geometry.
    ///
    /// # Parameters
    /// - `indices`: Range of indices to draw
    /// - `base_vertex`: Value added to each index
    /// - `instances`: Range of instances to draw
    pub fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
        self.encoder.draw_indexed(indices, base_vertex, instances);
    }

    // -------------------------------------------------------------------------
    // Indirect Draw Commands
    // -------------------------------------------------------------------------

    /// Draw with parameters read from a GPU buffer.
    ///
    /// The buffer must contain `DrawIndirectArgs` at the specified offset:
    /// ```text
    /// struct DrawIndirectArgs {
    ///     vertex_count: u32,
    ///     instance_count: u32,
    ///     first_vertex: u32,
    ///     first_instance: u32,
    /// }
    /// ```
    ///
    /// # Parameters
    /// - `indirect_buffer`: Buffer containing draw arguments
    /// - `indirect_offset`: Byte offset to the arguments (must be 4-byte aligned)
    pub fn draw_indirect(&mut self, indirect_buffer: &'a wgpu::Buffer, indirect_offset: u64) {
        self.encoder.draw_indirect(indirect_buffer, indirect_offset);
    }

    /// Draw indexed with parameters read from a GPU buffer.
    ///
    /// The buffer must contain `DrawIndexedIndirectArgs` at the specified offset:
    /// ```text
    /// struct DrawIndexedIndirectArgs {
    ///     index_count: u32,
    ///     instance_count: u32,
    ///     first_index: u32,
    ///     base_vertex: i32,
    ///     first_instance: u32,
    /// }
    /// ```
    ///
    /// # Parameters
    /// - `indirect_buffer`: Buffer containing draw arguments
    /// - `indirect_offset`: Byte offset to the arguments (must be 4-byte aligned)
    pub fn draw_indexed_indirect(
        &mut self,
        indirect_buffer: &'a wgpu::Buffer,
        indirect_offset: u64,
    ) {
        self.encoder
            .draw_indexed_indirect(indirect_buffer, indirect_offset);
    }

    // -------------------------------------------------------------------------
    // Finish
    // -------------------------------------------------------------------------

    /// Finish recording and produce the final render bundle.
    ///
    /// Consumes the recorder. The returned bundle can be executed in any
    /// render pass with matching format configuration.
    ///
    /// # Parameters
    /// - `desc`: Descriptor for the finished bundle (currently only label)
    pub fn finish(self) -> wgpu::RenderBundle {
        let desc = wgpu::RenderBundleDescriptor {
            label: self.label.as_deref(),
        };
        self.encoder.finish(&desc)
    }

    /// Finish recording with a custom label.
    pub fn finish_with_label(self, label: &str) -> wgpu::RenderBundle {
        let desc = wgpu::RenderBundleDescriptor { label: Some(label) };
        self.encoder.finish(&desc)
    }
}

impl fmt::Debug for RenderBundleRecorder<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RenderBundleRecorder")
            .field("label", &self.label)
            .finish_non_exhaustive()
    }
}

// ============================================================================
// BundleKey - Cache Key
// ============================================================================

/// Key for identifying cached render bundles.
///
/// A bundle key uniquely identifies a set of draw commands that can be cached.
/// The key should capture all inputs that affect the bundle content.
///
/// # Typical Key Components
///
/// - Mesh/geometry identifier
/// - Material/pipeline identifier
/// - Instance configuration
/// - LOD level
///
/// # Example
///
/// ```
/// use renderer_backend::render_pipeline::render_bundle::BundleKey;
///
/// // Simple mesh-based key
/// let key = BundleKey::from_u64(mesh_id);
///
/// // Compound key from multiple values
/// let key = BundleKey::from_parts(&[mesh_id, material_id, lod_level as u64]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BundleKey {
    /// The key data.
    data: BundleKeyData,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum BundleKeyData {
    /// Single u64 identifier.
    Simple(u64),
    /// Multiple u64 identifiers combined.
    Compound(Vec<u64>),
    /// String-based key (for named bundles).
    Named(String),
}

impl BundleKey {
    /// Create a key from a single u64 value.
    pub fn from_u64(id: u64) -> Self {
        Self {
            data: BundleKeyData::Simple(id),
        }
    }

    /// Create a key from multiple u64 values.
    ///
    /// Useful for compound keys like (mesh_id, material_id, lod).
    pub fn from_parts(parts: &[u64]) -> Self {
        if parts.len() == 1 {
            Self {
                data: BundleKeyData::Simple(parts[0]),
            }
        } else {
            Self {
                data: BundleKeyData::Compound(parts.to_vec()),
            }
        }
    }

    /// Create a key from a string name.
    pub fn from_name(name: impl Into<String>) -> Self {
        Self {
            data: BundleKeyData::Named(name.into()),
        }
    }

    /// Create a key by hashing arbitrary data.
    pub fn from_hash<T: Hash>(value: &T) -> Self {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        Self {
            data: BundleKeyData::Simple(hasher.finish()),
        }
    }
}

impl fmt::Display for BundleKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.data {
            BundleKeyData::Simple(id) => write!(f, "BundleKey({})", id),
            BundleKeyData::Compound(parts) => write!(f, "BundleKey({:?})", parts),
            BundleKeyData::Named(name) => write!(f, "BundleKey(\"{}\")", name),
        }
    }
}

impl From<u64> for BundleKey {
    fn from(id: u64) -> Self {
        Self::from_u64(id)
    }
}

impl From<&str> for BundleKey {
    fn from(name: &str) -> Self {
        Self::from_name(name)
    }
}

impl From<String> for BundleKey {
    fn from(name: String) -> Self {
        Self::from_name(name)
    }
}

// ============================================================================
// RenderBundleCache
// ============================================================================

/// Thread-safe cache for render bundles.
///
/// Stores render bundles by key and provides get-or-create semantics.
/// Uses `RwLock` for thread safety with read-biased access pattern.
///
/// # Thread Safety
///
/// - Multiple readers can access cached bundles concurrently
/// - Writers (create, invalidate) acquire exclusive lock
/// - Bundles are stored as `Arc` for safe cross-thread sharing
///
/// # Memory Management
///
/// The cache does not implement automatic eviction. Use [`invalidate`](Self::invalidate)
/// or [`invalidate_all`](Self::invalidate_all) to remove entries manually.
///
/// # Example
///
/// ```ignore
/// let cache = RenderBundleCache::new();
///
/// // Get or create a bundle
/// let bundle = cache.get_or_create(
///     BundleKey::from_u64(mesh_id),
///     || {
///         let mut recorder = RenderBundleRecorder::new(&device, &desc);
///         recorder.set_pipeline(&pipeline);
///         // ... record commands ...
///         recorder.finish()
///     }
/// );
///
/// // Invalidate when mesh changes
/// cache.invalidate(&BundleKey::from_u64(mesh_id));
/// ```
pub struct RenderBundleCache {
    bundles: RwLock<HashMap<BundleKey, Arc<wgpu::RenderBundle>>>,
}

impl Default for RenderBundleCache {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderBundleCache {
    /// Create a new empty cache.
    pub fn new() -> Self {
        Self {
            bundles: RwLock::new(HashMap::new()),
        }
    }

    /// Create a cache with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            bundles: RwLock::new(HashMap::with_capacity(capacity)),
        }
    }

    /// Get a cached bundle, or create and cache it if not present.
    ///
    /// # Parameters
    /// - `key`: The bundle key
    /// - `create`: Function to create the bundle if not cached
    ///
    /// # Returns
    /// An `Arc` to the cached bundle.
    ///
    /// # Thread Safety
    /// This method first attempts a read lock. If the bundle is not found,
    /// it upgrades to a write lock to insert the new bundle.
    pub fn get_or_create<F>(&self, key: BundleKey, create: F) -> Arc<wgpu::RenderBundle>
    where
        F: FnOnce() -> wgpu::RenderBundle,
    {
        // Fast path: try read lock first
        {
            let bundles = self.bundles.read().unwrap();
            if let Some(bundle) = bundles.get(&key) {
                return Arc::clone(bundle);
            }
        }

        // Slow path: acquire write lock and insert
        let mut bundles = self.bundles.write().unwrap();

        // Double-check after acquiring write lock (another thread may have inserted)
        if let Some(bundle) = bundles.get(&key) {
            return Arc::clone(bundle);
        }

        // Create and insert
        let bundle = Arc::new(create());
        bundles.insert(key, Arc::clone(&bundle));
        bundle
    }

    /// Get a cached bundle if it exists.
    ///
    /// Returns `None` if the bundle is not in the cache.
    pub fn get(&self, key: &BundleKey) -> Option<Arc<wgpu::RenderBundle>> {
        let bundles = self.bundles.read().unwrap();
        bundles.get(key).map(Arc::clone)
    }

    /// Check if a bundle is cached.
    pub fn contains(&self, key: &BundleKey) -> bool {
        let bundles = self.bundles.read().unwrap();
        bundles.contains_key(key)
    }

    /// Insert a bundle into the cache.
    ///
    /// Overwrites any existing bundle with the same key.
    pub fn insert(&self, key: BundleKey, bundle: wgpu::RenderBundle) {
        let mut bundles = self.bundles.write().unwrap();
        bundles.insert(key, Arc::new(bundle));
    }

    /// Invalidate (remove) a specific bundle from the cache.
    ///
    /// Returns `true` if the bundle was present and removed.
    pub fn invalidate(&self, key: &BundleKey) -> bool {
        let mut bundles = self.bundles.write().unwrap();
        bundles.remove(key).is_some()
    }

    /// Invalidate all bundles matching a predicate.
    ///
    /// Returns the number of bundles removed.
    pub fn invalidate_matching<F>(&self, predicate: F) -> usize
    where
        F: Fn(&BundleKey) -> bool,
    {
        let mut bundles = self.bundles.write().unwrap();
        let keys_to_remove: Vec<_> = bundles
            .keys()
            .filter(|k| predicate(k))
            .cloned()
            .collect();

        let count = keys_to_remove.len();
        for key in keys_to_remove {
            bundles.remove(&key);
        }
        count
    }

    /// Invalidate all cached bundles.
    ///
    /// Call this when render target formats change or on device lost.
    pub fn invalidate_all(&self) {
        let mut bundles = self.bundles.write().unwrap();
        bundles.clear();
    }

    /// Get the number of cached bundles.
    pub fn len(&self) -> usize {
        let bundles = self.bundles.read().unwrap();
        bundles.len()
    }

    /// Check if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get cache statistics.
    pub fn stats(&self) -> CacheStats {
        let bundles = self.bundles.read().unwrap();
        CacheStats {
            bundle_count: bundles.len(),
            // Note: Can't easily get bundle memory size from wgpu
        }
    }
}

impl fmt::Debug for RenderBundleCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let len = self.len();
        f.debug_struct("RenderBundleCache")
            .field("bundle_count", &len)
            .finish()
    }
}

/// Statistics about the render bundle cache.
#[derive(Debug, Clone, Copy, Default)]
pub struct CacheStats {
    /// Number of bundles currently cached.
    pub bundle_count: usize,
}

// ============================================================================
// Execute Bundles
// ============================================================================

/// Execute render bundles in a render pass.
///
/// This is a helper function that calls `render_pass.execute_bundles()`.
/// The bundles are executed in order.
///
/// # Requirements
///
/// - All bundles must have been created with formats matching the render pass
/// - The render pass must not have any active pipeline, bind groups, or buffers
///   (they will be reset after execution)
///
/// # Example
///
/// ```ignore
/// use renderer_backend::render_pipeline::render_bundle::execute_bundles;
///
/// // Execute multiple bundles
/// execute_bundles(&mut render_pass, &[&static_geometry, &ui_batch]);
/// ```
pub fn execute_bundles<'a>(
    render_pass: &mut wgpu::RenderPass<'a>,
    bundles: &[&'a wgpu::RenderBundle],
) {
    render_pass.execute_bundles(bundles.iter().copied());
}

/// Execute bundles from Arc references.
///
/// Convenience function for executing bundles stored in `Arc`.
pub fn execute_bundles_arc<'a>(
    render_pass: &mut wgpu::RenderPass<'a>,
    bundles: impl IntoIterator<Item = &'a Arc<wgpu::RenderBundle>>,
) {
    let refs: Vec<&wgpu::RenderBundle> = bundles.into_iter().map(|arc| arc.as_ref()).collect();
    render_pass.execute_bundles(refs.into_iter());
}

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during render bundle operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RenderBundleError {
    /// Sample count is not a valid power of 2 (1, 2, 4, 8, 16).
    InvalidSampleCount(u32),

    /// Too many color attachments (max 8).
    TooManyColorAttachments(usize),

    /// Bundle format mismatch with render pass.
    FormatMismatch {
        expected: String,
        actual: String,
    },
}

impl fmt::Display for RenderBundleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSampleCount(count) => {
                write!(f, "Invalid sample count {}, must be 1, 2, 4, 8, or 16", count)
            }
            Self::TooManyColorAttachments(count) => {
                write!(f, "Too many color attachments ({}), max is 8", count)
            }
            Self::FormatMismatch { expected, actual } => {
                write!(f, "Bundle format mismatch: expected {}, got {}", expected, actual)
            }
        }
    }
}

impl std::error::Error for RenderBundleError {}

// ============================================================================
// Preset Descriptors
// ============================================================================

/// Create a descriptor for a simple color-only bundle.
pub fn simple_color(format: wgpu::TextureFormat) -> RenderBundleEncoderDescriptor {
    RenderBundleEncoderDescriptor::new()
        .color_format(format)
        .sample_count(1)
}

/// Create a descriptor for color + depth bundle.
pub fn color_depth(
    color_format: wgpu::TextureFormat,
    depth_format: wgpu::TextureFormat,
) -> RenderBundleEncoderDescriptor {
    RenderBundleEncoderDescriptor::new()
        .color_format(color_format)
        .depth_stencil(depth_format, false, true)
        .sample_count(1)
}

/// Create a descriptor for depth-only bundle (shadow maps).
pub fn depth_only(depth_format: wgpu::TextureFormat) -> RenderBundleEncoderDescriptor {
    RenderBundleEncoderDescriptor::new()
        .depth_stencil(depth_format, false, true)
        .sample_count(1)
}

/// Create a descriptor for MSAA color + depth bundle.
pub fn msaa_color_depth(
    color_format: wgpu::TextureFormat,
    depth_format: wgpu::TextureFormat,
    sample_count: u32,
) -> RenderBundleEncoderDescriptor {
    RenderBundleEncoderDescriptor::new()
        .color_format(color_format)
        .depth_stencil(depth_format, false, true)
        .sample_count(sample_count)
}

/// Create a descriptor for G-buffer rendering (multiple color targets).
pub fn gbuffer(
    albedo_format: wgpu::TextureFormat,
    normal_format: wgpu::TextureFormat,
    material_format: wgpu::TextureFormat,
    depth_format: wgpu::TextureFormat,
) -> RenderBundleEncoderDescriptor {
    RenderBundleEncoderDescriptor::new()
        .color_format(albedo_format)
        .color_format(normal_format)
        .color_format(material_format)
        .depth_stencil(depth_format, false, true)
        .sample_count(1)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // RenderBundleEncoderDescriptor Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_descriptor_default() {
        let desc = RenderBundleEncoderDescriptor::default();
        assert!(desc.label.is_none());
        assert!(desc.color_formats.is_empty());
        assert!(desc.depth_stencil.is_none());
        assert_eq!(desc.sample_count, 1);
        assert!(desc.multiview.is_none());
    }

    #[test]
    fn test_descriptor_builder() {
        let desc = RenderBundleEncoderDescriptor::new()
            .label("test_bundle")
            .color_format(wgpu::TextureFormat::Bgra8UnormSrgb)
            .color_format(wgpu::TextureFormat::Rgba8Unorm)
            .depth_stencil(wgpu::TextureFormat::Depth32Float, false, true)
            .sample_count(4);

        assert_eq!(desc.label, Some("test_bundle".to_string()));
        assert_eq!(desc.color_formats.len(), 2);
        assert_eq!(desc.color_formats[0], Some(wgpu::TextureFormat::Bgra8UnormSrgb));
        assert_eq!(desc.color_formats[1], Some(wgpu::TextureFormat::Rgba8Unorm));
        assert!(desc.depth_stencil.is_some());
        let ds = desc.depth_stencil.unwrap();
        assert_eq!(ds.format, wgpu::TextureFormat::Depth32Float);
        assert!(!ds.depth_read_only);
        assert!(ds.stencil_read_only);
        assert_eq!(desc.sample_count, 4);
    }

    #[test]
    fn test_descriptor_validation() {
        // Valid descriptors
        assert!(RenderBundleEncoderDescriptor::new().sample_count(1).validate().is_ok());
        assert!(RenderBundleEncoderDescriptor::new().sample_count(4).validate().is_ok());
        assert!(RenderBundleEncoderDescriptor::new().sample_count(16).validate().is_ok());

        // Invalid sample count
        let mut desc = RenderBundleEncoderDescriptor::new();
        desc.sample_count = 3;
        assert!(matches!(
            desc.validate(),
            Err(RenderBundleError::InvalidSampleCount(3))
        ));

        // Too many color attachments
        let mut desc = RenderBundleEncoderDescriptor::new();
        desc.color_formats = vec![Some(wgpu::TextureFormat::Rgba8Unorm); 9];
        assert!(matches!(
            desc.validate(),
            Err(RenderBundleError::TooManyColorAttachments(9))
        ));
    }

    #[test]
    fn test_descriptor_display() {
        let desc = RenderBundleEncoderDescriptor::new()
            .label("test")
            .color_format(wgpu::TextureFormat::Bgra8UnormSrgb)
            .depth_stencil(wgpu::TextureFormat::Depth32Float, false, false)
            .sample_count(4);

        let display = format!("{}", desc);
        assert!(display.contains("test"));
        assert!(display.contains("colors: 1"));
        assert!(display.contains("depth: true"));
        assert!(display.contains("samples: 4"));
    }

    #[test]
    fn test_descriptor_multiview() {
        let desc = RenderBundleEncoderDescriptor::new()
            .multiview(NonZeroU32::new(2).unwrap());

        assert_eq!(desc.multiview, Some(NonZeroU32::new(2).unwrap()));

        let desc = desc.no_multiview();
        assert!(desc.multiview.is_none());
    }

    // -------------------------------------------------------------------------
    // BundleKey Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_bundle_key_simple() {
        let key1 = BundleKey::from_u64(42);
        let key2 = BundleKey::from_u64(42);
        let key3 = BundleKey::from_u64(43);

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_bundle_key_compound() {
        let key1 = BundleKey::from_parts(&[1, 2, 3]);
        let key2 = BundleKey::from_parts(&[1, 2, 3]);
        let key3 = BundleKey::from_parts(&[1, 2, 4]);

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_bundle_key_named() {
        let key1 = BundleKey::from_name("static_mesh");
        let key2 = BundleKey::from_name("static_mesh");
        let key3 = BundleKey::from_name("dynamic_mesh");

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_bundle_key_hash() {
        let key1 = BundleKey::from_hash(&("mesh", 42u64));
        let key2 = BundleKey::from_hash(&("mesh", 42u64));
        let key3 = BundleKey::from_hash(&("mesh", 43u64));

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_bundle_key_from_trait() {
        let key: BundleKey = 42u64.into();
        assert_eq!(key, BundleKey::from_u64(42));

        let key: BundleKey = "name".into();
        assert_eq!(key, BundleKey::from_name("name"));

        let key: BundleKey = String::from("name").into();
        assert_eq!(key, BundleKey::from_name("name"));
    }

    #[test]
    fn test_bundle_key_display() {
        assert_eq!(format!("{}", BundleKey::from_u64(42)), "BundleKey(42)");
        assert!(format!("{}", BundleKey::from_parts(&[1, 2])).contains("[1, 2]"));
        assert!(format!("{}", BundleKey::from_name("test")).contains("\"test\""));
    }

    #[test]
    fn test_bundle_key_hashable() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(BundleKey::from_u64(1));
        set.insert(BundleKey::from_u64(2));
        set.insert(BundleKey::from_u64(1)); // Duplicate

        assert_eq!(set.len(), 2);
    }

    // -------------------------------------------------------------------------
    // RenderBundleCache Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cache_new() {
        let cache = RenderBundleCache::new();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_cache_with_capacity() {
        let cache = RenderBundleCache::with_capacity(100);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_stats() {
        let cache = RenderBundleCache::new();
        let stats = cache.stats();
        assert_eq!(stats.bundle_count, 0);
    }

    #[test]
    fn test_cache_debug() {
        let cache = RenderBundleCache::new();
        let debug = format!("{:?}", cache);
        assert!(debug.contains("RenderBundleCache"));
        assert!(debug.contains("bundle_count"));
    }

    // -------------------------------------------------------------------------
    // Preset Descriptor Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_preset_simple_color() {
        let desc = simple_color(wgpu::TextureFormat::Bgra8UnormSrgb);
        assert_eq!(desc.color_formats.len(), 1);
        assert!(desc.depth_stencil.is_none());
        assert_eq!(desc.sample_count, 1);
    }

    #[test]
    fn test_preset_color_depth() {
        let desc = color_depth(
            wgpu::TextureFormat::Bgra8UnormSrgb,
            wgpu::TextureFormat::Depth32Float,
        );
        assert_eq!(desc.color_formats.len(), 1);
        assert!(desc.depth_stencil.is_some());
        assert_eq!(desc.sample_count, 1);
    }

    #[test]
    fn test_preset_depth_only() {
        let desc = depth_only(wgpu::TextureFormat::Depth32Float);
        assert!(desc.color_formats.is_empty());
        assert!(desc.depth_stencil.is_some());
        assert_eq!(desc.sample_count, 1);
    }

    #[test]
    fn test_preset_msaa() {
        let desc = msaa_color_depth(
            wgpu::TextureFormat::Bgra8UnormSrgb,
            wgpu::TextureFormat::Depth32Float,
            4,
        );
        assert_eq!(desc.sample_count, 4);
    }

    #[test]
    fn test_preset_gbuffer() {
        let desc = gbuffer(
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureFormat::Depth32Float,
        );
        assert_eq!(desc.color_formats.len(), 3);
        assert!(desc.depth_stencil.is_some());
    }

    // -------------------------------------------------------------------------
    // Error Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_error_display() {
        let err = RenderBundleError::InvalidSampleCount(3);
        assert!(format!("{}", err).contains("3"));

        let err = RenderBundleError::TooManyColorAttachments(9);
        assert!(format!("{}", err).contains("9"));

        let err = RenderBundleError::FormatMismatch {
            expected: "Rgba8".to_string(),
            actual: "Bgra8".to_string(),
        };
        assert!(format!("{}", err).contains("Rgba8"));
        assert!(format!("{}", err).contains("Bgra8"));
    }

    #[test]
    fn test_error_is_error() {
        fn assert_error<E: std::error::Error>() {}
        assert_error::<RenderBundleError>();
    }

    // -------------------------------------------------------------------------
    // RenderBundleEncoderDescriptor Builder Tests (Full Coverage)
    // -------------------------------------------------------------------------

    #[test]
    fn test_descriptor_new_returns_default() {
        let new_desc = RenderBundleEncoderDescriptor::new();
        let default_desc = RenderBundleEncoderDescriptor::default();

        assert_eq!(new_desc.label, default_desc.label);
        assert_eq!(new_desc.color_formats, default_desc.color_formats);
        assert_eq!(new_desc.depth_stencil, default_desc.depth_stencil);
        assert_eq!(new_desc.sample_count, default_desc.sample_count);
        assert_eq!(new_desc.multiview, default_desc.multiview);
    }

    #[test]
    fn test_descriptor_label_string_types() {
        // &str
        let desc = RenderBundleEncoderDescriptor::new().label("static");
        assert_eq!(desc.label, Some("static".to_string()));

        // String
        let desc = RenderBundleEncoderDescriptor::new().label(String::from("dynamic"));
        assert_eq!(desc.label, Some("dynamic".to_string()));

        // Cow<str> (via Into<String>)
        let label = std::borrow::Cow::Borrowed("cow_label");
        let desc = RenderBundleEncoderDescriptor::new().label(label.into_owned());
        assert_eq!(desc.label, Some("cow_label".to_string()));
    }

    #[test]
    fn test_descriptor_color_format_opt() {
        let desc = RenderBundleEncoderDescriptor::new()
            .color_format(wgpu::TextureFormat::Rgba8Unorm)
            .color_format_opt(None) // Sparse slot
            .color_format(wgpu::TextureFormat::Bgra8Unorm);

        assert_eq!(desc.color_formats.len(), 3);
        assert_eq!(desc.color_formats[0], Some(wgpu::TextureFormat::Rgba8Unorm));
        assert_eq!(desc.color_formats[1], None); // Sparse slot
        assert_eq!(desc.color_formats[2], Some(wgpu::TextureFormat::Bgra8Unorm));
    }

    #[test]
    fn test_descriptor_color_formats_bulk() {
        let formats = vec![
            Some(wgpu::TextureFormat::Rgba8Unorm),
            Some(wgpu::TextureFormat::Rgba16Float),
            None,
            Some(wgpu::TextureFormat::Rgba32Float),
        ];

        let desc = RenderBundleEncoderDescriptor::new().color_formats(formats.clone());
        assert_eq!(desc.color_formats, formats);
    }

    #[test]
    fn test_descriptor_depth_stencil_raw() {
        let ds = wgpu::RenderBundleDepthStencil {
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            depth_read_only: true,
            stencil_read_only: false,
        };

        let desc = RenderBundleEncoderDescriptor::new().depth_stencil_raw(ds);

        assert!(desc.depth_stencil.is_some());
        let result = desc.depth_stencil.unwrap();
        assert_eq!(result.format, wgpu::TextureFormat::Depth24PlusStencil8);
        assert!(result.depth_read_only);
        assert!(!result.stencil_read_only);
    }

    #[test]
    fn test_descriptor_no_depth_stencil() {
        let desc = RenderBundleEncoderDescriptor::new()
            .depth_stencil(wgpu::TextureFormat::Depth32Float, false, false)
            .no_depth_stencil();

        assert!(desc.depth_stencil.is_none());
    }

    #[test]
    fn test_descriptor_sample_count_valid_values() {
        for count in [1, 2, 4, 8, 16] {
            let desc = RenderBundleEncoderDescriptor::new().sample_count(count);
            assert_eq!(desc.sample_count, count);
            assert!(desc.validate().is_ok());
        }
    }

    #[test]
    fn test_descriptor_builder_chaining_full() {
        let desc = RenderBundleEncoderDescriptor::new()
            .label("full_chain")
            .color_format(wgpu::TextureFormat::Bgra8UnormSrgb)
            .color_format(wgpu::TextureFormat::Rgba8Unorm)
            .color_format_opt(Some(wgpu::TextureFormat::Rgba16Float))
            .depth_stencil(wgpu::TextureFormat::Depth32Float, false, true)
            .sample_count(4)
            .multiview(NonZeroU32::new(2).unwrap());

        assert_eq!(desc.label, Some("full_chain".to_string()));
        assert_eq!(desc.color_formats.len(), 3);
        assert!(desc.depth_stencil.is_some());
        assert_eq!(desc.sample_count, 4);
        assert_eq!(desc.multiview, Some(NonZeroU32::new(2).unwrap()));
    }

    #[test]
    fn test_descriptor_builder_overwrite_fields() {
        // Test that later calls overwrite earlier values
        let desc = RenderBundleEncoderDescriptor::new()
            .label("first")
            .label("second")
            .sample_count(1)
            .sample_count(4)
            .sample_count(8);

        assert_eq!(desc.label, Some("second".to_string()));
        assert_eq!(desc.sample_count, 8);
    }

    #[test]
    fn test_descriptor_to_wgpu() {
        let desc = RenderBundleEncoderDescriptor::new()
            .label("wgpu_test")
            .color_format(wgpu::TextureFormat::Bgra8UnormSrgb)
            .depth_stencil(wgpu::TextureFormat::Depth32Float, false, true)
            .sample_count(4);

        let wgpu_desc = desc.to_wgpu();

        assert_eq!(wgpu_desc.label, Some("wgpu_test"));
        assert_eq!(wgpu_desc.color_formats.len(), 1);
        assert_eq!(wgpu_desc.sample_count, 4);
        assert_eq!(wgpu_desc.depth_stencil, desc.depth_stencil);
        assert_eq!(wgpu_desc.multiview, None);
    }

    #[test]
    fn test_descriptor_to_wgpu_no_label() {
        let desc = RenderBundleEncoderDescriptor::new();
        let wgpu_desc = desc.to_wgpu();
        assert!(wgpu_desc.label.is_none());
    }

    #[test]
    fn test_descriptor_validate_edge_cases() {
        // Exactly 8 color attachments (max valid)
        let mut desc = RenderBundleEncoderDescriptor::new();
        desc.color_formats = vec![Some(wgpu::TextureFormat::Rgba8Unorm); 8];
        assert!(desc.validate().is_ok());

        // 0 color attachments (valid for depth-only)
        let desc = RenderBundleEncoderDescriptor::new()
            .depth_stencil(wgpu::TextureFormat::Depth32Float, false, false);
        assert!(desc.validate().is_ok());
    }

    #[test]
    fn test_descriptor_clone() {
        let desc = RenderBundleEncoderDescriptor::new()
            .label("clone_test")
            .color_format(wgpu::TextureFormat::Rgba8Unorm)
            .sample_count(4);

        let cloned = desc.clone();

        assert_eq!(desc.label, cloned.label);
        assert_eq!(desc.color_formats, cloned.color_formats);
        assert_eq!(desc.sample_count, cloned.sample_count);
    }

    #[test]
    fn test_descriptor_partial_eq() {
        let desc1 = RenderBundleEncoderDescriptor::new()
            .label("test")
            .sample_count(4);
        let desc2 = RenderBundleEncoderDescriptor::new()
            .label("test")
            .sample_count(4);
        let desc3 = RenderBundleEncoderDescriptor::new()
            .label("different")
            .sample_count(4);

        assert_eq!(desc1, desc2);
        assert_ne!(desc1, desc3);
    }

    #[test]
    fn test_descriptor_debug() {
        let desc = RenderBundleEncoderDescriptor::new()
            .label("debug_test")
            .color_format(wgpu::TextureFormat::Rgba8Unorm);

        let debug = format!("{:?}", desc);
        assert!(debug.contains("RenderBundleEncoderDescriptor"));
        assert!(debug.contains("debug_test"));
    }

    // -------------------------------------------------------------------------
    // BundleKey Comprehensive Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_bundle_key_simple_variant() {
        let key = BundleKey::from_u64(0);
        assert_eq!(format!("{}", key), "BundleKey(0)");

        let key = BundleKey::from_u64(u64::MAX);
        assert_eq!(format!("{}", key), format!("BundleKey({})", u64::MAX));
    }

    #[test]
    fn test_bundle_key_compound_single_element() {
        // Single element compound becomes Simple
        let key = BundleKey::from_parts(&[42]);
        assert_eq!(key, BundleKey::from_u64(42));
    }

    #[test]
    fn test_bundle_key_compound_empty() {
        // Empty parts should create Compound with empty vec
        let key = BundleKey::from_parts(&[]);
        // Verify it's hashable and displayable
        let _ = format!("{}", key);

        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
    }

    #[test]
    fn test_bundle_key_compound_many_parts() {
        let parts: Vec<u64> = (0..100).collect();
        let key = BundleKey::from_parts(&parts);

        let display = format!("{}", key);
        assert!(display.contains("[0, 1, 2"));
    }

    #[test]
    fn test_bundle_key_named_empty_string() {
        let key = BundleKey::from_name("");
        assert_eq!(format!("{}", key), "BundleKey(\"\")");
    }

    #[test]
    fn test_bundle_key_named_unicode() {
        let key = BundleKey::from_name("mesh_");
        let display = format!("{}", key);
        assert!(display.contains("mesh_"));
    }

    #[test]
    fn test_bundle_key_hash_consistency() {
        // Same input should produce same key
        let key1 = BundleKey::from_hash(&("mesh", 42u64, "material"));
        let key2 = BundleKey::from_hash(&("mesh", 42u64, "material"));
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_bundle_key_different_types_not_equal() {
        // Different key types should not be equal
        let simple = BundleKey::from_u64(42);
        let named = BundleKey::from_name("42");
        let compound = BundleKey::from_parts(&[42, 0]);

        assert_ne!(simple, named);
        assert_ne!(simple, compound);
        assert_ne!(named, compound);
    }

    #[test]
    fn test_bundle_key_clone() {
        let key = BundleKey::from_parts(&[1, 2, 3]);
        let cloned = key.clone();
        assert_eq!(key, cloned);
    }

    #[test]
    fn test_bundle_key_debug() {
        let key = BundleKey::from_name("debug_key");
        let debug = format!("{:?}", key);
        assert!(debug.contains("BundleKey"));
        assert!(debug.contains("Named"));
    }

    #[test]
    fn test_bundle_key_hash_in_hashmap() {
        use std::collections::HashMap;

        let mut map = HashMap::new();
        map.insert(BundleKey::from_u64(1), "value1");
        map.insert(BundleKey::from_name("key2"), "value2");
        map.insert(BundleKey::from_parts(&[1, 2]), "value3");

        assert_eq!(map.get(&BundleKey::from_u64(1)), Some(&"value1"));
        assert_eq!(map.get(&BundleKey::from_name("key2")), Some(&"value2"));
        assert_eq!(map.get(&BundleKey::from_parts(&[1, 2])), Some(&"value3"));
    }

    // -------------------------------------------------------------------------
    // RenderBundleCache get_or_create, get, contains, insert Tests
    // -------------------------------------------------------------------------

    // Note: Full cache tests with actual RenderBundles require wgpu device.
    // We test the synchronization and key mechanics without actual bundles.

    #[test]
    fn test_cache_contains_nonexistent() {
        let cache = RenderBundleCache::new();
        assert!(!cache.contains(&BundleKey::from_u64(999)));
    }

    #[test]
    fn test_cache_get_nonexistent() {
        let cache = RenderBundleCache::new();
        assert!(cache.get(&BundleKey::from_u64(999)).is_none());
    }

    #[test]
    fn test_cache_is_empty_after_new() {
        let cache = RenderBundleCache::new();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_cache_default_impl() {
        let cache = RenderBundleCache::default();
        assert!(cache.is_empty());
    }

    // -------------------------------------------------------------------------
    // Invalidation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cache_invalidate_nonexistent() {
        let cache = RenderBundleCache::new();
        let result = cache.invalidate(&BundleKey::from_u64(999));
        assert!(!result);
    }

    #[test]
    fn test_cache_invalidate_all_empty() {
        let cache = RenderBundleCache::new();
        cache.invalidate_all();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_invalidate_matching_no_matches() {
        let cache = RenderBundleCache::new();
        let count = cache.invalidate_matching(|_| false);
        assert_eq!(count, 0);
    }

    // -------------------------------------------------------------------------
    // Cache Statistics Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cache_stats_default() {
        let stats = CacheStats::default();
        assert_eq!(stats.bundle_count, 0);
    }

    #[test]
    fn test_cache_stats_debug() {
        let stats = CacheStats { bundle_count: 42 };
        let debug = format!("{:?}", stats);
        assert!(debug.contains("CacheStats"));
        assert!(debug.contains("42"));
    }

    #[test]
    fn test_cache_stats_clone() {
        let stats = CacheStats { bundle_count: 10 };
        let cloned = stats;
        assert_eq!(stats.bundle_count, cloned.bundle_count);
    }

    #[test]
    fn test_cache_stats_copy() {
        let stats = CacheStats { bundle_count: 5 };
        let copied: CacheStats = stats;
        assert_eq!(stats.bundle_count, copied.bundle_count);
    }

    // -------------------------------------------------------------------------
    // Thread Safety Tests (RwLock behavior)
    // -------------------------------------------------------------------------

    #[test]
    fn test_cache_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<RenderBundleCache>();
        assert_sync::<RenderBundleCache>();
    }

    #[test]
    fn test_cache_multiple_readers() {
        use std::thread;

        let cache = Arc::new(RenderBundleCache::new());

        let handles: Vec<_> = (0..10)
            .map(|i| {
                let cache_clone = Arc::clone(&cache);
                thread::spawn(move || {
                    // Multiple threads reading concurrently
                    let _ = cache_clone.contains(&BundleKey::from_u64(i));
                    let _ = cache_clone.get(&BundleKey::from_u64(i));
                    let _ = cache_clone.len();
                    let _ = cache_clone.is_empty();
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_cache_concurrent_invalidate_all() {
        use std::thread;

        let cache = Arc::new(RenderBundleCache::new());

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let cache_clone = Arc::clone(&cache);
                thread::spawn(move || {
                    cache_clone.invalidate_all();
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        assert!(cache.is_empty());
    }

    // -------------------------------------------------------------------------
    // Trait Coverage Tests (Debug, Clone, Hash, Eq, Default)
    // -------------------------------------------------------------------------

    #[test]
    fn test_traits_bundle_key_data_debug() {
        // BundleKeyData is private, but tested through BundleKey
        let key = BundleKey::from_parts(&[1, 2, 3]);
        let debug = format!("{:?}", key);
        assert!(debug.contains("Compound"));
    }

    #[test]
    fn test_traits_render_bundle_error_clone() {
        let err = RenderBundleError::InvalidSampleCount(3);
        let cloned = err.clone();
        assert_eq!(err, cloned);
    }

    #[test]
    fn test_traits_render_bundle_error_eq() {
        let err1 = RenderBundleError::TooManyColorAttachments(9);
        let err2 = RenderBundleError::TooManyColorAttachments(9);
        let err3 = RenderBundleError::TooManyColorAttachments(10);

        assert_eq!(err1, err2);
        assert_ne!(err1, err3);
    }

    #[test]
    fn test_traits_render_bundle_error_debug() {
        let err = RenderBundleError::FormatMismatch {
            expected: "A".to_string(),
            actual: "B".to_string(),
        };
        let debug = format!("{:?}", err);
        assert!(debug.contains("FormatMismatch"));
    }

    // -------------------------------------------------------------------------
    // Edge Cases Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_edge_case_empty_label() {
        let desc = RenderBundleEncoderDescriptor::new().label("");
        assert_eq!(desc.label, Some("".to_string()));
        let wgpu_desc = desc.to_wgpu();
        assert_eq!(wgpu_desc.label, Some(""));
    }

    #[test]
    fn test_edge_case_max_color_attachments() {
        let mut desc = RenderBundleEncoderDescriptor::new();
        for _ in 0..8 {
            desc = desc.color_format(wgpu::TextureFormat::Rgba8Unorm);
        }
        assert_eq!(desc.color_formats.len(), 8);
        assert!(desc.validate().is_ok());
    }

    #[test]
    fn test_edge_case_all_none_color_formats() {
        let desc = RenderBundleEncoderDescriptor::new()
            .color_format_opt(None)
            .color_format_opt(None)
            .color_format_opt(None);

        assert_eq!(desc.color_formats.len(), 3);
        assert!(desc.color_formats.iter().all(|f| f.is_none()));
    }

    #[test]
    fn test_edge_case_multiview_max_layers() {
        // Test with high layer count (GPU limits may vary)
        let layers = NonZeroU32::new(6).unwrap();
        let desc = RenderBundleEncoderDescriptor::new().multiview(layers);
        assert_eq!(desc.multiview, Some(layers));
    }

    #[test]
    fn test_edge_case_depth_stencil_read_only_combinations() {
        // Both read-only
        let desc = RenderBundleEncoderDescriptor::new()
            .depth_stencil(wgpu::TextureFormat::Depth24PlusStencil8, true, true);
        let ds = desc.depth_stencil.unwrap();
        assert!(ds.depth_read_only);
        assert!(ds.stencil_read_only);

        // Neither read-only
        let desc = RenderBundleEncoderDescriptor::new()
            .depth_stencil(wgpu::TextureFormat::Depth24PlusStencil8, false, false);
        let ds = desc.depth_stencil.unwrap();
        assert!(!ds.depth_read_only);
        assert!(!ds.stencil_read_only);
    }

    #[test]
    fn test_edge_case_bundle_key_from_parts_very_long() {
        let parts: Vec<u64> = (0..1000).collect();
        let key1 = BundleKey::from_parts(&parts);
        let key2 = BundleKey::from_parts(&parts);
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_edge_case_cache_stats_with_capacity() {
        let cache = RenderBundleCache::with_capacity(1000);
        let stats = cache.stats();
        assert_eq!(stats.bundle_count, 0);
    }

    // -------------------------------------------------------------------------
    // Memory Layout and Size Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_memory_size_bundle_key() {
        // BundleKey should be reasonably sized
        let size = std::mem::size_of::<BundleKey>();
        // Should be small enough to pass around by value efficiently
        assert!(size <= 48, "BundleKey is {} bytes, expected <= 48", size);
    }

    #[test]
    fn test_memory_size_descriptor() {
        let size = std::mem::size_of::<RenderBundleEncoderDescriptor>();
        // Descriptor contains Vecs so size is pointer-based
        assert!(size <= 128, "Descriptor is {} bytes, expected <= 128", size);
    }

    #[test]
    fn test_memory_size_cache_stats() {
        let size = std::mem::size_of::<CacheStats>();
        // CacheStats should be very small (just a usize)
        assert_eq!(size, std::mem::size_of::<usize>());
    }

    #[test]
    fn test_memory_size_render_bundle_error() {
        let size = std::mem::size_of::<RenderBundleError>();
        // Should be reasonably sized for stack allocation
        assert!(size <= 64, "RenderBundleError is {} bytes, expected <= 64", size);
    }

    #[test]
    fn test_memory_alignment_bundle_key() {
        let align = std::mem::align_of::<BundleKey>();
        // Should be word-aligned at minimum
        assert!(align >= std::mem::align_of::<usize>());
    }

    // -------------------------------------------------------------------------
    // Preset Descriptor Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_preset_depth_formats() {
        // Test various depth formats
        let formats = [
            wgpu::TextureFormat::Depth16Unorm,
            wgpu::TextureFormat::Depth32Float,
            wgpu::TextureFormat::Depth24Plus,
            wgpu::TextureFormat::Depth24PlusStencil8,
            wgpu::TextureFormat::Depth32FloatStencil8,
        ];

        for format in formats {
            let desc = depth_only(format);
            assert!(desc.depth_stencil.is_some());
            assert_eq!(desc.depth_stencil.unwrap().format, format);
        }
    }

    #[test]
    fn test_preset_color_formats() {
        let formats = [
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            wgpu::TextureFormat::Bgra8Unorm,
            wgpu::TextureFormat::Bgra8UnormSrgb,
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureFormat::Rgba32Float,
        ];

        for format in formats {
            let desc = simple_color(format);
            assert_eq!(desc.color_formats.len(), 1);
            assert_eq!(desc.color_formats[0], Some(format));
        }
    }

    // -------------------------------------------------------------------------
    // Error Variant Coverage
    // -------------------------------------------------------------------------

    #[test]
    fn test_error_variant_invalid_sample_count_values() {
        for count in [0, 3, 5, 6, 7, 9, 17, 32, 64, 100] {
            let err = RenderBundleError::InvalidSampleCount(count);
            let display = format!("{}", err);
            assert!(display.contains(&count.to_string()));
        }
    }

    #[test]
    fn test_error_variant_too_many_attachments_values() {
        for count in [9, 10, 16, 100] {
            let err = RenderBundleError::TooManyColorAttachments(count);
            let display = format!("{}", err);
            assert!(display.contains(&count.to_string()));
        }
    }

    #[test]
    fn test_error_variant_format_mismatch_long_strings() {
        let err = RenderBundleError::FormatMismatch {
            expected: "Rgba16Float".repeat(10),
            actual: "Bgra8UnormSrgb".repeat(10),
        };
        let display = format!("{}", err);
        assert!(display.contains("Rgba16Float"));
        assert!(display.contains("Bgra8UnormSrgb"));
    }

    // -------------------------------------------------------------------------
    // Display Trait Coverage
    // -------------------------------------------------------------------------

    #[test]
    fn test_display_descriptor_no_label() {
        let desc = RenderBundleEncoderDescriptor::new()
            .color_format(wgpu::TextureFormat::Rgba8Unorm);

        let display = format!("{}", desc);
        assert!(display.contains("None"));
        assert!(display.contains("colors: 1"));
    }

    #[test]
    fn test_display_descriptor_with_multiview() {
        let desc = RenderBundleEncoderDescriptor::new()
            .multiview(NonZeroU32::new(2).unwrap());

        let display = format!("{}", desc);
        assert!(display.contains("multiview: Some(2)"));
    }

    #[test]
    fn test_display_bundle_key_all_variants() {
        let simple_display = format!("{}", BundleKey::from_u64(123));
        assert!(simple_display.contains("123"));

        let compound_display = format!("{}", BundleKey::from_parts(&[1, 2, 3]));
        assert!(compound_display.contains("[1, 2, 3]"));

        let named_display = format!("{}", BundleKey::from_name("test_name"));
        assert!(named_display.contains("test_name"));
    }
}
