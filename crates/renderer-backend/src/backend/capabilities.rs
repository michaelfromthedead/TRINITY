//! Unified backend capabilities query system for TRINITY.
//!
//! This module provides a hardware-agnostic way to query GPU capabilities
//! across different graphics backends (Vulkan, Metal, DX12, WebGPU, OpenGL).
//! It abstracts over backend-specific feature detection to provide a unified
//! interface for capability queries.
//!
//! # Architecture
//!
//! ```text
//! UnifiedCapabilities
//!   - Hardware-agnostic feature queries
//!   - Backend-specific detection under the hood
//!   - Comparable across backends
//!
//! CapabilitiesQuery
//!   - Static methods for capability detection
//!   - Feature support checking
//!   - Backend comparison
//!
//! FeatureRequirement
//!   - Minimum/preferred feature levels
//!   - Satisfaction checking
//! ```
//!
//! # Example
//!
//! ```no_run
//! use renderer_backend::backend::capabilities::{
//!     CapabilitiesQuery, FeatureRequirement, FeatureLevel, UnifiedCapabilities,
//! };
//!
//! # async fn example() {
//! let instance = wgpu::Instance::default();
//! let adapter = instance
//!     .request_adapter(&wgpu::RequestAdapterOptions::default())
//!     .await
//!     .unwrap();
//!
//! let caps = CapabilitiesQuery::from_adapter(&adapter);
//!
//! // Check specific feature support
//! let rt_support = CapabilitiesQuery::supports_feature(&caps, "ray_tracing");
//! println!("Ray tracing: {:?}", rt_support);
//!
//! // Define requirements
//! let req = FeatureRequirement::new("ray_tracing", FeatureLevel::Hardware)
//!     .with_preferred(FeatureLevel::Optimal);
//!
//! if req.is_satisfied(rt_support) {
//!     println!("Ray tracing requirements met!");
//! }
//! # }
//! ```

use std::cmp::Ordering;
use wgpu::Adapter;

use super::{
    BackendType, D3D12Features, MetalFeatures, VulkanFeatures, WebGpuFeatures, WebGpuTier,
};

// ============================================================================
// FeatureLevel
// ============================================================================

/// Hardware feature support level.
///
/// Represents the level of support for a specific GPU feature, ranging from
/// unavailable to optimal hardware support. This enum is ordered from least
/// to most capable.
///
/// # Ordering
///
/// `Unavailable < Emulated < Hardware < Optimal`
///
/// This ordering allows comparison of feature support levels:
///
/// ```
/// use renderer_backend::backend::capabilities::FeatureLevel;
///
/// assert!(FeatureLevel::Hardware > FeatureLevel::Emulated);
/// assert!(FeatureLevel::Optimal > FeatureLevel::Hardware);
/// assert!(FeatureLevel::Unavailable < FeatureLevel::Emulated);
/// ```
///
/// # Example
///
/// ```
/// use renderer_backend::backend::capabilities::FeatureLevel;
///
/// let ray_tracing_support = FeatureLevel::Hardware;
///
/// match ray_tracing_support {
///     FeatureLevel::Unavailable => println!("No ray tracing"),
///     FeatureLevel::Emulated => println!("Software ray tracing (slow)"),
///     FeatureLevel::Hardware => println!("Native hardware ray tracing"),
///     FeatureLevel::Optimal => println!("Best-in-class ray tracing"),
/// }
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub enum FeatureLevel {
    /// Feature is not available.
    ///
    /// The hardware/driver does not support this feature at all.
    #[default]
    Unavailable,

    /// Feature is emulated in software.
    ///
    /// The feature works but may have significant performance overhead
    /// due to software emulation or fallback paths.
    Emulated,

    /// Feature has native hardware support.
    ///
    /// The feature is supported by the GPU hardware with good performance.
    Hardware,

    /// Feature has optimal/best-in-class support.
    ///
    /// The feature is supported with the highest quality and performance,
    /// typically representing the latest hardware generation.
    Optimal,
}

impl FeatureLevel {
    /// Check if the feature is at least minimally available.
    ///
    /// Returns `true` if the feature is at least emulated (not unavailable).
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::capabilities::FeatureLevel;
    ///
    /// assert!(!FeatureLevel::Unavailable.is_available());
    /// assert!(FeatureLevel::Emulated.is_available());
    /// assert!(FeatureLevel::Hardware.is_available());
    /// ```
    #[inline]
    pub const fn is_available(&self) -> bool {
        !matches!(self, FeatureLevel::Unavailable)
    }

    /// Check if the feature has native hardware support.
    ///
    /// Returns `true` if the feature is at least hardware-level (not emulated).
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::capabilities::FeatureLevel;
    ///
    /// assert!(!FeatureLevel::Emulated.is_hardware());
    /// assert!(FeatureLevel::Hardware.is_hardware());
    /// assert!(FeatureLevel::Optimal.is_hardware());
    /// ```
    #[inline]
    pub const fn is_hardware(&self) -> bool {
        matches!(self, FeatureLevel::Hardware | FeatureLevel::Optimal)
    }

    /// Check if the feature has optimal support.
    ///
    /// Returns `true` only for optimal-level support.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::capabilities::FeatureLevel;
    ///
    /// assert!(!FeatureLevel::Hardware.is_optimal());
    /// assert!(FeatureLevel::Optimal.is_optimal());
    /// ```
    #[inline]
    pub const fn is_optimal(&self) -> bool {
        matches!(self, FeatureLevel::Optimal)
    }

    /// Get a human-readable name for this feature level.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::capabilities::FeatureLevel;
    ///
    /// assert_eq!(FeatureLevel::Hardware.name(), "Hardware");
    /// ```
    #[inline]
    pub const fn name(&self) -> &'static str {
        match self {
            FeatureLevel::Unavailable => "Unavailable",
            FeatureLevel::Emulated => "Emulated",
            FeatureLevel::Hardware => "Hardware",
            FeatureLevel::Optimal => "Optimal",
        }
    }
}

impl std::fmt::Display for FeatureLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ============================================================================
// UnifiedCapabilities
// ============================================================================

/// Unified GPU capabilities across all backends.
///
/// This struct provides a hardware-agnostic view of GPU capabilities,
/// abstracting over the differences between Vulkan, Metal, DX12, and WebGPU.
/// It queries backend-specific features and normalizes them into a common
/// representation.
///
/// # Feature Support Levels
///
/// Each feature is represented as a [`FeatureLevel`], indicating whether
/// the feature is unavailable, emulated, has hardware support, or has
/// optimal support.
///
/// # Limits
///
/// In addition to feature support, this struct includes common GPU limits
/// that are useful for resource allocation and rendering decisions.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::backend::capabilities::{CapabilitiesQuery, FeatureLevel};
///
/// # async fn example() {
/// let instance = wgpu::Instance::default();
/// let adapter = instance
///     .request_adapter(&wgpu::RequestAdapterOptions::default())
///     .await
///     .unwrap();
///
/// let caps = CapabilitiesQuery::from_adapter(&adapter);
///
/// if caps.ray_tracing.is_hardware() {
///     println!("Hardware ray tracing available!");
///     println!("  Support level: {:?}", caps.ray_tracing);
/// }
///
/// println!("Max texture size: {}", caps.max_texture_size);
/// println!("Max buffer size: {} bytes", caps.max_buffer_size);
/// # }
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnifiedCapabilities {
    /// The detected backend type.
    pub backend: BackendType,

    /// Ray tracing support level.
    ///
    /// - `Unavailable`: No ray tracing support
    /// - `Emulated`: Software ray tracing (rare)
    /// - `Hardware`: Native RT cores (RTX, RDNA2+, M3+)
    /// - `Optimal`: Latest RT hardware (RTX 40+, RX 7000+)
    pub ray_tracing: FeatureLevel,

    /// Mesh shader support level.
    ///
    /// - `Unavailable`: No mesh shader support
    /// - `Emulated`: Geometry shader fallback (very slow)
    /// - `Hardware`: Native mesh shaders (Turing+, RDNA2+, M3+)
    /// - `Optimal`: Latest mesh shader hardware
    pub mesh_shaders: FeatureLevel,

    /// Variable rate shading support level.
    ///
    /// - `Unavailable`: No VRS support
    /// - `Hardware`: Tier 1 VRS (per-draw)
    /// - `Optimal`: Tier 2 VRS (per-primitive, image-based)
    pub variable_rate_shading: FeatureLevel,

    /// Bindless/descriptor indexing support level.
    ///
    /// - `Unavailable`: Traditional binding model only
    /// - `Hardware`: Basic bindless (limited descriptors)
    /// - `Optimal`: Full bindless (millions of descriptors)
    pub bindless: FeatureLevel,

    /// Async compute support level.
    ///
    /// - `Unavailable`: Single queue only
    /// - `Hardware`: Separate compute queue
    /// - `Optimal`: Multiple async compute queues
    pub async_compute: FeatureLevel,

    /// Conservative rasterization support level.
    ///
    /// - `Unavailable`: No conservative rasterization
    /// - `Hardware`: Tier 1 (overestimation only)
    /// - `Optimal`: Tier 3 (under/overestimation, uncertainty)
    pub conservative_rasterization: FeatureLevel,

    /// Sampler feedback support level.
    ///
    /// - `Unavailable`: No sampler feedback
    /// - `Hardware`: Basic feedback (texture streaming)
    /// - `Optimal`: Full feedback with min-mip
    pub sampler_feedback: FeatureLevel,

    /// Maximum texture dimension (width/height) in texels.
    pub max_texture_size: u32,

    /// Maximum buffer size in bytes.
    pub max_buffer_size: u64,

    /// Maximum compute workgroup size [x, y, z].
    pub max_compute_workgroup_size: [u32; 3],
}

impl Default for UnifiedCapabilities {
    fn default() -> Self {
        Self {
            backend: BackendType::Unknown,
            ray_tracing: FeatureLevel::Unavailable,
            mesh_shaders: FeatureLevel::Unavailable,
            variable_rate_shading: FeatureLevel::Unavailable,
            bindless: FeatureLevel::Unavailable,
            async_compute: FeatureLevel::Unavailable,
            conservative_rasterization: FeatureLevel::Unavailable,
            sampler_feedback: FeatureLevel::Unavailable,
            max_texture_size: 2048,
            max_buffer_size: 256 * 1024 * 1024, // 256 MB default
            max_compute_workgroup_size: [128, 128, 64],
        }
    }
}

impl UnifiedCapabilities {
    /// Check if full ray tracing pipeline is supported.
    ///
    /// Returns `true` if ray tracing has at least hardware-level support.
    #[inline]
    pub fn supports_full_rt(&self) -> bool {
        self.ray_tracing.is_hardware()
    }

    /// Check if GPU-driven rendering is well supported.
    ///
    /// GPU-driven rendering requires bindless resources and async compute.
    #[inline]
    pub fn supports_gpu_driven(&self) -> bool {
        self.bindless.is_hardware() && self.async_compute.is_available()
    }

    /// Check if this backend supports advanced features.
    ///
    /// Advanced features include ray tracing, mesh shaders, and VRS.
    #[inline]
    pub fn supports_advanced_features(&self) -> bool {
        self.ray_tracing.is_hardware()
            || self.mesh_shaders.is_hardware()
            || self.variable_rate_shading.is_hardware()
    }

    /// Get the total "feature score" for comparison purposes.
    ///
    /// Higher scores indicate more capable hardware. This is useful
    /// for sorting adapters by capability.
    pub fn feature_score(&self) -> u32 {
        let mut score = 0u32;

        // Weight features by importance
        score += self.ray_tracing as u32 * 10;
        score += self.mesh_shaders as u32 * 8;
        score += self.bindless as u32 * 7;
        score += self.async_compute as u32 * 6;
        score += self.variable_rate_shading as u32 * 5;
        score += self.conservative_rasterization as u32 * 3;
        score += self.sampler_feedback as u32 * 2;

        // Add limit-based scoring
        if self.max_texture_size >= 16384 {
            score += 5;
        } else if self.max_texture_size >= 8192 {
            score += 3;
        } else if self.max_texture_size >= 4096 {
            score += 1;
        }

        if self.max_buffer_size >= 4 * 1024 * 1024 * 1024 {
            score += 5;
        } else if self.max_buffer_size >= 1024 * 1024 * 1024 {
            score += 3;
        }

        score
    }
}

// ============================================================================
// CapabilitiesQuery
// ============================================================================

/// Static methods for querying and comparing GPU capabilities.
///
/// This struct provides utility methods for detecting capabilities from
/// adapters, querying specific features, and comparing backends.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::backend::capabilities::{CapabilitiesQuery, FeatureLevel};
///
/// # async fn example() {
/// let instance = wgpu::Instance::default();
/// let adapter = instance
///     .request_adapter(&wgpu::RequestAdapterOptions::default())
///     .await
///     .unwrap();
///
/// // Detect all capabilities
/// let caps = CapabilitiesQuery::from_adapter(&adapter);
///
/// // Query specific feature
/// let rt = CapabilitiesQuery::supports_feature(&caps, "ray_tracing");
/// println!("Ray tracing support: {:?}", rt);
///
/// // Detect backend type
/// let backend = CapabilitiesQuery::detect_backend(&adapter);
/// println!("Backend: {:?}", backend);
/// # }
/// ```
pub struct CapabilitiesQuery;

impl CapabilitiesQuery {
    /// Create unified capabilities from a wgpu adapter.
    ///
    /// This queries the adapter for all supported features and limits,
    /// then normalizes them into the unified representation.
    ///
    /// # Arguments
    ///
    /// * `adapter` - The wgpu adapter to query
    ///
    /// # Returns
    ///
    /// Unified capabilities for the adapter.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::backend::capabilities::CapabilitiesQuery;
    ///
    /// # async fn example() {
    /// let instance = wgpu::Instance::default();
    /// let adapter = instance
    ///     .request_adapter(&wgpu::RequestAdapterOptions::default())
    ///     .await
    ///     .unwrap();
    ///
    /// let caps = CapabilitiesQuery::from_adapter(&adapter);
    /// println!("Backend: {:?}", caps.backend);
    /// # }
    /// ```
    pub fn from_adapter(adapter: &Adapter) -> UnifiedCapabilities {
        let backend = Self::detect_backend(adapter);
        let features = adapter.features();
        let limits = adapter.limits();

        let mut caps = UnifiedCapabilities {
            backend,
            max_texture_size: limits.max_texture_dimension_2d,
            max_buffer_size: limits.max_buffer_size as u64,
            max_compute_workgroup_size: [
                limits.max_compute_workgroup_size_x,
                limits.max_compute_workgroup_size_y,
                limits.max_compute_workgroup_size_z,
            ],
            ..Default::default()
        };

        // Detect base wgpu features
        if features.contains(wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE)
            && features.contains(wgpu::Features::RAY_QUERY)
        {
            caps.ray_tracing = FeatureLevel::Hardware;
        }

        if features.contains(wgpu::Features::TEXTURE_BINDING_ARRAY)
            && features.contains(
                wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
            )
        {
            caps.bindless = FeatureLevel::Hardware;
        }

        if features.contains(wgpu::Features::CONSERVATIVE_RASTERIZATION) {
            caps.conservative_rasterization = FeatureLevel::Hardware;
        }

        // Backend-specific detection for more accurate feature levels
        match backend {
            BackendType::Vulkan => {
                Self::detect_vulkan_features(adapter, &mut caps);
            }
            BackendType::Metal => {
                Self::detect_metal_features(adapter, &mut caps);
            }
            BackendType::Dx12 => {
                Self::detect_dx12_features(adapter, &mut caps);
            }
            BackendType::WebGpu => {
                Self::detect_webgpu_features(adapter, &mut caps);
            }
            BackendType::Gl => {
                // OpenGL has limited feature support
                caps.async_compute = FeatureLevel::Unavailable;
                caps.ray_tracing = FeatureLevel::Unavailable;
                caps.mesh_shaders = FeatureLevel::Unavailable;
                caps.bindless = FeatureLevel::Unavailable;
            }
            _ => {
                // Empty/Unknown backends have no features
            }
        }

        caps
    }

    /// Detect the backend type from an adapter.
    ///
    /// # Arguments
    ///
    /// * `adapter` - The wgpu adapter to query
    ///
    /// # Returns
    ///
    /// The detected backend type.
    #[inline]
    pub fn detect_backend(adapter: &Adapter) -> BackendType {
        BackendType::from_adapter(adapter)
    }

    /// Query support level for a specific feature by name.
    ///
    /// # Arguments
    ///
    /// * `caps` - The capabilities to query
    /// * `feature` - Feature name (case-insensitive)
    ///
    /// # Supported Features
    ///
    /// - `ray_tracing` / `rt`
    /// - `mesh_shaders` / `mesh`
    /// - `variable_rate_shading` / `vrs`
    /// - `bindless`
    /// - `async_compute`
    /// - `conservative_rasterization` / `conservative_raster`
    /// - `sampler_feedback`
    ///
    /// # Returns
    ///
    /// The feature support level, or `Unavailable` for unknown features.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::capabilities::{
    ///     CapabilitiesQuery, FeatureLevel, UnifiedCapabilities,
    /// };
    ///
    /// let caps = UnifiedCapabilities::default();
    /// let rt = CapabilitiesQuery::supports_feature(&caps, "ray_tracing");
    /// assert_eq!(rt, FeatureLevel::Unavailable);
    /// ```
    pub fn supports_feature(caps: &UnifiedCapabilities, feature: &str) -> FeatureLevel {
        match feature.to_lowercase().as_str() {
            "ray_tracing" | "rt" | "raytracing" => caps.ray_tracing,
            "mesh_shaders" | "mesh" | "meshshaders" => caps.mesh_shaders,
            "variable_rate_shading" | "vrs" => caps.variable_rate_shading,
            "bindless" | "descriptor_indexing" => caps.bindless,
            "async_compute" | "asynccompute" => caps.async_compute,
            "conservative_rasterization" | "conservative_raster" => caps.conservative_rasterization,
            "sampler_feedback" | "samplerfeedback" => caps.sampler_feedback,
            _ => FeatureLevel::Unavailable,
        }
    }

    /// Compare two capability sets.
    ///
    /// Returns ordering based on overall capability level.
    /// Higher capability sets are `Greater`.
    ///
    /// # Arguments
    ///
    /// * `a` - First capabilities to compare
    /// * `b` - Second capabilities to compare
    ///
    /// # Returns
    ///
    /// `Ordering` indicating relative capability levels.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::capabilities::{
    ///     CapabilitiesQuery, UnifiedCapabilities, FeatureLevel,
    /// };
    /// use std::cmp::Ordering;
    ///
    /// let mut a = UnifiedCapabilities::default();
    /// let mut b = UnifiedCapabilities::default();
    ///
    /// a.ray_tracing = FeatureLevel::Hardware;
    /// // b has no ray tracing
    ///
    /// assert_eq!(CapabilitiesQuery::compare_backends(&a, &b), Ordering::Greater);
    /// ```
    pub fn compare_backends(a: &UnifiedCapabilities, b: &UnifiedCapabilities) -> Ordering {
        a.feature_score().cmp(&b.feature_score())
    }

    // ========================================================================
    // Backend-Specific Detection
    // ========================================================================

    fn detect_vulkan_features(adapter: &Adapter, caps: &mut UnifiedCapabilities) {
        let vk_features = VulkanFeatures::detect(adapter);

        // Ray tracing
        if vk_features.ray_tracing && vk_features.ray_query {
            // Check for optimal RT (RTX 40 series has RT tier 1.1)
            if vk_features.buffer_device_address {
                caps.ray_tracing = FeatureLevel::Optimal;
            } else {
                caps.ray_tracing = FeatureLevel::Hardware;
            }
        } else if vk_features.ray_query {
            caps.ray_tracing = FeatureLevel::Hardware;
        }

        // Mesh shaders
        if vk_features.mesh_shading {
            caps.mesh_shaders = FeatureLevel::Hardware;
        }

        // Bindless
        if vk_features.supports_bindless() {
            if vk_features.descriptor_indexing && vk_features.buffer_device_address {
                caps.bindless = FeatureLevel::Optimal;
            } else {
                caps.bindless = FeatureLevel::Hardware;
            }
        }

        // Async compute - Vulkan always has at least one compute queue
        if vk_features.timeline_semaphores {
            caps.async_compute = FeatureLevel::Optimal;
        } else {
            caps.async_compute = FeatureLevel::Hardware;
        }

        // VRS - detect from wgpu features since VulkanFeatures doesn't expose it directly
        // wgpu doesn't expose VRS features yet, so leave as Unavailable
        // Future: When wgpu exposes VRS, detect it here
    }

    fn detect_metal_features(adapter: &Adapter, caps: &mut UnifiedCapabilities) {
        let metal_features = MetalFeatures::detect(adapter);

        // Ray tracing (Metal 3+ on M1+)
        if metal_features.ray_tracing {
            // M3 has hardware RT accelerator
            if metal_features.gpu_family.supports_metal3()
                && adapter.get_info().name.contains("M3")
            {
                caps.ray_tracing = FeatureLevel::Optimal;
            } else if metal_features.ray_tracing {
                caps.ray_tracing = FeatureLevel::Hardware;
            }
        }

        // Mesh shaders
        if metal_features.mesh_shaders {
            caps.mesh_shaders = FeatureLevel::Hardware;
        }

        // Bindless (argument buffers tier 2)
        if metal_features.supports_bindless() {
            caps.bindless = FeatureLevel::Hardware;
            // Apple Silicon has excellent bindless support
            if metal_features.gpu_family.supports_metal3() {
                caps.bindless = FeatureLevel::Optimal;
            }
        }

        // Async compute
        caps.async_compute = FeatureLevel::Hardware;

        // VRS (Apple Silicon only, limited)
        if metal_features.gpu_family.supports_metal3() {
            caps.variable_rate_shading = FeatureLevel::Hardware;
        }
    }

    fn detect_dx12_features(adapter: &Adapter, caps: &mut UnifiedCapabilities) {
        let dx12_features = D3D12Features::detect(adapter);

        // Ray tracing
        if dx12_features.supports_rt() {
            if dx12_features.supports_inline_rt() {
                caps.ray_tracing = FeatureLevel::Optimal;
            } else {
                caps.ray_tracing = FeatureLevel::Hardware;
            }
        }

        // Mesh shaders
        if dx12_features.supports_mesh_shaders() {
            caps.mesh_shaders = FeatureLevel::Hardware;
        }

        // VRS
        if dx12_features.supports_vrs() {
            // DX12 tier detection - if feature level >= 12.2, likely Tier 2
            if dx12_features.feature_level >= super::D3D12FeatureLevel::FL_12_2 {
                caps.variable_rate_shading = FeatureLevel::Optimal;
            } else {
                caps.variable_rate_shading = FeatureLevel::Hardware;
            }
        }

        // Bindless
        if dx12_features.supports_bindless() {
            caps.bindless = FeatureLevel::Hardware;
        }

        // Async compute (D3D12 always has async compute queue)
        caps.async_compute = FeatureLevel::Hardware;

        // Conservative rasterization
        if dx12_features.supports_conservative_raster() {
            caps.conservative_rasterization = FeatureLevel::Hardware;
        }

        // Sampler feedback
        if dx12_features.supports_sampler_feedback() {
            caps.sampler_feedback = FeatureLevel::Hardware;
        }
    }

    fn detect_webgpu_features(adapter: &Adapter, caps: &mut UnifiedCapabilities) {
        let webgpu_features = WebGpuFeatures::detect(adapter);

        // WebGPU has very limited advanced features currently
        caps.ray_tracing = FeatureLevel::Unavailable;
        caps.mesh_shaders = FeatureLevel::Unavailable;
        caps.variable_rate_shading = FeatureLevel::Unavailable;
        caps.sampler_feedback = FeatureLevel::Unavailable;

        // Bindless is partially supported via texture binding arrays
        let features = adapter.features();
        if features.contains(wgpu::Features::TEXTURE_BINDING_ARRAY) {
            caps.bindless = FeatureLevel::Emulated;
        }

        // Async compute depends on tier
        match webgpu_features.tier {
            WebGpuTier::Tier3 => {
                caps.async_compute = FeatureLevel::Hardware;
            }
            WebGpuTier::Tier2 => {
                caps.async_compute = FeatureLevel::Emulated;
            }
            WebGpuTier::Tier1 => {
                caps.async_compute = FeatureLevel::Unavailable;
            }
        }
    }
}

// ============================================================================
// FeatureRequirement
// ============================================================================

/// A feature requirement with minimum and preferred support levels.
///
/// This struct represents a requirement for a specific GPU feature,
/// including both the minimum acceptable level and the preferred level.
/// It can be used to check whether an adapter meets the requirements
/// for a specific rendering path or feature.
///
/// # Example
///
/// ```
/// use renderer_backend::backend::capabilities::{FeatureRequirement, FeatureLevel};
///
/// // Require at least hardware ray tracing, prefer optimal
/// let req = FeatureRequirement::new("ray_tracing", FeatureLevel::Hardware)
///     .with_preferred(FeatureLevel::Optimal);
///
/// assert!(req.is_satisfied(FeatureLevel::Hardware));
/// assert!(req.is_satisfied(FeatureLevel::Optimal));
/// assert!(!req.is_satisfied(FeatureLevel::Emulated));
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FeatureRequirement {
    /// Name of the required feature.
    pub name: String,

    /// Minimum acceptable support level.
    pub minimum: FeatureLevel,

    /// Preferred support level for optimal experience.
    pub preferred: FeatureLevel,
}

impl FeatureRequirement {
    /// Create a new feature requirement with minimum level.
    ///
    /// The preferred level defaults to the same as minimum.
    ///
    /// # Arguments
    ///
    /// * `name` - Feature name (e.g., "ray_tracing")
    /// * `min` - Minimum acceptable support level
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::capabilities::{FeatureRequirement, FeatureLevel};
    ///
    /// let req = FeatureRequirement::new("bindless", FeatureLevel::Hardware);
    /// assert_eq!(req.minimum, FeatureLevel::Hardware);
    /// assert_eq!(req.preferred, FeatureLevel::Hardware);
    /// ```
    pub fn new(name: &str, min: FeatureLevel) -> Self {
        Self {
            name: name.to_string(),
            minimum: min,
            preferred: min,
        }
    }

    /// Set the preferred support level.
    ///
    /// # Arguments
    ///
    /// * `pref` - Preferred support level
    ///
    /// # Returns
    ///
    /// Self for method chaining.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::capabilities::{FeatureRequirement, FeatureLevel};
    ///
    /// let req = FeatureRequirement::new("mesh_shaders", FeatureLevel::Emulated)
    ///     .with_preferred(FeatureLevel::Hardware);
    ///
    /// assert_eq!(req.minimum, FeatureLevel::Emulated);
    /// assert_eq!(req.preferred, FeatureLevel::Hardware);
    /// ```
    pub fn with_preferred(mut self, pref: FeatureLevel) -> Self {
        self.preferred = pref;
        self
    }

    /// Check if the actual support level satisfies the minimum requirement.
    ///
    /// # Arguments
    ///
    /// * `actual` - The actual support level to check
    ///
    /// # Returns
    ///
    /// `true` if `actual >= minimum`.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::capabilities::{FeatureRequirement, FeatureLevel};
    ///
    /// let req = FeatureRequirement::new("ray_tracing", FeatureLevel::Hardware);
    ///
    /// assert!(!req.is_satisfied(FeatureLevel::Unavailable));
    /// assert!(!req.is_satisfied(FeatureLevel::Emulated));
    /// assert!(req.is_satisfied(FeatureLevel::Hardware));
    /// assert!(req.is_satisfied(FeatureLevel::Optimal));
    /// ```
    #[inline]
    pub fn is_satisfied(&self, actual: FeatureLevel) -> bool {
        actual >= self.minimum
    }

    /// Check if the actual support level meets the preferred requirement.
    ///
    /// # Arguments
    ///
    /// * `actual` - The actual support level to check
    ///
    /// # Returns
    ///
    /// `true` if `actual >= preferred`.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::capabilities::{FeatureRequirement, FeatureLevel};
    ///
    /// let req = FeatureRequirement::new("ray_tracing", FeatureLevel::Hardware)
    ///     .with_preferred(FeatureLevel::Optimal);
    ///
    /// assert!(req.is_satisfied(FeatureLevel::Hardware));
    /// assert!(!req.is_preferred(FeatureLevel::Hardware));
    /// assert!(req.is_preferred(FeatureLevel::Optimal));
    /// ```
    #[inline]
    pub fn is_preferred(&self, actual: FeatureLevel) -> bool {
        actual >= self.preferred
    }

    /// Get a quality rating for the actual support level.
    ///
    /// # Returns
    ///
    /// - `0` if below minimum (not satisfied)
    /// - `1` if meets minimum but not preferred
    /// - `2` if meets or exceeds preferred
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::capabilities::{FeatureRequirement, FeatureLevel};
    ///
    /// let req = FeatureRequirement::new("ray_tracing", FeatureLevel::Emulated)
    ///     .with_preferred(FeatureLevel::Hardware);
    ///
    /// assert_eq!(req.quality_rating(FeatureLevel::Unavailable), 0);
    /// assert_eq!(req.quality_rating(FeatureLevel::Emulated), 1);
    /// assert_eq!(req.quality_rating(FeatureLevel::Hardware), 2);
    /// assert_eq!(req.quality_rating(FeatureLevel::Optimal), 2);
    /// ```
    pub fn quality_rating(&self, actual: FeatureLevel) -> u8 {
        if actual >= self.preferred {
            2
        } else if actual >= self.minimum {
            1
        } else {
            0
        }
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // FeatureLevel Tests
    // ========================================================================

    #[test]
    fn test_feature_level_ordering() {
        assert!(FeatureLevel::Unavailable < FeatureLevel::Emulated);
        assert!(FeatureLevel::Emulated < FeatureLevel::Hardware);
        assert!(FeatureLevel::Hardware < FeatureLevel::Optimal);
    }

    #[test]
    fn test_feature_level_equality() {
        assert_eq!(FeatureLevel::Hardware, FeatureLevel::Hardware);
        assert_ne!(FeatureLevel::Hardware, FeatureLevel::Emulated);
    }

    #[test]
    fn test_feature_level_is_available() {
        assert!(!FeatureLevel::Unavailable.is_available());
        assert!(FeatureLevel::Emulated.is_available());
        assert!(FeatureLevel::Hardware.is_available());
        assert!(FeatureLevel::Optimal.is_available());
    }

    #[test]
    fn test_feature_level_is_hardware() {
        assert!(!FeatureLevel::Unavailable.is_hardware());
        assert!(!FeatureLevel::Emulated.is_hardware());
        assert!(FeatureLevel::Hardware.is_hardware());
        assert!(FeatureLevel::Optimal.is_hardware());
    }

    #[test]
    fn test_feature_level_is_optimal() {
        assert!(!FeatureLevel::Unavailable.is_optimal());
        assert!(!FeatureLevel::Emulated.is_optimal());
        assert!(!FeatureLevel::Hardware.is_optimal());
        assert!(FeatureLevel::Optimal.is_optimal());
    }

    #[test]
    fn test_feature_level_name() {
        assert_eq!(FeatureLevel::Unavailable.name(), "Unavailable");
        assert_eq!(FeatureLevel::Emulated.name(), "Emulated");
        assert_eq!(FeatureLevel::Hardware.name(), "Hardware");
        assert_eq!(FeatureLevel::Optimal.name(), "Optimal");
    }

    #[test]
    fn test_feature_level_display() {
        assert_eq!(format!("{}", FeatureLevel::Hardware), "Hardware");
        assert_eq!(format!("{}", FeatureLevel::Optimal), "Optimal");
    }

    #[test]
    fn test_feature_level_default() {
        assert_eq!(FeatureLevel::default(), FeatureLevel::Unavailable);
    }

    // ========================================================================
    // UnifiedCapabilities Tests
    // ========================================================================

    #[test]
    fn test_unified_capabilities_default() {
        let caps = UnifiedCapabilities::default();

        assert_eq!(caps.backend, BackendType::Unknown);
        assert_eq!(caps.ray_tracing, FeatureLevel::Unavailable);
        assert_eq!(caps.mesh_shaders, FeatureLevel::Unavailable);
        assert_eq!(caps.variable_rate_shading, FeatureLevel::Unavailable);
        assert_eq!(caps.bindless, FeatureLevel::Unavailable);
        assert_eq!(caps.async_compute, FeatureLevel::Unavailable);
        assert_eq!(caps.max_texture_size, 2048);
    }

    #[test]
    fn test_unified_capabilities_supports_full_rt() {
        let mut caps = UnifiedCapabilities::default();

        assert!(!caps.supports_full_rt());

        caps.ray_tracing = FeatureLevel::Emulated;
        assert!(!caps.supports_full_rt());

        caps.ray_tracing = FeatureLevel::Hardware;
        assert!(caps.supports_full_rt());

        caps.ray_tracing = FeatureLevel::Optimal;
        assert!(caps.supports_full_rt());
    }

    #[test]
    fn test_unified_capabilities_supports_gpu_driven() {
        let mut caps = UnifiedCapabilities::default();

        assert!(!caps.supports_gpu_driven());

        caps.bindless = FeatureLevel::Hardware;
        assert!(!caps.supports_gpu_driven()); // Still needs async_compute

        caps.async_compute = FeatureLevel::Emulated;
        assert!(caps.supports_gpu_driven()); // Emulated is available

        caps.async_compute = FeatureLevel::Hardware;
        assert!(caps.supports_gpu_driven());
    }

    #[test]
    fn test_unified_capabilities_supports_advanced_features() {
        let mut caps = UnifiedCapabilities::default();

        assert!(!caps.supports_advanced_features());

        caps.ray_tracing = FeatureLevel::Hardware;
        assert!(caps.supports_advanced_features());

        caps.ray_tracing = FeatureLevel::Unavailable;
        caps.mesh_shaders = FeatureLevel::Hardware;
        assert!(caps.supports_advanced_features());

        caps.mesh_shaders = FeatureLevel::Unavailable;
        caps.variable_rate_shading = FeatureLevel::Optimal;
        assert!(caps.supports_advanced_features());
    }

    #[test]
    fn test_unified_capabilities_feature_score() {
        let low_caps = UnifiedCapabilities::default();
        let mut high_caps = UnifiedCapabilities::default();

        high_caps.ray_tracing = FeatureLevel::Optimal;
        high_caps.mesh_shaders = FeatureLevel::Hardware;
        high_caps.bindless = FeatureLevel::Optimal;

        assert!(high_caps.feature_score() > low_caps.feature_score());

        // Test limits contribute to score
        high_caps.max_texture_size = 16384;
        let score_with_limits = high_caps.feature_score();
        high_caps.max_texture_size = 2048;
        let score_without_limits = high_caps.feature_score();

        assert!(score_with_limits > score_without_limits);
    }

    // ========================================================================
    // CapabilitiesQuery Tests
    // ========================================================================

    #[test]
    fn test_supports_feature_ray_tracing() {
        let mut caps = UnifiedCapabilities::default();
        caps.ray_tracing = FeatureLevel::Hardware;

        assert_eq!(
            CapabilitiesQuery::supports_feature(&caps, "ray_tracing"),
            FeatureLevel::Hardware
        );
        assert_eq!(
            CapabilitiesQuery::supports_feature(&caps, "rt"),
            FeatureLevel::Hardware
        );
        assert_eq!(
            CapabilitiesQuery::supports_feature(&caps, "RT"),
            FeatureLevel::Hardware
        );
    }

    #[test]
    fn test_supports_feature_mesh_shaders() {
        let mut caps = UnifiedCapabilities::default();
        caps.mesh_shaders = FeatureLevel::Optimal;

        assert_eq!(
            CapabilitiesQuery::supports_feature(&caps, "mesh_shaders"),
            FeatureLevel::Optimal
        );
        assert_eq!(
            CapabilitiesQuery::supports_feature(&caps, "mesh"),
            FeatureLevel::Optimal
        );
    }

    #[test]
    fn test_supports_feature_unknown() {
        let caps = UnifiedCapabilities::default();

        assert_eq!(
            CapabilitiesQuery::supports_feature(&caps, "unknown_feature"),
            FeatureLevel::Unavailable
        );
    }

    #[test]
    fn test_compare_backends_equal() {
        let a = UnifiedCapabilities::default();
        let b = UnifiedCapabilities::default();

        assert_eq!(
            CapabilitiesQuery::compare_backends(&a, &b),
            Ordering::Equal
        );
    }

    #[test]
    fn test_compare_backends_greater() {
        let mut a = UnifiedCapabilities::default();
        let b = UnifiedCapabilities::default();

        a.ray_tracing = FeatureLevel::Hardware;

        assert_eq!(
            CapabilitiesQuery::compare_backends(&a, &b),
            Ordering::Greater
        );
    }

    #[test]
    fn test_compare_backends_less() {
        let a = UnifiedCapabilities::default();
        let mut b = UnifiedCapabilities::default();

        b.ray_tracing = FeatureLevel::Optimal;
        b.mesh_shaders = FeatureLevel::Hardware;

        assert_eq!(CapabilitiesQuery::compare_backends(&a, &b), Ordering::Less);
    }

    // ========================================================================
    // FeatureRequirement Tests
    // ========================================================================

    #[test]
    fn test_feature_requirement_new() {
        let req = FeatureRequirement::new("ray_tracing", FeatureLevel::Hardware);

        assert_eq!(req.name, "ray_tracing");
        assert_eq!(req.minimum, FeatureLevel::Hardware);
        assert_eq!(req.preferred, FeatureLevel::Hardware);
    }

    #[test]
    fn test_feature_requirement_with_preferred() {
        let req = FeatureRequirement::new("ray_tracing", FeatureLevel::Hardware)
            .with_preferred(FeatureLevel::Optimal);

        assert_eq!(req.minimum, FeatureLevel::Hardware);
        assert_eq!(req.preferred, FeatureLevel::Optimal);
    }

    #[test]
    fn test_feature_requirement_is_satisfied() {
        let req = FeatureRequirement::new("bindless", FeatureLevel::Hardware);

        assert!(!req.is_satisfied(FeatureLevel::Unavailable));
        assert!(!req.is_satisfied(FeatureLevel::Emulated));
        assert!(req.is_satisfied(FeatureLevel::Hardware));
        assert!(req.is_satisfied(FeatureLevel::Optimal));
    }

    #[test]
    fn test_feature_requirement_is_preferred() {
        let req = FeatureRequirement::new("mesh_shaders", FeatureLevel::Emulated)
            .with_preferred(FeatureLevel::Hardware);

        assert!(!req.is_preferred(FeatureLevel::Emulated));
        assert!(req.is_preferred(FeatureLevel::Hardware));
        assert!(req.is_preferred(FeatureLevel::Optimal));
    }

    #[test]
    fn test_feature_requirement_quality_rating() {
        let req = FeatureRequirement::new("ray_tracing", FeatureLevel::Emulated)
            .with_preferred(FeatureLevel::Hardware);

        assert_eq!(req.quality_rating(FeatureLevel::Unavailable), 0);
        assert_eq!(req.quality_rating(FeatureLevel::Emulated), 1);
        assert_eq!(req.quality_rating(FeatureLevel::Hardware), 2);
        assert_eq!(req.quality_rating(FeatureLevel::Optimal), 2);
    }

    #[test]
    fn test_feature_requirement_edge_cases() {
        // Minimum at Unavailable - everything satisfies
        let req_any = FeatureRequirement::new("feature", FeatureLevel::Unavailable);
        assert!(req_any.is_satisfied(FeatureLevel::Unavailable));
        assert!(req_any.is_satisfied(FeatureLevel::Emulated));

        // Minimum at Optimal - only Optimal satisfies
        let req_optimal = FeatureRequirement::new("feature", FeatureLevel::Optimal);
        assert!(!req_optimal.is_satisfied(FeatureLevel::Hardware));
        assert!(req_optimal.is_satisfied(FeatureLevel::Optimal));
    }

    // ========================================================================
    // Integration Tests
    // ========================================================================

    #[test]
    fn test_backend_type_variants() {
        // Verify all expected backend types exist
        let backends = [
            BackendType::Vulkan,
            BackendType::Metal,
            BackendType::Dx12,
            BackendType::Gl,
            BackendType::WebGpu,
        ];

        for backend in backends {
            let mut caps = UnifiedCapabilities::default();
            caps.backend = backend;
            assert_eq!(caps.backend, backend);
        }
    }

    #[test]
    fn test_feature_level_conversion() {
        // Test that feature levels can be used as u32 for scoring
        assert_eq!(FeatureLevel::Unavailable as u32, 0);
        assert_eq!(FeatureLevel::Emulated as u32, 1);
        assert_eq!(FeatureLevel::Hardware as u32, 2);
        assert_eq!(FeatureLevel::Optimal as u32, 3);
    }

    #[test]
    fn test_limit_queries() {
        let mut caps = UnifiedCapabilities::default();

        // Test max_texture_size
        caps.max_texture_size = 8192;
        assert_eq!(caps.max_texture_size, 8192);

        // Test max_buffer_size
        caps.max_buffer_size = 4 * 1024 * 1024 * 1024; // 4 GB
        assert_eq!(caps.max_buffer_size, 4 * 1024 * 1024 * 1024);

        // Test max_compute_workgroup_size
        caps.max_compute_workgroup_size = [256, 256, 128];
        assert_eq!(caps.max_compute_workgroup_size[0], 256);
        assert_eq!(caps.max_compute_workgroup_size[1], 256);
        assert_eq!(caps.max_compute_workgroup_size[2], 128);
    }
}
