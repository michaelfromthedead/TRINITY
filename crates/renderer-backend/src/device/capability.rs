//! Capability tier detection for TRINITY.
//!
//! This module provides GPU capability tier classification based on hardware
//! features and limits. Unlike [`FeatureTier`] which focuses on feature count,
//! `CapabilityTier` classifies GPUs based on specific feature combinations that
//! enable different rendering paths.
//!
//! # Tier Classification
//!
//! | Tier | Description | Requirements |
//! |------|-------------|--------------|
//! | Full | Ray tracing capable | RT acceleration structures present |
//! | Advanced | Modern GPU-driven | Bindless + Multi-draw indirect count + Large workgroups |
//! | Standard | Typical desktop GPU | 8K textures + Compute shaders |
//! | Minimal | Fallback tier | Basic WebGPU support |
//!
//! # Example
//!
//! ```no_run
//! use renderer_backend::device::CapabilityTier;
//!
//! # async fn example() {
//! let instance = wgpu::Instance::default();
//! let adapter = instance
//!     .request_adapter(&wgpu::RequestAdapterOptions::default())
//!     .await
//!     .unwrap();
//!
//! let tier = CapabilityTier::from_adapter(&adapter);
//! println!("GPU capability tier: {}", tier.tier_name());
//!
//! if tier.supports_ray_tracing() {
//!     println!("Ray tracing available!");
//! }
//!
//! if tier.supports_bindless() {
//!     println!("Bindless rendering supported!");
//! }
//! # }
//! ```
//!
//! # Architecture
//!
//! The capability tier system is designed to enable graceful degradation:
//!
//! ```text
//! CapabilityTier::Full
//!   └── Ray tracing render path
//!   └── Bindless rendering
//!   └── GPU-driven culling
//!
//! CapabilityTier::Advanced
//!   └── Bindless rendering
//!   └── GPU-driven culling
//!   └── Multi-draw indirect
//!
//! CapabilityTier::Standard
//!   └── Traditional rendering
//!   └── High-res textures
//!   └── Compute shaders
//!
//! CapabilityTier::Minimal
//!   └── Basic WebGPU fallback
//!   └── Limited texture sizes
//!   └── No advanced features
//! ```

use log::{debug, info};
use wgpu::{Adapter, Features, Limits};

// ============================================================================
// Constants
// ============================================================================

/// Minimum 2D texture dimension required for Standard tier (8K).
const STANDARD_TIER_MIN_TEXTURE_2D: u32 = 8192;

/// Minimum compute workgroup invocations for Advanced tier.
const ADVANCED_TIER_MIN_WORKGROUP_INVOCATIONS: u32 = 1024;

/// Minimum storage buffer binding size for Standard tier (128 MB).
const STANDARD_TIER_MIN_STORAGE_BUFFER_SIZE: u32 = 128 * 1024 * 1024;

// ============================================================================
// CapabilityTier
// ============================================================================

/// GPU capability tier classification.
///
/// Capability tiers classify GPUs based on specific feature combinations that
/// enable different rendering paths. The tier determines which render techniques
/// and optimizations are available.
///
/// # Tier Ordering
///
/// Tiers are ordered from lowest to highest capability:
/// `Minimal < Standard < Advanced < Full`
///
/// This ordering allows code like:
/// ```
/// use renderer_backend::device::CapabilityTier;
///
/// let tier = CapabilityTier::Advanced;
/// assert!(tier >= CapabilityTier::Standard);
/// assert!(tier < CapabilityTier::Full);
/// ```
///
/// # Feature Requirements
///
/// - **Full**: Ray tracing acceleration structures (`RAY_TRACING_ACCELERATION_STRUCTURE`)
/// - **Advanced**: All of:
///   - Texture binding arrays (`TEXTURE_BINDING_ARRAY`)
///   - Multi-draw indirect count (`MULTI_DRAW_INDIRECT_COUNT`)
///   - Large workgroups (>= 1024 invocations)
/// - **Standard**: All of:
///   - 8K texture support (`max_texture_dimension_2d >= 8192`)
///   - Compute shaders (always available in wgpu)
///   - 128MB storage buffer binding
/// - **Minimal**: Everything else (basic WebGPU support)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CapabilityTier {
    /// Minimal tier - basic WebGPU support only.
    ///
    /// This tier represents the lowest common denominator across all wgpu
    /// backends. Use this tier for maximum compatibility at the cost of
    /// visual quality and performance.
    ///
    /// Typical hardware: Low-end mobile GPUs, WebGL2 contexts, software renderers.
    #[default]
    Minimal,

    /// Standard tier - typical desktop GPU capabilities.
    ///
    /// This tier provides good visual quality with standard rendering
    /// techniques. Most desktop and modern mobile GPUs qualify for this tier.
    ///
    /// Requirements:
    /// - 8K texture support
    /// - 128MB storage buffer binding
    /// - Compute shader support (always available)
    ///
    /// Typical hardware: Intel integrated GPUs, older discrete GPUs, modern mobile.
    Standard,

    /// Advanced tier - modern GPU-driven rendering.
    ///
    /// This tier enables GPU-driven rendering pipelines with bindless
    /// resources and efficient multi-draw operations. Requires relatively
    /// modern discrete GPUs.
    ///
    /// Requirements (all must be met):
    /// - Texture binding arrays (bindless)
    /// - Multi-draw indirect count
    /// - Large workgroups (>= 1024 invocations)
    ///
    /// Typical hardware: GTX 10-series+, RX 500+, Intel Arc, Apple M1+.
    Advanced,

    /// Full tier - ray tracing capable.
    ///
    /// This tier represents the highest capability level with hardware
    /// ray tracing support. Enables ray-traced shadows, reflections,
    /// and global illumination.
    ///
    /// Requirements:
    /// - Ray tracing acceleration structures
    ///
    /// Typical hardware: RTX 20/30/40-series, RX 6000+, Intel Arc.
    Full,
}

impl CapabilityTier {
    /// Detect capability tier from a wgpu adapter.
    ///
    /// This method queries the adapter's features and limits to determine
    /// the highest capability tier the hardware supports.
    ///
    /// # Arguments
    ///
    /// * `adapter` - The wgpu adapter to analyze
    ///
    /// # Returns
    ///
    /// The detected capability tier.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::device::CapabilityTier;
    ///
    /// # async fn example() {
    /// let instance = wgpu::Instance::default();
    /// let adapter = instance
    ///     .request_adapter(&wgpu::RequestAdapterOptions::default())
    ///     .await
    ///     .unwrap();
    ///
    /// let tier = CapabilityTier::from_adapter(&adapter);
    /// println!("Detected tier: {}", tier.tier_name());
    /// # }
    /// ```
    pub fn from_adapter(adapter: &Adapter) -> Self {
        let features = adapter.features();
        let limits = adapter.limits();

        Self::from_features_and_limits(&features, &limits)
    }

    /// Detect capability tier from features and limits.
    ///
    /// This is the core tier detection algorithm. It checks feature flags
    /// and hardware limits to determine the appropriate tier.
    ///
    /// # Arguments
    ///
    /// * `features` - The wgpu features supported by the device
    /// * `limits` - The wgpu limits of the device
    ///
    /// # Returns
    ///
    /// The detected capability tier.
    pub fn from_features_and_limits(features: &Features, limits: &Limits) -> Self {
        // Check for Full tier (ray tracing)
        if Self::check_full_tier(features) {
            debug!("CapabilityTier: Detected Full tier (RT support)");
            return CapabilityTier::Full;
        }

        // Check for Advanced tier (bindless + multi-draw + large workgroup)
        if Self::check_advanced_tier(features, limits) {
            debug!("CapabilityTier: Detected Advanced tier (bindless + multi-draw)");
            return CapabilityTier::Advanced;
        }

        // Check for Standard tier (8K textures + compute)
        if Self::check_standard_tier(limits) {
            debug!("CapabilityTier: Detected Standard tier (8K textures)");
            return CapabilityTier::Standard;
        }

        // Fallback to Minimal
        debug!("CapabilityTier: Detected Minimal tier (fallback)");
        CapabilityTier::Minimal
    }

    /// Check if features meet Full tier requirements.
    ///
    /// Full tier requires ray tracing acceleration structure support.
    fn check_full_tier(features: &Features) -> bool {
        features.contains(Features::RAY_TRACING_ACCELERATION_STRUCTURE)
    }

    /// Check if features and limits meet Advanced tier requirements.
    ///
    /// Advanced tier requires:
    /// - Texture binding arrays (bindless)
    /// - Multi-draw indirect count
    /// - Large workgroups (>= 1024 invocations)
    fn check_advanced_tier(features: &Features, limits: &Limits) -> bool {
        let has_bindless = features.contains(Features::TEXTURE_BINDING_ARRAY);
        let has_multi_draw = features.contains(Features::MULTI_DRAW_INDIRECT_COUNT);
        let has_large_workgroup =
            limits.max_compute_invocations_per_workgroup >= ADVANCED_TIER_MIN_WORKGROUP_INVOCATIONS;

        has_bindless && has_multi_draw && has_large_workgroup
    }

    /// Check if limits meet Standard tier requirements.
    ///
    /// Standard tier requires:
    /// - 8K texture support
    /// - 128MB storage buffer binding
    fn check_standard_tier(limits: &Limits) -> bool {
        let has_8k_textures = limits.max_texture_dimension_2d >= STANDARD_TIER_MIN_TEXTURE_2D;
        let has_large_storage =
            limits.max_storage_buffer_binding_size >= STANDARD_TIER_MIN_STORAGE_BUFFER_SIZE;

        has_8k_textures && has_large_storage
    }

    /// Get the human-readable tier name.
    ///
    /// # Returns
    ///
    /// A static string with the tier name.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::device::CapabilityTier;
    ///
    /// assert_eq!(CapabilityTier::Full.tier_name(), "Full");
    /// assert_eq!(CapabilityTier::Minimal.tier_name(), "Minimal");
    /// ```
    pub fn tier_name(&self) -> &'static str {
        match self {
            CapabilityTier::Full => "Full",
            CapabilityTier::Advanced => "Advanced",
            CapabilityTier::Standard => "Standard",
            CapabilityTier::Minimal => "Minimal",
        }
    }

    /// Get a detailed description of the tier.
    ///
    /// # Returns
    ///
    /// A static string describing the tier's capabilities.
    pub fn description(&self) -> &'static str {
        match self {
            CapabilityTier::Full => "Full tier: Ray tracing capable GPU",
            CapabilityTier::Advanced => "Advanced tier: Bindless + GPU-driven rendering",
            CapabilityTier::Standard => "Standard tier: 8K textures + compute shaders",
            CapabilityTier::Minimal => "Minimal tier: Basic WebGPU fallback",
        }
    }

    // ========================================================================
    // Capability Query Methods
    // ========================================================================

    /// Check if ray tracing is supported.
    ///
    /// Ray tracing is only available in the Full tier.
    ///
    /// # Returns
    ///
    /// `true` if the tier supports ray tracing.
    #[inline]
    pub fn supports_ray_tracing(&self) -> bool {
        matches!(self, CapabilityTier::Full)
    }

    /// Check if bindless rendering is supported.
    ///
    /// Bindless rendering is available in Advanced and Full tiers.
    ///
    /// # Returns
    ///
    /// `true` if the tier supports bindless rendering.
    #[inline]
    pub fn supports_bindless(&self) -> bool {
        matches!(self, CapabilityTier::Advanced | CapabilityTier::Full)
    }

    /// Check if GPU-driven rendering is supported.
    ///
    /// GPU-driven rendering (multi-draw indirect count) is available
    /// in Advanced and Full tiers.
    ///
    /// # Returns
    ///
    /// `true` if the tier supports GPU-driven rendering.
    #[inline]
    pub fn supports_gpu_driven(&self) -> bool {
        matches!(self, CapabilityTier::Advanced | CapabilityTier::Full)
    }

    /// Check if high-resolution (8K) textures are supported.
    ///
    /// 8K textures are available in Standard, Advanced, and Full tiers.
    ///
    /// # Returns
    ///
    /// `true` if the tier supports 8K textures.
    #[inline]
    pub fn supports_8k_textures(&self) -> bool {
        !matches!(self, CapabilityTier::Minimal)
    }

    /// Check if compute shaders are supported.
    ///
    /// Compute shaders are always available in wgpu, but this method
    /// returns `true` only for Standard tier and above where compute
    /// is practical (large storage buffers available).
    ///
    /// # Returns
    ///
    /// `true` if the tier has practical compute shader support.
    #[inline]
    pub fn supports_compute(&self) -> bool {
        !matches!(self, CapabilityTier::Minimal)
    }

    /// Check if this tier meets or exceeds a required tier.
    ///
    /// # Arguments
    ///
    /// * `required` - The minimum required tier
    ///
    /// # Returns
    ///
    /// `true` if this tier meets or exceeds the requirement.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::device::CapabilityTier;
    ///
    /// let tier = CapabilityTier::Advanced;
    /// assert!(tier.meets_requirement(CapabilityTier::Standard));
    /// assert!(tier.meets_requirement(CapabilityTier::Advanced));
    /// assert!(!tier.meets_requirement(CapabilityTier::Full));
    /// ```
    #[inline]
    pub fn meets_requirement(&self, required: CapabilityTier) -> bool {
        *self >= required
    }
}

// ============================================================================
// Trait Implementations
// ============================================================================

impl PartialOrd for CapabilityTier {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CapabilityTier {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let self_rank = self.rank();
        let other_rank = other.rank();
        self_rank.cmp(&other_rank)
    }
}

impl CapabilityTier {
    /// Get numeric rank for tier comparison.
    #[inline]
    fn rank(&self) -> u8 {
        match self {
            CapabilityTier::Minimal => 0,
            CapabilityTier::Standard => 1,
            CapabilityTier::Advanced => 2,
            CapabilityTier::Full => 3,
        }
    }
}

impl std::fmt::Display for CapabilityTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.tier_name())
    }
}

// ============================================================================
// CapabilityReport
// ============================================================================

/// Detailed capability report for an adapter.
///
/// This struct provides a comprehensive view of an adapter's capabilities,
/// including the detected tier and individual feature availability.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::device::{CapabilityTier, CapabilityReport};
///
/// # async fn example() {
/// let instance = wgpu::Instance::default();
/// let adapter = instance
///     .request_adapter(&wgpu::RequestAdapterOptions::default())
///     .await
///     .unwrap();
///
/// let report = CapabilityReport::from_adapter(&adapter);
/// println!("{}", report);
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct CapabilityReport {
    /// The detected capability tier.
    pub tier: CapabilityTier,

    /// Whether ray tracing acceleration structures are available.
    pub has_ray_tracing: bool,

    /// Whether texture binding arrays (bindless) are available.
    pub has_bindless: bool,

    /// Whether multi-draw indirect count is available.
    pub has_multi_draw_indirect_count: bool,

    /// Whether storage resource binding arrays are available.
    pub has_storage_binding_array: bool,

    /// Maximum 2D texture dimension.
    pub max_texture_dimension_2d: u32,

    /// Maximum compute workgroup invocations.
    pub max_compute_invocations: u32,

    /// Maximum storage buffer binding size in bytes.
    pub max_storage_buffer_binding_size: u32,

    /// Adapter name (for reporting).
    pub adapter_name: String,
}

impl CapabilityReport {
    /// Generate a capability report from an adapter.
    ///
    /// # Arguments
    ///
    /// * `adapter` - The wgpu adapter to analyze
    ///
    /// # Returns
    ///
    /// A detailed capability report.
    pub fn from_adapter(adapter: &Adapter) -> Self {
        let features = adapter.features();
        let limits = adapter.limits();
        let info = adapter.get_info();

        let tier = CapabilityTier::from_features_and_limits(&features, &limits);

        Self {
            tier,
            has_ray_tracing: features.contains(Features::RAY_TRACING_ACCELERATION_STRUCTURE),
            has_bindless: features.contains(Features::TEXTURE_BINDING_ARRAY),
            has_multi_draw_indirect_count: features.contains(Features::MULTI_DRAW_INDIRECT_COUNT),
            has_storage_binding_array: features.contains(Features::STORAGE_RESOURCE_BINDING_ARRAY),
            max_texture_dimension_2d: limits.max_texture_dimension_2d,
            max_compute_invocations: limits.max_compute_invocations_per_workgroup,
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
            adapter_name: info.name.clone(),
        }
    }

    /// Log the capability report at INFO level.
    pub fn log_report(&self) {
        info!("=== Capability Report: {} ===", self.adapter_name);
        info!("  Tier: {} ({})", self.tier.tier_name(), self.tier.description());
        info!("  Features:");
        info!("    Ray Tracing:     {}", self.has_ray_tracing);
        info!("    Bindless:        {}", self.has_bindless);
        info!("    Multi-Draw:      {}", self.has_multi_draw_indirect_count);
        info!("    Storage Arrays:  {}", self.has_storage_binding_array);
        info!("  Limits:");
        info!("    Max Texture 2D:  {}", self.max_texture_dimension_2d);
        info!("    Max Workgroup:   {}", self.max_compute_invocations);
        info!(
            "    Max Storage:     {} MB",
            self.max_storage_buffer_binding_size / (1024 * 1024)
        );
    }
}

impl std::fmt::Display for CapabilityReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Capability Report: {}", self.adapter_name)?;
        writeln!(f, "  Tier: {} - {}", self.tier.tier_name(), self.tier.description())?;
        writeln!(f)?;
        writeln!(f, "  Features:")?;
        writeln!(f, "    Ray Tracing:            {}", self.has_ray_tracing)?;
        writeln!(f, "    Bindless Textures:      {}", self.has_bindless)?;
        writeln!(f, "    Multi-Draw Indirect:    {}", self.has_multi_draw_indirect_count)?;
        writeln!(f, "    Storage Binding Array:  {}", self.has_storage_binding_array)?;
        writeln!(f)?;
        writeln!(f, "  Limits:")?;
        writeln!(f, "    Max 2D Texture:         {} px", self.max_texture_dimension_2d)?;
        writeln!(f, "    Max Workgroup Size:     {} invocations", self.max_compute_invocations)?;
        writeln!(
            f,
            "    Max Storage Binding:    {} MB",
            self.max_storage_buffer_binding_size / (1024 * 1024)
        )?;
        writeln!(f)?;
        writeln!(f, "  Capabilities:")?;
        writeln!(f, "    Ray Tracing:            {}", self.tier.supports_ray_tracing())?;
        writeln!(f, "    Bindless Rendering:     {}", self.tier.supports_bindless())?;
        writeln!(f, "    GPU-Driven Rendering:   {}", self.tier.supports_gpu_driven())?;
        writeln!(f, "    8K Textures:            {}", self.tier.supports_8k_textures())?;
        writeln!(f, "    Compute Shaders:        {}", self.tier.supports_compute())?;
        Ok(())
    }
}

// ============================================================================
// RenderPath
// ============================================================================

/// Render path selection based on capability tier.
///
/// The render path determines which rendering techniques are used.
/// Higher-tier paths provide better visual quality and performance
/// but require more capable hardware.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RenderPath {
    /// Ray-traced rendering with hardware acceleration.
    ///
    /// Uses ray tracing for shadows, reflections, and global illumination.
    /// Requires Full tier (ray tracing acceleration structures).
    RayTraced,

    /// GPU-driven rendering with bindless resources.
    ///
    /// Uses indirect draw calls and bindless textures for efficient
    /// rendering of large scenes. Requires Advanced tier.
    GPUDriven,

    /// Traditional forward/deferred rendering.
    ///
    /// Standard rendering pipeline with per-object draw calls.
    /// Works on Standard tier hardware.
    Traditional,

    /// Fallback rendering for minimal hardware.
    ///
    /// Simplified rendering with reduced features for
    /// low-end or WebGL2 hardware.
    Fallback,
}

impl RenderPath {
    /// Get the human-readable name of this render path.
    pub fn name(&self) -> &'static str {
        match self {
            RenderPath::RayTraced => "RayTraced",
            RenderPath::GPUDriven => "GPUDriven",
            RenderPath::Traditional => "Traditional",
            RenderPath::Fallback => "Fallback",
        }
    }

    /// Get a description of this render path.
    pub fn description(&self) -> &'static str {
        match self {
            RenderPath::RayTraced => "Hardware ray tracing for shadows, reflections, and GI",
            RenderPath::GPUDriven => "GPU-driven rendering with bindless resources",
            RenderPath::Traditional => "Traditional forward/deferred rendering",
            RenderPath::Fallback => "Fallback rendering for minimal hardware",
        }
    }
}

impl std::fmt::Display for RenderPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ============================================================================
// TextureCompression
// ============================================================================

/// Texture compression format selection.
///
/// Different platforms and hardware support different texture compression
/// formats. This enum represents the best available format for the hardware.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureCompression {
    /// BC (Block Compression) formats - DirectX/Vulkan desktop.
    ///
    /// Widely supported on desktop GPUs. Provides good compression
    /// ratios with acceptable quality.
    BC,

    /// ASTC (Adaptive Scalable Texture Compression) - mobile/modern desktop.
    ///
    /// Variable block size compression with excellent quality/size ratio.
    /// Common on mobile GPUs and newer desktop hardware.
    ASTC,

    /// ETC2 (Ericsson Texture Compression 2) - mobile fallback.
    ///
    /// Required by OpenGL ES 3.0 and WebGL2. Lower quality than BC/ASTC
    /// but widely supported on mobile.
    ETC2,

    /// No compression available.
    ///
    /// Use uncompressed textures. Higher memory usage and bandwidth
    /// but works everywhere.
    None,
}

impl TextureCompression {
    /// Get the human-readable name of this compression format.
    pub fn name(&self) -> &'static str {
        match self {
            TextureCompression::BC => "BC",
            TextureCompression::ASTC => "ASTC",
            TextureCompression::ETC2 => "ETC2",
            TextureCompression::None => "None",
        }
    }

    /// Get a description of this compression format.
    pub fn description(&self) -> &'static str {
        match self {
            TextureCompression::BC => "Block Compression (BC1-BC7) - desktop standard",
            TextureCompression::ASTC => "Adaptive Scalable Texture Compression - mobile/modern",
            TextureCompression::ETC2 => "Ericsson Texture Compression 2 - mobile fallback",
            TextureCompression::None => "No compression - maximum compatibility",
        }
    }

    /// Check if this format provides hardware-accelerated decompression.
    pub fn is_hardware_accelerated(&self) -> bool {
        !matches!(self, TextureCompression::None)
    }
}

impl std::fmt::Display for TextureCompression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ============================================================================
// CapabilityManager
// ============================================================================

/// Runtime capability manager for GPU feature queries.
///
/// `CapabilityManager` provides a centralized interface for querying GPU
/// capabilities at runtime. It wraps the detected capability tier and
/// cached feature/limit information to enable efficient capability queries
/// without repeated adapter/device inspection.
///
/// # Features
///
/// - Runtime feature queries (ray tracing, bindless, GPU culling, etc.)
/// - Render path selection based on hardware capabilities
/// - Texture compression format selection
/// - Bindless texture limit queries
/// - Comprehensive capability reporting
///
/// # Example
///
/// ```no_run
/// use renderer_backend::device::{CapabilityManager, RenderPath};
///
/// # async fn example() {
/// let instance = wgpu::Instance::default();
/// let adapter = instance
///     .request_adapter(&wgpu::RequestAdapterOptions::default())
///     .await
///     .unwrap();
///
/// let manager = CapabilityManager::from_adapter(&adapter);
///
/// // Query capabilities
/// if manager.supports_ray_tracing() {
///     println!("Ray tracing available!");
/// }
///
/// // Select render path
/// let path = manager.select_render_path();
/// println!("Using render path: {}", path);
///
/// // Get bindless limits
/// println!("Max bindless textures: {}", manager.max_bindless_textures());
///
/// // Generate full report
/// let report = manager.report();
/// println!("{}", report);
/// # }
/// ```
///
/// # Caching
///
/// All capability information is cached at construction time. Subsequent
/// queries are simple field lookups with no GPU communication overhead.
#[derive(Debug, Clone)]
pub struct CapabilityManager {
    /// The detected capability tier.
    tier: CapabilityTier,

    /// Cached feature flags from the adapter.
    features: Features,

    /// Cached limits from the adapter.
    limits: Limits,

    /// Adapter name for reporting.
    adapter_name: String,

    /// Whether timestamp queries are supported.
    has_timestamp_queries: bool,

    /// Whether push constants are supported.
    has_push_constants: bool,

    /// Maximum push constant size (0 if not supported).
    max_push_constant_size: u32,
}

impl CapabilityManager {
    /// Create a CapabilityManager from a wgpu adapter.
    ///
    /// This queries the adapter for features and limits, determines the
    /// capability tier, and caches all information for efficient runtime
    /// queries.
    ///
    /// # Arguments
    ///
    /// * `adapter` - The wgpu adapter to analyze
    ///
    /// # Returns
    ///
    /// A new CapabilityManager with cached capability information.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::device::CapabilityManager;
    ///
    /// # async fn example() {
    /// let instance = wgpu::Instance::default();
    /// let adapter = instance
    ///     .request_adapter(&wgpu::RequestAdapterOptions::default())
    ///     .await
    ///     .unwrap();
    ///
    /// let manager = CapabilityManager::from_adapter(&adapter);
    /// println!("Tier: {}", manager.tier());
    /// # }
    /// ```
    pub fn from_adapter(adapter: &Adapter) -> Self {
        let features = adapter.features();
        let limits = adapter.limits();
        let info = adapter.get_info();
        let tier = CapabilityTier::from_features_and_limits(&features, &limits);

        let has_timestamp_queries = features.contains(Features::TIMESTAMP_QUERY);
        let has_push_constants = features.contains(Features::PUSH_CONSTANTS);
        let max_push_constant_size = if has_push_constants {
            limits.max_push_constant_size
        } else {
            0
        };

        debug!(
            "CapabilityManager created for '{}': tier={}, timestamp_queries={}, push_constants={}",
            info.name,
            tier.tier_name(),
            has_timestamp_queries,
            has_push_constants
        );

        Self {
            tier,
            features,
            limits,
            adapter_name: info.name.clone(),
            has_timestamp_queries,
            has_push_constants,
            max_push_constant_size,
        }
    }

    /// Create a CapabilityManager from raw features and limits.
    ///
    /// This is useful for testing or when you already have the features
    /// and limits without an adapter reference.
    ///
    /// # Arguments
    ///
    /// * `features` - The wgpu features
    /// * `limits` - The wgpu limits
    /// * `adapter_name` - Name for reporting
    ///
    /// # Returns
    ///
    /// A new CapabilityManager with the given capabilities.
    pub fn from_features_and_limits(
        features: Features,
        limits: Limits,
        adapter_name: impl Into<String>,
    ) -> Self {
        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        let has_timestamp_queries = features.contains(Features::TIMESTAMP_QUERY);
        let has_push_constants = features.contains(Features::PUSH_CONSTANTS);
        let max_push_constant_size = if has_push_constants {
            limits.max_push_constant_size
        } else {
            0
        };

        Self {
            tier,
            features,
            limits,
            adapter_name: adapter_name.into(),
            has_timestamp_queries,
            has_push_constants,
            max_push_constant_size,
        }
    }

    // ========================================================================
    // Basic Accessors
    // ========================================================================

    /// Get the detected capability tier.
    #[inline]
    pub fn tier(&self) -> CapabilityTier {
        self.tier
    }

    /// Get the cached features.
    #[inline]
    pub fn features(&self) -> &Features {
        &self.features
    }

    /// Get the cached limits.
    #[inline]
    pub fn limits(&self) -> &Limits {
        &self.limits
    }

    /// Get the adapter name.
    #[inline]
    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    // ========================================================================
    // Feature Queries (Required by task)
    // ========================================================================

    /// Check if ray tracing is supported.
    ///
    /// Ray tracing requires hardware ray tracing acceleration structures.
    /// This is only available on Full tier hardware (RTX, RX 6000+, Intel Arc).
    ///
    /// # Returns
    ///
    /// `true` if ray tracing acceleration structures are available.
    #[inline]
    pub fn supports_ray_tracing(&self) -> bool {
        self.features
            .contains(Features::RAY_TRACING_ACCELERATION_STRUCTURE)
    }

    /// Check if bindless rendering is supported.
    ///
    /// Bindless rendering requires texture binding arrays, allowing shaders
    /// to access textures via indices rather than explicit bindings.
    ///
    /// # Returns
    ///
    /// `true` if texture binding arrays are available.
    #[inline]
    pub fn supports_bindless(&self) -> bool {
        self.features.contains(Features::TEXTURE_BINDING_ARRAY)
    }

    /// Check if GPU-driven culling is supported.
    ///
    /// GPU-driven culling requires multi-draw indirect count, which allows
    /// the GPU to determine draw call counts without CPU readback.
    ///
    /// # Returns
    ///
    /// `true` if multi-draw indirect count is available.
    #[inline]
    pub fn supports_gpu_culling(&self) -> bool {
        self.features.contains(Features::MULTI_DRAW_INDIRECT_COUNT)
    }

    /// Check if timestamp queries are supported.
    ///
    /// Timestamp queries allow measuring GPU execution time for profiling
    /// and performance analysis.
    ///
    /// # Returns
    ///
    /// `true` if timestamp queries are available.
    #[inline]
    pub fn supports_timestamp_queries(&self) -> bool {
        self.has_timestamp_queries
    }

    // ========================================================================
    // Selection Methods (Required by task)
    // ========================================================================

    /// Select the optimal render path based on capability tier.
    ///
    /// This method returns the highest-quality render path that the
    /// hardware can support:
    ///
    /// | Tier | Render Path |
    /// |------|-------------|
    /// | Full | RayTraced |
    /// | Advanced | GPUDriven |
    /// | Standard | Traditional |
    /// | Minimal | Fallback |
    ///
    /// # Returns
    ///
    /// The optimal [`RenderPath`] for this hardware.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::device::{CapabilityManager, RenderPath, CapabilityTier};
    /// use wgpu::{Features, Limits};
    ///
    /// // Full tier hardware
    /// let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE;
    /// let limits = Limits::default();
    /// let manager = CapabilityManager::from_features_and_limits(features, limits, "Test GPU");
    ///
    /// assert_eq!(manager.select_render_path(), RenderPath::RayTraced);
    /// ```
    pub fn select_render_path(&self) -> RenderPath {
        match self.tier {
            CapabilityTier::Full => RenderPath::RayTraced,
            CapabilityTier::Advanced => RenderPath::GPUDriven,
            CapabilityTier::Standard => RenderPath::Traditional,
            CapabilityTier::Minimal => RenderPath::Fallback,
        }
    }

    /// Select the best available texture compression format.
    ///
    /// This method checks for hardware support of various texture
    /// compression formats and returns the best available option:
    ///
    /// 1. BC (Block Compression) - desktop standard, best quality
    /// 2. ASTC - mobile/modern, variable quality
    /// 3. ETC2 - mobile fallback
    /// 4. None - no compression available
    ///
    /// # Returns
    ///
    /// The best available [`TextureCompression`] format.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::device::{CapabilityManager, TextureCompression};
    /// use wgpu::{Features, Limits};
    ///
    /// // Desktop GPU with BC support
    /// let features = Features::TEXTURE_COMPRESSION_BC;
    /// let limits = Limits::default();
    /// let manager = CapabilityManager::from_features_and_limits(features, limits, "Desktop GPU");
    ///
    /// assert_eq!(manager.select_texture_compression(), TextureCompression::BC);
    /// ```
    pub fn select_texture_compression(&self) -> TextureCompression {
        // Check in order of preference: BC (desktop) > ASTC (mobile/modern) > ETC2 (fallback)
        if self.features.contains(Features::TEXTURE_COMPRESSION_BC) {
            TextureCompression::BC
        } else if self.features.contains(Features::TEXTURE_COMPRESSION_ASTC) {
            TextureCompression::ASTC
        } else if self.features.contains(Features::TEXTURE_COMPRESSION_ETC2) {
            TextureCompression::ETC2
        } else {
            TextureCompression::None
        }
    }

    /// Get the maximum number of bindless textures supported.
    ///
    /// This returns the maximum number of sampled textures that can be
    /// bound in a single bind group. For bindless rendering, this determines
    /// the texture atlas capacity.
    ///
    /// # Returns
    ///
    /// The maximum number of sampled textures per bind group.
    /// Returns 0 if bindless is not supported (no texture binding arrays).
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::device::CapabilityManager;
    /// use wgpu::{Features, Limits};
    ///
    /// // GPU with bindless support
    /// let features = Features::TEXTURE_BINDING_ARRAY;
    /// let mut limits = Limits::default();
    /// limits.max_sampled_textures_per_shader_stage = 16384;
    ///
    /// let manager = CapabilityManager::from_features_and_limits(features, limits, "Modern GPU");
    /// assert!(manager.max_bindless_textures() > 0);
    /// ```
    pub fn max_bindless_textures(&self) -> u32 {
        if self.supports_bindless() {
            self.limits.max_sampled_textures_per_shader_stage
        } else {
            0
        }
    }

    /// Generate a comprehensive capability report.
    ///
    /// This creates a [`CapabilityReport`] containing all detected
    /// capabilities, suitable for logging or debugging.
    ///
    /// # Returns
    ///
    /// A detailed capability report.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::device::CapabilityManager;
    ///
    /// # async fn example() {
    /// let instance = wgpu::Instance::default();
    /// let adapter = instance
    ///     .request_adapter(&wgpu::RequestAdapterOptions::default())
    ///     .await
    ///     .unwrap();
    ///
    /// let manager = CapabilityManager::from_adapter(&adapter);
    /// let report = manager.report();
    /// println!("{}", report);
    /// # }
    /// ```
    pub fn report(&self) -> CapabilityReport {
        CapabilityReport {
            tier: self.tier,
            has_ray_tracing: self.supports_ray_tracing(),
            has_bindless: self.supports_bindless(),
            has_multi_draw_indirect_count: self.supports_gpu_culling(),
            has_storage_binding_array: self
                .features
                .contains(Features::STORAGE_RESOURCE_BINDING_ARRAY),
            max_texture_dimension_2d: self.limits.max_texture_dimension_2d,
            max_compute_invocations: self.limits.max_compute_invocations_per_workgroup,
            max_storage_buffer_binding_size: self.limits.max_storage_buffer_binding_size,
            adapter_name: self.adapter_name.clone(),
        }
    }

    // ========================================================================
    // Additional Capability Queries
    // ========================================================================

    /// Check if push constants are supported.
    ///
    /// Push constants allow small amounts of data to be sent to shaders
    /// without buffer bindings, reducing overhead for frequently-changed
    /// uniform data.
    ///
    /// # Returns
    ///
    /// `true` if push constants are available.
    #[inline]
    pub fn supports_push_constants(&self) -> bool {
        self.has_push_constants
    }

    /// Get the maximum push constant size in bytes.
    ///
    /// # Returns
    ///
    /// Maximum push constant size, or 0 if not supported.
    #[inline]
    pub fn max_push_constant_size(&self) -> u32 {
        self.max_push_constant_size
    }

    /// Check if storage buffer binding arrays are supported.
    ///
    /// This enables bindless storage buffers in addition to bindless textures.
    ///
    /// # Returns
    ///
    /// `true` if storage resource binding arrays are available.
    #[inline]
    pub fn supports_storage_binding_array(&self) -> bool {
        self.features
            .contains(Features::STORAGE_RESOURCE_BINDING_ARRAY)
    }

    /// Check if indirect first instance is supported.
    ///
    /// This enables the `first_instance` field in indirect draw calls,
    /// useful for GPU-driven rendering with instance culling.
    ///
    /// # Returns
    ///
    /// `true` if indirect first instance is available.
    #[inline]
    pub fn supports_indirect_first_instance(&self) -> bool {
        self.features.contains(Features::INDIRECT_FIRST_INSTANCE)
    }

    /// Get the maximum texture dimension for 2D textures.
    #[inline]
    pub fn max_texture_2d(&self) -> u32 {
        self.limits.max_texture_dimension_2d
    }

    /// Get the maximum compute workgroup invocations.
    #[inline]
    pub fn max_workgroup_invocations(&self) -> u32 {
        self.limits.max_compute_invocations_per_workgroup
    }

    /// Get the maximum storage buffer binding size.
    #[inline]
    pub fn max_storage_buffer_size(&self) -> u32 {
        self.limits.max_storage_buffer_binding_size
    }

    /// Check if the hardware meets a minimum tier requirement.
    ///
    /// # Arguments
    ///
    /// * `required` - The minimum required tier
    ///
    /// # Returns
    ///
    /// `true` if the hardware meets or exceeds the requirement.
    #[inline]
    pub fn meets_tier(&self, required: CapabilityTier) -> bool {
        self.tier >= required
    }

    /// Log capability summary at INFO level.
    pub fn log_capabilities(&self) {
        info!("=== CapabilityManager: {} ===", self.adapter_name);
        info!("  Tier: {}", self.tier.tier_name());
        info!("  Render Path: {}", self.select_render_path());
        info!("  Texture Compression: {}", self.select_texture_compression());
        info!("  Features:");
        info!("    Ray Tracing: {}", self.supports_ray_tracing());
        info!("    Bindless: {}", self.supports_bindless());
        info!("    GPU Culling: {}", self.supports_gpu_culling());
        info!("    Timestamp Queries: {}", self.supports_timestamp_queries());
        info!("    Push Constants: {}", self.supports_push_constants());
        info!("  Limits:");
        info!("    Max Bindless Textures: {}", self.max_bindless_textures());
        info!("    Max Texture 2D: {}", self.max_texture_2d());
        info!(
            "    Max Storage Buffer: {} MB",
            self.max_storage_buffer_size() / (1024 * 1024)
        );
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Detect capability tier from an adapter and log the result.
///
/// Convenience function that detects the tier and logs a summary.
///
/// # Arguments
///
/// * `adapter` - The wgpu adapter to analyze
///
/// # Returns
///
/// The detected capability tier.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::device::detect_capability_tier;
///
/// # async fn example() {
/// let instance = wgpu::Instance::default();
/// let adapter = instance
///     .request_adapter(&wgpu::RequestAdapterOptions::default())
///     .await
///     .unwrap();
///
/// let tier = detect_capability_tier(&adapter);
/// // Logs: "Adapter 'GPU Name' detected as Advanced tier"
/// # }
/// ```
pub fn detect_capability_tier(adapter: &Adapter) -> CapabilityTier {
    let tier = CapabilityTier::from_adapter(adapter);
    let name = adapter.get_info().name;
    info!(
        "Adapter '{}' detected as {} tier",
        name,
        tier.tier_name()
    );
    tier
}

/// Get the minimum wgpu features required for a given tier.
///
/// Returns the features that must be enabled to achieve the specified tier.
/// Note that this returns the features only, not the limits requirements.
///
/// # Arguments
///
/// * `tier` - The target capability tier
///
/// # Returns
///
/// The minimum wgpu features required.
pub fn features_for_tier(tier: CapabilityTier) -> Features {
    match tier {
        CapabilityTier::Full => Features::RAY_TRACING_ACCELERATION_STRUCTURE,
        CapabilityTier::Advanced => {
            Features::TEXTURE_BINDING_ARRAY | Features::MULTI_DRAW_INDIRECT_COUNT
        }
        CapabilityTier::Standard | CapabilityTier::Minimal => Features::empty(),
    }
}

/// Check if the given features and limits can achieve a target tier.
///
/// # Arguments
///
/// * `features` - The available wgpu features
/// * `limits` - The available wgpu limits
/// * `target` - The target capability tier
///
/// # Returns
///
/// `true` if the target tier can be achieved.
pub fn can_achieve_tier(features: &Features, limits: &Limits, target: CapabilityTier) -> bool {
    let actual = CapabilityTier::from_features_and_limits(features, limits);
    actual >= target
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // CapabilityTier Basic Tests
    // ========================================================================

    #[test]
    fn test_capability_tier_default() {
        let tier = CapabilityTier::default();
        assert_eq!(tier, CapabilityTier::Minimal);
    }

    #[test]
    fn test_capability_tier_ordering() {
        assert!(CapabilityTier::Minimal < CapabilityTier::Standard);
        assert!(CapabilityTier::Standard < CapabilityTier::Advanced);
        assert!(CapabilityTier::Advanced < CapabilityTier::Full);

        assert!(CapabilityTier::Full > CapabilityTier::Advanced);
        assert!(CapabilityTier::Full > CapabilityTier::Standard);
        assert!(CapabilityTier::Full > CapabilityTier::Minimal);
    }

    #[test]
    fn test_capability_tier_equality() {
        assert_eq!(CapabilityTier::Full, CapabilityTier::Full);
        assert_ne!(CapabilityTier::Full, CapabilityTier::Advanced);
    }

    #[test]
    fn test_capability_tier_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(CapabilityTier::Minimal);
        set.insert(CapabilityTier::Standard);
        set.insert(CapabilityTier::Advanced);
        set.insert(CapabilityTier::Full);
        assert_eq!(set.len(), 4);
    }

    #[test]
    fn test_capability_tier_names() {
        assert_eq!(CapabilityTier::Full.tier_name(), "Full");
        assert_eq!(CapabilityTier::Advanced.tier_name(), "Advanced");
        assert_eq!(CapabilityTier::Standard.tier_name(), "Standard");
        assert_eq!(CapabilityTier::Minimal.tier_name(), "Minimal");
    }

    #[test]
    fn test_capability_tier_descriptions() {
        assert!(!CapabilityTier::Full.description().is_empty());
        assert!(!CapabilityTier::Advanced.description().is_empty());
        assert!(!CapabilityTier::Standard.description().is_empty());
        assert!(!CapabilityTier::Minimal.description().is_empty());

        assert!(CapabilityTier::Full.description().contains("Ray tracing"));
        assert!(CapabilityTier::Advanced.description().contains("Bindless"));
    }

    #[test]
    fn test_capability_tier_display() {
        assert_eq!(format!("{}", CapabilityTier::Full), "Full");
        assert_eq!(format!("{}", CapabilityTier::Advanced), "Advanced");
        assert_eq!(format!("{}", CapabilityTier::Standard), "Standard");
        assert_eq!(format!("{}", CapabilityTier::Minimal), "Minimal");
    }

    // ========================================================================
    // Capability Query Tests
    // ========================================================================

    #[test]
    fn test_supports_ray_tracing() {
        assert!(CapabilityTier::Full.supports_ray_tracing());
        assert!(!CapabilityTier::Advanced.supports_ray_tracing());
        assert!(!CapabilityTier::Standard.supports_ray_tracing());
        assert!(!CapabilityTier::Minimal.supports_ray_tracing());
    }

    #[test]
    fn test_supports_bindless() {
        assert!(CapabilityTier::Full.supports_bindless());
        assert!(CapabilityTier::Advanced.supports_bindless());
        assert!(!CapabilityTier::Standard.supports_bindless());
        assert!(!CapabilityTier::Minimal.supports_bindless());
    }

    #[test]
    fn test_supports_gpu_driven() {
        assert!(CapabilityTier::Full.supports_gpu_driven());
        assert!(CapabilityTier::Advanced.supports_gpu_driven());
        assert!(!CapabilityTier::Standard.supports_gpu_driven());
        assert!(!CapabilityTier::Minimal.supports_gpu_driven());
    }

    #[test]
    fn test_supports_8k_textures() {
        assert!(CapabilityTier::Full.supports_8k_textures());
        assert!(CapabilityTier::Advanced.supports_8k_textures());
        assert!(CapabilityTier::Standard.supports_8k_textures());
        assert!(!CapabilityTier::Minimal.supports_8k_textures());
    }

    #[test]
    fn test_supports_compute() {
        assert!(CapabilityTier::Full.supports_compute());
        assert!(CapabilityTier::Advanced.supports_compute());
        assert!(CapabilityTier::Standard.supports_compute());
        assert!(!CapabilityTier::Minimal.supports_compute());
    }

    #[test]
    fn test_meets_requirement() {
        let full = CapabilityTier::Full;
        assert!(full.meets_requirement(CapabilityTier::Minimal));
        assert!(full.meets_requirement(CapabilityTier::Standard));
        assert!(full.meets_requirement(CapabilityTier::Advanced));
        assert!(full.meets_requirement(CapabilityTier::Full));

        let standard = CapabilityTier::Standard;
        assert!(standard.meets_requirement(CapabilityTier::Minimal));
        assert!(standard.meets_requirement(CapabilityTier::Standard));
        assert!(!standard.meets_requirement(CapabilityTier::Advanced));
        assert!(!standard.meets_requirement(CapabilityTier::Full));

        let minimal = CapabilityTier::Minimal;
        assert!(minimal.meets_requirement(CapabilityTier::Minimal));
        assert!(!minimal.meets_requirement(CapabilityTier::Standard));
    }

    // ========================================================================
    // Tier Detection Tests
    // ========================================================================

    #[test]
    fn test_detect_minimal_tier() {
        let features = Features::empty();
        let limits = Limits::downlevel_webgl2_defaults();

        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        assert_eq!(tier, CapabilityTier::Minimal);
    }

    #[test]
    fn test_detect_standard_tier() {
        let features = Features::empty();
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = 8192;
        limits.max_storage_buffer_binding_size = 128 * 1024 * 1024;

        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        assert_eq!(tier, CapabilityTier::Standard);
    }

    #[test]
    fn test_detect_advanced_tier() {
        let features =
            Features::TEXTURE_BINDING_ARRAY | Features::MULTI_DRAW_INDIRECT_COUNT;
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = 16384;
        limits.max_compute_invocations_per_workgroup = 1024;
        limits.max_storage_buffer_binding_size = 256 * 1024 * 1024;

        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        assert_eq!(tier, CapabilityTier::Advanced);
    }

    #[test]
    fn test_detect_full_tier() {
        let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE;
        let limits = Limits::default();

        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        assert_eq!(tier, CapabilityTier::Full);
    }

    #[test]
    fn test_full_tier_overrides_advanced() {
        // Even with all advanced features, RT should result in Full tier
        let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE
            | Features::TEXTURE_BINDING_ARRAY
            | Features::MULTI_DRAW_INDIRECT_COUNT;
        let mut limits = Limits::default();
        limits.max_compute_invocations_per_workgroup = 1024;

        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        assert_eq!(tier, CapabilityTier::Full);
    }

    #[test]
    fn test_advanced_needs_all_features() {
        // Only bindless - should fall to Standard or Minimal
        let features = Features::TEXTURE_BINDING_ARRAY;
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = 16384;
        limits.max_compute_invocations_per_workgroup = 1024;
        limits.max_storage_buffer_binding_size = 256 * 1024 * 1024;

        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        assert_ne!(tier, CapabilityTier::Advanced);
        // Should be Standard since we have 8K textures and storage
        assert_eq!(tier, CapabilityTier::Standard);
    }

    #[test]
    fn test_advanced_needs_large_workgroup() {
        // Has features but small workgroup
        let features =
            Features::TEXTURE_BINDING_ARRAY | Features::MULTI_DRAW_INDIRECT_COUNT;
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = 16384;
        limits.max_compute_invocations_per_workgroup = 256; // Too small
        limits.max_storage_buffer_binding_size = 256 * 1024 * 1024;

        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        // Should fall to Standard
        assert_eq!(tier, CapabilityTier::Standard);
    }

    #[test]
    fn test_standard_needs_8k_textures() {
        let features = Features::empty();
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = 4096; // Too small
        limits.max_storage_buffer_binding_size = 256 * 1024 * 1024;

        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        assert_eq!(tier, CapabilityTier::Minimal);
    }

    #[test]
    fn test_standard_needs_storage_buffer() {
        let features = Features::empty();
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = 8192;
        limits.max_storage_buffer_binding_size = 64 * 1024 * 1024; // Too small

        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        assert_eq!(tier, CapabilityTier::Minimal);
    }

    // ========================================================================
    // Utility Function Tests
    // ========================================================================

    #[test]
    fn test_features_for_tier() {
        let full_features = features_for_tier(CapabilityTier::Full);
        assert!(full_features.contains(Features::RAY_TRACING_ACCELERATION_STRUCTURE));

        let advanced_features = features_for_tier(CapabilityTier::Advanced);
        assert!(advanced_features.contains(Features::TEXTURE_BINDING_ARRAY));
        assert!(advanced_features.contains(Features::MULTI_DRAW_INDIRECT_COUNT));

        let standard_features = features_for_tier(CapabilityTier::Standard);
        assert_eq!(standard_features, Features::empty());

        let minimal_features = features_for_tier(CapabilityTier::Minimal);
        assert_eq!(minimal_features, Features::empty());
    }

    #[test]
    fn test_can_achieve_tier() {
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = 8192;
        limits.max_storage_buffer_binding_size = 128 * 1024 * 1024;

        // Empty features can achieve Minimal and Standard
        let empty_features = Features::empty();
        assert!(can_achieve_tier(&empty_features, &limits, CapabilityTier::Minimal));
        assert!(can_achieve_tier(&empty_features, &limits, CapabilityTier::Standard));
        assert!(!can_achieve_tier(&empty_features, &limits, CapabilityTier::Advanced));
        assert!(!can_achieve_tier(&empty_features, &limits, CapabilityTier::Full));

        // RT features can achieve all tiers
        let rt_features = Features::RAY_TRACING_ACCELERATION_STRUCTURE;
        assert!(can_achieve_tier(&rt_features, &limits, CapabilityTier::Minimal));
        assert!(can_achieve_tier(&rt_features, &limits, CapabilityTier::Standard));
        assert!(can_achieve_tier(&rt_features, &limits, CapabilityTier::Advanced));
        assert!(can_achieve_tier(&rt_features, &limits, CapabilityTier::Full));
    }

    // ========================================================================
    // CapabilityReport Tests
    // ========================================================================

    #[test]
    fn test_capability_report_display() {
        let report = CapabilityReport {
            tier: CapabilityTier::Advanced,
            has_ray_tracing: false,
            has_bindless: true,
            has_multi_draw_indirect_count: true,
            has_storage_binding_array: true,
            max_texture_dimension_2d: 16384,
            max_compute_invocations: 1024,
            max_storage_buffer_binding_size: 256 * 1024 * 1024,
            adapter_name: "Test GPU".to_string(),
        };

        let display = format!("{}", report);
        assert!(display.contains("Test GPU"));
        assert!(display.contains("Advanced"));
        assert!(display.contains("16384"));
        assert!(display.contains("1024"));
    }

    #[test]
    fn test_capability_report_clone() {
        let report = CapabilityReport {
            tier: CapabilityTier::Standard,
            has_ray_tracing: false,
            has_bindless: false,
            has_multi_draw_indirect_count: false,
            has_storage_binding_array: false,
            max_texture_dimension_2d: 8192,
            max_compute_invocations: 256,
            max_storage_buffer_binding_size: 128 * 1024 * 1024,
            adapter_name: "Clone Test".to_string(),
        };

        let cloned = report.clone();
        assert_eq!(cloned.tier, report.tier);
        assert_eq!(cloned.adapter_name, report.adapter_name);
        assert_eq!(cloned.max_texture_dimension_2d, report.max_texture_dimension_2d);
    }

    // ========================================================================
    // RenderPath Tests
    // ========================================================================

    #[test]
    fn test_render_path_names() {
        assert_eq!(RenderPath::RayTraced.name(), "RayTraced");
        assert_eq!(RenderPath::GPUDriven.name(), "GPUDriven");
        assert_eq!(RenderPath::Traditional.name(), "Traditional");
        assert_eq!(RenderPath::Fallback.name(), "Fallback");
    }

    #[test]
    fn test_render_path_descriptions() {
        assert!(!RenderPath::RayTraced.description().is_empty());
        assert!(!RenderPath::GPUDriven.description().is_empty());
        assert!(!RenderPath::Traditional.description().is_empty());
        assert!(!RenderPath::Fallback.description().is_empty());
    }

    #[test]
    fn test_render_path_display() {
        assert_eq!(format!("{}", RenderPath::RayTraced), "RayTraced");
        assert_eq!(format!("{}", RenderPath::GPUDriven), "GPUDriven");
    }

    // ========================================================================
    // TextureCompression Tests
    // ========================================================================

    #[test]
    fn test_texture_compression_names() {
        assert_eq!(TextureCompression::BC.name(), "BC");
        assert_eq!(TextureCompression::ASTC.name(), "ASTC");
        assert_eq!(TextureCompression::ETC2.name(), "ETC2");
        assert_eq!(TextureCompression::None.name(), "None");
    }

    #[test]
    fn test_texture_compression_hardware_accelerated() {
        assert!(TextureCompression::BC.is_hardware_accelerated());
        assert!(TextureCompression::ASTC.is_hardware_accelerated());
        assert!(TextureCompression::ETC2.is_hardware_accelerated());
        assert!(!TextureCompression::None.is_hardware_accelerated());
    }

    #[test]
    fn test_texture_compression_display() {
        assert_eq!(format!("{}", TextureCompression::BC), "BC");
        assert_eq!(format!("{}", TextureCompression::None), "None");
    }

    // ========================================================================
    // CapabilityManager Tests
    // ========================================================================

    #[test]
    fn test_capability_manager_from_features_and_limits_minimal() {
        let features = Features::empty();
        let limits = Limits::downlevel_webgl2_defaults();

        let manager =
            CapabilityManager::from_features_and_limits(features, limits, "Test Minimal GPU");

        assert_eq!(manager.tier(), CapabilityTier::Minimal);
        assert_eq!(manager.adapter_name(), "Test Minimal GPU");
        assert!(!manager.supports_ray_tracing());
        assert!(!manager.supports_bindless());
        assert!(!manager.supports_gpu_culling());
        assert!(!manager.supports_timestamp_queries());
    }

    #[test]
    fn test_capability_manager_from_features_and_limits_standard() {
        let features = Features::empty();
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = 8192;
        limits.max_storage_buffer_binding_size = 128 * 1024 * 1024;

        let manager =
            CapabilityManager::from_features_and_limits(features, limits, "Test Standard GPU");

        assert_eq!(manager.tier(), CapabilityTier::Standard);
        assert!(!manager.supports_ray_tracing());
        assert!(!manager.supports_bindless());
        assert!(!manager.supports_gpu_culling());
    }

    #[test]
    fn test_capability_manager_from_features_and_limits_advanced() {
        let features =
            Features::TEXTURE_BINDING_ARRAY | Features::MULTI_DRAW_INDIRECT_COUNT;
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = 16384;
        limits.max_compute_invocations_per_workgroup = 1024;
        limits.max_storage_buffer_binding_size = 256 * 1024 * 1024;
        limits.max_sampled_textures_per_shader_stage = 16384;

        let manager =
            CapabilityManager::from_features_and_limits(features, limits, "Test Advanced GPU");

        assert_eq!(manager.tier(), CapabilityTier::Advanced);
        assert!(!manager.supports_ray_tracing());
        assert!(manager.supports_bindless());
        assert!(manager.supports_gpu_culling());
        assert_eq!(manager.max_bindless_textures(), 16384);
    }

    #[test]
    fn test_capability_manager_from_features_and_limits_full() {
        let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE
            | Features::TEXTURE_BINDING_ARRAY
            | Features::MULTI_DRAW_INDIRECT_COUNT
            | Features::TIMESTAMP_QUERY;
        let mut limits = Limits::default();
        limits.max_compute_invocations_per_workgroup = 1024;

        let manager =
            CapabilityManager::from_features_and_limits(features, limits, "Test Full GPU");

        assert_eq!(manager.tier(), CapabilityTier::Full);
        assert!(manager.supports_ray_tracing());
        assert!(manager.supports_bindless());
        assert!(manager.supports_gpu_culling());
        assert!(manager.supports_timestamp_queries());
    }

    #[test]
    fn test_capability_manager_select_render_path() {
        // Full tier
        let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE;
        let limits = Limits::default();
        let manager =
            CapabilityManager::from_features_and_limits(features, limits, "Full GPU");
        assert_eq!(manager.select_render_path(), RenderPath::RayTraced);

        // Advanced tier
        let features =
            Features::TEXTURE_BINDING_ARRAY | Features::MULTI_DRAW_INDIRECT_COUNT;
        let mut limits = Limits::default();
        limits.max_compute_invocations_per_workgroup = 1024;
        limits.max_texture_dimension_2d = 8192;
        limits.max_storage_buffer_binding_size = 128 * 1024 * 1024;
        let manager =
            CapabilityManager::from_features_and_limits(features, limits, "Advanced GPU");
        assert_eq!(manager.select_render_path(), RenderPath::GPUDriven);

        // Standard tier
        let features = Features::empty();
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = 8192;
        limits.max_storage_buffer_binding_size = 128 * 1024 * 1024;
        let manager =
            CapabilityManager::from_features_and_limits(features, limits, "Standard GPU");
        assert_eq!(manager.select_render_path(), RenderPath::Traditional);

        // Minimal tier
        let features = Features::empty();
        let limits = Limits::downlevel_webgl2_defaults();
        let manager =
            CapabilityManager::from_features_and_limits(features, limits, "Minimal GPU");
        assert_eq!(manager.select_render_path(), RenderPath::Fallback);
    }

    #[test]
    fn test_capability_manager_select_texture_compression() {
        // BC compression (desktop)
        let features = Features::TEXTURE_COMPRESSION_BC;
        let limits = Limits::default();
        let manager =
            CapabilityManager::from_features_and_limits(features, limits, "Desktop GPU");
        assert_eq!(manager.select_texture_compression(), TextureCompression::BC);

        // ASTC compression (mobile/modern)
        let features = Features::TEXTURE_COMPRESSION_ASTC;
        let limits = Limits::default();
        let manager =
            CapabilityManager::from_features_and_limits(features, limits, "Mobile GPU");
        assert_eq!(manager.select_texture_compression(), TextureCompression::ASTC);

        // ETC2 compression (fallback)
        let features = Features::TEXTURE_COMPRESSION_ETC2;
        let limits = Limits::default();
        let manager =
            CapabilityManager::from_features_and_limits(features, limits, "ETC2 GPU");
        assert_eq!(manager.select_texture_compression(), TextureCompression::ETC2);

        // No compression
        let features = Features::empty();
        let limits = Limits::default();
        let manager =
            CapabilityManager::from_features_and_limits(features, limits, "Basic GPU");
        assert_eq!(manager.select_texture_compression(), TextureCompression::None);

        // BC preferred over ASTC
        let features = Features::TEXTURE_COMPRESSION_BC | Features::TEXTURE_COMPRESSION_ASTC;
        let limits = Limits::default();
        let manager =
            CapabilityManager::from_features_and_limits(features, limits, "Multi GPU");
        assert_eq!(manager.select_texture_compression(), TextureCompression::BC);
    }

    #[test]
    fn test_capability_manager_max_bindless_textures() {
        // With bindless support
        let features = Features::TEXTURE_BINDING_ARRAY;
        let mut limits = Limits::default();
        limits.max_sampled_textures_per_shader_stage = 16384;
        let manager =
            CapabilityManager::from_features_and_limits(features, limits, "Bindless GPU");
        assert_eq!(manager.max_bindless_textures(), 16384);

        // Without bindless support
        let features = Features::empty();
        let mut limits = Limits::default();
        limits.max_sampled_textures_per_shader_stage = 16384;
        let manager =
            CapabilityManager::from_features_and_limits(features, limits, "No Bindless GPU");
        assert_eq!(manager.max_bindless_textures(), 0);
    }

    #[test]
    fn test_capability_manager_report() {
        let features = Features::TEXTURE_BINDING_ARRAY
            | Features::MULTI_DRAW_INDIRECT_COUNT
            | Features::STORAGE_RESOURCE_BINDING_ARRAY;
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = 16384;
        limits.max_compute_invocations_per_workgroup = 1024;
        limits.max_storage_buffer_binding_size = 256 * 1024 * 1024;

        let manager =
            CapabilityManager::from_features_and_limits(features, limits, "Report Test GPU");

        let report = manager.report();

        assert_eq!(report.tier, CapabilityTier::Advanced);
        assert_eq!(report.adapter_name, "Report Test GPU");
        assert!(!report.has_ray_tracing);
        assert!(report.has_bindless);
        assert!(report.has_multi_draw_indirect_count);
        assert!(report.has_storage_binding_array);
        assert_eq!(report.max_texture_dimension_2d, 16384);
        assert_eq!(report.max_compute_invocations, 1024);
    }

    #[test]
    fn test_capability_manager_meets_tier() {
        let features = Features::TEXTURE_BINDING_ARRAY | Features::MULTI_DRAW_INDIRECT_COUNT;
        let mut limits = Limits::default();
        limits.max_compute_invocations_per_workgroup = 1024;
        limits.max_texture_dimension_2d = 8192;
        limits.max_storage_buffer_binding_size = 128 * 1024 * 1024;

        let manager =
            CapabilityManager::from_features_and_limits(features, limits, "Advanced GPU");

        assert!(manager.meets_tier(CapabilityTier::Minimal));
        assert!(manager.meets_tier(CapabilityTier::Standard));
        assert!(manager.meets_tier(CapabilityTier::Advanced));
        assert!(!manager.meets_tier(CapabilityTier::Full));
    }

    #[test]
    fn test_capability_manager_push_constants() {
        // With push constants
        let features = Features::PUSH_CONSTANTS;
        let mut limits = Limits::default();
        limits.max_push_constant_size = 128;
        let manager =
            CapabilityManager::from_features_and_limits(features, limits, "Push Constants GPU");
        assert!(manager.supports_push_constants());
        assert_eq!(manager.max_push_constant_size(), 128);

        // Without push constants
        let features = Features::empty();
        let limits = Limits::default();
        let manager =
            CapabilityManager::from_features_and_limits(features, limits, "No Push Constants GPU");
        assert!(!manager.supports_push_constants());
        assert_eq!(manager.max_push_constant_size(), 0);
    }

    #[test]
    fn test_capability_manager_clone() {
        let features = Features::TEXTURE_BINDING_ARRAY;
        let limits = Limits::default();
        let manager =
            CapabilityManager::from_features_and_limits(features, limits, "Clone Test GPU");

        let cloned = manager.clone();
        assert_eq!(cloned.tier(), manager.tier());
        assert_eq!(cloned.adapter_name(), manager.adapter_name());
        assert_eq!(cloned.supports_bindless(), manager.supports_bindless());
    }

    // ========================================================================
    // Integration Tests
    // ========================================================================

    #[cfg(not(feature = "ci"))]
    mod integration {
        use super::*;

        #[test]
        fn test_from_adapter_real_hardware() {
            let instance = wgpu::Instance::default();
            let adapters: Vec<wgpu::Adapter> =
                instance.enumerate_adapters(wgpu::Backends::PRIMARY);

            if adapters.is_empty() {
                println!("No adapters found, skipping test");
                return;
            }

            for adapter in &adapters {
                let tier = CapabilityTier::from_adapter(adapter);
                let name = adapter.get_info().name;

                println!("Adapter '{}' -> {} tier", name, tier.tier_name());

                // Tier should be valid (one of the four tiers)
                assert!(
                    matches!(
                        tier,
                        CapabilityTier::Minimal
                            | CapabilityTier::Standard
                            | CapabilityTier::Advanced
                            | CapabilityTier::Full
                    )
                );
            }
        }

        #[test]
        fn test_capability_report_from_adapter() {
            let instance = wgpu::Instance::default();
            let adapters: Vec<wgpu::Adapter> =
                instance.enumerate_adapters(wgpu::Backends::PRIMARY);

            if adapters.is_empty() {
                println!("No adapters found, skipping test");
                return;
            }

            let report = CapabilityReport::from_adapter(&adapters[0]);
            println!("{}", report);

            // Report should be consistent with tier
            assert_eq!(report.tier.supports_ray_tracing(), report.has_ray_tracing);

            if report.tier == CapabilityTier::Full {
                assert!(report.has_ray_tracing);
            }
        }

        #[test]
        fn test_detect_capability_tier_logging() {
            let instance = wgpu::Instance::default();
            let adapters: Vec<wgpu::Adapter> =
                instance.enumerate_adapters(wgpu::Backends::PRIMARY);

            if adapters.is_empty() {
                println!("No adapters found, skipping test");
                return;
            }

            // This should log the tier detection
            let tier = detect_capability_tier(&adapters[0]);
            assert!(tier >= CapabilityTier::Minimal);
        }
    }
}
