//! Metal-specific feature detection for TRINITY.
//!
//! This module provides detection of Metal GPU features and capabilities that
//! go beyond what wgpu exposes directly. Metal's feature set varies significantly
//! based on GPU family, and this module helps identify available functionality
//! for optimal rendering paths.
//!
//! # Metal GPU Families
//!
//! Apple organizes Metal capabilities into GPU families:
//!
//! | Family | Hardware | Key Features |
//! |--------|----------|--------------|
//! | Apple1-9 | A-series (iPhone/iPad) | Varies by generation |
//! | Mac1-2 | Intel Mac | Limited feature set |
//! | Common1-3 | Cross-platform | Shared baseline |
//! | Metal3 | M1+, A14+ | RT, mesh shaders, bindless |
//!
//! # Apple Silicon Generations
//!
//! | Generation | Chips | Notable Features |
//! |------------|-------|------------------|
//! | M1 | M1, M1 Pro/Max/Ultra | Metal 3, RT, mesh shaders |
//! | M2 | M2, M2 Pro/Max/Ultra | Improved RT, better cache |
//! | M3 | M3, M3 Pro/Max | Dynamic caching, hardware RT |
//! | M4 | M4, M4 Pro/Max | Enhanced neural engine |
//!
//! # Example
//!
//! ```no_run
//! use renderer_backend::backend::metal::{MetalFeatures, MetalGpuFamily};
//!
//! # async fn example() {
//! let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
//!     backends: wgpu::Backends::METAL,
//!     ..Default::default()
//! });
//!
//! let adapter = instance
//!     .request_adapter(&wgpu::RequestAdapterOptions::default())
//!     .await
//!     .unwrap();
//!
//! let metal_features = MetalFeatures::detect(&adapter);
//!
//! if metal_features.supports_rt() {
//!     println!("Metal ray tracing available!");
//! }
//!
//! if metal_features.is_m_series() {
//!     println!("Running on Apple Silicon!");
//! }
//! # }
//! ```

use log::debug;
use std::cmp::Ordering;
use wgpu::{Adapter, Features};

// ============================================================================
// MetalVersion
// ============================================================================

/// Metal API version.
///
/// Represents a specific version of the Metal API. Metal versions are tied to
/// macOS/iOS releases and determine which features are available.
///
/// # Version History
///
/// | Version | macOS | iOS | Key Features |
/// |---------|-------|-----|--------------|
/// | 2.0 | 10.13 | 11.0 | Argument buffers |
/// | 2.3 | 10.15 | 13.0 | Ray tracing basics |
/// | 2.4 | 11.0 | 14.0 | Apple Silicon support |
/// | 3.0 | 13.0 | 16.0 | Full ray tracing, mesh shaders |
/// | 3.1 | 14.0 | 17.0 | Hardware ray tracing |
///
/// # Example
///
/// ```
/// use renderer_backend::backend::metal::MetalVersion;
///
/// let v3 = MetalVersion::V3_0;
/// let v2 = MetalVersion::V2_4;
///
/// assert!(v3 > v2);
/// assert_eq!(v3.major, 3);
/// assert_eq!(v3.minor, 0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MetalVersion {
    /// Major version number.
    pub major: u32,
    /// Minor version number.
    pub minor: u32,
}

impl MetalVersion {
    /// Metal 2.0 (macOS 10.13, iOS 11.0).
    pub const V2_0: Self = Self { major: 2, minor: 0 };

    /// Metal 2.3 (macOS 10.15, iOS 13.0).
    pub const V2_3: Self = Self { major: 2, minor: 3 };

    /// Metal 2.4 (macOS 11.0, iOS 14.0).
    pub const V2_4: Self = Self { major: 2, minor: 4 };

    /// Metal 3.0 (macOS 13.0, iOS 16.0).
    pub const V3_0: Self = Self { major: 3, minor: 0 };

    /// Metal 3.1 (macOS 14.0, iOS 17.0).
    pub const V3_1: Self = Self { major: 3, minor: 1 };

    /// Create a new Metal version.
    ///
    /// # Arguments
    ///
    /// * `major` - Major version number
    /// * `minor` - Minor version number
    ///
    /// # Returns
    ///
    /// A new `MetalVersion` instance.
    #[inline]
    pub const fn new(major: u32, minor: u32) -> Self {
        Self { major, minor }
    }

    /// Check if this version supports ray tracing.
    ///
    /// Ray tracing requires Metal 3.0+.
    #[inline]
    pub const fn supports_ray_tracing(&self) -> bool {
        self.major >= 3
    }

    /// Check if this version supports mesh shaders.
    ///
    /// Mesh shaders require Metal 3.0+.
    #[inline]
    pub const fn supports_mesh_shaders(&self) -> bool {
        self.major >= 3
    }

    /// Check if this version supports argument buffers tier 2.
    ///
    /// Argument buffers tier 2 (bindless) requires Metal 2.0+.
    #[inline]
    pub const fn supports_argument_buffers_tier2(&self) -> bool {
        self.major >= 2
    }
}

impl Default for MetalVersion {
    fn default() -> Self {
        Self::V2_0
    }
}

impl PartialOrd for MetalVersion {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MetalVersion {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.major.cmp(&other.major) {
            Ordering::Equal => self.minor.cmp(&other.minor),
            other => other,
        }
    }
}

impl std::fmt::Display for MetalVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Metal {}.{}", self.major, self.minor)
    }
}

// ============================================================================
// AppleGpuFamily
// ============================================================================

/// Apple GPU family classification (hardware capability tiers).
///
/// This enum represents Apple's official GPU family classifications which
/// determine what Metal features are available on a given device.
///
/// # Ordering
///
/// Families are ordered by capability level. Higher families have more features.
/// Apple families (Apple1-Apple9) represent A-series and M-series chip generations.
/// Common families represent cross-platform baselines.
/// Mac families represent Intel Mac GPU capabilities.
///
/// # Example
///
/// ```
/// use renderer_backend::backend::metal::AppleGpuFamily;
///
/// let family = AppleGpuFamily::Apple7;
/// assert!(family.supports_ray_tracing());
/// assert!(family.supports_mesh_shaders());
/// assert!(family >= AppleGpuFamily::Apple1);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AppleGpuFamily {
    /// Apple GPU family 1 (A7, A8).
    ///
    /// First Metal-capable chips with basic feature set.
    Apple1,

    /// Apple GPU family 2 (A8X).
    ///
    /// Enhanced A8 with more GPU cores.
    Apple2,

    /// Apple GPU family 3 (A9, A10).
    ///
    /// Adds tessellation, resource heaps, SIMD groups.
    Apple3,

    /// Apple GPU family 4 (A11).
    ///
    /// Adds tile shaders, imageblock, raster order groups.
    Apple4,

    /// Apple GPU family 5 (A12).
    ///
    /// Adds sparse textures, depth/stencil resolve.
    Apple5,

    /// Apple GPU family 6 (A13).
    ///
    /// Adds ASTC HDR textures.
    Apple6,

    /// Apple GPU family 7 (A14, M1).
    ///
    /// First family with ray tracing and mesh shaders.
    Apple7,

    /// Apple GPU family 8 (A15, M2).
    ///
    /// Improved ray tracing performance and efficiency.
    Apple8,

    /// Apple GPU family 9 (A17, M3).
    ///
    /// Hardware-accelerated ray tracing, dynamic caching.
    Apple9,

    /// Common GPU family 1.
    ///
    /// Baseline cross-platform features.
    Common1,

    /// Common GPU family 2.
    ///
    /// Enhanced cross-platform features.
    Common2,

    /// Common GPU family 3.
    ///
    /// Modern cross-platform features (comparable to Vulkan 1.1).
    Common3,

    /// Mac GPU family 1 (older Intel Mac).
    ///
    /// Basic Metal support for Intel integrated GPUs.
    Mac1,

    /// Mac GPU family 2 (newer Intel Mac).
    ///
    /// Enhanced Metal support for Intel/AMD discrete GPUs.
    Mac2,
}

impl AppleGpuFamily {
    /// Check if this family supports ray tracing.
    ///
    /// Ray tracing requires Apple7+ (A14/M1 or later).
    ///
    /// # Returns
    ///
    /// `true` if ray tracing is supported.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::metal::AppleGpuFamily;
    ///
    /// assert!(AppleGpuFamily::Apple7.supports_ray_tracing());
    /// assert!(AppleGpuFamily::Apple9.supports_ray_tracing());
    /// assert!(!AppleGpuFamily::Apple6.supports_ray_tracing());
    /// ```
    #[inline]
    pub const fn supports_ray_tracing(&self) -> bool {
        matches!(
            self,
            AppleGpuFamily::Apple7 | AppleGpuFamily::Apple8 | AppleGpuFamily::Apple9
        )
    }

    /// Check if this family supports mesh shaders.
    ///
    /// Mesh shaders (object/mesh functions) require Apple7+ (A14/M1 or later).
    ///
    /// # Returns
    ///
    /// `true` if mesh shaders are supported.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::metal::AppleGpuFamily;
    ///
    /// assert!(AppleGpuFamily::Apple7.supports_mesh_shaders());
    /// assert!(!AppleGpuFamily::Apple6.supports_mesh_shaders());
    /// ```
    #[inline]
    pub const fn supports_mesh_shaders(&self) -> bool {
        matches!(
            self,
            AppleGpuFamily::Apple7 | AppleGpuFamily::Apple8 | AppleGpuFamily::Apple9
        )
    }

    /// Check if this family supports SIMD-group operations.
    ///
    /// SIMD-group (subgroup/wave) operations require Apple3+.
    ///
    /// # Returns
    ///
    /// `true` if SIMD-group operations are supported.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::metal::AppleGpuFamily;
    ///
    /// assert!(AppleGpuFamily::Apple3.supports_simd_groups());
    /// assert!(AppleGpuFamily::Apple7.supports_simd_groups());
    /// assert!(!AppleGpuFamily::Apple2.supports_simd_groups());
    /// ```
    #[inline]
    pub const fn supports_simd_groups(&self) -> bool {
        matches!(
            self,
            AppleGpuFamily::Apple3
                | AppleGpuFamily::Apple4
                | AppleGpuFamily::Apple5
                | AppleGpuFamily::Apple6
                | AppleGpuFamily::Apple7
                | AppleGpuFamily::Apple8
                | AppleGpuFamily::Apple9
                | AppleGpuFamily::Mac1
                | AppleGpuFamily::Mac2
        )
    }

    /// Get the minimum Metal version required for this GPU family.
    ///
    /// # Returns
    ///
    /// The minimum `MetalVersion` required.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::metal::{AppleGpuFamily, MetalVersion};
    ///
    /// assert_eq!(AppleGpuFamily::Apple7.min_metal_version(), MetalVersion::V3_0);
    /// assert_eq!(AppleGpuFamily::Apple3.min_metal_version(), MetalVersion::V2_0);
    /// ```
    #[inline]
    pub const fn min_metal_version(&self) -> MetalVersion {
        match self {
            AppleGpuFamily::Apple9 => MetalVersion::V3_1,
            AppleGpuFamily::Apple7 | AppleGpuFamily::Apple8 => MetalVersion::V3_0,
            AppleGpuFamily::Apple5 | AppleGpuFamily::Apple6 => MetalVersion::V2_3,
            AppleGpuFamily::Apple4 => MetalVersion::V2_0,
            AppleGpuFamily::Apple3 => MetalVersion::V2_0,
            AppleGpuFamily::Apple1 | AppleGpuFamily::Apple2 => MetalVersion::V2_0,
            AppleGpuFamily::Common1
            | AppleGpuFamily::Common2
            | AppleGpuFamily::Common3 => MetalVersion::V2_0,
            AppleGpuFamily::Mac1 | AppleGpuFamily::Mac2 => MetalVersion::V2_0,
        }
    }

    /// Get the family name as a string.
    #[inline]
    pub const fn name(&self) -> &'static str {
        match self {
            AppleGpuFamily::Apple1 => "Apple1",
            AppleGpuFamily::Apple2 => "Apple2",
            AppleGpuFamily::Apple3 => "Apple3",
            AppleGpuFamily::Apple4 => "Apple4",
            AppleGpuFamily::Apple5 => "Apple5",
            AppleGpuFamily::Apple6 => "Apple6",
            AppleGpuFamily::Apple7 => "Apple7",
            AppleGpuFamily::Apple8 => "Apple8",
            AppleGpuFamily::Apple9 => "Apple9",
            AppleGpuFamily::Common1 => "Common1",
            AppleGpuFamily::Common2 => "Common2",
            AppleGpuFamily::Common3 => "Common3",
            AppleGpuFamily::Mac1 => "Mac1",
            AppleGpuFamily::Mac2 => "Mac2",
        }
    }
}

impl Default for AppleGpuFamily {
    fn default() -> Self {
        AppleGpuFamily::Apple1
    }
}

impl std::fmt::Display for AppleGpuFamily {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ============================================================================
// PlatformLimits
// ============================================================================

/// Platform-specific resource limits.
///
/// iOS and macOS have different maximum resource limits for Metal.
/// This struct captures the platform-specific limits for vertex buffers,
/// fragment inputs, and compute threads.
///
/// # Example
///
/// ```
/// use renderer_backend::backend::metal::PlatformLimits;
///
/// let ios_limits = PlatformLimits::ios();
/// let macos_limits = PlatformLimits::macos();
///
/// assert!(macos_limits.max_vertex_buffers >= ios_limits.max_vertex_buffers);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlatformLimits {
    /// Maximum number of vertex buffers.
    pub max_vertex_buffers: u32,

    /// Maximum number of fragment shader inputs.
    pub max_fragment_inputs: u32,

    /// Maximum threads per compute threadgroup.
    pub max_compute_threads: u32,
}

impl PlatformLimits {
    /// Get default iOS limits.
    ///
    /// iOS devices have more constrained limits due to mobile power/thermal budgets.
    #[inline]
    pub const fn ios() -> Self {
        Self {
            max_vertex_buffers: 31,
            max_fragment_inputs: 60,
            max_compute_threads: 512,
        }
    }

    /// Get default macOS limits.
    ///
    /// macOS devices (especially Apple Silicon) have higher limits.
    #[inline]
    pub const fn macos() -> Self {
        Self {
            max_vertex_buffers: 31,
            max_fragment_inputs: 124,
            max_compute_threads: 1024,
        }
    }

    /// Get default limits (conservative cross-platform).
    #[inline]
    pub const fn default_limits() -> Self {
        Self::ios() // Use iOS limits as conservative default
    }
}

impl Default for PlatformLimits {
    fn default() -> Self {
        Self::default_limits()
    }
}

// ============================================================================
// MetalCapabilities
// ============================================================================

/// Metal-specific capability detection.
///
/// This struct captures comprehensive Metal capabilities including GPU family,
/// memory properties, feature support, and platform information.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::backend::metal::MetalCapabilities;
///
/// # async fn example() {
/// let instance = wgpu::Instance::default();
/// let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.unwrap();
///
/// let caps = MetalCapabilities::detect(&adapter);
/// if caps.ray_tracing {
///     println!("Ray tracing available!");
/// }
/// if caps.unified_memory {
///     println!("Unified memory architecture");
/// }
/// # }
/// ```
#[derive(Debug, Clone, Default)]
pub struct MetalCapabilities {
    // GPU Family
    /// Detected Apple GPU family.
    pub gpu_family: Option<AppleGpuFamily>,

    // Memory
    /// Whether the device uses unified memory architecture.
    pub unified_memory: bool,

    /// Maximum buffer length in bytes.
    pub max_buffer_length: u64,

    /// Maximum texture dimension (width/height).
    pub max_texture_size: u32,

    // Features
    /// Whether argument buffers (tier 1) are supported.
    pub argument_buffers: bool,

    /// Whether argument buffers tier 2 (bindless) is supported.
    pub argument_buffers_tier2: bool,

    /// Whether ray tracing is supported.
    pub ray_tracing: bool,

    /// Whether mesh shaders are supported.
    pub mesh_shaders: bool,

    /// Whether SIMD-group operations are supported.
    pub simd_groups: bool,

    // Platform
    /// Whether this is an iOS device.
    pub is_ios: bool,

    /// Whether this is a macOS device.
    pub is_macos: bool,

    /// Whether this is an iOS simulator.
    pub is_simulator: bool,
}

impl MetalCapabilities {
    /// Detect Metal capabilities from a wgpu adapter.
    ///
    /// # Arguments
    ///
    /// * `adapter` - The wgpu adapter to query
    ///
    /// # Returns
    ///
    /// The detected Metal capabilities.
    pub fn detect(adapter: &Adapter) -> Self {
        let info = adapter.get_info();
        let limits = adapter.limits();
        let features = adapter.features();

        // Detect GPU family from device name
        let gpu_family = Self::detect_gpu_family(&info.name);

        // Apple Silicon uses unified memory
        let unified_memory = gpu_family
            .map(|f| matches!(
                f,
                AppleGpuFamily::Apple7 | AppleGpuFamily::Apple8 | AppleGpuFamily::Apple9
            ))
            .unwrap_or(false);

        // Determine platform (this is compile-time, but we also check device name)
        let is_ios = cfg!(target_os = "ios")
            || info.name.to_lowercase().contains("iphone")
            || info.name.to_lowercase().contains("ipad");
        let is_macos = cfg!(target_os = "macos") && !is_ios;
        let is_simulator = cfg!(target_os = "ios") && cfg!(target_arch = "x86_64");

        // Feature detection
        let ray_tracing = features.contains(Features::RAY_TRACING_ACCELERATION_STRUCTURE)
            || features.contains(Features::RAY_QUERY)
            || gpu_family.map(|f| f.supports_ray_tracing()).unwrap_or(false);

        let mesh_shaders = gpu_family.map(|f| f.supports_mesh_shaders()).unwrap_or(false);

        let simd_groups = gpu_family.map(|f| f.supports_simd_groups()).unwrap_or(false);

        // Argument buffers detection
        let argument_buffers = gpu_family.is_some()
            && !matches!(
                gpu_family,
                Some(AppleGpuFamily::Apple1) | Some(AppleGpuFamily::Apple2)
            );

        let argument_buffers_tier2 = gpu_family
            .map(|f| matches!(
                f,
                AppleGpuFamily::Apple5
                    | AppleGpuFamily::Apple6
                    | AppleGpuFamily::Apple7
                    | AppleGpuFamily::Apple8
                    | AppleGpuFamily::Apple9
                    | AppleGpuFamily::Mac2
            ))
            .unwrap_or(false);

        // Get limits
        let max_buffer_length = if unified_memory {
            // Apple Silicon can address more memory
            u64::from(limits.max_buffer_size).min(256 * 1024 * 1024 * 1024) // 256 GB max
        } else {
            u64::from(limits.max_buffer_size).min(1024 * 1024 * 1024) // 1 GB for Intel
        };

        let max_texture_size = limits.max_texture_dimension_2d;

        Self {
            gpu_family,
            unified_memory,
            max_buffer_length,
            max_texture_size,
            argument_buffers,
            argument_buffers_tier2,
            ray_tracing,
            mesh_shaders,
            simd_groups,
            is_ios,
            is_macos,
            is_simulator,
        }
    }

    /// Detect GPU family from device name string.
    fn detect_gpu_family(name: &str) -> Option<AppleGpuFamily> {
        let name_lower = name.to_lowercase();

        // M-series chips (Apple Silicon Macs)
        if name_lower.contains("m4") || name_lower.contains("a17") {
            return Some(AppleGpuFamily::Apple9);
        }
        if name_lower.contains("m3") || name_lower.contains("a16") {
            return Some(AppleGpuFamily::Apple9);
        }
        if name_lower.contains("m2") || name_lower.contains("a15") {
            return Some(AppleGpuFamily::Apple8);
        }
        if name_lower.contains("m1") || name_lower.contains("a14") {
            return Some(AppleGpuFamily::Apple7);
        }
        if name_lower.contains("a13") {
            return Some(AppleGpuFamily::Apple6);
        }
        if name_lower.contains("a12") {
            return Some(AppleGpuFamily::Apple5);
        }
        if name_lower.contains("a11") {
            return Some(AppleGpuFamily::Apple4);
        }
        if name_lower.contains("a10") || name_lower.contains("a9") {
            return Some(AppleGpuFamily::Apple3);
        }
        if name_lower.contains("a8x") {
            return Some(AppleGpuFamily::Apple2);
        }
        if name_lower.contains("a8") || name_lower.contains("a7") {
            return Some(AppleGpuFamily::Apple1);
        }

        // Intel Mac
        if name_lower.contains("intel") {
            if name_lower.contains("uhd")
                || name_lower.contains("iris")
                || name_lower.contains("pro ")
            {
                return Some(AppleGpuFamily::Mac2);
            }
            return Some(AppleGpuFamily::Mac1);
        }

        // AMD (eGPU or Mac Pro)
        if name_lower.contains("amd") || name_lower.contains("radeon") {
            return Some(AppleGpuFamily::Mac2);
        }

        // Generic Apple GPU
        if name_lower.contains("apple") {
            return Some(AppleGpuFamily::Apple7); // Assume modern
        }

        None
    }

    /// Check if this is a discrete GPU (eGPU).
    ///
    /// # Returns
    ///
    /// `true` for external/discrete GPUs (AMD eGPUs, Mac Pro GPUs).
    #[inline]
    pub fn is_discrete(&self) -> bool {
        // Intel and AMD GPUs on Macs are discrete or eGPUs
        matches!(
            self.gpu_family,
            Some(AppleGpuFamily::Mac1) | Some(AppleGpuFamily::Mac2)
        )
    }

    /// Check if this is Apple Silicon (M1/M2/M3/M4).
    ///
    /// # Returns
    ///
    /// `true` for Apple Silicon (unified memory architecture).
    #[inline]
    pub fn is_apple_silicon(&self) -> bool {
        self.unified_memory
            && matches!(
                self.gpu_family,
                Some(AppleGpuFamily::Apple7)
                    | Some(AppleGpuFamily::Apple8)
                    | Some(AppleGpuFamily::Apple9)
            )
    }

    /// Get platform-specific limits.
    ///
    /// # Returns
    ///
    /// Platform limits for iOS or macOS.
    #[inline]
    pub fn platform_limits(&self) -> PlatformLimits {
        if self.is_ios {
            PlatformLimits::ios()
        } else {
            PlatformLimits::macos()
        }
    }
}

// ============================================================================
// MetalInfo
// ============================================================================

/// Comprehensive Metal device information.
///
/// This struct combines Metal version, capabilities, device name, and
/// registry ID for complete device identification and feature queries.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::backend::metal::MetalInfo;
///
/// # async fn example() {
/// let instance = wgpu::Instance::default();
/// let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.unwrap();
///
/// let info = MetalInfo::from_adapter(&adapter);
/// println!("Device: {}", info.device_name);
/// println!("Metal version: {}", info.version);
/// if info.supports_ray_tracing() {
///     println!("Ray tracing available!");
/// }
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct MetalInfo {
    /// Metal API version.
    pub version: MetalVersion,

    /// Metal capabilities.
    pub capabilities: MetalCapabilities,

    /// Device name.
    pub device_name: String,

    /// Device registry ID (unique identifier).
    pub registry_id: u64,
}

impl MetalInfo {
    /// Create MetalInfo from a wgpu adapter.
    ///
    /// # Arguments
    ///
    /// * `adapter` - The wgpu adapter to query
    ///
    /// # Returns
    ///
    /// A new `MetalInfo` instance.
    pub fn from_adapter(adapter: &Adapter) -> Self {
        let info = adapter.get_info();
        let capabilities = MetalCapabilities::detect(adapter);

        // Determine Metal version based on GPU family
        let version = capabilities
            .gpu_family
            .map(|f| f.min_metal_version())
            .unwrap_or(MetalVersion::V2_0);

        // Use vendor_id as registry_id (closest approximation in wgpu)
        let registry_id = u64::from(info.vendor);

        Self {
            version,
            capabilities,
            device_name: info.name.clone(),
            registry_id,
        }
    }

    /// Check if bindless resources are supported.
    ///
    /// Bindless requires argument buffers tier 2.
    ///
    /// # Returns
    ///
    /// `true` if bindless resources are available.
    #[inline]
    pub fn supports_bindless(&self) -> bool {
        self.capabilities.argument_buffers_tier2
    }

    /// Check if ray tracing is supported.
    ///
    /// # Returns
    ///
    /// `true` if ray tracing is available.
    #[inline]
    pub fn supports_ray_tracing(&self) -> bool {
        self.capabilities.ray_tracing
    }
}

// ============================================================================
// MetalGpuFamily
// ============================================================================

/// Metal GPU family classification.
///
/// Apple's Metal API organizes GPU capabilities into families. Each family
/// represents a set of features that are guaranteed to be available on
/// hardware in that family.
///
/// # Family Hierarchy
///
/// - **Apple families** (Apple1-9): iPhone/iPad A-series chips
/// - **Mac families** (Mac1-2): Intel-based Macs
/// - **Common families** (Common1-3): Cross-platform baseline
/// - **Metal3**: Latest feature tier (M1+, A14+)
///
/// # Example
///
/// ```
/// use renderer_backend::backend::metal::MetalGpuFamily;
///
/// let family = MetalGpuFamily::from_device_name("Apple M3 Pro");
/// assert!(family.is_apple_silicon());
/// assert!(family.supports_metal3());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MetalGpuFamily {
    /// Apple GPU family 1 (A7-A8)
    ///
    /// First Metal-capable chips. Limited feature set.
    Apple1,

    /// Apple GPU family 2 (A8-A8X)
    Apple2,

    /// Apple GPU family 3 (A9-A10)
    ///
    /// Adds tessellation and resource heaps.
    Apple3,

    /// Apple GPU family 4 (A11)
    ///
    /// Adds tile shaders and imageblock.
    Apple4,

    /// Apple GPU family 5 (A12)
    ///
    /// Adds sparse textures.
    Apple5,

    /// Apple GPU family 6 (A13)
    ///
    /// Adds ASTC HDR textures.
    Apple6,

    /// Apple GPU family 7 (A14, M1)
    ///
    /// First family with ray tracing and mesh shaders.
    Apple7,

    /// Apple GPU family 8 (A15, M2)
    ///
    /// Improved ray tracing performance.
    Apple8,

    /// Apple GPU family 9 (A16, A17, M3, M4)
    ///
    /// Hardware-accelerated ray tracing, dynamic caching.
    Apple9,

    /// Mac GPU family 1 (Intel Mac - older)
    ///
    /// Basic Metal support for Intel GPUs.
    Mac1,

    /// Mac GPU family 2 (Intel Mac - newer)
    ///
    /// Enhanced Metal support for Intel GPUs.
    Mac2,

    /// Common GPU family 1
    ///
    /// Baseline cross-platform features.
    Common1,

    /// Common GPU family 2
    ///
    /// Enhanced cross-platform features.
    Common2,

    /// Common GPU family 3
    ///
    /// Modern cross-platform features (Vulkan 1.1 equivalent).
    Common3,

    /// Metal 3 feature tier.
    ///
    /// Requires A14+/M1+ hardware. Includes:
    /// - Ray tracing
    /// - Mesh shaders (object/mesh functions)
    /// - Offline compilation
    /// - Bindless resources (argument buffers tier 2)
    Metal3,

    /// Unknown or undetected GPU family.
    #[default]
    Unknown,
}

impl MetalGpuFamily {
    /// Detect GPU family from a wgpu adapter.
    ///
    /// # Arguments
    ///
    /// * `adapter` - The wgpu adapter to query (should be a Metal adapter)
    ///
    /// # Returns
    ///
    /// The detected GPU family.
    #[inline]
    pub fn from_adapter(adapter: &Adapter) -> Self {
        let info = adapter.get_info();
        Self::from_device_name(&info.name)
    }

    /// Parse GPU family from device name string.
    ///
    /// # Arguments
    ///
    /// * `name` - The device name from adapter info
    ///
    /// # Returns
    ///
    /// The detected GPU family based on the device name.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::metal::MetalGpuFamily;
    ///
    /// assert!(MetalGpuFamily::from_device_name("Apple M3 Max").is_apple_silicon());
    /// assert!(MetalGpuFamily::from_device_name("Intel UHD Graphics 630").is_intel_mac());
    /// ```
    pub fn from_device_name(name: &str) -> Self {
        let name_lower = name.to_lowercase();

        // Check for M-series first (most common modern case)
        if name_lower.contains("m4") {
            return MetalGpuFamily::Apple9;
        }
        if name_lower.contains("m3") {
            return MetalGpuFamily::Apple9;
        }
        if name_lower.contains("m2") {
            return MetalGpuFamily::Apple8;
        }
        if name_lower.contains("m1") {
            return MetalGpuFamily::Apple7;
        }

        // Check for A-series chips
        if name_lower.contains("a17") {
            return MetalGpuFamily::Apple9;
        }
        if name_lower.contains("a16") {
            return MetalGpuFamily::Apple9;
        }
        if name_lower.contains("a15") {
            return MetalGpuFamily::Apple8;
        }
        if name_lower.contains("a14") {
            return MetalGpuFamily::Apple7;
        }
        if name_lower.contains("a13") {
            return MetalGpuFamily::Apple6;
        }
        if name_lower.contains("a12") {
            return MetalGpuFamily::Apple5;
        }
        if name_lower.contains("a11") {
            return MetalGpuFamily::Apple4;
        }
        if name_lower.contains("a10") {
            return MetalGpuFamily::Apple3;
        }
        if name_lower.contains("a9") {
            return MetalGpuFamily::Apple3;
        }
        if name_lower.contains("a8") {
            return MetalGpuFamily::Apple2;
        }
        if name_lower.contains("a7") {
            return MetalGpuFamily::Apple1;
        }

        // Check for Intel Mac
        if name_lower.contains("intel") {
            // Newer Intel GPUs
            if name_lower.contains("uhd")
                || name_lower.contains("iris")
                || name_lower.contains("pro ")
            {
                return MetalGpuFamily::Mac2;
            }
            return MetalGpuFamily::Mac1;
        }

        // Check for AMD (eGPU or pre-M1 Mac Pro)
        if name_lower.contains("amd") || name_lower.contains("radeon") {
            return MetalGpuFamily::Mac2;
        }

        // Generic Apple GPU mention
        if name_lower.contains("apple") {
            // Assume modern Apple Silicon if just "Apple GPU"
            return MetalGpuFamily::Apple7;
        }

        MetalGpuFamily::Unknown
    }

    /// Check if this is Apple Silicon (M-series or A14+).
    ///
    /// # Returns
    ///
    /// `true` for Apple7+ families which correspond to Apple Silicon.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::metal::MetalGpuFamily;
    ///
    /// assert!(MetalGpuFamily::Apple7.is_apple_silicon());
    /// assert!(MetalGpuFamily::Apple9.is_apple_silicon());
    /// assert!(!MetalGpuFamily::Mac2.is_apple_silicon());
    /// ```
    #[inline]
    pub const fn is_apple_silicon(&self) -> bool {
        matches!(
            self,
            MetalGpuFamily::Apple7
                | MetalGpuFamily::Apple8
                | MetalGpuFamily::Apple9
                | MetalGpuFamily::Metal3
        )
    }

    /// Check if this is an Intel Mac.
    ///
    /// # Returns
    ///
    /// `true` for Mac1 and Mac2 families (Intel-based Macs).
    #[inline]
    pub const fn is_intel_mac(&self) -> bool {
        matches!(self, MetalGpuFamily::Mac1 | MetalGpuFamily::Mac2)
    }

    /// Check if this family supports Metal 3 features.
    ///
    /// Metal 3 requires Apple7+ (A14/M1 or later).
    ///
    /// # Returns
    ///
    /// `true` if Metal 3 features are available.
    #[inline]
    pub const fn supports_metal3(&self) -> bool {
        matches!(
            self,
            MetalGpuFamily::Apple7
                | MetalGpuFamily::Apple8
                | MetalGpuFamily::Apple9
                | MetalGpuFamily::Metal3
        )
    }

    /// Check if this family supports ray tracing.
    ///
    /// Ray tracing requires Apple7+ (A14/M1 or later).
    ///
    /// # Returns
    ///
    /// `true` if ray tracing is supported.
    #[inline]
    pub const fn supports_ray_tracing(&self) -> bool {
        self.supports_metal3()
    }

    /// Check if this family supports mesh shaders.
    ///
    /// Mesh shaders (object/mesh functions) require Apple7+ (A14/M1 or later).
    ///
    /// # Returns
    ///
    /// `true` if mesh shaders are supported.
    #[inline]
    pub const fn supports_mesh_shaders(&self) -> bool {
        self.supports_metal3()
    }

    /// Get the family name as a string.
    ///
    /// # Returns
    ///
    /// A static string with the family name.
    #[inline]
    pub const fn name(&self) -> &'static str {
        match self {
            MetalGpuFamily::Apple1 => "Apple1",
            MetalGpuFamily::Apple2 => "Apple2",
            MetalGpuFamily::Apple3 => "Apple3",
            MetalGpuFamily::Apple4 => "Apple4",
            MetalGpuFamily::Apple5 => "Apple5",
            MetalGpuFamily::Apple6 => "Apple6",
            MetalGpuFamily::Apple7 => "Apple7",
            MetalGpuFamily::Apple8 => "Apple8",
            MetalGpuFamily::Apple9 => "Apple9",
            MetalGpuFamily::Mac1 => "Mac1",
            MetalGpuFamily::Mac2 => "Mac2",
            MetalGpuFamily::Common1 => "Common1",
            MetalGpuFamily::Common2 => "Common2",
            MetalGpuFamily::Common3 => "Common3",
            MetalGpuFamily::Metal3 => "Metal3",
            MetalGpuFamily::Unknown => "Unknown",
        }
    }

    /// Get the numeric family version (for Apple families).
    ///
    /// # Returns
    ///
    /// The Apple family version (1-9) or 0 for non-Apple families.
    #[inline]
    pub const fn apple_version(&self) -> u8 {
        match self {
            MetalGpuFamily::Apple1 => 1,
            MetalGpuFamily::Apple2 => 2,
            MetalGpuFamily::Apple3 => 3,
            MetalGpuFamily::Apple4 => 4,
            MetalGpuFamily::Apple5 => 5,
            MetalGpuFamily::Apple6 => 6,
            MetalGpuFamily::Apple7 => 7,
            MetalGpuFamily::Apple8 => 8,
            MetalGpuFamily::Apple9 => 9,
            _ => 0,
        }
    }
}

impl std::fmt::Display for MetalGpuFamily {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ============================================================================
// AppleSiliconGeneration
// ============================================================================

/// Apple Silicon chip generation.
///
/// This enum provides fine-grained identification of Apple Silicon chips,
/// including the specific tier (base, Pro, Max, Ultra).
///
/// # Generations
///
/// | Generation | Year | Process | Key Features |
/// |------------|------|---------|--------------|
/// | M1 | 2020 | 5nm | First Mac Apple Silicon |
/// | M2 | 2022 | 5nm | +18% CPU, +35% GPU |
/// | M3 | 2023 | 3nm | Dynamic caching, HW RT |
/// | M4 | 2024 | 3nm | Enhanced neural engine |
///
/// # Example
///
/// ```
/// use renderer_backend::backend::metal::AppleSiliconGeneration;
///
/// let chip = AppleSiliconGeneration::from_device_name("Apple M3 Max");
/// assert_eq!(chip, AppleSiliconGeneration::M3Max);
/// assert_eq!(chip.generation_number(), 3);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AppleSiliconGeneration {
    // A-series (iOS/iPadOS)
    /// A14 Bionic (iPhone 12, iPad Air 4)
    A14,

    /// A15 Bionic (iPhone 13/14, iPad mini 6)
    A15,

    /// A16 Bionic (iPhone 14 Pro)
    A16,

    /// A17 Pro (iPhone 15 Pro)
    A17Pro,

    // M1 Generation (2020)
    /// M1 base chip (MacBook Air, Mac mini, iMac)
    M1,

    /// M1 Pro (MacBook Pro 14/16")
    M1Pro,

    /// M1 Max (MacBook Pro 14/16")
    M1Max,

    /// M1 Ultra (Mac Studio)
    M1Ultra,

    // M2 Generation (2022)
    /// M2 base chip (MacBook Air, Mac mini, iPad Pro)
    M2,

    /// M2 Pro (MacBook Pro 14/16", Mac mini)
    M2Pro,

    /// M2 Max (MacBook Pro 14/16", Mac Studio)
    M2Max,

    /// M2 Ultra (Mac Studio, Mac Pro)
    M2Ultra,

    // M3 Generation (2023)
    /// M3 base chip (MacBook Air, iMac)
    M3,

    /// M3 Pro (MacBook Pro 14/16")
    M3Pro,

    /// M3 Max (MacBook Pro 14/16")
    M3Max,

    // M4 Generation (2024)
    /// M4 base chip (iPad Pro)
    M4,

    /// M4 Pro (MacBook Pro 14/16", Mac mini)
    M4Pro,

    /// M4 Max (MacBook Pro 14/16")
    M4Max,

    /// Unknown or non-Apple Silicon
    #[default]
    Unknown,
}

impl AppleSiliconGeneration {
    /// Parse chip generation from device name.
    ///
    /// # Arguments
    ///
    /// * `name` - The device name from adapter info
    ///
    /// # Returns
    ///
    /// The detected Apple Silicon generation.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::metal::AppleSiliconGeneration;
    ///
    /// assert_eq!(
    ///     AppleSiliconGeneration::from_device_name("Apple M3 Pro"),
    ///     AppleSiliconGeneration::M3Pro
    /// );
    /// ```
    pub fn from_device_name(name: &str) -> Self {
        let name_lower = name.to_lowercase();

        // M4 generation
        if name_lower.contains("m4") {
            if name_lower.contains("max") {
                return AppleSiliconGeneration::M4Max;
            }
            if name_lower.contains("pro") {
                return AppleSiliconGeneration::M4Pro;
            }
            return AppleSiliconGeneration::M4;
        }

        // M3 generation
        if name_lower.contains("m3") {
            if name_lower.contains("max") {
                return AppleSiliconGeneration::M3Max;
            }
            if name_lower.contains("pro") {
                return AppleSiliconGeneration::M3Pro;
            }
            return AppleSiliconGeneration::M3;
        }

        // M2 generation
        if name_lower.contains("m2") {
            if name_lower.contains("ultra") {
                return AppleSiliconGeneration::M2Ultra;
            }
            if name_lower.contains("max") {
                return AppleSiliconGeneration::M2Max;
            }
            if name_lower.contains("pro") {
                return AppleSiliconGeneration::M2Pro;
            }
            return AppleSiliconGeneration::M2;
        }

        // M1 generation
        if name_lower.contains("m1") {
            if name_lower.contains("ultra") {
                return AppleSiliconGeneration::M1Ultra;
            }
            if name_lower.contains("max") {
                return AppleSiliconGeneration::M1Max;
            }
            if name_lower.contains("pro") {
                return AppleSiliconGeneration::M1Pro;
            }
            return AppleSiliconGeneration::M1;
        }

        // A-series chips
        if name_lower.contains("a17") {
            return AppleSiliconGeneration::A17Pro;
        }
        if name_lower.contains("a16") {
            return AppleSiliconGeneration::A16;
        }
        if name_lower.contains("a15") {
            return AppleSiliconGeneration::A15;
        }
        if name_lower.contains("a14") {
            return AppleSiliconGeneration::A14;
        }

        AppleSiliconGeneration::Unknown
    }

    /// Get the M-series generation number.
    ///
    /// # Returns
    ///
    /// The generation number (1-4 for M1-M4, 0 for A-series or unknown).
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::metal::AppleSiliconGeneration;
    ///
    /// assert_eq!(AppleSiliconGeneration::M3Pro.generation_number(), 3);
    /// assert_eq!(AppleSiliconGeneration::M1Ultra.generation_number(), 1);
    /// assert_eq!(AppleSiliconGeneration::A17Pro.generation_number(), 0);
    /// ```
    #[inline]
    pub const fn generation_number(&self) -> u8 {
        match self {
            AppleSiliconGeneration::M1
            | AppleSiliconGeneration::M1Pro
            | AppleSiliconGeneration::M1Max
            | AppleSiliconGeneration::M1Ultra => 1,
            AppleSiliconGeneration::M2
            | AppleSiliconGeneration::M2Pro
            | AppleSiliconGeneration::M2Max
            | AppleSiliconGeneration::M2Ultra => 2,
            AppleSiliconGeneration::M3
            | AppleSiliconGeneration::M3Pro
            | AppleSiliconGeneration::M3Max => 3,
            AppleSiliconGeneration::M4
            | AppleSiliconGeneration::M4Pro
            | AppleSiliconGeneration::M4Max => 4,
            _ => 0,
        }
    }

    /// Check if this is an M-series chip.
    ///
    /// # Returns
    ///
    /// `true` for M1, M2, M3, or M4 series chips.
    #[inline]
    pub const fn is_m_series(&self) -> bool {
        self.generation_number() > 0
    }

    /// Check if this is an A-series chip.
    ///
    /// # Returns
    ///
    /// `true` for A14-A17 chips.
    #[inline]
    pub const fn is_a_series(&self) -> bool {
        matches!(
            self,
            AppleSiliconGeneration::A14
                | AppleSiliconGeneration::A15
                | AppleSiliconGeneration::A16
                | AppleSiliconGeneration::A17Pro
        )
    }

    /// Check if this is a Pro/Max/Ultra tier chip.
    ///
    /// # Returns
    ///
    /// `true` for Pro, Max, or Ultra variants.
    #[inline]
    pub const fn is_pro_tier(&self) -> bool {
        matches!(
            self,
            AppleSiliconGeneration::M1Pro
                | AppleSiliconGeneration::M1Max
                | AppleSiliconGeneration::M1Ultra
                | AppleSiliconGeneration::M2Pro
                | AppleSiliconGeneration::M2Max
                | AppleSiliconGeneration::M2Ultra
                | AppleSiliconGeneration::M3Pro
                | AppleSiliconGeneration::M3Max
                | AppleSiliconGeneration::M4Pro
                | AppleSiliconGeneration::M4Max
                | AppleSiliconGeneration::A17Pro
        )
    }

    /// Get GPU core count estimate for this chip.
    ///
    /// Note: Actual core counts vary by SKU; these are typical maximums.
    ///
    /// # Returns
    ///
    /// Estimated GPU core count.
    #[inline]
    pub const fn estimated_gpu_cores(&self) -> u32 {
        match self {
            AppleSiliconGeneration::A14 => 4,
            AppleSiliconGeneration::A15 => 5,
            AppleSiliconGeneration::A16 => 5,
            AppleSiliconGeneration::A17Pro => 6,
            AppleSiliconGeneration::M1 => 8,
            AppleSiliconGeneration::M1Pro => 16,
            AppleSiliconGeneration::M1Max => 32,
            AppleSiliconGeneration::M1Ultra => 64,
            AppleSiliconGeneration::M2 => 10,
            AppleSiliconGeneration::M2Pro => 19,
            AppleSiliconGeneration::M2Max => 38,
            AppleSiliconGeneration::M2Ultra => 76,
            AppleSiliconGeneration::M3 => 10,
            AppleSiliconGeneration::M3Pro => 18,
            AppleSiliconGeneration::M3Max => 40,
            AppleSiliconGeneration::M4 => 10,
            AppleSiliconGeneration::M4Pro => 20,
            AppleSiliconGeneration::M4Max => 40,
            AppleSiliconGeneration::Unknown => 0,
        }
    }

    /// Get the chip name as a string.
    ///
    /// # Returns
    ///
    /// A static string with the chip name.
    #[inline]
    pub const fn name(&self) -> &'static str {
        match self {
            AppleSiliconGeneration::A14 => "A14 Bionic",
            AppleSiliconGeneration::A15 => "A15 Bionic",
            AppleSiliconGeneration::A16 => "A16 Bionic",
            AppleSiliconGeneration::A17Pro => "A17 Pro",
            AppleSiliconGeneration::M1 => "M1",
            AppleSiliconGeneration::M1Pro => "M1 Pro",
            AppleSiliconGeneration::M1Max => "M1 Max",
            AppleSiliconGeneration::M1Ultra => "M1 Ultra",
            AppleSiliconGeneration::M2 => "M2",
            AppleSiliconGeneration::M2Pro => "M2 Pro",
            AppleSiliconGeneration::M2Max => "M2 Max",
            AppleSiliconGeneration::M2Ultra => "M2 Ultra",
            AppleSiliconGeneration::M3 => "M3",
            AppleSiliconGeneration::M3Pro => "M3 Pro",
            AppleSiliconGeneration::M3Max => "M3 Max",
            AppleSiliconGeneration::M4 => "M4",
            AppleSiliconGeneration::M4Pro => "M4 Pro",
            AppleSiliconGeneration::M4Max => "M4 Max",
            AppleSiliconGeneration::Unknown => "Unknown",
        }
    }
}

impl std::fmt::Display for AppleSiliconGeneration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ============================================================================
// MetalFeatures
// ============================================================================

/// Metal-specific feature detection.
///
/// This struct captures Metal capabilities beyond what wgpu exposes directly.
/// Detection is performed by inspecting wgpu's feature flags and adapter info,
/// mapping them to Metal-specific functionality.
///
/// # Feature Categories
///
/// | Category | Features |
/// |----------|----------|
/// | Ray Tracing | Intersection functions, acceleration structures |
/// | Mesh Shaders | Object/mesh functions |
/// | Argument Buffers | Tier 1 (limited), Tier 2 (bindless) |
/// | Memory | Memoryless attachments, heaps, lossless compression |
///
/// # Example
///
/// ```no_run
/// use renderer_backend::backend::metal::MetalFeatures;
///
/// # async fn example() {
/// let instance = wgpu::Instance::default();
/// let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.unwrap();
///
/// let metal = MetalFeatures::detect(&adapter);
/// println!("GPU Family: {}", metal.gpu_family);
/// println!("Argument Buffers Tier: {}", metal.argument_buffers_tier);
/// println!("Ray Tracing: {}", metal.ray_tracing);
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MetalFeatures {
    /// Detected GPU family.
    pub gpu_family: MetalGpuFamily,

    /// Detected Apple Silicon generation (if applicable).
    pub silicon_generation: AppleSiliconGeneration,

    /// Argument buffers tier.
    ///
    /// - 0: No argument buffers
    /// - 1: Basic argument buffers (limited indexing)
    /// - 2: Full argument buffers (bindless, unbounded arrays)
    pub argument_buffers_tier: u8,

    /// 32-bit float MSAA resolve support.
    ///
    /// Allows resolving 32-bit float render targets directly.
    pub float32_msaa_resolve: bool,

    /// Sparse texture support.
    ///
    /// Allows partially resident textures.
    pub sparse_textures: bool,

    /// Primitive motion blur support (ray tracing).
    ///
    /// Allows motion blur in ray tracing through motion transforms.
    pub primitive_motion_blur: bool,

    /// Ray tracing support.
    ///
    /// Includes acceleration structures and intersection functions.
    pub ray_tracing: bool,

    /// Mesh shaders support.
    ///
    /// Metal's object/mesh function pipeline.
    pub mesh_shaders: bool,

    /// Memoryless render targets support.
    ///
    /// Allows render targets that don't allocate memory (tile memory only).
    pub memoryless_render_targets: bool,

    /// Lossless texture compression support.
    ///
    /// Hardware lossless compression for textures and render targets.
    pub lossless_compression: bool,

    /// Function pointers support.
    ///
    /// Allows indirect function calls in shaders.
    pub function_pointers: bool,

    /// Primitive restart support with 32-bit indices.
    pub primitive_restart_32bit: bool,

    /// SIMD-group functions support.
    ///
    /// Subgroup operations (quad, wave).
    pub simd_group: bool,

    /// Read-write texture support.
    ///
    /// Allows reading and writing to the same texture in a shader.
    pub read_write_textures: bool,

    /// Tile shaders support.
    ///
    /// Apple's tile-based deferred rendering shader stage.
    pub tile_shaders: bool,

    /// Imageblock support.
    ///
    /// Allows explicit tile memory management.
    pub imageblock: bool,
}

impl MetalFeatures {
    /// Detect Metal features from a wgpu adapter.
    ///
    /// # Arguments
    ///
    /// * `adapter` - The wgpu adapter to query (should be a Metal adapter)
    ///
    /// # Returns
    ///
    /// The detected Metal features.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::backend::metal::MetalFeatures;
    ///
    /// # async fn example() {
    /// let instance = wgpu::Instance::default();
    /// let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.unwrap();
    ///
    /// let features = MetalFeatures::detect(&adapter);
    /// if features.supports_rt() {
    ///     println!("Metal ray tracing available!");
    /// }
    /// # }
    /// ```
    pub fn detect(adapter: &Adapter) -> Self {
        let wgpu_features = adapter.features();
        let info = adapter.get_info();
        Self::from_adapter_info(&info.name, wgpu_features)
    }

    /// Create features from device name and wgpu features.
    ///
    /// # Arguments
    ///
    /// * `device_name` - The device name from adapter info
    /// * `features` - The wgpu features from the adapter
    ///
    /// # Returns
    ///
    /// The detected Metal features.
    pub fn from_adapter_info(device_name: &str, features: Features) -> Self {
        let gpu_family = MetalGpuFamily::from_device_name(device_name);
        let silicon_generation = AppleSiliconGeneration::from_device_name(device_name);

        // Determine argument buffers tier
        let argument_buffers_tier = if gpu_family.supports_metal3() {
            2 // Metal 3 requires tier 2
        } else if gpu_family.is_apple_silicon()
            || matches!(gpu_family, MetalGpuFamily::Apple5 | MetalGpuFamily::Apple6)
        {
            2 // A12+ has tier 2
        } else if !matches!(
            gpu_family,
            MetalGpuFamily::Unknown
                | MetalGpuFamily::Apple1
                | MetalGpuFamily::Apple2
                | MetalGpuFamily::Apple3
        ) {
            1 // Most other Metal GPUs have tier 1
        } else {
            0
        };

        // Ray tracing detection
        let ray_tracing = features.contains(Features::RAY_TRACING_ACCELERATION_STRUCTURE)
            || features.contains(Features::RAY_QUERY)
            || gpu_family.supports_ray_tracing();

        // Mesh shaders - wgpu doesn't expose these yet, infer from family
        let mesh_shaders = gpu_family.supports_mesh_shaders();

        // Sparse textures - available from Apple5+
        let sparse_textures = matches!(
            gpu_family,
            MetalGpuFamily::Apple5
                | MetalGpuFamily::Apple6
                | MetalGpuFamily::Apple7
                | MetalGpuFamily::Apple8
                | MetalGpuFamily::Apple9
                | MetalGpuFamily::Metal3
        );

        // Motion blur requires Metal 3
        let primitive_motion_blur = gpu_family.supports_metal3();

        // Memoryless render targets - available on all iOS/Apple Silicon
        let memoryless_render_targets =
            gpu_family.is_apple_silicon() || gpu_family.apple_version() > 0;

        // Lossless compression - M1+ and A14+
        let lossless_compression = gpu_family.supports_metal3();

        // MSAA resolve for float32 - Apple4+
        let float32_msaa_resolve = gpu_family.apple_version() >= 4 || gpu_family.is_intel_mac();

        // Function pointers - Metal 3
        let function_pointers = gpu_family.supports_metal3();

        // Primitive restart with 32-bit - available on most Metal GPUs
        let primitive_restart_32bit =
            !matches!(gpu_family, MetalGpuFamily::Unknown | MetalGpuFamily::Apple1);

        // SIMD group - Apple3+
        let simd_group = gpu_family.apple_version() >= 3 || gpu_family.is_intel_mac();

        // Read-write textures - available from Apple3+ with certain formats
        let read_write_textures = gpu_family.apple_version() >= 3 || gpu_family.is_intel_mac();

        // Tile shaders - Apple4+
        let tile_shaders = gpu_family.apple_version() >= 4;

        // Imageblock - Apple4+
        let imageblock = gpu_family.apple_version() >= 4;

        let result = Self {
            gpu_family,
            silicon_generation,
            argument_buffers_tier,
            float32_msaa_resolve,
            sparse_textures,
            primitive_motion_blur,
            ray_tracing,
            mesh_shaders,
            memoryless_render_targets,
            lossless_compression,
            function_pointers,
            primitive_restart_32bit,
            simd_group,
            read_write_textures,
            tile_shaders,
            imageblock,
        };

        debug!(
            "MetalFeatures detected: family={}, RT={}, mesh={}, arg_tier={}",
            gpu_family.name(),
            ray_tracing,
            mesh_shaders,
            argument_buffers_tier
        );

        result
    }

    /// Create features by parsing device info string.
    ///
    /// # Arguments
    ///
    /// * `info` - Device info string to parse
    ///
    /// # Returns
    ///
    /// The detected Metal features.
    #[inline]
    pub fn from_device_info(info: &str) -> Self {
        Self::from_adapter_info(info, Features::empty())
    }

    /// Check if ray tracing is supported.
    ///
    /// # Returns
    ///
    /// `true` if Metal ray tracing is available.
    #[inline]
    pub const fn supports_rt(&self) -> bool {
        self.ray_tracing
    }

    /// Check if mesh shaders (object/mesh functions) are supported.
    ///
    /// # Returns
    ///
    /// `true` if Metal mesh shaders are available.
    #[inline]
    pub const fn supports_mesh_shaders(&self) -> bool {
        self.mesh_shaders
    }

    /// Check if any argument buffers are supported.
    ///
    /// # Returns
    ///
    /// `true` if argument buffers tier 1 or 2 is available.
    #[inline]
    pub const fn supports_argument_buffers(&self) -> bool {
        self.argument_buffers_tier >= 1
    }

    /// Check if bindless resources are supported.
    ///
    /// Bindless requires argument buffers tier 2.
    ///
    /// # Returns
    ///
    /// `true` if bindless resources are available.
    #[inline]
    pub const fn supports_bindless(&self) -> bool {
        self.argument_buffers_tier >= 2
    }

    /// Check if this is an M-series chip.
    ///
    /// # Returns
    ///
    /// `true` for M1, M2, M3, or M4 series chips.
    #[inline]
    pub const fn is_m_series(&self) -> bool {
        self.silicon_generation.is_m_series()
    }

    /// Get the minimum macOS version required for detected features.
    ///
    /// # Returns
    ///
    /// A tuple of (major, minor) version numbers.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::metal::MetalFeatures;
    ///
    /// let features = MetalFeatures::from_device_info("Apple M3");
    /// let (major, minor) = features.minimum_macos_version();
    /// println!("Requires macOS {}.{}", major, minor);
    /// ```
    #[inline]
    pub const fn minimum_macos_version(&self) -> (u32, u32) {
        // Metal 3 (M3 with hardware RT) requires Sonoma (14.0)
        if matches!(
            self.silicon_generation,
            AppleSiliconGeneration::M3
                | AppleSiliconGeneration::M3Pro
                | AppleSiliconGeneration::M3Max
                | AppleSiliconGeneration::M4
                | AppleSiliconGeneration::M4Pro
                | AppleSiliconGeneration::M4Max
        ) {
            return (14, 0);
        }

        // Metal 3 basics require Ventura (13.0)
        if self.ray_tracing || self.mesh_shaders {
            return (13, 0);
        }

        // Apple Silicon requires Big Sur (11.0)
        if self.gpu_family.is_apple_silicon() {
            return (11, 0);
        }

        // Basic Metal requires Mojave (10.14) for recent features
        if self.argument_buffers_tier >= 2 {
            return (10, 14);
        }

        // Fallback to Catalina for modern Metal
        (10, 15)
    }

    /// Create a summary string of detected features.
    ///
    /// # Returns
    ///
    /// A human-readable summary of available features.
    pub fn summary(&self) -> String {
        let mut features = Vec::new();

        features.push(self.gpu_family.name().to_string());

        if self.silicon_generation.is_m_series() {
            features.push(self.silicon_generation.name().to_string());
        }

        if self.ray_tracing {
            features.push("RT".to_string());
        }
        if self.mesh_shaders {
            features.push("Mesh".to_string());
        }
        if self.argument_buffers_tier > 0 {
            features.push(format!("ArgBuf-T{}", self.argument_buffers_tier));
        }
        if self.sparse_textures {
            features.push("Sparse".to_string());
        }
        if self.memoryless_render_targets {
            features.push("Memoryless".to_string());
        }
        if self.lossless_compression {
            features.push("LosslessComp".to_string());
        }

        features.join(", ")
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // MetalGpuFamily Tests
    // ========================================================================

    #[test]
    fn test_gpu_family_default() {
        let family = MetalGpuFamily::default();
        assert_eq!(family, MetalGpuFamily::Unknown);
    }

    #[test]
    fn test_gpu_family_from_m_series() {
        assert_eq!(
            MetalGpuFamily::from_device_name("Apple M1"),
            MetalGpuFamily::Apple7
        );
        assert_eq!(
            MetalGpuFamily::from_device_name("Apple M1 Pro"),
            MetalGpuFamily::Apple7
        );
        assert_eq!(
            MetalGpuFamily::from_device_name("Apple M1 Max"),
            MetalGpuFamily::Apple7
        );
        assert_eq!(
            MetalGpuFamily::from_device_name("Apple M1 Ultra"),
            MetalGpuFamily::Apple7
        );
        assert_eq!(
            MetalGpuFamily::from_device_name("Apple M2"),
            MetalGpuFamily::Apple8
        );
        assert_eq!(
            MetalGpuFamily::from_device_name("Apple M3"),
            MetalGpuFamily::Apple9
        );
        assert_eq!(
            MetalGpuFamily::from_device_name("Apple M3 Pro"),
            MetalGpuFamily::Apple9
        );
        assert_eq!(
            MetalGpuFamily::from_device_name("Apple M4"),
            MetalGpuFamily::Apple9
        );
    }

    #[test]
    fn test_gpu_family_from_a_series() {
        assert_eq!(
            MetalGpuFamily::from_device_name("Apple A14 Bionic GPU"),
            MetalGpuFamily::Apple7
        );
        assert_eq!(
            MetalGpuFamily::from_device_name("Apple A15"),
            MetalGpuFamily::Apple8
        );
        assert_eq!(
            MetalGpuFamily::from_device_name("Apple A16"),
            MetalGpuFamily::Apple9
        );
        assert_eq!(
            MetalGpuFamily::from_device_name("Apple A17 Pro"),
            MetalGpuFamily::Apple9
        );
        assert_eq!(
            MetalGpuFamily::from_device_name("Apple A13"),
            MetalGpuFamily::Apple6
        );
        assert_eq!(
            MetalGpuFamily::from_device_name("Apple A12"),
            MetalGpuFamily::Apple5
        );
    }

    #[test]
    fn test_gpu_family_from_intel() {
        assert_eq!(
            MetalGpuFamily::from_device_name("Intel UHD Graphics 630"),
            MetalGpuFamily::Mac2
        );
        assert_eq!(
            MetalGpuFamily::from_device_name("Intel Iris Pro Graphics"),
            MetalGpuFamily::Mac2
        );
        assert_eq!(
            MetalGpuFamily::from_device_name("Intel HD Graphics 4000"),
            MetalGpuFamily::Mac1
        );
    }

    #[test]
    fn test_gpu_family_from_amd() {
        assert_eq!(
            MetalGpuFamily::from_device_name("AMD Radeon Pro 5500M"),
            MetalGpuFamily::Mac2
        );
    }

    #[test]
    fn test_gpu_family_is_apple_silicon() {
        assert!(MetalGpuFamily::Apple7.is_apple_silicon());
        assert!(MetalGpuFamily::Apple8.is_apple_silicon());
        assert!(MetalGpuFamily::Apple9.is_apple_silicon());
        assert!(MetalGpuFamily::Metal3.is_apple_silicon());
        assert!(!MetalGpuFamily::Apple6.is_apple_silicon());
        assert!(!MetalGpuFamily::Mac2.is_apple_silicon());
        assert!(!MetalGpuFamily::Unknown.is_apple_silicon());
    }

    #[test]
    fn test_gpu_family_is_intel_mac() {
        assert!(MetalGpuFamily::Mac1.is_intel_mac());
        assert!(MetalGpuFamily::Mac2.is_intel_mac());
        assert!(!MetalGpuFamily::Apple7.is_intel_mac());
        assert!(!MetalGpuFamily::Unknown.is_intel_mac());
    }

    #[test]
    fn test_gpu_family_supports_metal3() {
        assert!(MetalGpuFamily::Apple7.supports_metal3());
        assert!(MetalGpuFamily::Apple8.supports_metal3());
        assert!(MetalGpuFamily::Apple9.supports_metal3());
        assert!(MetalGpuFamily::Metal3.supports_metal3());
        assert!(!MetalGpuFamily::Apple6.supports_metal3());
        assert!(!MetalGpuFamily::Mac2.supports_metal3());
    }

    #[test]
    fn test_gpu_family_supports_ray_tracing() {
        assert!(MetalGpuFamily::Apple7.supports_ray_tracing());
        assert!(MetalGpuFamily::Apple9.supports_ray_tracing());
        assert!(!MetalGpuFamily::Apple6.supports_ray_tracing());
        assert!(!MetalGpuFamily::Mac2.supports_ray_tracing());
    }

    #[test]
    fn test_gpu_family_supports_mesh_shaders() {
        assert!(MetalGpuFamily::Apple7.supports_mesh_shaders());
        assert!(MetalGpuFamily::Apple9.supports_mesh_shaders());
        assert!(!MetalGpuFamily::Apple6.supports_mesh_shaders());
        assert!(!MetalGpuFamily::Mac2.supports_mesh_shaders());
    }

    #[test]
    fn test_gpu_family_name() {
        assert_eq!(MetalGpuFamily::Apple7.name(), "Apple7");
        assert_eq!(MetalGpuFamily::Mac2.name(), "Mac2");
        assert_eq!(MetalGpuFamily::Metal3.name(), "Metal3");
        assert_eq!(MetalGpuFamily::Unknown.name(), "Unknown");
    }

    #[test]
    fn test_gpu_family_apple_version() {
        assert_eq!(MetalGpuFamily::Apple1.apple_version(), 1);
        assert_eq!(MetalGpuFamily::Apple7.apple_version(), 7);
        assert_eq!(MetalGpuFamily::Apple9.apple_version(), 9);
        assert_eq!(MetalGpuFamily::Mac2.apple_version(), 0);
        assert_eq!(MetalGpuFamily::Unknown.apple_version(), 0);
    }

    #[test]
    fn test_gpu_family_display() {
        assert_eq!(format!("{}", MetalGpuFamily::Apple7), "Apple7");
        assert_eq!(format!("{}", MetalGpuFamily::Metal3), "Metal3");
    }

    // ========================================================================
    // AppleSiliconGeneration Tests
    // ========================================================================

    #[test]
    fn test_silicon_generation_default() {
        let gen = AppleSiliconGeneration::default();
        assert_eq!(gen, AppleSiliconGeneration::Unknown);
    }

    #[test]
    fn test_silicon_generation_from_m_series() {
        assert_eq!(
            AppleSiliconGeneration::from_device_name("Apple M1"),
            AppleSiliconGeneration::M1
        );
        assert_eq!(
            AppleSiliconGeneration::from_device_name("Apple M1 Pro"),
            AppleSiliconGeneration::M1Pro
        );
        assert_eq!(
            AppleSiliconGeneration::from_device_name("Apple M1 Max"),
            AppleSiliconGeneration::M1Max
        );
        assert_eq!(
            AppleSiliconGeneration::from_device_name("Apple M1 Ultra"),
            AppleSiliconGeneration::M1Ultra
        );
        assert_eq!(
            AppleSiliconGeneration::from_device_name("Apple M2"),
            AppleSiliconGeneration::M2
        );
        assert_eq!(
            AppleSiliconGeneration::from_device_name("Apple M2 Pro"),
            AppleSiliconGeneration::M2Pro
        );
        assert_eq!(
            AppleSiliconGeneration::from_device_name("Apple M2 Max"),
            AppleSiliconGeneration::M2Max
        );
        assert_eq!(
            AppleSiliconGeneration::from_device_name("Apple M2 Ultra"),
            AppleSiliconGeneration::M2Ultra
        );
        assert_eq!(
            AppleSiliconGeneration::from_device_name("Apple M3"),
            AppleSiliconGeneration::M3
        );
        assert_eq!(
            AppleSiliconGeneration::from_device_name("Apple M3 Pro"),
            AppleSiliconGeneration::M3Pro
        );
        assert_eq!(
            AppleSiliconGeneration::from_device_name("Apple M3 Max"),
            AppleSiliconGeneration::M3Max
        );
        assert_eq!(
            AppleSiliconGeneration::from_device_name("Apple M4"),
            AppleSiliconGeneration::M4
        );
        assert_eq!(
            AppleSiliconGeneration::from_device_name("Apple M4 Pro"),
            AppleSiliconGeneration::M4Pro
        );
        assert_eq!(
            AppleSiliconGeneration::from_device_name("Apple M4 Max"),
            AppleSiliconGeneration::M4Max
        );
    }

    #[test]
    fn test_silicon_generation_from_a_series() {
        assert_eq!(
            AppleSiliconGeneration::from_device_name("Apple A14 Bionic"),
            AppleSiliconGeneration::A14
        );
        assert_eq!(
            AppleSiliconGeneration::from_device_name("Apple A15"),
            AppleSiliconGeneration::A15
        );
        assert_eq!(
            AppleSiliconGeneration::from_device_name("Apple A16"),
            AppleSiliconGeneration::A16
        );
        assert_eq!(
            AppleSiliconGeneration::from_device_name("Apple A17 Pro"),
            AppleSiliconGeneration::A17Pro
        );
    }

    #[test]
    fn test_silicon_generation_number() {
        assert_eq!(AppleSiliconGeneration::M1.generation_number(), 1);
        assert_eq!(AppleSiliconGeneration::M1Ultra.generation_number(), 1);
        assert_eq!(AppleSiliconGeneration::M2.generation_number(), 2);
        assert_eq!(AppleSiliconGeneration::M3Max.generation_number(), 3);
        assert_eq!(AppleSiliconGeneration::M4.generation_number(), 4);
        assert_eq!(AppleSiliconGeneration::A17Pro.generation_number(), 0);
        assert_eq!(AppleSiliconGeneration::Unknown.generation_number(), 0);
    }

    #[test]
    fn test_silicon_generation_is_m_series() {
        assert!(AppleSiliconGeneration::M1.is_m_series());
        assert!(AppleSiliconGeneration::M2Pro.is_m_series());
        assert!(AppleSiliconGeneration::M3Max.is_m_series());
        assert!(AppleSiliconGeneration::M4.is_m_series());
        assert!(!AppleSiliconGeneration::A17Pro.is_m_series());
        assert!(!AppleSiliconGeneration::Unknown.is_m_series());
    }

    #[test]
    fn test_silicon_generation_is_a_series() {
        assert!(AppleSiliconGeneration::A14.is_a_series());
        assert!(AppleSiliconGeneration::A15.is_a_series());
        assert!(AppleSiliconGeneration::A16.is_a_series());
        assert!(AppleSiliconGeneration::A17Pro.is_a_series());
        assert!(!AppleSiliconGeneration::M1.is_a_series());
        assert!(!AppleSiliconGeneration::Unknown.is_a_series());
    }

    #[test]
    fn test_silicon_generation_is_pro_tier() {
        assert!(AppleSiliconGeneration::M1Pro.is_pro_tier());
        assert!(AppleSiliconGeneration::M1Max.is_pro_tier());
        assert!(AppleSiliconGeneration::M1Ultra.is_pro_tier());
        assert!(AppleSiliconGeneration::M2Pro.is_pro_tier());
        assert!(AppleSiliconGeneration::M3Max.is_pro_tier());
        assert!(AppleSiliconGeneration::A17Pro.is_pro_tier());
        assert!(!AppleSiliconGeneration::M1.is_pro_tier());
        assert!(!AppleSiliconGeneration::M2.is_pro_tier());
        assert!(!AppleSiliconGeneration::A16.is_pro_tier());
    }

    #[test]
    fn test_silicon_generation_estimated_gpu_cores() {
        assert_eq!(AppleSiliconGeneration::M1.estimated_gpu_cores(), 8);
        assert_eq!(AppleSiliconGeneration::M1Pro.estimated_gpu_cores(), 16);
        assert_eq!(AppleSiliconGeneration::M1Max.estimated_gpu_cores(), 32);
        assert_eq!(AppleSiliconGeneration::M1Ultra.estimated_gpu_cores(), 64);
        assert_eq!(AppleSiliconGeneration::M3Max.estimated_gpu_cores(), 40);
        assert_eq!(AppleSiliconGeneration::Unknown.estimated_gpu_cores(), 0);
    }

    #[test]
    fn test_silicon_generation_name() {
        assert_eq!(AppleSiliconGeneration::M1.name(), "M1");
        assert_eq!(AppleSiliconGeneration::M3Pro.name(), "M3 Pro");
        assert_eq!(AppleSiliconGeneration::A17Pro.name(), "A17 Pro");
        assert_eq!(AppleSiliconGeneration::Unknown.name(), "Unknown");
    }

    #[test]
    fn test_silicon_generation_display() {
        assert_eq!(format!("{}", AppleSiliconGeneration::M3Max), "M3 Max");
        assert_eq!(format!("{}", AppleSiliconGeneration::A17Pro), "A17 Pro");
    }

    // ========================================================================
    // MetalFeatures Tests
    // ========================================================================

    #[test]
    fn test_metal_features_default() {
        let features = MetalFeatures::default();
        assert_eq!(features.gpu_family, MetalGpuFamily::Unknown);
        assert_eq!(features.silicon_generation, AppleSiliconGeneration::Unknown);
        assert_eq!(features.argument_buffers_tier, 0);
        assert!(!features.ray_tracing);
        assert!(!features.mesh_shaders);
    }

    #[test]
    fn test_metal_features_from_m1() {
        let features = MetalFeatures::from_device_info("Apple M1");
        assert_eq!(features.gpu_family, MetalGpuFamily::Apple7);
        assert_eq!(features.silicon_generation, AppleSiliconGeneration::M1);
        assert_eq!(features.argument_buffers_tier, 2);
        assert!(features.ray_tracing);
        assert!(features.mesh_shaders);
        assert!(features.memoryless_render_targets);
        assert!(features.lossless_compression);
    }

    #[test]
    fn test_metal_features_from_m3_max() {
        let features = MetalFeatures::from_device_info("Apple M3 Max");
        assert_eq!(features.gpu_family, MetalGpuFamily::Apple9);
        assert_eq!(features.silicon_generation, AppleSiliconGeneration::M3Max);
        assert_eq!(features.argument_buffers_tier, 2);
        assert!(features.ray_tracing);
        assert!(features.mesh_shaders);
        assert!(features.primitive_motion_blur);
        assert!(features.function_pointers);
    }

    #[test]
    fn test_metal_features_from_intel() {
        let features = MetalFeatures::from_device_info("Intel UHD Graphics 630");
        assert_eq!(features.gpu_family, MetalGpuFamily::Mac2);
        assert_eq!(features.silicon_generation, AppleSiliconGeneration::Unknown);
        assert!(!features.ray_tracing);
        assert!(!features.mesh_shaders);
        assert!(features.simd_group);
    }

    #[test]
    fn test_metal_features_supports_rt() {
        let m1 = MetalFeatures::from_device_info("Apple M1");
        assert!(m1.supports_rt());

        let intel = MetalFeatures::from_device_info("Intel UHD");
        assert!(!intel.supports_rt());
    }

    #[test]
    fn test_metal_features_supports_mesh_shaders() {
        let m3 = MetalFeatures::from_device_info("Apple M3");
        assert!(m3.supports_mesh_shaders());

        let a13 = MetalFeatures::from_device_info("Apple A13");
        assert!(!a13.supports_mesh_shaders());
    }

    #[test]
    fn test_metal_features_supports_argument_buffers() {
        let m1 = MetalFeatures::from_device_info("Apple M1");
        assert!(m1.supports_argument_buffers());
        assert!(m1.supports_bindless());

        let default = MetalFeatures::default();
        assert!(!default.supports_argument_buffers());
        assert!(!default.supports_bindless());
    }

    #[test]
    fn test_metal_features_is_m_series() {
        let m1 = MetalFeatures::from_device_info("Apple M1");
        assert!(m1.is_m_series());

        let a15 = MetalFeatures::from_device_info("Apple A15");
        assert!(!a15.is_m_series());
    }

    #[test]
    fn test_metal_features_minimum_macos_version_m3() {
        let m3 = MetalFeatures::from_device_info("Apple M3");
        assert_eq!(m3.minimum_macos_version(), (14, 0));
    }

    #[test]
    fn test_metal_features_minimum_macos_version_m1() {
        let m1 = MetalFeatures::from_device_info("Apple M1");
        // M1 with RT requires Ventura
        assert_eq!(m1.minimum_macos_version(), (13, 0));
    }

    #[test]
    fn test_metal_features_summary() {
        let m3 = MetalFeatures::from_device_info("Apple M3 Pro");
        let summary = m3.summary();
        assert!(summary.contains("Apple9"));
        assert!(summary.contains("M3 Pro"));
        assert!(summary.contains("RT"));
        assert!(summary.contains("Mesh"));
        assert!(summary.contains("ArgBuf-T2"));
    }

    #[test]
    fn test_metal_features_summary_intel() {
        let intel = MetalFeatures::from_device_info("Intel UHD Graphics 630");
        let summary = intel.summary();
        assert!(summary.contains("Mac2"));
        assert!(!summary.contains("RT"));
        assert!(!summary.contains("Mesh"));
    }

    // ========================================================================
    // Integration-style Tests
    // ========================================================================

    #[test]
    fn test_m4_has_all_latest_features() {
        let m4 = MetalFeatures::from_device_info("Apple M4 Max");

        // M4 should have all modern features
        assert!(m4.supports_rt());
        assert!(m4.supports_mesh_shaders());
        assert!(m4.supports_bindless());
        assert!(m4.lossless_compression);
        assert!(m4.primitive_motion_blur);
        assert!(m4.function_pointers);
        assert!(m4.sparse_textures);
        assert!(m4.tile_shaders);
        assert!(m4.imageblock);
    }

    #[test]
    fn test_feature_progression_a_series() {
        // Test that features increase with A-series generations
        let a12 = MetalFeatures::from_device_info("Apple A12");
        let a14 = MetalFeatures::from_device_info("Apple A14");
        let a17 = MetalFeatures::from_device_info("Apple A17 Pro");

        // A12 doesn't have RT
        assert!(!a12.supports_rt());
        // A14 has RT
        assert!(a14.supports_rt());
        // A17 has full Metal 3
        assert!(a17.supports_rt());
        assert!(a17.supports_mesh_shaders());
    }

    #[test]
    fn test_case_insensitive_parsing() {
        let lower = MetalFeatures::from_device_info("apple m3 pro");
        let upper = MetalFeatures::from_device_info("APPLE M3 PRO");
        let mixed = MetalFeatures::from_device_info("Apple M3 Pro");

        assert_eq!(lower.gpu_family, upper.gpu_family);
        assert_eq!(lower.gpu_family, mixed.gpu_family);
        assert_eq!(lower.silicon_generation, upper.silicon_generation);
    }

    // ========================================================================
    // AppleGpuFamily Tests (T-WGPU-P7.2.2)
    // ========================================================================

    #[test]
    fn test_apple_gpu_family_ordering() {
        // Test that families are ordered by capability
        assert!(AppleGpuFamily::Apple1 < AppleGpuFamily::Apple2);
        assert!(AppleGpuFamily::Apple2 < AppleGpuFamily::Apple3);
        assert!(AppleGpuFamily::Apple7 < AppleGpuFamily::Apple8);
        assert!(AppleGpuFamily::Apple8 < AppleGpuFamily::Apple9);
        assert!(AppleGpuFamily::Common1 < AppleGpuFamily::Common2);
        assert!(AppleGpuFamily::Mac1 < AppleGpuFamily::Mac2);
    }

    #[test]
    fn test_apple_gpu_family_ray_tracing() {
        // Ray tracing requires Apple7+
        assert!(AppleGpuFamily::Apple7.supports_ray_tracing());
        assert!(AppleGpuFamily::Apple8.supports_ray_tracing());
        assert!(AppleGpuFamily::Apple9.supports_ray_tracing());

        // Older families don't support RT
        assert!(!AppleGpuFamily::Apple1.supports_ray_tracing());
        assert!(!AppleGpuFamily::Apple6.supports_ray_tracing());
        assert!(!AppleGpuFamily::Mac1.supports_ray_tracing());
        assert!(!AppleGpuFamily::Mac2.supports_ray_tracing());
        assert!(!AppleGpuFamily::Common3.supports_ray_tracing());
    }

    #[test]
    fn test_apple_gpu_family_mesh_shaders() {
        // Mesh shaders require Apple7+
        assert!(AppleGpuFamily::Apple7.supports_mesh_shaders());
        assert!(AppleGpuFamily::Apple8.supports_mesh_shaders());
        assert!(AppleGpuFamily::Apple9.supports_mesh_shaders());

        // Older families don't support mesh shaders
        assert!(!AppleGpuFamily::Apple1.supports_mesh_shaders());
        assert!(!AppleGpuFamily::Apple6.supports_mesh_shaders());
        assert!(!AppleGpuFamily::Mac2.supports_mesh_shaders());
    }

    // ========================================================================
    // MetalVersion Tests (T-WGPU-P7.2.2)
    // ========================================================================

    #[test]
    fn test_metal_version_constants() {
        assert_eq!(MetalVersion::V2_0, MetalVersion { major: 2, minor: 0 });
        assert_eq!(MetalVersion::V2_3, MetalVersion { major: 2, minor: 3 });
        assert_eq!(MetalVersion::V2_4, MetalVersion { major: 2, minor: 4 });
        assert_eq!(MetalVersion::V3_0, MetalVersion { major: 3, minor: 0 });
        assert_eq!(MetalVersion::V3_1, MetalVersion { major: 3, minor: 1 });
    }

    #[test]
    fn test_metal_version_ordering() {
        assert!(MetalVersion::V2_0 < MetalVersion::V2_3);
        assert!(MetalVersion::V2_3 < MetalVersion::V2_4);
        assert!(MetalVersion::V2_4 < MetalVersion::V3_0);
        assert!(MetalVersion::V3_0 < MetalVersion::V3_1);

        // Test with new() constructor
        let v2_1 = MetalVersion::new(2, 1);
        let v2_2 = MetalVersion::new(2, 2);
        assert!(v2_1 < v2_2);
        assert!(MetalVersion::V2_0 < v2_1);
        assert!(v2_2 < MetalVersion::V2_3);
    }

    // ========================================================================
    // MetalCapabilities Tests (T-WGPU-P7.2.2)
    // ========================================================================

    #[test]
    fn test_metal_capabilities_default() {
        let caps = MetalCapabilities::default();
        assert!(caps.gpu_family.is_none());
        assert!(!caps.unified_memory);
        assert!(!caps.argument_buffers);
        assert!(!caps.argument_buffers_tier2);
        assert!(!caps.ray_tracing);
        assert!(!caps.mesh_shaders);
        assert!(!caps.simd_groups);
        assert!(!caps.is_ios);
        assert!(!caps.is_macos);
        assert!(!caps.is_simulator);
    }

    #[test]
    fn test_metal_capabilities_unified_memory() {
        // Create capabilities for Apple Silicon
        let mut caps = MetalCapabilities::default();
        caps.gpu_family = Some(AppleGpuFamily::Apple7);
        caps.unified_memory = true;

        assert!(caps.is_apple_silicon());
        assert!(caps.unified_memory);
    }

    #[test]
    fn test_metal_capabilities_argument_buffers() {
        let mut caps = MetalCapabilities::default();
        caps.argument_buffers = true;
        caps.argument_buffers_tier2 = false;
        assert!(caps.argument_buffers);
        assert!(!caps.argument_buffers_tier2);

        caps.argument_buffers_tier2 = true;
        assert!(caps.argument_buffers);
        assert!(caps.argument_buffers_tier2);
    }

    #[test]
    fn test_metal_capabilities_platform_ios() {
        let mut caps = MetalCapabilities::default();
        caps.is_ios = true;
        caps.is_macos = false;

        let limits = caps.platform_limits();
        assert_eq!(limits.max_vertex_buffers, PlatformLimits::ios().max_vertex_buffers);
        assert_eq!(limits.max_compute_threads, PlatformLimits::ios().max_compute_threads);
    }

    #[test]
    fn test_metal_capabilities_platform_macos() {
        let mut caps = MetalCapabilities::default();
        caps.is_ios = false;
        caps.is_macos = true;

        let limits = caps.platform_limits();
        assert_eq!(limits.max_vertex_buffers, PlatformLimits::macos().max_vertex_buffers);
        assert_eq!(limits.max_compute_threads, PlatformLimits::macos().max_compute_threads);
    }

    // ========================================================================
    // MetalInfo Tests (T-WGPU-P7.2.2)
    // ========================================================================

    #[test]
    fn test_metal_info_creation() {
        // Test creating MetalInfo directly
        let caps = MetalCapabilities {
            gpu_family: Some(AppleGpuFamily::Apple7),
            unified_memory: true,
            max_buffer_length: 1024 * 1024 * 1024,
            max_texture_size: 16384,
            argument_buffers: true,
            argument_buffers_tier2: true,
            ray_tracing: true,
            mesh_shaders: true,
            simd_groups: true,
            is_ios: false,
            is_macos: true,
            is_simulator: false,
        };

        let info = MetalInfo {
            version: MetalVersion::V3_0,
            capabilities: caps,
            device_name: "Apple M1".to_string(),
            registry_id: 12345,
        };

        assert_eq!(info.version, MetalVersion::V3_0);
        assert_eq!(info.device_name, "Apple M1");
        assert_eq!(info.registry_id, 12345);
    }

    #[test]
    fn test_metal_info_bindless() {
        let mut caps = MetalCapabilities::default();

        // Without tier 2, no bindless
        caps.argument_buffers_tier2 = false;
        let info = MetalInfo {
            version: MetalVersion::V2_0,
            capabilities: caps.clone(),
            device_name: "Test".to_string(),
            registry_id: 0,
        };
        assert!(!info.supports_bindless());

        // With tier 2, bindless available
        caps.argument_buffers_tier2 = true;
        let info2 = MetalInfo {
            version: MetalVersion::V3_0,
            capabilities: caps,
            device_name: "Test".to_string(),
            registry_id: 0,
        };
        assert!(info2.supports_bindless());
    }

    // ========================================================================
    // PlatformLimits Tests (T-WGPU-P7.2.2)
    // ========================================================================

    #[test]
    fn test_platform_limits_ios() {
        let limits = PlatformLimits::ios();
        assert_eq!(limits.max_vertex_buffers, 31);
        assert_eq!(limits.max_fragment_inputs, 60);
        assert_eq!(limits.max_compute_threads, 512);
    }

    #[test]
    fn test_platform_limits_macos() {
        let limits = PlatformLimits::macos();
        assert_eq!(limits.max_vertex_buffers, 31);
        assert_eq!(limits.max_fragment_inputs, 124);
        assert_eq!(limits.max_compute_threads, 1024);
    }

    #[test]
    fn test_is_apple_silicon() {
        // Apple Silicon: unified memory + Apple7/8/9
        let mut caps = MetalCapabilities::default();
        caps.unified_memory = true;
        caps.gpu_family = Some(AppleGpuFamily::Apple7);
        assert!(caps.is_apple_silicon());

        caps.gpu_family = Some(AppleGpuFamily::Apple8);
        assert!(caps.is_apple_silicon());

        caps.gpu_family = Some(AppleGpuFamily::Apple9);
        assert!(caps.is_apple_silicon());

        // Not Apple Silicon: Intel/AMD (no unified memory)
        caps.unified_memory = false;
        caps.gpu_family = Some(AppleGpuFamily::Mac2);
        assert!(!caps.is_apple_silicon());
    }

    #[test]
    fn test_is_discrete() {
        let mut caps = MetalCapabilities::default();

        // Intel Mac is discrete
        caps.gpu_family = Some(AppleGpuFamily::Mac1);
        assert!(caps.is_discrete());

        caps.gpu_family = Some(AppleGpuFamily::Mac2);
        assert!(caps.is_discrete());

        // Apple Silicon is not discrete
        caps.gpu_family = Some(AppleGpuFamily::Apple7);
        assert!(!caps.is_discrete());

        caps.gpu_family = Some(AppleGpuFamily::Apple9);
        assert!(!caps.is_discrete());
    }

    #[test]
    fn test_gpu_family_min_metal_version() {
        // Apple9 requires Metal 3.1
        assert_eq!(AppleGpuFamily::Apple9.min_metal_version(), MetalVersion::V3_1);

        // Apple7/8 require Metal 3.0
        assert_eq!(AppleGpuFamily::Apple7.min_metal_version(), MetalVersion::V3_0);
        assert_eq!(AppleGpuFamily::Apple8.min_metal_version(), MetalVersion::V3_0);

        // Apple5/6 require Metal 2.3
        assert_eq!(AppleGpuFamily::Apple5.min_metal_version(), MetalVersion::V2_3);
        assert_eq!(AppleGpuFamily::Apple6.min_metal_version(), MetalVersion::V2_3);

        // Older families require Metal 2.0
        assert_eq!(AppleGpuFamily::Apple1.min_metal_version(), MetalVersion::V2_0);
        assert_eq!(AppleGpuFamily::Apple3.min_metal_version(), MetalVersion::V2_0);
    }

    #[test]
    fn test_simd_groups_support() {
        // Apple3+ supports SIMD groups
        assert!(AppleGpuFamily::Apple3.supports_simd_groups());
        assert!(AppleGpuFamily::Apple7.supports_simd_groups());
        assert!(AppleGpuFamily::Apple9.supports_simd_groups());
        assert!(AppleGpuFamily::Mac1.supports_simd_groups());
        assert!(AppleGpuFamily::Mac2.supports_simd_groups());

        // Apple1/2 don't support SIMD groups
        assert!(!AppleGpuFamily::Apple1.supports_simd_groups());
        assert!(!AppleGpuFamily::Apple2.supports_simd_groups());

        // Common families don't have SIMD groups
        assert!(!AppleGpuFamily::Common1.supports_simd_groups());
    }

    // ========================================================================
    // Additional Integration Tests
    // ========================================================================

    #[test]
    fn test_metal_version_features() {
        // Metal 3.0+ supports RT and mesh shaders
        assert!(MetalVersion::V3_0.supports_ray_tracing());
        assert!(MetalVersion::V3_0.supports_mesh_shaders());
        assert!(MetalVersion::V3_1.supports_ray_tracing());
        assert!(MetalVersion::V3_1.supports_mesh_shaders());

        // Metal 2.x doesn't support RT/mesh
        assert!(!MetalVersion::V2_0.supports_ray_tracing());
        assert!(!MetalVersion::V2_0.supports_mesh_shaders());
        assert!(!MetalVersion::V2_4.supports_ray_tracing());
        assert!(!MetalVersion::V2_4.supports_mesh_shaders());

        // Metal 2.0+ supports argument buffers tier 2
        assert!(MetalVersion::V2_0.supports_argument_buffers_tier2());
        assert!(MetalVersion::V3_0.supports_argument_buffers_tier2());
    }

    #[test]
    fn test_metal_version_display() {
        assert_eq!(format!("{}", MetalVersion::V2_0), "Metal 2.0");
        assert_eq!(format!("{}", MetalVersion::V3_0), "Metal 3.0");
        assert_eq!(format!("{}", MetalVersion::V3_1), "Metal 3.1");
    }

    #[test]
    fn test_apple_gpu_family_display() {
        assert_eq!(format!("{}", AppleGpuFamily::Apple7), "Apple7");
        assert_eq!(format!("{}", AppleGpuFamily::Apple9), "Apple9");
        assert_eq!(format!("{}", AppleGpuFamily::Mac2), "Mac2");
    }

    #[test]
    fn test_metal_capabilities_gpu_family_detection() {
        // Test the static detection function
        let family_m1 = MetalCapabilities::detect_gpu_family("Apple M1");
        assert_eq!(family_m1, Some(AppleGpuFamily::Apple7));

        let family_m3 = MetalCapabilities::detect_gpu_family("Apple M3 Pro");
        assert_eq!(family_m3, Some(AppleGpuFamily::Apple9));

        let family_a14 = MetalCapabilities::detect_gpu_family("Apple A14 Bionic");
        assert_eq!(family_a14, Some(AppleGpuFamily::Apple7));

        let family_intel = MetalCapabilities::detect_gpu_family("Intel UHD Graphics 630");
        assert_eq!(family_intel, Some(AppleGpuFamily::Mac2));
    }
}
