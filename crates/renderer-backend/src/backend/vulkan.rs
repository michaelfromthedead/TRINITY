//! Vulkan-specific feature detection for TRINITY.
//!
//! This module provides detection of Vulkan extensions and features that
//! go beyond what wgpu exposes directly. While wgpu provides a unified API,
//! some Vulkan-specific capabilities require explicit detection for optimal
//! performance and feature usage.
//!
//! # Detected Extensions
//!
//! | Extension | Field | Description |
//! |-----------|-------|-------------|
//! | VK_KHR_ray_tracing_pipeline | `ray_tracing` | Full RT pipeline with hit/miss shaders |
//! | VK_KHR_ray_query | `ray_query` | Inline ray tracing in any shader |
//! | VK_EXT_descriptor_indexing | `descriptor_indexing` | Bindless resource access |
//! | VK_KHR_timeline_semaphore | `timeline_semaphores` | GPU timeline synchronization |
//! | VK_KHR_buffer_device_address | `buffer_device_address` | GPU pointer support |
//! | VK_EXT_mesh_shader | `mesh_shading` | Mesh/task shader pipeline |
//! | VK_KHR_dynamic_rendering | `dynamic_rendering` | Renderpass-less rendering |
//!
//! # Example
//!
//! ```no_run
//! use renderer_backend::backend::VulkanFeatures;
//!
//! # async fn example() {
//! let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
//!     backends: wgpu::Backends::VULKAN,
//!     ..Default::default()
//! });
//!
//! let adapter = instance
//!     .request_adapter(&wgpu::RequestAdapterOptions::default())
//!     .await
//!     .unwrap();
//!
//! let vk_features = VulkanFeatures::detect(&adapter);
//!
//! if vk_features.supports_rt_pipeline() {
//!     println!("Full Vulkan ray tracing pipeline available!");
//!     println!("  - Ray tracing: {}", vk_features.ray_tracing);
//!     println!("  - Ray query: {}", vk_features.ray_query);
//!     println!("  - Buffer device address: {}", vk_features.buffer_device_address);
//! }
//!
//! if vk_features.supports_bindless() {
//!     println!("Bindless resources available!");
//! }
//! # }
//! ```
//!
//! # Ray Tracing Tiers
//!
//! Vulkan ray tracing support is classified into tiers:
//!
//! - **Full**: `VK_KHR_ray_tracing_pipeline` + all dependencies
//! - **Query**: `VK_KHR_ray_query` only (inline RT in any shader)
//! - **None**: No ray tracing support
//!
//! # Vulkan Version
//!
//! The `VulkanVersion` struct provides version comparison and raw encoding:
//!
//! ```
//! use renderer_backend::backend::vulkan::VulkanVersion;
//!
//! let v1_2 = VulkanVersion::V1_2;
//! let v1_3 = VulkanVersion::V1_3;
//!
//! assert!(v1_3 > v1_2);
//! assert_eq!(v1_2.to_raw(), 0x00402000); // Vulkan version encoding
//! ```
//!
//! # VulkanInfo
//!
//! The `VulkanInfo` struct provides comprehensive adapter information:
//!
//! ```no_run
//! use renderer_backend::backend::vulkan::{VulkanInfo, VulkanDeviceType};
//!
//! # async fn example() {
//! # let instance = wgpu::Instance::default();
//! # let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.unwrap();
//! let info = VulkanInfo::from_adapter(&adapter);
//! println!("Device: {} ({:?})", info.device_name, info.device_type);
//! println!("Driver: {} v{}", info.driver_name, info.driver_version);
//! println!("Vulkan: {}", info.version);
//! # }
//! ```

use log::debug;
use std::ffi::c_void;
use wgpu::{Adapter, Features};

// ============================================================================
// VulkanVersion
// ============================================================================

/// Vulkan API version.
///
/// Represents a Vulkan version number with major, minor, and patch components.
/// Versions are comparable and can be converted to/from the raw Vulkan encoding.
///
/// # Vulkan Version Encoding
///
/// Vulkan encodes versions as a 32-bit integer:
/// - Bits 31-22: Major version (10 bits)
/// - Bits 21-12: Minor version (10 bits)
/// - Bits 11-0: Patch version (12 bits)
///
/// # Example
///
/// ```
/// use renderer_backend::backend::vulkan::VulkanVersion;
///
/// let v1_2 = VulkanVersion::V1_2;
/// let v1_3 = VulkanVersion::V1_3;
///
/// assert!(v1_3 > v1_2);
/// assert_eq!(v1_2.major, 1);
/// assert_eq!(v1_2.minor, 2);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VulkanVersion {
    /// Major version number.
    pub major: u32,
    /// Minor version number.
    pub minor: u32,
    /// Patch version number.
    pub patch: u32,
}

impl VulkanVersion {
    /// Vulkan 1.0
    pub const V1_0: Self = Self { major: 1, minor: 0, patch: 0 };
    /// Vulkan 1.1
    pub const V1_1: Self = Self { major: 1, minor: 1, patch: 0 };
    /// Vulkan 1.2
    pub const V1_2: Self = Self { major: 1, minor: 2, patch: 0 };
    /// Vulkan 1.3
    pub const V1_3: Self = Self { major: 1, minor: 3, patch: 0 };

    /// Create a new Vulkan version.
    ///
    /// # Arguments
    ///
    /// * `major` - Major version number
    /// * `minor` - Minor version number
    /// * `patch` - Patch version number
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::vulkan::VulkanVersion;
    ///
    /// let version = VulkanVersion::new(1, 3, 250);
    /// assert_eq!(version.major, 1);
    /// assert_eq!(version.minor, 3);
    /// assert_eq!(version.patch, 250);
    /// ```
    #[inline]
    pub const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self { major, minor, patch }
    }

    /// Decode a Vulkan version from its raw 32-bit encoding.
    ///
    /// # Arguments
    ///
    /// * `raw` - The raw Vulkan version integer
    ///
    /// # Returns
    ///
    /// The decoded version.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::vulkan::VulkanVersion;
    ///
    /// // Vulkan 1.2.0 raw encoding
    /// let version = VulkanVersion::from_raw(0x00402000);
    /// assert_eq!(version, VulkanVersion::V1_2);
    /// ```
    #[inline]
    pub const fn from_raw(raw: u32) -> Self {
        Self {
            major: (raw >> 22) & 0x3FF,
            minor: (raw >> 12) & 0x3FF,
            patch: raw & 0xFFF,
        }
    }

    /// Encode the version to its raw 32-bit Vulkan representation.
    ///
    /// # Returns
    ///
    /// The raw Vulkan version integer.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::vulkan::VulkanVersion;
    ///
    /// let version = VulkanVersion::V1_2;
    /// assert_eq!(version.to_raw(), 0x00402000);
    /// ```
    #[inline]
    pub const fn to_raw(&self) -> u32 {
        ((self.major & 0x3FF) << 22) | ((self.minor & 0x3FF) << 12) | (self.patch & 0xFFF)
    }

    /// Check if this version is at least the specified version.
    ///
    /// # Arguments
    ///
    /// * `major` - Minimum major version
    /// * `minor` - Minimum minor version
    ///
    /// # Returns
    ///
    /// `true` if this version meets or exceeds the minimum.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::vulkan::VulkanVersion;
    ///
    /// let version = VulkanVersion::V1_3;
    /// assert!(version.is_at_least(1, 2));
    /// assert!(version.is_at_least(1, 3));
    /// assert!(!version.is_at_least(1, 4));
    /// ```
    #[inline]
    pub const fn is_at_least(&self, major: u32, minor: u32) -> bool {
        self.major > major || (self.major == major && self.minor >= minor)
    }
}

impl Default for VulkanVersion {
    fn default() -> Self {
        Self::V1_0
    }
}

impl PartialOrd for VulkanVersion {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for VulkanVersion {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.to_raw().cmp(&other.to_raw())
    }
}

impl std::fmt::Display for VulkanVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

// ============================================================================
// VulkanDeviceType
// ============================================================================

/// Vulkan physical device type.
///
/// Corresponds to `VkPhysicalDeviceType` in the Vulkan specification.
/// The device type indicates the nature of the GPU hardware.
///
/// # Example
///
/// ```
/// use renderer_backend::backend::vulkan::VulkanDeviceType;
///
/// let device_type = VulkanDeviceType::DiscreteGpu;
/// assert!(device_type.is_gpu());
/// assert!(device_type.is_hardware());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum VulkanDeviceType {
    /// Discrete GPU (separate from CPU).
    ///
    /// Typically a dedicated graphics card with its own memory.
    /// Usually provides the best performance.
    DiscreteGpu,

    /// Integrated GPU (on the same die as CPU).
    ///
    /// Shares system memory with the CPU. Lower power but also
    /// lower performance than discrete GPUs.
    IntegratedGpu,

    /// Virtual GPU (virtualized environment).
    ///
    /// A GPU exposed through virtualization layers like VirtualBox,
    /// VMware, or cloud GPU instances.
    VirtualGpu,

    /// CPU-based software rendering.
    ///
    /// Software implementation of Vulkan running on the CPU.
    /// Very slow but useful for testing and fallback.
    Cpu,

    /// Unknown or unrecognized device type.
    #[default]
    Other,
}

impl VulkanDeviceType {
    /// Create from wgpu's DeviceType.
    ///
    /// # Arguments
    ///
    /// * `device_type` - The wgpu device type
    ///
    /// # Returns
    ///
    /// The corresponding Vulkan device type.
    pub fn from_wgpu(device_type: wgpu::DeviceType) -> Self {
        match device_type {
            wgpu::DeviceType::DiscreteGpu => Self::DiscreteGpu,
            wgpu::DeviceType::IntegratedGpu => Self::IntegratedGpu,
            wgpu::DeviceType::VirtualGpu => Self::VirtualGpu,
            wgpu::DeviceType::Cpu => Self::Cpu,
            wgpu::DeviceType::Other => Self::Other,
        }
    }

    /// Check if this is a hardware GPU (discrete or integrated).
    ///
    /// # Returns
    ///
    /// `true` for discrete or integrated GPUs.
    #[inline]
    pub const fn is_gpu(&self) -> bool {
        matches!(self, Self::DiscreteGpu | Self::IntegratedGpu | Self::VirtualGpu)
    }

    /// Check if this is real hardware (not software/CPU).
    ///
    /// # Returns
    ///
    /// `true` for discrete, integrated, or virtual GPUs.
    #[inline]
    pub const fn is_hardware(&self) -> bool {
        !matches!(self, Self::Cpu | Self::Other)
    }

    /// Get the device type name.
    ///
    /// # Returns
    ///
    /// Human-readable device type name.
    #[inline]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::DiscreteGpu => "Discrete GPU",
            Self::IntegratedGpu => "Integrated GPU",
            Self::VirtualGpu => "Virtual GPU",
            Self::Cpu => "CPU",
            Self::Other => "Other",
        }
    }
}

impl std::fmt::Display for VulkanDeviceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ============================================================================
// VulkanInfo
// ============================================================================

/// Comprehensive Vulkan adapter information.
///
/// This struct provides detailed information about a Vulkan adapter including
/// version, device type, driver information, and detected features.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::backend::vulkan::VulkanInfo;
///
/// # async fn example() {
/// let instance = wgpu::Instance::default();
/// let adapter = instance
///     .request_adapter(&wgpu::RequestAdapterOptions::default())
///     .await
///     .unwrap();
///
/// let info = VulkanInfo::from_adapter(&adapter);
/// println!("Device: {} ({:?})", info.device_name, info.device_type);
/// println!("Driver: {} v{}", info.driver_name, info.driver_version);
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct VulkanInfo {
    /// Detected Vulkan API version.
    pub version: VulkanVersion,

    /// Detected Vulkan features.
    pub features: VulkanFeatures,

    /// Driver name (e.g., "NVIDIA", "AMD", "Mesa").
    pub driver_name: String,

    /// Driver version as raw integer.
    ///
    /// The encoding is vendor-specific:
    /// - NVIDIA: (major << 22) | (minor << 14) | (patch << 6) | variant
    /// - AMD/Intel: VK_VERSION encoding
    pub driver_version: u32,

    /// Device name (e.g., "GeForce RTX 4090").
    pub device_name: String,

    /// Physical device type.
    pub device_type: VulkanDeviceType,

    /// Vendor ID (PCI vendor ID).
    pub vendor_id: u32,

    /// Device ID (PCI device ID).
    pub device_id: u32,
}

impl VulkanInfo {
    /// Create Vulkan info from a wgpu adapter.
    ///
    /// # Arguments
    ///
    /// * `adapter` - The wgpu adapter to query
    ///
    /// # Returns
    ///
    /// The detected Vulkan information.
    pub fn from_adapter(adapter: &Adapter) -> Self {
        let info = adapter.get_info();
        let features = VulkanFeatures::detect(adapter);

        // Infer Vulkan version from features
        let version = if features.supports_vulkan_1_3() {
            VulkanVersion::V1_3
        } else if features.supports_vulkan_1_2() {
            VulkanVersion::V1_2
        } else if features.supports_any_rt() || features.descriptor_indexing {
            VulkanVersion::V1_1
        } else {
            VulkanVersion::V1_0
        };

        Self {
            version,
            features,
            driver_name: info.driver.clone(),
            driver_version: info.driver_info.parse().unwrap_or(0),
            device_name: info.name.clone(),
            device_type: VulkanDeviceType::from_wgpu(info.device_type),
            vendor_id: info.vendor,
            device_id: info.device,
        }
    }

    /// Create a summary string of the Vulkan info.
    ///
    /// # Returns
    ///
    /// A human-readable summary string.
    pub fn summary(&self) -> String {
        format!(
            "{} ({}) - {} - Vulkan {}",
            self.device_name,
            self.device_type,
            self.driver_name,
            self.version
        )
    }

    /// Check if the device is suitable for TRINITY's requirements.
    ///
    /// TRINITY requires at least Vulkan 1.2 features for optimal operation.
    ///
    /// # Returns
    ///
    /// `true` if the device meets TRINITY's minimum requirements.
    #[inline]
    pub fn is_suitable(&self) -> bool {
        self.device_type.is_hardware() && self.features.supports_vulkan_1_2()
    }

    /// Check if the device supports ray tracing.
    ///
    /// # Returns
    ///
    /// `true` if any ray tracing capability is available.
    #[inline]
    pub fn supports_ray_tracing(&self) -> bool {
        self.features.supports_any_rt()
    }
}

impl Default for VulkanInfo {
    fn default() -> Self {
        Self {
            version: VulkanVersion::default(),
            features: VulkanFeatures::default(),
            driver_name: String::new(),
            driver_version: 0,
            device_name: String::new(),
            device_type: VulkanDeviceType::default(),
            vendor_id: 0,
            device_id: 0,
        }
    }
}

// ============================================================================
// VulkanRayTracingTier
// ============================================================================

/// Vulkan ray tracing capability tier.
///
/// Ray tracing in Vulkan comes in different capability levels:
///
/// - **Full**: Complete ray tracing pipeline with ray generation, hit/miss,
///   and callable shaders. Requires `VK_KHR_ray_tracing_pipeline`.
///
/// - **Query**: Inline ray tracing within any shader stage using ray queries.
///   Requires `VK_KHR_ray_query`. Simpler but less flexible than full RT.
///
/// - **None**: No ray tracing support available.
///
/// # Example
///
/// ```
/// use renderer_backend::backend::{VulkanFeatures, VulkanRayTracingTier};
///
/// # async fn example() {
/// # let instance = wgpu::Instance::default();
/// # let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.unwrap();
/// let features = VulkanFeatures::detect(&adapter);
/// let tier = features.ray_tracing_tier();
///
/// match tier {
///     VulkanRayTracingTier::Full => println!("Using full RT pipeline"),
///     VulkanRayTracingTier::Query => println!("Using inline ray queries"),
///     VulkanRayTracingTier::None => println!("RT not available"),
/// }
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum VulkanRayTracingTier {
    /// No ray tracing support.
    #[default]
    None,

    /// Ray query support only (VK_KHR_ray_query).
    ///
    /// Allows inline ray tracing in any shader stage but does not
    /// provide the full ray tracing pipeline with hit/miss shaders.
    Query,

    /// Full ray tracing pipeline (VK_KHR_ray_tracing_pipeline).
    ///
    /// Complete ray tracing support with ray generation, intersection,
    /// any-hit, closest-hit, miss, and callable shaders.
    Full,
}

impl VulkanRayTracingTier {
    /// Check if any ray tracing is available.
    #[inline]
    pub const fn is_available(&self) -> bool {
        !matches!(self, VulkanRayTracingTier::None)
    }

    /// Check if full ray tracing pipeline is available.
    #[inline]
    pub const fn is_full(&self) -> bool {
        matches!(self, VulkanRayTracingTier::Full)
    }

    /// Get tier name as a string.
    #[inline]
    pub const fn name(&self) -> &'static str {
        match self {
            VulkanRayTracingTier::None => "None",
            VulkanRayTracingTier::Query => "Ray Query",
            VulkanRayTracingTier::Full => "Full Pipeline",
        }
    }
}

impl std::fmt::Display for VulkanRayTracingTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ============================================================================
// VulkanFeatures
// ============================================================================

/// Vulkan-specific feature detection.
///
/// This struct captures Vulkan extension availability beyond what wgpu
/// exposes directly. Detection is performed by inspecting wgpu's feature
/// flags which map to underlying Vulkan extensions.
///
/// # Feature Mapping
///
/// | wgpu Feature | Vulkan Extension |
/// |--------------|------------------|
/// | `RAY_TRACING_ACCELERATION_STRUCTURE` | `VK_KHR_acceleration_structure` |
/// | `RAY_QUERY` | `VK_KHR_ray_query` |
/// | `TEXTURE_BINDING_ARRAY` | `VK_EXT_descriptor_indexing` (partial) |
/// | `BUFFER_BINDING_ARRAY` | `VK_EXT_descriptor_indexing` (partial) |
/// | `TIMESTAMP_QUERY` | Vulkan core 1.0 |
///
/// # Example
///
/// ```no_run
/// use renderer_backend::backend::VulkanFeatures;
///
/// # async fn example() {
/// let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
///     backends: wgpu::Backends::VULKAN,
///     ..Default::default()
/// });
/// let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.unwrap();
///
/// let vk = VulkanFeatures::detect(&adapter);
/// println!("Ray tracing: {}", vk.ray_tracing);
/// println!("Descriptor indexing: {}", vk.descriptor_indexing);
/// println!("Timeline semaphores: {}", vk.timeline_semaphores);
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VulkanFeatures {
    /// VK_KHR_ray_tracing_pipeline support.
    ///
    /// Indicates support for the full ray tracing pipeline with ray generation,
    /// intersection, any-hit, closest-hit, miss, and callable shaders.
    ///
    /// Dependencies: `VK_KHR_acceleration_structure`, `VK_KHR_spirv_1_4`,
    /// `VK_KHR_buffer_device_address`.
    pub ray_tracing: bool,

    /// VK_KHR_ray_query support.
    ///
    /// Indicates support for inline ray tracing within any shader stage.
    /// Simpler than full ray tracing pipeline but less flexible.
    ///
    /// Dependencies: `VK_KHR_acceleration_structure`, `VK_KHR_spirv_1_4`.
    pub ray_query: bool,

    /// VK_EXT_descriptor_indexing support.
    ///
    /// Enables bindless resource access with non-uniform indexing into
    /// descriptor arrays. Required for modern GPU-driven rendering.
    ///
    /// Key features:
    /// - Non-uniform indexing into sampled images/storage buffers
    /// - Unbounded descriptor arrays
    /// - Update-after-bind descriptors
    pub descriptor_indexing: bool,

    /// VK_KHR_timeline_semaphore support.
    ///
    /// Timeline semaphores provide GPU timeline synchronization with
    /// monotonically increasing counter values. More flexible than
    /// binary semaphores for complex async operations.
    ///
    /// Note: Promoted to Vulkan 1.2 core.
    pub timeline_semaphores: bool,

    /// VK_KHR_buffer_device_address support.
    ///
    /// Allows shaders to directly access buffer memory via 64-bit addresses.
    /// Required for ray tracing and useful for GPU-driven rendering.
    ///
    /// Note: Promoted to Vulkan 1.2 core.
    pub buffer_device_address: bool,

    /// VK_EXT_mesh_shader support.
    ///
    /// Enables mesh shader and task shader pipeline stages. Mesh shaders
    /// replace the traditional vertex/geometry pipeline with more flexible
    /// compute-like stages.
    pub mesh_shading: bool,

    /// VK_KHR_dynamic_rendering support.
    ///
    /// Allows render pass creation without VkRenderPass/VkFramebuffer objects.
    /// Simplifies rendering API and enables more dynamic render target usage.
    ///
    /// Note: Promoted to Vulkan 1.3 core.
    pub dynamic_rendering: bool,

    /// VK_KHR_synchronization2 support.
    ///
    /// Improved synchronization API with clearer semantics and better
    /// performance characteristics. Replaces legacy barriers.
    ///
    /// Note: Promoted to Vulkan 1.3 core.
    pub synchronization2: bool,

    /// VK_EXT_extended_dynamic_state support.
    ///
    /// Allows more pipeline state to be set dynamically, reducing
    /// pipeline object count and permutation explosion.
    pub extended_dynamic_state: bool,

    /// VK_KHR_maintenance4 support.
    ///
    /// Various quality-of-life improvements for Vulkan 1.3+.
    pub maintenance4: bool,
}

impl VulkanFeatures {
    /// Detect Vulkan features from a wgpu adapter.
    ///
    /// This method queries the adapter's features and maps them to
    /// Vulkan extension availability. Note that wgpu doesn't expose
    /// all Vulkan extensions directly, so some detection is approximate.
    ///
    /// # Arguments
    ///
    /// * `adapter` - The wgpu adapter to query (should be a Vulkan adapter)
    ///
    /// # Returns
    ///
    /// The detected Vulkan features.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::backend::VulkanFeatures;
    ///
    /// # async fn example() {
    /// let instance = wgpu::Instance::default();
    /// let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.unwrap();
    ///
    /// let features = VulkanFeatures::detect(&adapter);
    /// if features.supports_rt_pipeline() {
    ///     println!("RT pipeline available!");
    /// }
    /// # }
    /// ```
    pub fn detect(adapter: &Adapter) -> Self {
        let features = adapter.features();
        Self::from_features(features)
    }

    /// Create Vulkan features from wgpu feature flags.
    ///
    /// This performs the mapping from wgpu's unified feature flags to
    /// Vulkan-specific extension availability.
    ///
    /// # Arguments
    ///
    /// * `features` - The wgpu features from the adapter
    ///
    /// # Returns
    ///
    /// The mapped Vulkan features.
    pub fn from_features(features: Features) -> Self {
        // Ray tracing detection
        let ray_tracing = features.contains(Features::RAY_TRACING_ACCELERATION_STRUCTURE);
        let ray_query = features.contains(Features::RAY_QUERY);

        // Descriptor indexing detection
        // wgpu exposes this through multiple feature flags
        let descriptor_indexing = features.contains(Features::TEXTURE_BINDING_ARRAY)
            && features.contains(Features::BUFFER_BINDING_ARRAY)
            && features.contains(Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING);

        // Buffer device address - inferred from ray tracing support
        // In Vulkan, BDA is required for acceleration structures
        let buffer_device_address = ray_tracing || ray_query;

        // Timeline semaphores - wgpu doesn't expose this directly but uses them internally
        // We assume availability on modern Vulkan drivers (1.2+)
        // This is approximate detection based on ray tracing support as a proxy
        let timeline_semaphores = ray_tracing || ray_query;

        // Mesh shading - wgpu doesn't expose mesh shader feature yet
        // This remains false until wgpu adds mesh shader support
        let mesh_shading = false;

        // Dynamic rendering - assumed available with modern features
        let dynamic_rendering = ray_tracing || ray_query;

        // Synchronization2 - assumed available with modern features
        let synchronization2 = ray_tracing || ray_query;

        // Extended dynamic state - check through texture binding array as proxy
        let extended_dynamic_state = features.contains(Features::TEXTURE_BINDING_ARRAY);

        // Maintenance4 - assume with ray tracing support
        let maintenance4 = ray_tracing;

        let result = Self {
            ray_tracing,
            ray_query,
            descriptor_indexing,
            timeline_semaphores,
            buffer_device_address,
            mesh_shading,
            dynamic_rendering,
            synchronization2,
            extended_dynamic_state,
            maintenance4,
        };

        debug!(
            "VulkanFeatures detected: RT={}, RQ={}, DI={}, TS={}, BDA={}",
            ray_tracing, ray_query, descriptor_indexing, timeline_semaphores, buffer_device_address
        );

        result
    }

    /// Check if full ray tracing pipeline is supported.
    ///
    /// Full RT pipeline requires:
    /// - `VK_KHR_ray_tracing_pipeline` (ray_tracing)
    /// - `VK_KHR_acceleration_structure` (implied by ray_tracing)
    /// - `VK_KHR_buffer_device_address` (buffer_device_address)
    ///
    /// # Returns
    ///
    /// `true` if full ray tracing pipeline is available.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::backend::VulkanFeatures;
    ///
    /// # async fn example() {
    /// # let instance = wgpu::Instance::default();
    /// # let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.unwrap();
    /// let features = VulkanFeatures::detect(&adapter);
    /// if features.supports_rt_pipeline() {
    ///     println!("Can use ray tracing hit/miss shaders!");
    /// }
    /// # }
    /// ```
    #[inline]
    pub const fn supports_rt_pipeline(&self) -> bool {
        self.ray_tracing && self.buffer_device_address
    }

    /// Check if ray query (inline RT) is supported.
    ///
    /// Ray query allows inline ray tracing in any shader stage without
    /// requiring the full ray tracing pipeline.
    ///
    /// # Returns
    ///
    /// `true` if ray query is available.
    #[inline]
    pub const fn supports_ray_query(&self) -> bool {
        self.ray_query
    }

    /// Check if any ray tracing capability is available.
    ///
    /// # Returns
    ///
    /// `true` if either ray tracing pipeline or ray query is available.
    #[inline]
    pub const fn supports_any_rt(&self) -> bool {
        self.ray_tracing || self.ray_query
    }

    /// Check if bindless resource access is supported.
    ///
    /// Bindless rendering requires:
    /// - `VK_EXT_descriptor_indexing` (descriptor_indexing)
    /// - `VK_KHR_buffer_device_address` (buffer_device_address)
    ///
    /// Note: While bindless technically only requires descriptor indexing,
    /// modern bindless rendering typically uses buffer device addresses
    /// for optimal performance.
    ///
    /// # Returns
    ///
    /// `true` if bindless resources are available.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::backend::VulkanFeatures;
    ///
    /// # async fn example() {
    /// # let instance = wgpu::Instance::default();
    /// # let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.unwrap();
    /// let features = VulkanFeatures::detect(&adapter);
    /// if features.supports_bindless() {
    ///     println!("Can use GPU-driven rendering with bindless resources!");
    /// }
    /// # }
    /// ```
    #[inline]
    pub const fn supports_bindless(&self) -> bool {
        self.descriptor_indexing && self.buffer_device_address
    }

    /// Check if modern Vulkan 1.2+ features are available.
    ///
    /// Modern Vulkan includes timeline semaphores and buffer device addresses
    /// which are promoted to core in Vulkan 1.2.
    ///
    /// # Returns
    ///
    /// `true` if Vulkan 1.2+ features are available.
    #[inline]
    pub const fn supports_vulkan_1_2(&self) -> bool {
        self.timeline_semaphores && self.buffer_device_address
    }

    /// Check if Vulkan 1.3+ features are available.
    ///
    /// Vulkan 1.3 includes dynamic rendering and synchronization2 promoted
    /// to core.
    ///
    /// # Returns
    ///
    /// `true` if Vulkan 1.3+ features are available.
    #[inline]
    pub const fn supports_vulkan_1_3(&self) -> bool {
        self.dynamic_rendering && self.synchronization2
    }

    /// Get the ray tracing tier.
    ///
    /// # Returns
    ///
    /// The detected ray tracing capability tier.
    #[inline]
    pub const fn ray_tracing_tier(&self) -> VulkanRayTracingTier {
        if self.ray_tracing && self.buffer_device_address {
            VulkanRayTracingTier::Full
        } else if self.ray_query {
            VulkanRayTracingTier::Query
        } else {
            VulkanRayTracingTier::None
        }
    }

    /// Create a summary string of detected features.
    ///
    /// # Returns
    ///
    /// A human-readable summary of available features.
    pub fn summary(&self) -> String {
        let mut features = Vec::new();

        if self.ray_tracing {
            features.push("RT-Pipeline");
        }
        if self.ray_query {
            features.push("RT-Query");
        }
        if self.descriptor_indexing {
            features.push("Descriptor-Indexing");
        }
        if self.timeline_semaphores {
            features.push("Timeline-Semaphores");
        }
        if self.buffer_device_address {
            features.push("BDA");
        }
        if self.mesh_shading {
            features.push("Mesh-Shaders");
        }
        if self.dynamic_rendering {
            features.push("Dynamic-Rendering");
        }

        if features.is_empty() {
            "None".to_string()
        } else {
            features.join(", ")
        }
    }

    /// Check if mesh shading pipeline is supported.
    ///
    /// Mesh shading requires both mesh shaders and task shaders for the
    /// complete pipeline.
    ///
    /// # Returns
    ///
    /// `true` if mesh shading is available.
    #[inline]
    pub const fn supports_mesh_shading(&self) -> bool {
        self.mesh_shading
    }

    /// Check if modern synchronization features are supported.
    ///
    /// Modern sync includes timeline semaphores and synchronization2.
    ///
    /// # Returns
    ///
    /// `true` if modern synchronization is available.
    #[inline]
    pub const fn supports_modern_sync(&self) -> bool {
        self.timeline_semaphores && self.synchronization2
    }

    /// Get required Vulkan instance extensions for TRINITY.
    ///
    /// These extensions should be enabled when creating the Vulkan instance
    /// for optimal TRINITY operation.
    ///
    /// # Returns
    ///
    /// A slice of required instance extension names.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::VulkanFeatures;
    ///
    /// let extensions = VulkanFeatures::required_instance_extensions();
    /// assert!(extensions.contains(&"VK_KHR_get_physical_device_properties2"));
    /// ```
    #[inline]
    pub const fn required_instance_extensions() -> &'static [&'static str] {
        &[
            "VK_KHR_get_physical_device_properties2",
            "VK_KHR_external_memory_capabilities",
            "VK_KHR_external_semaphore_capabilities",
            "VK_EXT_debug_utils",
        ]
    }

    /// Get required Vulkan device extensions for ray tracing.
    ///
    /// These extensions must be enabled for ray tracing functionality.
    ///
    /// # Returns
    ///
    /// A slice of ray tracing extension names.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::VulkanFeatures;
    ///
    /// let extensions = VulkanFeatures::ray_tracing_extensions();
    /// assert!(extensions.contains(&"VK_KHR_ray_tracing_pipeline"));
    /// ```
    #[inline]
    pub const fn ray_tracing_extensions() -> &'static [&'static str] {
        &[
            "VK_KHR_ray_tracing_pipeline",
            "VK_KHR_acceleration_structure",
            "VK_KHR_ray_query",
            "VK_KHR_deferred_host_operations",
            "VK_KHR_buffer_device_address",
            "VK_KHR_spirv_1_4",
            "VK_KHR_shader_float_controls",
        ]
    }

    /// Get required Vulkan device extensions for mesh shading.
    ///
    /// These extensions must be enabled for mesh shader functionality.
    ///
    /// # Returns
    ///
    /// A slice of mesh shader extension names.
    #[inline]
    pub const fn mesh_shader_extensions() -> &'static [&'static str] {
        &[
            "VK_EXT_mesh_shader",
            "VK_KHR_spirv_1_4",
        ]
    }

    /// Get required Vulkan device extensions for bindless rendering.
    ///
    /// These extensions must be enabled for bindless resource access.
    ///
    /// # Returns
    ///
    /// A slice of bindless extension names.
    #[inline]
    pub const fn bindless_extensions() -> &'static [&'static str] {
        &[
            "VK_EXT_descriptor_indexing",
            "VK_KHR_buffer_device_address",
            "VK_KHR_maintenance3",
        ]
    }
}

// ============================================================================
// Raw Handle Access (unsafe)
// ============================================================================

/// Raw Vulkan handle access functions.
///
/// These functions provide access to underlying Vulkan handles when
/// direct Vulkan API calls are needed. This is primarily used for
/// interop with external libraries or debugging.
///
/// # Safety
///
/// All functions in this module are unsafe because:
/// - Handles are only valid while the wgpu objects are alive
/// - Using handles after the objects are dropped is undefined behavior
/// - Concurrent access must be properly synchronized
pub mod raw {
    use super::*;

    /// Get raw Vulkan instance handle.
    ///
    /// # Safety
    ///
    /// The returned handle is only valid while the adapter is alive.
    /// Using it after the adapter is dropped is undefined behavior.
    ///
    /// # Arguments
    ///
    /// * `adapter` - The wgpu adapter to query
    ///
    /// # Returns
    ///
    /// `Some` with the raw VkInstance handle if the adapter is Vulkan,
    /// `None` otherwise.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use renderer_backend::backend::vulkan::raw;
    ///
    /// # async unsafe fn example() {
    /// # let instance = wgpu::Instance::default();
    /// # let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.unwrap();
    /// let handle = unsafe { raw::get_instance_handle(&adapter) };
    /// if let Some(h) = handle {
    ///     // Use handle with external Vulkan library
    /// }
    /// # }
    /// ```
    #[inline]
    pub unsafe fn get_instance_handle(_adapter: &wgpu::Adapter) -> Option<*mut c_void> {
        // wgpu's as_hal API would be used here
        // This is a placeholder as the actual implementation depends on
        // wgpu's HAL access features
        None
    }

    /// Get raw Vulkan physical device handle.
    ///
    /// # Safety
    ///
    /// The returned handle is only valid while the adapter is alive.
    /// Using it after the adapter is dropped is undefined behavior.
    ///
    /// # Arguments
    ///
    /// * `adapter` - The wgpu adapter to query
    ///
    /// # Returns
    ///
    /// `Some` with the raw VkPhysicalDevice handle if the adapter is Vulkan,
    /// `None` otherwise.
    #[inline]
    pub unsafe fn get_physical_device_handle(_adapter: &wgpu::Adapter) -> Option<*mut c_void> {
        // wgpu's as_hal API would be used here
        None
    }

    /// Get raw Vulkan device handle.
    ///
    /// # Safety
    ///
    /// The returned handle is only valid while the device is alive.
    /// Using it after the device is dropped is undefined behavior.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device to query
    ///
    /// # Returns
    ///
    /// `Some` with the raw VkDevice handle if the device is Vulkan,
    /// `None` otherwise.
    #[inline]
    pub unsafe fn get_device_handle(_device: &wgpu::Device) -> Option<*mut c_void> {
        // wgpu's as_hal API would be used here
        None
    }

    /// Get raw Vulkan queue handle.
    ///
    /// # Safety
    ///
    /// The returned handle is only valid while the queue is alive.
    /// Using it after the queue is dropped is undefined behavior.
    ///
    /// # Arguments
    ///
    /// * `queue` - The wgpu queue to query
    ///
    /// # Returns
    ///
    /// `Some` with the raw VkQueue handle if the queue is Vulkan,
    /// `None` otherwise.
    #[inline]
    pub unsafe fn get_queue_handle(_queue: &wgpu::Queue) -> Option<*mut c_void> {
        // wgpu's as_hal API would be used here
        None
    }
}

// ============================================================================
// RawVulkanHandles (feature-gated)
// ============================================================================

/// Raw Vulkan handles for low-level access.
///
/// This struct provides access to the underlying Vulkan handles when
/// direct Vulkan API calls are needed. This is primarily used for
/// interop with external libraries or debugging.
///
/// # Safety
///
/// These handles must be used carefully:
/// - They are only valid while the wgpu device is alive
/// - Concurrent access must be properly synchronized
/// - Wrong usage can cause crashes or undefined behavior
///
/// # Feature Gate
///
/// This type is only available with the `vulkan-raw` feature.
///
/// # Example
///
/// ```ignore
/// // Only available with vulkan-raw feature
/// use renderer_backend::backend::RawVulkanHandles;
///
/// # unsafe fn example(device: &wgpu::Device) {
/// let handles = RawVulkanHandles::from_device(device);
/// if let Some(h) = handles {
///     // Use raw Vulkan handles carefully
///     println!("VkDevice: {:?}", h.device);
/// }
/// # }
/// ```
#[cfg(feature = "vulkan-raw")]
#[derive(Debug, Clone, Copy)]
pub struct RawVulkanHandles {
    /// Raw VkInstance handle.
    pub instance: u64,

    /// Raw VkPhysicalDevice handle.
    pub physical_device: u64,

    /// Raw VkDevice handle.
    pub device: u64,

    /// Raw VkQueue handle for the primary queue.
    pub queue: u64,

    /// Queue family index for the primary queue.
    pub queue_family_index: u32,
}

#[cfg(feature = "vulkan-raw")]
impl RawVulkanHandles {
    /// Attempt to extract raw Vulkan handles from a wgpu device.
    ///
    /// This uses wgpu's unsafe `as_hal` API to access the underlying
    /// Vulkan handles.
    ///
    /// # Safety
    ///
    /// The returned handles are only valid while the device is alive.
    /// Using them after the device is dropped is undefined behavior.
    ///
    /// # Returns
    ///
    /// `Some(handles)` if the device is a Vulkan device, `None` otherwise.
    #[inline]
    pub unsafe fn from_device(_device: &wgpu::Device) -> Option<Self> {
        // wgpu's as_hal API would be used here
        // This is a placeholder as the actual implementation depends on
        // wgpu's HAL access features
        None
    }

    /// Check if handles are valid (non-zero).
    #[inline]
    pub const fn is_valid(&self) -> bool {
        self.instance != 0 && self.physical_device != 0 && self.device != 0
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // VulkanRayTracingTier Tests
    // ========================================================================

    #[test]
    fn test_ray_tracing_tier_default() {
        let tier = VulkanRayTracingTier::default();
        assert_eq!(tier, VulkanRayTracingTier::None);
        assert!(!tier.is_available());
        assert!(!tier.is_full());
    }

    #[test]
    fn test_ray_tracing_tier_none() {
        let tier = VulkanRayTracingTier::None;
        assert!(!tier.is_available());
        assert!(!tier.is_full());
        assert_eq!(tier.name(), "None");
    }

    #[test]
    fn test_ray_tracing_tier_query() {
        let tier = VulkanRayTracingTier::Query;
        assert!(tier.is_available());
        assert!(!tier.is_full());
        assert_eq!(tier.name(), "Ray Query");
    }

    #[test]
    fn test_ray_tracing_tier_full() {
        let tier = VulkanRayTracingTier::Full;
        assert!(tier.is_available());
        assert!(tier.is_full());
        assert_eq!(tier.name(), "Full Pipeline");
    }

    #[test]
    fn test_ray_tracing_tier_display() {
        assert_eq!(format!("{}", VulkanRayTracingTier::None), "None");
        assert_eq!(format!("{}", VulkanRayTracingTier::Query), "Ray Query");
        assert_eq!(format!("{}", VulkanRayTracingTier::Full), "Full Pipeline");
    }

    // ========================================================================
    // VulkanFeatures Tests
    // ========================================================================

    #[test]
    fn test_vulkan_features_default() {
        let features = VulkanFeatures::default();
        assert!(!features.ray_tracing);
        assert!(!features.ray_query);
        assert!(!features.descriptor_indexing);
        assert!(!features.timeline_semaphores);
        assert!(!features.buffer_device_address);
        assert!(!features.mesh_shading);
        assert!(!features.dynamic_rendering);
    }

    #[test]
    fn test_vulkan_features_supports_rt_pipeline_requires_both() {
        let mut features = VulkanFeatures::default();
        assert!(!features.supports_rt_pipeline());

        features.ray_tracing = true;
        assert!(!features.supports_rt_pipeline());

        features.buffer_device_address = true;
        assert!(features.supports_rt_pipeline());
    }

    #[test]
    fn test_vulkan_features_supports_ray_query() {
        let mut features = VulkanFeatures::default();
        assert!(!features.supports_ray_query());

        features.ray_query = true;
        assert!(features.supports_ray_query());
    }

    #[test]
    fn test_vulkan_features_supports_any_rt() {
        let mut features = VulkanFeatures::default();
        assert!(!features.supports_any_rt());

        features.ray_query = true;
        assert!(features.supports_any_rt());

        features.ray_query = false;
        features.ray_tracing = true;
        assert!(features.supports_any_rt());

        features.ray_query = true;
        assert!(features.supports_any_rt());
    }

    #[test]
    fn test_vulkan_features_supports_bindless_requires_both() {
        let mut features = VulkanFeatures::default();
        assert!(!features.supports_bindless());

        features.descriptor_indexing = true;
        assert!(!features.supports_bindless());

        features.buffer_device_address = true;
        assert!(features.supports_bindless());
    }

    #[test]
    fn test_vulkan_features_supports_vulkan_1_2() {
        let mut features = VulkanFeatures::default();
        assert!(!features.supports_vulkan_1_2());

        features.timeline_semaphores = true;
        assert!(!features.supports_vulkan_1_2());

        features.buffer_device_address = true;
        assert!(features.supports_vulkan_1_2());
    }

    #[test]
    fn test_vulkan_features_supports_vulkan_1_3() {
        let mut features = VulkanFeatures::default();
        assert!(!features.supports_vulkan_1_3());

        features.dynamic_rendering = true;
        assert!(!features.supports_vulkan_1_3());

        features.synchronization2 = true;
        assert!(features.supports_vulkan_1_3());
    }

    #[test]
    fn test_vulkan_features_ray_tracing_tier_none() {
        let features = VulkanFeatures::default();
        assert_eq!(features.ray_tracing_tier(), VulkanRayTracingTier::None);
    }

    #[test]
    fn test_vulkan_features_ray_tracing_tier_query() {
        let mut features = VulkanFeatures::default();
        features.ray_query = true;
        assert_eq!(features.ray_tracing_tier(), VulkanRayTracingTier::Query);
    }

    #[test]
    fn test_vulkan_features_ray_tracing_tier_full() {
        let mut features = VulkanFeatures::default();
        features.ray_tracing = true;
        features.buffer_device_address = true;
        assert_eq!(features.ray_tracing_tier(), VulkanRayTracingTier::Full);
    }

    #[test]
    fn test_vulkan_features_ray_tracing_tier_full_takes_precedence() {
        // If both RT pipeline and ray query are available, tier is Full
        let mut features = VulkanFeatures::default();
        features.ray_tracing = true;
        features.ray_query = true;
        features.buffer_device_address = true;
        assert_eq!(features.ray_tracing_tier(), VulkanRayTracingTier::Full);
    }

    #[test]
    fn test_vulkan_features_summary_empty() {
        let features = VulkanFeatures::default();
        assert_eq!(features.summary(), "None");
    }

    #[test]
    fn test_vulkan_features_summary_with_features() {
        let mut features = VulkanFeatures::default();
        features.ray_tracing = true;
        features.descriptor_indexing = true;

        let summary = features.summary();
        assert!(summary.contains("RT-Pipeline"));
        assert!(summary.contains("Descriptor-Indexing"));
    }

    #[test]
    fn test_vulkan_features_summary_all_features() {
        let features = VulkanFeatures {
            ray_tracing: true,
            ray_query: true,
            descriptor_indexing: true,
            timeline_semaphores: true,
            buffer_device_address: true,
            mesh_shading: true,
            dynamic_rendering: true,
            synchronization2: false,
            extended_dynamic_state: false,
            maintenance4: false,
        };

        let summary = features.summary();
        assert!(summary.contains("RT-Pipeline"));
        assert!(summary.contains("RT-Query"));
        assert!(summary.contains("Descriptor-Indexing"));
        assert!(summary.contains("Timeline-Semaphores"));
        assert!(summary.contains("BDA"));
        assert!(summary.contains("Mesh-Shaders"));
        assert!(summary.contains("Dynamic-Rendering"));
    }

    #[test]
    fn test_vulkan_features_from_empty_wgpu_features() {
        let wgpu_features = Features::empty();
        let vk_features = VulkanFeatures::from_features(wgpu_features);

        assert!(!vk_features.ray_tracing);
        assert!(!vk_features.ray_query);
        assert!(!vk_features.descriptor_indexing);
        assert!(!vk_features.timeline_semaphores);
        assert!(!vk_features.buffer_device_address);
    }

    #[test]
    fn test_vulkan_features_from_rt_wgpu_features() {
        let wgpu_features = Features::RAY_TRACING_ACCELERATION_STRUCTURE | Features::RAY_QUERY;
        let vk_features = VulkanFeatures::from_features(wgpu_features);

        assert!(vk_features.ray_tracing);
        assert!(vk_features.ray_query);
        // RT implies BDA
        assert!(vk_features.buffer_device_address);
    }

    #[test]
    fn test_vulkan_features_from_bindless_wgpu_features() {
        let wgpu_features = Features::TEXTURE_BINDING_ARRAY
            | Features::BUFFER_BINDING_ARRAY
            | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
        let vk_features = VulkanFeatures::from_features(wgpu_features);

        assert!(vk_features.descriptor_indexing);
    }

    #[test]
    fn test_vulkan_features_partial_bindless_not_detected() {
        // Missing one required feature
        let wgpu_features = Features::TEXTURE_BINDING_ARRAY | Features::BUFFER_BINDING_ARRAY;
        let vk_features = VulkanFeatures::from_features(wgpu_features);

        assert!(!vk_features.descriptor_indexing);
    }

    #[test]
    fn test_vulkan_features_mesh_shading_check() {
        let mut features = VulkanFeatures::default();
        assert!(!features.supports_mesh_shading());

        features.mesh_shading = true;
        assert!(features.supports_mesh_shading());
    }

    #[test]
    fn test_vulkan_features_modern_sync_check() {
        let mut features = VulkanFeatures::default();
        assert!(!features.supports_modern_sync());

        features.timeline_semaphores = true;
        assert!(!features.supports_modern_sync());

        features.synchronization2 = true;
        assert!(features.supports_modern_sync());
    }

    #[test]
    fn test_required_instance_extensions() {
        let extensions = VulkanFeatures::required_instance_extensions();
        assert!(extensions.len() >= 3);
        assert!(extensions.contains(&"VK_KHR_get_physical_device_properties2"));
        assert!(extensions.contains(&"VK_EXT_debug_utils"));
    }

    #[test]
    fn test_ray_tracing_extensions() {
        let extensions = VulkanFeatures::ray_tracing_extensions();
        assert!(extensions.len() >= 5);
        assert!(extensions.contains(&"VK_KHR_ray_tracing_pipeline"));
        assert!(extensions.contains(&"VK_KHR_acceleration_structure"));
        assert!(extensions.contains(&"VK_KHR_ray_query"));
        assert!(extensions.contains(&"VK_KHR_buffer_device_address"));
    }

    #[test]
    fn test_features_all_false_default() {
        let features = VulkanFeatures::default();
        // Verify all fields are false by default
        assert!(!features.ray_tracing);
        assert!(!features.ray_query);
        assert!(!features.descriptor_indexing);
        assert!(!features.timeline_semaphores);
        assert!(!features.buffer_device_address);
        assert!(!features.mesh_shading);
        assert!(!features.dynamic_rendering);
        assert!(!features.synchronization2);
        assert!(!features.extended_dynamic_state);
        assert!(!features.maintenance4);
    }

    #[test]
    fn test_features_partial_ray_tracing() {
        // Ray tracing alone doesn't enable full RT pipeline
        let mut features = VulkanFeatures::default();
        features.ray_tracing = true;
        assert!(!features.supports_rt_pipeline());
        assert!(features.supports_any_rt());

        // Need buffer device address too
        features.buffer_device_address = true;
        assert!(features.supports_rt_pipeline());
    }

    #[test]
    fn test_features_partial_mesh_shading() {
        let mut features = VulkanFeatures::default();
        // Mesh shading requires the mesh_shading flag
        assert!(!features.supports_mesh_shading());

        features.mesh_shading = true;
        assert!(features.supports_mesh_shading());
    }

    // ========================================================================
    // VulkanVersion Tests
    // ========================================================================

    #[test]
    fn test_vulkan_version_constants() {
        assert_eq!(VulkanVersion::V1_0, VulkanVersion::new(1, 0, 0));
        assert_eq!(VulkanVersion::V1_1, VulkanVersion::new(1, 1, 0));
        assert_eq!(VulkanVersion::V1_2, VulkanVersion::new(1, 2, 0));
        assert_eq!(VulkanVersion::V1_3, VulkanVersion::new(1, 3, 0));
    }

    #[test]
    fn test_vulkan_version_ordering() {
        assert!(VulkanVersion::V1_0 < VulkanVersion::V1_1);
        assert!(VulkanVersion::V1_1 < VulkanVersion::V1_2);
        assert!(VulkanVersion::V1_2 < VulkanVersion::V1_3);
        assert!(VulkanVersion::V1_3 > VulkanVersion::V1_2);
        assert_eq!(VulkanVersion::V1_2, VulkanVersion::V1_2);
    }

    #[test]
    fn test_vulkan_version_from_raw() {
        // Vulkan 1.0.0 = 0x00400000
        let v1_0 = VulkanVersion::from_raw(0x00400000);
        assert_eq!(v1_0.major, 1);
        assert_eq!(v1_0.minor, 0);
        assert_eq!(v1_0.patch, 0);

        // Vulkan 1.2.0 = 0x00402000
        let v1_2 = VulkanVersion::from_raw(0x00402000);
        assert_eq!(v1_2.major, 1);
        assert_eq!(v1_2.minor, 2);
        assert_eq!(v1_2.patch, 0);

        // Vulkan 1.3.250
        let v1_3_250 = VulkanVersion::from_raw(0x004030FA);
        assert_eq!(v1_3_250.major, 1);
        assert_eq!(v1_3_250.minor, 3);
        assert_eq!(v1_3_250.patch, 250);
    }

    #[test]
    fn test_vulkan_version_to_raw() {
        assert_eq!(VulkanVersion::V1_0.to_raw(), 0x00400000);
        assert_eq!(VulkanVersion::V1_1.to_raw(), 0x00401000);
        assert_eq!(VulkanVersion::V1_2.to_raw(), 0x00402000);
        assert_eq!(VulkanVersion::V1_3.to_raw(), 0x00403000);

        // Round-trip
        let version = VulkanVersion::new(1, 3, 250);
        let raw = version.to_raw();
        let decoded = VulkanVersion::from_raw(raw);
        assert_eq!(version, decoded);
    }

    #[test]
    fn test_vulkan_version_is_at_least() {
        let v1_2 = VulkanVersion::V1_2;
        assert!(v1_2.is_at_least(1, 0));
        assert!(v1_2.is_at_least(1, 1));
        assert!(v1_2.is_at_least(1, 2));
        assert!(!v1_2.is_at_least(1, 3));
        assert!(!v1_2.is_at_least(2, 0));
    }

    #[test]
    fn test_vulkan_version_display() {
        assert_eq!(format!("{}", VulkanVersion::V1_0), "1.0.0");
        assert_eq!(format!("{}", VulkanVersion::V1_2), "1.2.0");
        assert_eq!(format!("{}", VulkanVersion::new(1, 3, 250)), "1.3.250");
    }

    #[test]
    fn test_vulkan_version_default() {
        let version = VulkanVersion::default();
        assert_eq!(version, VulkanVersion::V1_0);
    }

    // ========================================================================
    // VulkanDeviceType Tests
    // ========================================================================

    #[test]
    fn test_vulkan_device_type_variants() {
        // Test all variants exist and have correct properties
        assert!(VulkanDeviceType::DiscreteGpu.is_gpu());
        assert!(VulkanDeviceType::DiscreteGpu.is_hardware());

        assert!(VulkanDeviceType::IntegratedGpu.is_gpu());
        assert!(VulkanDeviceType::IntegratedGpu.is_hardware());

        assert!(VulkanDeviceType::VirtualGpu.is_gpu());
        assert!(VulkanDeviceType::VirtualGpu.is_hardware());

        assert!(!VulkanDeviceType::Cpu.is_gpu());
        assert!(!VulkanDeviceType::Cpu.is_hardware());

        assert!(!VulkanDeviceType::Other.is_gpu());
        assert!(!VulkanDeviceType::Other.is_hardware());
    }

    #[test]
    fn test_vulkan_device_type_names() {
        assert_eq!(VulkanDeviceType::DiscreteGpu.name(), "Discrete GPU");
        assert_eq!(VulkanDeviceType::IntegratedGpu.name(), "Integrated GPU");
        assert_eq!(VulkanDeviceType::VirtualGpu.name(), "Virtual GPU");
        assert_eq!(VulkanDeviceType::Cpu.name(), "CPU");
        assert_eq!(VulkanDeviceType::Other.name(), "Other");
    }

    #[test]
    fn test_vulkan_device_type_default() {
        let device_type = VulkanDeviceType::default();
        assert_eq!(device_type, VulkanDeviceType::Other);
    }

    #[test]
    fn test_vulkan_device_type_from_wgpu() {
        assert_eq!(VulkanDeviceType::from_wgpu(wgpu::DeviceType::DiscreteGpu), VulkanDeviceType::DiscreteGpu);
        assert_eq!(VulkanDeviceType::from_wgpu(wgpu::DeviceType::IntegratedGpu), VulkanDeviceType::IntegratedGpu);
        assert_eq!(VulkanDeviceType::from_wgpu(wgpu::DeviceType::VirtualGpu), VulkanDeviceType::VirtualGpu);
        assert_eq!(VulkanDeviceType::from_wgpu(wgpu::DeviceType::Cpu), VulkanDeviceType::Cpu);
        assert_eq!(VulkanDeviceType::from_wgpu(wgpu::DeviceType::Other), VulkanDeviceType::Other);
    }

    // ========================================================================
    // VulkanInfo Tests
    // ========================================================================

    #[test]
    fn test_vulkan_info_creation() {
        let info = VulkanInfo {
            version: VulkanVersion::V1_3,
            features: VulkanFeatures::default(),
            driver_name: "NVIDIA".to_string(),
            driver_version: 536870912,
            device_name: "GeForce RTX 4090".to_string(),
            device_type: VulkanDeviceType::DiscreteGpu,
            vendor_id: 0x10DE,
            device_id: 0x2684,
        };

        assert_eq!(info.version, VulkanVersion::V1_3);
        assert_eq!(info.driver_name, "NVIDIA");
        assert_eq!(info.device_name, "GeForce RTX 4090");
        assert_eq!(info.device_type, VulkanDeviceType::DiscreteGpu);
        assert_eq!(info.vendor_id, 0x10DE);
    }

    #[test]
    fn test_vulkan_info_summary() {
        let info = VulkanInfo {
            version: VulkanVersion::V1_3,
            features: VulkanFeatures::default(),
            driver_name: "NVIDIA".to_string(),
            driver_version: 0,
            device_name: "RTX 4090".to_string(),
            device_type: VulkanDeviceType::DiscreteGpu,
            vendor_id: 0,
            device_id: 0,
        };

        let summary = info.summary();
        assert!(summary.contains("RTX 4090"));
        assert!(summary.contains("Discrete GPU"));
        assert!(summary.contains("NVIDIA"));
        assert!(summary.contains("1.3.0"));
    }

    #[test]
    fn test_vulkan_info_default() {
        let info = VulkanInfo::default();
        assert_eq!(info.version, VulkanVersion::V1_0);
        assert!(info.driver_name.is_empty());
        assert!(info.device_name.is_empty());
        assert_eq!(info.device_type, VulkanDeviceType::Other);
    }

    #[test]
    fn test_vulkan_info_is_suitable() {
        let mut info = VulkanInfo::default();

        // Not suitable by default (no hardware, no 1.2 features)
        assert!(!info.is_suitable());

        // Add hardware type
        info.device_type = VulkanDeviceType::DiscreteGpu;
        assert!(!info.is_suitable()); // Still need Vulkan 1.2 features

        // Add Vulkan 1.2 features
        info.features.timeline_semaphores = true;
        info.features.buffer_device_address = true;
        assert!(info.is_suitable());
    }

    // ========================================================================
    // RawVulkanHandles Tests (feature-gated)
    // ========================================================================

    #[cfg(feature = "vulkan-raw")]
    #[test]
    fn test_raw_handles_is_valid() {
        let handles = RawVulkanHandles {
            instance: 0,
            physical_device: 0,
            device: 0,
            queue: 0,
            queue_family_index: 0,
        };
        assert!(!handles.is_valid());

        let handles = RawVulkanHandles {
            instance: 1,
            physical_device: 2,
            device: 3,
            queue: 4,
            queue_family_index: 0,
        };
        assert!(handles.is_valid());
    }
}
