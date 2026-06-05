//! Adapter enumeration and selection for TRINITY.
//!
//! This module provides enhanced adapter enumeration with logging,
//! filtering, and graceful zero-adapter handling.
//!
//! # Overview
//!
//! While `wgpu::Instance::enumerate_adapters()` returns a simple vector of adapters,
//! this module provides:
//!
//! - Detailed logging of each adapter's properties
//! - Backend-specific adapter counts
//! - Filtering by device type (discrete, integrated, software, etc.)
//! - Filtering by backend (Vulkan, Metal, DX12, etc.)
//! - Graceful handling when no adapters are found
//!
//! # Example
//!
//! ```no_run
//! use renderer_backend::device::{TrinityInstance, enumerate_adapters_with_info};
//!
//! let instance = TrinityInstance::new();
//! let result = enumerate_adapters_with_info(instance.inner(), instance.backends());
//!
//! println!("Found {} adapter(s)", result.adapters.len());
//! println!("Vulkan: {}, OpenGL: {}", result.backend_counts.vulkan, result.backend_counts.gl);
//! ```

use log::{debug, info, warn};
use wgpu::{Adapter, AdapterInfo, Backends, DeviceType, Instance};

// ============================================================================
// Vendor Classification
// ============================================================================

/// Known GPU vendor IDs for classification.
///
/// Vendor IDs are standardized PCI vendor identifiers. This enum classifies
/// common GPU vendors for easier handling in renderer logic.
///
/// # Example
///
/// ```
/// use renderer_backend::device::Vendor;
///
/// let vendor = Vendor::from_id(0x10DE);
/// assert_eq!(vendor, Vendor::Nvidia);
/// assert_eq!(vendor.name(), "NVIDIA");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Vendor {
    /// NVIDIA Corporation (vendor ID 0x10DE)
    Nvidia,
    /// Advanced Micro Devices (vendor IDs 0x1002, 0x1022)
    Amd,
    /// Intel Corporation (vendor ID 0x8086)
    Intel,
    /// Apple Inc. (vendor ID 0x106B)
    Apple,
    /// ARM Limited (vendor ID 0x13B5)
    Arm,
    /// Qualcomm (vendor ID 0x5143)
    Qualcomm,
    /// Microsoft Corporation (vendor ID 0x1414, used for WARP software renderer)
    Microsoft,
    /// Unknown vendor with raw vendor ID
    Unknown(u32),
}

impl Vendor {
    /// Classify vendor from a PCI vendor ID.
    ///
    /// # Arguments
    ///
    /// * `vendor_id` - The raw PCI vendor ID from the adapter info
    ///
    /// # Returns
    ///
    /// The classified vendor, or `Vendor::Unknown(id)` if not recognized.
    pub fn from_id(vendor_id: u32) -> Self {
        match vendor_id {
            0x10DE => Vendor::Nvidia,
            0x1002 | 0x1022 => Vendor::Amd,
            0x8086 => Vendor::Intel,
            0x106B => Vendor::Apple,
            0x13B5 => Vendor::Arm,
            0x5143 => Vendor::Qualcomm,
            0x1414 => Vendor::Microsoft,
            other => Vendor::Unknown(other),
        }
    }

    /// Get the human-readable vendor name.
    ///
    /// # Returns
    ///
    /// A static string with the vendor's common name.
    pub fn name(&self) -> &'static str {
        match self {
            Vendor::Nvidia => "NVIDIA",
            Vendor::Amd => "AMD",
            Vendor::Intel => "Intel",
            Vendor::Apple => "Apple",
            Vendor::Arm => "ARM",
            Vendor::Qualcomm => "Qualcomm",
            Vendor::Microsoft => "Microsoft",
            Vendor::Unknown(_) => "Unknown",
        }
    }

    /// Check if this is a known vendor (not Unknown).
    #[inline]
    pub fn is_known(&self) -> bool {
        !matches!(self, Vendor::Unknown(_))
    }

    /// Get the raw vendor ID.
    ///
    /// For known vendors, returns their standard PCI vendor ID.
    /// For unknown vendors, returns the ID passed to `from_id()`.
    pub fn id(&self) -> u32 {
        match self {
            Vendor::Nvidia => 0x10DE,
            Vendor::Amd => 0x1002,
            Vendor::Intel => 0x8086,
            Vendor::Apple => 0x106B,
            Vendor::Arm => 0x13B5,
            Vendor::Qualcomm => 0x5143,
            Vendor::Microsoft => 0x1414,
            Vendor::Unknown(id) => *id,
        }
    }
}

impl std::fmt::Display for Vendor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Vendor::Unknown(id) => write!(f, "Unknown (0x{:04X})", id),
            _ => write!(f, "{}", self.name()),
        }
    }
}

// ============================================================================
// Adapter Properties
// ============================================================================

/// Enhanced adapter properties wrapper.
///
/// This struct provides a convenient view of adapter information with
/// vendor classification, device type detection, and human-readable descriptions.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::device::{enumerate_adapters_with_info, AdapterProperties};
///
/// let instance = wgpu::Instance::default();
/// let result = enumerate_adapters_with_info(&instance, wgpu::Backends::PRIMARY);
///
/// for adapter in &result.adapters {
///     let props = AdapterProperties::from_adapter(adapter);
///     println!("{}", props.description());
///
///     if props.is_discrete() {
///         println!("  -> High-performance discrete GPU");
///     }
/// }
/// ```
#[derive(Debug, Clone)]
pub struct AdapterProperties {
    /// The adapter name (e.g., "NVIDIA GeForce RTX 4090").
    pub name: String,
    /// Classified vendor.
    pub vendor: Vendor,
    /// Raw vendor ID from the adapter.
    pub vendor_id: u32,
    /// Device ID (product identifier within the vendor).
    pub device_id: u32,
    /// Device type (discrete, integrated, virtual, CPU, other).
    pub device_type: DeviceType,
    /// Graphics backend (Vulkan, Metal, DX12, GL, WebGPU).
    pub backend: wgpu::Backend,
    /// Driver name (if available, may be empty).
    pub driver: String,
    /// Driver version/info (if available, may be empty).
    pub driver_info: String,
}

impl AdapterProperties {
    /// Extract properties from a wgpu Adapter.
    ///
    /// This queries the adapter's info and wraps it in a more convenient
    /// structure with vendor classification.
    ///
    /// # Arguments
    ///
    /// * `adapter` - The wgpu adapter to extract properties from
    ///
    /// # Returns
    ///
    /// An `AdapterProperties` instance with all available information.
    pub fn from_adapter(adapter: &wgpu::Adapter) -> Self {
        let info = adapter.get_info();
        AdapterProperties {
            name: info.name.clone(),
            vendor: Vendor::from_id(info.vendor),
            vendor_id: info.vendor,
            device_id: info.device,
            device_type: info.device_type,
            backend: info.backend,
            driver: info.driver.clone(),
            driver_info: info.driver_info.clone(),
        }
    }

    /// Create properties from raw AdapterInfo.
    ///
    /// Useful when you already have an `AdapterInfo` and want to avoid
    /// another `get_info()` call.
    pub fn from_info(info: &AdapterInfo) -> Self {
        AdapterProperties {
            name: info.name.clone(),
            vendor: Vendor::from_id(info.vendor),
            vendor_id: info.vendor,
            device_id: info.device,
            device_type: info.device_type,
            backend: info.backend,
            driver: info.driver.clone(),
            driver_info: info.driver_info.clone(),
        }
    }

    /// Generate a human-readable description of the adapter.
    ///
    /// The format varies based on available information:
    /// - With driver: `"NVIDIA GeForce RTX 4090 (Discrete GPU, Vulkan, driver: nvidia 535.86.05)"`
    /// - Without driver: `"NVIDIA GeForce RTX 4090 (Discrete GPU, Vulkan)"`
    ///
    /// # Returns
    ///
    /// A formatted string describing the adapter.
    pub fn description(&self) -> String {
        let type_str = device_type_short(self.device_type);
        if self.driver.is_empty() {
            format!(
                "{} ({}, {:?})",
                self.name, type_str, self.backend
            )
        } else if self.driver_info.is_empty() {
            format!(
                "{} ({}, {:?}, driver: {})",
                self.name, type_str, self.backend, self.driver
            )
        } else {
            format!(
                "{} ({}, {:?}, driver: {} {})",
                self.name, type_str, self.backend, self.driver, self.driver_info
            )
        }
    }

    /// Check if this is a discrete (dedicated) GPU.
    ///
    /// Discrete GPUs have their own dedicated video memory and are typically
    /// the highest-performance option for graphics workloads.
    #[inline]
    pub fn is_discrete(&self) -> bool {
        self.device_type == DeviceType::DiscreteGpu
    }

    /// Check if this is an integrated GPU.
    ///
    /// Integrated GPUs share system memory with the CPU and are typically
    /// more power-efficient but lower performance than discrete GPUs.
    #[inline]
    pub fn is_integrated(&self) -> bool {
        self.device_type == DeviceType::IntegratedGpu
    }

    /// Check if this is a software renderer.
    ///
    /// Software renderers (CPU) are fallbacks when no GPU is available.
    /// They are significantly slower but provide compatibility.
    #[inline]
    pub fn is_software(&self) -> bool {
        self.device_type == DeviceType::Cpu
    }

    /// Check if this is a virtual GPU.
    ///
    /// Virtual GPUs are typically found in virtualized environments
    /// (VMs, cloud instances) with GPU passthrough or virtualization.
    #[inline]
    pub fn is_virtual(&self) -> bool {
        self.device_type == DeviceType::VirtualGpu
    }

    /// Check if driver information is available.
    #[inline]
    pub fn has_driver_info(&self) -> bool {
        !self.driver.is_empty()
    }
}

impl std::fmt::Display for AdapterProperties {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.description())
    }
}

/// Get a short device type description (for compact formatting).
fn device_type_short(device_type: DeviceType) -> &'static str {
    match device_type {
        DeviceType::DiscreteGpu => "Discrete GPU",
        DeviceType::IntegratedGpu => "Integrated GPU",
        DeviceType::VirtualGpu => "Virtual GPU",
        DeviceType::Cpu => "Software",
        DeviceType::Other => "Other",
    }
}

/// Result of adapter enumeration with metadata.
///
/// This struct bundles the enumerated adapters with additional metadata
/// about the enumeration, such as per-backend adapter counts.
#[derive(Debug)]
pub struct EnumerationResult {
    /// All adapters found for the requested backends.
    pub adapters: Vec<Adapter>,
    /// Number of adapters per backend type.
    pub backend_counts: BackendCounts,
}

impl EnumerationResult {
    /// Returns `true` if no adapters were found.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.adapters.is_empty()
    }

    /// Returns the total number of adapters found.
    #[inline]
    pub fn len(&self) -> usize {
        self.adapters.len()
    }

    /// Get the first discrete GPU adapter, if any.
    ///
    /// Discrete GPUs are typically the highest performance option.
    pub fn first_discrete(&self) -> Option<&Adapter> {
        self.adapters
            .iter()
            .find(|a| a.get_info().device_type == DeviceType::DiscreteGpu)
    }

    /// Get the first integrated GPU adapter, if any.
    ///
    /// Integrated GPUs share memory with the CPU and are power-efficient.
    pub fn first_integrated(&self) -> Option<&Adapter> {
        self.adapters
            .iter()
            .find(|a| a.get_info().device_type == DeviceType::IntegratedGpu)
    }

    /// Get the best adapter based on device type priority.
    ///
    /// Priority order:
    /// 1. DiscreteGpu (highest performance)
    /// 2. IntegratedGpu (good balance)
    /// 3. VirtualGpu (virtualized environments)
    /// 4. Cpu (software fallback)
    /// 5. Other (unknown type)
    pub fn best_adapter(&self) -> Option<&Adapter> {
        // Try device types in priority order
        self.first_discrete()
            .or_else(|| self.first_integrated())
            .or_else(|| {
                self.adapters
                    .iter()
                    .find(|a| a.get_info().device_type == DeviceType::VirtualGpu)
            })
            .or_else(|| {
                self.adapters
                    .iter()
                    .find(|a| a.get_info().device_type == DeviceType::Cpu)
            })
            .or_else(|| self.adapters.first())
    }

    /// Get enhanced properties for all enumerated adapters.
    ///
    /// This provides a convenient way to access vendor classification,
    /// device type checks, and human-readable descriptions for all adapters.
    ///
    /// # Returns
    ///
    /// A vector of `AdapterProperties` for each adapter, in the same order.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::device::enumerate_adapters_with_info;
    ///
    /// let instance = wgpu::Instance::default();
    /// let result = enumerate_adapters_with_info(&instance, wgpu::Backends::PRIMARY);
    ///
    /// for props in result.properties() {
    ///     println!("{}: {}", props.vendor.name(), props.name);
    /// }
    /// ```
    pub fn properties(&self) -> Vec<AdapterProperties> {
        self.adapters
            .iter()
            .map(AdapterProperties::from_adapter)
            .collect()
    }

    /// Find the first adapter from a specific vendor.
    ///
    /// # Arguments
    ///
    /// * `vendor` - The vendor to search for
    ///
    /// # Returns
    ///
    /// The first adapter matching the vendor, or `None`.
    pub fn first_by_vendor(&self, vendor: Vendor) -> Option<&Adapter> {
        self.adapters
            .iter()
            .find(|a| Vendor::from_id(a.get_info().vendor) == vendor)
    }
}

/// Counts of adapters per backend type.
///
/// Useful for debugging and selecting adapters from specific backends.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct BackendCounts {
    /// Number of Vulkan adapters found.
    pub vulkan: usize,
    /// Number of Metal adapters found (macOS/iOS).
    pub metal: usize,
    /// Number of DirectX 12 adapters found (Windows).
    pub dx12: usize,
    /// Number of OpenGL/GLES adapters found.
    pub gl: usize,
    /// Number of WebGPU adapters found (WASM).
    pub webgpu: usize,
}

impl BackendCounts {
    /// Returns the total number of adapters across all backends.
    #[inline]
    pub fn total(&self) -> usize {
        self.vulkan + self.metal + self.dx12 + self.gl + self.webgpu
    }

    /// Returns `true` if no adapters were found for any backend.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.total() == 0
    }

    /// Returns a human-readable summary of the backend counts.
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        if self.vulkan > 0 {
            parts.push(format!("Vulkan: {}", self.vulkan));
        }
        if self.metal > 0 {
            parts.push(format!("Metal: {}", self.metal));
        }
        if self.dx12 > 0 {
            parts.push(format!("DX12: {}", self.dx12));
        }
        if self.gl > 0 {
            parts.push(format!("GL: {}", self.gl));
        }
        if self.webgpu > 0 {
            parts.push(format!("WebGPU: {}", self.webgpu));
        }
        if parts.is_empty() {
            "none".to_string()
        } else {
            parts.join(", ")
        }
    }
}

/// Enumerate adapters with logging and filtering.
///
/// This function wraps `wgpu::Instance::enumerate_adapters()` with additional
/// functionality:
///
/// - Logs detailed information about each adapter found
/// - Counts adapters per backend type
/// - Logs a warning if no adapters are found
///
/// # Arguments
///
/// * `instance` - The wgpu Instance to enumerate from
/// * `backends` - Backend filter (which backends to query)
///
/// # Returns
///
/// `EnumerationResult` containing all adapters and metadata.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::device::enumerate_adapters_with_info;
///
/// let instance = wgpu::Instance::default();
/// let result = enumerate_adapters_with_info(&instance, wgpu::Backends::PRIMARY);
///
/// if result.is_empty() {
///     eprintln!("No GPU adapters found!");
/// } else {
///     println!("Found {} adapter(s): {}", result.len(), result.backend_counts.summary());
/// }
/// ```
pub fn enumerate_adapters_with_info(instance: &Instance, backends: Backends) -> EnumerationResult {
    let adapters: Vec<Adapter> = instance.enumerate_adapters(backends);

    let mut counts = BackendCounts::default();

    for adapter in &adapters {
        let info = adapter.get_info();
        log_adapter_info(&info);

        match info.backend {
            wgpu::Backend::Vulkan => counts.vulkan += 1,
            wgpu::Backend::Metal => counts.metal += 1,
            wgpu::Backend::Dx12 => counts.dx12 += 1,
            wgpu::Backend::Gl => counts.gl += 1,
            wgpu::Backend::BrowserWebGpu => counts.webgpu += 1,
            wgpu::Backend::Empty => {}
        }
    }

    if adapters.is_empty() {
        warn!(
            "AdapterEnumerator: No adapters found for backends {:?}",
            backends
        );
        warn!("  Possible causes:");
        warn!("  - No GPU installed or detected");
        warn!("  - GPU drivers not installed or outdated");
        warn!("  - Requested backend not supported on this platform");
        #[cfg(target_os = "linux")]
        warn!("  - On Linux: ensure Vulkan ICD (e.g., libvulkan1) is installed");
    } else {
        info!(
            "AdapterEnumerator: Found {} adapter(s) for backends {:?}",
            adapters.len(),
            backends
        );
        debug!("  Backend breakdown: {}", counts.summary());
    }

    EnumerationResult {
        adapters,
        backend_counts: counts,
    }
}

/// Log detailed adapter information for debugging.
///
/// This logs at DEBUG level to avoid cluttering production logs while
/// still providing valuable information during development.
fn log_adapter_info(info: &AdapterInfo) {
    debug!(
        "  Adapter: {} | Backend: {:?} | Type: {:?} | Vendor: 0x{:04X} | Device: 0x{:04X}",
        info.name, info.backend, info.device_type, info.vendor, info.device
    );

    if !info.driver.is_empty() {
        debug!("    Driver: {} ({})", info.driver, info.driver_info);
    }
}

/// Filter adapters by device type.
///
/// Consumes the input vector and returns only adapters matching the specified
/// device type.
///
/// # Arguments
///
/// * `adapters` - Vector of adapters to filter
/// * `device_type` - The device type to filter by
///
/// # Returns
///
/// Vector containing only adapters of the specified device type.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::device::{enumerate_adapters_with_info, filter_by_device_type};
/// use wgpu::DeviceType;
///
/// let instance = wgpu::Instance::default();
/// let result = enumerate_adapters_with_info(&instance, wgpu::Backends::PRIMARY);
///
/// // Get only discrete GPUs
/// let discrete_gpus = filter_by_device_type(result.adapters, DeviceType::DiscreteGpu);
/// ```
pub fn filter_by_device_type(adapters: Vec<Adapter>, device_type: DeviceType) -> Vec<Adapter> {
    let filtered: Vec<Adapter> = adapters
        .into_iter()
        .filter(|a| a.get_info().device_type == device_type)
        .collect();

    debug!(
        "AdapterEnumerator: Filtered by device type {:?}: {} adapter(s)",
        device_type,
        filtered.len()
    );

    filtered
}

/// Filter adapters by backend.
///
/// Consumes the input vector and returns only adapters from the specified backend.
///
/// # Arguments
///
/// * `adapters` - Vector of adapters to filter
/// * `backend` - The backend to filter by
///
/// # Returns
///
/// Vector containing only adapters from the specified backend.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::device::{enumerate_adapters_with_info, filter_by_backend};
///
/// let instance = wgpu::Instance::default();
/// let result = enumerate_adapters_with_info(&instance, wgpu::Backends::all());
///
/// // Get only Vulkan adapters
/// let vulkan_adapters = filter_by_backend(result.adapters, wgpu::Backend::Vulkan);
/// ```
pub fn filter_by_backend(adapters: Vec<Adapter>, backend: wgpu::Backend) -> Vec<Adapter> {
    let filtered: Vec<Adapter> = adapters
        .into_iter()
        .filter(|a| a.get_info().backend == backend)
        .collect();

    debug!(
        "AdapterEnumerator: Filtered by backend {:?}: {} adapter(s)",
        backend,
        filtered.len()
    );

    filtered
}

/// Get a human-readable description of a device type.
///
/// Useful for logging and user-facing messages.
pub fn device_type_description(device_type: DeviceType) -> &'static str {
    match device_type {
        DeviceType::DiscreteGpu => "Discrete GPU (dedicated graphics card)",
        DeviceType::IntegratedGpu => "Integrated GPU (shared memory with CPU)",
        DeviceType::VirtualGpu => "Virtual GPU (virtualized environment)",
        DeviceType::Cpu => "CPU (software rendering)",
        DeviceType::Other => "Other (unknown device type)",
    }
}

// ============================================================================
// Feature Detection
// ============================================================================

/// Capability tier based on available features.
///
/// Feature tiers provide a simple way to categorize GPU capabilities for
/// making rendering decisions. Higher tiers indicate more advanced features.
///
/// # Tier Classification
///
/// - **Minimal**: Core WebGPU only, no optional features
/// - **Standard**: Common optional features (compression, indirect rendering)
/// - **Advanced**: Modern features (timestamp queries, f16 shaders)
/// - **Full**: All features available
///
/// # Example
///
/// ```no_run
/// use renderer_backend::device::{AdapterFeatures, FeatureTier};
///
/// # async fn example() {
/// let instance = wgpu::Instance::default();
/// let adapter = instance
///     .request_adapter(&wgpu::RequestAdapterOptions::default())
///     .await
///     .unwrap();
///
/// let features = AdapterFeatures::from_adapter(&adapter);
/// match features.tier() {
///     FeatureTier::Full => println!("All features available!"),
///     FeatureTier::Advanced => println!("Modern GPU features supported"),
///     FeatureTier::Standard => println!("Common features available"),
///     FeatureTier::Minimal => println!("Basic WebGPU only"),
/// }
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FeatureTier {
    /// Minimal tier - core WebGPU only, no optional features.
    Minimal,
    /// Standard tier - common optional features available.
    Standard,
    /// Advanced tier - modern GPU features supported.
    Advanced,
    /// Full tier - all features available.
    Full,
}

impl FeatureTier {
    /// Get a human-readable description of this tier.
    pub fn description(&self) -> &'static str {
        match self {
            FeatureTier::Minimal => "Minimal (core WebGPU only)",
            FeatureTier::Standard => "Standard (common optional features)",
            FeatureTier::Advanced => "Advanced (modern GPU features)",
            FeatureTier::Full => "Full (all features available)",
        }
    }
}

impl std::fmt::Display for FeatureTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Extracted adapter features with categorization.
///
/// This struct wraps `wgpu::Features` and provides convenient accessor methods
/// for querying optional GPU features. Features are organized by category:
///
/// - **Core Features**: Depth, texture compression
/// - **Rendering Features**: Indirect rendering, queries
/// - **Advanced Features**: Shader extensions, format support
///
/// # Example
///
/// ```no_run
/// use renderer_backend::device::AdapterFeatures;
///
/// # async fn example() {
/// let instance = wgpu::Instance::default();
/// let adapter = instance
///     .request_adapter(&wgpu::RequestAdapterOptions::default())
///     .await
///     .unwrap();
///
/// let features = AdapterFeatures::from_adapter(&adapter);
/// println!("Feature count: {}", features.count());
/// println!("Tier: {:?}", features.tier());
///
/// if features.has_timestamp_query() {
///     println!("GPU profiling available!");
/// }
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct AdapterFeatures {
    /// Raw wgpu Features bitflags.
    pub raw: wgpu::Features,
}

impl AdapterFeatures {
    /// Extract features from a wgpu Adapter.
    ///
    /// # Arguments
    ///
    /// * `adapter` - The wgpu adapter to query features from
    ///
    /// # Returns
    ///
    /// An `AdapterFeatures` instance containing the adapter's supported features.
    pub fn from_adapter(adapter: &wgpu::Adapter) -> Self {
        Self {
            raw: adapter.features(),
        }
    }

    // ========================================================================
    // Core Features
    // ========================================================================

    /// Check if depth clip control is available.
    ///
    /// Allows disabling depth clipping for techniques like shadow mapping
    /// or portals that render geometry behind the near plane.
    #[inline]
    pub fn has_depth_clip_control(&self) -> bool {
        self.raw.contains(wgpu::Features::DEPTH_CLIP_CONTROL)
    }

    /// Check if depth32float stencil8 format is available.
    ///
    /// Combined depth+stencil format with 32-bit float depth precision.
    /// Required for some advanced shadow mapping techniques.
    #[inline]
    pub fn has_depth32float_stencil8(&self) -> bool {
        self.raw.contains(wgpu::Features::DEPTH32FLOAT_STENCIL8)
    }

    // ========================================================================
    // Texture Compression Features
    // ========================================================================

    /// Check if BC (Block Compression) texture compression is available.
    ///
    /// BC formats (BC1-BC7) are widely supported on desktop GPUs (Windows, macOS, Linux).
    /// Essential for reducing VRAM usage and bandwidth.
    ///
    /// Platform availability:
    /// - Desktop: Almost universal
    /// - Mobile: Rare (use ETC2 or ASTC instead)
    /// - WebGPU: Varies by platform
    #[inline]
    pub fn has_texture_compression_bc(&self) -> bool {
        self.raw.contains(wgpu::Features::TEXTURE_COMPRESSION_BC)
    }

    /// Check if ETC2 texture compression is available.
    ///
    /// ETC2 is the standard compression format for OpenGL ES 3.0+ and mobile GPUs.
    ///
    /// Platform availability:
    /// - Desktop: Limited (Intel, some AMD)
    /// - Mobile: Universal on OpenGL ES 3.0+ devices
    /// - WebGPU: Mobile browsers
    #[inline]
    pub fn has_texture_compression_etc2(&self) -> bool {
        self.raw.contains(wgpu::Features::TEXTURE_COMPRESSION_ETC2)
    }

    /// Check if ASTC texture compression is available.
    ///
    /// ASTC provides high-quality compression with flexible block sizes.
    /// Supports both LDR and HDR content (see `has_texture_compression_astc_hdr`).
    ///
    /// Platform availability:
    /// - Desktop: Modern NVIDIA (Fermi+), AMD (GCN+), Intel (Haswell+)
    /// - Mobile: Most modern mobile GPUs (Mali, Adreno, PowerVR)
    /// - WebGPU: Varies
    #[inline]
    pub fn has_texture_compression_astc(&self) -> bool {
        self.raw.contains(wgpu::Features::TEXTURE_COMPRESSION_ASTC)
    }

    /// Check if ASTC HDR texture compression is available.
    ///
    /// ASTC HDR extends ASTC to support high dynamic range content.
    /// Requires base ASTC support.
    #[inline]
    pub fn has_texture_compression_astc_hdr(&self) -> bool {
        self.raw.contains(wgpu::Features::TEXTURE_COMPRESSION_ASTC_HDR)
    }

    /// Check if any texture compression format is available.
    ///
    /// Returns true if BC, ETC2, or ASTC compression is supported.
    #[inline]
    pub fn has_any_texture_compression(&self) -> bool {
        self.has_texture_compression_bc()
            || self.has_texture_compression_etc2()
            || self.has_texture_compression_astc()
    }

    // ========================================================================
    // Rendering Features
    // ========================================================================

    /// Check if indirect first instance is available.
    ///
    /// Allows the first instance index in indirect draw calls to be non-zero.
    /// Essential for GPU-driven rendering pipelines.
    #[inline]
    pub fn has_indirect_first_instance(&self) -> bool {
        self.raw.contains(wgpu::Features::INDIRECT_FIRST_INSTANCE)
    }

    /// Check if multiview rendering is available.
    ///
    /// Multiview allows rendering to multiple render targets in a single pass.
    /// Primary use case: VR stereo rendering (left/right eye).
    #[inline]
    pub fn has_multiview(&self) -> bool {
        self.raw.contains(wgpu::Features::MULTIVIEW)
    }

    // ========================================================================
    // Query Features
    // ========================================================================

    /// Check if timestamp query is available.
    ///
    /// Timestamp queries allow measuring GPU execution time for profiling.
    /// Essential for performance optimization and GPU profilers.
    ///
    /// Platform availability:
    /// - Desktop: Almost universal
    /// - Mobile: Limited
    /// - WebGPU: Requires `timestamp-query` feature
    #[inline]
    pub fn has_timestamp_query(&self) -> bool {
        self.raw.contains(wgpu::Features::TIMESTAMP_QUERY)
    }

    /// Check if pipeline statistics query is available.
    ///
    /// Pipeline statistics provide metrics like vertex/fragment invocations.
    /// Useful for debugging and optimization.
    #[inline]
    pub fn has_pipeline_statistics_query(&self) -> bool {
        self.raw.contains(wgpu::Features::PIPELINE_STATISTICS_QUERY)
    }

    // ========================================================================
    // Shader Features
    // ========================================================================

    /// Check if shader f16 is available.
    ///
    /// Allows use of 16-bit floating point types in shaders for improved
    /// performance and reduced register pressure.
    ///
    /// Platform availability:
    /// - Desktop: Modern NVIDIA (Turing+), AMD (RDNA+), Intel (Xe+)
    /// - Mobile: Many modern mobile GPUs
    #[inline]
    pub fn has_shader_f16(&self) -> bool {
        self.raw.contains(wgpu::Features::SHADER_F16)
    }

    /// Check if push constants are available.
    ///
    /// Push constants provide a fast path for small amounts of per-draw data
    /// without buffer updates. Limited to a small size (typically 128-256 bytes).
    ///
    /// Platform availability:
    /// - Vulkan: Universal
    /// - Metal: Emulated
    /// - DX12: Root constants
    #[inline]
    pub fn has_push_constants(&self) -> bool {
        self.raw.contains(wgpu::Features::PUSH_CONSTANTS)
    }

    // ========================================================================
    // Format Features
    // ========================================================================

    /// Check if RG11B10 float render is available.
    ///
    /// Allows rendering to RG11B10UFloat format, which provides HDR support
    /// with compact storage (32 bits per pixel vs 64 for RGBA16Float).
    #[inline]
    pub fn has_rg11b10ufloat_renderable(&self) -> bool {
        self.raw.contains(wgpu::Features::RG11B10UFLOAT_RENDERABLE)
    }

    /// Check if BGRA8 unorm storage is available.
    ///
    /// Allows using BGRA8Unorm format as storage texture.
    /// Useful for compute-based post-processing with swapchain-compatible output.
    #[inline]
    pub fn has_bgra8unorm_storage(&self) -> bool {
        self.raw.contains(wgpu::Features::BGRA8UNORM_STORAGE)
    }

    /// Check if float32 filterable is available.
    ///
    /// Allows linear filtering on 32-bit float textures.
    /// Required for some HDR rendering techniques.
    #[inline]
    pub fn has_float32_filterable(&self) -> bool {
        self.raw.contains(wgpu::Features::FLOAT32_FILTERABLE)
    }

    /// Check if 16-bit normalized texture formats are available.
    ///
    /// Adds support for R16Unorm, R16Snorm, Rg16Unorm, Rg16Snorm,
    /// Rgba16Unorm, Rgba16Snorm texture formats.
    #[inline]
    pub fn has_texture_format_16bit_norm(&self) -> bool {
        self.raw.contains(wgpu::Features::TEXTURE_FORMAT_16BIT_NORM)
    }

    // ========================================================================
    // Utility Methods
    // ========================================================================

    /// Count total available features.
    ///
    /// Returns the number of optional features supported by this adapter.
    #[inline]
    pub fn count(&self) -> u32 {
        self.raw.iter().count() as u32
    }

    /// Determine the capability tier based on available features.
    ///
    /// Tier classification considers both feature count and specific
    /// features that indicate advanced GPU capabilities.
    ///
    /// # Tier Criteria
    ///
    /// - **Full**: 12+ features including advanced features
    /// - **Advanced**: 8+ features including timestamp/pipeline queries or f16
    /// - **Standard**: 4+ features
    /// - **Minimal**: Fewer than 4 features
    pub fn tier(&self) -> FeatureTier {
        let count = self.count();
        let has_advanced = self.has_timestamp_query()
            || self.has_pipeline_statistics_query()
            || self.has_shader_f16();

        if count >= 12 && has_advanced {
            FeatureTier::Full
        } else if count >= 8 && has_advanced {
            FeatureTier::Advanced
        } else if count >= 4 {
            FeatureTier::Standard
        } else {
            FeatureTier::Minimal
        }
    }

    /// Get a summary of features by category.
    ///
    /// Returns a compact struct with key feature availability flags
    /// for quick capability checks.
    pub fn summary(&self) -> FeaturesSummary {
        FeaturesSummary {
            tier: self.tier(),
            total_count: self.count(),
            has_compression_bc: self.has_texture_compression_bc(),
            has_compression_etc2: self.has_texture_compression_etc2(),
            has_compression_astc: self.has_texture_compression_astc(),
            has_compression_astc_hdr: self.has_texture_compression_astc_hdr(),
            has_timestamp_query: self.has_timestamp_query(),
            has_pipeline_statistics: self.has_pipeline_statistics_query(),
            has_shader_f16: self.has_shader_f16(),
            has_push_constants: self.has_push_constants(),
            has_multiview: self.has_multiview(),
            has_indirect_first_instance: self.has_indirect_first_instance(),
        }
    }

    /// Check if all required features for a set are available.
    ///
    /// # Arguments
    ///
    /// * `required` - The features to check for
    ///
    /// # Returns
    ///
    /// `true` if all required features are present.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::device::AdapterFeatures;
    ///
    /// # async fn example() {
    /// # let instance = wgpu::Instance::default();
    /// # let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.unwrap();
    /// let features = AdapterFeatures::from_adapter(&adapter);
    ///
    /// let required = wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::PUSH_CONSTANTS;
    /// if features.supports(required) {
    ///     println!("GPU profiling with push constants available!");
    /// }
    /// # }
    /// ```
    #[inline]
    pub fn supports(&self, required: wgpu::Features) -> bool {
        self.raw.contains(required)
    }

    /// Get the best available texture compression format.
    ///
    /// Returns the recommended compression format based on platform conventions:
    /// - BC is preferred on desktop (highest quality, widest format support)
    /// - ASTC is preferred on mobile (flexible block sizes, HDR support)
    /// - ETC2 is fallback on mobile
    ///
    /// # Returns
    ///
    /// The name of the best available compression format, or "none".
    pub fn best_compression_format(&self) -> &'static str {
        // Desktop preference: BC > ASTC > ETC2
        if self.has_texture_compression_bc() {
            "BC"
        } else if self.has_texture_compression_astc() {
            "ASTC"
        } else if self.has_texture_compression_etc2() {
            "ETC2"
        } else {
            "none"
        }
    }
}

impl std::fmt::Display for AdapterFeatures {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Adapter Features ({} available):", self.count())?;
        writeln!(f, "  Tier: {}", self.tier().description())?;
        writeln!(f)?;
        writeln!(f, "  === Texture Compression ===")?;
        writeln!(f, "  BC (Desktop):        {}", self.has_texture_compression_bc())?;
        writeln!(f, "  ETC2 (Mobile):       {}", self.has_texture_compression_etc2())?;
        writeln!(f, "  ASTC:                {}", self.has_texture_compression_astc())?;
        writeln!(f, "  ASTC HDR:            {}", self.has_texture_compression_astc_hdr())?;
        writeln!(f, "  Best format:         {}", self.best_compression_format())?;
        writeln!(f)?;
        writeln!(f, "  === Rendering ===")?;
        writeln!(f, "  Indirect 1st inst:   {}", self.has_indirect_first_instance())?;
        writeln!(f, "  Multiview (VR):      {}", self.has_multiview())?;
        writeln!(f)?;
        writeln!(f, "  === Queries ===")?;
        writeln!(f, "  Timestamp:           {}", self.has_timestamp_query())?;
        writeln!(f, "  Pipeline stats:      {}", self.has_pipeline_statistics_query())?;
        writeln!(f)?;
        writeln!(f, "  === Shader ===")?;
        writeln!(f, "  Shader F16:          {}", self.has_shader_f16())?;
        writeln!(f, "  Push constants:      {}", self.has_push_constants())?;
        writeln!(f)?;
        writeln!(f, "  === Formats ===")?;
        writeln!(f, "  Depth clip control:  {}", self.has_depth_clip_control())?;
        writeln!(f, "  Depth32F+Stencil8:   {}", self.has_depth32float_stencil8())?;
        writeln!(f, "  RG11B10F renderable: {}", self.has_rg11b10ufloat_renderable())?;
        writeln!(f, "  BGRA8 storage:       {}", self.has_bgra8unorm_storage())?;
        writeln!(f, "  Float32 filterable:  {}", self.has_float32_filterable())?;
        writeln!(f, "  16-bit norm:         {}", self.has_texture_format_16bit_norm())?;
        Ok(())
    }
}

/// Summary of adapter features.
///
/// A compact representation of key feature availability for quick checks.
/// Use [`AdapterFeatures::summary()`] to create this from an adapter.
#[derive(Debug, Clone, Copy)]
pub struct FeaturesSummary {
    /// Computed feature tier.
    pub tier: FeatureTier,
    /// Total number of available features.
    pub total_count: u32,
    /// BC (Block Compression) support - desktop standard.
    pub has_compression_bc: bool,
    /// ETC2 compression support - mobile standard.
    pub has_compression_etc2: bool,
    /// ASTC compression support - modern mobile/desktop.
    pub has_compression_astc: bool,
    /// ASTC HDR compression support.
    pub has_compression_astc_hdr: bool,
    /// Timestamp query support - GPU profiling.
    pub has_timestamp_query: bool,
    /// Pipeline statistics query support.
    pub has_pipeline_statistics: bool,
    /// Shader F16 support - half-precision in shaders.
    pub has_shader_f16: bool,
    /// Push constants support - fast uniform updates.
    pub has_push_constants: bool,
    /// Multiview support - VR stereo rendering.
    pub has_multiview: bool,
    /// Indirect first instance support - GPU-driven rendering.
    pub has_indirect_first_instance: bool,
}

impl FeaturesSummary {
    /// Check if any texture compression is available.
    #[inline]
    pub fn has_any_compression(&self) -> bool {
        self.has_compression_bc || self.has_compression_etc2 || self.has_compression_astc
    }

    /// Check if GPU profiling is available.
    #[inline]
    pub fn has_profiling(&self) -> bool {
        self.has_timestamp_query
    }

    /// Check if GPU-driven rendering features are available.
    #[inline]
    pub fn has_gpu_driven(&self) -> bool {
        self.has_indirect_first_instance
    }
}

impl std::fmt::Display for FeaturesSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} ({} features) - Compression: BC={}, ETC2={}, ASTC={} | Profiling: {} | F16: {}",
            self.tier,
            self.total_count,
            self.has_compression_bc,
            self.has_compression_etc2,
            self.has_compression_astc,
            self.has_timestamp_query,
            self.has_shader_f16
        )
    }
}

/// Inspect adapter features and return formatted debug output.
///
/// This function queries the adapter's optional features and formats them
/// into a human-readable string organized by category.
///
/// # Arguments
///
/// * `adapter` - The wgpu adapter to inspect
///
/// # Returns
///
/// A formatted string containing feature availability information.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::device::inspect_features;
///
/// # async fn example() {
/// let instance = wgpu::Instance::default();
/// let adapter = instance
///     .request_adapter(&wgpu::RequestAdapterOptions::default())
///     .await
///     .unwrap();
///
/// println!("{}", inspect_features(&adapter));
/// # }
/// ```
pub fn inspect_features(adapter: &wgpu::Adapter) -> String {
    let features = AdapterFeatures::from_adapter(adapter);
    format!("{}", features)
}

// ============================================================================
// Adapter Limits
// ============================================================================

/// Extracted adapter limits with categorization for easy access.
///
/// This struct wraps `wgpu::Limits` and provides convenient accessor methods
/// organized by category (texture, buffer, bind group, compute, vertex).
///
/// # Example
///
/// ```no_run
/// use renderer_backend::device::AdapterLimits;
///
/// # async fn example() {
/// let instance = wgpu::Instance::default();
/// let adapter = instance
///     .request_adapter(&wgpu::RequestAdapterOptions::default())
///     .await
///     .unwrap();
///
/// let limits = AdapterLimits::from_adapter(&adapter);
/// println!("Max 2D texture size: {}", limits.max_texture_dimension_2d());
/// println!("Max buffer size: {} bytes", limits.max_buffer_size());
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct AdapterLimits {
    /// Raw wgpu Limits structure.
    pub raw: wgpu::Limits,
}

impl AdapterLimits {
    /// Extract limits from an adapter.
    ///
    /// # Arguments
    ///
    /// * `adapter` - The wgpu adapter to query limits from
    ///
    /// # Returns
    ///
    /// An `AdapterLimits` instance containing the adapter's hardware limits.
    pub fn from_adapter(adapter: &wgpu::Adapter) -> Self {
        Self {
            raw: adapter.limits(),
        }
    }

    // ========================================================================
    // Texture Limits
    // ========================================================================

    /// Maximum 1D texture dimension in texels.
    ///
    /// Typical values: 8192-16384 for most GPUs.
    #[inline]
    pub fn max_texture_dimension_1d(&self) -> u32 {
        self.raw.max_texture_dimension_1d
    }

    /// Maximum 2D texture dimension (width or height) in texels.
    ///
    /// Typical values: 8192-16384 for most GPUs.
    /// This affects maximum rendertarget size.
    #[inline]
    pub fn max_texture_dimension_2d(&self) -> u32 {
        self.raw.max_texture_dimension_2d
    }

    /// Maximum 3D texture dimension (width, height, or depth) in texels.
    ///
    /// Typical values: 2048-8192 for most GPUs.
    #[inline]
    pub fn max_texture_dimension_3d(&self) -> u32 {
        self.raw.max_texture_dimension_3d
    }

    /// Maximum number of array layers in a texture array.
    ///
    /// Typical values: 256-2048 for most GPUs.
    #[inline]
    pub fn max_texture_array_layers(&self) -> u32 {
        self.raw.max_texture_array_layers
    }

    // ========================================================================
    // Buffer Limits
    // ========================================================================

    /// Maximum buffer size in bytes.
    ///
    /// This is the maximum size of a single buffer allocation.
    /// Typical values: 256MB-2GB+ depending on GPU memory.
    #[inline]
    pub fn max_buffer_size(&self) -> u64 {
        self.raw.max_buffer_size
    }

    /// Maximum uniform buffer binding size in bytes.
    ///
    /// This limits how much data can be bound as a uniform buffer.
    /// Typical values: 16KB-64KB for most GPUs.
    #[inline]
    pub fn max_uniform_buffer_binding_size(&self) -> u32 {
        self.raw.max_uniform_buffer_binding_size
    }

    /// Maximum storage buffer binding size in bytes.
    ///
    /// Storage buffers can typically be much larger than uniform buffers.
    /// Typical values: 128MB-2GB for most GPUs.
    #[inline]
    pub fn max_storage_buffer_binding_size(&self) -> u32 {
        self.raw.max_storage_buffer_binding_size
    }

    // ========================================================================
    // Bind Group Limits
    // ========================================================================

    /// Maximum number of bind groups that can be used simultaneously.
    ///
    /// WebGPU minimum: 4. Typical values: 4-8.
    #[inline]
    pub fn max_bind_groups(&self) -> u32 {
        self.raw.max_bind_groups
    }

    /// Maximum number of bindings per bind group.
    ///
    /// This includes all binding types (buffers, textures, samplers).
    /// Typical values: 640-1000 for most GPUs.
    #[inline]
    pub fn max_bindings_per_bind_group(&self) -> u32 {
        self.raw.max_bindings_per_bind_group
    }

    /// Maximum number of dynamic uniform buffers per pipeline layout.
    ///
    /// Dynamic buffers allow changing the buffer offset without rebinding.
    /// Typical values: 8-16 for most GPUs.
    #[inline]
    pub fn max_dynamic_uniform_buffers_per_pipeline_layout(&self) -> u32 {
        self.raw.max_dynamic_uniform_buffers_per_pipeline_layout
    }

    /// Maximum number of dynamic storage buffers per pipeline layout.
    ///
    /// Dynamic buffers allow changing the buffer offset without rebinding.
    /// Typical values: 4-8 for most GPUs.
    #[inline]
    pub fn max_dynamic_storage_buffers_per_pipeline_layout(&self) -> u32 {
        self.raw.max_dynamic_storage_buffers_per_pipeline_layout
    }

    /// Maximum number of sampled textures per shader stage.
    ///
    /// This limits textures that can be sampled in a single shader.
    /// Typical values: 16-128 for most GPUs.
    #[inline]
    pub fn max_sampled_textures_per_shader_stage(&self) -> u32 {
        self.raw.max_sampled_textures_per_shader_stage
    }

    /// Maximum number of samplers per shader stage.
    ///
    /// Typical values: 16 for most GPUs (WebGPU minimum).
    #[inline]
    pub fn max_samplers_per_shader_stage(&self) -> u32 {
        self.raw.max_samplers_per_shader_stage
    }

    /// Maximum number of storage buffers per shader stage.
    ///
    /// Storage buffers provide read/write access from shaders.
    /// Typical values: 8-64 for most GPUs.
    #[inline]
    pub fn max_storage_buffers_per_shader_stage(&self) -> u32 {
        self.raw.max_storage_buffers_per_shader_stage
    }

    /// Maximum number of storage textures per shader stage.
    ///
    /// Storage textures provide read/write image access from shaders.
    /// Typical values: 4-8 for most GPUs.
    #[inline]
    pub fn max_storage_textures_per_shader_stage(&self) -> u32 {
        self.raw.max_storage_textures_per_shader_stage
    }

    /// Maximum number of uniform buffers per shader stage.
    ///
    /// Typical values: 12-14 for most GPUs.
    #[inline]
    pub fn max_uniform_buffers_per_shader_stage(&self) -> u32 {
        self.raw.max_uniform_buffers_per_shader_stage
    }

    // ========================================================================
    // Compute Limits
    // ========================================================================

    /// Maximum compute workgroup size in X dimension.
    ///
    /// Typical values: 256-1024 for most GPUs.
    #[inline]
    pub fn max_compute_workgroup_size_x(&self) -> u32 {
        self.raw.max_compute_workgroup_size_x
    }

    /// Maximum compute workgroup size in Y dimension.
    ///
    /// Typical values: 256-1024 for most GPUs.
    #[inline]
    pub fn max_compute_workgroup_size_y(&self) -> u32 {
        self.raw.max_compute_workgroup_size_y
    }

    /// Maximum compute workgroup size in Z dimension.
    ///
    /// Typical values: 64-256 for most GPUs.
    #[inline]
    pub fn max_compute_workgroup_size_z(&self) -> u32 {
        self.raw.max_compute_workgroup_size_z
    }

    /// Maximum compute invocations per workgroup.
    ///
    /// This is the product of all three workgroup dimensions.
    /// Typical values: 256-1024 for most GPUs.
    #[inline]
    pub fn max_compute_invocations_per_workgroup(&self) -> u32 {
        self.raw.max_compute_invocations_per_workgroup
    }

    /// Maximum compute workgroups per dimension.
    ///
    /// This limits the dispatch size in any dimension.
    /// Typical values: 65535 for most GPUs.
    #[inline]
    pub fn max_compute_workgroups_per_dimension(&self) -> u32 {
        self.raw.max_compute_workgroups_per_dimension
    }

    // ========================================================================
    // Vertex Limits
    // ========================================================================

    /// Maximum number of vertex buffers that can be bound simultaneously.
    ///
    /// Typical values: 8-16 for most GPUs.
    #[inline]
    pub fn max_vertex_buffers(&self) -> u32 {
        self.raw.max_vertex_buffers
    }

    /// Maximum number of vertex attributes.
    ///
    /// This limits the total attributes across all vertex buffers.
    /// Typical values: 16-32 for most GPUs.
    #[inline]
    pub fn max_vertex_attributes(&self) -> u32 {
        self.raw.max_vertex_attributes
    }

    /// Maximum vertex buffer array stride in bytes.
    ///
    /// This is the maximum step size between vertices in a buffer.
    /// Typical values: 2048-4096 for most GPUs.
    #[inline]
    pub fn max_vertex_buffer_array_stride(&self) -> u32 {
        self.raw.max_vertex_buffer_array_stride
    }

    // ========================================================================
    // Additional Limits
    // ========================================================================

    /// Minimum uniform buffer offset alignment in bytes.
    ///
    /// When binding a uniform buffer with an offset, the offset must be
    /// a multiple of this value. Typical values: 256 for most GPUs.
    #[inline]
    pub fn min_uniform_buffer_offset_alignment(&self) -> u32 {
        self.raw.min_uniform_buffer_offset_alignment
    }

    /// Minimum storage buffer offset alignment in bytes.
    ///
    /// When binding a storage buffer with an offset, the offset must be
    /// a multiple of this value. Typical values: 256 for most GPUs.
    #[inline]
    pub fn min_storage_buffer_offset_alignment(&self) -> u32 {
        self.raw.min_storage_buffer_offset_alignment
    }

    /// Maximum inter-stage shader components.
    ///
    /// This limits the data passed between shader stages (e.g., vertex to fragment).
    /// Typical values: 60-128 for most GPUs.
    #[inline]
    pub fn max_inter_stage_shader_components(&self) -> u32 {
        self.raw.max_inter_stage_shader_components
    }

    /// Maximum color attachments per render pass.
    ///
    /// Typical values: 8 for most GPUs.
    #[inline]
    pub fn max_color_attachments(&self) -> u32 {
        self.raw.max_color_attachments
    }

    /// Maximum color attachment bytes per sample.
    ///
    /// This limits the total size of all color attachments combined.
    /// Typical values: 32 for most GPUs.
    #[inline]
    pub fn max_color_attachment_bytes_per_sample(&self) -> u32 {
        self.raw.max_color_attachment_bytes_per_sample
    }

    // ========================================================================
    // Utility Methods
    // ========================================================================

    /// Check if the limits meet WebGPU minimum requirements.
    ///
    /// Returns `true` if all limits meet or exceed the WebGPU spec minimums.
    pub fn meets_webgpu_minimum(&self) -> bool {
        let min = wgpu::Limits::downlevel_webgl2_defaults();
        self.raw.max_texture_dimension_1d >= min.max_texture_dimension_1d
            && self.raw.max_texture_dimension_2d >= min.max_texture_dimension_2d
            && self.raw.max_bind_groups >= min.max_bind_groups
    }

    /// Get a categorized summary of all limits.
    ///
    /// Returns a struct with limits organized by category for easy access.
    pub fn summary(&self) -> LimitsSummary {
        LimitsSummary {
            texture: TextureLimits {
                max_1d: self.max_texture_dimension_1d(),
                max_2d: self.max_texture_dimension_2d(),
                max_3d: self.max_texture_dimension_3d(),
                max_array_layers: self.max_texture_array_layers(),
            },
            buffer: BufferLimits {
                max_size: self.max_buffer_size(),
                max_uniform_binding: self.max_uniform_buffer_binding_size(),
                max_storage_binding: self.max_storage_buffer_binding_size(),
                min_uniform_offset_alignment: self.min_uniform_buffer_offset_alignment(),
                min_storage_offset_alignment: self.min_storage_buffer_offset_alignment(),
            },
            bind_group: BindGroupLimits {
                max_bind_groups: self.max_bind_groups(),
                max_bindings_per_group: self.max_bindings_per_bind_group(),
                max_dynamic_uniform_buffers: self.max_dynamic_uniform_buffers_per_pipeline_layout(),
                max_dynamic_storage_buffers: self.max_dynamic_storage_buffers_per_pipeline_layout(),
                max_sampled_textures: self.max_sampled_textures_per_shader_stage(),
                max_samplers: self.max_samplers_per_shader_stage(),
                max_storage_buffers: self.max_storage_buffers_per_shader_stage(),
                max_storage_textures: self.max_storage_textures_per_shader_stage(),
                max_uniform_buffers: self.max_uniform_buffers_per_shader_stage(),
            },
            compute: ComputeLimits {
                max_workgroup_size_x: self.max_compute_workgroup_size_x(),
                max_workgroup_size_y: self.max_compute_workgroup_size_y(),
                max_workgroup_size_z: self.max_compute_workgroup_size_z(),
                max_invocations_per_workgroup: self.max_compute_invocations_per_workgroup(),
                max_workgroups_per_dimension: self.max_compute_workgroups_per_dimension(),
            },
            vertex: VertexLimits {
                max_buffers: self.max_vertex_buffers(),
                max_attributes: self.max_vertex_attributes(),
                max_buffer_array_stride: self.max_vertex_buffer_array_stride(),
            },
        }
    }
}

impl std::fmt::Display for AdapterLimits {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", format_limits_internal(self))
    }
}

/// Categorized summary of all adapter limits.
#[derive(Debug, Clone)]
pub struct LimitsSummary {
    /// Texture-related limits.
    pub texture: TextureLimits,
    /// Buffer-related limits.
    pub buffer: BufferLimits,
    /// Bind group-related limits.
    pub bind_group: BindGroupLimits,
    /// Compute-related limits.
    pub compute: ComputeLimits,
    /// Vertex-related limits.
    pub vertex: VertexLimits,
}

/// Texture limits summary.
#[derive(Debug, Clone, Copy)]
pub struct TextureLimits {
    /// Maximum 1D texture dimension.
    pub max_1d: u32,
    /// Maximum 2D texture dimension.
    pub max_2d: u32,
    /// Maximum 3D texture dimension.
    pub max_3d: u32,
    /// Maximum texture array layers.
    pub max_array_layers: u32,
}

/// Buffer limits summary.
#[derive(Debug, Clone, Copy)]
pub struct BufferLimits {
    /// Maximum buffer size in bytes.
    pub max_size: u64,
    /// Maximum uniform buffer binding size.
    pub max_uniform_binding: u32,
    /// Maximum storage buffer binding size.
    pub max_storage_binding: u32,
    /// Minimum uniform buffer offset alignment.
    pub min_uniform_offset_alignment: u32,
    /// Minimum storage buffer offset alignment.
    pub min_storage_offset_alignment: u32,
}

/// Bind group limits summary.
#[derive(Debug, Clone, Copy)]
pub struct BindGroupLimits {
    /// Maximum bind groups.
    pub max_bind_groups: u32,
    /// Maximum bindings per bind group.
    pub max_bindings_per_group: u32,
    /// Maximum dynamic uniform buffers.
    pub max_dynamic_uniform_buffers: u32,
    /// Maximum dynamic storage buffers.
    pub max_dynamic_storage_buffers: u32,
    /// Maximum sampled textures per stage.
    pub max_sampled_textures: u32,
    /// Maximum samplers per stage.
    pub max_samplers: u32,
    /// Maximum storage buffers per stage.
    pub max_storage_buffers: u32,
    /// Maximum storage textures per stage.
    pub max_storage_textures: u32,
    /// Maximum uniform buffers per stage.
    pub max_uniform_buffers: u32,
}

/// Compute limits summary.
#[derive(Debug, Clone, Copy)]
pub struct ComputeLimits {
    /// Maximum workgroup size X.
    pub max_workgroup_size_x: u32,
    /// Maximum workgroup size Y.
    pub max_workgroup_size_y: u32,
    /// Maximum workgroup size Z.
    pub max_workgroup_size_z: u32,
    /// Maximum invocations per workgroup.
    pub max_invocations_per_workgroup: u32,
    /// Maximum workgroups per dimension.
    pub max_workgroups_per_dimension: u32,
}

/// Vertex limits summary.
#[derive(Debug, Clone, Copy)]
pub struct VertexLimits {
    /// Maximum vertex buffers.
    pub max_buffers: u32,
    /// Maximum vertex attributes.
    pub max_attributes: u32,
    /// Maximum vertex buffer array stride.
    pub max_buffer_array_stride: u32,
}

/// Inspect adapter limits and return formatted debug output.
///
/// This function queries the adapter's hardware limits and formats them
/// into a human-readable string organized by category.
///
/// # Arguments
///
/// * `adapter` - The wgpu adapter to inspect
///
/// # Returns
///
/// A formatted string containing all adapter limits.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::device::inspect_limits;
///
/// # async fn example() {
/// let instance = wgpu::Instance::default();
/// let adapter = instance
///     .request_adapter(&wgpu::RequestAdapterOptions::default())
///     .await
///     .unwrap();
///
/// println!("{}", inspect_limits(&adapter));
/// # }
/// ```
pub fn inspect_limits(adapter: &wgpu::Adapter) -> String {
    let limits = AdapterLimits::from_adapter(adapter);
    format_limits_internal(&limits)
}

// ============================================================================
// Adapter Selection
// ============================================================================

/// Weights for scoring adapters by device type.
///
/// Higher weights indicate preference for that device type during selection.
/// The default weights favor discrete GPUs heavily, followed by integrated GPUs.
///
/// # Example
///
/// ```
/// use renderer_backend::device::DeviceTypeWeights;
///
/// let weights = DeviceTypeWeights::default();
/// assert!(weights.discrete > weights.integrated);
///
/// // Custom weights for power-saving mode (prefer integrated)
/// let power_save = DeviceTypeWeights {
///     discrete: 500,
///     integrated: 1000,
///     virtual_gpu: 200,
///     cpu: 100,
///     other: 50,
/// };
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceTypeWeights {
    /// Weight for discrete (dedicated) GPUs.
    pub discrete: u32,
    /// Weight for integrated GPUs.
    pub integrated: u32,
    /// Weight for virtual GPUs (in VMs).
    pub virtual_gpu: u32,
    /// Weight for CPU/software rendering.
    pub cpu: u32,
    /// Weight for unknown device types.
    pub other: u32,
}

impl Default for DeviceTypeWeights {
    fn default() -> Self {
        Self {
            discrete: 1000,
            integrated: 500,
            virtual_gpu: 300,
            cpu: 100,
            other: 50,
        }
    }
}

impl DeviceTypeWeights {
    /// Get weight for a specific device type.
    pub fn weight_for(&self, device_type: DeviceType) -> u32 {
        match device_type {
            DeviceType::DiscreteGpu => self.discrete,
            DeviceType::IntegratedGpu => self.integrated,
            DeviceType::VirtualGpu => self.virtual_gpu,
            DeviceType::Cpu => self.cpu,
            DeviceType::Other => self.other,
        }
    }

    /// Create weights that prefer integrated GPUs (power-saving mode).
    pub fn power_saving() -> Self {
        Self {
            discrete: 500,
            integrated: 1000,
            virtual_gpu: 200,
            cpu: 100,
            other: 50,
        }
    }

    /// Create weights that strongly prefer discrete GPUs (performance mode).
    pub fn performance() -> Self {
        Self {
            discrete: 2000,
            integrated: 400,
            virtual_gpu: 200,
            cpu: 50,
            other: 25,
        }
    }
}

/// Entry in the adapter blacklist for filtering out problematic adapters.
///
/// Blacklist entries can match by vendor, device name substring, or both.
/// Use this to filter out adapters with known driver bugs or compatibility issues.
///
/// # Example
///
/// ```
/// use renderer_backend::device::{AdapterBlacklistEntry, Vendor};
///
/// // Block all software renderers from a specific vendor
/// let entry = AdapterBlacklistEntry::new()
///     .with_vendor(Vendor::Microsoft)
///     .with_reason("WARP software renderer too slow");
///
/// // Block a specific problematic GPU by name
/// let entry = AdapterBlacklistEntry::new()
///     .with_name_contains("Buggy GPU 3000")
///     .with_reason("Known driver crash on Vulkan 1.3");
/// ```
#[derive(Debug, Clone)]
pub struct AdapterBlacklistEntry {
    /// Vendor to blacklist (if specified).
    pub vendor: Option<Vendor>,
    /// Substring that must appear in the adapter name (if specified).
    pub name_contains: Option<String>,
    /// Reason for blacklisting (for logging/debugging).
    pub reason: String,
}

impl AdapterBlacklistEntry {
    /// Create an empty blacklist entry.
    pub fn new() -> Self {
        Self {
            vendor: None,
            name_contains: None,
            reason: String::new(),
        }
    }

    /// Add a vendor filter.
    pub fn with_vendor(mut self, vendor: Vendor) -> Self {
        self.vendor = Some(vendor);
        self
    }

    /// Add a name substring filter.
    pub fn with_name_contains(mut self, name: impl Into<String>) -> Self {
        self.name_contains = Some(name.into());
        self
    }

    /// Set the reason for blacklisting.
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = reason.into();
        self
    }

    /// Check if an adapter matches this blacklist entry.
    ///
    /// Both vendor and name_contains must match if specified (AND logic).
    pub fn matches(&self, info: &AdapterInfo) -> bool {
        let vendor_matches = self
            .vendor
            .map(|v| Vendor::from_id(info.vendor) == v)
            .unwrap_or(true);

        let name_matches = self
            .name_contains
            .as_ref()
            .map(|n| info.name.to_lowercase().contains(&n.to_lowercase()))
            .unwrap_or(true);

        // Must have at least one filter specified to match
        let has_filter = self.vendor.is_some() || self.name_contains.is_some();
        has_filter && vendor_matches && name_matches
    }
}

impl Default for AdapterBlacklistEntry {
    fn default() -> Self {
        Self::new()
    }
}

/// Breakdown of how an adapter's score was calculated.
///
/// Useful for debugging and understanding why a particular adapter was chosen.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::device::AdapterSelector;
///
/// let selector = AdapterSelector::new();
/// // After selection, you can inspect the score breakdown
/// ```
#[derive(Debug, Clone)]
pub struct AdapterScore {
    /// Score contribution from device type.
    pub device_type_score: u32,
    /// Score contribution from feature tier.
    pub feature_score: u32,
    /// Score contribution from hardware limits.
    pub limits_score: u32,
    /// Score bonus from vendor preference.
    pub vendor_bonus: u32,
    /// Total combined score.
    pub total: u32,
    /// Whether this adapter was blacklisted.
    pub blacklisted: bool,
    /// Blacklist reason (if blacklisted).
    pub blacklist_reason: Option<String>,
}

impl AdapterScore {
    /// Create a zero score.
    fn zero() -> Self {
        Self {
            device_type_score: 0,
            feature_score: 0,
            limits_score: 0,
            vendor_bonus: 0,
            total: 0,
            blacklisted: false,
            blacklist_reason: None,
        }
    }

    /// Create a blacklisted score.
    fn blacklisted(reason: String) -> Self {
        Self {
            device_type_score: 0,
            feature_score: 0,
            limits_score: 0,
            vendor_bonus: 0,
            total: 0,
            blacklisted: true,
            blacklist_reason: Some(reason),
        }
    }

    /// Calculate total from components.
    fn calculate_total(&mut self) {
        self.total = self.device_type_score + self.feature_score + self.limits_score + self.vendor_bonus;
    }
}

impl std::fmt::Display for AdapterScore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.blacklisted {
            write!(
                f,
                "BLACKLISTED (reason: {})",
                self.blacklist_reason.as_deref().unwrap_or("unknown")
            )
        } else {
            write!(
                f,
                "{} (type: {}, features: {}, limits: {}, vendor: {})",
                self.total,
                self.device_type_score,
                self.feature_score,
                self.limits_score,
                self.vendor_bonus
            )
        }
    }
}

/// Result of adapter selection including the adapter and scoring details.
///
/// Provides full transparency into why an adapter was selected.
#[derive(Debug)]
pub struct SelectionResult<'a> {
    /// The selected adapter.
    pub adapter: &'a Adapter,
    /// Score breakdown for the selected adapter.
    pub score: AdapterScore,
    /// Scores for all evaluated adapters (for debugging).
    pub all_scores: Vec<(String, AdapterScore)>,
}

impl<'a> SelectionResult<'a> {
    /// Get the adapter's name.
    pub fn adapter_name(&self) -> String {
        self.adapter.get_info().name.clone()
    }

    /// Log selection results for debugging.
    pub fn log_results(&self) {
        info!("AdapterSelector: Selected '{}'", self.adapter_name());
        debug!("  Score breakdown: {}", self.score);
        debug!("  All adapter scores:");
        for (name, score) in &self.all_scores {
            debug!("    {}: {}", name, score);
        }
    }
}

/// Scoring-based adapter selector for choosing the best GPU.
///
/// The selector evaluates each available adapter and scores it based on:
/// - Device type (discrete > integrated > software)
/// - Available features (higher tier = higher score)
/// - Hardware limits (larger limits = higher score)
/// - Vendor preference (optional bonus for preferred vendor)
/// - Blacklist filtering (exclude known-problematic adapters)
///
/// # Example
///
/// ```no_run
/// use renderer_backend::device::{AdapterSelector, Vendor, AdapterBlacklistEntry};
///
/// // Create selector with NVIDIA preference and blacklist
/// let selector = AdapterSelector::new()
///     .with_vendor_preference(Vendor::Nvidia)
///     .with_blacklist_entry(
///         AdapterBlacklistEntry::new()
///             .with_vendor(Vendor::Microsoft)
///             .with_reason("Software renderer")
///     );
///
/// // Select from available adapters
/// let instance = wgpu::Instance::default();
/// let adapters: Vec<_> = instance.enumerate_adapters(wgpu::Backends::PRIMARY);
///
/// if let Some(result) = selector.select(&adapters) {
///     println!("Selected: {}", result.adapter_name());
///     result.log_results();
/// }
/// ```
#[derive(Debug, Clone)]
pub struct AdapterSelector {
    /// Blacklist entries for filtering adapters.
    blacklist: Vec<AdapterBlacklistEntry>,
    /// Preferred vendor (gets bonus score).
    vendor_preference: Option<Vendor>,
    /// Weights for device type scoring.
    device_type_weights: DeviceTypeWeights,
    /// Weight multiplier for feature tier scoring.
    feature_weight: u32,
    /// Weight multiplier for limits scoring.
    limit_weight: u32,
    /// Bonus score for preferred vendor.
    vendor_preference_bonus: u32,
}

impl Default for AdapterSelector {
    fn default() -> Self {
        Self::new()
    }
}

impl AdapterSelector {
    /// Create a new adapter selector with default settings.
    ///
    /// Default settings:
    /// - No blacklist entries
    /// - No vendor preference
    /// - Default device type weights (discrete > integrated > software)
    /// - Feature weight: 100 per tier level
    /// - Limit weight: 1 per unit
    /// - Vendor preference bonus: 200
    pub fn new() -> Self {
        Self {
            blacklist: Vec::new(),
            vendor_preference: None,
            device_type_weights: DeviceTypeWeights::default(),
            feature_weight: 100,
            limit_weight: 1,
            vendor_preference_bonus: 200,
        }
    }

    /// Set the vendor preference (gets bonus score).
    ///
    /// # Arguments
    ///
    /// * `vendor` - The preferred GPU vendor
    pub fn with_vendor_preference(mut self, vendor: Vendor) -> Self {
        self.vendor_preference = Some(vendor);
        self
    }

    /// Add a blacklist entry.
    ///
    /// Blacklisted adapters receive a score of 0 and are only selected
    /// as a last resort (if all adapters are blacklisted).
    ///
    /// # Arguments
    ///
    /// * `entry` - The blacklist entry to add
    pub fn with_blacklist_entry(mut self, entry: AdapterBlacklistEntry) -> Self {
        self.blacklist.push(entry);
        self
    }

    /// Set custom device type weights.
    ///
    /// # Arguments
    ///
    /// * `weights` - The custom weights to use
    pub fn with_device_type_weights(mut self, weights: DeviceTypeWeights) -> Self {
        self.device_type_weights = weights;
        self
    }

    /// Set the feature weight multiplier.
    ///
    /// Higher values make feature tier more important in selection.
    ///
    /// # Arguments
    ///
    /// * `weight` - Points per feature tier level (default: 100)
    pub fn with_feature_weight(mut self, weight: u32) -> Self {
        self.feature_weight = weight;
        self
    }

    /// Set the limits weight multiplier.
    ///
    /// Higher values make hardware limits more important in selection.
    ///
    /// # Arguments
    ///
    /// * `weight` - Weight for limit scoring (default: 1)
    pub fn with_limit_weight(mut self, weight: u32) -> Self {
        self.limit_weight = weight;
        self
    }

    /// Set the vendor preference bonus.
    ///
    /// # Arguments
    ///
    /// * `bonus` - Bonus points for preferred vendor (default: 200)
    pub fn with_vendor_preference_bonus(mut self, bonus: u32) -> Self {
        self.vendor_preference_bonus = bonus;
        self
    }

    /// Add multiple blacklist entries at once.
    pub fn with_blacklist(mut self, entries: Vec<AdapterBlacklistEntry>) -> Self {
        self.blacklist.extend(entries);
        self
    }

    /// Check if an adapter is blacklisted.
    ///
    /// Returns the reason if blacklisted, None otherwise.
    fn is_blacklisted(&self, info: &AdapterInfo) -> Option<String> {
        for entry in &self.blacklist {
            if entry.matches(info) {
                return Some(entry.reason.clone());
            }
        }
        None
    }

    /// Score an adapter.
    ///
    /// Returns a detailed score breakdown for the adapter.
    pub fn score_adapter(&self, adapter: &Adapter) -> AdapterScore {
        let info = adapter.get_info();

        // Check blacklist first
        if let Some(reason) = self.is_blacklisted(&info) {
            debug!(
                "AdapterSelector: Blacklisted '{}' - {}",
                info.name, reason
            );
            return AdapterScore::blacklisted(reason);
        }

        let mut score = AdapterScore::zero();

        // Score by device type
        score.device_type_score = self.device_type_weights.weight_for(info.device_type);

        // Score by feature tier
        let features = AdapterFeatures::from_adapter(adapter);
        score.feature_score = match features.tier() {
            FeatureTier::Full => self.feature_weight * 4,
            FeatureTier::Advanced => self.feature_weight * 3,
            FeatureTier::Standard => self.feature_weight * 2,
            FeatureTier::Minimal => self.feature_weight,
        };

        // Score by limits (normalized key limits)
        let limits = adapter.limits();
        score.limits_score = self.score_limits(&limits);

        // Vendor preference bonus
        if let Some(preferred) = self.vendor_preference {
            if Vendor::from_id(info.vendor) == preferred {
                score.vendor_bonus = self.vendor_preference_bonus;
            }
        }

        score.calculate_total();
        score
    }

    /// Score hardware limits.
    ///
    /// Uses key limits that indicate GPU capability:
    /// - Max texture dimension 2D (normalized to 0-100)
    /// - Max buffer size (normalized to 0-100)
    /// - Max compute workgroup invocations (normalized to 0-50)
    /// - Max storage buffer binding size (normalized to 0-50)
    fn score_limits(&self, limits: &wgpu::Limits) -> u32 {
        // Normalize texture size: 8192 = 50 points, 16384+ = 100 points
        let texture_score = (limits.max_texture_dimension_2d.min(16384) / 164) as u32;

        // Normalize buffer size: 256MB = 50 points, 1GB+ = 100 points
        let buffer_score = ((limits.max_buffer_size / (10 * 1024 * 1024)).min(100)) as u32;

        // Normalize compute: 256 invocations = 25 points, 1024+ = 50 points
        let compute_score = (limits.max_compute_invocations_per_workgroup / 20).min(50);

        // Normalize storage binding: 128MB = 25 points, 512MB+ = 50 points
        let storage_score = (limits.max_storage_buffer_binding_size / (10 * 1024 * 1024)).min(50);

        (texture_score + buffer_score + compute_score + storage_score) * self.limit_weight
    }

    /// Select the best adapter from a slice.
    ///
    /// Returns the adapter with the highest score. If all adapters are blacklisted,
    /// falls back to the first non-blacklisted adapter or the first adapter overall.
    ///
    /// # Arguments
    ///
    /// * `adapters` - Slice of adapters to choose from
    ///
    /// # Returns
    ///
    /// `None` if the slice is empty, otherwise `Some(SelectionResult)` with the
    /// selected adapter and full scoring information.
    pub fn select<'a>(&self, adapters: &'a [Adapter]) -> Option<SelectionResult<'a>> {
        if adapters.is_empty() {
            warn!("AdapterSelector: No adapters provided");
            return None;
        }

        // Score all adapters
        let mut scored: Vec<(&'a Adapter, AdapterScore)> = adapters
            .iter()
            .map(|a| (a, self.score_adapter(a)))
            .collect();

        // Build all_scores for debugging
        let all_scores: Vec<(String, AdapterScore)> = scored
            .iter()
            .map(|(a, s)| (a.get_info().name.clone(), s.clone()))
            .collect();

        // Sort by score descending (non-blacklisted first, then by total score)
        scored.sort_by(|(_, a), (_, b)| {
            // Non-blacklisted > blacklisted
            match (a.blacklisted, b.blacklisted) {
                (true, false) => std::cmp::Ordering::Greater,
                (false, true) => std::cmp::Ordering::Less,
                _ => b.total.cmp(&a.total),
            }
        });

        // Take the best (first after sorting)
        let (adapter, score) = scored.into_iter().next()?;

        Some(SelectionResult {
            adapter,
            score,
            all_scores,
        })
    }

    /// Select the best adapter, returning just the adapter reference.
    ///
    /// Convenience method when you don't need the scoring details.
    pub fn select_adapter<'a>(&self, adapters: &'a [Adapter]) -> Option<&'a Adapter> {
        self.select(adapters).map(|r| r.adapter)
    }
}

/// Internal function to format limits (used by both `inspect_limits` and `Display`).
fn format_limits_internal(limits: &AdapterLimits) -> String {
    format!(
        "Adapter Limits:\n\
         \n\
         === Texture Limits ===\n\
         Max 1D Dimension:        {}\n\
         Max 2D Dimension:        {}\n\
         Max 3D Dimension:        {}\n\
         Max Array Layers:        {}\n\
         \n\
         === Buffer Limits ===\n\
         Max Buffer Size:         {} bytes ({:.2} GB)\n\
         Max Uniform Binding:     {} bytes ({:.2} KB)\n\
         Max Storage Binding:     {} bytes ({:.2} MB)\n\
         Min Uniform Alignment:   {} bytes\n\
         Min Storage Alignment:   {} bytes\n\
         \n\
         === Bind Group Limits ===\n\
         Max Bind Groups:         {}\n\
         Max Bindings/Group:      {}\n\
         Max Dynamic Uniforms:    {}\n\
         Max Dynamic Storage:     {}\n\
         Max Sampled Textures:    {} (per stage)\n\
         Max Samplers:            {} (per stage)\n\
         Max Storage Buffers:     {} (per stage)\n\
         Max Storage Textures:    {} (per stage)\n\
         Max Uniform Buffers:     {} (per stage)\n\
         \n\
         === Compute Limits ===\n\
         Max Workgroup Size:      ({}, {}, {})\n\
         Max Invocations/WG:      {}\n\
         Max Workgroups/Dim:      {}\n\
         \n\
         === Vertex Limits ===\n\
         Max Vertex Buffers:      {}\n\
         Max Vertex Attributes:   {}\n\
         Max Buffer Stride:       {} bytes\n\
         \n\
         === Other Limits ===\n\
         Max Inter-Stage Comps:   {}\n\
         Max Color Attachments:   {}\n\
         Max Color Bytes/Sample:  {}",
        limits.max_texture_dimension_1d(),
        limits.max_texture_dimension_2d(),
        limits.max_texture_dimension_3d(),
        limits.max_texture_array_layers(),
        limits.max_buffer_size(),
        limits.max_buffer_size() as f64 / (1024.0 * 1024.0 * 1024.0),
        limits.max_uniform_buffer_binding_size(),
        limits.max_uniform_buffer_binding_size() as f64 / 1024.0,
        limits.max_storage_buffer_binding_size(),
        limits.max_storage_buffer_binding_size() as f64 / (1024.0 * 1024.0),
        limits.min_uniform_buffer_offset_alignment(),
        limits.min_storage_buffer_offset_alignment(),
        limits.max_bind_groups(),
        limits.max_bindings_per_bind_group(),
        limits.max_dynamic_uniform_buffers_per_pipeline_layout(),
        limits.max_dynamic_storage_buffers_per_pipeline_layout(),
        limits.max_sampled_textures_per_shader_stage(),
        limits.max_samplers_per_shader_stage(),
        limits.max_storage_buffers_per_shader_stage(),
        limits.max_storage_textures_per_shader_stage(),
        limits.max_uniform_buffers_per_shader_stage(),
        limits.max_compute_workgroup_size_x(),
        limits.max_compute_workgroup_size_y(),
        limits.max_compute_workgroup_size_z(),
        limits.max_compute_invocations_per_workgroup(),
        limits.max_compute_workgroups_per_dimension(),
        limits.max_vertex_buffers(),
        limits.max_vertex_attributes(),
        limits.max_vertex_buffer_array_stride(),
        limits.max_inter_stage_shader_components(),
        limits.max_color_attachments(),
        limits.max_color_attachment_bytes_per_sample()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_counts_default() {
        let counts = BackendCounts::default();
        assert_eq!(counts.vulkan, 0);
        assert_eq!(counts.metal, 0);
        assert_eq!(counts.dx12, 0);
        assert_eq!(counts.gl, 0);
        assert_eq!(counts.webgpu, 0);
        assert_eq!(counts.total(), 0);
        assert!(counts.is_empty());
    }

    #[test]
    fn test_backend_counts_total() {
        let counts = BackendCounts {
            vulkan: 2,
            metal: 0,
            dx12: 1,
            gl: 1,
            webgpu: 0,
        };
        assert_eq!(counts.total(), 4);
        assert!(!counts.is_empty());
    }

    #[test]
    fn test_backend_counts_summary() {
        let counts = BackendCounts::default();
        assert_eq!(counts.summary(), "none");

        let counts = BackendCounts {
            vulkan: 1,
            metal: 0,
            dx12: 0,
            gl: 2,
            webgpu: 0,
        };
        assert_eq!(counts.summary(), "Vulkan: 1, GL: 2");

        let counts = BackendCounts {
            vulkan: 1,
            metal: 1,
            dx12: 1,
            gl: 1,
            webgpu: 1,
        };
        assert_eq!(
            counts.summary(),
            "Vulkan: 1, Metal: 1, DX12: 1, GL: 1, WebGPU: 1"
        );
    }

    #[test]
    fn test_enumeration_result_is_empty() {
        let result = EnumerationResult {
            adapters: vec![],
            backend_counts: BackendCounts::default(),
        };
        assert!(result.is_empty());
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_device_type_description() {
        assert_eq!(
            device_type_description(DeviceType::DiscreteGpu),
            "Discrete GPU (dedicated graphics card)"
        );
        assert_eq!(
            device_type_description(DeviceType::IntegratedGpu),
            "Integrated GPU (shared memory with CPU)"
        );
        assert_eq!(
            device_type_description(DeviceType::VirtualGpu),
            "Virtual GPU (virtualized environment)"
        );
        assert_eq!(
            device_type_description(DeviceType::Cpu),
            "CPU (software rendering)"
        );
        assert_eq!(
            device_type_description(DeviceType::Other),
            "Other (unknown device type)"
        );
    }

    #[test]
    fn test_enumerate_adapters_with_info_empty_backends() {
        // Empty backends should return no adapters
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::empty(),
            ..Default::default()
        });
        let result = enumerate_adapters_with_info(&instance, wgpu::Backends::empty());
        assert!(result.is_empty());
    }

    // ========================================================================
    // Vendor Tests
    // ========================================================================

    #[test]
    fn test_vendor_from_id_known_vendors() {
        assert_eq!(Vendor::from_id(0x10DE), Vendor::Nvidia);
        assert_eq!(Vendor::from_id(0x1002), Vendor::Amd);
        assert_eq!(Vendor::from_id(0x1022), Vendor::Amd); // AMD alternate
        assert_eq!(Vendor::from_id(0x8086), Vendor::Intel);
        assert_eq!(Vendor::from_id(0x106B), Vendor::Apple);
        assert_eq!(Vendor::from_id(0x13B5), Vendor::Arm);
        assert_eq!(Vendor::from_id(0x5143), Vendor::Qualcomm);
        assert_eq!(Vendor::from_id(0x1414), Vendor::Microsoft);
    }

    #[test]
    fn test_vendor_from_id_unknown() {
        assert_eq!(Vendor::from_id(0x9999), Vendor::Unknown(0x9999));
        assert_eq!(Vendor::from_id(0x0000), Vendor::Unknown(0x0000));
    }

    #[test]
    fn test_vendor_name() {
        assert_eq!(Vendor::Nvidia.name(), "NVIDIA");
        assert_eq!(Vendor::Amd.name(), "AMD");
        assert_eq!(Vendor::Intel.name(), "Intel");
        assert_eq!(Vendor::Apple.name(), "Apple");
        assert_eq!(Vendor::Arm.name(), "ARM");
        assert_eq!(Vendor::Qualcomm.name(), "Qualcomm");
        assert_eq!(Vendor::Microsoft.name(), "Microsoft");
        assert_eq!(Vendor::Unknown(0x1234).name(), "Unknown");
    }

    #[test]
    fn test_vendor_is_known() {
        assert!(Vendor::Nvidia.is_known());
        assert!(Vendor::Amd.is_known());
        assert!(Vendor::Intel.is_known());
        assert!(!Vendor::Unknown(0x1234).is_known());
    }

    #[test]
    fn test_vendor_id() {
        assert_eq!(Vendor::Nvidia.id(), 0x10DE);
        assert_eq!(Vendor::Amd.id(), 0x1002);
        assert_eq!(Vendor::Intel.id(), 0x8086);
        assert_eq!(Vendor::Apple.id(), 0x106B);
        assert_eq!(Vendor::Arm.id(), 0x13B5);
        assert_eq!(Vendor::Qualcomm.id(), 0x5143);
        assert_eq!(Vendor::Microsoft.id(), 0x1414);
        assert_eq!(Vendor::Unknown(0xABCD).id(), 0xABCD);
    }

    #[test]
    fn test_vendor_display() {
        assert_eq!(format!("{}", Vendor::Nvidia), "NVIDIA");
        assert_eq!(format!("{}", Vendor::Amd), "AMD");
        assert_eq!(format!("{}", Vendor::Unknown(0x1234)), "Unknown (0x1234)");
    }

    #[test]
    fn test_vendor_equality() {
        assert_eq!(Vendor::Nvidia, Vendor::Nvidia);
        assert_ne!(Vendor::Nvidia, Vendor::Amd);
        assert_eq!(Vendor::Unknown(0x1234), Vendor::Unknown(0x1234));
        assert_ne!(Vendor::Unknown(0x1234), Vendor::Unknown(0x5678));
    }

    #[test]
    fn test_vendor_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(Vendor::Nvidia);
        set.insert(Vendor::Amd);
        set.insert(Vendor::Unknown(0x1234));
        assert_eq!(set.len(), 3);
        assert!(set.contains(&Vendor::Nvidia));
        assert!(set.contains(&Vendor::Amd));
        assert!(set.contains(&Vendor::Unknown(0x1234)));
    }

    // ========================================================================
    // AdapterProperties Tests
    // ========================================================================

    #[test]
    fn test_device_type_short() {
        assert_eq!(device_type_short(DeviceType::DiscreteGpu), "Discrete GPU");
        assert_eq!(device_type_short(DeviceType::IntegratedGpu), "Integrated GPU");
        assert_eq!(device_type_short(DeviceType::VirtualGpu), "Virtual GPU");
        assert_eq!(device_type_short(DeviceType::Cpu), "Software");
        assert_eq!(device_type_short(DeviceType::Other), "Other");
    }

    #[test]
    fn test_adapter_properties_device_type_checks() {
        // Create a mock properties struct for testing
        let make_props = |dt: DeviceType| AdapterProperties {
            name: "Test GPU".to_string(),
            vendor: Vendor::Nvidia,
            vendor_id: 0x10DE,
            device_id: 0x1234,
            device_type: dt,
            backend: wgpu::Backend::Vulkan,
            driver: String::new(),
            driver_info: String::new(),
        };

        let discrete = make_props(DeviceType::DiscreteGpu);
        assert!(discrete.is_discrete());
        assert!(!discrete.is_integrated());
        assert!(!discrete.is_software());
        assert!(!discrete.is_virtual());

        let integrated = make_props(DeviceType::IntegratedGpu);
        assert!(!integrated.is_discrete());
        assert!(integrated.is_integrated());
        assert!(!integrated.is_software());
        assert!(!integrated.is_virtual());

        let software = make_props(DeviceType::Cpu);
        assert!(!software.is_discrete());
        assert!(!software.is_integrated());
        assert!(software.is_software());
        assert!(!software.is_virtual());

        let virtual_gpu = make_props(DeviceType::VirtualGpu);
        assert!(!virtual_gpu.is_discrete());
        assert!(!virtual_gpu.is_integrated());
        assert!(!virtual_gpu.is_software());
        assert!(virtual_gpu.is_virtual());
    }

    #[test]
    fn test_adapter_properties_description_no_driver() {
        let props = AdapterProperties {
            name: "GeForce RTX 4090".to_string(),
            vendor: Vendor::Nvidia,
            vendor_id: 0x10DE,
            device_id: 0x2684,
            device_type: DeviceType::DiscreteGpu,
            backend: wgpu::Backend::Vulkan,
            driver: String::new(),
            driver_info: String::new(),
        };

        let desc = props.description();
        assert!(desc.contains("GeForce RTX 4090"));
        assert!(desc.contains("Discrete GPU"));
        assert!(desc.contains("Vulkan"));
        assert!(!desc.contains("driver"));
    }

    #[test]
    fn test_adapter_properties_description_with_driver() {
        let props = AdapterProperties {
            name: "GeForce RTX 4090".to_string(),
            vendor: Vendor::Nvidia,
            vendor_id: 0x10DE,
            device_id: 0x2684,
            device_type: DeviceType::DiscreteGpu,
            backend: wgpu::Backend::Vulkan,
            driver: "nvidia".to_string(),
            driver_info: "535.86.05".to_string(),
        };

        let desc = props.description();
        assert!(desc.contains("GeForce RTX 4090"));
        assert!(desc.contains("Discrete GPU"));
        assert!(desc.contains("Vulkan"));
        assert!(desc.contains("driver"));
        assert!(desc.contains("nvidia"));
        assert!(desc.contains("535.86.05"));
    }

    #[test]
    fn test_adapter_properties_description_driver_no_info() {
        let props = AdapterProperties {
            name: "Intel HD Graphics".to_string(),
            vendor: Vendor::Intel,
            vendor_id: 0x8086,
            device_id: 0x1234,
            device_type: DeviceType::IntegratedGpu,
            backend: wgpu::Backend::Vulkan,
            driver: "intel".to_string(),
            driver_info: String::new(),
        };

        let desc = props.description();
        assert!(desc.contains("Intel HD Graphics"));
        assert!(desc.contains("driver: intel"));
        // Should not have trailing space or double driver info
    }

    #[test]
    fn test_adapter_properties_has_driver_info() {
        let with_driver = AdapterProperties {
            name: "Test".to_string(),
            vendor: Vendor::Nvidia,
            vendor_id: 0x10DE,
            device_id: 0,
            device_type: DeviceType::DiscreteGpu,
            backend: wgpu::Backend::Vulkan,
            driver: "nvidia".to_string(),
            driver_info: String::new(),
        };
        assert!(with_driver.has_driver_info());

        let without_driver = AdapterProperties {
            name: "Test".to_string(),
            vendor: Vendor::Nvidia,
            vendor_id: 0x10DE,
            device_id: 0,
            device_type: DeviceType::DiscreteGpu,
            backend: wgpu::Backend::Vulkan,
            driver: String::new(),
            driver_info: String::new(),
        };
        assert!(!without_driver.has_driver_info());
    }

    #[test]
    fn test_adapter_properties_display() {
        let props = AdapterProperties {
            name: "Radeon RX 7900 XTX".to_string(),
            vendor: Vendor::Amd,
            vendor_id: 0x1002,
            device_id: 0x744C,
            device_type: DeviceType::DiscreteGpu,
            backend: wgpu::Backend::Vulkan,
            driver: String::new(),
            driver_info: String::new(),
        };

        let display = format!("{}", props);
        assert_eq!(display, props.description());
    }

    // ========================================================================
    // AdapterLimits Tests
    // ========================================================================

    #[test]
    fn test_adapter_limits_from_raw() {
        let raw_limits = wgpu::Limits::default();
        let limits = AdapterLimits { raw: raw_limits.clone() };

        // Verify texture limits
        assert_eq!(limits.max_texture_dimension_1d(), raw_limits.max_texture_dimension_1d);
        assert_eq!(limits.max_texture_dimension_2d(), raw_limits.max_texture_dimension_2d);
        assert_eq!(limits.max_texture_dimension_3d(), raw_limits.max_texture_dimension_3d);
        assert_eq!(limits.max_texture_array_layers(), raw_limits.max_texture_array_layers);

        // Verify buffer limits
        assert_eq!(limits.max_buffer_size(), raw_limits.max_buffer_size);
        assert_eq!(limits.max_uniform_buffer_binding_size(), raw_limits.max_uniform_buffer_binding_size);
        assert_eq!(limits.max_storage_buffer_binding_size(), raw_limits.max_storage_buffer_binding_size);

        // Verify bind group limits
        assert_eq!(limits.max_bind_groups(), raw_limits.max_bind_groups);
        assert_eq!(limits.max_bindings_per_bind_group(), raw_limits.max_bindings_per_bind_group);

        // Verify compute limits
        assert_eq!(limits.max_compute_workgroup_size_x(), raw_limits.max_compute_workgroup_size_x);
        assert_eq!(limits.max_compute_workgroup_size_y(), raw_limits.max_compute_workgroup_size_y);
        assert_eq!(limits.max_compute_workgroup_size_z(), raw_limits.max_compute_workgroup_size_z);

        // Verify vertex limits
        assert_eq!(limits.max_vertex_buffers(), raw_limits.max_vertex_buffers);
        assert_eq!(limits.max_vertex_attributes(), raw_limits.max_vertex_attributes);
    }

    #[test]
    fn test_adapter_limits_summary() {
        let limits = AdapterLimits {
            raw: wgpu::Limits::default(),
        };
        let summary = limits.summary();

        // Check texture summary
        assert_eq!(summary.texture.max_1d, limits.max_texture_dimension_1d());
        assert_eq!(summary.texture.max_2d, limits.max_texture_dimension_2d());
        assert_eq!(summary.texture.max_3d, limits.max_texture_dimension_3d());
        assert_eq!(summary.texture.max_array_layers, limits.max_texture_array_layers());

        // Check buffer summary
        assert_eq!(summary.buffer.max_size, limits.max_buffer_size());
        assert_eq!(summary.buffer.max_uniform_binding, limits.max_uniform_buffer_binding_size());
        assert_eq!(summary.buffer.max_storage_binding, limits.max_storage_buffer_binding_size());

        // Check bind group summary
        assert_eq!(summary.bind_group.max_bind_groups, limits.max_bind_groups());
        assert_eq!(summary.bind_group.max_bindings_per_group, limits.max_bindings_per_bind_group());

        // Check compute summary
        assert_eq!(summary.compute.max_workgroup_size_x, limits.max_compute_workgroup_size_x());
        assert_eq!(summary.compute.max_workgroup_size_y, limits.max_compute_workgroup_size_y());
        assert_eq!(summary.compute.max_workgroup_size_z, limits.max_compute_workgroup_size_z());

        // Check vertex summary
        assert_eq!(summary.vertex.max_buffers, limits.max_vertex_buffers());
        assert_eq!(summary.vertex.max_attributes, limits.max_vertex_attributes());
    }

    #[test]
    fn test_adapter_limits_meets_webgpu_minimum() {
        // Default limits should meet WebGPU minimum
        let limits = AdapterLimits {
            raw: wgpu::Limits::default(),
        };
        assert!(limits.meets_webgpu_minimum());
    }

    #[test]
    fn test_adapter_limits_display() {
        let limits = AdapterLimits {
            raw: wgpu::Limits::default(),
        };
        let display = format!("{}", limits);

        // Check that all sections are present
        assert!(display.contains("Adapter Limits:"));
        assert!(display.contains("=== Texture Limits ==="));
        assert!(display.contains("=== Buffer Limits ==="));
        assert!(display.contains("=== Bind Group Limits ==="));
        assert!(display.contains("=== Compute Limits ==="));
        assert!(display.contains("=== Vertex Limits ==="));
        assert!(display.contains("=== Other Limits ==="));

        // Check that key fields are present
        assert!(display.contains("Max 1D Dimension"));
        assert!(display.contains("Max 2D Dimension"));
        assert!(display.contains("Max Buffer Size"));
        assert!(display.contains("Max Bind Groups"));
        assert!(display.contains("Max Workgroup Size"));
        assert!(display.contains("Max Vertex Buffers"));
    }

    #[test]
    fn test_format_limits_internal() {
        let limits = AdapterLimits {
            raw: wgpu::Limits::default(),
        };
        let formatted = format_limits_internal(&limits);

        // Verify format includes human-readable sizes
        assert!(formatted.contains("bytes"));
        assert!(formatted.contains("GB"));
        assert!(formatted.contains("KB"));
        assert!(formatted.contains("MB"));
        assert!(formatted.contains("per stage"));
    }

    #[test]
    fn test_limits_summary_structs_are_copy() {
        // Verify that summary structs implement Copy
        let limits = AdapterLimits {
            raw: wgpu::Limits::default(),
        };
        let summary = limits.summary();

        let texture_copy = summary.texture;
        let buffer_copy = summary.buffer;
        let compute_copy = summary.compute;
        let vertex_copy = summary.vertex;

        // These would fail to compile if Copy wasn't implemented
        assert_eq!(texture_copy.max_2d, summary.texture.max_2d);
        assert_eq!(buffer_copy.max_size, summary.buffer.max_size);
        assert_eq!(compute_copy.max_workgroup_size_x, summary.compute.max_workgroup_size_x);
        assert_eq!(vertex_copy.max_buffers, summary.vertex.max_buffers);
    }

    #[test]
    fn test_adapter_limits_clone() {
        let limits = AdapterLimits {
            raw: wgpu::Limits::default(),
        };
        let cloned = limits.clone();

        assert_eq!(limits.max_texture_dimension_2d(), cloned.max_texture_dimension_2d());
        assert_eq!(limits.max_buffer_size(), cloned.max_buffer_size());
        assert_eq!(limits.max_bind_groups(), cloned.max_bind_groups());
    }

    // Integration tests that require actual GPU hardware
    #[cfg(not(feature = "ci"))]
    mod integration {
        use super::*;

        #[test]
        fn test_enumerate_adapters_with_info_primary() {
            let instance = wgpu::Instance::default();
            let result = enumerate_adapters_with_info(&instance, wgpu::Backends::PRIMARY);

            // We can't guarantee adapters exist, but we can verify the structure
            assert!(result.backend_counts.total() == result.adapters.len());

            // If we have adapters, verify best_adapter works
            if !result.is_empty() {
                assert!(result.best_adapter().is_some());
            }
        }

        #[test]
        fn test_filter_functions() {
            let instance = wgpu::Instance::default();
            let result = enumerate_adapters_with_info(&instance, wgpu::Backends::PRIMARY);
            let original_len = result.adapters.len();

            if !result.is_empty() {
                // Note: wgpu::Adapter doesn't implement Clone, so we test by
                // re-enumerating adapters for each filter test.

                // Test filter_by_device_type
                let result2 = enumerate_adapters_with_info(&instance, wgpu::Backends::PRIMARY);
                let discrete = filter_by_device_type(
                    result2.adapters,
                    DeviceType::DiscreteGpu,
                );
                assert!(discrete.len() <= original_len);

                // Test filter_by_backend
                #[cfg(target_os = "linux")]
                {
                    let result3 = enumerate_adapters_with_info(&instance, wgpu::Backends::PRIMARY);
                    let vulkan =
                        filter_by_backend(result3.adapters, wgpu::Backend::Vulkan);
                    assert!(vulkan.len() <= original_len);
                }
            }
        }
    }

    // ========================================================================
    // FeatureTier Tests
    // ========================================================================

    #[test]
    fn test_feature_tier_ordering() {
        // Tiers should be ordered from minimal to full
        assert!(FeatureTier::Minimal < FeatureTier::Standard);
        assert!(FeatureTier::Standard < FeatureTier::Advanced);
        assert!(FeatureTier::Advanced < FeatureTier::Full);
    }

    #[test]
    fn test_feature_tier_equality() {
        assert_eq!(FeatureTier::Minimal, FeatureTier::Minimal);
        assert_ne!(FeatureTier::Minimal, FeatureTier::Full);
    }

    #[test]
    fn test_feature_tier_description() {
        assert!(FeatureTier::Minimal.description().contains("Minimal"));
        assert!(FeatureTier::Standard.description().contains("Standard"));
        assert!(FeatureTier::Advanced.description().contains("Advanced"));
        assert!(FeatureTier::Full.description().contains("Full"));
    }

    #[test]
    fn test_feature_tier_display() {
        assert_eq!(format!("{}", FeatureTier::Minimal), "Minimal");
        assert_eq!(format!("{}", FeatureTier::Standard), "Standard");
        assert_eq!(format!("{}", FeatureTier::Advanced), "Advanced");
        assert_eq!(format!("{}", FeatureTier::Full), "Full");
    }

    #[test]
    fn test_feature_tier_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(FeatureTier::Minimal);
        set.insert(FeatureTier::Standard);
        set.insert(FeatureTier::Advanced);
        set.insert(FeatureTier::Full);
        assert_eq!(set.len(), 4);
    }

    // ========================================================================
    // AdapterFeatures Tests
    // ========================================================================

    #[test]
    fn test_adapter_features_empty() {
        let features = AdapterFeatures {
            raw: wgpu::Features::empty(),
        };

        assert_eq!(features.count(), 0);
        assert_eq!(features.tier(), FeatureTier::Minimal);
        assert!(!features.has_depth_clip_control());
        assert!(!features.has_texture_compression_bc());
        assert!(!features.has_timestamp_query());
        assert!(!features.has_shader_f16());
    }

    #[test]
    fn test_adapter_features_count() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::PUSH_CONSTANTS,
        };
        assert_eq!(features.count(), 2);

        let features = AdapterFeatures {
            raw: wgpu::Features::TIMESTAMP_QUERY
                | wgpu::Features::PUSH_CONSTANTS
                | wgpu::Features::SHADER_F16,
        };
        assert_eq!(features.count(), 3);
    }

    #[test]
    fn test_adapter_features_individual_checks() {
        let features = AdapterFeatures {
            raw: wgpu::Features::DEPTH_CLIP_CONTROL
                | wgpu::Features::DEPTH32FLOAT_STENCIL8
                | wgpu::Features::TEXTURE_COMPRESSION_BC
                | wgpu::Features::TEXTURE_COMPRESSION_ETC2
                | wgpu::Features::TEXTURE_COMPRESSION_ASTC
                | wgpu::Features::INDIRECT_FIRST_INSTANCE
                | wgpu::Features::TIMESTAMP_QUERY
                | wgpu::Features::PIPELINE_STATISTICS_QUERY
                | wgpu::Features::SHADER_F16
                | wgpu::Features::RG11B10UFLOAT_RENDERABLE
                | wgpu::Features::BGRA8UNORM_STORAGE
                | wgpu::Features::FLOAT32_FILTERABLE,
        };

        assert!(features.has_depth_clip_control());
        assert!(features.has_depth32float_stencil8());
        assert!(features.has_texture_compression_bc());
        assert!(features.has_texture_compression_etc2());
        assert!(features.has_texture_compression_astc());
        assert!(features.has_indirect_first_instance());
        assert!(features.has_timestamp_query());
        assert!(features.has_pipeline_statistics_query());
        assert!(features.has_shader_f16());
        assert!(features.has_rg11b10ufloat_renderable());
        assert!(features.has_bgra8unorm_storage());
        assert!(features.has_float32_filterable());
    }

    #[test]
    fn test_adapter_features_tier_minimal() {
        // Minimal: <4 features
        let features = AdapterFeatures {
            raw: wgpu::Features::DEPTH_CLIP_CONTROL | wgpu::Features::DEPTH32FLOAT_STENCIL8,
        };
        assert_eq!(features.tier(), FeatureTier::Minimal);
    }

    #[test]
    fn test_adapter_features_tier_standard() {
        // Standard: 4+ features without advanced
        let features = AdapterFeatures {
            raw: wgpu::Features::DEPTH_CLIP_CONTROL
                | wgpu::Features::DEPTH32FLOAT_STENCIL8
                | wgpu::Features::TEXTURE_COMPRESSION_BC
                | wgpu::Features::INDIRECT_FIRST_INSTANCE,
        };
        assert_eq!(features.tier(), FeatureTier::Standard);
    }

    #[test]
    fn test_adapter_features_tier_advanced() {
        // Advanced: 8+ features with timestamp/pipeline/f16
        let features = AdapterFeatures {
            raw: wgpu::Features::DEPTH_CLIP_CONTROL
                | wgpu::Features::DEPTH32FLOAT_STENCIL8
                | wgpu::Features::TEXTURE_COMPRESSION_BC
                | wgpu::Features::TEXTURE_COMPRESSION_ASTC
                | wgpu::Features::INDIRECT_FIRST_INSTANCE
                | wgpu::Features::RG11B10UFLOAT_RENDERABLE
                | wgpu::Features::BGRA8UNORM_STORAGE
                | wgpu::Features::TIMESTAMP_QUERY, // Advanced feature
        };
        assert_eq!(features.tier(), FeatureTier::Advanced);
    }

    #[test]
    fn test_adapter_features_tier_full() {
        // Full: 12+ features with advanced
        let features = AdapterFeatures {
            raw: wgpu::Features::DEPTH_CLIP_CONTROL
                | wgpu::Features::DEPTH32FLOAT_STENCIL8
                | wgpu::Features::TEXTURE_COMPRESSION_BC
                | wgpu::Features::TEXTURE_COMPRESSION_ETC2
                | wgpu::Features::TEXTURE_COMPRESSION_ASTC
                | wgpu::Features::INDIRECT_FIRST_INSTANCE
                | wgpu::Features::RG11B10UFLOAT_RENDERABLE
                | wgpu::Features::BGRA8UNORM_STORAGE
                | wgpu::Features::FLOAT32_FILTERABLE
                | wgpu::Features::MULTIVIEW
                | wgpu::Features::PUSH_CONSTANTS
                | wgpu::Features::TIMESTAMP_QUERY, // 12 features + advanced
        };
        assert_eq!(features.tier(), FeatureTier::Full);
    }

    #[test]
    fn test_adapter_features_supports() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::PUSH_CONSTANTS,
        };

        // Single feature
        assert!(features.supports(wgpu::Features::TIMESTAMP_QUERY));
        assert!(features.supports(wgpu::Features::PUSH_CONSTANTS));
        assert!(!features.supports(wgpu::Features::SHADER_F16));

        // Multiple features
        assert!(features.supports(wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::PUSH_CONSTANTS));
        assert!(!features.supports(wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::SHADER_F16));
    }

    #[test]
    fn test_adapter_features_any_compression() {
        let no_compression = AdapterFeatures {
            raw: wgpu::Features::TIMESTAMP_QUERY,
        };
        assert!(!no_compression.has_any_texture_compression());

        let bc_only = AdapterFeatures {
            raw: wgpu::Features::TEXTURE_COMPRESSION_BC,
        };
        assert!(bc_only.has_any_texture_compression());

        let etc2_only = AdapterFeatures {
            raw: wgpu::Features::TEXTURE_COMPRESSION_ETC2,
        };
        assert!(etc2_only.has_any_texture_compression());

        let astc_only = AdapterFeatures {
            raw: wgpu::Features::TEXTURE_COMPRESSION_ASTC,
        };
        assert!(astc_only.has_any_texture_compression());
    }

    #[test]
    fn test_adapter_features_best_compression_format() {
        let no_compression = AdapterFeatures {
            raw: wgpu::Features::TIMESTAMP_QUERY,
        };
        assert_eq!(no_compression.best_compression_format(), "none");

        // BC is preferred over others
        let all_compression = AdapterFeatures {
            raw: wgpu::Features::TEXTURE_COMPRESSION_BC
                | wgpu::Features::TEXTURE_COMPRESSION_ETC2
                | wgpu::Features::TEXTURE_COMPRESSION_ASTC,
        };
        assert_eq!(all_compression.best_compression_format(), "BC");

        // ASTC when BC not available
        let astc_etc2 = AdapterFeatures {
            raw: wgpu::Features::TEXTURE_COMPRESSION_ETC2
                | wgpu::Features::TEXTURE_COMPRESSION_ASTC,
        };
        assert_eq!(astc_etc2.best_compression_format(), "ASTC");

        // ETC2 as fallback
        let etc2_only = AdapterFeatures {
            raw: wgpu::Features::TEXTURE_COMPRESSION_ETC2,
        };
        assert_eq!(etc2_only.best_compression_format(), "ETC2");
    }

    #[test]
    fn test_adapter_features_display() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TEXTURE_COMPRESSION_BC,
        };
        let display = format!("{}", features);

        assert!(display.contains("Adapter Features"));
        assert!(display.contains("Tier:"));
        assert!(display.contains("Texture Compression"));
        assert!(display.contains("Rendering"));
        assert!(display.contains("Queries"));
        assert!(display.contains("Shader"));
        assert!(display.contains("Formats"));
    }

    #[test]
    fn test_adapter_features_clone() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::SHADER_F16,
        };
        let cloned = features.clone();

        assert_eq!(features.count(), cloned.count());
        assert_eq!(features.has_timestamp_query(), cloned.has_timestamp_query());
        assert_eq!(features.has_shader_f16(), cloned.has_shader_f16());
    }

    // ========================================================================
    // FeaturesSummary Tests
    // ========================================================================

    #[test]
    fn test_features_summary_creation() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TEXTURE_COMPRESSION_BC
                | wgpu::Features::TIMESTAMP_QUERY
                | wgpu::Features::SHADER_F16
                | wgpu::Features::PUSH_CONSTANTS,
        };
        let summary = features.summary();

        assert_eq!(summary.total_count, 4);
        assert!(summary.has_compression_bc);
        assert!(!summary.has_compression_etc2);
        assert!(!summary.has_compression_astc);
        assert!(summary.has_timestamp_query);
        assert!(summary.has_shader_f16);
        assert!(summary.has_push_constants);
    }

    #[test]
    fn test_features_summary_helper_methods() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TEXTURE_COMPRESSION_BC
                | wgpu::Features::TIMESTAMP_QUERY
                | wgpu::Features::INDIRECT_FIRST_INSTANCE,
        };
        let summary = features.summary();

        assert!(summary.has_any_compression());
        assert!(summary.has_profiling());
        assert!(summary.has_gpu_driven());
    }

    #[test]
    fn test_features_summary_display() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TEXTURE_COMPRESSION_BC
                | wgpu::Features::TIMESTAMP_QUERY
                | wgpu::Features::SHADER_F16
                | wgpu::Features::PUSH_CONSTANTS,
        };
        let summary = features.summary();
        let display = format!("{}", summary);

        assert!(display.contains("Compression"));
        assert!(display.contains("Profiling"));
        assert!(display.contains("F16"));
    }

    #[test]
    fn test_features_summary_copy() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TIMESTAMP_QUERY,
        };
        let summary = features.summary();

        // Copy semantics
        let copy = summary;
        assert_eq!(copy.total_count, summary.total_count);
        assert_eq!(copy.tier, summary.tier);
    }

    // ========================================================================
    // inspect_features Tests
    // ========================================================================

    #[test]
    fn test_inspect_features_format() {
        // We can't create an adapter without hardware, but we can test the
        // underlying Display implementation which inspect_features uses
        let features = AdapterFeatures {
            raw: wgpu::Features::TIMESTAMP_QUERY
                | wgpu::Features::TEXTURE_COMPRESSION_BC
                | wgpu::Features::SHADER_F16,
        };
        let output = format!("{}", features);

        // Verify structure
        assert!(output.contains("Adapter Features"));
        assert!(output.contains("available"));
        assert!(output.contains("Tier:"));
        assert!(output.contains("==="));
    }

    // ========================================================================
    // DeviceTypeWeights Tests
    // ========================================================================

    #[test]
    fn test_device_type_weights_default() {
        let weights = DeviceTypeWeights::default();
        assert_eq!(weights.discrete, 1000);
        assert_eq!(weights.integrated, 500);
        assert_eq!(weights.virtual_gpu, 300);
        assert_eq!(weights.cpu, 100);
        assert_eq!(weights.other, 50);
    }

    #[test]
    fn test_device_type_weights_weight_for() {
        let weights = DeviceTypeWeights::default();
        assert_eq!(weights.weight_for(DeviceType::DiscreteGpu), 1000);
        assert_eq!(weights.weight_for(DeviceType::IntegratedGpu), 500);
        assert_eq!(weights.weight_for(DeviceType::VirtualGpu), 300);
        assert_eq!(weights.weight_for(DeviceType::Cpu), 100);
        assert_eq!(weights.weight_for(DeviceType::Other), 50);
    }

    #[test]
    fn test_device_type_weights_power_saving() {
        let weights = DeviceTypeWeights::power_saving();
        // In power saving mode, integrated should be preferred over discrete
        assert!(weights.integrated > weights.discrete);
    }

    #[test]
    fn test_device_type_weights_performance() {
        let weights = DeviceTypeWeights::performance();
        // In performance mode, discrete should be strongly preferred
        assert!(weights.discrete > weights.integrated * 2);
    }

    // ========================================================================
    // AdapterBlacklistEntry Tests
    // ========================================================================

    #[test]
    fn test_blacklist_entry_new() {
        let entry = AdapterBlacklistEntry::new();
        assert!(entry.vendor.is_none());
        assert!(entry.name_contains.is_none());
        assert!(entry.reason.is_empty());
    }

    #[test]
    fn test_blacklist_entry_with_vendor() {
        let entry = AdapterBlacklistEntry::new()
            .with_vendor(Vendor::Microsoft)
            .with_reason("Software renderer");
        assert_eq!(entry.vendor, Some(Vendor::Microsoft));
        assert_eq!(entry.reason, "Software renderer");
    }

    #[test]
    fn test_blacklist_entry_with_name() {
        let entry = AdapterBlacklistEntry::new()
            .with_name_contains("WARP")
            .with_reason("Software renderer");
        assert_eq!(entry.name_contains, Some("WARP".to_string()));
    }

    #[test]
    fn test_blacklist_entry_matches_vendor() {
        let entry = AdapterBlacklistEntry::new()
            .with_vendor(Vendor::Microsoft)
            .with_reason("test");

        let microsoft_info = AdapterInfo {
            name: "Microsoft Basic Render Driver".to_string(),
            vendor: 0x1414, // Microsoft
            device: 0,
            device_type: DeviceType::Cpu,
            backend: wgpu::Backend::Vulkan,
            driver: String::new(),
            driver_info: String::new(),
        };

        let nvidia_info = AdapterInfo {
            name: "NVIDIA GeForce RTX 4090".to_string(),
            vendor: 0x10DE, // NVIDIA
            device: 0,
            device_type: DeviceType::DiscreteGpu,
            backend: wgpu::Backend::Vulkan,
            driver: String::new(),
            driver_info: String::new(),
        };

        assert!(entry.matches(&microsoft_info));
        assert!(!entry.matches(&nvidia_info));
    }

    #[test]
    fn test_blacklist_entry_matches_name() {
        let entry = AdapterBlacklistEntry::new()
            .with_name_contains("WARP")
            .with_reason("test");

        let warp_info = AdapterInfo {
            name: "Microsoft Basic WARP Driver".to_string(),
            vendor: 0x1414,
            device: 0,
            device_type: DeviceType::Cpu,
            backend: wgpu::Backend::Vulkan,
            driver: String::new(),
            driver_info: String::new(),
        };

        let nvidia_info = AdapterInfo {
            name: "NVIDIA GeForce RTX 4090".to_string(),
            vendor: 0x10DE,
            device: 0,
            device_type: DeviceType::DiscreteGpu,
            backend: wgpu::Backend::Vulkan,
            driver: String::new(),
            driver_info: String::new(),
        };

        assert!(entry.matches(&warp_info));
        assert!(!entry.matches(&nvidia_info));
    }

    #[test]
    fn test_blacklist_entry_matches_case_insensitive() {
        let entry = AdapterBlacklistEntry::new()
            .with_name_contains("warp")
            .with_reason("test");

        let warp_info = AdapterInfo {
            name: "Microsoft Basic WARP Driver".to_string(),
            vendor: 0x1414,
            device: 0,
            device_type: DeviceType::Cpu,
            backend: wgpu::Backend::Vulkan,
            driver: String::new(),
            driver_info: String::new(),
        };

        assert!(entry.matches(&warp_info));
    }

    #[test]
    fn test_blacklist_entry_matches_both() {
        // Entry requires both vendor AND name to match
        let entry = AdapterBlacklistEntry::new()
            .with_vendor(Vendor::Microsoft)
            .with_name_contains("WARP")
            .with_reason("test");

        let warp_microsoft = AdapterInfo {
            name: "Microsoft Basic WARP Driver".to_string(),
            vendor: 0x1414, // Microsoft
            device: 0,
            device_type: DeviceType::Cpu,
            backend: wgpu::Backend::Vulkan,
            driver: String::new(),
            driver_info: String::new(),
        };

        // Different vendor, same name pattern
        let warp_other = AdapterInfo {
            name: "Some WARP Implementation".to_string(),
            vendor: 0x10DE, // NVIDIA (wrong vendor)
            device: 0,
            device_type: DeviceType::Cpu,
            backend: wgpu::Backend::Vulkan,
            driver: String::new(),
            driver_info: String::new(),
        };

        assert!(entry.matches(&warp_microsoft));
        assert!(!entry.matches(&warp_other)); // Wrong vendor
    }

    #[test]
    fn test_blacklist_entry_empty_never_matches() {
        let entry = AdapterBlacklistEntry::new();

        let any_info = AdapterInfo {
            name: "Any GPU".to_string(),
            vendor: 0x10DE,
            device: 0,
            device_type: DeviceType::DiscreteGpu,
            backend: wgpu::Backend::Vulkan,
            driver: String::new(),
            driver_info: String::new(),
        };

        // Empty entry should not match anything (no filters specified)
        assert!(!entry.matches(&any_info));
    }

    // ========================================================================
    // AdapterScore Tests
    // ========================================================================

    #[test]
    fn test_adapter_score_zero() {
        let score = AdapterScore::zero();
        assert_eq!(score.device_type_score, 0);
        assert_eq!(score.feature_score, 0);
        assert_eq!(score.limits_score, 0);
        assert_eq!(score.vendor_bonus, 0);
        assert_eq!(score.total, 0);
        assert!(!score.blacklisted);
        assert!(score.blacklist_reason.is_none());
    }

    #[test]
    fn test_adapter_score_blacklisted() {
        let score = AdapterScore::blacklisted("Driver bug".to_string());
        assert!(score.blacklisted);
        assert_eq!(score.blacklist_reason, Some("Driver bug".to_string()));
        assert_eq!(score.total, 0);
    }

    #[test]
    fn test_adapter_score_calculate_total() {
        let mut score = AdapterScore::zero();
        score.device_type_score = 1000;
        score.feature_score = 400;
        score.limits_score = 100;
        score.vendor_bonus = 200;
        score.calculate_total();

        assert_eq!(score.total, 1700);
    }

    #[test]
    fn test_adapter_score_display() {
        let mut score = AdapterScore::zero();
        score.device_type_score = 1000;
        score.feature_score = 400;
        score.limits_score = 100;
        score.vendor_bonus = 200;
        score.calculate_total();

        let display = format!("{}", score);
        assert!(display.contains("1700"));
        assert!(display.contains("type: 1000"));
        assert!(display.contains("features: 400"));
        assert!(display.contains("limits: 100"));
        assert!(display.contains("vendor: 200"));
    }

    #[test]
    fn test_adapter_score_display_blacklisted() {
        let score = AdapterScore::blacklisted("Known issue".to_string());
        let display = format!("{}", score);
        assert!(display.contains("BLACKLISTED"));
        assert!(display.contains("Known issue"));
    }

    // ========================================================================
    // AdapterSelector Tests
    // ========================================================================

    #[test]
    fn test_adapter_selector_new() {
        let selector = AdapterSelector::new();
        assert!(selector.blacklist.is_empty());
        assert!(selector.vendor_preference.is_none());
        assert_eq!(selector.feature_weight, 100);
        assert_eq!(selector.limit_weight, 1);
        assert_eq!(selector.vendor_preference_bonus, 200);
    }

    #[test]
    fn test_adapter_selector_builder_pattern() {
        let selector = AdapterSelector::new()
            .with_vendor_preference(Vendor::Nvidia)
            .with_feature_weight(150)
            .with_limit_weight(2)
            .with_vendor_preference_bonus(300)
            .with_device_type_weights(DeviceTypeWeights::performance());

        assert_eq!(selector.vendor_preference, Some(Vendor::Nvidia));
        assert_eq!(selector.feature_weight, 150);
        assert_eq!(selector.limit_weight, 2);
        assert_eq!(selector.vendor_preference_bonus, 300);
        assert_eq!(selector.device_type_weights.discrete, 2000);
    }

    #[test]
    fn test_adapter_selector_with_blacklist() {
        let selector = AdapterSelector::new()
            .with_blacklist_entry(
                AdapterBlacklistEntry::new()
                    .with_vendor(Vendor::Microsoft)
                    .with_reason("Software renderer"),
            )
            .with_blacklist_entry(
                AdapterBlacklistEntry::new()
                    .with_name_contains("Buggy GPU")
                    .with_reason("Known driver crash"),
            );

        assert_eq!(selector.blacklist.len(), 2);
    }

    #[test]
    fn test_adapter_selector_with_blacklist_vec() {
        let entries = vec![
            AdapterBlacklistEntry::new()
                .with_vendor(Vendor::Microsoft)
                .with_reason("Software"),
            AdapterBlacklistEntry::new()
                .with_name_contains("Bad GPU")
                .with_reason("Driver issue"),
        ];

        let selector = AdapterSelector::new().with_blacklist(entries);
        assert_eq!(selector.blacklist.len(), 2);
    }

    #[test]
    fn test_adapter_selector_is_blacklisted() {
        let selector = AdapterSelector::new()
            .with_blacklist_entry(
                AdapterBlacklistEntry::new()
                    .with_vendor(Vendor::Microsoft)
                    .with_reason("Software renderer"),
            );

        let microsoft_info = AdapterInfo {
            name: "Microsoft Basic Render Driver".to_string(),
            vendor: 0x1414,
            device: 0,
            device_type: DeviceType::Cpu,
            backend: wgpu::Backend::Vulkan,
            driver: String::new(),
            driver_info: String::new(),
        };

        let nvidia_info = AdapterInfo {
            name: "NVIDIA GeForce RTX 4090".to_string(),
            vendor: 0x10DE,
            device: 0,
            device_type: DeviceType::DiscreteGpu,
            backend: wgpu::Backend::Vulkan,
            driver: String::new(),
            driver_info: String::new(),
        };

        assert!(selector.is_blacklisted(&microsoft_info).is_some());
        assert!(selector.is_blacklisted(&nvidia_info).is_none());
    }

    #[test]
    fn test_adapter_selector_select_empty() {
        let selector = AdapterSelector::new();
        let adapters: Vec<Adapter> = vec![];
        assert!(selector.select(&adapters).is_none());
    }

    #[test]
    fn test_adapter_selector_default() {
        let selector = AdapterSelector::default();
        assert!(selector.blacklist.is_empty());
        assert!(selector.vendor_preference.is_none());
    }

    #[test]
    fn test_adapter_selector_score_limits() {
        let selector = AdapterSelector::new();

        // Test with default limits
        let limits = wgpu::Limits::default();
        let score = selector.score_limits(&limits);

        // Score should be non-zero for default limits
        assert!(score > 0, "Default limits should produce non-zero score");
    }

    #[test]
    fn test_adapter_selector_score_limits_scaling() {
        let selector = AdapterSelector::new();

        // Low limits
        let low_limits = wgpu::Limits::downlevel_webgl2_defaults();
        let low_score = selector.score_limits(&low_limits);

        // High limits
        let high_limits = wgpu::Limits::default();
        let high_score = selector.score_limits(&high_limits);

        // Higher limits should produce higher score
        assert!(
            high_score >= low_score,
            "Higher limits should produce equal or higher score: high={}, low={}",
            high_score,
            low_score
        );
    }

    #[test]
    fn test_adapter_selector_clone() {
        let selector = AdapterSelector::new()
            .with_vendor_preference(Vendor::Nvidia)
            .with_blacklist_entry(
                AdapterBlacklistEntry::new()
                    .with_vendor(Vendor::Microsoft)
                    .with_reason("test"),
            );

        let cloned = selector.clone();
        assert_eq!(cloned.vendor_preference, selector.vendor_preference);
        assert_eq!(cloned.blacklist.len(), selector.blacklist.len());
    }

    // ========================================================================
    // Integration tests (require GPU hardware)
    // ========================================================================

    #[cfg(not(feature = "ci"))]
    mod selector_integration {
        use super::*;

        #[test]
        fn test_adapter_selector_score_real_adapter() {
            let instance = wgpu::Instance::default();
            let adapters: Vec<Adapter> = instance.enumerate_adapters(wgpu::Backends::PRIMARY);

            if adapters.is_empty() {
                println!("No adapters found, skipping test");
                return;
            }

            let selector = AdapterSelector::new();
            let score = selector.score_adapter(&adapters[0]);

            // Real adapters should have positive scores
            assert!(!score.blacklisted);
            assert!(score.total > 0, "Real adapter should have positive score");
            assert!(score.device_type_score > 0);
        }

        #[test]
        fn test_adapter_selector_select_real_adapters() {
            let instance = wgpu::Instance::default();
            let adapters: Vec<Adapter> = instance.enumerate_adapters(wgpu::Backends::PRIMARY);

            if adapters.is_empty() {
                println!("No adapters found, skipping test");
                return;
            }

            let selector = AdapterSelector::new();
            let result = selector.select(&adapters);

            assert!(result.is_some(), "Should select an adapter");

            let result = result.unwrap();
            assert!(!result.adapter_name().is_empty());
            assert_eq!(result.all_scores.len(), adapters.len());
        }

        #[test]
        fn test_adapter_selector_prefers_discrete() {
            let instance = wgpu::Instance::default();
            let adapters: Vec<Adapter> = instance.enumerate_adapters(wgpu::Backends::PRIMARY);

            if adapters.is_empty() {
                println!("No adapters found, skipping test");
                return;
            }

            let selector = AdapterSelector::new();

            // Score all adapters
            let mut has_discrete = false;
            let mut has_integrated = false;
            let mut discrete_score = 0u32;
            let mut integrated_score = 0u32;

            for adapter in &adapters {
                let info = adapter.get_info();
                let score = selector.score_adapter(adapter);

                match info.device_type {
                    DeviceType::DiscreteGpu => {
                        has_discrete = true;
                        discrete_score = discrete_score.max(score.total);
                    }
                    DeviceType::IntegratedGpu => {
                        has_integrated = true;
                        integrated_score = integrated_score.max(score.total);
                    }
                    _ => {}
                }
            }

            // If we have both types, discrete should score higher
            if has_discrete && has_integrated {
                assert!(
                    discrete_score > integrated_score,
                    "Discrete GPU should score higher than integrated: discrete={}, integrated={}",
                    discrete_score,
                    integrated_score
                );
            }
        }

        #[test]
        fn test_adapter_selector_vendor_preference() {
            let instance = wgpu::Instance::default();
            let adapters: Vec<Adapter> = instance.enumerate_adapters(wgpu::Backends::PRIMARY);

            if adapters.is_empty() {
                println!("No adapters found, skipping test");
                return;
            }

            // Find a vendor that exists
            let first_vendor = Vendor::from_id(adapters[0].get_info().vendor);

            let selector_no_pref = AdapterSelector::new();
            let selector_with_pref = AdapterSelector::new()
                .with_vendor_preference(first_vendor);

            let score_no_pref = selector_no_pref.score_adapter(&adapters[0]);
            let score_with_pref = selector_with_pref.score_adapter(&adapters[0]);

            // With preference matching, score should be higher
            assert!(
                score_with_pref.total > score_no_pref.total,
                "Vendor preference should increase score: with={}, without={}",
                score_with_pref.total,
                score_no_pref.total
            );
            assert!(score_with_pref.vendor_bonus > 0);
        }

        #[test]
        fn test_adapter_selector_blacklist_filtering() {
            let instance = wgpu::Instance::default();
            let adapters: Vec<Adapter> = instance.enumerate_adapters(wgpu::Backends::PRIMARY);

            if adapters.is_empty() {
                println!("No adapters found, skipping test");
                return;
            }

            let first_name = adapters[0].get_info().name.clone();

            // Blacklist the first adapter by name
            let selector = AdapterSelector::new()
                .with_blacklist_entry(
                    AdapterBlacklistEntry::new()
                        .with_name_contains(&first_name)
                        .with_reason("Test blacklist"),
                );

            let score = selector.score_adapter(&adapters[0]);
            assert!(score.blacklisted, "First adapter should be blacklisted");
            assert_eq!(score.total, 0, "Blacklisted adapters should have 0 score");
        }

        #[test]
        fn test_adapter_selector_fallback_when_all_blacklisted() {
            let instance = wgpu::Instance::default();
            let adapters: Vec<Adapter> = instance.enumerate_adapters(wgpu::Backends::PRIMARY);

            if adapters.is_empty() {
                println!("No adapters found, skipping test");
                return;
            }

            // Blacklist everything
            let mut selector = AdapterSelector::new();
            for adapter in &adapters {
                selector = selector.with_blacklist_entry(
                    AdapterBlacklistEntry::new()
                        .with_name_contains(&adapter.get_info().name)
                        .with_reason("Test"),
                );
            }

            // Should still return something (fallback to first)
            let result = selector.select(&adapters);
            assert!(result.is_some(), "Should fallback to first adapter even if all blacklisted");

            let result = result.unwrap();
            assert!(result.score.blacklisted);
        }

        #[test]
        fn test_selection_result_log() {
            let instance = wgpu::Instance::default();
            let adapters: Vec<Adapter> = instance.enumerate_adapters(wgpu::Backends::PRIMARY);

            if adapters.is_empty() {
                println!("No adapters found, skipping test");
                return;
            }

            let selector = AdapterSelector::new();
            if let Some(result) = selector.select(&adapters) {
                // This should not panic
                result.log_results();

                // Verify adapter_name works
                assert!(!result.adapter_name().is_empty());
            }
        }
    }
}
