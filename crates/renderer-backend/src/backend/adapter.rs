//! Adapter selection and ranking for TRINITY.
//!
//! This module provides GPU adapter selection with preference-based ranking
//! for choosing the best adapter from multiple available options.
//!
//! # Overview
//!
//! Modern systems often have multiple GPUs available:
//! - Discrete GPUs (NVIDIA, AMD)
//! - Integrated GPUs (Intel, AMD APU)
//! - Virtual GPUs (cloud/VM environments)
//! - Software renderers (CPU fallback)
//!
//! This module provides:
//! - [`AdapterInfo`] - Comprehensive adapter information
//! - [`GpuVendor`] - GPU vendor classification
//! - [`DeviceType`] - Device type classification
//! - [`AdapterPreference`] - User preferences for adapter selection
//! - [`PowerPreference`] - Power vs performance tradeoff
//! - [`AdapterSelector`] - Scoring and selection algorithm
//!
//! # Example
//!
//! ```no_run
//! use renderer_backend::backend::{
//!     AdapterInfo, AdapterPreference, AdapterSelector, GpuVendor, PowerPreference,
//! };
//!
//! # async fn example() {
//! let instance = wgpu::Instance::default();
//!
//! // Enumerate all adapters
//! let adapters = AdapterSelector::enumerate_adapters(&instance);
//! println!("Found {} adapters", adapters.len());
//!
//! // Create preferences
//! let prefs = AdapterPreference {
//!     preferred_vendor: Some(GpuVendor::Nvidia),
//!     power_preference: PowerPreference::HighPerformance,
//!     minimum_vram_mb: Some(4096),
//!     ..Default::default()
//! };
//!
//! // Select the best adapter
//! if let Some(best) = AdapterSelector::select_best(&adapters, &prefs) {
//!     println!("Selected: {} ({:?})", best.name, best.vendor);
//! }
//! # }
//! ```

use super::{BackendCapabilities, BackendType};
use log::{debug, info, warn};
use wgpu::{Adapter, Features, Limits};

// ============================================================================
// GpuVendor
// ============================================================================

/// GPU vendor classification.
///
/// Classifies GPU vendors by their PCI vendor ID. This is useful for
/// vendor-specific optimizations and workarounds.
///
/// # Vendor IDs
///
/// | Vendor | ID |
/// |--------|-----|
/// | NVIDIA | 0x10DE |
/// | AMD | 0x1002, 0x1022 |
/// | Intel | 0x8086 |
/// | Apple | 0x106B |
/// | ARM | 0x13B5 |
/// | Qualcomm | 0x5143 |
/// | Microsoft | 0x1414 |
///
/// # Example
///
/// ```
/// use renderer_backend::backend::GpuVendor;
///
/// let vendor = GpuVendor::from_vendor_id(0x10DE);
/// assert_eq!(vendor, GpuVendor::Nvidia);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GpuVendor {
    /// NVIDIA Corporation (vendor ID 0x10DE)
    Nvidia,
    /// Advanced Micro Devices (vendor IDs 0x1002, 0x1022)
    AMD,
    /// Intel Corporation (vendor ID 0x8086)
    Intel,
    /// Apple Inc. (vendor ID 0x106B)
    Apple,
    /// ARM Limited (vendor ID 0x13B5)
    ARM,
    /// Qualcomm (vendor ID 0x5143)
    Qualcomm,
    /// Microsoft Corporation (vendor ID 0x1414, used for WARP software renderer)
    Microsoft,
    /// Unknown vendor with raw vendor ID
    Unknown(u32),
}

impl GpuVendor {
    /// Classify vendor from a PCI vendor ID.
    ///
    /// # Arguments
    ///
    /// * `vendor_id` - The raw PCI vendor ID from the adapter info
    ///
    /// # Returns
    ///
    /// The classified vendor, or `GpuVendor::Unknown(id)` if not recognized.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::GpuVendor;
    ///
    /// assert_eq!(GpuVendor::from_vendor_id(0x10DE), GpuVendor::Nvidia);
    /// assert_eq!(GpuVendor::from_vendor_id(0x1002), GpuVendor::AMD);
    /// assert_eq!(GpuVendor::from_vendor_id(0x8086), GpuVendor::Intel);
    /// ```
    pub fn from_vendor_id(vendor_id: u32) -> Self {
        match vendor_id {
            0x10DE => GpuVendor::Nvidia,
            0x1002 | 0x1022 => GpuVendor::AMD,
            0x8086 => GpuVendor::Intel,
            0x106B => GpuVendor::Apple,
            0x13B5 => GpuVendor::ARM,
            0x5143 => GpuVendor::Qualcomm,
            0x1414 => GpuVendor::Microsoft,
            other => GpuVendor::Unknown(other),
        }
    }

    /// Get the human-readable vendor name.
    ///
    /// # Returns
    ///
    /// A static string with the vendor's common name.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::GpuVendor;
    ///
    /// assert_eq!(GpuVendor::Nvidia.name(), "NVIDIA");
    /// assert_eq!(GpuVendor::AMD.name(), "AMD");
    /// ```
    pub fn name(&self) -> &'static str {
        match self {
            GpuVendor::Nvidia => "NVIDIA",
            GpuVendor::AMD => "AMD",
            GpuVendor::Intel => "Intel",
            GpuVendor::Apple => "Apple",
            GpuVendor::ARM => "ARM",
            GpuVendor::Qualcomm => "Qualcomm",
            GpuVendor::Microsoft => "Microsoft",
            GpuVendor::Unknown(_) => "Unknown",
        }
    }

    /// Get the raw vendor ID.
    ///
    /// For known vendors, returns their standard PCI vendor ID.
    /// For unknown vendors, returns the ID passed to `from_vendor_id()`.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::GpuVendor;
    ///
    /// assert_eq!(GpuVendor::Nvidia.vendor_id(), 0x10DE);
    /// assert_eq!(GpuVendor::Unknown(0xABCD).vendor_id(), 0xABCD);
    /// ```
    pub fn vendor_id(&self) -> u32 {
        match self {
            GpuVendor::Nvidia => 0x10DE,
            GpuVendor::AMD => 0x1002,
            GpuVendor::Intel => 0x8086,
            GpuVendor::Apple => 0x106B,
            GpuVendor::ARM => 0x13B5,
            GpuVendor::Qualcomm => 0x5143,
            GpuVendor::Microsoft => 0x1414,
            GpuVendor::Unknown(id) => *id,
        }
    }

    /// Check if this is a known vendor (not Unknown).
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::GpuVendor;
    ///
    /// assert!(GpuVendor::Nvidia.is_known());
    /// assert!(!GpuVendor::Unknown(0x1234).is_known());
    /// ```
    #[inline]
    pub fn is_known(&self) -> bool {
        !matches!(self, GpuVendor::Unknown(_))
    }

    /// Check if this vendor typically produces high-performance discrete GPUs.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::GpuVendor;
    ///
    /// assert!(GpuVendor::Nvidia.is_discrete_vendor());
    /// assert!(GpuVendor::AMD.is_discrete_vendor());
    /// assert!(!GpuVendor::Intel.is_discrete_vendor()); // Intel Arc exists but typically integrated
    /// ```
    #[inline]
    pub fn is_discrete_vendor(&self) -> bool {
        matches!(self, GpuVendor::Nvidia | GpuVendor::AMD)
    }
}

impl Default for GpuVendor {
    fn default() -> Self {
        GpuVendor::Unknown(0)
    }
}

impl std::fmt::Display for GpuVendor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuVendor::Unknown(id) => write!(f, "Unknown (0x{:04X})", id),
            _ => write!(f, "{}", self.name()),
        }
    }
}

// ============================================================================
// DeviceType
// ============================================================================

/// GPU device type classification.
///
/// Classifies GPUs by their physical characteristics and capabilities.
/// This affects power consumption, performance, and memory architecture.
///
/// # Device Type Hierarchy
///
/// For adapter selection, device types are ranked:
/// 1. `DiscreteGpu` - Highest performance
/// 2. `IntegratedGpu` - Good balance of performance and power
/// 3. `VirtualGpu` - Virtualized environments
/// 4. `Cpu` - Software rendering fallback
/// 5. `Other` - Unknown type
///
/// # Example
///
/// ```
/// use renderer_backend::backend::DeviceType;
///
/// let discrete = DeviceType::DiscreteGpu;
/// let integrated = DeviceType::IntegratedGpu;
///
/// assert!(discrete.performance_rank() > integrated.performance_rank());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeviceType {
    /// Discrete (dedicated) GPU with its own video memory.
    ///
    /// Discrete GPUs offer the highest performance but consume more power.
    /// Examples: NVIDIA GeForce, AMD Radeon, Intel Arc.
    DiscreteGpu,

    /// Integrated GPU sharing system memory with the CPU.
    ///
    /// Integrated GPUs are more power-efficient but have lower performance.
    /// Examples: Intel UHD Graphics, AMD Radeon Graphics (APU).
    IntegratedGpu,

    /// Virtual GPU in a virtualized environment.
    ///
    /// Virtual GPUs provide GPU capabilities in VMs or cloud instances.
    /// Examples: NVIDIA vGPU, AMD MxGPU.
    VirtualGpu,

    /// CPU-based software renderer.
    ///
    /// Software renderers use the CPU for all graphics operations.
    /// Examples: Microsoft WARP, Mesa llvmpipe.
    Cpu,

    /// Unknown or unclassified device type.
    Other,
}

impl DeviceType {
    /// Convert from wgpu's DeviceType.
    ///
    /// # Arguments
    ///
    /// * `device_type` - The wgpu device type
    ///
    /// # Returns
    ///
    /// The corresponding `DeviceType`.
    pub fn from_wgpu(device_type: wgpu::DeviceType) -> Self {
        match device_type {
            wgpu::DeviceType::DiscreteGpu => DeviceType::DiscreteGpu,
            wgpu::DeviceType::IntegratedGpu => DeviceType::IntegratedGpu,
            wgpu::DeviceType::VirtualGpu => DeviceType::VirtualGpu,
            wgpu::DeviceType::Cpu => DeviceType::Cpu,
            wgpu::DeviceType::Other => DeviceType::Other,
        }
    }

    /// Get the human-readable name.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::DeviceType;
    ///
    /// assert_eq!(DeviceType::DiscreteGpu.name(), "Discrete GPU");
    /// assert_eq!(DeviceType::Cpu.name(), "CPU (Software)");
    /// ```
    pub fn name(&self) -> &'static str {
        match self {
            DeviceType::DiscreteGpu => "Discrete GPU",
            DeviceType::IntegratedGpu => "Integrated GPU",
            DeviceType::VirtualGpu => "Virtual GPU",
            DeviceType::Cpu => "CPU (Software)",
            DeviceType::Other => "Other",
        }
    }

    /// Get the performance rank (higher is better).
    ///
    /// This is used for scoring adapters during selection.
    ///
    /// # Returns
    ///
    /// A numeric rank from 0-100.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::DeviceType;
    ///
    /// assert!(DeviceType::DiscreteGpu.performance_rank() > DeviceType::IntegratedGpu.performance_rank());
    /// assert!(DeviceType::IntegratedGpu.performance_rank() > DeviceType::Cpu.performance_rank());
    /// ```
    pub fn performance_rank(&self) -> u32 {
        match self {
            DeviceType::DiscreteGpu => 100,
            DeviceType::IntegratedGpu => 60,
            DeviceType::VirtualGpu => 40,
            DeviceType::Cpu => 10,
            DeviceType::Other => 5,
        }
    }

    /// Get the power efficiency rank (higher is more efficient).
    ///
    /// This is used when `PowerPreference::LowPower` is selected.
    ///
    /// # Returns
    ///
    /// A numeric rank from 0-100.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::DeviceType;
    ///
    /// assert!(DeviceType::IntegratedGpu.power_efficiency_rank() > DeviceType::DiscreteGpu.power_efficiency_rank());
    /// ```
    pub fn power_efficiency_rank(&self) -> u32 {
        match self {
            DeviceType::IntegratedGpu => 100,
            DeviceType::VirtualGpu => 80,
            DeviceType::Cpu => 60,
            DeviceType::DiscreteGpu => 30,
            DeviceType::Other => 10,
        }
    }

    /// Check if this is a hardware GPU (not software).
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::DeviceType;
    ///
    /// assert!(DeviceType::DiscreteGpu.is_hardware_gpu());
    /// assert!(DeviceType::IntegratedGpu.is_hardware_gpu());
    /// assert!(!DeviceType::Cpu.is_hardware_gpu());
    /// ```
    #[inline]
    pub fn is_hardware_gpu(&self) -> bool {
        matches!(
            self,
            DeviceType::DiscreteGpu | DeviceType::IntegratedGpu | DeviceType::VirtualGpu
        )
    }

    /// Check if this is a discrete (dedicated) GPU.
    #[inline]
    pub fn is_discrete(&self) -> bool {
        matches!(self, DeviceType::DiscreteGpu)
    }

    /// Check if this is an integrated GPU.
    #[inline]
    pub fn is_integrated(&self) -> bool {
        matches!(self, DeviceType::IntegratedGpu)
    }

    /// Check if this is a software renderer.
    #[inline]
    pub fn is_software(&self) -> bool {
        matches!(self, DeviceType::Cpu)
    }
}

impl Default for DeviceType {
    fn default() -> Self {
        DeviceType::Other
    }
}

impl std::fmt::Display for DeviceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ============================================================================
// PowerPreference
// ============================================================================

/// Power preference for adapter selection.
///
/// This controls the balance between performance and power consumption
/// when selecting an adapter.
///
/// # Example
///
/// ```
/// use renderer_backend::backend::PowerPreference;
///
/// // For gaming or professional workloads
/// let gaming = PowerPreference::HighPerformance;
///
/// // For battery-powered devices
/// let mobile = PowerPreference::LowPower;
///
/// // Let the system decide
/// let auto = PowerPreference::None;
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PowerPreference {
    /// Prefer high-performance adapters.
    ///
    /// Selects discrete GPUs over integrated GPUs.
    /// Best for gaming, rendering, and compute-heavy workloads.
    #[default]
    HighPerformance,

    /// Prefer low-power adapters.
    ///
    /// Selects integrated GPUs over discrete GPUs.
    /// Best for laptops on battery or thermal-constrained devices.
    LowPower,

    /// No preference (use scoring only).
    ///
    /// Falls back to pure capability-based scoring without
    /// power preference adjustments.
    None,
}

impl PowerPreference {
    /// Get the human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            PowerPreference::HighPerformance => "High Performance",
            PowerPreference::LowPower => "Low Power",
            PowerPreference::None => "None",
        }
    }

    /// Convert to wgpu's PowerPreference.
    pub fn to_wgpu(&self) -> wgpu::PowerPreference {
        match self {
            PowerPreference::HighPerformance => wgpu::PowerPreference::HighPerformance,
            PowerPreference::LowPower => wgpu::PowerPreference::LowPower,
            PowerPreference::None => wgpu::PowerPreference::None,
        }
    }

    /// Create from wgpu's PowerPreference.
    pub fn from_wgpu(pref: wgpu::PowerPreference) -> Self {
        match pref {
            wgpu::PowerPreference::HighPerformance => PowerPreference::HighPerformance,
            wgpu::PowerPreference::LowPower => PowerPreference::LowPower,
            wgpu::PowerPreference::None => PowerPreference::None,
        }
    }
}

impl std::fmt::Display for PowerPreference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ============================================================================
// WgpuFeatureRequirement
// ============================================================================

/// wgpu feature requirement for adapter selection.
///
/// Specifies a wgpu feature that must be supported by the adapter.
/// This is distinct from `backend::WgpuFeatureRequirement` which uses
/// `FeatureLevel` for more nuanced capability queries.
///
/// # Example
///
/// ```
/// use renderer_backend::backend::WgpuFeatureRequirement;
///
/// // Require ray tracing
/// let rt = WgpuFeatureRequirement::new(
///     wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE,
///     "Ray tracing",
/// );
///
/// // Require timestamp queries
/// let timestamps = WgpuFeatureRequirement::new(
///     wgpu::Features::TIMESTAMP_QUERY,
///     "GPU profiling",
/// );
/// ```
#[derive(Clone, Debug)]
pub struct WgpuFeatureRequirement {
    /// The required feature.
    pub feature: Features,
    /// Human-readable description of why this feature is needed.
    pub description: String,
}

impl WgpuFeatureRequirement {
    /// Create a new feature requirement.
    ///
    /// # Arguments
    ///
    /// * `feature` - The wgpu feature to require
    /// * `description` - Why this feature is needed
    pub fn new(feature: Features, description: impl Into<String>) -> Self {
        Self {
            feature,
            description: description.into(),
        }
    }

    /// Check if an adapter supports this requirement.
    pub fn is_satisfied_by(&self, adapter_features: Features) -> bool {
        adapter_features.contains(self.feature)
    }
}

// ============================================================================
// AdapterInfo
// ============================================================================

/// Comprehensive adapter information.
///
/// This struct combines information from `wgpu::AdapterInfo` with
/// TRINITY's capability detection for a complete view of an adapter.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::backend::AdapterInfo;
///
/// # async fn example() {
/// let instance = wgpu::Instance::default();
/// let adapter = instance
///     .request_adapter(&wgpu::RequestAdapterOptions::default())
///     .await
///     .unwrap();
///
/// let info = AdapterInfo::from_adapter(&adapter);
/// println!("Name: {}", info.name);
/// println!("Vendor: {}", info.vendor);
/// println!("Type: {}", info.device_type);
/// println!("Backend: {}", info.backend);
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct AdapterInfo {
    /// The adapter name (e.g., "NVIDIA GeForce RTX 4090").
    pub name: String,

    /// Classified vendor.
    pub vendor: GpuVendor,

    /// Device type (discrete, integrated, virtual, CPU, other).
    pub device_type: DeviceType,

    /// Graphics backend (Vulkan, Metal, DX12, GL, WebGPU).
    pub backend: BackendType,

    /// Driver version/info (if available, may be empty).
    pub driver_version: String,

    /// Comprehensive capabilities from TRINITY's detection.
    pub capabilities: BackendCapabilities,

    /// Raw wgpu features.
    pub features: Features,

    /// Raw wgpu limits.
    pub limits: Limits,

    /// Raw vendor ID.
    pub vendor_id: u32,

    /// Raw device ID.
    pub device_id: u32,
}

impl AdapterInfo {
    /// Create AdapterInfo from a wgpu Adapter.
    ///
    /// # Arguments
    ///
    /// * `adapter` - The wgpu adapter to extract information from
    ///
    /// # Returns
    ///
    /// An `AdapterInfo` instance with all available information.
    pub fn from_adapter(adapter: &Adapter) -> Self {
        let info = adapter.get_info();
        let features = adapter.features();
        let limits = adapter.limits();
        let capabilities = BackendCapabilities::from_adapter(adapter);

        let driver_version = if info.driver_info.is_empty() {
            info.driver.clone()
        } else if info.driver.is_empty() {
            info.driver_info.clone()
        } else {
            format!("{} {}", info.driver, info.driver_info)
        };

        Self {
            name: info.name.clone(),
            vendor: GpuVendor::from_vendor_id(info.vendor),
            device_type: DeviceType::from_wgpu(info.device_type),
            backend: BackendType::from_adapter(adapter),
            driver_version,
            capabilities,
            features,
            limits,
            vendor_id: info.vendor,
            device_id: info.device,
        }
    }

    /// Get a human-readable description.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use renderer_backend::backend::AdapterInfo;
    /// # fn example(info: AdapterInfo) {
    /// println!("{}", info.description());
    /// // Output: "NVIDIA GeForce RTX 4090 (Discrete GPU, Vulkan)"
    /// # }
    /// ```
    pub fn description(&self) -> String {
        if self.driver_version.is_empty() {
            format!("{} ({}, {})", self.name, self.device_type, self.backend)
        } else {
            format!(
                "{} ({}, {}, driver: {})",
                self.name, self.device_type, self.backend, self.driver_version
            )
        }
    }

    /// Check if ray tracing is supported.
    #[inline]
    pub fn supports_ray_tracing(&self) -> bool {
        self.capabilities.ray_tracing
    }

    /// Check if bindless rendering is supported.
    #[inline]
    pub fn supports_bindless(&self) -> bool {
        self.capabilities.bindless
    }

    /// Check if mesh shaders are supported.
    #[inline]
    pub fn supports_mesh_shaders(&self) -> bool {
        self.capabilities.mesh_shaders
    }

    /// Check if this is a discrete GPU.
    #[inline]
    pub fn is_discrete(&self) -> bool {
        self.device_type.is_discrete()
    }

    /// Check if this is an integrated GPU.
    #[inline]
    pub fn is_integrated(&self) -> bool {
        self.device_type.is_integrated()
    }

    /// Check if this is a software renderer.
    #[inline]
    pub fn is_software(&self) -> bool {
        self.device_type.is_software()
    }

    /// Get estimated VRAM in MB (approximate).
    ///
    /// Note: wgpu doesn't expose actual VRAM directly. This returns
    /// an estimate based on `max_buffer_size` as a proxy.
    pub fn estimated_vram_mb(&self) -> u64 {
        // max_buffer_size is typically ~25% of VRAM on modern GPUs
        // This is a rough estimate
        (self.limits.max_buffer_size / (1024 * 1024)) * 4
    }
}

impl std::fmt::Display for AdapterInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.description())
    }
}

// ============================================================================
// AdapterPreference
// ============================================================================

/// User preferences for adapter selection.
///
/// This struct specifies the criteria for selecting the best adapter.
/// All fields are optional - unspecified fields don't affect scoring.
///
/// # Example
///
/// ```
/// use renderer_backend::backend::{
///     AdapterPreference, BackendType, DeviceType, WgpuFeatureRequirement, GpuVendor, PowerPreference,
/// };
///
/// // Gaming preferences
/// let gaming = AdapterPreference {
///     preferred_vendor: Some(GpuVendor::Nvidia),
///     preferred_type: Some(DeviceType::DiscreteGpu),
///     power_preference: PowerPreference::HighPerformance,
///     minimum_vram_mb: Some(8192),
///     ..Default::default()
/// };
///
/// // Mobile/laptop preferences
/// let mobile = AdapterPreference {
///     power_preference: PowerPreference::LowPower,
///     preferred_type: Some(DeviceType::IntegratedGpu),
///     ..Default::default()
/// };
/// ```
#[derive(Clone, Debug, Default)]
pub struct AdapterPreference {
    /// Preferred GPU vendor (if any).
    pub preferred_vendor: Option<GpuVendor>,

    /// Preferred device type (if any).
    pub preferred_type: Option<DeviceType>,

    /// Preferred backend (if any).
    pub preferred_backend: Option<BackendType>,

    /// Required features (all must be supported).
    pub require_features: Vec<WgpuFeatureRequirement>,

    /// Minimum VRAM in MB (if any).
    pub minimum_vram_mb: Option<u64>,

    /// Power preference for selection.
    pub power_preference: PowerPreference,
}

impl AdapterPreference {
    /// Create a new empty preference.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create preferences optimized for high performance.
    pub fn high_performance() -> Self {
        Self {
            preferred_type: Some(DeviceType::DiscreteGpu),
            power_preference: PowerPreference::HighPerformance,
            ..Default::default()
        }
    }

    /// Create preferences optimized for low power.
    pub fn low_power() -> Self {
        Self {
            preferred_type: Some(DeviceType::IntegratedGpu),
            power_preference: PowerPreference::LowPower,
            ..Default::default()
        }
    }

    /// Set the preferred vendor.
    pub fn with_vendor(mut self, vendor: GpuVendor) -> Self {
        self.preferred_vendor = Some(vendor);
        self
    }

    /// Set the preferred device type.
    pub fn with_device_type(mut self, device_type: DeviceType) -> Self {
        self.preferred_type = Some(device_type);
        self
    }

    /// Set the preferred backend.
    pub fn with_backend(mut self, backend: BackendType) -> Self {
        self.preferred_backend = Some(backend);
        self
    }

    /// Add a required feature.
    pub fn with_required_feature(mut self, feature: WgpuFeatureRequirement) -> Self {
        self.require_features.push(feature);
        self
    }

    /// Set the minimum VRAM.
    pub fn with_minimum_vram_mb(mut self, vram_mb: u64) -> Self {
        self.minimum_vram_mb = Some(vram_mb);
        self
    }

    /// Set the power preference.
    pub fn with_power_preference(mut self, pref: PowerPreference) -> Self {
        self.power_preference = pref;
        self
    }

    /// Require ray tracing support.
    pub fn require_ray_tracing(self) -> Self {
        self.with_required_feature(WgpuFeatureRequirement::new(
            Features::RAY_TRACING_ACCELERATION_STRUCTURE,
            "Ray tracing",
        ))
    }

    /// Require timestamp queries for profiling.
    pub fn require_timestamp_queries(self) -> Self {
        self.with_required_feature(WgpuFeatureRequirement::new(
            Features::TIMESTAMP_QUERY,
            "GPU profiling",
        ))
    }
}

// ============================================================================
// AdapterSelector
// ============================================================================

/// Adapter selection and ranking.
///
/// Provides methods to enumerate, score, and select the best GPU adapter
/// based on user preferences.
///
/// # Scoring Algorithm
///
/// The scoring algorithm considers:
/// 1. **Device type** (100 pts for discrete, 60 for integrated, etc.)
/// 2. **Power preference** (adjusts type scores)
/// 3. **Vendor preference** (+50 pts if matched)
/// 4. **Backend preference** (+30 pts if matched)
/// 5. **Features** (+10 pts per advanced feature)
/// 6. **Limits** (scaled by hardware capability)
///
/// # Example
///
/// ```no_run
/// use renderer_backend::backend::{AdapterInfo, AdapterPreference, AdapterSelector};
///
/// # async fn example() {
/// let instance = wgpu::Instance::default();
/// let adapters = AdapterSelector::enumerate_adapters(&instance);
/// let prefs = AdapterPreference::high_performance();
///
/// // Get scored rankings
/// let ranked = AdapterSelector::rank_adapters(&adapters, &prefs);
/// for (info, score) in ranked {
///     println!("{}: {} points", info.name, score);
/// }
///
/// // Get the best adapter
/// if let Some(best) = AdapterSelector::select_best(&adapters, &prefs) {
///     println!("Selected: {}", best.name);
/// }
/// # }
/// ```
pub struct AdapterSelector;

impl AdapterSelector {
    /// Enumerate all available adapters.
    ///
    /// # Arguments
    ///
    /// * `instance` - The wgpu instance to enumerate from
    ///
    /// # Returns
    ///
    /// A vector of `AdapterInfo` for all available adapters.
    pub fn enumerate_adapters(instance: &wgpu::Instance) -> Vec<AdapterInfo> {
        let wgpu_adapters: Vec<Adapter> = instance.enumerate_adapters(wgpu::Backends::all());

        let adapters: Vec<AdapterInfo> = wgpu_adapters
            .iter()
            .map(AdapterInfo::from_adapter)
            .collect();

        if adapters.is_empty() {
            warn!("AdapterSelector: No adapters found");
        } else {
            info!("AdapterSelector: Found {} adapter(s)", adapters.len());
            for adapter in &adapters {
                debug!(
                    "  - {} ({}, {})",
                    adapter.name, adapter.vendor, adapter.device_type
                );
            }
        }

        adapters
    }

    /// Select the best adapter based on preferences.
    ///
    /// # Arguments
    ///
    /// * `adapters` - Slice of available adapters
    /// * `prefs` - User preferences for selection
    ///
    /// # Returns
    ///
    /// The best adapter, or `None` if no adapters are available
    /// or none meet the requirements.
    pub fn select_best<'a>(
        adapters: &'a [AdapterInfo],
        prefs: &AdapterPreference,
    ) -> Option<&'a AdapterInfo> {
        if adapters.is_empty() {
            return None;
        }

        let ranked = Self::rank_adapters(adapters, prefs);

        // Filter to only adapters that meet requirements
        let valid: Vec<_> = ranked
            .into_iter()
            .filter(|(adapter, _)| Self::meets_requirements(adapter, prefs))
            .collect();

        valid.into_iter().next().map(|(adapter, score)| {
            info!(
                "AdapterSelector: Selected '{}' with score {}",
                adapter.name, score
            );
            adapter
        })
    }

    /// Rank adapters by score.
    ///
    /// # Arguments
    ///
    /// * `adapters` - Slice of available adapters
    /// * `prefs` - User preferences for scoring
    ///
    /// # Returns
    ///
    /// Vector of (adapter, score) tuples sorted by score descending.
    pub fn rank_adapters<'a>(
        adapters: &'a [AdapterInfo],
        prefs: &AdapterPreference,
    ) -> Vec<(&'a AdapterInfo, u32)> {
        let mut scored: Vec<_> = adapters
            .iter()
            .map(|adapter| (adapter, Self::score_adapter(adapter, prefs)))
            .collect();

        // Sort by score descending
        scored.sort_by(|(_, a), (_, b)| b.cmp(a));

        scored
    }

    /// Score an adapter based on preferences.
    ///
    /// # Arguments
    ///
    /// * `adapter` - The adapter to score
    /// * `prefs` - User preferences for scoring
    ///
    /// # Returns
    ///
    /// A numeric score (higher is better).
    pub fn score_adapter(adapter: &AdapterInfo, prefs: &AdapterPreference) -> u32 {
        let mut score = 0u32;

        // Device type score (base)
        let type_score = match prefs.power_preference {
            PowerPreference::HighPerformance => adapter.device_type.performance_rank(),
            PowerPreference::LowPower => adapter.device_type.power_efficiency_rank(),
            PowerPreference::None => adapter.device_type.performance_rank(),
        };
        score += type_score;

        // Preferred type bonus
        if let Some(preferred_type) = prefs.preferred_type {
            if adapter.device_type == preferred_type {
                score += 50;
            }
        }

        // Vendor preference bonus
        if let Some(preferred_vendor) = prefs.preferred_vendor {
            if adapter.vendor == preferred_vendor {
                score += 50;
            }
        }

        // Backend preference bonus
        if let Some(preferred_backend) = prefs.preferred_backend {
            if adapter.backend == preferred_backend {
                score += 30;
            }
        }

        // Feature bonuses
        if adapter.supports_ray_tracing() {
            score += 20;
        }
        if adapter.supports_bindless() {
            score += 15;
        }
        if adapter.supports_mesh_shaders() {
            score += 10;
        }
        if adapter.capabilities.ray_query {
            score += 10;
        }
        if adapter.capabilities.timeline_semaphores {
            score += 5;
        }

        // Limits bonus (scaled)
        let limits_score = Self::score_limits(&adapter.limits);
        score += limits_score;

        score
    }

    /// Score hardware limits.
    fn score_limits(limits: &Limits) -> u32 {
        let mut score = 0u32;

        // Texture dimension: 8192 = 10 pts, 16384 = 20 pts
        score += (limits.max_texture_dimension_2d / 800).min(25);

        // Buffer size: 256MB = 10 pts, 1GB = 20 pts
        score += ((limits.max_buffer_size / (50 * 1024 * 1024)) as u32).min(25);

        // Compute workgroup: 256 = 5 pts, 1024 = 15 pts
        score += (limits.max_compute_invocations_per_workgroup / 70).min(15);

        // Storage buffer: 128MB = 5 pts, 512MB = 15 pts
        score += (limits.max_storage_buffer_binding_size / (40 * 1024 * 1024)).min(15);

        score
    }

    /// Check if an adapter meets all requirements.
    ///
    /// # Arguments
    ///
    /// * `adapter` - The adapter to check
    /// * `prefs` - User preferences with requirements
    ///
    /// # Returns
    ///
    /// `true` if the adapter meets all requirements.
    pub fn meets_requirements(adapter: &AdapterInfo, prefs: &AdapterPreference) -> bool {
        // Check required features
        for req in &prefs.require_features {
            if !req.is_satisfied_by(adapter.features) {
                debug!(
                    "AdapterSelector: '{}' missing required feature: {}",
                    adapter.name, req.description
                );
                return false;
            }
        }

        // Check minimum VRAM
        if let Some(min_vram) = prefs.minimum_vram_mb {
            let estimated_vram = adapter.estimated_vram_mb();
            if estimated_vram < min_vram {
                debug!(
                    "AdapterSelector: '{}' has insufficient VRAM: {} MB (need {} MB)",
                    adapter.name, estimated_vram, min_vram
                );
                return false;
            }
        }

        true
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // GpuVendor Tests
    // ========================================================================

    #[test]
    fn test_gpu_vendor_from_vendor_id_known() {
        assert_eq!(GpuVendor::from_vendor_id(0x10DE), GpuVendor::Nvidia);
        assert_eq!(GpuVendor::from_vendor_id(0x1002), GpuVendor::AMD);
        assert_eq!(GpuVendor::from_vendor_id(0x1022), GpuVendor::AMD);
        assert_eq!(GpuVendor::from_vendor_id(0x8086), GpuVendor::Intel);
        assert_eq!(GpuVendor::from_vendor_id(0x106B), GpuVendor::Apple);
        assert_eq!(GpuVendor::from_vendor_id(0x13B5), GpuVendor::ARM);
        assert_eq!(GpuVendor::from_vendor_id(0x5143), GpuVendor::Qualcomm);
        assert_eq!(GpuVendor::from_vendor_id(0x1414), GpuVendor::Microsoft);
    }

    #[test]
    fn test_gpu_vendor_from_vendor_id_unknown() {
        assert_eq!(GpuVendor::from_vendor_id(0x9999), GpuVendor::Unknown(0x9999));
        assert_eq!(GpuVendor::from_vendor_id(0x0000), GpuVendor::Unknown(0x0000));
    }

    #[test]
    fn test_gpu_vendor_name() {
        assert_eq!(GpuVendor::Nvidia.name(), "NVIDIA");
        assert_eq!(GpuVendor::AMD.name(), "AMD");
        assert_eq!(GpuVendor::Intel.name(), "Intel");
        assert_eq!(GpuVendor::Apple.name(), "Apple");
        assert_eq!(GpuVendor::ARM.name(), "ARM");
        assert_eq!(GpuVendor::Qualcomm.name(), "Qualcomm");
        assert_eq!(GpuVendor::Microsoft.name(), "Microsoft");
        assert_eq!(GpuVendor::Unknown(0x1234).name(), "Unknown");
    }

    #[test]
    fn test_gpu_vendor_vendor_id() {
        assert_eq!(GpuVendor::Nvidia.vendor_id(), 0x10DE);
        assert_eq!(GpuVendor::AMD.vendor_id(), 0x1002);
        assert_eq!(GpuVendor::Intel.vendor_id(), 0x8086);
        assert_eq!(GpuVendor::Apple.vendor_id(), 0x106B);
        assert_eq!(GpuVendor::ARM.vendor_id(), 0x13B5);
        assert_eq!(GpuVendor::Qualcomm.vendor_id(), 0x5143);
        assert_eq!(GpuVendor::Microsoft.vendor_id(), 0x1414);
        assert_eq!(GpuVendor::Unknown(0xABCD).vendor_id(), 0xABCD);
    }

    #[test]
    fn test_gpu_vendor_is_known() {
        assert!(GpuVendor::Nvidia.is_known());
        assert!(GpuVendor::AMD.is_known());
        assert!(GpuVendor::Intel.is_known());
        assert!(!GpuVendor::Unknown(0x1234).is_known());
    }

    #[test]
    fn test_gpu_vendor_is_discrete_vendor() {
        assert!(GpuVendor::Nvidia.is_discrete_vendor());
        assert!(GpuVendor::AMD.is_discrete_vendor());
        assert!(!GpuVendor::Intel.is_discrete_vendor());
        assert!(!GpuVendor::Apple.is_discrete_vendor());
    }

    #[test]
    fn test_gpu_vendor_display() {
        assert_eq!(format!("{}", GpuVendor::Nvidia), "NVIDIA");
        assert_eq!(format!("{}", GpuVendor::AMD), "AMD");
        assert_eq!(format!("{}", GpuVendor::Unknown(0x1234)), "Unknown (0x1234)");
    }

    #[test]
    fn test_gpu_vendor_default() {
        assert_eq!(GpuVendor::default(), GpuVendor::Unknown(0));
    }

    #[test]
    fn test_gpu_vendor_equality() {
        assert_eq!(GpuVendor::Nvidia, GpuVendor::Nvidia);
        assert_ne!(GpuVendor::Nvidia, GpuVendor::AMD);
        assert_eq!(GpuVendor::Unknown(0x1234), GpuVendor::Unknown(0x1234));
        assert_ne!(GpuVendor::Unknown(0x1234), GpuVendor::Unknown(0x5678));
    }

    #[test]
    fn test_gpu_vendor_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(GpuVendor::Nvidia);
        set.insert(GpuVendor::AMD);
        set.insert(GpuVendor::Unknown(0x1234));
        assert_eq!(set.len(), 3);
        assert!(set.contains(&GpuVendor::Nvidia));
    }

    // ========================================================================
    // DeviceType Tests
    // ========================================================================

    #[test]
    fn test_device_type_from_wgpu() {
        assert_eq!(
            DeviceType::from_wgpu(wgpu::DeviceType::DiscreteGpu),
            DeviceType::DiscreteGpu
        );
        assert_eq!(
            DeviceType::from_wgpu(wgpu::DeviceType::IntegratedGpu),
            DeviceType::IntegratedGpu
        );
        assert_eq!(
            DeviceType::from_wgpu(wgpu::DeviceType::VirtualGpu),
            DeviceType::VirtualGpu
        );
        assert_eq!(DeviceType::from_wgpu(wgpu::DeviceType::Cpu), DeviceType::Cpu);
        assert_eq!(
            DeviceType::from_wgpu(wgpu::DeviceType::Other),
            DeviceType::Other
        );
    }

    #[test]
    fn test_device_type_name() {
        assert_eq!(DeviceType::DiscreteGpu.name(), "Discrete GPU");
        assert_eq!(DeviceType::IntegratedGpu.name(), "Integrated GPU");
        assert_eq!(DeviceType::VirtualGpu.name(), "Virtual GPU");
        assert_eq!(DeviceType::Cpu.name(), "CPU (Software)");
        assert_eq!(DeviceType::Other.name(), "Other");
    }

    #[test]
    fn test_device_type_performance_rank() {
        assert!(DeviceType::DiscreteGpu.performance_rank() > DeviceType::IntegratedGpu.performance_rank());
        assert!(DeviceType::IntegratedGpu.performance_rank() > DeviceType::VirtualGpu.performance_rank());
        assert!(DeviceType::VirtualGpu.performance_rank() > DeviceType::Cpu.performance_rank());
        assert!(DeviceType::Cpu.performance_rank() > DeviceType::Other.performance_rank());
    }

    #[test]
    fn test_device_type_power_efficiency_rank() {
        // Integrated should be most efficient
        assert!(DeviceType::IntegratedGpu.power_efficiency_rank() > DeviceType::DiscreteGpu.power_efficiency_rank());
        assert!(DeviceType::VirtualGpu.power_efficiency_rank() > DeviceType::DiscreteGpu.power_efficiency_rank());
    }

    #[test]
    fn test_device_type_is_hardware_gpu() {
        assert!(DeviceType::DiscreteGpu.is_hardware_gpu());
        assert!(DeviceType::IntegratedGpu.is_hardware_gpu());
        assert!(DeviceType::VirtualGpu.is_hardware_gpu());
        assert!(!DeviceType::Cpu.is_hardware_gpu());
        assert!(!DeviceType::Other.is_hardware_gpu());
    }

    #[test]
    fn test_device_type_specific_checks() {
        assert!(DeviceType::DiscreteGpu.is_discrete());
        assert!(!DeviceType::IntegratedGpu.is_discrete());

        assert!(DeviceType::IntegratedGpu.is_integrated());
        assert!(!DeviceType::DiscreteGpu.is_integrated());

        assert!(DeviceType::Cpu.is_software());
        assert!(!DeviceType::DiscreteGpu.is_software());
    }

    #[test]
    fn test_device_type_display() {
        assert_eq!(format!("{}", DeviceType::DiscreteGpu), "Discrete GPU");
        assert_eq!(format!("{}", DeviceType::Cpu), "CPU (Software)");
    }

    #[test]
    fn test_device_type_default() {
        assert_eq!(DeviceType::default(), DeviceType::Other);
    }

    // ========================================================================
    // PowerPreference Tests
    // ========================================================================

    #[test]
    fn test_power_preference_name() {
        assert_eq!(PowerPreference::HighPerformance.name(), "High Performance");
        assert_eq!(PowerPreference::LowPower.name(), "Low Power");
        assert_eq!(PowerPreference::None.name(), "None");
    }

    #[test]
    fn test_power_preference_to_wgpu() {
        assert_eq!(
            PowerPreference::HighPerformance.to_wgpu(),
            wgpu::PowerPreference::HighPerformance
        );
        assert_eq!(
            PowerPreference::LowPower.to_wgpu(),
            wgpu::PowerPreference::LowPower
        );
        assert_eq!(PowerPreference::None.to_wgpu(), wgpu::PowerPreference::None);
    }

    #[test]
    fn test_power_preference_from_wgpu() {
        assert_eq!(
            PowerPreference::from_wgpu(wgpu::PowerPreference::HighPerformance),
            PowerPreference::HighPerformance
        );
        assert_eq!(
            PowerPreference::from_wgpu(wgpu::PowerPreference::LowPower),
            PowerPreference::LowPower
        );
        assert_eq!(
            PowerPreference::from_wgpu(wgpu::PowerPreference::None),
            PowerPreference::None
        );
    }

    #[test]
    fn test_power_preference_default() {
        assert_eq!(PowerPreference::default(), PowerPreference::HighPerformance);
    }

    #[test]
    fn test_power_preference_display() {
        assert_eq!(format!("{}", PowerPreference::HighPerformance), "High Performance");
        assert_eq!(format!("{}", PowerPreference::LowPower), "Low Power");
    }

    // ========================================================================
    // WgpuFeatureRequirement Tests
    // ========================================================================

    #[test]
    fn test_feature_requirement_new() {
        let req = WgpuFeatureRequirement::new(Features::TIMESTAMP_QUERY, "GPU profiling");
        assert_eq!(req.feature, Features::TIMESTAMP_QUERY);
        assert_eq!(req.description, "GPU profiling");
    }

    #[test]
    fn test_feature_requirement_is_satisfied_by() {
        let req = WgpuFeatureRequirement::new(Features::TIMESTAMP_QUERY, "profiling");

        let features_with = Features::TIMESTAMP_QUERY | Features::PUSH_CONSTANTS;
        let features_without = Features::PUSH_CONSTANTS;

        assert!(req.is_satisfied_by(features_with));
        assert!(!req.is_satisfied_by(features_without));
    }

    // ========================================================================
    // AdapterPreference Tests
    // ========================================================================

    #[test]
    fn test_adapter_preference_default() {
        let prefs = AdapterPreference::default();
        assert!(prefs.preferred_vendor.is_none());
        assert!(prefs.preferred_type.is_none());
        assert!(prefs.preferred_backend.is_none());
        assert!(prefs.require_features.is_empty());
        assert!(prefs.minimum_vram_mb.is_none());
        assert_eq!(prefs.power_preference, PowerPreference::HighPerformance);
    }

    #[test]
    fn test_adapter_preference_high_performance() {
        let prefs = AdapterPreference::high_performance();
        assert_eq!(prefs.preferred_type, Some(DeviceType::DiscreteGpu));
        assert_eq!(prefs.power_preference, PowerPreference::HighPerformance);
    }

    #[test]
    fn test_adapter_preference_low_power() {
        let prefs = AdapterPreference::low_power();
        assert_eq!(prefs.preferred_type, Some(DeviceType::IntegratedGpu));
        assert_eq!(prefs.power_preference, PowerPreference::LowPower);
    }

    #[test]
    fn test_adapter_preference_builder_pattern() {
        let prefs = AdapterPreference::new()
            .with_vendor(GpuVendor::Nvidia)
            .with_device_type(DeviceType::DiscreteGpu)
            .with_backend(BackendType::Vulkan)
            .with_minimum_vram_mb(8192)
            .with_power_preference(PowerPreference::HighPerformance);

        assert_eq!(prefs.preferred_vendor, Some(GpuVendor::Nvidia));
        assert_eq!(prefs.preferred_type, Some(DeviceType::DiscreteGpu));
        assert_eq!(prefs.preferred_backend, Some(BackendType::Vulkan));
        assert_eq!(prefs.minimum_vram_mb, Some(8192));
        assert_eq!(prefs.power_preference, PowerPreference::HighPerformance);
    }

    #[test]
    fn test_adapter_preference_require_ray_tracing() {
        let prefs = AdapterPreference::new().require_ray_tracing();
        assert_eq!(prefs.require_features.len(), 1);
        assert_eq!(
            prefs.require_features[0].feature,
            Features::RAY_TRACING_ACCELERATION_STRUCTURE
        );
    }

    #[test]
    fn test_adapter_preference_require_timestamp_queries() {
        let prefs = AdapterPreference::new().require_timestamp_queries();
        assert_eq!(prefs.require_features.len(), 1);
        assert_eq!(prefs.require_features[0].feature, Features::TIMESTAMP_QUERY);
    }

    // ========================================================================
    // AdapterSelector Tests
    // ========================================================================

    #[test]
    fn test_adapter_selector_score_limits() {
        let limits = Limits::default();
        let score = AdapterSelector::score_limits(&limits);
        assert!(score > 0, "Default limits should produce non-zero score");
    }

    #[test]
    fn test_adapter_selector_score_limits_scaling() {
        let low_limits = Limits::downlevel_webgl2_defaults();
        let high_limits = Limits::default();

        let low_score = AdapterSelector::score_limits(&low_limits);
        let high_score = AdapterSelector::score_limits(&high_limits);

        assert!(
            high_score >= low_score,
            "Higher limits should produce equal or higher score: high={}, low={}",
            high_score,
            low_score
        );
    }

    #[test]
    fn test_adapter_selector_select_best_empty() {
        let adapters: Vec<AdapterInfo> = vec![];
        let prefs = AdapterPreference::default();
        assert!(AdapterSelector::select_best(&adapters, &prefs).is_none());
    }

    #[test]
    fn test_adapter_selector_rank_adapters_empty() {
        let adapters: Vec<AdapterInfo> = vec![];
        let prefs = AdapterPreference::default();
        let ranked = AdapterSelector::rank_adapters(&adapters, &prefs);
        assert!(ranked.is_empty());
    }

    // ========================================================================
    // Mock AdapterInfo Tests
    // ========================================================================

    fn make_mock_adapter_info(
        name: &str,
        vendor: GpuVendor,
        device_type: DeviceType,
        backend: BackendType,
    ) -> AdapterInfo {
        AdapterInfo {
            name: name.to_string(),
            vendor,
            device_type,
            backend,
            driver_version: String::new(),
            capabilities: BackendCapabilities {
                backend,
                ray_tracing: device_type == DeviceType::DiscreteGpu,
                ray_query: false,
                mesh_shaders: false,
                bindless: device_type == DeviceType::DiscreteGpu,
                timeline_semaphores: false,
                buffer_device_address: false,
                dynamic_rendering: false,
            },
            features: Features::empty(),
            limits: Limits::default(),
            vendor_id: vendor.vendor_id(),
            device_id: 0,
        }
    }

    #[test]
    fn test_adapter_selector_prefers_discrete() {
        let adapters = vec![
            make_mock_adapter_info(
                "Intel UHD",
                GpuVendor::Intel,
                DeviceType::IntegratedGpu,
                BackendType::Vulkan,
            ),
            make_mock_adapter_info(
                "NVIDIA RTX 4090",
                GpuVendor::Nvidia,
                DeviceType::DiscreteGpu,
                BackendType::Vulkan,
            ),
        ];

        let prefs = AdapterPreference::high_performance();
        let best = AdapterSelector::select_best(&adapters, &prefs).unwrap();

        assert_eq!(best.name, "NVIDIA RTX 4090");
    }

    #[test]
    fn test_adapter_selector_prefers_integrated_low_power() {
        let adapters = vec![
            make_mock_adapter_info(
                "NVIDIA RTX 4090",
                GpuVendor::Nvidia,
                DeviceType::DiscreteGpu,
                BackendType::Vulkan,
            ),
            make_mock_adapter_info(
                "Intel UHD",
                GpuVendor::Intel,
                DeviceType::IntegratedGpu,
                BackendType::Vulkan,
            ),
        ];

        let prefs = AdapterPreference::low_power();
        let best = AdapterSelector::select_best(&adapters, &prefs).unwrap();

        assert_eq!(best.name, "Intel UHD");
    }

    #[test]
    fn test_adapter_selector_vendor_preference() {
        let adapters = vec![
            make_mock_adapter_info(
                "AMD RX 7900",
                GpuVendor::AMD,
                DeviceType::DiscreteGpu,
                BackendType::Vulkan,
            ),
            make_mock_adapter_info(
                "NVIDIA RTX 4090",
                GpuVendor::Nvidia,
                DeviceType::DiscreteGpu,
                BackendType::Vulkan,
            ),
        ];

        let prefs = AdapterPreference::new().with_vendor(GpuVendor::AMD);
        let best = AdapterSelector::select_best(&adapters, &prefs).unwrap();

        // AMD should win due to vendor preference bonus
        assert_eq!(best.name, "AMD RX 7900");
    }

    #[test]
    fn test_adapter_selector_score_adapter() {
        let adapter = make_mock_adapter_info(
            "NVIDIA RTX 4090",
            GpuVendor::Nvidia,
            DeviceType::DiscreteGpu,
            BackendType::Vulkan,
        );

        let prefs = AdapterPreference::default();
        let score = AdapterSelector::score_adapter(&adapter, &prefs);

        // Discrete GPU should have high score
        assert!(score >= 100, "Discrete GPU should have score >= 100");
    }

    #[test]
    fn test_adapter_selector_meets_requirements_no_requirements() {
        let adapter = make_mock_adapter_info(
            "Test GPU",
            GpuVendor::Nvidia,
            DeviceType::DiscreteGpu,
            BackendType::Vulkan,
        );

        let prefs = AdapterPreference::default();
        assert!(AdapterSelector::meets_requirements(&adapter, &prefs));
    }

    #[test]
    fn test_adapter_selector_meets_requirements_missing_feature() {
        let adapter = make_mock_adapter_info(
            "Test GPU",
            GpuVendor::Nvidia,
            DeviceType::DiscreteGpu,
            BackendType::Vulkan,
        );

        let prefs = AdapterPreference::new().require_timestamp_queries();
        // Adapter has empty features, so doesn't have TIMESTAMP_QUERY
        assert!(!AdapterSelector::meets_requirements(&adapter, &prefs));
    }

    #[test]
    fn test_adapter_info_description_no_driver() {
        let adapter = make_mock_adapter_info(
            "Test GPU",
            GpuVendor::Nvidia,
            DeviceType::DiscreteGpu,
            BackendType::Vulkan,
        );

        let desc = adapter.description();
        assert!(desc.contains("Test GPU"));
        assert!(desc.contains("Discrete GPU"));
        assert!(desc.contains("Vulkan"));
        assert!(!desc.contains("driver"));
    }

    #[test]
    fn test_adapter_info_display() {
        let adapter = make_mock_adapter_info(
            "Test GPU",
            GpuVendor::Nvidia,
            DeviceType::DiscreteGpu,
            BackendType::Vulkan,
        );

        let display = format!("{}", adapter);
        assert_eq!(display, adapter.description());
    }

    #[test]
    fn test_adapter_info_supports_methods() {
        let adapter = make_mock_adapter_info(
            "Test GPU",
            GpuVendor::Nvidia,
            DeviceType::DiscreteGpu,
            BackendType::Vulkan,
        );

        // Mock sets ray_tracing and bindless true for discrete GPUs
        assert!(adapter.supports_ray_tracing());
        assert!(adapter.supports_bindless());
        assert!(!adapter.supports_mesh_shaders());
    }

    #[test]
    fn test_adapter_info_type_checks() {
        let discrete = make_mock_adapter_info(
            "Discrete",
            GpuVendor::Nvidia,
            DeviceType::DiscreteGpu,
            BackendType::Vulkan,
        );
        assert!(discrete.is_discrete());
        assert!(!discrete.is_integrated());
        assert!(!discrete.is_software());

        let integrated = make_mock_adapter_info(
            "Integrated",
            GpuVendor::Intel,
            DeviceType::IntegratedGpu,
            BackendType::Vulkan,
        );
        assert!(!integrated.is_discrete());
        assert!(integrated.is_integrated());
        assert!(!integrated.is_software());

        let software = make_mock_adapter_info(
            "Software",
            GpuVendor::Microsoft,
            DeviceType::Cpu,
            BackendType::Vulkan,
        );
        assert!(!software.is_discrete());
        assert!(!software.is_integrated());
        assert!(software.is_software());
    }

    #[test]
    fn test_adapter_info_estimated_vram() {
        let adapter = make_mock_adapter_info(
            "Test GPU",
            GpuVendor::Nvidia,
            DeviceType::DiscreteGpu,
            BackendType::Vulkan,
        );

        // With default limits, estimated VRAM should be > 0
        let vram = adapter.estimated_vram_mb();
        assert!(vram > 0, "Estimated VRAM should be > 0");
    }

    // ========================================================================
    // Integration Tests (require GPU hardware)
    // ========================================================================

    #[cfg(not(feature = "ci"))]
    mod integration {
        use super::*;

        #[test]
        fn test_enumerate_adapters() {
            let instance = wgpu::Instance::default();
            let adapters = AdapterSelector::enumerate_adapters(&instance);

            // We can't guarantee adapters exist, but the function should not panic
            println!("Found {} adapters", adapters.len());

            for adapter in &adapters {
                println!("  - {}", adapter.description());
            }
        }

        #[test]
        fn test_select_best_real_adapters() {
            let instance = wgpu::Instance::default();
            let adapters = AdapterSelector::enumerate_adapters(&instance);

            if adapters.is_empty() {
                println!("No adapters found, skipping test");
                return;
            }

            let prefs = AdapterPreference::high_performance();
            if let Some(best) = AdapterSelector::select_best(&adapters, &prefs) {
                println!("Selected: {}", best.description());
                assert!(!best.name.is_empty());
            }
        }

        #[test]
        fn test_rank_adapters_real() {
            let instance = wgpu::Instance::default();
            let adapters = AdapterSelector::enumerate_adapters(&instance);

            if adapters.is_empty() {
                println!("No adapters found, skipping test");
                return;
            }

            let prefs = AdapterPreference::default();
            let ranked = AdapterSelector::rank_adapters(&adapters, &prefs);

            println!("Adapter rankings:");
            for (adapter, score) in ranked {
                println!("  - {}: {} points", adapter.name, score);
            }
        }
    }
}
