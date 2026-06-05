//! Backend-specific abstractions for TRINITY.
//!
//! This module provides backend detection and platform-specific feature
//! exposure for wgpu's various rendering backends. While wgpu provides a
//! unified abstraction, some advanced features require backend-specific
//! knowledge.
//!
//! # Supported Backends
//!
//! | Backend | Platform | Notes |
//! |---------|----------|-------|
//! | Vulkan | Windows, Linux, Android | Primary backend, full feature support |
//! | Metal | macOS, iOS | Apple platforms |
//! | DX12 | Windows | DirectX 12 |
//! | DX11 | Windows (legacy) | DirectX 11 fallback |
//! | GL | Web, Linux fallback | OpenGL / WebGL |
//! | WebGPU | Web (browsers) | Native WebGPU |
//!
//! # Example
//!
//! ```no_run
//! use renderer_backend::backend::{BackendType, VulkanFeatures};
//!
//! # async fn example() {
//! let instance = wgpu::Instance::default();
//! let adapter = instance
//!     .request_adapter(&wgpu::RequestAdapterOptions::default())
//!     .await
//!     .unwrap();
//!
//! let backend = BackendType::from_adapter(&adapter);
//! println!("Running on: {} (native: {})", backend.name(), backend.is_native());
//!
//! if backend == BackendType::Vulkan {
//!     let vk_features = VulkanFeatures::detect(&adapter);
//!     if vk_features.supports_rt_pipeline() {
//!         println!("Full ray tracing pipeline available!");
//!     }
//! }
//! # }
//! ```
//!
//! # Architecture
//!
//! ```text
//! BackendType (mod.rs)
//!   - Backend detection from wgpu adapter
//!   - Platform classification (native/web)
//!   - Feature capability queries
//!
//! VulkanFeatures (vulkan.rs)
//!   - Vulkan-specific extension detection
//!   - Ray tracing pipeline support
//!   - Bindless/descriptor indexing support
//!   - Raw handle access (feature-gated)
//! ```

pub mod adapter;
pub mod capabilities;
pub mod dx12;
pub mod metal;
#[cfg(feature = "pyo3")]
pub mod python;
pub mod vulkan;
pub mod webgpu;

pub use adapter::{
    AdapterInfo, AdapterPreference, AdapterSelector, DeviceType, GpuVendor, PowerPreference,
    WgpuFeatureRequirement,
};
pub use capabilities::{
    CapabilitiesQuery, FeatureLevel, FeatureRequirement, UnifiedCapabilities,
};
pub use dx12::{
    D3D12FeatureLevel, D3D12Features, D3D12RayTracingTier, D3D12ShaderModel,
    D3DFeatureLevel, DX12Capabilities, DX12Info, MeshShaderTier, RayTracingTier,
    ShaderCompiler, ShaderModel,
};
pub use metal::{
    AppleGpuFamily, AppleSiliconGeneration, MetalCapabilities, MetalFeatures, MetalGpuFamily,
    MetalInfo, MetalVersion, PlatformLimits,
};
pub use vulkan::{VulkanDeviceType, VulkanFeatures, VulkanInfo, VulkanRayTracingTier, VulkanVersion};
pub use webgpu::{
    Browser, BrowserCapabilities, BrowserCompatibility, BrowserType, FeatureSupport,
    WebGPUFeature, WebGPULimitations, WebGPULimits, WebGpuFeatures, WebGpuLimits, WebGpuTier,
};

#[cfg(feature = "vulkan-raw")]
pub use vulkan::RawVulkanHandles;

#[cfg(feature = "pyo3")]
pub use python::{register_python_module, PyAdapterInfo, PyBackendType, PyFeatureLevel, PyUnifiedCapabilities};

use wgpu::{Adapter, Backend};

// ============================================================================
// BackendType
// ============================================================================

/// Graphics API backend type.
///
/// This enum represents the underlying graphics API that wgpu is using.
/// It's determined from the adapter info and can be used to enable
/// backend-specific code paths or features.
///
/// # Backend Detection
///
/// The backend is detected from `wgpu::AdapterInfo::backend` which provides
/// the actual graphics API in use for a given adapter.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::backend::BackendType;
///
/// # async fn example() {
/// let instance = wgpu::Instance::default();
/// let adapter = instance
///     .request_adapter(&wgpu::RequestAdapterOptions::default())
///     .await
///     .unwrap();
///
/// let backend = BackendType::from_adapter(&adapter);
///
/// match backend {
///     BackendType::Vulkan => println!("Using Vulkan"),
///     BackendType::Metal => println!("Using Metal"),
///     BackendType::Dx12 => println!("Using DirectX 12"),
///     _ => println!("Using: {}", backend.name()),
/// }
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendType {
    /// Vulkan graphics API.
    ///
    /// Available on Windows, Linux, and Android. Primary backend for TRINITY
    /// with full ray tracing and advanced feature support.
    Vulkan,

    /// Metal graphics API.
    ///
    /// Apple's graphics API for macOS and iOS. Provides excellent performance
    /// on Apple platforms with Metal 3 ray tracing support on newer hardware.
    Metal,

    /// DirectX 12 graphics API.
    ///
    /// Microsoft's modern graphics API for Windows. Full feature support
    /// including DXR ray tracing on compatible hardware.
    Dx12,

    /// DirectX 11 graphics API.
    ///
    /// Legacy DirectX backend for older Windows systems. Limited feature
    /// support compared to DX12.
    Dx11,

    /// OpenGL / OpenGL ES graphics API.
    ///
    /// Cross-platform fallback backend. Used for WebGL contexts on the web
    /// and as a fallback on systems without Vulkan/DX12 support.
    Gl,

    /// WebGPU graphics API.
    ///
    /// Native WebGPU implementation in browsers. This is distinct from wgpu
    /// running on the web (which uses GL/WebGL).
    WebGpu,

    /// Empty/null backend.
    ///
    /// Used when no adapter is available or for headless testing.
    Empty,

    /// Unknown or unrecognized backend.
    Unknown,
}

impl BackendType {
    /// Detect backend type from a wgpu adapter.
    ///
    /// # Arguments
    ///
    /// * `adapter` - The wgpu adapter to query
    ///
    /// # Returns
    ///
    /// The detected backend type.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::backend::BackendType;
    ///
    /// # async fn example() {
    /// let instance = wgpu::Instance::default();
    /// let adapter = instance
    ///     .request_adapter(&wgpu::RequestAdapterOptions::default())
    ///     .await
    ///     .unwrap();
    ///
    /// let backend = BackendType::from_adapter(&adapter);
    /// println!("Backend: {}", backend.name());
    /// # }
    /// ```
    #[inline]
    pub fn from_adapter(adapter: &Adapter) -> Self {
        Self::from_wgpu_backend(adapter.get_info().backend)
    }

    /// Convert from wgpu's Backend enum.
    ///
    /// # Arguments
    ///
    /// * `backend` - The wgpu backend enum value
    ///
    /// # Returns
    ///
    /// The corresponding `BackendType`.
    #[inline]
    pub fn from_wgpu_backend(backend: Backend) -> Self {
        match backend {
            Backend::Vulkan => BackendType::Vulkan,
            Backend::Metal => BackendType::Metal,
            Backend::Dx12 => BackendType::Dx12,
            Backend::Gl => BackendType::Gl,
            Backend::BrowserWebGpu => BackendType::WebGpu,
            Backend::Empty => BackendType::Empty,
        }
    }

    /// Get the human-readable name of this backend.
    ///
    /// # Returns
    ///
    /// A static string with the backend name.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::BackendType;
    ///
    /// assert_eq!(BackendType::Vulkan.name(), "Vulkan");
    /// assert_eq!(BackendType::Metal.name(), "Metal");
    /// assert_eq!(BackendType::Dx12.name(), "DirectX 12");
    /// ```
    #[inline]
    pub const fn name(&self) -> &'static str {
        match self {
            BackendType::Vulkan => "Vulkan",
            BackendType::Metal => "Metal",
            BackendType::Dx12 => "DirectX 12",
            BackendType::Dx11 => "DirectX 11",
            BackendType::Gl => "OpenGL",
            BackendType::WebGpu => "WebGPU",
            BackendType::Empty => "Empty",
            BackendType::Unknown => "Unknown",
        }
    }

    /// Check if this is a native (non-web) backend.
    ///
    /// Native backends run directly on the operating system's graphics stack
    /// without browser sandboxing or WebGL/WebGPU limitations.
    ///
    /// # Returns
    ///
    /// `true` for Vulkan, Metal, DX12, DX11; `false` for GL, WebGPU, Empty, Unknown.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::BackendType;
    ///
    /// assert!(BackendType::Vulkan.is_native());
    /// assert!(BackendType::Metal.is_native());
    /// assert!(!BackendType::WebGpu.is_native());
    /// assert!(!BackendType::Gl.is_native()); // GL can be native but we're conservative
    /// ```
    #[inline]
    pub const fn is_native(&self) -> bool {
        matches!(
            self,
            BackendType::Vulkan | BackendType::Metal | BackendType::Dx12 | BackendType::Dx11
        )
    }

    /// Check if this backend potentially supports ray tracing.
    ///
    /// This checks if the backend has the capability for ray tracing at the
    /// API level. Actual support depends on hardware and drivers.
    ///
    /// # Returns
    ///
    /// `true` for Vulkan, Metal, DX12; `false` for others.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::BackendType;
    ///
    /// assert!(BackendType::Vulkan.supports_ray_tracing());
    /// assert!(BackendType::Dx12.supports_ray_tracing());
    /// assert!(!BackendType::Gl.supports_ray_tracing());
    /// ```
    #[inline]
    pub const fn supports_ray_tracing(&self) -> bool {
        matches!(
            self,
            BackendType::Vulkan | BackendType::Metal | BackendType::Dx12
        )
    }

    /// Check if this backend supports mesh shaders.
    ///
    /// # Returns
    ///
    /// `true` for Vulkan, Metal, DX12; `false` for others.
    #[inline]
    pub const fn supports_mesh_shaders(&self) -> bool {
        matches!(
            self,
            BackendType::Vulkan | BackendType::Metal | BackendType::Dx12
        )
    }

    /// Check if this backend supports bindless/descriptor indexing.
    ///
    /// # Returns
    ///
    /// `true` for Vulkan, Metal, DX12; `false` for others.
    #[inline]
    pub const fn supports_bindless(&self) -> bool {
        matches!(
            self,
            BackendType::Vulkan | BackendType::Metal | BackendType::Dx12
        )
    }

    /// Convert to wgpu's Backends bitmask.
    ///
    /// # Returns
    ///
    /// A `wgpu::Backends` bitmask with only this backend enabled,
    /// or `Backends::empty()` for Empty/Unknown.
    #[inline]
    pub const fn to_backends(&self) -> wgpu::Backends {
        match self {
            BackendType::Vulkan => wgpu::Backends::VULKAN,
            BackendType::Metal => wgpu::Backends::METAL,
            BackendType::Dx12 => wgpu::Backends::DX12,
            BackendType::Dx11 => wgpu::Backends::empty(), // DX11 not in wgpu::Backends
            BackendType::Gl => wgpu::Backends::GL,
            BackendType::WebGpu => wgpu::Backends::BROWSER_WEBGPU,
            BackendType::Empty | BackendType::Unknown => wgpu::Backends::empty(),
        }
    }

    /// Check if this backend is available on the current platform.
    ///
    /// # Returns
    ///
    /// `true` if the backend can potentially be used on the current OS.
    #[inline]
    pub const fn is_available_on_platform(&self) -> bool {
        match self {
            BackendType::Vulkan => {
                cfg!(any(target_os = "windows", target_os = "linux", target_os = "android"))
            }
            BackendType::Metal => {
                cfg!(any(target_os = "macos", target_os = "ios"))
            }
            BackendType::Dx12 | BackendType::Dx11 => {
                cfg!(target_os = "windows")
            }
            BackendType::Gl => true, // GL is available on most platforms
            BackendType::WebGpu => {
                cfg!(target_arch = "wasm32")
            }
            BackendType::Empty | BackendType::Unknown => true,
        }
    }
}

impl std::fmt::Display for BackendType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl Default for BackendType {
    fn default() -> Self {
        BackendType::Unknown
    }
}

// ============================================================================
// BackendCapabilities
// ============================================================================

/// Combined backend capabilities summary.
///
/// This struct provides a unified view of what features are available
/// on a specific backend/adapter combination.
#[derive(Debug, Clone, Default)]
pub struct BackendCapabilities {
    /// The detected backend type.
    pub backend: BackendType,

    /// Whether ray tracing acceleration structures are supported.
    pub ray_tracing: bool,

    /// Whether ray queries (inline ray tracing) are supported.
    pub ray_query: bool,

    /// Whether mesh shaders are supported.
    pub mesh_shaders: bool,

    /// Whether bindless/descriptor indexing is supported.
    pub bindless: bool,

    /// Whether timeline semaphores are supported (Vulkan-specific).
    pub timeline_semaphores: bool,

    /// Whether buffer device addresses are supported (Vulkan-specific).
    pub buffer_device_address: bool,

    /// Whether dynamic rendering is supported (Vulkan-specific).
    pub dynamic_rendering: bool,
}

impl BackendCapabilities {
    /// Create capabilities from an adapter.
    ///
    /// This queries the adapter for all supported features and creates
    /// a unified capabilities view.
    ///
    /// # Arguments
    ///
    /// * `adapter` - The wgpu adapter to query
    ///
    /// # Returns
    ///
    /// The detected backend capabilities.
    pub fn from_adapter(adapter: &Adapter) -> Self {
        let backend = BackendType::from_adapter(adapter);
        let features = adapter.features();

        // Start with basic wgpu feature detection
        let mut caps = Self {
            backend,
            ray_tracing: features.contains(wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE),
            ray_query: features.contains(wgpu::Features::RAY_QUERY),
            mesh_shaders: false, // wgpu doesn't expose mesh shaders directly yet
            bindless: features.contains(wgpu::Features::TEXTURE_BINDING_ARRAY)
                && features.contains(wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING),
            timeline_semaphores: false,
            buffer_device_address: false,
            dynamic_rendering: false,
        };

        // Add Vulkan-specific detection if applicable
        if backend == BackendType::Vulkan {
            let vk_features = VulkanFeatures::detect(adapter);
            caps.timeline_semaphores = vk_features.timeline_semaphores;
            caps.buffer_device_address = vk_features.buffer_device_address;
            caps.dynamic_rendering = vk_features.dynamic_rendering;
            caps.mesh_shaders = vk_features.mesh_shading;

            // Vulkan may have better feature detection
            if !caps.ray_tracing {
                caps.ray_tracing = vk_features.ray_tracing;
            }
            if !caps.ray_query {
                caps.ray_query = vk_features.ray_query;
            }
            if !caps.bindless {
                caps.bindless = vk_features.supports_bindless();
            }
        }

        // Add Metal-specific detection if applicable
        if backend == BackendType::Metal {
            let metal_features = MetalFeatures::detect(adapter);
            caps.mesh_shaders = metal_features.mesh_shaders;

            // Metal may have better feature detection from GPU family
            if !caps.ray_tracing {
                caps.ray_tracing = metal_features.ray_tracing;
            }
            if !caps.bindless {
                caps.bindless = metal_features.supports_bindless();
            }

            // Metal 3 supports dynamic rendering patterns
            caps.dynamic_rendering = metal_features.gpu_family.supports_metal3();
        }

        // Add D3D12-specific detection if applicable
        if backend == BackendType::Dx12 {
            let dx12_features = D3D12Features::detect(adapter);
            caps.mesh_shaders = dx12_features.supports_mesh_shaders();

            // D3D12 may have better feature detection from feature level
            if !caps.ray_tracing {
                caps.ray_tracing = dx12_features.supports_rt();
            }
            if !caps.ray_query {
                caps.ray_query = dx12_features.supports_inline_rt();
            }
            if !caps.bindless {
                caps.bindless = dx12_features.supports_bindless();
            }

            // D3D12 FL 12.0+ supports dynamic rendering patterns
            caps.dynamic_rendering = dx12_features.feature_level >= D3D12FeatureLevel::FL_12_0;
        }

        // Add WebGPU-specific detection if applicable
        if backend == BackendType::WebGpu {
            let webgpu_features = WebGpuFeatures::detect(adapter);

            // WebGPU has limited ray tracing support currently
            // (no RT in base WebGPU spec as of 2024)
            caps.ray_tracing = false;
            caps.ray_query = false;

            // WebGPU doesn't expose mesh shaders yet
            caps.mesh_shaders = false;

            // Bindless is partially supported via texture binding arrays
            if !caps.bindless {
                caps.bindless = features.contains(wgpu::Features::TEXTURE_BINDING_ARRAY)
                    && features.contains(wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING);
            }

            // WebGPU uses render pass objects similar to dynamic rendering
            caps.dynamic_rendering = webgpu_features.tier >= WebGpuTier::Tier2;
        }

        caps
    }

    /// Check if full ray tracing pipeline is supported.
    ///
    /// Full RT pipeline requires both acceleration structures and ray query.
    #[inline]
    pub fn supports_full_rt(&self) -> bool {
        self.ray_tracing && self.ray_query
    }

    /// Check if GPU-driven rendering is well supported.
    ///
    /// GPU-driven rendering benefits from bindless resources and
    /// buffer device addresses.
    #[inline]
    pub fn supports_gpu_driven(&self) -> bool {
        self.bindless && (self.buffer_device_address || self.backend != BackendType::Vulkan)
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_type_name() {
        assert_eq!(BackendType::Vulkan.name(), "Vulkan");
        assert_eq!(BackendType::Metal.name(), "Metal");
        assert_eq!(BackendType::Dx12.name(), "DirectX 12");
        assert_eq!(BackendType::Dx11.name(), "DirectX 11");
        assert_eq!(BackendType::Gl.name(), "OpenGL");
        assert_eq!(BackendType::WebGpu.name(), "WebGPU");
        assert_eq!(BackendType::Empty.name(), "Empty");
        assert_eq!(BackendType::Unknown.name(), "Unknown");
    }

    #[test]
    fn test_backend_type_is_native() {
        assert!(BackendType::Vulkan.is_native());
        assert!(BackendType::Metal.is_native());
        assert!(BackendType::Dx12.is_native());
        assert!(BackendType::Dx11.is_native());
        assert!(!BackendType::Gl.is_native());
        assert!(!BackendType::WebGpu.is_native());
        assert!(!BackendType::Empty.is_native());
        assert!(!BackendType::Unknown.is_native());
    }

    #[test]
    fn test_backend_type_supports_ray_tracing() {
        assert!(BackendType::Vulkan.supports_ray_tracing());
        assert!(BackendType::Metal.supports_ray_tracing());
        assert!(BackendType::Dx12.supports_ray_tracing());
        assert!(!BackendType::Dx11.supports_ray_tracing());
        assert!(!BackendType::Gl.supports_ray_tracing());
        assert!(!BackendType::WebGpu.supports_ray_tracing());
    }

    #[test]
    fn test_backend_type_supports_mesh_shaders() {
        assert!(BackendType::Vulkan.supports_mesh_shaders());
        assert!(BackendType::Metal.supports_mesh_shaders());
        assert!(BackendType::Dx12.supports_mesh_shaders());
        assert!(!BackendType::Dx11.supports_mesh_shaders());
        assert!(!BackendType::Gl.supports_mesh_shaders());
    }

    #[test]
    fn test_backend_type_supports_bindless() {
        assert!(BackendType::Vulkan.supports_bindless());
        assert!(BackendType::Metal.supports_bindless());
        assert!(BackendType::Dx12.supports_bindless());
        assert!(!BackendType::Dx11.supports_bindless());
        assert!(!BackendType::Gl.supports_bindless());
    }

    #[test]
    fn test_backend_type_to_backends() {
        assert_eq!(BackendType::Vulkan.to_backends(), wgpu::Backends::VULKAN);
        assert_eq!(BackendType::Metal.to_backends(), wgpu::Backends::METAL);
        assert_eq!(BackendType::Dx12.to_backends(), wgpu::Backends::DX12);
        assert_eq!(BackendType::Gl.to_backends(), wgpu::Backends::GL);
        assert_eq!(BackendType::WebGpu.to_backends(), wgpu::Backends::BROWSER_WEBGPU);
        assert!(BackendType::Empty.to_backends().is_empty());
        assert!(BackendType::Unknown.to_backends().is_empty());
    }

    #[test]
    fn test_backend_type_from_wgpu_backend() {
        assert_eq!(BackendType::from_wgpu_backend(Backend::Vulkan), BackendType::Vulkan);
        assert_eq!(BackendType::from_wgpu_backend(Backend::Metal), BackendType::Metal);
        assert_eq!(BackendType::from_wgpu_backend(Backend::Dx12), BackendType::Dx12);
        assert_eq!(BackendType::from_wgpu_backend(Backend::Gl), BackendType::Gl);
        assert_eq!(BackendType::from_wgpu_backend(Backend::BrowserWebGpu), BackendType::WebGpu);
        assert_eq!(BackendType::from_wgpu_backend(Backend::Empty), BackendType::Empty);
    }

    #[test]
    fn test_backend_type_display() {
        assert_eq!(format!("{}", BackendType::Vulkan), "Vulkan");
        assert_eq!(format!("{}", BackendType::Dx12), "DirectX 12");
    }

    #[test]
    fn test_backend_type_default() {
        assert_eq!(BackendType::default(), BackendType::Unknown);
    }

    #[test]
    fn test_backend_capabilities_default() {
        let caps = BackendCapabilities::default();
        assert_eq!(caps.backend, BackendType::Unknown);
        assert!(!caps.ray_tracing);
        assert!(!caps.ray_query);
        assert!(!caps.mesh_shaders);
        assert!(!caps.bindless);
    }

    #[test]
    fn test_backend_capabilities_supports_full_rt() {
        let mut caps = BackendCapabilities::default();
        assert!(!caps.supports_full_rt());

        caps.ray_tracing = true;
        assert!(!caps.supports_full_rt());

        caps.ray_query = true;
        assert!(caps.supports_full_rt());
    }

    #[test]
    fn test_backend_capabilities_supports_gpu_driven() {
        let mut caps = BackendCapabilities::default();
        caps.backend = BackendType::Metal;
        caps.bindless = true;
        // Non-Vulkan backends don't require buffer_device_address
        assert!(caps.supports_gpu_driven());

        caps.backend = BackendType::Vulkan;
        // Vulkan requires buffer_device_address
        assert!(!caps.supports_gpu_driven());

        caps.buffer_device_address = true;
        assert!(caps.supports_gpu_driven());
    }

    #[test]
    fn test_backend_type_platform_availability() {
        // These tests verify the platform detection logic
        // The actual results depend on the compilation target

        #[cfg(target_os = "windows")]
        {
            assert!(BackendType::Vulkan.is_available_on_platform());
            assert!(BackendType::Dx12.is_available_on_platform());
            assert!(!BackendType::Metal.is_available_on_platform());
        }

        #[cfg(target_os = "macos")]
        {
            assert!(BackendType::Metal.is_available_on_platform());
            assert!(!BackendType::Vulkan.is_available_on_platform());
            assert!(!BackendType::Dx12.is_available_on_platform());
        }

        #[cfg(target_os = "linux")]
        {
            assert!(BackendType::Vulkan.is_available_on_platform());
            assert!(!BackendType::Metal.is_available_on_platform());
            assert!(!BackendType::Dx12.is_available_on_platform());
        }
    }
}
