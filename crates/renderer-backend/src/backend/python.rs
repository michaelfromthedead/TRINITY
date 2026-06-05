//! Python bindings for backend capabilities and adapter selection.
//!
//! This module provides PyO3-based Python bindings for querying GPU capabilities
//! and selecting adapters. It wraps the core Rust types to expose them to Python.
//!
//! # Available Classes
//!
//! - [`PyBackendType`] - Graphics API backend (Vulkan, Metal, DX12, WebGPU)
//! - [`PyFeatureLevel`] - Hardware feature support level (Unavailable to Optimal)
//! - [`PyUnifiedCapabilities`] - Unified GPU capabilities query
//! - [`PyAdapterInfo`] - Comprehensive adapter information
//!
//! # Example (Python)
//!
//! ```python
//! from renderer_backend import BackendType, FeatureLevel, UnifiedCapabilities
//!
//! # Check backend types
//! vulkan = BackendType.vulkan()
//! print(f"Backend: {vulkan.name()}")  # "Vulkan"
//!
//! # Feature level comparison
//! hardware = FeatureLevel.hardware()
//! optimal = FeatureLevel.optimal()
//! assert optimal >= hardware
//! ```
//!
//! # Module Registration
//!
//! The Python module is registered via the `register_python_module` function,
//! which should be called from the crate's main Python module setup.

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
#[cfg(feature = "pyo3")]
use pyo3::{types::PyModule, Bound};

#[cfg(feature = "pyo3")]
use super::{
    AdapterInfo, BackendCapabilities, BackendType, CapabilitiesQuery, DeviceType, FeatureLevel,
    GpuVendor, UnifiedCapabilities,
};

// ============================================================================
// PyBackendType
// ============================================================================

/// Python wrapper for BackendType.
///
/// Represents the graphics API backend (Vulkan, Metal, DX12, WebGPU, etc.).
///
/// # Python Example
///
/// ```python
/// from renderer_backend import BackendType
///
/// vulkan = BackendType.vulkan()
/// metal = BackendType.metal()
///
/// print(vulkan.name())  # "Vulkan"
/// print(metal)  # "BackendType.Metal"
///
/// assert vulkan != metal
/// ```
#[cfg(feature = "pyo3")]
#[pyclass(name = "BackendType")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PyBackendType(BackendType);

#[cfg(feature = "pyo3")]
#[pymethods]
impl PyBackendType {
    /// Create a Vulkan backend type.
    ///
    /// # Returns
    ///
    /// A `BackendType` representing Vulkan.
    ///
    /// # Python Example
    ///
    /// ```python
    /// backend = BackendType.vulkan()
    /// assert backend.name() == "Vulkan"
    /// ```
    #[staticmethod]
    pub fn vulkan() -> Self {
        PyBackendType(BackendType::Vulkan)
    }

    /// Create a Metal backend type.
    ///
    /// # Returns
    ///
    /// A `BackendType` representing Metal.
    ///
    /// # Python Example
    ///
    /// ```python
    /// backend = BackendType.metal()
    /// assert backend.name() == "Metal"
    /// ```
    #[staticmethod]
    pub fn metal() -> Self {
        PyBackendType(BackendType::Metal)
    }

    /// Create a DirectX 12 backend type.
    ///
    /// # Returns
    ///
    /// A `BackendType` representing DX12.
    ///
    /// # Python Example
    ///
    /// ```python
    /// backend = BackendType.dx12()
    /// assert backend.name() == "DirectX 12"
    /// ```
    #[staticmethod]
    pub fn dx12() -> Self {
        PyBackendType(BackendType::Dx12)
    }

    /// Create a WebGPU backend type.
    ///
    /// # Returns
    ///
    /// A `BackendType` representing WebGPU.
    ///
    /// # Python Example
    ///
    /// ```python
    /// backend = BackendType.webgpu()
    /// assert backend.name() == "WebGPU"
    /// ```
    #[staticmethod]
    pub fn webgpu() -> Self {
        PyBackendType(BackendType::WebGpu)
    }

    /// Create a DirectX 11 backend type.
    ///
    /// # Returns
    ///
    /// A `BackendType` representing DX11.
    #[staticmethod]
    pub fn dx11() -> Self {
        PyBackendType(BackendType::Dx11)
    }

    /// Create an OpenGL backend type.
    ///
    /// # Returns
    ///
    /// A `BackendType` representing OpenGL.
    #[staticmethod]
    pub fn gl() -> Self {
        PyBackendType(BackendType::Gl)
    }

    /// Create an Empty backend type.
    ///
    /// # Returns
    ///
    /// A `BackendType` representing an empty/null backend.
    #[staticmethod]
    pub fn empty() -> Self {
        PyBackendType(BackendType::Empty)
    }

    /// Create an Unknown backend type.
    ///
    /// # Returns
    ///
    /// A `BackendType` representing an unknown backend.
    #[staticmethod]
    pub fn unknown() -> Self {
        PyBackendType(BackendType::Unknown)
    }

    /// Get the human-readable name of this backend.
    ///
    /// # Returns
    ///
    /// A string with the backend name (e.g., "Vulkan", "Metal", "DirectX 12").
    ///
    /// # Python Example
    ///
    /// ```python
    /// backend = BackendType.vulkan()
    /// print(backend.name())  # "Vulkan"
    /// ```
    pub fn name(&self) -> &'static str {
        self.0.name()
    }

    /// Check if this is a native (non-web) backend.
    ///
    /// Native backends include Vulkan, Metal, DX12, and DX11.
    ///
    /// # Returns
    ///
    /// `True` if native, `False` otherwise.
    ///
    /// # Python Example
    ///
    /// ```python
    /// assert BackendType.vulkan().is_native()
    /// assert not BackendType.webgpu().is_native()
    /// ```
    pub fn is_native(&self) -> bool {
        self.0.is_native()
    }

    /// Check if this backend supports ray tracing at the API level.
    ///
    /// # Returns
    ///
    /// `True` if the backend API supports ray tracing.
    pub fn supports_ray_tracing(&self) -> bool {
        self.0.supports_ray_tracing()
    }

    /// Check if this backend supports mesh shaders at the API level.
    ///
    /// # Returns
    ///
    /// `True` if the backend API supports mesh shaders.
    pub fn supports_mesh_shaders(&self) -> bool {
        self.0.supports_mesh_shaders()
    }

    /// Check if this backend supports bindless/descriptor indexing.
    ///
    /// # Returns
    ///
    /// `True` if the backend API supports bindless resources.
    pub fn supports_bindless(&self) -> bool {
        self.0.supports_bindless()
    }

    /// Check if this backend is available on the current platform.
    ///
    /// # Returns
    ///
    /// `True` if the backend can potentially be used on the current OS.
    pub fn is_available_on_platform(&self) -> bool {
        self.0.is_available_on_platform()
    }

    /// Python representation.
    fn __repr__(&self) -> String {
        format!("BackendType.{}", self.0.name().replace(' ', ""))
    }

    /// Python string conversion.
    fn __str__(&self) -> String {
        self.0.name().to_string()
    }

    /// Python equality comparison.
    fn __eq__(&self, other: &Self) -> bool {
        self.0 == other.0
    }

    /// Python not-equal comparison.
    fn __ne__(&self, other: &Self) -> bool {
        self.0 != other.0
    }

    /// Python hash.
    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.0.hash(&mut hasher);
        hasher.finish()
    }
}

#[cfg(feature = "pyo3")]
impl From<BackendType> for PyBackendType {
    fn from(backend: BackendType) -> Self {
        PyBackendType(backend)
    }
}

#[cfg(feature = "pyo3")]
impl From<PyBackendType> for BackendType {
    fn from(py_backend: PyBackendType) -> Self {
        py_backend.0
    }
}

// ============================================================================
// PyFeatureLevel
// ============================================================================

/// Python wrapper for FeatureLevel.
///
/// Represents the level of support for a specific GPU feature, from
/// unavailable to optimal hardware support.
///
/// # Ordering
///
/// Feature levels are ordered: Unavailable < Emulated < Hardware < Optimal
///
/// # Python Example
///
/// ```python
/// from renderer_backend import FeatureLevel
///
/// unavailable = FeatureLevel.unavailable()
/// hardware = FeatureLevel.hardware()
/// optimal = FeatureLevel.optimal()
///
/// assert hardware > unavailable
/// assert optimal >= hardware
/// assert hardware.is_available()
/// assert not unavailable.is_available()
/// ```
#[cfg(feature = "pyo3")]
#[pyclass(name = "FeatureLevel")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PyFeatureLevel(FeatureLevel);

#[cfg(feature = "pyo3")]
#[pymethods]
impl PyFeatureLevel {
    /// Create an Unavailable feature level.
    ///
    /// # Returns
    ///
    /// A `FeatureLevel` indicating the feature is not available.
    #[staticmethod]
    pub fn unavailable() -> Self {
        PyFeatureLevel(FeatureLevel::Unavailable)
    }

    /// Create an Emulated feature level.
    ///
    /// # Returns
    ///
    /// A `FeatureLevel` indicating the feature is emulated in software.
    #[staticmethod]
    pub fn emulated() -> Self {
        PyFeatureLevel(FeatureLevel::Emulated)
    }

    /// Create a Hardware feature level.
    ///
    /// # Returns
    ///
    /// A `FeatureLevel` indicating native hardware support.
    #[staticmethod]
    pub fn hardware() -> Self {
        PyFeatureLevel(FeatureLevel::Hardware)
    }

    /// Create an Optimal feature level.
    ///
    /// # Returns
    ///
    /// A `FeatureLevel` indicating best-in-class hardware support.
    #[staticmethod]
    pub fn optimal() -> Self {
        PyFeatureLevel(FeatureLevel::Optimal)
    }

    /// Check if the feature is at least minimally available.
    ///
    /// # Returns
    ///
    /// `True` if the feature is at least emulated (not unavailable).
    ///
    /// # Python Example
    ///
    /// ```python
    /// assert FeatureLevel.emulated().is_available()
    /// assert not FeatureLevel.unavailable().is_available()
    /// ```
    pub fn is_available(&self) -> bool {
        self.0.is_available()
    }

    /// Check if the feature has native hardware support.
    ///
    /// # Returns
    ///
    /// `True` if the feature has at least hardware-level support.
    ///
    /// # Python Example
    ///
    /// ```python
    /// assert FeatureLevel.hardware().is_hardware()
    /// assert FeatureLevel.optimal().is_hardware()
    /// assert not FeatureLevel.emulated().is_hardware()
    /// ```
    pub fn is_hardware(&self) -> bool {
        self.0.is_hardware()
    }

    /// Check if the feature has optimal support.
    ///
    /// # Returns
    ///
    /// `True` only for optimal-level support.
    pub fn is_optimal(&self) -> bool {
        self.0.is_optimal()
    }

    /// Get the feature level name.
    ///
    /// # Returns
    ///
    /// The name as a string (e.g., "Hardware", "Optimal").
    pub fn name(&self) -> &'static str {
        self.0.name()
    }

    /// Python representation.
    fn __repr__(&self) -> String {
        format!("FeatureLevel.{}", self.0.name())
    }

    /// Python string conversion.
    fn __str__(&self) -> String {
        self.0.name().to_string()
    }

    /// Python equality comparison.
    fn __eq__(&self, other: &Self) -> bool {
        self.0 == other.0
    }

    /// Python not-equal comparison.
    fn __ne__(&self, other: &Self) -> bool {
        self.0 != other.0
    }

    /// Python greater-than-or-equal comparison.
    ///
    /// Allows `optimal >= hardware` style comparisons in Python.
    fn __ge__(&self, other: &Self) -> bool {
        self.0 >= other.0
    }

    /// Python greater-than comparison.
    fn __gt__(&self, other: &Self) -> bool {
        self.0 > other.0
    }

    /// Python less-than-or-equal comparison.
    fn __le__(&self, other: &Self) -> bool {
        self.0 <= other.0
    }

    /// Python less-than comparison.
    fn __lt__(&self, other: &Self) -> bool {
        self.0 < other.0
    }

    /// Python hash.
    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.0.hash(&mut hasher);
        hasher.finish()
    }
}

#[cfg(feature = "pyo3")]
impl From<FeatureLevel> for PyFeatureLevel {
    fn from(level: FeatureLevel) -> Self {
        PyFeatureLevel(level)
    }
}

#[cfg(feature = "pyo3")]
impl From<PyFeatureLevel> for FeatureLevel {
    fn from(py_level: PyFeatureLevel) -> Self {
        py_level.0
    }
}

// ============================================================================
// PyUnifiedCapabilities
// ============================================================================

/// Python wrapper for UnifiedCapabilities.
///
/// Provides a unified view of GPU capabilities across all backends.
/// This abstracts over backend-specific differences to provide a
/// consistent interface for capability queries.
///
/// # Python Example
///
/// ```python
/// from renderer_backend import UnifiedCapabilities
///
/// # Get capabilities from an adapter (typically via PyAdapterInfo)
/// caps = adapter_info.capabilities()
///
/// # Query feature support
/// if caps.ray_tracing().is_hardware():
///     print("Hardware ray tracing available!")
///
/// # Check limits
/// print(f"Max texture size: {caps.max_texture_size()}")
/// print(f"Feature score: {caps.feature_score()}")
/// ```
#[cfg(feature = "pyo3")]
#[pyclass(name = "UnifiedCapabilities")]
#[derive(Clone, Debug)]
pub struct PyUnifiedCapabilities {
    inner: UnifiedCapabilities,
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl PyUnifiedCapabilities {
    /// Get the backend type.
    ///
    /// # Returns
    ///
    /// The `BackendType` for this capability set.
    pub fn backend(&self) -> PyBackendType {
        PyBackendType(self.inner.backend)
    }

    /// Get ray tracing support level.
    ///
    /// # Returns
    ///
    /// The `FeatureLevel` for ray tracing support.
    ///
    /// # Python Example
    ///
    /// ```python
    /// if caps.ray_tracing() >= FeatureLevel.hardware():
    ///     print("Hardware RT available")
    /// ```
    pub fn ray_tracing(&self) -> PyFeatureLevel {
        PyFeatureLevel(self.inner.ray_tracing)
    }

    /// Get mesh shader support level.
    ///
    /// # Returns
    ///
    /// The `FeatureLevel` for mesh shader support.
    pub fn mesh_shaders(&self) -> PyFeatureLevel {
        PyFeatureLevel(self.inner.mesh_shaders)
    }

    /// Get variable rate shading support level.
    ///
    /// # Returns
    ///
    /// The `FeatureLevel` for VRS support.
    pub fn variable_rate_shading(&self) -> PyFeatureLevel {
        PyFeatureLevel(self.inner.variable_rate_shading)
    }

    /// Get bindless/descriptor indexing support level.
    ///
    /// # Returns
    ///
    /// The `FeatureLevel` for bindless resource support.
    pub fn bindless(&self) -> PyFeatureLevel {
        PyFeatureLevel(self.inner.bindless)
    }

    /// Get async compute support level.
    ///
    /// # Returns
    ///
    /// The `FeatureLevel` for async compute support.
    pub fn async_compute(&self) -> PyFeatureLevel {
        PyFeatureLevel(self.inner.async_compute)
    }

    /// Get conservative rasterization support level.
    ///
    /// # Returns
    ///
    /// The `FeatureLevel` for conservative rasterization.
    pub fn conservative_rasterization(&self) -> PyFeatureLevel {
        PyFeatureLevel(self.inner.conservative_rasterization)
    }

    /// Get sampler feedback support level.
    ///
    /// # Returns
    ///
    /// The `FeatureLevel` for sampler feedback.
    pub fn sampler_feedback(&self) -> PyFeatureLevel {
        PyFeatureLevel(self.inner.sampler_feedback)
    }

    /// Get maximum texture dimension (width/height) in texels.
    ///
    /// # Returns
    ///
    /// The maximum texture size.
    pub fn max_texture_size(&self) -> u32 {
        self.inner.max_texture_size
    }

    /// Get maximum buffer size in bytes.
    ///
    /// # Returns
    ///
    /// The maximum buffer size.
    pub fn max_buffer_size(&self) -> u64 {
        self.inner.max_buffer_size
    }

    /// Get maximum compute workgroup size [x, y, z].
    ///
    /// # Returns
    ///
    /// A tuple of (x, y, z) maximum workgroup dimensions.
    pub fn max_compute_workgroup_size(&self) -> (u32, u32, u32) {
        let size = self.inner.max_compute_workgroup_size;
        (size[0], size[1], size[2])
    }

    /// Check if full ray tracing pipeline is supported.
    ///
    /// Full RT requires at least hardware-level ray tracing support.
    ///
    /// # Returns
    ///
    /// `True` if full RT pipeline is available.
    ///
    /// # Python Example
    ///
    /// ```python
    /// if caps.supports_full_rt():
    ///     enable_ray_traced_shadows()
    /// ```
    pub fn supports_full_rt(&self) -> bool {
        self.inner.supports_full_rt()
    }

    /// Check if GPU-driven rendering is well supported.
    ///
    /// GPU-driven rendering requires bindless resources and async compute.
    ///
    /// # Returns
    ///
    /// `True` if GPU-driven rendering is viable.
    pub fn supports_gpu_driven(&self) -> bool {
        self.inner.supports_gpu_driven()
    }

    /// Check if advanced features are supported.
    ///
    /// Advanced features include ray tracing, mesh shaders, and VRS.
    ///
    /// # Returns
    ///
    /// `True` if any advanced features are available.
    pub fn supports_advanced_features(&self) -> bool {
        self.inner.supports_advanced_features()
    }

    /// Get the total feature score for comparison purposes.
    ///
    /// Higher scores indicate more capable hardware. Useful for
    /// sorting adapters by capability.
    ///
    /// # Returns
    ///
    /// A numeric score (higher is better).
    ///
    /// # Python Example
    ///
    /// ```python
    /// adapters.sort(key=lambda a: a.capabilities().feature_score(), reverse=True)
    /// ```
    pub fn feature_score(&self) -> u32 {
        self.inner.feature_score()
    }

    /// Python representation.
    fn __repr__(&self) -> String {
        format!(
            "UnifiedCapabilities(backend={}, rt={}, mesh={}, score={})",
            self.inner.backend.name(),
            self.inner.ray_tracing.name(),
            self.inner.mesh_shaders.name(),
            self.inner.feature_score()
        )
    }
}

#[cfg(feature = "pyo3")]
impl From<UnifiedCapabilities> for PyUnifiedCapabilities {
    fn from(caps: UnifiedCapabilities) -> Self {
        PyUnifiedCapabilities { inner: caps }
    }
}

#[cfg(feature = "pyo3")]
impl From<&UnifiedCapabilities> for PyUnifiedCapabilities {
    fn from(caps: &UnifiedCapabilities) -> Self {
        PyUnifiedCapabilities {
            inner: caps.clone(),
        }
    }
}

// ============================================================================
// PyAdapterInfo
// ============================================================================

/// Python wrapper for AdapterInfo.
///
/// Provides comprehensive information about a GPU adapter, including
/// name, vendor, device type, backend, and capabilities.
///
/// # Python Example
///
/// ```python
/// from renderer_backend import AdapterInfo
///
/// # Get adapter info (typically from enumerate_adapters)
/// info = adapter_info
///
/// print(f"Name: {info.name()}")
/// print(f"Vendor: {info.vendor()}")
/// print(f"Type: {info.device_type()}")
/// print(f"Backend: {info.backend().name()}")
///
/// if info.is_discrete():
///     print("This is a discrete GPU")
///
/// if info.supports_ray_tracing():
///     print("Ray tracing supported!")
///
/// caps = info.capabilities()
/// print(f"Feature score: {caps.feature_score()}")
/// ```
#[cfg(feature = "pyo3")]
#[pyclass(name = "AdapterInfo")]
#[derive(Clone, Debug)]
pub struct PyAdapterInfo {
    inner: AdapterInfo,
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl PyAdapterInfo {
    /// Get the adapter name.
    ///
    /// # Returns
    ///
    /// The adapter name (e.g., "NVIDIA GeForce RTX 4090").
    pub fn name(&self) -> &str {
        &self.inner.name
    }

    /// Get the vendor name.
    ///
    /// # Returns
    ///
    /// The vendor name (e.g., "NVIDIA", "AMD", "Intel").
    pub fn vendor(&self) -> &'static str {
        self.inner.vendor.name()
    }

    /// Get the device type name.
    ///
    /// # Returns
    ///
    /// The device type (e.g., "Discrete GPU", "Integrated GPU").
    pub fn device_type(&self) -> &'static str {
        self.inner.device_type.name()
    }

    /// Get the backend type.
    ///
    /// # Returns
    ///
    /// The `BackendType` for this adapter.
    pub fn backend(&self) -> PyBackendType {
        PyBackendType(self.inner.backend)
    }

    /// Get the driver version string.
    ///
    /// # Returns
    ///
    /// The driver version, or empty string if unavailable.
    pub fn driver_version(&self) -> &str {
        &self.inner.driver_version
    }

    /// Check if this is a discrete (dedicated) GPU.
    ///
    /// # Returns
    ///
    /// `True` if discrete GPU.
    pub fn is_discrete(&self) -> bool {
        self.inner.is_discrete()
    }

    /// Check if this is an integrated GPU.
    ///
    /// # Returns
    ///
    /// `True` if integrated GPU.
    pub fn is_integrated(&self) -> bool {
        self.inner.is_integrated()
    }

    /// Check if this is a software renderer.
    ///
    /// # Returns
    ///
    /// `True` if CPU/software renderer.
    pub fn is_software(&self) -> bool {
        self.inner.is_software()
    }

    /// Check if ray tracing is supported.
    ///
    /// # Returns
    ///
    /// `True` if ray tracing is available.
    pub fn supports_ray_tracing(&self) -> bool {
        self.inner.supports_ray_tracing()
    }

    /// Check if bindless rendering is supported.
    ///
    /// # Returns
    ///
    /// `True` if bindless is available.
    pub fn supports_bindless(&self) -> bool {
        self.inner.supports_bindless()
    }

    /// Check if mesh shaders are supported.
    ///
    /// # Returns
    ///
    /// `True` if mesh shaders are available.
    pub fn supports_mesh_shaders(&self) -> bool {
        self.inner.supports_mesh_shaders()
    }

    /// Get the unified capabilities for this adapter.
    ///
    /// # Returns
    ///
    /// A `UnifiedCapabilities` object with detailed capability information.
    ///
    /// # Python Example
    ///
    /// ```python
    /// caps = info.capabilities()
    /// print(f"Ray tracing: {caps.ray_tracing().name()}")
    /// ```
    pub fn capabilities(&self) -> PyUnifiedCapabilities {
        // Create UnifiedCapabilities from BackendCapabilities
        let backend_caps = &self.inner.capabilities;
        let limits = &self.inner.limits;

        let unified = UnifiedCapabilities {
            backend: self.inner.backend,
            ray_tracing: if backend_caps.ray_tracing && backend_caps.ray_query {
                FeatureLevel::Optimal
            } else if backend_caps.ray_tracing {
                FeatureLevel::Hardware
            } else {
                FeatureLevel::Unavailable
            },
            mesh_shaders: if backend_caps.mesh_shaders {
                FeatureLevel::Hardware
            } else {
                FeatureLevel::Unavailable
            },
            variable_rate_shading: FeatureLevel::Unavailable, // Not exposed in BackendCapabilities
            bindless: if backend_caps.bindless {
                FeatureLevel::Hardware
            } else {
                FeatureLevel::Unavailable
            },
            async_compute: if backend_caps.timeline_semaphores {
                FeatureLevel::Optimal
            } else {
                FeatureLevel::Hardware // Most modern GPUs have async compute
            },
            conservative_rasterization: FeatureLevel::Unavailable,
            sampler_feedback: FeatureLevel::Unavailable,
            max_texture_size: limits.max_texture_dimension_2d,
            max_buffer_size: limits.max_buffer_size as u64,
            max_compute_workgroup_size: [
                limits.max_compute_workgroup_size_x,
                limits.max_compute_workgroup_size_y,
                limits.max_compute_workgroup_size_z,
            ],
        };

        PyUnifiedCapabilities { inner: unified }
    }

    /// Get estimated VRAM in MB.
    ///
    /// Note: This is an estimate based on max_buffer_size.
    ///
    /// # Returns
    ///
    /// Estimated VRAM in megabytes.
    pub fn estimated_vram_mb(&self) -> u64 {
        self.inner.estimated_vram_mb()
    }

    /// Get the raw vendor ID.
    ///
    /// # Returns
    ///
    /// The PCI vendor ID.
    pub fn vendor_id(&self) -> u32 {
        self.inner.vendor_id
    }

    /// Get the raw device ID.
    ///
    /// # Returns
    ///
    /// The PCI device ID.
    pub fn device_id(&self) -> u32 {
        self.inner.device_id
    }

    /// Get a human-readable description.
    ///
    /// # Returns
    ///
    /// A description string including name, type, and backend.
    pub fn description(&self) -> String {
        self.inner.description()
    }

    /// Python representation.
    fn __repr__(&self) -> String {
        format!(
            "AdapterInfo(name='{}', vendor='{}', type='{}', backend='{}')",
            self.inner.name,
            self.inner.vendor.name(),
            self.inner.device_type.name(),
            self.inner.backend.name()
        )
    }

    /// Python string conversion.
    fn __str__(&self) -> String {
        self.inner.description()
    }
}

#[cfg(feature = "pyo3")]
impl From<AdapterInfo> for PyAdapterInfo {
    fn from(info: AdapterInfo) -> Self {
        PyAdapterInfo { inner: info }
    }
}

#[cfg(feature = "pyo3")]
impl From<&AdapterInfo> for PyAdapterInfo {
    fn from(info: &AdapterInfo) -> Self {
        PyAdapterInfo {
            inner: info.clone(),
        }
    }
}

// ============================================================================
// Module Registration
// ============================================================================

/// Register the backend Python classes with a PyO3 module.
///
/// This function adds all backend-related Python classes to the given module.
/// Call this from the crate's main Python module setup.
///
/// # Arguments
///
/// * `m` - The Python module to register classes with
///
/// # Returns
///
/// `PyResult<()>` indicating success or failure.
///
/// # Example
///
/// ```ignore
/// use pyo3::prelude::*;
///
/// #[pymodule]
/// fn renderer_backend(_py: Python, m: &PyModule) -> PyResult<()> {
///     backend::python::register_python_module(m)?;
///     Ok(())
/// }
/// ```
#[cfg(feature = "pyo3")]
pub fn register_python_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBackendType>()?;
    m.add_class::<PyFeatureLevel>()?;
    m.add_class::<PyUnifiedCapabilities>()?;
    m.add_class::<PyAdapterInfo>()?;
    Ok(())
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    #[cfg(feature = "pyo3")]
    use super::*;

    // ========================================================================
    // PyBackendType Tests
    // ========================================================================

    #[cfg(feature = "pyo3")]
    mod backend_type_tests {
        use super::*;

        #[test]
        fn test_vulkan_variant() {
            let backend = PyBackendType::vulkan();
            assert_eq!(backend.name(), "Vulkan");
            assert!(backend.is_native());
            assert!(backend.supports_ray_tracing());
            assert!(backend.supports_mesh_shaders());
            assert!(backend.supports_bindless());
        }

        #[test]
        fn test_metal_variant() {
            let backend = PyBackendType::metal();
            assert_eq!(backend.name(), "Metal");
            assert!(backend.is_native());
            assert!(backend.supports_ray_tracing());
        }

        #[test]
        fn test_dx12_variant() {
            let backend = PyBackendType::dx12();
            assert_eq!(backend.name(), "DirectX 12");
            assert!(backend.is_native());
            assert!(backend.supports_ray_tracing());
        }

        #[test]
        fn test_webgpu_variant() {
            let backend = PyBackendType::webgpu();
            assert_eq!(backend.name(), "WebGPU");
            assert!(!backend.is_native());
            assert!(!backend.supports_ray_tracing());
        }

        #[test]
        fn test_dx11_variant() {
            let backend = PyBackendType::dx11();
            assert_eq!(backend.name(), "DirectX 11");
            assert!(backend.is_native());
            assert!(!backend.supports_ray_tracing());
        }

        #[test]
        fn test_gl_variant() {
            let backend = PyBackendType::gl();
            assert_eq!(backend.name(), "OpenGL");
            assert!(!backend.is_native());
        }

        #[test]
        fn test_empty_variant() {
            let backend = PyBackendType::empty();
            assert_eq!(backend.name(), "Empty");
        }

        #[test]
        fn test_unknown_variant() {
            let backend = PyBackendType::unknown();
            assert_eq!(backend.name(), "Unknown");
        }

        #[test]
        fn test_repr() {
            let backend = PyBackendType::vulkan();
            assert_eq!(backend.__repr__(), "BackendType.Vulkan");

            let dx12 = PyBackendType::dx12();
            assert_eq!(dx12.__repr__(), "BackendType.DirectX12");
        }

        #[test]
        fn test_str() {
            let backend = PyBackendType::vulkan();
            assert_eq!(backend.__str__(), "Vulkan");
        }

        #[test]
        fn test_equality() {
            let a = PyBackendType::vulkan();
            let b = PyBackendType::vulkan();
            let c = PyBackendType::metal();

            assert!(a.__eq__(&b));
            assert!(!a.__eq__(&c));
            assert!(a.__ne__(&c));
            assert!(!a.__ne__(&b));
        }

        #[test]
        fn test_hash() {
            let a = PyBackendType::vulkan();
            let b = PyBackendType::vulkan();
            assert_eq!(a.__hash__(), b.__hash__());
        }

        #[test]
        fn test_from_backend_type() {
            let rust_backend = BackendType::Vulkan;
            let py_backend: PyBackendType = rust_backend.into();
            assert_eq!(py_backend.name(), "Vulkan");
        }

        #[test]
        fn test_into_backend_type() {
            let py_backend = PyBackendType::metal();
            let rust_backend: BackendType = py_backend.into();
            assert_eq!(rust_backend, BackendType::Metal);
        }
    }

    // ========================================================================
    // PyFeatureLevel Tests
    // ========================================================================

    #[cfg(feature = "pyo3")]
    mod feature_level_tests {
        use super::*;

        #[test]
        fn test_unavailable_level() {
            let level = PyFeatureLevel::unavailable();
            assert_eq!(level.name(), "Unavailable");
            assert!(!level.is_available());
            assert!(!level.is_hardware());
            assert!(!level.is_optimal());
        }

        #[test]
        fn test_emulated_level() {
            let level = PyFeatureLevel::emulated();
            assert_eq!(level.name(), "Emulated");
            assert!(level.is_available());
            assert!(!level.is_hardware());
            assert!(!level.is_optimal());
        }

        #[test]
        fn test_hardware_level() {
            let level = PyFeatureLevel::hardware();
            assert_eq!(level.name(), "Hardware");
            assert!(level.is_available());
            assert!(level.is_hardware());
            assert!(!level.is_optimal());
        }

        #[test]
        fn test_optimal_level() {
            let level = PyFeatureLevel::optimal();
            assert_eq!(level.name(), "Optimal");
            assert!(level.is_available());
            assert!(level.is_hardware());
            assert!(level.is_optimal());
        }

        #[test]
        fn test_ordering_ge() {
            let unavailable = PyFeatureLevel::unavailable();
            let emulated = PyFeatureLevel::emulated();
            let hardware = PyFeatureLevel::hardware();
            let optimal = PyFeatureLevel::optimal();

            assert!(optimal.__ge__(&hardware));
            assert!(optimal.__ge__(&optimal));
            assert!(hardware.__ge__(&emulated));
            assert!(emulated.__ge__(&unavailable));
            assert!(!unavailable.__ge__(&emulated));
        }

        #[test]
        fn test_ordering_gt() {
            let hardware = PyFeatureLevel::hardware();
            let optimal = PyFeatureLevel::optimal();

            assert!(optimal.__gt__(&hardware));
            assert!(!hardware.__gt__(&optimal));
            assert!(!optimal.__gt__(&optimal));
        }

        #[test]
        fn test_ordering_le() {
            let hardware = PyFeatureLevel::hardware();
            let optimal = PyFeatureLevel::optimal();

            assert!(hardware.__le__(&optimal));
            assert!(hardware.__le__(&hardware));
            assert!(!optimal.__le__(&hardware));
        }

        #[test]
        fn test_ordering_lt() {
            let hardware = PyFeatureLevel::hardware();
            let optimal = PyFeatureLevel::optimal();

            assert!(hardware.__lt__(&optimal));
            assert!(!optimal.__lt__(&hardware));
        }

        #[test]
        fn test_repr() {
            let level = PyFeatureLevel::hardware();
            assert_eq!(level.__repr__(), "FeatureLevel.Hardware");
        }

        #[test]
        fn test_str() {
            let level = PyFeatureLevel::optimal();
            assert_eq!(level.__str__(), "Optimal");
        }

        #[test]
        fn test_equality() {
            let a = PyFeatureLevel::hardware();
            let b = PyFeatureLevel::hardware();
            let c = PyFeatureLevel::optimal();

            assert!(a.__eq__(&b));
            assert!(!a.__eq__(&c));
            assert!(a.__ne__(&c));
        }

        #[test]
        fn test_hash() {
            let a = PyFeatureLevel::hardware();
            let b = PyFeatureLevel::hardware();
            assert_eq!(a.__hash__(), b.__hash__());
        }

        #[test]
        fn test_from_feature_level() {
            let rust_level = FeatureLevel::Hardware;
            let py_level: PyFeatureLevel = rust_level.into();
            assert_eq!(py_level.name(), "Hardware");
        }

        #[test]
        fn test_into_feature_level() {
            let py_level = PyFeatureLevel::optimal();
            let rust_level: FeatureLevel = py_level.into();
            assert_eq!(rust_level, FeatureLevel::Optimal);
        }
    }

    // ========================================================================
    // PyUnifiedCapabilities Tests
    // ========================================================================

    #[cfg(feature = "pyo3")]
    mod unified_capabilities_tests {
        use super::*;

        fn make_test_capabilities() -> PyUnifiedCapabilities {
            PyUnifiedCapabilities {
                inner: UnifiedCapabilities {
                    backend: BackendType::Vulkan,
                    ray_tracing: FeatureLevel::Hardware,
                    mesh_shaders: FeatureLevel::Hardware,
                    variable_rate_shading: FeatureLevel::Emulated,
                    bindless: FeatureLevel::Optimal,
                    async_compute: FeatureLevel::Hardware,
                    conservative_rasterization: FeatureLevel::Unavailable,
                    sampler_feedback: FeatureLevel::Unavailable,
                    max_texture_size: 16384,
                    max_buffer_size: 2 * 1024 * 1024 * 1024,
                    max_compute_workgroup_size: [256, 256, 64],
                },
            }
        }

        #[test]
        fn test_backend() {
            let caps = make_test_capabilities();
            assert_eq!(caps.backend().name(), "Vulkan");
        }

        #[test]
        fn test_ray_tracing() {
            let caps = make_test_capabilities();
            assert_eq!(caps.ray_tracing().name(), "Hardware");
            assert!(caps.ray_tracing().is_hardware());
        }

        #[test]
        fn test_mesh_shaders() {
            let caps = make_test_capabilities();
            assert_eq!(caps.mesh_shaders().name(), "Hardware");
        }

        #[test]
        fn test_variable_rate_shading() {
            let caps = make_test_capabilities();
            assert_eq!(caps.variable_rate_shading().name(), "Emulated");
        }

        #[test]
        fn test_bindless() {
            let caps = make_test_capabilities();
            assert_eq!(caps.bindless().name(), "Optimal");
        }

        #[test]
        fn test_async_compute() {
            let caps = make_test_capabilities();
            assert_eq!(caps.async_compute().name(), "Hardware");
        }

        #[test]
        fn test_conservative_rasterization() {
            let caps = make_test_capabilities();
            assert_eq!(caps.conservative_rasterization().name(), "Unavailable");
        }

        #[test]
        fn test_sampler_feedback() {
            let caps = make_test_capabilities();
            assert_eq!(caps.sampler_feedback().name(), "Unavailable");
        }

        #[test]
        fn test_max_texture_size() {
            let caps = make_test_capabilities();
            assert_eq!(caps.max_texture_size(), 16384);
        }

        #[test]
        fn test_max_buffer_size() {
            let caps = make_test_capabilities();
            assert_eq!(caps.max_buffer_size(), 2 * 1024 * 1024 * 1024);
        }

        #[test]
        fn test_max_compute_workgroup_size() {
            let caps = make_test_capabilities();
            assert_eq!(caps.max_compute_workgroup_size(), (256, 256, 64));
        }

        #[test]
        fn test_supports_full_rt() {
            let caps = make_test_capabilities();
            assert!(caps.supports_full_rt());
        }

        #[test]
        fn test_supports_gpu_driven() {
            let caps = make_test_capabilities();
            assert!(caps.supports_gpu_driven());
        }

        #[test]
        fn test_supports_advanced_features() {
            let caps = make_test_capabilities();
            assert!(caps.supports_advanced_features());
        }

        #[test]
        fn test_feature_score() {
            let caps = make_test_capabilities();
            let score = caps.feature_score();
            assert!(score > 0);
        }

        #[test]
        fn test_repr() {
            let caps = make_test_capabilities();
            let repr = caps.__repr__();
            assert!(repr.contains("UnifiedCapabilities"));
            assert!(repr.contains("Vulkan"));
        }

        #[test]
        fn test_from_unified_capabilities() {
            let rust_caps = UnifiedCapabilities::default();
            let py_caps: PyUnifiedCapabilities = rust_caps.into();
            assert_eq!(py_caps.backend().name(), "Unknown");
        }

        #[test]
        fn test_from_ref_unified_capabilities() {
            let rust_caps = UnifiedCapabilities::default();
            let py_caps: PyUnifiedCapabilities = (&rust_caps).into();
            assert_eq!(py_caps.backend().name(), "Unknown");
        }
    }

    // ========================================================================
    // PyAdapterInfo Tests
    // ========================================================================

    #[cfg(feature = "pyo3")]
    mod adapter_info_tests {
        use super::*;
        use wgpu::{Features, Limits};

        fn make_test_adapter_info() -> PyAdapterInfo {
            PyAdapterInfo {
                inner: AdapterInfo {
                    name: "NVIDIA GeForce RTX 4090".to_string(),
                    vendor: GpuVendor::Nvidia,
                    device_type: DeviceType::DiscreteGpu,
                    backend: BackendType::Vulkan,
                    driver_version: "545.92".to_string(),
                    capabilities: BackendCapabilities {
                        backend: BackendType::Vulkan,
                        ray_tracing: true,
                        ray_query: true,
                        mesh_shaders: true,
                        bindless: true,
                        timeline_semaphores: true,
                        buffer_device_address: true,
                        dynamic_rendering: true,
                    },
                    features: Features::empty(),
                    limits: Limits::default(),
                    vendor_id: 0x10DE,
                    device_id: 0x2684,
                },
            }
        }

        #[test]
        fn test_name() {
            let info = make_test_adapter_info();
            assert_eq!(info.name(), "NVIDIA GeForce RTX 4090");
        }

        #[test]
        fn test_vendor() {
            let info = make_test_adapter_info();
            assert_eq!(info.vendor(), "NVIDIA");
        }

        #[test]
        fn test_device_type() {
            let info = make_test_adapter_info();
            assert_eq!(info.device_type(), "Discrete GPU");
        }

        #[test]
        fn test_backend() {
            let info = make_test_adapter_info();
            assert_eq!(info.backend().name(), "Vulkan");
        }

        #[test]
        fn test_driver_version() {
            let info = make_test_adapter_info();
            assert_eq!(info.driver_version(), "545.92");
        }

        #[test]
        fn test_is_discrete() {
            let info = make_test_adapter_info();
            assert!(info.is_discrete());
            assert!(!info.is_integrated());
            assert!(!info.is_software());
        }

        #[test]
        fn test_supports_ray_tracing() {
            let info = make_test_adapter_info();
            assert!(info.supports_ray_tracing());
        }

        #[test]
        fn test_supports_bindless() {
            let info = make_test_adapter_info();
            assert!(info.supports_bindless());
        }

        #[test]
        fn test_supports_mesh_shaders() {
            let info = make_test_adapter_info();
            assert!(info.supports_mesh_shaders());
        }

        #[test]
        fn test_capabilities() {
            let info = make_test_adapter_info();
            let caps = info.capabilities();
            assert_eq!(caps.backend().name(), "Vulkan");
            // With both ray_tracing and ray_query true, should be Optimal
            assert_eq!(caps.ray_tracing().name(), "Optimal");
        }

        #[test]
        fn test_estimated_vram_mb() {
            let info = make_test_adapter_info();
            let vram = info.estimated_vram_mb();
            assert!(vram > 0);
        }

        #[test]
        fn test_vendor_id() {
            let info = make_test_adapter_info();
            assert_eq!(info.vendor_id(), 0x10DE);
        }

        #[test]
        fn test_device_id() {
            let info = make_test_adapter_info();
            assert_eq!(info.device_id(), 0x2684);
        }

        #[test]
        fn test_description() {
            let info = make_test_adapter_info();
            let desc = info.description();
            assert!(desc.contains("NVIDIA GeForce RTX 4090"));
            assert!(desc.contains("Discrete GPU"));
            assert!(desc.contains("Vulkan"));
        }

        #[test]
        fn test_repr() {
            let info = make_test_adapter_info();
            let repr = info.__repr__();
            assert!(repr.contains("AdapterInfo"));
            assert!(repr.contains("NVIDIA GeForce RTX 4090"));
            assert!(repr.contains("NVIDIA"));
        }

        #[test]
        fn test_str() {
            let info = make_test_adapter_info();
            let s = info.__str__();
            assert_eq!(s, info.description());
        }

        #[test]
        fn test_from_adapter_info() {
            let rust_info = AdapterInfo {
                name: "Test GPU".to_string(),
                vendor: GpuVendor::AMD,
                device_type: DeviceType::IntegratedGpu,
                backend: BackendType::Dx12,
                driver_version: "1.0".to_string(),
                capabilities: BackendCapabilities::default(),
                features: Features::empty(),
                limits: Limits::default(),
                vendor_id: 0x1002,
                device_id: 0x0000,
            };
            let py_info: PyAdapterInfo = rust_info.into();
            assert_eq!(py_info.name(), "Test GPU");
            assert_eq!(py_info.vendor(), "AMD");
        }

        #[test]
        fn test_from_ref_adapter_info() {
            let rust_info = AdapterInfo {
                name: "Test GPU".to_string(),
                vendor: GpuVendor::Intel,
                device_type: DeviceType::IntegratedGpu,
                backend: BackendType::Vulkan,
                driver_version: "".to_string(),
                capabilities: BackendCapabilities::default(),
                features: Features::empty(),
                limits: Limits::default(),
                vendor_id: 0x8086,
                device_id: 0x0000,
            };
            let py_info: PyAdapterInfo = (&rust_info).into();
            assert_eq!(py_info.vendor(), "Intel");
        }
    }

    // ========================================================================
    // Non-pyo3 fallback tests
    // ========================================================================

    #[cfg(not(feature = "pyo3"))]
    mod non_pyo3_tests {
        #[test]
        fn test_module_compiles_without_pyo3() {
            // This test just verifies the module compiles when pyo3 is disabled
            assert!(true);
        }
    }
}
