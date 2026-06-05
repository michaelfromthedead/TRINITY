//! Ray Tracing Capability Detection
//!
//! This module provides a capability detection system that gates ray tracing
//! features based on device capabilities. It queries wgpu feature flags and
//! provides appropriate fallback methods for rendering effects.
//!
//! # Architecture
//!
//! - `RTCapability`: Enum representing the level of RT support
//! - `RTCapabilityInfo`: Detailed capability information with query methods
//! - `RTEffectRouter`: Routes rendering effects to appropriate implementations
//! - Fallback enums: `ShadowMethod`, `ReflectionMethod`, `GIMethod`
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::rt_capability::{get_rt_capability, RTEffectRouter};
//!
//! let capability_info = get_rt_capability(&device_features);
//!
//! if capability_info.supports_shadows() {
//!     // Use ray-traced shadows
//! }
//!
//! let shadow_method = RTEffectRouter::route_shadows(capability_info.capability);
//! match shadow_method {
//!     ShadowMethod::RayTraced => use_rt_shadows(),
//!     ShadowMethod::CascadedShadowMaps => use_csm(),
//!     ShadowMethod::None => disable_shadows(),
//! }
//! ```

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default maximum ray recursion depth when RT is supported.
const DEFAULT_MAX_RAY_RECURSION_DEPTH: u32 = 1;

/// Maximum ray recursion depth for full RT pipeline support.
const FULL_RT_MAX_RAY_RECURSION_DEPTH: u32 = 8;

/// Default acceleration structure alignment in bytes.
const DEFAULT_AS_ALIGNMENT: u32 = 256;

// ---------------------------------------------------------------------------
// RTCapability
// ---------------------------------------------------------------------------

/// Level of ray tracing support available on the device.
///
/// This enum represents the three tiers of RT support:
/// - `None`: No ray tracing support available
/// - `RayQueryOnly`: Inline ray tracing via ray queries in compute/fragment shaders
/// - `Full`: Full ray tracing pipeline with ray generation/miss/closest hit shaders
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum RTCapability {
    /// No ray tracing support available.
    /// All RT effects must use rasterization fallbacks.
    #[default]
    None,

    /// Ray query support only (inline ray tracing).
    /// Supports ray queries in compute/fragment shaders but not
    /// dedicated RT pipeline stages.
    RayQueryOnly,

    /// Full ray tracing pipeline support.
    /// Includes ray generation, miss, closest hit, and any hit shaders.
    /// This is future functionality as wgpu matures.
    Full,
}

impl RTCapability {
    /// Returns `true` if any level of ray tracing is supported.
    pub fn is_supported(&self) -> bool {
        !matches!(self, RTCapability::None)
    }

    /// Returns `true` if ray queries are available (inline RT).
    pub fn has_ray_query(&self) -> bool {
        matches!(self, RTCapability::RayQueryOnly | RTCapability::Full)
    }

    /// Returns `true` if full RT pipeline is available.
    pub fn has_full_pipeline(&self) -> bool {
        matches!(self, RTCapability::Full)
    }

    /// Returns a human-readable description of the capability level.
    pub fn description(&self) -> &'static str {
        match self {
            RTCapability::None => "No ray tracing support",
            RTCapability::RayQueryOnly => "Ray query support (inline RT)",
            RTCapability::Full => "Full ray tracing pipeline",
        }
    }
}

impl std::fmt::Display for RTCapability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.description())
    }
}

// ---------------------------------------------------------------------------
// RTCapabilityInfo
// ---------------------------------------------------------------------------

/// Detailed ray tracing capability information.
///
/// Contains both the capability level and additional hardware parameters
/// that affect RT feature availability and performance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RTCapabilityInfo {
    /// The detected capability level.
    pub capability: RTCapability,

    /// Maximum ray recursion depth supported.
    /// For `RayQueryOnly`, this is typically 1 (no recursion in inline RT).
    /// For `Full`, this can be higher (typically 8-31).
    pub max_ray_recursion_depth: u32,

    /// Whether inline ray tracing is supported.
    /// This allows ray queries within compute/fragment shaders.
    pub supports_inline_ray_tracing: bool,

    /// Required alignment for acceleration structure buffers in bytes.
    pub acceleration_structure_align: u32,
}

impl RTCapabilityInfo {
    /// Create a new capability info with no RT support.
    pub const fn none() -> Self {
        Self {
            capability: RTCapability::None,
            max_ray_recursion_depth: 0,
            supports_inline_ray_tracing: false,
            acceleration_structure_align: 0,
        }
    }

    /// Create a new capability info with ray query support only.
    pub const fn ray_query_only() -> Self {
        Self {
            capability: RTCapability::RayQueryOnly,
            max_ray_recursion_depth: DEFAULT_MAX_RAY_RECURSION_DEPTH,
            supports_inline_ray_tracing: true,
            acceleration_structure_align: DEFAULT_AS_ALIGNMENT,
        }
    }

    /// Create a new capability info with full RT support.
    pub const fn full() -> Self {
        Self {
            capability: RTCapability::Full,
            max_ray_recursion_depth: FULL_RT_MAX_RAY_RECURSION_DEPTH,
            supports_inline_ray_tracing: true,
            acceleration_structure_align: DEFAULT_AS_ALIGNMENT,
        }
    }

    /// Create a custom capability info.
    pub const fn new(
        capability: RTCapability,
        max_ray_recursion_depth: u32,
        supports_inline_ray_tracing: bool,
        acceleration_structure_align: u32,
    ) -> Self {
        Self {
            capability,
            max_ray_recursion_depth,
            supports_inline_ray_tracing,
            acceleration_structure_align,
        }
    }

    /// Returns `true` if ray-traced shadows can be used.
    ///
    /// Requires at least ray query support for shadow rays.
    pub fn supports_shadows(&self) -> bool {
        self.capability.has_ray_query()
    }

    /// Returns `true` if ray-traced reflections can be used.
    ///
    /// Requires ray query support. Full RT pipeline is preferred
    /// for complex multi-bounce reflections.
    pub fn supports_reflections(&self) -> bool {
        self.capability.has_ray_query()
    }

    /// Returns `true` if ray-traced global illumination can be used.
    ///
    /// GI typically requires higher recursion depth for multi-bounce
    /// lighting. Returns `true` if recursion depth >= 2 or full RT.
    pub fn supports_gi(&self) -> bool {
        self.capability.has_full_pipeline() || self.max_ray_recursion_depth >= 2
    }

    /// Returns `true` if any RT features are available.
    pub fn is_supported(&self) -> bool {
        self.capability.is_supported()
    }

    /// Returns the recommended shadow method based on capabilities.
    pub fn recommended_shadow_method(&self) -> ShadowMethod {
        RTEffectRouter::route_shadows(self.capability)
    }

    /// Returns the recommended reflection method based on capabilities.
    pub fn recommended_reflection_method(&self) -> ReflectionMethod {
        RTEffectRouter::route_reflections(self.capability)
    }

    /// Returns the recommended GI method based on capabilities.
    pub fn recommended_gi_method(&self) -> GIMethod {
        RTEffectRouter::route_gi(self.capability)
    }
}

impl Default for RTCapabilityInfo {
    fn default() -> Self {
        Self::none()
    }
}

impl std::fmt::Display for RTCapabilityInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "RTCapabilityInfo {{ capability: {}, max_recursion: {}, inline_rt: {}, as_align: {} }}",
            self.capability,
            self.max_ray_recursion_depth,
            self.supports_inline_ray_tracing,
            self.acceleration_structure_align
        )
    }
}

// ---------------------------------------------------------------------------
// Shadow Method
// ---------------------------------------------------------------------------

/// Shadow rendering method based on RT capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ShadowMethod {
    /// Ray-traced shadows with pixel-perfect accuracy.
    RayTraced,

    /// Cascaded shadow maps for directional lights.
    /// Good quality fallback for outdoor scenes.
    #[default]
    CascadedShadowMaps,

    /// No shadows rendered.
    None,
}

impl ShadowMethod {
    /// Returns `true` if this method uses ray tracing.
    pub fn is_ray_traced(&self) -> bool {
        matches!(self, ShadowMethod::RayTraced)
    }

    /// Returns `true` if this method is a rasterization fallback.
    pub fn is_rasterized(&self) -> bool {
        matches!(self, ShadowMethod::CascadedShadowMaps)
    }

    /// Returns a human-readable description.
    pub fn description(&self) -> &'static str {
        match self {
            ShadowMethod::RayTraced => "Ray-traced shadows",
            ShadowMethod::CascadedShadowMaps => "Cascaded shadow maps",
            ShadowMethod::None => "Shadows disabled",
        }
    }
}

impl std::fmt::Display for ShadowMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.description())
    }
}

// ---------------------------------------------------------------------------
// Reflection Method
// ---------------------------------------------------------------------------

/// Reflection rendering method based on RT capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ReflectionMethod {
    /// Ray-traced reflections with accurate multi-bounce.
    RayTraced,

    /// Screen-space reflections (SSR).
    /// Limited to on-screen geometry but fast.
    #[default]
    ScreenSpace,

    /// Reflection probes / environment maps.
    /// Pre-baked or real-time cubemaps.
    Probes,

    /// No reflections rendered.
    None,
}

impl ReflectionMethod {
    /// Returns `true` if this method uses ray tracing.
    pub fn is_ray_traced(&self) -> bool {
        matches!(self, ReflectionMethod::RayTraced)
    }

    /// Returns `true` if this method uses screen-space techniques.
    pub fn is_screen_space(&self) -> bool {
        matches!(self, ReflectionMethod::ScreenSpace)
    }

    /// Returns `true` if this method uses probes/cubemaps.
    pub fn is_probe_based(&self) -> bool {
        matches!(self, ReflectionMethod::Probes)
    }

    /// Returns a human-readable description.
    pub fn description(&self) -> &'static str {
        match self {
            ReflectionMethod::RayTraced => "Ray-traced reflections",
            ReflectionMethod::ScreenSpace => "Screen-space reflections",
            ReflectionMethod::Probes => "Reflection probes",
            ReflectionMethod::None => "Reflections disabled",
        }
    }
}

impl std::fmt::Display for ReflectionMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.description())
    }
}

// ---------------------------------------------------------------------------
// GI Method
// ---------------------------------------------------------------------------

/// Global illumination method based on RT capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum GIMethod {
    /// Ray-traced global illumination.
    /// Most accurate but computationally expensive.
    RayTraced,

    /// Dynamic Diffuse Global Illumination (DDGI).
    /// Probe-based real-time GI with irradiance volumes.
    #[default]
    DDGI,

    /// Screen-space global illumination (SSGI).
    /// Fast approximation using screen-space data.
    SSGI,

    /// No global illumination.
    None,
}

impl GIMethod {
    /// Returns `true` if this method uses ray tracing.
    pub fn is_ray_traced(&self) -> bool {
        matches!(self, GIMethod::RayTraced)
    }

    /// Returns `true` if this method uses DDGI.
    pub fn is_ddgi(&self) -> bool {
        matches!(self, GIMethod::DDGI)
    }

    /// Returns `true` if this method uses screen-space techniques.
    pub fn is_screen_space(&self) -> bool {
        matches!(self, GIMethod::SSGI)
    }

    /// Returns a human-readable description.
    pub fn description(&self) -> &'static str {
        match self {
            GIMethod::RayTraced => "Ray-traced global illumination",
            GIMethod::DDGI => "Dynamic Diffuse GI (DDGI)",
            GIMethod::SSGI => "Screen-space global illumination",
            GIMethod::None => "Global illumination disabled",
        }
    }
}

impl std::fmt::Display for GIMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.description())
    }
}

// ---------------------------------------------------------------------------
// RTEffectRouter
// ---------------------------------------------------------------------------

/// Routes rendering effects to appropriate implementations based on RT capability.
///
/// This struct provides static methods to determine which rendering technique
/// should be used for various effects based on the device's RT capability level.
pub struct RTEffectRouter;

impl RTEffectRouter {
    /// Determine the shadow rendering method based on capability.
    ///
    /// - `Full` / `RayQueryOnly`: Use ray-traced shadows
    /// - `None`: Fall back to cascaded shadow maps
    pub fn route_shadows(capability: RTCapability) -> ShadowMethod {
        match capability {
            RTCapability::Full | RTCapability::RayQueryOnly => ShadowMethod::RayTraced,
            RTCapability::None => ShadowMethod::CascadedShadowMaps,
        }
    }

    /// Determine the reflection rendering method based on capability.
    ///
    /// - `Full`: Use ray-traced reflections (multi-bounce capable)
    /// - `RayQueryOnly`: Use ray-traced reflections (single bounce)
    /// - `None`: Fall back to screen-space reflections
    pub fn route_reflections(capability: RTCapability) -> ReflectionMethod {
        match capability {
            RTCapability::Full | RTCapability::RayQueryOnly => ReflectionMethod::RayTraced,
            RTCapability::None => ReflectionMethod::ScreenSpace,
        }
    }

    /// Determine the GI rendering method based on capability.
    ///
    /// - `Full`: Use ray-traced GI (most accurate)
    /// - `RayQueryOnly`: Use DDGI (probe-based, good quality)
    /// - `None`: Use SSGI (screen-space approximation)
    pub fn route_gi(capability: RTCapability) -> GIMethod {
        match capability {
            RTCapability::Full => GIMethod::RayTraced,
            RTCapability::RayQueryOnly => GIMethod::DDGI,
            RTCapability::None => GIMethod::SSGI,
        }
    }

    /// Route all effects at once and return a tuple of methods.
    pub fn route_all(
        capability: RTCapability,
    ) -> (ShadowMethod, ReflectionMethod, GIMethod) {
        (
            Self::route_shadows(capability),
            Self::route_reflections(capability),
            Self::route_gi(capability),
        )
    }

    /// Determine the best available capability from a set of features.
    pub fn detect_capability(features: &wgpu::Features) -> RTCapability {
        get_rt_capability(features).capability
    }
}

// ---------------------------------------------------------------------------
// Capability Detection
// ---------------------------------------------------------------------------

/// Detect ray tracing capability from wgpu device features.
///
/// Examines the provided feature flags to determine the level of RT support:
///
/// - `RAY_TRACING_ACCELERATION_STRUCTURE` + `RAY_TRACING_PIPELINE` = `Full`
/// - `RAY_TRACING_ACCELERATION_STRUCTURE` + `RAY_QUERY` = `RayQueryOnly`
/// - Neither = `None`
///
/// # Arguments
///
/// * `device_features` - The wgpu features supported by the device
///
/// # Returns
///
/// An `RTCapabilityInfo` struct with detailed capability information.
///
/// # Example
///
/// ```ignore
/// let adapter_features = adapter.features();
/// let rt_info = get_rt_capability(&adapter_features);
///
/// println!("RT Capability: {}", rt_info.capability);
/// println!("Supports shadows: {}", rt_info.supports_shadows());
/// ```
pub fn get_rt_capability(device_features: &wgpu::Features) -> RTCapabilityInfo {
    // Check for full RT pipeline support
    let has_as = device_features.contains(wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE);
    let has_ray_query = device_features.contains(wgpu::Features::RAY_QUERY);

    // Note: RAY_TRACING_PIPELINE is not yet stable in wgpu, so we check for it
    // but currently fall back to RayQueryOnly even if AS is present.
    // When wgpu stabilizes RT pipeline, uncomment and use this:
    // let has_rt_pipeline = device_features.contains(wgpu::Features::RAY_TRACING_PIPELINE);

    if has_as && has_ray_query {
        // Ray query support with acceleration structures
        RTCapabilityInfo::ray_query_only()
    } else if has_as {
        // Acceleration structure support but no ray query (unusual config)
        // Still report as ray query only since AS alone isn't useful
        RTCapabilityInfo::ray_query_only()
    } else {
        // No RT support
        RTCapabilityInfo::none()
    }
}

/// Check if ray tracing is available without full capability detection.
///
/// A quick check that returns `true` if any RT features are present.
pub fn has_rt_support(device_features: &wgpu::Features) -> bool {
    device_features.contains(wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE)
        || device_features.contains(wgpu::Features::RAY_QUERY)
}

/// Get the minimum required features for a given capability level.
///
/// Returns the wgpu features that must be enabled to achieve the
/// specified capability level.
pub fn required_features_for_capability(capability: RTCapability) -> wgpu::Features {
    match capability {
        RTCapability::None => wgpu::Features::empty(),
        RTCapability::RayQueryOnly => {
            wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE | wgpu::Features::RAY_QUERY
        }
        RTCapability::Full => {
            wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE | wgpu::Features::RAY_QUERY
            // When available: | wgpu::Features::RAY_TRACING_PIPELINE
        }
    }
}

/// Check if the given features meet the requirements for a capability level.
pub fn features_meet_capability(
    device_features: &wgpu::Features,
    required_capability: RTCapability,
) -> bool {
    let required = required_features_for_capability(required_capability);
    device_features.contains(required)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // RTCapability Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_rt_capability_none_default() {
        let cap = RTCapability::default();
        assert_eq!(cap, RTCapability::None);
        assert!(!cap.is_supported());
        assert!(!cap.has_ray_query());
        assert!(!cap.has_full_pipeline());
    }

    #[test]
    fn test_rt_capability_ray_query_only() {
        let cap = RTCapability::RayQueryOnly;
        assert!(cap.is_supported());
        assert!(cap.has_ray_query());
        assert!(!cap.has_full_pipeline());
    }

    #[test]
    fn test_rt_capability_full() {
        let cap = RTCapability::Full;
        assert!(cap.is_supported());
        assert!(cap.has_ray_query());
        assert!(cap.has_full_pipeline());
    }

    #[test]
    fn test_rt_capability_ordering() {
        assert!(RTCapability::None < RTCapability::RayQueryOnly);
        assert!(RTCapability::RayQueryOnly < RTCapability::Full);
        assert!(RTCapability::None < RTCapability::Full);
    }

    #[test]
    fn test_rt_capability_description() {
        assert!(!RTCapability::None.description().is_empty());
        assert!(!RTCapability::RayQueryOnly.description().is_empty());
        assert!(!RTCapability::Full.description().is_empty());
    }

    #[test]
    fn test_rt_capability_display() {
        let none_str = format!("{}", RTCapability::None);
        let rq_str = format!("{}", RTCapability::RayQueryOnly);
        let full_str = format!("{}", RTCapability::Full);

        assert!(!none_str.is_empty());
        assert!(!rq_str.is_empty());
        assert!(!full_str.is_empty());
    }

    // -------------------------------------------------------------------------
    // RTCapabilityInfo Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_rt_capability_info_none() {
        let info = RTCapabilityInfo::none();
        assert_eq!(info.capability, RTCapability::None);
        assert_eq!(info.max_ray_recursion_depth, 0);
        assert!(!info.supports_inline_ray_tracing);
        assert_eq!(info.acceleration_structure_align, 0);
        assert!(!info.is_supported());
    }

    #[test]
    fn test_rt_capability_info_ray_query_only() {
        let info = RTCapabilityInfo::ray_query_only();
        assert_eq!(info.capability, RTCapability::RayQueryOnly);
        assert_eq!(info.max_ray_recursion_depth, DEFAULT_MAX_RAY_RECURSION_DEPTH);
        assert!(info.supports_inline_ray_tracing);
        assert_eq!(info.acceleration_structure_align, DEFAULT_AS_ALIGNMENT);
        assert!(info.is_supported());
    }

    #[test]
    fn test_rt_capability_info_full() {
        let info = RTCapabilityInfo::full();
        assert_eq!(info.capability, RTCapability::Full);
        assert_eq!(info.max_ray_recursion_depth, FULL_RT_MAX_RAY_RECURSION_DEPTH);
        assert!(info.supports_inline_ray_tracing);
        assert_eq!(info.acceleration_structure_align, DEFAULT_AS_ALIGNMENT);
        assert!(info.is_supported());
    }

    #[test]
    fn test_rt_capability_info_custom() {
        let info = RTCapabilityInfo::new(RTCapability::RayQueryOnly, 4, true, 128);
        assert_eq!(info.capability, RTCapability::RayQueryOnly);
        assert_eq!(info.max_ray_recursion_depth, 4);
        assert!(info.supports_inline_ray_tracing);
        assert_eq!(info.acceleration_structure_align, 128);
    }

    #[test]
    fn test_rt_capability_info_supports_shadows() {
        assert!(!RTCapabilityInfo::none().supports_shadows());
        assert!(RTCapabilityInfo::ray_query_only().supports_shadows());
        assert!(RTCapabilityInfo::full().supports_shadows());
    }

    #[test]
    fn test_rt_capability_info_supports_reflections() {
        assert!(!RTCapabilityInfo::none().supports_reflections());
        assert!(RTCapabilityInfo::ray_query_only().supports_reflections());
        assert!(RTCapabilityInfo::full().supports_reflections());
    }

    #[test]
    fn test_rt_capability_info_supports_gi() {
        // None with 0 recursion depth
        assert!(!RTCapabilityInfo::none().supports_gi());

        // RayQueryOnly with depth 1 doesn't support GI
        assert!(!RTCapabilityInfo::ray_query_only().supports_gi());

        // Full always supports GI
        assert!(RTCapabilityInfo::full().supports_gi());

        // Custom with depth >= 2 supports GI
        let custom = RTCapabilityInfo::new(RTCapability::RayQueryOnly, 2, true, 256);
        assert!(custom.supports_gi());
    }

    #[test]
    fn test_rt_capability_info_default() {
        let info = RTCapabilityInfo::default();
        assert_eq!(info.capability, RTCapability::None);
        assert!(!info.is_supported());
    }

    #[test]
    fn test_rt_capability_info_display() {
        let info = RTCapabilityInfo::ray_query_only();
        let display = format!("{}", info);
        assert!(display.contains("RayQueryOnly") || display.contains("Ray query"));
    }

    // -------------------------------------------------------------------------
    // ShadowMethod Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_shadow_method_default() {
        assert_eq!(ShadowMethod::default(), ShadowMethod::CascadedShadowMaps);
    }

    #[test]
    fn test_shadow_method_ray_traced() {
        let method = ShadowMethod::RayTraced;
        assert!(method.is_ray_traced());
        assert!(!method.is_rasterized());
    }

    #[test]
    fn test_shadow_method_csm() {
        let method = ShadowMethod::CascadedShadowMaps;
        assert!(!method.is_ray_traced());
        assert!(method.is_rasterized());
    }

    #[test]
    fn test_shadow_method_none() {
        let method = ShadowMethod::None;
        assert!(!method.is_ray_traced());
        assert!(!method.is_rasterized());
    }

    #[test]
    fn test_shadow_method_description() {
        assert!(!ShadowMethod::RayTraced.description().is_empty());
        assert!(!ShadowMethod::CascadedShadowMaps.description().is_empty());
        assert!(!ShadowMethod::None.description().is_empty());
    }

    // -------------------------------------------------------------------------
    // ReflectionMethod Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_reflection_method_default() {
        assert_eq!(ReflectionMethod::default(), ReflectionMethod::ScreenSpace);
    }

    #[test]
    fn test_reflection_method_ray_traced() {
        let method = ReflectionMethod::RayTraced;
        assert!(method.is_ray_traced());
        assert!(!method.is_screen_space());
        assert!(!method.is_probe_based());
    }

    #[test]
    fn test_reflection_method_screen_space() {
        let method = ReflectionMethod::ScreenSpace;
        assert!(!method.is_ray_traced());
        assert!(method.is_screen_space());
        assert!(!method.is_probe_based());
    }

    #[test]
    fn test_reflection_method_probes() {
        let method = ReflectionMethod::Probes;
        assert!(!method.is_ray_traced());
        assert!(!method.is_screen_space());
        assert!(method.is_probe_based());
    }

    #[test]
    fn test_reflection_method_none() {
        let method = ReflectionMethod::None;
        assert!(!method.is_ray_traced());
        assert!(!method.is_screen_space());
        assert!(!method.is_probe_based());
    }

    // -------------------------------------------------------------------------
    // GIMethod Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_gi_method_default() {
        assert_eq!(GIMethod::default(), GIMethod::DDGI);
    }

    #[test]
    fn test_gi_method_ray_traced() {
        let method = GIMethod::RayTraced;
        assert!(method.is_ray_traced());
        assert!(!method.is_ddgi());
        assert!(!method.is_screen_space());
    }

    #[test]
    fn test_gi_method_ddgi() {
        let method = GIMethod::DDGI;
        assert!(!method.is_ray_traced());
        assert!(method.is_ddgi());
        assert!(!method.is_screen_space());
    }

    #[test]
    fn test_gi_method_ssgi() {
        let method = GIMethod::SSGI;
        assert!(!method.is_ray_traced());
        assert!(!method.is_ddgi());
        assert!(method.is_screen_space());
    }

    #[test]
    fn test_gi_method_none() {
        let method = GIMethod::None;
        assert!(!method.is_ray_traced());
        assert!(!method.is_ddgi());
        assert!(!method.is_screen_space());
    }

    // -------------------------------------------------------------------------
    // RTEffectRouter Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_route_shadows_none() {
        let method = RTEffectRouter::route_shadows(RTCapability::None);
        assert_eq!(method, ShadowMethod::CascadedShadowMaps);
    }

    #[test]
    fn test_route_shadows_ray_query() {
        let method = RTEffectRouter::route_shadows(RTCapability::RayQueryOnly);
        assert_eq!(method, ShadowMethod::RayTraced);
    }

    #[test]
    fn test_route_shadows_full() {
        let method = RTEffectRouter::route_shadows(RTCapability::Full);
        assert_eq!(method, ShadowMethod::RayTraced);
    }

    #[test]
    fn test_route_reflections_none() {
        let method = RTEffectRouter::route_reflections(RTCapability::None);
        assert_eq!(method, ReflectionMethod::ScreenSpace);
    }

    #[test]
    fn test_route_reflections_ray_query() {
        let method = RTEffectRouter::route_reflections(RTCapability::RayQueryOnly);
        assert_eq!(method, ReflectionMethod::RayTraced);
    }

    #[test]
    fn test_route_reflections_full() {
        let method = RTEffectRouter::route_reflections(RTCapability::Full);
        assert_eq!(method, ReflectionMethod::RayTraced);
    }

    #[test]
    fn test_route_gi_none() {
        let method = RTEffectRouter::route_gi(RTCapability::None);
        assert_eq!(method, GIMethod::SSGI);
    }

    #[test]
    fn test_route_gi_ray_query() {
        let method = RTEffectRouter::route_gi(RTCapability::RayQueryOnly);
        assert_eq!(method, GIMethod::DDGI);
    }

    #[test]
    fn test_route_gi_full() {
        let method = RTEffectRouter::route_gi(RTCapability::Full);
        assert_eq!(method, GIMethod::RayTraced);
    }

    #[test]
    fn test_route_all() {
        let (shadow, reflect, gi) = RTEffectRouter::route_all(RTCapability::None);
        assert_eq!(shadow, ShadowMethod::CascadedShadowMaps);
        assert_eq!(reflect, ReflectionMethod::ScreenSpace);
        assert_eq!(gi, GIMethod::SSGI);

        let (shadow, reflect, gi) = RTEffectRouter::route_all(RTCapability::RayQueryOnly);
        assert_eq!(shadow, ShadowMethod::RayTraced);
        assert_eq!(reflect, ReflectionMethod::RayTraced);
        assert_eq!(gi, GIMethod::DDGI);

        let (shadow, reflect, gi) = RTEffectRouter::route_all(RTCapability::Full);
        assert_eq!(shadow, ShadowMethod::RayTraced);
        assert_eq!(reflect, ReflectionMethod::RayTraced);
        assert_eq!(gi, GIMethod::RayTraced);
    }

    // -------------------------------------------------------------------------
    // Capability Detection Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_get_rt_capability_no_features() {
        let features = wgpu::Features::empty();
        let info = get_rt_capability(&features);
        assert_eq!(info.capability, RTCapability::None);
        assert!(!info.is_supported());
    }

    #[test]
    fn test_get_rt_capability_ray_query() {
        let features =
            wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE | wgpu::Features::RAY_QUERY;
        let info = get_rt_capability(&features);
        assert_eq!(info.capability, RTCapability::RayQueryOnly);
        assert!(info.is_supported());
        assert!(info.supports_inline_ray_tracing);
    }

    #[test]
    fn test_get_rt_capability_as_only() {
        // Unusual config: AS without ray query
        let features = wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE;
        let info = get_rt_capability(&features);
        // Should still work since AS alone triggers RayQueryOnly
        assert_eq!(info.capability, RTCapability::RayQueryOnly);
    }

    #[test]
    fn test_has_rt_support() {
        assert!(!has_rt_support(&wgpu::Features::empty()));
        assert!(has_rt_support(
            &wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE
        ));
        assert!(has_rt_support(&wgpu::Features::RAY_QUERY));
        assert!(has_rt_support(
            &(wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE | wgpu::Features::RAY_QUERY)
        ));
    }

    #[test]
    fn test_required_features_for_capability() {
        let none_features = required_features_for_capability(RTCapability::None);
        assert_eq!(none_features, wgpu::Features::empty());

        let rq_features = required_features_for_capability(RTCapability::RayQueryOnly);
        assert!(rq_features.contains(wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE));
        assert!(rq_features.contains(wgpu::Features::RAY_QUERY));

        let full_features = required_features_for_capability(RTCapability::Full);
        assert!(full_features.contains(wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE));
        assert!(full_features.contains(wgpu::Features::RAY_QUERY));
    }

    #[test]
    fn test_features_meet_capability() {
        let empty = wgpu::Features::empty();
        let with_rt =
            wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE | wgpu::Features::RAY_QUERY;

        // Empty features meet None requirement
        assert!(features_meet_capability(&empty, RTCapability::None));

        // Empty features don't meet RayQueryOnly
        assert!(!features_meet_capability(&empty, RTCapability::RayQueryOnly));

        // RT features meet all requirements
        assert!(features_meet_capability(&with_rt, RTCapability::None));
        assert!(features_meet_capability(&with_rt, RTCapability::RayQueryOnly));
        assert!(features_meet_capability(&with_rt, RTCapability::Full));
    }

    // -------------------------------------------------------------------------
    // Integration Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_recommended_methods_none() {
        let info = RTCapabilityInfo::none();
        assert_eq!(info.recommended_shadow_method(), ShadowMethod::CascadedShadowMaps);
        assert_eq!(info.recommended_reflection_method(), ReflectionMethod::ScreenSpace);
        assert_eq!(info.recommended_gi_method(), GIMethod::SSGI);
    }

    #[test]
    fn test_recommended_methods_ray_query() {
        let info = RTCapabilityInfo::ray_query_only();
        assert_eq!(info.recommended_shadow_method(), ShadowMethod::RayTraced);
        assert_eq!(info.recommended_reflection_method(), ReflectionMethod::RayTraced);
        assert_eq!(info.recommended_gi_method(), GIMethod::DDGI);
    }

    #[test]
    fn test_recommended_methods_full() {
        let info = RTCapabilityInfo::full();
        assert_eq!(info.recommended_shadow_method(), ShadowMethod::RayTraced);
        assert_eq!(info.recommended_reflection_method(), ReflectionMethod::RayTraced);
        assert_eq!(info.recommended_gi_method(), GIMethod::RayTraced);
    }

    #[test]
    fn test_detect_capability_via_router() {
        let features = wgpu::Features::empty();
        let cap = RTEffectRouter::detect_capability(&features);
        assert_eq!(cap, RTCapability::None);

        let features =
            wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE | wgpu::Features::RAY_QUERY;
        let cap = RTEffectRouter::detect_capability(&features);
        assert_eq!(cap, RTCapability::RayQueryOnly);
    }

    #[test]
    fn test_effect_methods_display() {
        // Ensure Display implementations work
        let shadow = format!("{}", ShadowMethod::RayTraced);
        let reflect = format!("{}", ReflectionMethod::RayTraced);
        let gi = format!("{}", GIMethod::RayTraced);

        assert!(!shadow.is_empty());
        assert!(!reflect.is_empty());
        assert!(!gi.is_empty());
    }

    #[test]
    fn test_capability_hash() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(RTCapability::None);
        set.insert(RTCapability::RayQueryOnly);
        set.insert(RTCapability::Full);

        assert_eq!(set.len(), 3);
        assert!(set.contains(&RTCapability::None));
        assert!(set.contains(&RTCapability::RayQueryOnly));
        assert!(set.contains(&RTCapability::Full));
    }

    #[test]
    fn test_method_hash() {
        use std::collections::HashSet;

        let mut shadow_set = HashSet::new();
        shadow_set.insert(ShadowMethod::RayTraced);
        shadow_set.insert(ShadowMethod::CascadedShadowMaps);
        shadow_set.insert(ShadowMethod::None);
        assert_eq!(shadow_set.len(), 3);

        let mut reflect_set = HashSet::new();
        reflect_set.insert(ReflectionMethod::RayTraced);
        reflect_set.insert(ReflectionMethod::ScreenSpace);
        reflect_set.insert(ReflectionMethod::Probes);
        reflect_set.insert(ReflectionMethod::None);
        assert_eq!(reflect_set.len(), 4);

        let mut gi_set = HashSet::new();
        gi_set.insert(GIMethod::RayTraced);
        gi_set.insert(GIMethod::DDGI);
        gi_set.insert(GIMethod::SSGI);
        gi_set.insert(GIMethod::None);
        assert_eq!(gi_set.len(), 4);
    }
}
