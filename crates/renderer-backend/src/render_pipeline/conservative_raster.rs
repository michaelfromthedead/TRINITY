//! Conservative rasterization configuration for render pipelines.
//!
//! This module provides conservative rasterization abstractions for wgpu 25.x
//! render pipelines with feature detection, builder patterns, and use case documentation.
//!
//! # Conservative Rasterization Overview
//!
//! Conservative rasterization is a technique that guarantees a fragment is generated
//! for any pixel that is at least partially covered by a primitive, rather than only
//! pixels whose center is covered. This is critical for algorithms that require
//! no-miss coverage guarantees.
//!
//! # Coverage Modes
//!
//! There are two main modes (wgpu uses overestimation by default):
//!
//! | Mode | Description | Fragment Generation |
//! |------|-------------|---------------------|
//! | **Overestimate** | Generates fragment if ANY part of pixel is covered | More fragments, no holes |
//! | **Underestimate** | Generates fragment only if ENTIRE pixel is covered | Fewer fragments, no overlap |
//!
//! wgpu's `conservative: true` enables overestimation mode.
//!
//! # Use Cases
//!
//! | Use Case | Benefit | Example |
//! |----------|---------|---------|
//! | **GPU Voxelization** | Guarantees thin geometry produces voxels | SVO/SDF construction |
//! | **Software Occlusion Culling** | No missed triangles in coverage buffer | Hierarchical Z-buffer |
//! | **Collision Detection** | Complete triangle coverage for accurate tests | GPU-accelerated physics |
//! | **Visibility Buffer** | All visible triangles contribute to V-buffer | Forward+ rendering |
//! | **Shadow Map Precision** | Prevents light leaking through thin geometry | High-quality shadows |
//! | **GPU Ray Tracing Prep** | Complete BVH population | Acceleration structures |
//! | **Pathfinding/Navigation** | Accurate navmesh rasterization | AI movement |
//!
//! # wgpu API Reference
//!
//! Conservative rasterization requires the `CONSERVATIVE_RASTERIZATION` feature:
//!
//! ```ignore
//! // Feature check
//! pub const CONSERVATIVE_RASTERIZATION: Features = Features::CONSERVATIVE_RASTERIZATION;
//!
//! // In PrimitiveState
//! pub struct PrimitiveState {
//!     pub conservative: bool,  // Enable conservative rasterization
//!     // ... other fields
//! }
//! ```
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::render_pipeline::conservative_raster::{
//!     ConservativeRasterization, ConservativeRasterBuilder, is_supported
//! };
//!
//! // Check if feature is supported
//! if is_supported(&adapter) {
//!     // Create conservative rasterization config
//!     let config = ConservativeRasterBuilder::new()
//!         .enable()
//!         .for_voxelization()
//!         .build();
//!
//!     // Apply to primitive state
//!     let primitive = PrimitiveStateDescriptor::new()
//!         .conservative(config.enabled());
//! }
//! ```
//!
//! # Hardware Support
//!
//! | Vendor | Support | Notes |
//! |--------|---------|-------|
//! | NVIDIA | Yes | Full support on Maxwell+ (GTX 900+) |
//! | AMD | Yes | GCN 4+ (RX 400+) |
//! | Intel | Partial | Gen9+ (Skylake+), varies by driver |
//! | Apple | No | Metal does not expose this feature |
//! | WebGPU | Limited | Browser/driver dependent |
//!
//! Always check feature support at runtime before enabling.

use std::fmt;

// ---------------------------------------------------------------------------
// Feature Detection
// ---------------------------------------------------------------------------

/// The wgpu feature flag for conservative rasterization.
///
/// This constant provides easy access to the feature flag for readability.
pub const CONSERVATIVE_RASTERIZATION_FEATURE: wgpu::Features = wgpu::Features::CONSERVATIVE_RASTERIZATION;

/// Check if the adapter supports conservative rasterization.
///
/// # Arguments
///
/// * `adapter` - The wgpu adapter to query
///
/// # Returns
///
/// `true` if conservative rasterization is supported.
///
/// # Example
///
/// ```ignore
/// if conservative_raster::is_supported(&adapter) {
///     println!("Conservative rasterization available!");
/// }
/// ```
pub fn is_supported(adapter: &wgpu::Adapter) -> bool {
    adapter.features().contains(wgpu::Features::CONSERVATIVE_RASTERIZATION)
}

/// Check if the device has conservative rasterization enabled.
///
/// # Arguments
///
/// * `device` - The wgpu device to query
///
/// # Returns
///
/// `true` if the device was created with conservative rasterization enabled.
pub fn is_enabled_on_device(device: &wgpu::Device) -> bool {
    device.features().contains(wgpu::Features::CONSERVATIVE_RASTERIZATION)
}

/// Get the required features for conservative rasterization.
///
/// Use this when creating a device that needs conservative rasterization:
///
/// ```ignore
/// let required = conservative_raster::required_features();
/// let device = adapter.request_device(&wgpu::DeviceDescriptor {
///     required_features: required | other_features,
///     ..Default::default()
/// }, None).await.unwrap();
/// ```
pub fn required_features() -> wgpu::Features {
    wgpu::Features::CONSERVATIVE_RASTERIZATION
}

// ---------------------------------------------------------------------------
// ConservativeRasterization
// ---------------------------------------------------------------------------

/// Configuration for conservative rasterization in render pipelines.
///
/// # Fields
///
/// | Field | Type | Description |
/// |-------|------|-------------|
/// | `enabled` | `bool` | Whether conservative rasterization is active |
/// | `use_case` | `Option<UseCase>` | The intended use case (for documentation/debugging) |
///
/// # Conservative Rasterization Behavior
///
/// When enabled, the GPU rasterizer guarantees that any pixel touched by a
/// primitive (even partially) will generate a fragment. This is critical for:
///
/// - **Voxelization**: Thin triangles still produce voxels
/// - **Coverage queries**: No triangles are missed
/// - **Occlusion culling**: Complete coverage masks
///
/// # Defaults
///
/// Default is disabled (`enabled: false`), as conservative rasterization
/// has performance overhead and requires hardware feature support.
///
/// # Performance Implications
///
/// | Aspect | Impact | Mitigation |
/// |--------|--------|------------|
/// | Fill rate | Higher (more fragments) | Use only where needed |
/// | Memory | More fragment shader invocations | Limit render target size |
/// | Bandwidth | Increased | Batch operations |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConservativeRasterization {
    /// Whether conservative rasterization is enabled.
    enabled: bool,
    /// Optional use case tag for debugging/tooling.
    use_case: Option<UseCase>,
}

/// Predefined use cases for conservative rasterization.
///
/// Tagging configurations with use cases helps with debugging,
/// profiling, and documentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UseCase {
    /// GPU voxelization (SVO, SDF construction).
    Voxelization,
    /// Software occlusion culling preparation.
    OcclusionCulling,
    /// Collision detection acceleration structures.
    CollisionDetection,
    /// Visibility buffer generation.
    VisibilityBuffer,
    /// Shadow map precision improvement.
    ShadowMapping,
    /// GPU ray tracing BVH population.
    RayTracingPrep,
    /// Navigation mesh rasterization.
    Pathfinding,
    /// Custom use case.
    Custom,
}

impl UseCase {
    /// Get the human-readable name for this use case.
    pub fn name(&self) -> &'static str {
        match self {
            UseCase::Voxelization => "Voxelization",
            UseCase::OcclusionCulling => "Occlusion Culling",
            UseCase::CollisionDetection => "Collision Detection",
            UseCase::VisibilityBuffer => "Visibility Buffer",
            UseCase::ShadowMapping => "Shadow Mapping",
            UseCase::RayTracingPrep => "Ray Tracing Prep",
            UseCase::Pathfinding => "Pathfinding",
            UseCase::Custom => "Custom",
        }
    }

    /// Get a description of why conservative rasterization helps this use case.
    pub fn description(&self) -> &'static str {
        match self {
            UseCase::Voxelization => {
                "Guarantees thin geometry produces voxels; prevents holes in SVO/SDF"
            }
            UseCase::OcclusionCulling => {
                "Ensures no triangles are missed in hierarchical Z-buffer construction"
            }
            UseCase::CollisionDetection => {
                "Complete triangle coverage for accurate GPU-accelerated physics tests"
            }
            UseCase::VisibilityBuffer => {
                "All visible triangles contribute to V-buffer for Forward+ rendering"
            }
            UseCase::ShadowMapping => {
                "Prevents light leaking through thin geometry in shadow maps"
            }
            UseCase::RayTracingPrep => {
                "Complete BVH population for acceleration structure construction"
            }
            UseCase::Pathfinding => {
                "Accurate navmesh rasterization for AI movement/pathfinding"
            }
            UseCase::Custom => {
                "User-defined conservative rasterization requirement"
            }
        }
    }
}

impl Default for ConservativeRasterization {
    fn default() -> Self {
        Self {
            enabled: false,
            use_case: None,
        }
    }
}

impl ConservativeRasterization {
    /// Create a new conservative rasterization config with defaults (disabled).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = ConservativeRasterization::new();
    /// assert!(!config.enabled());
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a disabled conservative rasterization config.
    ///
    /// Explicit alternative to `default()` for clarity.
    pub fn disabled() -> Self {
        Self::default()
    }

    /// Create an enabled conservative rasterization config.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = ConservativeRasterization::enabled();
    /// assert!(config.enabled());
    /// ```
    pub fn enabled_config() -> Self {
        Self {
            enabled: true,
            use_case: None,
        }
    }

    /// Create a config for GPU voxelization.
    ///
    /// Voxelization is one of the primary use cases for conservative rasterization.
    /// It ensures thin geometry (like walls viewed edge-on) still produces voxels.
    ///
    /// # Why Conservative Rasterization?
    ///
    /// Standard rasterization only generates fragments for pixels whose centers
    /// are inside the triangle. For thin triangles or those viewed edge-on,
    /// this can result in missing voxels, creating holes in the voxel grid.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let voxel_config = ConservativeRasterization::voxelization();
    /// render_pass.set_pipeline(&voxelization_pipeline);
    /// ```
    pub fn voxelization() -> Self {
        Self {
            enabled: true,
            use_case: Some(UseCase::Voxelization),
        }
    }

    /// Create a config for software occlusion culling.
    ///
    /// When building coverage buffers for hierarchical occlusion culling,
    /// conservative rasterization ensures no triangles are missed.
    pub fn occlusion_culling() -> Self {
        Self {
            enabled: true,
            use_case: Some(UseCase::OcclusionCulling),
        }
    }

    /// Create a config for collision detection.
    ///
    /// GPU-accelerated collision detection requires complete triangle coverage
    /// to avoid missed collisions.
    pub fn collision_detection() -> Self {
        Self {
            enabled: true,
            use_case: Some(UseCase::CollisionDetection),
        }
    }

    /// Create a config for visibility buffer generation.
    ///
    /// Visibility buffers benefit from conservative rasterization to ensure
    /// all visible triangles are captured for deferred rendering.
    pub fn visibility_buffer() -> Self {
        Self {
            enabled: true,
            use_case: Some(UseCase::VisibilityBuffer),
        }
    }

    /// Create a config for shadow map precision.
    ///
    /// Conservative rasterization can improve shadow quality by ensuring
    /// thin geometry casts proper shadows.
    pub fn shadow_mapping() -> Self {
        Self {
            enabled: true,
            use_case: Some(UseCase::ShadowMapping),
        }
    }

    /// Create a config for ray tracing preparation.
    ///
    /// When building acceleration structures on the GPU, conservative
    /// rasterization ensures complete BVH population.
    pub fn ray_tracing_prep() -> Self {
        Self {
            enabled: true,
            use_case: Some(UseCase::RayTracingPrep),
        }
    }

    /// Create a config for pathfinding/navigation.
    ///
    /// Navigation mesh rasterization benefits from conservative rasterization
    /// to ensure accurate obstacle representation.
    pub fn pathfinding() -> Self {
        Self {
            enabled: true,
            use_case: Some(UseCase::Pathfinding),
        }
    }

    /// Check if conservative rasterization is enabled.
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Get the use case, if set.
    pub fn use_case(&self) -> Option<UseCase> {
        self.use_case
    }

    /// Enable conservative rasterization.
    pub fn enable(mut self) -> Self {
        self.enabled = true;
        self
    }

    /// Disable conservative rasterization.
    pub fn disable(mut self) -> Self {
        self.enabled = false;
        self
    }

    /// Set the use case tag.
    pub fn with_use_case(mut self, use_case: UseCase) -> Self {
        self.use_case = Some(use_case);
        self
    }

    /// Clear the use case tag.
    pub fn without_use_case(mut self) -> Self {
        self.use_case = None;
        self
    }

    /// Convert to the boolean flag expected by wgpu's PrimitiveState.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let primitive = wgpu::PrimitiveState {
    ///     conservative: config.as_wgpu_flag(),
    ///     ..Default::default()
    /// };
    /// ```
    pub fn as_wgpu_flag(&self) -> bool {
        self.enabled
    }

    /// Validate the configuration against device capabilities.
    ///
    /// # Returns
    ///
    /// `Ok(())` if the configuration is valid for the device, or an error
    /// describing why it cannot be used.
    pub fn validate(&self, device: &wgpu::Device) -> Result<(), ConservativeRasterError> {
        if self.enabled && !is_enabled_on_device(device) {
            return Err(ConservativeRasterError::FeatureNotEnabled);
        }
        Ok(())
    }

    /// Check if the configuration can be used with the given device.
    pub fn is_valid_for_device(&self, device: &wgpu::Device) -> bool {
        self.validate(device).is_ok()
    }
}

// Thread-safety: ConservativeRasterization contains only Copy types
unsafe impl Send for ConservativeRasterization {}
unsafe impl Sync for ConservativeRasterization {}

impl fmt::Display for ConservativeRasterization {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.enabled {
            match &self.use_case {
                Some(uc) => write!(f, "ConservativeRasterization(enabled, {})", uc.name()),
                None => write!(f, "ConservativeRasterization(enabled)"),
            }
        } else {
            write!(f, "ConservativeRasterization(disabled)")
        }
    }
}

// ---------------------------------------------------------------------------
// ConservativeRasterBuilder
// ---------------------------------------------------------------------------

/// Builder for creating conservative rasterization configurations.
///
/// # Example
///
/// ```ignore
/// let config = ConservativeRasterBuilder::new()
///     .enable()
///     .for_voxelization()
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct ConservativeRasterBuilder {
    config: ConservativeRasterization,
}

impl Default for ConservativeRasterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ConservativeRasterBuilder {
    /// Create a new builder with defaults (disabled).
    pub fn new() -> Self {
        Self {
            config: ConservativeRasterization::default(),
        }
    }

    /// Start building from an existing configuration.
    pub fn from_config(config: ConservativeRasterization) -> Self {
        Self { config }
    }

    /// Enable conservative rasterization.
    pub fn enable(mut self) -> Self {
        self.config.enabled = true;
        self
    }

    /// Disable conservative rasterization.
    pub fn disable(mut self) -> Self {
        self.config.enabled = false;
        self
    }

    /// Set enabled state explicitly.
    pub fn enabled(mut self, enabled: bool) -> Self {
        self.config.enabled = enabled;
        self
    }

    /// Configure for voxelization use case.
    pub fn for_voxelization(mut self) -> Self {
        self.config.use_case = Some(UseCase::Voxelization);
        self
    }

    /// Configure for occlusion culling use case.
    pub fn for_occlusion_culling(mut self) -> Self {
        self.config.use_case = Some(UseCase::OcclusionCulling);
        self
    }

    /// Configure for collision detection use case.
    pub fn for_collision_detection(mut self) -> Self {
        self.config.use_case = Some(UseCase::CollisionDetection);
        self
    }

    /// Configure for visibility buffer use case.
    pub fn for_visibility_buffer(mut self) -> Self {
        self.config.use_case = Some(UseCase::VisibilityBuffer);
        self
    }

    /// Configure for shadow mapping use case.
    pub fn for_shadow_mapping(mut self) -> Self {
        self.config.use_case = Some(UseCase::ShadowMapping);
        self
    }

    /// Configure for ray tracing prep use case.
    pub fn for_ray_tracing_prep(mut self) -> Self {
        self.config.use_case = Some(UseCase::RayTracingPrep);
        self
    }

    /// Configure for pathfinding use case.
    pub fn for_pathfinding(mut self) -> Self {
        self.config.use_case = Some(UseCase::Pathfinding);
        self
    }

    /// Configure for a custom use case.
    pub fn for_custom(mut self) -> Self {
        self.config.use_case = Some(UseCase::Custom);
        self
    }

    /// Set use case explicitly.
    pub fn use_case(mut self, use_case: UseCase) -> Self {
        self.config.use_case = Some(use_case);
        self
    }

    /// Build the configuration.
    pub fn build(self) -> ConservativeRasterization {
        self.config
    }

    /// Build and validate against device capabilities.
    ///
    /// # Returns
    ///
    /// `Ok(ConservativeRasterization)` if valid, or an error if the
    /// device doesn't support the requested configuration.
    pub fn build_validated(
        self,
        device: &wgpu::Device,
    ) -> Result<ConservativeRasterization, ConservativeRasterError> {
        self.config.validate(device)?;
        Ok(self.config)
    }
}

// ---------------------------------------------------------------------------
// ConservativeRasterInfo
// ---------------------------------------------------------------------------

/// Metadata about a conservative rasterization use case.
///
/// Provides descriptive information for tooling, debugging, and documentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConservativeRasterInfo {
    /// Human-readable name for the use case.
    pub name: &'static str,
    /// Description of why conservative rasterization is beneficial.
    pub description: &'static str,
    /// Technical benefits of using conservative rasterization.
    pub benefits: &'static [&'static str],
    /// Performance considerations.
    pub performance_notes: &'static str,
    /// The use case enum value.
    pub use_case: UseCase,
}

/// All documented conservative rasterization use cases.
pub const CONSERVATIVE_RASTER_USE_CASES: [ConservativeRasterInfo; 7] = [
    ConservativeRasterInfo {
        name: "GPU Voxelization",
        description: "Convert triangle meshes to voxel grids on the GPU",
        benefits: &[
            "Guarantees thin geometry produces voxels",
            "Prevents holes in SVO/SDF construction",
            "Accurate volume representation",
        ],
        performance_notes: "Higher fill rate but essential for correctness",
        use_case: UseCase::Voxelization,
    },
    ConservativeRasterInfo {
        name: "Software Occlusion Culling",
        description: "Build coverage buffers for hierarchical occlusion testing",
        benefits: &[
            "No missed triangles in coverage buffer",
            "Accurate hierarchical Z-buffer",
            "Correct culling decisions",
        ],
        performance_notes: "One-time cost per frame, saves on draw calls",
        use_case: UseCase::OcclusionCulling,
    },
    ConservativeRasterInfo {
        name: "Collision Detection",
        description: "GPU-accelerated collision detection and physics",
        benefits: &[
            "Complete triangle coverage",
            "No missed collision events",
            "Accurate broadphase culling",
        ],
        performance_notes: "Critical for physics correctness",
        use_case: UseCase::CollisionDetection,
    },
    ConservativeRasterInfo {
        name: "Visibility Buffer",
        description: "Generate visibility buffers for deferred/Forward+ rendering",
        benefits: &[
            "All visible triangles captured",
            "Correct triangle IDs at all pixels",
            "Proper material sorting",
        ],
        performance_notes: "Slight overhead offset by deferred efficiency gains",
        use_case: UseCase::VisibilityBuffer,
    },
    ConservativeRasterInfo {
        name: "Shadow Mapping",
        description: "Improve shadow map precision for thin geometry",
        benefits: &[
            "Prevents light leaking",
            "Better thin geometry shadows",
            "Reduced shadow acne",
        ],
        performance_notes: "Use selectively for problematic geometry",
        use_case: UseCase::ShadowMapping,
    },
    ConservativeRasterInfo {
        name: "Ray Tracing Preparation",
        description: "Build acceleration structures on the GPU",
        benefits: &[
            "Complete BVH population",
            "No missing triangles in acceleration structure",
            "Correct ray-triangle intersections",
        ],
        performance_notes: "One-time build cost, essential for correctness",
        use_case: UseCase::RayTracingPrep,
    },
    ConservativeRasterInfo {
        name: "Pathfinding/Navigation",
        description: "Rasterize navigation meshes for AI movement",
        benefits: &[
            "Accurate obstacle representation",
            "No walkable area holes",
            "Correct navmesh rasterization",
        ],
        performance_notes: "Usually done offline or at level load",
        use_case: UseCase::Pathfinding,
    },
];

/// Get conservative rasterization info by use case name.
///
/// # Example
///
/// ```ignore
/// if let Some(info) = get_conservative_raster_info("GPU Voxelization") {
///     println!("Benefits: {:?}", info.benefits);
/// }
/// ```
pub fn get_conservative_raster_info(name: &str) -> Option<&'static ConservativeRasterInfo> {
    CONSERVATIVE_RASTER_USE_CASES.iter().find(|info| info.name == name)
}

/// Get conservative rasterization info by use case enum.
pub fn get_info_for_use_case(use_case: UseCase) -> Option<&'static ConservativeRasterInfo> {
    CONSERVATIVE_RASTER_USE_CASES
        .iter()
        .find(|info| info.use_case == use_case)
}

/// List all available use case names.
pub fn use_case_names() -> impl Iterator<Item = &'static str> {
    CONSERVATIVE_RASTER_USE_CASES.iter().map(|info| info.name)
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors related to conservative rasterization configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConservativeRasterError {
    /// Conservative rasterization was requested but the feature is not enabled on the device.
    FeatureNotEnabled,
    /// Conservative rasterization is not supported by the adapter.
    FeatureNotSupported,
}

impl fmt::Display for ConservativeRasterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConservativeRasterError::FeatureNotEnabled => {
                write!(
                    f,
                    "Conservative rasterization requested but CONSERVATIVE_RASTERIZATION \
                     feature is not enabled on the device"
                )
            }
            ConservativeRasterError::FeatureNotSupported => {
                write!(
                    f,
                    "Conservative rasterization is not supported by the adapter"
                )
            }
        }
    }
}

impl std::error::Error for ConservativeRasterError {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // ConservativeRasterization Basic Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_default_is_disabled() {
        let config = ConservativeRasterization::default();
        assert!(!config.enabled());
        assert!(config.use_case().is_none());
    }

    #[test]
    fn test_new_is_disabled() {
        let config = ConservativeRasterization::new();
        assert!(!config.enabled());
    }

    #[test]
    fn test_disabled_constructor() {
        let config = ConservativeRasterization::disabled();
        assert!(!config.enabled());
    }

    #[test]
    fn test_enabled_constructor() {
        let config = ConservativeRasterization::enabled_config();
        assert!(config.enabled());
        assert!(config.use_case().is_none());
    }

    // -------------------------------------------------------------------------
    // Preset Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_voxelization_preset() {
        let config = ConservativeRasterization::voxelization();
        assert!(config.enabled());
        assert_eq!(config.use_case(), Some(UseCase::Voxelization));
    }

    #[test]
    fn test_occlusion_culling_preset() {
        let config = ConservativeRasterization::occlusion_culling();
        assert!(config.enabled());
        assert_eq!(config.use_case(), Some(UseCase::OcclusionCulling));
    }

    #[test]
    fn test_collision_detection_preset() {
        let config = ConservativeRasterization::collision_detection();
        assert!(config.enabled());
        assert_eq!(config.use_case(), Some(UseCase::CollisionDetection));
    }

    #[test]
    fn test_visibility_buffer_preset() {
        let config = ConservativeRasterization::visibility_buffer();
        assert!(config.enabled());
        assert_eq!(config.use_case(), Some(UseCase::VisibilityBuffer));
    }

    #[test]
    fn test_shadow_mapping_preset() {
        let config = ConservativeRasterization::shadow_mapping();
        assert!(config.enabled());
        assert_eq!(config.use_case(), Some(UseCase::ShadowMapping));
    }

    #[test]
    fn test_ray_tracing_prep_preset() {
        let config = ConservativeRasterization::ray_tracing_prep();
        assert!(config.enabled());
        assert_eq!(config.use_case(), Some(UseCase::RayTracingPrep));
    }

    #[test]
    fn test_pathfinding_preset() {
        let config = ConservativeRasterization::pathfinding();
        assert!(config.enabled());
        assert_eq!(config.use_case(), Some(UseCase::Pathfinding));
    }

    // -------------------------------------------------------------------------
    // Fluent API Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_enable_disable_chain() {
        let config = ConservativeRasterization::new()
            .enable()
            .disable()
            .enable();
        assert!(config.enabled());
    }

    #[test]
    fn test_with_use_case() {
        let config = ConservativeRasterization::new()
            .enable()
            .with_use_case(UseCase::Voxelization);

        assert!(config.enabled());
        assert_eq!(config.use_case(), Some(UseCase::Voxelization));
    }

    #[test]
    fn test_without_use_case() {
        let config = ConservativeRasterization::voxelization()
            .without_use_case();

        assert!(config.enabled());
        assert!(config.use_case().is_none());
    }

    // -------------------------------------------------------------------------
    // wgpu Flag Conversion Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_as_wgpu_flag_enabled() {
        let config = ConservativeRasterization::enabled_config();
        assert!(config.as_wgpu_flag());
    }

    #[test]
    fn test_as_wgpu_flag_disabled() {
        let config = ConservativeRasterization::disabled();
        assert!(!config.as_wgpu_flag());
    }

    // -------------------------------------------------------------------------
    // Builder Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_builder_default() {
        let config = ConservativeRasterBuilder::new().build();
        assert!(!config.enabled());
    }

    #[test]
    fn test_builder_enable() {
        let config = ConservativeRasterBuilder::new()
            .enable()
            .build();
        assert!(config.enabled());
    }

    #[test]
    fn test_builder_disable() {
        let config = ConservativeRasterBuilder::new()
            .enable()
            .disable()
            .build();
        assert!(!config.enabled());
    }

    #[test]
    fn test_builder_enabled_explicit() {
        let config = ConservativeRasterBuilder::new()
            .enabled(true)
            .build();
        assert!(config.enabled());

        let config2 = ConservativeRasterBuilder::new()
            .enabled(false)
            .build();
        assert!(!config2.enabled());
    }

    #[test]
    fn test_builder_use_cases() {
        let config = ConservativeRasterBuilder::new()
            .enable()
            .for_voxelization()
            .build();
        assert_eq!(config.use_case(), Some(UseCase::Voxelization));

        let config2 = ConservativeRasterBuilder::new()
            .enable()
            .for_occlusion_culling()
            .build();
        assert_eq!(config2.use_case(), Some(UseCase::OcclusionCulling));

        let config3 = ConservativeRasterBuilder::new()
            .enable()
            .for_collision_detection()
            .build();
        assert_eq!(config3.use_case(), Some(UseCase::CollisionDetection));

        let config4 = ConservativeRasterBuilder::new()
            .enable()
            .for_visibility_buffer()
            .build();
        assert_eq!(config4.use_case(), Some(UseCase::VisibilityBuffer));

        let config5 = ConservativeRasterBuilder::new()
            .enable()
            .for_shadow_mapping()
            .build();
        assert_eq!(config5.use_case(), Some(UseCase::ShadowMapping));

        let config6 = ConservativeRasterBuilder::new()
            .enable()
            .for_ray_tracing_prep()
            .build();
        assert_eq!(config6.use_case(), Some(UseCase::RayTracingPrep));

        let config7 = ConservativeRasterBuilder::new()
            .enable()
            .for_pathfinding()
            .build();
        assert_eq!(config7.use_case(), Some(UseCase::Pathfinding));

        let config8 = ConservativeRasterBuilder::new()
            .enable()
            .for_custom()
            .build();
        assert_eq!(config8.use_case(), Some(UseCase::Custom));
    }

    #[test]
    fn test_builder_use_case_explicit() {
        let config = ConservativeRasterBuilder::new()
            .enable()
            .use_case(UseCase::Voxelization)
            .build();
        assert_eq!(config.use_case(), Some(UseCase::Voxelization));
    }

    #[test]
    fn test_builder_from_config() {
        let original = ConservativeRasterization::voxelization();
        let modified = ConservativeRasterBuilder::from_config(original)
            .disable()
            .build();

        assert!(!modified.enabled());
        assert_eq!(modified.use_case(), Some(UseCase::Voxelization));
    }

    // -------------------------------------------------------------------------
    // UseCase Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_use_case_names() {
        assert_eq!(UseCase::Voxelization.name(), "Voxelization");
        assert_eq!(UseCase::OcclusionCulling.name(), "Occlusion Culling");
        assert_eq!(UseCase::CollisionDetection.name(), "Collision Detection");
        assert_eq!(UseCase::VisibilityBuffer.name(), "Visibility Buffer");
        assert_eq!(UseCase::ShadowMapping.name(), "Shadow Mapping");
        assert_eq!(UseCase::RayTracingPrep.name(), "Ray Tracing Prep");
        assert_eq!(UseCase::Pathfinding.name(), "Pathfinding");
        assert_eq!(UseCase::Custom.name(), "Custom");
    }

    #[test]
    fn test_use_case_descriptions() {
        // Just verify they're non-empty
        assert!(!UseCase::Voxelization.description().is_empty());
        assert!(!UseCase::OcclusionCulling.description().is_empty());
        assert!(!UseCase::CollisionDetection.description().is_empty());
        assert!(!UseCase::VisibilityBuffer.description().is_empty());
        assert!(!UseCase::ShadowMapping.description().is_empty());
        assert!(!UseCase::RayTracingPrep.description().is_empty());
        assert!(!UseCase::Pathfinding.description().is_empty());
        assert!(!UseCase::Custom.description().is_empty());
    }

    // -------------------------------------------------------------------------
    // Info/Preset Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_get_conservative_raster_info() {
        let info = get_conservative_raster_info("GPU Voxelization").unwrap();
        assert_eq!(info.name, "GPU Voxelization");
        assert_eq!(info.use_case, UseCase::Voxelization);
        assert!(!info.benefits.is_empty());
    }

    #[test]
    fn test_get_conservative_raster_info_not_found() {
        assert!(get_conservative_raster_info("NonExistent").is_none());
    }

    #[test]
    fn test_get_info_for_use_case() {
        let info = get_info_for_use_case(UseCase::Voxelization).unwrap();
        assert_eq!(info.name, "GPU Voxelization");
    }

    #[test]
    fn test_use_case_names_iterator() {
        let names: Vec<_> = use_case_names().collect();
        assert_eq!(names.len(), 7);
        assert!(names.contains(&"GPU Voxelization"));
        assert!(names.contains(&"Software Occlusion Culling"));
        assert!(names.contains(&"Collision Detection"));
    }

    #[test]
    fn test_all_use_cases_have_info() {
        for info in &CONSERVATIVE_RASTER_USE_CASES {
            assert!(!info.name.is_empty());
            assert!(!info.description.is_empty());
            assert!(!info.benefits.is_empty());
            assert!(!info.performance_notes.is_empty());
        }
    }

    // -------------------------------------------------------------------------
    // Display Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_display_disabled() {
        let config = ConservativeRasterization::disabled();
        let display = format!("{}", config);
        assert!(display.contains("disabled"));
    }

    #[test]
    fn test_display_enabled_no_use_case() {
        let config = ConservativeRasterization::enabled_config();
        let display = format!("{}", config);
        assert!(display.contains("enabled"));
        assert!(!display.contains("Voxelization"));
    }

    #[test]
    fn test_display_enabled_with_use_case() {
        let config = ConservativeRasterization::voxelization();
        let display = format!("{}", config);
        assert!(display.contains("enabled"));
        assert!(display.contains("Voxelization"));
    }

    // -------------------------------------------------------------------------
    // Error Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_error_display_not_enabled() {
        let err = ConservativeRasterError::FeatureNotEnabled;
        let msg = format!("{}", err);
        assert!(msg.contains("CONSERVATIVE_RASTERIZATION"));
        assert!(msg.contains("not enabled"));
    }

    #[test]
    fn test_error_display_not_supported() {
        let err = ConservativeRasterError::FeatureNotSupported;
        let msg = format!("{}", err);
        assert!(msg.contains("not supported"));
    }

    // -------------------------------------------------------------------------
    // Thread Safety Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<ConservativeRasterization>();
        assert_sync::<ConservativeRasterization>();
        assert_send::<ConservativeRasterBuilder>();
        assert_sync::<ConservativeRasterBuilder>();
    }

    // -------------------------------------------------------------------------
    // Equality Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_equality() {
        let config1 = ConservativeRasterization::voxelization();
        let config2 = ConservativeRasterization::voxelization();
        let config3 = ConservativeRasterization::occlusion_culling();

        assert_eq!(config1, config2);
        assert_ne!(config1, config3);
    }

    #[test]
    fn test_clone() {
        let config = ConservativeRasterization::voxelization();
        let cloned = config.clone();
        assert_eq!(config, cloned);
    }

    #[test]
    fn test_copy() {
        let config = ConservativeRasterization::voxelization();
        let copied = config;
        assert_eq!(config, copied);
    }

    // -------------------------------------------------------------------------
    // Hash Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_hash() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(ConservativeRasterization::voxelization());
        set.insert(ConservativeRasterization::occlusion_culling());

        assert!(set.contains(&ConservativeRasterization::voxelization()));
        assert!(set.contains(&ConservativeRasterization::occlusion_culling()));
        assert!(!set.contains(&ConservativeRasterization::disabled()));
    }

    // -------------------------------------------------------------------------
    // Feature Constant Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_feature_constant() {
        // Verify the constant matches wgpu's feature
        assert_eq!(
            CONSERVATIVE_RASTERIZATION_FEATURE,
            wgpu::Features::CONSERVATIVE_RASTERIZATION
        );
    }

    #[test]
    fn test_required_features() {
        let features = required_features();
        assert!(features.contains(wgpu::Features::CONSERVATIVE_RASTERIZATION));
    }

    // -------------------------------------------------------------------------
    // WHITEBOX: Construction Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_new_equals_default() {
        let new_config = ConservativeRasterization::new();
        let default_config = ConservativeRasterization::default();
        assert_eq!(new_config, default_config);
    }

    #[test]
    fn test_disabled_equals_default() {
        let disabled_config = ConservativeRasterization::disabled();
        let default_config = ConservativeRasterization::default();
        assert_eq!(disabled_config, default_config);
    }

    #[test]
    fn test_enabled_config_differs_from_default() {
        let enabled = ConservativeRasterization::enabled_config();
        let default = ConservativeRasterization::default();
        assert_ne!(enabled, default);
    }

    #[test]
    fn test_all_preset_constructors_are_enabled() {
        let presets = [
            ConservativeRasterization::voxelization(),
            ConservativeRasterization::occlusion_culling(),
            ConservativeRasterization::collision_detection(),
            ConservativeRasterization::visibility_buffer(),
            ConservativeRasterization::shadow_mapping(),
            ConservativeRasterization::ray_tracing_prep(),
            ConservativeRasterization::pathfinding(),
        ];

        for preset in presets {
            assert!(preset.enabled(), "All preset constructors should be enabled");
            assert!(preset.use_case().is_some(), "All presets should have a use case");
        }
    }

    // -------------------------------------------------------------------------
    // WHITEBOX: Builder Pattern Chain Combinations
    // -------------------------------------------------------------------------

    #[test]
    fn test_builder_default_impl() {
        let builder1 = ConservativeRasterBuilder::default();
        let builder2 = ConservativeRasterBuilder::new();
        assert_eq!(builder1.build(), builder2.build());
    }

    #[test]
    fn test_builder_chain_enable_then_use_case() {
        let config = ConservativeRasterBuilder::new()
            .enable()
            .for_voxelization()
            .build();
        assert!(config.enabled());
        assert_eq!(config.use_case(), Some(UseCase::Voxelization));
    }

    #[test]
    fn test_builder_chain_use_case_then_enable() {
        let config = ConservativeRasterBuilder::new()
            .for_voxelization()
            .enable()
            .build();
        assert!(config.enabled());
        assert_eq!(config.use_case(), Some(UseCase::Voxelization));
    }

    #[test]
    fn test_builder_chain_enable_disable_enable() {
        let config = ConservativeRasterBuilder::new()
            .enable()
            .disable()
            .enable()
            .build();
        assert!(config.enabled());
    }

    #[test]
    fn test_builder_override_use_case() {
        let config = ConservativeRasterBuilder::new()
            .for_voxelization()
            .for_shadow_mapping()
            .build();
        assert_eq!(config.use_case(), Some(UseCase::ShadowMapping));
    }

    #[test]
    fn test_builder_all_use_case_methods() {
        let builders_and_expected = [
            (ConservativeRasterBuilder::new().for_voxelization().build(), UseCase::Voxelization),
            (ConservativeRasterBuilder::new().for_occlusion_culling().build(), UseCase::OcclusionCulling),
            (ConservativeRasterBuilder::new().for_collision_detection().build(), UseCase::CollisionDetection),
            (ConservativeRasterBuilder::new().for_visibility_buffer().build(), UseCase::VisibilityBuffer),
            (ConservativeRasterBuilder::new().for_shadow_mapping().build(), UseCase::ShadowMapping),
            (ConservativeRasterBuilder::new().for_ray_tracing_prep().build(), UseCase::RayTracingPrep),
            (ConservativeRasterBuilder::new().for_pathfinding().build(), UseCase::Pathfinding),
            (ConservativeRasterBuilder::new().for_custom().build(), UseCase::Custom),
        ];

        for (config, expected_use_case) in builders_and_expected {
            assert_eq!(config.use_case(), Some(expected_use_case));
        }
    }

    #[test]
    fn test_builder_enabled_true_false_explicit() {
        let true_config = ConservativeRasterBuilder::new().enabled(true).build();
        let false_config = ConservativeRasterBuilder::new().enabled(false).build();

        assert!(true_config.enabled());
        assert!(!false_config.enabled());
    }

    #[test]
    fn test_builder_from_config_preserves_state() {
        let original = ConservativeRasterization::voxelization();
        let rebuilt = ConservativeRasterBuilder::from_config(original).build();
        assert_eq!(original, rebuilt);
    }

    #[test]
    fn test_builder_from_config_allows_modification() {
        let original = ConservativeRasterization::voxelization();
        let modified = ConservativeRasterBuilder::from_config(original)
            .for_shadow_mapping()
            .build();

        assert!(modified.enabled()); // Still enabled
        assert_eq!(modified.use_case(), Some(UseCase::ShadowMapping)); // Changed
    }

    // -------------------------------------------------------------------------
    // WHITEBOX: Fluent API Additional Chains
    // -------------------------------------------------------------------------

    #[test]
    fn test_fluent_enable_then_disable() {
        let config = ConservativeRasterization::new().enable().disable();
        assert!(!config.enabled());
    }

    #[test]
    fn test_fluent_with_use_case_then_without() {
        let config = ConservativeRasterization::new()
            .with_use_case(UseCase::Voxelization)
            .without_use_case();
        assert!(config.use_case().is_none());
    }

    #[test]
    fn test_fluent_all_use_cases() {
        let use_cases = [
            UseCase::Voxelization,
            UseCase::OcclusionCulling,
            UseCase::CollisionDetection,
            UseCase::VisibilityBuffer,
            UseCase::ShadowMapping,
            UseCase::RayTracingPrep,
            UseCase::Pathfinding,
            UseCase::Custom,
        ];

        for uc in use_cases {
            let config = ConservativeRasterization::new().with_use_case(uc);
            assert_eq!(config.use_case(), Some(uc));
        }
    }

    #[test]
    fn test_fluent_multiple_enable_calls() {
        let config = ConservativeRasterization::new()
            .enable()
            .enable()
            .enable();
        assert!(config.enabled());
    }

    #[test]
    fn test_fluent_multiple_disable_calls() {
        let config = ConservativeRasterization::enabled_config()
            .disable()
            .disable()
            .disable();
        assert!(!config.enabled());
    }

    // -------------------------------------------------------------------------
    // WHITEBOX: UseCase Enum Coverage
    // -------------------------------------------------------------------------

    #[test]
    fn test_use_case_all_variants_name_non_empty() {
        let all_use_cases = [
            UseCase::Voxelization,
            UseCase::OcclusionCulling,
            UseCase::CollisionDetection,
            UseCase::VisibilityBuffer,
            UseCase::ShadowMapping,
            UseCase::RayTracingPrep,
            UseCase::Pathfinding,
            UseCase::Custom,
        ];

        for uc in all_use_cases {
            assert!(!uc.name().is_empty(), "UseCase::{:?} should have a name", uc);
            assert!(!uc.description().is_empty(), "UseCase::{:?} should have description", uc);
        }
    }

    #[test]
    fn test_use_case_names_unique() {
        let all_use_cases = [
            UseCase::Voxelization,
            UseCase::OcclusionCulling,
            UseCase::CollisionDetection,
            UseCase::VisibilityBuffer,
            UseCase::ShadowMapping,
            UseCase::RayTracingPrep,
            UseCase::Pathfinding,
            UseCase::Custom,
        ];

        let names: Vec<&str> = all_use_cases.iter().map(|uc| uc.name()).collect();
        let mut unique_names = names.clone();
        unique_names.sort();
        unique_names.dedup();

        assert_eq!(names.len(), unique_names.len(), "All UseCase names should be unique");
    }

    #[test]
    fn test_use_case_descriptions_unique() {
        let all_use_cases = [
            UseCase::Voxelization,
            UseCase::OcclusionCulling,
            UseCase::CollisionDetection,
            UseCase::VisibilityBuffer,
            UseCase::ShadowMapping,
            UseCase::RayTracingPrep,
            UseCase::Pathfinding,
            UseCase::Custom,
        ];

        let descs: Vec<&str> = all_use_cases.iter().map(|uc| uc.description()).collect();
        let mut unique_descs = descs.clone();
        unique_descs.sort();
        unique_descs.dedup();

        assert_eq!(descs.len(), unique_descs.len(), "All UseCase descriptions should be unique");
    }

    #[test]
    fn test_use_case_clone() {
        let uc = UseCase::Voxelization;
        let cloned = uc.clone();
        assert_eq!(uc, cloned);
    }

    #[test]
    fn test_use_case_copy() {
        let uc = UseCase::Voxelization;
        let copied = uc;
        assert_eq!(uc, copied);
    }

    // -------------------------------------------------------------------------
    // WHITEBOX: ConservativeRasterInfo Coverage
    // -------------------------------------------------------------------------

    #[test]
    fn test_info_array_length() {
        assert_eq!(CONSERVATIVE_RASTER_USE_CASES.len(), 7);
    }

    #[test]
    fn test_info_benefits_non_empty() {
        for info in &CONSERVATIVE_RASTER_USE_CASES {
            assert!(!info.benefits.is_empty(), "Info for {} should have benefits", info.name);
            for benefit in info.benefits {
                assert!(!benefit.is_empty(), "Benefits for {} should be non-empty strings", info.name);
            }
        }
    }

    #[test]
    fn test_info_performance_notes_non_empty() {
        for info in &CONSERVATIVE_RASTER_USE_CASES {
            assert!(
                !info.performance_notes.is_empty(),
                "Info for {} should have performance notes",
                info.name
            );
        }
    }

    #[test]
    fn test_info_use_case_matches_name() {
        // Verify use case enum matches the info name semantically
        for info in &CONSERVATIVE_RASTER_USE_CASES {
            let uc_name = info.use_case.name();
            // The info name should contain or relate to the use case name
            assert!(
                info.name.contains(uc_name) || info.description.to_lowercase().contains(&uc_name.to_lowercase()),
                "Info name '{}' should relate to UseCase::{}",
                info.name,
                uc_name
            );
        }
    }

    #[test]
    fn test_get_info_for_all_documented_use_cases() {
        let documented = [
            UseCase::Voxelization,
            UseCase::OcclusionCulling,
            UseCase::CollisionDetection,
            UseCase::VisibilityBuffer,
            UseCase::ShadowMapping,
            UseCase::RayTracingPrep,
            UseCase::Pathfinding,
        ];

        for uc in documented {
            let info = get_info_for_use_case(uc);
            assert!(info.is_some(), "UseCase::{:?} should have info", uc);
        }
    }

    #[test]
    fn test_get_info_for_custom_use_case_not_documented() {
        // Custom is not in CONSERVATIVE_RASTER_USE_CASES
        let info = get_info_for_use_case(UseCase::Custom);
        assert!(info.is_none(), "UseCase::Custom should not have documented info");
    }

    #[test]
    fn test_conservative_raster_info_clone() {
        let info = CONSERVATIVE_RASTER_USE_CASES[0].clone();
        assert_eq!(info, CONSERVATIVE_RASTER_USE_CASES[0]);
    }

    #[test]
    fn test_conservative_raster_info_copy() {
        let info = CONSERVATIVE_RASTER_USE_CASES[0];
        let copied = info;
        assert_eq!(info, copied);
    }

    #[test]
    fn test_use_case_names_iterator_count() {
        let count = use_case_names().count();
        assert_eq!(count, 7);
    }

    #[test]
    fn test_use_case_names_contains_all_documented() {
        let names: Vec<_> = use_case_names().collect();

        assert!(names.contains(&"GPU Voxelization"));
        assert!(names.contains(&"Software Occlusion Culling"));
        assert!(names.contains(&"Collision Detection"));
        assert!(names.contains(&"Visibility Buffer"));
        assert!(names.contains(&"Shadow Mapping"));
        assert!(names.contains(&"Ray Tracing Preparation"));
        assert!(names.contains(&"Pathfinding/Navigation"));
    }

    // -------------------------------------------------------------------------
    // WHITEBOX: wgpu Flag Conversion Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_wgpu_flag_after_enable_disable_chain() {
        let config = ConservativeRasterization::new()
            .enable()
            .disable()
            .enable()
            .disable();
        assert!(!config.as_wgpu_flag());
    }

    #[test]
    fn test_wgpu_flag_from_preset() {
        let presets = [
            ConservativeRasterization::voxelization(),
            ConservativeRasterization::occlusion_culling(),
            ConservativeRasterization::collision_detection(),
            ConservativeRasterization::visibility_buffer(),
            ConservativeRasterization::shadow_mapping(),
            ConservativeRasterization::ray_tracing_prep(),
            ConservativeRasterization::pathfinding(),
        ];

        for preset in presets {
            assert!(preset.as_wgpu_flag(), "All presets should return true for wgpu flag");
        }
    }

    #[test]
    fn test_wgpu_flag_consistency_with_enabled() {
        let configs = [
            ConservativeRasterization::new(),
            ConservativeRasterization::disabled(),
            ConservativeRasterization::enabled_config(),
            ConservativeRasterization::voxelization(),
            ConservativeRasterization::new().enable(),
            ConservativeRasterization::enabled_config().disable(),
        ];

        for config in configs {
            assert_eq!(
                config.enabled(),
                config.as_wgpu_flag(),
                "enabled() should always match as_wgpu_flag()"
            );
        }
    }

    // -------------------------------------------------------------------------
    // WHITEBOX: Display/Debug Trait Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_display_all_use_cases() {
        let use_cases = [
            UseCase::Voxelization,
            UseCase::OcclusionCulling,
            UseCase::CollisionDetection,
            UseCase::VisibilityBuffer,
            UseCase::ShadowMapping,
            UseCase::RayTracingPrep,
            UseCase::Pathfinding,
            UseCase::Custom,
        ];

        for uc in use_cases {
            let config = ConservativeRasterization::new().enable().with_use_case(uc);
            let display = format!("{}", config);
            assert!(display.contains("enabled"), "Display should show enabled");
            assert!(display.contains(uc.name()), "Display should show use case name");
        }
    }

    #[test]
    fn test_debug_format() {
        let config = ConservativeRasterization::voxelization();
        let debug = format!("{:?}", config);
        assert!(debug.contains("ConservativeRasterization"));
        assert!(debug.contains("enabled"));
        assert!(debug.contains("Voxelization"));
    }

    #[test]
    fn test_debug_format_disabled() {
        let config = ConservativeRasterization::disabled();
        let debug = format!("{:?}", config);
        assert!(debug.contains("ConservativeRasterization"));
        assert!(debug.contains("false") || debug.contains("disabled"));
    }

    #[test]
    fn test_builder_debug_format() {
        let builder = ConservativeRasterBuilder::new().enable().for_voxelization();
        let debug = format!("{:?}", builder);
        assert!(debug.contains("ConservativeRasterBuilder"));
    }

    // -------------------------------------------------------------------------
    // WHITEBOX: Error Type Coverage
    // -------------------------------------------------------------------------

    #[test]
    fn test_error_equality() {
        let err1 = ConservativeRasterError::FeatureNotEnabled;
        let err2 = ConservativeRasterError::FeatureNotEnabled;
        let err3 = ConservativeRasterError::FeatureNotSupported;

        assert_eq!(err1, err2);
        assert_ne!(err1, err3);
    }

    #[test]
    fn test_error_clone() {
        let err = ConservativeRasterError::FeatureNotEnabled;
        let cloned = err.clone();
        assert_eq!(err, cloned);
    }

    #[test]
    fn test_error_copy() {
        let err = ConservativeRasterError::FeatureNotEnabled;
        let copied = err;
        assert_eq!(err, copied);
    }

    #[test]
    fn test_error_debug_format() {
        let err = ConservativeRasterError::FeatureNotEnabled;
        let debug = format!("{:?}", err);
        assert!(debug.contains("FeatureNotEnabled"));

        let err2 = ConservativeRasterError::FeatureNotSupported;
        let debug2 = format!("{:?}", err2);
        assert!(debug2.contains("FeatureNotSupported"));
    }

    #[test]
    fn test_error_is_std_error() {
        fn assert_error<T: std::error::Error>() {}
        assert_error::<ConservativeRasterError>();
    }

    #[test]
    fn test_error_display_format_not_enabled() {
        let err = ConservativeRasterError::FeatureNotEnabled;
        let msg = err.to_string();
        assert!(msg.contains("not enabled"));
        assert!(msg.contains("CONSERVATIVE_RASTERIZATION"));
    }

    #[test]
    fn test_error_display_format_not_supported() {
        let err = ConservativeRasterError::FeatureNotSupported;
        let msg = err.to_string();
        assert!(msg.contains("not supported"));
    }

    // -------------------------------------------------------------------------
    // WHITEBOX: Thread Safety Verification
    // -------------------------------------------------------------------------

    #[test]
    fn test_config_send_across_threads() {
        use std::thread;

        let config = ConservativeRasterization::voxelization();
        let handle = thread::spawn(move || {
            assert!(config.enabled());
            config.use_case()
        });

        let result = handle.join().unwrap();
        assert_eq!(result, Some(UseCase::Voxelization));
    }

    #[test]
    fn test_builder_send_across_threads() {
        use std::thread;

        let builder = ConservativeRasterBuilder::new().enable();
        let handle = thread::spawn(move || {
            builder.for_voxelization().build()
        });

        let config = handle.join().unwrap();
        assert!(config.enabled());
    }

    #[test]
    fn test_config_sync_across_threads() {
        use std::sync::Arc;
        use std::thread;

        let config = Arc::new(ConservativeRasterization::voxelization());
        let config_clone = Arc::clone(&config);

        let handle = thread::spawn(move || {
            config_clone.enabled()
        });

        assert!(handle.join().unwrap());
        assert!(config.enabled());
    }

    // -------------------------------------------------------------------------
    // WHITEBOX: Hash Consistency
    // -------------------------------------------------------------------------

    #[test]
    fn test_hash_consistency_same_values() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        fn compute_hash<T: Hash>(t: &T) -> u64 {
            let mut s = DefaultHasher::new();
            t.hash(&mut s);
            s.finish()
        }

        let config1 = ConservativeRasterization::voxelization();
        let config2 = ConservativeRasterization::voxelization();

        assert_eq!(compute_hash(&config1), compute_hash(&config2));
    }

    #[test]
    fn test_hash_differs_for_different_configs() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        fn compute_hash<T: Hash>(t: &T) -> u64 {
            let mut s = DefaultHasher::new();
            t.hash(&mut s);
            s.finish()
        }

        let config1 = ConservativeRasterization::voxelization();
        let config2 = ConservativeRasterization::shadow_mapping();

        assert_ne!(compute_hash(&config1), compute_hash(&config2));
    }

    #[test]
    fn test_hash_set_operations() {
        use std::collections::HashSet;

        let mut set = HashSet::new();

        // Insert all presets
        set.insert(ConservativeRasterization::voxelization());
        set.insert(ConservativeRasterization::occlusion_culling());
        set.insert(ConservativeRasterization::collision_detection());
        set.insert(ConservativeRasterization::visibility_buffer());
        set.insert(ConservativeRasterization::shadow_mapping());
        set.insert(ConservativeRasterization::ray_tracing_prep());
        set.insert(ConservativeRasterization::pathfinding());
        set.insert(ConservativeRasterization::disabled());
        set.insert(ConservativeRasterization::enabled_config());

        // All should be unique
        assert_eq!(set.len(), 9);
    }

    // -------------------------------------------------------------------------
    // WHITEBOX: Equality Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_equality_enabled_vs_disabled() {
        let enabled = ConservativeRasterization::enabled_config();
        let disabled = ConservativeRasterization::disabled();
        assert_ne!(enabled, disabled);
    }

    #[test]
    fn test_equality_same_enabled_different_use_case() {
        let vox = ConservativeRasterization::voxelization();
        let shadow = ConservativeRasterization::shadow_mapping();
        assert_ne!(vox, shadow);
    }

    #[test]
    fn test_equality_same_use_case_different_enabled() {
        let enabled = ConservativeRasterization::voxelization();
        let disabled = ConservativeRasterization::voxelization().disable();
        assert_ne!(enabled, disabled);
    }

    #[test]
    fn test_equality_none_use_case_enabled() {
        let with_none = ConservativeRasterization::enabled_config();
        let with_vox = ConservativeRasterization::voxelization();
        assert_ne!(with_none, with_vox);
    }

    // -------------------------------------------------------------------------
    // WHITEBOX: Feature Constant Verification
    // -------------------------------------------------------------------------

    #[test]
    fn test_feature_constant_not_empty() {
        // Verify the feature is a valid non-empty feature set
        assert!(!CONSERVATIVE_RASTERIZATION_FEATURE.is_empty());
    }

    #[test]
    fn test_required_features_contains_only_conservative() {
        let features = required_features();
        // Should contain conservative rasterization
        assert!(features.contains(wgpu::Features::CONSERVATIVE_RASTERIZATION));
        // And should be exactly that feature
        assert_eq!(features, wgpu::Features::CONSERVATIVE_RASTERIZATION);
    }

    // -------------------------------------------------------------------------
    // WHITEBOX: Builder Clone
    // -------------------------------------------------------------------------

    #[test]
    fn test_builder_clone() {
        let builder1 = ConservativeRasterBuilder::new().enable().for_voxelization();
        let builder2 = builder1.clone();

        let config1 = builder1.build();
        let config2 = builder2.build();

        assert_eq!(config1, config2);
    }

    #[test]
    fn test_builder_clone_independence() {
        let builder1 = ConservativeRasterBuilder::new().enable();
        let builder2 = builder1.clone();

        let config1 = builder1.for_voxelization().build();
        let config2 = builder2.for_shadow_mapping().build();

        assert_ne!(config1, config2);
        assert_eq!(config1.use_case(), Some(UseCase::Voxelization));
        assert_eq!(config2.use_case(), Some(UseCase::ShadowMapping));
    }

    // -------------------------------------------------------------------------
    // WHITEBOX: Info Lookup Functions Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_get_info_by_exact_name_match() {
        let names = [
            "GPU Voxelization",
            "Software Occlusion Culling",
            "Collision Detection",
            "Visibility Buffer",
            "Shadow Mapping",
            "Ray Tracing Preparation",
            "Pathfinding/Navigation",
        ];

        for name in names {
            let info = get_conservative_raster_info(name);
            assert!(info.is_some(), "Should find info for '{}'", name);
            assert_eq!(info.unwrap().name, name);
        }
    }

    #[test]
    fn test_get_info_case_sensitive() {
        // Should not find with wrong case
        assert!(get_conservative_raster_info("gpu voxelization").is_none());
        assert!(get_conservative_raster_info("GPU VOXELIZATION").is_none());
    }

    #[test]
    fn test_get_info_partial_name_not_found() {
        assert!(get_conservative_raster_info("Voxel").is_none());
        assert!(get_conservative_raster_info("GPU").is_none());
    }

    #[test]
    fn test_get_info_empty_string() {
        assert!(get_conservative_raster_info("").is_none());
    }

    #[test]
    fn test_get_info_whitespace() {
        assert!(get_conservative_raster_info(" ").is_none());
        assert!(get_conservative_raster_info("  GPU Voxelization  ").is_none());
    }

    // -------------------------------------------------------------------------
    // WHITEBOX: Use Case Enum Hash
    // -------------------------------------------------------------------------

    #[test]
    fn test_use_case_hash_in_set() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(UseCase::Voxelization);
        set.insert(UseCase::OcclusionCulling);
        set.insert(UseCase::CollisionDetection);
        set.insert(UseCase::VisibilityBuffer);
        set.insert(UseCase::ShadowMapping);
        set.insert(UseCase::RayTracingPrep);
        set.insert(UseCase::Pathfinding);
        set.insert(UseCase::Custom);

        assert_eq!(set.len(), 8);
    }

    #[test]
    fn test_use_case_hash_as_map_key() {
        use std::collections::HashMap;

        let mut map = HashMap::new();
        map.insert(UseCase::Voxelization, "vox");
        map.insert(UseCase::ShadowMapping, "shadow");

        assert_eq!(map.get(&UseCase::Voxelization), Some(&"vox"));
        assert_eq!(map.get(&UseCase::ShadowMapping), Some(&"shadow"));
        assert_eq!(map.get(&UseCase::Custom), None);
    }

    // -------------------------------------------------------------------------
    // WHITEBOX: Configuration State Transitions
    // -------------------------------------------------------------------------

    #[test]
    fn test_state_transition_disabled_to_enabled() {
        let config = ConservativeRasterization::disabled().enable();
        assert!(config.enabled());
    }

    #[test]
    fn test_state_transition_enabled_to_disabled() {
        let config = ConservativeRasterization::enabled_config().disable();
        assert!(!config.enabled());
    }

    #[test]
    fn test_state_transition_add_use_case() {
        let config = ConservativeRasterization::enabled_config()
            .with_use_case(UseCase::Voxelization);
        assert_eq!(config.use_case(), Some(UseCase::Voxelization));
    }

    #[test]
    fn test_state_transition_change_use_case() {
        let config = ConservativeRasterization::voxelization()
            .with_use_case(UseCase::ShadowMapping);
        assert_eq!(config.use_case(), Some(UseCase::ShadowMapping));
    }

    #[test]
    fn test_state_transition_remove_use_case() {
        let config = ConservativeRasterization::voxelization().without_use_case();
        assert!(config.use_case().is_none());
    }

    #[test]
    fn test_complex_state_transition_chain() {
        let config = ConservativeRasterization::new()
            .enable()
            .with_use_case(UseCase::Voxelization)
            .disable()
            .with_use_case(UseCase::ShadowMapping)
            .enable()
            .without_use_case()
            .with_use_case(UseCase::Custom);

        assert!(config.enabled());
        assert_eq!(config.use_case(), Some(UseCase::Custom));
    }
}
