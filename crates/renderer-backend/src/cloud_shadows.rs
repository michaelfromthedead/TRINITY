//! Cloud Shadow System (T-ENV-2.4)
//!
//! Implements volumetric cloud shadows cast onto terrain for TRINITY's atmosphere system.
//! Uses density data from `cloud_raymarching.rs` to generate shadow maps and applies
//! them with soft falloff based on cloud height.
//!
//! # Overview
//!
//! The cloud shadow system provides:
//! - **CloudShadowConfig**: Configuration for shadow quality and behavior
//! - **CloudShadowCascade**: Single cascade of the shadow map
//! - **CloudShadowUniforms**: GPU-uploadable uniform data
//! - **CloudShadowRenderer**: Main API for shadow map generation
//!
//! # Shadow Generation
//!
//! The system generates shadow maps by:
//! 1. Ray marching through cloud volume along light direction
//! 2. Accumulating optical depth to compute transmittance
//! 3. Converting to shadow intensity with soft falloff
//! 4. Multi-cascade blending for large-area coverage
//!
//! # Temporal Stability
//!
//! Uses jittered sampling with TAA-friendly patterns:
//! - Halton sequence for sub-pixel jitter
//! - Temporal reprojection for cascade transitions
//! - Frame-to-frame stability via exponential moving average
//!
//! # References
//!
//! - Schneider & Vos, "The Real-time Volumetric Cloudscapes of Horizon: Zero Dawn"
//! - Hillaire, "A Scalable and Production Ready Sky and Atmosphere Rendering Technique"

use bytemuck::{Pod, Zeroable};

use crate::cloud_raymarching::{
    get_height_fraction, is_inside_cloud_layer, ray_intersect_layer, CloudLayerConfig, EPSILON,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default number of shadow cascades.
pub const DEFAULT_CASCADE_COUNT: u32 = 4;

/// Default shadow map resolution per cascade.
pub const DEFAULT_CASCADE_RESOLUTION: u32 = 1024;

/// Default shadow map coverage in meters for nearest cascade.
pub const DEFAULT_NEAR_COVERAGE: f32 = 500.0;

/// Default shadow map coverage in meters for farthest cascade.
pub const DEFAULT_FAR_COVERAGE: f32 = 50_000.0;

/// Default number of ray march steps for shadow generation.
pub const DEFAULT_SHADOW_STEPS: u32 = 32;

/// Default step size for shadow ray marching in meters.
pub const DEFAULT_SHADOW_STEP_SIZE: f32 = 150.0;

/// Default soft shadow falloff factor.
pub const DEFAULT_SOFT_SHADOW_FACTOR: f32 = 0.5;

/// Default temporal blend factor (lower = more stable, higher = more responsive).
pub const DEFAULT_TEMPORAL_BLEND: f32 = 0.1;

/// Default minimum shadow intensity (prevents pitch black).
pub const DEFAULT_MIN_SHADOW: f32 = 0.2;

/// Default jitter strength for temporal stability.
pub const DEFAULT_JITTER_STRENGTH: f32 = 1.0;

/// Earth radius in meters (for spherical atmosphere).
pub const EARTH_RADIUS: f32 = 6_371_000.0;

/// Pi constant.
pub const PI: f32 = std::f32::consts::PI;

// ---------------------------------------------------------------------------
// CloudShadowQuality - Quality presets
// ---------------------------------------------------------------------------

/// Shadow quality preset affecting resolution and step count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum CloudShadowQuality {
    /// Low quality: 512x512, 16 steps. Mobile/low-end.
    Low = 0,

    /// Medium quality: 1024x1024, 32 steps. Balanced.
    #[default]
    Medium = 1,

    /// High quality: 2048x2048, 64 steps. Desktop.
    High = 2,

    /// Ultra quality: 4096x4096, 128 steps. Maximum quality.
    Ultra = 3,
}

impl CloudShadowQuality {
    /// Get resolution per cascade for this quality level.
    #[inline]
    pub fn resolution(&self) -> u32 {
        match self {
            CloudShadowQuality::Low => 512,
            CloudShadowQuality::Medium => 1024,
            CloudShadowQuality::High => 2048,
            CloudShadowQuality::Ultra => 4096,
        }
    }

    /// Get ray march step count for this quality level.
    #[inline]
    pub fn step_count(&self) -> u32 {
        match self {
            CloudShadowQuality::Low => 16,
            CloudShadowQuality::Medium => 32,
            CloudShadowQuality::High => 64,
            CloudShadowQuality::Ultra => 128,
        }
    }

    /// Get quality from resolution (rounds down).
    #[inline]
    pub fn from_resolution(res: u32) -> Self {
        if res >= 4096 {
            CloudShadowQuality::Ultra
        } else if res >= 2048 {
            CloudShadowQuality::High
        } else if res >= 1024 {
            CloudShadowQuality::Medium
        } else {
            CloudShadowQuality::Low
        }
    }
}

// ---------------------------------------------------------------------------
// CloudShadowConfig - Shadow system configuration
// ---------------------------------------------------------------------------

/// Configuration for the cloud shadow system.
///
/// Controls cascade count, coverage, and quality settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CloudShadowConfig {
    /// Number of shadow cascades (1-4).
    pub cascade_count: u32,

    /// Resolution per cascade in texels.
    pub cascade_resolution: u32,

    /// Coverage distance in meters per cascade.
    /// Index 0 = nearest, last = farthest.
    pub cascade_coverage: [f32; 4],

    /// Number of ray march steps for shadow generation.
    pub shadow_steps: u32,

    /// Step size for ray marching in meters.
    pub shadow_step_size: f32,

    /// Soft shadow falloff factor (0 = hard, 1 = very soft).
    pub soft_shadow_factor: f32,

    /// Temporal blend factor for stability.
    pub temporal_blend: f32,

    /// Minimum shadow intensity (prevents pitch black).
    pub min_shadow: f32,

    /// Enable temporal jitter for TAA compatibility.
    pub enable_temporal_jitter: bool,

    /// Jitter strength for temporal stability.
    pub jitter_strength: f32,
}

impl Default for CloudShadowConfig {
    fn default() -> Self {
        Self::from_quality(CloudShadowQuality::Medium)
    }
}

impl CloudShadowConfig {
    /// Create configuration from a quality preset.
    pub fn from_quality(quality: CloudShadowQuality) -> Self {
        let coverage_base = DEFAULT_NEAR_COVERAGE;
        let coverage_far = DEFAULT_FAR_COVERAGE;

        // Exponential cascade distribution
        let ratio = (coverage_far / coverage_base).powf(1.0 / 3.0);

        Self {
            cascade_count: DEFAULT_CASCADE_COUNT,
            cascade_resolution: quality.resolution(),
            cascade_coverage: [
                coverage_base,
                coverage_base * ratio,
                coverage_base * ratio * ratio,
                coverage_far,
            ],
            shadow_steps: quality.step_count(),
            shadow_step_size: DEFAULT_SHADOW_STEP_SIZE,
            soft_shadow_factor: DEFAULT_SOFT_SHADOW_FACTOR,
            temporal_blend: DEFAULT_TEMPORAL_BLEND,
            min_shadow: DEFAULT_MIN_SHADOW,
            enable_temporal_jitter: true,
            jitter_strength: DEFAULT_JITTER_STRENGTH,
        }
    }

    /// Create a low quality configuration.
    #[inline]
    pub fn low() -> Self {
        Self::from_quality(CloudShadowQuality::Low)
    }

    /// Create a medium quality configuration.
    #[inline]
    pub fn medium() -> Self {
        Self::from_quality(CloudShadowQuality::Medium)
    }

    /// Create a high quality configuration.
    #[inline]
    pub fn high() -> Self {
        Self::from_quality(CloudShadowQuality::High)
    }

    /// Create an ultra quality configuration.
    #[inline]
    pub fn ultra() -> Self {
        Self::from_quality(CloudShadowQuality::Ultra)
    }

    /// Set cascade count (clamped to 1-4).
    #[inline]
    pub fn with_cascade_count(mut self, count: u32) -> Self {
        self.cascade_count = count.clamp(1, 4);
        self
    }

    /// Set cascade resolution.
    #[inline]
    pub fn with_resolution(mut self, resolution: u32) -> Self {
        self.cascade_resolution = resolution.max(64);
        self
    }

    /// Set shadow steps.
    #[inline]
    pub fn with_shadow_steps(mut self, steps: u32) -> Self {
        self.shadow_steps = steps.max(4);
        self
    }

    /// Set soft shadow factor.
    #[inline]
    pub fn with_soft_shadow(mut self, factor: f32) -> Self {
        self.soft_shadow_factor = factor.clamp(0.0, 1.0);
        self
    }

    /// Set temporal blend factor.
    #[inline]
    pub fn with_temporal_blend(mut self, blend: f32) -> Self {
        self.temporal_blend = blend.clamp(0.0, 1.0);
        self
    }

    /// Set minimum shadow intensity.
    #[inline]
    pub fn with_min_shadow(mut self, min: f32) -> Self {
        self.min_shadow = min.clamp(0.0, 1.0);
        self
    }

    /// Enable or disable temporal jitter.
    #[inline]
    pub fn with_temporal_jitter(mut self, enabled: bool) -> Self {
        self.enable_temporal_jitter = enabled;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> bool {
        self.cascade_count >= 1
            && self.cascade_count <= 4
            && self.cascade_resolution >= 64
            && self.shadow_steps >= 4
            && self.soft_shadow_factor >= 0.0
            && self.soft_shadow_factor <= 1.0
            && self.temporal_blend >= 0.0
            && self.temporal_blend <= 1.0
            && self.min_shadow >= 0.0
            && self.min_shadow <= 1.0
            && self.cascade_coverage
                .iter()
                .take(self.cascade_count as usize)
                .all(|&c| c > 0.0)
    }

    /// Get total texels for all cascades.
    pub fn total_texels(&self) -> u64 {
        let res = self.cascade_resolution as u64;
        res * res * self.cascade_count as u64
    }
}

// ---------------------------------------------------------------------------
// CloudShadowCascade - Single cascade data
// ---------------------------------------------------------------------------

/// Data for a single shadow map cascade.
///
/// Contains the light-space matrices and coverage bounds.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct CloudShadowCascade {
    /// Light view-projection matrix (column-major).
    pub light_view_proj: [[f32; 4]; 4],

    /// Inverse light view-projection matrix.
    pub light_view_proj_inv: [[f32; 4]; 4],

    /// World-space coverage bounds (min_x, min_z, max_x, max_z).
    pub coverage_bounds: [f32; 4],

    /// Coverage distance in meters.
    pub coverage_distance: f32,

    /// Cascade index.
    pub cascade_index: u32,

    /// Texel size in world units.
    pub texel_size: f32,

    /// Padding for alignment.
    pub _padding: f32,
}

// Size assertion: 64 + 64 + 16 + 4 + 4 + 4 + 4 = 160 bytes
const _: () = assert!(std::mem::size_of::<CloudShadowCascade>() == 160);

impl Default for CloudShadowCascade {
    fn default() -> Self {
        Self {
            light_view_proj: identity_matrix(),
            light_view_proj_inv: identity_matrix(),
            coverage_bounds: [0.0; 4],
            coverage_distance: 500.0,
            cascade_index: 0,
            texel_size: 1.0,
            _padding: 0.0,
        }
    }
}

impl CloudShadowCascade {
    /// Create a new cascade with given parameters.
    pub fn new(cascade_index: u32, coverage_distance: f32, resolution: u32) -> Self {
        let texel_size = (coverage_distance * 2.0) / resolution as f32;
        Self {
            light_view_proj: identity_matrix(),
            light_view_proj_inv: identity_matrix(),
            coverage_bounds: [
                -coverage_distance,
                -coverage_distance,
                coverage_distance,
                coverage_distance,
            ],
            coverage_distance,
            cascade_index,
            texel_size,
            _padding: 0.0,
        }
    }

    /// Update matrices from light direction and camera position.
    pub fn update_matrices(
        &mut self,
        light_dir: [f32; 3],
        camera_pos: [f32; 3],
        cloud_layer: &CloudLayerConfig,
    ) {
        let light_dir = normalize_vec3(light_dir);

        // Calculate center point (camera XZ, cloud layer center Y)
        let layer_center_y = (cloud_layer.min_height + cloud_layer.max_height) * 0.5;
        let center = [camera_pos[0], layer_center_y, camera_pos[2]];

        // Light position far along light direction
        let light_distance = cloud_layer.max_height * 2.0;
        let light_pos = [
            center[0] - light_dir[0] * light_distance,
            center[1] - light_dir[1] * light_distance,
            center[2] - light_dir[2] * light_distance,
        ];

        // Build view matrix (look-at)
        let view = look_at_matrix(light_pos, center, [0.0, 1.0, 0.0]);

        // Build orthographic projection
        let half_size = self.coverage_distance;
        let near = 0.1;
        let far = light_distance * 2.0 + cloud_layer.thickness();
        let proj = ortho_matrix(-half_size, half_size, -half_size, half_size, near, far);

        // Combine matrices
        self.light_view_proj = mat4_mul(proj, view);
        self.light_view_proj_inv = mat4_inverse(self.light_view_proj);

        // Update coverage bounds
        self.coverage_bounds = [
            camera_pos[0] - half_size,
            camera_pos[2] - half_size,
            camera_pos[0] + half_size,
            camera_pos[2] + half_size,
        ];
    }

    /// Project a world position to shadow map UV coordinates.
    #[inline]
    pub fn project_to_uv(&self, world_pos: [f32; 3]) -> [f32; 2] {
        let clip = mat4_transform_point(self.light_view_proj, world_pos);
        // NDC to UV: [-1,1] -> [0,1]
        [(clip[0] + 1.0) * 0.5, (clip[1] + 1.0) * 0.5]
    }

    /// Check if a world position is within cascade coverage.
    #[inline]
    pub fn contains(&self, world_pos: [f32; 3]) -> bool {
        world_pos[0] >= self.coverage_bounds[0]
            && world_pos[0] <= self.coverage_bounds[2]
            && world_pos[2] >= self.coverage_bounds[1]
            && world_pos[2] <= self.coverage_bounds[3]
    }

    /// Get blend factor for cascade transition (1.0 = fully this cascade).
    pub fn get_blend_factor(&self, world_pos: [f32; 3]) -> f32 {
        let center_x = (self.coverage_bounds[0] + self.coverage_bounds[2]) * 0.5;
        let center_z = (self.coverage_bounds[1] + self.coverage_bounds[3]) * 0.5;

        let dist_x = (world_pos[0] - center_x).abs();
        let dist_z = (world_pos[2] - center_z).abs();
        let max_dist = dist_x.max(dist_z);

        let half_size = self.coverage_distance;
        let blend_start = half_size * 0.8;

        if max_dist < blend_start {
            1.0
        } else {
            let t = (max_dist - blend_start) / (half_size - blend_start);
            1.0 - t.clamp(0.0, 1.0)
        }
    }
}

// ---------------------------------------------------------------------------
// CloudShadowUniforms - GPU uniform data
// ---------------------------------------------------------------------------

/// GPU uniform struct for cloud shadow rendering.
///
/// Contains all parameters needed for shadow map generation and sampling.
///
/// # Memory Layout (128 bytes)
///
/// | Offset | Field               | Size     |
/// |--------|---------------------|----------|
/// | 0      | light_direction     | 12 bytes |
/// | 12     | shadow_steps        | 4 bytes  |
/// | 16     | step_size           | 4 bytes  |
/// | 20     | soft_factor         | 4 bytes  |
/// | 24     | min_shadow          | 4 bytes  |
/// | 28     | jitter_strength     | 4 bytes  |
/// | 32     | frame_index         | 4 bytes  |
/// | 36     | cascade_count       | 4 bytes  |
/// | 40     | cascade_distances   | 16 bytes |
/// | 56     | cloud_min_height    | 4 bytes  |
/// | 60     | cloud_max_height    | 4 bytes  |
/// | 64     | cloud_density       | 4 bytes  |
/// | 68     | cloud_extinction    | 4 bytes  |
/// | 72     | temporal_blend      | 4 bytes  |
/// | 76     | jitter_offset       | 8 bytes  |
/// | 84     | _padding            | 44 bytes |
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct CloudShadowUniforms {
    /// Light direction (normalized, towards sun).
    pub light_direction: [f32; 3],

    /// Number of ray march steps.
    pub shadow_steps: u32,

    /// Step size in meters.
    pub step_size: f32,

    /// Soft shadow falloff factor.
    pub soft_factor: f32,

    /// Minimum shadow intensity.
    pub min_shadow: f32,

    /// Jitter strength for temporal stability.
    pub jitter_strength: f32,

    /// Current frame index for temporal jitter.
    pub frame_index: u32,

    /// Number of active cascades.
    pub cascade_count: u32,

    /// Coverage distances per cascade.
    pub cascade_distances: [f32; 4],

    /// Cloud layer minimum height.
    pub cloud_min_height: f32,

    /// Cloud layer maximum height.
    pub cloud_max_height: f32,

    /// Cloud density factor.
    pub cloud_density: f32,

    /// Cloud extinction coefficient.
    pub cloud_extinction: f32,

    /// Temporal blend factor.
    pub temporal_blend: f32,

    /// Jitter offset for current frame (x, y).
    pub jitter_offset: [f32; 2],

    /// Padding for 16-byte alignment.
    pub _padding: [u32; 11],
}

// Size assertion
const _: () = assert!(std::mem::size_of::<CloudShadowUniforms>() == 128);

impl Default for CloudShadowUniforms {
    fn default() -> Self {
        Self {
            light_direction: normalize_vec3([0.5, 0.7, -0.5]),
            shadow_steps: DEFAULT_SHADOW_STEPS,
            step_size: DEFAULT_SHADOW_STEP_SIZE,
            soft_factor: DEFAULT_SOFT_SHADOW_FACTOR,
            min_shadow: DEFAULT_MIN_SHADOW,
            jitter_strength: DEFAULT_JITTER_STRENGTH,
            frame_index: 0,
            cascade_count: DEFAULT_CASCADE_COUNT,
            cascade_distances: [500.0, 2500.0, 10000.0, 50000.0],
            cloud_min_height: 1500.0,
            cloud_max_height: 8000.0,
            cloud_density: 0.03,
            cloud_extinction: 0.1,
            temporal_blend: DEFAULT_TEMPORAL_BLEND,
            jitter_offset: [0.0, 0.0],
            _padding: [0; 11],
        }
    }
}

impl CloudShadowUniforms {
    /// Create uniforms from configuration and cloud layer.
    pub fn from_config(config: &CloudShadowConfig, layer: &CloudLayerConfig) -> Self {
        Self {
            light_direction: normalize_vec3([0.5, 0.7, -0.5]),
            shadow_steps: config.shadow_steps,
            step_size: config.shadow_step_size,
            soft_factor: config.soft_shadow_factor,
            min_shadow: config.min_shadow,
            jitter_strength: config.jitter_strength,
            frame_index: 0,
            cascade_count: config.cascade_count,
            cascade_distances: config.cascade_coverage,
            cloud_min_height: layer.min_height,
            cloud_max_height: layer.max_height,
            cloud_density: layer.density,
            cloud_extinction: layer.extinction_coeff,
            temporal_blend: config.temporal_blend,
            jitter_offset: [0.0, 0.0],
            _padding: [0; 11],
        }
    }

    /// Update light direction.
    #[inline]
    pub fn set_light_direction(&mut self, dir: [f32; 3]) {
        self.light_direction = normalize_vec3(dir);
    }

    /// Update frame index and jitter offset.
    pub fn advance_frame(&mut self) {
        self.frame_index = self.frame_index.wrapping_add(1);
        self.jitter_offset = halton_2d(self.frame_index, 2, 3);
    }

    /// Get the cloud layer thickness.
    #[inline]
    pub fn cloud_thickness(&self) -> f32 {
        self.cloud_max_height - self.cloud_min_height
    }
}

// ---------------------------------------------------------------------------
// CloudShadowSample - Shadow sampling result
// ---------------------------------------------------------------------------

/// Result of sampling cloud shadow at a world position.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CloudShadowSample {
    /// Shadow intensity (0 = full shadow, 1 = no shadow).
    pub shadow_factor: f32,

    /// Accumulated optical depth along light ray.
    pub optical_depth: f32,

    /// Distance to first cloud hit along light ray.
    pub cloud_distance: f32,

    /// Height fraction where shadow originates.
    pub shadow_height: f32,
}

impl CloudShadowSample {
    /// Create a sample with no shadow.
    #[inline]
    pub fn no_shadow() -> Self {
        Self {
            shadow_factor: 1.0,
            optical_depth: 0.0,
            cloud_distance: f32::MAX,
            shadow_height: 0.0,
        }
    }

    /// Create a sample with full shadow.
    #[inline]
    pub fn full_shadow() -> Self {
        Self {
            shadow_factor: 0.0,
            optical_depth: f32::MAX,
            cloud_distance: 0.0,
            shadow_height: 0.5,
        }
    }

    /// Check if this position is in shadow.
    #[inline]
    pub fn is_shadowed(&self, threshold: f32) -> bool {
        self.shadow_factor < threshold
    }

    /// Apply minimum shadow intensity.
    #[inline]
    pub fn clamp_to_min(&mut self, min_shadow: f32) {
        self.shadow_factor = self.shadow_factor.max(min_shadow);
    }

    /// Apply soft falloff based on cloud height.
    pub fn apply_height_falloff(&mut self, falloff_factor: f32) {
        // Higher clouds cast softer shadows
        let height_softness = 1.0 - (1.0 - self.shadow_height).powi(2);
        let softened = self.shadow_factor + (1.0 - self.shadow_factor) * height_softness * falloff_factor;
        self.shadow_factor = softened.clamp(0.0, 1.0);
    }
}

// ---------------------------------------------------------------------------
// CloudShadowRenderer - Main shadow rendering API
// ---------------------------------------------------------------------------

/// Main cloud shadow renderer.
///
/// Provides the API for generating and sampling cloud shadows.
///
/// # Example
///
/// ```
/// use renderer_backend::cloud_shadows::{
///     CloudShadowRenderer, CloudShadowConfig, CloudShadowQuality
/// };
/// use renderer_backend::cloud_raymarching::CloudLayerConfig;
///
/// let config = CloudShadowConfig::from_quality(CloudShadowQuality::High);
/// let layer = CloudLayerConfig::default();
/// let mut renderer = CloudShadowRenderer::new(config, layer);
///
/// // Update light direction
/// renderer.set_light_direction([0.5, 0.8, -0.3]);
///
/// // Sample shadow at a terrain position
/// let sample = renderer.sample_shadow([1000.0, 100.0, 2000.0]);
/// println!("Shadow factor: {}", sample.shadow_factor);
/// ```
#[derive(Debug, Clone)]
pub struct CloudShadowRenderer {
    /// Shadow configuration.
    config: CloudShadowConfig,

    /// Cloud layer configuration.
    cloud_layer: CloudLayerConfig,

    /// Light direction (normalized, towards sun).
    light_direction: [f32; 3],

    /// Shadow cascades.
    cascades: [CloudShadowCascade; 4],

    /// GPU uniforms.
    uniforms: CloudShadowUniforms,

    /// Current frame index.
    frame_index: u32,

    /// Camera position for cascade updates.
    camera_pos: [f32; 3],
}

impl CloudShadowRenderer {
    /// Create a new shadow renderer with given configuration.
    pub fn new(config: CloudShadowConfig, cloud_layer: CloudLayerConfig) -> Self {
        let mut renderer = Self {
            config,
            cloud_layer,
            light_direction: normalize_vec3([0.5, 0.7, -0.5]),
            cascades: [CloudShadowCascade::default(); 4],
            uniforms: CloudShadowUniforms::from_config(&config, &cloud_layer),
            frame_index: 0,
            camera_pos: [0.0; 3],
        };

        // Initialize cascades
        for i in 0..config.cascade_count as usize {
            renderer.cascades[i] = CloudShadowCascade::new(
                i as u32,
                config.cascade_coverage[i],
                config.cascade_resolution,
            );
        }

        renderer
    }

    /// Get the shadow configuration.
    #[inline]
    pub fn config(&self) -> &CloudShadowConfig {
        &self.config
    }

    /// Get the cloud layer configuration.
    #[inline]
    pub fn cloud_layer(&self) -> &CloudLayerConfig {
        &self.cloud_layer
    }

    /// Get the current GPU uniforms.
    #[inline]
    pub fn uniforms(&self) -> &CloudShadowUniforms {
        &self.uniforms
    }

    /// Get a specific cascade.
    #[inline]
    pub fn cascade(&self, index: usize) -> Option<&CloudShadowCascade> {
        if index < self.config.cascade_count as usize {
            Some(&self.cascades[index])
        } else {
            None
        }
    }

    /// Get all active cascades.
    pub fn cascades(&self) -> &[CloudShadowCascade] {
        &self.cascades[..self.config.cascade_count as usize]
    }

    /// Set light direction (will be normalized).
    pub fn set_light_direction(&mut self, direction: [f32; 3]) {
        self.light_direction = normalize_vec3(direction);
        self.uniforms.light_direction = self.light_direction;
    }

    /// Set camera position for cascade updates.
    pub fn set_camera_position(&mut self, pos: [f32; 3]) {
        self.camera_pos = pos;
    }

    /// Update cascade matrices for current camera position.
    pub fn update_cascades(&mut self) {
        for i in 0..self.config.cascade_count as usize {
            self.cascades[i].update_matrices(
                self.light_direction,
                self.camera_pos,
                &self.cloud_layer,
            );
        }
    }

    /// Advance to next frame (updates jitter).
    pub fn advance_frame(&mut self) {
        self.frame_index = self.frame_index.wrapping_add(1);
        self.uniforms.advance_frame();
    }

    /// Get temporal jitter offset for current frame.
    pub fn get_jitter_offset(&self) -> [f32; 2] {
        if self.config.enable_temporal_jitter {
            let jitter = halton_2d(self.frame_index, 2, 3);
            [
                jitter[0] * self.config.jitter_strength,
                jitter[1] * self.config.jitter_strength,
            ]
        } else {
            [0.0, 0.0]
        }
    }

    /// Sample cloud shadow at a world position.
    ///
    /// Ray marches from the position towards the sun through the cloud layer.
    pub fn sample_shadow(&self, world_pos: [f32; 3]) -> CloudShadowSample {
        // Check if position is above cloud layer (no shadow possible)
        if world_pos[1] >= self.cloud_layer.max_height {
            return CloudShadowSample::no_shadow();
        }

        // Ray march towards light
        self.ray_march_shadow(world_pos, self.light_direction)
    }

    /// Sample shadow with temporal jitter applied.
    pub fn sample_shadow_jittered(&self, world_pos: [f32; 3]) -> CloudShadowSample {
        let jitter = self.get_jitter_offset();

        // Apply jitter to horizontal position
        let jittered_pos = [
            world_pos[0] + jitter[0] * self.cascades[0].texel_size,
            world_pos[1],
            world_pos[2] + jitter[1] * self.cascades[0].texel_size,
        ];

        self.sample_shadow(jittered_pos)
    }

    /// Ray march through cloud layer to compute shadow.
    fn ray_march_shadow(&self, start_pos: [f32; 3], light_dir: [f32; 3]) -> CloudShadowSample {
        // Find intersection with cloud layer
        let intersection = ray_intersect_layer(start_pos, light_dir, &self.cloud_layer);

        if intersection.is_none() {
            return CloudShadowSample::no_shadow();
        }

        let (t_entry, t_exit) = intersection.unwrap();

        // If entry is behind us, no shadow
        if t_exit < 0.0 {
            return CloudShadowSample::no_shadow();
        }

        let t_entry = t_entry.max(0.0);
        let march_distance = t_exit - t_entry;

        if march_distance < EPSILON {
            return CloudShadowSample::no_shadow();
        }

        // Calculate step size
        let step_count = self.config.shadow_steps;
        let step_size = (march_distance / step_count as f32).min(self.config.shadow_step_size);

        // Initialize accumulator
        let mut optical_depth = 0.0f32;
        let mut first_hit_distance = f32::MAX;
        let mut first_hit_height = 0.0f32;

        // Ray march loop
        for i in 0..step_count {
            let t = t_entry + (i as f32 + 0.5) * step_size;
            let pos = [
                start_pos[0] + light_dir[0] * t,
                start_pos[1] + light_dir[1] * t,
                start_pos[2] + light_dir[2] * t,
            ];

            // Sample density (simplified procedural)
            let density = self.sample_cloud_density(pos);

            if density > EPSILON {
                // Record first hit
                if first_hit_distance == f32::MAX {
                    first_hit_distance = t;
                    first_hit_height = get_height_fraction(pos, &self.cloud_layer);
                }

                // Accumulate optical depth
                optical_depth += density * step_size * self.cloud_layer.extinction_coeff;
            }
        }

        // Convert optical depth to shadow factor using Beer-Lambert
        let transmittance = (-optical_depth).exp();

        let mut sample = CloudShadowSample {
            shadow_factor: transmittance,
            optical_depth,
            cloud_distance: first_hit_distance,
            shadow_height: first_hit_height,
        };

        // Apply soft shadow falloff based on height
        sample.apply_height_falloff(self.config.soft_shadow_factor);

        // Apply minimum shadow
        sample.clamp_to_min(self.config.min_shadow);

        sample
    }

    /// Sample cloud density at a position (simplified procedural).
    fn sample_cloud_density(&self, pos: [f32; 3]) -> f32 {
        if !is_inside_cloud_layer(pos, &self.cloud_layer) {
            return 0.0;
        }

        let height_frac = get_height_fraction(pos, &self.cloud_layer);

        // Simple procedural noise
        let tile_size = 6000.0;
        let scaled = [
            pos[0] / tile_size,
            pos[1] / tile_size,
            pos[2] / tile_size,
        ];

        let noise = ((scaled[0] * PI * 2.0).sin() * 0.5 + 0.5)
            * ((scaled[1] * PI * 4.0).sin() * 0.5 + 0.5)
            * ((scaled[2] * PI * 2.0).cos() * 0.5 + 0.5);

        // Apply coverage
        let coverage_threshold = 1.0 - self.cloud_layer.coverage;
        let density = if noise > coverage_threshold {
            (noise - coverage_threshold) / self.cloud_layer.coverage.max(EPSILON)
        } else {
            0.0
        };

        // Apply height gradient (cumulus-like)
        let height_mod = if height_frac < 0.2 {
            height_frac / 0.2
        } else if height_frac > 0.7 {
            1.0 - (height_frac - 0.7) / 0.3
        } else {
            1.0
        };

        density * height_mod * self.cloud_layer.density
    }

    /// Select appropriate cascade for a world position.
    pub fn select_cascade(&self, world_pos: [f32; 3]) -> usize {
        let dx = world_pos[0] - self.camera_pos[0];
        let dz = world_pos[2] - self.camera_pos[2];
        let distance = (dx * dx + dz * dz).sqrt();

        for i in 0..self.config.cascade_count as usize {
            if distance < self.config.cascade_coverage[i] {
                return i;
            }
        }

        (self.config.cascade_count - 1) as usize
    }

    /// Sample shadow with cascade blending.
    pub fn sample_shadow_cascaded(&self, world_pos: [f32; 3]) -> CloudShadowSample {
        let cascade_idx = self.select_cascade(world_pos);
        let cascade = &self.cascades[cascade_idx];

        // Get base sample
        let mut sample = self.sample_shadow(world_pos);

        // Apply cascade blend factor
        let blend = cascade.get_blend_factor(world_pos);
        if blend < 1.0 && cascade_idx + 1 < self.config.cascade_count as usize {
            // Blend with next cascade
            let next_sample = self.sample_shadow(world_pos);
            sample.shadow_factor =
                sample.shadow_factor * blend + next_sample.shadow_factor * (1.0 - blend);
        }

        sample
    }

    /// Compute shadow intensity for rendering.
    ///
    /// Convenience method that returns just the shadow factor.
    #[inline]
    pub fn get_shadow_intensity(&self, world_pos: [f32; 3]) -> f32 {
        self.sample_shadow(world_pos).shadow_factor
    }

    /// Get cascade count.
    #[inline]
    pub fn cascade_count(&self) -> u32 {
        self.config.cascade_count
    }

    /// Get cascade resolution.
    #[inline]
    pub fn cascade_resolution(&self) -> u32 {
        self.config.cascade_resolution
    }
}

// ---------------------------------------------------------------------------
// Temporal Jitter Patterns
// ---------------------------------------------------------------------------

/// Generate Halton sequence value for base b at index n.
///
/// Used for quasi-random jitter patterns that are TAA-friendly.
pub fn halton(n: u32, base: u32) -> f32 {
    let mut f = 1.0f32;
    let mut r = 0.0f32;
    let mut i = n;

    while i > 0 {
        f /= base as f32;
        r += f * (i % base) as f32;
        i /= base;
    }

    r
}

/// Generate 2D Halton point.
#[inline]
pub fn halton_2d(n: u32, base_x: u32, base_y: u32) -> [f32; 2] {
    [halton(n, base_x), halton(n, base_y)]
}

/// Generate jitter offset for a given frame.
///
/// Returns offset in range [-0.5, 0.5] for TAA compatibility.
pub fn taa_jitter_offset(frame: u32) -> [f32; 2] {
    let h = halton_2d(frame % 16 + 1, 2, 3);
    [h[0] - 0.5, h[1] - 0.5]
}

/// 8-sample jitter pattern (Halton 2,3 sequence).
pub const TAA_JITTER_8: [[f32; 2]; 8] = [
    [0.0, 0.0],
    [0.5, 0.333333],
    [0.25, 0.666667],
    [0.75, 0.111111],
    [0.125, 0.444444],
    [0.625, 0.777778],
    [0.375, 0.222222],
    [0.875, 0.555556],
];

/// Get jitter from 8-sample pattern.
#[inline]
pub fn get_jitter_8(frame: u32) -> [f32; 2] {
    TAA_JITTER_8[(frame % 8) as usize]
}

// ---------------------------------------------------------------------------
// Light-Space Projection Utilities
// ---------------------------------------------------------------------------

/// Build orthographic projection matrix.
///
/// # Arguments
///
/// * `left`, `right` - Horizontal bounds
/// * `bottom`, `top` - Vertical bounds
/// * `near`, `far` - Depth bounds
pub fn ortho_matrix(
    left: f32,
    right: f32,
    bottom: f32,
    top: f32,
    near: f32,
    far: f32,
) -> [[f32; 4]; 4] {
    let rml = right - left;
    let tmb = top - bottom;
    let fmn = far - near;

    [
        [2.0 / rml, 0.0, 0.0, 0.0],
        [0.0, 2.0 / tmb, 0.0, 0.0],
        [0.0, 0.0, -2.0 / fmn, 0.0],
        [
            -(right + left) / rml,
            -(top + bottom) / tmb,
            -(far + near) / fmn,
            1.0,
        ],
    ]
}

/// Build look-at view matrix.
///
/// # Arguments
///
/// * `eye` - Camera position
/// * `target` - Look-at target
/// * `up` - Up vector
pub fn look_at_matrix(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
    let f = normalize_vec3([
        target[0] - eye[0],
        target[1] - eye[1],
        target[2] - eye[2],
    ]);

    let s = normalize_vec3(cross(f, up));
    let u = cross(s, f);

    [
        [s[0], u[0], -f[0], 0.0],
        [s[1], u[1], -f[1], 0.0],
        [s[2], u[2], -f[2], 0.0],
        [-dot(s, eye), -dot(u, eye), dot(f, eye), 1.0],
    ]
}

/// Build identity matrix.
#[inline]
pub fn identity_matrix() -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// Multiply two 4x4 matrices (column-major).
pub fn mat4_mul(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut result = [[0.0f32; 4]; 4];

    for i in 0..4 {
        for j in 0..4 {
            result[i][j] = a[0][j] * b[i][0]
                + a[1][j] * b[i][1]
                + a[2][j] * b[i][2]
                + a[3][j] * b[i][3];
        }
    }

    result
}

/// Transform a point by a 4x4 matrix (assumes w=1).
pub fn mat4_transform_point(m: [[f32; 4]; 4], p: [f32; 3]) -> [f32; 3] {
    let w = m[0][3] * p[0] + m[1][3] * p[1] + m[2][3] * p[2] + m[3][3];
    let inv_w = if w.abs() > EPSILON { 1.0 / w } else { 1.0 };

    [
        (m[0][0] * p[0] + m[1][0] * p[1] + m[2][0] * p[2] + m[3][0]) * inv_w,
        (m[0][1] * p[0] + m[1][1] * p[1] + m[2][1] * p[2] + m[3][1]) * inv_w,
        (m[0][2] * p[0] + m[1][2] * p[1] + m[2][2] * p[2] + m[3][2]) * inv_w,
    ]
}

/// Compute inverse of a 4x4 matrix.
///
/// Returns identity if matrix is singular.
pub fn mat4_inverse(m: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut inv = [[0.0f32; 4]; 4];

    inv[0][0] = m[1][1] * m[2][2] * m[3][3] - m[1][1] * m[2][3] * m[3][2]
        - m[2][1] * m[1][2] * m[3][3]
        + m[2][1] * m[1][3] * m[3][2]
        + m[3][1] * m[1][2] * m[2][3]
        - m[3][1] * m[1][3] * m[2][2];

    inv[1][0] = -m[1][0] * m[2][2] * m[3][3]
        + m[1][0] * m[2][3] * m[3][2]
        + m[2][0] * m[1][2] * m[3][3]
        - m[2][0] * m[1][3] * m[3][2]
        - m[3][0] * m[1][2] * m[2][3]
        + m[3][0] * m[1][3] * m[2][2];

    inv[2][0] = m[1][0] * m[2][1] * m[3][3] - m[1][0] * m[2][3] * m[3][1]
        - m[2][0] * m[1][1] * m[3][3]
        + m[2][0] * m[1][3] * m[3][1]
        + m[3][0] * m[1][1] * m[2][3]
        - m[3][0] * m[1][3] * m[2][1];

    inv[3][0] = -m[1][0] * m[2][1] * m[3][2]
        + m[1][0] * m[2][2] * m[3][1]
        + m[2][0] * m[1][1] * m[3][2]
        - m[2][0] * m[1][2] * m[3][1]
        - m[3][0] * m[1][1] * m[2][2]
        + m[3][0] * m[1][2] * m[2][1];

    inv[0][1] = -m[0][1] * m[2][2] * m[3][3]
        + m[0][1] * m[2][3] * m[3][2]
        + m[2][1] * m[0][2] * m[3][3]
        - m[2][1] * m[0][3] * m[3][2]
        - m[3][1] * m[0][2] * m[2][3]
        + m[3][1] * m[0][3] * m[2][2];

    inv[1][1] = m[0][0] * m[2][2] * m[3][3] - m[0][0] * m[2][3] * m[3][2]
        - m[2][0] * m[0][2] * m[3][3]
        + m[2][0] * m[0][3] * m[3][2]
        + m[3][0] * m[0][2] * m[2][3]
        - m[3][0] * m[0][3] * m[2][2];

    inv[2][1] = -m[0][0] * m[2][1] * m[3][3]
        + m[0][0] * m[2][3] * m[3][1]
        + m[2][0] * m[0][1] * m[3][3]
        - m[2][0] * m[0][3] * m[3][1]
        - m[3][0] * m[0][1] * m[2][3]
        + m[3][0] * m[0][3] * m[2][1];

    inv[3][1] = m[0][0] * m[2][1] * m[3][2] - m[0][0] * m[2][2] * m[3][1]
        - m[2][0] * m[0][1] * m[3][2]
        + m[2][0] * m[0][2] * m[3][1]
        + m[3][0] * m[0][1] * m[2][2]
        - m[3][0] * m[0][2] * m[2][1];

    inv[0][2] = m[0][1] * m[1][2] * m[3][3] - m[0][1] * m[1][3] * m[3][2]
        - m[1][1] * m[0][2] * m[3][3]
        + m[1][1] * m[0][3] * m[3][2]
        + m[3][1] * m[0][2] * m[1][3]
        - m[3][1] * m[0][3] * m[1][2];

    inv[1][2] = -m[0][0] * m[1][2] * m[3][3]
        + m[0][0] * m[1][3] * m[3][2]
        + m[1][0] * m[0][2] * m[3][3]
        - m[1][0] * m[0][3] * m[3][2]
        - m[3][0] * m[0][2] * m[1][3]
        + m[3][0] * m[0][3] * m[1][2];

    inv[2][2] = m[0][0] * m[1][1] * m[3][3] - m[0][0] * m[1][3] * m[3][1]
        - m[1][0] * m[0][1] * m[3][3]
        + m[1][0] * m[0][3] * m[3][1]
        + m[3][0] * m[0][1] * m[1][3]
        - m[3][0] * m[0][3] * m[1][1];

    inv[3][2] = -m[0][0] * m[1][1] * m[3][2]
        + m[0][0] * m[1][2] * m[3][1]
        + m[1][0] * m[0][1] * m[3][2]
        - m[1][0] * m[0][2] * m[3][1]
        - m[3][0] * m[0][1] * m[1][2]
        + m[3][0] * m[0][2] * m[1][1];

    inv[0][3] = -m[0][1] * m[1][2] * m[2][3]
        + m[0][1] * m[1][3] * m[2][2]
        + m[1][1] * m[0][2] * m[2][3]
        - m[1][1] * m[0][3] * m[2][2]
        - m[2][1] * m[0][2] * m[1][3]
        + m[2][1] * m[0][3] * m[1][2];

    inv[1][3] = m[0][0] * m[1][2] * m[2][3] - m[0][0] * m[1][3] * m[2][2]
        - m[1][0] * m[0][2] * m[2][3]
        + m[1][0] * m[0][3] * m[2][2]
        + m[2][0] * m[0][2] * m[1][3]
        - m[2][0] * m[0][3] * m[1][2];

    inv[2][3] = -m[0][0] * m[1][1] * m[2][3]
        + m[0][0] * m[1][3] * m[2][1]
        + m[1][0] * m[0][1] * m[2][3]
        - m[1][0] * m[0][3] * m[2][1]
        - m[2][0] * m[0][1] * m[1][3]
        + m[2][0] * m[0][3] * m[1][1];

    inv[3][3] = m[0][0] * m[1][1] * m[2][2] - m[0][0] * m[1][2] * m[2][1]
        - m[1][0] * m[0][1] * m[2][2]
        + m[1][0] * m[0][2] * m[2][1]
        + m[2][0] * m[0][1] * m[1][2]
        - m[2][0] * m[0][2] * m[1][1];

    let det = m[0][0] * inv[0][0] + m[0][1] * inv[1][0] + m[0][2] * inv[2][0] + m[0][3] * inv[3][0];

    if det.abs() < EPSILON {
        return identity_matrix();
    }

    let inv_det = 1.0 / det;

    for i in 0..4 {
        for j in 0..4 {
            inv[i][j] *= inv_det;
        }
    }

    inv
}

/// Calculate cascade split distances using practical split scheme.
///
/// Combines logarithmic and linear distribution.
///
/// # Arguments
///
/// * `near` - Camera near plane
/// * `far` - Camera far plane
/// * `cascade_count` - Number of cascades
/// * `lambda` - Blend factor (0 = linear, 1 = logarithmic)
pub fn calculate_cascade_splits(near: f32, far: f32, cascade_count: u32, lambda: f32) -> [f32; 4] {
    let mut splits = [far; 4];

    for i in 1..=cascade_count {
        let p = i as f32 / cascade_count as f32;

        let log_split = near * (far / near).powf(p);
        let lin_split = near + (far - near) * p;

        let split = lambda * log_split + (1.0 - lambda) * lin_split;
        splits[(i - 1) as usize] = split;
    }

    splits
}

/// Compute texel-snapped position for shadow stability.
///
/// Prevents shadow shimmering by aligning to texel boundaries.
pub fn snap_to_texel(pos: [f32; 2], texel_size: f32) -> [f32; 2] {
    [
        (pos[0] / texel_size).floor() * texel_size,
        (pos[1] / texel_size).floor() * texel_size,
    ]
}

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

/// Normalize a 3D vector.
#[inline]
fn normalize_vec3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > EPSILON {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 1.0, 0.0]
    }
}

/// Cross product of two 3D vectors.
#[inline]
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Dot product of two 3D vectors.
#[inline]
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Smoothstep interpolation.
#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // CloudShadowQuality Tests
    // =========================================================================

    #[test]
    fn test_quality_resolution_low() {
        assert_eq!(CloudShadowQuality::Low.resolution(), 512);
    }

    #[test]
    fn test_quality_resolution_medium() {
        assert_eq!(CloudShadowQuality::Medium.resolution(), 1024);
    }

    #[test]
    fn test_quality_resolution_high() {
        assert_eq!(CloudShadowQuality::High.resolution(), 2048);
    }

    #[test]
    fn test_quality_resolution_ultra() {
        assert_eq!(CloudShadowQuality::Ultra.resolution(), 4096);
    }

    #[test]
    fn test_quality_step_count_low() {
        assert_eq!(CloudShadowQuality::Low.step_count(), 16);
    }

    #[test]
    fn test_quality_step_count_medium() {
        assert_eq!(CloudShadowQuality::Medium.step_count(), 32);
    }

    #[test]
    fn test_quality_step_count_high() {
        assert_eq!(CloudShadowQuality::High.step_count(), 64);
    }

    #[test]
    fn test_quality_step_count_ultra() {
        assert_eq!(CloudShadowQuality::Ultra.step_count(), 128);
    }

    #[test]
    fn test_quality_from_resolution() {
        assert_eq!(CloudShadowQuality::from_resolution(256), CloudShadowQuality::Low);
        assert_eq!(CloudShadowQuality::from_resolution(512), CloudShadowQuality::Low);
        assert_eq!(CloudShadowQuality::from_resolution(1024), CloudShadowQuality::Medium);
        assert_eq!(CloudShadowQuality::from_resolution(2048), CloudShadowQuality::High);
        assert_eq!(CloudShadowQuality::from_resolution(4096), CloudShadowQuality::Ultra);
        assert_eq!(CloudShadowQuality::from_resolution(8192), CloudShadowQuality::Ultra);
    }

    #[test]
    fn test_quality_default() {
        assert_eq!(CloudShadowQuality::default(), CloudShadowQuality::Medium);
    }

    // =========================================================================
    // CloudShadowConfig Tests
    // =========================================================================

    #[test]
    fn test_config_default() {
        let config = CloudShadowConfig::default();
        assert_eq!(config.cascade_count, 4);
        assert_eq!(config.cascade_resolution, 1024);
        assert_eq!(config.shadow_steps, 32);
        assert!(config.validate());
    }

    #[test]
    fn test_config_from_quality_low() {
        let config = CloudShadowConfig::low();
        assert_eq!(config.cascade_resolution, 512);
        assert_eq!(config.shadow_steps, 16);
    }

    #[test]
    fn test_config_from_quality_high() {
        let config = CloudShadowConfig::high();
        assert_eq!(config.cascade_resolution, 2048);
        assert_eq!(config.shadow_steps, 64);
    }

    #[test]
    fn test_config_from_quality_ultra() {
        let config = CloudShadowConfig::ultra();
        assert_eq!(config.cascade_resolution, 4096);
        assert_eq!(config.shadow_steps, 128);
    }

    #[test]
    fn test_config_with_cascade_count() {
        let config = CloudShadowConfig::default().with_cascade_count(2);
        assert_eq!(config.cascade_count, 2);
    }

    #[test]
    fn test_config_cascade_count_clamping() {
        let config = CloudShadowConfig::default().with_cascade_count(0);
        assert_eq!(config.cascade_count, 1);

        let config = CloudShadowConfig::default().with_cascade_count(10);
        assert_eq!(config.cascade_count, 4);
    }

    #[test]
    fn test_config_with_resolution() {
        let config = CloudShadowConfig::default().with_resolution(2048);
        assert_eq!(config.cascade_resolution, 2048);
    }

    #[test]
    fn test_config_resolution_minimum() {
        let config = CloudShadowConfig::default().with_resolution(10);
        assert_eq!(config.cascade_resolution, 64);
    }

    #[test]
    fn test_config_with_shadow_steps() {
        let config = CloudShadowConfig::default().with_shadow_steps(64);
        assert_eq!(config.shadow_steps, 64);
    }

    #[test]
    fn test_config_shadow_steps_minimum() {
        let config = CloudShadowConfig::default().with_shadow_steps(1);
        assert_eq!(config.shadow_steps, 4);
    }

    #[test]
    fn test_config_with_soft_shadow() {
        let config = CloudShadowConfig::default().with_soft_shadow(0.8);
        assert!((config.soft_shadow_factor - 0.8).abs() < EPSILON);
    }

    #[test]
    fn test_config_soft_shadow_clamping() {
        let config = CloudShadowConfig::default().with_soft_shadow(-0.5);
        assert_eq!(config.soft_shadow_factor, 0.0);

        let config = CloudShadowConfig::default().with_soft_shadow(1.5);
        assert_eq!(config.soft_shadow_factor, 1.0);
    }

    #[test]
    fn test_config_with_temporal_blend() {
        let config = CloudShadowConfig::default().with_temporal_blend(0.2);
        assert!((config.temporal_blend - 0.2).abs() < EPSILON);
    }

    #[test]
    fn test_config_with_min_shadow() {
        let config = CloudShadowConfig::default().with_min_shadow(0.3);
        assert!((config.min_shadow - 0.3).abs() < EPSILON);
    }

    #[test]
    fn test_config_with_temporal_jitter() {
        let config = CloudShadowConfig::default().with_temporal_jitter(false);
        assert!(!config.enable_temporal_jitter);
    }

    #[test]
    fn test_config_validate_valid() {
        let config = CloudShadowConfig::default();
        assert!(config.validate());
    }

    #[test]
    fn test_config_validate_invalid_cascade_count() {
        let mut config = CloudShadowConfig::default();
        config.cascade_count = 0;
        assert!(!config.validate());

        config.cascade_count = 5;
        assert!(!config.validate());
    }

    #[test]
    fn test_config_validate_invalid_resolution() {
        let mut config = CloudShadowConfig::default();
        config.cascade_resolution = 32;
        assert!(!config.validate());
    }

    #[test]
    fn test_config_validate_invalid_steps() {
        let mut config = CloudShadowConfig::default();
        config.shadow_steps = 2;
        assert!(!config.validate());
    }

    #[test]
    fn test_config_total_texels() {
        let config = CloudShadowConfig::default();
        let expected = 1024u64 * 1024 * 4;
        assert_eq!(config.total_texels(), expected);
    }

    // =========================================================================
    // CloudShadowCascade Tests
    // =========================================================================

    #[test]
    fn test_cascade_default() {
        let cascade = CloudShadowCascade::default();
        assert_eq!(cascade.cascade_index, 0);
        assert_eq!(cascade.coverage_distance, 500.0);
    }

    #[test]
    fn test_cascade_new() {
        let cascade = CloudShadowCascade::new(2, 1000.0, 1024);
        assert_eq!(cascade.cascade_index, 2);
        assert_eq!(cascade.coverage_distance, 1000.0);
        assert!((cascade.texel_size - 2000.0 / 1024.0).abs() < EPSILON);
    }

    #[test]
    fn test_cascade_coverage_bounds() {
        let cascade = CloudShadowCascade::new(0, 500.0, 1024);
        assert_eq!(cascade.coverage_bounds[0], -500.0);
        assert_eq!(cascade.coverage_bounds[1], -500.0);
        assert_eq!(cascade.coverage_bounds[2], 500.0);
        assert_eq!(cascade.coverage_bounds[3], 500.0);
    }

    #[test]
    fn test_cascade_contains_inside() {
        let cascade = CloudShadowCascade::new(0, 500.0, 1024);
        assert!(cascade.contains([0.0, 100.0, 0.0]));
        assert!(cascade.contains([250.0, 0.0, -250.0]));
    }

    #[test]
    fn test_cascade_contains_outside() {
        let cascade = CloudShadowCascade::new(0, 500.0, 1024);
        assert!(!cascade.contains([600.0, 0.0, 0.0]));
        assert!(!cascade.contains([0.0, 0.0, -600.0]));
    }

    #[test]
    fn test_cascade_contains_boundary() {
        let cascade = CloudShadowCascade::new(0, 500.0, 1024);
        assert!(cascade.contains([-500.0, 0.0, 500.0]));
    }

    #[test]
    fn test_cascade_blend_factor_center() {
        let cascade = CloudShadowCascade::new(0, 500.0, 1024);
        let blend = cascade.get_blend_factor([0.0, 0.0, 0.0]);
        assert!((blend - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_cascade_blend_factor_edge() {
        let cascade = CloudShadowCascade::new(0, 500.0, 1024);
        let blend = cascade.get_blend_factor([500.0, 0.0, 0.0]);
        assert!(blend < 0.1);
    }

    #[test]
    fn test_cascade_blend_factor_transition() {
        let cascade = CloudShadowCascade::new(0, 500.0, 1024);
        let blend = cascade.get_blend_factor([450.0, 0.0, 0.0]);
        assert!(blend < 1.0);
        assert!(blend > 0.0);
    }

    #[test]
    fn test_cascade_project_to_uv_center() {
        let cascade = CloudShadowCascade::default();
        let uv = cascade.project_to_uv([0.0, 0.0, 0.0]);
        // With identity matrix, center should map near 0.5, 0.5
        assert!(uv[0] >= 0.0 && uv[0] <= 1.0);
        assert!(uv[1] >= 0.0 && uv[1] <= 1.0);
    }

    #[test]
    fn test_cascade_pod() {
        let cascade = CloudShadowCascade::default();
        let bytes = bytemuck::bytes_of(&cascade);
        assert_eq!(bytes.len(), 160);
    }

    #[test]
    fn test_cascade_size() {
        assert_eq!(std::mem::size_of::<CloudShadowCascade>(), 160);
    }

    // =========================================================================
    // CloudShadowUniforms Tests
    // =========================================================================

    #[test]
    fn test_uniforms_default() {
        let uniforms = CloudShadowUniforms::default();
        assert_eq!(uniforms.shadow_steps, DEFAULT_SHADOW_STEPS);
        assert_eq!(uniforms.cascade_count, DEFAULT_CASCADE_COUNT);
    }

    #[test]
    fn test_uniforms_from_config() {
        let config = CloudShadowConfig::high();
        let layer = CloudLayerConfig::default();
        let uniforms = CloudShadowUniforms::from_config(&config, &layer);

        assert_eq!(uniforms.shadow_steps, 64);
        assert_eq!(uniforms.cloud_min_height, layer.min_height);
        assert_eq!(uniforms.cloud_max_height, layer.max_height);
    }

    #[test]
    fn test_uniforms_set_light_direction() {
        let mut uniforms = CloudShadowUniforms::default();
        uniforms.set_light_direction([1.0, 0.0, 0.0]);

        let len = (uniforms.light_direction[0].powi(2)
            + uniforms.light_direction[1].powi(2)
            + uniforms.light_direction[2].powi(2))
        .sqrt();
        assert!((len - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_uniforms_advance_frame() {
        let mut uniforms = CloudShadowUniforms::default();
        assert_eq!(uniforms.frame_index, 0);

        uniforms.advance_frame();
        assert_eq!(uniforms.frame_index, 1);

        uniforms.advance_frame();
        assert_eq!(uniforms.frame_index, 2);
    }

    #[test]
    fn test_uniforms_jitter_offset_changes() {
        let mut uniforms = CloudShadowUniforms::default();
        let jitter0 = uniforms.jitter_offset;

        uniforms.advance_frame();
        let jitter1 = uniforms.jitter_offset;

        // Jitter should change each frame
        assert!(jitter0[0] != jitter1[0] || jitter0[1] != jitter1[1]);
    }

    #[test]
    fn test_uniforms_cloud_thickness() {
        let uniforms = CloudShadowUniforms::default();
        let expected = uniforms.cloud_max_height - uniforms.cloud_min_height;
        assert!((uniforms.cloud_thickness() - expected).abs() < EPSILON);
    }

    #[test]
    fn test_uniforms_pod() {
        let uniforms = CloudShadowUniforms::default();
        let bytes = bytemuck::bytes_of(&uniforms);
        assert_eq!(bytes.len(), 128);
    }

    #[test]
    fn test_uniforms_size() {
        assert_eq!(std::mem::size_of::<CloudShadowUniforms>(), 128);
    }

    // =========================================================================
    // CloudShadowSample Tests
    // =========================================================================

    #[test]
    fn test_sample_no_shadow() {
        let sample = CloudShadowSample::no_shadow();
        assert_eq!(sample.shadow_factor, 1.0);
        assert_eq!(sample.optical_depth, 0.0);
    }

    #[test]
    fn test_sample_full_shadow() {
        let sample = CloudShadowSample::full_shadow();
        assert_eq!(sample.shadow_factor, 0.0);
    }

    #[test]
    fn test_sample_is_shadowed_threshold() {
        let sample = CloudShadowSample {
            shadow_factor: 0.3,
            ..Default::default()
        };
        assert!(sample.is_shadowed(0.5));
        assert!(!sample.is_shadowed(0.2));
    }

    #[test]
    fn test_sample_clamp_to_min() {
        let mut sample = CloudShadowSample::full_shadow();
        sample.clamp_to_min(0.2);
        assert_eq!(sample.shadow_factor, 0.2);
    }

    #[test]
    fn test_sample_clamp_to_min_above() {
        let mut sample = CloudShadowSample {
            shadow_factor: 0.5,
            ..Default::default()
        };
        sample.clamp_to_min(0.2);
        assert_eq!(sample.shadow_factor, 0.5);
    }

    #[test]
    fn test_sample_height_falloff_low() {
        let mut sample = CloudShadowSample {
            shadow_factor: 0.5,
            shadow_height: 0.0,
            ..Default::default()
        };
        let original = sample.shadow_factor;
        sample.apply_height_falloff(0.5);
        // Low clouds should have less softening
        assert!(sample.shadow_factor >= original);
    }

    #[test]
    fn test_sample_height_falloff_high() {
        let mut sample = CloudShadowSample {
            shadow_factor: 0.3,
            shadow_height: 1.0,
            ..Default::default()
        };
        sample.apply_height_falloff(0.5);
        // High clouds cast softer shadows
        assert!(sample.shadow_factor > 0.3);
    }

    // =========================================================================
    // CloudShadowRenderer Tests
    // =========================================================================

    #[test]
    fn test_renderer_new() {
        let config = CloudShadowConfig::default();
        let layer = CloudLayerConfig::default();
        let renderer = CloudShadowRenderer::new(config, layer);

        assert_eq!(renderer.cascade_count(), 4);
        assert_eq!(renderer.cascade_resolution(), 1024);
    }

    #[test]
    fn test_renderer_config() {
        let config = CloudShadowConfig::high();
        let renderer = CloudShadowRenderer::new(config, CloudLayerConfig::default());
        assert_eq!(renderer.config().cascade_resolution, 2048);
    }

    #[test]
    fn test_renderer_cloud_layer() {
        let layer = CloudLayerConfig::stratus();
        let renderer = CloudShadowRenderer::new(CloudShadowConfig::default(), layer);
        assert_eq!(renderer.cloud_layer().min_height, layer.min_height);
    }

    #[test]
    fn test_renderer_set_light_direction() {
        let mut renderer =
            CloudShadowRenderer::new(CloudShadowConfig::default(), CloudLayerConfig::default());
        renderer.set_light_direction([1.0, 1.0, 0.0]);

        let dir = renderer.uniforms().light_direction;
        let len = (dir[0].powi(2) + dir[1].powi(2) + dir[2].powi(2)).sqrt();
        assert!((len - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_renderer_set_camera_position() {
        let mut renderer =
            CloudShadowRenderer::new(CloudShadowConfig::default(), CloudLayerConfig::default());
        renderer.set_camera_position([100.0, 50.0, 200.0]);
        // Camera position should affect cascade selection
    }

    #[test]
    fn test_renderer_advance_frame() {
        let mut renderer =
            CloudShadowRenderer::new(CloudShadowConfig::default(), CloudLayerConfig::default());
        let frame0 = renderer.uniforms().frame_index;
        renderer.advance_frame();
        assert_eq!(renderer.uniforms().frame_index, frame0 + 1);
    }

    #[test]
    fn test_renderer_get_jitter_offset_enabled() {
        let renderer =
            CloudShadowRenderer::new(CloudShadowConfig::default(), CloudLayerConfig::default());
        let jitter = renderer.get_jitter_offset();
        // Jitter should be in reasonable range
        assert!(jitter[0].abs() <= 2.0);
        assert!(jitter[1].abs() <= 2.0);
    }

    #[test]
    fn test_renderer_get_jitter_offset_disabled() {
        let config = CloudShadowConfig::default().with_temporal_jitter(false);
        let renderer = CloudShadowRenderer::new(config, CloudLayerConfig::default());
        let jitter = renderer.get_jitter_offset();
        assert_eq!(jitter, [0.0, 0.0]);
    }

    #[test]
    fn test_renderer_cascade_access() {
        let renderer =
            CloudShadowRenderer::new(CloudShadowConfig::default(), CloudLayerConfig::default());
        assert!(renderer.cascade(0).is_some());
        assert!(renderer.cascade(3).is_some());
        assert!(renderer.cascade(4).is_none());
    }

    #[test]
    fn test_renderer_cascades_slice() {
        let renderer =
            CloudShadowRenderer::new(CloudShadowConfig::default(), CloudLayerConfig::default());
        assert_eq!(renderer.cascades().len(), 4);
    }

    #[test]
    fn test_renderer_sample_shadow_above_clouds() {
        let renderer =
            CloudShadowRenderer::new(CloudShadowConfig::default(), CloudLayerConfig::default());
        // Position above clouds should have no shadow
        let sample = renderer.sample_shadow([0.0, 10000.0, 0.0]);
        assert_eq!(sample.shadow_factor, 1.0);
    }

    #[test]
    fn test_renderer_sample_shadow_below_clouds() {
        let renderer =
            CloudShadowRenderer::new(CloudShadowConfig::default(), CloudLayerConfig::default());
        // Position below clouds may have shadow depending on density
        let sample = renderer.sample_shadow([0.0, 100.0, 0.0]);
        assert!(sample.shadow_factor >= 0.0);
        assert!(sample.shadow_factor <= 1.0);
    }

    #[test]
    fn test_renderer_sample_shadow_jittered() {
        let renderer =
            CloudShadowRenderer::new(CloudShadowConfig::default(), CloudLayerConfig::default());
        let sample = renderer.sample_shadow_jittered([0.0, 100.0, 0.0]);
        assert!(sample.shadow_factor >= 0.0);
        assert!(sample.shadow_factor <= 1.0);
    }

    #[test]
    fn test_renderer_select_cascade_near() {
        let mut renderer =
            CloudShadowRenderer::new(CloudShadowConfig::default(), CloudLayerConfig::default());
        renderer.set_camera_position([0.0, 0.0, 0.0]);
        let idx = renderer.select_cascade([100.0, 0.0, 0.0]);
        assert_eq!(idx, 0);
    }

    #[test]
    fn test_renderer_select_cascade_far() {
        let mut renderer =
            CloudShadowRenderer::new(CloudShadowConfig::default(), CloudLayerConfig::default());
        renderer.set_camera_position([0.0, 0.0, 0.0]);
        let idx = renderer.select_cascade([100000.0, 0.0, 0.0]);
        assert_eq!(idx, 3);
    }

    #[test]
    fn test_renderer_sample_shadow_cascaded() {
        let mut renderer =
            CloudShadowRenderer::new(CloudShadowConfig::default(), CloudLayerConfig::default());
        renderer.set_camera_position([0.0, 0.0, 0.0]);
        renderer.update_cascades();

        let sample = renderer.sample_shadow_cascaded([0.0, 100.0, 0.0]);
        assert!(sample.shadow_factor >= 0.0);
        assert!(sample.shadow_factor <= 1.0);
    }

    #[test]
    fn test_renderer_get_shadow_intensity() {
        let renderer =
            CloudShadowRenderer::new(CloudShadowConfig::default(), CloudLayerConfig::default());
        let intensity = renderer.get_shadow_intensity([0.0, 100.0, 0.0]);
        assert!(intensity >= 0.0);
        assert!(intensity <= 1.0);
    }

    // =========================================================================
    // Halton Sequence Tests
    // =========================================================================

    #[test]
    fn test_halton_base_2() {
        assert!((halton(0, 2) - 0.0).abs() < EPSILON);
        assert!((halton(1, 2) - 0.5).abs() < EPSILON);
        assert!((halton(2, 2) - 0.25).abs() < EPSILON);
        assert!((halton(3, 2) - 0.75).abs() < EPSILON);
    }

    #[test]
    fn test_halton_base_3() {
        assert!((halton(0, 3) - 0.0).abs() < EPSILON);
        assert!((halton(1, 3) - 0.333333).abs() < 0.001);
        assert!((halton(2, 3) - 0.666667).abs() < 0.001);
    }

    #[test]
    fn test_halton_range() {
        for i in 0..100 {
            let h = halton(i, 2);
            assert!(h >= 0.0 && h < 1.0);
        }
    }

    #[test]
    fn test_halton_2d() {
        let h = halton_2d(1, 2, 3);
        assert!((h[0] - 0.5).abs() < EPSILON);
        assert!((h[1] - 0.333333).abs() < 0.001);
    }

    #[test]
    fn test_taa_jitter_offset() {
        let jitter = taa_jitter_offset(0);
        assert!(jitter[0] >= -0.5 && jitter[0] <= 0.5);
        assert!(jitter[1] >= -0.5 && jitter[1] <= 0.5);
    }

    #[test]
    fn test_taa_jitter_8_pattern() {
        for jitter in TAA_JITTER_8.iter() {
            assert!(jitter[0] >= 0.0 && jitter[0] <= 1.0);
            assert!(jitter[1] >= 0.0 && jitter[1] <= 1.0);
        }
    }

    #[test]
    fn test_get_jitter_8() {
        assert_eq!(get_jitter_8(0), TAA_JITTER_8[0]);
        assert_eq!(get_jitter_8(8), TAA_JITTER_8[0]); // Wraps
    }

    // =========================================================================
    // Matrix Tests
    // =========================================================================

    #[test]
    fn test_identity_matrix() {
        let m = identity_matrix();
        assert_eq!(m[0][0], 1.0);
        assert_eq!(m[1][1], 1.0);
        assert_eq!(m[2][2], 1.0);
        assert_eq!(m[3][3], 1.0);
        assert_eq!(m[0][1], 0.0);
    }

    #[test]
    fn test_ortho_matrix_basic() {
        let m = ortho_matrix(-1.0, 1.0, -1.0, 1.0, 0.1, 100.0);
        // Check that diagonal elements are non-zero
        assert!(m[0][0].abs() > EPSILON);
        assert!(m[1][1].abs() > EPSILON);
        assert!(m[2][2].abs() > EPSILON);
    }

    #[test]
    fn test_ortho_matrix_symmetric() {
        let m = ortho_matrix(-100.0, 100.0, -100.0, 100.0, 1.0, 1000.0);
        // For symmetric bounds, translation should be zero
        assert!(m[3][0].abs() < EPSILON);
        assert!(m[3][1].abs() < EPSILON);
    }

    #[test]
    fn test_look_at_matrix_looking_down_z() {
        let m = look_at_matrix([0.0, 0.0, 5.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        // Forward should be -Z
        assert!(m[2][2].abs() > EPSILON);
    }

    #[test]
    fn test_mat4_mul_identity() {
        let id = identity_matrix();
        let result = mat4_mul(id, id);
        assert_eq!(result, id);
    }

    #[test]
    fn test_mat4_mul_order() {
        let a = ortho_matrix(-1.0, 1.0, -1.0, 1.0, 0.1, 10.0);
        let id = identity_matrix();
        let result = mat4_mul(a, id);
        assert_eq!(result, a);
    }

    #[test]
    fn test_mat4_transform_point_identity() {
        let id = identity_matrix();
        let p = [1.0, 2.0, 3.0];
        let result = mat4_transform_point(id, p);
        assert!((result[0] - p[0]).abs() < EPSILON);
        assert!((result[1] - p[1]).abs() < EPSILON);
        assert!((result[2] - p[2]).abs() < EPSILON);
    }

    #[test]
    fn test_mat4_inverse_identity() {
        let id = identity_matrix();
        let inv = mat4_inverse(id);
        assert_eq!(inv, id);
    }

    #[test]
    fn test_mat4_inverse_roundtrip() {
        let m = ortho_matrix(-10.0, 10.0, -10.0, 10.0, 0.1, 100.0);
        let inv = mat4_inverse(m);
        let result = mat4_mul(m, inv);

        // Should be close to identity
        let id = identity_matrix();
        for i in 0..4 {
            for j in 0..4 {
                assert!((result[i][j] - id[i][j]).abs() < 0.001);
            }
        }
    }

    // =========================================================================
    // Cascade Split Tests
    // =========================================================================

    #[test]
    fn test_cascade_splits_linear() {
        let splits = calculate_cascade_splits(0.1, 100.0, 4, 0.0);
        // Linear distribution
        assert!((splits[0] - 25.075).abs() < 0.1);
        assert!((splits[1] - 50.05).abs() < 0.1);
        assert!((splits[2] - 75.025).abs() < 0.1);
        assert!((splits[3] - 100.0).abs() < EPSILON);
    }

    #[test]
    fn test_cascade_splits_logarithmic() {
        let splits = calculate_cascade_splits(0.1, 100.0, 4, 1.0);
        // Logarithmic: more detail near camera
        assert!(splits[0] < 25.0); // First split closer
        assert!(splits[3] == 100.0);
    }

    #[test]
    fn test_cascade_splits_practical() {
        let splits = calculate_cascade_splits(0.1, 100.0, 4, 0.5);
        // Blend of linear and log
        assert!(splits[0] > 0.1);
        assert!(splits[0] < 25.0);
    }

    #[test]
    fn test_cascade_splits_ordering() {
        let splits = calculate_cascade_splits(1.0, 1000.0, 4, 0.5);
        assert!(splits[0] < splits[1]);
        assert!(splits[1] < splits[2]);
        assert!(splits[2] < splits[3]);
    }

    // =========================================================================
    // Snap to Texel Tests
    // =========================================================================

    #[test]
    fn test_snap_to_texel_aligned() {
        let snapped = snap_to_texel([10.0, 20.0], 10.0);
        assert_eq!(snapped, [10.0, 20.0]);
    }

    #[test]
    fn test_snap_to_texel_unaligned() {
        let snapped = snap_to_texel([15.5, 23.7], 10.0);
        assert_eq!(snapped, [10.0, 20.0]);
    }

    #[test]
    fn test_snap_to_texel_negative() {
        let snapped = snap_to_texel([-15.5, -23.7], 10.0);
        assert_eq!(snapped, [-20.0, -30.0]);
    }

    #[test]
    fn test_snap_to_texel_small() {
        let snapped = snap_to_texel([0.123, 0.456], 0.1);
        assert!((snapped[0] - 0.1).abs() < EPSILON);
        assert!((snapped[1] - 0.4).abs() < EPSILON);
    }

    // =========================================================================
    // Helper Function Tests
    // =========================================================================

    #[test]
    fn test_normalize_vec3_unit() {
        let v = normalize_vec3([1.0, 0.0, 0.0]);
        assert_eq!(v, [1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_normalize_vec3_scaled() {
        let v = normalize_vec3([3.0, 0.0, 4.0]);
        assert!((v[0] - 0.6).abs() < EPSILON);
        assert!(v[1].abs() < EPSILON);
        assert!((v[2] - 0.8).abs() < EPSILON);
    }

    #[test]
    fn test_normalize_vec3_zero() {
        let v = normalize_vec3([0.0, 0.0, 0.0]);
        assert_eq!(v, [0.0, 1.0, 0.0]); // Default to up
    }

    #[test]
    fn test_cross_basic() {
        let c = cross([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        assert!((c[2] - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_cross_anticommutative() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let ab = cross(a, b);
        let ba = cross(b, a);
        assert!((ab[0] + ba[0]).abs() < EPSILON);
        assert!((ab[1] + ba[1]).abs() < EPSILON);
        assert!((ab[2] + ba[2]).abs() < EPSILON);
    }

    #[test]
    fn test_dot_orthogonal() {
        let d = dot([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        assert!(d.abs() < EPSILON);
    }

    #[test]
    fn test_dot_parallel() {
        let d = dot([1.0, 0.0, 0.0], [2.0, 0.0, 0.0]);
        assert!((d - 2.0).abs() < EPSILON);
    }

    #[test]
    fn test_smoothstep_edges() {
        assert!((smoothstep(0.0, 1.0, 0.0) - 0.0).abs() < EPSILON);
        assert!((smoothstep(0.0, 1.0, 1.0) - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_smoothstep_middle() {
        let s = smoothstep(0.0, 1.0, 0.5);
        assert!((s - 0.5).abs() < EPSILON);
    }

    #[test]
    fn test_smoothstep_clamped() {
        assert!((smoothstep(0.0, 1.0, -0.5) - 0.0).abs() < EPSILON);
        assert!((smoothstep(0.0, 1.0, 1.5) - 1.0).abs() < EPSILON);
    }
}
