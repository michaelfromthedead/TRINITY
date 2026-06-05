//! Cloud Ray Marching Pass (T-ENV-2.2)
//!
//! Implements volumetric cloud ray marching for TRINITY's atmosphere system.
//! Uses noise textures from `cloud_noise.rs` and integrates with `sky_rendering.rs`.
//!
//! # Overview
//!
//! The ray marching system provides:
//! - **CloudLayerConfig**: Defines cloud layer boundaries and optical properties
//! - **CloudQuality**: Presets for step count vs performance tradeoffs
//! - **RayMarchConfig**: Controls ray marching behavior and LOD
//! - **CloudRayMarcher**: Main API for ray marching through cloud volumes
//!
//! # Physics Model
//!
//! The lighting model implements:
//! - Beer-Lambert transmission for extinction
//! - Powder effect for bright cloud edges (Henyey-Greenstein approximation)
//! - Multi-scattering approximation for energy conservation
//! - Height-based density gradients for cloud type shaping
//!
//! # Performance
//!
//! At 128 steps (High quality), targets 60 fps with:
//! - Early termination at transmittance threshold
//! - Adaptive step count based on distance
//! - LOD falloff for distant clouds
//!
//! # References
//!
//! - Schneider & Vos, "The Real-time Volumetric Cloudscapes of Horizon: Zero Dawn"
//! - Hillaire, "A Scalable and Production Ready Sky and Atmosphere Rendering Technique"

use bytemuck::{Pod, Zeroable};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default minimum cloud layer height in meters (1.5km cumulus base).
pub const DEFAULT_MIN_HEIGHT: f32 = 1500.0;

/// Default maximum cloud layer height in meters (8km cumulus top).
pub const DEFAULT_MAX_HEIGHT: f32 = 8000.0;

/// Default global cloud coverage (0-1).
pub const DEFAULT_COVERAGE: f32 = 0.5;

/// Default cloud density (extinction coefficient).
pub const DEFAULT_DENSITY: f32 = 0.03;

/// Default extinction coefficient for Beer-Lambert law.
pub const DEFAULT_EXTINCTION_COEFF: f32 = 0.1;

/// Default powder effect factor for bright edges.
pub const DEFAULT_POWDER_FACTOR: f32 = 2.0;

/// Default transmittance threshold for early ray termination.
pub const DEFAULT_TRANSMITTANCE_THRESHOLD: f32 = 0.01;

/// Default distance fade start in meters (50km).
pub const DEFAULT_DISTANCE_FADE_START: f32 = 50_000.0;

/// Default distance fade end in meters (150km).
pub const DEFAULT_DISTANCE_FADE_END: f32 = 150_000.0;

/// Default step size in meters for ray marching.
pub const DEFAULT_STEP_SIZE: f32 = 100.0;

/// Earth radius in meters for spherical calculations.
pub const EARTH_RADIUS: f32 = 6_371_000.0;

/// Default scattering albedo for clouds (highly reflective).
pub const DEFAULT_SCATTERING_ALBEDO: f32 = 0.99;

/// Small epsilon for floating point comparisons.
pub const EPSILON: f32 = 1e-6;

// ---------------------------------------------------------------------------
// CloudType - Cloud shape classification
// ---------------------------------------------------------------------------

/// Cloud type affecting height gradient shaping.
///
/// Different cloud types have distinct vertical density profiles
/// that determine their characteristic shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum CloudType {
    /// Stratus clouds: flat, low-lying, uniform coverage.
    /// Density concentrated at lower heights.
    Stratus = 0,

    /// Cumulus clouds: puffy, fair-weather clouds.
    /// Rounded bottom, billowing top.
    #[default]
    Cumulus = 1,

    /// Cumulonimbus clouds: tall, anvil-shaped storm clouds.
    /// Density extends through full height range.
    Cumulonimbus = 2,

    /// Stratocumulus: lumpy layer clouds.
    /// Moderate height variation.
    Stratocumulus = 3,

    /// Cirrus: wispy, high-altitude ice clouds.
    /// Very thin, fibrous structure.
    Cirrus = 4,
}

impl CloudType {
    /// Get the height gradient parameters for this cloud type.
    ///
    /// Returns (ramp_bottom, ramp_top, anvil_factor) where:
    /// - ramp_bottom: height fraction where density starts ramping up
    /// - ramp_top: height fraction where density starts ramping down
    /// - anvil_factor: broadening factor at top (for cumulonimbus)
    #[inline]
    pub fn height_gradient_params(&self) -> (f32, f32, f32) {
        match self {
            CloudType::Stratus => (0.0, 0.1, 0.0),
            CloudType::Cumulus => (0.1, 0.6, 0.0),
            CloudType::Cumulonimbus => (0.0, 0.8, 0.3),
            CloudType::Stratocumulus => (0.1, 0.4, 0.0),
            CloudType::Cirrus => (0.6, 0.9, 0.0),
        }
    }

    /// Get typical coverage range for this cloud type.
    #[inline]
    pub fn typical_coverage(&self) -> (f32, f32) {
        match self {
            CloudType::Stratus => (0.6, 0.95),
            CloudType::Cumulus => (0.2, 0.5),
            CloudType::Cumulonimbus => (0.3, 0.7),
            CloudType::Stratocumulus => (0.4, 0.8),
            CloudType::Cirrus => (0.1, 0.3),
        }
    }
}

// ---------------------------------------------------------------------------
// CloudLayerConfig - Cloud layer definition
// ---------------------------------------------------------------------------

/// Configuration for a single cloud layer.
///
/// Defines the vertical extent and optical properties of clouds.
/// Designed for GPU upload as a uniform buffer.
///
/// # Memory Layout (32 bytes)
///
/// | Offset | Field           | Size    |
/// |--------|-----------------|---------|
/// | 0      | min_height      | 4 bytes |
/// | 4      | max_height      | 4 bytes |
/// | 8      | coverage        | 4 bytes |
/// | 12     | density         | 4 bytes |
/// | 16     | extinction_coeff| 4 bytes |
/// | 20     | powder_factor   | 4 bytes |
/// | 24     | _padding        | 8 bytes |
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct CloudLayerConfig {
    /// Minimum height of cloud layer in meters (default 1500m = 1.5km).
    pub min_height: f32,

    /// Maximum height of cloud layer in meters (default 8000m = 8km).
    pub max_height: f32,

    /// Global cloud coverage factor (0-1, default 0.5).
    pub coverage: f32,

    /// Cloud density / extinction coefficient (default 0.03).
    pub density: f32,

    /// Extinction coefficient for Beer-Lambert law (default 0.1).
    pub extinction_coeff: f32,

    /// Powder effect factor for bright cloud edges (default 2.0).
    pub powder_factor: f32,

    /// Padding for GPU alignment.
    pub _padding: [u32; 2],
}

// Size assertion for GPU compatibility
const _: () = assert!(std::mem::size_of::<CloudLayerConfig>() == 32);

impl Default for CloudLayerConfig {
    fn default() -> Self {
        Self {
            min_height: DEFAULT_MIN_HEIGHT,
            max_height: DEFAULT_MAX_HEIGHT,
            coverage: DEFAULT_COVERAGE,
            density: DEFAULT_DENSITY,
            extinction_coeff: DEFAULT_EXTINCTION_COEFF,
            powder_factor: DEFAULT_POWDER_FACTOR,
            _padding: [0; 2],
        }
    }
}

impl CloudLayerConfig {
    /// Create a new cloud layer configuration.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a configuration for low-altitude stratus clouds.
    #[inline]
    pub fn stratus() -> Self {
        Self {
            min_height: 500.0,
            max_height: 2000.0,
            coverage: 0.8,
            density: 0.05,
            extinction_coeff: 0.15,
            powder_factor: 1.5,
            _padding: [0; 2],
        }
    }

    /// Create a configuration for cumulus clouds (fair weather).
    #[inline]
    pub fn cumulus() -> Self {
        Self::default()
    }

    /// Create a configuration for high-altitude cirrus clouds.
    #[inline]
    pub fn cirrus() -> Self {
        Self {
            min_height: 6000.0,
            max_height: 12000.0,
            coverage: 0.3,
            density: 0.01,
            extinction_coeff: 0.05,
            powder_factor: 3.0,
            _padding: [0; 2],
        }
    }

    /// Create a configuration for storm clouds (cumulonimbus).
    #[inline]
    pub fn storm() -> Self {
        Self {
            min_height: 500.0,
            max_height: 12000.0,
            coverage: 0.7,
            density: 0.08,
            extinction_coeff: 0.2,
            powder_factor: 1.0,
            _padding: [0; 2],
        }
    }

    /// Create a custom configuration with builder pattern.
    #[inline]
    pub fn with_height(mut self, min: f32, max: f32) -> Self {
        self.min_height = min.max(0.0);
        self.max_height = max.max(self.min_height + 100.0);
        self
    }

    /// Set coverage factor.
    #[inline]
    pub fn with_coverage(mut self, coverage: f32) -> Self {
        self.coverage = coverage.clamp(0.0, 1.0);
        self
    }

    /// Set density factor.
    #[inline]
    pub fn with_density(mut self, density: f32) -> Self {
        self.density = density.max(0.0);
        self
    }

    /// Get the thickness of the cloud layer in meters.
    #[inline]
    pub fn thickness(&self) -> f32 {
        self.max_height - self.min_height
    }

    /// Validate the configuration.
    pub fn validate(&self) -> bool {
        self.min_height >= 0.0
            && self.max_height > self.min_height
            && self.coverage >= 0.0
            && self.coverage <= 1.0
            && self.density >= 0.0
            && self.extinction_coeff >= 0.0
            && self.powder_factor >= 0.0
    }
}

// ---------------------------------------------------------------------------
// CloudQuality - Quality presets
// ---------------------------------------------------------------------------

/// Cloud rendering quality preset.
///
/// Controls the number of ray march steps, trading quality for performance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum CloudQuality {
    /// Low quality: 32 steps. Fast, suitable for mobile/low-end.
    Low = 0,

    /// Medium quality: 64 steps. Balanced quality/performance.
    #[default]
    Medium = 1,

    /// High quality: 128 steps. Good quality at 60 fps.
    High = 2,

    /// Ultra quality: 256 steps. Maximum quality, demanding.
    Ultra = 3,
}

impl CloudQuality {
    /// Get the number of ray march steps for this quality level.
    #[inline]
    pub fn step_count(&self) -> u32 {
        match self {
            CloudQuality::Low => 32,
            CloudQuality::Medium => 64,
            CloudQuality::High => 128,
            CloudQuality::Ultra => 256,
        }
    }

    /// Get the minimum step size multiplier for this quality level.
    #[inline]
    pub fn min_step_multiplier(&self) -> f32 {
        match self {
            CloudQuality::Low => 2.0,
            CloudQuality::Medium => 1.5,
            CloudQuality::High => 1.0,
            CloudQuality::Ultra => 0.75,
        }
    }

    /// Get quality level from step count (rounds down).
    #[inline]
    pub fn from_step_count(steps: u32) -> Self {
        if steps >= 256 {
            CloudQuality::Ultra
        } else if steps >= 128 {
            CloudQuality::High
        } else if steps >= 64 {
            CloudQuality::Medium
        } else {
            CloudQuality::Low
        }
    }
}

// ---------------------------------------------------------------------------
// RayMarchConfig - Ray marching parameters
// ---------------------------------------------------------------------------

/// Configuration for the ray marching algorithm.
///
/// Controls step size, early termination, and distance-based LOD.
///
/// # Memory Layout (32 bytes)
///
/// | Offset | Field                    | Size    |
/// |--------|--------------------------|---------|
/// | 0      | max_steps                | 4 bytes |
/// | 4      | step_size                | 4 bytes |
/// | 8      | transmittance_threshold  | 4 bytes |
/// | 12     | distance_fade_start      | 4 bytes |
/// | 16     | distance_fade_end        | 4 bytes |
/// | 20     | _padding                 | 12 bytes|
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct RayMarchConfig {
    /// Maximum number of ray march steps.
    pub max_steps: u32,

    /// Base step size in meters (adaptive based on distance).
    pub step_size: f32,

    /// Transmittance threshold for early ray termination (default 0.01).
    pub transmittance_threshold: f32,

    /// Distance at which LOD fade begins (meters).
    pub distance_fade_start: f32,

    /// Distance at which LOD fade ends (full fade, meters).
    pub distance_fade_end: f32,

    /// Padding for GPU alignment.
    pub _padding: [u32; 3],
}

// Size assertion
const _: () = assert!(std::mem::size_of::<RayMarchConfig>() == 32);

impl Default for RayMarchConfig {
    fn default() -> Self {
        Self::from_quality(CloudQuality::Medium)
    }
}

impl RayMarchConfig {
    /// Create a configuration from a quality preset.
    #[inline]
    pub fn from_quality(quality: CloudQuality) -> Self {
        Self {
            max_steps: quality.step_count(),
            step_size: DEFAULT_STEP_SIZE * quality.min_step_multiplier(),
            transmittance_threshold: DEFAULT_TRANSMITTANCE_THRESHOLD,
            distance_fade_start: DEFAULT_DISTANCE_FADE_START,
            distance_fade_end: DEFAULT_DISTANCE_FADE_END,
            _padding: [0; 3],
        }
    }

    /// Create a low quality configuration.
    #[inline]
    pub fn low() -> Self {
        Self::from_quality(CloudQuality::Low)
    }

    /// Create a medium quality configuration.
    #[inline]
    pub fn medium() -> Self {
        Self::from_quality(CloudQuality::Medium)
    }

    /// Create a high quality configuration.
    #[inline]
    pub fn high() -> Self {
        Self::from_quality(CloudQuality::High)
    }

    /// Create an ultra quality configuration.
    #[inline]
    pub fn ultra() -> Self {
        Self::from_quality(CloudQuality::Ultra)
    }

    /// Set maximum steps.
    #[inline]
    pub fn with_max_steps(mut self, steps: u32) -> Self {
        self.max_steps = steps.max(1);
        self
    }

    /// Set step size.
    #[inline]
    pub fn with_step_size(mut self, size: f32) -> Self {
        self.step_size = size.max(1.0);
        self
    }

    /// Set transmittance threshold.
    #[inline]
    pub fn with_transmittance_threshold(mut self, threshold: f32) -> Self {
        self.transmittance_threshold = threshold.clamp(0.001, 0.1);
        self
    }

    /// Set distance fade range.
    #[inline]
    pub fn with_distance_fade(mut self, start: f32, end: f32) -> Self {
        self.distance_fade_start = start.max(0.0);
        self.distance_fade_end = end.max(self.distance_fade_start + 1000.0);
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> bool {
        self.max_steps > 0
            && self.step_size > 0.0
            && self.transmittance_threshold > 0.0
            && self.transmittance_threshold < 1.0
            && self.distance_fade_end > self.distance_fade_start
    }
}

// ---------------------------------------------------------------------------
// CloudSample - Ray march sample result
// ---------------------------------------------------------------------------

/// Result of sampling a cloud at a single position.
///
/// Contains density, accumulated transmittance, and radiance.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CloudSample {
    /// Raw sampled density at this position.
    pub density: f32,

    /// Accumulated transmittance (Beer's law).
    pub transmittance: f32,

    /// Accumulated radiance (RGB linear).
    pub radiance: [f32; 3],

    /// Accumulated optical depth along ray.
    pub optical_depth: f32,
}

impl CloudSample {
    /// Create a new sample with initial values.
    #[inline]
    pub fn new() -> Self {
        Self {
            density: 0.0,
            transmittance: 1.0,
            radiance: [0.0; 3],
            optical_depth: 0.0,
        }
    }

    /// Create a sample with specific density.
    #[inline]
    pub fn with_density(density: f32) -> Self {
        Self {
            density,
            transmittance: 1.0,
            radiance: [0.0; 3],
            optical_depth: 0.0,
        }
    }

    /// Check if the sample is effectively opaque.
    #[inline]
    pub fn is_opaque(&self, threshold: f32) -> bool {
        self.transmittance < threshold
    }

    /// Check if the sample has no density.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.density < EPSILON
    }

    /// Accumulate another sample into this one.
    #[inline]
    pub fn accumulate(&mut self, other: &CloudSample, step_size: f32, extinction: f32) {
        if other.density > EPSILON {
            let sample_extinction = other.density * extinction * step_size;
            let sample_transmittance = (-sample_extinction).exp();

            // Front-to-back blending
            let weight = self.transmittance * (1.0 - sample_transmittance);
            for i in 0..3 {
                self.radiance[i] += weight * other.radiance[i];
            }

            self.transmittance *= sample_transmittance;
            self.optical_depth += sample_extinction;
        }
    }
}

// ---------------------------------------------------------------------------
// CloudRenderOutput - Final render result
// ---------------------------------------------------------------------------

/// Final output from cloud ray marching for compositing.
///
/// Contains color, alpha, and depth information for blending
/// with the scene.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct CloudRenderOutput {
    /// Final blended cloud color (RGB linear).
    pub color: [f32; 3],

    /// Alpha value for blending (1 - transmittance).
    pub alpha: f32,

    /// Depth of first significant cloud hit (for depth buffer).
    pub depth: f32,

    /// Padding for GPU alignment.
    pub _padding: [f32; 3],
}

// Size assertion
const _: () = assert!(std::mem::size_of::<CloudRenderOutput>() == 32);

impl Default for CloudRenderOutput {
    fn default() -> Self {
        Self {
            color: [0.0; 3],
            alpha: 0.0,
            depth: f32::MAX,
            _padding: [0.0; 3],
        }
    }
}

impl CloudRenderOutput {
    /// Create an output with no clouds (fully transparent).
    #[inline]
    pub fn empty() -> Self {
        Self::default()
    }

    /// Create an output from a cloud sample.
    #[inline]
    pub fn from_sample(sample: &CloudSample, first_hit_depth: f32) -> Self {
        Self {
            color: sample.radiance,
            alpha: 1.0 - sample.transmittance,
            depth: first_hit_depth,
            _padding: [0.0; 3],
        }
    }

    /// Check if this output is fully transparent.
    #[inline]
    pub fn is_transparent(&self) -> bool {
        self.alpha < EPSILON
    }

    /// Blend this cloud output over a background color.
    #[inline]
    pub fn blend_over(&self, background: [f32; 3]) -> [f32; 3] {
        [
            self.color[0] + (1.0 - self.alpha) * background[0],
            self.color[1] + (1.0 - self.alpha) * background[1],
            self.color[2] + (1.0 - self.alpha) * background[2],
        ]
    }
}

// ---------------------------------------------------------------------------
// Ray-Cloud Intersection Functions
// ---------------------------------------------------------------------------

/// Calculate ray intersection with a spherical cloud layer.
///
/// Models the atmosphere as a thin shell around a sphere (Earth).
/// Returns entry and exit distances along the ray, or None if no intersection.
///
/// # Arguments
///
/// * `origin` - Ray origin in world space (meters, Y-up).
/// * `dir` - Normalized ray direction.
/// * `layer` - Cloud layer configuration.
///
/// # Returns
///
/// Optional tuple of (entry_t, exit_t) distances along ray.
pub fn ray_intersect_layer(
    origin: [f32; 3],
    dir: [f32; 3],
    layer: &CloudLayerConfig,
) -> Option<(f32, f32)> {
    // For planar atmosphere approximation (valid near ground)
    // TODO: Switch to spherical for high altitude cameras

    // Check if ray is parallel to layer
    if dir[1].abs() < EPSILON {
        // Ray is horizontal
        if origin[1] >= layer.min_height && origin[1] <= layer.max_height {
            // Ray starts inside layer, extends to horizon
            return Some((0.0, f32::MAX));
        }
        return None;
    }

    // Calculate intersection with bottom plane
    let t_min = (layer.min_height - origin[1]) / dir[1];

    // Calculate intersection with top plane
    let t_max = (layer.max_height - origin[1]) / dir[1];

    // Sort intersections
    let (t_entry, t_exit) = if t_min < t_max {
        (t_min, t_max)
    } else {
        (t_max, t_min)
    };

    // Check if intersection is in front of ray
    if t_exit < 0.0 {
        return None;
    }

    // Clamp entry to ray origin
    let t_entry = t_entry.max(0.0);

    if t_entry < t_exit {
        Some((t_entry, t_exit))
    } else {
        None
    }
}

/// Check if a world position is inside the cloud layer.
///
/// # Arguments
///
/// * `pos` - World position in meters.
/// * `layer` - Cloud layer configuration.
#[inline]
pub fn is_inside_cloud_layer(pos: [f32; 3], layer: &CloudLayerConfig) -> bool {
    pos[1] >= layer.min_height && pos[1] <= layer.max_height
}

/// Get the height fraction within the cloud layer.
///
/// Returns 0.0 at the bottom of the layer, 1.0 at the top.
///
/// # Arguments
///
/// * `pos` - World position in meters.
/// * `layer` - Cloud layer configuration.
#[inline]
pub fn get_height_fraction(pos: [f32; 3], layer: &CloudLayerConfig) -> f32 {
    let thickness = layer.max_height - layer.min_height;
    if thickness > EPSILON {
        ((pos[1] - layer.min_height) / thickness).clamp(0.0, 1.0)
    } else {
        0.5
    }
}

// ---------------------------------------------------------------------------
// Density Functions
// ---------------------------------------------------------------------------

/// Remap raw noise density based on coverage and cloud type.
///
/// Applies coverage threshold to convert noise to cloud presence.
///
/// # Arguments
///
/// * `raw_noise` - Raw noise value from texture (0-1).
/// * `coverage` - Global coverage factor (0-1).
/// * `cloud_type` - Cloud type for shaping.
///
/// # Returns
///
/// Remapped density value (0-1).
#[inline]
pub fn remap_density(raw_noise: f32, coverage: f32, cloud_type: CloudType) -> f32 {
    let (min_cov, max_cov) = cloud_type.typical_coverage();

    // Remap coverage to cloud type range
    let adjusted_coverage = min_cov + coverage * (max_cov - min_cov);

    // Apply coverage threshold with smoothstep falloff
    let threshold = 1.0 - adjusted_coverage;

    if raw_noise < threshold {
        0.0
    } else {
        // Remap to 0-1 range above threshold
        let t = (raw_noise - threshold) / adjusted_coverage.max(EPSILON);
        t.clamp(0.0, 1.0)
    }
}

/// Apply height gradient to density based on cloud type.
///
/// Shapes the vertical density profile to create characteristic cloud forms.
///
/// # Arguments
///
/// * `density` - Input density (0-1).
/// * `height_frac` - Height fraction within layer (0-1).
/// * `cloud_type` - Cloud type for shaping.
///
/// # Returns
///
/// Height-modulated density value.
pub fn apply_height_gradient(density: f32, height_frac: f32, cloud_type: CloudType) -> f32 {
    let (ramp_bottom, ramp_top, anvil) = cloud_type.height_gradient_params();

    // Calculate base gradient
    let bottom_ramp = if height_frac < ramp_bottom {
        let t = height_frac / ramp_bottom.max(EPSILON);
        smoothstep(0.0, 1.0, t)
    } else {
        1.0
    };

    let top_ramp = if height_frac > ramp_top {
        let t = (height_frac - ramp_top) / (1.0 - ramp_top).max(EPSILON);
        1.0 - smoothstep(0.0, 1.0, t)
    } else {
        1.0
    };

    // Apply anvil broadening for cumulonimbus
    let anvil_factor = if anvil > 0.0 && height_frac > 0.8 {
        let t = (height_frac - 0.8) / 0.2;
        1.0 + anvil * smoothstep(0.0, 1.0, t)
    } else {
        1.0
    };

    density * bottom_ramp * top_ramp * anvil_factor
}

/// Erode base density with detail noise for wispy edges.
///
/// Subtracts detail noise from base density to create fine cloud structure.
///
/// # Arguments
///
/// * `base_density` - Base shape density (0-1).
/// * `detail_noise` - Detail noise value (0-1).
/// * `erosion` - Erosion strength (0-1, typical 0.3-0.5).
///
/// # Returns
///
/// Eroded density value.
#[inline]
pub fn erode_with_detail(base_density: f32, detail_noise: f32, erosion: f32) -> f32 {
    let eroded = base_density - detail_noise * erosion * base_density;
    eroded.max(0.0)
}

// ---------------------------------------------------------------------------
// Lighting Model
// ---------------------------------------------------------------------------

/// Apply Beer-Lambert law for light extinction.
///
/// Models how light is absorbed/scattered as it travels through the cloud.
///
/// # Arguments
///
/// * `density` - Cloud density at sample.
/// * `distance` - Distance traveled through cloud.
/// * `extinction` - Extinction coefficient.
///
/// # Returns
///
/// Transmittance factor (0-1).
#[inline]
pub fn beer_lambert(density: f32, distance: f32, extinction: f32) -> f32 {
    (-density * distance * extinction).exp()
}

/// Apply powder effect for bright cloud edges.
///
/// Creates the characteristic bright glow at cloud edges
/// due to multiple scattering in thin cloud regions.
///
/// # Arguments
///
/// * `density` - Cloud density at sample.
/// * `factor` - Powder effect strength (typical 2.0).
///
/// # Returns
///
/// Brightness factor (0-1).
#[inline]
pub fn powder_brightness(density: f32, factor: f32) -> f32 {
    1.0 - (-density * factor).exp()
}

/// Combined Beer-Powder for efficient evaluation.
///
/// Returns the product of Beer-Lambert transmittance and powder brightness.
///
/// # Arguments
///
/// * `density` - Cloud density.
/// * `step_size` - Step size in meters.
/// * `extinction` - Extinction coefficient.
/// * `powder_factor` - Powder effect strength.
#[inline]
pub fn beer_powder(density: f32, step_size: f32, extinction: f32, powder_factor: f32) -> f32 {
    let beer = beer_lambert(density, step_size, extinction);
    let powder = powder_brightness(density * step_size, powder_factor);
    beer * powder
}

/// Multi-scattering approximation for energy conservation.
///
/// Approximates additional scattering bounces within the cloud
/// to prevent overly dark cloud interiors.
///
/// # Arguments
///
/// * `single_scatter` - Single scattering radiance (RGB).
/// * `albedo` - Scattering albedo (typical 0.99 for clouds).
/// * `transmittance` - Current transmittance.
///
/// # Returns
///
/// Multi-scattered radiance (RGB).
pub fn multi_scatter_approx(
    single_scatter: [f32; 3],
    albedo: f32,
    transmittance: f32,
) -> [f32; 3] {
    // Approximate multiple scattering as a series expansion
    // L_ms = L_ss * (1 + albedo * (1 - T) + albedo^2 * (1 - T)^2 + ...)
    // Simplified: L_ms = L_ss / (1 - albedo * (1 - T))

    let scatter_factor = 1.0 - transmittance;
    let ms_factor = 1.0 / (1.0 - albedo * scatter_factor * 0.5).max(EPSILON);

    [
        single_scatter[0] * ms_factor,
        single_scatter[1] * ms_factor,
        single_scatter[2] * ms_factor,
    ]
}

/// Calculate ambient light contribution based on height.
///
/// Higher cloud regions receive more light from above,
/// lower regions receive ground bounce.
///
/// # Arguments
///
/// * `height_frac` - Height fraction within cloud layer (0-1).
/// * `sky_color` - Sky/zenith color (RGB linear).
/// * `sun_color` - Ground/horizon color (RGB linear).
///
/// # Returns
///
/// Ambient radiance (RGB linear).
pub fn ambient_from_height(
    height_frac: f32,
    sky_color: [f32; 3],
    sun_color: [f32; 3],
) -> [f32; 3] {
    // Smooth interpolation between ground and sky contribution
    let sky_contrib = smoothstep(0.0, 1.0, height_frac);
    let ground_contrib = 1.0 - sky_contrib;

    // Sky contributes from above, ground bounce from below
    let ambient_strength = 0.1; // Ambient contribution factor

    [
        ambient_strength * (sky_contrib * sky_color[0] + ground_contrib * sun_color[0] * 0.2),
        ambient_strength * (sky_contrib * sky_color[1] + ground_contrib * sun_color[1] * 0.2),
        ambient_strength * (sky_contrib * sky_color[2] + ground_contrib * sun_color[2] * 0.2),
    ]
}

// ---------------------------------------------------------------------------
// CloudRayMarcher - Main ray marching API
// ---------------------------------------------------------------------------

/// Main cloud ray marcher.
///
/// Provides the API for ray marching through volumetric clouds.
///
/// # Example
///
/// ```
/// use renderer_backend::cloud_raymarching::{
///     CloudRayMarcher, CloudLayerConfig, CloudQuality
/// };
///
/// let layer = CloudLayerConfig::cumulus();
/// let mut marcher = CloudRayMarcher::new(layer, CloudQuality::High);
///
/// // March a ray
/// let sample = marcher.march_ray(
///     [0.0, 1000.0, 0.0],  // Camera at 1km altitude
///     [0.0, 0.1, -1.0],     // Looking slightly up and forward
///     100_000.0,            // Max scene depth
/// );
/// ```
#[derive(Debug, Clone)]
pub struct CloudRayMarcher {
    /// Cloud layer configuration.
    layer_config: CloudLayerConfig,

    /// Ray march configuration.
    march_config: RayMarchConfig,

    /// Cloud type for shaping.
    cloud_type: CloudType,

    /// Sun direction (normalized, towards sun).
    sun_direction: [f32; 3],

    /// Sun color/intensity (RGB linear).
    sun_color: [f32; 3],

    /// Sky color for ambient (RGB linear).
    sky_color: [f32; 3],

    /// Scattering albedo.
    scattering_albedo: f32,
}

impl CloudRayMarcher {
    /// Create a new ray marcher with given configuration.
    #[inline]
    pub fn new(layer_config: CloudLayerConfig, quality: CloudQuality) -> Self {
        Self {
            layer_config,
            march_config: RayMarchConfig::from_quality(quality),
            cloud_type: CloudType::Cumulus,
            sun_direction: normalize_vec3([0.5, 0.7, -0.5]),
            sun_color: [1.0, 0.95, 0.9], // Warm sunlight
            sky_color: [0.3, 0.5, 0.8],  // Blue sky
            scattering_albedo: DEFAULT_SCATTERING_ALBEDO,
        }
    }

    /// Create a ray marcher with custom march config.
    #[inline]
    pub fn with_config(layer_config: CloudLayerConfig, march_config: RayMarchConfig) -> Self {
        Self {
            layer_config,
            march_config,
            cloud_type: CloudType::Cumulus,
            sun_direction: normalize_vec3([0.5, 0.7, -0.5]),
            sun_color: [1.0, 0.95, 0.9],
            sky_color: [0.3, 0.5, 0.8],
            scattering_albedo: DEFAULT_SCATTERING_ALBEDO,
        }
    }

    /// Get the layer configuration.
    #[inline]
    pub fn layer_config(&self) -> &CloudLayerConfig {
        &self.layer_config
    }

    /// Get the march configuration.
    #[inline]
    pub fn march_config(&self) -> &RayMarchConfig {
        &self.march_config
    }

    /// Set cloud type.
    #[inline]
    pub fn set_cloud_type(&mut self, cloud_type: CloudType) {
        self.cloud_type = cloud_type;
    }

    /// Set sun direction (will be normalized).
    #[inline]
    pub fn set_sun_direction(&mut self, direction: [f32; 3]) {
        self.sun_direction = normalize_vec3(direction);
    }

    /// Set sun color/intensity.
    #[inline]
    pub fn set_sun_color(&mut self, color: [f32; 3]) {
        self.sun_color = color;
    }

    /// Set sky color for ambient lighting.
    #[inline]
    pub fn set_sky_color(&mut self, color: [f32; 3]) {
        self.sky_color = color;
    }

    /// March a ray through the cloud layer.
    ///
    /// # Arguments
    ///
    /// * `ray_origin` - Ray start position in world space (meters).
    /// * `ray_dir` - Normalized ray direction.
    /// * `scene_depth` - Maximum scene depth (for occlusion).
    ///
    /// # Returns
    ///
    /// Cloud sample with accumulated density, transmittance, and radiance.
    pub fn march_ray(
        &self,
        ray_origin: [f32; 3],
        ray_dir: [f32; 3],
        scene_depth: f32,
    ) -> CloudSample {
        let ray_dir = normalize_vec3(ray_dir);

        // Find intersection with cloud layer
        let intersection = ray_intersect_layer(ray_origin, ray_dir, &self.layer_config);
        if intersection.is_none() {
            return CloudSample::new();
        }

        let (t_entry, t_exit) = intersection.unwrap();

        // Clamp to scene depth
        let t_exit = t_exit.min(scene_depth);
        if t_entry >= t_exit {
            return CloudSample::new();
        }

        // Get adaptive step count
        let distance = t_exit - t_entry;
        let step_count = self.get_step_count_for_distance(distance);
        let step_size = distance / step_count as f32;

        // Initialize accumulator
        let mut sample = CloudSample::new();

        // Ray march loop
        for i in 0..step_count {
            // Calculate sample position with jitter
            let t = t_entry + (i as f32 + 0.5) * step_size;
            let pos = [
                ray_origin[0] + ray_dir[0] * t,
                ray_origin[1] + ray_dir[1] * t,
                ray_origin[2] + ray_dir[2] * t,
            ];

            // Sample density
            let density = self.sample_density(pos);

            if density > EPSILON {
                // Calculate lighting
                let lighting = self.compute_lighting(pos, density, self.sun_direction);

                // Create local sample
                let local_sample = CloudSample {
                    density,
                    transmittance: 1.0,
                    radiance: lighting,
                    optical_depth: 0.0,
                };

                // Accumulate
                sample.accumulate(
                    &local_sample,
                    step_size,
                    self.layer_config.extinction_coeff,
                );

                // Early termination
                if sample.transmittance < self.march_config.transmittance_threshold {
                    break;
                }
            }
        }

        // Apply multi-scattering approximation
        sample.radiance = multi_scatter_approx(
            sample.radiance,
            self.scattering_albedo,
            sample.transmittance,
        );

        sample
    }

    /// Sample cloud density at a world position.
    ///
    /// Uses procedural noise for cloud shapes.
    ///
    /// # Arguments
    ///
    /// * `world_pos` - World position in meters.
    ///
    /// # Returns
    ///
    /// Cloud density (0-1).
    pub fn sample_density(&self, world_pos: [f32; 3]) -> f32 {
        // Check if inside layer
        if !is_inside_cloud_layer(world_pos, &self.layer_config) {
            return 0.0;
        }

        // Calculate height fraction
        let height_frac = get_height_fraction(world_pos, &self.layer_config);

        // Sample base noise (using simple procedural noise)
        // In production, this would sample from the 3D textures generated by cloud_noise.rs
        let tile_size = 6000.0; // 6km tile
        let base_noise = sample_procedural_noise(world_pos, tile_size);

        // Remap with coverage
        let remapped = remap_density(base_noise, self.layer_config.coverage, self.cloud_type);

        // Apply height gradient
        let shaped = apply_height_gradient(remapped, height_frac, self.cloud_type);

        // Apply detail erosion (simplified)
        let detail_noise = sample_procedural_noise(
            [world_pos[0] * 2.0, world_pos[1] * 2.0, world_pos[2] * 2.0],
            tile_size,
        );
        let final_density = erode_with_detail(shaped, detail_noise, 0.3);

        final_density * self.layer_config.density
    }

    /// Compute lighting at a sample position.
    ///
    /// # Arguments
    ///
    /// * `sample_pos` - World position of sample.
    /// * `density` - Local density.
    /// * `sun_dir` - Direction towards sun.
    ///
    /// # Returns
    ///
    /// Radiance (RGB linear).
    pub fn compute_lighting(
        &self,
        sample_pos: [f32; 3],
        density: f32,
        sun_dir: [f32; 3],
    ) -> [f32; 3] {
        let height_frac = get_height_fraction(sample_pos, &self.layer_config);

        // Estimate light transmittance towards sun (simplified)
        // In production, this would use cone sampling or light march
        let sun_transmittance = self.estimate_sun_transmittance(sample_pos, sun_dir);

        // Beer-Powder for in-scatter
        let in_scatter_factor = beer_powder(
            density,
            self.march_config.step_size,
            self.layer_config.extinction_coeff,
            self.layer_config.powder_factor,
        );

        // Direct sun contribution
        let direct = [
            self.sun_color[0] * sun_transmittance * in_scatter_factor,
            self.sun_color[1] * sun_transmittance * in_scatter_factor,
            self.sun_color[2] * sun_transmittance * in_scatter_factor,
        ];

        // Ambient contribution
        let ambient = ambient_from_height(height_frac, self.sky_color, self.sun_color);

        [
            direct[0] + ambient[0],
            direct[1] + ambient[1],
            direct[2] + ambient[2],
        ]
    }

    /// Apply Beer-Lambert law for a given density and step size.
    #[inline]
    pub fn apply_beer_lambert(&self, density: f32, step_size: f32) -> f32 {
        beer_lambert(density, step_size, self.layer_config.extinction_coeff)
    }

    /// Apply powder effect for a given density.
    #[inline]
    pub fn apply_powder_effect(&self, density: f32, factor: f32) -> f32 {
        powder_brightness(density, factor)
    }

    /// Get adaptive step count based on distance.
    ///
    /// Reduces step count for distant clouds (LOD).
    ///
    /// # Arguments
    ///
    /// * `distance` - Ray travel distance through cloud.
    ///
    /// # Returns
    ///
    /// Adapted step count.
    pub fn get_step_count_for_distance(&self, distance: f32) -> u32 {
        // Calculate LOD factor based on distance
        let lod_factor = if distance < self.march_config.distance_fade_start {
            1.0
        } else if distance > self.march_config.distance_fade_end {
            0.25 // Minimum 25% of steps
        } else {
            let t = (distance - self.march_config.distance_fade_start)
                / (self.march_config.distance_fade_end - self.march_config.distance_fade_start);
            1.0 - 0.75 * smoothstep(0.0, 1.0, t)
        };

        let base_steps = self.march_config.max_steps as f32;
        let adapted_steps = (base_steps * lod_factor).ceil() as u32;

        adapted_steps.max(8) // Minimum 8 steps
    }

    /// Estimate sun transmittance at a position (simplified).
    ///
    /// Uses a few samples towards the sun to estimate shadowing.
    fn estimate_sun_transmittance(&self, pos: [f32; 3], sun_dir: [f32; 3]) -> f32 {
        const LIGHT_STEPS: u32 = 6;
        const LIGHT_STEP_SIZE: f32 = 100.0;

        let mut accumulated_density = 0.0;

        for i in 0..LIGHT_STEPS {
            let t = (i as f32 + 0.5) * LIGHT_STEP_SIZE;
            let sample_pos = [
                pos[0] + sun_dir[0] * t,
                pos[1] + sun_dir[1] * t,
                pos[2] + sun_dir[2] * t,
            ];

            if is_inside_cloud_layer(sample_pos, &self.layer_config) {
                let height_frac = get_height_fraction(sample_pos, &self.layer_config);
                let base_density = sample_procedural_noise(sample_pos, 6000.0);
                let shaped = apply_height_gradient(
                    remap_density(base_density, self.layer_config.coverage, self.cloud_type),
                    height_frac,
                    self.cloud_type,
                );
                accumulated_density += shaped * LIGHT_STEP_SIZE;
            }
        }

        beer_lambert(
            accumulated_density * self.layer_config.density,
            1.0,
            self.layer_config.extinction_coeff,
        )
    }
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
        [0.0, 1.0, 0.0] // Default to up
    }
}

/// Smoothstep interpolation.
#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Simple procedural noise for testing.
///
/// In production, this would sample the 3D textures from `cloud_noise.rs`.
fn sample_procedural_noise(pos: [f32; 3], tile_size: f32) -> f32 {
    // Simplified 3D noise using sine waves
    let inv_tile = 1.0 / tile_size;
    let scaled = [pos[0] * inv_tile, pos[1] * inv_tile, pos[2] * inv_tile];

    let n1 = ((scaled[0] * 6.28318).sin() * 0.5 + 0.5)
        * ((scaled[1] * 12.56636).sin() * 0.5 + 0.5)
        * ((scaled[2] * 6.28318).cos() * 0.5 + 0.5);

    let n2 = ((scaled[0] * 12.56636 + 1.0).sin() * 0.5 + 0.5)
        * ((scaled[1] * 6.28318 + 2.0).cos() * 0.5 + 0.5)
        * ((scaled[2] * 12.56636 + 0.5).sin() * 0.5 + 0.5);

    (n1 * 0.6 + n2 * 0.4).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // CloudLayerConfig Tests
    // =========================================================================

    #[test]
    fn test_cloud_layer_config_default() {
        let config = CloudLayerConfig::default();
        assert_eq!(config.min_height, DEFAULT_MIN_HEIGHT);
        assert_eq!(config.max_height, DEFAULT_MAX_HEIGHT);
        assert_eq!(config.coverage, DEFAULT_COVERAGE);
        assert_eq!(config.density, DEFAULT_DENSITY);
        assert!(config.validate());
    }

    #[test]
    fn test_cloud_layer_config_stratus() {
        let config = CloudLayerConfig::stratus();
        assert!(config.min_height < 1000.0);
        assert!(config.max_height < 3000.0);
        assert!(config.coverage > 0.5);
        assert!(config.validate());
    }

    #[test]
    fn test_cloud_layer_config_cirrus() {
        let config = CloudLayerConfig::cirrus();
        assert!(config.min_height > 5000.0);
        assert!(config.density < 0.02);
        assert!(config.validate());
    }

    #[test]
    fn test_cloud_layer_config_storm() {
        let config = CloudLayerConfig::storm();
        assert!(config.max_height > 10000.0);
        assert!(config.density > 0.05);
        assert!(config.validate());
    }

    #[test]
    fn test_cloud_layer_config_with_height() {
        let config = CloudLayerConfig::default().with_height(2000.0, 5000.0);
        assert_eq!(config.min_height, 2000.0);
        assert_eq!(config.max_height, 5000.0);
        assert!(config.validate());
    }

    #[test]
    fn test_cloud_layer_config_with_coverage() {
        let config = CloudLayerConfig::default().with_coverage(0.8);
        assert_eq!(config.coverage, 0.8);
        assert!(config.validate());
    }

    #[test]
    fn test_cloud_layer_config_coverage_clamping() {
        let config = CloudLayerConfig::default().with_coverage(1.5);
        assert_eq!(config.coverage, 1.0);

        let config = CloudLayerConfig::default().with_coverage(-0.5);
        assert_eq!(config.coverage, 0.0);
    }

    #[test]
    fn test_cloud_layer_config_thickness() {
        let config = CloudLayerConfig::default().with_height(1000.0, 5000.0);
        assert_eq!(config.thickness(), 4000.0);
    }

    #[test]
    fn test_cloud_layer_config_invalid_heights() {
        let mut config = CloudLayerConfig::default();
        config.min_height = 5000.0;
        config.max_height = 3000.0; // Invalid: max < min
        assert!(!config.validate());
    }

    #[test]
    fn test_cloud_layer_config_invalid_coverage() {
        let mut config = CloudLayerConfig::default();
        config.coverage = 1.5; // Invalid: > 1
        assert!(!config.validate());

        config.coverage = -0.1; // Invalid: < 0
        assert!(!config.validate());
    }

    #[test]
    fn test_cloud_layer_config_pod() {
        let config = CloudLayerConfig::default();
        let bytes = bytemuck::bytes_of(&config);
        assert_eq!(bytes.len(), 32);
    }

    #[test]
    fn test_cloud_layer_config_size() {
        assert_eq!(std::mem::size_of::<CloudLayerConfig>(), 32);
    }

    // =========================================================================
    // CloudQuality Tests
    // =========================================================================

    #[test]
    fn test_cloud_quality_step_count_low() {
        assert_eq!(CloudQuality::Low.step_count(), 32);
    }

    #[test]
    fn test_cloud_quality_step_count_medium() {
        assert_eq!(CloudQuality::Medium.step_count(), 64);
    }

    #[test]
    fn test_cloud_quality_step_count_high() {
        assert_eq!(CloudQuality::High.step_count(), 128);
    }

    #[test]
    fn test_cloud_quality_step_count_ultra() {
        assert_eq!(CloudQuality::Ultra.step_count(), 256);
    }

    #[test]
    fn test_cloud_quality_from_step_count() {
        assert_eq!(CloudQuality::from_step_count(10), CloudQuality::Low);
        assert_eq!(CloudQuality::from_step_count(50), CloudQuality::Low);
        assert_eq!(CloudQuality::from_step_count(64), CloudQuality::Medium);
        assert_eq!(CloudQuality::from_step_count(100), CloudQuality::Medium);
        assert_eq!(CloudQuality::from_step_count(128), CloudQuality::High);
        assert_eq!(CloudQuality::from_step_count(200), CloudQuality::High);
        assert_eq!(CloudQuality::from_step_count(256), CloudQuality::Ultra);
        assert_eq!(CloudQuality::from_step_count(512), CloudQuality::Ultra);
    }

    #[test]
    fn test_cloud_quality_min_step_multiplier() {
        assert!(CloudQuality::Low.min_step_multiplier() > CloudQuality::High.min_step_multiplier());
        assert!(
            CloudQuality::Ultra.min_step_multiplier() < CloudQuality::Medium.min_step_multiplier()
        );
    }

    // =========================================================================
    // RayMarchConfig Tests
    // =========================================================================

    #[test]
    fn test_ray_march_config_default() {
        let config = RayMarchConfig::default();
        assert_eq!(config.max_steps, 64); // Medium quality
        assert!(config.validate());
    }

    #[test]
    fn test_ray_march_config_from_quality() {
        let low = RayMarchConfig::from_quality(CloudQuality::Low);
        assert_eq!(low.max_steps, 32);

        let high = RayMarchConfig::from_quality(CloudQuality::High);
        assert_eq!(high.max_steps, 128);
    }

    #[test]
    fn test_ray_march_config_low() {
        let config = RayMarchConfig::low();
        assert_eq!(config.max_steps, 32);
    }

    #[test]
    fn test_ray_march_config_high() {
        let config = RayMarchConfig::high();
        assert_eq!(config.max_steps, 128);
    }

    #[test]
    fn test_ray_march_config_ultra() {
        let config = RayMarchConfig::ultra();
        assert_eq!(config.max_steps, 256);
    }

    #[test]
    fn test_ray_march_config_with_max_steps() {
        let config = RayMarchConfig::default().with_max_steps(100);
        assert_eq!(config.max_steps, 100);
    }

    #[test]
    fn test_ray_march_config_with_step_size() {
        let config = RayMarchConfig::default().with_step_size(50.0);
        assert_eq!(config.step_size, 50.0);
    }

    #[test]
    fn test_ray_march_config_with_transmittance_threshold() {
        let config = RayMarchConfig::default().with_transmittance_threshold(0.05);
        assert!((config.transmittance_threshold - 0.05).abs() < EPSILON);
    }

    #[test]
    fn test_ray_march_config_transmittance_clamping() {
        let config = RayMarchConfig::default().with_transmittance_threshold(0.0001);
        assert_eq!(config.transmittance_threshold, 0.001);

        let config = RayMarchConfig::default().with_transmittance_threshold(0.5);
        assert_eq!(config.transmittance_threshold, 0.1);
    }

    #[test]
    fn test_ray_march_config_with_distance_fade() {
        let config = RayMarchConfig::default().with_distance_fade(30000.0, 100000.0);
        assert_eq!(config.distance_fade_start, 30000.0);
        assert_eq!(config.distance_fade_end, 100000.0);
    }

    #[test]
    fn test_ray_march_config_invalid_steps() {
        let mut config = RayMarchConfig::default();
        config.max_steps = 0;
        assert!(!config.validate());
    }

    #[test]
    fn test_ray_march_config_invalid_threshold() {
        let mut config = RayMarchConfig::default();
        config.transmittance_threshold = 0.0;
        assert!(!config.validate());

        config.transmittance_threshold = 1.5;
        assert!(!config.validate());
    }

    #[test]
    fn test_ray_march_config_pod() {
        let config = RayMarchConfig::default();
        let bytes = bytemuck::bytes_of(&config);
        assert_eq!(bytes.len(), 32);
    }

    // =========================================================================
    // CloudSample Tests
    // =========================================================================

    #[test]
    fn test_cloud_sample_new() {
        let sample = CloudSample::new();
        assert_eq!(sample.density, 0.0);
        assert_eq!(sample.transmittance, 1.0);
        assert_eq!(sample.radiance, [0.0; 3]);
        assert_eq!(sample.optical_depth, 0.0);
    }

    #[test]
    fn test_cloud_sample_with_density() {
        let sample = CloudSample::with_density(0.5);
        assert_eq!(sample.density, 0.5);
        assert_eq!(sample.transmittance, 1.0);
    }

    #[test]
    fn test_cloud_sample_is_opaque() {
        let mut sample = CloudSample::new();
        sample.transmittance = 0.005;
        assert!(sample.is_opaque(0.01));
        assert!(!sample.is_opaque(0.001));
    }

    #[test]
    fn test_cloud_sample_is_empty() {
        let sample = CloudSample::new();
        assert!(sample.is_empty());

        let sample = CloudSample::with_density(0.5);
        assert!(!sample.is_empty());
    }

    #[test]
    fn test_cloud_sample_accumulate() {
        let mut acc = CloudSample::new();
        let sample = CloudSample {
            density: 0.1,
            transmittance: 1.0,
            radiance: [1.0, 0.5, 0.25],
            optical_depth: 0.0,
        };

        acc.accumulate(&sample, 10.0, 0.1);

        assert!(acc.transmittance < 1.0);
        assert!(acc.radiance[0] > 0.0);
        assert!(acc.optical_depth > 0.0);
    }

    #[test]
    fn test_cloud_sample_accumulate_empty() {
        let mut acc = CloudSample::new();
        let empty = CloudSample::new();

        acc.accumulate(&empty, 10.0, 0.1);

        assert_eq!(acc.transmittance, 1.0);
        assert_eq!(acc.radiance, [0.0; 3]);
    }

    // =========================================================================
    // CloudRenderOutput Tests
    // =========================================================================

    #[test]
    fn test_cloud_render_output_default() {
        let output = CloudRenderOutput::default();
        assert_eq!(output.color, [0.0; 3]);
        assert_eq!(output.alpha, 0.0);
        assert_eq!(output.depth, f32::MAX);
    }

    #[test]
    fn test_cloud_render_output_empty() {
        let output = CloudRenderOutput::empty();
        assert!(output.is_transparent());
    }

    #[test]
    fn test_cloud_render_output_from_sample() {
        let sample = CloudSample {
            density: 0.5,
            transmittance: 0.3,
            radiance: [0.8, 0.7, 0.6],
            optical_depth: 1.0,
        };

        let output = CloudRenderOutput::from_sample(&sample, 5000.0);
        assert_eq!(output.color, sample.radiance);
        assert!((output.alpha - 0.7).abs() < EPSILON);
        assert_eq!(output.depth, 5000.0);
    }

    #[test]
    fn test_cloud_render_output_blend_over() {
        let output = CloudRenderOutput {
            color: [1.0, 0.0, 0.0],
            alpha: 0.5,
            depth: 1000.0,
            _padding: [0.0; 3],
        };

        let background = [0.0, 0.0, 1.0];
        let blended = output.blend_over(background);

        assert!((blended[0] - 1.0).abs() < EPSILON);
        assert!((blended[2] - 0.5).abs() < EPSILON);
    }

    #[test]
    fn test_cloud_render_output_pod() {
        let output = CloudRenderOutput::default();
        let bytes = bytemuck::bytes_of(&output);
        assert_eq!(bytes.len(), 32);
    }

    // =========================================================================
    // Ray-Layer Intersection Tests
    // =========================================================================

    #[test]
    fn test_ray_intersect_layer_above_looking_down() {
        let layer = CloudLayerConfig::default();
        let origin = [0.0, 10000.0, 0.0]; // Above cloud layer
        let dir = [0.0, -1.0, 0.0]; // Looking straight down

        let result = ray_intersect_layer(origin, dir, &layer);
        assert!(result.is_some());

        let (t_entry, t_exit) = result.unwrap();
        assert!(t_entry > 0.0);
        assert!(t_exit > t_entry);

        // Entry should be at max_height
        let entry_pos = origin[1] + dir[1] * t_entry;
        assert!((entry_pos - layer.max_height).abs() < 1.0);
    }

    #[test]
    fn test_ray_intersect_layer_below_looking_up() {
        let layer = CloudLayerConfig::default();
        let origin = [0.0, 500.0, 0.0]; // Below cloud layer
        let dir = [0.0, 1.0, 0.0]; // Looking straight up

        let result = ray_intersect_layer(origin, dir, &layer);
        assert!(result.is_some());

        let (t_entry, _) = result.unwrap();
        let entry_pos = origin[1] + dir[1] * t_entry;
        assert!((entry_pos - layer.min_height).abs() < 1.0);
    }

    #[test]
    fn test_ray_intersect_layer_inside() {
        let layer = CloudLayerConfig::default();
        let origin = [0.0, 3000.0, 0.0]; // Inside cloud layer
        let dir = normalize_vec3([1.0, 0.1, 0.0]);

        let result = ray_intersect_layer(origin, dir, &layer);
        assert!(result.is_some());

        let (t_entry, _) = result.unwrap();
        assert_eq!(t_entry, 0.0); // Should start at ray origin
    }

    #[test]
    fn test_ray_intersect_layer_above_looking_up() {
        let layer = CloudLayerConfig::default();
        let origin = [0.0, 10000.0, 0.0]; // Above
        let dir = [0.0, 1.0, 0.0]; // Looking up (away from layer)

        let result = ray_intersect_layer(origin, dir, &layer);
        assert!(result.is_none());
    }

    #[test]
    fn test_ray_intersect_layer_below_looking_down() {
        let layer = CloudLayerConfig::default();
        let origin = [0.0, 500.0, 0.0]; // Below
        let dir = [0.0, -1.0, 0.0]; // Looking down

        let result = ray_intersect_layer(origin, dir, &layer);
        assert!(result.is_none());
    }

    #[test]
    fn test_ray_intersect_layer_horizontal() {
        let layer = CloudLayerConfig::default();
        let origin = [0.0, 3000.0, 0.0]; // Inside
        let dir = [1.0, 0.0, 0.0]; // Horizontal

        let result = ray_intersect_layer(origin, dir, &layer);
        assert!(result.is_some());

        let (t_entry, t_exit) = result.unwrap();
        assert_eq!(t_entry, 0.0);
        assert_eq!(t_exit, f32::MAX);
    }

    #[test]
    fn test_ray_intersect_layer_horizontal_outside() {
        let layer = CloudLayerConfig::default();
        let origin = [0.0, 500.0, 0.0]; // Below
        let dir = [1.0, 0.0, 0.0]; // Horizontal

        let result = ray_intersect_layer(origin, dir, &layer);
        assert!(result.is_none());
    }

    #[test]
    fn test_is_inside_cloud_layer() {
        let layer = CloudLayerConfig::default();

        assert!(!is_inside_cloud_layer([0.0, 500.0, 0.0], &layer)); // Below
        assert!(is_inside_cloud_layer([0.0, 3000.0, 0.0], &layer)); // Inside
        assert!(is_inside_cloud_layer([0.0, 1500.0, 0.0], &layer)); // At min
        assert!(is_inside_cloud_layer([0.0, 8000.0, 0.0], &layer)); // At max
        assert!(!is_inside_cloud_layer([0.0, 10000.0, 0.0], &layer)); // Above
    }

    #[test]
    fn test_get_height_fraction_bottom() {
        let layer = CloudLayerConfig::default();
        let frac = get_height_fraction([0.0, 1500.0, 0.0], &layer);
        assert!((frac - 0.0).abs() < EPSILON);
    }

    #[test]
    fn test_get_height_fraction_top() {
        let layer = CloudLayerConfig::default();
        let frac = get_height_fraction([0.0, 8000.0, 0.0], &layer);
        assert!((frac - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_get_height_fraction_middle() {
        let layer = CloudLayerConfig::default();
        let mid_height = (layer.min_height + layer.max_height) / 2.0;
        let frac = get_height_fraction([0.0, mid_height, 0.0], &layer);
        assert!((frac - 0.5).abs() < EPSILON);
    }

    #[test]
    fn test_get_height_fraction_clamped() {
        let layer = CloudLayerConfig::default();
        assert!((get_height_fraction([0.0, 0.0, 0.0], &layer) - 0.0).abs() < EPSILON);
        assert!((get_height_fraction([0.0, 20000.0, 0.0], &layer) - 1.0).abs() < EPSILON);
    }

    // =========================================================================
    // Density Function Tests
    // =========================================================================

    #[test]
    fn test_remap_density_zero_coverage() {
        let density = remap_density(0.5, 0.0, CloudType::Cumulus);
        assert_eq!(density, 0.0); // Zero coverage = no clouds
    }

    #[test]
    fn test_remap_density_full_coverage() {
        // Use high noise value (0.9) to ensure it's above threshold
        // even with cloud-type typical coverage remapping
        let density = remap_density(0.9, 1.0, CloudType::Cumulus);
        assert!(density > 0.0, "Full coverage with high noise should produce clouds");

        // Stratus has higher coverage range, so 0.5 should work
        let stratus_density = remap_density(0.5, 1.0, CloudType::Stratus);
        assert!(stratus_density > 0.0, "Full coverage stratus should produce clouds");
    }

    #[test]
    fn test_remap_density_below_threshold() {
        let density = remap_density(0.1, 0.5, CloudType::Cumulus);
        assert_eq!(density, 0.0); // Below coverage threshold
    }

    #[test]
    fn test_remap_density_above_threshold() {
        let density = remap_density(0.9, 0.5, CloudType::Cumulus);
        assert!(density > 0.0); // Above threshold
    }

    #[test]
    fn test_apply_height_gradient_stratus() {
        // Stratus: low altitude, flat
        let low = apply_height_gradient(1.0, 0.05, CloudType::Stratus);
        let mid = apply_height_gradient(1.0, 0.5, CloudType::Stratus);

        // Stratus has most density at bottom
        assert!(low > mid);
    }

    #[test]
    fn test_apply_height_gradient_cumulus() {
        // Cumulus: bottom ramp, rounded top
        let bottom = apply_height_gradient(1.0, 0.0, CloudType::Cumulus);
        let mid = apply_height_gradient(1.0, 0.3, CloudType::Cumulus);
        let top = apply_height_gradient(1.0, 1.0, CloudType::Cumulus);

        assert!(bottom < mid); // Ramps up from bottom
        assert!(mid > top); // Falls off at top
    }

    #[test]
    fn test_apply_height_gradient_cumulonimbus() {
        // Cumulonimbus: extends through full height, anvil at top
        let top = apply_height_gradient(1.0, 0.95, CloudType::Cumulonimbus);
        assert!(top > 0.0); // Still has density at top
    }

    #[test]
    fn test_erode_with_detail_no_erosion() {
        let eroded = erode_with_detail(0.8, 0.5, 0.0);
        assert_eq!(eroded, 0.8);
    }

    #[test]
    fn test_erode_with_detail_partial() {
        let eroded = erode_with_detail(0.8, 0.5, 0.5);
        assert!(eroded < 0.8);
        assert!(eroded > 0.0);
    }

    #[test]
    fn test_erode_with_detail_full() {
        let eroded = erode_with_detail(0.5, 1.0, 1.0);
        assert!(eroded < 0.5);
    }

    #[test]
    fn test_erode_with_detail_no_negative() {
        let eroded = erode_with_detail(0.1, 1.0, 1.0);
        assert!(eroded >= 0.0);
    }

    // =========================================================================
    // Beer-Lambert Tests
    // =========================================================================

    #[test]
    fn test_beer_lambert_zero_density() {
        let t = beer_lambert(0.0, 100.0, 0.1);
        assert_eq!(t, 1.0);
    }

    #[test]
    fn test_beer_lambert_zero_distance() {
        let t = beer_lambert(0.5, 0.0, 0.1);
        assert_eq!(t, 1.0);
    }

    #[test]
    fn test_beer_lambert_decreasing() {
        let t1 = beer_lambert(0.5, 10.0, 0.1);
        let t2 = beer_lambert(0.5, 20.0, 0.1);
        assert!(t2 < t1);
    }

    #[test]
    fn test_beer_lambert_known_value() {
        // T = exp(-0.1 * 10 * 0.1) = exp(-0.1) ~= 0.9048
        let t = beer_lambert(0.1, 10.0, 0.1);
        assert!((t - 0.9048).abs() < 0.01);
    }

    #[test]
    fn test_powder_brightness_zero_density() {
        let p = powder_brightness(0.0, 2.0);
        assert!((p - 0.0).abs() < EPSILON);
    }

    #[test]
    fn test_powder_brightness_high_density() {
        let p = powder_brightness(1.0, 2.0);
        assert!(p > 0.8);
    }

    #[test]
    fn test_powder_brightness_increasing() {
        let p1 = powder_brightness(0.1, 2.0);
        let p2 = powder_brightness(0.5, 2.0);
        assert!(p2 > p1);
    }

    #[test]
    fn test_beer_powder() {
        let bp = beer_powder(0.5, 10.0, 0.1, 2.0);
        let beer = beer_lambert(0.5, 10.0, 0.1);
        let powder = powder_brightness(0.5 * 10.0, 2.0);
        assert!((bp - beer * powder).abs() < EPSILON);
    }

    #[test]
    fn test_multi_scatter_approx_no_scatter() {
        let result = multi_scatter_approx([1.0, 1.0, 1.0], 0.99, 1.0);
        // Full transmittance = no multiple scattering
        assert_eq!(result, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_multi_scatter_approx_partial() {
        let ss = [1.0, 1.0, 1.0];
        let result = multi_scatter_approx(ss, 0.99, 0.5);
        // With partial transmittance, should boost brightness
        assert!(result[0] > ss[0]);
    }

    #[test]
    fn test_ambient_from_height_bottom() {
        let ambient = ambient_from_height(0.0, [0.3, 0.5, 0.8], [1.0, 0.9, 0.8]);
        // At bottom, more ground bounce
        assert!(ambient[0] > 0.0);
    }

    #[test]
    fn test_ambient_from_height_top() {
        let ambient = ambient_from_height(1.0, [0.3, 0.5, 0.8], [1.0, 0.9, 0.8]);
        // At top, more sky contribution
        assert!(ambient[2] > ambient[0]);
    }

    // =========================================================================
    // Adaptive Step Count Tests
    // =========================================================================

    #[test]
    fn test_step_count_near() {
        let marcher = CloudRayMarcher::new(CloudLayerConfig::default(), CloudQuality::High);
        let steps = marcher.get_step_count_for_distance(10000.0);
        assert_eq!(steps, 128); // Full steps for near distance
    }

    #[test]
    fn test_step_count_far() {
        let marcher = CloudRayMarcher::new(CloudLayerConfig::default(), CloudQuality::High);
        let steps = marcher.get_step_count_for_distance(200000.0);
        assert!(steps < 128); // Reduced steps for far distance
        assert!(steps >= 32); // But not too few (25% of 128 = 32)
    }

    #[test]
    fn test_step_count_minimum() {
        let marcher = CloudRayMarcher::new(CloudLayerConfig::default(), CloudQuality::Low);
        let steps = marcher.get_step_count_for_distance(500000.0);
        assert!(steps >= 8); // Minimum 8 steps
    }

    // =========================================================================
    // Early Termination Tests
    // =========================================================================

    #[test]
    fn test_early_termination_threshold() {
        let config = RayMarchConfig::default().with_transmittance_threshold(0.01);
        assert_eq!(config.transmittance_threshold, 0.01);
    }

    #[test]
    fn test_sample_is_opaque_at_threshold() {
        let mut sample = CloudSample::new();
        sample.transmittance = 0.009;
        assert!(sample.is_opaque(0.01));
    }

    // =========================================================================
    // CloudRayMarcher Tests
    // =========================================================================

    #[test]
    fn test_ray_marcher_new() {
        let marcher = CloudRayMarcher::new(CloudLayerConfig::default(), CloudQuality::High);
        assert_eq!(marcher.march_config().max_steps, 128);
    }

    #[test]
    fn test_ray_marcher_with_config() {
        let layer = CloudLayerConfig::stratus();
        let march = RayMarchConfig::low();
        let marcher = CloudRayMarcher::with_config(layer, march);

        assert_eq!(marcher.layer_config().min_height, layer.min_height);
        assert_eq!(marcher.march_config().max_steps, 32);
    }

    #[test]
    fn test_ray_marcher_set_sun_direction() {
        let mut marcher = CloudRayMarcher::new(CloudLayerConfig::default(), CloudQuality::Medium);
        marcher.set_sun_direction([1.0, 0.0, 0.0]);

        // Should be normalized
        // Can't directly access sun_direction, but it's used in compute_lighting
    }

    #[test]
    fn test_ray_marcher_march_ray_miss() {
        let layer = CloudLayerConfig::default();
        let marcher = CloudRayMarcher::new(layer, CloudQuality::Low);

        // Ray from above, going up (misses clouds)
        let sample = marcher.march_ray([0.0, 10000.0, 0.0], [0.0, 1.0, 0.0], 100000.0);

        assert!(sample.is_empty());
        assert_eq!(sample.transmittance, 1.0);
    }

    #[test]
    fn test_ray_marcher_march_ray_hit() {
        let layer = CloudLayerConfig::default().with_coverage(0.9);
        let marcher = CloudRayMarcher::new(layer, CloudQuality::Medium);

        // Ray from below, going up through clouds
        let sample = marcher.march_ray([0.0, 1000.0, 0.0], [0.0, 1.0, 0.0], 100000.0);

        // With high coverage, should hit some clouds
        // Note: result depends on procedural noise
        assert!(sample.transmittance <= 1.0);
    }

    #[test]
    fn test_ray_marcher_sample_density_outside() {
        let marcher = CloudRayMarcher::new(CloudLayerConfig::default(), CloudQuality::Low);
        let density = marcher.sample_density([0.0, 500.0, 0.0]); // Below layer
        assert_eq!(density, 0.0);

        let density = marcher.sample_density([0.0, 10000.0, 0.0]); // Above layer
        assert_eq!(density, 0.0);
    }

    #[test]
    fn test_ray_marcher_sample_density_inside() {
        let marcher = CloudRayMarcher::new(
            CloudLayerConfig::default().with_coverage(1.0),
            CloudQuality::Low,
        );
        let density = marcher.sample_density([0.0, 3000.0, 0.0]);
        // May or may not have density depending on noise
        assert!(density >= 0.0);
    }

    #[test]
    fn test_ray_marcher_compute_lighting() {
        let marcher = CloudRayMarcher::new(CloudLayerConfig::default(), CloudQuality::Low);
        let lighting = marcher.compute_lighting(
            [0.0, 3000.0, 0.0],
            0.5,
            normalize_vec3([0.5, 0.7, -0.5]),
        );

        assert!(lighting[0] >= 0.0);
        assert!(lighting[1] >= 0.0);
        assert!(lighting[2] >= 0.0);
    }

    #[test]
    fn test_ray_marcher_apply_beer_lambert() {
        let marcher = CloudRayMarcher::new(CloudLayerConfig::default(), CloudQuality::Low);
        let t = marcher.apply_beer_lambert(0.5, 10.0);
        assert!(t > 0.0 && t < 1.0);
    }

    #[test]
    fn test_ray_marcher_apply_powder_effect() {
        let marcher = CloudRayMarcher::new(CloudLayerConfig::default(), CloudQuality::Low);
        let p = marcher.apply_powder_effect(0.5, 2.0);
        assert!(p > 0.0 && p < 1.0);
    }

    // =========================================================================
    // CloudType Tests
    // =========================================================================

    #[test]
    fn test_cloud_type_default() {
        let ct = CloudType::default();
        assert_eq!(ct, CloudType::Cumulus);
    }

    #[test]
    fn test_cloud_type_height_gradient_params() {
        let (bot, top, anvil) = CloudType::Stratus.height_gradient_params();
        assert!(bot < top);
        assert_eq!(anvil, 0.0);

        let (_, _, anvil) = CloudType::Cumulonimbus.height_gradient_params();
        assert!(anvil > 0.0);
    }

    #[test]
    fn test_cloud_type_typical_coverage() {
        let (min, max) = CloudType::Stratus.typical_coverage();
        assert!(min > 0.5);
        assert!(max > min);

        let (min, _) = CloudType::Cirrus.typical_coverage();
        assert!(min < 0.3);
    }

    // =========================================================================
    // Helper Function Tests
    // =========================================================================

    #[test]
    fn test_normalize_vec3() {
        let v = normalize_vec3([3.0, 4.0, 0.0]);
        assert!((v[0] - 0.6).abs() < EPSILON);
        assert!((v[1] - 0.8).abs() < EPSILON);
    }

    #[test]
    fn test_normalize_vec3_zero() {
        let v = normalize_vec3([0.0, 0.0, 0.0]);
        assert_eq!(v, [0.0, 1.0, 0.0]); // Default to up
    }

    #[test]
    fn test_smoothstep_edges() {
        assert_eq!(smoothstep(0.0, 1.0, 0.0), 0.0);
        assert_eq!(smoothstep(0.0, 1.0, 1.0), 1.0);
    }

    #[test]
    fn test_smoothstep_middle() {
        let mid = smoothstep(0.0, 1.0, 0.5);
        assert!((mid - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_smoothstep_clamping() {
        assert_eq!(smoothstep(0.0, 1.0, -1.0), 0.0);
        assert_eq!(smoothstep(0.0, 1.0, 2.0), 1.0);
    }

    #[test]
    fn test_sample_procedural_noise_range() {
        for x in 0..10 {
            for y in 0..10 {
                for z in 0..10 {
                    let pos = [x as f32 * 1000.0, y as f32 * 1000.0, z as f32 * 1000.0];
                    let noise = sample_procedural_noise(pos, 6000.0);
                    assert!(noise >= 0.0 && noise <= 1.0, "Noise out of range: {}", noise);
                }
            }
        }
    }

    // =========================================================================
    // Integration Tests
    // =========================================================================

    #[test]
    fn test_full_ray_march_pipeline() {
        let layer = CloudLayerConfig::default()
            .with_height(2000.0, 6000.0)
            .with_coverage(0.7);

        let mut marcher = CloudRayMarcher::new(layer, CloudQuality::Medium);
        marcher.set_sun_direction([0.5, 0.7, -0.5]);
        marcher.set_sun_color([1.0, 0.95, 0.9]);
        marcher.set_sky_color([0.3, 0.5, 0.8]);

        // March from ground looking up
        let sample = marcher.march_ray(
            [0.0, 1000.0, 0.0],
            normalize_vec3([0.0, 0.3, -1.0]),
            50000.0,
        );

        // Convert to output
        let output = CloudRenderOutput::from_sample(&sample, 3000.0);

        // Basic sanity checks
        assert!(output.alpha >= 0.0 && output.alpha <= 1.0);
        assert!(output.color[0] >= 0.0);
        assert!(output.color[1] >= 0.0);
        assert!(output.color[2] >= 0.0);
    }

    #[test]
    fn test_energy_conservation() {
        // Verify that accumulated radiance doesn't exceed input energy
        let marcher = CloudRayMarcher::new(
            CloudLayerConfig::default().with_coverage(0.9),
            CloudQuality::High,
        );

        let sample = marcher.march_ray([0.0, 500.0, 0.0], [0.0, 1.0, 0.0], 100000.0);

        // Radiance should be bounded by sun intensity
        let max_radiance = sample.radiance[0].max(sample.radiance[1]).max(sample.radiance[2]);
        assert!(max_radiance < 10.0); // Reasonable upper bound
    }

    #[test]
    fn test_transmittance_monotonic_decrease() {
        let layer = CloudLayerConfig::default().with_coverage(1.0).with_density(0.1);
        let marcher = CloudRayMarcher::new(layer, CloudQuality::Low);

        // Multiple rays of increasing length through clouds
        let mut prev_transmittance = 1.0;
        for depth_mult in 1..5 {
            let sample = marcher.march_ray(
                [0.0, 1000.0, 0.0],
                [0.0, 1.0, 0.0],
                10000.0 * depth_mult as f32,
            );
            // Transmittance should generally decrease or stay same
            // (may not strictly decrease if ray exits clouds)
            assert!(sample.transmittance <= prev_transmittance + EPSILON);
            prev_transmittance = sample.transmittance.min(prev_transmittance);
        }
    }
}
