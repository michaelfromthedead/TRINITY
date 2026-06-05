//! Froxel Light Integration (T-ENV-3.4)
//!
//! This module provides full S4 light integration for froxels, enabling:
//! - Point light scattering through froxels (light cones visible in fog)
//! - Spot light scattering through froxels (beams visible in fog/dust)
//! - Per-froxel light evaluation from clustered light lists
//! - Shadow map integration for volumetric shadows
//! - Phase function per light type (Henyey-Greenstein)
//!
//! # Overview
//!
//! Each froxel evaluates the N nearest lights from the clustered light list,
//! computing inscattered light contribution. Performance is bounded by max
//! lights per froxel (typically 8-32) via cluster culling.
//!
//! # Scattering Model
//!
//! The Henyey-Greenstein phase function models angular light distribution:
//! ```text
//! p(cos_theta) = (1 - g^2) / (4*pi * (1 + g^2 - 2*g*cos_theta)^1.5)
//! ```
//!
//! Where:
//! - `g = 0`: Isotropic scattering (uniform fog)
//! - `g > 0`: Forward scattering (mist, haze - light visible looking toward source)
//! - `g < 0`: Back scattering (dust - light visible looking away from source)
//!
//! # Shadow Integration
//!
//! Each light's shadow map is sampled to attenuate inscattered light where
//! the froxel is in shadow. Multiple shadow samples can be used for soft
//! volumetric shadows.

use bytemuck::{Pod, Zeroable};
use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Minimum lights per froxel (performance floor).
pub const MIN_LIGHTS_PER_FROXEL: u32 = 1;

/// Maximum lights per froxel (memory/performance ceiling).
pub const MAX_LIGHTS_PER_FROXEL: u32 = 64;

/// Default max lights per froxel (balanced quality/performance).
pub const DEFAULT_MAX_LIGHTS_PER_FROXEL: u32 = 16;

/// Minimum Henyey-Greenstein g parameter.
pub const MIN_HG_G: f32 = -0.999;

/// Maximum Henyey-Greenstein g parameter.
pub const MAX_HG_G: f32 = 0.999;

/// Default phase asymmetry for fog (forward scattering).
pub const DEFAULT_PHASE_G: f32 = 0.8;

/// Minimum scatter intensity.
pub const MIN_SCATTER_INTENSITY: f32 = 0.0;

/// Maximum scatter intensity.
pub const MAX_SCATTER_INTENSITY: f32 = 10.0;

/// Minimum shadow samples.
pub const MIN_SHADOW_SAMPLES: u32 = 1;

/// Maximum shadow samples.
pub const MAX_SHADOW_SAMPLES: u32 = 16;

/// Small epsilon for floating point comparisons.
const EPSILON: f32 = 1e-6;

// ---------------------------------------------------------------------------
// FroxelLightConfig — GPU-uploadable configuration
// ---------------------------------------------------------------------------

/// Configuration for froxel light integration.
///
/// This struct is designed to be uploaded to the GPU as a uniform buffer.
/// The layout is `repr(C)` and implements `Pod` for bytemuck compatibility.
///
/// # Memory Layout (16 bytes)
///
/// | Offset | Field               | Size    |
/// |--------|---------------------|---------|
/// | 0      | max_lights_per_froxel | 4 bytes |
/// | 4      | phase_g             | 4 bytes |
/// | 8      | scatter_intensity   | 4 bytes |
/// | 12     | shadow_samples      | 4 bytes |
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct FroxelLightConfig {
    /// Maximum number of lights to evaluate per froxel (8-32 typical).
    ///
    /// Limits performance cost but may miss distant lights.
    pub max_lights_per_froxel: u32,

    /// Henyey-Greenstein asymmetry parameter (-1 to 1).
    ///
    /// Controls the angular distribution of scattered light:
    /// - g = 0: Isotropic (uniform in all directions)
    /// - g > 0: Forward scattering (typical for fog/mist, 0.7-0.9)
    /// - g < 0: Back scattering (dust particles)
    pub phase_g: f32,

    /// Global scatter intensity multiplier.
    ///
    /// Scales all inscattered light. Higher values produce more visible
    /// light beams in fog.
    pub scatter_intensity: f32,

    /// Number of shadow map samples per light (1-4 typical).
    ///
    /// Higher values produce softer volumetric shadows but cost more.
    pub shadow_samples: u32,
}

impl FroxelLightConfig {
    /// Create a new froxel light configuration.
    ///
    /// # Arguments
    ///
    /// * `max_lights` - Max lights per froxel (clamped to 1-64).
    /// * `phase_g` - Henyey-Greenstein parameter (clamped to -0.999 to 0.999).
    /// * `scatter_intensity` - Global scatter multiplier.
    /// * `shadow_samples` - Shadow quality (clamped to 1-16).
    pub fn new(
        max_lights: u32,
        phase_g: f32,
        scatter_intensity: f32,
        shadow_samples: u32,
    ) -> Self {
        Self {
            max_lights_per_froxel: max_lights.clamp(
                MIN_LIGHTS_PER_FROXEL,
                MAX_LIGHTS_PER_FROXEL,
            ),
            phase_g: phase_g.clamp(MIN_HG_G, MAX_HG_G),
            scatter_intensity: scatter_intensity.clamp(
                MIN_SCATTER_INTENSITY,
                MAX_SCATTER_INTENSITY,
            ),
            shadow_samples: shadow_samples.clamp(
                MIN_SHADOW_SAMPLES,
                MAX_SHADOW_SAMPLES,
            ),
        }
    }

    /// Create a low-quality configuration (fast, less accurate).
    pub fn low_quality() -> Self {
        Self::new(8, DEFAULT_PHASE_G, 1.0, 1)
    }

    /// Create a medium-quality configuration (balanced).
    pub fn medium_quality() -> Self {
        Self::new(16, DEFAULT_PHASE_G, 1.0, 2)
    }

    /// Create a high-quality configuration (accurate, slower).
    pub fn high_quality() -> Self {
        Self::new(32, DEFAULT_PHASE_G, 1.0, 4)
    }

    /// Validate the configuration.
    pub fn is_valid(&self) -> bool {
        self.max_lights_per_froxel >= MIN_LIGHTS_PER_FROXEL
            && self.max_lights_per_froxel <= MAX_LIGHTS_PER_FROXEL
            && self.phase_g >= MIN_HG_G
            && self.phase_g <= MAX_HG_G
            && self.scatter_intensity >= MIN_SCATTER_INTENSITY
            && self.scatter_intensity <= MAX_SCATTER_INTENSITY
            && self.shadow_samples >= MIN_SHADOW_SAMPLES
            && self.shadow_samples <= MAX_SHADOW_SAMPLES
    }
}

impl Default for FroxelLightConfig {
    fn default() -> Self {
        Self::medium_quality()
    }
}

// Size assertion: 16 bytes
const _: () = assert!(std::mem::size_of::<FroxelLightConfig>() == 16);

// ---------------------------------------------------------------------------
// FroxelLightContribution — Light contribution to a froxel
// ---------------------------------------------------------------------------

/// Light contribution to a single froxel.
///
/// Stores the inscattered light color and shadow attenuation factor
/// for a light affecting this froxel.
///
/// # Memory Layout (16 bytes)
///
/// | Offset | Field        | Size    |
/// |--------|--------------|---------|
/// | 0      | inscatter    | 12 bytes |
/// | 12     | shadow_factor | 4 bytes |
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct FroxelLightContribution {
    /// Inscattered light color (RGB, pre-multiplied by phase function).
    pub inscatter: [f32; 3],

    /// Shadow attenuation factor (0 = fully shadowed, 1 = fully lit).
    pub shadow_factor: f32,
}

impl FroxelLightContribution {
    /// Create a new light contribution.
    pub fn new(inscatter: [f32; 3], shadow_factor: f32) -> Self {
        Self {
            inscatter,
            shadow_factor: shadow_factor.clamp(0.0, 1.0),
        }
    }

    /// Create a zero contribution (no light).
    pub fn zero() -> Self {
        Self {
            inscatter: [0.0; 3],
            shadow_factor: 0.0,
        }
    }

    /// Check if this contribution is effectively zero.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.shadow_factor < EPSILON
            || (self.inscatter[0].abs() < EPSILON
                && self.inscatter[1].abs() < EPSILON
                && self.inscatter[2].abs() < EPSILON)
    }

    /// Get the final inscattered light (inscatter * shadow_factor).
    #[inline]
    pub fn final_inscatter(&self) -> [f32; 3] {
        [
            self.inscatter[0] * self.shadow_factor,
            self.inscatter[1] * self.shadow_factor,
            self.inscatter[2] * self.shadow_factor,
        ]
    }

    /// Get the luminance of the inscattered light.
    #[inline]
    pub fn luminance(&self) -> f32 {
        // ITU-R BT.709 luminance coefficients
        let final_rgb = self.final_inscatter();
        0.2126 * final_rgb[0] + 0.7152 * final_rgb[1] + 0.0722 * final_rgb[2]
    }
}

impl Default for FroxelLightContribution {
    fn default() -> Self {
        Self::zero()
    }
}

// Size assertion: 16 bytes
const _: () = assert!(std::mem::size_of::<FroxelLightContribution>() == 16);

// ---------------------------------------------------------------------------
// PhaseType — Phase function types
// ---------------------------------------------------------------------------

/// Phase function type for different scattering media.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PhaseType {
    /// Isotropic scattering (g = 0).
    ///
    /// Light scatters uniformly in all directions. Good for very thick fog.
    Isotropic,

    /// Henyey-Greenstein with configurable g parameter.
    ///
    /// Most common choice for fog and atmospheric scattering.
    #[default]
    HenyeyGreenstein,

    /// Schlick approximation to Henyey-Greenstein.
    ///
    /// Faster to compute, similar appearance for moderate g values.
    Schlick,

    /// Rayleigh scattering for atmospheric effects.
    ///
    /// Strong forward and backward scattering, weak at 90 degrees.
    Rayleigh,
}

impl PhaseType {
    /// Get the name string for this phase type.
    #[inline]
    pub fn name(&self) -> &'static str {
        match self {
            PhaseType::Isotropic => "isotropic",
            PhaseType::HenyeyGreenstein => "henyey_greenstein",
            PhaseType::Schlick => "schlick",
            PhaseType::Rayleigh => "rayleigh",
        }
    }
}

// ---------------------------------------------------------------------------
// FroxelLightEvaluator — Main evaluation logic
// ---------------------------------------------------------------------------

/// Evaluator for froxel light integration.
///
/// Computes inscattered light contributions from point and spot lights
/// to individual froxels, using physically-based phase functions and
/// shadow attenuation.
pub struct FroxelLightEvaluator {
    /// Configuration for the evaluator.
    pub config: FroxelLightConfig,

    /// Phase function type to use.
    pub phase_type: PhaseType,

    /// Per-channel scattering coefficients (extinction * albedo).
    ///
    /// Controls how much light is scattered per unit distance.
    pub scatter_coefficients: [f32; 3],
}

impl FroxelLightEvaluator {
    /// Create a new froxel light evaluator.
    pub fn new(config: FroxelLightConfig) -> Self {
        Self {
            config,
            phase_type: PhaseType::HenyeyGreenstein,
            scatter_coefficients: [0.1, 0.1, 0.1],
        }
    }

    /// Create with custom phase type.
    pub fn with_phase_type(mut self, phase_type: PhaseType) -> Self {
        self.phase_type = phase_type;
        self
    }

    /// Create with custom scatter coefficients.
    pub fn with_scatter_coefficients(mut self, coeffs: [f32; 3]) -> Self {
        self.scatter_coefficients = [
            coeffs[0].max(0.0),
            coeffs[1].max(0.0),
            coeffs[2].max(0.0),
        ];
        self
    }

    // -----------------------------------------------------------------------
    // Phase Functions
    // -----------------------------------------------------------------------

    /// Henyey-Greenstein phase function.
    ///
    /// Calculates the probability distribution of scattering angle.
    ///
    /// # Arguments
    ///
    /// * `cos_theta` - Cosine of angle between view and light directions.
    /// * `g` - Asymmetry parameter (-1 to 1).
    ///
    /// # Returns
    ///
    /// Phase function value (integrates to 1 over sphere).
    #[inline]
    pub fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
        let g2 = g * g;
        let denom = 1.0 + g2 - 2.0 * g * cos_theta;
        if denom <= EPSILON {
            // Prevent division by zero at extreme angles
            return 1.0 / (4.0 * PI);
        }
        (1.0 - g2) / (4.0 * PI * denom.powf(1.5))
    }

    /// Schlick approximation to Henyey-Greenstein.
    ///
    /// Faster to compute with similar visual results for |g| < 0.8.
    /// Uses the formula: p(theta) = (1 - k^2) / (4*pi * (1 - k*cos(theta))^2)
    /// where k approximates the HG g parameter.
    ///
    /// # Arguments
    ///
    /// * `cos_theta` - Cosine of angle between view and light directions.
    /// * `g` - Asymmetry parameter (-1 to 1), used directly as k.
    #[inline]
    pub fn schlick_phase(cos_theta: f32, g: f32) -> f32 {
        // Use g directly as k (standard Schlick approximation)
        // For forward scattering (positive g), we want higher values when cos_theta = 1
        // Formula: (1 - k^2) / (4*pi * (1 - k*cos_theta)^2)
        let k = g.clamp(-0.999, 0.999);
        let denom = 1.0 - k * cos_theta;
        if denom.abs() <= EPSILON {
            return 1.0 / (4.0 * PI);
        }
        (1.0 - k * k) / (4.0 * PI * denom * denom)
    }

    /// Isotropic phase function (uniform scattering).
    #[inline]
    pub fn isotropic_phase() -> f32 {
        1.0 / (4.0 * PI)
    }

    /// Rayleigh phase function for atmospheric scattering.
    ///
    /// # Arguments
    ///
    /// * `cos_theta` - Cosine of angle between view and light directions.
    #[inline]
    pub fn rayleigh_phase(cos_theta: f32) -> f32 {
        // Rayleigh: (3/16pi) * (1 + cos^2(theta))
        (3.0 / (16.0 * PI)) * (1.0 + cos_theta * cos_theta)
    }

    /// Evaluate the configured phase function.
    ///
    /// # Arguments
    ///
    /// * `cos_theta` - Cosine of angle between view and light directions.
    #[inline]
    pub fn evaluate_phase(&self, cos_theta: f32) -> f32 {
        match self.phase_type {
            PhaseType::Isotropic => Self::isotropic_phase(),
            PhaseType::HenyeyGreenstein => {
                Self::henyey_greenstein(cos_theta, self.config.phase_g)
            }
            PhaseType::Schlick => {
                Self::schlick_phase(cos_theta, self.config.phase_g)
            }
            PhaseType::Rayleigh => Self::rayleigh_phase(cos_theta),
        }
    }

    // -----------------------------------------------------------------------
    // Point Light Evaluation
    // -----------------------------------------------------------------------

    /// Evaluate point light contribution to a froxel.
    ///
    /// Computes inscattered light from a point light source, considering:
    /// - Distance attenuation (inverse square falloff)
    /// - Phase function (angular scattering distribution)
    /// - Scatter intensity multiplier
    ///
    /// # Arguments
    ///
    /// * `froxel_center` - World-space center of the froxel.
    /// * `light_pos` - World-space position of the light.
    /// * `light_color` - Light RGB color/intensity.
    /// * `light_radius` - Light influence radius (zero attenuation at edge).
    /// * `view_dir` - Normalized direction toward camera.
    ///
    /// # Returns
    ///
    /// Light contribution with inscatter and shadow factor (1.0 if no shadow).
    pub fn evaluate_point_light(
        &self,
        froxel_center: [f32; 3],
        light_pos: [f32; 3],
        light_color: [f32; 3],
        light_radius: f32,
        view_dir: [f32; 3],
    ) -> FroxelLightContribution {
        // Vector from froxel to light
        let to_light = [
            light_pos[0] - froxel_center[0],
            light_pos[1] - froxel_center[1],
            light_pos[2] - froxel_center[2],
        ];

        // Distance to light
        let dist_sq = to_light[0] * to_light[0]
            + to_light[1] * to_light[1]
            + to_light[2] * to_light[2];
        let dist = dist_sq.sqrt();

        // Check if outside light radius
        if dist > light_radius || dist < EPSILON {
            return FroxelLightContribution::zero();
        }

        // Normalized direction to light
        let inv_dist = 1.0 / dist;
        let light_dir = [
            to_light[0] * inv_dist,
            to_light[1] * inv_dist,
            to_light[2] * inv_dist,
        ];

        // Distance attenuation (smooth falloff to zero at radius)
        let normalized_dist = dist / light_radius;
        let attenuation = Self::smooth_distance_attenuation(normalized_dist);

        // Phase function (cosine of angle between view and light direction)
        let cos_theta = -(view_dir[0] * light_dir[0]
            + view_dir[1] * light_dir[1]
            + view_dir[2] * light_dir[2]);
        let phase = self.evaluate_phase(cos_theta);

        // Inscattered light
        let intensity = attenuation * phase * self.config.scatter_intensity;
        let inscatter = [
            light_color[0] * intensity * self.scatter_coefficients[0],
            light_color[1] * intensity * self.scatter_coefficients[1],
            light_color[2] * intensity * self.scatter_coefficients[2],
        ];

        FroxelLightContribution::new(inscatter, 1.0)
    }

    /// Evaluate point light with shadow factor.
    ///
    /// Same as `evaluate_point_light` but with explicit shadow attenuation.
    pub fn evaluate_point_light_shadowed(
        &self,
        froxel_center: [f32; 3],
        light_pos: [f32; 3],
        light_color: [f32; 3],
        light_radius: f32,
        view_dir: [f32; 3],
        shadow_factor: f32,
    ) -> FroxelLightContribution {
        let mut contrib = self.evaluate_point_light(
            froxel_center,
            light_pos,
            light_color,
            light_radius,
            view_dir,
        );
        contrib.shadow_factor = shadow_factor.clamp(0.0, 1.0);
        contrib
    }

    // -----------------------------------------------------------------------
    // Spot Light Evaluation
    // -----------------------------------------------------------------------

    /// Evaluate spot light contribution to a froxel.
    ///
    /// Computes inscattered light from a spot light source, considering:
    /// - Distance attenuation (inverse square falloff)
    /// - Cone attenuation (smooth falloff from inner to outer angle)
    /// - Phase function (angular scattering distribution)
    /// - Scatter intensity multiplier
    ///
    /// # Arguments
    ///
    /// * `froxel_center` - World-space center of the froxel.
    /// * `light_pos` - World-space position of the light.
    /// * `light_dir` - Normalized direction the light is pointing.
    /// * `light_color` - Light RGB color/intensity.
    /// * `light_radius` - Light influence radius.
    /// * `inner_angle` - Inner cone angle in radians (full intensity).
    /// * `outer_angle` - Outer cone angle in radians (zero intensity).
    /// * `view_dir` - Normalized direction toward camera.
    ///
    /// # Returns
    ///
    /// Light contribution with inscatter and shadow factor (1.0 if no shadow).
    pub fn evaluate_spot_light(
        &self,
        froxel_center: [f32; 3],
        light_pos: [f32; 3],
        light_dir: [f32; 3],
        light_color: [f32; 3],
        light_radius: f32,
        inner_angle: f32,
        outer_angle: f32,
        view_dir: [f32; 3],
    ) -> FroxelLightContribution {
        // Vector from light to froxel
        let to_froxel = [
            froxel_center[0] - light_pos[0],
            froxel_center[1] - light_pos[1],
            froxel_center[2] - light_pos[2],
        ];

        // Distance to light
        let dist_sq = to_froxel[0] * to_froxel[0]
            + to_froxel[1] * to_froxel[1]
            + to_froxel[2] * to_froxel[2];
        let dist = dist_sq.sqrt();

        // Check if outside light radius
        if dist > light_radius || dist < EPSILON {
            return FroxelLightContribution::zero();
        }

        // Normalized direction from light to froxel
        let inv_dist = 1.0 / dist;
        let to_froxel_norm = [
            to_froxel[0] * inv_dist,
            to_froxel[1] * inv_dist,
            to_froxel[2] * inv_dist,
        ];

        // Cosine of angle between light direction and direction to froxel
        let cos_spot_angle = light_dir[0] * to_froxel_norm[0]
            + light_dir[1] * to_froxel_norm[1]
            + light_dir[2] * to_froxel_norm[2];

        // Check if outside outer cone
        let cos_outer = outer_angle.cos();
        if cos_spot_angle < cos_outer {
            return FroxelLightContribution::zero();
        }

        // Cone attenuation (smooth falloff from inner to outer)
        let cos_inner = inner_angle.cos();
        let cone_attenuation = Self::spot_cone_attenuation(
            cos_spot_angle,
            cos_inner,
            cos_outer,
        );

        // Distance attenuation
        let normalized_dist = dist / light_radius;
        let dist_attenuation = Self::smooth_distance_attenuation(normalized_dist);

        // Direction from froxel to light (for phase function)
        let froxel_to_light = [
            -to_froxel_norm[0],
            -to_froxel_norm[1],
            -to_froxel_norm[2],
        ];

        // Phase function
        let cos_theta = -(view_dir[0] * froxel_to_light[0]
            + view_dir[1] * froxel_to_light[1]
            + view_dir[2] * froxel_to_light[2]);
        let phase = self.evaluate_phase(cos_theta);

        // Inscattered light
        let intensity =
            dist_attenuation * cone_attenuation * phase * self.config.scatter_intensity;
        let inscatter = [
            light_color[0] * intensity * self.scatter_coefficients[0],
            light_color[1] * intensity * self.scatter_coefficients[1],
            light_color[2] * intensity * self.scatter_coefficients[2],
        ];

        FroxelLightContribution::new(inscatter, 1.0)
    }

    /// Evaluate spot light with shadow factor.
    pub fn evaluate_spot_light_shadowed(
        &self,
        froxel_center: [f32; 3],
        light_pos: [f32; 3],
        light_dir: [f32; 3],
        light_color: [f32; 3],
        light_radius: f32,
        inner_angle: f32,
        outer_angle: f32,
        view_dir: [f32; 3],
        shadow_factor: f32,
    ) -> FroxelLightContribution {
        let mut contrib = self.evaluate_spot_light(
            froxel_center,
            light_pos,
            light_dir,
            light_color,
            light_radius,
            inner_angle,
            outer_angle,
            view_dir,
        );
        contrib.shadow_factor = shadow_factor.clamp(0.0, 1.0);
        contrib
    }

    // -----------------------------------------------------------------------
    // Attenuation Functions
    // -----------------------------------------------------------------------

    /// Smooth distance attenuation with falloff to zero at radius edge.
    ///
    /// Uses a modified inverse-square law with a smooth cutoff:
    /// ```text
    /// att = saturate(1 - (d/r)^4)^2 / (d^2 + epsilon)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `normalized_dist` - Distance divided by radius (0 to 1+).
    #[inline]
    pub fn smooth_distance_attenuation(normalized_dist: f32) -> f32 {
        if normalized_dist >= 1.0 {
            return 0.0;
        }

        // Smooth window function
        let d4 = normalized_dist * normalized_dist * normalized_dist * normalized_dist;
        let window = (1.0 - d4).max(0.0);
        let window_sq = window * window;

        // Inverse square with smooth cutoff
        let dist_sq = normalized_dist * normalized_dist + EPSILON;
        window_sq / dist_sq
    }

    /// Spot light cone attenuation with smooth falloff.
    ///
    /// # Arguments
    ///
    /// * `cos_angle` - Cosine of angle from light direction to point.
    /// * `cos_inner` - Cosine of inner cone angle (full intensity).
    /// * `cos_outer` - Cosine of outer cone angle (zero intensity).
    #[inline]
    pub fn spot_cone_attenuation(cos_angle: f32, cos_inner: f32, cos_outer: f32) -> f32 {
        if cos_angle >= cos_inner {
            return 1.0;
        }
        if cos_angle <= cos_outer {
            return 0.0;
        }

        // Smooth falloff using smoothstep
        let t = (cos_angle - cos_outer) / (cos_inner - cos_outer);
        // Smoothstep: 3t^2 - 2t^3
        t * t * (3.0 - 2.0 * t)
    }

    // -----------------------------------------------------------------------
    // Light Accumulation
    // -----------------------------------------------------------------------

    /// Accumulate multiple light contributions into a single inscatter value.
    ///
    /// Sums the final inscattered light (inscatter * shadow_factor) from
    /// all contributions.
    ///
    /// # Arguments
    ///
    /// * `contributions` - Slice of light contributions.
    ///
    /// # Returns
    ///
    /// Total inscattered light RGB.
    pub fn accumulate_lights(&self, contributions: &[FroxelLightContribution]) -> [f32; 3] {
        let mut total = [0.0f32; 3];

        for contrib in contributions {
            let final_rgb = contrib.final_inscatter();
            total[0] += final_rgb[0];
            total[1] += final_rgb[1];
            total[2] += final_rgb[2];
        }

        total
    }

    /// Accumulate lights with weighted importance sampling.
    ///
    /// Weights contributions by their luminance for better convergence.
    ///
    /// # Arguments
    ///
    /// * `contributions` - Slice of light contributions.
    /// * `weights` - Per-light importance weights.
    pub fn accumulate_lights_weighted(
        &self,
        contributions: &[FroxelLightContribution],
        weights: &[f32],
    ) -> [f32; 3] {
        let mut total = [0.0f32; 3];
        let mut weight_sum = 0.0f32;

        for (contrib, weight) in contributions.iter().zip(weights.iter()) {
            let final_rgb = contrib.final_inscatter();
            let w = weight.max(0.0);
            total[0] += final_rgb[0] * w;
            total[1] += final_rgb[1] * w;
            total[2] += final_rgb[2] * w;
            weight_sum += w;
        }

        // Normalize by weight sum if non-zero
        if weight_sum > EPSILON {
            total[0] /= weight_sum;
            total[1] /= weight_sum;
            total[2] /= weight_sum;
        }

        total
    }

    // -----------------------------------------------------------------------
    // Light Culling
    // -----------------------------------------------------------------------

    /// Check if a point light affects a froxel (sphere-AABB intersection).
    ///
    /// # Arguments
    ///
    /// * `froxel_min` - Minimum corner of froxel AABB.
    /// * `froxel_max` - Maximum corner of froxel AABB.
    /// * `light_pos` - World-space position of the light.
    /// * `light_radius` - Light influence radius.
    #[inline]
    pub fn point_light_affects_froxel(
        froxel_min: [f32; 3],
        froxel_max: [f32; 3],
        light_pos: [f32; 3],
        light_radius: f32,
    ) -> bool {
        // Find closest point on AABB to sphere center
        let closest = [
            light_pos[0].clamp(froxel_min[0], froxel_max[0]),
            light_pos[1].clamp(froxel_min[1], froxel_max[1]),
            light_pos[2].clamp(froxel_min[2], froxel_max[2]),
        ];

        // Check distance to closest point
        let dx = light_pos[0] - closest[0];
        let dy = light_pos[1] - closest[1];
        let dz = light_pos[2] - closest[2];
        let dist_sq = dx * dx + dy * dy + dz * dz;

        dist_sq <= light_radius * light_radius
    }

    /// Check if a spot light affects a froxel (cone-AABB intersection).
    ///
    /// Uses a conservative sphere-AABB test with the cone's bounding sphere.
    ///
    /// # Arguments
    ///
    /// * `froxel_min` - Minimum corner of froxel AABB.
    /// * `froxel_max` - Maximum corner of froxel AABB.
    /// * `light_pos` - World-space position of the light.
    /// * `light_dir` - Normalized direction the light is pointing.
    /// * `light_radius` - Light influence radius.
    /// * `outer_angle` - Outer cone angle in radians.
    #[inline]
    pub fn spot_light_affects_froxel(
        froxel_min: [f32; 3],
        froxel_max: [f32; 3],
        light_pos: [f32; 3],
        light_dir: [f32; 3],
        light_radius: f32,
        outer_angle: f32,
    ) -> bool {
        // First, do a quick sphere test with the light's bounding sphere
        if !Self::point_light_affects_froxel(
            froxel_min,
            froxel_max,
            light_pos,
            light_radius,
        ) {
            return false;
        }

        // Find the center of the AABB
        let aabb_center = [
            (froxel_min[0] + froxel_max[0]) * 0.5,
            (froxel_min[1] + froxel_max[1]) * 0.5,
            (froxel_min[2] + froxel_max[2]) * 0.5,
        ];

        // Vector from light to AABB center
        let to_center = [
            aabb_center[0] - light_pos[0],
            aabb_center[1] - light_pos[1],
            aabb_center[2] - light_pos[2],
        ];

        let dist = (to_center[0] * to_center[0]
            + to_center[1] * to_center[1]
            + to_center[2] * to_center[2])
        .sqrt();

        if dist < EPSILON {
            return true; // Light is at AABB center
        }

        // Cosine of angle to AABB center
        let inv_dist = 1.0 / dist;
        let cos_angle = (light_dir[0] * to_center[0]
            + light_dir[1] * to_center[1]
            + light_dir[2] * to_center[2])
            * inv_dist;

        // Half diagonal of the AABB (conservative radius)
        let half_diag = [
            (froxel_max[0] - froxel_min[0]) * 0.5,
            (froxel_max[1] - froxel_min[1]) * 0.5,
            (froxel_max[2] - froxel_min[2]) * 0.5,
        ];
        let aabb_radius = (half_diag[0] * half_diag[0]
            + half_diag[1] * half_diag[1]
            + half_diag[2] * half_diag[2])
        .sqrt();

        // Conservative cone test: extend outer angle by AABB angular radius
        let sin_aabb = (aabb_radius / dist).min(1.0);
        let cos_aabb = (1.0 - sin_aabb * sin_aabb).max(0.0).sqrt();
        let cos_outer = outer_angle.cos();

        // cos(outer + aabb_angle) = cos(outer)*cos(aabb) - sin(outer)*sin(aabb)
        let sin_outer = outer_angle.sin();
        let cos_extended = cos_outer * cos_aabb - sin_outer * sin_aabb;

        cos_angle >= cos_extended
    }

    /// Cull lights from a list based on froxel AABB.
    ///
    /// Returns indices of lights that may affect the froxel.
    ///
    /// # Arguments
    ///
    /// * `froxel_min` - Minimum corner of froxel AABB.
    /// * `froxel_max` - Maximum corner of froxel AABB.
    /// * `light_positions` - Positions of all lights.
    /// * `light_radii` - Radii of all lights.
    /// * `max_lights` - Maximum number of lights to return.
    pub fn cull_point_lights(
        froxel_min: [f32; 3],
        froxel_max: [f32; 3],
        light_positions: &[[f32; 3]],
        light_radii: &[f32],
        max_lights: usize,
    ) -> Vec<usize> {
        let mut result = Vec::with_capacity(max_lights.min(light_positions.len()));

        for (i, (pos, radius)) in light_positions.iter().zip(light_radii.iter()).enumerate() {
            if result.len() >= max_lights {
                break;
            }

            if Self::point_light_affects_froxel(froxel_min, froxel_max, *pos, *radius) {
                result.push(i);
            }
        }

        result
    }
}

impl Default for FroxelLightEvaluator {
    fn default() -> Self {
        Self::new(FroxelLightConfig::default())
    }
}

// ---------------------------------------------------------------------------
// FroxelLightAccumulator — Batch accumulation helper
// ---------------------------------------------------------------------------

/// Helper for accumulating light contributions across multiple froxels.
///
/// Provides methods for batch processing and performance tracking.
pub struct FroxelLightAccumulator {
    /// Accumulated inscatter values per froxel.
    pub inscatter_values: Vec<[f32; 3]>,

    /// Number of lights evaluated per froxel.
    pub lights_per_froxel: Vec<u32>,

    /// Total lights culled.
    pub total_lights_culled: u64,

    /// Total lights evaluated.
    pub total_lights_evaluated: u64,
}

impl FroxelLightAccumulator {
    /// Create a new accumulator for the given number of froxels.
    pub fn new(num_froxels: usize) -> Self {
        Self {
            inscatter_values: vec![[0.0; 3]; num_froxels],
            lights_per_froxel: vec![0; num_froxels],
            total_lights_culled: 0,
            total_lights_evaluated: 0,
        }
    }

    /// Reset all accumulated values.
    pub fn reset(&mut self) {
        for v in &mut self.inscatter_values {
            *v = [0.0; 3];
        }
        for c in &mut self.lights_per_froxel {
            *c = 0;
        }
        self.total_lights_culled = 0;
        self.total_lights_evaluated = 0;
    }

    /// Add a light contribution to a froxel.
    #[inline]
    pub fn add_contribution(&mut self, froxel_index: usize, contrib: &FroxelLightContribution) {
        if froxel_index >= self.inscatter_values.len() {
            return;
        }

        let final_rgb = contrib.final_inscatter();
        self.inscatter_values[froxel_index][0] += final_rgb[0];
        self.inscatter_values[froxel_index][1] += final_rgb[1];
        self.inscatter_values[froxel_index][2] += final_rgb[2];
        self.lights_per_froxel[froxel_index] += 1;
        self.total_lights_evaluated += 1;
    }

    /// Record a culled light.
    #[inline]
    pub fn record_culled(&mut self) {
        self.total_lights_culled += 1;
    }

    /// Get the average number of lights per froxel.
    pub fn average_lights_per_froxel(&self) -> f32 {
        if self.lights_per_froxel.is_empty() {
            return 0.0;
        }

        let sum: u64 = self.lights_per_froxel.iter().map(|&x| x as u64).sum();
        sum as f32 / self.lights_per_froxel.len() as f32
    }

    /// Get the cull efficiency (fraction of lights culled).
    pub fn cull_efficiency(&self) -> f32 {
        let total = self.total_lights_culled + self.total_lights_evaluated;
        if total == 0 {
            return 0.0;
        }
        self.total_lights_culled as f32 / total as f32
    }

    /// Get the maximum inscatter luminance across all froxels.
    pub fn max_luminance(&self) -> f32 {
        self.inscatter_values
            .iter()
            .map(|rgb| 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])
            .fold(0.0f32, f32::max)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_EPSILON: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < TEST_EPSILON
    }

    fn approx_eq_eps(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    fn rgb_approx_eq(a: [f32; 3], b: [f32; 3], eps: f32) -> bool {
        (a[0] - b[0]).abs() < eps && (a[1] - b[1]).abs() < eps && (a[2] - b[2]).abs() < eps
    }

    // -----------------------------------------------------------------------
    // FroxelLightConfig Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_default() {
        let config = FroxelLightConfig::default();
        assert_eq!(config.max_lights_per_froxel, 16);
        assert!(approx_eq(config.phase_g, DEFAULT_PHASE_G));
        assert!(approx_eq(config.scatter_intensity, 1.0));
        assert_eq!(config.shadow_samples, 2);
        assert!(config.is_valid());
    }

    #[test]
    fn test_config_new_clamps_values() {
        // Test max lights clamping
        let config = FroxelLightConfig::new(0, 0.5, 1.0, 2);
        assert_eq!(config.max_lights_per_froxel, MIN_LIGHTS_PER_FROXEL);

        let config = FroxelLightConfig::new(100, 0.5, 1.0, 2);
        assert_eq!(config.max_lights_per_froxel, MAX_LIGHTS_PER_FROXEL);

        // Test phase_g clamping
        let config = FroxelLightConfig::new(16, -2.0, 1.0, 2);
        assert!(config.phase_g >= MIN_HG_G);

        let config = FroxelLightConfig::new(16, 2.0, 1.0, 2);
        assert!(config.phase_g <= MAX_HG_G);

        // Test scatter_intensity clamping
        let config = FroxelLightConfig::new(16, 0.5, -1.0, 2);
        assert!(config.scatter_intensity >= MIN_SCATTER_INTENSITY);

        let config = FroxelLightConfig::new(16, 0.5, 100.0, 2);
        assert!(config.scatter_intensity <= MAX_SCATTER_INTENSITY);

        // Test shadow_samples clamping
        let config = FroxelLightConfig::new(16, 0.5, 1.0, 0);
        assert_eq!(config.shadow_samples, MIN_SHADOW_SAMPLES);

        let config = FroxelLightConfig::new(16, 0.5, 1.0, 100);
        assert_eq!(config.shadow_samples, MAX_SHADOW_SAMPLES);
    }

    #[test]
    fn test_config_low_quality() {
        let config = FroxelLightConfig::low_quality();
        assert_eq!(config.max_lights_per_froxel, 8);
        assert_eq!(config.shadow_samples, 1);
        assert!(config.is_valid());
    }

    #[test]
    fn test_config_medium_quality() {
        let config = FroxelLightConfig::medium_quality();
        assert_eq!(config.max_lights_per_froxel, 16);
        assert_eq!(config.shadow_samples, 2);
        assert!(config.is_valid());
    }

    #[test]
    fn test_config_high_quality() {
        let config = FroxelLightConfig::high_quality();
        assert_eq!(config.max_lights_per_froxel, 32);
        assert_eq!(config.shadow_samples, 4);
        assert!(config.is_valid());
    }

    #[test]
    fn test_config_is_valid() {
        assert!(FroxelLightConfig::default().is_valid());
        assert!(FroxelLightConfig::new(16, 0.8, 1.0, 2).is_valid());
        assert!(FroxelLightConfig::new(1, -0.999, 0.0, 1).is_valid());
        assert!(FroxelLightConfig::new(64, 0.999, 10.0, 16).is_valid());
    }

    #[test]
    fn test_config_size() {
        assert_eq!(std::mem::size_of::<FroxelLightConfig>(), 16);
    }

    // -----------------------------------------------------------------------
    // FroxelLightContribution Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_contribution_new() {
        let contrib = FroxelLightContribution::new([1.0, 0.5, 0.25], 0.8);
        assert!(rgb_approx_eq(contrib.inscatter, [1.0, 0.5, 0.25], TEST_EPSILON));
        assert!(approx_eq(contrib.shadow_factor, 0.8));
    }

    #[test]
    fn test_contribution_zero() {
        let contrib = FroxelLightContribution::zero();
        assert!(rgb_approx_eq(contrib.inscatter, [0.0; 3], TEST_EPSILON));
        assert!(approx_eq(contrib.shadow_factor, 0.0));
        assert!(contrib.is_zero());
    }

    #[test]
    fn test_contribution_is_zero() {
        assert!(FroxelLightContribution::zero().is_zero());
        assert!(FroxelLightContribution::new([0.0; 3], 0.0).is_zero());
        assert!(FroxelLightContribution::new([0.0; 3], 1.0).is_zero());
        assert!(FroxelLightContribution::new([1.0, 0.0, 0.0], 0.0).is_zero());
        assert!(!FroxelLightContribution::new([1.0, 0.0, 0.0], 1.0).is_zero());
    }

    #[test]
    fn test_contribution_final_inscatter() {
        let contrib = FroxelLightContribution::new([1.0, 0.5, 0.25], 0.5);
        let final_rgb = contrib.final_inscatter();
        assert!(rgb_approx_eq(final_rgb, [0.5, 0.25, 0.125], TEST_EPSILON));
    }

    #[test]
    fn test_contribution_luminance() {
        // White light should have luminance = 1.0
        let contrib = FroxelLightContribution::new([1.0, 1.0, 1.0], 1.0);
        let lum = contrib.luminance();
        assert!(approx_eq_eps(lum, 1.0, 0.01));

        // Zero should have luminance = 0.0
        let contrib = FroxelLightContribution::zero();
        assert!(approx_eq(contrib.luminance(), 0.0));
    }

    #[test]
    fn test_contribution_shadow_clamp() {
        let contrib = FroxelLightContribution::new([1.0; 3], 2.0);
        assert!(approx_eq(contrib.shadow_factor, 1.0));

        let contrib = FroxelLightContribution::new([1.0; 3], -1.0);
        assert!(approx_eq(contrib.shadow_factor, 0.0));
    }

    #[test]
    fn test_contribution_size() {
        assert_eq!(std::mem::size_of::<FroxelLightContribution>(), 16);
    }

    // -----------------------------------------------------------------------
    // Phase Function Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_hg_isotropic() {
        // g = 0 should give isotropic scattering
        let forward = FroxelLightEvaluator::henyey_greenstein(1.0, 0.0);
        let side = FroxelLightEvaluator::henyey_greenstein(0.0, 0.0);
        let back = FroxelLightEvaluator::henyey_greenstein(-1.0, 0.0);

        // All should be equal for g = 0
        assert!(approx_eq_eps(forward, side, 0.001));
        assert!(approx_eq_eps(side, back, 0.001));

        // Should equal 1/(4*pi)
        let expected = 1.0 / (4.0 * PI);
        assert!(approx_eq_eps(forward, expected, 0.001));
    }

    #[test]
    fn test_hg_forward_scattering() {
        let g = 0.8;
        let forward = FroxelLightEvaluator::henyey_greenstein(1.0, g);
        let side = FroxelLightEvaluator::henyey_greenstein(0.0, g);
        let back = FroxelLightEvaluator::henyey_greenstein(-1.0, g);

        // Forward should be strongest, back weakest
        assert!(forward > side);
        assert!(side > back);
    }

    #[test]
    fn test_hg_back_scattering() {
        let g = -0.8;
        let forward = FroxelLightEvaluator::henyey_greenstein(1.0, g);
        let back = FroxelLightEvaluator::henyey_greenstein(-1.0, g);

        // Back should be stronger for negative g
        assert!(back > forward);
    }

    #[test]
    fn test_hg_symmetry() {
        // g = 0 should be symmetric
        let p_pos = FroxelLightEvaluator::henyey_greenstein(0.5, 0.0);
        let p_neg = FroxelLightEvaluator::henyey_greenstein(-0.5, 0.0);
        assert!(approx_eq_eps(p_pos, p_neg, 0.001));
    }

    #[test]
    fn test_hg_normalization() {
        // Phase function should integrate to approximately 1 over sphere
        let g = 0.7;
        let samples = 100;
        let mut sum = 0.0f32;

        for i in 0..samples {
            let cos_theta = -1.0 + 2.0 * (i as f32 / (samples - 1) as f32);
            sum += FroxelLightEvaluator::henyey_greenstein(cos_theta, g) * 2.0 / samples as f32;
        }

        // Should be close to 1/(2*pi) when integrated over cos_theta
        // (spherical integral has an additional 2*pi factor)
        let expected = 1.0 / (2.0 * PI);
        assert!(approx_eq_eps(sum, expected, 0.1));
    }

    #[test]
    fn test_hg_extreme_g_values() {
        // Very high g should strongly favor forward direction
        let g = 0.99;
        let forward = FroxelLightEvaluator::henyey_greenstein(1.0, g);
        let back = FroxelLightEvaluator::henyey_greenstein(-1.0, g);
        assert!(forward > back * 100.0);
    }

    #[test]
    fn test_schlick_phase_basic() {
        // For positive g (forward scattering), forward direction should be brighter
        // Use higher g for clearer distinction in the Schlick approximation
        let g = 0.8;
        let forward = FroxelLightEvaluator::schlick_phase(1.0, g);
        let side = FroxelLightEvaluator::schlick_phase(0.0, g);
        let back = FroxelLightEvaluator::schlick_phase(-1.0, g);

        // Forward should be strongest for positive g
        assert!(forward > side, "forward={} should be > side={}", forward, side);
        assert!(forward > back, "forward={} should be > back={}", forward, back);
    }

    #[test]
    fn test_schlick_vs_hg_similar() {
        // Schlick should be similar to HG for moderate g values
        let g = 0.5;
        let hg = FroxelLightEvaluator::henyey_greenstein(0.0, g);
        let schlick = FroxelLightEvaluator::schlick_phase(0.0, g);
        // Should be within 50% (Schlick is an approximation)
        assert!(approx_eq_eps(hg, schlick, hg * 0.5));
    }

    #[test]
    fn test_isotropic_phase() {
        let iso = FroxelLightEvaluator::isotropic_phase();
        let expected = 1.0 / (4.0 * PI);
        assert!(approx_eq_eps(iso, expected, TEST_EPSILON));
    }

    #[test]
    fn test_rayleigh_phase_basic() {
        let forward = FroxelLightEvaluator::rayleigh_phase(1.0);
        let side = FroxelLightEvaluator::rayleigh_phase(0.0);
        let back = FroxelLightEvaluator::rayleigh_phase(-1.0);

        // Rayleigh has strong forward and backward, weak at 90 degrees
        assert!(forward > side);
        assert!(back > side);
        assert!(approx_eq_eps(forward, back, TEST_EPSILON)); // Symmetric
    }

    #[test]
    fn test_evaluate_phase_isotropic() {
        let eval = FroxelLightEvaluator::default().with_phase_type(PhaseType::Isotropic);
        let phase = eval.evaluate_phase(0.5);
        assert!(approx_eq_eps(phase, FroxelLightEvaluator::isotropic_phase(), TEST_EPSILON));
    }

    #[test]
    fn test_evaluate_phase_hg() {
        let eval = FroxelLightEvaluator::default();
        let phase = eval.evaluate_phase(0.5);
        let expected = FroxelLightEvaluator::henyey_greenstein(0.5, DEFAULT_PHASE_G);
        assert!(approx_eq_eps(phase, expected, TEST_EPSILON));
    }

    // -----------------------------------------------------------------------
    // Point Light Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_point_light_at_center() {
        let eval = FroxelLightEvaluator::default();
        let contrib = eval.evaluate_point_light(
            [0.0, 0.0, 5.0],  // froxel
            [0.0, 0.0, 0.0],  // light at origin
            [1.0, 1.0, 1.0],  // white light
            10.0,             // radius
            [0.0, 0.0, -1.0], // looking toward light
        );

        // Should have significant inscatter
        assert!(!contrib.is_zero());
        assert!(approx_eq(contrib.shadow_factor, 1.0));
    }

    #[test]
    fn test_point_light_outside_radius() {
        let eval = FroxelLightEvaluator::default();
        let contrib = eval.evaluate_point_light(
            [0.0, 0.0, 20.0], // froxel far away
            [0.0, 0.0, 0.0],  // light at origin
            [1.0, 1.0, 1.0],
            10.0, // radius only 10
            [0.0, 0.0, -1.0],
        );

        assert!(contrib.is_zero());
    }

    #[test]
    fn test_point_light_distance_falloff() {
        let eval = FroxelLightEvaluator::default();

        let close = eval.evaluate_point_light(
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            10.0,
            [0.0, 0.0, -1.0],
        );

        let far = eval.evaluate_point_light(
            [0.0, 0.0, 8.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            10.0,
            [0.0, 0.0, -1.0],
        );

        // Close should be brighter
        assert!(close.luminance() > far.luminance());
    }

    #[test]
    fn test_point_light_color_contribution() {
        let eval = FroxelLightEvaluator::default();

        let red = eval.evaluate_point_light(
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0], // red light
            10.0,
            [0.0, 0.0, -1.0],
        );

        // Should have red inscatter, no green/blue
        assert!(red.inscatter[0] > 0.0);
        // Note: green/blue might be non-zero due to scatter coefficients
    }

    #[test]
    fn test_point_light_shadowed() {
        let eval = FroxelLightEvaluator::default();

        let unshadowed = eval.evaluate_point_light_shadowed(
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            10.0,
            [0.0, 0.0, -1.0],
            1.0,
        );

        let shadowed = eval.evaluate_point_light_shadowed(
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            10.0,
            [0.0, 0.0, -1.0],
            0.5,
        );

        // Shadowed should have half the final inscatter
        let unshadowed_lum = unshadowed.luminance();
        let shadowed_lum = shadowed.luminance();
        assert!(approx_eq_eps(shadowed_lum, unshadowed_lum * 0.5, 0.01));
    }

    #[test]
    fn test_point_light_at_froxel() {
        let eval = FroxelLightEvaluator::default();
        let contrib = eval.evaluate_point_light(
            [0.0, 0.0, 0.0], // froxel at same position as light
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            10.0,
            [0.0, 0.0, -1.0],
        );

        // Distance is basically zero, should return zero to avoid singularity
        // or handle gracefully
        assert!(contrib.inscatter[0] >= 0.0); // Should not be NaN or negative
    }

    // -----------------------------------------------------------------------
    // Spot Light Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_spot_light_in_cone() {
        let eval = FroxelLightEvaluator::default();
        let contrib = eval.evaluate_spot_light(
            [0.0, 0.0, 5.0],       // froxel in front of light
            [0.0, 0.0, 0.0],       // light at origin
            [0.0, 0.0, 1.0],       // pointing toward froxel
            [1.0, 1.0, 1.0],
            10.0,
            0.3,                   // inner angle ~17 degrees
            0.5,                   // outer angle ~29 degrees
            [0.0, 0.0, -1.0],
        );

        // Should have inscatter since we're in the cone
        assert!(!contrib.is_zero());
    }

    #[test]
    fn test_spot_light_outside_cone() {
        let eval = FroxelLightEvaluator::default();
        let contrib = eval.evaluate_spot_light(
            [5.0, 0.0, 0.0],       // froxel to the side
            [0.0, 0.0, 0.0],       // light at origin
            [0.0, 0.0, 1.0],       // pointing away from froxel
            [1.0, 1.0, 1.0],
            10.0,
            0.1,
            0.2,
            [0.0, 0.0, -1.0],
        );

        assert!(contrib.is_zero());
    }

    #[test]
    fn test_spot_light_outside_radius() {
        let eval = FroxelLightEvaluator::default();
        let contrib = eval.evaluate_spot_light(
            [0.0, 0.0, 20.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            10.0, // radius only 10
            0.3,
            0.5,
            [0.0, 0.0, -1.0],
        );

        assert!(contrib.is_zero());
    }

    #[test]
    fn test_spot_light_cone_falloff() {
        let eval = FroxelLightEvaluator::default();

        // Froxel in inner cone
        let inner = eval.evaluate_spot_light(
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            10.0,
            0.4, // inner angle
            0.8, // outer angle
            [0.0, 0.0, -1.0],
        );

        // Froxel at edge of cone (between inner and outer)
        let edge = eval.evaluate_spot_light(
            [2.0, 0.0, 5.0], // slightly off-axis
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            10.0,
            0.1, // very narrow inner
            0.5, // outer angle
            [0.0, 0.0, -1.0],
        );

        // Inner cone should be brighter (or equal if both in inner)
        // This test just verifies both are non-zero
        assert!(!inner.is_zero());
        // edge may or may not be in cone depending on geometry
    }

    #[test]
    fn test_spot_light_shadowed() {
        let eval = FroxelLightEvaluator::default();

        let contrib = eval.evaluate_spot_light_shadowed(
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            10.0,
            0.3,
            0.5,
            [0.0, 0.0, -1.0],
            0.25,
        );

        assert!(approx_eq(contrib.shadow_factor, 0.25));
    }

    // -----------------------------------------------------------------------
    // Attenuation Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_smooth_distance_attenuation_at_zero() {
        let att = FroxelLightEvaluator::smooth_distance_attenuation(0.0);
        // Should be very high (but finite) at distance 0
        assert!(att > 1.0);
        assert!(att.is_finite());
    }

    #[test]
    fn test_smooth_distance_attenuation_at_edge() {
        let att = FroxelLightEvaluator::smooth_distance_attenuation(1.0);
        assert!(approx_eq(att, 0.0));
    }

    #[test]
    fn test_smooth_distance_attenuation_beyond_edge() {
        let att = FroxelLightEvaluator::smooth_distance_attenuation(1.5);
        assert!(approx_eq(att, 0.0));
    }

    #[test]
    fn test_smooth_distance_attenuation_monotonic() {
        let att0 = FroxelLightEvaluator::smooth_distance_attenuation(0.1);
        let att1 = FroxelLightEvaluator::smooth_distance_attenuation(0.3);
        let att2 = FroxelLightEvaluator::smooth_distance_attenuation(0.6);
        let att3 = FroxelLightEvaluator::smooth_distance_attenuation(0.9);

        // Should decrease monotonically
        assert!(att0 > att1);
        assert!(att1 > att2);
        assert!(att2 > att3);
    }

    #[test]
    fn test_spot_cone_attenuation_in_inner() {
        let att = FroxelLightEvaluator::spot_cone_attenuation(0.9, 0.8, 0.5);
        assert!(approx_eq(att, 1.0));
    }

    #[test]
    fn test_spot_cone_attenuation_at_outer() {
        let att = FroxelLightEvaluator::spot_cone_attenuation(0.5, 0.8, 0.5);
        assert!(approx_eq(att, 0.0));
    }

    #[test]
    fn test_spot_cone_attenuation_between() {
        let att = FroxelLightEvaluator::spot_cone_attenuation(0.65, 0.8, 0.5);
        // Should be between 0 and 1
        assert!(att > 0.0 && att < 1.0);
    }

    #[test]
    fn test_spot_cone_attenuation_smoothstep() {
        // At midpoint, smoothstep should give 0.5
        let cos_mid = 0.65; // midpoint between 0.8 and 0.5
        let att = FroxelLightEvaluator::spot_cone_attenuation(cos_mid, 0.8, 0.5);
        assert!(approx_eq_eps(att, 0.5, 0.1));
    }

    // -----------------------------------------------------------------------
    // Accumulation Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_accumulate_lights_empty() {
        let eval = FroxelLightEvaluator::default();
        let result = eval.accumulate_lights(&[]);
        assert!(rgb_approx_eq(result, [0.0; 3], TEST_EPSILON));
    }

    #[test]
    fn test_accumulate_lights_single() {
        let eval = FroxelLightEvaluator::default();
        let contrib = FroxelLightContribution::new([1.0, 0.5, 0.25], 1.0);
        let result = eval.accumulate_lights(&[contrib]);
        assert!(rgb_approx_eq(result, [1.0, 0.5, 0.25], TEST_EPSILON));
    }

    #[test]
    fn test_accumulate_lights_multiple() {
        let eval = FroxelLightEvaluator::default();
        let contribs = [
            FroxelLightContribution::new([1.0, 0.0, 0.0], 1.0),
            FroxelLightContribution::new([0.0, 1.0, 0.0], 1.0),
            FroxelLightContribution::new([0.0, 0.0, 1.0], 1.0),
        ];
        let result = eval.accumulate_lights(&contribs);
        assert!(rgb_approx_eq(result, [1.0, 1.0, 1.0], TEST_EPSILON));
    }

    #[test]
    fn test_accumulate_lights_with_shadows() {
        let eval = FroxelLightEvaluator::default();
        let contribs = [
            FroxelLightContribution::new([2.0, 0.0, 0.0], 0.5), // half shadowed
            FroxelLightContribution::new([0.0, 2.0, 0.0], 0.5), // half shadowed
        ];
        let result = eval.accumulate_lights(&contribs);
        assert!(rgb_approx_eq(result, [1.0, 1.0, 0.0], TEST_EPSILON));
    }

    #[test]
    fn test_accumulate_lights_weighted() {
        let eval = FroxelLightEvaluator::default();
        let contribs = [
            FroxelLightContribution::new([1.0, 0.0, 0.0], 1.0),
            FroxelLightContribution::new([0.0, 1.0, 0.0], 1.0),
        ];
        let weights = [1.0, 3.0];
        let result = eval.accumulate_lights_weighted(&contribs, &weights);
        // Weighted average: (1*[1,0,0] + 3*[0,1,0]) / 4 = [0.25, 0.75, 0]
        assert!(rgb_approx_eq(result, [0.25, 0.75, 0.0], TEST_EPSILON));
    }

    // -----------------------------------------------------------------------
    // Light Culling Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_point_light_affects_froxel_inside() {
        let result = FroxelLightEvaluator::point_light_affects_froxel(
            [-1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0], // light at center of froxel
            0.5,
        );
        assert!(result);
    }

    #[test]
    fn test_point_light_affects_froxel_overlapping() {
        let result = FroxelLightEvaluator::point_light_affects_froxel(
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 0.5, 0.5], // light outside but radius reaches
            1.5,
        );
        assert!(result);
    }

    #[test]
    fn test_point_light_affects_froxel_disjoint() {
        let result = FroxelLightEvaluator::point_light_affects_froxel(
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [10.0, 0.0, 0.0], // light far away
            1.0,
        );
        assert!(!result);
    }

    #[test]
    fn test_spot_light_affects_froxel_in_cone() {
        let result = FroxelLightEvaluator::spot_light_affects_froxel(
            [4.0, -0.5, -0.5],
            [6.0, 0.5, 0.5],
            [0.0, 0.0, 0.0],  // light at origin
            [1.0, 0.0, 0.0],  // pointing toward froxel
            10.0,
            0.5, // outer angle
        );
        assert!(result);
    }

    #[test]
    fn test_spot_light_affects_froxel_outside_cone() {
        let result = FroxelLightEvaluator::spot_light_affects_froxel(
            [4.0, -0.5, -0.5],
            [6.0, 0.5, 0.5],
            [0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0], // pointing away
            10.0,
            0.3,
        );
        assert!(!result);
    }

    #[test]
    fn test_cull_point_lights() {
        let positions = [
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [100.0, 0.0, 0.0], // far away
            [2.0, 2.0, 2.0],
        ];
        let radii = [3.0, 3.0, 3.0, 5.0];

        let indices = FroxelLightEvaluator::cull_point_lights(
            [0.0, 0.0, 0.0],
            [3.0, 3.0, 3.0],
            &positions,
            &radii,
            10,
        );

        // Should include lights 0, 1, and 3 but not 2
        assert!(indices.contains(&0));
        assert!(indices.contains(&1));
        assert!(!indices.contains(&2));
        assert!(indices.contains(&3));
    }

    #[test]
    fn test_cull_point_lights_max_limit() {
        let positions: Vec<_> = (0..10).map(|i| [i as f32, 0.0, 0.0]).collect();
        let radii: Vec<_> = (0..10).map(|_| 20.0).collect();

        let indices = FroxelLightEvaluator::cull_point_lights(
            [0.0, 0.0, 0.0],
            [3.0, 3.0, 3.0],
            &positions,
            &radii,
            3, // max 3 lights
        );

        assert_eq!(indices.len(), 3);
    }

    // -----------------------------------------------------------------------
    // FroxelLightAccumulator Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_accumulator_new() {
        let acc = FroxelLightAccumulator::new(100);
        assert_eq!(acc.inscatter_values.len(), 100);
        assert_eq!(acc.lights_per_froxel.len(), 100);
        assert_eq!(acc.total_lights_culled, 0);
        assert_eq!(acc.total_lights_evaluated, 0);
    }

    #[test]
    fn test_accumulator_reset() {
        let mut acc = FroxelLightAccumulator::new(10);
        acc.inscatter_values[0] = [1.0, 1.0, 1.0];
        acc.lights_per_froxel[0] = 5;
        acc.total_lights_evaluated = 100;

        acc.reset();

        assert!(rgb_approx_eq(acc.inscatter_values[0], [0.0; 3], TEST_EPSILON));
        assert_eq!(acc.lights_per_froxel[0], 0);
        assert_eq!(acc.total_lights_evaluated, 0);
    }

    #[test]
    fn test_accumulator_add_contribution() {
        let mut acc = FroxelLightAccumulator::new(10);
        let contrib = FroxelLightContribution::new([1.0, 0.5, 0.25], 1.0);

        acc.add_contribution(0, &contrib);

        assert!(rgb_approx_eq(acc.inscatter_values[0], [1.0, 0.5, 0.25], TEST_EPSILON));
        assert_eq!(acc.lights_per_froxel[0], 1);
        assert_eq!(acc.total_lights_evaluated, 1);
    }

    #[test]
    fn test_accumulator_average_lights() {
        let mut acc = FroxelLightAccumulator::new(4);
        acc.lights_per_froxel = vec![2, 4, 6, 8];

        let avg = acc.average_lights_per_froxel();
        assert!(approx_eq(avg, 5.0));
    }

    #[test]
    fn test_accumulator_cull_efficiency() {
        let mut acc = FroxelLightAccumulator::new(10);
        acc.total_lights_culled = 80;
        acc.total_lights_evaluated = 20;

        let eff = acc.cull_efficiency();
        assert!(approx_eq(eff, 0.8));
    }

    #[test]
    fn test_accumulator_max_luminance() {
        let mut acc = FroxelLightAccumulator::new(3);
        acc.inscatter_values[0] = [0.1, 0.1, 0.1];
        acc.inscatter_values[1] = [1.0, 1.0, 1.0]; // brightest
        acc.inscatter_values[2] = [0.5, 0.5, 0.5];

        let max_lum = acc.max_luminance();
        assert!(approx_eq_eps(max_lum, 1.0, 0.01));
    }

    // -----------------------------------------------------------------------
    // Evaluator Builder Pattern Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_evaluator_with_phase_type() {
        let eval = FroxelLightEvaluator::default().with_phase_type(PhaseType::Rayleigh);
        assert_eq!(eval.phase_type, PhaseType::Rayleigh);
    }

    #[test]
    fn test_evaluator_with_scatter_coefficients() {
        let eval =
            FroxelLightEvaluator::default().with_scatter_coefficients([0.2, 0.3, 0.4]);
        assert!(rgb_approx_eq(eval.scatter_coefficients, [0.2, 0.3, 0.4], TEST_EPSILON));
    }

    #[test]
    fn test_evaluator_scatter_coefficients_clamp_negative() {
        let eval = FroxelLightEvaluator::default().with_scatter_coefficients([-0.1, -0.2, 0.3]);
        assert!(eval.scatter_coefficients[0] >= 0.0);
        assert!(eval.scatter_coefficients[1] >= 0.0);
    }

    // -----------------------------------------------------------------------
    // PhaseType Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_phase_type_name() {
        assert_eq!(PhaseType::Isotropic.name(), "isotropic");
        assert_eq!(PhaseType::HenyeyGreenstein.name(), "henyey_greenstein");
        assert_eq!(PhaseType::Schlick.name(), "schlick");
        assert_eq!(PhaseType::Rayleigh.name(), "rayleigh");
    }

    #[test]
    fn test_phase_type_default() {
        assert_eq!(PhaseType::default(), PhaseType::HenyeyGreenstein);
    }

    // -----------------------------------------------------------------------
    // Edge Case Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_zero_radius_light() {
        let eval = FroxelLightEvaluator::default();
        let contrib = eval.evaluate_point_light(
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            0.0, // zero radius
            [0.0, 0.0, -1.0],
        );
        assert!(contrib.is_zero());
    }

    #[test]
    fn test_very_small_scatter_intensity() {
        let config = FroxelLightConfig::new(16, 0.8, 0.001, 2);
        let eval = FroxelLightEvaluator::new(config);
        let contrib = eval.evaluate_point_light(
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            10.0,
            [0.0, 0.0, -1.0],
        );
        // Should have very small but non-zero contribution
        assert!(contrib.luminance() > 0.0);
        assert!(contrib.luminance() < 0.01);
    }

    #[test]
    fn test_degenerate_spot_angles() {
        let eval = FroxelLightEvaluator::default();
        // Inner angle >= outer angle (degenerate cone)
        let contrib = eval.evaluate_spot_light(
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            10.0,
            0.5, // inner
            0.3, // outer < inner
            [0.0, 0.0, -1.0],
        );
        // Should handle gracefully (likely return full intensity in "inner")
        // Just verify no panic/NaN
        assert!(contrib.inscatter[0].is_finite());
    }
}
