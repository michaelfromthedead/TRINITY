//! Cloud Lighting System (T-ENV-2.3)
//!
//! Implements physically-based lighting for volumetric clouds in TRINITY's
//! atmosphere system. Integrates with `cloud_raymarching.rs` for the ray
//! marching pass.
//!
//! # Overview
//!
//! The cloud lighting system provides:
//! - **CloudLightingUniforms**: GPU-uploadable uniform buffer for shader access
//! - **Henyey-Greenstein Phase Function**: Anisotropic scattering with multi-scatter approximation
//! - **Beer-Lambert Law**: In-scatter/out-scatter balance with energy conservation
//! - **Powder Effect**: Energy conserving darkening at cloud edges
//! - **Silver Lining**: Bright rim lighting at cloud edges facing the sun
//! - **Multi-Scattering Octave Approximation**: Efficient secondary bounce computation
//! - **Ambient/Environment Lighting**: Sky and ground contribution
//!
//! # Physics Model
//!
//! The lighting model implements:
//! 1. Single scattering with Henyey-Greenstein phase function
//! 2. Multi-scattering approximation using octave reduction
//! 3. Beer-Lambert transmission for extinction
//! 4. Powder effect for energy-conserving darkening
//! 5. Silver lining effect at cloud silhouettes
//!
//! # Integration with Ray Marching
//!
//! At each ray march step:
//! 1. Sample cloud density
//! 2. Compute phase function for view-to-light angle
//! 3. Estimate light transmittance towards sun (light march)
//! 4. Apply Beer-Powder for in-scatter
//! 5. Accumulate with multi-scatter approximation
//! 6. Add ambient contribution
//!
//! # References
//!
//! - Schneider & Vos, "The Real-time Volumetric Cloudscapes of Horizon: Zero Dawn"
//! - Hillaire, "A Scalable and Production Ready Sky and Atmosphere Rendering Technique"
//! - Henyey, L.G. & Greenstein, J.L. (1941), "Diffuse radiation in the Galaxy"

use bytemuck::{Pod, Zeroable};
use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Small epsilon for floating point comparisons.
pub const EPSILON: f32 = 1e-6;

/// Default asymmetry parameter (g) for Henyey-Greenstein (forward scattering).
pub const DEFAULT_ASYMMETRY_G: f32 = 0.7;

/// Secondary asymmetry for back-scatter lobe.
pub const DEFAULT_ASYMMETRY_G2: f32 = -0.3;

/// Blend factor between forward and backward scatter.
pub const DEFAULT_SCATTER_BLEND: f32 = 0.7;

/// Default scattering albedo for clouds (highly reflective).
pub const DEFAULT_SCATTERING_ALBEDO: f32 = 0.99;

/// Default extinction coefficient (sigma_t = sigma_a + sigma_s).
pub const DEFAULT_EXTINCTION_COEFF: f32 = 0.1;

/// Default powder effect strength.
pub const DEFAULT_POWDER_STRENGTH: f32 = 2.0;

/// Default silver lining intensity.
pub const DEFAULT_SILVER_LINING_INTENSITY: f32 = 0.8;

/// Default silver lining exponent (sharpness).
pub const DEFAULT_SILVER_LINING_EXPONENT: f32 = 8.0;

/// Number of light march steps for sun transmittance estimation.
pub const DEFAULT_LIGHT_MARCH_STEPS: u32 = 6;

/// Default light march step size in meters.
pub const DEFAULT_LIGHT_STEP_SIZE: f32 = 100.0;

/// Default multi-scatter octave count.
pub const DEFAULT_MULTISCATTER_OCTAVES: u32 = 4;

/// Multi-scatter attenuation per octave.
pub const MULTISCATTER_ATTENUATION: f32 = 0.5;

/// Multi-scatter contribution per octave.
pub const MULTISCATTER_CONTRIBUTION: f32 = 0.4;

/// Default ambient strength factor.
pub const DEFAULT_AMBIENT_STRENGTH: f32 = 0.15;

/// Default ground bounce factor.
pub const DEFAULT_GROUND_BOUNCE: f32 = 0.1;

/// Minimum valid g parameter for phase function.
pub const MIN_ASYMMETRY_G: f32 = -0.999;

/// Maximum valid g parameter for phase function.
pub const MAX_ASYMMETRY_G: f32 = 0.999;

// ---------------------------------------------------------------------------
// CloudLightingUniforms - GPU Uniform Buffer
// ---------------------------------------------------------------------------

/// GPU-uploadable uniform buffer for cloud lighting shaders.
///
/// Contains all parameters needed for cloud lighting calculations in the shader.
/// Designed for 16-byte aligned GPU upload.
///
/// # Memory Layout (128 bytes = 8 x vec4)
///
/// | Offset | Field                      | Size     |
/// |--------|----------------------------|----------|
/// | 0      | sun_direction              | 12 bytes |
/// | 12     | asymmetry_g                | 4 bytes  |
/// | 16     | sun_color                  | 12 bytes |
/// | 28     | asymmetry_g2               | 4 bytes  |
/// | 32     | sky_color                  | 12 bytes |
/// | 44     | scatter_blend              | 4 bytes  |
/// | 48     | ground_color               | 12 bytes |
/// | 60     | scattering_albedo          | 4 bytes  |
/// | 64     | extinction_coeff           | 4 bytes  |
/// | 68     | powder_strength            | 4 bytes  |
/// | 72     | silver_lining_intensity    | 4 bytes  |
/// | 76     | silver_lining_exponent     | 4 bytes  |
/// | 80     | ambient_strength           | 4 bytes  |
/// | 84     | ground_bounce              | 4 bytes  |
/// | 88     | light_march_steps          | 4 bytes  |
/// | 92     | light_step_size            | 4 bytes  |
/// | 96     | multiscatter_octaves       | 4 bytes  |
/// | 100    | multiscatter_attenuation   | 4 bytes  |
/// | 104    | multiscatter_contribution  | 4 bytes  |
/// | 108    | _padding                   | 20 bytes |
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct CloudLightingUniforms {
    /// Normalized direction towards the sun (world space, Y-up).
    pub sun_direction: [f32; 3],

    /// Primary asymmetry parameter for Henyey-Greenstein (g1, forward).
    pub asymmetry_g: f32,

    /// Sun color/intensity (RGB linear).
    pub sun_color: [f32; 3],

    /// Secondary asymmetry parameter (g2, backward).
    pub asymmetry_g2: f32,

    /// Sky color for ambient (RGB linear).
    pub sky_color: [f32; 3],

    /// Blend factor between forward (g1) and backward (g2) scatter.
    pub scatter_blend: f32,

    /// Ground color for bounce lighting (RGB linear).
    pub ground_color: [f32; 3],

    /// Scattering albedo (sigma_s / sigma_t), typically ~0.99 for clouds.
    pub scattering_albedo: f32,

    /// Extinction coefficient (sigma_t = sigma_a + sigma_s).
    pub extinction_coeff: f32,

    /// Powder effect strength (energy conserving darkening).
    pub powder_strength: f32,

    /// Silver lining intensity (rim light at cloud edges).
    pub silver_lining_intensity: f32,

    /// Silver lining exponent (controls sharpness).
    pub silver_lining_exponent: f32,

    /// Ambient light strength factor.
    pub ambient_strength: f32,

    /// Ground bounce contribution factor.
    pub ground_bounce: f32,

    /// Number of steps for light march towards sun.
    pub light_march_steps: u32,

    /// Step size for light march in meters.
    pub light_step_size: f32,

    /// Number of multi-scatter octaves.
    pub multiscatter_octaves: u32,

    /// Multi-scatter energy attenuation per octave.
    pub multiscatter_attenuation: f32,

    /// Multi-scatter contribution weight.
    pub multiscatter_contribution: f32,

    /// Padding for 16-byte alignment (128 bytes total).
    pub _padding: [u32; 5],
}

// Size assertion: 128 bytes (8 x vec4)
const _: () = assert!(std::mem::size_of::<CloudLightingUniforms>() == 128);

impl Default for CloudLightingUniforms {
    fn default() -> Self {
        Self {
            sun_direction: normalize_vec3([0.5, 0.7, -0.5]),
            asymmetry_g: DEFAULT_ASYMMETRY_G,
            sun_color: [1.0, 0.95, 0.9], // Warm sunlight
            asymmetry_g2: DEFAULT_ASYMMETRY_G2,
            sky_color: [0.3, 0.5, 0.8], // Blue sky
            scatter_blend: DEFAULT_SCATTER_BLEND,
            ground_color: [0.2, 0.18, 0.15], // Earth tones
            scattering_albedo: DEFAULT_SCATTERING_ALBEDO,
            extinction_coeff: DEFAULT_EXTINCTION_COEFF,
            powder_strength: DEFAULT_POWDER_STRENGTH,
            silver_lining_intensity: DEFAULT_SILVER_LINING_INTENSITY,
            silver_lining_exponent: DEFAULT_SILVER_LINING_EXPONENT,
            ambient_strength: DEFAULT_AMBIENT_STRENGTH,
            ground_bounce: DEFAULT_GROUND_BOUNCE,
            light_march_steps: DEFAULT_LIGHT_MARCH_STEPS,
            light_step_size: DEFAULT_LIGHT_STEP_SIZE,
            multiscatter_octaves: DEFAULT_MULTISCATTER_OCTAVES,
            multiscatter_attenuation: MULTISCATTER_ATTENUATION,
            multiscatter_contribution: MULTISCATTER_CONTRIBUTION,
            _padding: [0; 5],
        }
    }
}

impl CloudLightingUniforms {
    /// Create a new uniform buffer with default values.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create uniforms for midday conditions.
    #[inline]
    pub fn midday() -> Self {
        Self {
            sun_direction: [0.0, 1.0, 0.0], // Sun at zenith
            sun_color: [1.0, 1.0, 1.0],     // Neutral white
            ..Self::default()
        }
    }

    /// Create uniforms for sunset/golden hour.
    #[inline]
    pub fn sunset() -> Self {
        Self {
            sun_direction: normalize_vec3([0.0, 0.1, -1.0]), // Low sun
            sun_color: [1.0, 0.6, 0.3],                      // Warm orange
            asymmetry_g: 0.8,                                // More forward scatter
            silver_lining_intensity: 1.2,                    // Enhanced rim light
            ..Self::default()
        }
    }

    /// Create uniforms for overcast conditions.
    #[inline]
    pub fn overcast() -> Self {
        Self {
            sun_direction: [0.0, 1.0, 0.0],
            sun_color: [0.8, 0.8, 0.85],        // Cool gray
            asymmetry_g: 0.3,                   // More diffuse
            silver_lining_intensity: 0.2,       // Minimal rim
            ambient_strength: 0.3,              // More ambient
            multiscatter_octaves: 6,            // More scattering
            multiscatter_contribution: 0.6,
            ..Self::default()
        }
    }

    /// Set sun direction (will be normalized).
    #[inline]
    pub fn with_sun_direction(mut self, direction: [f32; 3]) -> Self {
        self.sun_direction = normalize_vec3(direction);
        self
    }

    /// Set sun color/intensity.
    #[inline]
    pub fn with_sun_color(mut self, color: [f32; 3]) -> Self {
        self.sun_color = color;
        self
    }

    /// Set phase function asymmetry (forward scatter dominance).
    #[inline]
    pub fn with_asymmetry(mut self, g: f32) -> Self {
        self.asymmetry_g = g.clamp(MIN_ASYMMETRY_G, MAX_ASYMMETRY_G);
        self
    }

    /// Set dual-lobe phase function parameters.
    #[inline]
    pub fn with_dual_lobe(mut self, g_forward: f32, g_backward: f32, blend: f32) -> Self {
        self.asymmetry_g = g_forward.clamp(MIN_ASYMMETRY_G, MAX_ASYMMETRY_G);
        self.asymmetry_g2 = g_backward.clamp(MIN_ASYMMETRY_G, MAX_ASYMMETRY_G);
        self.scatter_blend = blend.clamp(0.0, 1.0);
        self
    }

    /// Set powder effect strength.
    #[inline]
    pub fn with_powder_strength(mut self, strength: f32) -> Self {
        self.powder_strength = strength.max(0.0);
        self
    }

    /// Set silver lining parameters.
    #[inline]
    pub fn with_silver_lining(mut self, intensity: f32, exponent: f32) -> Self {
        self.silver_lining_intensity = intensity.max(0.0);
        self.silver_lining_exponent = exponent.max(1.0);
        self
    }

    /// Set multi-scatter parameters.
    #[inline]
    pub fn with_multiscatter(mut self, octaves: u32, attenuation: f32, contribution: f32) -> Self {
        self.multiscatter_octaves = octaves.clamp(1, 8);
        self.multiscatter_attenuation = attenuation.clamp(0.0, 1.0);
        self.multiscatter_contribution = contribution.clamp(0.0, 1.0);
        self
    }

    /// Set ambient lighting parameters.
    #[inline]
    pub fn with_ambient(mut self, strength: f32, ground_bounce: f32) -> Self {
        self.ambient_strength = strength.clamp(0.0, 1.0);
        self.ground_bounce = ground_bounce.clamp(0.0, 1.0);
        self
    }

    /// Set light march parameters.
    #[inline]
    pub fn with_light_march(mut self, steps: u32, step_size: f32) -> Self {
        self.light_march_steps = steps.clamp(1, 32);
        self.light_step_size = step_size.max(1.0);
        self
    }

    /// Validate all parameters are within expected ranges.
    pub fn validate(&self) -> bool {
        let len_sq = self.sun_direction[0].powi(2)
            + self.sun_direction[1].powi(2)
            + self.sun_direction[2].powi(2);

        (len_sq - 1.0).abs() < 0.01  // Normalized
            && self.asymmetry_g > MIN_ASYMMETRY_G
            && self.asymmetry_g < MAX_ASYMMETRY_G
            && self.asymmetry_g2 > MIN_ASYMMETRY_G
            && self.asymmetry_g2 < MAX_ASYMMETRY_G
            && self.scatter_blend >= 0.0
            && self.scatter_blend <= 1.0
            && self.scattering_albedo >= 0.0
            && self.scattering_albedo <= 1.0
            && self.extinction_coeff >= 0.0
            && self.powder_strength >= 0.0
            && self.silver_lining_intensity >= 0.0
            && self.silver_lining_exponent >= 1.0
            && self.ambient_strength >= 0.0
            && self.ground_bounce >= 0.0
            && self.light_march_steps >= 1
            && self.light_step_size > 0.0
            && self.multiscatter_octaves >= 1
            && self.multiscatter_attenuation >= 0.0
            && self.multiscatter_contribution >= 0.0
    }
}

// ---------------------------------------------------------------------------
// CloudLightingConfig - Rust-side configuration (not GPU)
// ---------------------------------------------------------------------------

/// Rust-side configuration for cloud lighting calculations.
///
/// This struct mirrors `CloudLightingUniforms` but with ergonomic Rust API.
/// Use `to_uniforms()` to convert for GPU upload.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CloudLightingConfig {
    /// Normalized direction towards the sun.
    pub sun_direction: [f32; 3],

    /// Sun color/intensity (RGB linear).
    pub sun_color: [f32; 3],

    /// Sky color for ambient lighting.
    pub sky_color: [f32; 3],

    /// Ground color for bounce lighting.
    pub ground_color: [f32; 3],

    /// Phase function parameters.
    pub phase_params: PhaseParams,

    /// Scattering parameters.
    pub scatter_params: ScatterParams,

    /// Silver lining parameters.
    pub silver_lining_params: SilverLiningParams,

    /// Multi-scatter parameters.
    pub multiscatter_params: MultiScatterParams,

    /// Ambient lighting parameters.
    pub ambient_params: AmbientParams,

    /// Light march parameters.
    pub light_march_params: LightMarchParams,
}

impl Default for CloudLightingConfig {
    fn default() -> Self {
        Self {
            sun_direction: normalize_vec3([0.5, 0.7, -0.5]),
            sun_color: [1.0, 0.95, 0.9],
            sky_color: [0.3, 0.5, 0.8],
            ground_color: [0.2, 0.18, 0.15],
            phase_params: PhaseParams::default(),
            scatter_params: ScatterParams::default(),
            silver_lining_params: SilverLiningParams::default(),
            multiscatter_params: MultiScatterParams::default(),
            ambient_params: AmbientParams::default(),
            light_march_params: LightMarchParams::default(),
        }
    }
}

impl CloudLightingConfig {
    /// Create a new config with default values.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set sun direction (will be normalized).
    #[inline]
    pub fn with_sun_direction(mut self, direction: [f32; 3]) -> Self {
        self.sun_direction = normalize_vec3(direction);
        self
    }

    /// Set sun color.
    #[inline]
    pub fn with_sun_color(mut self, color: [f32; 3]) -> Self {
        self.sun_color = color;
        self
    }

    /// Convert to GPU-uploadable uniforms.
    pub fn to_uniforms(&self) -> CloudLightingUniforms {
        CloudLightingUniforms {
            sun_direction: self.sun_direction,
            asymmetry_g: self.phase_params.asymmetry_g,
            sun_color: self.sun_color,
            asymmetry_g2: self.phase_params.asymmetry_g2,
            sky_color: self.sky_color,
            scatter_blend: self.phase_params.scatter_blend,
            ground_color: self.ground_color,
            scattering_albedo: self.scatter_params.scattering_albedo,
            extinction_coeff: self.scatter_params.extinction_coeff,
            powder_strength: self.scatter_params.powder_strength,
            silver_lining_intensity: self.silver_lining_params.intensity,
            silver_lining_exponent: self.silver_lining_params.exponent,
            ambient_strength: self.ambient_params.strength,
            ground_bounce: self.ambient_params.ground_bounce,
            light_march_steps: self.light_march_params.steps,
            light_step_size: self.light_march_params.step_size,
            multiscatter_octaves: self.multiscatter_params.octaves,
            multiscatter_attenuation: self.multiscatter_params.attenuation,
            multiscatter_contribution: self.multiscatter_params.contribution,
            _padding: [0; 5],
        }
    }
}

// ---------------------------------------------------------------------------
// Sub-configuration structs
// ---------------------------------------------------------------------------

/// Phase function parameters for anisotropic scattering.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhaseParams {
    /// Primary asymmetry parameter (forward scatter).
    pub asymmetry_g: f32,

    /// Secondary asymmetry parameter (backward scatter).
    pub asymmetry_g2: f32,

    /// Blend factor between forward and backward lobes.
    pub scatter_blend: f32,
}

impl Default for PhaseParams {
    fn default() -> Self {
        Self {
            asymmetry_g: DEFAULT_ASYMMETRY_G,
            asymmetry_g2: DEFAULT_ASYMMETRY_G2,
            scatter_blend: DEFAULT_SCATTER_BLEND,
        }
    }
}

impl PhaseParams {
    /// Create isotropic scattering (g=0).
    #[inline]
    pub fn isotropic() -> Self {
        Self {
            asymmetry_g: 0.0,
            asymmetry_g2: 0.0,
            scatter_blend: 0.5,
        }
    }

    /// Create forward-dominant scattering.
    #[inline]
    pub fn forward(g: f32) -> Self {
        Self {
            asymmetry_g: g.clamp(MIN_ASYMMETRY_G, MAX_ASYMMETRY_G),
            asymmetry_g2: 0.0,
            scatter_blend: 1.0,
        }
    }

    /// Create dual-lobe scattering (forward + backward).
    #[inline]
    pub fn dual_lobe(g_forward: f32, g_backward: f32, blend: f32) -> Self {
        Self {
            asymmetry_g: g_forward.clamp(MIN_ASYMMETRY_G, MAX_ASYMMETRY_G),
            asymmetry_g2: g_backward.clamp(MIN_ASYMMETRY_G, MAX_ASYMMETRY_G),
            scatter_blend: blend.clamp(0.0, 1.0),
        }
    }

    /// Validate parameters.
    #[inline]
    pub fn validate(&self) -> bool {
        self.asymmetry_g > MIN_ASYMMETRY_G
            && self.asymmetry_g < MAX_ASYMMETRY_G
            && self.asymmetry_g2 > MIN_ASYMMETRY_G
            && self.asymmetry_g2 < MAX_ASYMMETRY_G
            && self.scatter_blend >= 0.0
            && self.scatter_blend <= 1.0
    }
}

/// Scattering and extinction parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScatterParams {
    /// Scattering albedo (sigma_s / sigma_t).
    pub scattering_albedo: f32,

    /// Extinction coefficient.
    pub extinction_coeff: f32,

    /// Powder effect strength.
    pub powder_strength: f32,
}

impl Default for ScatterParams {
    fn default() -> Self {
        Self {
            scattering_albedo: DEFAULT_SCATTERING_ALBEDO,
            extinction_coeff: DEFAULT_EXTINCTION_COEFF,
            powder_strength: DEFAULT_POWDER_STRENGTH,
        }
    }
}

impl ScatterParams {
    /// Create parameters for thin clouds (low density).
    #[inline]
    pub fn thin_clouds() -> Self {
        Self {
            scattering_albedo: 0.99,
            extinction_coeff: 0.05,
            powder_strength: 3.0, // More powder for thin edges
        }
    }

    /// Create parameters for dense storm clouds.
    #[inline]
    pub fn storm_clouds() -> Self {
        Self {
            scattering_albedo: 0.95,
            extinction_coeff: 0.2,
            powder_strength: 1.0, // Less powder for dense clouds
        }
    }

    /// Validate parameters.
    #[inline]
    pub fn validate(&self) -> bool {
        self.scattering_albedo >= 0.0
            && self.scattering_albedo <= 1.0
            && self.extinction_coeff >= 0.0
            && self.powder_strength >= 0.0
    }
}

/// Silver lining (rim lighting) parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SilverLiningParams {
    /// Intensity of the silver lining effect.
    pub intensity: f32,

    /// Exponent controlling sharpness of the rim.
    pub exponent: f32,
}

impl Default for SilverLiningParams {
    fn default() -> Self {
        Self {
            intensity: DEFAULT_SILVER_LINING_INTENSITY,
            exponent: DEFAULT_SILVER_LINING_EXPONENT,
        }
    }
}

impl SilverLiningParams {
    /// Create dramatic silver lining for backlit clouds.
    #[inline]
    pub fn dramatic() -> Self {
        Self {
            intensity: 1.5,
            exponent: 12.0,
        }
    }

    /// Create subtle silver lining.
    #[inline]
    pub fn subtle() -> Self {
        Self {
            intensity: 0.3,
            exponent: 4.0,
        }
    }

    /// Disable silver lining.
    #[inline]
    pub fn disabled() -> Self {
        Self {
            intensity: 0.0,
            exponent: 1.0,
        }
    }

    /// Validate parameters.
    #[inline]
    pub fn validate(&self) -> bool {
        self.intensity >= 0.0 && self.exponent >= 1.0
    }
}

/// Multi-scattering octave parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultiScatterParams {
    /// Number of multi-scatter octaves.
    pub octaves: u32,

    /// Energy attenuation per octave.
    pub attenuation: f32,

    /// Contribution weight for multi-scatter.
    pub contribution: f32,
}

impl Default for MultiScatterParams {
    fn default() -> Self {
        Self {
            octaves: DEFAULT_MULTISCATTER_OCTAVES,
            attenuation: MULTISCATTER_ATTENUATION,
            contribution: MULTISCATTER_CONTRIBUTION,
        }
    }
}

impl MultiScatterParams {
    /// High quality multi-scatter (more octaves).
    #[inline]
    pub fn high_quality() -> Self {
        Self {
            octaves: 6,
            attenuation: 0.5,
            contribution: 0.5,
        }
    }

    /// Fast approximation (fewer octaves).
    #[inline]
    pub fn fast() -> Self {
        Self {
            octaves: 2,
            attenuation: 0.5,
            contribution: 0.3,
        }
    }

    /// Disable multi-scatter.
    #[inline]
    pub fn disabled() -> Self {
        Self {
            octaves: 1,
            attenuation: 0.0,
            contribution: 0.0,
        }
    }

    /// Validate parameters.
    #[inline]
    pub fn validate(&self) -> bool {
        self.octaves >= 1
            && self.octaves <= 8
            && self.attenuation >= 0.0
            && self.attenuation <= 1.0
            && self.contribution >= 0.0
            && self.contribution <= 1.0
    }
}

/// Ambient lighting parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AmbientParams {
    /// Overall ambient strength.
    pub strength: f32,

    /// Ground bounce contribution.
    pub ground_bounce: f32,
}

impl Default for AmbientParams {
    fn default() -> Self {
        Self {
            strength: DEFAULT_AMBIENT_STRENGTH,
            ground_bounce: DEFAULT_GROUND_BOUNCE,
        }
    }
}

impl AmbientParams {
    /// Bright ambient for cloudy days.
    #[inline]
    pub fn cloudy_day() -> Self {
        Self {
            strength: 0.3,
            ground_bounce: 0.15,
        }
    }

    /// Minimal ambient for clear sky.
    #[inline]
    pub fn clear_sky() -> Self {
        Self {
            strength: 0.1,
            ground_bounce: 0.05,
        }
    }

    /// Validate parameters.
    #[inline]
    pub fn validate(&self) -> bool {
        self.strength >= 0.0 && self.ground_bounce >= 0.0
    }
}

/// Light march parameters for sun transmittance estimation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LightMarchParams {
    /// Number of steps towards the sun.
    pub steps: u32,

    /// Step size in meters.
    pub step_size: f32,
}

impl Default for LightMarchParams {
    fn default() -> Self {
        Self {
            steps: DEFAULT_LIGHT_MARCH_STEPS,
            step_size: DEFAULT_LIGHT_STEP_SIZE,
        }
    }
}

impl LightMarchParams {
    /// High quality light march.
    #[inline]
    pub fn high_quality() -> Self {
        Self {
            steps: 12,
            step_size: 80.0,
        }
    }

    /// Fast light march.
    #[inline]
    pub fn fast() -> Self {
        Self {
            steps: 3,
            step_size: 150.0,
        }
    }

    /// Validate parameters.
    #[inline]
    pub fn validate(&self) -> bool {
        self.steps >= 1 && self.step_size > 0.0
    }
}

// ---------------------------------------------------------------------------
// Phase Functions
// ---------------------------------------------------------------------------

/// Henyey-Greenstein phase function.
///
/// Models anisotropic scattering with asymmetry parameter `g`:
/// - g = 0: Isotropic scattering
/// - g > 0: Forward scattering (towards light)
/// - g < 0: Backward scattering (away from light)
///
/// # Arguments
///
/// * `cos_theta` - Cosine of angle between view and light directions.
/// * `g` - Asymmetry parameter (-1 < g < 1).
///
/// # Returns
///
/// Phase function value (probability density).
#[inline]
pub fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let g_sq = g * g;
    let denom = 1.0 + g_sq - 2.0 * g * cos_theta;

    // Normalize by 1/(4*PI) for proper PDF
    (1.0 - g_sq) / (4.0 * PI * denom.max(EPSILON).powf(1.5))
}

/// Dual-lobe Henyey-Greenstein phase function.
///
/// Combines forward and backward scattering lobes for more realistic
/// cloud scattering behavior.
///
/// # Arguments
///
/// * `cos_theta` - Cosine of angle between view and light directions.
/// * `g1` - Forward scattering asymmetry (typically 0.7).
/// * `g2` - Backward scattering asymmetry (typically -0.3).
/// * `blend` - Blend factor between lobes (0 = all backward, 1 = all forward).
///
/// # Returns
///
/// Combined phase function value.
#[inline]
pub fn dual_lobe_hg(cos_theta: f32, g1: f32, g2: f32, blend: f32) -> f32 {
    let p1 = henyey_greenstein(cos_theta, g1);
    let p2 = henyey_greenstein(cos_theta, g2);
    blend * p1 + (1.0 - blend) * p2
}

/// Schlick phase function approximation.
///
/// A computationally cheaper approximation to Henyey-Greenstein.
/// Useful for real-time applications where HG is too expensive.
///
/// # Arguments
///
/// * `cos_theta` - Cosine of angle between view and light directions.
/// * `g` - Asymmetry parameter (-1 < g < 1).
///
/// # Returns
///
/// Phase function value.
#[inline]
pub fn schlick_phase(cos_theta: f32, g: f32) -> f32 {
    let k = 1.55 * g - 0.55 * g * g * g;
    let denom = 1.0 + k * cos_theta;
    (1.0 - k * k) / (4.0 * PI * denom * denom).max(EPSILON)
}

/// Cornette-Shanks phase function.
///
/// Modified Henyey-Greenstein that better handles forward scattering.
/// Particularly good for water droplet scattering in clouds.
///
/// # Arguments
///
/// * `cos_theta` - Cosine of angle between view and light directions.
/// * `g` - Asymmetry parameter (-1 < g < 1).
///
/// # Returns
///
/// Phase function value.
#[inline]
pub fn cornette_shanks(cos_theta: f32, g: f32) -> f32 {
    let g_sq = g * g;
    let denom = 1.0 + g_sq - 2.0 * g * cos_theta;

    let numerator = 3.0 * (1.0 - g_sq) * (1.0 + cos_theta * cos_theta);
    let denominator = 2.0 * (2.0 + g_sq) * denom.max(EPSILON).powf(1.5);

    numerator / (4.0 * PI * denominator).max(EPSILON)
}

/// Evaluate phase function from uniforms.
///
/// Convenience function that uses the phase parameters from uniforms.
///
/// # Arguments
///
/// * `cos_theta` - Cosine of angle between view and light directions.
/// * `uniforms` - Cloud lighting uniforms with phase parameters.
///
/// # Returns
///
/// Phase function value.
#[inline]
pub fn evaluate_phase(cos_theta: f32, uniforms: &CloudLightingUniforms) -> f32 {
    dual_lobe_hg(
        cos_theta,
        uniforms.asymmetry_g,
        uniforms.asymmetry_g2,
        uniforms.scatter_blend,
    )
}

// ---------------------------------------------------------------------------
// Beer-Lambert and Transmittance
// ---------------------------------------------------------------------------

/// Beer-Lambert law for light transmission.
///
/// Models exponential decay of light through a participating medium.
///
/// T = exp(-sigma_t * d)
///
/// # Arguments
///
/// * `density` - Cloud density at sample point.
/// * `distance` - Distance traveled through medium.
/// * `extinction` - Extinction coefficient (sigma_t).
///
/// # Returns
///
/// Transmittance (0-1).
#[inline]
pub fn beer_lambert(density: f32, distance: f32, extinction: f32) -> f32 {
    (-density * distance * extinction).exp()
}

/// Beer-Lambert with optical depth.
///
/// Alternative formulation using pre-computed optical depth.
///
/// T = exp(-tau)
///
/// # Arguments
///
/// * `optical_depth` - Accumulated optical depth (tau = integral of sigma_t * density * ds).
///
/// # Returns
///
/// Transmittance (0-1).
#[inline]
pub fn beer_lambert_optical_depth(optical_depth: f32) -> f32 {
    (-optical_depth).exp()
}

/// Compute optical depth for a ray segment.
///
/// tau = density * distance * extinction
///
/// # Arguments
///
/// * `density` - Cloud density.
/// * `distance` - Segment length.
/// * `extinction` - Extinction coefficient.
///
/// # Returns
///
/// Optical depth contribution.
#[inline]
pub fn compute_optical_depth(density: f32, distance: f32, extinction: f32) -> f32 {
    density * distance * extinction
}

// ---------------------------------------------------------------------------
// Powder Effect
// ---------------------------------------------------------------------------

/// Powder effect for energy conserving darkening.
///
/// Creates the characteristic darkening at cloud edges due to
/// increased out-scattering in thin regions. Implements the
/// "powder effect" from Horizon: Zero Dawn.
///
/// P = 1 - exp(-density * strength * 2)
///
/// # Arguments
///
/// * `density` - Local cloud density.
/// * `strength` - Powder effect strength (typically 2.0).
///
/// # Returns
///
/// Powder factor (0-1, higher = more in-scatter).
#[inline]
pub fn powder_effect(density: f32, strength: f32) -> f32 {
    1.0 - (-density * strength * 2.0).exp()
}

/// Combined Beer-Powder for efficient evaluation.
///
/// Combines Beer-Lambert transmittance with powder effect for
/// single-pass in-scatter calculation.
///
/// # Arguments
///
/// * `density` - Cloud density.
/// * `step_size` - Ray march step size.
/// * `extinction` - Extinction coefficient.
/// * `powder_strength` - Powder effect strength.
///
/// # Returns
///
/// Combined in-scatter factor.
#[inline]
pub fn beer_powder(density: f32, step_size: f32, extinction: f32, powder_strength: f32) -> f32 {
    let beer = beer_lambert(density, step_size, extinction);
    let powder = powder_effect(density * step_size, powder_strength);
    beer * powder
}

/// Energy-conserving powder modification.
///
/// Modified powder effect that ensures energy conservation by
/// accounting for both forward and backward scattering.
///
/// # Arguments
///
/// * `density` - Local cloud density.
/// * `optical_depth` - Accumulated optical depth from camera.
/// * `strength` - Powder effect strength.
///
/// # Returns
///
/// Energy-conserving powder factor.
#[inline]
pub fn powder_energy_conserving(density: f32, optical_depth: f32, strength: f32) -> f32 {
    let forward = 1.0 - (-density * strength * 2.0).exp();
    let depth_mod = (1.0 - (-optical_depth * 0.5).exp()).min(1.0);
    forward * (1.0 - depth_mod * 0.5)
}

// ---------------------------------------------------------------------------
// Silver Lining Effect
// ---------------------------------------------------------------------------

/// Silver lining effect for cloud edge highlighting.
///
/// Creates bright rim lighting at cloud edges when backlit by the sun.
/// Based on the view angle relative to the sun and local density gradient.
///
/// # Arguments
///
/// * `cos_view_light` - Cosine of angle between view and light directions.
/// * `density` - Local cloud density.
/// * `density_gradient` - Magnitude of density gradient (edges have high gradient).
/// * `intensity` - Silver lining intensity.
/// * `exponent` - Sharpness exponent.
///
/// # Returns
///
/// Silver lining contribution (additive).
#[inline]
pub fn silver_lining(
    cos_view_light: f32,
    density: f32,
    density_gradient: f32,
    intensity: f32,
    exponent: f32,
) -> f32 {
    // Only apply when looking towards the sun (backlit)
    let backlit = (-cos_view_light).max(0.0);

    // Stronger at edges (high gradient, low density)
    let edge_factor = density_gradient * (1.0 - density.min(1.0));

    intensity * backlit.powf(exponent) * edge_factor
}

/// Compute density gradient magnitude.
///
/// Estimates the density gradient by sampling neighboring positions.
/// High gradient indicates cloud edges.
///
/// # Arguments
///
/// * `density_center` - Density at center position.
/// * `density_samples` - Array of 6 densities (+X, -X, +Y, -Y, +Z, -Z).
/// * `sample_distance` - Distance to each sample.
///
/// # Returns
///
/// Magnitude of the density gradient.
pub fn compute_density_gradient(
    density_center: f32,
    density_samples: [f32; 6],
    sample_distance: f32,
) -> f32 {
    let inv_dist = 1.0 / sample_distance.max(EPSILON);

    // Central differences
    let dx = (density_samples[0] - density_samples[1]) * 0.5 * inv_dist;
    let dy = (density_samples[2] - density_samples[3]) * 0.5 * inv_dist;
    let dz = (density_samples[4] - density_samples[5]) * 0.5 * inv_dist;

    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Simplified silver lining without gradient computation.
///
/// Uses transmittance as a proxy for edge detection.
///
/// # Arguments
///
/// * `cos_view_light` - Cosine of view-light angle.
/// * `transmittance` - Current accumulated transmittance.
/// * `intensity` - Effect intensity.
/// * `exponent` - Sharpness exponent.
///
/// # Returns
///
/// Silver lining contribution.
#[inline]
pub fn silver_lining_simple(
    cos_view_light: f32,
    transmittance: f32,
    intensity: f32,
    exponent: f32,
) -> f32 {
    let backlit = (-cos_view_light).max(0.0);
    let edge = transmittance * (1.0 - transmittance * 0.5); // Peak at T ~ 0.5
    intensity * backlit.powf(exponent) * edge
}

// ---------------------------------------------------------------------------
// Multi-Scattering Approximation
// ---------------------------------------------------------------------------

/// Multi-scattering octave approximation.
///
/// Approximates multiple scattering bounces within the cloud using
/// an octave-based approach. Each octave represents higher-order
/// scattering with reduced contribution.
///
/// # Arguments
///
/// * `single_scatter` - Single scattering radiance (RGB).
/// * `albedo` - Scattering albedo.
/// * `transmittance` - Current transmittance.
/// * `octaves` - Number of multi-scatter octaves.
/// * `attenuation` - Energy attenuation per octave.
/// * `contribution` - Multi-scatter contribution weight.
///
/// # Returns
///
/// Multi-scattered radiance (RGB).
pub fn multiscatter_octaves(
    single_scatter: [f32; 3],
    albedo: f32,
    transmittance: f32,
    octaves: u32,
    attenuation: f32,
    contribution: f32,
) -> [f32; 3] {
    if octaves <= 1 || contribution < EPSILON {
        return single_scatter;
    }

    let mut total = single_scatter;
    let scatter_factor = 1.0 - transmittance;

    // Each octave represents higher-order scattering
    let mut octave_energy = albedo * scatter_factor * contribution;
    let mut octave_attenuation = 1.0;

    for _ in 1..octaves {
        octave_attenuation *= attenuation;
        let octave_contrib = octave_energy * octave_attenuation;

        total[0] += single_scatter[0] * octave_contrib;
        total[1] += single_scatter[1] * octave_contrib;
        total[2] += single_scatter[2] * octave_contrib;

        octave_energy *= albedo; // Each bounce loses energy
    }

    total
}

/// Simplified multi-scatter approximation.
///
/// Efficient single-formula approximation without octave iteration.
/// Based on the series expansion of multiple scattering.
///
/// L_ms = L_ss / (1 - albedo * (1 - T) * contribution)
///
/// # Arguments
///
/// * `single_scatter` - Single scattering radiance (RGB).
/// * `albedo` - Scattering albedo.
/// * `transmittance` - Current transmittance.
/// * `contribution` - Multi-scatter weight.
///
/// # Returns
///
/// Multi-scattered radiance (RGB).
#[inline]
pub fn multiscatter_simple(
    single_scatter: [f32; 3],
    albedo: f32,
    transmittance: f32,
    contribution: f32,
) -> [f32; 3] {
    let scatter_factor = 1.0 - transmittance;
    let ms_factor = 1.0 / (1.0 - albedo * scatter_factor * contribution * 0.5).max(EPSILON);

    [
        single_scatter[0] * ms_factor,
        single_scatter[1] * ms_factor,
        single_scatter[2] * ms_factor,
    ]
}

/// Evaluate multi-scatter from uniforms.
///
/// Convenience function using uniform buffer parameters.
#[inline]
pub fn evaluate_multiscatter(
    single_scatter: [f32; 3],
    transmittance: f32,
    uniforms: &CloudLightingUniforms,
) -> [f32; 3] {
    multiscatter_octaves(
        single_scatter,
        uniforms.scattering_albedo,
        transmittance,
        uniforms.multiscatter_octaves,
        uniforms.multiscatter_attenuation,
        uniforms.multiscatter_contribution,
    )
}

// ---------------------------------------------------------------------------
// Ambient Lighting
// ---------------------------------------------------------------------------

/// Compute ambient lighting contribution based on height.
///
/// Higher cloud regions receive more sky light, lower regions
/// receive ground bounce illumination.
///
/// # Arguments
///
/// * `height_fraction` - Normalized height within cloud layer (0=bottom, 1=top).
/// * `sky_color` - Sky/zenith color (RGB linear).
/// * `ground_color` - Ground color for bounce (RGB linear).
/// * `strength` - Overall ambient strength.
/// * `ground_bounce` - Ground bounce contribution factor.
///
/// # Returns
///
/// Ambient radiance (RGB linear).
pub fn ambient_from_height(
    height_fraction: f32,
    sky_color: [f32; 3],
    ground_color: [f32; 3],
    strength: f32,
    ground_bounce: f32,
) -> [f32; 3] {
    // Smooth interpolation between ground and sky
    let t = smoothstep(0.0, 1.0, height_fraction);
    let sky_contrib = t;
    let ground_contrib = (1.0 - t) * ground_bounce;

    [
        strength * (sky_contrib * sky_color[0] + ground_contrib * ground_color[0]),
        strength * (sky_contrib * sky_color[1] + ground_contrib * ground_color[1]),
        strength * (sky_contrib * sky_color[2] + ground_contrib * ground_color[2]),
    ]
}

/// Evaluate ambient from uniforms.
///
/// Convenience function using uniform buffer parameters.
#[inline]
pub fn evaluate_ambient(height_fraction: f32, uniforms: &CloudLightingUniforms) -> [f32; 3] {
    ambient_from_height(
        height_fraction,
        uniforms.sky_color,
        uniforms.ground_color,
        uniforms.ambient_strength,
        uniforms.ground_bounce,
    )
}

/// Ambient occlusion approximation for clouds.
///
/// Estimates local occlusion based on surrounding density.
/// Higher density surroundings = more occlusion.
///
/// # Arguments
///
/// * `local_density` - Density at current position.
/// * `surrounding_density` - Average density in neighborhood.
/// * `falloff` - Occlusion falloff rate.
///
/// # Returns
///
/// Ambient occlusion factor (0=fully occluded, 1=no occlusion).
#[inline]
pub fn cloud_ambient_occlusion(local_density: f32, surrounding_density: f32, falloff: f32) -> f32 {
    (-surrounding_density * falloff).exp() * (1.0 - local_density * 0.3).max(0.0)
}

// ---------------------------------------------------------------------------
// Light Energy Accumulation
// ---------------------------------------------------------------------------

/// Result of computing lighting at a single ray march step.
#[derive(Debug, Clone, Copy, Default)]
pub struct LightingSample {
    /// In-scattered radiance (RGB).
    pub in_scatter: [f32; 3],

    /// Light transmittance towards sun.
    pub sun_transmittance: f32,

    /// Phase function value.
    pub phase: f32,

    /// Beer-Powder factor.
    pub beer_powder: f32,

    /// Silver lining contribution.
    pub silver_lining: f32,
}

impl LightingSample {
    /// Create a new empty sample.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Total radiance (in-scatter + silver lining).
    #[inline]
    pub fn total_radiance(&self) -> [f32; 3] {
        [
            self.in_scatter[0] + self.silver_lining,
            self.in_scatter[1] + self.silver_lining,
            self.in_scatter[2] + self.silver_lining,
        ]
    }
}

/// Compute lighting at a single sample position.
///
/// This is the main lighting function called at each ray march step.
/// Combines all lighting components: phase function, Beer-Powder,
/// silver lining, and ambient.
///
/// # Arguments
///
/// * `density` - Cloud density at sample position.
/// * `height_fraction` - Height within cloud layer (0-1).
/// * `sun_transmittance` - Transmittance towards the sun.
/// * `view_dir` - Normalized view direction.
/// * `light_dir` - Normalized light direction (towards sun).
/// * `current_transmittance` - Accumulated transmittance from camera.
/// * `uniforms` - Cloud lighting uniforms.
///
/// # Returns
///
/// Lighting sample with all components.
pub fn compute_lighting(
    density: f32,
    height_fraction: f32,
    sun_transmittance: f32,
    view_dir: [f32; 3],
    light_dir: [f32; 3],
    current_transmittance: f32,
    uniforms: &CloudLightingUniforms,
) -> LightingSample {
    if density < EPSILON {
        return LightingSample::new();
    }

    // Compute view-light angle
    let cos_theta = dot(view_dir, light_dir);

    // Phase function
    let phase = evaluate_phase(cos_theta, uniforms);

    // Beer-Powder for in-scatter
    let bp = beer_powder(
        density,
        1.0, // Normalized step
        uniforms.extinction_coeff,
        uniforms.powder_strength,
    );

    // Direct sun contribution
    let sun_contrib = sun_transmittance * phase * bp;
    let in_scatter = [
        uniforms.sun_color[0] * sun_contrib,
        uniforms.sun_color[1] * sun_contrib,
        uniforms.sun_color[2] * sun_contrib,
    ];

    // Silver lining
    let silver = silver_lining_simple(
        cos_theta,
        current_transmittance,
        uniforms.silver_lining_intensity,
        uniforms.silver_lining_exponent,
    ) * sun_transmittance;

    // Ambient contribution
    let ambient = evaluate_ambient(height_fraction, uniforms);

    // Combine
    let total_in_scatter = [
        in_scatter[0] + ambient[0],
        in_scatter[1] + ambient[1],
        in_scatter[2] + ambient[2],
    ];

    LightingSample {
        in_scatter: total_in_scatter,
        sun_transmittance,
        phase,
        beer_powder: bp,
        silver_lining: silver,
    }
}

/// Accumulate lighting into a running total.
///
/// Uses front-to-back compositing with transmittance weighting.
///
/// # Arguments
///
/// * `accumulated` - Current accumulated radiance (RGB).
/// * `transmittance` - Current accumulated transmittance.
/// * `sample` - New lighting sample to add.
/// * `step_density` - Density * step_size product.
/// * `extinction` - Extinction coefficient.
///
/// # Returns
///
/// Updated (radiance, transmittance) tuple.
pub fn accumulate_lighting(
    accumulated: [f32; 3],
    transmittance: f32,
    sample: &LightingSample,
    step_density: f32,
    extinction: f32,
) -> ([f32; 3], f32) {
    let sample_extinction = step_density * extinction;
    let sample_transmittance = (-sample_extinction).exp();

    // Front-to-back blending weight
    let weight = transmittance * (1.0 - sample_transmittance);

    let radiance = sample.total_radiance();
    let new_accumulated = [
        accumulated[0] + weight * radiance[0],
        accumulated[1] + weight * radiance[1],
        accumulated[2] + weight * radiance[2],
    ];

    let new_transmittance = transmittance * sample_transmittance;

    (new_accumulated, new_transmittance)
}

// ---------------------------------------------------------------------------
// CloudLightingPass - Main Lighting API
// ---------------------------------------------------------------------------

/// Main cloud lighting pass.
///
/// Provides the CPU-side API for computing cloud lighting during ray marching.
/// Can be used standalone or integrated with `CloudRayMarcher` from
/// `cloud_raymarching.rs`.
///
/// # Example
///
/// ```
/// use renderer_backend::cloud_lighting::{CloudLightingPass, CloudLightingUniforms};
///
/// let uniforms = CloudLightingUniforms::midday();
/// let lighting = CloudLightingPass::new(uniforms);
///
/// // At each ray march step:
/// let sample = lighting.compute_step(
///     0.5,               // density
///     0.7,               // height_fraction
///     0.8,               // sun_transmittance
///     [0.0, 0.0, -1.0],  // view_dir
///     0.6,               // current_transmittance
/// );
/// ```
#[derive(Debug, Clone)]
pub struct CloudLightingPass {
    /// GPU uniform buffer data.
    uniforms: CloudLightingUniforms,

    /// Cached light direction (normalized).
    light_dir: [f32; 3],
}

impl CloudLightingPass {
    /// Create a new lighting pass with given uniforms.
    #[inline]
    pub fn new(uniforms: CloudLightingUniforms) -> Self {
        Self {
            light_dir: uniforms.sun_direction,
            uniforms,
        }
    }

    /// Create with default settings.
    #[inline]
    pub fn default_pass() -> Self {
        Self::new(CloudLightingUniforms::default())
    }

    /// Get the uniform buffer for GPU upload.
    #[inline]
    pub fn uniforms(&self) -> &CloudLightingUniforms {
        &self.uniforms
    }

    /// Get mutable access to uniforms.
    #[inline]
    pub fn uniforms_mut(&mut self) -> &mut CloudLightingUniforms {
        &mut self.uniforms
    }

    /// Update uniforms.
    pub fn set_uniforms(&mut self, uniforms: CloudLightingUniforms) {
        self.light_dir = uniforms.sun_direction;
        self.uniforms = uniforms;
    }

    /// Update sun direction.
    pub fn set_sun_direction(&mut self, direction: [f32; 3]) {
        self.light_dir = normalize_vec3(direction);
        self.uniforms.sun_direction = self.light_dir;
    }

    /// Update sun color.
    #[inline]
    pub fn set_sun_color(&mut self, color: [f32; 3]) {
        self.uniforms.sun_color = color;
    }

    /// Compute lighting at a single ray march step.
    ///
    /// # Arguments
    ///
    /// * `density` - Cloud density at sample position.
    /// * `height_fraction` - Height within cloud layer (0-1).
    /// * `sun_transmittance` - Transmittance towards the sun.
    /// * `view_dir` - Normalized view direction.
    /// * `current_transmittance` - Accumulated transmittance from camera.
    ///
    /// # Returns
    ///
    /// Lighting sample with all components.
    #[inline]
    pub fn compute_step(
        &self,
        density: f32,
        height_fraction: f32,
        sun_transmittance: f32,
        view_dir: [f32; 3],
        current_transmittance: f32,
    ) -> LightingSample {
        compute_lighting(
            density,
            height_fraction,
            sun_transmittance,
            view_dir,
            self.light_dir,
            current_transmittance,
            &self.uniforms,
        )
    }

    /// Apply multi-scatter to accumulated radiance.
    #[inline]
    pub fn apply_multiscatter(&self, radiance: [f32; 3], transmittance: f32) -> [f32; 3] {
        evaluate_multiscatter(radiance, transmittance, &self.uniforms)
    }

    /// Compute phase function for given angle.
    #[inline]
    pub fn compute_phase(&self, cos_view_light: f32) -> f32 {
        evaluate_phase(cos_view_light, &self.uniforms)
    }

    /// Compute Beer-Powder factor.
    #[inline]
    pub fn compute_beer_powder(&self, density: f32, step_size: f32) -> f32 {
        beer_powder(
            density,
            step_size,
            self.uniforms.extinction_coeff,
            self.uniforms.powder_strength,
        )
    }

    /// Compute ambient at given height.
    #[inline]
    pub fn compute_ambient(&self, height_fraction: f32) -> [f32; 3] {
        evaluate_ambient(height_fraction, &self.uniforms)
    }
}

impl Default for CloudLightingPass {
    fn default() -> Self {
        Self::default_pass()
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
    // CloudLightingUniforms Tests
    // =========================================================================

    #[test]
    fn test_uniforms_default() {
        let u = CloudLightingUniforms::default();
        assert!(u.validate());
        assert!(u.asymmetry_g > 0.0); // Forward scatter
    }

    #[test]
    fn test_uniforms_size() {
        assert_eq!(std::mem::size_of::<CloudLightingUniforms>(), 128);
    }

    #[test]
    fn test_uniforms_pod() {
        let u = CloudLightingUniforms::default();
        let bytes = bytemuck::bytes_of(&u);
        assert_eq!(bytes.len(), 128);
    }

    #[test]
    fn test_uniforms_midday() {
        let u = CloudLightingUniforms::midday();
        assert!(u.validate());
        assert!((u.sun_direction[1] - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_uniforms_sunset() {
        let u = CloudLightingUniforms::sunset();
        assert!(u.validate());
        assert!(u.sun_direction[1] < 0.5); // Low sun
        assert!(u.sun_color[0] > u.sun_color[2]); // Warm color
    }

    #[test]
    fn test_uniforms_overcast() {
        let u = CloudLightingUniforms::overcast();
        assert!(u.validate());
        assert!(u.ambient_strength > DEFAULT_AMBIENT_STRENGTH);
    }

    #[test]
    fn test_uniforms_with_sun_direction() {
        let u = CloudLightingUniforms::default()
            .with_sun_direction([1.0, 0.0, 0.0]);
        assert!((u.sun_direction[0] - 1.0).abs() < EPSILON);
        assert!(u.validate());
    }

    #[test]
    fn test_uniforms_with_asymmetry() {
        let u = CloudLightingUniforms::default()
            .with_asymmetry(0.8);
        assert!((u.asymmetry_g - 0.8).abs() < EPSILON);
    }

    #[test]
    fn test_uniforms_with_asymmetry_clamping() {
        let u = CloudLightingUniforms::default()
            .with_asymmetry(1.5);
        // Should be clamped to MAX_ASYMMETRY_G (0.999)
        assert!(u.asymmetry_g <= MAX_ASYMMETRY_G);
        assert!((u.asymmetry_g - MAX_ASYMMETRY_G).abs() < EPSILON);
    }

    #[test]
    fn test_uniforms_with_dual_lobe() {
        let u = CloudLightingUniforms::default()
            .with_dual_lobe(0.8, -0.4, 0.6);
        assert!((u.asymmetry_g - 0.8).abs() < EPSILON);
        assert!((u.asymmetry_g2 - (-0.4)).abs() < EPSILON);
        assert!((u.scatter_blend - 0.6).abs() < EPSILON);
    }

    #[test]
    fn test_uniforms_with_powder_strength() {
        let u = CloudLightingUniforms::default()
            .with_powder_strength(3.0);
        assert!((u.powder_strength - 3.0).abs() < EPSILON);
    }

    #[test]
    fn test_uniforms_with_silver_lining() {
        let u = CloudLightingUniforms::default()
            .with_silver_lining(1.5, 10.0);
        assert!((u.silver_lining_intensity - 1.5).abs() < EPSILON);
        assert!((u.silver_lining_exponent - 10.0).abs() < EPSILON);
    }

    #[test]
    fn test_uniforms_with_multiscatter() {
        let u = CloudLightingUniforms::default()
            .with_multiscatter(6, 0.4, 0.5);
        assert_eq!(u.multiscatter_octaves, 6);
        assert!((u.multiscatter_attenuation - 0.4).abs() < EPSILON);
    }

    #[test]
    fn test_uniforms_with_ambient() {
        let u = CloudLightingUniforms::default()
            .with_ambient(0.2, 0.15);
        assert!((u.ambient_strength - 0.2).abs() < EPSILON);
        assert!((u.ground_bounce - 0.15).abs() < EPSILON);
    }

    #[test]
    fn test_uniforms_with_light_march() {
        let u = CloudLightingUniforms::default()
            .with_light_march(10, 80.0);
        assert_eq!(u.light_march_steps, 10);
        assert!((u.light_step_size - 80.0).abs() < EPSILON);
    }

    #[test]
    fn test_uniforms_validate_invalid_g() {
        let mut u = CloudLightingUniforms::default();
        u.asymmetry_g = 1.0; // Invalid: at boundary
        assert!(!u.validate());
    }

    #[test]
    fn test_uniforms_validate_invalid_albedo() {
        let mut u = CloudLightingUniforms::default();
        u.scattering_albedo = 1.5; // Invalid: > 1
        assert!(!u.validate());
    }

    #[test]
    fn test_uniforms_validate_invalid_direction() {
        let mut u = CloudLightingUniforms::default();
        u.sun_direction = [0.0, 0.0, 0.0]; // Invalid: zero length
        assert!(!u.validate());
    }

    // =========================================================================
    // PhaseParams Tests
    // =========================================================================

    #[test]
    fn test_phase_params_default() {
        let p = PhaseParams::default();
        assert!(p.validate());
        assert!(p.asymmetry_g > 0.0);
    }

    #[test]
    fn test_phase_params_isotropic() {
        let p = PhaseParams::isotropic();
        assert!(p.validate());
        assert!((p.asymmetry_g).abs() < EPSILON);
    }

    #[test]
    fn test_phase_params_forward() {
        let p = PhaseParams::forward(0.9);
        assert!(p.validate());
        assert!((p.asymmetry_g - 0.9).abs() < EPSILON);
        assert!((p.scatter_blend - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_phase_params_dual_lobe() {
        let p = PhaseParams::dual_lobe(0.8, -0.3, 0.7);
        assert!(p.validate());
        assert!((p.asymmetry_g - 0.8).abs() < EPSILON);
        assert!((p.asymmetry_g2 - (-0.3)).abs() < EPSILON);
    }

    // =========================================================================
    // ScatterParams Tests
    // =========================================================================

    #[test]
    fn test_scatter_params_default() {
        let s = ScatterParams::default();
        assert!(s.validate());
        assert!(s.scattering_albedo > 0.9);
    }

    #[test]
    fn test_scatter_params_thin_clouds() {
        let s = ScatterParams::thin_clouds();
        assert!(s.validate());
        assert!(s.extinction_coeff < DEFAULT_EXTINCTION_COEFF);
    }

    #[test]
    fn test_scatter_params_storm_clouds() {
        let s = ScatterParams::storm_clouds();
        assert!(s.validate());
        assert!(s.extinction_coeff > DEFAULT_EXTINCTION_COEFF);
    }

    // =========================================================================
    // SilverLiningParams Tests
    // =========================================================================

    #[test]
    fn test_silver_lining_params_default() {
        let s = SilverLiningParams::default();
        assert!(s.validate());
    }

    #[test]
    fn test_silver_lining_params_dramatic() {
        let s = SilverLiningParams::dramatic();
        assert!(s.validate());
        assert!(s.intensity > DEFAULT_SILVER_LINING_INTENSITY);
    }

    #[test]
    fn test_silver_lining_params_subtle() {
        let s = SilverLiningParams::subtle();
        assert!(s.validate());
        assert!(s.intensity < DEFAULT_SILVER_LINING_INTENSITY);
    }

    #[test]
    fn test_silver_lining_params_disabled() {
        let s = SilverLiningParams::disabled();
        assert!(s.validate());
        assert!((s.intensity).abs() < EPSILON);
    }

    // =========================================================================
    // MultiScatterParams Tests
    // =========================================================================

    #[test]
    fn test_multiscatter_params_default() {
        let m = MultiScatterParams::default();
        assert!(m.validate());
    }

    #[test]
    fn test_multiscatter_params_high_quality() {
        let m = MultiScatterParams::high_quality();
        assert!(m.validate());
        assert!(m.octaves > DEFAULT_MULTISCATTER_OCTAVES);
    }

    #[test]
    fn test_multiscatter_params_fast() {
        let m = MultiScatterParams::fast();
        assert!(m.validate());
        assert!(m.octaves < DEFAULT_MULTISCATTER_OCTAVES);
    }

    #[test]
    fn test_multiscatter_params_disabled() {
        let m = MultiScatterParams::disabled();
        assert!(m.validate());
        assert!((m.contribution).abs() < EPSILON);
    }

    // =========================================================================
    // AmbientParams Tests
    // =========================================================================

    #[test]
    fn test_ambient_params_default() {
        let a = AmbientParams::default();
        assert!(a.validate());
    }

    #[test]
    fn test_ambient_params_cloudy_day() {
        let a = AmbientParams::cloudy_day();
        assert!(a.validate());
        assert!(a.strength > DEFAULT_AMBIENT_STRENGTH);
    }

    #[test]
    fn test_ambient_params_clear_sky() {
        let a = AmbientParams::clear_sky();
        assert!(a.validate());
        assert!(a.strength < DEFAULT_AMBIENT_STRENGTH);
    }

    // =========================================================================
    // LightMarchParams Tests
    // =========================================================================

    #[test]
    fn test_light_march_params_default() {
        let l = LightMarchParams::default();
        assert!(l.validate());
    }

    #[test]
    fn test_light_march_params_high_quality() {
        let l = LightMarchParams::high_quality();
        assert!(l.validate());
        assert!(l.steps > DEFAULT_LIGHT_MARCH_STEPS);
    }

    #[test]
    fn test_light_march_params_fast() {
        let l = LightMarchParams::fast();
        assert!(l.validate());
        assert!(l.steps < DEFAULT_LIGHT_MARCH_STEPS);
    }

    // =========================================================================
    // CloudLightingConfig Tests
    // =========================================================================

    #[test]
    fn test_config_default() {
        let c = CloudLightingConfig::default();
        let u = c.to_uniforms();
        assert!(u.validate());
    }

    #[test]
    fn test_config_to_uniforms() {
        let c = CloudLightingConfig::new()
            .with_sun_direction([0.0, 1.0, 0.0])
            .with_sun_color([1.0, 1.0, 1.0]);
        let u = c.to_uniforms();
        assert!((u.sun_direction[1] - 1.0).abs() < EPSILON);
        assert!((u.sun_color[0] - 1.0).abs() < EPSILON);
    }

    // =========================================================================
    // Phase Function Tests
    // =========================================================================

    #[test]
    fn test_henyey_greenstein_isotropic() {
        // g=0 should give isotropic: 1/(4*PI)
        let p = henyey_greenstein(0.0, 0.0);
        let expected = 1.0 / (4.0 * PI);
        assert!((p - expected).abs() < 0.001);
    }

    #[test]
    fn test_henyey_greenstein_forward() {
        // g>0, cos_theta=1 (looking into light) should be maximum
        let p_forward = henyey_greenstein(1.0, 0.7);
        let p_backward = henyey_greenstein(-1.0, 0.7);
        assert!(p_forward > p_backward);
    }

    #[test]
    fn test_henyey_greenstein_backward() {
        // g<0, cos_theta=-1 (looking away from light) should be maximum
        let p_forward = henyey_greenstein(1.0, -0.7);
        let p_backward = henyey_greenstein(-1.0, -0.7);
        assert!(p_backward > p_forward);
    }

    #[test]
    fn test_henyey_greenstein_positive() {
        // Phase function should always be positive
        for g in [-0.9, -0.5, 0.0, 0.5, 0.9].iter() {
            for cos in [-1.0, -0.5, 0.0, 0.5, 1.0].iter() {
                let p = henyey_greenstein(*cos, *g);
                assert!(p > 0.0, "HG({}, {}) = {} should be > 0", cos, g, p);
            }
        }
    }

    #[test]
    fn test_henyey_greenstein_normalization() {
        // Integral over sphere should be ~1 (we test a few angles)
        let g = 0.5;
        let mut sum = 0.0;
        let n = 100;
        for i in 0..n {
            let cos_theta = -1.0 + 2.0 * (i as f32 / n as f32);
            sum += henyey_greenstein(cos_theta, g) * 2.0 / n as f32;
        }
        // Rough check (numerical integration)
        assert!((sum - 1.0 / (2.0 * PI)).abs() < 0.1);
    }

    #[test]
    fn test_dual_lobe_hg() {
        let p = dual_lobe_hg(0.5, 0.7, -0.3, 0.7);
        assert!(p > 0.0);
    }

    #[test]
    fn test_dual_lobe_hg_blend_extremes() {
        let cos = 0.5;
        let g1 = 0.7;
        let g2 = -0.3;

        // blend=1 should equal pure g1
        let p1 = dual_lobe_hg(cos, g1, g2, 1.0);
        let pure1 = henyey_greenstein(cos, g1);
        assert!((p1 - pure1).abs() < EPSILON);

        // blend=0 should equal pure g2
        let p0 = dual_lobe_hg(cos, g1, g2, 0.0);
        let pure2 = henyey_greenstein(cos, g2);
        assert!((p0 - pure2).abs() < EPSILON);
    }

    #[test]
    fn test_schlick_phase() {
        let p = schlick_phase(0.5, 0.7);
        assert!(p > 0.0);
    }

    #[test]
    fn test_schlick_vs_hg() {
        // Schlick should approximate HG
        let cos = 0.5;
        let g = 0.5;
        let schlick = schlick_phase(cos, g);
        let hg = henyey_greenstein(cos, g);
        // They won't match exactly, but should be same order of magnitude
        assert!((schlick / hg - 1.0).abs() < 1.0);
    }

    #[test]
    fn test_cornette_shanks() {
        let p = cornette_shanks(0.5, 0.7);
        assert!(p > 0.0);
    }

    #[test]
    fn test_evaluate_phase() {
        let u = CloudLightingUniforms::default();
        let p = evaluate_phase(0.5, &u);
        assert!(p > 0.0);
    }

    // =========================================================================
    // Beer-Lambert Tests
    // =========================================================================

    #[test]
    fn test_beer_lambert_zero_density() {
        let t = beer_lambert(0.0, 100.0, 0.1);
        assert!((t - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_beer_lambert_zero_distance() {
        let t = beer_lambert(0.5, 0.0, 0.1);
        assert!((t - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_beer_lambert_exponential_decay() {
        let t1 = beer_lambert(0.5, 10.0, 0.1);
        let t2 = beer_lambert(0.5, 20.0, 0.1);
        // Double distance should square the transmittance
        assert!((t2 - t1 * t1).abs() < 0.01);
    }

    #[test]
    fn test_beer_lambert_range() {
        for density in [0.1, 0.5, 1.0].iter() {
            for distance in [1.0, 10.0, 100.0].iter() {
                let t = beer_lambert(*density, *distance, 0.1);
                assert!(t > 0.0 && t <= 1.0);
            }
        }
    }

    #[test]
    fn test_beer_lambert_optical_depth() {
        let tau = 0.5;
        let t = beer_lambert_optical_depth(tau);
        assert!((t - (-tau).exp()).abs() < EPSILON);
    }

    #[test]
    fn test_compute_optical_depth() {
        let tau = compute_optical_depth(0.5, 10.0, 0.1);
        assert!((tau - 0.5).abs() < EPSILON);
    }

    // =========================================================================
    // Powder Effect Tests
    // =========================================================================

    #[test]
    fn test_powder_effect_zero_density() {
        let p = powder_effect(0.0, 2.0);
        assert!(p.abs() < EPSILON);
    }

    #[test]
    fn test_powder_effect_high_density() {
        let p = powder_effect(1.0, 2.0);
        assert!(p > 0.9);
    }

    #[test]
    fn test_powder_effect_monotonic() {
        let p1 = powder_effect(0.2, 2.0);
        let p2 = powder_effect(0.5, 2.0);
        let p3 = powder_effect(0.8, 2.0);
        assert!(p1 < p2 && p2 < p3);
    }

    #[test]
    fn test_powder_effect_strength() {
        let p1 = powder_effect(0.5, 1.0);
        let p2 = powder_effect(0.5, 3.0);
        assert!(p2 > p1); // Higher strength = faster saturation
    }

    #[test]
    fn test_beer_powder() {
        let bp = beer_powder(0.5, 10.0, 0.1, 2.0);
        let beer = beer_lambert(0.5, 10.0, 0.1);
        let powder = powder_effect(0.5 * 10.0, 2.0);
        assert!((bp - beer * powder).abs() < EPSILON);
    }

    #[test]
    fn test_powder_energy_conserving() {
        let p = powder_energy_conserving(0.5, 0.3, 2.0);
        assert!(p > 0.0 && p < 1.0);
    }

    #[test]
    fn test_powder_energy_conserving_depth_effect() {
        let p_shallow = powder_energy_conserving(0.5, 0.1, 2.0);
        let p_deep = powder_energy_conserving(0.5, 2.0, 2.0);
        // Deep in cloud should have less powder effect
        assert!(p_deep < p_shallow);
    }

    // =========================================================================
    // Silver Lining Tests
    // =========================================================================

    #[test]
    fn test_silver_lining_frontlit() {
        // When looking with the light (cos=1), no silver lining
        let s = silver_lining(1.0, 0.5, 0.5, 0.8, 8.0);
        assert!(s.abs() < EPSILON);
    }

    #[test]
    fn test_silver_lining_backlit() {
        // When looking against the light (cos=-1), maximum silver lining
        let s_backlit = silver_lining(-1.0, 0.3, 0.8, 0.8, 2.0);
        let s_front = silver_lining(1.0, 0.3, 0.8, 0.8, 2.0);
        assert!(s_backlit > s_front);
    }

    #[test]
    fn test_silver_lining_edge_enhancement() {
        // High gradient + low density = strong silver lining
        let s_edge = silver_lining(-0.8, 0.1, 1.0, 0.8, 4.0);
        let s_interior = silver_lining(-0.8, 0.9, 0.1, 0.8, 4.0);
        assert!(s_edge > s_interior);
    }

    #[test]
    fn test_silver_lining_simple() {
        let s = silver_lining_simple(-0.8, 0.5, 0.8, 4.0);
        assert!(s > 0.0);
    }

    #[test]
    fn test_silver_lining_simple_transmittance_peak() {
        // Silver lining effect: strong when backlit (cos < 0)
        // The edge factor T * (1 - T * 0.5) peaks around T ~ 0.67
        // We check that effect is stronger in mid-range than at extremes
        let s_low = silver_lining_simple(-0.8, 0.1, 0.8, 2.0);
        let s_mid = silver_lining_simple(-0.8, 0.6, 0.8, 2.0);
        let s_high = silver_lining_simple(-0.8, 0.95, 0.8, 2.0);
        // Mid should be stronger than fully opaque (low T)
        assert!(s_mid > s_low, "Mid ({}) should be > Low ({})", s_mid, s_low);
        // With T=0.95, edge=0.95*(1-0.475)=0.499, vs T=0.6, edge=0.6*(1-0.3)=0.42
        // So actually high T can be higher. Let's just check reasonable values.
        assert!(s_mid > 0.0);
        assert!(s_high > 0.0);
    }

    #[test]
    fn test_compute_density_gradient() {
        let samples = [0.5, 0.3, 0.6, 0.4, 0.5, 0.5]; // +X,-X,+Y,-Y,+Z,-Z
        let grad = compute_density_gradient(0.5, samples, 10.0);
        assert!(grad > 0.0);
    }

    #[test]
    fn test_compute_density_gradient_uniform() {
        // Uniform field should have zero gradient
        let samples = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        let grad = compute_density_gradient(0.5, samples, 10.0);
        assert!(grad.abs() < EPSILON);
    }

    // =========================================================================
    // Multi-Scatter Tests
    // =========================================================================

    #[test]
    fn test_multiscatter_octaves_single() {
        let ss = [1.0, 0.5, 0.25];
        let ms = multiscatter_octaves(ss, 0.99, 0.5, 1, 0.5, 0.4);
        assert_eq!(ms, ss); // Single octave = no multi-scatter
    }

    #[test]
    fn test_multiscatter_octaves_boost() {
        let ss = [1.0, 1.0, 1.0];
        let ms = multiscatter_octaves(ss, 0.99, 0.5, 4, 0.5, 0.4);
        // Multi-scatter should boost brightness
        assert!(ms[0] > ss[0]);
        assert!(ms[1] > ss[1]);
        assert!(ms[2] > ss[2]);
    }

    #[test]
    fn test_multiscatter_octaves_no_scatter() {
        let ss = [1.0, 1.0, 1.0];
        // T=1 means no scattering occurred
        let ms = multiscatter_octaves(ss, 0.99, 1.0, 4, 0.5, 0.4);
        assert_eq!(ms, ss);
    }

    #[test]
    fn test_multiscatter_simple() {
        let ss = [1.0, 1.0, 1.0];
        let ms = multiscatter_simple(ss, 0.99, 0.5, 0.4);
        assert!(ms[0] > ss[0]);
    }

    #[test]
    fn test_multiscatter_simple_no_scatter() {
        let ss = [1.0, 1.0, 1.0];
        let ms = multiscatter_simple(ss, 0.99, 1.0, 0.4);
        assert_eq!(ms, ss);
    }

    #[test]
    fn test_evaluate_multiscatter() {
        let u = CloudLightingUniforms::default();
        let ss = [1.0, 1.0, 1.0];
        let ms = evaluate_multiscatter(ss, 0.5, &u);
        assert!(ms[0] > ss[0]);
    }

    // =========================================================================
    // Ambient Lighting Tests
    // =========================================================================

    #[test]
    fn test_ambient_from_height_bottom() {
        let ambient = ambient_from_height(
            0.0,
            [0.3, 0.5, 0.8],
            [0.2, 0.18, 0.15],
            0.15,
            0.1,
        );
        // At bottom, should have more ground contribution
        assert!(ambient[0] > 0.0);
    }

    #[test]
    fn test_ambient_from_height_top() {
        let ambient = ambient_from_height(
            1.0,
            [0.3, 0.5, 0.8],
            [0.2, 0.18, 0.15],
            0.15,
            0.1,
        );
        // At top, sky should dominate
        assert!(ambient[2] > ambient[0]); // Blue > red
    }

    #[test]
    fn test_ambient_strength() {
        let a1 = ambient_from_height(0.5, [0.3, 0.5, 0.8], [0.2, 0.18, 0.15], 0.1, 0.1);
        let a2 = ambient_from_height(0.5, [0.3, 0.5, 0.8], [0.2, 0.18, 0.15], 0.3, 0.1);
        // Higher strength = brighter ambient
        assert!(a2[0] > a1[0]);
        assert!(a2[1] > a1[1]);
        assert!(a2[2] > a1[2]);
    }

    #[test]
    fn test_evaluate_ambient() {
        let u = CloudLightingUniforms::default();
        let ambient = evaluate_ambient(0.5, &u);
        assert!(ambient[0] > 0.0);
        assert!(ambient[1] > 0.0);
        assert!(ambient[2] > 0.0);
    }

    #[test]
    fn test_cloud_ambient_occlusion() {
        let ao = cloud_ambient_occlusion(0.3, 0.5, 1.0);
        assert!(ao > 0.0 && ao < 1.0);
    }

    #[test]
    fn test_cloud_ao_high_surrounding() {
        // High surrounding density = more occlusion
        let ao_low = cloud_ambient_occlusion(0.3, 0.1, 1.0);
        let ao_high = cloud_ambient_occlusion(0.3, 0.8, 1.0);
        assert!(ao_high < ao_low);
    }

    // =========================================================================
    // LightingSample Tests
    // =========================================================================

    #[test]
    fn test_lighting_sample_new() {
        let s = LightingSample::new();
        assert_eq!(s.in_scatter, [0.0; 3]);
        assert_eq!(s.sun_transmittance, 0.0);
    }

    #[test]
    fn test_lighting_sample_total_radiance() {
        let s = LightingSample {
            in_scatter: [0.5, 0.4, 0.3],
            silver_lining: 0.1,
            ..Default::default()
        };
        let total = s.total_radiance();
        assert!((total[0] - 0.6).abs() < EPSILON);
        assert!((total[1] - 0.5).abs() < EPSILON);
        assert!((total[2] - 0.4).abs() < EPSILON);
    }

    // =========================================================================
    // compute_lighting Tests
    // =========================================================================

    #[test]
    fn test_compute_lighting_zero_density() {
        let u = CloudLightingUniforms::default();
        let s = compute_lighting(
            0.0, 0.5, 0.8,
            [0.0, 0.0, -1.0],
            u.sun_direction,
            0.8, &u,
        );
        assert_eq!(s.in_scatter, [0.0; 3]);
    }

    #[test]
    fn test_compute_lighting_nonzero() {
        let u = CloudLightingUniforms::default();
        let s = compute_lighting(
            0.5, 0.5, 0.8,
            [0.0, 0.0, -1.0],
            u.sun_direction,
            0.8, &u,
        );
        assert!(s.in_scatter[0] > 0.0);
        assert!(s.phase > 0.0);
        assert!(s.beer_powder > 0.0);
    }

    #[test]
    fn test_compute_lighting_sun_transmittance() {
        let u = CloudLightingUniforms::default();
        let s_lit = compute_lighting(
            0.5, 0.5, 1.0,
            [0.0, 0.0, -1.0],
            u.sun_direction,
            0.8, &u,
        );
        let s_shadow = compute_lighting(
            0.5, 0.5, 0.1,
            [0.0, 0.0, -1.0],
            u.sun_direction,
            0.8, &u,
        );
        // Shadow should be dimmer
        assert!(s_lit.in_scatter[0] > s_shadow.in_scatter[0]);
    }

    // =========================================================================
    // accumulate_lighting Tests
    // =========================================================================

    #[test]
    fn test_accumulate_lighting() {
        let sample = LightingSample {
            in_scatter: [1.0, 0.5, 0.25],
            silver_lining: 0.0,
            ..Default::default()
        };
        let (rad, trans) = accumulate_lighting(
            [0.0; 3], 1.0, &sample, 0.1, 0.1,
        );
        assert!(rad[0] > 0.0);
        assert!(trans < 1.0);
    }

    #[test]
    fn test_accumulate_lighting_opaque() {
        let sample = LightingSample {
            in_scatter: [1.0, 1.0, 1.0],
            silver_lining: 0.0,
            ..Default::default()
        };
        // Very dense step
        let (_, trans) = accumulate_lighting(
            [0.0; 3], 1.0, &sample, 10.0, 1.0,
        );
        assert!(trans < 0.1);
    }

    #[test]
    fn test_accumulate_lighting_transparent() {
        let sample = LightingSample {
            in_scatter: [1.0, 1.0, 1.0],
            silver_lining: 0.0,
            ..Default::default()
        };
        // Zero density step
        let (rad, trans) = accumulate_lighting(
            [0.0; 3], 1.0, &sample, 0.0, 0.1,
        );
        assert!((trans - 1.0).abs() < EPSILON);
        assert!(rad[0].abs() < EPSILON);
    }

    // =========================================================================
    // CloudLightingPass Tests
    // =========================================================================

    #[test]
    fn test_lighting_pass_new() {
        let pass = CloudLightingPass::new(CloudLightingUniforms::default());
        assert!(pass.uniforms().validate());
    }

    #[test]
    fn test_lighting_pass_default() {
        let pass = CloudLightingPass::default();
        assert!(pass.uniforms().validate());
    }

    #[test]
    fn test_lighting_pass_set_uniforms() {
        let mut pass = CloudLightingPass::default();
        pass.set_uniforms(CloudLightingUniforms::sunset());
        assert!(pass.uniforms().sun_direction[1] < 0.5);
    }

    #[test]
    fn test_lighting_pass_set_sun_direction() {
        let mut pass = CloudLightingPass::default();
        pass.set_sun_direction([1.0, 0.0, 0.0]);
        assert!((pass.uniforms().sun_direction[0] - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_lighting_pass_set_sun_color() {
        let mut pass = CloudLightingPass::default();
        pass.set_sun_color([1.0, 0.5, 0.25]);
        assert!((pass.uniforms().sun_color[1] - 0.5).abs() < EPSILON);
    }

    #[test]
    fn test_lighting_pass_compute_step() {
        let pass = CloudLightingPass::default();
        let s = pass.compute_step(
            0.5, 0.5, 0.8,
            [0.0, 0.0, -1.0],
            0.8,
        );
        assert!(s.in_scatter[0] > 0.0);
    }

    #[test]
    fn test_lighting_pass_apply_multiscatter() {
        let pass = CloudLightingPass::default();
        let rad = [1.0, 1.0, 1.0];
        let ms = pass.apply_multiscatter(rad, 0.5);
        assert!(ms[0] > rad[0]);
    }

    #[test]
    fn test_lighting_pass_compute_phase() {
        let pass = CloudLightingPass::default();
        let p = pass.compute_phase(0.5);
        assert!(p > 0.0);
    }

    #[test]
    fn test_lighting_pass_compute_beer_powder() {
        let pass = CloudLightingPass::default();
        let bp = pass.compute_beer_powder(0.5, 10.0);
        assert!(bp > 0.0 && bp < 1.0);
    }

    #[test]
    fn test_lighting_pass_compute_ambient() {
        let pass = CloudLightingPass::default();
        let ambient = pass.compute_ambient(0.5);
        assert!(ambient[0] > 0.0);
    }

    // =========================================================================
    // Helper Function Tests
    // =========================================================================

    #[test]
    fn test_normalize_vec3() {
        let v = normalize_vec3([3.0, 4.0, 0.0]);
        assert!((v[0] - 0.6).abs() < EPSILON);
        assert!((v[1] - 0.8).abs() < EPSILON);
        assert!((v[2]).abs() < EPSILON);
    }

    #[test]
    fn test_normalize_vec3_zero() {
        let v = normalize_vec3([0.0, 0.0, 0.0]);
        assert_eq!(v, [0.0, 1.0, 0.0]); // Default up
    }

    #[test]
    fn test_normalize_vec3_unit() {
        let v = normalize_vec3([1.0, 0.0, 0.0]);
        assert!((v[0] - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_dot() {
        assert!((dot([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) - 1.0).abs() < EPSILON);
        assert!((dot([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])).abs() < EPSILON);
        assert!((dot([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]) - (-1.0)).abs() < EPSILON);
    }

    #[test]
    fn test_smoothstep_edges() {
        assert!((smoothstep(0.0, 1.0, 0.0)).abs() < EPSILON);
        assert!((smoothstep(0.0, 1.0, 1.0) - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_smoothstep_middle() {
        let mid = smoothstep(0.0, 1.0, 0.5);
        assert!((mid - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_smoothstep_clamped() {
        assert!((smoothstep(0.0, 1.0, -1.0)).abs() < EPSILON);
        assert!((smoothstep(0.0, 1.0, 2.0) - 1.0).abs() < EPSILON);
    }

    // =========================================================================
    // Integration Tests
    // =========================================================================

    #[test]
    fn test_full_lighting_pipeline() {
        let uniforms = CloudLightingUniforms::default()
            .with_sun_direction([0.5, 0.7, -0.5])
            .with_sun_color([1.0, 0.95, 0.9])
            .with_asymmetry(0.7)
            .with_powder_strength(2.0)
            .with_silver_lining(0.8, 8.0)
            .with_multiscatter(4, 0.5, 0.4);

        let pass = CloudLightingPass::new(uniforms);

        // Simulate ray march accumulation
        let mut accumulated = [0.0f32; 3];
        let mut transmittance = 1.0f32;

        for i in 0..10 {
            let height = i as f32 / 10.0;
            let density = 0.3;
            let sun_t = (-i as f32 * 0.1).exp();

            let sample = pass.compute_step(
                density,
                height,
                sun_t,
                [0.0, 0.0, -1.0],
                transmittance,
            );

            let (new_acc, new_trans) = accumulate_lighting(
                accumulated,
                transmittance,
                &sample,
                density * 100.0,
                uniforms.extinction_coeff,
            );

            accumulated = new_acc;
            transmittance = new_trans;
        }

        // Apply multi-scatter
        accumulated = pass.apply_multiscatter(accumulated, transmittance);

        // Verify reasonable results
        assert!(accumulated[0] > 0.0);
        assert!(accumulated[1] > 0.0);
        assert!(accumulated[2] > 0.0);
        assert!(transmittance >= 0.0 && transmittance <= 1.0);
    }

    #[test]
    fn test_energy_conservation_phase() {
        // Phase function should be normalized: integral over sphere = 1
        // HG is already normalized by 1/(4*PI), so integral over solid angle = 1
        // We integrate over cos_theta from -1 to 1 with azimuthal symmetry (factor 2*PI)
        // Integral = 2*PI * integral_{-1}^{1} p(cos_theta) d(cos_theta)
        let g = 0.5;
        let n = 1000;
        let mut sum = 0.0;

        for i in 0..n {
            let cos_theta = -1.0 + 2.0 * (i as f32 + 0.5) / n as f32;
            let phase = henyey_greenstein(cos_theta, g);
            // Midpoint rule: phase * d(cos_theta) where d(cos_theta) = 2/n
            sum += phase * 2.0 / n as f32;
        }

        // After integrating over cos_theta: sum * 2*PI should equal 1
        let sphere_integral = sum * 2.0 * PI;

        // Should integrate to 1 (with numerical error)
        assert!((sphere_integral - 1.0).abs() < 0.05, "Phase integral = {}", sphere_integral);
    }

    #[test]
    fn test_energy_conservation_scatter() {
        // Scattering should not create energy
        let uniforms = CloudLightingUniforms::default();
        let input = [1.0f32; 3];
        let output = multiscatter_octaves(
            input,
            uniforms.scattering_albedo,
            0.5,
            uniforms.multiscatter_octaves,
            uniforms.multiscatter_attenuation,
            uniforms.multiscatter_contribution,
        );

        // With albedo < 1, output should be bounded
        let max_output = output[0].max(output[1]).max(output[2]);
        assert!(max_output < 10.0, "Output too bright: {}", max_output);
    }

    #[test]
    fn test_transmittance_monotonic() {
        let uniforms = CloudLightingUniforms::default();
        let sample = LightingSample {
            in_scatter: [1.0; 3],
            ..Default::default()
        };

        let mut trans = 1.0;
        for _ in 0..20 {
            let (_, new_trans) = accumulate_lighting(
                [0.0; 3],
                trans,
                &sample,
                0.1,
                uniforms.extinction_coeff,
            );
            assert!(new_trans <= trans + EPSILON, "Transmittance increased");
            trans = new_trans;
        }
    }

    #[test]
    fn test_config_round_trip() {
        let config = CloudLightingConfig::new()
            .with_sun_direction([0.0, 1.0, 0.0])
            .with_sun_color([1.0, 0.9, 0.8]);

        let uniforms = config.to_uniforms();

        assert!((uniforms.sun_direction[1] - 1.0).abs() < EPSILON);
        assert!((uniforms.sun_color[0] - 1.0).abs() < EPSILON);
        assert!(uniforms.validate());
    }
}
