//! Water Rendering System for TRINITY Engine.
//!
//! Provides physically-based water surface simulation including:
//! - `fft_multi_cascade`: Multi-cascade FFT ocean for full frequency coverage (T-ENV-3.8)
//! - `fft_ocean`: FFT-based ocean simulation using Phillips spectrum (T-ENV-2.8)
//! - `foam`: Ocean foam simulation for wave crests and shorelines (T-ENV-2.9)
//! - `gerstner`: Gerstner wave displacement for realistic ocean surfaces (T-ENV-1.7)
//! - `shading`: Water shading with Fresnel, refraction, SSS, and GGX specular (T-ENV-1.8)
//!
//! # Gerstner Waves
//!
//! Gerstner waves model the motion of water particles in deep water, where
//! particles move in circular orbits as waves pass. The key insight is that
//! particles are displaced both vertically AND horizontally, creating the
//! characteristic choppy appearance of ocean waves.
//!
//! # Mathematical Foundation
//!
//! For a single Gerstner wave:
//! - Horizontal displacement: `Q * A * D * cos(k * dot(D, p) - w * t)`
//! - Vertical displacement: `A * sin(k * dot(D, p) - w * t)`
//!
//! Where:
//! - `A` = amplitude (wave height)
//! - `Q` = steepness factor (0-1, controls choppiness)
//! - `D` = normalized direction vector
//! - `k` = 2*PI / wavelength (wave number)
//! - `w` = sqrt(g * k) (angular frequency from dispersion relation)
//! - `t` = time
//! - `p` = world position (x, z)
//!
//! # Usage
//!
//! ```ignore
//! use renderer_backend::water::{
//!     GerstnerWave, GerstnerWaveSet, WavePreset,
//! };
//!
//! // Create wave set from preset
//! let mut waves = GerstnerWaveSet::from_preset(WavePreset::Moderate);
//!
//! // Or build custom waves
//! let mut waves = GerstnerWaveSet::new();
//! waves.add_wave(GerstnerWave {
//!     amplitude: 0.5,
//!     wavelength: 10.0,
//!     steepness: 0.5,
//!     speed: 2.0,
//!     direction: [1.0, 0.0],
//!     _padding: [0.0; 2],
//! });
//!
//! // Animate waves
//! waves.set_time(elapsed_seconds);
//!
//! // Evaluate at a point
//! let result = waves.evaluate(x, z);
//! println!("Position: {:?}, Normal: {:?}", result.position, result.normal);
//!
//! // Batch evaluation for mesh
//! let grid = waves.evaluate_grid(64, 0.5, [0.0, 0.0]);
//! ```

pub mod fft_multi_cascade;
pub mod fft_ocean;
pub mod foam;
pub mod foam_advection;
pub mod gerstner;
pub mod shading;
pub mod underwater;

// Re-export main types
pub use gerstner::{
    GerstnerWave, GerstnerWaveSet, GerstnerResult, WavePreset,
    GERSTNER_WAVE_SIZE, MAX_WAVES, WORKGROUP_SIZE,
};

pub use shading::{
    WaterShadingConfig, WaterReflectionConfig, WaterShadingPass,
    fresnel_schlick, fresnel_dielectric, f0_from_ior,
    compute_refraction_offset, snells_law_direction,
    subsurface_color, water_specular_ggx, water_specular_anisotropic,
    WATER_IOR, AIR_IOR, DEFAULT_F0, WATER_SHADING_CONFIG_SIZE,
};

pub use fft_ocean::{
    Complex, FFTOcean, FFTOceanConfig, LCG,
    fft_1d, fft_2d, phillips_spectrum, dispersion_relation, is_power_of_2,
    GRAVITY, DEFAULT_FFT_SIZE, DEFAULT_PATCH_SIZE, FFT_OCEAN_CONFIG_SIZE,
};

pub use foam::{
    FoamConfig, FoamGenerator, FoamState, FoamMask,
    jacobian_from_grid, shore_distance_from_depth,
    DEFAULT_CREST_THRESHOLD, DEFAULT_DECAY_RATE, DEFAULT_SHORE_WIDTH,
    DEFAULT_SHORE_FOAM_MAX, DEFAULT_NOISE_SCALE, DEFAULT_NOISE_INTENSITY,
    FOAM_CONFIG_SIZE, FOAM_STATE_SIZE,
};

pub use fft_multi_cascade::{
    CascadeConfig, MultiCascadeConfig, MultiCascadeOcean, OceanSample,
    smoothstep, smootherstep, cascade_blend_weight,
    MAX_CASCADES, DEFAULT_CASCADE_COUNT, CASCADE_CONFIG_SIZE,
    MULTI_CASCADE_CONFIG_SIZE, OCEAN_SAMPLE_SIZE,
};

pub use underwater::{
    UnderwaterConfig, UnderwaterPostProcess, CausticsGenerator,
    UnderwaterFog, UnderwaterDistortion, GodRays,
    UNDERWATER_CONFIG_SIZE, TRANSITION_ZONE, DEFAULT_CAUSTICS_RESOLUTION,
};

pub use foam_advection::{
    FoamAdvectionConfig, FoamField, VelocityField, FoamAdvector,
    FoamBubble, FoamBubbles, FoamRenderer, TextureHandle,
    DEFAULT_GRID_RESOLUTION, DEFAULT_CELL_SIZE, DEFAULT_ADVECTION_SPEED,
    DEFAULT_DIFFUSION_RATE, DEFAULT_SPAWN_RATE, DEFAULT_BUBBLE_LIFETIME,
    MAX_BUBBLES, BUBBLE_GRAVITY, BUBBLE_DRAG, FOAM_ADVECTION_CONFIG_SIZE,
};
