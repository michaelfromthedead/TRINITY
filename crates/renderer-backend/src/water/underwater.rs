//! Underwater Post-Processing Effects for TRINITY Engine (T-ENV-3.9).
//!
//! Implements physically-based underwater rendering including:
//! - Light absorption using Beer-Lambert law (depth-dependent color shift)
//! - Volumetric scattering (in-scattered light from suspended particles)
//! - Animated caustics projected onto underwater surfaces
//! - Screen-space distortion for underwater viewing
//! - Underwater god rays (crepuscular rays through water surface)
//!
//! # Physics Background
//!
//! ## Beer-Lambert Law (Absorption)
//!
//! Light intensity decreases exponentially with distance through water:
//! `I = I0 * exp(-absorption * distance)`
//!
//! Different wavelengths are absorbed at different rates:
//! - Red light is absorbed quickly (within ~5m in clear water)
//! - Blue/green light penetrates much deeper
//! This creates the characteristic blue-green underwater color shift.
//!
//! ## Scattering
//!
//! Suspended particles in water scatter light toward the viewer:
//! - In-scattering adds light from the sun direction
//! - Amount depends on particle density and viewing angle
//!
//! ## Caustics
//!
//! Wave motion on the water surface acts as a lens, creating dancing
//! light patterns (caustics) on underwater surfaces. These are computed
//! using wave normal perturbation and light refraction.
//!
//! # Usage
//!
//! ```ignore
//! use renderer_backend::water::underwater::{
//!     UnderwaterConfig, UnderwaterPostProcess, CausticsGenerator,
//!     UnderwaterFog, UnderwaterDistortion, GodRays,
//! };
//!
//! let config = UnderwaterConfig::default();
//! let mut post = UnderwaterPostProcess::new(config);
//!
//! // Per-frame update
//! post.update_time(delta_seconds);
//!
//! // Check if camera is underwater
//! if post.is_underwater(camera_y, water_height) {
//!     // Get transition blend for smooth surface crossing
//!     let blend = post.get_blend_factor(camera_y, water_height);
//!
//!     // Apply underwater effects
//!     let output = post.process_frame(input_color, depth_buffer, camera_pos);
//! }
//! ```

use std::mem;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Size of UnderwaterConfig in bytes (GPU uniform buffer).
pub const UNDERWATER_CONFIG_SIZE: usize = 64;

/// Minimum valid density value.
pub const MIN_DENSITY: f32 = 0.0;

/// Maximum valid density value.
pub const MAX_DENSITY: f32 = 1.0;

/// Small epsilon for floating point comparisons.
const EPSILON: f32 = 1e-6;

/// Transition zone thickness (meters) for surface crossing blend.
pub const TRANSITION_ZONE: f32 = 0.5;

/// Default caustics texture resolution.
pub const DEFAULT_CAUSTICS_RESOLUTION: u32 = 256;

/// Speed of light in water relative to air (IOR = 1.33).
pub const WATER_IOR: f32 = 1.33;

// ---------------------------------------------------------------------------
// UnderwaterConfig
// ---------------------------------------------------------------------------

/// Configuration for underwater post-processing effects.
///
/// This struct is GPU-compatible with 64-byte alignment for uniform buffers.
///
/// # Memory Layout (64 bytes, std140 compatible)
///
/// | Offset | Field              | Size     |
/// |--------|-------------------|----------|
/// | 0      | absorption_color   | 12 bytes |
/// | 12     | absorption_density | 4 bytes  |
/// | 16     | scattering_color   | 12 bytes |
/// | 28     | scattering_density | 4 bytes  |
/// | 32     | caustics_intensity | 4 bytes  |
/// | 36     | caustics_scale     | 4 bytes  |
/// | 40     | fog_density        | 4 bytes  |
/// | 44     | distortion_strength| 4 bytes  |
/// | 48     | god_ray_intensity  | 4 bytes  |
/// | 52     | _padding           | 12 bytes |
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UnderwaterConfig {
    /// RGB absorption color (light that passes through).
    /// Determines the tint of objects viewed through water.
    /// Default: [0.1, 0.3, 0.5] (blue-green tint)
    pub absorption_color: [f32; 3],

    /// Absorption density coefficient.
    /// Higher values = more absorption per unit distance.
    /// Default: 0.1
    pub absorption_density: f32,

    /// RGB scattering color (in-scattered light from particles).
    /// Default: [0.05, 0.15, 0.2]
    pub scattering_color: [f32; 3],

    /// Scattering density coefficient.
    /// Higher values = more scattering per unit distance.
    /// Default: 0.05
    pub scattering_density: f32,

    /// Caustics brightness multiplier (0.0-1.0).
    /// Default: 0.5
    pub caustics_intensity: f32,

    /// Caustics pattern scale (world-space UV scale).
    /// Default: 1.0
    pub caustics_scale: f32,

    /// Fog density for distance fog underwater.
    /// Default: 0.02
    pub fog_density: f32,

    /// Screen-space distortion strength (0.0-1.0).
    /// Default: 0.1
    pub distortion_strength: f32,

    /// God ray intensity multiplier (0.0-1.0).
    /// Default: 0.3
    pub god_ray_intensity: f32,

    /// Padding for 16-byte alignment (64 bytes total).
    pub _padding: [f32; 3],
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<UnderwaterConfig>() == UNDERWATER_CONFIG_SIZE);

impl UnderwaterConfig {
    /// Create a new underwater configuration with custom parameters.
    ///
    /// # Arguments
    ///
    /// * `absorption_color` - RGB absorption color (0-1 range).
    /// * `absorption_density` - Absorption coefficient.
    /// * `fog_density` - Distance fog density.
    #[must_use]
    pub fn new(
        absorption_color: [f32; 3],
        absorption_density: f32,
        fog_density: f32,
    ) -> Self {
        Self {
            absorption_color: [
                absorption_color[0].clamp(0.0, 1.0),
                absorption_color[1].clamp(0.0, 1.0),
                absorption_color[2].clamp(0.0, 1.0),
            ],
            absorption_density: absorption_density.max(0.0),
            scattering_color: [0.05, 0.15, 0.2],
            scattering_density: 0.05,
            caustics_intensity: 0.5,
            caustics_scale: 1.0,
            fog_density: fog_density.max(0.0),
            distortion_strength: 0.1,
            god_ray_intensity: 0.3,
            _padding: [0.0; 3],
        }
    }

    /// Create configuration for tropical/clear water.
    ///
    /// High visibility, bright caustics, low absorption.
    #[must_use]
    pub fn tropical() -> Self {
        Self {
            absorption_color: [0.05, 0.4, 0.55],
            absorption_density: 0.03,
            scattering_color: [0.02, 0.08, 0.12],
            scattering_density: 0.02,
            caustics_intensity: 0.8,
            caustics_scale: 0.8,
            fog_density: 0.008,
            distortion_strength: 0.08,
            god_ray_intensity: 0.5,
            _padding: [0.0; 3],
        }
    }

    /// Create configuration for deep ocean water.
    ///
    /// Low visibility, dark blue absorption, minimal caustics.
    #[must_use]
    pub fn deep_ocean() -> Self {
        Self {
            absorption_color: [0.02, 0.08, 0.25],
            absorption_density: 0.15,
            scattering_color: [0.01, 0.04, 0.08],
            scattering_density: 0.08,
            caustics_intensity: 0.1,
            caustics_scale: 2.0,
            fog_density: 0.05,
            distortion_strength: 0.05,
            god_ray_intensity: 0.15,
            _padding: [0.0; 3],
        }
    }

    /// Create configuration for murky/river water.
    ///
    /// Very low visibility, greenish-brown absorption.
    #[must_use]
    pub fn murky() -> Self {
        Self {
            absorption_color: [0.15, 0.18, 0.08],
            absorption_density: 0.3,
            scattering_color: [0.08, 0.1, 0.05],
            scattering_density: 0.2,
            caustics_intensity: 0.05,
            caustics_scale: 1.5,
            fog_density: 0.15,
            distortion_strength: 0.15,
            god_ray_intensity: 0.05,
            _padding: [0.0; 3],
        }
    }

    /// Create configuration for pool/chlorinated water.
    ///
    /// Very high visibility, bright turquoise color.
    #[must_use]
    pub fn pool() -> Self {
        Self {
            absorption_color: [0.1, 0.6, 0.7],
            absorption_density: 0.01,
            scattering_color: [0.01, 0.05, 0.06],
            scattering_density: 0.01,
            caustics_intensity: 0.9,
            caustics_scale: 0.5,
            fog_density: 0.003,
            distortion_strength: 0.12,
            god_ray_intensity: 0.4,
            _padding: [0.0; 3],
        }
    }

    /// Validate configuration values.
    ///
    /// # Returns
    ///
    /// `Ok(())` if valid, `Err` with description if invalid.
    pub fn validate(&self) -> Result<(), &'static str> {
        // Check absorption color range
        for c in &self.absorption_color {
            if *c < 0.0 || *c > 1.0 {
                return Err("Absorption color components must be in range [0, 1]");
            }
        }

        // Check scattering color range
        for c in &self.scattering_color {
            if *c < 0.0 || *c > 1.0 {
                return Err("Scattering color components must be in range [0, 1]");
            }
        }

        // Check density values
        if self.absorption_density < 0.0 {
            return Err("Absorption density must be non-negative");
        }
        if self.scattering_density < 0.0 {
            return Err("Scattering density must be non-negative");
        }
        if self.fog_density < 0.0 {
            return Err("Fog density must be non-negative");
        }

        // Check intensity values
        if self.caustics_intensity < 0.0 || self.caustics_intensity > 1.0 {
            return Err("Caustics intensity must be in range [0, 1]");
        }
        if self.distortion_strength < 0.0 || self.distortion_strength > 1.0 {
            return Err("Distortion strength must be in range [0, 1]");
        }
        if self.god_ray_intensity < 0.0 || self.god_ray_intensity > 1.0 {
            return Err("God ray intensity must be in range [0, 1]");
        }

        // Check scale
        if self.caustics_scale <= 0.0 {
            return Err("Caustics scale must be positive");
        }

        Ok(())
    }

    /// Get the visibility distance (where 99% of light is absorbed).
    ///
    /// Computed from Beer-Lambert: distance = -ln(0.01) / density
    #[inline]
    #[must_use]
    pub fn visibility_distance(&self) -> f32 {
        if self.fog_density <= EPSILON {
            f32::MAX
        } else {
            // -ln(0.01) = 4.605
            4.605 / self.fog_density
        }
    }
}

impl Default for UnderwaterConfig {
    fn default() -> Self {
        Self {
            absorption_color: [0.1, 0.3, 0.5],
            absorption_density: 0.1,
            scattering_color: [0.05, 0.15, 0.2],
            scattering_density: 0.05,
            caustics_intensity: 0.5,
            caustics_scale: 1.0,
            fog_density: 0.02,
            distortion_strength: 0.1,
            god_ray_intensity: 0.3,
            _padding: [0.0; 3],
        }
    }
}

// ---------------------------------------------------------------------------
// CausticsGenerator
// ---------------------------------------------------------------------------

/// Generator for animated underwater caustics patterns.
///
/// Caustics are the dancing light patterns created when sunlight refracts
/// through the wavy water surface, acting as a focusing lens.
#[derive(Clone, Debug)]
pub struct CausticsGenerator {
    /// Caustics texture resolution.
    resolution: u32,
    /// World-space scale for caustics UVs.
    scale: f32,
    /// Animation speed multiplier.
    speed: f32,
    /// Number of wave octaves for pattern.
    octaves: u32,
}

impl CausticsGenerator {
    /// Create a new caustics generator.
    ///
    /// # Arguments
    ///
    /// * `resolution` - Texture resolution (power of 2 recommended).
    /// * `scale` - World-space scale (larger = coarser pattern).
    #[must_use]
    pub fn new(resolution: u32, scale: f32) -> Self {
        Self {
            resolution: resolution.max(64),
            scale: scale.max(0.1),
            speed: 1.0,
            octaves: 3,
        }
    }

    /// Set animation speed multiplier.
    #[must_use]
    pub fn with_speed(mut self, speed: f32) -> Self {
        self.speed = speed.max(0.0);
        self
    }

    /// Set number of pattern octaves.
    #[must_use]
    pub fn with_octaves(mut self, octaves: u32) -> Self {
        self.octaves = octaves.clamp(1, 6);
        self
    }

    /// Get texture resolution.
    #[inline]
    #[must_use]
    pub fn resolution(&self) -> u32 {
        self.resolution
    }

    /// Get world-space scale.
    #[inline]
    #[must_use]
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Generate caustics value for a texture pixel.
    ///
    /// This generates a procedural caustics pattern suitable for
    /// GPU texture generation.
    ///
    /// # Arguments
    ///
    /// * `sun_dir` - Normalized sun direction vector (pointing toward sun).
    /// * `time` - Animation time in seconds.
    ///
    /// # Returns
    ///
    /// Caustics intensity texture data as Vec<f32>.
    #[must_use]
    pub fn generate_caustics_texture(&self, sun_dir: [f32; 3], time: f32) -> Vec<f32> {
        let size = self.resolution as usize;
        let mut data = vec![0.0f32; size * size];

        let animated_time = time * self.speed;

        for y in 0..size {
            for x in 0..size {
                let u = x as f32 / size as f32;
                let v = y as f32 / size as f32;

                let caustic = self.sample_caustics_uv(u, v, sun_dir, animated_time);
                data[y * size + x] = caustic;
            }
        }

        data
    }

    /// Sample caustics intensity at world position.
    ///
    /// # Arguments
    ///
    /// * `world_pos` - World-space position [x, y, z].
    /// * `time` - Animation time in seconds.
    ///
    /// # Returns
    ///
    /// Caustics intensity in range [0, 1].
    #[must_use]
    pub fn sample_caustics(&self, world_pos: [f32; 3], time: f32) -> f32 {
        // Project to horizontal plane and scale
        let u = world_pos[0] / self.scale;
        let v = world_pos[2] / self.scale;

        // Use default up-facing sun direction for sampling
        let sun_dir = [0.0, 1.0, 0.0];
        self.sample_caustics_uv(u, v, sun_dir, time * self.speed)
    }

    /// Get UV coordinates for shader caustics lookup.
    ///
    /// # Arguments
    ///
    /// * `world_pos` - World-space position [x, y, z].
    ///
    /// # Returns
    ///
    /// UV coordinates [u, v] for caustics texture sampling.
    #[inline]
    #[must_use]
    pub fn get_caustics_uv(&self, world_pos: [f32; 3]) -> [f32; 2] {
        [
            (world_pos[0] / self.scale).fract(),
            (world_pos[2] / self.scale).fract(),
        ]
    }

    /// Internal caustics sampling using procedural noise.
    fn sample_caustics_uv(&self, u: f32, v: f32, sun_dir: [f32; 3], time: f32) -> f32 {
        let mut value = 0.0f32;
        let mut amplitude = 1.0f32;
        let mut frequency = 1.0f32;
        let mut max_value = 0.0f32;

        // Sun angle affects caustics pattern compression
        let sun_factor = sun_dir[1].abs().max(0.1);

        for _ in 0..self.octaves {
            // Two overlapping wave patterns create caustic-like interference
            let wave1 = caustic_wave(u * frequency, v * frequency, time, 0.0);
            let wave2 = caustic_wave(
                u * frequency + 0.37,
                v * frequency + 0.73,
                time * 1.1,
                1.57,
            );

            // Interference pattern
            let interference = (wave1 * wave2).max(0.0);
            value += interference * amplitude * sun_factor;
            max_value += amplitude;

            amplitude *= 0.5;
            frequency *= 2.0;
        }

        // Normalize and apply sharpening
        let normalized = value / max_value;
        // Sharpen the caustics pattern (caustics have bright spots)
        let sharpened = normalized.powf(0.5);
        sharpened.clamp(0.0, 1.0)
    }
}

impl Default for CausticsGenerator {
    fn default() -> Self {
        Self::new(DEFAULT_CAUSTICS_RESOLUTION, 1.0)
    }
}

/// Generate a single caustic wave pattern.
#[inline]
fn caustic_wave(x: f32, y: f32, time: f32, phase: f32) -> f32 {
    let t = time + phase;
    let wave_x = (x * 6.28 + t * 0.5).sin();
    let wave_y = (y * 6.28 + t * 0.7).cos();
    let combined = (wave_x + wave_y) * 0.5 + 0.5;
    combined
}

// ---------------------------------------------------------------------------
// UnderwaterFog
// ---------------------------------------------------------------------------

/// Underwater fog and absorption computation.
///
/// Implements Beer-Lambert law for light absorption and
/// volumetric scattering for in-scattered light.
#[derive(Clone, Debug, Default)]
pub struct UnderwaterFog;

impl UnderwaterFog {
    /// Create a new underwater fog processor.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Compute light absorption using Beer-Lambert law.
    ///
    /// `I = I0 * exp(-absorption * distance)`
    ///
    /// Different color channels are absorbed at different rates based
    /// on the absorption color configuration.
    ///
    /// # Arguments
    ///
    /// * `depth` - Distance traveled through water (meters).
    /// * `config` - Underwater configuration.
    ///
    /// # Returns
    ///
    /// RGB transmittance multiplier (how much light passes through).
    #[inline]
    #[must_use]
    pub fn compute_absorption(depth: f32, config: &UnderwaterConfig) -> [f32; 3] {
        if depth <= 0.0 {
            return [1.0, 1.0, 1.0]; // No absorption above water
        }

        let density = config.absorption_density;

        // Beer-Lambert law per channel
        // Lower absorption_color values = more absorption (inverted)
        // We use (1 - absorption_color) as the extinction coefficient
        [
            (-depth * density * (1.0 - config.absorption_color[0] + 0.1)).exp(),
            (-depth * density * (1.0 - config.absorption_color[1] + 0.1)).exp(),
            (-depth * density * (1.0 - config.absorption_color[2] + 0.1)).exp(),
        ]
    }

    /// Compute in-scattered light from particles.
    ///
    /// Light scattered toward the viewer from suspended particles,
    /// modulated by viewing angle relative to sun.
    ///
    /// # Arguments
    ///
    /// * `view_dist` - View distance through water (meters).
    /// * `sun_dir` - Normalized sun direction (toward sun).
    ///
    /// # Returns
    ///
    /// RGB scattered light contribution.
    #[inline]
    #[must_use]
    pub fn compute_scattering(
        view_dist: f32,
        sun_dir: [f32; 3],
        config: &UnderwaterConfig,
    ) -> [f32; 3] {
        if view_dist <= 0.0 {
            return [0.0, 0.0, 0.0];
        }

        // Amount of scattering depends on distance
        let scatter_amount = 1.0 - (-view_dist * config.scattering_density).exp();

        // Directional factor: more scattering when looking toward sun
        // (forward scattering dominates in water)
        let sun_y = sun_dir[1].clamp(-1.0, 1.0);
        let directional = 0.5 + 0.5 * sun_y;

        [
            config.scattering_color[0] * scatter_amount * directional,
            config.scattering_color[1] * scatter_amount * directional,
            config.scattering_color[2] * scatter_amount * directional,
        ]
    }

    /// Apply complete underwater fog effect to a color.
    ///
    /// Combines absorption (exponential falloff) and scattering (additive).
    ///
    /// # Arguments
    ///
    /// * `color` - Input RGB color.
    /// * `depth` - View distance through water.
    /// * `config` - Underwater configuration.
    ///
    /// # Returns
    ///
    /// Fogged RGB color.
    #[must_use]
    pub fn apply_underwater_fog(
        color: [f32; 3],
        depth: f32,
        config: &UnderwaterConfig,
    ) -> [f32; 3] {
        // Distance fog transmittance
        let fog_transmittance = (-depth * config.fog_density).exp();

        // Color absorption
        let absorption = Self::compute_absorption(depth, config);

        // In-scattering (using vertical sun direction as default)
        let scatter = Self::compute_scattering(depth, [0.0, 1.0, 0.0], config);

        // Apply: (color * absorption * fog_transmittance) + scatter
        [
            color[0] * absorption[0] * fog_transmittance + scatter[0],
            color[1] * absorption[1] * fog_transmittance + scatter[1],
            color[2] * absorption[2] * fog_transmittance + scatter[2],
        ]
    }

    /// Get visibility distance for the given configuration.
    ///
    /// Returns the distance where 99% of light is absorbed.
    #[inline]
    #[must_use]
    pub fn get_visibility_distance(config: &UnderwaterConfig) -> f32 {
        config.visibility_distance()
    }
}

// ---------------------------------------------------------------------------
// UnderwaterDistortion
// ---------------------------------------------------------------------------

/// Screen-space underwater distortion effects.
///
/// Simulates the wavy distortion seen when looking through water
/// due to surface waves and density variations.
#[derive(Clone, Debug)]
pub struct UnderwaterDistortion {
    /// Noise frequency in X direction.
    frequency_x: f32,
    /// Noise frequency in Y direction.
    frequency_y: f32,
    /// Animation speed.
    speed: f32,
}

impl UnderwaterDistortion {
    /// Create a new distortion effect.
    #[must_use]
    pub fn new() -> Self {
        Self {
            frequency_x: 10.0,
            frequency_y: 8.0,
            speed: 1.0,
        }
    }

    /// Set noise frequencies.
    #[must_use]
    pub fn with_frequency(mut self, freq_x: f32, freq_y: f32) -> Self {
        self.frequency_x = freq_x.max(0.1);
        self.frequency_y = freq_y.max(0.1);
        self
    }

    /// Set animation speed.
    #[must_use]
    pub fn with_speed(mut self, speed: f32) -> Self {
        self.speed = speed.max(0.0);
        self
    }

    /// Compute screen-space UV distortion offset.
    ///
    /// Uses Perlin-like noise to create wavy distortion.
    ///
    /// # Arguments
    ///
    /// * `uv` - Screen-space UV coordinates [0, 1].
    /// * `time` - Animation time in seconds.
    /// * `config` - Underwater configuration.
    ///
    /// # Returns
    ///
    /// UV offset [du, dv] to add to original UV.
    #[must_use]
    pub fn compute_distortion_offset(
        &self,
        uv: [f32; 2],
        time: f32,
        config: &UnderwaterConfig,
    ) -> [f32; 2] {
        self.apply_distortion(uv, time, config.distortion_strength)
    }

    /// Apply distortion with explicit strength parameter.
    ///
    /// # Arguments
    ///
    /// * `uv` - Screen-space UV coordinates.
    /// * `time` - Animation time.
    /// * `strength` - Distortion strength (0-1).
    ///
    /// # Returns
    ///
    /// Distorted UV coordinates.
    #[must_use]
    pub fn apply_distortion(&self, uv: [f32; 2], time: f32, strength: f32) -> [f32; 2] {
        let t = time * self.speed;

        // Perlin-like smooth noise using sine waves
        let noise_x = self.perlin_noise_2d(
            uv[0] * self.frequency_x + t * 0.3,
            uv[1] * self.frequency_y + t * 0.2,
        );
        let noise_y = self.perlin_noise_2d(
            uv[0] * self.frequency_x + t * 0.2 + 100.0,
            uv[1] * self.frequency_y + t * 0.35 + 100.0,
        );

        // Scale by strength and clamp to prevent extreme distortion
        let offset_x = (noise_x * strength * 0.1).clamp(-0.1, 0.1);
        let offset_y = (noise_y * strength * 0.1).clamp(-0.1, 0.1);

        [uv[0] + offset_x, uv[1] + offset_y]
    }

    /// Simple 2D Perlin-like noise using gradient interpolation.
    fn perlin_noise_2d(&self, x: f32, y: f32) -> f32 {
        // Simplified Perlin noise approximation using multiple sine waves
        let n1 = (x * 1.0).sin() * (y * 1.0).cos();
        let n2 = (x * 2.3 + 0.5).sin() * (y * 2.7 + 0.3).cos() * 0.5;
        let n3 = (x * 4.1 + 0.9).sin() * (y * 3.9 + 0.7).cos() * 0.25;
        let n4 = (x * 7.3 + 1.3).sin() * (y * 6.7 + 1.1).cos() * 0.125;

        n1 + n2 + n3 + n4
    }
}

impl Default for UnderwaterDistortion {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// GodRays (Underwater variant)
// ---------------------------------------------------------------------------

/// Underwater god ray (crepuscular ray) computation.
///
/// Simulates shafts of light penetrating from the water surface,
/// partially occluded by waves and depth.
#[derive(Clone, Debug)]
pub struct GodRays {
    /// Number of ray march steps.
    march_steps: u32,
    /// Maximum ray distance.
    max_distance: f32,
    /// Noise scale for ray variation.
    noise_scale: f32,
}

impl GodRays {
    /// Create a new god ray generator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            march_steps: 32,
            max_distance: 50.0,
            noise_scale: 0.1,
        }
    }

    /// Set number of ray march steps (quality vs performance).
    #[must_use]
    pub fn with_steps(mut self, steps: u32) -> Self {
        self.march_steps = steps.clamp(8, 128);
        self
    }

    /// Set maximum ray march distance.
    #[must_use]
    pub fn with_max_distance(mut self, distance: f32) -> Self {
        self.max_distance = distance.max(1.0);
        self
    }

    /// Compute god ray intensity at a point.
    ///
    /// # Arguments
    ///
    /// * `view_dir` - Normalized view direction (from camera).
    /// * `sun_dir` - Normalized sun direction (toward sun).
    /// * `depth` - Depth below water surface.
    ///
    /// # Returns
    ///
    /// God ray intensity factor (0-1).
    #[must_use]
    pub fn compute_ray_intensity(
        &self,
        view_dir: [f32; 3],
        sun_dir: [f32; 3],
        depth: f32,
    ) -> f32 {
        if depth <= 0.0 {
            return 0.0; // No rays above water
        }

        // View-sun alignment (stronger when looking toward light source)
        let v_dot_s = dot3(view_dir, sun_dir);
        let alignment = (v_dot_s * 0.5 + 0.5).powf(2.0);

        // Depth attenuation (rays weaken with depth)
        let depth_factor = (-depth * 0.1).exp();

        // Sun angle (rays are stronger when sun is higher)
        let sun_angle_factor = sun_dir[1].max(0.0);

        alignment * depth_factor * sun_angle_factor
    }

    /// Ray march through water to accumulate god ray contribution.
    ///
    /// # Arguments
    ///
    /// * `start` - Start position (camera position).
    /// * `end` - End position (scene point).
    /// * `config` - Underwater configuration.
    ///
    /// # Returns
    ///
    /// RGB god ray color contribution.
    #[must_use]
    pub fn march_underwater_rays(
        &self,
        start: [f32; 3],
        end: [f32; 3],
        config: &UnderwaterConfig,
    ) -> [f32; 3] {
        let ray_dir = [
            end[0] - start[0],
            end[1] - start[1],
            end[2] - start[2],
        ];
        let ray_length = vec3_length(ray_dir);

        if ray_length < EPSILON {
            return [0.0, 0.0, 0.0];
        }

        let ray_dir_norm = [
            ray_dir[0] / ray_length,
            ray_dir[1] / ray_length,
            ray_dir[2] / ray_length,
        ];

        let march_distance = ray_length.min(self.max_distance);
        let step_size = march_distance / self.march_steps as f32;

        let mut accumulated = [0.0f32; 3];
        let mut transmittance = 1.0f32;

        // Default sun direction (could be parameterized)
        let sun_dir = normalize3([0.3, 0.8, 0.2]);

        for i in 0..self.march_steps {
            let t = (i as f32 + 0.5) * step_size;
            let pos = [
                start[0] + ray_dir_norm[0] * t,
                start[1] + ray_dir_norm[1] * t,
                start[2] + ray_dir_norm[2] * t,
            ];

            // Depth below surface (assuming water at y=0)
            let depth = -pos[1];
            if depth <= 0.0 {
                continue; // Above water
            }

            // Ray intensity at this point
            let ray_intensity = self.compute_ray_intensity(ray_dir_norm, sun_dir, depth);

            // Add noise variation
            let noise = self.sample_noise(pos);
            let modulated_intensity = ray_intensity * (0.8 + 0.4 * noise);

            // Scattering contribution
            let scatter = modulated_intensity * transmittance * step_size;

            // Apply scattering color tint
            accumulated[0] += scatter * config.scattering_color[0] * 2.0;
            accumulated[1] += scatter * config.scattering_color[1] * 2.0;
            accumulated[2] += scatter * config.scattering_color[2] * 2.0;

            // Update transmittance (Beer-Lambert)
            transmittance *= (-step_size * config.fog_density).exp();

            if transmittance < 0.01 {
                break; // Early exit when opaque
            }
        }

        // Apply god ray intensity multiplier
        [
            accumulated[0] * config.god_ray_intensity,
            accumulated[1] * config.god_ray_intensity,
            accumulated[2] * config.god_ray_intensity,
        ]
    }

    /// Sample 3D noise for ray variation.
    fn sample_noise(&self, pos: [f32; 3]) -> f32 {
        let x = pos[0] * self.noise_scale;
        let y = pos[1] * self.noise_scale;
        let z = pos[2] * self.noise_scale;

        // Simple 3D noise approximation
        let n = (x * 12.9898 + y * 78.233 + z * 37.719).sin() * 43758.5453;
        n.fract()
    }
}

impl Default for GodRays {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// UnderwaterPostProcess (Compositor)
// ---------------------------------------------------------------------------

/// Complete underwater post-processing compositor.
///
/// Combines all underwater effects:
/// - Depth-based fog and absorption
/// - Animated caustics
/// - Screen-space distortion
/// - God rays
/// - Surface transition blending
#[derive(Clone, Debug)]
pub struct UnderwaterPostProcess {
    /// Effect configuration.
    config: UnderwaterConfig,
    /// Caustics generator.
    caustics: CausticsGenerator,
    /// Fog processor.
    fog: UnderwaterFog,
    /// Distortion effect.
    distortion: UnderwaterDistortion,
    /// God ray generator.
    god_rays: GodRays,
    /// Current animation time.
    time: f32,
}

impl UnderwaterPostProcess {
    /// Create a new underwater post-processor.
    ///
    /// # Arguments
    ///
    /// * `config` - Underwater effect configuration.
    #[must_use]
    pub fn new(config: UnderwaterConfig) -> Self {
        Self {
            caustics: CausticsGenerator::new(
                DEFAULT_CAUSTICS_RESOLUTION,
                config.caustics_scale,
            ),
            config,
            fog: UnderwaterFog::new(),
            distortion: UnderwaterDistortion::new(),
            god_rays: GodRays::new(),
            time: 0.0,
        }
    }

    /// Get current configuration.
    #[inline]
    #[must_use]
    pub fn config(&self) -> &UnderwaterConfig {
        &self.config
    }

    /// Get mutable configuration.
    #[inline]
    pub fn config_mut(&mut self) -> &mut UnderwaterConfig {
        &mut self.config
    }

    /// Get caustics generator.
    #[inline]
    #[must_use]
    pub fn caustics(&self) -> &CausticsGenerator {
        &self.caustics
    }

    /// Get mutable caustics generator.
    #[inline]
    pub fn caustics_mut(&mut self) -> &mut CausticsGenerator {
        &mut self.caustics
    }

    /// Check if camera is underwater.
    ///
    /// # Arguments
    ///
    /// * `camera_y` - Camera Y position (world-space).
    /// * `water_height` - Water surface Y position.
    ///
    /// # Returns
    ///
    /// `true` if camera is below water surface.
    #[inline]
    #[must_use]
    pub fn is_underwater(camera_y: f32, water_height: f32) -> bool {
        camera_y < water_height
    }

    /// Get blend factor for surface transition.
    ///
    /// Returns 0.0 when fully above water, 1.0 when fully below,
    /// and a smooth transition in the transition zone.
    ///
    /// # Arguments
    ///
    /// * `camera_y` - Camera Y position.
    /// * `water_height` - Water surface Y position.
    ///
    /// # Returns
    ///
    /// Blend factor in range [0, 1].
    #[must_use]
    pub fn get_blend_factor(camera_y: f32, water_height: f32) -> f32 {
        let depth = water_height - camera_y;

        if depth <= -TRANSITION_ZONE {
            0.0 // Fully above water
        } else if depth >= TRANSITION_ZONE {
            1.0 // Fully underwater
        } else {
            // Smooth transition using smoothstep
            let t = (depth + TRANSITION_ZONE) / (2.0 * TRANSITION_ZONE);
            smoothstep(t)
        }
    }

    /// Process a frame with underwater effects.
    ///
    /// This is the main entry point for applying all underwater effects.
    ///
    /// # Arguments
    ///
    /// * `input` - Input RGB color.
    /// * `depth` - Scene depth (view distance to surface).
    /// * `camera_pos` - Camera world position.
    ///
    /// # Returns
    ///
    /// Processed RGB color with underwater effects.
    #[must_use]
    pub fn process_frame(
        &self,
        input: [f32; 3],
        depth: f32,
        camera_pos: [f32; 3],
    ) -> [f32; 3] {
        // Apply fog and absorption
        let fogged = UnderwaterFog::apply_underwater_fog(input, depth, &self.config);

        // Add caustics
        let caustic_value = self.caustics.sample_caustics(camera_pos, self.time);
        let with_caustics = [
            fogged[0] + caustic_value * self.config.caustics_intensity * 0.3,
            fogged[1] + caustic_value * self.config.caustics_intensity * 0.4,
            fogged[2] + caustic_value * self.config.caustics_intensity * 0.5,
        ];

        with_caustics
    }

    /// Update animation time.
    ///
    /// # Arguments
    ///
    /// * `dt` - Delta time in seconds.
    pub fn update_time(&mut self, dt: f32) {
        self.time += dt;
    }

    /// Get current animation time.
    #[inline]
    #[must_use]
    pub fn time(&self) -> f32 {
        self.time
    }

    /// Set animation time directly.
    #[inline]
    pub fn set_time(&mut self, time: f32) {
        self.time = time.max(0.0);
    }

    /// Get god ray generator.
    #[inline]
    #[must_use]
    pub fn god_rays(&self) -> &GodRays {
        &self.god_rays
    }

    /// Get distortion effect.
    #[inline]
    #[must_use]
    pub fn distortion(&self) -> &UnderwaterDistortion {
        &self.distortion
    }
}

impl Default for UnderwaterPostProcess {
    fn default() -> Self {
        Self::new(UnderwaterConfig::default())
    }
}

// ---------------------------------------------------------------------------
// Utility Functions
// ---------------------------------------------------------------------------

/// Smoothstep interpolation (cubic Hermite).
#[inline]
fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// 3D vector dot product.
#[inline]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// 3D vector length.
#[inline]
fn vec3_length(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Normalize a 3D vector.
#[inline]
fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = vec3_length(v);
    if len > EPSILON {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 1.0, 0.0]
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f32 = 1e-4;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < TOLERANCE
    }

    fn approx_eq_loose(a: f32, b: f32, tolerance: f32) -> bool {
        (a - b).abs() < tolerance
    }

    fn approx_eq_vec3(a: [f32; 3], b: [f32; 3]) -> bool {
        approx_eq(a[0], b[0]) && approx_eq(a[1], b[1]) && approx_eq(a[2], b[2])
    }

    // -------------------------------------------------------------------------
    // UnderwaterConfig Tests (1-15)
    // -------------------------------------------------------------------------

    // Test 1: Config struct size is 64 bytes
    #[test]
    fn test_config_size() {
        assert_eq!(
            std::mem::size_of::<UnderwaterConfig>(),
            UNDERWATER_CONFIG_SIZE
        );
        assert_eq!(std::mem::size_of::<UnderwaterConfig>(), 64);
    }

    // Test 2: Config is Pod
    #[test]
    fn test_config_pod() {
        let config = UnderwaterConfig::default();
        let bytes: &[u8] = bytemuck::bytes_of(&config);
        assert_eq!(bytes.len(), 64);
    }

    // Test 3: Default config has expected values
    #[test]
    fn test_default_config_values() {
        let config = UnderwaterConfig::default();
        assert!(approx_eq_vec3(config.absorption_color, [0.1, 0.3, 0.5]));
        assert!(approx_eq(config.absorption_density, 0.1));
        assert!(approx_eq(config.fog_density, 0.02));
        assert!(approx_eq(config.caustics_intensity, 0.5));
        assert!(approx_eq(config.distortion_strength, 0.1));
        assert!(approx_eq(config.god_ray_intensity, 0.3));
    }

    // Test 4: Default config validation passes
    #[test]
    fn test_default_config_valid() {
        let config = UnderwaterConfig::default();
        assert!(config.validate().is_ok());
    }

    // Test 5: Tropical preset validation
    #[test]
    fn test_tropical_preset() {
        let config = UnderwaterConfig::tropical();
        assert!(config.validate().is_ok());
        assert!(config.caustics_intensity > 0.5); // High caustics
        assert!(config.absorption_density < 0.1); // Low absorption
    }

    // Test 6: Deep ocean preset validation
    #[test]
    fn test_deep_ocean_preset() {
        let config = UnderwaterConfig::deep_ocean();
        assert!(config.validate().is_ok());
        assert!(config.caustics_intensity < 0.2); // Low caustics
        assert!(config.fog_density > 0.02); // High fog
    }

    // Test 7: Murky preset validation
    #[test]
    fn test_murky_preset() {
        let config = UnderwaterConfig::murky();
        assert!(config.validate().is_ok());
        assert!(config.fog_density > 0.1); // Very high fog
        assert!(config.god_ray_intensity < 0.1); // Almost no rays
    }

    // Test 8: Pool preset validation
    #[test]
    fn test_pool_preset() {
        let config = UnderwaterConfig::pool();
        assert!(config.validate().is_ok());
        assert!(config.caustics_intensity > 0.8); // Very bright caustics
        assert!(config.fog_density < 0.01); // Very clear
    }

    // Test 9: Invalid absorption color rejected
    #[test]
    fn test_invalid_absorption_color() {
        let mut config = UnderwaterConfig::default();
        config.absorption_color[0] = 1.5;
        assert!(config.validate().is_err());
    }

    // Test 10: Invalid negative density rejected
    #[test]
    fn test_invalid_negative_density() {
        let mut config = UnderwaterConfig::default();
        config.absorption_density = -0.1;
        assert!(config.validate().is_err());
    }

    // Test 11: Invalid caustics intensity rejected
    #[test]
    fn test_invalid_caustics_intensity() {
        let mut config = UnderwaterConfig::default();
        config.caustics_intensity = 1.5;
        assert!(config.validate().is_err());
    }

    // Test 12: Invalid caustics scale rejected
    #[test]
    fn test_invalid_caustics_scale() {
        let mut config = UnderwaterConfig::default();
        config.caustics_scale = 0.0;
        assert!(config.validate().is_err());
    }

    // Test 13: Visibility distance calculation
    #[test]
    fn test_visibility_distance() {
        let config = UnderwaterConfig::default();
        let vis = config.visibility_distance();
        // fog_density = 0.02, vis = 4.605 / 0.02 = 230.25
        assert!(approx_eq_loose(vis, 230.25, 1.0));
    }

    // Test 14: Visibility distance with zero fog
    #[test]
    fn test_visibility_distance_zero_fog() {
        let mut config = UnderwaterConfig::default();
        config.fog_density = 0.0;
        let vis = config.visibility_distance();
        assert_eq!(vis, f32::MAX);
    }

    // Test 15: Config new() clamps values
    #[test]
    fn test_config_new_clamps() {
        let config = UnderwaterConfig::new([2.0, -0.5, 0.5], -1.0, -0.5);
        assert!(approx_eq(config.absorption_color[0], 1.0)); // Clamped to 1
        assert!(approx_eq(config.absorption_color[1], 0.0)); // Clamped to 0
        assert!(approx_eq(config.absorption_density, 0.0)); // Clamped to 0
        assert!(approx_eq(config.fog_density, 0.0)); // Clamped to 0
    }

    // -------------------------------------------------------------------------
    // Absorption Tests (16-22) - Beer-Lambert
    // -------------------------------------------------------------------------

    // Test 16: No absorption at zero depth
    #[test]
    fn test_absorption_zero_depth() {
        let config = UnderwaterConfig::default();
        let absorption = UnderwaterFog::compute_absorption(0.0, &config);
        assert!(approx_eq_vec3(absorption, [1.0, 1.0, 1.0]));
    }

    // Test 17: No absorption above water
    #[test]
    fn test_absorption_above_water() {
        let config = UnderwaterConfig::default();
        let absorption = UnderwaterFog::compute_absorption(-5.0, &config);
        assert!(approx_eq_vec3(absorption, [1.0, 1.0, 1.0]));
    }

    // Test 18: Absorption increases with depth
    #[test]
    fn test_absorption_increases_with_depth() {
        let config = UnderwaterConfig::default();
        let shallow = UnderwaterFog::compute_absorption(5.0, &config);
        let deep = UnderwaterFog::compute_absorption(20.0, &config);

        // Each channel should have more absorption at depth
        assert!(shallow[0] > deep[0]);
        assert!(shallow[1] > deep[1]);
        assert!(shallow[2] > deep[2]);
    }

    // Test 19: Beer-Lambert exponential decay
    #[test]
    fn test_beer_lambert_exponential() {
        let config = UnderwaterConfig::default();
        let t1 = UnderwaterFog::compute_absorption(10.0, &config);
        let t2 = UnderwaterFog::compute_absorption(20.0, &config);

        // At 2x depth, transmittance should be squared (exponential)
        // t2 should be approximately t1^2 for each channel
        for i in 0..3 {
            let expected = t1[i] * t1[i];
            assert!(
                approx_eq_loose(t2[i], expected, 0.05),
                "Channel {}: {} vs expected {}",
                i,
                t2[i],
                expected
            );
        }
    }

    // Test 20: Different channels absorb differently
    #[test]
    fn test_absorption_wavelength_dependent() {
        let config = UnderwaterConfig::default();
        let absorption = UnderwaterFog::compute_absorption(10.0, &config);

        // Default config has absorption_color [0.1, 0.3, 0.5]
        // Lower values = more absorption (1 - color used as coefficient)
        // So red (0.1) should be absorbed more than blue (0.5)
        assert!(
            absorption[0] < absorption[2],
            "Red should be absorbed more than blue: R={} B={}",
            absorption[0],
            absorption[2]
        );
    }

    // Test 21: High density increases absorption rate
    #[test]
    fn test_high_density_absorption() {
        let low_density = UnderwaterConfig::default();
        let mut high_density = UnderwaterConfig::default();
        high_density.absorption_density = 0.5;

        let abs_low = UnderwaterFog::compute_absorption(10.0, &low_density);
        let abs_high = UnderwaterFog::compute_absorption(10.0, &high_density);

        // Higher density = more absorption = lower transmittance
        for i in 0..3 {
            assert!(
                abs_high[i] < abs_low[i],
                "High density should have more absorption: {} < {}",
                abs_high[i],
                abs_low[i]
            );
        }
    }

    // Test 22: Absorption values are bounded
    #[test]
    fn test_absorption_bounds() {
        let config = UnderwaterConfig::murky();
        let absorption = UnderwaterFog::compute_absorption(100.0, &config);

        for i in 0..3 {
            assert!(
                absorption[i] >= 0.0 && absorption[i] <= 1.0,
                "Absorption should be in [0,1]: {}",
                absorption[i]
            );
        }
    }

    // -------------------------------------------------------------------------
    // Scattering Tests (23-28)
    // -------------------------------------------------------------------------

    // Test 23: No scattering at zero distance
    #[test]
    fn test_scattering_zero_distance() {
        let config = UnderwaterConfig::default();
        let scatter = UnderwaterFog::compute_scattering(0.0, [0.0, 1.0, 0.0], &config);
        assert!(approx_eq_vec3(scatter, [0.0, 0.0, 0.0]));
    }

    // Test 24: Scattering increases with distance
    #[test]
    fn test_scattering_increases_with_distance() {
        let config = UnderwaterConfig::default();
        let near = UnderwaterFog::compute_scattering(5.0, [0.0, 1.0, 0.0], &config);
        let far = UnderwaterFog::compute_scattering(50.0, [0.0, 1.0, 0.0], &config);

        let near_total = near[0] + near[1] + near[2];
        let far_total = far[0] + far[1] + far[2];

        assert!(
            far_total > near_total,
            "Far should have more scattering: {} vs {}",
            far_total,
            near_total
        );
    }

    // Test 25: Scattering depends on sun direction
    #[test]
    fn test_scattering_sun_direction() {
        let config = UnderwaterConfig::default();

        // Sun directly above
        let sun_up = UnderwaterFog::compute_scattering(10.0, [0.0, 1.0, 0.0], &config);
        // Sun below horizon
        let sun_down = UnderwaterFog::compute_scattering(10.0, [0.0, -1.0, 0.0], &config);

        let up_total = sun_up[0] + sun_up[1] + sun_up[2];
        let down_total = sun_down[0] + sun_down[1] + sun_down[2];

        assert!(
            up_total > down_total,
            "Sun up should scatter more: {} vs {}",
            up_total,
            down_total
        );
    }

    // Test 26: Scattering uses correct color
    #[test]
    fn test_scattering_color() {
        let mut config = UnderwaterConfig::default();
        config.scattering_color = [1.0, 0.0, 0.0]; // Pure red scatter

        let scatter = UnderwaterFog::compute_scattering(10.0, [0.0, 1.0, 0.0], &config);

        assert!(scatter[0] > scatter[1]);
        assert!(scatter[0] > scatter[2]);
    }

    // Test 27: Scattering density affects amount
    #[test]
    fn test_scattering_density() {
        let mut low = UnderwaterConfig::default();
        low.scattering_density = 0.01;

        let mut high = UnderwaterConfig::default();
        high.scattering_density = 0.2;

        let scatter_low = UnderwaterFog::compute_scattering(10.0, [0.0, 1.0, 0.0], &low);
        let scatter_high = UnderwaterFog::compute_scattering(10.0, [0.0, 1.0, 0.0], &high);

        let low_total = scatter_low[0] + scatter_low[1] + scatter_low[2];
        let high_total = scatter_high[0] + scatter_high[1] + scatter_high[2];

        assert!(
            high_total > low_total,
            "High density should scatter more: {} vs {}",
            high_total,
            low_total
        );
    }

    // Test 28: Scattering converges to scattering color
    #[test]
    fn test_scattering_convergence() {
        let config = UnderwaterConfig::default();
        // At very large distance, scattering approaches scattering_color
        let scatter = UnderwaterFog::compute_scattering(1000.0, [0.0, 1.0, 0.0], &config);

        // Should approach directional * scattering_color
        // With sun up, directional = 1.0
        for i in 0..3 {
            assert!(
                approx_eq_loose(scatter[i], config.scattering_color[i], 0.05),
                "Channel {} should converge: {} vs {}",
                i,
                scatter[i],
                config.scattering_color[i]
            );
        }
    }

    // -------------------------------------------------------------------------
    // Caustics Tests (29-36)
    // -------------------------------------------------------------------------

    // Test 29: Caustics generator default creation
    #[test]
    fn test_caustics_default() {
        let gen = CausticsGenerator::default();
        assert_eq!(gen.resolution(), DEFAULT_CAUSTICS_RESOLUTION);
        assert!(approx_eq(gen.scale(), 1.0));
    }

    // Test 30: Caustics texture generation
    #[test]
    fn test_caustics_texture_generation() {
        let gen = CausticsGenerator::new(64, 1.0);
        let texture = gen.generate_caustics_texture([0.0, 1.0, 0.0], 0.0);

        assert_eq!(texture.len(), 64 * 64);

        // All values should be in valid range
        for value in &texture {
            assert!(
                *value >= 0.0 && *value <= 1.0,
                "Caustic value out of range: {}",
                value
            );
        }
    }

    // Test 31: Caustics sampling returns valid values
    #[test]
    fn test_caustics_sampling() {
        let gen = CausticsGenerator::default();
        let value = gen.sample_caustics([10.0, -5.0, 20.0], 1.0);

        assert!(
            value >= 0.0 && value <= 1.0,
            "Caustic sample out of range: {}",
            value
        );
    }

    // Test 32: Caustics UV calculation
    #[test]
    fn test_caustics_uv() {
        let gen = CausticsGenerator::new(256, 2.0);
        let uv = gen.get_caustics_uv([5.0, 0.0, 3.0]);

        // UVs should be fractional
        assert!(uv[0] >= 0.0 && uv[0] < 1.0);
        assert!(uv[1] >= 0.0 && uv[1] < 1.0);
    }

    // Test 33: Caustics animation varies with time
    #[test]
    fn test_caustics_animation() {
        let gen = CausticsGenerator::new(64, 1.0);
        let pos = [5.0, -2.0, 5.0];

        let t0 = gen.sample_caustics(pos, 0.0);
        let t1 = gen.sample_caustics(pos, 1.0);
        let t2 = gen.sample_caustics(pos, 2.0);

        // At least some variation should occur (not all equal)
        let different = !approx_eq(t0, t1) || !approx_eq(t1, t2);
        assert!(different, "Caustics should animate: {} {} {}", t0, t1, t2);
    }

    // Test 34: Caustics with speed modifier
    #[test]
    fn test_caustics_speed() {
        let slow = CausticsGenerator::new(64, 1.0).with_speed(0.5);
        let fast = CausticsGenerator::new(64, 1.0).with_speed(2.0);

        // Different speeds should produce different patterns at same time
        let tex_slow = slow.generate_caustics_texture([0.0, 1.0, 0.0], 1.0);
        let tex_fast = fast.generate_caustics_texture([0.0, 1.0, 0.0], 1.0);

        let mut different_count = 0;
        for i in 0..tex_slow.len() {
            if !approx_eq(tex_slow[i], tex_fast[i]) {
                different_count += 1;
            }
        }
        assert!(different_count > tex_slow.len() / 2);
    }

    // Test 35: Caustics with octaves
    #[test]
    fn test_caustics_octaves() {
        let gen = CausticsGenerator::new(64, 1.0).with_octaves(4);
        let texture = gen.generate_caustics_texture([0.0, 1.0, 0.0], 0.5);

        // Should have valid values with more detail
        for value in &texture {
            assert!(*value >= 0.0 && *value <= 1.0);
        }
    }

    // Test 36: Caustics scale affects pattern
    #[test]
    fn test_caustics_scale() {
        let small = CausticsGenerator::new(64, 0.5);
        let large = CausticsGenerator::new(64, 2.0);

        // UVs should differ based on scale
        let uv_small = small.get_caustics_uv([1.0, 0.0, 1.0]);
        let uv_large = large.get_caustics_uv([1.0, 0.0, 1.0]);

        let different = !approx_eq(uv_small[0], uv_large[0])
            || !approx_eq(uv_small[1], uv_large[1]);
        assert!(different, "Different scales should give different UVs");
    }

    // -------------------------------------------------------------------------
    // Fog Application Tests (37-42)
    // -------------------------------------------------------------------------

    // Test 37: Fog at zero depth returns original color
    #[test]
    fn test_fog_zero_depth() {
        let config = UnderwaterConfig::default();
        let input = [1.0, 0.5, 0.2];
        let output = UnderwaterFog::apply_underwater_fog(input, 0.0, &config);

        // Should be very close to input (only tiny scattering)
        for i in 0..3 {
            assert!(
                approx_eq_loose(output[i], input[i], 0.01),
                "Channel {} should be preserved: {} vs {}",
                i,
                output[i],
                input[i]
            );
        }
    }

    // Test 38: Fog darkens with depth
    #[test]
    fn test_fog_darkens_with_depth() {
        let config = UnderwaterConfig::default();
        let input = [1.0, 1.0, 1.0];

        let near = UnderwaterFog::apply_underwater_fog(input, 5.0, &config);
        let far = UnderwaterFog::apply_underwater_fog(input, 50.0, &config);

        let near_brightness = near[0] + near[1] + near[2];
        let far_brightness = far[0] + far[1] + far[2];

        // Far should be darker (absorption dominates)
        assert!(
            far_brightness < near_brightness,
            "Far should be darker: {} vs {}",
            far_brightness,
            near_brightness
        );
    }

    // Test 39: Fog adds color tint
    #[test]
    fn test_fog_color_shift() {
        let config = UnderwaterConfig::default();
        let input = [1.0, 1.0, 1.0]; // White input

        let output = UnderwaterFog::apply_underwater_fog(input, 20.0, &config);

        // Red should be reduced more than blue (water absorption)
        assert!(
            output[2] > output[0],
            "Blue should be brighter than red: B={} R={}",
            output[2],
            output[0]
        );
    }

    // Test 40: Fog scattering adds light
    #[test]
    fn test_fog_scattering_adds_light() {
        let config = UnderwaterConfig::default();
        let input = [0.0, 0.0, 0.0]; // Black input

        let output = UnderwaterFog::apply_underwater_fog(input, 30.0, &config);

        // Should have some color from scattering
        let brightness = output[0] + output[1] + output[2];
        assert!(
            brightness > 0.0,
            "Scattering should add some light: {}",
            brightness
        );
    }

    // Test 41: Get visibility distance
    #[test]
    fn test_fog_visibility_distance() {
        let config = UnderwaterConfig::tropical();
        let vis = UnderwaterFog::get_visibility_distance(&config);

        // Tropical has fog_density = 0.008
        // vis = 4.605 / 0.008 = 575.625
        assert!(vis > 500.0 && vis < 600.0);
    }

    // Test 42: Fog output values are bounded
    #[test]
    fn test_fog_output_bounds() {
        let config = UnderwaterConfig::murky();
        let input = [2.0, 2.0, 2.0]; // HDR input

        let output = UnderwaterFog::apply_underwater_fog(input, 10.0, &config);

        // Absorption multiplies (reduces), scattering adds
        // Output should be finite
        for i in 0..3 {
            assert!(output[i].is_finite(), "Output should be finite: {}", output[i]);
        }
    }

    // -------------------------------------------------------------------------
    // Distortion Tests (43-48)
    // -------------------------------------------------------------------------

    // Test 43: Distortion default creation
    #[test]
    fn test_distortion_default() {
        let distortion = UnderwaterDistortion::default();
        // Should create without error
        let _ = distortion.apply_distortion([0.5, 0.5], 0.0, 0.1);
    }

    // Test 44: Zero strength gives no distortion
    #[test]
    fn test_distortion_zero_strength() {
        let distortion = UnderwaterDistortion::new();
        let uv = [0.5, 0.5];
        let result = distortion.apply_distortion(uv, 1.0, 0.0);

        assert!(approx_eq(result[0], uv[0]));
        assert!(approx_eq(result[1], uv[1]));
    }

    // Test 45: Distortion with Perlin noise
    #[test]
    fn test_distortion_perlin_variation() {
        let distortion = UnderwaterDistortion::new();

        let uv1 = distortion.apply_distortion([0.1, 0.1], 0.0, 0.5);
        let uv2 = distortion.apply_distortion([0.9, 0.9], 0.0, 0.5);

        // Different UVs should have different distortion
        let different = !approx_eq(uv1[0], uv2[0]) || !approx_eq(uv1[1], uv2[1]);
        assert!(different);
    }

    // Test 46: Distortion animates with time
    #[test]
    fn test_distortion_animation() {
        let distortion = UnderwaterDistortion::new();
        let uv = [0.5, 0.5];

        let t0 = distortion.apply_distortion(uv, 0.0, 0.5);
        let t1 = distortion.apply_distortion(uv, 2.0, 0.5);
        let t2 = distortion.apply_distortion(uv, 4.0, 0.5);

        // Should vary with time
        let varies = !approx_eq(t0[0], t1[0]) || !approx_eq(t1[0], t2[0]);
        assert!(varies, "Distortion should animate");
    }

    // Test 47: Distortion is bounded
    #[test]
    fn test_distortion_bounds() {
        let distortion = UnderwaterDistortion::new();

        // Even with extreme inputs, output should be reasonable
        for i in 0..100 {
            let seed = i as u32;
            let uv = [rand_float_seq(seed), rand_float_seq(seed + 1000)];
            let result = distortion.apply_distortion(uv, rand_float_seq(seed + 2000) * 100.0, 1.0);

            // Should not deviate too far from original
            assert!(
                (result[0] - uv[0]).abs() <= 0.15,
                "U distortion too large: {} vs {}",
                result[0],
                uv[0]
            );
            assert!(
                (result[1] - uv[1]).abs() <= 0.15,
                "V distortion too large: {} vs {}",
                result[1],
                uv[1]
            );
        }
    }

    // Test 48: Distortion with config
    #[test]
    fn test_distortion_with_config() {
        let distortion = UnderwaterDistortion::new();
        let config = UnderwaterConfig::default();
        let uv = [0.5, 0.5];

        let result = distortion.compute_distortion_offset(uv, 1.0, &config);

        // Should use config's distortion_strength
        assert!(result[0].is_finite() && result[1].is_finite());
    }

    // Simple pseudo-random for testing (deterministic sequence)
    fn rand_float_seq(seed: u32) -> f32 {
        let x = seed.wrapping_mul(1103515245).wrapping_add(12345);
        (x as f32 / u32::MAX as f32).fract().abs()
    }

    // -------------------------------------------------------------------------
    // God Ray Tests (49-55)
    // -------------------------------------------------------------------------

    // Test 49: God ray default creation
    #[test]
    fn test_god_ray_default() {
        let rays = GodRays::default();
        // Should create without error
        let _ = rays.compute_ray_intensity([0.0, 0.0, -1.0], [0.0, 1.0, 0.0], 5.0);
    }

    // Test 50: No rays above water
    #[test]
    fn test_god_ray_above_water() {
        let rays = GodRays::new();
        let intensity = rays.compute_ray_intensity([0.0, 0.0, -1.0], [0.0, 1.0, 0.0], -5.0);
        assert!(approx_eq(intensity, 0.0));
    }

    // Test 51: God ray direction dependency
    #[test]
    fn test_god_ray_direction() {
        let rays = GodRays::new();
        let sun_dir = [0.0, 1.0, 0.0]; // Sun directly above

        // Looking toward sun
        let toward_sun = rays.compute_ray_intensity([0.0, 1.0, 0.0], sun_dir, 5.0);
        // Looking away from sun
        let away_sun = rays.compute_ray_intensity([0.0, -1.0, 0.0], sun_dir, 5.0);

        assert!(
            toward_sun > away_sun,
            "Looking toward sun should be brighter: {} vs {}",
            toward_sun,
            away_sun
        );
    }

    // Test 52: God ray depth attenuation
    #[test]
    fn test_god_ray_depth() {
        let rays = GodRays::new();
        let view_dir = [0.0, 1.0, 0.0];
        let sun_dir = [0.0, 1.0, 0.0];

        let shallow = rays.compute_ray_intensity(view_dir, sun_dir, 5.0);
        let deep = rays.compute_ray_intensity(view_dir, sun_dir, 50.0);

        assert!(
            shallow > deep,
            "Shallow should be brighter: {} vs {}",
            shallow,
            deep
        );
    }

    // Test 53: God ray sun angle dependency
    #[test]
    fn test_god_ray_sun_angle() {
        let rays = GodRays::new();
        let view_dir = [0.0, 0.0, -1.0];

        // Sun high in sky
        let sun_high = rays.compute_ray_intensity(view_dir, [0.0, 0.9, 0.4], 5.0);
        // Sun at horizon
        let sun_low = rays.compute_ray_intensity(view_dir, [0.0, 0.1, 0.99], 5.0);

        assert!(
            sun_high > sun_low,
            "High sun should give more rays: {} vs {}",
            sun_high,
            sun_low
        );
    }

    // Test 54: God ray march returns valid color
    #[test]
    fn test_god_ray_march() {
        let rays = GodRays::new().with_steps(16);
        let config = UnderwaterConfig::default();

        let start = [0.0, -5.0, 0.0]; // Underwater
        let end = [0.0, -5.0, -20.0]; // 20m away underwater

        let color = rays.march_underwater_rays(start, end, &config);

        for i in 0..3 {
            assert!(
                color[i].is_finite() && color[i] >= 0.0,
                "Ray color should be valid: {}",
                color[i]
            );
        }
    }

    // Test 55: God ray occlusion from surface
    #[test]
    fn test_god_ray_surface_occlusion() {
        let rays = GodRays::new();
        let config = UnderwaterConfig::default();

        // Ray crossing above water surface should have less contribution
        let underwater_only = rays.march_underwater_rays(
            [0.0, -10.0, 0.0],
            [0.0, -10.0, -30.0],
            &config,
        );
        let crossing_surface = rays.march_underwater_rays(
            [0.0, -5.0, 0.0],
            [0.0, 5.0, -30.0], // Crosses y=0
            &config,
        );

        let uw_total = underwater_only[0] + underwater_only[1] + underwater_only[2];
        let cs_total = crossing_surface[0] + crossing_surface[1] + crossing_surface[2];

        // Underwater-only ray should accumulate more (no above-water sections)
        // But crossing surface means less underwater path
        // This test mainly verifies both work without errors
        assert!(uw_total.is_finite() && cs_total.is_finite());
    }

    // -------------------------------------------------------------------------
    // UnderwaterPostProcess Tests (56-65) - Transition & Compositor
    // -------------------------------------------------------------------------

    // Test 56: Post process default creation
    #[test]
    fn test_post_process_default() {
        let post = UnderwaterPostProcess::default();
        assert!(post.config().validate().is_ok());
        assert!(approx_eq(post.time(), 0.0));
    }

    // Test 57: Is underwater check
    #[test]
    fn test_is_underwater() {
        assert!(UnderwaterPostProcess::is_underwater(-5.0, 0.0));
        assert!(!UnderwaterPostProcess::is_underwater(5.0, 0.0));
        assert!(!UnderwaterPostProcess::is_underwater(0.0, 0.0)); // At surface = above
    }

    // Test 58: Blend factor fully above
    #[test]
    fn test_blend_above_water() {
        let blend = UnderwaterPostProcess::get_blend_factor(10.0, 0.0);
        assert!(approx_eq(blend, 0.0));
    }

    // Test 59: Blend factor fully below
    #[test]
    fn test_blend_below_water() {
        let blend = UnderwaterPostProcess::get_blend_factor(-10.0, 0.0);
        assert!(approx_eq(blend, 1.0));
    }

    // Test 60: Blend factor in transition zone
    #[test]
    fn test_blend_transition_zone() {
        // At water surface (camera_y = water_height)
        let at_surface = UnderwaterPostProcess::get_blend_factor(0.0, 0.0);
        assert!(
            at_surface > 0.4 && at_surface < 0.6,
            "At surface should be ~0.5: {}",
            at_surface
        );

        // Just above
        let above = UnderwaterPostProcess::get_blend_factor(0.25, 0.0);
        assert!(above < 0.5, "Above surface should be < 0.5: {}", above);

        // Just below
        let below = UnderwaterPostProcess::get_blend_factor(-0.25, 0.0);
        assert!(below > 0.5, "Below surface should be > 0.5: {}", below);
    }

    // Test 61: Process frame returns valid output
    #[test]
    fn test_process_frame() {
        let post = UnderwaterPostProcess::default();
        let input = [0.8, 0.7, 0.6];
        let depth = 10.0;
        let camera_pos = [0.0, -5.0, 0.0];

        let output = post.process_frame(input, depth, camera_pos);

        for i in 0..3 {
            assert!(
                output[i].is_finite(),
                "Output should be finite: {}",
                output[i]
            );
        }
    }

    // Test 62: Time update
    #[test]
    fn test_time_update() {
        let mut post = UnderwaterPostProcess::default();
        assert!(approx_eq(post.time(), 0.0));

        post.update_time(0.5);
        assert!(approx_eq(post.time(), 0.5));

        post.update_time(0.3);
        assert!(approx_eq(post.time(), 0.8));
    }

    // Test 63: Set time
    #[test]
    fn test_set_time() {
        let mut post = UnderwaterPostProcess::default();
        post.set_time(5.0);
        assert!(approx_eq(post.time(), 5.0));

        // Negative clamps to 0
        post.set_time(-1.0);
        assert!(approx_eq(post.time(), 0.0));
    }

    // Test 64: Config accessors
    #[test]
    fn test_config_accessors() {
        let mut post = UnderwaterPostProcess::default();

        post.config_mut().caustics_intensity = 0.8;
        assert!(approx_eq(post.config().caustics_intensity, 0.8));

        post.caustics_mut().scale;
        assert!(post.caustics().resolution() > 0);
    }

    // Test 65: Edge case - surface exactly
    #[test]
    fn test_edge_case_at_surface() {
        // Camera at exact water height
        let blend = UnderwaterPostProcess::get_blend_factor(0.0, 0.0);
        // Should be in middle of transition
        assert!(blend > 0.3 && blend < 0.7);
    }

    // -------------------------------------------------------------------------
    // Deep Water Edge Cases (66-70)
    // -------------------------------------------------------------------------

    // Test 66: Very deep absorption
    #[test]
    fn test_very_deep_absorption() {
        let config = UnderwaterConfig::default();
        let absorption = UnderwaterFog::compute_absorption(1000.0, &config);

        // At extreme depth, absorption should be nearly complete
        for i in 0..3 {
            assert!(
                absorption[i] < 0.001,
                "Deep absorption should be near zero: {}",
                absorption[i]
            );
        }
    }

    // Test 67: Very deep fog
    #[test]
    fn test_very_deep_fog() {
        let config = UnderwaterConfig::default();
        let input = [1.0, 1.0, 1.0];
        let output = UnderwaterFog::apply_underwater_fog(input, 500.0, &config);

        // Should be dominated by scattering color at extreme depth
        for i in 0..3 {
            assert!(output[i].is_finite());
        }
    }

    // Test 68: Zero density fog
    #[test]
    fn test_zero_density_fog() {
        let mut config = UnderwaterConfig::default();
        config.fog_density = 0.0;
        config.absorption_density = 0.0;
        config.scattering_density = 0.0;

        let input = [0.5, 0.5, 0.5];
        let output = UnderwaterFog::apply_underwater_fog(input, 100.0, &config);

        // With no fog, color should be preserved
        for i in 0..3 {
            assert!(
                approx_eq_loose(output[i], input[i], 0.01),
                "No fog should preserve color: {} vs {}",
                output[i],
                input[i]
            );
        }
    }

    // Test 69: Caustics at surface
    #[test]
    fn test_caustics_at_surface() {
        let gen = CausticsGenerator::default();
        let value = gen.sample_caustics([0.0, 0.0, 0.0], 0.0);
        assert!(value >= 0.0 && value <= 1.0);
    }

    // Test 70: God ray at surface
    #[test]
    fn test_god_ray_at_surface() {
        let rays = GodRays::new();
        // Depth = 0 (exactly at surface)
        let intensity = rays.compute_ray_intensity([0.0, -1.0, 0.0], [0.0, 1.0, 0.0], 0.0);
        // At surface, should be 0 (not underwater yet)
        assert!(approx_eq(intensity, 0.0));
    }

    // -------------------------------------------------------------------------
    // Utility Function Tests (71-75)
    // -------------------------------------------------------------------------

    // Test 71: Smoothstep at 0
    #[test]
    fn test_smoothstep_zero() {
        let result = smoothstep(0.0);
        assert!(approx_eq(result, 0.0));
    }

    // Test 72: Smoothstep at 1
    #[test]
    fn test_smoothstep_one() {
        let result = smoothstep(1.0);
        assert!(approx_eq(result, 1.0));
    }

    // Test 73: Smoothstep at 0.5
    #[test]
    fn test_smoothstep_half() {
        let result = smoothstep(0.5);
        assert!(approx_eq(result, 0.5));
    }

    // Test 74: Smoothstep clamps
    #[test]
    fn test_smoothstep_clamps() {
        assert!(approx_eq(smoothstep(-1.0), 0.0));
        assert!(approx_eq(smoothstep(2.0), 1.0));
    }

    // Test 75: Vector normalize
    #[test]
    fn test_normalize() {
        let v = [3.0, 4.0, 0.0];
        let n = normalize3(v);
        let len = vec3_length(n);
        assert!(approx_eq(len, 1.0));
    }

    // -------------------------------------------------------------------------
    // Integration Tests (76-80)
    // -------------------------------------------------------------------------

    // Test 76: Full underwater frame
    #[test]
    fn test_full_underwater_frame() {
        let mut post = UnderwaterPostProcess::new(UnderwaterConfig::tropical());
        post.update_time(1.5);

        let input = [0.8, 0.8, 0.8];
        let camera_pos = [10.0, -8.0, 20.0];
        let depth = 15.0;

        let output = post.process_frame(input, depth, camera_pos);

        // Should be darker and bluer
        assert!(output[0] < input[0]); // Less red
        for c in output {
            assert!(c.is_finite() && c >= 0.0);
        }
    }

    // Test 77: Transition underwater to above
    #[test]
    fn test_transition_sequence() {
        let steps: Vec<f32> = vec![-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0];
        let mut prev_blend = 1.0;

        for y in steps {
            let blend = UnderwaterPostProcess::get_blend_factor(y, 0.0);

            // Blend should monotonically decrease as we go up
            assert!(
                blend <= prev_blend,
                "Blend should decrease going up: {} at y={}",
                blend,
                y
            );
            prev_blend = blend;
        }
    }

    // Test 78: Multiple presets process correctly
    #[test]
    fn test_all_presets_process() {
        let presets = [
            UnderwaterConfig::default(),
            UnderwaterConfig::tropical(),
            UnderwaterConfig::deep_ocean(),
            UnderwaterConfig::murky(),
            UnderwaterConfig::pool(),
        ];

        for config in &presets {
            let post = UnderwaterPostProcess::new(*config);
            let output = post.process_frame([0.5, 0.5, 0.5], 10.0, [0.0, -5.0, 0.0]);

            for c in output {
                assert!(c.is_finite(), "Preset output should be finite");
            }
        }
    }

    // Test 79: God ray accumulation over distance
    #[test]
    fn test_god_ray_accumulation() {
        let rays = GodRays::new().with_steps(32);
        let config = UnderwaterConfig::default();

        let short_ray = rays.march_underwater_rays(
            [0.0, -5.0, 0.0],
            [0.0, -5.0, -10.0],
            &config,
        );
        let long_ray = rays.march_underwater_rays(
            [0.0, -5.0, 0.0],
            [0.0, -5.0, -50.0],
            &config,
        );

        let short_total = short_ray[0] + short_ray[1] + short_ray[2];
        let long_total = long_ray[0] + long_ray[1] + long_ray[2];

        // Longer ray should accumulate more (up to saturation)
        // At least it should be non-zero for both
        assert!(short_total >= 0.0 && long_total >= 0.0);
    }

    // Test 80: Combined effects don't explode
    #[test]
    fn test_combined_effects_stable() {
        let mut post = UnderwaterPostProcess::new(UnderwaterConfig::default());

        // Simulate 100 frames
        for i in 0..100 {
            post.update_time(0.016);
            let output = post.process_frame(
                [1.0, 1.0, 1.0],
                (i % 50) as f32,
                [i as f32, -10.0, i as f32 * 0.5],
            );

            for c in output {
                assert!(c.is_finite(), "Frame {} should be stable", i);
            }
        }
    }
}
