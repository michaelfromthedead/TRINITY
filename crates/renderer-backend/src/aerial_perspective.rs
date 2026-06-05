//! Aerial Perspective Rendering Integration
//!
//! This module provides aerial perspective (atmospheric haze) rendering for
//! terrain and distant objects. It implements the Bruneton approach to aerial
//! perspective using 3D LUTs for efficient real-time sampling.
//!
//! # Overview
//!
//! Aerial perspective simulates how the atmosphere affects distant objects:
//! - **Inscatter**: Light scattered into the view ray (additive, bluish)
//! - **Transmittance**: Light absorbed/scattered out (multiplicative, fade)
//!
//! As distance increases:
//! - Objects appear hazier and shift toward the sky color
//! - Contrast decreases due to inscattered light
//! - Colors fade toward the horizon color
//!
//! # Integration with Terrain
//!
//! The module provides specific helpers for terrain clipmap integration:
//! - Per-clipmap-level visibility adjustments
//! - Height-based density falloff
//! - Distance fog fallback for mobile/low quality
//!
//! # GPU Integration
//!
//! All configuration structs are `repr(C)` with `bytemuck::Pod` for direct
//! GPU upload. The 3D LUT is designed for hardware texture interpolation.
//!
//! # Physical Model
//!
//! Transmittance follows Beer-Lambert law:
//! ```text
//! T(d) = exp(-integral[0..d] sigma(s) ds)
//! ```
//!
//! Inscattered light accumulates along the ray:
//! ```text
//! L_in(d) = integral[0..d] T(s) * scatter(s) * phase(angle) ds
//! ```

use bytemuck::{Pod, Zeroable};
use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default maximum aerial perspective distance in meters (100km).
pub const DEFAULT_MAX_DISTANCE: f32 = 100_000.0;

/// Default density scale factor.
pub const DEFAULT_DENSITY_SCALE: f32 = 1.0;

/// Default height falloff rate (exponential decay coefficient).
pub const DEFAULT_HEIGHT_FALLOFF: f32 = 0.00012;

/// Default inscatter intensity multiplier.
pub const DEFAULT_INSCATTER_INTENSITY: f32 = 1.0;

/// Default blend start distance in meters (500m).
pub const DEFAULT_BLEND_START: f32 = 500.0;

/// Default blend range in meters (500m).
pub const DEFAULT_BLEND_RANGE: f32 = 500.0;

/// Minimum valid density scale.
pub const MIN_DENSITY_SCALE: f32 = 0.0;

/// Maximum valid density scale.
pub const MAX_DENSITY_SCALE: f32 = 10.0;

/// Minimum valid height falloff.
pub const MIN_HEIGHT_FALLOFF: f32 = 0.0;

/// Maximum valid height falloff.
pub const MAX_HEIGHT_FALLOFF: f32 = 1.0;

/// Default 3D LUT width (view direction samples).
pub const DEFAULT_LUT_WIDTH: u32 = 32;

/// Default 3D LUT height (sun angle samples).
pub const DEFAULT_LUT_HEIGHT: u32 = 32;

/// Default 3D LUT depth (distance samples).
pub const DEFAULT_LUT_DEPTH: u32 = 32;

/// Number of integration steps for inscatter accumulation.
pub const DEFAULT_INSCATTER_STEPS: u32 = 16;

/// Rayleigh scattering coefficients at sea level (RGB, per meter).
pub const RAYLEIGH_SCATTERING: [f32; 3] = [5.5e-6, 13.0e-6, 22.4e-6];

/// Mie scattering coefficient at sea level (per meter).
pub const MIE_SCATTERING: f32 = 21e-6;

/// Rayleigh scale height in meters.
pub const RAYLEIGH_SCALE_HEIGHT: f32 = 8000.0;

/// Mie scale height in meters.
pub const MIE_SCALE_HEIGHT: f32 = 1200.0;

/// Mie asymmetry parameter (Henyey-Greenstein g).
pub const MIE_ASYMMETRY_G: f32 = 0.8;

/// Small epsilon for numerical stability.
pub const EPSILON: f32 = 1e-6;

// ---------------------------------------------------------------------------
// AerialPerspectiveConfig
// ---------------------------------------------------------------------------

/// Configuration for aerial perspective rendering.
///
/// This struct is designed to be uploaded to the GPU as a uniform buffer.
/// The layout is `repr(C)` and implements `Pod` for bytemuck compatibility.
///
/// # Memory Layout (32 bytes)
///
/// | Offset | Field              | Size    |
/// |--------|--------------------|---------|
/// | 0      | max_distance       | 4 bytes |
/// | 4      | density_scale      | 4 bytes |
/// | 8      | height_falloff     | 4 bytes |
/// | 12     | inscatter_intensity| 4 bytes |
/// | 16     | blend_start        | 4 bytes |
/// | 20     | blend_range        | 4 bytes |
/// | 24     | _padding           | 8 bytes |
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct AerialPerspectiveConfig {
    /// Maximum distance for aerial perspective in meters.
    ///
    /// Beyond this distance, objects are fully blended with horizon color.
    /// Default: 100,000m (100km).
    pub max_distance: f32,

    /// Density scale factor (multiplier for scattering coefficients).
    ///
    /// Higher values = more haze. Range: 0.0 to 10.0.
    /// Default: 1.0.
    pub density_scale: f32,

    /// Exponential height falloff rate.
    ///
    /// Controls how quickly density decreases with altitude.
    /// Higher values = faster falloff (clearer at altitude).
    /// Default: 0.00012 (roughly 8km scale height).
    pub height_falloff: f32,

    /// Inscatter intensity multiplier.
    ///
    /// Scales the amount of light scattered into the view ray.
    /// Default: 1.0.
    pub inscatter_intensity: f32,

    /// Distance where aerial perspective blending begins (meters).
    ///
    /// Objects closer than this receive no aerial perspective effect.
    /// Default: 500m.
    pub blend_start: f32,

    /// Distance range over which blending occurs (meters).
    ///
    /// Aerial effect ramps from 0 to 1 over this range.
    /// Default: 500m.
    pub blend_range: f32,

    /// Padding for GPU alignment (vec4 boundary).
    pub _padding: [u32; 2],
}

impl Default for AerialPerspectiveConfig {
    fn default() -> Self {
        Self {
            max_distance: DEFAULT_MAX_DISTANCE,
            density_scale: DEFAULT_DENSITY_SCALE,
            height_falloff: DEFAULT_HEIGHT_FALLOFF,
            inscatter_intensity: DEFAULT_INSCATTER_INTENSITY,
            blend_start: DEFAULT_BLEND_START,
            blend_range: DEFAULT_BLEND_RANGE,
            _padding: [0; 2],
        }
    }
}

impl AerialPerspectiveConfig {
    /// Create a new aerial perspective configuration with validation.
    ///
    /// # Arguments
    ///
    /// * `max_distance` - Maximum aerial perspective distance in meters.
    /// * `density_scale` - Density scale factor.
    /// * `height_falloff` - Exponential height falloff rate.
    ///
    /// # Panics
    ///
    /// Panics if parameters are out of valid ranges.
    pub fn new(max_distance: f32, density_scale: f32, height_falloff: f32) -> Self {
        assert!(
            max_distance > 0.0,
            "max_distance must be positive, got {}",
            max_distance
        );
        assert!(
            density_scale >= MIN_DENSITY_SCALE && density_scale <= MAX_DENSITY_SCALE,
            "density_scale must be in range [{}, {}], got {}",
            MIN_DENSITY_SCALE,
            MAX_DENSITY_SCALE,
            density_scale
        );
        assert!(
            height_falloff >= MIN_HEIGHT_FALLOFF && height_falloff <= MAX_HEIGHT_FALLOFF,
            "height_falloff must be in range [{}, {}], got {}",
            MIN_HEIGHT_FALLOFF,
            MAX_HEIGHT_FALLOFF,
            height_falloff
        );

        Self {
            max_distance,
            density_scale,
            height_falloff,
            inscatter_intensity: DEFAULT_INSCATTER_INTENSITY,
            blend_start: DEFAULT_BLEND_START,
            blend_range: DEFAULT_BLEND_RANGE,
            _padding: [0; 2],
        }
    }

    /// Create a configuration with all parameters specified.
    pub fn with_full_params(
        max_distance: f32,
        density_scale: f32,
        height_falloff: f32,
        inscatter_intensity: f32,
        blend_start: f32,
        blend_range: f32,
    ) -> Self {
        assert!(
            max_distance > 0.0,
            "max_distance must be positive, got {}",
            max_distance
        );
        assert!(
            density_scale >= MIN_DENSITY_SCALE && density_scale <= MAX_DENSITY_SCALE,
            "density_scale must be in range [{}, {}], got {}",
            MIN_DENSITY_SCALE,
            MAX_DENSITY_SCALE,
            density_scale
        );
        assert!(
            inscatter_intensity >= 0.0,
            "inscatter_intensity must be non-negative, got {}",
            inscatter_intensity
        );
        assert!(
            blend_start >= 0.0,
            "blend_start must be non-negative, got {}",
            blend_start
        );
        assert!(
            blend_range > 0.0,
            "blend_range must be positive, got {}",
            blend_range
        );

        Self {
            max_distance,
            density_scale,
            height_falloff,
            inscatter_intensity,
            blend_start,
            blend_range,
            _padding: [0; 2],
        }
    }

    /// Builder: Set inscatter intensity.
    #[inline]
    pub fn with_inscatter_intensity(mut self, intensity: f32) -> Self {
        self.inscatter_intensity = intensity.max(0.0);
        self
    }

    /// Builder: Set blend parameters.
    #[inline]
    pub fn with_blend(mut self, start: f32, range: f32) -> Self {
        self.blend_start = start.max(0.0);
        self.blend_range = range.max(EPSILON);
        self
    }

    /// Calculate the blend factor for a given distance.
    ///
    /// Returns 0.0 for distances <= blend_start,
    /// 1.0 for distances >= blend_start + blend_range,
    /// and a smooth interpolation in between.
    #[inline]
    pub fn blend_factor(&self, distance: f32) -> f32 {
        if distance <= self.blend_start {
            0.0
        } else if distance >= self.blend_start + self.blend_range {
            1.0
        } else {
            let t = (distance - self.blend_start) / self.blend_range;
            // Smoothstep for smooth transition
            t * t * (3.0 - 2.0 * t)
        }
    }

    /// Calculate the normalized distance (0 to 1) for LUT sampling.
    #[inline]
    pub fn normalized_distance(&self, distance: f32) -> f32 {
        (distance / self.max_distance).clamp(0.0, 1.0)
    }

    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<(), AerialPerspectiveError> {
        if self.max_distance <= 0.0 {
            return Err(AerialPerspectiveError::InvalidParameter(
                "max_distance must be positive".into(),
            ));
        }
        if self.density_scale < MIN_DENSITY_SCALE || self.density_scale > MAX_DENSITY_SCALE {
            return Err(AerialPerspectiveError::InvalidParameter(format!(
                "density_scale {} out of range [{}, {}]",
                self.density_scale, MIN_DENSITY_SCALE, MAX_DENSITY_SCALE
            )));
        }
        if self.blend_range <= 0.0 {
            return Err(AerialPerspectiveError::InvalidParameter(
                "blend_range must be positive".into(),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// AerialSample
// ---------------------------------------------------------------------------

/// Sample result from aerial perspective evaluation.
///
/// Contains the accumulated inscatter and transmittance for a point.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable, Default)]
pub struct AerialSample {
    /// Accumulated inscattered light (RGB).
    ///
    /// This is additive light that should be added to the final color.
    pub inscatter: [f32; 3],

    /// Transmittance (0 to 1).
    ///
    /// Fraction of original light that survives the path.
    /// 1.0 = no extinction, 0.0 = fully absorbed.
    pub transmittance: f32,

    /// Optical depth along the path.
    ///
    /// Accumulated extinction: transmittance = exp(-optical_depth).
    pub optical_depth: f32,

    /// Padding for alignment.
    pub _padding: [u32; 3],
}

impl AerialSample {
    /// Create a new aerial sample.
    pub fn new(inscatter: [f32; 3], transmittance: f32, optical_depth: f32) -> Self {
        Self {
            inscatter,
            transmittance,
            optical_depth,
            _padding: [0; 3],
        }
    }

    /// Create an identity sample (no aerial effect).
    #[inline]
    pub fn identity() -> Self {
        Self {
            inscatter: [0.0, 0.0, 0.0],
            transmittance: 1.0,
            optical_depth: 0.0,
            _padding: [0; 3],
        }
    }

    /// Create a fully fogged sample (maximum aerial effect).
    #[inline]
    pub fn fully_fogged(fog_color: [f32; 3]) -> Self {
        Self {
            inscatter: fog_color,
            transmittance: 0.0,
            optical_depth: f32::MAX,
            _padding: [0; 3],
        }
    }

    /// Linearly interpolate between two samples.
    #[inline]
    pub fn lerp(a: &Self, b: &Self, t: f32) -> Self {
        let t_clamped = t.clamp(0.0, 1.0);
        let inv_t = 1.0 - t_clamped;
        Self {
            inscatter: [
                a.inscatter[0] * inv_t + b.inscatter[0] * t_clamped,
                a.inscatter[1] * inv_t + b.inscatter[1] * t_clamped,
                a.inscatter[2] * inv_t + b.inscatter[2] * t_clamped,
            ],
            transmittance: a.transmittance * inv_t + b.transmittance * t_clamped,
            optical_depth: a.optical_depth * inv_t + b.optical_depth * t_clamped,
            _padding: [0; 3],
        }
    }

    /// Combine this sample with another (for ray marching accumulation).
    ///
    /// Uses the over operator: T_new = T1 * T2, L_new = L1 + T1 * L2
    #[inline]
    pub fn combine(&self, other: &Self) -> Self {
        Self {
            inscatter: [
                self.inscatter[0] + self.transmittance * other.inscatter[0],
                self.inscatter[1] + self.transmittance * other.inscatter[1],
                self.inscatter[2] + self.transmittance * other.inscatter[2],
            ],
            transmittance: self.transmittance * other.transmittance,
            optical_depth: self.optical_depth + other.optical_depth,
            _padding: [0; 3],
        }
    }

    /// Check if this sample represents clear air (no fog effect).
    #[inline]
    pub fn is_clear(&self) -> bool {
        self.transmittance > 0.999
            && self.inscatter[0].abs() < EPSILON
            && self.inscatter[1].abs() < EPSILON
            && self.inscatter[2].abs() < EPSILON
    }

    /// Check if this sample represents fully fogged conditions.
    #[inline]
    pub fn is_fully_fogged(&self) -> bool {
        self.transmittance < 0.001
    }
}

// ---------------------------------------------------------------------------
// AerialPerspectiveLUT
// ---------------------------------------------------------------------------

/// 3D lookup table for aerial perspective.
///
/// The LUT is parameterized by:
/// - U: View-sun angle (horizontal dimension)
/// - V: Height/altitude (vertical dimension)
/// - W: Distance (depth dimension, exponentially mapped)
///
/// Each texel stores RGBA: RGB = inscatter, A = transmittance.
#[derive(Debug, Clone)]
pub struct AerialPerspectiveLUT {
    /// LUT data (RGBA per texel).
    pub data: Vec<f32>,

    /// Width of the LUT (view angle samples).
    pub width: u32,

    /// Height of the LUT (altitude samples).
    pub height: u32,

    /// Depth of the LUT (distance samples).
    pub depth: u32,

    /// Maximum distance represented in the LUT.
    pub max_distance: f32,

    /// Exponential mapping base for distance.
    /// depth_index = log(1 + distance * exp_base) / log(1 + max_distance * exp_base)
    pub exp_base: f32,
}

impl AerialPerspectiveLUT {
    /// Create a new aerial perspective LUT.
    ///
    /// # Arguments
    ///
    /// * `width` - Number of view angle samples.
    /// * `height` - Number of altitude samples.
    /// * `depth` - Number of distance samples.
    pub fn new(width: u32, height: u32, depth: u32) -> Self {
        let size = (width * height * depth * 4) as usize;
        Self {
            data: vec![0.0; size],
            width,
            height,
            depth,
            max_distance: DEFAULT_MAX_DISTANCE,
            exp_base: 0.001,
        }
    }

    /// Create a LUT with default dimensions.
    pub fn with_defaults() -> Self {
        Self::new(DEFAULT_LUT_WIDTH, DEFAULT_LUT_HEIGHT, DEFAULT_LUT_DEPTH)
    }

    /// Create a LUT with custom max distance.
    pub fn with_max_distance(width: u32, height: u32, depth: u32, max_distance: f32) -> Self {
        let mut lut = Self::new(width, height, depth);
        lut.max_distance = max_distance.max(EPSILON);
        lut
    }

    /// Get the total number of texels.
    #[inline]
    pub fn texel_count(&self) -> usize {
        (self.width * self.height * self.depth) as usize
    }

    /// Get the total data size in bytes.
    #[inline]
    pub fn byte_size(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }

    /// Convert depth index to world distance (exponential mapping).
    ///
    /// # Arguments
    ///
    /// * `depth_index` - Depth slice index (0 to depth-1).
    ///
    /// # Returns
    ///
    /// World-space distance in meters.
    #[inline]
    pub fn depth_to_distance(&self, depth_index: u32) -> f32 {
        if self.depth <= 1 {
            return self.max_distance * 0.5;
        }

        let t = depth_index as f32 / (self.depth - 1) as f32;
        // Exponential mapping for better near-field resolution
        // d = max_distance * (exp(t * k) - 1) / (exp(k) - 1)
        let k = 3.0; // Controls exponential curve steepness
        let exp_t = (t * k).exp();
        let exp_k = k.exp();
        self.max_distance * (exp_t - 1.0) / (exp_k - 1.0)
    }

    /// Convert world distance to depth index (inverse exponential mapping).
    ///
    /// # Arguments
    ///
    /// * `distance` - World-space distance in meters.
    ///
    /// # Returns
    ///
    /// Fractional depth index (can be used for interpolation).
    #[inline]
    pub fn distance_to_depth(&self, distance: f32) -> f32 {
        if self.depth <= 1 {
            return 0.0;
        }

        let clamped = distance.clamp(0.0, self.max_distance);
        let k: f32 = 3.0;
        let exp_k = k.exp();
        // Inverse of depth_to_distance
        // t = ln(1 + d * (exp(k) - 1) / max_distance) / k
        let ratio = clamped * (exp_k - 1.0) / self.max_distance;
        let t = (1.0 + ratio).ln() / k;
        t * (self.depth - 1) as f32
    }

    /// Get the index into the data array for a given texel.
    #[inline]
    pub fn texel_index(&self, u: u32, v: u32, w: u32) -> usize {
        ((w * self.height + v) * self.width + u) as usize * 4
    }

    /// Sample the LUT at a given world position.
    ///
    /// # Arguments
    ///
    /// * `camera_pos` - Camera position [x, y, z].
    /// * `world_pos` - World position to sample [x, y, z].
    /// * `sun_dir` - Normalized sun direction [x, y, z].
    ///
    /// # Returns
    ///
    /// Interpolated aerial sample.
    pub fn sample(
        &self,
        camera_pos: [f32; 3],
        world_pos: [f32; 3],
        sun_dir: [f32; 3],
    ) -> AerialSample {
        // Calculate view direction
        let dx = world_pos[0] - camera_pos[0];
        let dy = world_pos[1] - camera_pos[1];
        let dz = world_pos[2] - camera_pos[2];
        let distance = (dx * dx + dy * dy + dz * dz).sqrt();

        if distance < EPSILON {
            return AerialSample::identity();
        }

        let inv_dist = 1.0 / distance;
        let view_dir = [dx * inv_dist, dy * inv_dist, dz * inv_dist];

        // Calculate view-sun angle (dot product -> 0-1 range)
        let cos_angle = view_dir[0] * sun_dir[0]
            + view_dir[1] * sun_dir[1]
            + view_dir[2] * sun_dir[2];
        let u = (cos_angle * 0.5 + 0.5).clamp(0.0, 1.0);

        // Height (using Y as up, normalized to some max height)
        let avg_height = (camera_pos[1] + world_pos[1]) * 0.5;
        let max_height = 10000.0; // 10km normalization
        let v = (avg_height / max_height).clamp(0.0, 1.0);

        // Distance (exponential mapping)
        let w = self.distance_to_depth(distance) / (self.depth - 1).max(1) as f32;

        self.sample_uvw(u, v, w)
    }

    /// Sample the LUT using normalized UVW coordinates (0-1 range).
    pub fn sample_uvw(&self, u: f32, v: f32, w: f32) -> AerialSample {
        // Trilinear interpolation
        let u_scaled = u.clamp(0.0, 1.0) * (self.width - 1) as f32;
        let v_scaled = v.clamp(0.0, 1.0) * (self.height - 1) as f32;
        let w_scaled = w.clamp(0.0, 1.0) * (self.depth - 1) as f32;

        let u0 = (u_scaled as u32).min(self.width - 1);
        let v0 = (v_scaled as u32).min(self.height - 1);
        let w0 = (w_scaled as u32).min(self.depth - 1);

        let u1 = (u0 + 1).min(self.width - 1);
        let v1 = (v0 + 1).min(self.height - 1);
        let w1 = (w0 + 1).min(self.depth - 1);

        let fu = u_scaled.fract();
        let fv = v_scaled.fract();
        let fw = w_scaled.fract();

        // Sample 8 corners
        let c000 = self.sample_texel(u0, v0, w0);
        let c100 = self.sample_texel(u1, v0, w0);
        let c010 = self.sample_texel(u0, v1, w0);
        let c110 = self.sample_texel(u1, v1, w0);
        let c001 = self.sample_texel(u0, v0, w1);
        let c101 = self.sample_texel(u1, v0, w1);
        let c011 = self.sample_texel(u0, v1, w1);
        let c111 = self.sample_texel(u1, v1, w1);

        // Trilinear interpolation
        let c00 = AerialSample::lerp(&c000, &c100, fu);
        let c10 = AerialSample::lerp(&c010, &c110, fu);
        let c01 = AerialSample::lerp(&c001, &c101, fu);
        let c11 = AerialSample::lerp(&c011, &c111, fu);

        let c0 = AerialSample::lerp(&c00, &c10, fv);
        let c1 = AerialSample::lerp(&c01, &c11, fv);

        AerialSample::lerp(&c0, &c1, fw)
    }

    /// Sample a single texel from the LUT.
    fn sample_texel(&self, u: u32, v: u32, w: u32) -> AerialSample {
        let idx = self.texel_index(u, v, w);
        if idx + 3 >= self.data.len() {
            return AerialSample::identity();
        }

        AerialSample {
            inscatter: [self.data[idx], self.data[idx + 1], self.data[idx + 2]],
            transmittance: self.data[idx + 3],
            optical_depth: (-self.data[idx + 3].max(EPSILON).ln()).max(0.0),
            _padding: [0; 3],
        }
    }

    /// Get a reference to a depth slice of the LUT.
    ///
    /// # Arguments
    ///
    /// * `depth_index` - Depth slice index (0 to depth-1).
    ///
    /// # Returns
    ///
    /// Slice of the LUT data for this depth level.
    pub fn sample_depth_slice(&self, depth_index: u32) -> &[f32] {
        let slice_size = (self.width * self.height * 4) as usize;
        let start = depth_index as usize * slice_size;
        let end = (start + slice_size).min(self.data.len());
        &self.data[start..end]
    }

    /// Write a value to a texel.
    pub fn write_texel(&mut self, u: u32, v: u32, w: u32, sample: &AerialSample) {
        let idx = self.texel_index(u, v, w);
        if idx + 3 < self.data.len() {
            self.data[idx] = sample.inscatter[0];
            self.data[idx + 1] = sample.inscatter[1];
            self.data[idx + 2] = sample.inscatter[2];
            self.data[idx + 3] = sample.transmittance;
        }
    }

    /// Clear the LUT to identity values.
    pub fn clear(&mut self) {
        for i in 0..self.data.len() / 4 {
            let idx = i * 4;
            self.data[idx] = 0.0;
            self.data[idx + 1] = 0.0;
            self.data[idx + 2] = 0.0;
            self.data[idx + 3] = 1.0; // Full transmittance
        }
    }
}

impl Default for AerialPerspectiveLUT {
    fn default() -> Self {
        Self::with_defaults()
    }
}

// ---------------------------------------------------------------------------
// AerialPerspectiveRenderer
// ---------------------------------------------------------------------------

/// Renderer for aerial perspective effects.
///
/// Provides methods for computing and applying aerial perspective
/// to terrain and objects.
#[derive(Debug, Clone)]
pub struct AerialPerspectiveRenderer {
    /// Configuration for aerial perspective.
    pub config: AerialPerspectiveConfig,

    /// Optional bound LUT for fast sampling.
    lut: Option<AerialPerspectiveLUT>,

    /// Cached sun direction for inscatter calculations.
    sun_direction: [f32; 3],
}

impl AerialPerspectiveRenderer {
    /// Create a new aerial perspective renderer.
    pub fn new(config: AerialPerspectiveConfig) -> Self {
        Self {
            config,
            lut: None,
            sun_direction: [0.0, 1.0, 0.0], // Default: sun directly overhead
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(AerialPerspectiveConfig::default())
    }

    /// Bind a precomputed LUT for fast sampling.
    pub fn bind_lut(&mut self, lut: AerialPerspectiveLUT) {
        self.lut = Some(lut);
    }

    /// Unbind the current LUT.
    pub fn unbind_lut(&mut self) {
        self.lut = None;
    }

    /// Check if a LUT is currently bound.
    #[inline]
    pub fn has_lut(&self) -> bool {
        self.lut.is_some()
    }

    /// Set the sun direction (must be normalized).
    pub fn set_sun_direction(&mut self, direction: [f32; 3]) {
        let len = (direction[0] * direction[0]
            + direction[1] * direction[1]
            + direction[2] * direction[2])
        .sqrt();
        if len > EPSILON {
            self.sun_direction = [
                direction[0] / len,
                direction[1] / len,
                direction[2] / len,
            ];
        }
    }

    /// Compute aerial perspective effect for a world position.
    ///
    /// Uses the bound LUT if available, otherwise falls back to
    /// analytical evaluation.
    pub fn compute_aerial_effect(
        &self,
        camera_pos: [f32; 3],
        world_pos: [f32; 3],
        sun_dir: [f32; 3],
    ) -> AerialSample {
        // Fast path: use LUT if available
        if let Some(ref lut) = self.lut {
            return lut.sample(camera_pos, world_pos, sun_dir);
        }

        // Analytical fallback
        self.compute_aerial_effect_analytical(camera_pos, world_pos, sun_dir)
    }

    /// Compute aerial perspective using analytical integration.
    pub fn compute_aerial_effect_analytical(
        &self,
        camera_pos: [f32; 3],
        world_pos: [f32; 3],
        sun_dir: [f32; 3],
    ) -> AerialSample {
        let dx = world_pos[0] - camera_pos[0];
        let dy = world_pos[1] - camera_pos[1];
        let dz = world_pos[2] - camera_pos[2];
        let distance = (dx * dx + dy * dy + dz * dz).sqrt();

        if distance < EPSILON {
            return AerialSample::identity();
        }

        // Check blend factor
        let blend = self.config.blend_factor(distance);
        if blend < EPSILON {
            return AerialSample::identity();
        }

        // Compute transmittance and inscatter
        let inv_dist = 1.0 / distance;
        let view_dir = [dx * inv_dist, dy * inv_dist, dz * inv_dist];

        let inscatter = self.integrate_inscatter(camera_pos, world_pos, DEFAULT_INSCATTER_STEPS, sun_dir);
        let transmittance = self.compute_transmittance_along_ray(camera_pos, world_pos);
        let optical_depth = (-transmittance.max(EPSILON).ln()).max(0.0);

        // Apply blend factor
        let blended_inscatter = [
            inscatter[0] * blend * self.config.inscatter_intensity,
            inscatter[1] * blend * self.config.inscatter_intensity,
            inscatter[2] * blend * self.config.inscatter_intensity,
        ];
        let blended_transmittance = 1.0 - (1.0 - transmittance) * blend;

        AerialSample::new(blended_inscatter, blended_transmittance, optical_depth * blend)
    }

    /// Apply aerial perspective to a terrain color.
    ///
    /// # Arguments
    ///
    /// * `terrain_color` - Original terrain color (RGB).
    /// * `aerial_sample` - Aerial perspective sample.
    ///
    /// # Returns
    ///
    /// Modified color with aerial perspective applied.
    #[inline]
    pub fn apply_to_terrain_color(
        &self,
        terrain_color: [f32; 3],
        aerial_sample: &AerialSample,
    ) -> [f32; 3] {
        // Apply over operator: result = inscatter + transmittance * original
        [
            aerial_sample.inscatter[0] + aerial_sample.transmittance * terrain_color[0],
            aerial_sample.inscatter[1] + aerial_sample.transmittance * terrain_color[1],
            aerial_sample.inscatter[2] + aerial_sample.transmittance * terrain_color[2],
        ]
    }

    /// Apply aerial perspective to an object color based on distance and height.
    ///
    /// # Arguments
    ///
    /// * `object_color` - Original object color (RGB).
    /// * `distance` - Distance from camera in meters.
    /// * `height` - Height/altitude in meters.
    ///
    /// # Returns
    ///
    /// Modified color with aerial perspective applied.
    pub fn apply_to_object_color(
        &self,
        object_color: [f32; 3],
        distance: f32,
        height: f32,
    ) -> [f32; 3] {
        let blend = self.config.blend_factor(distance);
        if blend < EPSILON {
            return object_color;
        }

        // Simplified analytical model for objects
        let density_factor = self.height_density_ratio(height) * self.config.density_scale;
        let optical_depth = distance * density_factor * 1e-5;
        let transmittance = (-optical_depth).exp().clamp(0.0, 1.0);

        // Simple inscatter (blue-shifted)
        let inscatter_strength = (1.0 - transmittance) * self.config.inscatter_intensity * blend;
        let inscatter = [
            inscatter_strength * 0.6,  // R - least scatter
            inscatter_strength * 0.75, // G - medium scatter
            inscatter_strength * 1.0,  // B - most scatter (Rayleigh)
        ];

        let blended_transmittance = 1.0 - (1.0 - transmittance) * blend;

        [
            inscatter[0] + blended_transmittance * object_color[0],
            inscatter[1] + blended_transmittance * object_color[1],
            inscatter[2] + blended_transmittance * object_color[2],
        ]
    }

    /// Get the horizon blend factor for a given distance.
    ///
    /// Returns 0.0 at blend_start, 1.0 at blend_start + blend_range.
    #[inline]
    pub fn get_horizon_blend(&self, distance: f32) -> f32 {
        self.config.blend_factor(distance)
    }

    /// Get a simple distance fog factor.
    ///
    /// # Arguments
    ///
    /// * `distance` - Distance in meters.
    /// * `density` - Fog density coefficient.
    ///
    /// # Returns
    ///
    /// Fog factor (0 = clear, 1 = fully fogged).
    #[inline]
    pub fn get_distance_fog(&self, distance: f32, density: f32) -> f32 {
        // Exponential fog: fog = 1 - exp(-d * density)
        (1.0 - (-distance * density * self.config.density_scale).exp()).clamp(0.0, 1.0)
    }

    /// Integrate inscattered light along a ray.
    ///
    /// # Arguments
    ///
    /// * `start` - Ray start position.
    /// * `end` - Ray end position.
    /// * `num_steps` - Number of integration steps.
    /// * `sun_dir` - Normalized sun direction.
    ///
    /// # Returns
    ///
    /// Accumulated inscattered light (RGB).
    pub fn integrate_inscatter(
        &self,
        start: [f32; 3],
        end: [f32; 3],
        num_steps: u32,
        sun_dir: [f32; 3],
    ) -> [f32; 3] {
        let steps = num_steps.max(1);
        let step_size = 1.0 / steps as f32;

        // Ray direction
        let dx = end[0] - start[0];
        let dy = end[1] - start[1];
        let dz = end[2] - start[2];
        let ray_length = (dx * dx + dy * dy + dz * dz).sqrt();

        if ray_length < EPSILON {
            return [0.0, 0.0, 0.0];
        }

        let inv_len = 1.0 / ray_length;
        let view_dir = [dx * inv_len, dy * inv_len, dz * inv_len];

        // Calculate phase function (view-sun angle)
        let cos_angle = view_dir[0] * sun_dir[0]
            + view_dir[1] * sun_dir[1]
            + view_dir[2] * sun_dir[2];

        let rayleigh_phase = rayleigh_phase_function(cos_angle);
        let mie_phase = henyey_greenstein_phase(cos_angle, MIE_ASYMMETRY_G);

        let mut inscatter = [0.0f32; 3];
        let mut accumulated_transmittance = 1.0f32;

        for i in 0..steps {
            let t = (i as f32 + 0.5) * step_size;
            let pos = [
                start[0] + dx * t,
                start[1] + dy * t,
                start[2] + dz * t,
            ];

            let height = pos[1].max(0.0);
            let segment_length = ray_length * step_size;

            // Density at this height
            let rayleigh_density = (-height / RAYLEIGH_SCALE_HEIGHT).exp();
            let mie_density = (-height / MIE_SCALE_HEIGHT).exp();

            // Scattering contribution
            let rayleigh_scatter = [
                RAYLEIGH_SCATTERING[0] * rayleigh_density * rayleigh_phase,
                RAYLEIGH_SCATTERING[1] * rayleigh_density * rayleigh_phase,
                RAYLEIGH_SCATTERING[2] * rayleigh_density * rayleigh_phase,
            ];
            let mie_scatter = MIE_SCATTERING * mie_density * mie_phase;

            // Extinction for this segment
            let extinction =
                (RAYLEIGH_SCATTERING[1] * rayleigh_density + MIE_SCATTERING * mie_density)
                    * segment_length
                    * self.config.density_scale;

            let segment_transmittance = (-extinction).exp().clamp(0.0, 1.0);

            // Accumulate inscatter
            inscatter[0] += accumulated_transmittance * (rayleigh_scatter[0] + mie_scatter)
                * segment_length
                * self.config.density_scale;
            inscatter[1] += accumulated_transmittance * (rayleigh_scatter[1] + mie_scatter)
                * segment_length
                * self.config.density_scale;
            inscatter[2] += accumulated_transmittance * (rayleigh_scatter[2] + mie_scatter)
                * segment_length
                * self.config.density_scale;

            accumulated_transmittance *= segment_transmittance;

            // Early termination if fully attenuated
            if accumulated_transmittance < 0.001 {
                break;
            }
        }

        inscatter
    }

    /// Compute transmittance along a ray.
    ///
    /// # Arguments
    ///
    /// * `start` - Ray start position.
    /// * `end` - Ray end position.
    ///
    /// # Returns
    ///
    /// Transmittance (0 = fully absorbed, 1 = fully transmitted).
    pub fn compute_transmittance_along_ray(&self, start: [f32; 3], end: [f32; 3]) -> f32 {
        let dx = end[0] - start[0];
        let dy = end[1] - start[1];
        let dz = end[2] - start[2];
        let ray_length = (dx * dx + dy * dy + dz * dz).sqrt();

        if ray_length < EPSILON {
            return 1.0;
        }

        let steps = DEFAULT_INSCATTER_STEPS;
        let step_size = 1.0 / steps as f32;
        let mut optical_depth = 0.0f32;

        for i in 0..steps {
            let t = (i as f32 + 0.5) * step_size;
            let pos_y = start[1] + dy * t;
            let height = pos_y.max(0.0);

            let rayleigh_density = (-height / RAYLEIGH_SCALE_HEIGHT).exp();
            let mie_density = (-height / MIE_SCALE_HEIGHT).exp();

            let extinction = (RAYLEIGH_SCATTERING[1] * rayleigh_density
                + MIE_SCATTERING * mie_density)
                * ray_length
                * step_size
                * self.config.density_scale;

            optical_depth += extinction;
        }

        (-optical_depth).exp().clamp(0.0, 1.0)
    }

    /// Calculate the height-based density ratio.
    ///
    /// # Arguments
    ///
    /// * `altitude` - Height above ground in meters.
    ///
    /// # Returns
    ///
    /// Density ratio (1.0 at ground, exponentially decreasing with altitude).
    #[inline]
    pub fn height_density_ratio(&self, altitude: f32) -> f32 {
        (-altitude.max(0.0) * self.config.height_falloff).exp()
    }

    /// Apply a full aerial perspective pass to terrain.
    ///
    /// This method processes a terrain buffer and depth buffer to apply
    /// aerial perspective effects to the entire terrain.
    ///
    /// # Arguments
    ///
    /// * `terrain_buffer` - Mutable terrain color buffer (RGBA per pixel).
    /// * `depth_buffer` - Depth buffer (one f32 per pixel).
    /// * `width` - Buffer width in pixels.
    /// * `height` - Buffer height in pixels.
    /// * `camera_pos` - Camera world position.
    /// * `view_params` - View parameters for depth-to-world conversion.
    pub fn apply_aerial_perspective_pass(
        &self,
        terrain_buffer: &mut [f32],
        depth_buffer: &[f32],
        width: u32,
        height: u32,
        camera_pos: [f32; 3],
        near: f32,
        far: f32,
    ) {
        let pixel_count = (width * height) as usize;

        for i in 0..pixel_count {
            let idx = i * 4;
            if idx + 3 >= terrain_buffer.len() || i >= depth_buffer.len() {
                break;
            }

            let depth = depth_buffer[i];
            if depth >= 1.0 {
                continue; // Sky pixel
            }

            // Linearize depth
            let linear_depth = near * far / (far - depth * (far - near));
            let distance = linear_depth.max(0.0);

            // Apply simplified aerial perspective
            let blend = self.config.blend_factor(distance);
            if blend < EPSILON {
                continue;
            }

            let transmittance = 1.0 - self.get_distance_fog(distance, 1e-5);
            let inscatter_strength = (1.0 - transmittance) * self.config.inscatter_intensity * blend;

            // Blue-shifted inscatter
            let inscatter = [
                inscatter_strength * 0.6,
                inscatter_strength * 0.75,
                inscatter_strength * 1.0,
            ];

            let blended_transmittance = 1.0 - (1.0 - transmittance) * blend;

            terrain_buffer[idx] = inscatter[0] + blended_transmittance * terrain_buffer[idx];
            terrain_buffer[idx + 1] = inscatter[1] + blended_transmittance * terrain_buffer[idx + 1];
            terrain_buffer[idx + 2] = inscatter[2] + blended_transmittance * terrain_buffer[idx + 2];
        }
    }
}

impl Default for AerialPerspectiveRenderer {
    fn default() -> Self {
        Self::with_defaults()
    }
}

// ---------------------------------------------------------------------------
// Terrain Integration Helpers
// ---------------------------------------------------------------------------

/// Get visibility factor for terrain at a given clipmap level and distance.
///
/// Higher clipmap levels (coarser LOD) typically cover greater distances,
/// so their visibility is more affected by aerial perspective.
///
/// # Arguments
///
/// * `clipmap_level` - Clipmap LOD level (0 = finest).
/// * `distance` - Distance from camera in meters.
/// * `config` - Aerial perspective configuration.
///
/// # Returns
///
/// Visibility factor (0 = invisible, 1 = fully visible).
#[inline]
pub fn get_terrain_visibility(
    clipmap_level: u32,
    distance: f32,
    config: &AerialPerspectiveConfig,
) -> f32 {
    // Coarser levels (higher index) are at greater distances
    let level_distance_scale = 1 << clipmap_level;
    let effective_distance = distance;

    // Calculate transmittance-based visibility
    let density = config.density_scale * 1e-5 / (level_distance_scale as f32).sqrt();
    let optical_depth = effective_distance * density;
    (-optical_depth).exp().clamp(0.0, 1.0)
}

/// Adjust terrain color based on distance and altitude.
///
/// # Arguments
///
/// * `base_color` - Original terrain color (RGB).
/// * `distance` - Distance from camera in meters.
/// * `altitude` - Terrain altitude in meters.
/// * `config` - Aerial perspective configuration.
///
/// # Returns
///
/// Adjusted terrain color with aerial perspective.
pub fn adjust_terrain_color(
    base_color: [f32; 3],
    distance: f32,
    altitude: f32,
    config: &AerialPerspectiveConfig,
) -> [f32; 3] {
    let blend = config.blend_factor(distance);
    if blend < EPSILON {
        return base_color;
    }

    // Height-based density reduction
    let height_factor = (-altitude.max(0.0) * config.height_falloff).exp();
    let effective_density = config.density_scale * height_factor;

    // Simple exponential fog
    let optical_depth = distance * effective_density * 1e-5;
    let transmittance = (-optical_depth).exp().clamp(0.0, 1.0);

    // Inscatter (blue-shifted)
    let inscatter_strength = (1.0 - transmittance) * config.inscatter_intensity * blend;
    let inscatter = [
        inscatter_strength * 0.6,
        inscatter_strength * 0.75,
        inscatter_strength * 1.0,
    ];

    let blended_transmittance = 1.0 - (1.0 - transmittance) * blend;

    [
        inscatter[0] + blended_transmittance * base_color[0],
        inscatter[1] + blended_transmittance * base_color[1],
        inscatter[2] + blended_transmittance * base_color[2],
    ]
}

/// Compute fog factor for terrain at a given distance.
///
/// # Arguments
///
/// * `distance` - Distance from camera in meters.
/// * `config` - Aerial perspective configuration.
///
/// # Returns
///
/// Fog factor (0 = clear, 1 = fully fogged).
#[inline]
pub fn compute_terrain_fog_factor(distance: f32, config: &AerialPerspectiveConfig) -> f32 {
    let blend = config.blend_factor(distance);
    let optical_depth = distance * config.density_scale * 1e-5;
    let base_fog = 1.0 - (-optical_depth).exp();
    (base_fog * blend).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Phase Functions
// ---------------------------------------------------------------------------

/// Rayleigh phase function.
///
/// # Arguments
///
/// * `cos_angle` - Cosine of scattering angle.
///
/// # Returns
///
/// Phase function value.
#[inline]
pub fn rayleigh_phase_function(cos_angle: f32) -> f32 {
    (3.0 / (16.0 * PI)) * (1.0 + cos_angle * cos_angle)
}

/// Henyey-Greenstein phase function.
///
/// # Arguments
///
/// * `cos_angle` - Cosine of scattering angle.
/// * `g` - Asymmetry parameter (-1 to 1).
///
/// # Returns
///
/// Phase function value.
#[inline]
pub fn henyey_greenstein_phase(cos_angle: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_angle;
    if denom < EPSILON {
        return 1.0 / (4.0 * PI);
    }
    (1.0 - g2) / (4.0 * PI * denom.powf(1.5))
}

/// Cornette-Shanks phase function (improved Mie approximation).
///
/// # Arguments
///
/// * `cos_angle` - Cosine of scattering angle.
/// * `g` - Asymmetry parameter (-1 to 1).
///
/// # Returns
///
/// Phase function value.
#[inline]
pub fn cornette_shanks_phase(cos_angle: f32, g: f32) -> f32 {
    let g2 = g * g;
    let cos2 = cos_angle * cos_angle;
    let denom = 1.0 + g2 - 2.0 * g * cos_angle;
    if denom < EPSILON {
        return 1.0 / (4.0 * PI);
    }
    (3.0 / (8.0 * PI)) * ((1.0 - g2) * (1.0 + cos2)) / ((2.0 + g2) * denom.powf(1.5))
}

// ---------------------------------------------------------------------------
// Error Type
// ---------------------------------------------------------------------------

/// Errors that can occur in aerial perspective operations.
#[derive(Debug, Clone, PartialEq)]
pub enum AerialPerspectiveError {
    /// Invalid parameter value.
    InvalidParameter(String),

    /// LUT dimensions are invalid.
    InvalidLUTDimensions {
        width: u32,
        height: u32,
        depth: u32,
    },

    /// LUT data size mismatch.
    LUTDataSizeMismatch {
        expected: usize,
        actual: usize,
    },
}

impl std::fmt::Display for AerialPerspectiveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AerialPerspectiveError::InvalidParameter(msg) => {
                write!(f, "Invalid parameter: {}", msg)
            }
            AerialPerspectiveError::InvalidLUTDimensions { width, height, depth } => {
                write!(
                    f,
                    "Invalid LUT dimensions: {}x{}x{}",
                    width, height, depth
                )
            }
            AerialPerspectiveError::LUTDataSizeMismatch { expected, actual } => {
                write!(
                    f,
                    "LUT data size mismatch: expected {}, got {}",
                    expected, actual
                )
            }
        }
    }
}

impl std::error::Error for AerialPerspectiveError {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ===== Config Tests =====

    #[test]
    fn test_config_default() {
        let config = AerialPerspectiveConfig::default();
        assert_eq!(config.max_distance, DEFAULT_MAX_DISTANCE);
        assert_eq!(config.density_scale, DEFAULT_DENSITY_SCALE);
        assert_eq!(config.height_falloff, DEFAULT_HEIGHT_FALLOFF);
        assert_eq!(config.inscatter_intensity, DEFAULT_INSCATTER_INTENSITY);
        assert_eq!(config.blend_start, DEFAULT_BLEND_START);
        assert_eq!(config.blend_range, DEFAULT_BLEND_RANGE);
    }

    #[test]
    fn test_config_new() {
        let config = AerialPerspectiveConfig::new(50000.0, 2.0, 0.0001);
        assert_eq!(config.max_distance, 50000.0);
        assert_eq!(config.density_scale, 2.0);
        assert_eq!(config.height_falloff, 0.0001);
    }

    #[test]
    fn test_config_with_full_params() {
        let config = AerialPerspectiveConfig::with_full_params(
            80000.0, 1.5, 0.00015, 0.8, 100.0, 200.0,
        );
        assert_eq!(config.max_distance, 80000.0);
        assert_eq!(config.density_scale, 1.5);
        assert_eq!(config.inscatter_intensity, 0.8);
        assert_eq!(config.blend_start, 100.0);
        assert_eq!(config.blend_range, 200.0);
    }

    #[test]
    #[should_panic(expected = "max_distance must be positive")]
    fn test_config_invalid_max_distance() {
        AerialPerspectiveConfig::new(0.0, 1.0, 0.0001);
    }

    #[test]
    #[should_panic(expected = "density_scale must be in range")]
    fn test_config_invalid_density_scale() {
        AerialPerspectiveConfig::new(100000.0, 15.0, 0.0001);
    }

    #[test]
    fn test_config_validation() {
        let mut config = AerialPerspectiveConfig::default();
        assert!(config.validate().is_ok());

        config.max_distance = 0.0;
        assert!(config.validate().is_err());

        config.max_distance = 100000.0;
        config.blend_range = 0.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_blend_factor_below_start() {
        let config = AerialPerspectiveConfig::default();
        assert_eq!(config.blend_factor(0.0), 0.0);
        assert_eq!(config.blend_factor(500.0), 0.0); // At blend_start
    }

    #[test]
    fn test_config_blend_factor_above_range() {
        let config = AerialPerspectiveConfig::default();
        // blend_start = 500, blend_range = 500, so full blend at 1000+
        assert_eq!(config.blend_factor(1000.0), 1.0);
        assert_eq!(config.blend_factor(10000.0), 1.0);
    }

    #[test]
    fn test_config_blend_factor_smooth() {
        let config = AerialPerspectiveConfig::default();
        // Midpoint of blend (750m)
        let blend = config.blend_factor(750.0);
        assert!(blend > 0.0 && blend < 1.0);
        // Smoothstep at t=0.5 gives 0.5
        assert!((blend - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_config_normalized_distance() {
        let config = AerialPerspectiveConfig::default();
        assert_eq!(config.normalized_distance(0.0), 0.0);
        assert_eq!(config.normalized_distance(50000.0), 0.5);
        assert_eq!(config.normalized_distance(100000.0), 1.0);
        assert_eq!(config.normalized_distance(200000.0), 1.0); // Clamped
    }

    #[test]
    fn test_config_builder_pattern() {
        let config = AerialPerspectiveConfig::new(100000.0, 1.0, 0.0001)
            .with_inscatter_intensity(0.5)
            .with_blend(200.0, 300.0);

        assert_eq!(config.inscatter_intensity, 0.5);
        assert_eq!(config.blend_start, 200.0);
        assert_eq!(config.blend_range, 300.0);
    }

    // ===== AerialSample Tests =====

    #[test]
    fn test_sample_identity() {
        let sample = AerialSample::identity();
        assert_eq!(sample.inscatter, [0.0, 0.0, 0.0]);
        assert_eq!(sample.transmittance, 1.0);
        assert_eq!(sample.optical_depth, 0.0);
        assert!(sample.is_clear());
    }

    #[test]
    fn test_sample_fully_fogged() {
        let fog_color = [0.7, 0.8, 1.0];
        let sample = AerialSample::fully_fogged(fog_color);
        assert_eq!(sample.inscatter, fog_color);
        assert_eq!(sample.transmittance, 0.0);
        assert!(sample.is_fully_fogged());
    }

    #[test]
    fn test_sample_new() {
        let sample = AerialSample::new([0.1, 0.2, 0.3], 0.8, 0.5);
        assert_eq!(sample.inscatter, [0.1, 0.2, 0.3]);
        assert_eq!(sample.transmittance, 0.8);
        assert_eq!(sample.optical_depth, 0.5);
    }

    #[test]
    fn test_sample_lerp_at_zero() {
        let a = AerialSample::new([0.0, 0.0, 0.0], 1.0, 0.0);
        let b = AerialSample::new([1.0, 1.0, 1.0], 0.0, 2.0);
        let result = AerialSample::lerp(&a, &b, 0.0);
        assert_eq!(result.inscatter, a.inscatter);
        assert_eq!(result.transmittance, a.transmittance);
    }

    #[test]
    fn test_sample_lerp_at_one() {
        let a = AerialSample::new([0.0, 0.0, 0.0], 1.0, 0.0);
        let b = AerialSample::new([1.0, 1.0, 1.0], 0.0, 2.0);
        let result = AerialSample::lerp(&a, &b, 1.0);
        assert_eq!(result.inscatter, b.inscatter);
        assert_eq!(result.transmittance, b.transmittance);
    }

    #[test]
    fn test_sample_lerp_midpoint() {
        let a = AerialSample::new([0.0, 0.0, 0.0], 1.0, 0.0);
        let b = AerialSample::new([1.0, 1.0, 1.0], 0.0, 2.0);
        let result = AerialSample::lerp(&a, &b, 0.5);
        assert!((result.inscatter[0] - 0.5).abs() < EPSILON);
        assert!((result.transmittance - 0.5).abs() < EPSILON);
    }

    #[test]
    fn test_sample_combine() {
        let a = AerialSample::new([0.1, 0.1, 0.1], 0.8, 0.2);
        let b = AerialSample::new([0.2, 0.2, 0.2], 0.9, 0.1);
        let combined = a.combine(&b);

        // inscatter = a.inscatter + a.transmittance * b.inscatter
        let expected_inscatter = 0.1 + 0.8 * 0.2;
        assert!((combined.inscatter[0] - expected_inscatter).abs() < EPSILON);

        // transmittance = a.transmittance * b.transmittance
        assert!((combined.transmittance - 0.72).abs() < EPSILON);

        // optical_depth = sum
        assert!((combined.optical_depth - 0.3).abs() < EPSILON);
    }

    #[test]
    fn test_sample_is_clear() {
        let clear = AerialSample::new([0.0, 0.0, 0.0], 1.0, 0.0);
        assert!(clear.is_clear());

        let not_clear = AerialSample::new([0.1, 0.0, 0.0], 1.0, 0.0);
        assert!(!not_clear.is_clear());
    }

    #[test]
    fn test_sample_is_fully_fogged() {
        let fogged = AerialSample::new([0.5, 0.5, 0.5], 0.0001, 10.0);
        assert!(fogged.is_fully_fogged());

        let not_fogged = AerialSample::new([0.5, 0.5, 0.5], 0.5, 1.0);
        assert!(!not_fogged.is_fully_fogged());
    }

    // ===== LUT Tests =====

    #[test]
    fn test_lut_new() {
        let lut = AerialPerspectiveLUT::new(32, 32, 32);
        assert_eq!(lut.width, 32);
        assert_eq!(lut.height, 32);
        assert_eq!(lut.depth, 32);
        assert_eq!(lut.data.len(), 32 * 32 * 32 * 4);
    }

    #[test]
    fn test_lut_with_defaults() {
        let lut = AerialPerspectiveLUT::with_defaults();
        assert_eq!(lut.width, DEFAULT_LUT_WIDTH);
        assert_eq!(lut.height, DEFAULT_LUT_HEIGHT);
        assert_eq!(lut.depth, DEFAULT_LUT_DEPTH);
    }

    #[test]
    fn test_lut_with_max_distance() {
        let lut = AerialPerspectiveLUT::with_max_distance(16, 16, 16, 50000.0);
        assert_eq!(lut.max_distance, 50000.0);
    }

    #[test]
    fn test_lut_texel_count() {
        let lut = AerialPerspectiveLUT::new(8, 8, 8);
        assert_eq!(lut.texel_count(), 512);
    }

    #[test]
    fn test_lut_byte_size() {
        let lut = AerialPerspectiveLUT::new(8, 8, 8);
        assert_eq!(lut.byte_size(), 512 * 4 * 4); // 512 texels * 4 floats * 4 bytes
    }

    #[test]
    fn test_lut_depth_to_distance_zero() {
        let lut = AerialPerspectiveLUT::new(32, 32, 32);
        let dist = lut.depth_to_distance(0);
        assert!(dist < 100.0); // Near-zero distance
    }

    #[test]
    fn test_lut_depth_to_distance_max() {
        let lut = AerialPerspectiveLUT::new(32, 32, 32);
        let dist = lut.depth_to_distance(31);
        assert!((dist - lut.max_distance).abs() < 100.0);
    }

    #[test]
    fn test_lut_depth_distance_roundtrip() {
        let lut = AerialPerspectiveLUT::new(32, 32, 32);

        for d in [0, 8, 16, 24, 31] {
            let distance = lut.depth_to_distance(d);
            let recovered = lut.distance_to_depth(distance);
            assert!((recovered - d as f32).abs() < 0.1);
        }
    }

    #[test]
    fn test_lut_distance_to_depth_exponential() {
        let lut = AerialPerspectiveLUT::new(32, 32, 32);

        // Near distances should have higher resolution (more depth indices)
        let near_depth = lut.distance_to_depth(1000.0);
        let mid_depth = lut.distance_to_depth(50000.0);
        let far_depth = lut.distance_to_depth(100000.0);

        assert!(near_depth > 0.0);
        assert!(mid_depth > near_depth);
        assert!(far_depth > mid_depth);
    }

    #[test]
    fn test_lut_write_and_read_texel() {
        let mut lut = AerialPerspectiveLUT::new(8, 8, 8);
        let sample = AerialSample::new([0.5, 0.6, 0.7], 0.8, 0.5);

        lut.write_texel(3, 4, 5, &sample);
        let read = lut.sample_texel(3, 4, 5);

        assert!((read.inscatter[0] - 0.5).abs() < EPSILON);
        assert!((read.inscatter[1] - 0.6).abs() < EPSILON);
        assert!((read.inscatter[2] - 0.7).abs() < EPSILON);
        assert!((read.transmittance - 0.8).abs() < EPSILON);
    }

    #[test]
    fn test_lut_clear() {
        let mut lut = AerialPerspectiveLUT::new(4, 4, 4);

        // Write some data
        let sample = AerialSample::new([1.0, 1.0, 1.0], 0.5, 1.0);
        lut.write_texel(0, 0, 0, &sample);

        // Clear
        lut.clear();

        // Verify identity values
        let read = lut.sample_texel(0, 0, 0);
        assert!(read.is_clear());
    }

    #[test]
    fn test_lut_sample_at_origin() {
        let mut lut = AerialPerspectiveLUT::new(8, 8, 8);
        lut.clear();

        let camera = [0.0, 0.0, 0.0];
        let world = [0.0, 0.0, 0.0];
        let sun = [0.0, 1.0, 0.0];

        let sample = lut.sample(camera, world, sun);
        assert!(sample.is_clear());
    }

    #[test]
    fn test_lut_sample_uvw_corners() {
        let mut lut = AerialPerspectiveLUT::new(4, 4, 4);

        // Write a value at corner (0,0,0)
        let sample = AerialSample::new([1.0, 0.0, 0.0], 0.5, 0.5);
        lut.write_texel(0, 0, 0, &sample);

        let read = lut.sample_uvw(0.0, 0.0, 0.0);
        assert!((read.inscatter[0] - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_lut_sample_depth_slice() {
        let lut = AerialPerspectiveLUT::new(8, 8, 4);
        let slice = lut.sample_depth_slice(0);
        assert_eq!(slice.len(), 8 * 8 * 4);
    }

    // ===== Renderer Tests =====

    #[test]
    fn test_renderer_new() {
        let config = AerialPerspectiveConfig::default();
        let renderer = AerialPerspectiveRenderer::new(config);
        assert!(!renderer.has_lut());
    }

    #[test]
    fn test_renderer_with_defaults() {
        let renderer = AerialPerspectiveRenderer::with_defaults();
        assert_eq!(renderer.config.max_distance, DEFAULT_MAX_DISTANCE);
    }

    #[test]
    fn test_renderer_bind_unbind_lut() {
        let mut renderer = AerialPerspectiveRenderer::with_defaults();
        assert!(!renderer.has_lut());

        let lut = AerialPerspectiveLUT::with_defaults();
        renderer.bind_lut(lut);
        assert!(renderer.has_lut());

        renderer.unbind_lut();
        assert!(!renderer.has_lut());
    }

    #[test]
    fn test_renderer_set_sun_direction() {
        let mut renderer = AerialPerspectiveRenderer::with_defaults();
        renderer.set_sun_direction([1.0, 0.0, 0.0]);
        assert!((renderer.sun_direction[0] - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_renderer_set_sun_direction_normalizes() {
        let mut renderer = AerialPerspectiveRenderer::with_defaults();
        renderer.set_sun_direction([2.0, 0.0, 0.0]);
        assert!((renderer.sun_direction[0] - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_renderer_compute_aerial_effect_at_origin() {
        let renderer = AerialPerspectiveRenderer::with_defaults();
        let camera = [0.0, 0.0, 0.0];
        let world = [0.0, 0.0, 0.0];
        let sun = [0.0, 1.0, 0.0];

        let sample = renderer.compute_aerial_effect(camera, world, sun);
        assert!(sample.is_clear());
    }

    #[test]
    fn test_renderer_compute_aerial_effect_near_distance() {
        let renderer = AerialPerspectiveRenderer::with_defaults();
        let camera = [0.0, 100.0, 0.0];
        let world = [100.0, 100.0, 0.0]; // 100m away, within blend start
        let sun = [0.0, 1.0, 0.0];

        let sample = renderer.compute_aerial_effect(camera, world, sun);
        // Should have minimal effect at this distance (below blend start)
        assert!(sample.transmittance > 0.99);
    }

    #[test]
    fn test_renderer_compute_aerial_effect_far_distance() {
        let renderer = AerialPerspectiveRenderer::with_defaults();
        let camera = [0.0, 100.0, 0.0];
        let world = [50000.0, 100.0, 0.0]; // 50km away
        let sun = [0.0, 1.0, 0.0];

        let sample = renderer.compute_aerial_effect(camera, world, sun);
        // Should have significant effect
        assert!(sample.transmittance < 0.95);
        assert!(sample.inscatter[0] > 0.0 || sample.inscatter[1] > 0.0 || sample.inscatter[2] > 0.0);
    }

    #[test]
    fn test_renderer_inscatter_increases_with_distance() {
        let renderer = AerialPerspectiveRenderer::with_defaults();
        let camera = [0.0, 100.0, 0.0];
        let sun = [0.0, 1.0, 0.0];

        let sample_near = renderer.compute_aerial_effect(camera, [5000.0, 100.0, 0.0], sun);
        let sample_far = renderer.compute_aerial_effect(camera, [50000.0, 100.0, 0.0], sun);

        // Inscatter should increase with distance
        let inscatter_near = sample_near.inscatter[0] + sample_near.inscatter[1] + sample_near.inscatter[2];
        let inscatter_far = sample_far.inscatter[0] + sample_far.inscatter[1] + sample_far.inscatter[2];

        assert!(inscatter_far > inscatter_near);
    }

    #[test]
    fn test_renderer_transmittance_decreases_with_distance() {
        let renderer = AerialPerspectiveRenderer::with_defaults();
        let camera = [0.0, 100.0, 0.0];
        let sun = [0.0, 1.0, 0.0];

        let sample_near = renderer.compute_aerial_effect(camera, [5000.0, 100.0, 0.0], sun);
        let sample_far = renderer.compute_aerial_effect(camera, [50000.0, 100.0, 0.0], sun);

        // Transmittance should decrease with distance
        assert!(sample_near.transmittance > sample_far.transmittance);
    }

    #[test]
    fn test_renderer_apply_to_terrain_color() {
        let renderer = AerialPerspectiveRenderer::with_defaults();
        let terrain = [0.3, 0.5, 0.2]; // Green terrain
        let sample = AerialSample::new([0.1, 0.15, 0.2], 0.8, 0.5);

        let result = renderer.apply_to_terrain_color(terrain, &sample);

        // result = inscatter + transmittance * terrain
        assert!((result[0] - (0.1 + 0.8 * 0.3)).abs() < EPSILON);
        assert!((result[1] - (0.15 + 0.8 * 0.5)).abs() < EPSILON);
        assert!((result[2] - (0.2 + 0.8 * 0.2)).abs() < EPSILON);
    }

    #[test]
    fn test_renderer_apply_to_object_color_near() {
        let renderer = AerialPerspectiveRenderer::with_defaults();
        let color = [1.0, 0.0, 0.0]; // Red object

        // Very close distance - should return original color
        let result = renderer.apply_to_object_color(color, 100.0, 100.0);
        assert!((result[0] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_renderer_apply_to_object_color_far() {
        let renderer = AerialPerspectiveRenderer::with_defaults();
        let color = [1.0, 0.0, 0.0]; // Red object

        // Far distance - should be affected by aerial perspective
        let result = renderer.apply_to_object_color(color, 50000.0, 100.0);

        // Should have some blue inscatter
        assert!(result[2] > 0.0);
        // Red should be reduced
        assert!(result[0] < 1.0);
    }

    #[test]
    fn test_renderer_get_horizon_blend() {
        let renderer = AerialPerspectiveRenderer::with_defaults();

        assert_eq!(renderer.get_horizon_blend(0.0), 0.0);
        assert_eq!(renderer.get_horizon_blend(500.0), 0.0);
        assert_eq!(renderer.get_horizon_blend(1000.0), 1.0);
    }

    #[test]
    fn test_renderer_get_distance_fog() {
        let renderer = AerialPerspectiveRenderer::with_defaults();

        let fog_near = renderer.get_distance_fog(100.0, 0.001);
        let fog_far = renderer.get_distance_fog(10000.0, 0.001);

        assert!(fog_far > fog_near);
        assert!(fog_near >= 0.0);
        assert!(fog_far <= 1.0);
    }

    #[test]
    fn test_renderer_height_density_ratio_at_ground() {
        let renderer = AerialPerspectiveRenderer::with_defaults();
        let ratio = renderer.height_density_ratio(0.0);
        assert!((ratio - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_renderer_height_density_ratio_decreases_with_altitude() {
        let renderer = AerialPerspectiveRenderer::with_defaults();

        let ratio_low = renderer.height_density_ratio(1000.0);
        let ratio_high = renderer.height_density_ratio(5000.0);

        assert!(ratio_low > ratio_high);
        assert!(ratio_high > 0.0);
    }

    #[test]
    fn test_renderer_height_density_ratio_negative_altitude() {
        let renderer = AerialPerspectiveRenderer::with_defaults();
        // Negative altitude (underground) should be treated as 0
        let ratio = renderer.height_density_ratio(-100.0);
        assert!((ratio - 1.0).abs() < EPSILON);
    }

    // ===== Integration Tests =====

    #[test]
    fn test_integrate_inscatter_zero_distance() {
        let renderer = AerialPerspectiveRenderer::with_defaults();
        let pos = [0.0, 1000.0, 0.0];
        let sun = [0.0, 1.0, 0.0];

        let inscatter = renderer.integrate_inscatter(pos, pos, 16, sun);
        assert!((inscatter[0]).abs() < EPSILON);
        assert!((inscatter[1]).abs() < EPSILON);
        assert!((inscatter[2]).abs() < EPSILON);
    }

    #[test]
    fn test_integrate_inscatter_positive_values() {
        let renderer = AerialPerspectiveRenderer::with_defaults();
        let start = [0.0, 1000.0, 0.0];
        let end = [10000.0, 1000.0, 0.0];
        let sun = [0.0, 1.0, 0.0];

        let inscatter = renderer.integrate_inscatter(start, end, 16, sun);

        // All inscatter values should be positive
        assert!(inscatter[0] >= 0.0);
        assert!(inscatter[1] >= 0.0);
        assert!(inscatter[2] >= 0.0);

        // Blue should scatter more (Rayleigh)
        assert!(inscatter[2] >= inscatter[0]);
    }

    #[test]
    fn test_compute_transmittance_at_origin() {
        let renderer = AerialPerspectiveRenderer::with_defaults();
        let pos = [0.0, 1000.0, 0.0];

        let t = renderer.compute_transmittance_along_ray(pos, pos);
        assert!((t - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_compute_transmittance_decreases_with_distance() {
        let renderer = AerialPerspectiveRenderer::with_defaults();
        let start = [0.0, 1000.0, 0.0];

        let t_near = renderer.compute_transmittance_along_ray(start, [1000.0, 1000.0, 0.0]);
        let t_far = renderer.compute_transmittance_along_ray(start, [50000.0, 1000.0, 0.0]);

        assert!(t_near > t_far);
        assert!(t_far > 0.0);
        assert!(t_near <= 1.0);
    }

    // ===== Terrain Integration Tests =====

    #[test]
    fn test_get_terrain_visibility_level_0() {
        let config = AerialPerspectiveConfig::default();
        let vis = get_terrain_visibility(0, 1000.0, &config);
        assert!(vis > 0.0 && vis <= 1.0);
    }

    #[test]
    fn test_get_terrain_visibility_decreases_with_distance() {
        let config = AerialPerspectiveConfig::default();

        let vis_near = get_terrain_visibility(0, 1000.0, &config);
        let vis_far = get_terrain_visibility(0, 50000.0, &config);

        assert!(vis_near > vis_far);
    }

    #[test]
    fn test_adjust_terrain_color_near() {
        let config = AerialPerspectiveConfig::default();
        let color = [0.5, 0.5, 0.5];

        let adjusted = adjust_terrain_color(color, 100.0, 0.0, &config);

        // Near terrain should be mostly unchanged
        assert!((adjusted[0] - color[0]).abs() < 0.01);
    }

    #[test]
    fn test_adjust_terrain_color_far() {
        let config = AerialPerspectiveConfig::default();
        let color = [0.5, 0.5, 0.5];

        let adjusted = adjust_terrain_color(color, 50000.0, 0.0, &config);

        // Far terrain should have blue shift
        assert!(adjusted[2] > color[2]);
    }

    #[test]
    fn test_adjust_terrain_color_high_altitude() {
        let config = AerialPerspectiveConfig::default();
        let color = [0.5, 0.5, 0.5];

        let low_alt = adjust_terrain_color(color, 20000.0, 0.0, &config);
        let high_alt = adjust_terrain_color(color, 20000.0, 5000.0, &config);

        // High altitude should have less effect
        let diff_low = (low_alt[0] - color[0]).abs() + (low_alt[1] - color[1]).abs() + (low_alt[2] - color[2]).abs();
        let diff_high = (high_alt[0] - color[0]).abs() + (high_alt[1] - color[1]).abs() + (high_alt[2] - color[2]).abs();

        assert!(diff_high < diff_low);
    }

    #[test]
    fn test_compute_terrain_fog_factor_zero_distance() {
        let config = AerialPerspectiveConfig::default();
        let fog = compute_terrain_fog_factor(0.0, &config);
        assert!(fog < 0.01);
    }

    #[test]
    fn test_compute_terrain_fog_factor_increases() {
        let config = AerialPerspectiveConfig::default();

        let fog_near = compute_terrain_fog_factor(5000.0, &config);
        let fog_far = compute_terrain_fog_factor(50000.0, &config);

        assert!(fog_far > fog_near);
    }

    #[test]
    fn test_compute_terrain_fog_factor_clamped() {
        let config = AerialPerspectiveConfig::default();
        let fog = compute_terrain_fog_factor(1000000.0, &config);
        assert!(fog <= 1.0);
        assert!(fog >= 0.0);
    }

    // ===== Phase Function Tests =====

    #[test]
    fn test_rayleigh_phase_forward() {
        let phase = rayleigh_phase_function(1.0); // Forward scattering
        assert!(phase > 0.0);
    }

    #[test]
    fn test_rayleigh_phase_backward() {
        let phase = rayleigh_phase_function(-1.0); // Backward scattering
        assert!(phase > 0.0);
    }

    #[test]
    fn test_rayleigh_phase_symmetric() {
        let forward = rayleigh_phase_function(1.0);
        let backward = rayleigh_phase_function(-1.0);
        // Rayleigh is symmetric
        assert!((forward - backward).abs() < EPSILON);
    }

    #[test]
    fn test_rayleigh_phase_perpendicular() {
        let perp = rayleigh_phase_function(0.0); // 90 degrees
        let forward = rayleigh_phase_function(1.0);
        // Perpendicular scattering is weaker
        assert!(perp < forward);
    }

    #[test]
    fn test_henyey_greenstein_forward_dominant() {
        let g = 0.8; // Strong forward scattering
        let forward = henyey_greenstein_phase(1.0, g);
        let backward = henyey_greenstein_phase(-1.0, g);
        assert!(forward > backward);
    }

    #[test]
    fn test_henyey_greenstein_isotropic() {
        let g = 0.0; // Isotropic
        let forward = henyey_greenstein_phase(1.0, g);
        let backward = henyey_greenstein_phase(-1.0, g);
        let perp = henyey_greenstein_phase(0.0, g);

        // All should be equal for isotropic
        assert!((forward - backward).abs() < EPSILON);
        assert!((forward - perp).abs() < EPSILON);
    }

    #[test]
    fn test_cornette_shanks_forward_dominant() {
        let g = 0.8;
        let forward = cornette_shanks_phase(1.0, g);
        let backward = cornette_shanks_phase(-1.0, g);
        assert!(forward > backward);
    }

    // ===== Edge Cases =====

    #[test]
    fn test_edge_case_zero_distance() {
        let renderer = AerialPerspectiveRenderer::with_defaults();
        let pos = [1000.0, 500.0, 2000.0];
        let sun = [0.0, 1.0, 0.0];

        let sample = renderer.compute_aerial_effect(pos, pos, sun);
        assert!(sample.is_clear());
    }

    #[test]
    fn test_edge_case_max_distance() {
        let renderer = AerialPerspectiveRenderer::with_defaults();
        let camera = [0.0, 100.0, 0.0];
        let world = [100000.0, 100.0, 0.0]; // At max distance
        let sun = [0.0, 1.0, 0.0];

        let sample = renderer.compute_aerial_effect(camera, world, sun);
        // Should have significant effect but not be NaN or infinite
        assert!(sample.transmittance.is_finite());
        assert!(sample.inscatter[0].is_finite());
    }

    #[test]
    fn test_edge_case_underground() {
        let config = AerialPerspectiveConfig::default();

        // Underground point (negative Y)
        let color = [0.5, 0.5, 0.5];
        let adjusted = adjust_terrain_color(color, 10000.0, -100.0, &config);

        // Should still work, treating as ground level
        assert!(adjusted[0].is_finite());
        assert!(adjusted[1].is_finite());
        assert!(adjusted[2].is_finite());
    }

    #[test]
    fn test_edge_case_very_high_altitude() {
        let renderer = AerialPerspectiveRenderer::with_defaults();

        // Very high altitude - minimal atmospheric effect
        let ratio = renderer.height_density_ratio(50000.0);
        assert!(ratio < 0.01); // Very low density at 50km
    }

    // ===== Error Type Tests =====

    #[test]
    fn test_error_display_invalid_parameter() {
        let err = AerialPerspectiveError::InvalidParameter("test".to_string());
        let display = format!("{}", err);
        assert!(display.contains("Invalid parameter"));
    }

    #[test]
    fn test_error_display_invalid_lut_dimensions() {
        let err = AerialPerspectiveError::InvalidLUTDimensions {
            width: 0,
            height: 32,
            depth: 32,
        };
        let display = format!("{}", err);
        assert!(display.contains("Invalid LUT dimensions"));
    }

    #[test]
    fn test_error_display_lut_data_mismatch() {
        let err = AerialPerspectiveError::LUTDataSizeMismatch {
            expected: 1000,
            actual: 500,
        };
        let display = format!("{}", err);
        assert!(display.contains("LUT data size mismatch"));
    }

    // ===== Pod/Zeroable Tests =====

    #[test]
    fn test_config_is_pod() {
        // This test verifies that AerialPerspectiveConfig can be used with bytemuck
        let config = AerialPerspectiveConfig::default();
        let bytes: &[u8] = bytemuck::bytes_of(&config);
        assert_eq!(bytes.len(), std::mem::size_of::<AerialPerspectiveConfig>());
    }

    #[test]
    fn test_sample_is_pod() {
        let sample = AerialSample::default();
        let bytes: &[u8] = bytemuck::bytes_of(&sample);
        assert_eq!(bytes.len(), std::mem::size_of::<AerialSample>());
    }

    #[test]
    fn test_config_alignment() {
        // Config should be 32 bytes for GPU alignment
        assert_eq!(std::mem::size_of::<AerialPerspectiveConfig>(), 32);
    }

    #[test]
    fn test_sample_alignment() {
        // Sample should be 32 bytes for GPU alignment
        assert_eq!(std::mem::size_of::<AerialSample>(), 32);
    }

    // ===== Apply Pass Tests =====

    #[test]
    fn test_apply_aerial_perspective_pass_empty() {
        let renderer = AerialPerspectiveRenderer::with_defaults();
        let mut terrain_buffer: Vec<f32> = vec![];
        let depth_buffer: Vec<f32> = vec![];

        // Should not panic on empty buffers
        renderer.apply_aerial_perspective_pass(
            &mut terrain_buffer,
            &depth_buffer,
            0, 0,
            [0.0, 100.0, 0.0],
            0.1, 1000.0,
        );
    }

    #[test]
    fn test_apply_aerial_perspective_pass_basic() {
        let renderer = AerialPerspectiveRenderer::with_defaults();

        // 2x2 terrain buffer (RGBA)
        let mut terrain_buffer = vec![
            0.5, 0.5, 0.5, 1.0, // Pixel 0
            0.5, 0.5, 0.5, 1.0, // Pixel 1
            0.5, 0.5, 0.5, 1.0, // Pixel 2
            0.5, 0.5, 0.5, 1.0, // Pixel 3
        ];

        // Varying depths
        let depth_buffer = vec![0.1, 0.5, 0.9, 1.0];

        renderer.apply_aerial_perspective_pass(
            &mut terrain_buffer,
            &depth_buffer,
            2, 2,
            [0.0, 100.0, 0.0],
            0.1, 100000.0,
        );

        // Sky pixel (depth 1.0) should be unchanged
        assert!((terrain_buffer[12] - 0.5).abs() < EPSILON);
    }
}
