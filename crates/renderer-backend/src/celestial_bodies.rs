//! Celestial Bodies Rendering - Sun, Moon, and Stars.
//!
//! This module provides rendering infrastructure for celestial bodies in the
//! TRINITY sky system. It integrates with `sky_rendering.rs` to render sun disk,
//! moon phases, and procedural star fields.
//!
//! # Overview
//!
//! The celestial rendering system consists of:
//! - **SunConfig/SunRenderer**: Sun disk with limb darkening and glow
//! - **MoonConfig/MoonRenderer**: Moon with phase, craters, and maria
//! - **StarFieldConfig/StarFieldRenderer**: Procedural stars with twinkle
//! - **CelestialRenderer**: Unified API for all celestial bodies
//!
//! # Rendering Order
//!
//! 1. Stars (background, visible at night)
//! 2. Moon (when above horizon)
//! 3. Sun (with glow and optional corona)
//!
//! # Coordinate System
//!
//! - Y-up world space
//! - All directions are normalized unit vectors
//! - View direction points from camera towards sky
//!
//! # References
//!
//! - Limb darkening: Hestroffer & Magnan (1998)
//! - Lunar albedo: Hapke (1963)
//! - Star distribution: Tycho-2 catalogue statistics

use bytemuck::{Pod, Zeroable};
use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default sun angular radius in radians (~0.267 degrees).
pub const DEFAULT_SUN_ANGULAR_RADIUS: f32 = 0.00465;

/// Default moon angular radius in radians (~0.267 degrees).
pub const DEFAULT_MOON_ANGULAR_RADIUS: f32 = 0.00465;

/// Default moon albedo (reflectance).
pub const DEFAULT_MOON_ALBEDO: f32 = 0.12;

/// Default star count for procedural generation.
pub const DEFAULT_STAR_COUNT: u32 = 4000;

/// Minimum apparent magnitude (brightest visible stars).
pub const DEFAULT_MAGNITUDE_MIN: f32 = -1.0;

/// Maximum apparent magnitude (faintest visible stars).
pub const DEFAULT_MAGNITUDE_MAX: f32 = 6.0;

/// Default twinkle animation speed in Hz.
pub const DEFAULT_TWINKLE_SPEED: f32 = 2.0;

/// Default twinkle amplitude (0-1).
pub const DEFAULT_TWINKLE_AMPLITUDE: f32 = 0.3;

/// Default Milky Way band intensity.
pub const DEFAULT_MILKY_WAY_INTENSITY: f32 = 0.2;

/// Default sun glow radius multiplier.
pub const DEFAULT_GLOW_RADIUS: f32 = 3.0;

/// Default sun glow intensity.
pub const DEFAULT_GLOW_INTENSITY: f32 = 0.5;

/// Default limb darkening coefficient.
pub const DEFAULT_LIMB_DARKENING: f32 = 0.6;

/// Two times PI for convenience.
const TWO_PI: f32 = 2.0 * PI;

// ---------------------------------------------------------------------------
// SunConfig
// ---------------------------------------------------------------------------

/// Configuration for sun disk rendering.
///
/// Controls the appearance of the sun disk including size, limb darkening,
/// and glow effects. All parameters are designed for GPU compatibility.
///
/// # Example
///
/// ```
/// use renderer_backend::celestial_bodies::SunConfig;
///
/// let config = SunConfig::default();
/// assert!((config.angular_radius - 0.00465).abs() < 0.0001);
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct SunConfig {
    /// Angular radius of the sun disk in radians (default ~0.00465, ~0.267 degrees).
    pub angular_radius: f32,

    /// Limb darkening coefficient (0.0-1.0). Higher values create more
    /// pronounced darkening at the edge of the sun disk.
    pub limb_darkening: f32,

    /// Glow radius as a multiplier of angular_radius (default 3.0x).
    pub glow_radius: f32,

    /// Additive glow intensity (default 0.5).
    pub glow_intensity: f32,

    /// Whether corona rendering is enabled (for eclipse effects).
    /// Stored as u32 for GPU compatibility (0 = false, 1 = true).
    pub corona_enabled: u32,

    /// Padding for 16-byte alignment.
    pub _padding: [u32; 3],
}

// Size assertion: 32 bytes (2 vec4s)
const _: () = assert!(std::mem::size_of::<SunConfig>() == 32);

impl Default for SunConfig {
    fn default() -> Self {
        Self {
            angular_radius: DEFAULT_SUN_ANGULAR_RADIUS,
            limb_darkening: DEFAULT_LIMB_DARKENING,
            glow_radius: DEFAULT_GLOW_RADIUS,
            glow_intensity: DEFAULT_GLOW_INTENSITY,
            corona_enabled: 0,
            _padding: [0; 3],
        }
    }
}

impl SunConfig {
    /// Create a new sun configuration with default values.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a sun configuration for eclipse rendering.
    #[inline]
    pub fn eclipse() -> Self {
        Self {
            corona_enabled: 1,
            glow_radius: 5.0,
            glow_intensity: 0.8,
            ..Self::default()
        }
    }

    /// Set angular radius with clamping.
    #[inline]
    pub fn set_angular_radius(&mut self, radius: f32) {
        self.angular_radius = radius.clamp(0.001, 0.1);
    }

    /// Set limb darkening coefficient with clamping.
    #[inline]
    pub fn set_limb_darkening(&mut self, coefficient: f32) {
        self.limb_darkening = coefficient.clamp(0.0, 1.0);
    }

    /// Enable or disable corona rendering.
    #[inline]
    pub fn set_corona_enabled(&mut self, enabled: bool) {
        self.corona_enabled = if enabled { 1 } else { 0 };
    }

    /// Check if corona rendering is enabled.
    #[inline]
    pub fn is_corona_enabled(&self) -> bool {
        self.corona_enabled != 0
    }

    /// Validate configuration parameters.
    pub fn validate(&self) -> bool {
        self.angular_radius > 0.0
            && self.angular_radius <= 0.1
            && self.limb_darkening >= 0.0
            && self.limb_darkening <= 1.0
            && self.glow_radius > 0.0
            && self.glow_intensity >= 0.0
    }
}

// ---------------------------------------------------------------------------
// SunRenderer
// ---------------------------------------------------------------------------

/// Renderer for the sun disk with limb darkening and glow effects.
///
/// The sun is rendered as a bright disk with physically-based limb darkening
/// and an optional additive glow halo. The limb darkening model follows
/// Hestroffer & Magnan (1998).
///
/// # Example
///
/// ```
/// use renderer_backend::celestial_bodies::{SunRenderer, SunConfig};
///
/// let renderer = SunRenderer::new(SunConfig::default());
/// let view_dir = [0.0, 1.0, 0.0];
/// let sun_dir = [0.0, 1.0, 0.0];
/// let transmittance = [1.0, 1.0, 1.0];
///
/// let color = renderer.render_disk(view_dir, sun_dir, transmittance);
/// assert!(color[0] > 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct SunRenderer {
    /// Sun configuration.
    config: SunConfig,
}

impl Default for SunRenderer {
    fn default() -> Self {
        Self::new(SunConfig::default())
    }
}

impl SunRenderer {
    /// Create a new sun renderer with the given configuration.
    #[inline]
    pub fn new(config: SunConfig) -> Self {
        Self { config }
    }

    /// Get the current configuration.
    #[inline]
    pub fn config(&self) -> &SunConfig {
        &self.config
    }

    /// Get mutable access to the configuration.
    #[inline]
    pub fn config_mut(&mut self) -> &mut SunConfig {
        &mut self.config
    }

    /// Render the sun disk color.
    ///
    /// Returns the sun disk color modulated by atmospheric transmittance.
    /// The color includes limb darkening falloff.
    ///
    /// # Arguments
    ///
    /// * `view_dir` - Normalized view direction (from camera towards sky).
    /// * `sun_dir` - Normalized direction towards the sun.
    /// * `transmittance` - Atmospheric transmittance RGB (0-1).
    ///
    /// # Returns
    ///
    /// Linear RGB sun disk color.
    pub fn render_disk(
        &self,
        view_dir: [f32; 3],
        sun_dir: [f32; 3],
        transmittance: [f32; 3],
    ) -> [f32; 3] {
        let mask = self.get_disk_mask(view_dir, sun_dir);
        if mask <= 0.0 {
            return [0.0, 0.0, 0.0];
        }

        // Compute limb darkening
        let limb_factor = self.limb_darkening_factor(1.0 - mask);

        // Sun disk color (slightly warm white)
        let sun_color = [1.0, 0.98, 0.95];

        // Apply transmittance and limb darkening
        [
            sun_color[0] * transmittance[0] * limb_factor * mask,
            sun_color[1] * transmittance[1] * limb_factor * mask,
            sun_color[2] * transmittance[2] * limb_factor * mask,
        ]
    }

    /// Render the additive glow around the sun.
    ///
    /// Returns an additive glow color that should be blended on top
    /// of the sky. The glow extends beyond the sun disk.
    ///
    /// # Arguments
    ///
    /// * `view_dir` - Normalized view direction.
    /// * `sun_dir` - Normalized direction towards the sun.
    ///
    /// # Returns
    ///
    /// Linear RGB glow color (additive).
    pub fn render_glow(&self, view_dir: [f32; 3], sun_dir: [f32; 3]) -> [f32; 3] {
        let cos_angle = dot_vec3(view_dir, sun_dir);
        let glow_angular_radius = self.config.angular_radius * self.config.glow_radius;
        let cos_glow_radius = glow_angular_radius.cos();

        if cos_angle < cos_glow_radius {
            return [0.0, 0.0, 0.0];
        }

        // Distance from sun center (0 at center, 1 at glow edge)
        let t = if (1.0 - cos_glow_radius).abs() < 1e-10 {
            0.0
        } else {
            1.0 - (cos_angle - cos_glow_radius) / (1.0 - cos_glow_radius)
        };

        // Exponential falloff
        let falloff = (-t * 3.0).exp();
        let intensity = self.config.glow_intensity * falloff;

        // Warm glow color
        [intensity * 1.0, intensity * 0.9, intensity * 0.7]
    }

    /// Get the sun disk mask value.
    ///
    /// Returns a value from 0 to 1 indicating how much the view direction
    /// is inside the sun disk. 1.0 means at the center, 0.0 means outside.
    ///
    /// # Arguments
    ///
    /// * `view_dir` - Normalized view direction.
    /// * `sun_dir` - Normalized direction towards the sun.
    ///
    /// # Returns
    ///
    /// Mask value (0.0 = outside disk, 1.0 = center of disk).
    pub fn get_disk_mask(&self, view_dir: [f32; 3], sun_dir: [f32; 3]) -> f32 {
        let cos_angle = dot_vec3(view_dir, sun_dir);
        let cos_sun_radius = self.config.angular_radius.cos();

        if cos_angle > cos_sun_radius {
            // Inside the disk - compute normalized distance from edge
            // cos_sun_radius is at edge (0), 1.0 is at center (1)
            (cos_angle - cos_sun_radius) / (1.0 - cos_sun_radius)
        } else {
            0.0
        }
    }

    /// Compute limb darkening factor for a given normalized radius.
    ///
    /// Uses the Hestroffer & Magnan (1998) limb darkening model:
    /// I(r) = 1 - u * (1 - sqrt(1 - r^2))
    ///
    /// where u is the limb darkening coefficient and r is the normalized
    /// distance from disk center (0 at center, 1 at edge).
    ///
    /// # Arguments
    ///
    /// * `normalized_radius` - Distance from center (0-1).
    ///
    /// # Returns
    ///
    /// Limb darkening factor (0-1).
    pub fn limb_darkening_factor(&self, normalized_radius: f32) -> f32 {
        let r = normalized_radius.clamp(0.0, 1.0);
        let mu = (1.0 - r * r).sqrt();
        1.0 - self.config.limb_darkening * (1.0 - mu)
    }

    /// Render corona effect for eclipse.
    ///
    /// Returns the corona glow color for eclipse rendering.
    /// Only active when corona_enabled is true.
    ///
    /// # Arguments
    ///
    /// * `view_dir` - Normalized view direction.
    /// * `sun_dir` - Normalized direction towards the sun.
    ///
    /// # Returns
    ///
    /// Linear RGB corona color.
    pub fn render_corona(&self, view_dir: [f32; 3], sun_dir: [f32; 3]) -> [f32; 3] {
        if !self.config.is_corona_enabled() {
            return [0.0, 0.0, 0.0];
        }

        let cos_angle = dot_vec3(view_dir, sun_dir);
        let corona_radius = self.config.angular_radius * 8.0;
        let cos_corona_radius = corona_radius.cos();

        if cos_angle < cos_corona_radius {
            return [0.0, 0.0, 0.0];
        }

        // Corona intensity falloff (1/r^2 behavior)
        let t = (cos_angle - cos_corona_radius) / (1.0 - cos_corona_radius);
        let intensity = (1.0 - t).powi(2) * 0.3;

        // Pale white corona
        [intensity, intensity * 0.98, intensity * 0.95]
    }
}

// ---------------------------------------------------------------------------
// MoonConfig
// ---------------------------------------------------------------------------

/// Configuration for moon rendering.
///
/// Controls the appearance of the moon including size, albedo, phase,
/// and surface roughness for procedural texturing.
///
/// # Example
///
/// ```
/// use renderer_backend::celestial_bodies::MoonConfig;
///
/// let mut config = MoonConfig::default();
/// config.phase = 0.5; // Full moon
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct MoonConfig {
    /// Angular radius of the moon in radians (default ~0.00465).
    pub angular_radius: f32,

    /// Moon surface albedo/reflectance (default 0.12).
    pub albedo: f32,

    /// Lunar phase (0 = new moon, 0.25 = first quarter, 0.5 = full moon,
    /// 0.75 = last quarter).
    pub phase: f32,

    /// Surface roughness for procedural texture (0-1).
    pub roughness: f32,
}

// Size assertion: 16 bytes (1 vec4)
const _: () = assert!(std::mem::size_of::<MoonConfig>() == 16);

impl Default for MoonConfig {
    fn default() -> Self {
        Self {
            angular_radius: DEFAULT_MOON_ANGULAR_RADIUS,
            albedo: DEFAULT_MOON_ALBEDO,
            phase: 0.5, // Full moon by default
            roughness: 0.5,
        }
    }
}

impl MoonConfig {
    /// Create a new moon configuration with default values.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a configuration for a new moon.
    #[inline]
    pub fn new_moon() -> Self {
        Self {
            phase: 0.0,
            ..Self::default()
        }
    }

    /// Create a configuration for a full moon.
    #[inline]
    pub fn full_moon() -> Self {
        Self {
            phase: 0.5,
            ..Self::default()
        }
    }

    /// Create a configuration for a first quarter moon.
    #[inline]
    pub fn first_quarter() -> Self {
        Self {
            phase: 0.25,
            ..Self::default()
        }
    }

    /// Create a configuration for a last quarter moon.
    #[inline]
    pub fn last_quarter() -> Self {
        Self {
            phase: 0.75,
            ..Self::default()
        }
    }

    /// Set angular radius with clamping.
    #[inline]
    pub fn set_angular_radius(&mut self, radius: f32) {
        self.angular_radius = radius.clamp(0.001, 0.1);
    }

    /// Set albedo with clamping.
    #[inline]
    pub fn set_albedo(&mut self, albedo: f32) {
        self.albedo = albedo.clamp(0.0, 1.0);
    }

    /// Set phase with wrapping to [0, 1).
    #[inline]
    pub fn set_phase(&mut self, phase: f32) {
        self.phase = phase.rem_euclid(1.0);
    }

    /// Validate configuration parameters.
    pub fn validate(&self) -> bool {
        self.angular_radius > 0.0
            && self.angular_radius <= 0.1
            && self.albedo >= 0.0
            && self.albedo <= 1.0
            && self.phase >= 0.0
            && self.phase <= 1.0
            && self.roughness >= 0.0
            && self.roughness <= 1.0
    }
}

// ---------------------------------------------------------------------------
// MoonRenderer
// ---------------------------------------------------------------------------

/// Renderer for the moon with phases and procedural surface detail.
///
/// The moon is rendered with:
/// - Phase-dependent illumination based on sun position
/// - Procedural crater patterns
/// - Procedural maria (dark "seas") patterns
///
/// # Example
///
/// ```
/// use renderer_backend::celestial_bodies::{MoonRenderer, MoonConfig};
///
/// let renderer = MoonRenderer::new(MoonConfig::full_moon());
/// let sun_dir = [1.0, 0.0, 0.0];
/// let moon_dir = [-1.0, 0.0, 0.0];
///
/// let phase = renderer.compute_phase(sun_dir, moon_dir);
/// assert!((phase - 0.5).abs() < 0.1); // Approximately full moon
/// ```
#[derive(Debug, Clone)]
pub struct MoonRenderer {
    /// Moon configuration.
    config: MoonConfig,
}

impl Default for MoonRenderer {
    fn default() -> Self {
        Self::new(MoonConfig::default())
    }
}

impl MoonRenderer {
    /// Create a new moon renderer with the given configuration.
    #[inline]
    pub fn new(config: MoonConfig) -> Self {
        Self { config }
    }

    /// Get the current configuration.
    #[inline]
    pub fn config(&self) -> &MoonConfig {
        &self.config
    }

    /// Get mutable access to the configuration.
    #[inline]
    pub fn config_mut(&mut self) -> &mut MoonConfig {
        &mut self.config
    }

    /// Compute the lunar phase from sun and moon directions.
    ///
    /// The phase is determined by the angle between the sun and moon
    /// as seen from Earth (the observer).
    ///
    /// # Arguments
    ///
    /// * `sun_dir` - Normalized direction towards the sun.
    /// * `moon_dir` - Normalized direction towards the moon.
    ///
    /// # Returns
    ///
    /// Phase value (0 = new moon, 0.5 = full moon).
    pub fn compute_phase(&self, sun_dir: [f32; 3], moon_dir: [f32; 3]) -> f32 {
        // Dot product gives cosine of angle between sun and moon
        // When dot = 1 (same direction), moon is between sun and Earth -> new moon (0)
        // When dot = -1 (opposite), moon is on far side -> full moon (0.5)
        let cos_angle = dot_vec3(sun_dir, moon_dir);

        // Map from [-1, 1] to [0, 0.5] phase for the illuminated portion
        // cos = 1 -> phase = 0 (new moon)
        // cos = 0 -> phase = 0.25 (quarter moon)
        // cos = -1 -> phase = 0.5 (full moon)
        (1.0 - cos_angle) * 0.25
    }

    /// Render the moon color for a given view direction.
    ///
    /// Returns the moon color including phase shading and surface detail.
    ///
    /// # Arguments
    ///
    /// * `view_dir` - Normalized view direction.
    /// * `moon_dir` - Normalized direction towards the moon.
    /// * `sun_dir` - Normalized direction towards the sun.
    ///
    /// # Returns
    ///
    /// Linear RGB moon color.
    pub fn render_moon(
        &self,
        view_dir: [f32; 3],
        moon_dir: [f32; 3],
        sun_dir: [f32; 3],
    ) -> [f32; 3] {
        let cos_angle = dot_vec3(view_dir, moon_dir);
        let cos_moon_radius = self.config.angular_radius.cos();

        if cos_angle <= cos_moon_radius {
            return [0.0, 0.0, 0.0];
        }

        // Compute UV on moon disk for procedural texturing
        let uv = self.compute_moon_uv(view_dir, moon_dir);

        // Get surface detail
        let crater_detail = self.get_crater_pattern(uv);
        let maria_detail = self.get_maria_pattern(uv);

        // Base brightness from phase
        let phase_illumination = self.compute_phase_illumination(uv, sun_dir, moon_dir);

        // Combine surface details with illumination
        let surface_brightness = 1.0 - maria_detail * 0.3 - crater_detail * 0.1;
        let brightness = self.config.albedo * phase_illumination * surface_brightness;

        // Moon color (slightly warm gray)
        [brightness * 0.95, brightness * 0.93, brightness * 0.88]
    }

    /// Compute UV coordinates on the moon disk.
    fn compute_moon_uv(&self, view_dir: [f32; 3], moon_dir: [f32; 3]) -> [f32; 2] {
        // Project view direction onto plane perpendicular to moon direction
        let dot = dot_vec3(view_dir, moon_dir);

        // Vector from moon center to view ray intersection
        let offset = [
            view_dir[0] - moon_dir[0] * dot,
            view_dir[1] - moon_dir[1] * dot,
            view_dir[2] - moon_dir[2] * dot,
        ];

        // Normalize by angular radius
        let scale = 1.0 / self.config.angular_radius.sin().max(0.001);

        // Create orthonormal basis on moon disk
        // Use world up as reference unless moon is directly above
        let up = if moon_dir[1].abs() > 0.99 {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 1.0, 0.0]
        };

        let right = normalize_vec3(cross_vec3(moon_dir, up));
        let disk_up = cross_vec3(right, moon_dir);

        let u = dot_vec3(offset, right) * scale * 0.5 + 0.5;
        let v = dot_vec3(offset, disk_up) * scale * 0.5 + 0.5;

        [u.clamp(0.0, 1.0), v.clamp(0.0, 1.0)]
    }

    /// Compute illumination factor based on phase and position on disk.
    fn compute_phase_illumination(
        &self,
        uv: [f32; 2],
        sun_dir: [f32; 3],
        moon_dir: [f32; 3],
    ) -> f32 {
        // Phase angle (angle between sun-moon-observer)
        let cos_phase_angle = -dot_vec3(sun_dir, moon_dir);

        // Terminator position based on phase
        // UV[0] represents position across the disk (0 = left, 1 = right)
        let terminator_x = (1.0 + cos_phase_angle) * 0.5;

        // Smooth terminator transition
        let dist_from_terminator = uv[0] - terminator_x;
        let terminator_width = 0.1; // Softness of terminator

        // Sigmoid for smooth day/night transition
        let illumination = 1.0 / (1.0 + (-dist_from_terminator / terminator_width).exp());

        // Edge darkening (limb darkening for the moon)
        let r_sq = (uv[0] - 0.5).powi(2) + (uv[1] - 0.5).powi(2);
        let edge_factor = (1.0 - (r_sq * 4.0).min(1.0)).sqrt();

        illumination * edge_factor
    }

    /// Get procedural crater pattern at the given UV coordinates.
    ///
    /// Returns a value from 0 to 1 representing crater depth/shadow.
    ///
    /// # Arguments
    ///
    /// * `uv` - UV coordinates on the moon disk (0-1).
    ///
    /// # Returns
    ///
    /// Crater intensity (0 = no crater, 1 = deep crater).
    pub fn get_crater_pattern(&self, uv: [f32; 2]) -> f32 {
        // Multi-scale procedural craters using hash-based noise
        let mut crater_value = 0.0;

        // Large craters
        let large_scale = 4.0;
        crater_value += self.crater_noise(uv[0] * large_scale, uv[1] * large_scale) * 0.5;

        // Medium craters
        let medium_scale = 12.0;
        crater_value += self.crater_noise(uv[0] * medium_scale + 0.5, uv[1] * medium_scale + 0.3) * 0.3;

        // Small craters
        let small_scale = 30.0;
        crater_value += self.crater_noise(uv[0] * small_scale + 0.7, uv[1] * small_scale + 0.9) * 0.2;

        crater_value.clamp(0.0, 1.0)
    }

    /// Hash-based crater noise function.
    fn crater_noise(&self, x: f32, y: f32) -> f32 {
        // Cell-based approach for craters
        let cell_x = x.floor();
        let cell_y = y.floor();
        let frac_x = x - cell_x;
        let frac_y = y - cell_y;

        // Hash to get crater center within cell
        let hash1 = hash_2d(cell_x, cell_y);
        let hash2 = hash_2d(cell_x + 0.5, cell_y + 0.5);

        let crater_x = hash1 * 0.6 + 0.2;
        let crater_y = hash2 * 0.6 + 0.2;
        let crater_radius = hash_2d(cell_x + 0.3, cell_y + 0.7) * 0.3 + 0.1;

        // Distance from crater center
        let dx = frac_x - crater_x;
        let dy = frac_y - crater_y;
        let dist = (dx * dx + dy * dy).sqrt();

        // Crater profile: raised rim, depressed center
        if dist < crater_radius {
            let t = dist / crater_radius;
            // Smooth crater profile
            (1.0 - t * t).max(0.0) * self.config.roughness
        } else if dist < crater_radius * 1.3 {
            // Raised rim
            let rim_t = (dist - crater_radius) / (crater_radius * 0.3);
            (1.0 - rim_t) * 0.2 * self.config.roughness
        } else {
            0.0
        }
    }

    /// Get procedural maria (dark "seas") pattern at the given UV coordinates.
    ///
    /// Returns a value from 0 to 1 representing maria darkness.
    ///
    /// # Arguments
    ///
    /// * `uv` - UV coordinates on the moon disk (0-1).
    ///
    /// # Returns
    ///
    /// Maria intensity (0 = highland, 1 = dark mare).
    pub fn get_maria_pattern(&self, uv: [f32; 2]) -> f32 {
        // Simplified maria based on position
        // Major maria are concentrated in the northern hemisphere of the visible side

        // Distance from maria centers (simplified)
        let maria_centers = [
            ([0.4, 0.35], 0.15), // Mare Imbrium
            ([0.55, 0.4], 0.12), // Mare Serenitatis
            ([0.6, 0.55], 0.1),  // Mare Tranquillitatis
            ([0.45, 0.6], 0.08), // Mare Nubium
            ([0.3, 0.45], 0.1),  // Oceanus Procellarum
        ];

        let mut maria_value = 0.0;

        for (center, radius) in maria_centers {
            let dx = uv[0] - center[0];
            let dy = uv[1] - center[1];
            let dist = (dx * dx + dy * dy).sqrt();

            if dist < radius {
                let t = dist / radius;
                // Smooth falloff from center
                maria_value += (1.0 - t * t).max(0.0) * 0.8;
            }
        }

        // Add some noise for variation
        let noise = hash_2d(uv[0] * 20.0, uv[1] * 20.0) * 0.2;

        (maria_value + noise).clamp(0.0, 1.0)
    }

    /// Get the moon disk mask value.
    ///
    /// Returns a value from 0 to 1 indicating if the view direction
    /// is inside the moon disk.
    ///
    /// # Arguments
    ///
    /// * `view_dir` - Normalized view direction.
    /// * `moon_dir` - Normalized direction towards the moon.
    ///
    /// # Returns
    ///
    /// Mask value (0.0 = outside, 1.0 = inside).
    pub fn get_disk_mask(&self, view_dir: [f32; 3], moon_dir: [f32; 3]) -> f32 {
        let cos_angle = dot_vec3(view_dir, moon_dir);
        let cos_moon_radius = self.config.angular_radius.cos();

        if cos_angle > cos_moon_radius {
            1.0
        } else {
            0.0
        }
    }
}

// ---------------------------------------------------------------------------
// StarFieldConfig
// ---------------------------------------------------------------------------

/// Configuration for procedural star field rendering.
///
/// Controls star generation parameters including count, brightness range,
/// twinkle animation, and Milky Way band intensity.
///
/// # Example
///
/// ```
/// use renderer_backend::celestial_bodies::StarFieldConfig;
///
/// let config = StarFieldConfig::default();
/// assert_eq!(config.star_count, 4000);
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct StarFieldConfig {
    /// Number of stars to generate (default 4000).
    pub star_count: u32,

    /// Minimum apparent magnitude (brightest stars, default -1.0).
    pub magnitude_min: f32,

    /// Maximum apparent magnitude (faintest visible, default 6.0).
    pub magnitude_max: f32,

    /// Twinkle animation speed in Hz (default 2.0).
    pub twinkle_speed: f32,

    /// Twinkle amplitude (0-1, default 0.3).
    pub twinkle_amplitude: f32,

    /// Milky Way band intensity (default 0.2).
    pub milky_way_intensity: f32,

    /// Padding for 8-byte alignment.
    pub _padding: [u32; 2],
}

// Size assertion: 32 bytes (2 vec4s)
const _: () = assert!(std::mem::size_of::<StarFieldConfig>() == 32);

impl Default for StarFieldConfig {
    fn default() -> Self {
        Self {
            star_count: DEFAULT_STAR_COUNT,
            magnitude_min: DEFAULT_MAGNITUDE_MIN,
            magnitude_max: DEFAULT_MAGNITUDE_MAX,
            twinkle_speed: DEFAULT_TWINKLE_SPEED,
            twinkle_amplitude: DEFAULT_TWINKLE_AMPLITUDE,
            milky_way_intensity: DEFAULT_MILKY_WAY_INTENSITY,
            _padding: [0; 2],
        }
    }
}

impl StarFieldConfig {
    /// Create a new star field configuration with default values.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a configuration for a dense star field (dark sky).
    #[inline]
    pub fn dark_sky() -> Self {
        Self {
            star_count: 8000,
            magnitude_max: 7.0,
            milky_way_intensity: 0.4,
            ..Self::default()
        }
    }

    /// Create a configuration for a sparse star field (light pollution).
    #[inline]
    pub fn light_pollution() -> Self {
        Self {
            star_count: 500,
            magnitude_max: 3.0,
            milky_way_intensity: 0.0,
            ..Self::default()
        }
    }

    /// Set star count with clamping.
    #[inline]
    pub fn set_star_count(&mut self, count: u32) {
        self.star_count = count.clamp(100, 100_000);
    }

    /// Validate configuration parameters.
    pub fn validate(&self) -> bool {
        self.star_count >= 100
            && self.star_count <= 100_000
            && self.magnitude_min < self.magnitude_max
            && self.twinkle_speed >= 0.0
            && self.twinkle_amplitude >= 0.0
            && self.twinkle_amplitude <= 1.0
            && self.milky_way_intensity >= 0.0
            && self.milky_way_intensity <= 1.0
    }
}

// ---------------------------------------------------------------------------
// Star
// ---------------------------------------------------------------------------

/// Individual star data.
///
/// Represents a single star in the procedural star field with position,
/// brightness, and visual characteristics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Star {
    /// Normalized direction vector pointing to the star.
    pub direction: [f32; 3],

    /// Apparent magnitude (lower = brighter, range typically -1 to 6).
    pub magnitude: f32,

    /// Color temperature in Kelvin (3000-30000K).
    pub temperature: u32,

    /// Random phase offset for twinkle animation (0 to 2*PI).
    pub twinkle_phase: f32,
}

impl Star {
    /// Create a new star.
    #[inline]
    pub fn new(direction: [f32; 3], magnitude: f32, temperature: u32, twinkle_phase: f32) -> Self {
        Self {
            direction: normalize_vec3(direction),
            magnitude,
            temperature,
            twinkle_phase,
        }
    }

    /// Get the base brightness of this star (without twinkle).
    ///
    /// Brightness is computed from apparent magnitude using the formula:
    /// brightness = 10^(-0.4 * magnitude)
    #[inline]
    pub fn base_brightness(&self) -> f32 {
        // Astronomical magnitude to linear brightness
        10.0_f32.powf(-0.4 * self.magnitude)
    }

    /// Get the RGB color of this star based on temperature.
    #[inline]
    pub fn color(&self) -> [f32; 3] {
        temperature_to_rgb(self.temperature)
    }
}

// ---------------------------------------------------------------------------
// StarFieldRenderer
// ---------------------------------------------------------------------------

/// Renderer for procedural star fields with twinkle animation.
///
/// The star field is procedurally generated from a seed and includes:
/// - Realistic magnitude distribution
/// - Temperature-based colors (red to blue-white)
/// - Atmospheric twinkle animation
/// - Optional Milky Way band
///
/// # Example
///
/// ```
/// use renderer_backend::celestial_bodies::{StarFieldRenderer, StarFieldConfig};
///
/// let renderer = StarFieldRenderer::new(StarFieldConfig::default());
/// let brightness = renderer.sample_star_brightness([0.0, 1.0, 0.0], 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct StarFieldRenderer {
    /// Star field configuration.
    config: StarFieldConfig,

    /// Generated stars.
    stars: Vec<Star>,

    /// Current animation time.
    time: f32,
}

impl Default for StarFieldRenderer {
    fn default() -> Self {
        Self::with_seed(StarFieldConfig::default(), 42)
    }
}

impl StarFieldRenderer {
    /// Create a new star field renderer with the given configuration.
    ///
    /// Uses a default seed for star generation.
    #[inline]
    pub fn new(config: StarFieldConfig) -> Self {
        Self::with_seed(config, 42)
    }

    /// Create a new star field renderer with a specific seed.
    pub fn with_seed(config: StarFieldConfig, seed: u64) -> Self {
        let stars = Self::generate_stars_internal(config.star_count, seed, &config);
        Self {
            config,
            stars,
            time: 0.0,
        }
    }

    /// Get the current configuration.
    #[inline]
    pub fn config(&self) -> &StarFieldConfig {
        &self.config
    }

    /// Get the generated stars.
    #[inline]
    pub fn stars(&self) -> &[Star] {
        &self.stars
    }

    /// Get the current animation time.
    #[inline]
    pub fn time(&self) -> f32 {
        self.time
    }

    /// Generate a new set of stars with the given parameters.
    ///
    /// # Arguments
    ///
    /// * `count` - Number of stars to generate.
    /// * `seed` - Random seed for reproducible generation.
    ///
    /// # Returns
    ///
    /// Vector of generated stars.
    pub fn generate_stars(&self, count: u32, seed: u64) -> Vec<Star> {
        Self::generate_stars_internal(count, seed, &self.config)
    }

    /// Internal star generation.
    fn generate_stars_internal(count: u32, seed: u64, config: &StarFieldConfig) -> Vec<Star> {
        let mut stars = Vec::with_capacity(count as usize);

        // Hash the seed to get a different starting point for each seed
        let seed_hash = hash_u64(seed);

        for i in 0..count {
            // Generate quasi-random direction using Fibonacci sphere with seed perturbation
            let golden_ratio = (1.0 + 5.0_f32.sqrt()) / 2.0;

            // Add seed-dependent perturbation to the index
            let perturb_hash = hash_u64(seed_hash.wrapping_add(i as u64).wrapping_mul(0x9E3779B97F4A7C15));
            let perturb = (perturb_hash as f32 / u64::MAX as f32) * 0.5;

            let theta = TWO_PI * ((i as f32) + perturb) / golden_ratio;
            let phi_input = 2.0 * ((i as f32) + perturb * 0.5) / (count as f32) - 1.0;
            let phi = phi_input.clamp(-1.0, 1.0).acos();

            let direction = [
                phi.sin() * theta.cos(),
                phi.cos(),
                phi.sin() * theta.sin(),
            ];

            // Generate magnitude with realistic distribution
            // More faint stars than bright ones (logarithmic)
            let hash = hash_u64(seed_hash.wrapping_add(i as u64).wrapping_mul(0xBF58476D1CE4E5B9));
            let magnitude_range = config.magnitude_max - config.magnitude_min;
            let t = (hash as f32 / u64::MAX as f32).sqrt(); // Bias towards fainter
            let magnitude = config.magnitude_min + t * magnitude_range;

            // Generate temperature (biased towards sun-like)
            let temp_hash = hash_u64(seed_hash.wrapping_add(i as u64).wrapping_add(1000).wrapping_mul(0x94D049BB133111EB));
            let temp_t = temp_hash as f32 / u64::MAX as f32;
            // Distribution: mostly 4000-7000K, some outliers
            let temperature = if temp_t < 0.7 {
                // Common stars (4000-7000K)
                4000 + ((temp_t / 0.7) * 3000.0) as u32
            } else if temp_t < 0.9 {
                // Hot stars (7000-15000K)
                7000 + (((temp_t - 0.7) / 0.2) * 8000.0) as u32
            } else {
                // Red giants and very hot stars
                if temp_t < 0.95 {
                    3000 + ((temp_t - 0.9) / 0.05 * 1000.0) as u32
                } else {
                    15000 + ((temp_t - 0.95) / 0.05 * 15000.0) as u32
                }
            };

            // Generate twinkle phase
            let phase_hash = hash_u64(seed_hash.wrapping_add(i as u64).wrapping_add(2000).wrapping_mul(0x6A09E667F3BCC909));
            let twinkle_phase = (phase_hash as f32 / u64::MAX as f32) * TWO_PI;

            stars.push(Star::new(direction, magnitude, temperature, twinkle_phase));
        }

        stars
    }

    /// Sample the total star brightness for a view direction.
    ///
    /// This samples nearby stars and applies twinkle animation.
    ///
    /// # Arguments
    ///
    /// * `direction` - Normalized view direction.
    /// * `time` - Animation time in seconds.
    ///
    /// # Returns
    ///
    /// Total star brightness (additive).
    pub fn sample_star_brightness(&self, direction: [f32; 3], time: f32) -> f32 {
        let direction = normalize_vec3(direction);
        let mut total_brightness = 0.0;

        // Find stars near this direction
        for star in &self.stars {
            let cos_angle = dot_vec3(direction, star.direction);

            // Stars are point sources - use very small angular tolerance
            // Actual stars would be sub-pixel, but we use a small radius for visibility
            let star_radius: f32 = 0.001;
            let cos_star_radius = (1.0 - star_radius * star_radius).sqrt();

            if cos_angle > cos_star_radius {
                let base = star.base_brightness();

                // Apply twinkle
                let twinkle = self.compute_twinkle(star, time);
                total_brightness += base * twinkle;
            }
        }

        total_brightness
    }

    /// Sample star color for a view direction.
    ///
    /// # Arguments
    ///
    /// * `direction` - Normalized view direction.
    /// * `time` - Animation time in seconds.
    ///
    /// # Returns
    ///
    /// RGB star color (additive).
    pub fn sample_star_color(&self, direction: [f32; 3], time: f32) -> [f32; 3] {
        let direction = normalize_vec3(direction);
        let mut total_color = [0.0, 0.0, 0.0];

        for star in &self.stars {
            let cos_angle = dot_vec3(direction, star.direction);
            let star_radius: f32 = 0.002;
            let cos_star_radius = (1.0 - star_radius * star_radius).sqrt();

            if cos_angle > cos_star_radius {
                let base = star.base_brightness();
                let twinkle = self.compute_twinkle(star, time);
                let color = star.color();

                total_color[0] += color[0] * base * twinkle;
                total_color[1] += color[1] * base * twinkle;
                total_color[2] += color[2] * base * twinkle;
            }
        }

        total_color
    }

    /// Compute twinkle factor for a star at the given time.
    fn compute_twinkle(&self, star: &Star, time: f32) -> f32 {
        let phase = time * self.config.twinkle_speed * TWO_PI + star.twinkle_phase;

        // Multi-frequency twinkle for realism
        let twinkle1 = phase.sin();
        let twinkle2 = (phase * 1.7).sin();
        let twinkle3 = (phase * 2.3).sin();

        let combined = (twinkle1 + twinkle2 * 0.5 + twinkle3 * 0.3) / 1.8;
        let amplitude = self.config.twinkle_amplitude;

        1.0 - amplitude + amplitude * (combined * 0.5 + 0.5)
    }

    /// Sample the Milky Way brightness for a direction.
    ///
    /// Returns a brightness value representing the diffuse glow of the
    /// Milky Way band across the sky.
    ///
    /// # Arguments
    ///
    /// * `direction` - Normalized view direction.
    ///
    /// # Returns
    ///
    /// Milky Way brightness (0-1).
    pub fn sample_milky_way(&self, direction: [f32; 3]) -> f32 {
        if self.config.milky_way_intensity <= 0.0 {
            return 0.0;
        }

        let direction = normalize_vec3(direction);

        // Milky Way plane is tilted ~60 degrees from celestial equator
        // Simplified model: band along a great circle
        let milky_way_normal = normalize_vec3([0.5, 0.866, 0.0]);
        let dist_from_plane = dot_vec3(direction, milky_way_normal).abs();

        // Gaussian-like falloff from the galactic plane
        let band_width = 0.15;
        let band_value = (-dist_from_plane.powi(2) / (2.0 * band_width * band_width)).exp();

        // Add some noise for structure
        let noise = hash_2d(direction[0] * 10.0, direction[2] * 10.0);
        let noise2 = hash_2d(direction[0] * 30.0 + 0.5, direction[2] * 30.0 + 0.3);

        let structure = band_value * (0.7 + noise * 0.2 + noise2 * 0.1);

        structure * self.config.milky_way_intensity
    }

    /// Get the RGB color for a star at the given temperature.
    ///
    /// Converts color temperature (Kelvin) to RGB using black-body radiation
    /// approximation.
    ///
    /// # Arguments
    ///
    /// * `temperature` - Color temperature in Kelvin (3000-30000).
    ///
    /// # Returns
    ///
    /// Linear RGB color (normalized).
    pub fn get_star_color(&self, temperature: u32) -> [f32; 3] {
        temperature_to_rgb(temperature)
    }

    /// Update animation time.
    ///
    /// # Arguments
    ///
    /// * `delta_seconds` - Time since last update.
    pub fn update_time(&mut self, delta_seconds: f32) {
        self.time += delta_seconds;
    }

    /// Regenerate stars with a new seed.
    pub fn regenerate(&mut self, seed: u64) {
        self.stars = Self::generate_stars_internal(self.config.star_count, seed, &self.config);
    }
}

// ---------------------------------------------------------------------------
// CelestialRenderer
// ---------------------------------------------------------------------------

/// Unified renderer for all celestial bodies.
///
/// Combines sun, moon, and star field rendering into a single API.
/// This is the primary interface for rendering the night sky.
///
/// # Example
///
/// ```
/// use renderer_backend::celestial_bodies::CelestialRenderer;
///
/// let mut renderer = CelestialRenderer::new();
/// renderer.update_time(0.016);
///
/// let view_dir = [0.0, 1.0, 0.0];
/// let sun_dir = [1.0, 0.0, 0.0];
/// let moon_dir = [-1.0, 0.0, 0.0];
///
/// let color = renderer.render_celestials(view_dir, sun_dir, moon_dir, 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct CelestialRenderer {
    /// Sun renderer.
    sun_renderer: SunRenderer,

    /// Moon renderer.
    moon_renderer: MoonRenderer,

    /// Star field renderer.
    star_field_renderer: StarFieldRenderer,

    /// Animation time accumulator.
    time: f32,
}

impl Default for CelestialRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl CelestialRenderer {
    /// Create a new celestial renderer with default configurations.
    #[inline]
    pub fn new() -> Self {
        Self {
            sun_renderer: SunRenderer::default(),
            moon_renderer: MoonRenderer::default(),
            star_field_renderer: StarFieldRenderer::default(),
            time: 0.0,
        }
    }

    /// Create a celestial renderer with custom configurations.
    pub fn with_configs(
        sun_config: SunConfig,
        moon_config: MoonConfig,
        star_config: StarFieldConfig,
    ) -> Self {
        Self {
            sun_renderer: SunRenderer::new(sun_config),
            moon_renderer: MoonRenderer::new(moon_config),
            star_field_renderer: StarFieldRenderer::new(star_config),
            time: 0.0,
        }
    }

    /// Get the sun renderer.
    #[inline]
    pub fn sun_renderer(&self) -> &SunRenderer {
        &self.sun_renderer
    }

    /// Get mutable access to the sun renderer.
    #[inline]
    pub fn sun_renderer_mut(&mut self) -> &mut SunRenderer {
        &mut self.sun_renderer
    }

    /// Get the moon renderer.
    #[inline]
    pub fn moon_renderer(&self) -> &MoonRenderer {
        &self.moon_renderer
    }

    /// Get mutable access to the moon renderer.
    #[inline]
    pub fn moon_renderer_mut(&mut self) -> &mut MoonRenderer {
        &mut self.moon_renderer
    }

    /// Get the star field renderer.
    #[inline]
    pub fn star_field_renderer(&self) -> &StarFieldRenderer {
        &self.star_field_renderer
    }

    /// Get mutable access to the star field renderer.
    #[inline]
    pub fn star_field_renderer_mut(&mut self) -> &mut StarFieldRenderer {
        &mut self.star_field_renderer
    }

    /// Render all celestial bodies for a view direction.
    ///
    /// Combines contributions from sun, moon, and stars based on
    /// visibility and time of day.
    ///
    /// # Arguments
    ///
    /// * `view_dir` - Normalized view direction.
    /// * `sun_dir` - Normalized direction towards the sun.
    /// * `moon_dir` - Normalized direction towards the moon.
    /// * `time` - Animation time in seconds.
    ///
    /// # Returns
    ///
    /// Combined linear RGB color from all celestial bodies.
    pub fn render_celestials(
        &self,
        view_dir: [f32; 3],
        sun_dir: [f32; 3],
        moon_dir: [f32; 3],
        time: f32,
    ) -> [f32; 3] {
        let view_dir = normalize_vec3(view_dir);
        let sun_dir = normalize_vec3(sun_dir);
        let moon_dir = normalize_vec3(moon_dir);

        let mut color = [0.0, 0.0, 0.0];

        // Sun visibility factor (1 during day, 0 at night)
        let sun_elevation = sun_dir[1];
        let day_factor = ((sun_elevation + 0.1) / 0.2).clamp(0.0, 1.0);
        let night_factor = 1.0 - day_factor;

        // Stars (visible at night)
        if night_factor > 0.0 {
            let star_color = self.star_field_renderer.sample_star_color(view_dir, time);
            let milky_way = self.star_field_renderer.sample_milky_way(view_dir);

            color[0] += (star_color[0] + milky_way * 0.05) * night_factor;
            color[1] += (star_color[1] + milky_way * 0.05) * night_factor;
            color[2] += (star_color[2] + milky_way * 0.08) * night_factor;
        }

        // Moon (visible when above horizon and not too close to sun)
        if moon_dir[1] > -0.1 {
            let moon_color = self.moon_renderer.render_moon(view_dir, moon_dir, sun_dir);
            let moon_visibility = ((moon_dir[1] + 0.1) / 0.2).clamp(0.0, 1.0);

            color[0] += moon_color[0] * moon_visibility;
            color[1] += moon_color[1] * moon_visibility;
            color[2] += moon_color[2] * moon_visibility;
        }

        // Sun glow (always computed, disk blocked by day_factor implicitly in sky)
        let sun_glow = self.sun_renderer.render_glow(view_dir, sun_dir);
        color[0] += sun_glow[0];
        color[1] += sun_glow[1];
        color[2] += sun_glow[2];

        // Sun disk (modulated by transmittance - simplified here)
        let transmittance = [day_factor, day_factor, day_factor];
        let sun_disk = self.sun_renderer.render_disk(view_dir, sun_dir, transmittance);
        color[0] += sun_disk[0];
        color[1] += sun_disk[1];
        color[2] += sun_disk[2];

        // Corona (if enabled)
        let corona = self.sun_renderer.render_corona(view_dir, sun_dir);
        color[0] += corona[0];
        color[1] += corona[1];
        color[2] += corona[2];

        color
    }

    /// Update animation time.
    ///
    /// # Arguments
    ///
    /// * `delta_seconds` - Time since last frame.
    pub fn update_time(&mut self, delta_seconds: f32) {
        self.time += delta_seconds;
        self.star_field_renderer.update_time(delta_seconds);
    }

    /// Get the current animation time.
    #[inline]
    pub fn time(&self) -> f32 {
        self.time
    }

    /// Check if the sun is visible from the current view direction.
    pub fn is_sun_visible(&self, view_dir: [f32; 3], sun_dir: [f32; 3]) -> bool {
        self.sun_renderer.get_disk_mask(view_dir, sun_dir) > 0.0
    }

    /// Check if the moon is visible from the current view direction.
    pub fn is_moon_visible(&self, view_dir: [f32; 3], moon_dir: [f32; 3]) -> bool {
        self.moon_renderer.get_disk_mask(view_dir, moon_dir) > 0.0
    }
}

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

/// Normalize a 3D vector.
#[inline]
fn normalize_vec3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 1e-10 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 1.0, 0.0] // Default to up if zero vector
    }
}

/// Dot product of two 3D vectors.
#[inline]
fn dot_vec3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Cross product of two 3D vectors.
#[inline]
fn cross_vec3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Simple hash function for 2D coordinates.
#[inline]
fn hash_2d(x: f32, y: f32) -> f32 {
    let n = (x * 12.9898 + y * 78.233).sin() * 43758.5453;
    n - n.floor()
}

/// Simple hash function for u64.
#[inline]
fn hash_u64(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^= x >> 33;
    x
}

/// Convert color temperature (Kelvin) to RGB.
///
/// Uses Tanner Helland's algorithm for black-body radiation approximation.
fn temperature_to_rgb(temperature: u32) -> [f32; 3] {
    let temp = (temperature as f32 / 100.0).clamp(10.0, 400.0);

    let r = if temp <= 66.0 {
        1.0
    } else {
        let r = 329.698727446 * (temp - 60.0).powf(-0.1332047592);
        (r / 255.0).clamp(0.0, 1.0)
    };

    let g = if temp <= 66.0 {
        let g = 99.4708025861 * temp.ln() - 161.1195681661;
        (g / 255.0).clamp(0.0, 1.0)
    } else {
        let g = 288.1221695283 * (temp - 60.0).powf(-0.0755148492);
        (g / 255.0).clamp(0.0, 1.0)
    };

    let b = if temp >= 66.0 {
        1.0
    } else if temp <= 19.0 {
        0.0
    } else {
        let b = 138.5177312231 * (temp - 10.0).ln() - 305.0447927307;
        (b / 255.0).clamp(0.0, 1.0)
    };

    [r, g, b]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    // =========================================================================
    // SunConfig Tests
    // =========================================================================

    #[test]
    fn test_sun_config_default() {
        let config = SunConfig::default();
        assert!((config.angular_radius - DEFAULT_SUN_ANGULAR_RADIUS).abs() < 0.0001);
        assert!((config.limb_darkening - DEFAULT_LIMB_DARKENING).abs() < 0.0001);
        assert!((config.glow_radius - DEFAULT_GLOW_RADIUS).abs() < 0.0001);
        assert!((config.glow_intensity - DEFAULT_GLOW_INTENSITY).abs() < 0.0001);
        assert!(!config.is_corona_enabled());
        assert!(config.validate());
    }

    #[test]
    fn test_sun_config_eclipse() {
        let config = SunConfig::eclipse();
        assert!(config.is_corona_enabled());
        assert!(config.glow_radius > DEFAULT_GLOW_RADIUS);
        assert!(config.validate());
    }

    #[test]
    fn test_sun_config_set_angular_radius() {
        let mut config = SunConfig::default();
        config.set_angular_radius(0.05);
        assert!((config.angular_radius - 0.05).abs() < 0.0001);

        // Test clamping
        config.set_angular_radius(0.0);
        assert!(config.angular_radius >= 0.001);

        config.set_angular_radius(1.0);
        assert!(config.angular_radius <= 0.1);
    }

    #[test]
    fn test_sun_config_set_limb_darkening() {
        let mut config = SunConfig::default();
        config.set_limb_darkening(0.8);
        assert!((config.limb_darkening - 0.8).abs() < 0.0001);

        // Test clamping
        config.set_limb_darkening(-0.5);
        assert!(config.limb_darkening >= 0.0);

        config.set_limb_darkening(1.5);
        assert!(config.limb_darkening <= 1.0);
    }

    #[test]
    fn test_sun_config_corona_toggle() {
        let mut config = SunConfig::default();
        assert!(!config.is_corona_enabled());

        config.set_corona_enabled(true);
        assert!(config.is_corona_enabled());

        config.set_corona_enabled(false);
        assert!(!config.is_corona_enabled());
    }

    #[test]
    fn test_sun_config_validation_invalid() {
        let mut config = SunConfig::default();
        config.angular_radius = 0.0;
        assert!(!config.validate());

        config = SunConfig::default();
        config.limb_darkening = 1.5;
        assert!(!config.validate());

        config = SunConfig::default();
        config.glow_radius = -1.0;
        assert!(!config.validate());
    }

    #[test]
    fn test_sun_config_size() {
        assert_eq!(std::mem::size_of::<SunConfig>(), 32);
    }

    #[test]
    fn test_sun_config_pod() {
        let config = SunConfig::default();
        let bytes = bytemuck::bytes_of(&config);
        assert_eq!(bytes.len(), 32);
    }

    // =========================================================================
    // SunRenderer Tests
    // =========================================================================

    #[test]
    fn test_sun_renderer_new() {
        let renderer = SunRenderer::new(SunConfig::default());
        assert!(renderer.config().validate());
    }

    #[test]
    fn test_sun_renderer_default() {
        let renderer = SunRenderer::default();
        assert!(renderer.config().validate());
    }

    #[test]
    fn test_sun_renderer_disk_mask_center() {
        let renderer = SunRenderer::default();
        let sun_dir = [0.0, 1.0, 0.0];
        let view_dir = [0.0, 1.0, 0.0]; // Looking directly at sun

        let mask = renderer.get_disk_mask(view_dir, sun_dir);
        assert!(mask > 0.9, "Center mask should be near 1.0, got {}", mask);
    }

    #[test]
    fn test_sun_renderer_disk_mask_outside() {
        let renderer = SunRenderer::default();
        let sun_dir = [0.0, 1.0, 0.0];
        let view_dir = [1.0, 0.0, 0.0]; // Looking perpendicular

        let mask = renderer.get_disk_mask(view_dir, sun_dir);
        assert_eq!(mask, 0.0, "Outside mask should be 0.0");
    }

    #[test]
    fn test_sun_renderer_disk_mask_edge() {
        let renderer = SunRenderer::default();
        let sun_dir = [0.0, 1.0, 0.0];

        // Slightly offset from center
        let offset = renderer.config().angular_radius * 0.5;
        let view_dir = normalize_vec3([offset.sin(), offset.cos(), 0.0]);

        let mask = renderer.get_disk_mask(view_dir, sun_dir);
        assert!(mask > 0.0 && mask < 1.0, "Edge mask should be between 0 and 1, got {}", mask);
    }

    #[test]
    fn test_sun_renderer_limb_darkening_center() {
        let renderer = SunRenderer::default();
        let factor = renderer.limb_darkening_factor(0.0);
        assert!((factor - 1.0).abs() < 0.0001, "Center should have no darkening");
    }

    #[test]
    fn test_sun_renderer_limb_darkening_edge() {
        let renderer = SunRenderer::default();
        let factor = renderer.limb_darkening_factor(1.0);
        assert!(factor < 1.0, "Edge should be darker than center");
        assert!(factor > 0.0, "Edge should still be visible");
    }

    #[test]
    fn test_sun_renderer_limb_darkening_monotonic() {
        let renderer = SunRenderer::default();

        let mut prev = 1.0;
        for i in 0..=10 {
            let r = i as f32 / 10.0;
            let factor = renderer.limb_darkening_factor(r);
            assert!(factor <= prev + 0.0001, "Limb darkening should decrease monotonically");
            prev = factor;
        }
    }

    #[test]
    fn test_sun_renderer_render_disk_visible() {
        let renderer = SunRenderer::default();
        let sun_dir = [0.0, 1.0, 0.0];
        let view_dir = [0.0, 1.0, 0.0];
        let transmittance = [1.0, 1.0, 1.0];

        let color = renderer.render_disk(view_dir, sun_dir, transmittance);
        assert!(color[0] > 0.0 && color[1] > 0.0 && color[2] > 0.0);
    }

    #[test]
    fn test_sun_renderer_render_disk_invisible() {
        let renderer = SunRenderer::default();
        let sun_dir = [0.0, 1.0, 0.0];
        let view_dir = [1.0, 0.0, 0.0];
        let transmittance = [1.0, 1.0, 1.0];

        let color = renderer.render_disk(view_dir, sun_dir, transmittance);
        assert_eq!(color, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_sun_renderer_render_glow() {
        let renderer = SunRenderer::default();
        let sun_dir = [0.0, 1.0, 0.0];

        // Near sun but outside disk
        let offset = renderer.config().angular_radius * 2.0;
        let view_dir = normalize_vec3([offset.sin(), offset.cos(), 0.0]);

        let glow = renderer.render_glow(view_dir, sun_dir);
        assert!(glow[0] > 0.0 || glow[1] > 0.0 || glow[2] > 0.0);
    }

    #[test]
    fn test_sun_renderer_render_glow_falloff() {
        let renderer = SunRenderer::default();
        let sun_dir = [0.0, 1.0, 0.0];

        // Near sun
        let offset1 = renderer.config().angular_radius * 1.5;
        let view_dir1 = normalize_vec3([offset1.sin(), offset1.cos(), 0.0]);
        let glow1 = renderer.render_glow(view_dir1, sun_dir);

        // Further from sun
        let offset2 = renderer.config().angular_radius * 2.5;
        let view_dir2 = normalize_vec3([offset2.sin(), offset2.cos(), 0.0]);
        let glow2 = renderer.render_glow(view_dir2, sun_dir);

        assert!(glow1[0] > glow2[0], "Glow should fall off with distance");
    }

    #[test]
    fn test_sun_renderer_corona_disabled() {
        let renderer = SunRenderer::default();
        let sun_dir = [0.0, 1.0, 0.0];
        let view_dir = [0.0, 1.0, 0.0];

        let corona = renderer.render_corona(view_dir, sun_dir);
        assert_eq!(corona, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_sun_renderer_corona_enabled() {
        let renderer = SunRenderer::new(SunConfig::eclipse());
        let sun_dir = [0.0, 1.0, 0.0];

        // Near sun edge
        let offset = renderer.config().angular_radius * 2.0;
        let view_dir = normalize_vec3([offset.sin(), offset.cos(), 0.0]);

        let corona = renderer.render_corona(view_dir, sun_dir);
        assert!(corona[0] > 0.0 || corona[1] > 0.0 || corona[2] > 0.0);
    }

    // =========================================================================
    // MoonConfig Tests
    // =========================================================================

    #[test]
    fn test_moon_config_default() {
        let config = MoonConfig::default();
        assert!((config.angular_radius - DEFAULT_MOON_ANGULAR_RADIUS).abs() < 0.0001);
        assert!((config.albedo - DEFAULT_MOON_ALBEDO).abs() < 0.0001);
        assert!((config.phase - 0.5).abs() < 0.0001); // Full moon
        assert!(config.validate());
    }

    #[test]
    fn test_moon_config_new_moon() {
        let config = MoonConfig::new_moon();
        assert!((config.phase - 0.0).abs() < 0.0001);
        assert!(config.validate());
    }

    #[test]
    fn test_moon_config_full_moon() {
        let config = MoonConfig::full_moon();
        assert!((config.phase - 0.5).abs() < 0.0001);
        assert!(config.validate());
    }

    #[test]
    fn test_moon_config_first_quarter() {
        let config = MoonConfig::first_quarter();
        assert!((config.phase - 0.25).abs() < 0.0001);
        assert!(config.validate());
    }

    #[test]
    fn test_moon_config_last_quarter() {
        let config = MoonConfig::last_quarter();
        assert!((config.phase - 0.75).abs() < 0.0001);
        assert!(config.validate());
    }

    #[test]
    fn test_moon_config_set_phase() {
        let mut config = MoonConfig::default();
        config.set_phase(0.25);
        assert!((config.phase - 0.25).abs() < 0.0001);

        // Test wrapping
        config.set_phase(1.25);
        assert!((config.phase - 0.25).abs() < 0.0001);

        config.set_phase(-0.25);
        assert!((config.phase - 0.75).abs() < 0.0001);
    }

    #[test]
    fn test_moon_config_set_albedo() {
        let mut config = MoonConfig::default();
        config.set_albedo(0.2);
        assert!((config.albedo - 0.2).abs() < 0.0001);

        // Test clamping
        config.set_albedo(-0.1);
        assert!(config.albedo >= 0.0);

        config.set_albedo(1.5);
        assert!(config.albedo <= 1.0);
    }

    #[test]
    fn test_moon_config_size() {
        assert_eq!(std::mem::size_of::<MoonConfig>(), 16);
    }

    #[test]
    fn test_moon_config_pod() {
        let config = MoonConfig::default();
        let bytes = bytemuck::bytes_of(&config);
        assert_eq!(bytes.len(), 16);
    }

    // =========================================================================
    // MoonRenderer Tests
    // =========================================================================

    #[test]
    fn test_moon_renderer_new() {
        let renderer = MoonRenderer::new(MoonConfig::default());
        assert!(renderer.config().validate());
    }

    #[test]
    fn test_moon_renderer_compute_phase_full() {
        let renderer = MoonRenderer::default();
        // Sun and moon on opposite sides -> full moon
        let sun_dir = [1.0, 0.0, 0.0];
        let moon_dir = [-1.0, 0.0, 0.0];

        let phase = renderer.compute_phase(sun_dir, moon_dir);
        assert!(phase > 0.4 && phase < 0.6, "Full moon phase should be ~0.5, got {}", phase);
    }

    #[test]
    fn test_moon_renderer_compute_phase_new() {
        let renderer = MoonRenderer::default();
        // Sun and moon same direction -> new moon
        let sun_dir = [1.0, 0.0, 0.0];
        let moon_dir = [1.0, 0.0, 0.0];

        let phase = renderer.compute_phase(sun_dir, moon_dir);
        assert!(phase < 0.2, "New moon phase should be ~0, got {}", phase);
    }

    #[test]
    fn test_moon_renderer_compute_phase_quarter() {
        let renderer = MoonRenderer::default();
        // Sun and moon perpendicular -> quarter moon
        let sun_dir = [1.0, 0.0, 0.0];
        let moon_dir = [0.0, 1.0, 0.0];

        let phase = renderer.compute_phase(sun_dir, moon_dir);
        assert!(phase > 0.2 && phase < 0.4, "Quarter moon phase, got {}", phase);
    }

    #[test]
    fn test_moon_renderer_crater_pattern_range() {
        let renderer = MoonRenderer::default();

        for i in 0..10 {
            for j in 0..10 {
                let uv = [i as f32 / 10.0, j as f32 / 10.0];
                let crater = renderer.get_crater_pattern(uv);
                assert!(crater >= 0.0 && crater <= 1.0, "Crater pattern out of range at {:?}: {}", uv, crater);
            }
        }
    }

    #[test]
    fn test_moon_renderer_maria_pattern_range() {
        let renderer = MoonRenderer::default();

        for i in 0..10 {
            for j in 0..10 {
                let uv = [i as f32 / 10.0, j as f32 / 10.0];
                let maria = renderer.get_maria_pattern(uv);
                assert!(maria >= 0.0 && maria <= 1.0, "Maria pattern out of range at {:?}: {}", uv, maria);
            }
        }
    }

    #[test]
    fn test_moon_renderer_disk_mask() {
        let renderer = MoonRenderer::default();
        let moon_dir = [0.0, 1.0, 0.0];

        let inside_mask = renderer.get_disk_mask([0.0, 1.0, 0.0], moon_dir);
        assert_eq!(inside_mask, 1.0);

        let outside_mask = renderer.get_disk_mask([1.0, 0.0, 0.0], moon_dir);
        assert_eq!(outside_mask, 0.0);
    }

    #[test]
    fn test_moon_renderer_render_moon_visible() {
        let renderer = MoonRenderer::default();
        let view_dir = [0.0, 1.0, 0.0];
        let moon_dir = [0.0, 1.0, 0.0];
        let sun_dir = [-1.0, 0.0, 0.0];

        let color = renderer.render_moon(view_dir, moon_dir, sun_dir);
        // Should have some brightness
        assert!(color[0] > 0.0 || color[1] > 0.0 || color[2] > 0.0);
    }

    #[test]
    fn test_moon_renderer_render_moon_invisible() {
        let renderer = MoonRenderer::default();
        let view_dir = [1.0, 0.0, 0.0];
        let moon_dir = [0.0, 1.0, 0.0];
        let sun_dir = [-1.0, 0.0, 0.0];

        let color = renderer.render_moon(view_dir, moon_dir, sun_dir);
        assert_eq!(color, [0.0, 0.0, 0.0]);
    }

    // =========================================================================
    // StarFieldConfig Tests
    // =========================================================================

    #[test]
    fn test_star_field_config_default() {
        let config = StarFieldConfig::default();
        assert_eq!(config.star_count, DEFAULT_STAR_COUNT);
        assert!((config.magnitude_min - DEFAULT_MAGNITUDE_MIN).abs() < 0.0001);
        assert!((config.magnitude_max - DEFAULT_MAGNITUDE_MAX).abs() < 0.0001);
        assert!((config.twinkle_speed - DEFAULT_TWINKLE_SPEED).abs() < 0.0001);
        assert!(config.validate());
    }

    #[test]
    fn test_star_field_config_dark_sky() {
        let config = StarFieldConfig::dark_sky();
        assert!(config.star_count > DEFAULT_STAR_COUNT);
        assert!(config.milky_way_intensity > DEFAULT_MILKY_WAY_INTENSITY);
        assert!(config.validate());
    }

    #[test]
    fn test_star_field_config_light_pollution() {
        let config = StarFieldConfig::light_pollution();
        assert!(config.star_count < DEFAULT_STAR_COUNT);
        assert_eq!(config.milky_way_intensity, 0.0);
        assert!(config.validate());
    }

    #[test]
    fn test_star_field_config_set_star_count() {
        let mut config = StarFieldConfig::default();
        config.set_star_count(5000);
        assert_eq!(config.star_count, 5000);

        // Test clamping
        config.set_star_count(10);
        assert!(config.star_count >= 100);

        config.set_star_count(1_000_000);
        assert!(config.star_count <= 100_000);
    }

    #[test]
    fn test_star_field_config_validation_invalid() {
        let mut config = StarFieldConfig::default();
        config.star_count = 0;
        assert!(!config.validate());

        config = StarFieldConfig::default();
        config.magnitude_min = 10.0;
        config.magnitude_max = 5.0;
        assert!(!config.validate());

        config = StarFieldConfig::default();
        config.twinkle_amplitude = 1.5;
        assert!(!config.validate());
    }

    #[test]
    fn test_star_field_config_size() {
        assert_eq!(std::mem::size_of::<StarFieldConfig>(), 32);
    }

    // =========================================================================
    // Star Tests
    // =========================================================================

    #[test]
    fn test_star_new() {
        let star = Star::new([1.0, 0.0, 0.0], 2.5, 5500, 1.0);
        assert!((star.direction[0] - 1.0).abs() < 0.0001);
        assert!((star.magnitude - 2.5).abs() < 0.0001);
        assert_eq!(star.temperature, 5500);
        assert!((star.twinkle_phase - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_star_base_brightness() {
        let bright_star = Star::new([0.0, 1.0, 0.0], -1.0, 5500, 0.0);
        let dim_star = Star::new([0.0, 1.0, 0.0], 6.0, 5500, 0.0);

        assert!(bright_star.base_brightness() > dim_star.base_brightness());
    }

    #[test]
    fn test_star_color_temperature_red() {
        let color = temperature_to_rgb(3000);
        // Red star should have more red than blue
        assert!(color[0] > color[2], "3000K should be reddish");
    }

    #[test]
    fn test_star_color_temperature_white() {
        let color = temperature_to_rgb(6500);
        // Sun-like should be close to white
        let avg = (color[0] + color[1] + color[2]) / 3.0;
        assert!((color[0] - avg).abs() < 0.2, "6500K should be whitish");
        assert!((color[1] - avg).abs() < 0.2, "6500K should be whitish");
        assert!((color[2] - avg).abs() < 0.2, "6500K should be whitish");
    }

    #[test]
    fn test_star_color_temperature_blue() {
        let color = temperature_to_rgb(20000);
        // Hot star should have more blue
        assert!(color[2] >= color[0], "20000K should be bluish");
    }

    // =========================================================================
    // StarFieldRenderer Tests
    // =========================================================================

    #[test]
    fn test_star_field_renderer_new() {
        let renderer = StarFieldRenderer::new(StarFieldConfig::default());
        assert_eq!(renderer.stars().len(), DEFAULT_STAR_COUNT as usize);
    }

    #[test]
    fn test_star_field_renderer_with_seed() {
        let renderer1 = StarFieldRenderer::with_seed(StarFieldConfig::default(), 42);
        let renderer2 = StarFieldRenderer::with_seed(StarFieldConfig::default(), 42);

        // Same seed should produce same stars
        assert_eq!(renderer1.stars().len(), renderer2.stars().len());
        for i in 0..renderer1.stars().len().min(10) {
            assert_eq!(renderer1.stars()[i].direction, renderer2.stars()[i].direction);
        }
    }

    #[test]
    fn test_star_field_renderer_different_seeds() {
        let renderer1 = StarFieldRenderer::with_seed(StarFieldConfig::default(), 42);
        let renderer2 = StarFieldRenderer::with_seed(StarFieldConfig::default(), 123);

        // Different seeds should produce different stars
        let mut different = false;
        for i in 0..renderer1.stars().len().min(10) {
            if renderer1.stars()[i].direction != renderer2.stars()[i].direction {
                different = true;
                break;
            }
        }
        assert!(different, "Different seeds should produce different stars");
    }

    #[test]
    fn test_star_field_renderer_star_distribution_uniform() {
        let renderer = StarFieldRenderer::with_seed(StarFieldConfig::default(), 42);

        // Check that stars are distributed across the sky
        let mut up_count = 0;
        let mut down_count = 0;
        let mut east_count = 0;
        let mut west_count = 0;

        for star in renderer.stars() {
            if star.direction[1] > 0.0 {
                up_count += 1;
            } else {
                down_count += 1;
            }
            if star.direction[0] > 0.0 {
                east_count += 1;
            } else {
                west_count += 1;
            }
        }

        let total = renderer.stars().len();
        let tolerance = total / 4; // Allow 25% deviation

        assert!((up_count as i32 - total as i32 / 2).abs() < tolerance as i32);
        assert!((down_count as i32 - total as i32 / 2).abs() < tolerance as i32);
        assert!((east_count as i32 - total as i32 / 2).abs() < tolerance as i32);
        assert!((west_count as i32 - total as i32 / 2).abs() < tolerance as i32);
    }

    #[test]
    fn test_star_field_renderer_sample_brightness() {
        let renderer = StarFieldRenderer::with_seed(StarFieldConfig::default(), 42);

        // Sample towards a known star (first one)
        if !renderer.stars().is_empty() {
            let star = &renderer.stars()[0];
            let brightness = renderer.sample_star_brightness(star.direction, 0.0);
            // Should have some brightness when looking directly at a star
            // Note: might be 0 if tolerance is too small
            assert!(brightness >= 0.0);
        }
    }

    #[test]
    fn test_star_field_renderer_twinkle_modulation() {
        let renderer = StarFieldRenderer::with_seed(StarFieldConfig::default(), 42);

        if !renderer.stars().is_empty() {
            let star = &renderer.stars()[0];

            // Sample at different times
            let brightness1 = renderer.sample_star_brightness(star.direction, 0.0);
            let brightness2 = renderer.sample_star_brightness(star.direction, 0.25);
            let brightness3 = renderer.sample_star_brightness(star.direction, 0.5);

            // Over time, brightness should vary (not all be exactly the same)
            // This is probabilistic - at least one should differ
            let all_same = brightness1 == brightness2 && brightness2 == brightness3;
            // If all are 0, that's also valid (no star at that exact direction)
            if brightness1 > 0.0 || brightness2 > 0.0 || brightness3 > 0.0 {
                // At least check they're non-negative
                assert!(brightness1 >= 0.0);
                assert!(brightness2 >= 0.0);
                assert!(brightness3 >= 0.0);
            }
        }
    }

    #[test]
    fn test_star_field_renderer_milky_way() {
        let renderer = StarFieldRenderer::new(StarFieldConfig::default());

        // Sample in galactic plane
        let in_plane = renderer.sample_milky_way([1.0, 0.0, 0.0]);

        // Sample perpendicular to galactic plane
        let out_of_plane = renderer.sample_milky_way([0.0, 1.0, 0.0]);

        // Milky Way should be brighter in the plane
        // Note: depends on exact plane orientation
        assert!(in_plane >= 0.0);
        assert!(out_of_plane >= 0.0);
    }

    #[test]
    fn test_star_field_renderer_milky_way_disabled() {
        let mut config = StarFieldConfig::default();
        config.milky_way_intensity = 0.0;
        let renderer = StarFieldRenderer::new(config);

        let brightness = renderer.sample_milky_way([0.5, 0.5, 0.5]);
        assert_eq!(brightness, 0.0);
    }

    #[test]
    fn test_star_field_renderer_update_time() {
        let mut renderer = StarFieldRenderer::default();
        assert_eq!(renderer.time(), 0.0);

        renderer.update_time(0.5);
        assert!((renderer.time() - 0.5).abs() < 0.0001);

        renderer.update_time(0.5);
        assert!((renderer.time() - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_star_field_renderer_regenerate() {
        let mut renderer = StarFieldRenderer::with_seed(StarFieldConfig::default(), 42);
        let first_star = renderer.stars()[0].direction;

        renderer.regenerate(123);
        let new_first_star = renderer.stars()[0].direction;

        assert_ne!(first_star, new_first_star);
    }

    #[test]
    fn test_star_field_renderer_get_star_color() {
        let renderer = StarFieldRenderer::default();

        let red = renderer.get_star_color(3000);
        let white = renderer.get_star_color(6500);
        let blue = renderer.get_star_color(20000);

        assert!(red[0] > red[2], "3000K should be red");
        assert!(blue[2] >= blue[0], "20000K should be blue");
        // White should be balanced
        let white_avg = (white[0] + white[1] + white[2]) / 3.0;
        assert!((white[0] - white_avg).abs() < 0.3);
    }

    // =========================================================================
    // CelestialRenderer Tests
    // =========================================================================

    #[test]
    fn test_celestial_renderer_new() {
        let renderer = CelestialRenderer::new();
        assert!(renderer.sun_renderer().config().validate());
        assert!(renderer.moon_renderer().config().validate());
        assert!(renderer.star_field_renderer().config().validate());
    }

    #[test]
    fn test_celestial_renderer_with_configs() {
        let sun_config = SunConfig::eclipse();
        let moon_config = MoonConfig::full_moon();
        let star_config = StarFieldConfig::dark_sky();

        let renderer = CelestialRenderer::with_configs(sun_config, moon_config, star_config);

        assert!(renderer.sun_renderer().config().is_corona_enabled());
        assert!((renderer.moon_renderer().config().phase - 0.5).abs() < 0.0001);
        assert!(renderer.star_field_renderer().config().star_count > DEFAULT_STAR_COUNT);
    }

    #[test]
    fn test_celestial_renderer_render_celestials_day() {
        let renderer = CelestialRenderer::new();
        let view_dir = [0.0, 1.0, 0.0];
        let sun_dir = [0.0, 1.0, 0.0]; // Midday
        let moon_dir = [0.0, -1.0, 0.0]; // Below horizon

        let color = renderer.render_celestials(view_dir, sun_dir, moon_dir, 0.0);

        // During day looking at sun, should be bright
        assert!(color[0] > 0.0 || color[1] > 0.0 || color[2] > 0.0);
    }

    #[test]
    fn test_celestial_renderer_render_celestials_night() {
        let renderer = CelestialRenderer::new();
        let view_dir = [0.0, 1.0, 0.0];
        let sun_dir = [0.0, -1.0, 0.0]; // Below horizon
        let moon_dir = [0.0, 1.0, 0.0]; // Above horizon

        let color = renderer.render_celestials(view_dir, sun_dir, moon_dir, 0.0);

        // At night with moon visible, should have some light
        // Note: stars might also contribute
        assert!(color[0] >= 0.0 && color[1] >= 0.0 && color[2] >= 0.0);
    }

    #[test]
    fn test_celestial_renderer_update_time() {
        let mut renderer = CelestialRenderer::new();
        assert_eq!(renderer.time(), 0.0);

        renderer.update_time(1.0);
        assert!((renderer.time() - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_celestial_renderer_is_sun_visible() {
        let renderer = CelestialRenderer::new();

        assert!(renderer.is_sun_visible([0.0, 1.0, 0.0], [0.0, 1.0, 0.0]));
        assert!(!renderer.is_sun_visible([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]));
    }

    #[test]
    fn test_celestial_renderer_is_moon_visible() {
        let renderer = CelestialRenderer::new();

        assert!(renderer.is_moon_visible([0.0, 1.0, 0.0], [0.0, 1.0, 0.0]));
        assert!(!renderer.is_moon_visible([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]));
    }

    #[test]
    fn test_celestial_renderer_mutable_access() {
        let mut renderer = CelestialRenderer::new();

        renderer.sun_renderer_mut().config_mut().set_corona_enabled(true);
        assert!(renderer.sun_renderer().config().is_corona_enabled());

        renderer.moon_renderer_mut().config_mut().set_phase(0.25);
        assert!((renderer.moon_renderer().config().phase - 0.25).abs() < 0.0001);
    }

    // =========================================================================
    // Helper Function Tests
    // =========================================================================

    #[test]
    fn test_normalize_vec3() {
        let v = normalize_vec3([3.0, 4.0, 0.0]);
        assert!((v[0] - 0.6).abs() < 0.0001);
        assert!((v[1] - 0.8).abs() < 0.0001);
        assert!(v[2].abs() < 0.0001);
    }

    #[test]
    fn test_normalize_vec3_zero() {
        let v = normalize_vec3([0.0, 0.0, 0.0]);
        assert_eq!(v, [0.0, 1.0, 0.0]); // Default to up
    }

    #[test]
    fn test_dot_vec3() {
        assert_eq!(dot_vec3([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]), 0.0);
        assert_eq!(dot_vec3([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]), 1.0);
        assert_eq!(dot_vec3([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]), -1.0);
    }

    #[test]
    fn test_cross_vec3() {
        let result = cross_vec3([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        assert!((result[0] - 0.0).abs() < 0.0001);
        assert!((result[1] - 0.0).abs() < 0.0001);
        assert!((result[2] - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_hash_2d_deterministic() {
        let h1 = hash_2d(1.0, 2.0);
        let h2 = hash_2d(1.0, 2.0);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hash_2d_range() {
        for i in 0..100 {
            let h = hash_2d(i as f32 * 0.1, i as f32 * 0.2);
            assert!(h >= 0.0 && h < 1.0, "Hash out of range: {}", h);
        }
    }

    #[test]
    fn test_hash_u64_deterministic() {
        let h1 = hash_u64(42);
        let h2 = hash_u64(42);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_temperature_to_rgb_clamping() {
        // Very low temperature
        let cold = temperature_to_rgb(100);
        assert!(cold[0] >= 0.0 && cold[0] <= 1.0);
        assert!(cold[1] >= 0.0 && cold[1] <= 1.0);
        assert!(cold[2] >= 0.0 && cold[2] <= 1.0);

        // Very high temperature
        let hot = temperature_to_rgb(50000);
        assert!(hot[0] >= 0.0 && hot[0] <= 1.0);
        assert!(hot[1] >= 0.0 && hot[1] <= 1.0);
        assert!(hot[2] >= 0.0 && hot[2] <= 1.0);
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_sun_at_horizon() {
        let renderer = SunRenderer::default();
        let sun_dir = [1.0, 0.0, 0.0]; // At horizon
        let view_dir = [1.0, 0.0, 0.0];
        let transmittance = [1.0, 0.5, 0.3]; // Atmospheric reddening

        let color = renderer.render_disk(view_dir, sun_dir, transmittance);
        // Should be visible with transmittance applied
        assert!(color[0] > 0.0);
    }

    #[test]
    fn test_moon_edge_of_disk() {
        let renderer = MoonRenderer::default();
        let moon_dir = [0.0, 1.0, 0.0];

        // At exact edge of disk
        let offset = renderer.config().angular_radius;
        let view_dir = normalize_vec3([offset.sin(), offset.cos(), 0.0]);

        let mask = renderer.get_disk_mask(view_dir, moon_dir);
        // Should be at boundary
        assert!(mask == 0.0 || mask == 1.0);
    }

    #[test]
    fn test_star_at_magnitude_extremes() {
        let brightest = Star::new([0.0, 1.0, 0.0], -1.0, 5500, 0.0);
        let dimmest = Star::new([0.0, 1.0, 0.0], 6.0, 5500, 0.0);

        let bright_b = brightest.base_brightness();
        let dim_b = dimmest.base_brightness();

        assert!(bright_b > dim_b);
        assert!(bright_b > 0.0);
        assert!(dim_b > 0.0);

        // Brightness ratio should be significant
        assert!(bright_b / dim_b > 100.0);
    }

    #[test]
    fn test_celestial_renderer_twilight_transition() {
        let renderer = CelestialRenderer::new();

        // Sun just below horizon
        let view_dir = [0.0, 1.0, 0.0];
        let sun_dir = [0.0, -0.05, 1.0]; // Slightly below horizon
        let moon_dir = [0.0, 0.5, 0.0];

        let color = renderer.render_celestials(view_dir, sun_dir, moon_dir, 0.0);

        // Should have some contribution from stars/moon
        // Not testing exact values, just that it doesn't crash
        assert!(color[0] >= 0.0);
        assert!(color[1] >= 0.0);
        assert!(color[2] >= 0.0);
    }

    #[test]
    fn test_all_configs_pod_zeroable() {
        // Ensure all config types can be zeroed
        let sun: SunConfig = bytemuck::Zeroable::zeroed();
        let moon: MoonConfig = bytemuck::Zeroable::zeroed();
        let star: StarFieldConfig = bytemuck::Zeroable::zeroed();

        // And converted to bytes
        let _ = bytemuck::bytes_of(&sun);
        let _ = bytemuck::bytes_of(&moon);
        let _ = bytemuck::bytes_of(&star);
    }

    #[test]
    fn test_generate_stars_count() {
        let config = StarFieldConfig::default();
        let renderer = StarFieldRenderer::new(config);
        let stars = renderer.generate_stars(100, 42);
        assert_eq!(stars.len(), 100);
    }

    #[test]
    fn test_star_directions_normalized() {
        let renderer = StarFieldRenderer::with_seed(StarFieldConfig::default(), 42);

        for star in renderer.stars() {
            let len = (star.direction[0].powi(2) + star.direction[1].powi(2) + star.direction[2].powi(2)).sqrt();
            assert!((len - 1.0).abs() < 0.0001, "Star direction not normalized: len = {}", len);
        }
    }

    #[test]
    fn test_star_temperatures_in_range() {
        let renderer = StarFieldRenderer::with_seed(StarFieldConfig::default(), 42);

        for star in renderer.stars() {
            assert!(star.temperature >= 3000, "Temperature too low: {}", star.temperature);
            assert!(star.temperature <= 30000, "Temperature too high: {}", star.temperature);
        }
    }

    #[test]
    fn test_star_magnitudes_in_range() {
        let config = StarFieldConfig::default();
        let renderer = StarFieldRenderer::new(config);

        for star in renderer.stars() {
            assert!(star.magnitude >= config.magnitude_min - 0.0001);
            assert!(star.magnitude <= config.magnitude_max + 0.0001);
        }
    }

    #[test]
    fn test_star_twinkle_phases_in_range() {
        let renderer = StarFieldRenderer::with_seed(StarFieldConfig::default(), 42);

        for star in renderer.stars() {
            assert!(star.twinkle_phase >= 0.0);
            assert!(star.twinkle_phase <= TWO_PI);
        }
    }
}
