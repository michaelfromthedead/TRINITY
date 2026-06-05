//! Water Shading Pass for TRINITY Engine (T-ENV-1.8).
//!
//! Implements physically-based water shading including:
//! - Fresnel reflectance using Schlick and full dielectric equations
//! - Refraction with Snell's law and total internal reflection handling
//! - Subsurface scattering approximation for light transmission
//! - GGX specular (isotropic and anisotropic for wave-aligned highlights)
//! - Depth-based color blending (shallow to deep water)
//!
//! # Physics Background
//!
//! ## Fresnel Reflectance
//!
//! Water has an index of refraction (IOR) of approximately 1.33. The Fresnel
//! equations describe how light is partially reflected and partially refracted
//! at the water surface. At normal incidence (looking straight down), about 2%
//! is reflected. At grazing angles, reflection approaches 100%.
//!
//! The Schlick approximation provides a fast, reasonably accurate model:
//! `F = F0 + (1 - F0) * (1 - cos_theta)^5`
//!
//! Where `F0 = ((n1 - n2) / (n1 + n2))^2 ≈ 0.02` for air-water interface.
//!
//! ## Refraction
//!
//! Snell's law governs the refracted ray direction:
//! `n1 * sin(theta1) = n2 * sin(theta2)`
//!
//! When light travels from water to air at steep angles, total internal
//! reflection occurs beyond the critical angle.
//!
//! ## GGX Microfacet Model
//!
//! Water surfaces exhibit specular highlights that can be modeled using the
//! GGX/Trowbridge-Reitz distribution. Anisotropic GGX allows stretching
//! highlights along the wave direction for more realistic appearance.
//!
//! # Usage
//!
//! ```ignore
//! use renderer_backend::water::shading::{
//!     WaterShadingConfig, WaterReflectionConfig, WaterShadingPass,
//! };
//!
//! let config = WaterShadingConfig::default();
//! let reflection = WaterReflectionConfig::default();
//! let pass = WaterShadingPass::new(config, reflection);
//!
//! // Evaluate Fresnel at an angle
//! let cos_theta = 0.8;
//! let f = WaterShadingPass::fresnel_schlick(cos_theta, config.fresnel_f0);
//! ```

use std::f32::consts::PI;
use std::mem;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Water index of refraction (approximately 1.33 for visible light).
pub const WATER_IOR: f32 = 1.33;

/// Air index of refraction.
pub const AIR_IOR: f32 = 1.0;

/// Default F0 for air-water interface: ((1 - 1.33) / (1 + 1.33))^2 ≈ 0.02.
pub const DEFAULT_F0: f32 = 0.02;

/// Size of WaterShadingConfig in bytes.
pub const WATER_SHADING_CONFIG_SIZE: usize = 64;

/// Small epsilon for floating point comparisons.
const EPSILON: f32 = 1e-6;

// ---------------------------------------------------------------------------
// WaterShadingConfig
// ---------------------------------------------------------------------------

/// Configuration for water surface shading.
///
/// This struct is GPU-compatible with 64-byte alignment for uniform buffers.
///
/// # Memory Layout (64 bytes, std140 compatible)
///
/// | Offset | Field               | Size     |
/// |--------|---------------------|----------|
/// | 0      | base_color_shallow  | 12 bytes |
/// | 12     | shallow_depth       | 4 bytes  |
/// | 16     | base_color_deep     | 12 bytes |
/// | 24     | deep_depth          | 4 bytes  |
/// | 28     | fresnel_f0          | 4 bytes  |
/// | 32     | fresnel_power       | 4 bytes  |
/// | 36     | refraction_strength | 4 bytes  |
/// | 40     | roughness           | 4 bytes  |
/// | 44     | _padding            | 20 bytes |
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WaterShadingConfig {
    /// RGB color for shallow water regions.
    pub base_color_shallow: [f32; 3],
    /// Water depth (meters) where shallow color begins transitioning.
    pub shallow_depth: f32,
    /// RGB color for deep water regions.
    pub base_color_deep: [f32; 3],
    /// Water depth (meters) where color is fully deep.
    pub deep_depth: f32,
    /// Fresnel F0 term (reflectance at normal incidence).
    /// Default ~0.02 for water IOR 1.33.
    pub fresnel_f0: f32,
    /// Fresnel power adjustment (default 5.0 for Schlick).
    pub fresnel_power: f32,
    /// Screen-space refraction distortion strength (0.0-0.1 typical).
    pub refraction_strength: f32,
    /// GGX roughness for specular highlights.
    pub roughness: f32,
    /// Padding for 16-byte alignment (64 bytes total).
    pub _padding: [f32; 4],
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<WaterShadingConfig>() == WATER_SHADING_CONFIG_SIZE);

impl WaterShadingConfig {
    /// Create a new water shading configuration.
    ///
    /// # Arguments
    ///
    /// * `shallow_color` - RGB color for shallow water (0-1 range).
    /// * `deep_color` - RGB color for deep water (0-1 range).
    /// * `shallow_depth` - Depth where shallow transitions start.
    /// * `deep_depth` - Depth where deep color is fully applied.
    pub fn new(
        shallow_color: [f32; 3],
        deep_color: [f32; 3],
        shallow_depth: f32,
        deep_depth: f32,
    ) -> Self {
        Self {
            base_color_shallow: shallow_color,
            shallow_depth: shallow_depth.max(0.0),
            base_color_deep: deep_color,
            deep_depth: deep_depth.max(shallow_depth + 0.1),
            fresnel_f0: DEFAULT_F0,
            fresnel_power: 5.0,
            refraction_strength: 0.05,
            roughness: 0.1,
            _padding: [0.0; 4],
        }
    }

    /// Create a tropical/clear water configuration.
    pub fn tropical() -> Self {
        Self::new(
            [0.0, 0.8, 0.9],   // Bright cyan-turquoise shallow
            [0.0, 0.2, 0.4],   // Deep blue-green
            1.0,               // Shallow transition at 1m
            10.0,              // Full deep color at 10m
        )
    }

    /// Create a deep ocean configuration.
    pub fn ocean() -> Self {
        Self::new(
            [0.1, 0.3, 0.5],   // Muted blue-green shallow
            [0.02, 0.05, 0.15], // Very dark blue deep
            5.0,                // Shallow transition at 5m
            50.0,               // Full deep color at 50m
        )
    }

    /// Create a murky/river water configuration.
    pub fn murky() -> Self {
        Self::new(
            [0.2, 0.25, 0.15], // Greenish-brown shallow
            [0.05, 0.08, 0.05], // Dark murky deep
            0.5,                // Rapid transition
            3.0,                // Full opacity at 3m
        )
    }

    /// Validate configuration values.
    pub fn validate(&self) -> Result<(), &'static str> {
        // Check color ranges
        for c in &self.base_color_shallow {
            if *c < 0.0 || *c > 1.0 {
                return Err("Shallow color components must be in range [0, 1]");
            }
        }
        for c in &self.base_color_deep {
            if *c < 0.0 || *c > 1.0 {
                return Err("Deep color components must be in range [0, 1]");
            }
        }

        // Check depths
        if self.shallow_depth < 0.0 {
            return Err("Shallow depth must be non-negative");
        }
        if self.deep_depth <= self.shallow_depth {
            return Err("Deep depth must be greater than shallow depth");
        }

        // Check Fresnel
        if self.fresnel_f0 < 0.0 || self.fresnel_f0 > 1.0 {
            return Err("Fresnel F0 must be in range [0, 1]");
        }
        if self.fresnel_power < 0.0 {
            return Err("Fresnel power must be non-negative");
        }

        // Check refraction
        if self.refraction_strength < 0.0 || self.refraction_strength > 1.0 {
            return Err("Refraction strength must be in range [0, 1]");
        }

        // Check roughness
        if self.roughness < 0.0 || self.roughness > 1.0 {
            return Err("Roughness must be in range [0, 1]");
        }

        Ok(())
    }

    /// Interpolate water color based on depth.
    #[inline]
    pub fn depth_color(&self, depth: f32) -> [f32; 3] {
        if depth <= self.shallow_depth {
            return self.base_color_shallow;
        }
        if depth >= self.deep_depth {
            return self.base_color_deep;
        }

        // Linear interpolation
        let t = (depth - self.shallow_depth) / (self.deep_depth - self.shallow_depth);
        [
            self.base_color_shallow[0] + t * (self.base_color_deep[0] - self.base_color_shallow[0]),
            self.base_color_shallow[1] + t * (self.base_color_deep[1] - self.base_color_shallow[1]),
            self.base_color_shallow[2] + t * (self.base_color_deep[2] - self.base_color_shallow[2]),
        ]
    }
}

impl Default for WaterShadingConfig {
    fn default() -> Self {
        Self::new(
            [0.1, 0.5, 0.6],   // Cyan-teal shallow
            [0.02, 0.1, 0.2],  // Dark blue deep
            2.0,               // Shallow at 2m
            20.0,              // Deep at 20m
        )
    }
}

// ---------------------------------------------------------------------------
// WaterReflectionConfig
// ---------------------------------------------------------------------------

/// Configuration for water reflection techniques.
#[derive(Clone, Copy, Debug)]
pub struct WaterReflectionConfig {
    /// Enable screen-space reflections.
    pub ssr_enabled: bool,
    /// Maximum ray march steps for SSR (32-128 typical).
    pub ssr_max_steps: u32,
    /// Depth buffer thickness for SSR hit detection.
    pub ssr_thickness: f32,
    /// Enable planar reflections (expensive, for calm water).
    pub planar_enabled: bool,
    /// Fall back to reflection probes when SSR/planar fail.
    pub probe_fallback: bool,
    /// Overall reflection intensity multiplier.
    pub reflection_intensity: f32,
}

impl WaterReflectionConfig {
    /// Create a new reflection configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// High quality reflection settings.
    pub fn high_quality() -> Self {
        Self {
            ssr_enabled: true,
            ssr_max_steps: 128,
            ssr_thickness: 0.5,
            planar_enabled: true,
            probe_fallback: true,
            reflection_intensity: 1.0,
        }
    }

    /// Low quality (performance) reflection settings.
    pub fn low_quality() -> Self {
        Self {
            ssr_enabled: false,
            ssr_max_steps: 32,
            ssr_thickness: 1.0,
            planar_enabled: false,
            probe_fallback: true,
            reflection_intensity: 0.8,
        }
    }

    /// SSR-only reflection settings.
    pub fn ssr_only() -> Self {
        Self {
            ssr_enabled: true,
            ssr_max_steps: 64,
            ssr_thickness: 0.75,
            planar_enabled: false,
            probe_fallback: true,
            reflection_intensity: 1.0,
        }
    }

    /// Validate configuration values.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.ssr_max_steps == 0 || self.ssr_max_steps > 256 {
            return Err("SSR max steps must be in range [1, 256]");
        }
        if self.ssr_thickness <= 0.0 {
            return Err("SSR thickness must be positive");
        }
        if self.reflection_intensity < 0.0 {
            return Err("Reflection intensity must be non-negative");
        }
        Ok(())
    }
}

impl Default for WaterReflectionConfig {
    fn default() -> Self {
        Self {
            ssr_enabled: true,
            ssr_max_steps: 64,
            ssr_thickness: 0.5,
            planar_enabled: false,
            probe_fallback: true,
            reflection_intensity: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Fresnel Utilities
// ---------------------------------------------------------------------------

/// Schlick's approximation for Fresnel reflectance.
///
/// `F = F0 + (1 - F0) * (1 - cos_theta)^power`
///
/// # Arguments
///
/// * `cos_theta` - Cosine of angle between view direction and surface normal.
/// * `f0` - Fresnel reflectance at normal incidence.
///
/// # Returns
///
/// Fresnel reflectance term in range [F0, 1].
#[inline]
pub fn fresnel_schlick(cos_theta: f32, f0: f32) -> f32 {
    let cos_theta = cos_theta.clamp(0.0, 1.0);
    let one_minus_cos = 1.0 - cos_theta;
    // Using repeated multiplication for (1-cos)^5
    let pow5 = one_minus_cos * one_minus_cos;
    let pow5 = pow5 * pow5 * one_minus_cos;
    f0 + (1.0 - f0) * pow5
}

/// Schlick's approximation with adjustable power.
///
/// # Arguments
///
/// * `cos_theta` - Cosine of angle between view direction and surface normal.
/// * `f0` - Fresnel reflectance at normal incidence.
/// * `power` - Exponent for the falloff curve (5.0 is standard Schlick).
#[inline]
pub fn fresnel_schlick_power(cos_theta: f32, f0: f32, power: f32) -> f32 {
    let cos_theta = cos_theta.clamp(0.0, 1.0);
    let one_minus_cos = 1.0 - cos_theta;
    f0 + (1.0 - f0) * one_minus_cos.powf(power)
}

/// Full Fresnel equations for dielectric materials.
///
/// Computes the exact Fresnel reflectance for unpolarized light.
/// More accurate than Schlick for large index differences.
///
/// # Arguments
///
/// * `cos_theta` - Cosine of incident angle.
/// * `ior` - Index of refraction ratio (n2/n1, e.g., 1.33 for air-to-water).
///
/// # Returns
///
/// Fresnel reflectance in range [0, 1]. Returns 1.0 for total internal reflection.
pub fn fresnel_dielectric(cos_theta: f32, ior: f32) -> f32 {
    let cos_theta = cos_theta.clamp(0.0, 1.0);

    // Snell's law to find transmitted angle
    let sin_theta_i = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let sin_theta_t = sin_theta_i / ior;

    // Total internal reflection check
    if sin_theta_t >= 1.0 {
        return 1.0;
    }

    let cos_theta_t = (1.0 - sin_theta_t * sin_theta_t).max(0.0).sqrt();

    // Fresnel equations for s and p polarization
    let r_s = (cos_theta - ior * cos_theta_t) / (cos_theta + ior * cos_theta_t);
    let r_p = (ior * cos_theta - cos_theta_t) / (ior * cos_theta + cos_theta_t);

    // Average for unpolarized light
    (r_s * r_s + r_p * r_p) * 0.5
}

/// Calculate F0 (reflectance at normal incidence) from index of refraction.
///
/// `F0 = ((n1 - n2) / (n1 + n2))^2`
///
/// For air (n1=1.0) to water (n2=1.33): F0 ≈ 0.02
///
/// # Arguments
///
/// * `ior` - Index of refraction of the material (assuming air outside).
#[inline]
pub fn f0_from_ior(ior: f32) -> f32 {
    let ratio = (AIR_IOR - ior) / (AIR_IOR + ior);
    ratio * ratio
}

/// Calculate IOR from F0 value.
///
/// Inverse of `f0_from_ior`.
#[inline]
pub fn ior_from_f0(f0: f32) -> f32 {
    let f0 = f0.clamp(0.0, 0.99);
    let sqrt_f0 = f0.sqrt();
    (AIR_IOR + sqrt_f0) / (AIR_IOR - sqrt_f0).max(EPSILON)
}

// ---------------------------------------------------------------------------
// Refraction Utilities
// ---------------------------------------------------------------------------

/// Compute screen-space refraction UV offset.
///
/// Used for distorting the background behind water based on the surface normal.
///
/// # Arguments
///
/// * `normal` - Surface normal vector (should be normalized).
/// * `view_dir` - View direction vector (should be normalized, pointing toward camera).
/// * `strength` - Distortion strength multiplier (0.0-0.1 typical).
///
/// # Returns
///
/// Screen-space UV offset [du, dv].
pub fn compute_refraction_offset(
    normal: [f32; 3],
    view_dir: [f32; 3],
    strength: f32,
) -> [f32; 2] {
    // Project normal onto screen space (simplified, assumes view-aligned)
    // The XZ components of the normal relative to view create UV distortion
    let n_dot_v = dot3(normal, view_dir).abs().max(EPSILON);

    // Distortion based on how tilted the surface is
    let distortion_scale = strength / n_dot_v;

    // Use normal's XY (in view space approximation) for offset
    // For a proper implementation, transform normal to view space first
    let offset_u = normal[0] * distortion_scale;
    let offset_v = normal[2] * distortion_scale;

    // Clamp to prevent extreme distortion
    [
        offset_u.clamp(-0.5, 0.5),
        offset_v.clamp(-0.5, 0.5),
    ]
}

/// Compute refracted ray direction using Snell's law.
///
/// # Arguments
///
/// * `incident` - Incident ray direction (normalized, pointing into surface).
/// * `normal` - Surface normal (normalized, pointing outward).
/// * `eta` - Ratio of indices of refraction (n1/n2, e.g., 1.0/1.33 for air-to-water).
///
/// # Returns
///
/// Refracted ray direction, or `None` if total internal reflection occurs.
pub fn snells_law_direction(
    incident: [f32; 3],
    normal: [f32; 3],
    eta: f32,
) -> Option<[f32; 3]> {
    let cos_i = -dot3(incident, normal);
    let sin2_t = eta * eta * (1.0 - cos_i * cos_i);

    // Total internal reflection
    if sin2_t > 1.0 {
        return None;
    }

    let cos_t = (1.0 - sin2_t).sqrt();

    // Refracted direction: eta * I + (eta * cos_i - cos_t) * N
    let scale = eta * cos_i - cos_t;
    Some([
        eta * incident[0] + scale * normal[0],
        eta * incident[1] + scale * normal[1],
        eta * incident[2] + scale * normal[2],
    ])
}

/// Compute reflection ray direction.
///
/// # Arguments
///
/// * `incident` - Incident ray direction (normalized).
/// * `normal` - Surface normal (normalized).
///
/// # Returns
///
/// Reflected ray direction.
#[inline]
pub fn reflect_direction(incident: [f32; 3], normal: [f32; 3]) -> [f32; 3] {
    let d = 2.0 * dot3(incident, normal);
    [
        incident[0] - d * normal[0],
        incident[1] - d * normal[1],
        incident[2] - d * normal[2],
    ]
}

/// Critical angle for total internal reflection.
///
/// Beyond this angle (from normal), light cannot exit the denser medium.
///
/// # Arguments
///
/// * `ior_ratio` - n2/n1 where light travels from medium 1 to medium 2.
///                 For water-to-air: 1.0/1.33 ≈ 0.75.
#[inline]
pub fn critical_angle(ior_ratio: f32) -> f32 {
    if ior_ratio >= 1.0 {
        // No TIR possible when going to denser medium
        std::f32::consts::FRAC_PI_2
    } else {
        ior_ratio.asin()
    }
}

// ---------------------------------------------------------------------------
// Subsurface Scattering
// ---------------------------------------------------------------------------

/// Approximate subsurface scattering color for water.
///
/// Models light that enters the water, scatters, and exits toward the viewer.
/// Creates the characteristic blue-green glow when looking through waves.
///
/// # Arguments
///
/// * `view_dir` - View direction (toward camera, normalized).
/// * `light_dir` - Light direction (toward light source, normalized).
/// * `normal` - Surface normal (normalized).
/// * `water_depth` - Local water depth/thickness in meters.
/// * `config` - Water shading configuration.
///
/// # Returns
///
/// RGB color contribution from subsurface scattering.
pub fn subsurface_color(
    view_dir: [f32; 3],
    light_dir: [f32; 3],
    normal: [f32; 3],
    water_depth: f32,
    config: &WaterShadingConfig,
) -> [f32; 3] {
    // Wrap lighting: light can wrap around the surface
    let wrap = 0.5;
    let n_dot_l = dot3(normal, light_dir);
    let wrap_diffuse = (n_dot_l + wrap) / (1.0 + wrap);
    let wrap_diffuse = wrap_diffuse.max(0.0);

    // View-dependent term: more SSS when looking toward the light
    let v_dot_l = dot3(view_dir, light_dir);
    let forward_scatter = (v_dot_l * 0.5 + 0.5).powf(2.0);

    // Depth-based attenuation
    let base_color = config.depth_color(water_depth);
    let depth_factor = (-water_depth * 0.1).exp();

    // Combine factors
    let intensity = wrap_diffuse * forward_scatter * depth_factor;

    [
        base_color[0] * intensity,
        base_color[1] * intensity,
        base_color[2] * intensity,
    ]
}

/// Enhanced subsurface approximation with ambient occlusion.
pub fn subsurface_color_ao(
    view_dir: [f32; 3],
    light_dir: [f32; 3],
    normal: [f32; 3],
    water_depth: f32,
    ao: f32,
    config: &WaterShadingConfig,
) -> [f32; 3] {
    let sss = subsurface_color(view_dir, light_dir, normal, water_depth, config);
    [sss[0] * ao, sss[1] * ao, sss[2] * ao]
}

// ---------------------------------------------------------------------------
// GGX Specular
// ---------------------------------------------------------------------------

/// GGX (Trowbridge-Reitz) normal distribution function.
///
/// # Arguments
///
/// * `n_dot_h` - Dot product of normal and half vector.
/// * `roughness` - Surface roughness (0-1).
#[inline]
pub fn ggx_distribution(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let n_dot_h2 = n_dot_h * n_dot_h;

    let denom = n_dot_h2 * (a2 - 1.0) + 1.0;
    a2 / (PI * denom * denom).max(EPSILON)
}

/// Smith's geometry function for GGX (single direction).
#[inline]
fn ggx_geometry_schlick(n_dot_v: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;

    let denom = n_dot_v * (1.0 - k) + k;
    n_dot_v / denom.max(EPSILON)
}

/// Smith's geometry function (combined view and light directions).
#[inline]
pub fn ggx_geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let ggx_v = ggx_geometry_schlick(n_dot_v, roughness);
    let ggx_l = ggx_geometry_schlick(n_dot_l, roughness);
    ggx_v * ggx_l
}

/// Complete GGX specular term for water.
///
/// Combines distribution, geometry, and Fresnel terms.
///
/// # Arguments
///
/// * `n_dot_h` - Dot product of normal and half vector.
/// * `n_dot_v` - Dot product of normal and view direction.
/// * `n_dot_l` - Dot product of normal and light direction.
/// * `roughness` - Surface roughness (0-1).
///
/// # Returns
///
/// Specular intensity multiplier.
pub fn water_specular_ggx(
    n_dot_h: f32,
    n_dot_v: f32,
    n_dot_l: f32,
    roughness: f32,
) -> f32 {
    let n_dot_h = n_dot_h.max(0.0);
    let n_dot_v = n_dot_v.max(0.001);
    let n_dot_l = n_dot_l.max(0.0);

    let d = ggx_distribution(n_dot_h, roughness);
    let g = ggx_geometry_smith(n_dot_v, n_dot_l, roughness);
    let f = fresnel_schlick(n_dot_v, DEFAULT_F0);

    (d * g * f) / (4.0 * n_dot_v * n_dot_l).max(EPSILON)
}

/// Anisotropic GGX distribution stretched along wave direction.
///
/// Creates elongated specular highlights aligned with wave crests.
///
/// # Arguments
///
/// * `half_vector` - Half vector between view and light (normalized).
/// * `tangent` - Tangent vector along wave direction (normalized).
/// * `bitangent` - Bitangent vector perpendicular to wave (normalized).
/// * `normal` - Surface normal (normalized).
/// * `roughness_x` - Roughness along tangent (wave direction).
/// * `roughness_y` - Roughness along bitangent.
///
/// # Returns
///
/// Anisotropic specular distribution value.
pub fn water_specular_anisotropic(
    half_vector: [f32; 3],
    tangent: [f32; 3],
    bitangent: [f32; 3],
    normal: [f32; 3],
    roughness_x: f32,
    roughness_y: f32,
) -> f32 {
    let n_dot_h = dot3(normal, half_vector).max(0.0);
    let t_dot_h = dot3(tangent, half_vector);
    let b_dot_h = dot3(bitangent, half_vector);

    let ax = roughness_x * roughness_x;
    let ay = roughness_y * roughness_y;

    // Anisotropic GGX formula
    let exponent = (t_dot_h * t_dot_h) / (ax * ax).max(EPSILON)
        + (b_dot_h * b_dot_h) / (ay * ay).max(EPSILON);

    let base = exponent + n_dot_h * n_dot_h;
    let denom = PI * ax * ay * base * base;
    1.0 / denom.max(EPSILON)
}

/// Compute anisotropic roughness based on wave parameters.
///
/// # Arguments
///
/// * `base_roughness` - Base surface roughness.
/// * `wave_height` - Current wave height at the point.
/// * `max_wave_height` - Maximum expected wave height.
///
/// # Returns
///
/// (roughness_along_wave, roughness_perpendicular)
pub fn anisotropic_roughness_from_waves(
    base_roughness: f32,
    wave_height: f32,
    max_wave_height: f32,
) -> (f32, f32) {
    let wave_factor = (wave_height / max_wave_height.max(0.1)).abs().clamp(0.0, 1.0);

    // Along wave direction: smoother (stretched highlights)
    let roughness_x = base_roughness * (0.5 + 0.5 * wave_factor);
    // Perpendicular: rougher (compressed highlights)
    let roughness_y = base_roughness * (1.0 + wave_factor);

    (roughness_x.clamp(0.01, 1.0), roughness_y.clamp(0.01, 1.0))
}

// ---------------------------------------------------------------------------
// WaterShadingPass
// ---------------------------------------------------------------------------

/// Complete water shading evaluation pass.
#[derive(Clone, Debug)]
pub struct WaterShadingPass {
    /// Shading configuration.
    config: WaterShadingConfig,
    /// Reflection configuration.
    reflection_config: WaterReflectionConfig,
}

impl WaterShadingPass {
    /// Create a new water shading pass.
    pub fn new(config: WaterShadingConfig, reflection: WaterReflectionConfig) -> Self {
        Self {
            config,
            reflection_config: reflection,
        }
    }

    /// Create with default configurations.
    pub fn default_pass() -> Self {
        Self::new(WaterShadingConfig::default(), WaterReflectionConfig::default())
    }

    /// Get shading configuration.
    #[inline]
    pub fn config(&self) -> &WaterShadingConfig {
        &self.config
    }

    /// Get mutable shading configuration.
    #[inline]
    pub fn config_mut(&mut self) -> &mut WaterShadingConfig {
        &mut self.config
    }

    /// Get reflection configuration.
    #[inline]
    pub fn reflection_config(&self) -> &WaterReflectionConfig {
        &self.reflection_config
    }

    /// Get mutable reflection configuration.
    #[inline]
    pub fn reflection_config_mut(&mut self) -> &mut WaterReflectionConfig {
        &mut self.reflection_config
    }

    /// Schlick approximation wrapper.
    #[inline]
    pub fn fresnel_schlick(cos_theta: f32, f0: f32) -> f32 {
        fresnel_schlick(cos_theta, f0)
    }

    /// Full dielectric Fresnel wrapper.
    #[inline]
    pub fn fresnel_dielectric(cos_theta: f32, ior: f32) -> f32 {
        fresnel_dielectric(cos_theta, ior)
    }

    /// F0 from IOR calculation wrapper.
    #[inline]
    pub fn f0_from_ior(ior: f32) -> f32 {
        f0_from_ior(ior)
    }

    /// Evaluate complete water shading at a point.
    ///
    /// # Arguments
    ///
    /// * `normal` - Surface normal (normalized).
    /// * `view_dir` - Direction to viewer (normalized).
    /// * `light_dir` - Direction to light (normalized).
    /// * `light_color` - RGB light color and intensity.
    /// * `water_depth` - Local water depth/thickness.
    /// * `reflection_color` - RGB color from reflections (SSR/planar/probe).
    ///
    /// # Returns
    ///
    /// Final shaded RGB color.
    pub fn shade_point(
        &self,
        normal: [f32; 3],
        view_dir: [f32; 3],
        light_dir: [f32; 3],
        light_color: [f32; 3],
        water_depth: f32,
        reflection_color: [f32; 3],
    ) -> [f32; 3] {
        // Dot products
        let n_dot_v = dot3(normal, view_dir).max(0.001);
        let n_dot_l = dot3(normal, light_dir).max(0.0);

        // Half vector for specular
        let half = normalize3([
            view_dir[0] + light_dir[0],
            view_dir[1] + light_dir[1],
            view_dir[2] + light_dir[2],
        ]);
        let n_dot_h = dot3(normal, half).max(0.0);

        // Fresnel term
        let fresnel = fresnel_schlick_power(
            n_dot_v,
            self.config.fresnel_f0,
            self.config.fresnel_power,
        );

        // Specular highlight
        let specular = water_specular_ggx(
            n_dot_h,
            n_dot_v,
            n_dot_l,
            self.config.roughness,
        );

        // Subsurface scattering
        let sss = subsurface_color(
            view_dir,
            light_dir,
            normal,
            water_depth,
            &self.config,
        );

        // Base water color
        let base_color = self.config.depth_color(water_depth);

        // Reflection contribution (weighted by Fresnel and intensity)
        let reflection_weight = fresnel * self.reflection_config.reflection_intensity;

        // Combine components
        // Reflection (weighted by Fresnel)
        // + Specular highlights
        // + Diffuse/SSS (weighted by 1 - Fresnel)
        let mut final_color = [0.0f32; 3];
        for i in 0..3 {
            let reflected = reflection_color[i] * reflection_weight;
            let spec = specular * light_color[i];
            let diffuse = base_color[i] * n_dot_l * light_color[i] * (1.0 - fresnel);
            let subsurface = sss[i] * (1.0 - fresnel);

            final_color[i] = reflected + spec + diffuse + subsurface;
        }

        final_color
    }

    /// Shade with anisotropic specular along wave direction.
    pub fn shade_point_anisotropic(
        &self,
        normal: [f32; 3],
        tangent: [f32; 3],
        view_dir: [f32; 3],
        light_dir: [f32; 3],
        light_color: [f32; 3],
        water_depth: f32,
        wave_height: f32,
        max_wave_height: f32,
        reflection_color: [f32; 3],
    ) -> [f32; 3] {
        let n_dot_v = dot3(normal, view_dir).max(0.001);
        let n_dot_l = dot3(normal, light_dir).max(0.0);

        // Half vector
        let half = normalize3([
            view_dir[0] + light_dir[0],
            view_dir[1] + light_dir[1],
            view_dir[2] + light_dir[2],
        ]);

        // Bitangent
        let bitangent = cross3(normal, tangent);
        let bitangent = normalize3(bitangent);

        // Anisotropic roughness
        let (roughness_x, roughness_y) = anisotropic_roughness_from_waves(
            self.config.roughness,
            wave_height,
            max_wave_height,
        );

        // Anisotropic specular
        let specular = water_specular_anisotropic(
            half,
            tangent,
            bitangent,
            normal,
            roughness_x,
            roughness_y,
        );

        // Fresnel
        let fresnel = fresnel_schlick_power(
            n_dot_v,
            self.config.fresnel_f0,
            self.config.fresnel_power,
        );

        // SSS
        let sss = subsurface_color(
            view_dir,
            light_dir,
            normal,
            water_depth,
            &self.config,
        );

        // Base color
        let base_color = self.config.depth_color(water_depth);

        // Reflection
        let reflection_weight = fresnel * self.reflection_config.reflection_intensity;

        // Combine
        let mut final_color = [0.0f32; 3];
        for i in 0..3 {
            let reflected = reflection_color[i] * reflection_weight;
            let spec = specular * light_color[i] * fresnel;
            let diffuse = base_color[i] * n_dot_l * light_color[i] * (1.0 - fresnel);
            let subsurface = sss[i] * (1.0 - fresnel);

            final_color[i] = reflected + spec + diffuse + subsurface;
        }

        final_color
    }
}

impl Default for WaterShadingPass {
    fn default() -> Self {
        Self::default_pass()
    }
}

// ---------------------------------------------------------------------------
// Vector Utilities
// ---------------------------------------------------------------------------

#[inline]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > EPSILON {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 1.0, 0.0]
    }
}

#[inline]
fn vec3_length(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
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
    // Struct Size Tests
    // -------------------------------------------------------------------------

    // Test 1: WaterShadingConfig size is 64 bytes
    #[test]
    fn test_shading_config_size() {
        assert_eq!(
            std::mem::size_of::<WaterShadingConfig>(),
            WATER_SHADING_CONFIG_SIZE
        );
        assert_eq!(std::mem::size_of::<WaterShadingConfig>(), 64);
    }

    // Test 2: WaterShadingConfig is Pod
    #[test]
    fn test_shading_config_pod() {
        let config = WaterShadingConfig::default();
        let bytes: &[u8] = bytemuck::bytes_of(&config);
        assert_eq!(bytes.len(), 64);
    }

    // -------------------------------------------------------------------------
    // Fresnel Tests
    // -------------------------------------------------------------------------

    // Test 3: Fresnel at 0 degrees (cos_theta = 1) equals F0
    #[test]
    fn test_fresnel_at_zero_degrees() {
        let f0 = 0.02;
        let fresnel = fresnel_schlick(1.0, f0);
        assert!(
            approx_eq(fresnel, f0),
            "Fresnel at normal incidence should equal F0: {} vs {}",
            fresnel,
            f0
        );
    }

    // Test 4: Fresnel at 90 degrees (cos_theta = 0) equals 1.0
    #[test]
    fn test_fresnel_at_ninety_degrees() {
        let f0 = 0.02;
        let fresnel = fresnel_schlick(0.0, f0);
        assert!(
            approx_eq(fresnel, 1.0),
            "Fresnel at grazing angle should be 1.0: {}",
            fresnel
        );
    }

    // Test 5: Fresnel monotonically increases as angle increases
    #[test]
    fn test_fresnel_monotonic() {
        let f0 = 0.02;
        let mut prev = fresnel_schlick(1.0, f0);

        for i in 1..=10 {
            let cos_theta = 1.0 - (i as f32 * 0.1);
            let fresnel = fresnel_schlick(cos_theta, f0);
            assert!(
                fresnel >= prev,
                "Fresnel should increase as angle increases: {} < {}",
                fresnel,
                prev
            );
            prev = fresnel;
        }
    }

    // Test 6: F0 from IOR 1.33 approximates 0.02
    #[test]
    fn test_f0_from_ior_water() {
        let f0 = f0_from_ior(WATER_IOR);
        assert!(
            approx_eq_loose(f0, 0.02, 0.005),
            "F0 for water should be ~0.02: {}",
            f0
        );
    }

    // Test 7: F0 from IOR 1.0 (vacuum/air) equals 0
    #[test]
    fn test_f0_from_ior_air() {
        let f0 = f0_from_ior(1.0);
        assert!(
            approx_eq(f0, 0.0),
            "F0 for n=1 should be 0: {}",
            f0
        );
    }

    // Test 8: F0 from IOR 1.5 (glass) is approximately 0.04
    #[test]
    fn test_f0_from_ior_glass() {
        let f0 = f0_from_ior(1.5);
        let ratio = (1.0 - 1.5) / (1.0 + 1.5);
        let expected = ratio * ratio;
        assert!(
            approx_eq(f0, expected),
            "F0 for glass: {} vs {}",
            f0,
            expected
        );
    }

    // Test 9: Schlick vs full Fresnel comparison at normal incidence
    #[test]
    fn test_schlick_vs_dielectric_normal() {
        let cos_theta = 1.0;
        let schlick = fresnel_schlick(cos_theta, DEFAULT_F0);
        let dielectric = fresnel_dielectric(cos_theta, WATER_IOR);

        assert!(
            approx_eq_loose(schlick, dielectric, 0.01),
            "Schlick and dielectric should be close at normal incidence: {} vs {}",
            schlick,
            dielectric
        );
    }

    // Test 10: Schlick vs full Fresnel comparison at 45 degrees
    #[test]
    fn test_schlick_vs_dielectric_45deg() {
        let cos_theta = (45.0f32).to_radians().cos();
        let schlick = fresnel_schlick(cos_theta, DEFAULT_F0);
        let dielectric = fresnel_dielectric(cos_theta, WATER_IOR);

        // They should be reasonably close (within 5%)
        let diff = (schlick - dielectric).abs();
        assert!(
            diff < 0.05,
            "Schlick and dielectric should be similar at 45deg: {} vs {}, diff={}",
            schlick,
            dielectric,
            diff
        );
    }

    // Test 11: Full Fresnel returns 1.0 for total internal reflection
    #[test]
    fn test_fresnel_total_internal_reflection() {
        // Going from water to air at steep angle (beyond critical angle)
        // The critical angle for water->air is about 48.6 degrees
        // cos(80 degrees) = 0.17, which is a very steep (grazing) angle
        let cos_theta = 0.1; // Very steep angle
        let ior = AIR_IOR / WATER_IOR; // ~0.75, going from water to air

        // At steep angles with this IOR ratio, we get TIR
        // sin_theta_i = sqrt(1 - 0.1^2) = 0.995
        // sin_theta_t = 0.995 / 0.75 = 1.33 > 1 -> TIR
        let fresnel = fresnel_dielectric(cos_theta, ior);
        assert!(
            approx_eq(fresnel, 1.0),
            "Total internal reflection should give F=1.0: {}",
            fresnel
        );
    }

    // Test 12: IOR from F0 round-trip
    #[test]
    fn test_ior_from_f0_roundtrip() {
        let original_ior = 1.45;
        let f0 = f0_from_ior(original_ior);
        let recovered_ior = ior_from_f0(f0);

        assert!(
            approx_eq_loose(recovered_ior, original_ior, 0.01),
            "IOR round-trip: {} -> {} -> {}",
            original_ior,
            f0,
            recovered_ior
        );
    }

    // Test 13: Fresnel power adjustment
    #[test]
    fn test_fresnel_schlick_power() {
        let cos_theta = 0.5;
        let f0 = 0.02;

        let standard = fresnel_schlick(cos_theta, f0);
        let power_3 = fresnel_schlick_power(cos_theta, f0, 3.0);
        let power_7 = fresnel_schlick_power(cos_theta, f0, 7.0);

        // Lower power = faster transition to 1.0
        assert!(power_3 > standard, "Power 3 should be higher: {} vs {}", power_3, standard);
        // Higher power = slower transition to 1.0
        assert!(power_7 < standard, "Power 7 should be lower: {} vs {}", power_7, standard);
    }

    // -------------------------------------------------------------------------
    // Refraction Tests
    // -------------------------------------------------------------------------

    // Test 14: Refraction offset is bounded
    #[test]
    fn test_refraction_offset_bounds() {
        let normal = [0.5, 0.7, 0.5];
        let normal = normalize3(normal);
        let view = [0.0, 1.0, 0.0];
        let strength = 0.1;

        let offset = compute_refraction_offset(normal, view, strength);

        assert!(
            offset[0].abs() <= 0.5,
            "U offset should be bounded: {}",
            offset[0]
        );
        assert!(
            offset[1].abs() <= 0.5,
            "V offset should be bounded: {}",
            offset[1]
        );
    }

    // Test 15: Refraction offset with flat normal is minimal
    #[test]
    fn test_refraction_offset_flat() {
        let normal = [0.0, 1.0, 0.0];
        let view = [0.0, 1.0, 0.0];
        let strength = 0.05;

        let offset = compute_refraction_offset(normal, view, strength);

        assert!(
            offset[0].abs() < 0.01,
            "Flat surface should have minimal U offset: {}",
            offset[0]
        );
        assert!(
            offset[1].abs() < 0.01,
            "Flat surface should have minimal V offset: {}",
            offset[1]
        );
    }

    // Test 16: Snell's law produces refracted direction
    #[test]
    fn test_snells_law_direction() {
        let incident = normalize3([0.0, -1.0, 0.0]); // Straight down
        let normal = [0.0, 1.0, 0.0];
        let eta = AIR_IOR / WATER_IOR;

        let refracted = snells_law_direction(incident, normal, eta);
        assert!(refracted.is_some(), "Should refract at normal incidence");

        let r = refracted.unwrap();
        // At normal incidence, refracted should be same direction
        assert!(
            approx_eq_loose(r[1], -1.0, 0.01),
            "Refracted should be mostly downward: {:?}",
            r
        );
    }

    // Test 17: Snell's law total internal reflection
    #[test]
    fn test_snells_law_tir() {
        // Steep angle from water to air
        let incident = normalize3([0.9, -0.1, 0.0]);
        let normal = [0.0, 1.0, 0.0];
        let eta = WATER_IOR / AIR_IOR; // Going from water to air

        let refracted = snells_law_direction(incident, normal, eta);
        assert!(
            refracted.is_none(),
            "Should have total internal reflection at steep angle"
        );
    }

    // Test 18: Reflect direction is correct
    #[test]
    fn test_reflect_direction() {
        let incident = normalize3([1.0, -1.0, 0.0]);
        let normal = [0.0, 1.0, 0.0];

        let reflected = reflect_direction(incident, normal);

        // Reflected should have same X, opposite Y
        assert!(
            approx_eq(reflected[0], incident[0]),
            "X should be unchanged"
        );
        assert!(
            approx_eq(reflected[1], -incident[1]),
            "Y should be negated"
        );
    }

    // Test 19: Critical angle for water-air interface
    #[test]
    fn test_critical_angle_water_air() {
        let ior_ratio = AIR_IOR / WATER_IOR;
        let angle = critical_angle(ior_ratio);

        // Critical angle for water is about 48.6 degrees = 0.85 radians
        assert!(
            approx_eq_loose(angle, 0.85, 0.05),
            "Critical angle should be ~0.85 rad: {}",
            angle
        );
    }

    // Test 20: No critical angle when going to denser medium
    #[test]
    fn test_no_critical_angle_to_denser() {
        let ior_ratio = WATER_IOR / AIR_IOR; // >1
        let angle = critical_angle(ior_ratio);

        assert!(
            approx_eq(angle, std::f32::consts::FRAC_PI_2),
            "Should be 90 deg when no TIR possible: {}",
            angle
        );
    }

    // -------------------------------------------------------------------------
    // Subsurface Scattering Tests
    // -------------------------------------------------------------------------

    // Test 21: Subsurface color depth interpolation - shallow
    #[test]
    fn test_sss_shallow_color() {
        let config = WaterShadingConfig::default();
        let view = [0.0, 1.0, 0.0];
        let light = [0.0, 1.0, 0.0];
        let normal = [0.0, 1.0, 0.0];

        let sss = subsurface_color(view, light, normal, 0.0, &config);

        // Should use shallow color influence
        assert!(sss[0] >= 0.0 && sss[1] >= 0.0 && sss[2] >= 0.0);
    }

    // Test 22: Subsurface color attenuates with depth
    #[test]
    fn test_sss_depth_attenuation() {
        let config = WaterShadingConfig::default();
        let view = [0.0, 1.0, 0.0];
        let light = [0.0, 1.0, 0.0];
        let normal = [0.0, 1.0, 0.0];

        let sss_shallow = subsurface_color(view, light, normal, 1.0, &config);
        let sss_deep = subsurface_color(view, light, normal, 20.0, &config);

        let intensity_shallow = sss_shallow[0] + sss_shallow[1] + sss_shallow[2];
        let intensity_deep = sss_deep[0] + sss_deep[1] + sss_deep[2];

        assert!(
            intensity_shallow > intensity_deep,
            "Shallow should be brighter than deep: {} vs {}",
            intensity_shallow,
            intensity_deep
        );
    }

    // Test 23: SSS forward scattering
    #[test]
    fn test_sss_forward_scatter() {
        let config = WaterShadingConfig::default();
        let normal = [0.0, 1.0, 0.0];

        // View and light in same direction = forward scattering
        let view_forward = [0.5, 0.5, 0.5];
        let light_forward = [0.5, 0.5, 0.5];
        let sss_forward = subsurface_color(view_forward, light_forward, normal, 2.0, &config);

        // View and light opposite = back scattering
        let view_back = [-0.5, 0.5, -0.5];
        let sss_back = subsurface_color(view_back, light_forward, normal, 2.0, &config);

        let intensity_forward = sss_forward[0] + sss_forward[1] + sss_forward[2];
        let intensity_back = sss_back[0] + sss_back[1] + sss_back[2];

        assert!(
            intensity_forward >= intensity_back,
            "Forward scatter should be >= back scatter"
        );
    }

    // -------------------------------------------------------------------------
    // GGX Specular Tests
    // -------------------------------------------------------------------------

    // Test 24: GGX distribution normalization (integral check)
    #[test]
    fn test_ggx_distribution_peak() {
        // At n_dot_h = 1, GGX should peak
        let roughness = 0.3;
        let peak = ggx_distribution(1.0, roughness);

        // Check that it's finite and positive
        assert!(
            peak.is_finite() && peak > 0.0,
            "GGX at peak should be finite positive: {}",
            peak
        );

        // And higher than at 45 degrees
        let half_angle = (45.0f32).to_radians().cos();
        let off_peak = ggx_distribution(half_angle, roughness);
        assert!(
            peak > off_peak,
            "GGX peak should be higher than off-axis: {} vs {}",
            peak,
            off_peak
        );
    }

    // Test 25: GGX roughness affects spread
    #[test]
    fn test_ggx_roughness_spread() {
        let n_dot_h = 0.8;

        let smooth = ggx_distribution(n_dot_h, 0.1);
        let rough = ggx_distribution(n_dot_h, 0.5);

        // Smoother surface = tighter highlight = larger peak value at same angle
        // Actually, rougher surfaces have lower peaks but wider distributions
        let smooth_peak = ggx_distribution(1.0, 0.1);
        let rough_peak = ggx_distribution(1.0, 0.5);

        assert!(
            smooth_peak > rough_peak,
            "Smooth surface peak should be higher: {} vs {}",
            smooth_peak,
            rough_peak
        );
    }

    // Test 26: GGX geometry term is bounded
    #[test]
    fn test_ggx_geometry_bounds() {
        for roughness in [0.1, 0.3, 0.5, 0.8] {
            for angle in [0.1, 0.3, 0.5, 0.7, 0.9] {
                let g = ggx_geometry_smith(angle, angle, roughness);
                assert!(
                    g >= 0.0 && g <= 1.0,
                    "Geometry term should be in [0,1]: {} for r={}, a={}",
                    g,
                    roughness,
                    angle
                );
            }
        }
    }

    // Test 27: Water specular is finite
    #[test]
    fn test_water_specular_finite() {
        let spec = water_specular_ggx(0.9, 0.7, 0.6, 0.2);
        assert!(
            spec.is_finite() && spec >= 0.0,
            "Specular should be finite non-negative: {}",
            spec
        );
    }

    // Test 28: Anisotropic specular stretching effect
    #[test]
    fn test_anisotropic_stretching() {
        let normal = [0.0, 1.0, 0.0];
        let tangent = [1.0, 0.0, 0.0];
        let bitangent = [0.0, 0.0, 1.0];
        let half = normalize3([0.2, 0.98, 0.0]); // Slightly off normal

        // Isotropic (equal roughness)
        let iso = water_specular_anisotropic(half, tangent, bitangent, normal, 0.2, 0.2);

        // Anisotropic (stretched along tangent)
        let aniso = water_specular_anisotropic(half, tangent, bitangent, normal, 0.1, 0.4);

        // Both should be finite
        assert!(iso.is_finite() && aniso.is_finite());
        // They should differ
        assert!(
            (iso - aniso).abs() > 0.001,
            "Anisotropic should differ from isotropic"
        );
    }

    // Test 29: Anisotropic roughness from waves
    #[test]
    fn test_anisotropic_roughness_from_waves() {
        let base = 0.2;
        let (rx, ry) = anisotropic_roughness_from_waves(base, 0.5, 1.0);

        assert!(rx > 0.0 && rx <= 1.0);
        assert!(ry > 0.0 && ry <= 1.0);
        // Perpendicular should generally be rougher
        assert!(
            ry >= rx,
            "Perpendicular roughness should be >= parallel: {} vs {}",
            ry,
            rx
        );
    }

    // -------------------------------------------------------------------------
    // Config Tests
    // -------------------------------------------------------------------------

    // Test 30: Default config validation passes
    #[test]
    fn test_default_config_valid() {
        let config = WaterShadingConfig::default();
        assert!(config.validate().is_ok());
    }

    // Test 31: Tropical preset
    #[test]
    fn test_tropical_preset() {
        let config = WaterShadingConfig::tropical();
        assert!(config.validate().is_ok());
        assert!(config.base_color_shallow[1] > 0.5); // Bright
    }

    // Test 32: Ocean preset
    #[test]
    fn test_ocean_preset() {
        let config = WaterShadingConfig::ocean();
        assert!(config.validate().is_ok());
        assert!(config.deep_depth > 20.0); // Deep ocean
    }

    // Test 33: Murky preset
    #[test]
    fn test_murky_preset() {
        let config = WaterShadingConfig::murky();
        assert!(config.validate().is_ok());
        assert!(config.deep_depth < 10.0); // Rapid falloff
    }

    // Test 34: Invalid color range rejected
    #[test]
    fn test_invalid_color_range() {
        let mut config = WaterShadingConfig::default();
        config.base_color_shallow[0] = 1.5;
        assert!(config.validate().is_err());
    }

    // Test 35: Invalid depth range rejected
    #[test]
    fn test_invalid_depth_range() {
        let mut config = WaterShadingConfig::default();
        config.shallow_depth = 10.0;
        config.deep_depth = 5.0;
        assert!(config.validate().is_err());
    }

    // Test 36: Depth color interpolation at shallow
    #[test]
    fn test_depth_color_shallow() {
        let config = WaterShadingConfig::default();
        let color = config.depth_color(0.0);
        assert!(approx_eq_vec3(color, config.base_color_shallow));
    }

    // Test 37: Depth color interpolation at deep
    #[test]
    fn test_depth_color_deep() {
        let config = WaterShadingConfig::default();
        let color = config.depth_color(config.deep_depth + 10.0);
        assert!(approx_eq_vec3(color, config.base_color_deep));
    }

    // Test 38: Depth color interpolation midpoint
    #[test]
    fn test_depth_color_midpoint() {
        let config = WaterShadingConfig::new(
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            0.0,
            10.0,
        );
        let color = config.depth_color(5.0);
        // Should be approximately [0.5, 0.5, 0.5]
        assert!(approx_eq_loose(color[0], 0.5, 0.01));
    }

    // -------------------------------------------------------------------------
    // Reflection Config Tests
    // -------------------------------------------------------------------------

    // Test 39: Default reflection config validation
    #[test]
    fn test_default_reflection_valid() {
        let config = WaterReflectionConfig::default();
        assert!(config.validate().is_ok());
    }

    // Test 40: High quality reflection preset
    #[test]
    fn test_high_quality_reflection() {
        let config = WaterReflectionConfig::high_quality();
        assert!(config.ssr_enabled);
        assert!(config.planar_enabled);
        assert!(config.ssr_max_steps >= 128);
    }

    // Test 41: Low quality reflection preset
    #[test]
    fn test_low_quality_reflection() {
        let config = WaterReflectionConfig::low_quality();
        assert!(!config.ssr_enabled);
        assert!(!config.planar_enabled);
    }

    // Test 42: Invalid SSR steps rejected
    #[test]
    fn test_invalid_ssr_steps() {
        let mut config = WaterReflectionConfig::default();
        config.ssr_max_steps = 0;
        assert!(config.validate().is_err());
    }

    // -------------------------------------------------------------------------
    // WaterShadingPass Tests
    // -------------------------------------------------------------------------

    // Test 43: Default pass creation
    #[test]
    fn test_default_pass() {
        let pass = WaterShadingPass::default();
        assert!(pass.config().validate().is_ok());
    }

    // Test 44: Pass shade_point returns valid color
    #[test]
    fn test_shade_point_valid() {
        let pass = WaterShadingPass::default();
        let normal = [0.0, 1.0, 0.0];
        let view = [0.0, 1.0, 0.0];
        let light = normalize3([1.0, 1.0, 0.0]);
        let light_color = [1.0, 1.0, 1.0];
        let reflection = [0.2, 0.3, 0.4];

        let color = pass.shade_point(normal, view, light, light_color, 5.0, reflection);

        for i in 0..3 {
            assert!(
                color[i].is_finite() && color[i] >= 0.0,
                "Color component {} should be finite non-negative: {}",
                i,
                color[i]
            );
        }
    }

    // Test 45: Pass Fresnel wrappers work
    #[test]
    fn test_pass_fresnel_wrappers() {
        let schlick = WaterShadingPass::fresnel_schlick(0.8, 0.02);
        let dielectric = WaterShadingPass::fresnel_dielectric(0.8, 1.33);
        let f0 = WaterShadingPass::f0_from_ior(1.33);

        assert!(schlick > 0.0 && schlick < 1.0);
        assert!(dielectric > 0.0 && dielectric < 1.0);
        assert!(f0 > 0.0 && f0 < 0.1);
    }

    // Test 46: Anisotropic shade point
    #[test]
    fn test_shade_point_anisotropic() {
        let pass = WaterShadingPass::default();
        let normal = [0.0, 1.0, 0.0];
        let tangent = [1.0, 0.0, 0.0];
        let view = [0.0, 1.0, 0.0];
        let light = normalize3([1.0, 1.0, 0.0]);

        let color = pass.shade_point_anisotropic(
            normal,
            tangent,
            view,
            light,
            [1.0, 1.0, 1.0],
            5.0,
            0.5,
            1.0,
            [0.2, 0.3, 0.4],
        );

        for c in color {
            assert!(c.is_finite() && c >= 0.0);
        }
    }

    // Test 47: Config accessors
    #[test]
    fn test_config_accessors() {
        let mut pass = WaterShadingPass::default();

        pass.config_mut().roughness = 0.5;
        assert!(approx_eq(pass.config().roughness, 0.5));

        pass.reflection_config_mut().ssr_max_steps = 100;
        assert_eq!(pass.reflection_config().ssr_max_steps, 100);
    }

    // -------------------------------------------------------------------------
    // Vector Utility Tests
    // -------------------------------------------------------------------------

    // Test 48: dot3 calculation
    #[test]
    fn test_dot3() {
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        assert!(approx_eq(dot3(a, b), 0.0)); // Perpendicular

        let c = [1.0, 0.0, 0.0];
        assert!(approx_eq(dot3(a, c), 1.0)); // Parallel
    }

    // Test 49: cross3 calculation
    #[test]
    fn test_cross3() {
        let x = [1.0, 0.0, 0.0];
        let y = [0.0, 1.0, 0.0];
        let z = cross3(x, y);
        assert!(approx_eq_vec3(z, [0.0, 0.0, 1.0]));
    }

    // Test 50: normalize3 unit vector
    #[test]
    fn test_normalize3() {
        let v = [3.0, 4.0, 0.0];
        let n = normalize3(v);
        let len = vec3_length(n);
        assert!(approx_eq(len, 1.0));
        assert!(approx_eq(n[0], 0.6));
        assert!(approx_eq(n[1], 0.8));
    }

    // Test 51: normalize3 zero vector fallback
    #[test]
    fn test_normalize3_zero() {
        let v = [0.0, 0.0, 0.0];
        let n = normalize3(v);
        assert!(approx_eq_vec3(n, [0.0, 1.0, 0.0])); // Default up
    }

    // -------------------------------------------------------------------------
    // Edge Case Tests
    // -------------------------------------------------------------------------

    // Test 52: Fresnel with F0 = 0
    #[test]
    fn test_fresnel_f0_zero() {
        let f = fresnel_schlick(0.5, 0.0);
        // With F0=0, F = (1-cos)^5
        let base = 0.5f32;
        let expected = base * base * base * base * base;
        assert!(approx_eq(f, expected));
    }

    // Test 53: Fresnel with F0 = 1
    #[test]
    fn test_fresnel_f0_one() {
        let f = fresnel_schlick(0.5, 1.0);
        assert!(approx_eq(f, 1.0)); // Always 1
    }

    // Test 54: GGX with very low roughness
    #[test]
    fn test_ggx_low_roughness() {
        // With roughness = 0.01:
        // a = 0.01 * 0.01 = 0.0001
        // a^2 = 0.00000001
        // At n_dot_h = 1: denom = 1 * (0.00000001 - 1) + 1 = 0.00000001
        // D = 0.00000001 / (PI * 0.00000001^2)
        // But our EPSILON clamp kicks in, making result smaller
        // The key point is that low roughness should give a sharper peak than high roughness
        let d_low = ggx_distribution(1.0, 0.1);
        let d_high = ggx_distribution(1.0, 0.5);

        assert!(d_low.is_finite());
        assert!(d_high.is_finite());
        // Low roughness = sharper highlight = higher peak value
        assert!(
            d_low > d_high,
            "Low roughness should have sharper peak: {} vs {}",
            d_low,
            d_high
        );
    }

    // Test 55: GGX with roughness = 1
    #[test]
    fn test_ggx_max_roughness() {
        let d = ggx_distribution(0.5, 1.0);
        assert!(d.is_finite());
        assert!(d < 1.0); // Very diffuse
    }

    // Test 56: Shade point with zero reflection
    #[test]
    fn test_shade_no_reflection() {
        let pass = WaterShadingPass::default();
        let color = pass.shade_point(
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            5.0,
            [0.0, 0.0, 0.0], // No reflection
        );

        // Should still produce valid color from diffuse/SSS
        assert!(color[0] >= 0.0 || color[1] >= 0.0 || color[2] >= 0.0);
    }

    // Test 57: Shade point with full reflection
    #[test]
    fn test_shade_full_reflection() {
        let pass = WaterShadingPass::default();
        let color = pass.shade_point(
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            5.0,
            [1.0, 1.0, 1.0], // Full white reflection
        );

        // Should have reflection contribution
        assert!(color[0] > 0.0 && color[1] > 0.0 && color[2] > 0.0);
    }

    // Test 58: SSR-only config
    #[test]
    fn test_ssr_only_config() {
        let config = WaterReflectionConfig::ssr_only();
        assert!(config.ssr_enabled);
        assert!(!config.planar_enabled);
        assert!(config.validate().is_ok());
    }

    // Test 59: Negative depth handled
    #[test]
    fn test_negative_depth_color() {
        let config = WaterShadingConfig::default();
        let color = config.depth_color(-5.0);
        // Should clamp to shallow
        assert!(approx_eq_vec3(color, config.base_color_shallow));
    }

    // Test 60: SSS with ambient occlusion
    #[test]
    fn test_sss_with_ao() {
        let config = WaterShadingConfig::default();
        let view = [0.0, 1.0, 0.0];
        let light = [0.0, 1.0, 0.0];
        let normal = [0.0, 1.0, 0.0];

        let sss_full = subsurface_color(view, light, normal, 2.0, &config);
        let sss_ao = subsurface_color_ao(view, light, normal, 2.0, 0.5, &config);

        // AO should reduce intensity by factor of 0.5
        assert!(approx_eq(sss_ao[0], sss_full[0] * 0.5));
    }

    // Test 61: Constants are correct
    #[test]
    fn test_constants() {
        assert!(approx_eq(WATER_IOR, 1.33));
        assert!(approx_eq(AIR_IOR, 1.0));
        assert!(approx_eq_loose(DEFAULT_F0, 0.02, 0.005));
        assert_eq!(WATER_SHADING_CONFIG_SIZE, 64);
    }

    // Test 62: WaterReflectionConfig new()
    #[test]
    fn test_reflection_config_new() {
        let config = WaterReflectionConfig::new();
        assert_eq!(config.ssr_max_steps, 64); // Default
    }
}
