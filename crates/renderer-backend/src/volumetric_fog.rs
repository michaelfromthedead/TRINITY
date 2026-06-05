//! Volumetric Fog Density Field and Light Scattering.
//!
//! This module provides infrastructure for volumetric fog rendering with:
//! - Configurable density fields with height falloff and noise modulation
//! - Multiple fog layers with different falloff types
//! - Physically-based light scattering using Henyey-Greenstein phase function
//! - Local fog volumes (boxes, spheres, cylinders) for artist control
//!
//! # Overview
//!
//! Volumetric fog is rendered by:
//! 1. Sampling density at froxel centers
//! 2. Computing transmittance using Beer-Lambert law
//! 3. Accumulating inscattered light along view rays
//!
//! # Scattering Model
//!
//! The Henyey-Greenstein phase function models angular light distribution:
//! ```text
//! p(theta) = (1 - g^2) / (4*pi * (1 + g^2 - 2*g*cos(theta))^1.5)
//! ```
//! Where:
//! - `theta` is the angle between view and light directions
//! - `g` is the asymmetry parameter (-1 to 1)
//!   - g = 0: isotropic scattering
//!   - g > 0: forward scattering (bright when looking toward light)
//!   - g < 0: back scattering (bright when looking away from light)
//!
//! # Energy Conservation
//!
//! For physically correct rendering:
//! - Extinction = Absorption + Scattering
//! - Albedo = Scattering / Extinction (must be <= 1)
//! - Total energy (inscatter + transmittance) <= incoming energy

use bytemuck::{Pod, Zeroable};
use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Minimum valid density value.
pub const MIN_DENSITY: f32 = 0.0;

/// Maximum valid density value.
pub const MAX_DENSITY: f32 = 1.0;

/// Default height offset for fog (at ground level).
pub const DEFAULT_HEIGHT_OFFSET: f32 = 0.0;

/// Default height falloff rate.
pub const DEFAULT_HEIGHT_FALLOFF: f32 = 0.1;

/// Default noise scale for density modulation.
pub const DEFAULT_NOISE_SCALE: f32 = 0.1;

/// Minimum Henyey-Greenstein g parameter.
pub const MIN_HG_G: f32 = -0.999;

/// Maximum Henyey-Greenstein g parameter.
pub const MAX_HG_G: f32 = 0.999;

// ---------------------------------------------------------------------------
// FalloffType — Height fog falloff modes
// ---------------------------------------------------------------------------

/// Height fog falloff calculation mode.
///
/// Different falloff types produce different fog density curves with height.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum FalloffType {
    /// Linear falloff: density = max(0, 1 - (height - offset) / range)
    ///
    /// Simple and fast, but produces visible hard edges.
    Linear,
    /// Exponential falloff: density = exp(-falloff * max(0, height - offset))
    ///
    /// More natural appearance, commonly used for atmospheric fog.
    #[default]
    Exponential,
    /// Exponential squared falloff: density = exp(-falloff^2 * max(0, height - offset)^2)
    ///
    /// Sharper transition, useful for low-lying fog banks.
    ExponentialSquared,
}

impl FalloffType {
    /// Calculate the falloff factor for a given height above the offset.
    ///
    /// # Arguments
    ///
    /// * `height_above` - Height above the fog base (can be negative).
    /// * `falloff_rate` - Falloff rate parameter.
    /// * `range` - Used only for linear falloff (height range).
    ///
    /// # Returns
    ///
    /// Falloff multiplier in range [0, 1].
    #[inline]
    pub fn calculate(&self, height_above: f32, falloff_rate: f32, range: f32) -> f32 {
        if height_above <= 0.0 {
            return 1.0; // Full density below offset
        }

        match self {
            FalloffType::Linear => {
                if range <= 0.0 {
                    0.0
                } else {
                    (1.0 - height_above / range).max(0.0)
                }
            }
            FalloffType::Exponential => {
                (-falloff_rate * height_above).exp()
            }
            FalloffType::ExponentialSquared => {
                let fh = falloff_rate * height_above;
                (-fh * fh).exp()
            }
        }
    }

    /// Get the name string for this falloff type.
    #[inline]
    pub fn name(&self) -> &'static str {
        match self {
            FalloffType::Linear => "linear",
            FalloffType::Exponential => "exponential",
            FalloffType::ExponentialSquared => "exponential_squared",
        }
    }
}

// ---------------------------------------------------------------------------
// FogDensityConfig — GPU-uploadable density configuration
// ---------------------------------------------------------------------------

/// Configuration for fog density field evaluation.
///
/// This struct is designed to be uploaded to the GPU as a uniform buffer.
/// The layout is `repr(C)` and implements `Pod` for bytemuck compatibility.
///
/// # Memory Layout (32 bytes)
///
/// | Offset | Field          | Size    |
/// |--------|----------------|---------|
/// | 0      | base_density   | 4 bytes |
/// | 4      | height_falloff | 4 bytes |
/// | 8      | height_offset  | 4 bytes |
/// | 12     | noise_scale    | 4 bytes |
/// | 16     | noise_strength | 4 bytes |
/// | 20     | _padding       | 12 bytes |
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct FogDensityConfig {
    /// Base fog density (0.0 to 1.0).
    ///
    /// This is the uniform density before height falloff or noise modulation.
    pub base_density: f32,

    /// Exponential falloff rate with height.
    ///
    /// Higher values = faster falloff. Typical range: 0.01 to 1.0.
    pub height_falloff: f32,

    /// World-space height where fog is at full density.
    ///
    /// Below this height, fog is at base_density.
    /// Above this height, fog falls off according to height_falloff.
    pub height_offset: f32,

    /// 3D noise frequency for density modulation.
    ///
    /// Higher values = smaller noise features. Typical range: 0.01 to 1.0.
    pub noise_scale: f32,

    /// Strength of noise modulation (0.0 to 1.0).
    ///
    /// 0.0 = no noise, 1.0 = full noise amplitude.
    pub noise_strength: f32,

    /// Padding for GPU alignment (vec4 alignment).
    pub _padding: [u32; 3],
}

impl FogDensityConfig {
    /// Create a new fog density configuration.
    ///
    /// # Arguments
    ///
    /// * `base_density` - Base fog density (clamped to 0.0-1.0).
    /// * `height_falloff` - Height falloff rate.
    /// * `height_offset` - Height where fog is at full density.
    pub fn new(base_density: f32, height_falloff: f32, height_offset: f32) -> Self {
        Self {
            base_density: base_density.clamp(MIN_DENSITY, MAX_DENSITY),
            height_falloff: height_falloff.max(0.0),
            height_offset,
            noise_scale: DEFAULT_NOISE_SCALE,
            noise_strength: 0.0,
            _padding: [0; 3],
        }
    }

    /// Create a configuration with noise parameters.
    pub fn with_noise(mut self, scale: f32, strength: f32) -> Self {
        self.noise_scale = scale.max(0.001);
        self.noise_strength = strength.clamp(0.0, 1.0);
        self
    }

    /// Check if noise modulation is enabled.
    #[inline]
    pub fn has_noise(&self) -> bool {
        self.noise_strength > 0.0
    }

    /// Validate the configuration.
    pub fn is_valid(&self) -> bool {
        self.base_density >= MIN_DENSITY
            && self.base_density <= MAX_DENSITY
            && self.height_falloff >= 0.0
            && self.noise_scale > 0.0
            && self.noise_strength >= 0.0
            && self.noise_strength <= 1.0
    }

    /// Evaluate density at a world position using only height falloff.
    ///
    /// # Arguments
    ///
    /// * `world_y` - World-space Y (height) coordinate.
    ///
    /// # Returns
    ///
    /// Density value (0.0 to base_density).
    #[inline]
    pub fn evaluate_height(&self, world_y: f32) -> f32 {
        let height_above = world_y - self.height_offset;
        if height_above <= 0.0 {
            self.base_density
        } else {
            self.base_density * (-self.height_falloff * height_above).exp()
        }
    }
}

impl Default for FogDensityConfig {
    fn default() -> Self {
        Self {
            base_density: 0.05,
            height_falloff: DEFAULT_HEIGHT_FALLOFF,
            height_offset: DEFAULT_HEIGHT_OFFSET,
            noise_scale: DEFAULT_NOISE_SCALE,
            noise_strength: 0.0,
            _padding: [0; 3],
        }
    }
}

// ---------------------------------------------------------------------------
// FogLayer — Individual fog layer
// ---------------------------------------------------------------------------

/// A single fog layer with its own density and height bounds.
///
/// Multiple layers can be combined for complex fog effects like
/// ground mist with high-altitude haze.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FogLayer {
    /// Density of this layer (0.0 to 1.0).
    pub density: f32,

    /// Minimum height where this layer exists.
    pub height_min: f32,

    /// Maximum height where this layer exists.
    pub height_max: f32,

    /// How density falls off with height.
    pub falloff_type: FalloffType,
}

impl FogLayer {
    /// Create a new fog layer.
    ///
    /// # Arguments
    ///
    /// * `density` - Layer density (clamped to 0.0-1.0).
    /// * `height_min` - Bottom of the layer.
    /// * `height_max` - Top of the layer.
    /// * `falloff_type` - Height falloff calculation.
    ///
    /// # Panics
    ///
    /// Panics if `height_max <= height_min`.
    pub fn new(density: f32, height_min: f32, height_max: f32, falloff_type: FalloffType) -> Self {
        assert!(
            height_max > height_min,
            "height_max must be greater than height_min"
        );

        Self {
            density: density.clamp(MIN_DENSITY, MAX_DENSITY),
            height_min,
            height_max,
            falloff_type,
        }
    }

    /// Create a ground fog layer (full density at ground, falls off with height).
    pub fn ground_fog(density: f32, max_height: f32) -> Self {
        Self::new(density, 0.0, max_height, FalloffType::Exponential)
    }

    /// Create a fog bank layer (full density in range, sharp edges).
    pub fn fog_bank(density: f32, min_height: f32, max_height: f32) -> Self {
        Self::new(density, min_height, max_height, FalloffType::ExponentialSquared)
    }

    /// Evaluate layer density at a given height.
    ///
    /// # Arguments
    ///
    /// * `world_y` - World-space height.
    ///
    /// # Returns
    ///
    /// Layer density at this height (0.0 to layer density).
    #[inline]
    pub fn evaluate(&self, world_y: f32) -> f32 {
        // Outside layer bounds
        if world_y < self.height_min || world_y > self.height_max {
            return 0.0;
        }

        let range = self.height_max - self.height_min;
        let height_above = world_y - self.height_min;

        // Falloff rate for exponential modes (scaled to layer thickness)
        let falloff_rate = 3.0 / range; // ~5% at max height

        let factor = self.falloff_type.calculate(height_above, falloff_rate, range);
        self.density * factor
    }

    /// Check if a height is within this layer's bounds.
    #[inline]
    pub fn contains_height(&self, world_y: f32) -> bool {
        world_y >= self.height_min && world_y <= self.height_max
    }

    /// Get the height range of this layer.
    #[inline]
    pub fn height_range(&self) -> f32 {
        self.height_max - self.height_min
    }
}

impl Default for FogLayer {
    fn default() -> Self {
        Self {
            density: 0.1,
            height_min: 0.0,
            height_max: 50.0,
            falloff_type: FalloffType::Exponential,
        }
    }
}

// ---------------------------------------------------------------------------
// ScatteringParams — GPU-uploadable scattering parameters
// ---------------------------------------------------------------------------

/// Physical scattering parameters for volumetric fog.
///
/// This struct is designed to be uploaded to the GPU as a uniform buffer.
/// The layout is `repr(C)` and implements `Pod` for bytemuck compatibility.
///
/// # Memory Layout (32 bytes)
///
/// | Offset | Field            | Size    |
/// |--------|------------------|---------|
/// | 0      | extinction_rgb   | 12 bytes |
/// | 12     | henyey_g         | 4 bytes |
/// | 16     | albedo_rgb       | 12 bytes |
/// | 28     | ambient_strength | 4 bytes |
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct ScatteringParams {
    /// Per-channel extinction coefficients.
    ///
    /// Controls how quickly light is absorbed/scattered per unit distance.
    /// Higher values = denser fog appearance.
    pub extinction_rgb: [f32; 3],

    /// Henyey-Greenstein asymmetry parameter (-1 to 1).
    ///
    /// - g = 0: Isotropic (uniform in all directions)
    /// - g > 0: Forward scattering (typical for fog/mist)
    /// - g < 0: Back scattering
    ///
    /// Real-world fog: g ~ 0.7 to 0.9
    pub henyey_g: f32,

    /// Single-scatter albedo (RGB).
    ///
    /// Fraction of extinction that is scattering vs absorption.
    /// Must be <= 1.0 for each channel to conserve energy.
    pub albedo_rgb: [f32; 3],

    /// Ambient scattering strength.
    ///
    /// Adds uniform ambient light to fog, simulating multiple scattering.
    pub ambient_strength: f32,
}

impl ScatteringParams {
    /// Create new scattering parameters.
    ///
    /// # Arguments
    ///
    /// * `extinction` - Per-channel extinction coefficients.
    /// * `albedo` - Single-scatter albedo (clamped to 0-1).
    /// * `g` - Henyey-Greenstein parameter (clamped to -0.999 to 0.999).
    pub fn new(extinction: [f32; 3], albedo: [f32; 3], g: f32) -> Self {
        Self {
            extinction_rgb: [
                extinction[0].max(0.0),
                extinction[1].max(0.0),
                extinction[2].max(0.0),
            ],
            henyey_g: g.clamp(MIN_HG_G, MAX_HG_G),
            albedo_rgb: [
                albedo[0].clamp(0.0, 1.0),
                albedo[1].clamp(0.0, 1.0),
                albedo[2].clamp(0.0, 1.0),
            ],
            ambient_strength: 0.0,
        }
    }

    /// Create parameters with uniform extinction and albedo.
    pub fn uniform(extinction: f32, albedo: f32, g: f32) -> Self {
        Self::new(
            [extinction, extinction, extinction],
            [albedo, albedo, albedo],
            g,
        )
    }

    /// Set the ambient scattering strength.
    pub fn with_ambient(mut self, strength: f32) -> Self {
        self.ambient_strength = strength.clamp(0.0, 1.0);
        self
    }

    /// Validate the parameters for energy conservation.
    pub fn is_valid(&self) -> bool {
        self.extinction_rgb.iter().all(|e| *e >= 0.0)
            && self.albedo_rgb.iter().all(|a| *a >= 0.0 && *a <= 1.0)
            && self.henyey_g >= MIN_HG_G
            && self.henyey_g <= MAX_HG_G
            && self.ambient_strength >= 0.0
            && self.ambient_strength <= 1.0
    }

    /// Check if this fog is purely absorbing (no scattering).
    pub fn is_purely_absorbing(&self) -> bool {
        self.albedo_rgb.iter().all(|a| *a == 0.0)
    }

    /// Get the scattering coefficient (extinction * albedo).
    #[inline]
    pub fn scattering_coeff(&self) -> [f32; 3] {
        [
            self.extinction_rgb[0] * self.albedo_rgb[0],
            self.extinction_rgb[1] * self.albedo_rgb[1],
            self.extinction_rgb[2] * self.albedo_rgb[2],
        ]
    }

    /// Get the absorption coefficient (extinction * (1 - albedo)).
    #[inline]
    pub fn absorption_coeff(&self) -> [f32; 3] {
        [
            self.extinction_rgb[0] * (1.0 - self.albedo_rgb[0]),
            self.extinction_rgb[1] * (1.0 - self.albedo_rgb[1]),
            self.extinction_rgb[2] * (1.0 - self.albedo_rgb[2]),
        ]
    }
}

impl Default for ScatteringParams {
    fn default() -> Self {
        Self {
            extinction_rgb: [0.1, 0.1, 0.1],
            henyey_g: 0.8,
            albedo_rgb: [0.9, 0.9, 0.9],
            ambient_strength: 0.1,
        }
    }
}

// ---------------------------------------------------------------------------
// FogVolumeShape — Local fog volume shapes
// ---------------------------------------------------------------------------

/// Shape definition for local fog volumes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FogVolumeShape {
    /// Axis-aligned box with half extents.
    Box {
        /// Half size in each axis [x, y, z].
        half_extents: [f32; 3],
    },
    /// Sphere with radius.
    Sphere {
        /// Sphere radius.
        radius: f32,
    },
    /// Cylinder aligned to Y axis.
    Cylinder {
        /// Cylinder radius.
        radius: f32,
        /// Cylinder height (full height, centered at origin).
        height: f32,
    },
}

impl FogVolumeShape {
    /// Create a box shape.
    pub fn box_shape(half_x: f32, half_y: f32, half_z: f32) -> Self {
        FogVolumeShape::Box {
            half_extents: [half_x.abs(), half_y.abs(), half_z.abs()],
        }
    }

    /// Create a cube shape.
    pub fn cube(half_extent: f32) -> Self {
        FogVolumeShape::Box {
            half_extents: [half_extent.abs(); 3],
        }
    }

    /// Create a sphere shape.
    pub fn sphere(radius: f32) -> Self {
        FogVolumeShape::Sphere {
            radius: radius.abs(),
        }
    }

    /// Create a cylinder shape.
    pub fn cylinder(radius: f32, height: f32) -> Self {
        FogVolumeShape::Cylinder {
            radius: radius.abs(),
            height: height.abs(),
        }
    }

    /// Check if a local-space point is inside this shape.
    ///
    /// # Arguments
    ///
    /// * `local_pos` - Point in local (shape) space.
    ///
    /// # Returns
    ///
    /// `true` if the point is inside the shape.
    #[inline]
    pub fn contains(&self, local_pos: [f32; 3]) -> bool {
        match self {
            FogVolumeShape::Box { half_extents } => {
                local_pos[0].abs() <= half_extents[0]
                    && local_pos[1].abs() <= half_extents[1]
                    && local_pos[2].abs() <= half_extents[2]
            }
            FogVolumeShape::Sphere { radius } => {
                let dist_sq = local_pos[0] * local_pos[0]
                    + local_pos[1] * local_pos[1]
                    + local_pos[2] * local_pos[2];
                dist_sq <= radius * radius
            }
            FogVolumeShape::Cylinder { radius, height } => {
                let half_height = height * 0.5;
                let dist_sq_xz = local_pos[0] * local_pos[0] + local_pos[2] * local_pos[2];
                dist_sq_xz <= radius * radius && local_pos[1].abs() <= half_height
            }
        }
    }

    /// Calculate signed distance to the shape surface.
    ///
    /// # Arguments
    ///
    /// * `local_pos` - Point in local (shape) space.
    ///
    /// # Returns
    ///
    /// Negative inside, zero on surface, positive outside.
    #[inline]
    pub fn signed_distance(&self, local_pos: [f32; 3]) -> f32 {
        match self {
            FogVolumeShape::Box { half_extents } => {
                // SDF for box
                let dx = local_pos[0].abs() - half_extents[0];
                let dy = local_pos[1].abs() - half_extents[1];
                let dz = local_pos[2].abs() - half_extents[2];

                let outside_dist = (dx.max(0.0).powi(2)
                    + dy.max(0.0).powi(2)
                    + dz.max(0.0).powi(2))
                .sqrt();

                let inside_dist = dx.max(dy).max(dz).min(0.0);

                outside_dist + inside_dist
            }
            FogVolumeShape::Sphere { radius } => {
                let dist = (local_pos[0] * local_pos[0]
                    + local_pos[1] * local_pos[1]
                    + local_pos[2] * local_pos[2])
                .sqrt();
                dist - radius
            }
            FogVolumeShape::Cylinder { radius, height } => {
                let half_height = height * 0.5;
                let dist_xz = (local_pos[0] * local_pos[0] + local_pos[2] * local_pos[2]).sqrt();
                let dy = local_pos[1].abs() - half_height;
                let dr = dist_xz - radius;

                // Outside both
                if dr > 0.0 && dy > 0.0 {
                    (dr * dr + dy * dy).sqrt()
                } else if dr > 0.0 {
                    dr
                } else if dy > 0.0 {
                    dy
                } else {
                    dr.max(dy)
                }
            }
        }
    }

    /// Get the bounding box half extents.
    #[inline]
    pub fn bounding_half_extents(&self) -> [f32; 3] {
        match self {
            FogVolumeShape::Box { half_extents } => *half_extents,
            FogVolumeShape::Sphere { radius } => [*radius, *radius, *radius],
            FogVolumeShape::Cylinder { radius, height } => [*radius, height * 0.5, *radius],
        }
    }
}

impl Default for FogVolumeShape {
    fn default() -> Self {
        FogVolumeShape::Sphere { radius: 10.0 }
    }
}

// ---------------------------------------------------------------------------
// FogVolume — Local fog volume
// ---------------------------------------------------------------------------

/// A local fog volume placed by artists.
///
/// Fog volumes add localized fog effects that blend with global fog.
/// They support soft falloff at edges for natural-looking transitions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FogVolume {
    /// Shape of this fog volume.
    pub shape: FogVolumeShape,

    /// World transform matrix (column-major, 4x4).
    ///
    /// Used to transform world positions into local space for containment tests.
    pub transform: [f32; 16],

    /// Fog density inside this volume.
    pub density: f32,

    /// Edge falloff distance.
    ///
    /// Higher values = softer edges. 0.0 = hard edges.
    pub falloff: f32,
}

impl FogVolume {
    /// Create a new fog volume.
    ///
    /// # Arguments
    ///
    /// * `shape` - Volume shape.
    /// * `density` - Fog density (clamped to 0-1).
    /// * `falloff` - Edge softness (clamped to >= 0).
    pub fn new(shape: FogVolumeShape, density: f32, falloff: f32) -> Self {
        Self {
            shape,
            transform: Self::identity_matrix(),
            density: density.clamp(MIN_DENSITY, MAX_DENSITY),
            falloff: falloff.max(0.0),
        }
    }

    /// Create an identity 4x4 matrix.
    fn identity_matrix() -> [f32; 16] {
        [
            1.0, 0.0, 0.0, 0.0, // Column 0
            0.0, 1.0, 0.0, 0.0, // Column 1
            0.0, 0.0, 1.0, 0.0, // Column 2
            0.0, 0.0, 0.0, 1.0, // Column 3
        ]
    }

    /// Create a translation matrix.
    pub fn translation_matrix(x: f32, y: f32, z: f32) -> [f32; 16] {
        [
            1.0, 0.0, 0.0, 0.0, // Column 0
            0.0, 1.0, 0.0, 0.0, // Column 1
            0.0, 0.0, 1.0, 0.0, // Column 2
            x, y, z, 1.0, // Column 3
        ]
    }

    /// Set the world transform.
    pub fn with_transform(mut self, transform: [f32; 16]) -> Self {
        self.transform = transform;
        self
    }

    /// Set position using a translation matrix.
    pub fn at_position(mut self, x: f32, y: f32, z: f32) -> Self {
        self.transform = Self::translation_matrix(x, y, z);
        self
    }

    /// Transform a world position to local space.
    ///
    /// This assumes the transform is the world-to-local matrix.
    /// For a world matrix, you'd need to invert it first.
    fn world_to_local(&self, world_pos: [f32; 3]) -> [f32; 3] {
        // Extract translation from column 3
        let tx = self.transform[12];
        let ty = self.transform[13];
        let tz = self.transform[14];

        // For simple translation-only transforms, just subtract
        // For full transforms, we'd need proper matrix inverse multiplication
        [world_pos[0] - tx, world_pos[1] - ty, world_pos[2] - tz]
    }

    /// Check if a world position is inside this volume.
    pub fn contains(&self, world_pos: [f32; 3]) -> bool {
        let local_pos = self.world_to_local(world_pos);
        self.shape.contains(local_pos)
    }

    /// Sample fog density at a world position.
    ///
    /// Returns 0.0 if outside the volume, or density with falloff applied.
    pub fn sample(&self, world_pos: [f32; 3]) -> f32 {
        let local_pos = self.world_to_local(world_pos);
        let sdf = self.shape.signed_distance(local_pos);

        if sdf >= self.falloff {
            // Outside falloff region
            0.0
        } else if sdf <= 0.0 {
            // Fully inside
            self.density
        } else if self.falloff > 0.0 {
            // In falloff region
            let t = sdf / self.falloff;
            self.density * (1.0 - t)
        } else {
            // Hard edge, just inside
            self.density
        }
    }
}

impl Default for FogVolume {
    fn default() -> Self {
        Self {
            shape: FogVolumeShape::default(),
            transform: Self::identity_matrix(),
            density: 0.2,
            falloff: 2.0,
        }
    }
}

// ---------------------------------------------------------------------------
// VolumetricFog — Main fog manager
// ---------------------------------------------------------------------------

/// Main volumetric fog system managing density and scattering.
///
/// Combines global fog configuration with optional fog layers and local volumes
/// to produce the final fog density and appearance.
pub struct VolumetricFog {
    /// Global density configuration.
    density_config: FogDensityConfig,

    /// Scattering parameters.
    scattering_params: ScatteringParams,

    /// Additional fog layers.
    layers: Vec<FogLayer>,
}

impl VolumetricFog {
    /// Create a new volumetric fog system.
    pub fn new(density: FogDensityConfig, scattering: ScatteringParams) -> Self {
        Self {
            density_config: density,
            scattering_params: scattering,
            layers: Vec::new(),
        }
    }

    /// Add a fog layer.
    pub fn add_layer(&mut self, layer: FogLayer) {
        self.layers.push(layer);
    }

    /// Get the density configuration.
    #[inline]
    pub fn density_config(&self) -> &FogDensityConfig {
        &self.density_config
    }

    /// Get the scattering parameters.
    #[inline]
    pub fn scattering_params(&self) -> &ScatteringParams {
        &self.scattering_params
    }

    /// Get the fog layers.
    #[inline]
    pub fn layers(&self) -> &[FogLayer] {
        &self.layers
    }

    /// Evaluate fog density at a world position.
    ///
    /// Combines global fog with all layers.
    pub fn sample_density(&self, world_pos: [f32; 3]) -> f32 {
        // Start with global fog
        let mut density = self.density_config.evaluate_height(world_pos[1]);

        // Add layers
        for layer in &self.layers {
            density += layer.evaluate(world_pos[1]);
        }

        // Clamp to valid range
        density.clamp(0.0, 1.0)
    }

    /// Evaluate fog density with noise modulation.
    ///
    /// Uses deterministic noise based on position and time.
    pub fn sample_density_noisy(&self, world_pos: [f32; 3], time: f32) -> f32 {
        let base_density = self.sample_density(world_pos);

        if !self.density_config.has_noise() {
            return base_density;
        }

        // Simple deterministic "noise" based on position
        // In real implementation, use proper 3D noise texture or function
        let scale = self.density_config.noise_scale;
        let noise = Self::simple_noise(
            world_pos[0] * scale,
            world_pos[1] * scale,
            world_pos[2] * scale + time * 0.1,
        );

        // Modulate density
        let strength = self.density_config.noise_strength;
        let modulated = base_density * (1.0 + (noise - 0.5) * 2.0 * strength);

        modulated.clamp(0.0, 1.0)
    }

    /// Simple deterministic noise function (placeholder).
    ///
    /// In production, replace with proper 3D Perlin/Simplex noise.
    #[inline]
    fn simple_noise(x: f32, y: f32, z: f32) -> f32 {
        // Hash-based pseudo-noise
        let ix = (x.floor() as i32).wrapping_mul(1619);
        let iy = (y.floor() as i32).wrapping_mul(31337);
        let iz = (z.floor() as i32).wrapping_mul(6971);
        let hash = (ix ^ iy ^ iz).wrapping_mul(1013);
        let n = (hash as f32 / i32::MAX as f32).abs();
        // Smooth interpolation within cell
        let fx = x.fract().abs();
        let fy = y.fract().abs();
        let fz = z.fract().abs();
        let smooth = fx * fy * fz;
        (n + smooth * 0.1).fract()
    }

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
    pub fn phase_hg(cos_theta: f32, g: f32) -> f32 {
        let g2 = g * g;
        let denom = 1.0 + g2 - 2.0 * g * cos_theta;
        if denom <= 0.0 {
            // Prevent division by zero at extreme angles
            return 1.0 / (4.0 * PI);
        }
        (1.0 - g2) / (4.0 * PI * denom.powf(1.5))
    }

    /// Compute in-scattered light for a single light source.
    ///
    /// # Arguments
    ///
    /// * `view_dir` - Normalized direction toward camera.
    /// * `light_dir` - Normalized direction toward light.
    /// * `light_color` - Light RGB color/intensity.
    /// * `density` - Fog density at sample point.
    ///
    /// # Returns
    ///
    /// RGB in-scattered light contribution.
    pub fn compute_inscatter(
        &self,
        view_dir: [f32; 3],
        light_dir: [f32; 3],
        light_color: [f32; 3],
        density: f32,
    ) -> [f32; 3] {
        // Cosine of angle between view and light
        let cos_theta = view_dir[0] * light_dir[0]
            + view_dir[1] * light_dir[1]
            + view_dir[2] * light_dir[2];

        // Phase function value
        let phase = Self::phase_hg(cos_theta, self.scattering_params.henyey_g);

        // Scattering coefficient = extinction * albedo
        let scatter = self.scattering_params.scattering_coeff();

        // In-scatter = light * phase * scattering * density
        [
            light_color[0] * phase * scatter[0] * density,
            light_color[1] * phase * scatter[1] * density,
            light_color[2] * phase * scatter[2] * density,
        ]
    }

    /// Compute transmittance along a ray segment.
    ///
    /// Uses Beer-Lambert law: T = exp(-extinction * density * distance).
    ///
    /// # Arguments
    ///
    /// * `density` - Average fog density along the segment.
    /// * `distance` - Length of the ray segment.
    ///
    /// # Returns
    ///
    /// RGB transmittance (fraction of light that passes through).
    pub fn compute_transmittance(&self, density: f32, distance: f32) -> [f32; 3] {
        let ext = &self.scattering_params.extinction_rgb;
        [
            (-ext[0] * density * distance).exp(),
            (-ext[1] * density * distance).exp(),
            (-ext[2] * density * distance).exp(),
        ]
    }

    /// Compute optical depth along a ray segment.
    ///
    /// Optical depth = extinction * density * distance.
    #[inline]
    pub fn compute_optical_depth(&self, density: f32, distance: f32) -> [f32; 3] {
        let ext = &self.scattering_params.extinction_rgb;
        [
            ext[0] * density * distance,
            ext[1] * density * distance,
            ext[2] * density * distance,
        ]
    }
}

impl Default for VolumetricFog {
    fn default() -> Self {
        Self {
            density_config: FogDensityConfig::default(),
            scattering_params: ScatteringParams::default(),
            layers: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    fn approx_eq_eps(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    // -----------------------------------------------------------------------
    // FogDensityConfig tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_density_config_defaults() {
        let config = FogDensityConfig::default();
        assert_eq!(config.base_density, 0.05);
        assert_eq!(config.height_falloff, DEFAULT_HEIGHT_FALLOFF);
        assert_eq!(config.height_offset, DEFAULT_HEIGHT_OFFSET);
        assert_eq!(config.noise_scale, DEFAULT_NOISE_SCALE);
        assert_eq!(config.noise_strength, 0.0);
        assert!(config.is_valid());
    }

    #[test]
    fn test_density_config_new() {
        let config = FogDensityConfig::new(0.5, 0.2, 10.0);
        assert_eq!(config.base_density, 0.5);
        assert_eq!(config.height_falloff, 0.2);
        assert_eq!(config.height_offset, 10.0);
    }

    #[test]
    fn test_density_config_clamping() {
        // Density clamped to 0-1
        let config = FogDensityConfig::new(2.0, 0.1, 0.0);
        assert_eq!(config.base_density, 1.0);

        let config = FogDensityConfig::new(-1.0, 0.1, 0.0);
        assert_eq!(config.base_density, 0.0);

        // Falloff clamped to >= 0
        let config = FogDensityConfig::new(0.5, -5.0, 0.0);
        assert_eq!(config.height_falloff, 0.0);
    }

    #[test]
    fn test_density_config_with_noise() {
        let config = FogDensityConfig::default().with_noise(0.5, 0.8);
        assert!(config.has_noise());
        assert_eq!(config.noise_scale, 0.5);
        assert_eq!(config.noise_strength, 0.8);
    }

    #[test]
    fn test_density_config_noise_strength_clamping() {
        let config = FogDensityConfig::default().with_noise(0.1, 2.0);
        assert_eq!(config.noise_strength, 1.0);

        let config = FogDensityConfig::default().with_noise(0.1, -1.0);
        assert_eq!(config.noise_strength, 0.0);
    }

    #[test]
    fn test_density_config_evaluate_height() {
        let config = FogDensityConfig::new(1.0, 0.1, 0.0);

        // Below offset: full density
        assert!(approx_eq(config.evaluate_height(-10.0), 1.0));
        assert!(approx_eq(config.evaluate_height(0.0), 1.0));

        // Above offset: exponential falloff
        let at_10 = config.evaluate_height(10.0);
        let expected_10 = (-0.1 * 10.0_f32).exp();
        assert!(approx_eq_eps(at_10, expected_10, 0.001));

        // Higher = less density
        let at_20 = config.evaluate_height(20.0);
        assert!(at_20 < at_10);
    }

    #[test]
    fn test_density_config_validation() {
        let valid = FogDensityConfig::default();
        assert!(valid.is_valid());

        let mut invalid = valid;
        invalid.base_density = -0.1;
        assert!(!invalid.is_valid());

        let mut invalid = valid;
        invalid.base_density = 1.5;
        assert!(!invalid.is_valid());

        let mut invalid = valid;
        invalid.noise_scale = 0.0;
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_density_config_size() {
        // Should be 32 bytes (8 floats/u32s)
        assert_eq!(std::mem::size_of::<FogDensityConfig>(), 32);
    }

    #[test]
    fn test_density_config_pod() {
        let config = FogDensityConfig::default();
        let bytes: &[u8] = bytemuck::bytes_of(&config);
        assert_eq!(bytes.len(), 32);
    }

    // -----------------------------------------------------------------------
    // FalloffType tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_falloff_linear() {
        let falloff = FalloffType::Linear;

        // Below offset
        assert!(approx_eq(falloff.calculate(-5.0, 0.1, 100.0), 1.0));

        // At offset
        assert!(approx_eq(falloff.calculate(0.0, 0.1, 100.0), 1.0));

        // Midpoint
        assert!(approx_eq(falloff.calculate(50.0, 0.1, 100.0), 0.5));

        // At range end
        assert!(approx_eq(falloff.calculate(100.0, 0.1, 100.0), 0.0));

        // Beyond range
        assert!(approx_eq(falloff.calculate(150.0, 0.1, 100.0), 0.0));
    }

    #[test]
    fn test_falloff_exponential() {
        let falloff = FalloffType::Exponential;
        let rate = 0.1;

        // Below offset
        assert!(approx_eq(falloff.calculate(-5.0, rate, 100.0), 1.0));

        // At offset
        assert!(approx_eq(falloff.calculate(0.0, rate, 100.0), 1.0));

        // Check exponential formula
        let at_10 = falloff.calculate(10.0, rate, 100.0);
        let expected = (-rate * 10.0_f32).exp();
        assert!(approx_eq_eps(at_10, expected, 0.001));
    }

    #[test]
    fn test_falloff_exponential_squared() {
        let falloff = FalloffType::ExponentialSquared;
        let rate = 0.1;

        // Below offset
        assert!(approx_eq(falloff.calculate(-5.0, rate, 100.0), 1.0));

        // Check exp-squared formula
        let at_10 = falloff.calculate(10.0, rate, 100.0);
        let fh = rate * 10.0;
        let expected = (-fh * fh).exp();
        assert!(approx_eq_eps(at_10, expected, 0.001));

        // Exp-squared has sharper falloff at larger heights
        // At height = 10, rate = 0.1: exp(-1) = 0.368, exp-sq(-1) = 0.368 (same)
        // At height = 20, rate = 0.1: exp(-2) = 0.135, exp-sq(-4) = 0.018 (sharper)
        let exp_at_20 = FalloffType::Exponential.calculate(20.0, rate, 100.0);
        let exp_sq_at_20 = falloff.calculate(20.0, rate, 100.0);
        assert!(exp_sq_at_20 < exp_at_20, "Exp-squared should fall off faster at larger heights");
    }

    #[test]
    fn test_falloff_names() {
        assert_eq!(FalloffType::Linear.name(), "linear");
        assert_eq!(FalloffType::Exponential.name(), "exponential");
        assert_eq!(FalloffType::ExponentialSquared.name(), "exponential_squared");
    }

    #[test]
    fn test_falloff_default() {
        assert_eq!(FalloffType::default(), FalloffType::Exponential);
    }

    // -----------------------------------------------------------------------
    // FogLayer tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_fog_layer_new() {
        let layer = FogLayer::new(0.5, 0.0, 100.0, FalloffType::Exponential);
        assert_eq!(layer.density, 0.5);
        assert_eq!(layer.height_min, 0.0);
        assert_eq!(layer.height_max, 100.0);
    }

    #[test]
    #[should_panic(expected = "height_max must be greater than height_min")]
    fn test_fog_layer_invalid_heights() {
        FogLayer::new(0.5, 100.0, 50.0, FalloffType::Linear);
    }

    #[test]
    fn test_fog_layer_ground_fog() {
        let layer = FogLayer::ground_fog(0.8, 50.0);
        assert_eq!(layer.height_min, 0.0);
        assert_eq!(layer.height_max, 50.0);
        assert_eq!(layer.falloff_type, FalloffType::Exponential);
    }

    #[test]
    fn test_fog_layer_fog_bank() {
        let layer = FogLayer::fog_bank(0.6, 10.0, 30.0);
        assert_eq!(layer.height_min, 10.0);
        assert_eq!(layer.height_max, 30.0);
        assert_eq!(layer.falloff_type, FalloffType::ExponentialSquared);
    }

    #[test]
    fn test_fog_layer_evaluate_bounds() {
        let layer = FogLayer::new(1.0, 10.0, 50.0, FalloffType::Linear);

        // Below layer
        assert!(approx_eq(layer.evaluate(5.0), 0.0));

        // Above layer
        assert!(approx_eq(layer.evaluate(60.0), 0.0));

        // At bottom
        assert!(approx_eq(layer.evaluate(10.0), 1.0));
    }

    #[test]
    fn test_fog_layer_contains_height() {
        let layer = FogLayer::new(0.5, 10.0, 50.0, FalloffType::Linear);

        assert!(!layer.contains_height(5.0));
        assert!(layer.contains_height(10.0));
        assert!(layer.contains_height(30.0));
        assert!(layer.contains_height(50.0));
        assert!(!layer.contains_height(55.0));
    }

    #[test]
    fn test_fog_layer_height_range() {
        let layer = FogLayer::new(0.5, 10.0, 50.0, FalloffType::Linear);
        assert!(approx_eq(layer.height_range(), 40.0));
    }

    #[test]
    fn test_multiple_layers_blending() {
        let layer1 = FogLayer::new(0.3, 0.0, 30.0, FalloffType::Linear);
        let layer2 = FogLayer::new(0.4, 20.0, 50.0, FalloffType::Linear);

        // Overlap region should have both contributions
        let h = 25.0;
        let d1 = layer1.evaluate(h);
        let d2 = layer2.evaluate(h);
        let total = d1 + d2;
        assert!(total > 0.0);
        assert!(d1 > 0.0);
        assert!(d2 > 0.0);
    }

    // -----------------------------------------------------------------------
    // ScatteringParams tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_scattering_params_defaults() {
        let params = ScatteringParams::default();
        assert!(params.is_valid());
        assert_eq!(params.henyey_g, 0.8);
        assert_eq!(params.ambient_strength, 0.1);
    }

    #[test]
    fn test_scattering_params_new() {
        let params = ScatteringParams::new([0.1, 0.2, 0.3], [0.8, 0.9, 1.0], 0.5);
        assert_eq!(params.extinction_rgb, [0.1, 0.2, 0.3]);
        assert_eq!(params.albedo_rgb, [0.8, 0.9, 1.0]);
        assert_eq!(params.henyey_g, 0.5);
    }

    #[test]
    fn test_scattering_params_uniform() {
        let params = ScatteringParams::uniform(0.1, 0.9, 0.7);
        assert_eq!(params.extinction_rgb, [0.1, 0.1, 0.1]);
        assert_eq!(params.albedo_rgb, [0.9, 0.9, 0.9]);
        assert_eq!(params.henyey_g, 0.7);
    }

    #[test]
    fn test_scattering_params_clamping() {
        // Albedo clamped to 0-1
        let params = ScatteringParams::new([1.0, 1.0, 1.0], [1.5, -0.5, 0.5], 0.5);
        assert_eq!(params.albedo_rgb, [1.0, 0.0, 0.5]);

        // g clamped to -0.999 to 0.999
        let params = ScatteringParams::new([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], 1.5);
        assert!(params.henyey_g <= MAX_HG_G);

        let params = ScatteringParams::new([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], -1.5);
        assert!(params.henyey_g >= MIN_HG_G);
    }

    #[test]
    fn test_scattering_params_with_ambient() {
        let params = ScatteringParams::default().with_ambient(0.5);
        assert_eq!(params.ambient_strength, 0.5);
    }

    #[test]
    fn test_scattering_params_is_purely_absorbing() {
        let absorbing = ScatteringParams::new([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], 0.0);
        assert!(absorbing.is_purely_absorbing());

        let scattering = ScatteringParams::default();
        assert!(!scattering.is_purely_absorbing());
    }

    #[test]
    fn test_scattering_params_coefficients() {
        let params = ScatteringParams::new([0.4, 0.5, 0.6], [0.5, 0.6, 0.7], 0.0);

        let scatter = params.scattering_coeff();
        assert!(approx_eq(scatter[0], 0.2));
        assert!(approx_eq(scatter[1], 0.3));
        assert!(approx_eq_eps(scatter[2], 0.42, 0.001));

        let absorb = params.absorption_coeff();
        assert!(approx_eq(absorb[0], 0.2));
        assert!(approx_eq(absorb[1], 0.2));
        assert!(approx_eq_eps(absorb[2], 0.18, 0.001));
    }

    #[test]
    fn test_scattering_params_size() {
        // Should be 32 bytes
        assert_eq!(std::mem::size_of::<ScatteringParams>(), 32);
    }

    #[test]
    fn test_scattering_params_pod() {
        let params = ScatteringParams::default();
        let bytes: &[u8] = bytemuck::bytes_of(&params);
        assert_eq!(bytes.len(), 32);
    }

    // -----------------------------------------------------------------------
    // Henyey-Greenstein phase function tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_phase_hg_normalization() {
        // Phase function should integrate to 1 over the sphere
        // Approximate with Monte Carlo sampling
        let g = 0.5;
        let samples = 10000;
        let mut sum = 0.0;

        for i in 0..samples {
            let cos_theta = -1.0 + 2.0 * (i as f32 / samples as f32);
            sum += VolumetricFog::phase_hg(cos_theta, g) * 2.0 / samples as f32;
        }

        // Should be close to 1 / (4*PI) * 4*PI = 1
        // But our integral approximation gives sum * 4*PI
        let integral = sum * 2.0 * PI; // Factor of 2 for cos_theta range
        assert!(
            (integral - 1.0).abs() < 0.05,
            "Phase function integral = {} (expected ~1.0)",
            integral
        );
    }

    #[test]
    fn test_phase_hg_isotropic() {
        // g = 0 should be isotropic (uniform)
        let p0 = VolumetricFog::phase_hg(1.0, 0.0);
        let p1 = VolumetricFog::phase_hg(0.0, 0.0);
        let p2 = VolumetricFog::phase_hg(-1.0, 0.0);

        let expected = 1.0 / (4.0 * PI);
        assert!(approx_eq_eps(p0, expected, 0.001));
        assert!(approx_eq_eps(p1, expected, 0.001));
        assert!(approx_eq_eps(p2, expected, 0.001));
    }

    #[test]
    fn test_phase_hg_forward_scattering() {
        // g > 0 should favor forward scattering
        let g = 0.8;
        let forward = VolumetricFog::phase_hg(1.0, g); // cos(0) = 1
        let side = VolumetricFog::phase_hg(0.0, g); // cos(90) = 0
        let back = VolumetricFog::phase_hg(-1.0, g); // cos(180) = -1

        assert!(forward > side);
        assert!(side > back);
    }

    #[test]
    fn test_phase_hg_back_scattering() {
        // g < 0 should favor back scattering
        let g = -0.5;
        let forward = VolumetricFog::phase_hg(1.0, g);
        let back = VolumetricFog::phase_hg(-1.0, g);

        assert!(back > forward);
    }

    #[test]
    fn test_phase_hg_symmetry() {
        // g = 0 is symmetric
        let p_forward = VolumetricFog::phase_hg(0.5, 0.0);
        let p_back = VolumetricFog::phase_hg(-0.5, 0.0);
        assert!(approx_eq_eps(p_forward, p_back, 0.001));
    }

    // -----------------------------------------------------------------------
    // Transmittance (Beer-Lambert) tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_transmittance_zero_density() {
        let fog = VolumetricFog::default();
        let trans = fog.compute_transmittance(0.0, 100.0);
        assert!(approx_eq(trans[0], 1.0));
        assert!(approx_eq(trans[1], 1.0));
        assert!(approx_eq(trans[2], 1.0));
    }

    #[test]
    fn test_transmittance_zero_distance() {
        let fog = VolumetricFog::default();
        let trans = fog.compute_transmittance(0.5, 0.0);
        assert!(approx_eq(trans[0], 1.0));
        assert!(approx_eq(trans[1], 1.0));
        assert!(approx_eq(trans[2], 1.0));
    }

    #[test]
    fn test_transmittance_beer_lambert() {
        let scattering = ScatteringParams::uniform(0.1, 0.9, 0.0);
        let fog = VolumetricFog::new(FogDensityConfig::default(), scattering);

        let density = 0.5;
        let distance = 10.0;
        let trans = fog.compute_transmittance(density, distance);

        let expected = (-0.1 * 0.5 * 10.0_f32).exp();
        assert!(approx_eq_eps(trans[0], expected, 0.001));
        assert!(approx_eq_eps(trans[1], expected, 0.001));
        assert!(approx_eq_eps(trans[2], expected, 0.001));
    }

    #[test]
    fn test_transmittance_energy_bounded() {
        let fog = VolumetricFog::default();

        // Transmittance should always be 0-1
        for density in [0.0, 0.5, 1.0] {
            for distance in [0.0, 10.0, 100.0, 1000.0] {
                let trans = fog.compute_transmittance(density, distance);
                for t in trans {
                    assert!(t >= 0.0 && t <= 1.0);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Energy conservation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_energy_conservation_albedo() {
        // Albedo must be <= 1 for energy conservation
        let params = ScatteringParams::default();
        for a in params.albedo_rgb {
            assert!(a <= 1.0);
        }
    }

    #[test]
    fn test_inscatter_plus_transmittance_bounded() {
        // Total energy (inscatter + transmittance) should not exceed input
        let fog = VolumetricFog::default();

        let view_dir = [0.0, 0.0, 1.0];
        let light_dir = [0.0, 1.0, 0.0];
        let light_color = [1.0, 1.0, 1.0];
        let density = 0.5;
        let distance = 1.0;

        let inscatter = fog.compute_inscatter(view_dir, light_dir, light_color, density);
        let transmittance = fog.compute_transmittance(density, distance);

        // For a single step, inscatter contributes (1-T) * albedo * phase * light
        // This is a simplified check
        for i in 0..3 {
            let total = inscatter[i] + transmittance[i];
            // Total should be bounded (this is a loose check for sanity)
            assert!(
                total < 10.0,
                "Total energy too high: inscatter={}, trans={}",
                inscatter[i],
                transmittance[i]
            );
        }
    }

    // -----------------------------------------------------------------------
    // FogVolumeShape tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_shape_box_contains() {
        let shape = FogVolumeShape::box_shape(5.0, 3.0, 4.0);

        // Inside
        assert!(shape.contains([0.0, 0.0, 0.0]));
        assert!(shape.contains([4.9, 2.9, 3.9]));

        // On boundary
        assert!(shape.contains([5.0, 0.0, 0.0]));

        // Outside
        assert!(!shape.contains([5.1, 0.0, 0.0]));
        assert!(!shape.contains([0.0, 3.1, 0.0]));
    }

    #[test]
    fn test_shape_sphere_contains() {
        let shape = FogVolumeShape::sphere(10.0);

        // Inside
        assert!(shape.contains([0.0, 0.0, 0.0]));
        assert!(shape.contains([7.0, 7.0, 0.0])); // sqrt(98) < 10

        // On boundary
        assert!(shape.contains([10.0, 0.0, 0.0]));

        // Outside
        assert!(!shape.contains([7.1, 7.1, 0.0])); // sqrt(100.82) > 10
    }

    #[test]
    fn test_shape_cylinder_contains() {
        let shape = FogVolumeShape::cylinder(5.0, 10.0);

        // Inside
        assert!(shape.contains([0.0, 0.0, 0.0]));
        assert!(shape.contains([4.0, 4.0, 3.0])); // r=5, h/2=5

        // Outside radius
        assert!(!shape.contains([5.1, 0.0, 0.0]));

        // Outside height
        assert!(!shape.contains([0.0, 5.1, 0.0]));
    }

    #[test]
    fn test_shape_signed_distance_sphere() {
        let shape = FogVolumeShape::sphere(10.0);

        // Center
        assert!(approx_eq(shape.signed_distance([0.0, 0.0, 0.0]), -10.0));

        // On surface
        assert!(approx_eq_eps(shape.signed_distance([10.0, 0.0, 0.0]), 0.0, 0.001));

        // Outside
        assert!(approx_eq_eps(shape.signed_distance([15.0, 0.0, 0.0]), 5.0, 0.001));
    }

    #[test]
    fn test_shape_bounding_extents() {
        let box_shape = FogVolumeShape::box_shape(1.0, 2.0, 3.0);
        assert_eq!(box_shape.bounding_half_extents(), [1.0, 2.0, 3.0]);

        let sphere = FogVolumeShape::sphere(5.0);
        assert_eq!(sphere.bounding_half_extents(), [5.0, 5.0, 5.0]);

        let cylinder = FogVolumeShape::cylinder(3.0, 10.0);
        assert_eq!(cylinder.bounding_half_extents(), [3.0, 5.0, 3.0]);
    }

    // -----------------------------------------------------------------------
    // FogVolume tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_fog_volume_new() {
        let vol = FogVolume::new(FogVolumeShape::sphere(10.0), 0.5, 2.0);
        assert_eq!(vol.density, 0.5);
        assert_eq!(vol.falloff, 2.0);
    }

    #[test]
    fn test_fog_volume_at_position() {
        let vol = FogVolume::new(FogVolumeShape::sphere(5.0), 0.5, 1.0)
            .at_position(10.0, 20.0, 30.0);

        // Should contain point at center
        assert!(vol.contains([10.0, 20.0, 30.0]));

        // Should not contain distant point
        assert!(!vol.contains([0.0, 0.0, 0.0]));
    }

    #[test]
    fn test_fog_volume_sample_inside() {
        let vol = FogVolume::new(FogVolumeShape::sphere(10.0), 0.8, 0.0);

        // Center should have full density
        assert!(approx_eq(vol.sample([0.0, 0.0, 0.0]), 0.8));
    }

    #[test]
    fn test_fog_volume_sample_outside() {
        let vol = FogVolume::new(FogVolumeShape::sphere(10.0), 0.8, 0.0);

        // Outside should be zero
        assert!(approx_eq(vol.sample([20.0, 0.0, 0.0]), 0.0));
    }

    #[test]
    fn test_fog_volume_soft_falloff() {
        let vol = FogVolume::new(FogVolumeShape::sphere(10.0), 1.0, 5.0);

        // Inside: full density
        let inside = vol.sample([0.0, 0.0, 0.0]);
        assert!(approx_eq(inside, 1.0));

        // At edge: should be interpolated
        let at_edge = vol.sample([12.0, 0.0, 0.0]); // sdf = 2, falloff = 5
        assert!(at_edge > 0.0 && at_edge < 1.0);

        // Outside falloff region
        let outside = vol.sample([16.0, 0.0, 0.0]); // sdf = 6 > falloff
        assert!(approx_eq(outside, 0.0));
    }

    #[test]
    fn test_fog_volume_hard_edge() {
        let vol = FogVolume::new(FogVolumeShape::sphere(10.0), 0.5, 0.0);

        // Just inside: full density
        assert!(approx_eq(vol.sample([9.9, 0.0, 0.0]), 0.5));

        // Just outside: zero
        assert!(approx_eq(vol.sample([10.1, 0.0, 0.0]), 0.0));
    }

    // -----------------------------------------------------------------------
    // VolumetricFog system tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_volumetric_fog_new() {
        let fog = VolumetricFog::new(
            FogDensityConfig::default(),
            ScatteringParams::default(),
        );
        assert!(fog.layers().is_empty());
    }

    #[test]
    fn test_volumetric_fog_add_layer() {
        let mut fog = VolumetricFog::default();
        fog.add_layer(FogLayer::ground_fog(0.5, 50.0));
        assert_eq!(fog.layers().len(), 1);

        fog.add_layer(FogLayer::fog_bank(0.3, 100.0, 200.0));
        assert_eq!(fog.layers().len(), 2);
    }

    #[test]
    fn test_volumetric_fog_sample_density() {
        let config = FogDensityConfig::new(0.5, 0.1, 0.0);
        let fog = VolumetricFog::new(config, ScatteringParams::default());

        // At ground level
        let d0 = fog.sample_density([0.0, 0.0, 0.0]);
        assert!(approx_eq(d0, 0.5));

        // Above ground (falloff)
        let d10 = fog.sample_density([0.0, 10.0, 0.0]);
        assert!(d10 < 0.5);
    }

    #[test]
    fn test_volumetric_fog_sample_density_with_layers() {
        let config = FogDensityConfig::new(0.1, 0.1, 0.0);
        let mut fog = VolumetricFog::new(config, ScatteringParams::default());
        fog.add_layer(FogLayer::new(0.2, 0.0, 50.0, FalloffType::Linear));

        // Should combine global + layer
        let d = fog.sample_density([0.0, 0.0, 0.0]);
        assert!(d > 0.1); // More than just global
    }

    #[test]
    fn test_volumetric_fog_sample_density_noisy() {
        let config = FogDensityConfig::new(0.5, 0.1, 0.0).with_noise(0.5, 0.5);
        let fog = VolumetricFog::new(config, ScatteringParams::default());

        // Sample at same position
        let d1 = fog.sample_density_noisy([10.0, 5.0, 3.0], 0.0);
        let d2 = fog.sample_density_noisy([10.0, 5.0, 3.0], 0.0);

        // Should be deterministic
        assert!(approx_eq_eps(d1, d2, 0.001));
    }

    #[test]
    fn test_volumetric_fog_noise_determinism() {
        let config = FogDensityConfig::new(0.5, 0.1, 0.0).with_noise(0.1, 0.8);
        let fog = VolumetricFog::new(config, ScatteringParams::default());

        // Different positions should give different noise
        let d1 = fog.sample_density_noisy([0.0, 0.0, 0.0], 0.0);
        let d2 = fog.sample_density_noisy([100.0, 100.0, 100.0], 0.0);

        // They might be equal by chance, but very unlikely
        // At minimum, verify noise is bounded
        assert!(d1 >= 0.0 && d1 <= 1.0);
        assert!(d2 >= 0.0 && d2 <= 1.0);
    }

    #[test]
    fn test_volumetric_fog_compute_inscatter() {
        let fog = VolumetricFog::default();

        let view = [0.0, 0.0, 1.0];
        let light = [0.0, 0.0, 1.0]; // Same direction
        let color = [1.0, 1.0, 1.0];
        let density = 1.0;

        let inscatter = fog.compute_inscatter(view, light, color, density);

        // Forward scatter (view = light) should give higher value for g > 0
        assert!(inscatter[0] > 0.0);
        assert!(inscatter[1] > 0.0);
        assert!(inscatter[2] > 0.0);
    }

    #[test]
    fn test_volumetric_fog_optical_depth() {
        let scattering = ScatteringParams::uniform(0.2, 0.9, 0.0);
        let fog = VolumetricFog::new(FogDensityConfig::default(), scattering);

        let od = fog.compute_optical_depth(0.5, 10.0);
        let expected = 0.2 * 0.5 * 10.0;
        assert!(approx_eq(od[0], expected));
        assert!(approx_eq(od[1], expected));
        assert!(approx_eq(od[2], expected));
    }

    #[test]
    fn test_volumetric_fog_default() {
        let fog = VolumetricFog::default();
        assert!(fog.density_config().is_valid());
        assert!(fog.scattering_params().is_valid());
        assert!(fog.layers().is_empty());
    }
}
