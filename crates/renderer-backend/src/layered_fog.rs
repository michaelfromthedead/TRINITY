//! Layered Fog System (T-ENV-3.3)
//!
//! This module provides a multi-layer fog system for atmospheric rendering with:
//! - Ground fog: Dense, low altitude, sharp falloff (mist, morning fog)
//! - Mid haze: Subtle, mid altitude, gradual falloff (distant mountains)
//! - High haze: Thin, high altitude (upper atmosphere blue haze)
//!
//! # Overview
//!
//! Each fog layer has independent:
//! - Color (RGB)
//! - Density (max density at layer center)
//! - Height (base height and thickness)
//! - Falloff (rate at which density decreases)
//! - Quality toggle (for mobile optimization)
//!
//! Layers are composited additively into the froxel density field, allowing
//! complex atmospheric effects like morning mist with mountain haze.
//!
//! # GPU Integration
//!
//! All configuration structs are `repr(C)` with `bytemuck::Pod` for direct
//! GPU upload. The system integrates with the froxel volume system (T-ENV-1.5).
//!
//! # Quality Tiers
//!
//! Per-layer quality toggles allow disabling expensive layers on mobile:
//! - Ultra: All three layers with full quality
//! - High: Ground + Mid layers
//! - Medium: Ground + simplified Mid
//! - Low: Ground layer only

use bytemuck::{Pod, Zeroable};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Minimum valid density value.
pub const MIN_DENSITY: f32 = 0.0;

/// Maximum valid density value.
pub const MAX_DENSITY: f32 = 1.0;

/// Default ground fog base height (meters).
pub const DEFAULT_GROUND_BASE_HEIGHT: f32 = 0.0;

/// Default ground fog thickness (meters).
pub const DEFAULT_GROUND_THICKNESS: f32 = 50.0;

/// Default ground fog density.
pub const DEFAULT_GROUND_DENSITY: f32 = 0.3;

/// Default ground fog falloff rate.
pub const DEFAULT_GROUND_FALLOFF: f32 = 0.1;

/// Default mid haze base height (meters).
pub const DEFAULT_MID_BASE_HEIGHT: f32 = 100.0;

/// Default mid haze thickness (meters).
pub const DEFAULT_MID_THICKNESS: f32 = 500.0;

/// Default mid haze density.
pub const DEFAULT_MID_DENSITY: f32 = 0.05;

/// Default mid haze falloff rate.
pub const DEFAULT_MID_FALLOFF: f32 = 0.005;

/// Default high haze base height (meters).
pub const DEFAULT_HIGH_BASE_HEIGHT: f32 = 1000.0;

/// Default high haze thickness (meters).
pub const DEFAULT_HIGH_THICKNESS: f32 = 5000.0;

/// Default high haze density.
pub const DEFAULT_HIGH_DENSITY: f32 = 0.01;

/// Default high haze falloff rate.
pub const DEFAULT_HIGH_FALLOFF: f32 = 0.001;

/// Default white fog color.
pub const DEFAULT_FOG_COLOR_WHITE: [f32; 3] = [1.0, 1.0, 1.0];

/// Default bluish haze color for atmospheric scattering.
pub const DEFAULT_HAZE_COLOR_BLUE: [f32; 3] = [0.7, 0.8, 1.0];

// ---------------------------------------------------------------------------
// FalloffMode — Falloff curve types
// ---------------------------------------------------------------------------

/// Falloff curve types for fog density calculation.
///
/// Different falloff modes produce different density profiles with height.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u32)]
pub enum FalloffMode {
    /// Linear falloff: density decreases linearly with distance from layer center.
    /// density = max(0, 1 - distance / max_distance)
    Linear = 0,
    /// Exponential falloff: density decreases exponentially with distance.
    /// density = exp(-rate * distance)
    /// More natural appearance, commonly used for atmospheric effects.
    #[default]
    Exponential = 1,
    /// Exponential squared falloff: sharper transition at layer boundaries.
    /// density = exp(-(rate * distance)^2)
    /// Good for low-lying fog banks with sharp edges.
    ExponentialSquared = 2,
    /// Smooth step falloff: smooth transition using cubic Hermite interpolation.
    /// density = smoothstep(1, 0, distance / max_distance)
    SmoothStep = 3,
}

impl FalloffMode {
    /// Convert from u32 representation.
    pub fn from_u32(value: u32) -> Self {
        match value {
            0 => FalloffMode::Linear,
            1 => FalloffMode::Exponential,
            2 => FalloffMode::ExponentialSquared,
            3 => FalloffMode::SmoothStep,
            _ => FalloffMode::Exponential,
        }
    }

    /// Get the name string for this falloff mode.
    #[inline]
    pub fn name(&self) -> &'static str {
        match self {
            FalloffMode::Linear => "linear",
            FalloffMode::Exponential => "exponential",
            FalloffMode::ExponentialSquared => "exponential_squared",
            FalloffMode::SmoothStep => "smooth_step",
        }
    }
}

// ---------------------------------------------------------------------------
// BlendMode — Layer composition modes
// ---------------------------------------------------------------------------

/// How fog layers are composited together.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u32)]
pub enum BlendMode {
    /// Additive blending: layer densities are summed.
    /// Best for overlapping atmospheric effects.
    #[default]
    Additive = 0,
    /// Maximum blending: take the maximum density from any layer.
    /// Useful when layers shouldn't stack.
    Max = 1,
    /// Lerp blending: blend between layers based on height.
    /// Creates smooth transitions between distinct fog zones.
    Lerp = 2,
}

impl BlendMode {
    /// Convert from u32 representation.
    pub fn from_u32(value: u32) -> Self {
        match value {
            0 => BlendMode::Additive,
            1 => BlendMode::Max,
            2 => BlendMode::Lerp,
            _ => BlendMode::Additive,
        }
    }

    /// Get the name string for this blend mode.
    #[inline]
    pub fn name(&self) -> &'static str {
        match self {
            BlendMode::Additive => "additive",
            BlendMode::Max => "max",
            BlendMode::Lerp => "lerp",
        }
    }
}

// ---------------------------------------------------------------------------
// FogLayerConfig — GPU-uploadable layer configuration
// ---------------------------------------------------------------------------

/// Configuration for a single fog layer.
///
/// This struct is designed to be uploaded to the GPU as part of a uniform buffer.
/// The layout is `repr(C)` and implements `Pod` for bytemuck compatibility.
///
/// # Memory Layout (32 bytes)
///
/// | Offset | Field        | Size     |
/// |--------|--------------|----------|
/// | 0      | base_height  | 4 bytes  |
/// | 4      | thickness    | 4 bytes  |
/// | 8      | density      | 4 bytes  |
/// | 12     | falloff_rate | 4 bytes  |
/// | 16     | color        | 12 bytes |
/// | 28     | enabled      | 4 bytes  |
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct FogLayerConfig {
    /// Layer bottom height in world space (meters).
    pub base_height: f32,
    /// Layer thickness (meters). Fog extends from base_height to base_height + thickness.
    pub thickness: f32,
    /// Maximum density at layer center (0.0 to 1.0).
    pub density: f32,
    /// Falloff rate controlling how quickly density decreases with distance from center.
    pub falloff_rate: f32,
    /// Fog color (RGB, linear space).
    pub color: [f32; 3],
    /// Whether this layer is enabled (0 = disabled, non-zero = enabled).
    /// Used for quality tier toggling on mobile.
    pub enabled: u32,
}

impl FogLayerConfig {
    /// Create a new fog layer configuration.
    ///
    /// # Arguments
    ///
    /// * `base_height` - Bottom of the layer in world space.
    /// * `thickness` - Height range of the layer.
    /// * `density` - Maximum density (clamped to 0.0-1.0).
    /// * `falloff_rate` - Falloff rate (must be positive).
    /// * `color` - RGB color in linear space.
    pub fn new(
        base_height: f32,
        thickness: f32,
        density: f32,
        falloff_rate: f32,
        color: [f32; 3],
    ) -> Self {
        Self {
            base_height,
            thickness: thickness.max(0.0),
            density: density.clamp(MIN_DENSITY, MAX_DENSITY),
            falloff_rate: falloff_rate.max(0.0),
            color,
            enabled: 1,
        }
    }

    /// Create a disabled layer (zeroed out).
    pub fn disabled() -> Self {
        Self {
            base_height: 0.0,
            thickness: 0.0,
            density: 0.0,
            falloff_rate: 0.0,
            color: [0.0, 0.0, 0.0],
            enabled: 0,
        }
    }

    /// Create a ground fog layer preset.
    ///
    /// Ground fog is dense, low altitude fog with sharp falloff,
    /// simulating morning mist or valley fog.
    pub fn ground_fog() -> Self {
        Self::new(
            DEFAULT_GROUND_BASE_HEIGHT,
            DEFAULT_GROUND_THICKNESS,
            DEFAULT_GROUND_DENSITY,
            DEFAULT_GROUND_FALLOFF,
            DEFAULT_FOG_COLOR_WHITE,
        )
    }

    /// Create a mid haze layer preset.
    ///
    /// Mid haze is subtle, mid altitude haze with gradual falloff,
    /// simulating distant mountain haze or general atmospheric haze.
    pub fn mid_haze() -> Self {
        Self::new(
            DEFAULT_MID_BASE_HEIGHT,
            DEFAULT_MID_THICKNESS,
            DEFAULT_MID_DENSITY,
            DEFAULT_MID_FALLOFF,
            DEFAULT_FOG_COLOR_WHITE,
        )
    }

    /// Create a high haze layer preset.
    ///
    /// High haze is thin, high altitude haze simulating upper
    /// atmosphere Rayleigh scattering (blue haze).
    pub fn high_haze() -> Self {
        Self::new(
            DEFAULT_HIGH_BASE_HEIGHT,
            DEFAULT_HIGH_THICKNESS,
            DEFAULT_HIGH_DENSITY,
            DEFAULT_HIGH_FALLOFF,
            DEFAULT_HAZE_COLOR_BLUE,
        )
    }

    /// Enable or disable this layer.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = if enabled { 1 } else { 0 };
    }

    /// Check if this layer is enabled.
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.enabled != 0
    }

    /// Get the top height of this layer.
    #[inline]
    pub fn top_height(&self) -> f32 {
        self.base_height + self.thickness
    }

    /// Get the center height of this layer.
    #[inline]
    pub fn center_height(&self) -> f32 {
        self.base_height + self.thickness * 0.5
    }

    /// Check if a height is within this layer's bounds.
    #[inline]
    pub fn contains_height(&self, world_y: f32) -> bool {
        world_y >= self.base_height && world_y <= self.top_height()
    }

    /// Validate the layer configuration.
    pub fn is_valid(&self) -> bool {
        self.thickness >= 0.0
            && self.density >= MIN_DENSITY
            && self.density <= MAX_DENSITY
            && self.falloff_rate >= 0.0
            && self.color.iter().all(|c| *c >= 0.0)
    }
}

impl Default for FogLayerConfig {
    fn default() -> Self {
        Self::ground_fog()
    }
}

// ---------------------------------------------------------------------------
// LayeredFogConfig — GPU-uploadable multi-layer configuration
// ---------------------------------------------------------------------------

/// Configuration for the complete layered fog system.
///
/// This struct is designed to be uploaded to the GPU as a uniform buffer.
/// The layout is `repr(C)` and implements `Pod` for bytemuck compatibility.
///
/// # Memory Layout (100 bytes, padded to 112 for alignment)
///
/// | Offset | Field        | Size     |
/// |--------|--------------|----------|
/// | 0      | ground_layer | 32 bytes |
/// | 32     | mid_layer    | 32 bytes |
/// | 64     | high_layer   | 32 bytes |
/// | 96     | blend_mode   | 4 bytes  |
/// | 100    | falloff_mode | 4 bytes  |
/// | 104    | _padding     | 8 bytes  |
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct LayeredFogConfig {
    /// Ground fog layer: low altitude mist.
    pub ground_layer: FogLayerConfig,
    /// Mid haze layer: mountain/distance haze.
    pub mid_layer: FogLayerConfig,
    /// High haze layer: atmospheric/sky haze.
    pub high_layer: FogLayerConfig,
    /// Layer blending mode (0=additive, 1=max, 2=lerp).
    pub blend_mode: u32,
    /// Falloff calculation mode (0=linear, 1=exp, 2=exp2, 3=smooth).
    pub falloff_mode: u32,
    /// Padding for GPU alignment.
    pub _padding: [u32; 2],
}

impl LayeredFogConfig {
    /// Create a new layered fog configuration with all layers.
    pub fn new(
        ground_layer: FogLayerConfig,
        mid_layer: FogLayerConfig,
        high_layer: FogLayerConfig,
    ) -> Self {
        Self {
            ground_layer,
            mid_layer,
            high_layer,
            blend_mode: BlendMode::Additive as u32,
            falloff_mode: FalloffMode::Exponential as u32,
            _padding: [0; 2],
        }
    }

    /// Create a default configuration with all preset layers enabled.
    pub fn default_all_layers() -> Self {
        Self::new(
            FogLayerConfig::ground_fog(),
            FogLayerConfig::mid_haze(),
            FogLayerConfig::high_haze(),
        )
    }

    /// Create a configuration for mobile (ground layer only).
    pub fn mobile_quality() -> Self {
        let mut config = Self::default_all_layers();
        config.mid_layer.set_enabled(false);
        config.high_layer.set_enabled(false);
        config
    }

    /// Create a configuration for medium quality (ground + mid layers).
    pub fn medium_quality() -> Self {
        let mut config = Self::default_all_layers();
        config.high_layer.set_enabled(false);
        config
    }

    /// Set the blend mode.
    pub fn with_blend_mode(mut self, mode: BlendMode) -> Self {
        self.blend_mode = mode as u32;
        self
    }

    /// Set the falloff mode.
    pub fn with_falloff_mode(mut self, mode: FalloffMode) -> Self {
        self.falloff_mode = mode as u32;
        self
    }

    /// Get the blend mode enum.
    #[inline]
    pub fn get_blend_mode(&self) -> BlendMode {
        BlendMode::from_u32(self.blend_mode)
    }

    /// Get the falloff mode enum.
    #[inline]
    pub fn get_falloff_mode(&self) -> FalloffMode {
        FalloffMode::from_u32(self.falloff_mode)
    }

    /// Get the number of enabled layers.
    pub fn enabled_layer_count(&self) -> usize {
        let mut count = 0;
        if self.ground_layer.is_enabled() {
            count += 1;
        }
        if self.mid_layer.is_enabled() {
            count += 1;
        }
        if self.high_layer.is_enabled() {
            count += 1;
        }
        count
    }

    /// Validate all layer configurations.
    pub fn is_valid(&self) -> bool {
        self.ground_layer.is_valid()
            && self.mid_layer.is_valid()
            && self.high_layer.is_valid()
    }
}

impl Default for LayeredFogConfig {
    fn default() -> Self {
        Self::default_all_layers()
    }
}

// ---------------------------------------------------------------------------
// Falloff Functions
// ---------------------------------------------------------------------------

/// Calculate exponential falloff.
///
/// # Arguments
///
/// * `distance` - Distance from layer center (positive).
/// * `rate` - Falloff rate (higher = faster falloff).
///
/// # Returns
///
/// Falloff factor in range [0, 1].
#[inline]
pub fn exponential_falloff(distance: f32, rate: f32) -> f32 {
    if distance <= 0.0 {
        return 1.0;
    }
    (-rate * distance).exp()
}

/// Calculate exponential squared falloff.
///
/// Sharper transition than standard exponential.
///
/// # Arguments
///
/// * `distance` - Distance from layer center (positive).
/// * `rate` - Falloff rate (higher = faster falloff).
///
/// # Returns
///
/// Falloff factor in range [0, 1].
#[inline]
pub fn exponential_squared_falloff(distance: f32, rate: f32) -> f32 {
    if distance <= 0.0 {
        return 1.0;
    }
    let rd = rate * distance;
    (-rd * rd).exp()
}

/// Calculate linear falloff.
///
/// # Arguments
///
/// * `distance` - Distance from layer center (positive).
/// * `max_distance` - Maximum distance (at which falloff reaches 0).
///
/// # Returns
///
/// Falloff factor in range [0, 1].
#[inline]
pub fn linear_falloff(distance: f32, max_distance: f32) -> f32 {
    if distance <= 0.0 {
        return 1.0;
    }
    if max_distance <= 0.0 {
        return 0.0;
    }
    (1.0 - distance / max_distance).max(0.0)
}

/// Calculate smooth step falloff using cubic Hermite interpolation.
///
/// # Arguments
///
/// * `distance` - Distance from layer center (positive).
/// * `max_distance` - Maximum distance (at which falloff reaches 0).
///
/// # Returns
///
/// Falloff factor in range [0, 1].
#[inline]
pub fn smooth_step_falloff(distance: f32, max_distance: f32) -> f32 {
    if distance <= 0.0 {
        return 1.0;
    }
    if max_distance <= 0.0 || distance >= max_distance {
        return 0.0;
    }
    let t = distance / max_distance;
    // smoothstep(1, 0, t) = 1 - smoothstep(0, 1, t)
    let s = t * t * (3.0 - 2.0 * t);
    1.0 - s
}

// ---------------------------------------------------------------------------
// Color Blending Functions
// ---------------------------------------------------------------------------

/// Blend two colors linearly.
///
/// # Arguments
///
/// * `a` - First color (RGB).
/// * `b` - Second color (RGB).
/// * `t` - Blend factor (0.0 = a, 1.0 = b).
///
/// # Returns
///
/// Blended color (RGB).
#[inline]
pub fn blend_colors(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    let t = t.clamp(0.0, 1.0);
    let inv_t = 1.0 - t;
    [
        a[0] * inv_t + b[0] * t,
        a[1] * inv_t + b[1] * t,
        a[2] * inv_t + b[2] * t,
    ]
}

/// Add two colors (additive blending).
///
/// # Arguments
///
/// * `a` - First color (RGB).
/// * `b` - Second color (RGB).
///
/// # Returns
///
/// Sum of colors (not clamped).
#[inline]
pub fn add_colors(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Scale a color by a factor.
///
/// # Arguments
///
/// * `color` - Color (RGB).
/// * `scale` - Scale factor.
///
/// # Returns
///
/// Scaled color.
#[inline]
pub fn scale_color(color: [f32; 3], scale: f32) -> [f32; 3] {
    [color[0] * scale, color[1] * scale, color[2] * scale]
}

/// Take the maximum of two colors component-wise.
///
/// # Arguments
///
/// * `a` - First color (RGB).
/// * `b` - Second color (RGB).
///
/// # Returns
///
/// Maximum color.
#[inline]
pub fn max_colors(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0].max(b[0]), a[1].max(b[1]), a[2].max(b[2])]
}

// ---------------------------------------------------------------------------
// LayeredFogSystem — Runtime fog evaluation
// ---------------------------------------------------------------------------

/// Runtime system for evaluating layered fog density and color.
///
/// This system provides CPU-side fog evaluation for debugging and
/// integration testing. GPU shaders typically implement the same
/// logic for real-time rendering.
#[derive(Debug, Clone)]
pub struct LayeredFogSystem {
    /// Current fog configuration.
    pub config: LayeredFogConfig,
}

impl LayeredFogSystem {
    /// Create a new layered fog system.
    pub fn new(config: LayeredFogConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration.
    pub fn default_system() -> Self {
        Self::new(LayeredFogConfig::default())
    }

    /// Calculate density contribution from a single layer.
    ///
    /// # Arguments
    ///
    /// * `layer` - Layer configuration.
    /// * `height` - World height to sample.
    ///
    /// # Returns
    ///
    /// Density at the given height (0.0 to layer.density).
    #[inline]
    pub fn layer_density(&self, layer: &FogLayerConfig, height: f32) -> f32 {
        if !layer.is_enabled() || layer.density <= 0.0 || layer.thickness <= 0.0 {
            return 0.0;
        }

        // Calculate distance from layer center
        let center = layer.center_height();
        let half_thickness = layer.thickness * 0.5;
        let distance_from_center = (height - center).abs();

        // Outside layer bounds
        if distance_from_center > half_thickness {
            // Apply falloff outside layer
            let distance_outside = distance_from_center - half_thickness;
            let falloff = self.calculate_falloff(distance_outside, layer.falloff_rate, half_thickness);
            return layer.density * falloff;
        }

        // Inside layer - calculate falloff from edge
        let distance_from_edge = half_thickness - distance_from_center;
        let edge_ratio = distance_from_edge / half_thickness;

        // Full density at center, falling off toward edges
        layer.density * edge_ratio
    }

    /// Calculate falloff factor based on current falloff mode.
    #[inline]
    fn calculate_falloff(&self, distance: f32, rate: f32, max_distance: f32) -> f32 {
        match self.config.get_falloff_mode() {
            FalloffMode::Linear => linear_falloff(distance, max_distance),
            FalloffMode::Exponential => exponential_falloff(distance, rate),
            FalloffMode::ExponentialSquared => exponential_squared_falloff(distance, rate),
            FalloffMode::SmoothStep => smooth_step_falloff(distance, max_distance),
        }
    }

    /// Sample total fog density at a world height.
    ///
    /// Combines all enabled layers according to the blend mode.
    ///
    /// # Arguments
    ///
    /// * `world_height` - World-space Y coordinate.
    ///
    /// # Returns
    ///
    /// Total fog density at this height.
    pub fn sample_density(&self, world_height: f32) -> f32 {
        let ground_d = self.layer_density(&self.config.ground_layer, world_height);
        let mid_d = self.layer_density(&self.config.mid_layer, world_height);
        let high_d = self.layer_density(&self.config.high_layer, world_height);

        match self.config.get_blend_mode() {
            BlendMode::Additive => (ground_d + mid_d + high_d).min(1.0),
            BlendMode::Max => ground_d.max(mid_d).max(high_d),
            BlendMode::Lerp => {
                // Lerp mode: use height to blend between layers
                self.lerp_density(world_height, ground_d, mid_d, high_d)
            }
        }
    }

    /// Calculate lerp-blended density based on height.
    fn lerp_density(&self, height: f32, ground_d: f32, mid_d: f32, high_d: f32) -> f32 {
        let ground_top = self.config.ground_layer.top_height();
        let mid_base = self.config.mid_layer.base_height;
        let mid_top = self.config.mid_layer.top_height();
        let high_base = self.config.high_layer.base_height;

        if height < ground_top {
            // In ground layer
            ground_d
        } else if height < mid_base {
            // Between ground and mid - lerp
            let t = (height - ground_top) / (mid_base - ground_top).max(0.001);
            ground_d * (1.0 - t) + mid_d * t
        } else if height < mid_top {
            // In mid layer
            mid_d
        } else if height < high_base {
            // Between mid and high - lerp
            let t = (height - mid_top) / (high_base - mid_top).max(0.001);
            mid_d * (1.0 - t) + high_d * t
        } else {
            // In high layer
            high_d
        }
    }

    /// Sample fog color at a world height.
    ///
    /// Blends colors from enabled layers based on their density contribution.
    ///
    /// # Arguments
    ///
    /// * `world_height` - World-space Y coordinate.
    ///
    /// # Returns
    ///
    /// Fog color (RGB) at this height.
    pub fn sample_color(&self, world_height: f32) -> [f32; 3] {
        let ground_d = self.layer_density(&self.config.ground_layer, world_height);
        let mid_d = self.layer_density(&self.config.mid_layer, world_height);
        let high_d = self.layer_density(&self.config.high_layer, world_height);

        let total_d = ground_d + mid_d + high_d;
        if total_d <= 0.0 {
            return [0.0, 0.0, 0.0];
        }

        // Density-weighted color blend
        let ground_c = scale_color(self.config.ground_layer.color, ground_d);
        let mid_c = scale_color(self.config.mid_layer.color, mid_d);
        let high_c = scale_color(self.config.high_layer.color, high_d);

        let combined = add_colors(add_colors(ground_c, mid_c), high_c);

        // Normalize by total density
        scale_color(combined, 1.0 / total_d)
    }

    /// Compute extinction coefficient from density.
    ///
    /// Extinction controls how much light is absorbed/scattered per unit distance.
    ///
    /// # Arguments
    ///
    /// * `density` - Fog density (0.0 to 1.0).
    ///
    /// # Returns
    ///
    /// Extinction coefficient (RGB).
    #[inline]
    pub fn compute_extinction(&self, density: f32) -> [f32; 3] {
        // Base extinction scaled by density
        let extinction = density * 0.1; // Base extinction coefficient
        [extinction, extinction, extinction]
    }

    /// Compute transmittance over a distance through fog.
    ///
    /// Uses Beer-Lambert law: T = exp(-extinction * distance)
    ///
    /// # Arguments
    ///
    /// * `density` - Fog density.
    /// * `distance` - Ray travel distance.
    ///
    /// # Returns
    ///
    /// Transmittance (0.0 = fully absorbed, 1.0 = fully transmitted).
    #[inline]
    pub fn compute_transmittance(&self, density: f32, distance: f32) -> f32 {
        let extinction = density * 0.1;
        (-extinction * distance).exp()
    }

    /// Get all layer densities at a height.
    ///
    /// # Returns
    ///
    /// (ground_density, mid_density, high_density)
    pub fn layer_densities_at(&self, world_height: f32) -> (f32, f32, f32) {
        (
            self.layer_density(&self.config.ground_layer, world_height),
            self.layer_density(&self.config.mid_layer, world_height),
            self.layer_density(&self.config.high_layer, world_height),
        )
    }
}

impl Default for LayeredFogSystem {
    fn default() -> Self {
        Self::default_system()
    }
}

// ---------------------------------------------------------------------------
// Quality Tier Helpers
// ---------------------------------------------------------------------------

/// Quality tier for fog rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum FogQualityTier {
    /// Mobile quality: ground layer only.
    Mobile,
    /// Low quality: ground layer with reduced density.
    Low,
    /// Medium quality: ground + mid layers.
    Medium,
    /// High quality: all three layers.
    #[default]
    High,
    /// Ultra quality: all three layers with increased density.
    Ultra,
}

impl FogQualityTier {
    /// Create a layered fog config for this quality tier.
    pub fn create_config(self) -> LayeredFogConfig {
        match self {
            FogQualityTier::Mobile => LayeredFogConfig::mobile_quality(),
            FogQualityTier::Low => {
                let mut config = LayeredFogConfig::mobile_quality();
                config.ground_layer.density *= 0.5;
                config
            }
            FogQualityTier::Medium => LayeredFogConfig::medium_quality(),
            FogQualityTier::High => LayeredFogConfig::default_all_layers(),
            FogQualityTier::Ultra => {
                let mut config = LayeredFogConfig::default_all_layers();
                config.ground_layer.density = (config.ground_layer.density * 1.5).min(1.0);
                config.mid_layer.density = (config.mid_layer.density * 1.5).min(1.0);
                config.high_layer.density = (config.high_layer.density * 1.5).min(1.0);
                config
            }
        }
    }

    /// Get the name string for this quality tier.
    #[inline]
    pub fn name(self) -> &'static str {
        match self {
            FogQualityTier::Mobile => "mobile",
            FogQualityTier::Low => "low",
            FogQualityTier::Medium => "medium",
            FogQualityTier::High => "high",
            FogQualityTier::Ultra => "ultra",
        }
    }

    /// Get the number of fog layers enabled for this tier.
    #[inline]
    pub fn layer_count(self) -> usize {
        match self {
            FogQualityTier::Mobile | FogQualityTier::Low => 1,
            FogQualityTier::Medium => 2,
            FogQualityTier::High | FogQualityTier::Ultra => 3,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Falloff Function Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_exponential_falloff_at_zero_distance() {
        assert!((exponential_falloff(0.0, 0.1) - 1.0).abs() < 1e-6);
        assert!((exponential_falloff(0.0, 1.0) - 1.0).abs() < 1e-6);
        assert!((exponential_falloff(-1.0, 1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_exponential_falloff_decreases_with_distance() {
        let f1 = exponential_falloff(1.0, 0.5);
        let f2 = exponential_falloff(2.0, 0.5);
        let f3 = exponential_falloff(3.0, 0.5);
        assert!(f1 > f2);
        assert!(f2 > f3);
        assert!(f1 > 0.0 && f1 < 1.0);
    }

    #[test]
    fn test_exponential_falloff_rate_effect() {
        // Higher rate = faster falloff
        let slow = exponential_falloff(5.0, 0.1);
        let fast = exponential_falloff(5.0, 1.0);
        assert!(slow > fast);
    }

    #[test]
    fn test_exponential_squared_falloff_sharper() {
        // Squared falloff is sharper at longer distances where (rate * distance) > 1
        // At distance=3.0, rate=0.5: rd=1.5
        // exp(-1.5) = 0.223
        // exp(-2.25) = 0.105
        let exp = exponential_falloff(3.0, 0.5);
        let exp2 = exponential_squared_falloff(3.0, 0.5);
        assert!(exp2 < exp, "exp2={} should be < exp={}", exp2, exp);

        // But at short distances where (rate * distance) < 1, squared is gentler
        // At distance=1.0, rate=0.5: rd=0.5
        // exp(-0.5) = 0.606
        // exp(-0.25) = 0.779
        let exp_short = exponential_falloff(1.0, 0.5);
        let exp2_short = exponential_squared_falloff(1.0, 0.5);
        assert!(exp2_short > exp_short, "exp2_short={} should be > exp_short={}", exp2_short, exp_short);
    }

    #[test]
    fn test_linear_falloff_exact_values() {
        assert!((linear_falloff(0.0, 10.0) - 1.0).abs() < 1e-6);
        assert!((linear_falloff(5.0, 10.0) - 0.5).abs() < 1e-6);
        assert!((linear_falloff(10.0, 10.0) - 0.0).abs() < 1e-6);
        assert!((linear_falloff(15.0, 10.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_linear_falloff_negative_distance() {
        assert!((linear_falloff(-5.0, 10.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_linear_falloff_zero_max_distance() {
        assert!((linear_falloff(1.0, 0.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_smooth_step_falloff_at_boundaries() {
        assert!((smooth_step_falloff(0.0, 10.0) - 1.0).abs() < 1e-6);
        assert!((smooth_step_falloff(10.0, 10.0) - 0.0).abs() < 1e-6);
        // Mid-point should be 0.5 for smoothstep
        let mid = smooth_step_falloff(5.0, 10.0);
        assert!((mid - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_smooth_step_falloff_smooth_curve() {
        let f1 = smooth_step_falloff(2.5, 10.0);
        let f2 = smooth_step_falloff(5.0, 10.0);
        let f3 = smooth_step_falloff(7.5, 10.0);
        assert!(f1 > f2);
        assert!(f2 > f3);
    }

    // -----------------------------------------------------------------------
    // Color Blending Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_blend_colors_at_boundaries() {
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 0.0, 1.0];

        let at_a = blend_colors(a, b, 0.0);
        let at_b = blend_colors(a, b, 1.0);

        assert!((at_a[0] - 1.0).abs() < 1e-6);
        assert!((at_a[2] - 0.0).abs() < 1e-6);
        assert!((at_b[0] - 0.0).abs() < 1e-6);
        assert!((at_b[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_blend_colors_midpoint() {
        let a = [1.0, 0.0, 0.5];
        let b = [0.0, 1.0, 0.5];

        let mid = blend_colors(a, b, 0.5);

        assert!((mid[0] - 0.5).abs() < 1e-6);
        assert!((mid[1] - 0.5).abs() < 1e-6);
        assert!((mid[2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_blend_colors_clamps_t() {
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];

        let below = blend_colors(a, b, -0.5);
        let above = blend_colors(a, b, 1.5);

        assert!((below[0] - 1.0).abs() < 1e-6);
        assert!((above[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_add_colors() {
        let a = [0.3, 0.5, 0.2];
        let b = [0.2, 0.3, 0.4];

        let sum = add_colors(a, b);

        assert!((sum[0] - 0.5).abs() < 1e-6);
        assert!((sum[1] - 0.8).abs() < 1e-6);
        assert!((sum[2] - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_scale_color() {
        let c = [0.5, 0.25, 1.0];
        let scaled = scale_color(c, 2.0);

        assert!((scaled[0] - 1.0).abs() < 1e-6);
        assert!((scaled[1] - 0.5).abs() < 1e-6);
        assert!((scaled[2] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_max_colors() {
        let a = [0.3, 0.8, 0.1];
        let b = [0.5, 0.2, 0.9];

        let max = max_colors(a, b);

        assert!((max[0] - 0.5).abs() < 1e-6);
        assert!((max[1] - 0.8).abs() < 1e-6);
        assert!((max[2] - 0.9).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // FogLayerConfig Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_fog_layer_config_new() {
        let layer = FogLayerConfig::new(10.0, 50.0, 0.5, 0.1, [1.0, 1.0, 1.0]);

        assert!((layer.base_height - 10.0).abs() < 1e-6);
        assert!((layer.thickness - 50.0).abs() < 1e-6);
        assert!((layer.density - 0.5).abs() < 1e-6);
        assert!(layer.is_enabled());
    }

    #[test]
    fn test_fog_layer_config_density_clamped() {
        let layer = FogLayerConfig::new(0.0, 10.0, 1.5, 0.1, [1.0, 1.0, 1.0]);
        assert!((layer.density - 1.0).abs() < 1e-6);

        let layer2 = FogLayerConfig::new(0.0, 10.0, -0.5, 0.1, [1.0, 1.0, 1.0]);
        assert!((layer2.density - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_fog_layer_config_disabled() {
        let layer = FogLayerConfig::disabled();
        assert!(!layer.is_enabled());
        assert!((layer.density - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_fog_layer_config_presets() {
        let ground = FogLayerConfig::ground_fog();
        let mid = FogLayerConfig::mid_haze();
        let high = FogLayerConfig::high_haze();

        assert!(ground.is_enabled());
        assert!(mid.is_enabled());
        assert!(high.is_enabled());

        // Ground fog should be below mid, mid below high
        assert!(ground.top_height() <= mid.base_height || ground.top_height() <= mid.top_height());
        assert!(mid.top_height() <= high.base_height || mid.top_height() <= high.top_height());
    }

    #[test]
    fn test_fog_layer_config_height_methods() {
        let layer = FogLayerConfig::new(100.0, 50.0, 0.5, 0.1, [1.0, 1.0, 1.0]);

        assert!((layer.top_height() - 150.0).abs() < 1e-6);
        assert!((layer.center_height() - 125.0).abs() < 1e-6);
    }

    #[test]
    fn test_fog_layer_config_contains_height() {
        let layer = FogLayerConfig::new(100.0, 50.0, 0.5, 0.1, [1.0, 1.0, 1.0]);

        assert!(!layer.contains_height(99.0));
        assert!(layer.contains_height(100.0));
        assert!(layer.contains_height(125.0));
        assert!(layer.contains_height(150.0));
        assert!(!layer.contains_height(151.0));
    }

    #[test]
    fn test_fog_layer_config_enabled_toggle() {
        let mut layer = FogLayerConfig::ground_fog();
        assert!(layer.is_enabled());

        layer.set_enabled(false);
        assert!(!layer.is_enabled());

        layer.set_enabled(true);
        assert!(layer.is_enabled());
    }

    #[test]
    fn test_fog_layer_config_validation() {
        let valid = FogLayerConfig::new(0.0, 50.0, 0.5, 0.1, [1.0, 1.0, 1.0]);
        assert!(valid.is_valid());

        // Negative thickness is clamped to 0, which is valid
        let zero_thickness = FogLayerConfig::new(0.0, 0.0, 0.5, 0.1, [1.0, 1.0, 1.0]);
        assert!(zero_thickness.is_valid());
    }

    // -----------------------------------------------------------------------
    // LayeredFogConfig Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_layered_fog_config_default() {
        let config = LayeredFogConfig::default();

        assert!(config.ground_layer.is_enabled());
        assert!(config.mid_layer.is_enabled());
        assert!(config.high_layer.is_enabled());
        assert_eq!(config.enabled_layer_count(), 3);
    }

    #[test]
    fn test_layered_fog_config_mobile_quality() {
        let config = LayeredFogConfig::mobile_quality();

        assert!(config.ground_layer.is_enabled());
        assert!(!config.mid_layer.is_enabled());
        assert!(!config.high_layer.is_enabled());
        assert_eq!(config.enabled_layer_count(), 1);
    }

    #[test]
    fn test_layered_fog_config_medium_quality() {
        let config = LayeredFogConfig::medium_quality();

        assert!(config.ground_layer.is_enabled());
        assert!(config.mid_layer.is_enabled());
        assert!(!config.high_layer.is_enabled());
        assert_eq!(config.enabled_layer_count(), 2);
    }

    #[test]
    fn test_layered_fog_config_blend_mode() {
        let config = LayeredFogConfig::default()
            .with_blend_mode(BlendMode::Max);

        assert_eq!(config.get_blend_mode(), BlendMode::Max);
    }

    #[test]
    fn test_layered_fog_config_falloff_mode() {
        let config = LayeredFogConfig::default()
            .with_falloff_mode(FalloffMode::ExponentialSquared);

        assert_eq!(config.get_falloff_mode(), FalloffMode::ExponentialSquared);
    }

    #[test]
    fn test_layered_fog_config_validation() {
        let config = LayeredFogConfig::default();
        assert!(config.is_valid());
    }

    // -----------------------------------------------------------------------
    // LayeredFogSystem Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_layered_fog_system_creation() {
        let system = LayeredFogSystem::default();
        assert!(system.config.is_valid());
    }

    #[test]
    fn test_layer_density_disabled_layer() {
        let system = LayeredFogSystem::default();
        let mut disabled = FogLayerConfig::disabled();

        let density = system.layer_density(&disabled, 50.0);
        assert!((density - 0.0).abs() < 1e-6);

        disabled.density = 0.5;
        disabled.enabled = 0;
        let density2 = system.layer_density(&disabled, 50.0);
        assert!((density2 - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_layer_density_at_center() {
        let system = LayeredFogSystem::default();
        let layer = FogLayerConfig::new(0.0, 100.0, 0.5, 0.1, [1.0, 1.0, 1.0]);

        // At center (50.0), should get maximum density effect
        let center_density = system.layer_density(&layer, 50.0);
        assert!(center_density > 0.0);
    }

    #[test]
    fn test_layer_density_decreases_from_center() {
        let system = LayeredFogSystem::default();
        let layer = FogLayerConfig::new(0.0, 100.0, 0.5, 0.1, [1.0, 1.0, 1.0]);

        let d_center = system.layer_density(&layer, 50.0);
        let d_edge = system.layer_density(&layer, 10.0);
        let d_outside = system.layer_density(&layer, 150.0);

        assert!(d_center >= d_edge);
        assert!(d_edge > d_outside);
    }

    #[test]
    fn test_layer_density_outside_bounds() {
        let system = LayeredFogSystem::default();
        let layer = FogLayerConfig::new(100.0, 50.0, 0.5, 0.5, [1.0, 1.0, 1.0]);

        // Well outside layer bounds
        let d_below = system.layer_density(&layer, 0.0);
        let d_above = system.layer_density(&layer, 500.0);

        assert!(d_below < layer.density);
        assert!(d_above < layer.density);
    }

    #[test]
    fn test_sample_density_additive() {
        let mut config = LayeredFogConfig::default();
        config.blend_mode = BlendMode::Additive as u32;

        // Create overlapping layers
        config.ground_layer = FogLayerConfig::new(0.0, 100.0, 0.3, 0.01, [1.0, 1.0, 1.0]);
        config.mid_layer = FogLayerConfig::new(50.0, 100.0, 0.3, 0.01, [1.0, 1.0, 1.0]);
        config.high_layer.set_enabled(false);

        let system = LayeredFogSystem::new(config);

        // In overlap region, density should be sum of contributions
        let d_overlap = system.sample_density(75.0);
        let d_ground_only = system.sample_density(10.0);

        assert!(d_overlap > d_ground_only || d_overlap == 1.0); // Capped at 1.0
    }

    #[test]
    fn test_sample_density_max_blend() {
        let mut config = LayeredFogConfig::default();
        config.blend_mode = BlendMode::Max as u32;

        config.ground_layer = FogLayerConfig::new(0.0, 50.0, 0.3, 0.01, [1.0, 1.0, 1.0]);
        config.mid_layer = FogLayerConfig::new(25.0, 50.0, 0.5, 0.01, [1.0, 1.0, 1.0]);
        config.high_layer.set_enabled(false);

        let system = LayeredFogSystem::new(config);

        // In overlap, max mode should not exceed single layer max
        let d = system.sample_density(40.0);
        assert!(d <= 0.5);
    }

    #[test]
    fn test_sample_density_lerp_blend() {
        let mut config = LayeredFogConfig::default();
        config.blend_mode = BlendMode::Lerp as u32;

        let system = LayeredFogSystem::new(config);

        // Just verify it doesn't crash and returns valid values
        let d_low = system.sample_density(25.0);
        let d_mid = system.sample_density(300.0);
        let d_high = system.sample_density(3000.0);

        assert!(d_low >= 0.0 && d_low <= 1.0);
        assert!(d_mid >= 0.0 && d_mid <= 1.0);
        assert!(d_high >= 0.0 && d_high <= 1.0);
    }

    #[test]
    fn test_sample_color_density_weighted() {
        let mut config = LayeredFogConfig::default();
        config.ground_layer = FogLayerConfig::new(0.0, 50.0, 0.5, 0.01, [1.0, 0.0, 0.0]); // Red
        config.mid_layer = FogLayerConfig::new(50.0, 50.0, 0.5, 0.01, [0.0, 0.0, 1.0]); // Blue
        config.high_layer.set_enabled(false);

        let system = LayeredFogSystem::new(config);

        // Low height should be more red
        let color_low = system.sample_color(25.0);
        assert!(color_low[0] > color_low[2]); // More red than blue
    }

    #[test]
    fn test_sample_color_returns_zero_when_no_density() {
        let config = LayeredFogConfig::mobile_quality();
        let mut system = LayeredFogSystem::new(config);
        system.config.ground_layer.set_enabled(false);

        let color = system.sample_color(50.0);
        assert!((color[0] - 0.0).abs() < 1e-6);
        assert!((color[1] - 0.0).abs() < 1e-6);
        assert!((color[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_extinction() {
        let system = LayeredFogSystem::default();

        let ext = system.compute_extinction(0.5);
        assert!(ext[0] > 0.0);
        assert!((ext[0] - ext[1]).abs() < 1e-6);
        assert!((ext[1] - ext[2]).abs() < 1e-6);
    }

    #[test]
    fn test_compute_transmittance() {
        let system = LayeredFogSystem::default();

        // No density = full transmittance
        let t_clear = system.compute_transmittance(0.0, 100.0);
        assert!((t_clear - 1.0).abs() < 1e-6);

        // Higher density = lower transmittance
        let t_light = system.compute_transmittance(0.1, 100.0);
        let t_heavy = system.compute_transmittance(0.5, 100.0);
        assert!(t_light > t_heavy);

        // Longer distance = lower transmittance
        let t_short = system.compute_transmittance(0.3, 10.0);
        let t_long = system.compute_transmittance(0.3, 100.0);
        assert!(t_short > t_long);
    }

    #[test]
    fn test_layer_densities_at() {
        let system = LayeredFogSystem::default();

        let (g, m, h) = system.layer_densities_at(25.0);

        // At ground level, ground fog should dominate
        assert!(g > 0.0);
        // Mid and high may have some contribution depending on falloff
    }

    // -----------------------------------------------------------------------
    // FalloffMode and BlendMode Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_falloff_mode_from_u32() {
        assert_eq!(FalloffMode::from_u32(0), FalloffMode::Linear);
        assert_eq!(FalloffMode::from_u32(1), FalloffMode::Exponential);
        assert_eq!(FalloffMode::from_u32(2), FalloffMode::ExponentialSquared);
        assert_eq!(FalloffMode::from_u32(3), FalloffMode::SmoothStep);
        assert_eq!(FalloffMode::from_u32(99), FalloffMode::Exponential); // Default
    }

    #[test]
    fn test_blend_mode_from_u32() {
        assert_eq!(BlendMode::from_u32(0), BlendMode::Additive);
        assert_eq!(BlendMode::from_u32(1), BlendMode::Max);
        assert_eq!(BlendMode::from_u32(2), BlendMode::Lerp);
        assert_eq!(BlendMode::from_u32(99), BlendMode::Additive); // Default
    }

    #[test]
    fn test_falloff_mode_names() {
        assert_eq!(FalloffMode::Linear.name(), "linear");
        assert_eq!(FalloffMode::Exponential.name(), "exponential");
        assert_eq!(FalloffMode::ExponentialSquared.name(), "exponential_squared");
        assert_eq!(FalloffMode::SmoothStep.name(), "smooth_step");
    }

    #[test]
    fn test_blend_mode_names() {
        assert_eq!(BlendMode::Additive.name(), "additive");
        assert_eq!(BlendMode::Max.name(), "max");
        assert_eq!(BlendMode::Lerp.name(), "lerp");
    }

    // -----------------------------------------------------------------------
    // FogQualityTier Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_fog_quality_tier_configs() {
        let mobile = FogQualityTier::Mobile.create_config();
        let low = FogQualityTier::Low.create_config();
        let medium = FogQualityTier::Medium.create_config();
        let high = FogQualityTier::High.create_config();
        let ultra = FogQualityTier::Ultra.create_config();

        assert_eq!(mobile.enabled_layer_count(), 1);
        assert_eq!(low.enabled_layer_count(), 1);
        assert_eq!(medium.enabled_layer_count(), 2);
        assert_eq!(high.enabled_layer_count(), 3);
        assert_eq!(ultra.enabled_layer_count(), 3);
    }

    #[test]
    fn test_fog_quality_tier_layer_counts() {
        assert_eq!(FogQualityTier::Mobile.layer_count(), 1);
        assert_eq!(FogQualityTier::Low.layer_count(), 1);
        assert_eq!(FogQualityTier::Medium.layer_count(), 2);
        assert_eq!(FogQualityTier::High.layer_count(), 3);
        assert_eq!(FogQualityTier::Ultra.layer_count(), 3);
    }

    #[test]
    fn test_fog_quality_tier_names() {
        assert_eq!(FogQualityTier::Mobile.name(), "mobile");
        assert_eq!(FogQualityTier::Low.name(), "low");
        assert_eq!(FogQualityTier::Medium.name(), "medium");
        assert_eq!(FogQualityTier::High.name(), "high");
        assert_eq!(FogQualityTier::Ultra.name(), "ultra");
    }

    #[test]
    fn test_fog_quality_tier_density_scaling() {
        let low = FogQualityTier::Low.create_config();
        let high = FogQualityTier::High.create_config();
        let ultra = FogQualityTier::Ultra.create_config();

        // Low should have reduced ground density
        assert!(low.ground_layer.density < high.ground_layer.density);

        // Ultra should have increased density
        assert!(ultra.ground_layer.density > high.ground_layer.density);
    }

    // -----------------------------------------------------------------------
    // Bytemuck Pod/Zeroable Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_fog_layer_config_is_pod() {
        let layer = FogLayerConfig::ground_fog();
        let bytes = bytemuck::bytes_of(&layer);
        assert_eq!(bytes.len(), std::mem::size_of::<FogLayerConfig>());
    }

    #[test]
    fn test_layered_fog_config_is_pod() {
        let config = LayeredFogConfig::default();
        let bytes = bytemuck::bytes_of(&config);
        assert_eq!(bytes.len(), std::mem::size_of::<LayeredFogConfig>());
    }

    #[test]
    fn test_fog_layer_config_zeroed() {
        let zeroed: FogLayerConfig = bytemuck::Zeroable::zeroed();
        assert!((zeroed.density - 0.0).abs() < 1e-6);
        assert!((zeroed.base_height - 0.0).abs() < 1e-6);
        assert_eq!(zeroed.enabled, 0);
    }

    // -----------------------------------------------------------------------
    // Edge Case Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_zero_thickness_layer() {
        let system = LayeredFogSystem::default();
        let layer = FogLayerConfig::new(50.0, 0.0, 0.5, 0.1, [1.0, 1.0, 1.0]);

        // Zero thickness should return 0 density
        let d = system.layer_density(&layer, 50.0);
        assert!((d - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_zero_density_layer() {
        let system = LayeredFogSystem::default();
        let layer = FogLayerConfig::new(0.0, 100.0, 0.0, 0.1, [1.0, 1.0, 1.0]);

        let d = system.layer_density(&layer, 50.0);
        assert!((d - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_very_high_falloff_rate() {
        let system = LayeredFogSystem::default();
        let layer = FogLayerConfig::new(0.0, 100.0, 1.0, 100.0, [1.0, 1.0, 1.0]);

        // Very high falloff should make density drop quickly outside
        let d_outside = system.layer_density(&layer, 200.0);
        assert!(d_outside < 0.01);
    }

    #[test]
    fn test_very_low_falloff_rate() {
        let system = LayeredFogSystem::default();
        let layer = FogLayerConfig::new(0.0, 100.0, 1.0, 0.001, [1.0, 1.0, 1.0]);

        // Very low falloff should maintain density further out
        let d_outside = system.layer_density(&layer, 150.0);
        assert!(d_outside > 0.0);
    }

    #[test]
    fn test_negative_base_height() {
        let system = LayeredFogSystem::default();
        let layer = FogLayerConfig::new(-50.0, 100.0, 0.5, 0.1, [1.0, 1.0, 1.0]);

        // Should work with negative heights (underground fog)
        let d_center = system.layer_density(&layer, 0.0);
        assert!(d_center > 0.0);
    }

    #[test]
    fn test_extreme_heights() {
        let system = LayeredFogSystem::default();

        // Very high altitude
        let d_high = system.sample_density(100000.0);
        assert!(d_high >= 0.0 && d_high <= 1.0);

        // Very low altitude (below ground)
        let d_low = system.sample_density(-1000.0);
        assert!(d_low >= 0.0 && d_low <= 1.0);
    }

    #[test]
    fn test_density_never_negative() {
        let system = LayeredFogSystem::default();

        for height in [-100, 0, 50, 100, 500, 1000, 5000, 10000].iter() {
            let d = system.sample_density(*height as f32);
            assert!(d >= 0.0, "Density was negative at height {}", height);
        }
    }

    #[test]
    fn test_density_capped_at_one() {
        let mut config = LayeredFogConfig::default();
        config.ground_layer.density = 0.8;
        config.mid_layer.density = 0.8;
        config.high_layer.density = 0.8;

        let system = LayeredFogSystem::new(config);

        // Even with high densities, should be capped at 1.0 in additive mode
        for height in [0, 25, 50, 100, 200, 500, 1000].iter() {
            let d = system.sample_density(*height as f32);
            assert!(d <= 1.0, "Density exceeded 1.0 at height {}", height);
        }
    }

    // -----------------------------------------------------------------------
    // Integration Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_full_atmosphere_simulation() {
        let config = LayeredFogConfig::default_all_layers();
        let system = LayeredFogSystem::new(config);

        // Simulate sampling at various heights
        let heights = [0.0, 10.0, 25.0, 50.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0];

        let mut prev_ground = 1.0;
        for &h in &heights {
            let (g, m, high) = system.layer_densities_at(h);
            let total = system.sample_density(h);
            let color = system.sample_color(h);

            // All values should be valid
            assert!(g >= 0.0 && g <= 1.0);
            assert!(m >= 0.0 && m <= 1.0);
            assert!(high >= 0.0 && high <= 1.0);
            assert!(total >= 0.0 && total <= 1.0);
            assert!(color.iter().all(|c| *c >= 0.0));

            // Ground fog should generally decrease with height
            if h > 50.0 {
                assert!(g <= prev_ground + 0.01);
            }
            prev_ground = g;
        }
    }

    #[test]
    fn test_quality_tier_progression() {
        let tiers = [
            FogQualityTier::Mobile,
            FogQualityTier::Low,
            FogQualityTier::Medium,
            FogQualityTier::High,
            FogQualityTier::Ultra,
        ];

        for tier in tiers {
            let config = tier.create_config();
            assert!(config.is_valid());
            assert_eq!(config.enabled_layer_count(), tier.layer_count());
        }
    }
}
