//! Terrain Material Blending (T-ENV-1.10)
//!
//! This module provides GPU-compatible terrain material blending with:
//!
//! - **SplatMap**: Up to 8 terrain layers using 2x RGBA textures
//! - **Height-based blending**: Automatic layer selection based on terrain height
//! - **Slope-based blending**: Automatic layer selection based on terrain slope
//! - **Stochastic sampling**: Reduces visible texture tiling artifacts
//! - **Weight normalization**: Ensures layer weights sum to 1.0
//!
//! # SplatMap Architecture
//!
//! The splat map uses two RGBA textures to encode weights for 8 layers:
//! - Texture 0: Layers 0-3 (R=0, G=1, B=2, A=3)
//! - Texture 1: Layers 4-7 (R=4, G=5, B=6, A=7)
//!
//! # Auto-Blending
//!
//! Layers can be automatically blended based on:
//! - **Height**: Each layer defines height_min/height_max range
//! - **Slope**: Each layer defines slope_min/slope_max range (0-90 degrees)
//!
//! Blend weights are computed using smooth falloff curves at range boundaries.
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::terrain::terrain_material::{SplatMap, TerrainLayerDef, TerrainMaterialConfig};
//!
//! let config = TerrainMaterialConfig::new(1024, 8, 1.0);
//! let mut splat = SplatMap::new(config);
//!
//! // Define grass layer for low slopes
//! splat.set_layer(0, TerrainLayerDef {
//!     uv_scale: 0.1,
//!     height_min: 0.0,
//!     height_max: 100.0,
//!     slope_min: 0.0,
//!     slope_max: 30.0,
//!     ..Default::default()
//! });
//!
//! // Sample blended weights at a world position
//! let weights = splat.blend_weights([100.0, 50.0, 200.0], 15.0);
//! ```

use bytemuck::{Pod, Zeroable};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of terrain layers supported.
pub const MAX_TERRAIN_LAYERS: usize = 8;

/// Default splat map resolution.
pub const DEFAULT_SPLAT_RESOLUTION: u32 = 1024;

/// Default UV scale for terrain textures.
pub const DEFAULT_UV_SCALE: f32 = 1.0;

/// Minimum splat map resolution.
pub const MIN_SPLAT_RESOLUTION: u32 = 64;

/// Maximum splat map resolution.
pub const MAX_SPLAT_RESOLUTION: u32 = 8192;

/// Minimum UV scale.
pub const MIN_UV_SCALE: f32 = 0.001;

/// Maximum UV scale.
pub const MAX_UV_SCALE: f32 = 1000.0;

/// Epsilon for weight normalization (prevents division by zero).
pub const WEIGHT_EPSILON: f32 = 1e-6;

/// Default blend falloff for height/slope transitions.
pub const DEFAULT_BLEND_FALLOFF: f32 = 10.0;

/// Golden ratio for stochastic sampling.
pub const GOLDEN_RATIO: f32 = 1.618033988749895;

/// Stochastic noise scale factor.
pub const STOCHASTIC_SCALE: f32 = 0.15;

// ---------------------------------------------------------------------------
// Error Types
// ---------------------------------------------------------------------------

/// Errors that can occur during terrain material operations.
#[derive(Debug, Clone, PartialEq)]
pub enum TerrainMaterialError {
    /// Invalid splat resolution.
    InvalidResolution { value: u32, min: u32, max: u32 },
    /// Invalid UV scale.
    InvalidUvScale { value: f32, min: f32, max: f32 },
    /// Invalid layer index.
    InvalidLayerIndex { index: usize, max: usize },
    /// Invalid height range.
    InvalidHeightRange { min: f32, max: f32 },
    /// Invalid slope range.
    InvalidSlopeRange { min: f32, max: f32 },
    /// Layer count exceeds maximum.
    TooManyLayers { count: usize, max: usize },
    /// Invalid blend parameters.
    InvalidBlendParams { reason: &'static str },
}

impl std::fmt::Display for TerrainMaterialError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidResolution { value, min, max } => {
                write!(
                    f,
                    "Invalid splat resolution: {} (must be in [{}, {}])",
                    value, min, max
                )
            }
            Self::InvalidUvScale { value, min, max } => {
                write!(
                    f,
                    "Invalid UV scale: {} (must be in [{}, {}])",
                    value, min, max
                )
            }
            Self::InvalidLayerIndex { index, max } => {
                write!(f, "Invalid layer index: {} (max: {})", index, max)
            }
            Self::InvalidHeightRange { min, max } => {
                write!(f, "Invalid height range: [{}, {}] (min must be <= max)", min, max)
            }
            Self::InvalidSlopeRange { min, max } => {
                write!(f, "Invalid slope range: [{}, {}] (must be in [0, 90])", min, max)
            }
            Self::TooManyLayers { count, max } => {
                write!(f, "Too many layers: {} (max: {})", count, max)
            }
            Self::InvalidBlendParams { reason } => {
                write!(f, "Invalid blend parameters: {}", reason)
            }
        }
    }
}

impl std::error::Error for TerrainMaterialError {}

// ---------------------------------------------------------------------------
// GPU-side Configuration
// ---------------------------------------------------------------------------

/// Terrain material configuration (GPU-compatible).
///
/// This struct is designed for direct GPU upload with proper alignment.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct TerrainMaterialConfig {
    /// Splat map resolution in texels.
    pub splat_resolution: u32,
    /// Maximum number of active layers.
    pub max_layers: u32,
    /// Global UV scale multiplier.
    pub uv_scale: f32,
    /// Padding for 16-byte alignment.
    pub _padding: u32,
}

impl Default for TerrainMaterialConfig {
    fn default() -> Self {
        Self {
            splat_resolution: DEFAULT_SPLAT_RESOLUTION,
            max_layers: MAX_TERRAIN_LAYERS as u32,
            uv_scale: DEFAULT_UV_SCALE,
            _padding: 0,
        }
    }
}

impl TerrainMaterialConfig {
    /// Create a new terrain material configuration with validation.
    ///
    /// # Arguments
    ///
    /// * `splat_resolution` - Resolution of splat map texture
    /// * `max_layers` - Maximum number of active layers (1-8)
    /// * `uv_scale` - Global UV scale multiplier
    ///
    /// # Panics
    ///
    /// Panics if parameters are out of valid ranges.
    pub fn new(splat_resolution: u32, max_layers: u32, uv_scale: f32) -> Self {
        assert!(
            splat_resolution >= MIN_SPLAT_RESOLUTION && splat_resolution <= MAX_SPLAT_RESOLUTION,
            "splat_resolution must be in [{}, {}], got {}",
            MIN_SPLAT_RESOLUTION,
            MAX_SPLAT_RESOLUTION,
            splat_resolution
        );
        assert!(
            max_layers >= 1 && max_layers <= MAX_TERRAIN_LAYERS as u32,
            "max_layers must be in [1, {}], got {}",
            MAX_TERRAIN_LAYERS,
            max_layers
        );
        assert!(
            uv_scale >= MIN_UV_SCALE && uv_scale <= MAX_UV_SCALE,
            "uv_scale must be in [{}, {}], got {}",
            MIN_UV_SCALE,
            MAX_UV_SCALE,
            uv_scale
        );

        Self {
            splat_resolution,
            max_layers,
            uv_scale,
            _padding: 0,
        }
    }

    /// Try to create a new configuration with validation.
    pub fn try_new(
        splat_resolution: u32,
        max_layers: u32,
        uv_scale: f32,
    ) -> Result<Self, TerrainMaterialError> {
        if splat_resolution < MIN_SPLAT_RESOLUTION || splat_resolution > MAX_SPLAT_RESOLUTION {
            return Err(TerrainMaterialError::InvalidResolution {
                value: splat_resolution,
                min: MIN_SPLAT_RESOLUTION,
                max: MAX_SPLAT_RESOLUTION,
            });
        }
        if max_layers < 1 || max_layers > MAX_TERRAIN_LAYERS as u32 {
            return Err(TerrainMaterialError::TooManyLayers {
                count: max_layers as usize,
                max: MAX_TERRAIN_LAYERS,
            });
        }
        if uv_scale < MIN_UV_SCALE || uv_scale > MAX_UV_SCALE {
            return Err(TerrainMaterialError::InvalidUvScale {
                value: uv_scale,
                min: MIN_UV_SCALE,
                max: MAX_UV_SCALE,
            });
        }

        Ok(Self {
            splat_resolution,
            max_layers,
            uv_scale,
            _padding: 0,
        })
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), TerrainMaterialError> {
        Self::try_new(self.splat_resolution, self.max_layers, self.uv_scale)?;
        Ok(())
    }

    /// Calculate the number of texels in the splat map.
    #[inline]
    pub fn total_texels(&self) -> usize {
        (self.splat_resolution as usize) * (self.splat_resolution as usize)
    }
}

// ---------------------------------------------------------------------------
// Terrain Layer Definition
// ---------------------------------------------------------------------------

/// Terrain layer definition (GPU-compatible).
///
/// Defines properties for a single terrain texture layer including
/// UV scaling, height range, and slope range for auto-blending.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct TerrainLayerDef {
    /// UV scale for this layer's texture.
    pub uv_scale: f32,
    /// Minimum height for this layer (auto-blend).
    pub height_min: f32,
    /// Maximum height for this layer (auto-blend).
    pub height_max: f32,
    /// Minimum slope angle (degrees) for this layer.
    pub slope_min: f32,
    /// Maximum slope angle (degrees) for this layer.
    pub slope_max: f32,
    /// Blend falloff distance for height transitions.
    pub height_falloff: f32,
    /// Blend falloff distance for slope transitions.
    pub slope_falloff: f32,
    /// Layer priority (higher = painted on top).
    pub priority: f32,
    /// Texture array index for albedo.
    pub albedo_index: u32,
    /// Texture array index for normal map.
    pub normal_index: u32,
    /// Texture array index for roughness/metallic/AO.
    pub orm_index: u32,
    /// Stochastic tiling reduction factor (0 = off, 1 = max).
    pub stochastic_factor: f32,
    /// Triplanar projection scale (0 = no triplanar).
    pub triplanar_scale: f32,
    /// Height map influence for micro-detail.
    pub height_influence: f32,
    /// Padding for 16-byte alignment.
    pub _padding: [f32; 2],
}

impl Default for TerrainLayerDef {
    fn default() -> Self {
        Self {
            uv_scale: 1.0,
            height_min: f32::NEG_INFINITY,
            height_max: f32::INFINITY,
            slope_min: 0.0,
            slope_max: 90.0,
            height_falloff: DEFAULT_BLEND_FALLOFF,
            slope_falloff: 5.0,
            priority: 0.0,
            albedo_index: 0,
            normal_index: 0,
            orm_index: 0,
            stochastic_factor: 0.0,
            triplanar_scale: 0.0,
            height_influence: 0.0,
            _padding: [0.0; 2],
        }
    }
}

impl TerrainLayerDef {
    /// Create a new terrain layer with basic parameters.
    pub fn new(uv_scale: f32, height_min: f32, height_max: f32) -> Self {
        Self {
            uv_scale,
            height_min,
            height_max,
            ..Default::default()
        }
    }

    /// Create a layer optimized for flat areas (grass, sand).
    pub fn flat_layer(uv_scale: f32, height_min: f32, height_max: f32) -> Self {
        Self {
            uv_scale,
            height_min,
            height_max,
            slope_min: 0.0,
            slope_max: 25.0,
            slope_falloff: 5.0,
            ..Default::default()
        }
    }

    /// Create a layer optimized for slopes (rock, cliff).
    pub fn slope_layer(uv_scale: f32, min_slope: f32) -> Self {
        Self {
            uv_scale,
            height_min: f32::NEG_INFINITY,
            height_max: f32::INFINITY,
            slope_min: min_slope,
            slope_max: 90.0,
            slope_falloff: 5.0,
            triplanar_scale: 1.0, // Use triplanar for steep slopes
            ..Default::default()
        }
    }

    /// Set the texture indices for this layer.
    pub fn with_textures(mut self, albedo: u32, normal: u32, orm: u32) -> Self {
        self.albedo_index = albedo;
        self.normal_index = normal;
        self.orm_index = orm;
        self
    }

    /// Enable stochastic sampling for tiling reduction.
    pub fn with_stochastic(mut self, factor: f32) -> Self {
        self.stochastic_factor = factor.clamp(0.0, 1.0);
        self
    }

    /// Set layer priority for ordering.
    pub fn with_priority(mut self, priority: f32) -> Self {
        self.priority = priority;
        self
    }

    /// Validate the layer definition.
    pub fn validate(&self) -> Result<(), TerrainMaterialError> {
        if self.height_min > self.height_max {
            return Err(TerrainMaterialError::InvalidHeightRange {
                min: self.height_min,
                max: self.height_max,
            });
        }
        if self.slope_min < 0.0 || self.slope_max > 90.0 || self.slope_min > self.slope_max {
            return Err(TerrainMaterialError::InvalidSlopeRange {
                min: self.slope_min,
                max: self.slope_max,
            });
        }
        if self.uv_scale <= 0.0 {
            return Err(TerrainMaterialError::InvalidUvScale {
                value: self.uv_scale,
                min: MIN_UV_SCALE,
                max: MAX_UV_SCALE,
            });
        }
        Ok(())
    }

    /// Calculate the height-based weight for a given world height.
    ///
    /// Returns 1.0 when fully inside range, smoothly falls off at boundaries.
    #[inline]
    pub fn height_weight(&self, height: f32) -> f32 {
        if height < self.height_min - self.height_falloff {
            return 0.0;
        }
        if height > self.height_max + self.height_falloff {
            return 0.0;
        }

        let lower_blend = if height < self.height_min {
            smoothstep(
                self.height_min - self.height_falloff,
                self.height_min,
                height,
            )
        } else {
            1.0
        };

        let upper_blend = if height > self.height_max {
            1.0 - smoothstep(self.height_max, self.height_max + self.height_falloff, height)
        } else {
            1.0
        };

        lower_blend * upper_blend
    }

    /// Calculate the slope-based weight for a given slope angle (degrees).
    ///
    /// Returns 1.0 when fully inside range, smoothly falls off at boundaries.
    #[inline]
    pub fn slope_weight(&self, slope_degrees: f32) -> f32 {
        let slope = slope_degrees.clamp(0.0, 90.0);

        if slope < self.slope_min - self.slope_falloff {
            return 0.0;
        }
        if slope > self.slope_max + self.slope_falloff {
            return 0.0;
        }

        let lower_blend = if slope < self.slope_min {
            smoothstep(self.slope_min - self.slope_falloff, self.slope_min, slope)
        } else {
            1.0
        };

        let upper_blend = if slope > self.slope_max {
            1.0 - smoothstep(self.slope_max, self.slope_max + self.slope_falloff, slope)
        } else {
            1.0
        };

        lower_blend * upper_blend
    }

    /// Calculate combined height and slope weight.
    #[inline]
    pub fn combined_weight(&self, height: f32, slope_degrees: f32) -> f32 {
        self.height_weight(height) * self.slope_weight(slope_degrees)
    }
}

// ---------------------------------------------------------------------------
// GPU Splat Data
// ---------------------------------------------------------------------------

/// GPU splat map pixel data (2x RGBA8 for 8 layers).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
pub struct SplatPixel {
    /// Weights for layers 0-3 (RGBA).
    pub weights_0_3: [u8; 4],
    /// Weights for layers 4-7 (RGBA).
    pub weights_4_7: [u8; 4],
}

impl Default for SplatPixel {
    fn default() -> Self {
        Self {
            weights_0_3: [255, 0, 0, 0], // Full weight on layer 0
            weights_4_7: [0, 0, 0, 0],
        }
    }
}

impl SplatPixel {
    /// Create a new splat pixel with specified layer weights (0-255).
    pub fn new(weights: [u8; 8]) -> Self {
        Self {
            weights_0_3: [weights[0], weights[1], weights[2], weights[3]],
            weights_4_7: [weights[4], weights[5], weights[6], weights[7]],
        }
    }

    /// Create from normalized float weights (0.0-1.0).
    pub fn from_normalized(weights: [f32; 8]) -> Self {
        let to_u8 = |w: f32| (w.clamp(0.0, 1.0) * 255.0).round() as u8;
        Self {
            weights_0_3: [to_u8(weights[0]), to_u8(weights[1]), to_u8(weights[2]), to_u8(weights[3])],
            weights_4_7: [to_u8(weights[4]), to_u8(weights[5]), to_u8(weights[6]), to_u8(weights[7])],
        }
    }

    /// Get weights as normalized floats.
    pub fn to_normalized(&self) -> [f32; 8] {
        [
            self.weights_0_3[0] as f32 / 255.0,
            self.weights_0_3[1] as f32 / 255.0,
            self.weights_0_3[2] as f32 / 255.0,
            self.weights_0_3[3] as f32 / 255.0,
            self.weights_4_7[0] as f32 / 255.0,
            self.weights_4_7[1] as f32 / 255.0,
            self.weights_4_7[2] as f32 / 255.0,
            self.weights_4_7[3] as f32 / 255.0,
        ]
    }

    /// Get the weight for a specific layer.
    #[inline]
    pub fn get_weight(&self, layer: usize) -> u8 {
        match layer {
            0..=3 => self.weights_0_3[layer],
            4..=7 => self.weights_4_7[layer - 4],
            _ => 0,
        }
    }

    /// Set the weight for a specific layer.
    #[inline]
    pub fn set_weight(&mut self, layer: usize, weight: u8) {
        match layer {
            0..=3 => self.weights_0_3[layer] = weight,
            4..=7 => self.weights_4_7[layer - 4] = weight,
            _ => {}
        }
    }

    /// Get normalized weight for a specific layer.
    #[inline]
    pub fn get_weight_normalized(&self, layer: usize) -> f32 {
        self.get_weight(layer) as f32 / 255.0
    }
}

// ---------------------------------------------------------------------------
// Stochastic Sampling
// ---------------------------------------------------------------------------

/// Stochastic sampling parameters for tiling reduction.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct StochasticParams {
    /// UV offset scale.
    pub offset_scale: f32,
    /// Rotation angle scale (radians).
    pub rotation_scale: f32,
    /// Blend weight for stochastic contribution.
    pub blend_weight: f32,
    /// Hash seed for reproducibility.
    pub hash_seed: u32,
}

impl Default for StochasticParams {
    fn default() -> Self {
        Self {
            offset_scale: STOCHASTIC_SCALE,
            rotation_scale: std::f32::consts::PI * 0.25,
            blend_weight: 0.5,
            hash_seed: 0x12345678,
        }
    }
}

impl StochasticParams {
    /// Generate stochastic UV offset for a world position.
    ///
    /// Uses a deterministic hash to ensure consistent results.
    pub fn uv_offset(&self, world_pos: [f32; 2]) -> [f32; 2] {
        let hash = self.hash_position(world_pos);
        let angle = (hash as f32 / u32::MAX as f32) * std::f32::consts::TAU;
        let radius = self.offset_scale * ((hash >> 16) as f32 / u16::MAX as f32);

        [angle.cos() * radius, angle.sin() * radius]
    }

    /// Generate stochastic rotation for a world position (radians).
    pub fn rotation(&self, world_pos: [f32; 2]) -> f32 {
        let hash = self.hash_position([world_pos[0] + 100.0, world_pos[1] + 100.0]);
        (hash as f32 / u32::MAX as f32 - 0.5) * 2.0 * self.rotation_scale
    }

    /// Generate stochastic cell index for hex tiling.
    ///
    /// Returns the hex cell center and blend weight.
    pub fn hex_cell(&self, world_pos: [f32; 2], scale: f32) -> HexCell {
        let scaled = [world_pos[0] / scale, world_pos[1] / scale];

        // Skewed coordinates for hex grid
        let s = scaled[0] + scaled[1] / 1.732050808;
        let t = scaled[1] * 2.0 / 1.732050808;

        let si = s.floor() as i32;
        let ti = t.floor() as i32;
        let sf = s - si as f32;
        let tf = t - ti as f32;

        // Determine which hex cell we're in
        let (cell_i, cell_j) = if sf + tf < 1.0 {
            (si, ti)
        } else {
            (si + 1, ti + 1)
        };

        // Calculate barycentric-like weight within cell
        let weight = 1.0 - ((sf - 0.5).abs() + (tf - 0.5).abs()).min(1.0);

        HexCell {
            index: [cell_i, cell_j],
            weight,
            center: self.hex_center(cell_i, cell_j, scale),
        }
    }

    /// Get hex cell center in world coordinates.
    fn hex_center(&self, i: i32, j: i32, scale: f32) -> [f32; 2] {
        let x = (i as f32 - j as f32 * 0.5) * scale;
        let y = j as f32 * scale * 0.8660254; // sqrt(3)/2
        [x, y]
    }

    /// Hash a 2D position to a u32.
    fn hash_position(&self, pos: [f32; 2]) -> u32 {
        // Simple hash combining position bits with seed
        let xi = (pos[0] * 1000.0) as i32;
        let yi = (pos[1] * 1000.0) as i32;

        let mut h = self.hash_seed;
        h = h.wrapping_mul(0x9e3779b9).wrapping_add(xi as u32);
        h ^= h >> 16;
        h = h.wrapping_mul(0x85ebca6b).wrapping_add(yi as u32);
        h ^= h >> 13;
        h = h.wrapping_mul(0xc2b2ae35);
        h ^= h >> 16;
        h
    }
}

/// Hex cell information for stochastic tiling.
#[derive(Debug, Clone, Copy)]
pub struct HexCell {
    /// Cell grid index.
    pub index: [i32; 2],
    /// Blend weight within cell.
    pub weight: f32,
    /// Cell center in world coordinates.
    pub center: [f32; 2],
}

// ---------------------------------------------------------------------------
// SplatMap
// ---------------------------------------------------------------------------

/// Terrain splat map for material blending.
///
/// Manages up to 8 terrain layers with automatic height/slope-based
/// blending and manual painting support.
pub struct SplatMap {
    /// Configuration.
    config: TerrainMaterialConfig,
    /// Layer definitions.
    layers: [TerrainLayerDef; MAX_TERRAIN_LAYERS],
    /// Number of active layers.
    active_layers: usize,
    /// Splat pixel data (row-major).
    pixels: Vec<SplatPixel>,
    /// Stochastic sampling parameters.
    stochastic: StochasticParams,
    /// World-space bounds [min_x, min_z, max_x, max_z].
    world_bounds: [f32; 4],
}

impl SplatMap {
    /// Create a new splat map with default configuration.
    pub fn new(config: TerrainMaterialConfig) -> Self {
        let total_texels = config.total_texels();
        Self {
            config,
            layers: [TerrainLayerDef::default(); MAX_TERRAIN_LAYERS],
            active_layers: 1,
            pixels: vec![SplatPixel::default(); total_texels],
            stochastic: StochasticParams::default(),
            world_bounds: [0.0, 0.0, 1000.0, 1000.0],
        }
    }

    /// Create with specified world bounds.
    pub fn with_bounds(config: TerrainMaterialConfig, bounds: [f32; 4]) -> Self {
        let mut map = Self::new(config);
        map.world_bounds = bounds;
        map
    }

    /// Get the configuration.
    #[inline]
    pub fn config(&self) -> &TerrainMaterialConfig {
        &self.config
    }

    /// Get the world bounds.
    #[inline]
    pub fn world_bounds(&self) -> [f32; 4] {
        self.world_bounds
    }

    /// Set world bounds.
    pub fn set_world_bounds(&mut self, bounds: [f32; 4]) {
        self.world_bounds = bounds;
    }

    /// Get the number of active layers.
    #[inline]
    pub fn active_layers(&self) -> usize {
        self.active_layers
    }

    /// Get a layer definition.
    pub fn get_layer(&self, index: usize) -> Option<&TerrainLayerDef> {
        if index < MAX_TERRAIN_LAYERS {
            Some(&self.layers[index])
        } else {
            None
        }
    }

    /// Set a layer definition.
    pub fn set_layer(&mut self, index: usize, layer: TerrainLayerDef) -> Result<(), TerrainMaterialError> {
        if index >= MAX_TERRAIN_LAYERS {
            return Err(TerrainMaterialError::InvalidLayerIndex {
                index,
                max: MAX_TERRAIN_LAYERS - 1,
            });
        }
        layer.validate()?;
        self.layers[index] = layer;
        self.active_layers = self.active_layers.max(index + 1);
        Ok(())
    }

    /// Get stochastic sampling parameters.
    #[inline]
    pub fn stochastic(&self) -> &StochasticParams {
        &self.stochastic
    }

    /// Set stochastic sampling parameters.
    pub fn set_stochastic(&mut self, params: StochasticParams) {
        self.stochastic = params;
    }

    /// Get raw pixel data for GPU upload.
    pub fn pixels(&self) -> &[SplatPixel] {
        &self.pixels
    }

    /// Get mutable pixel data.
    pub fn pixels_mut(&mut self) -> &mut [SplatPixel] {
        &mut self.pixels
    }

    /// Convert world position to splat map texel coordinates.
    pub fn world_to_texel(&self, world_x: f32, world_z: f32) -> (i32, i32) {
        let norm_x = (world_x - self.world_bounds[0]) / (self.world_bounds[2] - self.world_bounds[0]);
        let norm_z = (world_z - self.world_bounds[1]) / (self.world_bounds[3] - self.world_bounds[1]);

        let res = self.config.splat_resolution as f32;
        let tx = (norm_x * res).floor() as i32;
        let tz = (norm_z * res).floor() as i32;

        (tx, tz)
    }

    /// Convert texel coordinates to world position (center of texel).
    pub fn texel_to_world(&self, tx: i32, tz: i32) -> (f32, f32) {
        let res = self.config.splat_resolution as f32;
        let norm_x = (tx as f32 + 0.5) / res;
        let norm_z = (tz as f32 + 0.5) / res;

        let world_x = self.world_bounds[0] + norm_x * (self.world_bounds[2] - self.world_bounds[0]);
        let world_z = self.world_bounds[1] + norm_z * (self.world_bounds[3] - self.world_bounds[1]);

        (world_x, world_z)
    }

    /// Get pixel at texel coordinates.
    pub fn get_pixel(&self, tx: i32, tz: i32) -> Option<&SplatPixel> {
        let res = self.config.splat_resolution as i32;
        if tx < 0 || tx >= res || tz < 0 || tz >= res {
            return None;
        }
        let idx = (tz * res + tx) as usize;
        self.pixels.get(idx)
    }

    /// Get mutable pixel at texel coordinates.
    pub fn get_pixel_mut(&mut self, tx: i32, tz: i32) -> Option<&mut SplatPixel> {
        let res = self.config.splat_resolution as i32;
        if tx < 0 || tx >= res || tz < 0 || tz >= res {
            return None;
        }
        let idx = (tz * res + tx) as usize;
        self.pixels.get_mut(idx)
    }

    /// Sample weights with bilinear interpolation.
    pub fn sample_bilinear(&self, world_x: f32, world_z: f32) -> [f32; 8] {
        let (tx, tz) = self.world_to_texel(world_x, world_z);

        // Get four neighboring pixels
        let p00 = self.get_pixel(tx, tz).map(|p| p.to_normalized()).unwrap_or([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let p10 = self.get_pixel(tx + 1, tz).map(|p| p.to_normalized()).unwrap_or(p00);
        let p01 = self.get_pixel(tx, tz + 1).map(|p| p.to_normalized()).unwrap_or(p00);
        let p11 = self.get_pixel(tx + 1, tz + 1).map(|p| p.to_normalized()).unwrap_or(p00);

        // Calculate fractional position
        let (wx, wz) = self.texel_to_world(tx, tz);
        let texel_size_x = (self.world_bounds[2] - self.world_bounds[0]) / self.config.splat_resolution as f32;
        let texel_size_z = (self.world_bounds[3] - self.world_bounds[1]) / self.config.splat_resolution as f32;
        let fx = ((world_x - wx) / texel_size_x + 0.5).clamp(0.0, 1.0);
        let fz = ((world_z - wz) / texel_size_z + 0.5).clamp(0.0, 1.0);

        // Bilinear interpolation
        let mut result = [0.0f32; 8];
        for i in 0..8 {
            let v0 = p00[i] * (1.0 - fx) + p10[i] * fx;
            let v1 = p01[i] * (1.0 - fx) + p11[i] * fx;
            result[i] = v0 * (1.0 - fz) + v1 * fz;
        }

        result
    }

    /// Calculate blended weights at a world position using auto-blending.
    ///
    /// Combines splat map weights with height and slope-based auto-blending.
    ///
    /// # Arguments
    ///
    /// * `world_pos` - [x, y, z] world position
    /// * `slope_degrees` - Terrain slope in degrees (0-90)
    pub fn blend_weights(&self, world_pos: [f32; 3], slope_degrees: f32) -> [f32; 8] {
        // Start with splat map weights
        let mut weights = self.sample_bilinear(world_pos[0], world_pos[2]);

        // Apply height and slope modulation from layer definitions
        let height = world_pos[1];
        for i in 0..self.active_layers.min(MAX_TERRAIN_LAYERS) {
            let layer = &self.layers[i];
            let auto_weight = layer.combined_weight(height, slope_degrees);
            weights[i] *= auto_weight;
        }

        // Normalize weights
        Self::normalize_weights(&mut weights);

        weights
    }

    /// Normalize weights so they sum to 1.0.
    ///
    /// If all weights are zero, sets layer 0 to 1.0.
    pub fn normalize_weights(weights: &mut [f32; 8]) {
        let sum: f32 = weights.iter().sum();

        if sum < WEIGHT_EPSILON {
            // All weights near zero, default to layer 0
            weights[0] = 1.0;
            for w in weights.iter_mut().skip(1) {
                *w = 0.0;
            }
        } else {
            let inv_sum = 1.0 / sum;
            for w in weights.iter_mut() {
                *w *= inv_sum;
            }
        }
    }

    /// Apply height-based auto-blending to entire splat map.
    ///
    /// Uses a height lookup function to determine height at each texel.
    pub fn auto_blend_height<F>(&mut self, height_fn: F)
    where
        F: Fn(f32, f32) -> f32,
    {
        let res = self.config.splat_resolution as i32;

        for tz in 0..res {
            for tx in 0..res {
                let (wx, wz) = self.texel_to_world(tx, tz);
                let height = height_fn(wx, wz);

                let mut weights = [0.0f32; 8];
                for i in 0..self.active_layers.min(MAX_TERRAIN_LAYERS) {
                    weights[i] = self.layers[i].height_weight(height);
                }
                Self::normalize_weights(&mut weights);

                if let Some(pixel) = self.get_pixel_mut(tx, tz) {
                    *pixel = SplatPixel::from_normalized(weights);
                }
            }
        }
    }

    /// Apply slope-based auto-blending to entire splat map.
    pub fn auto_blend_slope<F>(&mut self, slope_fn: F)
    where
        F: Fn(f32, f32) -> f32,
    {
        let res = self.config.splat_resolution as i32;

        for tz in 0..res {
            for tx in 0..res {
                let (wx, wz) = self.texel_to_world(tx, tz);
                let slope = slope_fn(wx, wz);

                let mut weights = [0.0f32; 8];
                for i in 0..self.active_layers.min(MAX_TERRAIN_LAYERS) {
                    weights[i] = self.layers[i].slope_weight(slope);
                }
                Self::normalize_weights(&mut weights);

                if let Some(pixel) = self.get_pixel_mut(tx, tz) {
                    *pixel = SplatPixel::from_normalized(weights);
                }
            }
        }
    }

    /// Apply combined height and slope auto-blending.
    pub fn auto_blend_combined<H, S>(&mut self, height_fn: H, slope_fn: S)
    where
        H: Fn(f32, f32) -> f32,
        S: Fn(f32, f32) -> f32,
    {
        let res = self.config.splat_resolution as i32;

        for tz in 0..res {
            for tx in 0..res {
                let (wx, wz) = self.texel_to_world(tx, tz);
                let height = height_fn(wx, wz);
                let slope = slope_fn(wx, wz);

                let mut weights = [0.0f32; 8];
                for i in 0..self.active_layers.min(MAX_TERRAIN_LAYERS) {
                    weights[i] = self.layers[i].combined_weight(height, slope);
                }
                Self::normalize_weights(&mut weights);

                if let Some(pixel) = self.get_pixel_mut(tx, tz) {
                    *pixel = SplatPixel::from_normalized(weights);
                }
            }
        }
    }

    /// Paint a circular brush stroke on the splat map.
    ///
    /// # Arguments
    ///
    /// * `center_x`, `center_z` - World position of brush center
    /// * `radius` - Brush radius in world units
    /// * `layer` - Target layer index (0-7)
    /// * `strength` - Paint strength (0.0-1.0)
    /// * `falloff` - Edge falloff (0.0 = hard, 1.0 = soft)
    pub fn paint_circle(
        &mut self,
        center_x: f32,
        center_z: f32,
        radius: f32,
        layer: usize,
        strength: f32,
        falloff: f32,
    ) {
        if layer >= MAX_TERRAIN_LAYERS {
            return;
        }

        let (center_tx, center_tz) = self.world_to_texel(center_x, center_z);
        let texel_size = (self.world_bounds[2] - self.world_bounds[0]) / self.config.splat_resolution as f32;
        let radius_texels = (radius / texel_size).ceil() as i32;

        let res = self.config.splat_resolution as i32;

        for dz in -radius_texels..=radius_texels {
            for dx in -radius_texels..=radius_texels {
                let tx = center_tx + dx;
                let tz = center_tz + dz;

                if tx < 0 || tx >= res || tz < 0 || tz >= res {
                    continue;
                }

                let (wx, wz) = self.texel_to_world(tx, tz);
                let dist = ((wx - center_x).powi(2) + (wz - center_z).powi(2)).sqrt();

                if dist > radius {
                    continue;
                }

                // Calculate falloff
                let falloff_start = radius * (1.0 - falloff);
                let brush_weight = if dist < falloff_start {
                    1.0
                } else {
                    1.0 - (dist - falloff_start) / (radius - falloff_start)
                };

                let paint_strength = strength * brush_weight;

                if let Some(pixel) = self.get_pixel_mut(tx, tz) {
                    let mut weights = pixel.to_normalized();

                    // Add to target layer
                    weights[layer] += paint_strength;

                    // Normalize
                    Self::normalize_weights(&mut weights);

                    *pixel = SplatPixel::from_normalized(weights);
                }
            }
        }
    }

    /// Fill entire splat map with a single layer.
    pub fn fill(&mut self, layer: usize) {
        if layer >= MAX_TERRAIN_LAYERS {
            return;
        }

        let mut weights = [0.0f32; 8];
        weights[layer] = 1.0;
        let pixel = SplatPixel::from_normalized(weights);

        for p in &mut self.pixels {
            *p = pixel;
        }
    }

    /// Generate stochastic UV offset for a world position.
    pub fn stochastic_offset(&self, world_x: f32, world_z: f32, layer: usize) -> [f32; 2] {
        if layer >= MAX_TERRAIN_LAYERS {
            return [0.0, 0.0];
        }

        let factor = self.layers[layer].stochastic_factor;
        if factor < WEIGHT_EPSILON {
            return [0.0, 0.0];
        }

        let offset = self.stochastic.uv_offset([world_x, world_z]);
        [offset[0] * factor, offset[1] * factor]
    }

    /// Sort layers by priority and return sorted indices.
    pub fn sorted_layer_indices(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.active_layers).collect();
        indices.sort_by(|&a, &b| {
            self.layers[b]
                .priority
                .partial_cmp(&self.layers[a].priority)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indices
    }

    /// Get scaled UV for a layer at world position.
    pub fn layer_uv(&self, layer: usize, world_x: f32, world_z: f32) -> [f32; 2] {
        if layer >= MAX_TERRAIN_LAYERS {
            return [0.0, 0.0];
        }

        let layer_def = &self.layers[layer];
        let base_scale = self.config.uv_scale * layer_def.uv_scale;

        let u = world_x * base_scale;
        let v = world_z * base_scale;

        // Apply stochastic offset
        let offset = self.stochastic_offset(world_x, world_z, layer);

        [u + offset[0], v + offset[1]]
    }
}

// ---------------------------------------------------------------------------
// Utility Functions
// ---------------------------------------------------------------------------

/// Smoothstep interpolation.
#[inline]
pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Smoother step (quintic) interpolation.
#[inline]
pub fn smootherstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// Calculate slope angle from surface normal.
///
/// Returns slope in degrees (0 = flat, 90 = vertical).
#[inline]
pub fn slope_from_normal(normal: [f32; 3]) -> f32 {
    // Normalize the normal vector
    let len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    if len < WEIGHT_EPSILON {
        return 0.0;
    }

    let ny = normal[1] / len;
    let angle_rad = ny.clamp(-1.0, 1.0).acos();
    angle_rad.to_degrees()
}

/// Calculate slope angle from height gradient.
///
/// # Arguments
///
/// * `dx` - Height change in X direction
/// * `dz` - Height change in Z direction
/// * `spacing` - Sample spacing in world units
#[inline]
pub fn slope_from_gradient(dx: f32, dz: f32, spacing: f32) -> f32 {
    let gradient_x = dx / spacing;
    let gradient_z = dz / spacing;
    let gradient_mag = (gradient_x * gradient_x + gradient_z * gradient_z).sqrt();
    gradient_mag.atan().to_degrees()
}

/// Compute terrain normal from 4 neighboring heights (cross pattern).
///
/// # Arguments
///
/// * `h_center` - Height at center
/// * `h_left`, `h_right` - Heights at x-1 and x+1
/// * `h_back`, `h_front` - Heights at z-1 and z+1
/// * `spacing` - Sample spacing
pub fn compute_normal_from_heights(
    _h_center: f32,
    h_left: f32,
    h_right: f32,
    h_back: f32,
    h_front: f32,
    spacing: f32,
) -> [f32; 3] {
    let dx = (h_right - h_left) / (2.0 * spacing);
    let dz = (h_front - h_back) / (2.0 * spacing);

    // Normal = cross(tangent_x, tangent_z) = (-dx, 1, -dz) normalized
    let nx = -dx;
    let ny = 1.0;
    let nz = -dz;

    let len = (nx * nx + ny * ny + nz * nz).sqrt();
    [nx / len, ny / len, nz / len]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // TerrainMaterialConfig Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_default() {
        let config = TerrainMaterialConfig::default();
        assert_eq!(config.splat_resolution, DEFAULT_SPLAT_RESOLUTION);
        assert_eq!(config.max_layers, MAX_TERRAIN_LAYERS as u32);
        assert_eq!(config.uv_scale, DEFAULT_UV_SCALE);
    }

    #[test]
    fn test_config_new_valid() {
        let config = TerrainMaterialConfig::new(512, 4, 2.0);
        assert_eq!(config.splat_resolution, 512);
        assert_eq!(config.max_layers, 4);
        assert_eq!(config.uv_scale, 2.0);
    }

    #[test]
    #[should_panic]
    fn test_config_new_invalid_resolution_low() {
        TerrainMaterialConfig::new(32, 8, 1.0);
    }

    #[test]
    #[should_panic]
    fn test_config_new_invalid_resolution_high() {
        TerrainMaterialConfig::new(16384, 8, 1.0);
    }

    #[test]
    #[should_panic]
    fn test_config_new_invalid_layers() {
        TerrainMaterialConfig::new(512, 0, 1.0);
    }

    #[test]
    #[should_panic]
    fn test_config_new_invalid_uv_scale() {
        TerrainMaterialConfig::new(512, 8, 0.0);
    }

    #[test]
    fn test_config_try_new_valid() {
        let result = TerrainMaterialConfig::try_new(256, 6, 0.5);
        assert!(result.is_ok());
        let config = result.unwrap();
        assert_eq!(config.splat_resolution, 256);
    }

    #[test]
    fn test_config_try_new_invalid() {
        let result = TerrainMaterialConfig::try_new(10, 8, 1.0);
        assert!(matches!(result, Err(TerrainMaterialError::InvalidResolution { .. })));
    }

    #[test]
    fn test_config_total_texels() {
        let config = TerrainMaterialConfig::new(128, 8, 1.0);
        assert_eq!(config.total_texels(), 128 * 128);
    }

    #[test]
    fn test_config_validate() {
        let config = TerrainMaterialConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_pod_alignment() {
        assert_eq!(std::mem::size_of::<TerrainMaterialConfig>(), 16);
    }

    // -----------------------------------------------------------------------
    // TerrainLayerDef Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_layer_default() {
        let layer = TerrainLayerDef::default();
        assert_eq!(layer.uv_scale, 1.0);
        assert_eq!(layer.slope_min, 0.0);
        assert_eq!(layer.slope_max, 90.0);
    }

    #[test]
    fn test_layer_new() {
        let layer = TerrainLayerDef::new(2.0, 10.0, 50.0);
        assert_eq!(layer.uv_scale, 2.0);
        assert_eq!(layer.height_min, 10.0);
        assert_eq!(layer.height_max, 50.0);
    }

    #[test]
    fn test_layer_flat() {
        let layer = TerrainLayerDef::flat_layer(1.0, 0.0, 100.0);
        assert_eq!(layer.slope_max, 25.0);
    }

    #[test]
    fn test_layer_slope() {
        let layer = TerrainLayerDef::slope_layer(0.5, 35.0);
        assert_eq!(layer.slope_min, 35.0);
        assert_eq!(layer.slope_max, 90.0);
        assert_eq!(layer.triplanar_scale, 1.0);
    }

    #[test]
    fn test_layer_with_textures() {
        let layer = TerrainLayerDef::default().with_textures(1, 2, 3);
        assert_eq!(layer.albedo_index, 1);
        assert_eq!(layer.normal_index, 2);
        assert_eq!(layer.orm_index, 3);
    }

    #[test]
    fn test_layer_with_stochastic() {
        let layer = TerrainLayerDef::default().with_stochastic(0.5);
        assert_eq!(layer.stochastic_factor, 0.5);
    }

    #[test]
    fn test_layer_with_stochastic_clamp() {
        let layer = TerrainLayerDef::default().with_stochastic(2.0);
        assert_eq!(layer.stochastic_factor, 1.0);
    }

    #[test]
    fn test_layer_with_priority() {
        let layer = TerrainLayerDef::default().with_priority(10.0);
        assert_eq!(layer.priority, 10.0);
    }

    #[test]
    fn test_layer_validate_valid() {
        let layer = TerrainLayerDef::new(1.0, 0.0, 100.0);
        assert!(layer.validate().is_ok());
    }

    #[test]
    fn test_layer_validate_invalid_height() {
        let layer = TerrainLayerDef::new(1.0, 100.0, 50.0);
        assert!(matches!(layer.validate(), Err(TerrainMaterialError::InvalidHeightRange { .. })));
    }

    #[test]
    fn test_layer_validate_invalid_slope() {
        let mut layer = TerrainLayerDef::default();
        layer.slope_min = -10.0;
        assert!(matches!(layer.validate(), Err(TerrainMaterialError::InvalidSlopeRange { .. })));
    }

    #[test]
    fn test_layer_validate_invalid_uv() {
        let mut layer = TerrainLayerDef::default();
        layer.uv_scale = 0.0;
        assert!(matches!(layer.validate(), Err(TerrainMaterialError::InvalidUvScale { .. })));
    }

    // -----------------------------------------------------------------------
    // Height Weight Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_height_weight_inside_range() {
        let layer = TerrainLayerDef::new(1.0, 10.0, 50.0);
        assert!((layer.height_weight(30.0) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_height_weight_outside_range() {
        let layer = TerrainLayerDef::new(1.0, 10.0, 50.0);
        assert!((layer.height_weight(0.0 - DEFAULT_BLEND_FALLOFF - 1.0)).abs() < 0.001);
        assert!((layer.height_weight(60.0 + DEFAULT_BLEND_FALLOFF + 1.0)).abs() < 0.001);
    }

    #[test]
    fn test_height_weight_at_boundary() {
        let layer = TerrainLayerDef::new(1.0, 10.0, 50.0);
        let w = layer.height_weight(10.0);
        assert!(w >= 0.5 && w <= 1.0);
    }

    #[test]
    fn test_height_weight_falloff() {
        let layer = TerrainLayerDef::new(1.0, 10.0, 50.0);
        let w1 = layer.height_weight(5.0);
        let w2 = layer.height_weight(7.0);
        let w3 = layer.height_weight(10.0);
        assert!(w1 < w2);
        assert!(w2 < w3);
    }

    // -----------------------------------------------------------------------
    // Slope Weight Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_slope_weight_inside_range() {
        let layer = TerrainLayerDef::flat_layer(1.0, 0.0, 100.0);
        assert!((layer.slope_weight(10.0) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_slope_weight_outside_range() {
        let layer = TerrainLayerDef::flat_layer(1.0, 0.0, 100.0);
        let w = layer.slope_weight(40.0);
        assert!(w < 0.5);
    }

    #[test]
    fn test_slope_weight_at_boundary() {
        let layer = TerrainLayerDef::flat_layer(1.0, 0.0, 100.0);
        let w = layer.slope_weight(25.0);
        assert!(w >= 0.5 && w <= 1.0);
    }

    #[test]
    fn test_slope_weight_clamped() {
        let layer = TerrainLayerDef::default();
        let w_neg = layer.slope_weight(-10.0);
        let w_over = layer.slope_weight(100.0);
        // Should clamp to 0-90 range
        assert!(w_neg >= 0.0);
        assert!(w_over >= 0.0);
    }

    #[test]
    fn test_slope_weight_gradient() {
        let layer = TerrainLayerDef::flat_layer(1.0, 0.0, 100.0);
        let w1 = layer.slope_weight(20.0);
        let w2 = layer.slope_weight(25.0);
        let w3 = layer.slope_weight(30.0);
        // Should decrease as slope increases past max
        assert!(w1 >= w2);
        assert!(w2 >= w3);
    }

    // -----------------------------------------------------------------------
    // Combined Weight Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_combined_weight_both_valid() {
        let layer = TerrainLayerDef::new(1.0, 0.0, 100.0);
        let w = layer.combined_weight(50.0, 10.0);
        assert!((w - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_combined_weight_height_invalid() {
        let layer = TerrainLayerDef::new(1.0, 50.0, 100.0);
        let w = layer.combined_weight(0.0, 10.0);
        assert!(w < 0.5);
    }

    #[test]
    fn test_combined_weight_slope_invalid() {
        let layer = TerrainLayerDef::flat_layer(1.0, 0.0, 100.0);
        let w = layer.combined_weight(50.0, 60.0);
        assert!(w < 0.5);
    }

    #[test]
    fn test_combined_weight_both_invalid() {
        let layer = TerrainLayerDef::flat_layer(1.0, 50.0, 100.0);
        let w = layer.combined_weight(0.0, 60.0);
        assert!(w < 0.1);
    }

    // -----------------------------------------------------------------------
    // SplatPixel Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_splat_pixel_default() {
        let pixel = SplatPixel::default();
        assert_eq!(pixel.weights_0_3[0], 255);
        assert_eq!(pixel.weights_0_3[1], 0);
    }

    #[test]
    fn test_splat_pixel_new() {
        let pixel = SplatPixel::new([100, 50, 50, 55, 0, 0, 0, 0]);
        assert_eq!(pixel.get_weight(0), 100);
        assert_eq!(pixel.get_weight(1), 50);
    }

    #[test]
    fn test_splat_pixel_from_normalized() {
        let pixel = SplatPixel::from_normalized([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(pixel.get_weight(0), 255);
    }

    #[test]
    fn test_splat_pixel_to_normalized() {
        let pixel = SplatPixel::new([255, 0, 0, 0, 0, 0, 0, 0]);
        let norm = pixel.to_normalized();
        assert!((norm[0] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_splat_pixel_get_set_weight() {
        let mut pixel = SplatPixel::default();
        pixel.set_weight(5, 128);
        assert_eq!(pixel.get_weight(5), 128);
    }

    #[test]
    fn test_splat_pixel_get_weight_out_of_bounds() {
        let pixel = SplatPixel::default();
        assert_eq!(pixel.get_weight(10), 0);
    }

    #[test]
    fn test_splat_pixel_normalized_roundtrip() {
        let original = [0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0];
        let pixel = SplatPixel::from_normalized(original);
        let result = pixel.to_normalized();
        for i in 0..8 {
            assert!((result[i] - original[i]).abs() < 0.01);
        }
    }

    #[test]
    fn test_splat_pixel_pod_size() {
        assert_eq!(std::mem::size_of::<SplatPixel>(), 8);
    }

    // -----------------------------------------------------------------------
    // Weight Normalization Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_normalize_weights_sum_to_one() {
        let mut weights = [0.5, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0];
        SplatMap::normalize_weights(&mut weights);
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_normalize_weights_preserves_ratios() {
        let mut weights = [2.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        SplatMap::normalize_weights(&mut weights);
        assert!((weights[0] - 0.5).abs() < 0.001);
        assert!((weights[1] - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_normalize_weights_all_zero() {
        let mut weights = [0.0; 8];
        SplatMap::normalize_weights(&mut weights);
        assert_eq!(weights[0], 1.0);
        assert_eq!(weights[1], 0.0);
    }

    #[test]
    fn test_normalize_weights_single_layer() {
        let mut weights = [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0];
        SplatMap::normalize_weights(&mut weights);
        assert!((weights[2] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_normalize_weights_already_normalized() {
        let mut weights = [0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0];
        let original = weights;
        SplatMap::normalize_weights(&mut weights);
        for i in 0..8 {
            assert!((weights[i] - original[i]).abs() < 0.001);
        }
    }

    // -----------------------------------------------------------------------
    // SplatMap Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_splatmap_new() {
        let config = TerrainMaterialConfig::new(64, 8, 1.0);
        let splat = SplatMap::new(config);
        assert_eq!(splat.pixels().len(), 64 * 64);
    }

    #[test]
    fn test_splatmap_with_bounds() {
        let config = TerrainMaterialConfig::new(64, 8, 1.0);
        let bounds = [0.0, 0.0, 500.0, 500.0];
        let splat = SplatMap::with_bounds(config, bounds);
        assert_eq!(splat.world_bounds(), bounds);
    }

    #[test]
    fn test_splatmap_set_layer() {
        let config = TerrainMaterialConfig::new(64, 8, 1.0);
        let mut splat = SplatMap::new(config);
        let layer = TerrainLayerDef::new(2.0, 0.0, 100.0);
        assert!(splat.set_layer(0, layer).is_ok());
        assert_eq!(splat.get_layer(0).unwrap().uv_scale, 2.0);
    }

    #[test]
    fn test_splatmap_set_layer_invalid_index() {
        let config = TerrainMaterialConfig::new(64, 8, 1.0);
        let mut splat = SplatMap::new(config);
        let layer = TerrainLayerDef::default();
        assert!(splat.set_layer(10, layer).is_err());
    }

    #[test]
    fn test_splatmap_active_layers() {
        let config = TerrainMaterialConfig::new(64, 8, 1.0);
        let mut splat = SplatMap::new(config);
        assert_eq!(splat.active_layers(), 1);

        let _ = splat.set_layer(3, TerrainLayerDef::default());
        assert_eq!(splat.active_layers(), 4);
    }

    #[test]
    fn test_splatmap_world_to_texel() {
        let config = TerrainMaterialConfig::new(100, 8, 1.0);
        let splat = SplatMap::with_bounds(config, [0.0, 0.0, 100.0, 100.0]);

        let (tx, tz) = splat.world_to_texel(50.0, 50.0);
        assert_eq!(tx, 50);
        assert_eq!(tz, 50);
    }

    #[test]
    fn test_splatmap_texel_to_world() {
        let config = TerrainMaterialConfig::new(100, 8, 1.0);
        let splat = SplatMap::with_bounds(config, [0.0, 0.0, 100.0, 100.0]);

        let (wx, wz) = splat.texel_to_world(50, 50);
        assert!((wx - 50.5).abs() < 0.1);
        assert!((wz - 50.5).abs() < 0.1);
    }

    #[test]
    fn test_splatmap_get_pixel() {
        let config = TerrainMaterialConfig::new(64, 8, 1.0);
        let splat = SplatMap::new(config);

        assert!(splat.get_pixel(0, 0).is_some());
        assert!(splat.get_pixel(63, 63).is_some());
        assert!(splat.get_pixel(64, 0).is_none());
        assert!(splat.get_pixel(-1, 0).is_none());
    }

    #[test]
    fn test_splatmap_fill() {
        let config = TerrainMaterialConfig::new(64, 8, 1.0);
        let mut splat = SplatMap::new(config);

        splat.fill(3);

        let pixel = splat.get_pixel(32, 32).unwrap();
        assert_eq!(pixel.get_weight(3), 255);
        assert_eq!(pixel.get_weight(0), 0);
    }

    #[test]
    fn test_splatmap_sample_bilinear_center() {
        let config = TerrainMaterialConfig::new(64, 8, 1.0);
        let splat = SplatMap::with_bounds(config, [0.0, 0.0, 64.0, 64.0]);

        let weights = splat.sample_bilinear(32.0, 32.0);
        // Default is layer 0 at full weight
        assert!(weights[0] > 0.9);
    }

    // -----------------------------------------------------------------------
    // Blend Weights Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_blend_weights_default() {
        let config = TerrainMaterialConfig::new(64, 8, 1.0);
        let splat = SplatMap::with_bounds(config, [0.0, 0.0, 64.0, 64.0]);

        let weights = splat.blend_weights([32.0, 50.0, 32.0], 0.0);
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_blend_weights_height_modulation() {
        let config = TerrainMaterialConfig::new(64, 8, 1.0);
        let mut splat = SplatMap::with_bounds(config, [0.0, 0.0, 64.0, 64.0]);

        // Layer 0: low altitude
        let _ = splat.set_layer(0, TerrainLayerDef::new(1.0, 0.0, 50.0));
        // Layer 1: high altitude
        let _ = splat.set_layer(1, TerrainLayerDef::new(1.0, 40.0, 100.0));

        // Fill with equal weights
        for p in splat.pixels_mut() {
            *p = SplatPixel::from_normalized([0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        }

        let low = splat.blend_weights([32.0, 10.0, 32.0], 0.0);
        let high = splat.blend_weights([32.0, 80.0, 32.0], 0.0);

        assert!(low[0] > low[1]); // Layer 0 dominates at low altitude
        assert!(high[1] > high[0]); // Layer 1 dominates at high altitude
    }

    #[test]
    fn test_blend_weights_slope_modulation() {
        let config = TerrainMaterialConfig::new(64, 8, 1.0);
        let mut splat = SplatMap::with_bounds(config, [0.0, 0.0, 64.0, 64.0]);

        // Layer 0: flat areas
        let _ = splat.set_layer(0, TerrainLayerDef::flat_layer(1.0, 0.0, 1000.0));
        // Layer 1: steep areas
        let _ = splat.set_layer(1, TerrainLayerDef::slope_layer(1.0, 30.0));

        for p in splat.pixels_mut() {
            *p = SplatPixel::from_normalized([0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        }

        let flat = splat.blend_weights([32.0, 50.0, 32.0], 5.0);
        let steep = splat.blend_weights([32.0, 50.0, 32.0], 60.0);

        assert!(flat[0] > flat[1]); // Layer 0 dominates on flat
        assert!(steep[1] > steep[0]); // Layer 1 dominates on steep
    }

    // -----------------------------------------------------------------------
    // Paint Circle Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_paint_circle_center() {
        let config = TerrainMaterialConfig::new(64, 8, 1.0);
        let mut splat = SplatMap::with_bounds(config, [0.0, 0.0, 64.0, 64.0]);

        // First clear to layer 0 with zero weight, then paint layer 1
        splat.fill(1);
        // Verify we have full weight on layer 1 after fill
        let pixel = splat.get_pixel(32, 32).unwrap();
        assert_eq!(pixel.get_weight(1), 255);

        // Now test painting from a clean slate
        let mut splat2 = SplatMap::with_bounds(config, [0.0, 0.0, 64.0, 64.0]);
        // Paint with high strength - adds to existing layer 0 (255)
        splat2.paint_circle(32.0, 32.0, 10.0, 1, 1.0, 0.0);

        let pixel2 = splat2.get_pixel(32, 32).unwrap();
        // Layer 1 should have significant weight after painting
        // Initial: [255, 0, ...], after adding 1.0: [255, 255, ...] (as floats)
        // After normalization: [127, 127, ...]
        assert!(pixel2.get_weight(1) > 100);
        assert!(pixel2.get_weight(0) > 100);
    }

    #[test]
    fn test_paint_circle_edge() {
        let config = TerrainMaterialConfig::new(64, 8, 1.0);
        let mut splat = SplatMap::with_bounds(config, [0.0, 0.0, 64.0, 64.0]);

        splat.paint_circle(32.0, 32.0, 5.0, 1, 1.0, 0.0);

        // Should not affect pixels outside radius
        let pixel = splat.get_pixel(50, 50).unwrap();
        assert_eq!(pixel.get_weight(1), 0);
    }

    #[test]
    fn test_paint_circle_falloff() {
        let config = TerrainMaterialConfig::new(64, 8, 1.0);
        let mut splat = SplatMap::with_bounds(config, [0.0, 0.0, 64.0, 64.0]);

        splat.paint_circle(32.0, 32.0, 10.0, 1, 1.0, 0.5);

        let center = splat.get_pixel(32, 32).unwrap();
        let edge = splat.get_pixel(37, 32).unwrap(); // Near edge

        assert!(center.get_weight(1) >= edge.get_weight(1));
    }

    #[test]
    fn test_paint_circle_invalid_layer() {
        let config = TerrainMaterialConfig::new(64, 8, 1.0);
        let mut splat = SplatMap::with_bounds(config, [0.0, 0.0, 64.0, 64.0]);

        // Should not panic
        splat.paint_circle(32.0, 32.0, 10.0, 100, 1.0, 0.0);
    }

    // -----------------------------------------------------------------------
    // Stochastic Sampling Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_stochastic_params_default() {
        let params = StochasticParams::default();
        assert_eq!(params.offset_scale, STOCHASTIC_SCALE);
    }

    #[test]
    fn test_stochastic_uv_offset_deterministic() {
        let params = StochasticParams::default();
        let pos = [100.0, 200.0];

        let offset1 = params.uv_offset(pos);
        let offset2 = params.uv_offset(pos);

        assert_eq!(offset1, offset2);
    }

    #[test]
    fn test_stochastic_uv_offset_varies() {
        let params = StochasticParams::default();

        let offset1 = params.uv_offset([100.0, 200.0]);
        let offset2 = params.uv_offset([150.0, 250.0]);

        assert!(offset1[0] != offset2[0] || offset1[1] != offset2[1]);
    }

    #[test]
    fn test_stochastic_rotation() {
        let params = StochasticParams::default();

        let rot = params.rotation([100.0, 200.0]);
        assert!(rot.abs() <= params.rotation_scale);
    }

    #[test]
    fn test_stochastic_hex_cell() {
        let params = StochasticParams::default();

        let cell = params.hex_cell([100.0, 200.0], 10.0);
        assert!(cell.weight >= 0.0 && cell.weight <= 1.0);
    }

    #[test]
    fn test_splatmap_stochastic_offset() {
        let config = TerrainMaterialConfig::new(64, 8, 1.0);
        let mut splat = SplatMap::new(config);

        let mut layer = TerrainLayerDef::default();
        layer.stochastic_factor = 0.5;
        let _ = splat.set_layer(0, layer);

        let offset = splat.stochastic_offset(100.0, 200.0, 0);
        assert!(offset[0] != 0.0 || offset[1] != 0.0);
    }

    #[test]
    fn test_splatmap_stochastic_offset_disabled() {
        let config = TerrainMaterialConfig::new(64, 8, 1.0);
        let splat = SplatMap::new(config);

        // Default layer has stochastic_factor = 0
        let offset = splat.stochastic_offset(100.0, 200.0, 0);
        assert_eq!(offset, [0.0, 0.0]);
    }

    // -----------------------------------------------------------------------
    // Layer UV Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_layer_uv_basic() {
        let config = TerrainMaterialConfig::new(64, 8, 2.0);
        let mut splat = SplatMap::new(config);

        let layer = TerrainLayerDef::new(0.5, 0.0, 100.0);
        let _ = splat.set_layer(0, layer);

        let uv = splat.layer_uv(0, 10.0, 20.0);
        // UV = world_pos * global_scale * layer_scale = 10 * 2 * 0.5 = 10
        assert!((uv[0] - 10.0).abs() < 0.001);
        assert!((uv[1] - 20.0).abs() < 0.001);
    }

    #[test]
    fn test_layer_uv_invalid_layer() {
        let config = TerrainMaterialConfig::new(64, 8, 1.0);
        let splat = SplatMap::new(config);

        let uv = splat.layer_uv(100, 10.0, 20.0);
        assert_eq!(uv, [0.0, 0.0]);
    }

    // -----------------------------------------------------------------------
    // Layer Priority/Ordering Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sorted_layer_indices() {
        let config = TerrainMaterialConfig::new(64, 8, 1.0);
        let mut splat = SplatMap::new(config);

        let _ = splat.set_layer(0, TerrainLayerDef::default().with_priority(1.0));
        let _ = splat.set_layer(1, TerrainLayerDef::default().with_priority(3.0));
        let _ = splat.set_layer(2, TerrainLayerDef::default().with_priority(2.0));

        let sorted = splat.sorted_layer_indices();
        assert_eq!(sorted[0], 1); // Highest priority first
        assert_eq!(sorted[1], 2);
        assert_eq!(sorted[2], 0);
    }

    #[test]
    fn test_sorted_layer_indices_equal_priority() {
        let config = TerrainMaterialConfig::new(64, 8, 1.0);
        let mut splat = SplatMap::new(config);

        let _ = splat.set_layer(0, TerrainLayerDef::default().with_priority(1.0));
        let _ = splat.set_layer(1, TerrainLayerDef::default().with_priority(1.0));

        let sorted = splat.sorted_layer_indices();
        assert_eq!(sorted.len(), 2);
    }

    // -----------------------------------------------------------------------
    // Auto-Blend Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_auto_blend_height() {
        let config = TerrainMaterialConfig::new(64, 8, 1.0);
        let mut splat = SplatMap::with_bounds(config, [0.0, 0.0, 64.0, 64.0]);

        let _ = splat.set_layer(0, TerrainLayerDef::new(1.0, 0.0, 30.0));
        let _ = splat.set_layer(1, TerrainLayerDef::new(1.0, 20.0, 100.0));

        splat.auto_blend_height(|x, z| (x + z) * 0.5);

        // Low corner should be mostly layer 0
        let low = splat.get_pixel(5, 5).unwrap();
        assert!(low.get_weight(0) > low.get_weight(1));

        // High corner should be mostly layer 1
        let high = splat.get_pixel(60, 60).unwrap();
        assert!(high.get_weight(1) > high.get_weight(0));
    }

    #[test]
    fn test_auto_blend_slope() {
        let config = TerrainMaterialConfig::new(64, 8, 1.0);
        let mut splat = SplatMap::with_bounds(config, [0.0, 0.0, 64.0, 64.0]);

        let _ = splat.set_layer(0, TerrainLayerDef::flat_layer(1.0, 0.0, 1000.0));
        let _ = splat.set_layer(1, TerrainLayerDef::slope_layer(1.0, 30.0));

        // Slope increases with x position
        splat.auto_blend_slope(|x, _z| x);

        let flat = splat.get_pixel(5, 32).unwrap();
        let steep = splat.get_pixel(60, 32).unwrap();

        assert!(flat.get_weight(0) > flat.get_weight(1));
        assert!(steep.get_weight(1) > steep.get_weight(0));
    }

    #[test]
    fn test_auto_blend_combined() {
        let config = TerrainMaterialConfig::new(64, 8, 1.0);
        let mut splat = SplatMap::with_bounds(config, [0.0, 0.0, 64.0, 64.0]);

        let _ = splat.set_layer(0, TerrainLayerDef::flat_layer(1.0, 0.0, 50.0));
        let _ = splat.set_layer(1, TerrainLayerDef::slope_layer(1.0, 30.0));

        splat.auto_blend_combined(
            |x, z| (x + z) * 0.5,
            |x, _z| x,
        );

        // Results should reflect both height and slope
        let pixel = splat.get_pixel(32, 32).unwrap();
        let sum: u16 = (0..8).map(|i| pixel.get_weight(i) as u16).sum();
        // Weights should still sum to ~255 after normalization
        assert!(sum > 250 && sum <= 256);
    }

    // -----------------------------------------------------------------------
    // Utility Function Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_smoothstep_bounds() {
        assert!((smoothstep(0.0, 1.0, 0.0) - 0.0).abs() < 0.001);
        assert!((smoothstep(0.0, 1.0, 1.0) - 1.0).abs() < 0.001);
        assert!((smoothstep(0.0, 1.0, 0.5) - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_smoothstep_clamping() {
        assert!((smoothstep(0.0, 1.0, -1.0) - 0.0).abs() < 0.001);
        assert!((smoothstep(0.0, 1.0, 2.0) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_smootherstep_bounds() {
        assert!((smootherstep(0.0, 1.0, 0.0) - 0.0).abs() < 0.001);
        assert!((smootherstep(0.0, 1.0, 1.0) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_slope_from_normal_flat() {
        let slope = slope_from_normal([0.0, 1.0, 0.0]);
        assert!(slope.abs() < 0.1);
    }

    #[test]
    fn test_slope_from_normal_vertical() {
        let slope = slope_from_normal([1.0, 0.0, 0.0]);
        assert!((slope - 90.0).abs() < 0.1);
    }

    #[test]
    fn test_slope_from_normal_45deg() {
        let slope = slope_from_normal([0.707, 0.707, 0.0]);
        assert!((slope - 45.0).abs() < 1.0);
    }

    #[test]
    fn test_slope_from_gradient_flat() {
        let slope = slope_from_gradient(0.0, 0.0, 1.0);
        assert!(slope.abs() < 0.1);
    }

    #[test]
    fn test_slope_from_gradient_steep() {
        let slope = slope_from_gradient(1.0, 0.0, 1.0);
        assert!((slope - 45.0).abs() < 0.1);
    }

    #[test]
    fn test_compute_normal_from_heights_flat() {
        let normal = compute_normal_from_heights(10.0, 10.0, 10.0, 10.0, 10.0, 1.0);
        assert!((normal[1] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_compute_normal_from_heights_sloped() {
        let normal = compute_normal_from_heights(10.0, 8.0, 12.0, 10.0, 10.0, 1.0);
        assert!(normal[0] < 0.0); // Slope in +X direction, normal points -X
        assert!(normal[1] > 0.0); // Still mostly up
    }

    // -----------------------------------------------------------------------
    // Pod/Zeroable Verification Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_terrain_material_config_pod() {
        let config = TerrainMaterialConfig::default();
        let bytes = bytemuck::bytes_of(&config);
        assert_eq!(bytes.len(), std::mem::size_of::<TerrainMaterialConfig>());
    }

    #[test]
    fn test_terrain_layer_def_pod() {
        let layer = TerrainLayerDef::default();
        let bytes = bytemuck::bytes_of(&layer);
        assert_eq!(bytes.len(), std::mem::size_of::<TerrainLayerDef>());
    }

    #[test]
    fn test_splat_pixel_pod() {
        let pixel = SplatPixel::default();
        let bytes = bytemuck::bytes_of(&pixel);
        assert_eq!(bytes.len(), 8);
    }

    #[test]
    fn test_stochastic_params_pod() {
        let params = StochasticParams::default();
        let bytes = bytemuck::bytes_of(&params);
        assert_eq!(bytes.len(), std::mem::size_of::<StochasticParams>());
    }
}
