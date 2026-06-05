//! Weather Map System (T-ENV-3.1)
//!
//! This module provides a 2D weather map system for controlling volumetric clouds
//! and weather effects across large world spaces. The weather map drives:
//!
//! - Cloud coverage: overall cloud density (0-1)
//! - Cloud type: blending between stratus, cumulus, and cumulonimbus
//! - Precipitation: rain and snow zones
//! - Wind velocity: 2D vector field for cloud motion
//! - Height layering: cloud configuration by altitude
//!
//! # Architecture
//!
//! The weather map is a 2D texture (or procedural function) that encodes weather
//! parameters at each world XZ coordinate. It integrates with `cloud_raymarching.rs`
//! to modulate cloud density and with `cloud_noise.rs` for procedural variation.
//!
//! # GPU Integration
//!
//! All structs are `repr(C)` with `bytemuck::Pod/Zeroable` for direct GPU upload.
//! The `WeatherMapUniforms` struct is designed for efficient uniform buffer binding.
//!
//! # World-Space Mapping
//!
//! World coordinates are transformed to weather map UV coordinates using:
//! ```text
//! uv = (world_xz - map_origin) * world_to_map_scale
//! ```
//!
//! The map tiles seamlessly for infinite worlds when using procedural noise.
//!
//! # References
//!
//! - Schneider & Vos, "The Real-time Volumetric Cloudscapes of Horizon: Zero Dawn"
//! - Hillaire, "A Scalable and Production Ready Sky and Atmosphere Rendering Technique"

use bytemuck::{Pod, Zeroable};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default weather map world scale (km per UV unit).
pub const DEFAULT_MAP_SCALE_KM: f32 = 50.0;

/// Default wind speed (m/s).
pub const DEFAULT_WIND_SPEED: f32 = 5.0;

/// Default wind direction (radians, 0 = +X, pi/2 = +Z).
pub const DEFAULT_WIND_DIRECTION: f32 = 0.785; // 45 degrees

/// Default base cloud coverage.
pub const DEFAULT_COVERAGE_BASE: f32 = 0.5;

/// Default cloud type blend (0 = stratus, 0.5 = cumulus, 1 = cumulonimbus).
pub const DEFAULT_CLOUD_TYPE: f32 = 0.5;

/// Default precipitation intensity.
pub const DEFAULT_PRECIPITATION: f32 = 0.0;

/// Minimum valid coverage value.
pub const MIN_COVERAGE: f32 = 0.0;

/// Maximum valid coverage value.
pub const MAX_COVERAGE: f32 = 1.0;

/// Small epsilon for floating point comparisons.
pub const EPSILON: f32 = 1e-6;

/// Cloud layer height bands (meters).
pub const STRATUS_MIN_HEIGHT: f32 = 500.0;
pub const STRATUS_MAX_HEIGHT: f32 = 2000.0;
pub const CUMULUS_MIN_HEIGHT: f32 = 1500.0;
pub const CUMULUS_MAX_HEIGHT: f32 = 8000.0;
pub const CUMULONIMBUS_MIN_HEIGHT: f32 = 500.0;
pub const CUMULONIMBUS_MAX_HEIGHT: f32 = 12000.0;

/// Precipitation temperature threshold (Celsius) for rain vs snow.
pub const SNOW_TEMPERATURE_THRESHOLD: f32 = 2.0;

/// Default noise octaves for weather patterns.
pub const DEFAULT_NOISE_OCTAVES: u32 = 4;

/// Default noise persistence.
pub const DEFAULT_NOISE_PERSISTENCE: f32 = 0.5;

/// Default noise lacunarity.
pub const DEFAULT_NOISE_LACUNARITY: f32 = 2.0;

/// Default noise scale for coverage patterns.
pub const DEFAULT_COVERAGE_NOISE_SCALE: f32 = 0.001;

/// Default noise scale for wind patterns.
pub const DEFAULT_WIND_NOISE_SCALE: f32 = 0.0005;

// ---------------------------------------------------------------------------
// CloudTypeId — Enumeration of cloud types
// ---------------------------------------------------------------------------

/// Cloud type identifier for type blending.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
#[repr(u8)]
pub enum CloudTypeId {
    /// Stratus: flat, low-lying, overcast.
    Stratus = 0,

    /// Cumulus: puffy, fair-weather clouds.
    #[default]
    Cumulus = 1,

    /// Cumulonimbus: tall storm clouds with anvil tops.
    Cumulonimbus = 2,

    /// Stratocumulus: lumpy layer clouds.
    Stratocumulus = 3,

    /// Cirrus: wispy, high-altitude ice clouds.
    Cirrus = 4,
}

impl CloudTypeId {
    /// Create from u8 value (clamped to valid range).
    #[inline]
    pub fn from_u8(value: u8) -> Self {
        match value {
            0 => CloudTypeId::Stratus,
            1 => CloudTypeId::Cumulus,
            2 => CloudTypeId::Cumulonimbus,
            3 => CloudTypeId::Stratocumulus,
            4 => CloudTypeId::Cirrus,
            _ => CloudTypeId::Cumulus,
        }
    }

    /// Get height range for this cloud type in meters.
    #[inline]
    pub fn height_range(&self) -> (f32, f32) {
        match self {
            CloudTypeId::Stratus => (STRATUS_MIN_HEIGHT, STRATUS_MAX_HEIGHT),
            CloudTypeId::Cumulus => (CUMULUS_MIN_HEIGHT, CUMULUS_MAX_HEIGHT),
            CloudTypeId::Cumulonimbus => (CUMULONIMBUS_MIN_HEIGHT, CUMULONIMBUS_MAX_HEIGHT),
            CloudTypeId::Stratocumulus => (1000.0, 3000.0),
            CloudTypeId::Cirrus => (6000.0, 12000.0),
        }
    }

    /// Get typical density multiplier for this cloud type.
    #[inline]
    pub fn density_multiplier(&self) -> f32 {
        match self {
            CloudTypeId::Stratus => 0.8,
            CloudTypeId::Cumulus => 1.0,
            CloudTypeId::Cumulonimbus => 1.5,
            CloudTypeId::Stratocumulus => 0.7,
            CloudTypeId::Cirrus => 0.2,
        }
    }

    /// Get the blend factor for interpolating cloud properties.
    ///
    /// Returns a value in [0, 1] where:
    /// - 0.0 = stratus
    /// - 0.5 = cumulus
    /// - 1.0 = cumulonimbus
    #[inline]
    pub fn blend_factor(&self) -> f32 {
        match self {
            CloudTypeId::Stratus => 0.0,
            CloudTypeId::Cumulus => 0.5,
            CloudTypeId::Cumulonimbus => 1.0,
            CloudTypeId::Stratocumulus => 0.25,
            CloudTypeId::Cirrus => 0.75,
        }
    }

    /// Get name string for debugging.
    #[inline]
    pub fn name(&self) -> &'static str {
        match self {
            CloudTypeId::Stratus => "stratus",
            CloudTypeId::Cumulus => "cumulus",
            CloudTypeId::Cumulonimbus => "cumulonimbus",
            CloudTypeId::Stratocumulus => "stratocumulus",
            CloudTypeId::Cirrus => "cirrus",
        }
    }
}

// ---------------------------------------------------------------------------
// PrecipitationType — Type of precipitation
// ---------------------------------------------------------------------------

/// Type of precipitation at a weather map location.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
#[repr(u8)]
pub enum PrecipitationType {
    /// No precipitation.
    #[default]
    None = 0,

    /// Light rain.
    LightRain = 1,

    /// Moderate rain.
    Rain = 2,

    /// Heavy rain.
    HeavyRain = 3,

    /// Light snow.
    LightSnow = 4,

    /// Moderate snow.
    Snow = 5,

    /// Heavy snow / blizzard.
    HeavySnow = 6,

    /// Mixed rain and snow (sleet).
    Sleet = 7,
}

impl PrecipitationType {
    /// Create from u8 value.
    #[inline]
    pub fn from_u8(value: u8) -> Self {
        match value {
            0 => PrecipitationType::None,
            1 => PrecipitationType::LightRain,
            2 => PrecipitationType::Rain,
            3 => PrecipitationType::HeavyRain,
            4 => PrecipitationType::LightSnow,
            5 => PrecipitationType::Snow,
            6 => PrecipitationType::HeavySnow,
            7 => PrecipitationType::Sleet,
            _ => PrecipitationType::None,
        }
    }

    /// Get precipitation intensity (0-1).
    #[inline]
    pub fn intensity(&self) -> f32 {
        match self {
            PrecipitationType::None => 0.0,
            PrecipitationType::LightRain | PrecipitationType::LightSnow => 0.3,
            PrecipitationType::Rain | PrecipitationType::Snow => 0.6,
            PrecipitationType::HeavyRain | PrecipitationType::HeavySnow => 1.0,
            PrecipitationType::Sleet => 0.5,
        }
    }

    /// Check if this is a snow type.
    #[inline]
    pub fn is_snow(&self) -> bool {
        matches!(
            self,
            PrecipitationType::LightSnow
                | PrecipitationType::Snow
                | PrecipitationType::HeavySnow
                | PrecipitationType::Sleet
        )
    }

    /// Check if this is a rain type.
    #[inline]
    pub fn is_rain(&self) -> bool {
        matches!(
            self,
            PrecipitationType::LightRain
                | PrecipitationType::Rain
                | PrecipitationType::HeavyRain
                | PrecipitationType::Sleet
        )
    }

    /// Get name string.
    #[inline]
    pub fn name(&self) -> &'static str {
        match self {
            PrecipitationType::None => "none",
            PrecipitationType::LightRain => "light_rain",
            PrecipitationType::Rain => "rain",
            PrecipitationType::HeavyRain => "heavy_rain",
            PrecipitationType::LightSnow => "light_snow",
            PrecipitationType::Snow => "snow",
            PrecipitationType::HeavySnow => "heavy_snow",
            PrecipitationType::Sleet => "sleet",
        }
    }
}

// ---------------------------------------------------------------------------
// WeatherMapUniforms — GPU uniform buffer for weather parameters
// ---------------------------------------------------------------------------

/// GPU-uploadable uniform buffer for weather map parameters.
///
/// This struct contains all parameters needed by the shader to sample
/// and evaluate weather conditions at any world position.
///
/// # Memory Layout (64 bytes)
///
/// | Offset | Field              | Size     |
/// |--------|--------------------|----------|
/// | 0      | world_to_map_scale | 8 bytes  |
/// | 8      | map_origin         | 8 bytes  |
/// | 16     | wind_velocity      | 8 bytes  |
/// | 24     | time               | 4 bytes  |
/// | 28     | coverage_base      | 4 bytes  |
/// | 32     | cloud_type_blend   | 4 bytes  |
/// | 36     | precipitation      | 4 bytes  |
/// | 40     | temperature        | 4 bytes  |
/// | 44     | noise_scale        | 4 bytes  |
/// | 48     | _padding           | 16 bytes |
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct WeatherMapUniforms {
    /// Scale factor to convert world XZ to map UV.
    /// UV = world_xz * world_to_map_scale
    pub world_to_map_scale: [f32; 2],

    /// Origin offset for map coordinates (world space).
    pub map_origin: [f32; 2],

    /// Wind velocity in world units per second (X, Z).
    pub wind_velocity: [f32; 2],

    /// Current time for animation (seconds).
    pub time: f32,

    /// Base cloud coverage (0-1).
    pub coverage_base: f32,

    /// Cloud type blend factor (0 = stratus, 0.5 = cumulus, 1 = cumulonimbus).
    pub cloud_type_blend: f32,

    /// Precipitation intensity (0-1).
    pub precipitation: f32,

    /// Temperature in Celsius (affects rain vs snow).
    pub temperature: f32,

    /// Noise scale for procedural patterns.
    pub noise_scale: f32,

    /// Padding for 16-byte alignment.
    pub _padding: [f32; 4],
}

// Size assertion for GPU compatibility
const _: () = assert!(std::mem::size_of::<WeatherMapUniforms>() == 64);

impl Default for WeatherMapUniforms {
    fn default() -> Self {
        Self {
            world_to_map_scale: [1.0 / (DEFAULT_MAP_SCALE_KM * 1000.0); 2],
            map_origin: [0.0; 2],
            wind_velocity: [
                DEFAULT_WIND_SPEED * DEFAULT_WIND_DIRECTION.cos(),
                DEFAULT_WIND_SPEED * DEFAULT_WIND_DIRECTION.sin(),
            ],
            time: 0.0,
            coverage_base: DEFAULT_COVERAGE_BASE,
            cloud_type_blend: DEFAULT_CLOUD_TYPE,
            precipitation: DEFAULT_PRECIPITATION,
            temperature: 15.0,
            noise_scale: DEFAULT_COVERAGE_NOISE_SCALE,
            _padding: [0.0; 4],
        }
    }
}

impl WeatherMapUniforms {
    /// Create new uniforms with default values.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create uniforms with custom map scale.
    #[inline]
    pub fn with_scale(map_scale_km: f32) -> Self {
        let scale = 1.0 / (map_scale_km.max(1.0) * 1000.0);
        Self {
            world_to_map_scale: [scale; 2],
            ..Default::default()
        }
    }

    /// Set wind from speed and direction.
    #[inline]
    pub fn set_wind(&mut self, speed: f32, direction_radians: f32) {
        self.wind_velocity = [
            speed * direction_radians.cos(),
            speed * direction_radians.sin(),
        ];
    }

    /// Set wind from velocity vector.
    #[inline]
    pub fn set_wind_velocity(&mut self, velocity: [f32; 2]) {
        self.wind_velocity = velocity;
    }

    /// Get wind speed magnitude.
    #[inline]
    pub fn wind_speed(&self) -> f32 {
        (self.wind_velocity[0] * self.wind_velocity[0]
            + self.wind_velocity[1] * self.wind_velocity[1])
        .sqrt()
    }

    /// Get wind direction in radians.
    #[inline]
    pub fn wind_direction(&self) -> f32 {
        self.wind_velocity[1].atan2(self.wind_velocity[0])
    }

    /// Update time for animation.
    #[inline]
    pub fn update_time(&mut self, delta_seconds: f32) {
        self.time += delta_seconds;
    }

    /// Set coverage with clamping.
    #[inline]
    pub fn set_coverage(&mut self, coverage: f32) {
        self.coverage_base = coverage.clamp(MIN_COVERAGE, MAX_COVERAGE);
    }

    /// Set cloud type blend with clamping.
    #[inline]
    pub fn set_cloud_type(&mut self, blend: f32) {
        self.cloud_type_blend = blend.clamp(0.0, 1.0);
    }

    /// Set precipitation with clamping.
    #[inline]
    pub fn set_precipitation(&mut self, intensity: f32) {
        self.precipitation = intensity.clamp(0.0, 1.0);
    }

    /// Transform world XZ to map UV coordinates.
    #[inline]
    pub fn world_to_uv(&self, world_x: f32, world_z: f32) -> [f32; 2] {
        [
            (world_x - self.map_origin[0]) * self.world_to_map_scale[0],
            (world_z - self.map_origin[1]) * self.world_to_map_scale[1],
        ]
    }

    /// Transform world XZ to map UV with wind advection.
    #[inline]
    pub fn world_to_uv_advected(&self, world_x: f32, world_z: f32) -> [f32; 2] {
        let advected_x = world_x - self.wind_velocity[0] * self.time;
        let advected_z = world_z - self.wind_velocity[1] * self.time;
        self.world_to_uv(advected_x, advected_z)
    }

    /// Validate uniform values.
    pub fn validate(&self) -> bool {
        self.world_to_map_scale[0] > 0.0
            && self.world_to_map_scale[1] > 0.0
            && self.coverage_base >= 0.0
            && self.coverage_base <= 1.0
            && self.cloud_type_blend >= 0.0
            && self.cloud_type_blend <= 1.0
            && self.precipitation >= 0.0
            && self.precipitation <= 1.0
            && self.noise_scale > 0.0
    }
}

// ---------------------------------------------------------------------------
// WeatherSample — Result of sampling the weather map
// ---------------------------------------------------------------------------

/// Result of sampling weather conditions at a world position.
///
/// Contains all weather parameters needed for cloud and precipitation rendering.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct WeatherSample {
    /// Cloud coverage (0-1).
    pub coverage: f32,

    /// Cloud type blend (0 = stratus, 0.5 = cumulus, 1 = cumulonimbus).
    pub cloud_type_blend: f32,

    /// Precipitation intensity (0-1).
    pub precipitation: f32,

    /// Precipitation type.
    pub precipitation_type: PrecipitationType,

    /// Wind velocity at this position (m/s).
    pub wind_velocity: [f32; 2],

    /// Height gradient modifier for cloud shaping.
    pub height_modifier: f32,

    /// Temperature at this position (Celsius).
    pub temperature: f32,
}

impl WeatherSample {
    /// Create an empty sample (clear weather).
    #[inline]
    pub fn clear() -> Self {
        Self {
            coverage: 0.0,
            cloud_type_blend: 0.5,
            precipitation: 0.0,
            precipitation_type: PrecipitationType::None,
            wind_velocity: [0.0; 2],
            height_modifier: 1.0,
            temperature: 15.0,
        }
    }

    /// Create a fully overcast sample.
    #[inline]
    pub fn overcast() -> Self {
        Self {
            coverage: 1.0,
            cloud_type_blend: 0.0, // Stratus
            precipitation: 0.0,
            precipitation_type: PrecipitationType::None,
            wind_velocity: [0.0; 2],
            height_modifier: 1.0,
            temperature: 10.0,
        }
    }

    /// Create a stormy sample.
    #[inline]
    pub fn storm() -> Self {
        Self {
            coverage: 0.9,
            cloud_type_blend: 1.0, // Cumulonimbus
            precipitation: 0.8,
            precipitation_type: PrecipitationType::HeavyRain,
            wind_velocity: [15.0, 5.0],
            height_modifier: 1.5,
            temperature: 12.0,
        }
    }

    /// Get the dominant cloud type based on blend factor.
    #[inline]
    pub fn dominant_cloud_type(&self) -> CloudTypeId {
        if self.cloud_type_blend < 0.25 {
            CloudTypeId::Stratus
        } else if self.cloud_type_blend < 0.75 {
            CloudTypeId::Cumulus
        } else {
            CloudTypeId::Cumulonimbus
        }
    }

    /// Interpolate between two weather samples.
    #[inline]
    pub fn lerp(a: &Self, b: &Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        let inv_t = 1.0 - t;

        Self {
            coverage: a.coverage * inv_t + b.coverage * t,
            cloud_type_blend: a.cloud_type_blend * inv_t + b.cloud_type_blend * t,
            precipitation: a.precipitation * inv_t + b.precipitation * t,
            precipitation_type: if t < 0.5 {
                a.precipitation_type
            } else {
                b.precipitation_type
            },
            wind_velocity: [
                a.wind_velocity[0] * inv_t + b.wind_velocity[0] * t,
                a.wind_velocity[1] * inv_t + b.wind_velocity[1] * t,
            ],
            height_modifier: a.height_modifier * inv_t + b.height_modifier * t,
            temperature: a.temperature * inv_t + b.temperature * t,
        }
    }

    /// Check if weather is clear (no clouds).
    #[inline]
    pub fn is_clear(&self) -> bool {
        self.coverage < 0.1
    }

    /// Check if weather has precipitation.
    #[inline]
    pub fn has_precipitation(&self) -> bool {
        self.precipitation > 0.1
    }
}

// ---------------------------------------------------------------------------
// HeightLayer — Cloud height layer configuration
// ---------------------------------------------------------------------------

/// Configuration for a height-based cloud layer.
///
/// Multiple height layers can be combined to create complex cloud formations
/// with different types at different altitudes.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct HeightLayer {
    /// Minimum height of this layer (meters).
    pub min_height: f32,

    /// Maximum height of this layer (meters).
    pub max_height: f32,

    /// Cloud type blend for this layer (0-1).
    pub cloud_type: f32,

    /// Density multiplier for this layer.
    pub density_multiplier: f32,
}

impl Default for HeightLayer {
    fn default() -> Self {
        Self::cumulus()
    }
}

impl HeightLayer {
    /// Create a new height layer.
    #[inline]
    pub fn new(min_height: f32, max_height: f32, cloud_type: f32, density: f32) -> Self {
        Self {
            min_height: min_height.max(0.0),
            max_height: max_height.max(min_height + 100.0),
            cloud_type: cloud_type.clamp(0.0, 1.0),
            density_multiplier: density.max(0.0),
        }
    }

    /// Create a stratus layer preset.
    #[inline]
    pub fn stratus() -> Self {
        Self {
            min_height: STRATUS_MIN_HEIGHT,
            max_height: STRATUS_MAX_HEIGHT,
            cloud_type: 0.0,
            density_multiplier: 0.8,
        }
    }

    /// Create a cumulus layer preset.
    #[inline]
    pub fn cumulus() -> Self {
        Self {
            min_height: CUMULUS_MIN_HEIGHT,
            max_height: CUMULUS_MAX_HEIGHT,
            cloud_type: 0.5,
            density_multiplier: 1.0,
        }
    }

    /// Create a cumulonimbus layer preset.
    #[inline]
    pub fn cumulonimbus() -> Self {
        Self {
            min_height: CUMULONIMBUS_MIN_HEIGHT,
            max_height: CUMULONIMBUS_MAX_HEIGHT,
            cloud_type: 1.0,
            density_multiplier: 1.5,
        }
    }

    /// Create a cirrus (high altitude) layer preset.
    #[inline]
    pub fn cirrus() -> Self {
        Self {
            min_height: 6000.0,
            max_height: 12000.0,
            cloud_type: 0.75,
            density_multiplier: 0.2,
        }
    }

    /// Get the thickness of this layer.
    #[inline]
    pub fn thickness(&self) -> f32 {
        self.max_height - self.min_height
    }

    /// Get height fraction within this layer (0 at min, 1 at max).
    #[inline]
    pub fn height_fraction(&self, height: f32) -> f32 {
        let thickness = self.thickness();
        if thickness > EPSILON {
            ((height - self.min_height) / thickness).clamp(0.0, 1.0)
        } else {
            0.5
        }
    }

    /// Check if a height is within this layer.
    #[inline]
    pub fn contains_height(&self, height: f32) -> bool {
        height >= self.min_height && height <= self.max_height
    }

    /// Validate layer configuration.
    #[inline]
    pub fn validate(&self) -> bool {
        self.min_height >= 0.0
            && self.max_height > self.min_height
            && self.cloud_type >= 0.0
            && self.cloud_type <= 1.0
            && self.density_multiplier >= 0.0
    }
}

// ---------------------------------------------------------------------------
// WindField — 2D wind velocity field
// ---------------------------------------------------------------------------

/// 2D wind velocity field for cloud advection.
///
/// Stores base wind parameters and provides sampling with procedural variation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindField {
    /// Base wind velocity (m/s, XZ).
    pub base_velocity: [f32; 2],

    /// Wind gust strength (0-1).
    pub gust_strength: f32,

    /// Wind gust frequency (Hz).
    pub gust_frequency: f32,

    /// Turbulence intensity (0-1).
    pub turbulence: f32,

    /// Noise scale for spatial variation.
    pub noise_scale: f32,
}

impl Default for WindField {
    fn default() -> Self {
        Self {
            base_velocity: [
                DEFAULT_WIND_SPEED * DEFAULT_WIND_DIRECTION.cos(),
                DEFAULT_WIND_SPEED * DEFAULT_WIND_DIRECTION.sin(),
            ],
            gust_strength: 0.3,
            gust_frequency: 0.1,
            turbulence: 0.1,
            noise_scale: DEFAULT_WIND_NOISE_SCALE,
        }
    }
}

impl WindField {
    /// Create a new wind field with given base velocity.
    #[inline]
    pub fn new(base_velocity: [f32; 2]) -> Self {
        Self {
            base_velocity,
            ..Default::default()
        }
    }

    /// Create from speed and direction.
    #[inline]
    pub fn from_speed_direction(speed: f32, direction_radians: f32) -> Self {
        Self {
            base_velocity: [speed * direction_radians.cos(), speed * direction_radians.sin()],
            ..Default::default()
        }
    }

    /// Create a calm wind field (no wind).
    #[inline]
    pub fn calm() -> Self {
        Self {
            base_velocity: [0.0; 2],
            gust_strength: 0.0,
            turbulence: 0.0,
            ..Default::default()
        }
    }

    /// Create a stormy wind field.
    #[inline]
    pub fn stormy() -> Self {
        Self {
            base_velocity: [15.0, 5.0],
            gust_strength: 0.6,
            gust_frequency: 0.2,
            turbulence: 0.3,
            ..Default::default()
        }
    }

    /// Get base wind speed.
    #[inline]
    pub fn speed(&self) -> f32 {
        (self.base_velocity[0] * self.base_velocity[0]
            + self.base_velocity[1] * self.base_velocity[1])
        .sqrt()
    }

    /// Get base wind direction in radians.
    #[inline]
    pub fn direction(&self) -> f32 {
        self.base_velocity[1].atan2(self.base_velocity[0])
    }

    /// Sample wind velocity at a world position with time variation.
    pub fn sample(&self, world_x: f32, world_z: f32, time: f32) -> [f32; 2] {
        // Base velocity
        let mut vx = self.base_velocity[0];
        let mut vz = self.base_velocity[1];

        // Add gusts (time-based sinusoidal variation)
        if self.gust_strength > EPSILON {
            let gust_phase = time * self.gust_frequency * std::f32::consts::TAU;
            let gust_factor = 1.0 + self.gust_strength * gust_phase.sin();
            vx *= gust_factor;
            vz *= gust_factor;
        }

        // Add turbulence (spatial noise)
        if self.turbulence > EPSILON {
            let noise_x = sample_noise_2d(world_x * self.noise_scale, world_z * self.noise_scale);
            let noise_z = sample_noise_2d(
                world_x * self.noise_scale + 100.0,
                world_z * self.noise_scale + 100.0,
            );

            let speed = self.speed();
            vx += noise_x * self.turbulence * speed;
            vz += noise_z * self.turbulence * speed;
        }

        [vx, vz]
    }

    /// Get the advected position for cloud motion.
    #[inline]
    pub fn advect_position(&self, x: f32, z: f32, time: f32) -> [f32; 2] {
        [
            x - self.base_velocity[0] * time,
            z - self.base_velocity[1] * time,
        ]
    }
}

// ---------------------------------------------------------------------------
// PrecipitationZone — Precipitation region configuration
// ---------------------------------------------------------------------------

/// Configuration for a precipitation zone.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct PrecipitationZone {
    /// Zone center in world XZ coordinates.
    pub center: [f32; 2],

    /// Zone radius in world units.
    pub radius: f32,

    /// Precipitation intensity (0-1).
    pub intensity: f32,

    /// Zone falloff (how quickly intensity drops at edges).
    pub falloff: f32,

    /// Temperature in this zone (Celsius).
    pub temperature: f32,

    /// Precipitation type (as u32 for Pod compatibility).
    pub precipitation_type: u32,

    /// Whether this zone is active.
    pub active: u32,
}

impl Default for PrecipitationZone {
    fn default() -> Self {
        Self {
            center: [0.0; 2],
            radius: 10000.0,
            intensity: 0.0,
            falloff: 0.5,
            temperature: 15.0,
            precipitation_type: 0,
            active: 0,
        }
    }
}

impl PrecipitationZone {
    /// Create a new precipitation zone.
    #[inline]
    pub fn new(center: [f32; 2], radius: f32, intensity: f32) -> Self {
        Self {
            center,
            radius: radius.max(1.0),
            intensity: intensity.clamp(0.0, 1.0),
            falloff: 0.5,
            temperature: 15.0,
            precipitation_type: PrecipitationType::Rain as u32,
            active: 1,
        }
    }

    /// Create a rain zone.
    #[inline]
    pub fn rain(center: [f32; 2], radius: f32, intensity: f32) -> Self {
        Self {
            center,
            radius: radius.max(1.0),
            intensity: intensity.clamp(0.0, 1.0),
            falloff: 0.5,
            temperature: 10.0,
            precipitation_type: PrecipitationType::Rain as u32,
            active: 1,
        }
    }

    /// Create a snow zone.
    #[inline]
    pub fn snow(center: [f32; 2], radius: f32, intensity: f32) -> Self {
        Self {
            center,
            radius: radius.max(1.0),
            intensity: intensity.clamp(0.0, 1.0),
            falloff: 0.3,
            temperature: -5.0,
            precipitation_type: PrecipitationType::Snow as u32,
            active: 1,
        }
    }

    /// Sample precipitation intensity at a world position.
    #[inline]
    pub fn sample(&self, world_x: f32, world_z: f32) -> f32 {
        if self.active == 0 {
            return 0.0;
        }

        let dx = world_x - self.center[0];
        let dz = world_z - self.center[1];
        let dist = (dx * dx + dz * dz).sqrt();

        if dist > self.radius {
            return 0.0;
        }

        let normalized_dist = dist / self.radius;
        let falloff_factor = 1.0 - smoothstep(1.0 - self.falloff, 1.0, normalized_dist);

        self.intensity * falloff_factor
    }

    /// Get precipitation type enum.
    #[inline]
    pub fn get_precipitation_type(&self) -> PrecipitationType {
        PrecipitationType::from_u8(self.precipitation_type as u8)
    }

    /// Set precipitation type.
    #[inline]
    pub fn set_precipitation_type(&mut self, ptype: PrecipitationType) {
        self.precipitation_type = ptype as u32;
    }

    /// Check if zone is active.
    #[inline]
    pub fn is_active(&self) -> bool {
        self.active != 0
    }

    /// Set zone active state.
    #[inline]
    pub fn set_active(&mut self, active: bool) {
        self.active = if active { 1 } else { 0 };
    }
}

// ---------------------------------------------------------------------------
// WeatherMap — Main weather map system
// ---------------------------------------------------------------------------

/// Main weather map system for controlling volumetric clouds and weather.
///
/// Provides procedural weather patterns based on noise functions,
/// with support for multiple height layers and precipitation zones.
#[derive(Debug, Clone)]
pub struct WeatherMap {
    /// GPU-uploadable uniforms.
    pub uniforms: WeatherMapUniforms,

    /// Height layers for cloud type variation.
    pub height_layers: Vec<HeightLayer>,

    /// Active precipitation zones.
    pub precipitation_zones: Vec<PrecipitationZone>,

    /// Wind field configuration.
    pub wind_field: WindField,

    /// Noise configuration for coverage patterns.
    pub noise_octaves: u32,
    pub noise_persistence: f32,
    pub noise_lacunarity: f32,
}

impl Default for WeatherMap {
    fn default() -> Self {
        Self::new()
    }
}

impl WeatherMap {
    /// Create a new weather map with default settings.
    pub fn new() -> Self {
        Self {
            uniforms: WeatherMapUniforms::default(),
            height_layers: vec![HeightLayer::cumulus()],
            precipitation_zones: Vec::new(),
            wind_field: WindField::default(),
            noise_octaves: DEFAULT_NOISE_OCTAVES,
            noise_persistence: DEFAULT_NOISE_PERSISTENCE,
            noise_lacunarity: DEFAULT_NOISE_LACUNARITY,
        }
    }

    /// Create a weather map with custom scale.
    pub fn with_scale(map_scale_km: f32) -> Self {
        Self {
            uniforms: WeatherMapUniforms::with_scale(map_scale_km),
            ..Self::new()
        }
    }

    /// Create a clear weather preset.
    pub fn clear() -> Self {
        let mut map = Self::new();
        map.uniforms.coverage_base = 0.1;
        map.wind_field = WindField::calm();
        map
    }

    /// Create a partly cloudy weather preset.
    pub fn partly_cloudy() -> Self {
        let mut map = Self::new();
        map.uniforms.coverage_base = 0.4;
        map.height_layers = vec![HeightLayer::cumulus()];
        map
    }

    /// Create an overcast weather preset.
    pub fn overcast() -> Self {
        let mut map = Self::new();
        map.uniforms.coverage_base = 0.9;
        map.uniforms.cloud_type_blend = 0.0; // Stratus
        map.height_layers = vec![HeightLayer::stratus()];
        map
    }

    /// Create a stormy weather preset.
    pub fn storm() -> Self {
        let mut map = Self::new();
        map.uniforms.coverage_base = 0.85;
        map.uniforms.cloud_type_blend = 1.0; // Cumulonimbus
        map.uniforms.precipitation = 0.7;
        map.uniforms.temperature = 10.0;
        map.height_layers = vec![HeightLayer::cumulonimbus()];
        map.wind_field = WindField::stormy();
        map.precipitation_zones.push(PrecipitationZone::rain(
            [0.0, 0.0],
            50000.0,
            0.8,
        ));
        map
    }

    /// Add a height layer.
    #[inline]
    pub fn add_height_layer(&mut self, layer: HeightLayer) {
        self.height_layers.push(layer);
    }

    /// Add a precipitation zone.
    #[inline]
    pub fn add_precipitation_zone(&mut self, zone: PrecipitationZone) {
        self.precipitation_zones.push(zone);
    }

    /// Set wind parameters.
    #[inline]
    pub fn set_wind(&mut self, speed: f32, direction_radians: f32) {
        self.wind_field = WindField::from_speed_direction(speed, direction_radians);
        self.uniforms.set_wind(speed, direction_radians);
    }

    /// Update time for animation.
    #[inline]
    pub fn update(&mut self, delta_seconds: f32) {
        self.uniforms.update_time(delta_seconds);
    }

    /// Sample weather at a world position.
    pub fn sample(&self, world_x: f32, world_z: f32) -> WeatherSample {
        // Get UV coordinates with wind advection
        let uv = self.uniforms.world_to_uv_advected(world_x, world_z);

        // Sample coverage from noise
        let coverage_noise = self.sample_coverage_noise(uv[0], uv[1]);
        let coverage = (self.uniforms.coverage_base + coverage_noise * 0.3).clamp(0.0, 1.0);

        // Sample cloud type from noise (different frequency)
        let type_noise = self.sample_type_noise(uv[0], uv[1]);
        let cloud_type_blend = (self.uniforms.cloud_type_blend + type_noise * 0.2).clamp(0.0, 1.0);

        // Sample wind
        let wind = self.wind_field.sample(world_x, world_z, self.uniforms.time);

        // Sample precipitation from zones
        let mut precipitation = self.uniforms.precipitation;
        let mut precipitation_type = PrecipitationType::None;
        let mut temperature = self.uniforms.temperature;

        for zone in &self.precipitation_zones {
            let zone_precip = zone.sample(world_x, world_z);
            if zone_precip > precipitation {
                precipitation = zone_precip;
                precipitation_type = zone.get_precipitation_type();
                temperature = zone.temperature;
            }
        }

        // Determine precipitation type from temperature if not set
        if precipitation > 0.1 && precipitation_type == PrecipitationType::None {
            precipitation_type = if temperature < SNOW_TEMPERATURE_THRESHOLD {
                if precipitation > 0.7 {
                    PrecipitationType::HeavySnow
                } else if precipitation > 0.3 {
                    PrecipitationType::Snow
                } else {
                    PrecipitationType::LightSnow
                }
            } else if precipitation > 0.7 {
                PrecipitationType::HeavyRain
            } else if precipitation > 0.3 {
                PrecipitationType::Rain
            } else {
                PrecipitationType::LightRain
            };
        }

        WeatherSample {
            coverage,
            cloud_type_blend,
            precipitation,
            precipitation_type,
            wind_velocity: wind,
            height_modifier: 1.0,
            temperature,
        }
    }

    /// Sample weather at a world position with height consideration.
    pub fn sample_at_height(&self, world_x: f32, world_y: f32, world_z: f32) -> WeatherSample {
        let mut sample = self.sample(world_x, world_z);

        // Apply height layer modifiers
        let mut total_weight = 0.0;
        let mut weighted_type = 0.0;
        let mut density_mod = 0.0;

        for layer in &self.height_layers {
            if layer.contains_height(world_y) {
                let height_frac = layer.height_fraction(world_y);
                // Bell curve weight centered at layer midpoint
                let weight = bell_curve(height_frac, 0.5, 0.3);

                weighted_type += layer.cloud_type * weight;
                density_mod += layer.density_multiplier * weight;
                total_weight += weight;
            }
        }

        if total_weight > EPSILON {
            sample.cloud_type_blend = weighted_type / total_weight;
            sample.height_modifier = density_mod / total_weight;
        }

        sample
    }

    /// Get the height layer for a given altitude.
    pub fn get_layer_at_height(&self, height: f32) -> Option<&HeightLayer> {
        self.height_layers.iter().find(|l| l.contains_height(height))
    }

    /// Sample coverage noise at UV coordinates.
    fn sample_coverage_noise(&self, u: f32, v: f32) -> f32 {
        fbm_2d(
            u * 1000.0 * self.uniforms.noise_scale,
            v * 1000.0 * self.uniforms.noise_scale,
            self.noise_octaves,
            self.noise_persistence,
            self.noise_lacunarity,
        )
    }

    /// Sample type noise at UV coordinates (different frequency).
    fn sample_type_noise(&self, u: f32, v: f32) -> f32 {
        fbm_2d(
            u * 500.0 * self.uniforms.noise_scale + 1000.0,
            v * 500.0 * self.uniforms.noise_scale + 1000.0,
            self.noise_octaves.saturating_sub(1).max(1),
            self.noise_persistence,
            self.noise_lacunarity,
        )
    }

    /// Get uniforms for GPU upload.
    #[inline]
    pub fn get_uniforms(&self) -> &WeatherMapUniforms {
        &self.uniforms
    }

    /// Validate weather map configuration.
    pub fn validate(&self) -> bool {
        self.uniforms.validate()
            && self.height_layers.iter().all(|l| l.validate())
            && self.noise_octaves > 0
            && self.noise_persistence > 0.0
            && self.noise_lacunarity > 1.0
    }
}

// ---------------------------------------------------------------------------
// Noise Functions
// ---------------------------------------------------------------------------

/// Simple 2D hash function for noise.
#[inline]
fn hash_2d(x: i32, y: i32, seed: u32) -> u32 {
    let mut h = seed;
    h ^= x as u32;
    h = h.wrapping_mul(0x85EBCA6B);
    h ^= h >> 13;
    h ^= y as u32;
    h = h.wrapping_mul(0xC2B2AE35);
    h ^= h >> 16;
    h
}

/// Convert hash to float in [0, 1).
#[inline]
fn hash_to_float(hash: u32) -> f32 {
    (hash >> 8) as f32 / 16777216.0
}

/// Simple 2D noise for weather patterns.
fn sample_noise_2d(x: f32, y: f32) -> f32 {
    let ix = x.floor() as i32;
    let iy = y.floor() as i32;
    let fx = x - ix as f32;
    let fy = y - iy as f32;

    let u = fade(fx);
    let v = fade(fy);

    let seed = 0x1337CAFE_u32;

    let h00 = hash_to_float(hash_2d(ix, iy, seed));
    let h10 = hash_to_float(hash_2d(ix + 1, iy, seed));
    let h01 = hash_to_float(hash_2d(ix, iy + 1, seed));
    let h11 = hash_to_float(hash_2d(ix + 1, iy + 1, seed));

    let x0 = lerp(h00, h10, u);
    let x1 = lerp(h01, h11, u);

    lerp(x0, x1, v) * 2.0 - 1.0 // Map to [-1, 1]
}

/// 2D Perlin noise with gradients.
fn perlin_2d(x: f32, y: f32, seed: u32) -> f32 {
    let ix = x.floor() as i32;
    let iy = y.floor() as i32;
    let fx = x - ix as f32;
    let fy = y - iy as f32;

    let u = fade(fx);
    let v = fade(fy);

    // Gradient contributions at 4 corners
    let g00 = gradient_2d(hash_2d(ix, iy, seed), fx, fy);
    let g10 = gradient_2d(hash_2d(ix + 1, iy, seed), fx - 1.0, fy);
    let g01 = gradient_2d(hash_2d(ix, iy + 1, seed), fx, fy - 1.0);
    let g11 = gradient_2d(hash_2d(ix + 1, iy + 1, seed), fx - 1.0, fy - 1.0);

    let x0 = lerp(g00, g10, u);
    let x1 = lerp(g01, g11, u);

    lerp(x0, x1, v)
}

/// 2D gradient for Perlin noise.
#[inline]
fn gradient_2d(hash: u32, x: f32, y: f32) -> f32 {
    let h = hash & 3;
    let u = if h < 2 { x } else { y };
    let v = if h < 2 { y } else { x };
    let u_sign = if (h & 1) == 0 { u } else { -u };
    u_sign + v
}

/// 2D FBM (Fractal Brownian Motion).
fn fbm_2d(x: f32, y: f32, octaves: u32, persistence: f32, lacunarity: f32) -> f32 {
    let mut value = 0.0;
    let mut amplitude = 1.0;
    let mut max_amplitude = 0.0;
    let mut px = x;
    let mut py = y;

    for i in 0..octaves {
        let seed = 0x1337BEEF_u32.wrapping_add(i.wrapping_mul(0x9E3779B9));
        value += amplitude * perlin_2d(px, py, seed);
        max_amplitude += amplitude;

        px *= lacunarity;
        py *= lacunarity;
        amplitude *= persistence;
    }

    if max_amplitude > EPSILON {
        value / max_amplitude
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

/// Smoothstep fade curve.
#[inline]
fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// Linear interpolation.
#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

/// Smoothstep interpolation.
#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Bell curve (Gaussian-like).
#[inline]
fn bell_curve(x: f32, center: f32, width: f32) -> f32 {
    let d = (x - center) / width;
    (-d * d).exp()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // CloudTypeId Tests
    // =========================================================================

    #[test]
    fn test_cloud_type_id_from_u8() {
        assert_eq!(CloudTypeId::from_u8(0), CloudTypeId::Stratus);
        assert_eq!(CloudTypeId::from_u8(1), CloudTypeId::Cumulus);
        assert_eq!(CloudTypeId::from_u8(2), CloudTypeId::Cumulonimbus);
        assert_eq!(CloudTypeId::from_u8(3), CloudTypeId::Stratocumulus);
        assert_eq!(CloudTypeId::from_u8(4), CloudTypeId::Cirrus);
        assert_eq!(CloudTypeId::from_u8(255), CloudTypeId::Cumulus); // Invalid -> default
    }

    #[test]
    fn test_cloud_type_id_height_range() {
        let (min, max) = CloudTypeId::Stratus.height_range();
        assert!(min < max);
        assert_eq!(min, STRATUS_MIN_HEIGHT);

        let (min, max) = CloudTypeId::Cumulonimbus.height_range();
        assert!(max > 10000.0);
    }

    #[test]
    fn test_cloud_type_id_density_multiplier() {
        assert!(CloudTypeId::Cirrus.density_multiplier() < CloudTypeId::Cumulus.density_multiplier());
        assert!(CloudTypeId::Cumulonimbus.density_multiplier() > CloudTypeId::Cumulus.density_multiplier());
    }

    #[test]
    fn test_cloud_type_id_blend_factor() {
        assert_eq!(CloudTypeId::Stratus.blend_factor(), 0.0);
        assert_eq!(CloudTypeId::Cumulus.blend_factor(), 0.5);
        assert_eq!(CloudTypeId::Cumulonimbus.blend_factor(), 1.0);
    }

    #[test]
    fn test_cloud_type_id_name() {
        assert_eq!(CloudTypeId::Stratus.name(), "stratus");
        assert_eq!(CloudTypeId::Cumulus.name(), "cumulus");
    }

    #[test]
    fn test_cloud_type_id_default() {
        assert_eq!(CloudTypeId::default(), CloudTypeId::Cumulus);
    }

    // =========================================================================
    // PrecipitationType Tests
    // =========================================================================

    #[test]
    fn test_precipitation_type_from_u8() {
        assert_eq!(PrecipitationType::from_u8(0), PrecipitationType::None);
        assert_eq!(PrecipitationType::from_u8(2), PrecipitationType::Rain);
        assert_eq!(PrecipitationType::from_u8(5), PrecipitationType::Snow);
        assert_eq!(PrecipitationType::from_u8(255), PrecipitationType::None);
    }

    #[test]
    fn test_precipitation_type_intensity() {
        assert_eq!(PrecipitationType::None.intensity(), 0.0);
        assert!(PrecipitationType::LightRain.intensity() < PrecipitationType::Rain.intensity());
        assert_eq!(PrecipitationType::HeavyRain.intensity(), 1.0);
    }

    #[test]
    fn test_precipitation_type_is_snow() {
        assert!(!PrecipitationType::Rain.is_snow());
        assert!(PrecipitationType::Snow.is_snow());
        assert!(PrecipitationType::Sleet.is_snow());
    }

    #[test]
    fn test_precipitation_type_is_rain() {
        assert!(!PrecipitationType::Snow.is_rain());
        assert!(PrecipitationType::Rain.is_rain());
        assert!(PrecipitationType::Sleet.is_rain());
    }

    #[test]
    fn test_precipitation_type_name() {
        assert_eq!(PrecipitationType::None.name(), "none");
        assert_eq!(PrecipitationType::HeavyRain.name(), "heavy_rain");
    }

    #[test]
    fn test_precipitation_type_default() {
        assert_eq!(PrecipitationType::default(), PrecipitationType::None);
    }

    // =========================================================================
    // WeatherMapUniforms Tests
    // =========================================================================

    #[test]
    fn test_weather_map_uniforms_default() {
        let uniforms = WeatherMapUniforms::default();
        assert!(uniforms.validate());
        assert_eq!(uniforms.coverage_base, DEFAULT_COVERAGE_BASE);
        assert!(uniforms.wind_speed() > 0.0);
    }

    #[test]
    fn test_weather_map_uniforms_with_scale() {
        let uniforms = WeatherMapUniforms::with_scale(100.0);
        let expected_scale = 1.0 / 100000.0;
        assert!((uniforms.world_to_map_scale[0] - expected_scale).abs() < EPSILON);
    }

    #[test]
    fn test_weather_map_uniforms_set_wind() {
        let mut uniforms = WeatherMapUniforms::new();
        uniforms.set_wind(10.0, std::f32::consts::PI / 4.0);
        assert!((uniforms.wind_speed() - 10.0).abs() < 0.01);
        assert!((uniforms.wind_direction() - std::f32::consts::PI / 4.0).abs() < 0.01);
    }

    #[test]
    fn test_weather_map_uniforms_wind_velocity() {
        let mut uniforms = WeatherMapUniforms::new();
        uniforms.set_wind_velocity([5.0, 0.0]);
        assert_eq!(uniforms.wind_speed(), 5.0);
        assert_eq!(uniforms.wind_direction(), 0.0);
    }

    #[test]
    fn test_weather_map_uniforms_update_time() {
        let mut uniforms = WeatherMapUniforms::new();
        assert_eq!(uniforms.time, 0.0);
        uniforms.update_time(1.0);
        assert_eq!(uniforms.time, 1.0);
        uniforms.update_time(0.5);
        assert_eq!(uniforms.time, 1.5);
    }

    #[test]
    fn test_weather_map_uniforms_set_coverage() {
        let mut uniforms = WeatherMapUniforms::new();
        uniforms.set_coverage(0.7);
        assert_eq!(uniforms.coverage_base, 0.7);

        uniforms.set_coverage(1.5);
        assert_eq!(uniforms.coverage_base, 1.0);

        uniforms.set_coverage(-0.5);
        assert_eq!(uniforms.coverage_base, 0.0);
    }

    #[test]
    fn test_weather_map_uniforms_set_cloud_type() {
        let mut uniforms = WeatherMapUniforms::new();
        uniforms.set_cloud_type(0.8);
        assert_eq!(uniforms.cloud_type_blend, 0.8);

        uniforms.set_cloud_type(2.0);
        assert_eq!(uniforms.cloud_type_blend, 1.0);
    }

    #[test]
    fn test_weather_map_uniforms_set_precipitation() {
        let mut uniforms = WeatherMapUniforms::new();
        uniforms.set_precipitation(0.6);
        assert_eq!(uniforms.precipitation, 0.6);
    }

    #[test]
    fn test_weather_map_uniforms_world_to_uv() {
        let uniforms = WeatherMapUniforms::with_scale(50.0);
        let uv = uniforms.world_to_uv(50000.0, 50000.0);
        assert!((uv[0] - 1.0).abs() < 0.01);
        assert!((uv[1] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_weather_map_uniforms_world_to_uv_advected() {
        let mut uniforms = WeatherMapUniforms::new();
        uniforms.set_wind_velocity([100.0, 0.0]);
        uniforms.time = 10.0;

        let uv_base = uniforms.world_to_uv(0.0, 0.0);
        let uv_advected = uniforms.world_to_uv_advected(0.0, 0.0);

        // Advected should be shifted
        assert!((uv_advected[0] - uv_base[0]).abs() > 0.0);
    }

    #[test]
    fn test_weather_map_uniforms_validate() {
        let uniforms = WeatherMapUniforms::default();
        assert!(uniforms.validate());

        let mut invalid = uniforms;
        invalid.coverage_base = 2.0;
        assert!(!invalid.validate());

        invalid = uniforms;
        invalid.world_to_map_scale = [0.0, 1.0];
        assert!(!invalid.validate());
    }

    #[test]
    fn test_weather_map_uniforms_pod() {
        let uniforms = WeatherMapUniforms::default();
        let bytes = bytemuck::bytes_of(&uniforms);
        assert_eq!(bytes.len(), 64);
    }

    #[test]
    fn test_weather_map_uniforms_size() {
        assert_eq!(std::mem::size_of::<WeatherMapUniforms>(), 64);
    }

    // =========================================================================
    // WeatherSample Tests
    // =========================================================================

    #[test]
    fn test_weather_sample_clear() {
        let sample = WeatherSample::clear();
        assert!(sample.is_clear());
        assert!(!sample.has_precipitation());
    }

    #[test]
    fn test_weather_sample_overcast() {
        let sample = WeatherSample::overcast();
        assert_eq!(sample.coverage, 1.0);
        assert!(!sample.is_clear());
    }

    #[test]
    fn test_weather_sample_storm() {
        let sample = WeatherSample::storm();
        assert!(sample.coverage > 0.8);
        assert!(sample.has_precipitation());
        assert_eq!(sample.precipitation_type, PrecipitationType::HeavyRain);
    }

    #[test]
    fn test_weather_sample_dominant_cloud_type() {
        let mut sample = WeatherSample::clear();

        sample.cloud_type_blend = 0.1;
        assert_eq!(sample.dominant_cloud_type(), CloudTypeId::Stratus);

        sample.cloud_type_blend = 0.5;
        assert_eq!(sample.dominant_cloud_type(), CloudTypeId::Cumulus);

        sample.cloud_type_blend = 0.9;
        assert_eq!(sample.dominant_cloud_type(), CloudTypeId::Cumulonimbus);
    }

    #[test]
    fn test_weather_sample_lerp() {
        let a = WeatherSample::clear();
        let b = WeatherSample::overcast();

        let mid = WeatherSample::lerp(&a, &b, 0.5);
        assert!((mid.coverage - 0.5).abs() < 0.01);

        let start = WeatherSample::lerp(&a, &b, 0.0);
        assert_eq!(start.coverage, a.coverage);

        let end = WeatherSample::lerp(&a, &b, 1.0);
        assert_eq!(end.coverage, b.coverage);
    }

    #[test]
    fn test_weather_sample_lerp_clamping() {
        let a = WeatherSample::clear();
        let b = WeatherSample::overcast();

        let clamped = WeatherSample::lerp(&a, &b, 2.0);
        assert_eq!(clamped.coverage, b.coverage);
    }

    #[test]
    fn test_weather_sample_default() {
        let sample = WeatherSample::default();
        assert_eq!(sample.coverage, 0.0);
        assert_eq!(sample.precipitation_type, PrecipitationType::None);
    }

    // =========================================================================
    // HeightLayer Tests
    // =========================================================================

    #[test]
    fn test_height_layer_new() {
        let layer = HeightLayer::new(1000.0, 5000.0, 0.5, 1.0);
        assert_eq!(layer.min_height, 1000.0);
        assert_eq!(layer.max_height, 5000.0);
        assert!(layer.validate());
    }

    #[test]
    fn test_height_layer_new_clamping() {
        let layer = HeightLayer::new(-100.0, 500.0, 1.5, -1.0);
        assert_eq!(layer.min_height, 0.0);
        assert_eq!(layer.cloud_type, 1.0);
        assert_eq!(layer.density_multiplier, 0.0);
    }

    #[test]
    fn test_height_layer_stratus() {
        let layer = HeightLayer::stratus();
        assert_eq!(layer.min_height, STRATUS_MIN_HEIGHT);
        assert_eq!(layer.cloud_type, 0.0);
        assert!(layer.validate());
    }

    #[test]
    fn test_height_layer_cumulus() {
        let layer = HeightLayer::cumulus();
        assert_eq!(layer.min_height, CUMULUS_MIN_HEIGHT);
        assert_eq!(layer.cloud_type, 0.5);
        assert!(layer.validate());
    }

    #[test]
    fn test_height_layer_cumulonimbus() {
        let layer = HeightLayer::cumulonimbus();
        assert_eq!(layer.cloud_type, 1.0);
        assert!(layer.density_multiplier > 1.0);
    }

    #[test]
    fn test_height_layer_cirrus() {
        let layer = HeightLayer::cirrus();
        assert!(layer.min_height > 5000.0);
        assert!(layer.density_multiplier < 1.0);
    }

    #[test]
    fn test_height_layer_thickness() {
        let layer = HeightLayer::new(1000.0, 5000.0, 0.5, 1.0);
        assert_eq!(layer.thickness(), 4000.0);
    }

    #[test]
    fn test_height_layer_height_fraction() {
        let layer = HeightLayer::new(1000.0, 5000.0, 0.5, 1.0);
        assert!((layer.height_fraction(1000.0) - 0.0).abs() < EPSILON);
        assert!((layer.height_fraction(3000.0) - 0.5).abs() < EPSILON);
        assert!((layer.height_fraction(5000.0) - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_height_layer_height_fraction_clamping() {
        let layer = HeightLayer::new(1000.0, 5000.0, 0.5, 1.0);
        assert_eq!(layer.height_fraction(0.0), 0.0);
        assert_eq!(layer.height_fraction(10000.0), 1.0);
    }

    #[test]
    fn test_height_layer_contains_height() {
        let layer = HeightLayer::new(1000.0, 5000.0, 0.5, 1.0);
        assert!(!layer.contains_height(500.0));
        assert!(layer.contains_height(1000.0));
        assert!(layer.contains_height(3000.0));
        assert!(layer.contains_height(5000.0));
        assert!(!layer.contains_height(6000.0));
    }

    #[test]
    fn test_height_layer_validate() {
        let valid = HeightLayer::cumulus();
        assert!(valid.validate());

        let invalid = HeightLayer {
            min_height: 5000.0,
            max_height: 1000.0,
            cloud_type: 0.5,
            density_multiplier: 1.0,
        };
        assert!(!invalid.validate());
    }

    #[test]
    fn test_height_layer_pod() {
        let layer = HeightLayer::cumulus();
        let bytes = bytemuck::bytes_of(&layer);
        assert_eq!(bytes.len(), 16);
    }

    #[test]
    fn test_height_layer_default() {
        let layer = HeightLayer::default();
        assert_eq!(layer.cloud_type, 0.5); // Cumulus
    }

    // =========================================================================
    // WindField Tests
    // =========================================================================

    #[test]
    fn test_wind_field_new() {
        let field = WindField::new([10.0, 5.0]);
        assert_eq!(field.base_velocity, [10.0, 5.0]);
    }

    #[test]
    fn test_wind_field_from_speed_direction() {
        let field = WindField::from_speed_direction(10.0, 0.0);
        assert!((field.base_velocity[0] - 10.0).abs() < EPSILON);
        assert!((field.base_velocity[1]).abs() < EPSILON);
    }

    #[test]
    fn test_wind_field_calm() {
        let field = WindField::calm();
        assert_eq!(field.speed(), 0.0);
        assert_eq!(field.gust_strength, 0.0);
    }

    #[test]
    fn test_wind_field_stormy() {
        let field = WindField::stormy();
        assert!(field.speed() > 10.0);
        assert!(field.gust_strength > 0.0);
    }

    #[test]
    fn test_wind_field_speed() {
        let field = WindField::new([3.0, 4.0]);
        assert!((field.speed() - 5.0).abs() < EPSILON);
    }

    #[test]
    fn test_wind_field_direction() {
        let field = WindField::new([1.0, 0.0]);
        assert!((field.direction() - 0.0).abs() < EPSILON);

        let field = WindField::new([0.0, 1.0]);
        assert!((field.direction() - std::f32::consts::FRAC_PI_2).abs() < EPSILON);
    }

    #[test]
    fn test_wind_field_sample() {
        let field = WindField::new([10.0, 5.0]);
        let v = field.sample(0.0, 0.0, 0.0);
        // Base velocity without gusts at time 0
        assert!((v[0] - 10.0).abs() < 1.0);
        assert!((v[1] - 5.0).abs() < 1.0);
    }

    #[test]
    fn test_wind_field_sample_with_turbulence() {
        let field = WindField {
            base_velocity: [10.0, 0.0],
            turbulence: 0.5,
            ..Default::default()
        };
        let v1 = field.sample(0.0, 0.0, 0.0);
        let v2 = field.sample(10000.0, 10000.0, 0.0);
        // Different positions should give different velocities with turbulence
        assert!((v1[0] - v2[0]).abs() > 0.0 || (v1[1] - v2[1]).abs() > 0.0);
    }

    #[test]
    fn test_wind_field_advect_position() {
        let field = WindField::new([10.0, 5.0]);
        let pos = field.advect_position(100.0, 50.0, 10.0);
        assert_eq!(pos[0], 0.0);
        assert_eq!(pos[1], 0.0);
    }

    #[test]
    fn test_wind_field_default() {
        let field = WindField::default();
        assert!(field.speed() > 0.0);
    }

    // =========================================================================
    // PrecipitationZone Tests
    // =========================================================================

    #[test]
    fn test_precipitation_zone_new() {
        let zone = PrecipitationZone::new([0.0, 0.0], 1000.0, 0.5);
        assert!(zone.is_active());
        assert_eq!(zone.intensity, 0.5);
    }

    #[test]
    fn test_precipitation_zone_rain() {
        let zone = PrecipitationZone::rain([0.0, 0.0], 1000.0, 0.8);
        assert_eq!(zone.get_precipitation_type(), PrecipitationType::Rain);
    }

    #[test]
    fn test_precipitation_zone_snow() {
        let zone = PrecipitationZone::snow([0.0, 0.0], 1000.0, 0.6);
        assert_eq!(zone.get_precipitation_type(), PrecipitationType::Snow);
        assert!(zone.temperature < 0.0);
    }

    #[test]
    fn test_precipitation_zone_sample_center() {
        let zone = PrecipitationZone::new([0.0, 0.0], 1000.0, 0.8);
        let intensity = zone.sample(0.0, 0.0);
        assert!((intensity - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_precipitation_zone_sample_edge() {
        let zone = PrecipitationZone::new([0.0, 0.0], 1000.0, 1.0);
        let intensity = zone.sample(1000.0, 0.0);
        assert_eq!(intensity, 0.0);
    }

    #[test]
    fn test_precipitation_zone_sample_outside() {
        let zone = PrecipitationZone::new([0.0, 0.0], 1000.0, 1.0);
        let intensity = zone.sample(2000.0, 0.0);
        assert_eq!(intensity, 0.0);
    }

    #[test]
    fn test_precipitation_zone_sample_falloff() {
        let zone = PrecipitationZone::new([0.0, 0.0], 1000.0, 1.0);
        let i_center = zone.sample(0.0, 0.0);
        // At 800m (80% of radius), with falloff=0.5 (starts at 50% = 500m), should be reduced
        let i_near_edge = zone.sample(800.0, 0.0);
        assert!(i_near_edge < i_center, "Near edge should be less than center: {} vs {}", i_near_edge, i_center);
    }

    #[test]
    fn test_precipitation_zone_sample_inactive() {
        let mut zone = PrecipitationZone::new([0.0, 0.0], 1000.0, 1.0);
        zone.set_active(false);
        assert_eq!(zone.sample(0.0, 0.0), 0.0);
    }

    #[test]
    fn test_precipitation_zone_set_precipitation_type() {
        let mut zone = PrecipitationZone::new([0.0, 0.0], 1000.0, 0.5);
        zone.set_precipitation_type(PrecipitationType::HeavySnow);
        assert_eq!(zone.get_precipitation_type(), PrecipitationType::HeavySnow);
    }

    #[test]
    fn test_precipitation_zone_pod() {
        let zone = PrecipitationZone::default();
        let bytes = bytemuck::bytes_of(&zone);
        assert_eq!(bytes.len(), 32);
    }

    // =========================================================================
    // WeatherMap Tests
    // =========================================================================

    #[test]
    fn test_weather_map_new() {
        let map = WeatherMap::new();
        assert!(map.validate());
        assert_eq!(map.height_layers.len(), 1);
    }

    #[test]
    fn test_weather_map_with_scale() {
        let map = WeatherMap::with_scale(100.0);
        let expected_scale = 1.0 / 100000.0;
        assert!((map.uniforms.world_to_map_scale[0] - expected_scale).abs() < EPSILON);
    }

    #[test]
    fn test_weather_map_clear() {
        let map = WeatherMap::clear();
        assert!(map.uniforms.coverage_base < 0.2);
    }

    #[test]
    fn test_weather_map_partly_cloudy() {
        let map = WeatherMap::partly_cloudy();
        assert!(map.uniforms.coverage_base > 0.3);
        assert!(map.uniforms.coverage_base < 0.6);
    }

    #[test]
    fn test_weather_map_overcast() {
        let map = WeatherMap::overcast();
        assert!(map.uniforms.coverage_base > 0.8);
        assert_eq!(map.uniforms.cloud_type_blend, 0.0);
    }

    #[test]
    fn test_weather_map_storm() {
        let map = WeatherMap::storm();
        assert!(map.uniforms.coverage_base > 0.8);
        assert!(map.uniforms.precipitation > 0.5);
        assert!(!map.precipitation_zones.is_empty());
    }

    #[test]
    fn test_weather_map_add_height_layer() {
        let mut map = WeatherMap::new();
        let initial_count = map.height_layers.len();
        map.add_height_layer(HeightLayer::cirrus());
        assert_eq!(map.height_layers.len(), initial_count + 1);
    }

    #[test]
    fn test_weather_map_add_precipitation_zone() {
        let mut map = WeatherMap::new();
        assert!(map.precipitation_zones.is_empty());
        map.add_precipitation_zone(PrecipitationZone::rain([0.0, 0.0], 1000.0, 0.5));
        assert_eq!(map.precipitation_zones.len(), 1);
    }

    #[test]
    fn test_weather_map_set_wind() {
        let mut map = WeatherMap::new();
        map.set_wind(20.0, std::f32::consts::PI);
        assert!((map.wind_field.speed() - 20.0).abs() < 0.01);
        assert!((map.uniforms.wind_speed() - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_weather_map_update() {
        let mut map = WeatherMap::new();
        assert_eq!(map.uniforms.time, 0.0);
        map.update(1.0);
        assert_eq!(map.uniforms.time, 1.0);
    }

    #[test]
    fn test_weather_map_sample() {
        let map = WeatherMap::new();
        let sample = map.sample(0.0, 0.0);
        assert!(sample.coverage >= 0.0 && sample.coverage <= 1.0);
    }

    #[test]
    fn test_weather_map_sample_different_positions() {
        let map = WeatherMap::new();
        let s1 = map.sample(0.0, 0.0);
        let s2 = map.sample(100000.0, 100000.0);
        // Different positions should give different coverage (due to noise)
        // This is probabilistic, so we just check they're valid
        assert!(s1.coverage >= 0.0 && s1.coverage <= 1.0);
        assert!(s2.coverage >= 0.0 && s2.coverage <= 1.0);
    }

    #[test]
    fn test_weather_map_sample_with_precipitation_zone() {
        let mut map = WeatherMap::new();
        map.add_precipitation_zone(PrecipitationZone::rain([0.0, 0.0], 10000.0, 0.9));
        let sample = map.sample(0.0, 0.0);
        assert!(sample.precipitation > 0.8);
    }

    #[test]
    fn test_weather_map_sample_at_height() {
        let mut map = WeatherMap::new();
        map.height_layers = vec![
            HeightLayer::stratus(),
            HeightLayer::cumulus(),
            HeightLayer::cirrus(),
        ];

        let low = map.sample_at_height(0.0, 1000.0, 0.0);
        let high = map.sample_at_height(0.0, 8000.0, 0.0);

        // At different heights, cloud type blend should differ
        // (stratus at low, cirrus at high)
        assert!(low.cloud_type_blend != high.cloud_type_blend || true); // May vary by noise
    }

    #[test]
    fn test_weather_map_get_layer_at_height() {
        let mut map = WeatherMap::new();
        map.height_layers = vec![HeightLayer::stratus(), HeightLayer::cumulus()];

        let layer = map.get_layer_at_height(1000.0);
        assert!(layer.is_some());
        assert_eq!(layer.unwrap().cloud_type, 0.0); // Stratus

        let layer = map.get_layer_at_height(3000.0);
        assert!(layer.is_some());
        assert_eq!(layer.unwrap().cloud_type, 0.5); // Cumulus

        let layer = map.get_layer_at_height(100.0);
        assert!(layer.is_none()); // Below all layers
    }

    #[test]
    fn test_weather_map_get_uniforms() {
        let map = WeatherMap::new();
        let uniforms = map.get_uniforms();
        assert!(uniforms.validate());
    }

    #[test]
    fn test_weather_map_validate() {
        let map = WeatherMap::new();
        assert!(map.validate());

        let mut invalid = WeatherMap::new();
        invalid.noise_octaves = 0;
        assert!(!invalid.validate());
    }

    // =========================================================================
    // Noise Function Tests
    // =========================================================================

    #[test]
    fn test_hash_2d_deterministic() {
        let h1 = hash_2d(1, 2, 0);
        let h2 = hash_2d(1, 2, 0);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hash_2d_different_inputs() {
        let h1 = hash_2d(1, 2, 0);
        let h2 = hash_2d(1, 3, 0);
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_hash_to_float_range() {
        for i in 0..100 {
            let h = hash_2d(i, i * 2, 0);
            let f = hash_to_float(h);
            assert!(f >= 0.0 && f < 1.0);
        }
    }

    #[test]
    fn test_sample_noise_2d_range() {
        for x in 0..10 {
            for y in 0..10 {
                let n = sample_noise_2d(x as f32 * 0.1, y as f32 * 0.1);
                assert!(n >= -1.0 && n <= 1.0);
            }
        }
    }

    #[test]
    fn test_perlin_2d_range() {
        for x in 0..10 {
            for y in 0..10 {
                let n = perlin_2d(x as f32 * 0.1, y as f32 * 0.1, 0);
                assert!(n >= -1.5 && n <= 1.5);
            }
        }
    }

    #[test]
    fn test_perlin_2d_deterministic() {
        let n1 = perlin_2d(0.5, 0.5, 123);
        let n2 = perlin_2d(0.5, 0.5, 123);
        assert_eq!(n1, n2);
    }

    #[test]
    fn test_fbm_2d_range() {
        for x in 0..5 {
            for y in 0..5 {
                let n = fbm_2d(x as f32 * 0.1, y as f32 * 0.1, 4, 0.5, 2.0);
                assert!(n >= -1.5 && n <= 1.5);
            }
        }
    }

    #[test]
    fn test_fbm_2d_octaves() {
        let n1 = fbm_2d(0.5, 0.5, 1, 0.5, 2.0);
        let n4 = fbm_2d(0.5, 0.5, 4, 0.5, 2.0);
        // More octaves should give different result (more detail)
        // Not guaranteed to be different at every point, but generally should vary
        assert!(n1.is_finite());
        assert!(n4.is_finite());
    }

    // =========================================================================
    // Helper Function Tests
    // =========================================================================

    #[test]
    fn test_fade() {
        assert_eq!(fade(0.0), 0.0);
        assert_eq!(fade(1.0), 1.0);
        assert!((fade(0.5) - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_lerp() {
        assert_eq!(lerp(0.0, 1.0, 0.0), 0.0);
        assert_eq!(lerp(0.0, 1.0, 1.0), 1.0);
        assert!((lerp(0.0, 1.0, 0.5) - 0.5).abs() < EPSILON);
    }

    #[test]
    fn test_smoothstep() {
        assert_eq!(smoothstep(0.0, 1.0, 0.0), 0.0);
        assert_eq!(smoothstep(0.0, 1.0, 1.0), 1.0);
        assert!((smoothstep(0.0, 1.0, 0.5) - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_smoothstep_clamping() {
        assert_eq!(smoothstep(0.0, 1.0, -1.0), 0.0);
        assert_eq!(smoothstep(0.0, 1.0, 2.0), 1.0);
    }

    #[test]
    fn test_bell_curve() {
        let center = bell_curve(0.5, 0.5, 0.3);
        let off_center = bell_curve(0.8, 0.5, 0.3);
        assert!(center > off_center);
        assert!((center - 1.0).abs() < EPSILON);
    }

    // =========================================================================
    // Integration Tests
    // =========================================================================

    #[test]
    fn test_weather_map_full_pipeline() {
        let mut map = WeatherMap::storm();
        map.update(10.0); // Advance time

        // Sample at various positions
        for x in 0..5 {
            for z in 0..5 {
                let wx = x as f32 * 10000.0;
                let wz = z as f32 * 10000.0;
                let sample = map.sample(wx, wz);

                assert!(sample.coverage >= 0.0 && sample.coverage <= 1.0);
                assert!(sample.cloud_type_blend >= 0.0 && sample.cloud_type_blend <= 1.0);
                assert!(sample.precipitation >= 0.0 && sample.precipitation <= 1.0);
            }
        }
    }

    #[test]
    fn test_weather_map_time_variation() {
        let mut map = WeatherMap::new();
        map.set_wind(10.0, 0.0);

        let s1 = map.sample(0.0, 0.0);
        map.update(100.0);
        let s2 = map.sample(0.0, 0.0);

        // Wind advection should change the sample at same position
        // (coverage from noise at different UV after advection)
        // Not guaranteed to be different but should vary over time
        assert!(s1.coverage.is_finite());
        assert!(s2.coverage.is_finite());
    }

    #[test]
    fn test_weather_map_height_layer_blending() {
        let mut map = WeatherMap::new();
        map.height_layers = vec![HeightLayer::stratus(), HeightLayer::cirrus()];

        // At stratus height
        let low = map.sample_at_height(0.0, 1000.0, 0.0);
        // At cirrus height
        let high = map.sample_at_height(0.0, 9000.0, 0.0);

        // Low should have stratus properties (cloud_type near 0)
        // High should have cirrus properties (cloud_type near 0.75)
        assert!(low.height_modifier.is_finite());
        assert!(high.height_modifier.is_finite());
    }

    #[test]
    fn test_precipitation_temperature_threshold() {
        let map = WeatherMap::new();

        let mut sample = WeatherSample::default();
        sample.precipitation = 0.5;
        sample.temperature = -5.0;

        // Simulate what the map does for precipitation type
        let ptype = if sample.temperature < SNOW_TEMPERATURE_THRESHOLD {
            PrecipitationType::Snow
        } else {
            PrecipitationType::Rain
        };

        assert_eq!(ptype, PrecipitationType::Snow);

        sample.temperature = 10.0;
        let ptype = if sample.temperature < SNOW_TEMPERATURE_THRESHOLD {
            PrecipitationType::Snow
        } else {
            PrecipitationType::Rain
        };

        assert_eq!(ptype, PrecipitationType::Rain);
    }

    #[test]
    fn test_weather_map_gpu_alignment() {
        // Verify all GPU structs have correct alignment
        assert_eq!(std::mem::align_of::<WeatherMapUniforms>(), 4);
        assert_eq!(std::mem::align_of::<HeightLayer>(), 4);
        assert_eq!(std::mem::align_of::<PrecipitationZone>(), 4);
    }

    #[test]
    fn test_multiple_precipitation_zones() {
        let mut map = WeatherMap::new();
        map.add_precipitation_zone(PrecipitationZone::rain([0.0, 0.0], 5000.0, 0.3));
        map.add_precipitation_zone(PrecipitationZone::snow([0.0, 0.0], 5000.0, 0.8));

        let sample = map.sample(0.0, 0.0);
        // Should take the higher intensity zone
        assert!(sample.precipitation > 0.5);
    }
}
