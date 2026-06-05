//! Shoreline Interaction System for TRINITY Engine (T-ENV-3.10).
//!
//! Implements realistic shoreline wave-terrain interaction effects:
//! - Wave breaking detection and physics (McCowan criterion)
//! - Wet sand material modification
//! - Tidal simulation (12-hour cycle)
//! - Foam distribution along shorelines
//!
//! # Overview
//!
//! The shoreline system bridges the water simulation (Gerstner waves, FFT ocean)
//! with terrain rendering, creating realistic coastal environments. Key features:
//!
//! 1. **Shoreline Detection**: Finds water-terrain intersection from heightmaps
//! 2. **Wave Breaking**: Detects when waves break using depth-based criteria
//! 3. **Wet Sand**: Modifies terrain material based on wave runup and tides
//! 4. **Tidal Simulation**: 12-hour tidal cycle with configurable amplitude
//!
//! # Physics
//!
//! ## Wave Breaking (McCowan Criterion)
//!
//! Waves break when the ratio of wave height to water depth exceeds a threshold:
//! ```text
//! H_b / d_b = 0.78 (McCowan, 1891)
//! ```
//! Where `H_b` is breaking wave height and `d_b` is breaking depth.
//!
//! ## Break Type Classification
//!
//! The Iribarren number (surf similarity parameter) determines break type:
//! ```text
//! ξ = tan(β) / sqrt(H/L)
//! ```
//! - ξ < 0.5: Spilling breaker (gentle slope)
//! - ξ > 2.0: Plunging breaker (steep slope)
//! - Between: Collapsing/surging
//!
//! ## Wave Runup (Hunt Formula)
//!
//! Maximum wave runup on a beach:
//! ```text
//! R = 0.35 * β * sqrt(H * L)
//! ```
//! Where β is beach slope, H is wave height, L is wavelength.
//!
//! # Usage
//!
//! ```ignore
//! use renderer_backend::shoreline::{
//!     ShorelineConfig, ShorelineDetector, WaveBreaking, WetSand,
//!     TideSimulation, ShorelineInteraction,
//! };
//!
//! // Initialize
//! let config = ShorelineConfig::default();
//! let mut interaction = ShorelineInteraction::new(config);
//!
//! // Update each frame
//! interaction.update(dt);
//!
//! // Query effects at world position
//! let foam = interaction.get_shore_foam([x, y, z]);
//! let wetness = interaction.get_wet_sand_factor([x, y, z]);
//! ```

use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// McCowan breaking criterion: H/d ratio at breaking.
pub const MCCOWAN_RATIO: f32 = 0.78;

/// Default wave break depth coefficient (McCowan).
pub const DEFAULT_BREAK_DEPTH_COEFF: f32 = 0.78;

/// Default foam spread distance in meters.
pub const DEFAULT_FOAM_SPREAD: f32 = 5.0;

/// Default wet sand distance in meters.
pub const DEFAULT_WET_SAND_DISTANCE: f32 = 3.0;

/// Default wave runup height in meters.
pub const DEFAULT_WAVE_RUN_UP: f32 = 1.5;

/// Default tidal amplitude in meters.
pub const DEFAULT_TIDE_AMPLITUDE: f32 = 0.5;

/// Default tidal period in seconds (12 hours).
pub const DEFAULT_TIDE_PERIOD: f32 = 43200.0;

/// ShorelineConfig struct size in bytes.
pub const SHORELINE_CONFIG_SIZE: usize = 32;

/// Small epsilon for floating point comparisons.
const EPSILON: f32 = 1e-6;

/// Gravitational acceleration (m/s^2).
const GRAVITY: f32 = 9.81;

// ---------------------------------------------------------------------------
// ShorelineConfig
// ---------------------------------------------------------------------------

/// Configuration for shoreline interaction effects.
///
/// GPU-compatible layout for uniform buffers.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShorelineConfig {
    /// Wave break depth coefficient (default: 0.78, McCowan).
    /// Waves break when depth < wave_height * this value.
    pub wave_break_depth: f32,

    /// Foam spread distance from shoreline in meters (default: 5.0).
    pub foam_spread: f32,

    /// Wet sand distance from waterline in meters (default: 3.0).
    pub wet_sand_distance: f32,

    /// Maximum wave runup height in meters (default: 1.5).
    pub wave_run_up: f32,

    /// Tidal amplitude in meters (default: 0.5).
    pub tide_amplitude: f32,

    /// Tidal period in seconds (default: 43200 = 12 hours).
    pub tide_period: f32,

    /// Padding for 16-byte alignment.
    pub _padding: [f32; 2],
}

const _: () = assert!(std::mem::size_of::<ShorelineConfig>() == SHORELINE_CONFIG_SIZE);

impl Default for ShorelineConfig {
    fn default() -> Self {
        Self {
            wave_break_depth: DEFAULT_BREAK_DEPTH_COEFF,
            foam_spread: DEFAULT_FOAM_SPREAD,
            wet_sand_distance: DEFAULT_WET_SAND_DISTANCE,
            wave_run_up: DEFAULT_WAVE_RUN_UP,
            tide_amplitude: DEFAULT_TIDE_AMPLITUDE,
            tide_period: DEFAULT_TIDE_PERIOD,
            _padding: [0.0; 2],
        }
    }
}

impl ShorelineConfig {
    /// Create a new configuration with custom values.
    pub fn new(
        wave_break_depth: f32,
        foam_spread: f32,
        wet_sand_distance: f32,
        wave_run_up: f32,
        tide_amplitude: f32,
        tide_period: f32,
    ) -> Self {
        Self {
            wave_break_depth: wave_break_depth.max(EPSILON),
            foam_spread: foam_spread.max(EPSILON),
            wet_sand_distance: wet_sand_distance.max(EPSILON),
            wave_run_up: wave_run_up.max(0.0),
            tide_amplitude: tide_amplitude.max(0.0),
            tide_period: tide_period.max(1.0),
            _padding: [0.0; 2],
        }
    }

    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.wave_break_depth <= 0.0 {
            return Err("Wave break depth coefficient must be positive");
        }
        if self.foam_spread <= 0.0 {
            return Err("Foam spread must be positive");
        }
        if self.wet_sand_distance <= 0.0 {
            return Err("Wet sand distance must be positive");
        }
        if self.wave_run_up < 0.0 {
            return Err("Wave runup must be non-negative");
        }
        if self.tide_amplitude < 0.0 {
            return Err("Tide amplitude must be non-negative");
        }
        if self.tide_period <= 0.0 {
            return Err("Tide period must be positive");
        }
        Ok(())
    }

    /// Configuration for calm seas with minimal effects.
    pub fn calm() -> Self {
        Self {
            wave_break_depth: 0.85,
            foam_spread: 3.0,
            wet_sand_distance: 2.0,
            wave_run_up: 0.5,
            tide_amplitude: 0.3,
            tide_period: DEFAULT_TIDE_PERIOD,
            _padding: [0.0; 2],
        }
    }

    /// Configuration for stormy conditions.
    pub fn stormy() -> Self {
        Self {
            wave_break_depth: 0.65,
            foam_spread: 10.0,
            wet_sand_distance: 6.0,
            wave_run_up: 3.0,
            tide_amplitude: 0.8,
            tide_period: DEFAULT_TIDE_PERIOD,
            _padding: [0.0; 2],
        }
    }
}

// ---------------------------------------------------------------------------
// ShorelinePoint
// ---------------------------------------------------------------------------

/// A point along the shoreline.
#[derive(Clone, Copy, Debug, Default)]
pub struct ShorelinePoint {
    /// World position [x, y, z].
    pub position: [f32; 3],
    /// Normal vector perpendicular to shore (pointing inland).
    pub normal: [f32; 3],
    /// Local beach slope angle (radians).
    pub slope: f32,
    /// Distance along shoreline from reference point.
    pub arc_length: f32,
}

impl ShorelinePoint {
    /// Create a new shoreline point.
    pub fn new(position: [f32; 3], normal: [f32; 3], slope: f32) -> Self {
        Self {
            position,
            normal,
            slope,
            arc_length: 0.0,
        }
    }

    /// Create a shoreline point with arc length.
    pub fn with_arc_length(mut self, arc_length: f32) -> Self {
        self.arc_length = arc_length;
        self
    }
}

// ---------------------------------------------------------------------------
// ShorelineDetector
// ---------------------------------------------------------------------------

/// Detects shoreline from terrain heightmap and water level.
#[derive(Clone, Debug)]
pub struct ShorelineDetector {
    /// Configuration.
    config: ShorelineConfig,
    /// Cached shoreline points (cleared on water level change).
    cached_points: Vec<ShorelinePoint>,
    /// Last water level used for cache.
    cached_water_level: f32,
}

impl ShorelineDetector {
    /// Create a new shoreline detector.
    pub fn new(config: ShorelineConfig) -> Self {
        Self {
            config,
            cached_points: Vec::new(),
            cached_water_level: f32::NAN,
        }
    }

    /// Find shoreline points from terrain heightmap.
    ///
    /// # Arguments
    ///
    /// * `heightmap` - 2D terrain heightmap (row-major)
    /// * `width` - Heightmap width in pixels
    /// * `height` - Heightmap height in pixels
    /// * `world_scale` - World units per pixel
    /// * `world_origin` - World origin [x, z]
    /// * `water_level` - Current water level
    ///
    /// # Returns
    ///
    /// Vec of shoreline points where terrain intersects water.
    pub fn find_shoreline(
        &mut self,
        heightmap: &[f32],
        width: usize,
        height: usize,
        world_scale: f32,
        world_origin: [f32; 2],
        water_level: f32,
    ) -> Vec<ShorelinePoint> {
        if heightmap.len() < width * height {
            return Vec::new();
        }

        // Check cache
        if (self.cached_water_level - water_level).abs() < EPSILON && !self.cached_points.is_empty()
        {
            return self.cached_points.clone();
        }

        let mut points: Vec<ShorelinePoint> = Vec::new();
        let mut arc_length = 0.0f32;

        // March through heightmap looking for water crossings
        for y in 0..height.saturating_sub(1) {
            for x in 0..width.saturating_sub(1) {
                let idx = y * width + x;
                let h00 = heightmap[idx];
                let h10 = heightmap[idx + 1];
                let h01 = heightmap[idx + width];
                let h11 = heightmap[idx + width + 1];

                // Check for water level crossing in this cell
                let above = [
                    h00 > water_level,
                    h10 > water_level,
                    h01 > water_level,
                    h11 > water_level,
                ];

                let count_above = above.iter().filter(|&&b| b).count();

                // Mixed cell (some above, some below water)
                if count_above > 0 && count_above < 4 {
                    // Interpolate crossing position
                    let world_x = world_origin[0] + (x as f32 + 0.5) * world_scale;
                    let world_z = world_origin[1] + (y as f32 + 0.5) * world_scale;

                    // Compute local normal from gradient
                    let dx = (h10 - h00 + h11 - h01) * 0.5;
                    let dz = (h01 - h00 + h11 - h10) * 0.5;
                    let slope = (dx * dx + dz * dz).sqrt().atan();

                    // Normal perpendicular to shoreline (pointing inland)
                    let normal = normalize_vec3([-dx, 0.0, -dz]);

                    let point = ShorelinePoint {
                        position: [world_x, water_level, world_z],
                        normal,
                        slope,
                        arc_length,
                    };

                    // Update arc length
                    if let Some(last) = points.last() {
                        let dx = point.position[0] - last.position[0];
                        let dz = point.position[2] - last.position[2];
                        arc_length += (dx * dx + dz * dz).sqrt();
                    }

                    points.push(point.with_arc_length(arc_length));
                }
            }
        }

        // Cache results
        self.cached_points = points.clone();
        self.cached_water_level = water_level;

        points
    }

    /// Sample signed distance to shoreline from world position.
    ///
    /// Returns positive values for land, negative for water.
    pub fn sample_shore_distance(&self, world_pos: [f32; 3], water_level: f32) -> f32 {
        if self.cached_points.is_empty() {
            // Fallback: use height difference
            return world_pos[1] - water_level;
        }

        // Find nearest shoreline point
        let mut min_dist = f32::MAX;
        for point in &self.cached_points {
            let dx = world_pos[0] - point.position[0];
            let dz = world_pos[2] - point.position[2];
            let dist = (dx * dx + dz * dz).sqrt();
            min_dist = min_dist.min(dist);
        }

        // Sign based on height relative to water
        let sign = if world_pos[1] > water_level { 1.0 } else { -1.0 };
        min_dist * sign
    }

    /// Check if position is in wave breaking zone.
    pub fn is_breaking_zone(&self, _pos: [f32; 3], wave_height: f32, depth: f32) -> bool {
        if depth <= 0.0 {
            return false;
        }

        // McCowan criterion: waves break when H/d > 0.78
        let ratio = wave_height / depth;
        ratio > self.config.wave_break_depth
    }

    /// Get shore normal at position (perpendicular to shoreline).
    pub fn get_shore_normal(&self, pos: [f32; 3]) -> [f32; 3] {
        if self.cached_points.is_empty() {
            return [0.0, 0.0, 1.0]; // Default: pointing in +Z
        }

        // Find nearest shoreline point and return its normal
        let mut nearest_idx = 0;
        let mut min_dist = f32::MAX;

        for (i, point) in self.cached_points.iter().enumerate() {
            let dx = pos[0] - point.position[0];
            let dz = pos[2] - point.position[2];
            let dist = dx * dx + dz * dz;
            if dist < min_dist {
                min_dist = dist;
                nearest_idx = i;
            }
        }

        self.cached_points[nearest_idx].normal
    }

    /// Clear cached shoreline points.
    pub fn clear_cache(&mut self) {
        self.cached_points.clear();
        self.cached_water_level = f32::NAN;
    }

    /// Get configuration.
    pub fn config(&self) -> &ShorelineConfig {
        &self.config
    }

    /// Get cached shoreline points.
    pub fn cached_points(&self) -> &[ShorelinePoint] {
        &self.cached_points
    }
}

// ---------------------------------------------------------------------------
// BreakType
// ---------------------------------------------------------------------------

/// Type of wave breaking.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BreakType {
    /// Gentle breaking with foam rolling down wave face (ξ < 0.5).
    Spilling,
    /// Dramatic breaking with wave curling over (ξ > 2.0).
    Plunging,
    /// Intermediate breaking (0.5 < ξ < 2.0).
    Collapsing,
    /// Wave surges up beach without breaking (very steep beach).
    Surging,
}

impl BreakType {
    /// Get foam intensity multiplier for this break type.
    pub fn foam_multiplier(&self) -> f32 {
        match self {
            BreakType::Spilling => 0.6,
            BreakType::Plunging => 1.0,
            BreakType::Collapsing => 0.8,
            BreakType::Surging => 0.3,
        }
    }

    /// Get spray intensity for this break type.
    pub fn spray_intensity(&self) -> f32 {
        match self {
            BreakType::Spilling => 0.2,
            BreakType::Plunging => 1.0,
            BreakType::Collapsing => 0.5,
            BreakType::Surging => 0.1,
        }
    }
}

impl Default for BreakType {
    fn default() -> Self {
        BreakType::Spilling
    }
}

// ---------------------------------------------------------------------------
// BreakPoint
// ---------------------------------------------------------------------------

/// Information about where a wave breaks.
#[derive(Clone, Copy, Debug, Default)]
pub struct BreakPoint {
    /// Position where wave breaks [x, y, z].
    pub position: [f32; 3],
    /// Breaking wave height.
    pub wave_height: f32,
    /// Water depth at break point.
    pub depth: f32,
    /// Type of breaking.
    pub break_type: BreakType,
    /// Breaking intensity (0-1).
    pub intensity: f32,
}

// ---------------------------------------------------------------------------
// WaveBreaking
// ---------------------------------------------------------------------------

/// Wave breaking physics and detection.
#[derive(Clone, Debug)]
pub struct WaveBreaking {
    /// Configuration.
    config: ShorelineConfig,
}

impl WaveBreaking {
    /// Create a new wave breaking calculator.
    pub fn new(config: ShorelineConfig) -> Self {
        Self { config }
    }

    /// Compute where a wave will break given depth profile.
    ///
    /// # Arguments
    ///
    /// * `wave_height` - Incident wave height in deep water
    /// * `wavelength` - Wave wavelength
    /// * `depth` - Current water depth at evaluation point
    ///
    /// # Returns
    ///
    /// Break point information if wave is breaking, None otherwise.
    pub fn compute_break_point(
        &self,
        wave_height: f32,
        wavelength: f32,
        depth: f32,
    ) -> Option<BreakPoint> {
        if wave_height <= 0.0 || depth <= 0.0 {
            return None;
        }

        // McCowan criterion: H/d = 0.78
        let break_depth = wave_height / self.config.wave_break_depth;

        // Wave is breaking if current depth is at or below break depth
        if depth <= break_depth {
            let intensity = self.breaking_intensity(depth, wave_height);
            let break_type = self.classify_break(0.1, wave_height, wavelength); // Assume 0.1 slope

            Some(BreakPoint {
                position: [0.0, 0.0, 0.0], // Position must be set by caller
                wave_height,
                depth,
                break_type,
                intensity,
            })
        } else {
            None
        }
    }

    /// Calculate breaking intensity based on depth and wave height.
    ///
    /// Returns value in range [0, 1].
    pub fn breaking_intensity(&self, depth: f32, wave_height: f32) -> f32 {
        if depth <= 0.0 || wave_height <= 0.0 {
            return 0.0;
        }

        // Ratio of wave height to depth
        let ratio = wave_height / depth;

        // Intensity increases as ratio exceeds McCowan threshold
        let excess = (ratio - self.config.wave_break_depth) / self.config.wave_break_depth;
        excess.clamp(0.0, 1.0)
    }

    /// Classify wave break type based on beach slope and wave steepness.
    ///
    /// Uses Iribarren number (surf similarity parameter):
    /// ξ = tan(β) / sqrt(H/L)
    pub fn plunge_vs_spill(&self, beach_slope: f32, wave_steepness: f32) -> BreakType {
        self.classify_break(beach_slope, 1.0, 1.0 / wave_steepness.max(EPSILON))
    }

    /// Classify break type using Iribarren number.
    fn classify_break(&self, beach_slope: f32, wave_height: f32, wavelength: f32) -> BreakType {
        if wavelength <= 0.0 || wave_height <= 0.0 {
            return BreakType::Spilling;
        }

        // Iribarren number: ξ = tan(β) / sqrt(H/L)
        let steepness = wave_height / wavelength;
        let iribarren = beach_slope.tan() / steepness.sqrt();

        if iribarren < 0.5 {
            BreakType::Spilling
        } else if iribarren < 2.0 {
            BreakType::Collapsing
        } else if iribarren < 3.3 {
            BreakType::Plunging
        } else {
            BreakType::Surging
        }
    }

    /// Get foam spawn rate based on breaking intensity.
    pub fn get_foam_spawn_rate(&self, breaking_intensity: f32) -> f32 {
        // Exponential ramp-up of foam with intensity
        let intensity = breaking_intensity.clamp(0.0, 1.0);
        intensity * intensity * 10.0 // Max 10 foam particles per unit time
    }

    /// Calculate wave runup height using Hunt formula.
    ///
    /// R = 0.35 * β * sqrt(H * L)
    pub fn wave_runup(&self, beach_slope: f32, wave_height: f32, wavelength: f32) -> f32 {
        if wave_height <= 0.0 || wavelength <= 0.0 {
            return 0.0;
        }

        0.35 * beach_slope * (wave_height * wavelength).sqrt()
    }

    /// Get configuration.
    pub fn config(&self) -> &ShorelineConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// WetSand
// ---------------------------------------------------------------------------

/// Wet sand material modification.
#[derive(Clone, Debug)]
pub struct WetSand {
    /// Configuration.
    config: ShorelineConfig,
}

impl WetSand {
    /// Create a new wet sand calculator.
    pub fn new(config: ShorelineConfig) -> Self {
        Self { config }
    }

    /// Compute sand wetness based on shore distance, tide, and wave runup.
    ///
    /// Returns wetness in range [0, 1].
    pub fn compute_wetness(&self, shore_distance: f32, tide_phase: f32, wave_runup: f32) -> f32 {
        if shore_distance < 0.0 {
            // Underwater
            return 1.0;
        }

        // Wet zone extends based on tide and wave runup
        let tide_offset = tide_phase.sin() * self.config.tide_amplitude;
        let wet_extent = self.config.wet_sand_distance + tide_offset + wave_runup;

        if shore_distance > wet_extent {
            return 0.0;
        }

        // Gradient from waterline to edge of wet zone
        let t = shore_distance / wet_extent.max(EPSILON);
        1.0 - smoothstep(0.0, 1.0, t)
    }

    /// Get albedo multiplier for wet sand (darker when wet).
    ///
    /// Wet sand is typically 30-50% darker than dry sand.
    pub fn wet_sand_albedo_multiplier(&self, wetness: f32) -> f32 {
        let wetness = wetness.clamp(0.0, 1.0);
        // Linear interpolation: dry=1.0, wet=0.5
        1.0 - wetness * 0.5
    }

    /// Get roughness for wet sand (smoother when wet).
    ///
    /// Wet sand has lower roughness due to water film.
    pub fn wet_sand_roughness(&self, wetness: f32) -> f32 {
        let wetness = wetness.clamp(0.0, 1.0);
        // Dry sand roughness ~0.8, wet sand ~0.3
        0.8 - wetness * 0.5
    }

    /// Generate puddle mask based on position and shore distance.
    ///
    /// Returns puddle intensity in range [0, 1].
    pub fn puddle_mask(&self, pos: [f32; 3], shore_distance: f32) -> f32 {
        if shore_distance < 0.0 || shore_distance > self.config.wet_sand_distance {
            return 0.0;
        }

        // Simple noise-based puddle pattern
        let noise = pseudo_noise_2d(pos[0] * 0.5, pos[2] * 0.5);

        // Puddles more likely near waterline
        let distance_factor = 1.0 - (shore_distance / self.config.wet_sand_distance);

        // Threshold noise to create discrete puddles
        let puddle_threshold = 0.6 - distance_factor * 0.3;
        if noise > puddle_threshold {
            (noise - puddle_threshold) / (1.0 - puddle_threshold)
        } else {
            0.0
        }
    }

    /// Get configuration.
    pub fn config(&self) -> &ShorelineConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// TideSimulation
// ---------------------------------------------------------------------------

/// Tidal simulation with configurable period and amplitude.
#[derive(Clone, Debug)]
pub struct TideSimulation {
    /// Tidal amplitude in meters.
    amplitude: f32,
    /// Tidal period in seconds.
    period: f32,
    /// Current simulation time.
    time: f32,
    /// Current tidal phase (0 to 2*PI).
    phase: f32,
}

impl TideSimulation {
    /// Create a new tide simulation.
    pub fn new(amplitude: f32, period: f32) -> Self {
        Self {
            amplitude: amplitude.max(0.0),
            period: period.max(1.0),
            time: 0.0,
            phase: 0.0,
        }
    }

    /// Update simulation by delta time.
    pub fn update(&mut self, dt: f32) {
        self.time += dt;
        self.phase = (self.time / self.period) * 2.0 * PI;
    }

    /// Get current water level offset from base level.
    pub fn get_water_level(&self, base_level: f32) -> f32 {
        base_level + self.amplitude * self.phase.sin()
    }

    /// Get current tide velocity (rate of change).
    ///
    /// Positive = rising tide, negative = falling tide.
    pub fn get_tide_velocity(&self) -> f32 {
        // Derivative of sin is cos
        let angular_freq = 2.0 * PI / self.period;
        self.amplitude * angular_freq * self.phase.cos()
    }

    /// Check if currently at high tide.
    pub fn is_high_tide(&self) -> bool {
        // High tide when phase near PI/2
        let normalized = self.phase % (2.0 * PI);
        (normalized - PI / 2.0).abs() < 0.1 || (normalized - PI / 2.0 + 2.0 * PI).abs() < 0.1
    }

    /// Check if currently at low tide.
    pub fn is_low_tide(&self) -> bool {
        // Low tide when phase near 3*PI/2
        let normalized = self.phase % (2.0 * PI);
        (normalized - 3.0 * PI / 2.0).abs() < 0.1
    }

    /// Get current time.
    pub fn time(&self) -> f32 {
        self.time
    }

    /// Get current phase.
    pub fn phase(&self) -> f32 {
        self.phase
    }

    /// Get amplitude.
    pub fn amplitude(&self) -> f32 {
        self.amplitude
    }

    /// Get period.
    pub fn period(&self) -> f32 {
        self.period
    }

    /// Set simulation time directly.
    pub fn set_time(&mut self, time: f32) {
        self.time = time;
        self.phase = (self.time / self.period) * 2.0 * PI;
    }

    /// Reset simulation to initial state.
    pub fn reset(&mut self) {
        self.time = 0.0;
        self.phase = 0.0;
    }
}

impl Default for TideSimulation {
    fn default() -> Self {
        Self::new(DEFAULT_TIDE_AMPLITUDE, DEFAULT_TIDE_PERIOD)
    }
}

// ---------------------------------------------------------------------------
// WaveModification
// ---------------------------------------------------------------------------

/// Modification to apply to Gerstner wave sample near shore.
#[derive(Clone, Copy, Debug, Default)]
pub struct WaveModification {
    /// Height attenuation factor (0-1).
    pub height_scale: f32,
    /// Steepness modification.
    pub steepness_scale: f32,
    /// Phase shift from shoaling.
    pub phase_shift: f32,
    /// Wave is breaking.
    pub is_breaking: bool,
}

// ---------------------------------------------------------------------------
// ShorelineInteraction
// ---------------------------------------------------------------------------

/// Main compositor for all shoreline effects.
#[derive(Clone, Debug)]
pub struct ShorelineInteraction {
    /// Configuration.
    config: ShorelineConfig,
    /// Shoreline detector.
    detector: ShorelineDetector,
    /// Wave breaking physics.
    wave_breaking: WaveBreaking,
    /// Wet sand material.
    wet_sand: WetSand,
    /// Tide simulation.
    tide: TideSimulation,
    /// Current simulation time.
    time: f32,
    /// Base water level.
    base_water_level: f32,
}

impl ShorelineInteraction {
    /// Create a new shoreline interaction system.
    pub fn new(config: ShorelineConfig) -> Self {
        let tide = TideSimulation::new(config.tide_amplitude, config.tide_period);

        Self {
            detector: ShorelineDetector::new(config),
            wave_breaking: WaveBreaking::new(config),
            wet_sand: WetSand::new(config),
            config,
            tide,
            time: 0.0,
            base_water_level: 0.0,
        }
    }

    /// Update simulation by delta time.
    pub fn update(&mut self, dt: f32) {
        self.time += dt;
        self.tide.update(dt);
    }

    /// Set base water level.
    pub fn set_base_water_level(&mut self, level: f32) {
        self.base_water_level = level;
    }

    /// Get current water level (base + tide).
    pub fn current_water_level(&self) -> f32 {
        self.tide.get_water_level(self.base_water_level)
    }

    /// Get shore foam intensity at world position.
    ///
    /// Returns foam value in range [0, 1].
    pub fn get_shore_foam(&self, world_pos: [f32; 3]) -> f32 {
        let water_level = self.current_water_level();
        let shore_dist = self.detector.sample_shore_distance(world_pos, water_level);

        if shore_dist < 0.0 {
            // Underwater - less foam
            let depth = -shore_dist;
            if depth < self.config.foam_spread {
                return (1.0 - depth / self.config.foam_spread) * 0.5;
            }
            return 0.0;
        }

        // On land - foam from runup
        if shore_dist > self.config.foam_spread {
            return 0.0;
        }

        // Foam gradient from waterline
        let t = shore_dist / self.config.foam_spread;
        let base_foam = 1.0 - smoothstep(0.0, 1.0, t);

        // Add noise variation
        let noise = pseudo_noise_2d(world_pos[0] * 0.3 + self.time * 0.5, world_pos[2] * 0.3);
        (base_foam * (0.7 + noise * 0.3)).clamp(0.0, 1.0)
    }

    /// Get wet sand factor at world position.
    ///
    /// Returns wetness in range [0, 1].
    pub fn get_wet_sand_factor(&self, world_pos: [f32; 3]) -> f32 {
        let water_level = self.current_water_level();
        let shore_dist = self.detector.sample_shore_distance(world_pos, water_level);

        self.wet_sand
            .compute_wetness(shore_dist, self.tide.phase(), self.config.wave_run_up)
    }

    /// Get wave modification for a Gerstner sample based on depth.
    pub fn get_wave_modification(&self, _gerstner_sample: [f32; 3], depth: f32) -> WaveModification {
        if depth <= 0.0 {
            return WaveModification {
                height_scale: 0.0,
                steepness_scale: 0.0,
                phase_shift: 0.0,
                is_breaking: false,
            };
        }

        // Shoaling: waves slow down and increase in height in shallow water
        // Using simplified linear shoaling coefficient
        let deep_water_depth = 10.0; // Reference depth
        let shoaling_coeff = if depth < deep_water_depth {
            (deep_water_depth / depth).sqrt().min(2.0)
        } else {
            1.0
        };

        // Check for breaking
        let effective_height = 1.0 * shoaling_coeff; // Assume unit wave height
        let is_breaking = self.detector.is_breaking_zone([0.0, 0.0, 0.0], effective_height, depth);

        let height_scale = if is_breaking {
            // Waves dissipate after breaking
            (depth / (effective_height / self.config.wave_break_depth)).min(1.0)
        } else {
            shoaling_coeff
        };

        // Waves become steeper in shallow water
        let steepness_scale = shoaling_coeff;

        // Phase shift from wave slowing
        let phase_shift = if depth < deep_water_depth {
            (1.0 - depth / deep_water_depth) * 0.5
        } else {
            0.0
        };

        WaveModification {
            height_scale,
            steepness_scale,
            phase_shift,
            is_breaking,
        }
    }

    /// Apply terrain wetness modification to material parameters.
    ///
    /// Returns (albedo_multiplier, roughness).
    pub fn apply_terrain_wetness(&self, shore_distance: f32) -> (f32, f32) {
        let wetness = self.wet_sand.compute_wetness(
            shore_distance,
            self.tide.phase(),
            self.config.wave_run_up,
        );

        let albedo_mult = self.wet_sand.wet_sand_albedo_multiplier(wetness);
        let roughness = self.wet_sand.wet_sand_roughness(wetness);

        (albedo_mult, roughness)
    }

    /// Get configuration.
    pub fn config(&self) -> &ShorelineConfig {
        &self.config
    }

    /// Get shoreline detector.
    pub fn detector(&self) -> &ShorelineDetector {
        &self.detector
    }

    /// Get mutable shoreline detector.
    pub fn detector_mut(&mut self) -> &mut ShorelineDetector {
        &mut self.detector
    }

    /// Get wave breaking.
    pub fn wave_breaking(&self) -> &WaveBreaking {
        &self.wave_breaking
    }

    /// Get wet sand.
    pub fn wet_sand(&self) -> &WetSand {
        &self.wet_sand
    }

    /// Get tide simulation.
    pub fn tide(&self) -> &TideSimulation {
        &self.tide
    }

    /// Get mutable tide simulation.
    pub fn tide_mut(&mut self) -> &mut TideSimulation {
        &mut self.tide
    }

    /// Get current time.
    pub fn time(&self) -> f32 {
        self.time
    }
}

impl Default for ShorelineInteraction {
    fn default() -> Self {
        Self::new(ShorelineConfig::default())
    }
}

// ---------------------------------------------------------------------------
// Utility Functions
// ---------------------------------------------------------------------------

/// Smoothstep interpolation.
#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Normalize a 3D vector.
#[inline]
fn normalize_vec3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > EPSILON {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 1.0, 0.0]
    }
}

/// Simple 2D pseudo-noise function.
/// Returns value in range [0, 1].
#[inline]
fn pseudo_noise_2d(x: f32, y: f32) -> f32 {
    let n = (x * 12.9898 + y * 78.233).sin() * 43758.5453;
    // fract() can return negative values for negative inputs
    // Use abs() after fract() to ensure [0, 1) range
    n.fract().abs()
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

    // -----------------------------------------------------------------------
    // ShorelineConfig Tests
    // -----------------------------------------------------------------------

    // Test 1: Config struct size
    #[test]
    fn test_config_struct_size() {
        assert_eq!(
            std::mem::size_of::<ShorelineConfig>(),
            SHORELINE_CONFIG_SIZE
        );
        assert_eq!(std::mem::size_of::<ShorelineConfig>(), 32);
    }

    // Test 2: Config default values
    #[test]
    fn test_config_default_values() {
        let config = ShorelineConfig::default();
        assert!(approx_eq(config.wave_break_depth, MCCOWAN_RATIO));
        assert!(approx_eq(config.foam_spread, DEFAULT_FOAM_SPREAD));
        assert!(approx_eq(config.wet_sand_distance, DEFAULT_WET_SAND_DISTANCE));
        assert!(approx_eq(config.wave_run_up, DEFAULT_WAVE_RUN_UP));
        assert!(approx_eq(config.tide_amplitude, DEFAULT_TIDE_AMPLITUDE));
        assert!(approx_eq(config.tide_period, DEFAULT_TIDE_PERIOD));
    }

    // Test 3: Config validation success
    #[test]
    fn test_config_validation_success() {
        let config = ShorelineConfig::default();
        assert!(config.validate().is_ok());
    }

    // Test 4: Config validation failure - wave break depth
    #[test]
    fn test_config_validation_wave_break_depth() {
        let mut config = ShorelineConfig::default();
        config.wave_break_depth = 0.0;
        assert!(config.validate().is_err());
    }

    // Test 5: Config validation failure - foam spread
    #[test]
    fn test_config_validation_foam_spread() {
        let mut config = ShorelineConfig::default();
        config.foam_spread = -1.0;
        assert!(config.validate().is_err());
    }

    // Test 6: Config calm preset
    #[test]
    fn test_config_calm() {
        let config = ShorelineConfig::calm();
        assert!(config.wave_break_depth > MCCOWAN_RATIO); // Higher threshold = calmer
        assert!(config.foam_spread < DEFAULT_FOAM_SPREAD);
    }

    // Test 7: Config stormy preset
    #[test]
    fn test_config_stormy() {
        let config = ShorelineConfig::stormy();
        assert!(config.wave_break_depth < MCCOWAN_RATIO); // Lower threshold = more breaking
        assert!(config.foam_spread > DEFAULT_FOAM_SPREAD);
    }

    // Test 8: Config new clamps values
    #[test]
    fn test_config_new_clamps() {
        let config = ShorelineConfig::new(-1.0, -1.0, -1.0, -1.0, -1.0, 0.0);
        assert!(config.wave_break_depth > 0.0);
        assert!(config.foam_spread > 0.0);
        assert!(config.wet_sand_distance > 0.0);
        assert!(config.tide_period >= 1.0);
    }

    // Test 9: Config bytemuck pod
    #[test]
    fn test_config_bytemuck() {
        let config = ShorelineConfig::default();
        let bytes: &[u8] = bytemuck::bytes_of(&config);
        assert_eq!(bytes.len(), SHORELINE_CONFIG_SIZE);
    }

    // -----------------------------------------------------------------------
    // ShorelineDetector Tests
    // -----------------------------------------------------------------------

    // Test 10: Detector creation
    #[test]
    fn test_detector_creation() {
        let config = ShorelineConfig::default();
        let detector = ShorelineDetector::new(config);
        assert!(detector.cached_points().is_empty());
    }

    // Test 11: Shoreline detection basic
    #[test]
    fn test_shoreline_detection_basic() {
        let config = ShorelineConfig::default();
        let mut detector = ShorelineDetector::new(config);

        // Simple heightmap: slope from 0 to 2
        let heightmap: Vec<f32> = (0..16)
            .map(|i| {
                let x = i % 4;
                x as f32 * 0.5
            })
            .collect();

        let points = detector.find_shoreline(&heightmap, 4, 4, 1.0, [0.0, 0.0], 0.75);

        // Should find crossing points
        assert!(!points.is_empty());
    }

    // Test 12: Shoreline detection empty heightmap
    #[test]
    fn test_shoreline_detection_empty() {
        let config = ShorelineConfig::default();
        let mut detector = ShorelineDetector::new(config);
        let points = detector.find_shoreline(&[], 0, 0, 1.0, [0.0, 0.0], 0.0);
        assert!(points.is_empty());
    }

    // Test 13: Shoreline detection all underwater
    #[test]
    fn test_shoreline_detection_all_underwater() {
        let config = ShorelineConfig::default();
        let mut detector = ShorelineDetector::new(config);
        let heightmap = vec![0.0; 16]; // All at sea level
        let points = detector.find_shoreline(&heightmap, 4, 4, 1.0, [0.0, 0.0], 1.0);
        assert!(points.is_empty()); // No crossings when all underwater
    }

    // Test 14: Shoreline detection all above water
    #[test]
    fn test_shoreline_detection_all_above() {
        let config = ShorelineConfig::default();
        let mut detector = ShorelineDetector::new(config);
        let heightmap = vec![10.0; 16]; // All above water
        let points = detector.find_shoreline(&heightmap, 4, 4, 1.0, [0.0, 0.0], 0.0);
        assert!(points.is_empty()); // No crossings when all above
    }

    // Test 15: Shore distance underwater
    #[test]
    fn test_shore_distance_underwater() {
        let config = ShorelineConfig::default();
        let detector = ShorelineDetector::new(config);
        let dist = detector.sample_shore_distance([0.0, -5.0, 0.0], 0.0);
        assert!(dist < 0.0); // Negative = underwater
    }

    // Test 16: Shore distance above water
    #[test]
    fn test_shore_distance_above() {
        let config = ShorelineConfig::default();
        let detector = ShorelineDetector::new(config);
        let dist = detector.sample_shore_distance([0.0, 5.0, 0.0], 0.0);
        assert!(dist > 0.0); // Positive = above water
    }

    // Test 17: Breaking zone detection
    #[test]
    fn test_breaking_zone_detection() {
        let config = ShorelineConfig::default();
        let detector = ShorelineDetector::new(config);

        // Wave height 1.0, depth 1.0 => H/d = 1.0 > 0.78 => breaking
        assert!(detector.is_breaking_zone([0.0, 0.0, 0.0], 1.0, 1.0));

        // Wave height 0.5, depth 2.0 => H/d = 0.25 < 0.78 => not breaking
        assert!(!detector.is_breaking_zone([0.0, 0.0, 0.0], 0.5, 2.0));
    }

    // Test 18: Breaking zone zero depth
    #[test]
    fn test_breaking_zone_zero_depth() {
        let config = ShorelineConfig::default();
        let detector = ShorelineDetector::new(config);
        assert!(!detector.is_breaking_zone([0.0, 0.0, 0.0], 1.0, 0.0));
    }

    // Test 19: Shore normal default
    #[test]
    fn test_shore_normal_default() {
        let config = ShorelineConfig::default();
        let detector = ShorelineDetector::new(config);
        let normal = detector.get_shore_normal([0.0, 0.0, 0.0]);
        // Default when no cache
        assert!(approx_eq(normal[2], 1.0));
    }

    // Test 20: Cache clearing
    #[test]
    fn test_cache_clearing() {
        let config = ShorelineConfig::default();
        let mut detector = ShorelineDetector::new(config);

        let heightmap: Vec<f32> = (0..16).map(|i| (i % 4) as f32 * 0.5).collect();
        detector.find_shoreline(&heightmap, 4, 4, 1.0, [0.0, 0.0], 0.75);
        assert!(!detector.cached_points().is_empty());

        detector.clear_cache();
        assert!(detector.cached_points().is_empty());
    }

    // -----------------------------------------------------------------------
    // WaveBreaking Tests
    // -----------------------------------------------------------------------

    // Test 21: Wave breaking creation
    #[test]
    fn test_wave_breaking_creation() {
        let config = ShorelineConfig::default();
        let breaking = WaveBreaking::new(config);
        assert!(approx_eq(breaking.config().wave_break_depth, MCCOWAN_RATIO));
    }

    // Test 22: Break point computation - breaking
    #[test]
    fn test_break_point_breaking() {
        let config = ShorelineConfig::default();
        let breaking = WaveBreaking::new(config);

        // Wave height 1.0 breaks at depth ~1.28m (1.0/0.78)
        // At depth 1.0m, should be breaking
        let bp = breaking.compute_break_point(1.0, 10.0, 1.0);
        assert!(bp.is_some());
    }

    // Test 23: Break point computation - not breaking
    #[test]
    fn test_break_point_not_breaking() {
        let config = ShorelineConfig::default();
        let breaking = WaveBreaking::new(config);

        // Wave height 0.5 breaks at depth ~0.64m
        // At depth 2.0m, should not be breaking
        let bp = breaking.compute_break_point(0.5, 10.0, 2.0);
        assert!(bp.is_none());
    }

    // Test 24: Break point zero wave height
    #[test]
    fn test_break_point_zero_wave() {
        let config = ShorelineConfig::default();
        let breaking = WaveBreaking::new(config);
        let bp = breaking.compute_break_point(0.0, 10.0, 1.0);
        assert!(bp.is_none());
    }

    // Test 25: Break point zero depth
    #[test]
    fn test_break_point_zero_depth() {
        let config = ShorelineConfig::default();
        let breaking = WaveBreaking::new(config);
        let bp = breaking.compute_break_point(1.0, 10.0, 0.0);
        assert!(bp.is_none());
    }

    // Test 26: Breaking intensity calculation
    #[test]
    fn test_breaking_intensity() {
        let config = ShorelineConfig::default();
        let breaking = WaveBreaking::new(config);

        // H/d = 1.0/0.78 = 1.28 => excess = (1.28 - 0.78)/0.78 = 0.64
        let intensity = breaking.breaking_intensity(0.78, 1.0);
        assert!(intensity > 0.0 && intensity <= 1.0);
    }

    // Test 27: Breaking intensity zero depth
    #[test]
    fn test_breaking_intensity_zero() {
        let config = ShorelineConfig::default();
        let breaking = WaveBreaking::new(config);
        let intensity = breaking.breaking_intensity(0.0, 1.0);
        assert!(approx_eq(intensity, 0.0));
    }

    // Test 28: Break type classification - spilling
    #[test]
    fn test_break_type_spilling() {
        let config = ShorelineConfig::default();
        let breaking = WaveBreaking::new(config);

        // Gentle slope, steep wave => spilling
        let break_type = breaking.plunge_vs_spill(0.02, 0.04);
        assert_eq!(break_type, BreakType::Spilling);
    }

    // Test 29: Break type classification - plunging
    #[test]
    fn test_break_type_plunging() {
        let config = ShorelineConfig::default();
        let breaking = WaveBreaking::new(config);

        // Steep slope, low steepness => plunging
        let break_type = breaking.plunge_vs_spill(0.3, 0.01);
        assert_eq!(break_type, BreakType::Plunging);
    }

    // Test 30: Break type foam multiplier
    #[test]
    fn test_break_type_foam_multiplier() {
        assert!(BreakType::Plunging.foam_multiplier() > BreakType::Spilling.foam_multiplier());
        assert!(BreakType::Plunging.foam_multiplier() == 1.0);
    }

    // Test 31: Break type spray intensity
    #[test]
    fn test_break_type_spray() {
        assert!(BreakType::Plunging.spray_intensity() > BreakType::Surging.spray_intensity());
    }

    // Test 32: Foam spawn rate
    #[test]
    fn test_foam_spawn_rate() {
        let config = ShorelineConfig::default();
        let breaking = WaveBreaking::new(config);

        let rate_low = breaking.get_foam_spawn_rate(0.1);
        let rate_high = breaking.get_foam_spawn_rate(1.0);
        assert!(rate_high > rate_low);
        assert!(approx_eq(rate_high, 10.0));
    }

    // Test 33: Wave runup calculation
    #[test]
    fn test_wave_runup() {
        let config = ShorelineConfig::default();
        let breaking = WaveBreaking::new(config);

        let runup = breaking.wave_runup(0.1, 1.0, 50.0);
        // R = 0.35 * 0.1 * sqrt(50) = 0.247
        assert!(approx_eq_loose(runup, 0.247, 0.01));
    }

    // Test 34: Wave runup zero values
    #[test]
    fn test_wave_runup_zero() {
        let config = ShorelineConfig::default();
        let breaking = WaveBreaking::new(config);

        assert!(approx_eq(breaking.wave_runup(0.1, 0.0, 10.0), 0.0));
        assert!(approx_eq(breaking.wave_runup(0.1, 1.0, 0.0), 0.0));
    }

    // -----------------------------------------------------------------------
    // WetSand Tests
    // -----------------------------------------------------------------------

    // Test 35: Wet sand creation
    #[test]
    fn test_wet_sand_creation() {
        let config = ShorelineConfig::default();
        let wet_sand = WetSand::new(config);
        assert!(approx_eq(
            wet_sand.config().wet_sand_distance,
            DEFAULT_WET_SAND_DISTANCE
        ));
    }

    // Test 36: Wetness underwater
    #[test]
    fn test_wetness_underwater() {
        let config = ShorelineConfig::default();
        let wet_sand = WetSand::new(config);
        let wetness = wet_sand.compute_wetness(-1.0, 0.0, 0.0);
        assert!(approx_eq(wetness, 1.0));
    }

    // Test 37: Wetness far from shore
    #[test]
    fn test_wetness_far() {
        let config = ShorelineConfig::default();
        let wet_sand = WetSand::new(config);
        let wetness = wet_sand.compute_wetness(100.0, 0.0, 0.0);
        assert!(approx_eq(wetness, 0.0));
    }

    // Test 38: Wetness gradient
    #[test]
    fn test_wetness_gradient() {
        let config = ShorelineConfig::default();
        let wet_sand = WetSand::new(config);

        let wetness_near = wet_sand.compute_wetness(0.5, 0.0, 0.0);
        let wetness_mid = wet_sand.compute_wetness(1.5, 0.0, 0.0);
        let wetness_far = wet_sand.compute_wetness(2.5, 0.0, 0.0);

        assert!(wetness_near > wetness_mid);
        assert!(wetness_mid > wetness_far);
    }

    // Test 39: Wet sand albedo multiplier
    #[test]
    fn test_wet_sand_albedo() {
        let config = ShorelineConfig::default();
        let wet_sand = WetSand::new(config);

        let dry = wet_sand.wet_sand_albedo_multiplier(0.0);
        let wet = wet_sand.wet_sand_albedo_multiplier(1.0);

        assert!(approx_eq(dry, 1.0));
        assert!(approx_eq(wet, 0.5)); // 50% darker when fully wet
    }

    // Test 40: Wet sand roughness
    #[test]
    fn test_wet_sand_roughness() {
        let config = ShorelineConfig::default();
        let wet_sand = WetSand::new(config);

        let dry = wet_sand.wet_sand_roughness(0.0);
        let wet = wet_sand.wet_sand_roughness(1.0);

        assert!(dry > wet); // Wet sand is smoother
        assert!(approx_eq(dry, 0.8));
        assert!(approx_eq(wet, 0.3));
    }

    // Test 41: Puddle mask underwater
    #[test]
    fn test_puddle_mask_underwater() {
        let config = ShorelineConfig::default();
        let wet_sand = WetSand::new(config);
        let puddle = wet_sand.puddle_mask([0.0, 0.0, 0.0], -1.0);
        assert!(approx_eq(puddle, 0.0));
    }

    // Test 42: Puddle mask far from shore
    #[test]
    fn test_puddle_mask_far() {
        let config = ShorelineConfig::default();
        let wet_sand = WetSand::new(config);
        let puddle = wet_sand.puddle_mask([0.0, 0.0, 0.0], 100.0);
        assert!(approx_eq(puddle, 0.0));
    }

    // -----------------------------------------------------------------------
    // TideSimulation Tests
    // -----------------------------------------------------------------------

    // Test 43: Tide creation
    #[test]
    fn test_tide_creation() {
        let tide = TideSimulation::new(0.5, 43200.0);
        assert!(approx_eq(tide.amplitude(), 0.5));
        assert!(approx_eq(tide.period(), 43200.0));
    }

    // Test 44: Tide default
    #[test]
    fn test_tide_default() {
        let tide = TideSimulation::default();
        assert!(approx_eq(tide.amplitude(), DEFAULT_TIDE_AMPLITUDE));
        assert!(approx_eq(tide.period(), DEFAULT_TIDE_PERIOD));
    }

    // Test 45: Tide update
    #[test]
    fn test_tide_update() {
        let mut tide = TideSimulation::new(0.5, 100.0);
        tide.update(25.0); // Quarter period
        assert!(approx_eq(tide.phase(), PI / 2.0));
    }

    // Test 46: Tide water level
    #[test]
    fn test_tide_water_level() {
        let mut tide = TideSimulation::new(0.5, 100.0);

        // At time 0, sin(0) = 0
        assert!(approx_eq(tide.get_water_level(10.0), 10.0));

        // At quarter period, sin(PI/2) = 1
        tide.update(25.0);
        assert!(approx_eq(tide.get_water_level(10.0), 10.5));
    }

    // Test 47: Tide velocity
    #[test]
    fn test_tide_velocity() {
        let mut tide = TideSimulation::new(0.5, 100.0);

        // At time 0, derivative cos(0) = 1, rising tide
        let vel_0 = tide.get_tide_velocity();
        assert!(vel_0 > 0.0);

        // At half period, derivative cos(PI) = -1, falling tide
        tide.update(50.0);
        let vel_half = tide.get_tide_velocity();
        assert!(vel_half < 0.0);
    }

    // Test 48: High tide detection
    #[test]
    fn test_high_tide() {
        let mut tide = TideSimulation::new(0.5, 100.0);

        // At time 0, not high tide
        assert!(!tide.is_high_tide());

        // At quarter period, high tide
        tide.update(25.0);
        assert!(tide.is_high_tide());
    }

    // Test 49: Low tide detection
    #[test]
    fn test_low_tide() {
        let mut tide = TideSimulation::new(0.5, 100.0);

        // At 3/4 period, low tide (3*PI/2)
        tide.update(75.0);
        assert!(tide.is_low_tide());
    }

    // Test 50: Tide 12-hour cycle
    #[test]
    fn test_tide_12_hour_cycle() {
        let mut tide = TideSimulation::new(0.5, DEFAULT_TIDE_PERIOD);

        let level_start = tide.get_water_level(0.0);

        // After full cycle, should return to same level
        tide.update(DEFAULT_TIDE_PERIOD);
        let level_end = tide.get_water_level(0.0);

        assert!(approx_eq_loose(level_start, level_end, 0.001));
    }

    // Test 51: Tide reset
    #[test]
    fn test_tide_reset() {
        let mut tide = TideSimulation::new(0.5, 100.0);
        tide.update(50.0);
        tide.reset();
        assert!(approx_eq(tide.time(), 0.0));
        assert!(approx_eq(tide.phase(), 0.0));
    }

    // Test 52: Tide set time
    #[test]
    fn test_tide_set_time() {
        let mut tide = TideSimulation::new(0.5, 100.0);
        tide.set_time(50.0);
        assert!(approx_eq(tide.time(), 50.0));
        assert!(approx_eq(tide.phase(), PI));
    }

    // -----------------------------------------------------------------------
    // ShorelineInteraction Tests
    // -----------------------------------------------------------------------

    // Test 53: Interaction creation
    #[test]
    fn test_interaction_creation() {
        let config = ShorelineConfig::default();
        let interaction = ShorelineInteraction::new(config);
        assert!(approx_eq(interaction.time(), 0.0));
    }

    // Test 54: Interaction default
    #[test]
    fn test_interaction_default() {
        let interaction = ShorelineInteraction::default();
        assert!(approx_eq(
            interaction.config().wave_break_depth,
            MCCOWAN_RATIO
        ));
    }

    // Test 55: Interaction update
    #[test]
    fn test_interaction_update() {
        let mut interaction = ShorelineInteraction::default();
        interaction.update(1.0);
        assert!(approx_eq(interaction.time(), 1.0));
    }

    // Test 56: Interaction water level
    #[test]
    fn test_interaction_water_level() {
        let mut interaction = ShorelineInteraction::default();
        interaction.set_base_water_level(10.0);

        // At time 0, tide offset is 0
        assert!(approx_eq(interaction.current_water_level(), 10.0));
    }

    // Test 57: Shore foam gradient
    #[test]
    fn test_shore_foam_gradient() {
        let interaction = ShorelineInteraction::default();

        // Foam should decrease with distance from shore
        let foam_near = interaction.get_shore_foam([0.0, 0.1, 0.0]);
        let foam_far = interaction.get_shore_foam([0.0, 10.0, 0.0]);

        assert!(foam_near >= foam_far);
    }

    // Test 58: Wet sand factor
    #[test]
    fn test_wet_sand_factor() {
        let interaction = ShorelineInteraction::default();

        let wet_near = interaction.get_wet_sand_factor([0.0, 0.5, 0.0]);
        let wet_far = interaction.get_wet_sand_factor([0.0, 10.0, 0.0]);

        assert!(wet_near > wet_far);
    }

    // Test 59: Wave modification deep water
    #[test]
    fn test_wave_modification_deep() {
        let interaction = ShorelineInteraction::default();
        let mod_result = interaction.get_wave_modification([0.0, 0.0, 0.0], 20.0);

        // Deep water: minimal modification
        assert!(approx_eq(mod_result.height_scale, 1.0));
        assert!(!mod_result.is_breaking);
    }

    // Test 60: Wave modification shallow water
    #[test]
    fn test_wave_modification_shallow() {
        let interaction = ShorelineInteraction::default();
        // Use depth 5.0 which is shallow enough for shoaling but deep enough to not break
        // shoaling_coeff = sqrt(10/5) = sqrt(2) = 1.41
        // effective_height = 1.0 * 1.41 = 1.41
        // H/d = 1.41/5 = 0.28 < 0.78 => not breaking
        let mod_result = interaction.get_wave_modification([0.0, 0.0, 0.0], 5.0);

        // Shallow water: shoaling increases height
        assert!(mod_result.height_scale > 1.0);
        assert!(!mod_result.is_breaking);
    }

    // Test 61: Terrain wetness application
    #[test]
    fn test_terrain_wetness_application() {
        let interaction = ShorelineInteraction::default();

        let (albedo_near, roughness_near) = interaction.apply_terrain_wetness(0.5);
        let (albedo_far, roughness_far) = interaction.apply_terrain_wetness(10.0);

        // Near shore: darker, smoother
        assert!(albedo_near < albedo_far);
        assert!(roughness_near < roughness_far);
    }

    // Test 62: Component accessors
    #[test]
    fn test_component_accessors() {
        let interaction = ShorelineInteraction::default();

        let _ = interaction.detector();
        let _ = interaction.wave_breaking();
        let _ = interaction.wet_sand();
        let _ = interaction.tide();
        let _ = interaction.config();
    }

    // Test 63: Mutable accessors
    #[test]
    fn test_mutable_accessors() {
        let mut interaction = ShorelineInteraction::default();

        interaction.detector_mut().clear_cache();
        interaction.tide_mut().reset();
    }

    // -----------------------------------------------------------------------
    // Edge Cases and Integration Tests
    // -----------------------------------------------------------------------

    // Test 64: Cliff shoreline (vertical)
    #[test]
    fn test_cliff_shoreline() {
        let config = ShorelineConfig::default();
        let mut detector = ShorelineDetector::new(config);

        // Cliff: sudden jump from 0 to 10
        let heightmap = vec![0.0, 0.0, 10.0, 10.0, 0.0, 0.0, 10.0, 10.0, 0.0, 0.0, 10.0, 10.0, 0.0, 0.0, 10.0, 10.0];

        let points = detector.find_shoreline(&heightmap, 4, 4, 1.0, [0.0, 0.0], 5.0);

        // Should detect shoreline at cliff edge
        assert!(!points.is_empty());
    }

    // Test 65: Island (closed shoreline)
    #[test]
    fn test_island_shoreline() {
        let config = ShorelineConfig::default();
        let mut detector = ShorelineDetector::new(config);

        // Island: high in center, low at edges
        let heightmap = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];

        let points = detector.find_shoreline(&heightmap, 4, 4, 1.0, [0.0, 0.0], 1.0);

        // Should detect closed shoreline around island
        assert!(!points.is_empty());
    }

    // Test 66: Tidal pool (inland water)
    #[test]
    fn test_tidal_pool() {
        let config = ShorelineConfig::default();
        let mut detector = ShorelineDetector::new(config);

        // Pool: low spot surrounded by high terrain
        let heightmap = vec![
            2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        ];

        let points = detector.find_shoreline(&heightmap, 4, 4, 1.0, [0.0, 0.0], 1.0);

        // Should detect shoreline around pool
        assert!(!points.is_empty());
    }

    // Test 67: Full cycle integration
    #[test]
    fn test_full_cycle_integration() {
        let mut interaction = ShorelineInteraction::default();
        interaction.set_base_water_level(5.0);

        // Simulate one full tidal cycle
        let dt = 100.0;
        let steps = (DEFAULT_TIDE_PERIOD / dt) as usize;

        let mut min_level = f32::MAX;
        let mut max_level = f32::MIN;

        for _ in 0..steps {
            interaction.update(dt);
            let level = interaction.current_water_level();
            min_level = min_level.min(level);
            max_level = max_level.max(level);
        }

        // Should see full tidal range
        let range = max_level - min_level;
        assert!(approx_eq_loose(range, DEFAULT_TIDE_AMPLITUDE * 2.0, 0.1));
    }

    // Test 68: Shoreline point arc length
    #[test]
    fn test_shoreline_point_arc_length() {
        let point = ShorelinePoint::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.1);
        let point_with_arc = point.with_arc_length(5.0);
        assert!(approx_eq(point_with_arc.arc_length, 5.0));
    }

    // Test 69: Break type default
    #[test]
    fn test_break_type_default() {
        let bt = BreakType::default();
        assert_eq!(bt, BreakType::Spilling);
    }

    // Test 70: Smoothstep function
    #[test]
    fn test_smoothstep() {
        assert!(approx_eq(smoothstep(0.0, 1.0, 0.0), 0.0));
        assert!(approx_eq(smoothstep(0.0, 1.0, 1.0), 1.0));
        assert!(approx_eq(smoothstep(0.0, 1.0, 0.5), 0.5));
    }

    // Test 71: Normalize vector
    #[test]
    fn test_normalize_vector() {
        let v = normalize_vec3([3.0, 0.0, 4.0]);
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        assert!(approx_eq(len, 1.0));
    }

    // Test 72: Normalize zero vector
    #[test]
    fn test_normalize_zero_vector() {
        let v = normalize_vec3([0.0, 0.0, 0.0]);
        // Should return up vector
        assert!(approx_eq(v[1], 1.0));
    }

    // Test 73: Wave modification zero depth
    #[test]
    fn test_wave_modification_zero_depth() {
        let interaction = ShorelineInteraction::default();
        let mod_result = interaction.get_wave_modification([0.0, 0.0, 0.0], 0.0);
        assert!(approx_eq(mod_result.height_scale, 0.0));
    }

    // Test 74: Collapsing break type
    #[test]
    fn test_break_type_collapsing() {
        let config = ShorelineConfig::default();
        let breaking = WaveBreaking::new(config);

        // Intermediate slope
        let break_type = breaking.plunge_vs_spill(0.1, 0.02);
        assert_eq!(break_type, BreakType::Collapsing);
    }

    // Test 75: Surging break type
    #[test]
    fn test_break_type_surging() {
        let config = ShorelineConfig::default();
        let breaking = WaveBreaking::new(config);

        // Very steep slope, low steepness
        let break_type = breaking.plunge_vs_spill(0.5, 0.005);
        assert_eq!(break_type, BreakType::Surging);
    }

    // Test 76: Shore foam underwater
    #[test]
    fn test_shore_foam_underwater() {
        let interaction = ShorelineInteraction::default();
        let foam = interaction.get_shore_foam([0.0, -2.0, 0.0]);
        // Should have some foam near surface
        assert!(foam >= 0.0);
        assert!(foam < 1.0);
    }

    // Test 77: Pseudo noise deterministic
    #[test]
    fn test_pseudo_noise_deterministic() {
        let n1 = pseudo_noise_2d(1.0, 2.0);
        let n2 = pseudo_noise_2d(1.0, 2.0);
        assert!(approx_eq(n1, n2));
    }

    // Test 78: Pseudo noise range
    #[test]
    fn test_pseudo_noise_range() {
        for i in 0..100 {
            let n = pseudo_noise_2d(i as f32 * 0.1, i as f32 * 0.2);
            // fract() returns values in [0, 1), but floating point precision
            // can cause edge cases, so we allow a small tolerance
            assert!(n >= 0.0 && n <= 1.0, "Noise value {} out of range", n);
        }
    }
}
