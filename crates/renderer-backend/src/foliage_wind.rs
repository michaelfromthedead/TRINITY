//! Foliage Wind Animation System (T-ENV-3.11)
//!
//! This module provides realistic wind animation for foliage by integrating with
//! the weather map system. It implements:
//!
//! - Wind field sampling from weather map
//! - Procedural wind gusts and turbulence
//! - Hierarchical wind (trunk -> branch -> leaf)
//! - Pivot-based bending for grass and plants
//! - Vertex displacement shader integration
//! - Detail oscillation (high-frequency flutter)
//!
//! # Architecture
//!
//! The system uses a multi-layer approach:
//!
//! 1. **Base Wind**: Global wind from weather map
//! 2. **Gusts**: Time-varying sinusoidal modulation
//! 3. **Turbulence**: Spatial noise for local variation
//! 4. **Hierarchical Bending**: Different bend amounts for trunk/branch/leaf
//! 5. **Detail Flutter**: High-frequency oscillation for leaves
//!
//! # GPU Integration
//!
//! All structs are `repr(C)` with `bytemuck::Pod/Zeroable` for direct GPU upload.
//! The `FoliageWindUniforms` struct is designed for efficient uniform buffer binding.
//!
//! # Algorithms
//!
//! - Gerstner-style wave displacement for grass sway
//! - Perlin/simplex noise for turbulence variation
//! - Phase offsets based on world position (prevents synchronized motion)
//! - LOD-based wind detail reduction
//! - Wind occlusion estimation from nearby geometry
//!
//! # References
//!
//! - GPU Gems 3: "Animated Grass with Instancing"
//! - "Real-time Realistic Rendering of Nature Scenes with Optimized GPU"
//! - Unreal Engine foliage shader documentation

use bytemuck::{Pod, Zeroable};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default wind speed (m/s).
pub const DEFAULT_WIND_SPEED: f32 = 5.0;

/// Default wind direction (radians, 0 = +X, pi/2 = +Z).
pub const DEFAULT_WIND_DIRECTION: f32 = 0.785; // 45 degrees

/// Default gust frequency (Hz).
pub const DEFAULT_GUST_FREQUENCY: f32 = 0.15;

/// Default gust strength (0-1).
pub const DEFAULT_GUST_STRENGTH: f32 = 0.3;

/// Default turbulence scale (world units).
pub const DEFAULT_TURBULENCE_SCALE: f32 = 50.0;

/// Default turbulence intensity (0-1).
pub const DEFAULT_TURBULENCE_INTENSITY: f32 = 0.2;

/// Default trunk bend factor.
pub const DEFAULT_TRUNK_BEND: f32 = 0.02;

/// Default branch bend factor.
pub const DEFAULT_BRANCH_BEND: f32 = 0.1;

/// Default leaf bend factor.
pub const DEFAULT_LEAF_BEND: f32 = 0.3;

/// Default detail oscillation frequency (Hz).
pub const DEFAULT_DETAIL_FREQUENCY: f32 = 2.5;

/// Default detail oscillation amplitude.
pub const DEFAULT_DETAIL_AMPLITUDE: f32 = 0.05;

/// Small epsilon for floating point comparisons.
pub const EPSILON: f32 = 1e-6;

/// Pi constant.
pub const PI: f32 = std::f32::consts::PI;

/// Two Pi constant.
pub const TAU: f32 = std::f32::consts::TAU;

/// Maximum wind speed before clamping (m/s).
pub const MAX_WIND_SPEED: f32 = 50.0;

/// Minimum wind update delta time.
pub const MIN_DELTA_TIME: f32 = 0.0001;

/// Golden ratio for phase distribution.
pub const GOLDEN_RATIO: f32 = 1.618033988749895;

/// Maximum LOD level for wind calculations.
pub const MAX_WIND_LOD: u8 = 4;

// ---------------------------------------------------------------------------
// FoliageType - Classification of foliage for wind behavior
// ---------------------------------------------------------------------------

/// Classification of foliage for wind behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
#[repr(u8)]
pub enum FoliageType {
    /// Grass - pivot at base, full bend.
    Grass = 0,

    /// Small plant - pivot at base, moderate bend.
    #[default]
    SmallPlant = 1,

    /// Bush - multi-pivot, moderate bend.
    Bush = 2,

    /// Small tree - trunk/branch/leaf hierarchy.
    SmallTree = 3,

    /// Large tree - reduced trunk movement.
    LargeTree = 4,

    /// Palm tree - frond-based animation.
    Palm = 5,

    /// Hanging foliage (vines, moss) - top-pivot.
    Hanging = 6,

    /// Aquatic plants - underwater wave motion.
    Aquatic = 7,
}

impl FoliageType {
    /// Create from u8 value (clamped to valid range).
    #[inline]
    pub fn from_u8(value: u8) -> Self {
        match value {
            0 => FoliageType::Grass,
            1 => FoliageType::SmallPlant,
            2 => FoliageType::Bush,
            3 => FoliageType::SmallTree,
            4 => FoliageType::LargeTree,
            5 => FoliageType::Palm,
            6 => FoliageType::Hanging,
            7 => FoliageType::Aquatic,
            _ => FoliageType::SmallPlant,
        }
    }

    /// Get the base bend stiffness for this foliage type (0 = stiff, 1 = flexible).
    #[inline]
    pub fn stiffness(&self) -> f32 {
        match self {
            FoliageType::Grass => 0.1,
            FoliageType::SmallPlant => 0.3,
            FoliageType::Bush => 0.4,
            FoliageType::SmallTree => 0.6,
            FoliageType::LargeTree => 0.85,
            FoliageType::Palm => 0.5,
            FoliageType::Hanging => 0.2,
            FoliageType::Aquatic => 0.15,
        }
    }

    /// Get the trunk bend multiplier.
    #[inline]
    pub fn trunk_bend_multiplier(&self) -> f32 {
        match self {
            FoliageType::Grass => 0.0,
            FoliageType::SmallPlant => 0.1,
            FoliageType::Bush => 0.2,
            FoliageType::SmallTree => 0.3,
            FoliageType::LargeTree => 0.1,
            FoliageType::Palm => 0.2,
            FoliageType::Hanging => 0.0,
            FoliageType::Aquatic => 0.0,
        }
    }

    /// Get the branch bend multiplier.
    #[inline]
    pub fn branch_bend_multiplier(&self) -> f32 {
        match self {
            FoliageType::Grass => 0.0,
            FoliageType::SmallPlant => 0.3,
            FoliageType::Bush => 0.5,
            FoliageType::SmallTree => 0.6,
            FoliageType::LargeTree => 0.4,
            FoliageType::Palm => 0.8,
            FoliageType::Hanging => 0.2,
            FoliageType::Aquatic => 0.3,
        }
    }

    /// Get the leaf bend multiplier.
    #[inline]
    pub fn leaf_bend_multiplier(&self) -> f32 {
        match self {
            FoliageType::Grass => 1.0,
            FoliageType::SmallPlant => 0.8,
            FoliageType::Bush => 0.7,
            FoliageType::SmallTree => 1.0,
            FoliageType::LargeTree => 0.9,
            FoliageType::Palm => 1.0,
            FoliageType::Hanging => 1.0,
            FoliageType::Aquatic => 0.8,
        }
    }

    /// Get whether this type uses pivot-based bending (vs. hierarchical).
    #[inline]
    pub fn uses_pivot_bending(&self) -> bool {
        matches!(
            self,
            FoliageType::Grass | FoliageType::SmallPlant | FoliageType::Hanging
        )
    }

    /// Get the detail flutter scale for this type.
    #[inline]
    pub fn detail_flutter_scale(&self) -> f32 {
        match self {
            FoliageType::Grass => 1.5,
            FoliageType::SmallPlant => 1.0,
            FoliageType::Bush => 0.8,
            FoliageType::SmallTree => 1.2,
            FoliageType::LargeTree => 0.6,
            FoliageType::Palm => 1.0,
            FoliageType::Hanging => 1.3,
            FoliageType::Aquatic => 0.5,
        }
    }

    /// Get name string for debugging.
    #[inline]
    pub fn name(&self) -> &'static str {
        match self {
            FoliageType::Grass => "grass",
            FoliageType::SmallPlant => "small_plant",
            FoliageType::Bush => "bush",
            FoliageType::SmallTree => "small_tree",
            FoliageType::LargeTree => "large_tree",
            FoliageType::Palm => "palm",
            FoliageType::Hanging => "hanging",
            FoliageType::Aquatic => "aquatic",
        }
    }
}

// ---------------------------------------------------------------------------
// FoliageWindUniforms - GPU uniform buffer for wind parameters
// ---------------------------------------------------------------------------

/// GPU-uploadable uniform buffer for foliage wind parameters.
///
/// This struct contains all parameters needed by the shader to calculate
/// wind displacement for foliage vertices.
///
/// # Memory Layout (96 bytes)
///
/// | Offset | Field               | Size     |
/// |--------|---------------------|----------|
/// | 0      | wind_direction      | 8 bytes  |
/// | 8      | wind_strength       | 4 bytes  |
/// | 12     | time                | 4 bytes  |
/// | 16     | gust_frequency      | 4 bytes  |
/// | 20     | gust_strength       | 4 bytes  |
/// | 24     | turbulence_scale    | 4 bytes  |
/// | 28     | turbulence_intensity| 4 bytes  |
/// | 32     | trunk_bend          | 4 bytes  |
/// | 36     | branch_bend         | 4 bytes  |
/// | 40     | leaf_bend           | 4 bytes  |
/// | 44     | detail_frequency    | 4 bytes  |
/// | 48     | detail_amplitude    | 4 bytes  |
/// | 52     | phase_offset_scale  | 4 bytes  |
/// | 56     | lod_detail_falloff  | 4 bytes  |
/// | 60     | wind_occlusion      | 4 bytes  |
/// | 64     | gerstner_params     | 16 bytes |
/// | 80     | _padding            | 16 bytes |
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct FoliageWindUniforms {
    /// Wind direction (normalized XZ).
    pub wind_direction: [f32; 2],
    /// Wind strength (m/s).
    pub wind_strength: f32,
    /// Current time for animation (seconds).
    pub time: f32,

    /// Gust frequency (Hz).
    pub gust_frequency: f32,
    /// Gust strength multiplier (0-1).
    pub gust_strength: f32,
    /// Turbulence spatial scale (world units).
    pub turbulence_scale: f32,
    /// Turbulence intensity (0-1).
    pub turbulence_intensity: f32,

    /// Trunk bend factor.
    pub trunk_bend: f32,
    /// Branch bend factor.
    pub branch_bend: f32,
    /// Leaf bend factor.
    pub leaf_bend: f32,
    /// Detail oscillation frequency (Hz).
    pub detail_frequency: f32,

    /// Detail oscillation amplitude.
    pub detail_amplitude: f32,
    /// Phase offset scale based on world position.
    pub phase_offset_scale: f32,
    /// LOD detail falloff (higher = faster falloff).
    pub lod_detail_falloff: f32,
    /// Wind occlusion factor (0 = full occlusion, 1 = no occlusion).
    pub wind_occlusion: f32,

    /// Gerstner wave parameters: [wavelength, steepness, speed, _].
    pub gerstner_params: [f32; 4],

    /// Padding for 16-byte alignment (96 bytes total).
    pub _padding: [f32; 4],
}

// Size assertion for GPU compatibility
const _: () = assert!(std::mem::size_of::<FoliageWindUniforms>() == 96);

impl Default for FoliageWindUniforms {
    fn default() -> Self {
        let dir = DEFAULT_WIND_DIRECTION;
        Self {
            wind_direction: [dir.cos(), dir.sin()],
            wind_strength: DEFAULT_WIND_SPEED,
            time: 0.0,

            gust_frequency: DEFAULT_GUST_FREQUENCY,
            gust_strength: DEFAULT_GUST_STRENGTH,
            turbulence_scale: DEFAULT_TURBULENCE_SCALE,
            turbulence_intensity: DEFAULT_TURBULENCE_INTENSITY,

            trunk_bend: DEFAULT_TRUNK_BEND,
            branch_bend: DEFAULT_BRANCH_BEND,
            leaf_bend: DEFAULT_LEAF_BEND,
            detail_frequency: DEFAULT_DETAIL_FREQUENCY,

            detail_amplitude: DEFAULT_DETAIL_AMPLITUDE,
            phase_offset_scale: 0.01,
            lod_detail_falloff: 2.0,
            wind_occlusion: 1.0,

            gerstner_params: [2.0, 0.5, 1.0, 0.0],

            _padding: [0.0; 4],
        }
    }
}

impl FoliageWindUniforms {
    /// Create new uniforms with default values.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create uniforms from wind velocity.
    pub fn from_velocity(velocity: [f32; 2]) -> Self {
        let strength = (velocity[0] * velocity[0] + velocity[1] * velocity[1]).sqrt();
        let (dir_x, dir_y) = if strength > EPSILON {
            (velocity[0] / strength, velocity[1] / strength)
        } else {
            (1.0, 0.0)
        };

        Self {
            wind_direction: [dir_x, dir_y],
            wind_strength: strength,
            ..Default::default()
        }
    }

    /// Create uniforms from speed and direction.
    pub fn from_speed_direction(speed: f32, direction_radians: f32) -> Self {
        Self {
            wind_direction: [direction_radians.cos(), direction_radians.sin()],
            wind_strength: speed.clamp(0.0, MAX_WIND_SPEED),
            ..Default::default()
        }
    }

    /// Set wind from speed and direction.
    #[inline]
    pub fn set_wind(&mut self, speed: f32, direction_radians: f32) {
        self.wind_strength = speed.clamp(0.0, MAX_WIND_SPEED);
        self.wind_direction = [direction_radians.cos(), direction_radians.sin()];
    }

    /// Set wind from velocity vector.
    #[inline]
    pub fn set_wind_velocity(&mut self, velocity: [f32; 2]) {
        self.wind_strength =
            (velocity[0] * velocity[0] + velocity[1] * velocity[1]).sqrt().min(MAX_WIND_SPEED);
        if self.wind_strength > EPSILON {
            self.wind_direction = [
                velocity[0] / self.wind_strength,
                velocity[1] / self.wind_strength,
            ];
        }
    }

    /// Update time for animation.
    #[inline]
    pub fn update_time(&mut self, delta_seconds: f32) {
        self.time += delta_seconds.max(MIN_DELTA_TIME);
    }

    /// Set hierarchical bend factors.
    #[inline]
    pub fn set_bend_factors(&mut self, trunk: f32, branch: f32, leaf: f32) {
        self.trunk_bend = trunk.clamp(0.0, 1.0);
        self.branch_bend = branch.clamp(0.0, 1.0);
        self.leaf_bend = leaf.clamp(0.0, 1.0);
    }

    /// Set bend factors from foliage type.
    #[inline]
    pub fn set_foliage_type(&mut self, foliage_type: FoliageType) {
        self.trunk_bend = DEFAULT_TRUNK_BEND * foliage_type.trunk_bend_multiplier();
        self.branch_bend = DEFAULT_BRANCH_BEND * foliage_type.branch_bend_multiplier();
        self.leaf_bend = DEFAULT_LEAF_BEND * foliage_type.leaf_bend_multiplier();
        self.detail_amplitude = DEFAULT_DETAIL_AMPLITUDE * foliage_type.detail_flutter_scale();
    }

    /// Set gust parameters.
    #[inline]
    pub fn set_gust(&mut self, frequency: f32, strength: f32) {
        self.gust_frequency = frequency.max(0.0);
        self.gust_strength = strength.clamp(0.0, 1.0);
    }

    /// Set turbulence parameters.
    #[inline]
    pub fn set_turbulence(&mut self, scale: f32, intensity: f32) {
        self.turbulence_scale = scale.max(0.1);
        self.turbulence_intensity = intensity.clamp(0.0, 1.0);
    }

    /// Set detail oscillation parameters.
    #[inline]
    pub fn set_detail(&mut self, frequency: f32, amplitude: f32) {
        self.detail_frequency = frequency.max(0.0);
        self.detail_amplitude = amplitude.clamp(0.0, 1.0);
    }

    /// Set Gerstner wave parameters.
    #[inline]
    pub fn set_gerstner(&mut self, wavelength: f32, steepness: f32, speed: f32) {
        self.gerstner_params = [wavelength.max(0.1), steepness.clamp(0.0, 1.0), speed, 0.0];
    }

    /// Get wind velocity vector.
    #[inline]
    pub fn wind_velocity(&self) -> [f32; 2] {
        [
            self.wind_direction[0] * self.wind_strength,
            self.wind_direction[1] * self.wind_strength,
        ]
    }

    /// Get wind direction in radians.
    #[inline]
    pub fn wind_direction_radians(&self) -> f32 {
        self.wind_direction[1].atan2(self.wind_direction[0])
    }

    /// Calculate gust factor at current time.
    #[inline]
    pub fn gust_factor(&self) -> f32 {
        if self.gust_strength < EPSILON {
            return 1.0;
        }
        let gust_phase = self.time * self.gust_frequency * TAU;
        1.0 + self.gust_strength * (gust_phase.sin() * 0.5 + 0.5)
    }

    /// Validate uniform values.
    pub fn validate(&self) -> bool {
        // Check wind direction is normalized
        let dir_len = (self.wind_direction[0] * self.wind_direction[0]
            + self.wind_direction[1] * self.wind_direction[1])
        .sqrt();

        (dir_len - 1.0).abs() < 0.01
            && self.wind_strength >= 0.0
            && self.wind_strength <= MAX_WIND_SPEED
            && self.gust_frequency >= 0.0
            && self.gust_strength >= 0.0
            && self.gust_strength <= 1.0
            && self.turbulence_scale > 0.0
            && self.turbulence_intensity >= 0.0
            && self.turbulence_intensity <= 1.0
            && self.trunk_bend >= 0.0
            && self.branch_bend >= 0.0
            && self.leaf_bend >= 0.0
            && self.detail_frequency >= 0.0
            && self.detail_amplitude >= 0.0
            && self.wind_occlusion >= 0.0
            && self.wind_occlusion <= 1.0
    }
}

// ---------------------------------------------------------------------------
// WindSample - Result of sampling wind at a position
// ---------------------------------------------------------------------------

/// Result of sampling wind conditions at a world position.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct WindSample {
    /// Wind velocity (XZ, m/s).
    pub velocity: [f32; 2],
    /// Wind strength with gusts (m/s).
    pub strength: f32,
    /// Wind direction (radians).
    pub direction: f32,
    /// Turbulence offset (XZ).
    pub turbulence: [f32; 2],
    /// Phase offset for this position.
    pub phase_offset: f32,
    /// Effective gust factor.
    pub gust_factor: f32,
}

impl WindSample {
    /// Create a calm wind sample.
    #[inline]
    pub fn calm() -> Self {
        Self {
            velocity: [0.0; 2],
            strength: 0.0,
            direction: 0.0,
            turbulence: [0.0; 2],
            phase_offset: 0.0,
            gust_factor: 1.0,
        }
    }

    /// Create a wind sample from velocity.
    pub fn from_velocity(velocity: [f32; 2]) -> Self {
        let strength = (velocity[0] * velocity[0] + velocity[1] * velocity[1]).sqrt();
        let direction = velocity[1].atan2(velocity[0]);
        Self {
            velocity,
            strength,
            direction,
            turbulence: [0.0; 2],
            phase_offset: 0.0,
            gust_factor: 1.0,
        }
    }

    /// Get total velocity including turbulence.
    #[inline]
    pub fn total_velocity(&self) -> [f32; 2] {
        [
            self.velocity[0] + self.turbulence[0],
            self.velocity[1] + self.turbulence[1],
        ]
    }

    /// Get effective strength including gust factor.
    #[inline]
    pub fn effective_strength(&self) -> f32 {
        self.strength * self.gust_factor
    }

    /// Interpolate between two wind samples.
    pub fn lerp(a: &Self, b: &Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        let inv_t = 1.0 - t;

        Self {
            velocity: [
                a.velocity[0] * inv_t + b.velocity[0] * t,
                a.velocity[1] * inv_t + b.velocity[1] * t,
            ],
            strength: a.strength * inv_t + b.strength * t,
            direction: lerp_angle(a.direction, b.direction, t),
            turbulence: [
                a.turbulence[0] * inv_t + b.turbulence[0] * t,
                a.turbulence[1] * inv_t + b.turbulence[1] * t,
            ],
            phase_offset: a.phase_offset * inv_t + b.phase_offset * t,
            gust_factor: a.gust_factor * inv_t + b.gust_factor * t,
        }
    }
}

// ---------------------------------------------------------------------------
// PivotPoint - Pivot configuration for bending
// ---------------------------------------------------------------------------

/// Pivot point configuration for bending calculations.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct PivotPoint {
    /// Pivot position (local space XYZ).
    pub position: [f32; 3],
    /// Bend influence radius.
    pub radius: f32,
    /// Bend stiffness (0 = flexible, 1 = stiff).
    pub stiffness: f32,
    /// Bend weight (0-1).
    pub weight: f32,
    /// Height influence (how much height affects bend).
    pub height_influence: f32,
    /// Padding.
    pub _padding: f32,
}

impl Default for PivotPoint {
    fn default() -> Self {
        Self::at_base()
    }
}

impl PivotPoint {
    /// Create a pivot at the base (ground level).
    #[inline]
    pub fn at_base() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            radius: 1.0,
            stiffness: 0.3,
            weight: 1.0,
            height_influence: 1.0,
            _padding: 0.0,
        }
    }

    /// Create a pivot at a specific height.
    #[inline]
    pub fn at_height(height: f32) -> Self {
        Self {
            position: [0.0, height, 0.0],
            radius: 1.0,
            stiffness: 0.3,
            weight: 1.0,
            height_influence: 1.0,
            _padding: 0.0,
        }
    }

    /// Create a pivot for hanging foliage (top pivot).
    #[inline]
    pub fn at_top(height: f32) -> Self {
        Self {
            position: [0.0, height, 0.0],
            radius: 1.0,
            stiffness: 0.2,
            weight: 1.0,
            height_influence: -1.0, // Inverse height influence for hanging
            _padding: 0.0,
        }
    }

    /// Calculate bend amount for a vertex at given height.
    #[inline]
    pub fn bend_amount(&self, vertex_height: f32) -> f32 {
        let height_diff = (vertex_height - self.position[1]).max(0.0);
        let normalized_height = (height_diff / self.radius.max(EPSILON)).min(1.0);
        let bend = normalized_height * self.height_influence * self.weight;
        bend * (1.0 - self.stiffness)
    }

    /// Validate pivot configuration.
    #[inline]
    pub fn validate(&self) -> bool {
        self.radius > 0.0
            && self.stiffness >= 0.0
            && self.stiffness <= 1.0
            && self.weight >= 0.0
            && self.weight <= 1.0
    }
}

// ---------------------------------------------------------------------------
// HierarchicalBend - Hierarchical bending configuration
// ---------------------------------------------------------------------------

/// Hierarchical bending configuration for trees.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct HierarchicalBend {
    /// Trunk bend origin (local Y).
    pub trunk_origin: f32,
    /// Branch bend origin (relative to trunk).
    pub branch_origin: f32,
    /// Leaf attachment point (relative to branch).
    pub leaf_origin: f32,
    /// _padding.
    pub _padding: f32,

    /// Trunk bend strength.
    pub trunk_strength: f32,
    /// Branch bend strength.
    pub branch_strength: f32,
    /// Leaf bend strength.
    pub leaf_strength: f32,
    /// Phase separation between levels.
    pub phase_separation: f32,

    /// Trunk oscillation frequency.
    pub trunk_frequency: f32,
    /// Branch oscillation frequency.
    pub branch_frequency: f32,
    /// Leaf oscillation frequency.
    pub leaf_frequency: f32,
    /// Overall scale.
    pub scale: f32,
}

impl Default for HierarchicalBend {
    fn default() -> Self {
        Self {
            trunk_origin: 0.0,
            branch_origin: 0.3,
            leaf_origin: 0.7,
            _padding: 0.0,

            trunk_strength: DEFAULT_TRUNK_BEND,
            branch_strength: DEFAULT_BRANCH_BEND,
            leaf_strength: DEFAULT_LEAF_BEND,
            phase_separation: PI / 3.0,

            trunk_frequency: 0.3,
            branch_frequency: 0.8,
            leaf_frequency: 2.0,
            scale: 1.0,
        }
    }
}

impl HierarchicalBend {
    /// Create hierarchical bend for a specific foliage type.
    pub fn for_foliage_type(foliage_type: FoliageType) -> Self {
        let mut bend = Self::default();
        bend.trunk_strength *= foliage_type.trunk_bend_multiplier();
        bend.branch_strength *= foliage_type.branch_bend_multiplier();
        bend.leaf_strength *= foliage_type.leaf_bend_multiplier();

        // Adjust frequencies based on type
        match foliage_type {
            FoliageType::LargeTree => {
                bend.trunk_frequency *= 0.5;
                bend.branch_frequency *= 0.7;
            }
            FoliageType::Palm => {
                bend.branch_origin = 0.8;
                bend.branch_frequency *= 0.6;
            }
            _ => {}
        }

        bend
    }

    /// Calculate bend displacement for a vertex.
    ///
    /// `normalized_height` is 0 at base, 1 at top.
    /// `phase` is the base phase offset for this instance.
    /// `time` is the current animation time.
    /// `wind_strength` is the current wind strength.
    pub fn calculate_displacement(
        &self,
        normalized_height: f32,
        phase: f32,
        time: f32,
        wind_strength: f32,
    ) -> [f32; 2] {
        let wind_factor = (wind_strength / DEFAULT_WIND_SPEED).min(2.0);

        let mut dx = 0.0;
        let mut dz = 0.0;

        // Trunk contribution
        if normalized_height > self.trunk_origin {
            let trunk_height =
                ((normalized_height - self.trunk_origin) / (1.0 - self.trunk_origin)).min(1.0);
            let trunk_phase = time * self.trunk_frequency * TAU + phase;
            let trunk_bend =
                trunk_phase.sin() * self.trunk_strength * trunk_height * wind_factor * self.scale;
            dx += trunk_bend;
            dz += trunk_bend * 0.3; // Slight perpendicular motion
        }

        // Branch contribution
        if normalized_height > self.branch_origin {
            let branch_height =
                ((normalized_height - self.branch_origin) / (1.0 - self.branch_origin)).min(1.0);
            let branch_phase =
                time * self.branch_frequency * TAU + phase + self.phase_separation;
            let branch_bend = branch_phase.sin()
                * self.branch_strength
                * branch_height
                * wind_factor
                * self.scale;
            dx += branch_bend;
            dz += branch_bend * 0.5;
        }

        // Leaf contribution
        if normalized_height > self.leaf_origin {
            let leaf_height =
                ((normalized_height - self.leaf_origin) / (1.0 - self.leaf_origin)).min(1.0);
            let leaf_phase =
                time * self.leaf_frequency * TAU + phase + self.phase_separation * 2.0;
            let leaf_bend =
                leaf_phase.sin() * self.leaf_strength * leaf_height * wind_factor * self.scale;
            dx += leaf_bend;
            dz += leaf_bend * 0.7;
        }

        [dx, dz]
    }

    /// Validate configuration.
    #[inline]
    pub fn validate(&self) -> bool {
        self.trunk_origin >= 0.0
            && self.trunk_origin < self.branch_origin
            && self.branch_origin < self.leaf_origin
            && self.leaf_origin <= 1.0
            && self.trunk_strength >= 0.0
            && self.branch_strength >= 0.0
            && self.leaf_strength >= 0.0
            && self.scale > 0.0
    }
}

// ---------------------------------------------------------------------------
// GerstnerWave - Gerstner wave for grass sway
// ---------------------------------------------------------------------------

/// Gerstner wave parameters for realistic grass sway.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct GerstnerWave {
    /// Wave direction (normalized XZ).
    pub direction: [f32; 2],
    /// Wavelength (world units).
    pub wavelength: f32,
    /// Wave steepness (0-1).
    pub steepness: f32,
    /// Wave speed (units/sec).
    pub speed: f32,
    /// Amplitude.
    pub amplitude: f32,
    /// Phase offset.
    pub phase_offset: f32,
    /// Padding.
    pub _padding: f32,
}

impl Default for GerstnerWave {
    fn default() -> Self {
        Self {
            direction: [1.0, 0.0],
            wavelength: 2.0,
            steepness: 0.5,
            speed: 1.0,
            amplitude: 0.1,
            phase_offset: 0.0,
            _padding: 0.0,
        }
    }
}

impl GerstnerWave {
    /// Create a Gerstner wave aligned with wind direction.
    pub fn from_wind(wind_direction: [f32; 2], wind_strength: f32) -> Self {
        let len = (wind_direction[0] * wind_direction[0] + wind_direction[1] * wind_direction[1])
            .sqrt();
        let dir = if len > EPSILON {
            [wind_direction[0] / len, wind_direction[1] / len]
        } else {
            [1.0, 0.0]
        };

        Self {
            direction: dir,
            wavelength: 2.0 / (1.0 + wind_strength * 0.1),
            steepness: (0.3 + wind_strength * 0.05).min(0.8),
            speed: 1.0 + wind_strength * 0.2,
            amplitude: 0.05 + wind_strength * 0.02,
            phase_offset: 0.0,
            _padding: 0.0,
        }
    }

    /// Calculate displacement at a world position.
    ///
    /// Returns (dx, dy, dz) displacement.
    pub fn calculate(&self, world_x: f32, world_z: f32, time: f32) -> [f32; 3] {
        let k = TAU / self.wavelength;
        let c = self.speed;

        // Phase at this position
        let dot = self.direction[0] * world_x + self.direction[1] * world_z;
        let phase = k * dot - c * time + self.phase_offset;

        // Gerstner formula
        let q = self.steepness / (k * self.amplitude).max(EPSILON);
        let cos_phase = phase.cos();
        let sin_phase = phase.sin();

        let dx = q * self.amplitude * self.direction[0] * cos_phase;
        let dy = self.amplitude * sin_phase;
        let dz = q * self.amplitude * self.direction[1] * cos_phase;

        [dx, dy, dz]
    }

    /// Calculate wave frequency (Hz).
    #[inline]
    pub fn frequency(&self) -> f32 {
        self.speed / self.wavelength
    }

    /// Validate wave parameters.
    #[inline]
    pub fn validate(&self) -> bool {
        let dir_len = (self.direction[0] * self.direction[0]
            + self.direction[1] * self.direction[1])
        .sqrt();

        (dir_len - 1.0).abs() < 0.01
            && self.wavelength > 0.0
            && self.steepness >= 0.0
            && self.steepness <= 1.0
            && self.amplitude >= 0.0
    }
}

// ---------------------------------------------------------------------------
// DetailOscillation - High-frequency flutter for leaves
// ---------------------------------------------------------------------------

/// High-frequency detail oscillation for leaves and small elements.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct DetailOscillation {
    /// Primary oscillation frequency (Hz).
    pub primary_frequency: f32,
    /// Secondary oscillation frequency (Hz).
    pub secondary_frequency: f32,
    /// Primary amplitude.
    pub primary_amplitude: f32,
    /// Secondary amplitude.
    pub secondary_amplitude: f32,
    /// Phase variation scale.
    pub phase_scale: f32,
    /// Wind influence (how much wind affects detail).
    pub wind_influence: f32,
    /// Padding.
    pub _padding: [f32; 2],
}

impl Default for DetailOscillation {
    fn default() -> Self {
        Self {
            primary_frequency: DEFAULT_DETAIL_FREQUENCY,
            secondary_frequency: DEFAULT_DETAIL_FREQUENCY * GOLDEN_RATIO,
            primary_amplitude: DEFAULT_DETAIL_AMPLITUDE,
            secondary_amplitude: DEFAULT_DETAIL_AMPLITUDE * 0.5,
            phase_scale: 0.1,
            wind_influence: 0.5,
            _padding: [0.0; 2],
        }
    }
}

impl DetailOscillation {
    /// Calculate detail oscillation at a position.
    pub fn calculate(&self, world_x: f32, world_z: f32, time: f32, wind_strength: f32) -> f32 {
        // Position-based phase offset for variation
        let phase1 = world_x * self.phase_scale + world_z * self.phase_scale * GOLDEN_RATIO;
        let phase2 = world_x * self.phase_scale * GOLDEN_RATIO - world_z * self.phase_scale;

        // Primary oscillation
        let primary = (time * self.primary_frequency * TAU + phase1).sin() * self.primary_amplitude;

        // Secondary oscillation (different frequency for complexity)
        let secondary =
            (time * self.secondary_frequency * TAU + phase2).sin() * self.secondary_amplitude;

        // Wind influence modulation
        let wind_mod = 1.0 + (wind_strength / DEFAULT_WIND_SPEED) * self.wind_influence;

        (primary + secondary) * wind_mod
    }

    /// Calculate 2D detail oscillation.
    pub fn calculate_2d(
        &self,
        world_x: f32,
        world_z: f32,
        time: f32,
        wind_strength: f32,
    ) -> [f32; 2] {
        let phase1 = world_x * self.phase_scale + world_z * self.phase_scale * GOLDEN_RATIO;
        let phase2 = world_z * self.phase_scale - world_x * self.phase_scale * 0.7;

        let wind_mod = 1.0 + (wind_strength / DEFAULT_WIND_SPEED) * self.wind_influence;

        let dx = (time * self.primary_frequency * TAU + phase1).sin()
            * self.primary_amplitude
            * wind_mod;
        let dz = (time * self.secondary_frequency * TAU + phase2).sin()
            * self.secondary_amplitude
            * wind_mod;

        [dx, dz]
    }

    /// Validate parameters.
    #[inline]
    pub fn validate(&self) -> bool {
        self.primary_frequency > 0.0
            && self.secondary_frequency > 0.0
            && self.primary_amplitude >= 0.0
            && self.secondary_amplitude >= 0.0
            && self.wind_influence >= 0.0
    }
}

// ---------------------------------------------------------------------------
// WindOcclusion - Estimation of wind blocking from geometry
// ---------------------------------------------------------------------------

/// Wind occlusion estimation.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct WindOcclusion {
    /// Occlusion factor (0 = fully blocked, 1 = no occlusion).
    pub factor: f32,
    /// Occlusion direction (where wind is coming from).
    pub direction: [f32; 2],
    /// Distance to nearest occluder.
    pub distance: f32,
}

impl WindOcclusion {
    /// Full wind (no occlusion).
    #[inline]
    pub fn full_wind() -> Self {
        Self {
            factor: 1.0,
            direction: [0.0, 0.0],
            distance: f32::MAX,
        }
    }

    /// Fully occluded.
    #[inline]
    pub fn fully_occluded() -> Self {
        Self {
            factor: 0.0,
            direction: [0.0, 0.0],
            distance: 0.0,
        }
    }

    /// Calculate occlusion from distance and size.
    pub fn from_occluder(occluder_distance: f32, occluder_size: f32, falloff: f32) -> Self {
        if occluder_distance <= 0.0 {
            return Self::fully_occluded();
        }

        let normalized_distance = occluder_distance / occluder_size.max(EPSILON);
        let factor = (normalized_distance * falloff).min(1.0);

        Self {
            factor,
            direction: [0.0, 0.0],
            distance: occluder_distance,
        }
    }

    /// Combine multiple occlusion sources.
    pub fn combine(sources: &[Self]) -> Self {
        if sources.is_empty() {
            return Self::full_wind();
        }

        let mut min_factor = 1.0;
        let mut closest_dist = f32::MAX;
        let mut dir = [0.0f32; 2];

        for src in sources {
            if src.factor < min_factor {
                min_factor = src.factor;
                dir = src.direction;
            }
            if src.distance < closest_dist {
                closest_dist = src.distance;
            }
        }

        Self {
            factor: min_factor,
            direction: dir,
            distance: closest_dist,
        }
    }
}

// ---------------------------------------------------------------------------
// FoliageWindSystem - Main wind system
// ---------------------------------------------------------------------------

/// Main foliage wind system that integrates all wind components.
#[derive(Debug, Clone)]
pub struct FoliageWindSystem {
    /// GPU uniforms.
    pub uniforms: FoliageWindUniforms,
    /// Gerstner wave configuration.
    pub gerstner: GerstnerWave,
    /// Hierarchical bend configuration.
    pub hierarchical: HierarchicalBend,
    /// Detail oscillation configuration.
    pub detail: DetailOscillation,
    /// Current foliage type.
    pub foliage_type: FoliageType,
    /// Noise seed for turbulence.
    pub noise_seed: u32,
}

impl Default for FoliageWindSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl FoliageWindSystem {
    /// Create a new wind system with default settings.
    pub fn new() -> Self {
        Self {
            uniforms: FoliageWindUniforms::default(),
            gerstner: GerstnerWave::default(),
            hierarchical: HierarchicalBend::default(),
            detail: DetailOscillation::default(),
            foliage_type: FoliageType::default(),
            noise_seed: 0x1337CAFE,
        }
    }

    /// Create a wind system for a specific foliage type.
    pub fn for_foliage_type(foliage_type: FoliageType) -> Self {
        let mut system = Self::new();
        system.set_foliage_type(foliage_type);
        system
    }

    /// Create from weather map wind velocity.
    pub fn from_weather_wind(wind_velocity: [f32; 2]) -> Self {
        let mut system = Self::new();
        system.set_wind_velocity(wind_velocity);
        system
    }

    /// Set the foliage type and update parameters accordingly.
    pub fn set_foliage_type(&mut self, foliage_type: FoliageType) {
        self.foliage_type = foliage_type;
        self.uniforms.set_foliage_type(foliage_type);
        self.hierarchical = HierarchicalBend::for_foliage_type(foliage_type);
        self.detail.primary_amplitude =
            DEFAULT_DETAIL_AMPLITUDE * foliage_type.detail_flutter_scale();
        self.detail.secondary_amplitude =
            DEFAULT_DETAIL_AMPLITUDE * foliage_type.detail_flutter_scale() * 0.5;
    }

    /// Set wind from velocity vector (typically from weather map).
    pub fn set_wind_velocity(&mut self, velocity: [f32; 2]) {
        self.uniforms.set_wind_velocity(velocity);
        self.gerstner = GerstnerWave::from_wind(self.uniforms.wind_direction, self.uniforms.wind_strength);
    }

    /// Set wind from speed and direction.
    pub fn set_wind(&mut self, speed: f32, direction_radians: f32) {
        self.uniforms.set_wind(speed, direction_radians);
        self.gerstner = GerstnerWave::from_wind(self.uniforms.wind_direction, speed);
    }

    /// Update the system for a new frame.
    pub fn update(&mut self, delta_time: f32) {
        self.uniforms.update_time(delta_time);
    }

    /// Sample wind at a world position.
    pub fn sample(&self, world_x: f32, world_z: f32) -> WindSample {
        // Calculate phase offset based on position
        let phase_offset = calculate_phase_offset(world_x, world_z, self.uniforms.phase_offset_scale);

        // Calculate gust factor
        let gust_phase = self.uniforms.time * self.uniforms.gust_frequency * TAU + phase_offset;
        let gust_factor = 1.0 + self.uniforms.gust_strength * (gust_phase.sin() * 0.5 + 0.5);

        // Calculate turbulence
        let turbulence = self.sample_turbulence(world_x, world_z);

        // Base velocity with gust
        let base_velocity = self.uniforms.wind_velocity();
        let velocity = [
            base_velocity[0] * gust_factor,
            base_velocity[1] * gust_factor,
        ];

        WindSample {
            velocity,
            strength: self.uniforms.wind_strength * gust_factor,
            direction: self.uniforms.wind_direction_radians(),
            turbulence,
            phase_offset,
            gust_factor,
        }
    }

    /// Sample turbulence at a position.
    fn sample_turbulence(&self, world_x: f32, world_z: f32) -> [f32; 2] {
        if self.uniforms.turbulence_intensity < EPSILON {
            return [0.0; 2];
        }

        let scale = 1.0 / self.uniforms.turbulence_scale;
        let time_offset = self.uniforms.time * 0.1;

        // Sample 2D noise for turbulence
        let noise_x = sample_noise_2d(
            world_x * scale + time_offset,
            world_z * scale,
            self.noise_seed,
        );
        let noise_z = sample_noise_2d(
            world_x * scale + 100.0,
            world_z * scale + time_offset,
            self.noise_seed.wrapping_add(0x12345678),
        );

        let intensity = self.uniforms.turbulence_intensity * self.uniforms.wind_strength;

        [noise_x * intensity, noise_z * intensity]
    }

    /// Calculate vertex displacement for foliage.
    ///
    /// `world_pos`: World position of the vertex
    /// `normalized_height`: Height from 0 (base) to 1 (top)
    /// `lod`: Current LOD level
    pub fn calculate_displacement(
        &self,
        world_pos: [f32; 3],
        normalized_height: f32,
        lod: u8,
    ) -> [f32; 3] {
        // LOD-based detail falloff
        let lod_factor = calculate_lod_factor(lod, self.uniforms.lod_detail_falloff);
        if lod_factor < EPSILON {
            return [0.0; 3];
        }

        let wind_sample = self.sample(world_pos[0], world_pos[2]);

        let mut displacement = [0.0f32; 3];

        // Apply displacement based on foliage type
        if self.foliage_type.uses_pivot_bending() {
            // Pivot-based bending for grass/small plants
            let gerstner_disp =
                self.gerstner
                    .calculate(world_pos[0], world_pos[2], self.uniforms.time);

            // Scale by height (more movement at top)
            let height_factor = normalized_height * normalized_height;

            displacement[0] += gerstner_disp[0] * height_factor * wind_sample.gust_factor;
            displacement[1] += gerstner_disp[1] * height_factor * wind_sample.gust_factor * 0.3;
            displacement[2] += gerstner_disp[2] * height_factor * wind_sample.gust_factor;
        } else {
            // Hierarchical bending for trees
            let hier_disp = self.hierarchical.calculate_displacement(
                normalized_height,
                wind_sample.phase_offset,
                self.uniforms.time,
                wind_sample.strength,
            );

            displacement[0] += hier_disp[0];
            displacement[2] += hier_disp[1];
        }

        // Add turbulence
        let turb_factor = normalized_height * 0.5;
        displacement[0] += wind_sample.turbulence[0] * turb_factor;
        displacement[2] += wind_sample.turbulence[1] * turb_factor;

        // Add detail oscillation
        let detail_disp =
            self.detail
                .calculate_2d(world_pos[0], world_pos[2], self.uniforms.time, wind_sample.strength);
        displacement[0] += detail_disp[0] * normalized_height;
        displacement[2] += detail_disp[1] * normalized_height;

        // Apply LOD factor
        displacement[0] *= lod_factor;
        displacement[1] *= lod_factor;
        displacement[2] *= lod_factor;

        // Apply wind occlusion
        displacement[0] *= self.uniforms.wind_occlusion;
        displacement[1] *= self.uniforms.wind_occlusion;
        displacement[2] *= self.uniforms.wind_occlusion;

        displacement
    }

    /// Get the GPU uniforms for shader upload.
    #[inline]
    pub fn get_uniforms(&self) -> &FoliageWindUniforms {
        &self.uniforms
    }

    /// Validate the entire system configuration.
    pub fn validate(&self) -> bool {
        self.uniforms.validate()
            && self.gerstner.validate()
            && self.hierarchical.validate()
            && self.detail.validate()
    }
}

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

/// Calculate phase offset based on world position.
///
/// Uses a hash-like function to ensure unique phases for each position.
#[inline]
pub fn calculate_phase_offset(world_x: f32, world_z: f32, scale: f32) -> f32 {
    let px = (world_x * scale * 1000.0) as i32;
    let pz = (world_z * scale * 1000.0) as i32;

    // Simple hash-like combination
    let hash = px.wrapping_mul(73856093) ^ pz.wrapping_mul(19349663);
    let normalized = (hash as u32 as f32) / (u32::MAX as f32);

    normalized * TAU
}

/// Calculate LOD detail factor.
///
/// Higher LOD = less detail. Returns 0-1 factor.
#[inline]
pub fn calculate_lod_factor(lod: u8, falloff: f32) -> f32 {
    if lod >= MAX_WIND_LOD {
        return 0.0;
    }
    let lod_normalized = lod as f32 / MAX_WIND_LOD as f32;
    (1.0 - lod_normalized * falloff).max(0.0)
}

/// Linearly interpolate angles (handles wraparound).
#[inline]
fn lerp_angle(a: f32, b: f32, t: f32) -> f32 {
    let mut diff = b - a;
    while diff > PI {
        diff -= TAU;
    }
    while diff < -PI {
        diff += TAU;
    }
    a + diff * t
}

/// Simple 2D noise for turbulence.
fn sample_noise_2d(x: f32, y: f32, seed: u32) -> f32 {
    let ix = x.floor() as i32;
    let iy = y.floor() as i32;
    let fx = x - ix as f32;
    let fy = y - iy as f32;

    let u = fade(fx);
    let v = fade(fy);

    let h00 = hash_to_float(hash_2d(ix, iy, seed));
    let h10 = hash_to_float(hash_2d(ix + 1, iy, seed));
    let h01 = hash_to_float(hash_2d(ix, iy + 1, seed));
    let h11 = hash_to_float(hash_2d(ix + 1, iy + 1, seed));

    let x0 = lerp(h00, h10, u);
    let x1 = lerp(h01, h11, u);

    lerp(x0, x1, v) * 2.0 - 1.0 // Map to [-1, 1]
}

/// Hash function for noise.
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // FoliageType Tests
    // =========================================================================

    #[test]
    fn test_foliage_type_from_u8() {
        assert_eq!(FoliageType::from_u8(0), FoliageType::Grass);
        assert_eq!(FoliageType::from_u8(1), FoliageType::SmallPlant);
        assert_eq!(FoliageType::from_u8(7), FoliageType::Aquatic);
        assert_eq!(FoliageType::from_u8(255), FoliageType::SmallPlant);
    }

    #[test]
    fn test_foliage_type_stiffness() {
        assert!(FoliageType::Grass.stiffness() < FoliageType::LargeTree.stiffness());
        assert!(FoliageType::SmallPlant.stiffness() > 0.0);
        assert!(FoliageType::LargeTree.stiffness() < 1.0);
    }

    #[test]
    fn test_foliage_type_trunk_bend_multiplier() {
        assert_eq!(FoliageType::Grass.trunk_bend_multiplier(), 0.0);
        assert!(FoliageType::SmallTree.trunk_bend_multiplier() > 0.0);
    }

    #[test]
    fn test_foliage_type_branch_bend_multiplier() {
        assert_eq!(FoliageType::Grass.branch_bend_multiplier(), 0.0);
        assert!(FoliageType::Palm.branch_bend_multiplier() > FoliageType::SmallTree.branch_bend_multiplier());
    }

    #[test]
    fn test_foliage_type_leaf_bend_multiplier() {
        assert!(FoliageType::Grass.leaf_bend_multiplier() > 0.0);
        assert!(FoliageType::LargeTree.leaf_bend_multiplier() > 0.0);
    }

    #[test]
    fn test_foliage_type_uses_pivot_bending() {
        assert!(FoliageType::Grass.uses_pivot_bending());
        assert!(FoliageType::SmallPlant.uses_pivot_bending());
        assert!(FoliageType::Hanging.uses_pivot_bending());
        assert!(!FoliageType::SmallTree.uses_pivot_bending());
        assert!(!FoliageType::LargeTree.uses_pivot_bending());
    }

    #[test]
    fn test_foliage_type_detail_flutter_scale() {
        assert!(FoliageType::Grass.detail_flutter_scale() > FoliageType::LargeTree.detail_flutter_scale());
    }

    #[test]
    fn test_foliage_type_name() {
        assert_eq!(FoliageType::Grass.name(), "grass");
        assert_eq!(FoliageType::Palm.name(), "palm");
    }

    #[test]
    fn test_foliage_type_default() {
        assert_eq!(FoliageType::default(), FoliageType::SmallPlant);
    }

    // =========================================================================
    // FoliageWindUniforms Tests
    // =========================================================================

    #[test]
    fn test_uniforms_default() {
        let uniforms = FoliageWindUniforms::default();
        assert!(uniforms.validate());
        assert!(uniforms.wind_strength > 0.0);
    }

    #[test]
    fn test_uniforms_struct_size() {
        assert_eq!(std::mem::size_of::<FoliageWindUniforms>(), 96);
    }

    #[test]
    fn test_uniforms_pod() {
        let uniforms = FoliageWindUniforms::default();
        let bytes = bytemuck::bytes_of(&uniforms);
        assert_eq!(bytes.len(), 96);
    }

    #[test]
    fn test_uniforms_from_velocity() {
        let uniforms = FoliageWindUniforms::from_velocity([10.0, 0.0]);
        assert!((uniforms.wind_strength - 10.0).abs() < EPSILON);
        assert!((uniforms.wind_direction[0] - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_uniforms_from_velocity_diagonal() {
        let uniforms = FoliageWindUniforms::from_velocity([3.0, 4.0]);
        assert!((uniforms.wind_strength - 5.0).abs() < EPSILON);
        assert!((uniforms.wind_direction[0] - 0.6).abs() < 0.01);
        assert!((uniforms.wind_direction[1] - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_uniforms_from_velocity_zero() {
        let uniforms = FoliageWindUniforms::from_velocity([0.0, 0.0]);
        assert_eq!(uniforms.wind_strength, 0.0);
        assert_eq!(uniforms.wind_direction, [1.0, 0.0]); // Default direction
    }

    #[test]
    fn test_uniforms_from_speed_direction() {
        let uniforms = FoliageWindUniforms::from_speed_direction(10.0, 0.0);
        assert!((uniforms.wind_strength - 10.0).abs() < EPSILON);
        assert!((uniforms.wind_direction[0] - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_uniforms_set_wind() {
        let mut uniforms = FoliageWindUniforms::new();
        uniforms.set_wind(15.0, PI / 4.0);
        assert!((uniforms.wind_strength - 15.0).abs() < EPSILON);
        assert!((uniforms.wind_direction_radians() - PI / 4.0).abs() < 0.01);
    }

    #[test]
    fn test_uniforms_set_wind_velocity() {
        let mut uniforms = FoliageWindUniforms::new();
        uniforms.set_wind_velocity([5.0, 5.0]);
        let expected_strength = (50.0f32).sqrt();
        assert!((uniforms.wind_strength - expected_strength).abs() < 0.01);
    }

    #[test]
    fn test_uniforms_update_time() {
        let mut uniforms = FoliageWindUniforms::new();
        assert_eq!(uniforms.time, 0.0);
        uniforms.update_time(1.0);
        assert_eq!(uniforms.time, 1.0);
        uniforms.update_time(0.5);
        assert_eq!(uniforms.time, 1.5);
    }

    #[test]
    fn test_uniforms_set_bend_factors() {
        let mut uniforms = FoliageWindUniforms::new();
        uniforms.set_bend_factors(0.1, 0.2, 0.3);
        assert_eq!(uniforms.trunk_bend, 0.1);
        assert_eq!(uniforms.branch_bend, 0.2);
        assert_eq!(uniforms.leaf_bend, 0.3);
    }

    #[test]
    fn test_uniforms_set_bend_factors_clamping() {
        let mut uniforms = FoliageWindUniforms::new();
        uniforms.set_bend_factors(1.5, -0.5, 2.0);
        assert_eq!(uniforms.trunk_bend, 1.0);
        assert_eq!(uniforms.branch_bend, 0.0);
        assert_eq!(uniforms.leaf_bend, 1.0);
    }

    #[test]
    fn test_uniforms_set_foliage_type() {
        let mut uniforms = FoliageWindUniforms::new();
        uniforms.set_foliage_type(FoliageType::Grass);
        assert_eq!(uniforms.trunk_bend, 0.0); // Grass has no trunk
    }

    #[test]
    fn test_uniforms_set_gust() {
        let mut uniforms = FoliageWindUniforms::new();
        uniforms.set_gust(0.5, 0.7);
        assert_eq!(uniforms.gust_frequency, 0.5);
        assert_eq!(uniforms.gust_strength, 0.7);
    }

    #[test]
    fn test_uniforms_set_turbulence() {
        let mut uniforms = FoliageWindUniforms::new();
        uniforms.set_turbulence(100.0, 0.5);
        assert_eq!(uniforms.turbulence_scale, 100.0);
        assert_eq!(uniforms.turbulence_intensity, 0.5);
    }

    #[test]
    fn test_uniforms_set_detail() {
        let mut uniforms = FoliageWindUniforms::new();
        uniforms.set_detail(3.0, 0.1);
        assert_eq!(uniforms.detail_frequency, 3.0);
        assert_eq!(uniforms.detail_amplitude, 0.1);
    }

    #[test]
    fn test_uniforms_set_gerstner() {
        let mut uniforms = FoliageWindUniforms::new();
        uniforms.set_gerstner(3.0, 0.6, 2.0);
        assert_eq!(uniforms.gerstner_params[0], 3.0);
        assert_eq!(uniforms.gerstner_params[1], 0.6);
        assert_eq!(uniforms.gerstner_params[2], 2.0);
    }

    #[test]
    fn test_uniforms_wind_velocity() {
        let uniforms = FoliageWindUniforms::from_speed_direction(10.0, 0.0);
        let vel = uniforms.wind_velocity();
        assert!((vel[0] - 10.0).abs() < EPSILON);
        assert!(vel[1].abs() < EPSILON);
    }

    #[test]
    fn test_uniforms_gust_factor() {
        let mut uniforms = FoliageWindUniforms::new();
        uniforms.gust_strength = 0.5;
        uniforms.gust_frequency = 1.0;
        uniforms.time = 0.0;

        let factor = uniforms.gust_factor();
        assert!(factor >= 1.0);
        assert!(factor <= 1.5);
    }

    #[test]
    fn test_uniforms_gust_factor_no_gust() {
        let mut uniforms = FoliageWindUniforms::new();
        uniforms.gust_strength = 0.0;
        assert_eq!(uniforms.gust_factor(), 1.0);
    }

    #[test]
    fn test_uniforms_validate_valid() {
        let uniforms = FoliageWindUniforms::default();
        assert!(uniforms.validate());
    }

    #[test]
    fn test_uniforms_validate_invalid_direction() {
        let mut uniforms = FoliageWindUniforms::default();
        uniforms.wind_direction = [0.0, 0.0];
        assert!(!uniforms.validate());
    }

    #[test]
    fn test_uniforms_validate_invalid_speed() {
        let mut uniforms = FoliageWindUniforms::default();
        uniforms.wind_strength = -1.0;
        assert!(!uniforms.validate());
    }

    #[test]
    fn test_uniforms_validate_invalid_gust() {
        let mut uniforms = FoliageWindUniforms::default();
        uniforms.gust_strength = 2.0;
        assert!(!uniforms.validate());
    }

    // =========================================================================
    // WindSample Tests
    // =========================================================================

    #[test]
    fn test_wind_sample_calm() {
        let sample = WindSample::calm();
        assert_eq!(sample.velocity, [0.0, 0.0]);
        assert_eq!(sample.strength, 0.0);
    }

    #[test]
    fn test_wind_sample_from_velocity() {
        let sample = WindSample::from_velocity([3.0, 4.0]);
        assert!((sample.strength - 5.0).abs() < EPSILON);
    }

    #[test]
    fn test_wind_sample_total_velocity() {
        let mut sample = WindSample::from_velocity([5.0, 0.0]);
        sample.turbulence = [1.0, 2.0];
        let total = sample.total_velocity();
        assert_eq!(total, [6.0, 2.0]);
    }

    #[test]
    fn test_wind_sample_effective_strength() {
        let mut sample = WindSample::from_velocity([10.0, 0.0]);
        sample.gust_factor = 1.5;
        assert!((sample.effective_strength() - 15.0).abs() < EPSILON);
    }

    #[test]
    fn test_wind_sample_lerp() {
        let a = WindSample::from_velocity([0.0, 0.0]);
        let b = WindSample::from_velocity([10.0, 0.0]);

        let mid = WindSample::lerp(&a, &b, 0.5);
        assert!((mid.velocity[0] - 5.0).abs() < EPSILON);
    }

    #[test]
    fn test_wind_sample_lerp_clamping() {
        let a = WindSample::from_velocity([0.0, 0.0]);
        let b = WindSample::from_velocity([10.0, 0.0]);

        let over = WindSample::lerp(&a, &b, 2.0);
        assert!((over.velocity[0] - 10.0).abs() < EPSILON);
    }

    // =========================================================================
    // PivotPoint Tests
    // =========================================================================

    #[test]
    fn test_pivot_point_at_base() {
        let pivot = PivotPoint::at_base();
        assert_eq!(pivot.position, [0.0, 0.0, 0.0]);
        assert!(pivot.validate());
    }

    #[test]
    fn test_pivot_point_at_height() {
        let pivot = PivotPoint::at_height(1.5);
        assert_eq!(pivot.position[1], 1.5);
    }

    #[test]
    fn test_pivot_point_at_top() {
        let pivot = PivotPoint::at_top(2.0);
        assert_eq!(pivot.position[1], 2.0);
        assert!(pivot.height_influence < 0.0); // Inverse for hanging
    }

    #[test]
    fn test_pivot_point_bend_amount() {
        let pivot = PivotPoint::at_base();
        let bend_at_base = pivot.bend_amount(0.0);
        let bend_at_top = pivot.bend_amount(1.0);

        assert!(bend_at_base.abs() < EPSILON);
        assert!(bend_at_top > bend_at_base);
    }

    #[test]
    fn test_pivot_point_validate() {
        let valid = PivotPoint::at_base();
        assert!(valid.validate());

        let invalid = PivotPoint {
            radius: 0.0,
            ..PivotPoint::at_base()
        };
        assert!(!invalid.validate());
    }

    #[test]
    fn test_pivot_point_pod() {
        let pivot = PivotPoint::at_base();
        let bytes = bytemuck::bytes_of(&pivot);
        assert_eq!(bytes.len(), 32);
    }

    // =========================================================================
    // HierarchicalBend Tests
    // =========================================================================

    #[test]
    fn test_hierarchical_bend_default() {
        let bend = HierarchicalBend::default();
        assert!(bend.validate());
    }

    #[test]
    fn test_hierarchical_bend_for_foliage_type() {
        let grass = HierarchicalBend::for_foliage_type(FoliageType::Grass);
        let tree = HierarchicalBend::for_foliage_type(FoliageType::SmallTree);

        assert!(grass.trunk_strength < tree.trunk_strength);
    }

    #[test]
    fn test_hierarchical_bend_calculate_displacement() {
        let bend = HierarchicalBend::default();

        // At base, should be zero
        let disp_base = bend.calculate_displacement(0.0, 0.0, 0.0, 5.0);
        assert!(disp_base[0].abs() < 0.1);

        // At top, should be non-zero (depending on time)
        let disp_top = bend.calculate_displacement(1.0, 0.0, 1.0, 5.0);
        // Just verify it's finite
        assert!(disp_top[0].is_finite());
        assert!(disp_top[1].is_finite());
    }

    #[test]
    fn test_hierarchical_bend_wind_strength_scaling() {
        let bend = HierarchicalBend::default();

        let disp_weak = bend.calculate_displacement(1.0, 0.0, 1.0, 1.0);
        let disp_strong = bend.calculate_displacement(1.0, 0.0, 1.0, 10.0);

        // Stronger wind should produce larger displacement
        let mag_weak = (disp_weak[0] * disp_weak[0] + disp_weak[1] * disp_weak[1]).sqrt();
        let mag_strong = (disp_strong[0] * disp_strong[0] + disp_strong[1] * disp_strong[1]).sqrt();

        assert!(mag_strong > mag_weak);
    }

    #[test]
    fn test_hierarchical_bend_validate() {
        let valid = HierarchicalBend::default();
        assert!(valid.validate());

        let invalid = HierarchicalBend {
            trunk_origin: 0.5,
            branch_origin: 0.3, // Less than trunk
            ..Default::default()
        };
        assert!(!invalid.validate());
    }

    #[test]
    fn test_hierarchical_bend_pod() {
        let bend = HierarchicalBend::default();
        let bytes = bytemuck::bytes_of(&bend);
        assert_eq!(bytes.len(), 48);
    }

    // =========================================================================
    // GerstnerWave Tests
    // =========================================================================

    #[test]
    fn test_gerstner_wave_default() {
        let wave = GerstnerWave::default();
        assert!(wave.validate());
    }

    #[test]
    fn test_gerstner_wave_from_wind() {
        let wave = GerstnerWave::from_wind([1.0, 0.0], 10.0);
        assert!((wave.direction[0] - 1.0).abs() < EPSILON);
        assert!(wave.amplitude > GerstnerWave::default().amplitude);
    }

    #[test]
    fn test_gerstner_wave_from_wind_diagonal() {
        let wave = GerstnerWave::from_wind([1.0, 1.0], 5.0);
        let dir_len = (wave.direction[0] * wave.direction[0]
            + wave.direction[1] * wave.direction[1])
        .sqrt();
        assert!((dir_len - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_gerstner_wave_calculate() {
        let wave = GerstnerWave::default();
        let disp = wave.calculate(0.0, 0.0, 0.0);

        assert!(disp[0].is_finite());
        assert!(disp[1].is_finite());
        assert!(disp[2].is_finite());
    }

    #[test]
    fn test_gerstner_wave_calculate_varies_with_position() {
        let wave = GerstnerWave::default();
        let disp1 = wave.calculate(0.0, 0.0, 0.0);
        let disp2 = wave.calculate(5.0, 0.0, 0.0);

        // Different positions should give different displacements
        assert!((disp1[0] - disp2[0]).abs() > EPSILON || (disp1[1] - disp2[1]).abs() > EPSILON);
    }

    #[test]
    fn test_gerstner_wave_calculate_varies_with_time() {
        let wave = GerstnerWave::default();
        let disp1 = wave.calculate(0.0, 0.0, 0.0);
        let disp2 = wave.calculate(0.0, 0.0, 1.0);

        assert!((disp1[0] - disp2[0]).abs() > EPSILON || (disp1[1] - disp2[1]).abs() > EPSILON);
    }

    #[test]
    fn test_gerstner_wave_frequency() {
        let wave = GerstnerWave {
            speed: 2.0,
            wavelength: 4.0,
            ..Default::default()
        };
        assert!((wave.frequency() - 0.5).abs() < EPSILON);
    }

    #[test]
    fn test_gerstner_wave_validate() {
        let valid = GerstnerWave::default();
        assert!(valid.validate());

        let invalid = GerstnerWave {
            wavelength: 0.0,
            ..Default::default()
        };
        assert!(!invalid.validate());
    }

    #[test]
    fn test_gerstner_wave_pod() {
        let wave = GerstnerWave::default();
        let bytes = bytemuck::bytes_of(&wave);
        assert_eq!(bytes.len(), 32);
    }

    // =========================================================================
    // DetailOscillation Tests
    // =========================================================================

    #[test]
    fn test_detail_oscillation_default() {
        let detail = DetailOscillation::default();
        assert!(detail.validate());
    }

    #[test]
    fn test_detail_oscillation_calculate() {
        let detail = DetailOscillation::default();
        let osc = detail.calculate(0.0, 0.0, 0.0, 5.0);
        assert!(osc.is_finite());
    }

    #[test]
    fn test_detail_oscillation_calculate_varies() {
        let detail = DetailOscillation::default();
        let osc1 = detail.calculate(0.0, 0.0, 0.0, 5.0);
        let osc2 = detail.calculate(10.0, 10.0, 1.0, 5.0);

        assert!((osc1 - osc2).abs() > EPSILON);
    }

    #[test]
    fn test_detail_oscillation_calculate_2d() {
        let detail = DetailOscillation::default();
        let osc = detail.calculate_2d(0.0, 0.0, 0.0, 5.0);
        assert!(osc[0].is_finite());
        assert!(osc[1].is_finite());
    }

    #[test]
    fn test_detail_oscillation_wind_influence() {
        let detail = DetailOscillation::default();
        let weak = detail.calculate(0.0, 0.0, 0.5, 1.0);
        let strong = detail.calculate(0.0, 0.0, 0.5, 20.0);

        assert!(strong.abs() > weak.abs());
    }

    #[test]
    fn test_detail_oscillation_validate() {
        let valid = DetailOscillation::default();
        assert!(valid.validate());

        let invalid = DetailOscillation {
            primary_frequency: 0.0,
            ..Default::default()
        };
        assert!(!invalid.validate());
    }

    #[test]
    fn test_detail_oscillation_pod() {
        let detail = DetailOscillation::default();
        let bytes = bytemuck::bytes_of(&detail);
        assert_eq!(bytes.len(), 32);
    }

    // =========================================================================
    // WindOcclusion Tests
    // =========================================================================

    #[test]
    fn test_wind_occlusion_full_wind() {
        let occ = WindOcclusion::full_wind();
        assert_eq!(occ.factor, 1.0);
        assert_eq!(occ.distance, f32::MAX);
    }

    #[test]
    fn test_wind_occlusion_fully_occluded() {
        let occ = WindOcclusion::fully_occluded();
        assert_eq!(occ.factor, 0.0);
        assert_eq!(occ.distance, 0.0);
    }

    #[test]
    fn test_wind_occlusion_from_occluder() {
        let occ = WindOcclusion::from_occluder(5.0, 10.0, 1.0);
        assert!(occ.factor > 0.0);
        assert!(occ.factor < 1.0);
    }

    #[test]
    fn test_wind_occlusion_from_occluder_zero_distance() {
        let occ = WindOcclusion::from_occluder(0.0, 10.0, 1.0);
        assert_eq!(occ.factor, 0.0);
    }

    #[test]
    fn test_wind_occlusion_from_occluder_far_away() {
        let occ = WindOcclusion::from_occluder(100.0, 10.0, 1.0);
        assert_eq!(occ.factor, 1.0);
    }

    #[test]
    fn test_wind_occlusion_combine_empty() {
        let occ = WindOcclusion::combine(&[]);
        assert_eq!(occ.factor, 1.0);
    }

    #[test]
    fn test_wind_occlusion_combine_single() {
        let sources = [WindOcclusion::from_occluder(5.0, 10.0, 1.0)];
        let combined = WindOcclusion::combine(&sources);
        assert_eq!(combined.factor, sources[0].factor);
    }

    #[test]
    fn test_wind_occlusion_combine_multiple() {
        let sources = [
            WindOcclusion { factor: 0.8, direction: [0.0, 0.0], distance: 10.0 },
            WindOcclusion { factor: 0.5, direction: [1.0, 0.0], distance: 5.0 },
        ];
        let combined = WindOcclusion::combine(&sources);
        assert_eq!(combined.factor, 0.5); // Takes minimum
    }

    // =========================================================================
    // FoliageWindSystem Tests
    // =========================================================================

    #[test]
    fn test_wind_system_new() {
        let system = FoliageWindSystem::new();
        assert!(system.validate());
    }

    #[test]
    fn test_wind_system_for_foliage_type() {
        let grass = FoliageWindSystem::for_foliage_type(FoliageType::Grass);
        let tree = FoliageWindSystem::for_foliage_type(FoliageType::LargeTree);

        assert_eq!(grass.foliage_type, FoliageType::Grass);
        assert_eq!(tree.foliage_type, FoliageType::LargeTree);
    }

    #[test]
    fn test_wind_system_from_weather_wind() {
        let system = FoliageWindSystem::from_weather_wind([10.0, 5.0]);
        let vel = system.uniforms.wind_velocity();
        assert!((vel[0] - 10.0).abs() < 0.1);
        assert!((vel[1] - 5.0).abs() < 0.1);
    }

    #[test]
    fn test_wind_system_set_foliage_type() {
        let mut system = FoliageWindSystem::new();
        system.set_foliage_type(FoliageType::Palm);
        assert_eq!(system.foliage_type, FoliageType::Palm);
    }

    #[test]
    fn test_wind_system_set_wind_velocity() {
        let mut system = FoliageWindSystem::new();
        system.set_wind_velocity([15.0, 0.0]);
        assert!((system.uniforms.wind_strength - 15.0).abs() < EPSILON);
    }

    #[test]
    fn test_wind_system_set_wind() {
        let mut system = FoliageWindSystem::new();
        system.set_wind(20.0, PI / 2.0);
        assert!((system.uniforms.wind_strength - 20.0).abs() < EPSILON);
    }

    #[test]
    fn test_wind_system_update() {
        let mut system = FoliageWindSystem::new();
        system.update(1.0);
        assert!((system.uniforms.time - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_wind_system_sample() {
        let system = FoliageWindSystem::new();
        let sample = system.sample(0.0, 0.0);
        assert!(sample.velocity[0].is_finite());
        assert!(sample.strength.is_finite());
    }

    #[test]
    fn test_wind_system_sample_different_positions() {
        let system = FoliageWindSystem::new();
        let s1 = system.sample(0.0, 0.0);
        let s2 = system.sample(1000.0, 1000.0);

        // Phase offsets should be different
        assert!((s1.phase_offset - s2.phase_offset).abs() > EPSILON);
    }

    #[test]
    fn test_wind_system_calculate_displacement() {
        let mut system = FoliageWindSystem::new();
        system.update(1.0);

        let disp = system.calculate_displacement([0.0, 0.0, 0.0], 0.5, 0);
        assert!(disp[0].is_finite());
        assert!(disp[1].is_finite());
        assert!(disp[2].is_finite());
    }

    #[test]
    fn test_wind_system_calculate_displacement_at_base() {
        let mut system = FoliageWindSystem::new();
        system.update(1.0);

        let disp = system.calculate_displacement([0.0, 0.0, 0.0], 0.0, 0);
        // At base (height 0), displacement should be minimal
        let magnitude = (disp[0] * disp[0] + disp[1] * disp[1] + disp[2] * disp[2]).sqrt();
        assert!(magnitude < 0.5);
    }

    #[test]
    fn test_wind_system_calculate_displacement_at_top() {
        let mut system = FoliageWindSystem::new();
        system.set_wind(15.0, 0.0);
        system.update(1.0);

        let disp_base = system.calculate_displacement([0.0, 0.0, 0.0], 0.0, 0);
        let disp_top = system.calculate_displacement([0.0, 1.0, 0.0], 1.0, 0);

        let mag_base = (disp_base[0] * disp_base[0] + disp_base[2] * disp_base[2]).sqrt();
        let mag_top = (disp_top[0] * disp_top[0] + disp_top[2] * disp_top[2]).sqrt();

        // Top should have more displacement than base
        assert!(mag_top >= mag_base);
    }

    #[test]
    fn test_wind_system_calculate_displacement_lod_reduction() {
        let mut system = FoliageWindSystem::new();
        system.set_wind(10.0, 0.0);
        system.update(1.0);

        let disp_lod0 = system.calculate_displacement([0.0, 1.0, 0.0], 1.0, 0);
        let disp_lod3 = system.calculate_displacement([0.0, 1.0, 0.0], 1.0, 3);

        let mag_lod0 = (disp_lod0[0] * disp_lod0[0] + disp_lod0[2] * disp_lod0[2]).sqrt();
        let mag_lod3 = (disp_lod3[0] * disp_lod3[0] + disp_lod3[2] * disp_lod3[2]).sqrt();

        // Higher LOD should have less detail (smaller displacement)
        assert!(mag_lod3 <= mag_lod0);
    }

    #[test]
    fn test_wind_system_calculate_displacement_max_lod() {
        let mut system = FoliageWindSystem::new();
        system.update(1.0);

        let disp = system.calculate_displacement([0.0, 1.0, 0.0], 1.0, MAX_WIND_LOD);
        // At max LOD, displacement should be zero
        assert!(disp[0].abs() < EPSILON);
        assert!(disp[1].abs() < EPSILON);
        assert!(disp[2].abs() < EPSILON);
    }

    #[test]
    fn test_wind_system_get_uniforms() {
        let system = FoliageWindSystem::new();
        let uniforms = system.get_uniforms();
        assert!(uniforms.validate());
    }

    #[test]
    fn test_wind_system_validate() {
        let system = FoliageWindSystem::new();
        assert!(system.validate());
    }

    // =========================================================================
    // Helper Function Tests
    // =========================================================================

    #[test]
    fn test_calculate_phase_offset() {
        let phase1 = calculate_phase_offset(0.0, 0.0, 0.01);
        let phase2 = calculate_phase_offset(100.0, 0.0, 0.01);

        // Different positions should have different phases
        assert!((phase1 - phase2).abs() > EPSILON);

        // Phase should be in range [0, TAU)
        assert!(phase1 >= 0.0 && phase1 < TAU);
        assert!(phase2 >= 0.0 && phase2 < TAU);
    }

    #[test]
    fn test_calculate_phase_offset_deterministic() {
        let phase1 = calculate_phase_offset(50.0, 50.0, 0.01);
        let phase2 = calculate_phase_offset(50.0, 50.0, 0.01);
        assert_eq!(phase1, phase2);
    }

    #[test]
    fn test_calculate_lod_factor() {
        assert_eq!(calculate_lod_factor(0, 2.0), 1.0);
        assert!(calculate_lod_factor(2, 2.0) < 1.0);
        assert_eq!(calculate_lod_factor(MAX_WIND_LOD, 2.0), 0.0);
    }

    #[test]
    fn test_calculate_lod_factor_falloff() {
        let low_falloff = calculate_lod_factor(2, 1.0);
        let high_falloff = calculate_lod_factor(2, 3.0);

        assert!(high_falloff < low_falloff);
    }

    #[test]
    fn test_lerp_angle() {
        let a = lerp_angle(0.0, PI, 0.5);
        assert!((a - PI / 2.0).abs() < 0.01);
    }

    #[test]
    fn test_lerp_angle_wraparound() {
        // Test crossing the -PI/PI boundary
        let a = lerp_angle(-PI + 0.1, PI - 0.1, 0.5);
        // Should take the short path
        assert!(a.abs() > PI - 0.5);
    }

    #[test]
    fn test_sample_noise_2d_range() {
        for x in 0..10 {
            for y in 0..10 {
                let n = sample_noise_2d(x as f32 * 0.1, y as f32 * 0.1, 0x12345678);
                assert!(n >= -1.0 && n <= 1.0);
            }
        }
    }

    #[test]
    fn test_sample_noise_2d_deterministic() {
        let n1 = sample_noise_2d(0.5, 0.5, 123);
        let n2 = sample_noise_2d(0.5, 0.5, 123);
        assert_eq!(n1, n2);
    }

    #[test]
    fn test_sample_noise_2d_different_seeds() {
        let n1 = sample_noise_2d(0.5, 0.5, 123);
        let n2 = sample_noise_2d(0.5, 0.5, 456);
        assert!((n1 - n2).abs() > EPSILON);
    }

    #[test]
    fn test_hash_2d_deterministic() {
        let h1 = hash_2d(1, 2, 0);
        let h2 = hash_2d(1, 2, 0);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hash_2d_different_inputs() {
        let h1 = hash_2d(1, 2, 0);
        let h2 = hash_2d(2, 1, 0);
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

    // =========================================================================
    // Integration Tests
    // =========================================================================

    #[test]
    fn test_full_wind_pipeline() {
        let mut system = FoliageWindSystem::for_foliage_type(FoliageType::Grass);
        system.set_wind_velocity([10.0, 5.0]);

        // Simulate several frames
        for frame in 0..100 {
            system.update(0.016); // ~60fps

            // Sample at various positions
            for x in 0..5 {
                for z in 0..5 {
                    let wx = x as f32 * 10.0;
                    let wz = z as f32 * 10.0;

                    let sample = system.sample(wx, wz);
                    assert!(sample.velocity[0].is_finite());
                    assert!(sample.strength.is_finite());

                    // Calculate displacement at different heights
                    for h in [0.0, 0.5, 1.0] {
                        let disp = system.calculate_displacement([wx, h, wz], h, 0);
                        assert!(disp[0].is_finite());
                        assert!(disp[1].is_finite());
                        assert!(disp[2].is_finite());
                    }
                }
            }
        }
    }

    #[test]
    fn test_different_foliage_types_produce_different_results() {
        let types = [
            FoliageType::Grass,
            FoliageType::SmallTree,
            FoliageType::LargeTree,
            FoliageType::Palm,
        ];

        let mut results = Vec::new();

        for ft in &types {
            let mut system = FoliageWindSystem::for_foliage_type(*ft);
            system.set_wind(10.0, 0.0);
            system.update(1.0);

            let disp = system.calculate_displacement([0.0, 1.0, 0.0], 1.0, 0);
            results.push((ft, disp));
        }

        // Different foliage types should produce at least some different results
        let unique_count = results
            .iter()
            .map(|(_, d)| ((d[0] * 1000.0) as i32, (d[2] * 1000.0) as i32))
            .collect::<std::collections::HashSet<_>>()
            .len();

        assert!(unique_count > 1, "Different foliage types should produce varied results");
    }

    #[test]
    fn test_wind_strength_affects_displacement() {
        let mut weak_system = FoliageWindSystem::new();
        weak_system.set_wind(1.0, 0.0);
        weak_system.update(1.0);

        let mut strong_system = FoliageWindSystem::new();
        strong_system.set_wind(20.0, 0.0);
        strong_system.update(1.0);

        let weak_disp = weak_system.calculate_displacement([0.0, 1.0, 0.0], 1.0, 0);
        let strong_disp = strong_system.calculate_displacement([0.0, 1.0, 0.0], 1.0, 0);

        let weak_mag = (weak_disp[0] * weak_disp[0] + weak_disp[2] * weak_disp[2]).sqrt();
        let strong_mag = (strong_disp[0] * strong_disp[0] + strong_disp[2] * strong_disp[2]).sqrt();

        assert!(strong_mag > weak_mag);
    }

    #[test]
    fn test_gpu_struct_alignment() {
        // Verify all GPU structs have correct alignment
        assert_eq!(std::mem::align_of::<FoliageWindUniforms>(), 4);
        assert_eq!(std::mem::align_of::<PivotPoint>(), 4);
        assert_eq!(std::mem::align_of::<HierarchicalBend>(), 4);
        assert_eq!(std::mem::align_of::<GerstnerWave>(), 4);
        assert_eq!(std::mem::align_of::<DetailOscillation>(), 4);
    }
}
