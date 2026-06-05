//! Gerstner Wave Compute Shader for TRINITY Engine (T-ENV-1.7).
//!
//! Implements Gerstner wave displacement for realistic water surfaces.
//! Gerstner waves model the orbital motion of water particles, creating
//! the characteristic choppiness of ocean waves.
//!
//! # Overview
//!
//! The Gerstner wave model displaces water surface vertices both vertically
//! (wave height) and horizontally (choppiness). Multiple waves are
//! superimposed to create complex, realistic ocean surfaces.
//!
//! # Physics
//!
//! Deep water dispersion relation: `omega^2 = g * k`
//! - `omega` = angular frequency (rad/s)
//! - `g` = gravitational acceleration (9.81 m/s^2)
//! - `k` = wave number = 2*PI / wavelength
//!
//! Phase velocity: `c = omega / k = sqrt(g / k)`
//!
//! # Steepness Control
//!
//! The steepness parameter (Q) controls horizontal displacement:
//! - Q = 0: Pure sine wave (no horizontal displacement)
//! - Q = 1: Maximum steepness (wave about to break)
//!
//! To prevent wave looping (particle paths crossing), the sum of all
//! wave steepnesses must not exceed 1:
//! `Q_i = steepness_i / (wavelength_i * amplitude_i * num_waves)`
//!
//! # Usage
//!
//! ```ignore
//! // Create from preset
//! let mut waves = GerstnerWaveSet::from_preset(WavePreset::Choppy);
//!
//! // Animate
//! waves.set_time(elapsed_seconds);
//!
//! // Evaluate single point
//! let result = waves.evaluate(10.0, 5.0);
//!
//! // Batch evaluate grid
//! let grid = waves.evaluate_grid(64, 0.5, [0.0, 0.0]);
//! ```

use std::f32::consts::{PI, TAU};
use std::mem;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Compute shader workgroup size (must match WGSL constant).
pub const WORKGROUP_SIZE: u32 = 8;

/// GerstnerWave struct size in bytes.
pub const GERSTNER_WAVE_SIZE: usize = 32;

/// Maximum waves supported by compute shader.
pub const MAX_WAVES: usize = 32;

/// Gravitational acceleration (m/s^2).
pub const GRAVITY: f32 = 9.81;

/// Small epsilon for floating point comparisons.
const EPSILON: f32 = 1e-6;

// ---------------------------------------------------------------------------
// GerstnerWave
// ---------------------------------------------------------------------------

/// Single Gerstner wave parameters.
///
/// Matches the WGSL `GerstnerWave` struct layout.
///
/// # Memory Layout (32 bytes, std140 compatible)
///
/// | Offset | Field      | Size     |
/// |--------|------------|----------|
/// | 0      | amplitude  | 4 bytes  |
/// | 4      | wavelength | 4 bytes  |
/// | 8      | steepness  | 4 bytes  |
/// | 12     | speed      | 4 bytes  |
/// | 16     | direction  | 8 bytes  |
/// | 24     | _padding   | 8 bytes  |
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GerstnerWave {
    /// Wave amplitude (height from trough to crest / 2).
    pub amplitude: f32,
    /// Wavelength (horizontal distance between crests).
    pub wavelength: f32,
    /// Steepness factor (0-1, controls choppiness).
    pub steepness: f32,
    /// Phase velocity multiplier for animation speed.
    pub speed: f32,
    /// Normalized direction vector (XZ plane).
    pub direction: [f32; 2],
    /// Padding for 16-byte alignment.
    pub _padding: [f32; 2],
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<GerstnerWave>() == GERSTNER_WAVE_SIZE);

impl GerstnerWave {
    /// Create a new Gerstner wave.
    ///
    /// # Arguments
    ///
    /// * `amplitude` - Wave height (meters).
    /// * `wavelength` - Distance between crests (meters).
    /// * `steepness` - Choppiness factor (0-1).
    /// * `speed` - Animation speed multiplier.
    /// * `direction` - Wave travel direction (will be normalized).
    pub fn new(
        amplitude: f32,
        wavelength: f32,
        steepness: f32,
        speed: f32,
        direction: [f32; 2],
    ) -> Self {
        let dir_len = (direction[0] * direction[0] + direction[1] * direction[1]).sqrt();
        let normalized = if dir_len > EPSILON {
            [direction[0] / dir_len, direction[1] / dir_len]
        } else {
            [1.0, 0.0]
        };

        Self {
            amplitude: amplitude.max(0.0),
            wavelength: wavelength.max(0.1),
            steepness: steepness.clamp(0.0, 1.0),
            speed,
            direction: normalized,
            _padding: [0.0; 2],
        }
    }

    /// Create a wave with default parameters.
    pub fn default_wave() -> Self {
        Self::new(0.5, 10.0, 0.5, 1.0, [1.0, 0.0])
    }

    /// Wave number k = 2*PI / wavelength.
    #[inline]
    pub fn wave_number(&self) -> f32 {
        TAU / self.wavelength
    }

    /// Angular frequency from deep water dispersion relation.
    ///
    /// omega^2 = g * k, so omega = sqrt(g * k)
    #[inline]
    pub fn angular_frequency(&self) -> f32 {
        let k = self.wave_number();
        (GRAVITY * k).sqrt()
    }

    /// Phase velocity c = omega / k = sqrt(g / k).
    #[inline]
    pub fn phase_velocity(&self) -> f32 {
        let k = self.wave_number();
        (GRAVITY / k).sqrt()
    }

    /// Clamp steepness to prevent wave looping.
    ///
    /// The maximum safe steepness for a wave in a set is:
    /// Q_max = 1 / (k * A * n)
    ///
    /// Where k = wave number, A = amplitude, n = number of waves.
    pub fn clamp_steepness(&mut self, num_waves: usize) {
        if num_waves == 0 || self.amplitude < EPSILON {
            return;
        }

        let k = self.wave_number();
        let n = num_waves as f32;
        let q_max = 1.0 / (k * self.amplitude * n);
        self.steepness = self.steepness.min(q_max);
    }

    /// Compute wave phase at position (x, z) and time t.
    #[inline]
    fn phase(&self, x: f32, z: f32, time: f32) -> f32 {
        let k = self.wave_number();
        let omega = self.angular_frequency();
        let dot = self.direction[0] * x + self.direction[1] * z;
        k * dot - omega * time * self.speed
    }

    /// Evaluate displacement at position (x, z).
    fn evaluate_single(&self, x: f32, z: f32, time: f32) -> [f32; 3] {
        let phase = self.phase(x, z, time);
        let cos_phase = phase.cos();
        let sin_phase = phase.sin();

        // Q factor for horizontal displacement
        let q = self.steepness;

        // Displacement
        let dx = q * self.amplitude * self.direction[0] * cos_phase;
        let dz = q * self.amplitude * self.direction[1] * cos_phase;
        let dy = self.amplitude * sin_phase;

        [dx, dy, dz]
    }

    /// Evaluate displacement gradient for normal calculation.
    fn evaluate_gradient(&self, x: f32, z: f32, time: f32, _num_waves: usize) -> [[f32; 3]; 2] {
        let phase = self.phase(x, z, time);
        let cos_phase = phase.cos();
        let sin_phase = phase.sin();
        let k = self.wave_number();
        let q = self.steepness;

        // Partial derivatives
        // d/dx of position
        let wa = k * self.amplitude;

        // dP/dx
        let dx_dx = -q * wa * self.direction[0] * self.direction[0] * sin_phase;
        let dy_dx = wa * self.direction[0] * cos_phase;
        let dz_dx = -q * wa * self.direction[0] * self.direction[1] * sin_phase;

        // dP/dz
        let dx_dz = -q * wa * self.direction[0] * self.direction[1] * sin_phase;
        let dy_dz = wa * self.direction[1] * cos_phase;
        let dz_dz = -q * wa * self.direction[1] * self.direction[1] * sin_phase;

        [
            [dx_dx, dy_dx, dz_dx], // partial derivative wrt x
            [dx_dz, dy_dz, dz_dz], // partial derivative wrt z
        ]
    }
}

impl Default for GerstnerWave {
    fn default() -> Self {
        Self::default_wave()
    }
}

// ---------------------------------------------------------------------------
// GerstnerResult
// ---------------------------------------------------------------------------

/// Result of Gerstner wave evaluation at a point.
#[derive(Clone, Copy, Debug, Default)]
pub struct GerstnerResult {
    /// Displaced world position [x, y, z].
    pub position: [f32; 3],
    /// Surface normal (normalized).
    pub normal: [f32; 3],
}

impl GerstnerResult {
    /// Create a result with default values (flat surface at origin).
    pub fn flat() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            normal: [0.0, 1.0, 0.0],
        }
    }
}

// ---------------------------------------------------------------------------
// WavePreset
// ---------------------------------------------------------------------------

/// Preset wave configurations for common ocean states.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WavePreset {
    /// Calm conditions: 4 waves, small amplitude.
    Calm,
    /// Moderate conditions: 8 waves, medium amplitude.
    Moderate,
    /// Choppy conditions: 16 waves, high steepness.
    Choppy,
    /// Storm conditions: 24 waves, large amplitude + high steepness.
    Storm,
}

impl WavePreset {
    /// Get waves for this preset.
    pub fn waves(&self) -> Vec<GerstnerWave> {
        match self {
            WavePreset::Calm => Self::calm_waves(),
            WavePreset::Moderate => Self::moderate_waves(),
            WavePreset::Choppy => Self::choppy_waves(),
            WavePreset::Storm => Self::storm_waves(),
        }
    }

    fn calm_waves() -> Vec<GerstnerWave> {
        vec![
            GerstnerWave::new(0.1, 20.0, 0.3, 1.0, [1.0, 0.0]),
            GerstnerWave::new(0.08, 15.0, 0.25, 1.1, [0.8, 0.6]),
            GerstnerWave::new(0.06, 10.0, 0.2, 0.9, [0.5, -0.866]),
            GerstnerWave::new(0.04, 8.0, 0.15, 1.2, [-0.3, 0.95]),
        ]
    }

    fn moderate_waves() -> Vec<GerstnerWave> {
        vec![
            GerstnerWave::new(0.5, 40.0, 0.4, 1.0, [1.0, 0.0]),
            GerstnerWave::new(0.4, 30.0, 0.35, 1.1, [0.866, 0.5]),
            GerstnerWave::new(0.3, 25.0, 0.3, 0.95, [0.5, 0.866]),
            GerstnerWave::new(0.25, 20.0, 0.28, 1.05, [0.0, 1.0]),
            GerstnerWave::new(0.2, 15.0, 0.25, 1.15, [-0.5, 0.866]),
            GerstnerWave::new(0.15, 12.0, 0.22, 0.9, [-0.866, 0.5]),
            GerstnerWave::new(0.1, 10.0, 0.2, 1.2, [-1.0, 0.0]),
            GerstnerWave::new(0.08, 8.0, 0.18, 1.0, [0.707, -0.707]),
        ]
    }

    fn choppy_waves() -> Vec<GerstnerWave> {
        let mut waves = Vec::with_capacity(16);
        let base_amplitude = 0.8;
        let base_wavelength = 50.0;

        for i in 0..16 {
            let angle = (i as f32 / 16.0) * TAU;
            let scale = 1.0 - (i as f32 / 16.0) * 0.6;
            let amplitude = base_amplitude * scale;
            let wavelength = base_wavelength * scale.max(0.3);
            let steepness = 0.5 + (i as f32 / 16.0) * 0.3;
            let speed = 0.8 + (i as f32 / 16.0) * 0.4;
            let direction = [angle.cos(), angle.sin()];

            waves.push(GerstnerWave::new(
                amplitude,
                wavelength,
                steepness.min(0.9),
                speed,
                direction,
            ));
        }

        waves
    }

    fn storm_waves() -> Vec<GerstnerWave> {
        let mut waves = Vec::with_capacity(24);
        let base_amplitude = 2.0;
        let base_wavelength = 80.0;

        for i in 0..24 {
            let angle = (i as f32 / 24.0) * TAU + 0.1 * (i as f32);
            let scale = 1.0 - (i as f32 / 24.0) * 0.7;
            let amplitude = base_amplitude * scale;
            let wavelength = base_wavelength * scale.max(0.2);
            let steepness = 0.6 + (i as f32 / 24.0) * 0.35;
            let speed = 1.0 + (i as f32 / 24.0) * 0.5;
            let direction = [angle.cos(), angle.sin()];

            waves.push(GerstnerWave::new(
                amplitude,
                wavelength,
                steepness.min(0.95),
                speed,
                direction,
            ));
        }

        waves
    }
}

impl Default for WavePreset {
    fn default() -> Self {
        WavePreset::Moderate
    }
}

// ---------------------------------------------------------------------------
// GerstnerWaveSet
// ---------------------------------------------------------------------------

/// Collection of Gerstner waves with animation state.
#[derive(Clone, Debug)]
pub struct GerstnerWaveSet {
    /// Individual waves in this set.
    waves: Vec<GerstnerWave>,
    /// Current animation time.
    time: f32,
}

impl GerstnerWaveSet {
    /// Create an empty wave set.
    pub fn new() -> Self {
        Self {
            waves: Vec::new(),
            time: 0.0,
        }
    }

    /// Create a wave set with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            waves: Vec::with_capacity(capacity),
            time: 0.0,
        }
    }

    /// Create a wave set from a preset configuration.
    pub fn from_preset(preset: WavePreset) -> Self {
        let waves = preset.waves();
        let mut set = Self {
            waves,
            time: 0.0,
        };
        set.clamp_all_steepness();
        set
    }

    /// Add a wave to the set.
    ///
    /// # Panics
    ///
    /// Panics if adding would exceed MAX_WAVES.
    pub fn add_wave(&mut self, wave: GerstnerWave) {
        assert!(
            self.waves.len() < MAX_WAVES,
            "Cannot add wave: maximum of {} waves exceeded",
            MAX_WAVES
        );
        self.waves.push(wave);
        self.clamp_all_steepness();
    }

    /// Get number of waves in the set.
    #[inline]
    pub fn wave_count(&self) -> usize {
        self.waves.len()
    }

    /// Check if the wave set is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.waves.is_empty()
    }

    /// Get a reference to the waves slice.
    #[inline]
    pub fn waves(&self) -> &[GerstnerWave] {
        &self.waves
    }

    /// Get a mutable reference to the waves slice.
    #[inline]
    pub fn waves_mut(&mut self) -> &mut [GerstnerWave] {
        &mut self.waves
    }

    /// Set the animation time.
    #[inline]
    pub fn set_time(&mut self, time: f32) {
        self.time = time;
    }

    /// Get the current animation time.
    #[inline]
    pub fn time(&self) -> f32 {
        self.time
    }

    /// Advance the animation time by delta.
    #[inline]
    pub fn advance_time(&mut self, delta: f32) {
        self.time += delta;
    }

    /// Clear all waves.
    pub fn clear(&mut self) {
        self.waves.clear();
    }

    /// Clamp all wave steepness values to prevent looping.
    fn clamp_all_steepness(&mut self) {
        let n = self.waves.len();
        for wave in &mut self.waves {
            wave.clamp_steepness(n);
        }
    }

    /// Compute displacement at (x, z) for current time.
    ///
    /// Returns the displaced position and surface normal.
    pub fn evaluate(&self, x: f32, z: f32) -> GerstnerResult {
        self.evaluate_at_time(x, z, self.time)
    }

    /// Compute displacement at (x, z) for specified time.
    pub fn evaluate_at_time(&self, x: f32, z: f32, time: f32) -> GerstnerResult {
        if self.waves.is_empty() {
            return GerstnerResult {
                position: [x, 0.0, z],
                normal: [0.0, 1.0, 0.0],
            };
        }

        // Sum displacements from all waves
        let mut total_x = x;
        let mut total_y = 0.0f32;
        let mut total_z = z;

        for wave in &self.waves {
            let [dx, dy, dz] = wave.evaluate_single(x, z, time);
            total_x += dx;
            total_y += dy;
            total_z += dz;
        }

        // Compute normal from gradients
        let normal = self.evaluate_normal_at_time(x, z, time);

        GerstnerResult {
            position: [total_x, total_y, total_z],
            normal,
        }
    }

    /// Compute surface normal at (x, z) for current time.
    pub fn evaluate_normal(&self, x: f32, z: f32) -> [f32; 3] {
        self.evaluate_normal_at_time(x, z, self.time)
    }

    /// Compute analytic surface normal from wave gradients.
    pub fn evaluate_normal_at_time(&self, x: f32, z: f32, time: f32) -> [f32; 3] {
        if self.waves.is_empty() {
            return [0.0, 1.0, 0.0];
        }

        let n = self.waves.len();

        // Sum gradients from all waves
        let mut dx_total = [0.0f32, 0.0, 0.0];
        let mut dz_total = [0.0f32, 0.0, 0.0];

        for wave in &self.waves {
            let [dx, dz] = wave.evaluate_gradient(x, z, time, n);
            dx_total[0] += dx[0];
            dx_total[1] += dx[1];
            dx_total[2] += dx[2];
            dz_total[0] += dz[0];
            dz_total[1] += dz[1];
            dz_total[2] += dz[2];
        }

        // Tangent vectors
        // T_x = (1 + sum(dx_dx), sum(dy_dx), sum(dz_dx))
        // T_z = (sum(dx_dz), sum(dy_dz), 1 + sum(dz_dz))
        let t_x = [1.0 + dx_total[0], dx_total[1], dx_total[2]];
        let t_z = [dz_total[0], dz_total[1], 1.0 + dz_total[2]];

        // Normal = T_z cross T_x (order for upward-facing normal)
        let normal = [
            t_z[1] * t_x[2] - t_z[2] * t_x[1],
            t_z[2] * t_x[0] - t_z[0] * t_x[2],
            t_z[0] * t_x[1] - t_z[1] * t_x[0],
        ];

        // Normalize
        let len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        if len > EPSILON {
            [normal[0] / len, normal[1] / len, normal[2] / len]
        } else {
            [0.0, 1.0, 0.0]
        }
    }

    /// Batch evaluate a grid of points.
    ///
    /// # Arguments
    ///
    /// * `grid_size` - Number of points per side (total = grid_size^2).
    /// * `spacing` - Distance between grid points.
    /// * `origin` - Grid origin (x, z).
    ///
    /// # Returns
    ///
    /// Vec of GerstnerResult, row-major order (z varies fastest).
    pub fn evaluate_grid(
        &self,
        grid_size: usize,
        spacing: f32,
        origin: [f32; 2],
    ) -> Vec<GerstnerResult> {
        let total = grid_size * grid_size;
        let mut results = Vec::with_capacity(total);

        for row in 0..grid_size {
            let x = origin[0] + (row as f32) * spacing;
            for col in 0..grid_size {
                let z = origin[1] + (col as f32) * spacing;
                results.push(self.evaluate(x, z));
            }
        }

        results
    }

    /// Get GPU-compatible wave data as byte slice.
    ///
    /// Pads to MAX_WAVES for uniform buffer requirements.
    pub fn as_gpu_bytes(&self) -> Vec<u8> {
        let mut padded = vec![GerstnerWave::default(); MAX_WAVES];
        let copy_count = self.waves.len().min(MAX_WAVES);
        padded[..copy_count].copy_from_slice(&self.waves[..copy_count]);

        bytemuck::cast_slice(&padded).to_vec()
    }

    /// Get wave count and time as GPU-compatible uniform.
    pub fn wave_params(&self) -> GerstnerWaveParams {
        GerstnerWaveParams {
            wave_count: self.waves.len() as u32,
            time: self.time,
            _padding: [0; 2],
        }
    }
}

impl Default for GerstnerWaveSet {
    fn default() -> Self {
        Self::new()
    }
}

impl From<WavePreset> for GerstnerWaveSet {
    fn from(preset: WavePreset) -> Self {
        Self::from_preset(preset)
    }
}

// ---------------------------------------------------------------------------
// GerstnerWaveParams
// ---------------------------------------------------------------------------

/// GPU uniform buffer for wave evaluation parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GerstnerWaveParams {
    /// Number of active waves.
    pub wave_count: u32,
    /// Current animation time.
    pub time: f32,
    /// Padding for 16-byte alignment.
    pub _padding: [u32; 2],
}

// ---------------------------------------------------------------------------
// GridParams
// ---------------------------------------------------------------------------

/// GPU uniform buffer for grid evaluation parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GridParams {
    /// Number of vertices per side.
    pub grid_size: u32,
    /// Spacing between vertices.
    pub spacing: f32,
    /// Grid origin X.
    pub origin_x: f32,
    /// Grid origin Z.
    pub origin_z: f32,
}

impl GridParams {
    /// Create grid parameters.
    pub fn new(grid_size: u32, spacing: f32, origin: [f32; 2]) -> Self {
        Self {
            grid_size,
            spacing,
            origin_x: origin[0],
            origin_z: origin[1],
        }
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

    fn approx_eq_vec3(a: [f32; 3], b: [f32; 3]) -> bool {
        approx_eq(a[0], b[0]) && approx_eq(a[1], b[1]) && approx_eq(a[2], b[2])
    }

    fn normalize_vec3(v: [f32; 3]) -> [f32; 3] {
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if len > EPSILON {
            [v[0] / len, v[1] / len, v[2] / len]
        } else {
            [0.0, 1.0, 0.0]
        }
    }

    fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    fn vec3_length(v: [f32; 3]) -> f32 {
        (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
    }

    // Test 1: GerstnerWave struct size
    #[test]
    fn test_wave_struct_size() {
        assert_eq!(mem::size_of::<GerstnerWave>(), GERSTNER_WAVE_SIZE);
        assert_eq!(mem::size_of::<GerstnerWave>(), 32);
    }

    // Test 2: Wave number calculation
    #[test]
    fn test_wave_number() {
        let wave = GerstnerWave::new(1.0, 10.0, 0.5, 1.0, [1.0, 0.0]);
        let k = wave.wave_number();
        assert!(approx_eq(k, TAU / 10.0));
    }

    // Test 3: Angular frequency (dispersion relation)
    #[test]
    fn test_angular_frequency() {
        let wave = GerstnerWave::new(1.0, 10.0, 0.5, 1.0, [1.0, 0.0]);
        let k = wave.wave_number();
        let omega = wave.angular_frequency();
        // omega^2 = g * k
        let expected = (GRAVITY * k).sqrt();
        assert!(approx_eq(omega, expected));
    }

    // Test 4: Phase velocity
    #[test]
    fn test_phase_velocity() {
        let wave = GerstnerWave::new(1.0, 10.0, 0.5, 1.0, [1.0, 0.0]);
        let k = wave.wave_number();
        let c = wave.phase_velocity();
        // c = sqrt(g / k)
        let expected = (GRAVITY / k).sqrt();
        assert!(approx_eq(c, expected));
    }

    // Test 5: Steepness clamping
    #[test]
    fn test_steepness_clamping() {
        let mut wave = GerstnerWave::new(1.0, 10.0, 1.0, 1.0, [1.0, 0.0]);
        let k = wave.wave_number();
        let n = 4;
        wave.clamp_steepness(n);

        // Q_max = 1 / (k * A * n)
        let q_max = 1.0 / (k * 1.0 * n as f32);
        assert!(wave.steepness <= q_max + EPSILON);
    }

    // Test 6: Direction normalization
    #[test]
    fn test_direction_normalization() {
        let wave = GerstnerWave::new(1.0, 10.0, 0.5, 1.0, [3.0, 4.0]);
        let dir = wave.direction;
        let len = (dir[0] * dir[0] + dir[1] * dir[1]).sqrt();
        assert!(approx_eq(len, 1.0));
        assert!(approx_eq(dir[0], 0.6));
        assert!(approx_eq(dir[1], 0.8));
    }

    // Test 7: Zero direction falls back to [1, 0]
    #[test]
    fn test_zero_direction_fallback() {
        let wave = GerstnerWave::new(1.0, 10.0, 0.5, 1.0, [0.0, 0.0]);
        assert!(approx_eq(wave.direction[0], 1.0));
        assert!(approx_eq(wave.direction[1], 0.0));
    }

    // Test 8: Single wave evaluation at origin, time=0
    #[test]
    fn test_single_wave_evaluation_origin() {
        let wave = GerstnerWave::new(0.5, 10.0, 0.0, 1.0, [1.0, 0.0]);
        let mut set = GerstnerWaveSet::new();
        set.waves.push(wave);
        set.set_time(0.0);

        let result = set.evaluate(0.0, 0.0);
        // At x=0, z=0, t=0: phase = 0, sin(0)=0, cos(0)=1
        // dy = A * sin(0) = 0
        // With steepness=0, dx=dz=0
        assert!(approx_eq(result.position[1], 0.0));
    }

    // Test 9: Single wave evaluation with displacement
    #[test]
    fn test_single_wave_displacement() {
        let wave = GerstnerWave::new(1.0, TAU, 0.0, 1.0, [1.0, 0.0]);
        // wavelength = 2*PI, so k = 1
        // At x = PI/2: phase = 1 * PI/2 - omega * 0 = PI/2
        // sin(PI/2) = 1, so dy = A = 1.0

        let mut set = GerstnerWaveSet::new();
        set.waves.push(wave);
        set.set_time(0.0);

        let result = set.evaluate(PI / 2.0, 0.0);
        assert!(approx_eq(result.position[1], 1.0));
    }

    // Test 10: Multi-wave superposition
    #[test]
    fn test_multi_wave_superposition() {
        // Two identical waves should double the displacement
        let wave = GerstnerWave::new(0.5, TAU, 0.0, 1.0, [1.0, 0.0]);

        let mut set = GerstnerWaveSet::new();
        set.waves.push(wave);
        set.waves.push(wave);
        set.set_time(0.0);

        let single = GerstnerWave::new(0.5, TAU, 0.0, 1.0, [1.0, 0.0])
            .evaluate_single(PI / 2.0, 0.0, 0.0);

        let result = set.evaluate(PI / 2.0, 0.0);
        assert!(approx_eq(result.position[1], 2.0 * single[1]));
    }

    // Test 11: Normal at flat surface
    #[test]
    fn test_normal_flat_surface() {
        let set = GerstnerWaveSet::new();
        let normal = set.evaluate_normal(0.0, 0.0);
        assert!(approx_eq_vec3(normal, [0.0, 1.0, 0.0]));
    }

    // Test 12: Normal approximately up for small amplitude
    #[test]
    fn test_normal_small_amplitude() {
        let wave = GerstnerWave::new(0.01, 100.0, 0.1, 1.0, [1.0, 0.0]);
        let mut set = GerstnerWaveSet::new();
        set.waves.push(wave);
        set.set_time(0.0);

        let normal = set.evaluate_normal(0.0, 0.0);
        // Should be close to up vector
        let up = [0.0f32, 1.0, 0.0];
        let d = dot(normal, up);
        assert!(d > 0.99, "Normal should be approximately up, dot = {}", d);
    }

    // Test 13: Normal is normalized
    #[test]
    fn test_normal_normalized() {
        let mut set = GerstnerWaveSet::from_preset(WavePreset::Choppy);
        set.set_time(1.5);

        for x in [0.0, 5.0, -3.0, 10.0] {
            for z in [0.0, 2.0, -7.0, 15.0] {
                let normal = set.evaluate_normal(x, z);
                let len = vec3_length(normal);
                assert!(
                    approx_eq(len, 1.0),
                    "Normal at ({}, {}) has length {}",
                    x,
                    z,
                    len
                );
            }
        }
    }

    // Test 14: WavePreset::Calm wave count
    #[test]
    fn test_preset_calm() {
        let set = GerstnerWaveSet::from_preset(WavePreset::Calm);
        assert_eq!(set.wave_count(), 4);
    }

    // Test 15: WavePreset::Moderate wave count
    #[test]
    fn test_preset_moderate() {
        let set = GerstnerWaveSet::from_preset(WavePreset::Moderate);
        assert_eq!(set.wave_count(), 8);
    }

    // Test 16: WavePreset::Choppy wave count
    #[test]
    fn test_preset_choppy() {
        let set = GerstnerWaveSet::from_preset(WavePreset::Choppy);
        assert_eq!(set.wave_count(), 16);
    }

    // Test 17: WavePreset::Storm wave count
    #[test]
    fn test_preset_storm() {
        let set = GerstnerWaveSet::from_preset(WavePreset::Storm);
        assert_eq!(set.wave_count(), 24);
    }

    // Test 18: Grid evaluation size
    #[test]
    fn test_grid_evaluation_size() {
        let set = GerstnerWaveSet::from_preset(WavePreset::Calm);
        let grid = set.evaluate_grid(8, 1.0, [0.0, 0.0]);
        assert_eq!(grid.len(), 64);
    }

    // Test 19: Grid evaluation positions
    #[test]
    fn test_grid_evaluation_positions() {
        let set = GerstnerWaveSet::new(); // No waves = no displacement
        let grid = set.evaluate_grid(4, 2.0, [10.0, 20.0]);

        // First point should be at origin
        assert!(approx_eq(grid[0].position[0], 10.0));
        assert!(approx_eq(grid[0].position[2], 20.0));

        // Last point should be at (10 + 3*2, 20 + 3*2) = (16, 26)
        let last = &grid[15];
        assert!(approx_eq(last.position[0], 16.0));
        assert!(approx_eq(last.position[2], 26.0));
    }

    // Test 20: Time animation changes result
    #[test]
    fn test_time_animation() {
        let mut set = GerstnerWaveSet::from_preset(WavePreset::Moderate);

        set.set_time(0.0);
        let result_t0 = set.evaluate(5.0, 5.0);

        set.set_time(1.0);
        let result_t1 = set.evaluate(5.0, 5.0);

        // Results should differ
        assert!(!approx_eq(result_t0.position[1], result_t1.position[1]));
    }

    // Test 21: Zero amplitude wave has no effect
    #[test]
    fn test_zero_amplitude() {
        let wave = GerstnerWave::new(0.0, 10.0, 0.5, 1.0, [1.0, 0.0]);
        let mut set = GerstnerWaveSet::new();
        set.waves.push(wave);
        set.set_time(1.0);

        let result = set.evaluate(5.0, 5.0);
        assert!(approx_eq(result.position[0], 5.0));
        assert!(approx_eq(result.position[1], 0.0));
        assert!(approx_eq(result.position[2], 5.0));
    }

    // Test 22: Very long wavelength is nearly flat
    #[test]
    fn test_very_long_wavelength() {
        let wave = GerstnerWave::new(0.5, 10000.0, 0.5, 1.0, [1.0, 0.0]);
        let mut set = GerstnerWaveSet::new();
        set.waves.push(wave);
        set.set_time(0.0);

        // Sample points should have similar heights
        let r1 = set.evaluate(0.0, 0.0);
        let r2 = set.evaluate(10.0, 0.0);

        let height_diff = (r1.position[1] - r2.position[1]).abs();
        assert!(height_diff < 0.01, "Height diff = {}", height_diff);
    }

    // Test 23: Empty wave set returns original position
    #[test]
    fn test_empty_wave_set() {
        let set = GerstnerWaveSet::new();
        let result = set.evaluate(100.0, 50.0);

        assert!(approx_eq(result.position[0], 100.0));
        assert!(approx_eq(result.position[1], 0.0));
        assert!(approx_eq(result.position[2], 50.0));
    }

    // Test 24: MAX_WAVES limit
    #[test]
    fn test_max_waves_constant() {
        assert_eq!(MAX_WAVES, 32);
    }

    // Test 25: add_wave increases count
    #[test]
    fn test_add_wave() {
        let mut set = GerstnerWaveSet::new();
        assert_eq!(set.wave_count(), 0);

        set.add_wave(GerstnerWave::default());
        assert_eq!(set.wave_count(), 1);

        set.add_wave(GerstnerWave::default());
        assert_eq!(set.wave_count(), 2);
    }

    // Test 26: clear removes all waves
    #[test]
    fn test_clear_waves() {
        let mut set = GerstnerWaveSet::from_preset(WavePreset::Storm);
        assert_eq!(set.wave_count(), 24);

        set.clear();
        assert_eq!(set.wave_count(), 0);
        assert!(set.is_empty());
    }

    // Test 27: advance_time works
    #[test]
    fn test_advance_time() {
        let mut set = GerstnerWaveSet::new();
        set.set_time(5.0);
        set.advance_time(1.5);
        assert!(approx_eq(set.time(), 6.5));
    }

    // Test 28: GPU bytes has correct size
    #[test]
    fn test_gpu_bytes_size() {
        let set = GerstnerWaveSet::from_preset(WavePreset::Calm);
        let bytes = set.as_gpu_bytes();
        assert_eq!(bytes.len(), MAX_WAVES * GERSTNER_WAVE_SIZE);
    }

    // Test 29: wave_params returns correct values
    #[test]
    fn test_wave_params() {
        let mut set = GerstnerWaveSet::from_preset(WavePreset::Moderate);
        set.set_time(3.14);

        let params = set.wave_params();
        assert_eq!(params.wave_count, 8);
        assert!(approx_eq(params.time, 3.14));
    }

    // Test 30: Perpendicular wave directions
    #[test]
    fn test_perpendicular_waves() {
        let wave_x = GerstnerWave::new(0.5, 10.0, 0.3, 1.0, [1.0, 0.0]);
        let wave_z = GerstnerWave::new(0.5, 10.0, 0.3, 1.0, [0.0, 1.0]);

        let mut set = GerstnerWaveSet::new();
        set.waves.push(wave_x);
        set.waves.push(wave_z);
        set.clamp_all_steepness();
        set.set_time(0.5);

        // Result should be different from either wave alone
        let result = set.evaluate(5.0, 5.0);
        assert!(!approx_eq(result.position[1], 0.0));
    }

    // Test 31: Opposite wave directions
    #[test]
    fn test_opposite_wave_directions() {
        let wave_pos = GerstnerWave::new(0.5, 10.0, 0.0, 1.0, [1.0, 0.0]);
        let wave_neg = GerstnerWave::new(0.5, 10.0, 0.0, 1.0, [-1.0, 0.0]);

        // At origin with t=0, both have phase=0, both add sin(0)=0
        // But at x=PI/(2k), one has sin(PI/2)=1, other has sin(-PI/2)=-1

        let mut set_pos = GerstnerWaveSet::new();
        set_pos.waves.push(wave_pos);

        let mut set_neg = GerstnerWaveSet::new();
        set_neg.waves.push(wave_neg);

        let k = TAU / 10.0;
        let x_test = PI / (2.0 * k);

        let r_pos = set_pos.evaluate(x_test, 0.0);
        let r_neg = set_neg.evaluate(x_test, 0.0);

        // Heights should be opposite
        assert!(approx_eq(r_pos.position[1], -r_neg.position[1]));
    }

    // Test 32: Steepness affects horizontal displacement
    #[test]
    fn test_steepness_horizontal_displacement() {
        let wave_steep = GerstnerWave::new(0.5, 10.0, 0.8, 1.0, [1.0, 0.0]);
        let wave_flat = GerstnerWave::new(0.5, 10.0, 0.0, 1.0, [1.0, 0.0]);

        let mut set_steep = GerstnerWaveSet::new();
        set_steep.waves.push(wave_steep);

        let mut set_flat = GerstnerWaveSet::new();
        set_flat.waves.push(wave_flat);

        let r_steep = set_steep.evaluate(5.0, 0.0);
        let r_flat = set_flat.evaluate(5.0, 0.0);

        // Flat wave should have minimal X displacement
        let dx_flat = (r_flat.position[0] - 5.0).abs();
        let dx_steep = (r_steep.position[0] - 5.0).abs();

        assert!(
            dx_steep > dx_flat,
            "Steep wave should displace more: {} vs {}",
            dx_steep,
            dx_flat
        );
    }

    // Test 33: Negative amplitude handled
    #[test]
    fn test_negative_amplitude_clamped() {
        let wave = GerstnerWave::new(-1.0, 10.0, 0.5, 1.0, [1.0, 0.0]);
        assert!(wave.amplitude >= 0.0);
    }

    // Test 34: Wavelength minimum enforced
    #[test]
    fn test_wavelength_minimum() {
        let wave = GerstnerWave::new(1.0, 0.001, 0.5, 1.0, [1.0, 0.0]);
        assert!(wave.wavelength >= 0.1);
    }

    // Test 35: Steepness clamped to [0, 1]
    #[test]
    fn test_steepness_range() {
        let wave_high = GerstnerWave::new(1.0, 10.0, 1.5, 1.0, [1.0, 0.0]);
        let wave_low = GerstnerWave::new(1.0, 10.0, -0.5, 1.0, [1.0, 0.0]);

        assert!(wave_high.steepness <= 1.0);
        assert!(wave_low.steepness >= 0.0);
    }

    // Test 36: Default wave values
    #[test]
    fn test_default_wave() {
        let wave = GerstnerWave::default();
        assert!(approx_eq(wave.amplitude, 0.5));
        assert!(approx_eq(wave.wavelength, 10.0));
        assert!(approx_eq(wave.steepness, 0.5));
        assert!(approx_eq(wave.speed, 1.0));
    }

    // Test 37: GerstnerResult default
    #[test]
    fn test_gerstner_result_default() {
        let result = GerstnerResult::default();
        assert!(approx_eq_vec3(result.position, [0.0, 0.0, 0.0]));
        assert!(approx_eq_vec3(result.normal, [0.0, 0.0, 0.0]));
    }

    // Test 38: GerstnerResult::flat()
    #[test]
    fn test_gerstner_result_flat() {
        let result = GerstnerResult::flat();
        assert!(approx_eq_vec3(result.position, [0.0, 0.0, 0.0]));
        assert!(approx_eq_vec3(result.normal, [0.0, 1.0, 0.0]));
    }

    // Test 39: GridParams creation
    #[test]
    fn test_grid_params() {
        let params = GridParams::new(64, 0.5, [10.0, 20.0]);
        assert_eq!(params.grid_size, 64);
        assert!(approx_eq(params.spacing, 0.5));
        assert!(approx_eq(params.origin_x, 10.0));
        assert!(approx_eq(params.origin_z, 20.0));
    }

    // Test 40: WavePreset From conversion
    #[test]
    fn test_wave_preset_from() {
        let set: GerstnerWaveSet = WavePreset::Choppy.into();
        assert_eq!(set.wave_count(), 16);
    }

    // Test 41: waves() accessor
    #[test]
    fn test_waves_accessor() {
        let set = GerstnerWaveSet::from_preset(WavePreset::Calm);
        let waves = set.waves();
        assert_eq!(waves.len(), 4);
    }

    // Test 42: waves_mut() accessor
    #[test]
    fn test_waves_mut_accessor() {
        let mut set = GerstnerWaveSet::from_preset(WavePreset::Calm);
        let waves = set.waves_mut();
        waves[0].amplitude = 10.0;
        assert!(approx_eq(set.waves()[0].amplitude, 10.0));
    }

    // Test 43: with_capacity constructor
    #[test]
    fn test_with_capacity() {
        let set = GerstnerWaveSet::with_capacity(16);
        assert!(set.is_empty());
        // Can't easily test capacity, but should not panic
    }

    // Test 44: Deep water speed relationship
    #[test]
    fn test_deep_water_speed() {
        // Phase velocity should increase with wavelength
        let wave_short = GerstnerWave::new(1.0, 10.0, 0.5, 1.0, [1.0, 0.0]);
        let wave_long = GerstnerWave::new(1.0, 100.0, 0.5, 1.0, [1.0, 0.0]);

        let c_short = wave_short.phase_velocity();
        let c_long = wave_long.phase_velocity();

        assert!(
            c_long > c_short,
            "Longer waves should be faster: {} vs {}",
            c_long,
            c_short
        );
    }

    // Test 45: Wave period relationship
    #[test]
    fn test_wave_period() {
        let wave = GerstnerWave::new(1.0, 10.0, 0.5, 1.0, [1.0, 0.0]);
        let omega = wave.angular_frequency();
        let period = TAU / omega;

        // For wavelength 10m, period should be around 2.5 seconds
        assert!(period > 2.0 && period < 3.0);
    }

    // Test 46: Evaluate at multiple time steps shows periodicity
    #[test]
    fn test_wave_periodicity() {
        let wave = GerstnerWave::new(1.0, TAU, 0.0, 1.0, [1.0, 0.0]);
        // k = 1, omega = sqrt(g) ~= 3.13
        let omega = wave.angular_frequency();
        let period = TAU / omega;

        let mut set = GerstnerWaveSet::new();
        set.waves.push(wave);

        let x = 0.0;
        let z = 0.0;

        set.set_time(0.0);
        let r0 = set.evaluate(x, z);

        set.set_time(period);
        let r1 = set.evaluate(x, z);

        // After one period, should be back to same position
        assert!(
            approx_eq(r0.position[1], r1.position[1]),
            "Heights should match after one period: {} vs {}",
            r0.position[1],
            r1.position[1]
        );
    }

    // Test 47: Diagonal wave direction
    #[test]
    fn test_diagonal_direction() {
        let wave = GerstnerWave::new(0.5, 10.0, 0.3, 1.0, [1.0, 1.0]);
        // Direction should be normalized to [1/sqrt(2), 1/sqrt(2)]
        let expected = 1.0 / 2.0f32.sqrt();
        assert!(approx_eq(wave.direction[0], expected));
        assert!(approx_eq(wave.direction[1], expected));
    }

    // Test 48: Position displacement in wave direction
    #[test]
    fn test_displacement_follows_direction() {
        let wave = GerstnerWave::new(1.0, 10.0, 0.5, 1.0, [1.0, 0.0]);
        let mut set = GerstnerWaveSet::new();
        set.waves.push(wave);
        set.set_time(0.0);

        // With direction [1, 0] and steepness > 0, X should be displaced
        // but Z should remain unchanged
        let result = set.evaluate(0.0, 5.0);
        assert!(approx_eq(result.position[2], 5.0)); // Z unchanged
    }

    // Test 49: Speed multiplier affects animation
    #[test]
    fn test_speed_multiplier() {
        let wave_fast = GerstnerWave::new(1.0, TAU, 0.0, 2.0, [1.0, 0.0]);
        let wave_slow = GerstnerWave::new(1.0, TAU, 0.0, 0.5, [1.0, 0.0]);

        let mut set_fast = GerstnerWaveSet::new();
        set_fast.waves.push(wave_fast);

        let mut set_slow = GerstnerWaveSet::new();
        set_slow.waves.push(wave_slow);

        // At time 1, fast wave should be at a different phase than slow
        set_fast.set_time(1.0);
        set_slow.set_time(1.0);

        let r_fast = set_fast.evaluate(0.0, 0.0);
        let r_slow = set_slow.evaluate(0.0, 0.0);

        assert!(!approx_eq(r_fast.position[1], r_slow.position[1]));
    }

    // Test 50: GerstnerWaveParams size
    #[test]
    fn test_wave_params_size() {
        assert_eq!(mem::size_of::<GerstnerWaveParams>(), 16);
    }

    // Test 51: GridParams size
    #[test]
    fn test_grid_params_size() {
        assert_eq!(mem::size_of::<GridParams>(), 16);
    }

    // Test 52: Performance - evaluate grid doesn't panic on large size
    #[test]
    fn test_grid_performance() {
        let set = GerstnerWaveSet::from_preset(WavePreset::Moderate);
        let grid = set.evaluate_grid(32, 1.0, [0.0, 0.0]);
        assert_eq!(grid.len(), 1024);
    }

    // Test 53: WGSL shader validation
    #[test]
    fn test_shader_validation() {
        use naga::front::wgsl;
        use naga::valid::{Capabilities, ValidationFlags, Validator};

        let shader_src = include_str!("../../shaders/water/gerstner.comp.wgsl");

        let module = wgsl::parse_str(shader_src).expect("Failed to parse WGSL shader");

        let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
        validator
            .validate(&module)
            .expect("Failed to validate WGSL shader");
    }

    // Test 54: Shader has main entry point
    #[test]
    fn test_shader_entry_points() {
        use naga::front::wgsl;

        let shader_src = include_str!("../../shaders/water/gerstner.comp.wgsl");
        let module = wgsl::parse_str(shader_src).expect("Failed to parse WGSL shader");

        let entry_points: Vec<_> = module
            .entry_points
            .iter()
            .map(|ep| ep.name.as_str())
            .collect();

        assert!(
            entry_points.contains(&"main"),
            "Shader must have 'main' entry point"
        );
        assert!(
            entry_points.contains(&"evaluate_points"),
            "Shader must have 'evaluate_points' entry point"
        );
    }

    // Test 55: Shader workgroup size matches constant
    #[test]
    fn test_shader_workgroup_size() {
        use naga::front::wgsl;

        let shader_src = include_str!("../../shaders/water/gerstner.comp.wgsl");
        let module = wgsl::parse_str(shader_src).expect("Failed to parse WGSL shader");

        // Find the main entry point
        let main_entry = module
            .entry_points
            .iter()
            .find(|ep| ep.name == "main")
            .expect("No main entry point");

        // Check workgroup size (should be 8x8x1)
        assert_eq!(main_entry.workgroup_size[0], 8);
        assert_eq!(main_entry.workgroup_size[1], 8);
        assert_eq!(main_entry.workgroup_size[2], 1);
    }

    // Test 56: Bytemuck derive correctness
    #[test]
    fn test_bytemuck_cast() {
        let wave = GerstnerWave::default();
        let bytes: &[u8] = bytemuck::bytes_of(&wave);
        assert_eq!(bytes.len(), GERSTNER_WAVE_SIZE);

        let params = GerstnerWaveParams {
            wave_count: 8,
            time: 1.5,
            _padding: [0; 2],
        };
        let param_bytes: &[u8] = bytemuck::bytes_of(&params);
        assert_eq!(param_bytes.len(), 16);
    }
}
