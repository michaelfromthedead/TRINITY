//! Foam Generation for TRINITY Engine (T-ENV-2.9).
//!
//! Implements foam generation from wave dynamics:
//! - Crest foam from Jacobian determinant (wave folding)
//! - Shore foam from distance to shoreline
//! - Procedural noise for foam detail
//!
//! # Overview
//!
//! Foam appears on ocean surfaces in two primary locations:
//! 1. **Wave crests**: When waves fold over (Jacobian < 0), whitecaps form
//! 2. **Shoreline**: Breaking waves create foam bands near beaches
//!
//! # Physics
//!
//! The Jacobian determinant of the displacement field indicates wave folding:
//! ```text
//! J = (1 + dx/du) * (1 + dz/dv) - (dx/dv) * (dz/du)
//! ```
//!
//! When J < 0, the wave surface has folded over itself, creating foam.
//! Foam intensity: `foam = clamp(1 - J/threshold, 0, 1)`
//!
//! # Usage
//!
//! ```ignore
//! use renderer_backend::water::foam::{FoamConfig, FoamGenerator, FoamState};
//!
//! let config = FoamConfig::default();
//! let generator = FoamGenerator::new(config);
//!
//! // Compute Jacobian from wave displacement gradients
//! let jacobian = FoamGenerator::compute_jacobian([dx_du, dx_dv], [dz_du, dz_dv]);
//!
//! // Generate crest foam from Jacobian
//! let crest_foam = generator.crest_foam_from_jacobian(jacobian);
//!
//! // Generate shore foam
//! let shore_foam = generator.shore_foam(distance_to_shore, wave_height);
//! ```

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default crest threshold for foam generation.
pub const DEFAULT_CREST_THRESHOLD: f32 = 0.7;

/// Default foam decay rate per second.
pub const DEFAULT_DECAY_RATE: f32 = 2.0;

/// Default shore foam band width in meters.
pub const DEFAULT_SHORE_WIDTH: f32 = 5.0;

/// Default maximum shore foam intensity.
pub const DEFAULT_SHORE_FOAM_MAX: f32 = 1.0;

/// Default noise scale for foam detail.
pub const DEFAULT_NOISE_SCALE: f32 = 10.0;

/// Default noise intensity contribution.
pub const DEFAULT_NOISE_INTENSITY: f32 = 0.3;

/// FoamConfig struct size in bytes.
pub const FOAM_CONFIG_SIZE: usize = 32;

/// FoamState struct size in bytes.
pub const FOAM_STATE_SIZE: usize = 16;

/// Small epsilon for floating point comparisons.
const EPSILON: f32 = 1e-6;

// ---------------------------------------------------------------------------
// FoamConfig
// ---------------------------------------------------------------------------

/// Configuration for foam generation.
///
/// GPU-compatible layout for uniform buffers.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FoamConfig {
    /// Jacobian threshold for foam generation (default: 0.7).
    /// Lower values produce more foam, higher values less foam.
    pub crest_threshold: f32,

    /// Foam decay rate per second (default: 2.0).
    /// Higher values cause foam to dissipate faster.
    pub decay_rate: f32,

    /// Shore foam band width in meters (default: 5.0).
    /// Width of the foam band near shoreline.
    pub shore_width: f32,

    /// Maximum shore foam intensity (default: 1.0).
    pub shore_foam_max: f32,

    /// Noise scale for procedural detail (default: 10.0).
    /// Higher values produce finer noise patterns.
    pub noise_scale: f32,

    /// Noise intensity contribution (default: 0.3).
    /// How much noise affects final foam appearance.
    pub noise_intensity: f32,

    /// Padding for 16-byte alignment.
    pub _padding: [f32; 2],
}

impl Default for FoamConfig {
    fn default() -> Self {
        Self {
            crest_threshold: DEFAULT_CREST_THRESHOLD,
            decay_rate: DEFAULT_DECAY_RATE,
            shore_width: DEFAULT_SHORE_WIDTH,
            shore_foam_max: DEFAULT_SHORE_FOAM_MAX,
            noise_scale: DEFAULT_NOISE_SCALE,
            noise_intensity: DEFAULT_NOISE_INTENSITY,
            _padding: [0.0; 2],
        }
    }
}

impl FoamConfig {
    /// Create a new foam configuration with custom values.
    pub fn new(
        crest_threshold: f32,
        decay_rate: f32,
        shore_width: f32,
        shore_foam_max: f32,
        noise_scale: f32,
        noise_intensity: f32,
    ) -> Self {
        Self {
            crest_threshold: crest_threshold.max(EPSILON),
            decay_rate: decay_rate.max(0.0),
            shore_width: shore_width.max(EPSILON),
            shore_foam_max: shore_foam_max.clamp(0.0, 1.0),
            noise_scale: noise_scale.max(EPSILON),
            noise_intensity: noise_intensity.clamp(0.0, 1.0),
            _padding: [0.0; 2],
        }
    }

    /// Create configuration for calm water with minimal foam.
    pub fn calm() -> Self {
        Self {
            crest_threshold: 0.9,
            decay_rate: 3.0,
            shore_width: 3.0,
            shore_foam_max: 0.5,
            noise_scale: 15.0,
            noise_intensity: 0.2,
            _padding: [0.0; 2],
        }
    }

    /// Create configuration for moderate foam.
    pub fn moderate() -> Self {
        Self::default()
    }

    /// Create configuration for stormy conditions with heavy foam.
    pub fn stormy() -> Self {
        Self {
            crest_threshold: 0.5,
            decay_rate: 1.0,
            shore_width: 10.0,
            shore_foam_max: 1.0,
            noise_scale: 8.0,
            noise_intensity: 0.5,
            _padding: [0.0; 2],
        }
    }

    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.crest_threshold <= 0.0 {
            return Err("Crest threshold must be positive");
        }
        if self.decay_rate < 0.0 {
            return Err("Decay rate must be non-negative");
        }
        if self.shore_width <= 0.0 {
            return Err("Shore width must be positive");
        }
        if self.shore_foam_max < 0.0 || self.shore_foam_max > 1.0 {
            return Err("Shore foam max must be in [0, 1]");
        }
        if self.noise_scale <= 0.0 {
            return Err("Noise scale must be positive");
        }
        if self.noise_intensity < 0.0 || self.noise_intensity > 1.0 {
            return Err("Noise intensity must be in [0, 1]");
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// FoamState
// ---------------------------------------------------------------------------

/// Per-vertex or per-pixel foam state.
///
/// GPU-compatible layout for storage buffers.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FoamState {
    /// Current foam amount (0.0 to 1.0).
    pub foam_amount: f32,

    /// Decay timer for persistent foam.
    pub decay_timer: f32,

    /// Jacobian determinant at this point.
    pub jacobian: f32,

    /// Distance to nearest shoreline.
    pub shore_distance: f32,
}

impl FoamState {
    /// Create a new foam state.
    pub fn new(foam_amount: f32, decay_timer: f32, jacobian: f32, shore_distance: f32) -> Self {
        Self {
            foam_amount: foam_amount.clamp(0.0, 1.0),
            decay_timer: decay_timer.max(0.0),
            jacobian,
            shore_distance: shore_distance.max(0.0),
        }
    }

    /// Create a foam state with no foam.
    pub fn empty() -> Self {
        Self {
            foam_amount: 0.0,
            decay_timer: 0.0,
            jacobian: 1.0,
            shore_distance: f32::MAX,
        }
    }

    /// Check if this state has any visible foam.
    #[inline]
    pub fn has_foam(&self) -> bool {
        self.foam_amount > EPSILON
    }

    /// Check if this state represents a folded wave (Jacobian < 0).
    #[inline]
    pub fn is_folded(&self) -> bool {
        self.jacobian < 0.0
    }

    /// Check if this state is near the shore.
    #[inline]
    pub fn is_near_shore(&self, shore_width: f32) -> bool {
        self.shore_distance < shore_width
    }
}

// ---------------------------------------------------------------------------
// FoamGenerator
// ---------------------------------------------------------------------------

/// Foam generation system.
///
/// Generates foam from wave displacement Jacobian and shore distance.
#[derive(Clone, Debug)]
pub struct FoamGenerator {
    /// Configuration parameters.
    pub config: FoamConfig,
}

impl FoamGenerator {
    /// Create a new foam generator with the given configuration.
    pub fn new(config: FoamConfig) -> Self {
        Self { config }
    }

    /// Create a foam generator with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(FoamConfig::default())
    }

    /// Compute the Jacobian determinant of the wave displacement field.
    ///
    /// The Jacobian measures how the displacement field transforms area.
    /// When J < 0, the surface has folded over, indicating whitecap foam.
    ///
    /// # Arguments
    ///
    /// * `dx_duv` - Partial derivatives of X displacement [dx/du, dx/dv]
    /// * `dz_duv` - Partial derivatives of Z displacement [dz/du, dz/dv]
    ///
    /// # Returns
    ///
    /// Jacobian determinant: J = (1 + dx/du)(1 + dz/dv) - (dx/dv)(dz/du)
    ///
    /// # Physics
    ///
    /// - J > 1: Surface is stretched
    /// - J = 1: No deformation
    /// - 0 < J < 1: Surface is compressed
    /// - J < 0: Surface has folded (whitecap)
    #[inline]
    pub fn compute_jacobian(dx_duv: [f32; 2], dz_duv: [f32; 2]) -> f32 {
        let [dx_du, dx_dv] = dx_duv;
        let [dz_du, dz_dv] = dz_duv;

        // J = (1 + dx/du) * (1 + dz/dv) - (dx/dv) * (dz/du)
        (1.0 + dx_du) * (1.0 + dz_dv) - dx_dv * dz_du
    }

    /// Compute Jacobian from full 2x2 displacement gradient matrix.
    ///
    /// # Arguments
    ///
    /// * `gradient` - 2x2 matrix [[dx/du, dx/dv], [dz/du, dz/dv]]
    #[inline]
    pub fn compute_jacobian_from_matrix(gradient: [[f32; 2]; 2]) -> f32 {
        Self::compute_jacobian(gradient[0], gradient[1])
    }

    /// Generate crest foam intensity from Jacobian.
    ///
    /// # Arguments
    ///
    /// * `jacobian` - Jacobian determinant from displacement field
    ///
    /// # Returns
    ///
    /// Foam intensity in [0, 1]. Higher values for more negative Jacobian.
    #[inline]
    pub fn crest_foam_from_jacobian(&self, jacobian: f32) -> f32 {
        if jacobian >= self.config.crest_threshold {
            return 0.0;
        }

        // foam = clamp(1 - J/threshold, 0, 1)
        // When J = threshold: foam = 0
        // When J = 0: foam = 1
        // When J < 0: foam > 1, clamped to 1
        let foam = 1.0 - jacobian / self.config.crest_threshold;
        foam.clamp(0.0, 1.0)
    }

    /// Update foam decay over time.
    ///
    /// Foam persists briefly after whitecap formation, then fades.
    ///
    /// # Arguments
    ///
    /// * `current_foam` - Current foam amount
    /// * `dt` - Time step in seconds
    ///
    /// # Returns
    ///
    /// Updated foam amount after decay
    #[inline]
    pub fn update_decay(&self, current_foam: f32, dt: f32) -> f32 {
        let decayed = current_foam - self.config.decay_rate * dt;
        decayed.max(0.0)
    }

    /// Generate shore foam based on distance to shoreline.
    ///
    /// Shore foam increases as water gets shallower and waves break.
    ///
    /// # Arguments
    ///
    /// * `shore_distance` - Distance to nearest shoreline in meters
    /// * `wave_height` - Current wave height at this point
    ///
    /// # Returns
    ///
    /// Shore foam intensity in [0, 1]
    pub fn shore_foam(&self, shore_distance: f32, wave_height: f32) -> f32 {
        if shore_distance >= self.config.shore_width {
            return 0.0;
        }

        // Linear falloff from shore
        let distance_factor = 1.0 - (shore_distance / self.config.shore_width);

        // Wave height contribution (breaking waves have more foam)
        // Normalize wave height to a reasonable range (assume typical waves 0-3m)
        let height_factor = (wave_height.abs() / 3.0).clamp(0.0, 1.0);

        // Combined foam: stronger near shore, especially with high waves
        let foam = distance_factor * (0.5 + 0.5 * height_factor) * self.config.shore_foam_max;
        foam.clamp(0.0, 1.0)
    }

    /// Generate shore foam with breaking wave pattern.
    ///
    /// Adds wave breaking effect where foam peaks at a certain depth.
    ///
    /// # Arguments
    ///
    /// * `shore_distance` - Distance to nearest shoreline
    /// * `wave_height` - Current wave height
    /// * `water_depth` - Water depth at this point
    pub fn shore_foam_breaking(&self, shore_distance: f32, wave_height: f32, water_depth: f32) -> f32 {
        if shore_distance >= self.config.shore_width {
            return 0.0;
        }

        // Waves break when depth < wave_height * 1.28 (McCowan criterion)
        let breaking_ratio = wave_height.abs() / (water_depth.max(EPSILON) * 1.28);
        let breaking_factor = breaking_ratio.clamp(0.0, 1.0);

        // Distance falloff
        let distance_factor = 1.0 - (shore_distance / self.config.shore_width);

        // Breaking waves create maximum foam
        let foam = distance_factor * breaking_factor * self.config.shore_foam_max;
        foam.clamp(0.0, 1.0)
    }

    /// Sample procedural noise for foam detail.
    ///
    /// Uses a simple pseudo-random noise pattern for foam texture variation.
    ///
    /// # Arguments
    ///
    /// * `uv` - Texture coordinates [u, v]
    /// * `time` - Current animation time
    ///
    /// # Returns
    ///
    /// Noise value in [0, 1]
    pub fn sample_foam_noise(&self, uv: [f32; 2], time: f32) -> f32 {
        let scale = self.config.noise_scale;
        let intensity = self.config.noise_intensity;

        // Simple value noise using trig functions
        let u = uv[0] * scale + time * 0.1;
        let v = uv[1] * scale + time * 0.15;

        // Layer multiple frequencies
        let n1 = (u.sin() * v.cos()).abs();
        let n2 = ((u * 2.3).cos() * (v * 2.7).sin()).abs();
        let n3 = ((u * 4.1 + v * 3.9).sin()).abs();

        // Combine with falloff
        let noise = n1 * 0.5 + n2 * 0.3 + n3 * 0.2;
        noise * intensity
    }

    /// Sample foam noise with turbulence (multiple octaves).
    ///
    /// # Arguments
    ///
    /// * `uv` - Texture coordinates
    /// * `time` - Animation time
    /// * `octaves` - Number of noise octaves (1-4)
    pub fn sample_foam_noise_turbulent(&self, uv: [f32; 2], time: f32, octaves: u32) -> f32 {
        let octaves = octaves.clamp(1, 4);
        let scale = self.config.noise_scale;
        let intensity = self.config.noise_intensity;

        let mut noise = 0.0;
        let mut amplitude = 0.5;
        let mut frequency = 1.0;
        let mut max_amplitude = 0.0;

        for i in 0..octaves {
            let u = uv[0] * scale * frequency + time * 0.1 * (i as f32 + 1.0);
            let v = uv[1] * scale * frequency + time * 0.15 * (i as f32 + 1.0);

            // Pseudo-random based on position
            let hash = ((u * 127.1 + v * 311.7).sin() * 43758.5453).fract();
            noise += hash.abs() * amplitude;
            max_amplitude += amplitude;

            amplitude *= 0.5;
            frequency *= 2.0;
        }

        (noise / max_amplitude) * intensity
    }

    /// Compute the final combined foam value.
    ///
    /// Combines crest foam, shore foam, and noise.
    ///
    /// # Arguments
    ///
    /// * `state` - Current foam state
    ///
    /// # Returns
    ///
    /// Final foam intensity in [0, 1]
    pub fn compute_final_foam(&self, state: &FoamState) -> f32 {
        // Start with existing foam amount (includes decay)
        let mut foam = state.foam_amount;

        // Add crest foam from Jacobian
        let crest = self.crest_foam_from_jacobian(state.jacobian);
        foam = foam.max(crest);

        // Clamp final result
        foam.clamp(0.0, 1.0)
    }

    /// Compute final foam with UV coordinates for noise.
    ///
    /// # Arguments
    ///
    /// * `state` - Current foam state
    /// * `uv` - Texture coordinates
    /// * `time` - Animation time
    pub fn compute_final_foam_with_noise(
        &self,
        state: &FoamState,
        uv: [f32; 2],
        time: f32,
    ) -> f32 {
        let base_foam = self.compute_final_foam(state);

        if base_foam < EPSILON {
            return 0.0;
        }

        // Modulate with noise
        let noise = self.sample_foam_noise(uv, time);
        let foam = base_foam * (1.0 - self.config.noise_intensity + noise);

        foam.clamp(0.0, 1.0)
    }

    /// Update foam state for one frame.
    ///
    /// # Arguments
    ///
    /// * `state` - Current foam state (modified in place)
    /// * `jacobian` - Current Jacobian from wave displacement
    /// * `shore_distance` - Distance to shore
    /// * `wave_height` - Current wave height
    /// * `dt` - Time step in seconds
    pub fn update_state(
        &self,
        state: &mut FoamState,
        jacobian: f32,
        shore_distance: f32,
        wave_height: f32,
        dt: f32,
    ) {
        // Update stored values
        state.jacobian = jacobian;
        state.shore_distance = shore_distance;

        // Compute new foam sources
        let crest_foam = self.crest_foam_from_jacobian(jacobian);
        let shore_foam = self.shore_foam(shore_distance, wave_height);

        // Take maximum of current (decayed) and new foam
        let decayed = self.update_decay(state.foam_amount, dt);
        let new_foam = crest_foam.max(shore_foam);
        state.foam_amount = decayed.max(new_foam);

        // Update decay timer
        if new_foam > decayed {
            state.decay_timer = 0.0; // Reset timer when new foam appears
        } else {
            state.decay_timer += dt;
        }
    }

    /// Batch update multiple foam states.
    ///
    /// # Arguments
    ///
    /// * `states` - Slice of foam states to update
    /// * `jacobians` - Jacobian values for each state
    /// * `shore_distances` - Shore distances for each state
    /// * `wave_heights` - Wave heights for each state
    /// * `dt` - Time step in seconds
    pub fn update_states_batch(
        &self,
        states: &mut [FoamState],
        jacobians: &[f32],
        shore_distances: &[f32],
        wave_heights: &[f32],
        dt: f32,
    ) {
        debug_assert_eq!(states.len(), jacobians.len());
        debug_assert_eq!(states.len(), shore_distances.len());
        debug_assert_eq!(states.len(), wave_heights.len());

        for i in 0..states.len() {
            self.update_state(
                &mut states[i],
                jacobians[i],
                shore_distances[i],
                wave_heights[i],
                dt,
            );
        }
    }

    /// Estimate foam coverage (percentage of surface with foam).
    ///
    /// # Arguments
    ///
    /// * `states` - Slice of foam states
    /// * `threshold` - Minimum foam amount to count as covered
    pub fn estimate_coverage(&self, states: &[FoamState], threshold: f32) -> f32 {
        if states.is_empty() {
            return 0.0;
        }

        let covered = states.iter().filter(|s| s.foam_amount >= threshold).count();
        covered as f32 / states.len() as f32
    }

    /// Get average foam intensity.
    pub fn average_foam(&self, states: &[FoamState]) -> f32 {
        if states.is_empty() {
            return 0.0;
        }

        let sum: f32 = states.iter().map(|s| s.foam_amount).sum();
        sum / states.len() as f32
    }
}

impl Default for FoamGenerator {
    fn default() -> Self {
        Self::with_defaults()
    }
}

// ---------------------------------------------------------------------------
// FoamMask
// ---------------------------------------------------------------------------

/// Foam mask render target data.
///
/// Stores foam values for a 2D grid, suitable for GPU texture upload.
#[derive(Clone, Debug)]
pub struct FoamMask {
    /// Width of the foam mask.
    pub width: u32,

    /// Height of the foam mask.
    pub height: u32,

    /// Foam values (row-major, R8 normalized to 0-255).
    pub data: Vec<u8>,
}

impl FoamMask {
    /// Create a new foam mask with given dimensions.
    pub fn new(width: u32, height: u32) -> Self {
        let size = (width * height) as usize;
        Self {
            width,
            height,
            data: vec![0; size],
        }
    }

    /// Clear the foam mask to zero.
    pub fn clear(&mut self) {
        self.data.fill(0);
    }

    /// Set foam value at (x, y).
    ///
    /// # Arguments
    ///
    /// * `x` - X coordinate
    /// * `y` - Y coordinate
    /// * `value` - Foam intensity (0.0 to 1.0)
    #[inline]
    pub fn set(&mut self, x: u32, y: u32, value: f32) {
        if x < self.width && y < self.height {
            let idx = (y * self.width + x) as usize;
            self.data[idx] = (value.clamp(0.0, 1.0) * 255.0) as u8;
        }
    }

    /// Get foam value at (x, y).
    #[inline]
    pub fn get(&self, x: u32, y: u32) -> f32 {
        if x < self.width && y < self.height {
            let idx = (y * self.width + x) as usize;
            self.data[idx] as f32 / 255.0
        } else {
            0.0
        }
    }

    /// Sample with bilinear filtering.
    pub fn sample_bilinear(&self, u: f32, v: f32) -> f32 {
        let x = u * (self.width as f32 - 1.0);
        let y = v * (self.height as f32 - 1.0);

        let x0 = x.floor() as u32;
        let y0 = y.floor() as u32;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);

        let fx = x.fract();
        let fy = y.fract();

        let v00 = self.get(x0, y0);
        let v10 = self.get(x1, y0);
        let v01 = self.get(x0, y1);
        let v11 = self.get(x1, y1);

        let v0 = v00 * (1.0 - fx) + v10 * fx;
        let v1 = v01 * (1.0 - fx) + v11 * fx;

        v0 * (1.0 - fy) + v1 * fy
    }

    /// Fill from foam states.
    pub fn fill_from_states(&mut self, states: &[FoamState]) {
        let expected = (self.width * self.height) as usize;
        debug_assert_eq!(states.len(), expected);

        for (i, state) in states.iter().enumerate() {
            self.data[i] = (state.foam_amount.clamp(0.0, 1.0) * 255.0) as u8;
        }
    }

    /// Get raw bytes for GPU upload (R8 format).
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Total pixel count.
    #[inline]
    pub fn pixel_count(&self) -> usize {
        (self.width * self.height) as usize
    }
}

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

/// Compute Jacobian determinant from finite differences.
///
/// # Arguments
///
/// * `displacements` - 2D array of displacement vectors [x, z]
/// * `grid_spacing` - Distance between grid points
/// * `x` - X index in grid
/// * `z` - Z index in grid
/// * `width` - Grid width
/// * `height` - Grid height
pub fn jacobian_from_grid(
    displacements: &[[f32; 2]],
    grid_spacing: f32,
    x: usize,
    z: usize,
    width: usize,
    height: usize,
) -> f32 {
    let idx = |xi: usize, zi: usize| zi * width + xi;

    // Central differences where possible
    let x_prev = if x > 0 { x - 1 } else { x };
    let x_next = if x + 1 < width { x + 1 } else { x };
    let z_prev = if z > 0 { z - 1 } else { z };
    let z_next = if z + 1 < height { z + 1 } else { z };

    let h = grid_spacing * if x > 0 && x + 1 < width { 2.0 } else { 1.0 };
    let v = grid_spacing * if z > 0 && z + 1 < height { 2.0 } else { 1.0 };

    let dx_du = (displacements[idx(x_next, z)][0] - displacements[idx(x_prev, z)][0]) / h;
    let dx_dv = (displacements[idx(x, z_next)][0] - displacements[idx(x, z_prev)][0]) / v;
    let dz_du = (displacements[idx(x_next, z)][1] - displacements[idx(x_prev, z)][1]) / h;
    let dz_dv = (displacements[idx(x, z_next)][1] - displacements[idx(x, z_prev)][1]) / v;

    FoamGenerator::compute_jacobian([dx_du, dx_dv], [dz_du, dz_dv])
}

/// Compute shoreline distance for a point given a depth map.
///
/// # Arguments
///
/// * `depth_map` - 2D depth values (negative = underwater)
/// * `x` - X index
/// * `z` - Z index
/// * `width` - Grid width
/// * `height` - Grid height
/// * `cell_size` - Size of each cell in world units
pub fn shore_distance_from_depth(
    depth_map: &[f32],
    x: usize,
    z: usize,
    width: usize,
    height: usize,
    cell_size: f32,
) -> f32 {
    let idx = z * width + x;

    // If we're on land, distance is 0
    if depth_map[idx] >= 0.0 {
        return 0.0;
    }

    // Simple search for nearest shoreline
    let max_search = (width.max(height) / 4).max(16);
    let mut min_dist = f32::MAX;

    for radius in 1..=max_search {
        let r = radius as i32;

        // Check perimeter of search square
        for offset in -r..=r {
            for (dx, dz) in [(r, offset), (-r, offset), (offset, r), (offset, -r)] {
                let nx = x as i32 + dx;
                let nz = z as i32 + dz;

                if nx >= 0 && nx < width as i32 && nz >= 0 && nz < height as i32 {
                    let ni = nz as usize * width + nx as usize;
                    if depth_map[ni] >= 0.0 {
                        let dist = ((dx * dx + dz * dz) as f32).sqrt() * cell_size;
                        min_dist = min_dist.min(dist);
                    }
                }
            }
        }

        // Early exit if we found shoreline
        if min_dist < f32::MAX {
            break;
        }
    }

    min_dist
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

    const TOLERANCE: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < TOLERANCE
    }

    // ===== FoamConfig Tests =====

    #[test]
    fn test_foam_config_size() {
        assert_eq!(mem::size_of::<FoamConfig>(), FOAM_CONFIG_SIZE);
    }

    #[test]
    fn test_foam_config_default() {
        let config = FoamConfig::default();
        assert!(approx_eq(config.crest_threshold, DEFAULT_CREST_THRESHOLD));
        assert!(approx_eq(config.decay_rate, DEFAULT_DECAY_RATE));
        assert!(approx_eq(config.shore_width, DEFAULT_SHORE_WIDTH));
        assert!(approx_eq(config.shore_foam_max, DEFAULT_SHORE_FOAM_MAX));
        assert!(approx_eq(config.noise_scale, DEFAULT_NOISE_SCALE));
        assert!(approx_eq(config.noise_intensity, DEFAULT_NOISE_INTENSITY));
    }

    #[test]
    fn test_foam_config_presets() {
        let calm = FoamConfig::calm();
        let moderate = FoamConfig::moderate();
        let stormy = FoamConfig::stormy();

        // Calm has higher threshold (less foam)
        assert!(calm.crest_threshold > moderate.crest_threshold);
        // Stormy has lower threshold (more foam)
        assert!(stormy.crest_threshold < moderate.crest_threshold);

        // Stormy has slower decay (foam persists)
        assert!(stormy.decay_rate < calm.decay_rate);
    }

    #[test]
    fn test_foam_config_validate_valid() {
        assert!(FoamConfig::default().validate().is_ok());
        assert!(FoamConfig::calm().validate().is_ok());
        assert!(FoamConfig::stormy().validate().is_ok());
    }

    #[test]
    fn test_foam_config_validate_invalid_threshold() {
        let mut config = FoamConfig::default();
        config.crest_threshold = 0.0;
        assert!(config.validate().is_err());

        config.crest_threshold = -0.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_foam_config_validate_invalid_decay() {
        let mut config = FoamConfig::default();
        config.decay_rate = -1.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_foam_config_validate_invalid_shore_width() {
        let mut config = FoamConfig::default();
        config.shore_width = 0.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_foam_config_validate_invalid_shore_foam_max() {
        let mut config = FoamConfig::default();
        config.shore_foam_max = 1.5;
        assert!(config.validate().is_err());

        config.shore_foam_max = -0.1;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_foam_config_validate_invalid_noise() {
        let mut config = FoamConfig::default();
        config.noise_scale = 0.0;
        assert!(config.validate().is_err());

        config.noise_scale = 1.0;
        config.noise_intensity = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_foam_config_new_clamps() {
        let config = FoamConfig::new(-1.0, -1.0, -1.0, 2.0, -1.0, 2.0);
        assert!(config.crest_threshold > 0.0);
        assert!(config.decay_rate >= 0.0);
        assert!(config.shore_width > 0.0);
        assert!(config.shore_foam_max <= 1.0);
        assert!(config.noise_scale > 0.0);
        assert!(config.noise_intensity <= 1.0);
    }

    #[test]
    fn test_foam_config_bytemuck() {
        let config = FoamConfig::default();
        let bytes: &[u8] = bytemuck::bytes_of(&config);
        assert_eq!(bytes.len(), FOAM_CONFIG_SIZE);
    }

    // ===== FoamState Tests =====

    #[test]
    fn test_foam_state_size() {
        assert_eq!(mem::size_of::<FoamState>(), FOAM_STATE_SIZE);
    }

    #[test]
    fn test_foam_state_default() {
        let state = FoamState::default();
        assert!(approx_eq(state.foam_amount, 0.0));
        assert!(approx_eq(state.decay_timer, 0.0));
        assert!(approx_eq(state.jacobian, 0.0));
        assert!(approx_eq(state.shore_distance, 0.0));
    }

    #[test]
    fn test_foam_state_empty() {
        let state = FoamState::empty();
        assert!(approx_eq(state.foam_amount, 0.0));
        assert!(approx_eq(state.jacobian, 1.0));
        assert_eq!(state.shore_distance, f32::MAX);
    }

    #[test]
    fn test_foam_state_new_clamps() {
        let state = FoamState::new(2.0, -1.0, -0.5, -10.0);
        assert!(state.foam_amount <= 1.0);
        assert!(state.decay_timer >= 0.0);
        assert!(state.shore_distance >= 0.0);
    }

    #[test]
    fn test_foam_state_has_foam() {
        let state_with = FoamState::new(0.5, 0.0, 0.0, 0.0);
        let state_without = FoamState::new(0.0, 0.0, 0.0, 0.0);

        assert!(state_with.has_foam());
        assert!(!state_without.has_foam());
    }

    #[test]
    fn test_foam_state_is_folded() {
        let folded = FoamState::new(0.0, 0.0, -0.5, 0.0);
        let not_folded = FoamState::new(0.0, 0.0, 0.5, 0.0);

        assert!(folded.is_folded());
        assert!(!not_folded.is_folded());
    }

    #[test]
    fn test_foam_state_is_near_shore() {
        let near = FoamState::new(0.0, 0.0, 0.0, 3.0);
        let far = FoamState::new(0.0, 0.0, 0.0, 10.0);

        assert!(near.is_near_shore(5.0));
        assert!(!far.is_near_shore(5.0));
    }

    #[test]
    fn test_foam_state_bytemuck() {
        let state = FoamState::default();
        let bytes: &[u8] = bytemuck::bytes_of(&state);
        assert_eq!(bytes.len(), FOAM_STATE_SIZE);
    }

    // ===== Jacobian Tests =====

    #[test]
    fn test_jacobian_no_displacement() {
        // No displacement: J = (1+0)(1+0) - 0*0 = 1
        let j = FoamGenerator::compute_jacobian([0.0, 0.0], [0.0, 0.0]);
        assert!(approx_eq(j, 1.0));
    }

    #[test]
    fn test_jacobian_uniform_stretch() {
        // Uniform 2x stretch: J = (1+1)(1+1) - 0*0 = 4
        let j = FoamGenerator::compute_jacobian([1.0, 0.0], [0.0, 1.0]);
        assert!(approx_eq(j, 4.0));
    }

    #[test]
    fn test_jacobian_uniform_compression() {
        // Uniform 0.5x compression: J = (1-0.5)(1-0.5) - 0*0 = 0.25
        let j = FoamGenerator::compute_jacobian([-0.5, 0.0], [0.0, -0.5]);
        assert!(approx_eq(j, 0.25));
    }

    #[test]
    fn test_jacobian_negative_folded() {
        // Wave folding: J becomes negative when shear dominates
        // J = (1 + dx_du)(1 + dz_dv) - dx_dv * dz_du
        // For J < 0, we need dx_dv * dz_du > (1 + dx_du)(1 + dz_dv)
        // Example: dx_dv = 2, dz_du = 2 gives shear term = 4
        //          dx_du = 0, dz_dv = 0 gives main term = 1
        //          J = 1 - 4 = -3
        let j = FoamGenerator::compute_jacobian([0.0, 2.0], [2.0, 0.0]);
        assert!(j < 0.0, "Jacobian should be negative: {}", j);
    }

    #[test]
    fn test_jacobian_shear() {
        // Pure shear: J = 1*1 - 0.5*0.5 = 0.75
        let j = FoamGenerator::compute_jacobian([0.0, 0.5], [0.5, 0.0]);
        assert!(approx_eq(j, 0.75));
    }

    #[test]
    fn test_jacobian_from_matrix() {
        let matrix = [[0.5, 0.1], [0.2, 0.3]];
        let j1 = FoamGenerator::compute_jacobian([0.5, 0.1], [0.2, 0.3]);
        let j2 = FoamGenerator::compute_jacobian_from_matrix(matrix);
        assert!(approx_eq(j1, j2));
    }

    #[test]
    fn test_jacobian_critical_value() {
        // Exactly at zero: complete compression
        let j = FoamGenerator::compute_jacobian([-1.0, 0.0], [0.0, -1.0]);
        assert!(approx_eq(j, 0.0));
    }

    // ===== Crest Foam Tests =====

    #[test]
    fn test_crest_foam_above_threshold() {
        let gen = FoamGenerator::with_defaults();
        let foam = gen.crest_foam_from_jacobian(0.8);
        assert!(approx_eq(foam, 0.0));
    }

    #[test]
    fn test_crest_foam_at_threshold() {
        let gen = FoamGenerator::with_defaults();
        let foam = gen.crest_foam_from_jacobian(DEFAULT_CREST_THRESHOLD);
        assert!(approx_eq(foam, 0.0));
    }

    #[test]
    fn test_crest_foam_below_threshold() {
        let gen = FoamGenerator::with_defaults();
        let foam = gen.crest_foam_from_jacobian(0.35);
        // foam = 1 - 0.35/0.7 = 0.5
        assert!(approx_eq(foam, 0.5));
    }

    #[test]
    fn test_crest_foam_at_zero() {
        let gen = FoamGenerator::with_defaults();
        let foam = gen.crest_foam_from_jacobian(0.0);
        // foam = 1 - 0/0.7 = 1.0
        assert!(approx_eq(foam, 1.0));
    }

    #[test]
    fn test_crest_foam_negative_clamped() {
        let gen = FoamGenerator::with_defaults();
        let foam = gen.crest_foam_from_jacobian(-0.5);
        // foam = 1 - (-0.5)/0.7 > 1, clamped to 1
        assert!(approx_eq(foam, 1.0));
    }

    #[test]
    fn test_crest_foam_very_negative() {
        let gen = FoamGenerator::with_defaults();
        let foam = gen.crest_foam_from_jacobian(-10.0);
        assert!(approx_eq(foam, 1.0));
    }

    #[test]
    fn test_crest_foam_custom_threshold() {
        let config = FoamConfig::new(0.5, 2.0, 5.0, 1.0, 10.0, 0.3);
        let gen = FoamGenerator::new(config);

        let foam = gen.crest_foam_from_jacobian(0.25);
        // foam = 1 - 0.25/0.5 = 0.5
        assert!(approx_eq(foam, 0.5));
    }

    // ===== Decay Tests =====

    #[test]
    fn test_decay_reduces_foam() {
        let gen = FoamGenerator::with_defaults();
        let decayed = gen.update_decay(1.0, 0.1);
        // decayed = 1.0 - 2.0 * 0.1 = 0.8
        assert!(approx_eq(decayed, 0.8));
    }

    #[test]
    fn test_decay_clamps_to_zero() {
        let gen = FoamGenerator::with_defaults();
        let decayed = gen.update_decay(0.1, 1.0);
        // decayed = 0.1 - 2.0 * 1.0 = -1.9, clamped to 0
        assert!(approx_eq(decayed, 0.0));
    }

    #[test]
    fn test_decay_no_time() {
        let gen = FoamGenerator::with_defaults();
        let decayed = gen.update_decay(0.5, 0.0);
        assert!(approx_eq(decayed, 0.5));
    }

    #[test]
    fn test_decay_custom_rate() {
        let config = FoamConfig::new(0.7, 5.0, 5.0, 1.0, 10.0, 0.3);
        let gen = FoamGenerator::new(config);

        let decayed = gen.update_decay(1.0, 0.1);
        // decayed = 1.0 - 5.0 * 0.1 = 0.5
        assert!(approx_eq(decayed, 0.5));
    }

    #[test]
    fn test_decay_zero_rate() {
        let config = FoamConfig::new(0.7, 0.0, 5.0, 1.0, 10.0, 0.3);
        let gen = FoamGenerator::new(config);

        let decayed = gen.update_decay(0.8, 10.0);
        assert!(approx_eq(decayed, 0.8));
    }

    // ===== Shore Foam Tests =====

    #[test]
    fn test_shore_foam_far_from_shore() {
        let gen = FoamGenerator::with_defaults();
        let foam = gen.shore_foam(10.0, 1.0);
        assert!(approx_eq(foam, 0.0));
    }

    #[test]
    fn test_shore_foam_at_shore() {
        let gen = FoamGenerator::with_defaults();
        let foam = gen.shore_foam(0.0, 1.0);
        assert!(foam > 0.0);
    }

    #[test]
    fn test_shore_foam_at_shore_width() {
        let gen = FoamGenerator::with_defaults();
        let foam = gen.shore_foam(DEFAULT_SHORE_WIDTH, 1.0);
        assert!(approx_eq(foam, 0.0));
    }

    #[test]
    fn test_shore_foam_increases_near_shore() {
        let gen = FoamGenerator::with_defaults();
        let foam_far = gen.shore_foam(4.0, 1.0);
        let foam_near = gen.shore_foam(1.0, 1.0);

        assert!(foam_near > foam_far);
    }

    #[test]
    fn test_shore_foam_increases_with_wave_height() {
        let gen = FoamGenerator::with_defaults();
        let foam_low = gen.shore_foam(2.0, 0.5);
        let foam_high = gen.shore_foam(2.0, 2.5);

        assert!(foam_high > foam_low);
    }

    #[test]
    fn test_shore_foam_breaking_deep_water() {
        let gen = FoamGenerator::with_defaults();
        // Deep water: no breaking
        let foam = gen.shore_foam_breaking(2.0, 1.0, 10.0);
        assert!(foam < 0.5);
    }

    #[test]
    fn test_shore_foam_breaking_shallow_water() {
        let gen = FoamGenerator::with_defaults();
        // Shallow water: waves break
        let foam = gen.shore_foam_breaking(2.0, 1.0, 0.5);
        assert!(foam > 0.3);
    }

    #[test]
    fn test_shore_foam_custom_width() {
        let config = FoamConfig::new(0.7, 2.0, 10.0, 1.0, 10.0, 0.3);
        let gen = FoamGenerator::new(config);

        let foam = gen.shore_foam(8.0, 1.0);
        assert!(foam > 0.0); // Within wider band
    }

    // ===== Noise Tests =====

    #[test]
    fn test_noise_in_range() {
        let gen = FoamGenerator::with_defaults();

        for i in 0..100 {
            let u = i as f32 / 10.0;
            let v = i as f32 / 15.0;
            let noise = gen.sample_foam_noise([u, v], 0.0);
            assert!(noise >= 0.0 && noise <= 1.0);
        }
    }

    #[test]
    fn test_noise_varies_with_position() {
        let gen = FoamGenerator::with_defaults();

        let n1 = gen.sample_foam_noise([0.0, 0.0], 0.0);
        let n2 = gen.sample_foam_noise([0.5, 0.5], 0.0);
        let n3 = gen.sample_foam_noise([1.0, 1.0], 0.0);

        // At least one should be different
        assert!(n1 != n2 || n2 != n3);
    }

    #[test]
    fn test_noise_varies_with_time() {
        let gen = FoamGenerator::with_defaults();

        let n1 = gen.sample_foam_noise([0.5, 0.5], 0.0);
        let n2 = gen.sample_foam_noise([0.5, 0.5], 10.0);

        // Should be different at different times
        assert!(!approx_eq(n1, n2));
    }

    #[test]
    fn test_noise_turbulent_octaves() {
        let gen = FoamGenerator::with_defaults();

        let n1 = gen.sample_foam_noise_turbulent([0.5, 0.5], 0.0, 1);
        let n4 = gen.sample_foam_noise_turbulent([0.5, 0.5], 0.0, 4);

        // More octaves changes the result
        assert!(!approx_eq(n1, n4));
    }

    #[test]
    fn test_noise_turbulent_in_range() {
        let gen = FoamGenerator::with_defaults();

        for octaves in 1..=4 {
            let noise = gen.sample_foam_noise_turbulent([0.3, 0.7], 1.5, octaves);
            assert!(noise >= 0.0 && noise <= 1.0);
        }
    }

    // ===== Final Foam Tests =====

    #[test]
    fn test_final_foam_empty_state() {
        let gen = FoamGenerator::with_defaults();
        let state = FoamState::empty();
        let foam = gen.compute_final_foam(&state);
        assert!(approx_eq(foam, 0.0));
    }

    #[test]
    fn test_final_foam_from_crest() {
        let gen = FoamGenerator::with_defaults();
        let state = FoamState::new(0.0, 0.0, -0.5, f32::MAX);
        let foam = gen.compute_final_foam(&state);
        assert!(approx_eq(foam, 1.0));
    }

    #[test]
    fn test_final_foam_takes_max() {
        let gen = FoamGenerator::with_defaults();
        // State has existing foam, Jacobian would give less
        let state = FoamState::new(0.8, 0.0, 0.5, f32::MAX);
        let foam = gen.compute_final_foam(&state);
        assert!(approx_eq(foam, 0.8));
    }

    #[test]
    fn test_final_foam_with_noise_zero_foam() {
        let gen = FoamGenerator::with_defaults();
        let state = FoamState::empty();
        let foam = gen.compute_final_foam_with_noise(&state, [0.5, 0.5], 0.0);
        assert!(approx_eq(foam, 0.0));
    }

    #[test]
    fn test_final_foam_with_noise_modulates() {
        let gen = FoamGenerator::with_defaults();
        let state = FoamState::new(1.0, 0.0, -1.0, f32::MAX);

        let foam1 = gen.compute_final_foam_with_noise(&state, [0.0, 0.0], 0.0);
        let foam2 = gen.compute_final_foam_with_noise(&state, [0.5, 0.5], 0.0);

        // Both should be valid
        assert!(foam1 >= 0.0 && foam1 <= 1.0);
        assert!(foam2 >= 0.0 && foam2 <= 1.0);
    }

    // ===== State Update Tests =====

    #[test]
    fn test_update_state_new_crest_foam() {
        let gen = FoamGenerator::with_defaults();
        let mut state = FoamState::empty();

        gen.update_state(&mut state, -0.5, f32::MAX, 0.0, 0.016);

        assert!(state.has_foam());
        assert!(approx_eq(state.jacobian, -0.5));
    }

    #[test]
    fn test_update_state_decays() {
        let gen = FoamGenerator::with_defaults();
        let mut state = FoamState::new(1.0, 0.0, 1.0, f32::MAX);

        // No new foam, existing should decay
        gen.update_state(&mut state, 1.0, f32::MAX, 0.0, 0.5);

        assert!(state.foam_amount < 1.0);
    }

    #[test]
    fn test_update_state_shore_foam() {
        let gen = FoamGenerator::with_defaults();
        let mut state = FoamState::empty();

        gen.update_state(&mut state, 1.0, 1.0, 2.0, 0.016);

        assert!(state.has_foam());
        assert!(approx_eq(state.shore_distance, 1.0));
    }

    #[test]
    fn test_update_state_timer_resets() {
        let gen = FoamGenerator::with_defaults();
        let mut state = FoamState::new(0.5, 1.0, 1.0, f32::MAX);

        // New crest foam should reset timer
        gen.update_state(&mut state, -0.5, f32::MAX, 0.0, 0.016);

        assert!(approx_eq(state.decay_timer, 0.0));
    }

    #[test]
    fn test_update_states_batch() {
        let gen = FoamGenerator::with_defaults();
        let mut states = vec![FoamState::empty(); 4];
        let jacobians = [-0.5, 0.5, 0.0, 1.0];
        let shore_dists = [1.0, 10.0, 2.0, 20.0];
        let wave_heights = [1.0, 1.0, 1.0, 1.0];

        gen.update_states_batch(&mut states, &jacobians, &shore_dists, &wave_heights, 0.016);

        // First should have foam (negative Jacobian)
        assert!(states[0].has_foam());
        // Second should have less (high Jacobian, far from shore)
        assert!(states[1].foam_amount < states[0].foam_amount);
    }

    // ===== Coverage/Stats Tests =====

    #[test]
    fn test_estimate_coverage_empty() {
        let gen = FoamGenerator::with_defaults();
        let coverage = gen.estimate_coverage(&[], 0.1);
        assert!(approx_eq(coverage, 0.0));
    }

    #[test]
    fn test_estimate_coverage_all_foam() {
        let gen = FoamGenerator::with_defaults();
        let states = vec![FoamState::new(1.0, 0.0, 0.0, 0.0); 10];
        let coverage = gen.estimate_coverage(&states, 0.1);
        assert!(approx_eq(coverage, 1.0));
    }

    #[test]
    fn test_estimate_coverage_half() {
        let gen = FoamGenerator::with_defaults();
        let mut states = vec![FoamState::empty(); 10];
        for i in 0..5 {
            states[i].foam_amount = 0.5;
        }
        let coverage = gen.estimate_coverage(&states, 0.1);
        assert!(approx_eq(coverage, 0.5));
    }

    #[test]
    fn test_average_foam_empty() {
        let gen = FoamGenerator::with_defaults();
        let avg = gen.average_foam(&[]);
        assert!(approx_eq(avg, 0.0));
    }

    #[test]
    fn test_average_foam() {
        let gen = FoamGenerator::with_defaults();
        let states = vec![
            FoamState::new(0.2, 0.0, 0.0, 0.0),
            FoamState::new(0.4, 0.0, 0.0, 0.0),
            FoamState::new(0.6, 0.0, 0.0, 0.0),
            FoamState::new(0.8, 0.0, 0.0, 0.0),
        ];
        let avg = gen.average_foam(&states);
        assert!(approx_eq(avg, 0.5));
    }

    // ===== FoamMask Tests =====

    #[test]
    fn test_foam_mask_new() {
        let mask = FoamMask::new(64, 64);
        assert_eq!(mask.width, 64);
        assert_eq!(mask.height, 64);
        assert_eq!(mask.data.len(), 64 * 64);
    }

    #[test]
    fn test_foam_mask_clear() {
        let mut mask = FoamMask::new(4, 4);
        mask.set(0, 0, 1.0);
        mask.clear();
        assert!(approx_eq(mask.get(0, 0), 0.0));
    }

    #[test]
    fn test_foam_mask_set_get() {
        let mut mask = FoamMask::new(4, 4);
        mask.set(1, 2, 0.5);
        let value = mask.get(1, 2);
        // Allow for quantization error (255 levels)
        assert!((value - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_foam_mask_clamps_value() {
        let mut mask = FoamMask::new(4, 4);
        mask.set(0, 0, 2.0);
        mask.set(1, 1, -1.0);

        assert!(mask.get(0, 0) <= 1.0);
        assert!(mask.get(1, 1) >= 0.0);
    }

    #[test]
    fn test_foam_mask_bounds() {
        let mut mask = FoamMask::new(4, 4);
        mask.set(10, 10, 1.0); // Out of bounds, should not crash
        let value = mask.get(10, 10);
        assert!(approx_eq(value, 0.0));
    }

    #[test]
    fn test_foam_mask_bilinear_corners() {
        let mut mask = FoamMask::new(2, 2);
        mask.set(0, 0, 1.0);
        mask.set(1, 0, 0.0);
        mask.set(0, 1, 0.0);
        mask.set(1, 1, 0.0);

        let v00 = mask.sample_bilinear(0.0, 0.0);
        let v11 = mask.sample_bilinear(1.0, 1.0);

        assert!((v00 - 1.0).abs() < 0.01);
        assert!((v11 - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_foam_mask_bilinear_center() {
        let mut mask = FoamMask::new(2, 2);
        mask.set(0, 0, 1.0);
        mask.set(1, 0, 1.0);
        mask.set(0, 1, 0.0);
        mask.set(1, 1, 0.0);

        let v_center = mask.sample_bilinear(0.5, 0.5);
        // Should be interpolated to ~0.5
        assert!((v_center - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_foam_mask_fill_from_states() {
        let mut mask = FoamMask::new(2, 2);
        let states = vec![
            FoamState::new(0.0, 0.0, 0.0, 0.0),
            FoamState::new(0.5, 0.0, 0.0, 0.0),
            FoamState::new(0.5, 0.0, 0.0, 0.0),
            FoamState::new(1.0, 0.0, 0.0, 0.0),
        ];

        mask.fill_from_states(&states);

        assert!((mask.get(0, 0) - 0.0).abs() < 0.01);
        assert!((mask.get(1, 1) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_foam_mask_as_bytes() {
        let mask = FoamMask::new(4, 4);
        let bytes = mask.as_bytes();
        assert_eq!(bytes.len(), 16);
    }

    #[test]
    fn test_foam_mask_pixel_count() {
        let mask = FoamMask::new(8, 16);
        assert_eq!(mask.pixel_count(), 128);
    }

    // ===== Grid Helper Tests =====

    #[test]
    fn test_jacobian_from_grid_no_displacement() {
        let displacements = vec![[0.0, 0.0]; 9];
        let j = jacobian_from_grid(&displacements, 1.0, 1, 1, 3, 3);
        assert!(approx_eq(j, 1.0));
    }

    #[test]
    fn test_jacobian_from_grid_uniform_displacement() {
        // Constant displacement = no gradient = J = 1
        let displacements = vec![[1.0, 1.0]; 9];
        let j = jacobian_from_grid(&displacements, 1.0, 1, 1, 3, 3);
        assert!(approx_eq(j, 1.0));
    }

    #[test]
    fn test_jacobian_from_grid_linear_gradient() {
        // Linear gradient in X: displacement increases by 1 per cell
        let mut displacements = vec![[0.0, 0.0]; 9];
        for z in 0..3 {
            for x in 0..3 {
                displacements[z * 3 + x] = [x as f32, 0.0];
            }
        }

        let j = jacobian_from_grid(&displacements, 1.0, 1, 1, 3, 3);
        // At center (1,1), using central differences:
        // dx/du = (disp[2] - disp[0]) / (2 * spacing) = (2 - 0) / 2 = 1
        // dz/dv = 0 (no z displacement)
        // J = (1 + 1) * (1 + 0) - 0 * 0 = 2
        assert!((j - 2.0).abs() < 0.1, "Jacobian should be ~2.0, got {}", j);
    }

    #[test]
    fn test_jacobian_from_grid_corner() {
        let displacements = vec![[0.0, 0.0]; 9];
        // Corner uses one-sided differences
        let j = jacobian_from_grid(&displacements, 1.0, 0, 0, 3, 3);
        assert!(approx_eq(j, 1.0));
    }

    // ===== Shore Distance Tests =====

    #[test]
    fn test_shore_distance_on_land() {
        let depth_map = vec![1.0; 9]; // All land
        let dist = shore_distance_from_depth(&depth_map, 1, 1, 3, 3, 1.0);
        assert!(approx_eq(dist, 0.0));
    }

    #[test]
    fn test_shore_distance_deep_water() {
        let depth_map = vec![-10.0; 9]; // All deep water
        let dist = shore_distance_from_depth(&depth_map, 1, 1, 3, 3, 1.0);
        assert_eq!(dist, f32::MAX); // No shoreline found
    }

    #[test]
    fn test_shore_distance_near_shore() {
        let mut depth_map = vec![-1.0; 9];
        depth_map[0] = 1.0; // Land at (0,0)

        let dist = shore_distance_from_depth(&depth_map, 1, 1, 3, 3, 1.0);
        // Distance from (1,1) to (0,0) = sqrt(2) ~= 1.414
        assert!((dist - 2.0_f32.sqrt()).abs() < 0.1);
    }

    // ===== Generator Default Tests =====

    #[test]
    fn test_generator_default() {
        let gen = FoamGenerator::default();
        assert!(gen.config.validate().is_ok());
    }

    #[test]
    fn test_generator_with_defaults() {
        let gen = FoamGenerator::with_defaults();
        assert!(approx_eq(gen.config.crest_threshold, DEFAULT_CREST_THRESHOLD));
    }
}
