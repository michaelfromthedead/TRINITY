//! Advection-Based Foam Transport Simulation for TRINITY Engine (T-ENV-3.12).
//!
//! Extends the basic foam system (T-ENV-2.9) with physically accurate transport:
//! - Semi-Lagrangian advection for foam movement
//! - Gaussian diffusion for foam spreading
//! - Exponential decay for foam lifetime
//! - Turbulence-driven spawn from wave Jacobian
//! - Obstacle collision foam generation
//! - Particle-based bubble detail layer
//!
//! # Overview
//!
//! Foam on ocean surfaces doesn't just appear and disappear - it moves with
//! the water surface, spreads out over time, and decays. This module simulates
//! these transport phenomena using a 2D scalar field (concentration) advected
//! by a velocity field derived from wave motion.
//!
//! # Semi-Lagrangian Advection
//!
//! Instead of moving foam forward in time (which can be unstable), we trace
//! backward from each cell to find where the foam came from:
//!
//! ```text
//! new_value[x,y] = old_value[x - vx*dt, y - vy*dt]
//! ```
//!
//! This is unconditionally stable for any time step.
//!
//! # Usage
//!
//! ```ignore
//! use renderer_backend::water::foam_advection::{
//!     FoamAdvectionConfig, FoamField, VelocityField, FoamAdvector,
//! };
//!
//! let config = FoamAdvectionConfig::default();
//! let mut foam = FoamField::new(512, 512);
//! let mut velocity = VelocityField::new(512, 512);
//!
//! // Derive velocity from wave gradients
//! velocity.from_wave_gradient(&gerstner_results, &wave_params);
//!
//! // Spawn foam from wave turbulence
//! let advector = FoamAdvector::new(config);
//! advector.spawn_from_turbulence(&mut foam, &jacobian_field);
//!
//! // Simulate one frame
//! advector.step(&mut foam, &velocity, dt);
//!
//! // Sample for shading
//! let foam_amount = foam.sample_bilinear(u, v);
//! ```

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default grid resolution for foam field.
pub const DEFAULT_GRID_RESOLUTION: u32 = 512;

/// Default cell size in meters.
pub const DEFAULT_CELL_SIZE: f32 = 0.5;

/// Default advection speed multiplier.
pub const DEFAULT_ADVECTION_SPEED: f32 = 1.0;

/// Default diffusion rate per second.
pub const DEFAULT_DIFFUSION_RATE: f32 = 0.01;

/// Default foam spawn rate per second.
pub const DEFAULT_SPAWN_RATE: f32 = 1.0;

/// Default foam decay rate per second.
pub const DEFAULT_DECAY_RATE: f32 = 0.5;

/// Default maximum foam concentration.
pub const DEFAULT_MAX_CONCENTRATION: f32 = 1.0;

/// FoamAdvectionConfig struct size in bytes (must be 16-byte aligned).
pub const FOAM_ADVECTION_CONFIG_SIZE: usize = 32;

/// Default bubble lifetime in seconds.
pub const DEFAULT_BUBBLE_LIFETIME: f32 = 2.0;

/// Maximum number of active bubbles.
pub const MAX_BUBBLES: usize = 10000;

/// Gravity for bubble physics (m/s^2).
pub const BUBBLE_GRAVITY: f32 = 9.81;

/// Water drag coefficient for bubbles.
pub const BUBBLE_DRAG: f32 = 0.5;

/// Small epsilon for floating point comparisons.
const EPSILON: f32 = 1e-6;

// ---------------------------------------------------------------------------
// FoamAdvectionConfig
// ---------------------------------------------------------------------------

/// Configuration for advection-based foam simulation.
///
/// GPU-compatible layout for uniform buffers.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FoamAdvectionConfig {
    /// Grid resolution (cells per side).
    pub grid_resolution: u32,

    /// Cell size in world units (meters).
    pub cell_size: f32,

    /// Advection speed multiplier (default: 1.0).
    pub advection_speed: f32,

    /// Diffusion rate per second (default: 0.01).
    pub diffusion_rate: f32,

    /// Foam spawn rate per second (default: 1.0).
    pub spawn_rate: f32,

    /// Foam decay rate per second (default: 0.5).
    pub decay_rate: f32,

    /// Maximum foam concentration (default: 1.0).
    pub max_concentration: f32,

    /// Padding for 16-byte alignment.
    pub _padding: f32,
}

impl Default for FoamAdvectionConfig {
    fn default() -> Self {
        Self {
            grid_resolution: DEFAULT_GRID_RESOLUTION,
            cell_size: DEFAULT_CELL_SIZE,
            advection_speed: DEFAULT_ADVECTION_SPEED,
            diffusion_rate: DEFAULT_DIFFUSION_RATE,
            spawn_rate: DEFAULT_SPAWN_RATE,
            decay_rate: DEFAULT_DECAY_RATE,
            max_concentration: DEFAULT_MAX_CONCENTRATION,
            _padding: 0.0,
        }
    }
}

impl FoamAdvectionConfig {
    /// Create a new configuration with custom parameters.
    pub fn new(
        grid_resolution: u32,
        cell_size: f32,
        advection_speed: f32,
        diffusion_rate: f32,
        spawn_rate: f32,
        decay_rate: f32,
        max_concentration: f32,
    ) -> Self {
        Self {
            grid_resolution: grid_resolution.max(2),
            cell_size: cell_size.max(EPSILON),
            advection_speed: advection_speed.max(0.0),
            diffusion_rate: diffusion_rate.max(0.0),
            spawn_rate: spawn_rate.max(0.0),
            decay_rate: decay_rate.max(0.0),
            max_concentration: max_concentration.max(EPSILON),
            _padding: 0.0,
        }
    }

    /// Create configuration for high-quality simulation.
    pub fn high_quality() -> Self {
        Self {
            grid_resolution: 1024,
            cell_size: 0.25,
            advection_speed: 1.0,
            diffusion_rate: 0.005,
            spawn_rate: 1.5,
            decay_rate: 0.3,
            max_concentration: 1.0,
            _padding: 0.0,
        }
    }

    /// Create configuration for performance-oriented simulation.
    pub fn performance() -> Self {
        Self {
            grid_resolution: 256,
            cell_size: 1.0,
            advection_speed: 1.0,
            diffusion_rate: 0.02,
            spawn_rate: 0.8,
            decay_rate: 0.8,
            max_concentration: 1.0,
            _padding: 0.0,
        }
    }

    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.grid_resolution < 2 {
            return Err("Grid resolution must be at least 2");
        }
        if self.cell_size <= 0.0 {
            return Err("Cell size must be positive");
        }
        if self.advection_speed < 0.0 {
            return Err("Advection speed must be non-negative");
        }
        if self.diffusion_rate < 0.0 {
            return Err("Diffusion rate must be non-negative");
        }
        if self.spawn_rate < 0.0 {
            return Err("Spawn rate must be non-negative");
        }
        if self.decay_rate < 0.0 {
            return Err("Decay rate must be non-negative");
        }
        if self.max_concentration <= 0.0 {
            return Err("Max concentration must be positive");
        }
        Ok(())
    }

    /// Get the total world size covered by the grid.
    #[inline]
    pub fn world_size(&self) -> f32 {
        self.grid_resolution as f32 * self.cell_size
    }
}

// ---------------------------------------------------------------------------
// FoamField
// ---------------------------------------------------------------------------

/// 2D scalar field storing foam concentration values.
///
/// Each cell contains a concentration value from 0.0 (no foam) to
/// max_concentration (fully saturated). The field supports bilinear
/// sampling for smooth interpolation during advection.
#[derive(Clone, Debug)]
pub struct FoamField {
    /// Width of the field in cells.
    pub width: u32,

    /// Height of the field in cells.
    pub height: u32,

    /// Concentration values (row-major order).
    data: Vec<f32>,
}

impl FoamField {
    /// Create a new foam field initialized to zero.
    pub fn new(width: u32, height: u32) -> Self {
        let size = (width * height) as usize;
        Self {
            width,
            height,
            data: vec![0.0; size],
        }
    }

    /// Create a foam field from existing data.
    pub fn from_data(width: u32, height: u32, data: Vec<f32>) -> Option<Self> {
        if data.len() != (width * height) as usize {
            return None;
        }
        Some(Self { width, height, data })
    }

    /// Get foam concentration at integer coordinates.
    ///
    /// Returns 0.0 for out-of-bounds coordinates.
    #[inline]
    pub fn get(&self, x: u32, y: u32) -> f32 {
        if x < self.width && y < self.height {
            self.data[(y * self.width + x) as usize]
        } else {
            0.0
        }
    }

    /// Get foam concentration at signed integer coordinates.
    ///
    /// Returns 0.0 for out-of-bounds coordinates.
    #[inline]
    pub fn get_i32(&self, x: i32, y: i32) -> f32 {
        if x >= 0 && y >= 0 && x < self.width as i32 && y < self.height as i32 {
            self.data[(y as u32 * self.width + x as u32) as usize]
        } else {
            0.0
        }
    }

    /// Set foam concentration at integer coordinates.
    #[inline]
    pub fn set(&mut self, x: u32, y: u32, value: f32) {
        if x < self.width && y < self.height {
            self.data[(y * self.width + x) as usize] = value;
        }
    }

    /// Add foam to a cell (accumulates, does not replace).
    #[inline]
    pub fn add(&mut self, x: u32, y: u32, amount: f32) {
        if x < self.width && y < self.height {
            let idx = (y * self.width + x) as usize;
            self.data[idx] += amount;
        }
    }

    /// Sample with bilinear interpolation.
    ///
    /// Coordinates are in normalized [0, 1] range.
    pub fn sample_bilinear(&self, u: f32, v: f32) -> f32 {
        // Clamp to valid range
        let u = u.clamp(0.0, 1.0);
        let v = v.clamp(0.0, 1.0);

        // Convert to cell coordinates
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

        // Bilinear interpolation
        let v0 = v00 * (1.0 - fx) + v10 * fx;
        let v1 = v01 * (1.0 - fx) + v11 * fx;

        v0 * (1.0 - fy) + v1 * fy
    }

    /// Sample with bilinear interpolation at cell coordinates.
    ///
    /// Coordinates are in cell space (0 to width/height).
    pub fn sample_bilinear_cell(&self, x: f32, y: f32) -> f32 {
        // Clamp to valid range
        let x = x.clamp(0.0, (self.width - 1) as f32);
        let y = y.clamp(0.0, (self.height - 1) as f32);

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

    /// Clear the entire field to zero.
    pub fn clear(&mut self) {
        self.data.fill(0.0);
    }

    /// Fill the entire field with a value.
    pub fn fill(&mut self, value: f32) {
        self.data.fill(value);
    }

    /// Clamp all values to a maximum.
    pub fn clamp_max(&mut self, max_value: f32) {
        for v in &mut self.data {
            *v = v.min(max_value);
        }
    }

    /// Clamp all values to [0, max].
    pub fn clamp_range(&mut self, max_value: f32) {
        for v in &mut self.data {
            *v = v.clamp(0.0, max_value);
        }
    }

    /// Get the total foam mass (sum of all concentrations).
    pub fn total_mass(&self) -> f32 {
        self.data.iter().sum()
    }

    /// Get the average foam concentration.
    pub fn average(&self) -> f32 {
        if self.data.is_empty() {
            return 0.0;
        }
        self.total_mass() / self.data.len() as f32
    }

    /// Get the maximum foam concentration.
    pub fn max_value(&self) -> f32 {
        self.data.iter().cloned().fold(0.0, f32::max)
    }

    /// Get raw data slice.
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Get mutable raw data slice.
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Get pixel count.
    #[inline]
    pub fn cell_count(&self) -> usize {
        (self.width * self.height) as usize
    }

    /// Copy data from another field of the same size.
    pub fn copy_from(&mut self, other: &FoamField) {
        debug_assert_eq!(self.width, other.width);
        debug_assert_eq!(self.height, other.height);
        self.data.copy_from_slice(&other.data);
    }

    /// Swap data with another field.
    pub fn swap(&mut self, other: &mut FoamField) {
        debug_assert_eq!(self.width, other.width);
        debug_assert_eq!(self.height, other.height);
        std::mem::swap(&mut self.data, &mut other.data);
    }
}

// ---------------------------------------------------------------------------
// VelocityField
// ---------------------------------------------------------------------------

/// 2D vector field storing velocity values for advection.
///
/// Each cell contains a 2D velocity vector [vx, vy] representing
/// the flow direction and speed at that point.
#[derive(Clone, Debug)]
pub struct VelocityField {
    /// Width of the field in cells.
    pub width: u32,

    /// Height of the field in cells.
    pub height: u32,

    /// Velocity vectors (row-major order, [vx, vy] per cell).
    data: Vec<[f32; 2]>,
}

impl VelocityField {
    /// Create a new velocity field initialized to zero.
    pub fn new(width: u32, height: u32) -> Self {
        let size = (width * height) as usize;
        Self {
            width,
            height,
            data: vec![[0.0, 0.0]; size],
        }
    }

    /// Get velocity at integer coordinates.
    ///
    /// Returns [0, 0] for out-of-bounds coordinates.
    #[inline]
    pub fn get(&self, x: u32, y: u32) -> [f32; 2] {
        if x < self.width && y < self.height {
            self.data[(y * self.width + x) as usize]
        } else {
            [0.0, 0.0]
        }
    }

    /// Get velocity at signed integer coordinates.
    #[inline]
    pub fn get_i32(&self, x: i32, y: i32) -> [f32; 2] {
        if x >= 0 && y >= 0 && x < self.width as i32 && y < self.height as i32 {
            self.data[(y as u32 * self.width + x as u32) as usize]
        } else {
            [0.0, 0.0]
        }
    }

    /// Set velocity at integer coordinates.
    #[inline]
    pub fn set(&mut self, x: u32, y: u32, vx: f32, vy: f32) {
        if x < self.width && y < self.height {
            self.data[(y * self.width + x) as usize] = [vx, vy];
        }
    }

    /// Sample with bilinear interpolation.
    ///
    /// Coordinates are in normalized [0, 1] range.
    pub fn sample_bilinear(&self, u: f32, v: f32) -> [f32; 2] {
        let u = u.clamp(0.0, 1.0);
        let v = v.clamp(0.0, 1.0);

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

        // Bilinear interpolation for each component
        let vx0 = v00[0] * (1.0 - fx) + v10[0] * fx;
        let vx1 = v01[0] * (1.0 - fx) + v11[0] * fx;
        let vy0 = v00[1] * (1.0 - fx) + v10[1] * fx;
        let vy1 = v01[1] * (1.0 - fx) + v11[1] * fx;

        [
            vx0 * (1.0 - fy) + vx1 * fy,
            vy0 * (1.0 - fy) + vy1 * fy,
        ]
    }

    /// Sample with bilinear interpolation at cell coordinates.
    pub fn sample_bilinear_cell(&self, x: f32, y: f32) -> [f32; 2] {
        let x = x.clamp(0.0, (self.width - 1) as f32);
        let y = y.clamp(0.0, (self.height - 1) as f32);

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

        let vx0 = v00[0] * (1.0 - fx) + v10[0] * fx;
        let vx1 = v01[0] * (1.0 - fx) + v11[0] * fx;
        let vy0 = v00[1] * (1.0 - fx) + v10[1] * fx;
        let vy1 = v01[1] * (1.0 - fx) + v11[1] * fx;

        [
            vx0 * (1.0 - fy) + vx1 * fy,
            vy0 * (1.0 - fy) + vy1 * fy,
        ]
    }

    /// Clear the entire field to zero velocity.
    pub fn clear(&mut self) {
        self.data.fill([0.0, 0.0]);
    }

    /// Set uniform velocity across the entire field.
    pub fn set_uniform(&mut self, vx: f32, vy: f32) {
        self.data.fill([vx, vy]);
    }

    /// Derive velocity from wave displacement gradients.
    ///
    /// The horizontal velocity of water particles in Gerstner waves
    /// is proportional to the time derivative of horizontal displacement.
    /// For simulation purposes, we approximate this as the spatial gradient
    /// of the height field times wave speed.
    ///
    /// # Arguments
    ///
    /// * `height_field` - Wave heights at each cell
    /// * `wave_speed` - Characteristic wave speed
    /// * `cell_size` - Size of each cell in world units
    pub fn from_wave_gradient(
        &mut self,
        height_field: &[f32],
        wave_speed: f32,
        cell_size: f32,
    ) {
        debug_assert_eq!(height_field.len(), self.cell_count());

        let inv_cell_size = 1.0 / cell_size;

        for y in 0..self.height {
            for x in 0..self.width {
                let idx = (y * self.width + x) as usize;

                // Central differences for gradient
                let x_prev = if x > 0 { x - 1 } else { x };
                let x_next = if x + 1 < self.width { x + 1 } else { x };
                let y_prev = if y > 0 { y - 1 } else { y };
                let y_next = if y + 1 < self.height { y + 1 } else { y };

                let h_x0 = height_field[(y * self.width + x_prev) as usize];
                let h_x1 = height_field[(y * self.width + x_next) as usize];
                let h_y0 = height_field[(y_prev * self.width + x) as usize];
                let h_y1 = height_field[(y_next * self.width + x) as usize];

                // Gradient scale depends on whether we used central or one-sided
                let dx = if x > 0 && x + 1 < self.width { 2.0 } else { 1.0 };
                let dy = if y > 0 && y + 1 < self.height { 2.0 } else { 1.0 };

                // Velocity from gradient (downhill flow)
                let dh_dx = (h_x1 - h_x0) / (dx * cell_size);
                let dh_dy = (h_y1 - h_y0) / (dy * cell_size);

                // Water flows "downhill" on wave surface, perpendicular to gradient
                // For realistic foam motion, we use gradient direction
                self.data[idx] = [
                    -dh_dx * wave_speed * inv_cell_size,
                    -dh_dy * wave_speed * inv_cell_size,
                ];
            }
        }
    }

    /// Get average velocity magnitude.
    pub fn average_speed(&self) -> f32 {
        if self.data.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.data.iter()
            .map(|v| (v[0] * v[0] + v[1] * v[1]).sqrt())
            .sum();
        sum / self.data.len() as f32
    }

    /// Get maximum velocity magnitude.
    pub fn max_speed(&self) -> f32 {
        self.data.iter()
            .map(|v| (v[0] * v[0] + v[1] * v[1]).sqrt())
            .fold(0.0, f32::max)
    }

    /// Get cell count.
    #[inline]
    pub fn cell_count(&self) -> usize {
        (self.width * self.height) as usize
    }

    /// Get raw data slice.
    pub fn data(&self) -> &[[f32; 2]] {
        &self.data
    }

    /// Get mutable raw data slice.
    pub fn data_mut(&mut self) -> &mut [[f32; 2]] {
        &mut self.data
    }
}

// ---------------------------------------------------------------------------
// FoamAdvector
// ---------------------------------------------------------------------------

/// Foam advection simulator.
///
/// Handles the transport of foam concentration using semi-Lagrangian
/// advection, diffusion, decay, and spawn from various sources.
#[derive(Clone, Debug)]
pub struct FoamAdvector {
    /// Configuration parameters.
    pub config: FoamAdvectionConfig,

    /// Scratch buffer for double-buffering during advection.
    scratch: Vec<f32>,
}

impl FoamAdvector {
    /// Create a new foam advector with the given configuration.
    pub fn new(config: FoamAdvectionConfig) -> Self {
        let size = (config.grid_resolution * config.grid_resolution) as usize;
        Self {
            config,
            scratch: vec![0.0; size],
        }
    }

    /// Create a foam advector with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(FoamAdvectionConfig::default())
    }

    /// Resize scratch buffer if needed.
    fn ensure_scratch_size(&mut self, size: usize) {
        if self.scratch.len() < size {
            self.scratch.resize(size, 0.0);
        }
    }

    /// Perform semi-Lagrangian advection.
    ///
    /// Traces particles backward in time to find where the foam
    /// at each cell came from, providing unconditional stability.
    ///
    /// # Arguments
    ///
    /// * `field` - Foam field to advect (modified in place)
    /// * `velocity` - Velocity field for transport
    /// * `dt` - Time step in seconds
    pub fn advect(&mut self, field: &mut FoamField, velocity: &VelocityField, dt: f32) {
        let width = field.width;
        let height = field.height;
        let size = field.cell_count();

        self.ensure_scratch_size(size);

        let speed = self.config.advection_speed;

        // Semi-Lagrangian: trace backward from each cell
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;

                // Get velocity at this cell
                let vel = velocity.get(x, y);

                // Trace back in time (in cell coordinates)
                let src_x = x as f32 - vel[0] * speed * dt;
                let src_y = y as f32 - vel[1] * speed * dt;

                // Sample the source value with bilinear interpolation
                self.scratch[idx] = field.sample_bilinear_cell(src_x, src_y);
            }
        }

        // Copy scratch back to field
        field.data_mut().copy_from_slice(&self.scratch[..size]);
    }

    /// Apply Gaussian diffusion to spread foam.
    ///
    /// Uses a simple 3x3 Gaussian kernel for local spreading.
    ///
    /// # Arguments
    ///
    /// * `field` - Foam field to diffuse (modified in place)
    /// * `dt` - Time step in seconds
    pub fn diffuse(&mut self, field: &mut FoamField, dt: f32) {
        let width = field.width;
        let height = field.height;
        let size = field.cell_count();

        self.ensure_scratch_size(size);

        // Diffusion amount this frame
        let diffusion = self.config.diffusion_rate * dt;

        // How much to spread to neighbors
        let spread = diffusion.min(0.25); // Clamp for stability

        // Weights: center keeps (1 - 4*spread), each neighbor gets spread/4
        let center_weight = 1.0 - 4.0 * spread;
        let neighbor_weight = spread;

        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;

                let center = field.get(x, y);
                let left = field.get_i32(x as i32 - 1, y as i32);
                let right = field.get_i32(x as i32 + 1, y as i32);
                let up = field.get_i32(x as i32, y as i32 - 1);
                let down = field.get_i32(x as i32, y as i32 + 1);

                self.scratch[idx] = center * center_weight
                    + (left + right + up + down) * neighbor_weight;
            }
        }

        field.data_mut().copy_from_slice(&self.scratch[..size]);
    }

    /// Apply exponential decay to foam.
    ///
    /// # Arguments
    ///
    /// * `field` - Foam field to decay (modified in place)
    /// * `dt` - Time step in seconds
    pub fn decay(&mut self, field: &mut FoamField, dt: f32) {
        let decay_factor = (-self.config.decay_rate * dt).exp();

        for v in field.data_mut() {
            *v *= decay_factor;
        }
    }

    /// Spawn foam from wave turbulence (Jacobian field).
    ///
    /// Foam is generated where waves are folding (Jacobian < 0) or
    /// heavily compressed (Jacobian < threshold).
    ///
    /// # Arguments
    ///
    /// * `field` - Foam field to spawn into (modified in place)
    /// * `jacobian_field` - Jacobian determinant values
    pub fn spawn_from_turbulence(&self, field: &mut FoamField, jacobian_field: &[f32]) {
        debug_assert_eq!(field.cell_count(), jacobian_field.len());

        let spawn_threshold = 0.7; // Same as foam.rs DEFAULT_CREST_THRESHOLD

        for (i, &jacobian) in jacobian_field.iter().enumerate() {
            if jacobian < spawn_threshold {
                // Foam intensity: more foam for more negative Jacobian
                let foam = (1.0 - jacobian / spawn_threshold).clamp(0.0, 1.0);
                let spawn_amount = foam * self.config.spawn_rate;

                field.data_mut()[i] += spawn_amount;
            }
        }

        // Clamp to max concentration
        field.clamp_max(self.config.max_concentration);
    }

    /// Spawn foam from collision with obstacles.
    ///
    /// Foam is generated where the water surface intersects obstacles
    /// (negative SDF values near zero crossing).
    ///
    /// # Arguments
    ///
    /// * `field` - Foam field to spawn into (modified in place)
    /// * `obstacle_sdf` - Signed distance field of obstacles
    pub fn spawn_from_collision(&self, field: &mut FoamField, obstacle_sdf: &[f32]) {
        debug_assert_eq!(field.cell_count(), obstacle_sdf.len());

        let collision_threshold = 0.5; // Distance threshold for collision

        for (i, &sdf) in obstacle_sdf.iter().enumerate() {
            // Foam at collision points (SDF near zero, underwater)
            if sdf < 0.0 && sdf > -collision_threshold {
                // Stronger foam closer to surface
                let proximity = 1.0 - (-sdf / collision_threshold);
                let spawn_amount = proximity * self.config.spawn_rate;

                field.data_mut()[i] += spawn_amount;
            }
        }

        field.clamp_max(self.config.max_concentration);
    }

    /// Perform a full simulation step.
    ///
    /// Executes advection, diffusion, and decay in order.
    ///
    /// # Arguments
    ///
    /// * `field` - Foam field to simulate (modified in place)
    /// * `velocity` - Velocity field for advection
    /// * `dt` - Time step in seconds
    pub fn step(&mut self, field: &mut FoamField, velocity: &VelocityField, dt: f32) {
        self.advect(field, velocity, dt);
        self.diffuse(field, dt);
        self.decay(field, dt);

        // Ensure valid range
        field.clamp_range(self.config.max_concentration);
    }

    /// Perform a full step with turbulence spawn.
    ///
    /// # Arguments
    ///
    /// * `field` - Foam field to simulate
    /// * `velocity` - Velocity field for advection
    /// * `jacobian_field` - Jacobian values for spawn
    /// * `dt` - Time step in seconds
    pub fn step_with_spawn(
        &mut self,
        field: &mut FoamField,
        velocity: &VelocityField,
        jacobian_field: &[f32],
        dt: f32,
    ) {
        self.spawn_from_turbulence(field, jacobian_field);
        self.step(field, velocity, dt);
    }
}

impl Default for FoamAdvector {
    fn default() -> Self {
        Self::with_defaults()
    }
}

// ---------------------------------------------------------------------------
// FoamBubble
// ---------------------------------------------------------------------------

/// Individual bubble particle for detail layer.
#[derive(Clone, Copy, Debug, Default)]
pub struct FoamBubble {
    /// Position [x, y, z] in world space.
    pub position: [f32; 3],

    /// Velocity [vx, vy, vz] in world space.
    pub velocity: [f32; 3],

    /// Bubble size (diameter in meters).
    pub size: f32,

    /// Remaining lifetime in seconds.
    pub lifetime: f32,
}

impl FoamBubble {
    /// Create a new bubble.
    pub fn new(position: [f32; 3], velocity: [f32; 3], size: f32, lifetime: f32) -> Self {
        Self {
            position,
            velocity,
            size: size.max(0.001),
            lifetime: lifetime.max(0.0),
        }
    }

    /// Check if the bubble has expired.
    #[inline]
    pub fn is_expired(&self) -> bool {
        self.lifetime <= 0.0
    }

    /// Update bubble physics for one frame.
    ///
    /// # Arguments
    ///
    /// * `dt` - Time step in seconds
    /// * `gravity` - Gravity acceleration (default: 9.81)
    /// * `drag` - Drag coefficient (default: 0.5)
    pub fn update(&mut self, dt: f32, gravity: f32, drag: f32) {
        // Apply buoyancy (bubbles rise)
        // Buoyancy is proportional to size^3, drag to size^2*velocity^2
        // For simplicity, we use a linear model
        let buoyancy = gravity * 0.5; // Reduced effective gravity due to buoyancy

        // Update velocity with buoyancy and drag
        self.velocity[1] += buoyancy * dt; // Y-up

        // Apply drag (proportional to velocity squared)
        let speed_sq = self.velocity[0].powi(2)
            + self.velocity[1].powi(2)
            + self.velocity[2].powi(2);

        if speed_sq > EPSILON {
            let speed = speed_sq.sqrt();
            let drag_factor = (1.0 - drag * speed * dt).max(0.0);
            self.velocity[0] *= drag_factor;
            self.velocity[1] *= drag_factor;
            self.velocity[2] *= drag_factor;
        }

        // Update position
        self.position[0] += self.velocity[0] * dt;
        self.position[1] += self.velocity[1] * dt;
        self.position[2] += self.velocity[2] * dt;

        // Update lifetime
        self.lifetime -= dt;
    }
}

// ---------------------------------------------------------------------------
// FoamBubbles
// ---------------------------------------------------------------------------

/// Particle system for foam bubble detail.
///
/// Manages a collection of individual bubble particles that provide
/// fine detail on top of the advected foam field.
#[derive(Clone, Debug)]
pub struct FoamBubbles {
    /// Active bubbles.
    bubbles: Vec<FoamBubble>,

    /// Maximum number of bubbles.
    max_bubbles: usize,

    /// Default bubble lifetime.
    default_lifetime: f32,

    /// Gravity for physics.
    gravity: f32,

    /// Drag coefficient.
    drag: f32,
}

impl FoamBubbles {
    /// Create a new bubble system.
    pub fn new(max_bubbles: usize) -> Self {
        Self {
            bubbles: Vec::with_capacity(max_bubbles.min(MAX_BUBBLES)),
            max_bubbles: max_bubbles.min(MAX_BUBBLES),
            default_lifetime: DEFAULT_BUBBLE_LIFETIME,
            gravity: BUBBLE_GRAVITY,
            drag: BUBBLE_DRAG,
        }
    }

    /// Create with default settings.
    pub fn with_defaults() -> Self {
        Self::new(MAX_BUBBLES)
    }

    /// Spawn a new bubble.
    ///
    /// # Arguments
    ///
    /// * `pos` - Initial position [x, y, z]
    /// * `velocity` - Initial velocity [vx, vy, vz]
    /// * `size` - Bubble diameter in meters
    ///
    /// Returns true if bubble was spawned, false if at capacity.
    pub fn spawn_bubble(&mut self, pos: [f32; 3], velocity: [f32; 3], size: f32) -> bool {
        if self.bubbles.len() >= self.max_bubbles {
            return false;
        }

        self.bubbles.push(FoamBubble::new(
            pos,
            velocity,
            size,
            self.default_lifetime,
        ));
        true
    }

    /// Spawn a bubble with custom lifetime.
    pub fn spawn_bubble_with_lifetime(
        &mut self,
        pos: [f32; 3],
        velocity: [f32; 3],
        size: f32,
        lifetime: f32,
    ) -> bool {
        if self.bubbles.len() >= self.max_bubbles {
            return false;
        }

        self.bubbles.push(FoamBubble::new(pos, velocity, size, lifetime));
        true
    }

    /// Update all bubble physics.
    pub fn update(&mut self, dt: f32) {
        for bubble in &mut self.bubbles {
            bubble.update(dt, self.gravity, self.drag);
        }
    }

    /// Remove all expired bubbles.
    ///
    /// Returns the number of bubbles removed.
    pub fn pop_expired(&mut self) -> usize {
        let initial_count = self.bubbles.len();
        self.bubbles.retain(|b| !b.is_expired());
        initial_count - self.bubbles.len()
    }

    /// Update physics and remove expired bubbles.
    pub fn step(&mut self, dt: f32) {
        self.update(dt);
        self.pop_expired();
    }

    /// Get the number of active bubbles.
    #[inline]
    pub fn get_active_count(&self) -> usize {
        self.bubbles.len()
    }

    /// Get bubble positions for rendering.
    pub fn get_positions(&self) -> Vec<[f32; 3]> {
        self.bubbles.iter().map(|b| b.position).collect()
    }

    /// Get bubble positions as a flat slice (for GPU upload).
    pub fn get_positions_flat(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.bubbles.len() * 3);
        for b in &self.bubbles {
            result.push(b.position[0]);
            result.push(b.position[1]);
            result.push(b.position[2]);
        }
        result
    }

    /// Get bubble sizes for rendering.
    pub fn get_sizes(&self) -> Vec<f32> {
        self.bubbles.iter().map(|b| b.size).collect()
    }

    /// Clear all bubbles.
    pub fn clear(&mut self) {
        self.bubbles.clear();
    }

    /// Get reference to all bubbles.
    pub fn bubbles(&self) -> &[FoamBubble] {
        &self.bubbles
    }

    /// Get mutable reference to all bubbles.
    pub fn bubbles_mut(&mut self) -> &mut [FoamBubble] {
        &mut self.bubbles
    }

    /// Set default bubble lifetime.
    pub fn set_default_lifetime(&mut self, lifetime: f32) {
        self.default_lifetime = lifetime.max(0.0);
    }

    /// Set physics parameters.
    pub fn set_physics(&mut self, gravity: f32, drag: f32) {
        self.gravity = gravity;
        self.drag = drag.max(0.0);
    }
}

impl Default for FoamBubbles {
    fn default() -> Self {
        Self::with_defaults()
    }
}

// ---------------------------------------------------------------------------
// FoamRenderer (GPU Integration Placeholder)
// ---------------------------------------------------------------------------

/// Handle to a GPU texture resource.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct TextureHandle(pub u32);

/// Foam renderer for GPU integration.
///
/// Manages GPU resources for foam rendering, including texture uploads
/// and shader bindings.
#[derive(Clone, Debug)]
pub struct FoamRenderer {
    /// Configuration.
    pub config: FoamAdvectionConfig,

    /// CPU-side staging buffer.
    staging_buffer: Vec<f32>,

    /// Texture handle (placeholder for actual GPU resource).
    texture_handle: TextureHandle,

    /// Texture dimensions.
    texture_width: u32,
    texture_height: u32,
}

impl FoamRenderer {
    /// Create a new foam renderer.
    pub fn new(config: FoamAdvectionConfig) -> Self {
        let size = (config.grid_resolution * config.grid_resolution) as usize;
        Self {
            config,
            staging_buffer: vec![0.0; size],
            texture_handle: TextureHandle(0),
            texture_width: config.grid_resolution,
            texture_height: config.grid_resolution,
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(FoamAdvectionConfig::default())
    }

    /// Upload foam field data to GPU staging buffer.
    ///
    /// In a real implementation, this would upload to a GPU buffer/texture.
    pub fn upload_foam_field(&mut self, field: &FoamField) {
        debug_assert_eq!(field.width, self.texture_width);
        debug_assert_eq!(field.height, self.texture_height);

        self.staging_buffer.copy_from_slice(field.data());
    }

    /// Get the foam texture handle.
    pub fn get_foam_texture_handle(&self) -> TextureHandle {
        self.texture_handle
    }

    /// Set the texture handle (for external GPU resource management).
    pub fn set_foam_texture_handle(&mut self, handle: TextureHandle) {
        self.texture_handle = handle;
    }

    /// Sample foam value for shading (from staging buffer).
    ///
    /// In a real implementation, this would sample from GPU texture.
    pub fn sample_foam_for_shading(&self, uv: [f32; 2]) -> f32 {
        let u = uv[0].clamp(0.0, 1.0);
        let v = uv[1].clamp(0.0, 1.0);

        let x = u * (self.texture_width as f32 - 1.0);
        let y = v * (self.texture_height as f32 - 1.0);

        let x0 = x.floor() as u32;
        let y0 = y.floor() as u32;
        let x1 = (x0 + 1).min(self.texture_width - 1);
        let y1 = (y0 + 1).min(self.texture_height - 1);

        let fx = x.fract();
        let fy = y.fract();

        let idx00 = (y0 * self.texture_width + x0) as usize;
        let idx10 = (y0 * self.texture_width + x1) as usize;
        let idx01 = (y1 * self.texture_width + x0) as usize;
        let idx11 = (y1 * self.texture_width + x1) as usize;

        let v00 = self.staging_buffer[idx00];
        let v10 = self.staging_buffer[idx10];
        let v01 = self.staging_buffer[idx01];
        let v11 = self.staging_buffer[idx11];

        let v0 = v00 * (1.0 - fx) + v10 * fx;
        let v1 = v01 * (1.0 - fx) + v11 * fx;

        v0 * (1.0 - fy) + v1 * fy
    }

    /// Get staging buffer as bytes (for GPU upload).
    pub fn staging_buffer_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.staging_buffer)
    }

    /// Get staging buffer as R8 normalized bytes (for texture upload).
    pub fn staging_buffer_r8(&self) -> Vec<u8> {
        self.staging_buffer
            .iter()
            .map(|&v| (v.clamp(0.0, 1.0) * 255.0) as u8)
            .collect()
    }

    /// Get texture dimensions.
    pub fn texture_dimensions(&self) -> (u32, u32) {
        (self.texture_width, self.texture_height)
    }
}

impl Default for FoamRenderer {
    fn default() -> Self {
        Self::with_defaults()
    }
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

    fn approx_eq_eps(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    // ========================================================================
    // FoamAdvectionConfig Tests
    // ========================================================================

    #[test]
    fn test_config_size() {
        assert_eq!(mem::size_of::<FoamAdvectionConfig>(), FOAM_ADVECTION_CONFIG_SIZE);
    }

    #[test]
    fn test_config_default() {
        let config = FoamAdvectionConfig::default();
        assert_eq!(config.grid_resolution, DEFAULT_GRID_RESOLUTION);
        assert!(approx_eq(config.cell_size, DEFAULT_CELL_SIZE));
        assert!(approx_eq(config.advection_speed, DEFAULT_ADVECTION_SPEED));
        assert!(approx_eq(config.diffusion_rate, DEFAULT_DIFFUSION_RATE));
        assert!(approx_eq(config.spawn_rate, DEFAULT_SPAWN_RATE));
        assert!(approx_eq(config.decay_rate, DEFAULT_DECAY_RATE));
        assert!(approx_eq(config.max_concentration, DEFAULT_MAX_CONCENTRATION));
    }

    #[test]
    fn test_config_new_clamps() {
        let config = FoamAdvectionConfig::new(1, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0);
        assert!(config.grid_resolution >= 2);
        assert!(config.cell_size > 0.0);
        assert!(config.advection_speed >= 0.0);
        assert!(config.diffusion_rate >= 0.0);
        assert!(config.spawn_rate >= 0.0);
        assert!(config.decay_rate >= 0.0);
        assert!(config.max_concentration > 0.0);
    }

    #[test]
    fn test_config_presets() {
        let hq = FoamAdvectionConfig::high_quality();
        let perf = FoamAdvectionConfig::performance();

        assert!(hq.grid_resolution > perf.grid_resolution);
        assert!(hq.cell_size < perf.cell_size);
    }

    #[test]
    fn test_config_validate_valid() {
        assert!(FoamAdvectionConfig::default().validate().is_ok());
        assert!(FoamAdvectionConfig::high_quality().validate().is_ok());
        assert!(FoamAdvectionConfig::performance().validate().is_ok());
    }

    #[test]
    fn test_config_validate_invalid_resolution() {
        let mut config = FoamAdvectionConfig::default();
        config.grid_resolution = 1;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_invalid_cell_size() {
        let mut config = FoamAdvectionConfig::default();
        config.cell_size = 0.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_invalid_advection_speed() {
        let mut config = FoamAdvectionConfig::default();
        config.advection_speed = -1.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_invalid_diffusion() {
        let mut config = FoamAdvectionConfig::default();
        config.diffusion_rate = -0.1;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_invalid_spawn() {
        let mut config = FoamAdvectionConfig::default();
        config.spawn_rate = -0.1;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_invalid_decay() {
        let mut config = FoamAdvectionConfig::default();
        config.decay_rate = -0.1;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_invalid_max_concentration() {
        let mut config = FoamAdvectionConfig::default();
        config.max_concentration = 0.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_world_size() {
        let config = FoamAdvectionConfig::new(100, 0.5, 1.0, 0.01, 1.0, 0.5, 1.0);
        assert!(approx_eq(config.world_size(), 50.0));
    }

    #[test]
    fn test_config_bytemuck() {
        let config = FoamAdvectionConfig::default();
        let bytes: &[u8] = bytemuck::bytes_of(&config);
        assert_eq!(bytes.len(), FOAM_ADVECTION_CONFIG_SIZE);
    }

    // ========================================================================
    // FoamField Tests
    // ========================================================================

    #[test]
    fn test_field_new() {
        let field = FoamField::new(64, 64);
        assert_eq!(field.width, 64);
        assert_eq!(field.height, 64);
        assert_eq!(field.cell_count(), 64 * 64);
    }

    #[test]
    fn test_field_get_set() {
        let mut field = FoamField::new(4, 4);
        field.set(1, 2, 0.75);
        assert!(approx_eq(field.get(1, 2), 0.75));
    }

    #[test]
    fn test_field_get_out_of_bounds() {
        let field = FoamField::new(4, 4);
        assert!(approx_eq(field.get(10, 10), 0.0));
    }

    #[test]
    fn test_field_get_i32_negative() {
        let field = FoamField::new(4, 4);
        assert!(approx_eq(field.get_i32(-1, 0), 0.0));
        assert!(approx_eq(field.get_i32(0, -1), 0.0));
    }

    #[test]
    fn test_field_add() {
        let mut field = FoamField::new(4, 4);
        field.set(1, 1, 0.5);
        field.add(1, 1, 0.3);
        assert!(approx_eq(field.get(1, 1), 0.8));
    }

    #[test]
    fn test_field_add_out_of_bounds() {
        let mut field = FoamField::new(4, 4);
        field.add(10, 10, 1.0); // Should not crash
        assert!(approx_eq(field.get(10, 10), 0.0));
    }

    #[test]
    fn test_field_bilinear_corners() {
        let mut field = FoamField::new(2, 2);
        field.set(0, 0, 1.0);
        field.set(1, 0, 0.0);
        field.set(0, 1, 0.0);
        field.set(1, 1, 0.0);

        assert!(approx_eq(field.sample_bilinear(0.0, 0.0), 1.0));
        assert!(approx_eq(field.sample_bilinear(1.0, 1.0), 0.0));
    }

    #[test]
    fn test_field_bilinear_center() {
        let mut field = FoamField::new(2, 2);
        field.set(0, 0, 1.0);
        field.set(1, 0, 1.0);
        field.set(0, 1, 0.0);
        field.set(1, 1, 0.0);

        let center = field.sample_bilinear(0.5, 0.5);
        assert!(approx_eq_eps(center, 0.5, 0.1));
    }

    #[test]
    fn test_field_bilinear_cell_coords() {
        let mut field = FoamField::new(4, 4);
        field.set(1, 1, 1.0);

        let sample = field.sample_bilinear_cell(1.0, 1.0);
        assert!(approx_eq(sample, 1.0));
    }

    #[test]
    fn test_field_clear() {
        let mut field = FoamField::new(4, 4);
        field.set(1, 1, 0.5);
        field.clear();
        assert!(approx_eq(field.get(1, 1), 0.0));
    }

    #[test]
    fn test_field_fill() {
        let mut field = FoamField::new(4, 4);
        field.fill(0.7);
        assert!(approx_eq(field.get(0, 0), 0.7));
        assert!(approx_eq(field.get(3, 3), 0.7));
    }

    #[test]
    fn test_field_clamp_max() {
        let mut field = FoamField::new(4, 4);
        field.set(0, 0, 2.0);
        field.set(1, 1, 0.5);
        field.clamp_max(1.0);
        assert!(approx_eq(field.get(0, 0), 1.0));
        assert!(approx_eq(field.get(1, 1), 0.5));
    }

    #[test]
    fn test_field_clamp_range() {
        let mut field = FoamField::new(4, 4);
        field.set(0, 0, 2.0);
        field.set(1, 1, -0.5);
        field.clamp_range(1.0);
        assert!(approx_eq(field.get(0, 0), 1.0));
        assert!(approx_eq(field.get(1, 1), 0.0));
    }

    #[test]
    fn test_field_total_mass() {
        let mut field = FoamField::new(2, 2);
        field.set(0, 0, 0.25);
        field.set(1, 0, 0.25);
        field.set(0, 1, 0.25);
        field.set(1, 1, 0.25);
        assert!(approx_eq(field.total_mass(), 1.0));
    }

    #[test]
    fn test_field_average() {
        let mut field = FoamField::new(2, 2);
        field.set(0, 0, 0.0);
        field.set(1, 0, 0.0);
        field.set(0, 1, 1.0);
        field.set(1, 1, 1.0);
        assert!(approx_eq(field.average(), 0.5));
    }

    #[test]
    fn test_field_max_value() {
        let mut field = FoamField::new(4, 4);
        field.set(2, 2, 0.8);
        assert!(approx_eq(field.max_value(), 0.8));
    }

    #[test]
    fn test_field_from_data() {
        let data = vec![0.1, 0.2, 0.3, 0.4];
        let field = FoamField::from_data(2, 2, data).unwrap();
        assert!(approx_eq(field.get(0, 0), 0.1));
        assert!(approx_eq(field.get(1, 1), 0.4));
    }

    #[test]
    fn test_field_from_data_invalid_size() {
        let data = vec![0.1, 0.2, 0.3];
        assert!(FoamField::from_data(2, 2, data).is_none());
    }

    #[test]
    fn test_field_copy_from() {
        let mut field1 = FoamField::new(4, 4);
        let mut field2 = FoamField::new(4, 4);
        field2.set(1, 1, 0.9);

        field1.copy_from(&field2);
        assert!(approx_eq(field1.get(1, 1), 0.9));
    }

    #[test]
    fn test_field_swap() {
        let mut field1 = FoamField::new(4, 4);
        let mut field2 = FoamField::new(4, 4);
        field1.set(0, 0, 1.0);
        field2.set(1, 1, 2.0);

        field1.swap(&mut field2);
        assert!(approx_eq(field1.get(1, 1), 2.0));
        assert!(approx_eq(field2.get(0, 0), 1.0));
    }

    // ========================================================================
    // VelocityField Tests
    // ========================================================================

    #[test]
    fn test_velocity_new() {
        let vel = VelocityField::new(32, 32);
        assert_eq!(vel.width, 32);
        assert_eq!(vel.height, 32);
        assert_eq!(vel.cell_count(), 32 * 32);
    }

    #[test]
    fn test_velocity_get_set() {
        let mut vel = VelocityField::new(4, 4);
        vel.set(1, 2, 0.5, -0.3);
        let v = vel.get(1, 2);
        assert!(approx_eq(v[0], 0.5));
        assert!(approx_eq(v[1], -0.3));
    }

    #[test]
    fn test_velocity_get_out_of_bounds() {
        let vel = VelocityField::new(4, 4);
        let v = vel.get(10, 10);
        assert!(approx_eq(v[0], 0.0));
        assert!(approx_eq(v[1], 0.0));
    }

    #[test]
    fn test_velocity_get_i32_negative() {
        let vel = VelocityField::new(4, 4);
        let v = vel.get_i32(-1, 0);
        assert!(approx_eq(v[0], 0.0));
        assert!(approx_eq(v[1], 0.0));
    }

    #[test]
    fn test_velocity_bilinear() {
        let mut vel = VelocityField::new(2, 2);
        vel.set(0, 0, 1.0, 0.0);
        vel.set(1, 0, 0.0, 0.0);
        vel.set(0, 1, 0.0, 1.0);
        vel.set(1, 1, 0.0, 0.0);

        let v00 = vel.sample_bilinear(0.0, 0.0);
        assert!(approx_eq(v00[0], 1.0));
        assert!(approx_eq(v00[1], 0.0));
    }

    #[test]
    fn test_velocity_bilinear_interpolates() {
        let mut vel = VelocityField::new(2, 2);
        vel.set(0, 0, 1.0, 1.0);
        vel.set(1, 0, 1.0, 1.0);
        vel.set(0, 1, -1.0, -1.0);
        vel.set(1, 1, -1.0, -1.0);

        let center = vel.sample_bilinear(0.5, 0.5);
        assert!(approx_eq_eps(center[0], 0.0, 0.1));
        assert!(approx_eq_eps(center[1], 0.0, 0.1));
    }

    #[test]
    fn test_velocity_bilinear_cell() {
        let mut vel = VelocityField::new(4, 4);
        vel.set(1, 1, 0.5, 0.5);

        let v = vel.sample_bilinear_cell(1.0, 1.0);
        assert!(approx_eq(v[0], 0.5));
        assert!(approx_eq(v[1], 0.5));
    }

    #[test]
    fn test_velocity_clear() {
        let mut vel = VelocityField::new(4, 4);
        vel.set(1, 1, 1.0, 1.0);
        vel.clear();
        let v = vel.get(1, 1);
        assert!(approx_eq(v[0], 0.0));
        assert!(approx_eq(v[1], 0.0));
    }

    #[test]
    fn test_velocity_set_uniform() {
        let mut vel = VelocityField::new(4, 4);
        vel.set_uniform(0.5, -0.5);
        let v = vel.get(3, 3);
        assert!(approx_eq(v[0], 0.5));
        assert!(approx_eq(v[1], -0.5));
    }

    #[test]
    fn test_velocity_from_wave_gradient_flat() {
        let mut vel = VelocityField::new(4, 4);
        let heights = vec![0.0; 16]; // Flat surface

        vel.from_wave_gradient(&heights, 1.0, 1.0);

        // Flat surface = no gradient = zero velocity
        for y in 0..4 {
            for x in 0..4 {
                let v = vel.get(x, y);
                assert!(approx_eq(v[0], 0.0));
                assert!(approx_eq(v[1], 0.0));
            }
        }
    }

    #[test]
    fn test_velocity_from_wave_gradient_sloped() {
        let mut vel = VelocityField::new(4, 4);
        // Height increases in X direction
        let heights: Vec<f32> = (0..16)
            .map(|i| (i % 4) as f32)
            .collect();

        vel.from_wave_gradient(&heights, 1.0, 1.0);

        // Should have negative X velocity (downhill flow)
        let v = vel.get(2, 2);
        assert!(v[0] < 0.0, "Expected negative X velocity, got {}", v[0]);
    }

    #[test]
    fn test_velocity_average_speed() {
        let mut vel = VelocityField::new(2, 2);
        vel.set(0, 0, 3.0, 4.0); // Speed = 5
        vel.set(1, 0, 0.0, 0.0); // Speed = 0
        vel.set(0, 1, 0.0, 0.0);
        vel.set(1, 1, 0.0, 0.0);

        let avg = vel.average_speed();
        assert!(approx_eq(avg, 1.25)); // (5 + 0 + 0 + 0) / 4
    }

    #[test]
    fn test_velocity_max_speed() {
        let mut vel = VelocityField::new(4, 4);
        vel.set(1, 1, 3.0, 4.0); // Speed = 5
        assert!(approx_eq(vel.max_speed(), 5.0));
    }

    // ========================================================================
    // FoamAdvector Tests
    // ========================================================================

    #[test]
    fn test_advector_new() {
        let advector = FoamAdvector::new(FoamAdvectionConfig::default());
        assert!(advector.config.validate().is_ok());
    }

    #[test]
    fn test_advector_default() {
        let advector = FoamAdvector::default();
        assert_eq!(advector.config.grid_resolution, DEFAULT_GRID_RESOLUTION);
    }

    #[test]
    fn test_advect_stationary() {
        let config = FoamAdvectionConfig::new(4, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        let mut advector = FoamAdvector::new(config);

        let mut field = FoamField::new(4, 4);
        field.set(2, 2, 1.0);

        let velocity = VelocityField::new(4, 4); // Zero velocity

        let mass_before = field.total_mass();
        advector.advect(&mut field, &velocity, 0.1);
        let mass_after = field.total_mass();

        // Mass should be conserved (approximately, due to bilinear sampling)
        assert!(approx_eq_eps(mass_before, mass_after, 0.01));
    }

    #[test]
    fn test_advect_transport() {
        let config = FoamAdvectionConfig::new(8, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        let mut advector = FoamAdvector::new(config);

        let mut field = FoamField::new(8, 8);
        field.set(4, 4, 1.0);

        let mut velocity = VelocityField::new(8, 8);
        velocity.set_uniform(1.0, 0.0); // Move right

        advector.advect(&mut field, &velocity, 1.0);

        // Foam should have moved right (x+1 direction)
        // Due to semi-Lagrangian, value at (5,4) should be higher
        assert!(field.get(4, 4) < 1.0, "Original position should decrease");
    }

    #[test]
    fn test_advect_stability() {
        // Semi-Lagrangian should be stable for large dt
        let config = FoamAdvectionConfig::new(4, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        let mut advector = FoamAdvector::new(config);

        let mut field = FoamField::new(4, 4);
        field.set(2, 2, 1.0);

        let mut velocity = VelocityField::new(4, 4);
        velocity.set_uniform(10.0, 10.0); // Large velocity

        advector.advect(&mut field, &velocity, 10.0); // Large dt

        // Should not explode
        assert!(field.max_value() <= 1.0);
        assert!(field.max_value() >= 0.0);
    }

    #[test]
    fn test_diffuse_spreads() {
        let config = FoamAdvectionConfig::new(8, 1.0, 0.0, 0.1, 0.0, 0.0, 1.0);
        let mut advector = FoamAdvector::new(config);

        let mut field = FoamField::new(8, 8);
        field.set(4, 4, 1.0);

        let initial_peak = field.get(4, 4);
        advector.diffuse(&mut field, 1.0);
        let final_peak = field.get(4, 4);

        // Peak should decrease as foam spreads
        assert!(final_peak < initial_peak);

        // Neighbors should increase
        assert!(field.get(3, 4) > 0.0);
        assert!(field.get(5, 4) > 0.0);
        assert!(field.get(4, 3) > 0.0);
        assert!(field.get(4, 5) > 0.0);
    }

    #[test]
    fn test_diffuse_conserves_mass() {
        let config = FoamAdvectionConfig::new(8, 1.0, 0.0, 0.1, 0.0, 0.0, 1.0);
        let mut advector = FoamAdvector::new(config);

        let mut field = FoamField::new(8, 8);
        field.set(4, 4, 1.0);

        let mass_before = field.total_mass();
        advector.diffuse(&mut field, 0.5);
        let mass_after = field.total_mass();

        // Mass should be approximately conserved (boundary effects may cause small loss)
        assert!(approx_eq_eps(mass_before, mass_after, 0.1));
    }

    #[test]
    fn test_decay_exponential() {
        let config = FoamAdvectionConfig::new(4, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0);
        let mut advector = FoamAdvector::new(config);

        let mut field = FoamField::new(4, 4);
        field.fill(1.0);

        advector.decay(&mut field, 1.0);

        // After 1 second with decay_rate=1.0, value should be e^(-1) ~= 0.368
        let expected = (-1.0_f32).exp();
        assert!(approx_eq_eps(field.get(2, 2), expected, 0.01));
    }

    #[test]
    fn test_decay_zero_rate() {
        let config = FoamAdvectionConfig::new(4, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        let mut advector = FoamAdvector::new(config);

        let mut field = FoamField::new(4, 4);
        field.fill(0.5);

        advector.decay(&mut field, 10.0);

        // No decay should occur
        assert!(approx_eq(field.get(2, 2), 0.5));
    }

    #[test]
    fn test_spawn_from_turbulence() {
        let config = FoamAdvectionConfig::new(4, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0);
        let advector = FoamAdvector::new(config);

        let mut field = FoamField::new(4, 4);
        let mut jacobian = vec![1.0; 16]; // No folding
        jacobian[5] = -0.5; // Folded wave at (1,1)

        advector.spawn_from_turbulence(&mut field, &jacobian);

        // Foam should appear where Jacobian is negative
        assert!(field.get(1, 1) > 0.0);
        assert!(approx_eq(field.get(0, 0), 0.0));
    }

    #[test]
    fn test_spawn_from_collision() {
        let config = FoamAdvectionConfig::new(4, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0);
        let advector = FoamAdvector::new(config);

        let mut field = FoamField::new(4, 4);
        let mut sdf = vec![1.0; 16]; // No obstacles
        sdf[5] = -0.2; // Obstacle collision at (1,1)

        advector.spawn_from_collision(&mut field, &sdf);

        // Foam should appear at collision point
        assert!(field.get(1, 1) > 0.0);
        assert!(approx_eq(field.get(0, 0), 0.0));
    }

    #[test]
    fn test_step_full() {
        let config = FoamAdvectionConfig::new(8, 1.0, 1.0, 0.01, 0.0, 0.1, 1.0);
        let mut advector = FoamAdvector::new(config);

        let mut field = FoamField::new(8, 8);
        field.set(4, 4, 1.0);

        let velocity = VelocityField::new(8, 8);

        advector.step(&mut field, &velocity, 0.016);

        // Field should still be valid
        assert!(field.max_value() <= 1.0);
        assert!(field.max_value() >= 0.0);
    }

    #[test]
    fn test_step_with_spawn() {
        let config = FoamAdvectionConfig::new(4, 1.0, 1.0, 0.01, 1.0, 0.1, 1.0);
        let mut advector = FoamAdvector::new(config);

        let mut field = FoamField::new(4, 4);
        let velocity = VelocityField::new(4, 4);
        let mut jacobian = vec![1.0; 16];
        jacobian[5] = -0.5;

        advector.step_with_spawn(&mut field, &velocity, &jacobian, 0.016);

        // Foam should have been spawned and processed
        assert!(field.get(1, 1) > 0.0);
    }

    // ========================================================================
    // FoamBubble Tests
    // ========================================================================

    #[test]
    fn test_bubble_new() {
        let bubble = FoamBubble::new([1.0, 2.0, 3.0], [0.1, 0.2, 0.3], 0.05, 2.0);
        assert!(approx_eq(bubble.position[0], 1.0));
        assert!(approx_eq(bubble.size, 0.05));
        assert!(approx_eq(bubble.lifetime, 2.0));
    }

    #[test]
    fn test_bubble_is_expired() {
        let live = FoamBubble::new([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.05, 1.0);
        let dead = FoamBubble::new([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.05, 0.0);

        assert!(!live.is_expired());
        assert!(dead.is_expired());
    }

    #[test]
    fn test_bubble_update_position() {
        let mut bubble = FoamBubble::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.05, 2.0);
        bubble.update(0.5, 0.0, 0.0); // No gravity/drag

        // Position should change based on velocity
        assert!(bubble.position[0] > 0.0);
    }

    #[test]
    fn test_bubble_update_lifetime() {
        let mut bubble = FoamBubble::new([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.05, 1.0);
        bubble.update(0.5, 0.0, 0.0);

        assert!(approx_eq(bubble.lifetime, 0.5));
    }

    #[test]
    fn test_bubble_update_buoyancy() {
        let mut bubble = FoamBubble::new([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.05, 2.0);
        bubble.update(0.1, BUBBLE_GRAVITY, 0.0);

        // Bubble should rise (positive Y)
        assert!(bubble.velocity[1] > 0.0);
    }

    #[test]
    fn test_bubble_update_drag() {
        let mut bubble = FoamBubble::new([0.0, 0.0, 0.0], [10.0, 0.0, 0.0], 0.05, 2.0);
        let initial_speed = bubble.velocity[0];

        bubble.update(0.1, 0.0, 0.5); // With drag

        // Speed should decrease
        assert!(bubble.velocity[0] < initial_speed);
    }

    // ========================================================================
    // FoamBubbles Tests
    // ========================================================================

    #[test]
    fn test_bubbles_new() {
        let bubbles = FoamBubbles::new(1000);
        assert_eq!(bubbles.get_active_count(), 0);
    }

    #[test]
    fn test_bubbles_spawn() {
        let mut bubbles = FoamBubbles::new(100);
        let spawned = bubbles.spawn_bubble([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.05);

        assert!(spawned);
        assert_eq!(bubbles.get_active_count(), 1);
    }

    #[test]
    fn test_bubbles_spawn_at_capacity() {
        let mut bubbles = FoamBubbles::new(2);
        bubbles.spawn_bubble([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.05);
        bubbles.spawn_bubble([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.05);

        let spawned = bubbles.spawn_bubble([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.05);
        assert!(!spawned);
        assert_eq!(bubbles.get_active_count(), 2);
    }

    #[test]
    fn test_bubbles_spawn_with_lifetime() {
        let mut bubbles = FoamBubbles::new(100);
        bubbles.spawn_bubble_with_lifetime([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.05, 5.0);

        assert!(approx_eq(bubbles.bubbles()[0].lifetime, 5.0));
    }

    #[test]
    fn test_bubbles_update() {
        let mut bubbles = FoamBubbles::new(100);
        bubbles.spawn_bubble([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.05);

        let initial_x = bubbles.bubbles()[0].position[0];
        bubbles.update(0.1);

        assert!(bubbles.bubbles()[0].position[0] != initial_x);
    }

    #[test]
    fn test_bubbles_pop_expired() {
        let mut bubbles = FoamBubbles::new(100);
        bubbles.spawn_bubble_with_lifetime([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.05, 0.5);
        bubbles.spawn_bubble_with_lifetime([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.05, 2.0);

        // Advance time past first bubble's lifetime
        bubbles.update(1.0);
        let removed = bubbles.pop_expired();

        assert_eq!(removed, 1);
        assert_eq!(bubbles.get_active_count(), 1);
    }

    #[test]
    fn test_bubbles_step() {
        let mut bubbles = FoamBubbles::new(100);
        bubbles.spawn_bubble_with_lifetime([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.05, 0.05);

        bubbles.step(0.1); // Should update and remove expired

        assert_eq!(bubbles.get_active_count(), 0);
    }

    #[test]
    fn test_bubbles_get_positions() {
        let mut bubbles = FoamBubbles::new(100);
        bubbles.spawn_bubble([1.0, 2.0, 3.0], [0.0, 0.0, 0.0], 0.05);
        bubbles.spawn_bubble([4.0, 5.0, 6.0], [0.0, 0.0, 0.0], 0.05);

        let positions = bubbles.get_positions();
        assert_eq!(positions.len(), 2);
        assert!(approx_eq(positions[0][0], 1.0));
        assert!(approx_eq(positions[1][0], 4.0));
    }

    #[test]
    fn test_bubbles_get_positions_flat() {
        let mut bubbles = FoamBubbles::new(100);
        bubbles.spawn_bubble([1.0, 2.0, 3.0], [0.0, 0.0, 0.0], 0.05);

        let flat = bubbles.get_positions_flat();
        assert_eq!(flat.len(), 3);
        assert!(approx_eq(flat[0], 1.0));
        assert!(approx_eq(flat[1], 2.0));
        assert!(approx_eq(flat[2], 3.0));
    }

    #[test]
    fn test_bubbles_get_sizes() {
        let mut bubbles = FoamBubbles::new(100);
        bubbles.spawn_bubble([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.05);
        bubbles.spawn_bubble([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.1);

        let sizes = bubbles.get_sizes();
        assert_eq!(sizes.len(), 2);
        assert!(approx_eq(sizes[0], 0.05));
        assert!(approx_eq(sizes[1], 0.1));
    }

    #[test]
    fn test_bubbles_clear() {
        let mut bubbles = FoamBubbles::new(100);
        bubbles.spawn_bubble([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.05);
        bubbles.spawn_bubble([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.05);

        bubbles.clear();
        assert_eq!(bubbles.get_active_count(), 0);
    }

    #[test]
    fn test_bubbles_set_physics() {
        let mut bubbles = FoamBubbles::new(100);
        bubbles.set_physics(5.0, 0.2);

        bubbles.spawn_bubble([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.05);
        bubbles.update(0.1);

        // Should use custom gravity (buoyancy derived from it)
        // Just verify it doesn't crash and produces reasonable values
        assert!(bubbles.bubbles()[0].velocity[1].is_finite());
    }

    #[test]
    fn test_bubbles_set_default_lifetime() {
        let mut bubbles = FoamBubbles::new(100);
        bubbles.set_default_lifetime(10.0);
        bubbles.spawn_bubble([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.05);

        assert!(approx_eq(bubbles.bubbles()[0].lifetime, 10.0));
    }

    // ========================================================================
    // FoamRenderer Tests
    // ========================================================================

    #[test]
    fn test_renderer_new() {
        let renderer = FoamRenderer::new(FoamAdvectionConfig::default());
        assert_eq!(renderer.texture_dimensions(), (DEFAULT_GRID_RESOLUTION, DEFAULT_GRID_RESOLUTION));
    }

    #[test]
    fn test_renderer_upload_foam_field() {
        let mut renderer = FoamRenderer::new(FoamAdvectionConfig::new(4, 1.0, 1.0, 0.01, 1.0, 0.5, 1.0));
        let mut field = FoamField::new(4, 4);
        field.set(1, 1, 0.75);

        renderer.upload_foam_field(&field);

        // Verify data was copied
        let sample = renderer.sample_foam_for_shading([0.33, 0.33]); // Roughly at (1,1)
        assert!(sample > 0.0);
    }

    #[test]
    fn test_renderer_sample_foam_for_shading() {
        let mut renderer = FoamRenderer::new(FoamAdvectionConfig::new(2, 1.0, 1.0, 0.01, 1.0, 0.5, 1.0));
        let mut field = FoamField::new(2, 2);
        field.set(0, 0, 1.0);
        field.set(1, 0, 0.0);
        field.set(0, 1, 0.0);
        field.set(1, 1, 0.0);

        renderer.upload_foam_field(&field);

        let corner = renderer.sample_foam_for_shading([0.0, 0.0]);
        assert!(approx_eq(corner, 1.0));

        let opposite = renderer.sample_foam_for_shading([1.0, 1.0]);
        assert!(approx_eq(opposite, 0.0));
    }

    #[test]
    fn test_renderer_texture_handle() {
        let mut renderer = FoamRenderer::with_defaults();
        renderer.set_foam_texture_handle(TextureHandle(42));
        assert_eq!(renderer.get_foam_texture_handle(), TextureHandle(42));
    }

    #[test]
    fn test_renderer_staging_buffer_r8() {
        let mut renderer = FoamRenderer::new(FoamAdvectionConfig::new(2, 1.0, 1.0, 0.01, 1.0, 0.5, 1.0));
        let mut field = FoamField::new(2, 2);
        field.set(0, 0, 0.5);
        field.set(1, 1, 1.0);

        renderer.upload_foam_field(&field);
        let r8 = renderer.staging_buffer_r8();

        assert_eq!(r8.len(), 4);
        assert_eq!(r8[0], 127); // 0.5 * 255 = 127.5
        assert_eq!(r8[3], 255); // 1.0 * 255 = 255
    }

    // ========================================================================
    // Grid Boundary Tests
    // ========================================================================

    #[test]
    fn test_boundary_wrap_advection() {
        // Test that advection handles boundaries gracefully
        let config = FoamAdvectionConfig::new(4, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        let mut advector = FoamAdvector::new(config);

        let mut field = FoamField::new(4, 4);
        field.set(0, 0, 1.0); // Corner

        let mut velocity = VelocityField::new(4, 4);
        velocity.set_uniform(-1.0, -1.0); // Point outside grid

        advector.advect(&mut field, &velocity, 1.0);

        // Should clamp to boundary, not crash
        assert!(field.max_value() <= 1.0);
    }

    #[test]
    fn test_boundary_clamp_bilinear() {
        let field = FoamField::new(4, 4);

        // Out of range samples should be clamped
        let low = field.sample_bilinear(-0.5, -0.5);
        let high = field.sample_bilinear(1.5, 1.5);

        assert!(low >= 0.0);
        assert!(high >= 0.0);
    }

    // ========================================================================
    // Mass Conservation Tests
    // ========================================================================

    #[test]
    fn test_mass_conservation_no_decay() {
        let config = FoamAdvectionConfig::new(16, 1.0, 1.0, 0.0, 0.0, 0.0, 10.0);
        let mut advector = FoamAdvector::new(config);

        let mut field = FoamField::new(16, 16);
        // Put foam in center (away from boundaries)
        for y in 6..10 {
            for x in 6..10 {
                field.set(x, y, 1.0);
            }
        }

        let velocity = VelocityField::new(16, 16);
        let mass_before = field.total_mass();

        // Just advection and diffusion (no decay, no spawn)
        for _ in 0..10 {
            advector.advect(&mut field, &velocity, 0.1);
            advector.diffuse(&mut field, 0.1);
        }

        let mass_after = field.total_mass();

        // Mass should be approximately conserved
        assert!(approx_eq_eps(mass_before, mass_after, 1.0));
    }

    #[test]
    fn test_mass_decreases_with_decay() {
        let config = FoamAdvectionConfig::new(4, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0);
        let mut advector = FoamAdvector::new(config);

        let mut field = FoamField::new(4, 4);
        field.fill(1.0);

        let mass_before = field.total_mass();
        advector.decay(&mut field, 1.0);
        let mass_after = field.total_mass();

        assert!(mass_after < mass_before);
    }

    // ========================================================================
    // Integration Tests
    // ========================================================================

    #[test]
    fn test_full_simulation_cycle() {
        let config = FoamAdvectionConfig::new(32, 0.5, 1.0, 0.01, 0.5, 0.2, 1.0);
        let mut advector = FoamAdvector::new(config);

        let mut field = FoamField::new(32, 32);
        let mut velocity = VelocityField::new(32, 32);
        let mut jacobian = vec![1.0; 32 * 32];

        // Set up some wave folding
        for y in 10..22 {
            for x in 10..22 {
                let idx = y * 32 + x;
                jacobian[idx] = -0.3;
            }
        }

        // Set up flow velocity
        velocity.set_uniform(0.5, 0.3);

        // Simulate several frames
        for _ in 0..60 {
            advector.step_with_spawn(&mut field, &velocity, &jacobian, 0.016);
        }

        // Foam should have spawned in the turbulent region
        let center_foam = field.get(16, 16);
        assert!(center_foam > 0.0, "Foam should spawn in turbulent region");

        // And should have spread/moved
        assert!(field.max_value() > 0.0);
    }

    #[test]
    fn test_bubbles_integration() {
        let mut bubbles = FoamBubbles::new(100);

        // Spawn bubbles
        for i in 0..10 {
            bubbles.spawn_bubble(
                [i as f32 * 0.1, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                0.02 + i as f32 * 0.002,
            );
        }

        assert_eq!(bubbles.get_active_count(), 10);

        // Simulate
        for _ in 0..100 {
            bubbles.step(0.016);
        }

        // Some bubbles should have expired (default lifetime = 2.0s, we simulated ~1.6s)
        // All should still be alive
        assert!(bubbles.get_active_count() > 0);

        // Bubbles should have risen
        for bubble in bubbles.bubbles() {
            assert!(bubble.position[1] > 0.0);
        }
    }

    // ========================================================================
    // Performance / Stress Tests
    // ========================================================================

    #[test]
    fn test_large_field_operations() {
        let config = FoamAdvectionConfig::new(256, 0.5, 1.0, 0.01, 1.0, 0.5, 1.0);
        let mut advector = FoamAdvector::new(config);

        let mut field = FoamField::new(256, 256);
        let velocity = VelocityField::new(256, 256);

        field.fill(0.5);

        // Should complete without timeout
        for _ in 0..10 {
            advector.step(&mut field, &velocity, 0.016);
        }

        assert!(field.max_value() <= 1.0);
    }

    #[test]
    fn test_many_bubbles() {
        let mut bubbles = FoamBubbles::new(5000);

        for i in 0..5000 {
            bubbles.spawn_bubble(
                [(i % 100) as f32, (i / 100) as f32, 0.0],
                [0.0, 0.1, 0.0],
                0.01,
            );
        }

        assert_eq!(bubbles.get_active_count(), 5000);

        // Simulate
        for _ in 0..10 {
            bubbles.step(0.016);
        }

        assert_eq!(bubbles.get_active_count(), 5000);
    }
}
