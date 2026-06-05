//! DDGI Infinite Scrolling Volume Manager.
//!
//! This module provides the [`DDGIScrollManager`] which handles infinite scrolling
//! for DDGI probe volumes. When the camera moves, probes at the trailing edge
//! "scroll" to the leading edge and are re-seeded with interpolated data.
//!
//! # Architecture
//!
//! The scrolling system operates in two phases:
//!
//! 1. **CPU Detection**: [`DDGIScrollManager::update`] detects when the camera
//!    has moved beyond the scroll threshold and computes the new scroll offset.
//!
//! 2. **GPU Re-seeding**: [`DDGIScrollManager::create_shift_pass`] creates a
//!    compute pass that runs `ddgi_grid_shift.comp.wgsl` to re-seed probes.
//!
//! # Scroll Threshold Recommendations
//!
//! The scroll threshold should be tuned based on:
//! - **Cell size**: Threshold should be < cell_size to prevent multiple scrolls
//! - **Camera speed**: Higher speeds need larger thresholds to reduce churn
//! - **Visual quality**: Lower thresholds give smoother lighting transitions
//!
//! | Scenario       | Cell Size | Recommended Threshold |
//! |----------------|-----------|----------------------|
//! | Walking/Indoor | 2.0m      | 0.5-1.0m             |
//! | Driving/Outdoor| 4.0m      | 1.0-2.0m             |
//! | Flying         | 8.0m      | 2.0-4.0m             |
//!
//! # Example
//!
//! ```ignore
//! let mut manager = DDGIScrollManager::new(2.0); // 2m threshold
//!
//! // Each frame:
//! if let Some(delta) = manager.update(camera_pos, &grid) {
//!     let pass = manager.create_shift_pass(delta, &grid);
//!     queue.submit(std::iter::once(pass.into_command_buffer(&device)));
//! }
//! ```

use bytemuck::{Pod, Zeroable};
use std::fmt;

use super::probe_grid::{ProbeGridGpu, ProbeSH};

// ============================================================================
// Scroll Delta
// ============================================================================

/// Represents a scroll operation with delta offsets.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ScrollDelta {
    /// Scroll delta in X axis (grid cells)
    pub dx: i32,
    /// Scroll delta in Y axis (grid cells)
    pub dy: i32,
    /// Scroll delta in Z axis (grid cells)
    pub dz: i32,
}

impl ScrollDelta {
    /// Zero delta (no scrolling).
    pub const ZERO: Self = Self { dx: 0, dy: 0, dz: 0 };

    /// Create a new scroll delta.
    pub fn new(dx: i32, dy: i32, dz: i32) -> Self {
        Self { dx, dy, dz }
    }

    /// Check if any scrolling is needed.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.dx == 0 && self.dy == 0 && self.dz == 0
    }

    /// Get the total number of probes affected by this scroll.
    ///
    /// This is an approximation: probes are affected if they lie in the
    /// scroll-in region on any axis.
    pub fn affected_probe_count(&self, dimensions: [u32; 3]) -> u32 {
        let mut count = 0u32;

        // X slab
        if self.dx != 0 {
            count += (self.dx.unsigned_abs()) * dimensions[1] * dimensions[2];
        }
        // Y slab
        if self.dy != 0 {
            count += dimensions[0] * (self.dy.unsigned_abs()) * dimensions[2];
        }
        // Z slab
        if self.dz != 0 {
            count += dimensions[0] * dimensions[1] * (self.dz.unsigned_abs());
        }

        count
    }

    /// Get the magnitude of the scroll (max absolute delta).
    #[inline]
    pub fn magnitude(&self) -> u32 {
        self.dx
            .unsigned_abs()
            .max(self.dy.unsigned_abs())
            .max(self.dz.unsigned_abs())
    }
}

impl fmt::Display for ScrollDelta {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ScrollDelta({}, {}, {})", self.dx, self.dy, self.dz)
    }
}

// ============================================================================
// GPU Uniform Structure
// ============================================================================

/// GPU-uploadable parameters for the grid shift compute shader.
///
/// Matches the WGSL `GridShiftParams` struct layout.
/// Size: 64 bytes (4 vec4s).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct GridShiftParams {
    /// Previous scroll offset (grid cell units)
    pub old_scroll_offset: [i32; 3],
    /// Padding for vec4 alignment
    pub _pad0: i32,
    /// New scroll offset (grid cell units)
    pub new_scroll_offset: [i32; 3],
    /// Padding for vec4 alignment
    pub _pad1: i32,
    /// Grid dimensions (probes per axis)
    pub dimensions: [u32; 3],
    /// Padding for vec4 alignment
    pub _pad2: u32,
    /// Frame index for noise dithering
    pub frame_index: u32,
    /// Grid cell spacing in world units
    pub cell_spacing: f32,
    /// Seed blend factor (0.0 = keep old, 1.0 = full reseed)
    pub seed_blend_factor: f32,
    /// Reserved
    pub _reserved: f32,
}

impl Default for GridShiftParams {
    fn default() -> Self {
        Self {
            old_scroll_offset: [0; 3],
            _pad0: 0,
            new_scroll_offset: [0; 3],
            _pad1: 0,
            dimensions: [8, 4, 8],
            _pad2: 0,
            frame_index: 0,
            cell_spacing: 2.0,
            seed_blend_factor: 0.8,
            _reserved: 0.0,
        }
    }
}

impl GridShiftParams {
    /// Create params for a scroll operation.
    pub fn new(
        old_offset: [i32; 3],
        new_offset: [i32; 3],
        dimensions: [u32; 3],
        frame_index: u32,
        cell_spacing: f32,
    ) -> Self {
        Self {
            old_scroll_offset: old_offset,
            _pad0: 0,
            new_scroll_offset: new_offset,
            _pad1: 0,
            dimensions,
            _pad2: 0,
            frame_index,
            cell_spacing,
            seed_blend_factor: 0.8,
            _reserved: 0.0,
        }
    }

    /// Compute dispatch size for the grid shift shader.
    ///
    /// Returns (workgroups_x, workgroups_y, workgroups_z) for an 8x8x1 workgroup.
    pub fn dispatch_size(&self) -> (u32, u32, u32) {
        let wg_x = (self.dimensions[0] + 7) / 8;
        let wg_y = (self.dimensions[1] + 7) / 8;
        let wg_z = self.dimensions[2];
        (wg_x, wg_y, wg_z)
    }
}

// ============================================================================
// Probe Status
// ============================================================================

/// Per-probe status for tracking re-seed state.
///
/// Matches WGSL `ProbeStatus` struct. Size: 16 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct ProbeStatus {
    /// Needs re-seeding this frame (0 or 1)
    pub needs_reseed: u32,
    /// Confidence level (0 = stale, 255 = fully converged)
    pub confidence: u32,
    /// Frames since last update
    pub stale_frames: u32,
    /// Reserved
    pub _reserved: u32,
}

impl ProbeStatus {
    /// Create a fresh probe status (high confidence).
    pub fn fresh() -> Self {
        Self {
            needs_reseed: 0,
            confidence: 255,
            stale_frames: 0,
            _reserved: 0,
        }
    }

    /// Create a stale probe status (low confidence).
    pub fn stale() -> Self {
        Self {
            needs_reseed: 1,
            confidence: 0,
            stale_frames: 0,
            _reserved: 0,
        }
    }

    /// Check if probe is reliable for sampling.
    #[inline]
    pub fn is_reliable(&self) -> bool {
        self.confidence > 128 && self.needs_reseed == 0
    }
}

// ============================================================================
// Scroll Manager
// ============================================================================

/// Scroll hysteresis mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ScrollHysteresis {
    /// Scroll immediately when threshold crossed
    #[default]
    Immediate,
    /// Require sustained movement before scrolling
    Sustained,
    /// Predictive scrolling based on velocity
    Predictive,
}

/// DDGI infinite scrolling volume manager.
///
/// Manages the scroll offset for a probe grid and detects when re-seeding
/// is needed based on camera movement.
#[derive(Clone, Debug)]
pub struct DDGIScrollManager {
    /// Current scroll offset in grid cell units.
    current_offset: [i32; 3],
    /// Pending scroll offset (when hysteresis is active).
    pending_offset: [i32; 3],
    /// Last known camera position.
    last_camera_pos: [f32; 3],
    /// Distance threshold before scroll triggers (in world units).
    shift_threshold: f32,
    /// Accumulated camera movement since last scroll.
    accumulated_delta: [f32; 3],
    /// Hysteresis mode for scroll triggering.
    hysteresis: ScrollHysteresis,
    /// Number of frames to sustain movement (for Sustained mode).
    sustain_frames: u32,
    /// Counter for sustained movement detection.
    sustain_counter: u32,
    /// Whether the manager has been initialized.
    initialized: bool,
    /// Current frame index.
    frame_index: u32,
}

impl Default for DDGIScrollManager {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl DDGIScrollManager {
    /// Create a new scroll manager with the given threshold.
    ///
    /// # Arguments
    ///
    /// * `shift_threshold` - Distance in world units before scrolling triggers.
    ///   Should be less than the probe grid cell size.
    ///
    /// # Panics
    ///
    /// Panics if `shift_threshold` is not positive.
    pub fn new(shift_threshold: f32) -> Self {
        assert!(shift_threshold > 0.0, "Shift threshold must be positive");
        Self {
            current_offset: [0; 3],
            pending_offset: [0; 3],
            last_camera_pos: [0.0; 3],
            shift_threshold,
            accumulated_delta: [0.0; 3],
            hysteresis: ScrollHysteresis::default(),
            sustain_frames: 3,
            sustain_counter: 0,
            initialized: false,
            frame_index: 0,
        }
    }

    /// Create with custom hysteresis settings.
    pub fn with_hysteresis(mut self, hysteresis: ScrollHysteresis, sustain_frames: u32) -> Self {
        self.hysteresis = hysteresis;
        self.sustain_frames = sustain_frames;
        self
    }

    /// Get current scroll offset.
    #[inline]
    pub fn current_offset(&self) -> [i32; 3] {
        self.current_offset
    }

    /// Get shift threshold.
    #[inline]
    pub fn shift_threshold(&self) -> f32 {
        self.shift_threshold
    }

    /// Set shift threshold.
    ///
    /// # Panics
    ///
    /// Panics if threshold is not positive.
    pub fn set_shift_threshold(&mut self, threshold: f32) {
        assert!(threshold > 0.0, "Shift threshold must be positive");
        self.shift_threshold = threshold;
    }

    /// Reset the scroll manager state.
    pub fn reset(&mut self) {
        self.current_offset = [0; 3];
        self.pending_offset = [0; 3];
        self.accumulated_delta = [0.0; 3];
        self.sustain_counter = 0;
        self.initialized = false;
    }

    /// Update scroll state based on camera movement.
    ///
    /// Returns `Some(ScrollDelta)` if scrolling is needed, `None` otherwise.
    ///
    /// # Arguments
    ///
    /// * `camera_pos` - Current camera world position.
    /// * `grid` - Reference to the probe grid for cell size information.
    pub fn update(&mut self, camera_pos: [f32; 3], grid: &ProbeGridGpu) -> Option<ScrollDelta> {
        self.frame_index = self.frame_index.wrapping_add(1);

        // First update: just store position
        if !self.initialized {
            self.last_camera_pos = camera_pos;
            self.initialized = true;
            return None;
        }

        // Compute camera delta
        let delta = [
            camera_pos[0] - self.last_camera_pos[0],
            camera_pos[1] - self.last_camera_pos[1],
            camera_pos[2] - self.last_camera_pos[2],
        ];

        // Accumulate delta
        self.accumulated_delta[0] += delta[0];
        self.accumulated_delta[1] += delta[1];
        self.accumulated_delta[2] += delta[2];

        // Update last position
        self.last_camera_pos = camera_pos;

        // Check if accumulated delta exceeds cell size (scroll in cell increments)
        let cell_size = grid.cell_size;

        let mut cell_delta = [0i32; 3];
        let mut any_scroll = false;

        for i in 0..3 {
            // Scroll when we've moved at least one cell
            // Use floor for positive and ceil for negative to get cell crossings
            let cells = if self.accumulated_delta[i] >= 0.0 {
                (self.accumulated_delta[i] / cell_size[i]).floor() as i32
            } else {
                (self.accumulated_delta[i] / cell_size[i]).ceil() as i32
            };

            if cells != 0 {
                cell_delta[i] = cells;
                // Subtract the consumed delta
                self.accumulated_delta[i] -= (cells as f32) * cell_size[i];
                any_scroll = true;
            }
        }

        if !any_scroll {
            // No scroll needed
            match self.hysteresis {
                ScrollHysteresis::Sustained => {
                    self.sustain_counter = 0;
                }
                _ => {}
            }
            return None;
        }

        // Apply hysteresis
        match self.hysteresis {
            ScrollHysteresis::Immediate => {
                // Scroll immediately
                self.current_offset[0] += cell_delta[0];
                self.current_offset[1] += cell_delta[1];
                self.current_offset[2] += cell_delta[2];

                Some(ScrollDelta::new(cell_delta[0], cell_delta[1], cell_delta[2]))
            }
            ScrollHysteresis::Sustained => {
                // Require multiple frames of movement
                self.pending_offset[0] += cell_delta[0];
                self.pending_offset[1] += cell_delta[1];
                self.pending_offset[2] += cell_delta[2];
                self.sustain_counter += 1;

                if self.sustain_counter >= self.sustain_frames {
                    // Apply accumulated scroll
                    let delta = ScrollDelta::new(
                        self.pending_offset[0],
                        self.pending_offset[1],
                        self.pending_offset[2],
                    );

                    self.current_offset[0] += self.pending_offset[0];
                    self.current_offset[1] += self.pending_offset[1];
                    self.current_offset[2] += self.pending_offset[2];

                    self.pending_offset = [0; 3];
                    self.sustain_counter = 0;

                    if !delta.is_zero() {
                        Some(delta)
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            ScrollHysteresis::Predictive => {
                // TODO: Implement velocity-based prediction
                // For now, fall back to immediate
                self.current_offset[0] += cell_delta[0];
                self.current_offset[1] += cell_delta[1];
                self.current_offset[2] += cell_delta[2];

                Some(ScrollDelta::new(cell_delta[0], cell_delta[1], cell_delta[2]))
            }
        }
    }

    /// Create GPU parameters for a grid shift operation.
    ///
    /// # Arguments
    ///
    /// * `delta` - The scroll delta from `update()`.
    /// * `grid` - Reference to the probe grid.
    /// * `old_offset` - Previous scroll offset (before this delta).
    pub fn create_shift_params(&self, delta: ScrollDelta, grid: &ProbeGridGpu) -> GridShiftParams {
        let old_offset = [
            self.current_offset[0] - delta.dx,
            self.current_offset[1] - delta.dy,
            self.current_offset[2] - delta.dz,
        ];

        GridShiftParams::new(
            old_offset,
            self.current_offset,
            grid.dimensions,
            self.frame_index,
            grid.cell_size[0], // Assuming uniform spacing
        )
    }

    /// Get current frame index.
    #[inline]
    pub fn frame_index(&self) -> u32 {
        self.frame_index
    }

    /// Manually set the scroll offset (for loading saved state).
    pub fn set_offset(&mut self, offset: [i32; 3]) {
        self.current_offset = offset;
    }

    /// Check if a probe index is in the scroll-in region.
    ///
    /// This is the CPU-side equivalent of `probe_needs_reseed` in the shader.
    pub fn probe_needs_reseed(
        &self,
        probe_idx: [u32; 3],
        delta: &ScrollDelta,
        dimensions: [u32; 3],
    ) -> bool {
        // X axis
        if delta.dx > 0 {
            let threshold_x = dimensions[0].saturating_sub(delta.dx as u32);
            if probe_idx[0] >= threshold_x {
                return true;
            }
        } else if delta.dx < 0 {
            if probe_idx[0] < (-delta.dx) as u32 {
                return true;
            }
        }

        // Y axis
        if delta.dy > 0 {
            let threshold_y = dimensions[1].saturating_sub(delta.dy as u32);
            if probe_idx[1] >= threshold_y {
                return true;
            }
        } else if delta.dy < 0 {
            if probe_idx[1] < (-delta.dy) as u32 {
                return true;
            }
        }

        // Z axis
        if delta.dz > 0 {
            let threshold_z = dimensions[2].saturating_sub(delta.dz as u32);
            if probe_idx[2] >= threshold_z {
                return true;
            }
        } else if delta.dz < 0 {
            if probe_idx[2] < (-delta.dz) as u32 {
                return true;
            }
        }

        false
    }

    /// Get list of probe indices that need re-seeding.
    ///
    /// Returns indices in linear buffer order.
    pub fn get_probes_to_reseed(&self, delta: &ScrollDelta, grid: &ProbeGridGpu) -> Vec<u32> {
        let dimensions = grid.dimensions;
        let mut result = Vec::new();

        for z in 0..dimensions[2] {
            for y in 0..dimensions[1] {
                for x in 0..dimensions[0] {
                    let idx = [x, y, z];
                    if self.probe_needs_reseed(idx, delta, dimensions) {
                        let linear = x + y * dimensions[0] + z * dimensions[0] * dimensions[1];
                        result.push(linear);
                    }
                }
            }
        }

        result
    }
}

// ============================================================================
// CPU-Side Probe Seeding (for testing/fallback)
// ============================================================================

/// CPU-side implementation of probe re-seeding from neighbors.
///
/// This is primarily for testing and as a fallback when GPU compute is unavailable.
pub fn seed_probe_from_neighbors_cpu(
    probe_idx: [u32; 3],
    delta: &ScrollDelta,
    probes: &[ProbeSH],
    dimensions: [u32; 3],
) -> ProbeSH {
    let mut accum_irradiance = [[0.0f32; 4]; 9];
    let mut accum_visibility = [0.0f32; 9];
    let mut total_weight = 0.0f32;

    // Sample 3x3x3 neighborhood
    for dz in -1i32..=1 {
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                // Skip self
                if dx == 0 && dy == 0 && dz == 0 {
                    continue;
                }

                let nx = probe_idx[0] as i32 + dx;
                let ny = probe_idx[1] as i32 + dy;
                let nz = probe_idx[2] as i32 + dz;

                // Bounds check
                if nx < 0
                    || nx >= dimensions[0] as i32
                    || ny < 0
                    || ny >= dimensions[1] as i32
                    || nz < 0
                    || nz >= dimensions[2] as i32
                {
                    continue;
                }

                let neighbor_idx = [nx as u32, ny as u32, nz as u32];

                // Skip neighbors that also need re-seeding
                if probe_needs_reseed_static(neighbor_idx, delta, dimensions) {
                    continue;
                }

                // Distance weight
                let dist = ((dx * dx + dy * dy + dz * dz) as f32).sqrt();
                let weight = (1.0 - dist / 2.0).max(0.0);

                if weight < 0.001 {
                    continue;
                }

                // Accumulate
                let linear = neighbor_idx[0]
                    + neighbor_idx[1] * dimensions[0]
                    + neighbor_idx[2] * dimensions[0] * dimensions[1];
                let neighbor = &probes[linear as usize];

                for i in 0..9 {
                    for j in 0..4 {
                        accum_irradiance[i][j] += neighbor.irradiance.coeffs[i][j] * weight;
                    }
                    accum_visibility[i] += neighbor.visibility[i] * weight;
                }
                total_weight += weight;
            }
        }
    }

    // Normalize
    if total_weight > 0.001 {
        let inv_weight = 1.0 / total_weight;
        for i in 0..9 {
            for j in 0..4 {
                accum_irradiance[i][j] *= inv_weight;
            }
            accum_visibility[i] *= inv_weight;
        }
    } else {
        // No valid neighbors - minimal ambient
        accum_irradiance[0] = [0.01, 0.01, 0.01, 0.0];
        accum_visibility = [1.0; 9];
    }

    ProbeSH {
        irradiance: crate::sh::SHCoefficientsL2 {
            coeffs: accum_irradiance,
        },
        visibility: accum_visibility,
        _pad: [0.0; 3],
    }
}

/// Static version of probe_needs_reseed for CPU use.
fn probe_needs_reseed_static(
    probe_idx: [u32; 3],
    delta: &ScrollDelta,
    dimensions: [u32; 3],
) -> bool {
    // X axis
    if delta.dx > 0 {
        let threshold_x = dimensions[0].saturating_sub(delta.dx as u32);
        if probe_idx[0] >= threshold_x {
            return true;
        }
    } else if delta.dx < 0 && probe_idx[0] < (-delta.dx) as u32 {
        return true;
    }

    // Y axis
    if delta.dy > 0 {
        let threshold_y = dimensions[1].saturating_sub(delta.dy as u32);
        if probe_idx[1] >= threshold_y {
            return true;
        }
    } else if delta.dy < 0 && probe_idx[1] < (-delta.dy) as u32 {
        return true;
    }

    // Z axis
    if delta.dz > 0 {
        let threshold_z = dimensions[2].saturating_sub(delta.dz as u32);
        if probe_idx[2] >= threshold_z {
            return true;
        }
    } else if delta.dz < 0 && probe_idx[2] < (-delta.dz) as u32 {
        return true;
    }

    false
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-6;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    fn create_test_grid() -> ProbeGridGpu {
        ProbeGridGpu::new([0.0; 3], [2.0, 2.0, 2.0], [8, 4, 8])
    }

    // ── ScrollDelta tests ───────────────────────────────────────────────────

    #[test]
    fn test_scroll_delta_zero() {
        let delta = ScrollDelta::ZERO;
        assert!(delta.is_zero());
    }

    #[test]
    fn test_scroll_delta_non_zero() {
        let delta = ScrollDelta::new(1, 0, 0);
        assert!(!delta.is_zero());
    }

    #[test]
    fn test_scroll_delta_affected_count_x() {
        let delta = ScrollDelta::new(2, 0, 0);
        let dims = [8, 4, 8];
        // X slab: 2 * 4 * 8 = 64
        assert_eq!(delta.affected_probe_count(dims), 64);
    }

    #[test]
    fn test_scroll_delta_affected_count_combined() {
        let delta = ScrollDelta::new(1, 1, 1);
        let dims = [8, 4, 8];
        // X: 1*4*8=32, Y: 8*1*8=64, Z: 8*4*1=32 = 128
        assert_eq!(delta.affected_probe_count(dims), 128);
    }

    #[test]
    fn test_scroll_delta_magnitude() {
        let delta = ScrollDelta::new(-3, 2, -5);
        assert_eq!(delta.magnitude(), 5);
    }

    // ── GridShiftParams tests ───────────────────────────────────────────────

    #[test]
    fn test_grid_shift_params_size() {
        assert_eq!(std::mem::size_of::<GridShiftParams>(), 64);
    }

    #[test]
    fn test_grid_shift_params_pod() {
        let params = GridShiftParams::default();
        let bytes: &[u8] = bytemuck::bytes_of(&params);
        assert_eq!(bytes.len(), 64);
    }

    #[test]
    fn test_grid_shift_params_dispatch_size() {
        let params = GridShiftParams::new([0; 3], [1, 0, 0], [32, 16, 8], 0, 2.0);
        let (wg_x, wg_y, wg_z) = params.dispatch_size();
        assert_eq!(wg_x, 4); // 32/8
        assert_eq!(wg_y, 2); // 16/8
        assert_eq!(wg_z, 8);
    }

    // ── ProbeStatus tests ───────────────────────────────────────────────────

    #[test]
    fn test_probe_status_size() {
        assert_eq!(std::mem::size_of::<ProbeStatus>(), 16);
    }

    #[test]
    fn test_probe_status_fresh() {
        let status = ProbeStatus::fresh();
        assert!(status.is_reliable());
        assert_eq!(status.needs_reseed, 0);
    }

    #[test]
    fn test_probe_status_stale() {
        let status = ProbeStatus::stale();
        assert!(!status.is_reliable());
        assert_eq!(status.needs_reseed, 1);
    }

    // ── DDGIScrollManager creation tests ────────────────────────────────────

    #[test]
    fn test_scroll_manager_new() {
        let manager = DDGIScrollManager::new(1.0);
        assert!(approx_eq(manager.shift_threshold(), 1.0));
        assert_eq!(manager.current_offset(), [0, 0, 0]);
    }

    #[test]
    #[should_panic(expected = "Shift threshold must be positive")]
    fn test_scroll_manager_invalid_threshold() {
        DDGIScrollManager::new(0.0);
    }

    #[test]
    fn test_scroll_manager_default() {
        let manager = DDGIScrollManager::default();
        assert!(approx_eq(manager.shift_threshold(), 1.0));
    }

    // ── DDGIScrollManager update tests ──────────────────────────────────────

    #[test]
    fn test_scroll_manager_first_update_returns_none() {
        let mut manager = DDGIScrollManager::new(1.0);
        let grid = create_test_grid();
        let result = manager.update([0.0, 0.0, 0.0], &grid);
        assert!(result.is_none());
    }

    #[test]
    fn test_scroll_manager_no_movement() {
        let mut manager = DDGIScrollManager::new(1.0);
        let grid = create_test_grid();

        manager.update([0.0, 0.0, 0.0], &grid); // Initialize
        let result = manager.update([0.0, 0.0, 0.0], &grid);

        assert!(result.is_none());
        assert_eq!(manager.current_offset(), [0, 0, 0]);
    }

    #[test]
    fn test_scroll_manager_small_movement() {
        let mut manager = DDGIScrollManager::new(1.0);
        let grid = create_test_grid();

        manager.update([0.0, 0.0, 0.0], &grid);
        let result = manager.update([0.5, 0.0, 0.0], &grid); // Less than threshold

        assert!(result.is_none());
        assert_eq!(manager.current_offset(), [0, 0, 0]);
    }

    #[test]
    fn test_scroll_manager_triggers_scroll_x() {
        let mut manager = DDGIScrollManager::new(1.0);
        let grid = create_test_grid();

        manager.update([0.0, 0.0, 0.0], &grid);
        let result = manager.update([2.5, 0.0, 0.0], &grid); // More than cell size

        assert!(result.is_some());
        let delta = result.unwrap();
        assert_eq!(delta.dx, 1);
        assert_eq!(delta.dy, 0);
        assert_eq!(delta.dz, 0);
    }

    #[test]
    fn test_scroll_manager_triggers_scroll_negative() {
        let mut manager = DDGIScrollManager::new(1.0);
        let grid = create_test_grid();

        manager.update([0.0, 0.0, 0.0], &grid);
        let result = manager.update([-2.5, 0.0, 0.0], &grid);

        assert!(result.is_some());
        let delta = result.unwrap();
        assert_eq!(delta.dx, -1);
    }

    #[test]
    fn test_scroll_manager_accumulates_offset() {
        let mut manager = DDGIScrollManager::new(1.0);
        let grid = create_test_grid();

        manager.update([0.0, 0.0, 0.0], &grid);
        manager.update([2.5, 0.0, 0.0], &grid); // +1
        manager.update([4.5, 0.0, 0.0], &grid); // +1

        assert_eq!(manager.current_offset()[0], 2);
    }

    #[test]
    fn test_scroll_manager_multi_axis() {
        let mut manager = DDGIScrollManager::new(1.0);
        let grid = create_test_grid();

        manager.update([0.0, 0.0, 0.0], &grid);
        let result = manager.update([2.5, 2.5, 2.5], &grid);

        assert!(result.is_some());
        let delta = result.unwrap();
        assert_eq!(delta.dx, 1);
        assert_eq!(delta.dy, 1);
        assert_eq!(delta.dz, 1);
    }

    // ── DDGIScrollManager hysteresis tests ──────────────────────────────────

    #[test]
    fn test_scroll_manager_sustained_hysteresis() {
        let mut manager = DDGIScrollManager::new(1.0)
            .with_hysteresis(ScrollHysteresis::Sustained, 3);
        let grid = create_test_grid();

        manager.update([0.0, 0.0, 0.0], &grid);

        // First scroll triggers pendng
        let r1 = manager.update([2.5, 0.0, 0.0], &grid);
        assert!(r1.is_none()); // Not yet

        let r2 = manager.update([4.5, 0.0, 0.0], &grid);
        assert!(r2.is_none()); // Still not

        let r3 = manager.update([6.5, 0.0, 0.0], &grid);
        assert!(r3.is_some()); // Now triggers
    }

    // ── probe_needs_reseed tests ────────────────────────────────────────────

    #[test]
    fn test_probe_needs_reseed_positive_x() {
        let manager = DDGIScrollManager::new(1.0);
        let delta = ScrollDelta::new(2, 0, 0);
        let dims = [8, 4, 8];

        // Probes at x=6,7 should need reseed (8-2=6)
        assert!(manager.probe_needs_reseed([6, 0, 0], &delta, dims));
        assert!(manager.probe_needs_reseed([7, 0, 0], &delta, dims));
        assert!(!manager.probe_needs_reseed([5, 0, 0], &delta, dims));
    }

    #[test]
    fn test_probe_needs_reseed_negative_x() {
        let manager = DDGIScrollManager::new(1.0);
        let delta = ScrollDelta::new(-2, 0, 0);
        let dims = [8, 4, 8];

        // Probes at x=0,1 should need reseed
        assert!(manager.probe_needs_reseed([0, 0, 0], &delta, dims));
        assert!(manager.probe_needs_reseed([1, 0, 0], &delta, dims));
        assert!(!manager.probe_needs_reseed([2, 0, 0], &delta, dims));
    }

    #[test]
    fn test_probe_needs_reseed_no_scroll() {
        let manager = DDGIScrollManager::new(1.0);
        let delta = ScrollDelta::ZERO;
        let dims = [8, 4, 8];

        // No probes should need reseed
        assert!(!manager.probe_needs_reseed([0, 0, 0], &delta, dims));
        assert!(!manager.probe_needs_reseed([7, 3, 7], &delta, dims));
    }

    #[test]
    fn test_probe_needs_reseed_multi_axis() {
        let manager = DDGIScrollManager::new(1.0);
        let delta = ScrollDelta::new(1, 1, 0);
        let dims = [8, 4, 8];

        // Should trigger on either x or y boundary
        assert!(manager.probe_needs_reseed([7, 0, 0], &delta, dims)); // X boundary
        assert!(manager.probe_needs_reseed([0, 3, 0], &delta, dims)); // Y boundary
        assert!(!manager.probe_needs_reseed([0, 0, 0], &delta, dims)); // Neither
    }

    // ── get_probes_to_reseed tests ──────────────────────────────────────────

    #[test]
    fn test_get_probes_to_reseed_count() {
        let manager = DDGIScrollManager::new(1.0);
        let grid = create_test_grid();
        let delta = ScrollDelta::new(1, 0, 0);

        let probes = manager.get_probes_to_reseed(&delta, &grid);

        // X slab with delta=1: 1 * 4 * 8 = 32 probes
        assert_eq!(probes.len(), 32);
    }

    #[test]
    fn test_get_probes_to_reseed_empty() {
        let manager = DDGIScrollManager::new(1.0);
        let grid = create_test_grid();
        let delta = ScrollDelta::ZERO;

        let probes = manager.get_probes_to_reseed(&delta, &grid);

        assert!(probes.is_empty());
    }

    // ── create_shift_params tests ───────────────────────────────────────────

    #[test]
    fn test_create_shift_params() {
        let mut manager = DDGIScrollManager::new(1.0);
        let grid = create_test_grid();

        manager.update([0.0, 0.0, 0.0], &grid);
        let delta_result = manager.update([2.5, 0.0, 0.0], &grid);
        let delta = delta_result.unwrap();

        let params = manager.create_shift_params(delta, &grid);

        assert_eq!(params.dimensions, [8, 4, 8]);
        assert_eq!(params.new_scroll_offset, manager.current_offset());
        assert!(approx_eq(params.cell_spacing, 2.0));
    }

    // ── Reset tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_scroll_manager_reset() {
        let mut manager = DDGIScrollManager::new(1.0);
        let grid = create_test_grid();

        manager.update([0.0, 0.0, 0.0], &grid);
        manager.update([5.0, 5.0, 5.0], &grid);

        assert_ne!(manager.current_offset(), [0, 0, 0]);

        manager.reset();

        assert_eq!(manager.current_offset(), [0, 0, 0]);
        assert!(!manager.initialized);
    }

    // ── Set offset tests ────────────────────────────────────────────────────

    #[test]
    fn test_scroll_manager_set_offset() {
        let mut manager = DDGIScrollManager::new(1.0);

        manager.set_offset([10, 20, 30]);

        assert_eq!(manager.current_offset(), [10, 20, 30]);
    }
}
