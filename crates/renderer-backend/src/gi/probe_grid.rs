//! GPU storage structures for DDGI probe grids.
//!
//! This module defines the GPU-uploadable structures for probe-based global
//! illumination. These structures are designed to match WGSL shader layouts
//! with proper alignment and padding for efficient GPU access.
//!
//! # Memory Layout
//!
//! All structures are `repr(C)` and derive `Pod`/`Zeroable` for safe
//! GPU buffer uploads via bytemuck.
//!
//! | Struct | Size | Alignment |
//! |--------|------|-----------|
//! | `ProbeGridGpu` | 64 bytes | 16 bytes |
//! | `ProbeSH` | 192 bytes | 16 bytes |
//! | `ProbeVis` | 16 bytes | 4 bytes |
//!
//! # Coordinate Systems
//!
//! The probe grid uses a right-handed coordinate system with Y-up:
//! - World position to grid index: `(world - origin) / cell_size`
//! - Grid index to linear index: `x + y * dim.x + z * dim.x * dim.y`
//!
//! # Infinite Scrolling
//!
//! The `scroll_offset` field enables infinite scrolling volumes where probes
//! wrap around the grid as the camera moves. The `ProbeRingBuffer` provides
//! a CPU-side implementation for managing scrolled probe data.

use crate::sh::SHCoefficientsL2;
use bytemuck::{Pod, Zeroable};

// ============================================================================
// GPU Structures
// ============================================================================

/// GPU probe grid metadata.
///
/// This structure is uploaded to a uniform buffer and accessed by all probe
/// update and sampling shaders. Size: 64 bytes.
///
/// # WGSL Layout
///
/// ```wgsl
/// struct ProbeGridGpu {
///     origin: vec3<f32>,
///     _pad0: f32,
///     cell_size: vec3<f32>,
///     _pad1: f32,
///     dimensions: vec3<u32>,
///     total_probes: u32,
///     scroll_offset: vec3<i32>,
///     frame_index: u32,
/// }
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct ProbeGridGpu {
    /// World-space origin (minimum corner) of the probe volume.
    pub origin: [f32; 3],
    /// Padding for vec4 alignment.
    pub _pad0: f32,
    /// Cell size (spacing) between adjacent probes in world units.
    pub cell_size: [f32; 3],
    /// Padding for vec4 alignment.
    pub _pad1: f32,
    /// Number of probes along each axis (X, Y, Z).
    pub dimensions: [u32; 3],
    /// Total number of probes in the volume (cached for convenience).
    pub total_probes: u32,
    /// Scroll offset for infinite scrolling volumes (in probe grid units).
    pub scroll_offset: [i32; 3],
    /// Current frame index for temporal updates.
    pub frame_index: u32,
}

impl Default for ProbeGridGpu {
    fn default() -> Self {
        Self {
            origin: [0.0, 0.0, 0.0],
            _pad0: 0.0,
            cell_size: [2.0, 2.0, 2.0],
            _pad1: 0.0,
            dimensions: [8, 4, 8],
            total_probes: 8 * 4 * 8,
            scroll_offset: [0, 0, 0],
            frame_index: 0,
        }
    }
}

impl ProbeGridGpu {
    /// Create a new probe grid with the given parameters.
    pub fn new(
        origin: [f32; 3],
        cell_size: [f32; 3],
        dimensions: [u32; 3],
    ) -> Self {
        Self {
            origin,
            _pad0: 0.0,
            cell_size,
            _pad1: 0.0,
            dimensions,
            total_probes: dimensions[0] * dimensions[1] * dimensions[2],
            scroll_offset: [0, 0, 0],
            frame_index: 0,
        }
    }

    /// Calculate the total number of probes from dimensions.
    #[inline]
    pub fn calculate_total_probes(&self) -> u32 {
        self.dimensions[0] * self.dimensions[1] * self.dimensions[2]
    }

    /// Convert world position to grid coordinates (floating-point).
    #[inline]
    pub fn world_to_grid(&self, world: [f32; 3]) -> [f32; 3] {
        [
            (world[0] - self.origin[0]) / self.cell_size[0],
            (world[1] - self.origin[1]) / self.cell_size[1],
            (world[2] - self.origin[2]) / self.cell_size[2],
        ]
    }

    /// Convert grid coordinates to world position.
    #[inline]
    pub fn grid_to_world(&self, grid: [u32; 3]) -> [f32; 3] {
        [
            self.origin[0] + (grid[0] as f32) * self.cell_size[0],
            self.origin[1] + (grid[1] as f32) * self.cell_size[1],
            self.origin[2] + (grid[2] as f32) * self.cell_size[2],
        ]
    }

    /// Convert 3D grid index to linear buffer index.
    #[inline]
    pub fn grid_to_linear(&self, grid: [u32; 3]) -> u32 {
        grid[0] + grid[1] * self.dimensions[0] + grid[2] * self.dimensions[0] * self.dimensions[1]
    }

    /// Convert linear buffer index to 3D grid index.
    #[inline]
    pub fn linear_to_grid(&self, linear: u32) -> [u32; 3] {
        let z = linear / (self.dimensions[0] * self.dimensions[1]);
        let rem = linear % (self.dimensions[0] * self.dimensions[1]);
        let y = rem / self.dimensions[0];
        let x = rem % self.dimensions[0];
        [x, y, z]
    }

    /// Apply scroll offset to grid coordinates (wrapping).
    #[inline]
    pub fn apply_scroll(&self, grid: [u32; 3]) -> [u32; 3] {
        let wrap = |val: u32, offset: i32, dim: u32| -> u32 {
            let signed = (val as i32) + offset;
            let modded = signed.rem_euclid(dim as i32);
            modded as u32
        };
        [
            wrap(grid[0], self.scroll_offset[0], self.dimensions[0]),
            wrap(grid[1], self.scroll_offset[1], self.dimensions[1]),
            wrap(grid[2], self.scroll_offset[2], self.dimensions[2]),
        ]
    }

    /// Update the scroll offset when the camera moves.
    pub fn update_scroll(&mut self, camera_pos: [f32; 3], prev_camera_pos: [f32; 3]) {
        let delta = [
            camera_pos[0] - prev_camera_pos[0],
            camera_pos[1] - prev_camera_pos[1],
            camera_pos[2] - prev_camera_pos[2],
        ];

        // Convert delta to grid cells
        let cell_delta = [
            (delta[0] / self.cell_size[0]).floor() as i32,
            (delta[1] / self.cell_size[1]).floor() as i32,
            (delta[2] / self.cell_size[2]).floor() as i32,
        ];

        self.scroll_offset[0] += cell_delta[0];
        self.scroll_offset[1] += cell_delta[1];
        self.scroll_offset[2] += cell_delta[2];
    }

    /// Advance frame index (for temporal updates).
    #[inline]
    pub fn advance_frame(&mut self) {
        self.frame_index = self.frame_index.wrapping_add(1);
    }
}

// ============================================================================
// Per-Probe Storage
// ============================================================================

/// Per-probe SH irradiance storage.
///
/// Stores L2 (9-coefficient) spherical harmonics for irradiance plus
/// L2 visibility coefficients for soft shadowing. Size: 192 bytes.
///
/// # Memory Layout
///
/// - `irradiance`: 144 bytes (9 RGB coefficients as vec4)
/// - `visibility`: 36 bytes (9 f32 coefficients)
/// - `_pad`: 12 bytes (align to 192)
///
/// # WGSL Layout
///
/// ```wgsl
/// struct ProbeSH {
///     irradiance: array<vec4<f32>, 9>,  // 144 bytes
///     visibility: array<f32, 9>,         // 36 bytes
///     _pad: array<f32, 3>,               // 12 bytes
/// }
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct ProbeSH {
    /// L2 spherical harmonics irradiance (RGB, 9 coefficients).
    pub irradiance: SHCoefficientsL2,
    /// L2 visibility coefficients for soft shadowing.
    pub visibility: [f32; 9],
    /// Padding to align to 192 bytes.
    pub _pad: [f32; 3],
}

impl Default for ProbeSH {
    fn default() -> Self {
        Self::ZERO
    }
}

impl ProbeSH {
    /// Zero-initialized probe SH.
    pub const ZERO: Self = Self {
        irradiance: SHCoefficientsL2::ZERO,
        visibility: [0.0; 9],
        _pad: [0.0; 3],
    };

    /// Create a new ProbeSH with the given irradiance.
    pub fn with_irradiance(irradiance: SHCoefficientsL2) -> Self {
        Self {
            irradiance,
            visibility: [1.0; 9], // Default full visibility
            _pad: [0.0; 3],
        }
    }

    /// Blend with another probe using linear interpolation.
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        let inv_t = 1.0 - t;
        let mut result = Self::ZERO;

        result.irradiance = self.irradiance.lerp(&other.irradiance, t);

        for i in 0..9 {
            result.visibility[i] = self.visibility[i] * inv_t + other.visibility[i] * t;
        }

        result
    }

    /// Reset probe to zero state (for invalidation).
    pub fn reset(&mut self) {
        *self = Self::ZERO;
    }
}

// ============================================================================
// Visibility/Occlusion Data
// ============================================================================

/// Per-probe occlusion/visibility data.
///
/// Stores statistical information about ray hit distances for each probe,
/// used for probe validity testing and soft shadowing. Size: 16 bytes.
///
/// # WGSL Layout
///
/// ```wgsl
/// struct ProbeVis {
///     mean_distance: f32,
///     variance: f32,
///     confidence: f32,
///     update_frame: u32,
/// }
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct ProbeVis {
    /// Mean hit distance from probe rays.
    pub mean_distance: f32,
    /// Variance of hit distances.
    pub variance: f32,
    /// Confidence in probe data (0.0 = unreliable, 1.0 = fully converged).
    pub confidence: f32,
    /// Frame when this probe was last updated.
    pub update_frame: u32,
}

impl Default for ProbeVis {
    fn default() -> Self {
        Self {
            mean_distance: f32::MAX,
            variance: 0.0,
            confidence: 0.0,
            update_frame: 0,
        }
    }
}

impl ProbeVis {
    /// Create with initial values after first ray trace.
    pub fn new(mean_distance: f32, variance: f32, frame: u32) -> Self {
        Self {
            mean_distance,
            variance,
            confidence: 0.1, // Low initial confidence
            update_frame: frame,
        }
    }

    /// Update running statistics with Welford's online algorithm.
    pub fn update(&mut self, new_distance: f32, frame: u32) {
        // Exponential moving average for mean
        let alpha = 0.1;
        let delta = new_distance - self.mean_distance;
        self.mean_distance += alpha * delta;

        // Update variance estimate
        let delta2 = new_distance - self.mean_distance;
        self.variance = (1.0 - alpha) * self.variance + alpha * delta * delta2;

        // Increase confidence with each update (capped at 1.0)
        self.confidence = (self.confidence + 0.05).min(1.0);
        self.update_frame = frame;
    }

    /// Check if probe needs update based on frame age.
    #[inline]
    pub fn needs_update(&self, current_frame: u32, max_age: u32) -> bool {
        current_frame.wrapping_sub(self.update_frame) > max_age
    }

    /// Check if probe data is reliable.
    #[inline]
    pub fn is_reliable(&self) -> bool {
        self.confidence > 0.5
    }

    /// Reset probe visibility (for invalidation).
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

// ============================================================================
// Ring Buffer for Scrolling Volumes
// ============================================================================

/// Ring buffer for scrolling probe volumes.
///
/// Provides a CPU-side circular buffer for managing probe data in infinite
/// scrolling volumes. When the camera moves beyond the grid bounds, old
/// probes wrap around and are re-used for new positions.
///
/// # Usage
///
/// ```ignore
/// let mut ring = ProbeRingBuffer::new(256);
/// ring.push(ProbeSH::ZERO);
/// let probe = ring.get(0);
/// ```
#[derive(Clone, Debug)]
pub struct ProbeRingBuffer {
    /// Maximum number of probes in the buffer.
    pub capacity: usize,
    /// Current write head position.
    pub head: usize,
    /// Number of valid probes in the buffer.
    pub len: usize,
    /// Probe data storage.
    pub probes: Vec<ProbeSH>,
}

impl ProbeRingBuffer {
    /// Create a new ring buffer with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            head: 0,
            len: 0,
            probes: vec![ProbeSH::ZERO; capacity],
        }
    }

    /// Push a new probe, wrapping around if at capacity.
    pub fn push(&mut self, probe: ProbeSH) {
        self.probes[self.head] = probe;
        self.head = (self.head + 1) % self.capacity;
        if self.len < self.capacity {
            self.len += 1;
        }
    }

    /// Get probe at index (relative to head).
    #[inline]
    pub fn get(&self, index: usize) -> Option<&ProbeSH> {
        if index >= self.len {
            return None;
        }
        let actual_index = (self.head + self.capacity - self.len + index) % self.capacity;
        Some(&self.probes[actual_index])
    }

    /// Get mutable probe at index.
    #[inline]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut ProbeSH> {
        if index >= self.len {
            return None;
        }
        let actual_index = (self.head + self.capacity - self.len + index) % self.capacity;
        Some(&mut self.probes[actual_index])
    }

    /// Check if buffer is full.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.len == self.capacity
    }

    /// Check if buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Clear all probes.
    pub fn clear(&mut self) {
        self.head = 0;
        self.len = 0;
        for probe in &mut self.probes {
            *probe = ProbeSH::ZERO;
        }
    }

    /// Get the current length.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Get raw slice for GPU upload.
    pub fn as_slice(&self) -> &[ProbeSH] {
        &self.probes[..self.len.min(self.capacity)]
    }

    /// Invalidate probes in a region (for scrolling).
    pub fn invalidate_region(&mut self, start: usize, count: usize) {
        for i in 0..count {
            let idx = (start + i) % self.capacity;
            if idx < self.probes.len() {
                self.probes[idx].reset();
            }
        }
    }
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

    // ── Struct size tests ───────────────────────────────────────────────────

    #[test]
    fn test_probe_grid_gpu_size() {
        assert_eq!(std::mem::size_of::<ProbeGridGpu>(), 64);
    }

    #[test]
    fn test_probe_sh_size() {
        assert_eq!(std::mem::size_of::<ProbeSH>(), 192);
    }

    #[test]
    fn test_probe_vis_size() {
        assert_eq!(std::mem::size_of::<ProbeVis>(), 16);
    }

    #[test]
    fn test_probe_grid_gpu_alignment() {
        assert_eq!(std::mem::align_of::<ProbeGridGpu>(), 4);
    }

    #[test]
    fn test_probe_sh_alignment() {
        assert_eq!(std::mem::align_of::<ProbeSH>(), 4);
    }

    #[test]
    fn test_probe_vis_alignment() {
        assert_eq!(std::mem::align_of::<ProbeVis>(), 4);
    }

    // ── Pod/Zeroable trait compliance ───────────────────────────────────────

    #[test]
    fn test_probe_grid_gpu_pod_cast() {
        let grid = ProbeGridGpu::default();
        let bytes: &[u8] = bytemuck::bytes_of(&grid);
        assert_eq!(bytes.len(), 64);
        let _restored: &ProbeGridGpu = bytemuck::from_bytes(bytes);
    }

    #[test]
    fn test_probe_sh_pod_cast() {
        let probe = ProbeSH::ZERO;
        let bytes: &[u8] = bytemuck::bytes_of(&probe);
        assert_eq!(bytes.len(), 192);
        let _restored: &ProbeSH = bytemuck::from_bytes(bytes);
    }

    #[test]
    fn test_probe_vis_pod_cast() {
        let vis = ProbeVis::default();
        let bytes: &[u8] = bytemuck::bytes_of(&vis);
        assert_eq!(bytes.len(), 16);
        let _restored: &ProbeVis = bytemuck::from_bytes(bytes);
    }

    #[test]
    fn test_probe_grid_gpu_zeroable() {
        let zero: ProbeGridGpu = bytemuck::Zeroable::zeroed();
        assert_eq!(zero.origin, [0.0, 0.0, 0.0]);
        assert_eq!(zero.dimensions, [0, 0, 0]);
        assert_eq!(zero.total_probes, 0);
    }

    #[test]
    fn test_probe_sh_zeroable() {
        let zero: ProbeSH = bytemuck::Zeroable::zeroed();
        for i in 0..9 {
            assert_eq!(zero.visibility[i], 0.0);
        }
    }

    #[test]
    fn test_probe_vis_zeroable() {
        let zero: ProbeVis = bytemuck::Zeroable::zeroed();
        assert_eq!(zero.mean_distance, 0.0);
        assert_eq!(zero.variance, 0.0);
        assert_eq!(zero.confidence, 0.0);
        assert_eq!(zero.update_frame, 0);
    }

    // ── ProbeGridGpu tests ──────────────────────────────────────────────────

    #[test]
    fn test_probe_grid_gpu_default() {
        let grid = ProbeGridGpu::default();
        assert_eq!(grid.dimensions, [8, 4, 8]);
        assert_eq!(grid.total_probes, 256);
    }

    #[test]
    fn test_probe_grid_gpu_new() {
        let grid = ProbeGridGpu::new(
            [1.0, 2.0, 3.0],
            [0.5, 0.5, 0.5],
            [4, 4, 4],
        );
        assert_eq!(grid.origin, [1.0, 2.0, 3.0]);
        assert_eq!(grid.cell_size, [0.5, 0.5, 0.5]);
        assert_eq!(grid.total_probes, 64);
    }

    #[test]
    fn test_probe_grid_gpu_calculate_total_probes() {
        let grid = ProbeGridGpu::new([0.0; 3], [1.0; 3], [10, 10, 10]);
        assert_eq!(grid.calculate_total_probes(), 1000);
    }

    #[test]
    fn test_world_to_grid_origin() {
        let grid = ProbeGridGpu::default();
        let world = grid.origin;
        let grid_pos = grid.world_to_grid(world);
        assert!(approx_eq(grid_pos[0], 0.0));
        assert!(approx_eq(grid_pos[1], 0.0));
        assert!(approx_eq(grid_pos[2], 0.0));
    }

    #[test]
    fn test_world_to_grid_offset() {
        let grid = ProbeGridGpu::new([0.0, 0.0, 0.0], [2.0, 2.0, 2.0], [8, 4, 8]);
        let world = [4.0, 2.0, 6.0];
        let grid_pos = grid.world_to_grid(world);
        assert!(approx_eq(grid_pos[0], 2.0));
        assert!(approx_eq(grid_pos[1], 1.0));
        assert!(approx_eq(grid_pos[2], 3.0));
    }

    #[test]
    fn test_grid_to_world() {
        let grid = ProbeGridGpu::new([1.0, 2.0, 3.0], [0.5, 0.5, 0.5], [8, 4, 8]);
        let grid_idx = [2, 4, 6];
        let world = grid.grid_to_world(grid_idx);
        assert!(approx_eq(world[0], 2.0));
        assert!(approx_eq(world[1], 4.0));
        assert!(approx_eq(world[2], 6.0));
    }

    #[test]
    fn test_grid_to_linear_origin() {
        let grid = ProbeGridGpu::default();
        assert_eq!(grid.grid_to_linear([0, 0, 0]), 0);
    }

    #[test]
    fn test_grid_to_linear_x_axis() {
        let grid = ProbeGridGpu::default();
        assert_eq!(grid.grid_to_linear([1, 0, 0]), 1);
        assert_eq!(grid.grid_to_linear([7, 0, 0]), 7);
    }

    #[test]
    fn test_grid_to_linear_y_axis() {
        let grid = ProbeGridGpu::default();
        // y=1 should be offset by dimensions.x = 8
        assert_eq!(grid.grid_to_linear([0, 1, 0]), 8);
        assert_eq!(grid.grid_to_linear([0, 3, 0]), 24);
    }

    #[test]
    fn test_grid_to_linear_z_axis() {
        let grid = ProbeGridGpu::default();
        // z=1 should be offset by dimensions.x * dimensions.y = 8 * 4 = 32
        assert_eq!(grid.grid_to_linear([0, 0, 1]), 32);
        assert_eq!(grid.grid_to_linear([0, 0, 7]), 224);
    }

    #[test]
    fn test_grid_to_linear_combined() {
        let grid = ProbeGridGpu::default();
        // [3, 2, 1] = 3 + 2*8 + 1*32 = 3 + 16 + 32 = 51
        assert_eq!(grid.grid_to_linear([3, 2, 1]), 51);
    }

    #[test]
    fn test_linear_to_grid_roundtrip() {
        let grid = ProbeGridGpu::default();
        for z in 0..grid.dimensions[2] {
            for y in 0..grid.dimensions[1] {
                for x in 0..grid.dimensions[0] {
                    let linear = grid.grid_to_linear([x, y, z]);
                    let back = grid.linear_to_grid(linear);
                    assert_eq!(back, [x, y, z]);
                }
            }
        }
    }

    #[test]
    fn test_apply_scroll_no_offset() {
        let grid = ProbeGridGpu::default();
        let scrolled = grid.apply_scroll([3, 2, 1]);
        assert_eq!(scrolled, [3, 2, 1]);
    }

    #[test]
    fn test_apply_scroll_positive() {
        let mut grid = ProbeGridGpu::default();
        grid.scroll_offset = [2, 1, 3];
        let scrolled = grid.apply_scroll([0, 0, 0]);
        assert_eq!(scrolled, [2, 1, 3]);
    }

    #[test]
    fn test_apply_scroll_wrap() {
        let mut grid = ProbeGridGpu::default();
        grid.scroll_offset = [10, 5, 10]; // Wraps around
        let scrolled = grid.apply_scroll([0, 0, 0]);
        assert_eq!(scrolled, [2, 1, 2]); // 10 % 8 = 2, 5 % 4 = 1, 10 % 8 = 2
    }

    #[test]
    fn test_apply_scroll_negative() {
        let mut grid = ProbeGridGpu::default();
        grid.scroll_offset = [-1, -1, -1];
        let scrolled = grid.apply_scroll([0, 0, 0]);
        assert_eq!(scrolled, [7, 3, 7]); // -1 mod 8 = 7, -1 mod 4 = 3
    }

    #[test]
    fn test_update_scroll() {
        let mut grid = ProbeGridGpu::new([0.0; 3], [2.0; 3], [8, 4, 8]);
        grid.update_scroll([4.0, 2.0, 6.0], [0.0, 0.0, 0.0]);
        assert_eq!(grid.scroll_offset, [2, 1, 3]);
    }

    #[test]
    fn test_advance_frame() {
        let mut grid = ProbeGridGpu::default();
        assert_eq!(grid.frame_index, 0);
        grid.advance_frame();
        assert_eq!(grid.frame_index, 1);
        grid.advance_frame();
        assert_eq!(grid.frame_index, 2);
    }

    #[test]
    fn test_advance_frame_wraps() {
        let mut grid = ProbeGridGpu::default();
        grid.frame_index = u32::MAX;
        grid.advance_frame();
        assert_eq!(grid.frame_index, 0);
    }

    // ── ProbeSH tests ───────────────────────────────────────────────────────

    #[test]
    fn test_probe_sh_zero() {
        let probe = ProbeSH::ZERO;
        for i in 0..9 {
            assert_eq!(probe.visibility[i], 0.0);
        }
    }

    #[test]
    fn test_probe_sh_with_irradiance() {
        let irr = SHCoefficientsL2::new([[1.0; 3]; 9]);
        let probe = ProbeSH::with_irradiance(irr);
        for i in 0..9 {
            assert_eq!(probe.visibility[i], 1.0); // Default full visibility
        }
    }

    #[test]
    fn test_probe_sh_lerp() {
        let a = ProbeSH::ZERO;
        let mut b = ProbeSH::ZERO;
        b.visibility = [1.0; 9];

        let mid = a.lerp(&b, 0.5);
        for i in 0..9 {
            assert!(approx_eq(mid.visibility[i], 0.5));
        }
    }

    #[test]
    fn test_probe_sh_reset() {
        let mut probe = ProbeSH::with_irradiance(SHCoefficientsL2::new([[1.0; 3]; 9]));
        probe.reset();
        assert_eq!(probe, ProbeSH::ZERO);
    }

    // ── ProbeVis tests ──────────────────────────────────────────────────────

    #[test]
    fn test_probe_vis_default() {
        let vis = ProbeVis::default();
        assert_eq!(vis.mean_distance, f32::MAX);
        assert_eq!(vis.variance, 0.0);
        assert_eq!(vis.confidence, 0.0);
    }

    #[test]
    fn test_probe_vis_new() {
        let vis = ProbeVis::new(5.0, 1.0, 100);
        assert_eq!(vis.mean_distance, 5.0);
        assert_eq!(vis.variance, 1.0);
        assert_eq!(vis.update_frame, 100);
    }

    #[test]
    fn test_probe_vis_update() {
        let mut vis = ProbeVis::new(10.0, 0.0, 0);
        vis.update(12.0, 1);
        // Mean should move towards 12
        assert!(vis.mean_distance > 10.0);
        assert!(vis.mean_distance < 12.0);
        assert_eq!(vis.update_frame, 1);
    }

    #[test]
    fn test_probe_vis_needs_update() {
        let vis = ProbeVis::new(5.0, 1.0, 10);
        assert!(!vis.needs_update(10, 5)); // Same frame
        assert!(!vis.needs_update(15, 5)); // Within max age
        assert!(vis.needs_update(16, 5)); // Beyond max age
    }

    #[test]
    fn test_probe_vis_is_reliable() {
        let mut vis = ProbeVis::default();
        assert!(!vis.is_reliable()); // 0.0 confidence

        vis.confidence = 0.5;
        assert!(!vis.is_reliable()); // Exactly 0.5

        vis.confidence = 0.51;
        assert!(vis.is_reliable()); // Above 0.5
    }

    #[test]
    fn test_probe_vis_reset() {
        let mut vis = ProbeVis::new(5.0, 1.0, 100);
        vis.reset();
        assert_eq!(vis, ProbeVis::default());
    }

    // ── ProbeRingBuffer tests ───────────────────────────────────────────────

    #[test]
    fn test_ring_buffer_new() {
        let ring = ProbeRingBuffer::new(16);
        assert_eq!(ring.capacity, 16);
        assert_eq!(ring.len(), 0);
        assert!(ring.is_empty());
    }

    #[test]
    fn test_ring_buffer_push() {
        let mut ring = ProbeRingBuffer::new(4);
        ring.push(ProbeSH::ZERO);
        assert_eq!(ring.len(), 1);
        assert!(!ring.is_empty());
    }

    #[test]
    fn test_ring_buffer_push_multiple() {
        let mut ring = ProbeRingBuffer::new(4);
        for _ in 0..3 {
            ring.push(ProbeSH::ZERO);
        }
        assert_eq!(ring.len(), 3);
    }

    #[test]
    fn test_ring_buffer_is_full() {
        let mut ring = ProbeRingBuffer::new(4);
        assert!(!ring.is_full());
        for _ in 0..4 {
            ring.push(ProbeSH::ZERO);
        }
        assert!(ring.is_full());
    }

    #[test]
    fn test_ring_buffer_wrap() {
        let mut ring = ProbeRingBuffer::new(4);
        for _ in 0..6 {
            ring.push(ProbeSH::ZERO);
        }
        assert_eq!(ring.len(), 4); // Capped at capacity
        assert!(ring.is_full());
    }

    #[test]
    fn test_ring_buffer_get() {
        let mut ring = ProbeRingBuffer::new(4);
        let mut probe = ProbeSH::ZERO;
        probe.visibility[0] = 1.0;
        ring.push(probe);

        let retrieved = ring.get(0);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().visibility[0], 1.0);
    }

    #[test]
    fn test_ring_buffer_get_out_of_bounds() {
        let ring = ProbeRingBuffer::new(4);
        assert!(ring.get(0).is_none());
    }

    #[test]
    fn test_ring_buffer_get_mut() {
        let mut ring = ProbeRingBuffer::new(4);
        ring.push(ProbeSH::ZERO);

        if let Some(probe) = ring.get_mut(0) {
            probe.visibility[0] = 2.0;
        }

        assert_eq!(ring.get(0).unwrap().visibility[0], 2.0);
    }

    #[test]
    fn test_ring_buffer_clear() {
        let mut ring = ProbeRingBuffer::new(4);
        ring.push(ProbeSH::ZERO);
        ring.push(ProbeSH::ZERO);
        ring.clear();
        assert!(ring.is_empty());
        assert_eq!(ring.len(), 0);
    }

    #[test]
    fn test_ring_buffer_as_slice() {
        let mut ring = ProbeRingBuffer::new(4);
        ring.push(ProbeSH::ZERO);
        ring.push(ProbeSH::ZERO);
        let slice = ring.as_slice();
        assert_eq!(slice.len(), 2);
    }

    #[test]
    fn test_ring_buffer_invalidate_region() {
        let mut ring = ProbeRingBuffer::new(4);
        for _ in 0..4 {
            let mut probe = ProbeSH::ZERO;
            probe.visibility[0] = 1.0;
            ring.push(probe);
        }
        ring.invalidate_region(1, 2);
        assert_eq!(ring.probes[1].visibility[0], 0.0);
        assert_eq!(ring.probes[2].visibility[0], 0.0);
        assert_eq!(ring.probes[0].visibility[0], 1.0); // Not invalidated
        assert_eq!(ring.probes[3].visibility[0], 1.0); // Not invalidated
    }

    // ── Coordinate conversion integration tests ─────────────────────────────

    #[test]
    fn test_world_to_linear_roundtrip() {
        let grid = ProbeGridGpu::new([0.0; 3], [1.0; 3], [8, 4, 8]);
        let world = [3.0, 2.0, 5.0];

        // World -> grid -> linear -> grid
        let grid_pos = grid.world_to_grid(world);
        let grid_idx = [grid_pos[0] as u32, grid_pos[1] as u32, grid_pos[2] as u32];
        let linear = grid.grid_to_linear(grid_idx);
        let back = grid.linear_to_grid(linear);

        assert_eq!(back, grid_idx);
    }

    #[test]
    fn test_scroll_preserves_grid_structure() {
        let mut grid = ProbeGridGpu::default();
        grid.scroll_offset = [3, 2, 1];

        // All scrolled coordinates should remain valid
        for z in 0..grid.dimensions[2] {
            for y in 0..grid.dimensions[1] {
                for x in 0..grid.dimensions[0] {
                    let scrolled = grid.apply_scroll([x, y, z]);
                    assert!(scrolled[0] < grid.dimensions[0]);
                    assert!(scrolled[1] < grid.dimensions[1]);
                    assert!(scrolled[2] < grid.dimensions[2]);
                }
            }
        }
    }
}
