//! Cloud Temporal Reprojection (T-ENV-2.6)
//!
//! This module provides temporal reprojection for volumetric cloud rendering,
//! amortizing the expensive ray marching cost over multiple frames.
//!
//! # Overview
//!
//! Temporal reprojection reuses cloud data from previous frames by:
//! 1. Reprojecting pixel positions using view/projection matrices
//! 2. Sampling the history buffer with bilinear filtering
//! 3. Rejecting invalid samples using neighborhood clamping (YCoCg AABB)
//! 4. Blending history with current frame based on confidence
//!
//! # Features
//!
//! * **Motion Vector Calculation**: Computes parallax-aware motion vectors
//! * **History Buffer Management**: Ping-pong buffer system with jitter offsets
//! * **Neighborhood Clamping**: AABB clamping in YCoCg color space for ghosting rejection
//! * **Variance-Based Blending**: Adaptive blend weight based on sample variance
//! * **Checkerboard Rendering**: Half-resolution rendering each frame (even/odd pixels)
//!
//! # Performance
//!
//! With temporal reprojection enabled:
//! - Cloud ray marching can run at half resolution
//! - Typical blend factor: 0.9 (90% history reuse)
//! - Effective sample count increase: 4-8x over multiple frames
//!
//! # References
//!
//! - Karis, "High-Quality Temporal Supersampling" (SIGGRAPH 2014)
//! - Salvi, "An Excursion in Temporal Supersampling" (GDC 2016)
//! - Schneider & Vos, "The Real-time Volumetric Cloudscapes of Horizon: Zero Dawn"

use bytemuck::{Pod, Zeroable};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default blend factor for temporal accumulation (90% history).
pub const DEFAULT_BLEND_FACTOR: f32 = 0.9;

/// Minimum blend factor (fastest response to changes).
pub const MIN_BLEND_FACTOR: f32 = 0.0;

/// Maximum blend factor (slowest response, highest quality).
pub const MAX_BLEND_FACTOR: f32 = 0.98;

/// Default jitter sequence length (Halton 2,3).
pub const DEFAULT_JITTER_SEQUENCE_LENGTH: u32 = 8;

/// Maximum jitter sequence length.
pub const MAX_JITTER_SEQUENCE_LENGTH: u32 = 64;

/// Default variance threshold for ghosting rejection.
pub const DEFAULT_VARIANCE_THRESHOLD: f32 = 0.1;

/// Default neighborhood clamp gamma (sharpness factor).
pub const DEFAULT_CLAMP_GAMMA: f32 = 1.0;

/// Small epsilon for floating point comparisons.
pub const EPSILON: f32 = 1e-6;

/// Maximum disocclusion distance threshold (in NDC space).
pub const MAX_DISOCCLUSION_THRESHOLD: f32 = 0.1;

/// Default cloud parallax depth for motion vectors.
pub const DEFAULT_CLOUD_DEPTH: f32 = 10000.0;

// ---------------------------------------------------------------------------
// CloudTemporalUniforms - GPU-uploadable temporal state
// ---------------------------------------------------------------------------

/// GPU uniforms for cloud temporal reprojection.
///
/// This struct is designed for direct GPU upload as a uniform buffer.
/// All matrices are stored in column-major order for WGSL compatibility.
///
/// # Memory Layout (192 bytes)
///
/// | Offset | Field            | Size      |
/// |--------|------------------|-----------|
/// | 0      | prev_view_proj   | 64 bytes  |
/// | 64     | curr_view_proj   | 64 bytes  |
/// | 128    | jitter_offset    | 8 bytes   |
/// | 136    | blend_factor     | 4 bytes   |
/// | 140    | frame_index      | 4 bytes   |
/// | 144    | resolution       | 8 bytes   |
/// | 152    | cloud_depth      | 4 bytes   |
/// | 156    | variance_threshold | 4 bytes |
/// | 160    | clamp_gamma      | 4 bytes   |
/// | 164    | enable_checkerboard | 4 bytes |
/// | 168    | _padding         | 24 bytes  |
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct CloudTemporalUniforms {
    /// Previous frame's combined view-projection matrix.
    pub prev_view_proj: [[f32; 4]; 4],

    /// Current frame's combined view-projection matrix.
    pub curr_view_proj: [[f32; 4]; 4],

    /// Sub-pixel jitter offset for the current frame (x, y in pixels).
    pub jitter_offset: [f32; 2],

    /// Temporal blend factor (0 = current only, 1 = history only).
    pub blend_factor: f32,

    /// Frame index for checkerboard and jitter sequence.
    pub frame_index: u32,

    /// Render resolution (width, height).
    pub resolution: [f32; 2],

    /// Cloud depth for motion vector calculation (meters).
    pub cloud_depth: f32,

    /// Variance threshold for ghosting rejection.
    pub variance_threshold: f32,

    /// Neighborhood clamp gamma (sharpness).
    pub clamp_gamma: f32,

    /// Enable checkerboard rendering (1 = enabled, 0 = disabled).
    pub enable_checkerboard: u32,

    /// Padding for 16-byte alignment.
    pub _padding: [u32; 6],
}

// Size assertion for GPU compatibility (must be multiple of 16)
const _: () = assert!(std::mem::size_of::<CloudTemporalUniforms>() == 192);

impl Default for CloudTemporalUniforms {
    fn default() -> Self {
        Self {
            prev_view_proj: IDENTITY_MATRIX,
            curr_view_proj: IDENTITY_MATRIX,
            jitter_offset: [0.0, 0.0],
            blend_factor: DEFAULT_BLEND_FACTOR,
            frame_index: 0,
            resolution: [1920.0, 1080.0],
            cloud_depth: DEFAULT_CLOUD_DEPTH,
            variance_threshold: DEFAULT_VARIANCE_THRESHOLD,
            clamp_gamma: DEFAULT_CLAMP_GAMMA,
            enable_checkerboard: 0,
            _padding: [0; 6],
        }
    }
}

impl CloudTemporalUniforms {
    /// Create new temporal uniforms with default settings.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create temporal uniforms with specified resolution.
    #[inline]
    pub fn with_resolution(width: u32, height: u32) -> Self {
        Self {
            resolution: [width as f32, height as f32],
            ..Default::default()
        }
    }

    /// Set the previous and current view-projection matrices.
    #[inline]
    pub fn set_matrices(
        &mut self,
        prev_view_proj: [[f32; 4]; 4],
        curr_view_proj: [[f32; 4]; 4],
    ) {
        self.prev_view_proj = prev_view_proj;
        self.curr_view_proj = curr_view_proj;
    }

    /// Update for a new frame.
    #[inline]
    pub fn advance_frame(&mut self, new_view_proj: [[f32; 4]; 4]) {
        self.prev_view_proj = self.curr_view_proj;
        self.curr_view_proj = new_view_proj;
        self.frame_index = self.frame_index.wrapping_add(1);
    }

    /// Set the jitter offset for the current frame.
    #[inline]
    pub fn set_jitter(&mut self, x: f32, y: f32) {
        self.jitter_offset = [x, y];
    }

    /// Set blend factor (clamped to valid range).
    #[inline]
    pub fn set_blend_factor(&mut self, factor: f32) {
        self.blend_factor = factor.clamp(MIN_BLEND_FACTOR, MAX_BLEND_FACTOR);
    }

    /// Enable or disable checkerboard rendering.
    #[inline]
    pub fn set_checkerboard(&mut self, enabled: bool) {
        self.enable_checkerboard = if enabled { 1 } else { 0 };
    }

    /// Get whether checkerboard is enabled.
    #[inline]
    pub fn is_checkerboard_enabled(&self) -> bool {
        self.enable_checkerboard != 0
    }

    /// Validate the uniforms.
    pub fn validate(&self) -> bool {
        self.blend_factor >= MIN_BLEND_FACTOR
            && self.blend_factor <= MAX_BLEND_FACTOR
            && self.resolution[0] > 0.0
            && self.resolution[1] > 0.0
            && self.cloud_depth > 0.0
            && self.variance_threshold > 0.0
            && self.clamp_gamma > 0.0
    }
}

// ---------------------------------------------------------------------------
// HistoryBuffer - Ping-pong buffer management
// ---------------------------------------------------------------------------

/// Index for ping-pong buffer selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum BufferIndex {
    /// Buffer A (even frames).
    #[default]
    A = 0,
    /// Buffer B (odd frames).
    B = 1,
}

impl BufferIndex {
    /// Get the other buffer index.
    #[inline]
    pub fn other(&self) -> Self {
        match self {
            BufferIndex::A => BufferIndex::B,
            BufferIndex::B => BufferIndex::A,
        }
    }

    /// Get buffer index from frame number.
    #[inline]
    pub fn from_frame(frame: u32) -> Self {
        if frame % 2 == 0 {
            BufferIndex::A
        } else {
            BufferIndex::B
        }
    }
}

/// History buffer state for temporal reprojection.
///
/// Manages ping-pong buffers and tracks buffer validity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HistoryBufferState {
    /// Current buffer being written to.
    pub current: BufferIndex,

    /// Resolution of history buffers (width, height).
    pub resolution: (u32, u32),

    /// Whether the history is valid (false on first frame or resolution change).
    pub valid: bool,

    /// Frame count since last invalidation.
    pub frames_accumulated: u32,
}

impl Default for HistoryBufferState {
    fn default() -> Self {
        Self {
            current: BufferIndex::A,
            resolution: (1920, 1080),
            valid: false,
            frames_accumulated: 0,
        }
    }
}

impl HistoryBufferState {
    /// Create a new history buffer state.
    #[inline]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            resolution: (width, height),
            ..Default::default()
        }
    }

    /// Advance to the next frame.
    #[inline]
    pub fn advance(&mut self) {
        self.current = self.current.other();
        if !self.valid {
            self.valid = true;
        }
        self.frames_accumulated = self.frames_accumulated.saturating_add(1);
    }

    /// Get the history (read) buffer index.
    #[inline]
    pub fn history_index(&self) -> BufferIndex {
        self.current.other()
    }

    /// Invalidate the history (e.g., on resolution change or camera cut).
    #[inline]
    pub fn invalidate(&mut self) {
        self.valid = false;
        self.frames_accumulated = 0;
    }

    /// Check if resolution matches and optionally invalidate.
    pub fn check_resolution(&mut self, width: u32, height: u32) -> bool {
        if self.resolution != (width, height) {
            self.resolution = (width, height);
            self.invalidate();
            false
        } else {
            true
        }
    }

    /// Get the blend factor adjusted for history validity.
    #[inline]
    pub fn effective_blend_factor(&self, base_factor: f32) -> f32 {
        if !self.valid {
            0.0 // No history available
        } else if self.frames_accumulated < 4 {
            // Ramp up blend factor over first few frames
            base_factor * (self.frames_accumulated as f32 / 4.0)
        } else {
            base_factor
        }
    }
}

// ---------------------------------------------------------------------------
// Jitter Sequence - Sub-pixel jitter for temporal AA
// ---------------------------------------------------------------------------

/// Pre-computed Halton(2,3) jitter sequence.
///
/// Provides low-discrepancy sub-pixel offsets for temporal anti-aliasing.
#[derive(Debug, Clone)]
pub struct JitterSequence {
    /// Pre-computed jitter offsets (x, y pairs).
    offsets: Vec<[f32; 2]>,
    /// Current index in the sequence.
    index: usize,
}

impl Default for JitterSequence {
    fn default() -> Self {
        Self::halton(DEFAULT_JITTER_SEQUENCE_LENGTH as usize)
    }
}

impl JitterSequence {
    /// Create a Halton(2,3) jitter sequence.
    pub fn halton(length: usize) -> Self {
        let length = length.clamp(1, MAX_JITTER_SEQUENCE_LENGTH as usize);
        let mut offsets = Vec::with_capacity(length);

        for i in 0..length {
            let x = halton_sequence(i as u32 + 1, 2) - 0.5;
            let y = halton_sequence(i as u32 + 1, 3) - 0.5;
            offsets.push([x, y]);
        }

        Self { offsets, index: 0 }
    }

    /// Create a simple 2x2 grid jitter sequence.
    pub fn grid_2x2() -> Self {
        Self {
            offsets: vec![
                [-0.25, -0.25],
                [0.25, -0.25],
                [-0.25, 0.25],
                [0.25, 0.25],
            ],
            index: 0,
        }
    }

    /// Create a simple 4x4 grid jitter sequence.
    pub fn grid_4x4() -> Self {
        let mut offsets = Vec::with_capacity(16);
        for y in 0..4 {
            for x in 0..4 {
                let jx = (x as f32 + 0.5) / 4.0 - 0.5;
                let jy = (y as f32 + 0.5) / 4.0 - 0.5;
                offsets.push([jx, jy]);
            }
        }
        Self { offsets, index: 0 }
    }

    /// Create a no-jitter sequence (single center sample).
    pub fn none() -> Self {
        Self {
            offsets: vec![[0.0, 0.0]],
            index: 0,
        }
    }

    /// Get the current jitter offset.
    #[inline]
    pub fn current(&self) -> [f32; 2] {
        self.offsets[self.index]
    }

    /// Get jitter offset for a specific frame.
    #[inline]
    pub fn for_frame(&self, frame: u32) -> [f32; 2] {
        self.offsets[(frame as usize) % self.offsets.len()]
    }

    /// Advance to the next jitter offset.
    #[inline]
    pub fn advance(&mut self) {
        self.index = (self.index + 1) % self.offsets.len();
    }

    /// Reset to the beginning of the sequence.
    #[inline]
    pub fn reset(&mut self) {
        self.index = 0;
    }

    /// Get the sequence length.
    #[inline]
    pub fn len(&self) -> usize {
        self.offsets.len()
    }

    /// Check if sequence is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.offsets.is_empty()
    }

    /// Get jitter scaled to pixel coordinates.
    #[inline]
    pub fn current_pixels(&self, width: f32, height: f32) -> [f32; 2] {
        let [jx, jy] = self.current();
        [jx / width, jy / height]
    }
}

/// Calculate Halton sequence value.
///
/// Returns a value in [0, 1) using the given base.
#[inline]
pub fn halton_sequence(mut index: u32, base: u32) -> f32 {
    let mut result = 0.0;
    let mut f = 1.0 / base as f32;

    while index > 0 {
        result += f * (index % base) as f32;
        index /= base;
        f /= base as f32;
    }

    result
}

// ---------------------------------------------------------------------------
// Motion Vector Calculation
// ---------------------------------------------------------------------------

/// Calculate motion vector for cloud reprojection.
///
/// Uses the cloud depth assumption to compute parallax-aware motion.
///
/// # Arguments
///
/// * `pixel` - Current pixel position (0 to resolution).
/// * `resolution` - Screen resolution (width, height).
/// * `prev_view_proj` - Previous frame's view-projection matrix.
/// * `curr_view_proj_inv` - Inverse of current frame's view-projection matrix.
/// * `cloud_depth` - Assumed cloud depth in world units.
///
/// # Returns
///
/// Motion vector (dx, dy) in pixels, or None if reprojection fails.
pub fn calculate_motion_vector(
    pixel: [f32; 2],
    resolution: [f32; 2],
    prev_view_proj: &[[f32; 4]; 4],
    curr_view_proj_inv: &[[f32; 4]; 4],
    cloud_depth: f32,
) -> Option<[f32; 2]> {
    // Convert pixel to NDC (-1 to 1)
    let ndc_x = (pixel[0] / resolution[0]) * 2.0 - 1.0;
    let ndc_y = 1.0 - (pixel[1] / resolution[1]) * 2.0; // Y flipped

    // Unproject to world space at cloud depth
    let clip_pos = [ndc_x, ndc_y, 0.5, 1.0]; // Mid-depth
    let world_pos = mat4_transform_point(curr_view_proj_inv, clip_pos)?;

    // Scale to cloud depth (assuming forward direction)
    let ray_length = (world_pos[0] * world_pos[0]
        + world_pos[1] * world_pos[1]
        + world_pos[2] * world_pos[2])
        .sqrt()
        .max(EPSILON);
    let scale = cloud_depth / ray_length;
    let cloud_pos = [
        world_pos[0] * scale,
        world_pos[1] * scale,
        world_pos[2] * scale,
        1.0,
    ];

    // Project to previous frame
    let prev_clip = mat4_transform_vec4(prev_view_proj, cloud_pos);
    if prev_clip[3].abs() < EPSILON {
        return None; // Behind camera
    }

    let prev_ndc_x = prev_clip[0] / prev_clip[3];
    let prev_ndc_y = prev_clip[1] / prev_clip[3];

    // Check if in valid range
    if prev_ndc_x.abs() > 1.5 || prev_ndc_y.abs() > 1.5 {
        return None; // Outside screen bounds
    }

    // Convert back to pixel coordinates
    let prev_pixel_x = (prev_ndc_x + 1.0) * 0.5 * resolution[0];
    let prev_pixel_y = (1.0 - prev_ndc_y) * 0.5 * resolution[1];

    Some([pixel[0] - prev_pixel_x, pixel[1] - prev_pixel_y])
}

/// Calculate reprojected UV coordinates for sampling history.
///
/// # Arguments
///
/// * `uv` - Current UV coordinates (0 to 1).
/// * `prev_view_proj` - Previous frame's view-projection matrix.
/// * `curr_view_proj_inv` - Inverse of current frame's view-projection matrix.
/// * `cloud_depth` - Assumed cloud depth in world units.
///
/// # Returns
///
/// Reprojected UV coordinates, or None if reprojection fails.
pub fn reproject_uv(
    uv: [f32; 2],
    prev_view_proj: &[[f32; 4]; 4],
    curr_view_proj_inv: &[[f32; 4]; 4],
    cloud_depth: f32,
) -> Option<[f32; 2]> {
    // Convert UV to NDC
    let ndc_x = uv[0] * 2.0 - 1.0;
    let ndc_y = 1.0 - uv[1] * 2.0; // Y flipped

    // Unproject to world space
    let clip_pos = [ndc_x, ndc_y, 0.5, 1.0];
    let world_pos = mat4_transform_point(curr_view_proj_inv, clip_pos)?;

    // Scale to cloud depth
    let ray_length = (world_pos[0] * world_pos[0]
        + world_pos[1] * world_pos[1]
        + world_pos[2] * world_pos[2])
        .sqrt()
        .max(EPSILON);
    let scale = cloud_depth / ray_length;
    let cloud_pos = [
        world_pos[0] * scale,
        world_pos[1] * scale,
        world_pos[2] * scale,
        1.0,
    ];

    // Project to previous frame
    let prev_clip = mat4_transform_vec4(prev_view_proj, cloud_pos);
    if prev_clip[3].abs() < EPSILON {
        return None;
    }

    let prev_ndc_x = prev_clip[0] / prev_clip[3];
    let prev_ndc_y = prev_clip[1] / prev_clip[3];

    // Convert to UV
    let prev_u = (prev_ndc_x + 1.0) * 0.5;
    let prev_v = (1.0 - prev_ndc_y) * 0.5;

    // Check bounds
    if prev_u < 0.0 || prev_u > 1.0 || prev_v < 0.0 || prev_v > 1.0 {
        None
    } else {
        Some([prev_u, prev_v])
    }
}

// ---------------------------------------------------------------------------
// Neighborhood Clamping - Ghosting Rejection
// ---------------------------------------------------------------------------

/// RGB color as 3 floats.
pub type RgbColor = [f32; 3];

/// Convert RGB to YCoCg color space.
///
/// YCoCg provides better decorrelation for color clamping.
#[inline]
pub fn rgb_to_ycocg(rgb: RgbColor) -> [f32; 3] {
    let y = 0.25 * rgb[0] + 0.5 * rgb[1] + 0.25 * rgb[2];
    let co = 0.5 * rgb[0] - 0.5 * rgb[2];
    let cg = -0.25 * rgb[0] + 0.5 * rgb[1] - 0.25 * rgb[2];
    [y, co, cg]
}

/// Convert YCoCg to RGB color space.
#[inline]
pub fn ycocg_to_rgb(ycocg: [f32; 3]) -> RgbColor {
    let y = ycocg[0];
    let co = ycocg[1];
    let cg = ycocg[2];

    let r = y + co - cg;
    let g = y + cg;
    let b = y - co - cg;

    [r, g, b]
}

/// AABB (axis-aligned bounding box) for color clamping.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ColorAABB {
    /// Minimum color values.
    pub min: [f32; 3],
    /// Maximum color values.
    pub max: [f32; 3],
}

impl ColorAABB {
    /// Create an AABB from a set of color samples.
    pub fn from_samples(samples: &[[f32; 3]]) -> Self {
        if samples.is_empty() {
            return Self::default();
        }

        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];

        for sample in samples {
            for i in 0..3 {
                min[i] = min[i].min(sample[i]);
                max[i] = max[i].max(sample[i]);
            }
        }

        Self { min, max }
    }

    /// Create an AABB from mean and variance.
    pub fn from_statistics(mean: [f32; 3], variance: [f32; 3], gamma: f32) -> Self {
        let mut min = [0.0; 3];
        let mut max = [0.0; 3];

        for i in 0..3 {
            let stddev = variance[i].sqrt() * gamma;
            min[i] = mean[i] - stddev;
            max[i] = mean[i] + stddev;
        }

        Self { min, max }
    }

    /// Expand the AABB by a factor.
    #[inline]
    pub fn expand(&mut self, factor: f32) {
        for i in 0..3 {
            let center = (self.min[i] + self.max[i]) * 0.5;
            let half_size = (self.max[i] - self.min[i]) * 0.5 * factor;
            self.min[i] = center - half_size;
            self.max[i] = center + half_size;
        }
    }

    /// Clamp a color to this AABB.
    #[inline]
    pub fn clamp(&self, color: [f32; 3]) -> [f32; 3] {
        [
            color[0].clamp(self.min[0], self.max[0]),
            color[1].clamp(self.min[1], self.max[1]),
            color[2].clamp(self.min[2], self.max[2]),
        ]
    }

    /// Clip a color towards the center of the AABB.
    ///
    /// Uses ray-AABB intersection to find the closest point on the AABB
    /// along the line from the AABB center to the color.
    pub fn clip_towards_center(&self, color: [f32; 3]) -> [f32; 3] {
        let center = [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        ];

        // Direction from center to color
        let dir = [
            color[0] - center[0],
            color[1] - center[1],
            color[2] - center[2],
        ];

        // Check if already inside
        if self.contains(color) {
            return color;
        }

        // Find intersection with AABB
        let mut t_min = 0.0_f32;
        let mut t_max = 1.0_f32;

        for i in 0..3 {
            if dir[i].abs() > EPSILON {
                let t1 = (self.min[i] - center[i]) / dir[i];
                let t2 = (self.max[i] - center[i]) / dir[i];
                let (t_near, t_far) = if t1 < t2 { (t1, t2) } else { (t2, t1) };
                t_min = t_min.max(t_near);
                t_max = t_max.min(t_far);
            }
        }

        let t = t_max.min(1.0).max(0.0);

        [
            center[0] + dir[0] * t,
            center[1] + dir[1] * t,
            center[2] + dir[2] * t,
        ]
    }

    /// Check if a color is inside the AABB.
    #[inline]
    pub fn contains(&self, color: [f32; 3]) -> bool {
        color[0] >= self.min[0]
            && color[0] <= self.max[0]
            && color[1] >= self.min[1]
            && color[1] <= self.max[1]
            && color[2] >= self.min[2]
            && color[2] <= self.max[2]
    }

    /// Get the size of the AABB.
    #[inline]
    pub fn size(&self) -> [f32; 3] {
        [
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        ]
    }

    /// Get the center of the AABB.
    #[inline]
    pub fn center(&self) -> [f32; 3] {
        [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        ]
    }
}

/// Neighborhood sampling result for clamping.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct NeighborhoodStats {
    /// Mean color of the neighborhood.
    pub mean: [f32; 3],
    /// Variance of the neighborhood.
    pub variance: [f32; 3],
    /// Minimum color in neighborhood.
    pub min: [f32; 3],
    /// Maximum color in neighborhood.
    pub max: [f32; 3],
    /// Number of samples.
    pub count: u32,
}

impl NeighborhoodStats {
    /// Compute statistics from a 3x3 neighborhood.
    pub fn from_3x3(samples: &[[f32; 3]; 9]) -> Self {
        let count = 9u32;

        // Compute mean
        let mut mean = [0.0; 3];
        for sample in samples {
            for i in 0..3 {
                mean[i] += sample[i];
            }
        }
        for m in &mut mean {
            *m /= count as f32;
        }

        // Compute variance and min/max
        let mut variance = [0.0; 3];
        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];

        for sample in samples {
            for i in 0..3 {
                let diff = sample[i] - mean[i];
                variance[i] += diff * diff;
                min[i] = min[i].min(sample[i]);
                max[i] = max[i].max(sample[i]);
            }
        }

        for v in &mut variance {
            *v /= count as f32;
        }

        Self {
            mean,
            variance,
            min,
            max,
            count,
        }
    }

    /// Compute statistics from a plus-shaped (5-tap) neighborhood.
    pub fn from_plus(samples: &[[f32; 3]; 5]) -> Self {
        let count = 5u32;

        let mut mean = [0.0; 3];
        for sample in samples {
            for i in 0..3 {
                mean[i] += sample[i];
            }
        }
        for m in &mut mean {
            *m /= count as f32;
        }

        let mut variance = [0.0; 3];
        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];

        for sample in samples {
            for i in 0..3 {
                let diff = sample[i] - mean[i];
                variance[i] += diff * diff;
                min[i] = min[i].min(sample[i]);
                max[i] = max[i].max(sample[i]);
            }
        }

        for v in &mut variance {
            *v /= count as f32;
        }

        Self {
            mean,
            variance,
            min,
            max,
            count,
        }
    }

    /// Get AABB from min/max bounds.
    #[inline]
    pub fn aabb(&self) -> ColorAABB {
        ColorAABB {
            min: self.min,
            max: self.max,
        }
    }

    /// Get variance-based AABB using mean and standard deviation.
    #[inline]
    pub fn variance_aabb(&self, gamma: f32) -> ColorAABB {
        ColorAABB::from_statistics(self.mean, self.variance, gamma)
    }

    /// Clamp a history sample to the neighborhood bounds.
    #[inline]
    pub fn clamp_history(&self, history: [f32; 3], gamma: f32) -> [f32; 3] {
        self.variance_aabb(gamma).clip_towards_center(history)
    }

    /// Calculate confidence based on variance.
    ///
    /// High variance = low confidence (fast motion/edges).
    #[inline]
    pub fn confidence(&self, threshold: f32) -> f32 {
        let total_variance = self.variance[0] + self.variance[1] + self.variance[2];
        1.0 / (1.0 + total_variance / threshold)
    }
}

// ---------------------------------------------------------------------------
// Variance-Based Blend Weight
// ---------------------------------------------------------------------------

/// Calculate temporal blend weight based on variance and disocclusion.
///
/// Lower weight = more current frame, higher weight = more history.
///
/// # Arguments
///
/// * `stats` - Neighborhood statistics from current frame.
/// * `history_color` - Reprojected history sample.
/// * `base_blend` - Base blend factor (typically 0.9).
/// * `variance_threshold` - Threshold for variance-based rejection.
///
/// # Returns
///
/// Blend weight in [0, base_blend].
pub fn calculate_blend_weight(
    stats: &NeighborhoodStats,
    history_color: [f32; 3],
    base_blend: f32,
    variance_threshold: f32,
) -> f32 {
    // Start with base blend factor
    let mut weight = base_blend;

    // Reduce weight based on variance (high variance = uncertain neighborhood)
    let confidence = stats.confidence(variance_threshold);
    weight *= confidence;

    // Reduce weight if history is far from neighborhood center
    let center = stats.mean;
    let dist_sq = (history_color[0] - center[0]).powi(2)
        + (history_color[1] - center[1]).powi(2)
        + (history_color[2] - center[2]).powi(2);

    let aabb = stats.aabb();
    let aabb_size = aabb.size();
    let max_dist_sq = (aabb_size[0].powi(2) + aabb_size[1].powi(2) + aabb_size[2].powi(2)) * 0.25;

    if max_dist_sq > EPSILON {
        let dist_factor = 1.0 - (dist_sq / max_dist_sq).min(1.0);
        weight *= dist_factor;
    }

    weight.clamp(0.0, base_blend)
}

/// Blend current and history colors.
#[inline]
pub fn blend_colors(current: [f32; 3], history: [f32; 3], weight: f32) -> [f32; 3] {
    let inv_weight = 1.0 - weight;
    [
        current[0] * inv_weight + history[0] * weight,
        current[1] * inv_weight + history[1] * weight,
        current[2] * inv_weight + history[2] * weight,
    ]
}

// ---------------------------------------------------------------------------
// Checkerboard Rendering
// ---------------------------------------------------------------------------

/// Checkerboard pattern for half-resolution rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CheckerboardPhase {
    /// Render even pixels (x + y) % 2 == 0.
    Even = 0,
    /// Render odd pixels (x + y) % 2 == 1.
    Odd = 1,
}

impl CheckerboardPhase {
    /// Get the phase for a given frame index.
    #[inline]
    pub fn from_frame(frame: u32) -> Self {
        if frame % 2 == 0 {
            CheckerboardPhase::Even
        } else {
            CheckerboardPhase::Odd
        }
    }

    /// Get the other phase.
    #[inline]
    pub fn other(&self) -> Self {
        match self {
            CheckerboardPhase::Even => CheckerboardPhase::Odd,
            CheckerboardPhase::Odd => CheckerboardPhase::Even,
        }
    }

    /// Check if a pixel should be rendered in this phase.
    #[inline]
    pub fn should_render(&self, x: u32, y: u32) -> bool {
        let parity = (x + y) % 2;
        match self {
            CheckerboardPhase::Even => parity == 0,
            CheckerboardPhase::Odd => parity == 1,
        }
    }

    /// Get the offset needed to find the corresponding pixel in the other phase.
    #[inline]
    pub fn neighbor_offset(&self) -> (i32, i32) {
        // Return offset to nearest rendered neighbor
        (1, 0) // Horizontal neighbor
    }
}

/// Checkerboard reconstruction helper.
#[derive(Debug, Clone, Copy)]
pub struct CheckerboardReconstruct {
    /// Current phase being rendered.
    pub phase: CheckerboardPhase,
    /// Frame index.
    pub frame: u32,
}

impl CheckerboardReconstruct {
    /// Create a new checkerboard reconstructor.
    #[inline]
    pub fn new(frame: u32) -> Self {
        Self {
            phase: CheckerboardPhase::from_frame(frame),
            frame,
        }
    }

    /// Get the UV offset to sample history for a non-rendered pixel.
    ///
    /// Returns the offset in normalized UV coordinates.
    #[inline]
    pub fn history_offset(&self, x: u32, y: u32, resolution: [f32; 2]) -> [f32; 2] {
        if self.phase.should_render(x, y) {
            // This pixel was rendered, no offset needed
            [0.0, 0.0]
        } else {
            // This pixel was not rendered, sample from history at this location
            [0.0, 0.0] // History should have valid data from previous frame
        }
    }

    /// Check if a pixel needs reconstruction from history.
    #[inline]
    pub fn needs_reconstruction(&self, x: u32, y: u32) -> bool {
        !self.phase.should_render(x, y)
    }

    /// Blend newly rendered pixel with history.
    #[inline]
    pub fn blend_rendered(
        &self,
        current: [f32; 3],
        history: [f32; 3],
        blend_factor: f32,
    ) -> [f32; 3] {
        blend_colors(current, history, blend_factor)
    }

    /// Reconstruct non-rendered pixel from neighbors and history.
    ///
    /// Uses bilinear filtering of rendered neighbors weighted with history.
    pub fn reconstruct_pixel(
        &self,
        x: u32,
        y: u32,
        neighbors: &[[f32; 3]; 4], // left, right, up, down
        history: [f32; 3],
        history_valid: bool,
    ) -> [f32; 3] {
        // Average of horizontal and vertical neighbors
        let horizontal = [
            (neighbors[0][0] + neighbors[1][0]) * 0.5,
            (neighbors[0][1] + neighbors[1][1]) * 0.5,
            (neighbors[0][2] + neighbors[1][2]) * 0.5,
        ];

        let vertical = [
            (neighbors[2][0] + neighbors[3][0]) * 0.5,
            (neighbors[2][1] + neighbors[3][1]) * 0.5,
            (neighbors[2][2] + neighbors[3][2]) * 0.5,
        ];

        let spatial = [
            (horizontal[0] + vertical[0]) * 0.5,
            (horizontal[1] + vertical[1]) * 0.5,
            (horizontal[2] + vertical[2]) * 0.5,
        ];

        if history_valid {
            // Blend spatial with history
            blend_colors(spatial, history, 0.5)
        } else {
            spatial
        }
    }
}

// ---------------------------------------------------------------------------
// Bilinear Sampling
// ---------------------------------------------------------------------------

/// Bilinear sample from a 2x2 grid of samples.
///
/// # Arguments
///
/// * `samples` - 2x2 grid in row-major order [top-left, top-right, bottom-left, bottom-right].
/// * `frac` - Fractional position within the cell (0-1 for each axis).
#[inline]
pub fn bilinear_sample(samples: &[[f32; 3]; 4], frac: [f32; 2]) -> [f32; 3] {
    let fx = frac[0];
    let fy = frac[1];

    let top = [
        samples[0][0] * (1.0 - fx) + samples[1][0] * fx,
        samples[0][1] * (1.0 - fx) + samples[1][1] * fx,
        samples[0][2] * (1.0 - fx) + samples[1][2] * fx,
    ];

    let bottom = [
        samples[2][0] * (1.0 - fx) + samples[3][0] * fx,
        samples[2][1] * (1.0 - fx) + samples[3][1] * fx,
        samples[2][2] * (1.0 - fx) + samples[3][2] * fx,
    ];

    [
        top[0] * (1.0 - fy) + bottom[0] * fy,
        top[1] * (1.0 - fy) + bottom[1] * fy,
        top[2] * (1.0 - fy) + bottom[2] * fy,
    ]
}

/// Calculate bilinear weights for a position.
///
/// Returns (base_x, base_y, frac_x, frac_y) where base is the integer coordinate
/// and frac is the fractional offset.
#[inline]
pub fn bilinear_coordinates(uv: [f32; 2], resolution: [f32; 2]) -> (i32, i32, f32, f32) {
    let px = uv[0] * resolution[0] - 0.5;
    let py = uv[1] * resolution[1] - 0.5;

    let base_x = px.floor() as i32;
    let base_y = py.floor() as i32;

    let frac_x = px - base_x as f32;
    let frac_y = py - base_y as f32;

    (base_x, base_y, frac_x, frac_y)
}

// ---------------------------------------------------------------------------
// Matrix Utilities
// ---------------------------------------------------------------------------

/// Identity 4x4 matrix.
pub const IDENTITY_MATRIX: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

/// Transform a vec4 by a 4x4 matrix.
#[inline]
pub fn mat4_transform_vec4(m: &[[f32; 4]; 4], v: [f32; 4]) -> [f32; 4] {
    [
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2] + m[3][0] * v[3],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2] + m[3][1] * v[3],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2] + m[3][2] * v[3],
        m[0][3] * v[0] + m[1][3] * v[1] + m[2][3] * v[2] + m[3][3] * v[3],
    ]
}

/// Transform a point by a 4x4 matrix, performing perspective divide.
///
/// Returns None if the w component is too small (behind camera).
#[inline]
pub fn mat4_transform_point(m: &[[f32; 4]; 4], p: [f32; 4]) -> Option<[f32; 3]> {
    let result = mat4_transform_vec4(m, p);
    if result[3].abs() < EPSILON {
        return None;
    }
    let inv_w = 1.0 / result[3];
    Some([result[0] * inv_w, result[1] * inv_w, result[2] * inv_w])
}

/// Multiply two 4x4 matrices.
#[inline]
pub fn mat4_mul(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut result = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            result[i][j] = a[0][j] * b[i][0]
                + a[1][j] * b[i][1]
                + a[2][j] * b[i][2]
                + a[3][j] * b[i][3];
        }
    }
    result
}

/// Invert a 4x4 matrix.
///
/// Returns None if the matrix is singular.
pub fn mat4_inverse(m: &[[f32; 4]; 4]) -> Option<[[f32; 4]; 4]> {
    let mut inv = [[0.0f32; 4]; 4];

    inv[0][0] = m[1][1] * m[2][2] * m[3][3]
        - m[1][1] * m[2][3] * m[3][2]
        - m[2][1] * m[1][2] * m[3][3]
        + m[2][1] * m[1][3] * m[3][2]
        + m[3][1] * m[1][2] * m[2][3]
        - m[3][1] * m[1][3] * m[2][2];

    inv[1][0] = -m[1][0] * m[2][2] * m[3][3]
        + m[1][0] * m[2][3] * m[3][2]
        + m[2][0] * m[1][2] * m[3][3]
        - m[2][0] * m[1][3] * m[3][2]
        - m[3][0] * m[1][2] * m[2][3]
        + m[3][0] * m[1][3] * m[2][2];

    inv[2][0] = m[1][0] * m[2][1] * m[3][3]
        - m[1][0] * m[2][3] * m[3][1]
        - m[2][0] * m[1][1] * m[3][3]
        + m[2][0] * m[1][3] * m[3][1]
        + m[3][0] * m[1][1] * m[2][3]
        - m[3][0] * m[1][3] * m[2][1];

    inv[3][0] = -m[1][0] * m[2][1] * m[3][2]
        + m[1][0] * m[2][2] * m[3][1]
        + m[2][0] * m[1][1] * m[3][2]
        - m[2][0] * m[1][2] * m[3][1]
        - m[3][0] * m[1][1] * m[2][2]
        + m[3][0] * m[1][2] * m[2][1];

    inv[0][1] = -m[0][1] * m[2][2] * m[3][3]
        + m[0][1] * m[2][3] * m[3][2]
        + m[2][1] * m[0][2] * m[3][3]
        - m[2][1] * m[0][3] * m[3][2]
        - m[3][1] * m[0][2] * m[2][3]
        + m[3][1] * m[0][3] * m[2][2];

    inv[1][1] = m[0][0] * m[2][2] * m[3][3]
        - m[0][0] * m[2][3] * m[3][2]
        - m[2][0] * m[0][2] * m[3][3]
        + m[2][0] * m[0][3] * m[3][2]
        + m[3][0] * m[0][2] * m[2][3]
        - m[3][0] * m[0][3] * m[2][2];

    inv[2][1] = -m[0][0] * m[2][1] * m[3][3]
        + m[0][0] * m[2][3] * m[3][1]
        + m[2][0] * m[0][1] * m[3][3]
        - m[2][0] * m[0][3] * m[3][1]
        - m[3][0] * m[0][1] * m[2][3]
        + m[3][0] * m[0][3] * m[2][1];

    inv[3][1] = m[0][0] * m[2][1] * m[3][2]
        - m[0][0] * m[2][2] * m[3][1]
        - m[2][0] * m[0][1] * m[3][2]
        + m[2][0] * m[0][2] * m[3][1]
        + m[3][0] * m[0][1] * m[2][2]
        - m[3][0] * m[0][2] * m[2][1];

    inv[0][2] = m[0][1] * m[1][2] * m[3][3]
        - m[0][1] * m[1][3] * m[3][2]
        - m[1][1] * m[0][2] * m[3][3]
        + m[1][1] * m[0][3] * m[3][2]
        + m[3][1] * m[0][2] * m[1][3]
        - m[3][1] * m[0][3] * m[1][2];

    inv[1][2] = -m[0][0] * m[1][2] * m[3][3]
        + m[0][0] * m[1][3] * m[3][2]
        + m[1][0] * m[0][2] * m[3][3]
        - m[1][0] * m[0][3] * m[3][2]
        - m[3][0] * m[0][2] * m[1][3]
        + m[3][0] * m[0][3] * m[1][2];

    inv[2][2] = m[0][0] * m[1][1] * m[3][3]
        - m[0][0] * m[1][3] * m[3][1]
        - m[1][0] * m[0][1] * m[3][3]
        + m[1][0] * m[0][3] * m[3][1]
        + m[3][0] * m[0][1] * m[1][3]
        - m[3][0] * m[0][3] * m[1][1];

    inv[3][2] = -m[0][0] * m[1][1] * m[3][2]
        + m[0][0] * m[1][2] * m[3][1]
        + m[1][0] * m[0][1] * m[3][2]
        - m[1][0] * m[0][2] * m[3][1]
        - m[3][0] * m[0][1] * m[1][2]
        + m[3][0] * m[0][2] * m[1][1];

    inv[0][3] = -m[0][1] * m[1][2] * m[2][3]
        + m[0][1] * m[1][3] * m[2][2]
        + m[1][1] * m[0][2] * m[2][3]
        - m[1][1] * m[0][3] * m[2][2]
        - m[2][1] * m[0][2] * m[1][3]
        + m[2][1] * m[0][3] * m[1][2];

    inv[1][3] = m[0][0] * m[1][2] * m[2][3]
        - m[0][0] * m[1][3] * m[2][2]
        - m[1][0] * m[0][2] * m[2][3]
        + m[1][0] * m[0][3] * m[2][2]
        + m[2][0] * m[0][2] * m[1][3]
        - m[2][0] * m[0][3] * m[1][2];

    inv[2][3] = -m[0][0] * m[1][1] * m[2][3]
        + m[0][0] * m[1][3] * m[2][1]
        + m[1][0] * m[0][1] * m[2][3]
        - m[1][0] * m[0][3] * m[2][1]
        - m[2][0] * m[0][1] * m[1][3]
        + m[2][0] * m[0][3] * m[1][1];

    inv[3][3] = m[0][0] * m[1][1] * m[2][2]
        - m[0][0] * m[1][2] * m[2][1]
        - m[1][0] * m[0][1] * m[2][2]
        + m[1][0] * m[0][2] * m[2][1]
        + m[2][0] * m[0][1] * m[1][2]
        - m[2][0] * m[0][2] * m[1][1];

    let det = m[0][0] * inv[0][0] + m[0][1] * inv[1][0] + m[0][2] * inv[2][0] + m[0][3] * inv[3][0];

    if det.abs() < EPSILON {
        return None;
    }

    let inv_det = 1.0 / det;
    for i in 0..4 {
        for j in 0..4 {
            inv[i][j] *= inv_det;
        }
    }

    Some(inv)
}

// ---------------------------------------------------------------------------
// CloudTemporalState - Complete temporal reprojection state
// ---------------------------------------------------------------------------

/// Complete temporal reprojection state.
///
/// Combines all components needed for temporal cloud reprojection.
#[derive(Debug, Clone)]
pub struct CloudTemporalState {
    /// GPU uniforms for shader upload.
    pub uniforms: CloudTemporalUniforms,

    /// History buffer state.
    pub history: HistoryBufferState,

    /// Jitter sequence for TAA.
    pub jitter: JitterSequence,

    /// Cached inverse of current view-projection.
    pub curr_view_proj_inv: Option<[[f32; 4]; 4]>,
}

impl Default for CloudTemporalState {
    fn default() -> Self {
        Self::new()
    }
}

impl CloudTemporalState {
    /// Create a new temporal state with default settings.
    pub fn new() -> Self {
        Self {
            uniforms: CloudTemporalUniforms::default(),
            history: HistoryBufferState::default(),
            jitter: JitterSequence::default(),
            curr_view_proj_inv: None,
        }
    }

    /// Create a temporal state with specified resolution.
    pub fn with_resolution(width: u32, height: u32) -> Self {
        Self {
            uniforms: CloudTemporalUniforms::with_resolution(width, height),
            history: HistoryBufferState::new(width, height),
            jitter: JitterSequence::default(),
            curr_view_proj_inv: None,
        }
    }

    /// Update for a new frame with a new view-projection matrix.
    pub fn update(&mut self, view_proj: [[f32; 4]; 4]) {
        // Advance history and jitter
        self.history.advance();
        self.jitter.advance();

        // Update uniforms
        self.uniforms.advance_frame(view_proj);
        let jitter = self.jitter.current_pixels(
            self.uniforms.resolution[0],
            self.uniforms.resolution[1],
        );
        self.uniforms.set_jitter(jitter[0], jitter[1]);

        // Update effective blend factor based on history validity
        let effective_blend = self.history.effective_blend_factor(self.uniforms.blend_factor);
        self.uniforms.blend_factor = effective_blend;

        // Cache inverse matrix
        self.curr_view_proj_inv = mat4_inverse(&view_proj);
    }

    /// Check and update resolution, invalidating history if changed.
    pub fn set_resolution(&mut self, width: u32, height: u32) {
        self.history.check_resolution(width, height);
        self.uniforms.resolution = [width as f32, height as f32];
    }

    /// Invalidate history (e.g., on camera cut).
    pub fn invalidate(&mut self) {
        self.history.invalidate();
    }

    /// Get reprojected UV for sampling history.
    pub fn reproject(&self, uv: [f32; 2]) -> Option<[f32; 2]> {
        let inv = self.curr_view_proj_inv.as_ref()?;
        reproject_uv(
            uv,
            &self.uniforms.prev_view_proj,
            inv,
            self.uniforms.cloud_depth,
        )
    }

    /// Check if history is valid for reprojection.
    #[inline]
    pub fn is_history_valid(&self) -> bool {
        self.history.valid
    }

    /// Get the current frame index.
    #[inline]
    pub fn frame_index(&self) -> u32 {
        self.uniforms.frame_index
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // CloudTemporalUniforms tests
    // ========================================================================

    #[test]
    fn test_uniforms_default() {
        let uniforms = CloudTemporalUniforms::default();
        assert_eq!(uniforms.blend_factor, DEFAULT_BLEND_FACTOR);
        assert_eq!(uniforms.frame_index, 0);
        assert_eq!(uniforms.resolution, [1920.0, 1080.0]);
    }

    #[test]
    fn test_uniforms_with_resolution() {
        let uniforms = CloudTemporalUniforms::with_resolution(3840, 2160);
        assert_eq!(uniforms.resolution, [3840.0, 2160.0]);
    }

    #[test]
    fn test_uniforms_set_matrices() {
        let mut uniforms = CloudTemporalUniforms::default();
        let prev = [[1.0; 4]; 4];
        let curr = [[2.0; 4]; 4];
        uniforms.set_matrices(prev, curr);
        assert_eq!(uniforms.prev_view_proj, prev);
        assert_eq!(uniforms.curr_view_proj, curr);
    }

    #[test]
    fn test_uniforms_advance_frame() {
        let mut uniforms = CloudTemporalUniforms::default();
        uniforms.curr_view_proj = [[2.0; 4]; 4];
        let new_matrix = [[3.0; 4]; 4];
        uniforms.advance_frame(new_matrix);

        assert_eq!(uniforms.prev_view_proj, [[2.0; 4]; 4]);
        assert_eq!(uniforms.curr_view_proj, [[3.0; 4]; 4]);
        assert_eq!(uniforms.frame_index, 1);
    }

    #[test]
    fn test_uniforms_set_jitter() {
        let mut uniforms = CloudTemporalUniforms::default();
        uniforms.set_jitter(0.25, -0.25);
        assert_eq!(uniforms.jitter_offset, [0.25, -0.25]);
    }

    #[test]
    fn test_uniforms_set_blend_factor() {
        let mut uniforms = CloudTemporalUniforms::default();
        uniforms.set_blend_factor(0.5);
        assert_eq!(uniforms.blend_factor, 0.5);

        uniforms.set_blend_factor(2.0);
        assert_eq!(uniforms.blend_factor, MAX_BLEND_FACTOR);

        uniforms.set_blend_factor(-1.0);
        assert_eq!(uniforms.blend_factor, MIN_BLEND_FACTOR);
    }

    #[test]
    fn test_uniforms_checkerboard() {
        let mut uniforms = CloudTemporalUniforms::default();
        assert!(!uniforms.is_checkerboard_enabled());

        uniforms.set_checkerboard(true);
        assert!(uniforms.is_checkerboard_enabled());
        assert_eq!(uniforms.enable_checkerboard, 1);

        uniforms.set_checkerboard(false);
        assert!(!uniforms.is_checkerboard_enabled());
    }

    #[test]
    fn test_uniforms_validate() {
        let uniforms = CloudTemporalUniforms::default();
        assert!(uniforms.validate());

        let mut invalid = uniforms;
        invalid.blend_factor = 2.0;
        assert!(!invalid.validate());

        let mut invalid2 = CloudTemporalUniforms::default();
        invalid2.resolution = [0.0, 1080.0];
        assert!(!invalid2.validate());

        let mut invalid3 = CloudTemporalUniforms::default();
        invalid3.cloud_depth = 0.0;
        assert!(!invalid3.validate());
    }

    #[test]
    fn test_uniforms_size() {
        assert_eq!(std::mem::size_of::<CloudTemporalUniforms>(), 192);
    }

    #[test]
    fn test_uniforms_alignment() {
        assert_eq!(std::mem::align_of::<CloudTemporalUniforms>(), 4);
    }

    #[test]
    fn test_uniforms_pod_zeroable() {
        let zeroed: CloudTemporalUniforms = bytemuck::Zeroable::zeroed();
        assert_eq!(zeroed.frame_index, 0);
        assert_eq!(zeroed.blend_factor, 0.0);
    }

    // ========================================================================
    // BufferIndex tests
    // ========================================================================

    #[test]
    fn test_buffer_index_other() {
        assert_eq!(BufferIndex::A.other(), BufferIndex::B);
        assert_eq!(BufferIndex::B.other(), BufferIndex::A);
    }

    #[test]
    fn test_buffer_index_from_frame() {
        assert_eq!(BufferIndex::from_frame(0), BufferIndex::A);
        assert_eq!(BufferIndex::from_frame(1), BufferIndex::B);
        assert_eq!(BufferIndex::from_frame(2), BufferIndex::A);
        assert_eq!(BufferIndex::from_frame(100), BufferIndex::A);
        assert_eq!(BufferIndex::from_frame(101), BufferIndex::B);
    }

    // ========================================================================
    // HistoryBufferState tests
    // ========================================================================

    #[test]
    fn test_history_default() {
        let state = HistoryBufferState::default();
        assert_eq!(state.current, BufferIndex::A);
        assert_eq!(state.resolution, (1920, 1080));
        assert!(!state.valid);
        assert_eq!(state.frames_accumulated, 0);
    }

    #[test]
    fn test_history_new() {
        let state = HistoryBufferState::new(3840, 2160);
        assert_eq!(state.resolution, (3840, 2160));
        assert!(!state.valid);
    }

    #[test]
    fn test_history_advance() {
        let mut state = HistoryBufferState::default();
        assert_eq!(state.current, BufferIndex::A);
        assert!(!state.valid);

        state.advance();
        assert_eq!(state.current, BufferIndex::B);
        assert!(state.valid);
        assert_eq!(state.frames_accumulated, 1);

        state.advance();
        assert_eq!(state.current, BufferIndex::A);
        assert_eq!(state.frames_accumulated, 2);
    }

    #[test]
    fn test_history_index() {
        let state = HistoryBufferState::default();
        assert_eq!(state.history_index(), BufferIndex::B);
    }

    #[test]
    fn test_history_invalidate() {
        let mut state = HistoryBufferState::default();
        state.advance();
        state.advance();
        assert!(state.valid);
        assert_eq!(state.frames_accumulated, 2);

        state.invalidate();
        assert!(!state.valid);
        assert_eq!(state.frames_accumulated, 0);
    }

    #[test]
    fn test_history_check_resolution() {
        let mut state = HistoryBufferState::new(1920, 1080);
        state.advance();
        assert!(state.valid);

        assert!(state.check_resolution(1920, 1080));
        assert!(state.valid);

        assert!(!state.check_resolution(3840, 2160));
        assert!(!state.valid);
        assert_eq!(state.resolution, (3840, 2160));
    }

    #[test]
    fn test_history_effective_blend_factor() {
        let mut state = HistoryBufferState::default();
        assert_eq!(state.effective_blend_factor(0.9), 0.0); // No history

        state.advance();
        assert!((state.effective_blend_factor(0.9) - 0.225).abs() < 0.01); // 1/4 of 0.9

        state.advance();
        assert!((state.effective_blend_factor(0.9) - 0.45).abs() < 0.01); // 2/4 of 0.9

        state.advance();
        state.advance();
        assert!((state.effective_blend_factor(0.9) - 0.9).abs() < 0.01); // Full blend
    }

    // ========================================================================
    // JitterSequence tests
    // ========================================================================

    #[test]
    fn test_jitter_halton_default() {
        let jitter = JitterSequence::default();
        assert_eq!(jitter.len(), DEFAULT_JITTER_SEQUENCE_LENGTH as usize);
    }

    #[test]
    fn test_jitter_halton_values() {
        let jitter = JitterSequence::halton(8);

        // Verify all values are in [-0.5, 0.5]
        for i in 0..jitter.len() {
            let [x, y] = jitter.for_frame(i as u32);
            assert!(x >= -0.5 && x <= 0.5, "x={} out of range", x);
            assert!(y >= -0.5 && y <= 0.5, "y={} out of range", y);
        }
    }

    #[test]
    fn test_jitter_grid_2x2() {
        let jitter = JitterSequence::grid_2x2();
        assert_eq!(jitter.len(), 4);

        // Check expected values
        assert_eq!(jitter.for_frame(0), [-0.25, -0.25]);
        assert_eq!(jitter.for_frame(1), [0.25, -0.25]);
        assert_eq!(jitter.for_frame(2), [-0.25, 0.25]);
        assert_eq!(jitter.for_frame(3), [0.25, 0.25]);
    }

    #[test]
    fn test_jitter_grid_4x4() {
        let jitter = JitterSequence::grid_4x4();
        assert_eq!(jitter.len(), 16);
    }

    #[test]
    fn test_jitter_none() {
        let jitter = JitterSequence::none();
        assert_eq!(jitter.len(), 1);
        assert_eq!(jitter.current(), [0.0, 0.0]);
    }

    #[test]
    fn test_jitter_advance() {
        let mut jitter = JitterSequence::halton(4);
        let first = jitter.current();
        jitter.advance();
        let second = jitter.current();
        assert_ne!(first, second);

        jitter.reset();
        assert_eq!(jitter.current(), first);
    }

    #[test]
    fn test_jitter_for_frame() {
        let jitter = JitterSequence::halton(4);
        assert_eq!(jitter.for_frame(0), jitter.for_frame(4));
        assert_eq!(jitter.for_frame(1), jitter.for_frame(5));
    }

    #[test]
    fn test_jitter_current_pixels() {
        let jitter = JitterSequence::grid_2x2();
        let [x, y] = jitter.current_pixels(1920.0, 1080.0);
        assert!((x - (-0.25 / 1920.0)).abs() < 1e-6);
        assert!((y - (-0.25 / 1080.0)).abs() < 1e-6);
    }

    #[test]
    fn test_jitter_is_empty() {
        let jitter = JitterSequence::halton(4);
        assert!(!jitter.is_empty());
    }

    // ========================================================================
    // Halton sequence tests
    // ========================================================================

    #[test]
    fn test_halton_sequence_base2() {
        assert!((halton_sequence(1, 2) - 0.5).abs() < 1e-6);
        assert!((halton_sequence(2, 2) - 0.25).abs() < 1e-6);
        assert!((halton_sequence(3, 2) - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_halton_sequence_base3() {
        assert!((halton_sequence(1, 3) - (1.0 / 3.0)).abs() < 1e-6);
        assert!((halton_sequence(2, 3) - (2.0 / 3.0)).abs() < 1e-6);
    }

    #[test]
    fn test_halton_sequence_range() {
        for i in 1..100 {
            let h2 = halton_sequence(i, 2);
            let h3 = halton_sequence(i, 3);
            assert!(h2 >= 0.0 && h2 < 1.0, "h2={} out of range", h2);
            assert!(h3 >= 0.0 && h3 < 1.0, "h3={} out of range", h3);
        }
    }

    // ========================================================================
    // Color space conversion tests
    // ========================================================================

    #[test]
    fn test_rgb_to_ycocg_black() {
        let rgb = [0.0, 0.0, 0.0];
        let ycocg = rgb_to_ycocg(rgb);
        assert_eq!(ycocg, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_rgb_to_ycocg_white() {
        let rgb = [1.0, 1.0, 1.0];
        let ycocg = rgb_to_ycocg(rgb);
        assert!((ycocg[0] - 1.0).abs() < 1e-6); // Y = 1
        assert!((ycocg[1] - 0.0).abs() < 1e-6); // Co = 0
        assert!((ycocg[2] - 0.0).abs() < 1e-6); // Cg = 0
    }

    #[test]
    fn test_rgb_to_ycocg_red() {
        let rgb = [1.0, 0.0, 0.0];
        let ycocg = rgb_to_ycocg(rgb);
        assert!((ycocg[0] - 0.25).abs() < 1e-6); // Y
        assert!((ycocg[1] - 0.5).abs() < 1e-6);  // Co (positive for red)
        assert!((ycocg[2] - (-0.25)).abs() < 1e-6); // Cg
    }

    #[test]
    fn test_ycocg_to_rgb_roundtrip() {
        let colors = [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.5],
            [0.2, 0.6, 0.9],
        ];

        for rgb in colors {
            let ycocg = rgb_to_ycocg(rgb);
            let rgb2 = ycocg_to_rgb(ycocg);
            for i in 0..3 {
                assert!(
                    (rgb[i] - rgb2[i]).abs() < 1e-5,
                    "Roundtrip failed for {:?}: got {:?}",
                    rgb,
                    rgb2
                );
            }
        }
    }

    // ========================================================================
    // ColorAABB tests
    // ========================================================================

    #[test]
    fn test_aabb_from_samples() {
        let samples = [
            [0.0, 0.2, 0.4],
            [0.1, 0.3, 0.5],
            [0.2, 0.1, 0.3],
        ];
        let aabb = ColorAABB::from_samples(&samples);
        assert_eq!(aabb.min, [0.0, 0.1, 0.3]);
        assert_eq!(aabb.max, [0.2, 0.3, 0.5]);
    }

    #[test]
    fn test_aabb_from_samples_empty() {
        let aabb = ColorAABB::from_samples(&[]);
        assert_eq!(aabb.min, [0.0, 0.0, 0.0]);
        assert_eq!(aabb.max, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_aabb_from_statistics() {
        let mean = [0.5, 0.5, 0.5];
        let variance = [0.01, 0.04, 0.09];
        let gamma = 1.0;
        let aabb = ColorAABB::from_statistics(mean, variance, gamma);

        assert!((aabb.min[0] - 0.4).abs() < 1e-5); // 0.5 - 0.1
        assert!((aabb.max[0] - 0.6).abs() < 1e-5); // 0.5 + 0.1
        assert!((aabb.min[1] - 0.3).abs() < 1e-5); // 0.5 - 0.2
        assert!((aabb.max[1] - 0.7).abs() < 1e-5);
    }

    #[test]
    fn test_aabb_expand() {
        let mut aabb = ColorAABB {
            min: [0.0, 0.0, 0.0],
            max: [1.0, 1.0, 1.0],
        };
        aabb.expand(2.0);
        assert_eq!(aabb.min, [-0.5, -0.5, -0.5]);
        assert_eq!(aabb.max, [1.5, 1.5, 1.5]);
    }

    #[test]
    fn test_aabb_clamp() {
        let aabb = ColorAABB {
            min: [0.0, 0.0, 0.0],
            max: [1.0, 1.0, 1.0],
        };

        assert_eq!(aabb.clamp([0.5, 0.5, 0.5]), [0.5, 0.5, 0.5]);
        assert_eq!(aabb.clamp([1.5, 0.5, 0.5]), [1.0, 0.5, 0.5]);
        assert_eq!(aabb.clamp([-0.5, 0.5, 0.5]), [0.0, 0.5, 0.5]);
    }

    #[test]
    fn test_aabb_contains() {
        let aabb = ColorAABB {
            min: [0.0, 0.0, 0.0],
            max: [1.0, 1.0, 1.0],
        };

        assert!(aabb.contains([0.5, 0.5, 0.5]));
        assert!(aabb.contains([0.0, 0.0, 0.0]));
        assert!(aabb.contains([1.0, 1.0, 1.0]));
        assert!(!aabb.contains([1.1, 0.5, 0.5]));
        assert!(!aabb.contains([-0.1, 0.5, 0.5]));
    }

    #[test]
    fn test_aabb_clip_towards_center() {
        let aabb = ColorAABB {
            min: [0.0, 0.0, 0.0],
            max: [1.0, 1.0, 1.0],
        };

        // Inside: should return unchanged
        let inside = [0.5, 0.5, 0.5];
        assert_eq!(aabb.clip_towards_center(inside), inside);

        // Outside: should clip to boundary
        let outside = [2.0, 0.5, 0.5];
        let clipped = aabb.clip_towards_center(outside);
        assert!((clipped[0] - 1.0).abs() < 1e-5);
        assert!((clipped[1] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_aabb_size() {
        let aabb = ColorAABB {
            min: [0.0, 0.1, 0.2],
            max: [1.0, 0.5, 0.8],
        };
        let size = aabb.size();
        assert!((size[0] - 1.0).abs() < 1e-5);
        assert!((size[1] - 0.4).abs() < 1e-5);
        assert!((size[2] - 0.6).abs() < 1e-5);
    }

    #[test]
    fn test_aabb_center() {
        let aabb = ColorAABB {
            min: [0.0, 0.0, 0.0],
            max: [1.0, 1.0, 1.0],
        };
        assert_eq!(aabb.center(), [0.5, 0.5, 0.5]);
    }

    // ========================================================================
    // NeighborhoodStats tests
    // ========================================================================

    #[test]
    fn test_neighborhood_from_3x3() {
        let samples = [
            [0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2],
            [0.3, 0.3, 0.3],
            [0.1, 0.1, 0.1],
            [0.5, 0.5, 0.5], // Center
            [0.1, 0.1, 0.1],
            [0.3, 0.3, 0.3],
            [0.2, 0.2, 0.2],
            [0.1, 0.1, 0.1],
        ];
        let stats = NeighborhoodStats::from_3x3(&samples);

        assert_eq!(stats.count, 9);
        assert_eq!(stats.min, [0.1, 0.1, 0.1]);
        assert_eq!(stats.max, [0.5, 0.5, 0.5]);
    }

    #[test]
    fn test_neighborhood_from_plus() {
        let samples = [
            [0.0, 0.0, 0.0], // left
            [1.0, 1.0, 1.0], // right
            [0.0, 0.0, 0.0], // up
            [1.0, 1.0, 1.0], // down
            [0.5, 0.5, 0.5], // center
        ];
        let stats = NeighborhoodStats::from_plus(&samples);

        assert_eq!(stats.count, 5);
        assert_eq!(stats.min, [0.0, 0.0, 0.0]);
        assert_eq!(stats.max, [1.0, 1.0, 1.0]);
        assert!((stats.mean[0] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_neighborhood_aabb() {
        let samples: [[f32; 3]; 9] = [[0.5; 3]; 9];
        let stats = NeighborhoodStats::from_3x3(&samples);
        let aabb = stats.aabb();

        assert_eq!(aabb.min, [0.5, 0.5, 0.5]);
        assert_eq!(aabb.max, [0.5, 0.5, 0.5]);
    }

    #[test]
    fn test_neighborhood_variance_aabb() {
        let samples: [[f32; 3]; 9] = [[0.5; 3]; 9];
        let stats = NeighborhoodStats::from_3x3(&samples);
        let aabb = stats.variance_aabb(1.0);

        // Zero variance means AABB collapses to mean
        assert!((aabb.min[0] - 0.5).abs() < 1e-5);
        assert!((aabb.max[0] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_neighborhood_confidence() {
        let uniform: [[f32; 3]; 9] = [[0.5; 3]; 9];
        let stats = NeighborhoodStats::from_3x3(&uniform);
        assert!((stats.confidence(0.1) - 1.0).abs() < 1e-5); // Zero variance = max confidence
    }

    #[test]
    fn test_neighborhood_clamp_history() {
        let samples = [
            [0.4, 0.4, 0.4],
            [0.5, 0.5, 0.5],
            [0.6, 0.6, 0.6],
            [0.4, 0.4, 0.4],
            [0.5, 0.5, 0.5],
            [0.6, 0.6, 0.6],
            [0.4, 0.4, 0.4],
            [0.5, 0.5, 0.5],
            [0.6, 0.6, 0.6],
        ];
        let stats = NeighborhoodStats::from_3x3(&samples);

        // History inside neighborhood should pass through
        let inside = [0.5, 0.5, 0.5];
        let clamped = stats.clamp_history(inside, 1.0);
        assert!((clamped[0] - 0.5).abs() < 1e-5);
    }

    // ========================================================================
    // Blend weight calculation tests
    // ========================================================================

    #[test]
    fn test_calculate_blend_weight_perfect_match() {
        let samples: [[f32; 3]; 9] = [[0.5; 3]; 9];
        let stats = NeighborhoodStats::from_3x3(&samples);
        let history = [0.5, 0.5, 0.5];
        let weight = calculate_blend_weight(&stats, history, 0.9, 0.1);

        // Perfect match should give high weight
        assert!(weight > 0.8);
    }

    #[test]
    fn test_calculate_blend_weight_mismatch() {
        // Use samples with variation to get a meaningful AABB
        let samples = [
            [0.4, 0.4, 0.4],
            [0.5, 0.5, 0.5],
            [0.6, 0.6, 0.6],
            [0.4, 0.5, 0.6],
            [0.5, 0.5, 0.5],
            [0.6, 0.5, 0.4],
            [0.45, 0.45, 0.45],
            [0.55, 0.55, 0.55],
            [0.5, 0.5, 0.5],
        ];
        let stats = NeighborhoodStats::from_3x3(&samples);
        let history = [1.0, 0.0, 0.5]; // Very different from [0.4-0.6] range
        let weight = calculate_blend_weight(&stats, history, 0.9, 0.1);

        // Mismatch should reduce weight
        assert!(weight < 0.5, "weight={} should be < 0.5 for mismatched history", weight);
    }

    #[test]
    fn test_blend_colors() {
        let current = [1.0, 0.0, 0.0];
        let history = [0.0, 1.0, 0.0];

        let blend_0 = blend_colors(current, history, 0.0);
        assert_eq!(blend_0, current);

        let blend_1 = blend_colors(current, history, 1.0);
        assert_eq!(blend_1, history);

        let blend_half = blend_colors(current, history, 0.5);
        assert_eq!(blend_half, [0.5, 0.5, 0.0]);
    }

    // ========================================================================
    // Checkerboard tests
    // ========================================================================

    #[test]
    fn test_checkerboard_phase_from_frame() {
        assert_eq!(CheckerboardPhase::from_frame(0), CheckerboardPhase::Even);
        assert_eq!(CheckerboardPhase::from_frame(1), CheckerboardPhase::Odd);
        assert_eq!(CheckerboardPhase::from_frame(2), CheckerboardPhase::Even);
    }

    #[test]
    fn test_checkerboard_phase_other() {
        assert_eq!(CheckerboardPhase::Even.other(), CheckerboardPhase::Odd);
        assert_eq!(CheckerboardPhase::Odd.other(), CheckerboardPhase::Even);
    }

    #[test]
    fn test_checkerboard_should_render() {
        let even = CheckerboardPhase::Even;
        assert!(even.should_render(0, 0));
        assert!(!even.should_render(1, 0));
        assert!(!even.should_render(0, 1));
        assert!(even.should_render(1, 1));
        assert!(even.should_render(2, 2));

        let odd = CheckerboardPhase::Odd;
        assert!(!odd.should_render(0, 0));
        assert!(odd.should_render(1, 0));
        assert!(odd.should_render(0, 1));
        assert!(!odd.should_render(1, 1));
    }

    #[test]
    fn test_checkerboard_half_pixels() {
        let even = CheckerboardPhase::Even;
        let odd = CheckerboardPhase::Odd;

        let mut even_count = 0;
        let mut odd_count = 0;

        for y in 0..10 {
            for x in 0..10 {
                if even.should_render(x, y) {
                    even_count += 1;
                }
                if odd.should_render(x, y) {
                    odd_count += 1;
                }
            }
        }

        // Should render exactly half the pixels each
        assert_eq!(even_count, 50);
        assert_eq!(odd_count, 50);
    }

    #[test]
    fn test_checkerboard_reconstruct_new() {
        let reconstruct = CheckerboardReconstruct::new(0);
        assert_eq!(reconstruct.phase, CheckerboardPhase::Even);
        assert_eq!(reconstruct.frame, 0);
    }

    #[test]
    fn test_checkerboard_needs_reconstruction() {
        let reconstruct = CheckerboardReconstruct::new(0);
        assert!(!reconstruct.needs_reconstruction(0, 0)); // Even pixel
        assert!(reconstruct.needs_reconstruction(1, 0));  // Odd pixel
    }

    #[test]
    fn test_checkerboard_blend_rendered() {
        let reconstruct = CheckerboardReconstruct::new(0);
        let current = [1.0, 0.0, 0.0];
        let history = [0.0, 1.0, 0.0];
        let blended = reconstruct.blend_rendered(current, history, 0.5);
        assert_eq!(blended, [0.5, 0.5, 0.0]);
    }

    #[test]
    fn test_checkerboard_reconstruct_pixel() {
        let reconstruct = CheckerboardReconstruct::new(0);
        let neighbors = [
            [1.0, 0.0, 0.0], // left
            [1.0, 0.0, 0.0], // right
            [0.0, 1.0, 0.0], // up
            [0.0, 1.0, 0.0], // down
        ];
        let history = [0.5, 0.5, 0.5];

        let result = reconstruct.reconstruct_pixel(1, 0, &neighbors, history, true);
        // Should blend spatial average with history
        assert!(result[0] > 0.0 && result[0] < 1.0);
    }

    // ========================================================================
    // Bilinear sampling tests
    // ========================================================================

    #[test]
    fn test_bilinear_sample_corners() {
        let samples = [
            [1.0, 0.0, 0.0], // top-left
            [0.0, 1.0, 0.0], // top-right
            [0.0, 0.0, 1.0], // bottom-left
            [1.0, 1.0, 1.0], // bottom-right
        ];

        // At top-left corner
        let tl = bilinear_sample(&samples, [0.0, 0.0]);
        assert!((tl[0] - 1.0).abs() < 1e-5);
        assert!((tl[1] - 0.0).abs() < 1e-5);
        assert!((tl[2] - 0.0).abs() < 1e-5);

        // At bottom-right corner
        let br = bilinear_sample(&samples, [1.0, 1.0]);
        assert!((br[0] - 1.0).abs() < 1e-5);
        assert!((br[1] - 1.0).abs() < 1e-5);
        assert!((br[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_bilinear_sample_center() {
        let samples = [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ];

        let center = bilinear_sample(&samples, [0.5, 0.5]);
        assert!((center[0] - 0.5).abs() < 1e-5);
        assert!((center[1] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_bilinear_coordinates() {
        let (base_x, base_y, frac_x, frac_y) = bilinear_coordinates([0.5, 0.5], [1920.0, 1080.0]);
        assert_eq!(base_x, 959);
        assert_eq!(base_y, 539);
        assert!(frac_x >= 0.0 && frac_x < 1.0);
        assert!(frac_y >= 0.0 && frac_y < 1.0);
    }

    #[test]
    fn test_bilinear_coordinates_edge() {
        let (base_x, base_y, frac_x, frac_y) = bilinear_coordinates([0.0, 0.0], [100.0, 100.0]);
        assert_eq!(base_x, -1);
        assert!(frac_x >= 0.0);
    }

    // ========================================================================
    // Matrix utility tests
    // ========================================================================

    #[test]
    fn test_mat4_transform_vec4() {
        let identity = IDENTITY_MATRIX;
        let v = [1.0, 2.0, 3.0, 1.0];
        let result = mat4_transform_vec4(&identity, v);
        assert_eq!(result, v);
    }

    #[test]
    fn test_mat4_transform_vec4_scale() {
        let scale = [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let v = [1.0, 2.0, 3.0, 1.0];
        let result = mat4_transform_vec4(&scale, v);
        assert_eq!(result, [2.0, 4.0, 6.0, 1.0]);
    }

    #[test]
    fn test_mat4_transform_point() {
        let identity = IDENTITY_MATRIX;
        let p = [1.0, 2.0, 3.0, 1.0];
        let result = mat4_transform_point(&identity, p);
        assert!(result.is_some());
        let r = result.unwrap();
        assert_eq!(r, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_mat4_transform_point_perspective() {
        let perspective = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 2.0], // w=2
        ];
        let p = [2.0, 4.0, 6.0, 1.0];
        let result = mat4_transform_point(&perspective, p).unwrap();
        assert_eq!(result, [1.0, 2.0, 3.0]); // Divided by w=2
    }

    #[test]
    fn test_mat4_transform_point_behind_camera() {
        let m = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0], // w=0
        ];
        let p = [1.0, 2.0, 3.0, 1.0];
        assert!(mat4_transform_point(&m, p).is_none());
    }

    #[test]
    fn test_mat4_mul_identity() {
        let result = mat4_mul(&IDENTITY_MATRIX, &IDENTITY_MATRIX);
        assert_eq!(result, IDENTITY_MATRIX);
    }

    #[test]
    fn test_mat4_mul_scale() {
        let scale2 = [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let result = mat4_mul(&scale2, &scale2);
        assert_eq!(result[0][0], 4.0);
        assert_eq!(result[1][1], 4.0);
        assert_eq!(result[2][2], 4.0);
    }

    #[test]
    fn test_mat4_inverse_identity() {
        let inv = mat4_inverse(&IDENTITY_MATRIX);
        assert!(inv.is_some());
        let result = inv.unwrap();
        for i in 0..4 {
            for j in 0..4 {
                if i == j {
                    assert!((result[i][j] - 1.0).abs() < 1e-5);
                } else {
                    assert!(result[i][j].abs() < 1e-5);
                }
            }
        }
    }

    #[test]
    fn test_mat4_inverse_scale() {
        let scale2 = [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let inv = mat4_inverse(&scale2).unwrap();
        assert!((inv[0][0] - 0.5).abs() < 1e-5);
        assert!((inv[1][1] - 0.5).abs() < 1e-5);
        assert!((inv[2][2] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_mat4_inverse_singular() {
        let singular = [[0.0; 4]; 4];
        assert!(mat4_inverse(&singular).is_none());
    }

    #[test]
    fn test_mat4_inverse_roundtrip() {
        let m = [
            [1.0, 2.0, 3.0, 4.0],
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 0.0, 1.0, 2.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let inv = mat4_inverse(&m).unwrap();
        let result = mat4_mul(&m, &inv);

        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (result[i][j] - expected).abs() < 1e-4,
                    "result[{}][{}] = {}, expected {}",
                    i, j, result[i][j], expected
                );
            }
        }
    }

    // ========================================================================
    // Motion vector tests
    // ========================================================================

    #[test]
    fn test_calculate_motion_vector_stationary() {
        let prev = IDENTITY_MATRIX;
        let curr_inv = IDENTITY_MATRIX;
        let pixel = [960.0, 540.0];
        let resolution = [1920.0, 1080.0];

        let mv = calculate_motion_vector(pixel, resolution, &prev, &curr_inv, 10000.0);
        // With identical matrices, motion should be ~zero
        assert!(mv.is_some());
        let [dx, dy] = mv.unwrap();
        assert!(dx.abs() < 10.0); // Allow some tolerance
        assert!(dy.abs() < 10.0);
    }

    #[test]
    fn test_reproject_uv_stationary() {
        let prev = IDENTITY_MATRIX;
        let curr_inv = IDENTITY_MATRIX;
        let uv = [0.5, 0.5];

        let result = reproject_uv(uv, &prev, &curr_inv, 10000.0);
        assert!(result.is_some());
        // With identical matrices, UV should be similar
    }

    #[test]
    fn test_reproject_uv_out_of_bounds() {
        // Create a matrix that projects behind the camera
        let extreme = [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ];

        let result = reproject_uv([0.5, 0.5], &extreme, &IDENTITY_MATRIX, 10000.0);
        assert!(result.is_none());
    }

    // ========================================================================
    // CloudTemporalState tests
    // ========================================================================

    #[test]
    fn test_temporal_state_new() {
        let state = CloudTemporalState::new();
        assert_eq!(state.uniforms.frame_index, 0);
        assert!(!state.history.valid);
    }

    #[test]
    fn test_temporal_state_with_resolution() {
        let state = CloudTemporalState::with_resolution(3840, 2160);
        assert_eq!(state.uniforms.resolution, [3840.0, 2160.0]);
        assert_eq!(state.history.resolution, (3840, 2160));
    }

    #[test]
    fn test_temporal_state_update() {
        let mut state = CloudTemporalState::new();
        assert_eq!(state.uniforms.frame_index, 0);
        assert!(!state.history.valid);

        state.update(IDENTITY_MATRIX);
        assert_eq!(state.uniforms.frame_index, 1);
        assert!(state.history.valid);
        assert!(state.curr_view_proj_inv.is_some());
    }

    #[test]
    fn test_temporal_state_set_resolution() {
        let mut state = CloudTemporalState::with_resolution(1920, 1080);
        state.update(IDENTITY_MATRIX);
        assert!(state.history.valid);

        state.set_resolution(3840, 2160);
        assert!(!state.history.valid); // Should invalidate
        assert_eq!(state.uniforms.resolution, [3840.0, 2160.0]);
    }

    #[test]
    fn test_temporal_state_invalidate() {
        let mut state = CloudTemporalState::new();
        state.update(IDENTITY_MATRIX);
        assert!(state.is_history_valid());

        state.invalidate();
        assert!(!state.is_history_valid());
    }

    #[test]
    fn test_temporal_state_reproject() {
        let mut state = CloudTemporalState::new();
        state.update(IDENTITY_MATRIX);

        let result = state.reproject([0.5, 0.5]);
        assert!(result.is_some());
    }

    #[test]
    fn test_temporal_state_frame_index() {
        let mut state = CloudTemporalState::new();
        assert_eq!(state.frame_index(), 0);

        state.update(IDENTITY_MATRIX);
        assert_eq!(state.frame_index(), 1);

        state.update(IDENTITY_MATRIX);
        assert_eq!(state.frame_index(), 2);
    }

    // ========================================================================
    // Additional edge case tests
    // ========================================================================

    #[test]
    fn test_uniforms_frame_wrapping() {
        let mut uniforms = CloudTemporalUniforms::default();
        uniforms.frame_index = u32::MAX;
        uniforms.advance_frame(IDENTITY_MATRIX);
        assert_eq!(uniforms.frame_index, 0); // Should wrap
    }

    #[test]
    fn test_history_frames_accumulated_saturation() {
        let mut state = HistoryBufferState::default();
        state.frames_accumulated = u32::MAX;
        state.advance();
        assert_eq!(state.frames_accumulated, u32::MAX); // Saturating add
    }

    #[test]
    fn test_jitter_clamped_length() {
        let jitter = JitterSequence::halton(1000); // Way over limit
        assert_eq!(jitter.len(), MAX_JITTER_SEQUENCE_LENGTH as usize);
    }

    #[test]
    fn test_neighborhood_stats_high_variance() {
        let samples = [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5],
        ];
        let stats = NeighborhoodStats::from_3x3(&samples);

        // High variance should result in low confidence
        let confidence = stats.confidence(0.1);
        assert!(confidence < 0.5);
    }

    #[test]
    fn test_aabb_expand_zero() {
        let mut aabb = ColorAABB {
            min: [0.5, 0.5, 0.5],
            max: [0.5, 0.5, 0.5],
        };
        aabb.expand(2.0);
        // Expanding a zero-size AABB should still result in zero size
        assert_eq!(aabb.min, [0.5, 0.5, 0.5]);
        assert_eq!(aabb.max, [0.5, 0.5, 0.5]);
    }

    #[test]
    fn test_bilinear_sample_uniform() {
        let samples = [[0.5, 0.5, 0.5]; 4];
        let result = bilinear_sample(&samples, [0.3, 0.7]);
        assert_eq!(result, [0.5, 0.5, 0.5]);
    }

    #[test]
    fn test_checkerboard_neighbor_offset() {
        let phase = CheckerboardPhase::Even;
        let (dx, dy) = phase.neighbor_offset();
        assert_eq!((dx, dy), (1, 0));
    }

    #[test]
    fn test_checkerboard_history_offset() {
        let reconstruct = CheckerboardReconstruct::new(0);
        let offset = reconstruct.history_offset(0, 0, [1920.0, 1080.0]);
        assert_eq!(offset, [0.0, 0.0]);
    }

    // ========================================================================
    // GPU struct alignment tests
    // ========================================================================

    #[test]
    fn test_uniforms_byte_offset_prev_view_proj() {
        let uniforms = CloudTemporalUniforms::default();
        let base = &uniforms as *const _ as usize;
        let field = &uniforms.prev_view_proj as *const _ as usize;
        assert_eq!(field - base, 0);
    }

    #[test]
    fn test_uniforms_byte_offset_curr_view_proj() {
        let uniforms = CloudTemporalUniforms::default();
        let base = &uniforms as *const _ as usize;
        let field = &uniforms.curr_view_proj as *const _ as usize;
        assert_eq!(field - base, 64);
    }

    #[test]
    fn test_uniforms_byte_offset_jitter() {
        let uniforms = CloudTemporalUniforms::default();
        let base = &uniforms as *const _ as usize;
        let field = &uniforms.jitter_offset as *const _ as usize;
        assert_eq!(field - base, 128);
    }

    #[test]
    fn test_uniforms_byte_offset_blend_factor() {
        let uniforms = CloudTemporalUniforms::default();
        let base = &uniforms as *const _ as usize;
        let field = &uniforms.blend_factor as *const _ as usize;
        assert_eq!(field - base, 136);
    }

    #[test]
    fn test_uniforms_16_byte_multiple() {
        let size = std::mem::size_of::<CloudTemporalUniforms>();
        assert_eq!(size % 16, 0, "Size {} is not a multiple of 16", size);
    }
}
