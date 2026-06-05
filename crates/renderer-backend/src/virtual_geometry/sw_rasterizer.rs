//! Software Rasterizer for TRINITY Engine (T-GPU-8.2).
//!
//! Compute shader-based triangle rasterization for virtual geometry systems.
//! Software rasterization in compute shaders handles tiny triangles more
//! efficiently than hardware rasterization, avoiding quad overshading.
//!
//! # Overview
//!
//! The software rasterizer pipeline:
//! 1. Project triangles from clip space to screen space
//! 2. Compute bounding boxes and cull backfacing/degenerate triangles
//! 3. Set up edge functions (Pineda's algorithm)
//! 4. Iterate pixels within bounding box using tile-based approach
//! 5. Test each pixel using edge functions
//! 6. Perform atomic depth test via atomicMin
//! 7. Write to visibility buffer on depth test pass
//!
//! # Performance
//!
//! - Work complexity: O(n * area) where n = triangles, area = avg triangle pixels
//! - Target: < 0.1ms for 50K triangles on RTX 3080
//! - Best for: triangles < 8x8 pixels (sub-pixel to small)
//!
//! # Usage
//!
//! ```ignore
//! use renderer_backend::virtual_geometry::{SoftwareRasterizerPipeline, RasterizerResources};
//!
//! let pipeline = SoftwareRasterizerPipeline::new(&device);
//! let resources = RasterizerResources::new(&device, 1920, 1080, 100_000);
//!
//! // Upload triangles and metadata
//! resources.upload_triangles(&queue, &clip_triangles);
//! resources.upload_metadata(&queue, &triangle_meta);
//!
//! // Dispatch rasterization
//! let params = RasterizerParams::new(1920.0, 1080.0, num_triangles);
//! pipeline.dispatch(&mut encoder, &resources, &params);
//! ```

use std::mem;

use bytemuck::{Pod, Zeroable};
use wgpu::{Buffer, BufferUsages, Device, Queue};

// =============================================================================
// CONSTANTS
// =============================================================================

/// Compute shader workgroup size for tiled rasterization.
pub const TILE_WORKGROUP_SIZE: u32 = 64;

/// Compute shader workgroup size for linear (small triangle) rasterization.
pub const LINEAR_WORKGROUP_SIZE: u32 = 256;

/// Tile size in pixels for tile-based rasterization.
pub const DEFAULT_TILE_SIZE: u32 = 8;

/// Invalid instance ID sentinel value.
pub const INVALID_INSTANCE_ID: u32 = 0xFFFF_FFFF;

/// Invalid primitive ID sentinel value.
pub const INVALID_PRIMITIVE_ID: u32 = 0x00FF_FFFF; // 24 bits max

/// Default maximum triangles.
pub const DEFAULT_MAX_TRIANGLES: u32 = 262_144; // 256K triangles

/// Minimum triangle area in pixels (degenerate threshold).
pub const MIN_TRIANGLE_AREA: f32 = 0.0001;

/// Depth clear value (far plane in u32 bits).
pub const DEPTH_CLEAR_VALUE: u32 = 0xFFFF_FFFF;

/// Flag: enable depth testing.
pub const FLAG_DEPTH_TEST: u32 = 1;

/// Flag: enable backface culling.
pub const FLAG_BACKFACE_CULL: u32 = 2;

// =============================================================================
// RASTERIZER PARAMS
// =============================================================================

/// GPU uniform buffer for software rasterizer parameters.
///
/// Matches the WGSL `RasterizerParams` struct layout (64 bytes).
///
/// # Memory Layout
///
/// 64 bytes, std140/std430 compatible:
/// | Offset | Field          | Size |
/// |--------|----------------|------|
/// | 0      | viewport       | 16   |
/// | 16     | depth_bias     | 4    |
/// | 20     | tile_size      | 4    |
/// | 24     | num_triangles  | 4    |
/// | 28     | screen_width   | 4    |
/// | 32     | screen_height  | 4    |
/// | 36     | near_plane     | 4    |
/// | 40     | far_plane      | 4    |
/// | 44     | flags          | 4    |
/// | 48     | _pad           | 16   |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct RasterizerParams {
    /// Viewport: (x, y, width, height).
    pub viewport: [f32; 4],
    /// Depth bias for shadow mapping / z-fighting avoidance.
    pub depth_bias: f32,
    /// Tile size in pixels (8 or 16).
    pub tile_size: u32,
    /// Number of triangles to rasterize.
    pub num_triangles: u32,
    /// Screen width in pixels.
    pub screen_width: u32,
    /// Screen height in pixels.
    pub screen_height: u32,
    /// Near plane depth value.
    pub near_plane: f32,
    /// Far plane depth value.
    pub far_plane: f32,
    /// Flags: bit 0 = enable depth test, bit 1 = enable backface cull.
    pub flags: u32,
    /// Padding for 64-byte alignment.
    pub _pad: [f32; 4],
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<RasterizerParams>() == 64);

impl RasterizerParams {
    /// Create parameters for given screen dimensions and triangle count.
    pub fn new(screen_width: f32, screen_height: f32, num_triangles: u32) -> Self {
        Self {
            viewport: [0.0, 0.0, screen_width, screen_height],
            depth_bias: 0.0,
            tile_size: DEFAULT_TILE_SIZE,
            num_triangles,
            screen_width: screen_width as u32,
            screen_height: screen_height as u32,
            near_plane: 0.0,
            far_plane: 1.0,
            flags: FLAG_DEPTH_TEST | FLAG_BACKFACE_CULL,
            _pad: [0.0; 4],
        }
    }

    /// Create parameters with custom viewport.
    pub fn with_viewport(
        viewport_x: f32,
        viewport_y: f32,
        viewport_width: f32,
        viewport_height: f32,
        screen_width: u32,
        screen_height: u32,
        num_triangles: u32,
    ) -> Self {
        Self {
            viewport: [viewport_x, viewport_y, viewport_width, viewport_height],
            depth_bias: 0.0,
            tile_size: DEFAULT_TILE_SIZE,
            num_triangles,
            screen_width,
            screen_height,
            near_plane: 0.0,
            far_plane: 1.0,
            flags: FLAG_DEPTH_TEST | FLAG_BACKFACE_CULL,
            _pad: [0.0; 4],
        }
    }

    /// Set depth bias for shadow mapping.
    #[inline]
    pub fn with_depth_bias(mut self, bias: f32) -> Self {
        self.depth_bias = bias;
        self
    }

    /// Set tile size (8 or 16).
    #[inline]
    pub fn with_tile_size(mut self, size: u32) -> Self {
        self.tile_size = size;
        self
    }

    /// Enable or disable depth testing.
    #[inline]
    pub fn with_depth_test(mut self, enabled: bool) -> Self {
        if enabled {
            self.flags |= FLAG_DEPTH_TEST;
        } else {
            self.flags &= !FLAG_DEPTH_TEST;
        }
        self
    }

    /// Enable or disable backface culling.
    #[inline]
    pub fn with_backface_cull(mut self, enabled: bool) -> Self {
        if enabled {
            self.flags |= FLAG_BACKFACE_CULL;
        } else {
            self.flags &= !FLAG_BACKFACE_CULL;
        }
        self
    }

    /// Get the number of workgroups for tiled dispatch.
    #[inline]
    pub fn num_workgroups_tiled(&self) -> u32 {
        self.num_triangles
    }

    /// Get the number of workgroups for linear dispatch.
    #[inline]
    pub fn num_workgroups_linear(&self) -> u32 {
        (self.num_triangles + LINEAR_WORKGROUP_SIZE - 1) / LINEAR_WORKGROUP_SIZE
    }

    /// Get the number of workgroups for clearing depth/visibility.
    #[inline]
    pub fn num_workgroups_clear(&self) -> u32 {
        let total_pixels = self.screen_width * self.screen_height;
        (total_pixels + LINEAR_WORKGROUP_SIZE - 1) / LINEAR_WORKGROUP_SIZE
    }

    /// Check if no work is needed.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.num_triangles == 0
    }

    /// Check if depth testing is enabled.
    #[inline]
    pub fn depth_test_enabled(&self) -> bool {
        (self.flags & FLAG_DEPTH_TEST) != 0
    }

    /// Check if backface culling is enabled.
    #[inline]
    pub fn backface_cull_enabled(&self) -> bool {
        (self.flags & FLAG_BACKFACE_CULL) != 0
    }
}

// =============================================================================
// CLIP SPACE TRIANGLE
// =============================================================================

/// Clip-space triangle with 4D homogeneous coordinates.
///
/// # Memory Layout
///
/// 48 bytes:
/// | Offset | Field | Size |
/// |--------|-------|------|
/// | 0      | v0    | 16   |
/// | 16     | v1    | 16   |
/// | 32     | v2    | 16   |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct ClipSpaceTriangle {
    /// Vertex 0 in clip space (x, y, z, w).
    pub v0: [f32; 4],
    /// Vertex 1 in clip space (x, y, z, w).
    pub v1: [f32; 4],
    /// Vertex 2 in clip space (x, y, z, w).
    pub v2: [f32; 4],
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<ClipSpaceTriangle>() == 48);

impl ClipSpaceTriangle {
    /// Create a new clip-space triangle.
    pub const fn new(v0: [f32; 4], v1: [f32; 4], v2: [f32; 4]) -> Self {
        Self { v0, v1, v2 }
    }

    /// Create from separate x, y, z, w components.
    pub const fn from_components(
        x0: f32, y0: f32, z0: f32, w0: f32,
        x1: f32, y1: f32, z1: f32, w1: f32,
        x2: f32, y2: f32, z2: f32, w2: f32,
    ) -> Self {
        Self {
            v0: [x0, y0, z0, w0],
            v1: [x1, y1, z1, w1],
            v2: [x2, y2, z2, w2],
        }
    }

    /// Get vertex by index (0, 1, or 2).
    #[inline]
    pub fn vertex(&self, index: usize) -> [f32; 4] {
        match index {
            0 => self.v0,
            1 => self.v1,
            2 => self.v2,
            _ => panic!("vertex index out of bounds"),
        }
    }

    /// Check if any vertex has w <= 0 (behind camera).
    #[inline]
    pub fn is_behind_camera(&self) -> bool {
        self.v0[3] <= 0.0 || self.v1[3] <= 0.0 || self.v2[3] <= 0.0
    }
}

// =============================================================================
// TRIANGLE METADATA
// =============================================================================

/// Triangle metadata for visibility buffer output.
///
/// # Memory Layout
///
/// 8 bytes:
/// | Offset | Field        | Size |
/// |--------|--------------|------|
/// | 0      | instance_id  | 4    |
/// | 4      | primitive_id | 4    |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct TriangleMeta {
    /// Instance ID for material/transform lookup.
    pub instance_id: u32,
    /// Primitive ID within the mesh.
    pub primitive_id: u32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<TriangleMeta>() == 8);

impl TriangleMeta {
    /// Create new triangle metadata.
    pub const fn new(instance_id: u32, primitive_id: u32) -> Self {
        Self { instance_id, primitive_id }
    }

    /// Create an invalid metadata entry.
    pub const fn invalid() -> Self {
        Self {
            instance_id: INVALID_INSTANCE_ID,
            primitive_id: INVALID_PRIMITIVE_ID,
        }
    }

    /// Check if this entry is valid.
    #[inline]
    pub const fn is_valid(&self) -> bool {
        self.instance_id != INVALID_INSTANCE_ID
    }
}

// =============================================================================
// RASTERIZER TILE
// =============================================================================

/// Tile for tile-based triangle binning.
///
/// Used for hierarchical rasterization where triangles are first binned
/// to tiles, then each tile is processed by a workgroup.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct RasterizerTile {
    /// Tile X coordinate (in tiles, not pixels).
    pub tile_x: u32,
    /// Tile Y coordinate (in tiles, not pixels).
    pub tile_y: u32,
    /// Index of first triangle in this tile.
    pub first_triangle: u32,
    /// Number of triangles in this tile.
    pub triangle_count: u32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<RasterizerTile>() == 16);

impl RasterizerTile {
    /// Create a new tile entry.
    pub const fn new(tile_x: u32, tile_y: u32, first_triangle: u32, triangle_count: u32) -> Self {
        Self {
            tile_x,
            tile_y,
            first_triangle,
            triangle_count,
        }
    }

    /// Get pixel coordinates of tile's top-left corner.
    #[inline]
    pub const fn pixel_origin(&self, tile_size: u32) -> (u32, u32) {
        (self.tile_x * tile_size, self.tile_y * tile_size)
    }

    /// Check if tile is empty.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.triangle_count == 0
    }
}

// =============================================================================
// RASTERIZER STATISTICS
// =============================================================================

/// Statistics from software rasterization.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct RasterizerStats {
    /// Number of triangles processed.
    pub triangles_processed: u32,
    /// Number of triangles culled (backface/degenerate).
    pub triangles_culled: u32,
    /// Number of fragments tested.
    pub fragments_tested: u32,
    /// Number of fragments written.
    pub fragments_written: u32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<RasterizerStats>() == 16);

impl RasterizerStats {
    /// Get cull rate as percentage.
    #[inline]
    pub fn cull_rate(&self) -> f32 {
        if self.triangles_processed == 0 {
            0.0
        } else {
            self.triangles_culled as f32 / (self.triangles_processed + self.triangles_culled) as f32
        }
    }

    /// Get depth test pass rate as percentage.
    #[inline]
    pub fn depth_pass_rate(&self) -> f32 {
        if self.fragments_tested == 0 {
            0.0
        } else {
            self.fragments_written as f32 / self.fragments_tested as f32
        }
    }

    /// Get average fragments per triangle.
    #[inline]
    pub fn avg_fragments_per_triangle(&self) -> f32 {
        if self.triangles_processed == 0 {
            0.0
        } else {
            self.fragments_tested as f32 / self.triangles_processed as f32
        }
    }
}

// =============================================================================
// BOUNDING BOX
// =============================================================================

/// Screen-space bounding box.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BoundingBox {
    /// Minimum X coordinate (inclusive).
    pub min_x: i32,
    /// Minimum Y coordinate (inclusive).
    pub min_y: i32,
    /// Maximum X coordinate (inclusive).
    pub max_x: i32,
    /// Maximum Y coordinate (inclusive).
    pub max_y: i32,
}

impl BoundingBox {
    /// Create a new bounding box.
    pub const fn new(min_x: i32, min_y: i32, max_x: i32, max_y: i32) -> Self {
        Self { min_x, min_y, max_x, max_y }
    }

    /// Check if bounding box is empty (no pixels).
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.max_x < self.min_x || self.max_y < self.min_y
    }

    /// Get width in pixels.
    #[inline]
    pub const fn width(&self) -> i32 {
        if self.is_empty() { 0 } else { self.max_x - self.min_x + 1 }
    }

    /// Get height in pixels.
    #[inline]
    pub const fn height(&self) -> i32 {
        if self.is_empty() { 0 } else { self.max_y - self.min_y + 1 }
    }

    /// Get area in pixels.
    #[inline]
    pub const fn area(&self) -> i32 {
        self.width() * self.height()
    }

    /// Clamp bounding box to screen bounds.
    pub fn clamp(&self, screen_width: u32, screen_height: u32) -> Self {
        Self {
            min_x: self.min_x.max(0),
            min_y: self.min_y.max(0),
            max_x: self.max_x.min(screen_width as i32 - 1),
            max_y: self.max_y.min(screen_height as i32 - 1),
        }
    }
}

// =============================================================================
// CPU REFERENCE IMPLEMENTATIONS
// =============================================================================

/// Compute 2D edge function (orient2d).
///
/// E(p) = (p.x - v0.x) * (v1.y - v0.y) - (p.y - v0.y) * (v1.x - v0.x)
///
/// Returns:
/// - Positive if p is to the left of edge (v0 -> v1)
/// - Negative if p is to the right of edge
/// - Zero if p is on the edge
#[inline]
pub fn cpu_edge_function(v0: (f32, f32), v1: (f32, f32), p: (f32, f32)) -> f32 {
    (p.0 - v0.0) * (v1.1 - v0.1) - (p.1 - v0.1) * (v1.0 - v0.0)
}

/// Compute bounding box for a triangle.
pub fn cpu_triangle_bbox(
    p0: (f32, f32),
    p1: (f32, f32),
    p2: (f32, f32),
    screen_width: u32,
    screen_height: u32,
) -> BoundingBox {
    let min_x = p0.0.min(p1.0).min(p2.0);
    let min_y = p0.1.min(p1.1).min(p2.1);
    let max_x = p0.0.max(p1.0).max(p2.0);
    let max_y = p0.1.max(p1.1).max(p2.1);

    BoundingBox {
        min_x: (min_x.floor() as i32).max(0),
        min_y: (min_y.floor() as i32).max(0),
        max_x: (max_x.ceil() as i32).min(screen_width as i32 - 1),
        max_y: (max_y.ceil() as i32).min(screen_height as i32 - 1),
    }
}

/// CPU reference depth test.
///
/// Returns true if the new depth passes (is closer in reverse-Z).
#[inline]
pub fn cpu_depth_test(new_depth: f32, old_depth: f32) -> bool {
    new_depth < old_depth
}

/// Encode depth as u32 for atomic comparison.
#[inline]
pub fn cpu_encode_depth(depth: f32) -> u32 {
    depth.clamp(0.0, 1.0).to_bits()
}

/// Decode u32 depth back to float.
#[inline]
pub fn cpu_decode_depth(encoded: u32) -> f32 {
    f32::from_bits(encoded)
}

/// Project clip-space vertex to screen-space.
pub fn cpu_project_vertex(clip: [f32; 4], viewport: [f32; 4]) -> (f32, f32, f32) {
    let inv_w = 1.0 / clip[3];
    let ndc_x = clip[0] * inv_w;
    let ndc_y = clip[1] * inv_w;
    let ndc_z = clip[2] * inv_w;

    let screen_x = (ndc_x * 0.5 + 0.5) * viewport[2] + viewport[0];
    let screen_y = (ndc_y * 0.5 + 0.5) * viewport[3] + viewport[1];

    (screen_x, screen_y, ndc_z)
}

/// Check if triangle is backfacing (CW winding in screen space).
#[inline]
pub fn cpu_is_backfacing(p0: (f32, f32), p1: (f32, f32), p2: (f32, f32)) -> bool {
    cpu_edge_function(p0, p1, p2) <= 0.0
}

/// Compute triangle area in pixels.
#[inline]
pub fn cpu_triangle_area(p0: (f32, f32), p1: (f32, f32), p2: (f32, f32)) -> f32 {
    cpu_edge_function(p0, p1, p2).abs() * 0.5
}

/// Pack instance_id and primitive_id for visibility buffer.
#[inline]
pub fn cpu_pack_visibility(instance_id: u32, primitive_id: u32) -> u32 {
    ((instance_id & 0xFFFF) << 16) | (primitive_id & 0xFFFF)
}

/// Unpack instance_id from visibility buffer.
#[inline]
pub fn cpu_unpack_instance_id(packed: u32) -> u32 {
    packed >> 16
}

/// Unpack primitive_id from visibility buffer.
#[inline]
pub fn cpu_unpack_primitive_id(packed: u32) -> u32 {
    packed & 0xFFFF
}

/// Compute barycentric coordinates for a point inside a triangle.
pub fn cpu_compute_barycentrics(
    p0: (f32, f32),
    p1: (f32, f32),
    p2: (f32, f32),
    point: (f32, f32),
) -> Option<(f32, f32, f32)> {
    let area = cpu_edge_function(p0, p1, p2);

    if area.abs() < MIN_TRIANGLE_AREA {
        return None;
    }

    let e0 = cpu_edge_function(p1, p2, point);
    let e1 = cpu_edge_function(p2, p0, point);
    let e2 = cpu_edge_function(p0, p1, point);

    let area_inv = 1.0 / area;
    let b0 = e0 * area_inv;
    let b1 = e1 * area_inv;
    let b2 = e2 * area_inv;

    Some((b0, b1, b2))
}

/// Interpolate depth using barycentric coordinates.
#[inline]
pub fn cpu_interpolate_depth(bary: (f32, f32, f32), z0: f32, z1: f32, z2: f32) -> f32 {
    bary.0 * z0 + bary.1 * z1 + bary.2 * z2
}

/// Perspective-correct depth interpolation.
pub fn cpu_interpolate_depth_perspective(
    bary: (f32, f32, f32),
    z0: f32, z1: f32, z2: f32,
    inv_w0: f32, inv_w1: f32, inv_w2: f32,
) -> f32 {
    let inv_w = bary.0 * inv_w0 + bary.1 * inv_w1 + bary.2 * inv_w2;
    let z_over_w = bary.0 * (z0 * inv_w0) + bary.1 * (z1 * inv_w1) + bary.2 * (z2 * inv_w2);
    z_over_w / inv_w
}

/// CPU reference implementation for software rasterization of a single triangle.
///
/// Returns a list of (pixel_x, pixel_y, depth, instance_id, primitive_id) for all
/// fragments that pass the edge function test.
pub fn cpu_rasterize_triangle(
    triangle: &ClipSpaceTriangle,
    meta: &TriangleMeta,
    viewport: [f32; 4],
    screen_width: u32,
    screen_height: u32,
    backface_cull: bool,
) -> Vec<(u32, u32, f32, u32, u32)> {
    let mut fragments = Vec::new();

    // Skip invalid triangles
    if !meta.is_valid() || triangle.is_behind_camera() {
        return fragments;
    }

    // Project vertices
    let (sx0, sy0, z0) = cpu_project_vertex(triangle.v0, viewport);
    let (sx1, sy1, z1) = cpu_project_vertex(triangle.v1, viewport);
    let (sx2, sy2, z2) = cpu_project_vertex(triangle.v2, viewport);

    let p0 = (sx0, sy0);
    let p1 = (sx1, sy1);
    let p2 = (sx2, sy2);

    // Backface culling
    if backface_cull && cpu_is_backfacing(p0, p1, p2) {
        return fragments;
    }

    // Degenerate check
    let area = cpu_edge_function(p0, p1, p2);
    if area.abs() < MIN_TRIANGLE_AREA {
        return fragments;
    }

    // Compute bounding box
    let bbox = cpu_triangle_bbox(p0, p1, p2, screen_width, screen_height);

    if bbox.is_empty() {
        return fragments;
    }

    let area_inv = 1.0 / area;
    let inv_w0 = 1.0 / triangle.v0[3];
    let inv_w1 = 1.0 / triangle.v1[3];
    let inv_w2 = 1.0 / triangle.v2[3];

    // Iterate over bounding box
    for py in bbox.min_y..=bbox.max_y {
        for px in bbox.min_x..=bbox.max_x {
            let point = (px as f32 + 0.5, py as f32 + 0.5);

            // Edge function tests
            let e0 = cpu_edge_function(p1, p2, point);
            let e1 = cpu_edge_function(p2, p0, point);
            let e2 = cpu_edge_function(p0, p1, point);

            // Check if inside (all same sign)
            let inside = (e0 >= 0.0 && e1 >= 0.0 && e2 >= 0.0) ||
                         (e0 <= 0.0 && e1 <= 0.0 && e2 <= 0.0);

            if !inside {
                continue;
            }

            // Compute barycentrics
            let bary = (e0 * area_inv, e1 * area_inv, e2 * area_inv);

            // Interpolate depth with perspective correction
            let depth = cpu_interpolate_depth_perspective(bary, z0, z1, z2, inv_w0, inv_w1, inv_w2);

            fragments.push((px as u32, py as u32, depth, meta.instance_id, meta.primitive_id));
        }
    }

    fragments
}

// =============================================================================
// RASTERIZER RESOURCES
// =============================================================================

/// GPU resources for software rasterizer operations.
pub struct RasterizerResources {
    /// Uniform buffer for parameters.
    pub params_buffer: Buffer,
    /// Input triangles in clip space.
    pub triangles_buffer: Buffer,
    /// Triangle metadata (instance_id, primitive_id).
    pub metadata_buffer: Buffer,
    /// Depth buffer (atomic u32).
    pub depth_buffer: Buffer,
    /// Visibility buffer output (atomic u32).
    pub visibility_buffer: Buffer,
    /// Statistics counters (4 x atomic u32).
    pub stats_buffer: Buffer,
    /// Staging buffer for reading stats.
    pub stats_staging: Buffer,
    /// Maximum triangles.
    pub max_triangles: u32,
    /// Screen width.
    pub screen_width: u32,
    /// Screen height.
    pub screen_height: u32,
}

impl RasterizerResources {
    /// Create resources for software rasterizer.
    pub fn new(device: &Device, screen_width: u32, screen_height: u32, max_triangles: u32) -> Self {
        let total_pixels = (screen_width * screen_height) as u64;

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sw_rasterizer_params"),
            size: mem::size_of::<RasterizerParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let triangles_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sw_rasterizer_triangles"),
            size: (max_triangles as u64) * (mem::size_of::<ClipSpaceTriangle>() as u64),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let metadata_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sw_rasterizer_metadata"),
            size: (max_triangles as u64) * (mem::size_of::<TriangleMeta>() as u64),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let depth_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sw_rasterizer_depth"),
            size: total_pixels * 4, // u32 per pixel
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let visibility_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sw_rasterizer_visibility"),
            size: total_pixels * 4, // u32 per pixel
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let stats_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sw_rasterizer_stats"),
            size: mem::size_of::<RasterizerStats>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let stats_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sw_rasterizer_stats_staging"),
            size: mem::size_of::<RasterizerStats>() as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            params_buffer,
            triangles_buffer,
            metadata_buffer,
            depth_buffer,
            visibility_buffer,
            stats_buffer,
            stats_staging,
            max_triangles,
            screen_width,
            screen_height,
        }
    }

    /// Upload parameters to the GPU.
    pub fn upload_params(&self, queue: &Queue, params: &RasterizerParams) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Upload triangles to the GPU.
    pub fn upload_triangles(&self, queue: &Queue, triangles: &[ClipSpaceTriangle]) {
        let byte_len = triangles.len() * mem::size_of::<ClipSpaceTriangle>();
        assert!(byte_len <= self.triangles_buffer.size() as usize);
        queue.write_buffer(&self.triangles_buffer, 0, bytemuck::cast_slice(triangles));
    }

    /// Upload triangle metadata to the GPU.
    pub fn upload_metadata(&self, queue: &Queue, metadata: &[TriangleMeta]) {
        let byte_len = metadata.len() * mem::size_of::<TriangleMeta>();
        assert!(byte_len <= self.metadata_buffer.size() as usize);
        queue.write_buffer(&self.metadata_buffer, 0, bytemuck::cast_slice(metadata));
    }

    /// Clear statistics counters.
    pub fn clear_stats(&self, queue: &Queue) {
        queue.write_buffer(&self.stats_buffer, 0, &[0u8; 16]);
    }

    /// Read statistics back from GPU.
    pub fn read_stats(&self, device: &Device, queue: &Queue) -> RasterizerStats {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("read_rasterizer_stats"),
        });
        encoder.copy_buffer_to_buffer(
            &self.stats_buffer,
            0,
            &self.stats_staging,
            0,
            mem::size_of::<RasterizerStats>() as u64,
        );
        queue.submit([encoder.finish()]);

        let buffer_slice = self.stats_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let stats = *bytemuck::from_bytes::<RasterizerStats>(&data);
        drop(data);
        self.stats_staging.unmap();

        stats
    }

    /// Get total pixel count.
    #[inline]
    pub fn total_pixels(&self) -> u32 {
        self.screen_width * self.screen_height
    }
}

// =============================================================================
// SOFTWARE RASTERIZER PIPELINE
// =============================================================================

/// Compute pipeline for software rasterization.
pub struct SoftwareRasterizerPipeline {
    /// Tiled rasterization pipeline (one workgroup per triangle).
    tiled_pipeline: wgpu::ComputePipeline,
    /// Linear rasterization pipeline (one thread per triangle).
    linear_pipeline: wgpu::ComputePipeline,
    /// Clear depth buffer pipeline.
    clear_depth_pipeline: wgpu::ComputePipeline,
    /// Clear visibility buffer pipeline.
    clear_visibility_pipeline: wgpu::ComputePipeline,
    /// Clear statistics pipeline.
    clear_stats_pipeline: wgpu::ComputePipeline,
    /// Bind group layout.
    bind_group_layout: wgpu::BindGroupLayout,
}

impl SoftwareRasterizerPipeline {
    /// Create a new software rasterizer pipeline.
    pub fn new(device: &Device) -> Self {
        let bind_group_layout = Self::create_bind_group_layout(device);
        let pipeline_layout = Self::create_pipeline_layout(device, &bind_group_layout);

        let shader_source = include_str!("../../shaders/virtual_geometry/sw_rasterizer.comp.wgsl");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sw_rasterizer_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let tiled_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sw_rasterizer_tiled_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "rasterize_triangle_tiled",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let linear_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sw_rasterizer_linear_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "rasterize_small_triangle",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let clear_depth_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sw_rasterizer_clear_depth_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "clear_depth_buffer",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let clear_visibility_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sw_rasterizer_clear_visibility_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "clear_visibility_buffer",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let clear_stats_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sw_rasterizer_clear_stats_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "clear_stats",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            tiled_pipeline,
            linear_pipeline,
            clear_depth_pipeline,
            clear_visibility_pipeline,
            clear_stats_pipeline,
            bind_group_layout,
        }
    }

    /// Get the bind group layout.
    #[inline]
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    /// Create a bind group for the given resources.
    pub fn create_bind_group(
        &self,
        device: &Device,
        resources: &RasterizerResources,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sw_rasterizer_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: resources.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: resources.triangles_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resources.metadata_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: resources.depth_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: resources.visibility_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: resources.stats_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Dispatch tiled rasterization (better for larger triangles).
    pub fn dispatch_tiled(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        params: &RasterizerParams,
    ) {
        if params.is_empty() {
            return;
        }

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sw_rasterizer_tiled_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.tiled_pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(params.num_workgroups_tiled(), 1, 1);
    }

    /// Dispatch linear rasterization (better for small triangles).
    pub fn dispatch_linear(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        params: &RasterizerParams,
    ) {
        if params.is_empty() {
            return;
        }

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sw_rasterizer_linear_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.linear_pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(params.num_workgroups_linear(), 1, 1);
    }

    /// Dispatch clear operations (depth + visibility + stats).
    pub fn dispatch_clear(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        params: &RasterizerParams,
    ) {
        let num_clear_workgroups = params.num_workgroups_clear();

        // Clear stats
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sw_rasterizer_clear_stats_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.clear_stats_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        // Clear depth buffer
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sw_rasterizer_clear_depth_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.clear_depth_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(num_clear_workgroups, 1, 1);
        }

        // Clear visibility buffer
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sw_rasterizer_clear_visibility_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.clear_visibility_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(num_clear_workgroups, 1, 1);
        }
    }

    /// Create the bind group layout.
    fn create_bind_group_layout(device: &Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sw_rasterizer_bind_group_layout"),
            entries: &[
                // binding 0: params (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: triangles (storage, read)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 2: metadata (storage, read)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 3: depth_buffer (storage, read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 4: visibility_buffer (storage, read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 5: stats (storage, read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create the pipeline layout.
    fn create_pipeline_layout(
        device: &Device,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::PipelineLayout {
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sw_rasterizer_pipeline_layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        })
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Size/Layout Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_rasterizer_params_size() {
        assert_eq!(mem::size_of::<RasterizerParams>(), 64);
    }

    #[test]
    fn test_clip_space_triangle_size() {
        assert_eq!(mem::size_of::<ClipSpaceTriangle>(), 48);
    }

    #[test]
    fn test_triangle_meta_size() {
        assert_eq!(mem::size_of::<TriangleMeta>(), 8);
    }

    #[test]
    fn test_rasterizer_tile_size() {
        assert_eq!(mem::size_of::<RasterizerTile>(), 16);
    }

    #[test]
    fn test_rasterizer_stats_size() {
        assert_eq!(mem::size_of::<RasterizerStats>(), 16);
    }

    #[test]
    fn test_rasterizer_params_pod() {
        let params = RasterizerParams::new(1920.0, 1080.0, 1000);
        let bytes = bytemuck::bytes_of(&params);
        assert_eq!(bytes.len(), 64);
    }

    #[test]
    fn test_clip_space_triangle_pod() {
        let tri = ClipSpaceTriangle::new(
            [0.0, 0.0, 0.5, 1.0],
            [1.0, 0.0, 0.5, 1.0],
            [0.5, 1.0, 0.5, 1.0],
        );
        let bytes = bytemuck::bytes_of(&tri);
        assert_eq!(bytes.len(), 48);
    }

    #[test]
    fn test_triangle_meta_pod() {
        let meta = TriangleMeta::new(42, 100);
        let bytes = bytemuck::bytes_of(&meta);
        assert_eq!(bytes.len(), 8);
    }

    // -------------------------------------------------------------------------
    // Edge Function Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_edge_function_ccw() {
        // CCW triangle: (0,0) -> (1,0) -> (0.5, 1)
        // Using Pineda's edge function: E(p) = (p.x - v0.x)*(v1.y - v0.y) - (p.y - v0.y)*(v1.x - v0.x)
        // For horizontal edge from (0,0) to (1,0), dy=0, dx=1
        // E(p) = (p.x - 0)*(0) - (p.y - 0)*(1) = -p.y
        // Point above edge (p.y > 0) gives negative value
        // Point below edge (p.y < 0) gives positive value
        let v0 = (0.0, 0.0);
        let v1 = (1.0, 0.0);
        let p = (0.5, 0.3);

        let e = cpu_edge_function(v0, v1, p);
        // Point at y=0.3 gives e = -0.3
        assert!(e < 0.0, "Point above horizontal edge should have negative edge value");
    }

    #[test]
    fn test_edge_function_on_edge() {
        let v0 = (0.0, 0.0);
        let v1 = (1.0, 0.0);
        let p = (0.5, 0.0); // On the edge

        let e = cpu_edge_function(v0, v1, p);
        assert!((e).abs() < 1e-6, "Point on edge should have zero edge value");
    }

    #[test]
    fn test_edge_function_outside() {
        let v0 = (0.0, 0.0);
        let v1 = (1.0, 0.0);
        let p = (0.5, -1.0); // Below the edge

        let e = cpu_edge_function(v0, v1, p);
        // Point below edge (y < 0) gives positive value for horizontal edge
        assert!(e > 0.0, "Point below horizontal edge should have positive edge value");
    }

    // -------------------------------------------------------------------------
    // Bounding Box Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_triangle_bbox_basic() {
        let p0 = (10.0, 20.0);
        let p1 = (50.0, 10.0);
        let p2 = (30.0, 60.0);

        let bbox = cpu_triangle_bbox(p0, p1, p2, 1920, 1080);

        assert_eq!(bbox.min_x, 10);
        assert_eq!(bbox.min_y, 10);
        assert_eq!(bbox.max_x, 50);
        assert_eq!(bbox.max_y, 60);
    }

    #[test]
    fn test_triangle_bbox_clamped() {
        let p0 = (-10.0, -20.0);
        let p1 = (100.0, 50.0);
        let p2 = (50.0, 200.0);

        let bbox = cpu_triangle_bbox(p0, p1, p2, 100, 100);

        assert_eq!(bbox.min_x, 0);
        assert_eq!(bbox.min_y, 0);
        assert_eq!(bbox.max_x, 99);
        assert_eq!(bbox.max_y, 99);
    }

    #[test]
    fn test_triangle_bbox_offscreen() {
        let p0 = (-100.0, -100.0);
        let p1 = (-50.0, -50.0);
        let p2 = (-75.0, -25.0);

        let bbox = cpu_triangle_bbox(p0, p1, p2, 1920, 1080);

        // All negative coordinates, clamped to 0
        assert!(bbox.is_empty() || bbox.max_x < 0 || bbox.max_y < 0);
    }

    #[test]
    fn test_bbox_area() {
        let bbox = BoundingBox::new(0, 0, 9, 9);
        assert_eq!(bbox.width(), 10);
        assert_eq!(bbox.height(), 10);
        assert_eq!(bbox.area(), 100);
    }

    #[test]
    fn test_bbox_empty() {
        let empty = BoundingBox::new(10, 10, 5, 5);
        assert!(empty.is_empty());
        assert_eq!(empty.area(), 0);
    }

    // -------------------------------------------------------------------------
    // Depth Test Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_depth_test_closer() {
        assert!(cpu_depth_test(0.3, 0.5)); // 0.3 is closer in reverse-Z
    }

    #[test]
    fn test_depth_test_farther() {
        assert!(!cpu_depth_test(0.7, 0.5)); // 0.7 is farther
    }

    #[test]
    fn test_depth_test_equal() {
        assert!(!cpu_depth_test(0.5, 0.5)); // Equal, no pass
    }

    #[test]
    fn test_depth_encode_decode() {
        let depths = [0.0, 0.25, 0.5, 0.75, 1.0];
        for d in depths {
            let encoded = cpu_encode_depth(d);
            let decoded = cpu_decode_depth(encoded);
            assert!((d - decoded).abs() < 1e-6, "Depth encode/decode roundtrip failed");
        }
    }

    #[test]
    fn test_depth_encode_ordering() {
        // For reverse-Z, smaller depth = closer
        let near = cpu_encode_depth(0.1);
        let far = cpu_encode_depth(0.9);

        // In float bit representation, smaller float has smaller bits (for positive values)
        assert!(near < far, "Near depth should encode to smaller value");
    }

    // -------------------------------------------------------------------------
    // Visibility Packing Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_visibility_pack_unpack() {
        let instance_id = 0x1234;
        let primitive_id = 0x5678;

        let packed = cpu_pack_visibility(instance_id, primitive_id);
        let unpacked_inst = cpu_unpack_instance_id(packed);
        let unpacked_prim = cpu_unpack_primitive_id(packed);

        assert_eq!(unpacked_inst, instance_id);
        assert_eq!(unpacked_prim, primitive_id);
    }

    #[test]
    fn test_visibility_pack_max_values() {
        let instance_id = 0xFFFF;
        let primitive_id = 0xFFFF;

        let packed = cpu_pack_visibility(instance_id, primitive_id);
        assert_eq!(cpu_unpack_instance_id(packed), 0xFFFF);
        assert_eq!(cpu_unpack_primitive_id(packed), 0xFFFF);
    }

    #[test]
    fn test_visibility_pack_zero() {
        let packed = cpu_pack_visibility(0, 0);
        assert_eq!(packed, 0);
        assert_eq!(cpu_unpack_instance_id(packed), 0);
        assert_eq!(cpu_unpack_primitive_id(packed), 0);
    }

    // -------------------------------------------------------------------------
    // Barycentric Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_barycentrics_vertex() {
        let p0 = (0.0, 0.0);
        let p1 = (10.0, 0.0);
        let p2 = (5.0, 10.0);

        // At vertex 0
        let bary = cpu_compute_barycentrics(p0, p1, p2, p0).unwrap();
        assert!((bary.0 - 1.0).abs() < 0.01);
        assert!(bary.1.abs() < 0.01);
        assert!(bary.2.abs() < 0.01);
    }

    #[test]
    fn test_barycentrics_center() {
        let p0 = (0.0, 0.0);
        let p1 = (3.0, 0.0);
        let p2 = (0.0, 3.0);

        // Centroid
        let center = (1.0, 1.0);
        let bary = cpu_compute_barycentrics(p0, p1, p2, center).unwrap();

        // At centroid, all barycentrics should be ~1/3
        assert!((bary.0 - 1.0/3.0).abs() < 0.1);
        assert!((bary.1 - 1.0/3.0).abs() < 0.1);
        assert!((bary.2 - 1.0/3.0).abs() < 0.1);
    }

    #[test]
    fn test_barycentrics_sum_to_one() {
        let p0 = (0.0, 0.0);
        let p1 = (10.0, 0.0);
        let p2 = (5.0, 10.0);

        let test_points = [
            (3.0, 2.0),
            (5.0, 5.0),
            (7.0, 3.0),
        ];

        for point in test_points {
            if let Some(bary) = cpu_compute_barycentrics(p0, p1, p2, point) {
                let sum = bary.0 + bary.1 + bary.2;
                assert!((sum - 1.0).abs() < 0.01, "Barycentrics should sum to 1");
            }
        }
    }

    #[test]
    fn test_barycentrics_degenerate() {
        // Collinear points
        let p0 = (0.0, 0.0);
        let p1 = (5.0, 0.0);
        let p2 = (10.0, 0.0);

        let result = cpu_compute_barycentrics(p0, p1, p2, (5.0, 0.0));
        assert!(result.is_none(), "Degenerate triangle should return None");
    }

    // -------------------------------------------------------------------------
    // Depth Interpolation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_depth_interpolation_vertex() {
        let bary = (1.0, 0.0, 0.0);
        let z0 = 0.1;
        let z1 = 0.5;
        let z2 = 0.9;

        let depth = cpu_interpolate_depth(bary, z0, z1, z2);
        assert!((depth - z0).abs() < 1e-6);
    }

    #[test]
    fn test_depth_interpolation_center() {
        let bary = (1.0/3.0, 1.0/3.0, 1.0/3.0);
        let z0 = 0.0;
        let z1 = 0.3;
        let z2 = 0.6;

        let depth = cpu_interpolate_depth(bary, z0, z1, z2);
        let expected = (z0 + z1 + z2) / 3.0;
        assert!((depth - expected).abs() < 0.01);
    }

    // -------------------------------------------------------------------------
    // Backface Culling Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_backface_ccw() {
        // In screen space with Y-down, CCW triangles have negative area
        // The is_backfacing function checks if area <= 0
        // So CCW triangles ARE considered backfacing in Y-down space
        // This is the expected behavior: we assume Y-up in NDC, but Y-down in screen
        let p0 = (0.0, 0.0);
        let p1 = (1.0, 0.0);
        let p2 = (0.5, 1.0);

        // Area = edge_function(p0, p1, p2) = (0.5)(0) - (1)(1) = -1 (negative)
        // So this is backfacing according to our convention
        let area = cpu_edge_function(p0, p1, p2);
        assert!(area < 0.0, "CCW in Y-down screen space should have negative area");
    }

    #[test]
    fn test_backface_cw() {
        // CW triangle in Y-down screen space has positive area (front-facing)
        let p0 = (0.0, 0.0);
        let p1 = (0.5, 1.0);
        let p2 = (1.0, 0.0);

        let area = cpu_edge_function(p0, p1, p2);
        assert!(area > 0.0, "CW in Y-down screen space should have positive area");
    }

    // -------------------------------------------------------------------------
    // Triangle Area Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_triangle_area() {
        // Right triangle with legs of 3 and 4
        let p0 = (0.0, 0.0);
        let p1 = (3.0, 0.0);
        let p2 = (0.0, 4.0);

        let area = cpu_triangle_area(p0, p1, p2);
        assert!((area - 6.0).abs() < 0.01, "Area should be 6");
    }

    #[test]
    fn test_triangle_area_degenerate() {
        // Collinear points
        let p0 = (0.0, 0.0);
        let p1 = (1.0, 0.0);
        let p2 = (2.0, 0.0);

        let area = cpu_triangle_area(p0, p1, p2);
        assert!(area < MIN_TRIANGLE_AREA, "Degenerate triangle should have ~0 area");
    }

    // -------------------------------------------------------------------------
    // Projection Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_project_vertex_center() {
        // Clip space (0, 0, 0.5, 1) should project to center of viewport
        let clip = [0.0, 0.0, 0.5, 1.0];
        let viewport = [0.0, 0.0, 1920.0, 1080.0];

        let (sx, sy, _sz) = cpu_project_vertex(clip, viewport);

        assert!((sx - 960.0).abs() < 1.0, "Should be at center X");
        assert!((sy - 540.0).abs() < 1.0, "Should be at center Y");
    }

    #[test]
    fn test_project_vertex_corner() {
        // Clip space (-1, -1, 0.5, 1) should project to top-left
        let clip = [-1.0, -1.0, 0.5, 1.0];
        let viewport = [0.0, 0.0, 100.0, 100.0];

        let (sx, sy, _sz) = cpu_project_vertex(clip, viewport);

        assert!(sx.abs() < 1.0, "Should be at left edge");
        assert!(sy.abs() < 1.0, "Should be at top edge");
    }

    // -------------------------------------------------------------------------
    // RasterizerParams Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_rasterizer_params_new() {
        let params = RasterizerParams::new(1920.0, 1080.0, 1000);

        assert_eq!(params.viewport, [0.0, 0.0, 1920.0, 1080.0]);
        assert_eq!(params.screen_width, 1920);
        assert_eq!(params.screen_height, 1080);
        assert_eq!(params.num_triangles, 1000);
        assert!(params.depth_test_enabled());
        assert!(params.backface_cull_enabled());
    }

    #[test]
    fn test_rasterizer_params_builders() {
        let params = RasterizerParams::new(1920.0, 1080.0, 1000)
            .with_depth_bias(0.001)
            .with_tile_size(16)
            .with_depth_test(false)
            .with_backface_cull(false);

        assert!((params.depth_bias - 0.001).abs() < 1e-6);
        assert_eq!(params.tile_size, 16);
        assert!(!params.depth_test_enabled());
        assert!(!params.backface_cull_enabled());
    }

    #[test]
    fn test_rasterizer_params_workgroups() {
        let params = RasterizerParams::new(1920.0, 1080.0, 1000);

        assert_eq!(params.num_workgroups_tiled(), 1000);
        assert_eq!(params.num_workgroups_linear(), 4); // ceil(1000/256)

        let clear_wg = params.num_workgroups_clear();
        let expected = (1920 * 1080 + 255) / 256;
        assert_eq!(clear_wg, expected);
    }

    #[test]
    fn test_rasterizer_params_empty() {
        let empty = RasterizerParams::new(1920.0, 1080.0, 0);
        assert!(empty.is_empty());

        let not_empty = RasterizerParams::new(1920.0, 1080.0, 1);
        assert!(!not_empty.is_empty());
    }

    // -------------------------------------------------------------------------
    // ClipSpaceTriangle Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_clip_space_triangle_vertex_access() {
        let tri = ClipSpaceTriangle::new(
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        );

        assert_eq!(tri.vertex(0), [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(tri.vertex(1), [5.0, 6.0, 7.0, 8.0]);
        assert_eq!(tri.vertex(2), [9.0, 10.0, 11.0, 12.0]);
    }

    #[test]
    fn test_clip_space_triangle_behind_camera() {
        // All vertices in front
        let front = ClipSpaceTriangle::new(
            [0.0, 0.0, 0.5, 1.0],
            [1.0, 0.0, 0.5, 1.0],
            [0.5, 1.0, 0.5, 1.0],
        );
        assert!(!front.is_behind_camera());

        // One vertex behind (w <= 0)
        let behind = ClipSpaceTriangle::new(
            [0.0, 0.0, 0.5, -1.0], // w < 0
            [1.0, 0.0, 0.5, 1.0],
            [0.5, 1.0, 0.5, 1.0],
        );
        assert!(behind.is_behind_camera());
    }

    // -------------------------------------------------------------------------
    // TriangleMeta Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_triangle_meta_new() {
        let meta = TriangleMeta::new(42, 100);
        assert_eq!(meta.instance_id, 42);
        assert_eq!(meta.primitive_id, 100);
        assert!(meta.is_valid());
    }

    #[test]
    fn test_triangle_meta_invalid() {
        let meta = TriangleMeta::invalid();
        assert!(!meta.is_valid());
        assert_eq!(meta.instance_id, INVALID_INSTANCE_ID);
    }

    // -------------------------------------------------------------------------
    // RasterizerTile Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_rasterizer_tile() {
        let tile = RasterizerTile::new(5, 10, 100, 50);

        assert_eq!(tile.pixel_origin(8), (40, 80));
        assert!(!tile.is_empty());
    }

    #[test]
    fn test_rasterizer_tile_empty() {
        let empty = RasterizerTile::new(0, 0, 0, 0);
        assert!(empty.is_empty());
    }

    // -------------------------------------------------------------------------
    // RasterizerStats Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_rasterizer_stats_rates() {
        let stats = RasterizerStats {
            triangles_processed: 80,
            triangles_culled: 20,
            fragments_tested: 1000,
            fragments_written: 500,
        };

        assert!((stats.cull_rate() - 0.2).abs() < 0.01);
        assert!((stats.depth_pass_rate() - 0.5).abs() < 0.01);
        assert!((stats.avg_fragments_per_triangle() - 12.5).abs() < 0.01);
    }

    #[test]
    fn test_rasterizer_stats_zero() {
        let stats = RasterizerStats::default();

        assert_eq!(stats.cull_rate(), 0.0);
        assert_eq!(stats.depth_pass_rate(), 0.0);
        assert_eq!(stats.avg_fragments_per_triangle(), 0.0);
    }

    // -------------------------------------------------------------------------
    // CPU Rasterization Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cpu_rasterize_simple_triangle() {
        // A triangle in clip space that projects to visible screen area
        // NDC: center of screen, with positive w
        let triangle = ClipSpaceTriangle::new(
            [-0.5, -0.5, 0.5, 1.0],   // Bottom-left in NDC
            [0.5, -0.5, 0.5, 1.0],    // Bottom-right in NDC
            [0.0, 0.5, 0.5, 1.0],     // Top-center in NDC
        );
        let meta = TriangleMeta::new(1, 0);
        let viewport = [0.0, 0.0, 100.0, 100.0];

        // Without backface culling to ensure we get fragments regardless of winding
        let fragments = cpu_rasterize_triangle(&triangle, &meta, viewport, 100, 100, false);

        assert!(!fragments.is_empty(), "Should produce some fragments");

        // All fragments should have the correct instance/primitive IDs
        for (_, _, _, inst, prim) in &fragments {
            assert_eq!(*inst, 1);
            assert_eq!(*prim, 0);
        }
    }

    #[test]
    fn test_cpu_rasterize_invalid_triangle() {
        let triangle = ClipSpaceTriangle::new(
            [0.0, 0.0, 0.5, 1.0],
            [1.0, 0.0, 0.5, 1.0],
            [0.5, 1.0, 0.5, 1.0],
        );
        let meta = TriangleMeta::invalid();
        let viewport = [0.0, 0.0, 100.0, 100.0];

        let fragments = cpu_rasterize_triangle(&triangle, &meta, viewport, 100, 100, false);

        assert!(fragments.is_empty(), "Invalid triangle should produce no fragments");
    }

    #[test]
    fn test_cpu_rasterize_backfacing() {
        // Create two triangles with opposite windings
        // Triangle 1: will produce fragments without culling
        let triangle1 = ClipSpaceTriangle::new(
            [-0.5, -0.5, 0.5, 1.0],
            [0.5, -0.5, 0.5, 1.0],
            [0.0, 0.5, 0.5, 1.0],
        );

        // Triangle 2: reversed winding (swap v1 and v2)
        let triangle2 = ClipSpaceTriangle::new(
            [-0.5, -0.5, 0.5, 1.0],
            [0.0, 0.5, 0.5, 1.0],
            [0.5, -0.5, 0.5, 1.0],
        );

        let meta = TriangleMeta::new(1, 0);
        let viewport = [0.0, 0.0, 100.0, 100.0];

        // Without backface culling - both should produce fragments
        let frag1_no_cull = cpu_rasterize_triangle(&triangle1, &meta, viewport, 100, 100, false);
        let frag2_no_cull = cpu_rasterize_triangle(&triangle2, &meta, viewport, 100, 100, false);

        assert!(!frag1_no_cull.is_empty(), "Triangle 1 should produce fragments without culling");
        assert!(!frag2_no_cull.is_empty(), "Triangle 2 should produce fragments without culling");

        // With backface culling - one should be culled
        let frag1_cull = cpu_rasterize_triangle(&triangle1, &meta, viewport, 100, 100, true);
        let frag2_cull = cpu_rasterize_triangle(&triangle2, &meta, viewport, 100, 100, true);

        // Exactly one of them should be empty (the backfacing one)
        let one_culled = frag1_cull.is_empty() != frag2_cull.is_empty();
        assert!(one_culled, "Exactly one triangle should be culled with backface culling enabled");
    }

    // -------------------------------------------------------------------------
    // Shader Validation Tests (using naga)
    // -------------------------------------------------------------------------

    #[test]
    fn test_sw_rasterizer_shader_parses() {
        let shader_source =
            include_str!("../../shaders/virtual_geometry/sw_rasterizer.comp.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("sw_rasterizer shader should parse without errors");

        let entry_names: Vec<_> = module.entry_points.iter().map(|ep| &ep.name).collect();

        assert!(
            entry_names.iter().any(|n| *n == "rasterize_triangle_tiled"),
            "Should have rasterize_triangle_tiled entry point"
        );
        assert!(
            entry_names.iter().any(|n| *n == "rasterize_small_triangle"),
            "Should have rasterize_small_triangle entry point"
        );
        assert!(
            entry_names.iter().any(|n| *n == "clear_depth_buffer"),
            "Should have clear_depth_buffer entry point"
        );
        assert!(
            entry_names.iter().any(|n| *n == "clear_visibility_buffer"),
            "Should have clear_visibility_buffer entry point"
        );
        assert!(
            entry_names.iter().any(|n| *n == "clear_stats"),
            "Should have clear_stats entry point"
        );
    }

    #[test]
    fn test_sw_rasterizer_shader_validates() {
        let shader_source =
            include_str!("../../shaders/virtual_geometry/sw_rasterizer.comp.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("sw_rasterizer shader should parse without errors");

        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );

        validator
            .validate(&module)
            .expect("sw_rasterizer shader should validate without errors");
    }

    #[test]
    fn test_sw_rasterizer_shader_workgroup_sizes() {
        let shader_source =
            include_str!("../../shaders/virtual_geometry/sw_rasterizer.comp.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("sw_rasterizer shader should parse without errors");

        for ep in &module.entry_points {
            match ep.name.as_str() {
                "rasterize_triangle_tiled" => {
                    assert_eq!(ep.workgroup_size, [64, 1, 1]);
                }
                "rasterize_small_triangle" | "clear_depth_buffer" | "clear_visibility_buffer" => {
                    assert_eq!(ep.workgroup_size, [256, 1, 1]);
                }
                "clear_stats" => {
                    assert_eq!(ep.workgroup_size, [1, 1, 1]);
                }
                _ => {}
            }
        }
    }

    #[test]
    fn test_sw_rasterizer_shader_entry_points_are_compute() {
        let shader_source =
            include_str!("../../shaders/virtual_geometry/sw_rasterizer.comp.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("sw_rasterizer shader should parse without errors");

        for ep in &module.entry_points {
            assert_eq!(
                ep.stage,
                naga::ShaderStage::Compute,
                "Entry point {} should be a compute shader",
                ep.name
            );
        }
    }
}
