// SPDX-License-Identifier: MIT
//
// gpu_cull_small_triangle.comp.wgsl - Small Triangle Culling for TRINITY Engine (T-GPU-3.8)
//
// Culls triangles smaller than a configurable pixel threshold.
// Uses screen-space projected area estimation via 2D cross product.
//
// Algorithm:
// 1. Read pre-projected triangle screen positions (from prior projection pass)
// 2. Compute 2D triangle area using cross product: area = |e1 x e2| / 2
// 3. If area < min_pixel_area: cull (set flag to 0)
// 4. Otherwise: mark visible (set flag to 1)
// 5. Track culled count via atomic counter for statistics
//
// Performance: O(n) work, single dispatch, <0.02ms for 100K triangles

// ============================================================================
// Constants
// ============================================================================

const WORKGROUP_SIZE: u32 = 256u;

// ============================================================================
// Structs
// ============================================================================

/// Culling parameters uniform buffer.
///
/// Memory Layout (80 bytes, std140 aligned):
/// | Offset | Field             | Size |
/// |--------|-------------------|------|
/// | 0      | num_triangles     | 4    |
/// | 4      | _pad0             | 12   |
/// | 16     | view_proj (col0)  | 16   |
/// | 32     | view_proj (col1)  | 16   |
/// | 48     | view_proj (col2)  | 16   |
/// | 64     | view_proj (col3)  | 16   |
/// | 80     | viewport_width    | 4    |
/// | 84     | viewport_height   | 4    |
/// | 88     | min_pixel_area    | 4    |
/// | 92     | _pad1             | 4    |
struct SmallTriangleCullParams {
    /// Number of triangles to process.
    num_triangles: u32,
    /// Padding for mat4 alignment.
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    /// View-projection matrix (column-major).
    view_proj: mat4x4<f32>,
    /// Viewport width in pixels.
    viewport_width: f32,
    /// Viewport height in pixels.
    viewport_height: f32,
    /// Minimum visible area in pixels (e.g., 1.0 for single pixel threshold).
    min_pixel_area: f32,
    /// Padding for 16-byte alignment.
    _pad3: f32,
}

/// Pre-projected triangle with screen-space positions.
///
/// Memory Layout (32 bytes):
/// | Offset | Field        | Size |
/// |--------|--------------|------|
/// | 0      | p0           | 8    |
/// | 8      | p1           | 8    |
/// | 16     | p2           | 8    |
/// | 24     | instance_id  | 4    |
/// | 28     | primitive_id | 4    |
struct ProjectedTriangle {
    /// Screen-space position of vertex 0 (after perspective divide).
    p0: vec2<f32>,
    /// Screen-space position of vertex 1 (after perspective divide).
    p1: vec2<f32>,
    /// Screen-space position of vertex 2 (after perspective divide).
    p2: vec2<f32>,
    /// Instance ID for draw call lookup.
    instance_id: u32,
    /// Primitive ID within the mesh.
    primitive_id: u32,
}

/// Atomic counter for statistics.
struct AtomicCounter {
    value: atomic<u32>,
}

// ============================================================================
// Bindings
// ============================================================================

/// Culling parameters (uniform buffer).
@group(0) @binding(0) var<uniform> params: SmallTriangleCullParams;

/// Input triangles with projected screen positions (read-only storage buffer).
@group(0) @binding(1) var<storage, read> triangles: array<ProjectedTriangle>;

/// Output visibility flags: 1 = visible, 0 = culled (read-write storage buffer).
@group(0) @binding(2) var<storage, read_write> visible_flags: array<u32>;

/// Atomic counter for number of culled triangles (for statistics).
@group(0) @binding(3) var<storage, read_write> culled_count: AtomicCounter;

// ============================================================================
// Helper Functions
// ============================================================================

/// Compute screen-space triangle area using 2D cross product.
///
/// Uses the "shoelace formula" variant: area = |e1 x e2| / 2
/// where e1 = p1 - p0, e2 = p2 - p0.
///
/// The 2D cross product gives the signed area of the parallelogram:
/// cross = e1.x * e2.y - e1.y * e2.x
/// Triangle area = |cross| / 2
///
/// This is more efficient than computing the full 3D area because:
/// - No square root needed
/// - Works directly in screen space
/// - Correctly handles perspective projection
fn triangle_area_2d(p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>) -> f32 {
    let e1 = p1 - p0;
    let e2 = p2 - p0;
    // 2D cross product gives signed parallelogram area
    let cross = e1.x * e2.y - e1.y * e2.x;
    // Triangle is half the parallelogram, take absolute value
    return abs(cross) * 0.5;
}

/// Check if triangle is degenerate (zero or near-zero area).
///
/// A triangle is degenerate if:
/// - All three vertices are coincident
/// - Two vertices are coincident (line)
/// - Vertices are collinear
fn is_degenerate(p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>) -> bool {
    // Use a very small epsilon for numerical stability
    let epsilon = 1e-10;
    let area = triangle_area_2d(p0, p1, p2);
    return area < epsilon;
}

/// Convert NDC coordinates to pixel coordinates.
///
/// NDC range is [-1, 1], pixel range is [0, width/height].
/// x_pixel = (ndc_x + 1) * 0.5 * width
/// y_pixel = (ndc_y + 1) * 0.5 * height
fn ndc_to_pixels(ndc: vec2<f32>, viewport_width: f32, viewport_height: f32) -> vec2<f32> {
    return vec2<f32>(
        (ndc.x + 1.0) * 0.5 * viewport_width,
        (ndc.y + 1.0) * 0.5 * viewport_height
    );
}

// ============================================================================
// Main Compute Kernel
// ============================================================================

/// Per-triangle small triangle culling kernel.
///
/// Each thread processes one triangle:
/// 1. Read pre-projected screen coordinates
/// 2. Convert NDC to pixel space
/// 3. Compute triangle area in pixels
/// 4. If area < min_pixel_area or degenerate: cull
/// 5. Otherwise: mark visible
@compute @workgroup_size(256)
fn cull_small_triangle(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;

    // Bounds check: skip threads beyond triangle count.
    if (idx >= params.num_triangles) {
        return;
    }

    let tri = triangles[idx];

    // Convert from NDC to pixel coordinates for area calculation
    let p0_pixels = ndc_to_pixels(tri.p0, params.viewport_width, params.viewport_height);
    let p1_pixels = ndc_to_pixels(tri.p1, params.viewport_width, params.viewport_height);
    let p2_pixels = ndc_to_pixels(tri.p2, params.viewport_width, params.viewport_height);

    // Compute triangle area in pixels
    let area = triangle_area_2d(p0_pixels, p1_pixels, p2_pixels);

    // Cull if area is below threshold (includes degenerate triangles)
    if (area < params.min_pixel_area) {
        visible_flags[idx] = 0u;
        atomicAdd(&culled_count.value, 1u);
    } else {
        visible_flags[idx] = 1u;
    }
}

// ============================================================================
// Alternative Entry Points
// ============================================================================

/// Small triangle culling with degenerate check only.
/// Culls only zero-area triangles, ignores min_pixel_area threshold.
/// Useful for preprocessing pass before rasterization.
@compute @workgroup_size(256)
fn cull_degenerate_only(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;

    if (idx >= params.num_triangles) {
        return;
    }

    let tri = triangles[idx];

    // Check for degenerate triangle in NDC space (scale-independent)
    if (is_degenerate(tri.p0, tri.p1, tri.p2)) {
        visible_flags[idx] = 0u;
        atomicAdd(&culled_count.value, 1u);
    } else {
        visible_flags[idx] = 1u;
    }
}

/// Small triangle culling without atomic counter.
/// Faster when statistics are not needed.
@compute @workgroup_size(256)
fn cull_small_triangle_no_stats(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;

    if (idx >= params.num_triangles) {
        return;
    }

    let tri = triangles[idx];

    let p0_pixels = ndc_to_pixels(tri.p0, params.viewport_width, params.viewport_height);
    let p1_pixels = ndc_to_pixels(tri.p1, params.viewport_width, params.viewport_height);
    let p2_pixels = ndc_to_pixels(tri.p2, params.viewport_width, params.viewport_height);

    let area = triangle_area_2d(p0_pixels, p1_pixels, p2_pixels);

    if (area < params.min_pixel_area) {
        visible_flags[idx] = 0u;
    } else {
        visible_flags[idx] = 1u;
    }
}
