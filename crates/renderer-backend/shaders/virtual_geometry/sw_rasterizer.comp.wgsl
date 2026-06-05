// SPDX-License-Identifier: MIT
//
// sw_rasterizer.comp.wgsl - Software Rasterizer for TRINITY Engine (T-GPU-8.2)
//
// Compute shader-based triangle rasterization for virtual geometry systems.
// Software rasterization in compute shaders handles tiny triangles more
// efficiently than hardware rasterization, avoiding quad overshading.
//
// Algorithm:
// 1. Compute triangle bounding box in screen space
// 2. Set up edge functions for inside/outside tests (orient2d)
// 3. Iterate pixels within bounding box using tile-based approach
// 4. Test each pixel against all three edge functions
// 5. For pixels inside, interpolate depth via barycentrics
// 6. Perform atomic depth test using atomicMin on u32 depth buffer
// 7. On pass, write to visibility buffer (pack instance_id + primitive_id)
//
// Performance Target: <0.1ms for 50K triangles on RTX 3080
//
// References:
// - "A Parallel Algorithm for Polygon Rasterization" (Pineda, 1988)
// - "Software Occlusion Culling" (Intel, 2014)
// - "Nanite: A Deep Dive" (Epic Games, 2021)

// ============================================================================
// Constants
// ============================================================================

/// Workgroup size for tile-based rasterization.
/// 8x8 = 64 threads, matches GPU warp/wavefront size.
const TILE_SIZE_X: u32 = 8u;
const TILE_SIZE_Y: u32 = 8u;
const THREADS_PER_TILE: u32 = 64u;

/// Linear workgroup for per-triangle dispatch.
const WORKGROUP_SIZE_LINEAR: u32 = 256u;

/// Sub-pixel precision bits for fixed-point rasterization.
/// 8 bits = 256 sub-pixel positions, standard for hardware rasterizers.
const SUBPIXEL_BITS: u32 = 8u;
const SUBPIXEL_SCALE: f32 = 256.0;

/// Invalid ID sentinel values.
const INVALID_INSTANCE_ID: u32 = 0xFFFFFFFFu;
const INVALID_PRIMITIVE_ID: u32 = 0xFFFFFFu; // 24 bits max

/// Depth buffer encoding: use float bits as u32 for atomicMin.
/// Far plane = 1.0, near plane = 0.0 (reverse-Z convention).
const DEPTH_CLEAR_VALUE: u32 = 0xFFFFFFFFu; // ~infinity in float bits

/// Minimum triangle area in pixels to avoid degenerate cases.
const MIN_TRIANGLE_AREA: f32 = 0.0001;

// ============================================================================
// Structures
// ============================================================================

/// Software rasterizer parameters.
///
/// Memory Layout (64 bytes, std140 aligned):
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
struct RasterizerParams {
    /// Viewport: (x, y, width, height).
    viewport: vec4<f32>,
    /// Depth bias for shadow mapping / z-fighting avoidance.
    depth_bias: f32,
    /// Tile size in pixels (8 or 16).
    tile_size: u32,
    /// Number of triangles to rasterize.
    num_triangles: u32,
    /// Screen width in pixels.
    screen_width: u32,
    /// Screen height in pixels.
    screen_height: u32,
    /// Near plane depth value.
    near_plane: f32,
    /// Far plane depth value.
    far_plane: f32,
    /// Flags: bit 0 = enable depth test, bit 1 = enable backface cull.
    flags: u32,
    /// Padding for 64-byte alignment.
    _pad: vec4<f32>,
}

/// Clip-space triangle with 4D homogeneous coordinates.
///
/// Memory Layout (48 bytes):
/// | Offset | Field | Size |
/// |--------|-------|------|
/// | 0      | v0    | 16   |
/// | 16     | v1    | 16   |
/// | 32     | v2    | 16   |
struct ClipSpaceTriangle {
    /// Vertex 0 in clip space (x, y, z, w).
    v0: vec4<f32>,
    /// Vertex 1 in clip space (x, y, z, w).
    v1: vec4<f32>,
    /// Vertex 2 in clip space (x, y, z, w).
    v2: vec4<f32>,
}

/// Triangle metadata for visibility buffer output.
///
/// Memory Layout (8 bytes):
/// | Offset | Field        | Size |
/// |--------|--------------|------|
/// | 0      | instance_id  | 4    |
/// | 4      | primitive_id | 4    |
struct TriangleMeta {
    /// Instance ID for material/transform lookup.
    instance_id: u32,
    /// Primitive ID within the mesh.
    primitive_id: u32,
}

/// Screen-space triangle with projected coordinates and depth.
///
/// Memory Layout (32 bytes):
/// | Offset | Field | Size |
/// |--------|-------|------|
/// | 0      | p0    | 8    |
/// | 8      | p1    | 8    |
/// | 16     | p2    | 8    |
/// | 24     | z     | 12   | (3 x f32)
/// | 36     | _pad  | 4    |
struct ScreenTriangle {
    /// Screen-space position of vertex 0 (in pixels).
    p0: vec2<f32>,
    /// Screen-space position of vertex 1 (in pixels).
    p1: vec2<f32>,
    /// Screen-space position of vertex 2 (in pixels).
    p2: vec2<f32>,
    /// Depth values at each vertex (for interpolation).
    z0: f32,
    z1: f32,
    z2: f32,
    /// One-over-w for perspective-correct interpolation.
    inv_w0: f32,
    inv_w1: f32,
    inv_w2: f32,
}

/// Visibility buffer entry (packed format).
///
/// Memory Layout (8 bytes):
/// | Offset | Field              | Size |
/// |--------|--------------------| -----|
/// | 0      | instance_primitive | 4    |
/// | 4      | depth_barycentrics | 4    |
///
/// Encoding:
/// - instance_primitive: bits [31:24] unused, [23:0] instance_id (24 bits)
/// - For tiny triangles, we pack instance_id (24 bits) + primitive_id (8 bits)
struct PackedVisibility {
    /// bits [31:8] = instance_id (24 bits), bits [7:0] = primitive_id low (8 bits)
    instance_primitive: u32,
    /// primitive_id high (16 bits) | flags (16 bits)
    primitive_flags: u32,
}

/// Edge function coefficients for triangle rasterization.
///
/// Edge function: E(x,y) = a*x + b*y + c
/// Point is inside if all three edge functions have the same sign.
struct EdgeFunction {
    a: f32,
    b: f32,
    c: f32,
}

/// Bounding box in screen space.
struct BoundingBox {
    min_x: i32,
    min_y: i32,
    max_x: i32,
    max_y: i32,
}

/// Atomic depth buffer entry.
struct AtomicDepth {
    value: atomic<u32>,
}

/// Atomic visibility buffer entry.
struct AtomicVisibility {
    value: atomic<u32>,
}

// ============================================================================
// Bindings
// ============================================================================

/// Rasterizer parameters (uniform buffer).
@group(0) @binding(0) var<uniform> params: RasterizerParams;

/// Input triangles in clip space (read-only storage buffer).
@group(0) @binding(1) var<storage, read> triangles: array<ClipSpaceTriangle>;

/// Triangle metadata: instance_id, primitive_id (read-only storage buffer).
@group(0) @binding(2) var<storage, read> triangle_meta: array<TriangleMeta>;

/// Depth buffer with atomic access (read-write storage buffer).
/// Stored as u32 with float bits for atomicMin comparison.
@group(0) @binding(3) var<storage, read_write> depth_buffer: array<AtomicDepth>;

/// Visibility buffer output (read-write storage buffer).
/// Stores packed instance_id + primitive_id for winning fragments.
@group(0) @binding(4) var<storage, read_write> visibility_buffer: array<atomic<u32>>;

/// Statistics counters (read-write storage buffer).
/// [0] = triangles processed
/// [1] = triangles culled (backface/degenerate)
/// [2] = fragments tested
/// [3] = fragments written
@group(0) @binding(5) var<storage, read_write> stats: array<atomic<u32>>;

// ============================================================================
// Helper Functions: Coordinate Transforms
// ============================================================================

/// Transform clip-space to normalized device coordinates (NDC).
///
/// NDC = clip.xyz / clip.w
/// Range: [-1, 1] for x, y; [0, 1] for z (reverse-Z convention).
fn clip_to_ndc(clip: vec4<f32>) -> vec3<f32> {
    let inv_w = 1.0 / clip.w;
    return vec3<f32>(clip.x * inv_w, clip.y * inv_w, clip.z * inv_w);
}

/// Transform NDC to screen space (pixel coordinates).
///
/// screen.x = (ndc.x + 1) * 0.5 * viewport.width + viewport.x
/// screen.y = (ndc.y + 1) * 0.5 * viewport.height + viewport.y
fn ndc_to_screen(ndc: vec3<f32>, viewport: vec4<f32>) -> vec2<f32> {
    return vec2<f32>(
        (ndc.x * 0.5 + 0.5) * viewport.z + viewport.x,
        (ndc.y * 0.5 + 0.5) * viewport.w + viewport.y
    );
}

/// Project clip-space vertex to screen-space with depth.
fn project_vertex(clip: vec4<f32>, viewport: vec4<f32>) -> vec3<f32> {
    let ndc = clip_to_ndc(clip);
    let screen = ndc_to_screen(ndc, viewport);
    return vec3<f32>(screen.x, screen.y, ndc.z);
}

// ============================================================================
// Helper Functions: Edge Functions (Pineda's Algorithm)
// ============================================================================

/// Compute 2D edge function (orient2d) for a point relative to an edge.
///
/// E(p) = (p.x - v0.x) * (v1.y - v0.y) - (p.y - v0.y) * (v1.x - v0.x)
///
/// Returns:
/// - Positive if p is to the left of edge (v0 -> v1)
/// - Negative if p is to the right of edge
/// - Zero if p is on the edge
fn edge_function(v0: vec2<f32>, v1: vec2<f32>, p: vec2<f32>) -> f32 {
    return (p.x - v0.x) * (v1.y - v0.y) - (p.y - v0.y) * (v1.x - v0.x);
}

/// Set up edge function coefficients for incremental evaluation.
///
/// Edge function: E(x,y) = a*x + b*y + c
/// Where:
///   a = v1.y - v0.y (dy)
///   b = v0.x - v1.x (-dx)
///   c = v1.x * v0.y - v0.x * v1.y
fn setup_edge(v0: vec2<f32>, v1: vec2<f32>) -> EdgeFunction {
    let a = v1.y - v0.y;
    let b = v0.x - v1.x;
    let c = v1.x * v0.y - v0.x * v1.y;
    return EdgeFunction(a, b, c);
}

/// Evaluate edge function at a point.
fn eval_edge(edge: EdgeFunction, p: vec2<f32>) -> f32 {
    return edge.a * p.x + edge.b * p.y + edge.c;
}

/// Step edge function in X direction.
fn step_edge_x(edge: EdgeFunction, value: f32) -> f32 {
    return value + edge.a;
}

/// Step edge function in Y direction.
fn step_edge_y(edge: EdgeFunction, value: f32) -> f32 {
    return value + edge.b;
}

// ============================================================================
// Helper Functions: Bounding Box and Clipping
// ============================================================================

/// Compute screen-space bounding box for a triangle.
///
/// The bounding box is clamped to the viewport and rounded to pixel boundaries.
fn compute_triangle_bbox(
    p0: vec2<f32>,
    p1: vec2<f32>,
    p2: vec2<f32>,
    screen_width: u32,
    screen_height: u32
) -> BoundingBox {
    // Find min/max extents
    var min_x = min(min(p0.x, p1.x), p2.x);
    var min_y = min(min(p0.y, p1.y), p2.y);
    var max_x = max(max(p0.x, p1.x), p2.x);
    var max_y = max(max(p0.y, p1.y), p2.y);

    // Clamp to screen bounds
    min_x = max(min_x, 0.0);
    min_y = max(min_y, 0.0);
    max_x = min(max_x, f32(screen_width) - 1.0);
    max_y = min(max_y, f32(screen_height) - 1.0);

    // Round to pixel boundaries (floor for min, ceil for max)
    return BoundingBox(
        i32(floor(min_x)),
        i32(floor(min_y)),
        i32(ceil(max_x)),
        i32(ceil(max_y))
    );
}

/// Check if bounding box is empty (no pixels to rasterize).
fn bbox_is_empty(bbox: BoundingBox) -> bool {
    return bbox.max_x < bbox.min_x || bbox.max_y < bbox.min_y;
}

/// Compute triangle area (signed, in pixels squared).
///
/// Area = 0.5 * |edge_function(v0, v1, v2)|
fn triangle_area(p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>) -> f32 {
    return abs(edge_function(p0, p1, p2)) * 0.5;
}

/// Check if triangle is backfacing (clockwise winding in screen space).
///
/// For screen space with Y-down, positive edge function = front-facing.
/// We assume counter-clockwise winding for front faces.
fn is_backfacing(p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>) -> bool {
    let area = edge_function(p0, p1, p2);
    return area <= 0.0;
}

// ============================================================================
// Helper Functions: Depth and Barycentrics
// ============================================================================

/// Compute barycentric coordinates for a point inside a triangle.
///
/// Uses edge function values normalized by total triangle area.
fn compute_barycentrics(
    e0: f32, // edge function for edge (v1, v2) at p
    e1: f32, // edge function for edge (v2, v0) at p
    e2: f32, // edge function for edge (v0, v1) at p
    area_inv: f32 // 1.0 / (2 * triangle_area)
) -> vec3<f32> {
    return vec3<f32>(e0, e1, e2) * area_inv;
}

/// Interpolate depth using barycentric coordinates.
///
/// depth = bary.x * z0 + bary.y * z1 + bary.z * z2
fn interpolate_depth(bary: vec3<f32>, z0: f32, z1: f32, z2: f32) -> f32 {
    return bary.x * z0 + bary.y * z1 + bary.z * z2;
}

/// Perspective-correct depth interpolation.
///
/// For perspective-correct interpolation:
/// 1. Interpolate 1/w at each vertex
/// 2. Interpolate z/w at each vertex
/// 3. Final z = (z/w) / (1/w)
fn interpolate_depth_perspective(
    bary: vec3<f32>,
    z0: f32, z1: f32, z2: f32,
    inv_w0: f32, inv_w1: f32, inv_w2: f32
) -> f32 {
    let inv_w = bary.x * inv_w0 + bary.y * inv_w1 + bary.z * inv_w2;
    let z_over_w = bary.x * (z0 * inv_w0) + bary.y * (z1 * inv_w1) + bary.z * (z2 * inv_w2);
    return z_over_w / inv_w;
}

/// Encode depth as u32 for atomicMin comparison.
///
/// Uses IEEE 754 float bits directly - for positive floats, bit comparison
/// gives the same order as float comparison (reverse-Z: smaller = closer).
fn encode_depth(depth: f32) -> u32 {
    // Clamp to [0, 1] range, apply bias
    let clamped = clamp(depth, 0.0, 1.0);
    // Use bitcast for exact IEEE 754 representation
    return bitcast<u32>(clamped);
}

/// Decode u32 depth back to float.
fn decode_depth(encoded: u32) -> f32 {
    return bitcast<f32>(encoded);
}

// ============================================================================
// Helper Functions: Visibility Buffer Packing
// ============================================================================

/// Pack instance_id and primitive_id into visibility buffer format.
///
/// Format: 32 bits
/// - bits [31:24]: unused (8 bits)
/// - bits [23:0]: instance_id (24 bits, max 16M instances)
///
/// Second word:
/// - bits [31:8]: primitive_id (24 bits, max 16M triangles per mesh)
/// - bits [7:0]: unused (8 bits)
fn pack_visibility(instance_id: u32, primitive_id: u32) -> u32 {
    // Simple packing: instance_id in upper 16 bits, primitive_id in lower 16 bits
    // Supports up to 65K instances and 65K triangles per draw
    return ((instance_id & 0xFFFFu) << 16u) | (primitive_id & 0xFFFFu);
}

/// Unpack instance_id from visibility buffer.
fn unpack_instance_id(packed: u32) -> u32 {
    return packed >> 16u;
}

/// Unpack primitive_id from visibility buffer.
fn unpack_primitive_id(packed: u32) -> u32 {
    return packed & 0xFFFFu;
}

// ============================================================================
// Main Compute Kernel: Per-Triangle Rasterization
// ============================================================================

/// Per-triangle software rasterizer kernel.
///
/// Each workgroup processes one triangle. Threads within the workgroup
/// cooperatively rasterize pixels within the triangle's bounding box.
@compute @workgroup_size(64)
fn rasterize_triangle_tiled(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let triangle_idx = wid.x;

    // Bounds check
    if (triangle_idx >= params.num_triangles) {
        return;
    }

    // Load triangle data
    let clip_tri = triangles[triangle_idx];
    let tri_info = triangle_meta[triangle_idx];

    // Skip if instance/primitive is invalid
    if (tri_info.instance_id == INVALID_INSTANCE_ID) {
        return;
    }

    // Project vertices to screen space
    let s0 = project_vertex(clip_tri.v0, params.viewport);
    let s1 = project_vertex(clip_tri.v1, params.viewport);
    let s2 = project_vertex(clip_tri.v2, params.viewport);

    let p0 = s0.xy;
    let p1 = s1.xy;
    let p2 = s2.xy;

    // Backface culling (if enabled)
    if ((params.flags & 2u) != 0u && is_backfacing(p0, p1, p2)) {
        if (lid.x == 0u) {
            atomicAdd(&stats[1], 1u);
        }
        return;
    }

    // Compute triangle area for degenerate check and barycentric normalization
    let signed_area = edge_function(p0, p1, p2);
    let area = abs(signed_area);

    // Skip degenerate triangles
    if (area < MIN_TRIANGLE_AREA) {
        if (lid.x == 0u) {
            atomicAdd(&stats[1], 1u);
        }
        return;
    }

    // Update processed counter (first thread only)
    if (lid.x == 0u) {
        atomicAdd(&stats[0], 1u);
    }

    // Compute bounding box
    let bbox = compute_triangle_bbox(p0, p1, p2, params.screen_width, params.screen_height);

    // Skip if bbox is off-screen
    if (bbox_is_empty(bbox)) {
        return;
    }

    // Set up edge functions for incremental evaluation
    let edge0 = setup_edge(p1, p2); // Edge opposite to v0
    let edge1 = setup_edge(p2, p0); // Edge opposite to v1
    let edge2 = setup_edge(p0, p1); // Edge opposite to v2

    // Area inverse for barycentric normalization
    let area_inv = 1.0 / signed_area;

    // Calculate how many pixels in the bounding box
    let bbox_width = bbox.max_x - bbox.min_x + 1;
    let bbox_height = bbox.max_y - bbox.min_y + 1;
    let total_pixels = u32(bbox_width * bbox_height);

    // Each thread processes a subset of pixels
    let thread_id = lid.x;
    let num_threads = 64u;

    // Depth values from projected vertices
    let z0 = s0.z;
    let z1 = s1.z;
    let z2 = s2.z;

    // One-over-w for perspective-correct interpolation
    let inv_w0 = 1.0 / clip_tri.v0.w;
    let inv_w1 = 1.0 / clip_tri.v1.w;
    let inv_w2 = 1.0 / clip_tri.v2.w;

    // Pack visibility data once
    let visibility_packed = pack_visibility(tri_info.instance_id, tri_info.primitive_id);

    // Each thread iterates over its assigned pixels
    for (var pixel_idx = thread_id; pixel_idx < total_pixels; pixel_idx += num_threads) {
        // Convert linear index to 2D coordinates
        let local_x = i32(pixel_idx % u32(bbox_width));
        let local_y = i32(pixel_idx / u32(bbox_width));
        let px = bbox.min_x + local_x;
        let py = bbox.min_y + local_y;

        // Sample at pixel center
        let p = vec2<f32>(f32(px) + 0.5, f32(py) + 0.5);

        // Evaluate edge functions
        let e0 = eval_edge(edge0, p);
        let e1 = eval_edge(edge1, p);
        let e2 = eval_edge(edge2, p);

        // Check if point is inside triangle
        // For CCW winding, all edge values should be >= 0 (or <= 0 for CW)
        let inside = (e0 >= 0.0 && e1 >= 0.0 && e2 >= 0.0) ||
                     (e0 <= 0.0 && e1 <= 0.0 && e2 <= 0.0);

        if (!inside) {
            continue;
        }

        // Compute barycentric coordinates
        let bary = vec3<f32>(e0, e1, e2) * area_inv;

        // Interpolate depth with perspective correction
        let depth = interpolate_depth_perspective(bary, z0, z1, z2, inv_w0, inv_w1, inv_w2);

        // Apply depth bias
        let biased_depth = depth + params.depth_bias;

        // Encode depth for atomic comparison
        let encoded_depth = encode_depth(biased_depth);

        // Compute buffer index
        let buffer_idx = u32(py) * params.screen_width + u32(px);

        // Update statistics (fragments tested)
        atomicAdd(&stats[2], 1u);

        // Depth test using atomicMin
        if ((params.flags & 1u) != 0u) {
            // Atomic depth test: only write if we're closer
            let prev_depth = atomicMin(&depth_buffer[buffer_idx].value, encoded_depth);

            if (encoded_depth < prev_depth) {
                // We won the depth test - write visibility data
                atomicStore(&visibility_buffer[buffer_idx], visibility_packed);
                atomicAdd(&stats[3], 1u);
            }
        } else {
            // No depth test - always write
            atomicStore(&depth_buffer[buffer_idx].value, encoded_depth);
            atomicStore(&visibility_buffer[buffer_idx], visibility_packed);
            atomicAdd(&stats[3], 1u);
        }
    }
}

// ============================================================================
// Alternative Entry Point: Linear Per-Triangle
// ============================================================================

/// Linear per-triangle rasterizer for small triangles.
///
/// Each thread processes exactly one triangle. More efficient for
/// triangles that cover few pixels (< 8x8).
@compute @workgroup_size(256)
fn rasterize_small_triangle(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let triangle_idx = gid.x;

    if (triangle_idx >= params.num_triangles) {
        return;
    }

    let clip_tri = triangles[triangle_idx];
    let tri_info = triangle_meta[triangle_idx];

    if (tri_info.instance_id == INVALID_INSTANCE_ID) {
        return;
    }

    // Project vertices
    let s0 = project_vertex(clip_tri.v0, params.viewport);
    let s1 = project_vertex(clip_tri.v1, params.viewport);
    let s2 = project_vertex(clip_tri.v2, params.viewport);

    let p0 = s0.xy;
    let p1 = s1.xy;
    let p2 = s2.xy;

    // Backface culling
    if ((params.flags & 2u) != 0u && is_backfacing(p0, p1, p2)) {
        atomicAdd(&stats[1], 1u);
        return;
    }

    // Degenerate check
    let signed_area = edge_function(p0, p1, p2);
    let area = abs(signed_area);

    if (area < MIN_TRIANGLE_AREA) {
        atomicAdd(&stats[1], 1u);
        return;
    }

    atomicAdd(&stats[0], 1u);

    // Compute bounding box
    let bbox = compute_triangle_bbox(p0, p1, p2, params.screen_width, params.screen_height);

    if (bbox_is_empty(bbox)) {
        return;
    }

    // Set up for rasterization
    let edge0 = setup_edge(p1, p2);
    let edge1 = setup_edge(p2, p0);
    let edge2 = setup_edge(p0, p1);
    let area_inv = 1.0 / signed_area;

    let z0 = s0.z;
    let z1 = s1.z;
    let z2 = s2.z;

    let inv_w0 = 1.0 / clip_tri.v0.w;
    let inv_w1 = 1.0 / clip_tri.v1.w;
    let inv_w2 = 1.0 / clip_tri.v2.w;

    let visibility_packed = pack_visibility(tri_info.instance_id, tri_info.primitive_id);

    // Iterate over bounding box (scanline approach for small triangles)
    for (var py = bbox.min_y; py <= bbox.max_y; py++) {
        for (var px = bbox.min_x; px <= bbox.max_x; px++) {
            let p = vec2<f32>(f32(px) + 0.5, f32(py) + 0.5);

            let e0 = eval_edge(edge0, p);
            let e1 = eval_edge(edge1, p);
            let e2 = eval_edge(edge2, p);

            let inside = (e0 >= 0.0 && e1 >= 0.0 && e2 >= 0.0) ||
                         (e0 <= 0.0 && e1 <= 0.0 && e2 <= 0.0);

            if (!inside) {
                continue;
            }

            let bary = vec3<f32>(e0, e1, e2) * area_inv;
            let depth = interpolate_depth_perspective(bary, z0, z1, z2, inv_w0, inv_w1, inv_w2);
            let biased_depth = depth + params.depth_bias;
            let encoded_depth = encode_depth(biased_depth);

            let buffer_idx = u32(py) * params.screen_width + u32(px);

            atomicAdd(&stats[2], 1u);

            if ((params.flags & 1u) != 0u) {
                let prev_depth = atomicMin(&depth_buffer[buffer_idx].value, encoded_depth);
                if (encoded_depth < prev_depth) {
                    atomicStore(&visibility_buffer[buffer_idx], visibility_packed);
                    atomicAdd(&stats[3], 1u);
                }
            } else {
                atomicStore(&depth_buffer[buffer_idx].value, encoded_depth);
                atomicStore(&visibility_buffer[buffer_idx], visibility_packed);
                atomicAdd(&stats[3], 1u);
            }
        }
    }
}

// ============================================================================
// Utility Entry Points
// ============================================================================

/// Clear depth buffer to far plane.
@compute @workgroup_size(256)
fn clear_depth_buffer(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    let total_pixels = params.screen_width * params.screen_height;

    if (idx >= total_pixels) {
        return;
    }

    atomicStore(&depth_buffer[idx].value, DEPTH_CLEAR_VALUE);
}

/// Clear visibility buffer to invalid state.
@compute @workgroup_size(256)
fn clear_visibility_buffer(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    let total_pixels = params.screen_width * params.screen_height;

    if (idx >= total_pixels) {
        return;
    }

    atomicStore(&visibility_buffer[idx], INVALID_INSTANCE_ID);
}

/// Reset statistics counters.
@compute @workgroup_size(1)
fn clear_stats() {
    atomicStore(&stats[0], 0u);
    atomicStore(&stats[1], 0u);
    atomicStore(&stats[2], 0u);
    atomicStore(&stats[3], 0u);
}
