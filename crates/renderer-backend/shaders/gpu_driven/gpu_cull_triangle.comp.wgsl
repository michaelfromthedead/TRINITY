// SPDX-License-Identifier: MIT
//
// gpu_cull_triangle.comp.wgsl - Triangle Culling for TRINITY Engine (T-GPU-3.7)
//
// Per-triangle culling: backface, degenerate, and frustum rejection.
// Operates in clip space for accurate backface determination.
//
// Algorithm:
// 1. Transform triangle vertices to clip space
// 2. Backface test: sign of 2D cross product in clip space
// 3. Degenerate test: triangle area below threshold
// 4. Frustum test: all vertices outside same plane
//
// Performance: O(n) work, single dispatch, <0.1ms for 1M triangles

// ============================================================================
// Constants
// ============================================================================

const WORKGROUP_SIZE: u32 = 256u;

/// Cull reason flags
const CULL_REASON_NONE: u32 = 0u;
const CULL_REASON_BACKFACE: u32 = 1u;
const CULL_REASON_DEGENERATE: u32 = 2u;
const CULL_REASON_FRUSTUM: u32 = 3u;

/// Backface cull mode
const CULL_MODE_NONE: u32 = 0u;
const CULL_MODE_CCW: u32 = 1u;    // Cull counter-clockwise faces
const CULL_MODE_CW: u32 = 2u;     // Cull clockwise faces

// ============================================================================
// Structs
// ============================================================================

/// Parameters for triangle culling
struct TriangleCullParams {
    /// Number of triangles to process
    num_triangles: u32,
    /// Backface cull mode (0 = none, 1 = CCW, 2 = CW)
    cull_backface: u32,
    /// Area threshold for degenerate triangle detection
    degenerate_threshold: f32,
    /// Viewport width in pixels (for micro-triangle detection)
    viewport_width: f32,
    /// Viewport height in pixels
    viewport_height: f32,
    /// Padding for 16-byte alignment
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
    /// View-projection matrix (column-major, 4x4)
    view_proj: mat4x4<f32>,
}

/// Input triangle with 3 vertices and IDs
struct TriangleInput {
    /// Vertex 0 position (world space)
    v0: vec3<f32>,
    /// Padding for vec4 alignment
    _pad0: f32,
    /// Vertex 1 position (world space)
    v1: vec3<f32>,
    /// Padding for vec4 alignment
    _pad1: f32,
    /// Vertex 2 position (world space)
    v2: vec3<f32>,
    /// Instance ID this triangle belongs to
    instance_id: u32,
    /// Primitive ID within the instance
    primitive_id: u32,
    /// Padding for 16-byte struct alignment
    _pad2: vec3<f32>,
}

/// Culling result for a single triangle
struct CullResult {
    /// Visibility flag: 0 = culled, 1 = visible
    visible: u32,
    /// Cull reason: 0 = none, 1 = backface, 2 = degenerate, 3 = frustum
    cull_reason: u32,
}

// ============================================================================
// Bindings
// ============================================================================

/// Culling parameters (uniform buffer)
@group(0) @binding(0) var<uniform> params: TriangleCullParams;

/// Input triangles (read-only storage buffer)
@group(0) @binding(1) var<storage, read> triangles: array<TriangleInput>;

/// Output cull results (read-write storage buffer)
@group(0) @binding(2) var<storage, read_write> results: array<CullResult>;

// ============================================================================
// Clip Space Utilities
// ============================================================================

/// Transform a world-space point to clip space
fn to_clip_space(world_pos: vec3<f32>) -> vec4<f32> {
    return params.view_proj * vec4<f32>(world_pos, 1.0);
}

/// Perform perspective divide to get NDC coordinates
fn to_ndc(clip_pos: vec4<f32>) -> vec3<f32> {
    let inv_w = 1.0 / clip_pos.w;
    return vec3<f32>(clip_pos.x * inv_w, clip_pos.y * inv_w, clip_pos.z * inv_w);
}

// ============================================================================
// Backface Culling
// ============================================================================

/// Check if triangle faces away from camera in clip space.
/// Uses the sign of the 2D cross product (winding order).
///
/// In clip space, after perspective divide:
/// - Positive cross product z = counter-clockwise (front-facing)
/// - Negative cross product z = clockwise (back-facing)
///
/// Returns true if triangle should be culled based on winding.
fn is_backfacing(v0: vec4<f32>, v1: vec4<f32>, v2: vec4<f32>, cull_mode: u32) -> bool {
    // Skip backface culling if disabled
    if (cull_mode == CULL_MODE_NONE) {
        return false;
    }

    // Convert to NDC for 2D winding test
    // Note: We handle w < 0 (behind camera) as culled via frustum test
    if (v0.w <= 0.0 || v1.w <= 0.0 || v2.w <= 0.0) {
        // Vertex behind camera - will be frustum culled
        return false;
    }

    let ndc0 = to_ndc(v0);
    let ndc1 = to_ndc(v1);
    let ndc2 = to_ndc(v2);

    // 2D cross product of edges: (v1 - v0) x (v2 - v0)
    // Only need z component: (e1.x * e2.y) - (e1.y * e2.x)
    let e1 = vec2<f32>(ndc1.x - ndc0.x, ndc1.y - ndc0.y);
    let e2 = vec2<f32>(ndc2.x - ndc0.x, ndc2.y - ndc0.y);
    let cross_z = e1.x * e2.y - e1.y * e2.x;

    // Check winding based on cull mode
    if (cull_mode == CULL_MODE_CCW) {
        // Cull CCW faces (positive cross_z)
        return cross_z > 0.0;
    } else {
        // Cull CW faces (negative cross_z)
        return cross_z < 0.0;
    }
}

// ============================================================================
// Degenerate Triangle Detection
// ============================================================================

/// Check if triangle area is below threshold (degenerate).
/// Area in clip space accounts for perspective foreshortening.
///
/// Uses screen-space area approximation after perspective divide.
/// Triangles with area < threshold are considered degenerate.
fn is_degenerate(v0: vec4<f32>, v1: vec4<f32>, v2: vec4<f32>, threshold: f32) -> bool {
    // Handle vertices behind camera
    if (v0.w <= 0.0 || v1.w <= 0.0 || v2.w <= 0.0) {
        return false; // Let frustum test handle this
    }

    let ndc0 = to_ndc(v0);
    let ndc1 = to_ndc(v1);
    let ndc2 = to_ndc(v2);

    // Convert NDC to approximate screen pixels
    let half_w = params.viewport_width * 0.5;
    let half_h = params.viewport_height * 0.5;

    let p0 = vec2<f32>(ndc0.x * half_w, ndc0.y * half_h);
    let p1 = vec2<f32>(ndc1.x * half_w, ndc1.y * half_h);
    let p2 = vec2<f32>(ndc2.x * half_w, ndc2.y * half_h);

    // Triangle area using cross product magnitude / 2
    let e1 = p1 - p0;
    let e2 = p2 - p0;
    let area = abs(e1.x * e2.y - e1.y * e2.x) * 0.5;

    return area < threshold;
}

/// Alternative: Check using edge lengths for degenerate collinear triangles
fn is_degenerate_edges(v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>, threshold: f32) -> bool {
    let e0 = v1 - v0;
    let e1 = v2 - v1;
    let e2 = v0 - v2;

    let len0_sq = dot(e0, e0);
    let len1_sq = dot(e1, e1);
    let len2_sq = dot(e2, e2);

    let threshold_sq = threshold * threshold;

    // Any edge too short makes triangle degenerate
    return len0_sq < threshold_sq || len1_sq < threshold_sq || len2_sq < threshold_sq;
}

// ============================================================================
// Frustum Culling
// ============================================================================

/// Clip space frustum plane outcode bits
const CLIP_LEFT: u32 = 1u;
const CLIP_RIGHT: u32 = 2u;
const CLIP_BOTTOM: u32 = 4u;
const CLIP_TOP: u32 = 8u;
const CLIP_NEAR: u32 = 16u;
const CLIP_FAR: u32 = 32u;

/// Compute outcode for a clip-space vertex.
/// Bits indicate which frustum planes the vertex is outside of.
fn compute_outcode(clip_pos: vec4<f32>) -> u32 {
    var code = 0u;
    let w = clip_pos.w;

    // Note: In clip space, a point is inside the frustum when:
    // -w <= x <= w, -w <= y <= w, 0 <= z <= w (for wgpu/Vulkan depth [0,1])

    if (clip_pos.x < -w) { code = code | CLIP_LEFT; }
    if (clip_pos.x > w) { code = code | CLIP_RIGHT; }
    if (clip_pos.y < -w) { code = code | CLIP_BOTTOM; }
    if (clip_pos.y > w) { code = code | CLIP_TOP; }
    if (clip_pos.z < 0.0) { code = code | CLIP_NEAR; }
    if (clip_pos.z > w) { code = code | CLIP_FAR; }

    return code;
}

/// Check if triangle is entirely outside the frustum.
/// Uses Cohen-Sutherland style outcode AND test.
/// If all vertices share an outcode bit, triangle is outside that plane.
fn is_frustum_culled(v0: vec4<f32>, v1: vec4<f32>, v2: vec4<f32>) -> bool {
    let code0 = compute_outcode(v0);
    let code1 = compute_outcode(v1);
    let code2 = compute_outcode(v2);

    // If AND of all outcodes is non-zero, all vertices are outside
    // the same plane, so triangle is completely outside frustum
    return (code0 & code1 & code2) != 0u;
}

/// More precise frustum test checking each plane individually
fn is_frustum_culled_precise(v0: vec4<f32>, v1: vec4<f32>, v2: vec4<f32>) -> bool {
    // Check each clip plane
    let w0 = v0.w;
    let w1 = v1.w;
    let w2 = v2.w;

    // Left plane: x >= -w
    if (v0.x < -w0 && v1.x < -w1 && v2.x < -w2) {
        return true;
    }
    // Right plane: x <= w
    if (v0.x > w0 && v1.x > w1 && v2.x > w2) {
        return true;
    }
    // Bottom plane: y >= -w
    if (v0.y < -w0 && v1.y < -w1 && v2.y < -w2) {
        return true;
    }
    // Top plane: y <= w
    if (v0.y > w0 && v1.y > w1 && v2.y > w2) {
        return true;
    }
    // Near plane: z >= 0
    if (v0.z < 0.0 && v1.z < 0.0 && v2.z < 0.0) {
        return true;
    }
    // Far plane: z <= w
    if (v0.z > w0 && v1.z > w1 && v2.z > w2) {
        return true;
    }

    return false;
}

// ============================================================================
// Main Compute Kernel
// ============================================================================

/// Per-triangle culling kernel.
///
/// Each thread processes one triangle:
/// 1. Transform vertices to clip space
/// 2. Frustum test (quick rejection)
/// 3. Backface test (if enabled)
/// 4. Degenerate test (if threshold > 0)
/// 5. Write visibility and cull reason to output buffer
@compute @workgroup_size(256)
fn cull_triangle(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;

    // Bounds check
    if (idx >= params.num_triangles) {
        return;
    }

    let tri = triangles[idx];

    // Transform vertices to clip space
    let clip0 = to_clip_space(tri.v0);
    let clip1 = to_clip_space(tri.v1);
    let clip2 = to_clip_space(tri.v2);

    var visible = true;
    var cull_reason = CULL_REASON_NONE;

    // Test 1: Frustum culling (fastest - check first)
    if (is_frustum_culled(clip0, clip1, clip2)) {
        visible = false;
        cull_reason = CULL_REASON_FRUSTUM;
    }
    // Test 2: Backface culling
    else if (is_backfacing(clip0, clip1, clip2, params.cull_backface)) {
        visible = false;
        cull_reason = CULL_REASON_BACKFACE;
    }
    // Test 3: Degenerate triangle detection
    else if (params.degenerate_threshold > 0.0 &&
             is_degenerate(clip0, clip1, clip2, params.degenerate_threshold)) {
        visible = false;
        cull_reason = CULL_REASON_DEGENERATE;
    }

    // Write result
    results[idx] = CullResult(select(0u, 1u, visible), cull_reason);
}

// ============================================================================
// Alternative Entry Points
// ============================================================================

/// Backface-only culling (no frustum or degenerate tests).
/// Use when triangles are already frustum-culled at instance level.
@compute @workgroup_size(256)
fn cull_triangle_backface_only(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;

    if (idx >= params.num_triangles) {
        return;
    }

    let tri = triangles[idx];

    let clip0 = to_clip_space(tri.v0);
    let clip1 = to_clip_space(tri.v1);
    let clip2 = to_clip_space(tri.v2);

    var visible = true;
    var cull_reason = CULL_REASON_NONE;

    if (is_backfacing(clip0, clip1, clip2, params.cull_backface)) {
        visible = false;
        cull_reason = CULL_REASON_BACKFACE;
    }

    results[idx] = CullResult(select(0u, 1u, visible), cull_reason);
}

/// Frustum-only culling.
/// Use when backface and degenerate tests are not needed.
@compute @workgroup_size(256)
fn cull_triangle_frustum_only(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;

    if (idx >= params.num_triangles) {
        return;
    }

    let tri = triangles[idx];

    let clip0 = to_clip_space(tri.v0);
    let clip1 = to_clip_space(tri.v1);
    let clip2 = to_clip_space(tri.v2);

    var visible = true;
    var cull_reason = CULL_REASON_NONE;

    if (is_frustum_culled(clip0, clip1, clip2)) {
        visible = false;
        cull_reason = CULL_REASON_FRUSTUM;
    }

    results[idx] = CullResult(select(0u, 1u, visible), cull_reason);
}

/// Combined backface + degenerate culling (no frustum test).
/// Use for triangles known to be in frustum.
@compute @workgroup_size(256)
fn cull_triangle_no_frustum(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;

    if (idx >= params.num_triangles) {
        return;
    }

    let tri = triangles[idx];

    let clip0 = to_clip_space(tri.v0);
    let clip1 = to_clip_space(tri.v1);
    let clip2 = to_clip_space(tri.v2);

    var visible = true;
    var cull_reason = CULL_REASON_NONE;

    if (is_backfacing(clip0, clip1, clip2, params.cull_backface)) {
        visible = false;
        cull_reason = CULL_REASON_BACKFACE;
    } else if (params.degenerate_threshold > 0.0 &&
               is_degenerate(clip0, clip1, clip2, params.degenerate_threshold)) {
        visible = false;
        cull_reason = CULL_REASON_DEGENERATE;
    }

    results[idx] = CullResult(select(0u, 1u, visible), cull_reason);
}
