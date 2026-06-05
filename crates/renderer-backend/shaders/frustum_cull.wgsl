// SPDX-License-Identifier: MIT
//
// frustum_cull.wgsl - AABB-Frustum Intersection Test (T-WGPU-P6.3.2)
//
// Implements optimized AABB-frustum intersection testing for GPU-based culling.
// Uses the p-vertex optimization: for each plane, only test the corner most
// aligned with the plane normal for early rejection.
//
// Memory Layout:
//   FrustumPlane: 16 bytes (vec3 normal + f32 distance)
//   FrustumPlanes: 96 bytes (6 planes x 16 bytes)
//   AABB: 32 bytes (vec3 min + pad + vec3 max + pad)
//
// Performance:
//   - 6 plane tests with early-out on first cull
//   - P-vertex optimization reduces per-plane work
//   - Branch-free select() for p-vertex computation
//   - O(1) per AABB, O(n) for n objects

// ============================================================================
// Constants
// ============================================================================

/// Number of frustum planes (left, right, bottom, top, near, far)
const NUM_FRUSTUM_PLANES: u32 = 6u;

/// Plane indices for readability
const PLANE_LEFT: u32 = 0u;
const PLANE_RIGHT: u32 = 1u;
const PLANE_BOTTOM: u32 = 2u;
const PLANE_TOP: u32 = 3u;
const PLANE_NEAR: u32 = 4u;
const PLANE_FAR: u32 = 5u;

/// Visibility result constants for test_aabb_frustum_detailed
const VISIBILITY_OUTSIDE: u32 = 0u;      // Fully outside frustum (culled)
const VISIBILITY_INTERSECTING: u32 = 1u; // Partially inside frustum
const VISIBILITY_INSIDE: u32 = 2u;        // Fully inside frustum

// ============================================================================
// Structs
// ============================================================================

/// A frustum plane in Hessian normal form.
///
/// Plane equation: dot(normal, point) + distance = 0
/// Points with dot(normal, point) + distance > 0 are on the positive side
/// (inside the frustum).
///
/// All plane normals point inward toward the frustum interior.
struct FrustumPlane {
    /// Normalized plane normal pointing into frustum
    normal: vec3<f32>,
    /// Signed distance from origin (d in ax+by+cz+d=0)
    distance: f32,
}

/// The 6 frustum planes for visibility testing.
///
/// Plane order: left, right, bottom, top, near, far
/// Total size: 96 bytes (6 x 16 bytes)
struct FrustumPlanes {
    planes: array<FrustumPlane, 6>,
}

/// Axis-Aligned Bounding Box for culling.
///
/// Stored as min/max corners in world space.
/// Total size: 32 bytes (with padding for GPU alignment)
struct AABB {
    /// Minimum corner (x_min, y_min, z_min)
    min: vec3<f32>,
    /// Padding for 16-byte alignment
    _pad0: f32,
    /// Maximum corner (x_max, y_max, z_max)
    max: vec3<f32>,
    /// Padding for 16-byte alignment
    _pad1: f32,
}

// ============================================================================
// Bindings
// ============================================================================

/// Frustum planes uniform buffer (96 bytes)
@group(0) @binding(0) var<uniform> frustum: FrustumPlanes;

// ============================================================================
// Helper Functions
// ============================================================================

/// Compute signed distance from a point to a plane.
///
/// Returns:
///   > 0: point is on positive side (inside frustum)
///   = 0: point is on the plane
///   < 0: point is on negative side (outside frustum)
fn signed_distance_to_plane(point: vec3<f32>, plane: FrustumPlane) -> f32 {
    return dot(plane.normal, point) + plane.distance;
}

/// Get the p-vertex (positive vertex) of an AABB for a given plane.
///
/// The p-vertex is the corner of the AABB most aligned with the plane normal.
/// If the p-vertex is outside the plane, the entire AABB is outside.
///
/// This uses branch-free select() for optimal GPU performance.
fn get_p_vertex(aabb_min: vec3<f32>, aabb_max: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        select(aabb_min.x, aabb_max.x, normal.x >= 0.0),
        select(aabb_min.y, aabb_max.y, normal.y >= 0.0),
        select(aabb_min.z, aabb_max.z, normal.z >= 0.0),
    );
}

/// Get the n-vertex (negative vertex) of an AABB for a given plane.
///
/// The n-vertex is the corner of the AABB least aligned with the plane normal.
/// If the n-vertex is inside the plane, the entire AABB is inside that plane.
fn get_n_vertex(aabb_min: vec3<f32>, aabb_max: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        select(aabb_max.x, aabb_min.x, normal.x >= 0.0),
        select(aabb_max.y, aabb_min.y, normal.y >= 0.0),
        select(aabb_max.z, aabb_min.z, normal.z >= 0.0),
    );
}

// ============================================================================
// Core AABB-Frustum Test Functions
// ============================================================================

/// Test if an AABB is visible (not culled) against the frustum.
///
/// Uses the p-vertex optimization: for each frustum plane, only test the
/// corner most aligned with the plane normal. If that corner is outside
/// the plane, the entire AABB is outside.
///
/// Returns:
///   true  - AABB is visible (inside or intersecting frustum)
///   false - AABB is fully outside frustum (culled)
///
/// Performance:
///   - Early out on first cull (avg ~3 plane tests for culled objects)
///   - 6 plane tests maximum for visible objects
///   - Branch-free p-vertex computation
fn test_aabb_frustum(aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> bool {
    // Test against each of the 6 frustum planes
    for (var i = 0u; i < NUM_FRUSTUM_PLANES; i = i + 1u) {
        let plane = frustum.planes[i];

        // P-vertex: corner most aligned with plane normal
        // If normal component is positive, use max; otherwise use min
        let p = vec3<f32>(
            select(aabb_min.x, aabb_max.x, plane.normal.x >= 0.0),
            select(aabb_min.y, aabb_max.y, plane.normal.y >= 0.0),
            select(aabb_min.z, aabb_max.z, plane.normal.z >= 0.0),
        );

        // If p-vertex is outside plane, entire AABB is outside
        if (dot(plane.normal, p) + plane.distance < 0.0) {
            return false; // Culled
        }
    }
    return true; // Visible
}

/// Test AABB against frustum with detailed result.
///
/// Returns visibility classification:
///   0 (VISIBILITY_OUTSIDE)      - AABB is fully outside frustum (culled)
///   1 (VISIBILITY_INTERSECTING) - AABB intersects frustum boundary
///   2 (VISIBILITY_INSIDE)       - AABB is fully inside frustum
///
/// This is useful for LOD selection or hierarchical culling where you want
/// to know if children can skip culling tests (fully inside parent).
///
/// Performance: ~2x slower than basic test due to n-vertex checks
fn test_aabb_frustum_detailed(aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> u32 {
    // Track if AABB is fully inside all planes
    var fully_inside = true;

    for (var i = 0u; i < NUM_FRUSTUM_PLANES; i = i + 1u) {
        let plane = frustum.planes[i];

        // P-vertex: corner most aligned with normal (furthest in normal direction)
        let p = vec3<f32>(
            select(aabb_min.x, aabb_max.x, plane.normal.x >= 0.0),
            select(aabb_min.y, aabb_max.y, plane.normal.y >= 0.0),
            select(aabb_min.z, aabb_max.z, plane.normal.z >= 0.0),
        );

        let p_dist = dot(plane.normal, p) + plane.distance;

        // If p-vertex is outside, entire AABB is outside
        if (p_dist < 0.0) {
            return VISIBILITY_OUTSIDE;
        }

        // N-vertex: corner least aligned with normal (closest in normal direction)
        let n = vec3<f32>(
            select(aabb_max.x, aabb_min.x, plane.normal.x >= 0.0),
            select(aabb_max.y, aabb_min.y, plane.normal.y >= 0.0),
            select(aabb_max.z, aabb_min.z, plane.normal.z >= 0.0),
        );

        let n_dist = dot(plane.normal, n) + plane.distance;

        // If n-vertex is outside, AABB straddles this plane (intersecting)
        if (n_dist < 0.0) {
            fully_inside = false;
        }
    }

    return select(VISIBILITY_INTERSECTING, VISIBILITY_INSIDE, fully_inside);
}

/// Test AABB against a single frustum plane.
///
/// Returns:
///   true  - AABB is on positive side or intersecting the plane
///   false - AABB is fully on negative side (outside)
///
/// Useful for custom culling with subset of planes.
fn test_aabb_plane(aabb_min: vec3<f32>, aabb_max: vec3<f32>, plane_index: u32) -> bool {
    let plane = frustum.planes[plane_index];

    let p = vec3<f32>(
        select(aabb_min.x, aabb_max.x, plane.normal.x >= 0.0),
        select(aabb_min.y, aabb_max.y, plane.normal.y >= 0.0),
        select(aabb_min.z, aabb_max.z, plane.normal.z >= 0.0),
    );

    return dot(plane.normal, p) + plane.distance >= 0.0;
}

/// Test AABB against frustum using raw plane data (no uniform binding).
///
/// This variant accepts plane data directly, useful for:
///   - Shadow cascades with different frustums
///   - Portal culling with arbitrary frustums
///   - Testing against user-defined clip planes
///
/// planes: array of 6 FrustumPlane structs
fn test_aabb_frustum_raw(
    aabb_min: vec3<f32>,
    aabb_max: vec3<f32>,
    planes: array<FrustumPlane, 6>
) -> bool {
    for (var i = 0u; i < 6u; i = i + 1u) {
        let plane = planes[i];

        let p = vec3<f32>(
            select(aabb_min.x, aabb_max.x, plane.normal.x >= 0.0),
            select(aabb_min.y, aabb_max.y, plane.normal.y >= 0.0),
            select(aabb_min.z, aabb_max.z, plane.normal.z >= 0.0),
        );

        if (dot(plane.normal, p) + plane.distance < 0.0) {
            return false;
        }
    }
    return true;
}

// ============================================================================
// Transformed AABB Support
// ============================================================================

/// Test an oriented bounding box (OBB) against the frustum.
///
/// For transformed AABBs (rotated/scaled), compute world-space AABB bounds
/// and call this function. The local_to_world matrix transforms the local
/// AABB to world space.
///
/// Note: For best performance, pre-compute world-space AABBs on CPU/GPU
/// during instance update and use test_aabb_frustum() directly.
fn test_obb_frustum(
    local_min: vec3<f32>,
    local_max: vec3<f32>,
    local_to_world: mat4x4<f32>
) -> bool {
    // Compute the 8 corners of the local AABB
    let corners = array<vec3<f32>, 8>(
        vec3<f32>(local_min.x, local_min.y, local_min.z),
        vec3<f32>(local_max.x, local_min.y, local_min.z),
        vec3<f32>(local_min.x, local_max.y, local_min.z),
        vec3<f32>(local_max.x, local_max.y, local_min.z),
        vec3<f32>(local_min.x, local_min.y, local_max.z),
        vec3<f32>(local_max.x, local_min.y, local_max.z),
        vec3<f32>(local_min.x, local_max.y, local_max.z),
        vec3<f32>(local_max.x, local_max.y, local_max.z),
    );

    // Transform corners to world space and compute world AABB
    var world_min = vec3<f32>(1e30, 1e30, 1e30);
    var world_max = vec3<f32>(-1e30, -1e30, -1e30);

    for (var i = 0u; i < 8u; i = i + 1u) {
        let world_pos = (local_to_world * vec4<f32>(corners[i], 1.0)).xyz;
        world_min = min(world_min, world_pos);
        world_max = max(world_max, world_pos);
    }

    // Test the world-space AABB
    return test_aabb_frustum(world_min, world_max);
}

// ============================================================================
// Batch Culling (Compute Shader Entry Points)
// ============================================================================

/// Parameters for batch frustum culling
struct CullParams {
    /// Number of AABBs to process
    num_objects: u32,
    /// Flags (reserved for future use)
    flags: u32,
    /// Padding for 16-byte alignment
    _pad0: u32,
    _pad1: u32,
}

/// Input AABB for batch culling (matches InstanceBounds in frustum_cull.rs)
struct InputAABB {
    min: vec3<f32>,
    _pad0: f32,
    max: vec3<f32>,
    _pad1: f32,
}

// Batch culling bindings (used by compute entry points)
@group(1) @binding(0) var<uniform> cull_params: CullParams;
@group(1) @binding(1) var<storage, read> input_aabbs: array<InputAABB>;
@group(1) @binding(2) var<storage, read_write> visibility_flags: array<u32>;

/// Workgroup size for compute shaders
const WORKGROUP_SIZE: u32 = 256u;

/// Batch AABB frustum culling compute shader.
///
/// One thread per AABB. Outputs 1 for visible, 0 for culled.
@compute @workgroup_size(256)
fn cull_aabb_batch(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;

    // Bounds check
    if (idx >= cull_params.num_objects) {
        return;
    }

    let aabb = input_aabbs[idx];
    let visible = test_aabb_frustum(aabb.min, aabb.max);

    visibility_flags[idx] = select(0u, 1u, visible);
}

/// Batch AABB frustum culling with detailed visibility output.
///
/// Outputs: 0 = outside, 1 = intersecting, 2 = inside
@compute @workgroup_size(256)
fn cull_aabb_batch_detailed(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;

    if (idx >= cull_params.num_objects) {
        return;
    }

    let aabb = input_aabbs[idx];
    let result = test_aabb_frustum_detailed(aabb.min, aabb.max);

    visibility_flags[idx] = result;
}
