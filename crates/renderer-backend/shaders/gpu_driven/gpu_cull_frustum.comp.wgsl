// SPDX-License-Identifier: MIT
//
// gpu_cull_frustum.comp.wgsl - GPU Frustum Culling for TRINITY Engine (T-GPU-3.1)
//
// Performs sphere-then-AABB frustum culling for GPU-driven rendering.
// One thread per instance, outputs visibility flags.
//
// Algorithm:
// 1. Quick bounding sphere test against all 6 frustum planes
// 2. If sphere passes, precise AABB test using p/n vertex optimization
// 3. Zero-radius spheres fall back to AABB-only test
//
// Performance: O(n) work, single dispatch, <0.1ms for 100K instances

// ============================================================================
// Constants
// ============================================================================

const WORKGROUP_SIZE: u32 = 256u;

/// Number of frustum planes (left, right, bottom, top, near, far)
const NUM_PLANES: u32 = 6u;

// ============================================================================
// Structs
// ============================================================================

/// Frustum plane in Hessian normal form: (normal.xyz, distance)
/// Plane equation: dot(normal, point) + distance = 0
/// Points on positive side (inside frustum) have dot(normal, point) + distance > 0
struct FrustumPlane {
    /// Plane normal (normalized, pointing inward toward frustum interior)
    normal: vec3<f32>,
    /// Signed distance from origin (d in plane equation)
    distance: f32,
}

/// Culling parameters uniform buffer
struct CullParams {
    /// Number of instances to process
    num_instances: u32,
    /// Reserved for future flags (e.g., enable sphere test, debug mode)
    flags: u32,
    /// Padding for 16-byte alignment
    _pad0: u32,
    _pad1: u32,
}

/// View frustum with 6 planes
/// Order: left, right, bottom, top, near, far
struct Frustum {
    planes: array<FrustumPlane, 6>,
}

/// Bounding data for a single instance
/// Contains both sphere (fast) and AABB (precise) bounds
struct InstanceBounds {
    // Bounding sphere for quick rejection test
    /// Center of bounding sphere in world space
    sphere_center: vec3<f32>,
    /// Radius of bounding sphere (0 = use AABB only)
    sphere_radius: f32,

    // Axis-aligned bounding box for precise test
    /// Minimum corner of AABB in world space
    aabb_min: vec3<f32>,
    /// Padding for vec4 alignment
    _pad0: f32,

    /// Maximum corner of AABB in world space
    aabb_max: vec3<f32>,
    /// Padding for vec4 alignment
    _pad1: f32,
}

// ============================================================================
// Bindings
// ============================================================================

/// Culling parameters (uniform buffer)
@group(0) @binding(0) var<uniform> params: CullParams;

/// View frustum planes (uniform buffer)
@group(0) @binding(1) var<uniform> frustum: Frustum;

/// Instance bounds array (read-only storage buffer)
@group(0) @binding(2) var<storage, read> instances: array<InstanceBounds>;

/// Output visibility flags: 1 = visible, 0 = culled (read-write storage buffer)
@group(0) @binding(3) var<storage, read_write> visibility: array<u32>;

// ============================================================================
// Plane Tests
// ============================================================================

/// Compute signed distance from point to plane.
/// Positive = inside frustum (on normal side), Negative = outside
fn point_plane_distance(point: vec3<f32>, plane: FrustumPlane) -> f32 {
    return dot(plane.normal, point) + plane.distance;
}

/// Test bounding sphere against a single frustum plane.
/// Returns the signed distance from sphere surface to plane.
/// If result < 0, sphere is entirely outside this plane.
fn sphere_plane_test(center: vec3<f32>, radius: f32, plane: FrustumPlane) -> f32 {
    let center_dist = point_plane_distance(center, plane);
    // If center_dist + radius < 0, the entire sphere is on the negative side
    return center_dist + radius;
}

/// Test bounding sphere against all 6 frustum planes.
/// Returns true if sphere is visible (inside or intersecting frustum).
fn sphere_frustum_test(center: vec3<f32>, radius: f32) -> bool {
    // Test against each plane in sequence
    // Early exit on first failure (sphere entirely outside a plane)
    for (var i = 0u; i < NUM_PLANES; i++) {
        if (sphere_plane_test(center, radius, frustum.planes[i]) < 0.0) {
            return false; // Sphere is entirely outside this plane
        }
    }
    return true; // Sphere passed all plane tests
}

// ============================================================================
// AABB Tests
// ============================================================================

/// Get the "positive vertex" (p-vertex) of an AABB for a given plane normal.
/// This is the corner of the AABB most aligned with the plane normal.
/// If p-vertex is outside the plane, the entire AABB is outside.
fn aabb_positive_vertex(aabb_min: vec3<f32>, aabb_max: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        select(aabb_min.x, aabb_max.x, normal.x >= 0.0),
        select(aabb_min.y, aabb_max.y, normal.y >= 0.0),
        select(aabb_min.z, aabb_max.z, normal.z >= 0.0)
    );
}

/// Get the "negative vertex" (n-vertex) of an AABB for a given plane normal.
/// This is the corner of the AABB least aligned with the plane normal.
/// Used for determining if AABB is entirely inside the plane.
fn aabb_negative_vertex(aabb_min: vec3<f32>, aabb_max: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        select(aabb_max.x, aabb_min.x, normal.x >= 0.0),
        select(aabb_max.y, aabb_min.y, normal.y >= 0.0),
        select(aabb_max.z, aabb_min.z, normal.z >= 0.0)
    );
}

/// Test AABB against all 6 frustum planes using p/n vertex optimization.
/// Returns true if AABB is visible (inside or intersecting frustum).
fn aabb_frustum_test(aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> bool {
    for (var i = 0u; i < NUM_PLANES; i++) {
        let plane = frustum.planes[i];

        // Get the corner most aligned with plane normal
        let p_vertex = aabb_positive_vertex(aabb_min, aabb_max, plane.normal);

        // If p-vertex is outside the plane, entire AABB is outside
        if (point_plane_distance(p_vertex, plane) < 0.0) {
            return false; // AABB entirely outside this plane
        }
    }
    return true; // AABB passed all plane tests
}

// ============================================================================
// Culling Flags
// ============================================================================

/// Flag bit: Enable sphere test (skip if bounds.sphere_radius == 0)
const FLAG_USE_SPHERE: u32 = 1u;
/// Flag bit: Debug mode (always mark visible for verification)
const FLAG_DEBUG_VISIBLE: u32 = 2u;

// ============================================================================
// Main Compute Kernel
// ============================================================================

/// Per-instance frustum culling kernel.
///
/// Each thread processes one instance:
/// 1. If sphere_radius > 0: Quick sphere test first
/// 2. If sphere passes (or radius == 0): Precise AABB test
/// 3. Write visibility flag to output buffer
@compute @workgroup_size(256)
fn cull_frustum(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;

    // Bounds check: skip threads beyond instance count
    if (idx >= params.num_instances) {
        return;
    }

    let bounds = instances[idx];
    var visible: bool = false;

    // Debug mode: always mark visible
    if ((params.flags & FLAG_DEBUG_VISIBLE) != 0u) {
        visible = true;
    } else if (bounds.sphere_radius <= 0.0) {
        // No valid bounding sphere: use AABB test only
        // This handles degenerate cases and point-like objects
        visible = aabb_frustum_test(bounds.aabb_min, bounds.aabb_max);
    } else {
        // Two-phase test: sphere first (cheap), then AABB (precise)

        // Phase 1: Quick sphere rejection test
        let sphere_visible = sphere_frustum_test(bounds.sphere_center, bounds.sphere_radius);

        if (sphere_visible) {
            // Phase 2: Precise AABB test (sphere might be false positive)
            // Bounding sphere is conservative, so we refine with AABB
            visible = aabb_frustum_test(bounds.aabb_min, bounds.aabb_max);
        }
        // If sphere failed, definitely culled (no need for AABB test)
    }

    // Write visibility flag
    visibility[idx] = select(0u, 1u, visible);
}

// ============================================================================
// Alternative Entry Points
// ============================================================================

/// Sphere-only culling (faster, less precise).
/// Use when AABB refinement isn't needed.
@compute @workgroup_size(256)
fn cull_frustum_sphere_only(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;

    if (idx >= params.num_instances) {
        return;
    }

    let bounds = instances[idx];
    var visible: bool = false;

    if (bounds.sphere_radius <= 0.0) {
        // No sphere: fall back to AABB
        visible = aabb_frustum_test(bounds.aabb_min, bounds.aabb_max);
    } else {
        visible = sphere_frustum_test(bounds.sphere_center, bounds.sphere_radius);
    }

    visibility[idx] = select(0u, 1u, visible);
}

/// AABB-only culling (more precise, slightly slower).
/// Use when bounding spheres are not available.
@compute @workgroup_size(256)
fn cull_frustum_aabb_only(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;

    if (idx >= params.num_instances) {
        return;
    }

    let bounds = instances[idx];
    let visible = aabb_frustum_test(bounds.aabb_min, bounds.aabb_max);
    visibility[idx] = select(0u, 1u, visible);
}
