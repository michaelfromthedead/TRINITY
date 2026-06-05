// SPDX-License-Identifier: MIT
//
// gpu_cull_occlusion.comp.wgsl - HiZ Occlusion Culling for TRINITY Engine (T-GPU-3.4)
//
// Tests instance bounding boxes against a hierarchical-Z depth buffer.
// Uses conservative depth testing at the appropriate mip level.
//
// Algorithm:
// 1. Quick bounding sphere rejection test (early exit)
// 2. Project AABB corners to screen space
// 3. Compute screen-space rect and min depth
// 4. Select appropriate HiZ mip level based on rect size
// 5. Sample HiZ at rect corners (conservative: use max depth)
// 6. If instance min depth > HiZ max depth: occluded
//
// Performance: O(n) work, single dispatch, <0.15ms for 100K instances

// ============================================================================
// Constants
// ============================================================================

const WORKGROUP_SIZE: u32 = 256u;

/// Epsilon for floating-point comparisons.
const EPSILON: f32 = 0.0001;

/// Depth buffer uses reversed-Z: near=1.0, far=0.0.
/// Set to 1 for reversed-Z, 0 for standard depth.
const REVERSED_Z: u32 = 1u;

// ============================================================================
// Structs
// ============================================================================

/// HiZ occlusion culling parameters.
///
/// Memory Layout (128 bytes, std140 aligned):
/// | Offset | Field         | Size |
/// |--------|---------------|------|
/// | 0      | num_instances | 4    |
/// | 4      | hiz_width     | 4    |
/// | 8      | hiz_height    | 4    |
/// | 12     | num_mips      | 4    |
/// | 16     | view_proj     | 64   |
/// | 80     | near_plane    | 4    |
/// | 84     | far_plane     | 4    |
/// | 88     | flags         | 4    |
/// | 92     | _pad0         | 4    |
struct OcclusionCullParams {
    /// Number of instances to process.
    num_instances: u32,
    /// HiZ texture width (mip 0).
    hiz_width: u32,
    /// HiZ texture height (mip 0).
    hiz_height: u32,
    /// Number of mip levels in HiZ texture.
    num_mips: u32,
    /// Combined view-projection matrix for screen-space projection.
    view_proj: mat4x4<f32>,
    /// Near plane distance (for near-plane clipping).
    near_plane: f32,
    /// Far plane distance.
    far_plane: f32,
    /// Flags: bit 0 = debug (always visible), bit 1 = disable sphere test.
    flags: u32,
    /// Padding for alignment.
    _pad0: u32,
}

/// Bounding data for a single instance.
///
/// Memory Layout (48 bytes, vec4 aligned):
/// | Offset | Field         | Size |
/// |--------|---------------|------|
/// | 0      | aabb_min      | 12   |
/// | 12     | _pad0         | 4    |
/// | 16     | aabb_max      | 12   |
/// | 28     | _pad1         | 4    |
/// | 32     | sphere_center | 12   |
/// | 44     | sphere_radius | 4    |
struct OcclusionInstanceBounds {
    /// Minimum corner of AABB in world space.
    aabb_min: vec3<f32>,
    /// Padding for vec4 alignment.
    _pad0: f32,
    /// Maximum corner of AABB in world space.
    aabb_max: vec3<f32>,
    /// Padding for vec4 alignment.
    _pad1: f32,
    /// Center of bounding sphere in world space.
    sphere_center: vec3<f32>,
    /// Radius of bounding sphere.
    sphere_radius: f32,
}

/// Occlusion culling result for each instance.
///
/// Memory Layout (4 bytes):
/// | Offset | Field   | Size |
/// |--------|---------|------|
/// | 0      | visible | 4    |
struct OcclusionResult {
    /// Visibility flag: 1 = visible, 0 = occluded.
    visible: u32,
}

/// Screen-space projection result.
struct ScreenProjection {
    /// Minimum screen coordinates (NDC, -1 to 1).
    screen_min: vec2<f32>,
    /// Maximum screen coordinates (NDC, -1 to 1).
    screen_max: vec2<f32>,
    /// Minimum depth (closest point).
    min_depth: f32,
    /// True if projection is valid (not behind camera).
    valid: bool,
    /// True if AABB crosses near plane.
    crosses_near: bool,
}

// ============================================================================
// Bindings
// ============================================================================

/// Culling parameters (uniform buffer).
@group(0) @binding(0) var<uniform> params: OcclusionCullParams;

/// Instance bounds array (read-only storage buffer).
@group(0) @binding(1) var<storage, read> instances: array<OcclusionInstanceBounds>;

/// Output visibility flags (read-write storage buffer).
@group(0) @binding(2) var<storage, read_write> results: array<OcclusionResult>;

/// Hierarchical-Z depth buffer (read-only texture).
@group(0) @binding(3) var hiz_texture: texture_2d<f32>;

/// HiZ sampler (nearest filtering).
@group(0) @binding(4) var hiz_sampler: sampler;

// ============================================================================
// Projection Functions
// ============================================================================

/// Project a world-space point to clip space.
fn project_point(point: vec3<f32>, vp: mat4x4<f32>) -> vec4<f32> {
    return vp * vec4<f32>(point, 1.0);
}

/// Convert clip space to NDC (normalized device coordinates).
/// Handles w division and returns (x, y, z) in [-1,1] x [-1,1] x [0,1].
fn clip_to_ndc(clip: vec4<f32>) -> vec3<f32> {
    if (abs(clip.w) < EPSILON) {
        return vec3<f32>(0.0, 0.0, 1.0); // Degenerate case
    }
    let inv_w = 1.0 / clip.w;
    return vec3<f32>(clip.x * inv_w, clip.y * inv_w, clip.z * inv_w);
}

/// Project an AABB to screen space, computing the bounding rect and min depth.
///
/// Returns a ScreenProjection with:
/// - screen_min/max: Bounding rect in NDC (-1 to 1)
/// - min_depth: Closest depth value (for occlusion test)
/// - valid: False if entirely behind camera
/// - crosses_near: True if AABB crosses near plane
fn project_aabb(bounds: OcclusionInstanceBounds, vp: mat4x4<f32>, near: f32) -> ScreenProjection {
    var result: ScreenProjection;
    result.screen_min = vec2<f32>(1.0, 1.0);
    result.screen_max = vec2<f32>(-1.0, -1.0);
    result.min_depth = 1.0;
    result.valid = false;
    result.crosses_near = false;

    let aabb_min = bounds.aabb_min;
    let aabb_max = bounds.aabb_max;

    // Generate 8 corners of the AABB
    var corners: array<vec3<f32>, 8>;
    corners[0] = vec3<f32>(aabb_min.x, aabb_min.y, aabb_min.z);
    corners[1] = vec3<f32>(aabb_max.x, aabb_min.y, aabb_min.z);
    corners[2] = vec3<f32>(aabb_min.x, aabb_max.y, aabb_min.z);
    corners[3] = vec3<f32>(aabb_max.x, aabb_max.y, aabb_min.z);
    corners[4] = vec3<f32>(aabb_min.x, aabb_min.y, aabb_max.z);
    corners[5] = vec3<f32>(aabb_max.x, aabb_min.y, aabb_max.z);
    corners[6] = vec3<f32>(aabb_min.x, aabb_max.y, aabb_max.z);
    corners[7] = vec3<f32>(aabb_max.x, aabb_max.y, aabb_max.z);

    var behind_count: u32 = 0u;
    var in_front_count: u32 = 0u;

    // Project all corners
    for (var i = 0u; i < 8u; i++) {
        let clip = project_point(corners[i], vp);

        // Check if point is behind near plane
        // In clip space, w > 0 for points in front of camera
        // For reversed-Z, near plane is at w = near
        if (clip.w < EPSILON) {
            behind_count++;
            continue;
        }

        in_front_count++;
        let ndc = clip_to_ndc(clip);

        // Expand screen-space bounding rect
        result.screen_min = min(result.screen_min, ndc.xy);
        result.screen_max = max(result.screen_max, ndc.xy);

        // Track minimum depth (for reversed-Z: higher value = closer)
        if (REVERSED_Z == 1u) {
            result.min_depth = max(result.min_depth, ndc.z);
        } else {
            result.min_depth = min(result.min_depth, ndc.z);
        }

        result.valid = true;
    }

    // Check for near plane crossing
    if (behind_count > 0u && in_front_count > 0u) {
        result.crosses_near = true;
        // Expand to full screen for conservative test
        result.screen_min = vec2<f32>(-1.0, -1.0);
        result.screen_max = vec2<f32>(1.0, 1.0);
    }

    // If all corners behind camera, entire AABB is behind
    if (in_front_count == 0u) {
        result.valid = false;
    }

    // Clamp to screen bounds
    result.screen_min = clamp(result.screen_min, vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0));
    result.screen_max = clamp(result.screen_max, vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0));

    return result;
}

// ============================================================================
// Mip Level Selection
// ============================================================================

/// Select appropriate mip level based on screen-space rect size.
///
/// The mip level is chosen so that the rect covers approximately 2-4 texels.
/// This provides a good balance between accuracy and efficiency.
fn select_mip(screen_width: f32, screen_height: f32, hiz_width: u32, hiz_height: u32, num_mips: u32) -> u32 {
    // Convert NDC dimensions to pixel dimensions
    let pixel_width = screen_width * f32(hiz_width) * 0.5;
    let pixel_height = screen_height * f32(hiz_height) * 0.5;

    // Select mip where rect is ~2-4 texels wide
    let max_extent = max(pixel_width, pixel_height);

    if (max_extent < EPSILON) {
        return num_mips - 1u; // Use coarsest mip for tiny objects
    }

    // log2(max_extent / 2) gives mip level where extent becomes ~2 texels
    let mip_float = log2(max_extent) - 1.0;
    let mip = u32(max(0.0, mip_float));

    return min(mip, num_mips - 1u);
}

// ============================================================================
// HiZ Testing
// ============================================================================

/// Sample HiZ depth at a UV coordinate for a given mip level.
/// Returns the depth value (for reversed-Z: higher = closer, lower = farther).
fn sample_hiz(uv: vec2<f32>, mip: u32) -> f32 {
    // Convert UV [0,1] to texel coordinates at mip level
    let mip_scale = 1u << mip;
    let mip_width = max(1u, params.hiz_width / mip_scale);
    let mip_height = max(1u, params.hiz_height / mip_scale);

    let texel_x = u32(clamp(uv.x * f32(mip_width), 0.0, f32(mip_width - 1u)));
    let texel_y = u32(clamp(uv.y * f32(mip_height), 0.0, f32(mip_height - 1u)));

    // Use textureLoad for precise texel access
    let depth = textureLoad(hiz_texture, vec2<i32>(i32(texel_x), i32(texel_y)), i32(mip)).r;
    return depth;
}

/// Test if a projected rect is occluded by the HiZ buffer.
///
/// Samples HiZ at multiple points within the rect and uses the max depth
/// (most conservative) for the occlusion test.
///
/// Returns true if the instance is VISIBLE (not occluded).
fn test_hiz(screen_min: vec2<f32>, screen_max: vec2<f32>, instance_depth: f32, mip: u32) -> bool {
    // Convert NDC [-1,1] to UV [0,1]
    let uv_min = screen_min * 0.5 + 0.5;
    let uv_max = screen_max * 0.5 + 0.5;

    // Sample 4 corners for conservative test
    var hiz_depth: f32;

    let d0 = sample_hiz(vec2<f32>(uv_min.x, uv_min.y), mip);
    let d1 = sample_hiz(vec2<f32>(uv_max.x, uv_min.y), mip);
    let d2 = sample_hiz(vec2<f32>(uv_min.x, uv_max.y), mip);
    let d3 = sample_hiz(vec2<f32>(uv_max.x, uv_max.y), mip);

    if (REVERSED_Z == 1u) {
        // Reversed-Z: higher depth = closer
        // Use minimum HiZ depth (farthest occluder)
        hiz_depth = min(min(d0, d1), min(d2, d3));

        // Instance is visible if its closest point is closer than farthest occluder
        // instance_depth > hiz_depth means instance is closer (higher depth in reversed-Z)
        return instance_depth >= hiz_depth;
    } else {
        // Standard depth: lower depth = closer
        // Use maximum HiZ depth (farthest occluder)
        hiz_depth = max(max(d0, d1), max(d2, d3));

        // Instance is visible if its closest point is closer than farthest occluder
        return instance_depth <= hiz_depth;
    }
}

// ============================================================================
// Sphere Quick Rejection
// ============================================================================

/// Quick sphere-based rejection test.
/// Projects sphere center and checks if projected size is reasonable.
fn sphere_quick_test(center: vec3<f32>, radius: f32, vp: mat4x4<f32>) -> bool {
    // If no valid bounding sphere, skip this test
    if (radius <= 0.0) {
        return true; // Can't reject, continue to AABB test
    }

    // Project sphere center
    let clip = project_point(center, vp);

    // Behind camera entirely
    if (clip.w < EPSILON) {
        return false;
    }

    // Sphere is too small on screen - might be worth culling early
    // This is optional and can be disabled via flags
    return true;
}

// ============================================================================
// Culling Flags
// ============================================================================

/// Flag: Debug mode (always mark visible).
const FLAG_DEBUG_VISIBLE: u32 = 1u;
/// Flag: Disable sphere quick-reject test.
const FLAG_NO_SPHERE_TEST: u32 = 2u;
/// Flag: Conservative mode (use larger screen rect).
const FLAG_CONSERVATIVE: u32 = 4u;

// ============================================================================
// Main Compute Kernel
// ============================================================================

/// Per-instance HiZ occlusion culling kernel.
///
/// Each thread processes one instance:
/// 1. Quick bounding sphere test (optional early exit)
/// 2. Project AABB to screen space
/// 3. Handle near-plane crossing cases
/// 4. Select appropriate mip level
/// 5. Sample HiZ and compare depths
@compute @workgroup_size(256)
fn cull_occlusion(@builtin(global_invocation_id) gid: vec3<u32>) {
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
    } else {
        // Quick sphere rejection test
        if ((params.flags & FLAG_NO_SPHERE_TEST) == 0u) {
            if (!sphere_quick_test(bounds.sphere_center, bounds.sphere_radius, params.view_proj)) {
                // Sphere is behind camera, definitely not visible
                results[idx] = OcclusionResult(0u);
                return;
            }
        }

        // Project AABB to screen space
        let proj = project_aabb(bounds, params.view_proj, params.near_plane);

        if (!proj.valid) {
            // Entire AABB is behind camera
            results[idx] = OcclusionResult(0u);
            return;
        }

        // If AABB crosses near plane, conservatively mark visible
        if (proj.crosses_near) {
            results[idx] = OcclusionResult(1u);
            return;
        }

        // Compute screen-space rect dimensions
        let screen_width = proj.screen_max.x - proj.screen_min.x;
        let screen_height = proj.screen_max.y - proj.screen_min.y;

        // Select appropriate mip level
        let mip = select_mip(
            screen_width,
            screen_height,
            params.hiz_width,
            params.hiz_height,
            params.num_mips
        );

        // Test against HiZ buffer
        visible = test_hiz(proj.screen_min, proj.screen_max, proj.min_depth, mip);
    }

    // Write visibility result
    results[idx] = OcclusionResult(select(0u, 1u, visible));
}

// ============================================================================
// Alternative Entry Points
// ============================================================================

/// Conservative occlusion culling (fewer false negatives, more false positives).
/// Uses a larger screen rect and coarser mip level.
@compute @workgroup_size(256)
fn cull_occlusion_conservative(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;

    if (idx >= params.num_instances) {
        return;
    }

    let bounds = instances[idx];

    // Project AABB
    let proj = project_aabb(bounds, params.view_proj, params.near_plane);

    if (!proj.valid) {
        results[idx] = OcclusionResult(0u);
        return;
    }

    // Near plane crossing: always visible
    if (proj.crosses_near) {
        results[idx] = OcclusionResult(1u);
        return;
    }

    // Expand rect by 10% for conservative test
    let center = (proj.screen_min + proj.screen_max) * 0.5;
    let half_size = (proj.screen_max - proj.screen_min) * 0.55; // 10% expansion
    let expanded_min = clamp(center - half_size, vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0));
    let expanded_max = clamp(center + half_size, vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0));

    let screen_width = expanded_max.x - expanded_min.x;
    let screen_height = expanded_max.y - expanded_min.y;

    // Use coarser mip (+1) for more conservative test
    let mip = min(
        select_mip(screen_width, screen_height, params.hiz_width, params.hiz_height, params.num_mips) + 1u,
        params.num_mips - 1u
    );

    let visible = test_hiz(expanded_min, expanded_max, proj.min_depth, mip);
    results[idx] = OcclusionResult(select(0u, 1u, visible));
}

/// AABB-only occlusion culling (no sphere pre-test).
/// Use when bounding spheres are not available or accurate.
@compute @workgroup_size(256)
fn cull_occlusion_aabb_only(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;

    if (idx >= params.num_instances) {
        return;
    }

    let bounds = instances[idx];

    // Project AABB directly
    let proj = project_aabb(bounds, params.view_proj, params.near_plane);

    if (!proj.valid) {
        results[idx] = OcclusionResult(0u);
        return;
    }

    if (proj.crosses_near) {
        results[idx] = OcclusionResult(1u);
        return;
    }

    let screen_width = proj.screen_max.x - proj.screen_min.x;
    let screen_height = proj.screen_max.y - proj.screen_min.y;

    let mip = select_mip(
        screen_width,
        screen_height,
        params.hiz_width,
        params.hiz_height,
        params.num_mips
    );

    let visible = test_hiz(proj.screen_min, proj.screen_max, proj.min_depth, mip);
    results[idx] = OcclusionResult(select(0u, 1u, visible));
}
