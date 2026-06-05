// SPDX-License-Identifier: MIT
//
// hiz_occlusion.wgsl - HiZ Occlusion Test for GPU-Driven Occlusion Culling (T-WGPU-P6.4.3).
//
// Implements hierarchical-Z occlusion testing for GPU-driven rendering:
// 1. Project world-space AABB to screen-space rectangle
// 2. Select appropriate HiZ mip level based on rectangle size
// 3. Sample HiZ depth at rectangle corners (conservative max)
// 4. Compare AABB near depth against HiZ depth (reverse-Z)
//
// Depth Convention (Reverse-Z):
//   TRINITY uses reversed-Z (near=1.0, far=0.0):
//   - Larger Z values are closer to the camera
//   - For occlusion: if object_near_depth < hiz_depth, object is occluded
//   - HiZ stores MAX depth per region (closest visible surface)
//
// Algorithm:
//   1. Transform all 8 AABB corners to clip space
//   2. Clip against near plane (w > 0 check)
//   3. Project to NDC, then to screen coordinates
//   4. Compute screen-space bounding rect and nearest depth
//   5. Select mip level: log2(max(rect_width, rect_height))
//   6. Sample HiZ at 4 corners of the rect
//   7. Take MAX of samples (conservative depth in reverse-Z)
//   8. Compare: if aabb_near_depth < hiz_max_depth, occluded
//
// Conservative Occlusion:
//   The test is conservative: false positives (marking occluded as visible)
//   never occur. False negatives (marking visible as occluded) are minimized
//   by sampling at rect corners and taking max depth.
//
// Workgroup Size: 256 threads for batch processing.

// ============================================================================
// Constants
// ============================================================================

/// Workgroup size for batch occlusion testing.
const WORKGROUP_SIZE: u32 = 256u;

/// Number of AABB corners.
const NUM_CORNERS: u32 = 8u;

/// Small epsilon for floating point comparisons.
const EPSILON: f32 = 1e-6;

/// Maximum mip level supported.
const MAX_MIP_LEVEL: u32 = 14u;

/// Screen-space expansion factor for conservative bounds.
/// Slightly expand the screen rect to handle sub-pixel rasterization.
const CONSERVATIVE_EXPAND: f32 = 1.0;

// ============================================================================
// Structs
// ============================================================================

/// Parameters for HiZ occlusion testing.
///
/// Memory Layout (48 bytes, 16-byte aligned):
/// | Offset | Field           | Size | Description                    |
/// |--------|-----------------|------|--------------------------------|
/// | 0      | view_projection | 64   | Combined VP matrix (column-major) |
/// | 64     | hiz_size        | 8    | HiZ base resolution (width, height) |
/// | 72     | near_plane      | 4    | Near plane distance            |
/// | 76     | max_mip         | 4    | Maximum mip level (num_mips - 1) |
struct HiZOcclusionParams {
    /// Combined view-projection matrix (column-major).
    view_projection: mat4x4<f32>,
    /// HiZ pyramid base resolution (mip 0).
    hiz_size: vec2<f32>,
    /// Near plane distance for clipping.
    near_plane: f32,
    /// Maximum mip level (num_mips - 1).
    max_mip: u32,
}

/// Projected AABB result.
///
/// Contains screen-space bounds and nearest depth after projection.
struct ProjectedAABB {
    /// Minimum screen coordinates (x, y).
    min_xy: vec2<f32>,
    /// Maximum screen coordinates (x, y).
    max_xy: vec2<f32>,
    /// Nearest depth in NDC (closest corner to camera).
    /// In reverse-Z, higher values are nearer.
    near_depth: f32,
    /// True if AABB is at least partially in front of camera.
    valid: bool,
}

/// Occlusion test result.
struct OcclusionResult {
    /// True if visible (not occluded).
    visible: bool,
    /// Mip level used for testing.
    mip_level: u32,
    /// Screen-space rect size used.
    rect_size: vec2<f32>,
}

// ============================================================================
// Bindings: Group 0 - HiZ Pyramid
// ============================================================================

/// HiZ pyramid texture (all mip levels).
@group(0) @binding(0) var hiz_texture: texture_2d<f32>;

/// Linear sampler for HiZ sampling.
@group(0) @binding(1) var hiz_sampler: sampler;

// ============================================================================
// Bindings: Group 1 - Parameters
// ============================================================================

/// Occlusion test parameters.
@group(1) @binding(0) var<uniform> params: HiZOcclusionParams;

// ============================================================================
// Helper Functions: Coordinate Transforms
// ============================================================================

/// Transform a world-space point to clip space.
fn world_to_clip(world_pos: vec3<f32>) -> vec4<f32> {
    return params.view_projection * vec4<f32>(world_pos, 1.0);
}

/// Check if a clip-space point is in front of the near plane.
/// In clip space, w > 0 means the point is in front of the camera.
fn is_in_front_of_near(clip: vec4<f32>) -> bool {
    return clip.w > EPSILON;
}

/// Project clip space to NDC (perspective divide).
/// Returns (x, y, z) in [-1, 1] for visible points.
fn clip_to_ndc(clip: vec4<f32>) -> vec3<f32> {
    let inv_w = 1.0 / clip.w;
    return vec3<f32>(clip.x * inv_w, clip.y * inv_w, clip.z * inv_w);
}

/// Convert NDC to screen coordinates.
/// NDC: (-1,-1) bottom-left, (1,1) top-right
/// Screen: (0,0) top-left, (width, height) bottom-right
fn ndc_to_screen(ndc: vec2<f32>, screen_size: vec2<f32>) -> vec2<f32> {
    // NDC to [0, 1] range
    let uv = (ndc + vec2<f32>(1.0, 1.0)) * 0.5;
    // Flip Y for screen coordinates (top-left origin)
    let screen_uv = vec2<f32>(uv.x, 1.0 - uv.y);
    // Scale to screen size
    return screen_uv * screen_size;
}

/// Convert NDC to UV coordinates (for texture sampling).
/// NDC: (-1,-1) bottom-left, (1,1) top-right
/// UV: (0,0) top-left, (1,1) bottom-right
fn ndc_to_uv(ndc: vec2<f32>) -> vec2<f32> {
    let uv = (ndc + vec2<f32>(1.0, 1.0)) * 0.5;
    // Flip Y for texture coordinates
    return vec2<f32>(uv.x, 1.0 - uv.y);
}

// ============================================================================
// Helper Functions: AABB Corners
// ============================================================================

/// Get the i-th corner of an AABB (0-7).
/// Uses bit pattern: x = bit 0, y = bit 1, z = bit 2.
fn get_aabb_corner(aabb_min: vec3<f32>, aabb_max: vec3<f32>, index: u32) -> vec3<f32> {
    return vec3<f32>(
        select(aabb_min.x, aabb_max.x, (index & 1u) != 0u),
        select(aabb_min.y, aabb_max.y, (index & 2u) != 0u),
        select(aabb_min.z, aabb_max.z, (index & 4u) != 0u),
    );
}

// ============================================================================
// Core Functions: AABB Projection
// ============================================================================

/// Project world-space AABB to screen-space rectangle.
///
/// Transforms all 8 corners to clip space, clips against near plane,
/// and computes the screen-space bounding rectangle and nearest depth.
///
/// # Arguments
/// * `aabb_min` - Minimum corner of AABB in world space.
/// * `aabb_max` - Maximum corner of AABB in world space.
///
/// # Returns
/// ProjectedAABB containing screen bounds, near depth, and validity.
///
/// # Notes
/// - If any corner is behind the camera, the AABB is marked as visible
///   (conservative: don't cull potentially visible objects).
/// - Screen coordinates are in pixels, origin at top-left.
/// - near_depth is in NDC space [0, 1] with reverse-Z (1.0 = near).
fn project_aabb_to_screen(aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> ProjectedAABB {
    var result: ProjectedAABB;

    // Initialize bounds
    result.min_xy = vec2<f32>(1e30, 1e30);
    result.max_xy = vec2<f32>(-1e30, -1e30);
    result.near_depth = 0.0;  // In reverse-Z, 0 is far, 1 is near
    result.valid = true;

    // Track if all corners are behind camera
    var all_behind = true;
    var any_behind = false;

    // Process all 8 corners
    for (var i = 0u; i < NUM_CORNERS; i = i + 1u) {
        let corner = get_aabb_corner(aabb_min, aabb_max, i);
        let clip = world_to_clip(corner);

        if (is_in_front_of_near(clip)) {
            all_behind = false;
            let ndc = clip_to_ndc(clip);
            let screen = ndc_to_screen(ndc.xy, params.hiz_size);

            // Expand screen bounds
            result.min_xy = min(result.min_xy, screen);
            result.max_xy = max(result.max_xy, screen);

            // Track nearest depth (max in reverse-Z)
            // NDC z is in [0, 1] with reverse-Z (near = 1)
            let depth = clamp(ndc.z, 0.0, 1.0);
            result.near_depth = max(result.near_depth, depth);
        } else {
            any_behind = true;
        }
    }

    // Handle edge cases
    if (all_behind) {
        // All corners behind camera: fully occluded (or degenerate)
        result.valid = false;
        return result;
    }

    if (any_behind) {
        // Some corners behind camera: conservatively mark as visible
        // by extending bounds to screen edges
        result.min_xy = vec2<f32>(0.0, 0.0);
        result.max_xy = params.hiz_size;
        result.near_depth = 1.0;  // Near plane in reverse-Z
    }

    // Clamp to screen bounds
    result.min_xy = clamp(result.min_xy, vec2<f32>(0.0, 0.0), params.hiz_size);
    result.max_xy = clamp(result.max_xy, vec2<f32>(0.0, 0.0), params.hiz_size);

    // Apply conservative expansion (sub-pixel coverage)
    result.min_xy = max(result.min_xy - vec2<f32>(CONSERVATIVE_EXPAND), vec2<f32>(0.0, 0.0));
    result.max_xy = min(result.max_xy + vec2<f32>(CONSERVATIVE_EXPAND), params.hiz_size);

    return result;
}

// ============================================================================
// Core Functions: Mip Level Selection
// ============================================================================

/// Select appropriate HiZ mip level based on screen-space rect size.
///
/// The mip level is chosen so that the rect covers at most a small number
/// of texels (ideally 2x2 or less) at that mip level, enabling efficient
/// depth testing with minimal texture reads.
///
/// # Arguments
/// * `rect_size` - Size of screen-space rect (width, height) in pixels.
///
/// # Returns
/// Mip level to use for HiZ sampling (0 = base resolution).
///
/// # Algorithm
/// mip = floor(log2(max(width, height)))
/// This ensures the rect maps to roughly 1-2 texels at the selected mip.
fn select_mip_level(rect_size: vec2<f32>) -> u32 {
    let max_dim = max(rect_size.x, rect_size.y);

    // Handle degenerate cases
    if (max_dim <= 1.0) {
        return 0u;
    }

    // log2(max_dim), floor
    let mip = u32(log2(max_dim));

    // Clamp to valid range
    return min(mip, params.max_mip);
}

/// Calculate mip dimensions for a given mip level.
fn mip_size(level: u32) -> vec2<f32> {
    let divisor = f32(1u << level);
    return max(params.hiz_size / divisor, vec2<f32>(1.0, 1.0));
}

// ============================================================================
// Core Functions: HiZ Depth Sampling
// ============================================================================

/// Sample HiZ depth at a single point.
/// Uses textureLoad for exact texel access.
fn sample_hiz_point(uv: vec2<f32>, level: u32) -> f32 {
    let mip_dims = mip_size(level);
    let texel = vec2<i32>(uv * mip_dims);
    let clamped = clamp(texel, vec2<i32>(0, 0), vec2<i32>(mip_dims) - vec2<i32>(1, 1));
    return textureLoad(hiz_texture, clamped, i32(level)).r;
}

/// Sample HiZ depth at rect corners and return conservative (max) depth.
///
/// For reverse-Z, we want the MAX depth from the HiZ region, which
/// represents the closest visible surface. If the object's near depth
/// is less than this max, the object is fully behind existing geometry.
///
/// # Arguments
/// * `min_uv` - Minimum UV of the screen rect (top-left).
/// * `max_uv` - Maximum UV of the screen rect (bottom-right).
/// * `level` - Mip level to sample.
///
/// # Returns
/// Maximum (closest) depth from the sampled region.
fn sample_hiz_rect_max(min_uv: vec2<f32>, max_uv: vec2<f32>, level: u32) -> f32 {
    // Sample at 4 corners
    let d00 = sample_hiz_point(min_uv, level);
    let d10 = sample_hiz_point(vec2<f32>(max_uv.x, min_uv.y), level);
    let d01 = sample_hiz_point(vec2<f32>(min_uv.x, max_uv.y), level);
    let d11 = sample_hiz_point(max_uv, level);

    // Return max (closest in reverse-Z)
    return max(max(d00, d10), max(d01, d11));
}

/// Sample HiZ with bilinear filtering for smoother results.
/// Uses textureSampleLevel for hardware bilinear.
fn sample_hiz_bilinear(uv: vec2<f32>, level: f32) -> f32 {
    return textureSampleLevel(hiz_texture, hiz_sampler, uv, level).r;
}

/// Sample HiZ rect with bilinear filtering at corners.
fn sample_hiz_rect_bilinear_max(min_uv: vec2<f32>, max_uv: vec2<f32>, level: f32) -> f32 {
    let d00 = sample_hiz_bilinear(min_uv, level);
    let d10 = sample_hiz_bilinear(vec2<f32>(max_uv.x, min_uv.y), level);
    let d01 = sample_hiz_bilinear(vec2<f32>(min_uv.x, max_uv.y), level);
    let d11 = sample_hiz_bilinear(max_uv, level);

    return max(max(d00, d10), max(d01, d11));
}

// ============================================================================
// Core Functions: Occlusion Test
// ============================================================================

/// Test if an AABB is occluded using the HiZ pyramid.
///
/// # Algorithm
/// 1. Project AABB to screen-space rect and get near depth
/// 2. Select appropriate mip level based on rect size
/// 3. Sample HiZ at rect corners, take max (closest visible depth)
/// 4. Compare: if aabb_near_depth < hiz_max_depth, object is occluded
///
/// # Arguments
/// * `aabb_min` - Minimum corner of AABB in world space.
/// * `aabb_max` - Maximum corner of AABB in world space.
///
/// # Returns
/// true if the AABB is visible (not occluded), false if occluded.
///
/// # Depth Convention (Reverse-Z)
/// - HiZ stores MAX depth per region (closest visible surface)
/// - AABB near_depth is the closest AABB point to camera
/// - If near_depth < hiz_depth: AABB is entirely behind visible geometry
/// - If near_depth >= hiz_depth: AABB may be visible (conservative)
fn test_hiz_occlusion(aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> bool {
    // Step 1: Project AABB to screen
    let projected = project_aabb_to_screen(aabb_min, aabb_max);

    // Handle invalid projections (behind camera, etc.)
    if (!projected.valid) {
        // If all behind camera, it's not visible
        return false;
    }

    // Step 2: Calculate rect size and select mip level
    let rect_size = projected.max_xy - projected.min_xy;

    // Degenerate rect: mark as visible (conservative)
    if (rect_size.x < 1.0 || rect_size.y < 1.0) {
        return true;
    }

    let mip_level = select_mip_level(rect_size);

    // Step 3: Convert to UVs for texture sampling
    let min_uv = projected.min_xy / params.hiz_size;
    let max_uv = projected.max_xy / params.hiz_size;

    // Step 4: Sample HiZ depth (max of corners for conservative test)
    let hiz_depth = sample_hiz_rect_max(min_uv, max_uv, mip_level);

    // Step 5: Occlusion test (reverse-Z)
    // In reverse-Z: near=1.0, far=0.0
    // If object's near depth < HiZ depth, object is behind all visible geometry
    // Add small epsilon to avoid precision issues
    return projected.near_depth >= (hiz_depth - EPSILON);
}

/// Extended occlusion test returning detailed results.
///
/// Same as test_hiz_occlusion but returns additional debug info.
fn test_hiz_occlusion_detailed(aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> OcclusionResult {
    var result: OcclusionResult;
    result.visible = true;
    result.mip_level = 0u;
    result.rect_size = vec2<f32>(0.0, 0.0);

    let projected = project_aabb_to_screen(aabb_min, aabb_max);

    if (!projected.valid) {
        result.visible = false;
        return result;
    }

    let rect_size = projected.max_xy - projected.min_xy;
    result.rect_size = rect_size;

    if (rect_size.x < 1.0 || rect_size.y < 1.0) {
        return result;  // Visible (conservative)
    }

    let mip_level = select_mip_level(rect_size);
    result.mip_level = mip_level;

    let min_uv = projected.min_xy / params.hiz_size;
    let max_uv = projected.max_xy / params.hiz_size;

    let hiz_depth = sample_hiz_rect_max(min_uv, max_uv, mip_level);

    result.visible = projected.near_depth >= (hiz_depth - EPSILON);
    return result;
}

// ============================================================================
// Batch Processing Bindings (Group 2)
// ============================================================================

/// Batch processing parameters.
struct BatchParams {
    /// Number of AABBs to process.
    num_objects: u32,
    /// Flags (reserved).
    flags: u32,
    /// Padding for 16-byte alignment.
    _pad0: u32,
    _pad1: u32,
}

/// Input AABB for batch processing.
struct InputAABB {
    /// Minimum corner of AABB.
    min: vec3<f32>,
    /// Padding.
    _pad0: f32,
    /// Maximum corner of AABB.
    max: vec3<f32>,
    /// Padding.
    _pad1: f32,
}

@group(2) @binding(0) var<uniform> batch_params: BatchParams;
@group(2) @binding(1) var<storage, read> input_aabbs: array<InputAABB>;
@group(2) @binding(2) var<storage, read_write> visibility_results: array<u32>;

// ============================================================================
// Compute Shader Entry Points
// ============================================================================

/// Batch HiZ occlusion culling compute shader.
///
/// One thread per AABB. Outputs 1 for visible, 0 for occluded.
@compute @workgroup_size(256)
fn hiz_occlusion_cull(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;

    // Bounds check
    if (idx >= batch_params.num_objects) {
        return;
    }

    let aabb = input_aabbs[idx];
    let visible = test_hiz_occlusion(aabb.min, aabb.max);

    visibility_results[idx] = select(0u, 1u, visible);
}

/// Batch HiZ occlusion culling with detailed output.
///
/// Outputs packed visibility info: [mip_level:8 | visible:1 | reserved:23]
@compute @workgroup_size(256)
fn hiz_occlusion_cull_detailed(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;

    if (idx >= batch_params.num_objects) {
        return;
    }

    let aabb = input_aabbs[idx];
    let result = test_hiz_occlusion_detailed(aabb.min, aabb.max);

    // Pack result: [mip_level:8 | visible:1 | reserved:23]
    let packed = (result.mip_level << 24u) | select(0u, 1u, result.visible);
    visibility_results[idx] = packed;
}

// ============================================================================
// Notes on Usage
// ============================================================================
//
// Dispatch calculation:
//   workgroups_x = (num_objects + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE
//   workgroups_y = 1
//   workgroups_z = 1
//
// Bind group layout:
//   Group 0: HiZ pyramid texture + sampler
//   Group 1: HiZOcclusionParams uniform buffer
//   Group 2: BatchParams + input AABBs + visibility results
//
// Integration with HiZ pyramid:
//   The HiZ pyramid must be built before occlusion testing.
//   Use hiz_downsample.wgsl to generate mip levels from depth buffer.
//
// Reverse-Z reminder:
//   - Near plane depth = 1.0
//   - Far plane depth = 0.0
//   - Higher values = closer to camera
//   - HiZ stores MAX = closest visible surface per region
//
// Performance tips:
//   - Process all AABBs in one dispatch
//   - Combine with frustum culling (test frustum first, then HiZ)
//   - Use temporal coherence: objects visible last frame likely visible now
