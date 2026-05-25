// SPDX-License-Identifier: MIT
//
// light_culling.wgsl — Forward+ clustered light culling (T-BRG-6.3).
//
// Partitions the view frustum into 3D froxels (frustum voxels) and
// assigns lights to each froxel. One workgroup per 16×16 pixel tile.

// ── Constants ──

const TILE_SIZE: u32 = 16u;
const MAX_LIGHTS_PER_FROXEL: u32 = 64u;
const NUM_DEPTH_SLICES: u32 = 32u;
const MAX_LIGHTS_PER_TILE: u32 = 256u;

// ── Shared memory for depth reduction ──

var<workgroup> shared_min_depth: f32;
var<workgroup> shared_max_depth: f32;

// ── Data structures ──

struct PointLight {
    position: vec3<f32>,
    radius: f32,
    color: vec3<f32>,
    intensity: f32,
}

struct SpotLight {
    position: vec3<f32>,
    radius: f32,
    direction: vec3<f32>,
    cos_outer_angle: f32,
    cos_inner_angle: f32,
    _pad0: f32,
    _pad1: f32,
    color: vec3<f32>,
    intensity: f32,
}

struct Froxel {
    light_offset: u32,  // offset into light_index_list
    light_count: u32,   // number of lights in this froxel
}

struct LightCounts {
    num_point: u32,
    num_spot: u32,
    _pad0: u32,
    _pad1: u32,
}

struct CullingParams {
    screen_width: u32,
    screen_height: u32,
    num_depth_slices: u32,
    tile_size: u32,
    depth_slice_scale: f32,
    depth_slice_bias: f32,
    near_plane: f32,
    far_plane: f32,
}

// ── Uniforms and buffers ──

@group(0) @binding(0) var<uniform> params: CullingParams;
@group(0) @binding(1) var depth_texture: texture_depth_2d;
@group(0) @binding(2) var<uniform> light_counts: LightCounts;
@group(0) @binding(3) var<storage, read> point_lights: array<PointLight>;
@group(0) @binding(4) var<storage, read> spot_lights: array<SpotLight>;
@group(0) @binding(5) var<storage, read_write> froxel_grid: array<Froxel>;
@group(0) @binding(6) var<storage, read_write> light_index_list: array<u32>;
@group(0) @binding(7) var<storage, read_write> global_light_counter: array<atomic<u32>>;

// ── Utility functions ──

// Converts a depth value and screen position to a froxel Z slice.
fn depth_to_slice(depth: f32) -> u32 {
    // Exponential depth slicing for better near-field distribution.
    let linear_depth = params.near_plane / (params.far_plane - depth * (params.far_plane - params.near_plane));

    let slice_f = log2(linear_depth) * params.depth_slice_scale + params.depth_slice_bias;
    let slice = u32(clamp(slice_f, 0.0, f32(params.num_depth_slices - 1u)));
    return slice;
}

// Tests sphere-AABB intersection for point light vs froxel.
fn sphere_aabb_intersect(center: vec3<f32>, radius: f32, aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> bool {
    var closest = clamp(center, aabb_min, aabb_max);
    let dist_sq = dot(center - closest, center - closest);
    return dist_sq <= radius * radius;
}

// Tests cone-AABB intersection for spot light vs froxel.
fn cone_aabb_intersect(
    cone_tip: vec3<f32>,
    cone_dir: vec3<f32>,
    cos_angle: f32,
    range: f32,
    aabb_min: vec3<f32>,
    aabb_max: vec3<f32>,
) -> bool {
    // First check: bounding sphere test (conservative).
    let center = (aabb_min + aabb_max) * 0.5;
    let extent = length(aabb_max - aabb_min) * 0.5;
    let to_center = center - cone_tip;
    let dist_to_center = length(to_center);

    if dist_to_center > range + extent {
        return false;
    }

    // Check if any AABB corner is inside the cone frustum.
    let corners: array<vec3<f32>, 8> = array<vec3<f32>, 8>(
        vec3<f32>(aabb_min.x, aabb_min.y, aabb_min.z),
        vec3<f32>(aabb_max.x, aabb_min.y, aabb_min.z),
        vec3<f32>(aabb_min.x, aabb_max.y, aabb_min.z),
        vec3<f32>(aabb_max.x, aabb_max.y, aabb_min.z),
        vec3<f32>(aabb_min.x, aabb_min.y, aabb_max.z),
        vec3<f32>(aabb_max.x, aabb_min.y, aabb_max.z),
        vec3<f32>(aabb_min.x, aabb_max.y, aabb_max.z),
        vec3<f32>(aabb_max.x, aabb_max.y, aabb_max.z),
    );

    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let to_corner = corners[i] - cone_tip;
        let dist = length(to_corner);
        if dist <= range {
            let cos_to_corner = dot(normalize(to_corner), cone_dir);
            if cos_to_corner >= cos_angle {
                return true;
            }
        }
    }

    // Check if cone direction points toward AABB (conservative).
    let to_closest = clamp(center - cone_tip, -extent, extent) + (center - cone_tip);
    return dot(normalize(to_closest), cone_dir) >= cos_angle;
}

// ── Main compute entry point ──

@compute @workgroup_size(TILE_SIZE, TILE_SIZE, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
) {
    let tile_x = wg_id.x;
    let tile_y = wg_id.y;

    let px = tile_x * TILE_SIZE + local_id.x;
    let py = tile_y * TILE_SIZE + local_id.y;

    // Step 1: Compute tile depth bounds.
    var min_depth: f32 = 1.0;
    var max_depth: f32 = 0.0;

    if px < params.screen_width && py < params.screen_height {
        let depth = textureLoad(depth_texture, vec2<i32>(i32(px), i32(py)), 0);
        min_depth = depth;
        max_depth = depth;
    }

    // Initialize shared memory from thread 0.
    if local_idx == 0u {
        shared_min_depth = 1.0;
        shared_max_depth = 0.0;
    }
    workgroupBarrier();

    // Parallel reduction: min depth.
    atomicMin(&bitcast<atomic<i32>>((&shared_min_depth)), bitcast<i32>(min_depth));
    // Parallel reduction: max depth.
    atomicMax(&bitcast<atomic<i32>>((&shared_max_depth)), bitcast<i32>(max_depth));
    workgroupBarrier();

    let tile_min_depth = shared_min_depth;
    let tile_max_depth = shared_max_depth;

    // Step 2: Compute froxel Z range for this tile.
    let min_slice = depth_to_slice(tile_min_depth);
    let max_slice = depth_to_slice(tile_max_depth);

    // Step 3: Assign lights to froxels (single thread per workgroup).
    if local_idx == 0u {
        let num_tiles_x = (params.screen_width + TILE_SIZE - 1u) / TILE_SIZE;
        let tile_index = tile_y * num_tiles_x + tile_x;

        for (var slice: u32 = min_slice; slice <= max_slice; slice = slice + 1u) {
            let froxel_idx = tile_index * params.num_depth_slices + slice;

            // Compute froxel AABB (simplified: use near/far plane estimate).
            let near_depth = params.near_plane * exp2(f32(slice) / params.depth_slice_scale);
            let far_depth = params.near_plane * exp2(f32(slice + 1u) / params.depth_slice_scale);

            // Simplified AABB for light testing (view-space).
            let aabb_min = vec3<f32>(-far_depth, -far_depth, near_depth);
            let aabb_max = vec3<f32>(far_depth, far_depth, far_depth);

            var light_offset = froxel_grid[froxel_idx].light_offset;
            var light_count = 0u;

            // Cull point lights.
            for (var i: u32 = 0u; i < light_counts.num_point && light_count < MAX_LIGHTS_PER_FROXEL; i = i + 1u) {
                let light = point_lights[i];
                if sphere_aabb_intersect(light.position, light.radius, aabb_min, aabb_max) {
                    light_index_list[light_offset + light_count] = i;
                    light_count = light_count + 1u;
                }
            }

            // Cull spot lights.
            for (var j: u32 = 0u; j < light_counts.num_spot && light_count < MAX_LIGHTS_PER_FROXEL; j = j + 1u) {
                let light = spot_lights[j];
                if cone_aabb_intersect(
                    light.position, normalize(-light.direction),
                    light.cos_outer_angle, light.radius,
                    aabb_min, aabb_max
                ) {
                    light_index_list[light_offset + light_count] = j;
                    light_count = light_count + 1u;
                }
            }

            froxel_grid[froxel_idx].light_count = light_count;
        }
    }
}
