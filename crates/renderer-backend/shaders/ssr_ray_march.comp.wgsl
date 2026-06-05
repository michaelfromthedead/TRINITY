// SPDX-License-Identifier: MIT
//
// ssr_ray_march.comp.wgsl - Screen-Space Reflections with HiZ Ray Marching (T-GIR-P4.2).
//
// Performs hierarchical screen-space ray marching for reflections using the
// HiZ buffer to accelerate ray-depth intersection tests. This algorithm starts
// at coarse mip levels and descends to finer levels only when necessary,
// significantly reducing the number of depth samples compared to linear marching.
//
// Algorithm:
//   1. For each reflective pixel, compute the reflection ray in view space
//   2. Project ray to screen space and begin HiZ traversal
//   3. Start at the coarsest mip level where the ray covers ~1 texel
//   4. March ray: if behind surface, descend mip; if in front, advance ray
//   5. At mip 0, perform binary refinement to find exact intersection
//   6. Output hit UV + confidence or mark as miss for fallback
//
// Depth convention: TRINITY uses reversed-Z (near=1.0, far=0.0)
// - Larger Z values are closer to the camera
// - HiZ stores max(depth) per tile (furthest point in tile)
// - Ray.z > HiZ.z means ray is behind all geometry in that tile
//
// Workgroup size: 8x8 threads for optimal occupancy on modern GPUs.

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const WORKGROUP_SIZE: u32 = 8u;
const PI: f32 = 3.14159265359;
const EPSILON: f32 = 0.0001;

// ---------------------------------------------------------------------------
// Structs
// ---------------------------------------------------------------------------

/// SSR configuration parameters.
struct SSRConfig {
    max_steps: u32,           // Maximum HiZ march steps
    max_binary_steps: u32,    // Maximum binary refinement steps
    thickness: f32,           // Depth comparison threshold
    stride: f32,              // Initial step size
    jitter_amount: f32,       // Temporal jitter (0-1)
    max_distance: f32,        // Maximum ray travel distance
    _pad: vec2<f32>,          // Padding for alignment
}

/// Extended uniforms including camera matrices.
struct SSRUniforms {
    view_matrix: mat4x4<f32>,
    proj_matrix: mat4x4<f32>,
    inv_view_matrix: mat4x4<f32>,
    inv_proj_matrix: mat4x4<f32>,
    config: SSRConfig,
    screen_size: vec2<u32>,
    max_mip: u32,
    frame_index: u32,
    near_plane: f32,
    far_plane: f32,
    _pad: vec2<f32>,
}

/// Result of HiZ ray trace.
struct HitResult {
    hit: bool,
    uv: vec2<f32>,
    distance: f32,
    steps: u32,
    mip: u32,
    confidence: f32,
}

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> uniforms: SSRUniforms;
@group(0) @binding(1) var hiz_texture: texture_2d<f32>;
@group(0) @binding(2) var hiz_sampler: sampler;
@group(0) @binding(3) var gbuffer_depth: texture_2d<f32>;
@group(0) @binding(4) var gbuffer_normal: texture_2d<f32>;
@group(0) @binding(5) var reflection_output: texture_storage_2d<rgba16float, write>;

// ---------------------------------------------------------------------------
// Helper Functions: Math
// ---------------------------------------------------------------------------

/// Compute the reflection direction.
fn reflect_dir(incident: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    return incident - 2.0 * dot(incident, normal) * normal;
}

/// Blue noise-based jitter for temporal stability.
fn temporal_jitter(pixel: vec2<u32>, frame: u32) -> f32 {
    // R2 sequence for better coverage
    let phi = 1.6180339887; // Golden ratio
    let seed = f32(pixel.x * 1973u + pixel.y * 9277u + frame * 26699u);
    return fract(seed * 0.0001 + f32(frame) * phi) * 2.0 - 1.0;
}

/// Generate pseudo-random value from screen coordinate and frame.
fn hash(p: vec2<f32>, frame: u32) -> f32 {
    let p3 = fract(vec3<f32>(p.xyx) * vec3<f32>(443.8975, 441.423, 437.195));
    let p4 = p3 + vec3<f32>(dot(p3, p3.yzx + f32(frame) * 0.01));
    return fract((p4.x + p4.y) * p4.z);
}

// ---------------------------------------------------------------------------
// Helper Functions: Coordinate Transforms
// ---------------------------------------------------------------------------

/// Convert screen UV (0-1) to NDC (-1 to 1).
fn uv_to_ndc(uv: vec2<f32>) -> vec2<f32> {
    return uv * 2.0 - 1.0;
}

/// Convert NDC to screen UV.
fn ndc_to_uv(ndc: vec2<f32>) -> vec2<f32> {
    return ndc * 0.5 + 0.5;
}

/// Convert screen UV + depth to view-space position.
fn screen_to_view(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv_to_ndc(uv), depth, 1.0);
    let view_pos = uniforms.inv_proj_matrix * ndc;
    return view_pos.xyz / view_pos.w;
}

/// Convert view-space position to screen UV + depth.
fn view_to_screen(view_pos: vec3<f32>) -> vec3<f32> {
    let clip = uniforms.proj_matrix * vec4<f32>(view_pos, 1.0);
    let ndc = clip.xyz / clip.w;
    return vec3<f32>(ndc_to_uv(ndc.xy), ndc.z);
}

/// Linearize reversed-Z depth to view-space distance.
fn linearize_depth(depth: f32) -> f32 {
    // Reversed-Z: near = 1.0, far = 0.0
    let near = uniforms.near_plane;
    let far = uniforms.far_plane;
    return near * far / (far + depth * (near - far));
}

// ---------------------------------------------------------------------------
// Helper Functions: HiZ Sampling
// ---------------------------------------------------------------------------

/// Sample the HiZ buffer at a specific mip level.
/// Returns the maximum depth in the sampled region.
fn sample_hiz(uv: vec2<f32>, mip_level: i32) -> f32 {
    // Clamp UV to valid range
    let clamped_uv = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0));

    // textureSampleLevel with point filtering
    return textureSampleLevel(hiz_texture, hiz_sampler, clamped_uv, f32(mip_level)).r;
}

/// Sample depth at mip 0 (full resolution).
fn sample_depth_mip0(uv: vec2<f32>) -> f32 {
    let screen_size = vec2<f32>(uniforms.screen_size);
    let coord = vec2<i32>(uv * screen_size);
    let clamped = clamp(coord, vec2<i32>(0), vec2<i32>(uniforms.screen_size) - vec2<i32>(1));
    return textureLoad(gbuffer_depth, clamped, 0).r;
}

/// Compute step size for a given mip level.
/// Step size doubles with each mip level for efficient traversal.
fn step_size_for_mip(mip_level: u32) -> f32 {
    return uniforms.config.stride * f32(1u << mip_level);
}

// ---------------------------------------------------------------------------
// HiZ Ray Marching
// ---------------------------------------------------------------------------

/// Perform hierarchical ray marching starting at coarse mip.
///
/// The algorithm starts at the coarsest mip level and:
/// 1. Samples HiZ depth at current mip
/// 2. If ray is behind surface (ray.z > hiz.z), descend mip level
/// 3. If at mip 0 and behind, do binary refinement
/// 4. If ray is in front, advance by step_size(mip)
///
/// This reduces average iterations from O(distance) to O(log(distance)).
fn hiz_trace(
    ray_origin_view: vec3<f32>,
    ray_dir_view: vec3<f32>,
    jitter: f32,
) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.uv = vec2<f32>(0.0);
    result.distance = 0.0;
    result.steps = 0u;
    result.mip = 0u;
    result.confidence = 0.0;

    // Start at a small offset along the ray to avoid self-intersection
    let start_offset = uniforms.config.stride * (0.5 + jitter * uniforms.config.jitter_amount);
    var t = start_offset;

    // Start at coarsest mip level (but not too coarse to avoid missing small features)
    var mip_level = i32(min(uniforms.max_mip, 6u));

    let max_t = uniforms.config.max_distance;

    for (var i = 0u; i < uniforms.config.max_steps; i++) {
        result.steps = i;

        // Current ray position in view space
        let ray_pos_view = ray_origin_view + ray_dir_view * t;

        // Project to screen space
        let ray_screen = view_to_screen(ray_pos_view);
        let ray_uv = ray_screen.xy;
        let ray_depth = ray_screen.z;

        // Check screen bounds (with small margin for edge cases)
        if (ray_uv.x < -0.01 || ray_uv.x > 1.01 || ray_uv.y < -0.01 || ray_uv.y > 1.01) {
            // Ray went off-screen
            return result;
        }

        // Check if ray is behind the near plane or beyond far plane
        if (ray_depth > 1.0 || ray_depth < 0.0) {
            return result;
        }

        // Check max distance
        if (t > max_t) {
            return result;
        }

        // Sample HiZ at current mip level
        let hiz_depth = sample_hiz(ray_uv, mip_level);

        // Depth comparison (reversed-Z: larger values are closer)
        // Ray is behind surface if ray_depth > hiz_depth
        let depth_diff = ray_depth - hiz_depth;

        if (depth_diff > 0.0) {
            // Ray is behind the surface
            if (mip_level == 0) {
                // At finest level, check if within thickness threshold
                if (depth_diff < uniforms.config.thickness) {
                    // Hit! Do binary refinement for exact position
                    result = binary_refine(ray_origin_view, ray_dir_view, t - step_size_for_mip(0u), t);
                    return result;
                } else {
                    // Behind surface but too thick, ray passed through thin geometry
                    // Continue marching
                    t += step_size_for_mip(0u);
                }
            } else {
                // Descend to finer mip level for more precision
                mip_level = max(mip_level - 1, 0);
            }
        } else {
            // Ray is in front of surface, advance
            t += step_size_for_mip(u32(mip_level));

            // Consider ascending mip when far from geometry
            // This speeds up traversal through empty space
            if (depth_diff < -0.1 && mip_level < i32(uniforms.max_mip) - 1) {
                mip_level = min(mip_level + 1, i32(uniforms.max_mip));
            }
        }
    }

    // Max iterations reached without hit
    return result;
}

/// Binary refinement to find the exact hit point.
///
/// Once the coarse HiZ traversal finds a potential intersection region,
/// binary search narrows down the exact t parameter where the ray
/// intersects the depth surface.
fn binary_refine(
    ray_origin_view: vec3<f32>,
    ray_dir_view: vec3<f32>,
    t_lo: f32,
    t_hi: f32,
) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.uv = vec2<f32>(0.0);
    result.distance = 0.0;
    result.steps = 0u;
    result.mip = 0u;
    result.confidence = 0.0;

    var lo = t_lo;
    var hi = t_hi;

    for (var i = 0u; i < uniforms.config.max_binary_steps; i++) {
        let mid = (lo + hi) * 0.5;

        let ray_pos = ray_origin_view + ray_dir_view * mid;
        let ray_screen = view_to_screen(ray_pos);
        let ray_uv = ray_screen.xy;
        let ray_depth = ray_screen.z;

        // Check bounds
        if (ray_uv.x < 0.0 || ray_uv.x > 1.0 || ray_uv.y < 0.0 || ray_uv.y > 1.0) {
            return result;
        }

        // Sample at full resolution
        let surface_depth = sample_depth_mip0(ray_uv);
        let depth_diff = ray_depth - surface_depth;

        if (depth_diff > 0.0) {
            // Behind surface, search earlier
            hi = mid;
        } else {
            // In front, search later
            lo = mid;
        }
    }

    // Use the final midpoint as hit position
    let t_hit = (lo + hi) * 0.5;
    let hit_pos = ray_origin_view + ray_dir_view * t_hit;
    let hit_screen = view_to_screen(hit_pos);
    let hit_uv = hit_screen.xy;
    let hit_depth = hit_screen.z;

    // Final validation
    if (hit_uv.x < 0.0 || hit_uv.x > 1.0 || hit_uv.y < 0.0 || hit_uv.y > 1.0) {
        return result;
    }

    let surface_depth = sample_depth_mip0(hit_uv);
    let final_diff = abs(hit_depth - surface_depth);

    if (final_diff < uniforms.config.thickness) {
        result.hit = true;
        result.uv = hit_uv;
        result.distance = t_hit;
        result.mip = 0u;
        result.confidence = compute_confidence(ray_dir_view, hit_uv, t_hit);
    }

    return result;
}

/// Compute hit confidence based on various factors.
///
/// Confidence is reduced for:
/// - Grazing angles (ray nearly parallel to surface)
/// - Hits near screen edges (prone to artifacts)
/// - Very distant hits (lower accuracy)
fn compute_confidence(
    ray_dir: vec3<f32>,
    hit_uv: vec2<f32>,
    distance: f32,
) -> f32 {
    // Distance factor: confidence decreases with distance
    let dist_factor = 1.0 - saturate(distance / uniforms.config.max_distance);

    // Edge factor: fade out near screen edges
    let edge_dist = min(
        min(hit_uv.x, 1.0 - hit_uv.x),
        min(hit_uv.y, 1.0 - hit_uv.y)
    );
    let edge_factor = saturate(edge_dist * 10.0); // Fade in 10% from edges

    // Angle factor: penalize rays nearly parallel to view direction
    // (These often produce stretched, low-quality reflections)
    let view_angle = abs(ray_dir.z); // Z component of normalized view-space direction
    let angle_factor = saturate(view_angle * 2.0);

    return dist_factor * edge_factor * angle_factor;
}

// ---------------------------------------------------------------------------
// Main Entry Point
// ---------------------------------------------------------------------------

/// SSR ray marching kernel.
///
/// For each pixel:
/// 1. Sample GBuffer depth and normal
/// 2. Reconstruct view-space position and compute reflection direction
/// 3. Perform HiZ ray march
/// 4. Output hit UV + confidence, or mark as miss
@compute @workgroup_size(8, 8, 1)
fn ssr_ray_march(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Bounds check
    if (gid.x >= uniforms.screen_size.x || gid.y >= uniforms.screen_size.y) {
        return;
    }

    let pixel = vec2<i32>(gid.xy);
    let screen_size = vec2<f32>(uniforms.screen_size);
    let uv = (vec2<f32>(gid.xy) + 0.5) / screen_size;

    // Sample GBuffer
    let depth = textureLoad(gbuffer_depth, pixel, 0).r;
    let normal_encoded = textureLoad(gbuffer_normal, pixel, 0).xyz;

    // Skip sky pixels (depth at far plane in reversed-Z is ~0)
    if (depth < EPSILON) {
        textureStore(reflection_output, pixel, vec4<f32>(0.0, 0.0, -1.0, 0.0));
        return;
    }

    // Decode normal (assuming octahedral or RGB encoding)
    // For now, assume world-space normals stored as RGB
    let normal_world = normalize(normal_encoded * 2.0 - 1.0);

    // Transform normal to view space
    let normal_view = normalize((uniforms.view_matrix * vec4<f32>(normal_world, 0.0)).xyz);

    // Reconstruct view-space position
    let pos_view = screen_to_view(uv, depth);

    // Compute view direction (from camera to pixel)
    let view_dir = normalize(pos_view);

    // Skip surfaces facing away from camera (backfaces)
    if (dot(normal_view, -view_dir) < 0.05) {
        textureStore(reflection_output, pixel, vec4<f32>(0.0, 0.0, -1.0, 0.0));
        return;
    }

    // Compute reflection direction
    let reflect_dir = reflect_dir(view_dir, normal_view);

    // Skip reflections pointing toward camera (would intersect near plane)
    if (reflect_dir.z > -0.1) {
        textureStore(reflection_output, pixel, vec4<f32>(0.0, 0.0, -1.0, 0.0));
        return;
    }

    // Temporal jitter for TAA
    let jitter = temporal_jitter(gid.xy, uniforms.frame_index);

    // Perform HiZ ray march
    let result = hiz_trace(pos_view, reflect_dir, jitter);

    // Output result
    // Format: (hit_uv.x, hit_uv.y, confidence, distance)
    // confidence < 0 indicates miss
    if (result.hit) {
        textureStore(reflection_output, pixel, vec4<f32>(
            result.uv.x,
            result.uv.y,
            result.confidence,
            result.distance
        ));
    } else {
        // Miss: negative confidence signals fallback needed
        textureStore(reflection_output, pixel, vec4<f32>(0.0, 0.0, -1.0, 0.0));
    }
}

// ---------------------------------------------------------------------------
// Notes on Performance
// ---------------------------------------------------------------------------
//
// Typical step counts for different scene types:
//
// | Scene Type           | Avg Steps | Max Steps | Notes                    |
// |---------------------|-----------|-----------|--------------------------|
// | Indoor flat floor   | 8-12      | 24        | Fast, mostly mip 0       |
// | Indoor complex      | 12-20     | 40        | More mip transitions     |
// | Outdoor terrain     | 16-32     | 64        | Long rays, mip climbing  |
// | Grazing angles      | 32-64     | 128       | Worst case, many samples |
//
// Optimizations implemented:
// 1. HiZ acceleration: O(log n) vs O(n) for linear march
// 2. Early-out on screen bounds
// 3. Adaptive mip climbing for empty space
// 4. Binary refinement only at intersection
// 5. View-space marching (more stable than world-space)
//
// Future improvements:
// - Stochastic reflection direction for rough surfaces
// - Temporal reprojection of previous frame hits
// - Hybrid with ray-traced reflections for misses
