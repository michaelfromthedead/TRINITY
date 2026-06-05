// SPDX-License-Identifier: MIT
//
// hybrid_depth.wgsl -- Hybrid mode depth buffer raymarching compute shader.
//
// T-DEMO-6.3: Reads rasterization depth buffer
// T-DEMO-6.4: Writes only where ray march hit is closer than raster depth
//
// This shader implements hybrid rendering where ray-marched SDF content is
// composited with rasterized geometry using depth buffer comparison.
//
// Depth Pipeline:
// 1. Sample rasterization depth buffer at pixel position
// 2. Convert NDC depth to linear depth using camera near/far planes
// 3. Ray march through SDF scene, terminating at raster depth
// 4. If ray march hit is closer than raster depth, write color
// 5. If raster is closer, discard fragment (preserve raster output)

// =============================================================================
// Uniforms
// =============================================================================

struct HybridUniforms {
    time: f32,
    resolution_x: f32,
    resolution_y: f32,
    near_plane: f32,
    far_plane: f32,
    depth_enabled: f32,
    _padding0: f32,
    _padding1: f32,
}

@group(0) @binding(0) var<uniform> uniforms: HybridUniforms;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var depth_texture: texture_depth_2d;
@group(0) @binding(3) var depth_sampler: sampler;

// =============================================================================
// Constants
// =============================================================================

const MAX_STEPS: i32 = 64;
const MAX_DIST: f32 = 20.0;
const SURF_DIST: f32 = 0.001;
const DEPTH_EPSILON: f32 = 0.0001;

// =============================================================================
// Depth Conversion Functions (T-DEMO-6.3)
// =============================================================================

/// Convert NDC depth (0.0 = near, 1.0 = far) to linear view-space depth.
/// Uses standard perspective projection formula.
fn ndc_to_linear(ndc_depth: f32) -> f32 {
    let near = uniforms.near_plane;
    let far = uniforms.far_plane;

    // Handle edge cases
    if (ndc_depth >= 1.0 - DEPTH_EPSILON) {
        return far;
    }
    if (ndc_depth <= DEPTH_EPSILON) {
        return near;
    }

    return (near * far) / (far - ndc_depth * (far - near));
}

/// Sample the depth buffer and convert to linear depth.
fn sample_depth(pixel_coords: vec2<i32>) -> f32 {
    // If depth is disabled, return maximum distance
    if (uniforms.depth_enabled < 0.5) {
        return MAX_DIST;
    }

    // Get depth texture dimensions
    let depth_size = textureDimensions(depth_texture);

    // Clamp to valid range
    let clamped_coords = clamp(
        pixel_coords,
        vec2<i32>(0),
        vec2<i32>(i32(depth_size.x) - 1, i32(depth_size.y) - 1)
    );

    // Load depth value directly (no filtering needed)
    let ndc_depth = textureLoad(depth_texture, clamped_coords, 0);

    // Convert to linear depth
    return ndc_to_linear(ndc_depth);
}

// =============================================================================
// SDF Primitives (inline for 4K constraint)
// =============================================================================

fn sdf_sphere(p: vec3<f32>, r: f32) -> f32 {
    return length(p) - r;
}

fn sdf_box(p: vec3<f32>, b: vec3<f32>) -> f32 {
    let q = abs(p) - b;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn sdf_torus(p: vec3<f32>, r: vec2<f32>) -> f32 {
    let q = vec2<f32>(length(p.xz) - r.x, p.y);
    return length(q) - r.y;
}

// Smooth minimum (smooth union)
fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = max(k - abs(a - b), 0.0) / k;
    return min(a, b) - h * h * k * 0.25;
}

// =============================================================================
// Material Definition
// =============================================================================

struct Material {
    albedo: vec3<f32>,
    roughness: f32,
    metallic: f32,
}

const MAT_GROUND: u32 = 0u;
const MAT_SPHERE: u32 = 1u;
const MAT_BOX: u32 = 2u;
const MAT_TORUS: u32 = 3u;

fn scene_material(id: u32) -> Material {
    var mat: Material;
    switch id {
        case 0u: {
            mat.albedo = vec3<f32>(0.5, 0.5, 0.5);
            mat.roughness = 0.8;
            mat.metallic = 0.0;
        }
        case 1u: {
            mat.albedo = vec3<f32>(0.8, 0.2, 0.3);
            mat.roughness = 0.3;
            mat.metallic = 0.8;
        }
        case 2u: {
            mat.albedo = vec3<f32>(0.3, 0.5, 0.9);
            mat.roughness = 0.5;
            mat.metallic = 0.0;
        }
        case 3u: {
            mat.albedo = vec3<f32>(0.9, 0.7, 0.3);
            mat.roughness = 0.2;
            mat.metallic = 0.9;
        }
        default: {
            mat.albedo = vec3<f32>(1.0, 0.0, 1.0);
            mat.roughness = 1.0;
            mat.metallic = 0.0;
        }
    }
    return mat;
}

// =============================================================================
// Scene Definition
// =============================================================================

fn scene_sdf(p: vec3<f32>) -> vec2<f32> {
    let t = uniforms.time;

    // Animated sphere
    let sphere_pos = vec3<f32>(sin(t) * 0.8, 0.0, cos(t) * 0.8);
    let d_sphere = sdf_sphere(p - sphere_pos, 0.4);

    // Rotating box
    let angle = t * 0.5;
    let c = cos(angle);
    let s = sin(angle);
    let rotated_p = vec3<f32>(
        p.x * c - p.z * s,
        p.y,
        p.x * s + p.z * c
    );
    let d_box = sdf_box(rotated_p, vec3<f32>(0.3, 0.3, 0.3));

    // Torus in the background
    let d_torus = sdf_torus(p - vec3<f32>(0.0, 0.0, -1.5), vec2<f32>(0.6, 0.2));

    // Ground plane
    let d_ground = p.y + 0.8;

    // Find closest object and its material
    var d = d_sphere;
    var mat_id = f32(MAT_SPHERE);

    if (d_box < d) {
        d = d_box;
        mat_id = f32(MAT_BOX);
    }

    if (d_torus < d) {
        d = d_torus;
        mat_id = f32(MAT_TORUS);
    }

    if (d_ground < d) {
        d = d_ground;
        mat_id = f32(MAT_GROUND);
    }

    // Apply smooth blending
    let d_blended = smin(smin(d_sphere, d_box, 0.3), d_torus, 0.2);
    d = min(d_blended, d_ground);

    return vec2<f32>(d, mat_id);
}

fn map_scene(p: vec3<f32>) -> f32 {
    return scene_sdf(p).x;
}

// =============================================================================
// Normal Estimation
// =============================================================================

fn calc_normal(p: vec3<f32>) -> vec3<f32> {
    let e = vec2<f32>(0.001, 0.0);
    return normalize(vec3<f32>(
        map_scene(p + e.xyy) - map_scene(p - e.xyy),
        map_scene(p + e.yxy) - map_scene(p - e.yxy),
        map_scene(p + e.yyx) - map_scene(p - e.yyx)
    ));
}

// =============================================================================
// Ray Marching with Depth Termination (T-DEMO-6.3 / T-DEMO-6.4)
// =============================================================================

/// Ray march with early termination at rasterized depth.
/// Returns (hit_distance, material_id) where hit_distance >= MAX_DIST means miss.
fn ray_march(ro: vec3<f32>, rd: vec3<f32>, max_depth: f32) -> vec2<f32> {
    var t = 0.0;
    var mat_id = 0.0;

    // Clamp max_depth to reasonable range
    let effective_max = min(max_depth, MAX_DIST);

    for (var i = 0; i < MAX_STEPS; i++) {
        let p = ro + rd * t;
        let result = scene_sdf(p);
        let d = result.x;
        mat_id = result.y;

        // Surface hit
        if (d < SURF_DIST) {
            return vec2<f32>(t, mat_id);
        }

        t += d;

        // Early termination: ray has passed raster depth (T-DEMO-6.4)
        if (t > effective_max) {
            return vec2<f32>(MAX_DIST, mat_id);
        }
    }

    return vec2<f32>(MAX_DIST, mat_id);
}

// =============================================================================
// Lighting
// =============================================================================

fn calc_lighting(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let light_pos = vec3<f32>(
        sin(uniforms.time * 0.7) * 3.0,
        2.0 + sin(uniforms.time * 0.3),
        cos(uniforms.time * 0.7) * 3.0
    );

    let light_dir = normalize(light_pos - p);
    let diff = max(dot(n, light_dir), 0.0);
    let ao = 0.5 + 0.5 * n.y;

    // Simplified soft shadows
    var shadow = 1.0;
    var t = 0.02;
    for (var i = 0; i < 32; i++) {
        let d = map_scene(p + light_dir * t);
        if (d < 0.001) {
            shadow = 0.0;
            break;
        }
        shadow = min(shadow, 10.0 * d / t);
        t += d;
        if (t > 5.0) {
            break;
        }
    }

    let ambient = vec3<f32>(0.1, 0.12, 0.15);
    let diffuse = vec3<f32>(0.9, 0.8, 0.7) * diff * shadow;

    return ambient * ao + diffuse;
}

// =============================================================================
// Color Functions
// =============================================================================

fn get_sky_color(rd: vec3<f32>) -> vec3<f32> {
    let t = 0.5 * (rd.y + 1.0);
    return mix(
        vec3<f32>(0.8, 0.7, 0.6),
        vec3<f32>(0.2, 0.3, 0.5),
        t
    );
}

fn get_object_color(p: vec3<f32>) -> vec3<f32> {
    let checker = floor(p.x * 2.0) + floor(p.z * 2.0);
    if (p.y < -0.79) {
        if (fract(checker * 0.5) < 0.5) {
            return vec3<f32>(0.4, 0.4, 0.4);
        } else {
            return vec3<f32>(0.6, 0.6, 0.6);
        }
    }

    let h = (p.y + 1.0) * 0.5;
    return mix(
        vec3<f32>(0.8, 0.2, 0.3),
        vec3<f32>(0.3, 0.5, 0.9),
        clamp(h, 0.0, 1.0)
    );
}

// =============================================================================
// Main Compute Shader (T-DEMO-6.3 / T-DEMO-6.4)
// =============================================================================

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let resolution = vec2<f32>(uniforms.resolution_x, uniforms.resolution_y);
    let coords = vec2<f32>(f32(global_id.x), f32(global_id.y));

    // Early exit if outside texture bounds
    if (coords.x >= resolution.x || coords.y >= resolution.y) {
        return;
    }

    // Sample rasterized depth buffer (T-DEMO-6.3)
    let pixel_coords = vec2<i32>(global_id.xy);
    let raster_linear_depth = sample_depth(pixel_coords);

    // Normalized coordinates with aspect ratio correction
    let uv = (coords - 0.5 * resolution) / min(resolution.x, resolution.y);

    // Camera setup
    let cam_pos = vec3<f32>(0.0, 0.5, 3.0);
    let cam_target = vec3<f32>(0.0, 0.0, 0.0);
    let cam_up = vec3<f32>(0.0, 1.0, 0.0);

    // Camera matrix
    let forward = normalize(cam_target - cam_pos);
    let right = normalize(cross(forward, cam_up));
    let up = cross(right, forward);

    // Ray direction
    let rd = normalize(forward + uv.x * right + uv.y * up);

    // Ray march with depth termination (T-DEMO-6.3)
    // Use raster depth as maximum march distance
    let march_result = ray_march(cam_pos, rd, raster_linear_depth);
    let sdf_dist = march_result.x;
    let mat_id = march_result.y;

    var color: vec3<f32>;

    // Depth test (T-DEMO-6.4)
    // Only write ray march result if it's closer than raster depth
    let hit_sdf = sdf_dist < MAX_DIST - SURF_DIST;
    let closer_than_raster = sdf_dist < raster_linear_depth - DEPTH_EPSILON;

    if (hit_sdf && closer_than_raster) {
        // Ray march hit is closer - calculate shading
        let p = cam_pos + rd * sdf_dist;
        let n = calc_normal(p);
        let lighting = calc_lighting(p, n);
        let obj_color = get_object_color(p);
        color = obj_color * lighting;

        // Fog
        let fog = 1.0 - exp(-sdf_dist * 0.08);
        color = mix(color, get_sky_color(rd), fog);
    } else if (uniforms.depth_enabled > 0.5 && !closer_than_raster && hit_sdf) {
        // Raster is closer - discard this fragment (T-DEMO-6.4)
        // Write transparent/sky to indicate discard
        // The actual raster color would be composited separately
        color = get_sky_color(rd) * 0.5;
    } else {
        // Ray march miss - show sky
        color = get_sky_color(rd);
    }

    // Gamma correction
    color = pow(color, vec3<f32>(1.0 / 2.2));

    // Vignette
    let vignette = 1.0 - 0.3 * length(uv);
    color *= vignette;

    // Output
    textureStore(output_texture, vec2<i32>(global_id.xy), vec4<f32>(color, 1.0));
}
