// SPDX-License-Identifier: MIT
//
// demo.wgsl -- Minimal demoscene raymarching compute shader.
//
// This shader implements a basic SDF ray marcher for 4K demoscene effects.
// Designed for real-time execution with @workgroup_size(8, 8, 1).
//
// Uniforms:
//   - time: Animation time in seconds
//   - resolution: Output texture resolution (width, height)
//   - _padding: Alignment padding
//
// Output:
//   - RGBA storage texture written via textureStore

// =============================================================================
// Uniforms
// =============================================================================

struct DemoUniforms {
    time: f32,
    resolution_x: f32,
    resolution_y: f32,
    _padding: f32,
}

@group(0) @binding(0) var<uniform> uniforms: DemoUniforms;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba8unorm, write>;

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

/// Material IDs for scene objects.
const MAT_GROUND: u32 = 0u;
const MAT_SPHERE: u32 = 1u;
const MAT_BOX: u32 = 2u;
const MAT_TORUS: u32 = 3u;

/// Get material properties by ID.
fn scene_material(id: u32) -> Material {
    var mat: Material;
    switch id {
        case 0u: {
            // Ground - gray checkerboard
            mat.albedo = vec3<f32>(0.5, 0.5, 0.5);
            mat.roughness = 0.8;
            mat.metallic = 0.0;
        }
        case 1u: {
            // Sphere - red metallic
            mat.albedo = vec3<f32>(0.8, 0.2, 0.3);
            mat.roughness = 0.3;
            mat.metallic = 0.8;
        }
        case 2u: {
            // Box - blue plastic
            mat.albedo = vec3<f32>(0.3, 0.5, 0.9);
            mat.roughness = 0.5;
            mat.metallic = 0.0;
        }
        case 3u: {
            // Torus - gold metallic
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

/// Scene SDF returning (distance, material_id).
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

    // Apply smooth blending for visual effect (but return closest material)
    let d_blended = smin(smin(d_sphere, d_box, 0.3), d_torus, 0.2);
    d = min(d_blended, d_ground);

    return vec2<f32>(d, mat_id);
}

/// Legacy map_scene for backward compatibility.
fn map_scene(p: vec3<f32>) -> f32 {
    return scene_sdf(p).x;
}

// =============================================================================
// Normal Estimation (central differences)
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
// Ray Marching
// =============================================================================

const MAX_STEPS: i32 = 64;
const MAX_DIST: f32 = 20.0;
const SURF_DIST: f32 = 0.001;

fn ray_march(ro: vec3<f32>, rd: vec3<f32>) -> f32 {
    var t = 0.0;

    for (var i = 0; i < MAX_STEPS; i++) {
        let p = ro + rd * t;
        let d = map_scene(p);

        if (d < SURF_DIST) {
            break;
        }

        t += d;

        if (t > MAX_DIST) {
            break;
        }
    }

    return t;
}

// =============================================================================
// Lighting
// =============================================================================

fn calc_lighting(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    // Animated light position
    let light_pos = vec3<f32>(
        sin(uniforms.time * 0.7) * 3.0,
        2.0 + sin(uniforms.time * 0.3),
        cos(uniforms.time * 0.7) * 3.0
    );

    let light_dir = normalize(light_pos - p);
    let diff = max(dot(n, light_dir), 0.0);

    // Ambient occlusion approximation
    let ao = 0.5 + 0.5 * n.y;

    // Soft shadows
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

    // Combine lighting
    let ambient = vec3<f32>(0.1, 0.12, 0.15);
    let diffuse = vec3<f32>(0.9, 0.8, 0.7) * diff * shadow;

    return ambient * ao + diffuse;
}

// =============================================================================
// Color Gradient
// =============================================================================

fn get_sky_color(rd: vec3<f32>) -> vec3<f32> {
    let t = 0.5 * (rd.y + 1.0);
    return mix(
        vec3<f32>(0.8, 0.7, 0.6),  // Horizon
        vec3<f32>(0.2, 0.3, 0.5),  // Zenith
        t
    );
}

fn get_object_color(p: vec3<f32>) -> vec3<f32> {
    // Simple material based on position
    let checker = floor(p.x * 2.0) + floor(p.z * 2.0);
    if (p.y < -0.79) {
        // Ground plane - checkerboard
        if (fract(checker * 0.5) < 0.5) {
            return vec3<f32>(0.4, 0.4, 0.4);
        } else {
            return vec3<f32>(0.6, 0.6, 0.6);
        }
    }

    // Objects - gradient based on height
    let h = (p.y + 1.0) * 0.5;
    return mix(
        vec3<f32>(0.8, 0.2, 0.3),
        vec3<f32>(0.3, 0.5, 0.9),
        clamp(h, 0.0, 1.0)
    );
}

// =============================================================================
// Main Compute Shader
// =============================================================================

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let resolution = vec2<f32>(uniforms.resolution_x, uniforms.resolution_y);
    let coords = vec2<f32>(f32(global_id.x), f32(global_id.y));

    // Early exit if outside texture bounds
    if (coords.x >= resolution.x || coords.y >= resolution.y) {
        return;
    }

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

    // Ray march
    let t = ray_march(cam_pos, rd);

    var color: vec3<f32>;

    if (t < MAX_DIST) {
        // Hit - calculate shading
        let p = cam_pos + rd * t;
        let n = calc_normal(p);
        let lighting = calc_lighting(p, n);
        let obj_color = get_object_color(p);
        color = obj_color * lighting;

        // Fog
        let fog = 1.0 - exp(-t * 0.08);
        color = mix(color, get_sky_color(rd), fog);
    } else {
        // Miss - sky gradient
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
