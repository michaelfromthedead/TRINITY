// SPDX-License-Identifier: MIT
//
// pbr.frag.wgsl — PBR fragment shader (T-BRG-6.2).
//
// Cook-Torrance BRDF with GGX normal distribution, Smith-GGX geometry,
// and Schlick Fresnel. Supports bindless material table, point/directional/spot
// lights, and optional CSM shadow sampling.

// ── Material table entry (matches material_table.wgsl layout, 80 bytes) ──

struct MaterialTableEntry {
    base_color: vec4<f32>,
    emissive: vec4<f32>,
    metallic: f32,
    roughness: f32,
    occlusion: f32,
    normal_scale: f32,
    albedo_texture_id: u32,
    normal_texture_id: u32,
    metallic_roughness_tex_id: u32,
    emissive_texture_id: u32,
    flags: u32,
    alpha_cutoff: f32,
}

// ── Light types ──

struct DirectionalLight {
    direction: vec3<f32>,
    _pad0: f32,
    color: vec3<f32>,
    intensity: f32,
}

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

struct LightCounts {
    num_directional: u32,
    num_point: u32,
    num_spot: u32,
    _pad: u32,
}

// ── Camera uniforms (group 0) ──

struct CameraUniforms {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    view_projection: mat4x4<f32>,
    camera_position: vec3<f32>,
    ambient_intensity: f32,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;

// ── Material table (group 1, bindless) ──

@group(1) @binding(0) var<storage, read> material_table: array<MaterialTableEntry>;

// ── Light buffers (group 2) ──

@group(2) @binding(0) var<uniform> light_counts: LightCounts;
@group(2) @binding(1) var<storage, read> dir_lights: array<DirectionalLight>;
@group(2) @binding(2) var<storage, read> point_lights: array<PointLight>;
@group(2) @binding(3) var<storage, read> spot_lights: array<SpotLight>;

// ── Shadow map resources (group 3) ──

struct CascadeData {
    light_view_proj: mat4x4<f32>,
    split_depth: f32,
    shadow_map_index: u32,
    _pad0: f32,
    _pad1: f32,
}

@group(3) @binding(0) var<uniform> cascade_data: array<CascadeData, 4>;
@group(3) @binding(1) var shadow_maps: texture_depth_2d_array;
@group(3) @binding(2) var shadow_sampler: sampler_comparison;
@group(3) @binding(3) var<uniform> shadow_params: vec4<f32>; // (bias, slope_bias, normal_bias, pcf_radius)

// ── Fragment inputs ──

struct FragmentInput {
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) world_tangent: vec4<f32>,
    @location(3) texcoord: vec2<f32>,
    @location(4) material_index: u32,
}

// ── Constants ──

const PI: f32 = 3.14159265359;
const EPSILON: f32 = 0.00001;
const MAX_REFLECTION_LOD: f32 = 4.0;

// ── Cook-Torrance BRDF functions ──

// Trowbridge-Reitz (GGX) normal distribution function.
fn distribution_ggx(n: vec3<f32>, h: vec3<f32>, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let ndoth = max(dot(n, h), 0.0);
    let ndoth2 = ndoth * ndoth;

    let denom = ndoth2 * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

// Smith-GGX geometry function for a single direction.
fn geometry_schlick_ggx(ndotv: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return ndotv / (ndotv * (1.0 - k) + k);
}

// Smith-GGX geometry function combining view and light directions.
fn geometry_smith(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, roughness: f32) -> f32 {
    let ndotv = max(dot(n, v), 0.0);
    let ndotl = max(dot(n, l), 0.0);
    return geometry_schlick_ggx(ndotv, roughness) * geometry_schlick_ggx(ndotl, roughness);
}

// Schlick Fresnel approximation.
fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// ── Shadow sampling ──

// Selects the CSM cascade index based on view-space depth.
fn select_cascade(view_depth: f32) -> u32 {
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        if view_depth < cascade_data[i].split_depth {
            return i;
        }
    }
    return 3u; // last cascade
}

// Computes shadow factor [0,1] using PCF on the selected cascade.
fn shadow_factor(world_pos: vec3<f32>, normal: vec3<f32>, light_dir: vec3<f32>) -> f32 {
    // Compute view-space depth for cascade selection.
    let view_pos = camera.view * vec4<f32>(world_pos, 1.0);
    let view_depth = abs(view_pos.z);
    let cascade_idx = select_cascade(view_depth);

    let c = cascade_data[cascade_idx];

    // Transform to light clip space.
    let light_clip = c.light_view_proj * vec4<f32>(world_pos, 1.0);
    if light_clip.w <= 0.0 {
        return 1.0; // behind light, not shadowed
    }

    var uv = light_clip.xyz / light_clip.w;
    if any(uv.xyz < vec3<f32>(-1.0)) || any(uv.xyz > vec3<f32>(1.0)) {
        return 1.0; // outside cascade frustum
    }

    // Apply depth biases.
    let cos_theta = max(dot(normal, light_dir), 0.0);
    let slope_bias = shadow_params.y * tan(acos(cos_theta));
    let bias = shadow_params.x + slope_bias;
    let normal_bias = shadow_params.z * (1.0 - cos_theta);
    uv.z = uv.z - bias - normal_bias;

    // Transform [-1,1] to [0,1] UV, depth stays [0,1] for comparison.
    let shadow_uv = vec3<f32>(uv.xy * 0.5 + 0.5, cascade_idx);

    // PCF with configurable kernel.
    let pcf_r = u32(shadow_params.w);
    let texel_size = 1.0 / 2048.0; // shadow map resolution

    var shadow: f32 = 0.0;
    let kernel_size = 2 * pcf_r + 1u;
    for (var x: u32 = 0u; x < kernel_size; x = x + 1u) {
        for (var y: u32 = 0u; y < kernel_size; y = y + 1u) {
            let offset = vec2<f32>(f32(x) - f32(pcf_r), f32(y) - f32(pcf_r)) * texel_size;
            let sample_uv = vec3<f32>(shadow_uv.xy + offset, shadow_uv.z);
            shadow = shadow + textureSampleCompare(
                shadow_maps, shadow_sampler, sample_uv.xy, i32(sample_uv.z), uv.z
            );
        }
    }
    let sample_count = f32(kernel_size * kernel_size);
    return shadow / sample_count;
}

// ── Lighting computation ──

// Computes radiance from a point light with distance attenuation.
fn eval_point_light(
    light: PointLight,
    n: vec3<f32>,
    v: vec3<f32>,
    world_pos: vec3<f32>,
    albedo: vec3<f32>,
    f0: vec3<f32>,
    roughness: f32,
) -> vec3<f32> {
    let to_light = light.position - world_pos;
    let dist = length(to_light);
    if dist > light.radius {
        return vec3<f32>(0.0);
    }
    let l = to_light / dist;
    let h = normalize(v + l);

    // Distance attenuation (smooth falloff).
    let attenuation = pow(clamp(1.0 - (dist * dist) / (light.radius * light.radius), 0.0, 1.0), 2.0);

    let radiance = light.color * light.intensity * attenuation;

    return eval_brdf(n, v, l, h, albedo, f0, roughness, radiance);
}

// Computes radiance from a directional light.
fn eval_directional_light(
    light: DirectionalLight,
    n: vec3<f32>,
    v: vec3<f32>,
    world_pos: vec3<f32>,
    albedo: vec3<f32>,
    f0: vec3<f32>,
    roughness: f32,
) -> vec3<f32> {
    let l = normalize(-light.direction);
    let h = normalize(v + l);

    let radiance = light.color * light.intensity;

    // Apply shadow factor for the first directional light (sun).
    let shadow = shadow_factor(world_pos, n, l);

    return eval_brdf(n, v, l, h, albedo, f0, roughness, radiance * shadow);
}

// Computes radiance from a spot light with cone attenuation.
fn eval_spot_light(
    light: SpotLight,
    n: vec3<f32>,
    v: vec3<f32>,
    world_pos: vec3<f32>,
    albedo: vec3<f32>,
    f0: vec3<f32>,
    roughness: f32,
) -> vec3<f32> {
    let to_light = light.position - world_pos;
    let dist = length(to_light);
    if dist > light.radius {
        return vec3<f32>(0.0);
    }
    let l = to_light / dist;
    let h = normalize(v + l);

    // Distance attenuation.
    let dist_att = pow(clamp(1.0 - (dist * dist) / (light.radius * light.radius), 0.0, 1.0), 2.0);

    // Cone attenuation (smooth inner-to-outer).
    let spot_dir = normalize(-light.direction);
    let cos_theta = dot(-l, spot_dir);
    let cone_att = smoothstep(light.cos_outer_angle, light.cos_inner_angle, cos_theta);

    let radiance = light.color * light.intensity * dist_att * cone_att;

    return eval_brdf(n, v, l, h, albedo, f0, roughness, radiance);
}

// Core BRDF evaluation (shared by all light types).
fn eval_brdf(
    n: vec3<f32>,
    v: vec3<f32>,
    l: vec3<f32>,
    h: vec3<f32>,
    albedo: vec3<f32>,
    f0: vec3<f32>,
    roughness: f32,
    radiance: vec3<f32>,
) -> vec3<f32> {
    let ndotl = max(dot(n, l), 0.0);
    let ndotv = max(dot(n, v), 0.0);
    if ndotl <= 0.0 || ndotv <= 0.0 {
        return vec3<f32>(0.0);
    }

    // Cook-Torrance specular.
    let ndf = distribution_ggx(n, h, roughness);
    let g = geometry_smith(n, v, l, roughness);
    let f = fresnel_schlick(max(dot(h, v), 0.0), f0);

    let specular = (ndf * g * f) / max(4.0 * ndotv * ndotl, EPSILON);

    // Lambertian diffuse (energy-conserving).
    let kd = (1.0 - f) * (1.0 - albedo_to_metallic_f0(albedo).g); // approximate metallic factor

    return (kd * albedo / PI + specular) * radiance * ndotl;
}

// Helper: approximate F0 from albedo (used to scale diffuse for energy conservation).
fn albedo_to_metallic_f0(albedo: vec3<f32>) -> vec3<f32> {
    return albedo; // simplified — metallic controls F0 directly in the main path
}

// ── Main ──

@fragment
fn fs_main(input: FragmentInput) -> @location(0) vec4<f32> {
    let material = material_table[input.material_index];

    // Extract material parameters.
    let albedo = material.base_color.rgb;
    let metallic = material.metallic;
    let roughness = max(material.roughness, 0.04); // minimum roughness to avoid sparkles
    let ao = material.occlusion;
    let emissive = material.emissive.rgb * material.emissive.a;

    // Compute world-space normal (with optional normal mapping would go here).
    let n = normalize(input.world_normal);

    // Compute view direction.
    let v = normalize(camera.camera_position - input.world_position);

    // Compute F0 (surface reflectance at normal incidence).
    // Dielectric F0 is typically ~0.04; metals use albedo as F0.
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);

    var lo: vec3<f32> = vec3<f32>(0.0);

    // Accumulate directional lights.
    for (var i: u32 = 0u; i < light_counts.num_directional; i = i + 1u) {
        lo = lo + eval_directional_light(
            dir_lights[i], n, v, input.world_position, albedo, f0, roughness
        );
    }

    // Accumulate point lights.
    for (var j: u32 = 0u; j < light_counts.num_point; j = j + 1u) {
        lo = lo + eval_point_light(
            point_lights[j], n, v, input.world_position, albedo, f0, roughness
        );
    }

    // Accumulate spot lights.
    for (var k: u32 = 0u; k < light_counts.num_spot; k = k + 1u) {
        lo = lo + eval_spot_light(
            spot_lights[k], n, v, input.world_position, albedo, f0, roughness
        );
    }

    // Ambient term.
    let ambient = camera.ambient_intensity * albedo * ao;

    // Final colour = emissive + ambient + direct lighting.
    let color = emissive + ambient + lo;

    return vec4<f32>(color, material.base_color.a);
}
