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
    // T-MAT-4.2: Clear coat layer parameters
    clear_coat: f32,           // 0.0 = no coat, 1.0 = full coat
    clear_coat_roughness: f32, // Roughness of clear coat layer [0,1]
    // T-MAT-4.3: Anisotropy parameters
    anisotropy: f32,           // -1.0 to 1.0, 0.0 = isotropic
    anisotropy_rotation: f32,  // 0.0 to 2*PI radians
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
@group(3) @binding(4) var<uniform> cascade_blend_range: f32; // T-LIT-4.4: blend range in world units (default 2.0)

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

// T-MAT-4.2: Clear coat F0 for IOR 1.5 (polyurethane/lacquer)
// Computed as: ((n-1)/(n+1))^2 = ((1.5-1)/(1.5+1))^2 = 0.04
const CLEAR_COAT_F0: f32 = 0.04;

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

// T-MAT-4.3: Anisotropic GGX Normal Distribution Function.
// Produces elliptically stretched specular highlights on surfaces like brushed metal.
// Formula: D = 1 / (PI * alpha_t * alpha_b * denom^2)
// where denom = (ToH/alpha_t)^2 + (BoH/alpha_b)^2 + NoH^2
fn distribution_ggx_anisotropic(
    NoH: f32,
    ToH: f32,
    BoH: f32,
    alpha_t: f32,
    alpha_b: f32
) -> f32 {
    let at = max(alpha_t, EPSILON);
    let ab = max(alpha_b, EPSILON);

    let at2 = at * at;
    let ab2 = ab * ab;

    let term_t = ToH * ToH / at2;
    let term_b = BoH * BoH / ab2;
    let term_n = NoH * NoH;

    let denom = term_t + term_b + term_n;
    return 1.0 / (PI * at * ab * denom * denom + EPSILON);
}

// T-MAT-4.3: Anisotropic Smith-GGX geometry function.
// Height-correlated form combining view and light masking/shadowing.
fn geometry_smith_anisotropic(
    NoV: f32,
    NoL: f32,
    ToV: f32,
    BoV: f32,
    ToL: f32,
    BoL: f32,
    alpha_t: f32,
    alpha_b: f32
) -> f32 {
    let at2 = alpha_t * alpha_t;
    let ab2 = alpha_b * alpha_b;

    // Compute projected roughness for view direction
    let lambdaV = NoL * sqrt(at2 * ToV * ToV + ab2 * BoV * BoV + NoV * NoV);
    // Compute projected roughness for light direction
    let lambdaL = NoV * sqrt(at2 * ToL * ToL + ab2 * BoL * BoL + NoL * NoL);

    return 0.5 / (lambdaV + lambdaL + EPSILON);
}

// T-MAT-4.3: Rotate tangent vector by anisotropy rotation angle.
fn rotate_tangent_by_angle(tangent: vec3<f32>, bitangent: vec3<f32>, angle: f32) -> vec3<f32> {
    let cos_a = cos(angle);
    let sin_a = sin(angle);
    return tangent * cos_a + bitangent * sin_a;
}

// T-MAT-4.3: Rotate bitangent vector by anisotropy rotation angle.
fn rotate_bitangent_by_angle(tangent: vec3<f32>, bitangent: vec3<f32>, angle: f32) -> vec3<f32> {
    let cos_a = cos(angle);
    let sin_a = sin(angle);
    return -tangent * sin_a + bitangent * cos_a;
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

// ── T-MAT-4.2: Clear Coat BRDF Functions ──

// Schlick Fresnel for clear coat layer (scalar, fixed F0 = 0.04).
fn fresnel_clear_coat(VoH: f32) -> f32 {
    let Fc = pow(1.0 - VoH, 5.0);
    return CLEAR_COAT_F0 + (1.0 - CLEAR_COAT_F0) * Fc;
}

// GGX NDF for clear coat layer with Disney roughness remapping.
fn distribution_ggx_clear_coat(NoH: f32, cc_roughness: f32) -> f32 {
    let a = cc_roughness * cc_roughness;
    let a2 = a * a;
    let NoH2 = NoH * NoH;
    let denom = NoH2 * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom + EPSILON);
}

// Kelemen visibility function for clear coat (simplified form).
// V = 1 / (4 * VoH^2), suitable for thin dielectric layers.
fn geometry_clear_coat_kelemen(VoH: f32) -> f32 {
    return 0.25 / (VoH * VoH + EPSILON);
}

// Evaluate clear coat BRDF contribution.
// Returns vec4: xyz = clear coat specular (grayscale), w = Fresnel factor * intensity.
fn eval_clear_coat(
    n: vec3<f32>,
    v: vec3<f32>,
    l: vec3<f32>,
    cc_intensity: f32,
    cc_roughness: f32,
) -> vec4<f32> {
    // Early exit if clear coat is disabled
    if cc_intensity < EPSILON {
        return vec4<f32>(0.0);
    }

    let h = normalize(v + l);
    let NoL = max(dot(n, l), 0.0);
    let NoV = max(dot(n, v), 0.0);
    let NoH = max(dot(n, h), 0.0);
    let VoH = max(dot(v, h), 0.0);

    // Early exit for grazing angles
    if NoL < EPSILON || NoV < EPSILON {
        return vec4<f32>(0.0);
    }

    // Evaluate clear coat BRDF: D * G * F
    let D = distribution_ggx_clear_coat(NoH, cc_roughness);
    let G = geometry_clear_coat_kelemen(VoH);
    let F = fresnel_clear_coat(VoH);

    let cc_brdf = D * G * F * cc_intensity;

    // Return grayscale specular with Fresnel factor for layer blending
    return vec4<f32>(cc_brdf, cc_brdf, cc_brdf, F * cc_intensity);
}

// ── Shadow sampling ──

// T-LIT-4.4: Cascade selection result with blend information.
struct CascadeSelection {
    primary_idx: u32,      // Primary cascade index
    secondary_idx: u32,    // Secondary cascade for blending (if in blend zone)
    blend_factor: f32,     // 0.0 = use primary only, 1.0 = use secondary only
}

// Selects the CSM cascade index based on view-space depth.
// Returns primary cascade, and blend information for soft transitions.
fn select_cascade_blended(view_depth: f32, blend_range: f32) -> CascadeSelection {
    var result: CascadeSelection;
    result.blend_factor = 0.0;
    result.secondary_idx = 3u;

    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let split_depth = cascade_data[i].split_depth;
        if view_depth < split_depth {
            result.primary_idx = i;

            // Check if within blend range of cascade boundary
            if blend_range > 0.0 && i < 3u {
                let blend_start = split_depth - blend_range;
                if view_depth > blend_start {
                    // Linear interpolation within blend zone
                    result.blend_factor = (view_depth - blend_start) / blend_range;
                    result.secondary_idx = i + 1u;
                }
            }
            return result;
        }
    }
    result.primary_idx = 3u;
    return result;
}

// Legacy select_cascade for compatibility.
fn select_cascade(view_depth: f32) -> u32 {
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        if view_depth < cascade_data[i].split_depth {
            return i;
        }
    }
    return 3u; // last cascade
}

// T-LIT-4.4: Samples shadow from a single cascade with PCF.
// Extracted to allow reuse for cascade blending.
fn sample_cascade_shadow(
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    light_dir: vec3<f32>,
    cascade_idx: u32
) -> f32 {
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

// Computes shadow factor [0,1] using PCF on the selected cascade.
// T-LIT-4.4: Now supports soft cascade transitions via cascade_blend_range.
fn shadow_factor(world_pos: vec3<f32>, normal: vec3<f32>, light_dir: vec3<f32>) -> f32 {
    // Compute view-space depth for cascade selection.
    let view_pos = camera.view * vec4<f32>(world_pos, 1.0);
    let view_depth = abs(view_pos.z);

    // Select cascade with blend information.
    let selection = select_cascade_blended(view_depth, cascade_blend_range);

    // Sample primary cascade.
    let primary_shadow = sample_cascade_shadow(world_pos, normal, light_dir, selection.primary_idx);

    // If not in blend zone, return primary result only.
    if selection.blend_factor <= 0.0 {
        return primary_shadow;
    }

    // Sample secondary cascade and blend.
    let secondary_shadow = sample_cascade_shadow(world_pos, normal, light_dir, selection.secondary_idx);

    // Linear interpolation between cascades for smooth transition.
    return mix(primary_shadow, secondary_shadow, selection.blend_factor);
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
    cc_intensity: f32,
    cc_roughness: f32,
    tangent: vec3<f32>,
    bitangent: vec3<f32>,
    anisotropy: f32,
    anisotropy_rotation: f32,
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

    return eval_brdf_full(n, v, l, h, albedo, f0, roughness, radiance,
                          cc_intensity, cc_roughness,
                          tangent, bitangent, anisotropy, anisotropy_rotation);
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
    cc_intensity: f32,
    cc_roughness: f32,
    tangent: vec3<f32>,
    bitangent: vec3<f32>,
    anisotropy: f32,
    anisotropy_rotation: f32,
) -> vec3<f32> {
    let l = normalize(-light.direction);
    let h = normalize(v + l);

    let radiance = light.color * light.intensity;

    // Apply shadow factor for the first directional light (sun).
    let shadow = shadow_factor(world_pos, n, l);

    return eval_brdf_full(n, v, l, h, albedo, f0, roughness, radiance * shadow,
                          cc_intensity, cc_roughness,
                          tangent, bitangent, anisotropy, anisotropy_rotation);
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
    cc_intensity: f32,
    cc_roughness: f32,
    tangent: vec3<f32>,
    bitangent: vec3<f32>,
    anisotropy: f32,
    anisotropy_rotation: f32,
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

    return eval_brdf_full(n, v, l, h, albedo, f0, roughness, radiance,
                          cc_intensity, cc_roughness,
                          tangent, bitangent, anisotropy, anisotropy_rotation);
}

// Core BRDF evaluation (shared by all light types).
// Now supports optional clear coat layer (T-MAT-4.2).
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
    return eval_brdf_with_clear_coat(n, v, l, h, albedo, f0, roughness, radiance, 0.0, 0.0);
}

// Core BRDF evaluation with clear coat support (T-MAT-4.2).
// Clear coat is a dual-layer BRDF with energy conservation:
//   final = clear_coat_brdf + (1 - Fc * intensity) * base_brdf
fn eval_brdf_with_clear_coat(
    n: vec3<f32>,
    v: vec3<f32>,
    l: vec3<f32>,
    h: vec3<f32>,
    albedo: vec3<f32>,
    f0: vec3<f32>,
    roughness: f32,
    radiance: vec3<f32>,
    cc_intensity: f32,
    cc_roughness: f32,
) -> vec3<f32> {
    // Delegate to full version with zero anisotropy
    return eval_brdf_full(n, v, l, h, albedo, f0, roughness, radiance,
                          cc_intensity, cc_roughness,
                          vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0),
                          0.0, 0.0);
}

// T-MAT-4.3: Full BRDF evaluation with clear coat and anisotropy support.
// Combines:
//   - Cook-Torrance specular (isotropic or anisotropic based on anisotropy param)
//   - Optional clear coat layer (T-MAT-4.2)
//   - Energy-conserving diffuse
fn eval_brdf_full(
    n: vec3<f32>,
    v: vec3<f32>,
    l: vec3<f32>,
    h: vec3<f32>,
    albedo: vec3<f32>,
    f0: vec3<f32>,
    roughness: f32,
    radiance: vec3<f32>,
    cc_intensity: f32,
    cc_roughness: f32,
    tangent: vec3<f32>,
    bitangent: vec3<f32>,
    anisotropy: f32,
    anisotropy_rotation: f32,
) -> vec3<f32> {
    let ndotl = max(dot(n, l), 0.0);
    let ndotv = max(dot(n, v), 0.0);
    if ndotl <= 0.0 || ndotv <= 0.0 {
        return vec3<f32>(0.0);
    }

    // Fresnel (same for both paths)
    let f = fresnel_schlick(max(dot(h, v), 0.0), f0);

    var specular: vec3<f32>;

    // T-MAT-4.3: Use anisotropic GGX when anisotropy is non-zero
    if abs(anisotropy) > 0.001 {
        // Rotate tangent basis by anisotropy rotation
        let t = rotate_tangent_by_angle(tangent, bitangent, anisotropy_rotation);
        let b = rotate_bitangent_by_angle(tangent, bitangent, anisotropy_rotation);

        // Compute anisotropic alpha values
        // alpha_t = roughness * (1 + anisotropy), alpha_b = roughness * (1 - anisotropy)
        let a = roughness * roughness;
        let alpha_t = max(a * (1.0 + anisotropy), EPSILON);
        let alpha_b = max(a * (1.0 - anisotropy), EPSILON);

        // Compute dot products with tangent basis
        let NoH = max(dot(n, h), 0.0);
        let ToH = dot(t, h);
        let BoH = dot(b, h);
        let ToV = dot(t, v);
        let BoV = dot(b, v);
        let ToL = dot(t, l);
        let BoL = dot(b, l);

        // Anisotropic NDF and geometry
        let ndf = distribution_ggx_anisotropic(NoH, ToH, BoH, alpha_t, alpha_b);
        let g = geometry_smith_anisotropic(ndotv, ndotl, ToV, BoV, ToL, BoL, alpha_t, alpha_b);

        // Cook-Torrance: D * G * F (G already includes 1/4*NoV*NoL factor)
        specular = ndf * g * f;
    } else {
        // Fallback to isotropic GGX
        let ndf = distribution_ggx(n, h, roughness);
        let g = geometry_smith(n, v, l, roughness);
        specular = (ndf * g * f) / max(4.0 * ndotv * ndotl, EPSILON);
    }

    // Lambertian diffuse (energy-conserving).
    let kd = (1.0 - f) * (1.0 - albedo_to_metallic_f0(albedo).g); // approximate metallic factor

    // Base layer BRDF contribution
    var base_brdf = (kd * albedo / PI + specular) * radiance * ndotl;

    // Apply clear coat if enabled (T-MAT-4.2)
    if cc_intensity > EPSILON {
        // Evaluate clear coat layer
        let cc_result = eval_clear_coat(n, v, l, cc_intensity, cc_roughness);

        // Energy conservation: base layer is attenuated by clear coat Fresnel
        // cc_result.w = Fc * cc_intensity
        let base_attenuation = 1.0 - cc_result.w;

        // Combine layers: clear_coat + attenuated_base
        base_brdf = cc_result.xyz * radiance * ndotl + base_brdf * base_attenuation;
    }

    return base_brdf;
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

    // T-MAT-4.2: Extract clear coat parameters
    let cc_intensity = material.clear_coat;
    let cc_roughness = max(material.clear_coat_roughness, 0.04); // minimum roughness

    // T-MAT-4.3: Extract anisotropy parameters
    let anisotropy = material.anisotropy;
    let anisotropy_rotation = material.anisotropy_rotation;

    // Compute world-space normal (with optional normal mapping would go here).
    let n = normalize(input.world_normal);

    // T-MAT-4.3: Compute tangent and bitangent for anisotropic BRDF
    // world_tangent.xyz is the tangent, world_tangent.w is the bitangent sign
    let tangent = normalize(input.world_tangent.xyz);
    let bitangent = normalize(cross(n, tangent) * input.world_tangent.w);

    // Compute view direction.
    let v = normalize(camera.camera_position - input.world_position);

    // Compute F0 (surface reflectance at normal incidence).
    // Dielectric F0 is typically ~0.04; metals use albedo as F0.
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);

    var lo: vec3<f32> = vec3<f32>(0.0);

    // Accumulate directional lights.
    for (var i: u32 = 0u; i < light_counts.num_directional; i = i + 1u) {
        lo = lo + eval_directional_light(
            dir_lights[i], n, v, input.world_position, albedo, f0, roughness,
            cc_intensity, cc_roughness,
            tangent, bitangent, anisotropy, anisotropy_rotation
        );
    }

    // Accumulate point lights.
    for (var j: u32 = 0u; j < light_counts.num_point; j = j + 1u) {
        lo = lo + eval_point_light(
            point_lights[j], n, v, input.world_position, albedo, f0, roughness,
            cc_intensity, cc_roughness,
            tangent, bitangent, anisotropy, anisotropy_rotation
        );
    }

    // Accumulate spot lights.
    for (var k: u32 = 0u; k < light_counts.num_spot; k = k + 1u) {
        lo = lo + eval_spot_light(
            spot_lights[k], n, v, input.world_position, albedo, f0, roughness,
            cc_intensity, cc_roughness,
            tangent, bitangent, anisotropy, anisotropy_rotation
        );
    }

    // Ambient term.
    let ambient = camera.ambient_intensity * albedo * ao;

    // Final colour = emissive + ambient + direct lighting.
    let color = emissive + ambient + lo;

    return vec4<f32>(color, material.base_color.a);
}
