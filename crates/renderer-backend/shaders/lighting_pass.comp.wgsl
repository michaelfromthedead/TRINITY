// SPDX-License-Identifier: MIT
//
// lighting_pass.comp.wgsl - Deferred PBR Lighting Pass (T-LIT-3.6).
//
// Reads G-Buffer textures and evaluates Cook-Torrance BRDF for all lights
// using froxel-based light culling (from T-LIT-2.x). Outputs HDR lighting.
//
// Pipeline:
//   1. Sample G-Buffer (albedo, normal, roughness/metallic, depth)
//   2. Reconstruct world position from depth
//   3. Determine froxel index from screen position and depth
//   4. Loop over lights assigned to this froxel
//   5. Accumulate BRDF contributions
//   6. Write HDR output

// ============================================================================
// Constants
// ============================================================================

const PI: f32 = 3.14159265359;
const EPSILON: f32 = 0.00001;
const TILE_SIZE: u32 = 16u;
const MAX_LIGHTS_PER_FROXEL: u32 = 64u;

// ============================================================================
// Data Structures (matches light_culling.wgsl and pbr.frag.wgsl)
// ============================================================================

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

// Froxel structure from light culling pass
struct Froxel {
    light_offset: u32,  // Offset into light_index_list
    light_count: u32,   // Number of lights in this froxel
}

struct CameraUniforms {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    view_projection: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    inv_projection: mat4x4<f32>,
    inv_view_projection: mat4x4<f32>,
    camera_position: vec3<f32>,
    ambient_intensity: f32,
    near_plane: f32,
    far_plane: f32,
    _pad0: f32,
    _pad1: f32,
}

struct FroxelGridConfig {
    screen_width: u32,
    screen_height: u32,
    num_depth_slices: u32,
    tile_size: u32,
    depth_slice_scale: f32,
    depth_slice_bias: f32,
    near_plane: f32,
    far_plane: f32,
}

// CSM cascade data for directional light shadows
struct CascadeData {
    light_view_proj: mat4x4<f32>,
    split_depth: f32,
    shadow_map_index: u32,
    _pad0: f32,
    _pad1: f32,
}

// ============================================================================
// Bind Groups
// ============================================================================

// Group 0: G-Buffer textures
@group(0) @binding(0) var g_albedo: texture_2d<f32>;
@group(0) @binding(1) var g_normal: texture_2d<f32>;
@group(0) @binding(2) var g_roughness_metallic: texture_2d<f32>;
@group(0) @binding(3) var g_depth: texture_depth_2d;
@group(0) @binding(4) var g_sampler: sampler;

// Group 1: Light data and froxel grid
@group(1) @binding(0) var<storage, read> light_buffer_point: array<PointLight>;
@group(1) @binding(1) var<storage, read> light_buffer_spot: array<SpotLight>;
@group(1) @binding(2) var<storage, read> light_buffer_directional: array<DirectionalLight>;
@group(1) @binding(3) var<uniform> light_counts: LightCounts;
@group(1) @binding(4) var<storage, read> froxel_grid: array<Froxel>;
@group(1) @binding(5) var<storage, read> froxel_light_indices: array<u32>;
@group(1) @binding(6) var<uniform> camera: CameraUniforms;
@group(1) @binding(7) var<uniform> grid_config: FroxelGridConfig;

// Group 2: Output HDR texture
@group(2) @binding(0) var output_hdr: texture_storage_2d<rgba16float, write>;

// Group 3: Shadow resources (optional)
@group(3) @binding(0) var<uniform> cascade_data: array<CascadeData, 4>;
@group(3) @binding(1) var shadow_maps: texture_depth_2d_array;
@group(3) @binding(2) var shadow_sampler: sampler_comparison;
@group(3) @binding(3) var<uniform> shadow_params: vec4<f32>; // (bias, slope_bias, normal_bias, pcf_radius)
@group(3) @binding(4) var<uniform> cascade_blend_range: f32; // T-LIT-4.4: cascade blend range in world units

// Group 4: Contact shadow resources (T-LIT-8.2)
// Contact shadows enhance shadow quality by screen-space ray marching
@group(4) @binding(0) var contact_shadow_tex: texture_2d<f32>;
@group(4) @binding(1) var contact_shadow_sampler: sampler;
@group(4) @binding(2) var<uniform> contact_shadow_config: vec4<f32>; // (blend_mode, intensity, fallback, _pad)

// Contact shadow blend modes (T-LIT-8.2)
const BLEND_MODE_MIN: u32 = 0u;
const BLEND_MODE_MULTIPLY: u32 = 1u;
const BLEND_MODE_REPLACE: u32 = 2u;

// ============================================================================
// Cook-Torrance BRDF Functions
// ============================================================================

// Trowbridge-Reitz (GGX) normal distribution function.
// Models the microfacet distribution on the surface.
fn distribution_ggx(n: vec3<f32>, h: vec3<f32>, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let ndoth = max(dot(n, h), 0.0);
    let ndoth2 = ndoth * ndoth;

    let denom = ndoth2 * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom + EPSILON);
}

// Smith-GGX geometry function for a single direction.
// Accounts for self-shadowing of microfacets.
fn geometry_schlick_ggx(ndotv: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return ndotv / (ndotv * (1.0 - k) + k + EPSILON);
}

// Smith-GGX geometry function combining view and light directions.
fn geometry_smith(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, roughness: f32) -> f32 {
    let ndotv = max(dot(n, v), 0.0);
    let ndotl = max(dot(n, l), 0.0);
    return geometry_schlick_ggx(ndotv, roughness) * geometry_schlick_ggx(ndotl, roughness);
}

// Schlick Fresnel approximation.
// Models how reflectivity increases at grazing angles.
fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// Core BRDF evaluation for a single light.
// Returns the outgoing radiance contribution.
fn eval_brdf(
    n: vec3<f32>,      // Surface normal
    v: vec3<f32>,      // View direction (toward camera)
    l: vec3<f32>,      // Light direction (toward light)
    albedo: vec3<f32>, // Base color
    f0: vec3<f32>,     // Fresnel reflectance at normal incidence
    roughness: f32,    // Surface roughness [0,1]
    metallic: f32,     // Metallic factor [0,1]
    radiance: vec3<f32>, // Incoming light radiance
) -> vec3<f32> {
    let ndotl = max(dot(n, l), 0.0);
    let ndotv = max(dot(n, v), 0.0);

    // Early exit for backfacing or grazing angles
    if ndotl <= 0.0 || ndotv <= 0.0 {
        return vec3<f32>(0.0);
    }

    let h = normalize(v + l);

    // Specular BRDF (Cook-Torrance)
    let ndf = distribution_ggx(n, h, roughness);
    let g = geometry_smith(n, v, l, roughness);
    let f = fresnel_schlick(max(dot(h, v), 0.0), f0);

    let numerator = ndf * g * f;
    let denominator = 4.0 * ndotv * ndotl;
    let specular = numerator / max(denominator, EPSILON);

    // Diffuse BRDF (energy-conserving Lambertian)
    // kS = Fresnel represents reflected light, kD = 1 - kS is refracted/absorbed
    let kd = (vec3<f32>(1.0) - f) * (1.0 - metallic);
    let diffuse = kd * albedo / PI;

    return (diffuse + specular) * radiance * ndotl;
}

// ============================================================================
// World Position Reconstruction
// ============================================================================

// Reconstructs world position from depth buffer value and screen coordinates.
// Uses the inverse view-projection matrix.
fn reconstruct_world_position(
    screen_pos: vec2<f32>,  // Screen position in [0, width/height]
    depth: f32,              // Depth buffer value [0, 1]
    screen_size: vec2<f32>,
) -> vec3<f32> {
    // Convert screen position to NDC [-1, 1]
    let ndc_x = (screen_pos.x / screen_size.x) * 2.0 - 1.0;
    let ndc_y = 1.0 - (screen_pos.y / screen_size.y) * 2.0; // Flip Y for Vulkan/Metal
    let ndc_z = depth; // Depth is already in [0, 1] for reverse-Z

    let clip_pos = vec4<f32>(ndc_x, ndc_y, ndc_z, 1.0);
    let world_pos = camera.inv_view_projection * clip_pos;

    return world_pos.xyz / world_pos.w;
}

// ============================================================================
// Froxel Index Calculation
// ============================================================================

// Converts depth to exponential slice index for better near-field precision.
fn depth_to_slice(depth: f32) -> u32 {
    // Linear depth from reverse-Z depth buffer
    let linear_depth = grid_config.near_plane / (grid_config.far_plane - depth * (grid_config.far_plane - grid_config.near_plane));

    // Exponential slicing: slice = log2(depth) * scale + bias
    let slice_f = log2(max(linear_depth, grid_config.near_plane)) * grid_config.depth_slice_scale + grid_config.depth_slice_bias;
    return u32(clamp(slice_f, 0.0, f32(grid_config.num_depth_slices - 1u)));
}

// Computes the 3D froxel index from screen position and depth.
fn compute_froxel_index(screen_pos: vec2<u32>, depth: f32) -> u32 {
    let tile_x = screen_pos.x / grid_config.tile_size;
    let tile_y = screen_pos.y / grid_config.tile_size;
    let slice = depth_to_slice(depth);

    let num_tiles_x = (grid_config.screen_width + grid_config.tile_size - 1u) / grid_config.tile_size;
    let tile_index = tile_y * num_tiles_x + tile_x;

    return tile_index * grid_config.num_depth_slices + slice;
}

// ============================================================================
// Contact Shadow Sampling (T-LIT-8.2)
// ============================================================================

// Samples the contact shadow texture at the given pixel coordinate.
// Returns shadow factor: 0.0 = full shadow, 1.0 = no shadow.
fn sample_contact_shadow(pixel_coord: vec2<u32>) -> f32 {
    // Sample from the contact shadow texture
    // The texture is R8Unorm, so we only need the .r component
    return textureLoad(contact_shadow_tex, vec2<i32>(pixel_coord), 0).r;
}

// Blends contact shadow with shadow map result based on configured blend mode.
// Both shadow_map_result and contact_shadow_result are in [0, 1] range.
// - 0.0 = fully shadowed (no light)
// - 1.0 = fully lit (no shadow)
fn compute_combined_shadow(
    shadow_map_result: f32,
    contact_shadow_result: f32
) -> f32 {
    let blend_mode = u32(contact_shadow_config.x);
    let intensity = contact_shadow_config.y;

    // Apply intensity to contact shadow (lerp toward 1.0 = no shadow)
    let adjusted_contact = mix(1.0, contact_shadow_result, intensity);

    // Blend based on mode
    var result: f32;
    if blend_mode == BLEND_MODE_MULTIPLY {
        // Multiply: shadow_map * contact
        // Produces softer shadows, darker where both see shadow
        result = shadow_map_result * adjusted_contact;
    } else if blend_mode == BLEND_MODE_REPLACE {
        // Replace: ignore shadow map, use contact only
        // For debugging/visualization
        result = adjusted_contact;
    } else {
        // Default: Min blend
        // min(shadow_map, contact) - most conservative
        // Pixel is only lit if BOTH methods say it's lit
        result = min(shadow_map_result, adjusted_contact);
    }

    return result;
}

// ============================================================================
// Shadow Sampling
// ============================================================================

// T-LIT-4.4: Cascade selection result with blend information for soft transitions.
struct CascadeSelection {
    primary_idx: u32,      // Primary cascade index
    secondary_idx: u32,    // Secondary cascade for blending (if in blend zone)
    blend_factor: f32,     // 0.0 = use primary only, 1.0 = use secondary only
}

// Selects CSM cascade based on view-space depth.
fn select_cascade(view_depth: f32) -> u32 {
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        if view_depth < cascade_data[i].split_depth {
            return i;
        }
    }
    return 3u;
}

// T-LIT-4.4: Selects cascade with blend information for soft transitions.
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

// T-LIT-4.4: Samples shadow from a single cascade with PCF.
fn sample_cascade_shadow(
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    light_dir: vec3<f32>,
    cascade_idx: u32
) -> f32 {
    let c = cascade_data[cascade_idx];

    // Transform to light clip space
    let light_clip = c.light_view_proj * vec4<f32>(world_pos, 1.0);
    if light_clip.w <= 0.0 {
        return 1.0; // Behind light, not shadowed
    }

    var uv = light_clip.xyz / light_clip.w;
    if any(uv.xyz < vec3<f32>(-1.0)) || any(uv.xyz > vec3<f32>(1.0)) {
        return 1.0; // Outside cascade frustum
    }

    // Apply depth biases
    let cos_theta = max(dot(normal, light_dir), 0.0);
    let slope_bias = shadow_params.y * tan(acos(cos_theta));
    let bias = shadow_params.x + slope_bias;
    let normal_bias = shadow_params.z * (1.0 - cos_theta);
    uv.z = uv.z - bias - normal_bias;

    // Transform [-1,1] to [0,1] UV
    let shadow_uv = vec2<f32>(uv.xy * 0.5 + 0.5);

    // PCF with configurable kernel
    let pcf_r = u32(shadow_params.w);
    let texel_size = 1.0 / 2048.0; // Shadow map resolution

    var shadow: f32 = 0.0;
    let kernel_size = 2u * pcf_r + 1u;
    for (var x: u32 = 0u; x < kernel_size; x = x + 1u) {
        for (var y: u32 = 0u; y < kernel_size; y = y + 1u) {
            let offset = vec2<f32>(f32(x) - f32(pcf_r), f32(y) - f32(pcf_r)) * texel_size;
            shadow = shadow + textureSampleCompare(
                shadow_maps, shadow_sampler, shadow_uv + offset, i32(cascade_idx), uv.z
            );
        }
    }

    return shadow / f32(kernel_size * kernel_size);
}

// Computes shadow factor using PCF (Percentage Closer Filtering).
// T-LIT-4.4: Now supports soft cascade transitions via cascade_blend_range.
fn compute_shadow_factor(world_pos: vec3<f32>, normal: vec3<f32>, light_dir: vec3<f32>) -> f32 {
    // Compute view-space depth for cascade selection
    let view_pos = camera.view * vec4<f32>(world_pos, 1.0);
    let view_depth = abs(view_pos.z);

    // Select cascade with blend information
    let selection = select_cascade_blended(view_depth, cascade_blend_range);

    // Sample primary cascade
    let primary_shadow = sample_cascade_shadow(world_pos, normal, light_dir, selection.primary_idx);

    // If not in blend zone, return primary result only (fast path)
    if selection.blend_factor <= 0.0 {
        return primary_shadow;
    }

    // Sample secondary cascade and blend
    let secondary_shadow = sample_cascade_shadow(world_pos, normal, light_dir, selection.secondary_idx);

    // Linear interpolation between cascades for smooth transition
    return mix(primary_shadow, secondary_shadow, selection.blend_factor);
}

// ============================================================================
// Light Evaluation Functions
// ============================================================================

// Evaluates contribution from a directional light (sun/moon).
// pixel_coord is used for contact shadow lookup (T-LIT-8.2).
fn eval_directional_light(
    light: DirectionalLight,
    n: vec3<f32>,
    v: vec3<f32>,
    world_pos: vec3<f32>,
    albedo: vec3<f32>,
    f0: vec3<f32>,
    roughness: f32,
    metallic: f32,
    pixel_coord: vec2<u32>,
) -> vec3<f32> {
    let l = normalize(-light.direction);
    let radiance = light.color * light.intensity;

    // Compute shadow factor from shadow map
    let shadow_map_factor = compute_shadow_factor(world_pos, n, l);

    // Sample contact shadow (T-LIT-8.2)
    let contact_shadow_factor = sample_contact_shadow(pixel_coord);

    // Combine shadow map and contact shadow results
    let combined_shadow = compute_combined_shadow(shadow_map_factor, contact_shadow_factor);

    return eval_brdf(n, v, l, albedo, f0, roughness, metallic, radiance * combined_shadow);
}

// Evaluates contribution from a point light.
// Uses smooth inverse-square falloff with radius cutoff.
fn eval_point_light(
    light: PointLight,
    n: vec3<f32>,
    v: vec3<f32>,
    world_pos: vec3<f32>,
    albedo: vec3<f32>,
    f0: vec3<f32>,
    roughness: f32,
    metallic: f32,
) -> vec3<f32> {
    let to_light = light.position - world_pos;
    let dist = length(to_light);

    // Early exit if outside light radius
    if dist > light.radius {
        return vec3<f32>(0.0);
    }

    let l = to_light / dist;

    // Smooth attenuation: (1 - (d/r)^2)^2
    // Reaches zero exactly at radius, avoiding hard cutoffs
    let d_over_r = dist / light.radius;
    let d_over_r_sq = d_over_r * d_over_r;
    let attenuation = pow(clamp(1.0 - d_over_r_sq, 0.0, 1.0), 2.0);

    let radiance = light.color * light.intensity * attenuation;

    return eval_brdf(n, v, l, albedo, f0, roughness, metallic, radiance);
}

// Evaluates contribution from a spot light.
// Combines distance and angular attenuation.
fn eval_spot_light(
    light: SpotLight,
    n: vec3<f32>,
    v: vec3<f32>,
    world_pos: vec3<f32>,
    albedo: vec3<f32>,
    f0: vec3<f32>,
    roughness: f32,
    metallic: f32,
) -> vec3<f32> {
    let to_light = light.position - world_pos;
    let dist = length(to_light);

    // Early exit if outside light radius
    if dist > light.radius {
        return vec3<f32>(0.0);
    }

    let l = to_light / dist;

    // Distance attenuation (same as point light)
    let d_over_r = dist / light.radius;
    let d_over_r_sq = d_over_r * d_over_r;
    let dist_att = pow(clamp(1.0 - d_over_r_sq, 0.0, 1.0), 2.0);

    // Angular attenuation (smoothstep between inner and outer cone)
    let spot_dir = normalize(-light.direction);
    let cos_theta = dot(-l, spot_dir);
    let cone_att = smoothstep(light.cos_outer_angle, light.cos_inner_angle, cos_theta);

    // Early exit if outside cone
    if cone_att <= 0.0 {
        return vec3<f32>(0.0);
    }

    let radiance = light.color * light.intensity * dist_att * cone_att;

    return eval_brdf(n, v, l, albedo, f0, roughness, metallic, radiance);
}

// ============================================================================
// G-Buffer Decoding
// ============================================================================

// Decodes world-space normal from G-Buffer.
// Assumes normals are stored as (n * 0.5 + 0.5) in RGB.
fn decode_normal(encoded: vec3<f32>) -> vec3<f32> {
    return normalize(encoded * 2.0 - 1.0);
}

// ============================================================================
// Main Compute Entry Point
// ============================================================================

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let screen_size = vec2<f32>(f32(grid_config.screen_width), f32(grid_config.screen_height));

    // Step 1: Bounds check - ensure we're within screen dimensions
    if gid.x >= grid_config.screen_width || gid.y >= grid_config.screen_height {
        return;
    }

    let pixel_coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let screen_pos = vec2<f32>(f32(gid.x), f32(gid.y));

    // Step 2: Sample G-Buffer textures
    let albedo_sample = textureLoad(g_albedo, pixel_coord, 0);
    let normal_sample = textureLoad(g_normal, pixel_coord, 0);
    let rm_sample = textureLoad(g_roughness_metallic, pixel_coord, 0);
    let depth = textureLoad(g_depth, pixel_coord, 0);

    // Early exit for sky pixels (depth = 0 in reverse-Z or no geometry)
    if depth <= 0.0 || depth >= 1.0 {
        // Write ambient sky color or background
        textureStore(output_hdr, pixel_coord, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }

    // Decode G-Buffer data
    let albedo = albedo_sample.rgb;
    let normal = decode_normal(normal_sample.rgb);
    let roughness = max(rm_sample.r, 0.04); // Minimum roughness to avoid numerical issues
    let metallic = rm_sample.g;
    let ao = rm_sample.b; // Ambient occlusion in blue channel (if present)

    // Step 3: Reconstruct world position from depth
    let world_pos = reconstruct_world_position(screen_pos + 0.5, depth, screen_size);

    // Compute view direction (from surface to camera)
    let v = normalize(camera.camera_position - world_pos);

    // Compute F0 (Fresnel reflectance at normal incidence)
    // Dielectric surfaces have F0 ~= 0.04, metals use albedo as F0
    let dielectric_f0 = vec3<f32>(0.04);
    let f0 = mix(dielectric_f0, albedo, metallic);

    // Step 4: Determine froxel index for light lookup
    let froxel_idx = compute_froxel_index(vec2<u32>(gid.x, gid.y), depth);
    let froxel = froxel_grid[froxel_idx];

    // Initialize light accumulator
    var lo = vec3<f32>(0.0);

    // Step 5a: Evaluate directional lights (not in froxel grid, always visible)
    // Pass pixel coordinate for contact shadow lookup (T-LIT-8.2)
    let pixel_uv = vec2<u32>(gid.x, gid.y);
    for (var i: u32 = 0u; i < light_counts.num_directional; i = i + 1u) {
        lo = lo + eval_directional_light(
            light_buffer_directional[i],
            normal, v, world_pos, albedo, f0, roughness, metallic,
            pixel_uv
        );
    }

    // Step 5b: Loop over lights assigned to this froxel
    let light_end = min(froxel.light_offset + froxel.light_count, froxel.light_offset + MAX_LIGHTS_PER_FROXEL);
    for (var i: u32 = froxel.light_offset; i < light_end; i = i + 1u) {
        let packed_index = froxel_light_indices[i];

        // Unpack light type and index from packed value
        // High 2 bits: light type (0=point, 1=spot)
        // Low 30 bits: light index
        let light_type = packed_index >> 30u;
        let light_index = packed_index & 0x3FFFFFFFu;

        if light_type == 0u {
            // Point light
            if light_index < light_counts.num_point {
                lo = lo + eval_point_light(
                    light_buffer_point[light_index],
                    normal, v, world_pos, albedo, f0, roughness, metallic
                );
            }
        } else if light_type == 1u {
            // Spot light
            if light_index < light_counts.num_spot {
                lo = lo + eval_spot_light(
                    light_buffer_spot[light_index],
                    normal, v, world_pos, albedo, f0, roughness, metallic
                );
            }
        }
    }

    // Step 6: Add ambient lighting
    let ambient = camera.ambient_intensity * albedo * ao;

    // Final HDR color
    let hdr_color = lo + ambient;

    // Step 7: Write HDR output
    textureStore(output_hdr, pixel_coord, vec4<f32>(hdr_color, 1.0));
}
