// SPDX-License-Identifier: MIT
//
// decal.frag.wgsl - Deferred Decal Fragment Shader (T-GPU-6.4)
//
// Projects decal textures onto geometry using deferred rendering.
// Reconstructs world position from depth buffer, transforms to decal space,
// and applies texture with configurable blend modes.

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const EPSILON: f32 = 0.0001;
const PI: f32 = 3.14159265359;

/// Blend modes for decal application.
const BLEND_MODE_ALBEDO: u32 = 0u;
const BLEND_MODE_NORMAL: u32 = 1u;
const BLEND_MODE_BOTH: u32 = 2u;
const BLEND_MODE_EMISSIVE: u32 = 3u;

// ---------------------------------------------------------------------------
// Uniform Structures (must match vertex shader)
// ---------------------------------------------------------------------------

struct DecalParams {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_position: vec3<f32>,
    _pad: f32,
}

struct DecalInstance {
    world_to_decal: mat4x4<f32>,
    decal_to_world: mat4x4<f32>,
    color: vec4<f32>,
    atlas_rect: vec4<f32>,
    blend_mode: u32,
    normal_strength: f32,
    fade: f32,
    _pad: f32,
}

// ---------------------------------------------------------------------------
// Bind Groups
// ---------------------------------------------------------------------------

// Group 0: Decal parameters and instance buffer (shared with vertex shader)
@group(0) @binding(0) var<uniform> params: DecalParams;
@group(0) @binding(1) var<storage, read> decals: array<DecalInstance>;

// Group 1: GBuffer and decal textures
@group(1) @binding(0) var depth_texture: texture_depth_2d;
@group(1) @binding(1) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(2) var decal_atlas: texture_2d<f32>;
@group(1) @binding(3) var decal_normal_atlas: texture_2d<f32>;
@group(1) @binding(4) var decal_sampler: sampler;

// ---------------------------------------------------------------------------
// Fragment Input
// ---------------------------------------------------------------------------

struct FragmentInput {
    @builtin(position) frag_coord: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) @interpolate(flat) instance_index: u32,
}

// ---------------------------------------------------------------------------
// Output Targets
// ---------------------------------------------------------------------------

struct FragmentOutput {
    /// Modified albedo output (blended with existing GBuffer albedo).
    @location(0) albedo: vec4<f32>,
    /// Modified normal output (blended with existing GBuffer normal).
    @location(1) normal: vec4<f32>,
}

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

/// Reconstructs world position from depth buffer sample.
/// Uses inverse view-projection to transform from NDC to world space.
fn reconstruct_world_position(frag_coord: vec2<f32>, depth: f32) -> vec3<f32> {
    // Get screen dimensions from depth texture
    let dims = vec2<f32>(textureDimensions(depth_texture));

    // Convert fragment coordinates to NDC [-1, 1]
    let ndc_xy = (frag_coord / dims) * 2.0 - 1.0;
    // Note: Y is flipped in Vulkan/WGPU conventions
    let ndc = vec4<f32>(ndc_xy.x, -ndc_xy.y, depth, 1.0);

    // Transform to world space
    let world_h = params.inv_view_proj * ndc;
    return world_h.xyz / world_h.w;
}

/// Checks if a point in decal space is inside the unit cube [-0.5, 0.5].
fn is_inside_decal_volume(decal_pos: vec3<f32>) -> bool {
    return all(decal_pos >= vec3<f32>(-0.5)) && all(decal_pos <= vec3<f32>(0.5));
}

/// Converts decal-space position to UV coordinates.
/// Decal XY [-0.5, 0.5] maps to UV [0, 1].
fn decal_pos_to_uv(decal_pos: vec3<f32>) -> vec2<f32> {
    return decal_pos.xy + 0.5;
}

/// Applies atlas rectangle transformation to UV coordinates.
fn apply_atlas_rect(uv: vec2<f32>, rect: vec4<f32>) -> vec2<f32> {
    return rect.xy + uv * rect.zw;
}

/// Decode normal from [0,1] texture space to [-1,1] world space.
fn decode_normal(encoded: vec3<f32>) -> vec3<f32> {
    return encoded * 2.0 - 1.0;
}

/// Encode normal from [-1,1] world space to [0,1] texture space.
fn encode_normal(n: vec3<f32>) -> vec3<f32> {
    return n * 0.5 + 0.5;
}

/// Builds a tangent-to-world matrix from the decal orientation.
/// Z-axis of decal is projection direction, XY are the decal plane.
fn build_decal_tbn(decal: DecalInstance) -> mat3x3<f32> {
    // Extract tangent, bitangent, normal from decal_to_world
    // Columns 0, 1, 2 are the local X, Y, Z axes in world space
    let tangent = normalize(decal.decal_to_world[0].xyz);
    let bitangent = normalize(decal.decal_to_world[1].xyz);
    let normal = normalize(decal.decal_to_world[2].xyz);

    return mat3x3<f32>(tangent, bitangent, normal);
}

/// Blends decal normal with surface normal using reoriented normal mapping.
fn blend_normals_rnm(surface_normal: vec3<f32>, decal_normal: vec3<f32>, strength: f32) -> vec3<f32> {
    // Reoriented Normal Mapping blend
    // Based on "Reoriented Normal Mapping" by Colin Barr-Brisebois & Stephen Hill
    let t = surface_normal + vec3<f32>(0.0, 0.0, 1.0);
    let u = decal_normal * vec3<f32>(-1.0, -1.0, 1.0);
    let blended = t * dot(t, u) - u * t.z;

    // Interpolate between surface normal and blended based on strength
    return normalize(mix(surface_normal, normalize(blended), strength));
}

// ---------------------------------------------------------------------------
// Fragment Shader Entry Point
// ---------------------------------------------------------------------------

@fragment
fn fs_decal(input: FragmentInput) -> FragmentOutput {
    var output: FragmentOutput;

    // Fetch decal instance data
    let decal = decals[input.instance_index];

    // Early discard if fade is zero
    if decal.fade < EPSILON {
        discard;
    }

    // Sample depth at current fragment
    let frag_coord_2d = input.frag_coord.xy;
    let depth = textureLoad(depth_texture, vec2<i32>(frag_coord_2d), 0);

    // Discard if no geometry (sky/background)
    if depth >= 1.0 {
        discard;
    }

    // Reconstruct world position from depth
    let world_pos = reconstruct_world_position(frag_coord_2d, depth);

    // Transform to decal local space
    let decal_pos_h = decal.world_to_decal * vec4<f32>(world_pos, 1.0);
    let decal_pos = decal_pos_h.xyz / decal_pos_h.w;

    // Reject fragments outside the decal volume
    if !is_inside_decal_volume(decal_pos) {
        discard;
    }

    // Compute UV coordinates from decal position
    let base_uv = decal_pos_to_uv(decal_pos);
    let atlas_uv = apply_atlas_rect(base_uv, decal.atlas_rect);

    // Sample decal color texture
    let decal_color = textureSample(decal_atlas, decal_sampler, atlas_uv);

    // Discard if decal texture is transparent
    if decal_color.a < EPSILON {
        discard;
    }

    // Apply tint and fade
    let final_alpha = decal_color.a * decal.color.a * decal.fade;
    let final_color = decal_color.rgb * decal.color.rgb;

    // Handle different blend modes
    switch decal.blend_mode {
        case BLEND_MODE_ALBEDO: {
            output.albedo = vec4<f32>(final_color, final_alpha);
            output.normal = vec4<f32>(0.0);
        }
        case BLEND_MODE_NORMAL: {
            // Sample normal map
            let normal_sample = textureSample(decal_normal_atlas, decal_sampler, atlas_uv);
            let decal_normal = decode_normal(normal_sample.rgb);

            // Transform decal normal to world space
            let tbn = build_decal_tbn(decal);
            let world_decal_normal = normalize(tbn * decal_normal);

            output.albedo = vec4<f32>(0.0);
            output.normal = vec4<f32>(encode_normal(world_decal_normal), final_alpha * decal.normal_strength);
        }
        case BLEND_MODE_BOTH: {
            // Sample and apply both albedo and normal
            let normal_sample = textureSample(decal_normal_atlas, decal_sampler, atlas_uv);
            let decal_normal = decode_normal(normal_sample.rgb);

            let tbn = build_decal_tbn(decal);
            let world_decal_normal = normalize(tbn * decal_normal);

            output.albedo = vec4<f32>(final_color, final_alpha);
            output.normal = vec4<f32>(encode_normal(world_decal_normal), final_alpha * decal.normal_strength);
        }
        case BLEND_MODE_EMISSIVE: {
            // Emissive decals add light, stored in alpha for HDR
            output.albedo = vec4<f32>(final_color * final_alpha, 0.0);
            output.normal = vec4<f32>(0.0);
        }
        default: {
            output.albedo = vec4<f32>(final_color, final_alpha);
            output.normal = vec4<f32>(0.0);
        }
    }

    return output;
}
