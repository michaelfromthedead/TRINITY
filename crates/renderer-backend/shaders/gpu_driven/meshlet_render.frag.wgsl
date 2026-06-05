// SPDX-License-Identifier: MIT
//
// meshlet_render.frag.wgsl - Meshlet Rendering Fragment Shader (T-GPU-4.5)
//
// Fragment shading for meshlet-rendered geometry. Supports multiple output modes:
// 1. Visibility buffer output (pack instance_id + primitive_id)
// 2. G-buffer output (albedo, normal, metallic, roughness)
// 3. Shadow pass (depth only, no color output)
//
// Material data is fetched from the bindless material table using the
// material_id passed from the vertex shader.
//
// Performance Target: <0.2ms for 100K visible meshlets

// ============================================================================
// Constants
// ============================================================================

const PI: f32 = 3.14159265359;
const EPSILON: f32 = 0.00001;

/// Invalid visibility ID (used for discarded fragments).
const INVALID_VISIBILITY_ID: u32 = 0xFFFFFFFFu;

// ============================================================================
// Material Flags
// ============================================================================

/// Material has albedo texture.
const MAT_FLAG_HAS_ALBEDO: u32 = 1u;
/// Material has normal map texture.
const MAT_FLAG_HAS_NORMAL: u32 = 2u;
/// Material has metallic-roughness texture.
const MAT_FLAG_HAS_METALLIC_ROUGHNESS: u32 = 4u;
/// Material has emissive texture.
const MAT_FLAG_HAS_EMISSIVE: u32 = 8u;
/// Material uses alpha masking.
const MAT_FLAG_ALPHA_MASK: u32 = 16u;
/// Material uses alpha blending.
const MAT_FLAG_ALPHA_BLEND: u32 = 32u;
/// Material is double-sided.
const MAT_FLAG_DOUBLE_SIDED: u32 = 64u;

// ============================================================================
// Structures
// ============================================================================

/// Material table entry (matches material_table.wgsl layout, 80 bytes).
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
    clear_coat: f32,
    clear_coat_roughness: f32,
    anisotropy: f32,
    anisotropy_rotation: f32,
    _pad0: f32,
    _pad1: f32,
}

/// Meshlet render parameters.
struct MeshletRenderParams {
    num_meshlets: u32,
    flags: u32,
    alpha_cutoff: f32,
    _pad: u32,
    viewport_size: vec2<f32>,
    near_plane: f32,
    far_plane: f32,
}

/// Camera uniforms.
struct CameraUniforms {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    view_projection: mat4x4<f32>,
    camera_position: vec3<f32>,
    _pad: f32,
}

// ============================================================================
// Bind Groups
// ============================================================================

// Group 0: Camera and render parameters
@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> params: MeshletRenderParams;

// Group 1: Material table (bindless)
@group(1) @binding(0) var<storage, read> material_table: array<MaterialTableEntry>;

// Group 3: Textures
@group(3) @binding(0) var texture_sampler: sampler;
@group(3) @binding(1) var texture_array: texture_2d_array<f32>;

// ============================================================================
// Fragment Input (from vertex shader)
// ============================================================================

struct FragmentInput {
    @builtin(position) frag_coord: vec4<f32>,
    @builtin(front_facing) front_facing: bool,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) world_tangent: vec4<f32>,
    @location(3) texcoord: vec2<f32>,
    @location(4) @interpolate(flat) material_id: u32,
    @location(5) @interpolate(flat) instance_id: u32,
    @location(6) @interpolate(flat) primitive_id: u32,
}

/// Visibility pass input (reduced).
struct VisibilityInput {
    @builtin(position) frag_coord: vec4<f32>,
    @location(0) @interpolate(flat) visibility_id: u32,
}

// ============================================================================
// Fragment Output
// ============================================================================

/// G-buffer output (deferred rendering).
struct GBufferOutput {
    /// Albedo RGB + alpha.
    @location(0) albedo: vec4<f32>,
    /// World normal (encoded).
    @location(1) normal: vec4<f32>,
    /// Metallic (R), Roughness (G), AO (B), Material flags (A).
    @location(2) material: vec4<f32>,
    /// Emissive RGB + unused.
    @location(3) emissive: vec4<f32>,
}

/// Visibility buffer output.
struct VisibilityOutput {
    /// Packed visibility ID: instance_id (20 bits) | primitive_id (12 bits).
    @location(0) visibility_id: u32,
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Encodes a normalized vector to octahedral representation.
fn encode_octahedral(n: vec3<f32>) -> vec2<f32> {
    let sum = abs(n.x) + abs(n.y) + abs(n.z);
    var oct = n.xy / sum;

    if (n.z < 0.0) {
        let sign_x = select(-1.0, 1.0, oct.x >= 0.0);
        let sign_y = select(-1.0, 1.0, oct.y >= 0.0);
        oct = (1.0 - abs(oct.yx)) * vec2<f32>(sign_x, sign_y);
    }

    return oct * 0.5 + 0.5; // Map to [0, 1]
}

/// Decodes octahedral representation to normalized vector.
fn decode_octahedral(oct_encoded: vec2<f32>) -> vec3<f32> {
    let oct = oct_encoded * 2.0 - 1.0;
    var n = vec3<f32>(oct.x, oct.y, 1.0 - abs(oct.x) - abs(oct.y));

    if (n.z < 0.0) {
        let sign_x = select(-1.0, 1.0, n.x >= 0.0);
        let sign_y = select(-1.0, 1.0, n.y >= 0.0);
        n = vec3<f32>(
            (1.0 - abs(oct.y)) * sign_x,
            (1.0 - abs(oct.x)) * sign_y,
            n.z
        );
    }

    return normalize(n);
}

/// Samples normal map and applies tangent-space transform.
fn sample_normal_map(
    uv: vec2<f32>,
    texture_id: u32,
    normal_scale: f32,
    world_normal: vec3<f32>,
    world_tangent: vec4<f32>
) -> vec3<f32> {
    // Sample normal map
    let normal_sample = textureSample(texture_array, texture_sampler, uv, texture_id);
    let raw_normal = normal_sample.xyz * 2.0 - 1.0;
    var tangent_normal = vec3<f32>(raw_normal.x * normal_scale, raw_normal.y * normal_scale, raw_normal.z);

    // Build TBN matrix
    let n = normalize(world_normal);
    let t = normalize(world_tangent.xyz);
    let b = cross(n, t) * world_tangent.w;

    // Transform to world space
    let tbn = mat3x3<f32>(t, b, n);
    return normalize(tbn * tangent_normal);
}

/// Pack visibility ID from instance and primitive.
fn pack_visibility_id(instance_id: u32, primitive_id: u32) -> u32 {
    return ((instance_id & 0xFFFFFu) << 12u) | (primitive_id & 0xFFFu);
}

/// Check if a material flag is set.
fn has_flag(flags: u32, flag: u32) -> bool {
    return (flags & flag) != 0u;
}

// ============================================================================
// Main Fragment Shader: G-Buffer Output (Deferred)
// ============================================================================

@fragment
fn fs_gbuffer(input: FragmentInput) -> GBufferOutput {
    var output: GBufferOutput;

    // Fetch material data
    let mat = material_table[input.material_id];

    // Start with base color
    var albedo = mat.base_color;

    // Sample albedo texture if present
    if (has_flag(mat.flags, MAT_FLAG_HAS_ALBEDO)) {
        let tex_albedo = textureSample(
            texture_array,
            texture_sampler,
            input.texcoord,
            mat.albedo_texture_id
        );
        albedo *= tex_albedo;
    }

    // Alpha test
    if (has_flag(mat.flags, MAT_FLAG_ALPHA_MASK)) {
        if (albedo.a < mat.alpha_cutoff) {
            discard;
        }
    }

    // Normal handling
    var world_normal = normalize(input.world_normal);

    // Flip normal for back faces if double-sided
    if (!input.front_facing && has_flag(mat.flags, MAT_FLAG_DOUBLE_SIDED)) {
        world_normal = -world_normal;
    }

    // Apply normal map if present
    if (has_flag(mat.flags, MAT_FLAG_HAS_NORMAL)) {
        world_normal = sample_normal_map(
            input.texcoord,
            mat.normal_texture_id,
            mat.normal_scale,
            world_normal,
            input.world_tangent
        );
    }

    // Metallic-roughness
    var metallic = mat.metallic;
    var roughness = mat.roughness;
    var ao = mat.occlusion;

    if (has_flag(mat.flags, MAT_FLAG_HAS_METALLIC_ROUGHNESS)) {
        let mr_sample = textureSample(
            texture_array,
            texture_sampler,
            input.texcoord,
            mat.metallic_roughness_tex_id
        );
        // glTF convention: B = metallic, G = roughness
        metallic *= mr_sample.b;
        roughness *= mr_sample.g;
        ao *= mr_sample.r; // AO often in R channel
    }

    // Emissive
    var emissive = mat.emissive.rgb;
    if (has_flag(mat.flags, MAT_FLAG_HAS_EMISSIVE)) {
        let emissive_sample = textureSample(
            texture_array,
            texture_sampler,
            input.texcoord,
            mat.emissive_texture_id
        );
        emissive *= emissive_sample.rgb;
    }

    // Write G-buffer outputs
    output.albedo = albedo;
    output.normal = vec4<f32>(encode_octahedral(world_normal), 0.0, 1.0);
    output.material = vec4<f32>(metallic, roughness, ao, f32(input.material_id) / 65535.0);
    output.emissive = vec4<f32>(emissive, 1.0);

    return output;
}

// ============================================================================
// Visibility Buffer Fragment Shader
// ============================================================================

@fragment
fn fs_visibility(input: VisibilityInput) -> VisibilityOutput {
    var output: VisibilityOutput;
    output.visibility_id = input.visibility_id;
    return output;
}

// ============================================================================
// Visibility Buffer with Alpha Test Fragment Shader
// ============================================================================

struct VisibilityAlphaInput {
    @builtin(position) frag_coord: vec4<f32>,
    @location(0) @interpolate(flat) visibility_id: u32,
    @location(1) texcoord: vec2<f32>,
    @location(2) @interpolate(flat) material_id: u32,
}

@fragment
fn fs_visibility_alpha_test(input: VisibilityAlphaInput) -> VisibilityOutput {
    var output: VisibilityOutput;

    // Fetch material for alpha test
    let mat = material_table[input.material_id];

    // Sample albedo for alpha value
    var alpha = mat.base_color.a;
    if (has_flag(mat.flags, MAT_FLAG_HAS_ALBEDO)) {
        let tex_albedo = textureSample(
            texture_array,
            texture_sampler,
            input.texcoord,
            mat.albedo_texture_id
        );
        alpha *= tex_albedo.a;
    }

    // Alpha test
    if (alpha < mat.alpha_cutoff) {
        discard;
    }

    output.visibility_id = input.visibility_id;
    return output;
}

// ============================================================================
// Shadow Pass Fragment Shader (Depth only, optional alpha test)
// ============================================================================

// Note: Shadow pass typically uses depth-only rendering with no color output.
// This entry point is for alpha-tested shadows.

struct ShadowAlphaInput {
    @builtin(position) frag_coord: vec4<f32>,
    @location(0) texcoord: vec2<f32>,
    @location(1) @interpolate(flat) material_id: u32,
}

@fragment
fn fs_shadow_alpha_test(input: ShadowAlphaInput) {
    let mat = material_table[input.material_id];

    // Only do alpha test if material requires it
    if (has_flag(mat.flags, MAT_FLAG_ALPHA_MASK)) {
        var alpha = mat.base_color.a;
        if (has_flag(mat.flags, MAT_FLAG_HAS_ALBEDO)) {
            let tex_albedo = textureSample(
                texture_array,
                texture_sampler,
                input.texcoord,
                mat.albedo_texture_id
            );
            alpha *= tex_albedo.a;
        }

        if (alpha < mat.alpha_cutoff) {
            discard;
        }
    }

    // No color output - depth only
}
