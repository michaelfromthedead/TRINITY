// Bindless Texture Array for TRINITY Material System (T-MAT-5.7)
//
// This module provides WGSL declarations and helper functions for sampling
// from the bindless texture array. Textures are indexed via the material
// table's texture_index fields rather than individual per-draw bindings.
//
// Usage:
//   1. Include this file in your shader
//   2. Access textures via sample_bindless() using indices from MaterialTableEntry
//
// Example:
//   let albedo = sample_bindless(material.albedo_texture_id, 0u, uv);
//
// Note: The binding_array feature requires WebGPU support. Check adapter
// limits (maxTexturesPerShaderStage) before use. When unsupported, fall back
// to traditional per-draw texture bindings.

// =============================================================================
// Constants
// =============================================================================

/// Maximum number of textures in the bindless array.
/// Matches MAX_BINDLESS_TEXTURES in Rust/Python (4096).
const MAX_BINDLESS_TEXTURES: u32 = 4096u;

/// Sentinel value indicating no texture bound (u32::MAX).
const INVALID_TEXTURE_INDEX: u32 = 0xFFFFFFFFu;

/// Default sampler index for standard linear filtering.
const DEFAULT_SAMPLER_INDEX: u32 = 0u;

/// Error color returned when sampling invalid texture (magenta).
const ERROR_COLOR: vec4<f32> = vec4<f32>(1.0, 0.0, 1.0, 1.0);

/// Transparent black for optional texture sampling.
const TRANSPARENT_BLACK: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);

/// Default normal map value (flat normal pointing up).
const FLAT_NORMAL: vec4<f32> = vec4<f32>(0.5, 0.5, 1.0, 1.0);

// =============================================================================
// Bindless Resource Declarations
// =============================================================================

// Note: These declarations use binding_array which requires WebGPU support.
// The array sizes should match the limits set during pipeline creation.
// Uncomment the appropriate declarations based on your binding layout.

// Group 1 is typically used for material textures:
// @group(1) @binding(0) var textures: binding_array<texture_2d<f32>, 4096>;
// @group(1) @binding(1) var samplers: binding_array<sampler, 16>;

// Alternative: storage textures for compute access
// @group(1) @binding(2) var storage_textures: binding_array<texture_storage_2d<rgba8unorm, read_write>, 256>;

// =============================================================================
// Texture Table Entry (matches Rust TextureTableEntry)
// =============================================================================

/// Metadata for a single texture in the bindless array.
/// This struct mirrors the Rust TextureTableEntry layout (24 bytes).
struct TextureTableEntry {
    /// Width of the texture in pixels.
    width: u32,
    /// Height of the texture in pixels.
    height: u32,
    /// Number of mip levels.
    mip_levels: u32,
    /// Packed texture format identifier.
    format: u32,
    /// Number of array layers (1 for 2D textures).
    layer_count: u32,
    /// Flags field (bit 0 = valid).
    flags: u32,
}

/// Check if a texture table entry is valid.
fn texture_entry_is_valid(entry: TextureTableEntry) -> bool {
    return (entry.flags & 1u) != 0u;
}

/// Get texture dimensions as vec2.
fn texture_entry_dimensions(entry: TextureTableEntry) -> vec2<u32> {
    return vec2<u32>(entry.width, entry.height);
}

/// Get texture aspect ratio (width / height).
fn texture_entry_aspect_ratio(entry: TextureTableEntry) -> f32 {
    if entry.height == 0u {
        return 1.0;
    }
    return f32(entry.width) / f32(entry.height);
}

// =============================================================================
// Texture Index Validation
// =============================================================================

/// Check if a texture index is valid (not the sentinel value).
fn is_valid_texture_index(index: u32) -> bool {
    return index != INVALID_TEXTURE_INDEX;
}

/// Check if a texture index is within bounds of the bindless array.
fn is_texture_index_in_bounds(index: u32) -> bool {
    return index < MAX_BINDLESS_TEXTURES && index != INVALID_TEXTURE_INDEX;
}

/// Validate texture index, returning default value if invalid.
fn validate_texture_index(index: u32, default_value: u32) -> u32 {
    if is_valid_texture_index(index) {
        return index;
    }
    return default_value;
}

// =============================================================================
// Bindless Sampling Functions
// =============================================================================

// Note: The following functions assume the binding_array declarations above
// are active. Uncomment the appropriate version for your binding layout.

/// Sample from the bindless texture array.
///
/// @param textures: The bindless texture array binding.
/// @param samplers: The sampler array binding.
/// @param texture_index: Index into the bindless texture array.
/// @param sampler_index: Index into the sampler array.
/// @param uv: Texture coordinates in [0, 1] range.
/// @returns: Sampled RGBA color, or ERROR_COLOR if index is invalid.
fn sample_bindless_impl(
    textures: binding_array<texture_2d<f32>>,
    samplers: binding_array<sampler>,
    texture_index: u32,
    sampler_index: u32,
    uv: vec2<f32>
) -> vec4<f32> {
    // Validate texture index
    if texture_index == INVALID_TEXTURE_INDEX {
        return ERROR_COLOR;
    }

    // Sample from the bindless array
    return textureSample(textures[texture_index], samplers[sampler_index], uv);
}

/// Sample from bindless array with default sampler (index 0).
fn sample_bindless_default_impl(
    textures: binding_array<texture_2d<f32>>,
    samplers: binding_array<sampler>,
    texture_index: u32,
    uv: vec2<f32>
) -> vec4<f32> {
    return sample_bindless_impl(textures, samplers, texture_index, DEFAULT_SAMPLER_INDEX, uv);
}

/// Sample from bindless array with explicit LOD level.
fn sample_bindless_lod_impl(
    textures: binding_array<texture_2d<f32>>,
    samplers: binding_array<sampler>,
    texture_index: u32,
    sampler_index: u32,
    uv: vec2<f32>,
    lod: f32
) -> vec4<f32> {
    if texture_index == INVALID_TEXTURE_INDEX {
        return ERROR_COLOR;
    }
    return textureSampleLevel(textures[texture_index], samplers[sampler_index], uv, lod);
}

/// Sample from bindless array with gradient for anisotropic filtering.
fn sample_bindless_grad_impl(
    textures: binding_array<texture_2d<f32>>,
    samplers: binding_array<sampler>,
    texture_index: u32,
    sampler_index: u32,
    uv: vec2<f32>,
    ddx: vec2<f32>,
    ddy: vec2<f32>
) -> vec4<f32> {
    if texture_index == INVALID_TEXTURE_INDEX {
        return ERROR_COLOR;
    }
    return textureSampleGrad(textures[texture_index], samplers[sampler_index], uv, ddx, ddy);
}

/// Sample from bindless array with bias.
fn sample_bindless_bias_impl(
    textures: binding_array<texture_2d<f32>>,
    samplers: binding_array<sampler>,
    texture_index: u32,
    sampler_index: u32,
    uv: vec2<f32>,
    bias: f32
) -> vec4<f32> {
    if texture_index == INVALID_TEXTURE_INDEX {
        return ERROR_COLOR;
    }
    return textureSampleBias(textures[texture_index], samplers[sampler_index], uv, bias);
}

// =============================================================================
// Optional Texture Sampling (returns default if texture not bound)
// =============================================================================

/// Sample optional texture, returning fallback color if not bound.
fn sample_optional(
    textures: binding_array<texture_2d<f32>>,
    samplers: binding_array<sampler>,
    texture_index: u32,
    sampler_index: u32,
    uv: vec2<f32>,
    fallback: vec4<f32>
) -> vec4<f32> {
    if !is_valid_texture_index(texture_index) {
        return fallback;
    }
    return textureSample(textures[texture_index], samplers[sampler_index], uv);
}

/// Sample optional albedo texture (fallback: white).
fn sample_optional_albedo(
    textures: binding_array<texture_2d<f32>>,
    samplers: binding_array<sampler>,
    texture_index: u32,
    uv: vec2<f32>
) -> vec4<f32> {
    return sample_optional(textures, samplers, texture_index, DEFAULT_SAMPLER_INDEX, uv, vec4<f32>(1.0));
}

/// Sample optional normal texture (fallback: flat normal).
fn sample_optional_normal(
    textures: binding_array<texture_2d<f32>>,
    samplers: binding_array<sampler>,
    texture_index: u32,
    uv: vec2<f32>
) -> vec4<f32> {
    return sample_optional(textures, samplers, texture_index, DEFAULT_SAMPLER_INDEX, uv, FLAT_NORMAL);
}

/// Sample optional metallic-roughness texture (fallback: non-metallic, mid-roughness).
fn sample_optional_metallic_roughness(
    textures: binding_array<texture_2d<f32>>,
    samplers: binding_array<sampler>,
    texture_index: u32,
    uv: vec2<f32>
) -> vec4<f32> {
    // glTF convention: roughness in G channel, metallic in B channel
    return sample_optional(textures, samplers, texture_index, DEFAULT_SAMPLER_INDEX, uv, vec4<f32>(0.0, 0.5, 0.0, 1.0));
}

/// Sample optional emissive texture (fallback: no emission).
fn sample_optional_emissive(
    textures: binding_array<texture_2d<f32>>,
    samplers: binding_array<sampler>,
    texture_index: u32,
    uv: vec2<f32>
) -> vec4<f32> {
    return sample_optional(textures, samplers, texture_index, DEFAULT_SAMPLER_INDEX, uv, TRANSPARENT_BLACK);
}

/// Sample optional occlusion texture (fallback: no occlusion).
fn sample_optional_occlusion(
    textures: binding_array<texture_2d<f32>>,
    samplers: binding_array<sampler>,
    texture_index: u32,
    uv: vec2<f32>
) -> vec4<f32> {
    return sample_optional(textures, samplers, texture_index, DEFAULT_SAMPLER_INDEX, uv, vec4<f32>(1.0));
}

// =============================================================================
// Material Table Integration
// =============================================================================

// The MaterialTableEntry struct is defined in material_table.wgsl.
// These functions provide convenience wrappers for sampling material textures.

/// Check if a material has an albedo texture.
fn material_has_albedo_texture(albedo_texture_id: u32) -> bool {
    return is_valid_texture_index(albedo_texture_id);
}

/// Check if a material has a normal texture.
fn material_has_normal_texture(normal_texture_id: u32) -> bool {
    return is_valid_texture_index(normal_texture_id);
}

/// Check if a material has a metallic-roughness texture.
fn material_has_metallic_roughness_texture(mr_texture_id: u32) -> bool {
    return is_valid_texture_index(mr_texture_id);
}

/// Check if a material has an emissive texture.
fn material_has_emissive_texture(emissive_texture_id: u32) -> bool {
    return is_valid_texture_index(emissive_texture_id);
}

// =============================================================================
// Texture Coordinate Utilities
// =============================================================================

/// Transform UV coordinates with offset and scale.
fn transform_uv(uv: vec2<f32>, offset: vec2<f32>, scale: vec2<f32>) -> vec2<f32> {
    return uv * scale + offset;
}

/// Wrap UV coordinates to [0, 1] range (repeat mode).
fn wrap_uv_repeat(uv: vec2<f32>) -> vec2<f32> {
    return fract(uv);
}

/// Wrap UV coordinates with mirror repeat.
fn wrap_uv_mirror(uv: vec2<f32>) -> vec2<f32> {
    let t = floor(uv);
    let f = fract(uv);
    let mirror = vec2<f32>(
        select(f.x, 1.0 - f.x, (i32(t.x) & 1) != 0),
        select(f.y, 1.0 - f.y, (i32(t.y) & 1) != 0)
    );
    return mirror;
}

/// Clamp UV coordinates to [0, 1] range.
fn clamp_uv(uv: vec2<f32>) -> vec2<f32> {
    return clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0));
}

/// Check if UV coordinates are within [0, 1] range.
fn uv_in_range(uv: vec2<f32>) -> bool {
    return all(uv >= vec2<f32>(0.0)) && all(uv <= vec2<f32>(1.0));
}

// =============================================================================
// LOD Calculation
// =============================================================================

/// Calculate texture LOD from screen-space derivatives.
///
/// @param ddx: dFdx of UV coordinates.
/// @param ddy: dFdy of UV coordinates.
/// @param texture_size: Texture dimensions in pixels.
/// @returns: LOD level (0 = base level).
fn calculate_lod(ddx: vec2<f32>, ddy: vec2<f32>, texture_size: vec2<f32>) -> f32 {
    let dx = ddx * texture_size;
    let dy = ddy * texture_size;
    let d = max(dot(dx, dx), dot(dy, dy));
    return 0.5 * log2(max(d, 1.0));
}

/// Calculate anisotropic LOD with anisotropy ratio.
fn calculate_lod_anisotropic(
    ddx: vec2<f32>,
    ddy: vec2<f32>,
    texture_size: vec2<f32>,
    max_anisotropy: f32
) -> vec2<f32> {
    let dx = ddx * texture_size;
    let dy = ddy * texture_size;

    let px = dot(dx, dx);
    let py = dot(dy, dy);

    let major = max(px, py);
    let minor = min(px, py);

    let ratio = clamp(sqrt(major / max(minor, 0.0001)), 1.0, max_anisotropy);
    let lod = 0.5 * log2(major / (ratio * ratio));

    return vec2<f32>(max(lod, 0.0), ratio);
}

// =============================================================================
// Triplanar Mapping Utilities
// =============================================================================

/// Generate triplanar blend weights from world normal.
fn triplanar_weights(world_normal: vec3<f32>, sharpness: f32) -> vec3<f32> {
    let weights = pow(abs(world_normal), vec3<f32>(sharpness));
    return weights / (weights.x + weights.y + weights.z);
}

/// Sample texture with triplanar projection.
fn sample_triplanar(
    textures: binding_array<texture_2d<f32>>,
    samplers: binding_array<sampler>,
    texture_index: u32,
    world_pos: vec3<f32>,
    world_normal: vec3<f32>,
    scale: f32,
    sharpness: f32
) -> vec4<f32> {
    if !is_valid_texture_index(texture_index) {
        return ERROR_COLOR;
    }

    let weights = triplanar_weights(world_normal, sharpness);

    // Sample along each axis
    let uv_x = world_pos.yz * scale;
    let uv_y = world_pos.xz * scale;
    let uv_z = world_pos.xy * scale;

    let sample_x = textureSample(textures[texture_index], samplers[DEFAULT_SAMPLER_INDEX], uv_x);
    let sample_y = textureSample(textures[texture_index], samplers[DEFAULT_SAMPLER_INDEX], uv_y);
    let sample_z = textureSample(textures[texture_index], samplers[DEFAULT_SAMPLER_INDEX], uv_z);

    // Blend based on normal weights
    return sample_x * weights.x + sample_y * weights.y + sample_z * weights.z;
}

// =============================================================================
// Parallax Mapping Utilities
// =============================================================================

/// Simple parallax offset calculation.
fn parallax_offset(
    view_tangent: vec3<f32>,
    height: f32,
    scale: f32
) -> vec2<f32> {
    return view_tangent.xy * (height * scale) / view_tangent.z;
}

/// Steep parallax mapping step.
fn steep_parallax_step(
    textures: binding_array<texture_2d<f32>>,
    samplers: binding_array<sampler>,
    height_texture_index: u32,
    uv: vec2<f32>,
    view_tangent: vec3<f32>,
    scale: f32,
    num_layers: u32
) -> vec2<f32> {
    if !is_valid_texture_index(height_texture_index) {
        return uv;
    }

    let layer_depth = 1.0 / f32(num_layers);
    var current_depth = 0.0;
    var current_uv = uv;

    let delta_uv = (view_tangent.xy / view_tangent.z) * scale / f32(num_layers);

    for (var i = 0u; i < num_layers; i = i + 1u) {
        let height = textureSample(textures[height_texture_index], samplers[DEFAULT_SAMPLER_INDEX], current_uv).r;
        if current_depth >= height {
            break;
        }
        current_uv = current_uv - delta_uv;
        current_depth = current_depth + layer_depth;
    }

    return current_uv;
}

// =============================================================================
// Detail Texture Utilities
// =============================================================================

/// Blend detail texture with base using overlay mode.
fn blend_detail_overlay(base: vec4<f32>, detail: vec4<f32>, strength: f32) -> vec4<f32> {
    let overlay = select(
        2.0 * base * detail,
        vec4<f32>(1.0) - 2.0 * (vec4<f32>(1.0) - base) * (vec4<f32>(1.0) - detail),
        base.rgb > vec3<f32>(0.5)
    );
    return mix(base, vec4<f32>(overlay.rgb, base.a), strength);
}

/// Blend detail normal with base normal.
fn blend_detail_normal(base_normal: vec3<f32>, detail_normal: vec3<f32>) -> vec3<f32> {
    // Reoriented Normal Mapping (RNM)
    let t = base_normal + vec3<f32>(0.0, 0.0, 1.0);
    let u = detail_normal * vec3<f32>(-1.0, -1.0, 1.0);
    return normalize(t * dot(t, u) - u * t.z);
}
