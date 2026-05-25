// SPDX-License-Identifier: MIT
//
// material_table.wgsl -- Bindless material table entry definition (T-GPU-1.4).
//
// GPU buffer layout as `array<MaterialTableEntry>` in a storage buffer.
// Shaders reference materials by a u32 index into this array.
//
// Each entry is 80 bytes (aligned to 16 for vec4<f32> compatibility).
// The CPU-side MaterialTableEntry Rust struct at `material_table.rs` uses
// the identical layout, verified by unit tests.

/// A single PBR material descriptor in the bindless material table.
///
/// Fields are laid out with vec4<f32> members first for alignment:
///
/// | Offset | Size | Field                       | Description                     |
/// |--------|------|-----------------------------|---------------------------------|
/// | 0      | 16   | base_color                  | RGBA base colour (vec4<f32>)    |
/// | 16     | 16   | emissive                    | RGB emissive + intensity (.a)   |
/// | 32     | 4    | metallic                    | Metalness (0-1)                 |
/// | 36     | 4    | roughness                   | Roughness (0-1)                 |
/// | 40     | 4    | occlusion                   | Ambient occlusion (0-1)         |
/// | 44     | 4    | normal_scale                | Normal map intensity scale      |
/// | 48     | 4    | albedo_texture_id           | Index into bindless texture arr |
/// | 52     | 4    | normal_texture_id           | Index into bindless texture arr |
/// | 56     | 4    | metallic_roughness_tex_id   | Index into bindless texture arr |
/// | 60     | 4    | emissive_texture_id         | Index into bindless texture arr |
/// | 64     | 4    | flags                       | Bit 0 = visible, bit 31 = dirty |
/// | 68     | 4    | alpha_cutoff                | Alpha-mask threshold            |
/// | 72     | 8    | (implicit padding)          | Rounds to 80 (align 16)         |
///
/// Total: 80 bytes. Storage-buffer array stride: 80.
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

/// Returns `true` when the material's visible bit (flags bit 0) is set.
fn material_is_visible(entry: MaterialTableEntry) -> bool {
    return (entry.flags & 1u) != 0u;
}

/// Returns `true` when the material's dirty bit (flags bit 31) is set.
fn material_is_dirty(entry: MaterialTableEntry) -> bool {
    return (entry.flags & 0x80000000u) != 0u;
}

/// Extracts the base colour (vec4<f32>) from a material entry.
fn material_get_base_color(entry: MaterialTableEntry) -> vec4<f32> {
    return entry.base_color;
}

/// Extracts the emissive RGB (ignoring intensity) from a material entry.
fn material_get_emissive_rgb(entry: MaterialTableEntry) -> vec3<f32> {
    return entry.emissive.rgb;
}

/// Extracts the emissive intensity from a material entry (stored in .a).
fn material_get_emissive_intensity(entry: MaterialTableEntry) -> f32 {
    return entry.emissive.a;
}

/// Returns the final emissive colour = emissive.rgb * emissive.a.
fn material_get_emissive_final(entry: MaterialTableEntry) -> vec3<f32> {
    return entry.emissive.rgb * entry.emissive.a;
}

/// Returns `true` when the material has a valid albedo texture bound.
fn material_has_albedo_texture(entry: MaterialTableEntry) -> bool {
    return entry.albedo_texture_id != 0xFFFFFFFFu;
}

/// Returns `true` when the material has a valid normal texture bound.
fn material_has_normal_texture(entry: MaterialTableEntry) -> bool {
    return entry.normal_texture_id != 0xFFFFFFFFu;
}

/// Returns `true` when the material has a valid metallic-roughness texture.
fn material_has_mr_texture(entry: MaterialTableEntry) -> bool {
    return entry.metallic_roughness_tex_id != 0xFFFFFFFFu;
}

/// Returns `true` when the material has a valid emissive texture.
fn material_has_emissive_texture(entry: MaterialTableEntry) -> bool {
    return entry.emissive_texture_id != 0xFFFFFFFFu;
}
