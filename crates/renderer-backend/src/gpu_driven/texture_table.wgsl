// SPDX-License-Identifier: MIT
//
// texture_table.wgsl -- Bindless texture table entry definition (T-GPU-1.5).
//
// GPU resource is `texture_2d_array<f32>` bound as a bindless texture array.
// Shaders reference textures by a u32 index (array layer) into this resource.
// The CPU-side metadata buffer (`array<TextureTableEntry>`) is used for
// introspection (size, format queries) and can be uploaded via the staging
// pipeline.
//
// Each entry is 24 bytes (6 x u32, 4-byte aligned). The CPU-side
// `TextureTableEntry` Rust struct at `texture_table.rs` uses the identical
// layout, verified by unit tests.

/// Metadata for a single texture in the bindless texture array.
///
/// Fields are tightly packed (no padding between u32 members):
///
/// | Offset | Size | Field        | Description                          |
/// |--------|------|--------------|--------------------------------------|
/// | 0      | 4    | width        | Texture width in pixels              |
/// | 4      | 4    | height       | Texture height in pixels             |
/// | 8      | 4    | mip_levels   | Number of mip levels                 |
/// | 12     | 4    | format       | Packed texture format identifier     |
/// | 16     | 4    | layer_count  | Number of array layers (1 for 2D)    |
/// | 20     | 4    | flags        | Bitmask (bit 0 = valid)              |
///
/// Total: 24 bytes. In a storage buffer this forms `array<TextureTableEntry>`.
struct TextureTableEntry {
    width: u32,
    height: u32,
    mip_levels: u32,
    format: u32,
    layer_count: u32,
    flags: u32,
}

/// Maximum number of bindless textures supported.
///
/// Must match `MAX_BINDLESS_TEXTURES` in `texture_table.rs`.
const MAX_BINDLESS_TEXTURES: u32 = 4096u;

/// Returns `true` when the texture index is within the valid bindless range.
fn texture_index_valid(index: u32) -> bool {
    return index < MAX_BINDLESS_TEXTURES;
}

/// Returns `true` when the texture's valid bit (flags bit 0) is set.
fn texture_is_valid(entry: TextureTableEntry) -> bool {
    return (entry.flags & 1u) != 0u;
}

/// Returns `true` when the texture has more than one mip level.
fn texture_has_mips(entry: TextureTableEntry) -> bool {
    return entry.mip_levels > 1u;
}

/// Returns the total number of texels for a given texture entry.
///
/// The intermediate multiplication is promoted to `u64` to avoid overflow
/// when width, height, or layer_count exceed 2^16 (e.g. 4096 x 4096 x N).
fn texture_total_texels(entry: TextureTableEntry) -> u64 {
    return u64(entry.width) * u64(entry.height) * u64(entry.layer_count);
}
