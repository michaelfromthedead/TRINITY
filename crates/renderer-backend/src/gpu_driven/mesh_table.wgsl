// SPDX-License-Identifier: MIT
//
// mesh_table.wgsl -- Bindless mesh table entry definition (T-GPU-1.3).
//
// GPU buffer layout as `array<MeshTableEntry>` in a storage buffer. Shaders
// reference meshes by a u32 index into this array.
//
// Each entry is 24 bytes (6 x u32, 4-byte aligned). The CPU-side
// `MeshTableEntry` Rust struct at `mesh_table.rs` uses the identical layout,
// verified by unit tests.

/// A single mesh descriptor in the bindless mesh table.
///
/// Fields are tightly packed (no padding between u32 members):
///
/// | Offset | Size | Field          | Description                    |
/// |--------|------|----------------|--------------------------------|
/// | 0      | 4    | index_offset   | Byte offset in GPU index buffer |
/// | 4      | 4    | vertex_offset  | Byte offset in GPU vertex buffer|
/// | 8      | 4    | index_count    | Number of indices to draw       |
/// | 12     | 4    | vertex_count   | Number of vertices in this mesh |
/// | 16     | 4    | material_id    | Index into the material table   |
/// | 20     | 4    | flags          | Bitmask (bit 0 = visible)       |
///
/// Total: 24 bytes. In a storage buffer this forms `array<MeshTableEntry>`.
struct MeshTableEntry {
    index_offset: u32,
    vertex_offset: u32,
    index_count: u32,
    vertex_count: u32,
    material_id: u32,
    flags: u32,
}

/// Returns `true` when the mesh's visible bit (flags bit 0) is set.
fn mesh_is_visible(entry: MeshTableEntry) -> bool {
    return (entry.flags & 1u) != 0u;
}

/// Returns the `[start, end)` byte range of index data for this mesh.
///
/// Each index is 4 bytes (u32). The range is:
///   start = entry.index_offset
///   end   = entry.index_offset + entry.index_count * 4
fn mesh_index_byte_range(entry: MeshTableEntry) -> vec2<u32> {
    return vec2<u32>(
        entry.index_offset,
        entry.index_offset + entry.index_count * 4u,
    );
}

/// Returns the `[start, end)` byte range of vertex data for this mesh.
///
/// Each vertex position is assumed to be 12 bytes (vec3<f32>). The range is:
///   start = entry.vertex_offset
///   end   = entry.vertex_offset + entry.vertex_count * 12u
fn mesh_vertex_byte_range(entry: MeshTableEntry) -> vec2<u32> {
    return vec2<u32>(
        entry.vertex_offset,
        entry.vertex_offset + entry.vertex_count * 12u,
    );
}
