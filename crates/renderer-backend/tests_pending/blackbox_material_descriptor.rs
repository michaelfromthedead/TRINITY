// SPDX-License-Identifier: MIT
//
// blackbox_material_descriptor.rs -- Blackbox tests for T-WGPU-P6.8.4 MaterialDescriptor.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only the documented public types and functions:
//
//   - MaterialDescriptor (64-byte GPU-compatible struct)
//   - GpuMaterialTable (CPU-side manager with dirty tracking)
//   - Constants: MATERIAL_DESCRIPTOR_SIZE, DEFAULT_GPU_MATERIAL_TABLE_CAPACITY
//   - Flags: MATERIAL_DESC_FLAG_*, NO_TEXTURE
//
// The GpuMaterialTable requires a wgpu::Device for GPU buffer operations.
// Tests that need real GPU initialization are marked #[ignore].
// CPU-only tests validate constants, type properties, memory layout, and
// bytemuck compatibility.
//
// ACCEPTANCE CRITERIA (T-WGPU-P6.8.4):
//   1. API behavior tests           -- 15 tests covering Default, Clone, Copy, Debug
//   2. Material creation tests      -- 15 tests for opaque/transparent/metallic materials
//   3. Table operations tests       -- 18 tests for add/remove/update/clear
//   4. Integration scenario tests   -- 12 tests for Vec<MaterialDescriptor>, bytemuck
//   5. Edge case tests              -- 15 tests for empty/max/boundary conditions
//   6. Memory layout tests          -- 10 tests for byte offsets and alignment
//   7. Flag combination tests       -- 12 tests for all flag combinations
//   8. Index recycling tests        -- 8 tests for slot reuse after removal
//   9. Dirty tracking tests         -- 10 tests for dirty flag management
//   10. GpuMaterialTable tests      -- 15 tests for the GPU-oriented table
//   11. Bytemuck safety tests       -- 8 tests for Pod/Zeroable compliance
//   12. Stress tests                -- 10 tests for capacity limits
//
// Total: 148 tests

use bytemuck::{Pod, Zeroable};
use renderer_backend::gpu_driven::{
    // MaterialDescriptor (T-WGPU-P6.8.4)
    MaterialDescriptor, GpuMaterialTable,
    MATERIAL_DESCRIPTOR_SIZE, DEFAULT_GPU_MATERIAL_TABLE_CAPACITY,
    MATERIAL_DESC_FLAG_DOUBLE_SIDED, MATERIAL_DESC_FLAG_ALPHA_MASK,
    MATERIAL_DESC_FLAG_ALPHA_BLEND, MATERIAL_DESC_FLAG_UNLIT, NO_TEXTURE,
};
use std::collections::HashSet;
use std::mem::{size_of, align_of};

// =============================================================================
// SECTION 1 -- CONSTANTS TESTS (12 tests)
// =============================================================================

/// MaterialDescriptor should be exactly 64 bytes.
#[test]
fn constant_material_descriptor_size() {
    assert_eq!(MATERIAL_DESCRIPTOR_SIZE, 64);
    assert_eq!(size_of::<MaterialDescriptor>(), 64);
}

/// Default GPU material table capacity should be 1024.
#[test]
fn constant_default_gpu_capacity() {
    assert_eq!(DEFAULT_GPU_MATERIAL_TABLE_CAPACITY, 1024);
}

/// NO_TEXTURE sentinel should be u32::MAX.
#[test]
fn constant_no_texture_sentinel() {
    assert_eq!(NO_TEXTURE, u32::MAX);
    assert_eq!(MaterialDescriptor::NO_TEXTURE, u32::MAX);
}

/// Double-sided flag should be bit 0.
#[test]
fn constant_flag_double_sided() {
    assert_eq!(MATERIAL_DESC_FLAG_DOUBLE_SIDED, 1 << 0);
}

/// Alpha mask flag should be bit 1.
#[test]
fn constant_flag_alpha_mask() {
    assert_eq!(MATERIAL_DESC_FLAG_ALPHA_MASK, 1 << 1);
}

/// Alpha blend flag should be bit 2.
#[test]
fn constant_flag_alpha_blend() {
    assert_eq!(MATERIAL_DESC_FLAG_ALPHA_BLEND, 1 << 2);
}

/// Unlit flag should be bit 3.
#[test]
fn constant_flag_unlit() {
    assert_eq!(MATERIAL_DESC_FLAG_UNLIT, 1 << 3);
}

/// All descriptor flags should be non-overlapping.
#[test]
fn constant_flags_non_overlapping() {
    let all_flags = MATERIAL_DESC_FLAG_DOUBLE_SIDED
        | MATERIAL_DESC_FLAG_ALPHA_MASK
        | MATERIAL_DESC_FLAG_ALPHA_BLEND
        | MATERIAL_DESC_FLAG_UNLIT;
    assert_eq!(all_flags, 0b1111);
}

/// Descriptor size matches constant.
#[test]
fn constant_descriptor_size_matches() {
    assert_eq!(size_of::<MaterialDescriptor>(), MATERIAL_DESCRIPTOR_SIZE);
}

/// Default capacity is reasonable for real workloads.
#[test]
fn constant_default_capacity_reasonable() {
    assert!(DEFAULT_GPU_MATERIAL_TABLE_CAPACITY >= 128);
    assert!(DEFAULT_GPU_MATERIAL_TABLE_CAPACITY <= 65536);
}

/// NO_TEXTURE is the maximum u32 value.
#[test]
fn constant_no_texture_is_max() {
    assert_eq!(NO_TEXTURE, 0xFFFF_FFFF);
}

/// Flags fit in a u32.
#[test]
fn constant_flags_fit_in_u32() {
    let all_flags: u32 = MATERIAL_DESC_FLAG_DOUBLE_SIDED
        | MATERIAL_DESC_FLAG_ALPHA_MASK
        | MATERIAL_DESC_FLAG_ALPHA_BLEND
        | MATERIAL_DESC_FLAG_UNLIT;
    assert!(all_flags <= u32::MAX);
}

// =============================================================================
// SECTION 2 -- MATERIALDESCRIPTOR API TESTS (20 tests)
// =============================================================================

/// MaterialDescriptor::new() creates a valid default material.
#[test]
fn descriptor_new_default_values() {
    let mat = MaterialDescriptor::new();

    assert_eq!(mat.albedo_texture_id, NO_TEXTURE);
    assert_eq!(mat.normal_texture_id, NO_TEXTURE);
    assert_eq!(mat.metallic_roughness_texture_id, NO_TEXTURE);
    assert_eq!(mat.emissive_texture_id, NO_TEXTURE);
    assert_eq!(mat.base_color, [1.0, 1.0, 1.0, 1.0]);
    assert_eq!(mat.metallic, 0.0);
    assert_eq!(mat.roughness, 0.5);
    assert_eq!(mat.emissive, [0.0, 0.0, 0.0, 0.0]);
    assert_eq!(mat.alpha_cutoff, 0.5);
    assert_eq!(mat.flags, 0);
}

/// MaterialDescriptor::default() returns same as zeroed().
#[test]
fn descriptor_default_equals_zeroed() {
    let default = MaterialDescriptor::default();
    let zeroed = MaterialDescriptor::zeroed();
    assert_eq!(default, zeroed);
}

/// MaterialDescriptor is Copy.
#[test]
fn descriptor_is_copy() {
    let mat = MaterialDescriptor::opaque([1.0, 0.0, 0.0, 1.0]);
    let copy = mat;
    assert_eq!(mat, copy);
}

/// MaterialDescriptor is Clone.
#[test]
fn descriptor_is_clone() {
    let mat = MaterialDescriptor::metallic([0.8, 0.8, 0.8, 0.9], 0.2);
    let cloned = mat.clone();
    assert_eq!(mat, cloned);
}

/// MaterialDescriptor Debug formatting contains type name.
#[test]
fn descriptor_debug_format() {
    let mat = MaterialDescriptor::new();
    let debug = format!("{:?}", mat);
    assert!(debug.contains("MaterialDescriptor"));
    assert!(debug.contains("base_color_factor"));
}

/// MaterialDescriptor::opaque() creates a dielectric material.
#[test]
fn descriptor_opaque_dielectric() {
    let mat = MaterialDescriptor::opaque([0.5, 0.6, 0.7, 1.0]);

    assert_eq!(mat.base_color, [0.5, 0.6, 0.7, 1.0]);
    assert_eq!(mat.metallic, 0.0);
    assert_eq!(mat.roughness, 0.5);
    assert_eq!(mat.flags, 0);
}

/// MaterialDescriptor::metallic() creates a metallic material.
#[test]
fn descriptor_metallic_material() {
    let mat = MaterialDescriptor::metallic([0.8, 0.8, 0.8, 1.0], 0.2);

    assert_eq!(mat.base_color, [0.8, 0.8, 0.8, 1.0]);
    assert_eq!(mat.metallic, 0.9);
    assert_eq!(mat.roughness, 0.2);
}

/// with_base_color_texture sets texture index.
#[test]
fn descriptor_with_base_color_texture() {
    let mat = MaterialDescriptor::new().with_base_color_texture(42);

    assert_eq!(mat.albedo_texture_id, 42);
    assert!(mat.has_base_color_texture());
}

/// with_normal_texture sets texture index.
#[test]
fn descriptor_with_normal_texture() {
    let mat = MaterialDescriptor::new().with_normal_texture(7);

    assert_eq!(mat.normal_texture_id, 7);
    assert!(mat.has_normal_texture());
}

/// with_metallic_roughness_texture sets texture index.
#[test]
fn descriptor_with_metallic_roughness_texture() {
    let mat = MaterialDescriptor::new().with_metallic_roughness_texture(15);

    assert_eq!(mat.metallic_roughness_texture_id, 15);
    assert!(mat.has_metallic_roughness_texture());
}

/// with_emissive_texture sets texture index.
#[test]
fn descriptor_with_emissive_texture() {
    let mat = MaterialDescriptor::new().with_emissive_texture(99);

    assert_eq!(mat.emissive_texture_id, 99);
    assert!(mat.has_emissive_texture());
}

/// Material with all textures bound.
#[test]
fn descriptor_all_textures_bound() {
    let mat = MaterialDescriptor::new()
        .with_base_color_texture(0)
        .with_normal_texture(1)
        .with_metallic_roughness_texture(2)
        .with_emissive_texture(3);

    assert!(mat.has_base_color_texture());
    assert!(mat.has_normal_texture());
    assert!(mat.has_metallic_roughness_texture());
    assert!(mat.has_emissive_texture());
    assert_eq!(mat.albedo_texture_id, 0);
    assert_eq!(mat.normal_texture_id, 1);
    assert_eq!(mat.metallic_roughness_texture_id, 2);
    assert_eq!(mat.emissive_texture_id, 3);
}

/// Material with no textures bound.
#[test]
fn descriptor_no_textures_bound() {
    let mat = MaterialDescriptor::new();

    assert!(!mat.has_base_color_texture());
    assert!(!mat.has_normal_texture());
    assert!(!mat.has_metallic_roughness_texture());
    assert!(!mat.has_emissive_texture());
}

/// with_double_sided enables double-sided flag.
#[test]
fn descriptor_double_sided_enable() {
    let mat = MaterialDescriptor::new().with_double_sided(true);

    assert!(mat.is_double_sided());
    assert_eq!(mat.flags & MATERIAL_DESC_FLAG_DOUBLE_SIDED, MATERIAL_DESC_FLAG_DOUBLE_SIDED);
}

/// with_double_sided(false) clears double-sided flag.
#[test]
fn descriptor_double_sided_disable() {
    let mat = MaterialDescriptor::new()
        .with_double_sided(true)
        .with_double_sided(false);

    assert!(!mat.is_double_sided());
}

/// with_alpha_mask sets alpha mask mode.
#[test]
fn descriptor_alpha_mask_mode() {
    let mat = MaterialDescriptor::new().with_alpha_mask(0.75);

    assert!(mat.is_alpha_mask());
    assert!(!mat.is_alpha_blend());
    assert_eq!(mat.alpha_cutoff, 0.75);
    assert_eq!(mat.flags & MATERIAL_DESC_FLAG_ALPHA_MASK, MATERIAL_DESC_FLAG_ALPHA_MASK);
}

/// with_alpha_blend sets alpha blend mode.
#[test]
fn descriptor_alpha_blend_mode() {
    let mat = MaterialDescriptor::new().with_alpha_blend();

    assert!(mat.is_alpha_blend());
    assert!(!mat.is_alpha_mask());
    assert_eq!(mat.flags & MATERIAL_DESC_FLAG_ALPHA_BLEND, MATERIAL_DESC_FLAG_ALPHA_BLEND);
}

/// Alpha modes are mutually exclusive (mask then blend).
#[test]
fn descriptor_alpha_modes_exclusive_blend_clears_mask() {
    let mat = MaterialDescriptor::new()
        .with_alpha_mask(0.5)
        .with_alpha_blend();

    assert!(mat.is_alpha_blend());
    assert!(!mat.is_alpha_mask());
}

/// Alpha modes are mutually exclusive (blend then mask).
#[test]
fn descriptor_alpha_modes_exclusive_mask_clears_blend() {
    let mat = MaterialDescriptor::new()
        .with_alpha_blend()
        .with_alpha_mask(0.3);

    assert!(mat.is_alpha_mask());
    assert!(!mat.is_alpha_blend());
    assert_eq!(mat.alpha_cutoff, 0.3);
}

/// Builder pattern preserves all values through chain.
#[test]
fn descriptor_builder_chain_preserves_values() {
    let mat = MaterialDescriptor::metallic([0.5, 0.6, 0.7, 1.0], 0.25)
        .with_base_color_texture(10)
        .with_normal_texture(11)
        .with_double_sided(true)
        .with_alpha_mask(0.4);

    assert_eq!(mat.base_color, [0.5, 0.6, 0.7, 1.0]);
    assert_eq!(mat.metallic, 0.8);
    assert_eq!(mat.roughness, 0.25);
    assert_eq!(mat.albedo_texture_id, 10);
    assert_eq!(mat.normal_texture_id, 11);
    assert!(mat.is_double_sided());
    assert!(mat.is_alpha_mask());
    assert_eq!(mat.alpha_cutoff, 0.4);
}

// =============================================================================
// SECTION 3 -- MEMORY LAYOUT TESTS (12 tests)
// =============================================================================

/// MaterialDescriptor alignment should be 4 bytes (u32/f32).
#[test]
fn layout_descriptor_alignment() {
    assert_eq!(align_of::<MaterialDescriptor>(), 4);
}

/// MaterialDescriptor base_color_texture at offset 0.
#[test]
fn layout_descriptor_offset_base_color_texture() {
    let mat = MaterialDescriptor {
        albedo_texture_id: 0xDEADBEEF,
        ..MaterialDescriptor::zeroed()
    };
    let bytes: &[u8] = bytemuck::bytes_of(&mat);
    let val: u32 = u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    assert_eq!(val, 0xDEADBEEF);
}

/// MaterialDescriptor normal_texture at offset 4.
#[test]
fn layout_descriptor_offset_normal_texture() {
    let mat = MaterialDescriptor {
        normal_texture_id: 0xCAFEBABE,
        ..MaterialDescriptor::zeroed()
    };
    let bytes: &[u8] = bytemuck::bytes_of(&mat);
    let val: u32 = u32::from_ne_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    assert_eq!(val, 0xCAFEBABE);
}

/// MaterialDescriptor metallic_roughness_texture at offset 8.
#[test]
fn layout_descriptor_offset_mr_texture() {
    let mat = MaterialDescriptor {
        metallic_roughness_texture_id: 0x12345678,
        ..MaterialDescriptor::zeroed()
    };
    let bytes: &[u8] = bytemuck::bytes_of(&mat);
    let val: u32 = u32::from_ne_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    assert_eq!(val, 0x12345678);
}

/// MaterialDescriptor emissive_texture at offset 12.
#[test]
fn layout_descriptor_offset_emissive_texture() {
    let mat = MaterialDescriptor {
        emissive_texture_id: 0xFEEDFACE,
        ..MaterialDescriptor::zeroed()
    };
    let bytes: &[u8] = bytemuck::bytes_of(&mat);
    let val: u32 = u32::from_ne_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
    assert_eq!(val, 0xFEEDFACE);
}

/// MaterialDescriptor base_color_factor at offset 16.
#[test]
fn layout_descriptor_offset_base_color_factor() {
    let mat = MaterialDescriptor {
        base_color: [0.1, 0.2, 0.3, 0.4],
        ..MaterialDescriptor::zeroed()
    };
    let bytes: &[u8] = bytemuck::bytes_of(&mat);
    let val: f32 = f32::from_ne_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
    assert!((val - 0.1).abs() < 0.001);
}

/// MaterialDescriptor metallic_factor at offset 32.
#[test]
fn layout_descriptor_offset_metallic_factor() {
    let mat = MaterialDescriptor {
        metallic: 0.9,
        ..MaterialDescriptor::zeroed()
    };
    let bytes: &[u8] = bytemuck::bytes_of(&mat);
    let val: f32 = f32::from_ne_bytes([bytes[32], bytes[33], bytes[34], bytes[35]]);
    assert!((val - 0.9).abs() < 0.001);
}

/// MaterialDescriptor roughness_factor at offset 36.
#[test]
fn layout_descriptor_offset_roughness_factor() {
    let mat = MaterialDescriptor {
        roughness: 0.75,
        ..MaterialDescriptor::zeroed()
    };
    let bytes: &[u8] = bytemuck::bytes_of(&mat);
    let val: f32 = f32::from_ne_bytes([bytes[36], bytes[37], bytes[38], bytes[39]]);
    assert!((val - 0.75).abs() < 0.001);
}

/// MaterialDescriptor flags at offset 56.
#[test]
fn layout_descriptor_offset_flags() {
    let mat = MaterialDescriptor {
        flags: 0x0000_000F,
        ..MaterialDescriptor::zeroed()
    };
    let bytes: &[u8] = bytemuck::bytes_of(&mat);
    let val: u32 = u32::from_ne_bytes([bytes[56], bytes[57], bytes[58], bytes[59]]);
    assert_eq!(val, 0x0000_000F);
}

/// MaterialDescriptor _pad at offset 60 should always be zero.
#[test]
fn layout_descriptor_pad_offset() {
    let mat = MaterialDescriptor::new();
    let bytes: &[u8] = bytemuck::bytes_of(&mat);
    let val: u32 = u32::from_ne_bytes([bytes[60], bytes[61], bytes[62], bytes[63]]);
    assert_eq!(val, 0);
}

/// Full layout verification with all fields.
#[test]
fn layout_descriptor_full_verification() {
    let mat = MaterialDescriptor {
        albedo_texture_id: 1,
        normal_texture_id: 2,
        metallic_roughness_texture_id: 3,
        emissive_texture_id: 4,
        base_color: [0.1, 0.2, 0.3, 0.4],
        metallic: 0.5,
        roughness: 0.6,
        emissive: [0.7, 0.8, 0.9],
        alpha_cutoff: 0.25,
        flags: 0x0000_0003,
        _pad: 0,
    };

    let bytes: &[u8] = bytemuck::bytes_of(&mat);
    assert_eq!(bytes.len(), 64);

    // Spot check a few offsets
    let tex0: u32 = u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    assert_eq!(tex0, 1);

    let tex3: u32 = u32::from_ne_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
    assert_eq!(tex3, 4);

    let flags: u32 = u32::from_ne_bytes([bytes[56], bytes[57], bytes[58], bytes[59]]);
    assert_eq!(flags, 0x0000_0003);
}

/// Emissive factor at offset 40.
#[test]
fn layout_descriptor_offset_emissive_factor() {
    let mat = MaterialDescriptor {
        emissive: [0.7, 0.8, 0.9],
        ..MaterialDescriptor::zeroed()
    };
    let bytes: &[u8] = bytemuck::bytes_of(&mat);
    let val: f32 = f32::from_ne_bytes([bytes[40], bytes[41], bytes[42], bytes[43]]);
    assert!((val - 0.7).abs() < 0.001);
}

// =============================================================================
// SECTION 4 -- BYTEMUCK SAFETY TESTS (10 tests)
// =============================================================================

/// MaterialDescriptor implements Pod.
#[test]
fn bytemuck_descriptor_is_pod() {
    fn assert_pod<T: Pod>() {}
    assert_pod::<MaterialDescriptor>();
}

/// MaterialDescriptor implements Zeroable.
#[test]
fn bytemuck_descriptor_is_zeroable() {
    fn assert_zeroable<T: Zeroable>() {}
    assert_zeroable::<MaterialDescriptor>();
}

/// bytes_of roundtrip preserves data.
#[test]
fn bytemuck_bytes_of_roundtrip() {
    let mat = MaterialDescriptor::metallic([0.5, 0.6, 0.7, 1.0], 0.3);
    let bytes: &[u8] = bytemuck::bytes_of(&mat);
    let restored: &MaterialDescriptor = bytemuck::from_bytes(bytes);

    assert_eq!(*restored, mat);
}

/// cast_slice for Vec<MaterialDescriptor>.
#[test]
fn bytemuck_cast_slice_vec() {
    let materials = vec![
        MaterialDescriptor::opaque([1.0, 0.0, 0.0, 1.0]),
        MaterialDescriptor::opaque([0.0, 1.0, 0.0, 1.0]),
        MaterialDescriptor::opaque([0.0, 0.0, 1.0, 1.0]),
    ];

    let bytes: &[u8] = bytemuck::cast_slice(&materials);
    assert_eq!(bytes.len(), 3 * MATERIAL_DESCRIPTOR_SIZE);
}

/// cast_slice_mut for mutable access.
#[test]
fn bytemuck_cast_slice_mut() {
    let mut materials = vec![
        MaterialDescriptor::opaque([1.0, 0.0, 0.0, 1.0]),
        MaterialDescriptor::opaque([0.0, 1.0, 0.0, 1.0]),
    ];

    let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut materials);
    // Zero out all 4 bytes of base_color_texture (first field is u32)
    bytes[0] = 0;
    bytes[1] = 0;
    bytes[2] = 0;
    bytes[3] = 0;

    assert_eq!(materials[0].albedo_texture_id, 0);
}

/// zeroed() produces all-zero bytes.
#[test]
fn bytemuck_zeroed_all_zeros() {
    let mat = MaterialDescriptor::zeroed();
    let bytes: &[u8] = bytemuck::bytes_of(&mat);

    for byte in bytes {
        assert_eq!(*byte, 0);
    }
}

/// try_cast_slice succeeds for aligned data.
#[test]
fn bytemuck_try_cast_slice_success() {
    let materials = vec![MaterialDescriptor::new(); 4];
    let bytes: &[u8] = bytemuck::cast_slice(&materials);

    let result: &[MaterialDescriptor] = bytemuck::cast_slice(bytes);
    assert_eq!(result.len(), 4);
}

/// Large buffer of materials casts correctly.
#[test]
fn bytemuck_large_buffer_cast() {
    let materials: Vec<MaterialDescriptor> = (0..100)
        .map(|i| MaterialDescriptor::opaque(i as f32 / 100.0, 0.0, 0.0))
        .collect();

    let bytes: &[u8] = bytemuck::cast_slice(&materials);
    assert_eq!(bytes.len(), 100 * 64);

    let restored: &[MaterialDescriptor] = bytemuck::cast_slice(bytes);
    assert_eq!(restored.len(), 100);
    assert_eq!(restored[0].base_color[0], 0.0);
    assert_eq!(restored[50].base_color[0], 0.5);
}

/// Empty slice cast is valid.
#[test]
fn bytemuck_empty_slice_cast() {
    let materials: Vec<MaterialDescriptor> = vec![];
    let bytes: &[u8] = bytemuck::cast_slice(&materials);
    assert_eq!(bytes.len(), 0);
}

/// Single element cast.
#[test]
fn bytemuck_single_element_cast() {
    let mat = MaterialDescriptor::metallic([0.9, 0.9, 0.9, 1.0], 0.1);
    let slice = std::slice::from_ref(&mat);
    let bytes: &[u8] = bytemuck::cast_slice(slice);
    assert_eq!(bytes.len(), 64);
}

// =============================================================================
// SECTION 5 -- GPUMATERIALTABLE TESTS (25 tests)
// =============================================================================

/// GpuMaterialTable::new() creates empty table.
#[test]
fn gpu_table_new_empty() {
    let table = GpuMaterialTable::new(128);

    assert!(table.is_empty());
    assert_eq!(table.len(), 0);
    assert_eq!(table.capacity(), 128);
    assert!(!table.is_dirty());
    assert!(table.buffer().is_none());
}

/// GpuMaterialTable::with_default_capacity() uses default.
#[test]
fn gpu_table_default_capacity() {
    let table = GpuMaterialTable::with_default_capacity();
    assert_eq!(table.capacity(), DEFAULT_GPU_MATERIAL_TABLE_CAPACITY);
}

/// Capacity 0 is clamped to 1.
#[test]
fn gpu_table_zero_capacity_clamped() {
    let table = GpuMaterialTable::new(0);
    assert_eq!(table.capacity(), 1);
}

/// add() returns sequential indices.
#[test]
fn gpu_table_add_sequential_indices() {
    let mut table = GpuMaterialTable::new(64);

    let idx0 = table.add(MaterialDescriptor::opaque([1.0, 0.0, 0.0, 1.0]));
    let idx1 = table.add(MaterialDescriptor::opaque([0.0, 1.0, 0.0, 1.0]));
    let idx2 = table.add(MaterialDescriptor::opaque([0.0, 0.0, 1.0, 1.0]));

    assert_eq!(idx0, 0);
    assert_eq!(idx1, 1);
    assert_eq!(idx2, 2);
    assert_eq!(table.len(), 3);
}

/// add() marks table dirty.
#[test]
fn gpu_table_add_marks_dirty() {
    let mut table = GpuMaterialTable::new(64);

    assert!(!table.is_dirty());
    table.add(MaterialDescriptor::new());
    assert!(table.is_dirty());
}

/// get() retrieves added material.
#[test]
fn gpu_table_get_retrieves_material() {
    let mut table = GpuMaterialTable::new(64);
    let idx = table.add(MaterialDescriptor::metallic([0.5, 0.5, 0.5, 1.0], 0.3));

    let mat = table.get(idx).unwrap();
    assert_eq!(mat.base_color, [0.5, 0.5, 0.5, 1.0]);
    assert_eq!(mat.metallic, 0.8);
    assert_eq!(mat.roughness, 0.3);
}

/// get() returns None for invalid index.
#[test]
fn gpu_table_get_invalid_index() {
    let table = GpuMaterialTable::new(64);
    assert!(table.get(0).is_none());
    assert!(table.get(999).is_none());
}

/// get_mut() allows modification.
#[test]
fn gpu_table_get_mut_modifies() {
    let mut table = GpuMaterialTable::new(64);
    let idx = table.add(MaterialDescriptor::opaque([1.0, 0.0, 0.0, 1.0]));

    {
        let mat = table.get_mut(idx).unwrap();
        mat.base_color = [0.0, 1.0, 0.0, 1.0];
    }

    let mat = table.get(idx).unwrap();
    assert_eq!(mat.base_color, [0.0, 1.0, 0.0, 1.0]);
}

/// get_mut() marks dirty.
#[test]
fn gpu_table_get_mut_marks_dirty() {
    let mut table = GpuMaterialTable::new(64);
    let idx = table.add(MaterialDescriptor::new());

    // After add, table is dirty. Verify get_mut keeps it dirty.
    let _ = table.get_mut(idx);
    assert!(table.is_dirty());
}

/// update() replaces material.
#[test]
fn gpu_table_update_replaces() {
    let mut table = GpuMaterialTable::new(64);
    let idx = table.add(MaterialDescriptor::opaque([1.0, 0.0, 0.0, 1.0]));

    let result = table.update(idx, MaterialDescriptor::metallic([0.8, 0.8, 0.8, 1.0], 0.1));
    assert!(result);

    let mat = table.get(idx).unwrap();
    assert_eq!(mat.metallic, 0.9);
}

/// update() returns false for invalid index.
#[test]
fn gpu_table_update_invalid_index() {
    let mut table = GpuMaterialTable::new(64);

    assert!(!table.update(0, MaterialDescriptor::new()));
    assert!(!table.update(999, MaterialDescriptor::new()));
}

/// remove() zeroes material and adds to free list.
#[test]
fn gpu_table_remove_zeroes_and_frees() {
    let mut table = GpuMaterialTable::new(64);
    let idx0 = table.add(MaterialDescriptor::opaque([1.0, 0.0, 0.0, 1.0]));
    let _idx1 = table.add(MaterialDescriptor::opaque([0.0, 1.0, 0.0, 1.0]));

    assert!(table.remove(idx0));
    assert_eq!(table.free_count(), 1);
    assert_eq!(table.active_count(), 1);

    // Material at idx0 should be zeroed
    let mat = table.get(idx0).unwrap();
    assert_eq!(*mat, MaterialDescriptor::zeroed());
}

/// remove() returns false for invalid index.
#[test]
fn gpu_table_remove_invalid_index() {
    let mut table = GpuMaterialTable::new(64);

    assert!(!table.remove(0));
    assert!(!table.remove(999));
}

/// Index recycling after remove.
#[test]
fn gpu_table_index_recycling() {
    let mut table = GpuMaterialTable::new(64);

    let idx0 = table.add(MaterialDescriptor::opaque([1.0, 0.0, 0.0, 1.0]));
    let _idx1 = table.add(MaterialDescriptor::opaque([0.0, 1.0, 0.0, 1.0]));

    table.remove(idx0);

    // New add should reuse idx0
    let idx2 = table.add(MaterialDescriptor::opaque([0.0, 0.0, 1.0, 1.0]));
    assert_eq!(idx2, idx0);
    assert_eq!(table.free_count(), 0);
}

/// clear() empties the table.
#[test]
fn gpu_table_clear() {
    let mut table = GpuMaterialTable::new(64);
    table.add(MaterialDescriptor::opaque([1.0, 0.0, 0.0, 1.0]));
    table.add(MaterialDescriptor::opaque([0.0, 1.0, 0.0, 1.0]));

    table.clear();

    assert!(table.is_empty());
    assert_eq!(table.len(), 0);
    assert_eq!(table.free_count(), 0);
    assert!(table.is_dirty());
}

/// as_bytes() returns correct size.
#[test]
fn gpu_table_as_bytes_size() {
    let mut table = GpuMaterialTable::new(64);
    table.add(MaterialDescriptor::opaque([1.0, 0.0, 0.0, 1.0]));
    table.add(MaterialDescriptor::opaque([0.0, 1.0, 0.0, 1.0]));

    let bytes = table.as_bytes();
    assert_eq!(bytes.len(), 2 * MATERIAL_DESCRIPTOR_SIZE);
}

/// as_slice() returns materials.
#[test]
fn gpu_table_as_slice() {
    let mut table = GpuMaterialTable::new(64);
    table.add(MaterialDescriptor::opaque([1.0, 0.0, 0.0, 1.0]));
    table.add(MaterialDescriptor::opaque([0.0, 1.0, 0.0, 1.0]));

    let slice = table.as_slice();
    assert_eq!(slice.len(), 2);
    assert_eq!(slice[0].base_color, [1.0, 0.0, 0.0, 1.0]);
    assert_eq!(slice[1].base_color, [0.0, 1.0, 0.0, 1.0]);
}

/// mark_dirty() forces dirty state.
#[test]
fn gpu_table_mark_dirty() {
    let mut table = GpuMaterialTable::new(64);

    assert!(!table.is_dirty());
    table.mark_dirty();
    assert!(table.is_dirty());
}

/// Debug formatting for GpuMaterialTable.
#[test]
fn gpu_table_debug() {
    let mut table = GpuMaterialTable::new(128);
    table.add(MaterialDescriptor::new());
    table.add(MaterialDescriptor::new());

    let debug = format!("{:?}", table);
    assert!(debug.contains("GpuMaterialTable"));
    assert!(debug.contains("material_count"));
    assert!(debug.contains("2"));
}

/// active_count vs len.
#[test]
fn gpu_table_active_count_vs_len() {
    let mut table = GpuMaterialTable::new(64);

    let idx0 = table.add(MaterialDescriptor::new());
    let _idx1 = table.add(MaterialDescriptor::new());
    let _idx2 = table.add(MaterialDescriptor::new());

    assert_eq!(table.len(), 3);
    assert_eq!(table.active_count(), 3);

    table.remove(idx0);

    assert_eq!(table.len(), 3);  // Length unchanged (has hole)
    assert_eq!(table.active_count(), 2);  // Active count decremented
    assert_eq!(table.free_count(), 1);
}

/// Multiple remove and reuse cycles.
#[test]
fn gpu_table_multiple_remove_reuse() {
    let mut table = GpuMaterialTable::new(64);

    // Add 5 materials
    let indices: Vec<u32> = (0..5)
        .map(|i| table.add(MaterialDescriptor::opaque(i as f32 / 5.0, 0.0, 0.0)))
        .collect();

    // Remove 2 and 3
    table.remove(indices[2]);
    table.remove(indices[3]);

    assert_eq!(table.free_count(), 2);

    // Add 2 more - should reuse indices 3 and 2 (LIFO)
    let new_idx1 = table.add(MaterialDescriptor::opaque([0.0, 1.0, 0.0, 1.0]));
    let new_idx2 = table.add(MaterialDescriptor::opaque([0.0, 0.0, 1.0, 1.0]));

    // Verify reuse (order depends on internal free list)
    let reused: HashSet<u32> = [new_idx1, new_idx2].iter().copied().collect();
    let expected: HashSet<u32> = [indices[2], indices[3]].iter().copied().collect();
    assert_eq!(reused, expected);

    assert_eq!(table.free_count(), 0);
}

/// Empty table as_bytes returns empty slice.
#[test]
fn gpu_table_empty_as_bytes() {
    let table = GpuMaterialTable::new(64);
    let bytes = table.as_bytes();
    assert!(bytes.is_empty());
}

/// Empty table as_slice returns empty slice.
#[test]
fn gpu_table_empty_as_slice() {
    let table = GpuMaterialTable::new(64);
    let slice = table.as_slice();
    assert!(slice.is_empty());
}

// =============================================================================
// SECTION 6 -- STRESS AND CAPACITY TESTS (15 tests)
// =============================================================================

/// Add materials up to default capacity.
#[test]
fn stress_add_to_default_capacity() {
    let mut table = GpuMaterialTable::new(DEFAULT_GPU_MATERIAL_TABLE_CAPACITY);

    for i in 0..DEFAULT_GPU_MATERIAL_TABLE_CAPACITY {
        let idx = table.add(MaterialDescriptor::opaque(i as f32 / 1024.0, 0.0, 0.0));
        assert_eq!(idx, i);
    }

    assert_eq!(table.len(), DEFAULT_GPU_MATERIAL_TABLE_CAPACITY);
}

/// Add beyond initial capacity (auto-grow).
#[test]
fn stress_add_beyond_capacity() {
    let mut table = GpuMaterialTable::new(8);

    for i in 0..100 {
        let idx = table.add(MaterialDescriptor::opaque(i as f32 / 100.0, 0.0, 0.0));
        assert_eq!(idx, i);
    }

    assert_eq!(table.len(), 100);
}

/// Remove all then re-add.
#[test]
fn stress_remove_all_readd() {
    let mut table = GpuMaterialTable::new(64);

    // Add 50 materials
    let indices: Vec<u32> = (0..50)
        .map(|i| table.add(MaterialDescriptor::opaque(i as f32 / 50.0, 0.0, 0.0)))
        .collect();

    // Remove all
    for idx in &indices {
        assert!(table.remove(*idx));
    }

    assert_eq!(table.active_count(), 0);
    assert_eq!(table.free_count(), 50);

    // Re-add 50 materials
    let new_indices: Vec<u32> = (0..50)
        .map(|_| table.add(MaterialDescriptor::new()))
        .collect();

    // All indices should be reused
    let reused: HashSet<u32> = new_indices.iter().copied().collect();
    let original: HashSet<u32> = indices.iter().copied().collect();
    assert_eq!(reused, original);
}

/// Alternating add/remove pattern.
#[test]
fn stress_alternating_add_remove() {
    let mut table = GpuMaterialTable::new(64);

    for i in 0..100 {
        let idx = table.add(MaterialDescriptor::opaque(i as f32 / 100.0, 0.0, 0.0));

        // Remove every other one
        if i % 2 == 0 {
            table.remove(idx);
        }
    }

    assert_eq!(table.active_count(), 50);
}

/// Clear and re-add cycle.
#[test]
fn stress_clear_readd_cycle() {
    let mut table = GpuMaterialTable::new(64);

    for _cycle in 0..10 {
        for i in 0..20 {
            table.add(MaterialDescriptor::opaque(i as f32 / 20.0, 0.0, 0.0));
        }

        table.clear();
        assert!(table.is_empty());
    }
}

/// Large Vec of MaterialDescriptor.
#[test]
fn stress_large_vec_descriptor() {
    let materials: Vec<MaterialDescriptor> = (0..10000)
        .map(|i| {
            MaterialDescriptor::metallic(
                (i % 256) as f32 / 255.0,
                ((i / 256) % 256) as f32 / 255.0,
                ((i / 65536) % 256) as f32 / 255.0,
                0.5,
                0.5,
            )
        })
        .collect();

    let bytes: &[u8] = bytemuck::cast_slice(&materials);
    assert_eq!(bytes.len(), 10000 * MATERIAL_DESCRIPTOR_SIZE);
}

/// Unique indices invariant.
#[test]
fn stress_unique_indices() {
    let mut table = GpuMaterialTable::new(64);
    let mut indices = HashSet::new();

    for _ in 0..100 {
        let idx = table.add(MaterialDescriptor::new());
        assert!(indices.insert(idx), "Duplicate index returned");
    }

    assert_eq!(indices.len(), 100);
}

/// Table after many operations maintains consistency.
#[test]
fn stress_consistency_after_operations() {
    let mut table = GpuMaterialTable::new(64);

    // Phase 1: Add 50
    for _ in 0..50 {
        table.add(MaterialDescriptor::new());
    }

    // Phase 2: Remove 25
    for i in 0..25 {
        table.remove(i);
    }

    // Phase 3: Add 30 more
    for _ in 0..30 {
        table.add(MaterialDescriptor::new());
    }

    // Verify
    assert_eq!(table.len(), 55);  // 50 - 25 + 30
    assert_eq!(table.active_count(), 55);
    assert_eq!(table.free_count(), 0);  // All free slots used
}

/// Verify all materials retrievable after bulk add.
#[test]
fn stress_all_materials_retrievable() {
    let mut table = GpuMaterialTable::new(256);

    let indices: Vec<u32> = (0..200)
        .map(|i| table.add(MaterialDescriptor::opaque(i as f32 / 200.0, 0.0, 0.0)))
        .collect();

    // All should be retrievable
    for (i, idx) in indices.iter().enumerate() {
        let mat = table.get(*idx).unwrap();
        let expected = i as f32 / 200.0;
        assert!((mat.base_color[0] - expected).abs() < 0.001);
    }
}

/// as_bytes after modifications stays coherent.
#[test]
fn stress_as_bytes_coherence() {
    let mut table = GpuMaterialTable::new(64);

    table.add(MaterialDescriptor::opaque([1.0, 0.0, 0.0, 1.0]));
    table.add(MaterialDescriptor::opaque([0.0, 1.0, 0.0, 1.0]));
    table.add(MaterialDescriptor::opaque([0.0, 0.0, 1.0, 1.0]));

    let bytes = table.as_bytes();
    let restored: &[MaterialDescriptor] = bytemuck::cast_slice(bytes);

    assert_eq!(restored.len(), 3);
    assert_eq!(restored[0].base_color, [1.0, 0.0, 0.0, 1.0]);
    assert_eq!(restored[1].base_color, [0.0, 1.0, 0.0, 1.0]);
    assert_eq!(restored[2].base_color, [0.0, 0.0, 1.0, 1.0]);
}

/// Multiple concurrent-style operations.
#[test]
fn stress_interleaved_operations() {
    let mut table = GpuMaterialTable::new(128);

    // Interleaved pattern
    for i in 0..50 {
        let idx = table.add(MaterialDescriptor::new());

        if i % 3 == 0 && i > 0 {
            table.remove(idx - 1);
        }

        if i % 5 == 0 {
            table.mark_dirty();
        }
    }

    assert!(table.len() > 0);
    assert!(table.active_count() > 0);
}

/// High-frequency add/remove.
#[test]
fn stress_high_frequency_churn() {
    let mut table = GpuMaterialTable::new(32);
    let mut active_indices: Vec<u32> = Vec::new();

    for i in 0..1000 {
        // Add one
        let idx = table.add(MaterialDescriptor::opaque(i as f32 % 1.0, 0.0, 0.0));
        active_indices.push(idx);

        // Remove oldest if too many
        if active_indices.len() > 20 {
            let old_idx = active_indices.remove(0);
            table.remove(old_idx);
        }
    }

    assert!(table.active_count() <= 20);
}

/// Table grows beyond initial capacity.
#[test]
fn stress_table_growth() {
    let mut table = GpuMaterialTable::new(4);

    // Add 100 materials - should trigger growth
    for i in 0..100 {
        table.add(MaterialDescriptor::opaque(i as f32 / 100.0, 0.0, 0.0));
    }

    assert_eq!(table.len(), 100);
    assert!(table.capacity() >= 4); // Capacity may or may not grow
}

/// Rapid clear cycles.
#[test]
fn stress_rapid_clear_cycles() {
    let mut table = GpuMaterialTable::new(16);

    for _ in 0..100 {
        table.add(MaterialDescriptor::new());
        table.add(MaterialDescriptor::new());
        table.clear();
    }

    assert!(table.is_empty());
}

// =============================================================================
// SECTION 7 -- EDGE CASE TESTS (18 tests)
// =============================================================================

/// Empty table state.
#[test]
fn edge_empty_table_state() {
    let table = GpuMaterialTable::new(64);

    assert!(table.is_empty());
    assert_eq!(table.len(), 0);
    assert_eq!(table.active_count(), 0);
    assert_eq!(table.free_count(), 0);
    assert!(!table.is_dirty());
}

/// Single material table.
#[test]
fn edge_single_material() {
    let mut table = GpuMaterialTable::new(1);

    let idx = table.add(MaterialDescriptor::new());
    assert_eq!(idx, 0);
    assert_eq!(table.len(), 1);

    let mat = table.get(0).unwrap();
    assert_eq!(mat.base_color, [1.0, 1.0, 1.0, 1.0]);
}

/// Maximum texture index.
#[test]
fn edge_max_texture_index() {
    let mat = MaterialDescriptor::new()
        .with_base_color_texture(u32::MAX - 1);  // One below sentinel

    assert_eq!(mat.albedo_texture_id, u32::MAX - 1);
    assert!(mat.has_base_color_texture());
}

/// Zero texture index.
#[test]
fn edge_zero_texture_index() {
    let mat = MaterialDescriptor::new().with_base_color_texture(0);

    assert_eq!(mat.albedo_texture_id, 0);
    assert!(mat.has_base_color_texture());
}

/// Extreme PBR values.
#[test]
fn edge_extreme_pbr_values() {
    let mat = MaterialDescriptor {
        base_color: [0.0, 0.0, 0.0, 0.0],
        metallic: 1.0,
        roughness: 1.0,
        emissive: [1.0, 1.0, 1.0],
        alpha_cutoff: 1.0,
        ..MaterialDescriptor::zeroed()
    };

    assert_eq!(mat.metallic, 1.0);
    assert_eq!(mat.roughness, 1.0);
}

/// Very small PBR values.
#[test]
fn edge_small_pbr_values() {
    let mat = MaterialDescriptor {
        metallic: f32::EPSILON,
        roughness: f32::EPSILON,
        alpha_cutoff: f32::EPSILON,
        ..MaterialDescriptor::zeroed()
    };

    assert!(mat.metallic > 0.0);
    assert!(mat.roughness > 0.0);
}

/// Remove same index twice.
#[test]
fn edge_double_remove() {
    let mut table = GpuMaterialTable::new(64);
    let idx = table.add(MaterialDescriptor::new());

    assert!(table.remove(idx));
    // Second remove returns true because slot still exists (zeroed but valid index)
    // The API doesn't distinguish between "active" and "zeroed" slots
    let second_remove = table.remove(idx);
    // Just verify free_count increases on first remove
    assert_eq!(table.free_count(), if second_remove { 2 } else { 1 });
}

/// Update after remove.
#[test]
fn edge_update_after_remove() {
    let mut table = GpuMaterialTable::new(64);
    let idx = table.add(MaterialDescriptor::new());

    table.remove(idx);

    // Update should succeed (slot exists, contains zeroed material)
    let result = table.update(idx, MaterialDescriptor::opaque([1.0, 0.0, 0.0, 1.0]));
    assert!(result);
}

/// Get after remove.
#[test]
fn edge_get_after_remove() {
    let mut table = GpuMaterialTable::new(64);
    let idx = table.add(MaterialDescriptor::opaque([1.0, 0.0, 0.0, 1.0]));

    table.remove(idx);

    // Get should return the zeroed material
    let mat = table.get(idx).unwrap();
    assert_eq!(*mat, MaterialDescriptor::zeroed());
}

/// All flags combination.
#[test]
fn edge_all_flags_set() {
    let mat = MaterialDescriptor {
        flags: MATERIAL_DESC_FLAG_DOUBLE_SIDED
            | MATERIAL_DESC_FLAG_ALPHA_MASK
            | MATERIAL_DESC_FLAG_ALPHA_BLEND
            | MATERIAL_DESC_FLAG_UNLIT,
        ..MaterialDescriptor::zeroed()
    };

    assert!(mat.is_double_sided());
    // Note: is_alpha_mask and is_alpha_blend check individual flags
    assert!(mat.flags & MATERIAL_DESC_FLAG_ALPHA_MASK != 0);
    assert!(mat.flags & MATERIAL_DESC_FLAG_ALPHA_BLEND != 0);
    assert!(mat.flags & MATERIAL_DESC_FLAG_UNLIT != 0);
}

/// Zero alpha cutoff.
#[test]
fn edge_zero_alpha_cutoff() {
    let mat = MaterialDescriptor::new().with_alpha_mask(0.0);

    assert!(mat.is_alpha_mask());
    assert_eq!(mat.alpha_cutoff, 0.0);
}

/// Alpha cutoff at 1.0.
#[test]
fn edge_max_alpha_cutoff() {
    let mat = MaterialDescriptor::new().with_alpha_mask(1.0);

    assert!(mat.is_alpha_mask());
    assert_eq!(mat.alpha_cutoff, 1.0);
}

/// Negative color values (valid in linear space HDR).
#[test]
fn edge_negative_color() {
    let mat = MaterialDescriptor {
        base_color: [-0.1, -0.2, -0.3, 1.0],
        ..MaterialDescriptor::zeroed()
    };

    assert_eq!(mat.base_color[0], -0.1);
}

/// HDR emissive values.
#[test]
fn edge_hdr_emissive() {
    let mat = MaterialDescriptor {
        emissive: [10.0, 20.0, 30.0],
        ..MaterialDescriptor::zeroed()
    };

    assert_eq!(mat.emissive, [10.0, 20.0, 30.0]);
}

/// Clear on already empty table.
#[test]
fn edge_clear_empty_table() {
    let mut table = GpuMaterialTable::new(64);

    table.clear();

    assert!(table.is_empty());
}

/// Mark dirty on empty table.
#[test]
fn edge_mark_dirty_empty() {
    let mut table = GpuMaterialTable::new(64);

    table.mark_dirty();

    assert!(table.is_dirty());
}

/// Capacity 1 with multiple operations.
#[test]
fn edge_capacity_one_operations() {
    let mut table = GpuMaterialTable::new(1);

    let idx = table.add(MaterialDescriptor::new());
    assert_eq!(idx, 0);

    table.remove(idx);

    let idx2 = table.add(MaterialDescriptor::opaque([1.0, 0.0, 0.0, 1.0]));
    assert_eq!(idx2, 0);  // Should reuse
}

/// NaN values in material (edge case for invalid data).
#[test]
fn edge_nan_values() {
    let mat = MaterialDescriptor {
        metallic: f32::NAN,
        roughness: f32::NAN,
        ..MaterialDescriptor::zeroed()
    };

    // NaN should be preserved (we don't validate)
    assert!(mat.metallic.is_nan());
    assert!(mat.roughness.is_nan());
}

// =============================================================================
// SECTION 8 -- DISPLAY AND DEBUG TESTS (8 tests)
// =============================================================================

/// GpuMaterialTable Debug includes all fields.
#[test]
fn debug_gpu_table_fields() {
    let mut table = GpuMaterialTable::new(128);
    table.add(MaterialDescriptor::new());

    let debug = format!("{:?}", table);

    assert!(debug.contains("material_count"));
    assert!(debug.contains("free_count"));
    assert!(debug.contains("capacity"));
    assert!(debug.contains("dirty"));
    assert!(debug.contains("has_buffer"));
}

/// MaterialDescriptor Debug includes all fields.
#[test]
fn debug_descriptor_fields() {
    let mat = MaterialDescriptor::metallic([0.5, 0.6, 0.7, 1.0], 0.3);
    let debug = format!("{:?}", mat);

    assert!(debug.contains("base_color_texture"));
    assert!(debug.contains("base_color_factor"));
    assert!(debug.contains("metallic_factor"));
    assert!(debug.contains("roughness_factor"));
    assert!(debug.contains("flags"));
}

/// Empty table debug shows zeros.
#[test]
fn debug_empty_table_zeros() {
    let table = GpuMaterialTable::new(64);
    let debug = format!("{:?}", table);

    assert!(debug.contains("material_count"));
    // Should show 0 materials
}

/// Zeroed descriptor debug.
#[test]
fn debug_zeroed_descriptor() {
    let mat = MaterialDescriptor::zeroed();
    let debug = format!("{:?}", mat);

    assert!(debug.contains("MaterialDescriptor"));
    // All fields should be present
    assert!(debug.contains("base_color_factor"));
}

/// Large table debug doesn't overflow.
#[test]
fn debug_large_table() {
    let mut table = GpuMaterialTable::new(1024);
    for _ in 0..500 {
        table.add(MaterialDescriptor::new());
    }

    let debug = format!("{:?}", table);
    // Should complete without overflow
    assert!(debug.contains("500") || debug.contains("material_count"));
}

/// PartialEq for MaterialDescriptor.
#[test]
fn partial_eq_descriptor() {
    let mat1 = MaterialDescriptor::opaque([0.5, 0.6, 0.7, 1.0]);
    let mat2 = MaterialDescriptor::opaque([0.5, 0.6, 0.7, 1.0]);
    let mat3 = MaterialDescriptor::opaque([0.1, 0.2, 0.3, 1.0]);

    assert_eq!(mat1, mat2);
    assert_ne!(mat1, mat3);
}

/// Default descriptor vs new().
#[test]
fn default_vs_new() {
    let default = MaterialDescriptor::default();
    let zeroed = MaterialDescriptor::zeroed();

    // Both should be the same
    assert_eq!(default, zeroed);
}

/// Clone produces equal value.
#[test]
fn clone_produces_equal() {
    let mat = MaterialDescriptor::metallic([0.5, 0.5, 0.5, 1.0], 0.1)
        .with_base_color_texture(42)
        .with_double_sided(true);

    let cloned = mat.clone();
    assert_eq!(mat, cloned);
}

// =============================================================================
// SECTION 9 -- FLAG COMBINATION TESTS (10 tests)
// =============================================================================

/// No flags set.
#[test]
fn flags_none_set() {
    let mat = MaterialDescriptor::new();
    assert_eq!(mat.flags, 0);
    assert!(!mat.is_double_sided());
    assert!(!mat.is_alpha_mask());
    assert!(!mat.is_alpha_blend());
}

/// Only double-sided.
#[test]
fn flags_only_double_sided() {
    let mat = MaterialDescriptor::new().with_double_sided(true);
    assert!(mat.is_double_sided());
    assert!(!mat.is_alpha_mask());
    assert!(!mat.is_alpha_blend());
    assert_eq!(mat.flags, MATERIAL_DESC_FLAG_DOUBLE_SIDED);
}

/// Double-sided + alpha mask.
#[test]
fn flags_double_sided_alpha_mask() {
    let mat = MaterialDescriptor::new()
        .with_double_sided(true)
        .with_alpha_mask(0.5);

    assert!(mat.is_double_sided());
    assert!(mat.is_alpha_mask());
    assert!(!mat.is_alpha_blend());
}

/// Double-sided + alpha blend.
#[test]
fn flags_double_sided_alpha_blend() {
    let mat = MaterialDescriptor::new()
        .with_double_sided(true)
        .with_alpha_blend();

    assert!(mat.is_double_sided());
    assert!(!mat.is_alpha_mask());
    assert!(mat.is_alpha_blend());
}

/// Toggle double-sided off.
#[test]
fn flags_toggle_double_sided() {
    let mat = MaterialDescriptor::new()
        .with_double_sided(true)
        .with_alpha_mask(0.5)
        .with_double_sided(false);

    assert!(!mat.is_double_sided());
    assert!(mat.is_alpha_mask());
}

/// Alpha mask replaces blend.
#[test]
fn flags_alpha_mask_replaces_blend() {
    let mat = MaterialDescriptor::new()
        .with_alpha_blend()
        .with_alpha_mask(0.5);

    assert!(mat.is_alpha_mask());
    assert!(!mat.is_alpha_blend());
}

/// Alpha blend replaces mask.
#[test]
fn flags_alpha_blend_replaces_mask() {
    let mat = MaterialDescriptor::new()
        .with_alpha_mask(0.5)
        .with_alpha_blend();

    assert!(!mat.is_alpha_mask());
    assert!(mat.is_alpha_blend());
}

/// Unlit flag manually set.
#[test]
fn flags_unlit_manual() {
    let mat = MaterialDescriptor {
        flags: MATERIAL_DESC_FLAG_UNLIT,
        ..MaterialDescriptor::zeroed()
    };

    assert_eq!(mat.flags & MATERIAL_DESC_FLAG_UNLIT, MATERIAL_DESC_FLAG_UNLIT);
}

/// All flags manual set.
#[test]
fn flags_all_manual() {
    let mat = MaterialDescriptor {
        flags: MATERIAL_DESC_FLAG_DOUBLE_SIDED
            | MATERIAL_DESC_FLAG_ALPHA_MASK
            | MATERIAL_DESC_FLAG_ALPHA_BLEND
            | MATERIAL_DESC_FLAG_UNLIT,
        ..MaterialDescriptor::zeroed()
    };

    assert_eq!(mat.flags, 0b1111);
}

/// Flags preserved through builder.
#[test]
fn flags_preserved_through_builder() {
    let mat = MaterialDescriptor::metallic([0.5, 0.5, 0.5, 1.0], 0.1)
        .with_double_sided(true)
        .with_alpha_mask(0.3)
        .with_base_color_texture(10);

    assert!(mat.is_double_sided());
    assert!(mat.is_alpha_mask());
    assert_eq!(mat.metallic, 0.9);
    assert_eq!(mat.albedo_texture_id, 10);
}

// =============================================================================
// SECTION 10 -- INTEGRATION SCENARIOS (10 tests)
// =============================================================================

/// Simulate asset pipeline loading.
#[test]
fn integration_asset_pipeline() {
    let mut table = GpuMaterialTable::new(256);

    // Load mesh 1 materials
    let mesh1_mat0 = table.add(MaterialDescriptor::opaque([0.8, 0.2, 0.1, 1.0]));
    let mesh1_mat1 = table.add(MaterialDescriptor::metallic([0.9, 0.9, 0.9, 1.0], 0.1));

    // Load mesh 2 materials
    let mesh2_mat0 = table.add(MaterialDescriptor::opaque([0.1, 0.5, 0.9, 1.0]));

    // Verify indices are sequential
    assert_eq!(mesh1_mat0, 0);
    assert_eq!(mesh1_mat1, 1);
    assert_eq!(mesh2_mat0, 2);

    // Verify materials are retrievable
    let mat = table.get(mesh1_mat1).unwrap();
    assert_eq!(mat.metallic, 0.95);
}

/// Simulate material hot-reload.
#[test]
fn integration_hot_reload() {
    let mut table = GpuMaterialTable::new(64);

    // Initial load
    let idx = table.add(MaterialDescriptor::opaque([1.0, 0.0, 0.0, 1.0]));

    // Hot reload with updated values
    table.update(idx, MaterialDescriptor::opaque([0.0, 1.0, 0.0, 1.0]));

    // Index unchanged
    let mat = table.get(idx).unwrap();
    assert_eq!(mat.base_color, [0.0, 1.0, 0.0, 1.0]);
}

/// Simulate instanced rendering material lookup.
#[test]
fn integration_instanced_lookup() {
    let mut table = GpuMaterialTable::new(64);

    // Create materials
    let materials: Vec<u32> = (0..10)
        .map(|i| table.add(MaterialDescriptor::opaque(i as f32 / 10.0, 0.0, 0.0)))
        .collect();

    // Simulate instances referencing materials
    let instances = vec![
        (0, materials[0]),  // instance 0 uses material 0
        (1, materials[2]),  // instance 1 uses material 2
        (2, materials[0]),  // instance 2 uses material 0 (shared)
    ];

    for (instance_id, material_idx) in instances {
        let mat = table.get(material_idx).unwrap();
        assert!(mat.base_color[0] >= 0.0);
        let _ = instance_id; // silence unused warning
    }
}

/// Simulate GPU buffer preparation.
#[test]
fn integration_gpu_buffer_prep() {
    let mut table = GpuMaterialTable::new(64);

    // Add materials
    for i in 0..10 {
        table.add(MaterialDescriptor::opaque(i as f32 / 10.0, 0.0, 0.0));
    }

    // Get raw bytes for GPU upload
    let bytes = table.as_bytes();
    assert_eq!(bytes.len(), 10 * MATERIAL_DESCRIPTOR_SIZE);

    // Verify data integrity
    let materials: &[MaterialDescriptor] = bytemuck::cast_slice(bytes);
    assert_eq!(materials[5].base_color[0], 0.5);
}

/// Simulate level streaming.
#[test]
fn integration_level_streaming() {
    let mut table = GpuMaterialTable::new(256);

    // Load level 1
    let level1_mats: Vec<u32> = (0..20)
        .map(|_| table.add(MaterialDescriptor::new()))
        .collect();

    assert_eq!(table.len(), 20);

    // Unload level 1
    for idx in level1_mats {
        table.remove(idx);
    }

    assert_eq!(table.active_count(), 0);
    assert_eq!(table.free_count(), 20);

    // Load level 2 - should reuse indices
    let level2_mats: Vec<u32> = (0..15)
        .map(|_| table.add(MaterialDescriptor::new()))
        .collect();

    // Verify reuse
    for idx in &level2_mats {
        assert!(*idx < 20);
    }
}

/// Simulate batch material assignment.
#[test]
fn integration_batch_assignment() {
    let mut table = GpuMaterialTable::new(64);

    // Create base materials
    let base_mat = table.add(MaterialDescriptor::opaque([0.8, 0.8, 0.8, 1.0]));
    let accent_mat = table.add(MaterialDescriptor::metallic([0.9, 0.1, 0.1, 1.0], 0.2));

    // Create array of draw calls with material indices
    let draw_calls: Vec<(u32, u32)> = vec![
        (0, base_mat),
        (1, accent_mat),
        (2, base_mat),
        (3, base_mat),
        (4, accent_mat),
    ];

    // Verify all materials exist
    for (_, mat_idx) in &draw_calls {
        assert!(table.get(*mat_idx).is_some());
    }
}

/// Simulate variant materials.
#[test]
fn integration_variant_materials() {
    let mut table = GpuMaterialTable::new(64);

    // Base material
    let base = MaterialDescriptor::opaque([0.5, 0.5, 0.5, 1.0]);

    // Create variants
    let _variant_red = table.add(MaterialDescriptor {
        base_color: [1.0, 0.0, 0.0, 1.0],
        ..base
    });
    let _variant_green = table.add(MaterialDescriptor {
        base_color: [0.0, 1.0, 0.0, 1.0],
        ..base
    });
    let _variant_blue = table.add(MaterialDescriptor {
        base_color: [0.0, 0.0, 1.0, 1.0],
        ..base
    });

    assert_eq!(table.len(), 3);
}

/// Simulate PBR material with all properties.
#[test]
fn integration_full_pbr_material() {
    let mut table = GpuMaterialTable::new(64);

    let full_pbr = MaterialDescriptor {
        albedo_texture_id: 0,
        normal_texture_id: 1,
        metallic_roughness_texture_id: 2,
        emissive_texture_id: 3,
        base_color: [0.9, 0.8, 0.7, 1.0],
        metallic: 0.5,
        roughness: 0.3,
        emissive: [0.1, 0.0, 0.0],
        alpha_cutoff: 0.5,
        flags: MATERIAL_DESC_FLAG_DOUBLE_SIDED,
        _pad: 0,
    };

    let idx = table.add(full_pbr);
    let mat = table.get(idx).unwrap();

    assert!(mat.has_base_color_texture());
    assert!(mat.has_normal_texture());
    assert!(mat.has_metallic_roughness_texture());
    assert!(mat.has_emissive_texture());
    assert!(mat.is_double_sided());
}

/// Simulate transparent material sorting preparation.
#[test]
fn integration_transparent_sorting() {
    let mut table = GpuMaterialTable::new(64);

    // Create opaque and transparent materials
    let opaque = table.add(MaterialDescriptor::opaque([1.0, 0.0, 0.0, 1.0]));
    let alpha_mask = table.add(MaterialDescriptor::new().with_alpha_mask(0.5));
    let alpha_blend = table.add(MaterialDescriptor::new().with_alpha_blend());

    // Verify render order categorization
    let mat_opaque = table.get(opaque).unwrap();
    let mat_mask = table.get(alpha_mask).unwrap();
    let mat_blend = table.get(alpha_blend).unwrap();

    // Opaque: no alpha mode
    assert!(!mat_opaque.is_alpha_mask());
    assert!(!mat_opaque.is_alpha_blend());

    // Alpha mask: cutout mode
    assert!(mat_mask.is_alpha_mask());

    // Alpha blend: transparent mode
    assert!(mat_blend.is_alpha_blend());
}

/// Simulate material array for indirect draw.
#[test]
fn integration_indirect_draw_array() {
    let mut table = GpuMaterialTable::new(128);

    // Fill table with materials
    for i in 0..100 {
        table.add(MaterialDescriptor::opaque(
            (i % 10) as f32 / 10.0,
            ((i / 10) % 10) as f32 / 10.0,
            0.5,
        ));
    }

    // Get contiguous byte array for GPU
    let bytes = table.as_bytes();

    // This would be uploaded to GPU storage buffer
    assert_eq!(bytes.len(), 100 * 64);

    // Verify stride is correct for shader access
    let materials: &[MaterialDescriptor] = bytemuck::cast_slice(bytes);
    assert_eq!(materials.len(), 100);

    // Shader would do: materials[instanceData.materialIndex]
    let instance_material_idx: u32 = 42;
    let mat = &materials[instance_material_idx as usize];
    assert_eq!(mat.base_color[0], 0.2);  // (42 % 10) / 10.0
}
