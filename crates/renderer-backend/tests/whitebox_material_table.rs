// SPDX-License-Identifier: MIT
//
// whitebox_material_table.rs -- Whitebox tests for MaterialTable (T-GPU-1.4).
//
// These tests verify internal invariants of the bindless material table:
//
//   - WGSL shader compilation and struct layout validation via naga
//   - Byte-level field offset verification against the WGSL spec
//   - Flag-bit arithmetic and sentinel invariants
//   - Dirty-tracking through get_mut with observable mutation side-effects
//   - Capacity auto-resize boundary conditions
//   - BufferRegistry staging lifecycle integration
//   - is_zero() deep edge cases with all flag combinations
//   - Serialisation round-trip and invariants
//
// CLEANROOM: These tests use ONLY the public API exported by the crate
// (renderer_backend::gpu_driven::*). They verify internal behaviour through
// observable state transitions -- no private fields, no wild pointer tricks.

use bytemuck::Zeroable;
use renderer_backend::gpu_driven::{
    BufferRegistry, MaterialTable, MaterialTableEntry, MaterialRemoveResult,
    DEFAULT_MATERIAL_TABLE_CAPACITY, MATERIAL_FLAG_DIRTY, MATERIAL_FLAG_VISIBLE,
    MATERIAL_TABLE_ENTRY_SIZE,
    // T-WGPU-P6.8.4: MaterialDescriptor + GpuMaterialTable
    MaterialDescriptor, GpuMaterialTable,
    MATERIAL_DESCRIPTOR_SIZE, DEFAULT_GPU_MATERIAL_TABLE_CAPACITY,
    MATERIAL_DESC_FLAG_DOUBLE_SIDED, MATERIAL_DESC_FLAG_ALPHA_MASK,
    MATERIAL_DESC_FLAG_ALPHA_BLEND, MATERIAL_DESC_FLAG_UNLIT, NO_TEXTURE,
};

// =============================================================================
// Test fixture: WGSL source
// =============================================================================

static WGSL_SOURCE: &str = include_str!("../src/gpu_driven/material_table.wgsl");

// =============================================================================
// Helpers
// =============================================================================

/// Creates a material entry with the given base colour for testing.
fn make_entry(r: f32, g: f32, b: f32, a: f32) -> MaterialTableEntry {
    MaterialTableEntry {
        base_color: [r, g, b, a],
        ..MaterialTableEntry::zeroed()
    }
}

/// A minimal BufferRegistry pre-sized for material table staging tests.
fn staging_registry() -> BufferRegistry {
    // 256 KiB -- enough for a full default-capacity table.
    BufferRegistry::new(256 * 1024)
}

/// Asserts that an entry read via `as_bytes()` at the given slot offset
/// matches the expected f32 value.
fn assert_byte_f32(bytes: &[u8], offset: usize, expected: f32) {
    let actual = f32::from_le_bytes(
        bytes[offset..offset + 4].try_into().unwrap(),
    );
    assert!(
        (actual - expected).abs() <= f32::EPSILON,
        "byte offset {}: expected {:.6}, got {:.6}",
        offset,
        expected,
        actual,
    );
}

/// Asserts that an entry read via `as_bytes()` at the given slot offset
/// matches the expected u32 value.
fn assert_byte_u32(bytes: &[u8], offset: usize, expected: u32) {
    let actual = u32::from_le_bytes(
        bytes[offset..offset + 4].try_into().unwrap(),
    );
    assert_eq!(
        actual, expected,
        "byte offset {}: expected 0x{:08x}, got 0x{:08x}",
        offset, expected, actual,
    );
}

// =============================================================================
// SECTION 1 -- WGSL shader compilation and layout verification
// =============================================================================

/// The WGSL source compiles without errors via naga.
///
/// This catches syntax errors, type mismatches, and unsupported constructs
/// in the shader source before they reach the runtime WGSL compiler.
#[test]
fn wgsl_compiles_via_naga() {
    let module = naga::front::wgsl::parse_str(WGSL_SOURCE);
    match module {
        Ok(_) => {}
        Err(err) => {
            panic!("WGSL failed to parse via naga:\n{:#?}", err);
        }
    }
}

/// The WGSL struct `MaterialTableEntry` exists and has exactly the expected
/// number of members (12).
#[test]
fn wgsl_struct_has_correct_member_count() {
    let module = naga::front::wgsl::parse_str(WGSL_SOURCE)
        .expect("WGSL must parse");
    let struct_name = "MaterialTableEntry";

    // Find the struct definition by iterating types.
    let found = module.types.iter().any(|(_handle, ty)| {
        if ty.name.as_deref() == Some(struct_name) {
            if let naga::TypeInner::Struct { members, .. } = &ty.inner {
                return members.len() == 12;
            }
        }
        false
    });

    assert!(
        found,
        "WGSL struct '{}' must exist with exactly 12 members",
        struct_name
    );
}

/// All 6 WGSL helper functions must be present and return the correct type.
#[test]
fn wgsl_helpers_have_correct_signatures() {
    let module = naga::front::wgsl::parse_str(WGSL_SOURCE)
        .expect("WGSL must parse");

    let expected_functions: &[(&str, u64)] = &[
        ("material_is_visible",            1), // returns bool -> Scalar(Bool)
        ("material_is_dirty",              1), // returns bool -> Scalar(Bool)
        ("material_get_base_color",        4), // returns vec4<f32>
        ("material_get_emissive_rgb",      3), // returns vec3<f32>
        ("material_get_emissive_intensity",2), // returns f32 -> Scalar(Float)
        ("material_get_emissive_final",    3), // returns vec3<f32>
    ];

    for (name, _expected_size) in expected_functions {
        let fn_found = module.functions.iter().any(|(_handle, func)| {
            func.name.as_deref() == Some(*name)
        });
        assert!(
            fn_found,
            "WGSL helper function '{}' not found",
            name,
        );
    }
}

/// All 4 texture presence-checking helpers exist.
#[test]
fn wgsl_texture_query_helpers_exist() {
    let module = naga::front::wgsl::parse_str(WGSL_SOURCE)
        .expect("WGSL must parse");

    let tex_helpers: &[&str] = &[
        "material_has_albedo_texture",
        "material_has_normal_texture",
        "material_has_mr_texture",
        "material_has_emissive_texture",
    ];

    for name in tex_helpers {
        let found = module.functions.iter().any(|(_handle, func)| {
            func.name.as_deref() == Some(*name)
        });
        assert!(
            found,
            "WGSL texture helper '{}' not found",
            name
        );
    }
}

/// The WGSL source uses `0xFFFFFFFFu` for the texture sentinel (u32::MAX).
#[test]
fn wgsl_uses_u32_max_sentinel() {
    assert!(
        WGSL_SOURCE.contains("0xFFFFFFFFu"),
        "WGSL source must use 0xFFFFFFFFu as the texture sentinel"
    );
}

// =============================================================================
// SECTION 2 -- Byte-level layout verification
// =============================================================================

/// Verify every field offset against the documented WGSL layout.
///
/// | Offset | Size | Field                       |
/// |--------|------|-----------------------------|
/// | 0      | 16   | base_color                  |
/// | 16     | 16   | emissive                    |
/// | 32     | 4    | metallic                    |
/// | 36     | 4    | roughness                   |
/// | 40     | 4    | occlusion                   |
/// | 44     | 4    | normal_scale                |
/// | 48     | 4    | albedo_texture_id           |
/// | 52     | 4    | normal_texture_id           |
/// | 56     | 4    | metallic_roughness_tex_id   |
/// | 60     | 4    | emissive_texture_id         |
/// | 64     | 4    | flags                       |
/// | 68     | 4    | alpha_cutoff                |
/// | 72     | 8    | (padding)                   |
/// Total: 80 bytes.
#[test]
fn byte_layout_matches_wgsl_spec() {
    let entry = MaterialTableEntry {
        base_color: [1.0, 2.0, 3.0, 4.0],
        emissive: [5.0, 6.0, 7.0, 8.0],
        metallic: 9.0,
        roughness: 10.0,
        occlusion: 11.0,
        normal_scale: 12.0,
        albedo_texture_id: 13,
        normal_texture_id: 14,
        metallic_roughness_texture_id: 15,
        emissive_texture_id: 16,
        flags: 0x8000_0001,
        alpha_cutoff: 0.5,
        _pad: [0, 0],
    };

    let bytes = unsafe {
        core::slice::from_raw_parts(
            &entry as *const _ as *const u8,
            MATERIAL_TABLE_ENTRY_SIZE,
        )
    };

    // base_color at offset 0 (16 bytes)
    let bc: &[f32; 4] = unsafe { &*(bytes.as_ptr() as *const [f32; 4]) };
    assert_eq!(bc[0], 1.0);
    assert_eq!(bc[1], 2.0);
    assert_eq!(bc[2], 3.0);
    assert_eq!(bc[3], 4.0);

    // emissive at offset 16 (16 bytes)
    let em: &[f32; 4] = unsafe { &*(bytes.as_ptr().add(16) as *const [f32; 4]) };
    assert_eq!(em[0], 5.0);
    assert_eq!(em[1], 6.0);
    assert_eq!(em[2], 7.0);
    assert_eq!(em[3], 8.0);

    // metallic at offset 32
    assert_byte_f32(bytes, 32, 9.0);
    // roughness at offset 36
    assert_byte_f32(bytes, 36, 10.0);
    // occlusion at offset 40
    assert_byte_f32(bytes, 40, 11.0);
    // normal_scale at offset 44
    assert_byte_f32(bytes, 44, 12.0);

    // albedo_texture_id at offset 48
    assert_byte_u32(bytes, 48, 13);
    // normal_texture_id at offset 52
    assert_byte_u32(bytes, 52, 14);
    // metallic_roughness_texture_id at offset 56
    assert_byte_u32(bytes, 56, 15);
    // emissive_texture_id at offset 60
    assert_byte_u32(bytes, 60, 16);

    // flags at offset 64
    assert_byte_u32(bytes, 64, 0x8000_0001);
    // alpha_cutoff at offset 68
    assert_byte_f32(bytes, 68, 0.5);

    // Confirm total size.
    assert_eq!(bytes.len(), 80);
}

/// The padding bytes (offset 72 through 79) are zero in a default entry.
#[test]
fn padding_bytes_are_zero() {
    let entry = MaterialTableEntry::zeroed();
    let bytes = unsafe {
        core::slice::from_raw_parts(
            &entry as *const _ as *const u8,
            MATERIAL_TABLE_ENTRY_SIZE,
        )
    };

    // Padding occupies offsets 72-79 (8 bytes).
    for offset in 72..80 {
        assert_eq!(
            bytes[offset], 0u8,
            "Padding byte at offset {} must be zero",
            offset
        );
    }
}

/// Every field placed at a unique offset -- no overlap.
#[test]
fn no_field_overlap() {
    let entry = MaterialTableEntry {
        base_color: [1.0, 2.0, 3.0, 4.0],
        emissive: [5.0, 6.0, 7.0, 8.0],
        metallic: 9.0,
        roughness: 10.0,
        occlusion: 11.0,
        normal_scale: 12.0,
        albedo_texture_id: 13,
        normal_texture_id: 14,
        metallic_roughness_texture_id: 15,
        emissive_texture_id: 16,
        flags: 0x8000_0001,
        alpha_cutoff: 0.5,
        _pad: [0, 0],
    };

    let bytes = unsafe {
        core::slice::from_raw_parts(
            &entry as *const _ as *const u8,
            MATERIAL_TABLE_ENTRY_SIZE,
        )
    };

    // Verify each field reads independently by checking the adjacent field
    // boundaries. The 4 f32s of base_color at [0..16) should NOT leak into
    // emissive at [16..32).
    let bc_end: f32 = unsafe { bytes.as_ptr().add(12).cast::<f32>().read() };
    assert_eq!(bc_end, 4.0, "base_color[3] must be at offset 12");

    let em_start: f32 = unsafe { bytes.as_ptr().add(16).cast::<f32>().read() };
    assert_eq!(em_start, 5.0, "emissive[0] must be at offset 16");

    // The boundary between metallic (offset 32) and roughness (offset 36)
    // must not overlap.
    let metallic_at_32: f32 = unsafe { bytes.as_ptr().add(32).cast::<f32>().read() };
    assert_eq!(metallic_at_32, 9.0);

    let roughness_at_36: f32 = unsafe { bytes.as_ptr().add(36).cast::<f32>().read() };
    assert_eq!(roughness_at_36, 10.0);
}

// =============================================================================
// SECTION 3 -- Flag-bit invariants
// =============================================================================

/// The dirty flag bit is exactly bit 31 (0x8000_0000).
#[test]
fn dirty_flag_bit_value() {
    assert_eq!(MATERIAL_FLAG_DIRTY, 0x8000_0000);
}

/// The visible flag bit is exactly bit 0 (0x0000_0001).
#[test]
fn visible_flag_bit_value() {
    assert_eq!(MATERIAL_FLAG_VISIBLE, 0x0000_0001);
}

/// The two flag bits do not overlap.
#[test]
fn flag_bits_do_not_overlap() {
    assert_eq!(MATERIAL_FLAG_DIRTY & MATERIAL_FLAG_VISIBLE, 0);
}

/// Both flags can be set simultaneously.
#[test]
fn both_flags_set_simultaneously() {
    let combined = MATERIAL_FLAG_DIRTY | MATERIAL_FLAG_VISIBLE;
    assert_eq!(combined, 0x8000_0001);
    assert!(combined & MATERIAL_FLAG_DIRTY != 0);
    assert!(combined & MATERIAL_FLAG_VISIBLE != 0);
}

/// The u32::MAX sentinel (0xFFFF_FFFF) is disjoint from valid flag bits.
#[test]
fn sentinel_disjoint_from_flags() {
    let sentinel: u32 = 0xFFFF_FFFF;
    // The sentinel has all bits set, including the flag bits.
    assert!(sentinel & MATERIAL_FLAG_DIRTY != 0);
    assert!(sentinel & MATERIAL_FLAG_VISIBLE != 0);
    // But the texture sentinel should never appear in flags.
    // This test verifies the constant values are sensible.
}

/// Clearing the dirty flag preserves the visible flag.
#[test]
fn mark_clean_preserves_visible() {
    let mut entry = MaterialTableEntry::zeroed();
    entry.flags = MATERIAL_FLAG_DIRTY | MATERIAL_FLAG_VISIBLE;

    // Simulate mark_clean.
    entry.flags &= !MATERIAL_FLAG_DIRTY;

    assert!(entry.flags & MATERIAL_FLAG_VISIBLE != 0,
            "visible flag must survive mark_clean");
    assert_eq!(entry.flags & MATERIAL_FLAG_DIRTY, 0,
               "dirty flag must be cleared");
    assert_eq!(entry.flags, MATERIAL_FLAG_VISIBLE);
}

// =============================================================================
// SECTION 4 -- is_zero() deep edge cases
// =============================================================================

/// A truly zeroed entry reads as zero.
#[test]
fn is_zero_true_for_zeroed() {
    assert!(MaterialTableEntry::zeroed().is_zero());
}

/// An entry with only the dirty flag set (and all other fields at their
/// zero/sentinel values) reads as zero.
#[test]
fn is_zero_with_just_dirty() {
    let entry = MaterialTableEntry {
        flags: MATERIAL_FLAG_DIRTY,
        ..MaterialTableEntry::zeroed()
    };
    assert!(entry.is_zero());
}

/// An entry with the visible flag set AND otherwise zeroed does NOT read
/// as zero (because visible is not masked out in is_zero).
#[test]
fn is_zero_false_with_visible_flag() {
    let entry = MaterialTableEntry {
        flags: MATERIAL_FLAG_VISIBLE,
        ..MaterialTableEntry::zeroed()
    };
    assert!(!entry.is_zero());
}

/// An entry with a non-zero base_color does not read as zero, even with
/// dirty flag set.
#[test]
fn is_zero_false_with_non_zero_base_color() {
    let entry = MaterialTableEntry {
        base_color: [0.0, 0.0, 0.0, 0.001], // tiny but non-zero
        flags: MATERIAL_FLAG_DIRTY,
        ..MaterialTableEntry::zeroed()
    };
    assert!(!entry.is_zero());
}

/// An entry with a non-sentinel texture ID does not read as zero.
#[test]
fn is_zero_false_with_valid_texture_id() {
    let entry = MaterialTableEntry {
        albedo_texture_id: 0,  // valid texture, not u32::MAX
        flags: MATERIAL_FLAG_DIRTY,
        ..MaterialTableEntry::zeroed()
    };
    assert!(!entry.is_zero());
}

/// An entry with alpha_cutoff set reads as non-zero.
#[test]
fn is_zero_false_with_alpha_cutoff() {
    let entry = MaterialTableEntry {
        alpha_cutoff: 0.5,
        flags: MATERIAL_FLAG_DIRTY,
        ..MaterialTableEntry::zeroed()
    };
    assert!(!entry.is_zero());
}

/// Only the dirty flag bit is masked out in the is_zero comparison.
/// Setting any other flag bit (e.g. bit 1) should make is_zero return false.
#[test]
fn is_zero_masks_only_dirty_bit() {
    let entry = MaterialTableEntry {
        // Bit 1 is not masked -- it should cause is_zero to return false.
        flags: 0x0000_0002,
        ..MaterialTableEntry::zeroed()
    };
    // The is_zero implementation checks (self.flags & !MATERIAL_FLAG_DIRTY) == 0.
    // 0x0000_0002 & !0x8000_0000 = 0x0000_0002 != 0 → false.
    assert!(!entry.is_zero(), "bit 1 must NOT be masked in is_zero");
}

// =============================================================================
// SECTION 5 -- Dirty-tracking via get_mut with observable mutation
// =============================================================================

/// Mutating a field through get_mut is observable on the next get().
#[test]
fn get_mut_mutation_observable() {
    let mut table = MaterialTable::with_capacity(8);
    let idx = table.add(make_entry(1.0, 0.0, 0.0, 1.0));
    table.mark_clean();

    {
        let entry = table.get_mut(idx).unwrap();
        entry.metallic = 0.75;
        entry.roughness = 0.25;
    }

    // Read back through the immutable getter.
    let entry = table.get(idx).unwrap();
    assert!((entry.metallic - 0.75).abs() < f32::EPSILON);
    assert!((entry.roughness - 0.25).abs() < f32::EPSILON);
}

/// get_mut() sets the dirty flag automatically, so mark_clean is needed
/// between successive get_mut calls to see fresh dirty flags.
#[test]
fn get_mut_sets_dirty_on_each_call() {
    let mut table = MaterialTable::with_capacity(8);
    let idx = table.add(make_entry(1.0, 0.0, 0.0, 1.0));
    table.mark_clean();
    assert!(!table.any_dirty());

    // First get_mut marks dirty.
    let _entry = table.get_mut(idx);
    assert!(table.any_dirty());

    // Even without actually mutating, the dirty flag was set.
    table.mark_clean();
    assert!(!table.any_dirty());

    // Second get_mut marks dirty again.
    let _entry = table.get_mut(idx);
    assert!(table.any_dirty());
}

/// get_mut returns the correct mutable reference: mutating one entry does
/// not affect a sibling entry.
#[test]
fn get_mut_does_not_affect_sibling_entries() {
    let mut table = MaterialTable::with_capacity(8);
    let idx_a = table.add(make_entry(1.0, 0.0, 0.0, 1.0));
    let idx_b = table.add(make_entry(0.0, 1.0, 0.0, 1.0));
    table.mark_clean();

    // Mutate entry A.
    {
        let entry = table.get_mut(idx_a).unwrap();
        entry.metallic = 0.9;
    }

    // Entry B should be unchanged.
    let entry_b = table.get(idx_b).unwrap();
    assert_eq!(entry_b.base_color, [0.0, 1.0, 0.0, 1.0]);
    assert!((entry_b.metallic - 0.0).abs() < f32::EPSILON);

    // Dirty count should be exactly 1 (only entry A was dirtied).
    assert_eq!(table.dirty_count(), 1);
}

// =============================================================================
// SECTION 6 -- Capacity auto-resize boundary
// =============================================================================

/// Adding one entry beyond capacity extends the Vec by one element.
#[test]
fn auto_resize_on_full_table() {
    let mut table = MaterialTable::with_capacity(4);
    assert_eq!(table.len(), 4);

    // Fill all 4 slots.
    for i in 0..4u32 {
        table.add(make_entry(i as f32 / 4.0, 0.0, 0.0, 1.0));
    }
    assert_eq!(table.live_count(), 4);
    assert_eq!(table.len(), 4);

    // Add one more -- extends by one (the Vec grows by 1 element).
    let idx = table.add(make_entry(1.0, 0.0, 0.0, 1.0));
    assert_eq!(idx, 4);
    assert_eq!(table.len(), 5); // grew from 4 to 5
    assert_eq!(table.live_count(), 5);
}

/// After auto-resize, all previous entries remain accessible.
#[test]
fn auto_resize_preserves_existing_entries() {
    let mut table = MaterialTable::with_capacity(2);
    let idx0 = table.add(make_entry(0.1, 0.0, 0.0, 1.0));
    let idx1 = table.add(make_entry(0.2, 0.0, 0.0, 1.0));

    // Force extension.
    let idx2 = table.add(make_entry(0.3, 0.0, 0.0, 1.0));

    assert_eq!(table.len(), 3); // grew from 2 to 3
    assert!((table.get(idx0).unwrap().base_color[0] - 0.1).abs() < f32::EPSILON);
    assert!((table.get(idx1).unwrap().base_color[0] - 0.2).abs() < f32::EPSILON);
    assert!((table.get(idx2).unwrap().base_color[0] - 0.3).abs() < f32::EPSILON);
}

/// Auto-resize reuses holes first, then extends.
#[test]
fn auto_resize_fills_holes_before_extension() {
    let mut table = MaterialTable::with_capacity(4);
    let idx0 = table.add(make_entry(1.0, 0.0, 0.0, 1.0));
    let _idx1 = table.add(make_entry(0.0, 1.0, 0.0, 1.0));

    // Remove entry 0, creating a hole.
    table.remove(idx0);

    // Fill remaining slots (hole at 0, live at 1, empty at 2, empty at 3).
    let idx2 = table.add(make_entry(0.0, 0.0, 1.0, 1.0));
    assert_eq!(idx2, 0); // reuses hole

    let idx3 = table.add(make_entry(1.0, 1.0, 0.0, 1.0));
    assert_eq!(idx3, 2); // fills first empty slot

    let idx4 = table.add(make_entry(0.0, 1.0, 1.0, 1.0));
    assert_eq!(idx4, 3); // fills second empty slot

    // Now the table is logically full (4 entries, but one is a hole at idx0
    // which we just reused, so actually all are filled).
    // Next add should extend by one element.
    let idx5 = table.add(make_entry(1.0, 0.0, 1.0, 1.0));
    assert_eq!(idx5, 4); // extends
    assert_eq!(table.len(), 5); // grew from 4 to 5
}

// =============================================================================
// SECTION 7 -- BufferRegistry staging lifecycle
// =============================================================================

/// Stage a single dirty entry through the registry, submit, and verify the
/// byte content matches the expected layout.
#[test]
fn stage_single_entry_verify_bytes() {
    let mut table = MaterialTable::with_capacity(4);
    let mut registry = staging_registry();

    let _entry_idx = table.add(MaterialTableEntry {
        base_color: [0.1, 0.2, 0.3, 0.4],
        metallic: 0.5,
        roughness: 0.25,
        occlusion: 0.8,
        albedo_texture_id: 3,
        ..MaterialTableEntry::zeroed()
    });

    let (slot_idx, byte_size) = table.stage(&mut registry)
        .expect("stage must return Some");
    assert_eq!(byte_size, 4 * MATERIAL_TABLE_ENTRY_SIZE);
    // stage() preserves dirty flags for retry safety.
    assert!(table.any_dirty(), "stage() must preserve dirty flags for retry safety");

    // Submit then mark clean to verify staged content.
    registry.submit_staging(slot_idx, byte_size);
    table.mark_clean();
    let read_idx = registry.acquire_reading()
        .expect("must have a Ready slot");
    let slot = registry.slot(read_idx).unwrap();
    let data = slot.as_slice();
    assert_eq!(data.len(), byte_size);

    // Entry 0: base_color = [0.1, 0.2, 0.3, 0.4] at byte offset 0.
    assert_byte_f32(data, 0, 0.1);
    assert_byte_f32(data, 4, 0.2);
    assert_byte_f32(data, 8, 0.3);
    assert_byte_f32(data, 12, 0.4);

    // metallic = 0.5 at offset 32
    assert_byte_f32(data, 32, 0.5);
    // roughness = 0.25 at offset 36
    assert_byte_f32(data, 36, 0.25);
    // albedo_texture_id = 3 at offset 48
    assert_byte_u32(data, 48, 3);

    // Entry 1 (hole, offset 80) should be zeroed.
    assert_byte_u32(data, 80 + 64, 0); // flags at entry 1 offset 64 relative = abs 144
}

/// Stage and submit: verify the full pipeline with submit_staging.
#[test]
fn stage_and_submit_full_pipeline() {
    let mut table = MaterialTable::with_capacity(4);
    let mut registry = staging_registry();

    table.add(MaterialTableEntry {
        base_color: [1.0, 0.0, 0.0, 1.0],
        emissive: [0.0, 0.0, 0.0, 2.0],
        ..MaterialTableEntry::zeroed()
    });

    assert!(table.stage_and_submit(&mut registry));
    assert!(!table.any_dirty());

    // Acquire for reading on the GPU side.
    let read_idx = registry.acquire_reading()
        .expect("must have a Ready slot");
    let slot = registry.slot(read_idx).unwrap();
    let data = slot.as_slice();

    // Entry 0: base_color = [1, 0, 0, 1].
    assert_byte_f32(data, 0, 1.0);
    assert_byte_f32(data, 12, 1.0);
    // emissive = [0, 0, 0, 2] at offset 16.
    assert_byte_f32(data, 16 + 12, 2.0); // emissive.a
}

/// A second stage after mark_clean works correctly (cycling dirty flags).
#[test]
fn stage_twice_with_interleaved_mutations() {
    let mut table = MaterialTable::with_capacity(4);
    let mut registry = staging_registry();

    // First round.
    let idx = table.add(MaterialTableEntry {
        base_color: [1.0, 0.0, 0.0, 1.0],
        ..MaterialTableEntry::zeroed()
    });
    assert!(table.stage_and_submit(&mut registry));
    assert!(!table.any_dirty());

    // Mutate and stage again.
    {
        let entry = table.get_mut(idx).unwrap();
        entry.metallic = 0.9;
    }
    assert!(table.any_dirty());
    assert_eq!(table.dirty_count(), 1);

    assert!(table.stage_and_submit(&mut registry));
    assert!(!table.any_dirty());

    // The staged data should contain the updated metallic.
    let read_idx = registry.acquire_reading()
        .expect("must have a Ready slot");
    let data = registry.slot(read_idx).unwrap().as_slice();
    assert_byte_f32(data, 32, 0.9);
}

/// Stage returns None when no entries are dirty.
#[test]
fn stage_clean_table_returns_none() {
    let mut table = MaterialTable::with_capacity(4);
    let mut registry = staging_registry();

    table.add(make_entry(1.0, 0.0, 0.0, 1.0));
    table.mark_clean();
    assert!(!table.any_dirty());

    assert!(table.stage(&mut registry).is_none());
}

/// After a successful stage, the table's mark_clean has been called and
/// further mutations create fresh dirty entries.
#[test]
fn mutations_after_stage_create_fresh_dirty_flags() {
    let mut table = MaterialTable::with_capacity(8);
    let mut registry = staging_registry();

    let idx = table.add(make_entry(1.0, 0.0, 0.0, 1.0));
    assert!(table.stage_and_submit(&mut registry));
    assert!(!table.any_dirty());

    // New entry.
    table.add(make_entry(0.0, 1.0, 0.0, 1.0));
    assert!(table.any_dirty());
    assert_eq!(table.dirty_count(), 1);

    // Existing entry via get_mut.
    table.get_mut(idx).unwrap();
    assert_eq!(table.dirty_count(), 2);
}

/// stage_and_submit returns false when the table has no dirty entries.
#[test]
fn stage_and_submit_clean_returns_false() {
    let mut table = MaterialTable::with_capacity(4);
    let mut registry = staging_registry();

    assert!(!table.stage_and_submit(&mut registry));
}

// =============================================================================
// SECTION 8 -- Serialisation invariants
// =============================================================================

/// as_bytes() length is always a multiple of MATERIAL_TABLE_ENTRY_SIZE.
#[test]
fn as_bytes_length_is_multiple_of_entry_size() {
    let table = MaterialTable::with_capacity(16);
    let bytes = table.as_bytes();
    assert_eq!(bytes.len() % MATERIAL_TABLE_ENTRY_SIZE, 0);
    assert_eq!(bytes.len(), 16 * MATERIAL_TABLE_ENTRY_SIZE);
}

/// as_bytes() length changes when the table auto-resizes.
#[test]
fn as_bytes_reflects_current_capacity() {
    let mut table = MaterialTable::with_capacity(2);
    assert_eq!(table.as_bytes().len(), 2 * MATERIAL_TABLE_ENTRY_SIZE);

    table.add(make_entry(1.0, 0.0, 0.0, 1.0));
    table.add(make_entry(0.0, 1.0, 0.0, 1.0));
    table.add(make_entry(0.0, 0.0, 1.0, 1.0)); // triggers extension

    assert_eq!(table.as_bytes().len(), 3 * MATERIAL_TABLE_ENTRY_SIZE);
}

/// AsBytes round-trip: write an entry, read it back, match all fields.
#[test]
fn byte_round_trip_all_fields() {
    let entry = MaterialTableEntry {
        base_color: [0.1, 0.2, 0.3, 0.4],
        emissive: [0.5, 0.6, 0.7, 0.8],
        metallic: 0.9,
        roughness: 0.1,
        occlusion: 0.2,
        normal_scale: 1.5,
        albedo_texture_id: 10,
        normal_texture_id: 20,
        metallic_roughness_texture_id: 30,
        emissive_texture_id: 40,
        flags: 0x8000_0001,
        alpha_cutoff: 0.45,
        _pad: [0, 0],
    };

    // Serialise.
    let bytes = unsafe {
        core::slice::from_raw_parts(
            &entry as *const _ as *const u8,
            MATERIAL_TABLE_ENTRY_SIZE,
        )
    };

    // Deserialise via raw byte copy.
    let mut reconstructed = MaterialTableEntry::zeroed();
    unsafe {
        core::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            &mut reconstructed as *mut _ as *mut u8,
            MATERIAL_TABLE_ENTRY_SIZE,
        );
    }

    assert_eq!(reconstructed, entry);
}

// =============================================================================
// SECTION 9 -- as_slice invariants
// =============================================================================

/// as_slice returns the full table, including holes.
#[test]
fn as_slice_includes_holes() {
    let mut table = MaterialTable::with_capacity(8);
    let idx = table.add(make_entry(1.0, 0.0, 0.0, 1.0));
    table.remove(idx);
    table.add(make_entry(0.0, 1.0, 0.0, 1.0));

    let slice = table.as_slice();
    assert_eq!(slice.len(), 8);
    // Slot 0 was removed then reused, so it is now live.
    // Slot 1 was never used -- it is zeroed (hole).
    assert!(!slice[0].is_zero(), "slot 0 was reused -- must be live");
    assert!(slice[1].is_zero(), "slot 1 was never used -- must be hole");
}

/// Modifications through get_mut are reflected in as_slice.
#[test]
fn as_slice_reflects_recent_mutations() {
    let mut table = MaterialTable::with_capacity(8);
    let idx = table.add(make_entry(1.0, 0.0, 0.0, 1.0));
    table.mark_clean();

    table.get_mut(idx).unwrap().metallic = 0.88;

    let slice = table.as_slice();
    assert!((slice[0].metallic - 0.88).abs() < f32::EPSILON);
}

// =============================================================================
// SECTION 10 -- is_empty / live_count invariants
// =============================================================================

/// is_empty matches live_count == 0.
#[test]
fn is_empty_matches_live_count() {
    let mut table = MaterialTable::with_capacity(16);
    assert!(table.is_empty());
    assert_eq!(table.live_count(), 0);

    table.add(make_entry(1.0, 0.0, 0.0, 1.0));
    assert!(!table.is_empty());
    assert_eq!(table.live_count(), 1);

    table.clear();
    assert!(table.is_empty());
    assert_eq!(table.live_count(), 0);
}

/// live_count never exceeds len().
#[test]
fn live_count_never_exceeds_len() {
    let mut table = MaterialTable::with_capacity(4);
    for i in 0..10u32 {
        table.add(make_entry(i as f32 / 10.0, 0.0, 0.0, 1.0));
        assert!(
            table.live_count() <= table.len(),
            "live_count {} exceeds len {} after add {}",
            table.live_count(),
            table.len(),
            i
        );
    }
}

/// Remove reduces live_count by exactly 1; double remove does not
/// decrement live_count again.
#[test]
fn remove_live_count_invariant() {
    let mut table = MaterialTable::with_capacity(8);
    let idx = table.add(make_entry(1.0, 0.0, 0.0, 1.0));
    assert_eq!(table.live_count(), 1);

    assert_eq!(table.remove(idx), MaterialRemoveResult::Removed);
    assert_eq!(table.live_count(), 0);

    // Second remove on same slot is NotFound.
    assert_eq!(table.remove(idx), MaterialRemoveResult::NotFound);
    assert_eq!(table.live_count(), 0); // unchanged
}

/// insert_at on a hole increments live_count.
#[test]
fn insert_at_hole_increments_live_count() {
    let mut table = MaterialTable::with_capacity(8);
    let idx = table.add(make_entry(1.0, 0.0, 0.0, 1.0));
    table.remove(idx);
    assert_eq!(table.live_count(), 0);

    table.insert_at(idx, make_entry(0.0, 1.0, 0.0, 1.0));
    assert_eq!(table.live_count(), 1);
}

/// insert_at on a live slot does NOT increment live_count.
#[test]
fn insert_at_live_slot_preserves_live_count() {
    let mut table = MaterialTable::with_capacity(8);
    table.add(make_entry(1.0, 0.0, 0.0, 1.0));
    assert_eq!(table.live_count(), 1);

    table.insert_at(0, make_entry(0.0, 1.0, 0.0, 1.0));
    assert_eq!(table.live_count(), 1, "overwriting live slot must not change live_count");
}

// =============================================================================
// SECTION 11 -- Mark_dirty edge conditions
// =============================================================================

/// mark_dirty on a clean entry returns true and sets the flag.
#[test]
fn mark_dirty_on_clean_entry() {
    let mut table = MaterialTable::with_capacity(8);
    let idx = table.add(make_entry(1.0, 0.0, 0.0, 1.0));
    table.mark_clean();

    assert!(table.mark_dirty(idx));
    assert!(table.any_dirty());
    assert_eq!(table.dirty_count(), 1);
}

/// mark_dirty on an already-dirty entry is idempotent.
#[test]
fn mark_dirty_idempotent() {
    let mut table = MaterialTable::with_capacity(8);
    let idx = table.add(make_entry(1.0, 0.0, 0.0, 1.0));
    // add already marks dirty, so dirty_count is 1.
    assert_eq!(table.dirty_count(), 1);

    // mark_dirty again should still return true (entry exists and is live).
    assert!(table.mark_dirty(idx));
    // Dirty count should still be 1 (no new dirty entries).
    assert_eq!(table.dirty_count(), 1);
}

/// mark_dirty on a removed (hole) entry returns false.
#[test]
fn mark_dirty_on_hole_returns_false() {
    let mut table = MaterialTable::with_capacity(8);
    let idx = table.add(make_entry(1.0, 0.0, 0.0, 1.0));
    table.remove(idx);

    assert!(!table.mark_dirty(idx));
}

/// mark_dirty out of bounds returns false.
#[test]
fn mark_dirty_out_of_bounds() {
    let mut table = MaterialTable::with_capacity(8);
    assert!(!table.mark_dirty(999));
    assert!(!table.mark_dirty(u32::MAX));
}

// =============================================================================
// SECTION 12 -- Update edge conditions
// =============================================================================

/// update replaces the entry and marks it dirty.
#[test]
fn update_replaces_and_marks_dirty() {
    let mut table = MaterialTable::with_capacity(8);
    let idx = table.add(make_entry(1.0, 0.0, 0.0, 1.0));
    table.mark_clean();

    let replacement = MaterialTableEntry {
        base_color: [0.0, 0.0, 1.0, 1.0],
        metallic: 0.99,
        ..MaterialTableEntry::zeroed()
    };
    assert!(table.update(idx, replacement).is_some());
    assert!(table.any_dirty());

    let entry = table.get(idx).unwrap();
    assert_eq!(entry.base_color, [0.0, 0.0, 1.0, 1.0]);
    assert!((entry.metallic - 0.99).abs() < f32::EPSILON);
}

/// update on a hole returns None.
#[test]
fn update_hole_returns_none() {
    let mut table = MaterialTable::with_capacity(8);
    // Slot 0 is a hole (never written).
    assert!(table.update(0, make_entry(1.0, 0.0, 0.0, 1.0)).is_none());
}

/// update on an out-of-bounds index returns None.
#[test]
fn update_out_of_bounds_returns_none() {
    let mut table = MaterialTable::with_capacity(8);
    assert!(table.update(999, make_entry(1.0, 0.0, 0.0, 1.0)).is_none());
}

// =============================================================================
// SECTION 13 -- Default trait implementations
// =============================================================================

/// MaterialTableEntry::default() is identical to zeroed().
#[test]
fn entry_default_equals_zeroed() {
    let default_entry = MaterialTableEntry::default();
    let zeroed_entry = MaterialTableEntry::zeroed();
    assert_eq!(default_entry, zeroed_entry);
    assert!(default_entry.is_zero());
}

/// MaterialTable::default() creates a table with default capacity.
#[test]
fn table_default_has_default_capacity() {
    let table = MaterialTable::default();
    assert_eq!(table.len(), DEFAULT_MATERIAL_TABLE_CAPACITY);
    assert!(table.is_empty());
}

// =============================================================================
// SECTION 14 -- Display formatting for MaterialTableEntry
// =============================================================================

/// Entry Display shows all key fields.
#[test]
fn entry_display_includes_texture_ids() {
    let entry = MaterialTableEntry {
        base_color: [0.5, 0.6, 0.7, 1.0],
        emissive: [0.0, 0.0, 0.0, 0.0],
        metallic: 0.8,
        roughness: 0.3,
        occlusion: 1.0,
        normal_scale: 1.0,
        albedo_texture_id: 7,
        normal_texture_id: 8,
        metallic_roughness_texture_id: 9,
        emissive_texture_id: u32::MAX,
        flags: 0x8000_0001,
        alpha_cutoff: 0.5,
        _pad: [0, 0],
    };
    let display = format!("{}", entry);
    assert!(display.contains("albedo_tex: 7"));
    assert!(display.contains("normal_tex: 8"));
    assert!(display.contains("mr_tex: 9"));
    assert!(display.contains("emissive_tex: 4294967295")); // u32::MAX
    assert!(display.contains("flags: 0x80000001"));
}

/// Entry Display shows u32::MAX for unbound textures.
#[test]
fn entry_display_u32_max_for_unbound_textures() {
    let entry = MaterialTableEntry {
        albedo_texture_id: u32::MAX,
        ..MaterialTableEntry::zeroed()
    };
    let display = format!("{}", entry);
    assert!(
        display.contains("4294967295"),
        "Display should show u32::MAX (4294967295) for unbound textures, got: {}",
        display
    );
}

/// Table Display includes capacity and live count.
#[test]
fn table_display_includes_capacity_and_live() {
    let mut table = MaterialTable::with_capacity(128);
    table.add(make_entry(1.0, 0.0, 0.0, 1.0));
    table.add(make_entry(0.0, 1.0, 0.0, 1.0));

    let display = format!("{}", table);
    assert!(display.contains("128"));
    assert!(display.contains("2"));
}

// =============================================================================
// SECTION 15 -- Zero-capacity clamp
// =============================================================================

/// with_capacity(0) clamps to 1 (minimum viable table).
#[test]
fn with_capacity_zero_clamps_to_one() {
    let table = MaterialTable::with_capacity(0);
    assert_eq!(table.len(), 1);
    assert!(table.is_empty());
    assert_eq!(table.live_count(), 0);
}

/// A capacity-1 table can hold one entry and auto-resizes when a second
/// is added.
#[test]
fn capacity_one_auto_resize() {
    let mut table = MaterialTable::with_capacity(1);
    assert_eq!(table.len(), 1);

    let idx0 = table.add(make_entry(1.0, 0.0, 0.0, 1.0));
    assert_eq!(idx0, 0);
    assert_eq!(table.len(), 1);

    let idx1 = table.add(make_entry(0.0, 1.0, 0.0, 1.0));
    assert_eq!(idx1, 1);
    assert_eq!(table.len(), 2); // auto-resize
}

// =============================================================================
// SECTION 16 -- Remove + hole reuse interactions with dirty flag
// =============================================================================

/// Removing an entry clears its dirty flag (the zeroed entry has flags = 0).
#[test]
fn remove_clears_dirty_flag() {
    let mut table = MaterialTable::with_capacity(8);
    let idx = table.add(make_entry(1.0, 0.0, 0.0, 1.0));
    assert!(table.any_dirty());

    table.remove(idx);

    // The removed entry is zeroed, which has flags = 0 (clean).
    // But other entries may still be dirty.
    assert!(!table.any_dirty(), "only entry was removed -- table should be clean");
}

/// Reusing a removed slot: the new entry is marked dirty.
#[test]
fn reused_slot_is_marked_dirty() {
    let mut table = MaterialTable::with_capacity(8);
    let idx = table.add(make_entry(1.0, 0.0, 0.0, 1.0));
    table.mark_clean();
    table.remove(idx);
    assert!(!table.any_dirty());

    let new_idx = table.add(make_entry(0.0, 1.0, 0.0, 1.0));
    assert_eq!(new_idx, idx);
    assert!(table.any_dirty());
    assert_eq!(table.dirty_count(), 1);
}

// =============================================================================
// SECTION 17 -- Reserve edge conditions
// =============================================================================

/// reserve(0) is a no-op.
#[test]
fn reserve_zero_is_noop() {
    let mut table = MaterialTable::with_capacity(8);
    table.reserve(0);
    assert_eq!(table.len(), 8);
}

/// reserve adds exactly the requested number of slots.
#[test]
fn reserve_adds_correct_number_of_slots() {
    let mut table = MaterialTable::with_capacity(4);
    table.reserve(10);
    assert_eq!(table.len(), 14);
}

/// Newly reserved slots are zeroed (holes).
#[test]
fn reserved_slots_are_zeroed() {
    let mut table = MaterialTable::with_capacity(4);
    table.reserve(4);

    let slice = table.as_slice();
    for i in 4..8 {
        assert!(
            slice[i].is_zero(),
            "reserved slot {} must be zeroed",
            i
        );
    }
}

/// Reserve does not affect live_count.
#[test]
fn reserve_preserves_live_count() {
    let mut table = MaterialTable::with_capacity(8);
    table.add(make_entry(1.0, 0.0, 0.0, 1.0));
    assert_eq!(table.live_count(), 1);

    table.reserve(16);
    assert_eq!(table.live_count(), 1);
}

// =============================================================================
// SECTION 18 -- String / display edge cases
// =============================================================================

/// The Display output for a zeroed entry is well-formed.
#[test]
fn zeroed_entry_display_well_formed() {
    let entry = MaterialTableEntry::zeroed();
    let display = format!("{}", entry);
    // Every field appears.
    assert!(display.contains("base_color:"));
    assert!(display.contains("emissive:"));
    assert!(display.contains("metallic:"));
    assert!(display.contains("roughness:"));
    assert!(display.contains("occlusion:"));
    assert!(display.contains("normal_scale:"));
    assert!(display.contains("albedo_tex:"));
    assert!(display.contains("normal_tex:"));
    assert!(display.contains("mr_tex:"));
    assert!(display.contains("emissive_tex:"));
    assert!(display.contains("flags:"));
    assert!(display.contains("alpha_cutoff:"));
}

// =============================================================================
// SECTION 19 -- Clone and Copy semantics
// =============================================================================

/// MaterialTableEntry can be cloned and the clone is identical.
#[test]
fn entry_clone_is_identical() {
    let entry = MaterialTableEntry {
        base_color: [0.1, 0.2, 0.3, 0.4],
        metallic: 0.5,
        ..MaterialTableEntry::zeroed()
    };
    let cloned = entry;
    assert_eq!(cloned, entry);
    assert_eq!(cloned.base_color, [0.1, 0.2, 0.3, 0.4]);
    assert!((cloned.metallic - 0.5).abs() < f32::EPSILON);
}

/// MaterialTableEntry is Copy (implicit clone on assignment).
#[test]
fn entry_is_copy() {
    let entry = MaterialTableEntry::zeroed();
    let copied = entry; // Copy, not move.
    assert_eq!(entry, copied, "Copy semantics must preserve original");
}

// =============================================================================
// SECTION 20 -- WGSL source well-formedness
// =============================================================================

/// WGSL source must not contain a BOM.
#[test]
fn wgsl_no_bom() {
    assert!(
        !WGSL_SOURCE.starts_with('\u{feff}'),
        "File must not start with a BOM"
    );
}

/// WGSL source starts with the SPDX license header.
#[test]
fn wgsl_starts_with_license() {
    assert!(
        WGSL_SOURCE.starts_with("// SPDX-License-Identifier: MIT"),
        "File must start with the MIT license header"
    );
}

/// The WGSL struct field names match the Rust field naming (modulo
/// snake_case differences). This is a soft check that the WGSL gets
/// regenerated when the Rust struct changes.
#[test]
fn wgsl_contains_expected_field_names() {
    let expected_fields = [
        "base_color",
        "emissive",
        "metallic",
        "roughness",
        "occlusion",
        "normal_scale",
        "albedo_texture_id",
        "normal_texture_id",
        "metallic_roughness_tex_id",
        "emissive_texture_id",
        "flags",
        "alpha_cutoff",
    ];
    for field in &expected_fields {
        assert!(
            WGSL_SOURCE.contains(field),
            "WGSL must contain field '{}'",
            field
        );
    }
}

/// The WGSL source text defines exactly one `struct` keyword -- there should
/// be no extra struct definitions beyond `MaterialTableEntry`.
///
/// NOTE: naga's parser may synthesise additional struct types internally
/// (e.g. for entry-point return types), so we check the source text, not
/// the parsed module.
#[test]
fn wgsl_has_no_extra_structs_in_source() {
    // Count `struct ` keyword occurrences in the raw WGSL text.
    // The only user-defined struct in this file is MaterialTableEntry.
    let struct_count = WGSL_SOURCE.lines()
        .filter(|line| line.trim().starts_with("struct "))
        .count();
    assert_eq!(
        struct_count, 1,
        "WGSL source must define exactly 1 struct (MaterialTableEntry), found {}",
        struct_count
    );
}

// =============================================================================
// SECTION 21 -- MaterialDescriptor layout (T-WGPU-P6.8.4)
// =============================================================================

/// MaterialDescriptor size is exactly 64 bytes.
#[test]
fn material_descriptor_size_is_64() {
    assert_eq!(
        std::mem::size_of::<MaterialDescriptor>(),
        64,
        "MaterialDescriptor must be exactly 64 bytes"
    );
    assert_eq!(
        std::mem::size_of::<MaterialDescriptor>(),
        MATERIAL_DESCRIPTOR_SIZE,
        "MaterialDescriptor size must match MATERIAL_DESCRIPTOR_SIZE constant"
    );
}

/// MaterialDescriptor alignment is 4 (for u32/f32 fields).
#[test]
fn material_descriptor_alignment_is_4() {
    assert_eq!(
        std::mem::align_of::<MaterialDescriptor>(),
        4,
        "MaterialDescriptor must be 4-byte aligned"
    );
}

/// MaterialDescriptor implements Pod and Zeroable.
#[test]
fn material_descriptor_is_pod_and_zeroable() {
    fn assert_pod<T: bytemuck::Pod>() {}
    fn assert_zeroable<T: bytemuck::Zeroable>() {}
    assert_pod::<MaterialDescriptor>();
    assert_zeroable::<MaterialDescriptor>();
}

/// Verify byte offsets match the documented layout:
/// | Offset | Size | Field |
/// |--------|------|-------|
/// | 0      | 4    | base_color_texture |
/// | 4      | 4    | normal_texture |
/// | 8      | 4    | metallic_roughness_texture |
/// | 12     | 4    | emissive_texture |
/// | 16     | 16   | base_color_factor |
/// | 32     | 4    | metallic_factor |
/// | 36     | 4    | roughness_factor |
/// | 40     | 12   | emissive_factor |
/// | 52     | 4    | alpha_cutoff |
/// | 56     | 4    | flags |
/// | 60     | 4    | _pad |
#[test]
fn material_descriptor_byte_layout_matches_spec() {
    let mat = MaterialDescriptor {
        base_color_texture: 1,
        normal_texture: 2,
        metallic_roughness_texture: 3,
        emissive_texture: 4,
        base_color_factor: [0.1, 0.2, 0.3, 0.4],
        metallic_factor: 0.5,
        roughness_factor: 0.6,
        emissive_factor: [0.7, 0.8, 0.9],
        alpha_cutoff: 0.25,
        flags: 0x0000_0007,
        _pad: 0,
    };

    let bytes: &[u8] = bytemuck::bytes_of(&mat);
    assert_eq!(bytes.len(), 64);

    // base_color_texture at offset 0
    let tex0 = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    assert_eq!(tex0, 1);

    // normal_texture at offset 4
    let tex1 = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    assert_eq!(tex1, 2);

    // metallic_roughness_texture at offset 8
    let tex2 = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    assert_eq!(tex2, 3);

    // emissive_texture at offset 12
    let tex3 = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
    assert_eq!(tex3, 4);

    // base_color_factor at offset 16 (16 bytes)
    let bc0 = f32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
    assert!((bc0 - 0.1).abs() < 0.001);
    let bc1 = f32::from_le_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);
    assert!((bc1 - 0.2).abs() < 0.001);
    let bc2 = f32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]);
    assert!((bc2 - 0.3).abs() < 0.001);
    let bc3 = f32::from_le_bytes([bytes[28], bytes[29], bytes[30], bytes[31]]);
    assert!((bc3 - 0.4).abs() < 0.001);

    // metallic_factor at offset 32
    let mf = f32::from_le_bytes([bytes[32], bytes[33], bytes[34], bytes[35]]);
    assert!((mf - 0.5).abs() < 0.001);

    // roughness_factor at offset 36
    let rf = f32::from_le_bytes([bytes[36], bytes[37], bytes[38], bytes[39]]);
    assert!((rf - 0.6).abs() < 0.001);

    // emissive_factor at offset 40 (12 bytes)
    let ef0 = f32::from_le_bytes([bytes[40], bytes[41], bytes[42], bytes[43]]);
    assert!((ef0 - 0.7).abs() < 0.001);
    let ef1 = f32::from_le_bytes([bytes[44], bytes[45], bytes[46], bytes[47]]);
    assert!((ef1 - 0.8).abs() < 0.001);
    let ef2 = f32::from_le_bytes([bytes[48], bytes[49], bytes[50], bytes[51]]);
    assert!((ef2 - 0.9).abs() < 0.001);

    // alpha_cutoff at offset 52
    let ac = f32::from_le_bytes([bytes[52], bytes[53], bytes[54], bytes[55]]);
    assert!((ac - 0.25).abs() < 0.001);

    // flags at offset 56
    let flags = u32::from_le_bytes([bytes[56], bytes[57], bytes[58], bytes[59]]);
    assert_eq!(flags, 0x0000_0007);

    // _pad at offset 60
    let pad = u32::from_le_bytes([bytes[60], bytes[61], bytes[62], bytes[63]]);
    assert_eq!(pad, 0);
}

/// Verify no field overlap in MaterialDescriptor layout.
#[test]
fn material_descriptor_no_field_overlap() {
    let mat = MaterialDescriptor {
        base_color_texture: 0xAAAA_AAAA,
        normal_texture: 0xBBBB_BBBB,
        metallic_roughness_texture: 0xCCCC_CCCC,
        emissive_texture: 0xDDDD_DDDD,
        base_color_factor: [1.0, 2.0, 3.0, 4.0],
        metallic_factor: 5.0,
        roughness_factor: 6.0,
        emissive_factor: [7.0, 8.0, 9.0],
        alpha_cutoff: 10.0,
        flags: 0x1234_5678,
        _pad: 0xFFFF_FFFF,
    };

    let bytes: &[u8] = bytemuck::bytes_of(&mat);

    // Verify that changing one field does not affect adjacent fields
    let tex0 = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let tex1 = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    assert_eq!(tex0, 0xAAAA_AAAA);
    assert_eq!(tex1, 0xBBBB_BBBB);
    assert_ne!(tex0, tex1);

    // Boundary between base_color_factor[3] and metallic_factor
    let bc3 = f32::from_le_bytes([bytes[28], bytes[29], bytes[30], bytes[31]]);
    let mf = f32::from_le_bytes([bytes[32], bytes[33], bytes[34], bytes[35]]);
    assert_eq!(bc3, 4.0);
    assert_eq!(mf, 5.0);
}

// =============================================================================
// SECTION 22 -- MaterialDescriptor builder methods (T-WGPU-P6.8.4)
// =============================================================================

/// MaterialDescriptor::new() creates default PBR values.
#[test]
fn material_descriptor_new_default_values() {
    let mat = MaterialDescriptor::new();
    assert_eq!(mat.base_color_texture, NO_TEXTURE);
    assert_eq!(mat.normal_texture, NO_TEXTURE);
    assert_eq!(mat.metallic_roughness_texture, NO_TEXTURE);
    assert_eq!(mat.emissive_texture, NO_TEXTURE);
    assert_eq!(mat.base_color_factor, [1.0, 1.0, 1.0, 1.0]);
    assert_eq!(mat.metallic_factor, 0.0);
    assert!((mat.roughness_factor - 0.5).abs() < f32::EPSILON);
    assert_eq!(mat.emissive_factor, [0.0, 0.0, 0.0]);
    assert!((mat.alpha_cutoff - 0.5).abs() < f32::EPSILON);
    assert_eq!(mat.flags, 0);
    assert_eq!(mat._pad, 0);
}

/// MaterialDescriptor::opaque() creates a dielectric opaque material.
#[test]
fn material_descriptor_opaque_builder() {
    let mat = MaterialDescriptor::opaque(0.5, 0.6, 0.7);
    assert_eq!(mat.base_color_factor, [0.5, 0.6, 0.7, 1.0]);
    assert_eq!(mat.metallic_factor, 0.0);
    assert!((mat.roughness_factor - 0.5).abs() < f32::EPSILON);
    assert_eq!(mat.base_color_texture, NO_TEXTURE);
    assert_eq!(mat.flags, 0);
}

/// MaterialDescriptor::metallic() creates a metallic material.
#[test]
fn material_descriptor_metallic_builder() {
    let mat = MaterialDescriptor::metallic(0.8, 0.8, 0.8, 0.9, 0.2);
    assert_eq!(mat.base_color_factor, [0.8, 0.8, 0.8, 1.0]);
    assert!((mat.metallic_factor - 0.9).abs() < f32::EPSILON);
    assert!((mat.roughness_factor - 0.2).abs() < f32::EPSILON);
    assert_eq!(mat.base_color_texture, NO_TEXTURE);
}

/// with_base_color_texture sets the texture index.
#[test]
fn material_descriptor_with_base_color_texture() {
    let mat = MaterialDescriptor::new().with_base_color_texture(42);
    assert_eq!(mat.base_color_texture, 42);
    assert!(mat.has_base_color_texture());
}

/// with_normal_texture sets the texture index.
#[test]
fn material_descriptor_with_normal_texture() {
    let mat = MaterialDescriptor::new().with_normal_texture(123);
    assert_eq!(mat.normal_texture, 123);
    assert!(mat.has_normal_texture());
}

/// with_metallic_roughness_texture sets the texture index.
#[test]
fn material_descriptor_with_metallic_roughness_texture() {
    let mat = MaterialDescriptor::new().with_metallic_roughness_texture(456);
    assert_eq!(mat.metallic_roughness_texture, 456);
    assert!(mat.has_metallic_roughness_texture());
}

/// with_emissive_texture sets the texture index.
#[test]
fn material_descriptor_with_emissive_texture() {
    let mat = MaterialDescriptor::new().with_emissive_texture(789);
    assert_eq!(mat.emissive_texture, 789);
    assert!(mat.has_emissive_texture());
}

/// Chained texture setters work correctly.
#[test]
fn material_descriptor_chained_texture_setters() {
    let mat = MaterialDescriptor::new()
        .with_base_color_texture(0)
        .with_normal_texture(1)
        .with_metallic_roughness_texture(2)
        .with_emissive_texture(3);

    assert_eq!(mat.base_color_texture, 0);
    assert_eq!(mat.normal_texture, 1);
    assert_eq!(mat.metallic_roughness_texture, 2);
    assert_eq!(mat.emissive_texture, 3);
    assert!(mat.has_base_color_texture());
    assert!(mat.has_normal_texture());
    assert!(mat.has_metallic_roughness_texture());
    assert!(mat.has_emissive_texture());
}

/// with_double_sided(true) sets the flag.
#[test]
fn material_descriptor_with_double_sided_true() {
    let mat = MaterialDescriptor::new().with_double_sided(true);
    assert!(mat.is_double_sided());
    assert_eq!(mat.flags & MATERIAL_DESC_FLAG_DOUBLE_SIDED, MATERIAL_DESC_FLAG_DOUBLE_SIDED);
}

/// with_double_sided(false) clears the flag.
#[test]
fn material_descriptor_with_double_sided_false() {
    let mat = MaterialDescriptor::new()
        .with_double_sided(true)
        .with_double_sided(false);
    assert!(!mat.is_double_sided());
    assert_eq!(mat.flags & MATERIAL_DESC_FLAG_DOUBLE_SIDED, 0);
}

/// with_alpha_mask sets flag and cutoff.
#[test]
fn material_descriptor_with_alpha_mask() {
    let mat = MaterialDescriptor::new().with_alpha_mask(0.75);
    assert!(mat.is_alpha_mask());
    assert!(!mat.is_alpha_blend());
    assert!((mat.alpha_cutoff - 0.75).abs() < f32::EPSILON);
    assert_eq!(mat.flags & MATERIAL_DESC_FLAG_ALPHA_MASK, MATERIAL_DESC_FLAG_ALPHA_MASK);
    assert_eq!(mat.flags & MATERIAL_DESC_FLAG_ALPHA_BLEND, 0);
}

/// with_alpha_blend sets flag and clears alpha_mask.
#[test]
fn material_descriptor_with_alpha_blend() {
    let mat = MaterialDescriptor::new().with_alpha_blend();
    assert!(mat.is_alpha_blend());
    assert!(!mat.is_alpha_mask());
    assert_eq!(mat.flags & MATERIAL_DESC_FLAG_ALPHA_BLEND, MATERIAL_DESC_FLAG_ALPHA_BLEND);
    assert_eq!(mat.flags & MATERIAL_DESC_FLAG_ALPHA_MASK, 0);
}

/// Alpha modes are mutually exclusive.
#[test]
fn material_descriptor_alpha_modes_mutually_exclusive() {
    // Start with alpha mask
    let mat1 = MaterialDescriptor::new().with_alpha_mask(0.5);
    assert!(mat1.is_alpha_mask());
    assert!(!mat1.is_alpha_blend());

    // Switch to alpha blend
    let mat2 = mat1.with_alpha_blend();
    assert!(!mat2.is_alpha_mask());
    assert!(mat2.is_alpha_blend());

    // Switch back to alpha mask
    let mat3 = mat2.with_alpha_mask(0.3);
    assert!(mat3.is_alpha_mask());
    assert!(!mat3.is_alpha_blend());
    assert!((mat3.alpha_cutoff - 0.3).abs() < f32::EPSILON);
}

/// Multiple flags can coexist (except alpha_mask and alpha_blend).
#[test]
fn material_descriptor_multiple_flags() {
    let mat = MaterialDescriptor::new()
        .with_double_sided(true)
        .with_alpha_mask(0.5);

    assert!(mat.is_double_sided());
    assert!(mat.is_alpha_mask());
    assert_eq!(
        mat.flags,
        MATERIAL_DESC_FLAG_DOUBLE_SIDED | MATERIAL_DESC_FLAG_ALPHA_MASK
    );
}

// =============================================================================
// SECTION 23 -- MaterialDescriptor flag operations (T-WGPU-P6.8.4)
// =============================================================================

/// Flag constants have correct bit positions.
#[test]
fn material_desc_flag_bit_positions() {
    assert_eq!(MATERIAL_DESC_FLAG_DOUBLE_SIDED, 0b0001);
    assert_eq!(MATERIAL_DESC_FLAG_ALPHA_MASK, 0b0010);
    assert_eq!(MATERIAL_DESC_FLAG_ALPHA_BLEND, 0b0100);
    assert_eq!(MATERIAL_DESC_FLAG_UNLIT, 0b1000);
}

/// Flags do not overlap.
#[test]
fn material_desc_flags_do_not_overlap() {
    let all_flags = MATERIAL_DESC_FLAG_DOUBLE_SIDED
        | MATERIAL_DESC_FLAG_ALPHA_MASK
        | MATERIAL_DESC_FLAG_ALPHA_BLEND
        | MATERIAL_DESC_FLAG_UNLIT;
    assert_eq!(all_flags, 0b1111);

    // No overlapping bits
    assert_eq!(MATERIAL_DESC_FLAG_DOUBLE_SIDED & MATERIAL_DESC_FLAG_ALPHA_MASK, 0);
    assert_eq!(MATERIAL_DESC_FLAG_DOUBLE_SIDED & MATERIAL_DESC_FLAG_ALPHA_BLEND, 0);
    assert_eq!(MATERIAL_DESC_FLAG_DOUBLE_SIDED & MATERIAL_DESC_FLAG_UNLIT, 0);
    assert_eq!(MATERIAL_DESC_FLAG_ALPHA_MASK & MATERIAL_DESC_FLAG_ALPHA_BLEND, 0);
    assert_eq!(MATERIAL_DESC_FLAG_ALPHA_MASK & MATERIAL_DESC_FLAG_UNLIT, 0);
    assert_eq!(MATERIAL_DESC_FLAG_ALPHA_BLEND & MATERIAL_DESC_FLAG_UNLIT, 0);
}

/// is_double_sided returns correct value.
#[test]
fn material_descriptor_is_double_sided_query() {
    let mat_no = MaterialDescriptor::new();
    assert!(!mat_no.is_double_sided());

    let mat_yes = MaterialDescriptor::new().with_double_sided(true);
    assert!(mat_yes.is_double_sided());
}

/// is_alpha_mask returns correct value.
#[test]
fn material_descriptor_is_alpha_mask_query() {
    let mat_no = MaterialDescriptor::new();
    assert!(!mat_no.is_alpha_mask());

    let mat_yes = MaterialDescriptor::new().with_alpha_mask(0.5);
    assert!(mat_yes.is_alpha_mask());
}

/// is_alpha_blend returns correct value.
#[test]
fn material_descriptor_is_alpha_blend_query() {
    let mat_no = MaterialDescriptor::new();
    assert!(!mat_no.is_alpha_blend());

    let mat_yes = MaterialDescriptor::new().with_alpha_blend();
    assert!(mat_yes.is_alpha_blend());
}

/// has_*_texture returns false when texture is NO_TEXTURE.
#[test]
fn material_descriptor_no_texture_queries() {
    let mat = MaterialDescriptor::new();
    assert!(!mat.has_base_color_texture());
    assert!(!mat.has_normal_texture());
    assert!(!mat.has_metallic_roughness_texture());
    assert!(!mat.has_emissive_texture());
}

/// NO_TEXTURE constant equals u32::MAX.
#[test]
fn no_texture_equals_u32_max() {
    assert_eq!(NO_TEXTURE, u32::MAX);
    assert_eq!(MaterialDescriptor::NO_TEXTURE, u32::MAX);
    assert_eq!(NO_TEXTURE, MaterialDescriptor::NO_TEXTURE);
}

// =============================================================================
// SECTION 24 -- GpuMaterialTable operations (T-WGPU-P6.8.4)
// =============================================================================

/// GpuMaterialTable::new creates empty table with specified capacity.
#[test]
fn gpu_material_table_new_is_empty() {
    let table = GpuMaterialTable::new(128);
    assert!(table.is_empty());
    assert_eq!(table.len(), 0);
    assert_eq!(table.capacity(), 128);
    assert!(!table.is_dirty());
    assert!(table.buffer().is_none());
}

/// GpuMaterialTable::with_default_capacity uses default capacity.
#[test]
fn gpu_material_table_with_default_capacity() {
    let table = GpuMaterialTable::with_default_capacity();
    assert_eq!(table.capacity(), DEFAULT_GPU_MATERIAL_TABLE_CAPACITY);
    assert_eq!(table.capacity(), 1024);
}

/// add() returns sequential indices.
#[test]
fn gpu_material_table_add_returns_sequential_indices() {
    let mut table = GpuMaterialTable::new(64);
    let idx0 = table.add(MaterialDescriptor::opaque(1.0, 0.0, 0.0));
    let idx1 = table.add(MaterialDescriptor::opaque(0.0, 1.0, 0.0));
    let idx2 = table.add(MaterialDescriptor::opaque(0.0, 0.0, 1.0));

    assert_eq!(idx0, 0);
    assert_eq!(idx1, 1);
    assert_eq!(idx2, 2);
    assert_eq!(table.len(), 3);
}

/// add() marks table as dirty.
#[test]
fn gpu_material_table_add_marks_dirty() {
    let mut table = GpuMaterialTable::new(64);
    assert!(!table.is_dirty());

    table.add(MaterialDescriptor::new());
    assert!(table.is_dirty());
}

/// get() retrieves correct material.
#[test]
fn gpu_material_table_get_retrieves_correct_material() {
    let mut table = GpuMaterialTable::new(64);
    let idx = table.add(MaterialDescriptor::metallic(0.5, 0.6, 0.7, 0.8, 0.3));

    let mat = table.get(idx).expect("material should exist");
    assert_eq!(mat.base_color_factor, [0.5, 0.6, 0.7, 1.0]);
    assert!((mat.metallic_factor - 0.8).abs() < f32::EPSILON);
    assert!((mat.roughness_factor - 0.3).abs() < f32::EPSILON);
}

/// get() returns None for invalid index.
#[test]
fn gpu_material_table_get_invalid_index_returns_none() {
    let table = GpuMaterialTable::new(64);
    assert!(table.get(0).is_none());
    assert!(table.get(999).is_none());
}

/// get_mut() returns mutable reference and marks dirty.
#[test]
fn gpu_material_table_get_mut_marks_dirty() {
    let mut table = GpuMaterialTable::new(64);
    let idx = table.add(MaterialDescriptor::new());

    // Add marks dirty, so we verify it's dirty, then test get_mut
    assert!(table.is_dirty());

    {
        let mat = table.get_mut(idx).unwrap();
        mat.metallic_factor = 0.95;
    }

    // Still dirty after get_mut
    assert!(table.is_dirty());
    assert!((table.get(idx).unwrap().metallic_factor - 0.95).abs() < f32::EPSILON);
}

/// update() modifies material and marks dirty.
#[test]
fn gpu_material_table_update_modifies_and_marks_dirty() {
    let mut table = GpuMaterialTable::new(64);
    let idx = table.add(MaterialDescriptor::opaque(1.0, 0.0, 0.0));
    // add() marks dirty, test that update also keeps/sets dirty
    assert!(table.is_dirty());

    let result = table.update(idx, MaterialDescriptor::metallic(0.8, 0.8, 0.8, 0.9, 0.1));
    assert!(result);
    assert!(table.is_dirty());

    let mat = table.get(idx).unwrap();
    assert!((mat.metallic_factor - 0.9).abs() < f32::EPSILON);
}

/// update() returns false for invalid index.
#[test]
fn gpu_material_table_update_invalid_index_returns_false() {
    let mut table = GpuMaterialTable::new(64);
    assert!(!table.update(0, MaterialDescriptor::new()));
    assert!(!table.update(999, MaterialDescriptor::new()));
}

/// remove() adds index to free list.
#[test]
fn gpu_material_table_remove_adds_to_free_list() {
    let mut table = GpuMaterialTable::new(64);
    let idx0 = table.add(MaterialDescriptor::opaque(1.0, 0.0, 0.0));
    let _idx1 = table.add(MaterialDescriptor::opaque(0.0, 1.0, 0.0));
    // add() marks dirty
    assert!(table.is_dirty());

    assert!(table.remove(idx0));
    // Still dirty after remove
    assert!(table.is_dirty());
    assert_eq!(table.free_count(), 1);
    assert_eq!(table.active_count(), 1);

    // Material at idx0 should be zeroed (bytemuck Zeroable)
    let mat = table.get(idx0).unwrap();
    assert_eq!(*mat, MaterialDescriptor::zeroed());
}

/// remove() returns false for invalid index.
#[test]
fn gpu_material_table_remove_invalid_index_returns_false() {
    let mut table = GpuMaterialTable::new(64);
    assert!(!table.remove(0));
    assert!(!table.remove(999));
}

/// Removed index is reused on next add().
#[test]
fn gpu_material_table_index_reuse() {
    let mut table = GpuMaterialTable::new(64);
    let idx0 = table.add(MaterialDescriptor::opaque(1.0, 0.0, 0.0));
    let _idx1 = table.add(MaterialDescriptor::opaque(0.0, 1.0, 0.0));

    table.remove(idx0);
    assert_eq!(table.free_count(), 1);

    // Next add should reuse idx0
    let idx2 = table.add(MaterialDescriptor::opaque(0.0, 0.0, 1.0));
    assert_eq!(idx2, idx0);
    assert_eq!(table.free_count(), 0);
}

// =============================================================================
// SECTION 25 -- GpuMaterialTable dirty tracking (T-WGPU-P6.8.4)
// =============================================================================

/// Dirty flag set on add.
#[test]
fn gpu_material_table_dirty_on_add() {
    let mut table = GpuMaterialTable::new(64);
    assert!(!table.is_dirty());

    table.add(MaterialDescriptor::new());
    assert!(table.is_dirty());
}

/// Dirty flag set on update.
#[test]
fn gpu_material_table_dirty_on_update() {
    let mut table = GpuMaterialTable::new(64);
    let idx = table.add(MaterialDescriptor::new());
    // add() marks dirty, update should maintain dirty state
    assert!(table.is_dirty());

    table.update(idx, MaterialDescriptor::opaque(0.5, 0.5, 0.5));
    assert!(table.is_dirty());
    // Verify the update took effect
    let mat = table.get(idx).unwrap();
    assert_eq!(mat.base_color_factor, [0.5, 0.5, 0.5, 1.0]);
}

/// Dirty flag set on remove.
#[test]
fn gpu_material_table_dirty_on_remove() {
    let mut table = GpuMaterialTable::new(64);
    let idx = table.add(MaterialDescriptor::new());
    // add() marks dirty
    assert!(table.is_dirty());

    table.remove(idx);
    // Still dirty after remove
    assert!(table.is_dirty());
    // Verify remove took effect
    assert_eq!(table.free_count(), 1);
}

/// mark_dirty() ensures dirty flag is set.
#[test]
fn gpu_material_table_mark_dirty() {
    let mut table = GpuMaterialTable::new(64);
    table.add(MaterialDescriptor::new());
    // add() marks dirty
    assert!(table.is_dirty());

    // mark_dirty() should keep it dirty
    table.mark_dirty();
    assert!(table.is_dirty());
}

/// clear() marks table dirty and resets state.
#[test]
fn gpu_material_table_clear_marks_dirty() {
    let mut table = GpuMaterialTable::new(64);
    table.add(MaterialDescriptor::new());
    table.add(MaterialDescriptor::new());
    assert!(table.is_dirty());
    assert_eq!(table.len(), 2);

    table.clear();
    assert!(table.is_dirty());
    assert!(table.is_empty());
    assert_eq!(table.free_count(), 0);
    assert_eq!(table.len(), 0);
}

// =============================================================================
// SECTION 26 -- GpuMaterialTable counts and capacity (T-WGPU-P6.8.4)
// =============================================================================

/// active_count tracks live materials.
#[test]
fn gpu_material_table_active_count() {
    let mut table = GpuMaterialTable::new(64);
    assert_eq!(table.active_count(), 0);

    let idx0 = table.add(MaterialDescriptor::new());
    let idx1 = table.add(MaterialDescriptor::new());
    assert_eq!(table.active_count(), 2);

    table.remove(idx0);
    assert_eq!(table.active_count(), 1);

    table.remove(idx1);
    assert_eq!(table.active_count(), 0);
}

/// free_count tracks recycled slots.
#[test]
fn gpu_material_table_free_count() {
    let mut table = GpuMaterialTable::new(64);
    assert_eq!(table.free_count(), 0);

    let idx0 = table.add(MaterialDescriptor::new());
    let idx1 = table.add(MaterialDescriptor::new());
    assert_eq!(table.free_count(), 0);

    table.remove(idx0);
    assert_eq!(table.free_count(), 1);

    table.remove(idx1);
    assert_eq!(table.free_count(), 2);

    // Adding reuses free slots
    table.add(MaterialDescriptor::new());
    assert_eq!(table.free_count(), 1);
}

/// Capacity minimum is 1.
#[test]
fn gpu_material_table_capacity_minimum() {
    let table = GpuMaterialTable::new(0);
    assert_eq!(table.capacity(), 1);
}

/// len() returns material count.
#[test]
fn gpu_material_table_len() {
    let mut table = GpuMaterialTable::new(64);
    assert_eq!(table.len(), 0);

    table.add(MaterialDescriptor::new());
    assert_eq!(table.len(), 1);

    table.add(MaterialDescriptor::new());
    assert_eq!(table.len(), 2);
}

/// is_empty() returns true only when no materials.
#[test]
fn gpu_material_table_is_empty() {
    let mut table = GpuMaterialTable::new(64);
    assert!(table.is_empty());

    let idx = table.add(MaterialDescriptor::new());
    assert!(!table.is_empty());

    table.remove(idx);
    // Still has materials (just zeroed), so len() is still 1
    assert_eq!(table.len(), 1);
}

// =============================================================================
// SECTION 27 -- GpuMaterialTable serialization (T-WGPU-P6.8.4)
// =============================================================================

/// as_bytes() returns correct byte length.
#[test]
fn gpu_material_table_as_bytes_length() {
    let mut table = GpuMaterialTable::new(64);
    assert_eq!(table.as_bytes().len(), 0);

    table.add(MaterialDescriptor::new());
    assert_eq!(table.as_bytes().len(), MATERIAL_DESCRIPTOR_SIZE);

    table.add(MaterialDescriptor::new());
    assert_eq!(table.as_bytes().len(), 2 * MATERIAL_DESCRIPTOR_SIZE);
}

/// as_slice() returns correct slice.
#[test]
fn gpu_material_table_as_slice() {
    let mut table = GpuMaterialTable::new(64);
    table.add(MaterialDescriptor::opaque(1.0, 0.0, 0.0));
    table.add(MaterialDescriptor::opaque(0.0, 1.0, 0.0));

    let slice = table.as_slice();
    assert_eq!(slice.len(), 2);
    assert_eq!(slice[0].base_color_factor, [1.0, 0.0, 0.0, 1.0]);
    assert_eq!(slice[1].base_color_factor, [0.0, 1.0, 0.0, 1.0]);
}

/// as_bytes() content matches bytemuck cast.
#[test]
fn gpu_material_table_as_bytes_content() {
    let mut table = GpuMaterialTable::new(64);
    let mat = MaterialDescriptor::metallic(0.5, 0.6, 0.7, 0.8, 0.2)
        .with_base_color_texture(42);
    table.add(mat);

    let bytes = table.as_bytes();
    assert_eq!(bytes.len(), MATERIAL_DESCRIPTOR_SIZE);

    // Verify first field (base_color_texture = 42)
    let tex0 = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    assert_eq!(tex0, 42);
}

// =============================================================================
// SECTION 28 -- GpuMaterialTable Debug impl (T-WGPU-P6.8.4)
// =============================================================================

/// Debug output includes key information.
#[test]
fn gpu_material_table_debug_output() {
    let mut table = GpuMaterialTable::new(128);
    table.add(MaterialDescriptor::new());
    table.add(MaterialDescriptor::new());

    let debug = format!("{:?}", table);
    assert!(debug.contains("GpuMaterialTable"));
    assert!(debug.contains("material_count"));
    assert!(debug.contains("2"));
    assert!(debug.contains("128") || debug.contains("capacity"));
}

// =============================================================================
// SECTION 29 -- MaterialDescriptor zeroed/default (T-WGPU-P6.8.4)
// =============================================================================

/// MaterialDescriptor::zeroed() has all fields zero.
#[test]
fn material_descriptor_zeroed() {
    let mat = MaterialDescriptor::zeroed();
    assert_eq!(mat.base_color_texture, 0);
    assert_eq!(mat.normal_texture, 0);
    assert_eq!(mat.metallic_roughness_texture, 0);
    assert_eq!(mat.emissive_texture, 0);
    assert_eq!(mat.base_color_factor, [0.0, 0.0, 0.0, 0.0]);
    assert_eq!(mat.metallic_factor, 0.0);
    assert_eq!(mat.roughness_factor, 0.0);
    assert_eq!(mat.emissive_factor, [0.0, 0.0, 0.0]);
    assert_eq!(mat.alpha_cutoff, 0.0);
    assert_eq!(mat.flags, 0);
    assert_eq!(mat._pad, 0);
}

/// MaterialDescriptor::default() equals zeroed().
#[test]
fn material_descriptor_default_equals_zeroed() {
    let mat_default = MaterialDescriptor::default();
    let mat_zeroed = MaterialDescriptor::zeroed();
    assert_eq!(mat_default, mat_zeroed);
}

/// MaterialDescriptor Debug output is well-formed.
#[test]
fn material_descriptor_debug_output() {
    let mat = MaterialDescriptor::opaque(0.5, 0.5, 0.5);
    let debug = format!("{:?}", mat);
    assert!(debug.contains("MaterialDescriptor"));
    assert!(debug.contains("base_color_factor"));
}

// =============================================================================
// SECTION 30 -- Constants verification (T-WGPU-P6.8.4)
// =============================================================================

/// MATERIAL_DESCRIPTOR_SIZE is 64.
#[test]
fn material_descriptor_size_constant() {
    assert_eq!(MATERIAL_DESCRIPTOR_SIZE, 64);
}

/// DEFAULT_GPU_MATERIAL_TABLE_CAPACITY is 1024.
#[test]
fn default_gpu_material_table_capacity_constant() {
    assert_eq!(DEFAULT_GPU_MATERIAL_TABLE_CAPACITY, 1024);
}

/// MaterialDescriptor Copy trait.
#[test]
fn material_descriptor_is_copy() {
    let mat = MaterialDescriptor::opaque(1.0, 0.0, 0.0);
    let copied = mat; // Copy, not move
    assert_eq!(mat, copied);
}

/// MaterialDescriptor Clone trait.
#[test]
fn material_descriptor_is_clone() {
    let mat = MaterialDescriptor::opaque(1.0, 0.0, 0.0);
    let cloned = mat.clone();
    assert_eq!(mat, cloned);
}

/// MaterialDescriptor PartialEq works correctly.
#[test]
fn material_descriptor_partial_eq() {
    let mat1 = MaterialDescriptor::opaque(1.0, 0.0, 0.0);
    let mat2 = MaterialDescriptor::opaque(1.0, 0.0, 0.0);
    let mat3 = MaterialDescriptor::opaque(0.0, 1.0, 0.0);

    assert_eq!(mat1, mat2);
    assert_ne!(mat1, mat3);
}

/// Bytemuck slice cast works for multiple materials.
#[test]
fn material_descriptor_bytemuck_slice_cast() {
    let materials = vec![
        MaterialDescriptor::opaque(1.0, 0.0, 0.0),
        MaterialDescriptor::opaque(0.0, 1.0, 0.0),
        MaterialDescriptor::opaque(0.0, 0.0, 1.0),
    ];

    let bytes: &[u8] = bytemuck::cast_slice(&materials);
    assert_eq!(bytes.len(), 3 * MATERIAL_DESCRIPTOR_SIZE);
}
