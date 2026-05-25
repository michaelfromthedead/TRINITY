// Blackbox contract tests for MaterialTable (T-GPU-1.4).
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::gpu_driven::*` -- no internal fields,
// no private methods, no implementation details.
//
// Acceptance criterion (T-GPU-1.4):
//   Materials loaded via asset pipeline auto-append to MaterialTable with
//   dirty flag management.
//
// The MaterialTable guarantees:
//   - add() returns sequential u32 indices that shaders can use for bindless
//     access via the GPU-side `array<MaterialTableEntry>` storage buffer.
//   - add() automatically marks the entry dirty (bit 31 of `flags`).
//   - Entries are retrievable by index via get() / as_slice().
//   - The dirty flag is cleared after staging (`mark_clean()`), enabling
//     incremental uploads.
//   - Removing an entry zeroes it (hole) but preserves all other indices.
//   - Updating an entry in place keeps its index stable.
//   - The table supports `stage()` / `stage_and_submit()` through the
//     triple-buffered BufferRegistry for GPU upload.
//
// Coverage:
//   1.  Auto-append acceptance: sequential add() returns indices 0, 1, 2, ...
//   2.  Dirty flag on add: every new entry is automatically marked dirty
//   3.  Material property preservation: all fields retrievable by index
//   4.  Texture ID sentinel: u32::MAX == no texture bound
//   5.  Full frame loop: add -> stage -> submit -> GPU read-back
//   6.  Incremental loading across multiple frames
//   7.  Dirty flag lifecycle: add -> dirty, stage -> clean, update -> dirty
//   8.  get_mut marks dirty
//   9.  Hole creation and index stability after removal
//  10.  Hole reuse on subsequent add
//  11.  insert_at for specific index assignment
//  12.  Clear and re-add: indices restart from 0
//  13.  Bulk staging integrity through BufferRegistry
//  14.  Dense loading at scale (1000 entries)
//  15.  Empty table stage returns None
//  16.  Display/formatting for table and entries
//  17.  Default capacity and entry size constants
//  18.  Update preserves index
//  19.  Zero capacity clamping to 1
//  20.  reserve() extends capacity without breaking existing entries

use renderer_backend::gpu_driven::{
    BufferRegistry, MaterialTable, MaterialTableEntry,
    MaterialRemoveResult, SubmitResult,
    DEFAULT_MATERIAL_TABLE_CAPACITY, MATERIAL_FLAG_DIRTY, MATERIAL_FLAG_VISIBLE,
    MATERIAL_TABLE_ENTRY_SIZE,
};

// =============================================================================
// Test helpers
// =============================================================================

/// Creates a default PBR material entry with standard PBR properties.
fn default_pbr() -> MaterialTableEntry {
    MaterialTableEntry {
        base_color: [0.8, 0.8, 0.8, 1.0],
        emissive: [0.0, 0.0, 0.0, 0.0],
        metallic: 0.5,
        roughness: 0.3,
        occlusion: 1.0,
        normal_scale: 1.0,
        albedo_texture_id: u32::MAX,
        normal_texture_id: u32::MAX,
        metallic_roughness_texture_id: u32::MAX,
        emissive_texture_id: u32::MAX,
        flags: MATERIAL_FLAG_VISIBLE,
        alpha_cutoff: 0.5,
    }
}

/// Creates a red metallic material.
fn red_metal() -> MaterialTableEntry {
    MaterialTableEntry {
        base_color: [0.9, 0.1, 0.1, 1.0],
        metallic: 0.9,
        roughness: 0.1,
        occlusion: 1.0,
        normal_scale: 1.0,
        ..MaterialTableEntry::zeroed()
    }
}

/// Creates a blue rough material.
fn blue_rough() -> MaterialTableEntry {
    MaterialTableEntry {
        base_color: [0.1, 0.2, 0.9, 1.0],
        metallic: 0.0,
        roughness: 0.8,
        occlusion: 0.5,
        normal_scale: 0.5,
        ..MaterialTableEntry::zeroed()
    }
}

/// Returns a small `BufferRegistry` for staging tests.
fn small_registry() -> BufferRegistry {
    BufferRegistry::new(8192) // 8 KiB -- fits a few entries
}

// =============================================================================
// SECTION 1 -- Auto-append acceptance: the asset pipeline contract
// =============================================================================

/// THE ACCEPTANCE TEST: Materials loaded via the asset pipeline auto-append
/// to the MaterialTable and return sequential u32 indices.
///
/// This validates the core contract of T-GPU-1.4: when the asset pipeline
/// loads materials and calls add(), each material receives a unique, sequential
/// u32 index. Shaders reference materials by these indices via the bindless
/// `array<MaterialTableEntry>` storage buffer.
#[test]
fn materials_auto_append_with_sequential_indices() {
    let mut table = MaterialTable::new();

    // Simulate asset pipeline loading 5 materials.
    let idx_a = table.add(default_pbr());
    let idx_b = table.add(red_metal());
    let idx_c = table.add(blue_rough());
    let idx_d = table.add(default_pbr());
    let idx_e = table.add(red_metal());

    // Indices must be sequential starting from 0.
    assert_eq!(idx_a, 0, "First material gets index 0");
    assert_eq!(idx_b, 1, "Second material gets index 1");
    assert_eq!(idx_c, 2, "Third material gets index 2");
    assert_eq!(idx_d, 3, "Fourth material gets index 3");
    assert_eq!(idx_e, 4, "Fifth material gets index 4");

    // live_count must reflect all 5 materials.
    assert_eq!(table.live_count(), 5, "5 live materials");
    assert!(!table.is_empty());
}

/// Every material added via add() is automatically marked dirty.
/// The dirty flag is the mechanism by which the staging pipeline knows which
/// entries to upload. This is the core of "dirty flag management" in the
/// acceptance criterion.
#[test]
fn add_marks_entry_dirty() {
    let mut table = MaterialTable::new();
    table.add(default_pbr());
    assert!(table.any_dirty(), "Table must have dirty entries after add");
    assert_eq!(table.dirty_count(), 1, "Exactly 1 dirty entry");
}

/// After adding multiple materials, dirty_count reflects all of them.
#[test]
fn add_multiple_all_dirty() {
    let mut table = MaterialTable::with_capacity(16);
    table.add(default_pbr());
    table.add(red_metal());
    table.add(blue_rough());
    assert_eq!(table.dirty_count(), 3, "All 3 entries are dirty after add");
}

// =============================================================================
// SECTION 2 -- Material property preservation
// =============================================================================

/// Shaders reference materials by u32 index via the bindless material table.
/// get(index) must return the correct entry with all PBR fields intact.
#[test]
fn entry_retrievable_by_index_with_all_pbr_fields() {
    let mut table = MaterialTable::new();

    let entry = MaterialTableEntry {
        base_color: [0.2, 0.4, 0.6, 1.0],
        emissive: [0.1, 0.0, 0.0, 2.0],
        metallic: 0.8,
        roughness: 0.15,
        occlusion: 0.9,
        normal_scale: 1.5,
        albedo_texture_id: 3,
        normal_texture_id: 7,
        metallic_roughness_texture_id: 11,
        emissive_texture_id: 15,
        flags: MATERIAL_FLAG_VISIBLE | MATERIAL_FLAG_DIRTY,
        alpha_cutoff: 0.45,
    };

    let idx = table.add(entry);

    // Retrieve by index -- this is what the GPU does with the u32 handle.
    let retrieved = table.get(idx).expect("Entry must be retrievable by index");

    // PBR colour fields.
    assert_eq!(retrieved.base_color, [0.2, 0.4, 0.6, 1.0]);
    assert_eq!(retrieved.emissive, [0.1, 0.0, 0.0, 2.0]);

    // PBR scalar fields.
    assert!((retrieved.metallic - 0.8).abs() < f32::EPSILON);
    assert!((retrieved.roughness - 0.15).abs() < f32::EPSILON);
    assert!((retrieved.occlusion - 0.9).abs() < f32::EPSILON);
    assert!((retrieved.normal_scale - 1.5).abs() < f32::EPSILON);

    // Texture bindings.
    assert_eq!(retrieved.albedo_texture_id, 3);
    assert_eq!(retrieved.normal_texture_id, 7);
    assert_eq!(retrieved.metallic_roughness_texture_id, 11);
    assert_eq!(retrieved.emissive_texture_id, 15);

    // Flags and alpha.
    assert!(retrieved.flags & MATERIAL_FLAG_VISIBLE != 0);
    assert!(retrieved.flags & MATERIAL_FLAG_DIRTY != 0);
    assert!((retrieved.alpha_cutoff - 0.45).abs() < f32::EPSILON);
}

/// Multiple shader invocations (e.g., fragment shader for every pixel) can
/// reference the same material by index and get consistent data.
#[test]
fn multiple_references_to_same_material_via_index() {
    let mut table = MaterialTable::new();

    let shared_mat = table.add(default_pbr());

    // Simulate 10 shader invocations all reading the same material.
    for invocation in 0..10u32 {
        let mat = table
            .get(shared_mat)
            .unwrap_or_else(|| panic!("Invocation {}: material index must be valid", invocation));
        assert_eq!(mat.base_color, [0.8, 0.8, 0.8, 1.0]);
        assert!((mat.metallic - 0.5).abs() < f32::EPSILON);
    }

    assert_eq!(table.live_count(), 1, "Only one material in the table");
}

// =============================================================================
// SECTION 3 -- Texture ID sentinel
// =============================================================================

/// u32::MAX (0xFFFF_FFFF) is the sentinel for "no texture bound." This must
/// be the default for entries created via zeroed(). WGSL shader helpers check
/// this sentinel before dereferencing the bindless texture array.
#[test]
fn texture_id_sentinel_is_u32_max() {
    let entry = MaterialTableEntry::zeroed();
    assert_eq!(
        entry.albedo_texture_id, u32::MAX,
        "albedo sentinel must be u32::MAX"
    );
    assert_eq!(
        entry.normal_texture_id, u32::MAX,
        "normal sentinel must be u32::MAX"
    );
    assert_eq!(
        entry.metallic_roughness_texture_id, u32::MAX,
        "metallic-roughness sentinel must be u32::MAX"
    );
    assert_eq!(
        entry.emissive_texture_id, u32::MAX,
        "emissive sentinel must be u32::MAX"
    );
}

/// The is_zero() check must account for u32::MAX texture IDs (a zeroed entry
/// has all texture IDs set to the sentinel, not 0).
#[test]
fn zeroed_entry_has_texture_sentinels() {
    let entry = MaterialTableEntry::zeroed();
    assert!(entry.is_zero(), "zeroed entry must pass is_zero() check");
}

// =============================================================================
// SECTION 4 -- Full frame loop: add -> stage -> submit -> GPU read-back
// =============================================================================

/// Full frame cycle simulating the complete CPU-to-GPU pipeline for materials.
///
/// The asset pipeline loads materials, appends them to the MaterialTable, then
/// stages the table through BufferRegistry for GPU upload. The GPU reads the
/// data back and uses the u32 indices for bindless material lookups.
#[test]
fn full_frame_loop_add_stage_submit_readback() {
    let mut table = MaterialTable::with_capacity(4);
    let mut registry = small_registry();

    // Load materials (simulating asset pipeline).
    let idx0 = table.add(default_pbr());
    let idx1 = table.add(red_metal());

    // Stage and submit for GPU upload.
    let (slot_index, written) = table
        .stage(&mut registry)
        .expect("Must acquire staging slot");
    assert_eq!(
        written,
        MATERIAL_TABLE_ENTRY_SIZE * 4,
        "Whole table (4 entries) staged, not just live ones"
    );

    assert!(matches!(
        registry.submit_staging(slot_index, written),
        SubmitResult::Submitted
    ));

    // GPU side: acquire the ready slot and read the data.
    let read_idx = registry
        .acquire_reading()
        .expect("Must have a ready slot for GPU to read");
    let slot = registry.slot(read_idx).unwrap();

    // Verify total byte size.
    assert_eq!(slot.size(), MATERIAL_TABLE_ENTRY_SIZE * 4);
    let bytes = slot.as_slice();

    // Entry 0 at offset 0: base_color = [0.8, 0.8, 0.8, 1.0]
    let e0_base_color: &[f32; 4] =
        unsafe { &*(bytes.as_ptr() as *const [f32; 4]) };
    assert!((e0_base_color[0] - 0.8).abs() < f32::EPSILON, "entry[0].base_color.r");
    assert!((e0_base_color[3] - 1.0).abs() < f32::EPSILON, "entry[0].base_color.a");

    // Entry 0: metallic at offset 32 = 0.5
    let e0_metallic: f32 =
        unsafe { bytes.as_ptr().add(32).cast::<f32>().read() };
    assert!((e0_metallic - 0.5).abs() < f32::EPSILON, "entry[0].metallic");

    // Entry 1 at offset 80: base_color = [0.9, 0.1, 0.1, 1.0]
    let e1_base_color: &[f32; 4] =
        unsafe { &*(bytes.as_ptr().add(MATERIAL_TABLE_ENTRY_SIZE) as *const [f32; 4]) };
    assert!((e1_base_color[0] - 0.9).abs() < f32::EPSILON, "entry[1].base_color.r");

    // Entry 1: metallic at offset 32+80 = 112 = 0.9
    let e1_metallic: f32 =
        unsafe { bytes.as_ptr().add(MATERIAL_TABLE_ENTRY_SIZE + 32).cast::<f32>().read() };
    assert!((e1_metallic - 0.9).abs() < f32::EPSILON, "entry[1].metallic");

    // GPU would use indices 0 and 1 to reference these materials.
    assert_eq!(idx0, 0, "GPU references material A by index 0");
    assert_eq!(idx1, 1, "GPU references material B by index 1");
}

// =============================================================================
// SECTION 5 -- Incremental loading across multiple frames
// =============================================================================

/// Incremental loading: materials are added across multiple frames, each frame
/// stages the updated table for GPU upload. All indices remain stable and
/// correct across frames.
///
/// This simulates a real-world scenario where the asset pipeline streams
/// materials in over time while the engine is running.
#[test]
fn incremental_material_loading_across_frames() {
    let mut table = MaterialTable::with_capacity(16);
    let mut registry = BufferRegistry::new(16384);

    // --- Frame 1: load 3 materials, stage, submit --------------------------
    let idx_a = table.add(default_pbr());
    let idx_b = table.add(red_metal());
    let idx_c = table.add(blue_rough());

    // Stage frame 1 (live count = 3, capacity = 16).
    let (slot_1, written_1) = table.stage(&mut registry)
        .expect("Frame 1: must acquire slot");
    assert_eq!(
        written_1,
        MATERIAL_TABLE_ENTRY_SIZE * 16,
        "Frame 1: whole table (16 entries) staged"
    );
    assert!(matches!(
        registry.submit_staging(slot_1, written_1),
        SubmitResult::Submitted
    ));

    // GPU reads frame 1.
    let read_1 = registry.acquire_reading()
        .expect("Frame 1: GPU must acquire");
    let data_1 = registry.slot(read_1).unwrap();
    assert_eq!(data_1.size(), MATERIAL_TABLE_ENTRY_SIZE * 16);

    // --- Frame 2: load 2 more materials, stage again -----------------------
    let idx_d = table.add(default_pbr());
    let idx_e = table.add(red_metal());

    // Old indices must still be valid.
    assert_eq!(idx_a, 0, "Index 0 stable across frames");
    assert_eq!(idx_b, 1, "Index 1 stable across frames");
    assert_eq!(idx_c, 2, "Index 2 stable across frames");
    // New indices continue sequentially.
    assert_eq!(idx_d, 3, "New material gets next sequential index");
    assert_eq!(idx_e, 4, "New material gets next sequential index");

    assert_eq!(table.live_count(), 5, "5 live materials across two frames");

    // Stage frame 2 (now 5 live entries, 16 capacity).
    let (slot_2, written_2) = table.stage(&mut registry)
        .expect("Frame 2: must acquire slot");
    assert_eq!(written_2, MATERIAL_TABLE_ENTRY_SIZE * 16);
    assert!(matches!(
        registry.submit_staging(slot_2, written_2),
        SubmitResult::Submitted
    ));

    // GPU reads frame 2 -- all 5 materials available.
    let read_2 = registry.acquire_reading()
        .expect("Frame 2: GPU must acquire");
    let data_2 = registry.slot(read_2).unwrap();
    assert_eq!(data_2.size(), MATERIAL_TABLE_ENTRY_SIZE * 16);

    // Verify each entry via the CPU-side table.
    let mat_a = table.get(0).expect("idx_a must exist");
    assert_eq!(mat_a.base_color, [0.8, 0.8, 0.8, 1.0]);

    let mat_b = table.get(1).expect("idx_b must exist");
    assert_eq!(mat_b.base_color, [0.9, 0.1, 0.1, 1.0]);

    let mat_c = table.get(2).expect("idx_c must exist");
    assert_eq!(mat_c.base_color, [0.1, 0.2, 0.9, 1.0]);
    assert!((mat_c.metallic - 0.0).abs() < f32::EPSILON);
}

// =============================================================================
// SECTION 6 -- Dirty flag lifecycle
// =============================================================================

/// The dirty flag lifecycle: add() marks dirty, stage_and_submit() clears
/// dirty only after successful submit, update() marks dirty again.
/// stage() preserves dirty flags for retry safety (T-GPU-1.4 fix).
#[test]
fn dirty_flag_lifecycle_add_stage_submit_update() {
    let mut table = MaterialTable::with_capacity(4);
    let mut registry = small_registry();

    // Phase 1: add marks dirty.
    let idx = table.add(default_pbr());
    assert!(table.any_dirty(), "After add: table must be dirty");
    assert_eq!(table.dirty_count(), 1);

    // Phase 2: stage_and_submit clears dirty after successful submit.
    assert!(table.stage_and_submit(&mut registry), "stage_and_submit must succeed");
    assert!(!table.any_dirty(), "After submit: table must be clean");
    assert_eq!(table.dirty_count(), 0);

    // Phase 3: update marks dirty again.
    table.update(idx, red_metal()).expect("Update must succeed");
    assert!(table.any_dirty(), "After update: table must be dirty");
    assert_eq!(table.dirty_count(), 1);
}

/// mark_clean() explicitly clears all dirty flags.
#[test]
fn mark_clean_clears_all_dirty_flags() {
    let mut table = MaterialTable::with_capacity(8);
    table.add(default_pbr());
    table.add(red_metal());
    table.add(blue_rough());
    assert_eq!(table.dirty_count(), 3);

    table.mark_clean();
    assert!(!table.any_dirty());
    assert_eq!(table.dirty_count(), 0);
}

/// mark_dirty() can re-dirty a specific entry after it was cleaned.
#[test]
fn mark_dirty_on_specific_entry() {
    let mut table = MaterialTable::with_capacity(4);
    let idx = table.add(default_pbr());
    table.mark_clean();
    assert!(!table.any_dirty());

    assert!(table.mark_dirty(idx));
    assert!(table.any_dirty());
    assert_eq!(table.dirty_count(), 1);
}

/// mark_dirty() on a hole or out-of-bounds index returns false.
#[test]
fn mark_dirty_nonexistent_returns_false() {
    let mut table = MaterialTable::with_capacity(4);
    // index 0 is a hole (no material added yet).
    assert!(!table.mark_dirty(0), "Hole must return false");
    assert!(!table.mark_dirty(999), "Out of bounds must return false");
}

// =============================================================================
// SECTION 7 -- get_mut marks dirty
// =============================================================================

/// get_mut returns a mutable reference and automatically marks the entry dirty.
/// The caller can then mutate the entry in-place without calling separate
/// update().
#[test]
fn get_mut_marks_dirty_and_allows_in_place_mutation() {
    let mut table = MaterialTable::with_capacity(8);
    let idx = table.add(default_pbr());
    table.mark_clean();

    // Mutate visibility via mutable reference.
    {
        let entry = table.get_mut(idx).expect("get_mut must succeed");
        entry.flags &= !MATERIAL_FLAG_VISIBLE; // make invisible
    }

    // The entry must now be dirty.
    assert!(table.any_dirty(), "get_mut must mark entry dirty");
    let entry = table.get(idx).unwrap();
    assert!(entry.flags & MATERIAL_FLAG_VISIBLE == 0, "Visibility toggled off");

    // Index unchanged.
    assert_eq!(idx, 0);
    assert_eq!(table.live_count(), 1);
}

/// get_mut on a hole or out-of-bounds returns None.
#[test]
fn get_mut_nonexistent_returns_none() {
    let mut table = MaterialTable::with_capacity(4);
    assert!(table.get_mut(0).is_none(), "Hole must return None");
    assert!(table.get_mut(999).is_none(), "Out of bounds must return None");
}

// =============================================================================
// SECTION 8 -- Hole creation and index stability after removal
// =============================================================================

/// Removing a material zeroes the slot (hole) but does NOT change the indices
/// of other materials. This is critical for GPU-driven rendering where in-flight
/// GPU commands reference materials by index.
#[test]
fn remove_creates_hole_preserves_other_indices() {
    let mut table = MaterialTable::with_capacity(8);

    let idx_a = table.add(default_pbr());
    let idx_b = table.add(red_metal());
    let idx_c = table.add(blue_rough());

    // Remove the middle material.
    assert_eq!(table.remove(idx_b), MaterialRemoveResult::Removed);

    // live_count decreased.
    assert_eq!(table.live_count(), 2, "Two live materials remain");

    // All indices unchanged (no shift).
    assert_eq!(table.get(idx_a).unwrap().base_color, [0.8, 0.8, 0.8, 1.0], "idx_a stable");
    assert_eq!(table.get(idx_c).unwrap().base_color, [0.1, 0.2, 0.9, 1.0], "idx_c stable");

    // The removed slot must be a hole (get returns None).
    assert!(table.get(idx_b).is_none(), "Removed slot must be a hole");
}

/// Removing a nonexistent index or a hole returns NotFound.
#[test]
fn remove_nonexistent_returns_not_found() {
    let mut table = MaterialTable::with_capacity(4);
    assert_eq!(
        table.remove(0),
        MaterialRemoveResult::NotFound,
        "Removing index 0 from empty table must be NotFound"
    );
    assert_eq!(
        table.remove(999),
        MaterialRemoveResult::NotFound,
        "Removing out of bounds must be NotFound"
    );
}

/// Removing an entry and then adding a new entry reuses the hole (the new
/// entry goes into the first available zeroed slot).
#[test]
fn remove_then_add_reuses_hole() {
    let mut table = MaterialTable::with_capacity(8);

    let idx0 = table.add(default_pbr());
    let idx1 = table.add(red_metal());
    assert_eq!(idx0, 0);
    assert_eq!(idx1, 1);

    // Remove material at index 0.
    assert_eq!(table.remove(idx0), MaterialRemoveResult::Removed);
    assert_eq!(table.live_count(), 1);

    // Adding a new material should reuse slot 0 (first hole).
    let idx_new = table.add(blue_rough());
    assert_eq!(idx_new, 0, "New material reuses freed slot 0");
    assert_eq!(table.live_count(), 2);

    // Slot 0 now has the new material.
    assert_eq!(table.get(0).unwrap().base_color, [0.1, 0.2, 0.9, 1.0]);
}

// =============================================================================
// SECTION 9 -- insert_at for specific index assignment
// =============================================================================

/// insert_at allows the asset pipeline to place a material at a specific index,
/// for example when slot indices are pre-determined by a data-driven config.
#[test]
fn insert_at_specific_index() {
    let mut table = MaterialTable::with_capacity(16);

    // Insert at index 5.
    table.insert_at(5, red_metal());

    // Table capacity is 16, index 5 must have the material.
    let entry = table.get(5).expect("Index 5 must have the material");
    assert_eq!(entry.base_color, [0.9, 0.1, 0.1, 1.0]);
    assert_eq!(table.live_count(), 1);

    // Indices 0..4 are zeroed holes.
    for i in 0..5u32 {
        assert!(table.get(i).is_none(), "Index {} must be a hole", i);
    }
}

/// insert_at overwriting a live slot does not increase live_count.
#[test]
fn insert_at_overwrites_live_slot() {
    let mut table = MaterialTable::with_capacity(8);
    table.add(default_pbr()); // index 0
    table.add(red_metal());   // index 1
    assert_eq!(table.live_count(), 2);

    // Overwrite index 1.
    table.insert_at(1, blue_rough());
    assert_eq!(table.live_count(), 2, "Live count unchanged after overwrite");
    assert_eq!(table.get(1).unwrap().base_color, [0.1, 0.2, 0.9, 1.0]);
}

/// insert_at fills a hole and increments live_count.
#[test]
fn insert_at_fills_hole() {
    let mut table = MaterialTable::with_capacity(8);
    let idx = table.add(default_pbr());
    assert_eq!(idx, 0);
    table.remove(idx);
    assert_eq!(table.live_count(), 0);

    // insert_at the hole.
    table.insert_at(0, red_metal());
    assert_eq!(table.live_count(), 1);
    assert_eq!(table.get(0).unwrap().base_color, [0.9, 0.1, 0.1, 1.0]);
}

// =============================================================================
// SECTION 10 -- Clear and re-add
// =============================================================================

/// After clear(), the table is fully reset. New materials get indices starting
/// from 0 again. This simulates unloading a scene and loading a new one.
#[test]
fn clear_and_reload_indices_restart() {
    let mut table = MaterialTable::with_capacity(16);

    // Load scene A (3 materials).
    let scene_a: Vec<u32> = (0..3).map(|_| table.add(default_pbr())).collect();
    assert_eq!(scene_a, [0, 1, 2], "Scene A indices 0..2");

    // Clear (simulating scene unload).
    table.clear();
    assert!(table.is_empty());
    assert_eq!(table.live_count(), 0);
    // Capacity preserved.
    assert_eq!(table.len(), 16);

    // Load scene B -- indices restart from 0.
    let scene_b: Vec<u32> = (0..2).map(|_| table.add(red_metal())).collect();
    assert_eq!(scene_b, [0, 1], "Scene B indices restart from 0");

    // Scene B data is correct at indices 0 and 1.
    assert_eq!(table.get(0).unwrap().base_color, [0.9, 0.1, 0.1, 1.0]);
    assert_eq!(table.get(1).unwrap().base_color, [0.9, 0.1, 0.1, 1.0]);
}

/// Staging after clear: the table transitions from populated -> empty ->
/// repopulated without errors.
#[test]
fn clear_then_stage_works_correctly() {
    let mut table = MaterialTable::with_capacity(4);
    let mut registry = small_registry();

    // Load initial materials and stage.
    table.add(default_pbr());
    let (s0, w0) = table.stage(&mut registry).expect("Initial stage");
    assert!(matches!(registry.submit_staging(s0, w0), SubmitResult::Submitted));

    // Clear.
    table.clear();

    // Empty table cannot be staged (no dirty entries).
    assert!(
        table.stage(&mut registry).is_none(),
        "Empty table after clear must return None"
    );

    // Reload with different data.
    table.add(red_metal());

    // Stage the new data.
    let (s1, w1) = table.stage(&mut registry).expect("Repopulated stage");
    assert_eq!(w1, MATERIAL_TABLE_ENTRY_SIZE * 4, "Whole table (4 entries) staged");
    assert!(matches!(registry.submit_staging(s1, w1), SubmitResult::Submitted));

    // GPU reads back -- must be the new data.
    let read_idx = registry.acquire_reading().expect("Must acquire read slot");
    let bytes = registry.slot(read_idx).unwrap().as_slice();

    // Entry 0: base_color = [0.9, 0.1, 0.1, 1.0]
    let e0_bc: &[f32; 4] = unsafe { &*(bytes.as_ptr() as *const [f32; 4]) };
    assert!((e0_bc[0] - 0.9).abs() < f32::EPSILON, "After reload: base_color.r=0.9");
    assert!((e0_bc[2] - 0.1).abs() < f32::EPSILON, "After reload: base_color.b=0.1");
}

// =============================================================================
// SECTION 11 -- Bulk staging integrity through BufferRegistry
// =============================================================================

/// Stage and submit a table with diverse materials through the full triple-buffer
/// pipeline and verify that every byte arrives intact at the GPU side.
/// This is the end-to-end data-path validation for the material table.
#[test]
fn bulk_staging_integrity_full_pipeline() {
    let mut table = MaterialTable::with_capacity(64);
    let mut registry = BufferRegistry::new(65536); // 64 KiB

    // Load 50 materials with deterministic patterns.
    for i in 0..50u32 {
        let mat = MaterialTableEntry {
            base_color: [
                i as f32 / 50.0,
                (i + 1) as f32 / 50.0,
                (i + 2) as f32 / 50.0,
                1.0,
            ],
            metallic: (i % 10) as f32 / 10.0,
            roughness: ((i + 1) % 10) as f32 / 10.0,
            occlusion: 1.0,
            normal_scale: 1.0,
            albedo_texture_id: i,
            normal_texture_id: i + 100,
            metallic_roughness_texture_id: i + 200,
            emissive_texture_id: i + 300,
            flags: MATERIAL_FLAG_VISIBLE,
            alpha_cutoff: 0.5,
            ..MaterialTableEntry::zeroed()
        };
        table.add(mat);
    }

    // Stage the full table.
    let (slot_index, written) = table.stage(&mut registry).expect("Must acquire slot");
    assert_eq!(
        written,
        MATERIAL_TABLE_ENTRY_SIZE * 64,
        "Whole table (64 entries) staged"
    );
    assert!(matches!(
        registry.submit_staging(slot_index, written),
        SubmitResult::Submitted
    ));

    // Read back and verify entries.
    let read_idx = registry.acquire_reading().expect("Must acquire read slot");
    let bytes = registry.slot(read_idx).unwrap().as_slice();
    assert_eq!(bytes.len(), MATERIAL_TABLE_ENTRY_SIZE * 64);

    for i in 0..50u32 {
        let offset = (i as usize) * MATERIAL_TABLE_ENTRY_SIZE;

        // base_color at offset 0 (4 x f32).
        let bc_r: f32 = unsafe { bytes.as_ptr().add(offset).cast::<f32>().read() };
        let bc_g: f32 = unsafe { bytes.as_ptr().add(offset + 4).cast::<f32>().read() };
        assert!(
            (bc_r - i as f32 / 50.0).abs() < 0.001,
            "entry[{}].base_color.r mismatch",
            i
        );
        assert!(
            (bc_g - (i + 1) as f32 / 50.0).abs() < 0.001,
            "entry[{}].base_color.g mismatch",
            i
        );

        // metallic at offset 32.
        let metallic: f32 = unsafe { bytes.as_ptr().add(offset + 32).cast::<f32>().read() };
        let expected_metal = (i % 10) as f32 / 10.0;
        assert!(
            (metallic - expected_metal).abs() < f32::EPSILON,
            "entry[{}].metallic mismatch: expected {}, got {}",
            i, expected_metal, metallic
        );

        // roughness at offset 36.
        let roughness: f32 = unsafe { bytes.as_ptr().add(offset + 36).cast::<f32>().read() };
        let expected_rough = ((i + 1) % 10) as f32 / 10.0;
        assert!(
            (roughness - expected_rough).abs() < f32::EPSILON,
            "entry[{}].roughness mismatch",
            i
        );

        // albedo_texture_id at offset 48.
        let albedo: u32 = unsafe { bytes.as_ptr().add(offset + 48).cast::<u32>().read() };
        assert_eq!(albedo, i, "entry[{}].albedo_texture_id mismatch", i);

        // flags at offset 64.
        let flags: u32 = unsafe { bytes.as_ptr().add(offset + 64).cast::<u32>().read() };
        assert!(
            flags & MATERIAL_FLAG_VISIBLE != 0,
            "entry[{}] must be visible",
            i
        );
    }
}

// =============================================================================
// SECTION 12 -- Dense loading at scale (1000 entries)
// =============================================================================

/// Loading 1000 materials -- simulating a real scene load with many PBR
/// materials -- must produce sequential indices 0..999 and all entries must
/// be retrievable by index without error.
#[test]
fn dense_loading_at_scale_1000_materials() {
    let mut table = MaterialTable::with_capacity(1024);

    // Simulate bulk asset pipeline load.
    for i in 0..1000u32 {
        let mat = MaterialTableEntry {
            base_color: [
                (i % 256) as f32 / 255.0,
                ((i + 85) % 256) as f32 / 255.0,
                ((i + 170) % 256) as f32 / 255.0,
                1.0,
            ],
            metallic: (i % 10) as f32 / 10.0,
            roughness: ((i + 1) % 10) as f32 / 10.0,
            occlusion: 1.0,
            normal_scale: 1.0,
            albedo_texture_id: i % 512,
            normal_texture_id: (i + 100) % 512,
            metallic_roughness_texture_id: (i + 200) % 512,
            emissive_texture_id: (i + 300) % 512,
            flags: MATERIAL_FLAG_VISIBLE,
            alpha_cutoff: 0.5,
            ..MaterialTableEntry::zeroed()
        };
        let idx = table.add(mat);
        assert_eq!(
            idx, i,
            "Material {} must get sequential index {}, got {}",
            i, i, idx
        );
    }

    assert_eq!(table.live_count(), 1000);
    assert_eq!(table.len(), 1024); // capacity unchanged

    // All 1000 materials retrievable by index.
    for i in 0..1000u32 {
        let entry = table
            .get(i)
            .unwrap_or_else(|| panic!("Material at index {} must be retrievable", i));
        let expected_r = (i % 256) as f32 / 255.0;
        assert!(
            (entry.base_color[0] - expected_r).abs() < 0.001,
            "base_color.r for material {}",
            i
        );
        assert_eq!(
            entry.metallic,
            (i % 10) as f32 / 10.0,
            "metallic for material {}",
            i
        );
        assert_eq!(
            entry.albedo_texture_id,
            i % 512,
            "albedo_texture_id for material {}",
            i
        );
    }

    // as_bytes must produce the correct total size.
    assert_eq!(
        table.as_bytes().len(),
        1024 * MATERIAL_TABLE_ENTRY_SIZE,
        "Byte length must be 1024 * 80 = 81920"
    );
}

// =============================================================================
// SECTION 13 -- Empty table edge cases
// =============================================================================

/// An empty (all-zero) material table cannot be staged (no dirty entries).
/// Nothing to upload.
#[test]
fn empty_table_stage_returns_none() {
    let mut table = MaterialTable::with_capacity(4);
    let mut registry = small_registry();

    assert!(
        table.stage(&mut registry).is_none(),
        "Empty table (no dirty entries) must return None from stage()"
    );
}

/// An empty table that has never had entries added has dirty_count 0.
#[test]
fn empty_table_has_no_dirty_entries() {
    let table = MaterialTable::with_capacity(4);
    assert!(!table.any_dirty());
    assert_eq!(table.dirty_count(), 0);
}

/// Getting an entry from a table with only holes returns None.
#[test]
fn get_from_holed_table_returns_none() {
    let table = MaterialTable::with_capacity(4);
    assert!(table.get(0).is_none(), "Index 0 on all-hole table must be None");
    assert!(table.get(999).is_none(), "Out of bounds must be None");
}

/// stage_and_submit on an empty table returns false.
#[test]
fn stage_and_submit_clean_table_returns_false() {
    let mut table = MaterialTable::with_capacity(4);
    let mut registry = small_registry();
    assert!(
        !table.stage_and_submit(&mut registry),
        "Clean table must return false from stage_and_submit"
    );
}

// =============================================================================
// SECTION 14 -- Update preserves index
// =============================================================================

/// Updating a material entry in place preserves its index. This allows the
/// asset pipeline to stream updated material data (e.g., PBR parameter
/// tweaks at runtime) without invalidating GPU-side references.
#[test]
fn update_preserves_index() {
    let mut table = MaterialTable::with_capacity(8);

    let idx = table.add(default_pbr());
    assert_eq!(idx, 0, "Initial index is 0");

    // Update to a different material.
    let updated = red_metal();
    assert!(table.update(idx, updated).is_some(), "Update must succeed");

    // Index unchanged.
    assert_eq!(table.live_count(), 1, "Live count unchanged after update");
    assert!(table.get(idx).is_some(), "Entry still retrievable at same index");

    // Entry has new data.
    let entry = table.get(idx).unwrap();
    assert_eq!(entry.base_color, [0.9, 0.1, 0.1, 1.0]);
    assert!((entry.metallic - 0.9).abs() < f32::EPSILON);
}

/// Update on a hole or out-of-bounds index returns None.
#[test]
fn update_nonexistent_returns_none() {
    let mut table = MaterialTable::with_capacity(4);
    assert!(
        table.update(0, default_pbr()).is_none(),
        "Update on hole must return None"
    );
    assert!(
        table.update(999, default_pbr()).is_none(),
        "Update out of bounds must return None"
    );
}

// =============================================================================
// SECTION 15 -- Display and formatting
// =============================================================================

/// Display output for the table must contain capacity and live count.
#[test]
fn display_shows_table_state() {
    let mut table = MaterialTable::with_capacity(16);
    let empty_display = format!("{}", table);
    assert!(
        empty_display.contains("capacity: 16"),
        "Display must show capacity"
    );
    assert!(
        empty_display.contains("live: 0"),
        "Display must show live count"
    );

    table.add(default_pbr());
    table.add(red_metal());
    let populated_display = format!("{}", table);
    assert!(
        populated_display.contains("live: 2"),
        "Display must show live=2"
    );
}

/// Display output for MaterialTableEntry must show all PBR fields.
#[test]
fn entry_display_shows_all_fields() {
    let entry = default_pbr();
    let s = format!("{}", entry);
    assert!(s.contains("base_color"), "Display must contain base_color");
    assert!(s.contains("emissive"), "Display must contain emissive");
    assert!(s.contains("metallic"), "Display must contain metallic");
    assert!(s.contains("roughness"), "Display must contain roughness");
    assert!(s.contains("occlusion"), "Display must contain occlusion");
    assert!(s.contains("albedo_tex"), "Display must contain albedo_tex");
    assert!(s.contains("flags"), "Display must contain flags");
}

// =============================================================================
// SECTION 16 -- Constants and size layout
// =============================================================================

/// DEFAULT_MATERIAL_TABLE_CAPACITY constant is exported and positive.
#[test]
fn default_capacity_constant_is_positive() {
    assert!(
        DEFAULT_MATERIAL_TABLE_CAPACITY > 0,
        "DEFAULT_MATERIAL_TABLE_CAPACITY must be > 0"
    );
    // The default capacity of 1024 supports real-world scenes.
    assert!(
        DEFAULT_MATERIAL_TABLE_CAPACITY >= 1024,
        "DEFAULT_MATERIAL_TABLE_CAPACITY should be at least 1024 for real scenes"
    );
}

/// MATERIAL_TABLE_ENTRY_SIZE constant is exported and matches the WGSL
/// struct layout (80 bytes).
#[test]
fn entry_size_constant_matches_wgsl() {
    assert_eq!(
        MATERIAL_TABLE_ENTRY_SIZE, 80,
        "MATERIAL_TABLE_ENTRY_SIZE must be 80 bytes per the WGSL layout spec"
    );
}

/// MATERIAL_FLAG constants are exported.
#[test]
fn material_flag_constants_are_exported() {
    assert_eq!(
        MATERIAL_FLAG_DIRTY, 0x8000_0000,
        "MATERIAL_FLAG_DIRTY must be bit 31"
    );
    assert_eq!(
        MATERIAL_FLAG_VISIBLE, 0x0000_0001,
        "MATERIAL_FLAG_VISIBLE must be bit 0"
    );
}

// =============================================================================
// SECTION 17 -- Zero and default entries
// =============================================================================

/// Zeroed entry has all fields at default sentinel values and is_zero()
/// returns true.
#[test]
fn zeroed_entry_is_zero() {
    let z = MaterialTableEntry::zeroed();
    assert!(z.is_zero());
    assert_eq!(z.base_color, [0.0; 4]);
    assert_eq!(z.emissive, [0.0; 4]);
    assert_eq!(z.metallic, 0.0);
    assert_eq!(z.roughness, 0.0);
    assert_eq!(z.occlusion, 0.0);
    assert_eq!(z.normal_scale, 0.0);
    assert_eq!(z.albedo_texture_id, u32::MAX);
    assert_eq!(z.normal_texture_id, u32::MAX);
    assert_eq!(z.metallic_roughness_texture_id, u32::MAX);
    assert_eq!(z.emissive_texture_id, u32::MAX);
    assert_eq!(z.flags, 0);
    assert_eq!(z.alpha_cutoff, 0.0);
}

/// Default entry is a zeroed entry.
#[test]
fn default_entry_is_zeroed() {
    let d: MaterialTableEntry = Default::default();
    assert!(d.is_zero());
}

/// A zeroed entry with only the dirty flag set still reads as zero.
#[test]
fn zeroed_with_dirty_flag_still_zero() {
    let entry = MaterialTableEntry {
        flags: MATERIAL_FLAG_DIRTY,
        ..MaterialTableEntry::zeroed()
    };
    assert!(
        entry.is_zero(),
        "Entry with only dirty flag set must still be is_zero()"
    );
}

// =============================================================================
// SECTION 18 -- zero_capacity table clamps to 1
// =============================================================================

/// with_capacity(0) must clamp to capacity 1 (minimum viable table).
#[test]
fn zero_capacity_clamps_to_one() {
    let table = MaterialTable::with_capacity(0);
    assert_eq!(table.len(), 1, "Zero capacity must clamp to 1");
    assert!(table.is_empty());
}

// =============================================================================
// SECTION 19 -- Reserve extends capacity
// =============================================================================

/// reserve() extends the table capacity without breaking existing entries.
#[test]
fn reserve_extends_capacity() {
    let mut table = MaterialTable::with_capacity(4);
    assert_eq!(table.len(), 4);

    // Add an entry.
    let idx = table.add(default_pbr());
    assert_eq!(idx, 0);

    // Reserve space for 8 more.
    table.reserve(8);
    assert_eq!(table.len(), 12, "Capacity extended to 12");

    // Existing entry still valid.
    let entry = table.get(0).expect("Existing entry must still be valid");
    assert_eq!(entry.base_color, [0.8, 0.8, 0.8, 1.0]);
    assert_eq!(table.live_count(), 1);

    // New entries can be added into the reserved space.
    let idx2 = table.add(red_metal());
    assert_eq!(idx2, 1, "Second entry at index 1 (first hole after index 0)");
    assert_eq!(table.live_count(), 2);
}

// =============================================================================
// SECTION 20 -- Auto-resize when full
// =============================================================================

/// When the table fills up (all slots occupied), add() must still succeed
/// by finding the first available slot or returning an appropriate result.
/// The spec says: "If the table is full (no holes available), it doubles in
/// capacity." So we verify that adding beyond initial capacity works.
#[test]
fn table_auto_grows_when_full_with_no_holes() {
    // Small table with capacity 2.
    let mut table = MaterialTable::with_capacity(2);

    // Fill both slots.
    let idx0 = table.add(default_pbr());
    let idx1 = table.add(red_metal());
    assert_eq!(idx0, 0);
    assert_eq!(idx1, 1);
    assert_eq!(table.live_count(), 2);

    // Add a third entry -- must succeed (table grows).
    let idx2 = table.add(blue_rough());
    // The spec says auto-resize doubles capacity: capacity becomes 4.
    // The new entry goes into the first slot beyond the original capacity.
    // Since fill was at indices 0 and 1 with no holes, add_inner extends.
    assert_eq!(idx2, 2, "Third entry gets index 2 after auto-grow");
    assert_eq!(table.live_count(), 3);
    assert!(
        table.len() >= 3,
        "Table must grow to accommodate all entries"
    );

    // All entries retrievable.
    assert_eq!(table.get(0).unwrap().base_color, [0.8, 0.8, 0.8, 1.0]);
    assert_eq!(table.get(1).unwrap().base_color, [0.9, 0.1, 0.1, 1.0]);
    assert_eq!(table.get(2).unwrap().base_color, [0.1, 0.2, 0.9, 1.0]);
}

// =============================================================================
// SECTION 21 -- Stage with multiple submissions (frame pipeline)
// =============================================================================

/// Simulate a multi-frame submission pattern: the asset pipeline loads
/// materials incrementally and stages them each frame. Each frame's GPU data
/// must contain all materials loaded up to that point.
#[test]
fn multi_frame_incremental_staging() {
    let mut table = MaterialTable::with_capacity(16);
    let mut registry = BufferRegistry::new(16384);

    // Frame 1: 2 materials.
    table.add(default_pbr());
    table.add(red_metal());
    let (s1, w1) = table.stage(&mut registry).expect("Frame 1 stage");
    assert_eq!(w1, MATERIAL_TABLE_ENTRY_SIZE * 16);
    assert!(matches!(registry.submit_staging(s1, w1), SubmitResult::Submitted));

    // GPU reads frame 1.
    let r1 = registry.acquire_reading().expect("Frame 1 read");
    assert_eq!(
        registry.slot(r1).unwrap().size(),
        MATERIAL_TABLE_ENTRY_SIZE * 16
    );

    // Frame 2: add 1 more material (3 total).
    table.add(blue_rough());
    let (s2, w2) = table.stage(&mut registry).expect("Frame 2 stage");
    assert_eq!(w2, MATERIAL_TABLE_ENTRY_SIZE * 16);
    assert!(matches!(registry.submit_staging(s2, w2), SubmitResult::Submitted));

    // GPU reads frame 2 (3 materials).
    let r2 = registry.acquire_reading().expect("Frame 2 read");
    let bytes2 = registry.slot(r2).unwrap().as_slice();
    assert_eq!(bytes2.len(), MATERIAL_TABLE_ENTRY_SIZE * 16);

    // Verify entry 0 at offset 0: default_pbr has base_color [0.8, 0.8, 0.8, 1.0].
    let e0_bc: &[f32; 4] = unsafe { &*(bytes2.as_ptr() as *const [f32; 4]) };
    assert!((e0_bc[0] - 0.8).abs() < f32::EPSILON, "Frame 2 entry[0].r");

    // Entry 1 at offset 80: red_metal has base_color [0.9, 0.1, 0.1, 1.0].
    let e1_bc: &[f32; 4] =
        unsafe { &*(bytes2.as_ptr().add(MATERIAL_TABLE_ENTRY_SIZE) as *const [f32; 4]) };
    assert!((e1_bc[0] - 0.9).abs() < f32::EPSILON, "Frame 2 entry[1].r");

    // Entry 2 at offset 160: blue_rough has base_color [0.1, 0.2, 0.9, 1.0].
    let e2_bc: &[f32; 4] =
        unsafe { &*(bytes2.as_ptr().add(2 * MATERIAL_TABLE_ENTRY_SIZE) as *const [f32; 4]) };
    assert!((e2_bc[0] - 0.1).abs() < f32::EPSILON, "Frame 2 entry[2].r");

    // Verify all 3 entries via CPU side.
    for i in 0..3u32 {
        assert!(
            table.get(i).is_some(),
            "Entry {} must exist on CPU side",
            i
        );
    }

    // Frame count is 2.
    assert_eq!(registry.frame_count(), 2);
}

// =============================================================================
// SECTION 22 -- stage_and_submit convenience method
// =============================================================================

/// stage_and_submit combines stage + submit into one call.
#[test]
fn stage_and_submit_works() {
    let mut table = MaterialTable::with_capacity(4);
    let mut registry = small_registry();

    table.add(default_pbr());
    assert!(
        table.stage_and_submit(&mut registry),
        "stage_and_submit must succeed"
    );

    // GPU reads the submitted data.
    let read_idx = registry.acquire_reading().expect("Must have ready slot");
    let bytes = registry.slot(read_idx).unwrap().as_slice();
    assert_eq!(bytes.len(), MATERIAL_TABLE_ENTRY_SIZE * 4);

    // Verify entry 0.
    let e0_bc: &[f32; 4] = unsafe { &*(bytes.as_ptr() as *const [f32; 4]) };
    assert!((e0_bc[0] - 0.8).abs() < f32::EPSILON);
}

/// stage_and_submit with all slots occupied returns false.
#[test]
fn stage_and_submit_fails_when_no_slot() {
    let mut table = MaterialTable::with_capacity(4);
    let mut registry = BufferRegistry::new(4096);

    table.add(default_pbr());

    // Occupy all 3 staging slots.
    let _ = registry.acquire_staging();
    let _ = registry.acquire_staging();
    let _ = registry.acquire_staging();

    assert!(
        !table.stage_and_submit(&mut registry),
        "stage_and_submit must fail when all slots occupied"
    );
}

// =============================================================================
// SECTION 23 -- Staging after update shows new data
// =============================================================================

/// Staging after update: the GPU buffer must contain the updated material
/// data at the same index position.
#[test]
fn staging_after_update_shows_new_data() {
    let mut table = MaterialTable::with_capacity(4);
    let mut registry = small_registry();

    table.add(default_pbr());

    // Update index 0 with red_metal.
    table.update(0, red_metal()).expect("Update must succeed");

    // Stage through BufferRegistry.
    let (slot_index, written) = table.stage(&mut registry)
        .expect("Must acquire slot");
    assert!(matches!(
        registry.submit_staging(slot_index, written),
        SubmitResult::Submitted
    ));

    let read_idx = registry.acquire_reading().expect("Must acquire read slot");
    let bytes = registry.slot(read_idx).unwrap().as_slice();

    // Entry 0 must be red_metal: base_color = [0.9, 0.1, 0.1, 1.0].
    let e0_bc: &[f32; 4] = unsafe { &*(bytes.as_ptr() as *const [f32; 4]) };
    assert!((e0_bc[0] - 0.9).abs() < f32::EPSILON, "Updated base_color.r=0.9");
    assert!((e0_bc[1] - 0.1).abs() < f32::EPSILON, "Updated base_color.g=0.1");

    // metallic at offset 32 should be 0.9 (red_metal is metallic).
    let metallic: f32 = unsafe { bytes.as_ptr().add(32).cast::<f32>().read() };
    assert!((metallic - 0.9).abs() < f32::EPSILON, "Updated metallic=0.9");
}

/// Update then stage across multiple frames increments frame_count.
#[test]
fn update_then_stage_increments_frame_count() {
    let mut table = MaterialTable::with_capacity(4);
    let mut registry = small_registry();

    let idx = table.add(default_pbr());

    // Stage frame 1.
    assert!(table.stage_and_submit(&mut registry));
    assert_eq!(registry.frame_count(), 1);

    // Update and stage frame 2.
    table.update(idx, red_metal()).expect("Update must succeed");
    assert!(table.stage_and_submit(&mut registry));
    assert_eq!(registry.frame_count(), 2);

    // GPU reads frame 2.
    let read_idx = registry.acquire_reading().expect("Must have ready slot");
    let bytes = registry.slot(read_idx).unwrap().as_slice();
    let e0_bc: &[f32; 4] = unsafe { &*(bytes.as_ptr() as *const [f32; 4]) };
    assert!((e0_bc[0] - 0.9).abs() < f32::EPSILON, "Frame 2 has updated data");
}

// =============================================================================
// SECTION 24 -- Dirty and hole interaction
// =============================================================================

/// Removing a cleaned entry produces a hole; that hole is not dirty.
#[test]
fn remove_cleaned_entry_creates_clean_hole() {
    let mut table = MaterialTable::with_capacity(4);
    let idx = table.add(default_pbr());
    table.mark_clean();

    table.remove(idx);
    // The hole must not be dirty.
    assert!(!table.any_dirty(), "Remove must not produce dirty entries");
}

/// Double stage-and-submit cycle: add, submit (clears dirty), add again,
/// submit again (clears dirty). stage() alone preserves dirty flags.
#[test]
fn double_stage_and_submit_cycle() {
    let mut table = MaterialTable::with_capacity(4);
    let mut registry = small_registry();

    // First cycle.
    table.add(default_pbr());
    assert!(table.stage_and_submit(&mut registry));
    assert!(!table.any_dirty());

    // Second cycle.
    table.add(red_metal());
    assert!(table.stage_and_submit(&mut registry));
    assert!(!table.any_dirty());

    assert_eq!(table.live_count(), 2);
}

// =============================================================================
// SECTION 25 -- as_bytes and as_slice consistency
// =============================================================================

/// The byte representation and the slice representation must agree on the
/// number of entries.
#[test]
fn as_bytes_and_as_slice_agree_on_count() {
    let mut table = MaterialTable::with_capacity(16);
    for _ in 0..5u32 {
        table.add(default_pbr());
    }

    let bytes = table.as_bytes();
    let slice = table.as_slice();

    assert_eq!(slice.len() * MATERIAL_TABLE_ENTRY_SIZE, bytes.len());
    assert_eq!(slice.len(), 16); // capacity

    // Spot-check entries via bytes match the slice.
    for (i, entry) in slice.iter().enumerate().take(5) {
        let bc_addr = i * MATERIAL_TABLE_ENTRY_SIZE;
        let bc_r: f32 = unsafe { bytes.as_ptr().add(bc_addr).cast::<f32>().read() };
        assert!(
            (bc_r - entry.base_color[0]).abs() < f32::EPSILON,
            "Byte/slice agreement for entry[{}].base_color.r",
            i
        );
    }
}
