// Blackbox contract tests for TextureTable (T-GPU-1.5).
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::gpu_driven::*` -- no internal fields,
// no private methods, no implementation details.
//
// Acceptance criterion (T-GPU-1.5):
//   Bindless Texture Table with free-list management.
//   MAX_BINDLESS_TEXTURES = 4096.
//
// The TextureTable guarantees:
//   - add() allocates from the free-list first (LIFO), or extends the entries vec
//   - add() returns Some(u32) on success, None when the table is full and the
//     free-list is empty (hard cap at MAX_BINDLESS_TEXTURES)
//   - remove() zeroes the entry and pushes its index onto the free-list
//   - Entries are retrievable by index via get() / as_slice()
//   - get() returns None for out-of-range indices
//   - The table can be staged through BufferRegistry for GPU upload
//   - Updating an entry in place keeps its index stable
//   - insert_at places an entry at a specific index, extending if needed
//   - clear() resets the table entirely
//
// Coverage:
//   1.  Sequential indices on fresh table
//   2.  Free-list reuse: remove then add reuses freed slot
//   3.  Free-list LIFO ordering (last removed pops first)
//   4.  Field preservation: all six TextureTableEntry fields retrievable
//   5.  Multiple shader references to the same texture via index
//   6.  Full frame loop: add -> stage -> submit -> GPU read-back
//   7.  Incremental texture loading across multiple frames
//   8.  Hole preserves other indices (index stability after remove)
//   9.  is_full() at MAX_BINDLESS_TEXTURES
//  10.  add returns None when table is full
//  11.  Full table remove then re-add
//  12.  insert_at specific index assignment
//  13.  insert_at at or beyond MAX_BINDLESS_TEXTURES returns false
//  14.  Clear and re-add: indices restart from 0
//  15.  update preserves index
//  16.  update live_count tracking (zero / non-zero transitions)
//  17.  Empty table properties (is_empty, len, live_count)
//  18.  Stage empty table returns None
//  19.  Display/formatting for table and entry
//  20.  Constants: MAX_BINDLESS_TEXTURES=4096, TEXTURE_TABLE_ENTRY_SIZE=24
//  21.  Dense loading at scale (1000+ entries)
//  22.  Bulk staging integrity through BufferRegistry
//  23.  Free-list no-duplicates invariant
//  24.  reserve() extends capacity
//  25.  as_bytes / as_slice consistency
//  26.  Zero / default entries and is_zero()
//  27.  staging after remove and update mutations

use renderer_backend::gpu_driven::{
    BufferRegistry, TextureTable, TextureTableEntry, TextureRemoveResult,
    SubmitResult,
    DEFAULT_TEXTURE_TABLE_CAPACITY, MAX_BINDLESS_TEXTURES, TEXTURE_TABLE_ENTRY_SIZE,
};

// =============================================================================
// Helpers
// =============================================================================

/// Creates a basic texture entry with given dimensions and a valid flag.
fn tex(width: u32, height: u32, mip_levels: u32) -> TextureTableEntry {
    TextureTableEntry::new(width, height, mip_levels, 0, 1, 1)
}

/// Creates a texture entry with full custom fields.
fn tex_full(
    width: u32,
    height: u32,
    mip_levels: u32,
    format: u32,
    layer_count: u32,
    flags: u32,
) -> TextureTableEntry {
    TextureTableEntry::new(width, height, mip_levels, format, layer_count, flags)
}

/// Returns a small BufferRegistry for staging tests.
fn small_registry() -> BufferRegistry {
    BufferRegistry::new(8192)
}

// =============================================================================
// SECTION 1 -- Sequential indices on a fresh table
// =============================================================================

/// THE ACCEPTANCE TEST: Textures added to a fresh table receive sequential
/// u32 indices starting from 0. This is the core contract for bindless access:
/// shaders reference textures by these indices into the bindless texture array.
#[test]
fn textures_get_sequential_indices_on_fresh_table() {
    let mut table = TextureTable::new();

    let idx_a = table.add(tex(1024, 768, 10)).unwrap();
    let idx_b = table.add(tex(512, 512, 8)).unwrap();
    let idx_c = table.add(tex(2048, 2048, 12)).unwrap();
    let idx_d = table.add(tex(128, 128, 1)).unwrap();
    let idx_e = table.add(tex(4096, 4096, 14)).unwrap();

    assert_eq!(idx_a, 0, "First texture gets index 0");
    assert_eq!(idx_b, 1, "Second texture gets index 1");
    assert_eq!(idx_c, 2, "Third texture gets index 2");
    assert_eq!(idx_d, 3, "Fourth texture gets index 3");
    assert_eq!(idx_e, 4, "Fifth texture gets index 4");

    assert_eq!(table.live_count(), 5, "5 live textures");
    assert_eq!(table.len(), 5);
    assert!(!table.is_empty());
}

// =============================================================================
// SECTION 2 -- Free-list reuse: remove then add reuses freed slot
// =============================================================================

/// Free-list management is the defining feature of TextureTable (vs MeshTable
/// which only creates holes). When a texture is removed, its index goes onto
/// the free-list. The next add() pops from the free-list first, reusing the
/// freed slot in O(1).
#[test]
fn remove_then_add_reuses_freed_slot() {
    let mut table = TextureTable::new();

    let idx_a = table.add(tex(1024, 768, 10)).unwrap();
    let idx_b = table.add(tex(512, 512, 8)).unwrap();
    let idx_c = table.add(tex(2048, 2048, 12)).unwrap();
    assert_eq!(table.live_count(), 3);

    // Remove the first entry (index 0).
    assert_eq!(table.remove(idx_a), TextureRemoveResult::Removed);
    assert_eq!(table.live_count(), 2);
    assert_eq!(table.free_count(), 1);

    // Add should reuse the freed slot at index 0.
    let recycled = table.add(tex(4096, 4096, 14)).unwrap();
    assert_eq!(recycled, 0, "free-list must reuse index 0");
    assert_eq!(table.live_count(), 3);
    assert_eq!(table.free_count(), 0);

    // Verify the recycled slot has the new data.
    let entry = table.get(recycled).unwrap();
    assert_eq!(entry.width, 4096);
    assert_eq!(entry.height, 4096);
    assert_eq!(entry.mip_levels, 14);
}

/// Removing the last entry pushes its index onto the free-list; the next
/// add reuses that index (LIFO).
#[test]
fn remove_last_then_add_reuses_last_index() {
    let mut table = TextureTable::new();

    table.add(tex(100, 100, 1)).unwrap();
    let idx = table.add(tex(200, 200, 2)).unwrap();
    assert_eq!(idx, 1);

    // Remove the last entry (index 1).
    assert_eq!(table.remove(idx), TextureRemoveResult::Removed);

    // Add reuses index 1 (LIFO free-list).
    let recycled = table.add(tex(300, 300, 3)).unwrap();
    assert_eq!(recycled, 1, "free-list LIFO must reuse last removed index");
    assert_eq!(table.len(), 2);
    assert_eq!(table.live_count(), 2);
}

// =============================================================================
// SECTION 3 -- Free-list LIFO ordering
// =============================================================================

/// The free-list is a LIFO stack. The last index pushed (most recently removed)
/// is the first one popped (reused). This test validates the ordering contract.
#[test]
fn free_list_lifo_ordering() {
    let mut table = TextureTable::new();

    // Add 10 textures.
    let mut indices = Vec::new();
    for i in 0..10u32 {
        indices.push(table.add(tex(i * 100, 100, 1)).unwrap());
    }
    assert_eq!(table.live_count(), 10);

    // Remove the even indices in reverse order: 8, 6, 4, 2, 0.
    // Free-list stack (after removes): [8, 6, 4, 2, 0] (8 on top).
    let mut freed = Vec::new();
    for i in (0..10).step_by(2).rev() {
        let idx = indices[i];
        table.remove(idx);
        freed.push(idx);
    }
    assert_eq!(table.free_count(), 5);

    // LIFO: first pop returns 8 (last removed even index), then 6, 4, 2, 0.
    // freed was pushed in reverse order: [8, 6, 4, 2, 0]
    for &expected_idx in &freed {
        let new_idx = table.add(tex(1, 1, 1)).unwrap();
        assert_eq!(
            new_idx, expected_idx,
            "free-list LIFO: expected index {}, got {}",
            expected_idx, new_idx
        );
    }
    assert_eq!(table.free_count(), 0);
    assert_eq!(table.live_count(), 10);
}

// =============================================================================
// SECTION 4 -- Field preservation
// =============================================================================

/// Shaders reference textures by u32 index into the bindless
/// `texture_2d_array<f32>`. get(index) must return the correct entry with all
/// six fields intact.
#[test]
fn entry_retrievable_by_index_with_all_fields() {
    let mut table = TextureTable::new();

    let idx = table.add(tex_full(
        1920,   // width
        1080,   // height
        11,     // mip_levels
        0x0A,   // format
        4,      // layer_count (array texture with 4 layers)
        0x01,   // flags (bit 0 = valid)
    )).unwrap();

    let entry = table.get(idx).expect("Entry must be retrievable by index");
    assert_eq!(entry.width, 1920, "width preserved");
    assert_eq!(entry.height, 1080, "height preserved");
    assert_eq!(entry.mip_levels, 11, "mip_levels preserved");
    assert_eq!(entry.format, 0x0A, "format preserved");
    assert_eq!(entry.layer_count, 4, "layer_count preserved");
    assert_eq!(entry.flags, 0x01, "flags preserved");

    // The entry must also be findable via as_slice() at the correct position.
    let slice = table.as_slice();
    assert_eq!(slice.len(), 1, "One entry in the slice");
    assert_eq!(slice[idx as usize].width, 1920);
}

/// get() returns None for out-of-range indices.
#[test]
fn get_out_of_range_returns_none() {
    let table = TextureTable::new();
    assert!(table.get(0).is_none(), "Empty table: index 0 must be None");
    assert!(table.get(u32::MAX).is_none(), "u32::MAX must be None");
}

// =============================================================================
// SECTION 5 -- Multiple shader references to same texture by index
// =============================================================================

/// Many shader invocations (fragment shaders across many pixels) can reference
/// the same texture by its u32 index and get consistent metadata.
#[test]
fn multiple_references_to_same_texture_via_index() {
    let mut table = TextureTable::new();

    let shared_tex = table.add(tex(1024, 768, 10)).unwrap();

    // Simulate 10 shader invocations all reading the same texture metadata.
    for invocation in 0..10u32 {
        let entry = table.get(shared_tex)
            .unwrap_or_else(|| panic!("Invocation {}: texture index must be valid", invocation));
        assert_eq!(entry.width, 1024, "Invocation {} sees correct width", invocation);
        assert_eq!(entry.height, 768, "Invocation {} sees correct height", invocation);
        assert_eq!(entry.mip_levels, 10, "Invocation {} sees correct mip_levels", invocation);
    }

    assert_eq!(table.live_count(), 1, "Only one texture in the table");
    assert_eq!(table.len(), 1);
}

// =============================================================================
// SECTION 6 -- Full frame loop: add -> stage -> submit -> read-back
// =============================================================================

/// Full frame cycle simulating the complete CPU-to-GPU pipeline for textures.
///
/// The asset pipeline loads textures, appends them to the TextureTable, then
/// stages the table through BufferRegistry for GPU upload. The GPU reads the
/// data back and uses the u32 indices for bindless texture lookups into the
/// `texture_2d_array<f32>`.
#[test]
fn full_frame_loop_add_stage_submit_readback() {
    let mut table = TextureTable::new();
    let mut registry = BufferRegistry::new(4096);

    // Load textures (simulating asset pipeline).
    let idx0 = table.add(tex(1024, 768, 10)).unwrap();
    let idx1 = table.add(tex(512, 512, 8)).unwrap();

    // Stage and submit for GPU upload.
    let (slot_index, written) = table.stage(&mut registry)
        .expect("Must acquire staging slot");
    assert_eq!(
        written,
        TEXTURE_TABLE_ENTRY_SIZE * 2,
        "Two entries, 48 bytes"
    );

    assert!(matches!(
        registry.submit_staging(slot_index, written),
        SubmitResult::Submitted
    ));

    // GPU side: acquire the ready slot and read the data.
    let read_idx = registry.acquire_reading()
        .expect("Must have a ready slot for GPU to read");
    let slot = registry.slot(read_idx).unwrap();

    // Verify total byte size.
    assert_eq!(slot.size(), TEXTURE_TABLE_ENTRY_SIZE * 2);
    let bytes = slot.as_slice();

    // Entry 0 at offset 0: width=1024 (0x400), height=768 (0x300),
    //                      mip_levels=10, format=0, layer_count=1, flags=1
    assert_eq!(&bytes[0..4], &[0x00, 0x04, 0, 0], "entry[0].width=1024");
    assert_eq!(&bytes[4..8], &[0x00, 0x03, 0, 0], "entry[0].height=768");
    assert_eq!(&bytes[8..12], &[10, 0, 0, 0], "entry[0].mip_levels=10");
    assert_eq!(&bytes[12..16], &[0, 0, 0, 0], "entry[0].format=0");
    assert_eq!(&bytes[16..20], &[1, 0, 0, 0], "entry[0].layer_count=1");
    assert_eq!(&bytes[20..24], &[1, 0, 0, 0], "entry[0].flags=1");

    // Entry 1 at offset 24: width=512 (0x200), height=512 (0x200)
    assert_eq!(&bytes[24..28], &[0x00, 0x02, 0, 0], "entry[1].width=512");
    assert_eq!(&bytes[28..32], &[0x00, 0x02, 0, 0], "entry[1].height=512");
    assert_eq!(&bytes[32..36], &[8, 0, 0, 0], "entry[1].mip_levels=8");

    // GPU would use indices 0 and 1 to reference these textures.
    assert_eq!(idx0, 0, "GPU references texture A by index 0");
    assert_eq!(idx1, 1, "GPU references texture B by index 1");
}

// =============================================================================
// SECTION 7 -- Incremental texture loading across multiple frames
// =============================================================================

/// Incremental loading: textures are added across multiple frames, each frame
/// stages the updated table for GPU upload. All indices remain stable and
/// correct across frames.
#[test]
fn incremental_texture_loading_across_frames() {
    let mut table = TextureTable::new();
    let mut registry = BufferRegistry::new(16384);

    // --- Frame 1: load 3 textures, stage, submit -----------------------------
    let idx_a = table.add(tex(1024, 768, 10)).unwrap();
    let idx_b = table.add(tex(512, 512, 8)).unwrap();
    let idx_c = table.add(tex(2048, 2048, 12)).unwrap();

    // Stage frame 1.
    let (slot_1, written_1) = table.stage(&mut registry)
        .expect("Frame 1: must acquire slot");
    assert_eq!(
        written_1,
        TEXTURE_TABLE_ENTRY_SIZE * 3,
        "Frame 1: 3 entries (72 bytes)"
    );
    assert!(matches!(registry.submit_staging(slot_1, written_1), SubmitResult::Submitted));

    // GPU reads frame 1.
    let read_1 = registry.acquire_reading().expect("Frame 1: GPU must acquire");
    let data_1 = registry.slot(read_1).unwrap();
    assert_eq!(data_1.size(), TEXTURE_TABLE_ENTRY_SIZE * 3);
    // Verify entry 0 width via bytes.
    assert_eq!(&data_1.as_slice()[0..4], &[0x00, 0x04, 0, 0]);

    // --- Frame 2: load 2 more textures, stage again --------------------------
    let idx_d = table.add(tex(4096, 4096, 14)).unwrap();
    let idx_e = table.add(tex(128, 128, 1)).unwrap();

    // Old indices must still be valid.
    assert_eq!(idx_a, 0, "Index 0 stable across frames");
    assert_eq!(idx_b, 1, "Index 1 stable across frames");
    assert_eq!(idx_c, 2, "Index 2 stable across frames");
    // New indices continue sequentially.
    assert_eq!(idx_d, 3, "New texture gets next sequential index");
    assert_eq!(idx_e, 4, "New texture gets next sequential index");

    assert_eq!(table.live_count(), 5, "5 live textures across two frames");

    // Stage frame 2 (now 5 entries).
    let (slot_2, written_2) = table.stage(&mut registry)
        .expect("Frame 2: must acquire slot");
    assert_eq!(
        written_2,
        TEXTURE_TABLE_ENTRY_SIZE * 5,
        "Frame 2: 5 entries (120 bytes)"
    );
    assert!(matches!(registry.submit_staging(slot_2, written_2), SubmitResult::Submitted));

    // GPU reads frame 2 -- all 5 textures available.
    let read_2 = registry.acquire_reading().expect("Frame 2: GPU must acquire");
    let data_2 = registry.slot(read_2).unwrap();
    assert_eq!(data_2.size(), TEXTURE_TABLE_ENTRY_SIZE * 5);

    // Verify each entry via the CPU-side table.
    assert_eq!(table.get(0).unwrap().width, 1024, "idx_a preserved");
    assert_eq!(table.get(1).unwrap().width, 512, "idx_b preserved");
    assert_eq!(table.get(2).unwrap().width, 2048, "idx_c preserved");
    assert_eq!(table.get(3).unwrap().width, 4096, "idx_d correct");
    assert_eq!(table.get(4).unwrap().width, 128, "idx_e correct");
    assert_eq!(table.get(4).unwrap().mip_levels, 1, "idx_e mips correct");
}

// =============================================================================
// SECTION 8 -- Hole preserves other indices after removal
// =============================================================================

/// Removing a texture creates a zeroed hole but does NOT change the indices of
/// other textures. This preserves GPU-side references that may be in-flight.
#[test]
fn remove_creates_hole_does_not_invalidate_other_indices() {
    let mut table = TextureTable::new();

    let idx_a = table.add(tex(1024, 768, 10)).unwrap();
    let idx_b = table.add(tex(512, 512, 8)).unwrap();
    let idx_c = table.add(tex(2048, 2048, 12)).unwrap();

    // Remove the middle texture.
    assert_eq!(table.remove(idx_b), TextureRemoveResult::Removed);

    // live_count decreased.
    assert_eq!(table.live_count(), 2, "Two live textures remain");

    // All indices unchanged (no shift).
    assert_eq!(table.len(), 3, "Length unchanged after removal (hole)");
    assert_eq!(table.get(idx_a).unwrap().width, 1024, "Texture A index 0 stable");
    assert_eq!(table.get(idx_c).unwrap().width, 2048, "Texture C index 2 stable");

    // The removed slot is a zero hole.
    let hole = table.get(idx_b).unwrap();
    assert!(hole.is_zero(), "Removed entry must be a hole");
}

/// Removing a texture that was already removed (a hole) returns NotFound.
#[test]
fn remove_hole_returns_not_found() {
    let mut table = TextureTable::new();
    let idx = table.add(tex(100, 100, 1)).unwrap();
    assert_eq!(table.remove(idx), TextureRemoveResult::Removed);
    assert_eq!(
        table.remove(idx),
        TextureRemoveResult::NotFound,
        "Removing a hole must return NotFound"
    );
}

/// Removing a nonexistent index returns NotFound.
#[test]
fn remove_nonexistent_returns_not_found() {
    let mut table = TextureTable::new();
    assert_eq!(
        table.remove(0),
        TextureRemoveResult::NotFound,
        "Removing index 0 from empty table must be NotFound"
    );
    assert_eq!(
        table.remove(u32::MAX),
        TextureRemoveResult::NotFound,
        "Removing u32::MAX must be NotFound"
    );
}

// =============================================================================
// SECTION 9 -- is_full() at MAX_BINDLESS_TEXTURES
// =============================================================================

/// is_full() must return true only when the table has reached MAX_BINDLESS_TEXTURES
/// entries and no free slots remain.
#[test]
fn is_full_reflects_capacity_limit() {
    let mut table = TextureTable::with_capacity(4);
    for _ in 0..4u32 {
        table.add(tex(100, 100, 1)).unwrap();
    }
    // Table has capacity for MAX_BINDLESS_TEXTURES (4096), so 4 adds does
    // not fill it. Verify is_full is false.
    assert!(!table.is_full());
    assert_eq!(table.live_count(), 4);
}

/// An empty table with free slots is not full.
#[test]
fn empty_table_is_not_full() {
    let table = TextureTable::new();
    assert!(!table.is_full());
}

// =============================================================================
// SECTION 10 -- add returns None when table is full
// =============================================================================

/// add() must return None when the table has reached MAX_BINDLESS_TEXTURES
/// and no free slots remain in the free-list.
#[test]
fn add_returns_none_when_table_full() {
    let mut table = TextureTable::with_capacity(MAX_BINDLESS_TEXTURES);

    // Fill the table to max capacity.
    for i in 0..MAX_BINDLESS_TEXTURES as u32 {
        let result = table.add(tex(i, i, 1));
        assert!(result.is_some(), "add {} must succeed", i);
    }

    assert!(table.is_full(), "Table must be full after {} adds", MAX_BINDLESS_TEXTURES);
    assert_eq!(table.live_count(), MAX_BINDLESS_TEXTURES);
    assert_eq!(table.free_count(), 0);

    // One more must fail.
    assert!(
        table.add(tex(0, 0, 0)).is_none(),
        "add must return None when table is full"
    );
}

/// add() on a table with all free-list slots recycled but no room to extend
/// returns None.
#[test]
fn add_returns_none_when_full_with_recycled_slots() {
    let mut table = TextureTable::with_capacity(MAX_BINDLESS_TEXTURES);

    // Fill to max.
    for i in 0..MAX_BINDLESS_TEXTURES as u32 {
        table.add(tex(i, i, 1)).unwrap();
    }
    assert!(table.is_full());

    // Remove one to free a slot -- table no longer full.
    table.remove(0);
    assert!(!table.is_full());

    // Add reuses the freed slot.
    let recycled = table.add(tex(9999, 9999, 1)).unwrap();
    assert_eq!(recycled, 0, "Must reuse freed slot at index 0");
    assert!(table.is_full());

    // One more must fail.
    assert!(
        table.add(tex(0, 0, 0)).is_none(),
        "add must return None when table is full even after recycle"
    );
}

// =============================================================================
// SECTION 11 -- Full table remove then re-add
// =============================================================================

/// Remove from a full table and re-add via free-list.
#[test]
fn full_table_remove_and_reuse() {
    let mut table = TextureTable::with_capacity(MAX_BINDLESS_TEXTURES);

    // Fill to max.
    for i in 0..MAX_BINDLESS_TEXTURES as u32 {
        table.add(tex(i, i, 1)).unwrap();
    }
    assert!(table.is_full());

    // Remove the last 100 entries.
    for i in (MAX_BINDLESS_TEXTURES - 100) as u32..MAX_BINDLESS_TEXTURES as u32 {
        assert_eq!(table.remove(i), TextureRemoveResult::Removed);
    }
    assert_eq!(table.free_count(), 100);
    assert!(!table.is_full());

    // Re-add 100, should reuse free-list slots in LIFO order (last removed pops first).
    for i in 0..100u32 {
        let idx = table.add(tex(i + 9999, 0, 1)).unwrap();
        // LIFO: last removed was MAX_BINDLESS_TEXTURES-1, so that pops first.
        assert!(
            idx >= (MAX_BINDLESS_TEXTURES - 100) as u32,
            "Reused index {} should be in the freed range",
            idx
        );
    }
    assert_eq!(table.free_count(), 0);
    assert!(table.is_full());
    assert_eq!(table.live_count(), MAX_BINDLESS_TEXTURES);
}

// =============================================================================
// SECTION 12 -- insert_at for specific index assignment
// =============================================================================

/// insert_at places a texture at a specific index, extending the entries vec
/// with zero holes to fill the gap if needed.
#[test]
fn insert_at_specific_index() {
    let mut table = TextureTable::new();
    table.add(tex(100, 100, 1)).unwrap();
    table.add(tex(200, 200, 2)).unwrap();

    // Insert at index 5.
    assert!(table.insert_at(5, tex(999, 999, 9)));

    // Table has 6 entries (indices 0..5).
    assert_eq!(table.len(), 6);
    assert_eq!(table.live_count(), 3);

    // Index 0 and 1 have original data.
    assert_eq!(table.get(0).unwrap().width, 100);
    assert_eq!(table.get(1).unwrap().width, 200);

    // Indices 2..4 are zero holes.
    for i in 2..5u32 {
        assert!(table.get(i).unwrap().is_zero(), "Index {} must be a zero hole", i);
    }

    // Index 5 has the inserted data.
    let entry = table.get(5).unwrap();
    assert_eq!(entry.width, 999);
    assert_eq!(entry.height, 999);
    assert_eq!(entry.mip_levels, 9);
}

/// insert_at on index 0 of an empty table produces a single-entry table.
#[test]
fn insert_at_index_zero_on_empty() {
    let mut table = TextureTable::new();
    assert!(table.is_empty());

    assert!(table.insert_at(0, tex(42, 84, 3)));
    assert_eq!(table.len(), 1);
    assert_eq!(table.live_count(), 1);
    assert_eq!(table.get(0).unwrap().width, 42);
}

/// insert_at with a zero entry does not increment live_count.
#[test]
fn insert_at_zero_entry_does_not_increment_live_count() {
    let mut table = TextureTable::new();
    table.insert_at(0, TextureTableEntry::zero());
    assert_eq!(table.len(), 1);
    assert_eq!(table.live_count(), 0, "Zero entry must not increment live_count");
    assert!(table.get(0).unwrap().is_zero());
}

/// insert_at fills a hole previously created by remove.
#[test]
fn insert_at_fills_hole() {
    let mut table = TextureTable::new();
    let idx = table.add(tex(100, 100, 1)).unwrap();
    assert_eq!(idx, 0);
    table.remove(idx);
    assert_eq!(table.live_count(), 0);

    // insert_at the hole.
    table.insert_at(0, tex(999, 888, 7));
    assert_eq!(table.live_count(), 1);
    assert_eq!(table.get(0).unwrap().width, 999);
}

// =============================================================================
// SECTION 13 -- insert_at beyond MAX_BINDLESS_TEXTURES returns false
// =============================================================================

/// insert_at at MAX_BINDLESS_TEXTURES or beyond must return false, preserving
/// the hard cap.
#[test]
fn insert_at_beyond_max_returns_false() {
    let mut table = TextureTable::with_capacity(4);
    assert!(
        !table.insert_at(MAX_BINDLESS_TEXTURES as u32, tex(1, 1, 1)),
        "insert_at(MAX_BINDLESS_TEXTURES) must return false"
    );
    assert!(
        !table.insert_at(u32::MAX, tex(1, 1, 1)),
        "insert_at(u32::MAX) must return false"
    );
}

/// insert_at at the last valid index (MAX_BINDLESS_TEXTURES - 1) must succeed.
#[test]
fn insert_at_last_valid_index_succeeds() {
    let mut table = TextureTable::new();
    let last = (MAX_BINDLESS_TEXTURES - 1) as u32;
    assert!(
        table.insert_at(last, tex(1, 1, 1)),
        "insert_at(last valid index) must succeed"
    );
    assert_eq!(table.len(), MAX_BINDLESS_TEXTURES);
    assert_eq!(table.live_count(), 1);
    assert_eq!(table.get(last).unwrap().width, 1);
}

// =============================================================================
// SECTION 14 -- Clear and re-add: indices restart from 0
// =============================================================================

/// After clear(), the table is fully reset. New textures get indices starting
/// from 0 again.
#[test]
fn clear_and_reload_indices_restart() {
    let mut table = TextureTable::new();

    // Load scene A (3 textures).
    let scene_a: Vec<u32> = (0..3).map(|i| table.add(tex(i * 100, 100, 1)).unwrap()).collect();
    assert_eq!(scene_a, [0, 1, 2], "Scene A indices 0..2");

    // Clear (simulating scene unload).
    table.clear();
    assert!(table.is_empty());
    assert_eq!(table.live_count(), 0);
    assert_eq!(table.len(), 0);
    assert_eq!(table.free_count(), 0);

    // Load scene B -- indices restart from 0.
    let scene_b: Vec<u32> = (0..2).map(|i| table.add(tex(5000 + i * 100, 100, 1)).unwrap()).collect();
    assert_eq!(scene_b, [0, 1], "Scene B indices restart from 0");

    // Scene B data is correct at indices 0 and 1.
    assert_eq!(table.get(0).unwrap().width, 5000);
    assert_eq!(table.get(1).unwrap().width, 5100);
}

/// Staging after clear: the table transitions from populated -> empty ->
/// repopulated without errors.
#[test]
fn clear_then_stage_works_correctly() {
    let mut table = TextureTable::new();
    let mut registry = BufferRegistry::new(4096);

    // Load initial textures and stage.
    table.add(tex(100, 100, 1)).unwrap();
    let (s0, w0) = table.stage(&mut registry).expect("Initial stage");
    assert!(matches!(registry.submit_staging(s0, w0), SubmitResult::Submitted));

    // Clear.
    table.clear();

    // Empty table cannot be staged.
    assert!(
        table.stage(&mut registry).is_none(),
        "Empty table after clear must return None"
    );

    // Reload with different data.
    table.add(tex(9999, 8888, 7)).unwrap();

    // Stage the new data.
    let (s1, w1) = table.stage(&mut registry).expect("Repopulated stage");
    assert_eq!(
        w1,
        TEXTURE_TABLE_ENTRY_SIZE,
        "One entry after repopulation"
    );
    assert!(matches!(registry.submit_staging(s1, w1), SubmitResult::Submitted));

    // GPU reads back -- must be the new data, not the old.
    let read_idx = registry.acquire_reading().expect("Must acquire read slot");
    let bytes = registry.slot(read_idx).unwrap().as_slice();
    assert_eq!(&bytes[0..4], &[0x0F, 0x27, 0, 0], "width=9999");
    assert_eq!(&bytes[4..8], &[0xB8, 0x22, 0, 0], "height=8888");
}

// =============================================================================
// SECTION 15 -- update preserves index
// =============================================================================

/// Updating a texture entry in place preserves its index. This allows the asset
/// pipeline to stream updated texture data (e.g., mip level changes) without
/// invalidating GPU-side references.
#[test]
fn update_preserves_index() {
    let mut table = TextureTable::new();

    let idx = table.add(tex(100, 200, 3)).unwrap();
    assert_eq!(idx, 0, "Initial index is 0");

    // Update to new texture data.
    assert!(table.update(idx, tex(1920, 1080, 11)), "Update must succeed");

    // Index unchanged.
    assert_eq!(table.len(), 1, "Length unchanged after update");
    assert_eq!(table.live_count(), 1, "Live count unchanged after update");

    // Entry has new data.
    let entry = table.get(idx).unwrap();
    assert_eq!(entry.width, 1920, "width updated");
    assert_eq!(entry.height, 1080, "height updated");
    assert_eq!(entry.mip_levels, 11, "mip_levels updated");
}

/// update on a nonexistent index returns false.
#[test]
fn update_out_of_range_returns_false() {
    let mut table = TextureTable::new();
    assert!(
        !table.update(0, tex(1, 1, 1)),
        "Update on empty table must return false"
    );
    assert!(
        !table.update(u32::MAX, tex(1, 1, 1)),
        "Update on u32::MAX must return false"
    );
}

// =============================================================================
// SECTION 16 -- update live_count tracking
// =============================================================================

/// update correctly adjusts live_count and free-list when transitioning between
/// zero and non-zero entries.
#[test]
fn update_live_count_tracking() {
    let mut table = TextureTable::new();

    // Add a live entry.
    let idx = table.add(tex(1, 2, 3)).unwrap();
    assert_eq!(table.live_count(), 1);

    // Update to zero: live_count decrements, index goes to free-list.
    table.update(idx, TextureTableEntry::zero());
    assert_eq!(table.live_count(), 0);
    assert_eq!(table.free_count(), 1);

    // Update from zero to non-zero: live_count increments, free-list cleared.
    table.update(idx, tex(7, 8, 9));
    assert_eq!(table.live_count(), 1);
    assert_eq!(table.free_count(), 0);

    // Update non-zero to different non-zero: live_count unchanged.
    table.update(idx, tex(13, 14, 15));
    assert_eq!(table.live_count(), 1);

    // Update zero to zero: no change.
    table.update(idx, TextureTableEntry::zero());
    assert_eq!(table.live_count(), 0);
    assert_eq!(table.free_count(), 1);
    table.update(idx, TextureTableEntry::zero());
    assert_eq!(table.live_count(), 0);
}

// =============================================================================
// SECTION 17 -- Empty table properties
// =============================================================================

/// An empty table has zero live count, zero length, and empty byte/slice views.
#[test]
fn empty_table_properties() {
    let table = TextureTable::new();
    assert!(table.is_empty());
    assert_eq!(table.live_count(), 0);
    assert_eq!(table.len(), 0);
    assert_eq!(table.free_count(), 0);
    assert!(!table.is_full());
    assert!(table.as_slice().is_empty());
    assert!(table.as_bytes().is_empty());
}

/// with_capacity creates an empty table with pre-allocated capacity.
#[test]
fn with_capacity_creates_empty_table() {
    let table = TextureTable::with_capacity(2048);
    assert!(table.is_empty());
    assert_eq!(table.len(), 0);
    assert_eq!(table.live_count(), 0);
}

/// with_capacity(0) clamps to minimum 1 (valid empty table).
#[test]
fn with_capacity_zero_clamps() {
    let table = TextureTable::with_capacity(0);
    assert!(table.is_empty());
    assert_eq!(table.len(), 0);
}

// =============================================================================
// SECTION 18 -- Stage empty table returns None
// =============================================================================

/// An empty texture table cannot be staged (no data to upload).
#[test]
fn empty_table_stage_returns_none() {
    let table = TextureTable::new();
    let mut registry = BufferRegistry::new(4096);

    assert!(
        table.stage(&mut registry).is_none(),
        "Empty table must return None from stage()"
    );
    assert!(
        !table.stage_and_submit(&mut registry),
        "Empty table must return false from stage_and_submit()"
    );
}

// =============================================================================
// SECTION 19 -- Display and formatting
// =============================================================================

/// Display output for the table must show entries, live, free, and max counts.
#[test]
fn texture_table_display_shows_state() {
    let mut table = TextureTable::new();
    let empty_display = format!("{}", table);
    assert!(
        empty_display.starts_with("TextureTable("),
        "Display must start with 'TextureTable('"
    );
    assert!(
        empty_display.contains("entries=0"),
        "Empty table display must show entries=0"
    );
    assert!(
        empty_display.contains("live=0"),
        "Empty table display must show live=0"
    );
    assert!(
        empty_display.contains("free="),
        "Display must show free count"
    );
    assert!(
        empty_display.contains("max="),
        "Display must show max textures"
    );

    table.add(tex(1024, 768, 10)).unwrap();
    table.add(tex(512, 512, 8)).unwrap();
    let populated_display = format!("{}", table);
    assert!(
        populated_display.contains("entries=2"),
        "Display must show entries=2"
    );
    assert!(
        populated_display.contains("live=2"),
        "Display must show live=2"
    );
}

/// Display for TextureTableEntry must show all field values.
#[test]
fn texture_table_entry_display_shows_fields() {
    let e = tex_full(1024, 768, 10, 1, 4, 0xFF);
    let s = format!("{}", e);
    assert!(s.contains("1024"), "Display must contain width");
    assert!(s.contains("768"), "Display must contain height");
    assert!(s.contains("mips=10"), "Display must contain mip levels");
    assert!(s.contains("fmt="), "Display must contain format");
    assert!(s.contains("flags=0x"), "Display must contain flags hex");
}

// =============================================================================
// SECTION 20 -- Constants
// =============================================================================

/// MAX_BINDLESS_TEXTURES must be 4096, matching typical hardware limits for
/// texture_2d_array<f32> array layers.
#[test]
fn max_bindless_textures_constant() {
    assert_eq!(
        MAX_BINDLESS_TEXTURES, 4096,
        "MAX_BINDLESS_TEXTURES must be 4096"
    );
}

/// TEXTURE_TABLE_ENTRY_SIZE must be 24 bytes (6 x u32, 4-byte aligned),
/// matching the WGSL struct layout.
#[test]
fn texture_table_entry_size_constant() {
    assert_eq!(
        TEXTURE_TABLE_ENTRY_SIZE, 24,
        "TEXTURE_TABLE_ENTRY_SIZE must be 24 bytes (6 * u32)"
    );
}

/// DEFAULT_TEXTURE_TABLE_CAPACITY constant is exported and positive.
#[test]
fn default_texture_table_capacity_constant_is_positive() {
    assert!(
        DEFAULT_TEXTURE_TABLE_CAPACITY > 0,
        "DEFAULT_TEXTURE_TABLE_CAPACITY must be > 0"
    );
    assert!(
        DEFAULT_TEXTURE_TABLE_CAPACITY >= 1024,
        "DEFAULT_TEXTURE_TABLE_CAPACITY should be at least 1024 for real scenes"
    );
}

// =============================================================================
// SECTION 21 -- Dense loading at scale (1000+ entries)
// =============================================================================

/// Loading 1000 textures -- simulating a real scene load -- must produce
/// sequential indices 0..999 and all entries must be retrievable by index.
#[test]
fn dense_loading_at_scale_1000_textures() {
    let mut table = TextureTable::with_capacity(1024);

    // Simulate bulk asset pipeline load.
    for i in 0..1000u32 {
        let idx = table.add(tex_full(
            i * 10,            // width
            i * 5,             // height
            (i % 12) + 1,      // mip_levels 1..12
            i % 256,           // format cycles 0..255
            1 + (i % 4),       // layer_count 1..4
            1,                 // valid
        )).unwrap();
        assert_eq!(
            idx, i,
            "Texture {} must get sequential index {}, got {}",
            i, i, idx
        );
    }

    assert_eq!(table.live_count(), 1000);
    assert_eq!(table.len(), 1000);

    // All 1000 textures retrievable by index.
    for i in 0..1000u32 {
        let entry = table.get(i)
            .unwrap_or_else(|| panic!("Texture at index {} must be retrievable", i));
        assert_eq!(entry.width, i * 10, "width for texture {}", i);
        assert_eq!(entry.height, i * 5, "height for texture {}", i);
        assert_eq!(entry.mip_levels, (i % 12) + 1, "mip_levels for texture {}", i);
        assert_eq!(entry.format, i % 256, "format for texture {}", i);
        assert_eq!(entry.layer_count, 1 + (i % 4), "layer_count for texture {}", i);
        assert_eq!(entry.flags, 1, "flags for texture {}", i);
    }

    // as_bytes must produce the correct total size.
    assert_eq!(
        table.as_bytes().len(),
        1000 * TEXTURE_TABLE_ENTRY_SIZE,
        "Byte length must be 1000 * 24 = 24000"
    );

    // as_slice must have the correct length.
    assert_eq!(table.as_slice().len(), 1000);
}

// =============================================================================
// SECTION 22 -- Bulk staging integrity through BufferRegistry
// =============================================================================

/// Stage and submit a table with diverse entries through the full triple-buffer
/// pipeline and verify that every byte arrives intact at the GPU side.
/// This is the end-to-end data-path validation for the texture table.
#[test]
fn bulk_staging_integrity_full_pipeline() {
    let mut table = TextureTable::with_capacity(128);
    let mut registry = BufferRegistry::new(65536); // 64 KiB

    // Load 50 textures with deterministic patterns.
    for i in 0..50u32 {
        table.add(tex_full(
            i * 100,       // width
            (i + 1) * 50,  // height
            (i % 12) + 1,  // mip_levels 1..12
            i % 16,        // format 0..15
            1 + (i % 4),   // layer_count 1..4
            if i % 2 == 0 { 1 } else { 0 }, // flags: even=valid, odd=invalid
        )).unwrap();
    }

    // Stage the full table.
    let (slot_index, written) = table.stage(&mut registry).expect("Must acquire slot");
    assert_eq!(written, 50 * TEXTURE_TABLE_ENTRY_SIZE);
    assert!(matches!(registry.submit_staging(slot_index, written), SubmitResult::Submitted));

    // Read back and verify every byte.
    let read_idx = registry.acquire_reading().expect("Must acquire read slot");
    let bytes = registry.slot(read_idx).unwrap().as_slice();
    assert_eq!(bytes.len(), 50 * TEXTURE_TABLE_ENTRY_SIZE);

    for i in 0..50u32 {
        let offset = (i as usize) * TEXTURE_TABLE_ENTRY_SIZE;

        let actual_width = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        let actual_height = u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap());
        let actual_mip_levels = u32::from_le_bytes(bytes[offset + 8..offset + 12].try_into().unwrap());
        let actual_format = u32::from_le_bytes(bytes[offset + 12..offset + 16].try_into().unwrap());
        let actual_layer_count = u32::from_le_bytes(bytes[offset + 16..offset + 20].try_into().unwrap());
        let actual_flags = u32::from_le_bytes(bytes[offset + 20..offset + 24].try_into().unwrap());

        assert_eq!(actual_width, i * 100, "entry[{}].width", i);
        assert_eq!(actual_height, (i + 1) * 50, "entry[{}].height", i);
        assert_eq!(actual_mip_levels, (i % 12) + 1, "entry[{}].mip_levels", i);
        assert_eq!(actual_format, i % 16, "entry[{}].format", i);
        assert_eq!(actual_layer_count, 1 + (i % 4), "entry[{}].layer_count", i);
        assert_eq!(actual_flags, if i % 2 == 0 { 1 } else { 0 }, "entry[{}].flags", i);
    }
}

// =============================================================================
// SECTION 23 -- Free-list no-duplicates invariant
// =============================================================================

/// The free-list must never contain duplicate indices.
#[test]
fn free_list_no_duplicates() {
    let mut table = TextureTable::new();

    for i in 0..10u32 {
        table.add(tex(i, i, 1)).unwrap();
    }

    // Remove all entries.
    for i in 0..10u32 {
        table.remove(i);
    }
    assert_eq!(table.free_count(), 10);

    // Sort and dedup to verify no duplicates. We cannot access the free_list
    // directly, but we can verify the invariant indirectly: add 10 textures and
    // check all indices are unique and in the expected range.
    let mut seen = std::collections::HashSet::new();
    for _ in 0..10u32 {
        let idx = table.add(tex(1, 1, 1)).unwrap();
        assert!(seen.insert(idx), "Duplicate index {} found in free-list", idx);
    }
    assert_eq!(seen.len(), 10, "All 10 indices must be unique");
    assert_eq!(table.free_count(), 0);
}

// =============================================================================
// SECTION 24 -- reserve() extends capacity
// =============================================================================

/// reserve() ensures subsequent add() calls do not reallocate. This is useful
/// when the total texture count is known ahead of time.
#[test]
fn reserve_after_add_does_not_break_existing_indices() {
    let mut table = TextureTable::new();

    table.add(tex(100, 100, 1)).unwrap();
    table.add(tex(200, 200, 2)).unwrap();

    // Reserve space for 500 more.
    table.reserve(500);

    // Existing indices still valid.
    assert_eq!(table.get(0).unwrap().width, 100);
    assert_eq!(table.get(1).unwrap().width, 200);

    // Adds after reserve continue correctly.
    for i in 2..102u32 {
        let idx = table.add(tex(i * 10, 0, 1)).unwrap();
        assert_eq!(idx, i, "Reserved space: sequential index {}", i);
    }
    assert_eq!(table.live_count(), 102);
}

/// reserve on an empty table works.
#[test]
fn reserve_on_empty_table() {
    let mut table = TextureTable::new();
    table.reserve(50);
    for i in 0..50u32 {
        let idx = table.add(tex(i, 0, 1)).unwrap();
        assert_eq!(idx, i);
    }
    assert_eq!(table.len(), 50);
}

/// reserve(0) is a no-op.
#[test]
fn reserve_zero_is_noop() {
    let mut table = TextureTable::new();
    table.add(tex(1, 1, 1)).unwrap();
    table.reserve(0);
    assert_eq!(table.live_count(), 1);
}

// =============================================================================
// SECTION 25 -- as_bytes and as_slice consistency
// =============================================================================

/// The byte representation and the slice representation must agree on the
/// number of entries.
#[test]
fn as_bytes_and_as_slice_agree_on_count() {
    let mut table = TextureTable::new();

    for i in 0..10u32 {
        table.add(tex(i, i * 2, 1)).unwrap();
    }

    let bytes = table.as_bytes();
    let slice = table.as_slice();

    assert_eq!(slice.len() * TEXTURE_TABLE_ENTRY_SIZE, bytes.len());
    assert_eq!(slice.len(), 10);

    // The first field of each entry in bytes should match the slice.
    for (i, entry) in slice.iter().enumerate() {
        let width_bytes = u32::from_le_bytes(
            bytes[i * TEXTURE_TABLE_ENTRY_SIZE..i * TEXTURE_TABLE_ENTRY_SIZE + 4]
                .try_into()
                .unwrap(),
        );
        assert_eq!(
            width_bytes,
            entry.width,
            "Byte/slice agreement for entry[{}].width",
            i
        );
    }
}

/// as_bytes with a table that has holes: holes must be zero in the byte stream.
#[test]
fn as_bytes_with_holes() {
    let mut table = TextureTable::new();
    table.add(tex(100, 100, 1)).unwrap();
    let idx = table.add(tex(200, 200, 2)).unwrap();
    table.add(tex(300, 300, 3)).unwrap();
    table.remove(idx); // Index 1 becomes a hole.

    let bytes = table.as_bytes();
    assert_eq!(bytes.len(), TEXTURE_TABLE_ENTRY_SIZE * 3);

    // Entry 0 intact.
    assert_eq!(&bytes[0..4], &[100, 0, 0, 0], "entry 0 width intact");
    // Entry 1 is a hole (all zeros).
    for byte_offset in 0..TEXTURE_TABLE_ENTRY_SIZE {
        assert_eq!(
            bytes[TEXTURE_TABLE_ENTRY_SIZE + byte_offset], 0,
            "hole byte {} must be zero",
            byte_offset
        );
    }
    // Entry 2 intact.
    assert_eq!(&bytes[48..52], &[44, 1, 0, 0], "entry 2 width=300 (0x012C)");
}

// =============================================================================
// SECTION 26 -- Zero / default entries and is_zero()
// =============================================================================

/// Zero entry has all fields at zero and is_zero() returns true.
#[test]
fn zero_entry_is_zero() {
    let z = TextureTableEntry::zero();
    assert!(z.is_zero());
    assert_eq!(z.width, 0);
    assert_eq!(z.height, 0);
    assert_eq!(z.mip_levels, 0);
    assert_eq!(z.format, 0);
    assert_eq!(z.layer_count, 0);
    assert_eq!(z.flags, 0);
}

/// Default entry is a zero entry.
#[test]
fn default_entry_is_zero() {
    let d: TextureTableEntry = Default::default();
    assert!(d.is_zero());
    assert_eq!(d.width, 0);
}

/// Non-zero fields cause is_zero() to return false.
#[test]
fn non_zero_entry_is_not_zero() {
    let e = tex(1, 0, 0);
    assert!(!e.is_zero(), "width=1 must make is_zero false");

    let e2 = tex_full(0, 0, 0, 0, 0, 1);
    assert!(!e2.is_zero(), "flags=1 must make is_zero false");

    let e3 = tex_full(0, 0, 0, 0, 1, 0);
    assert!(!e3.is_zero(), "layer_count=1 must make is_zero false");
}

/// TextureTableEntry::new with all zeros must match zero().
#[test]
fn new_with_all_zeros_is_zero() {
    let e = TextureTableEntry::new(0, 0, 0, 0, 0, 0);
    assert!(e.is_zero());
}

// =============================================================================
// SECTION 27 -- Staging after remove and update mutations
// =============================================================================

/// Staging after complex mutation sequence: remove, update, insert_at.
/// The GPU buffer must faithfully represent the mutated state.
#[test]
fn staging_after_complex_mutations() {
    let mut registry = BufferRegistry::new(4096);
    let mut table = TextureTable::new();

    for i in 0..5u32 {
        table.add(tex(i * 10, i * 20, (i + 1) as u32)).unwrap();
    }

    // Mutate: remove index 0, remove index 3, update index 1, insert_at 5.
    table.remove(0);
    table.remove(3);
    table.update(1, tex(999, 888, 7)).unwrap();
    table.insert_at(5, tex(42, 84, 3));

    let (slot_idx, byte_size) = table.stage(&mut registry).expect("Staging must succeed");
    assert_eq!(byte_size, 6 * TEXTURE_TABLE_ENTRY_SIZE);

    // Read back and verify.
    assert!(matches!(registry.submit_staging(slot_idx, byte_size), SubmitResult::Submitted));
    let read_idx = registry.acquire_reading().expect("Must acquire read slot");
    let staged: &[TextureTableEntry] = unsafe {
        std::slice::from_raw_parts(
            registry.slot(read_idx).unwrap().as_slice().as_ptr() as *const TextureTableEntry,
            6,
        )
    };
    assert!(staged[0].is_zero(), "entry 0 removed -> zero hole");
    assert_eq!(staged[1].width, 999, "entry 1 updated");
    assert_eq!(staged[2].width, 20, "entry 2 intact");
    assert!(staged[3].is_zero(), "entry 3 removed -> zero hole");
    assert_eq!(staged[4].width, 40, "entry 4 intact");
    assert_eq!(staged[5].width, 42, "entry 5 insert_at");
}

/// Multiple stage-and-submit cycles across frames.
#[test]
fn multiple_stage_and_submit_cycles() {
    let mut registry = BufferRegistry::new(4096);
    let mut table = TextureTable::new();

    assert!(!table.stage_and_submit(&mut registry), "Empty table must not stage");

    // Cycle 1: add 3 textures.
    for i in 0..3u32 {
        table.add(tex(i, i, 1)).unwrap();
    }
    assert!(table.stage_and_submit(&mut registry));

    // Mutate and cycle 2.
    table.remove(1);
    table.add(tex(100, 100, 5)).unwrap();
    assert!(table.stage_and_submit(&mut registry));

    // Clear and cycle 3.
    table.clear();
    assert!(!table.stage_and_submit(&mut registry), "Cleared table must not stage");
}

/// Staging after update shows new data in the GPU buffer.
#[test]
fn staging_after_update_shows_new_data() {
    let mut table = TextureTable::new();
    let mut registry = BufferRegistry::new(4096);

    // Add a texture.
    let idx = table.add(tex(100, 200, 3)).unwrap();

    // Update index 0 with new dimensions.
    table.update(idx, tex(1920, 1080, 11)).unwrap();

    // Stage through BufferRegistry.
    let (slot_index, written) = table.stage(&mut registry).expect("Must acquire slot");
    assert!(matches!(registry.submit_staging(slot_index, written), SubmitResult::Submitted));

    let read_idx = registry.acquire_reading().expect("Must acquire read slot");
    let bytes = registry.slot(read_idx).unwrap().as_slice();

    // Verify updated data is what the GPU sees.
    assert_eq!(&bytes[0..4], &[0x80, 0x07, 0, 0], "width=1920 (0x0780)");
    assert_eq!(&bytes[4..8], &[0x38, 0x04, 0, 0], "height=1080 (0x0438)");
    assert_eq!(&bytes[8..12], &[11, 0, 0, 0], "mip_levels=11");
}

// =============================================================================
// SECTION 28 -- get_mut allows in-place mutation
// =============================================================================

/// get_mut returns a mutable reference for in-place mutation of flags or
/// other fields without changing the index.
#[test]
fn get_mut_updates_flags_in_place() {
    let mut table = TextureTable::new();

    let idx = table.add(tex(1024, 768, 10)).unwrap();
    assert_eq!(table.get(idx).unwrap().flags, 1, "Initially valid");

    // Toggle validity flag off via mutable reference.
    {
        let entry = table.get_mut(idx).expect("get_mut must succeed");
        entry.flags = 0;
    }

    // Verify the change.
    assert_eq!(table.get(idx).unwrap().flags, 0, "Flag toggled off");

    // Index unchanged.
    assert_eq!(idx, 0);
    assert_eq!(table.len(), 1);
}

/// get_mut on a hole or out-of-bounds returns None.
#[test]
fn get_mut_out_of_range_returns_none() {
    let mut table = TextureTable::new();
    assert!(table.get_mut(0).is_none(), "Empty table: get_mut(0) must be None");
    assert!(table.get_mut(u32::MAX).is_none(), "get_mut(u32::MAX) must be None");
}

// =============================================================================
// SECTION 29 -- Hybrid behavior: free-list interaction with insert_at
// =============================================================================

/// When a freed slot exists on the free-list and insert_at targets that slot,
/// the free-list entry must be removed (defensive cleanup in insert_at).
#[test]
fn insert_at_clears_free_list_entry() {
    let mut table = TextureTable::new();

    // Add a texture, remove it (goes to free-list).
    let idx = table.add(tex(100, 100, 1)).unwrap();
    table.remove(idx);
    assert_eq!(table.free_count(), 1);

    // insert_at the same index: should fill the hole and clear it from free-list.
    table.insert_at(idx, tex(999, 999, 9));
    assert_eq!(table.free_count(), 0, "insert_at must clear free-list entry for target index");
    assert_eq!(table.live_count(), 1);
    assert_eq!(table.get(idx).unwrap().width, 999);
}

// =============================================================================
// SECTION 30 -- Default trait implementations
// =============================================================================

/// TextureTable and TextureTableEntry both implement Default.
#[test]
fn texture_table_implements_default() {
    let table: TextureTable = Default::default();
    assert!(table.is_empty());
    assert_eq!(table.len(), 0);
}

/// TextureTableEntry is Copy + Clone.
#[test]
fn texture_table_entry_is_copy_and_clone() {
    let a = tex_full(1024, 768, 10, 1, 4, 0xFF);
    let b = a; // Copy.
    assert_eq!(a.width, b.width);
    assert_eq!(a.flags, b.flags);
    assert_eq!(a.format, b.format);

    let c = a.clone(); // Clone.
    assert_eq!(a.width, c.width);
    assert_eq!(a.layer_count, c.layer_count);
}

// =============================================================================
// SECTION 31 -- max_textures accessor
// =============================================================================

/// max_textures() must always return MAX_BINDLESS_TEXTURES (4096), regardless
/// of the initial capacity parameter.
#[test]
fn max_textures_is_always_4096() {
    let table = TextureTable::with_capacity(1);
    assert_eq!(
        table.max_textures(),
        MAX_BINDLESS_TEXTURES,
        "with_capacity(1) must still limit to 4096"
    );

    let large = TextureTable::with_capacity(MAX_BINDLESS_TEXTURES + 100);
    assert_eq!(
        large.max_textures(),
        MAX_BINDLESS_TEXTURES,
        "with_capacity(4196) must still limit to 4096"
    );

    let default_table = TextureTable::new();
    assert_eq!(
        default_table.max_textures(),
        MAX_BINDLESS_TEXTURES,
        "new() must limit to 4096"
    );
}

// =============================================================================
// SECTION 32 -- Hard cap enforcement via add after full to 4096
// =============================================================================

/// Fill all 4096 slots and verify that the 4097th add returns None.
#[test]
fn add_up_to_max_textures_then_fails() {
    let mut table = TextureTable::with_capacity(MAX_BINDLESS_TEXTURES);

    // Fill exactly to MAX_BINDLESS_TEXTURES.
    for i in 0..MAX_BINDLESS_TEXTURES as u32 {
        let result = table.add(tex(i, i * 2, (i % 12) + 1));
        assert!(result.is_some(), "add {} must succeed", i);
    }
    assert!(table.is_full());
    assert_eq!(table.live_count(), MAX_BINDLESS_TEXTURES);

    // Verify the last entry is correct.
    let last = table.get((MAX_BINDLESS_TEXTURES - 1) as u32).unwrap();
    assert_eq!(last.width, (MAX_BINDLESS_TEXTURES - 1) as u32);

    // 4097th add must fail.
    assert!(table.add(tex(0, 0, 0)).is_none());
}
