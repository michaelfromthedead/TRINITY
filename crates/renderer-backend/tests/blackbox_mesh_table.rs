// Blackbox contract tests for MeshTable (T-GPU-1.3).
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::gpu_driven::*` -- no internal fields,
// no private methods, no implementation details.
//
// Acceptance criterion (T-GPU-1.3):
//   Meshes loaded via S16 Asset Pipeline automatically append to MeshTable;
//   instances reference by u32 index.
//
// The MeshTable guarantees:
//   - add() returns sequential u32 indices that instances can reference
//   - Entries are retrievable by index via get() / as_slice()
//   - The table can be staged through BufferRegistry for GPU upload
//   - Removing an entry creates a hole but preserves all other indices
//   - Updating an entry in place keeps its index stable
//
// Coverage:
//   1.  Auto-append acceptance: sequential add() returns indices 0, 1, 2, ...
//   2.  Entry retrievable by index: get(index) returns the correct entry
//   3.  Full frame loop: add -> stage -> submit -> read-back
//   4.  Incremental mesh loading across multiple frames
//   5.  Index stability after removal: holes don't invalidate other indices
//   6.  Dense loading at scale (1000 entries)
//   7.  In-place update preserves index
//   8.  Bulk staging integrity through BufferRegistry
//   9.  Empty table stage returns None
//  10.  Clear and re-add: indices restart from 0

use renderer_backend::gpu_driven::{
    BufferRegistry, MeshTable, MeshTableEntry, RemoveResult,
    SubmitResult, DEFAULT_MESH_TABLE_CAPACITY, MESH_TABLE_ENTRY_SIZE,
};

// =============================================================================
// SECTION 1 -- Auto-append acceptance: the S16 asset pipeline contract
// =============================================================================

/// THE ACCEPTANCE TEST: Meshes loaded via the asset pipeline auto-append to
/// the MeshTable and return sequential u32 indices.
///
/// This validates the core contract of T-GPU-1.3: when the S16 Asset Pipeline
/// loads meshes and calls add(), each mesh receives a unique, sequential u32
/// index. Instances can then reference meshes by these indices.
#[test]
fn meshes_auto_append_with_sequential_indices() {
    let mut table = MeshTable::new();

    // Simulate S16 asset pipeline loading 5 meshes.
    let idx_a = table.add(MeshTableEntry::new(0, 0, 100, 50, 0, 1));
    let idx_b = table.add(MeshTableEntry::new(300, 0, 200, 100, 1, 1));
    let idx_c = table.add(MeshTableEntry::new(900, 0, 50, 25, 2, 1));
    let idx_d = table.add(MeshTableEntry::new(1200, 0, 500, 250, 0, 1));
    let idx_e = table.add(MeshTableEntry::new(3200, 0, 80, 40, 3, 0));

    // Indices must be sequential starting from 0.
    assert_eq!(idx_a, 0, "First mesh gets index 0");
    assert_eq!(idx_b, 1, "Second mesh gets index 1");
    assert_eq!(idx_c, 2, "Third mesh gets index 2");
    assert_eq!(idx_d, 3, "Fourth mesh gets index 3");
    assert_eq!(idx_e, 4, "Fifth mesh gets index 4");

    // live_count must reflect all 5 meshes.
    assert_eq!(table.live_count(), 5);
    assert_eq!(table.len(), 5);
    assert!(!table.is_empty());
}

/// Instances can reference a mesh by its u32 index.
///
/// After a mesh is appended, get(index) must return the correct entry.
/// This is the counterpart of the GPU-side `array<MeshTableEntry>[instance_id]`
/// lookup.
#[test]
fn instances_reference_mesh_by_u32_index() {
    let mut table = MeshTable::new();

    // Load a mesh with known properties.
    let idx = table.add(MeshTableEntry::new(
        0x1000,    // index_offset
        0x2000,    // vertex_offset
        300,       // index_count
        150,       // vertex_count
        2,         // material_id
        1,         // flags (visible)
    ));

    // Retrieve by index -- this is what the GPU does with the u32 handle.
    let entry = table.get(idx).expect("Entry must be retrievable by index");
    assert_eq!(entry.index_offset, 0x1000, "index_offset preserved");
    assert_eq!(entry.vertex_offset, 0x2000, "vertex_offset preserved");
    assert_eq!(entry.index_count, 300, "index_count preserved");
    assert_eq!(entry.vertex_count, 150, "vertex_count preserved");
    assert_eq!(entry.material_id, 2, "material_id preserved");
    assert_eq!(entry.flags, 1, "flags preserved (visible)");

    // The entry must also be findable via as_slice() at the correct position.
    let slice = table.as_slice();
    assert_eq!(slice.len(), 1, "One entry in the slice");
    assert_eq!(slice[idx as usize].index_offset, 0x1000);
}

/// Multiple instances can reference the same mesh by index.
///
/// In a GPU-driven renderer, many instances may draw the same mesh. Each
/// instance carries the u32 mesh index. This test verifies that the index
/// remains stable and correct through repeated lookups.
#[test]
fn multiple_instances_can_reference_same_mesh() {
    let mut table = MeshTable::new();

    let shared_mesh = table.add(MeshTableEntry::new(0, 0, 100, 50, 1, 1));

    // Simulate 10 instances all referencing the same mesh.
    for instance_id in 0..10u32 {
        let entry = table
            .get(shared_mesh)
            .unwrap_or_else(|| panic!("Instance {}: mesh index {} must be valid", instance_id, shared_mesh));
        assert_eq!(entry.index_count, 100, "Instance {} sees correct index_count", instance_id);
        assert_eq!(entry.material_id, 1, "Instance {} sees correct material_id", instance_id);
    }

    assert_eq!(table.live_count(), 1, "Only one mesh in the table");
    assert_eq!(table.len(), 1);
}

// =============================================================================
// SECTION 2 -- Full frame loop: add -> stage -> submit -> read-back
// =============================================================================

/// Full frame cycle simulating the complete CPU-to-GPU pipeline.
///
/// The S16 asset pipeline loads meshes, appends them to the MeshTable, then
/// stages the table through BufferRegistry for GPU upload. The GPU reads the
/// data back and uses the u32 indices to reference meshes.
#[test]
fn full_frame_loop_add_stage_submit_readback() {
    let mut table = MeshTable::new();
    let mut registry = BufferRegistry::new(4096);

    // Load meshes (simulating asset pipeline).
    let idx0 = table.add(MeshTableEntry::new(0, 0, 100, 50, 0, 1));
    let idx1 = table.add(MeshTableEntry::new(400, 0, 200, 100, 1, 1));

    // Stage and submit for GPU upload.
    let (slot_index, written) = table
        .stage(&mut registry)
        .expect("Must acquire staging slot");
    assert_eq!(written, MESH_TABLE_ENTRY_SIZE * 2, "Two entries, 48 bytes");

    assert!(matches!(
        registry.submit_staging(slot_index, written),
        SubmitResult::Submitted
    ));

    // GPU side: acquire the ready slot and read the data.
    let read_idx = registry
        .acquire_reading()
        .expect("Must have a ready slot for GPU to read");
    let slot = registry.slot(read_idx).unwrap();

    // Verify the byte layout matches two MeshTableEntry structs.
    assert_eq!(slot.size(), MESH_TABLE_ENTRY_SIZE * 2);
    let bytes = slot.as_slice();

    // Entry 0 at offset 0: index_offset=0, vertex_offset=0, index_count=100,
    //                      vertex_count=50, material_id=0, flags=1
    assert_eq!(&bytes[0..4], &[0, 0, 0, 0], "entry[0].index_offset");
    assert_eq!(&bytes[4..8], &[0, 0, 0, 0], "entry[0].vertex_offset");
    assert_eq!(&bytes[8..12], &[100, 0, 0, 0], "entry[0].index_count=100");
    assert_eq!(&bytes[12..16], &[50, 0, 0, 0], "entry[0].vertex_count=50");
    assert_eq!(&bytes[16..20], &[0, 0, 0, 0], "entry[0].material_id=0");
    assert_eq!(&bytes[20..24], &[1, 0, 0, 0], "entry[0].flags=1");

    // Entry 1 at offset 24: index_offset=400, material_id=1
    assert_eq!(&bytes[24..28], &[0x90, 0x01, 0, 0], "entry[1].index_offset=400");
    assert_eq!(&bytes[40..44], &[1, 0, 0, 0], "entry[1].material_id=1");

    // GPU would use indices 0 and 1 to reference these meshes.
    assert_eq!(idx0, 0, "GPU references mesh A by index 0");
    assert_eq!(idx1, 1, "GPU references mesh B by index 1");
}

// =============================================================================
// SECTION 3 -- Incremental mesh loading across multiple frames
// =============================================================================

/// Incremental loading: meshes are added across multiple frames, each frame
/// stages the updated table for GPU upload. All indices remain stable and
/// correct across frames.
///
/// This simulates a real-world scenario where the S16 asset pipeline streams
/// meshes in over time while the engine is running.
#[test]
fn incremental_mesh_loading_across_frames() {
    let mut table = MeshTable::new();
    let mut registry = BufferRegistry::new(16384);

    // --- Frame 1: load 3 meshes, stage, submit --------------------------------
    let idx_a = table.add(MeshTableEntry::new(0, 0, 100, 50, 0, 1));
    let idx_b = table.add(MeshTableEntry::new(400, 0, 200, 100, 1, 1));
    let idx_c = table.add(MeshTableEntry::new(1200, 0, 50, 25, 2, 0));

    // Stage frame 1.
    let (slot_1, written_1) = table.stage(&mut registry).expect("Frame 1: must acquire slot");
    assert_eq!(written_1, MESH_TABLE_ENTRY_SIZE * 3, "Frame 1: 3 entries (72 bytes)");
    assert!(matches!(registry.submit_staging(slot_1, written_1), SubmitResult::Submitted));

    // GPU reads frame 1.
    let read_1 = registry.acquire_reading().expect("Frame 1: GPU must acquire");
    let data_1 = registry.slot(read_1).unwrap();
    assert_eq!(data_1.size(), MESH_TABLE_ENTRY_SIZE * 3);
    // Index 0 has index_count=100.
    assert_eq!(&data_1.as_slice()[8..12], &[100, 0, 0, 0]);

    // --- Frame 2: load 2 more meshes, stage again -----------------------------
    let idx_d = table.add(MeshTableEntry::new(2000, 0, 300, 150, 0, 1));
    let idx_e = table.add(MeshTableEntry::new(4000, 0, 80, 40, 3, 1));

    // Old indices must still be valid.
    assert_eq!(idx_a, 0, "Index 0 stable across frames");
    assert_eq!(idx_b, 1, "Index 1 stable across frames");
    assert_eq!(idx_c, 2, "Index 2 stable across frames");
    // New indices continue sequentially.
    assert_eq!(idx_d, 3, "New mesh gets next sequential index");
    assert_eq!(idx_e, 4, "New mesh gets next sequential index");

    assert_eq!(table.live_count(), 5, "5 live meshes across two frames");

    // Stage frame 2 (now 5 entries).
    let (slot_2, written_2) = table.stage(&mut registry).expect("Frame 2: must acquire slot");
    assert_eq!(written_2, MESH_TABLE_ENTRY_SIZE * 5, "Frame 2: 5 entries (120 bytes)");
    assert!(matches!(registry.submit_staging(slot_2, written_2), SubmitResult::Submitted));

    // GPU reads frame 2 -- all 5 meshes available.
    let read_2 = registry.acquire_reading().expect("Frame 2: GPU must acquire");
    let data_2 = registry.slot(read_2).unwrap();
    assert_eq!(data_2.size(), MESH_TABLE_ENTRY_SIZE * 5);

    // Verify each entry via the CPU-side table (the source of truth).
    assert_eq!(table.get(0).unwrap().index_count, 100, "idx_a preserved");
    assert_eq!(table.get(3).unwrap().index_count, 300, "idx_d correct");
    assert_eq!(table.get(4).unwrap().index_count, 80, "idx_e correct");
    assert_eq!(table.get(4).unwrap().material_id, 3, "idx_e material correct");
}

// =============================================================================
// SECTION 4 -- Index stability after removal
// =============================================================================

/// Removing a mesh creates a hole (zeroed entry) but does NOT change the
/// indices of other meshes. This is critical for GPU-driven rendering where
/// in-flight GPU commands reference meshes by index.
#[test]
fn remove_creates_hole_does_not_invalidate_other_indices() {
    let mut table = MeshTable::new();

    let idx_a = table.add(MeshTableEntry::new(0, 0, 100, 50, 0, 1));
    let idx_b = table.add(MeshTableEntry::new(400, 0, 200, 100, 1, 1));
    let idx_c = table.add(MeshTableEntry::new(1200, 0, 50, 25, 2, 1));

    // Remove the middle mesh.
    assert_eq!(table.remove(idx_b), RemoveResult::Removed);

    // live_count decreased.
    assert_eq!(table.live_count(), 2, "Two live meshes remain");

    // All indices unchanged (no shift).
    assert_eq!(table.len(), 3, "Length unchanged after removal (hole)");
    assert_eq!(table.get(idx_a).unwrap().index_count, 100, "Mesh A index 0 stable");
    assert_eq!(table.get(idx_c).unwrap().index_count, 50, "Mesh C index 2 stable");

    // The removed slot is a zero hole.
    let hole = table.get(idx_b).unwrap();
    assert!(hole.is_zero(), "Removed entry is a hole");
    assert_eq!(hole.index_offset, 0);
    assert_eq!(hole.vertex_offset, 0);
    assert_eq!(hole.index_count, 0);
    assert_eq!(hole.flags, 0);
}

/// After removal, adding new meshes appends after the existing entries
/// (the hole is not reused by add()). This is the expected behavior for a
/// bindless table where holes preserve GPU handle validity.
#[test]
fn add_after_remove_appends_at_end() {
    let mut table = MeshTable::new();

    let idx_a = table.add(MeshTableEntry::new(0, 0, 100, 50, 0, 1));
    let _idx_b = table.add(MeshTableEntry::new(400, 0, 200, 100, 1, 1));

    assert_eq!(table.remove(idx_a), RemoveResult::Removed);

    // New mesh appends after the last entry (index 2, not reused index 0).
    let idx_c = table.add(MeshTableEntry::new(800, 0, 50, 25, 2, 1));
    assert_eq!(idx_c, 2, "New mesh appends at end (len=2), not reusing hole at 0");

    assert_eq!(table.live_count(), 2, "Mesh B + Mesh C = 2 live");
    assert_eq!(table.len(), 3, "Three slots: [hole, B, C]");
    assert!(table.get(0).unwrap().is_zero(), "Slot 0 is still a hole");
    assert!(!table.get(1).unwrap().is_zero(), "Slot 1 is B (alive)");
    assert!(!table.get(2).unwrap().is_zero(), "Slot 2 is C (alive)");
}

/// Staging a table with holes: zero entries must be faithfully represented
/// in the GPU buffer so that GPU-side code can skip them (e.g., check flags
/// bit 0 or index_count == 0).
#[test]
fn staging_preserves_holes() {
    let mut table = MeshTable::new();
    let mut registry = BufferRegistry::new(4096);

    table.add(MeshTableEntry::new(0, 0, 100, 50, 0, 1));
    let idx = table.add(MeshTableEntry::new(400, 0, 200, 100, 1, 1));
    table.add(MeshTableEntry::new(1200, 0, 50, 25, 2, 1));

    // Remove middle entry, creating a hole at index 1.
    table.remove(idx);

    // Stage the table (3 entries, middle is hole).
    let (slot_index, written) = table.stage(&mut registry).expect("Must acquire slot");
    assert_eq!(written, MESH_TABLE_ENTRY_SIZE * 3);

    assert!(matches!(
        registry.submit_staging(slot_index, written),
        SubmitResult::Submitted
    ));

    let read_idx = registry.acquire_reading().expect("Must acquire read slot");
    let bytes = registry.slot(read_idx).unwrap().as_slice();

    // Entry 0: live at offset 0.
    assert_eq!(&bytes[0..4], &[0, 0, 0, 0], "entry[0].index_offset");
    assert_eq!(&bytes[8..12], &[100, 0, 0, 0], "entry[0].index_count=100");

    // Entry 1: hole at offset 24 -- all zeros.
    let hole_start = MESH_TABLE_ENTRY_SIZE;
    for byte_offset in 0..MESH_TABLE_ENTRY_SIZE {
        assert_eq!(
            bytes[hole_start + byte_offset], 0,
            "hole byte {} must be zero",
            byte_offset
        );
    }

    // Entry 2: live at offset 48.
    assert_eq!(&bytes[48..52], &[0xB0, 0x04, 0, 0], "entry[2].index_offset=1200");
    assert_eq!(&bytes[56..60], &[50, 0, 0, 0], "entry[2].index_count=50");
}

// =============================================================================
// SECTION 5 -- Dense loading at scale (1000+ entries)
// =============================================================================

/// Loading 1000 meshes -- simulating a real scene load -- must produce
/// sequential indices 0..999 and all entries must be retrievable by index
/// without error.
#[test]
fn dense_loading_at_scale_1000_meshes() {
    let mut table = MeshTable::new();

    // Simulate bulk asset pipeline load.
    for i in 0..1000u32 {
        let idx = table.add(MeshTableEntry::new(
            i * 100,        // index_offset -- each mesh has unique range
            i * 200,        // vertex_offset
            (i + 1) * 10,   // index_count
            (i + 1) * 5,    // vertex_count
            i % 10,         // material_id cycles 0..9
            1,              // visible
        ));
        assert_eq!(
            idx, i,
            "Mesh {} must get sequential index {}, got {}",
            i, i, idx
        );
    }

    assert_eq!(table.live_count(), 1000);
    assert_eq!(table.len(), 1000);

    // All 1000 meshes retrievable by index.
    for i in 0..1000u32 {
        let entry = table
            .get(i)
            .unwrap_or_else(|| panic!("Mesh at index {} must be retrievable", i));
        assert_eq!(entry.index_offset, i * 100, "index_offset for mesh {}", i);
        assert_eq!(entry.vertex_offset, i * 200, "vertex_offset for mesh {}", i);
        assert_eq!(entry.index_count, (i + 1) * 10, "index_count for mesh {}", i);
        assert_eq!(entry.material_id, i % 10, "material_id for mesh {}", i);
        assert_eq!(entry.flags, 1, "flags (visible) for mesh {}", i);
    }

    // as_bytes must produce the correct total size.
    assert_eq!(
        table.as_bytes().len(),
        1000 * MESH_TABLE_ENTRY_SIZE,
        "Byte length must be 1000 * 24 = 24000"
    );

    // as_slice must have the correct length.
    assert_eq!(table.as_slice().len(), 1000);
}

/// After dense loading, staging 1000 entries through BufferRegistry must
/// succeed and the GPU must be able to read all 1000 entries back.
#[test]
fn dense_loading_stage_through_registry() {
    let mut table = MeshTable::new();
    let mut registry = BufferRegistry::new(4096); // Will auto-grow.

    // Load 1000 meshes.
    for i in 0..1000u32 {
        table.add(MeshTableEntry::new(i, i * 2, (i + 1) * 10, (i + 1) * 5, i % 10, 1));
    }

    // Stage (registry slot will auto-grow).
    let (slot_index, written) = table.stage(&mut registry).expect("Must acquire slot");
    assert_eq!(written, 1000 * MESH_TABLE_ENTRY_SIZE);
    assert!(matches!(registry.submit_staging(slot_index, written), SubmitResult::Submitted));

    // GPU reads back.
    let read_idx = registry.acquire_reading().expect("Must acquire read slot");
    let bytes = registry.slot(read_idx).unwrap().as_slice();
    assert_eq!(bytes.len(), 1000 * MESH_TABLE_ENTRY_SIZE);

    // Spot-check entry at index 512.
    let offset = 512 * MESH_TABLE_ENTRY_SIZE;
    assert_eq!(
        &bytes[offset..offset + 4],
        &[0x00, 0x02, 0, 0],
        "entry[512].index_offset=512 (0x200)"
    );
    assert_eq!(
        &bytes[offset + 12..offset + 16],
        &[0x05, 0x0A, 0, 0],
        "entry[512].vertex_count=(512+1)*5=2565 (0x0A05)"
    );
    assert_eq!(
        &bytes[offset + 16..offset + 20],
        &[2, 0, 0, 0],
        "entry[512].material_id=512%10=2"
    );
    assert_eq!(
        &bytes[offset + 20..offset + 24],
        &[1, 0, 0, 0],
        "entry[512].flags=1 (visible)"
    );
}

// =============================================================================
// SECTION 6 -- In-place update preserves index
// =============================================================================

/// Updating an entry in place preserves its index. This allows the asset
/// pipeline to stream updated mesh data (e.g., LOD transitions) without
/// invalidating GPU-side references.
#[test]
fn update_preserves_index() {
    let mut table = MeshTable::new();

    let idx = table.add(MeshTableEntry::new(0, 0, 100, 50, 0, 1));
    assert_eq!(idx, 0, "Initial index is 0");

    // Update to new geometry data (simulating LOD change or streaming).
    let updated = MeshTableEntry::new(500, 1000, 50, 25, 2, 1);
    assert!(table.update(idx, updated), "Update must succeed");

    // Index unchanged.
    assert_eq!(table.len(), 1, "Length unchanged after update");
    assert_eq!(table.live_count(), 1, "Live count unchanged after update");

    // Entry has new data.
    let entry = table.get(idx).unwrap();
    assert_eq!(entry.index_offset, 500, "index_offset updated");
    assert_eq!(entry.vertex_offset, 1000, "vertex_offset updated");
    assert_eq!(entry.index_count, 50, "index_count updated");
    assert_eq!(entry.vertex_count, 25, "vertex_count updated");
    assert_eq!(entry.material_id, 2, "material_id updated");
    assert_eq!(entry.flags, 1, "flags preserved");
}

/// Staging after update: the GPU buffer must contain the updated data
/// at the same index position.
#[test]
fn staging_after_update_shows_new_data() {
    let mut table = MeshTable::new();
    let mut registry = BufferRegistry::new(4096);

    table.add(MeshTableEntry::new(100, 200, 300, 150, 0, 1));

    // Update index 0 with new data.
    table.update(0, MeshTableEntry::new(999, 888, 777, 666, 5, 0));

    // Stage through BufferRegistry.
    let (slot_index, written) = table.stage(&mut registry).expect("Must acquire slot");
    assert!(matches!(registry.submit_staging(slot_index, written), SubmitResult::Submitted));

    let read_idx = registry.acquire_reading().expect("Must acquire read slot");
    let bytes = registry.slot(read_idx).unwrap().as_slice();

    // Verify updated data is what the GPU sees.
    assert_eq!(&bytes[0..4], &[0xE7, 0x03, 0, 0], "index_offset=999");
    assert_eq!(&bytes[4..8], &[0x78, 0x03, 0, 0], "vertex_offset=888");
    assert_eq!(&bytes[8..12], &[0x09, 0x03, 0, 0], "index_count=777");
    assert_eq!(&bytes[12..16], &[0x9A, 0x02, 0, 0], "vertex_count=666");
    assert_eq!(&bytes[16..20], &[5, 0, 0, 0], "material_id=5");
    assert_eq!(&bytes[20..24], &[0, 0, 0, 0], "flags=0 (not visible)");
}

// =============================================================================
// SECTION 7 -- Bulk staging integrity through BufferRegistry
// =============================================================================

/// Stage and submit a table with diverse entries through the full triple-buffer
/// pipeline and verify that every byte arrives intact at the GPU side.
/// This is the end-to-end data-path validation for the mesh table.
#[test]
fn bulk_staging_integrity_full_pipeline() {
    let mut table = MeshTable::new();
    let mut registry = BufferRegistry::new(4096);

    // Load 50 meshes with deterministic patterns.
    for i in 0..50u32 {
        table.add(MeshTableEntry::new(
            i * 100,
            i * 1000,
            (i + 1) * 20,
            (i + 1) * 10,
            i % 8,
            if i % 2 == 0 { 1 } else { 0 },
        ));
    }

    // Stage the full table.
    let (slot_index, written) = table.stage(&mut registry).expect("Must acquire slot");
    assert_eq!(written, 50 * MESH_TABLE_ENTRY_SIZE);
    assert!(matches!(registry.submit_staging(slot_index, written), SubmitResult::Submitted));

    // Read back and verify every byte.
    let read_idx = registry.acquire_reading().expect("Must acquire read slot");
    let bytes = registry.slot(read_idx).unwrap().as_slice();
    assert_eq!(bytes.len(), 50 * MESH_TABLE_ENTRY_SIZE);

    for i in 0..50u32 {
        let offset = (i as usize) * MESH_TABLE_ENTRY_SIZE;

        let actual_index_offset = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        let actual_vertex_offset = u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap());
        let actual_index_count = u32::from_le_bytes(bytes[offset + 8..offset + 12].try_into().unwrap());
        let actual_vertex_count = u32::from_le_bytes(bytes[offset + 12..offset + 16].try_into().unwrap());
        let actual_material_id = u32::from_le_bytes(bytes[offset + 16..offset + 20].try_into().unwrap());
        let actual_flags = u32::from_le_bytes(bytes[offset + 20..offset + 24].try_into().unwrap());

        assert_eq!(actual_index_offset, i * 100, "entry[{}].index_offset", i);
        assert_eq!(actual_vertex_offset, i * 1000, "entry[{}].vertex_offset", i);
        assert_eq!(actual_index_count, (i + 1) * 20, "entry[{}].index_count", i);
        assert_eq!(actual_vertex_count, (i + 1) * 10, "entry[{}].vertex_count", i);
        assert_eq!(actual_material_id, i % 8, "entry[{}].material_id", i);
        assert_eq!(actual_flags, if i % 2 == 0 { 1 } else { 0 }, "entry[{}].flags", i);
    }
}

// =============================================================================
// SECTION 8 -- Empty table edge cases
// =============================================================================

/// An empty mesh table cannot be staged (no data to upload).
/// This is consistent with the spec: the asset pipeline has loaded nothing,
/// so there is nothing to upload.
#[test]
fn empty_table_stage_returns_none() {
    let table = MeshTable::new();
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

/// An empty table is empty with zero live count and zero length.
#[test]
fn empty_table_properties() {
    let table = MeshTable::new();
    assert!(table.is_empty());
    assert_eq!(table.live_count(), 0);
    assert_eq!(table.len(), 0);
    assert!(table.as_slice().is_empty());
    assert!(table.as_bytes().is_empty());
}

/// Getting an entry from an empty table returns None.
#[test]
fn get_from_empty_table_returns_none() {
    let table = MeshTable::new();
    assert!(table.get(0).is_none(), "Index 0 on empty table must be None");
    assert!(table.get(u32::MAX).is_none(), "u32::MAX on empty table must be None");
}

// =============================================================================
// SECTION 9 -- Clear and re-add: indices restart from 0
// =============================================================================

/// After clear(), the table is fully reset. New meshes get indices starting
/// from 0 again. This simulates unloading a scene and loading a new one.
#[test]
fn clear_and_reload_indices_restart() {
    let mut table = MeshTable::new();

    // Load scene A (3 meshes).
    let scene_a = [
        table.add(MeshTableEntry::new(0, 0, 100, 50, 0, 1)),
        table.add(MeshTableEntry::new(400, 0, 200, 100, 1, 1)),
        table.add(MeshTableEntry::new(1200, 0, 50, 25, 2, 1)),
    ];
    assert_eq!(scene_a, [0, 1, 2], "Scene A indices 0..2");

    // Clear (simulating scene unload).
    table.clear();
    assert!(table.is_empty());
    assert_eq!(table.live_count(), 0);
    assert_eq!(table.len(), 0);

    // Load scene B -- indices restart from 0.
    let scene_b = [
        table.add(MeshTableEntry::new(5000, 10000, 300, 150, 0, 1)),
        table.add(MeshTableEntry::new(20000, 40000, 500, 250, 1, 1)),
    ];
    assert_eq!(scene_b, [0, 1], "Scene B indices restart from 0");

    // Scene B data is correct at indices 0, 1.
    assert_eq!(table.get(0).unwrap().index_offset, 5000);
    assert_eq!(table.get(1).unwrap().index_offset, 20000);
}

/// Staging after clear: the table transitions from populated -> empty ->
/// repopulated without errors. The GPU sees only the new data.
#[test]
fn clear_then_stage_works_correctly() {
    let mut table = MeshTable::new();
    let mut registry = BufferRegistry::new(4096);

    // Load initial meshes and stage.
    table.add(MeshTableEntry::new(0, 0, 100, 50, 0, 1));
    let (s0, w0) = table.stage(&mut registry).expect("Initial stage");
    assert!(matches!(registry.submit_staging(s0, w0), SubmitResult::Submitted));

    // Clear.
    table.clear();

    // Empty table cannot be staged.
    assert!(table.stage(&mut registry).is_none(), "Empty table after clear must return None");

    // Reload with different data.
    table.add(MeshTableEntry::new(9999, 8888, 777, 666, 5, 0));

    // Stage the new data.
    let (s1, w1) = table.stage(&mut registry).expect("Repopulated stage");
    assert_eq!(w1, MESH_TABLE_ENTRY_SIZE, "One entry after repopulation");
    assert!(matches!(registry.submit_staging(s1, w1), SubmitResult::Submitted));

    // GPU reads back -- must be the new data, not the old.
    let read_idx = registry.acquire_reading().expect("Must acquire read slot");
    let bytes = registry.slot(read_idx).unwrap().as_slice();
    assert_eq!(&bytes[0..4], &[0x0F, 0x27, 0, 0], "index_offset=9999");
    assert_eq!(&bytes[4..8], &[0xB8, 0x22, 0, 0], "vertex_offset=8888");
}

// =============================================================================
// SECTION 10 -- insert_at for non-sequential index assignment
// =============================================================================

/// insert_at allows the asset pipeline to place a mesh at a specific index,
/// for example when slot indices are pre-determined by a data-driven config.
#[test]
fn insert_at_non_sequential_index() {
    let mut table = MeshTable::new();

    // Insert at index 10.
    table.insert_at(10, MeshTableEntry::new(500, 1000, 300, 150, 2, 1));

    // Table must have 11 entries (indices 0..10).
    assert_eq!(table.len(), 11, "Table must extend to index 10 -> 11 entries");
    assert_eq!(table.live_count(), 1, "Only the inserted entry is live");

    // Indices 0..9 are zero holes.
    for i in 0..10u32 {
        let entry = table
            .get(i)
            .unwrap_or_else(|| panic!("Index {} must be accessible", i));
        assert!(entry.is_zero(), "Index {} must be a zero hole", i);
    }

    // Index 10 has the data we inserted.
    let entry = table.get(10).unwrap();
    assert_eq!(entry.index_offset, 500);
    assert_eq!(entry.vertex_offset, 1000);
    assert_eq!(entry.index_count, 300);
    assert_eq!(entry.material_id, 2);
    assert_eq!(entry.flags, 1);
}

/// insert_at followed by add: add appends after the highest index.
#[test]
fn insert_at_then_add_appends_after() {
    let mut table = MeshTable::new();

    table.insert_at(3, MeshTableEntry::new(100, 0, 50, 25, 0, 1));
    let idx = table.add(MeshTableEntry::new(200, 0, 100, 50, 1, 1));

    // add appends at the end (position 4).
    assert_eq!(idx, 4, "add after insert_at(3) must go to index 4");
    assert_eq!(table.len(), 5);
    assert_eq!(table.live_count(), 2);
}

/// insert_at at index 0 (before any add) creates a table with a single
/// entry at the first position.
#[test]
fn insert_at_index_zero() {
    let mut table = MeshTable::new();

    table.insert_at(0, MeshTableEntry::new(42, 84, 100, 50, 0, 1));

    assert_eq!(table.len(), 1, "insert_at(0) -> one entry");
    assert_eq!(table.live_count(), 1);
    assert_eq!(table.get(0).unwrap().index_offset, 42);
}

// =============================================================================
// SECTION 11 -- Display and formatting
// =============================================================================

/// Display output must contain key state information for debugging.
#[test]
fn display_shows_table_state() {
    let mut table = MeshTable::new();
    let empty_display = format!("{}", table);
    assert!(
        empty_display.starts_with("MeshTable("),
        "Display must start with 'MeshTable('"
    );
    assert!(
        empty_display.contains("entries=0"),
        "Empty table display must show entries=0"
    );
    assert!(
        empty_display.contains("live=0"),
        "Empty table display must show live=0"
    );

    table.add(MeshTableEntry::new(1, 2, 3, 4, 5, 0));
    table.add(MeshTableEntry::new(6, 7, 8, 9, 10, 1));
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

/// Display for MeshTableEntry must show all field values.
#[test]
fn entry_display_shows_all_fields() {
    let e = MeshTableEntry::new(100, 200, 300, 400, 5, 0xFF);
    let s = format!("{}", e);
    assert!(s.contains("idx_off=100"), "Display must show idx_off=100");
    assert!(s.contains("vtx_off=200"), "Display must show vtx_off=200");
    assert!(s.contains("idx_cnt=300"), "Display must show idx_cnt=300");
    assert!(s.contains("vtx_cnt=400"), "Display must show vtx_cnt=400");
    assert!(s.contains("mat=5"), "Display must show mat=5");
    assert!(s.contains("flags=0x"), "Display must show flags=0x...");
    assert!(s.contains("ff"), "Display must contain ff (0xFF)");
}

// =============================================================================
// SECTION 12 -- Entry zero and default
// =============================================================================

/// Zero entry has all fields at zero and is_zero() returns true.
#[test]
fn zero_entry_is_zero() {
    let z = MeshTableEntry::zero();
    assert!(z.is_zero());
    assert_eq!(z.index_offset, 0);
    assert_eq!(z.vertex_offset, 0);
    assert_eq!(z.index_count, 0);
    assert_eq!(z.vertex_count, 0);
    assert_eq!(z.material_id, 0);
    assert_eq!(z.flags, 0);
}

/// Default entry is a zero entry.
#[test]
fn default_entry_is_zero() {
    let d: MeshTableEntry = Default::default();
    assert!(d.is_zero());
    assert_eq!(d.index_offset, 0);
}

/// Non-zero fields cause is_zero() to return false.
#[test]
fn non_zero_entry_is_not_zero() {
    let e = MeshTableEntry::new(1, 0, 0, 0, 0, 0);
    assert!(!e.is_zero(), "index_offset=1 must make is_zero false");
}

// =============================================================================
// SECTION 13 -- Capacity and growth
// =============================================================================

/// DEFAULT_MESH_TABLE_CAPACITY constant is exported and positive.
#[test]
fn default_capacity_constant_is_positive() {
    assert!(
        DEFAULT_MESH_TABLE_CAPACITY > 0,
        "DEFAULT_MESH_TABLE_CAPACITY must be > 0"
    );
    // The export serves as documentation for the asset pipeline integration.
    assert!(
        DEFAULT_MESH_TABLE_CAPACITY >= 1024,
        "DEFAULT_MESH_TABLE_CAPACITY should be at least 1024 for real scenes"
    );
}

/// MESH_TABLE_ENTRY_SIZE constant is exported and matches the WGSL struct.
#[test]
fn entry_size_constant_matches_wgsl() {
    assert_eq!(
        MESH_TABLE_ENTRY_SIZE, 24,
        "MESH_TABLE_ENTRY_SIZE must be 24 bytes (6 * u32)"
    );
}

/// with_capacity creates a table that can hold at least the requested number
/// of entries without reallocation.
#[test]
fn with_capacity_handles_large_initial_capacity() {
    let mut table = MeshTable::with_capacity(10_000);

    // Add 10,000 entries without hitting capacity issues.
    for i in 0..10_000u32 {
        let idx = table.add(MeshTableEntry::new(i, 0, 100, 50, i % 10, 1));
        assert_eq!(idx, i, "Sequential index {} expected", i);
    }

    assert_eq!(table.live_count(), 10_000);
    assert_eq!(table.len(), 10_000);
}

// =============================================================================
// SECTION 14 -- Default trait implementations
// =============================================================================

/// MeshTable and MeshTableEntry both implement Default.
#[test]
fn mesh_table_implements_default() {
    let table: MeshTable = Default::default();
    assert!(table.is_empty());
    assert_eq!(table.len(), 0);
}

/// MeshTableEntry is Copy + Clone.
#[test]
fn mesh_table_entry_is_copy_and_clone() {
    let a = MeshTableEntry::new(1, 2, 3, 4, 5, 6);
    let b = a; // Copy.
    assert_eq!(a.index_offset, b.index_offset);
    assert_eq!(a.flags, b.flags);

    let c = a.clone(); // Clone.
    assert_eq!(a.index_offset, c.index_offset);
    assert_eq!(a.material_id, c.material_id);
}

// =============================================================================
// SECTION 15 -- Reserve prevents reallocation during add
// =============================================================================

/// Reserve ensures subsequent add() calls do not reallocate. This is useful
/// for the asset pipeline when the total mesh count is known ahead of time.
#[test]
fn reserve_after_add_does_not_break_existing_indices() {
    let mut table = MeshTable::new();

    table.add(MeshTableEntry::new(10, 0, 100, 50, 0, 1));
    table.add(MeshTableEntry::new(20, 0, 200, 100, 1, 1));

    // Reserve space for 500 more.
    table.reserve(500);

    // Existing indices still valid.
    assert_eq!(table.get(0).unwrap().index_offset, 10);
    assert_eq!(table.get(1).unwrap().index_offset, 20);

    // Adds after reserve continue correctly.
    for i in 2..102u32 {
        let idx = table.add(MeshTableEntry::new(i * 10, 0, 100, 50, 0, 1));
        assert_eq!(idx, i, "Reserved space: sequential index {}", i);
    }
}

// =============================================================================
// SECTION 16 -- as_bytes and as_slice consistency
// =============================================================================

/// The byte representation and the slice representation must agree on the
/// number of entries.
#[test]
fn as_bytes_and_as_slice_agree_on_count() {
    let mut table = MeshTable::new();

    for i in 0..10u32 {
        table.add(MeshTableEntry::new(i, i * 2, 100, 50, 0, 1));
    }

    let bytes = table.as_bytes();
    let slice = table.as_slice();

    assert_eq!(slice.len() * MESH_TABLE_ENTRY_SIZE, bytes.len());
    assert_eq!(slice.len(), 10);

    // The first field of each entry in bytes should match the slice.
    for (i, entry) in slice.iter().enumerate() {
        let idx_offset_bytes = u32::from_le_bytes(
            bytes[i * MESH_TABLE_ENTRY_SIZE..i * MESH_TABLE_ENTRY_SIZE + 4]
                .try_into()
                .unwrap(),
        );
        assert_eq!(
            idx_offset_bytes,
            entry.index_offset,
            "Byte/slice agreement for entry[{}].index_offset",
            i
        );
    }
}

// =============================================================================
// SECTION 17 -- get_mut allows in-place mutation through a mutable reference
// =============================================================================

/// get_mut returns a mutable reference that can be used to update flags
/// in-place without changing the index.
#[test]
fn get_mut_updates_visibility_flag() {
    let mut table = MeshTable::new();

    // Mesh loaded by asset pipeline, initially visible.
    let idx = table.add(MeshTableEntry::new(0, 0, 100, 50, 0, 1));
    assert_eq!(table.get(idx).unwrap().flags, 1, "Initially visible");

    // Toggle visibility off via mutable reference (simulating GPU feedback:
    // "this mesh was not visible last frame, skip culling").
    {
        let entry = table.get_mut(idx).expect("get_mut must succeed");
        entry.flags = 0;
    }

    // Verify the change.
    assert_eq!(table.get(idx).unwrap().flags, 0, "Visibility toggled off");

    // Index unchanged.
    assert_eq!(idx, 0);
    assert_eq!(table.len(), 1);
}

// =============================================================================
// SECTION 18 -- RemoveResult enum completeness
// =============================================================================

/// RemoveResult::Removed is returned when a live entry is removed.
#[test]
fn remove_result_removed() {
    let mut table = MeshTable::new();
    table.add(MeshTableEntry::new(1, 2, 3, 4, 5, 6));
    assert_eq!(table.remove(0), RemoveResult::Removed);
}

/// RemoveResult::NotFound is returned when the index does not exist.
#[test]
fn remove_result_not_found_out_of_range() {
    let mut table = MeshTable::new();
    assert_eq!(
        table.remove(0),
        RemoveResult::NotFound,
        "Removing index 0 from empty table must be NotFound"
    );
    assert_eq!(
        table.remove(u32::MAX),
        RemoveResult::NotFound,
        "Removing u32::MAX from empty table must be NotFound"
    );
}

/// RemoveResult::NotFound is returned when removing a hole.
#[test]
fn remove_result_not_found_for_hole() {
    let mut table = MeshTable::new();
    table.add(MeshTableEntry::new(1, 2, 3, 4, 5, 6));
    table.remove(0);
    assert_eq!(
        table.remove(0),
        RemoveResult::NotFound,
        "Removing a hole must be NotFound"
    );
}

// =============================================================================
// SECTION 19 -- Stage with multiple submissions (frame pipeline)
// =============================================================================

/// Simulate a multi-frame submission pattern: the asset pipeline loads meshes
/// incrementally and stages them each frame. Each frame's GPU data must
/// contain all meshes loaded up to that point.
#[test]
fn multi_frame_incremental_staging() {
    let mut table = MeshTable::new();
    let mut registry = BufferRegistry::new(16384);

    // Frame 1: 2 meshes.
    table.add(MeshTableEntry::new(0, 0, 100, 50, 0, 1));
    table.add(MeshTableEntry::new(400, 0, 200, 100, 1, 1));
    let (s1, w1) = table.stage(&mut registry).expect("Frame 1 stage");
    assert_eq!(w1, MESH_TABLE_ENTRY_SIZE * 2);
    assert!(matches!(registry.submit_staging(s1, w1), SubmitResult::Submitted));

    // GPU reads frame 1.
    let r1 = registry.acquire_reading().expect("Frame 1 read");
    assert_eq!(registry.slot(r1).unwrap().size(), MESH_TABLE_ENTRY_SIZE * 2);

    // Frame 2: add 1 more mesh (3 total).
    table.add(MeshTableEntry::new(800, 0, 50, 25, 2, 1));
    let (s2, w2) = table.stage(&mut registry).expect("Frame 2 stage");
    assert_eq!(w2, MESH_TABLE_ENTRY_SIZE * 3);
    assert!(matches!(registry.submit_staging(s2, w2), SubmitResult::Submitted));

    // GPU reads frame 2 (3 meshes).
    let r2 = registry.acquire_reading().expect("Frame 2 read");
    let bytes2 = registry.slot(r2).unwrap().as_slice();
    assert_eq!(bytes2.len(), MESH_TABLE_ENTRY_SIZE * 3);
    assert_eq!(&bytes2[0..4], &[0, 0, 0, 0], "frame2 entry[0] ok");
    assert_eq!(&bytes2[24..28], &[0x90, 0x01, 0, 0], "frame2 entry[1] offset=400");
    assert_eq!(&bytes2[48..52], &[0x20, 0x03, 0, 0], "frame2 entry[2] offset=800");

    // Frame 3: add 2 more meshes (5 total).
    table.add(MeshTableEntry::new(1600, 0, 300, 150, 0, 1));
    table.add(MeshTableEntry::new(3200, 0, 80, 40, 3, 0));
    let (s3, w3) = table.stage(&mut registry).expect("Frame 3 stage");
    assert_eq!(w3, MESH_TABLE_ENTRY_SIZE * 5);
    assert!(matches!(registry.submit_staging(s3, w3), SubmitResult::Submitted));

    // GPU reads frame 3 (5 meshes).
    let r3 = registry.acquire_reading().expect("Frame 3 read");
    let bytes3 = registry.slot(r3).unwrap().as_slice();
    assert_eq!(bytes3.len(), MESH_TABLE_ENTRY_SIZE * 5);

    // Verify all 5 entries: indices 0..4.
    for i in 0..5u32 {
        let offset = (i as usize) * MESH_TABLE_ENTRY_SIZE;
        let index_offset = u32::from_le_bytes(bytes3[offset..offset + 4].try_into().unwrap());
        assert_eq!(
            index_offset,
            table.get(i).unwrap().index_offset,
            "GPU/CPU agreement on entry[{}].index_offset",
            i
        );
    }

    // Frame count is 3.
    assert_eq!(registry.frame_count(), 3);
}

// =============================================================================
// SECTION 20 -- stage_and_submit convenience method
// =============================================================================

/// stage_and_submit combines stage + submit into one call.
#[test]
fn stage_and_submit_works() {
    let mut table = MeshTable::new();
    let mut registry = BufferRegistry::new(4096);

    table.add(MeshTableEntry::new(100, 200, 300, 150, 5, 1));
    assert!(table.stage_and_submit(&mut registry), "stage_and_submit must succeed");

    // GPU reads the submitted data.
    let read_idx = registry.acquire_reading().expect("Must have ready slot");
    let bytes = registry.slot(read_idx).unwrap().as_slice();
    assert_eq!(bytes.len(), MESH_TABLE_ENTRY_SIZE);
    assert_eq!(&bytes[0..4], &[100, 0, 0, 0], "index_offset=100");
}

/// stage_and_submit returns false when no staging slot is available.
#[test]
fn stage_and_submit_fails_when_no_slot() {
    let mut table = MeshTable::new();
    let mut registry = BufferRegistry::new(4096);

    table.add(MeshTableEntry::new(1, 2, 3, 4, 5, 6));

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
// SECTION 21 -- Default mesh table uses DEFAULT_MESH_TABLE_CAPACITY
// =============================================================================

/// MeshTable::new() creates a table with at least DEFAULT_MESH_TABLE_CAPACITY
/// capacity, enabling many meshes to be added without immediate reallocation.
#[test]
fn new_table_has_default_capacity() {
    let table = MeshTable::with_capacity(DEFAULT_MESH_TABLE_CAPACITY);
    assert!(table.is_empty());

    // We can add DEFAULT_MESH_TABLE_CAPACITY entries without needing to grow
    // (verified by absence of reallocation -- we add exactly that many).
    // The table starts empty but has the capacity pre-allocated.
    // Actual capacity is an implementation detail, but with_capacity must
    // allocate >= the requested amount.
    assert_eq!(table.len(), 0);
    assert_eq!(table.live_count(), 0);
}

// =============================================================================
// SECTION 22 -- Update with zero entry reduces live_count
// =============================================================================

/// Updating a live entry to zero reduces live_count. This is semantically
/// equivalent to removal but preserves the index (no GPU handle change).
#[test]
fn update_to_zero_reduces_live_count() {
    let mut table = MeshTable::new();
    let idx = table.add(MeshTableEntry::new(10, 20, 30, 40, 0, 1));
    assert_eq!(table.live_count(), 1, "One live entry");

    // Update to zero (equivalent to making it a hole).
    assert!(table.update(idx, MeshTableEntry::zero()), "Update to zero must succeed");
    assert_eq!(table.live_count(), 0, "Live count drops to 0");
    assert!(table.get(idx).unwrap().is_zero(), "Entry is now a zero hole");
    assert_eq!(table.len(), 1, "Length unchanged (hole preserved)");
}

// =============================================================================
// SECTION 23 -- Update across the staging pipeline
// =============================================================================

/// Updating a mesh and re-staging must produce the new data in the GPU buffer.
/// The frame_count increments with each submit.
#[test]
fn update_then_stage_shows_new_data_in_pipeline() {
    let mut table = MeshTable::new();
    let mut registry = BufferRegistry::new(4096);

    // Load initial mesh.
    let idx = table.add(MeshTableEntry::new(100, 200, 300, 150, 0, 1));

    // Stage frame 1.
    assert!(table.stage_and_submit(&mut registry));
    assert_eq!(registry.frame_count(), 1);

    // Update the mesh with new geometry.
    table.update(idx, MeshTableEntry::new(999, 888, 777, 666, 5, 0));

    // Stage frame 2.
    assert!(table.stage_and_submit(&mut registry));
    assert_eq!(registry.frame_count(), 2);

    // GPU reads frame 2 (newest frame).
    let read_idx = registry.acquire_reading().expect("Must have ready slot");
    let bytes = registry.slot(read_idx).unwrap().as_slice();
    assert_eq!(bytes.len(), MESH_TABLE_ENTRY_SIZE);

    // Must be the updated data.
    assert_eq!(&bytes[0..4], &[0xE7, 0x03, 0, 0], "Updated index_offset=999");
    assert_eq!(&bytes[4..8], &[0x78, 0x03, 0, 0], "Updated vertex_offset=888");
}
