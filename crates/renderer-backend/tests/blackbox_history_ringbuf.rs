// Blackbox contract tests for HistorySlotManager and ResourceLifetime::History (T-FG-3.5).
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::frame_graph::*` -- no internal fields,
// no private methods, no implementation details.
//
// DEV added: HistorySlotManager struct and ResourceLifetime::History variant
// for N-slot ring buffer history resources.
//
// Contract (T-FG-3.5):
//   HistorySlotManager manages slot assignment for history resources.
//   - HistorySlotManager::new() creates an empty manager with no tracked resources
//   - HistorySlotManager::from_resources(resources) detects resources with
//     ResourceLifetime::History(n) and tracks their slot counts
//   - slot_for(handle, frame) returns frame % history_length for history resources
//   - Non-history resources have slot_for always 0
//   - history_length returns the history length for history resources
//   - is_history correctly identifies history resources
//   - Empty manager returns None for all queries
//
// Coverage:
//   1.  HistorySlotManager can be constructed (no panic)
//   2.  HistorySlotManager::new() creates empty manager
//   3.  from_resources() detects a resource with ResourceLifetime::History(3)
//   4.  slot_for() returns correct slot: frame 0->0, 1->1, 2->2, 3->0
//   5.  Non-history resources have slot_for always 0
//   6.  history_length returns correct values
//   7.  is_history correctly distinguishes history from non-history
//   8.  Multiple history resources with different lengths
//   9.  Empty manager returns None for all queries

use renderer_backend::frame_graph::{
    BufferDesc, HistorySlotManager, IrResource, ResourceDesc, ResourceHandle,
    ResourceLifetime, ResourceState,
};

// =============================================================================
// Helpers
// =============================================================================

/// Creates a buffer resource with the given handle, name, and lifetime.
fn make_resource(handle: u32, name: &str, lifetime: ResourceLifetime) -> IrResource {
    IrResource::new(
        ResourceHandle(handle),
        name,
        ResourceDesc::Buffer(BufferDesc {
            size: 4096,
            usage: "storage".into(),
            is_indirect_arg: false,
        }),
        lifetime,
        ResourceState::Uninitialized,
    )
}

/// Creates a history resource with the given history length.
fn history_resource(handle: u32, name: &str, history_len: usize) -> IrResource {
    make_resource(handle, name, ResourceLifetime::History(history_len))
}

/// Creates a transient (non-history) resource.
fn transient_resource(handle: u32, name: &str) -> IrResource {
    make_resource(handle, name, ResourceLifetime::Transient)
}

// =============================================================================
// SECTION 1 -- Construction
// =============================================================================

/// HistorySlotManager::new() can be called without panicking.
#[test]
fn construction_new_does_not_panic() {
    let manager = HistorySlotManager::new();
    let h = ResourceHandle(0);
    assert!(!manager.is_history(h), "Empty manager: is_history must be false");
}

/// HistorySlotManager::new() creates an empty manager that returns None/false
/// for all queries.
#[test]
fn new_creates_empty_manager() {
    let manager = HistorySlotManager::new();
    let h = ResourceHandle(42);

    assert!(
        !manager.is_history(h),
        "Empty manager: is_history must be false",
    );
    assert_eq!(
        manager.history_length(h),
        None,
        "Empty manager: history_length must be None",
    );
    assert_eq!(
        manager.slot_for(h, 0),
        None,
        "Empty manager: slot_for at frame 0 must be None",
    );
    assert_eq!(
        manager.slot_for(h, 5),
        None,
        "Empty manager: slot_for at any frame must be None",
    );
}

/// HistorySlotManager::from_resources with an empty slice produces an empty
/// manager.
#[test]
fn from_resources_empty_slice() {
    let resources: Vec<IrResource> = vec![];
    let manager = HistorySlotManager::from_resources(&resources);
    let h = ResourceHandle(0);

    assert!(!manager.is_history(h));
    assert_eq!(manager.history_length(h), None);
    assert_eq!(manager.slot_for(h, 0), None);
}

/// HistorySlotManager::from_resources with only non-history resources produces
/// a manager where is_history is false and slot_for returns Some(0).
#[test]
fn from_resources_only_transient() {
    let resources = vec![
        transient_resource(0, "a"),
        transient_resource(1, "b"),
    ];
    let manager = HistorySlotManager::from_resources(&resources);

    assert!(!manager.is_history(ResourceHandle(0)));
    assert!(!manager.is_history(ResourceHandle(1)));
    assert_eq!(manager.slot_for(ResourceHandle(0), 0), Some(0));
    assert_eq!(manager.slot_for(ResourceHandle(1), 0), Some(0));
}

// =============================================================================
// SECTION 2 -- from_resources detects history resources
// =============================================================================

/// from_resources detects a single resource with ResourceLifetime::History(3).
#[test]
fn from_resources_detects_history_3() {
    let resources = vec![history_resource(0, "hdr_color", 3)];
    let manager = HistorySlotManager::from_resources(&resources);
    let h = ResourceHandle(0);

    assert!(
        manager.is_history(h),
        "History(3) resource must be detected as history",
    );
    assert_eq!(
        manager.history_length(h),
        Some(3),
        "History(3) resource must report length 3",
    );
}

/// from_resources detects a resource with ResourceLifetime::History(1).
#[test]
fn from_resources_detects_history_1() {
    let resources = vec![history_resource(0, "single_slot", 1)];
    let manager = HistorySlotManager::from_resources(&resources);
    let h = ResourceHandle(0);

    assert!(manager.is_history(h));
    assert_eq!(manager.history_length(h), Some(1));
}

/// from_resources detects a resource with ResourceLifetime::History(8).
#[test]
fn from_resources_detects_history_8() {
    let resources = vec![history_resource(0, "eight_slot", 8)];
    let manager = HistorySlotManager::from_resources(&resources);
    let h = ResourceHandle(0);

    assert!(manager.is_history(h));
    assert_eq!(manager.history_length(h), Some(8));
}

/// from_resources detects a resource with ResourceLifetime::History(64).
#[test]
fn from_resources_detects_history_64() {
    let resources = vec![history_resource(0, "sixty_four_slot", 64)];
    let manager = HistorySlotManager::from_resources(&resources);
    let h = ResourceHandle(0);

    assert!(manager.is_history(h));
    assert_eq!(manager.history_length(h), Some(64));
}

// =============================================================================
// SECTION 3 -- slot_for slot mapping
// =============================================================================

/// slot_for returns correct slot for a 3-slot ring buffer:
/// frame 0 -> slot 0, frame 1 -> slot 1, frame 2 -> slot 2, frame 3 -> slot 0.
#[test]
fn slot_for_three_slot_ring() {
    let resources = vec![history_resource(0, "triple", 3)];
    let manager = HistorySlotManager::from_resources(&resources);
    let h = ResourceHandle(0);

    assert_eq!(manager.slot_for(h, 0), Some(0), "Frame 0 -> slot 0");
    assert_eq!(manager.slot_for(h, 1), Some(1), "Frame 1 -> slot 1");
    assert_eq!(manager.slot_for(h, 2), Some(2), "Frame 2 -> slot 2");
    assert_eq!(manager.slot_for(h, 3), Some(0), "Frame 3 -> slot 0 (wrap)");
}

/// slot_for maps correctly for a 2-slot history resource (double-buffering).
#[test]
fn slot_for_two_slot_ring() {
    let resources = vec![history_resource(0, "double", 2)];
    let manager = HistorySlotManager::from_resources(&resources);
    let h = ResourceHandle(0);

    for frame in 0..10 {
        let expected = frame % 2;
        assert_eq!(
            manager.slot_for(h, frame),
            Some(expected),
            "2-slot ring: frame {} -> slot {}",
            frame,
            expected,
        );
    }
}

/// slot_for maps correctly for a 5-slot history resource.
#[test]
fn slot_for_five_slot_ring() {
    let resources = vec![history_resource(0, "penta", 5)];
    let manager = HistorySlotManager::from_resources(&resources);
    let h = ResourceHandle(0);

    for frame in 0..25 {
        let expected = frame % 5;
        assert_eq!(
            manager.slot_for(h, frame),
            Some(expected),
            "5-slot ring: frame {} -> slot {}",
            frame,
            expected,
        );
    }
}

/// slot_for with large frame indices wraps correctly via modulo.
#[test]
fn slot_for_large_frame_indices_wrap() {
    let resources = vec![history_resource(0, "wrap_test", 4)];
    let manager = HistorySlotManager::from_resources(&resources);
    let h = ResourceHandle(0);

    assert_eq!(manager.slot_for(h, 1000), Some(0));
    assert_eq!(manager.slot_for(h, 1001), Some(1));
    assert_eq!(manager.slot_for(h, 1002), Some(2));
    assert_eq!(manager.slot_for(h, 1003), Some(3));
    assert_eq!(manager.slot_for(h, 1004), Some(0));

    assert_eq!(manager.slot_for(h, usize::MAX), Some(usize::MAX % 4));
}

// =============================================================================
// SECTION 4 -- Non-history resources have slot_for always 0
// =============================================================================

/// Transient resources have slot_for always returning Some(0) regardless of
/// frame index.
#[test]
fn transient_resource_slot_for_always_zero() {
    let resources = vec![transient_resource(0, "color_rt")];
    let manager = HistorySlotManager::from_resources(&resources);
    let h = ResourceHandle(0);

    assert!(!manager.is_history(h), "Transient resource is not history");
    for frame in 0..10 {
        assert_eq!(
            manager.slot_for(h, frame),
            Some(0),
            "Transient resource slot_for at frame {} must be 0",
            frame,
        );
    }
}

/// Multiple transient resources all return slot 0.
#[test]
fn multiple_transient_resources_all_slot_zero() {
    let resources = vec![
        transient_resource(0, "a"),
        transient_resource(1, "b"),
        transient_resource(2, "c"),
    ];
    let manager = HistorySlotManager::from_resources(&resources);

    for h in 0..3 {
        let handle = ResourceHandle(h);
        assert!(!manager.is_history(handle), "Transient {} is not history", h);
        assert_eq!(manager.slot_for(handle, 0), Some(0));
        assert_eq!(manager.slot_for(handle, 42), Some(0));
    }
}

/// Imported resources have slot_for always returning Some(0).
#[test]
fn imported_resource_slot_for_always_zero() {
    let resources = vec![make_resource(
        0,
        "imported_buf",
        ResourceLifetime::Imported,
    )];
    let manager = HistorySlotManager::from_resources(&resources);
    let h = ResourceHandle(0);

    assert!(!manager.is_history(h), "Imported resource is not history");
    assert_eq!(
        manager.slot_for(h, 0),
        Some(0),
        "Imported resource slot_for must be 0",
    );
    assert_eq!(
        manager.slot_for(h, 99),
        Some(0),
        "Imported resource slot_for at any frame must be 0",
    );
}

// =============================================================================
// SECTION 5 -- history_length returns correct values
// =============================================================================

/// history_length returns the correct length for various history sizes.
#[test]
fn history_length_various_values() {
    let resources = vec![
        history_resource(0, "h2", 2),
        history_resource(1, "h4", 4),
        history_resource(2, "h8", 8),
        history_resource(3, "h16", 16),
        history_resource(4, "h64", 64),
    ];
    let manager = HistorySlotManager::from_resources(&resources);

    assert_eq!(manager.history_length(ResourceHandle(0)), Some(2));
    assert_eq!(manager.history_length(ResourceHandle(1)), Some(4));
    assert_eq!(manager.history_length(ResourceHandle(2)), Some(8));
    assert_eq!(manager.history_length(ResourceHandle(3)), Some(16));
    assert_eq!(manager.history_length(ResourceHandle(4)), Some(64));
}

/// history_length returns None for a transient resource.
#[test]
fn history_length_for_transient_is_none() {
    let resources = vec![transient_resource(0, "plain")];
    let manager = HistorySlotManager::from_resources(&resources);
    assert_eq!(
        manager.history_length(ResourceHandle(0)),
        None,
        "Transient resource must have history_length = None",
    );
}

/// history_length returns None for an imported resource.
#[test]
fn history_length_for_imported_is_none() {
    let resources = vec![make_resource(0, "imported", ResourceLifetime::Imported)];
    let manager = HistorySlotManager::from_resources(&resources);
    assert_eq!(
        manager.history_length(ResourceHandle(0)),
        None,
        "Imported resource must have history_length = None",
    );
}

/// history_length returns None for a handle not in the manager.
#[test]
fn history_length_unknown_handle_none() {
    let resources = vec![history_resource(0, "h3", 3)];
    let manager = HistorySlotManager::from_resources(&resources);
    assert_eq!(
        manager.history_length(ResourceHandle(99)),
        None,
        "Unknown handle must return None",
    );
    assert_eq!(
        manager.history_length(ResourceHandle(u32::MAX)),
        None,
        "Non-existent handle must return None",
    );
}

// =============================================================================
// SECTION 6 -- is_history correctly distinguishes
// =============================================================================

/// is_history correctly distinguishes history from transient resources.
#[test]
fn is_history_distinguishes_types() {
    let resources = vec![
        history_resource(0, "history", 3),
        transient_resource(1, "transient"),
    ];
    let manager = HistorySlotManager::from_resources(&resources);

    assert!(
        manager.is_history(ResourceHandle(0)),
        "Resource with History lifetime must be identified as history",
    );
    assert!(
        !manager.is_history(ResourceHandle(1)),
        "Resource with Transient lifetime must NOT be identified as history",
    );
}

/// is_history returns false for imported resources.
#[test]
fn is_history_false_for_imported() {
    let resources = vec![make_resource(0, "imported", ResourceLifetime::Imported)];
    let manager = HistorySlotManager::from_resources(&resources);
    assert!(
        !manager.is_history(ResourceHandle(0)),
        "Imported resource must NOT be identified as history",
    );
}

/// is_history returns false for handles not in the manager.
#[test]
fn is_history_unknown_returns_false() {
    let resources = vec![history_resource(0, "h3", 3)];
    let manager = HistorySlotManager::from_resources(&resources);
    assert!(
        !manager.is_history(ResourceHandle(99)),
        "is_history must return false for unknown handle",
    );
}

/// is_history returns false for all handles on an empty manager.
#[test]
fn is_history_empty_manager() {
    let manager = HistorySlotManager::new();
    assert!(!manager.is_history(ResourceHandle(0)));
    assert!(!manager.is_history(ResourceHandle(1)));
    assert!(!manager.is_history(ResourceHandle(999)));
}

// =============================================================================
// SECTION 7 -- Multiple history resources with different lengths
// =============================================================================

/// Multiple history resources with different lengths are tracked independently.
#[test]
fn multiple_history_resources_different_lengths() {
    let resources = vec![
        history_resource(0, "double", 2),
        history_resource(1, "triple", 3),
        history_resource(2, "quad", 4),
    ];
    let manager = HistorySlotManager::from_resources(&resources);

    let h0 = ResourceHandle(0);
    let h1 = ResourceHandle(1);
    let h2 = ResourceHandle(2);

    // All three are detected as history.
    assert!(manager.is_history(h0));
    assert!(manager.is_history(h1));
    assert!(manager.is_history(h2));

    // Each has its own history_length.
    assert_eq!(manager.history_length(h0), Some(2));
    assert_eq!(manager.history_length(h1), Some(3));
    assert_eq!(manager.history_length(h2), Some(4));

    // 2-slot: 0->0, 1->1, 2->0, 3->1
    assert_eq!(manager.slot_for(h0, 0), Some(0));
    assert_eq!(manager.slot_for(h0, 1), Some(1));
    assert_eq!(manager.slot_for(h0, 2), Some(0));
    assert_eq!(manager.slot_for(h0, 3), Some(1));

    // 3-slot: 0->0, 1->1, 2->2, 3->0
    assert_eq!(manager.slot_for(h1, 0), Some(0));
    assert_eq!(manager.slot_for(h1, 1), Some(1));
    assert_eq!(manager.slot_for(h1, 2), Some(2));
    assert_eq!(manager.slot_for(h1, 3), Some(0));

    // 4-slot: 0->0, 1->1, 2->2, 3->3, 4->0
    assert_eq!(manager.slot_for(h2, 0), Some(0));
    assert_eq!(manager.slot_for(h2, 1), Some(1));
    assert_eq!(manager.slot_for(h2, 2), Some(2));
    assert_eq!(manager.slot_for(h2, 3), Some(3));
    assert_eq!(manager.slot_for(h2, 4), Some(0));
}

/// Multiple history resources with the same length all work correctly.
#[test]
fn multiple_history_resources_same_length() {
    let resources = vec![
        history_resource(0, "a", 3),
        history_resource(1, "b", 3),
        history_resource(2, "c", 3),
    ];
    let manager = HistorySlotManager::from_resources(&resources);

    for i in 0..3 {
        let h = ResourceHandle(i);
        assert!(manager.is_history(h), "Resource {} must be history", i);
        assert_eq!(
            manager.history_length(h),
            Some(3),
            "Resource {} history length must be 3",
            i,
        );
    }

    // All have the same slot mapping for their shared history length.
    for &(frame, expected_slot) in &[(0, 0), (1, 1), (2, 2), (3, 0), (4, 1), (5, 2)] {
        assert_eq!(
            manager.slot_for(ResourceHandle(0), frame),
            Some(expected_slot),
            "Resource 0 slot at frame {}",
            frame,
        );
        assert_eq!(
            manager.slot_for(ResourceHandle(1), frame),
            Some(expected_slot),
            "Resource 1 slot at frame {}",
            frame,
        );
        assert_eq!(
            manager.slot_for(ResourceHandle(2), frame),
            Some(expected_slot),
            "Resource 2 slot at frame {}",
            frame,
        );
    }
}

// =============================================================================
// SECTION 8 -- Mixed history and non-history resources
// =============================================================================

/// Manager correctly handles a mix of history, transient, and imported resources.
#[test]
fn mixed_history_and_non_history() {
    let resources = vec![
        history_resource(0, "hdr_accum", 2),
        transient_resource(1, "albedo_rt"),
        history_resource(2, "motion_vectors", 4),
        transient_resource(3, "depth_rt"),
        make_resource(4, "external", ResourceLifetime::Imported),
        history_resource(5, "history_buffer", 3),
    ];
    let manager = HistorySlotManager::from_resources(&resources);

    // History resources.
    assert!(manager.is_history(ResourceHandle(0)));
    assert_eq!(manager.history_length(ResourceHandle(0)), Some(2));

    assert!(manager.is_history(ResourceHandle(2)));
    assert_eq!(manager.history_length(ResourceHandle(2)), Some(4));

    assert!(manager.is_history(ResourceHandle(5)));
    assert_eq!(manager.history_length(ResourceHandle(5)), Some(3));

    // Transient resources.
    assert!(!manager.is_history(ResourceHandle(1)));
    assert_eq!(manager.history_length(ResourceHandle(1)), None);
    assert_eq!(manager.slot_for(ResourceHandle(1), 0), Some(0));
    assert_eq!(manager.slot_for(ResourceHandle(1), 7), Some(0));

    assert!(!manager.is_history(ResourceHandle(3)));
    assert_eq!(manager.history_length(ResourceHandle(3)), None);
    assert_eq!(manager.slot_for(ResourceHandle(3), 0), Some(0));
    assert_eq!(manager.slot_for(ResourceHandle(3), 42), Some(0));

    // Imported resources.
    assert!(!manager.is_history(ResourceHandle(4)));
    assert_eq!(manager.history_length(ResourceHandle(4)), None);
    assert_eq!(manager.slot_for(ResourceHandle(4), 0), Some(0));
    assert_eq!(manager.slot_for(ResourceHandle(4), 99), Some(0));
}

// =============================================================================
// SECTION 9 -- Empty manager returns None for all queries
// =============================================================================

/// Empty manager (created via new()) returns None for slot_for and
/// history_length, and false for is_history, for any handle.
#[test]
fn empty_manager_all_queries_none_or_false() {
    let manager = HistorySlotManager::new();

    // Test multiple handles on empty manager.
    for handle in &[0u32, 1, 42, 100, u32::MAX] {
        let h = ResourceHandle(*handle);
        assert!(
            !manager.is_history(h),
            "Empty manager: is_history({:?}) must be false",
            h,
        );
        assert_eq!(
            manager.history_length(h),
            None,
            "Empty manager: history_length({:?}) must be None",
            h,
        );
        assert_eq!(
            manager.slot_for(h, 0),
            None,
            "Empty manager: slot_for({:?}, 0) must be None",
            h,
        );
        assert_eq!(
            manager.slot_for(h, 7),
            None,
            "Empty manager: slot_for({:?}, 7) must be None",
            h,
        );
    }
}

/// Empty manager (created via from_resources(&[])) also returns None for all
/// queries -- both constructors produce the same empty state.
#[test]
fn empty_from_resources_all_queries_none() {
    let resources: Vec<IrResource> = vec![];
    let manager = HistorySlotManager::from_resources(&resources);

    let h = ResourceHandle(0);
    assert!(!manager.is_history(h));
    assert_eq!(manager.history_length(h), None);
    assert_eq!(manager.slot_for(h, 0), None);
}

// =============================================================================
// SECTION 10 -- Edge cases
// =============================================================================

/// Unknown handle queries on a populated manager.
#[test]
fn unknown_handle_on_populated_manager() {
    let resources = vec![
        history_resource(0, "known", 3),
        transient_resource(1, "known_transient"),
    ];
    let manager = HistorySlotManager::from_resources(&resources);

    let unknown = ResourceHandle(99);
    assert!(!manager.is_history(unknown));
    assert_eq!(manager.history_length(unknown), None);
    assert_eq!(manager.slot_for(unknown, 0), None);
    assert_eq!(manager.slot_for(unknown, 5), None);
}

/// ResourceLifetime::History(0) -- edge case for zero-length history.
#[test]
fn history_length_zero_does_not_panic() {
    let resources = vec![history_resource(0, "zero_len", 0)];
    let manager = HistorySlotManager::from_resources(&resources);
    let h = ResourceHandle(0);

    // Must be detected as a history resource.
    assert!(manager.is_history(h), "History(0) resource is history");
    // Reports length 0.
    assert_eq!(
        manager.history_length(h),
        Some(0),
        "History(0) resource reports length 0",
    );
}

/// Non-contiguous resource handles are all tracked correctly.
#[test]
fn non_contiguous_handles() {
    let resources = vec![
        history_resource(10, "a", 2),
        transient_resource(20, "b"),
        history_resource(30, "c", 4),
        make_resource(40, "d", ResourceLifetime::Imported),
    ];
    let manager = HistorySlotManager::from_resources(&resources);

    assert!(manager.is_history(ResourceHandle(10)));
    assert!(!manager.is_history(ResourceHandle(20)));
    assert!(manager.is_history(ResourceHandle(30)));
    assert!(!manager.is_history(ResourceHandle(40)));

    assert_eq!(manager.history_length(ResourceHandle(10)), Some(2));
    assert_eq!(manager.history_length(ResourceHandle(20)), None);
    assert_eq!(manager.history_length(ResourceHandle(30)), Some(4));
    assert_eq!(manager.history_length(ResourceHandle(40)), None);

    assert_eq!(manager.slot_for(ResourceHandle(10), 1), Some(1));
    assert_eq!(manager.slot_for(ResourceHandle(20), 99), Some(0));
    assert_eq!(manager.slot_for(ResourceHandle(30), 4), Some(0));
    assert_eq!(manager.slot_for(ResourceHandle(40), 77), Some(0));
}

/// Large number of history resources are all tracked correctly.
#[test]
fn many_history_resources() {
    let n = 50;
    let resources: Vec<IrResource> = (0..n)
        .map(|i| history_resource(i, &format!("h{}", i), (i % 16) + 1))
        .collect();
    let manager = HistorySlotManager::from_resources(&resources);

    for i in 0..n {
        let h = ResourceHandle(i);
        assert!(
            manager.is_history(h),
            "Resource {} must be history",
            i,
        );
        let expected_len = (i % 16) + 1;
        assert_eq!(
            manager.history_length(h),
            Some(expected_len),
            "Resource {} history length must be {}",
            i,
            expected_len,
        );
        assert_eq!(
            manager.slot_for(h, 0),
            Some(0),
            "Resource {} slot at frame 0 must be 0",
            i,
        );
        assert_eq!(
            manager.slot_for(h, expected_len),
            Some(0),
            "Resource {} slot at frame = length must wrap to 0",
            i,
        );
    }
}
