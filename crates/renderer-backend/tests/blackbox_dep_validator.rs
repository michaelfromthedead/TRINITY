// SPDX-License-Identifier: MIT
//
// blackbox_dep_validator.rs -- Blackbox contract tests for T-FG-2.6
// (DependencyValidator).
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests construct IR data directly -- no implementation details, no internal
// fields, no private methods.
//
// Public API under test:
//   DependencyValidator::validate(passes, resources, edges)
//     -> Result<(), Vec<String>>
//
//   Performs four structural checks on the edge list:
//
//   1. No dangling pass references  -- every `from` and `to` in every edge
//      exists in the provided pass list.
//   2. All resource handles exist   -- every resource referenced by an edge
//      exists in the provided resource list (ResourceHandle::NONE is flagged).
//   3. RAW edges follow Write->Read  -- for every RAW edge, the source pass
//      must list the resource in its write-set and the target pass must list
//      it in its read-set.
//   4. No self-loops                -- no edge has `from == to`.
//
// Acceptance criteria (T-FG-2.6):
//   1.  Valid edge with valid passes and resources -> Ok(())
//   2.  Edge from-pass missing from passes -> Err
//   3.  Edge to-pass missing from passes -> Err
//   4.  Both from and to pass missing -> Err with two messages
//   5.  Edge references resource not in resources -> Err
//   6.  ResourceHandle::NONE in edge -> Err
//   7.  RAW edge where source does not write the resource -> Err
//   8.  RAW edge where target does not read the resource -> Err
//   9.  RAW edge where both source write and target read are wrong -> Err
//  10.  Self-loop edge (from == to) -> Err
//  11.  Self-loop is caught before dangling-ref check (continue skips)
//  12.  Empty edge list passes validation -> Ok(())
//  13.  Multiple edges all valid -> Ok(())
//  14.  Multiple violations accumulate into single Err(Vec<String>)
//  15.  RAW edge where source writes and target reads passes -> Ok(())
//  16.  WAR and WAW edges are not checked for Write->Read pattern -> Ok(())
//  17.  Error messages contain relevant context (pass index, resource handle)
//  18.  Dangling from-pass still detected when self-loop is also present
//  19.  Single edge with all four violations -> four error messages
//  20.  Large batch of valid edges -> Ok(())

use renderer_backend::frame_graph::{
    DependencyValidator, DispatchSource, EdgeType, IrEdge, IrPass, IrResource,
    PassIndex, ResourceDesc, ResourceHandle, ResourceLifetime, ResourceState,
};

// =============================================================================
// Helpers
// =============================================================================

/// Creates a storage-buffer IrResource at the given handle.
fn storage_buf(handle: u32, name: &str, size: u64) -> IrResource {
    IrResource::new(
        ResourceHandle(handle),
        name,
        ResourceDesc::Buffer(renderer_backend::frame_graph::BufferDesc {
            size,
            usage: "storage".into(),
            is_indirect_arg: false,
        }),
        ResourceLifetime::Transient,
        ResourceState::Uninitialized,
    )
}

/// Creates a compute pass that writes the listed resources and reads none.
fn write_pass(index: usize, name: &str, writes: &[u32]) -> IrPass {
    let mut pass = IrPass::compute(
        PassIndex(index),
        name,
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        renderer_backend::frame_graph::ViewType::Storage,
    );
    for &w in writes {
        pass.access_set.writes.push(ResourceHandle(w));
    }
    pass
}

/// Creates a compute pass that reads the listed resources and writes none.
fn read_pass(index: usize, name: &str, reads: &[u32]) -> IrPass {
    let mut pass = IrPass::compute(
        PassIndex(index),
        name,
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        renderer_backend::frame_graph::ViewType::Storage,
    );
    for &r in reads {
        pass.access_set.reads.push(ResourceHandle(r));
    }
    pass
}

/// Creates a compute pass that both reads and writes the listed resources.
fn read_write_pass(index: usize, name: &str, reads: &[u32], writes: &[u32]) -> IrPass {
    let mut pass = IrPass::compute(
        PassIndex(index),
        name,
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        renderer_backend::frame_graph::ViewType::Storage,
    );
    for &r in reads {
        pass.access_set.reads.push(ResourceHandle(r));
    }
    for &w in writes {
        pass.access_set.writes.push(ResourceHandle(w));
    }
    pass
}

/// Shorthand for DependencyValidator::validate.
fn validate(
    passes: &[IrPass],
    resources: &[IrResource],
    edges: &[IrEdge],
) -> Result<(), Vec<String>> {
    DependencyValidator::validate(passes, resources, edges)
}

/// Creates a ResourceHandle from a u32.
fn res(handle: u32) -> ResourceHandle {
    ResourceHandle(handle)
}

// =============================================================================
// SECTION 1 -- Happy path: valid passes, resources, and edges
// =============================================================================

/// A single valid edge with all referenced passes and resources present
/// produces Ok(()).
#[test]
fn valid_single_edge_returns_ok() {
    let passes = vec![
        write_pass(0, "writer", &[1]),
        read_pass(1, "reader", &[1]),
    ];
    let resources = vec![storage_buf(1, "buf", 256)];
    let edges = vec![IrEdge::new(PassIndex(0), PassIndex(1), res(1), EdgeType::RAW)];

    let result = validate(&passes, &resources, &edges);
    assert!(result.is_ok(), "Valid edge must produce Ok(()); got {:?}", result);
}

/// An empty edge list passes validation even with passes and resources present.
#[test]
fn empty_edges_returns_ok() {
    let passes = vec![write_pass(0, "p0", &[])];
    let resources = vec![storage_buf(1, "orphan", 64)];
    let edges = vec![];

    let result = validate(&passes, &resources, &edges);
    assert!(result.is_ok(), "Empty edges must produce Ok(())");
}

/// Multiple valid RAW edges across different resources all pass.
#[test]
fn multiple_valid_raw_edges_returns_ok() {
    let passes = vec![
        write_pass(0, "producer_a", &[1]),
        write_pass(1, "producer_b", &[2]),
        read_pass(2, "consumer", &[1, 2]),
    ];
    let resources = vec![
        storage_buf(1, "resource_a", 64),
        storage_buf(2, "resource_b", 128),
    ];
    let edges = vec![
        IrEdge::new(PassIndex(0), PassIndex(2), res(1), EdgeType::RAW),
        IrEdge::new(PassIndex(1), PassIndex(2), res(2), EdgeType::RAW),
    ];

    let result = validate(&passes, &resources, &edges);
    assert!(result.is_ok(), "Multiple valid RAW edges must produce Ok(()); got {:?}", result);
}

/// WAR and WAW edges are accepted without a Write->Read pattern check.
#[test]
fn war_and_waw_edges_accepted() {
    let passes = vec![
        read_pass(0, "reader", &[1]),
        write_pass(1, "writer", &[1]),
    ];
    let resources = vec![storage_buf(1, "buf", 64)];
    let edges = vec![
        IrEdge::new(PassIndex(0), PassIndex(1), res(1), EdgeType::WAR),
        IrEdge::new(PassIndex(0), PassIndex(1), res(1), EdgeType::WAW),
    ];

    let result = validate(&passes, &resources, &edges);
    assert!(result.is_ok(), "WAR and WAW edges must pass without Write->Read check; got {:?}", result);
}

/// A valid RAW edge where the source writes and the target reads passes.
#[test]
fn valid_raw_edge_pattern_passes() {
    let passes = vec![
        write_pass(0, "writer", &[1]),
        read_pass(1, "reader", &[1]),
    ];
    let resources = vec![storage_buf(1, "buf", 64)];
    let edges = vec![IrEdge::new(PassIndex(0), PassIndex(1), res(1), EdgeType::RAW)];

    let result = validate(&passes, &resources, &edges);
    assert!(result.is_ok(), "Valid RAW W->R pattern must pass; got {:?}", result);
}

/// A ReadWrite pass on the source side satisfies the RAW Write check.
#[test]
fn raw_from_readwrite_pass_passes() {
    let passes = vec![
        read_write_pass(0, "rmw", &[1], &[1]),
        read_pass(1, "reader", &[1]),
    ];
    let resources = vec![storage_buf(1, "buf", 64)];
    let edges = vec![IrEdge::new(PassIndex(0), PassIndex(1), res(1), EdgeType::RAW)];

    let result = validate(&passes, &resources, &edges);
    assert!(result.is_ok(), "RAW from ReadWrite pass must pass; got {:?}", result);
}

// =============================================================================
// SECTION 2 -- Dangling pass references (Check 1)
// =============================================================================

/// An edge whose from-pass index does not exist in the pass list is caught.
#[test]
fn dangling_from_pass_caught() {
    let passes = vec![read_pass(0, "p0", &[])];
    let resources = vec![];
    let edges = vec![IrEdge::new(PassIndex(99), PassIndex(0), res(1), EdgeType::RAW)];

    let result = validate(&passes, &resources, &edges);
    assert!(result.is_err(), "Dangling from-pass must produce errors");
}

/// An edge whose to-pass index does not exist in the pass list is caught.
#[test]
fn dangling_to_pass_caught() {
    let passes = vec![write_pass(0, "p0", &[1])];
    let resources = vec![storage_buf(1, "buf", 64)];
    let edges = vec![IrEdge::new(PassIndex(0), PassIndex(999), res(1), EdgeType::RAW)];

    let result = validate(&passes, &resources, &edges);
    assert!(result.is_err(), "Dangling to-pass must produce errors");
}

/// Both from-pass and to-pass dangling produce two error messages.
#[test]
fn both_ends_dangling_two_errors() {
    let passes = vec![];
    let resources = vec![];
    let edges = vec![IrEdge::new(PassIndex(10), PassIndex(20), res(1), EdgeType::RAW)];

    let result = validate(&passes, &resources, &edges);
    assert!(result.is_err(), "Both ends dangling must produce errors");
    let errors = result.unwrap_err();
    // Edge has from=10 and to=20 missing + resource not found => 3 errors expected.
    assert!(
        errors.len() >= 2,
        "Expected >=2 error messages for dangling from+to; got {}",
        errors.len(),
    );
}

/// Multiple edges with dangling references produce multiple messages.
#[test]
fn multiple_dangling_refs_caught() {
    let passes = vec![write_pass(0, "only_pass", &[])];
    let resources = vec![];
    let edges = vec![
        IrEdge::new(PassIndex(99), PassIndex(0), res(1), EdgeType::RAW),
        IrEdge::new(PassIndex(0), PassIndex(88), res(2), EdgeType::RAW),
    ];

    let result = validate(&passes, &resources, &edges);
    assert!(result.is_err(), "Multiple dangling refs must produce errors");
    let errors = result.unwrap_err();
    // Edge 0: from=99 dangling, res(1) missing. Edge 1: to=88 dangling, res(2) missing.
    assert!(
        errors.len() >= 4,
        "Expected >=4 messages (2 dangling + 2 missing resources); got {}",
        errors.len(),
    );
}

// =============================================================================
// SECTION 3 -- Missing resource handles (Check 2)
// =============================================================================

/// An edge referencing a resource handle that does not exist in the resource
/// list is caught.
#[test]
fn missing_resource_handle_caught() {
    let passes = vec![
        write_pass(0, "writer", &[1]),
        read_pass(1, "reader", &[1]),
    ];
    let resources = vec![]; // no resources at all
    let edges = vec![IrEdge::new(PassIndex(0), PassIndex(1), res(1), EdgeType::RAW)];

    let result = validate(&passes, &resources, &edges);
    assert!(result.is_err(), "Missing resource handle must produce errors");
}

/// ResourceHandle::NONE in an edge is flagged as invalid.
#[test]
fn none_resource_handle_caught() {
    let passes = vec![
        write_pass(0, "writer", &[]),
        read_pass(1, "reader", &[]),
    ];
    let resources = vec![];
    let edges = vec![IrEdge::new(
        PassIndex(0),
        PassIndex(1),
        ResourceHandle::NONE,
        EdgeType::RAW,
    )];

    let result = validate(&passes, &resources, &edges);
    assert!(result.is_err(), "ResourceHandle::NONE must be flagged");
}

/// Multiple edges with missing resources are all reported.
#[test]
fn multiple_missing_resources_caught() {
    let passes = vec![
        write_pass(0, "w0", &[]),
        read_pass(1, "r1", &[]),
        write_pass(2, "w2", &[]),
        read_pass(3, "r3", &[]),
    ];
    let resources = vec![];
    let edges = vec![
        IrEdge::new(PassIndex(0), PassIndex(1), res(10), EdgeType::RAW),
        IrEdge::new(PassIndex(2), PassIndex(3), res(20), EdgeType::RAW),
    ];

    let result = validate(&passes, &resources, &edges);
    assert!(result.is_err(), "Multiple missing resources must produce errors");
    let errors = result.unwrap_err();
    // Each edge has a missing resource, and each resource handle is absent.
    assert!(
        errors.len() >= 2,
        "Expected >=2 errors for 2 missing resources; got {}",
        errors.len(),
    );
}

/// A resource handle that exists in the resource list does not trigger an error.
#[test]
fn existing_resource_handle_ok() {
    let passes = vec![
        write_pass(0, "writer", &[1]),
        read_pass(1, "reader", &[1]),
    ];
    let resources = vec![storage_buf(1, "buf", 64)];
    let edges = vec![IrEdge::new(PassIndex(0), PassIndex(1), res(1), EdgeType::RAW)];

    let result = validate(&passes, &resources, &edges);
    assert!(result.is_ok(), "Existing resource handle must pass; got {:?}", result);
}

// =============================================================================
// SECTION 4 -- RAW pattern: Write->Read verification (Check 3)
// =============================================================================

/// A RAW edge where the source pass does NOT write the resource is caught.
#[test]
fn raw_source_does_not_write_caught() {
    let passes = vec![
        read_pass(0, "source", &[1]),    // reads but does NOT write res(1)
        read_pass(1, "target", &[1]),
    ];
    let resources = vec![storage_buf(1, "buf", 64)];
    let edges = vec![IrEdge::new(PassIndex(0), PassIndex(1), res(1), EdgeType::RAW)];

    let result = validate(&passes, &resources, &edges);
    assert!(result.is_err(), "RAW source not writing must produce errors");
}

/// A RAW edge where the target pass does NOT read the resource is caught.
#[test]
fn raw_target_does_not_read_caught() {
    let passes = vec![
        write_pass(0, "source", &[1]),
        write_pass(1, "target", &[2]),   // writes res(2) but does NOT read res(1)
    ];
    let resources = vec![
        storage_buf(1, "buf_a", 64),
        storage_buf(2, "buf_b", 64),
    ];
    let edges = vec![IrEdge::new(PassIndex(0), PassIndex(1), res(1), EdgeType::RAW)];

    let result = validate(&passes, &resources, &edges);
    assert!(result.is_err(), "RAW target not reading must produce errors");
}

/// A RAW edge where both the source does not write AND the target does not
/// read is caught with two error messages.
#[test]
fn raw_both_ends_wrong_two_errors() {
    let passes = vec![
        read_pass(0, "source", &[2]),    // reads res(2), not res(1)
        write_pass(1, "target", &[2]),   // writes res(2), not res(1)
    ];
    let resources = vec![
        storage_buf(1, "buf", 64),
        storage_buf(2, "other", 64),
    ];
    let edges = vec![IrEdge::new(PassIndex(0), PassIndex(1), res(1), EdgeType::RAW)];

    let result = validate(&passes, &resources, &edges);
    assert!(result.is_err(), "RAW both ends wrong must produce errors");
    let errors = result.unwrap_err();
    // Expected: 1 "source does not write" + 1 "target does not read".
    assert!(
        errors.len() >= 2,
        "Expected >=2 errors for both ends of RAW wrong; got {}",
        errors.len(),
    );
}

/// A RAW edge where the source pass is dangling skips the RAW Write->Read
/// check (the dangling ref error is already reported).
#[test]
fn raw_dangling_source_skips_raw_check() {
    // Edge from-pass(99) doesn't exist, so RAW check is skipped.
    let passes = vec![read_pass(0, "target", &[1])];
    let resources = vec![storage_buf(1, "buf", 64)];
    let edges = vec![IrEdge::new(PassIndex(99), PassIndex(0), res(1), EdgeType::RAW)];

    let result = validate(&passes, &resources, &edges);
    assert!(result.is_err(), "RAW with dangling source must produce errors");
    let errors = result.unwrap_err();
    // Only the dangling ref error; no RAW Write->Read error because the
    // from-pass is not found, so the check is skipped.
    let has_dangling = errors.iter().any(|e| e.contains("from-pass"));
    assert!(has_dangling, "Dangling ref error must be present; got {:?}", errors);
}

// =============================================================================
// SECTION 5 -- Self-loops (Check 4)
// =============================================================================

/// A self-loop edge (from == to) is caught.
#[test]
fn self_loop_caught() {
    let passes = vec![write_pass(0, "p0", &[1])];
    let resources = vec![storage_buf(1, "buf", 64)];
    let edges = vec![IrEdge::new(PassIndex(0), PassIndex(0), res(1), EdgeType::RAW)];

    let result = validate(&passes, &resources, &edges);
    assert!(result.is_err(), "Self-loop edge must produce errors");
}

/// A self-loop is reported before dangling-ref checks (the `continue` in the
/// implementation skips further checks for that edge).
#[test]
fn self_loop_skips_dangling_and_raw_checks() {
    let passes = vec![];
    let resources = vec![];
    // from == to == 5, which is also dangling (no passes), but self-loop
    // should be the only error since it's checked first.
    let edges = vec![IrEdge::new(PassIndex(5), PassIndex(5), res(99), EdgeType::RAW)];

    let result = validate(&passes, &resources, &edges);
    assert!(result.is_err(), "Self-loop must produce errors");
    let errors = result.unwrap_err();
    // Only the self-loop message, no dangling ref or missing resource messages.
    assert_eq!(
        errors.len(),
        1,
        "Self-loop must report only itself, skipping other checks; got {:?}",
        errors,
    );
    let first = &errors[0];
    assert!(
        first.to_lowercase().contains("self-loop"),
        "Self-loop message must contain 'self-loop': {}",
        first,
    );
}

/// Multiple self-loops across different edges are all reported.
#[test]
fn multiple_self_loops_caught() {
    let passes = vec![
        write_pass(0, "p0", &[1]),
        write_pass(1, "p1", &[2]),
    ];
    let resources = vec![
        storage_buf(1, "buf_a", 64),
        storage_buf(2, "buf_b", 64),
    ];
    let edges = vec![
        IrEdge::new(PassIndex(0), PassIndex(0), res(1), EdgeType::RAW),
        IrEdge::new(PassIndex(1), PassIndex(1), res(2), EdgeType::WAR),
    ];

    let result = validate(&passes, &resources, &edges);
    assert!(result.is_err(), "Multiple self-loops must produce errors");
    let errors = result.unwrap_err();
    assert_eq!(
        errors.len(),
        2,
        "Two self-loop edges must produce exactly 2 errors; got {}",
        errors.len(),
    );
}

// =============================================================================
// SECTION 6 -- Combined violations (multiple checks)
// =============================================================================

/// A single edge that simultaneously has a dangling from-pass AND a missing
/// resource AND a RAW pattern violation produces error messages for each.
#[test]
fn single_edge_multiple_violations() {
    let passes = vec![read_pass(0, "target", &[1])];
    let resources = vec![]; // no resources
    // from=99 dangling, res(1) missing, RAW but from-pass doesn't write res(1)
    let edges = vec![IrEdge::new(PassIndex(99), PassIndex(0), res(1), EdgeType::RAW)];

    let result = validate(&passes, &resources, &edges);
    assert!(result.is_err(), "Multiple violations must produce errors");
    let errors = result.unwrap_err();
    // Expected: dangling from-pass(99) + missing resource(1) = 2.
    // RAW check is skipped because from-pass is dangling.
    assert!(
        errors.len() >= 2,
        "Expected >=2 errors (dangling + missing resource); got {}",
        errors.len(),
    );
}

/// Two edges each with different violation types produce all errors.
#[test]
fn two_edges_different_violations_accumulated() {
    let passes = vec![write_pass(0, "p0", &[1])];
    let resources = vec![storage_buf(1, "buf", 64)];
    let edges = vec![
        // Edge 0: self-loop
        IrEdge::new(PassIndex(0), PassIndex(0), res(1), EdgeType::RAW),
        // Edge 1: dangling to-pass + RAW target doesn't read
        IrEdge::new(PassIndex(0), PassIndex(99), res(1), EdgeType::RAW),
    ];

    let result = validate(&passes, &resources, &edges);
    assert!(result.is_err(), "Combined violations must produce errors");
    let errors = result.unwrap_err();
    // Edge 0: self-loop -> 1 error (continue skips other checks).
    // Edge 1: dangling to-pass(99) -> 1 error (RAW check skipped because
    //         to-pass is not found; resource check passes since res(1) exists).
    // Total = 2.
    assert_eq!(
        errors.len(),
        2,
        "Expected 2 errors (self-loop + dangling to-pass); got {}",
        errors.len(),
    );
}

/// All four check categories violated across multiple edges.
#[test]
fn all_four_checks_violated() {
    let passes = vec![
        write_pass(0, "writer", &[1]),
        read_pass(1, "reader", &[]),  // does not read anything
    ];
    let resources = vec![]; // no resources
    let edges = vec![
        // Self-loop (check 4)
        IrEdge::new(PassIndex(0), PassIndex(0), res(1), EdgeType::RAW),
        // Dangling to-pass (check 1) + missing resource (check 2)
        IrEdge::new(PassIndex(0), PassIndex(99), res(2), EdgeType::RAW),
        // RAW: source writes but target doesn't read (check 3)
        IrEdge::new(PassIndex(0), PassIndex(1), res(1), EdgeType::RAW),
    ];

    let result = validate(&passes, &resources, &edges);
    assert!(result.is_err(), "All four violations must produce errors");
    let errors = result.unwrap_err();
    assert!(
        errors.len() >= 4,
        "Expected >=4 errors across all check categories; got {}",
        errors.len(),
    );
}

// =============================================================================
// SECTION 7 -- Error message content
// =============================================================================

/// Dangling from-pass message contains the pass index.
#[test]
fn dangling_from_message_contains_pass_index() {
    let passes = vec![write_pass(0, "p0", &[])];
    let resources = vec![];
    let edges = vec![IrEdge::new(PassIndex(42), PassIndex(0), res(1), EdgeType::RAW)];

    let result = validate(&passes, &resources, &edges);
    let errors = result.unwrap_err();
    let all = errors.join(" ");
    assert!(all.contains("42"), "Dangling ref error should mention pass index 42: {}", all);
}

/// Dangling to-pass message contains the pass index.
#[test]
fn dangling_to_message_contains_pass_index() {
    let passes = vec![write_pass(0, "p0", &[])];
    let resources = vec![];
    let edges = vec![IrEdge::new(PassIndex(0), PassIndex(77), res(1), EdgeType::RAW)];

    let result = validate(&passes, &resources, &edges);
    let errors = result.unwrap_err();
    let all = errors.join(" ");
    assert!(all.contains("77"), "Dangling to-pass error should mention pass index 77: {}", all);
}

/// Missing resource message contains the resource handle.
#[test]
fn missing_resource_message_contains_handle() {
    let passes = vec![
        write_pass(0, "p0", &[]),
        read_pass(1, "p1", &[]),
    ];
    let resources = vec![];
    let edges = vec![IrEdge::new(PassIndex(0), PassIndex(1), res(55), EdgeType::RAW)];

    let result = validate(&passes, &resources, &edges);
    let errors = result.unwrap_err();
    let all = errors.join(" ");
    assert!(all.contains("55"), "Missing resource error should mention handle 55: {}", all);
}

/// Self-loop message contains "self-loop" (case-insensitive).
#[test]
fn self_loop_message_contains_self_loop() {
    let passes = vec![write_pass(0, "p0", &[1])];
    let resources = vec![storage_buf(1, "buf", 64)];
    let edges = vec![IrEdge::new(PassIndex(0), PassIndex(0), res(1), EdgeType::RAW)];

    let result = validate(&passes, &resources, &edges);
    let errors = result.unwrap_err();
    let all = errors.join(" ").to_lowercase();
    assert!(all.contains("self-loop"), "Self-loop error must contain 'self-loop': {:?}", errors);
}

/// RAW source-not-write message contains "does not write".
#[test]
fn raw_source_not_write_message_content() {
    let passes = vec![
        read_pass(0, "source", &[1]),
        read_pass(1, "target", &[1]),
    ];
    let resources = vec![storage_buf(1, "buf", 64)];
    let edges = vec![IrEdge::new(PassIndex(0), PassIndex(1), res(1), EdgeType::RAW)];

    let result = validate(&passes, &resources, &edges);
    let errors = result.unwrap_err();
    let all = errors.join(" ").to_lowercase();
    assert!(all.contains("does not write"), "RAW source error must mention 'does not write': {:?}", errors);
}

/// RAW target-not-read message contains "does not read".
#[test]
fn raw_target_not_read_message_content() {
    let passes = vec![
        write_pass(0, "source", &[1]),
        write_pass(1, "target", &[2]),
    ];
    let resources = vec![
        storage_buf(1, "buf", 64),
        storage_buf(2, "other", 64),
    ];
    let edges = vec![IrEdge::new(PassIndex(0), PassIndex(1), res(1), EdgeType::RAW)];

    let result = validate(&passes, &resources, &edges);
    let errors = result.unwrap_err();
    let all = errors.join(" ").to_lowercase();
    assert!(all.contains("does not read"), "RAW target error must mention 'does not read': {:?}", errors);
}

/// Error messages mention the pass name for context.
#[test]
fn raw_pattern_message_contains_pass_name() {
    let passes = vec![
        read_pass(0, "my_source_pass", &[2]),
        read_pass(1, "my_target_pass", &[2]),
    ];
    let resources = vec![
        storage_buf(1, "buf", 64),
        storage_buf(2, "other", 64),
    ];
    let edges = vec![IrEdge::new(PassIndex(0), PassIndex(1), res(1), EdgeType::RAW)];

    let result = validate(&passes, &resources, &edges);
    let errors = result.unwrap_err();
    let all = errors.join(" ");
    assert!(
        all.contains("my_source_pass"),
        "RAW source error should mention pass name 'my_source_pass': {}",
        all,
    );
}

// =============================================================================
// SECTION 8 -- Boundary and edge cases
// =============================================================================

/// An edge referencing the first and last valid pass indices at boundary
/// passes validation.
#[test]
fn edge_boundary_valid_indices_ok() {
    let passes = vec![
        write_pass(0, "first", &[1]),
        read_pass(999, "last", &[1]),
    ];
    let resources = vec![storage_buf(1, "buf", 64)];
    let edges = vec![IrEdge::new(PassIndex(0), PassIndex(999), res(1), EdgeType::RAW)];

    let result = validate(&passes, &resources, &edges);
    assert!(result.is_ok(), "Boundary valid indices must pass; got {:?}", result);
}

/// A self-loop at the boundary (single pass, self-loop on itself).
#[test]
fn self_loop_single_pass_system() {
    let passes = vec![write_pass(0, "lonely", &[1])];
    let resources = vec![storage_buf(1, "buf", 64)];
    let edges = vec![IrEdge::new(PassIndex(0), PassIndex(0), res(1), EdgeType::RAW)];

    let result = validate(&passes, &resources, &edges);
    assert!(result.is_err(), "Self-loop on only pass must be caught");
}

/// An edge with a resource handle that exists passes, even when the passes
/// have empty access sets (no RAW pattern to check).
#[test]
fn edge_no_raw_check_for_non_raw_edges() {
    let passes = vec![
        write_pass(0, "p0", &[]),
        write_pass(1, "p1", &[]),
    ];
    let resources = vec![storage_buf(1, "buf", 64)];
    let edges = vec![
        IrEdge::new(PassIndex(0), PassIndex(1), res(1), EdgeType::WAR),
        IrEdge::new(PassIndex(0), PassIndex(1), res(1), EdgeType::WAW),
    ];

    let result = validate(&passes, &resources, &edges);
    assert!(result.is_ok(), "WAR/WAW edges with empty access sets must pass; got {:?}", result);
}

/// A zero-value pass index (PassIndex(0)) works correctly as an edge endpoint.
#[test]
fn pass_index_zero_as_endpoint_ok() {
    let passes = vec![
        write_pass(0, "p0", &[1]),
        read_pass(1, "p1", &[1]),
    ];
    let resources = vec![storage_buf(1, "buf", 64)];
    let edges = vec![IrEdge::new(PassIndex(0), PassIndex(1), res(1), EdgeType::RAW)];

    let result = validate(&passes, &resources, &edges);
    assert!(result.is_ok(), "PassIndex(0) as endpoint must pass; got {:?}", result);
}

// =============================================================================
// SECTION 9 -- Large batch stress test
// =============================================================================

/// 50 passes in a linear chain with 49 valid RAW edges passes validation.
#[test]
fn fifty_edges_linear_chain_ok() {
    // Linear chain: P0 writes res1; P1 reads res1 and writes res2; ...;
    // P48 reads res48 and writes res49; P49 reads res49.
    // Edge `i`: Pi -> P(i+1) on resource (i+1) as RAW.
    let passes: Vec<IrPass> = (0..50)
        .map(|i| {
            let name = format!("p{}", i);
            match i {
                0 => write_pass(i, &name, &[1]),
                49 => read_pass(i, &name, &[49]),
                n => read_write_pass(n, &name, &[n as u32], &[n as u32 + 1]),
            }
        })
        .collect();
    let resources: Vec<IrResource> = (1..=49)
        .map(|i| storage_buf(i, &format!("res{}", i), 64))
        .collect();
    let edges: Vec<IrEdge> = (0..49)
        .map(|i| IrEdge::new(PassIndex(i), PassIndex(i + 1), res(i as u32 + 1), EdgeType::RAW))
        .collect();

    let result = validate(&passes, &resources, &edges);
    assert!(result.is_ok(), "50-pass linear chain must pass; got {:?}", result);
}

// =============================================================================
// SECTION 10 -- Exact error count tests
// =============================================================================

/// A self-loop on an edge that also has a valid from/to produces exactly 1
/// error (the self-loop; other checks are skipped).
#[test]
fn self_loop_exact_one_error() {
    let passes = vec![write_pass(0, "p0", &[1])];
    let resources = vec![storage_buf(1, "buf", 64)];
    let edges = vec![IrEdge::new(PassIndex(0), PassIndex(0), res(1), EdgeType::RAW)];

    let result = validate(&passes, &resources, &edges);
    let errors = result.unwrap_err();
    assert_eq!(errors.len(), 1, "Self-loop must produce exactly 1 error");
}

/// A single dangling from-pass on an otherwise valid edge produces exactly 1
/// error for the dangling ref (plus 1 for missing resource if resource is gone).
#[test]
fn single_dangling_from_exact_errors() {
    let passes = vec![read_pass(0, "p0", &[])];
    let resources = vec![];
    let edges = vec![IrEdge::new(PassIndex(99), PassIndex(0), res(1), EdgeType::RAW)];

    let result = validate(&passes, &resources, &edges);
    let errors = result.unwrap_err();
    // 1 dangling from-pass + 1 missing resource = 2.
    assert_eq!(errors.len(), 2, "Dangling from + missing resource = 2 errors; got {}", errors.len());
}
