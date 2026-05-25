// Blackbox contract tests for T-FG-4.4 BarrierOptimizer.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::frame_graph::*` -- no internal fields,
// no private methods, no implementation details.
//
// Acceptance criteria (contract):
//   BarrierOptimizer::optimize() removes redundant barriers:
//
//   1. Same-state elimination   -- before == after => no transition needed.
//   2. Read-read elimination    -- both before and after are read-only states.
//   3. Deduplication            -- same (from, to, resource) triple, keep first.
//
// Read-only states (is_read_only):
//   VertexBuffer, IndexBuffer, IndirectArgument, DepthStencilReadOnly,
//   ShaderRead, TransferSrc
//
// Coverage:
//   1.  BarrierOptimizer new + default construction
//   2.  Debug + Clone traits
//   3.  Same-state elimination -- all same-state pairs
//   4.  Same-state preserves differing-state barriers in the same batch
//   5.  Read-read elimination  -- ShaderRead -> ShaderRead
//   6.  Read-read elimination  -- all read-only x read-only pairs
//   7.  Read-read preserves read -> write   (ShaderRead -> ColorAttachment)
//   8.  Read-read preserves write -> read   (ColorAttachment -> ShaderRead)
//   9.  Read-read preserves write -> write  (ColorAttachment -> ShaderReadWrite)
//  10.  Deduplication -- same (from, to, resource) triple collapsed
//  11.  Deduplication preserves distinct resources at same boundary
//  12.  Deduplication preserves distinct boundaries for same resource
//  13.  All three rules interact on a single input batch
//  14.  Empty input produces empty output
//  15.  Complex stress test -- many barriers of all categories
//  16.  Deduplication preserves first-occurrence state pair
//  17.  Read-only pairs interleaved with read-write pairs
//  18.  Output ordering is stable (preserves input order)
//  19.  Deduplication with state pairs that would otherwise survive rules 1+2
//  20.  TransferDst (write) paired with ShaderRead (read) -- NOT eliminated

use renderer_backend::frame_graph::{BarrierOptimizer, EdgeType, PassIndex, ResourceHandle, ResourceState};

type BarrierTuple = (PassIndex, PassIndex, ResourceHandle, EdgeType, ResourceState, ResourceState);

// =========================================================================
// SECTION 1 -- Construction traits
// =========================================================================

#[test]
fn barrier_optimizer_new_creates_instance() {
    let opt = BarrierOptimizer::new();
    let result = opt.optimize(&vec![]);
    assert!(result.is_empty(), "new optimizer on empty input");
}

#[test]
fn barrier_optimizer_default_creates_instance() {
    let opt = BarrierOptimizer::default();
    let result = opt.optimize(&vec![]);
    assert!(result.is_empty(), "default optimizer on empty input");
}

#[test]
fn barrier_optimizer_debug_format() {
    let opt = BarrierOptimizer::new();
    let s = format!("{:?}", opt);
    assert!(s.contains("BarrierOptimizer"), "Debug output contains type name");
}

#[test]
fn barrier_optimizer_clone_produces_independent_instance() {
    let a = BarrierOptimizer::new();
    let b = a.clone();
    // Both should behave identically on the same input.
    let input: Vec<BarrierTuple> = vec![(
        PassIndex(0),
        PassIndex(1),
        ResourceHandle(1),
        EdgeType::RAW,
        ResourceState::ColorAttachment,
        ResourceState::ColorAttachment, // same-state
    )];
    let result_a = a.optimize(&input.clone());
    let result_b = b.optimize(&input);
    assert_eq!(result_a, result_b);
}

// =========================================================================
// SECTION 2 -- Same-state elimination (Rule 1)
// =========================================================================

#[test]
fn same_state_before_equals_after_is_removed() {
    let opt = BarrierOptimizer::new();
    let input: Vec<BarrierTuple> = vec![(
        PassIndex(0),
        PassIndex(1),
        ResourceHandle(1),
        EdgeType::RAW,
        ResourceState::ShaderRead,
        ResourceState::ShaderRead,
    )];
    let result = opt.optimize(&input);
    assert!(
        result.is_empty(),
        "same-state barrier (ShaderRead -> ShaderRead) eliminated",
    );
}

#[test]
fn same_state_color_attachment_eliminated() {
    let opt = BarrierOptimizer::new();
    let input: Vec<BarrierTuple> = vec![(
        PassIndex(0),
        PassIndex(1),
        ResourceHandle(1),
        EdgeType::RAW,
        ResourceState::ColorAttachment,
        ResourceState::ColorAttachment,
    )];
    let result = opt.optimize(&input);
    assert!(result.is_empty());
}

#[test]
fn same_state_all_variants_eliminated() {
    let opt = BarrierOptimizer::new();
    let states = vec![
        ResourceState::Uninitialized,
        ResourceState::VertexBuffer,
        ResourceState::IndexBuffer,
        ResourceState::IndirectArgument,
        ResourceState::ColorAttachment,
        ResourceState::DepthStencilAttachment,
        ResourceState::DepthStencilReadOnly,
        ResourceState::ShaderRead,
        ResourceState::ShaderReadWrite,
        ResourceState::TransferSrc,
        ResourceState::TransferDst,
        ResourceState::AccelerationStructure,
        ResourceState::Present,
    ];
    let input: Vec<BarrierTuple> = states
        .iter()
        .enumerate()
        .map(|(i, s)| {
            (
                PassIndex(0),
                PassIndex(1),
                ResourceHandle(i as u32),
                EdgeType::RAW,
                s.clone(),
                s.clone(),
            )
        })
        .collect();

    let result = opt.optimize(&input);
    assert!(
        result.is_empty(),
        "every same-state barrier is eliminated regardless of variant",
    );
}

#[test]
fn same_state_preserves_differing_barriers_in_batch() {
    let opt = BarrierOptimizer::new();
    let input: Vec<BarrierTuple> = vec![
        // Same-state (should be removed)
        (
            PassIndex(0),
            PassIndex(1),
            ResourceHandle(1),
            EdgeType::RAW,
            ResourceState::ShaderRead,
            ResourceState::ShaderRead,
        ),
        // Genuine transition (should survive)
        (
            PassIndex(0),
            PassIndex(1),
            ResourceHandle(2),
            EdgeType::RAW,
            ResourceState::ColorAttachment,
            ResourceState::ShaderRead,
        ),
        // Same-state (should be removed)
        (
            PassIndex(1),
            PassIndex(2),
            ResourceHandle(3),
            EdgeType::RAW,
            ResourceState::ShaderReadWrite,
            ResourceState::ShaderReadWrite,
        ),
        // Another genuine transition (should survive)
        (
            PassIndex(1),
            PassIndex(2),
            ResourceHandle(4),
            EdgeType::RAW,
            ResourceState::Uninitialized,
            ResourceState::ShaderRead,
        ),
    ];
    let result = opt.optimize(&input);
    assert_eq!(result.len(), 2, "two same-state removed, two survive");
    // Surviving entries must have differing before/after states.
    for (_from, _to, handle, _edge_type, before, after) in &result {
        assert_ne!(before, after, "surviving barrier must not be same-state");
        // Verify which ones survived by handle.
        assert!(
            *handle == ResourceHandle(2) || *handle == ResourceHandle(4),
            "only barriers with actual transitions survive (handle={:?})",
            handle,
        );
    }
}

// =========================================================================
// SECTION 3 -- Read-read elimination (Rule 2)
// =========================================================================

#[test]
fn read_read_shader_read_to_shader_read_eliminated() {
    let opt = BarrierOptimizer::new();
    let input: Vec<BarrierTuple> = vec![(
        PassIndex(0),
        PassIndex(1),
        ResourceHandle(1),
        EdgeType::RAW,
        ResourceState::ShaderRead,
        ResourceState::ShaderRead,
    )];
    let result = opt.optimize(&input);
    assert!(
        result.is_empty(),
        "read-read barrier eliminated (ShaderRead -> ShaderRead)",
    );
}

#[test]
fn read_read_vertex_buffer_to_shader_read_eliminated() {
    let opt = BarrierOptimizer::new();
    let input: Vec<BarrierTuple> = vec![(
        PassIndex(0),
        PassIndex(1),
        ResourceHandle(1),
        EdgeType::RAW,
        ResourceState::VertexBuffer,
        ResourceState::ShaderRead,
    )];
    let result = opt.optimize(&input);
    assert!(
        result.is_empty(),
        "read-read eliminated (VertexBuffer -> ShaderRead)",
    );
}

#[test]
fn read_read_index_buffer_to_indirect_argument_eliminated() {
    let opt = BarrierOptimizer::new();
    let input: Vec<BarrierTuple> = vec![(
        PassIndex(0),
        PassIndex(1),
        ResourceHandle(1),
        EdgeType::RAW,
        ResourceState::IndexBuffer,
        ResourceState::IndirectArgument,
    )];
    let result = opt.optimize(&input);
    assert!(result.is_empty(), "read-read: IndexBuffer -> IndirectArgument");
}

#[test]
fn read_read_depth_stencil_read_only_to_shader_read_eliminated() {
    let opt = BarrierOptimizer::new();
    let input: Vec<BarrierTuple> = vec![(
        PassIndex(0),
        PassIndex(1),
        ResourceHandle(1),
        EdgeType::RAW,
        ResourceState::DepthStencilReadOnly,
        ResourceState::ShaderRead,
    )];
    let result = opt.optimize(&input);
    assert!(result.is_empty(), "read-read: DepthStencilReadOnly -> ShaderRead");
}

#[test]
fn read_read_transfer_src_to_shader_read_eliminated() {
    let opt = BarrierOptimizer::new();
    let input: Vec<BarrierTuple> = vec![(
        PassIndex(0),
        PassIndex(1),
        ResourceHandle(1),
        EdgeType::RAW,
        ResourceState::TransferSrc,
        ResourceState::ShaderRead,
    )];
    let result = opt.optimize(&input);
    assert!(result.is_empty(), "read-read: TransferSrc -> ShaderRead");
}

#[test]
fn read_read_shader_read_to_depth_stencil_read_only_eliminated() {
    let opt = BarrierOptimizer::new();
    let input: Vec<BarrierTuple> = vec![(
        PassIndex(0),
        PassIndex(1),
        ResourceHandle(1),
        EdgeType::RAW,
        ResourceState::ShaderRead,
        ResourceState::DepthStencilReadOnly,
    )];
    let result = opt.optimize(&input);
    assert!(result.is_empty(), "read-read: ShaderRead -> DepthStencilReadOnly");
}

#[test]
fn read_read_all_read_only_pairs_eliminated() {
    let opt = BarrierOptimizer::new();
    let read_only = vec![
        ResourceState::VertexBuffer,
        ResourceState::IndexBuffer,
        ResourceState::IndirectArgument,
        ResourceState::DepthStencilReadOnly,
        ResourceState::ShaderRead,
        ResourceState::TransferSrc,
    ];
    let mut idx = 0u32;
    let input: Vec<BarrierTuple> = read_only
        .iter()
        .flat_map(|before| {
            read_only.iter().map(move |after| {
                let entry = (
                    PassIndex(0),
                    PassIndex(1),
                    ResourceHandle(idx),
                    EdgeType::RAW,
                    before.clone(),
                    after.clone(),
                );
                idx += 1;
                entry
            })
        })
        .collect();

    // 6 read-only states => 6x6 = 36 pairs; all should be eliminated.
    let result = opt.optimize(&input);
    assert!(
        result.is_empty(),
        "all {} read-only x read-only pairs eliminated",
        36,
    );
}

#[test]
fn read_read_preserves_read_to_write() {
    let opt = BarrierOptimizer::new();
    let input: Vec<BarrierTuple> = vec![(
        PassIndex(0),
        PassIndex(1),
        ResourceHandle(1),
        EdgeType::RAW,
        ResourceState::ShaderRead,         // read-only
        ResourceState::ColorAttachment,    // write
    )];
    let result = opt.optimize(&input);
    assert_eq!(result.len(), 1, "read -> write barrier survives");
    assert_eq!(result[0].4, ResourceState::ShaderRead);
    assert_eq!(result[0].5, ResourceState::ColorAttachment);
}

#[test]
fn read_read_preserves_write_to_read() {
    let opt = BarrierOptimizer::new();
    let input: Vec<BarrierTuple> = vec![(
        PassIndex(0),
        PassIndex(1),
        ResourceHandle(1),
        EdgeType::RAW,
        ResourceState::ColorAttachment,    // write
        ResourceState::ShaderRead,         // read-only
    )];
    let result = opt.optimize(&input);
    assert_eq!(result.len(), 1, "write -> read barrier survives");
    assert_eq!(result[0].4, ResourceState::ColorAttachment);
    assert_eq!(result[0].5, ResourceState::ShaderRead);
}

#[test]
fn read_read_preserves_write_to_write() {
    let opt = BarrierOptimizer::new();
    let input: Vec<BarrierTuple> = vec![
        // ColorAttachment -> ShaderReadWrite (write-to-write, should survive)
        (
            PassIndex(0),
            PassIndex(1),
            ResourceHandle(1),
            EdgeType::RAW,
            ResourceState::ColorAttachment,
            ResourceState::ShaderReadWrite,
        ),
        // TransferDst -> ColorAttachment (write-to-write, should survive)
        (
            PassIndex(0),
            PassIndex(1),
            ResourceHandle(2),
            EdgeType::RAW,
            ResourceState::TransferDst,
            ResourceState::ColorAttachment,
        ),
    ];
    let result = opt.optimize(&input);
    assert_eq!(result.len(), 2, "write-to-write barriers survive");
}

#[test]
fn read_read_does_not_eliminate_transfer_dst_to_shader_read() {
    // TransferDst is a write state -- not read-only.
    let opt = BarrierOptimizer::new();
    let input: Vec<BarrierTuple> = vec![(
        PassIndex(0),
        PassIndex(1),
        ResourceHandle(1),
        EdgeType::RAW,
        ResourceState::TransferDst,
        ResourceState::ShaderRead,
    )];
    let result = opt.optimize(&input);
    assert_eq!(
        result.len(),
        1,
        "TransferDst (write) -> ShaderRead (read) is NOT a read-read pair",
    );
}

#[test]
fn read_read_does_not_eliminate_uninitialized_to_shader_read() {
    // Uninitialized is not read-only.
    let opt = BarrierOptimizer::new();
    let input: Vec<BarrierTuple> = vec![(
        PassIndex(0),
        PassIndex(1),
        ResourceHandle(1),
        EdgeType::RAW,
        ResourceState::Uninitialized,
        ResourceState::ShaderRead,
    )];
    let result = opt.optimize(&input);
    assert_eq!(result.len(), 1, "Uninitialized -> ShaderRead survives");
}

#[test]
fn read_read_mixed_batch_preserves_write_eliminates_read_only() {
    let opt = BarrierOptimizer::new();
    let input: Vec<BarrierTuple> = vec![
        // Read-read (eliminated)
        (
            PassIndex(0),
            PassIndex(1),
            ResourceHandle(1),
            EdgeType::RAW,
            ResourceState::ShaderRead,
            ResourceState::DepthStencilReadOnly,
        ),
        // Write-read (survives)
        (
            PassIndex(0),
            PassIndex(1),
            ResourceHandle(2),
            EdgeType::RAW,
            ResourceState::ColorAttachment,
            ResourceState::ShaderRead,
        ),
        // Read-write (survives)
        (
            PassIndex(1),
            PassIndex(2),
            ResourceHandle(3),
            EdgeType::RAW,
            ResourceState::ShaderRead,
            ResourceState::ShaderReadWrite,
        ),
        // Write-write (survives)
        (
            PassIndex(1),
            PassIndex(2),
            ResourceHandle(4),
            EdgeType::RAW,
            ResourceState::ShaderReadWrite,
            ResourceState::ColorAttachment,
        ),
    ];
    let result = opt.optimize(&input);
    assert_eq!(result.len(), 3, "3 write-involving barriers survive, 1 read-read eliminated");
}

// =========================================================================
// SECTION 4 -- Deduplication (Rule 3)
// =========================================================================

#[test]
fn dedup_exact_duplicate_collapsed() {
    let opt = BarrierOptimizer::new();
    let input: Vec<BarrierTuple> = vec![
        (
            PassIndex(0),
            PassIndex(1),
            ResourceHandle(1),
            EdgeType::RAW,
            ResourceState::ColorAttachment,
            ResourceState::ShaderRead,
        ),
        // Exact duplicate of the above
        (
            PassIndex(0),
            PassIndex(1),
            ResourceHandle(1),
            EdgeType::RAW,
            ResourceState::ColorAttachment,
            ResourceState::ShaderRead,
        ),
    ];
    let result = opt.optimize(&input);
    assert_eq!(result.len(), 1, "duplicate barrier collapsed to one");
    assert_eq!(result[0].2, ResourceHandle(1));
}

#[test]
fn dedup_same_from_to_resource_different_state_keeps_first() {
    // Duplicate detected by (from, to, resource) triple -- only the first
    // occurrence is preserved even if the state pairs differ.
    let opt = BarrierOptimizer::new();
    let input: Vec<BarrierTuple> = vec![
        // First occurrence -- has differing state
        (
            PassIndex(0),
            PassIndex(1),
            ResourceHandle(1),
            EdgeType::RAW,
            ResourceState::ColorAttachment,
            ResourceState::ShaderRead,
        ),
        // Second occurrence -- same (from, to, resource) but different states.
        // This would NOT be eliminated by same-state or read-read rules, but
        // IS eliminated by dedup.
        (
            PassIndex(0),
            PassIndex(1),
            ResourceHandle(1),
            EdgeType::RAW,
            ResourceState::ShaderRead,
            ResourceState::ShaderReadWrite,
        ),
    ];
    let result = opt.optimize(&input);
    assert_eq!(result.len(), 1, "dedup by triple keeps first occurrence only");
    // The first occurrence's state pair is preserved.
    assert_eq!(result[0].4, ResourceState::ColorAttachment);
    assert_eq!(result[0].5, ResourceState::ShaderRead);
}

#[test]
fn dedup_preserves_distinct_resources_at_same_boundary() {
    let opt = BarrierOptimizer::new();
    let input: Vec<BarrierTuple> = vec![
        (
            PassIndex(0),
            PassIndex(1),
            ResourceHandle(1),
            EdgeType::RAW,
            ResourceState::ColorAttachment,
            ResourceState::ShaderRead,
        ),
        (
            PassIndex(0),
            PassIndex(1),
            ResourceHandle(2),
            EdgeType::RAW,
            ResourceState::ColorAttachment,
            ResourceState::ShaderRead,
        ),
        (
            PassIndex(0),
            PassIndex(1),
            ResourceHandle(3),
            EdgeType::RAW,
            ResourceState::ColorAttachment,
            ResourceState::ShaderRead,
        ),
    ];
    let result = opt.optimize(&input);
    assert_eq!(
        result.len(),
        3,
        "three distinct resources at same boundary -> all three survive",
    );
    let handles: Vec<ResourceHandle> = result.iter().map(|e| e.2).collect();
    assert!(handles.contains(&ResourceHandle(1)));
    assert!(handles.contains(&ResourceHandle(2)));
    assert!(handles.contains(&ResourceHandle(3)));
}

#[test]
fn dedup_preserves_same_resource_at_distinct_boundaries() {
    let opt = BarrierOptimizer::new();
    let input: Vec<BarrierTuple> = vec![
        // P0 -> P1
        (
            PassIndex(0),
            PassIndex(1),
            ResourceHandle(1),
            EdgeType::RAW,
            ResourceState::Uninitialized,
            ResourceState::ShaderRead,
        ),
        // P1 -> P2 (same resource, different boundary)
        (
            PassIndex(1),
            PassIndex(2),
            ResourceHandle(1),
            EdgeType::RAW,
            ResourceState::ShaderRead,
            ResourceState::ColorAttachment,
        ),
        // P2 -> P3 (same resource, yet another boundary)
        (
            PassIndex(2),
            PassIndex(3),
            ResourceHandle(1),
            EdgeType::RAW,
            ResourceState::ColorAttachment,
            ResourceState::ShaderRead,
        ),
    ];
    let result = opt.optimize(&input);
    assert_eq!(
        result.len(),
        3,
        "same resource at three distinct boundaries -> all three survive",
    );
    // Verify boundaries are distinct.
    let boundaries: Vec<(PassIndex, PassIndex)> = result.iter().map(|e| (e.0, e.1)).collect();
    assert!(boundaries.contains(&(PassIndex(0), PassIndex(1))));
    assert!(boundaries.contains(&(PassIndex(1), PassIndex(2))));
    assert!(boundaries.contains(&(PassIndex(2), PassIndex(3))));
}

#[test]
fn dedup_mixed_with_distinct_and_duplicate() {
    let opt = BarrierOptimizer::new();
    let input: Vec<BarrierTuple> = vec![
        // Distinct resources at P0->P1 (all survive)
        (
            PassIndex(0),
            PassIndex(1),
            ResourceHandle(1),
            EdgeType::RAW,
            ResourceState::ColorAttachment,
            ResourceState::ShaderRead,
        ),
        (
            PassIndex(0),
            PassIndex(1),
            ResourceHandle(2),
            EdgeType::RAW,
            ResourceState::ColorAttachment,
            ResourceState::ShaderRead,
        ),
        // Duplicate of ResourceHandle(1) at P0->P1 (collapsed)
        (
            PassIndex(0),
            PassIndex(1),
            ResourceHandle(1),
            EdgeType::RAW,
            ResourceState::ColorAttachment,
            ResourceState::ShaderRead,
        ),
        // Same resource at different boundary (P1->P2, survives)
        (
            PassIndex(1),
            PassIndex(2),
            ResourceHandle(1),
            EdgeType::RAW,
            ResourceState::ShaderRead,
            ResourceState::ColorAttachment,
        ),
    ];
    let result = opt.optimize(&input);
    assert_eq!(result.len(), 3, "three unique (from, to, resource) triples survive");
}

// =========================================================================
// SECTION 5 -- Combined rule interaction
// =========================================================================

#[test]
fn all_three_rules_on_same_input() {
    let opt = BarrierOptimizer::new();
    let input: Vec<BarrierTuple> = vec![
        // 1. Same-state (eliminated by Rule 1)
        (
            PassIndex(0),
            PassIndex(1),
            ResourceHandle(1),
            EdgeType::RAW,
            ResourceState::ShaderRead,
            ResourceState::ShaderRead,
        ),
        // 2. Read-read (eliminated by Rule 2)
        (
            PassIndex(0),
            PassIndex(1),
            ResourceHandle(2),
            EdgeType::RAW,
            ResourceState::VertexBuffer,
            ResourceState::ShaderRead,
        ),
        // 3. Valid barrier (survives)
        (
            PassIndex(0),
            PassIndex(1),
            ResourceHandle(3),
            EdgeType::RAW,
            ResourceState::Uninitialized,
            ResourceState::ShaderRead,
        ),
        // 4. Duplicate of #3 (eliminated by Rule 3)
        (
            PassIndex(0),
            PassIndex(1),
            ResourceHandle(3),
            EdgeType::RAW,
            ResourceState::Uninitialized,
            ResourceState::ShaderRead,
        ),
        // 5. Another valid barrier at different boundary (survives)
        (
            PassIndex(1),
            PassIndex(2),
            ResourceHandle(4),
            EdgeType::RAW,
            ResourceState::ColorAttachment,
            ResourceState::ShaderRead,
        ),
        // 6. Same-state of #5 but different resource id (eliminated by Rule 1)
        (
            PassIndex(1),
            PassIndex(2),
            ResourceHandle(5),
            EdgeType::RAW,
            ResourceState::ColorAttachment,
            ResourceState::ColorAttachment,
        ),
    ];

    let result = opt.optimize(&input);
    // Expected survivors: #3 (P0->P1, R3, Uninit->SR) and #5 (P1->P2, R4, CA->SR)
    assert_eq!(
        result.len(),
        2,
        "all three rules active: 6 input -> 2 survivors",
    );

    // Survivor 1: P0->P1, R3
    assert_eq!(result[0].0, PassIndex(0));
    assert_eq!(result[0].1, PassIndex(1));
    assert_eq!(result[0].2, ResourceHandle(3));
    assert_eq!(result[0].4, ResourceState::Uninitialized);
    assert_eq!(result[0].5, ResourceState::ShaderRead);

    // Survivor 2: P1->P2, R4
    assert_eq!(result[1].0, PassIndex(1));
    assert_eq!(result[1].1, PassIndex(2));
    assert_eq!(result[1].2, ResourceHandle(4));
    assert_eq!(result[1].4, ResourceState::ColorAttachment);
    assert_eq!(result[1].5, ResourceState::ShaderRead);
}

// =========================================================================
// SECTION 6 -- Edge cases
// =========================================================================

#[test]
fn empty_input_returns_empty() {
    let opt = BarrierOptimizer::new();
    let result = opt.optimize(&vec![]);
    assert!(result.is_empty(), "empty input -> empty output");
}

#[test]
fn output_preserves_input_ordering() {
    let opt = BarrierOptimizer::new();
    // Seven barriers where none are redundant. Order must be preserved.
    let input: Vec<BarrierTuple> = vec![
        (
            PassIndex(0),
            PassIndex(1),
            ResourceHandle(1),
            EdgeType::RAW,
            ResourceState::Uninitialized,
            ResourceState::ShaderRead,
        ),
        (
            PassIndex(1),
            PassIndex(2),
            ResourceHandle(1),
            EdgeType::RAW,
            ResourceState::ShaderRead,
            ResourceState::ColorAttachment,
        ),
        (
            PassIndex(2),
            PassIndex(3),
            ResourceHandle(1),
            EdgeType::RAW,
            ResourceState::ColorAttachment,
            ResourceState::ShaderRead,
        ),
        (
            PassIndex(3),
            PassIndex(4),
            ResourceHandle(1),
            EdgeType::RAW,
            ResourceState::ShaderRead,
            ResourceState::TransferDst,
        ),
        (
            PassIndex(4),
            PassIndex(5),
            ResourceHandle(1),
            EdgeType::RAW,
            ResourceState::TransferDst,
            ResourceState::ShaderRead,
        ),
        (
            PassIndex(5),
            PassIndex(6),
            ResourceHandle(1),
            EdgeType::RAW,
            ResourceState::ShaderRead,
            ResourceState::Present,
        ),
    ];
    let result = opt.optimize(&input.clone());
    assert_eq!(
        result.len(),
        input.len(),
        "no redundant barriers -> all survive in order",
    );
    for (i, (expected, actual)) in input.iter().zip(result.iter()).enumerate() {
        assert_eq!(
            (actual.0, actual.1, actual.2, actual.3, actual.4),
            (expected.0, expected.1, expected.2, expected.3, expected.4),
            "entry {} order preserved",
            i,
        );
    }
}

#[test]
fn complex_stress_test() {
    // A large batch mixing all categories of redundant and non-redundant
    // barriers across multiple boundaries and resources.
    let opt = BarrierOptimizer::new();
    let input: Vec<BarrierTuple> = vec![
        // Boundary P0->P1
        (PassIndex(0), PassIndex(1), ResourceHandle(1), EdgeType::RAW, ResourceState::Uninitialized, ResourceState::ShaderRead),       // valid
        (PassIndex(0), PassIndex(1), ResourceHandle(2), EdgeType::RAW, ResourceState::Uninitialized, ResourceState::ShaderRead),       // valid
        (PassIndex(0), PassIndex(1), ResourceHandle(1), EdgeType::RAW, ResourceState::Uninitialized, ResourceState::ShaderRead),       // dup of R1
        (PassIndex(0), PassIndex(1), ResourceHandle(3), EdgeType::RAW, ResourceState::ShaderRead, ResourceState::ShaderRead),         // same-state
        (PassIndex(0), PassIndex(1), ResourceHandle(4), EdgeType::RAW, ResourceState::ShaderRead, ResourceState::DepthStencilReadOnly), // read-read
        (PassIndex(0), PassIndex(1), ResourceHandle(5), EdgeType::RAW, ResourceState::ColorAttachment, ResourceState::ShaderRead),     // valid

        // Boundary P1->P2
        (PassIndex(1), PassIndex(2), ResourceHandle(1), EdgeType::RAW, ResourceState::ShaderRead, ResourceState::ColorAttachment),     // valid
        (PassIndex(1), PassIndex(2), ResourceHandle(6), EdgeType::RAW, ResourceState::TransferDst, ResourceState::ShaderRead),         // valid
        (PassIndex(1), PassIndex(2), ResourceHandle(6), EdgeType::RAW, ResourceState::TransferDst, ResourceState::ShaderRead),         // dup of R6
        (PassIndex(1), PassIndex(2), ResourceHandle(7), EdgeType::RAW, ResourceState::VertexBuffer, ResourceState::IndexBuffer),       // read-read
        (PassIndex(1), PassIndex(2), ResourceHandle(8), EdgeType::RAW, ResourceState::ShaderReadWrite, ResourceState::ShaderReadWrite), // same-state
        (PassIndex(1), PassIndex(2), ResourceHandle(9), EdgeType::RAW, ResourceState::Uninitialized, ResourceState::TransferDst),       // valid

        // Boundary P2->P3
        (PassIndex(2), PassIndex(3), ResourceHandle(1), EdgeType::RAW, ResourceState::ColorAttachment, ResourceState::ShaderRead),     // valid
        (PassIndex(2), PassIndex(3), ResourceHandle(1), EdgeType::RAW, ResourceState::ColorAttachment, ResourceState::ShaderRead),     // dup of above
        (PassIndex(2), PassIndex(3), ResourceHandle(9), EdgeType::RAW, ResourceState::TransferDst, ResourceState::ShaderRead),         // valid
        (PassIndex(2), PassIndex(3), ResourceHandle(10), EdgeType::RAW, ResourceState::DepthStencilReadOnly, ResourceState::ShaderRead), // read-read
    ];

    let result = opt.optimize(&input);

    // Count expected survivors:
    // P0->P1: R1, R2, R5 = 3
    // P1->P2: R1, R6, R9 = 3
    // P2->P3: R1, R9    = 2
    // Total: 8
    assert_eq!(result.len(), 8, "complex stress test: 16 in -> 8 survivors");

    // Verify no same-state barriers survive.
    for (_, _, _, _edge_type, before, after) in &result {
        assert_ne!(before, after, "no same-state barrier survives");
    }

    // Verify no read-read barriers survive.
    for (_, _, _, _edge_type, before, after) in &result {
        let before_read_only = matches!(
            before,
            ResourceState::VertexBuffer
                | ResourceState::IndexBuffer
                | ResourceState::IndirectArgument
                | ResourceState::DepthStencilReadOnly
                | ResourceState::ShaderRead
                | ResourceState::TransferSrc
        );
        let after_read_only = matches!(
            after,
            ResourceState::VertexBuffer
                | ResourceState::IndexBuffer
                | ResourceState::IndirectArgument
                | ResourceState::DepthStencilReadOnly
                | ResourceState::ShaderRead
                | ResourceState::TransferSrc
        );
        assert!(
            !(before_read_only && after_read_only),
            "no read-read barrier survives",
        );
    }

    // Verify no duplicate (from, to, resource) triples.
    let mut seen_triples = std::collections::HashSet::new();
    for (from, to, resource, _edge_type, _, _) in &result {
        assert!(
            seen_triples.insert((*from, *to, *resource)),
            "no duplicate (from, to, resource) triple in output",
        );
    }

    // Verify each boundary has the right resources.
    let p0p1: Vec<ResourceHandle> = result.iter()
        .filter(|e| e.0 == PassIndex(0) && e.1 == PassIndex(1))
        .map(|e| e.2)
        .collect();
    assert_eq!(p0p1.len(), 3, "P0->P1 has 3 unique resources");
    assert!(p0p1.contains(&ResourceHandle(1)));
    assert!(p0p1.contains(&ResourceHandle(2)));
    assert!(p0p1.contains(&ResourceHandle(5)));

    let p1p2: Vec<ResourceHandle> = result.iter()
        .filter(|e| e.0 == PassIndex(1) && e.1 == PassIndex(2))
        .map(|e| e.2)
        .collect();
    assert_eq!(p1p2.len(), 3, "P1->P2 has 3 unique resources");
    assert!(p1p2.contains(&ResourceHandle(1)));
    assert!(p1p2.contains(&ResourceHandle(6)));
    assert!(p1p2.contains(&ResourceHandle(9)));

    let p2p3: Vec<ResourceHandle> = result.iter()
        .filter(|e| e.0 == PassIndex(2) && e.1 == PassIndex(3))
        .map(|e| e.2)
        .collect();
    assert_eq!(p2p3.len(), 2, "P2->P3 has 2 unique resources");
    assert!(p2p3.contains(&ResourceHandle(1)));
    assert!(p2p3.contains(&ResourceHandle(9)));
}

// =========================================================================
// SECTION 7 -- Dedup preserves first-occurrence state (corner cases)
// =========================================================================

#[test]
fn dedup_with_same_state_first_then_valid_state_pair() {
    // First occurrence has before==after (would be same-state). But because
    // Rule 3 checks the triple BEFORE Rule 1, this first occurrence should
    // still be removed by Rule 1.
    //
    // More precisely: Rules 1+2 are checked per-entry BEFORE Rule 3 in the
    // implementation. So a barrier that is same-state or read-read is skipped
    // on its own merits, and its triple is never inserted into `seen`.
    // A subsequent valid entry with the same triple IS inserted and kept.
    let opt = BarrierOptimizer::new();
    let input: Vec<BarrierTuple> = vec![
        // First: same-state (eliminated by Rule 1, NOT inserted into seen)
        (
            PassIndex(0),
            PassIndex(1),
            ResourceHandle(1),
            EdgeType::RAW,
            ResourceState::ShaderRead,
            ResourceState::ShaderRead,
        ),
        // Second: valid pair, same triple (inserted and kept because triple
        // was never in `seen` since the first entry was skipped by Rule 1).
        (
            PassIndex(0),
            PassIndex(1),
            ResourceHandle(1),
            EdgeType::RAW,
            ResourceState::ColorAttachment,
            ResourceState::ShaderRead,
        ),
    ];
    let result = opt.optimize(&input);
    assert_eq!(
        result.len(),
        1,
        "same-state first entry is skipped, valid second entry with same triple survives",
    );
    assert_eq!(result[0].4, ResourceState::ColorAttachment);
    assert_eq!(result[0].5, ResourceState::ShaderRead);
}

#[test]
fn dedup_with_read_read_first_then_valid_state_pair() {
    let opt = BarrierOptimizer::new();
    let input: Vec<BarrierTuple> = vec![
        // First: read-read (eliminated by Rule 2)
        (
            PassIndex(0),
            PassIndex(1),
            ResourceHandle(1),
            EdgeType::RAW,
            ResourceState::ShaderRead,
            ResourceState::DepthStencilReadOnly,
        ),
        // Second: valid, same triple
        (
            PassIndex(0),
            PassIndex(1),
            ResourceHandle(1),
            EdgeType::RAW,
            ResourceState::ColorAttachment,
            ResourceState::ShaderRead,
        ),
    ];
    let result = opt.optimize(&input);
    assert_eq!(
        result.len(),
        1,
        "read-read first entry is skipped, valid second entry survives",
    );
    assert_eq!(result[0].4, ResourceState::ColorAttachment);
    assert_eq!(result[0].5, ResourceState::ShaderRead);
}

// =========================================================================
// SECTION 8 -- All read-only states are individually recognized
// =========================================================================

#[test]
fn read_only_vertex_buffer_is_read_only() {
    // VertexBuffer -> ShaderRead is read-read, should be eliminated.
    let opt = BarrierOptimizer::new();
    let input: Vec<BarrierTuple> = vec![(
        PassIndex(0),
        PassIndex(1),
        ResourceHandle(1),
        EdgeType::RAW,
        ResourceState::VertexBuffer,
        ResourceState::ShaderRead,
    )];
    assert!(opt.optimize(&input).is_empty(), "VertexBuffer is read-only");
}

#[test]
fn read_only_index_buffer_is_read_only() {
    let opt = BarrierOptimizer::new();
    let input: Vec<BarrierTuple> = vec![(
        PassIndex(0),
        PassIndex(1),
        ResourceHandle(1),
        EdgeType::RAW,
        ResourceState::IndexBuffer,
        ResourceState::ShaderRead,
    )];
    assert!(opt.optimize(&input).is_empty(), "IndexBuffer is read-only");
}

#[test]
fn read_only_indirect_argument_is_read_only() {
    let opt = BarrierOptimizer::new();
    let input: Vec<BarrierTuple> = vec![(
        PassIndex(0),
        PassIndex(1),
        ResourceHandle(1),
        EdgeType::RAW,
        ResourceState::IndirectArgument,
        ResourceState::ShaderRead,
    )];
    assert!(opt.optimize(&input).is_empty(), "IndirectArgument is read-only");
}

#[test]
fn read_only_depth_stencil_read_only_is_read_only() {
    let opt = BarrierOptimizer::new();
    let input: Vec<BarrierTuple> = vec![(
        PassIndex(0),
        PassIndex(1),
        ResourceHandle(1),
        EdgeType::RAW,
        ResourceState::DepthStencilReadOnly,
        ResourceState::ShaderRead,
    )];
    assert!(opt.optimize(&input).is_empty(), "DepthStencilReadOnly is read-only");
}

#[test]
fn read_only_shader_read_is_read_only() {
    let opt = BarrierOptimizer::new();
    let input: Vec<BarrierTuple> = vec![(
        PassIndex(0),
        PassIndex(1),
        ResourceHandle(1),
        EdgeType::RAW,
        ResourceState::ShaderRead,
        ResourceState::VertexBuffer,
    )];
    assert!(opt.optimize(&input).is_empty(), "ShaderRead is read-only");
}

#[test]
fn read_only_transfer_src_is_read_only() {
    let opt = BarrierOptimizer::new();
    let input: Vec<BarrierTuple> = vec![(
        PassIndex(0),
        PassIndex(1),
        ResourceHandle(1),
        EdgeType::RAW,
        ResourceState::TransferSrc,
        ResourceState::ShaderRead,
    )];
    assert!(opt.optimize(&input).is_empty(), "TransferSrc is read-only");
}

// =========================================================================
// SECTION 9 -- Write states are NOT treated as read-only
// =========================================================================

#[test]
fn write_states_not_read_only_color_attachment() {
    let opt = BarrierOptimizer::new();
    let input: Vec<BarrierTuple> = vec![
        (
            PassIndex(0),
            PassIndex(1),
            ResourceHandle(1),
            EdgeType::RAW,
            ResourceState::ColorAttachment,
            ResourceState::ShaderRead,
        ),
        (
            PassIndex(0),
            PassIndex(1),
            ResourceHandle(2),
            EdgeType::RAW,
            ResourceState::ShaderRead,
            ResourceState::ColorAttachment,
        ),
    ];
    let result = opt.optimize(&input);
    assert_eq!(result.len(), 2, "ColorAttachment barriers survive");
}

#[test]
fn write_states_not_read_only_depth_stencil_attachment() {
    let opt = BarrierOptimizer::new();
    let input: Vec<BarrierTuple> = vec![(
        PassIndex(0),
        PassIndex(1),
        ResourceHandle(1),
        EdgeType::RAW,
        ResourceState::DepthStencilAttachment,
        ResourceState::ShaderRead,
    )];
    let result = opt.optimize(&input);
    assert_eq!(result.len(), 1, "DepthStencilAttachment barrier survives");
}

#[test]
fn write_states_not_read_only_shader_read_write() {
    let opt = BarrierOptimizer::new();
    let input: Vec<BarrierTuple> = vec![(
        PassIndex(0),
        PassIndex(1),
        ResourceHandle(1),
        EdgeType::RAW,
        ResourceState::ShaderReadWrite,
        ResourceState::ShaderRead,
    )];
    let result = opt.optimize(&input);
    assert_eq!(result.len(), 1, "ShaderReadWrite barrier survives");
}

#[test]
fn write_states_not_read_only_transfer_dst() {
    let opt = BarrierOptimizer::new();
    let input: Vec<BarrierTuple> = vec![(
        PassIndex(0),
        PassIndex(1),
        ResourceHandle(1),
        EdgeType::RAW,
        ResourceState::TransferDst,
        ResourceState::ShaderRead,
    )];
    let result = opt.optimize(&input);
    assert_eq!(result.len(), 1, "TransferDst barrier survives");
}

#[test]
fn write_states_not_read_only_present() {
    let opt = BarrierOptimizer::new();
    let input: Vec<BarrierTuple> = vec![(
        PassIndex(0),
        PassIndex(1),
        ResourceHandle(1),
        EdgeType::RAW,
        ResourceState::Present,
        ResourceState::ShaderRead,
    )];
    let result = opt.optimize(&input);
    assert_eq!(result.len(), 1, "Present barrier survives");
}

#[test]
fn write_states_not_read_only_acceleration_structure() {
    let opt = BarrierOptimizer::new();
    let input: Vec<BarrierTuple> = vec![(
        PassIndex(0),
        PassIndex(1),
        ResourceHandle(1),
        EdgeType::RAW,
        ResourceState::AccelerationStructure,
        ResourceState::ShaderRead,
    )];
    let result = opt.optimize(&input);
    assert_eq!(result.len(), 1, "AccelerationStructure barrier survives");
}

#[test]
fn write_states_not_read_only_uninitialized() {
    let opt = BarrierOptimizer::new();
    let input: Vec<BarrierTuple> = vec![(
        PassIndex(0),
        PassIndex(1),
        ResourceHandle(1),
        EdgeType::RAW,
        ResourceState::Uninitialized,
        ResourceState::ShaderRead,
    )];
    let result = opt.optimize(&input);
    assert_eq!(result.len(), 1, "Uninitialized barrier survives");
}

// =========================================================================
// SECTION 10 -- BarrierOptimizer is idempotent
// =========================================================================

#[test]
fn optimizer_is_idempotent() {
    // Running the optimizer twice on the same input should produce the same
    // result as running it once (since no redundant entries should remain).
    let opt = BarrierOptimizer::new();
    let input: Vec<BarrierTuple> = vec![
        // Mix of valid and redundant
        (
            PassIndex(0),
            PassIndex(1),
            ResourceHandle(1),
            EdgeType::RAW,
            ResourceState::Uninitialized,
            ResourceState::ShaderRead,
        ),
        (
            PassIndex(0),
            PassIndex(1),
            ResourceHandle(1),
            EdgeType::RAW,
            ResourceState::Uninitialized,
            ResourceState::ShaderRead,
        ), // duplicate
        (
            PassIndex(0),
            PassIndex(1),
            ResourceHandle(2),
            EdgeType::RAW,
            ResourceState::ColorAttachment,
            ResourceState::ColorAttachment,
        ), // same-state
        (
            PassIndex(0),
            PassIndex(1),
            ResourceHandle(3),
            EdgeType::RAW,
            ResourceState::ShaderRead,
            ResourceState::DepthStencilReadOnly,
        ), // read-read
        (
            PassIndex(1),
            PassIndex(2),
            ResourceHandle(4),
            EdgeType::RAW,
            ResourceState::ColorAttachment,
            ResourceState::ShaderRead,
        ), // valid
    ];

    let once = opt.optimize(&input.clone());
    let twice = opt.optimize(&opt.optimize(&input));

    assert_eq!(
        once, twice,
        "optimizer is idempotent: second pass produces no change",
    );
}
