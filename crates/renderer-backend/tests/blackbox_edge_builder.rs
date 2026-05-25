// Blackbox contract tests for T-FG-2.2 EdgeBuilder.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::frame_graph::*` -- no internal fields,
// no private methods, no implementation details.
//
// Contract:
//   EdgeBuilder builds Vec<IrEdge> programmatically.
//
//   Methods:
//     - add_edge(from, to, resource, edge_type) -- appends a fully-specified
//       edge with the given PassIndex, ResourceHandle, and EdgeType.
//     - add_raw(from, to, resource)             -- convenience for RAW edges.
//     - add_war(from, to, resource)             -- convenience for WAR edges.
//     - add_waw(from, to, resource)             -- convenience for WAW edges.
//     - build(&mut self) -> Vec<IrEdge>         -- drains accumulated edges
//       and returns them. The builder can be reused after build().
//     - chain(passes, resource)                 -- convenience: connects a
//       sequence of passes with RAW edges over the given resource. For N
//       passes, adds N-1 edges: passes[0]->passes[1], passes[1]->passes[2],
//       etc.
//
//   EdgeBuilder derives Clone, Debug, Default. All add_* methods return
//   &mut Self for method chaining.
//
// Coverage:
//   1.  EdgeBuilder::new() creates an empty builder
//   2.  EdgeBuilder::default() creates an empty builder
//   3.  add_edge with each EdgeType variant (RAW, WAR, WAW)
//   4.  add_edge preserves from, to, resource, edge_type on the IrEdge
//   5.  add_raw produces EdgeType::RAW
//   6.  add_war produces EdgeType::WAR
//   7.  add_waw produces EdgeType::WAW
//   8.  Method chaining -- add_raw().add_war().add_waw() returns &mut Self
//   9.  build() returns accumulated edges
//  10.  build() clears internal state (subsequent build returns empty)
//  11.  Builder reuse after build()
//  12.  Insertion order preserved in build() output
//  13.  Multiple edges with same type and different passes
//  14.  Multiple edges over different resources
//  15.  chain() with 2 passes produces a single RAW edge
//  16.  chain() with 3 passes produces two RAW edges
//  17.  chain() with 4 passes produces three RAW edges
//  18.  chain() with single pass produces no edges
//  19.  chain() preserves resource handle on all edges
//  20.  chain() preserves EdgeType::RAW on all edges
//  21.  EdgeBuilder derives Clone (clone produces equal builder)
//  22.  EdgeBuilder derives Debug (non-empty output)
//  23.  EdgeBuilder derives Default (same as new())
//  24.  EdgeBuilder Display / Debug round-trip for edges
//  25.  Mixed usage: add_edge + add_raw + chain + build
//  26.  Large batch: many edges all accumulated correctly
//  27.  chain() with PassIndex values that are not sequential

use renderer_backend::frame_graph::{EdgeBuilder, EdgeType, IrEdge, PassIndex, ResourceHandle};

// =============================================================================
// SECTION 1 -- Construction
// =============================================================================

#[test]
fn edge_builder_new_creates_empty() {
    let mut builder = EdgeBuilder::new();
    let edges = builder.build();
    assert!(edges.is_empty(), "EdgeBuilder::new() starts with no edges",);
}

#[test]
fn edge_builder_default_creates_empty() {
    let mut builder: EdgeBuilder = Default::default();
    let edges = builder.build();
    assert!(
        edges.is_empty(),
        "EdgeBuilder::default() starts with no edges",
    );
}

#[test]
fn edge_builder_new_and_default_are_equivalent() {
    let from_new = EdgeBuilder::new().build();
    let from_default = EdgeBuilder::default().build();
    assert_eq!(
        from_new, from_default,
        "new() and default() produce equivalent builders",
    );
}

// =============================================================================
// SECTION 2 -- add_edge with all EdgeType variants
// =============================================================================

#[test]
fn add_edge_raw_variant() {
    let mut builder = EdgeBuilder::new();
    builder.add_edge(PassIndex(0), PassIndex(1), ResourceHandle(1), EdgeType::RAW);
    let edges = builder.build();
    assert_eq!(edges.len(), 1, "one edge added");
    assert_eq!(edges[0].edge_type, EdgeType::RAW, "edge type is RAW",);
}

#[test]
fn add_edge_war_variant() {
    let mut builder = EdgeBuilder::new();
    builder.add_edge(PassIndex(2), PassIndex(3), ResourceHandle(5), EdgeType::WAR);
    let edges = builder.build();
    assert_eq!(edges.len(), 1, "one edge added");
    assert_eq!(edges[0].edge_type, EdgeType::WAR, "edge type is WAR",);
}

#[test]
fn add_edge_waw_variant() {
    let mut builder = EdgeBuilder::new();
    builder.add_edge(PassIndex(7), PassIndex(8), ResourceHandle(9), EdgeType::WAW);
    let edges = builder.build();
    assert_eq!(edges.len(), 1, "one edge added");
    assert_eq!(edges[0].edge_type, EdgeType::WAW, "edge type is WAW",);
}

// =============================================================================
// SECTION 3 -- add_edge preserves all fields
// =============================================================================

#[test]
fn add_edge_preserves_from_field() {
    let mut builder = EdgeBuilder::new();
    builder.add_edge(
        PassIndex(42),
        PassIndex(99),
        ResourceHandle(7),
        EdgeType::RAW,
    );
    let edges = builder.build();
    assert_eq!(edges[0].from, PassIndex(42), "from pass index is preserved",);
}

#[test]
fn add_edge_preserves_to_field() {
    let mut builder = EdgeBuilder::new();
    builder.add_edge(
        PassIndex(0),
        PassIndex(255),
        ResourceHandle(3),
        EdgeType::WAR,
    );
    let edges = builder.build();
    assert_eq!(edges[0].to, PassIndex(255), "to pass index is preserved",);
}

#[test]
fn add_edge_preserves_resource_handle() {
    let mut builder = EdgeBuilder::new();
    builder.add_edge(
        PassIndex(0),
        PassIndex(1),
        ResourceHandle(65535),
        EdgeType::WAW,
    );
    let edges = builder.build();
    assert_eq!(
        edges[0].resource,
        ResourceHandle(65535),
        "resource handle is preserved",
    );
}

#[test]
fn add_edge_preserves_all_fields() {
    let mut builder = EdgeBuilder::new();
    builder.add_edge(
        PassIndex(10),
        PassIndex(20),
        ResourceHandle(30),
        EdgeType::RAW,
    );
    let edges = builder.build();

    assert_eq!(edges[0].from, PassIndex(10), "from matches");
    assert_eq!(edges[0].to, PassIndex(20), "to matches");
    assert_eq!(edges[0].resource, ResourceHandle(30), "resource matches");
    assert_eq!(edges[0].edge_type, EdgeType::RAW, "edge_type matches");
}

// =============================================================================
// SECTION 4 -- Convenience methods (add_raw, add_war, add_waw)
// =============================================================================

#[test]
fn add_raw_produces_raw_edge() {
    let mut builder = EdgeBuilder::new();
    builder.add_raw(PassIndex(0), PassIndex(1), ResourceHandle(1));
    let edges = builder.build();
    assert_eq!(edges.len(), 1, "add_raw adds one edge");
    assert_eq!(
        edges[0].edge_type,
        EdgeType::RAW,
        "add_raw produces EdgeType::RAW",
    );
}

#[test]
fn add_war_produces_war_edge() {
    let mut builder = EdgeBuilder::new();
    builder.add_war(PassIndex(5), PassIndex(6), ResourceHandle(2));
    let edges = builder.build();
    assert_eq!(edges.len(), 1, "add_war adds one edge");
    assert_eq!(
        edges[0].edge_type,
        EdgeType::WAR,
        "add_war produces EdgeType::WAR",
    );
}

#[test]
fn add_waw_produces_waw_edge() {
    let mut builder = EdgeBuilder::new();
    builder.add_waw(PassIndex(3), PassIndex(4), ResourceHandle(3));
    let edges = builder.build();
    assert_eq!(edges.len(), 1, "add_waw adds one edge");
    assert_eq!(
        edges[0].edge_type,
        EdgeType::WAW,
        "add_waw produces EdgeType::WAW",
    );
}

#[test]
fn convenience_methods_preserve_pass_indices() {
    let mut builder = EdgeBuilder::new();
    builder.add_raw(PassIndex(0), PassIndex(1), ResourceHandle(10));
    builder.add_war(PassIndex(2), PassIndex(3), ResourceHandle(20));
    builder.add_waw(PassIndex(4), PassIndex(5), ResourceHandle(30));
    let edges = builder.build();

    assert_eq!(edges[0].from, PassIndex(0));
    assert_eq!(edges[0].to, PassIndex(1));
    assert_eq!(edges[1].from, PassIndex(2));
    assert_eq!(edges[1].to, PassIndex(3));
    assert_eq!(edges[2].from, PassIndex(4));
    assert_eq!(edges[2].to, PassIndex(5));
}

// =============================================================================
// SECTION 5 -- Method chaining
// =============================================================================

#[test]
fn add_edge_returns_mut_self_for_chaining() {
    let mut builder = EdgeBuilder::new();
    // Chaining add_edge calls.
    builder
        .add_edge(PassIndex(0), PassIndex(1), ResourceHandle(1), EdgeType::RAW)
        .add_edge(PassIndex(2), PassIndex(3), ResourceHandle(2), EdgeType::WAR);
    let edges = builder.build();
    assert_eq!(edges.len(), 2, "chained add_edge calls accumulate");
}

#[test]
fn convenience_methods_chain_together() {
    let mut builder = EdgeBuilder::new();
    builder
        .add_raw(PassIndex(0), PassIndex(1), ResourceHandle(1))
        .add_war(PassIndex(2), PassIndex(3), ResourceHandle(2))
        .add_waw(PassIndex(4), PassIndex(5), ResourceHandle(3));
    let edges = builder.build();
    assert_eq!(edges.len(), 3, "chain of raw/war/waw produces three edges");
}

#[test]
fn mixed_chaining() {
    let mut builder = EdgeBuilder::new();
    builder
        .add_edge(PassIndex(0), PassIndex(1), ResourceHandle(1), EdgeType::RAW)
        .add_raw(PassIndex(2), PassIndex(3), ResourceHandle(2))
        .add_war(PassIndex(4), PassIndex(5), ResourceHandle(3))
        .add_waw(PassIndex(6), PassIndex(7), ResourceHandle(4));
    let edges = builder.build();
    assert_eq!(edges.len(), 4, "mixed chaining produces four edges");
}

// =============================================================================
// SECTION 6 -- build() drains edges
// =============================================================================

#[test]
fn build_returns_all_edges() {
    let mut builder = EdgeBuilder::new();
    builder.add_raw(PassIndex(0), PassIndex(1), ResourceHandle(1));
    builder.add_raw(PassIndex(1), PassIndex(2), ResourceHandle(1));
    builder.add_raw(PassIndex(2), PassIndex(3), ResourceHandle(1));
    let edges = builder.build();
    assert_eq!(edges.len(), 3, "build() returns all three edges");
}

#[test]
fn build_clears_internal_state() {
    let mut builder = EdgeBuilder::new();
    builder.add_raw(PassIndex(0), PassIndex(1), ResourceHandle(1));
    let _first = builder.build();
    let second = builder.build();
    assert!(
        second.is_empty(),
        "second build() returns empty vec after drain",
    );
}

#[test]
fn builder_reuse_after_build() {
    let mut builder = EdgeBuilder::new();
    builder.add_raw(PassIndex(0), PassIndex(1), ResourceHandle(1));
    let _first_batch = builder.build();

    // Reuse the same builder for a second batch.
    builder.add_war(PassIndex(2), PassIndex(3), ResourceHandle(2));
    let second_batch = builder.build();
    assert_eq!(second_batch.len(), 1, "builder works after reuse",);
    assert_eq!(
        second_batch[0].edge_type,
        EdgeType::WAR,
        "second batch has correct edge type",
    );
}

// =============================================================================
// SECTION 7 -- Insertion order
// =============================================================================

#[test]
fn build_preserves_insertion_order() {
    let mut builder = EdgeBuilder::new();
    builder.add_raw(PassIndex(0), PassIndex(1), ResourceHandle(1));
    builder.add_war(PassIndex(2), PassIndex(3), ResourceHandle(2));
    builder.add_waw(PassIndex(4), PassIndex(5), ResourceHandle(3));
    builder.add_raw(PassIndex(6), PassIndex(7), ResourceHandle(4));

    let edges = builder.build();
    assert_eq!(edges.len(), 4, "four edges in order");

    assert_eq!(edges[0].edge_type, EdgeType::RAW);
    assert_eq!(edges[0].from, PassIndex(0));
    assert_eq!(edges[0].to, PassIndex(1));

    assert_eq!(edges[1].edge_type, EdgeType::WAR);
    assert_eq!(edges[1].from, PassIndex(2));
    assert_eq!(edges[1].to, PassIndex(3));

    assert_eq!(edges[2].edge_type, EdgeType::WAW);
    assert_eq!(edges[2].from, PassIndex(4));
    assert_eq!(edges[2].to, PassIndex(5));

    assert_eq!(edges[3].edge_type, EdgeType::RAW);
    assert_eq!(edges[3].from, PassIndex(6));
    assert_eq!(edges[3].to, PassIndex(7));
}

// =============================================================================
// SECTION 8 -- Multiple edges over same and different resources
// =============================================================================

#[test]
fn multiple_edges_same_resource() {
    let mut builder = EdgeBuilder::new();
    let res = ResourceHandle(7);
    builder.add_raw(PassIndex(0), PassIndex(1), res);
    builder.add_raw(PassIndex(1), PassIndex(2), res);
    builder.add_raw(PassIndex(2), PassIndex(3), res);
    let edges = builder.build();

    assert_eq!(edges.len(), 3, "three edges over same resource");
    for (i, edge) in edges.iter().enumerate() {
        assert_eq!(edge.resource, res, "edge[{}] has correct resource", i,);
    }
}

#[test]
fn multiple_edges_different_resources() {
    let mut builder = EdgeBuilder::new();
    builder.add_raw(PassIndex(0), PassIndex(1), ResourceHandle(10));
    builder.add_raw(PassIndex(2), PassIndex(3), ResourceHandle(20));
    builder.add_raw(PassIndex(4), PassIndex(5), ResourceHandle(30));
    let edges = builder.build();

    assert_eq!(edges.len(), 3);
    assert_eq!(edges[0].resource, ResourceHandle(10));
    assert_eq!(edges[1].resource, ResourceHandle(20));
    assert_eq!(edges[2].resource, ResourceHandle(30));
}

// =============================================================================
// SECTION 9 -- chain() convenience method
// =============================================================================

#[test]
fn chain_two_passes_produces_single_raw_edge() {
    let mut builder = EdgeBuilder::new();
    builder.chain(&[PassIndex(0), PassIndex(1)], ResourceHandle(1));
    let edges = builder.build();
    assert_eq!(edges.len(), 1, "chain of 2 passes produces 1 edge",);
    assert_eq!(edges[0].from, PassIndex(0), "chain from is first pass",);
    assert_eq!(edges[0].to, PassIndex(1), "chain to is second pass",);
}

#[test]
fn chain_three_passes_produces_two_raw_edges() {
    let mut builder = EdgeBuilder::new();
    builder.chain(
        &[PassIndex(0), PassIndex(1), PassIndex(2)],
        ResourceHandle(5),
    );
    let edges = builder.build();
    assert_eq!(edges.len(), 2, "chain of 3 passes produces 2 edges",);

    assert_eq!(edges[0].from, PassIndex(0), "edge0 from = P0");
    assert_eq!(edges[0].to, PassIndex(1), "edge0 to = P1");
    assert_eq!(edges[1].from, PassIndex(1), "edge1 from = P1");
    assert_eq!(edges[1].to, PassIndex(2), "edge1 to = P2");
}

#[test]
fn chain_four_passes_produces_three_raw_edges() {
    let mut builder = EdgeBuilder::new();
    builder.chain(
        &[PassIndex(0), PassIndex(1), PassIndex(2), PassIndex(3)],
        ResourceHandle(10),
    );
    let edges = builder.build();
    assert_eq!(edges.len(), 3, "chain of 4 passes produces 3 edges",);

    assert_eq!(edges[0].from, PassIndex(0));
    assert_eq!(edges[0].to, PassIndex(1));
    assert_eq!(edges[1].from, PassIndex(1));
    assert_eq!(edges[1].to, PassIndex(2));
    assert_eq!(edges[2].from, PassIndex(2));
    assert_eq!(edges[2].to, PassIndex(3));
}

#[test]
fn chain_single_pass_produces_no_edges() {
    let mut builder = EdgeBuilder::new();
    builder.chain(&[PassIndex(0)], ResourceHandle(1));
    let edges = builder.build();
    assert!(edges.is_empty(), "chain with single pass produces no edges",);
}

#[test]
fn chain_empty_slice_produces_no_edges() {
    let mut builder = EdgeBuilder::new();
    builder.chain(&[], ResourceHandle(1));
    let edges = builder.build();
    assert!(edges.is_empty(), "chain with empty slice produces no edges",);
}

#[test]
fn chain_preserves_resource_on_all_edges() {
    let mut builder = EdgeBuilder::new();
    let res = ResourceHandle(42);
    builder.chain(
        &[PassIndex(0), PassIndex(1), PassIndex(2), PassIndex(3)],
        res,
    );
    let edges = builder.build();

    assert_eq!(edges.len(), 3);
    for (i, edge) in edges.iter().enumerate() {
        assert_eq!(edge.resource, res, "chain edge[{}] has correct resource", i,);
    }
}

#[test]
fn chain_edges_are_all_raw() {
    let mut builder = EdgeBuilder::new();
    builder.chain(
        &[PassIndex(0), PassIndex(1), PassIndex(2), PassIndex(3)],
        ResourceHandle(1),
    );
    let edges = builder.build();

    assert_eq!(edges.len(), 3);
    for (i, edge) in edges.iter().enumerate() {
        assert_eq!(edge.edge_type, EdgeType::RAW, "chain edge[{}] is RAW", i,);
    }
}

#[test]
fn chain_with_non_sequential_pass_indices() {
    let mut builder = EdgeBuilder::new();
    builder.chain(
        &[PassIndex(10), PassIndex(55), PassIndex(200)],
        ResourceHandle(7),
    );
    let edges = builder.build();
    assert_eq!(
        edges.len(),
        2,
        "chain with non-seq indices produces 2 edges"
    );
    assert_eq!(edges[0].from, PassIndex(10));
    assert_eq!(edges[0].to, PassIndex(55));
    assert_eq!(edges[1].from, PassIndex(55));
    assert_eq!(edges[1].to, PassIndex(200));
}

#[test]
fn chain_returns_mut_self_for_chaining() {
    let mut builder = EdgeBuilder::new();
    builder
        .chain(&[PassIndex(0), PassIndex(1)], ResourceHandle(1))
        .chain(&[PassIndex(2), PassIndex(3)], ResourceHandle(2));
    let edges = builder.build();
    assert_eq!(edges.len(), 2, "chained chain() calls accumulate",);
}

// =============================================================================
// SECTION 10 -- EdgeType variant discrimination
// =============================================================================

#[test]
fn edge_type_variants_are_distinct() {
    assert_ne!(EdgeType::RAW, EdgeType::WAR, "RAW != WAR",);
    assert_ne!(EdgeType::WAR, EdgeType::WAW, "WAR != WAW",);
    assert_ne!(EdgeType::WAW, EdgeType::RAW, "WAW != RAW",);
}

#[test]
fn edge_type_derives_clone_copy() {
    let original = EdgeType::RAW;
    let cloned = original;
    assert_eq!(original, cloned, "EdgeType is Copy");
}

#[test]
fn edge_type_display_formats_correctly() {
    assert_eq!(format!("{}", EdgeType::RAW), "RAW");
    assert_eq!(format!("{}", EdgeType::WAR), "WAR");
    assert_eq!(format!("{}", EdgeType::WAW), "WAW");
}

// =============================================================================
// SECTION 11 -- IrEdge Display / Debug
// =============================================================================

#[test]
fn ir_edge_display_format() {
    let edge = IrEdge::new(PassIndex(0), PassIndex(2), ResourceHandle(7), EdgeType::RAW);
    let display = format!("{}", edge);
    assert!(!display.is_empty(), "IrEdge Display is non-empty",);
    assert!(display.contains("RAW"), "IrEdge Display contains edge type",);
}

#[test]
fn ir_edge_debug_format() {
    let edge = IrEdge::new(PassIndex(3), PassIndex(4), ResourceHandle(5), EdgeType::WAR);
    let debug = format!("{:?}", edge);
    assert!(!debug.is_empty(), "IrEdge Debug is non-empty",);
}

// =============================================================================
// SECTION 12 -- EdgeBuilder derives
// =============================================================================

#[test]
fn edge_builder_clone_produces_equal_builder() {
    let mut builder_a = EdgeBuilder::new();
    builder_a.add_raw(PassIndex(0), PassIndex(1), ResourceHandle(1));
    builder_a.add_war(PassIndex(2), PassIndex(3), ResourceHandle(2));

    let mut builder_b = builder_a.clone();
    let edges_a = builder_a.build();
    let edges_b = builder_b.build();

    assert_eq!(edges_a, edges_b, "cloned builder produces identical edges",);
}

#[test]
fn edge_builder_clone_is_independent() {
    let mut builder_a = EdgeBuilder::new();
    builder_a.add_raw(PassIndex(0), PassIndex(1), ResourceHandle(1));

    let mut builder_b = builder_a.clone();
    builder_b.add_war(PassIndex(2), PassIndex(3), ResourceHandle(2));

    // builder_a should not have the edge added to builder_b.
    let edges_a = builder_a.build();
    let edges_b = builder_b.build();

    assert_eq!(
        edges_a.len(),
        1,
        "original builder unaffected by clone mutation",
    );
    assert_eq!(edges_b.len(), 2, "clone has its own edges",);
}

#[test]
fn edge_builder_debug_non_empty() {
    let mut builder = EdgeBuilder::new();
    builder.add_raw(PassIndex(0), PassIndex(1), ResourceHandle(1));
    let debug = format!("{:?}", builder);
    assert!(!debug.is_empty(), "EdgeBuilder Debug output is non-empty",);
}

// =============================================================================
// SECTION 13 -- Mixed usage
// =============================================================================

#[test]
fn mixed_add_edge_add_raw_add_war_add_waw() {
    let mut builder = EdgeBuilder::new();

    // Use all four mutation methods.
    builder.add_edge(
        PassIndex(0),
        PassIndex(1),
        ResourceHandle(10),
        EdgeType::RAW,
    );
    builder.add_raw(PassIndex(2), PassIndex(3), ResourceHandle(20));
    builder.add_war(PassIndex(4), PassIndex(5), ResourceHandle(30));
    builder.add_waw(PassIndex(6), PassIndex(7), ResourceHandle(40));
    builder.chain(
        &[PassIndex(8), PassIndex(9), PassIndex(10)],
        ResourceHandle(50),
    );

    let edges = builder.build();
    // 1 (add_edge) + 1 (add_raw) + 1 (add_war) + 1 (add_waw) + 2 (chain 3 passes)
    assert_eq!(
        edges.len(),
        6,
        "mixed usage produces correct total edge count",
    );

    // Verify each edge's type.
    assert_eq!(edges[0].edge_type, EdgeType::RAW);
    assert_eq!(edges[0].resource, ResourceHandle(10));
    assert_eq!(edges[0].from, PassIndex(0));
    assert_eq!(edges[0].to, PassIndex(1));

    assert_eq!(edges[1].edge_type, EdgeType::RAW);
    assert_eq!(edges[1].resource, ResourceHandle(20));
    assert_eq!(edges[1].from, PassIndex(2));
    assert_eq!(edges[1].to, PassIndex(3));

    assert_eq!(edges[2].edge_type, EdgeType::WAR);
    assert_eq!(edges[2].resource, ResourceHandle(30));
    assert_eq!(edges[2].from, PassIndex(4));
    assert_eq!(edges[2].to, PassIndex(5));

    assert_eq!(edges[3].edge_type, EdgeType::WAW);
    assert_eq!(edges[3].resource, ResourceHandle(40));
    assert_eq!(edges[3].from, PassIndex(6));
    assert_eq!(edges[3].to, PassIndex(7));

    // Chain edges.
    assert_eq!(edges[4].edge_type, EdgeType::RAW);
    assert_eq!(edges[4].resource, ResourceHandle(50));
    assert_eq!(edges[4].from, PassIndex(8));
    assert_eq!(edges[4].to, PassIndex(9));

    assert_eq!(edges[5].edge_type, EdgeType::RAW);
    assert_eq!(edges[5].resource, ResourceHandle(50));
    assert_eq!(edges[5].from, PassIndex(9));
    assert_eq!(edges[5].to, PassIndex(10));
}

// =============================================================================
// SECTION 14 -- Large batch
// =============================================================================

#[test]
fn large_batch_of_edges() {
    let mut builder = EdgeBuilder::new();
    let count = 100;

    for i in 0..count {
        let from = PassIndex(i * 2);
        let to = PassIndex(i * 2 + 1);
        let res = ResourceHandle(i as u32);
        builder.add_raw(from, to, res);
    }

    let edges = builder.build();
    assert_eq!(
        edges.len(),
        count,
        "batch of {} edges accumulated correctly",
        count,
    );

    // Verify every edge.
    for (i, edge) in edges.iter().enumerate() {
        assert_eq!(edge.from, PassIndex(i * 2), "edge[{}] from", i);
        assert_eq!(edge.to, PassIndex(i * 2 + 1), "edge[{}] to", i);
        assert_eq!(
            edge.resource,
            ResourceHandle(i as u32),
            "edge[{}] resource",
            i
        );
        assert_eq!(edge.edge_type, EdgeType::RAW, "edge[{}] type", i);
    }
}

#[test]
fn large_batch_of_edges_different_types() {
    let mut builder = EdgeBuilder::new();
    let count = 30;

    for i in 0..count {
        let from = PassIndex(i);
        let to = PassIndex(i + 1);
        let res = ResourceHandle(i as u32);
        let etype = match i % 3 {
            0 => EdgeType::RAW,
            1 => EdgeType::WAR,
            _ => EdgeType::WAW,
        };
        builder.add_edge(from, to, res, etype);
    }

    let edges = builder.build();
    assert_eq!(edges.len(), count, "batch of {} edges", count);

    for (i, edge) in edges.iter().enumerate() {
        assert_eq!(edge.from, PassIndex(i), "edge[{}] from", i);
        assert_eq!(edge.to, PassIndex(i + 1), "edge[{}] to", i);
        assert_eq!(
            edge.resource,
            ResourceHandle(i as u32),
            "edge[{}] resource",
            i
        );
        let expected = match i % 3 {
            0 => EdgeType::RAW,
            1 => EdgeType::WAR,
            _ => EdgeType::WAW,
        };
        assert_eq!(edge.edge_type, expected, "edge[{}] type", i);
    }
}

// =============================================================================
// SECTION 15 -- PassIndex and ResourceHandle edge cases
// =============================================================================

#[test]
fn edge_with_none_resource_handle() {
    let mut builder = EdgeBuilder::new();
    builder.add_edge(
        PassIndex(0),
        PassIndex(1),
        ResourceHandle::NONE,
        EdgeType::RAW,
    );
    let edges = builder.build();
    assert_eq!(edges.len(), 1, "edge with NONE resource created");
    assert_eq!(
        edges[0].resource,
        ResourceHandle::NONE,
        "resource is NONE sentinel",
    );
}

#[test]
fn edge_with_zero_pass_index() {
    let mut builder = EdgeBuilder::new();
    builder.add_edge(PassIndex(0), PassIndex(0), ResourceHandle(1), EdgeType::RAW);
    let edges = builder.build();
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].from, PassIndex(0));
    assert_eq!(edges[0].to, PassIndex(0));
}

// =============================================================================
// SECTION 16 -- Build after chain
// =============================================================================

#[test]
fn chain_then_build_then_chain_again() {
    let mut builder = EdgeBuilder::new();

    // First cycle.
    builder.chain(&[PassIndex(0), PassIndex(1)], ResourceHandle(1));
    let first = builder.build();
    assert_eq!(first.len(), 1);

    // Second cycle on same builder.
    builder.chain(&[PassIndex(2), PassIndex(3)], ResourceHandle(2));
    let second = builder.build();
    assert_eq!(second.len(), 1);
    assert_eq!(second[0].from, PassIndex(2));
    assert_eq!(second[0].to, PassIndex(3));
    assert_eq!(second[0].resource, ResourceHandle(2));
}

// =============================================================================
// SECTION 17 -- Clone with edges then build both
// =============================================================================

#[test]
fn clone_builder_with_edges_then_build_both() {
    let mut builder = EdgeBuilder::new();
    builder.add_raw(PassIndex(0), PassIndex(1), ResourceHandle(1));
    builder.add_war(PassIndex(2), PassIndex(3), ResourceHandle(2));

    let mut cloned = builder.clone();

    // Add different edges to each.
    builder.add_waw(PassIndex(4), PassIndex(5), ResourceHandle(3));
    cloned.add_raw(PassIndex(6), PassIndex(7), ResourceHandle(4));

    let original_edges = builder.build();
    let cloned_edges = cloned.build();

    assert_eq!(original_edges.len(), 3, "original has 3 edges");
    assert_eq!(cloned_edges.len(), 3, "clone has 3 edges");

    // The first two edges (shared pre-clone state) are identical.
    assert_eq!(original_edges[0], cloned_edges[0]);
    assert_eq!(original_edges[1], cloned_edges[1]);

    // The third edge differs.
    assert_ne!(original_edges[2], cloned_edges[2]);
}

// =============================================================================
// SECTION 18 -- Multiple chain calls
// =============================================================================

#[test]
fn multiple_chain_calls_accumulate() {
    let mut builder = EdgeBuilder::new();
    builder
        .chain(
            &[PassIndex(0), PassIndex(1), PassIndex(2)],
            ResourceHandle(1),
        )
        .chain(
            &[PassIndex(3), PassIndex(4), PassIndex(5)],
            ResourceHandle(2),
        )
        .chain(&[PassIndex(6), PassIndex(7)], ResourceHandle(3));

    let edges = builder.build();
    // First chain: 2 edges, Second chain: 2 edges, Third chain: 1 edge
    assert_eq!(edges.len(), 5, "three chain calls accumulate");

    // First chain: passes 0->1, 1->2 on resource 1
    assert_eq!(edges[0].from, PassIndex(0));
    assert_eq!(edges[0].to, PassIndex(1));
    assert_eq!(edges[0].resource, ResourceHandle(1));
    assert_eq!(edges[1].from, PassIndex(1));
    assert_eq!(edges[1].to, PassIndex(2));
    assert_eq!(edges[1].resource, ResourceHandle(1));

    // Second chain: passes 3->4, 4->5 on resource 2
    assert_eq!(edges[2].from, PassIndex(3));
    assert_eq!(edges[2].to, PassIndex(4));
    assert_eq!(edges[2].resource, ResourceHandle(2));
    assert_eq!(edges[3].from, PassIndex(4));
    assert_eq!(edges[3].to, PassIndex(5));
    assert_eq!(edges[3].resource, ResourceHandle(2));

    // Third chain: passes 6->7 on resource 3
    assert_eq!(edges[4].from, PassIndex(6));
    assert_eq!(edges[4].to, PassIndex(7));
    assert_eq!(edges[4].resource, ResourceHandle(3));
}

// =============================================================================
// SECTION 19 -- EdgeType Hash and Eq (stored in HashSet)
// =============================================================================

#[test]
fn edge_type_can_be_stored_in_hashset() {
    use std::collections::HashSet;

    let mut set = HashSet::new();
    set.insert(EdgeType::RAW);
    set.insert(EdgeType::WAR);
    set.insert(EdgeType::WAW);

    assert_eq!(set.len(), 3, "all three EdgeType variants in HashSet");

    assert!(set.contains(&EdgeType::RAW));
    assert!(set.contains(&EdgeType::WAR));
    assert!(set.contains(&EdgeType::WAW));
}

// =============================================================================
// SECTION 20 -- Default is consistent with new
// =============================================================================

#[test]
fn default_builder_is_empty() {
    let mut builder: EdgeBuilder = Default::default();
    assert!(builder.build().is_empty(), "Default builder is empty");
}

#[test]
fn default_builder_clone_is_empty() {
    let builder: EdgeBuilder = Default::default();
    let mut cloned = builder.clone();
    assert!(
        cloned.build().is_empty(),
        "Clone of Default builder is empty"
    );
}
