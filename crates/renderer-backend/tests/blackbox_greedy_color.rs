// SPDX-License-Identifier: MIT
//
// blackbox_greedy_color.rs -- Blackbox contract tests for T-FG-3.3 GreedyColor.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only the public types and functions exported by
// `renderer_backend::frame_graph` -- no internal fields, no private methods.
//
// Acceptance criterion (T-FG-3.3):
//   greedy_color_resources(interference, resources) -> HashMap<ResourceHandle, u32>
//     assigns colours to resources using greedy largest-first graph colouring:
//     resources are sorted by estimated byte size descending, then each is
//     assigned the smallest non-negative integer colour not used by any
//     already-coloured neighbour in the interference graph.
//   num_colours(map) -> u32 returns the number of distinct colours in the map.
//
// Contract:
//   - Empty resource list produces an empty colour map.
//   - Single resource receives colour 0.
//   - Non-interfering resources may share the same colour.
//   - Interfering resources always receive distinct colours.
//   - Larger (by estimated_bytes) resources are coloured before smaller ones.
//   - Colour numbers are the smallest non-negative integer not used by any
//     neighbour -- no colour gaps are introduced beyond what interference
//     forces.
//   - num_colours counts distinct colour values in the map.
//   - Texture resources with different GPU formats interfere even when their
//     lifetime intervals are disjoint (format-mismatch rule).
//   - Buffer resources never format-interfere; they only interfere via
//     lifetime overlap.
//   - Resources not present in the interference graph are absent from the
//     colour map (they are not coloured).
//
// Coverage:
//   1.  Empty input -> empty colour map
//   2.  Single resource -> colour 0
//   3.  Non-interfering resources share colour 0
//   4.  Interfering resources get distinct colours (0 and 1)
//   5.  Greedy largest-first across multiple interfering resources
//   6.  Chain interference reuses colour across non-interfering ends
//   7.  Complete graph K5 -- every resource gets a unique colour
//   8.  Disconnected interference components reuse colours independently
//   9.  Size ordering: larger resource coloured before smaller
//  10.  Format mismatch causes interference despite disjoint lifetimes
//  11.  num_colours on empty map returns 0
//  12.  num_colours on single colour returns 1
//  13.  num_colours counts distinct values, not handles
//  14.  Mixed buffer and texture resources
//  15.  Resource not in any pass has no lifetime, is absent from colour map

use renderer_backend::frame_graph::{
    BufferDesc, DispatchSource, EmptyView, InstanceSource, InterferenceGraph, IrPass, IrResource,
    PassIndex, PassType, ResourceAccessSet, ResourceDesc, ResourceHandle, ResourceLifetime,
    ResourceState, TextureDesc, ViewType,
    compute_lifetimes, greedy_color_resources, num_colors,
};
use std::collections::HashMap;

// =============================================================================
// Helpers
// =============================================================================

/// Creates a texture resource with the given parameters.
fn tex(
    handle: u32,
    name: &str,
    format: &str,
    width: u32,
    height: u32,
) -> IrResource {
    IrResource::new(
        ResourceHandle(handle),
        name,
        ResourceDesc::Texture2D(TextureDesc {
            width,
            height,
            mip_levels: 1,
            array_layers: 1,
            format: format.into(),
        }),
        ResourceLifetime::Transient,
        ResourceState::Uninitialized,
    )
}

/// Creates a buffer resource with the given size.
fn buf(handle: u32, name: &str, size: u64) -> IrResource {
    IrResource::new(
        ResourceHandle(handle),
        name,
        ResourceDesc::Buffer(BufferDesc {
            size,
            usage: "storage".into(),
            is_indirect_arg: false,
        }),
        ResourceLifetime::Transient,
        ResourceState::Uninitialized,
    )
}

/// Creates a compute pass that reads and writes the given resources.
fn compute_pass(idx: usize, reads: &[ResourceHandle], writes: &[ResourceHandle]) -> IrPass {
    IrPass {
        index: PassIndex(idx),
        name: format!("p{}", idx),
        pass_type: PassType::Compute,
        access_set: ResourceAccessSet {
            reads: reads.to_vec(),
            writes: writes.to_vec(),
        },
        color_attachments: Vec::new(),
        depth_stencil: None,
        instance_source: InstanceSource::Direct {
            index_count: 0,
            instance_count: 1,
            base_vertex: 0,
            first_index: 0,
            first_instance: 0,
        },
        dispatch_source: Some(DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        }),
        view_type: ViewType::Storage,
        tags: Vec::new(),
        feature_flags: 0,
        view: std::sync::Arc::new(EmptyView { name: format!("pass{}", idx) }),
    }
}

/// Creates a copy pass that reads from `src` and writes to `dst`.
fn copy_pass(idx: usize, src: ResourceHandle, dst: ResourceHandle) -> IrPass {
    IrPass {
        index: PassIndex(idx),
        name: format!("copy{}", idx),
        pass_type: PassType::Copy,
        access_set: ResourceAccessSet {
            reads: vec![src],
            writes: vec![dst],
        },
        color_attachments: Vec::new(),
        depth_stencil: None,
        instance_source: InstanceSource::Direct {
            index_count: 0,
            instance_count: 1,
            base_vertex: 0,
            first_index: 0,
            first_instance: 0,
        },
        dispatch_source: None,
        view_type: ViewType::StorageBuffer,
        tags: Vec::new(),
        feature_flags: 0,
        view: std::sync::Arc::new(EmptyView { name: format!("copy{}", idx) }),
    }
}

/// Builds an interference graph from resources and passes.
fn build_interference(
    resources: &[IrResource],
    passes: &[IrPass],
) -> InterferenceGraph {
    let lifetimes = compute_lifetimes(passes, &[], resources);
    InterferenceGraph::build(resources, &lifetimes)
}

/// Runs greedy colouring and returns the colour map.
fn run_greedy(
    resources: &[IrResource],
    passes: &[IrPass],
) -> HashMap<ResourceHandle, u32> {
    let lifetimes = compute_lifetimes(passes, &[], resources);
    let ig = InterferenceGraph::build(resources, &lifetimes);
    greedy_color_resources(&ig, resources)
}

// =============================================================================
// SECTION 1 -- Empty input and edge cases
// =============================================================================

/// Empty resources with no passes produces an empty colour map.
#[test]
fn empty_input_produces_empty_colour_map() {
    let resources: Vec<IrResource> = vec![];
    let passes: Vec<IrPass> = vec![];

    let colours = run_greedy(&resources, &passes);

    assert!(
        colours.is_empty(),
        "Empty input must produce an empty colour map, got {} entries",
        colours.len(),
    );
}

/// num_colours on an empty map returns 0.
#[test]
fn num_colors_empty_map_is_zero() {
    let map: HashMap<ResourceHandle, u32> = HashMap::new();
    assert_eq!(num_colors(&map), 0, "Empty map has 0 colours");
}

// =============================================================================
// SECTION 2 -- Single resource
// =============================================================================

/// A single resource used by one pass receives colour 0.
#[test]
fn single_resource_gets_colour_zero() {
    let r0 = ResourceHandle(0);
    let resources = vec![buf(0, "single", 4096)];
    let passes = vec![compute_pass(0, &[r0], &[])];

    let colours = run_greedy(&resources, &passes);

    assert_eq!(colours.len(), 1, "One resource coloured");
    assert_eq!(
        colours.get(&r0),
        Some(&0),
        "Single resource receives colour 0",
    );
    assert_eq!(num_colors(&colours), 1, "One distinct colour");
}

/// num_colours returns 1 when the map has one entry.
#[test]
fn num_colors_single_entry() {
    let mut map = HashMap::new();
    map.insert(ResourceHandle(0), 0u32);
    assert_eq!(num_colors(&map), 1);
}

// =============================================================================
// SECTION 3 -- Non-interfering resources share colour 0
// =============================================================================

/// Two buffers with disjoint lifetimes (non-overlapping passes) can share
/// colour 0.
#[test]
fn non_interfering_buffers_share_colour() {
    let r0 = ResourceHandle(0);
    let r1 = ResourceHandle(1);

    // R0 used in pass 0, R1 used in pass 1 -- intervals (0,0) and (1,1) do
    // NOT overlap.
    let resources = vec![buf(0, "buf_a", 4096), buf(1, "buf_b", 4096)];
    let passes = vec![
        compute_pass(0, &[r0], &[]),
        compute_pass(1, &[r1], &[]),
    ];

    let colours = run_greedy(&resources, &passes);

    assert_eq!(colours.len(), 2, "Both resources coloured");
    assert_eq!(
        colours.get(&r0),
        Some(&0),
        "R0 gets colour 0",
    );
    assert_eq!(
        colours.get(&r1),
        Some(&0),
        "R1 also gets colour 0 (non-interfering)",
    );
    assert_eq!(num_colors(&colours), 1, "Only one distinct colour needed");
}

/// Two textures with the *same* format and disjoint lifetimes share colour.
#[test]
fn non_interfering_same_format_textures_share_colour() {
    let r0 = ResourceHandle(0);
    let r1 = ResourceHandle(1);

    let resources = vec![
        tex(0, "tex_a", "rgba8unorm", 100, 100),
        tex(1, "tex_b", "rgba8unorm", 200, 200),
    ];
    let passes = vec![
        compute_pass(0, &[r0], &[]),
        compute_pass(2, &[r1], &[]),
    ];

    let colours = run_greedy(&resources, &passes);

    assert_eq!(*colours.get(&r0).unwrap(), 0, "R0 gets colour 0");
    assert_eq!(*colours.get(&r1).unwrap(), 0, "R1 gets colour 0 (same format, disjoint lifetimes)");
    assert_eq!(num_colors(&colours), 1, "One colour suffices");
}

// =============================================================================
// SECTION 4 -- Interfering resources get distinct colours
// =============================================================================

/// Two buffers with overlapping lifetimes get distinct colours.
#[test]
fn interfering_buffers_get_distinct_colours() {
    let r0 = ResourceHandle(0);
    let r1 = ResourceHandle(1);

    // Both buffers used by the same pass -> overlapping lifetime.
    let resources = vec![buf(0, "buf_a", 4096), buf(1, "buf_b", 4096)];
    let passes = vec![compute_pass(0, &[r0, r1], &[])];

    let colours = run_greedy(&resources, &passes);

    assert_eq!(colours.len(), 2, "Both resources coloured");
    assert_ne!(
        colours.get(&r0),
        colours.get(&r1),
        "Interfering resources must get different colours",
    );

    // Colours should be 0 and 1 (smallest available).
    let mut vals: Vec<u32> = colours.values().copied().collect();
    vals.sort();
    assert_eq!(vals, vec![0, 1], "Colours are 0 and 1");
    assert_eq!(num_colors(&colours), 2, "Two distinct colours");
}

/// Two textures with overlapping lifetimes get distinct colours.
#[test]
fn interfering_textures_get_distinct_colours() {
    let r0 = ResourceHandle(0);
    let r1 = ResourceHandle(1);

    let resources = vec![
        tex(0, "tex_a", "rgba8unorm", 100, 100),
        tex(1, "tex_b", "r32float", 100, 100),
    ];
    let passes = vec![compute_pass(0, &[r0, r1], &[])];

    let colours = run_greedy(&resources, &passes);

    assert_eq!(colours.len(), 2);
    assert_ne!(colours.get(&r0), colours.get(&r1));
    assert_eq!(num_colors(&colours), 2);
}

// =============================================================================
// SECTION 5 -- Greedy largest-first ordering
// =============================================================================

/// Three resources all interfering: greedy largest-first assigns distinct
/// colours with the largest resource getting colour 0.
#[test]
fn greedy_largest_first_three_interfering() {
    let r_small = ResourceHandle(0);
    let r_medium = ResourceHandle(1);
    let r_large = ResourceHandle(2);

    // Sizes: large=1_000_000, medium=100_000, small=10_000.
    let resources = vec![
        buf(0, "small", 10_000),
        buf(1, "medium", 100_000),
        buf(2, "large", 1_000_000),
    ];

    // All three used by the same pass -> all interfere.
    let passes = vec![compute_pass(
        0,
        &[r_small, r_medium, r_large],
        &[],
    )];

    let colours = run_greedy(&resources, &passes);

    assert_eq!(colours.len(), 3, "All three resources coloured");

    // Greedy sorts by size descending: large, medium, small.
    // - large (largest) gets colour 0.
    // - medium interferes with large -> colour 1.
    // - small interferes with both -> colour 2.
    assert_eq!(
        *colours.get(&r_large).unwrap(),
        0,
        "Largest resource gets colour 0",
    );
    assert_eq!(
        *colours.get(&r_medium).unwrap(),
        1,
        "Medium resource gets colour 1",
    );
    assert_eq!(
        *colours.get(&r_small).unwrap(),
        2,
        "Smallest resource gets colour 2",
    );
    assert_eq!(num_colors(&colours), 3, "Three distinct colours");
}

/// Greedy picks the *first available* colour (not necessarily colour 0 if
/// neighbours block it).
#[test]
fn greedy_picks_first_available_not_always_zero() {
    // R0 and R1 interfere; R0 is larger.
    // R0 gets 0.
    // R1 sees neighbour R0 has 0 -> picks 1.
    //
    // R2 (smallest) does NOT interfere with R0 but DOES interfere with R1.
    // R2 sees neighbour R1 has 1, but 0 is available (R0 is not a neighbour).
    // So R2 gets 0.
    let r0 = ResourceHandle(0);
    let r1 = ResourceHandle(1);
    let r2 = ResourceHandle(2);

    let resources = vec![
        buf(0, "large", 1_000_000),
        buf(1, "medium", 100_000),
        buf(2, "small", 10_000),
    ];

    // Lifetime pattern: R0 and R1 overlap, R1 and R2 overlap, but R0 and
    // R2 do NOT overlap.
    let passes = vec![
        compute_pass(0, &[r0], &[]),       // R0 used in p0
        compute_pass(1, &[r0, r1], &[]),   // R0+R1 overlap
        compute_pass(2, &[r1, r2], &[]),   // R1+R2 overlap
        compute_pass(3, &[r2], &[]),       // R2 used in p3
    ];

    let colours = run_greedy(&resources, &passes);

    assert_eq!(colours.len(), 3);

    // large (size=1M) -> colour 0
    assert_eq!(*colours.get(&r0).unwrap(), 0, "R0 (largest) gets colour 0");

    // medium (size=100K) interferes with R0 -> colour 1
    assert_eq!(*colours.get(&r1).unwrap(), 1, "R1 gets colour 1");

    // small (size=10K) interferes with R1 but NOT R0 -> colour 0 is available
    assert_eq!(
        *colours.get(&r2).unwrap(),
        0,
        "R2 reuses colour 0 (does not interfere with R0)",
    );

    assert_eq!(num_colors(&colours), 2, "Two distinct colours reused");
}

// =============================================================================
// SECTION 6 -- Chain interference: non-interfering ends share colour
// =============================================================================

/// Chain: R0-R1 interfere, R1-R2 interfere, R0-R2 do NOT interfere.
/// Greedy largest-first: R0=0, R1=1, R2=0.
#[test]
fn chain_interference_reuses_colour() {
    let r0 = ResourceHandle(0);
    let r1 = ResourceHandle(1);
    let r2 = ResourceHandle(2);

    // Sizes descending: R0 (largest), R1 (medium), R2 (smallest).
    let resources = vec![
        buf(0, "chain_a", 1_000_000),
        buf(1, "chain_b", 100_000),
        buf(2, "chain_c", 10_000),
    ];

    // R0 read in p0, p1
    // R1 read in p1, p2
    // R2 read in p2, p3
    // R0 and R2 DO NOT overlap (R0 last=p1, R2 first=p2 -> no overlap).
    let passes = vec![
        compute_pass(0, &[r0], &[]),
        compute_pass(1, &[r0, r1], &[]),
        compute_pass(2, &[r1, r2], &[]),
        compute_pass(3, &[r2], &[]),
    ];

    let colours = run_greedy(&resources, &passes);

    assert_eq!(colours.len(), 3);

    assert_eq!(*colours.get(&r0).unwrap(), 0, "R0 gets colour 0");
    assert_eq!(*colours.get(&r1).unwrap(), 1, "R1 gets colour 1 (interferes with R0)");
    assert_eq!(
        *colours.get(&r2).unwrap(),
        0,
        "R2 reuses colour 0 (no interference with R0)",
    );

    assert_eq!(num_colors(&colours), 2, "Two colours for chain");
}

// =============================================================================
// SECTION 7 -- Complete graph K5: all distinct colours
// =============================================================================

/// A complete graph of 5 resources (all pairwise interference) must assign
/// 5 distinct colours.
#[test]
fn complete_graph_k5_all_distinct() {
    let handles: Vec<ResourceHandle> = (0..5).map(ResourceHandle).collect();

    // Buffers of equal size; all used by the same pass (all interfere).
    let resources: Vec<IrResource> = (0..5)
        .map(|i| buf(i, &format!("buf_{}", i), 4096))
        .collect();

    let passes = vec![compute_pass(0, &handles, &[])];

    let colours = run_greedy(&resources, &passes);

    assert_eq!(colours.len(), 5, "All 5 resources coloured");
    assert_eq!(num_colors(&colours), 5, "Five distinct colours required");

    // Colours should be exactly {0, 1, 2, 3, 4}.
    let mut vals: Vec<u32> = colours.values().copied().collect();
    vals.sort();
    assert_eq!(vals, vec![0, 1, 2, 3, 4], "Colours 0..4 assigned");
}

// =============================================================================
// SECTION 8 -- Disconnected components reuse colours independently
// =============================================================================

/// Two disjoint interference groups each get colours 0 and 1 independently.
#[test]
fn disconnected_components_reuse_colours() {
    // Group A: R0, R1 (interfere with each other, larger group A resources)
    // Group B: R2, R3 (interfere with each other, disjoint from group A)
    //
    // R0 (largest in group A) -> 0
    // R1 -> 1
    // R2 (largest in group B) -> 0 (no interference with group A)
    // R3 -> 1

    let r0 = ResourceHandle(0);
    let r1 = ResourceHandle(1);
    let r2 = ResourceHandle(2);
    let r3 = ResourceHandle(3);

    let resources = vec![
        buf(0, "A_large", 1_000_000),
        buf(1, "A_small", 10_000),
        buf(2, "B_large", 500_000),
        buf(3, "B_small", 5_000),
    ];

    // Group A used in passes 0-1, group B used in passes 2-3.
    // The lifetimes of group A and B DO NOT overlap.
    let passes = vec![
        compute_pass(0, &[r0], &[]),
        compute_pass(1, &[r0, r1], &[]),
        compute_pass(2, &[r2], &[]),
        compute_pass(3, &[r2, r3], &[]),
    ];

    let colours = run_greedy(&resources, &passes);

    assert_eq!(colours.len(), 4);

    // Group A: R0 gets 0 (largest in A), R1 gets 1 (interferes with R0).
    assert_eq!(*colours.get(&r0).unwrap(), 0, "R0 (group A largest) gets 0");
    assert_eq!(*colours.get(&r1).unwrap(), 1, "R1 (group A) gets 1");

    // Group B: R2 gets 0 (no interference with group A), R3 gets 1.
    assert_eq!(*colours.get(&r2).unwrap(), 0, "R2 (group B) reuses colour 0");
    assert_eq!(*colours.get(&r3).unwrap(), 1, "R3 (group B) gets colour 1");

    assert_eq!(num_colors(&colours), 2, "Two colours suffice for two groups");
}

// =============================================================================
// SECTION 9 -- Size ordering verification
// =============================================================================

/// When two resources interfere, the larger one is coloured first and gets
/// colour 0, while the smaller one gets colour 1.
#[test]
fn larger_resource_coloured_before_smaller() {
    let r_small = ResourceHandle(0);
    let r_large = ResourceHandle(1);

    let resources = vec![
        buf(0, "small_40k", 40_000),
        buf(1, "large_1M", 1_000_000),
    ];
    let passes = vec![compute_pass(0, &[r_small, r_large], &[])];

    let colours = run_greedy(&resources, &passes);

    // Large gets color 0 (coloured first), small gets 1.
    assert_eq!(
        *colours.get(&r_large).unwrap(),
        0,
        "Larger resource (1M) gets colour 0",
    );
    assert_eq!(
        *colours.get(&r_small).unwrap(),
        1,
        "Smaller resource (40K) gets colour 1",
    );
}

/// Texture with larger estimated bytes gets higher colour priority.
#[test]
fn larger_texture_coloured_first() {
    let r_small = ResourceHandle(0);
    let r_large = ResourceHandle(1);

    // rgba8unorm = 4 bytes/texel.
    // small: 16x16 = 256 px * 4 = 1024 bytes.
    // large: 1920x1080 = 2_073_600 px * 4 = 8_294_400 bytes.
    let resources = vec![
        tex(0, "small_tex", "rgba8unorm", 16, 16),
        tex(1, "large_tex", "rgba8unorm", 1920, 1080),
    ];
    let passes = vec![compute_pass(0, &[r_small, r_large], &[])];

    let colours = run_greedy(&resources, &passes);

    assert_eq!(
        *colours.get(&r_large).unwrap(),
        0,
        "Larger texture gets colour 0",
    );
    assert_eq!(
        *colours.get(&r_small).unwrap(),
        1,
        "Smaller texture gets colour 1",
    );
}

/// Texture estimated_bytes accounts for mip_levels and array_layers.
#[test]
fn texture_estimated_bytes_includes_mips_and_layers() {
    let r0 = ResourceHandle(0);
    let r1 = ResourceHandle(1);

    // r0: 4x4 rgba8unorm, 4 mips, 1 layer = 4*4*(1+1/4+1/16+1/64)*4 ≈ 85 bytes
    // r1: 4x4 rgba8unorm, 1 mip, 6 layers = 4*4*1*6*4 = 384 bytes (cube-like)
    // r1 > r0 in estimated bytes due to array_layers.
    let resources = vec![
        IrResource::new(
            ResourceHandle(0),
            "multi_mip",
            ResourceDesc::Texture2D(TextureDesc {
                width: 4,
                height: 4,
                mip_levels: 4,
                array_layers: 1,
                format: "rgba8unorm".into(),
            }),
            ResourceLifetime::Transient,
            ResourceState::Uninitialized,
        ),
        IrResource::new(
            ResourceHandle(1),
            "array_tex",
            ResourceDesc::Texture2D(TextureDesc {
                width: 4,
                height: 4,
                mip_levels: 1,
                array_layers: 6,
                format: "rgba8unorm".into(),
            }),
            ResourceLifetime::Transient,
            ResourceState::Uninitialized,
        ),
    ];
    let passes = vec![compute_pass(0, &[r0, r1], &[])];

    let colours = run_greedy(&resources, &passes);

    // r1 (6-layer array) has larger estimated_bytes -> colour 0.
    assert_eq!(*colours.get(&r1).unwrap(), 0, "6-layer texture gets colour 0");
    assert_eq!(*colours.get(&r0).unwrap(), 1, "multi-mip texture gets colour 1");
}

// =============================================================================
// SECTION 10 -- Format mismatch causes interference
// =============================================================================

/// Two textures with different GPU formats interfere even when their
/// lifetimes are disjoint.
#[test]
fn format_mismatch_causes_interference_despite_disjoint_lifetimes() {
    let r0 = ResourceHandle(0);
    let r1 = ResourceHandle(1);

    // R0: rgba8unorm, used in pass 0 only.
    // R1: r32float, used in pass 1 only.
    // Lifetimes: (0,0) and (1,1) -> NO overlap.
    // BUT formats differ -> format_mismatch = true -> interfere.
    let resources = vec![
        tex(0, "colour_rt", "rgba8unorm", 1920, 1080),
        tex(1, "depth_rt", "r32float", 1920, 1080),
    ];
    let passes = vec![
        compute_pass(0, &[r0], &[]),
        compute_pass(1, &[r1], &[]),
    ];

    let colours = run_greedy(&resources, &passes);

    assert_eq!(
        colours.len(),
        2,
        "Both textures coloured despite disjoint lifetimes",
    );
    assert_ne!(
        colours.get(&r0),
        colours.get(&r1),
        "Format-mismatched textures must get different colours",
    );
    assert_eq!(num_colors(&colours), 2, "Two distinct colours required");
}

/// Two textures with the SAME format and disjoint lifetimes do NOT
/// interfere (already tested above in the non-interfering section).
/// This test contrasts with the format-mismatch test.
#[test]
fn same_format_textures_with_disjoint_lifetimes_share_colour() {
    let r0 = ResourceHandle(0);
    let r1 = ResourceHandle(1);

    let resources = vec![
        tex(0, "rt_a", "rgba8unorm", 800, 600),
        tex(1, "rt_b", "rgba8unorm", 400, 300),
    ];
    let passes = vec![
        compute_pass(0, &[r0], &[]),
        compute_pass(2, &[r1], &[]),
    ];

    let colours = run_greedy(&resources, &passes);

    assert_eq!(
        *colours.get(&r0).unwrap(),
        0,
        "R0 gets colour 0",
    );
    assert_eq!(
        *colours.get(&r1).unwrap(),
        0,
        "R1 gets colour 0 (same format, disjoint lifetimes)",
    );
}

/// Buffers never format-interfere; they only interfere via lifetime overlap.
#[test]
fn buffers_do_not_format_interfere() {
    let r0 = ResourceHandle(0);
    let r1 = ResourceHandle(1);
    let r2 = ResourceHandle(2);

    // Three buffers used in non-overlapping passes -> no interference.
    // All get colour 0.
    let resources = vec![
        buf(0, "buf_a", 4096),
        buf(1, "buf_b", 8192),
        buf(2, "buf_c", 16384),
    ];
    let passes = vec![
        compute_pass(0, &[r0], &[]),
        compute_pass(1, &[r1], &[]),
        compute_pass(2, &[r2], &[]),
    ];

    let colours = run_greedy(&resources, &passes);

    assert_eq!(*colours.get(&r0).unwrap(), 0, "R0 gets colour 0");
    assert_eq!(*colours.get(&r1).unwrap(), 0, "R1 gets colour 0 (no format interference)");
    assert_eq!(*colours.get(&r2).unwrap(), 0, "R2 gets colour 0");
    assert_eq!(num_colors(&colours), 1, "One colour for all buffers");
}

// =============================================================================
// SECTION 11 -- num_colour edge cases and counting
// =============================================================================

/// num_colours returns correct count for a map with distinct colour values.
#[test]
fn num_colors_multiple_distinct() {
    let mut map = HashMap::new();
    map.insert(ResourceHandle(0), 0);
    map.insert(ResourceHandle(1), 1);
    map.insert(ResourceHandle(2), 2);

    assert_eq!(num_colors(&map), 3, "Three distinct colours");
}

/// num_colours returns correct count when some colours are reused.
#[test]
fn num_colors_with_reused_colours() {
    let mut map = HashMap::new();
    map.insert(ResourceHandle(0), 0);
    map.insert(ResourceHandle(1), 1);
    map.insert(ResourceHandle(2), 0); // reused
    map.insert(ResourceHandle(3), 1); // reused

    assert_eq!(num_colors(&map), 2, "Two distinct colours (0 and 1)");
}

/// num_colours with a single colour value across many handles.
#[test]
fn num_colors_single_colour_many_handles() {
    let mut map = HashMap::new();
    for i in 0..100 {
        map.insert(ResourceHandle(i), 0u32);
    }
    assert_eq!(num_colors(&map), 1, "All resources share colour 0");
}

// =============================================================================
// SECTION 12 -- Mixed resource types
// =============================================================================

/// Buffers and textures mixed: buffers and textures can interfere if
/// lifetimes overlap.
#[test]
fn mixed_buffer_and_texture_interference() {
    let r0 = ResourceHandle(0);
    let r1 = ResourceHandle(1);

    // Buffer and texture with overlapping lifetimes -> interfere.
    let resources = vec![
        buf(0, "storage", 65536),
        tex(1, "albedo", "rgba8unorm", 100, 100),
    ];
    let passes = vec![compute_pass(0, &[r0, r1], &[])];

    let colours = run_greedy(&resources, &passes);

    assert_eq!(colours.len(), 2);
    assert_ne!(
        colours.get(&r0),
        colours.get(&r1),
        "Buffer and texture with overlapping lifetimes get distinct colours",
    );
}

/// Buffer and texture with disjoint lifetimes do NOT interfere (buffer has
/// no format to mismatch).
#[test]
fn mixed_buffer_and_texture_disjoint_lifetimes() {
    let r0 = ResourceHandle(0);
    let r1 = ResourceHandle(1);

    let resources = vec![
        buf(0, "storage", 65536),
        tex(1, "albedo", "rgba8unorm", 100, 100),
    ];
    let passes = vec![
        compute_pass(0, &[r0], &[]),
        compute_pass(1, &[r1], &[]),
    ];

    let colours = run_greedy(&resources, &passes);

    // No interference -> both can share colour 0.
    assert_eq!(*colours.get(&r0).unwrap(), 0, "Buffer gets colour 0");
    assert_eq!(*colours.get(&r1).unwrap(), 0, "Texture gets colour 0 (disjoint lifetimes with buffer)");
    assert_eq!(num_colors(&colours), 1, "One colour");
}

// =============================================================================
// SECTION 13 -- Resources with no lifetime still get a colour
// =============================================================================

/// greedy_color_resources iterates over all resources in the input slice,
/// not just those with lifetime entries. A resource never touched by any
/// pass has no neighbours in the interference graph, so it receives colour 0.
#[test]
fn resource_without_lifetime_still_gets_colour() {
    let r0 = ResourceHandle(0);
    let r1 = ResourceHandle(1);

    // R1 is never used in any pass -> no lifetime entries -> no neighbours.
    let resources = vec![
        buf(0, "used", 4096),
        buf(1, "unused", 8192),
    ];
    let passes = vec![compute_pass(0, &[r0], &[])];

    let lifetimes = compute_lifetimes(&passes, &[], &resources);
    let ig = InterferenceGraph::build(&resources, &lifetimes);
    let colours = greedy_color_resources(&ig, &resources);

    assert_eq!(
        colours.len(),
        2,
        "Both resources appear in colour map (greedy colors all input resources)",
    );
    assert_eq!(
        *colours.get(&r0).unwrap(),
        0u32,
        "R0 gets colour 0",
    );
    assert_eq!(
        *colours.get(&r1).unwrap(),
        0,
        "R1 gets colour 0 (no neighbours to force a different colour)",
    );
}

// =============================================================================
// SECTION 14 -- Interference graph construction from compute_lifetimes
// =============================================================================

/// Interference graph built from passes where resources are read vs. written
/// produces correct edges.
#[test]
fn interference_graph_reflects_read_write_lifetimes() {
    let r0 = ResourceHandle(0);
    let r1 = ResourceHandle(1);
    let r2 = ResourceHandle(2);

    // R0 written by p0, read by p2    -> lifetime (0, 2)
    // R1 read by p1, read by p2        -> lifetime (1, 2)
    // R2 written by p2                  -> lifetime (2, 2)
    //
    // R0 and R1: overlap (R0.last=2 >= R1.first=1, R1.last=2 >= R0.first=0) -> interfere
    // R0 and R2: overlap (both at p2) -> interfere
    // R1 and R2: overlap (both at p2) -> interfere
    let resources = vec![
        buf(0, "write_then_read", 4096),
        buf(1, "read_twice", 4096),
        buf(2, "write_only", 4096),
    ];
    let passes = vec![
        compute_pass(0, &[], &[r0]),
        compute_pass(1, &[r1], &[]),
        compute_pass(2, &[r0, r1, r2], &[r2]),
    ];

    let ig = build_interference(&resources, &passes);

    // All pairs interfere.
    assert!(ig.interfere(r0, r1), "R0 and R1 interfere");
    assert!(ig.interfere(r0, r2), "R0 and R2 interfere");
    assert!(ig.interfere(r1, r2), "R1 and R2 interfere");

    // Greedy coloring gives all three distinct colours.
    let _lifetimes = compute_lifetimes(&passes, &[], &resources);
    let colours = greedy_color_resources(&ig, &resources);
    assert_eq!(num_colors(&colours), 3);
}

/// Copy pass lifetimes contribute to interference graph correctly.
#[test]
fn copy_pass_lifetimes_in_interference_graph() {
    let r0 = ResourceHandle(0);
    let r1 = ResourceHandle(1);
    let r2 = ResourceHandle(2);

    // p0: copy R0 -> R1
    // p1: copy R1 -> R2
    //
    // R0 lifetime: (0, 0)
    // R1 lifetime: (0, 1) -- read by p0, written by p1
    // R2 lifetime: (1, 1)
    //
    // R0 and R1: overlap (both in p0) -> interfere
    // R1 and R2: overlap (both in p1) -> interfere
    // R0 and R2: no overlap (0<1, 0<1) -> no interference
    let resources = vec![
        buf(0, "src", 4096),
        buf(1, "intermediate", 4096),
        buf(2, "dst", 4096),
    ];
    let passes = vec![
        copy_pass(0, r0, r1),
        copy_pass(1, r1, r2),
    ];

    let colours = run_greedy(&resources, &passes);

    assert_eq!(colours.len(), 3);
    assert_eq!(*colours.get(&r0).unwrap(), 0, "R0 gets colour 0");
    assert_ne!(*colours.get(&r1).unwrap(), *colours.get(&r0).unwrap(), "R1 diff from R0");
    assert_eq!(*colours.get(&r2).unwrap(), 0, "R2 reuses colour 0 (no interference with R0)");
}

// =============================================================================
// SECTION 15 -- Large-scale: many non-interfering resources
// =============================================================================

/// A large number of non-interfering resources all get colour 0.
#[test]
fn many_non_interfering_resources_single_colour() {
    let n: usize = 50;
    let resources: Vec<IrResource> = (0..n)
        .map(|i| buf(i as u32, &format!("res_{}", i), 1024))
        .collect();
    let handles: Vec<ResourceHandle> = (0..n).map(|i| ResourceHandle(i as u32)).collect();

    // Each resource used in its own dedicated pass -> no interference.
    let passes: Vec<IrPass> = (0..n)
        .map(|i| compute_pass(i, &[ResourceHandle(i as u32)], &[]))
        .collect();

    let colours = run_greedy(&resources, &passes);

    assert_eq!(colours.len(), n, "All {} resources coloured", n);
    assert_eq!(num_colors(&colours), 1, "All share colour 0");

    // Verify every resource has colour 0.
    for h in &handles {
        assert_eq!(
            colours.get(h),
            Some(&0),
            "Resource {:?} has colour 0",
            h,
        );
    }
}

/// Colours are non-negative and sequential from 0 with no gaps -- greedy
/// always picks the smallest available colour.
#[test]
fn colours_are_sequential_without_gaps() {
    // Build a graph where each new resource interferes with all previous ones.
    let n = 8;
    let resources: Vec<IrResource> = (0..n)
        .map(|i| buf(i, &format!("r{}", i), 4096))
        .collect();
    let handles: Vec<ResourceHandle> = (0..n).map(ResourceHandle).collect();

    // All used by the same pass -> complete interference subgraph.
    let passes = vec![compute_pass(0, &handles, &[])];

    let colours = run_greedy(&resources, &passes);

    assert_eq!(num_colors(&colours), n as u32, "{} distinct colours", n);

    // Colours should be exactly {0, 1, ..., n-1}.
    let mut vals: Vec<u32> = colours.values().copied().collect();
    vals.sort();
    let expected: Vec<u32> = (0..n as u32).collect();
    assert_eq!(vals, expected, "Colours are 0..{} without gaps", n - 1);
}

// =============================================================================
// SECTION 16 -- Interference graph edge cases
// =============================================================================

/// `interfere` returns false for resources not in the graph.
#[test]
fn interfere_returns_false_for_unknown_handle() {
    let r0 = ResourceHandle(0);
    let r1 = ResourceHandle(1);

    let resources = vec![buf(0, "only_res", 1024)];
    let passes = vec![compute_pass(0, &[r0], &[])];

    let ig = build_interference(&resources, &passes);

    // R1 was never registered -> interfere should be false.
    assert!(
        !ig.interfere(r0, r1),
        "interfere returns false when one handle is unknown",
    );
    assert!(
        !ig.interfere(r1, r0),
        "interfere is symmetric; unknown handle returns false",
    );
    assert!(
        !ig.interfere(r1, r1),
        "interfere for two unknown handles returns false",
    );
}

/// `neighbors` returns an empty slice for a handle not in the graph.
#[test]
fn neighbors_empty_for_unknown_handle() {
    let r0 = ResourceHandle(0);
    let r1 = ResourceHandle(42);

    let resources = vec![buf(0, "res", 1024)];
    let passes = vec![compute_pass(0, &[r0], &[])];

    let ig = build_interference(&resources, &passes);

    assert!(
        ig.neighbors(r1).is_empty(),
        "neighbors returns empty slice for unknown handle",
    );
    assert!(
        ig.neighbors(ResourceHandle(999)).is_empty(),
        "neighbors returns empty for non-existent handle",
    );
}

// =============================================================================
// SECTION 17 -- Correctness: no two interfering resources share a colour
// =============================================================================

/// Invariant check: for every pair of interfering resources in the graph,
/// their assigned colours must be different.
#[test]
fn interfering_resources_never_share_a_colour() {
    let r0 = ResourceHandle(0);
    let r1 = ResourceHandle(1);
    let r2 = ResourceHandle(2);
    let r3 = ResourceHandle(3);
    let r4 = ResourceHandle(4);

    // Mix of interference patterns:
    // p0 touches R0, R1, R2  -> R0,R1,R2 all interfere
    // p1 touches R2, R3       -> R2,R3 interfere
    // p2 touches R3, R4       -> R3,R4 interfere
    // R0 and R4 have no direct overlap and same format (none for buffers).
    let resources = vec![
        buf(0, "a", 1_000_000),
        buf(1, "b", 800_000),
        buf(2, "c", 600_000),
        buf(3, "d", 400_000),
        buf(4, "e", 200_000),
    ];
    let passes = vec![
        compute_pass(0, &[r0, r1, r2], &[]),
        compute_pass(1, &[r2, r3], &[]),
        compute_pass(2, &[r3, r4], &[]),
    ];

    let colours = run_greedy(&resources, &passes);
    let ig = build_interference(&resources, &passes);

    // Verify the invariant for every pair in the colour map.
    for (&ha, &ca) in &colours {
        for (&hb, &cb) in &colours {
            if ha < hb && ig.interfere(ha, hb) {
                assert_ne!(
                    ca, cb,
                    "Interfering resources {:?} and {:?} must not share colour {}",
                    ha, hb, ca,
                );
            }
        }
    }
}
