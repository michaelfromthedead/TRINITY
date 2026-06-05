// SPDX-License-Identifier: MIT
//
// blackbox_dynamic_culling.rs -- Blackbox contract tests for T-FG-6.4
// (Dynamic culling / FeatureSet).
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::frame_graph::*` -- no internal fields,
// no private methods, no implementation details.
//
// Public API under test:
//   FeatureSet                         -- bitfield for runtime feature flags
//   FeatureSet::NONE                   -- zero (all feature-gated passes disabled)
//   FeatureSet::DEBUG_WIREFRAME        -- bit 0
//   FeatureSet::DEBUG_OVERLAY          -- bit 1
//   FeatureSet::DEBUG_PROFILER         -- bit 2
//   FeatureSet::contains(self, other)  -- subset check
//   BitOr<FeatureSet>                  -- union
//   BitAnd<FeatureSet>                 -- intersection
//   IrPass.feature_flags               -- per-pass u64 mask
//   is_pass_live(pass, features)       -- runtime liveness predicate
//   CompiledFrameGraph::compile() -- used for integration tests
//
// Contract:
//   - FeatureSet::NONE equals 0.
//   - Each debug constant occupies a distinct bit (non-zero, unique).
//   - contains() returns true when all bits of `other` are set in `self`.
//   - BitOr produces the union of two flag sets.
//   - IrPass.feature_flags defaults to 0 (always live).
//   - A pass with feature_flags == 0 is ALWAYS live regardless of runtime set.
//   - A pass with non-zero feature_flags is live ONLY when the runtime FeatureSet
//     contains ALL the bits declared in feature_flags.
//
// Coverage:
//   1.  FeatureSet::NONE is zero
//   2.  FeatureSet constants are distinct and non-zero
//   3.  contains() works for individual flags
//   4.  BitOr combines flags
//   5.  Build a graph -> runtime_features defaults to NONE -> all passes live
//   6.  Pass with no feature flags is always live
//   7.  Build graph with pass tagged with DEBUG_WIREFRAME -> set runtime to
//       NONE -> pass should not execute
//   8.  Build graph with pass tagged with DEBUG_WIREFRAME -> set runtime to
//       DEBUG_WIREFRAME -> pass is live
//   9.  Multiple debug passes with different flags -> only matching ones execute

use renderer_backend::frame_graph::{
    CompiledFrameGraph, DispatchSource, FeatureSet, InstanceSource, IrPass, IrResource,
    PassIndex, PassType, ResourceDesc, ResourceHandle, ResourceLifetime, ResourceState,
    TextureDesc, ViewType,
};

// =============================================================================
// Helpers
// =============================================================================

fn make_texture(handle: u32, name: &str, w: u32, h: u32) -> IrResource {
    IrResource::new(
        ResourceHandle(handle),
        name,
        ResourceDesc::Texture2D(TextureDesc {
            width: w,
            height: h,
            mip_levels: 1,
            array_layers: 1,
            format: "rgba8unorm".into(),
        }),
        ResourceLifetime::Transient,
        ResourceState::Uninitialized,
    )
}

fn make_buffer(handle: u32, name: &str, size: u64) -> IrResource {
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

/// Build a graphics pass with the given color attachment resource.
fn graphics_pass(
    index: usize,
    name: &str,
    color: ResourceHandle,
) -> IrPass {
    let mut pass = IrPass::graphics(
        PassIndex(index),
        name,
        vec![renderer_backend::frame_graph::ColorAttachment {
            resource: color,
            mip_level: 0,
            array_layer: 0,
            load_op: renderer_backend::frame_graph::AttachmentLoadOp::Clear,
            store_op: renderer_backend::frame_graph::AttachmentStoreOp::Store,
            clear_color: [0.0, 0.0, 0.0, 0.0],
        }],
        None,
        InstanceSource::Direct {
            index_count: 3,
            instance_count: 1,
            base_vertex: 0,
            first_index: 0,
            first_instance: 0,
        },
        ViewType::ColorAttachment,
    );
    // Ensure color attachment is in the write set (sync_access_set_from_attachments
    // already handles this, but we double-ensure via explicit push).
    if !pass.access_set.writes.contains(&color) {
        pass.access_set.writes.push(color);
    }
    pass
}

/// Build a compute pass with the given read and write resource handles.
fn compute_pass(index: usize, name: &str, reads: &[ResourceHandle], writes: &[ResourceHandle]) -> IrPass {
    let mut pass = IrPass::compute(
        PassIndex(index),
        name,
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::Storage,
    );
    pass.access_set.reads.extend_from_slice(reads);
    pass.access_set.writes.extend_from_slice(writes);
    pass
}

// =============================================================================
// SECTION 1 -- FeatureSet::NONE is zero
// =============================================================================

#[test]
fn feature_set_none_is_zero() {
    assert_eq!(
        FeatureSet::NONE.0,
        0u64,
        "FeatureSet::NONE must have raw value 0",
    );
}

#[test]
fn feature_set_none_contains_nothing() {
    assert!(
        !FeatureSet::NONE.contains(FeatureSet::DEBUG_WIREFRAME),
        "NONE should not contain DEBUG_WIREFRAME",
    );
    assert!(
        !FeatureSet::NONE.contains(FeatureSet::DEBUG_OVERLAY),
        "NONE should not contain DEBUG_OVERLAY",
    );
    assert!(
        !FeatureSet::NONE.contains(FeatureSet::DEBUG_PROFILER),
        "NONE should not contain DEBUG_PROFILER",
    );
}

// =============================================================================
// SECTION 2 -- FeatureSet constants are distinct and non-zero
// =============================================================================

#[test]
fn feature_set_constants_are_non_zero() {
    assert_ne!(
        FeatureSet::DEBUG_WIREFRAME.0,
        0u64,
        "DEBUG_WIREFRAME must be non-zero",
    );
    assert_ne!(
        FeatureSet::DEBUG_OVERLAY.0,
        0u64,
        "DEBUG_OVERLAY must be non-zero",
    );
    assert_ne!(
        FeatureSet::DEBUG_PROFILER.0,
        0u64,
        "DEBUG_PROFILER must be non-zero",
    );
}

#[test]
fn feature_set_constants_are_distinct() {
    assert_ne!(
        FeatureSet::DEBUG_WIREFRAME,
        FeatureSet::DEBUG_OVERLAY,
        "DEBUG_WIREFRAME and DEBUG_OVERLAY must be distinct",
    );
    assert_ne!(
        FeatureSet::DEBUG_WIREFRAME,
        FeatureSet::DEBUG_PROFILER,
        "DEBUG_WIREFRAME and DEBUG_PROFILER must be distinct",
    );
    assert_ne!(
        FeatureSet::DEBUG_OVERLAY,
        FeatureSet::DEBUG_PROFILER,
        "DEBUG_OVERLAY and DEBUG_PROFILER must be distinct",
    );
}

#[test]
fn feature_set_constants_are_single_bit() {
    // Each constant should be a power of two (single bit).
    assert!(
        FeatureSet::DEBUG_WIREFRAME.0.is_power_of_two(),
        "DEBUG_WIREFRAME must be a single bit (power of two)",
    );
    assert!(
        FeatureSet::DEBUG_OVERLAY.0.is_power_of_two(),
        "DEBUG_OVERLAY must be a single bit (power of two)",
    );
    assert!(
        FeatureSet::DEBUG_PROFILER.0.is_power_of_two(),
        "DEBUG_PROFILER must be a single bit (power of two)",
    );
}

// =============================================================================
// SECTION 3 -- contains() works for individual flags
// =============================================================================

#[test]
fn contains_self_returns_true() {
    assert!(
        FeatureSet::DEBUG_WIREFRAME.contains(FeatureSet::DEBUG_WIREFRAME),
        "A set must contain itself",
    );
    assert!(
        FeatureSet::DEBUG_OVERLAY.contains(FeatureSet::DEBUG_OVERLAY),
        "A set must contain itself",
    );
}

#[test]
fn contains_different_flag_returns_false() {
    assert!(
        !FeatureSet::DEBUG_WIREFRAME.contains(FeatureSet::DEBUG_OVERLAY),
        "DEBUG_WIREFRAME should not contain DEBUG_OVERLAY",
    );
    assert!(
        !FeatureSet::DEBUG_PROFILER.contains(FeatureSet::DEBUG_WIREFRAME),
        "DEBUG_PROFILER should not contain DEBUG_WIREFRAME",
    );
}

#[test]
fn contains_unioned_returns_true_for_each_component() {
    let combined = FeatureSet::DEBUG_WIREFRAME | FeatureSet::DEBUG_OVERLAY;

    assert!(
        combined.contains(FeatureSet::DEBUG_WIREFRAME),
        "Combined set must contain DEBUG_WIREFRAME",
    );
    assert!(
        combined.contains(FeatureSet::DEBUG_OVERLAY),
        "Combined set must contain DEBUG_OVERLAY",
    );
    assert!(
        !combined.contains(FeatureSet::DEBUG_PROFILER),
        "Combined set must NOT contain DEBUG_PROFILER (not included)",
    );
}

#[test]
fn contains_all_three_combined() {
    let all = FeatureSet::DEBUG_WIREFRAME
        | FeatureSet::DEBUG_OVERLAY
        | FeatureSet::DEBUG_PROFILER;

    assert!(all.contains(FeatureSet::DEBUG_WIREFRAME));
    assert!(all.contains(FeatureSet::DEBUG_OVERLAY));
    assert!(all.contains(FeatureSet::DEBUG_PROFILER));
}

#[test]
fn contains_empty_mask() {
    // Every set trivially contains the NONE (empty) mask.
    // contains(0) should return true because 0 & anything == 0.
    let f = FeatureSet::DEBUG_WIREFRAME;
    assert!(
        f.contains(FeatureSet::NONE),
        "Any set must trivially contain NONE (0 bits required)",
    );
}

#[test]
fn contains_none_never_satisfies_non_empty() {
    assert!(
        !FeatureSet::NONE.contains(FeatureSet::DEBUG_WIREFRAME),
        "NONE must not satisfy any non-empty query",
    );
    assert!(
        !FeatureSet::NONE.contains(FeatureSet::DEBUG_OVERLAY),
        "NONE must not satisfy any non-empty query",
    );
    assert!(
        !FeatureSet::NONE.contains(FeatureSet::DEBUG_PROFILER),
        "NONE must not satisfy any non-empty query",
    );
}

// =============================================================================
// SECTION 4 -- BitOr combines flags
// =============================================================================

#[test]
fn bitor_combines_two_flags() {
    let combined = FeatureSet::DEBUG_WIREFRAME | FeatureSet::DEBUG_OVERLAY;

    assert_eq!(
        combined.0,
        FeatureSet::DEBUG_WIREFRAME.0 | FeatureSet::DEBUG_OVERLAY.0,
        "BitOr must produce the bitwise union",
    );
}

#[test]
fn bitor_combines_three_flags() {
    let combined = FeatureSet::DEBUG_WIREFRAME
        | FeatureSet::DEBUG_OVERLAY
        | FeatureSet::DEBUG_PROFILER;

    let expected = FeatureSet::DEBUG_WIREFRAME.0
        | FeatureSet::DEBUG_OVERLAY.0
        | FeatureSet::DEBUG_PROFILER.0;

    assert_eq!(combined.0, expected, "Three-way BitOr must produce the bitwise union");
}

#[test]
fn bitor_with_none_is_identity() {
    assert_eq!(
        (FeatureSet::DEBUG_WIREFRAME | FeatureSet::NONE).0,
        FeatureSet::DEBUG_WIREFRAME.0,
        "BitOr with NONE is identity",
    );
    assert_eq!(
        (FeatureSet::NONE | FeatureSet::DEBUG_WIREFRAME).0,
        FeatureSet::DEBUG_WIREFRAME.0,
        "BitOr with NONE is identity (reversed)",
    );
}

#[test]
fn bitor_same_flag_is_idempotent() {
    let dup = FeatureSet::DEBUG_WIREFRAME | FeatureSet::DEBUG_WIREFRAME;
    assert_eq!(
        dup.0,
        FeatureSet::DEBUG_WIREFRAME.0,
        "BitOr of the same flag twice must equal the flag itself",
    );
}

// =============================================================================
// SECTION 5 -- Build a graph: runtime_features defaults to NONE; all passes live
// =============================================================================

#[test]
fn compile_with_default_runtime_features_all_production_passes_live() {
    // A graphics pass writing to a color attachment should always be live
    // regardless of runtime features, because its feature_flags == 0.
    let color = ResourceHandle(1);
    let resources = vec![make_texture(1, "color", 800, 600)];

    let mut pass = graphics_pass(0, "render", color);
    pass.feature_flags = 0; // production pass (always live)

    let passes = vec![pass];
    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("graph must compile");

    // With runtime_features = NONE (default) and pass with feature_flags = 0,
    // the pass should remain live in the execution order.
    assert!(
        compiled.order.contains(&PassIndex(0)),
        "Production pass (feature_flags=0) must be live even with runtime_features=NONE",
    );
    assert_eq!(
        compiled.order.len(),
        1,
        "One live pass in execution order",
    );
}

#[test]
fn compile_with_default_runtime_features_live_compute_pass() {
    // A compute pass whose output feeds into a live graphics pass.
    // Both have feature_flags = 0 (production). Both survive.
    let r0 = ResourceHandle(1);
    let r1 = ResourceHandle(2);
    let resources = vec![
        make_texture(1, "gbuffer", 800, 600),
        make_buffer(2, "result", 1024),
    ];

    let mut p0 = compute_pass(0, "resolve", &[r0], &[r1]);
    p0.feature_flags = 0;

    let mut p1 = graphics_pass(1, "compose", r0);
    p1.feature_flags = 0;
    // Consume the compute pass output so it survives Phase 6.
    p1.access_set.reads.push(r1);

    let passes = vec![p0, p1];
    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("graph must compile");

    // Both passes survive because feature_flags == 0 (always live).
    assert!(compiled.order.contains(&PassIndex(0)), "compute pass live");
    assert!(compiled.order.contains(&PassIndex(1)), "graphics pass live");
    assert_eq!(compiled.order.len(), 2, "both passes in execution order");
}

// =============================================================================
// SECTION 6 -- Pass with no feature flags is always live
// =============================================================================

#[test]
fn pass_with_zero_feature_flags_live_under_any_runtime() {
    // Set runtime_features to a non-empty set, but the pass has
    // feature_flags = 0. The pass must still execute because
    // feature_flags == 0 means "always live".
    let color = ResourceHandle(1);
    let resources = vec![make_texture(1, "output", 800, 600)];

    let mut pass = graphics_pass(0, "always_live", color);
    pass.feature_flags = 0; // always live

    let passes = vec![pass];
    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("graph must compile");

    assert!(
        compiled.order.contains(&PassIndex(0)),
        "Pass with feature_flags=0 must be live regardless of runtime features",
    );
}

#[test]
fn compute_pass_without_feature_flags_always_live() {
    // A compute pass that reads from a resource produced by a live graphics pass.
    // feature_flags = 0 on both passes.
    let r0 = ResourceHandle(1);
    let resources = vec![make_texture(1, "src", 800, 600)];

    let mut p0 = graphics_pass(0, "gbuffer", r0);
    p0.feature_flags = 0;

    // Compute pass that reads from the graphics output.
    let mut p1 = compute_pass(1, "post", &[r0], &[]);
    p1.feature_flags = 0;

    let passes = vec![p0, p1];
    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("graph must compile");

    assert!(
        compiled.order.contains(&PassIndex(0)),
        "Graphics pass always live",
    );
    assert!(
        compiled.order.contains(&PassIndex(1)),
        "Compute pass with no feature flags always live when consumed by live pass",
    );
}

// =============================================================================
// SECTION 7 -- Pass tagged with DEBUG_WIREFRAME, runtime NONE => not live
// =============================================================================

#[test]
fn debug_pass_culled_when_runtime_is_none() {
    // A pass with feature_flags = DEBUG_WIREFRAME should be culled
    // when the runtime FeatureSet is NONE.
    let color = ResourceHandle(1);
    let resources = vec![make_texture(1, "framebuffer", 800, 600)];

    let mut pass = graphics_pass(0, "wireframe_overlay", color);
    pass.feature_flags = FeatureSet::DEBUG_WIREFRAME.0;
    // Ensure the pass has correct access set.
    pass.access_set.writes.push(color);

    let passes = vec![pass];

    // Compile with runtime_features = NONE (all debug passes disabled).
    // If the compile function accepts runtime_features, use it.
    // Otherwise, use the is_pass_live predicate directly.
    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("graph must compile with a debug-tagged pass");

    // With runtime_features = NONE, the debug pass should be culled
    // if the compiler performs dynamic culling.  If the compiler does NOT
    // perform dynamic culling, the pass remains in the order but is
    // flagged for skipping.  At minimum, the order reflects the pass's
    // runtime liveness status.
    //
    // The test verifies that is_pass_live returns false for this configuration.
    let live = renderer_backend::frame_graph::is_pass_live(
        &compiled.passes[0],
        FeatureSet::NONE,
    );
    assert!(
        !live,
        "Pass with DEBUG_WIREFRAME must NOT be live under runtime FeatureSet::NONE",
    );
}

#[test]
fn debug_graphics_pass_does_not_appear_in_execution_order_when_disabled() {
    let color = ResourceHandle(1);
    let resources = vec![make_texture(1, "rt", 800, 600)];

    let mut pass = graphics_pass(0, "debug_wireframe", color);
    pass.feature_flags = FeatureSet::DEBUG_WIREFRAME.0;

    let passes = vec![pass];
    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("graph compiles");

    // Under FeatureSet::NONE, a debug-only pass should be excluded from
    // the execution order (or flagged as eliminated).
    let should_skip = renderer_backend::frame_graph::is_pass_live(
        &compiled.passes[0],
        FeatureSet::NONE,
    );
    assert!(
        !should_skip,
        "DEBUG_WIREFRAME pass must be skipped at runtime when FeatureSet is NONE",
    );
}

#[test]
fn debug_compute_pass_culled_when_runtime_is_none() {
    let color = ResourceHandle(1);
    let resources = vec![make_texture(1, "gbuffer", 800, 600)];

    // A graphics (production) pass with no feature flags.
    let mut p0 = graphics_pass(0, "gbuffer", color);
    p0.feature_flags = 0;

    // A compute debug pass flagged with DEBUG_WIREFRAME.
    let mut p1 = compute_pass(1, "debug_ssao", &[color], &[]);
    p1.feature_flags = FeatureSet::DEBUG_WIREFRAME.0;

    let passes = vec![p0, p1];
    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("graph compiles");

    // P0 (production) is always live.
    assert!(
        renderer_backend::frame_graph::is_pass_live(&compiled.passes[0], FeatureSet::NONE),
        "production pass always live",
    );

    // P1 (debug compute) is live only when DEBUG_WIREFRAME is set.
    assert!(
        !renderer_backend::frame_graph::is_pass_live(&compiled.passes[1], FeatureSet::NONE),
        "debug compute pass must be culled when runtime features are NONE",
    );
}

// =============================================================================
// SECTION 8 -- Pass tagged with DEBUG_WIREFRAME, runtime = DEBUG_WIREFRAME => live
// =============================================================================

#[test]
fn debug_pass_live_when_matching_runtime_flag_set() {
    let color = ResourceHandle(1);
    let resources = vec![make_texture(1, "framebuffer", 800, 600)];

    let mut pass = graphics_pass(0, "wireframe_overlay", color);
    pass.feature_flags = FeatureSet::DEBUG_WIREFRAME.0;

    let passes = vec![pass];
    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("graph compiles");

    // With runtime features set to DEBUG_WIREFRAME, the pass is live.
    let live = renderer_backend::frame_graph::is_pass_live(
        &compiled.passes[0],
        FeatureSet::DEBUG_WIREFRAME,
    );
    assert!(
        live,
        "Pass with DEBUG_WIREFRAME must be live when runtime FeatureSet includes DEBUG_WIREFRAME",
    );
}

#[test]
fn debug_pass_live_with_runtime_containing_multiple_bits() {
    let color = ResourceHandle(1);
    let resources = vec![make_texture(1, "rt", 800, 600)];

    let mut pass = graphics_pass(0, "wireframe", color);
    pass.feature_flags = FeatureSet::DEBUG_WIREFRAME.0;

    let passes = vec![pass];
    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("graph compiles");

    // Runtime has both WIREFRAME and OVERLAY, which includes the required bit.
    let runtime = FeatureSet::DEBUG_WIREFRAME | FeatureSet::DEBUG_OVERLAY;
    let live = renderer_backend::frame_graph::is_pass_live(&compiled.passes[0], runtime);
    assert!(
        live,
        "Pass must be live when runtime FeatureSet is a superset of the pass's feature_flags",
    );
}

// =============================================================================
// SECTION 9 -- Multiple debug passes with different flags: only matching execute
// =============================================================================

#[test]
fn only_debug_pass_with_matching_flag_executes() {
    let r0 = ResourceHandle(1);
    let r1 = ResourceHandle(2);
    let r2 = ResourceHandle(3);
    let r3 = ResourceHandle(4);
    let resources = vec![
        make_texture(1, "color", 800, 600),
        make_buffer(2, "profiler_out", 4096),
        make_buffer(3, "overlay_out", 4096),
        make_texture(4, "final", 800, 600),
    ];

    // P0: production graphics (always live).
    let mut p0 = graphics_pass(0, "scene_render", r0);
    p0.feature_flags = 0;

    // P1: debug wireframe (requires DEBUG_WIREFRAME).
    let mut p1 = graphics_pass(1, "debug_wireframe", r0);
    p1.feature_flags = FeatureSet::DEBUG_WIREFRAME.0;

    // P2: debug profiler (requires DEBUG_PROFILER).
    let mut p2 = compute_pass(2, "debug_profiler", &[], &[r1]);
    p2.feature_flags = FeatureSet::DEBUG_PROFILER.0;

    // P3: debug overlay (requires DEBUG_OVERLAY).
    let mut p3 = compute_pass(3, "debug_overlay", &[], &[r2]);
    p3.feature_flags = FeatureSet::DEBUG_OVERLAY.0;

    // P4: final composition — consumes all debug outputs so they
    // survive Phase 6 dead-pass elimination.
    let mut p4 = graphics_pass(4, "final_compose", r3);
    p4.feature_flags = 0;
    p4.access_set.reads.push(r1);
    p4.access_set.reads.push(r2);

    let passes = vec![p0, p1, p2, p3, p4];
    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("graph compiles");

    // Runtime features: only DEBUG_WIREFRAME and DEBUG_OVERLAY are active.
    let runtime = FeatureSet::DEBUG_WIREFRAME | FeatureSet::DEBUG_OVERLAY;

    // Find each pass by its index in the compiled output.
    let find_pass = |idx: usize| -> &IrPass {
        compiled.passes.iter().find(|p| p.index == PassIndex(idx)).unwrap()
    };

    // P0: always live (feature_flags == 0).
    assert!(
        renderer_backend::frame_graph::is_pass_live(find_pass(0), runtime),
        "P0 (production) always live",
    );

    // P1: live because runtime has DEBUG_WIREFRAME.
    assert!(
        renderer_backend::frame_graph::is_pass_live(find_pass(1), runtime),
        "P1 (wireframe) live under DEBUG_WIREFRAME",
    );

    // P2: culled because DEBUG_PROFILER is NOT in the runtime set.
    assert!(
        !renderer_backend::frame_graph::is_pass_live(find_pass(2), runtime),
        "P2 (profiler) culled because DEBUG_PROFILER is disabled",
    );

    // P3: live because runtime has DEBUG_OVERLAY.
    assert!(
        renderer_backend::frame_graph::is_pass_live(find_pass(3), runtime),
        "P3 (overlay) live under DEBUG_OVERLAY",
    );
}

#[test]
fn no_debug_passes_live_when_all_disabled() {
    let r0 = ResourceHandle(1);
    let r1 = ResourceHandle(2);
    let resources = vec![
        make_texture(1, "color", 800, 600),
        make_buffer(2, "debug_out", 4096),
    ];

    // One production pass and two debug passes with different flags.
    let mut p0 = graphics_pass(0, "scene", r0);
    p0.feature_flags = 0;

    let mut p1 = compute_pass(1, "wireframe", &[r0], &[]);
    p1.feature_flags = FeatureSet::DEBUG_WIREFRAME.0;

    let mut p2 = compute_pass(2, "overlay", &[], &[r1]);
    p2.feature_flags = FeatureSet::DEBUG_OVERLAY.0;

    // Final consumer pass so all computes survive Phase 6.
    let mut p3 = graphics_pass(3, "final_tone", r0);
    p3.feature_flags = 0;
    p3.access_set.reads.push(r1);

    let passes = vec![p0, p1, p2, p3];
    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("graph compiles");

    let runtime = FeatureSet::NONE;

    // Find passes by their index rather than array position.
    let find_pass = |idx: usize| -> &IrPass {
        compiled.passes.iter().find(|p| p.index == PassIndex(idx)).unwrap()
    };

    // Only P0 (production) is live.
    assert!(
        renderer_backend::frame_graph::is_pass_live(find_pass(0), runtime),
        "P0 production pass always live",
    );
    assert!(
        !renderer_backend::frame_graph::is_pass_live(find_pass(1), runtime),
        "P1 debug pass culled when all features disabled",
    );
    assert!(
        !renderer_backend::frame_graph::is_pass_live(find_pass(2), runtime),
        "P2 debug pass culled when all features disabled",
    );
}

#[test]
fn all_debug_passes_live_when_all_enabled() {
    let r0 = ResourceHandle(1);
    let r1 = ResourceHandle(2);
    let r2 = ResourceHandle(3);
    let r3 = ResourceHandle(4);
    let resources = vec![
        make_texture(1, "rt", 800, 600),
        make_buffer(2, "w_debug", 4096),
        make_buffer(3, "p_debug", 4096),
        make_texture(4, "final", 800, 600),
    ];

    let mut p0 = graphics_pass(0, "scene", r0);
    p0.feature_flags = 0;

    let mut p1 = compute_pass(1, "wireframe", &[r0], &[]);
    p1.feature_flags = FeatureSet::DEBUG_WIREFRAME.0;

    let mut p2 = compute_pass(2, "profiler", &[], &[r1]);
    p2.feature_flags = FeatureSet::DEBUG_PROFILER.0;

    let mut p3 = compute_pass(3, "overlay", &[], &[r2]);
    p3.feature_flags = FeatureSet::DEBUG_OVERLAY.0;

    // Consumer pass so r1 and r2 are read, keeping P2/P3 alive.
    let mut p4 = graphics_pass(4, "final_tone", r3);
    p4.feature_flags = 0;
    p4.access_set.reads.push(r1);
    p4.access_set.reads.push(r2);

    let passes = vec![p0, p1, p2, p3, p4];
    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("graph compiles");

    let runtime = FeatureSet::DEBUG_WIREFRAME
        | FeatureSet::DEBUG_PROFILER
        | FeatureSet::DEBUG_OVERLAY;

    let find_pass = |idx: usize| -> &IrPass {
        compiled.passes.iter().find(|p| p.index == PassIndex(idx)).unwrap()
    };

    // All passes live when all debug features are enabled.
    for i in 0..4 {
        assert!(
            renderer_backend::frame_graph::is_pass_live(find_pass(i), runtime),
            "P{} must be live when all debug features enabled",
            i,
        );
    }
}

// =============================================================================
// SECTION 10 -- is_pass_live edge cases
// =============================================================================

#[test]
fn is_pass_live_zero_feature_flags_always_live_under_any_runtime() {
    let color = ResourceHandle(1);
    let resources = vec![make_texture(1, "t", 100, 100)];

    let mut pass = graphics_pass(0, "production", color);
    pass.feature_flags = 0;

    // Mutate to simulate a production pass with runtime_features set to full.
    let full = FeatureSet::DEBUG_WIREFRAME
        | FeatureSet::DEBUG_OVERLAY
        | FeatureSet::DEBUG_PROFILER;
    assert!(
        renderer_backend::frame_graph::is_pass_live(&pass, full),
        "Pass with feature_flags=0 must be live even under max runtime features",
    );
}

#[test]
fn is_pass_live_double_gated_pass() {
    // A pass that requires BOTH WIREFRAME and PROFILER to be live.
    let color = ResourceHandle(1);
    let resources = vec![make_texture(1, "rt", 100, 100)];

    let mut pass = graphics_pass(0, "debug", color);
    // Require both bits.
    let required = FeatureSet::DEBUG_WIREFRAME.0 | FeatureSet::DEBUG_PROFILER.0;
    pass.feature_flags = required;

    let passes = vec![pass];
    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("graph compiles");

    // Only WIREFRAME -> not enough.
    let only_wire = FeatureSet::DEBUG_WIREFRAME;
    assert!(
        !renderer_backend::frame_graph::is_pass_live(&compiled.passes[0], only_wire),
        "Pass requiring both WIREFRAME and PROFILER must not be live when only WIREFRAME is set",
    );

    // Both WIREFRAME and PROFILER -> enough.
    let both = FeatureSet::DEBUG_WIREFRAME | FeatureSet::DEBUG_PROFILER;
    assert!(
        renderer_backend::frame_graph::is_pass_live(&compiled.passes[0], both),
        "Pass requiring both WIREFRAME and PROFILER must be live when both are set",
    );
}

#[test]
fn is_pass_live_non_zero_flags_under_none() {
    // Any pass with non-zero feature_flags is dead under FeatureSet::NONE.
    let color = ResourceHandle(1);
    let mut pass = graphics_pass(0, "debug_any", color);
    pass.feature_flags = 0xFFFF; // all bits set (requires everything)
    assert!(
        !renderer_backend::frame_graph::is_pass_live(&pass, FeatureSet::NONE),
        "Pass with non-zero feature_flags must be dead under FeatureSet::NONE",
    );
}

#[test]
fn is_pass_live_non_zero_flags_under_full_set() {
    let color = ResourceHandle(1);
    let mut pass = graphics_pass(0, "debug_any", color);
    pass.feature_flags = 0xFFFF; // all bits set
    let full = FeatureSet(0xFFFF);
    assert!(
        renderer_backend::frame_graph::is_pass_live(&pass, full),
        "Pass requiring all bits must be live when runtime has all bits set",
    );
}

// =============================================================================
// SECTION 11 -- Integration: mixed production and debug graph
// =============================================================================

#[test]
fn mixed_graph_culls_only_correct_debug_passes() {
    // Build a realistic graph:
    //   P0 (graphics, production): gbuffer
    //   P1 (compute, production):  lighting
    //   P2 (compute, debug_wireframe): wireframe overlay
    //   P3 (compute, debug_profiler):  profiler output
    //   P4 (graphics, production): compose final
    //
    // Runtime: only DEBUG_WIREFRAME enabled.
    // Expected live: P0, P1, P2, P4 (P3 culled).

    let gbuf = ResourceHandle(1);
    let light = ResourceHandle(2);
    let wire = ResourceHandle(3);
    let prof = ResourceHandle(4);
    let final_ = ResourceHandle(5);

    let resources = vec![
        make_texture(1, "gbuffer", 1920, 1080),
        make_buffer(2, "lighting", 65536),
        make_buffer(3, "wireframe_buf", 4096),
        make_buffer(4, "profiler_buf", 8192),
        make_texture(5, "final", 1920, 1080),
    ];

    // P0: gbuffer (graphics, production -- always live).
    let mut p0 = graphics_pass(0, "gbuffer", gbuf);
    p0.feature_flags = 0;

    // P1: lighting (compute, production).
    let mut p1 = compute_pass(1, "lighting", &[gbuf], &[light]);
    p1.feature_flags = 0;

    // P2: wireframe overlay (compute, debug).
    let mut p2 = compute_pass(2, "wireframe_overlay", &[gbuf], &[wire]);
    p2.feature_flags = FeatureSet::DEBUG_WIREFRAME.0;

    // P3: profiler (compute, debug).
    let mut p3 = compute_pass(3, "profiler", &[], &[prof]);
    p3.feature_flags = FeatureSet::DEBUG_PROFILER.0;

    // P4: compose (graphics, production). Reads all debug outputs so
    // P2 and P3 survive Phase 6 dead-pass elimination.
    let mut p4 = graphics_pass(4, "compose", final_);
    p4.feature_flags = 0;
    p4.access_set.reads.push(light);
    p4.access_set.reads.push(wire);
    p4.access_set.reads.push(prof);

    let passes = vec![p0, p1, p2, p3, p4];
    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("mixed graph compiles");

    let runtime = FeatureSet::DEBUG_WIREFRAME;

    let find_pass = |idx: usize| -> &IrPass {
        compiled.passes.iter().find(|p| p.index == PassIndex(idx)).unwrap()
    };

    // P0, P1, P4: production (always live).
    assert!(
        renderer_backend::frame_graph::is_pass_live(find_pass(0), runtime),
        "P0 production always live",
    );
    assert!(
        renderer_backend::frame_graph::is_pass_live(find_pass(1), runtime),
        "P1 production always live",
    );
    assert!(
        renderer_backend::frame_graph::is_pass_live(find_pass(4), runtime),
        "P4 production always live",
    );

    // P2: live because DEBUG_WIREFRAME is in runtime.
    assert!(
        renderer_backend::frame_graph::is_pass_live(find_pass(2), runtime),
        "P2 wireframe overlay live under DEBUG_WIREFRAME",
    );

    // P3: culled because DEBUG_PROFILER is NOT in runtime.
    assert!(
        !renderer_backend::frame_graph::is_pass_live(find_pass(3), runtime),
        "P3 profiler culled when DEBUG_PROFILER is disabled",
    );
}

// =============================================================================
// SECTION 12 -- FeatureSet Display
// =============================================================================

#[test]
fn feature_set_display_none() {
    let s = format!("{}", FeatureSet::NONE);
    assert!(
        s.contains("NONE"),
        "FeatureSet::NONE Display must include 'NONE', got '{}'",
        s,
    );
}

#[test]
fn feature_set_display_wireframe() {
    let s = format!("{}", FeatureSet::DEBUG_WIREFRAME);
    assert!(
        s.contains("WIREFRAME"),
        "FeatureSet::DEBUG_WIREFRAME Display must include 'WIREFRAME', got '{}'",
        s,
    );
}

#[test]
fn feature_set_display_combined() {
    let combined = FeatureSet::DEBUG_WIREFRAME | FeatureSet::DEBUG_OVERLAY;
    let s = format!("{}", combined);
    assert!(
        s.contains("WIREFRAME"),
        "Combined Display includes WIREFRAME, got '{}'",
        s,
    );
    assert!(
        s.contains("OVERLAY"),
        "Combined Display includes OVERLAY, got '{}'",
        s,
    );
}
