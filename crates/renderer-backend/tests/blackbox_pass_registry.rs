// SPDX-License-Identifier: MIT
//
// blackbox_pass_registry.rs -- Blackbox contract tests for T-FG-2.1 PassRegistry.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::frame_graph::*` -- no internal fields,
// no private methods, no implementation details.
//
// Public API under test:
//   PassRegistry::new()
//   PassRegistry::register(&mut self, pass: IrPass) -> PassIndex
//   PassRegistry::get(&self, index: PassIndex) -> Option<&IrPass>
//   PassRegistry::len(&self) -> usize
//   PassRegistry::is_empty(&self) -> bool
//   PassRegistry::iter(&self) -> std::slice::Iter<'_, IrPass>
//   Default for PassRegistry
//
// Coverage:
//   1.  new() creates an empty registry
//   2.  Default::default() creates an empty registry
//   3.  new() and Default::default() are equivalent
//   4.  register returns PassIndex(0) for the first pass
//   5.  register returns incrementing indices
//   6.  register accepts all four pass types (Graphics, Compute, Copy, RayTracing)
//   7.  get with valid index returns Some containing the registered pass
//   8.  get preserves pass identity (name, type, attachments)
//   9.  get returns None for an out-of-bounds index
//  10.  get returns None on an empty registry
//  11.  len returns 0 for a new registry
//  12.  len increments with each registration
//  13.  len is accurate after multiple registrations of mixed types
//  14.  is_empty returns true for a new registry
//  15.  is_empty returns false after a single registration
//  16.  is_empty returns true after all passes are gone (logical: re-creating)
//  17.  iter yields no items on an empty registry
//  18.  iter yields passes in registration order
//  19.  iter yields the correct number of items matching len()
//  20.  iter yields passes with correct type variants in order
//  21.  Clone preserves the full registry state
//  22.  Debug output is structurally valid
//  23.  register-and-get round-trips for a Graphics pass
//  24.  register-and-get round-trips for a Compute pass
//  25.  register-and-get round-trips for a Copy pass
//  26.  register-and-get round-trips for a RayTracing pass

use renderer_backend::frame_graph::{
    AttachmentLoadOp, AttachmentStoreOp, ColorAttachment, DepthStencilAttachment, DispatchSource,
    InstanceSource, IrPass, PassIndex, PassRegistry, ViewType,
};

// =============================================================================
// Helpers
// =============================================================================

/// A minimal graphics pass suitable for registration tests.
fn make_graphics_pass(name: &str) -> IrPass {
    IrPass::graphics(
        PassIndex(0),
        name,
        vec![ColorAttachment::default()],
        Some(DepthStencilAttachment::default()),
        InstanceSource::Direct {
            index_count: 3,
            instance_count: 1,
            base_vertex: 0,
            first_index: 0,
            first_instance: 0,
        },
        ViewType::ColorAttachment,
    )
}

/// A minimal compute pass suitable for registration tests.
fn make_compute_pass(name: &str) -> IrPass {
    IrPass::compute(
        PassIndex(0),
        name,
        DispatchSource::Direct {
            group_count_x: 8,
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::Storage,
    )
}

/// A minimal copy pass suitable for registration tests.
fn make_copy_pass(name: &str) -> IrPass {
    IrPass::copy(PassIndex(0), name)
}

/// A minimal ray-tracing pass suitable for registration tests.
fn make_ray_tracing_pass(name: &str) -> IrPass {
    IrPass::ray_tracing(
        PassIndex(0),
        name,
        DispatchSource::Direct {
            group_count_x: 16,
            group_count_y: 1,
            group_count_z: 1,
        },
    )
}

// =============================================================================
// SECTION 1 -- Construction: new(), Default, equivalence
// =============================================================================

#[test]
fn new_creates_empty_registry() {
    let reg = PassRegistry::new();
    assert_eq!(reg.len(), 0);
    assert!(reg.is_empty());
}

#[test]
fn default_creates_empty_registry() {
    let reg: PassRegistry = Default::default();
    assert_eq!(reg.len(), 0);
    assert!(reg.is_empty());
}

#[test]
fn new_and_default_are_equivalent() {
    let reg_new = PassRegistry::new();
    let reg_default: PassRegistry = Default::default();
    assert_eq!(reg_new.len(), reg_default.len());
    assert_eq!(reg_new.is_empty(), reg_default.is_empty());
}

// =============================================================================
// SECTION 2 -- register() returns correct indices
// =============================================================================

#[test]
fn register_returns_index_zero_for_first_pass() {
    let mut reg = PassRegistry::new();
    let idx = reg.register(make_graphics_pass("gbuffer"));
    assert_eq!(idx, PassIndex(0));
}

#[test]
fn register_returns_incrementing_indices() {
    let mut reg = PassRegistry::new();
    let idx0 = reg.register(make_graphics_pass("gbuffer"));
    let idx1 = reg.register(make_compute_pass("post_fx"));
    let idx2 = reg.register(make_copy_pass("copy_final"));

    assert_eq!(idx0, PassIndex(0));
    assert_eq!(idx1, PassIndex(1));
    assert_eq!(idx2, PassIndex(2));
}

#[test]
fn register_accepts_all_four_pass_types() {
    let mut reg = PassRegistry::new();

    let g_idx = reg.register(make_graphics_pass("gbuffer"));
    let c_idx = reg.register(make_compute_pass("compute_ao"));
    let cp_idx = reg.register(make_copy_pass("copy_resolve"));
    let rt_idx = reg.register(make_ray_tracing_pass("rt_reflections"));

    assert_eq!(g_idx, PassIndex(0));
    assert_eq!(c_idx, PassIndex(1));
    assert_eq!(cp_idx, PassIndex(2));
    assert_eq!(rt_idx, PassIndex(3));
}

// =============================================================================
// SECTION 3 -- get() indexed lookup
// =============================================================================

#[test]
fn get_returns_some_with_registered_pass() {
    let mut reg = PassRegistry::new();
    let idx = reg.register(make_graphics_pass("shadow_map"));
    let retrieved = reg.get(idx);

    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().name, "shadow_map");
}

#[test]
fn get_preserves_pass_identity_and_attachments() {
    let mut reg = PassRegistry::new();

    let color_att = ColorAttachment {
        resource: renderer_backend::frame_graph::ResourceHandle(42),
        mip_level: 1,
        array_layer: 0,
        load_op: AttachmentLoadOp::Clear,
        store_op: AttachmentStoreOp::Store,
        clear_color: [0.1, 0.2, 0.3, 1.0],
    };
    let ds = DepthStencilAttachment {
        resource: renderer_backend::frame_graph::ResourceHandle(7),
        depth_load_op: AttachmentLoadOp::Clear,
        depth_store_op: AttachmentStoreOp::Store,
        stencil_load_op: AttachmentLoadOp::DontCare,
        stencil_store_op: AttachmentStoreOp::DontCare,
        clear_depth: 0.5,
        clear_stencil: 0,
        depth_test_enabled: true,
        depth_write_enabled: true,
    };

    let pass = IrPass::graphics(
        PassIndex(0),
        "custom_gbuffer",
        vec![color_att.clone()],
        Some(ds.clone()),
        InstanceSource::Direct {
            index_count: 6,
            instance_count: 2,
            base_vertex: 0,
            first_index: 0,
            first_instance: 0,
        },
        ViewType::ColorAttachment,
    );

    let idx = reg.register(pass);
    let retrieved = reg.get(idx).expect("get returned None for valid index");

    assert_eq!(retrieved.name, "custom_gbuffer");
    assert_eq!(retrieved.color_attachments.len(), 1);
    assert_eq!(retrieved.color_attachments[0].resource, color_att.resource);
    assert_eq!(retrieved.color_attachments[0].load_op, AttachmentLoadOp::Clear);
    assert_eq!(retrieved.color_attachments[0].store_op, AttachmentStoreOp::Store);
    assert_eq!(retrieved.color_attachments[0].clear_color, [0.1, 0.2, 0.3, 1.0]);
    assert!(retrieved.depth_stencil.is_some());
    assert_eq!(retrieved.depth_stencil.as_ref().unwrap().resource, ds.resource);
}

#[test]
fn get_returns_none_for_out_of_bounds_index() {
    let mut reg = PassRegistry::new();
    reg.register(make_graphics_pass("pass_a"));
    reg.register(make_compute_pass("pass_b"));

    assert!(reg.get(PassIndex(2)).is_none());
    assert!(reg.get(PassIndex(10)).is_none());
    assert!(reg.get(PassIndex(usize::MAX)).is_none());
}

#[test]
fn get_returns_none_on_empty_registry() {
    let reg = PassRegistry::new();
    assert!(reg.get(PassIndex(0)).is_none());
    assert!(reg.get(PassIndex(1)).is_none());
}

// =============================================================================
// SECTION 4 -- len()
// =============================================================================

#[test]
fn len_returns_zero_for_new_registry() {
    let reg = PassRegistry::new();
    assert_eq!(reg.len(), 0);
}

#[test]
fn len_increments_with_each_registration() {
    let mut reg = PassRegistry::new();
    assert_eq!(reg.len(), 0);

    reg.register(make_graphics_pass("a"));
    assert_eq!(reg.len(), 1);

    reg.register(make_compute_pass("b"));
    assert_eq!(reg.len(), 2);

    reg.register(make_copy_pass("c"));
    assert_eq!(reg.len(), 3);

    reg.register(make_ray_tracing_pass("d"));
    assert_eq!(reg.len(), 4);
}

#[test]
fn len_accurate_after_mixed_registrations() {
    let mut reg = PassRegistry::new();
    let total = 16;

    for i in 0..total {
        let name = format!("pass_{}", i);
        match i % 4 {
            0 => reg.register(make_graphics_pass(&name)),
            1 => reg.register(make_compute_pass(&name)),
            2 => reg.register(make_copy_pass(&name)),
            _ => reg.register(make_ray_tracing_pass(&name)),
        };
    }

    assert_eq!(reg.len(), total);
}

// =============================================================================
// SECTION 5 -- is_empty()
// =============================================================================

#[test]
fn is_empty_true_for_new_registry() {
    let reg = PassRegistry::new();
    assert!(reg.is_empty());
}

#[test]
fn is_empty_false_after_single_registration() {
    let mut reg = PassRegistry::new();
    reg.register(make_graphics_pass("any"));
    assert!(!reg.is_empty());
}

#[test]
fn is_empty_false_after_multiple_registrations() {
    let mut reg = PassRegistry::new();
    reg.register(make_graphics_pass("a"));
    reg.register(make_compute_pass("b"));
    reg.register(make_copy_pass("c"));
    assert!(!reg.is_empty());
}

#[test]
fn is_empty_true_for_new_default_registry() {
    let reg: PassRegistry = Default::default();
    assert!(reg.is_empty());
}

// =============================================================================
// SECTION 6 -- iter()
// =============================================================================

#[test]
fn iter_yields_nothing_on_empty_registry() {
    let reg = PassRegistry::new();
    let collected: Vec<&IrPass> = reg.iter().collect();
    assert!(collected.is_empty());
}

#[test]
fn iter_yields_passes_in_registration_order() {
    let mut reg = PassRegistry::new();
    reg.register(make_graphics_pass("first"));
    reg.register(make_compute_pass("second"));
    reg.register(make_copy_pass("third"));

    let names: Vec<&str> = reg.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(names, vec!["first", "second", "third"]);
}

#[test]
fn iter_yields_correct_count_matching_len() {
    let mut reg = PassRegistry::new();
    reg.register(make_graphics_pass("a"));
    reg.register(make_compute_pass("b"));
    reg.register(make_copy_pass("c"));
    reg.register(make_ray_tracing_pass("d"));

    let count = reg.iter().count();
    assert_eq!(count, reg.len());
    assert_eq!(count, 4);
}

#[test]
fn iter_yields_correct_type_variants_in_order() {
    use renderer_backend::frame_graph::PassType;

    let mut reg = PassRegistry::new();
    reg.register(make_graphics_pass("g"));
    reg.register(make_compute_pass("c"));
    reg.register(make_copy_pass("cp"));
    reg.register(make_ray_tracing_pass("rt"));

    let types: Vec<PassType> = reg.iter().map(|p| p.pass_type).collect();
    assert_eq!(
        types,
        vec![PassType::Graphics, PassType::Compute, PassType::Copy, PassType::RayTracing]
    );
}

#[test]
fn iter_supports_collect_into_vec() {
    let mut reg = PassRegistry::new();
    reg.register(make_graphics_pass("a"));
    reg.register(make_compute_pass("b"));

    let passes: Vec<&IrPass> = reg.iter().collect();
    assert_eq!(passes.len(), 2);
    assert_eq!(passes[0].name, "a");
    assert_eq!(passes[1].name, "b");
}

// =============================================================================
// SECTION 7 -- Clone
// =============================================================================

#[test]
fn clone_preserves_all_passes() {
    let mut reg = PassRegistry::new();
    reg.register(make_graphics_pass("gbuffer"));
    reg.register(make_compute_pass("ao"));
    reg.register(make_copy_pass("resolve"));

    let cloned = reg.clone();

    assert_eq!(cloned.len(), 3);
    let names: Vec<&str> = cloned.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(names, vec!["gbuffer", "ao", "resolve"]);
}

#[test]
fn cloned_registry_is_independent() {
    let mut reg = PassRegistry::new();
    reg.register(make_graphics_pass("original"));

    let mut cloned = reg.clone();
    cloned.register(make_compute_pass("extra"));

    // Original should remain unchanged.
    assert_eq!(reg.len(), 1);
    assert_eq!(cloned.len(), 2);
}

// =============================================================================
// SECTION 8 -- Debug formatting
// =============================================================================

#[test]
fn debug_formatting_is_valid() {
    let mut reg = PassRegistry::new();
    reg.register(make_graphics_pass("a"));
    reg.register(make_compute_pass("b"));

    let debug_str = format!("{:?}", reg);
    assert!(!debug_str.is_empty());
    assert!(debug_str.contains("PassRegistry"));
}

// =============================================================================
// SECTION 9 -- Register-and-get round-trip for each pass type
// =============================================================================

#[test]
fn round_trip_graphics_pass() {
    let mut reg = PassRegistry::new();
    let pass = make_graphics_pass("gbuffer_main");
    let idx = reg.register(pass);
    let retrieved = reg.get(idx).expect("get returned None");

    assert_eq!(retrieved.name, "gbuffer_main");
    assert!(retrieved.pass_type == renderer_backend::frame_graph::PassType::Graphics);
}

#[test]
fn round_trip_compute_pass() {
    let mut reg = PassRegistry::new();
    let pass = make_compute_pass("cs_lighting");
    let idx = reg.register(pass);
    let retrieved = reg.get(idx).expect("get returned None");

    assert_eq!(retrieved.name, "cs_lighting");
    assert!(retrieved.pass_type == renderer_backend::frame_graph::PassType::Compute);
}

#[test]
fn round_trip_copy_pass() {
    let mut reg = PassRegistry::new();
    let pass = make_copy_pass("copy_backbuffer");
    let idx = reg.register(pass);
    let retrieved = reg.get(idx).expect("get returned None");

    assert_eq!(retrieved.name, "copy_backbuffer");
    assert!(retrieved.pass_type == renderer_backend::frame_graph::PassType::Copy);
}

#[test]
fn round_trip_ray_tracing_pass() {
    let mut reg = PassRegistry::new();
    let pass = make_ray_tracing_pass("rt_gi");
    let idx = reg.register(pass);
    let retrieved = reg.get(idx).expect("get returned None");

    assert_eq!(retrieved.name, "rt_gi");
    assert!(retrieved.pass_type == renderer_backend::frame_graph::PassType::RayTracing);
}

// =============================================================================
// SECTION 10 -- Edge cases and integration
// =============================================================================

#[test]
fn register_many_passes_retrieve_by_index() {
    let mut reg = PassRegistry::new();
    let count = 100;

    for i in 0..count {
        reg.register(make_graphics_pass(&format!("pass_{}", i)));
    }

    assert_eq!(reg.len(), count);

    // Spot-check scattered indices.
    assert_eq!(reg.get(PassIndex(0)).unwrap().name, "pass_0");
    assert_eq!(reg.get(PassIndex(50)).unwrap().name, "pass_50");
    assert_eq!(reg.get(PassIndex(99)).unwrap().name, "pass_99");
    assert!(reg.get(PassIndex(100)).is_none());
}

#[test]
fn len_is_empty_consistent_cycle() {
    let mut reg = PassRegistry::new();
    assert!(reg.is_empty());
    assert_eq!(reg.len(), 0);

    reg.register(make_graphics_pass("p"));
    assert!(!reg.is_empty());
    assert_eq!(reg.len(), 1);

    reg.register(make_compute_pass("q"));
    assert!(!reg.is_empty());
    assert_eq!(reg.len(), 2);
}

#[test]
fn iter_on_cloned_registry_matches_original() {
    let mut reg = PassRegistry::new();
    reg.register(make_graphics_pass("a"));
    reg.register(make_compute_pass("b"));
    reg.register(make_copy_pass("c"));

    let cloned = reg.clone();

    let orig_names: Vec<&str> = reg.iter().map(|p| p.name.as_str()).collect();
    let clone_names: Vec<&str> = cloned.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(orig_names, clone_names);
}

#[test]
fn get_does_not_mutate_registry() {
    let mut reg = PassRegistry::new();
    reg.register(make_graphics_pass("stable"));

    let len_before = reg.len();
    let _ = reg.get(PassIndex(0));
    let len_after = reg.len();

    assert_eq!(len_before, len_after);
}
