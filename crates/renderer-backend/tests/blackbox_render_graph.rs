// SPDX-License-Identifier: MIT
//
// blackbox_render_graph.rs -- Blackbox contract tests for T-FG-4.1 RenderGraphBuilder.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::frame_graph::RenderGraphBuilder` and the
// IR types it produces -- no internal fields, no private methods.
//
// Acceptance criterion (T-FG-4.1):
//   RenderGraphBuilder::new() creates an empty builder.
//   create_texture() / create_buffer() declare resources and return handles.
//   add_graphics_pass() / add_compute_pass() / add_copy_pass() declare passes
//   and return indices.
//   finalize() consumes the builder and returns (Vec<IrPass>, Vec<IrResource>).
//
// Contract:
//   - Texture resources have Texture2D descriptor, 1 mip level, 1 array layer,
//     Transient lifetime, Uninitialized initial state.
//   - Buffer resources have usage "storage | copy_src | copy_dst", are NOT
//     indirect-argument buffers, Transient, Uninitialized.
//   - Graphics passes use ColorAttachment view type, Clear/Store load/store ops,
//     Direct instance source with index_count=0 and instance_count=1.
//   - Compute passes use Storage view type, Direct dispatch source with the
//     caller-provided workgroup counts.
//   - Copy passes use StorageBuffer view type, no dispatch source.
//   - Resource handles start at 0 and increment sequentially across all types.
//   - Pass indices start at 0 and increment sequentially across all types.
//   - Empty builder produces empty vectors from finalize().
//
// Coverage:
//   1.  Builder construction and empty finalize
//   2.  Texture creation -- handle value, resource fields
//   3.  Buffer creation -- handle value, resource fields
//   4.  Graphics pass -- color attachments, depth-stencil, type, view, instance
//   5.  Compute pass -- reads, writes, workgroup, type, view, dispatch
//   6.  Copy pass -- source, dest, type, view
//   7.  Method chaining -- interleaved resources and passes
//   8.  Multiple passes -- correct count and ordering
//   9.  Integration -- full pipeline produces well-structured frame graph

use renderer_backend::frame_graph::{
    DispatchSource, InstanceSource, IrPass, IrResource, PassIndex, PassType,
    RenderGraphBuilder, ResourceDesc, ResourceHandle, ResourceLifetime,
    ResourceState, TextureDesc, ViewType,
};

// =============================================================================
// SECTION 1 -- Builder construction and empty finalize
// =============================================================================

/// A newly constructed builder finalizes to empty vectors.
#[test]
fn builder_new_then_finalize_returns_empty() {
    let builder = RenderGraphBuilder::new();
    let (passes, resources) = builder.finalize();

    assert!(
        passes.is_empty(),
        "New builder must produce zero passes, got {}",
        passes.len(),
    );
    assert!(
        resources.is_empty(),
        "New builder must produce zero resources, got {}",
        resources.len(),
    );
}

/// Explicitly verifies both returned vectors are empty (length check).
#[test]
fn empty_builder_finalize_produces_empty_vectors() {
    let (passes, resources) = RenderGraphBuilder::new().finalize();

    assert_eq!(passes.len(), 0, "passes vec must be empty");
    assert_eq!(resources.len(), 0, "resources vec must be empty");
}

// =============================================================================
// SECTION 2 -- Texture creation: handle value, resource fields
// =============================================================================

/// Texture creation returns a valid ResourceHandle that starts at 0 and
/// increments on successive calls.
#[test]
fn create_texture_returns_incrementing_handles() {
    let mut builder = RenderGraphBuilder::new();

    let h0 = builder.create_texture("tex_a", 1920, 1080, "rgba8unorm");
    let h1 = builder.create_texture("tex_b", 800, 600, "bgra8unorm-srgb");
    let h2 = builder.create_texture("tex_c", 256, 256, "r32float");

    assert_eq!(h0, ResourceHandle(0), "First texture gets handle 0");
    assert_eq!(h1, ResourceHandle(1), "Second texture gets handle 1");
    assert_eq!(h2, ResourceHandle(2), "Third texture gets handle 2");
}

/// The created texture IrResource has the correct descriptor fields.
#[test]
fn create_texture_produces_expected_resource() {
    let mut builder = RenderGraphBuilder::new();
    let handle = builder.create_texture("color_rt", 1920, 1080, "rgba8unorm");
    let (_, resources) = builder.finalize();

    assert_eq!(resources.len(), 1, "One resource created");

    let res = &resources[0];
    assert_eq!(res.handle, handle, "Resource handle matches returned handle");
    assert_eq!(res.name, "color_rt", "Resource name preserved");

    match &res.desc {
        ResourceDesc::Texture2D(desc) => {
            assert_eq!(desc.width, 1920, "Texture width preserved");
            assert_eq!(desc.height, 1080, "Texture height preserved");
            assert_eq!(desc.mip_levels, 1, "Default mip_levels is 1");
            assert_eq!(desc.array_layers, 1, "Default array_layers is 1");
            assert_eq!(desc.format, "rgba8unorm", "Texture format preserved");
        }
        other => panic!("Expected Texture2D desc, got {:?}", other),
    }

    assert_eq!(
        res.lifetime,
        ResourceLifetime::Transient,
        "Texture lifetime is Transient",
    );
    assert_eq!(
        res.initial_state,
        ResourceState::Uninitialized,
        "Texture initial state is Uninitialized",
    );
    assert!(
        res.view_format_override.is_none(),
        "No view format override by default",
    );
}

/// Resource name is preserved verbatim in the output.
#[test]
fn create_texture_name_preserved_exactly() {
    let mut builder = RenderGraphBuilder::new();
    builder.create_texture("custom_name_with_underscores_123", 64, 64, "r8unorm");
    let (_, resources) = builder.finalize();

    assert_eq!(
        resources[0].name, "custom_name_with_underscores_123",
        "Resource name must be preserved verbatim",
    );
}

/// Texture with non-square dimensions preserves exact values.
#[test]
fn create_texture_non_square_dimensions() {
    let mut builder = RenderGraphBuilder::new();
    builder.create_texture("non_square", 800, 600, "rgba8unorm");
    let (_, resources) = builder.finalize();

    match &resources[0].desc {
        ResourceDesc::Texture2D(desc) => {
            assert_eq!(desc.width, 800);
            assert_eq!(desc.height, 600);
        }
        _ => panic!("Expected Texture2D"),
    }
}

/// Texture with zero dimensions produces a valid resource.
#[test]
fn create_texture_zero_dimensions() {
    let mut builder = RenderGraphBuilder::new();
    builder.create_texture("zero_sized", 0, 0, "r8unorm");
    let (_, resources) = builder.finalize();

    match &resources[0].desc {
        ResourceDesc::Texture2D(desc) => {
            assert_eq!(desc.width, 0, "zero width is permitted");
            assert_eq!(desc.height, 0, "zero height is permitted");
        }
        _ => panic!("Expected Texture2D"),
    }
}

// =============================================================================
// SECTION 3 -- Buffer creation: handle value, resource fields
// =============================================================================

/// Buffer creation returns a valid ResourceHandle, continuing from the
/// highest texture handle.
#[test]
fn create_buffer_returns_incrementing_handles() {
    let mut builder = RenderGraphBuilder::new();

    let tex = builder.create_texture("tex", 100, 100, "r8unorm");
    let buf0 = builder.create_buffer("buf_a", 4096);
    let buf1 = builder.create_buffer("buf_b", 65536);
    let buf2 = builder.create_buffer("buf_c", 1048576);

    assert_eq!(tex, ResourceHandle(0), "Texture handle is 0");
    assert_eq!(buf0, ResourceHandle(1), "First buffer gets handle 1");
    assert_eq!(buf1, ResourceHandle(2), "Second buffer gets handle 2");
    assert_eq!(buf2, ResourceHandle(3), "Third buffer gets handle 3");
}

/// The created buffer IrResource has the default descriptor fields.
#[test]
fn create_buffer_produces_expected_resource() {
    let mut builder = RenderGraphBuilder::new();
    let handle = builder.create_buffer("storage_buf", 262144);
    let (_, resources) = builder.finalize();

    assert_eq!(resources.len(), 1, "One resource created");

    let res = &resources[0];
    assert_eq!(res.handle, handle, "Resource handle matches returned handle");
    assert_eq!(res.name, "storage_buf", "Buffer name preserved");

    match &res.desc {
        ResourceDesc::Buffer(desc) => {
            assert_eq!(desc.size, 262144, "Buffer size preserved");
            assert_eq!(
                desc.usage, "storage | copy_src | copy_dst",
                "Default buffer usage string",
            );
            assert!(
                !desc.is_indirect_arg,
                "Buffer is NOT an indirect argument by default",
            );
        }
        other => panic!("Expected Buffer desc, got {:?}", other),
    }

    assert_eq!(
        res.lifetime,
        ResourceLifetime::Transient,
        "Buffer lifetime is Transient",
    );
    assert_eq!(
        res.initial_state,
        ResourceState::Uninitialized,
        "Buffer initial state is Uninitialized",
    );
}

/// Zero-sized buffer creates a valid resource.
#[test]
fn create_buffer_zero_size() {
    let mut builder = RenderGraphBuilder::new();
    builder.create_buffer("empty_buf", 0);
    let (_, resources) = builder.finalize();

    match &resources[0].desc {
        ResourceDesc::Buffer(desc) => {
            assert_eq!(desc.size, 0, "zero-sized buffer is valid");
        }
        _ => panic!("Expected Buffer"),
    }
}

/// Large buffer sizes are preserved exactly.
#[test]
fn create_buffer_large_size() {
    let mut builder = RenderGraphBuilder::new();
    builder.create_buffer("large_buf", 0xFFFF_FFFF);
    let (_, resources) = builder.finalize();

    match &resources[0].desc {
        ResourceDesc::Buffer(desc) => {
            assert_eq!(desc.size, 0xFFFF_FFFF, "large buffer size preserved");
        }
        _ => panic!("Expected Buffer"),
    }
}

// =============================================================================
// SECTION 4 -- Graphics pass: color attachments, depth-stencil, type, view
// =============================================================================

/// Adding a graphics pass returns a PassIndex starting at 0.
#[test]
fn add_graphics_pass_returns_pass_index() {
    let mut builder = RenderGraphBuilder::new();
    let color = builder.create_texture("color", 100, 100, "rgba8unorm");

    let p0 = builder.add_graphics_pass("pass_a", &[color], None);
    let p1 = builder.add_graphics_pass("pass_b", &[], None);
    let p2 = builder.add_graphics_pass("pass_c", &[color], None);

    assert_eq!(p0, PassIndex(0), "First pass gets index 0");
    assert_eq!(p1, PassIndex(1), "Second pass gets index 1");
    assert_eq!(p2, PassIndex(2), "Third pass gets index 2");
}

/// Graphics pass with color attachments produces the correct IrPass structure.
#[test]
fn add_graphics_pass_with_color_attachments() {
    let mut builder = RenderGraphBuilder::new();
    let c0 = builder.create_texture("albedo", 1920, 1080, "rgba8unorm");
    let c1 = builder.create_texture("normal", 1920, 1080, "rgba8unorm");
    let c2 = builder.create_texture("roughness", 1920, 1080, "rgba8unorm");

    let idx = builder.add_graphics_pass("gbuffer", &[c0, c1, c2], None);
    let (passes, _) = builder.finalize();

    assert_eq!(passes.len(), 1);

    let pass = &passes[0];
    assert_eq!(pass.index, idx, "Pass index matches returned index");
    assert_eq!(pass.name, "gbuffer", "Pass name preserved");
    assert_eq!(pass.pass_type, PassType::Graphics, "Pass type is Graphics");

    // Color attachments.
    assert_eq!(
        pass.color_attachments.len(),
        3,
        "Three color attachments",
    );

    // Each attachment has correct resource handle and default Clear/Store.
    for (i, expected_handle) in [c0, c1, c2].iter().enumerate() {
        let ca = &pass.color_attachments[i];
        assert_eq!(ca.resource, *expected_handle, "CA {} resource handle", i);
        assert_eq!(
            ca.mip_level, 0,
            "CA {} default mip_level is 0", i,
        );
        assert_eq!(
            ca.array_layer, 0,
            "CA {} default array_layer is 0", i,
        );
        assert_eq!(
            ca.load_op,
            renderer_backend::frame_graph::AttachmentLoadOp::Clear,
            "CA {} load_op is Clear", i,
        );
        assert_eq!(
            ca.store_op,
            renderer_backend::frame_graph::AttachmentStoreOp::Store,
            "CA {} store_op is Store", i,
        );
    }

    // Depth-stencil: None.
    assert!(
        pass.depth_stencil.is_none(),
        "Graphics pass without depth_stencil argument has None",
    );

    // Instance source: Direct with default values.
    match &pass.instance_source {
        InstanceSource::Direct {
            index_count,
            instance_count,
            ..
        } => {
            assert_eq!(*index_count, 0, "Default index_count is 0");
            assert_eq!(*instance_count, 1, "Default instance_count is 1");
        }
        other => panic!("Expected Direct instance source, got {:?}", other),
    }

    // View type.
    assert_eq!(
        pass.view_type,
        ViewType::ColorAttachment,
        "Graphics pass uses ColorAttachment view type",
    );

    // Dispatch source: None.
    assert!(
        pass.dispatch_source.is_none(),
        "Graphics pass has no dispatch source",
    );

    // Tags: empty.
    assert!(pass.tags.is_empty(), "Graphics pass has no tags by default");
}

/// Graphics pass with depth-stencil attachment.
#[test]
fn add_graphics_pass_with_depth_stencil() {
    let mut builder = RenderGraphBuilder::new();
    let color = builder.create_texture("albedo", 1920, 1080, "rgba8unorm");
    let depth = builder.create_texture("depth", 1920, 1080, "depth32float");

    let _idx = builder.add_graphics_pass("depth_pass", &[color], Some(depth));
    let (passes, _) = builder.finalize();

    assert_eq!(passes.len(), 1);
    let pass = &passes[0];

    assert!(pass.depth_stencil.is_some(), "Depth-stencil is present");
    let ds = pass.depth_stencil.as_ref().unwrap();
    assert_eq!(ds.resource, depth, "Depth-stencil resource handle");
    assert_eq!(
        ds.depth_load_op,
        renderer_backend::frame_graph::AttachmentLoadOp::Load,
        "Default depth_load_op is Load",
    );
    assert_eq!(
        ds.depth_store_op,
        renderer_backend::frame_graph::AttachmentStoreOp::Store,
        "Default depth_store_op is Store",
    );
    assert!(
        ds.depth_test_enabled,
        "Depth test enabled by default",
    );
    assert!(
        ds.depth_write_enabled,
        "Depth write enabled by default",
    );
}

/// Graphics pass without color attachments (only depth-stencil).
#[test]
fn add_graphics_pass_empty_color_attachments() {
    let mut builder = RenderGraphBuilder::new();
    let depth = builder.create_texture("depth", 100, 100, "depth32float");

    let _idx = builder.add_graphics_pass("depth_only", &[], Some(depth));
    let (passes, _) = builder.finalize();

    assert_eq!(passes.len(), 1);
    let pass = &passes[0];

    assert!(
        pass.color_attachments.is_empty(),
        "No color attachments",
    );
    assert!(pass.depth_stencil.is_some(), "Depth-stencil is present");
    assert_eq!(
        pass.depth_stencil.as_ref().unwrap().resource,
        depth,
        "Depth-stencil handle",
    );
}

/// Graphics pass with neither color attachments nor depth-stencil.
#[test]
fn add_graphics_pass_no_attachments() {
    let mut builder = RenderGraphBuilder::new();
    let _idx = builder.add_graphics_pass("empty_gfx", &[], None);
    let (passes, _) = builder.finalize();

    assert_eq!(passes.len(), 1);
    let pass = &passes[0];

    assert!(pass.color_attachments.is_empty(), "No color attachments");
    assert!(pass.depth_stencil.is_none(), "No depth-stencil");
    assert_eq!(pass.pass_type, PassType::Graphics, "Type is Graphics");
}

/// Access set is correctly synced from color attachments (Clear=no read, Store=write).
#[test]
fn add_graphics_pass_access_set_from_attachments() {
    let mut builder = RenderGraphBuilder::new();
    let c0 = builder.create_texture("rt0", 100, 100, "rgba8unorm");
    let c1 = builder.create_texture("rt1", 100, 100, "rgba8unorm");

    builder.add_graphics_pass("mrt", &[c0, c1], None);
    let (passes, _) = builder.finalize();

    let pass = &passes[0];

    // With Clear load op, resources are NOT in reads.
    assert!(
        !pass.access_set.reads.contains(&c0),
        "Clear load = no read for c0",
    );
    assert!(
        !pass.access_set.reads.contains(&c1),
        "Clear load = no read for c1",
    );

    // With Store store op, resources ARE in writes.
    assert!(
        pass.access_set.writes.contains(&c0),
        "Store store = write for c0",
    );
    assert!(
        pass.access_set.writes.contains(&c1),
        "Store store = write for c1",
    );
}

// =============================================================================
// SECTION 5 -- Compute pass: reads, writes, workgroup, type, view, dispatch
// =============================================================================

/// Adding a compute pass returns a sequential PassIndex.
#[test]
fn add_compute_pass_returns_pass_index() {
    let mut builder = RenderGraphBuilder::new();
    let buf = builder.create_buffer("data", 4096);

    let p0 = builder.add_compute_pass("comp_a", &[buf], &[], (1, 1, 1));
    let p1 = builder.add_compute_pass("comp_b", &[], &[buf], (8, 8, 1));
    let p2 = builder.add_compute_pass("comp_c", &[buf], &[buf], (16, 16, 1));

    assert_eq!(p0, PassIndex(0), "First compute pass gets index 0");
    assert_eq!(p1, PassIndex(1), "Second compute pass gets index 1");
    assert_eq!(p2, PassIndex(2), "Third compute pass gets index 2");
}

/// Compute pass with reads and writes produces the correct IrPass structure.
#[test]
fn add_compute_pass_with_reads_and_writes() {
    let mut builder = RenderGraphBuilder::new();
    let input = builder.create_texture("input", 100, 100, "rgba8unorm");
    let output = builder.create_buffer("output", 65536);

    let idx = builder.add_compute_pass("postfx", &[input], &[output], (8, 8, 1));
    let (passes, _) = builder.finalize();

    assert_eq!(passes.len(), 1);
    let pass = &passes[0];

    assert_eq!(pass.index, idx, "Pass index matches");
    assert_eq!(pass.name, "postfx", "Pass name preserved");
    assert_eq!(pass.pass_type, PassType::Compute, "Pass type is Compute");

    // Access set.
    assert!(
        pass.access_set.reads.contains(&input),
        "Input resource is in reads",
    );
    assert!(
        pass.access_set.writes.contains(&output),
        "Output resource is in writes",
    );
    assert_eq!(pass.access_set.reads.len(), 1, "One read");
    assert_eq!(pass.access_set.writes.len(), 1, "One write");

    // Color attachments: empty.
    assert!(
        pass.color_attachments.is_empty(),
        "Compute pass has no color attachments",
    );

    // Depth-stencil: None.
    assert!(pass.depth_stencil.is_none(), "Compute pass has no depth-stencil");

    // View type.
    assert_eq!(
        pass.view_type,
        ViewType::Storage,
        "Compute pass uses Storage view type",
    );

    // Tags: empty.
    assert!(pass.tags.is_empty(), "Compute pass has no tags by default");
}

/// Compute pass workgroup sizes preserved in the dispatch source.
#[test]
fn add_compute_pass_workgroup_sizes_preserved() {
    let mut builder = RenderGraphBuilder::new();
    let buf = builder.create_buffer("data", 4096);

    builder.add_compute_pass("large_comp", &[buf], &[], (64, 32, 16));
    let (passes, _) = builder.finalize();

    let pass = &passes[0];

    match &pass.dispatch_source {
        Some(DispatchSource::Direct {
            group_count_x,
            group_count_y,
            group_count_z,
        }) => {
            assert_eq!(*group_count_x, 64, "Workgroup X preserved");
            assert_eq!(*group_count_y, 32, "Workgroup Y preserved");
            assert_eq!(*group_count_z, 16, "Workgroup Z preserved");
        }
        other => panic!("Expected Direct dispatch source, got {:?}", other),
    }
}

/// Compute pass with no reads (write-only).
#[test]
fn add_compute_pass_no_reads() {
    let mut builder = RenderGraphBuilder::new();
    let output = builder.create_buffer("output", 4096);

    builder.add_compute_pass("write_only", &[], &[output], (4, 4, 1));
    let (passes, _) = builder.finalize();

    assert!(
        passes[0].access_set.reads.is_empty(),
        "No reads in write-only compute pass",
    );
    assert!(
        passes[0].access_set.writes.contains(&output),
        "Output still in writes",
    );
}

/// Compute pass with no writes (read-only).
#[test]
fn add_compute_pass_no_writes() {
    let mut builder = RenderGraphBuilder::new();
    let input = builder.create_texture("input", 100, 100, "r8unorm");

    builder.add_compute_pass("read_only", &[input], &[], (1, 1, 1));
    let (passes, _) = builder.finalize();

    assert!(
        passes[0].access_set.writes.is_empty(),
        "No writes in read-only compute pass",
    );
    assert!(
        passes[0].access_set.reads.contains(&input),
        "Input still in reads",
    );
}

/// Compute pass with single-element workgroup (1, 1, 1).
#[test]
fn add_compute_pass_minimal_workgroup() {
    let mut builder = RenderGraphBuilder::new();
    let buf = builder.create_buffer("data", 1024);

    builder.add_compute_pass("minimal", &[buf], &[], (1, 1, 1));
    let (passes, _) = builder.finalize();

    match &passes[0].dispatch_source {
        Some(DispatchSource::Direct {
            group_count_x,
            group_count_y,
            group_count_z,
        }) => {
            assert_eq!(*group_count_x, 1);
            assert_eq!(*group_count_y, 1);
            assert_eq!(*group_count_z, 1);
        }
        _ => panic!("Expected Direct dispatch"),
    }
}

// =============================================================================
// SECTION 6 -- Copy pass: source, dest, type, view
// =============================================================================

/// Adding a copy pass returns a sequential PassIndex.
#[test]
fn add_copy_pass_returns_pass_index() {
    let mut builder = RenderGraphBuilder::new();
    let src = builder.create_texture("src", 100, 100, "rgba8unorm");
    let dst = builder.create_buffer("dst", 65536);

    let p0 = builder.add_copy_pass("copy_0", src, dst);
    let p1 = builder.add_copy_pass("copy_1", src, dst);

    assert_eq!(p0, PassIndex(0), "First copy pass gets index 0");
    assert_eq!(p1, PassIndex(1), "Second copy pass gets index 1");
}

/// Copy pass with source and dest produces the correct IrPass structure.
#[test]
fn add_copy_pass_with_source_and_dest() {
    let mut builder = RenderGraphBuilder::new();
    let src = builder.create_texture("src_tex", 100, 100, "rgba8unorm");
    let dst = builder.create_buffer("dst_buf", 4096);

    let idx = builder.add_copy_pass("copy_out", src, dst);
    let (passes, _) = builder.finalize();

    assert_eq!(passes.len(), 1);
    let pass = &passes[0];

    assert_eq!(pass.index, idx, "Pass index matches");
    assert_eq!(pass.name, "copy_out", "Pass name preserved");
    assert_eq!(pass.pass_type, PassType::Copy, "Pass type is Copy");

    // Access set: reads = [src], writes = [dst].
    assert!(
        pass.access_set.reads.contains(&src),
        "Source is in reads",
    );
    assert!(
        pass.access_set.writes.contains(&dst),
        "Dest is in writes",
    );
    assert_eq!(pass.access_set.reads.len(), 1, "One read (source)");
    assert_eq!(pass.access_set.writes.len(), 1, "One write (dest)");

    // No color attachments, no depth-stencil.
    assert!(
        pass.color_attachments.is_empty(),
        "Copy pass has no color attachments",
    );
    assert!(
        pass.depth_stencil.is_none(),
        "Copy pass has no depth-stencil",
    );

    // View type.
    assert_eq!(
        pass.view_type,
        ViewType::StorageBuffer,
        "Copy pass uses StorageBuffer view type",
    );

    // No dispatch source.
    assert!(
        pass.dispatch_source.is_none(),
        "Copy pass has no dispatch source",
    );

    // Tags: empty.
    assert!(pass.tags.is_empty(), "Copy pass has no tags by default");
}

/// Copy pass where source and dest are the same resource.
#[test]
fn add_copy_pass_same_source_and_dest() {
    let mut builder = RenderGraphBuilder::new();
    let buf = builder.create_buffer("self_copy", 4096);

    builder.add_copy_pass("self_copy", buf, buf);
    let (passes, _) = builder.finalize();

    let pass = &passes[0];

    assert!(
        pass.access_set.reads.contains(&buf),
        "Source is in reads",
    );
    assert!(
        pass.access_set.writes.contains(&buf),
        "Dest is in writes",
    );
}

// =============================================================================
// SECTION 7 -- Method chaining: interleaved resources and passes
// =============================================================================

/// Multiple operations chained in sequence produce correct final output.
#[test]
fn method_chaining_multiple_operations() {
    let mut builder = RenderGraphBuilder::new();

    let color = builder.create_texture("color", 1920, 1080, "rgba8unorm");
    let depth = builder.create_texture("depth", 1920, 1080, "depth32float");
    let output = builder.create_buffer("output", 1048576);

    let _gfx = builder.add_graphics_pass("render", &[color], Some(depth));
    let _comp = builder.add_compute_pass("process", &[color], &[output], (8, 8, 1));
    let _copy = builder.add_copy_pass("copy_out", color, output);

    let (passes, resources) = builder.finalize();

    // Three resources (two textures + one buffer).
    assert_eq!(resources.len(), 3, "Three resources created");
    assert_eq!(resources[0].handle, ResourceHandle(0));
    assert_eq!(resources[0].name, "color");
    assert_eq!(resources[1].handle, ResourceHandle(1));
    assert_eq!(resources[1].name, "depth");
    assert_eq!(resources[2].handle, ResourceHandle(2));
    assert_eq!(resources[2].name, "output");

    // Three passes (graphics + compute + copy).
    assert_eq!(passes.len(), 3, "Three passes created");
    assert_eq!(passes[0].name, "render");
    assert_eq!(passes[0].pass_type, PassType::Graphics);
    assert_eq!(passes[1].name, "process");
    assert_eq!(passes[1].pass_type, PassType::Compute);
    assert_eq!(passes[2].name, "copy_out");
    assert_eq!(passes[2].pass_type, PassType::Copy);
}

/// Passes and resources reference each other correctly through handles.
#[test]
fn method_chaining_resource_handle_connectivity() {
    let mut builder = RenderGraphBuilder::new();

    let tex = builder.create_texture("framebuffer", 800, 600, "rgba16float");
    let buf = builder.create_buffer("result", 4096);

    let gfx_idx = builder.add_graphics_pass("render", &[tex], None);
    let comp_idx = builder.add_compute_pass("compute", &[tex], &[buf], (4, 4, 1));
    let copy_idx = builder.add_copy_pass("copy", tex, buf);

    let (passes, resources) = builder.finalize();

    // Verify resource handles connectivity.
    assert_eq!(resources[0].handle, tex, "Texture handle is tex");
    assert_eq!(resources[1].handle, buf, "Buffer handle is buf");

    // Graphics pass writes tex (Clear+Store = no read, write).
    assert!(!passes[0].access_set.reads.contains(&tex));
    assert!(passes[0].access_set.writes.contains(&tex));

    // Compute pass reads tex, writes buf.
    assert!(passes[1].access_set.reads.contains(&tex));
    assert!(passes[1].access_set.writes.contains(&buf));

    // Copy pass reads tex, writes buf.
    assert!(passes[2].access_set.reads.contains(&tex));
    assert!(passes[2].access_set.writes.contains(&buf));

    // Pass indices.
    assert_eq!(gfx_idx, PassIndex(0));
    assert_eq!(comp_idx, PassIndex(1));
    assert_eq!(copy_idx, PassIndex(2));
}

/// Interleaved resource creation and pass addition preserves order.
#[test]
fn method_chaining_interleaved_operations() {
    let mut builder = RenderGraphBuilder::new();

    let r0 = builder.create_texture("tex0", 100, 100, "r8unorm");
    let _p0 = builder.add_graphics_pass("pass0", &[r0], None);
    let r1 = builder.create_buffer("buf0", 1024);
    let _p1 = builder.add_compute_pass("pass1", &[r0], &[r1], (1, 1, 1));
    let r2 = builder.create_texture("tex1", 200, 200, "rgba8unorm");
    let _p2 = builder.add_copy_pass("pass2", r0, r1);
    let _ = r2;

    let (passes, resources) = builder.finalize();

    // Resources in creation order: tex0, buf0, tex1.
    assert_eq!(resources.len(), 3, "Three resources");
    assert_eq!(resources[0].handle, ResourceHandle(0));
    assert_eq!(resources[0].name, "tex0");
    assert_eq!(resources[1].handle, ResourceHandle(1));
    assert_eq!(resources[1].name, "buf0");
    assert_eq!(resources[2].handle, ResourceHandle(2));
    assert_eq!(resources[2].name, "tex1");

    // Passes in creation order: pass0, pass1, pass2.
    assert_eq!(passes.len(), 3, "Three passes");
    assert_eq!(passes[0].name, "pass0");
    assert_eq!(passes[0].pass_type, PassType::Graphics);
    assert_eq!(passes[1].name, "pass1");
    assert_eq!(passes[1].pass_type, PassType::Compute);
    assert_eq!(passes[2].name, "pass2");
    assert_eq!(passes[2].pass_type, PassType::Copy);
}

// =============================================================================
// SECTION 8 -- Multiple passes: correct count, ordering, and types
// =============================================================================

/// Creating multiple passes of the same type produces correct count.
#[test]
fn multiple_graphics_passes_produce_correct_count() {
    let mut builder = RenderGraphBuilder::new();
    let color = builder.create_texture("rt", 100, 100, "rgba8unorm");

    let n = 5;
    for i in 0..n {
        let name = format!("gfx_pass_{}", i);
        builder.add_graphics_pass(&name, &[color], None);
    }

    let (passes, _) = builder.finalize();

    assert_eq!(passes.len(), n, "{} graphics passes created", n);
    for i in 0..n {
        assert_eq!(passes[i].pass_type, PassType::Graphics, "Pass {} is Graphics", i);
        assert_eq!(passes[i].index, PassIndex(i), "Pass {} index", i);
    }
}

/// Multiple compute passes produce correct count.
#[test]
fn multiple_compute_passes_produce_correct_count() {
    let mut builder = RenderGraphBuilder::new();
    let buf = builder.create_buffer("data", 256);

    let n = 4;
    for i in 0..n {
        let name = format!("comp_pass_{}", i);
        builder.add_compute_pass(&name, &[], &[buf], (1, 1, 1));
    }

    let (passes, _) = builder.finalize();

    assert_eq!(passes.len(), n, "{} compute passes created", n);
    for i in 0..n {
        assert_eq!(passes[i].pass_type, PassType::Compute, "Pass {} is Compute", i);
        assert_eq!(passes[i].index, PassIndex(i), "Pass {} index", i);
    }
}

/// Multiple copy passes produce correct count.
#[test]
fn multiple_copy_passes_produce_correct_count() {
    let mut builder = RenderGraphBuilder::new();
    let src = builder.create_texture("src", 100, 100, "r8unorm");
    let dst = builder.create_texture("dst", 100, 100, "r8unorm");

    let n = 3;
    for i in 0..n {
        let name = format!("copy_pass_{}", i);
        builder.add_copy_pass(&name, src, dst);
    }

    let (passes, _) = builder.finalize();

    assert_eq!(passes.len(), n, "{} copy passes created", n);
    for i in 0..n {
        assert_eq!(passes[i].pass_type, PassType::Copy, "Pass {} is Copy", i);
        assert_eq!(passes[i].index, PassIndex(i), "Pass {} index", i);
    }
}

/// Mixed pass types all appear in the correct order.
#[test]
fn mixed_pass_types_in_order() {
    let mut builder = RenderGraphBuilder::new();
    let tex = builder.create_texture("tex", 100, 100, "rgba8unorm");
    let buf = builder.create_buffer("buf", 4096);

    let pass_types = [
        PassType::Graphics,
        PassType::Compute,
        PassType::Copy,
        PassType::Compute,
        PassType::Graphics,
        PassType::Copy,
    ];

    for (i, pt) in pass_types.iter().enumerate() {
        let name = format!("pass_{}", i);
        match pt {
            PassType::Graphics => {
                builder.add_graphics_pass(&name, &[tex], None);
            }
            PassType::Compute => {
                builder.add_compute_pass(&name, &[tex], &[buf], (1, 1, 1));
            }
            PassType::Copy => {
                builder.add_copy_pass(&name, tex, buf);
            }
            _ => panic!("Unexpected pass type in test"),
        }
    }

    let (passes, _) = builder.finalize();

    assert_eq!(passes.len(), pass_types.len(), "All passes created");
    for (i, expected_type) in pass_types.iter().enumerate() {
        assert_eq!(
            passes[i].pass_type, *expected_type,
            "Pass {} type mismatch", i,
        );
    }
}

// =============================================================================
// SECTION 9 -- Integration: full pipeline produces well-structured frame graph
// =============================================================================

/// Full pipeline: textures + buffers + all pass types, verify every field.
#[test]
fn full_pipeline_produces_valid_frame_graph() {
    let mut builder = RenderGraphBuilder::new();

    // --- Declare resources ---
    let albedo = builder.create_texture("albedo", 1920, 1080, "rgba8unorm");
    let normal = builder.create_texture("normal", 1920, 1080, "rgba8unorm");
    let depth = builder.create_texture("depth", 1920, 1080, "depth32float");
    let light_accum = builder.create_texture("light_accum", 1920, 1080, "rgba16float");
    let _vertex_buf = builder.create_buffer("vertices", 262144);
    let _indirect_buf = builder.create_buffer("indirect_args", 4096);
    let output_buf = builder.create_buffer("final_output", 1048576);

    // --- Declare passes ---
    // Pass 0: G-buffer render
    let _gbuf_idx = builder.add_graphics_pass("gbuffer", &[albedo, normal], Some(depth));

    // Pass 1: Lighting compute (reads gbuffer, writes light_accum)
    let _light_idx = builder.add_compute_pass(
        "lighting",
        &[albedo, normal, depth],
        &[light_accum],
        (16, 16, 1),
    );

    // Pass 2: Copy depth buffer to output
    let _depth_copy_idx = builder.add_copy_pass("depth_copy", depth, output_buf);

    // Pass 3: Post-process compute
    let _post_idx = builder.add_compute_pass(
        "post_process",
        &[light_accum],
        &[],
        (8, 8, 1),
    );

    // --- Finalize ---
    let (passes, resources) = builder.finalize();

    // --- Assert resources ---
    assert_eq!(resources.len(), 7, "Seven resources declared");

    // Check resource handles and names by index.
    let expected_resources = [
        ("albedo", ResourceHandle(0)),
        ("normal", ResourceHandle(1)),
        ("depth", ResourceHandle(2)),
        ("light_accum", ResourceHandle(3)),
        ("vertices", ResourceHandle(4)),
        ("indirect_args", ResourceHandle(5)),
        ("final_output", ResourceHandle(6)),
    ];

    for (i, (expected_name, expected_handle)) in expected_resources.iter().enumerate() {
        assert_eq!(
            resources[i].name, *expected_name,
            "Resource {} name", i,
        );
        assert_eq!(
            resources[i].handle, *expected_handle,
            "Resource {} handle", i,
        );
    }

    // Verify texture resources have Texture2D desc.
    for i in 0..4 {
        match &resources[i].desc {
            ResourceDesc::Texture2D(_) => {} // OK
            other => panic!(
                "Resource {} '{}': expected Texture2D, got {:?}",
                i, resources[i].name, other,
            ),
        }
    }

    // Verify buffer resources have Buffer desc.
    for i in 4..7 {
        match &resources[i].desc {
            ResourceDesc::Buffer(_) => {} // OK
            other => panic!(
                "Resource {} '{}': expected Buffer, got {:?}",
                i, resources[i].name, other,
            ),
        }
    }

    // --- Assert passes ---
    assert_eq!(passes.len(), 4, "Four passes declared");

    // Pass 0: Graphics -- gbuffer
    assert_eq!(passes[0].name, "gbuffer");
    assert_eq!(passes[0].pass_type, PassType::Graphics);
    assert_eq!(passes[0].index, PassIndex(0));
    assert_eq!(passes[0].color_attachments.len(), 2, "Two color attachments");
    assert_eq!(passes[0].color_attachments[0].resource, albedo);
    assert_eq!(passes[0].color_attachments[1].resource, normal);
    assert!(passes[0].depth_stencil.is_some(), "Depth-stencil present");
    assert_eq!(
        passes[0].depth_stencil.as_ref().unwrap().resource,
        depth,
    );
    assert_eq!(passes[0].view_type, ViewType::ColorAttachment);
    assert!(passes[0].dispatch_source.is_none());

    // Pass 1: Compute -- lighting
    assert_eq!(passes[1].name, "lighting");
    assert_eq!(passes[1].pass_type, PassType::Compute);
    assert_eq!(passes[1].index, PassIndex(1));
    assert!(passes[1].access_set.reads.contains(&albedo));
    assert!(passes[1].access_set.reads.contains(&normal));
    assert!(passes[1].access_set.reads.contains(&depth));
    assert!(passes[1].access_set.writes.contains(&light_accum));
    assert_eq!(passes[1].view_type, ViewType::Storage);
    match &passes[1].dispatch_source {
        Some(DispatchSource::Direct {
            group_count_x,
            group_count_y,
            group_count_z,
        }) => {
            assert_eq!(*group_count_x, 16);
            assert_eq!(*group_count_y, 16);
            assert_eq!(*group_count_z, 1);
        }
        _ => panic!("Expected Direct dispatch"),
    }

    // Pass 2: Copy -- depth_copy
    assert_eq!(passes[2].name, "depth_copy");
    assert_eq!(passes[2].pass_type, PassType::Copy);
    assert_eq!(passes[2].index, PassIndex(2));
    assert!(passes[2].access_set.reads.contains(&depth));
    assert!(passes[2].access_set.writes.contains(&output_buf));
    assert_eq!(passes[2].view_type, ViewType::StorageBuffer);
    assert!(passes[2].dispatch_source.is_none());

    // Pass 3: Compute -- post_process
    assert_eq!(passes[3].name, "post_process");
    assert_eq!(passes[3].pass_type, PassType::Compute);
    assert_eq!(passes[3].index, PassIndex(3));
    assert!(passes[3].access_set.reads.contains(&light_accum));
    assert!(passes[3].access_set.writes.is_empty(), "No writes in post_process");
    assert_eq!(passes[3].view_type, ViewType::Storage);
    match &passes[3].dispatch_source {
        Some(DispatchSource::Direct {
            group_count_x,
            group_count_y,
            ..
        }) => {
            assert_eq!(*group_count_x, 8);
            assert_eq!(*group_count_y, 8);
        }
        _ => panic!("Expected Direct dispatch"),
    }

    // All passes have empty tags.
    for (i, pass) in passes.iter().enumerate() {
        assert!(
            pass.tags.is_empty(),
            "Pass {} '{}' has no tags",
            i, pass.name,
        );
    }
}

/// Multiple builders can be used independently.
#[test]
fn independent_builders_produce_independent_outputs() {
    let mut builder_a = RenderGraphBuilder::new();
    let tex_a = builder_a.create_texture("tex_a", 100, 100, "r8unorm");
    builder_a.add_graphics_pass("pass_a", &[tex_a], None);

    let mut builder_b = RenderGraphBuilder::new();
    let tex_b = builder_b.create_texture("tex_b", 200, 200, "rgba8unorm");
    builder_b.add_compute_pass("pass_b", &[tex_b], &[], (1, 1, 1));

    let (passes_a, resources_a) = builder_a.finalize();
    let (passes_b, resources_b) = builder_b.finalize();

    // Builder A.
    assert_eq!(passes_a.len(), 1);
    assert_eq!(resources_a.len(), 1);
    assert_eq!(passes_a[0].name, "pass_a");
    assert_eq!(passes_a[0].pass_type, PassType::Graphics);
    assert_eq!(resources_a[0].name, "tex_a");

    // Builder B.
    assert_eq!(passes_b.len(), 1);
    assert_eq!(resources_b.len(), 1);
    assert_eq!(passes_b[0].name, "pass_b");
    assert_eq!(passes_b[0].pass_type, PassType::Compute);
    assert_eq!(resources_b[0].name, "tex_b");

    // Handles restart from 0 in each builder.
    assert_eq!(resources_a[0].handle, ResourceHandle(0));
    assert_eq!(resources_b[0].handle, ResourceHandle(0));
    assert_eq!(passes_a[0].index, PassIndex(0));
    assert_eq!(passes_b[0].index, PassIndex(0));
}

/// finalize consumes the builder (cannot call methods after).
/// This test verifies the return type is owned data, not a reference.
#[test]
fn finalize_returns_owned_vectors() {
    let mut builder = RenderGraphBuilder::new();
    builder.create_texture("tex", 100, 100, "r8unorm");
    builder.add_graphics_pass("pass", &[ResourceHandle(0)], None);

    let (passes, resources) = builder.finalize();

    // Owned data: we can move and modify the returned vectors.
    let mut passes = passes;
    let mut resources = resources;

    passes.push(
        IrPass::compute(
            PassIndex(42),
            "late_added",
            DispatchSource::Direct {
                group_count_x: 1,
                group_count_y: 1,
                group_count_z: 1,
            },
            ViewType::Storage,
        ),
    );

    resources.push(
        IrResource::new(
            ResourceHandle(99),
            "late_resource",
            ResourceDesc::Texture2D(TextureDesc {
                width: 1,
                height: 1,
                mip_levels: 1,
                array_layers: 1,
                format: "r8unorm".into(),
            }),
            ResourceLifetime::Transient,
            ResourceState::Uninitialized,
        ),
    );

    assert_eq!(passes.len(), 2, "Can push to passes after finalize");
    assert_eq!(resources.len(), 2, "Can push to resources after finalize");
    assert_eq!(passes[1].index, PassIndex(42));
    assert_eq!(resources[1].handle, ResourceHandle(99));
}
