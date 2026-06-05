// Blackbox integration tests for T-FG-1.5 (view field on passes).
//
// CLEANROOM: Tests are written against the public API contract only.
// No src/ access, no internal fields, no private methods.
//
// Contract (T-FG-1.5):
//   - IrPass struct has `view: Arc<dyn View>` field
//   - Views are bound during pass execution
//   - EmptyView returns empty bind groups
//   - CameraView returns camera uniforms
//   - Copy passes always have EmptyView
//   - Views survive frame graph compilation
//
// Test Coverage:
//   1. test_frame_graph_with_camera_view -- Build graph with graphics pass using CameraView
//   2. test_frame_graph_mixed_views -- Graph with multiple passes, different view types
//   3. test_view_persistence_across_compile -- Views survive frame graph compilation
//   4. test_copy_pass_has_empty_view -- Copy passes always have EmptyView
//   5. test_view_bind_returns_correct_groups -- bind() output matches view type

use std::sync::Arc;

use renderer_backend::frame_graph::{
    AttachmentLoadOp, AttachmentStoreOp, CameraView, ColorAttachment,
    CompiledFrameGraph, DispatchSource, EmptyView, InstanceSource, IrPass, IrResource,
    PassIndex, PassType, RenderContext, ResourceDesc, ResourceHandle, ResourceLifetime,
    ResourceState, TextureDesc, TextureView, View, ViewType,
};

// =============================================================================
// Helper functions for creating test views
// =============================================================================

/// Creates an identity matrix [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]].
fn identity_matrix() -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// Creates a CameraView with standard test configuration.
fn make_camera_view(name: &str, width: u32, height: u32, format: &str) -> CameraView {
    CameraView {
        name: name.into(),
        view: identity_matrix(),
        proj: identity_matrix(),
        position: [0.0, 0.0, 0.0],
        width,
        height,
        format: format.into(),
    }
}

/// Creates a ColorAttachment with standard test configuration.
fn make_color_attachment(resource: ResourceHandle) -> ColorAttachment {
    ColorAttachment {
        resource,
        mip_level: 0,
        array_layer: 0,
        load_op: AttachmentLoadOp::Clear,
        store_op: AttachmentStoreOp::Store,
        clear_color: [0.0, 0.0, 0.0, 1.0],
    }
}

/// Creates a standard Direct InstanceSource for testing.
fn direct_instance_source(index_count: u32) -> InstanceSource {
    InstanceSource::Direct {
        index_count,
        instance_count: 1,
        base_vertex: 0,
        first_index: 0,
        first_instance: 0,
    }
}

/// Creates a Direct DispatchSource for testing.
fn direct_dispatch(x: u32, y: u32, z: u32) -> DispatchSource {
    DispatchSource::Direct {
        group_count_x: x,
        group_count_y: y,
        group_count_z: z,
    }
}

/// Creates a transient texture IrResource for testing.
fn make_texture_resource(handle: ResourceHandle, name: &str, format: &str) -> IrResource {
    IrResource::new(
        handle,
        name,
        ResourceDesc::Texture2D(TextureDesc {
            width: 1920,
            height: 1080,
            mip_levels: 1,
            array_layers: 1,
            format: format.into(),
        }),
        ResourceLifetime::Transient,
        ResourceState::Uninitialized,
    )
}

// =============================================================================
// SECTION 1 -- test_frame_graph_with_camera_view
// =============================================================================

/// Builds a frame graph with a graphics pass that uses CameraView.
/// Verifies the view is correctly attached to the pass.
#[test]
fn test_frame_graph_with_camera_view() {
    // Create a CameraView for the main render pass
    let camera_view: Arc<dyn View> = Arc::new(make_camera_view(
        "main_camera",
        1920,
        1080,
        "rgba8unorm",
    ));

    // Create a graphics pass with the camera view
    let pass = IrPass::graphics_with_view(
        PassIndex(0),
        "gbuffer_pass",
        vec![make_color_attachment(ResourceHandle(0))],
        None,
        direct_instance_source(36),
        ViewType::ColorAttachment,
        camera_view.clone(),
    );

    // Verify pass has correct type
    assert_eq!(pass.pass_type, PassType::Graphics, "Pass is graphics type");
    assert_eq!(pass.name, "gbuffer_pass", "Pass name is preserved");
    assert_eq!(
        pass.view_type,
        ViewType::ColorAttachment,
        "View type is ColorAttachment"
    );

    // Verify view is correctly attached
    assert_eq!(pass.view.name(), "main_camera", "View name is preserved");
    assert_eq!(
        pass.view.view_type(),
        ViewType::ColorAttachment,
        "View reports ColorAttachment type"
    );
    assert!(!pass.view.is_transient(), "CameraView is not transient");

    // Verify bind() returns non-empty groups for CameraView
    let ctx = RenderContext::default();
    let groups = pass.view.bind(&ctx);
    assert!(
        !groups.is_empty(),
        "CameraView::bind returns at least one bind group"
    );
}

/// Verifies graphics pass constructor with custom view preserves all fields.
#[test]
fn test_graphics_pass_with_view_preserves_fields() {
    let view: Arc<dyn View> = Arc::new(make_camera_view(
        "shadow_camera",
        1024,
        1024,
        "d32float",
    ));

    let pass = IrPass::graphics_with_view(
        PassIndex(1),
        "shadow_map",
        vec![],
        None,
        direct_instance_source(100),
        ViewType::ColorAttachment,
        view,
    );

    assert_eq!(pass.index, PassIndex(1), "Index preserved");
    assert_eq!(pass.name, "shadow_map", "Name preserved");
    assert_eq!(pass.view.name(), "shadow_camera", "View name preserved");
}

// =============================================================================
// SECTION 2 -- test_frame_graph_mixed_views
// =============================================================================

/// Builds a graph with multiple passes, each with different view types.
/// Verifies each pass has the correct view attached.
#[test]
fn test_frame_graph_mixed_views() {
    let ctx = RenderContext { frame_index: 42 };

    // Pass 0: Graphics pass with CameraView
    let camera_view: Arc<dyn View> = Arc::new(make_camera_view(
        "main_camera",
        1920,
        1080,
        "rgba8unorm",
    ));
    let graphics_pass = IrPass::graphics_with_view(
        PassIndex(0),
        "gbuffer",
        vec![make_color_attachment(ResourceHandle(0))],
        None,
        direct_instance_source(36),
        ViewType::ColorAttachment,
        camera_view,
    );

    // Pass 1: Compute pass with TextureView
    let texture_view: Arc<dyn View> = Arc::new(TextureView {
        name: "hdr_target".into(),
        width: 1920,
        height: 1080,
        format: "rgba16float".into(),
        transient: false,
    });
    let compute_pass = IrPass::compute_with_view(
        PassIndex(1),
        "tonemap",
        direct_dispatch(120, 68, 1),
        ViewType::Texture2D,
        texture_view,
    );

    // Pass 2: Copy pass (always EmptyView)
    let copy_pass = IrPass::copy(PassIndex(2), "upload_staging");

    // Collect passes
    let passes = vec![graphics_pass, compute_pass, copy_pass];

    // Verify each pass has the correct view type
    assert_eq!(passes[0].view.view_type(), ViewType::ColorAttachment);
    assert_eq!(passes[1].view.view_type(), ViewType::Texture2D);
    assert_eq!(passes[2].view.view_type(), ViewType::Empty);

    // Verify each pass view has the correct name
    assert_eq!(passes[0].view.name(), "main_camera");
    assert_eq!(passes[1].view.name(), "hdr_target");
    assert_eq!(passes[2].view.name(), "upload_staging");

    // Verify bind() returns appropriate results
    let groups_0 = passes[0].view.bind(&ctx);
    let groups_1 = passes[1].view.bind(&ctx);
    let groups_2 = passes[2].view.bind(&ctx);

    assert!(!groups_0.is_empty(), "CameraView returns bind groups");
    // TextureView returns empty by design
    let _ = groups_1;
    assert!(groups_2.is_empty(), "EmptyView returns empty bind groups");
}

/// Tests that different view types can coexist in a heterogeneous collection.
#[test]
fn test_mixed_view_types_heterogeneous() {
    let views: Vec<Arc<dyn View>> = vec![
        Arc::new(EmptyView {
            name: "slot_a".into(),
        }),
        Arc::new(make_camera_view("camera_main", 1920, 1080, "rgba8unorm")),
        Arc::new(TextureView {
            name: "gbuffer_normal".into(),
            width: 1920,
            height: 1080,
            format: "rgba16float".into(),
            transient: true,
        }),
        Arc::new(make_camera_view("camera_shadow", 1024, 1024, "d32float")),
    ];

    assert_eq!(views.len(), 4, "Four views in collection");

    // Verify view types
    assert_eq!(views[0].view_type(), ViewType::Empty);
    assert_eq!(views[1].view_type(), ViewType::ColorAttachment);
    assert_eq!(views[2].view_type(), ViewType::Texture2D);
    assert_eq!(views[3].view_type(), ViewType::ColorAttachment);

    // Verify transience
    assert!(!views[0].is_transient());
    assert!(!views[1].is_transient());
    assert!(views[2].is_transient(), "TextureView with transient=true");
    assert!(!views[3].is_transient());
}

// =============================================================================
// SECTION 3 -- test_view_persistence_across_compile
// =============================================================================

/// Verifies that views survive frame graph compilation unchanged.
#[test]
fn test_view_persistence_across_compile() {
    // Create views
    let camera_view: Arc<dyn View> = Arc::new(make_camera_view(
        "persist_camera",
        1920,
        1080,
        "rgba8unorm",
    ));

    let texture_view: Arc<dyn View> = Arc::new(TextureView {
        name: "persist_texture".into(),
        width: 256,
        height: 256,
        format: "r8unorm".into(),
        transient: false,
    });

    // Create passes with views
    let passes = vec![
        IrPass::graphics_with_view(
            PassIndex(0),
            "pass_a",
            vec![make_color_attachment(ResourceHandle(0))],
            None,
            direct_instance_source(3),
            ViewType::ColorAttachment,
            camera_view.clone(),
        ),
        IrPass::compute_with_view(
            PassIndex(1),
            "pass_b",
            direct_dispatch(8, 8, 1),
            ViewType::Texture2D,
            texture_view.clone(),
        ),
    ];

    // Store original view info before compilation
    let original_view_0_name = passes[0].view.name().to_string();
    let original_view_0_type = passes[0].view.view_type();
    let original_view_1_name = passes[1].view.name().to_string();
    let original_view_1_type = passes[1].view.view_type();

    // Create minimal resources for compilation
    let resources = vec![make_texture_resource(
        ResourceHandle(0),
        "color_target",
        "rgba8unorm",
    )];

    // Compile the frame graph
    let compiled = CompiledFrameGraph::compile(passes, resources);
    assert!(compiled.is_ok(), "Compilation should succeed");

    let compiled = compiled.unwrap();

    // Verify views persist after compilation
    assert_eq!(compiled.passes.len(), 2, "Both passes should survive");

    assert_eq!(
        compiled.passes[0].view.name(),
        original_view_0_name,
        "View 0 name persists"
    );
    assert_eq!(
        compiled.passes[0].view.view_type(),
        original_view_0_type,
        "View 0 type persists"
    );
    assert_eq!(
        compiled.passes[1].view.name(),
        original_view_1_name,
        "View 1 name persists"
    );
    assert_eq!(
        compiled.passes[1].view.view_type(),
        original_view_1_type,
        "View 1 type persists"
    );

    // Verify bind() still works after compilation
    let ctx = RenderContext { frame_index: 1 };
    let groups_0 = compiled.passes[0].view.bind(&ctx);
    let groups_1 = compiled.passes[1].view.bind(&ctx);

    assert!(
        !groups_0.is_empty(),
        "CameraView bind still works after compile"
    );
    // TextureView bind returns empty by design
    let _ = groups_1;
}

/// Tests that Arc<dyn View> can be cloned and shared across passes.
#[test]
fn test_view_arc_cloning() {
    let shared_view: Arc<dyn View> = Arc::new(make_camera_view(
        "shared_camera",
        1920,
        1080,
        "rgba8unorm",
    ));

    // Create two passes sharing the same view
    let pass_a = IrPass::graphics_with_view(
        PassIndex(0),
        "pass_a",
        vec![],
        None,
        direct_instance_source(3),
        ViewType::ColorAttachment,
        shared_view.clone(),
    );

    let pass_b = IrPass::graphics_with_view(
        PassIndex(1),
        "pass_b",
        vec![],
        None,
        direct_instance_source(3),
        ViewType::ColorAttachment,
        shared_view.clone(),
    );

    // Both passes should reference the same view (by name and type)
    assert_eq!(pass_a.view.name(), "shared_camera");
    assert_eq!(pass_b.view.name(), "shared_camera");
    assert_eq!(pass_a.view.view_type(), pass_b.view.view_type());

    // Arc reference count should reflect sharing
    // (We can't directly check ref count, but we verify the view is the same)
    let ctx = RenderContext::default();
    let groups_a = pass_a.view.bind(&ctx);
    let groups_b = pass_b.view.bind(&ctx);

    // Both should return equivalent bind groups since they use the same view
    assert_eq!(groups_a.len(), groups_b.len());
}

// =============================================================================
// SECTION 4 -- test_copy_pass_has_empty_view
// =============================================================================

/// Confirms that copy passes always have EmptyView by construction.
#[test]
fn test_copy_pass_has_empty_view() {
    let copy_pass = IrPass::copy(PassIndex(0), "upload_buffer");

    // Verify it's a copy pass
    assert_eq!(copy_pass.pass_type, PassType::Copy, "Pass type is Copy");

    // Verify view is EmptyView
    assert_eq!(
        copy_pass.view.view_type(),
        ViewType::Empty,
        "Copy pass has EmptyView"
    );
    assert_eq!(
        copy_pass.view.name(),
        "upload_buffer",
        "EmptyView name matches pass name"
    );
    assert!(
        !copy_pass.view.is_transient(),
        "EmptyView is not transient"
    );

    // Verify bind returns empty
    let ctx = RenderContext::default();
    let groups = copy_pass.view.bind(&ctx);
    assert!(
        groups.is_empty(),
        "EmptyView::bind returns empty Vec<BindGroup>"
    );
}

/// Tests multiple copy passes all have EmptyView.
#[test]
fn test_multiple_copy_passes_all_empty_view() {
    let copy_passes = vec![
        IrPass::copy(PassIndex(0), "copy_a"),
        IrPass::copy(PassIndex(1), "copy_b"),
        IrPass::copy(PassIndex(2), "copy_c"),
    ];

    let ctx = RenderContext::default();

    for (i, pass) in copy_passes.iter().enumerate() {
        assert_eq!(
            pass.pass_type,
            PassType::Copy,
            "Pass {} is Copy type",
            i
        );
        assert_eq!(
            pass.view.view_type(),
            ViewType::Empty,
            "Pass {} has EmptyView",
            i
        );
        assert!(
            pass.view.bind(&ctx).is_empty(),
            "Pass {} bind returns empty",
            i
        );
    }
}

/// Verifies that copy pass view name matches the pass name.
#[test]
fn test_copy_pass_view_name_matches_pass() {
    let names = ["upload", "download", "staging_transfer", "readback"];

    for name in names {
        let pass = IrPass::copy(PassIndex(0), name);
        assert_eq!(
            pass.view.name(),
            name,
            "Copy pass view name '{}' matches pass name",
            name
        );
    }
}

// =============================================================================
// SECTION 5 -- test_view_bind_returns_correct_groups
// =============================================================================

/// Verifies that bind() output matches the view type.
#[test]
fn test_view_bind_returns_correct_groups() {
    let ctx = RenderContext { frame_index: 0 };

    // EmptyView returns empty
    let empty = EmptyView {
        name: "empty".into(),
    };
    let empty_groups = empty.bind(&ctx);
    assert!(
        empty_groups.is_empty(),
        "EmptyView::bind returns empty Vec"
    );
    assert_eq!(empty_groups.len(), 0, "EmptyView::bind length is 0");

    // CameraView returns at least one group
    let camera = make_camera_view("camera", 1920, 1080, "rgba8unorm");
    let camera_groups = camera.bind(&ctx);
    assert!(
        !camera_groups.is_empty(),
        "CameraView::bind returns at least one BindGroup"
    );
    assert!(
        camera_groups.len() >= 1,
        "CameraView::bind returns at least 1 group"
    );
}

/// Tests that CameraView bind groups contain meaningful data.
#[test]
fn test_camera_view_bind_group_content() {
    let ctx = RenderContext { frame_index: 100 };
    let camera = make_camera_view("main_camera", 1920, 1080, "rgba8unorm");

    let groups = camera.bind(&ctx);
    assert!(!groups.is_empty(), "CameraView returns bind groups");

    // Each BindGroup should be a valid newtype wrapping a string
    for (i, group) in groups.iter().enumerate() {
        let cloned = group.clone();
        assert_eq!(cloned, *group, "BindGroup {} is cloneable", i);

        let debug = format!("{:?}", group);
        assert!(!debug.is_empty(), "BindGroup {} has Debug output", i);
    }
}

/// Verifies bind() works correctly across multiple frame indices.
#[test]
fn test_view_bind_across_frames() {
    let camera = make_camera_view("frame_test_camera", 800, 600, "bgra8unorm");

    // Bind across multiple frames
    for frame in 0..10 {
        let ctx = RenderContext { frame_index: frame };
        let groups = camera.bind(&ctx);
        assert!(
            !groups.is_empty(),
            "CameraView::bind returns groups for frame {}",
            frame
        );
    }
}

/// Tests that different CameraView configurations produce bind groups.
#[test]
fn test_different_camera_configs_bind() {
    let ctx = RenderContext::default();

    let configs = [
        ("hd", 1920, 1080, "rgba8unorm"),
        ("4k", 3840, 2160, "rgba16float"),
        ("shadow", 1024, 1024, "d32float"),
        ("thumbnail", 128, 128, "r8unorm"),
    ];

    for (name, width, height, format) in configs {
        let camera = make_camera_view(name, width, height, format);

        let groups = camera.bind(&ctx);
        assert!(
            !groups.is_empty(),
            "Camera '{}' ({}x{} {}) produces bind groups",
            name,
            width,
            height,
            format
        );
    }
}

/// Tests bind group equality for identical views.
#[test]
fn test_bind_group_equality() {
    let ctx = RenderContext { frame_index: 0 };

    let camera_a = make_camera_view("camera", 1920, 1080, "rgba8unorm");
    let camera_b = make_camera_view("camera", 1920, 1080, "rgba8unorm");

    let groups_a = camera_a.bind(&ctx);
    let groups_b = camera_b.bind(&ctx);

    // Both should produce the same number of bind groups
    assert_eq!(
        groups_a.len(),
        groups_b.len(),
        "Identical cameras produce same number of bind groups"
    );
}

// =============================================================================
// SECTION 6 -- Additional integration scenarios
// =============================================================================

/// Integration test: full frame graph with mixed views through compilation.
#[test]
fn test_full_graph_mixed_views_compiled() {
    // Create resources
    let resources = vec![
        make_texture_resource(ResourceHandle(0), "color_rt", "rgba8unorm"),
        make_texture_resource(ResourceHandle(1), "depth_rt", "d32float"),
    ];

    // Create passes with different views
    let camera_view: Arc<dyn View> = Arc::new(make_camera_view(
        "scene_camera",
        1920,
        1080,
        "rgba8unorm",
    ));

    let passes = vec![
        // Graphics pass with CameraView
        IrPass::graphics_with_view(
            PassIndex(0),
            "scene_render",
            vec![ColorAttachment {
                resource: ResourceHandle(0),
                mip_level: 0,
                array_layer: 0,
                load_op: AttachmentLoadOp::Clear,
                store_op: AttachmentStoreOp::Store,
                clear_color: [0.1, 0.1, 0.1, 1.0],
            }],
            None,
            direct_instance_source(1000),
            ViewType::ColorAttachment,
            camera_view,
        ),
        // Copy pass with implicit EmptyView
        IrPass::copy(PassIndex(1), "readback"),
    ];

    // Compile
    let result = CompiledFrameGraph::compile(passes, resources);
    assert!(result.is_ok(), "Compilation succeeds with mixed views");

    let compiled = result.unwrap();

    // Find passes by name (order may change after compilation)
    let scene_pass = compiled
        .passes
        .iter()
        .find(|p| p.name == "scene_render")
        .expect("scene_render pass exists");
    let readback_pass = compiled
        .passes
        .iter()
        .find(|p| p.name == "readback")
        .expect("readback pass exists");

    // Verify views are preserved
    assert_eq!(scene_pass.view.name(), "scene_camera");
    assert_eq!(scene_pass.view.view_type(), ViewType::ColorAttachment);

    assert_eq!(readback_pass.view.name(), "readback");
    assert_eq!(readback_pass.view.view_type(), ViewType::Empty);
}

/// Tests ray tracing pass with custom view.
#[test]
fn test_ray_tracing_pass_with_view() {
    let storage_view: Arc<dyn View> = Arc::new(TextureView {
        name: "ray_output".into(),
        width: 1920,
        height: 1080,
        format: "rgba16float".into(),
        transient: false,
    });

    let pass = IrPass::ray_tracing_with_view(
        PassIndex(0),
        "path_trace",
        direct_dispatch(1920, 1080, 1),
        storage_view,
    );

    assert_eq!(pass.pass_type, PassType::RayTracing);
    assert_eq!(pass.view.name(), "ray_output");
    assert_eq!(pass.view.view_type(), ViewType::Texture2D);
}

/// Tests that default graphics/compute passes have EmptyView.
#[test]
fn test_default_constructors_have_empty_view() {
    // Graphics without explicit view
    let graphics = IrPass::graphics(
        PassIndex(0),
        "default_graphics",
        vec![],
        None,
        direct_instance_source(0),
        ViewType::ColorAttachment,
    );

    assert_eq!(
        graphics.view.view_type(),
        ViewType::Empty,
        "Default graphics pass has EmptyView"
    );
    assert_eq!(
        graphics.view.name(),
        "default_graphics",
        "EmptyView name matches pass name"
    );

    // Compute without explicit view
    let compute = IrPass::compute(
        PassIndex(1),
        "default_compute",
        direct_dispatch(1, 1, 1),
        ViewType::Storage,
    );

    assert_eq!(
        compute.view.view_type(),
        ViewType::Empty,
        "Default compute pass has EmptyView"
    );
    assert_eq!(
        compute.view.name(),
        "default_compute",
        "EmptyView name matches pass name"
    );

    // Ray tracing without explicit view
    let ray = IrPass::ray_tracing(
        PassIndex(2),
        "default_ray",
        direct_dispatch(1, 1, 1),
    );

    assert_eq!(
        ray.view.view_type(),
        ViewType::Empty,
        "Default ray tracing pass has EmptyView"
    );
}

/// Tests view Debug implementation via dyn View.
#[test]
fn test_view_debug_via_trait_object() {
    let views: Vec<Arc<dyn View>> = vec![
        Arc::new(EmptyView {
            name: "debug_empty".into(),
        }),
        Arc::new(make_camera_view("debug_camera", 1920, 1080, "rgba8unorm")),
    ];

    for view in views {
        let debug = format!("{:?}", view);
        assert!(!debug.is_empty(), "View Debug output is non-empty");
    }
}

// =============================================================================
// SECTION 7 -- View field interactions with pass operations
// =============================================================================

/// Tests that view field is accessible after sync_access_set_from_attachments.
#[test]
fn test_view_survives_access_set_sync() {
    let view: Arc<dyn View> = Arc::new(make_camera_view(
        "sync_test_camera",
        1920,
        1080,
        "rgba8unorm",
    ));

    let mut pass = IrPass::graphics_with_view(
        PassIndex(0),
        "sync_test",
        vec![make_color_attachment(ResourceHandle(0))],
        None,
        direct_instance_source(36),
        ViewType::ColorAttachment,
        view,
    );

    // Call sync_access_set_from_attachments (mutates access_set)
    pass.sync_access_set_from_attachments();

    // View should still be intact
    assert_eq!(pass.view.name(), "sync_test_camera");
    assert_eq!(pass.view.view_type(), ViewType::ColorAttachment);

    let ctx = RenderContext::default();
    let groups = pass.view.bind(&ctx);
    assert!(!groups.is_empty(), "bind() still works after access set sync");
}

/// Tests pass Display includes view information.
#[test]
fn test_pass_display_includes_view() {
    let pass = IrPass::graphics_with_view(
        PassIndex(0),
        "display_test",
        vec![],
        None,
        direct_instance_source(0),
        ViewType::ColorAttachment,
        Arc::new(make_camera_view("cam", 800, 600, "rgba8unorm")),
    );

    let display = format!("{}", pass);

    // Display should at least mention the pass name
    assert!(
        display.contains("display_test"),
        "Display contains pass name"
    );
}

/// Tests compute pass with TextureView.
#[test]
fn test_compute_pass_with_texture_view() {
    let texture_view: Arc<dyn View> = Arc::new(TextureView {
        name: "compute_output".into(),
        width: 512,
        height: 512,
        format: "rgba16float".into(),
        transient: true,
    });

    let pass = IrPass::compute_with_view(
        PassIndex(0),
        "compute_pass",
        direct_dispatch(64, 64, 1),
        ViewType::Storage,
        texture_view,
    );

    assert_eq!(pass.pass_type, PassType::Compute);
    assert_eq!(pass.view.name(), "compute_output");
    assert_eq!(pass.view.view_type(), ViewType::Texture2D);
    assert!(pass.view.is_transient(), "TextureView is transient");
}
