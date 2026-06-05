// Blackbox contract tests for T-FG-1.4 View trait.
//
// CLEANROOM: Tests are written against the public trait contract only. No src/
// access. No internal fields, no private methods.
//
// Contract (T-FG-1.4):
//   trait View: Send + Sync + Debug {
//       fn view_type(&self) -> ViewType;
//       fn name(&self) -> &str;
//       fn is_transient(&self) -> bool;
//       fn bind(&self, ctx: &RenderContext) -> Vec<BindGroup>;
//   }
//
//   struct RenderContext { pub frame_index: u64 }         // Clone, Debug, Default
//   struct BindGroup(pub String);                          // Clone, Debug, PartialEq, Eq, Hash
//
//   struct EmptyView  { pub name: String }                 // Clone, Debug
//   struct CameraView { pub name, width, height, format }  // Clone, Debug
//   struct TextureView{ pub name, width, height, format, transient }  // Clone, Debug
//
//   ViewType: Empty, Camera, Texture2D, TextureCube, Texture3D, Storage,
//             UniformTexel, StorageTexel, UniformBuffer, StorageBuffer,
//             AccelerationStructure, ColorAttachment
//
// View implementations:
//   - EmptyView:  view_type() -> ViewType::Empty,         is_transient() -> false
//   - CameraView: view_type() -> ViewType::ColorAttachment, is_transient() -> false
//   - TextureView: view_type() -> ViewType::Texture2D,    is_transient() -> self.transient
//
// Scenarios:
//   1.  EmptyView can be constructed and implements View
//   2.  CameraView can be constructed and implements View
//   3.  View trait exposes all four methods via trait object
//   4.  View trait is object-safe (Box<dyn View>)
//   5.  EmptyView::bind returns empty Vec<BindGroup>
//   6.  CameraView::bind returns Vec<BindGroup> with one entry
//   7.  Multiple views can be collected in Vec<Box<dyn View>>
//   8.  RenderContext can be constructed
//   9.  BindGroup newtype wraps a String
//  10.  View types implement Debug
//  11.  View trait has Send + Sync bounds (cross-thread usability)
//  12.  EmptyView and CameraView have different view_type variants
//  13.  Multiple camera views with distinct configurations
//  14.  EmptyView name is preserved
//  15.  CameraView name/width/height/format are preserved
//
// =============================================================================
// Imports
// =============================================================================

use renderer_backend::frame_graph::{
    BindGroup, CameraView, EmptyView, RenderContext, TextureView, View, ViewType,
};

// =============================================================================
// SECTION 1 -- EmptyView: construction and View trait implementation
// =============================================================================

#[test]
fn empty_view_constructs_and_implements_view() {
    let view = EmptyView {
        name: "empty".into(),
    };

    assert_eq!(
        view.view_type(),
        ViewType::Empty,
        "EmptyView reports ViewType::Empty",
    );
    assert_eq!(view.name(), "empty", "EmptyView name is preserved",);
    assert!(!view.is_transient(), "EmptyView is_transient() is false",);
}

// =============================================================================
// SECTION 2 -- CameraView: construction and View trait implementation
// =============================================================================

#[test]
fn camera_view_constructs_and_implements_view() {
    let view = CameraView {
        name: "main_camera".into(),
        width: 1920,
        height: 1080,
        format: "rgba8unorm".into(),
        ..Default::default()
    };

    assert_eq!(
        view.view_type(),
        ViewType::ColorAttachment,
        "CameraView reports ViewType::ColorAttachment",
    );
    assert_eq!(view.name(), "main_camera", "CameraView name is preserved",);
    assert_eq!(view.width, 1920, "CameraView width is preserved");
    assert_eq!(view.height, 1080, "CameraView height is preserved");
    assert_eq!(view.format, "rgba8unorm", "CameraView format is preserved");
    assert!(!view.is_transient(), "CameraView is_transient() is false",);
}

// =============================================================================
// SECTION 3 -- View trait exposes all four methods via trait object
// =============================================================================

#[test]
fn view_trait_four_methods_accessible() {
    // Verify the View trait contract has all expected methods by using a
    // concrete type through dynamic dispatch.

    let ctx = RenderContext::default();
    let empty: Box<dyn View> = Box::new(EmptyView { name: "e".into() });

    // view_type
    assert_eq!(empty.view_type(), ViewType::Empty);

    // name
    assert_eq!(empty.name(), "e");

    // is_transient
    assert!(!empty.is_transient());

    // bind
    let bind_groups = empty.bind(&ctx);
    assert!(
        bind_groups.is_empty(),
        "EmptyView::bind returns empty Vec<BindGroup>",
    );
}

// =============================================================================
// SECTION 4 -- View trait is object-safe (dynamic dispatch via Box<dyn View>)
// =============================================================================

#[test]
fn view_trait_is_object_safe() {
    let ctx = RenderContext::default();

    let empty: Box<dyn View> = Box::new(EmptyView {
        name: "empty_obj".into(),
    });
    let camera: Box<dyn View> = Box::new(CameraView {
        name: "cam_obj".into(),
        width: 800,
        height: 600,
        format: "bgra8unorm".into(),
        ..Default::default()
    });

    // All four trait methods accessible via dynamic dispatch.
    assert_eq!(
        empty.view_type(),
        ViewType::Empty,
        "dyn View: EmptyView.view_type()",
    );
    assert_eq!(
        camera.view_type(),
        ViewType::ColorAttachment,
        "dyn View: CameraView.view_type()",
    );

    assert_eq!(empty.name(), "empty_obj");
    assert_eq!(camera.name(), "cam_obj");

    assert!(!empty.is_transient());
    assert!(!camera.is_transient());

    // bind() works through trait object.
    let empty_groups = empty.bind(&ctx);
    let camera_groups = camera.bind(&ctx);
    assert!(
        empty_groups.is_empty(),
        "EmptyView::bind through trait object returns empty"
    );
    assert!(
        !camera_groups.is_empty(),
        "CameraView::bind through trait object returns groups"
    );
}

// =============================================================================
// SECTION 5 -- EmptyView::bind returns empty Vec<BindGroup>
// =============================================================================

#[test]
fn empty_view_bind_returns_empty() {
    let view = EmptyView {
        name: "empty_bind".into(),
    };
    let ctx = RenderContext::default();

    let groups = view.bind(&ctx);

    assert!(
        groups.is_empty(),
        "EmptyView::bind() returns zero BindGroups",
    );
    assert_eq!(groups.len(), 0, "EmptyView::bind() length is 0",);
}

// =============================================================================
// SECTION 6 -- CameraView::bind returns Vec<BindGroup> with one entry
// =============================================================================

#[test]
fn camera_view_bind_returns_non_empty() {
    let view = CameraView {
        name: "camera_bind".into(),
        width: 1920,
        height: 1080,
        format: "rgba8unorm".into(),
        ..Default::default()
    };
    let ctx = RenderContext::default();

    let groups = view.bind(&ctx);

    assert!(
        !groups.is_empty(),
        "CameraView::bind() returns at least one BindGroup",
    );
}

// =============================================================================
// SECTION 7 -- Heterogeneous collection of view trait objects
// =============================================================================

#[test]
fn heterogeneous_view_collection() {
    let ctx = RenderContext::default();

    let views: Vec<Box<dyn View>> = vec![
        Box::new(EmptyView {
            name: "slot_a".into(),
        }),
        Box::new(CameraView {
            name: "camera_main".into(),
            width: 1920,
            height: 1080,
            format: "rgba8unorm".into(),
        ..Default::default()
    }),
        Box::new(CameraView {
            name: "camera_shadow".into(),
            width: 1024,
            height: 1024,
            format: "d32float".into(),
        ..Default::default()
    }),
    ];

    assert_eq!(views.len(), 3, "three view trait objects in collection");

    // All trait methods accessible on each element.
    assert_eq!(views[0].view_type(), ViewType::Empty);
    assert_eq!(views[1].view_type(), ViewType::ColorAttachment);
    assert_eq!(views[2].view_type(), ViewType::ColorAttachment);

    assert_eq!(views[0].name(), "slot_a");
    assert_eq!(views[1].name(), "camera_main");
    assert_eq!(views[2].name(), "camera_shadow");

    assert!(!views[0].is_transient());
    assert!(!views[1].is_transient());
    assert!(!views[2].is_transient());

    // bind() works on each element through dynamic dispatch.
    let results: Vec<usize> = views.iter().map(|v| v.bind(&ctx).len()).collect();

    assert_eq!(results[0], 0, "EmptyView bind -> 0 groups");
    assert!(results[1] > 0, "CameraView bind -> >0 groups");
    assert!(results[2] > 0, "CameraView bind -> >0 groups");
}

// =============================================================================
// SECTION 8 -- RenderContext can be constructed
// =============================================================================

#[test]
fn render_context_default_construction() {
    let ctx = RenderContext::default();
    assert_eq!(ctx.frame_index, 0, "Default RenderContext frame_index is 0",);
}

#[test]
fn render_context_struct_literal_construction() {
    let ctx = RenderContext { frame_index: 42 };
    assert_eq!(
        ctx.frame_index, 42,
        "RenderContext frame_index set via struct literal",
    );
}

#[test]
fn render_context_frame_index_advances() {
    let ctx_a = RenderContext { frame_index: 0 };
    let ctx_b = RenderContext { frame_index: 1 };
    let ctx_c = RenderContext { frame_index: 100 };

    assert_eq!(ctx_a.frame_index, 0);
    assert_eq!(ctx_b.frame_index, 1);
    assert_eq!(ctx_c.frame_index, 100);
    assert!(
        ctx_c.frame_index > ctx_b.frame_index,
        "frame_index can advance",
    );
}

#[test]
fn render_context_is_cloneable() {
    let ctx = RenderContext { frame_index: 7 };
    let cloned = ctx.clone();
    assert_eq!(
        cloned.frame_index, ctx.frame_index,
        "Clone preserves frame_index",
    );
}

#[test]
fn render_context_is_debug_printable() {
    let ctx = RenderContext { frame_index: 99 };
    let debug = format!("{:?}", ctx);
    assert!(!debug.is_empty(), "RenderContext Debug output is non-empty");
}

// =============================================================================
// SECTION 9 -- BindGroup newtype wraps a String
// =============================================================================

#[test]
fn bind_group_constructs_with_string() {
    let bg = BindGroup("camera_uniform".into());
    assert_eq!(bg.0, "camera_uniform", "BindGroup inner string accessible");
}

#[test]
fn bind_group_clone_preserves_value() {
    let bg = BindGroup("test".into());
    let cloned = bg.clone();
    assert_eq!(cloned.0, bg.0, "Clone preserves inner string");
}

#[test]
fn bind_group_debug_output() {
    let bg = BindGroup("shadow_map".into());
    let debug = format!("{:?}", bg);
    assert!(!debug.is_empty(), "BindGroup Debug output is non-empty");
}

#[test]
fn bind_group_partial_eq_same_value() {
    let a = BindGroup("uniforms".into());
    let b = BindGroup("uniforms".into());
    assert_eq!(a, b, "BindGroups with same string are equal");
}

#[test]
fn bind_group_partial_eq_different_value() {
    let a = BindGroup("camera_a".into());
    let b = BindGroup("camera_b".into());
    assert_ne!(a, b, "BindGroups with different strings are not equal");
}

#[test]
fn bind_group_hash_consistent() {
    use std::collections::HashSet;

    let mut set = HashSet::new();
    set.insert(BindGroup("a".into()));
    set.insert(BindGroup("b".into()));
    set.insert(BindGroup("a".into())); // duplicate

    assert_eq!(
        set.len(),
        2,
        "HashSet deduplicates BindGroups by inner string",
    );
}

// =============================================================================
// SECTION 10 -- View types implement Debug
// =============================================================================

#[test]
fn empty_view_debug_output() {
    let view = EmptyView {
        name: "empty_dbg".into(),
    };
    let debug = format!("{:?}", view);
    assert!(!debug.is_empty(), "EmptyView Debug output is non-empty",);
    assert!(
        debug.contains("empty_dbg") || debug.contains("EmptyView"),
        "EmptyView Debug contains type or name info: got '{}'",
        debug,
    );
}

#[test]
fn camera_view_debug_output() {
    let view = CameraView {
        name: "cam_dbg".into(),
        width: 800,
        height: 600,
        format: "r8unorm".into(),
        ..Default::default()
    };
    let debug = format!("{:?}", view);
    assert!(!debug.is_empty(), "CameraView Debug output is non-empty",);
    assert!(
        debug.contains("cam_dbg") || debug.contains("CameraView"),
        "CameraView Debug contains type or name info: got '{}'",
        debug,
    );
}

// =============================================================================
// SECTION 11 -- View trait has Send + Sync bounds (cross-thread usability)
// =============================================================================

#[test]
fn empty_view_is_send_sync() {
    fn assert_send<T: Send>(_: &T) {}
    fn assert_sync<T: Sync>(_: &T) {}

    let view = EmptyView {
        name: "send_sync".into(),
    };
    assert_send(&view);
    assert_sync(&view);
}

#[test]
fn camera_view_is_send_sync() {
    fn assert_send<T: Send>(_: &T) {}
    fn assert_sync<T: Sync>(_: &T) {}

    let view = CameraView {
        name: "cam_send".into(),
        width: 640,
        height: 480,
        format: "rgba8unorm".into(),
        ..Default::default()
    };
    assert_send(&view);
    assert_sync(&view);
}

// =============================================================================
// SECTION 12 -- ViewType variants: Empty vs ColorAttachment
// =============================================================================

#[test]
fn empty_view_type_is_empty() {
    let view = EmptyView {
        name: "type_check".into(),
    };
    assert_eq!(
        view.view_type(),
        ViewType::Empty,
        "EmptyView.view_type() = Empty",
    );
}

#[test]
fn camera_view_type_is_color_attachment() {
    let view = CameraView {
        name: "type_check".into(),
        width: 1920,
        height: 1080,
        format: "rgba8unorm".into(),
        ..Default::default()
    };
    assert_eq!(
        view.view_type(),
        ViewType::ColorAttachment,
        "CameraView.view_type() = ColorAttachment",
    );
}

#[test]
fn empty_and_camera_view_types_differ() {
    let empty = EmptyView { name: "a".into() };
    let camera = CameraView {
        name: "a".into(),
        width: 1920,
        height: 1080,
        format: "rgba8unorm".into(),
        ..Default::default()
    };

    assert_ne!(
        empty.view_type(),
        camera.view_type(),
        "EmptyView and CameraView have different ViewType values",
    );
    assert_ne!(ViewType::Empty, ViewType::ColorAttachment);
}

// =============================================================================
// SECTION 13 -- Multiple camera views with distinct configurations
// =============================================================================

#[test]
fn multiple_camera_views_distinct_configs() {
    let cameras = vec![
        CameraView {
            name: "main".into(),
            width: 1920,
            height: 1080,
            format: "rgba8unorm".into(),
        ..Default::default()
    },
        CameraView {
            name: "shadow".into(),
            width: 1024,
            height: 1024,
            format: "d32float".into(),
        ..Default::default()
    },
        CameraView {
            name: "ui".into(),
            width: 1920,
            height: 1080,
            format: "bgra8unorm-srgb".into(),
        ..Default::default()
    },
    ];

    assert_eq!(cameras.len(), 3, "three camera views");

    let expected_names = ["main", "shadow", "ui"];
    let expected_widths = [1920, 1024, 1920];
    let expected_heights = [1080, 1024, 1080];
    let expected_formats = ["rgba8unorm", "d32float", "bgra8unorm-srgb"];

    for (i, cam) in cameras.iter().enumerate() {
        assert_eq!(cam.name(), expected_names[i], "CameraView[{}] name", i,);
        assert_eq!(cam.width, expected_widths[i], "CameraView[{}] width", i);
        assert_eq!(cam.height, expected_heights[i], "CameraView[{}] height", i,);
        assert_eq!(cam.format, expected_formats[i], "CameraView[{}] format", i,);
        assert_eq!(cam.view_type(), ViewType::ColorAttachment);
        assert!(!cam.is_transient());
    }
}

// =============================================================================
// SECTION 14 -- EmptyView name is preserved
// =============================================================================

#[test]
fn empty_view_name_preserved() {
    let names = ["slot_a", "slot_b", "unbound", "optional"];
    for &expected in &names {
        let view = EmptyView {
            name: expected.into(),
        };
        assert_eq!(
            view.name(),
            expected,
            "EmptyView name '{}' preserved",
            expected,
        );
    }
}

// =============================================================================
// SECTION 15 -- CameraView all fields preserved
// =============================================================================

#[test]
fn camera_view_fields_preserved() {
    let view = CameraView {
        name: "hdr_camera".into(),
        width: 3840,
        height: 2160,
        format: "rgba16float".into(),
        ..Default::default()
    };

    assert_eq!(view.name(), "hdr_camera");
    assert_eq!(view.width, 3840);
    assert_eq!(view.height, 2160);
    assert_eq!(view.format, "rgba16float");
}

// =============================================================================
// SECTION 16 -- ViewType is Debug-printable
// =============================================================================

#[test]
fn view_type_debug_output() {
    let debug_empty = format!("{:?}", ViewType::Empty);
    let debug_camera = format!("{:?}", ViewType::ColorAttachment);

    assert!(
        !debug_empty.is_empty(),
        "ViewType::Empty Debug is non-empty"
    );
    assert!(
        !debug_camera.is_empty(),
        "ViewType::ColorAttachment Debug is non-empty",
    );
}

// =============================================================================
// SECTION 17 -- RenderContext passed to bind is consumed correctly
// =============================================================================

#[test]
fn bind_accepts_render_context_reference() {
    // Verify that bind() accepts &RenderContext correctly for multiple types.
    let ctx = RenderContext::default();

    let empty = EmptyView {
        name: "bind_ctx".into(),
    };
    let camera = CameraView {
        name: "bind_ctx_cam".into(),
        width: 640,
        height: 480,
        format: "r8unorm".into(),
        ..Default::default()
    };

    // Both calls should succeed (different implementations handle ctx differently).
    let _empty_groups = empty.bind(&ctx);
    let _camera_groups = camera.bind(&ctx);

    // RenderContext can be reused across bind calls.
    let _empty_again = empty.bind(&ctx);
    let _camera_again = camera.bind(&ctx);
}

// =============================================================================
// SECTION 18 -- EmptyView implements Clone
// =============================================================================

#[test]
fn empty_view_clone_preserves_contract() {
    let original = EmptyView {
        name: "clone_me".into(),
    };
    let cloned = original.clone();

    assert_eq!(
        cloned.view_type(),
        original.view_type(),
        "Clone preserves view_type",
    );
    assert_eq!(cloned.name(), original.name(), "Clone preserves name");
    assert_eq!(
        cloned.is_transient(),
        original.is_transient(),
        "Clone preserves is_transient",
    );
}

// =============================================================================
// SECTION 19 -- CameraView implements Clone
// =============================================================================

#[test]
fn camera_view_clone_preserves_contract() {
    let original = CameraView {
        name: "cam_clone".into(),
        width: 1920,
        height: 1080,
        format: "rgba8unorm".into(),
        ..Default::default()
    };
    let cloned = original.clone();

    assert_eq!(
        cloned.view_type(),
        original.view_type(),
        "Clone preserves view_type",
    );
    assert_eq!(cloned.name(), original.name(), "Clone preserves name");
    assert_eq!(cloned.width, original.width, "Clone preserves width");
    assert_eq!(cloned.height, original.height, "Clone preserves height");
    assert_eq!(cloned.format, original.format, "Clone preserves format",);
    assert_eq!(
        cloned.is_transient(),
        original.is_transient(),
        "Clone preserves is_transient",
    );
}

// =============================================================================
// SECTION 20 -- TextureView basic contract (if available)
// =============================================================================

#[test]
fn texture_view_constructs_and_implements_view() {
    let view = TextureView {
        name: "hdr_target".into(),
        width: 1920,
        height: 1080,
        format: "rgba16float".into(),
        transient: false,
    };

    assert_eq!(
        view.view_type(),
        ViewType::Texture2D,
        "TextureView reports ViewType::Texture2D",
    );
    assert_eq!(view.name(), "hdr_target", "TextureView name is preserved",);
    assert!(
        !view.is_transient(),
        "TextureView is_transient() follows transient field",
    );
}

#[test]
fn texture_view_transient_flag() {
    let transient_tex = TextureView {
        name: "tmp_rt".into(),
        width: 256,
        height: 256,
        format: "r16float".into(),
        transient: true,
    };

    assert!(
        transient_tex.is_transient(),
        "TextureView with transient=true reports is_transient() true",
    );
    assert_eq!(
        transient_tex.name(),
        "tmp_rt",
        "Transient TextureView preserves name",
    );
    assert_eq!(
        transient_tex.view_type(),
        ViewType::Texture2D,
        "Transient TextureView still reports ViewType::Texture2D",
    );
}

#[test]
fn texture_view_fields_preserved() {
    let view = TextureView {
        name: "albedo".into(),
        width: 512,
        height: 512,
        format: "bc7_unorm".into(),
        transient: false,
    };

    assert_eq!(view.name(), "albedo");
    assert_eq!(view.width, 512);
    assert_eq!(view.height, 512);
    assert_eq!(view.format, "bc7_unorm");
    assert!(!view.transient);
}

#[test]
fn texture_view_clone_preserves_contract() {
    let original = TextureView {
        name: "gbuffer".into(),
        width: 1920,
        height: 1080,
        format: "rgba8unorm".into(),
        transient: false,
    };
    let cloned = original.clone();

    assert_eq!(cloned.view_type(), original.view_type());
    assert_eq!(cloned.name(), original.name());
    assert_eq!(cloned.width, original.width);
    assert_eq!(cloned.height, original.height);
    assert_eq!(cloned.format, original.format);
    assert_eq!(cloned.transient, original.transient);

    // Transient variant also clones correctly.
    let transient = TextureView {
        name: "tmp".into(),
        width: 128,
        height: 128,
        format: "r32float".into(),
        transient: true,
    };
    let cloned_transient = transient.clone();
    assert!(
        cloned_transient.is_transient(),
        "Clone preserves transient flag",
    );
    assert_eq!(cloned_transient.name(), "tmp");
}

// =============================================================================
// SECTION 21 -- Integration: bind results are BindGroup values
// =============================================================================

#[test]
fn bind_returns_bind_group_type() {
    // Verify that bind() returns Vec<BindGroup> — the correct type.
    let ctx = RenderContext { frame_index: 1 };
    let camera = CameraView {
        name: "integ".into(),
        width: 1920,
        height: 1080,
        format: "rgba8unorm".into(),
        ..Default::default()
    };

    let groups: Vec<BindGroup> = camera.bind(&ctx);
    assert!(!groups.is_empty(), "bind returns BindGroup values");

    // Each element should be a valid BindGroup (clonable, debuggable).
    for (i, bg) in groups.iter().enumerate() {
        let _cloned = bg.clone();
        let debug = format!("{:?}", bg);
        assert!(!debug.is_empty(), "BindGroup[{}] Debug is non-empty", i,);
    }
}

// =============================================================================
// SECTION 22 -- View references via &dyn View function parameter
// =============================================================================

/// Helper that accepts any View via trait reference and queries all methods.
fn describe_view(view: &dyn View, ctx: &RenderContext) -> String {
    let bind_count = view.bind(ctx).len();
    format!(
        "View<{:?}> name='{}' transient={} bind_count={}",
        view.view_type(),
        view.name(),
        view.is_transient(),
        bind_count,
    )
}

#[test]
fn view_trait_as_function_parameter() {
    let ctx = RenderContext::default();

    let empty = EmptyView {
        name: "empty_slot".into(),
    };
    let camera = CameraView {
        name: "main_cam".into(),
        width: 1920,
        height: 1080,
        format: "rgba8unorm".into(),
        ..Default::default()
    };

    let desc_empty = describe_view(&empty, &ctx);
    let desc_camera = describe_view(&camera, &ctx);

    assert!(
        desc_empty.contains("Empty"),
        "empty view description contains variant: got '{}'",
        desc_empty,
    );
    assert!(
        desc_camera.contains("ColorAttachment"),
        "camera view description contains variant: got '{}'",
        desc_camera,
    );
    assert!(
        desc_camera.contains("bind_count="),
        "camera view description contains bind_count: got '{}'",
        desc_camera,
    );
}

// =============================================================================
// SECTION 23 -- BindGroup stored in collections
// =============================================================================

#[test]
fn bind_group_collection() {
    use std::collections::HashSet;

    let mut set = HashSet::new();
    set.insert(BindGroup("camera_uniform".into()));
    set.insert(BindGroup("shadow_map".into()));
    set.insert(BindGroup("camera_uniform".into())); // duplicate

    assert_eq!(set.len(), 2, "BindGroup HashSet deduplicates correctly",);
    assert!(set.contains(&BindGroup("camera_uniform".into())));
    assert!(set.contains(&BindGroup("shadow_map".into())));
}

// =============================================================================
// SECTION 24 -- TextureView Debug impl
// =============================================================================

#[test]
fn texture_view_debug_output() {
    let view = TextureView {
        name: "hdr".into(),
        width: 1920,
        height: 1080,
        format: "rgba16float".into(),
        transient: false,
    };
    let debug = format!("{:?}", view);
    assert!(!debug.is_empty(), "TextureView Debug output is non-empty",);
}

// =============================================================================
// SECTION 25 -- TextureView is_transient obeys constructor parameter
// =============================================================================

#[test]
fn texture_view_transient_obeys_flag() {
    let persistent = TextureView {
        name: "persistent".into(),
        width: 64,
        height: 64,
        format: "r8unorm".into(),
        transient: false,
    };
    let transient = TextureView {
        name: "transient".into(),
        width: 64,
        height: 64,
        format: "r8unorm".into(),
        transient: true,
    };

    assert!(!persistent.is_transient(), "Persistent TextureView");
    assert!(transient.is_transient(), "Transient TextureView");
}
