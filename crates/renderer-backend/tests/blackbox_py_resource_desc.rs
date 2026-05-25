// Blackbox contract tests for T-FG-1.3 PyResourceDesc converter.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::frame_graph::*` -- no internal fields,
// no private methods, no implementation details.
//
// Acceptance criterion (T-FG-1.3):
//   PyResourceDesc::to_ir_resource(handle, is_transient) converts a Python JSON
//   resource descriptor into an IrResource with the correct:
//     - ResourceDesc variant (Texture2D / Texture3D / TextureCube / Buffer)
//     - Dimensional fields (width, height, depth, mip_levels, array_layers, size)
//     - Format string preservation
//     - Lifetime (Transient vs Imported from is_transient)
//     - Initial state (always Uninitialized)
//     - Handle and name preservation
//
// The contract (from mod.rs docs):
//   resource_type | IR variant
//   --------------|------------
//   "Texture3D"   | ResourceDesc::Texture3D  (width, height, depth, mip_levels=1)
//   "TextureCube" | ResourceDesc::TextureCube (width, height, mip_levels=1, array_layers=6)
//   "Buffer"      | ResourceDesc::Buffer      (size = width, usage = "storage")
//   _any other_   | ResourceDesc::Texture2D   (width, height, mip_levels=1, array_layers=1)
//                 |   (includes "Texture2D" and unknown types)
//
// Coverage:
//   1.  Texture2D: width, height, array_layers=1, mip_levels=1, format preserved
//   2.  Texture3D: width, height, depth, mip_levels=1, format preserved
//   3.  TextureCube: width, height, array_layers=6, mip_levels=1, format preserved
//   4.  Buffer: size = width (as u64), usage = "storage", is_indirect_arg = false
//   5.  Transient lifetime  (is_transient = true)
//   6.  Imported lifetime   (is_transient = false)
//   7.  Handle preservation (handle.0 matches input)
//   8.  Name preservation (exact string match)
//   9.  Initial state is always Uninitialized
//  10.  View format override is None
//  11.  Format string passed through for all texture variants
//  12.  Buffer ignores height, depth, format
//  13.  Unknown resource_type falls back to Texture2D
//  14.  Zero dimensions (width=0, height=0) produce correct IR
//  15.  Large dimensions (u32::MAX) produce correct IR
//  16.  Empty name string is preserved
//  17.  Depth field semantics: Texture3D uses it, Texture2D ignores it
//  18.  Explicit "Texture2D" string matches default path
//  19.  Texture2D depth = 0 still produces valid desc (not used)
//  20.  Integration: to_ir_resource output round-trips through emit_resource_bridge

use renderer_backend::frame_graph::{
    PyResourceDesc, ResourceDesc, ResourceHandle,
    ResourceLifetime, ResourceState,
};

// =============================================================================
// Helpers
// =============================================================================

/// Build a PyResourceDesc with given fields.
fn make_desc(
    name: &str,
    resource_type: &str,
    width: u32,
    height: u32,
    depth: u32,
    format: &str,
) -> PyResourceDesc {
    PyResourceDesc {
        name: name.to_string(),
        resource_type: resource_type.to_string(),
        width,
        height,
        depth,
        format: format.to_string(),
        ..Default::default()
    }
}

/// Shorthand: Texture2D descriptor.
fn tex2d(name: &str, w: u32, h: u32, fmt: &str) -> PyResourceDesc {
    make_desc(name, "Texture2D", w, h, 1, fmt)
}

/// Shorthand: Texture3D descriptor.
fn tex3d(name: &str, w: u32, h: u32, d: u32, fmt: &str) -> PyResourceDesc {
    make_desc(name, "Texture3D", w, h, d, fmt)
}

/// Shorthand: TextureCube descriptor.
fn texcube(name: &str, w: u32, h: u32, fmt: &str) -> PyResourceDesc {
    make_desc(name, "TextureCube", w, h, 1, fmt)
}

/// Shorthand: Buffer descriptor.
fn buffer_desc(name: &str, size: u32, fmt: &str) -> PyResourceDesc {
    make_desc(name, "Buffer", size, 1, 1, fmt)
}

// =============================================================================
// SECTION 1 -- Texture2D conversion
// =============================================================================

/// Texture2D conversion sets the correct ResourceDesc variant and preserves
/// width, height, array_layers=1, mip_levels=1, and the format string.
#[test]
fn texture2d_produces_correct_variant_and_fields() {
    let desc = tex2d("gbuffer_albedo", 1920, 1080, "rgba8unorm");
    let ir = desc.to_ir_resource(ResourceHandle(1), true);

    assert_eq!(ir.handle, ResourceHandle(1), "Handle preserved");
    assert_eq!(ir.name, "gbuffer_albedo", "Name preserved");

    match &ir.desc {
        ResourceDesc::Texture2D(t) => {
            assert_eq!(t.width, 1920, "Texture2D width");
            assert_eq!(t.height, 1080, "Texture2D height");
            assert_eq!(t.mip_levels, 1, "Texture2D mip_levels defaults to 1");
            assert_eq!(t.array_layers, 1, "Texture2D array_layers defaults to 1");
            assert_eq!(t.format, "rgba8unorm", "Texture2D format");
        }
        other => panic!("Expected ResourceDesc::Texture2D, got {:?}", other),
    }
}

/// Texture2D with non-square dimensions.
#[test]
fn texture2d_non_square_dimensions() {
    let desc = tex2d("non_square", 800, 600, "bgra8unorm-srgb");
    let ir = desc.to_ir_resource(ResourceHandle(2), true);

    match &ir.desc {
        ResourceDesc::Texture2D(t) => {
            assert_eq!(t.width, 800);
            assert_eq!(t.height, 600);
        }
        _ => panic!("Expected Texture2D"),
    }
}

/// Texture2D with a non-"Texture2D" resource_type string (the default/fallback
/// branch) also produces Texture2D -- this tests that any unknown type falls
/// back to Texture2D.
#[test]
fn texture2d_unknown_resource_type_falls_back() {
    let desc = make_desc("unknown_fallback", "UnknownType", 512, 512, 1, "r8unorm");
    let ir = desc.to_ir_resource(ResourceHandle(3), true);

    match &ir.desc {
        ResourceDesc::Texture2D(t) => {
            assert_eq!(t.width, 512, "Fallback produces Texture2D");
            assert_eq!(t.height, 512);
            assert_eq!(t.format, "r8unorm");
        }
        _ => panic!("Unknown type should fall back to Texture2D"),
    }
}

/// Explicit "Texture2D" resource_type produces identical output to the fallback
/// path -- both go through the same `_` arm.
#[test]
fn texture2d_explicit_vs_fallback_identical() {
    let explicit = make_desc("explicit", "Texture2D", 256, 256, 1, "r32float");
    let fallback = make_desc("fallback", "NonExistent", 256, 256, 1, "r32float");

    let ir_explicit = explicit.to_ir_resource(ResourceHandle(4), true);
    let ir_fallback = fallback.to_ir_resource(ResourceHandle(4), true);

    // Both should produce Texture2D with same dimensions.
    match (&ir_explicit.desc, &ir_fallback.desc) {
        (ResourceDesc::Texture2D(a), ResourceDesc::Texture2D(b)) => {
            assert_eq!(a.width, b.width);
            assert_eq!(a.height, b.height);
            assert_eq!(a.mip_levels, b.mip_levels);
            assert_eq!(a.array_layers, b.array_layers);
            assert_eq!(a.format, b.format);
        }
        _ => panic!("Both must produce Texture2D"),
    }
}

// =============================================================================
// SECTION 2 -- Texture3D conversion
// =============================================================================

/// Texture3D conversion produces the correct variant and uses the depth field.
#[test]
fn texture3d_produces_correct_variant_and_fields() {
    let desc = tex3d("volume_data", 256, 256, 128, "r16float");
    let ir = desc.to_ir_resource(ResourceHandle(10), true);

    assert_eq!(ir.handle, ResourceHandle(10));
    assert_eq!(ir.name, "volume_data");

    match &ir.desc {
        ResourceDesc::Texture3D(t) => {
            assert_eq!(t.width, 256, "Texture3D width");
            assert_eq!(t.height, 256, "Texture3D height");
            assert_eq!(t.depth, 128, "Texture3D depth from PyResourceDesc.depth");
            assert_eq!(t.mip_levels, 1, "Texture3D mip_levels defaults to 1");
            assert_eq!(t.format, "r16float", "Texture3D format");
        }
        other => panic!("Expected ResourceDesc::Texture3D, got {:?}", other),
    }
}

/// Texture3D with asymmetric dimensions.
#[test]
fn texture3d_asymmetric_dimensions() {
    let desc = tex3d("asymmetric", 512, 256, 64, "rgba16float");
    let ir = desc.to_ir_resource(ResourceHandle(11), true);

    match &ir.desc {
        ResourceDesc::Texture3D(t) => {
            assert_eq!(t.width, 512, "Texture3D asymmetric width");
            assert_eq!(t.height, 256, "Texture3D asymmetric height");
            assert_eq!(t.depth, 64, "Texture3D asymmetric depth");
            assert_eq!(t.format, "rgba16float");
        }
        _ => panic!("Expected Texture3D"),
    }
}

/// Texture3D depth=1 produces a valid 3D descriptor (thin volume).
#[test]
fn texture3d_depth_one() {
    let desc = tex3d("thin_volume", 64, 64, 1, "r32float");
    let ir = desc.to_ir_resource(ResourceHandle(12), true);

    match &ir.desc {
        ResourceDesc::Texture3D(t) => {
            assert_eq!(t.depth, 1, "Texture3D depth=1 is allowed");
        }
        _ => panic!("Expected Texture3D"),
    }
}

// =============================================================================
// SECTION 3 -- TextureCube conversion
// =============================================================================

/// TextureCube conversion produces the correct variant with array_layers=6.
#[test]
fn texturecube_produces_correct_variant_and_fields() {
    let desc = texcube("env_map", 1024, 1024, "rgba8unorm");
    let ir = desc.to_ir_resource(ResourceHandle(20), true);

    assert_eq!(ir.handle, ResourceHandle(20));
    assert_eq!(ir.name, "env_map");

    match &ir.desc {
        ResourceDesc::TextureCube(t) => {
            assert_eq!(t.width, 1024, "TextureCube width");
            assert_eq!(t.height, 1024, "TextureCube height");
            assert_eq!(t.mip_levels, 1, "TextureCube mip_levels defaults to 1");
            assert_eq!(t.array_layers, 6, "TextureCube array_layers=6 (six faces)");
            assert_eq!(t.format, "rgba8unorm", "TextureCube format");
        }
        other => panic!("Expected ResourceDesc::TextureCube, got {:?}", other),
    }
}

/// TextureCube with non-square dimensions.
#[test]
fn texturecube_non_square() {
    let desc = texcube("irr_map", 2048, 1024, "rgba16float");
    let ir = desc.to_ir_resource(ResourceHandle(21), true);

    match &ir.desc {
        ResourceDesc::TextureCube(t) => {
            assert_eq!(t.width, 2048);
            assert_eq!(t.height, 1024);
            assert_eq!(t.array_layers, 6, "Always 6 regardless of dimensions");
        }
        _ => panic!("Expected TextureCube"),
    }
}

// =============================================================================
// SECTION 4 -- Buffer conversion
// =============================================================================

/// Buffer conversion produces the correct variant with size = width as u64,
/// usage = "storage", is_indirect_arg = false.
#[test]
fn buffer_produces_correct_variant_and_fields() {
    let desc = buffer_desc("particle_buf", 65536, "ignored_format");
    let ir = desc.to_ir_resource(ResourceHandle(30), true);

    assert_eq!(ir.handle, ResourceHandle(30));
    assert_eq!(ir.name, "particle_buf");

    match &ir.desc {
        ResourceDesc::Buffer(b) => {
            assert_eq!(b.size, 65536u64, "Buffer size = width as u64");
            assert_eq!(b.usage, "storage", "Buffer usage defaults to \"storage\"");
            assert!(!b.is_indirect_arg, "Buffer is_indirect_arg defaults to false");
        }
        other => panic!("Expected ResourceDesc::Buffer, got {:?}", other),
    }
}

/// Buffer with width=0 produces size=0 (edge case).
#[test]
fn buffer_zero_size() {
    let desc = buffer_desc("zero_buf", 0, "ignored");
    let ir = desc.to_ir_resource(ResourceHandle(31), true);

    match &ir.desc {
        ResourceDesc::Buffer(b) => {
            assert_eq!(b.size, 0, "Buffer size can be 0");
        }
        _ => panic!("Expected Buffer"),
    }
}

/// Buffer ignores the height, depth, and format fields of PyResourceDesc.
#[test]
fn buffer_ignores_height_depth_format() {
    // Construct a buffer desc with non-trivial height, depth, and format.
    let desc = make_desc("buf_ignores_extra", "Buffer", 4096, 9999, 8888, "rgba8unorm");
    let ir = desc.to_ir_resource(ResourceHandle(32), true);

    match &ir.desc {
        ResourceDesc::Buffer(b) => {
            assert_eq!(b.size, 4096, "Buffer uses width only for size");
            // height, depth, format are all ignored for buffers
        }
        _ => panic!("Expected Buffer"),
    }
}

// =============================================================================
// SECTION 5 -- Transient lifetime (is_transient = true)
// =============================================================================

/// When is_transient is true, the IrResource lifetime is Transient.
#[test]
fn is_transient_true_creates_transient_lifetime() {
    let desc = tex2d("temp_rt", 800, 600, "rgba8unorm");
    let ir = desc.to_ir_resource(ResourceHandle(40), true);

    assert_eq!(
        ir.lifetime,
        ResourceLifetime::Transient,
        "is_transient=true must produce Transient lifetime"
    );
}

/// All resource types with is_transient=true produce Transient lifetime.
#[test]
fn transient_lifetime_for_all_resource_types() {
    let cases: Vec<(PyResourceDesc, &str)> = vec![
        (tex2d("t2d", 1, 1, "r8unorm"), "Texture2D"),
        (tex3d("t3d", 1, 1, 1, "r32float"), "Texture3D"),
        (texcube("tc", 1, 1, "rgba8unorm"), "TextureCube"),
        (buffer_desc("buf", 64, "ignored"), "Buffer"),
    ];

    for (desc, type_name) in &cases {
        let ir = desc.to_ir_resource(ResourceHandle(0), true);
        assert_eq!(
            ir.lifetime,
            ResourceLifetime::Transient,
            "{} with is_transient=true must be Transient",
            type_name,
        );
    }
}

// =============================================================================
// SECTION 6 -- Imported lifetime (is_transient = false)
// =============================================================================

/// When is_transient is false, the IrResource lifetime is Imported.
#[test]
fn is_transient_false_creates_imported_lifetime() {
    let desc = tex2d("swapchain_image", 1920, 1080, "bgra8unorm-srgb");
    let ir = desc.to_ir_resource(ResourceHandle(50), false);

    assert_eq!(
        ir.lifetime,
        ResourceLifetime::Imported,
        "is_transient=false must produce Imported lifetime"
    );
}

/// All resource types with is_transient=false produce Imported lifetime.
#[test]
fn imported_lifetime_for_all_resource_types() {
    let cases: Vec<(PyResourceDesc, &str)> = vec![
        (tex2d("t2d", 1, 1, "r8unorm"), "Texture2D"),
        (tex3d("t3d", 1, 1, 1, "r32float"), "Texture3D"),
        (texcube("tc", 1, 1, "rgba8unorm"), "TextureCube"),
        (buffer_desc("buf", 64, "ignored"), "Buffer"),
    ];

    for (desc, type_name) in &cases {
        let ir = desc.to_ir_resource(ResourceHandle(0), false);
        assert_eq!(
            ir.lifetime,
            ResourceLifetime::Imported,
            "{} with is_transient=false must be Imported",
            type_name,
        );
    }
}

// =============================================================================
// SECTION 7 -- Handle preservation
// =============================================================================

/// The handle passed to to_ir_resource is preserved in the output.
#[test]
fn handle_preserved_in_output() {
    let desc = tex2d("h_test", 100, 100, "r8unorm");

    let handles: Vec<u32> = vec![0, 1, 42, 255, 65535, 0x7FFFFFFF, 0xFFFFFFFF];

    for &handle in &handles {
        let ir = desc.to_ir_resource(ResourceHandle(handle), true);
        assert_eq!(
            ir.handle,
            ResourceHandle(handle),
            "Handle {} must be preserved",
            handle,
        );
    }
}

// =============================================================================
// SECTION 8 -- Name preservation
// =============================================================================

/// The name string is preserved verbatim in the output IrResource.
#[test]
fn name_preserved_verbatim() {
    let desc = make_desc("custom_resource_name_123", "Texture2D", 1, 1, 1, "r8unorm");
    let ir = desc.to_ir_resource(ResourceHandle(0), true);
    assert_eq!(ir.name, "custom_resource_name_123");
}

/// Name with special characters is preserved.
#[test]
fn name_with_special_characters() {
    let desc = make_desc("res_01.diffuse#2", "Texture2D", 1, 1, 1, "r8unorm");
    let ir = desc.to_ir_resource(ResourceHandle(0), true);
    assert_eq!(ir.name, "res_01.diffuse#2");
}

/// Empty name string is preserved as-is.
#[test]
fn empty_name_is_preserved() {
    let desc = make_desc("", "Texture2D", 1, 1, 1, "r8unorm");
    let ir = desc.to_ir_resource(ResourceHandle(0), true);
    assert_eq!(ir.name, "", "Empty name string must be preserved");
}

// =============================================================================
// SECTION 9 -- Initial state and view_format_override
// =============================================================================

/// The initial_state is always Uninitialized, regardless of resource type or
/// is_transient value.
#[test]
fn initial_state_always_uninitialized() {
    let cases: Vec<(PyResourceDesc, bool)> = vec![
        (tex2d("a", 1, 1, "r8unorm"), true),
        (tex2d("b", 1, 1, "r8unorm"), false),
        (tex3d("c", 1, 1, 1, "r32float"), true),
        (texcube("d", 1, 1, "rgba8unorm"), false),
        (buffer_desc("e", 64, "ignored"), true),
    ];

    for (desc, is_transient) in &cases {
        let ir = desc.to_ir_resource(ResourceHandle(0), *is_transient);
        assert_eq!(
            ir.initial_state,
            ResourceState::Uninitialized,
            "Resource '{}' must have initial_state=Uninitialized",
            ir.name,
        );
    }
}

/// view_format_override is always None after to_ir_resource conversion.
#[test]
fn view_format_override_is_none() {
    let desc = tex2d("no_override", 800, 600, "rgba8unorm");
    let ir = desc.to_ir_resource(ResourceHandle(0), true);
    assert!(
        ir.view_format_override.is_none(),
        "view_format_override must be None after conversion"
    );
}

// =============================================================================
// SECTION 10 -- Format string preservation for textures
// =============================================================================

/// All texture variants preserve their format string.
#[test]
fn texture_format_preserved_all_variants() {
    let cases: Vec<(PyResourceDesc, &str)> = vec![
        (tex2d("a", 100, 100, "rgba8unorm-srgb"), "rgba8unorm-srgb"),
        (tex2d("b", 100, 100, "r32float"), "r32float"),
        (tex2d("c", 100, 100, "depth32float"), "depth32float"),
        (tex2d("d", 100, 100, "bc7_unorm"), "bc7_unorm"),
        (tex3d("e", 64, 64, 32, "r16float"), "r16float"),
        (tex3d("f", 64, 64, 32, "rgba16float"), "rgba16float"),
        (texcube("g", 512, 512, "bgra8unorm"), "bgra8unorm"),
        (texcube("h", 512, 512, "rgba8unorm-srgb"), "rgba8unorm-srgb"),
    ];

    for (desc, expected_format) in &cases {
        let ir = desc.to_ir_resource(ResourceHandle(0), true);
        match &ir.desc {
            ResourceDesc::Texture2D(t) => {
                assert_eq!(t.format, *expected_format, "Texture2D format for '{}'", ir.name);
            }
            ResourceDesc::Texture3D(t) => {
                assert_eq!(t.format, *expected_format, "Texture3D format for '{}'", ir.name);
            }
            ResourceDesc::TextureCube(t) => {
                assert_eq!(t.format, *expected_format, "TextureCube format for '{}'", ir.name);
            }
            ResourceDesc::Buffer(_) => {
                // Buffer does not carry format.
            }
        }
    }
}

// =============================================================================
// SECTION 11 -- Field-level access: all public fields on output IrResource
// =============================================================================

/// Every field on the output IrResource is accessible through the public API.
#[test]
fn all_ir_resource_fields_accessible() {
    let desc = tex2d("full_check", 1920, 1080, "rgba16float");
    let ir = desc.to_ir_resource(ResourceHandle(77), true);

    // handle
    assert_eq!(ir.handle.0, 77);
    // name
    assert_eq!(ir.name, "full_check");
    // desc -- checked via pattern match
    assert!(
        matches!(ir.desc, ResourceDesc::Texture2D(_)),
        "desc must be Texture2D"
    );
    // lifetime
    assert_eq!(ir.lifetime, ResourceLifetime::Transient);
    // initial_state
    assert_eq!(ir.initial_state, ResourceState::Uninitialized);
    // view_format_override
    assert!(ir.view_format_override.is_none());
}

// =============================================================================
// SECTION 12 -- Texture2D ignores depth field
// =============================================================================

/// The depth field of PyResourceDesc is NOT used for Texture2D conversion.
/// Texture2D always uses array_layers=1 regardless of depth.
#[test]
fn texture2d_ignores_depth_field() {
    // Construct with depth=99 -- should be ignored for Texture2D.
    let desc = make_desc("ignores_depth", "Texture2D", 800, 600, 99, "r8unorm");
    let ir = desc.to_ir_resource(ResourceHandle(80), true);

    match &ir.desc {
        ResourceDesc::Texture2D(t) => {
            // Depth from PyResourceDesc should NOT leak into array_layers or
            // any other field. Texture2D has no depth field.
            assert_eq!(t.width, 800, "width unchanged");
            assert_eq!(t.height, 600, "height unchanged");
            assert_eq!(t.array_layers, 1, "array_layers=1 regardless of depth");
            assert_eq!(t.mip_levels, 1, "mip_levels=1");
        }
        _ => panic!("Expected Texture2D"),
    }
}

/// Texture2D with zero depth still produces a valid descriptor.
#[test]
fn texture2d_zero_depth_still_valid() {
    let desc = make_desc("zero_depth", "Texture2D", 800, 600, 0, "r8unorm");
    let ir = desc.to_ir_resource(ResourceHandle(81), true);

    match &ir.desc {
        ResourceDesc::Texture2D(t) => {
            assert_eq!(t.width, 800);
            assert_eq!(t.height, 600);
            assert_eq!(t.array_layers, 1);
        }
        _ => panic!("Expected Texture2D"),
    }
}

// =============================================================================
// SECTION 13 -- TextureCube ignores depth field
// =============================================================================

/// TextureCube uses array_layers=6 always, regardless of PyResourceDesc.depth.
#[test]
fn texturecube_uses_six_layers_regardless_of_depth() {
    // depth=99 should not affect TextureCube's array_layers.
    let desc = make_desc("cube_ignores_depth", "TextureCube", 1024, 1024, 99, "rgba8unorm");
    let ir = desc.to_ir_resource(ResourceHandle(85), true);

    match &ir.desc {
        ResourceDesc::TextureCube(t) => {
            assert_eq!(t.width, 1024);
            assert_eq!(t.height, 1024);
            assert_eq!(t.array_layers, 6, "TextureCube always 6 layers");
            assert_eq!(t.mip_levels, 1);
        }
        _ => panic!("Expected TextureCube"),
    }
}

// =============================================================================
// SECTION 14 -- Large dimensions (boundary values)
// =============================================================================

/// Large u32 dimension values are preserved correctly.
#[test]
fn large_dimensions_preserved() {
    let desc = make_desc("large", "Texture2D", u32::MAX, u32::MAX, u32::MAX, "r8unorm");
    let ir = desc.to_ir_resource(ResourceHandle(90), true);

    match &ir.desc {
        ResourceDesc::Texture2D(t) => {
            assert_eq!(t.width, u32::MAX, "Max width");
            assert_eq!(t.height, u32::MAX, "Max height");
        }
        _ => panic!("Expected Texture2D"),
    }
}

/// Large buffer size (from width) is preserved.
#[test]
fn large_buffer_size() {
    let desc = make_desc("big_buf", "Buffer", u32::MAX, 1, 1, "ignored");
    let ir = desc.to_ir_resource(ResourceHandle(91), true);

    match &ir.desc {
        ResourceDesc::Buffer(b) => {
            assert_eq!(b.size, u32::MAX as u64, "Buffer size from u32::MAX width");
        }
        _ => panic!("Expected Buffer"),
    }
}

/// Large Texture3D depth is preserved.
#[test]
fn large_texture3d_depth() {
    let desc = make_desc("big_volume", "Texture3D", 4096, 4096, u32::MAX, "r32float");
    let ir = desc.to_ir_resource(ResourceHandle(92), true);

    match &ir.desc {
        ResourceDesc::Texture3D(t) => {
            assert_eq!(t.depth, u32::MAX, "Max depth for Texture3D");
        }
        _ => panic!("Expected Texture3D"),
    }
}

// =============================================================================
// SECTION 15 -- Zero dimensions
// =============================================================================

/// Zero-width and zero-height textures are valid edge cases.
#[test]
fn zero_dimension_texture2d() {
    let desc = tex2d("zero_tex", 0, 0, "r8unorm");
    let ir = desc.to_ir_resource(ResourceHandle(100), true);

    match &ir.desc {
        ResourceDesc::Texture2D(t) => {
            assert_eq!(t.width, 0, "Zero width");
            assert_eq!(t.height, 0, "Zero height");
            assert_eq!(t.mip_levels, 1);
            assert_eq!(t.array_layers, 1);
        }
        _ => panic!("Expected Texture2D"),
    }
}

/// Zero-dimension Texture3D.
#[test]
fn zero_dimension_texture3d() {
    let desc = tex3d("zero_vol", 0, 0, 0, "r32float");
    let ir = desc.to_ir_resource(ResourceHandle(101), true);

    match &ir.desc {
        ResourceDesc::Texture3D(t) => {
            assert_eq!(t.width, 0);
            assert_eq!(t.height, 0);
            assert_eq!(t.depth, 0);
        }
        _ => panic!("Expected Texture3D"),
    }
}

// =============================================================================
// SECTION 16 -- resource_type string is case-sensitive
// =============================================================================

/// The resource_type string matching is case-sensitive. "texture2d" (lowercase)
/// does NOT match "Texture2D" and falls through to the default Texture2D arm
/// (same result, but hits a different code path).
#[test]
fn resource_type_case_sensitivity_lowercase_falls_to_default() {
    // Lowercase "texture2d" is not "Texture2D", "Texture3D", "TextureCube",
    // or "Buffer" -- so it hits the default arm, which produces Texture2D.
    let desc = make_desc("lower_t2d", "texture2d", 256, 256, 1, "r8unorm");
    let ir = desc.to_ir_resource(ResourceHandle(110), true);

    match &ir.desc {
        ResourceDesc::Texture2D(t) => {
            assert_eq!(t.width, 256, "Fallback to Texture2D for lowercase");
        }
        _ => panic!("Expected Texture2D from fallback"),
    }
}

/// "buffer" (lowercase) does not match "Buffer", so it falls back to
/// Texture2D (not Buffer).
#[test]
fn lowercase_buffer_falls_to_texture2d() {
    let desc = make_desc("lower_buf", "buffer", 4096, 1, 1, "ignored");
    let ir = desc.to_ir_resource(ResourceHandle(111), true);

    // Lowercase "buffer" is NOT matched by the "Buffer" arm, so it falls back
    // to Texture2D.
    match &ir.desc {
        ResourceDesc::Texture2D(_) => {} // Expected fallback
        ResourceDesc::Buffer(_) => {
            panic!("Lowercase 'buffer' should NOT match Buffer arm");
        }
        _ => panic!("Expected Texture2D fallback"),
    }
}

// =============================================================================
// SECTION 17 -- All four resource_type strings are distinct
// =============================================================================

/// Each of the four documented resource_type strings produces a distinct
/// ResourceDesc variant.
#[test]
fn each_resource_type_produces_correct_distinct_variant() {
    let cases: Vec<(PyResourceDesc, &str)> = vec![
        (tex2d("t2d", 1, 1, "r8unorm"), "Texture2D"),
        (tex3d("t3d", 1, 1, 1, "r32float"), "Texture3D"),
        (texcube("tc", 1, 1, "rgba8unorm"), "TextureCube"),
        (buffer_desc("buf", 64, "ignored"), "Buffer"),
    ];

    for (desc, type_name) in &cases {
        let ir = desc.to_ir_resource(ResourceHandle(0), true);
        match *type_name {
            "Texture2D" => assert!(
                matches!(ir.desc, ResourceDesc::Texture2D(_)),
                "Must be Texture2D"
            ),
            "Texture3D" => assert!(
                matches!(ir.desc, ResourceDesc::Texture3D(_)),
                "Must be Texture3D"
            ),
            "TextureCube" => assert!(
                matches!(ir.desc, ResourceDesc::TextureCube(_)),
                "Must be TextureCube"
            ),
            "Buffer" => assert!(
                matches!(ir.desc, ResourceDesc::Buffer(_)),
                "Must be Buffer"
            ),
            _ => unreachable!(),
        }
    }
}

// =============================================================================
// SECTION 18 -- Default field values: mip_levels=1 across all texture variants
// =============================================================================

/// All texture variants from PyResourceDesc produce mip_levels=1.
#[test]
fn texture_mip_levels_default_to_one() {
    let cases: Vec<(PyResourceDesc, &str)> = vec![
        (tex2d("a", 512, 512, "r8unorm"), "Texture2D"),
        (tex3d("b", 64, 64, 32, "r32float"), "Texture3D"),
        (texcube("c", 1024, 1024, "rgba8unorm"), "TextureCube"),
    ];

    for (desc, _type_name) in &cases {
        let ir = desc.to_ir_resource(ResourceHandle(0), true);
        match &ir.desc {
            ResourceDesc::Texture2D(t) => assert_eq!(t.mip_levels, 1, "Texture2D mip_levels"),
            ResourceDesc::Texture3D(t) => assert_eq!(t.mip_levels, 1, "Texture3D mip_levels"),
            ResourceDesc::TextureCube(t) => assert_eq!(t.mip_levels, 1, "TextureCube mip_levels"),
            ResourceDesc::Buffer(_) => {} // Buffer has no mip_levels
        }
    }
}

// =============================================================================
// SECTION 19 -- ResourceDesc is used in IrResource::new correctly
// =============================================================================

/// Verify that the desc field of the output IrResource routes through
/// IrResource::new by checking that all constructor invariants hold.
/// IrResource::new stores desc verbatim, so each field inside the desc
/// variant should match the PyResourceDesc-derived values.
#[test]
fn ir_resource_new_fields_are_correct() {
    // Test that the desc is stored (IrResource::new is the underlying
    // constructor called by to_ir_resource).
    let desc = tex2d("check_new", 1920, 1080, "rgba8unorm");
    let ir = desc.to_ir_resource(ResourceHandle(5), true);

    // Verify IrResource::new was used (handle, name, lifetime, initial_state
    // all match the to_ir_resource contract).
    assert_eq!(ir.handle, ResourceHandle(5));
    assert_eq!(ir.name, "check_new");
    assert_eq!(ir.lifetime, ResourceLifetime::Transient);
    assert_eq!(ir.initial_state, ResourceState::Uninitialized);
}

// =============================================================================
// SECTION 20 -- Integration: to_ir_resource round-trips through
//               emit_resource_bridge (schema consistency)
// =============================================================================

/// The IrResource produced by PyResourceDesc::to_ir_resource can be serialized
/// by emit_resource_bridge and all key fields are present in the JSON output.
#[test]
fn to_ir_resource_output_serializes_via_emit_bridge() {
    use renderer_backend::frame_graph::emit_resource_bridge;

    let desc = tex2d("roundtrip_check", 1920, 1080, "rgba8unorm");
    let ir = desc.to_ir_resource(ResourceHandle(42), true);
    let json = emit_resource_bridge(&ir);

    // Must be a JSON object with expected keys.
    assert!(json.is_object(), "emit_resource_bridge must produce an object");
    let obj = json.as_object().unwrap();
    assert!(obj.contains_key("name"), "Must have 'name' key");
    assert!(obj.contains_key("handle"), "Must have 'handle' key");
    assert!(obj.contains_key("resource_type"), "Must have 'resource_type' key");
    assert!(obj.contains_key("dimensions"), "Must have 'dimensions' key");
    assert!(obj.contains_key("format"), "Must have 'format' key");
    assert!(obj.contains_key("transient"), "Must have 'transient' key");
    assert!(obj.contains_key("initial_state"), "Must have 'initial_state' key");

    // Verify field values are consistent with PyResourceDesc inputs.
    assert_eq!(json["name"].as_str(), Some("roundtrip_check"));
    assert_eq!(json["handle"].as_u64(), Some(42));
    assert_eq!(json["resource_type"].as_str(), Some("texture2d"));
    assert_eq!(json["dimensions"]["width"].as_u64(), Some(1920));
    assert_eq!(json["dimensions"]["height"].as_u64(), Some(1080));
    assert_eq!(json["format"].as_str(), Some("rgba8unorm"));
    assert_eq!(json["transient"].as_bool(), Some(true));
    assert_eq!(json["initial_state"].as_str(), Some("Uninitialized"));
}

/// Texture3D from PyResourceDesc serializes correctly through emit_resource_bridge.
#[test]
fn texture3d_to_ir_resource_serializes_via_emit_bridge() {
    use renderer_backend::frame_graph::emit_resource_bridge;

    let desc = tex3d("volume_scan", 256, 256, 64, "r16float");
    let ir = desc.to_ir_resource(ResourceHandle(55), false);
    let json = emit_resource_bridge(&ir);

    assert_eq!(json["resource_type"].as_str(), Some("texture3d"));
    assert_eq!(json["dimensions"]["depth"].as_u64(), Some(64));
    assert_eq!(json["format"].as_str(), Some("r16float"));
    assert_eq!(json["transient"].as_bool(), Some(false));
}

/// Buffer from PyResourceDesc serializes correctly through emit_resource_bridge.
#[test]
fn buffer_to_ir_resource_serializes_via_emit_bridge() {
    use renderer_backend::frame_graph::emit_resource_bridge;

    let desc = buffer_desc("storage_buf", 65536, "ignored");
    let ir = desc.to_ir_resource(ResourceHandle(60), true);
    let json = emit_resource_bridge(&ir);

    assert_eq!(json["resource_type"].as_str(), Some("buffer"));
    assert_eq!(json["dimensions"]["size"].as_u64(), Some(65536));
    assert!(json["format"].is_null(), "Buffer format must be null");
    assert_eq!(json["transient"].as_bool(), Some(true));
}

/// TextureCube from PyResourceDesc serializes correctly through emit_resource_bridge.
#[test]
fn texturecube_to_ir_resource_serializes_via_emit_bridge() {
    use renderer_backend::frame_graph::emit_resource_bridge;

    let desc = texcube("skybox", 2048, 2048, "rgba8unorm");
    let ir = desc.to_ir_resource(ResourceHandle(70), false);
    let json = emit_resource_bridge(&ir);

    assert_eq!(json["resource_type"].as_str(), Some("texturecube"));
    assert_eq!(json["dimensions"]["depth"].as_u64(), Some(6));
    assert_eq!(json["format"].as_str(), Some("rgba8unorm"));
    assert_eq!(json["transient"].as_bool(), Some(false));
}
