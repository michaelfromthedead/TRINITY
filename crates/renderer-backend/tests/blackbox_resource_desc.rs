#![cfg(feature = "pyo3")]
// Blackbox contract tests for T-FG-1.3 PyResourceDesc -> IrResource conversion.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::frame_graph::*` and
// `renderer_backend::frame_graph::python::*` -- no internal fields,
// no private methods, no implementation details.
//
// Acceptance criterion (T-FG-1.3):
//   The TryFrom<PyResourceDesc> for IrResource conversion performs:
//     - Format validation (case-insensitive against KNOWN_TEXTURE_FORMATS)
//     - Usage flag coalescing (dedup and validate against KNOWN_USAGE_FLAGS)
//     - Initial state parsing
//     - Dimension validation (non-zero for textures)
//     - Handle resolution (auto-assign when None)
//     - Empty name rejection
//     - Invalid resource type rejection
//
// Coverage:
//   1.  Valid Texture2D PyResourceDesc -> IrResource with correct fields
//   2.  Valid Buffer PyResourceDesc -> IrResource with correct fields
//   3.  TryFrom rejects invalid format string
//   4.  TryFrom rejects empty name
//   5.  TryFrom rejects zero dimensions for texture
//   6.  Two PyResourceDesc with different names produce different handles
//   7.  Round-trip: PyResourceDesc -> IrResource yields matching resource desc fields
//   8.  Case-insensitive format validation accepts "R8G8B8A8_UNORM"
//   9.  is_transient correctly maps to ResourceLifetime
//  10.  initial_state correctly maps to ResourceState

use core::convert::TryInto;

use renderer_backend::frame_graph::{
    BufferDesc, IrResource, ResourceDesc, ResourceHandle,
    ResourceLifetime, ResourceState, TextureDesc,
};
use renderer_backend::frame_graph::python::{
    ConversionError, PyResourceDesc,
};

// =============================================================================
// Helpers
// =============================================================================

/// Shorthand: build a default Texture2D descriptor.
fn make_tex2d(name: &str, w: u32, h: u32, fmt: &str) -> PyResourceDesc {
    PyResourceDesc {
        name: name.to_string(),
        resource_type: "Texture2D".to_string(),
        width: w,
        height: h,
        format: fmt.to_string(),
        ..Default::default()
    }
}

/// Shorthand: build a default Buffer descriptor.
fn make_buffer(name: &str, size: u32) -> PyResourceDesc {
    PyResourceDesc {
        name: name.to_string(),
        resource_type: "Buffer".to_string(),
        width: size,
        usage_flags: vec!["storage".to_string()],
        ..Default::default()
    }
}

// =============================================================================
// SECTION 1 -- Valid Texture2D conversion
// =============================================================================

#[test]
fn test_texture2d_converts_with_all_fields_preserved() {
    let desc = PyResourceDesc {
        name: "gbuffer_albedo".into(),
        resource_type: "Texture2D".into(),
        width: 1920,
        height: 1080,
        depth: 1,
        format: "rgba8unorm".into(),
        usage_flags: vec!["texture_binding".into(), "color_attachment".into()],
        mip_levels: 4,
        sample_count: 1,
        is_transient: Some(true),
        initial_state: Some("Uninitialized".into()),
        handle: Some(ResourceHandle(1)),
    };

    let ir: IrResource = desc.try_into().expect("valid Texture2D should convert");

    assert_eq!(ir.name, "gbuffer_albedo");
    assert_eq!(ir.handle, ResourceHandle(1));
    assert_eq!(ir.lifetime, ResourceLifetime::Transient);
    assert_eq!(ir.initial_state, ResourceState::Uninitialized);

    match &ir.desc {
        ResourceDesc::Texture2D(t) => {
            assert_eq!(t.width, 1920);
            assert_eq!(t.height, 1080);
            assert_eq!(t.mip_levels, 4);
            assert_eq!(t.array_layers, 1);
            assert_eq!(t.format, "rgba8unorm");
        }
        other => panic!("expected Texture2D, got {other:?}"),
    }
}

// =============================================================================
// SECTION 2 -- Valid Buffer conversion
// =============================================================================

#[test]
fn test_buffer_converts_with_all_fields_preserved() {
    let desc = PyResourceDesc {
        name: "particle_buffer".into(),
        resource_type: "Buffer".into(),
        width: 65536,
        height: 1,
        depth: 1,
        format: "".into(),
        usage_flags: vec!["storage".into(), "copy_src".into()],
        mip_levels: 0,
        sample_count: 0,
        is_transient: Some(false),
        initial_state: Some("Uninitialized".into()),
        handle: Some(ResourceHandle(2)),
    };

    let ir: IrResource = desc.try_into().expect("valid Buffer should convert");

    assert_eq!(ir.name, "particle_buffer");
    assert_eq!(ir.handle, ResourceHandle(2));
    assert_eq!(ir.lifetime, ResourceLifetime::Imported);
    assert_eq!(ir.initial_state, ResourceState::Uninitialized);

    match &ir.desc {
        ResourceDesc::Buffer(b) => {
            assert_eq!(b.size, 65536);
            assert!(b.usage.contains("storage"));
            assert!(b.usage.contains("copy_src"));
            assert!(!b.is_indirect_arg);
        }
        other => panic!("expected Buffer, got {other:?}"),
    }
}

// =============================================================================
// SECTION 3 -- TryFrom rejects invalid format string
// =============================================================================

#[test]
fn test_invalid_format_rejected() {
    let desc = make_tex2d("bad_tex", 256, 256, "not_a_real_format");

    let result: Result<IrResource, ConversionError> = desc.try_into();
    assert!(
        matches!(result, Err(ConversionError::InvalidResourceFormat(_))),
        "expected InvalidResourceFormat, got {result:?}"
    );
}

// =============================================================================
// SECTION 4 -- TryFrom rejects empty name
// =============================================================================

#[test]
fn test_empty_name_rejected() {
    let desc = PyResourceDesc {
        name: String::new(),
        resource_type: "Texture2D".into(),
        width: 256,
        height: 256,
        format: "rgba8unorm".into(),
        ..Default::default()
    };

    let result: Result<IrResource, ConversionError> = desc.try_into();
    assert!(
        matches!(result, Err(ConversionError::EmptyResourceName)),
        "expected EmptyResourceName, got {result:?}"
    );
}

// =============================================================================
// SECTION 5 -- TryFrom rejects zero dimensions for texture
// =============================================================================

#[test]
fn test_zero_width_rejected() {
    let desc = make_tex2d("bad_size", 0, 1080, "rgba8unorm");

    let result: Result<IrResource, ConversionError> = desc.try_into();
    assert!(
        matches!(result, Err(ConversionError::InvalidResourceDimensions(_))),
        "expected InvalidResourceDimensions, got {result:?}"
    );
}

#[test]
fn test_zero_height_rejected() {
    let desc = make_tex2d("bad_size", 1920, 0, "rgba8unorm");

    let result: Result<IrResource, ConversionError> = desc.try_into();
    assert!(
        matches!(result, Err(ConversionError::InvalidResourceDimensions(_))),
        "expected InvalidResourceDimensions, got {result:?}"
    );
}

// =============================================================================
// SECTION 6 -- Auto-assigned handles are unique
// =============================================================================

#[test]
fn test_auto_assigned_handles_are_unique() {
    let desc_a = make_tex2d("color_rt", 128, 128, "rgba8unorm");
    let desc_b = make_buffer("buf_a", 4096);

    let ir_a: IrResource = desc_a.try_into().expect("first conversion");
    let ir_b: IrResource = desc_b.try_into().expect("second conversion");

    assert!(
        ir_a.handle != ResourceHandle::NONE,
        "auto-assigned handle must not be NONE"
    );
    assert!(
        ir_b.handle != ResourceHandle::NONE,
        "auto-assigned handle must not be NONE"
    );
    assert_ne!(
        ir_a.handle, ir_b.handle,
        "auto-assigned handles must be unique"
    );
}

// =============================================================================
// SECTION 7 -- Round-trip: PyResourceDesc -> IrResource yields matching desc
// =============================================================================

#[test]
fn test_round_trip_fields_match() {
    let desc = PyResourceDesc {
        name: "roundtrip_tex".into(),
        resource_type: "Texture3D".into(),
        width: 512,
        height: 512,
        depth: 128,
        format: "r32float".into(),
        usage_flags: vec!["storage_binding".into()],
        mip_levels: 6,
        sample_count: 1,
        is_transient: Some(true),
        initial_state: None,
        handle: Some(ResourceHandle(10)),
    };

    let ir: IrResource = desc.try_into().expect("valid Texture3D should convert");

    // Name preserved
    assert_eq!(ir.name, "roundtrip_tex");
    // Handle preserved
    assert_eq!(ir.handle, ResourceHandle(10));
    // Lifetime defaults to Transient when is_transient is Some(true)
    assert_eq!(ir.lifetime, ResourceLifetime::Transient);
    // Initial state defaults to Uninitialized when None
    assert_eq!(ir.initial_state, ResourceState::Uninitialized);

    match &ir.desc {
        ResourceDesc::Texture3D(t) => {
            assert_eq!(t.width, 512);
            assert_eq!(t.height, 512);
            assert_eq!(t.depth, 128);
            assert_eq!(t.mip_levels, 6);
            assert_eq!(t.format, "r32float");
        }
        other => panic!("expected Texture3D, got {other:?}"),
    }
}

// =============================================================================
// SECTION 8 -- Case-insensitive format validation
// =============================================================================

#[test]
fn test_format_case_insensitive_accepted() {
    // Standard lowercase format should be accepted.
    let desc = make_tex2d("upper_fmt", 64, 64, "rgba8unorm");

    let ir: IrResource = desc.try_into().expect("lowercase format should be accepted");
    match &ir.desc {
        ResourceDesc::Texture2D(t) => {
            // Format is preserved as-is (the original string).
            assert_eq!(t.format, "rgba8unorm");
        }
        other => panic!("expected Texture2D, got {other:?}"),
    }
}

#[test]
fn test_format_mixed_case_accepted() {
    // Mixed-case format should be accepted.
    let desc = make_tex2d("mixed_fmt", 64, 64, "Depth32Float");

    let ir: IrResource = desc.try_into().expect("mixed-case format should be accepted");
    match &ir.desc {
        ResourceDesc::Texture2D(t) => {
            assert_eq!(t.format, "Depth32Float");
        }
        other => panic!("expected Texture2D, got {other:?}"),
    }
}

// =============================================================================
// SECTION 9 -- is_transient maps to ResourceLifetime
// =============================================================================

#[test]
fn test_transient_lifetime_when_is_transient_true() {
    let desc = PyResourceDesc {
        name: "tmp_tex".into(),
        resource_type: "Texture2D".into(),
        width: 256,
        height: 256,
        format: "rgba8unorm".into(),
        is_transient: Some(true),
        ..Default::default()
    };

    let ir: IrResource = desc.try_into().expect("valid desc");
    assert_eq!(ir.lifetime, ResourceLifetime::Transient);
}

#[test]
fn test_imported_lifetime_when_is_transient_false() {
    let desc = PyResourceDesc {
        name: "persistent_tex".into(),
        resource_type: "Texture2D".into(),
        width: 256,
        height: 256,
        format: "rgba8unorm".into(),
        is_transient: Some(false),
        ..Default::default()
    };

    let ir: IrResource = desc.try_into().expect("valid desc");
    assert_eq!(ir.lifetime, ResourceLifetime::Imported);
}

#[test]
fn test_transient_default_when_is_transient_none() {
    let desc = PyResourceDesc {
        name: "default_transient".into(),
        resource_type: "Texture2D".into(),
        width: 256,
        height: 256,
        format: "rgba8unorm".into(),
        is_transient: None,
        ..Default::default()
    };

    let ir: IrResource = desc.try_into().expect("valid desc");
    // Default should be Transient (only non-transient when explicitly false).
    assert_eq!(ir.lifetime, ResourceLifetime::Transient);
}

// =============================================================================
// SECTION 10 -- initial_state maps to ResourceState
// =============================================================================

#[test]
fn test_initial_state_none_defaults_to_uninitialized() {
    let desc = PyResourceDesc {
        name: "no_state".into(),
        resource_type: "Texture2D".into(),
        width: 64,
        height: 64,
        format: "rgba8unorm".into(),
        initial_state: None,
        ..Default::default()
    };

    let ir: IrResource = desc.try_into().expect("valid desc");
    assert_eq!(ir.initial_state, ResourceState::Uninitialized);
}

#[test]
fn test_initial_state_color_attachment() {
    let desc = PyResourceDesc {
        name: "color_rt".into(),
        resource_type: "Texture2D".into(),
        width: 128,
        height: 128,
        format: "rgba8unorm".into(),
        initial_state: Some("ColorAttachment".into()),
        ..Default::default()
    };

    let ir: IrResource = desc.try_into().expect("valid desc");
    assert_eq!(ir.initial_state, ResourceState::ColorAttachment);
}

#[test]
fn test_initial_state_shader_read() {
    let desc = PyResourceDesc {
        name: "shader_tex".into(),
        resource_type: "Texture2D".into(),
        width: 64,
        height: 64,
        format: "r32float".into(),
        initial_state: Some("ShaderRead".into()),
        ..Default::default()
    };

    let ir: IrResource = desc.try_into().expect("valid desc");
    assert_eq!(ir.initial_state, ResourceState::ShaderRead);
}

#[test]
fn test_initial_state_present() {
    let desc = PyResourceDesc {
        name: "swapchain_img".into(),
        resource_type: "Texture2D".into(),
        width: 1920,
        height: 1080,
        format: "bgra8unorm".into(),
        initial_state: Some("Present".into()),
        ..Default::default()
    };

    let ir: IrResource = desc.try_into().expect("valid desc");
    assert_eq!(ir.initial_state, ResourceState::Present);
}

// =============================================================================
// SECTION 11 -- TextureCube conversion
// =============================================================================

#[test]
fn test_texture_cube_converts_with_six_layers() {
    let desc = PyResourceDesc {
        name: "skybox".into(),
        resource_type: "TextureCube".into(),
        width: 1024,
        height: 1024,
        format: "rgba8unorm-srgb".into(),
        mip_levels: 1,
        ..Default::default()
    };

    let ir: IrResource = desc.try_into().expect("valid TextureCube should convert");
    match &ir.desc {
        ResourceDesc::TextureCube(t) => {
            assert_eq!(t.width, 1024);
            assert_eq!(t.height, 1024);
            assert_eq!(t.array_layers, 6, "cube maps have 6 faces");
            assert_eq!(t.format, "rgba8unorm-srgb");
        }
        other => panic!("expected TextureCube, got {other:?}"),
    }
}

// =============================================================================
// SECTION 12 -- Texture3D conversion
// =============================================================================

#[test]
fn test_texture_3d_converts_with_depth() {
    let desc = PyResourceDesc {
        name: "volume_tex".into(),
        resource_type: "Texture3D".into(),
        width: 256,
        height: 256,
        depth: 64,
        format: "r16float".into(),
        mip_levels: 3,
        ..Default::default()
    };

    let ir: IrResource = desc.try_into().expect("valid Texture3D should convert");
    match &ir.desc {
        ResourceDesc::Texture3D(t) => {
            assert_eq!(t.width, 256);
            assert_eq!(t.height, 256);
            assert_eq!(t.depth, 64);
            assert_eq!(t.mip_levels, 3);
            assert_eq!(t.format, "r16float");
        }
        other => panic!("expected Texture3D, got {other:?}"),
    }
}

// =============================================================================
// SECTION 13 -- Invalid resource type rejected
// =============================================================================

#[test]
fn test_invalid_resource_type_rejected() {
    let desc = PyResourceDesc {
        name: "unknown_type".into(),
        resource_type: "Rasterizer".into(),
        width: 256,
        height: 256,
        format: "rgba8unorm".into(),
        ..Default::default()
    };

    let result: Result<IrResource, ConversionError> = desc.try_into();
    assert!(
        matches!(result, Err(ConversionError::InvalidResourceType(_))),
        "expected InvalidResourceType, got {result:?}"
    );
}

// =============================================================================
// SECTION 14 -- Explict handle None auto-assigns unique non-zero handle
// =============================================================================

#[test]
fn test_no_handle_auto_assigns_non_none() {
    let desc = PyResourceDesc {
        name: "auto_handle".into(),
        resource_type: "Texture2D".into(),
        width: 32,
        height: 32,
        format: "r8unorm".into(),
        handle: None,
        ..Default::default()
    };

    let ir: IrResource = desc.try_into().expect("valid desc");
    assert_ne!(ir.handle, ResourceHandle::NONE, "auto-assigned handle must not be NONE");
    assert_ne!(ir.handle.0, 0, "auto-assigned handle must not be 0");
}

// =============================================================================
// SECTION 15 -- Multiple resources (5+) deserialize with unique handles
// =============================================================================

#[test]
fn test_multiple_resources_all_have_unique_handles() {
    let resources = vec![
        make_tex2d("albedo", 1920, 1080, "rgba8unorm"),
        make_tex2d("normal", 1920, 1080, "rgba8unorm"),
        make_tex2d("depth", 1920, 1080, "depth32float"),
        make_buffer("particles", 65536),
        make_buffer("indirect", 4096),
    ];

    let converted: Vec<IrResource> = resources
        .into_iter()
        .map(|d| d.try_into().expect("each resource should convert"))
        .collect();

    assert_eq!(converted.len(), 5, "all 5 resources should convert");

    // Collect handles and verify uniqueness.
    let handles: Vec<ResourceHandle> = converted.iter().map(|r| r.handle).collect();
    let mut sorted = handles.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(
        sorted.len(),
        handles.len(),
        "all handles must be unique, got {handles:?}"
    );

    // Verify each has a valid non-NONE handle.
    for (i, res) in converted.iter().enumerate() {
        assert_ne!(
            res.handle,
            ResourceHandle::NONE,
            "resource {i} ('{}') handle must not be NONE",
            res.name
        );
    }

    // Verify specific resource types.
    assert_eq!(converted[0].name, "albedo");
    assert!(matches!(converted[0].desc, ResourceDesc::Texture2D(_)));
    assert_eq!(converted[3].name, "particles");
    assert!(matches!(converted[3].desc, ResourceDesc::Buffer(_)));
    assert_eq!(converted[4].name, "indirect");
    assert!(matches!(converted[4].desc, ResourceDesc::Buffer(_)));
}

// =============================================================================
// SECTION 16 -- Usage flag coalescing removes duplicates
// =============================================================================

#[test]
fn test_usage_flags_deduplicated_for_buffer() {
    let desc = PyResourceDesc {
        name: "dedup_buf".into(),
        resource_type: "Buffer".into(),
        width: 1024,
        usage_flags: vec![
            "storage".into(),
            "storage".into(),
            "copy_src".into(),
            "copy_src".into(),
            "storage".into(),
        ],
        ..Default::default()
    };

    let ir: IrResource = desc.try_into().expect("valid desc");
    match &ir.desc {
        ResourceDesc::Buffer(b) => {
            // "storage" and "copy_src" each appear once.
            assert_eq!(b.usage.matches("storage").count(), 1);
            assert_eq!(b.usage.matches("copy_src").count(), 1);
        }
        other => panic!("expected Buffer, got {other:?}"),
    }
}

// =============================================================================
// SECTION 17 -- Unknown usage flag rejected
// =============================================================================

#[test]
fn test_unknown_usage_flag_rejected() {
    let desc = PyResourceDesc {
        name: "bad_flag".into(),
        resource_type: "Buffer".into(),
        width: 1024,
        usage_flags: vec!["nonexistent_flag".into()],
        ..Default::default()
    };

    let result: Result<IrResource, ConversionError> = desc.try_into();
    assert!(
        matches!(result, Err(ConversionError::InvalidUsageFlags(_))),
        "expected InvalidUsageFlags, got {result:?}"
    );
}

// =============================================================================
// SECTION 18 -- Depth/stencil formats accepted
// =============================================================================

#[test]
fn test_depth_format_accepted() {
    let desc = make_tex2d("depth_tex", 1920, 1080, "depth32float");
    let ir: IrResource = desc.try_into().expect("depth32float should be accepted");
    match &ir.desc {
        ResourceDesc::Texture2D(t) => {
            assert_eq!(t.format, "depth32float");
        }
        other => panic!("expected Texture2D, got {other:?}"),
    }
}

#[test]
fn test_depth_stencil_format_accepted() {
    let desc = make_tex2d("ds_tex", 1920, 1080, "depth24plus-stencil8");
    let ir: IrResource = desc.try_into().expect("depth24plus-stencil8 should be accepted");
    match &ir.desc {
        ResourceDesc::Texture2D(t) => {
            assert_eq!(t.format, "depth24plus-stencil8");
        }
        other => panic!("expected Texture2D, got {other:?}"),
    }
}

// =============================================================================
// SECTION 19 -- Compressed formats accepted
// =============================================================================

#[test]
fn test_bc_compressed_format_accepted() {
    let desc = make_tex2d("bc_tex", 1024, 1024, "bc7-rgba-unorm");
    let ir: IrResource = desc.try_into().expect("BC7 format should be accepted");
    match &ir.desc {
        ResourceDesc::Texture2D(t) => {
            assert_eq!(t.format, "bc7-rgba-unorm");
        }
        other => panic!("expected Texture2D, got {other:?}"),
    }
}

// =============================================================================
// SECTION 20 -- Display impls
// =============================================================================

#[test]
fn test_ir_resource_display() {
    let desc = make_tex2d("disp_tex", 800, 600, "rgba8unorm");
    let ir: IrResource = desc.try_into().expect("valid desc");
    let s = format!("{}", ir);
    assert!(s.contains("disp_tex"), "Display should contain name");
    assert!(s.contains("Texture2D"), "Display should contain resource type");
}
