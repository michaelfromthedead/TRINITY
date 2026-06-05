// Blackbox contract tests for T-WGPU-P2.3.3 Texture Views API
//
// CLEANROOM: No access to implementation files. Tests use only the public API
// exported by `renderer_backend::resources::{CubeFace, TrinityTextureViewDescriptor,
// validate_view_dimensions, validate_mip_range, validate_array_range, ...}`.
//
// Forbidden files (per TESTDEV_BLACKBOX prompt):
//   - crates/renderer-backend/src/resources/texture_views.rs (implementation)
//   - Any WHITEBOX test file for this task
//
// Contract sources:
//   - Task acceptance criteria (T-WGPU-P2.3.3)
//   - Public API documentation in resources/mod.rs
//
// Public API under test:
//   - CubeFace: PosX, NegX, PosY, NegY, PosZ, NegZ, layer_index()
//   - TrinityTextureViewDescriptor: with default, builder pattern
//   - validate_view_dimensions(texture_dim, view_dim) -> bool
//   - validate_mip_range(texture_mips, base_mip, count) -> bool
//   - validate_array_range(texture_layers, base_layer, count) -> bool
//   - is_depth_stencil_format(format) -> bool
//   - has_stencil_component(format) -> bool
//   - has_depth_component(format) -> bool
//   - native_view_dimension(dim, layers, sample_count) -> TextureViewDimension
//   - TrinityTexture methods: create_trinity_view, create_mip_view, create_layer_view,
//     create_cube_face_view
//
// Test design rationale:
//   Equivalence partitioning:
//     - CubeFace variants (6 faces: +X, -X, +Y, -Y, +Z, -Z)
//     - Mip range validation (valid/invalid combinations)
//     - Array range validation (valid/invalid combinations)
//     - Depth/stencil format detection
//   Edge cases:
//     - Zero mip levels
//     - Maximum array layers
//     - Cube face index boundaries (0-5)

use renderer_backend::resources::{
    has_depth_component, has_stencil_component, is_depth_stencil_format, native_view_dimension,
    validate_array_range, validate_mip_range, validate_view_dimensions, CubeFace,
    TrinityTextureViewDescriptor,
};
use wgpu::{TextureAspect, TextureDimension, TextureFormat, TextureViewDimension};

// =============================================================================
// SECTION 1: API CONTRACT TESTS - CubeFace Enum (no GPU required)
// =============================================================================

// -----------------------------------------------------------------------------
// CubeFace Enum Variant Tests
// -----------------------------------------------------------------------------

/// Test: CubeFace enum has PosX variant.
#[test]
fn test_cubeface_has_posx_variant() {
    let face = CubeFace::PosX;
    // Just verify it compiles and can be used
    assert!(matches!(face, CubeFace::PosX));
}

/// Test: CubeFace enum has NegX variant.
#[test]
fn test_cubeface_has_negx_variant() {
    let face = CubeFace::NegX;
    assert!(matches!(face, CubeFace::NegX));
}

/// Test: CubeFace enum has PosY variant.
#[test]
fn test_cubeface_has_posy_variant() {
    let face = CubeFace::PosY;
    assert!(matches!(face, CubeFace::PosY));
}

/// Test: CubeFace enum has NegY variant.
#[test]
fn test_cubeface_has_negy_variant() {
    let face = CubeFace::NegY;
    assert!(matches!(face, CubeFace::NegY));
}

/// Test: CubeFace enum has PosZ variant.
#[test]
fn test_cubeface_has_posz_variant() {
    let face = CubeFace::PosZ;
    assert!(matches!(face, CubeFace::PosZ));
}

/// Test: CubeFace enum has NegZ variant.
#[test]
fn test_cubeface_has_negz_variant() {
    let face = CubeFace::NegZ;
    assert!(matches!(face, CubeFace::NegZ));
}

/// Test: CubeFace has exactly 6 variants (by pattern exhaustiveness).
#[test]
fn test_cubeface_has_exactly_six_variants() {
    // This test ensures pattern matching is exhaustive for exactly 6 variants
    fn face_to_string(face: CubeFace) -> &'static str {
        match face {
            CubeFace::PosX => "+X",
            CubeFace::NegX => "-X",
            CubeFace::PosY => "+Y",
            CubeFace::NegY => "-Y",
            CubeFace::PosZ => "+Z",
            CubeFace::NegZ => "-Z",
        }
    }

    assert_eq!(face_to_string(CubeFace::PosX), "+X");
    assert_eq!(face_to_string(CubeFace::NegX), "-X");
    assert_eq!(face_to_string(CubeFace::PosY), "+Y");
    assert_eq!(face_to_string(CubeFace::NegY), "-Y");
    assert_eq!(face_to_string(CubeFace::PosZ), "+Z");
    assert_eq!(face_to_string(CubeFace::NegZ), "-Z");
}

// -----------------------------------------------------------------------------
// CubeFace::layer_index() Tests
// -----------------------------------------------------------------------------

/// Test: CubeFace::PosX maps to layer index 0.
#[test]
fn test_cubeface_posx_layer_index_is_zero() {
    assert_eq!(CubeFace::PosX.layer_index(), 0);
}

/// Test: CubeFace::NegX maps to layer index 1.
#[test]
fn test_cubeface_negx_layer_index_is_one() {
    assert_eq!(CubeFace::NegX.layer_index(), 1);
}

/// Test: CubeFace::PosY maps to layer index 2.
#[test]
fn test_cubeface_posy_layer_index_is_two() {
    assert_eq!(CubeFace::PosY.layer_index(), 2);
}

/// Test: CubeFace::NegY maps to layer index 3.
#[test]
fn test_cubeface_negy_layer_index_is_three() {
    assert_eq!(CubeFace::NegY.layer_index(), 3);
}

/// Test: CubeFace::PosZ maps to layer index 4.
#[test]
fn test_cubeface_posz_layer_index_is_four() {
    assert_eq!(CubeFace::PosZ.layer_index(), 4);
}

/// Test: CubeFace::NegZ maps to layer index 5.
#[test]
fn test_cubeface_negz_layer_index_is_five() {
    assert_eq!(CubeFace::NegZ.layer_index(), 5);
}

/// Test: All CubeFace layer indices are unique and cover 0-5.
#[test]
fn test_cubeface_layer_indices_are_unique_and_contiguous() {
    let mut indices = vec![
        CubeFace::PosX.layer_index(),
        CubeFace::NegX.layer_index(),
        CubeFace::PosY.layer_index(),
        CubeFace::NegY.layer_index(),
        CubeFace::PosZ.layer_index(),
        CubeFace::NegZ.layer_index(),
    ];
    indices.sort();
    assert_eq!(indices, vec![0, 1, 2, 3, 4, 5]);
}

// =============================================================================
// SECTION 2: TrinityTextureViewDescriptor Tests
// =============================================================================

/// Test: TrinityTextureViewDescriptor has a Default implementation.
#[test]
fn test_descriptor_has_default() {
    let desc = TrinityTextureViewDescriptor::default();
    // Default should provide sensible values
    assert!(desc.label.is_none());
    assert!(desc.format.is_none());
    assert!(desc.dimension.is_none());
    assert_eq!(desc.base_mip_level, 0);
    assert_eq!(desc.base_array_layer, 0);
}

/// Test: Descriptor can be constructed with custom label.
#[test]
fn test_descriptor_custom_label() {
    let desc = TrinityTextureViewDescriptor {
        label: Some("test_view"),
        ..Default::default()
    };
    assert_eq!(desc.label, Some("test_view"));
}

/// Test: Descriptor can be constructed with custom format.
#[test]
fn test_descriptor_custom_format() {
    let desc = TrinityTextureViewDescriptor {
        format: Some(TextureFormat::Rgba8Unorm),
        ..Default::default()
    };
    assert_eq!(desc.format, Some(TextureFormat::Rgba8Unorm));
}

/// Test: Descriptor can be constructed with custom dimension.
#[test]
fn test_descriptor_custom_dimension() {
    let desc = TrinityTextureViewDescriptor {
        dimension: Some(TextureViewDimension::D2),
        ..Default::default()
    };
    assert_eq!(desc.dimension, Some(TextureViewDimension::D2));
}

/// Test: Descriptor aspect field defaults to All.
#[test]
fn test_descriptor_aspect_default() {
    let desc = TrinityTextureViewDescriptor::default();
    assert_eq!(desc.aspect, TextureAspect::All);
}

/// Test: Descriptor can specify depth aspect.
#[test]
fn test_descriptor_depth_aspect() {
    let desc = TrinityTextureViewDescriptor {
        aspect: TextureAspect::DepthOnly,
        ..Default::default()
    };
    assert_eq!(desc.aspect, TextureAspect::DepthOnly);
}

/// Test: Descriptor can specify stencil aspect.
#[test]
fn test_descriptor_stencil_aspect() {
    let desc = TrinityTextureViewDescriptor {
        aspect: TextureAspect::StencilOnly,
        ..Default::default()
    };
    assert_eq!(desc.aspect, TextureAspect::StencilOnly);
}

/// Test: Descriptor mip_level_count can be None (all remaining mips).
#[test]
fn test_descriptor_mip_count_none() {
    let desc = TrinityTextureViewDescriptor::default();
    assert!(desc.mip_level_count.is_none());
}

/// Test: Descriptor mip_level_count can be Some.
#[test]
fn test_descriptor_mip_count_some() {
    let desc = TrinityTextureViewDescriptor {
        mip_level_count: Some(4),
        ..Default::default()
    };
    assert_eq!(desc.mip_level_count, Some(4));
}

/// Test: Descriptor array_layer_count can be None.
#[test]
fn test_descriptor_array_count_none() {
    let desc = TrinityTextureViewDescriptor::default();
    assert!(desc.array_layer_count.is_none());
}

/// Test: Descriptor array_layer_count can be Some.
#[test]
fn test_descriptor_array_count_some() {
    let desc = TrinityTextureViewDescriptor {
        array_layer_count: Some(6),
        ..Default::default()
    };
    assert_eq!(desc.array_layer_count, Some(6));
}

// =============================================================================
// SECTION 3: Mip Range Validation Tests
// =============================================================================

/// Test: Valid mip range at base level 0 with all mips.
#[test]
fn test_validate_mip_range_valid_base_zero_all() {
    // 4 mip levels, starting at 0, taking all (None)
    assert!(validate_mip_range(4, 0, None));
}

/// Test: Valid mip range with explicit count.
#[test]
fn test_validate_mip_range_valid_explicit_count() {
    // 4 mip levels, starting at 0, taking 2
    assert!(validate_mip_range(4, 0, Some(2)));
}

/// Test: Valid mip range at non-zero base.
#[test]
fn test_validate_mip_range_valid_nonzero_base() {
    // 4 mip levels, starting at 2, taking 2
    assert!(validate_mip_range(4, 2, Some(2)));
}

/// Test: Valid single mip selection.
#[test]
fn test_validate_mip_range_valid_single_mip() {
    // 4 mip levels, starting at 3, taking 1
    assert!(validate_mip_range(4, 3, Some(1)));
}

/// Test: Invalid mip range - base exceeds texture mips.
#[test]
fn test_validate_mip_range_invalid_base_exceeds() {
    // 4 mip levels, but base is 4 (out of bounds)
    assert!(!validate_mip_range(4, 4, None));
}

/// Test: Invalid mip range - base + count exceeds texture mips.
#[test]
fn test_validate_mip_range_invalid_count_exceeds() {
    // 4 mip levels, base 2, count 3 would need mips 2,3,4 but max is 3
    assert!(!validate_mip_range(4, 2, Some(3)));
}

/// Test: Invalid mip range - zero texture mips.
#[test]
fn test_validate_mip_range_zero_texture_mips() {
    // Edge case: texture has 0 mip levels
    assert!(!validate_mip_range(0, 0, None));
}

/// Test: Valid mip range at last mip level.
#[test]
fn test_validate_mip_range_last_mip() {
    // 4 mip levels (0-3), selecting just the last one
    assert!(validate_mip_range(4, 3, Some(1)));
}

/// Test: Invalid mip range - zero count.
#[test]
fn test_validate_mip_range_zero_count() {
    // Requesting 0 mip levels should be invalid
    assert!(!validate_mip_range(4, 0, Some(0)));
}

// =============================================================================
// SECTION 4: Array Range Validation Tests
// =============================================================================

/// Test: Valid array range at base layer 0 with all layers.
#[test]
fn test_validate_array_range_valid_base_zero_all() {
    // 6 layers (cubemap), starting at 0, taking all (None)
    assert!(validate_array_range(6, 0, None));
}

/// Test: Valid array range with explicit count.
#[test]
fn test_validate_array_range_valid_explicit_count() {
    // 6 layers, starting at 0, taking 3
    assert!(validate_array_range(6, 0, Some(3)));
}

/// Test: Valid array range at non-zero base.
#[test]
fn test_validate_array_range_valid_nonzero_base() {
    // 6 layers, starting at 2, taking 4
    assert!(validate_array_range(6, 2, Some(4)));
}

/// Test: Valid single layer selection.
#[test]
fn test_validate_array_range_valid_single_layer() {
    // 6 layers, selecting just layer 5
    assert!(validate_array_range(6, 5, Some(1)));
}

/// Test: Invalid array range - base exceeds texture layers.
#[test]
fn test_validate_array_range_invalid_base_exceeds() {
    // 6 layers, but base is 6 (out of bounds)
    assert!(!validate_array_range(6, 6, None));
}

/// Test: Invalid array range - base + count exceeds texture layers.
#[test]
fn test_validate_array_range_invalid_count_exceeds() {
    // 6 layers, base 4, count 3 would need layers 4,5,6 but max is 5
    assert!(!validate_array_range(6, 4, Some(3)));
}

/// Test: Invalid array range - zero texture layers.
#[test]
fn test_validate_array_range_zero_texture_layers() {
    // Edge case: texture has 0 layers (invalid texture)
    assert!(!validate_array_range(0, 0, None));
}

/// Test: Valid array range with single layer texture.
#[test]
fn test_validate_array_range_single_layer_texture() {
    // Single layer texture, base 0, count 1
    assert!(validate_array_range(1, 0, Some(1)));
}

/// Test: Invalid array range - zero count.
#[test]
fn test_validate_array_range_zero_count() {
    // Requesting 0 layers should be invalid
    assert!(!validate_array_range(6, 0, Some(0)));
}

// =============================================================================
// SECTION 5: View Dimension Validation Tests
// =============================================================================

/// Test: 1D texture supports 1D view dimension.
#[test]
fn test_validate_view_dimensions_1d_to_1d() {
    assert!(validate_view_dimensions(
        TextureDimension::D1,
        TextureViewDimension::D1,
        1
    ));
}

/// Test: 2D texture supports 2D view dimension.
#[test]
fn test_validate_view_dimensions_2d_to_2d() {
    assert!(validate_view_dimensions(
        TextureDimension::D2,
        TextureViewDimension::D2,
        1
    ));
}

/// Test: 2D array texture supports 2DArray view dimension.
#[test]
fn test_validate_view_dimensions_2d_to_2d_array() {
    assert!(validate_view_dimensions(
        TextureDimension::D2,
        TextureViewDimension::D2Array,
        4
    ));
}

/// Test: 2D texture with 6 layers supports Cube view dimension.
#[test]
fn test_validate_view_dimensions_2d_to_cube() {
    assert!(validate_view_dimensions(
        TextureDimension::D2,
        TextureViewDimension::Cube,
        6
    ));
}

/// Test: 2D texture with 12 layers supports CubeArray view dimension.
#[test]
fn test_validate_view_dimensions_2d_to_cube_array() {
    assert!(validate_view_dimensions(
        TextureDimension::D2,
        TextureViewDimension::CubeArray,
        12
    ));
}

/// Test: 3D texture supports 3D view dimension.
#[test]
fn test_validate_view_dimensions_3d_to_3d() {
    assert!(validate_view_dimensions(
        TextureDimension::D3,
        TextureViewDimension::D3,
        1
    ));
}

/// Test: Invalid - 1D texture cannot be viewed as 2D.
#[test]
fn test_validate_view_dimensions_1d_to_2d_invalid() {
    assert!(!validate_view_dimensions(
        TextureDimension::D1,
        TextureViewDimension::D2,
        1
    ));
}

/// Test: Invalid - 2D texture with non-cube layer count viewed as Cube.
#[test]
fn test_validate_view_dimensions_2d_to_cube_wrong_layers() {
    // Cube requires exactly 6 layers (or multiple of 6 for CubeArray)
    assert!(!validate_view_dimensions(
        TextureDimension::D2,
        TextureViewDimension::Cube,
        4
    ));
}

/// Test: Invalid - 3D texture cannot be viewed as 2D.
#[test]
fn test_validate_view_dimensions_3d_to_2d_invalid() {
    assert!(!validate_view_dimensions(
        TextureDimension::D3,
        TextureViewDimension::D2,
        1
    ));
}

// =============================================================================
// SECTION 6: Depth/Stencil Format Detection Tests
// =============================================================================

/// Test: Depth32Float is a depth-stencil format.
#[test]
fn test_depth32float_is_depth_stencil() {
    assert!(is_depth_stencil_format(TextureFormat::Depth32Float));
}

/// Test: Depth24Plus is a depth-stencil format.
#[test]
fn test_depth24plus_is_depth_stencil() {
    assert!(is_depth_stencil_format(TextureFormat::Depth24Plus));
}

/// Test: Depth24PlusStencil8 is a depth-stencil format.
#[test]
fn test_depth24plus_stencil8_is_depth_stencil() {
    assert!(is_depth_stencil_format(TextureFormat::Depth24PlusStencil8));
}

/// Test: Depth32FloatStencil8 is a depth-stencil format.
#[test]
fn test_depth32float_stencil8_is_depth_stencil() {
    assert!(is_depth_stencil_format(TextureFormat::Depth32FloatStencil8));
}

/// Test: Rgba8Unorm is not a depth-stencil format.
#[test]
fn test_rgba8unorm_is_not_depth_stencil() {
    assert!(!is_depth_stencil_format(TextureFormat::Rgba8Unorm));
}

/// Test: Bgra8Unorm is not a depth-stencil format.
#[test]
fn test_bgra8unorm_is_not_depth_stencil() {
    assert!(!is_depth_stencil_format(TextureFormat::Bgra8Unorm));
}

// -----------------------------------------------------------------------------
// has_depth_component Tests
// -----------------------------------------------------------------------------

/// Test: Depth32Float has depth component.
#[test]
fn test_depth32float_has_depth() {
    assert!(has_depth_component(TextureFormat::Depth32Float));
}

/// Test: Depth24PlusStencil8 has depth component.
#[test]
fn test_depth24plus_stencil8_has_depth() {
    assert!(has_depth_component(TextureFormat::Depth24PlusStencil8));
}

/// Test: Rgba8Unorm does not have depth component.
#[test]
fn test_rgba8unorm_has_no_depth() {
    assert!(!has_depth_component(TextureFormat::Rgba8Unorm));
}

// -----------------------------------------------------------------------------
// has_stencil_component Tests
// -----------------------------------------------------------------------------

/// Test: Depth24PlusStencil8 has stencil component.
#[test]
fn test_depth24plus_stencil8_has_stencil() {
    assert!(has_stencil_component(TextureFormat::Depth24PlusStencil8));
}

/// Test: Depth32FloatStencil8 has stencil component.
#[test]
fn test_depth32float_stencil8_has_stencil() {
    assert!(has_stencil_component(TextureFormat::Depth32FloatStencil8));
}

/// Test: Depth32Float does not have stencil component.
#[test]
fn test_depth32float_has_no_stencil() {
    assert!(!has_stencil_component(TextureFormat::Depth32Float));
}

/// Test: Depth24Plus does not have stencil component.
#[test]
fn test_depth24plus_has_no_stencil() {
    assert!(!has_stencil_component(TextureFormat::Depth24Plus));
}

/// Test: Rgba8Unorm does not have stencil component.
#[test]
fn test_rgba8unorm_has_no_stencil() {
    assert!(!has_stencil_component(TextureFormat::Rgba8Unorm));
}

// =============================================================================
// SECTION 7: native_view_dimension Tests
// =============================================================================

/// Test: 1D texture with 1 layer -> D1.
#[test]
fn test_native_view_dimension_1d() {
    assert_eq!(
        native_view_dimension(TextureDimension::D1, 1),
        TextureViewDimension::D1
    );
}

/// Test: 2D texture with 1 layer -> D2.
#[test]
fn test_native_view_dimension_2d_single_layer() {
    assert_eq!(
        native_view_dimension(TextureDimension::D2, 1),
        TextureViewDimension::D2
    );
}

/// Test: 2D texture with multiple layers -> D2Array.
#[test]
fn test_native_view_dimension_2d_multi_layer() {
    assert_eq!(
        native_view_dimension(TextureDimension::D2, 4),
        TextureViewDimension::D2Array
    );
}

/// Test: 2D texture with 6 layers -> Cube (special case).
#[test]
fn test_native_view_dimension_2d_cube() {
    // 6 layers typically indicates a cube map
    let dim = native_view_dimension(TextureDimension::D2, 6);
    // Could be Cube or D2Array depending on implementation
    assert!(dim == TextureViewDimension::Cube || dim == TextureViewDimension::D2Array);
}

/// Test: 3D texture -> D3.
#[test]
fn test_native_view_dimension_3d() {
    assert_eq!(
        native_view_dimension(TextureDimension::D3, 1),
        TextureViewDimension::D3
    );
}

/// Test: 2D texture with 1 layer returns D2 (regardless of multisampling).
#[test]
fn test_native_view_dimension_2d_single() {
    // Native view dimension is based on texture dimension and layer count
    assert_eq!(
        native_view_dimension(TextureDimension::D2, 1),
        TextureViewDimension::D2
    );
}

// =============================================================================
// SECTION 8: Edge Case Tests
// =============================================================================

/// Test: Maximum mip level selection.
#[test]
fn test_mip_range_max_level() {
    // Test with a high mip count (e.g., 4K texture = 13 mips for 4096x4096)
    assert!(validate_mip_range(13, 12, Some(1)));
}

/// Test: Maximum array layer selection.
#[test]
fn test_array_range_max_layer() {
    // 2048 layers is a practical maximum
    assert!(validate_array_range(2048, 2047, Some(1)));
}

/// Test: Mip range with large texture.
#[test]
fn test_mip_range_large_texture() {
    // 16K texture = 15 mips for 16384x16384
    assert!(validate_mip_range(15, 0, Some(15)));
}

/// Test: Array range spanning all cube faces.
#[test]
fn test_array_range_full_cubemap() {
    // Full cubemap selection
    assert!(validate_array_range(6, 0, Some(6)));
}

/// Test: Array range for cube array (multiple cubes).
#[test]
fn test_array_range_cube_array() {
    // 4 cubemaps = 24 layers
    assert!(validate_array_range(24, 0, Some(24)));
}

/// Test: Descriptor with all fields specified.
#[test]
fn test_descriptor_fully_specified() {
    let desc = TrinityTextureViewDescriptor {
        label: Some("full_spec_view"),
        format: Some(TextureFormat::Rgba16Float),
        dimension: Some(TextureViewDimension::D2),
        aspect: TextureAspect::All,
        base_mip_level: 2,
        mip_level_count: Some(3),
        base_array_layer: 1,
        array_layer_count: Some(2),
    };

    assert_eq!(desc.label, Some("full_spec_view"));
    assert_eq!(desc.format, Some(TextureFormat::Rgba16Float));
    assert_eq!(desc.dimension, Some(TextureViewDimension::D2));
    assert_eq!(desc.aspect, TextureAspect::All);
    assert_eq!(desc.base_mip_level, 2);
    assert_eq!(desc.mip_level_count, Some(3));
    assert_eq!(desc.base_array_layer, 1);
    assert_eq!(desc.array_layer_count, Some(2));
}

/// Test: CubeFace layer indices are within valid cube range.
#[test]
fn test_cubeface_indices_within_cube_range() {
    for face in [
        CubeFace::PosX,
        CubeFace::NegX,
        CubeFace::PosY,
        CubeFace::NegY,
        CubeFace::PosZ,
        CubeFace::NegZ,
    ] {
        let index = face.layer_index();
        assert!(index < 6, "CubeFace {:?} has index {} >= 6", face, index);
    }
}

// =============================================================================
// SECTION 9: Integration Tests (require GPU)
// =============================================================================

// Note: Integration tests requiring real GPU hardware use TrinityInstance and
// TrinityDevice from renderer_backend::device module. These tests are ignored
// by default since they require GPU access.

/// Test: Real texture view creation with GPU.
/// This test requires a GPU and is ignored by default.
#[test]

fn test_integration_create_texture_view() {
    use renderer_backend::device::TrinityInstance;
    use renderer_backend::resources::{create_texture, TrinityTextureDescriptor};
    use wgpu::TextureUsages;

    // Create Trinity wgpu instance
    let instance = TrinityInstance::new();
    let adapters = instance.enumerate_adapters();
    if adapters.is_empty() {
        eprintln!("No GPU adapters found, skipping test");
        return;
    }

    let adapter = &adapters[0];
    let (device, _queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
            .expect("Failed to create device");

    // Create texture
    let texture = create_texture(
        &device,
        &TrinityTextureDescriptor {
            label: Some("test_texture"),
            size: wgpu::Extent3d {
                width: 256,
                height: 256,
                depth_or_array_layers: 1,
            },
            mip_level_count: 4,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        },
    );

    // Create view using TrinityTexture methods
    let _view = texture.create_trinity_view(&TrinityTextureViewDescriptor::default());
}

/// Test: Create mip view for specific level.
#[test]

fn test_integration_create_mip_view() {
    use renderer_backend::device::TrinityInstance;
    use renderer_backend::resources::{create_texture, TrinityTextureDescriptor};
    use wgpu::TextureUsages;

    let instance = TrinityInstance::new();
    let adapters = instance.enumerate_adapters();
    if adapters.is_empty() {
        eprintln!("No GPU adapters found, skipping test");
        return;
    }

    let adapter = &adapters[0];
    let (device, _queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
            .expect("Failed to create device");

    let texture = create_texture(
        &device,
        &TrinityTextureDescriptor {
            label: Some("mip_test_texture"),
            size: wgpu::Extent3d {
                width: 256,
                height: 256,
                depth_or_array_layers: 1,
            },
            mip_level_count: 4,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
    );

    // Create views for each mip level
    let _mip0 = texture.create_mip_view(0);
    let _mip1 = texture.create_mip_view(1);
    let _mip2 = texture.create_mip_view(2);
    let _mip3 = texture.create_mip_view(3);
}

/// Test: Create layer view for texture array.
#[test]

fn test_integration_create_layer_view() {
    use renderer_backend::device::TrinityInstance;
    use renderer_backend::resources::{create_texture, TrinityTextureDescriptor};
    use wgpu::TextureUsages;

    let instance = TrinityInstance::new();
    let adapters = instance.enumerate_adapters();
    if adapters.is_empty() {
        eprintln!("No GPU adapters found, skipping test");
        return;
    }

    let adapter = &adapters[0];
    let (device, _queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
            .expect("Failed to create device");

    let texture = create_texture(
        &device,
        &TrinityTextureDescriptor {
            label: Some("array_test_texture"),
            size: wgpu::Extent3d {
                width: 256,
                height: 256,
                depth_or_array_layers: 4,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
    );

    // Create views for each layer
    let _layer0 = texture.create_layer_view(0);
    let _layer1 = texture.create_layer_view(1);
    let _layer2 = texture.create_layer_view(2);
    let _layer3 = texture.create_layer_view(3);
}

/// Test: Create cube face views for cubemap.
#[test]

fn test_integration_create_cube_face_views() {
    use renderer_backend::device::TrinityInstance;
    use renderer_backend::resources::{create_texture, TrinityTextureDescriptor};
    use wgpu::TextureUsages;

    let instance = TrinityInstance::new();
    let adapters = instance.enumerate_adapters();
    if adapters.is_empty() {
        eprintln!("No GPU adapters found, skipping test");
        return;
    }

    let adapter = &adapters[0];
    let (device, _queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
            .expect("Failed to create device");

    let texture = create_texture(
        &device,
        &TrinityTextureDescriptor {
            label: Some("cube_test_texture"),
            size: wgpu::Extent3d {
                width: 256,
                height: 256,
                depth_or_array_layers: 6, // 6 faces for cubemap
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
    );

    // Create views for each cube face
    let _pos_x = texture.create_cube_face_view(CubeFace::PosX);
    let _neg_x = texture.create_cube_face_view(CubeFace::NegX);
    let _pos_y = texture.create_cube_face_view(CubeFace::PosY);
    let _neg_y = texture.create_cube_face_view(CubeFace::NegY);
    let _pos_z = texture.create_cube_face_view(CubeFace::PosZ);
    let _neg_z = texture.create_cube_face_view(CubeFace::NegZ);
}

/// Test: Create depth texture view with depth-only aspect.
#[test]

fn test_integration_depth_only_view() {
    use renderer_backend::device::TrinityInstance;
    use renderer_backend::resources::{create_texture, TrinityTextureDescriptor};
    use wgpu::TextureUsages;

    let instance = TrinityInstance::new();
    let adapters = instance.enumerate_adapters();
    if adapters.is_empty() {
        eprintln!("No GPU adapters found, skipping test");
        return;
    }

    let adapter = &adapters[0];
    let (device, _queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
            .expect("Failed to create device");

    let texture = create_texture(
        &device,
        &TrinityTextureDescriptor {
            label: Some("depth_test_texture"),
            size: wgpu::Extent3d {
                width: 256,
                height: 256,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth24PlusStencil8,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
    );

    // Create depth-only view
    let _depth_view = texture.create_trinity_view(&TrinityTextureViewDescriptor {
        aspect: TextureAspect::DepthOnly,
        ..Default::default()
    });
}
