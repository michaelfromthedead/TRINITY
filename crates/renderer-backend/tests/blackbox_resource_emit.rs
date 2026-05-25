// Blackbox contract tests for T-FG-7.4 Resource emit bridge.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::frame_graph::*` -- no internal fields,
// no private methods, no implementation details.
//
// Acceptance criterion (T-FG-7.4):
//   emit_resource_bridge(resource) -> JSON with structured resource descriptors
//   emit_resource_table(compiled)  -> [JSON] sorted by handle
//
// The resource emit bridge guarantees:
//   - Output is valid JSON with expected schema keys
//   - Texture resources include width/height/depth in dimensions
//   - Buffer resources include size in dimensions (not width/height)
//   - Format strings present for textures, null for buffers
//   - Transient flag correctly reflects ResourceLifetime
//   - Resource name and handle are preserved in output
//   - emit_resource_table produces output sorted by handle index
//   - emit_resource_table populates first_use_pass/last_use_pass from lifetime analysis
//   - Imported resources receive an "imported" import_path
//   - mip_levels and sample_count are present for textures, null for buffers
//
// Coverage:
//   1.  Output is a JSON object with all expected top-level schema keys
//   2.  Texture2D dimensions: width, height, depth=1
//   3.  Texture3D dimensions: width, height, depth (actual depth value)
//   4.  TextureCube dimensions: width, height, depth=6
//   5.  Buffer dimensions: size (no width/height/depth)
//   6.  Format is a non-null string for textures
//   7.  Format is null for buffers
//   8.  Transient resource emits transient=true
//   9.  Imported resource emits transient=false
//  10.  Resource name preserved in output
//  11.  Resource handle preserved in output
//  12.  emit_resource_table sorts resources by handle
//  13.  emit_resource_table populates first_use_pass/last_use_pass
//  14.  emit_resource_table sets import_path for imported resources
//  15.  emit_resource_table leaves import_path null for transient resources
//  16.  mip_levels is a number for textures, null for buffers
//  17.  sample_count is 1 for textures, null for buffers
//  18.  initial_state is a readable string matching ResourceState display
//  19.  view_format_override is null when not set, populated when set
//  20.  resource_type discriminator is correct per variant
//  21.  Empty resource table produces empty array
//  22.  Multiple resources with mixed types in emit_resource_table
//  23.  Round-trip: IrResource -> JSON -> field values match originals

use renderer_backend::frame_graph::{
    BufferDesc, CompiledFrameGraph, DispatchSource, IrPass, IrResource,
    PassIndex, ResourceDesc, ResourceHandle, ResourceLifetime, ResourceState,
    Texture3DDesc, TextureDesc, ViewType,
    emit_resource_bridge, emit_resource_table, mock_resource_texture,
};

// =============================================================================
// Helpers
// =============================================================================

/// Creates a Texture2D IrResource with full control over fields.
fn tex2d(
    handle: u32,
    name: &str,
    width: u32,
    height: u32,
    mip_levels: u32,
    array_layers: u32,
    format: &str,
    lifetime: ResourceLifetime,
    initial_state: ResourceState,
) -> IrResource {
    IrResource::new(
        ResourceHandle(handle),
        name,
        ResourceDesc::Texture2D(TextureDesc {
            width,
            height,
            mip_levels,
            array_layers,
            format: format.into(),
        }),
        lifetime,
        initial_state,
    )
}

/// Creates a Texture3D IrResource.
fn tex3d(
    handle: u32,
    name: &str,
    width: u32,
    height: u32,
    depth: u32,
    mip_levels: u32,
    format: &str,
) -> IrResource {
    IrResource::new(
        ResourceHandle(handle),
        name,
        ResourceDesc::Texture3D(Texture3DDesc {
            width,
            height,
            depth,
            mip_levels,
            format: format.into(),
        }),
        ResourceLifetime::Transient,
        ResourceState::Uninitialized,
    )
}

/// Creates a TextureCube IrResource.
fn texcube(
    handle: u32,
    name: &str,
    width: u32,
    height: u32,
    mip_levels: u32,
    format: &str,
) -> IrResource {
    IrResource::new(
        ResourceHandle(handle),
        name,
        ResourceDesc::TextureCube(TextureDesc {
            width,
            height,
            mip_levels,
            array_layers: 6,
            format: format.into(),
        }),
        ResourceLifetime::Transient,
        ResourceState::Uninitialized,
    )
}

/// Creates a Buffer IrResource with full control.
fn buffer_res(
    handle: u32,
    name: &str,
    size: u64,
    usage: &str,
    lifetime: ResourceLifetime,
) -> IrResource {
    IrResource::new(
        ResourceHandle(handle),
        name,
        ResourceDesc::Buffer(BufferDesc {
            size,
            usage: usage.into(),
            is_indirect_arg: false,
        }),
        lifetime,
        ResourceState::Uninitialized,
    )
}

/// Builds a minimal CompiledFrameGraph from passes and resources using
/// CompiledFrameGraph::compile (runs all six phases internally).
fn compile(passes: Vec<IrPass>, resources: Vec<IrResource>) -> CompiledFrameGraph {
    CompiledFrameGraph::compile(passes, resources)
        .expect("Compilation must succeed for test helpers")
}

// =============================================================================
// SECTION 1 -- Output is a JSON object with all expected schema keys
// =============================================================================

/// THE ACCEPTANCE TEST: emit_resource_bridge produces a JSON object with the
/// documented schema keys. Every resource type must emit these top-level keys.
#[test]
fn emit_resource_bridge_has_all_expected_schema_keys() {
    let resource = tex2d(
        1, "gbuffer_albedo", 1920, 1080, 10, 1, "rgba8unorm",
        ResourceLifetime::Transient, ResourceState::Uninitialized,
    );
    let json = emit_resource_bridge(&resource);

    // Must be a JSON object.
    assert!(json.is_object(), "Output must be a JSON object");

    // Verify all documented schema keys exist.
    let obj = json.as_object().unwrap();
    assert!(obj.contains_key("name"), "Must contain 'name'");
    assert!(obj.contains_key("handle"), "Must contain 'handle'");
    assert!(obj.contains_key("resource_type"), "Must contain 'resource_type'");
    assert!(obj.contains_key("dimensions"), "Must contain 'dimensions'");
    assert!(obj.contains_key("format"), "Must contain 'format'");
    assert!(obj.contains_key("mip_levels"), "Must contain 'mip_levels'");
    assert!(obj.contains_key("sample_count"), "Must contain 'sample_count'");
    assert!(obj.contains_key("transient"), "Must contain 'transient'");
    assert!(obj.contains_key("initial_state"), "Must contain 'initial_state'");
    assert!(obj.contains_key("view_format_override"), "Must contain 'view_format_override'");
    assert!(obj.contains_key("first_use_pass"), "Must contain 'first_use_pass'");
    assert!(obj.contains_key("last_use_pass"), "Must contain 'last_use_pass'");
    assert!(obj.contains_key("import_path"), "Must contain 'import_path'");
}

/// All four resource_type variants produce all 13 schema keys.
#[test]
fn all_resource_variants_have_full_schema() {
    let resources: Vec<IrResource> = vec![
        tex2d(1, "t2d", 100, 100, 1, 1, "rgba8unorm", ResourceLifetime::Transient, ResourceState::Uninitialized),
        tex3d(2, "t3d", 64, 64, 32, 1, "r32float"),
        texcube(3, "cube", 512, 512, 1, "rgba8unorm"),
        buffer_res(4, "buf", 4096, "storage", ResourceLifetime::Transient),
    ];

    let expected_keys: Vec<&str> = vec![
        "name", "handle", "resource_type", "dimensions", "format",
        "mip_levels", "sample_count", "transient", "initial_state",
        "view_format_override", "first_use_pass", "last_use_pass", "import_path",
    ];

    for (i, resource) in resources.iter().enumerate() {
        let json = emit_resource_bridge(resource);
        let obj = json.as_object().unwrap_or_else(|| {
            panic!("Resource {} (handle={}): output must be an object", i, resource.handle.0)
        });
        for key in &expected_keys {
            assert!(
                obj.contains_key(*key),
                "Resource {} (handle={}): missing key '{}'",
                i, resource.handle.0, key,
            );
        }
        assert_eq!(
            obj.len(),
            expected_keys.len(),
            "Resource {} (handle={}): expected {} keys, got {}",
            i, resource.handle.0, expected_keys.len(), obj.len(),
        );
    }
}

// =============================================================================
// SECTION 2 -- Texture2D dimensions: width, height, depth=1
// =============================================================================

/// Texture2D resources must emit dimensions with width, height, and depth=1.
#[test]
fn texture2d_dimensions_include_width_height_depth_one() {
    let resource = tex2d(
        5, "color_rt", 1920, 1080, 1, 1, "rgba8unorm",
        ResourceLifetime::Transient, ResourceState::Uninitialized,
    );
    let json = emit_resource_bridge(&resource);
    let dims = &json["dimensions"];

    assert!(dims.is_object(), "dimensions must be an object");
    assert_eq!(dims["width"].as_u64(), Some(1920), "width must be 1920");
    assert_eq!(dims["height"].as_u64(), Some(1080), "height must be 1080");
    assert_eq!(dims["depth"].as_u64(), Some(1), "Texture2D depth must be 1");
}

/// Texture2D with non-square dimensions preserves exact values.
#[test]
fn texture2d_non_square_dimensions() {
    let resource = tex2d(
        7, "non_square", 800, 600, 1, 1, "bgra8unorm-srgb",
        ResourceLifetime::Transient, ResourceState::Uninitialized,
    );
    let json = emit_resource_bridge(&resource);
    let dims = &json["dimensions"];

    assert_eq!(dims["width"].as_u64(), Some(800));
    assert_eq!(dims["height"].as_u64(), Some(600));
    assert_eq!(dims["depth"].as_u64(), Some(1));
}

/// Texture2D depth field exactly equals 1 (not missing, not null).
#[test]
fn texture2d_depth_is_exactly_one() {
    let resource = mock_resource_texture(ResourceHandle(0), "default_tex", 256, 256);
    let json = emit_resource_bridge(&resource);
    let dims = &json["dimensions"];

    assert_eq!(
        dims["depth"].as_u64(),
        Some(1),
        "Texture2D depth must be exactly 1, not null or absent"
    );
}

// =============================================================================
// SECTION 3 -- Texture3D dimensions: width, height, depth (actual depth)
// =============================================================================

/// Texture3D resources must emit dimensions with width, height, and depth
/// set to the actual volume depth (not 1).
#[test]
fn texture3d_dimensions_include_width_height_actual_depth() {
    let resource = tex3d(10, "volume", 256, 256, 128, 8, "r32float");
    let json = emit_resource_bridge(&resource);
    let dims = &json["dimensions"];

    assert_eq!(dims["width"].as_u64(), Some(256), "width must be 256");
    assert_eq!(dims["height"].as_u64(), Some(256), "height must be 256");
    assert_eq!(dims["depth"].as_u64(), Some(128), "Texture3D depth must be actual depth (128)");
}

/// Texture3D with different width, height, depth values.
#[test]
fn texture3d_asymmetric_dimensions() {
    let resource = tex3d(11, "asymmetric_volume", 512, 256, 64, 4, "rgba16float");
    let json = emit_resource_bridge(&resource);
    let dims = &json["dimensions"];

    assert_eq!(dims["width"].as_u64(), Some(512));
    assert_eq!(dims["height"].as_u64(), Some(256));
    assert_eq!(dims["depth"].as_u64(), Some(64));
}

// =============================================================================
// SECTION 4 -- TextureCube dimensions: width, height, depth=6
// =============================================================================

/// TextureCube resources must emit dimensions with width, height, and depth=6
/// (representing the six cube faces).
#[test]
fn texturecube_dimensions_include_width_height_depth_six() {
    let resource = texcube(15, "skybox", 2048, 2048, 1, "rgba8unorm");
    let json = emit_resource_bridge(&resource);
    let dims = &json["dimensions"];

    assert_eq!(dims["width"].as_u64(), Some(2048), "width must be 2048");
    assert_eq!(dims["height"].as_u64(), Some(2048), "height must be 2048");
    assert_eq!(dims["depth"].as_u64(), Some(6), "TextureCube depth must be 6 (six faces)");
}

/// TextureCube depth is exactly 6 regardless of dimensions.
#[test]
fn texturecube_depth_always_six() {
    let resource = texcube(16, "irr_map", 128, 128, 1, "rgba16float");
    let json = emit_resource_bridge(&resource);
    assert_eq!(
        json["dimensions"]["depth"].as_u64(),
        Some(6),
        "TextureCube depth must always be 6"
    );
}

// =============================================================================
// SECTION 5 -- Buffer dimensions: size (no width/height/depth)
// =============================================================================

/// Buffer resources must emit dimensions with `size` and must NOT contain
/// width, height, or depth fields.
#[test]
fn buffer_dimensions_include_size_not_width_height_or_depth() {
    let resource = buffer_res(20, "particle_buf", 65536, "storage", ResourceLifetime::Transient);
    let json = emit_resource_bridge(&resource);
    let dims = &json["dimensions"];

    assert!(dims.is_object(), "Buffer dimensions must be an object");
    assert_eq!(dims["size"].as_u64(), Some(65536), "size must be 65536");

    // Buffer dimensions must NOT have texture-specific fields.
    assert!(
        dims.get("width").is_none(),
        "Buffer must NOT have 'width' in dimensions"
    );
    assert!(
        dims.get("height").is_none(),
        "Buffer must NOT have 'height' in dimensions"
    );
    assert!(
        dims.get("depth").is_none(),
        "Buffer must NOT have 'depth' in dimensions"
    );
}

/// Buffer with different sizes produces correct size field.
#[test]
fn buffer_size_field_matches_various_sizes() {
    let sizes: Vec<(u32, u64)> = vec![
        (25, 0),
        (26, 1),
        (27, 256),
        (28, 1024 * 1024),
        (29, 0xFFFF_FFFF),
    ];

    for (handle, size) in &sizes {
        let resource = buffer_res(*handle, "buf", *size, "uniform", ResourceLifetime::Transient);
        let json = emit_resource_bridge(&resource);
        assert_eq!(
            json["dimensions"]["size"].as_u64(),
            Some(*size),
            "Buffer handle {}: size must be {}",
            handle, size,
        );
    }
}

/// Buffer with is_indirect_arg retains the correct size in dimensions.
#[test]
fn buffer_indirect_arg_still_has_size() {
    let resource = IrResource::new(
        ResourceHandle(30),
        "indirect_buf",
        ResourceDesc::Buffer(BufferDesc {
            size: 4096,
            usage: "indirect".into(),
            is_indirect_arg: true,
        }),
        ResourceLifetime::Transient,
        ResourceState::IndirectArgument,
    );
    let json = emit_resource_bridge(&resource);
    assert_eq!(
        json["dimensions"]["size"].as_u64(),
        Some(4096),
        "Indirect buffer must still have size in dimensions"
    );
}

// =============================================================================
// SECTION 6 -- Format strings present for textures, null for buffers
// =============================================================================

/// Texture resources emit `format` as a non-null string matching the texture
/// descriptor format.
#[test]
fn texture_format_is_non_null_string() {
    let resource = tex2d(
        35, "hdr_target", 1920, 1080, 1, 1, "rgba16float",
        ResourceLifetime::Transient, ResourceState::ColorAttachment,
    );
    let json = emit_resource_bridge(&resource);

    let fmt = json["format"].as_str();
    assert!(fmt.is_some(), "Texture format must be a string, not null");
    assert_eq!(fmt.unwrap(), "rgba16float");
}

/// All three texture variants emit their specific format string.
#[test]
fn texture_variants_all_have_format_string() {
    let resources: Vec<(IrResource, &str)> = vec![
        (
            tex2d(40, "a", 100, 100, 1, 1, "rgba8unorm-srgb", ResourceLifetime::Transient, ResourceState::ShaderRead),
            "rgba8unorm-srgb",
        ),
        (
            tex3d(41, "b", 64, 64, 32, 1, "r32float"),
            "r32float",
        ),
        (
            texcube(42, "c", 512, 512, 1, "bgra8unorm"),
            "bgra8unorm",
        ),
    ];

    for (resource, expected_format) in &resources {
        let json = emit_resource_bridge(resource);
        assert_eq!(
            json["format"].as_str(),
            Some(*expected_format),
            "Resource {}: expected format '{}'",
            resource.handle.0, expected_format,
        );
    }
}

/// Buffer resources emit `format` as JSON null.
#[test]
fn buffer_format_is_null() {
    let resource = buffer_res(45, "ssbo", 8192, "storage", ResourceLifetime::Transient);
    let json = emit_resource_bridge(&resource);
    assert!(
        json["format"].is_null(),
        "Buffer format must be null, got {:?}",
        json["format"],
    );
}

// =============================================================================
// SECTION 7 -- Transient flag correctly set
// =============================================================================

/// A transient resource emits `transient: true`.
#[test]
fn transient_resource_emits_transient_true() {
    let resource = tex2d(
        50, "temp_depth", 1024, 1024, 1, 1, "depth32float",
        ResourceLifetime::Transient, ResourceState::Uninitialized,
    );
    let json = emit_resource_bridge(&resource);
    assert_eq!(
        json["transient"].as_bool(),
        Some(true),
        "Transient resource must emit transient=true"
    );
}

/// An imported resource emits `transient: false`.
#[test]
fn imported_resource_emits_transient_false() {
    let resource = tex2d(
        51, "swapchain_image", 1920, 1080, 1, 1, "rgba8unorm",
        ResourceLifetime::Imported, ResourceState::Present,
    );
    let json = emit_resource_bridge(&resource);
    assert_eq!(
        json["transient"].as_bool(),
        Some(false),
        "Imported resource must emit transient=false"
    );
}

/// Transient flag for buffer resources follows the same rule.
#[test]
fn buffer_transient_flag() {
    let transient_buf = buffer_res(52, "scratch", 4096, "storage", ResourceLifetime::Transient);
    assert_eq!(
        emit_resource_bridge(&transient_buf)["transient"].as_bool(),
        Some(true),
        "Transient buffer must emit transient=true"
    );

    let imported_buf = buffer_res(53, "persistent", 4096, "uniform", ResourceLifetime::Imported);
    assert_eq!(
        emit_resource_bridge(&imported_buf)["transient"].as_bool(),
        Some(false),
        "Imported buffer must emit transient=false"
    );
}

// =============================================================================
// SECTION 8 -- Resource name and handle present in output
// =============================================================================

/// Resource name is preserved verbatim in the JSON output.
#[test]
fn resource_name_preserved_in_output() {
    let resource = tex2d(
        60, "custom_name_with_underscores", 800, 600, 1, 1, "r8unorm",
        ResourceLifetime::Transient, ResourceState::ShaderRead,
    );
    let json = emit_resource_bridge(&resource);
    assert_eq!(
        json["name"].as_str(),
        Some("custom_name_with_underscores"),
        "Resource name must be preserved verbatim"
    );
}

/// Resource handle (u32) is preserved in the output as a number.
#[test]
fn resource_handle_preserved_in_output() {
    let handles: Vec<u32> = vec![0, 1, 100, 255, 65535, 0x7FFFFFFF, 0xFFFFFFFF];

    for &handle in &handles {
        let resource = buffer_res(handle, "buf", 64, "storage", ResourceLifetime::Transient);
        let json = emit_resource_bridge(&resource);
        assert_eq!(
            json["handle"].as_u64(),
            Some(handle as u64),
            "Handle {} must be preserved in output as u64-compatible number",
            handle,
        );
    }
}

// =============================================================================
// SECTION 9 -- emit_resource_table produces sorted output
// =============================================================================

/// emit_resource_table must return resources sorted by handle index,
/// regardless of input order.
#[test]
fn emit_resource_table_sorted_by_handle() {
    // Resources in arbitrary order (non-sequential handles, out of order).
    let resources = vec![
        buffer_res(10, "buf_c", 1024, "storage", ResourceLifetime::Transient),
        tex2d(3, "tex_a", 100, 100, 1, 1, "rgba8unorm", ResourceLifetime::Transient, ResourceState::Uninitialized),
        buffer_res(7, "buf_b", 4096, "uniform", ResourceLifetime::Transient),
        tex2d(1, "tex_a", 1920, 1080, 1, 1, "rgba8unorm", ResourceLifetime::Transient, ResourceState::Uninitialized),
    ];

    // One pass that reads all resources so compilation completes.
    let pass = mock_compute_pass_reading_all(&resources);
    let compiled = compile(vec![pass], resources);

    let table = emit_resource_table(&compiled);
    assert_eq!(table.len(), 4, "Must have 4 entries");

    // Verify handles are in ascending order: 1, 3, 7, 10.
    let handle_values: Vec<u64> = table
        .iter()
        .map(|entry| entry["handle"].as_u64().unwrap())
        .collect();
    assert_eq!(handle_values, vec![1, 3, 7, 10], "Must be sorted by handle ascending");

    // Verify the names are still correct per handle.
    assert_eq!(table[0]["name"].as_str(), Some("tex_a"), "Handle 1 name");
    assert_eq!(table[1]["name"].as_str(), Some("tex_a"), "Handle 3 name");
    assert_eq!(table[2]["name"].as_str(), Some("buf_b"), "Handle 7 name");
    assert_eq!(table[3]["name"].as_str(), Some("buf_c"), "Handle 10 name");
}

/// Helper: creates a compute pass that reads all given resources (required so
/// that CompiledFrameGraph::compile does not abort on isolated resources).
fn mock_compute_pass_reading_all(resources: &[IrResource]) -> IrPass {
    let reads: Vec<ResourceHandle> = resources.iter().map(|r| r.handle).collect();
    let mut pass = IrPass::compute(
        PassIndex(0),
        "pass_reader",
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::Storage,
    );
    for &h in &reads {
        pass.access_set.reads.push(h);
    }
    pass
}

// =============================================================================
// SECTION 10 -- emit_resource_table populates lifetime information
// =============================================================================

/// When resources are referenced by passes, emit_resource_table populates
/// first_use_pass and last_use_pass with the actual pass indices.
#[test]
fn emit_resource_table_populates_lifetime_from_passes() {
    // Two resources, two passes: pass 0 writes both, pass 1 reads both.
    let r1 = ResourceHandle(1);
    let r2 = ResourceHandle(2);

    let resources = vec![
        buffer_res(1, "buf_1", 1024, "storage", ResourceLifetime::Transient),
        buffer_res(2, "buf_2", 2048, "storage", ResourceLifetime::Transient),
    ];

    let pass_write = {
        let mut p = IrPass::compute(
            PassIndex(0),
            "writer",
            DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 },
            ViewType::Storage,
        );
        p.access_set.writes.push(r1);
        p.access_set.writes.push(r2);
        p
    };

    let pass_read = {
        let mut p = IrPass::compute(
            PassIndex(1),
            "reader",
            DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 },
            ViewType::Storage,
        );
        p.access_set.reads.push(r1);
        p.access_set.reads.push(r2);
        p
    };

    let compiled = compile(vec![pass_write, pass_read], resources);
    let table = emit_resource_table(&compiled);
    assert_eq!(table.len(), 2);

    // Both resources are first used in pass 0 and last used in pass 1.
    for entry in &table {
        assert_eq!(
            entry["first_use_pass"].as_u64(),
            Some(0),
            "Resource {}: first_use_pass should be 0",
            entry["handle"].as_u64().unwrap(),
        );
        assert_eq!(
            entry["last_use_pass"].as_u64(),
            Some(1),
            "Resource {}: last_use_pass should be 1",
            entry["handle"].as_u64().unwrap(),
        );
    }
}

/// A resource used only in a single pass has first_use_pass == last_use_pass.
#[test]
fn single_pass_use_has_equal_first_and_last() {
    let r1 = ResourceHandle(1);
    let resources = vec![
        mock_resource_texture(r1, "single_use", 800, 600),
    ];

    let pass = {
        let mut p = IrPass::compute(
            PassIndex(0),
            "sole_user",
            DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 },
            ViewType::Storage,
        );
        p.access_set.writes.push(r1);
        p
    };

    let compiled = compile(vec![pass], resources);
    let table = emit_resource_table(&compiled);
    assert_eq!(table.len(), 1);

    assert_eq!(table[0]["first_use_pass"].as_u64(), Some(0));
    assert_eq!(table[0]["last_use_pass"].as_u64(), Some(0));
}

// =============================================================================
// SECTION 11 -- emit_resource_table sets import_path for imported resources
// =============================================================================

/// Imported resources receive import_path = "imported".
#[test]
fn imported_resource_gets_import_path() {
    let resource = tex2d(
        80, "external_tex", 1920, 1080, 1, 1, "rgba8unorm",
        ResourceLifetime::Imported, ResourceState::Present,
    );
    let compiled = compile(
        vec![mock_compute_pass_reading_all(&[resource.clone()])],
        vec![resource],
    );
    let table = emit_resource_table(&compiled);
    assert_eq!(table.len(), 1);

    assert_eq!(
        table[0]["import_path"].as_str(),
        Some("imported"),
        "Imported resource must have import_path='imported'"
    );
}

/// Transient resources have import_path = null.
#[test]
fn transient_resource_import_path_is_null() {
    let resource = tex2d(
        81, "internal_rt", 800, 600, 1, 1, "rgba8unorm",
        ResourceLifetime::Transient, ResourceState::Uninitialized,
    );
    let compiled = compile(
        vec![mock_compute_pass_reading_all(&[resource.clone()])],
        vec![resource],
    );
    let table = emit_resource_table(&compiled);
    assert_eq!(table.len(), 1);

    assert!(
        table[0]["import_path"].is_null(),
        "Transient resource must have import_path=null"
    );
}

/// emit_resource_table also sets import_path for imported buffers.
#[test]
fn imported_buffer_gets_import_path() {
    let resource = buffer_res(82, "persistent_ssbo", 16384, "storage", ResourceLifetime::Imported);
    let compiled = compile(
        vec![mock_compute_pass_reading_all(&[resource.clone()])],
        vec![resource],
    );
    let table = emit_resource_table(&compiled);
    assert_eq!(table[0]["import_path"].as_str(), Some("imported"));
}

// =============================================================================
// SECTION 12 -- mip_levels and sample_count
// =============================================================================

/// Texture resources emit mip_levels as a non-null number matching the
/// descriptor, and sample_count as 1.
#[test]
fn texture_mip_levels_and_sample_count() {
    let resource = tex2d(
        90, "mip_chain", 1024, 1024, 12, 1, "rgba8unorm",
        ResourceLifetime::Transient, ResourceState::ShaderRead,
    );
    let json = emit_resource_bridge(&resource);

    assert_eq!(
        json["mip_levels"].as_u64(),
        Some(12),
        "mip_levels must be 12"
    );
    assert_eq!(
        json["sample_count"].as_u64(),
        Some(1),
        "sample_count must be 1 (MSAA not modelled at IR level)"
    );
}

/// Texture3D and TextureCube also emit mip_levels and sample_count.
#[test]
fn texture3d_and_cube_mip_levels_and_sample_count() {
    let t3d = tex3d(91, "vol", 64, 64, 32, 8, "r32float");
    let j3d = emit_resource_bridge(&t3d);
    assert_eq!(j3d["mip_levels"].as_u64(), Some(8), "Texture3D mip_levels");
    assert_eq!(j3d["sample_count"].as_u64(), Some(1), "Texture3D sample_count");

    let tcube = texcube(92, "env", 512, 512, 4, "rgba16float");
    let jcube = emit_resource_bridge(&tcube);
    assert_eq!(jcube["mip_levels"].as_u64(), Some(4), "TextureCube mip_levels");
    assert_eq!(jcube["sample_count"].as_u64(), Some(1), "TextureCube sample_count");
}

/// Buffer resources emit mip_levels and sample_count as null.
#[test]
fn buffer_mip_levels_and_sample_count_are_null() {
    let resource = buffer_res(95, "storage_buf", 8192, "storage", ResourceLifetime::Transient);
    let json = emit_resource_bridge(&resource);

    assert!(
        json["mip_levels"].is_null(),
        "Buffer mip_levels must be null"
    );
    assert!(
        json["sample_count"].is_null(),
        "Buffer sample_count must be null"
    );
}

// =============================================================================
// SECTION 13 -- initial_state is a readable string
// =============================================================================

/// initial_state must be a string matching ResourceState Display output.
#[test]
fn initial_state_is_readable_string_for_all_states() {
    use ResourceState::*;
    let states: Vec<(ResourceState, &str)> = vec![
        (Uninitialized, "Uninitialized"),
        (VertexBuffer, "VertexBuffer"),
        (IndexBuffer, "IndexBuffer"),
        (IndirectArgument, "IndirectArgument"),
        (ColorAttachment, "ColorAttachment"),
        (DepthStencilAttachment, "DepthStencilAttachment"),
        (DepthStencilReadOnly, "DepthStencilReadOnly"),
        (ShaderRead, "ShaderRead"),
        (ShaderReadWrite, "ShaderReadWrite"),
        (TransferSrc, "TransferSrc"),
        (TransferDst, "TransferDst"),
        (AccelerationStructure, "AccelerationStructure"),
        (Present, "Present"),
    ];

    for (state, expected) in &states {
        let resource = tex2d(
            100, "state_test", 1, 1, 1, 1, "rgba8unorm",
            ResourceLifetime::Transient, *state,
        );
        let json = emit_resource_bridge(&resource);
        assert_eq!(
            json["initial_state"].as_str(),
            Some(*expected),
            "initial_state for {:?} must be '{}'",
            state, expected,
        );
    }
}

// =============================================================================
// SECTION 14 -- view_format_override
// =============================================================================

/// view_format_override is null when the resource has no override set.
#[test]
fn view_format_override_is_null_by_default() {
    let resource = tex2d(
        110, "no_override", 800, 600, 1, 1, "rgba8unorm",
        ResourceLifetime::Transient, ResourceState::ShaderRead,
    );
    let json = emit_resource_bridge(&resource);
    assert!(
        json["view_format_override"].is_null(),
        "view_format_override must be null when not set"
    );
}

/// view_format_override is populated when set on the resource.
#[test]
fn view_format_override_populated_when_set() {
    let mut resource = tex2d(
        111, "with_override", 800, 600, 1, 1, "rgba8unorm",
        ResourceLifetime::Transient, ResourceState::ShaderRead,
    );
    resource.view_format_override = Some("bgra8unorm".into());

    let json = emit_resource_bridge(&resource);
    assert_eq!(
        json["view_format_override"].as_str(),
        Some("bgra8unorm"),
        "view_format_override must be 'bgra8unorm'"
    );
}

// =============================================================================
// SECTION 15 -- resource_type discriminator
// =============================================================================

/// resource_type must be the correct string discriminator per variant.
#[test]
fn resource_type_discriminator_for_all_variants() {
    let cases: Vec<(IrResource, &str)> = vec![
        (
            tex2d(200, "t2d", 1, 1, 1, 1, "r8unorm", ResourceLifetime::Transient, ResourceState::Uninitialized),
            "texture2d",
        ),
        (
            tex3d(201, "t3d", 1, 1, 1, 1, "r32float"),
            "texture3d",
        ),
        (
            texcube(202, "cube", 1, 1, 1, "rgba8unorm"),
            "texturecube",
        ),
        (
            buffer_res(203, "buf", 64, "storage", ResourceLifetime::Transient),
            "buffer",
        ),
    ];

    for (resource, expected_type) in &cases {
        let json = emit_resource_bridge(resource);
        assert_eq!(
            json["resource_type"].as_str(),
            Some(*expected_type),
            "Resource {} (handle={}): expected resource_type '{}'",
            resource.name, resource.handle.0, expected_type,
        );
    }
}

// =============================================================================
// SECTION 16 -- Empty resource table
// =============================================================================

/// An empty compiled frame graph (no passes, no resources) produces an empty
/// array from emit_resource_table.
#[test]
fn empty_resource_table_returns_empty_array() {
    let compiled = compile(vec![], vec![]);
    let table = emit_resource_table(&compiled);
    assert!(
        table.is_empty(),
        "Empty frame graph must produce empty array, got {} entries",
        table.len(),
    );
}

// =============================================================================
// SECTION 17 -- Multiple resources with mixed types in emit_resource_table
// =============================================================================

/// emit_resource_table correctly handles a mix of texture and buffer resources.
#[test]
fn mixed_texture_and_buffer_table() {
    let resources: Vec<IrResource> = vec![
        tex2d(5, "color_out", 1920, 1080, 1, 1, "rgba8unorm", ResourceLifetime::Transient, ResourceState::Uninitialized),
        buffer_res(3, "particles", 65536, "storage", ResourceLifetime::Transient),
        texcube(7, "env_map", 1024, 1024, 1, "rgba16float"),
        buffer_res(1, "constants", 256, "uniform", ResourceLifetime::Imported),
    ];

    let pass = mock_compute_pass_reading_all(&resources);
    let compiled = compile(vec![pass], resources);
    let table = emit_resource_table(&compiled);

    assert_eq!(table.len(), 4);

    // Check each type is correct at its sorted position.
    // Sorted handles: 1 (constants, buffer), 3 (particles, buffer),
    //                 5 (color_out, texture2d), 7 (env_map, texturecube)
    assert_eq!(table[0]["resource_type"].as_str(), Some("buffer"),  "entry 0 = buffer");
    assert_eq!(table[0]["handle"].as_u64(), Some(1), "entry 0 = handle 1");
    assert_eq!(table[0]["name"].as_str(), Some("constants"));

    assert_eq!(table[1]["resource_type"].as_str(), Some("buffer"),  "entry 1 = buffer");
    assert_eq!(table[1]["handle"].as_u64(), Some(3));

    assert_eq!(table[2]["resource_type"].as_str(), Some("texture2d"), "entry 2 = texture2d");
    assert_eq!(table[2]["handle"].as_u64(), Some(5));

    assert_eq!(table[3]["resource_type"].as_str(), Some("texturecube"), "entry 3 = texturecube");
    assert_eq!(table[3]["handle"].as_u64(), Some(7));
}

// =============================================================================
// SECTION 18 -- Round-trip: IrResource -> JSON fields match originals
// =============================================================================

/// Round-trip verification: constructing a resource and serialising it must
/// preserve all scalar and string fields in the JSON output.
#[test]
fn texture2d_roundtrip_preserves_all_fields() {
    let resource = tex2d(
        300, "roundtrip_tex", 1920, 1080, 10, 4, "rgba8unorm-srgb",
        ResourceLifetime::Transient, ResourceState::ColorAttachment,
    );
    let json = emit_resource_bridge(&resource);

    assert_eq!(json["name"].as_str(), Some("roundtrip_tex"));
    assert_eq!(json["handle"].as_u64(), Some(300));
    assert_eq!(json["resource_type"].as_str(), Some("texture2d"));
    assert_eq!(json["dimensions"]["width"].as_u64(), Some(1920));
    assert_eq!(json["dimensions"]["height"].as_u64(), Some(1080));
    assert_eq!(json["dimensions"]["depth"].as_u64(), Some(1));
    assert_eq!(json["format"].as_str(), Some("rgba8unorm-srgb"));
    assert_eq!(json["mip_levels"].as_u64(), Some(10));
    assert_eq!(json["sample_count"].as_u64(), Some(1));
    assert_eq!(json["transient"].as_bool(), Some(true));
    assert_eq!(json["initial_state"].as_str(), Some("ColorAttachment"));
    assert!(json["view_format_override"].is_null());
    assert!(json["first_use_pass"].is_null());
    assert!(json["last_use_pass"].is_null());
    assert!(json["import_path"].is_null());
}

/// Buffer round-trip: all buffer-specific fields are preserved.
#[test]
fn buffer_roundtrip_preserves_all_fields() {
    let resource = IrResource::new(
        ResourceHandle(301),
        "roundtrip_buf",
        ResourceDesc::Buffer(BufferDesc {
            size: 1048576,
            usage: "storage | indirect".into(),
            is_indirect_arg: true,
        }),
        ResourceLifetime::Imported,
        ResourceState::IndirectArgument,
    );
    let json = emit_resource_bridge(&resource);

    assert_eq!(json["name"].as_str(), Some("roundtrip_buf"));
    assert_eq!(json["handle"].as_u64(), Some(301));
    assert_eq!(json["resource_type"].as_str(), Some("buffer"));
    assert_eq!(json["dimensions"]["size"].as_u64(), Some(1048576));
    assert!(json["dimensions"].get("width").is_none());
    assert!(json["dimensions"].get("height").is_none());
    assert!(json["dimensions"].get("depth").is_none());
    assert!(json["format"].is_null());
    assert!(json["mip_levels"].is_null());
    assert!(json["sample_count"].is_null());
    assert_eq!(json["transient"].as_bool(), Some(false));
    assert_eq!(json["initial_state"].as_str(), Some("IndirectArgument"));
    assert!(json["view_format_override"].is_null());
    assert!(json["first_use_pass"].is_null());
    assert!(json["last_use_pass"].is_null());
}

/// Texture3D round-trip: depth field is preserved as actual depth.
#[test]
fn texture3d_roundtrip_preserves_depth() {
    let resource = tex3d(302, "volume_scan", 256, 256, 64, 5, "r16float");
    let json = emit_resource_bridge(&resource);

    assert_eq!(json["name"].as_str(), Some("volume_scan"));
    assert_eq!(json["resource_type"].as_str(), Some("texture3d"));
    assert_eq!(json["dimensions"]["depth"].as_u64(), Some(64), "Texture3D depth=64");
    assert_eq!(json["format"].as_str(), Some("r16float"));
    assert_eq!(json["mip_levels"].as_u64(), Some(5));
}

/// TextureCube round-trip: depth is 6, format is preserved.
#[test]
fn texturecube_roundtrip_preserves_depth_six() {
    let resource = texcube(303, "skybox_pan", 2048, 2048, 8, "rgba8unorm");
    let json = emit_resource_bridge(&resource);

    assert_eq!(json["name"].as_str(), Some("skybox_pan"));
    assert_eq!(json["resource_type"].as_str(), Some("texturecube"));
    assert_eq!(json["dimensions"]["depth"].as_u64(), Some(6), "TextureCube depth=6");
    assert_eq!(json["format"].as_str(), Some("rgba8unorm"));
    assert_eq!(json["mip_levels"].as_u64(), Some(8));
}

// =============================================================================
// SECTION 19 -- emit_resource_bridge handles zero values correctly
// =============================================================================

/// Zero-sized dimensions are serialised correctly (not omitted).
#[test]
fn zero_sized_dimensions_are_serialised() {
    let resource = tex2d(
        400, "zero_sized", 0, 0, 0, 0, "r8unorm",
        ResourceLifetime::Transient, ResourceState::Uninitialized,
    );
    let json = emit_resource_bridge(&resource);

    assert_eq!(json["dimensions"]["width"].as_u64(), Some(0), "zero width");
    assert_eq!(json["dimensions"]["height"].as_u64(), Some(0), "zero height");
    assert_eq!(json["dimensions"]["depth"].as_u64(), Some(1), "depth still 1 for Texture2D");
    assert_eq!(json["mip_levels"].as_u64(), Some(0), "zero mip_levels");
}

/// Buffer with size 0 is serialised correctly.
#[test]
fn zero_sized_buffer() {
    let resource = buffer_res(401, "empty_buf", 0, "storage", ResourceLifetime::Transient);
    let json = emit_resource_bridge(&resource);
    assert_eq!(json["dimensions"]["size"].as_u64(), Some(0), "zero buffer size");
}

// =============================================================================
// SECTION 20 -- Display and resource_type consistency
// =============================================================================

/// resource_type is always lowercase for all variants.
#[test]
fn resource_type_is_always_lowercase() {
    let types: Vec<IrResource> = vec![
        tex2d(500, "a", 1, 1, 1, 1, "r8unorm", ResourceLifetime::Transient, ResourceState::Uninitialized),
        tex3d(501, "b", 1, 1, 1, 1, "r32float"),
        texcube(502, "c", 1, 1, 1, "rgba8unorm"),
        buffer_res(503, "d", 64, "storage", ResourceLifetime::Transient),
    ];

    for resource in &types {
        let json = emit_resource_bridge(resource);
        let rt = json["resource_type"].as_str().unwrap();
        assert_eq!(
            rt.to_lowercase(),
            rt,
            "resource_type '{}' must be lowercase for resource {}",
            rt, resource.name,
        );
    }
}

// =============================================================================
// SECTION 21 -- Edge cases: ResourceHandle::NONE and large handles
// =============================================================================

/// ResourceHandle::NONE (u32::MAX) is a valid handle that should serialise
/// correctly without panicking.
#[test]
fn resource_handle_none_is_serialised() {
    let resource = buffer_res(
        ResourceHandle::NONE.0, "none_handle", 1024, "storage", ResourceLifetime::Transient,
    );
    let json = emit_resource_bridge(&resource);
    assert_eq!(
        json["handle"].as_u64(),
        Some(ResourceHandle::NONE.0 as u64),
        "ResourceHandle::NONE must serialise as u32::MAX"
    );
}

// =============================================================================
// SECTION 22 -- emit_resource_table handles resources not referenced by passes
// =============================================================================

/// Resources that are not referenced by any pass still appear in the table
/// (with null first_use_pass and last_use_pass from standalone emit, but
/// note: emit_resource_table calls compute_lifetimes internally, which scans
/// passes. If no pass references a resource, the resource gets no lifetime
/// entry and keeps null for first/last_use_pass).
#[test]
fn resource_not_referenced_by_any_pass() {
    let resources = vec![
        tex2d(1, "orphan_tex", 800, 600, 1, 1, "rgba8unorm", ResourceLifetime::Transient, ResourceState::Uninitialized),
    ];

    // Compile with an unrelated pass (does not reference the resource).
    let unrelated_pass = IrPass::compute(
        PassIndex(0),
        "unrelated",
        DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 },
        ViewType::Storage,
    );

    let compiled = compile(vec![unrelated_pass], resources);
    let table = emit_resource_table(&compiled);
    assert_eq!(table.len(), 1, "Unreferenced resource must still appear in table");

    // The resource appears with null lifetimes (compute_lifetimes did not
    // find it in any pass).
    assert_eq!(table[0]["handle"].as_u64(), Some(1));
    assert_eq!(table[0]["name"].as_str(), Some("orphan_tex"));
}

// =============================================================================
// SECTION 23 -- emit_resource_table preserves individual resource bridge output
// =============================================================================

/// Each entry in emit_resource_table is produced by emit_resource_bridge, so
/// the per-resource fields (name, handle, resource_type, dimensions, format,
/// transient, initial_state) must match the standalone bridge output.
#[test]
fn table_entries_match_standalone_bridge_output() {
    let resources = vec![
        tex2d(10, "albedo", 1920, 1080, 8, 1, "rgba8unorm", ResourceLifetime::Transient, ResourceState::Uninitialized),
        buffer_res(5, "vbo", 65536, "vertex", ResourceLifetime::Imported),
    ];

    let pass = mock_compute_pass_reading_all(&resources);
    let compiled = compile(vec![pass], resources.clone());
    let table = emit_resource_table(&compiled);

    // Compare each table entry against standalone emit_resource_bridge.
    for resource in &resources {
        let standalone = emit_resource_bridge(resource);
        let table_entry = table.iter()
            .find(|e| e["handle"].as_u64() == Some(resource.handle.0 as u64))
            .unwrap_or_else(|| panic!("Resource {} must appear in table", resource.handle.0));

        // Fields that should match exactly.
        assert_eq!(table_entry["name"], standalone["name"], "name for handle {}", resource.handle.0);
        assert_eq!(table_entry["handle"], standalone["handle"]);
        assert_eq!(table_entry["resource_type"], standalone["resource_type"]);
        assert_eq!(table_entry["dimensions"], standalone["dimensions"]);
        assert_eq!(table_entry["format"], standalone["format"]);
        assert_eq!(table_entry["mip_levels"], standalone["mip_levels"]);
        assert_eq!(table_entry["sample_count"], standalone["sample_count"]);
        assert_eq!(table_entry["transient"], standalone["transient"]);
        assert_eq!(table_entry["initial_state"], standalone["initial_state"]);
        assert_eq!(table_entry["view_format_override"], standalone["view_format_override"]);
    }
}

// =============================================================================
// SECTION 24 -- Multiple resources with same handle (deduplication note)
// =============================================================================

/// When multiple resources share the same handle, the later one in the
/// resources vec overwrites in the compiled output (Vec storage, not a map).
/// emit_resource_table processes all resources in the compiled vec; duplicates
/// are preserved. This test documents the current behaviour.
#[test]
fn duplicate_handles_are_all_emitted() {
    let resources = vec![
        buffer_res(1, "first_buf", 1024, "storage", ResourceLifetime::Transient),
        buffer_res(1, "second_buf", 2048, "storage", ResourceLifetime::Transient),
    ];

    let pass = mock_compute_pass_reading_all(&resources);
    let compiled = compile(vec![pass], resources);
    let table = emit_resource_table(&compiled);

    // Both resources with handle 1 are emitted (the vec is sorted by handle,
    // but both have handle 1 so they remain adjacent and order-preserved).
    assert_eq!(table.len(), 2, "Both duplicate-handle resources must be emitted");
    assert_eq!(table[0]["name"].as_str(), Some("first_buf"));
    assert_eq!(table[1]["name"].as_str(), Some("second_buf"));
    assert_eq!(table[0]["handle"].as_u64(), Some(1));
    assert_eq!(table[1]["handle"].as_u64(), Some(1));
}
