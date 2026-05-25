// Blackbox contract tests for T-FG-6.4 PassValidator.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::frame_graph::*` -- no internal fields,
// no private methods, no implementation details.
//
// Contract:
//   PassValidator::validate(pass: &IrPass, resources: &[IrResource]) -> Result<(), Vec<String>>
//   validates a single pass before graph compilation. Returns Ok(()) when all
//   checks pass, or Err containing one or more human-readable error messages.
//
// Checks performed:
//
//   A. Attachment resource validity
//      - Colour attachments must not use ResourceHandle::NONE
//      - Colour attachments must reference existing resources
//      - Depth-stencil attachment must not use ResourceHandle::NONE
//      - Depth-stencil attachment must reference an existing resource
//
//   B. Dispatch source validity
//      - Compute and RayTracing passes MUST have dispatch_source = Some
//      - Graphics and Copy passes MUST NOT have dispatch_source = Some
//      - Direct dispatch requires group_count_x, group_count_y, group_count_z
//        all > 0
//      - Indirect dispatch must not use ResourceHandle::NONE as buffer
//      - Indirect dispatch must reference an existing resource as buffer
//      - InstanceSource::Indirect for graphics passes must have valid buffer
//        (not NONE and must reference existing resource)
//
//   C. No self-referencing read+write
//      - A resource handle appearing in both `reads` and `writes` is flagged
//        UNLESS it is also referenced as a colour or depth-stencil attachment
//        (where load+store semantics make dual access legitimate)
//      - Multiple self-references each produce separate error messages
//
// Scenarios:
//   1.  Valid graphics pass with colour attachment -> Ok
//   2.  Valid graphics pass with colour + depth-stencil -> Ok
//   3.  Valid compute pass with direct dispatch -> Ok
//   4.  Valid compute pass with indirect dispatch -> Ok
//   5.  Valid copy pass -> Ok
//   6.  Valid ray-tracing pass -> Ok
//   7.  Colour attachment with ResourceHandle::NONE -> error
//   8.  Colour attachment referencing unknown resource -> error
//   9.  Depth-stencil with ResourceHandle::NONE -> error
//  10.  Depth-stencil referencing unknown resource -> error
//  11.  Compute pass without dispatch_source -> error
//  12.  RayTracing pass without dispatch_source -> error
//  13.  Graphics pass with dispatch_source -> error
//  14.  Copy pass with dispatch_source -> error
//  15.  Direct dispatch with group_count_x = 0 -> error
//  16.  Direct dispatch with group_count_y = 0 -> error
//  17.  Direct dispatch with group_count_z = 0 -> error
//  18.  Indirect dispatch with ResourceHandle::NONE buffer -> error
//  19.  Indirect dispatch with unknown buffer handle -> error
//  20.  Graphics pass with InstanceSource::Indirect, NONE buffer -> error
//  21.  Graphics pass with InstanceSource::Indirect, unknown buffer -> error
//  22.  Pass reads and writes same non-attachment resource -> error
//  23.  Pass reads and writes same resource used as colour attachment -> Ok
//  24.  Pass reads and writes same resource used as depth-stencil -> Ok
//  25.  Multiple self-references produce multiple errors
//  26.  Combination of errors (invalid resource + self-ref) produces both
//  27.  Valid pass with no resource references -> Ok
//  28.  Valid pass with many resources -> Ok
//  29.  Empty resource list with valid pass that needs no resources -> Ok
//  30.  Empty resource list with pass referencing resource -> error
//
use renderer_backend::frame_graph::{
    mock_pass_compute, mock_pass_graphics, mock_resource_buffer, mock_resource_texture,
    ColorAttachment, DepthStencilAttachment, DispatchSource, InstanceSource, IrPass, IrResource,
    PassIndex, PassValidator, ResourceHandle, ViewType,
};

// =============================================================================
// SECTION 1 -- Valid passes
// =============================================================================

#[test]
fn valid_graphics_pass_with_colour_attachment() {
    // A graphics pass writing one texture as a colour attachment.
    let r = ResourceHandle(1);
    let resources = vec![mock_resource_texture(r, "color_rt", 1920, 1080)];
    let pass = mock_pass_graphics(PassIndex(0), "render_scene", &[r]);

    let result = PassValidator::validate(&pass, &resources);
    assert!(result.is_ok(), "valid graphics pass passes validation");
}

#[test]
fn valid_graphics_pass_with_colour_and_depth_stencil() {
    // A graphics pass with both a colour attachment and a depth-stencil
    // attachment, both referencing real resources.
    let color = ResourceHandle(1);
    let depth = ResourceHandle(2);
    let resources = vec![
        mock_resource_texture(color, "albedo", 1920, 1080),
        mock_resource_texture(depth, "depth_buffer", 1920, 1080),
    ];
    let pass = IrPass::graphics(
        PassIndex(0),
        "scene_depth",
        vec![ColorAttachment {
            resource: color,
            ..Default::default()
        }],
        Some(DepthStencilAttachment {
            resource: depth,
            ..Default::default()
        }),
        InstanceSource::Direct {
            index_count: 6,
            instance_count: 1,
            base_vertex: 0,
            first_index: 0,
            first_instance: 0,
        },
        ViewType::Texture2D,
    );

    let result = PassValidator::validate(&pass, &resources);
    assert!(result.is_ok(), "graphics pass with colour + depth passes");
}

#[test]
fn valid_compute_pass_with_direct_dispatch() {
    // A compute pass with a valid direct dispatch source.
    let r = ResourceHandle(1);
    let resources = vec![mock_resource_buffer(r, "data", 4096)];
    let mut pass = mock_pass_compute(PassIndex(0), "compute_task", &[r], &[]);

    let result = PassValidator::validate(&pass, &resources);
    assert!(result.is_ok(), "compute pass with direct dispatch passes");
}

#[test]
fn valid_compute_pass_with_indirect_dispatch() {
    // A compute pass with a valid indirect dispatch source referencing a
    // real buffer resource.
    let buf = ResourceHandle(1);
    let resources = vec![mock_resource_buffer(buf, "indirect_buf", 256)];
    let pass = IrPass::compute(
        PassIndex(0),
        "indirect_compute",
        DispatchSource::Indirect {
            buffer: buf,
            offset: 0,
        },
        ViewType::Storage,
    );

    let result = PassValidator::validate(&pass, &resources);
    assert!(result.is_ok(), "compute pass with indirect dispatch passes");
}

#[test]
fn valid_copy_pass() {
    // A copy pass has no dispatch source and no attachments.
    let pass = IrPass::copy(PassIndex(0), "buffer_copy");

    let result = PassValidator::validate(&pass, &[]);
    assert!(result.is_ok(), "copy pass always passes validation");
}

#[test]
fn valid_ray_tracing_pass() {
    // A ray-tracing pass with a valid direct dispatch.
    let pass = IrPass::ray_tracing(
        PassIndex(0),
        "rt_shadows",
        DispatchSource::Direct {
            group_count_x: 8,
            group_count_y: 8,
            group_count_z: 1,
        },
    );

    let result = PassValidator::validate(&pass, &[]);
    assert!(result.is_ok(), "ray-tracing pass with dispatch passes");
}

#[test]
fn valid_graphics_pass_mesh_shader_instance_source() {
    // Graphics pass using mesh shader instance source (valid).
    let r = ResourceHandle(1);
    let resources = vec![mock_resource_texture(r, "mesh_out", 800, 600)];
    let pass = IrPass::graphics(
        PassIndex(0),
        "mesh_render",
        vec![ColorAttachment {
            resource: r,
            ..Default::default()
        }],
        None,
        InstanceSource::Mesh {
            group_count_x: 4,
            group_count_y: 4,
            group_count_z: 1,
        },
        ViewType::Texture2D,
    );

    let result = PassValidator::validate(&pass, &resources);
    assert!(result.is_ok(), "graphics with Mesh instance source passes");
}

// =============================================================================
// SECTION 2 -- Attachment resource validity
// =============================================================================

#[test]
fn colour_attachment_with_none_handle_fails() {
    // A colour attachment using ResourceHandle::NONE must be caught.
    let pass = IrPass::graphics(
        PassIndex(0),
        "bad_attach",
        vec![ColorAttachment {
            resource: ResourceHandle::NONE,
            ..Default::default()
        }],
        None,
        InstanceSource::Direct {
            index_count: 6,
            instance_count: 1,
            base_vertex: 0,
            first_index: 0,
            first_instance: 0,
        },
        ViewType::Texture2D,
    );

    let result = PassValidator::validate(&pass, &[]);
    assert!(result.is_err(), "NONE colour attachment rejected");

    let errors = result.unwrap_err();
    assert!(
        errors.iter().any(|e| e.contains("NONE")),
        "error message mentions NONE: {:?}",
        errors,
    );
}

#[test]
fn colour_attachment_with_unknown_resource_fails() {
    // A colour attachment referencing a resource handle not present in the
    // resources list must be caught.
    let r = ResourceHandle(1); // not in resources
    let resources = vec![mock_resource_texture(ResourceHandle(2), "other", 64, 64)];
    let pass = mock_pass_graphics(PassIndex(0), "bad_ref", &[r]);

    let result = PassValidator::validate(&pass, &resources);
    assert!(result.is_err(), "unknown colour attachment resource rejected");

    let errors = result.unwrap_err();
    assert!(
        errors.iter().any(|e| e.contains("unknown") || e.contains("not found") || e.contains("ResourceHandle(1)")),
        "error message identifies the unknown handle: {:?}",
        errors,
    );
}

#[test]
fn depth_stencil_with_none_handle_fails() {
    // A depth-stencil attachment using ResourceHandle::NONE must be caught.
    let color = ResourceHandle(1);
    let resources = vec![mock_resource_texture(color, "color", 64, 64)];
    let pass = IrPass::graphics(
        PassIndex(0),
        "bad_depth",
        vec![ColorAttachment {
            resource: color,
            ..Default::default()
        }],
        Some(DepthStencilAttachment {
            resource: ResourceHandle::NONE,
            ..Default::default()
        }),
        InstanceSource::Direct {
            index_count: 6,
            instance_count: 1,
            base_vertex: 0,
            first_index: 0,
            first_instance: 0,
        },
        ViewType::Texture2D,
    );

    let result = PassValidator::validate(&pass, &resources);
    assert!(result.is_err(), "NONE depth-stencil rejected");

    let errors = result.unwrap_err();
    assert!(
        errors.iter().any(|e| e.contains("NONE")),
        "error message mentions NONE: {:?}",
        errors,
    );
}

#[test]
fn depth_stencil_with_unknown_resource_fails() {
    // A depth-stencil attachment referencing an unknown resource must be caught.
    let color = ResourceHandle(1);
    let bad_depth = ResourceHandle(99); // not in resources
    let resources = vec![mock_resource_texture(color, "color", 64, 64)];
    let pass = IrPass::graphics(
        PassIndex(0),
        "bad_depth_ref",
        vec![ColorAttachment {
            resource: color,
            ..Default::default()
        }],
        Some(DepthStencilAttachment {
            resource: bad_depth,
            ..Default::default()
        }),
        InstanceSource::Direct {
            index_count: 6,
            instance_count: 1,
            base_vertex: 0,
            first_index: 0,
            first_instance: 0,
        },
        ViewType::Texture2D,
    );

    let result = PassValidator::validate(&pass, &resources);
    assert!(result.is_err(), "unknown depth-stencil resource rejected");

    let errors = result.unwrap_err();
    assert!(
        errors.iter().any(|e| e.contains("99") || e.contains("unknown") || e.contains("not found")),
        "error message identifies the bad handle: {:?}",
        errors,
    );
}

#[test]
fn multiple_bad_attachments_all_reported() {
    // Two colour attachments both referencing unknown resources must each
    // produce an error.
    let r1 = ResourceHandle(1);
    let r2 = ResourceHandle(2);
    let resources = vec![]; // no resources registered
    let pass = mock_pass_graphics(PassIndex(0), "double_bad", &[r1, r2]);

    let result = PassValidator::validate(&pass, &resources);
    assert!(result.is_err(), "multiple bad attachments rejected");

    let errors = result.unwrap_err();
    // Should have at least 2 errors (one per bad attachment).
    assert!(
        errors.len() >= 2,
        "at least 2 errors for 2 bad attachments: got {}",
        errors.len(),
    );
}

// =============================================================================
// SECTION 3 -- Dispatch source validity (type-level checks)
// =============================================================================

#[test]
fn compute_pass_without_dispatch_source_fails() {
    // A compute pass with dispatch_source = None must be flagged.
    let pass = IrPass {
        dispatch_source: None,
        ..IrPass::compute(
            PassIndex(0),
            "no_dispatch",
            DispatchSource::Direct {
                group_count_x: 1,
                group_count_y: 1,
                group_count_z: 1,
            },
            ViewType::Storage,
        )
    };

    let result = PassValidator::validate(&pass, &[]);
    assert!(result.is_err(), "compute pass without dispatch rejected");

    let errors = result.unwrap_err();
    assert!(
        errors.iter().any(|e| e.contains("dispatch_source") && e.contains("None")),
        "error message mentions missing dispatch_source: {:?}",
        errors,
    );
}

#[test]
fn ray_tracing_pass_without_dispatch_source_fails() {
    // A ray-tracing pass with dispatch_source = None must be flagged.
    let pass = IrPass {
        dispatch_source: None,
        ..IrPass::ray_tracing(
            PassIndex(0),
            "rt_no_dispatch",
            DispatchSource::Direct {
                group_count_x: 1,
                group_count_y: 1,
                group_count_z: 1,
            },
        )
    };

    let result = PassValidator::validate(&pass, &[]);
    assert!(result.is_err(), "ray-tracing pass without dispatch rejected");

    let errors = result.unwrap_err();
    assert!(
        errors.iter().any(|e| e.contains("dispatch_source") && e.contains("None") && (e.contains("RayTracing") || e.contains("Compute"))),
        "error mentions missing dispatch for ray-tracing: {:?}",
        errors,
    );
}

#[test]
fn graphics_pass_with_dispatch_source_fails() {
    // A graphics pass with dispatch_source = Some must be flagged.
    let r = ResourceHandle(1);
    let resources = vec![mock_resource_texture(r, "color", 64, 64)];
    let pass = IrPass {
        dispatch_source: Some(DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        }),
        ..IrPass::graphics(
            PassIndex(0),
            "gfx_with_dispatch",
            vec![ColorAttachment {
                resource: r,
                ..Default::default()
            }],
            None,
            InstanceSource::Direct {
                index_count: 6,
                instance_count: 1,
                base_vertex: 0,
                first_index: 0,
                first_instance: 0,
            },
            ViewType::Texture2D,
        )
    };

    let result = PassValidator::validate(&pass, &resources);
    assert!(result.is_err(), "graphics pass with dispatch rejected");

    let errors = result.unwrap_err();
    assert!(
        errors.iter().any(|e| e.contains("dispatch_source") && e.contains("Some")),
        "error mentions unexpected dispatch_source: {:?}",
        errors,
    );
}

#[test]
fn copy_pass_with_dispatch_source_fails() {
    // A copy pass with dispatch_source = Some must be flagged.
    let pass = IrPass {
        dispatch_source: Some(DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        }),
        ..IrPass::copy(PassIndex(0), "copy_with_dispatch")
    };

    let result = PassValidator::validate(&pass, &[]);
    assert!(result.is_err(), "copy pass with dispatch rejected");

    let errors = result.unwrap_err();
    assert!(
        errors.iter().any(|e| e.contains("dispatch_source") && e.contains("Some")),
        "error mentions unexpected dispatch_source: {:?}",
        errors,
    );
}

// =============================================================================
// SECTION 4 -- Dispatch source validity (value-level: direct dispatch)
// =============================================================================

#[test]
fn direct_dispatch_with_zero_group_count_x_fails() {
    // Direct dispatch with group_count_x = 0 must be caught.
    let pass = IrPass::compute(
        PassIndex(0),
        "bad_dispatch_x",
        DispatchSource::Direct {
            group_count_x: 0,
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::Storage,
    );

    let result = PassValidator::validate(&pass, &[]);
    assert!(result.is_err(), "dispatch with x=0 rejected");

    let errors = result.unwrap_err();
    assert!(
        errors.iter().any(|e| e.contains("group_count_x") && e.contains("0")),
        "error mentions group_count_x = 0: {:?}",
        errors,
    );
}

#[test]
fn direct_dispatch_with_zero_group_count_y_fails() {
    // Direct dispatch with group_count_y = 0 must be caught.
    let pass = IrPass::compute(
        PassIndex(0),
        "bad_dispatch_y",
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 0,
            group_count_z: 1,
        },
        ViewType::Storage,
    );

    let result = PassValidator::validate(&pass, &[]);
    assert!(result.is_err(), "dispatch with y=0 rejected");

    let errors = result.unwrap_err();
    assert!(
        errors.iter().any(|e| e.contains("group_count_y") && e.contains("0")),
        "error mentions group_count_y = 0: {:?}",
        errors,
    );
}

#[test]
fn direct_dispatch_with_zero_group_count_z_fails() {
    // Direct dispatch with group_count_z = 0 must be caught.
    let pass = IrPass::compute(
        PassIndex(0),
        "bad_dispatch_z",
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 0,
        },
        ViewType::Storage,
    );

    let result = PassValidator::validate(&pass, &[]);
    assert!(result.is_err(), "dispatch with z=0 rejected");

    let errors = result.unwrap_err();
    assert!(
        errors.iter().any(|e| e.contains("group_count_z") && e.contains("0")),
        "error mentions group_count_z = 0: {:?}",
        errors,
    );
}

#[test]
fn direct_dispatch_with_all_zeros_reports_all_dimensions() {
    // All three dimensions zero should produce errors for x, y, and z.
    let pass = IrPass::compute(
        PassIndex(0),
        "all_zero_dispatch",
        DispatchSource::Direct {
            group_count_x: 0,
            group_count_y: 0,
            group_count_z: 0,
        },
        ViewType::Storage,
    );

    let result = PassValidator::validate(&pass, &[]);
    assert!(result.is_err(), "all-zero dispatch rejected");

    let errors = result.unwrap_err();
    // Should report at least 3 errors (one per zero dimension).
    assert!(
        errors.len() >= 3,
        "at least 3 errors for all-zero dispatch: got {}",
        errors.len(),
    );
}

// =============================================================================
// SECTION 5 -- Dispatch source validity (value-level: indirect dispatch)
// =============================================================================

#[test]
fn indirect_dispatch_with_none_buffer_fails() {
    // Indirect dispatch using ResourceHandle::NONE as buffer must be caught.
    let pass = IrPass::compute(
        PassIndex(0),
        "bad_indirect",
        DispatchSource::Indirect {
            buffer: ResourceHandle::NONE,
            offset: 0,
        },
        ViewType::Storage,
    );

    let result = PassValidator::validate(&pass, &[]);
    assert!(result.is_err(), "indirect with NONE buffer rejected");

    let errors = result.unwrap_err();
    assert!(
        errors.iter().any(|e| e.contains("NONE")),
        "error mentions NONE buffer for indirect dispatch: {:?}",
        errors,
    );
}

#[test]
fn indirect_dispatch_with_unknown_buffer_fails() {
    // Indirect dispatch referencing a buffer handle not in the resource list.
    let pass = IrPass::compute(
        PassIndex(0),
        "unknown_indirect",
        DispatchSource::Indirect {
            buffer: ResourceHandle(99),
            offset: 0,
        },
        ViewType::Storage,
    );

    let result = PassValidator::validate(&pass, &[]);
    assert!(result.is_err(), "indirect with unknown buffer rejected");

    let errors = result.unwrap_err();
    assert!(
        errors.iter().any(|e| e.contains("99") || e.contains("unknown")),
        "error identifies the unknown buffer: {:?}",
        errors,
    );
}

#[test]
fn indirect_dispatch_with_valid_buffer_passes() {
    // Indirect dispatch referencing a real buffer resource must pass.
    let buf = ResourceHandle(1);
    let resources = vec![mock_resource_buffer(buf, "indirect_args", 256)];
    let pass = IrPass::compute(
        PassIndex(0),
        "good_indirect",
        DispatchSource::Indirect {
            buffer: buf,
            offset: 0,
        },
        ViewType::Storage,
    );

    let result = PassValidator::validate(&pass, &resources);
    assert!(result.is_ok(), "indirect dispatch with valid buffer passes");
}

// =============================================================================
// SECTION 6 -- Graphics pass InstanceSource::Indirect validation
// =============================================================================

#[test]
fn graphics_pass_indirect_instance_none_buffer_fails() {
    // Graphics pass with InstanceSource::Indirect using ResourceHandle::NONE.
    let r = ResourceHandle(1);
    let resources = vec![mock_resource_texture(r, "color", 64, 64)];
    let pass = IrPass::graphics(
        PassIndex(0),
        "bad_indirect_draw",
        vec![ColorAttachment {
            resource: r,
            ..Default::default()
        }],
        None,
        InstanceSource::Indirect {
            buffer: ResourceHandle::NONE,
            offset: 0,
            draw_count: 1,
            stride: 16,
        },
        ViewType::Texture2D,
    );

    let result = PassValidator::validate(&pass, &resources);
    assert!(result.is_err(), "indirect draw with NONE buffer rejected");

    let errors = result.unwrap_err();
    assert!(
        errors.iter().any(|e| e.contains("NONE")),
        "error mentions NONE for indirect draw: {:?}",
        errors,
    );
}

#[test]
fn graphics_pass_indirect_instance_unknown_buffer_fails() {
    // Graphics pass with InstanceSource::Indirect referencing unknown buffer.
    let r = ResourceHandle(1);
    let resources = vec![mock_resource_texture(r, "color", 64, 64)];
    let pass = IrPass::graphics(
        PassIndex(0),
        "unknown_indirect_draw",
        vec![ColorAttachment {
            resource: r,
            ..Default::default()
        }],
        None,
        InstanceSource::Indirect {
            buffer: ResourceHandle(99),
            offset: 0,
            draw_count: 1,
            stride: 16,
        },
        ViewType::Texture2D,
    );

    let result = PassValidator::validate(&pass, &resources);
    assert!(result.is_err(), "indirect draw with unknown buffer rejected");

    let errors = result.unwrap_err();
    assert!(
        errors.iter().any(|e| e.contains("99") || e.contains("unknown")),
        "error identifies unknown buffer for indirect draw: {:?}",
        errors,
    );
}

// =============================================================================
// SECTION 7 -- Self-referencing read+write
// =============================================================================

#[test]
fn pass_reads_and_writes_same_non_attachment_resource_fails() {
    // A compute pass that reads and writes the same resource (not an
    // attachment) must be flagged.
    let r = ResourceHandle(1);
    let resources = vec![mock_resource_buffer(r, "rw_data", 4096)];
    let mut pass = mock_pass_compute(PassIndex(0), "self_ref", &[], &[]);
    pass.access_set.reads.push(r);
    pass.access_set.writes.push(r);

    let result = PassValidator::validate(&pass, &resources);
    assert!(result.is_err(), "self-referencing read+write rejected");

    let errors = result.unwrap_err();
    assert!(
        errors.iter().any(|e| e.contains("reads") && e.contains("writes")),
        "error mentions read+write conflict: {:?}",
        errors,
    );
}

#[test]
fn same_resource_read_and_write_as_colour_attachment_ok() {
    // When the dual-access resource is a colour attachment, the validator
    // must allow it (load+store semantics).
    let r = ResourceHandle(1);
    let resources = vec![mock_resource_texture(r, "color_rt", 1920, 1080)];

    // Build a graphics pass where the colour attachment resource is also
    // explicitly listed in access_set.reads (simulating load).
    let mut pass = IrPass::graphics(
        PassIndex(0),
        "load_store",
        vec![ColorAttachment {
            resource: r,
            load_op: renderer_backend::frame_graph::AttachmentLoadOp::Load,
            ..Default::default()
        }],
        None,
        InstanceSource::Direct {
            index_count: 6,
            instance_count: 1,
            base_vertex: 0,
            first_index: 0,
            first_instance: 0,
        },
        ViewType::Texture2D,
    );
    // Explicitly add r to the reads set (it should already be there after
    // sync_access_set_from_attachments, but be explicit).
    pass.access_set.reads.push(r);

    let result = PassValidator::validate(&pass, &resources);
    assert!(
        result.is_ok(),
        "colour attachment read+write is allowed (load+store semantics)",
    );
}

#[test]
fn same_resource_read_and_write_as_depth_stencil_ok() {
    // When the dual-access resource is a depth-stencil attachment, the
    // validator must allow it.
    let color = ResourceHandle(1);
    let depth = ResourceHandle(2);
    let resources = vec![
        mock_resource_texture(color, "color", 1920, 1080),
        mock_resource_texture(depth, "depth", 1920, 1080),
    ];

    let mut pass = IrPass::graphics(
        PassIndex(0),
        "depth_load_store",
        vec![ColorAttachment {
            resource: color,
            ..Default::default()
        }],
        Some(DepthStencilAttachment {
            resource: depth,
            depth_load_op: renderer_backend::frame_graph::AttachmentLoadOp::Load,
            ..Default::default()
        }),
        InstanceSource::Direct {
            index_count: 6,
            instance_count: 1,
            base_vertex: 0,
            first_index: 0,
            first_instance: 0,
        },
        ViewType::Texture2D,
    );
    // Depth buffer is also in the read set.
    pass.access_set.reads.push(depth);

    let result = PassValidator::validate(&pass, &resources);
    assert!(
        result.is_ok(),
        "depth-stencil attachment read+write is allowed",
    );
}

#[test]
fn multiple_self_references_produce_multiple_errors() {
    // If the pass reads+write two distinct non-attachment resources, both
    // must produce error messages.
    let r1 = ResourceHandle(1);
    let r2 = ResourceHandle(2);
    let resources = vec![
        mock_resource_buffer(r1, "buf_a", 1024),
        mock_resource_buffer(r2, "buf_b", 2048),
    ];
    let mut pass = mock_pass_compute(PassIndex(0), "multi_self_ref", &[], &[]);
    pass.access_set.reads.push(r1);
    pass.access_set.writes.push(r1);
    pass.access_set.reads.push(r2);
    pass.access_set.writes.push(r2);

    let result = PassValidator::validate(&pass, &resources);
    assert!(result.is_err(), "multiple self-refs rejected");

    let errors = result.unwrap_err();
    assert!(
        errors.len() >= 2,
        "at least 2 errors for 2 self-refs: got {}",
        errors.len(),
    );
}

// =============================================================================
// SECTION 8 -- Combination and edge cases
// =============================================================================

#[test]
fn combination_invalid_resource_and_self_ref() {
    // Pass with both an invalid resource reference and a self-ref.
    // Both should be reported.
    let r_valid = ResourceHandle(1);
    let r_self = ResourceHandle(2);
    let r_missing = ResourceHandle(99);
    let resources = vec![
        mock_resource_buffer(r_valid, "valid", 64),
        mock_resource_buffer(r_self, "self_ref", 128),
    ];
    let mut pass = mock_pass_compute(PassIndex(0), "combo", &[], &[]);
    pass.access_set.reads.push(r_valid);
    pass.access_set.reads.push(r_missing); // unknown
    pass.access_set.writes.push(r_self);
    pass.access_set.reads.push(r_self);

    let result = PassValidator::validate(&pass, &resources);
    assert!(result.is_err(), "combination of errors rejected");

    let errors = result.unwrap_err();
    // At least 2 errors: one for unknown resource, one for self-ref.
    assert!(
        errors.len() >= 2,
        "at least 2 errors for combination: got {}",
        errors.len(),
    );
}

#[test]
fn pass_with_no_resource_references_valid() {
    // A pass with no reads, no writes, no attachments, no dispatch.
    // Copy pass is the canonical example.
    let pass = IrPass::copy(PassIndex(0), "noop_copy");

    let result = PassValidator::validate(&pass, &[]);
    assert!(result.is_ok(), "pass with no resource references passes");
}

#[test]
fn pass_with_all_valid_resource_references_ok() {
    // A pass that reads and writes many valid resources but has no
    // self-ref issues should pass.
    let resources: Vec<IrResource> = (0..10)
        .map(|i| mock_resource_buffer(ResourceHandle(i), &format!("buf_{}", i), 64))
        .collect();
    let reads: Vec<ResourceHandle> = (0..5).map(ResourceHandle).collect();
    let writes: Vec<ResourceHandle> = (5..10).map(ResourceHandle).collect();

    let mut pass = mock_pass_compute(PassIndex(0), "many_resources", &[], &[]);
    pass.access_set.reads.extend(reads);
    pass.access_set.writes.extend(writes);

    let result = PassValidator::validate(&pass, &resources);
    assert!(result.is_ok(), "many valid resource references pass");
}

#[test]
fn empty_resource_list_with_pass_needing_none_ok() {
    // Copy pass with empty resource list is valid.
    let pass = IrPass::copy(PassIndex(0), "empty_copy");
    assert!(PassValidator::validate(&pass, &[]).is_ok());
}

#[test]
fn empty_resource_list_with_pass_referencing_resource_fails() {
    // A pass that references a resource when the resource list is empty.
    let mut pass = mock_pass_compute(PassIndex(0), "missing_res", &[], &[]);
    pass.access_set.writes.push(ResourceHandle(1));

    let result = PassValidator::validate(&pass, &[]);
    assert!(result.is_err(), "resource not found in empty list");
}

#[test]
fn colour_attachment_resource_also_in_writes_not_double_reported() {
    // When a colour attachment resource is also in access_set.writes, the
    // resource should be validated as a colour attachment (not flagged as
    // self-ref). The validator should produce exactly the expected attachment
    // errors, not duplicate self-ref errors.
    let r = ResourceHandle(1);
    let resources = vec![mock_resource_texture(r, "tex", 64, 64)];

    let pass = IrPass::graphics(
        PassIndex(0),
        "attachment_in_writes",
        vec![ColorAttachment {
            resource: r,
            ..Default::default()
        }],
        None,
        InstanceSource::Direct {
            index_count: 6,
            instance_count: 1,
            base_vertex: 0,
            first_index: 0,
            first_instance: 0,
        },
        ViewType::Texture2D,
    );
    // The colour attachment resource appears in writes (that's normal).
    // The sync_access_set_from_attachments call in the constructor adds it.
    // So this should pass without errors.

    let result = PassValidator::validate(&pass, &resources);
    assert!(result.is_ok(), "attachment resource in writes is allowed");
}

#[test]
fn ray_tracing_pass_with_indirect_dispatch_passes() {
    // RayTracing pass with valid indirect dispatch.
    let buf = ResourceHandle(1);
    let resources = vec![mock_resource_buffer(buf, "rt_indirect", 256)];
    let pass = IrPass::ray_tracing(
        PassIndex(0),
        "rt_indirect",
        DispatchSource::Indirect {
            buffer: buf,
            offset: 0,
        },
    );

    let result = PassValidator::validate(&pass, &resources);
    assert!(result.is_ok(), "ray-tracing with indirect dispatch passes");
}

#[test]
fn compute_pass_with_valid_dispatch_and_attachments_passes() {
    // Compute pass reads a texture, writes a buffer, has valid dispatch.
    let tex = ResourceHandle(1);
    let buf = ResourceHandle(2);
    let resources = vec![
        mock_resource_texture(tex, "input", 1920, 1080),
        mock_resource_buffer(buf, "output", 4096),
    ];
    let mut pass = IrPass::compute(
        PassIndex(0),
        "post_process",
        DispatchSource::Direct {
            group_count_x: 16,
            group_count_y: 16,
            group_count_z: 1,
        },
        ViewType::Storage,
    );
    pass.access_set.reads.push(tex);
    pass.access_set.writes.push(buf);

    let result = PassValidator::validate(&pass, &resources);
    assert!(result.is_ok(), "compute pass with valid dispatch and resources passes");
}

#[test]
fn graphics_pass_without_any_attachment_no_dispatch_passes() {
    // A graphics pass with zero colour attachments and no depth-stencil.
    // This is unusual but structurally valid: no attachments = no NONE check
    // triggered, no dispatch_source = Ok for Graphics.
    let pass = IrPass::graphics(
        PassIndex(0),
        "empty_gfx",
        vec![],
        None,
        InstanceSource::Direct {
            index_count: 0,
            instance_count: 0,
            base_vertex: 0,
            first_index: 0,
            first_instance: 0,
        },
        ViewType::Texture2D,
    );

    let result = PassValidator::validate(&pass, &[]);
    assert!(result.is_ok(), "graphics pass with no attachments and no dispatch passes");
}

#[test]
fn validate_returns_multiple_errors_in_single_call() {
    // Trigger multiple validation failures in one call and verify all
    // are returned.
    let r = ResourceHandle(1); // not registered
    let mut pass = IrPass::compute(
        PassIndex(0),
        "multi_error",
        DispatchSource::Direct {
            group_count_x: 0, // invalid
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::Storage,
    );
    pass.access_set.writes.push(r); // unknown resource
    pass.access_set.reads.push(r);

    let result = PassValidator::validate(&pass, &[]);
    assert!(result.is_err(), "multiple errors returned");

    let errors = result.unwrap_err();
    // At minimum: 1 for zero group_count_x + 1 for unknown resource
    // (self-ref may or may not fire since the resource is unknown).
    assert!(
        errors.len() >= 2,
        "at least 2 errors from multi-error pass: got {}",
        errors.len(),
    );
}

#[test]
fn error_messages_are_human_readable() {
    // Verify error messages contain descriptive text rather than raw
    // debug output.
    let pass = IrPass::compute(
        PassIndex(5),
        "test_pass",
        DispatchSource::Direct {
            group_count_x: 0,
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::Storage,
    );

    let result = PassValidator::validate(&pass, &[]);
    let errors = result.unwrap_err();

    assert!(!errors.is_empty(), "at least one error");
    for msg in &errors {
        // Each message should contain the pass index or name for context.
        let has_pass_ref = msg.contains("5") || msg.contains("test_pass");
        assert!(has_pass_ref, "error message references the pass: '{}'", msg);
    }
}
