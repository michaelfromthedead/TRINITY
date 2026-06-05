// Blackbox contract tests for T-WGPU-P7.5.5 Render Pass Declaration
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::frame_graph::passes::*` and
// `renderer_backend::frame_graph::graph::{ResourceId, PassId}` -- no internal
// fields, no private methods, no implementation details.
//
// Contract:
//   Render pass declaration provides structs and builders for declaring render
//   passes within the frame graph. Components include:
//
//   PassLoadOp     - Load operation (Clear, Load, DontCare)
//   PassStoreOp    - Store operation (Store, Discard)
//   PassViewport   - Viewport and depth range configuration
//   PassColorAttachment - Color output target with load/store ops
//   PassDepthAttachment - Depth/stencil target with separate ops
//   RenderPassConfig    - Complete render pass configuration
//   RenderPassBuilder   - Fluent API for constructing passes
//   PassExecutor        - Trait for custom pass execution
//   RenderPassNode      - Fully configured pass for scheduling
//
// Test Categories:
//   1. API Contract Tests (20+): Enum variants, struct construction, defaults
//   2. Render Pass Scenarios (30+): Forward, deferred, post-process, shadow, MSAA
//   3. Builder Workflows (25+): Fluent API, chaining, resource tracking
//   4. Integration with FrameGraph (15+): Pass creation, dependency tracking
//   5. Edge Cases (15+): Boundaries, validation, error handling

use renderer_backend::frame_graph::graph::{PassId, RenderContext, ResourceId};
use renderer_backend::frame_graph::passes::{
    FnExecutor, NoOpExecutor, PassColorAttachment, PassDepthAttachment, PassExecutor, PassLoadOp,
    PassStoreOp, PassViewport, RenderPassBuilder, RenderPassConfig, RenderPassNode,
};

// ===========================================================================
// CATEGORY 1: API CONTRACT TESTS (20+)
// ===========================================================================

// Test 1: PassLoadOp::Clear variant exists and is default
#[test]
fn test_pass_load_op_clear_is_default() {
    let op = PassLoadOp::default();
    assert_eq!(op, PassLoadOp::Clear);
    assert!(op.is_clear());
}

// Test 2: PassLoadOp::Load variant exists
#[test]
fn test_pass_load_op_load_variant() {
    let op = PassLoadOp::Load;
    assert!(op.is_load());
    assert!(!op.is_clear());
    assert!(!op.is_dont_care());
}

// Test 3: PassLoadOp::DontCare variant exists
#[test]
fn test_pass_load_op_dont_care_variant() {
    let op = PassLoadOp::DontCare;
    assert!(op.is_dont_care());
    assert!(!op.is_clear());
    assert!(!op.is_load());
}

// Test 4: PassStoreOp::Store variant exists and is default
#[test]
fn test_pass_store_op_store_is_default() {
    let op = PassStoreOp::default();
    assert_eq!(op, PassStoreOp::Store);
    assert!(op.is_store());
}

// Test 5: PassStoreOp::Discard variant exists
#[test]
fn test_pass_store_op_discard_variant() {
    let op = PassStoreOp::Discard;
    assert!(op.is_discard());
    assert!(!op.is_store());
}

// Test 6: PassViewport construction with new()
#[test]
fn test_pass_viewport_new() {
    let vp = PassViewport::new(10.0, 20.0, 800.0, 600.0);
    assert_eq!(vp.x, 10.0);
    assert_eq!(vp.y, 20.0);
    assert_eq!(vp.width, 800.0);
    assert_eq!(vp.height, 600.0);
    assert_eq!(vp.min_depth, 0.0);
    assert_eq!(vp.max_depth, 1.0);
}

// Test 7: PassViewport construction with with_size()
#[test]
fn test_pass_viewport_with_size() {
    let vp = PassViewport::with_size(1920.0, 1080.0);
    assert_eq!(vp.x, 0.0);
    assert_eq!(vp.y, 0.0);
    assert_eq!(vp.width, 1920.0);
    assert_eq!(vp.height, 1080.0);
}

// Test 8: PassViewport default values
#[test]
fn test_pass_viewport_default() {
    let vp = PassViewport::default();
    assert_eq!(vp.x, 0.0);
    assert_eq!(vp.y, 0.0);
    assert_eq!(vp.width, 1920.0);
    assert_eq!(vp.height, 1080.0);
    assert_eq!(vp.min_depth, 0.0);
    assert_eq!(vp.max_depth, 1.0);
    assert!(vp.is_valid());
}

// Test 9: PassColorAttachment::new() construction
#[test]
fn test_pass_color_attachment_new() {
    let res = ResourceId::new(42);
    let att = PassColorAttachment::new(res);
    assert_eq!(att.resource, res);
    assert_eq!(att.load_op, PassLoadOp::Clear);
    assert_eq!(att.store_op, PassStoreOp::Store);
    assert_eq!(att.clear_color, Some([0.0, 0.0, 0.0, 1.0]));
    assert!(att.resolve_target.is_none());
}

// Test 10: PassColorAttachment::load() construction
#[test]
fn test_pass_color_attachment_load() {
    let res = ResourceId::new(10);
    let att = PassColorAttachment::load(res);
    assert_eq!(att.resource, res);
    assert_eq!(att.load_op, PassLoadOp::Load);
    assert_eq!(att.store_op, PassStoreOp::Store);
    assert!(att.clear_color.is_none());
}

// Test 11: PassColorAttachment::clear() construction
#[test]
fn test_pass_color_attachment_clear() {
    let res = ResourceId::new(11);
    let color = [1.0, 0.5, 0.25, 0.75];
    let att = PassColorAttachment::clear(res, color);
    assert_eq!(att.resource, res);
    assert_eq!(att.load_op, PassLoadOp::Clear);
    assert_eq!(att.clear_color, Some(color));
}

// Test 12: PassColorAttachment::transient() construction
#[test]
fn test_pass_color_attachment_transient() {
    let res = ResourceId::new(12);
    let att = PassColorAttachment::transient(res);
    assert_eq!(att.resource, res);
    assert_eq!(att.load_op, PassLoadOp::DontCare);
    assert_eq!(att.store_op, PassStoreOp::Discard);
    assert!(att.clear_color.is_none());
}

// Test 13: PassDepthAttachment::new() construction
#[test]
fn test_pass_depth_attachment_new() {
    let res = ResourceId::new(100);
    let att = PassDepthAttachment::new(res);
    assert_eq!(att.resource, res);
    assert_eq!(att.depth_load_op, PassLoadOp::Clear);
    assert_eq!(att.depth_store_op, PassStoreOp::Store);
    assert_eq!(att.stencil_load_op, PassLoadOp::DontCare);
    assert_eq!(att.stencil_store_op, PassStoreOp::Discard);
    assert_eq!(att.clear_depth, 1.0);
    assert_eq!(att.clear_stencil, 0);
    assert!(!att.read_only);
}

// Test 14: PassDepthAttachment::load() construction
#[test]
fn test_pass_depth_attachment_load() {
    let res = ResourceId::new(101);
    let att = PassDepthAttachment::load(res);
    assert_eq!(att.resource, res);
    assert_eq!(att.depth_load_op, PassLoadOp::Load);
    assert!(!att.read_only);
}

// Test 15: PassDepthAttachment::read_only() construction
#[test]
fn test_pass_depth_attachment_read_only() {
    let res = ResourceId::new(102);
    let att = PassDepthAttachment::read_only(res);
    assert_eq!(att.resource, res);
    assert_eq!(att.depth_load_op, PassLoadOp::Load);
    assert!(att.read_only);
    assert!(!att.writes_depth());
    assert!(!att.writes_stencil());
}

// Test 16: PassDepthAttachment::with_stencil() construction
#[test]
fn test_pass_depth_attachment_with_stencil() {
    let res = ResourceId::new(103);
    let att = PassDepthAttachment::with_stencil(res);
    assert_eq!(att.stencil_load_op, PassLoadOp::Clear);
    assert_eq!(att.stencil_store_op, PassStoreOp::Store);
    assert!(att.writes_stencil());
}

// Test 17: RenderPassConfig::new() construction
#[test]
fn test_render_pass_config_new() {
    let config = RenderPassConfig::new("my_pass");
    assert_eq!(config.name, "my_pass");
    assert!(config.color_attachments.is_empty());
    assert!(config.depth_attachment.is_none());
    assert_eq!(config.sample_count, 1);
    assert!(config.viewport.is_none());
}

// Test 18: RenderPassConfig::with_color() construction
#[test]
fn test_render_pass_config_with_color() {
    let color = PassColorAttachment::new(ResourceId::new(1));
    let config = RenderPassConfig::with_color("color_pass", color);
    assert_eq!(config.name, "color_pass");
    assert_eq!(config.color_attachments.len(), 1);
    assert!(config.has_color());
    assert!(!config.has_depth());
}

// Test 19: RenderPassConfig::with_color_and_depth() construction
#[test]
fn test_render_pass_config_with_color_and_depth() {
    let color = PassColorAttachment::new(ResourceId::new(1));
    let depth = PassDepthAttachment::new(ResourceId::new(2));
    let config = RenderPassConfig::with_color_and_depth("full_pass", color, depth);
    assert!(config.has_color());
    assert!(config.has_depth());
    assert_eq!(config.color_attachments.len(), 1);
}

// Test 20: RenderPassBuilder::new() construction
#[test]
fn test_render_pass_builder_new() {
    let builder = RenderPassBuilder::new("builder_pass");
    let config = builder.build();
    assert_eq!(config.name, "builder_pass");
    assert!(config.color_attachments.is_empty());
}

// Test 21: NoOpExecutor construction
#[test]
fn test_no_op_executor_construction() {
    let executor = NoOpExecutor;
    assert_eq!(executor.name(), "NoOpExecutor");
}

// Test 22: RenderPassNode::new() construction
#[test]
fn test_render_pass_node_new() {
    let config = RenderPassConfig::with_color("test", PassColorAttachment::new(ResourceId::new(1)));
    let executor = Box::new(NoOpExecutor);
    let node = RenderPassNode::new(PassId::new(99), config.clone(), executor);
    assert_eq!(node.id, PassId::new(99));
    assert_eq!(node.name(), "test");
}

// Test 23: RenderPassNode::empty() construction
#[test]
fn test_render_pass_node_empty() {
    let config = RenderPassConfig::with_color("empty", PassColorAttachment::new(ResourceId::new(1)));
    let node = RenderPassNode::empty(PassId::new(0), config);
    assert_eq!(node.executor.name(), "NoOpExecutor");
}

// ===========================================================================
// CATEGORY 2: RENDER PASS SCENARIOS (30+)
// ===========================================================================

// Test 24: Forward rendering - single color attachment
#[test]
fn test_forward_rendering_single_color() {
    let color = PassColorAttachment::clear(ResourceId::new(1), [0.1, 0.2, 0.3, 1.0]);
    let config = RenderPassConfig::with_color("forward_pass", color);

    assert_eq!(config.color_attachment_count(), 1);
    assert!(!config.has_depth());
    assert!(!config.is_multisampled());
    assert!(config.validate().is_none());
}

// Test 25: Forward rendering - with depth
#[test]
fn test_forward_rendering_with_depth() {
    let color = PassColorAttachment::clear(ResourceId::new(1), [0.0, 0.0, 0.0, 1.0]);
    let depth = PassDepthAttachment::new(ResourceId::new(2));
    let config = RenderPassConfig::with_color_and_depth("forward_depth", color, depth);

    assert!(config.has_color());
    assert!(config.has_depth());
    assert!(config.validate().is_none());
}

// Test 26: Forward rendering - clear both attachments
#[test]
fn test_forward_rendering_clear_both() {
    let color = PassColorAttachment::clear(ResourceId::new(1), [0.5, 0.5, 0.5, 1.0]);
    let depth = PassDepthAttachment::new(ResourceId::new(2)).with_clear_depth(1.0);
    let config = RenderPassConfig::with_color_and_depth("forward_clear", color, depth);

    assert_eq!(config.color_attachments[0].load_op, PassLoadOp::Clear);
    assert_eq!(config.depth_attachment.as_ref().unwrap().depth_load_op, PassLoadOp::Clear);
}

// Test 27: Deferred G-Buffer - multiple color attachments (albedo, normal, material)
#[test]
fn test_deferred_gbuffer_multiple_colors() {
    let albedo = PassColorAttachment::clear(ResourceId::new(1), [0.0, 0.0, 0.0, 1.0]);
    let normal = PassColorAttachment::clear(ResourceId::new(2), [0.5, 0.5, 1.0, 1.0]);
    let material = PassColorAttachment::clear(ResourceId::new(3), [0.0, 0.0, 0.0, 0.0]);

    let config = RenderPassBuilder::new("gbuffer_pass")
        .add_color_attachment(albedo)
        .add_color_attachment(normal)
        .add_color_attachment(material)
        .build();

    assert_eq!(config.color_attachment_count(), 3);
    assert!(config.validate().is_none());
}

// Test 28: Deferred G-Buffer - with depth attachment
#[test]
fn test_deferred_gbuffer_with_depth() {
    let albedo = PassColorAttachment::new(ResourceId::new(1));
    let normal = PassColorAttachment::new(ResourceId::new(2));
    let depth = PassDepthAttachment::new(ResourceId::new(10));

    let config = RenderPassBuilder::new("gbuffer_depth")
        .add_color_attachment(albedo)
        .add_color_attachment(normal)
        .set_depth_attachment(depth)
        .build();

    assert!(config.has_color());
    assert!(config.has_depth());
}

// Test 29: Deferred G-Buffer - clear all attachments
#[test]
fn test_deferred_gbuffer_clear_all() {
    let config = RenderPassBuilder::new("gbuffer_clear")
        .add_color_attachment(PassColorAttachment::clear(ResourceId::new(1), [0.0, 0.0, 0.0, 1.0]))
        .add_color_attachment(PassColorAttachment::clear(ResourceId::new(2), [0.5, 0.5, 1.0, 1.0]))
        .add_color_attachment(PassColorAttachment::clear(ResourceId::new(3), [1.0, 1.0, 1.0, 1.0]))
        .set_depth_attachment(PassDepthAttachment::new(ResourceId::new(4)))
        .build();

    for att in &config.color_attachments {
        assert!(att.load_op.is_clear());
    }
    assert!(config.depth_attachment.as_ref().unwrap().depth_load_op.is_clear());
}

// Test 30: Post-process - load from previous pass
#[test]
fn test_post_process_load_previous() {
    let color = PassColorAttachment::load(ResourceId::new(1));
    let config = RenderPassConfig::with_color("post_process", color);

    assert_eq!(config.color_attachments[0].load_op, PassLoadOp::Load);
    assert_eq!(config.color_attachments[0].store_op, PassStoreOp::Store);
}

// Test 31: Post-process - no depth
#[test]
fn test_post_process_no_depth() {
    let color = PassColorAttachment::load(ResourceId::new(1));
    let config = RenderPassConfig::with_color("post_no_depth", color);

    assert!(!config.has_depth());
}

// Test 32: Post-process - store result
#[test]
fn test_post_process_store_result() {
    let config = RenderPassBuilder::new("post_store")
        .add_color_attachment(PassColorAttachment {
            resource: ResourceId::new(1),
            load_op: PassLoadOp::Load,
            store_op: PassStoreOp::Store,
            clear_color: None,
            resolve_target: None,
        })
        .build();

    assert!(config.color_attachments[0].store_op.is_store());
}

// Test 33: Shadow map - depth only
#[test]
fn test_shadow_map_depth_only() {
    let depth = PassDepthAttachment::new(ResourceId::new(100));
    let mut config = RenderPassConfig::new("shadow_pass");
    config.depth_attachment = Some(depth);

    assert!(!config.has_color());
    assert!(config.has_depth());
    assert!(config.validate().is_none());
}

// Test 34: Shadow map - no color attachments
#[test]
fn test_shadow_map_no_color() {
    let config = RenderPassBuilder::new("shadow_no_color")
        .set_depth_attachment(PassDepthAttachment::new(ResourceId::new(1)))
        .build();

    assert!(config.color_attachments.is_empty());
    assert!(config.has_depth());
}

// Test 35: Shadow map - clear depth
#[test]
fn test_shadow_map_clear_depth() {
    let depth = PassDepthAttachment::new(ResourceId::new(1)).with_clear_depth(1.0);
    let mut config = RenderPassConfig::new("shadow_clear");
    config.depth_attachment = Some(depth);

    let d = config.depth_attachment.as_ref().unwrap();
    assert!(d.depth_load_op.is_clear());
    assert_eq!(d.clear_depth, 1.0);
}

// Test 36: MSAA - color with resolve target
#[test]
fn test_msaa_color_with_resolve() {
    let color = PassColorAttachment::new(ResourceId::new(1))
        .with_resolve(ResourceId::new(2));
    let config = RenderPassBuilder::new("msaa_pass")
        .add_color_attachment(color)
        .sample_count(4)
        .build();

    assert!(config.is_multisampled());
    assert_eq!(config.sample_count, 4);
    assert!(config.color_attachments[0].has_resolve());
}

// Test 37: MSAA - sample count 4
#[test]
fn test_msaa_sample_count_4() {
    let config = RenderPassBuilder::new("msaa4")
        .add_color_attachment(PassColorAttachment::new(ResourceId::new(1)))
        .sample_count(4)
        .build();

    assert_eq!(config.sample_count, 4);
    assert!(config.is_multisampled());
}

// Test 38: MSAA - sample count 8
#[test]
fn test_msaa_sample_count_8() {
    let config = RenderPassBuilder::new("msaa8")
        .add_color_attachment(PassColorAttachment::new(ResourceId::new(1)))
        .sample_count(8)
        .build();

    assert_eq!(config.sample_count, 8);
    assert!(config.validate().is_none());
}

// Test 39: MSAA - sample count 16
#[test]
fn test_msaa_sample_count_16() {
    let config = RenderPassBuilder::new("msaa16")
        .add_color_attachment(PassColorAttachment::new(ResourceId::new(1)))
        .sample_count(16)
        .build();

    assert_eq!(config.sample_count, 16);
}

// Test 40: Skybox pass - load color, no depth write
#[test]
fn test_skybox_pass() {
    let config = RenderPassBuilder::new("skybox")
        .add_color_attachment(PassColorAttachment::load(ResourceId::new(1)))
        .set_depth_attachment(PassDepthAttachment::read_only(ResourceId::new(2)))
        .build();

    assert!(config.depth_attachment.as_ref().unwrap().read_only);
}

// Test 41: Transparent pass - load color and depth, no depth write
#[test]
fn test_transparent_pass() {
    let config = RenderPassBuilder::new("transparent")
        .add_color_attachment(PassColorAttachment::load(ResourceId::new(1)))
        .set_depth_attachment(PassDepthAttachment::load(ResourceId::new(2)).make_read_only())
        .build();

    let depth = config.depth_attachment.as_ref().unwrap();
    assert!(depth.read_only);
    assert!(!depth.writes_depth());
}

// Test 42: UI overlay pass - load color, no depth
#[test]
fn test_ui_overlay_pass() {
    let config = RenderPassBuilder::new("ui_overlay")
        .add_color_attachment(PassColorAttachment::load(ResourceId::new(1)))
        .build();

    assert!(!config.has_depth());
    assert_eq!(config.color_attachments[0].load_op, PassLoadOp::Load);
}

// Test 43: Depth prepass - depth only, store
#[test]
fn test_depth_prepass() {
    let config = RenderPassBuilder::new("depth_prepass")
        .set_depth_attachment(PassDepthAttachment::new(ResourceId::new(1)))
        .build();

    let depth = config.depth_attachment.as_ref().unwrap();
    assert!(depth.writes_depth());
    assert_eq!(depth.depth_store_op, PassStoreOp::Store);
}

// Test 44: Stencil write pass
#[test]
fn test_stencil_write_pass() {
    let depth = PassDepthAttachment::with_stencil(ResourceId::new(1))
        .with_clear_stencil(0);
    let config = RenderPassBuilder::new("stencil_write")
        .add_color_attachment(PassColorAttachment::new(ResourceId::new(2)))
        .set_depth_attachment(depth)
        .build();

    let d = config.depth_attachment.as_ref().unwrap();
    assert!(d.writes_stencil());
    assert_eq!(d.clear_stencil, 0);
}

// Test 45: Full deferred pipeline - 4 MRT + depth
#[test]
fn test_full_deferred_mrt() {
    let config = RenderPassBuilder::new("deferred_mrt")
        .add_color_attachment(PassColorAttachment::new(ResourceId::new(1))) // albedo
        .add_color_attachment(PassColorAttachment::new(ResourceId::new(2))) // normal
        .add_color_attachment(PassColorAttachment::new(ResourceId::new(3))) // material
        .add_color_attachment(PassColorAttachment::new(ResourceId::new(4))) // emission
        .set_depth_attachment(PassDepthAttachment::new(ResourceId::new(5)))
        .build();

    assert_eq!(config.color_attachment_count(), 4);
    assert!(config.has_depth());
}

// Test 46: HDR to LDR tonemap pass
#[test]
fn test_hdr_tonemap_pass() {
    let config = RenderPassBuilder::new("tonemap")
        .add_color_attachment(PassColorAttachment::new(ResourceId::new(1)))
        .read_resource(ResourceId::new(2)) // HDR input
        .build();

    assert!(config.has_color());
}

// Test 47: Bloom downsample pass
#[test]
fn test_bloom_downsample() {
    let config = RenderPassBuilder::new("bloom_down")
        .add_color_attachment(PassColorAttachment::new(ResourceId::new(1)))
        .read_resource(ResourceId::new(2))
        .viewport(PassViewport::with_size(960.0, 540.0))
        .build();

    assert!(config.viewport.is_some());
    let vp = config.viewport.as_ref().unwrap();
    assert_eq!(vp.width, 960.0);
    assert_eq!(vp.height, 540.0);
}

// Test 48: Bloom upsample pass
#[test]
fn test_bloom_upsample() {
    let config = RenderPassBuilder::new("bloom_up")
        .add_color_attachment(PassColorAttachment::load(ResourceId::new(1)))
        .read_resource(ResourceId::new(2))
        .viewport(PassViewport::with_size(1920.0, 1080.0))
        .build();

    assert!(config.color_attachments[0].load_op.is_load());
}

// Test 49: Ambient occlusion pass
#[test]
fn test_ao_pass() {
    let config = RenderPassBuilder::new("ssao")
        .add_color_attachment(PassColorAttachment::new(ResourceId::new(1)))
        .read_resource(ResourceId::new(2)) // depth
        .read_resource(ResourceId::new(3)) // normal
        .build();

    let (_, reads, _) = RenderPassBuilder::new("ssao_check")
        .add_color_attachment(PassColorAttachment::new(ResourceId::new(1)))
        .read_resource(ResourceId::new(2))
        .read_resource(ResourceId::new(3))
        .build_with_deps();

    assert!(reads.contains(&ResourceId::new(2)));
    assert!(reads.contains(&ResourceId::new(3)));
}

// Test 50: Motion vectors pass
#[test]
fn test_motion_vectors_pass() {
    let config = RenderPassBuilder::new("motion_vectors")
        .add_color_attachment(PassColorAttachment::new(ResourceId::new(1)))
        .set_depth_attachment(PassDepthAttachment::read_only(ResourceId::new(2)))
        .build();

    assert!(config.depth_attachment.as_ref().unwrap().read_only);
}

// Test 51: Deferred lighting pass - read g-buffer
#[test]
fn test_deferred_lighting() {
    let (config, reads, writes) = RenderPassBuilder::new("deferred_light")
        .add_color_attachment(PassColorAttachment::new(ResourceId::new(10)))
        .read_resource(ResourceId::new(1)) // albedo
        .read_resource(ResourceId::new(2)) // normal
        .read_resource(ResourceId::new(3)) // material
        .read_resource(ResourceId::new(4)) // depth
        .build_with_deps();

    assert_eq!(reads.len(), 4);
    assert!(writes.contains(&ResourceId::new(10)));
}

// Test 52: TAA resolve pass
#[test]
fn test_taa_resolve() {
    let config = RenderPassBuilder::new("taa_resolve")
        .add_color_attachment(PassColorAttachment::new(ResourceId::new(1)))
        .read_resource(ResourceId::new(2)) // current frame
        .read_resource(ResourceId::new(3)) // history
        .read_resource(ResourceId::new(4)) // motion
        .build();

    assert!(config.has_color());
}

// Test 53: Cascaded shadow map - multiple passes
#[test]
fn test_csm_cascade_0() {
    let config = RenderPassBuilder::new("csm_cascade_0")
        .set_depth_attachment(PassDepthAttachment::new(ResourceId::new(100)))
        .viewport(PassViewport::with_size(2048.0, 2048.0))
        .build();

    assert!(!config.has_color());
    assert!(config.has_depth());
}

// ===========================================================================
// CATEGORY 3: BUILDER WORKFLOWS (25+)
// ===========================================================================

// Test 54: Minimal pass construction
#[test]
fn test_builder_minimal() {
    let config = RenderPassBuilder::new("minimal")
        .add_color_attachment(PassColorAttachment::new(ResourceId::new(1)))
        .build();

    assert_eq!(config.name, "minimal");
    assert_eq!(config.sample_count, 1);
    assert!(config.viewport.is_none());
}

// Test 55: Full pass construction
#[test]
fn test_builder_full() {
    let config = RenderPassBuilder::new("full_pass")
        .add_color_attachment(PassColorAttachment::clear(ResourceId::new(1), [0.0, 0.0, 0.0, 1.0]))
        .set_depth_attachment(PassDepthAttachment::new(ResourceId::new(2)))
        .sample_count(4)
        .viewport(PassViewport::new(0.0, 0.0, 1920.0, 1080.0))
        .build();

    assert!(config.has_color());
    assert!(config.has_depth());
    assert!(config.is_multisampled());
    assert!(config.viewport.is_some());
}

// Test 56: Chained method calls
#[test]
fn test_builder_chained_calls() {
    let config = RenderPassBuilder::new("chained")
        .add_color_attachment(PassColorAttachment::new(ResourceId::new(1)))
        .add_color_attachment(PassColorAttachment::new(ResourceId::new(2)))
        .add_color_attachment(PassColorAttachment::new(ResourceId::new(3)))
        .set_depth_attachment(PassDepthAttachment::new(ResourceId::new(4)))
        .sample_count(2)
        .viewport(PassViewport::default())
        .build();

    assert_eq!(config.color_attachment_count(), 3);
}

// Test 57: Multiple color attachments via builder
#[test]
fn test_builder_multiple_colors() {
    let builder = RenderPassBuilder::new("multi_color");
    let config = builder
        .add_color_attachment(PassColorAttachment::new(ResourceId::new(1)))
        .add_color_attachment(PassColorAttachment::new(ResourceId::new(2)))
        .build();

    assert_eq!(config.color_attachment_count(), 2);
}

// Test 58: Read resource tracking
#[test]
fn test_builder_read_resource_tracking() {
    let builder = RenderPassBuilder::new("read_track")
        .add_color_attachment(PassColorAttachment::new(ResourceId::new(1)))
        .read_resource(ResourceId::new(10))
        .read_resource(ResourceId::new(11));

    let reads = builder.get_reads();
    assert!(reads.contains(&ResourceId::new(10)));
    assert!(reads.contains(&ResourceId::new(11)));
}

// Test 59: Write resource tracking
#[test]
fn test_builder_write_resource_tracking() {
    let builder = RenderPassBuilder::new("write_track")
        .add_color_attachment(PassColorAttachment::new(ResourceId::new(1)))
        .write_resource(ResourceId::new(20))
        .write_resource(ResourceId::new(21));

    let writes = builder.get_writes();
    assert!(writes.contains(&ResourceId::new(20)));
    assert!(writes.contains(&ResourceId::new(21)));
}

// Test 60: Viewport configuration
#[test]
fn test_builder_viewport_config() {
    let config = RenderPassBuilder::new("viewport_test")
        .add_color_attachment(PassColorAttachment::new(ResourceId::new(1)))
        .viewport(PassViewport::new(100.0, 50.0, 800.0, 600.0).with_depth_range(0.1, 0.9))
        .build();

    let vp = config.viewport.as_ref().unwrap();
    assert_eq!(vp.x, 100.0);
    assert_eq!(vp.y, 50.0);
    assert_eq!(vp.min_depth, 0.1);
    assert_eq!(vp.max_depth, 0.9);
}

// Test 61: Sample count configuration
#[test]
fn test_builder_sample_count_config() {
    for count in [1, 2, 4, 8, 16] {
        let config = RenderPassBuilder::new("sample_test")
            .add_color_attachment(PassColorAttachment::new(ResourceId::new(1)))
            .sample_count(count)
            .build();

        assert_eq!(config.sample_count, count);
        assert!(config.validate().is_none());
    }
}

// Test 62: build_with_deps returns config and dependencies
#[test]
fn test_builder_build_with_deps() {
    let (config, reads, writes) = RenderPassBuilder::new("deps_test")
        .add_color_attachment(PassColorAttachment::load(ResourceId::new(1)))
        .set_depth_attachment(PassDepthAttachment::new(ResourceId::new(2)))
        .read_resource(ResourceId::new(10))
        .write_resource(ResourceId::new(20))
        .build_with_deps();

    assert_eq!(config.name, "deps_test");
    assert!(reads.contains(&ResourceId::new(1))); // Load implies read
    assert!(reads.contains(&ResourceId::new(10)));
    assert!(writes.contains(&ResourceId::new(2))); // Store implies write
    assert!(writes.contains(&ResourceId::new(20)));
}

// Test 63: Automatic read tracking from LoadOp::Load
#[test]
fn test_builder_auto_read_from_load() {
    let (_, reads, _) = RenderPassBuilder::new("auto_read")
        .add_color_attachment(PassColorAttachment::load(ResourceId::new(5)))
        .build_with_deps();

    assert!(reads.contains(&ResourceId::new(5)));
}

// Test 64: Automatic write tracking from StoreOp::Store
#[test]
fn test_builder_auto_write_from_store() {
    let (_, _, writes) = RenderPassBuilder::new("auto_write")
        .add_color_attachment(PassColorAttachment::new(ResourceId::new(5)))
        .build_with_deps();

    assert!(writes.contains(&ResourceId::new(5)));
}

// Test 65: Resolve target auto-tracked as write
#[test]
fn test_builder_resolve_target_tracking() {
    let (_, _, writes) = RenderPassBuilder::new("resolve_track")
        .add_color_attachment(PassColorAttachment::new(ResourceId::new(1)).with_resolve(ResourceId::new(2)))
        .build_with_deps();

    assert!(writes.contains(&ResourceId::new(1)));
    assert!(writes.contains(&ResourceId::new(2)));
}

// Test 66: Depth read tracking from read_only
#[test]
fn test_builder_depth_read_only_tracking() {
    let (_, reads, writes) = RenderPassBuilder::new("depth_ro")
        .add_color_attachment(PassColorAttachment::new(ResourceId::new(1)))
        .set_depth_attachment(PassDepthAttachment::read_only(ResourceId::new(10)))
        .build_with_deps();

    assert!(reads.contains(&ResourceId::new(10)));
    assert!(!writes.contains(&ResourceId::new(10)));
}

// Test 67: Depth write tracking from write
#[test]
fn test_builder_depth_write_tracking() {
    let (_, _, writes) = RenderPassBuilder::new("depth_wr")
        .add_color_attachment(PassColorAttachment::new(ResourceId::new(1)))
        .set_depth_attachment(PassDepthAttachment::new(ResourceId::new(10)))
        .build_with_deps();

    assert!(writes.contains(&ResourceId::new(10)));
}

// Test 68: No duplicate read entries
#[test]
fn test_builder_no_duplicate_reads() {
    let (_, reads, _) = RenderPassBuilder::new("no_dup")
        .add_color_attachment(PassColorAttachment::new(ResourceId::new(1)))
        .read_resource(ResourceId::new(10))
        .read_resource(ResourceId::new(10)) // duplicate
        .build_with_deps();

    let count = reads.iter().filter(|&&r| r == ResourceId::new(10)).count();
    assert_eq!(count, 1);
}

// Test 69: No duplicate write entries
#[test]
fn test_builder_no_duplicate_writes() {
    let (_, _, writes) = RenderPassBuilder::new("no_dup_w")
        .add_color_attachment(PassColorAttachment::new(ResourceId::new(1)))
        .write_resource(ResourceId::new(20))
        .write_resource(ResourceId::new(20))
        .build_with_deps();

    let count = writes.iter().filter(|&&w| w == ResourceId::new(20)).count();
    assert_eq!(count, 1);
}

// Test 70: Builder can be cloned
#[test]
fn test_builder_clone() {
    let builder1 = RenderPassBuilder::new("clone_test")
        .add_color_attachment(PassColorAttachment::new(ResourceId::new(1)));

    let builder2 = builder1.clone();
    let config2 = builder2.build();

    assert_eq!(config2.name, "clone_test");
    assert_eq!(config2.color_attachment_count(), 1);
}

// Test 71: Builder Default trait
#[test]
fn test_builder_default() {
    let builder = RenderPassBuilder::default();
    let config = builder.build();

    // Default name is "unnamed_pass" from config default
    assert!(config.color_attachments.is_empty());
}

// Test 72: Color attachment with_clear_color chain
#[test]
fn test_color_attachment_with_clear_color() {
    let att = PassColorAttachment::new(ResourceId::new(1))
        .with_clear_color([1.0, 0.0, 0.0, 1.0]);

    assert_eq!(att.clear_color, Some([1.0, 0.0, 0.0, 1.0]));
    assert!(att.load_op.is_clear());
}

// Test 73: Depth attachment with_clear_depth chain
#[test]
fn test_depth_attachment_with_clear_depth() {
    let att = PassDepthAttachment::new(ResourceId::new(1))
        .with_clear_depth(0.5);

    assert_eq!(att.clear_depth, 0.5);
    assert!(att.depth_load_op.is_clear());
}

// Test 74: Depth attachment with_clear_stencil chain
#[test]
fn test_depth_attachment_with_clear_stencil() {
    let att = PassDepthAttachment::with_stencil(ResourceId::new(1))
        .with_clear_stencil(255);

    assert_eq!(att.clear_stencil, 255);
    assert!(att.stencil_load_op.is_clear());
}

// Test 75: Depth attachment make_read_only chain
#[test]
fn test_depth_attachment_make_read_only() {
    let att = PassDepthAttachment::new(ResourceId::new(1))
        .make_read_only();

    assert!(att.read_only);
}

// Test 76: Viewport with_depth_range chain
#[test]
fn test_viewport_depth_range_chain() {
    let vp = PassViewport::default()
        .with_depth_range(0.25, 0.75);

    assert_eq!(vp.min_depth, 0.25);
    assert_eq!(vp.max_depth, 0.75);
}

// Test 77: Multiple chained operations on builder
#[test]
fn test_builder_extensive_chaining() {
    let (config, reads, writes) = RenderPassBuilder::new("extensive")
        .add_color_attachment(
            PassColorAttachment::new(ResourceId::new(1))
                .with_clear_color([0.1, 0.2, 0.3, 1.0])
                .with_resolve(ResourceId::new(2)),
        )
        .add_color_attachment(PassColorAttachment::load(ResourceId::new(3)))
        .set_depth_attachment(
            PassDepthAttachment::with_stencil(ResourceId::new(4))
                .with_clear_depth(1.0)
                .with_clear_stencil(128),
        )
        .sample_count(4)
        .viewport(PassViewport::with_size(1920.0, 1080.0).with_depth_range(0.0, 1.0))
        .read_resource(ResourceId::new(100))
        .write_resource(ResourceId::new(101))
        .build_with_deps();

    assert_eq!(config.color_attachment_count(), 2);
    assert!(config.has_depth());
    assert!(config.is_multisampled());
    assert!(config.viewport.is_some());
    assert!(reads.contains(&ResourceId::new(3))); // load
    assert!(reads.contains(&ResourceId::new(100)));
    assert!(writes.contains(&ResourceId::new(1)));
    assert!(writes.contains(&ResourceId::new(2))); // resolve
    assert!(writes.contains(&ResourceId::new(4)));
    assert!(writes.contains(&ResourceId::new(101)));
}

// Test 78: Builder preserves insertion order
#[test]
fn test_builder_preserves_order() {
    let config = RenderPassBuilder::new("order_test")
        .add_color_attachment(PassColorAttachment::new(ResourceId::new(1)))
        .add_color_attachment(PassColorAttachment::new(ResourceId::new(2)))
        .add_color_attachment(PassColorAttachment::new(ResourceId::new(3)))
        .build();

    assert_eq!(config.color_attachments[0].resource, ResourceId::new(1));
    assert_eq!(config.color_attachments[1].resource, ResourceId::new(2));
    assert_eq!(config.color_attachments[2].resource, ResourceId::new(3));
}

// ===========================================================================
// CATEGORY 4: INTEGRATION WITH FRAMEGRAPH (15+)
// ===========================================================================

// Test 79: RenderPassConfig written_resources accuracy
#[test]
fn test_config_written_resources() {
    let color = PassColorAttachment::new(ResourceId::new(1));
    let depth = PassDepthAttachment::new(ResourceId::new(2));
    let config = RenderPassConfig::with_color_and_depth("write_test", color, depth);

    let writes = config.written_resources();
    assert!(writes.contains(&ResourceId::new(1)));
    assert!(writes.contains(&ResourceId::new(2)));
}

// Test 80: RenderPassConfig written_resources with resolve
#[test]
fn test_config_written_resources_with_resolve() {
    let color = PassColorAttachment::new(ResourceId::new(1))
        .with_resolve(ResourceId::new(3));
    let config = RenderPassConfig::with_color("resolve_write", color);

    let writes = config.written_resources();
    assert!(writes.contains(&ResourceId::new(1)));
    assert!(writes.contains(&ResourceId::new(3)));
}

// Test 81: RenderPassConfig written_resources excludes discard
#[test]
fn test_config_written_resources_excludes_discard() {
    let color = PassColorAttachment::transient(ResourceId::new(1));
    let config = RenderPassConfig::with_color("discard_test", color);

    let writes = config.written_resources();
    assert!(!writes.contains(&ResourceId::new(1)));
}

// Test 82: RenderPassConfig read_resources accuracy
#[test]
fn test_config_read_resources() {
    let color = PassColorAttachment::load(ResourceId::new(1));
    let depth = PassDepthAttachment::read_only(ResourceId::new(2));
    let config = RenderPassConfig::with_color_and_depth("read_test", color, depth);

    let reads = config.read_resources();
    assert!(reads.contains(&ResourceId::new(1)));
    assert!(reads.contains(&ResourceId::new(2)));
}

// Test 83: RenderPassConfig read_resources excludes clear
#[test]
fn test_config_read_resources_excludes_clear() {
    let color = PassColorAttachment::clear(ResourceId::new(1), [0.0; 4]);
    let config = RenderPassConfig::with_color("clear_no_read", color);

    let reads = config.read_resources();
    assert!(!reads.contains(&ResourceId::new(1)));
}

// Test 84: RenderPassNode written_resources delegates to config
#[test]
fn test_node_written_resources() {
    let config = RenderPassConfig::with_color("node_write", PassColorAttachment::new(ResourceId::new(10)));
    let node = RenderPassNode::empty(PassId::new(1), config);

    let writes = node.written_resources();
    assert!(writes.contains(&ResourceId::new(10)));
}

// Test 85: RenderPassNode read_resources delegates to config
#[test]
fn test_node_read_resources() {
    let config = RenderPassConfig::with_color("node_read", PassColorAttachment::load(ResourceId::new(20)));
    let node = RenderPassNode::empty(PassId::new(2), config);

    let reads = node.read_resources();
    assert!(reads.contains(&ResourceId::new(20)));
}

// Test 86: RenderPassNode name accessor
#[test]
fn test_node_name_accessor() {
    let config = RenderPassConfig::with_color("accessor_test", PassColorAttachment::new(ResourceId::new(1)));
    let node = RenderPassNode::empty(PassId::new(3), config);

    assert_eq!(node.name(), "accessor_test");
}

// Test 87: RenderPassNode with function executor
#[test]
fn test_node_with_fn_executor() {
    let config = RenderPassConfig::with_color("fn_exec", PassColorAttachment::new(ResourceId::new(1)));
    let executed = std::sync::atomic::AtomicBool::new(false);

    let node = RenderPassNode::with_fn(PassId::new(4), config, |_ctx, _pass| {
        // This would execute during actual rendering
    });

    assert_eq!(node.executor.name(), "FnExecutor");
}

// Test 88: PassColorAttachment referenced_resources
#[test]
fn test_color_attachment_referenced_resources() {
    let att = PassColorAttachment::new(ResourceId::new(1))
        .with_resolve(ResourceId::new(2));

    let refs = att.referenced_resources();
    assert_eq!(refs.len(), 2);
    assert!(refs.contains(&ResourceId::new(1)));
    assert!(refs.contains(&ResourceId::new(2)));
}

// Test 89: PassColorAttachment referenced_resources without resolve
#[test]
fn test_color_attachment_referenced_resources_no_resolve() {
    let att = PassColorAttachment::new(ResourceId::new(5));

    let refs = att.referenced_resources();
    assert_eq!(refs.len(), 1);
    assert_eq!(refs[0], ResourceId::new(5));
}

// Test 90: PassDepthAttachment writes_depth check
#[test]
fn test_depth_writes_depth_check() {
    // Normal depth attachment writes depth
    let att1 = PassDepthAttachment::new(ResourceId::new(1));
    assert!(att1.writes_depth());

    // Read-only does not write
    let att2 = PassDepthAttachment::read_only(ResourceId::new(2));
    assert!(!att2.writes_depth());

    // Discard does not write
    let att3 = PassDepthAttachment {
        resource: ResourceId::new(3),
        depth_load_op: PassLoadOp::Clear,
        depth_store_op: PassStoreOp::Discard,
        stencil_load_op: PassLoadOp::DontCare,
        stencil_store_op: PassStoreOp::Discard,
        clear_depth: 1.0,
        clear_stencil: 0,
        read_only: false,
    };
    assert!(!att3.writes_depth());
}

// Test 91: PassDepthAttachment writes_stencil check
#[test]
fn test_depth_writes_stencil_check() {
    // Default does not write stencil (discard)
    let att1 = PassDepthAttachment::new(ResourceId::new(1));
    assert!(!att1.writes_stencil());

    // With stencil writes stencil
    let att2 = PassDepthAttachment::with_stencil(ResourceId::new(2));
    assert!(att2.writes_stencil());

    // Read-only does not write even with store
    let att3 = PassDepthAttachment::with_stencil(ResourceId::new(3)).make_read_only();
    assert!(!att3.writes_stencil());
}

// Test 92: RenderPassConfig validate returns None for valid
#[test]
fn test_config_validate_valid() {
    let config = RenderPassConfig::with_color("valid", PassColorAttachment::new(ResourceId::new(1)));
    assert!(config.validate().is_none());
}

// Test 93: Complex multi-attachment dependency tracking
#[test]
fn test_complex_dependency_tracking() {
    let (config, reads, writes) = RenderPassBuilder::new("complex_deps")
        // Load from previous pass
        .add_color_attachment(PassColorAttachment::load(ResourceId::new(1)))
        // Clear new output
        .add_color_attachment(PassColorAttachment::new(ResourceId::new(2)))
        // Transient - no read/write tracking
        .add_color_attachment(PassColorAttachment::transient(ResourceId::new(3)))
        // Read-only depth
        .set_depth_attachment(PassDepthAttachment::read_only(ResourceId::new(10)))
        // Explicit reads
        .read_resource(ResourceId::new(100))
        .read_resource(ResourceId::new(101))
        .build_with_deps();

    // Reads: load color (1), read-only depth (10), explicit (100, 101)
    assert!(reads.contains(&ResourceId::new(1)));
    assert!(reads.contains(&ResourceId::new(10)));
    assert!(reads.contains(&ResourceId::new(100)));
    assert!(reads.contains(&ResourceId::new(101)));

    // Writes: store colors (1, 2) - but 3 is discarded
    // Note: color 1 is both read (load) and written (store)
    assert!(writes.contains(&ResourceId::new(1)));
    assert!(writes.contains(&ResourceId::new(2)));
}

// ===========================================================================
// CATEGORY 5: EDGE CASES (15+)
// ===========================================================================

// Test 94: Empty name
#[test]
fn test_empty_name() {
    let config = RenderPassConfig::with_color("", PassColorAttachment::new(ResourceId::new(1)));
    assert_eq!(config.name, "");
    assert!(config.validate().is_none()); // Empty name is technically valid
}

// Test 95: Very long name
#[test]
fn test_long_name() {
    let long_name = "a".repeat(1000);
    let config = RenderPassConfig::with_color(long_name.clone(), PassColorAttachment::new(ResourceId::new(1)));
    assert_eq!(config.name.len(), 1000);
}

// Test 96: Zero sample count validation
#[test]
fn test_zero_sample_count_invalid() {
    let mut config = RenderPassConfig::with_color("zero_sample", PassColorAttachment::new(ResourceId::new(1)));
    config.sample_count = 0;

    let error = config.validate();
    assert!(error.is_some());
    assert!(error.unwrap().contains("Invalid sample count"));
}

// Test 97: Invalid sample count (3) validation
#[test]
fn test_invalid_sample_count_3() {
    let mut config = RenderPassConfig::with_color("sample_3", PassColorAttachment::new(ResourceId::new(1)));
    config.sample_count = 3;

    assert!(config.validate().is_some());
}

// Test 98: Invalid sample count (32) validation
#[test]
fn test_invalid_sample_count_32() {
    let mut config = RenderPassConfig::with_color("sample_32", PassColorAttachment::new(ResourceId::new(1)));
    config.sample_count = 32;

    assert!(config.validate().is_some());
}

// Test 99: Max color attachments (8) is valid
#[test]
fn test_max_color_attachments_valid() {
    let mut config = RenderPassConfig::new("max_colors");
    for i in 0..8 {
        config.color_attachments.push(PassColorAttachment::new(ResourceId::new(i)));
    }

    assert_eq!(config.color_attachment_count(), 8);
    assert!(config.validate().is_none());
}

// Test 100: Too many color attachments (9) is invalid
#[test]
fn test_too_many_color_attachments_invalid() {
    let mut config = RenderPassConfig::new("too_many");
    for i in 0..9 {
        config.color_attachments.push(PassColorAttachment::new(ResourceId::new(i)));
    }

    let error = config.validate();
    assert!(error.is_some());
    assert!(error.unwrap().contains("Too many color attachments"));
}

// Test 101: Depth-only pass is valid
#[test]
fn test_depth_only_pass_valid() {
    let mut config = RenderPassConfig::new("depth_only");
    config.depth_attachment = Some(PassDepthAttachment::new(ResourceId::new(1)));

    assert!(!config.has_color());
    assert!(config.has_depth());
    assert!(config.validate().is_none());
}

// Test 102: No attachments is invalid
#[test]
fn test_no_attachments_invalid() {
    let config = RenderPassConfig::new("no_attachments");

    let error = config.validate();
    assert!(error.is_some());
    assert!(error.unwrap().contains("at least one attachment"));
}

// Test 103: Invalid viewport (zero width)
#[test]
fn test_invalid_viewport_zero_width() {
    let vp = PassViewport::new(0.0, 0.0, 0.0, 100.0);
    assert!(!vp.is_valid());

    let mut config = RenderPassConfig::with_color("vp_zero_w", PassColorAttachment::new(ResourceId::new(1)));
    config.viewport = Some(vp);

    assert!(config.validate().is_some());
}

// Test 104: Invalid viewport (zero height)
#[test]
fn test_invalid_viewport_zero_height() {
    let vp = PassViewport::new(0.0, 0.0, 100.0, 0.0);
    assert!(!vp.is_valid());
}

// Test 105: Invalid viewport (inverted depth range)
#[test]
fn test_invalid_viewport_inverted_depth() {
    let vp = PassViewport::new(0.0, 0.0, 100.0, 100.0).with_depth_range(1.0, 0.0);
    assert!(!vp.is_valid());
}

// Test 106: Viewport aspect ratio calculation
#[test]
fn test_viewport_aspect_ratio() {
    let vp = PassViewport::with_size(1920.0, 1080.0);
    let aspect = vp.aspect_ratio();
    assert!((aspect - 16.0 / 9.0).abs() < 0.01);
}

// Test 107: Viewport aspect ratio with zero height
#[test]
fn test_viewport_aspect_ratio_zero_height() {
    let vp = PassViewport {
        x: 0.0,
        y: 0.0,
        width: 100.0,
        height: 0.0,
        min_depth: 0.0,
        max_depth: 1.0,
    };

    // Should return 1.0 for zero height to avoid division by zero
    assert_eq!(vp.aspect_ratio(), 1.0);
}

// Test 108: ResourceId edge cases
#[test]
fn test_resource_id_edge_cases() {
    let r0 = ResourceId::new(0);
    let r_max = ResourceId::new(u64::MAX - 1);
    let r_invalid = ResourceId::INVALID;

    assert_eq!(r0.raw(), 0);
    assert!(!r0.is_invalid());
    assert!(r_invalid.is_invalid());

    // Can use max-1 as valid
    let config = RenderPassConfig::with_color("edge", PassColorAttachment::new(r_max));
    assert!(config.validate().is_none());
}

// Test 109: PassId edge cases
#[test]
fn test_pass_id_edge_cases() {
    let p0 = PassId::new(0);
    let p_max = PassId::new(u64::MAX - 1);
    let p_invalid = PassId::INVALID;

    assert_eq!(p0.raw(), 0);
    assert!(!p0.is_invalid());
    assert!(p_invalid.is_invalid());
}

// Test 110: Display trait for PassLoadOp
#[test]
fn test_display_pass_load_op() {
    assert_eq!(format!("{}", PassLoadOp::Clear), "Clear");
    assert_eq!(format!("{}", PassLoadOp::Load), "Load");
    assert_eq!(format!("{}", PassLoadOp::DontCare), "DontCare");
}

// Test 111: Display trait for PassStoreOp
#[test]
fn test_display_pass_store_op() {
    assert_eq!(format!("{}", PassStoreOp::Store), "Store");
    assert_eq!(format!("{}", PassStoreOp::Discard), "Discard");
}

// Test 112: Display trait for PassViewport
#[test]
fn test_display_pass_viewport() {
    let vp = PassViewport::new(10.0, 20.0, 800.0, 600.0);
    let s = format!("{}", vp);
    assert!(s.contains("10"));
    assert!(s.contains("20"));
    assert!(s.contains("800"));
    assert!(s.contains("600"));
}

// Test 113: Display trait for PassColorAttachment
#[test]
fn test_display_pass_color_attachment() {
    let att = PassColorAttachment::new(ResourceId::new(42));
    let s = format!("{}", att);
    assert!(s.contains("42") || s.contains("ColorAttachment"));
}

// Test 114: Display trait for PassDepthAttachment
#[test]
fn test_display_pass_depth_attachment() {
    let att = PassDepthAttachment::new(ResourceId::new(99));
    let s = format!("{}", att);
    assert!(s.contains("99") || s.contains("DepthAttachment"));
}

// Test 115: Display trait for RenderPassConfig
#[test]
fn test_display_render_pass_config() {
    let config = RenderPassConfig::with_color("display_test", PassColorAttachment::new(ResourceId::new(1)));
    let s = format!("{}", config);
    assert!(s.contains("display_test"));
}

// Test 116: Debug trait for RenderPassNode
#[test]
fn test_debug_render_pass_node() {
    let config = RenderPassConfig::with_color("debug_test", PassColorAttachment::new(ResourceId::new(1)));
    let node = RenderPassNode::empty(PassId::new(5), config);
    let s = format!("{:?}", node);
    assert!(s.contains("RenderPassNode"));
    assert!(s.contains("debug_test"));
}

// Test 117: Display trait for RenderPassNode
#[test]
fn test_display_render_pass_node() {
    let config = RenderPassConfig::with_color("node_display", PassColorAttachment::new(ResourceId::new(1)));
    let node = RenderPassNode::empty(PassId::new(6), config);
    let s = format!("{}", node);
    assert!(s.contains("node_display") || s.contains("RenderPassNode"));
}

// Test 118: FnExecutor with custom name
#[test]
fn test_fn_executor_named() {
    let executor = FnExecutor::named("custom_executor", |_ctx: &mut RenderContext, _pass: &mut wgpu::RenderPass| {
        // execution logic
    });
    assert_eq!(executor.name(), "custom_executor");
}

// Test 119: Multiple passes sharing same resource
#[test]
fn test_multiple_passes_same_resource() {
    let shared_color = ResourceId::new(100);

    let config1 = RenderPassBuilder::new("pass1")
        .add_color_attachment(PassColorAttachment::new(shared_color))
        .build();

    let config2 = RenderPassBuilder::new("pass2")
        .add_color_attachment(PassColorAttachment::load(shared_color))
        .build();

    // Both reference the same resource
    assert_eq!(config1.color_attachments[0].resource, shared_color);
    assert_eq!(config2.color_attachments[0].resource, shared_color);

    // First clears, second loads
    assert!(config1.color_attachments[0].load_op.is_clear());
    assert!(config2.color_attachments[0].load_op.is_load());
}

// Test 120: Clone traits work correctly
#[test]
fn test_clone_traits() {
    let load_op = PassLoadOp::Clear;
    let load_op_clone = load_op.clone();
    assert_eq!(load_op, load_op_clone);

    let store_op = PassStoreOp::Store;
    let store_op_clone = store_op.clone();
    assert_eq!(store_op, store_op_clone);

    let viewport = PassViewport::default();
    let viewport_clone = viewport.clone();
    assert_eq!(viewport, viewport_clone);

    let color_att = PassColorAttachment::new(ResourceId::new(1));
    let color_att_clone = color_att.clone();
    assert_eq!(color_att, color_att_clone);

    let depth_att = PassDepthAttachment::new(ResourceId::new(2));
    let depth_att_clone = depth_att.clone();
    assert_eq!(depth_att, depth_att_clone);

    let config = RenderPassConfig::with_color("clone", PassColorAttachment::new(ResourceId::new(3)));
    let config_clone = config.clone();
    assert_eq!(config, config_clone);
}

// Test 121: Copy traits work correctly
#[test]
fn test_copy_traits() {
    let load_op = PassLoadOp::Clear;
    let load_op_copy: PassLoadOp = load_op; // Copy
    assert_eq!(load_op, load_op_copy);

    let store_op = PassStoreOp::Store;
    let store_op_copy: PassStoreOp = store_op;
    assert_eq!(store_op, store_op_copy);

    let viewport = PassViewport::default();
    let viewport_copy: PassViewport = viewport;
    assert_eq!(viewport, viewport_copy);
}

// Test 122: PartialEq traits work correctly
#[test]
fn test_partial_eq_traits() {
    assert_eq!(PassLoadOp::Clear, PassLoadOp::Clear);
    assert_ne!(PassLoadOp::Clear, PassLoadOp::Load);

    assert_eq!(PassStoreOp::Store, PassStoreOp::Store);
    assert_ne!(PassStoreOp::Store, PassStoreOp::Discard);

    let vp1 = PassViewport::with_size(100.0, 100.0);
    let vp2 = PassViewport::with_size(100.0, 100.0);
    let vp3 = PassViewport::with_size(200.0, 100.0);
    assert_eq!(vp1, vp2);
    assert_ne!(vp1, vp3);
}

// Test 123: Hash traits work correctly
#[test]
fn test_hash_traits() {
    use std::collections::HashSet;

    let mut load_ops: HashSet<PassLoadOp> = HashSet::new();
    load_ops.insert(PassLoadOp::Clear);
    load_ops.insert(PassLoadOp::Load);
    load_ops.insert(PassLoadOp::DontCare);
    assert_eq!(load_ops.len(), 3);

    let mut store_ops: HashSet<PassStoreOp> = HashSet::new();
    store_ops.insert(PassStoreOp::Store);
    store_ops.insert(PassStoreOp::Discard);
    assert_eq!(store_ops.len(), 2);
}

// Test 124: wgpu StoreOp conversion
#[test]
fn test_wgpu_store_op_conversion() {
    assert_eq!(PassStoreOp::Store.to_wgpu(), wgpu::StoreOp::Store);
    assert_eq!(PassStoreOp::Discard.to_wgpu(), wgpu::StoreOp::Discard);
}

// Test 125: wgpu color LoadOp conversion
#[test]
fn test_wgpu_color_load_op_conversion() {
    let clear_color = Some([1.0, 0.5, 0.25, 1.0]);

    match PassLoadOp::Clear.to_wgpu_color(clear_color) {
        wgpu::LoadOp::Clear(color) => {
            assert!((color.r - 1.0).abs() < 0.001);
            assert!((color.g - 0.5).abs() < 0.001);
            assert!((color.b - 0.25).abs() < 0.001);
            assert!((color.a - 1.0).abs() < 0.001);
        }
        _ => panic!("Expected Clear"),
    }

    match PassLoadOp::Load.to_wgpu_color(None) {
        wgpu::LoadOp::Load => {}
        _ => panic!("Expected Load"),
    }
}

// Test 126: wgpu depth LoadOp conversion
#[test]
fn test_wgpu_depth_load_op_conversion() {
    match PassLoadOp::Clear.to_wgpu_depth(1.0) {
        wgpu::LoadOp::Clear(depth) => {
            assert!((depth - 1.0).abs() < 0.001);
        }
        _ => panic!("Expected Clear"),
    }

    match PassLoadOp::Load.to_wgpu_depth(0.0) {
        wgpu::LoadOp::Load => {}
        _ => panic!("Expected Load"),
    }
}

// Test 127: wgpu stencil LoadOp conversion
#[test]
fn test_wgpu_stencil_load_op_conversion() {
    match PassLoadOp::Clear.to_wgpu_stencil(128) {
        wgpu::LoadOp::Clear(stencil) => {
            assert_eq!(stencil, 128);
        }
        _ => panic!("Expected Clear"),
    }

    match PassLoadOp::Load.to_wgpu_stencil(0) {
        wgpu::LoadOp::Load => {}
        _ => panic!("Expected Load"),
    }
}

// Test 128: Default clear color is black with alpha 1
#[test]
fn test_default_clear_color() {
    let att = PassColorAttachment::new(ResourceId::new(1));
    assert_eq!(att.clear_color, Some([0.0, 0.0, 0.0, 1.0]));
}

// Test 129: Default clear depth is 1.0 (far plane)
#[test]
fn test_default_clear_depth() {
    let att = PassDepthAttachment::new(ResourceId::new(1));
    assert_eq!(att.clear_depth, 1.0);
}

// Test 130: Default clear stencil is 0
#[test]
fn test_default_clear_stencil() {
    let att = PassDepthAttachment::new(ResourceId::new(1));
    assert_eq!(att.clear_stencil, 0);
}

// Test 131: Config color_attachment_count helper
#[test]
fn test_config_color_attachment_count() {
    let mut config = RenderPassConfig::new("count_test");
    assert_eq!(config.color_attachment_count(), 0);

    config.color_attachments.push(PassColorAttachment::new(ResourceId::new(1)));
    assert_eq!(config.color_attachment_count(), 1);

    config.color_attachments.push(PassColorAttachment::new(ResourceId::new(2)));
    assert_eq!(config.color_attachment_count(), 2);
}

// Test 132: Config has_color helper
#[test]
fn test_config_has_color_helper() {
    let config_empty = RenderPassConfig::new("empty");
    assert!(!config_empty.has_color());

    let config_with = RenderPassConfig::with_color("with", PassColorAttachment::new(ResourceId::new(1)));
    assert!(config_with.has_color());
}

// Test 133: Config has_depth helper
#[test]
fn test_config_has_depth_helper() {
    let config_no = RenderPassConfig::with_color("no_depth", PassColorAttachment::new(ResourceId::new(1)));
    assert!(!config_no.has_depth());

    let config_with = RenderPassConfig::with_color_and_depth(
        "with_depth",
        PassColorAttachment::new(ResourceId::new(1)),
        PassDepthAttachment::new(ResourceId::new(2)),
    );
    assert!(config_with.has_depth());
}

// Test 134: Config is_multisampled helper
#[test]
fn test_config_is_multisampled_helper() {
    let config_1x = RenderPassBuilder::new("1x")
        .add_color_attachment(PassColorAttachment::new(ResourceId::new(1)))
        .sample_count(1)
        .build();
    assert!(!config_1x.is_multisampled());

    let config_4x = RenderPassBuilder::new("4x")
        .add_color_attachment(PassColorAttachment::new(ResourceId::new(1)))
        .sample_count(4)
        .build();
    assert!(config_4x.is_multisampled());
}

// Test 135: PassColorAttachment has_resolve helper
#[test]
fn test_color_attachment_has_resolve_helper() {
    let att_no = PassColorAttachment::new(ResourceId::new(1));
    assert!(!att_no.has_resolve());

    let att_yes = PassColorAttachment::new(ResourceId::new(1))
        .with_resolve(ResourceId::new(2));
    assert!(att_yes.has_resolve());
}
