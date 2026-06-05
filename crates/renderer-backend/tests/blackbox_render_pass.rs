// SPDX-License-Identifier: MIT
//
// blackbox_render_pass.rs -- Blackbox tests for T-WGPU-P3.8.1 Render Pass Creation.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only the documented public types and functions:
//
//   - RenderPassDescriptor -- High-level render pass descriptor
//   - RenderPassBuilder -- Fluent builder for render passes
//   - RenderPassColorAttachment -- Color attachment configuration
//   - RenderPassDepthStencilAttachment -- Depth/stencil attachment
//   - TimestampWrites -- GPU profiling timestamps
//   - OcclusionQuerySet -- Visibility query configuration
//   - LoadOp -- Load operation at pass start
//   - StoreOp -- Store operation at pass end
//   - Operations -- Combined load/store operations
//   - RenderPassInfo -- Preset metadata
//   - RenderPassError -- Error types
//
// PUBLIC FUNCTIONS:
//   - get_render_pass_preset_info, render_pass_preset_names
//   - validate_color_attachment_count, validate_render_pass_descriptor
//
// CONSTANTS:
//   - DEFAULT_CLEAR_COLOR, DEFAULT_CLEAR_DEPTH, DEFAULT_CLEAR_STENCIL
//   - RENDER_PASS_MAX_COLOR_ATTACHMENTS, RENDER_PASS_PRESETS
//
// ACCEPTANCE CRITERIA (T-WGPU-P3.8.1):
//   1. RenderPassDescriptor construction
//   2. Color attachments array (up to 8)
//   3. Depth/stencil attachment (optional)
//   4. Timestamp writes (optional)
//   5. Occlusion query set (optional)
//
// TEST CATEGORIES:
//   1. API Tests (10 tests) - Constructor, builder, defaults
//   2. Descriptor Tests (12 tests) - Fields, configuration
//   3. ColorAttachment Tests (10 tests) - Clear, load, resolve
//   4. DepthStencil Tests (10 tests) - Depth-only, stencil, combined
//   5. TimestampWrites Tests (8 tests) - Both, beginning, end
//   6. Builder Tests (10 tests) - Fluent API, presets
//   7. Operations Tests (8 tests) - LoadOp, StoreOp
//   8. Validation Tests (6 tests) - Error conditions
//   9. Real-world Scenarios (8 tests) - Forward, deferred, shadow
//   10. Thread Safety (4 tests) - Send + Sync
//
// Total target: 80+ tests

use renderer_backend::render_pipeline::{
    get_render_pass_preset_info, render_pass_preset_names,
    validate_color_attachment_count, validate_render_pass_descriptor,
    RenderPassColorAttachment, RenderPassDepthStencilAttachment,
    LoadOp, OcclusionQuerySet, Operations, RenderPassBuilder, RenderPassDescriptor,
    RenderPassError, RenderPassInfo, StoreOp, TimestampWrites,
    DEFAULT_CLEAR_COLOR, DEFAULT_CLEAR_DEPTH, DEFAULT_CLEAR_STENCIL,
    RENDER_PASS_MAX_COLOR_ATTACHMENTS, RENDER_PASS_PRESETS,
};

// =============================================================================
// CATEGORY 1: API TESTS - Public Interface (10 tests)
// =============================================================================

mod api_tests {
    use super::*;

    #[test]
    fn test_render_pass_descriptor_is_public() {
        // Verify RenderPassDescriptor struct is accessible
        let desc = RenderPassDescriptor::new();
        assert!(desc.label.is_none());
    }

    #[test]
    fn test_render_pass_builder_is_public() {
        // Verify RenderPassBuilder is accessible
        let builder = RenderPassBuilder::new();
        let desc = builder.build();
        assert!(desc.color_attachments.is_empty());
    }

    #[test]
    fn test_color_attachment_is_public() {
        // Verify RenderPassColorAttachment is accessible
        let att = RenderPassColorAttachment::new();
        assert_eq!(att.store_op, StoreOp::Store);
    }

    #[test]
    fn test_depth_stencil_attachment_is_public() {
        // Verify RenderPassDepthStencilAttachment is accessible
        let att = RenderPassDepthStencilAttachment::new();
        assert!(att.depth_ops.is_some());
    }

    #[test]
    fn test_timestamp_writes_is_public() {
        // Verify TimestampWrites is accessible
        let ts = TimestampWrites::new();
        assert!(ts.beginning_of_pass_write_index.is_none());
    }

    #[test]
    fn test_occlusion_query_set_is_public() {
        // Verify OcclusionQuerySet is accessible
        let oqs = OcclusionQuerySet::new();
        assert!(oqs.enabled);
    }

    #[test]
    fn test_load_op_is_public() {
        // Verify LoadOp enum is accessible
        let op: LoadOp<f32> = LoadOp::Clear(1.0);
        assert!(matches!(op, LoadOp::Clear(_)));
    }

    #[test]
    fn test_store_op_is_public() {
        // Verify StoreOp enum is accessible
        let op = StoreOp::Store;
        assert_eq!(op, StoreOp::Store);
    }

    #[test]
    fn test_operations_is_public() {
        // Verify Operations struct is accessible
        let ops = Operations::<f32>::clear(1.0);
        assert!(matches!(ops.load, LoadOp::Clear(_)));
    }

    #[test]
    fn test_render_pass_error_is_public() {
        // Verify RenderPassError enum is accessible
        let err = RenderPassError::NoAttachments;
        let msg = format!("{}", err);
        assert!(msg.contains("at least one"));
    }
}

// =============================================================================
// CATEGORY 2: DESCRIPTOR TESTS - Fields and Configuration (12 tests)
// =============================================================================

mod descriptor_tests {
    use super::*;

    #[test]
    fn test_descriptor_new_empty() {
        let desc = RenderPassDescriptor::new();
        assert!(desc.label.is_none());
        assert!(desc.color_attachments.is_empty());
        assert!(desc.depth_stencil_attachment.is_none());
        assert!(desc.timestamp_writes.is_none());
        assert!(!desc.occlusion_query_enabled);
    }

    #[test]
    fn test_descriptor_label() {
        let desc = RenderPassDescriptor::new().label("forward_pass");
        assert_eq!(desc.label, Some("forward_pass".to_string()));
    }

    #[test]
    fn test_descriptor_single_color_attachment() {
        let desc = RenderPassDescriptor::new()
            .color_attachment(RenderPassColorAttachment::new());
        assert_eq!(desc.color_attachments.len(), 1);
    }

    #[test]
    fn test_descriptor_multiple_color_attachments() {
        let desc = RenderPassDescriptor::new()
            .color_attachment(RenderPassColorAttachment::new())
            .color_attachment(RenderPassColorAttachment::new())
            .color_attachment(RenderPassColorAttachment::new());
        assert_eq!(desc.color_attachments.len(), 3);
    }

    #[test]
    fn test_descriptor_empty_color_slot() {
        let desc = RenderPassDescriptor::new()
            .color_attachment(RenderPassColorAttachment::new())
            .empty_color_slot()
            .color_attachment(RenderPassColorAttachment::new());
        assert_eq!(desc.color_attachments.len(), 3);
        assert!(desc.color_attachments[1].is_none());
    }

    #[test]
    fn test_descriptor_set_color_attachments() {
        let atts = vec![
            Some(RenderPassColorAttachment::new()),
            None,
            Some(RenderPassColorAttachment::new()),
        ];
        let desc = RenderPassDescriptor::new().color_attachments(atts);
        assert_eq!(desc.color_attachments.len(), 3);
    }

    #[test]
    fn test_descriptor_depth_stencil() {
        let desc = RenderPassDescriptor::new()
            .depth_stencil(RenderPassDepthStencilAttachment::new());
        assert!(desc.depth_stencil_attachment.is_some());
    }

    #[test]
    fn test_descriptor_timestamp_writes() {
        let desc = RenderPassDescriptor::new()
            .timestamp_writes(TimestampWrites::both(0, 1));
        assert!(desc.timestamp_writes.is_some());
    }

    #[test]
    fn test_descriptor_occlusion_queries() {
        let desc = RenderPassDescriptor::new().with_occlusion_queries();
        assert!(desc.occlusion_query_enabled);
    }

    #[test]
    fn test_descriptor_color_attachment_count() {
        let desc = RenderPassDescriptor::new()
            .color_attachment(RenderPassColorAttachment::new())
            .color_attachment(RenderPassColorAttachment::new());
        assert_eq!(desc.color_attachment_count(), 2);
    }

    #[test]
    fn test_descriptor_has_depth_stencil() {
        let desc_no = RenderPassDescriptor::new();
        assert!(!desc_no.has_depth_stencil());

        let desc_yes = RenderPassDescriptor::new()
            .depth_stencil(RenderPassDepthStencilAttachment::new());
        assert!(desc_yes.has_depth_stencil());
    }

    #[test]
    fn test_descriptor_display_format() {
        let desc = RenderPassDescriptor::new()
            .label("test")
            .color_attachment(RenderPassColorAttachment::new());
        let s = format!("{}", desc);
        assert!(s.contains("RenderPassDescriptor"));
    }
}

// =============================================================================
// CATEGORY 3: COLOR ATTACHMENT TESTS (10 tests)
// =============================================================================

mod color_attachment_tests {
    use super::*;

    #[test]
    fn test_color_attachment_new() {
        let att = RenderPassColorAttachment::new();
        assert_eq!(att.store_op, StoreOp::Store);
        assert!(!att.has_resolve);
    }

    #[test]
    fn test_color_attachment_default() {
        let att = RenderPassColorAttachment::default();
        assert_eq!(att.clear_color, DEFAULT_CLEAR_COLOR);
    }

    #[test]
    fn test_color_attachment_with_clear() {
        let att = RenderPassColorAttachment::with_clear(wgpu::Color::RED);
        if let LoadOp::Clear(color) = att.load_op {
            assert_eq!(color, wgpu::Color::RED);
        } else {
            panic!("Expected Clear");
        }
    }

    #[test]
    fn test_color_attachment_with_load() {
        let att = RenderPassColorAttachment::with_load();
        assert!(matches!(att.load_op, LoadOp::Load));
    }

    #[test]
    fn test_color_attachment_clear_color_builder() {
        let att = RenderPassColorAttachment::new()
            .clear_color(wgpu::Color::GREEN);
        assert_eq!(att.clear_color, wgpu::Color::GREEN);
    }

    #[test]
    fn test_color_attachment_load_builder() {
        let att = RenderPassColorAttachment::new().load();
        assert!(matches!(att.load_op, LoadOp::Load));
    }

    #[test]
    fn test_color_attachment_store_discard() {
        let att = RenderPassColorAttachment::new().store(StoreOp::Discard);
        assert_eq!(att.store_op, StoreOp::Discard);
    }

    #[test]
    fn test_color_attachment_with_resolve() {
        let att = RenderPassColorAttachment::new().with_resolve();
        assert!(att.has_resolve);
    }

    #[test]
    fn test_color_attachment_operations() {
        let att = RenderPassColorAttachment::with_clear(wgpu::Color::BLUE);
        let ops = att.operations();
        assert!(matches!(ops.load, LoadOp::Clear(_)));
        assert_eq!(ops.store, StoreOp::Store);
    }

    #[test]
    fn test_color_attachment_display() {
        let att = RenderPassColorAttachment::new();
        let s = format!("{}", att);
        assert!(s.contains("ColorAttachment"));
    }
}

// =============================================================================
// CATEGORY 4: DEPTH STENCIL TESTS (10 tests)
// =============================================================================

mod depth_stencil_tests {
    use super::*;

    #[test]
    fn test_depth_stencil_new() {
        let att = RenderPassDepthStencilAttachment::new();
        assert!(att.depth_ops.is_some());
        assert!(att.stencil_ops.is_none());
    }

    #[test]
    fn test_depth_stencil_depth_only() {
        let att = RenderPassDepthStencilAttachment::depth_only(0.5);
        assert!(att.depth_ops.is_some());
        assert!(att.stencil_ops.is_none());
        if let Some(ops) = att.depth_ops {
            if let LoadOp::Clear(depth) = ops.load {
                assert_eq!(depth, 0.5);
            }
        }
    }

    #[test]
    fn test_depth_stencil_depth_only_reverse_z() {
        let att = RenderPassDepthStencilAttachment::depth_only_reverse_z();
        if let Some(ops) = att.depth_ops {
            if let LoadOp::Clear(depth) = ops.load {
                assert_eq!(depth, 0.0);
            }
        }
    }

    #[test]
    fn test_depth_stencil_depth_read_only() {
        let att = RenderPassDepthStencilAttachment::depth_read_only();
        assert!(att.depth_ops.is_none());
        assert!(att.stencil_ops.is_none());
        assert!(att.is_read_only());
    }

    #[test]
    fn test_depth_stencil_combined() {
        let att = RenderPassDepthStencilAttachment::depth_stencil(1.0, 128);
        assert!(att.depth_ops.is_some());
        assert!(att.stencil_ops.is_some());
    }

    #[test]
    fn test_depth_stencil_stencil_only() {
        let att = RenderPassDepthStencilAttachment::stencil_only(255);
        assert!(att.depth_ops.is_none());
        assert!(att.stencil_ops.is_some());
    }

    #[test]
    fn test_depth_stencil_is_depth_writable() {
        let att = RenderPassDepthStencilAttachment::depth_only(1.0);
        assert!(att.is_depth_writable());

        let att_ro = RenderPassDepthStencilAttachment::depth_read_only();
        assert!(!att_ro.is_depth_writable());
    }

    #[test]
    fn test_depth_stencil_is_stencil_writable() {
        let att = RenderPassDepthStencilAttachment::depth_stencil(1.0, 0);
        assert!(att.is_stencil_writable());

        let att_no = RenderPassDepthStencilAttachment::depth_only(1.0);
        assert!(!att_no.is_stencil_writable());
    }

    #[test]
    fn test_depth_stencil_builder_chains() {
        let att = RenderPassDepthStencilAttachment::new()
            .depth_clear(0.75)
            .stencil_clear(64);
        assert!(att.depth_ops.is_some());
        assert!(att.stencil_ops.is_some());
    }

    #[test]
    fn test_depth_stencil_display() {
        let att = RenderPassDepthStencilAttachment::depth_stencil(1.0, 0);
        let s = format!("{}", att);
        assert!(s.contains("DepthStencilAttachment"));
    }
}

// =============================================================================
// CATEGORY 5: TIMESTAMP WRITES TESTS (8 tests)
// =============================================================================

mod timestamp_writes_tests {
    use super::*;

    #[test]
    fn test_timestamp_writes_new() {
        let ts = TimestampWrites::new();
        assert!(ts.beginning_of_pass_write_index.is_none());
        assert!(ts.end_of_pass_write_index.is_none());
    }

    #[test]
    fn test_timestamp_writes_both() {
        let ts = TimestampWrites::both(0, 1);
        assert_eq!(ts.beginning_of_pass_write_index, Some(0));
        assert_eq!(ts.end_of_pass_write_index, Some(1));
    }

    #[test]
    fn test_timestamp_writes_beginning_only() {
        let ts = TimestampWrites::beginning_only(5);
        assert_eq!(ts.beginning_of_pass_write_index, Some(5));
        assert!(ts.end_of_pass_write_index.is_none());
    }

    #[test]
    fn test_timestamp_writes_end_only() {
        let ts = TimestampWrites::end_only(10);
        assert!(ts.beginning_of_pass_write_index.is_none());
        assert_eq!(ts.end_of_pass_write_index, Some(10));
    }

    #[test]
    fn test_timestamp_writes_builder_beginning() {
        let ts = TimestampWrites::new().beginning(3);
        assert_eq!(ts.beginning_of_pass_write_index, Some(3));
    }

    #[test]
    fn test_timestamp_writes_builder_end() {
        let ts = TimestampWrites::new().end(7);
        assert_eq!(ts.end_of_pass_write_index, Some(7));
    }

    #[test]
    fn test_timestamp_writes_is_enabled() {
        let ts_empty = TimestampWrites::new();
        assert!(!ts_empty.is_enabled());

        let ts_begin = TimestampWrites::beginning_only(0);
        assert!(ts_begin.is_enabled());

        let ts_both = TimestampWrites::both(0, 1);
        assert!(ts_both.is_enabled());
    }

    #[test]
    fn test_timestamp_writes_display() {
        let ts = TimestampWrites::both(0, 1);
        let s = format!("{}", ts);
        assert!(s.contains("TimestampWrites"));
    }
}

// =============================================================================
// CATEGORY 6: BUILDER TESTS - Fluent API and Presets (10 tests)
// =============================================================================

mod builder_tests {
    use super::*;

    #[test]
    fn test_builder_new() {
        let builder = RenderPassBuilder::new();
        let desc = builder.build();
        assert!(desc.label.is_none());
    }

    #[test]
    fn test_builder_label() {
        let desc = RenderPassBuilder::new()
            .label("my_pass")
            .build();
        assert_eq!(desc.label, Some("my_pass".to_string()));
    }

    #[test]
    fn test_builder_color_attachment() {
        let desc = RenderPassBuilder::new()
            .color_attachment(Operations::clear_black())
            .build();
        assert_eq!(desc.color_attachments.len(), 1);
    }

    #[test]
    fn test_builder_color_attachment_msaa() {
        let desc = RenderPassBuilder::new()
            .color_attachment_msaa(Operations::clear_black())
            .build();
        if let Some(Some(att)) = desc.color_attachments.first() {
            assert!(att.has_resolve);
        }
    }

    #[test]
    fn test_builder_depth_stencil() {
        let desc = RenderPassBuilder::new()
            .depth_stencil(
                Some(Operations::clear_depth()),
                Some(Operations::clear_stencil()),
            )
            .build();
        assert!(desc.depth_stencil_attachment.is_some());
    }

    #[test]
    fn test_builder_depth_only() {
        let desc = RenderPassBuilder::new()
            .depth_only(Operations::clear_depth())
            .build();
        if let Some(att) = desc.depth_stencil_attachment {
            assert!(att.depth_ops.is_some());
            assert!(att.stencil_ops.is_none());
        }
    }

    #[test]
    fn test_builder_depth_read_only() {
        let desc = RenderPassBuilder::new()
            .depth_read_only()
            .build();
        if let Some(att) = desc.depth_stencil_attachment {
            assert!(att.is_read_only());
        }
    }

    #[test]
    fn test_builder_timestamp_writes() {
        let desc = RenderPassBuilder::new()
            .timestamp_writes(TimestampWrites::both(0, 1))
            .build();
        assert!(desc.timestamp_writes.is_some());
    }

    #[test]
    fn test_builder_with_occlusion_queries() {
        let desc = RenderPassBuilder::new()
            .with_occlusion_queries()
            .build();
        assert!(desc.occlusion_query_enabled);
    }

    #[test]
    fn test_builder_fluent_chain() {
        let desc = RenderPassBuilder::new()
            .label("combined_pass")
            .color_attachment(Operations::clear_black())
            .color_attachment(Operations::clear_black())
            .depth_only(Operations::clear_depth())
            .timestamp_writes(TimestampWrites::both(0, 1))
            .with_occlusion_queries()
            .build();
        assert_eq!(desc.label, Some("combined_pass".to_string()));
        assert_eq!(desc.color_attachments.len(), 2);
        assert!(desc.has_depth_stencil());
        assert!(desc.has_timestamp_writes());
        assert!(desc.has_occlusion_queries());
    }
}

// =============================================================================
// CATEGORY 7: OPERATIONS TESTS - LoadOp and StoreOp (8 tests)
// =============================================================================

mod operations_tests {
    use super::*;

    #[test]
    fn test_load_op_clear() {
        let op = LoadOp::Clear(1.0f32);
        assert!(matches!(op, LoadOp::Clear(1.0)));
    }

    #[test]
    fn test_load_op_load() {
        let op: LoadOp<f32> = LoadOp::Load;
        assert!(matches!(op, LoadOp::Load));
    }

    #[test]
    fn test_store_op_store() {
        let op = StoreOp::Store;
        assert_eq!(op, StoreOp::Store);
    }

    #[test]
    fn test_store_op_discard() {
        let op = StoreOp::Discard;
        assert_eq!(op, StoreOp::Discard);
    }

    #[test]
    fn test_operations_clear() {
        let ops = Operations::clear(wgpu::Color::RED);
        assert!(matches!(ops.load, LoadOp::Clear(_)));
        assert_eq!(ops.store, StoreOp::Store);
    }

    #[test]
    fn test_operations_clear_discard() {
        let ops = Operations::clear_discard(1.0f32);
        assert!(matches!(ops.load, LoadOp::Clear(1.0)));
        assert_eq!(ops.store, StoreOp::Discard);
    }

    #[test]
    fn test_operations_load_store() {
        let ops: Operations<f32> = Operations::load_store();
        assert!(matches!(ops.load, LoadOp::Load));
        assert_eq!(ops.store, StoreOp::Store);
    }

    #[test]
    fn test_operations_display() {
        let ops = Operations::clear(1.0f32);
        let s = format!("{}", ops);
        assert!(s.contains("Operations"));
        assert!(s.contains("Clear"));
    }
}

// =============================================================================
// CATEGORY 8: VALIDATION TESTS - Error Conditions (6 tests)
// =============================================================================

mod validation_tests {
    use super::*;

    #[test]
    fn test_validate_color_count_valid() {
        assert!(validate_color_attachment_count(0).is_ok());
        assert!(validate_color_attachment_count(4).is_ok());
        assert!(validate_color_attachment_count(8).is_ok());
    }

    #[test]
    fn test_validate_color_count_invalid() {
        let result = validate_color_attachment_count(9);
        assert!(result.is_err());
        if let Err(RenderPassError::TooManyColorAttachments { count, max }) = result {
            assert_eq!(count, 9);
            assert_eq!(max, 8);
        }
    }

    #[test]
    fn test_validate_descriptor_color_only() {
        let desc = RenderPassDescriptor::new()
            .color_attachment(RenderPassColorAttachment::new());
        assert!(validate_render_pass_descriptor(&desc).is_ok());
    }

    #[test]
    fn test_validate_descriptor_depth_only() {
        let desc = RenderPassDescriptor::new()
            .depth_stencil(RenderPassDepthStencilAttachment::new());
        assert!(validate_render_pass_descriptor(&desc).is_ok());
    }

    #[test]
    fn test_validate_descriptor_no_attachments() {
        let desc = RenderPassDescriptor::new();
        let result = validate_render_pass_descriptor(&desc);
        assert!(matches!(result, Err(RenderPassError::NoAttachments)));
    }

    #[test]
    fn test_validate_descriptor_too_many_colors() {
        let mut desc = RenderPassDescriptor::new();
        for _ in 0..9 {
            desc.color_attachments.push(Some(RenderPassColorAttachment::new()));
        }
        let result = validate_render_pass_descriptor(&desc);
        assert!(matches!(result, Err(RenderPassError::TooManyColorAttachments { .. })));
    }
}

// =============================================================================
// CATEGORY 9: REAL-WORLD SCENARIOS (8 tests)
// =============================================================================

mod real_world_tests {
    use super::*;

    #[test]
    fn test_scenario_forward_rendering() {
        // Forward rendering: single color + depth
        let desc = RenderPassBuilder::color_depth().build();
        assert_eq!(desc.color_attachments.len(), 1);
        assert!(desc.has_depth_stencil());
    }

    #[test]
    fn test_scenario_deferred_gbuffer() {
        // Deferred rendering: multiple color targets (G-buffer)
        let desc = RenderPassBuilder::gbuffer().build();
        assert_eq!(desc.color_attachments.len(), 3);
        assert!(desc.has_depth_stencil());
    }

    #[test]
    fn test_scenario_shadow_map() {
        // Shadow mapping: depth only, no color
        let desc = RenderPassBuilder::shadow_map().build();
        assert!(desc.color_attachments.is_empty());
        assert!(desc.has_depth_stencil());
    }

    #[test]
    fn test_scenario_post_process() {
        // Post-processing: load existing, no depth
        let desc = RenderPassBuilder::post_process().build();
        assert_eq!(desc.color_attachments.len(), 1);
        if let Some(Some(att)) = desc.color_attachments.first() {
            assert!(matches!(att.load_op, LoadOp::Load));
        }
    }

    #[test]
    fn test_scenario_transparent_pass() {
        // Transparent rendering: load color, read-only depth
        let desc = RenderPassBuilder::transparent().build();
        assert!(desc.has_depth_stencil());
        if let Some(att) = desc.depth_stencil_attachment {
            assert!(att.is_read_only());
        }
    }

    #[test]
    fn test_scenario_ui_overlay() {
        // UI overlay: load existing color, no depth
        let desc = RenderPassBuilder::ui().build();
        if let Some(Some(att)) = desc.color_attachments.first() {
            assert!(matches!(att.load_op, LoadOp::Load));
        }
    }

    #[test]
    fn test_scenario_msaa_resolve() {
        // MSAA rendering with resolve
        let desc = RenderPassBuilder::msaa_resolve().build();
        if let Some(Some(att)) = desc.color_attachments.first() {
            assert!(att.has_resolve);
        }
    }

    #[test]
    fn test_scenario_depth_prepass() {
        // Depth pre-pass for early-Z
        let desc = RenderPassBuilder::depth_prepass().build();
        assert!(desc.color_attachments.is_empty());
        assert!(desc.has_depth_stencil());
    }
}

// =============================================================================
// CATEGORY 10: THREAD SAFETY TESTS (4 tests)
// =============================================================================

mod thread_safety_tests {
    use super::*;

    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    #[test]
    fn test_render_pass_descriptor_send() {
        assert_send::<RenderPassDescriptor>();
    }

    #[test]
    fn test_render_pass_descriptor_sync() {
        assert_sync::<RenderPassDescriptor>();
    }

    #[test]
    fn test_render_pass_builder_send() {
        assert_send::<RenderPassBuilder>();
    }

    #[test]
    fn test_render_pass_builder_sync() {
        assert_sync::<RenderPassBuilder>();
    }
}

// =============================================================================
// ADDITIONAL: CONSTANTS AND PRESETS TESTS
// =============================================================================

mod constants_tests {
    use super::*;

    #[test]
    fn test_max_color_attachments_constant() {
        assert_eq!(RENDER_PASS_MAX_COLOR_ATTACHMENTS, 8);
    }

    #[test]
    fn test_default_clear_color() {
        assert_eq!(DEFAULT_CLEAR_COLOR.r, 0.0);
        assert_eq!(DEFAULT_CLEAR_COLOR.g, 0.0);
        assert_eq!(DEFAULT_CLEAR_COLOR.b, 0.0);
        assert_eq!(DEFAULT_CLEAR_COLOR.a, 1.0);
    }

    #[test]
    fn test_default_clear_depth() {
        assert_eq!(DEFAULT_CLEAR_DEPTH, 1.0);
    }

    #[test]
    fn test_default_clear_stencil() {
        assert_eq!(DEFAULT_CLEAR_STENCIL, 0);
    }
}

mod preset_tests {
    use super::*;

    #[test]
    fn test_render_pass_presets_count() {
        assert_eq!(RENDER_PASS_PRESETS.len(), 10);
    }

    #[test]
    fn test_get_preset_info_shadow_map() {
        let info = get_render_pass_preset_info("shadow_map");
        assert!(info.is_some());
        if let Some(info) = info {
            assert_eq!(info.name, "shadow_map");
            assert_eq!(info.color_count, 0);
            assert!(info.has_depth);
        }
    }

    #[test]
    fn test_get_preset_info_gbuffer() {
        let info = get_render_pass_preset_info("gbuffer");
        assert!(info.is_some());
        if let Some(info) = info {
            assert_eq!(info.color_count, 3);
            assert!(info.has_depth);
        }
    }

    #[test]
    fn test_get_preset_info_nonexistent() {
        let info = get_render_pass_preset_info("nonexistent");
        assert!(info.is_none());
    }

    #[test]
    fn test_preset_names_iterator() {
        let names: Vec<_> = render_pass_preset_names().collect();
        assert!(names.contains(&"simple_color"));
        assert!(names.contains(&"shadow_map"));
        assert!(names.contains(&"gbuffer"));
        assert!(names.contains(&"transparent"));
    }

    #[test]
    fn test_preset_info_simple_color() {
        let info = get_render_pass_preset_info("simple_color").unwrap();
        assert_eq!(info.color_count, 1);
        assert!(!info.has_depth);
        assert!(!info.has_stencil);
    }

    #[test]
    fn test_preset_info_msaa_resolve() {
        let info = get_render_pass_preset_info("msaa_resolve").unwrap();
        assert!(info.has_resolve);
    }

    #[test]
    fn test_preset_info_stencil_only() {
        let info = get_render_pass_preset_info("stencil_only").unwrap();
        assert!(!info.has_depth);
        assert!(info.has_stencil);
    }
}

// =============================================================================
// ADDITIONAL: BUILDER PRESETS TESTS
// =============================================================================

mod builder_presets_tests {
    use super::*;

    #[test]
    fn test_builder_simple_color() {
        let desc = RenderPassBuilder::simple_color().build();
        assert_eq!(desc.label, Some("simple_color".to_string()));
        assert_eq!(desc.color_attachments.len(), 1);
        assert!(!desc.has_depth_stencil());
    }

    #[test]
    fn test_builder_simple_color_clear() {
        let desc = RenderPassBuilder::simple_color_clear(wgpu::Color::RED).build();
        assert_eq!(desc.color_attachments.len(), 1);
    }

    #[test]
    fn test_builder_color_depth_reverse_z() {
        let desc = RenderPassBuilder::color_depth_reverse_z().build();
        assert!(desc.has_depth_stencil());
        if let Some(att) = desc.depth_stencil_attachment {
            if let Some(ops) = att.depth_ops {
                if let LoadOp::Clear(depth) = ops.load {
                    assert_eq!(depth, 0.0);
                }
            }
        }
    }

    #[test]
    fn test_builder_shadow_map_reverse_z() {
        let desc = RenderPassBuilder::shadow_map_reverse_z().build();
        assert!(desc.has_depth_stencil());
        if let Some(att) = desc.depth_stencil_attachment {
            if let Some(ops) = att.depth_ops {
                if let LoadOp::Clear(depth) = ops.load {
                    assert_eq!(depth, 0.0);
                }
            }
        }
    }

    #[test]
    fn test_builder_gbuffer_reverse_z() {
        let desc = RenderPassBuilder::gbuffer_reverse_z().build();
        assert_eq!(desc.color_attachments.len(), 3);
        assert!(desc.has_depth_stencil());
    }

    #[test]
    fn test_builder_fullscreen() {
        let desc = RenderPassBuilder::fullscreen().build();
        assert_eq!(desc.label, Some("fullscreen".to_string()));
        assert_eq!(desc.color_attachments.len(), 1);
        assert!(!desc.has_depth_stencil());
    }

    #[test]
    fn test_builder_stencil_only() {
        let desc = RenderPassBuilder::stencil_only().build();
        assert_eq!(desc.label, Some("stencil_only".to_string()));
        if let Some(att) = desc.depth_stencil_attachment {
            assert!(att.depth_ops.is_none());
            assert!(att.stencil_ops.is_some());
        }
    }
}

// =============================================================================
// ADDITIONAL: CLONE AND DEBUG TESTS
// =============================================================================

mod clone_debug_tests {
    use super::*;

    #[test]
    fn test_load_op_clone() {
        let op = LoadOp::Clear(wgpu::Color::RED);
        let cloned = op.clone();
        assert_eq!(op, cloned);
    }

    #[test]
    fn test_store_op_clone() {
        let op = StoreOp::Store;
        let cloned = op.clone();
        assert_eq!(op, cloned);
    }

    #[test]
    fn test_operations_clone() {
        let ops = Operations::clear_black();
        let cloned = ops.clone();
        assert_eq!(ops, cloned);
    }

    #[test]
    fn test_color_attachment_clone() {
        let att = RenderPassColorAttachment::new();
        let cloned = att.clone();
        assert_eq!(att.store_op, cloned.store_op);
    }

    #[test]
    fn test_depth_stencil_attachment_clone() {
        let att = RenderPassDepthStencilAttachment::new();
        let cloned = att.clone();
        assert_eq!(att, cloned);
    }

    #[test]
    fn test_timestamp_writes_clone() {
        let ts = TimestampWrites::both(0, 1);
        let cloned = ts.clone();
        assert_eq!(ts, cloned);
    }

    #[test]
    fn test_render_pass_descriptor_clone() {
        let desc = RenderPassDescriptor::new().label("test");
        let cloned = desc.clone();
        assert_eq!(desc.label, cloned.label);
    }

    #[test]
    fn test_render_pass_builder_clone() {
        let builder = RenderPassBuilder::new().label("test");
        let cloned = builder.clone();
        assert_eq!(builder.build().label, cloned.build().label);
    }

    #[test]
    fn test_debug_implementations() {
        // Verify Debug is implemented
        let _ = format!("{:?}", LoadOp::Clear(1.0f32));
        let _ = format!("{:?}", StoreOp::Store);
        let _ = format!("{:?}", Operations::clear_depth());
        let _ = format!("{:?}", RenderPassColorAttachment::new());
        let _ = format!("{:?}", RenderPassDepthStencilAttachment::new());
        let _ = format!("{:?}", TimestampWrites::new());
        let _ = format!("{:?}", OcclusionQuerySet::new());
        let _ = format!("{:?}", RenderPassDescriptor::new());
        let _ = format!("{:?}", RenderPassBuilder::new());
        let _ = format!("{:?}", RenderPassError::NoAttachments);
    }
}

// =============================================================================
// ADDITIONAL: ERROR DISPLAY TESTS
// =============================================================================

mod error_tests {
    use super::*;

    #[test]
    fn test_error_display_too_many_attachments() {
        let err = RenderPassError::TooManyColorAttachments { count: 10, max: 8 };
        let s = format!("{}", err);
        assert!(s.contains("10"));
        assert!(s.contains("8"));
    }

    #[test]
    fn test_error_display_no_attachments() {
        let err = RenderPassError::NoAttachments;
        let s = format!("{}", err);
        assert!(s.contains("at least one"));
    }

    #[test]
    fn test_error_display_invalid_timestamp() {
        let err = RenderPassError::InvalidTimestampIndex {
            index: 5,
            query_set_size: 4,
        };
        let s = format!("{}", err);
        assert!(s.contains("5"));
        assert!(s.contains("4"));
    }

    #[test]
    fn test_error_is_error_trait() {
        let err: Box<dyn std::error::Error> = Box::new(RenderPassError::NoAttachments);
        assert!(!err.to_string().is_empty());
    }
}

// =============================================================================
// ADDITIONAL: OCCLUSION QUERY SET TESTS
// =============================================================================

mod occlusion_query_tests {
    use super::*;

    #[test]
    fn test_occlusion_query_set_disabled() {
        let oqs = OcclusionQuerySet::disabled();
        assert!(!oqs.enabled);
    }

    #[test]
    fn test_occlusion_query_set_enable() {
        let oqs = OcclusionQuerySet::disabled().enable();
        assert!(oqs.enabled);
    }

    #[test]
    fn test_occlusion_query_set_disable() {
        let oqs = OcclusionQuerySet::new().disable();
        assert!(!oqs.enabled);
    }

    #[test]
    fn test_occlusion_query_set_display() {
        let oqs = OcclusionQuerySet::new();
        let s = format!("{}", oqs);
        assert!(s.contains("OcclusionQuerySet"));
    }
}

// =============================================================================
// ADDITIONAL: DEPTH STENCIL BUILDER TESTS
// =============================================================================

mod depth_stencil_builder_tests {
    use super::*;

    #[test]
    fn test_depth_load() {
        let att = RenderPassDepthStencilAttachment::new().depth_load();
        if let Some(ops) = att.depth_ops {
            assert!(matches!(ops.load, LoadOp::Load));
        }
    }

    #[test]
    fn test_no_depth_ops() {
        let att = RenderPassDepthStencilAttachment::new().no_depth_ops();
        assert!(att.depth_ops.is_none());
    }

    #[test]
    fn test_stencil_load() {
        let att = RenderPassDepthStencilAttachment::new().stencil_load();
        if let Some(ops) = att.stencil_ops {
            assert!(matches!(ops.load, LoadOp::Load));
        }
    }

    #[test]
    fn test_no_stencil_ops() {
        let att = RenderPassDepthStencilAttachment::depth_stencil(1.0, 0)
            .no_stencil_ops();
        assert!(att.stencil_ops.is_none());
    }

    #[test]
    fn test_with_depth_ops() {
        let att = RenderPassDepthStencilAttachment::new()
            .with_depth_ops(Operations::clear(0.25));
        if let Some(ops) = att.depth_ops {
            if let LoadOp::Clear(depth) = ops.load {
                assert_eq!(depth, 0.25);
            }
        }
    }

    #[test]
    fn test_with_stencil_ops() {
        let att = RenderPassDepthStencilAttachment::new()
            .with_stencil_ops(Operations::clear(64u32));
        assert!(att.stencil_ops.is_some());
    }
}

// =============================================================================
// ADDITIONAL: OPERATIONS CONVENIENCE METHODS TESTS
// =============================================================================

mod operations_convenience_tests {
    use super::*;

    #[test]
    fn test_clear_black() {
        let ops = Operations::clear_black();
        if let LoadOp::Clear(color) = ops.load {
            assert_eq!(color, wgpu::Color::BLACK);
        } else {
            panic!("Expected Clear");
        }
    }

    #[test]
    fn test_clear_opaque_black() {
        let ops = Operations::clear_opaque_black();
        if let LoadOp::Clear(color) = ops.load {
            assert_eq!(color.r, 0.0);
            assert_eq!(color.g, 0.0);
            assert_eq!(color.b, 0.0);
            assert_eq!(color.a, 1.0);
        } else {
            panic!("Expected Clear");
        }
    }

    #[test]
    fn test_clear_white() {
        let ops = Operations::clear_white();
        if let LoadOp::Clear(color) = ops.load {
            assert_eq!(color, wgpu::Color::WHITE);
        } else {
            panic!("Expected Clear");
        }
    }

    #[test]
    fn test_clear_rgb() {
        let ops = Operations::clear_rgb(0.5, 0.6, 0.7);
        if let LoadOp::Clear(color) = ops.load {
            assert_eq!(color.r, 0.5);
            assert_eq!(color.g, 0.6);
            assert_eq!(color.b, 0.7);
            assert_eq!(color.a, 1.0);
        } else {
            panic!("Expected Clear");
        }
    }

    #[test]
    fn test_clear_rgba() {
        let ops = Operations::clear_rgba(0.1, 0.2, 0.3, 0.4);
        if let LoadOp::Clear(color) = ops.load {
            assert_eq!(color.r, 0.1);
            assert_eq!(color.g, 0.2);
            assert_eq!(color.b, 0.3);
            assert_eq!(color.a, 0.4);
        } else {
            panic!("Expected Clear");
        }
    }

    #[test]
    fn test_clear_depth() {
        let ops = Operations::clear_depth();
        if let LoadOp::Clear(depth) = ops.load {
            assert_eq!(depth, 1.0);
        } else {
            panic!("Expected Clear");
        }
    }

    #[test]
    fn test_clear_depth_reverse_z() {
        let ops = Operations::clear_depth_reverse_z();
        if let LoadOp::Clear(depth) = ops.load {
            assert_eq!(depth, 0.0);
        } else {
            panic!("Expected Clear");
        }
    }

    #[test]
    fn test_clear_depth_transient() {
        let ops = Operations::clear_depth_transient();
        assert!(matches!(ops.load, LoadOp::Clear(1.0)));
        assert_eq!(ops.store, StoreOp::Discard);
    }

    #[test]
    fn test_clear_stencil() {
        let ops = Operations::clear_stencil();
        if let LoadOp::Clear(stencil) = ops.load {
            assert_eq!(stencil, 0);
        } else {
            panic!("Expected Clear");
        }
    }

    #[test]
    fn test_clear_stencil_transient() {
        let ops = Operations::clear_stencil_transient();
        assert!(matches!(ops.load, LoadOp::Clear(0)));
        assert_eq!(ops.store, StoreOp::Discard);
    }

    #[test]
    fn test_load_discard() {
        let ops: Operations<f32> = Operations::load_discard();
        assert!(matches!(ops.load, LoadOp::Load));
        assert_eq!(ops.store, StoreOp::Discard);
    }
}
