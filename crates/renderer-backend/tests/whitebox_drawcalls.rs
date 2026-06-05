//! Whitebox tests for Draw Call Statistics (T-WGPU-P7.4.4)
//!
//! Comprehensive testing of draw call tracking, batching, and analysis
//! for the TRINITY renderer's profiling subsystem.

use std::collections::VecDeque;
use std::time::Duration;

use renderer_backend::profiling::drawcalls::{
    DrawBatch, DrawCall, DrawCallAnalyzer, DrawCallTracker, DrawFrameStats, DrawType,
    FrameAnalysis, FrameComparison, PassStats, PassType, PrimitiveTopology, TrendAnalysis,
    DEFAULT_HISTORY_SIZE, HEAVY_PASS_DRAW_THRESHOLD, HEAVY_PASS_VERTEX_THRESHOLD,
    MAX_HISTORY_SIZE, MIN_HISTORY_SIZE, STATE_THRASHING_THRESHOLD,
};

// ============================================================================
// DrawType Tests (20+)
// ============================================================================

mod draw_type_tests {
    use super::*;

    // ----- is_indexed() tests -----

    #[test]
    fn test_draw_is_not_indexed() {
        assert!(!DrawType::Draw.is_indexed());
    }

    #[test]
    fn test_draw_indexed_is_indexed() {
        assert!(DrawType::DrawIndexed.is_indexed());
    }

    #[test]
    fn test_draw_indirect_is_not_indexed() {
        assert!(!DrawType::DrawIndirect.is_indexed());
    }

    #[test]
    fn test_draw_indexed_indirect_is_indexed() {
        assert!(DrawType::DrawIndexedIndirect.is_indexed());
    }

    #[test]
    fn test_multi_draw_indirect_is_not_indexed() {
        assert!(!DrawType::MultiDrawIndirect.is_indexed());
    }

    #[test]
    fn test_multi_draw_indexed_indirect_is_indexed() {
        assert!(DrawType::MultiDrawIndexedIndirect.is_indexed());
    }

    #[test]
    fn test_dispatch_is_not_indexed() {
        assert!(!DrawType::Dispatch.is_indexed());
    }

    #[test]
    fn test_dispatch_indirect_is_not_indexed() {
        assert!(!DrawType::DispatchIndirect.is_indexed());
    }

    // ----- is_indirect() tests -----

    #[test]
    fn test_draw_is_not_indirect() {
        assert!(!DrawType::Draw.is_indirect());
    }

    #[test]
    fn test_draw_indexed_is_not_indirect() {
        assert!(!DrawType::DrawIndexed.is_indirect());
    }

    #[test]
    fn test_draw_indirect_is_indirect() {
        assert!(DrawType::DrawIndirect.is_indirect());
    }

    #[test]
    fn test_draw_indexed_indirect_is_indirect() {
        assert!(DrawType::DrawIndexedIndirect.is_indirect());
    }

    #[test]
    fn test_multi_draw_indirect_is_indirect() {
        assert!(DrawType::MultiDrawIndirect.is_indirect());
    }

    #[test]
    fn test_multi_draw_indexed_indirect_is_indirect() {
        assert!(DrawType::MultiDrawIndexedIndirect.is_indirect());
    }

    #[test]
    fn test_dispatch_is_not_indirect() {
        assert!(!DrawType::Dispatch.is_indirect());
    }

    #[test]
    fn test_dispatch_indirect_is_indirect() {
        assert!(DrawType::DispatchIndirect.is_indirect());
    }

    // ----- is_compute() / is_render() tests -----

    #[test]
    fn test_draw_is_render() {
        assert!(DrawType::Draw.is_render());
        assert!(!DrawType::Draw.is_compute());
    }

    #[test]
    fn test_draw_indexed_is_render() {
        assert!(DrawType::DrawIndexed.is_render());
        assert!(!DrawType::DrawIndexed.is_compute());
    }

    #[test]
    fn test_draw_indirect_is_render() {
        assert!(DrawType::DrawIndirect.is_render());
        assert!(!DrawType::DrawIndirect.is_compute());
    }

    #[test]
    fn test_draw_indexed_indirect_is_render() {
        assert!(DrawType::DrawIndexedIndirect.is_render());
        assert!(!DrawType::DrawIndexedIndirect.is_compute());
    }

    #[test]
    fn test_multi_draw_indirect_is_render() {
        assert!(DrawType::MultiDrawIndirect.is_render());
        assert!(!DrawType::MultiDrawIndirect.is_compute());
    }

    #[test]
    fn test_multi_draw_indexed_indirect_is_render() {
        assert!(DrawType::MultiDrawIndexedIndirect.is_render());
        assert!(!DrawType::MultiDrawIndexedIndirect.is_compute());
    }

    #[test]
    fn test_dispatch_is_compute() {
        assert!(DrawType::Dispatch.is_compute());
        assert!(!DrawType::Dispatch.is_render());
    }

    #[test]
    fn test_dispatch_indirect_is_compute() {
        assert!(DrawType::DispatchIndirect.is_compute());
        assert!(!DrawType::DispatchIndirect.is_render());
    }

    // ----- name() tests -----

    #[test]
    fn test_name_draw() {
        assert_eq!(DrawType::Draw.name(), "Draw");
    }

    #[test]
    fn test_name_draw_indexed() {
        assert_eq!(DrawType::DrawIndexed.name(), "DrawIndexed");
    }

    #[test]
    fn test_name_draw_indirect() {
        assert_eq!(DrawType::DrawIndirect.name(), "DrawIndirect");
    }

    #[test]
    fn test_name_draw_indexed_indirect() {
        assert_eq!(DrawType::DrawIndexedIndirect.name(), "DrawIndexedIndirect");
    }

    #[test]
    fn test_name_multi_draw_indirect() {
        assert_eq!(DrawType::MultiDrawIndirect.name(), "MultiDrawIndirect");
    }

    #[test]
    fn test_name_multi_draw_indexed_indirect() {
        assert_eq!(
            DrawType::MultiDrawIndexedIndirect.name(),
            "MultiDrawIndexedIndirect"
        );
    }

    #[test]
    fn test_name_dispatch() {
        assert_eq!(DrawType::Dispatch.name(), "Dispatch");
    }

    #[test]
    fn test_name_dispatch_indirect() {
        assert_eq!(DrawType::DispatchIndirect.name(), "DispatchIndirect");
    }

    #[test]
    fn test_display_matches_name() {
        for draw_type in [
            DrawType::Draw,
            DrawType::DrawIndexed,
            DrawType::DrawIndirect,
            DrawType::DrawIndexedIndirect,
            DrawType::MultiDrawIndirect,
            DrawType::MultiDrawIndexedIndirect,
            DrawType::Dispatch,
            DrawType::DispatchIndirect,
        ] {
            assert_eq!(draw_type.to_string(), draw_type.name());
        }
    }

    #[test]
    fn test_draw_type_clone() {
        let dt = DrawType::DrawIndexed;
        let cloned = dt.clone();
        assert_eq!(dt, cloned);
    }

    #[test]
    fn test_draw_type_debug() {
        let debug_str = format!("{:?}", DrawType::Dispatch);
        assert!(debug_str.contains("Dispatch"));
    }
}

// ============================================================================
// DrawCall Tests (30+)
// ============================================================================

mod draw_call_tests {
    use super::*;

    #[test]
    fn test_new_basic() {
        let draw = DrawCall::new(DrawType::Draw, 100, 1);
        assert_eq!(draw.draw_type, DrawType::Draw);
        assert_eq!(draw.vertex_count, 100);
        assert_eq!(draw.instance_count, 1);
        assert!(draw.index_count.is_none());
        assert!(draw.pipeline_id.is_none());
        assert!(draw.label.is_none());
    }

    #[test]
    fn test_new_with_multiple_instances() {
        let draw = DrawCall::new(DrawType::Draw, 50, 10);
        assert_eq!(draw.vertex_count, 50);
        assert_eq!(draw.instance_count, 10);
    }

    #[test]
    fn test_total_vertices_single_instance() {
        let draw = DrawCall::new(DrawType::Draw, 100, 1);
        assert_eq!(draw.total_vertices(), 100);
    }

    #[test]
    fn test_total_vertices_multiple_instances() {
        let draw = DrawCall::new(DrawType::Draw, 100, 10);
        assert_eq!(draw.total_vertices(), 1000);
    }

    #[test]
    fn test_total_vertices_zero_vertex_count() {
        let draw = DrawCall::new(DrawType::Draw, 0, 10);
        assert_eq!(draw.total_vertices(), 0);
    }

    #[test]
    fn test_total_vertices_zero_instance_count() {
        let draw = DrawCall::new(DrawType::Draw, 100, 0);
        assert_eq!(draw.total_vertices(), 0);
    }

    #[test]
    fn test_total_vertices_max_u32() {
        let draw = DrawCall::new(DrawType::Draw, u32::MAX, 1);
        assert_eq!(draw.total_vertices(), u32::MAX as u64);
    }

    #[test]
    fn test_total_vertices_overflow_prevention() {
        // Two large values that would overflow u32 but fit in u64
        let draw = DrawCall::new(DrawType::Draw, 100000, 100000);
        assert_eq!(draw.total_vertices(), 10_000_000_000u64);
    }

    // ----- total_primitives() tests -----

    #[test]
    fn test_total_primitives_point_list() {
        let draw = DrawCall::new(DrawType::Draw, 10, 1);
        assert_eq!(draw.total_primitives(PrimitiveTopology::PointList), 10);
    }

    #[test]
    fn test_total_primitives_line_list() {
        let draw = DrawCall::new(DrawType::Draw, 10, 1);
        assert_eq!(draw.total_primitives(PrimitiveTopology::LineList), 5);
    }

    #[test]
    fn test_total_primitives_line_strip() {
        let draw = DrawCall::new(DrawType::Draw, 10, 1);
        assert_eq!(draw.total_primitives(PrimitiveTopology::LineStrip), 9);
    }

    #[test]
    fn test_total_primitives_triangle_list() {
        let draw = DrawCall::new(DrawType::Draw, 12, 1);
        assert_eq!(draw.total_primitives(PrimitiveTopology::TriangleList), 4);
    }

    #[test]
    fn test_total_primitives_triangle_strip() {
        let draw = DrawCall::new(DrawType::Draw, 10, 1);
        assert_eq!(draw.total_primitives(PrimitiveTopology::TriangleStrip), 8);
    }

    #[test]
    fn test_total_primitives_with_instances() {
        let draw = DrawCall::new(DrawType::Draw, 12, 5);
        assert_eq!(draw.total_primitives(PrimitiveTopology::TriangleList), 20);
    }

    #[test]
    fn test_total_primitives_indexed_uses_index_count() {
        let draw = DrawCall::indexed(100, 12, 1);
        // Should use index_count (12), not vertex_count (100)
        assert_eq!(draw.total_primitives(PrimitiveTopology::TriangleList), 4);
    }

    #[test]
    fn test_total_primitives_line_strip_minimum_vertices() {
        let draw = DrawCall::new(DrawType::Draw, 1, 1);
        // 1 - 1 = 0 lines (saturating_sub)
        assert_eq!(draw.total_primitives(PrimitiveTopology::LineStrip), 0);
    }

    #[test]
    fn test_total_primitives_triangle_strip_minimum_vertices() {
        let draw = DrawCall::new(DrawType::Draw, 2, 1);
        // 2 - 2 = 0 triangles (saturating_sub)
        assert_eq!(draw.total_primitives(PrimitiveTopology::TriangleStrip), 0);
    }

    // ----- indexed() constructor tests -----

    #[test]
    fn test_indexed_constructor() {
        let draw = DrawCall::indexed(100, 300, 5);
        assert_eq!(draw.draw_type, DrawType::DrawIndexed);
        assert_eq!(draw.vertex_count, 100);
        assert_eq!(draw.index_count, Some(300));
        assert_eq!(draw.instance_count, 5);
    }

    #[test]
    fn test_indexed_total_vertices() {
        let draw = DrawCall::indexed(100, 300, 5);
        // total_vertices = vertex_count * instance_count
        assert_eq!(draw.total_vertices(), 500);
    }

    // ----- dispatch() constructor tests -----

    #[test]
    fn test_dispatch_constructor() {
        let draw = DrawCall::dispatch(8, 8, 1);
        assert_eq!(draw.draw_type, DrawType::Dispatch);
        assert!(draw.draw_type.is_compute());
    }

    #[test]
    fn test_dispatch_workgroups() {
        let draw = DrawCall::dispatch(8, 16, 4);
        assert_eq!(draw.workgroups(), Some((8, 16, 4)));
    }

    #[test]
    fn test_workgroups_none_for_render_draw() {
        let draw = DrawCall::new(DrawType::Draw, 100, 1);
        assert!(draw.workgroups().is_none());
    }

    #[test]
    fn test_workgroups_for_dispatch_indirect() {
        let mut draw = DrawCall::new(DrawType::DispatchIndirect, 4, 4);
        draw.index_count = Some(2);
        assert_eq!(draw.workgroups(), Some((4, 4, 2)));
    }

    // ----- Builder pattern tests -----

    #[test]
    fn test_with_pipeline() {
        let draw = DrawCall::new(DrawType::Draw, 100, 1).with_pipeline(42);
        assert_eq!(draw.pipeline_id, Some(42));
    }

    #[test]
    fn test_with_label() {
        let draw = DrawCall::new(DrawType::Draw, 100, 1).with_label("test_draw");
        assert_eq!(draw.label.as_deref(), Some("test_draw"));
    }

    #[test]
    fn test_with_label_string() {
        let draw = DrawCall::new(DrawType::Draw, 100, 1).with_label(String::from("owned_label"));
        assert_eq!(draw.label.as_deref(), Some("owned_label"));
    }

    #[test]
    fn test_builder_chaining() {
        let draw = DrawCall::new(DrawType::Draw, 100, 1)
            .with_pipeline(123)
            .with_label("chained");
        assert_eq!(draw.pipeline_id, Some(123));
        assert_eq!(draw.label.as_deref(), Some("chained"));
    }

    #[test]
    fn test_default_draw_call() {
        let draw = DrawCall::default();
        assert_eq!(draw.draw_type, DrawType::Draw);
        assert_eq!(draw.vertex_count, 0);
        assert_eq!(draw.instance_count, 1);
        assert!(draw.index_count.is_none());
    }

    #[test]
    fn test_draw_call_clone() {
        let original = DrawCall::new(DrawType::DrawIndexed, 100, 5)
            .with_pipeline(42)
            .with_label("original");
        let cloned = original.clone();
        assert_eq!(cloned.draw_type, original.draw_type);
        assert_eq!(cloned.vertex_count, original.vertex_count);
        assert_eq!(cloned.pipeline_id, original.pipeline_id);
        assert_eq!(cloned.label, original.label);
    }

    #[test]
    fn test_draw_call_debug() {
        let draw = DrawCall::new(DrawType::Draw, 100, 1);
        let debug_str = format!("{:?}", draw);
        assert!(debug_str.contains("Draw"));
        assert!(debug_str.contains("100"));
    }
}

// ============================================================================
// DrawBatch Tests (20+)
// ============================================================================

mod draw_batch_tests {
    use super::*;

    #[test]
    fn test_new_batch() {
        let batch = DrawBatch::new(42);
        assert_eq!(batch.pipeline_id, 42);
        assert!(batch.draws.is_empty());
        assert!(batch.end_time.is_none());
    }

    #[test]
    fn test_empty_batch_total_draw_count() {
        let batch = DrawBatch::new(1);
        assert_eq!(batch.total_draw_count(), 0);
    }

    #[test]
    fn test_empty_batch_total_vertex_count() {
        let batch = DrawBatch::new(1);
        assert_eq!(batch.total_vertex_count(), 0);
    }

    #[test]
    fn test_empty_batch_avg_vertices_per_draw() {
        let batch = DrawBatch::new(1);
        assert_eq!(batch.avg_vertices_per_draw(), 0.0);
    }

    #[test]
    fn test_single_draw_batch() {
        let mut batch = DrawBatch::new(1);
        batch.add_draw(DrawCall::new(DrawType::Draw, 100, 1));
        assert_eq!(batch.total_draw_count(), 1);
        assert_eq!(batch.total_vertex_count(), 100);
    }

    #[test]
    fn test_multiple_draws_batch() {
        let mut batch = DrawBatch::new(1);
        batch.add_draw(DrawCall::new(DrawType::Draw, 100, 1));
        batch.add_draw(DrawCall::new(DrawType::Draw, 200, 1));
        batch.add_draw(DrawCall::new(DrawType::Draw, 300, 1));
        assert_eq!(batch.total_draw_count(), 3);
        assert_eq!(batch.total_vertex_count(), 600);
    }

    #[test]
    fn test_batch_with_instances() {
        let mut batch = DrawBatch::new(1);
        batch.add_draw(DrawCall::new(DrawType::Draw, 100, 5));
        batch.add_draw(DrawCall::new(DrawType::Draw, 200, 2));
        // total = 100*5 + 200*2 = 500 + 400 = 900
        assert_eq!(batch.total_vertex_count(), 900);
    }

    #[test]
    fn test_total_instance_count() {
        let mut batch = DrawBatch::new(1);
        batch.add_draw(DrawCall::new(DrawType::Draw, 100, 5));
        batch.add_draw(DrawCall::new(DrawType::Draw, 200, 3));
        assert_eq!(batch.total_instance_count(), 8);
    }

    #[test]
    fn test_avg_vertices_per_draw() {
        let mut batch = DrawBatch::new(1);
        batch.add_draw(DrawCall::new(DrawType::Draw, 100, 1));
        batch.add_draw(DrawCall::new(DrawType::Draw, 200, 1));
        batch.add_draw(DrawCall::new(DrawType::Draw, 300, 1));
        // avg = 600 / 3 = 200
        assert_eq!(batch.avg_vertices_per_draw(), 200.0);
    }

    #[test]
    fn test_batch_finish_sets_end_time() {
        let mut batch = DrawBatch::new(1);
        assert!(batch.end_time.is_none());
        batch.finish();
        assert!(batch.end_time.is_some());
    }

    #[test]
    fn test_duration_before_finish() {
        let batch = DrawBatch::new(1);
        assert!(batch.duration().is_none());
    }

    #[test]
    fn test_duration_after_finish() {
        let mut batch = DrawBatch::new(1);
        batch.finish();
        let duration = batch.duration();
        assert!(duration.is_some());
        // Duration should be very small (microseconds)
        assert!(duration.unwrap() < Duration::from_millis(100));
    }

    #[test]
    fn test_batch_default() {
        let batch = DrawBatch::default();
        assert_eq!(batch.pipeline_id, 0);
        assert!(batch.draws.is_empty());
    }

    #[test]
    fn test_batch_clone() {
        let mut batch = DrawBatch::new(42);
        batch.add_draw(DrawCall::new(DrawType::Draw, 100, 1));
        batch.finish();

        let cloned = batch.clone();
        assert_eq!(cloned.pipeline_id, batch.pipeline_id);
        assert_eq!(cloned.draws.len(), batch.draws.len());
    }

    #[test]
    fn test_batch_debug() {
        let batch = DrawBatch::new(42);
        let debug_str = format!("{:?}", batch);
        assert!(debug_str.contains("42"));
        assert!(debug_str.contains("DrawBatch"));
    }

    #[test]
    fn test_batch_with_indexed_draws() {
        let mut batch = DrawBatch::new(1);
        batch.add_draw(DrawCall::indexed(100, 300, 1));
        batch.add_draw(DrawCall::indexed(50, 150, 2));
        // total_vertices = vertex_count * instance_count
        // 100*1 + 50*2 = 200
        assert_eq!(batch.total_vertex_count(), 200);
    }

    #[test]
    fn test_batch_mixed_draw_types() {
        let mut batch = DrawBatch::new(1);
        batch.add_draw(DrawCall::new(DrawType::Draw, 100, 1));
        batch.add_draw(DrawCall::indexed(50, 150, 1));
        batch.add_draw(DrawCall::new(DrawType::DrawIndirect, 200, 1));
        assert_eq!(batch.total_draw_count(), 3);
    }

    #[test]
    fn test_batch_large_draw_count() {
        let mut batch = DrawBatch::new(1);
        for _ in 0..1000 {
            batch.add_draw(DrawCall::new(DrawType::Draw, 10, 1));
        }
        assert_eq!(batch.total_draw_count(), 1000);
        assert_eq!(batch.total_vertex_count(), 10000);
    }
}

// ============================================================================
// PassStats Tests (25+)
// ============================================================================

mod pass_stats_tests {
    use super::*;

    #[test]
    fn test_new_render_pass() {
        let pass = PassStats::new(PassType::Render, Some("GBuffer".to_string()));
        assert_eq!(pass.pass_type, PassType::Render);
        assert_eq!(pass.label.as_deref(), Some("GBuffer"));
    }

    #[test]
    fn test_new_compute_pass() {
        let pass = PassStats::new(PassType::Compute, Some("Culling".to_string()));
        assert_eq!(pass.pass_type, PassType::Compute);
    }

    #[test]
    fn test_new_pass_no_label() {
        let pass = PassStats::new(PassType::Render, None);
        assert!(pass.label.is_none());
    }

    #[test]
    fn test_record_draw_increments_count() {
        let mut pass = PassStats::new(PassType::Render, None);
        pass.record_draw(&DrawCall::new(DrawType::Draw, 100, 1));
        assert_eq!(pass.draw_count, 1);
    }

    #[test]
    fn test_record_draw_accumulates_vertices() {
        let mut pass = PassStats::new(PassType::Render, None);
        pass.record_draw(&DrawCall::new(DrawType::Draw, 100, 2));
        pass.record_draw(&DrawCall::new(DrawType::Draw, 50, 3));
        // 100*2 + 50*3 = 200 + 150 = 350
        assert_eq!(pass.vertex_count, 350);
    }

    #[test]
    fn test_record_draw_accumulates_instances() {
        let mut pass = PassStats::new(PassType::Render, None);
        pass.record_draw(&DrawCall::new(DrawType::Draw, 100, 5));
        pass.record_draw(&DrawCall::new(DrawType::Draw, 100, 3));
        assert_eq!(pass.instance_count, 8);
    }

    #[test]
    fn test_record_dispatch_increments_count() {
        let mut pass = PassStats::new(PassType::Compute, None);
        pass.record_draw(&DrawCall::dispatch(8, 8, 1));
        assert_eq!(pass.dispatch_count, 1);
        assert_eq!(pass.draw_count, 0); // Should not count as draw
    }

    #[test]
    fn test_record_dispatch_accumulates_workgroups() {
        let mut pass = PassStats::new(PassType::Compute, None);
        pass.record_draw(&DrawCall::dispatch(8, 8, 4));
        pass.record_draw(&DrawCall::dispatch(4, 4, 2));
        // Workgroups are accumulated per dimension
        assert_eq!(pass.workgroup_count.0, 12); // 8 + 4
        assert_eq!(pass.workgroup_count.1, 12); // 8 + 4
        assert_eq!(pass.workgroup_count.2, 6); // 4 + 2
    }

    #[test]
    fn test_total_workgroup_invocations() {
        let mut pass = PassStats::new(PassType::Compute, None);
        pass.workgroup_count = (8, 8, 4);
        assert_eq!(pass.total_workgroup_invocations(), 256); // 8*8*4
    }

    #[test]
    fn test_record_pipeline_switch() {
        let mut pass = PassStats::new(PassType::Render, None);
        pass.record_pipeline_switch();
        pass.record_pipeline_switch();
        assert_eq!(pass.pipeline_switches, 2);
    }

    #[test]
    fn test_record_bind_group_set() {
        let mut pass = PassStats::new(PassType::Render, None);
        pass.record_bind_group_set();
        pass.record_bind_group_set();
        pass.record_bind_group_set();
        assert_eq!(pass.bind_group_sets, 3);
    }

    #[test]
    fn test_state_changes_per_draw_zero_draws() {
        let pass = PassStats::new(PassType::Render, None);
        assert_eq!(pass.state_changes_per_draw(), 0.0);
    }

    #[test]
    fn test_state_changes_per_draw() {
        let mut pass = PassStats::new(PassType::Render, None);
        pass.record_draw(&DrawCall::new(DrawType::Draw, 100, 1));
        pass.record_draw(&DrawCall::new(DrawType::Draw, 100, 1));
        pass.record_pipeline_switch();
        pass.record_bind_group_set();
        // 2 state changes / 2 draws = 1.0
        assert_eq!(pass.state_changes_per_draw(), 1.0);
    }

    #[test]
    fn test_avg_vertices_per_draw_zero_draws() {
        let pass = PassStats::new(PassType::Render, None);
        assert_eq!(pass.avg_vertices_per_draw(), 0.0);
    }

    #[test]
    fn test_avg_vertices_per_draw() {
        let mut pass = PassStats::new(PassType::Render, None);
        pass.record_draw(&DrawCall::new(DrawType::Draw, 100, 1));
        pass.record_draw(&DrawCall::new(DrawType::Draw, 200, 1));
        // avg = 300 / 2 = 150
        assert_eq!(pass.avg_vertices_per_draw(), 150.0);
    }

    #[test]
    fn test_avg_instances_per_draw() {
        let mut pass = PassStats::new(PassType::Render, None);
        pass.record_draw(&DrawCall::new(DrawType::Draw, 100, 5));
        pass.record_draw(&DrawCall::new(DrawType::Draw, 100, 3));
        // avg = 8 / 2 = 4
        assert_eq!(pass.avg_instances_per_draw(), 4.0);
    }

    #[test]
    fn test_finish_sets_duration() {
        let mut pass = PassStats::new(PassType::Render, None);
        std::thread::sleep(Duration::from_millis(1));
        pass.finish();
        assert!(pass.duration > Duration::ZERO);
    }

    #[test]
    fn test_pass_stats_display() {
        let mut pass = PassStats::new(PassType::Render, Some("test".to_string()));
        pass.record_draw(&DrawCall::new(DrawType::Draw, 100, 1));
        pass.finish();
        let display = pass.to_string();
        assert!(display.contains("Render"));
        assert!(display.contains("test"));
        assert!(display.contains("1 draws"));
    }

    #[test]
    fn test_pass_stats_default() {
        let pass = PassStats::default();
        assert_eq!(pass.pass_type, PassType::Render);
        assert!(pass.label.is_none());
        assert_eq!(pass.draw_count, 0);
    }

    #[test]
    fn test_pass_type_display() {
        assert_eq!(PassType::Render.to_string(), "Render");
        assert_eq!(PassType::Compute.to_string(), "Compute");
    }

    #[test]
    fn test_pipeline_switch_tracking_via_draw() {
        let mut pass = PassStats::new(PassType::Render, None);

        // First draw with pipeline 1
        pass.record_draw(&DrawCall::new(DrawType::Draw, 100, 1).with_pipeline(1));
        assert_eq!(pass.pipeline_switches, 0); // No switch yet

        // Second draw with same pipeline
        pass.record_draw(&DrawCall::new(DrawType::Draw, 100, 1).with_pipeline(1));
        assert_eq!(pass.pipeline_switches, 0); // Still no switch

        // Third draw with different pipeline
        pass.record_draw(&DrawCall::new(DrawType::Draw, 100, 1).with_pipeline(2));
        assert_eq!(pass.pipeline_switches, 1); // Now we have a switch
    }

    #[test]
    fn test_pass_stats_clone() {
        let mut pass = PassStats::new(PassType::Render, Some("original".to_string()));
        pass.record_draw(&DrawCall::new(DrawType::Draw, 100, 1));
        pass.finish();

        let cloned = pass.clone();
        assert_eq!(cloned.draw_count, pass.draw_count);
        assert_eq!(cloned.label, pass.label);
    }
}

// ============================================================================
// DrawFrameStats Tests (25+)
// ============================================================================

mod draw_frame_stats_tests {
    use super::*;

    #[test]
    fn test_new_frame_stats() {
        let frame = DrawFrameStats::new(42);
        assert_eq!(frame.frame_number, 42);
        assert!(frame.passes.is_empty());
    }

    #[test]
    fn test_frame_number_tracking() {
        let frame = DrawFrameStats::new(123);
        assert_eq!(frame.frame_number, 123);
    }

    #[test]
    fn test_add_pass_aggregates_draws() {
        let mut frame = DrawFrameStats::new(0);

        let mut pass1 = PassStats::new(PassType::Render, None);
        pass1.record_draw(&DrawCall::new(DrawType::Draw, 100, 1));
        pass1.finish();
        frame.add_pass(pass1);

        let mut pass2 = PassStats::new(PassType::Render, None);
        pass2.record_draw(&DrawCall::new(DrawType::Draw, 200, 1));
        pass2.record_draw(&DrawCall::new(DrawType::Draw, 300, 1));
        pass2.finish();
        frame.add_pass(pass2);

        assert_eq!(frame.total_draw_calls, 3);
    }

    #[test]
    fn test_add_pass_aggregates_dispatches() {
        let mut frame = DrawFrameStats::new(0);

        let mut pass = PassStats::new(PassType::Compute, None);
        pass.record_draw(&DrawCall::dispatch(8, 8, 1));
        pass.record_draw(&DrawCall::dispatch(4, 4, 1));
        pass.finish();
        frame.add_pass(pass);

        assert_eq!(frame.total_dispatches, 2);
    }

    #[test]
    fn test_add_pass_aggregates_vertices() {
        let mut frame = DrawFrameStats::new(0);

        let mut pass1 = PassStats::new(PassType::Render, None);
        pass1.record_draw(&DrawCall::new(DrawType::Draw, 100, 2)); // 200 vertices
        pass1.finish();
        frame.add_pass(pass1);

        let mut pass2 = PassStats::new(PassType::Render, None);
        pass2.record_draw(&DrawCall::new(DrawType::Draw, 150, 2)); // 300 vertices
        pass2.finish();
        frame.add_pass(pass2);

        assert_eq!(frame.total_vertices, 500);
    }

    #[test]
    fn test_add_pass_aggregates_instances() {
        let mut frame = DrawFrameStats::new(0);

        let mut pass = PassStats::new(PassType::Render, None);
        pass.record_draw(&DrawCall::new(DrawType::Draw, 100, 5));
        pass.record_draw(&DrawCall::new(DrawType::Draw, 100, 3));
        pass.finish();
        frame.add_pass(pass);

        assert_eq!(frame.total_instances, 8);
    }

    #[test]
    fn test_add_pass_aggregates_pipeline_switches() {
        let mut frame = DrawFrameStats::new(0);

        let mut pass = PassStats::new(PassType::Render, None);
        pass.record_pipeline_switch();
        pass.record_pipeline_switch();
        pass.finish();
        frame.add_pass(pass);

        assert_eq!(frame.total_pipeline_switches, 2);
    }

    #[test]
    fn test_add_pass_aggregates_bind_groups() {
        let mut frame = DrawFrameStats::new(0);

        let mut pass = PassStats::new(PassType::Render, None);
        pass.record_bind_group_set();
        pass.record_bind_group_set();
        pass.record_bind_group_set();
        pass.finish();
        frame.add_pass(pass);

        assert_eq!(frame.total_bind_group_sets, 3);
    }

    #[test]
    fn test_total_state_changes() {
        let mut frame = DrawFrameStats::new(0);

        let mut pass = PassStats::new(PassType::Render, None);
        pass.record_pipeline_switch();
        pass.record_bind_group_set();
        pass.record_bind_group_set();
        pass.finish();
        frame.add_pass(pass);

        assert_eq!(frame.total_state_changes(), 3);
    }

    #[test]
    fn test_state_changes_per_draw_zero_draws() {
        let frame = DrawFrameStats::new(0);
        assert_eq!(frame.state_changes_per_draw(), 0.0);
    }

    #[test]
    fn test_state_changes_per_draw() {
        let mut frame = DrawFrameStats::new(0);
        frame.total_draw_calls = 10;
        frame.total_pipeline_switches = 5;
        frame.total_bind_group_sets = 5;
        // 10 state changes / 10 draws = 1.0
        assert_eq!(frame.state_changes_per_draw(), 1.0);
    }

    #[test]
    fn test_avg_vertices_per_draw_zero_draws() {
        let frame = DrawFrameStats::new(0);
        assert_eq!(frame.avg_vertices_per_draw(), 0.0);
    }

    #[test]
    fn test_avg_vertices_per_draw() {
        let mut frame = DrawFrameStats::new(0);
        frame.total_draw_calls = 4;
        frame.total_vertices = 1000;
        assert_eq!(frame.avg_vertices_per_draw(), 250.0);
    }

    #[test]
    fn test_draws_per_ms_zero_time() {
        let frame = DrawFrameStats::new(0);
        assert_eq!(frame.draws_per_ms(), 0.0);
    }

    #[test]
    fn test_draws_per_ms() {
        let mut frame = DrawFrameStats::new(0);
        frame.total_draw_calls = 100;
        frame.frame_time = Duration::from_millis(10);
        assert_eq!(frame.draws_per_ms(), 10.0);
    }

    #[test]
    fn test_finish_sets_frame_time() {
        let mut frame = DrawFrameStats::new(0);
        std::thread::sleep(Duration::from_millis(1));
        frame.finish();
        assert!(frame.frame_time > Duration::ZERO);
    }

    #[test]
    fn test_summary_format() {
        let mut frame = DrawFrameStats::new(42);
        frame.total_draw_calls = 100;
        frame.total_dispatches = 10;
        frame.total_vertices = 50000;
        frame.total_instances = 1000;
        frame.total_pipeline_switches = 20;
        frame.total_bind_group_sets = 50;
        frame.frame_time = Duration::from_millis(16);

        let summary = frame.summary();
        assert!(summary.contains("Frame 42"));
        assert!(summary.contains("100 draws"));
        assert!(summary.contains("10 dispatches"));
        assert!(summary.contains("50000 vertices"));
    }

    #[test]
    fn test_frame_stats_display() {
        let mut frame = DrawFrameStats::new(0);
        frame.total_draw_calls = 50;
        let display = frame.to_string();
        assert!(display.contains("50 draws"));
    }

    #[test]
    fn test_frame_stats_default() {
        let frame = DrawFrameStats::default();
        assert_eq!(frame.frame_number, 0);
        assert_eq!(frame.total_draw_calls, 0);
    }

    #[test]
    fn test_frame_stats_clone() {
        let mut frame = DrawFrameStats::new(42);
        frame.total_draw_calls = 100;
        frame.finish();

        let cloned = frame.clone();
        assert_eq!(cloned.frame_number, frame.frame_number);
        assert_eq!(cloned.total_draw_calls, frame.total_draw_calls);
    }

    #[test]
    fn test_multi_pass_frame() {
        let mut frame = DrawFrameStats::new(0);

        // GBuffer pass
        let mut gbuffer = PassStats::new(PassType::Render, Some("GBuffer".to_string()));
        gbuffer.record_draw(&DrawCall::new(DrawType::Draw, 1000, 1));
        gbuffer.finish();
        frame.add_pass(gbuffer);

        // Lighting pass
        let mut lighting = PassStats::new(PassType::Render, Some("Lighting".to_string()));
        lighting.record_draw(&DrawCall::new(DrawType::Draw, 4, 1));
        lighting.finish();
        frame.add_pass(lighting);

        // Post-process compute
        let mut postprocess = PassStats::new(PassType::Compute, Some("PostProcess".to_string()));
        postprocess.record_draw(&DrawCall::dispatch(8, 8, 1));
        postprocess.finish();
        frame.add_pass(postprocess);

        assert_eq!(frame.passes.len(), 3);
        assert_eq!(frame.total_draw_calls, 2);
        assert_eq!(frame.total_dispatches, 1);
    }
}

// ============================================================================
// DrawCallTracker Tests (30+)
// ============================================================================

mod draw_call_tracker_tests {
    use super::*;

    #[test]
    fn test_new_tracker() {
        let tracker = DrawCallTracker::new();
        assert_eq!(tracker.current_frame_number(), 0);
        assert!(tracker.history().is_empty());
        assert!(tracker.is_enabled());
    }

    #[test]
    fn test_with_history_size() {
        let tracker = DrawCallTracker::with_history_size(50);
        assert_eq!(tracker.history().capacity(), 50);
    }

    #[test]
    fn test_with_history_size_clamps_min() {
        let tracker = DrawCallTracker::with_history_size(0);
        // Should clamp to MIN_HISTORY_SIZE
        assert!(tracker.history().capacity() >= MIN_HISTORY_SIZE);
    }

    #[test]
    fn test_with_history_size_clamps_max() {
        let tracker = DrawCallTracker::with_history_size(10000);
        // Should clamp to MAX_HISTORY_SIZE
        assert!(tracker.history().capacity() <= MAX_HISTORY_SIZE);
    }

    #[test]
    fn test_begin_end_frame_lifecycle() {
        let mut tracker = DrawCallTracker::new();

        tracker.begin_frame();
        tracker.begin_pass(PassType::Render, None);
        tracker.record_draw_non_indexed(100, 1);
        tracker.end_pass();
        let stats = tracker.end_frame();

        assert_eq!(stats.total_draw_calls, 1);
        assert_eq!(tracker.current_frame_number(), 1);
    }

    #[test]
    fn test_begin_pass_end_pass() {
        let mut tracker = DrawCallTracker::new();
        tracker.begin_frame();
        tracker.begin_pass(PassType::Render, Some("TestPass"));
        tracker.record_draw_non_indexed(100, 1);
        let pass_stats = tracker.end_pass();

        assert!(pass_stats.is_some());
        assert_eq!(pass_stats.unwrap().draw_count, 1);
    }

    #[test]
    fn test_record_draw() {
        let mut tracker = DrawCallTracker::new();
        tracker.begin_frame();
        tracker.begin_pass(PassType::Render, None);
        tracker.record_draw(DrawCall::new(DrawType::Draw, 100, 1));
        let pass = tracker.end_pass().unwrap();

        assert_eq!(pass.draw_count, 1);
        assert_eq!(pass.vertex_count, 100);
    }

    #[test]
    fn test_record_draw_indexed() {
        let mut tracker = DrawCallTracker::new();
        tracker.begin_frame();
        tracker.begin_pass(PassType::Render, None);
        tracker.record_draw_indexed(100, 300, 5);
        let pass = tracker.end_pass().unwrap();

        assert_eq!(pass.draw_count, 1);
        assert_eq!(pass.vertex_count, 500); // 100 * 5
    }

    #[test]
    fn test_record_dispatch() {
        let mut tracker = DrawCallTracker::new();
        tracker.begin_frame();
        tracker.begin_pass(PassType::Compute, None);
        tracker.record_dispatch(8, 8, 4);
        let pass = tracker.end_pass().unwrap();

        assert_eq!(pass.dispatch_count, 1);
        assert_eq!(pass.workgroup_count, (8, 8, 4));
    }

    #[test]
    fn test_record_pipeline_switch() {
        let mut tracker = DrawCallTracker::new();
        tracker.begin_frame();
        tracker.begin_pass(PassType::Render, None);
        tracker.record_pipeline_switch();
        tracker.record_pipeline_switch();
        let pass = tracker.end_pass().unwrap();

        assert_eq!(pass.pipeline_switches, 2);
    }

    #[test]
    fn test_record_bind_group_set() {
        let mut tracker = DrawCallTracker::new();
        tracker.begin_frame();
        tracker.begin_pass(PassType::Render, None);
        tracker.record_bind_group_set();
        tracker.record_bind_group_set();
        tracker.record_bind_group_set();
        let pass = tracker.end_pass().unwrap();

        assert_eq!(pass.bind_group_sets, 3);
    }

    #[test]
    fn test_history_ring_buffer() {
        let mut tracker = DrawCallTracker::with_history_size(5);

        for i in 0..10 {
            tracker.begin_frame();
            tracker.begin_pass(PassType::Render, None);
            tracker.record_draw_non_indexed((i + 1) * 10, 1);
            tracker.end_pass();
            tracker.end_frame();
        }

        // Should only keep last 5 frames
        assert_eq!(tracker.history().len(), 5);
        // First frame in history should be frame 5
        assert_eq!(tracker.history().front().unwrap().frame_number, 5);
    }

    #[test]
    fn test_avg_draw_calls_per_frame() {
        let mut tracker = DrawCallTracker::with_history_size(10);

        for _ in 0..5 {
            tracker.begin_frame();
            tracker.begin_pass(PassType::Render, None);
            tracker.record_draw_non_indexed(100, 1);
            tracker.record_draw_non_indexed(100, 1);
            tracker.end_pass();
            tracker.end_frame();
        }

        assert_eq!(tracker.avg_draw_calls_per_frame(), 2.0);
    }

    #[test]
    fn test_avg_draw_calls_empty_history() {
        let tracker = DrawCallTracker::new();
        assert_eq!(tracker.avg_draw_calls_per_frame(), 0.0);
    }

    #[test]
    fn test_avg_vertices_per_frame() {
        let mut tracker = DrawCallTracker::with_history_size(10);

        for _ in 0..5 {
            tracker.begin_frame();
            tracker.begin_pass(PassType::Render, None);
            tracker.record_draw_non_indexed(100, 1);
            tracker.record_draw_non_indexed(100, 1);
            tracker.end_pass();
            tracker.end_frame();
        }

        assert_eq!(tracker.avg_vertices_per_frame(), 200.0);
    }

    #[test]
    fn test_avg_dispatches_per_frame() {
        let mut tracker = DrawCallTracker::with_history_size(10);

        for _ in 0..5 {
            tracker.begin_frame();
            tracker.begin_pass(PassType::Compute, None);
            tracker.record_dispatch(8, 8, 1);
            tracker.record_dispatch(4, 4, 1);
            tracker.end_pass();
            tracker.end_frame();
        }

        assert_eq!(tracker.avg_dispatches_per_frame(), 2.0);
    }

    #[test]
    fn test_multiple_passes_per_frame() {
        let mut tracker = DrawCallTracker::new();
        tracker.begin_frame();

        tracker.begin_pass(PassType::Render, Some("Pass1"));
        tracker.record_draw_non_indexed(100, 1);
        tracker.end_pass();

        tracker.begin_pass(PassType::Render, Some("Pass2"));
        tracker.record_draw_non_indexed(200, 1);
        tracker.end_pass();

        tracker.begin_pass(PassType::Compute, Some("Pass3"));
        tracker.record_dispatch(8, 8, 1);
        tracker.end_pass();

        let stats = tracker.end_frame();

        assert_eq!(stats.passes.len(), 3);
        assert_eq!(stats.total_draw_calls, 2);
        assert_eq!(stats.total_dispatches, 1);
    }

    #[test]
    fn test_empty_frame() {
        let mut tracker = DrawCallTracker::new();
        tracker.begin_frame();
        let stats = tracker.end_frame();

        assert_eq!(stats.total_draw_calls, 0);
        assert_eq!(stats.total_vertices, 0);
    }

    #[test]
    fn test_disabled_tracker() {
        let mut tracker = DrawCallTracker::new();
        tracker.set_enabled(false);

        tracker.begin_frame();
        tracker.begin_pass(PassType::Render, None);
        tracker.record_draw_non_indexed(100, 1);
        let stats = tracker.end_frame();

        // Should return default stats when disabled
        assert_eq!(stats.total_draw_calls, 0);
    }

    #[test]
    fn test_enable_disable_tracking() {
        let mut tracker = DrawCallTracker::new();
        assert!(tracker.is_enabled());

        tracker.set_enabled(false);
        assert!(!tracker.is_enabled());

        tracker.set_enabled(true);
        assert!(tracker.is_enabled());
    }

    #[test]
    fn test_last_frame_stats() {
        let mut tracker = DrawCallTracker::new();

        assert!(tracker.last_frame_stats().is_none());

        tracker.begin_frame();
        tracker.begin_pass(PassType::Render, None);
        tracker.record_draw_non_indexed(100, 1);
        tracker.end_pass();
        tracker.end_frame();

        let last = tracker.last_frame_stats();
        assert!(last.is_some());
        assert_eq!(last.unwrap().total_draw_calls, 1);
    }

    #[test]
    fn test_clear_history() {
        let mut tracker = DrawCallTracker::new();

        for _ in 0..5 {
            tracker.begin_frame();
            tracker.end_frame();
        }

        assert_eq!(tracker.history().len(), 5);

        tracker.clear_history();
        assert!(tracker.history().is_empty());
    }

    #[test]
    fn test_reset() {
        let mut tracker = DrawCallTracker::new();

        for _ in 0..5 {
            tracker.begin_frame();
            tracker.end_frame();
        }

        tracker.reset();

        assert_eq!(tracker.current_frame_number(), 0);
        assert!(tracker.history().is_empty());
    }

    #[test]
    fn test_current_frame_stats() {
        let mut tracker = DrawCallTracker::new();
        tracker.begin_frame();
        tracker.begin_pass(PassType::Render, None);
        tracker.record_draw_non_indexed(100, 1);

        let current = tracker.current_frame_stats();
        assert_eq!(current.frame_number, 0);
    }

    #[test]
    fn test_nested_begin_pass_closes_previous() {
        let mut tracker = DrawCallTracker::new();
        tracker.begin_frame();

        tracker.begin_pass(PassType::Render, Some("First"));
        tracker.record_draw_non_indexed(100, 1);
        // Don't explicitly end pass

        tracker.begin_pass(PassType::Render, Some("Second"));
        tracker.record_draw_non_indexed(200, 1);
        tracker.end_pass();

        let stats = tracker.end_frame();
        assert_eq!(stats.passes.len(), 2);
    }

    #[test]
    fn test_begin_frame_closes_previous_pass() {
        let mut tracker = DrawCallTracker::new();

        tracker.begin_frame();
        tracker.begin_pass(PassType::Render, None);
        tracker.record_draw_non_indexed(100, 1);
        // Don't explicitly end pass or frame

        tracker.begin_frame();
        // Previous pass should have been closed
    }

    #[test]
    fn test_default_tracker() {
        let tracker = DrawCallTracker::default();
        assert_eq!(tracker.current_frame_number(), 0);
        assert!(tracker.is_enabled());
    }

    #[test]
    fn test_avg_frame_time_ms() {
        let mut tracker = DrawCallTracker::with_history_size(10);

        for _ in 0..3 {
            tracker.begin_frame();
            tracker.begin_pass(PassType::Render, None);
            tracker.record_draw_non_indexed(100, 1);
            tracker.end_pass();
            tracker.end_frame();
        }

        // Frame time should be very small but > 0
        let avg_time = tracker.avg_frame_time_ms();
        assert!(avg_time >= 0.0);
    }
}

// ============================================================================
// DrawCallAnalyzer Tests (20+)
// ============================================================================

mod draw_call_analyzer_tests {
    use super::*;

    fn create_test_frame(draw_count: usize, verts_per_draw: u32) -> DrawFrameStats {
        let mut frame = DrawFrameStats::new(0);
        let mut pass = PassStats::new(PassType::Render, None);

        for _ in 0..draw_count {
            pass.record_draw(&DrawCall::new(DrawType::Draw, verts_per_draw, 1));
        }
        pass.finish();
        frame.add_pass(pass);
        frame.finish();
        frame
    }

    #[test]
    fn test_analyze_frame_basic() {
        let frame = create_test_frame(100, 100);
        let analysis = DrawCallAnalyzer::analyze_frame(&frame);

        assert_eq!(analysis.frame_number, 0);
    }

    #[test]
    fn test_analyze_frame_draw_call_density() {
        let mut frame = create_test_frame(100, 100);
        frame.frame_time = Duration::from_millis(10);

        let analysis = DrawCallAnalyzer::analyze_frame(&frame);
        assert_eq!(analysis.draw_call_density, 10.0); // 100 draws / 10ms
    }

    #[test]
    fn test_find_heavy_passes_by_draws() {
        let mut frame = DrawFrameStats::new(0);

        let mut light = PassStats::new(PassType::Render, Some("light".to_string()));
        for _ in 0..10 {
            light.record_draw(&DrawCall::new(DrawType::Draw, 100, 1));
        }
        light.finish();
        frame.add_pass(light);

        let mut heavy = PassStats::new(PassType::Render, Some("heavy".to_string()));
        for _ in 0..2000 {
            heavy.record_draw(&DrawCall::new(DrawType::Draw, 100, 1));
        }
        heavy.finish();
        frame.add_pass(heavy);

        let heavy_passes = DrawCallAnalyzer::find_heavy_passes(&frame, 1000);
        assert_eq!(heavy_passes.len(), 1);
        assert_eq!(heavy_passes[0].label.as_deref(), Some("heavy"));
    }

    #[test]
    fn test_find_heavy_passes_by_vertices() {
        let mut frame = DrawFrameStats::new(0);

        let mut heavy = PassStats::new(PassType::Render, Some("heavy".to_string()));
        // Single draw with lots of vertices
        heavy.record_draw(&DrawCall::new(DrawType::Draw, 100, 15000)); // 1.5M vertices
        heavy.finish();
        frame.add_pass(heavy);

        let heavy_passes = DrawCallAnalyzer::find_heavy_passes(&frame, 10000); // Threshold higher than draw count
        assert_eq!(heavy_passes.len(), 1); // Found by vertex count
    }

    #[test]
    fn test_detect_state_thrashing_true() {
        let mut frame = DrawFrameStats::new(0);
        let mut pass = PassStats::new(PassType::Render, None);

        // Many state changes relative to draws
        pass.record_draw(&DrawCall::new(DrawType::Draw, 100, 1));
        pass.record_pipeline_switch();
        pass.record_bind_group_set();
        pass.record_draw(&DrawCall::new(DrawType::Draw, 100, 1));
        pass.finish();
        frame.add_pass(pass);
        frame.finish();

        // State changes per draw = 2/2 = 1.0 > 0.5 threshold
        assert!(DrawCallAnalyzer::detect_state_thrashing(&frame));
    }

    #[test]
    fn test_detect_state_thrashing_false() {
        let mut frame = DrawFrameStats::new(0);
        let mut pass = PassStats::new(PassType::Render, None);

        // Many draws with few state changes
        for _ in 0..100 {
            pass.record_draw(&DrawCall::new(DrawType::Draw, 100, 1));
        }
        pass.record_pipeline_switch();
        pass.finish();
        frame.add_pass(pass);
        frame.finish();

        // State changes per draw = 1/100 = 0.01 < 0.5 threshold
        assert!(!DrawCallAnalyzer::detect_state_thrashing(&frame));
    }

    #[test]
    fn test_batching_efficiency_good() {
        let frame = create_test_frame(10, 1000); // High verts per draw
        let efficiency = DrawCallAnalyzer::batching_efficiency(&frame);
        assert!(efficiency >= 1.0);
    }

    #[test]
    fn test_batching_efficiency_poor() {
        let frame = create_test_frame(1000, 10); // Low verts per draw
        let efficiency = DrawCallAnalyzer::batching_efficiency(&frame);
        assert!(efficiency < 0.1);
    }

    #[test]
    fn test_batching_efficiency_no_draws() {
        let frame = DrawFrameStats::new(0);
        let efficiency = DrawCallAnalyzer::batching_efficiency(&frame);
        assert_eq!(efficiency, 1.0); // No draws = perfect efficiency
    }

    #[test]
    fn test_recommendations_cpu_bound() {
        let frame = create_test_frame(2000, 10); // Many small draws
        let recs = DrawCallAnalyzer::recommendations(&frame);
        assert!(!recs.is_empty());
        // Should recommend batching/instancing
        assert!(recs
            .iter()
            .any(|r| r.contains("instancing") || r.contains("batching")));
    }

    #[test]
    fn test_recommendations_state_thrashing() {
        let mut frame = DrawFrameStats::new(0);
        let mut pass = PassStats::new(PassType::Render, None);

        for _ in 0..10 {
            pass.record_draw(&DrawCall::new(DrawType::Draw, 100, 1));
            pass.record_pipeline_switch();
        }
        pass.finish();
        frame.add_pass(pass);
        frame.finish();

        let recs = DrawCallAnalyzer::recommendations(&frame);
        assert!(recs.iter().any(|r| r.contains("state change")));
    }

    #[test]
    fn test_compare_frames_draw_delta() {
        let frame_a = create_test_frame(10, 100);
        let frame_b = create_test_frame(20, 100);

        let comparison = DrawCallAnalyzer::compare_frames(&frame_a, &frame_b);
        assert_eq!(comparison.draw_delta, 10);
    }

    #[test]
    fn test_compare_frames_vertex_delta() {
        let frame_a = create_test_frame(10, 100); // 1000 vertices
        let frame_b = create_test_frame(10, 200); // 2000 vertices

        let comparison = DrawCallAnalyzer::compare_frames(&frame_a, &frame_b);
        assert_eq!(comparison.vertex_delta, 1000);
    }

    #[test]
    fn test_compare_frames_negative_delta() {
        let frame_a = create_test_frame(20, 100);
        let frame_b = create_test_frame(10, 100);

        let comparison = DrawCallAnalyzer::compare_frames(&frame_a, &frame_b);
        assert_eq!(comparison.draw_delta, -10);
    }

    #[test]
    fn test_analyze_trend_empty_history() {
        let history: VecDeque<DrawFrameStats> = VecDeque::new();
        let trend = DrawCallAnalyzer::analyze_trend(&history);

        assert_eq!(trend.samples, 0);
        assert_eq!(trend.avg_draw_calls, 0.0);
    }

    #[test]
    fn test_analyze_trend_single_frame() {
        let mut history = VecDeque::new();
        history.push_back(create_test_frame(100, 100));

        let trend = DrawCallAnalyzer::analyze_trend(&history);
        assert_eq!(trend.samples, 1);
        assert_eq!(trend.avg_draw_calls, 100.0);
    }

    #[test]
    fn test_analyze_trend_increasing() {
        let mut history = VecDeque::new();

        for i in 0..10 {
            let frame = create_test_frame((i + 1) * 10, 100);
            history.push_back(frame);
        }

        let trend = DrawCallAnalyzer::analyze_trend(&history);
        assert!(trend.is_increasing());
    }

    #[test]
    fn test_analyze_trend_decreasing() {
        let mut history = VecDeque::new();

        for i in (0..10).rev() {
            let frame = create_test_frame((i + 1) * 10, 100);
            history.push_back(frame);
        }

        let trend = DrawCallAnalyzer::analyze_trend(&history);
        assert!(trend.is_decreasing());
    }

    #[test]
    fn test_analyze_trend_stable() {
        let mut history = VecDeque::new();

        for _ in 0..10 {
            let frame = create_test_frame(100, 100);
            history.push_back(frame);
        }

        let trend = DrawCallAnalyzer::analyze_trend(&history);
        // Draw count doesn't change much, so variance should be low
        assert!(trend.draw_variance < 1.0);
    }

    #[test]
    fn test_analyze_frame_bottleneck_excessive_draws() {
        let frame = create_test_frame(15000, 10);
        let analysis = DrawCallAnalyzer::analyze_frame(&frame);

        assert!(analysis
            .bottlenecks
            .iter()
            .any(|b| b.contains("Excessive draw calls")));
    }
}

// ============================================================================
// FrameAnalysis Tests (15+)
// ============================================================================

mod frame_analysis_tests {
    use super::*;

    fn create_frame_with_draws(count: usize, verts: u32) -> DrawFrameStats {
        let mut frame = DrawFrameStats::new(0);
        let mut pass = PassStats::new(PassType::Render, None);

        for _ in 0..count {
            pass.record_draw(&DrawCall::new(DrawType::Draw, verts, 1));
        }
        pass.finish();
        frame.add_pass(pass);
        frame.finish();
        frame
    }

    #[test]
    fn test_is_cpu_bound_many_small_draws() {
        // Many draws with few vertices = CPU bound
        let frame = create_frame_with_draws(2000, 10);
        let analysis = DrawCallAnalyzer::analyze_frame(&frame);

        assert!(analysis.is_cpu_bound());
        assert!(!analysis.is_gpu_bound());
    }

    #[test]
    fn test_is_cpu_bound_threshold() {
        // Exactly at threshold
        let frame = create_frame_with_draws(1001, 99);
        let analysis = DrawCallAnalyzer::analyze_frame(&frame);

        assert!(analysis.is_cpu_bound());
    }

    #[test]
    fn test_is_not_cpu_bound_few_draws() {
        let frame = create_frame_with_draws(100, 10);
        let analysis = DrawCallAnalyzer::analyze_frame(&frame);

        assert!(!analysis.is_cpu_bound());
    }

    #[test]
    fn test_is_gpu_bound_few_large_draws() {
        // Few draws with lots of vertices = GPU bound
        let mut frame = DrawFrameStats::new(0);
        let mut pass = PassStats::new(PassType::Render, None);

        // 50 draws with 30K vertices each = 1.5M vertices
        for _ in 0..50 {
            pass.record_draw(&DrawCall::new(DrawType::Draw, 30000, 1));
        }
        pass.finish();
        frame.add_pass(pass);
        frame.finish();

        let analysis = DrawCallAnalyzer::analyze_frame(&frame);
        assert!(analysis.is_gpu_bound());
    }

    #[test]
    fn test_is_not_gpu_bound_many_draws() {
        let frame = create_frame_with_draws(500, 10000);
        let analysis = DrawCallAnalyzer::analyze_frame(&frame);

        // Too many draws to be GPU bound
        assert!(!analysis.is_gpu_bound());
    }

    #[test]
    fn test_batching_score_high() {
        let frame = create_frame_with_draws(10, 5000);
        let analysis = DrawCallAnalyzer::analyze_frame(&frame);

        // High average vertices = good batching
        assert!(analysis.batching_score >= 1.0);
    }

    #[test]
    fn test_batching_score_low() {
        let frame = create_frame_with_draws(100, 50);
        let analysis = DrawCallAnalyzer::analyze_frame(&frame);

        // Low average vertices = poor batching
        assert!(analysis.batching_score < 0.1);
    }

    #[test]
    fn test_state_change_ratio() {
        let mut frame = DrawFrameStats::new(0);
        let mut pass = PassStats::new(PassType::Render, None);

        for _ in 0..10 {
            pass.record_draw(&DrawCall::new(DrawType::Draw, 100, 1));
            pass.record_pipeline_switch();
        }
        pass.finish();
        frame.add_pass(pass);
        frame.finish();

        let analysis = DrawCallAnalyzer::analyze_frame(&frame);
        assert_eq!(analysis.state_change_ratio, 1.0);
    }

    #[test]
    fn test_recommendations_low_batching() {
        let frame = create_frame_with_draws(100, 50); // Low batching score
        let analysis = DrawCallAnalyzer::analyze_frame(&frame);
        let recs = analysis.recommendations();

        assert!(recs.iter().any(|r| r.contains("batching")));
    }

    #[test]
    fn test_recommendations_high_density() {
        let mut frame = create_frame_with_draws(1000, 100);
        frame.frame_time = Duration::from_millis(1); // 1000 draws in 1ms = high density

        let analysis = DrawCallAnalyzer::analyze_frame(&frame);
        let recs = analysis.recommendations();

        assert!(recs.iter().any(|r| r.contains("GPU-driven")));
    }

    #[test]
    fn test_frame_analysis_display() {
        let frame = create_frame_with_draws(100, 100);
        let analysis = DrawCallAnalyzer::analyze_frame(&frame);
        let display = analysis.to_string();

        assert!(display.contains("Frame 0 Analysis"));
        assert!(display.contains("draws/ms"));
    }

    #[test]
    fn test_frame_analysis_clone() {
        let frame = create_frame_with_draws(100, 100);
        let analysis = DrawCallAnalyzer::analyze_frame(&frame);
        let cloned = analysis.clone();

        assert_eq!(cloned.frame_number, analysis.frame_number);
        assert_eq!(cloned.batching_score, analysis.batching_score);
    }

    #[test]
    fn test_both_bounds_mutually_exclusive() {
        // A frame shouldn't be both CPU and GPU bound
        let frame = create_frame_with_draws(500, 500);
        let analysis = DrawCallAnalyzer::analyze_frame(&frame);

        // Neither extreme
        assert!(!analysis.is_cpu_bound() || !analysis.is_gpu_bound());
    }

    #[test]
    fn test_recommendations_empty_for_optimal_frame() {
        // Good batching, low state changes, moderate density
        let mut frame = DrawFrameStats::new(0);
        let mut pass = PassStats::new(PassType::Render, None);

        for _ in 0..50 {
            pass.record_draw(&DrawCall::new(DrawType::Draw, 2000, 1));
        }
        pass.finish();
        frame.add_pass(pass);
        frame.frame_time = Duration::from_millis(16);
        frame.finish();

        let analysis = DrawCallAnalyzer::analyze_frame(&frame);
        let recs = analysis.recommendations();

        // Should have few or no recommendations
        assert!(recs.len() < 3);
    }
}

// ============================================================================
// Additional Edge Case and Constants Tests
// ============================================================================

mod edge_case_tests {
    use super::*;

    #[test]
    fn test_default_history_size_constant() {
        assert_eq!(DEFAULT_HISTORY_SIZE, 120);
    }

    #[test]
    fn test_min_history_size_constant() {
        assert_eq!(MIN_HISTORY_SIZE, 1);
    }

    #[test]
    fn test_max_history_size_constant() {
        assert_eq!(MAX_HISTORY_SIZE, 3600);
    }

    #[test]
    fn test_state_thrashing_threshold_constant() {
        assert_eq!(STATE_THRASHING_THRESHOLD, 0.5);
    }

    #[test]
    fn test_heavy_pass_draw_threshold_constant() {
        assert_eq!(HEAVY_PASS_DRAW_THRESHOLD, 1000);
    }

    #[test]
    fn test_heavy_pass_vertex_threshold_constant() {
        assert_eq!(HEAVY_PASS_VERTEX_THRESHOLD, 1_000_000);
    }

    #[test]
    fn test_primitive_topology_default() {
        let topo = PrimitiveTopology::default();
        assert_eq!(topo, PrimitiveTopology::TriangleList);
    }

    #[test]
    fn test_pass_type_default() {
        let pass_type = PassType::default();
        assert_eq!(pass_type, PassType::Render);
    }

    #[test]
    fn test_frame_comparison_display() {
        let comparison = FrameComparison {
            frame_a: 0,
            frame_b: 1,
            draw_delta: 10,
            vertex_delta: -500,
            time_delta_ms: 2.5,
            state_change_delta: 5,
        };

        let display = comparison.to_string();
        assert!(display.contains("Frame 0 vs 1"));
        assert!(display.contains("+10"));
        assert!(display.contains("-500"));
    }

    #[test]
    fn test_frame_comparison_default() {
        let comparison = FrameComparison::default();
        assert_eq!(comparison.frame_a, 0);
        assert_eq!(comparison.draw_delta, 0);
    }

    #[test]
    fn test_trend_analysis_is_stable() {
        let trend = TrendAnalysis {
            avg_draw_calls: 100.0,
            avg_frame_time_ms: 16.0,
            draw_variance: 10.0,
            time_variance: 0.5,
            draw_trend: 0.0,
            samples: 60,
        };

        assert!(trend.is_stable());
    }

    #[test]
    fn test_trend_analysis_not_stable_high_variance() {
        let trend = TrendAnalysis {
            avg_draw_calls: 100.0,
            avg_frame_time_ms: 16.0,
            draw_variance: 500.0, // High variance
            time_variance: 0.5,
            draw_trend: 0.0,
            samples: 60,
        };

        assert!(!trend.is_stable());
    }

    #[test]
    fn test_trend_analysis_display() {
        let trend = TrendAnalysis {
            avg_draw_calls: 150.0,
            avg_frame_time_ms: 16.67,
            draw_variance: 10.0,
            time_variance: 0.5,
            draw_trend: 1.0,
            samples: 60,
        };

        let display = trend.to_string();
        assert!(display.contains("60 samples"));
        assert!(display.contains("increasing"));
    }

    #[test]
    fn test_trend_analysis_default() {
        let trend = TrendAnalysis::default();
        assert_eq!(trend.samples, 0);
        assert_eq!(trend.avg_draw_calls, 0.0);
    }

    #[test]
    fn test_zero_vertex_primitives() {
        let draw = DrawCall::new(DrawType::Draw, 0, 1);
        assert_eq!(draw.total_primitives(PrimitiveTopology::TriangleList), 0);
    }

    #[test]
    fn test_very_large_workgroups() {
        let draw = DrawCall::dispatch(65535, 65535, 1024);
        let (x, y, z) = draw.workgroups().unwrap();
        assert_eq!(x, 65535);
        assert_eq!(y, 65535);
        assert_eq!(z, 1024);
    }

    #[test]
    fn test_pass_stats_workgroup_accumulation_saturation() {
        let mut pass = PassStats::new(PassType::Compute, None);

        // Dispatch with max u32 workgroups
        let mut draw = DrawCall::dispatch(u32::MAX, 1, 1);
        draw.draw_type = DrawType::Dispatch;
        pass.record_draw(&draw);

        // Should saturate, not overflow
        assert_eq!(pass.workgroup_count.0, u32::MAX);
    }
}

// ============================================================================
// Integration / Workflow Tests
// ============================================================================

mod integration_tests {
    use super::*;

    #[test]
    fn test_full_frame_workflow() {
        let mut tracker = DrawCallTracker::new();

        // Simulate a typical frame
        tracker.begin_frame();

        // Shadow pass
        tracker.begin_pass(PassType::Render, Some("Shadows"));
        for _ in 0..50 {
            tracker.record_draw_indexed(1000, 3000, 1);
        }
        tracker.end_pass();

        // GBuffer pass
        tracker.begin_pass(PassType::Render, Some("GBuffer"));
        tracker.record_pipeline_switch();
        for _ in 0..100 {
            tracker.record_draw_indexed(500, 1500, 1);
            tracker.record_bind_group_set();
        }
        tracker.end_pass();

        // Lighting compute
        tracker.begin_pass(PassType::Compute, Some("Lighting"));
        tracker.record_dispatch(32, 32, 1);
        tracker.end_pass();

        // Post-process
        tracker.begin_pass(PassType::Render, Some("PostProcess"));
        tracker.record_draw_non_indexed(4, 1); // Fullscreen quad
        tracker.end_pass();

        let stats = tracker.end_frame();

        assert_eq!(stats.passes.len(), 4);
        assert_eq!(stats.total_draw_calls, 151); // 50 + 100 + 1
        assert_eq!(stats.total_dispatches, 1);

        // Analyze the frame
        let analysis = DrawCallAnalyzer::analyze_frame(&stats);
        let _ = analysis.recommendations();
    }

    #[test]
    fn test_multi_frame_analysis() {
        let mut tracker = DrawCallTracker::with_history_size(60);

        // Simulate varying workloads
        for i in 0..30 {
            tracker.begin_frame();
            tracker.begin_pass(PassType::Render, None);

            let draw_count = 100 + (i % 10) * 10;
            for _ in 0..draw_count {
                tracker.record_draw_non_indexed(100, 1);
            }

            tracker.end_pass();
            tracker.end_frame();
        }

        // Analyze trends
        let trend = DrawCallAnalyzer::analyze_trend(tracker.history());
        assert_eq!(trend.samples, 30);

        // Check averages
        let avg_draws = tracker.avg_draw_calls_per_frame();
        assert!(avg_draws > 0.0);
    }
}
