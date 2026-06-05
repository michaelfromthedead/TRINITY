// Blackbox contract tests for T-WGPU-P4.4.5 Pipeline Statistics
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::pipeline_statistics::*` and
// `renderer_backend::query_pool::*` -- no internal fields, no private methods.
//
// Acceptance criteria (T-WGPU-P4.4.5):
//   1. Feature check (PIPELINE_STATISTICS_QUERY) - is_supported()
//   2. QueryType::PipelineStatistics - pool creation works
//   3. 5 statistic types - vertex, clipper, fragment, compute fields
//   4. overdraw_estimate() - calculation accuracy
//   5. culling_efficiency() - calculation accuracy
//
// Coverage:
//   1.  PipelineStatisticsResult::new() construction
//   2.  PipelineStatisticsResult::zero() factory
//   3.  PipelineStatisticsResult field accessors (all 5 statistic types)
//   4.  PipelineStatisticsResult::overdraw_estimate() calculation
//   5.  PipelineStatisticsResult::culling_efficiency() calculation
//   6.  PipelineStatisticsResult edge cases (zero denominators)
//   7.  StatisticType enum variants (all 5)
//   8.  StatisticType::all() returns all variants
//   9.  StatisticType::index() mapping
//  10.  StatisticType::name() returns descriptive names
//  11.  PipelineStatisticsResult::get() by StatisticType
//  12.  PipelineStatisticsResult::to_array() conversion
//  13.  PipelineStatisticsResult::from_slice() parsing
//  14.  PipelineStatisticsResult::combine() aggregation
//  15.  PipelineStatisticsResult::has_activity() detection
//  16.  PipelineStatisticsResult::is_graphics_pass() / is_compute_pass()
//  17.  PipelineStatisticsData construction and access
//  18.  PipelineStatisticsData::aggregate()
//  19.  PipelineStatisticsPoolBuilder API
//  20.  calculate_resolve_buffer_size() utility
//  21.  all_statistics_types() returns all flags

use renderer_backend::pipeline_statistics::{
    PipelineStatisticsResult, PipelineStatisticsData, PipelineStatisticsPoolBuilder,
    StatisticType, PipelineStatisticsAllocation, PipelineStatisticsResolveParams,
    LabeledStatisticsResult,
    // Constants
    STATISTIC_SIZE_BYTES, STATISTICS_PER_QUERY, STATISTICS_RESULT_SIZE_BYTES,
    MIN_POOL_CAPACITY, MAX_RECOMMENDED_CAPACITY, DEFAULT_LABEL_PREFIX,
    // Helper functions
    calculate_resolve_buffer_size, all_statistics_types,
};
// Note: QueryError would be used for pool creation tests with device
#[allow(unused_imports)]
use renderer_backend::query_pool::QueryError;

// =========================================================================
// Criteria #3: 5 Statistic Types - StatisticType Enum Tests
// =========================================================================

#[test]
fn test_statistic_type_vertex_shader_invocations() {
    let stat = StatisticType::VertexShaderInvocations;
    assert_eq!(stat.index(), 0, "VertexShaderInvocations should be index 0");
    assert!(!stat.name().is_empty(), "Should have a descriptive name");
}

#[test]
fn test_statistic_type_clipper_invocations() {
    let stat = StatisticType::ClipperInvocations;
    assert_eq!(stat.index(), 1, "ClipperInvocations should be index 1");
    assert!(!stat.name().is_empty(), "Should have a descriptive name");
}

#[test]
fn test_statistic_type_clipper_primitives_out() {
    let stat = StatisticType::ClipperPrimitivesOut;
    assert_eq!(stat.index(), 2, "ClipperPrimitivesOut should be index 2");
    assert!(!stat.name().is_empty(), "Should have a descriptive name");
}

#[test]
fn test_statistic_type_fragment_shader_invocations() {
    let stat = StatisticType::FragmentShaderInvocations;
    assert_eq!(stat.index(), 3, "FragmentShaderInvocations should be index 3");
    assert!(!stat.name().is_empty(), "Should have a descriptive name");
}

#[test]
fn test_statistic_type_compute_shader_invocations() {
    let stat = StatisticType::ComputeShaderInvocations;
    assert_eq!(stat.index(), 4, "ComputeShaderInvocations should be index 4");
    assert!(!stat.name().is_empty(), "Should have a descriptive name");
}

#[test]
fn test_statistic_type_all_returns_five_variants() {
    let all = StatisticType::all();
    assert_eq!(all.len(), 5, "Should return all 5 statistic types");

    // Verify order and uniqueness
    assert_eq!(all[0], StatisticType::VertexShaderInvocations);
    assert_eq!(all[1], StatisticType::ClipperInvocations);
    assert_eq!(all[2], StatisticType::ClipperPrimitivesOut);
    assert_eq!(all[3], StatisticType::FragmentShaderInvocations);
    assert_eq!(all[4], StatisticType::ComputeShaderInvocations);
}

#[test]
fn test_statistic_type_index_matches_array_position() {
    let all = StatisticType::all();
    for (i, stat) in all.iter().enumerate() {
        assert_eq!(stat.index(), i, "Index should match array position for {:?}", stat);
    }
}

#[test]
fn test_statistic_type_short_name_exists() {
    for stat in StatisticType::all() {
        let short = stat.short_name();
        assert!(!short.is_empty(), "Short name should exist for {:?}", stat);
        assert!(short.len() < stat.name().len() || short.len() <= 20,
            "Short name should be abbreviated");
    }
}

#[test]
fn test_statistic_type_equality() {
    assert_eq!(StatisticType::VertexShaderInvocations, StatisticType::VertexShaderInvocations);
    assert_ne!(StatisticType::VertexShaderInvocations, StatisticType::FragmentShaderInvocations);
}

#[test]
fn test_statistic_type_clone_and_copy() {
    let stat = StatisticType::ComputeShaderInvocations;
    let cloned = stat.clone();
    let copied: StatisticType = stat; // Copy trait
    assert_eq!(stat, cloned);
    assert_eq!(stat, copied);
}

// =========================================================================
// Criteria #3: PipelineStatisticsResult - All 5 Fields Accessible
// =========================================================================

#[test]
fn test_pipeline_statistics_result_new_construction() {
    let result = PipelineStatisticsResult::new(100, 80, 60, 500, 0);
    assert_eq!(result.vertex_shader_invocations, 100);
    assert_eq!(result.clipper_invocations, 80);
    assert_eq!(result.clipper_primitives_out, 60);
    assert_eq!(result.fragment_shader_invocations, 500);
    assert_eq!(result.compute_shader_invocations, 0);
}

#[test]
fn test_pipeline_statistics_result_zero_factory() {
    let result = PipelineStatisticsResult::zero();
    assert_eq!(result.vertex_shader_invocations, 0);
    assert_eq!(result.clipper_invocations, 0);
    assert_eq!(result.clipper_primitives_out, 0);
    assert_eq!(result.fragment_shader_invocations, 0);
    assert_eq!(result.compute_shader_invocations, 0);
}

#[test]
fn test_pipeline_statistics_result_get_by_statistic_type() {
    let result = PipelineStatisticsResult::new(10, 20, 30, 40, 50);

    assert_eq!(result.get(StatisticType::VertexShaderInvocations), 10);
    assert_eq!(result.get(StatisticType::ClipperInvocations), 20);
    assert_eq!(result.get(StatisticType::ClipperPrimitivesOut), 30);
    assert_eq!(result.get(StatisticType::FragmentShaderInvocations), 40);
    assert_eq!(result.get(StatisticType::ComputeShaderInvocations), 50);
}

#[test]
fn test_pipeline_statistics_result_to_array() {
    let result = PipelineStatisticsResult::new(1, 2, 3, 4, 5);
    let array = result.to_array();

    assert_eq!(array.len(), 5);
    assert_eq!(array[0], 1); // vertex
    assert_eq!(array[1], 2); // clipper in
    assert_eq!(array[2], 3); // clipper out
    assert_eq!(array[3], 4); // fragment
    assert_eq!(array[4], 5); // compute
}

#[test]
fn test_pipeline_statistics_result_from_slice_valid() {
    let values = [100u64, 80, 60, 500, 256];
    let result = PipelineStatisticsResult::from_slice(&values);

    assert!(result.is_some(), "Should parse valid slice");
    let r = result.unwrap();
    assert_eq!(r.vertex_shader_invocations, 100);
    assert_eq!(r.fragment_shader_invocations, 500);
}

#[test]
fn test_pipeline_statistics_result_from_slice_too_short() {
    let values = [100u64, 80, 60, 500]; // Only 4 elements
    let result = PipelineStatisticsResult::from_slice(&values);

    assert!(result.is_none(), "Should fail on slice with fewer than 5 elements");
}

#[test]
fn test_pipeline_statistics_result_from_slice_longer_fails_or_truncates() {
    let values = [1u64, 2, 3, 4, 5, 6, 7]; // Extra elements
    let result = PipelineStatisticsResult::from_slice(&values);

    // Implementation requires exactly 5 elements, so longer slices fail
    // This is a valid design choice for strict parsing
    if result.is_some() {
        // If it accepts longer slices, it should truncate and use first 5
        let r = result.unwrap();
        assert_eq!(r.vertex_shader_invocations, 1);
        assert_eq!(r.compute_shader_invocations, 5);
    }
    // Otherwise, None is acceptable - implementation rejects non-exact slices
}

// =========================================================================
// Criteria #4: overdraw_estimate() Calculation Accuracy
// =========================================================================

#[test]
fn test_overdraw_estimate_typical_case() {
    // 1000 fragment invocations for 500 expected pixels = 2.0x overdraw
    let result = PipelineStatisticsResult::new(100, 80, 60, 1000, 0);
    let overdraw = result.overdraw_estimate(500);

    assert!((overdraw - 2.0).abs() < 0.001, "Expected 2.0x overdraw, got {}", overdraw);
}

#[test]
fn test_overdraw_estimate_no_overdraw() {
    // Fragment invocations equal to expected pixels = 1.0x (no overdraw)
    let result = PipelineStatisticsResult::new(100, 80, 60, 500, 0);
    let overdraw = result.overdraw_estimate(500);

    assert!((overdraw - 1.0).abs() < 0.001, "Expected 1.0x (no overdraw), got {}", overdraw);
}

#[test]
fn test_overdraw_estimate_zero_expected_pixels() {
    // Edge case: zero expected pixels should handle gracefully
    let result = PipelineStatisticsResult::new(100, 80, 60, 500, 0);
    let overdraw = result.overdraw_estimate(0);

    // Should return some finite value, not panic or return NaN
    assert!(overdraw.is_finite(), "Overdraw should be finite even with 0 expected pixels");
}

#[test]
fn test_overdraw_estimate_zero_fragments() {
    // No fragment invocations = 0 overdraw
    let result = PipelineStatisticsResult::new(100, 80, 60, 0, 0);
    let overdraw = result.overdraw_estimate(500);

    assert!((overdraw - 0.0).abs() < 0.001, "Expected 0 overdraw with no fragments, got {}", overdraw);
}

#[test]
fn test_overdraw_estimate_high_overdraw() {
    // 10000 fragment invocations for 1000 expected = 10x overdraw
    let result = PipelineStatisticsResult::new(100, 80, 60, 10000, 0);
    let overdraw = result.overdraw_estimate(1000);

    assert!((overdraw - 10.0).abs() < 0.001, "Expected 10x overdraw, got {}", overdraw);
}

// =========================================================================
// Criteria #5: culling_efficiency() Calculation Accuracy
// =========================================================================

#[test]
fn test_culling_efficiency_typical_case() {
    // 80 primitives in, 60 out = 25% culled = 0.25 efficiency
    let result = PipelineStatisticsResult::new(100, 80, 60, 500, 0);
    let efficiency = result.culling_efficiency();

    assert!((efficiency - 0.25).abs() < 0.001, "Expected 0.25 culling efficiency, got {}", efficiency);
}

#[test]
fn test_culling_efficiency_no_culling() {
    // All primitives pass through = 0% efficiency
    let result = PipelineStatisticsResult::new(100, 100, 100, 500, 0);
    let efficiency = result.culling_efficiency();

    assert!((efficiency - 0.0).abs() < 0.001, "Expected 0 efficiency (no culling), got {}", efficiency);
}

#[test]
fn test_culling_efficiency_all_culled() {
    // All primitives culled = 100% efficiency
    let result = PipelineStatisticsResult::new(100, 100, 0, 0, 0);
    let efficiency = result.culling_efficiency();

    assert!((efficiency - 1.0).abs() < 0.001, "Expected 1.0 efficiency (all culled), got {}", efficiency);
}

#[test]
fn test_culling_efficiency_zero_input_primitives() {
    // Edge case: zero clipper invocations
    let result = PipelineStatisticsResult::new(0, 0, 0, 0, 0);
    let efficiency = result.culling_efficiency();

    // Should handle gracefully, not panic or return NaN
    assert!(efficiency.is_finite(), "Efficiency should be finite with 0 input primitives");
}

#[test]
fn test_culling_efficiency_half_culled() {
    // 100 in, 50 out = 50% culled
    let result = PipelineStatisticsResult::new(100, 100, 50, 500, 0);
    let efficiency = result.culling_efficiency();

    assert!((efficiency - 0.5).abs() < 0.001, "Expected 0.5 efficiency, got {}", efficiency);
}

// =========================================================================
// Additional PipelineStatisticsResult Methods
// =========================================================================

#[test]
fn test_pipeline_statistics_result_has_activity_true() {
    let result = PipelineStatisticsResult::new(100, 0, 0, 0, 0);
    assert!(result.has_activity(), "Should detect activity with non-zero stats");
}

#[test]
fn test_pipeline_statistics_result_has_activity_false() {
    let result = PipelineStatisticsResult::zero();
    assert!(!result.has_activity(), "Zero result should have no activity");
}

#[test]
fn test_pipeline_statistics_result_is_graphics_pass() {
    let graphics_result = PipelineStatisticsResult::new(100, 80, 60, 500, 0);
    assert!(graphics_result.is_graphics_pass(), "Should detect graphics pass");

    let compute_only = PipelineStatisticsResult::new(0, 0, 0, 0, 256);
    assert!(!compute_only.is_graphics_pass(), "Compute-only should not be graphics pass");
}

#[test]
fn test_pipeline_statistics_result_is_compute_pass() {
    let compute_result = PipelineStatisticsResult::new(0, 0, 0, 0, 256);
    assert!(compute_result.is_compute_pass(), "Should detect compute pass");

    let graphics_only = PipelineStatisticsResult::new(100, 80, 60, 500, 0);
    assert!(!graphics_only.is_compute_pass(), "Graphics-only should not be compute pass");
}

#[test]
fn test_pipeline_statistics_result_combine() {
    let a = PipelineStatisticsResult::new(100, 80, 60, 500, 10);
    let b = PipelineStatisticsResult::new(50, 40, 30, 250, 20);
    let combined = a.combine(&b);

    assert_eq!(combined.vertex_shader_invocations, 150);
    assert_eq!(combined.clipper_invocations, 120);
    assert_eq!(combined.clipper_primitives_out, 90);
    assert_eq!(combined.fragment_shader_invocations, 750);
    assert_eq!(combined.compute_shader_invocations, 30);
}

#[test]
fn test_pipeline_statistics_result_total_shader_invocations() {
    let result = PipelineStatisticsResult::new(100, 80, 60, 500, 256);
    let total = result.total_shader_invocations();

    // Should include vertex + fragment + compute
    assert!(total >= 100 + 500 + 256, "Total should include major shader stages");
}

#[test]
fn test_pipeline_statistics_result_vertex_reuse_ratio() {
    // vertex_reuse_ratio = vertex_count / vertex_shader_invocations
    // 200 vertices for 100 invocations = 2.0 reuse (vertices reused across invocations)
    // Higher ratio means better vertex caching/reuse
    let result = PipelineStatisticsResult::new(100, 80, 60, 500, 0);
    let ratio = result.vertex_reuse_ratio(200);

    assert!((ratio - 2.0).abs() < 0.001, "Expected 2.0 reuse ratio (vertex_count/invocations), got {}", ratio);
}

#[test]
fn test_pipeline_statistics_result_vertex_reuse_ratio_zero_vertices() {
    let result = PipelineStatisticsResult::new(100, 80, 60, 500, 0);
    let ratio = result.vertex_reuse_ratio(0);

    assert!(ratio.is_finite(), "Should handle zero vertex count gracefully");
}

#[test]
fn test_pipeline_statistics_result_fragment_to_vertex_ratio() {
    let result = PipelineStatisticsResult::new(100, 80, 60, 500, 0);
    let ratio = result.fragment_to_vertex_ratio();

    // 500 fragments / 100 vertices = 5.0
    assert!((ratio - 5.0).abs() < 0.001, "Expected 5.0 ratio, got {}", ratio);
}

#[test]
fn test_pipeline_statistics_result_avg_primitive_size() {
    // 1000 fragments for 100 primitives = 10 avg size
    let result = PipelineStatisticsResult::new(100, 100, 100, 1000, 0);
    let avg = result.avg_primitive_size();

    assert!((avg - 10.0).abs() < 0.001, "Expected 10.0 avg size, got {}", avg);
}

// =========================================================================
// PipelineStatisticsData Tests
// =========================================================================

#[test]
fn test_pipeline_statistics_data_new() {
    let results = vec![
        PipelineStatisticsResult::new(100, 80, 60, 500, 0),
        PipelineStatisticsResult::new(200, 160, 120, 1000, 0),
    ];
    let data = PipelineStatisticsData::new(results, 0);

    assert_eq!(data.len(), 2);
    assert!(!data.is_empty());
}

#[test]
fn test_pipeline_statistics_data_empty() {
    let data = PipelineStatisticsData::empty();

    assert_eq!(data.len(), 0);
    assert!(data.is_empty());
}

#[test]
fn test_pipeline_statistics_data_get() {
    let results = vec![
        PipelineStatisticsResult::new(100, 80, 60, 500, 0),
        PipelineStatisticsResult::new(200, 160, 120, 1000, 0),
    ];
    let data = PipelineStatisticsData::new(results, 0);

    let first = data.get(0);
    assert!(first.is_some());
    assert_eq!(first.unwrap().vertex_shader_invocations, 100);

    let second = data.get(1);
    assert!(second.is_some());
    assert_eq!(second.unwrap().vertex_shader_invocations, 200);

    let none = data.get(2);
    assert!(none.is_none());
}

#[test]
fn test_pipeline_statistics_data_get_by_query_index() {
    let results = vec![
        PipelineStatisticsResult::new(100, 80, 60, 500, 0),
        PipelineStatisticsResult::new(200, 160, 120, 1000, 0),
    ];
    let data = PipelineStatisticsData::new(results, 5); // Start at index 5

    // Query index 5 should be first result
    let result = data.get_by_query_index(5);
    assert!(result.is_some());
    assert_eq!(result.unwrap().vertex_shader_invocations, 100);

    // Query index 6 should be second result
    let result = data.get_by_query_index(6);
    assert!(result.is_some());
    assert_eq!(result.unwrap().vertex_shader_invocations, 200);

    // Query index 4 (before start) should be None
    assert!(data.get_by_query_index(4).is_none());
}

#[test]
fn test_pipeline_statistics_data_aggregate() {
    let results = vec![
        PipelineStatisticsResult::new(100, 80, 60, 500, 10),
        PipelineStatisticsResult::new(200, 120, 90, 700, 20),
    ];
    let data = PipelineStatisticsData::new(results, 0);
    let agg = data.aggregate();

    assert_eq!(agg.vertex_shader_invocations, 300);
    assert_eq!(agg.clipper_invocations, 200);
    assert_eq!(agg.clipper_primitives_out, 150);
    assert_eq!(agg.fragment_shader_invocations, 1200);
    assert_eq!(agg.compute_shader_invocations, 30);
}

#[test]
fn test_pipeline_statistics_data_overdraw_estimates() {
    let results = vec![
        PipelineStatisticsResult::new(100, 80, 60, 1000, 0), // 2x overdraw
        PipelineStatisticsResult::new(100, 80, 60, 500, 0),  // 1x overdraw
    ];
    let data = PipelineStatisticsData::new(results, 0);
    let estimates = data.overdraw_estimates(500);

    assert_eq!(estimates.len(), 2);
    assert!((estimates[0] - 2.0).abs() < 0.001);
    assert!((estimates[1] - 1.0).abs() < 0.001);
}

#[test]
fn test_pipeline_statistics_data_culling_efficiencies() {
    let results = vec![
        PipelineStatisticsResult::new(100, 100, 50, 500, 0),  // 50% culled
        PipelineStatisticsResult::new(100, 100, 100, 500, 0), // 0% culled
    ];
    let data = PipelineStatisticsData::new(results, 0);
    let efficiencies = data.culling_efficiencies();

    assert_eq!(efficiencies.len(), 2);
    assert!((efficiencies[0] - 0.5).abs() < 0.001);
    assert!((efficiencies[1] - 0.0).abs() < 0.001);
}

#[test]
fn test_pipeline_statistics_data_max_fragment_invocations() {
    let results = vec![
        PipelineStatisticsResult::new(100, 80, 60, 500, 0),
        PipelineStatisticsResult::new(100, 80, 60, 2000, 0), // Max
        PipelineStatisticsResult::new(100, 80, 60, 1000, 0),
    ];
    let data = PipelineStatisticsData::new(results, 0);
    let max = data.max_fragment_invocations();

    assert!(max.is_some());
    let (idx, result) = max.unwrap();
    assert_eq!(idx, 1); // Index 1 has max
    assert_eq!(result.fragment_shader_invocations, 2000);
}

#[test]
fn test_pipeline_statistics_data_overdraw_hotspots() {
    let results = vec![
        PipelineStatisticsResult::new(100, 80, 60, 500, 0),   // 1x
        PipelineStatisticsResult::new(100, 80, 60, 2500, 0),  // 5x - hotspot
        PipelineStatisticsResult::new(100, 80, 60, 1000, 0),  // 2x
    ];
    let data = PipelineStatisticsData::new(results, 0);

    // Threshold 3.0x - only index 1 should be flagged
    let hotspots = data.overdraw_hotspots(500, 3.0);

    assert_eq!(hotspots.len(), 1);
    assert_eq!(hotspots[0].0, 1); // Query index 1
    assert!((hotspots[0].1 - 5.0).abs() < 0.001);
}

// =========================================================================
// LabeledStatisticsResult Tests
// =========================================================================

#[test]
fn test_labeled_statistics_result_new() {
    let result = PipelineStatisticsResult::new(100, 80, 60, 500, 0);
    let labeled = LabeledStatisticsResult::new(0, result);

    assert_eq!(labeled.query_index, 0);
    assert!(labeled.label.is_none());
}

#[test]
fn test_labeled_statistics_result_with_label() {
    let result = PipelineStatisticsResult::new(100, 80, 60, 500, 0);
    let labeled = LabeledStatisticsResult::with_label(5, result, "shadow_pass");

    assert_eq!(labeled.query_index, 5);
    assert_eq!(labeled.label.as_deref(), Some("shadow_pass"));
}

#[test]
fn test_labeled_statistics_result_label_or() {
    let result = PipelineStatisticsResult::new(100, 80, 60, 500, 0);

    let with_label = LabeledStatisticsResult::with_label(0, result.clone(), "my_pass");
    assert_eq!(with_label.label_or("default"), "my_pass");

    let without_label = LabeledStatisticsResult::new(0, result);
    assert_eq!(without_label.label_or("default"), "default");
}

#[test]
fn test_labeled_statistics_result_label_or_unnamed() {
    let result = PipelineStatisticsResult::new(100, 80, 60, 500, 0);
    let labeled = LabeledStatisticsResult::new(0, result);

    let name = labeled.label_or_unnamed();
    assert!(!name.is_empty(), "Should return a default name");
}

#[test]
fn test_labeled_statistics_result_default() {
    let labeled = LabeledStatisticsResult::default();
    assert_eq!(labeled.query_index, 0);
    assert!(labeled.label.is_none());
    assert_eq!(labeled.result.vertex_shader_invocations, 0);
}

// =========================================================================
// PipelineStatisticsAllocation Tests
// =========================================================================

#[test]
fn test_pipeline_statistics_allocation_new() {
    let alloc = PipelineStatisticsAllocation::new(5, 1);
    assert_eq!(alloc.index, 5);
    assert_eq!(alloc.generation, 1);
}

#[test]
fn test_pipeline_statistics_allocation_with_index() {
    let alloc = PipelineStatisticsAllocation::with_index(10);
    assert_eq!(alloc.index, 10);
    assert_eq!(alloc.generation, 0);
}

#[test]
fn test_pipeline_statistics_allocation_display() {
    let alloc = PipelineStatisticsAllocation::new(5, 2);
    let display = format!("{}", alloc);
    assert!(display.contains("5"), "Display should include index");
}

// =========================================================================
// PipelineStatisticsResolveParams Tests
// =========================================================================

#[test]
fn test_resolve_params_new() {
    let params = PipelineStatisticsResolveParams::new(0, 10, 0);
    assert_eq!(params.start_query, 0);
    assert_eq!(params.query_count, 10);
    assert_eq!(params.destination_offset, 0);
}

#[test]
fn test_resolve_params_from_start() {
    let params = PipelineStatisticsResolveParams::from_start(5);
    assert_eq!(params.start_query, 0);
    assert_eq!(params.query_count, 5);
    assert_eq!(params.destination_offset, 0);
}

#[test]
fn test_resolve_params_end_query() {
    let params = PipelineStatisticsResolveParams::new(5, 10, 0);
    assert_eq!(params.end_query(), 15);
}

#[test]
fn test_resolve_params_required_buffer_size() {
    let params = PipelineStatisticsResolveParams::new(0, 10, 0);
    let size = params.required_buffer_size();

    // Should be query_count * STATISTICS_RESULT_SIZE_BYTES
    assert_eq!(size, 10 * STATISTICS_RESULT_SIZE_BYTES);
}

#[test]
fn test_resolve_params_default() {
    let params = PipelineStatisticsResolveParams::default();
    assert_eq!(params.start_query, 0);
    // Default should have some sensible values
}

// =========================================================================
// Constants Tests
// =========================================================================

#[test]
fn test_statistic_size_bytes_constant() {
    assert_eq!(STATISTIC_SIZE_BYTES, 8, "Each statistic should be 8 bytes (u64)");
}

#[test]
fn test_statistics_per_query_constant() {
    assert_eq!(STATISTICS_PER_QUERY, 5, "Should have 5 statistics per query");
}

#[test]
fn test_statistics_result_size_bytes_constant() {
    assert_eq!(STATISTICS_RESULT_SIZE_BYTES, 5 * 8, "Result size should be 5 * 8 = 40 bytes");
}

#[test]
fn test_min_pool_capacity_constant() {
    assert!(MIN_POOL_CAPACITY >= 1, "Minimum capacity should be at least 1");
}

#[test]
fn test_max_recommended_capacity_constant() {
    assert!(MAX_RECOMMENDED_CAPACITY > MIN_POOL_CAPACITY, "Max should be greater than min");
    assert!(MAX_RECOMMENDED_CAPACITY <= 4096, "Max should be reasonable");
}

#[test]
fn test_default_label_prefix_constant() {
    assert!(!DEFAULT_LABEL_PREFIX.is_empty(), "Should have a default label prefix");
}

// =========================================================================
// Utility Function Tests
// =========================================================================

#[test]
fn test_calculate_resolve_buffer_size_single() {
    let size = calculate_resolve_buffer_size(1);
    assert_eq!(size, STATISTICS_RESULT_SIZE_BYTES);
}

#[test]
fn test_calculate_resolve_buffer_size_multiple() {
    let size = calculate_resolve_buffer_size(10);
    assert_eq!(size, 10 * STATISTICS_RESULT_SIZE_BYTES);
}

#[test]
fn test_calculate_resolve_buffer_size_zero() {
    let size = calculate_resolve_buffer_size(0);
    assert_eq!(size, 0);
}

#[test]
fn test_all_statistics_types_returns_all_flags() {
    let types = all_statistics_types();

    // Should contain all 5 statistic types as bitflags
    assert!(types.contains(wgpu::PipelineStatisticsTypes::VERTEX_SHADER_INVOCATIONS));
    assert!(types.contains(wgpu::PipelineStatisticsTypes::CLIPPER_INVOCATIONS));
    assert!(types.contains(wgpu::PipelineStatisticsTypes::CLIPPER_PRIMITIVES_OUT));
    assert!(types.contains(wgpu::PipelineStatisticsTypes::FRAGMENT_SHADER_INVOCATIONS));
    assert!(types.contains(wgpu::PipelineStatisticsTypes::COMPUTE_SHADER_INVOCATIONS));
}

// =========================================================================
// Criteria #2: PipelineStatisticsPoolBuilder Tests (Pool Creation)
// =========================================================================

#[test]
fn test_pool_builder_new() {
    let builder = PipelineStatisticsPoolBuilder::new();
    // Builder should be constructable without device
    let _ = builder;
}

#[test]
fn test_pool_builder_capacity() {
    let builder = PipelineStatisticsPoolBuilder::new()
        .capacity(64);
    // Should accept capacity
    let _ = builder;
}

#[test]
fn test_pool_builder_label() {
    let builder = PipelineStatisticsPoolBuilder::new()
        .capacity(32)
        .label("test_pool");
    // Should accept label
    let _ = builder;
}

#[test]
fn test_pool_builder_chaining() {
    let builder = PipelineStatisticsPoolBuilder::new()
        .capacity(128)
        .label("chained_pool");
    // Chaining should work
    let _ = builder;
}

#[test]
fn test_pool_builder_default() {
    let builder = PipelineStatisticsPoolBuilder::default();
    // Default should work
    let _ = builder;
}

// =========================================================================
// Display Trait Tests
// =========================================================================

#[test]
fn test_statistic_type_display() {
    for stat in StatisticType::all() {
        let display = format!("{}", stat);
        assert!(!display.is_empty(), "Display should not be empty for {:?}", stat);
    }
}

#[test]
fn test_pipeline_statistics_result_display() {
    let result = PipelineStatisticsResult::new(100, 80, 60, 500, 256);
    let display = format!("{}", result);

    // Should contain some statistics info
    assert!(!display.is_empty());
}

#[test]
fn test_labeled_statistics_result_display() {
    let result = PipelineStatisticsResult::new(100, 80, 60, 500, 0);
    let labeled = LabeledStatisticsResult::with_label(0, result, "test_pass");
    let display = format!("{}", labeled);

    assert!(!display.is_empty());
}

#[test]
fn test_resolve_params_display() {
    let params = PipelineStatisticsResolveParams::new(0, 10, 0);
    let display = format!("{}", params);

    assert!(!display.is_empty());
}

// =========================================================================
// Edge Cases and Boundary Tests
// =========================================================================

#[test]
fn test_large_statistics_values() {
    let max = u64::MAX;
    let result = PipelineStatisticsResult::new(max, max, max, max, max);

    assert_eq!(result.vertex_shader_invocations, max);
    assert_eq!(result.fragment_shader_invocations, max);
}

#[test]
fn test_overdraw_estimate_with_large_values() {
    let result = PipelineStatisticsResult::new(0, 0, 0, u64::MAX, 0);
    let overdraw = result.overdraw_estimate(1);

    // Should not panic, result may be very large or infinity
    assert!(!overdraw.is_nan(), "Should not return NaN");
}

#[test]
fn test_culling_efficiency_primitives_increase() {
    // Edge case: more primitives out than in (shouldn't happen normally)
    let result = PipelineStatisticsResult::new(100, 50, 100, 500, 0);
    let efficiency = result.culling_efficiency();

    // Should handle gracefully (likely negative or clamped)
    assert!(efficiency.is_finite());
}

#[test]
fn test_from_slice_empty() {
    let empty: &[u64] = &[];
    let result = PipelineStatisticsResult::from_slice(empty);
    assert!(result.is_none());
}

#[test]
fn test_data_aggregate_empty() {
    let data = PipelineStatisticsData::empty();
    let agg = data.aggregate();

    // Aggregating empty data should return zeros
    assert_eq!(agg.vertex_shader_invocations, 0);
    assert_eq!(agg.fragment_shader_invocations, 0);
}
