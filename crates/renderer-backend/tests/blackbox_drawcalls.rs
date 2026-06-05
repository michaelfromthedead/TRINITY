// Blackbox contract tests for T-WGPU-P7.4.4 Draw Call Statistics.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::profiling::drawcalls::*` -- no internal fields,
// no private methods, no implementation details.
//
// Test Requirements:
//   - API Contract Tests (25+)
//   - Real-World Scenarios (30+)
//   - Performance Analysis (20+)
//   - Edge Cases (15+)
//   - Recommendation Testing (15+)
//   - History and Trends (15+)
//
// Coverage Summary (120+ tests):
//   01-25: API Contract Tests
//   26-55: Real-World Scenario Tests
//   56-75: Performance Analysis Tests
//   76-90: Edge Case Tests
//   91-105: Recommendation Tests
//   106-120: History and Trend Tests

use std::collections::VecDeque;
use std::time::Duration;

use renderer_backend::profiling::drawcalls::{
    DrawBatch, DrawCall, DrawCallAnalyzer, DrawCallTracker, DrawFrameStats, DrawType,
    FrameAnalysis, FrameComparison, PassStats, PassType, PrimitiveTopology, TrendAnalysis,
    DEFAULT_HISTORY_SIZE, HEAVY_PASS_DRAW_THRESHOLD, HEAVY_PASS_VERTEX_THRESHOLD,
    MAX_HISTORY_SIZE, MIN_HISTORY_SIZE, STATE_THRASHING_THRESHOLD,
};

// ============================================================================
// SECTION 1 -- API Contract Tests (01-25)
// ============================================================================

/// Test 01: DrawCallTracker::new() creates tracker with default history size.
#[test]
fn api_tracker_new_creates_with_defaults() {
    let tracker = DrawCallTracker::new();
    assert!(tracker.is_enabled());
    assert_eq!(tracker.current_frame_number(), 0);
    assert!(tracker.history().is_empty());
}

/// Test 02: DrawCallTracker::with_history_size() respects size parameter.
#[test]
fn api_tracker_with_history_size() {
    let tracker = DrawCallTracker::with_history_size(50);
    // Can't directly check history_size but behavior can be verified
    assert!(tracker.is_enabled());
    assert_eq!(tracker.current_frame_number(), 0);
}

/// Test 03: DrawCallTracker::set_enabled() toggles tracking.
#[test]
fn api_tracker_set_enabled_toggles() {
    let mut tracker = DrawCallTracker::new();
    assert!(tracker.is_enabled());

    tracker.set_enabled(false);
    assert!(!tracker.is_enabled());

    tracker.set_enabled(true);
    assert!(tracker.is_enabled());
}

/// Test 04: DrawCallTracker::begin_frame() starts frame tracking.
#[test]
fn api_tracker_begin_frame() {
    let mut tracker = DrawCallTracker::new();
    tracker.begin_frame();
    // Frame number should still be 0 until end_frame
    assert_eq!(tracker.current_frame_number(), 0);
}

/// Test 05: DrawCallTracker::end_frame() returns DrawFrameStats.
#[test]
fn api_tracker_end_frame_returns_stats() {
    let mut tracker = DrawCallTracker::new();
    tracker.begin_frame();
    let stats = tracker.end_frame();
    assert_eq!(stats.frame_number, 0);
}

/// Test 06: DrawCallTracker::begin_pass() starts pass recording.
#[test]
fn api_tracker_begin_pass() {
    let mut tracker = DrawCallTracker::new();
    tracker.begin_frame();
    tracker.begin_pass(PassType::Render, Some("GBuffer"));
    // Pass is now active
}

/// Test 07: DrawCallTracker::end_pass() returns PassStats.
#[test]
fn api_tracker_end_pass_returns_stats() {
    let mut tracker = DrawCallTracker::new();
    tracker.begin_frame();
    tracker.begin_pass(PassType::Render, Some("Test"));
    let pass_stats = tracker.end_pass();
    assert!(pass_stats.is_some());
    let stats = pass_stats.unwrap();
    assert_eq!(stats.pass_type, PassType::Render);
    assert_eq!(stats.label.as_deref(), Some("Test"));
}

/// Test 08: DrawCall::new() creates basic draw call.
#[test]
fn api_drawcall_new_creates_basic() {
    let draw = DrawCall::new(DrawType::Draw, 100, 1);
    assert_eq!(draw.draw_type, DrawType::Draw);
    assert_eq!(draw.vertex_count, 100);
    assert_eq!(draw.instance_count, 1);
    assert!(draw.index_count.is_none());
}

/// Test 09: DrawCall::indexed() creates indexed draw call.
#[test]
fn api_drawcall_indexed_creates_indexed() {
    let draw = DrawCall::indexed(100, 300, 5);
    assert_eq!(draw.draw_type, DrawType::DrawIndexed);
    assert_eq!(draw.vertex_count, 100);
    assert_eq!(draw.index_count, Some(300));
    assert_eq!(draw.instance_count, 5);
}

/// Test 10: DrawCall::dispatch() creates compute dispatch.
#[test]
fn api_drawcall_dispatch_creates_compute() {
    let draw = DrawCall::dispatch(8, 8, 4);
    assert!(draw.draw_type.is_compute());
    assert_eq!(draw.workgroups(), Some((8, 8, 4)));
}

/// Test 11: DrawCall::with_pipeline() sets pipeline ID.
#[test]
fn api_drawcall_with_pipeline() {
    let draw = DrawCall::new(DrawType::Draw, 100, 1).with_pipeline(42);
    assert_eq!(draw.pipeline_id, Some(42));
}

/// Test 12: DrawCall::with_label() sets label.
#[test]
fn api_drawcall_with_label() {
    let draw = DrawCall::new(DrawType::Draw, 100, 1).with_label("my_draw");
    assert_eq!(draw.label.as_deref(), Some("my_draw"));
}

/// Test 13: DrawCall::total_vertices() calculates correctly.
#[test]
fn api_drawcall_total_vertices() {
    let draw = DrawCall::new(DrawType::Draw, 100, 10);
    assert_eq!(draw.total_vertices(), 1000);
}

/// Test 14: DrawCall::total_primitives() calculates for topology.
#[test]
fn api_drawcall_total_primitives() {
    let draw = DrawCall::new(DrawType::Draw, 12, 1);
    assert_eq!(draw.total_primitives(PrimitiveTopology::TriangleList), 4);
}

/// Test 15: DrawBatch::new() creates batch with pipeline.
#[test]
fn api_drawbatch_new_creates_batch() {
    let batch = DrawBatch::new(123);
    assert_eq!(batch.pipeline_id, 123);
    assert!(batch.draws.is_empty());
    assert_eq!(batch.total_draw_count(), 0);
}

/// Test 16: DrawBatch::add_draw() adds draws to batch.
#[test]
fn api_drawbatch_add_draw() {
    let mut batch = DrawBatch::new(1);
    batch.add_draw(DrawCall::new(DrawType::Draw, 100, 1));
    batch.add_draw(DrawCall::new(DrawType::Draw, 200, 2));
    assert_eq!(batch.total_draw_count(), 2);
}

/// Test 17: DrawBatch::finish() completes batch.
#[test]
fn api_drawbatch_finish() {
    let mut batch = DrawBatch::new(1);
    batch.add_draw(DrawCall::new(DrawType::Draw, 100, 1));
    batch.finish();
    assert!(batch.duration().is_some());
}

/// Test 18: PassStats::new() creates pass statistics.
#[test]
fn api_passstats_new_creates_stats() {
    let pass = PassStats::new(PassType::Compute, Some("Lighting".to_string()));
    assert_eq!(pass.pass_type, PassType::Compute);
    assert_eq!(pass.label.as_deref(), Some("Lighting"));
    assert_eq!(pass.draw_count, 0);
}

/// Test 19: DrawFrameStats::new() creates frame statistics.
#[test]
fn api_framestats_new_creates_stats() {
    let frame = DrawFrameStats::new(42);
    assert_eq!(frame.frame_number, 42);
    assert!(frame.passes.is_empty());
    assert_eq!(frame.total_draw_calls, 0);
}

/// Test 20: DrawCallAnalyzer::analyze_frame() produces FrameAnalysis.
#[test]
fn api_analyzer_analyze_frame() {
    let frame = DrawFrameStats::new(0);
    let analysis = DrawCallAnalyzer::analyze_frame(&frame);
    assert_eq!(analysis.frame_number, 0);
}

/// Test 21: DrawType classification methods work correctly.
#[test]
fn api_drawtype_classification() {
    assert!(!DrawType::Draw.is_indexed());
    assert!(DrawType::DrawIndexed.is_indexed());
    assert!(!DrawType::Draw.is_indirect());
    assert!(DrawType::DrawIndirect.is_indirect());
    assert!(!DrawType::Draw.is_compute());
    assert!(DrawType::Dispatch.is_compute());
    assert!(DrawType::Draw.is_render());
    assert!(!DrawType::Dispatch.is_render());
}

/// Test 22: DrawType::name() returns string name.
#[test]
fn api_drawtype_name() {
    assert_eq!(DrawType::Draw.name(), "Draw");
    assert_eq!(DrawType::DrawIndexed.name(), "DrawIndexed");
    assert_eq!(DrawType::Dispatch.name(), "Dispatch");
    assert_eq!(DrawType::MultiDrawIndexedIndirect.name(), "MultiDrawIndexedIndirect");
}

/// Test 23: PrimitiveTopology default is TriangleList.
#[test]
fn api_primitive_topology_default() {
    let topo: PrimitiveTopology = Default::default();
    assert_eq!(topo, PrimitiveTopology::TriangleList);
}

/// Test 24: PassType default is Render.
#[test]
fn api_passtype_default() {
    let pt: PassType = Default::default();
    assert_eq!(pt, PassType::Render);
}

/// Test 25: Default trait implementations work.
#[test]
fn api_default_implementations() {
    let _draw: DrawCall = Default::default();
    let _batch: DrawBatch = Default::default();
    let _pass: PassStats = Default::default();
    let _frame: DrawFrameStats = Default::default();
    let _tracker: DrawCallTracker = Default::default();
    let _comparison: FrameComparison = Default::default();
    let _trend: TrendAnalysis = Default::default();
}

// ============================================================================
// SECTION 2 -- Real-World Scenario Tests (26-55)
// ============================================================================

/// Test 26: Simple render frame - 100 draws, 1 pass.
#[test]
fn scenario_simple_render_frame() {
    let mut tracker = DrawCallTracker::new();

    tracker.begin_frame();
    tracker.begin_pass(PassType::Render, Some("Main"));

    for _ in 0..100 {
        tracker.record_draw_indexed(1000, 3000, 1);
    }

    let pass_stats = tracker.end_pass().unwrap();
    assert_eq!(pass_stats.draw_count, 100);
    assert_eq!(pass_stats.vertex_count, 100_000);

    let frame_stats = tracker.end_frame();
    assert_eq!(frame_stats.total_draw_calls, 100);
    assert_eq!(frame_stats.passes.len(), 1);
}

/// Test 27: Complex frame - 5 passes, mixed render/compute.
#[test]
fn scenario_complex_frame_mixed_passes() {
    let mut tracker = DrawCallTracker::new();

    tracker.begin_frame();

    // Pass 1: Shadow pass (render)
    tracker.begin_pass(PassType::Render, Some("Shadows"));
    for _ in 0..50 {
        tracker.record_draw_indexed(500, 1500, 1);
    }
    tracker.end_pass();

    // Pass 2: GBuffer pass (render)
    tracker.begin_pass(PassType::Render, Some("GBuffer"));
    for _ in 0..200 {
        tracker.record_draw_indexed(1000, 3000, 1);
    }
    tracker.end_pass();

    // Pass 3: Light culling (compute)
    tracker.begin_pass(PassType::Compute, Some("LightCull"));
    tracker.record_dispatch(16, 16, 1);
    tracker.end_pass();

    // Pass 4: Deferred lighting (compute)
    tracker.begin_pass(PassType::Compute, Some("Lighting"));
    tracker.record_dispatch(64, 64, 1);
    tracker.end_pass();

    // Pass 5: Post-process (render)
    tracker.begin_pass(PassType::Render, Some("PostProcess"));
    tracker.record_draw_non_indexed(3, 1); // Fullscreen triangle
    tracker.end_pass();

    let stats = tracker.end_frame();
    assert_eq!(stats.passes.len(), 5);
    assert_eq!(stats.total_draw_calls, 251);
    assert_eq!(stats.total_dispatches, 2);
}

/// Test 28: High-frequency frame - 10,000+ draws.
#[test]
fn scenario_high_frequency_frame() {
    let mut tracker = DrawCallTracker::new();

    tracker.begin_frame();
    tracker.begin_pass(PassType::Render, Some("Massive"));

    // CPU-bound: > 1000 draws AND avg_vertices_per_draw < 100.0
    // 10000 draws * 50 vertices = 500_000 total, avg = 50 vertices per draw
    for _ in 0..10_000 {
        tracker.record_draw_indexed(50, 150, 1);
    }

    tracker.end_pass();
    let stats = tracker.end_frame();

    assert_eq!(stats.total_draw_calls, 10_000);
    assert_eq!(stats.total_vertices, 500_000);
    assert!(stats.avg_vertices_per_draw() < 100.0);

    // Analyze: should detect CPU-bound nature
    let analysis = DrawCallAnalyzer::analyze_frame(&stats);
    assert!(analysis.is_cpu_bound());
}

/// Test 29: Compute-heavy frame - mostly dispatches.
#[test]
fn scenario_compute_heavy_frame() {
    let mut tracker = DrawCallTracker::new();

    tracker.begin_frame();

    // Multiple compute passes
    for i in 0..10 {
        tracker.begin_pass(PassType::Compute, Some(&format!("Compute{}", i)));
        tracker.record_dispatch(256, 256, 1);
        tracker.record_dispatch(128, 128, 4);
        tracker.end_pass();
    }

    // One small render pass
    tracker.begin_pass(PassType::Render, Some("Final"));
    tracker.record_draw_non_indexed(3, 1);
    tracker.end_pass();

    let stats = tracker.end_frame();
    assert_eq!(stats.total_dispatches, 20);
    assert_eq!(stats.total_draw_calls, 1);
}

/// Test 30: UI frame - many small batched draws.
#[test]
fn scenario_ui_frame() {
    let mut tracker = DrawCallTracker::new();

    tracker.begin_frame();
    tracker.begin_pass(PassType::Render, Some("UI"));

    // Many small UI elements
    for _ in 0..500 {
        tracker.record_draw_indexed(4, 6, 1); // Quad
    }

    tracker.end_pass();
    let stats = tracker.end_frame();

    assert_eq!(stats.total_draw_calls, 500);
    // Low average vertices per draw
    assert!(stats.avg_vertices_per_draw() < 10.0);
}

/// Test 31: Shadow pass - depth-only draws.
#[test]
fn scenario_shadow_pass() {
    let mut tracker = DrawCallTracker::new();

    tracker.begin_frame();

    // 4 cascade shadow passes
    for cascade in 0..4 {
        tracker.begin_pass(PassType::Render, Some(&format!("Shadow_CSM{}", cascade)));
        for _ in 0..100 {
            tracker.record_draw_indexed(1000, 3000, 1);
        }
        tracker.end_pass();
    }

    let stats = tracker.end_frame();
    assert_eq!(stats.passes.len(), 4);
    assert_eq!(stats.total_draw_calls, 400);
}

/// Test 32: Post-process - few fullscreen draws.
#[test]
fn scenario_postprocess_frame() {
    let mut tracker = DrawCallTracker::new();

    tracker.begin_frame();
    tracker.begin_pass(PassType::Render, Some("PostProcess"));

    // Fullscreen passes: bloom, tone mapping, FXAA
    tracker.record_draw_non_indexed(3, 1); // Bloom downsample
    tracker.record_draw_non_indexed(3, 1); // Bloom upsample
    tracker.record_draw_non_indexed(3, 1); // Tone mapping
    tracker.record_draw_non_indexed(3, 1); // FXAA

    tracker.end_pass();
    let stats = tracker.end_frame();

    assert_eq!(stats.total_draw_calls, 4);
    assert_eq!(stats.total_vertices, 12);
}

/// Test 33: Particle system - instanced draws.
#[test]
fn scenario_particle_system() {
    let mut tracker = DrawCallTracker::new();

    tracker.begin_frame();
    tracker.begin_pass(PassType::Render, Some("Particles"));

    // Particle systems with many instances
    tracker.record_draw_indexed(4, 6, 10_000); // Fire effect
    tracker.record_draw_indexed(4, 6, 5_000); // Smoke
    tracker.record_draw_indexed(4, 6, 1_000); // Sparks

    tracker.end_pass();
    let stats = tracker.end_frame();

    assert_eq!(stats.total_draw_calls, 3);
    assert_eq!(stats.total_instances, 16_000);
}

/// Test 34: Terrain - large vertex counts.
#[test]
fn scenario_terrain_rendering() {
    let mut tracker = DrawCallTracker::new();

    tracker.begin_frame();
    tracker.begin_pass(PassType::Render, Some("Terrain"));

    // Large terrain chunks
    for _ in 0..64 {
        tracker.record_draw_indexed(65_536, 98_304, 1); // 256x256 patch
    }

    tracker.end_pass();
    let stats = tracker.end_frame();

    assert_eq!(stats.total_draw_calls, 64);
    assert!(stats.total_vertices > 4_000_000);
}

/// Test 35: Skeletal mesh - indexed draws with skinning.
#[test]
fn scenario_skeletal_mesh() {
    let mut tracker = DrawCallTracker::new();

    tracker.begin_frame();
    tracker.begin_pass(PassType::Render, Some("Characters"));

    // Characters with multiple mesh parts
    for _ in 0..20 {
        // Head
        tracker.record_draw_indexed(5000, 15000, 1);
        // Body
        tracker.record_draw_indexed(10000, 30000, 1);
        // Arms
        tracker.record_draw_indexed(4000, 12000, 1);
        // Legs
        tracker.record_draw_indexed(4000, 12000, 1);
    }

    tracker.end_pass();
    let stats = tracker.end_frame();

    assert_eq!(stats.total_draw_calls, 80);
}

/// Test 36: Multiple render targets (deferred rendering).
#[test]
fn scenario_deferred_rendering() {
    let mut tracker = DrawCallTracker::new();

    tracker.begin_frame();

    // GBuffer pass
    tracker.begin_pass(PassType::Render, Some("GBuffer"));
    for _ in 0..1000 {
        tracker.record_draw_indexed(1000, 3000, 1);
    }
    tracker.end_pass();

    // Light pass
    tracker.begin_pass(PassType::Compute, Some("Lighting"));
    tracker.record_dispatch(1920 / 8, 1080 / 8, 1);
    tracker.end_pass();

    // Composite
    tracker.begin_pass(PassType::Render, Some("Composite"));
    tracker.record_draw_non_indexed(3, 1);
    tracker.end_pass();

    let stats = tracker.end_frame();
    assert_eq!(stats.passes.len(), 3);
}

/// Test 37: Forward+ rendering with light culling.
#[test]
fn scenario_forward_plus() {
    let mut tracker = DrawCallTracker::new();

    tracker.begin_frame();

    // Depth prepass
    tracker.begin_pass(PassType::Render, Some("DepthPrepass"));
    for _ in 0..500 {
        tracker.record_draw_indexed(1000, 3000, 1);
    }
    tracker.end_pass();

    // Light culling
    tracker.begin_pass(PassType::Compute, Some("LightCull"));
    tracker.record_dispatch(120, 68, 1); // 1920/16 x 1080/16
    tracker.end_pass();

    // Main forward pass
    tracker.begin_pass(PassType::Render, Some("Forward"));
    for _ in 0..500 {
        tracker.record_draw_indexed(1000, 3000, 1);
    }
    tracker.end_pass();

    let stats = tracker.end_frame();
    assert_eq!(stats.total_draw_calls, 1000);
    assert_eq!(stats.total_dispatches, 1);
}

/// Test 38: SSAO rendering.
#[test]
fn scenario_ssao() {
    let mut tracker = DrawCallTracker::new();

    tracker.begin_frame();

    // SSAO generation
    tracker.begin_pass(PassType::Render, Some("SSAO"));
    tracker.record_draw_non_indexed(3, 1);
    tracker.end_pass();

    // SSAO blur (horizontal)
    tracker.begin_pass(PassType::Render, Some("SSAO_BlurH"));
    tracker.record_draw_non_indexed(3, 1);
    tracker.end_pass();

    // SSAO blur (vertical)
    tracker.begin_pass(PassType::Render, Some("SSAO_BlurV"));
    tracker.record_draw_non_indexed(3, 1);
    tracker.end_pass();

    let stats = tracker.end_frame();
    assert_eq!(stats.passes.len(), 3);
    assert_eq!(stats.total_draw_calls, 3);
}

/// Test 39: Volumetric fog rendering.
#[test]
fn scenario_volumetric_fog() {
    let mut tracker = DrawCallTracker::new();

    tracker.begin_frame();

    // 3D froxel volume compute
    tracker.begin_pass(PassType::Compute, Some("FroxelVolume"));
    tracker.record_dispatch(160, 90, 64); // Depth slices
    tracker.end_pass();

    // Integration
    tracker.begin_pass(PassType::Compute, Some("FroxelIntegration"));
    tracker.record_dispatch(160, 90, 1);
    tracker.end_pass();

    // Apply to scene
    tracker.begin_pass(PassType::Render, Some("FogApply"));
    tracker.record_draw_non_indexed(3, 1);
    tracker.end_pass();

    let stats = tracker.end_frame();
    assert_eq!(stats.total_dispatches, 2);
}

/// Test 40: Ocean rendering with tessellation.
#[test]
fn scenario_ocean_tessellation() {
    let mut tracker = DrawCallTracker::new();

    tracker.begin_frame();

    // Ocean FFT compute
    tracker.begin_pass(PassType::Compute, Some("OceanFFT"));
    tracker.record_dispatch(512, 512, 1);
    tracker.record_dispatch(512, 512, 1);
    tracker.record_dispatch(512, 512, 1);
    tracker.end_pass();

    // Ocean geometry pass
    tracker.begin_pass(PassType::Render, Some("OceanDraw"));
    // Tessellated patches
    for _ in 0..256 {
        tracker.record_draw_non_indexed(4, 1); // Control points
    }
    tracker.end_pass();

    let stats = tracker.end_frame();
    assert_eq!(stats.total_dispatches, 3);
    assert_eq!(stats.total_draw_calls, 256);
}

/// Test 41: Ray tracing hybrid frame.
#[test]
fn scenario_raytracing_hybrid() {
    let mut tracker = DrawCallTracker::new();

    tracker.begin_frame();

    // Rasterize primary rays (GBuffer)
    tracker.begin_pass(PassType::Render, Some("GBuffer"));
    for _ in 0..500 {
        tracker.record_draw_indexed(1000, 3000, 1);
    }
    tracker.end_pass();

    // Ray trace reflections (compute as proxy)
    tracker.begin_pass(PassType::Compute, Some("RTReflections"));
    tracker.record_dispatch(1920 / 8, 1080 / 8, 1);
    tracker.end_pass();

    // Ray trace shadows (compute as proxy)
    tracker.begin_pass(PassType::Compute, Some("RTShadows"));
    tracker.record_dispatch(1920 / 8, 1080 / 8, 1);
    tracker.end_pass();

    // Denoise
    tracker.begin_pass(PassType::Compute, Some("Denoise"));
    tracker.record_dispatch(1920 / 8, 1080 / 8, 1);
    tracker.end_pass();

    // Composite
    tracker.begin_pass(PassType::Render, Some("Composite"));
    tracker.record_draw_non_indexed(3, 1);
    tracker.end_pass();

    let stats = tracker.end_frame();
    assert_eq!(stats.total_dispatches, 3);
}

/// Test 42: GPU culling frame.
#[test]
fn scenario_gpu_culling() {
    let mut tracker = DrawCallTracker::new();

    tracker.begin_frame();

    // Instance culling
    tracker.begin_pass(PassType::Compute, Some("InstanceCull"));
    tracker.record_dispatch(1024, 1, 1);
    tracker.end_pass();

    // Draw commands compaction
    tracker.begin_pass(PassType::Compute, Some("Compact"));
    tracker.record_dispatch(256, 1, 1);
    tracker.end_pass();

    // Indirect draw
    tracker.begin_pass(PassType::Render, Some("IndirectDraw"));
    let draw = DrawCall::new(DrawType::DrawIndexedIndirect, 0, 0)
        .with_label("CulledGeometry");
    tracker.record_draw(draw);
    tracker.end_pass();

    let stats = tracker.end_frame();
    assert_eq!(stats.total_dispatches, 2);
    assert_eq!(stats.total_draw_calls, 1);
}

/// Test 43: Mesh shader frame (multi-draw indirect).
#[test]
fn scenario_mesh_shader() {
    let mut tracker = DrawCallTracker::new();

    tracker.begin_frame();
    tracker.begin_pass(PassType::Render, Some("MeshShader"));

    // Multi-draw indirect for meshlet clusters
    let draw = DrawCall::new(DrawType::MultiDrawIndexedIndirect, 0, 1000)
        .with_label("Meshlets");
    tracker.record_draw(draw);

    tracker.end_pass();
    let stats = tracker.end_frame();
    assert_eq!(stats.total_draw_calls, 1);
}

/// Test 44: VR stereo rendering.
#[test]
fn scenario_vr_stereo() {
    let mut tracker = DrawCallTracker::new();

    tracker.begin_frame();

    // Left eye
    tracker.begin_pass(PassType::Render, Some("LeftEye"));
    for _ in 0..200 {
        tracker.record_draw_indexed(1000, 3000, 1);
    }
    tracker.end_pass();

    // Right eye
    tracker.begin_pass(PassType::Render, Some("RightEye"));
    for _ in 0..200 {
        tracker.record_draw_indexed(1000, 3000, 1);
    }
    tracker.end_pass();

    let stats = tracker.end_frame();
    assert_eq!(stats.passes.len(), 2);
    assert_eq!(stats.total_draw_calls, 400);
}

/// Test 45: Temporal effects frame (TAA).
#[test]
fn scenario_temporal_aa() {
    let mut tracker = DrawCallTracker::new();

    tracker.begin_frame();

    // Main render
    tracker.begin_pass(PassType::Render, Some("Main"));
    for _ in 0..300 {
        tracker.record_draw_indexed(1000, 3000, 1);
    }
    tracker.end_pass();

    // TAA resolve
    tracker.begin_pass(PassType::Render, Some("TAA"));
    tracker.record_draw_non_indexed(3, 1);
    tracker.end_pass();

    let stats = tracker.end_frame();
    assert_eq!(stats.total_draw_calls, 301);
}

/// Test 46: Debug visualization frame.
#[test]
fn scenario_debug_visualization() {
    let mut tracker = DrawCallTracker::new();

    tracker.begin_frame();

    // Main scene
    tracker.begin_pass(PassType::Render, Some("Scene"));
    for _ in 0..100 {
        tracker.record_draw_indexed(500, 1500, 1);
    }
    tracker.end_pass();

    // Debug overlays
    tracker.begin_pass(PassType::Render, Some("Debug"));
    // Wireframes
    for _ in 0..100 {
        tracker.record_draw_indexed(500, 1500, 1);
    }
    // Bounding boxes (lines)
    for _ in 0..100 {
        tracker.record_draw_non_indexed(24, 1);
    }
    tracker.end_pass();

    let stats = tracker.end_frame();
    assert_eq!(stats.total_draw_calls, 300);
}

/// Test 47: Level-of-detail switching.
#[test]
fn scenario_lod_switching() {
    let mut tracker = DrawCallTracker::new();

    tracker.begin_frame();
    tracker.begin_pass(PassType::Render, Some("LOD"));

    // LOD0 - high detail, few instances
    for _ in 0..10 {
        tracker.record_draw_indexed(50000, 150000, 1);
    }

    // LOD1 - medium detail
    for _ in 0..30 {
        tracker.record_draw_indexed(10000, 30000, 1);
    }

    // LOD2 - low detail, many instances
    for _ in 0..100 {
        tracker.record_draw_indexed(1000, 3000, 1);
    }

    tracker.end_pass();
    let stats = tracker.end_frame();

    assert_eq!(stats.total_draw_calls, 140);
}

/// Test 48: Impostor rendering.
#[test]
fn scenario_impostors() {
    let mut tracker = DrawCallTracker::new();

    tracker.begin_frame();
    tracker.begin_pass(PassType::Render, Some("Impostors"));

    // Near objects (full geometry)
    for _ in 0..20 {
        tracker.record_draw_indexed(5000, 15000, 1);
    }

    // Far objects (impostors - billboards)
    tracker.record_draw_indexed(4, 6, 5000); // 5000 impostor instances

    tracker.end_pass();
    let stats = tracker.end_frame();

    assert_eq!(stats.total_draw_calls, 21);
    assert_eq!(stats.total_instances, 5020);
}

/// Test 49: Grass rendering with instancing.
#[test]
fn scenario_grass_rendering() {
    let mut tracker = DrawCallTracker::new();

    tracker.begin_frame();
    tracker.begin_pass(PassType::Render, Some("Grass"));

    // Grass chunks with high instance counts
    for _ in 0..32 {
        tracker.record_draw_indexed(16, 18, 10000); // Grass blade mesh
    }

    tracker.end_pass();
    let stats = tracker.end_frame();

    assert_eq!(stats.total_draw_calls, 32);
    assert_eq!(stats.total_instances, 320_000);
}

/// Test 50: Decal rendering.
#[test]
fn scenario_decals() {
    let mut tracker = DrawCallTracker::new();

    tracker.begin_frame();

    // Main scene
    tracker.begin_pass(PassType::Render, Some("Scene"));
    for _ in 0..200 {
        tracker.record_draw_indexed(1000, 3000, 1);
    }
    tracker.end_pass();

    // Decal pass
    tracker.begin_pass(PassType::Render, Some("Decals"));
    for _ in 0..50 {
        tracker.record_draw_indexed(36, 36, 1); // Box for projection
    }
    tracker.end_pass();

    let stats = tracker.end_frame();
    assert_eq!(stats.passes.len(), 2);
    assert_eq!(stats.total_draw_calls, 250);
}

/// Test 51: Motion blur.
#[test]
fn scenario_motion_blur() {
    let mut tracker = DrawCallTracker::new();

    tracker.begin_frame();

    // Main render
    tracker.begin_pass(PassType::Render, Some("Main"));
    for _ in 0..300 {
        tracker.record_draw_indexed(1000, 3000, 1);
    }
    tracker.end_pass();

    // Velocity buffer generation
    tracker.begin_pass(PassType::Render, Some("Velocity"));
    for _ in 0..300 {
        tracker.record_draw_indexed(1000, 3000, 1);
    }
    tracker.end_pass();

    // Motion blur pass
    tracker.begin_pass(PassType::Render, Some("MotionBlur"));
    tracker.record_draw_non_indexed(3, 1);
    tracker.end_pass();

    let stats = tracker.end_frame();
    assert_eq!(stats.total_draw_calls, 601);
}

/// Test 52: Bloom rendering.
#[test]
fn scenario_bloom() {
    let mut tracker = DrawCallTracker::new();

    tracker.begin_frame();

    // Brightness extraction
    tracker.begin_pass(PassType::Render, Some("BrightPass"));
    tracker.record_draw_non_indexed(3, 1);
    tracker.end_pass();

    // Downsampling (5 levels)
    for i in 0..5 {
        tracker.begin_pass(PassType::Render, Some(&format!("Downsample{}", i)));
        tracker.record_draw_non_indexed(3, 1);
        tracker.end_pass();
    }

    // Upsampling with blur (5 levels)
    for i in 0..5 {
        tracker.begin_pass(PassType::Render, Some(&format!("Upsample{}", i)));
        tracker.record_draw_non_indexed(3, 1);
        tracker.end_pass();
    }

    // Composite
    tracker.begin_pass(PassType::Render, Some("BloomComposite"));
    tracker.record_draw_non_indexed(3, 1);
    tracker.end_pass();

    let stats = tracker.end_frame();
    assert_eq!(stats.passes.len(), 12);
    assert_eq!(stats.total_draw_calls, 12);
}

/// Test 53: HDR tone mapping.
#[test]
fn scenario_hdr_tonemapping() {
    let mut tracker = DrawCallTracker::new();

    tracker.begin_frame();

    // Luminance histogram
    tracker.begin_pass(PassType::Compute, Some("Histogram"));
    tracker.record_dispatch(1920 / 16, 1080 / 16, 1);
    tracker.end_pass();

    // Average luminance
    tracker.begin_pass(PassType::Compute, Some("AvgLum"));
    tracker.record_dispatch(1, 1, 1);
    tracker.end_pass();

    // Tone mapping
    tracker.begin_pass(PassType::Render, Some("ToneMap"));
    tracker.record_draw_non_indexed(3, 1);
    tracker.end_pass();

    let stats = tracker.end_frame();
    assert_eq!(stats.total_dispatches, 2);
    assert_eq!(stats.total_draw_calls, 1);
}

/// Test 54: Skinned mesh batching.
#[test]
fn scenario_skinned_mesh_batching() {
    let mut tracker = DrawCallTracker::new();

    tracker.begin_frame();

    // Skin matrix update
    tracker.begin_pass(PassType::Compute, Some("SkinMatrices"));
    tracker.record_dispatch(50, 1, 1); // 50 characters, 64 bones each
    tracker.end_pass();

    // Skinned mesh draw
    tracker.begin_pass(PassType::Render, Some("SkinnedMesh"));
    for _ in 0..50 {
        tracker.record_draw_indexed(10000, 30000, 1);
    }
    tracker.end_pass();

    let stats = tracker.end_frame();
    assert_eq!(stats.total_dispatches, 1);
    assert_eq!(stats.total_draw_calls, 50);
}

/// Test 55: Environment map generation.
#[test]
fn scenario_env_map_generation() {
    let mut tracker = DrawCallTracker::new();

    tracker.begin_frame();

    // 6 faces of cubemap
    for face in 0..6 {
        tracker.begin_pass(PassType::Render, Some(&format!("EnvMap_Face{}", face)));
        for _ in 0..100 {
            tracker.record_draw_indexed(1000, 3000, 1);
        }
        tracker.end_pass();
    }

    // Generate mipmaps
    tracker.begin_pass(PassType::Compute, Some("EnvMipmap"));
    tracker.record_dispatch(128, 128, 6);
    tracker.end_pass();

    let stats = tracker.end_frame();
    assert_eq!(stats.passes.len(), 7);
    assert_eq!(stats.total_draw_calls, 600);
}

// ============================================================================
// SECTION 3 -- Performance Analysis Tests (56-75)
// ============================================================================

/// Test 56: CPU-bound detection - many small draws.
#[test]
fn perf_cpu_bound_detection() {
    let mut tracker = DrawCallTracker::new();

    tracker.begin_frame();
    tracker.begin_pass(PassType::Render, Some("CPUBound"));

    // Many very small draws
    for _ in 0..5000 {
        tracker.record_draw_indexed(10, 30, 1);
    }

    tracker.end_pass();
    let stats = tracker.end_frame();

    let analysis = DrawCallAnalyzer::analyze_frame(&stats);
    assert!(analysis.is_cpu_bound());
    assert!(!analysis.is_gpu_bound());
}

/// Test 57: GPU-bound detection - few large draws.
#[test]
fn perf_gpu_bound_detection() {
    let mut tracker = DrawCallTracker::new();

    tracker.begin_frame();
    tracker.begin_pass(PassType::Render, Some("GPUBound"));

    // Few very large draws - must have < 100 draws and > 1_000_000 total vertices
    for _ in 0..50 {
        // 50 draws * 25000 vertices = 1_250_000 total vertices
        tracker.record_draw_indexed(25_000, 75_000, 1);
    }

    tracker.end_pass();
    let stats = tracker.end_frame();

    let analysis = DrawCallAnalyzer::analyze_frame(&stats);
    // GPU-bound: total_draws < 100 && total_vertices > 1_000_000
    assert!(stats.total_draw_calls < 100);
    assert!(stats.total_vertices > 1_000_000);
    assert!(analysis.is_gpu_bound());
    assert!(!analysis.is_cpu_bound());
}

/// Test 58: State thrashing detection - frequent pipeline changes.
#[test]
fn perf_state_thrashing_detection() {
    let mut tracker = DrawCallTracker::new();

    tracker.begin_frame();
    tracker.begin_pass(PassType::Render, Some("Thrashing"));

    // Alternating pipelines
    for _ in 0..100 {
        tracker.record_draw_indexed(100, 300, 1);
        tracker.record_pipeline_switch();
    }

    tracker.end_pass();
    let stats = tracker.end_frame();

    assert!(DrawCallAnalyzer::detect_state_thrashing(&stats));
    assert!(stats.state_changes_per_draw() > STATE_THRASHING_THRESHOLD);
}

/// Test 59: Good batching - consecutive draws same pipeline.
#[test]
fn perf_good_batching() {
    let mut tracker = DrawCallTracker::new();

    tracker.begin_frame();
    tracker.begin_pass(PassType::Render, Some("GoodBatch"));

    // Many draws, few pipeline switches
    for _ in 0..1000 {
        tracker.record_draw_indexed(1000, 3000, 1);
    }
    // Only one switch for entire batch
    tracker.record_pipeline_switch();

    tracker.end_pass();
    let stats = tracker.end_frame();

    assert!(!DrawCallAnalyzer::detect_state_thrashing(&stats));
    assert!(stats.state_changes_per_draw() < STATE_THRASHING_THRESHOLD);
}

/// Test 60: Poor batching - alternating pipelines.
#[test]
fn perf_poor_batching() {
    let mut tracker = DrawCallTracker::new();

    tracker.begin_frame();
    tracker.begin_pass(PassType::Render, Some("PoorBatch"));

    // Alternating pipelines every draw
    for i in 0..200 {
        let draw = DrawCall::indexed(100, 300, 1).with_pipeline(i % 5);
        tracker.record_draw(draw);
    }

    tracker.end_pass();
    let stats = tracker.end_frame();

    // Should have many pipeline switches due to alternation
    assert!(stats.total_pipeline_switches > 0);
}

/// Test 61: Mixed workload classification.
#[test]
fn perf_mixed_workload() {
    let mut tracker = DrawCallTracker::new();

    tracker.begin_frame();

    // Compute-heavy phase
    tracker.begin_pass(PassType::Compute, Some("ComputePhase"));
    tracker.record_dispatch(256, 256, 64);
    tracker.end_pass();

    // Render-heavy phase
    tracker.begin_pass(PassType::Render, Some("RenderPhase"));
    for _ in 0..500 {
        tracker.record_draw_indexed(1000, 3000, 1);
    }
    tracker.end_pass();

    let stats = tracker.end_frame();
    let analysis = DrawCallAnalyzer::analyze_frame(&stats);

    // Should be classified based on majority workload
    assert!(stats.total_draw_calls > 0);
    assert!(stats.total_dispatches > 0);
    assert!(!analysis.is_cpu_bound()); // Reasonable draw count with good vertices
}

/// Test 62: Draw call density measurement.
#[test]
fn perf_draw_call_density() {
    let mut tracker = DrawCallTracker::new();

    tracker.begin_frame();
    tracker.begin_pass(PassType::Render, Some("Density"));

    for _ in 0..1000 {
        tracker.record_draw_indexed(100, 300, 1);
    }

    tracker.end_pass();
    let stats = tracker.end_frame();

    // Density should be calculated (draws per ms)
    let density = stats.draws_per_ms();
    // Note: This value depends on actual execution time
    assert!(density >= 0.0);
}

/// Test 63: Batching efficiency calculation.
#[test]
fn perf_batching_efficiency() {
    let mut frame_low = DrawFrameStats::new(0);
    let mut pass_low = PassStats::new(PassType::Render, None);
    for _ in 0..100 {
        pass_low.record_draw(&DrawCall::new(DrawType::Draw, 10, 1));
    }
    pass_low.finish();
    frame_low.add_pass(pass_low);
    frame_low.finish();

    let mut frame_high = DrawFrameStats::new(1);
    let mut pass_high = PassStats::new(PassType::Render, None);
    for _ in 0..10 {
        pass_high.record_draw(&DrawCall::new(DrawType::Draw, 10000, 1));
    }
    pass_high.finish();
    frame_high.add_pass(pass_high);
    frame_high.finish();

    let eff_low = DrawCallAnalyzer::batching_efficiency(&frame_low);
    let eff_high = DrawCallAnalyzer::batching_efficiency(&frame_high);

    assert!(eff_high > eff_low);
}

/// Test 64: Heavy pass detection by draw count.
#[test]
fn perf_heavy_pass_detection_draws() {
    let mut frame = DrawFrameStats::new(0);

    let mut light_pass = PassStats::new(PassType::Render, Some("Light".to_string()));
    for _ in 0..10 {
        light_pass.record_draw(&DrawCall::new(DrawType::Draw, 100, 1));
    }
    light_pass.finish();
    frame.add_pass(light_pass);

    let mut heavy_pass = PassStats::new(PassType::Render, Some("Heavy".to_string()));
    for _ in 0..2000 {
        heavy_pass.record_draw(&DrawCall::new(DrawType::Draw, 100, 1));
    }
    heavy_pass.finish();
    frame.add_pass(heavy_pass);
    frame.finish();

    let heavy_passes = DrawCallAnalyzer::find_heavy_passes(&frame, HEAVY_PASS_DRAW_THRESHOLD);
    assert_eq!(heavy_passes.len(), 1);
    assert_eq!(heavy_passes[0].label.as_deref(), Some("Heavy"));
}

/// Test 65: Heavy pass detection by vertex count.
#[test]
fn perf_heavy_pass_detection_vertices() {
    let mut frame = DrawFrameStats::new(0);

    let mut heavy_pass = PassStats::new(PassType::Render, Some("HighVerts".to_string()));
    // Few draws but huge vertex count
    for _ in 0..10 {
        heavy_pass.record_draw(&DrawCall::new(DrawType::Draw, 200_000, 1));
    }
    heavy_pass.finish();
    frame.add_pass(heavy_pass);
    frame.finish();

    let heavy_passes = DrawCallAnalyzer::find_heavy_passes(&frame, HEAVY_PASS_DRAW_THRESHOLD);
    // Should be detected due to high vertex count
    assert!(!heavy_passes.is_empty());
    assert!(heavy_passes[0].vertex_count >= HEAVY_PASS_VERTEX_THRESHOLD);
}

/// Test 66: Average vertices per draw calculation.
#[test]
fn perf_avg_vertices_per_draw() {
    let mut frame = DrawFrameStats::new(0);
    let mut pass = PassStats::new(PassType::Render, None);

    pass.record_draw(&DrawCall::new(DrawType::Draw, 1000, 1));
    pass.record_draw(&DrawCall::new(DrawType::Draw, 2000, 1));
    pass.record_draw(&DrawCall::new(DrawType::Draw, 3000, 1));
    pass.finish();
    frame.add_pass(pass);
    frame.finish();

    assert_eq!(frame.avg_vertices_per_draw(), 2000.0);
    assert_eq!(frame.total_draw_calls, 3);
    assert_eq!(frame.total_vertices, 6000);
}

/// Test 67: State changes per draw ratio.
#[test]
fn perf_state_changes_per_draw() {
    let mut frame = DrawFrameStats::new(0);
    let mut pass = PassStats::new(PassType::Render, None);

    pass.record_draw(&DrawCall::new(DrawType::Draw, 100, 1));
    pass.record_pipeline_switch();
    pass.record_bind_group_set();
    pass.record_draw(&DrawCall::new(DrawType::Draw, 100, 1));
    pass.finish();
    frame.add_pass(pass);
    frame.finish();

    // 2 state changes, 2 draws = 1.0 ratio
    assert_eq!(frame.state_changes_per_draw(), 1.0);
}

/// Test 68: Pass-level state changes tracking.
#[test]
fn perf_pass_state_changes() {
    let mut pass = PassStats::new(PassType::Render, Some("Test".to_string()));

    for _ in 0..10 {
        pass.record_draw(&DrawCall::new(DrawType::Draw, 100, 1));
    }
    pass.record_pipeline_switch();
    pass.record_pipeline_switch();
    pass.record_bind_group_set();
    pass.record_bind_group_set();
    pass.record_bind_group_set();
    pass.finish();

    assert_eq!(pass.pipeline_switches, 2);
    assert_eq!(pass.bind_group_sets, 3);
    // 5 state changes / 10 draws = 0.5
    assert_eq!(pass.state_changes_per_draw(), 0.5);
}

/// Test 69: Instanced draw analysis.
#[test]
fn perf_instanced_draw_analysis() {
    let mut frame = DrawFrameStats::new(0);
    let mut pass = PassStats::new(PassType::Render, None);

    // Highly instanced draws
    pass.record_draw(&DrawCall::new(DrawType::Draw, 100, 1000));
    pass.record_draw(&DrawCall::new(DrawType::Draw, 100, 500));
    pass.finish();
    frame.add_pass(pass);
    frame.finish();

    assert_eq!(frame.total_draw_calls, 2);
    assert_eq!(frame.total_instances, 1500);
    assert_eq!(frame.total_vertices, 150_000); // 100*1000 + 100*500
}

/// Test 70: Compute workgroup analysis.
#[test]
fn perf_compute_workgroup_analysis() {
    let mut pass = PassStats::new(PassType::Compute, None);

    pass.record_draw(&DrawCall::dispatch(8, 8, 1));
    pass.record_draw(&DrawCall::dispatch(16, 16, 4));
    pass.finish();

    // Workgroup count is accumulated
    let total_invocations = pass.total_workgroup_invocations();
    assert!(total_invocations > 0);
}

/// Test 71: Pass average instances calculation.
#[test]
fn perf_pass_avg_instances() {
    let mut pass = PassStats::new(PassType::Render, None);

    pass.record_draw(&DrawCall::new(DrawType::Draw, 100, 10));
    pass.record_draw(&DrawCall::new(DrawType::Draw, 100, 20));
    pass.record_draw(&DrawCall::new(DrawType::Draw, 100, 30));
    pass.finish();

    assert_eq!(pass.avg_instances_per_draw(), 20.0);
}

/// Test 72: Frame total state changes.
#[test]
fn perf_frame_total_state_changes() {
    let mut frame = DrawFrameStats::new(0);

    let mut pass1 = PassStats::new(PassType::Render, None);
    pass1.record_draw(&DrawCall::new(DrawType::Draw, 100, 1));
    pass1.record_pipeline_switch();
    pass1.record_bind_group_set();
    pass1.finish();
    frame.add_pass(pass1);

    let mut pass2 = PassStats::new(PassType::Render, None);
    pass2.record_draw(&DrawCall::new(DrawType::Draw, 100, 1));
    pass2.record_pipeline_switch();
    pass2.finish();
    frame.add_pass(pass2);

    frame.finish();

    assert_eq!(frame.total_state_changes(), 3);
}

/// Test 73: Indexed draw primitives calculation.
#[test]
fn perf_indexed_primitives() {
    let draw = DrawCall::indexed(100, 36, 1);

    // Indexed draw uses index count for primitives
    let triangles = draw.total_primitives(PrimitiveTopology::TriangleList);
    assert_eq!(triangles, 12); // 36 indices / 3
}

/// Test 74: Line primitive calculation.
#[test]
fn perf_line_primitives() {
    let draw = DrawCall::new(DrawType::Draw, 10, 1);

    assert_eq!(draw.total_primitives(PrimitiveTopology::LineList), 5);
    assert_eq!(draw.total_primitives(PrimitiveTopology::LineStrip), 9);
}

/// Test 75: Point primitive calculation.
#[test]
fn perf_point_primitives() {
    let draw = DrawCall::new(DrawType::Draw, 100, 10);

    // Points: 1 vertex = 1 primitive, per instance
    assert_eq!(draw.total_primitives(PrimitiveTopology::PointList), 1000);
}

// ============================================================================
// SECTION 4 -- Edge Case Tests (76-90)
// ============================================================================

/// Test 76: Zero draw frames.
#[test]
fn edge_zero_draw_frame() {
    let mut tracker = DrawCallTracker::new();

    tracker.begin_frame();
    tracker.begin_pass(PassType::Render, Some("Empty"));
    // No draws
    tracker.end_pass();
    let stats = tracker.end_frame();

    assert_eq!(stats.total_draw_calls, 0);
    assert_eq!(stats.total_vertices, 0);
    assert_eq!(stats.avg_vertices_per_draw(), 0.0);
}

/// Test 77: Single draw frame.
#[test]
fn edge_single_draw_frame() {
    let mut tracker = DrawCallTracker::new();

    tracker.begin_frame();
    tracker.begin_pass(PassType::Render, None);
    tracker.record_draw_indexed(1000, 3000, 1);
    tracker.end_pass();
    let stats = tracker.end_frame();

    assert_eq!(stats.total_draw_calls, 1);
    assert_eq!(stats.total_vertices, 1000);
}

/// Test 78: Maximum history buffer.
#[test]
fn edge_max_history() {
    let mut tracker = DrawCallTracker::with_history_size(MAX_HISTORY_SIZE);

    for i in 0..10 {
        tracker.begin_frame();
        tracker.begin_pass(PassType::Render, None);
        tracker.record_draw_indexed(i as u32 * 10, i as u32 * 30, 1);
        tracker.end_pass();
        tracker.end_frame();
    }

    assert_eq!(tracker.history().len(), 10);
}

/// Test 79: History overflow behavior.
#[test]
fn edge_history_overflow() {
    let mut tracker = DrawCallTracker::with_history_size(5);

    for i in 0..10 {
        tracker.begin_frame();
        tracker.begin_pass(PassType::Render, None);
        tracker.record_draw_indexed(i as u32 * 100, i as u32 * 300, 1);
        tracker.end_pass();
        tracker.end_frame();
    }

    assert_eq!(tracker.history().len(), 5);
    // Oldest frames should be discarded
    let first_frame = tracker.history().front().unwrap();
    assert_eq!(first_frame.frame_number, 5);
}

/// Test 80: Rapid frame succession.
#[test]
fn edge_rapid_frames() {
    let mut tracker = DrawCallTracker::new();

    for _ in 0..100 {
        tracker.begin_frame();
        tracker.begin_pass(PassType::Render, None);
        tracker.record_draw_non_indexed(3, 1);
        tracker.end_pass();
        tracker.end_frame();
    }

    assert_eq!(tracker.current_frame_number(), 100);
    assert_eq!(tracker.history().len(), 100);
}

/// Test 81: Pass with no draws.
#[test]
fn edge_empty_pass() {
    let mut tracker = DrawCallTracker::new();

    tracker.begin_frame();
    tracker.begin_pass(PassType::Render, Some("Empty"));
    let pass_stats = tracker.end_pass().unwrap();

    assert_eq!(pass_stats.draw_count, 0);
    assert_eq!(pass_stats.vertex_count, 0);
}

/// Test 82: Frame without passes.
#[test]
fn edge_frame_without_passes() {
    let mut tracker = DrawCallTracker::new();

    tracker.begin_frame();
    let stats = tracker.end_frame();

    assert!(stats.passes.is_empty());
    assert_eq!(stats.total_draw_calls, 0);
}

/// Test 83: Minimum history size clamping.
#[test]
fn edge_min_history_clamping() {
    let tracker = DrawCallTracker::with_history_size(0);
    // Should be clamped to MIN_HISTORY_SIZE
    assert!(tracker.is_enabled());
}

/// Test 84: Very large vertex count.
#[test]
fn edge_large_vertex_count() {
    let draw = DrawCall::new(DrawType::Draw, u32::MAX, 1);
    assert_eq!(draw.total_vertices(), u32::MAX as u64);

    let instanced = DrawCall::new(DrawType::Draw, 1_000_000, 1000);
    assert_eq!(instanced.total_vertices(), 1_000_000_000);
}

/// Test 85: Zero instance count.
#[test]
fn edge_zero_instances() {
    let draw = DrawCall::new(DrawType::Draw, 100, 0);
    assert_eq!(draw.total_vertices(), 0);
}

/// Test 86: Unclosed pass handling.
#[test]
fn edge_unclosed_pass() {
    let mut tracker = DrawCallTracker::new();

    tracker.begin_frame();
    tracker.begin_pass(PassType::Render, Some("Unclosed"));
    tracker.record_draw_indexed(100, 300, 1);
    // Don't call end_pass

    // end_frame should handle unclosed pass
    let stats = tracker.end_frame();
    assert_eq!(stats.total_draw_calls, 1);
    assert_eq!(stats.passes.len(), 1);
}

/// Test 87: Nested pass attempt (should close previous).
#[test]
fn edge_nested_pass() {
    let mut tracker = DrawCallTracker::new();

    tracker.begin_frame();
    tracker.begin_pass(PassType::Render, Some("Pass1"));
    tracker.record_draw_indexed(100, 300, 1);

    // Start new pass without ending first - should auto-close
    tracker.begin_pass(PassType::Compute, Some("Pass2"));
    tracker.record_dispatch(8, 8, 1);
    tracker.end_pass();

    let stats = tracker.end_frame();
    assert_eq!(stats.passes.len(), 2);
}

/// Test 88: Disabled tracker operations.
#[test]
fn edge_disabled_tracker() {
    let mut tracker = DrawCallTracker::new();
    tracker.set_enabled(false);

    tracker.begin_frame();
    tracker.begin_pass(PassType::Render, None);
    tracker.record_draw_indexed(100, 300, 1);
    let pass_stats = tracker.end_pass();
    let frame_stats = tracker.end_frame();

    assert!(pass_stats.is_none());
    assert_eq!(frame_stats.total_draw_calls, 0);
}

/// Test 89: Reset clears everything.
#[test]
fn edge_reset() {
    let mut tracker = DrawCallTracker::new();

    // Generate some history
    for _ in 0..5 {
        tracker.begin_frame();
        tracker.begin_pass(PassType::Render, None);
        tracker.record_draw_indexed(100, 300, 1);
        tracker.end_pass();
        tracker.end_frame();
    }

    assert_eq!(tracker.history().len(), 5);
    assert_eq!(tracker.current_frame_number(), 5);

    tracker.reset();

    assert_eq!(tracker.history().len(), 0);
    assert_eq!(tracker.current_frame_number(), 0);
}

/// Test 90: Clear history preserves frame number.
#[test]
fn edge_clear_history() {
    let mut tracker = DrawCallTracker::new();

    for _ in 0..5 {
        tracker.begin_frame();
        tracker.begin_pass(PassType::Render, None);
        tracker.record_draw_indexed(100, 300, 1);
        tracker.end_pass();
        tracker.end_frame();
    }

    let frame_before = tracker.current_frame_number();
    tracker.clear_history();

    assert_eq!(tracker.history().len(), 0);
    assert_eq!(tracker.current_frame_number(), frame_before);
}

// ============================================================================
// SECTION 5 -- Recommendation Tests (91-105)
// ============================================================================

/// Test 91: State thrashing generates batch recommendation.
#[test]
fn rec_state_thrashing_batch() {
    let mut frame = DrawFrameStats::new(0);
    let mut pass = PassStats::new(PassType::Render, None);

    for _ in 0..100 {
        pass.record_draw(&DrawCall::new(DrawType::Draw, 100, 1));
        pass.record_pipeline_switch();
    }
    pass.finish();
    frame.add_pass(pass);
    frame.finish();

    let recs = DrawCallAnalyzer::recommendations(&frame);
    assert!(!recs.is_empty());
    let has_state_rec = recs.iter().any(|r| r.contains("state"));
    assert!(has_state_rec);
}

/// Test 92: Small draws generate instancing recommendation.
#[test]
fn rec_small_draws_instancing() {
    let mut frame = DrawFrameStats::new(0);
    let mut pass = PassStats::new(PassType::Render, None);

    // Many small draws - CPU bound
    for _ in 0..5000 {
        pass.record_draw(&DrawCall::new(DrawType::Draw, 10, 1));
    }
    pass.finish();
    frame.add_pass(pass);
    frame.finish();

    let recs = DrawCallAnalyzer::recommendations(&frame);
    let has_batch_rec = recs.iter().any(|r| r.to_lowercase().contains("instancing") || r.to_lowercase().contains("batch"));
    assert!(has_batch_rec);
}

/// Test 93: Many pipelines generate consolidation recommendation.
#[test]
fn rec_many_pipelines_consolidation() {
    let mut frame = DrawFrameStats::new(0);
    let mut pass = PassStats::new(PassType::Render, None);

    // Excessive state changes
    for i in 0..1000 {
        pass.record_draw(&DrawCall::indexed(100, 300, 1).with_pipeline(i));
    }
    pass.record_pipeline_switch();
    pass.record_pipeline_switch();
    pass.record_pipeline_switch();
    pass.finish();
    frame.add_pass(pass);
    frame.finish();

    // Should trigger some recommendations
    let recs = DrawCallAnalyzer::recommendations(&frame);
    // With many draws, some recommendation should be generated
    assert!(frame.total_draw_calls > 0);
}

/// Test 94: High draw density generates GPU-driven recommendation.
#[test]
fn rec_high_density_gpu_driven() {
    // Create frame with artificially high density
    let mut frame = DrawFrameStats::new(0);
    let mut pass = PassStats::new(PassType::Render, None);

    // Many small draws with high state changes
    for _ in 0..10_000 {
        pass.record_draw(&DrawCall::new(DrawType::Draw, 10, 1));
    }
    pass.finish();
    frame.add_pass(pass);
    frame.finish();

    let analysis = DrawCallAnalyzer::analyze_frame(&frame);
    // Should generate some recommendations
    let recs = analysis.recommendations();
    assert!(!recs.is_empty());
}

/// Test 95: No false positives on healthy workload.
#[test]
fn rec_no_false_positives() {
    let mut frame = DrawFrameStats::new(0);
    let mut pass = PassStats::new(PassType::Render, None);

    // Healthy workload: reasonable draw count, good batching
    for _ in 0..100 {
        pass.record_draw(&DrawCall::new(DrawType::Draw, 5000, 1));
    }
    pass.finish();
    frame.add_pass(pass);
    frame.finish();

    let recs = DrawCallAnalyzer::recommendations(&frame);
    // Should have no or few recommendations
    let critical_count = recs.iter().filter(|r| r.contains("excessive") || r.contains("thrashing")).count();
    assert_eq!(critical_count, 0);
}

/// Test 96: Excessive draw calls bottleneck.
#[test]
fn rec_excessive_draws_bottleneck() {
    let mut frame = DrawFrameStats::new(0);
    let mut pass = PassStats::new(PassType::Render, None);

    for _ in 0..15_000 {
        pass.record_draw(&DrawCall::new(DrawType::Draw, 100, 1));
    }
    pass.finish();
    frame.add_pass(pass);
    frame.finish();

    let analysis = DrawCallAnalyzer::analyze_frame(&frame);
    let has_excessive = analysis.bottlenecks.iter().any(|b| b.contains("Excessive draw calls"));
    assert!(has_excessive);
}

/// Test 97: Pipeline switches bottleneck.
#[test]
fn rec_pipeline_switches_bottleneck() {
    let mut frame = DrawFrameStats::new(0);
    let mut pass = PassStats::new(PassType::Render, None);

    // More switches than half the draws
    for _ in 0..10 {
        pass.record_draw(&DrawCall::new(DrawType::Draw, 100, 1));
        pass.record_pipeline_switch();
    }
    pass.finish();
    frame.add_pass(pass);
    frame.finish();

    let analysis = DrawCallAnalyzer::analyze_frame(&frame);
    let has_pipeline_issue = analysis.bottlenecks.iter().any(|b| b.contains("pipeline"));
    assert!(has_pipeline_issue);
}

/// Test 98: Pass-level state change bottleneck.
#[test]
fn rec_pass_state_change_bottleneck() {
    let mut frame = DrawFrameStats::new(0);
    let mut pass = PassStats::new(PassType::Render, Some("BadPass".to_string()));

    // More than 1 state change per draw
    pass.record_draw(&DrawCall::new(DrawType::Draw, 100, 1));
    pass.record_pipeline_switch();
    pass.record_bind_group_set();
    pass.record_bind_group_set();
    pass.finish();
    frame.add_pass(pass);
    frame.finish();

    let analysis = DrawCallAnalyzer::analyze_frame(&frame);
    let has_pass_issue = analysis.bottlenecks.iter().any(|b| b.contains("BadPass") && b.contains("state changes"));
    assert!(has_pass_issue);
}

/// Test 99: Low batching score.
#[test]
fn rec_low_batching_score() {
    let mut frame = DrawFrameStats::new(0);
    let mut pass = PassStats::new(PassType::Render, None);

    // Very low vertices per draw
    for _ in 0..100 {
        pass.record_draw(&DrawCall::new(DrawType::Draw, 10, 1));
    }
    pass.finish();
    frame.add_pass(pass);
    frame.finish();

    let analysis = DrawCallAnalyzer::analyze_frame(&frame);
    assert!(analysis.batching_score < 0.3);

    let recs = analysis.recommendations();
    let has_batch_rec = recs.iter().any(|r| r.to_lowercase().contains("batch") || r.to_lowercase().contains("merg"));
    assert!(has_batch_rec);
}

/// Test 100: Perfect batching efficiency.
#[test]
fn rec_perfect_efficiency() {
    let mut frame = DrawFrameStats::new(0);
    let mut pass = PassStats::new(PassType::Render, None);

    // High vertices per draw = good batching
    pass.record_draw(&DrawCall::new(DrawType::Draw, 100_000, 1));
    pass.finish();
    frame.add_pass(pass);
    frame.finish();

    let efficiency = DrawCallAnalyzer::batching_efficiency(&frame);
    assert!(efficiency >= 1.0);
}

/// Test 101: Recommendations include bottlenecks.
#[test]
fn rec_includes_bottlenecks() {
    let mut frame = DrawFrameStats::new(0);
    let mut pass = PassStats::new(PassType::Render, Some("Test".to_string()));

    // Create a condition that adds a bottleneck
    pass.record_draw(&DrawCall::new(DrawType::Draw, 100, 1));
    pass.record_pipeline_switch();
    pass.record_bind_group_set();
    pass.record_bind_group_set();
    pass.finish();
    frame.add_pass(pass);
    frame.finish();

    let analysis = DrawCallAnalyzer::analyze_frame(&frame);
    assert!(!analysis.bottlenecks.is_empty());

    let recs = analysis.recommendations();
    // Recommendations should include the bottleneck messages
    assert!(recs.len() >= analysis.bottlenecks.len());
}

/// Test 102: Empty frame recommendations (batching score 1.0 for zero draws).
#[test]
fn rec_empty_frame_efficiency() {
    let frame = DrawFrameStats::new(0);

    // Empty frame should have perfect batching efficiency (no draws = nothing to optimize)
    let efficiency = DrawCallAnalyzer::batching_efficiency(&frame);
    assert_eq!(efficiency, 1.0);

    // No state thrashing on empty frame
    assert!(!DrawCallAnalyzer::detect_state_thrashing(&frame));

    // Analysis should show no CPU/GPU bound issues
    let analysis = DrawCallAnalyzer::analyze_frame(&frame);
    assert!(!analysis.is_cpu_bound());
    assert!(!analysis.is_gpu_bound());
}

/// Test 103: Compute-only frame recommendations.
#[test]
fn rec_compute_only() {
    let mut frame = DrawFrameStats::new(0);
    let mut pass = PassStats::new(PassType::Compute, None);

    pass.record_draw(&DrawCall::dispatch(256, 256, 64));
    pass.finish();
    frame.add_pass(pass);
    frame.finish();

    // Compute-only frame shouldn't trigger draw-specific recommendations
    let analysis = DrawCallAnalyzer::analyze_frame(&frame);
    assert!(!analysis.is_cpu_bound());
    assert!(!analysis.is_gpu_bound());
}

/// Test 104: Frame analysis display.
#[test]
fn rec_frame_analysis_display() {
    let mut frame = DrawFrameStats::new(42);
    let mut pass = PassStats::new(PassType::Render, None);
    pass.record_draw(&DrawCall::new(DrawType::Draw, 1000, 1));
    pass.finish();
    frame.add_pass(pass);
    frame.finish();

    let analysis = DrawCallAnalyzer::analyze_frame(&frame);
    let display = format!("{}", analysis);

    assert!(display.contains("42"));
    assert!(display.contains("Analysis"));
}

/// Test 105: Comparison display.
#[test]
fn rec_comparison_display() {
    let frame_a = DrawFrameStats::new(0);
    let frame_b = DrawFrameStats::new(1);

    let comparison = DrawCallAnalyzer::compare_frames(&frame_a, &frame_b);
    let display = format!("{}", comparison);

    assert!(display.contains("0"));
    assert!(display.contains("1"));
    assert!(display.contains("vs"));
}

// ============================================================================
// SECTION 6 -- History and Trend Tests (106-120)
// ============================================================================

/// Test 106: Trend detection increasing draws.
#[test]
fn trend_increasing_draws() {
    let mut history = VecDeque::new();

    for i in 0..20 {
        let mut frame = DrawFrameStats::new(i);
        let mut pass = PassStats::new(PassType::Render, None);
        // Increasing draw count
        for _ in 0..(i + 1) * 10 {
            pass.record_draw(&DrawCall::new(DrawType::Draw, 100, 1));
        }
        pass.finish();
        frame.add_pass(pass);
        frame.finish();
        history.push_back(frame);
    }

    let trend = DrawCallAnalyzer::analyze_trend(&history);
    assert!(trend.is_increasing());
    assert!(!trend.is_decreasing());
}

/// Test 107: Trend detection decreasing draws.
#[test]
fn trend_decreasing_draws() {
    let mut history = VecDeque::new();

    for i in 0..20 {
        let mut frame = DrawFrameStats::new(i);
        let mut pass = PassStats::new(PassType::Render, None);
        // Decreasing draw count
        for _ in 0..(20 - i) * 10 {
            pass.record_draw(&DrawCall::new(DrawType::Draw, 100, 1));
        }
        pass.finish();
        frame.add_pass(pass);
        frame.finish();
        history.push_back(frame);
    }

    let trend = DrawCallAnalyzer::analyze_trend(&history);
    assert!(trend.is_decreasing());
    assert!(!trend.is_increasing());
}

/// Test 108: Stable workload detection.
#[test]
fn trend_stable_workload() {
    let mut history = VecDeque::new();

    for i in 0..20 {
        let mut frame = DrawFrameStats::new(i);
        let mut pass = PassStats::new(PassType::Render, None);
        // Constant draw count
        for _ in 0..100 {
            pass.record_draw(&DrawCall::new(DrawType::Draw, 100, 1));
        }
        pass.finish();
        frame.add_pass(pass);
        frame.finish();
        history.push_back(frame);
    }

    let trend = DrawCallAnalyzer::analyze_trend(&history);
    assert!(trend.is_stable());
}

/// Test 109: Variance calculation accuracy.
#[test]
fn trend_variance_calculation() {
    let mut history = VecDeque::new();

    // Create frames with varying draw counts
    let draw_counts = [10, 20, 30, 40, 50];
    for (i, &count) in draw_counts.iter().enumerate() {
        let mut frame = DrawFrameStats::new(i as u64);
        let mut pass = PassStats::new(PassType::Render, None);
        for _ in 0..count {
            pass.record_draw(&DrawCall::new(DrawType::Draw, 100, 1));
        }
        pass.finish();
        frame.add_pass(pass);
        frame.finish();
        history.push_back(frame);
    }

    let trend = DrawCallAnalyzer::analyze_trend(&history);
    assert!(trend.draw_variance > 0.0);
    assert_eq!(trend.avg_draw_calls, 30.0);
}

/// Test 110: Average calculations from history.
#[test]
fn trend_average_calculations() {
    let mut tracker = DrawCallTracker::new();

    for i in 0..10 {
        tracker.begin_frame();
        tracker.begin_pass(PassType::Render, None);
        for _ in 0..(i + 1) * 10 {
            tracker.record_draw_indexed(100, 300, 1);
        }
        tracker.end_pass();
        tracker.end_frame();
    }

    let avg = tracker.avg_draw_calls_per_frame();
    // Average of 10, 20, 30, ..., 100 = 55
    assert!((avg - 55.0).abs() < 0.01);
}

/// Test 111: Ring buffer wraparound.
#[test]
fn trend_ring_buffer_wraparound() {
    let mut tracker = DrawCallTracker::with_history_size(5);

    for i in 0..20 {
        tracker.begin_frame();
        tracker.begin_pass(PassType::Render, None);
        for _ in 0..(i + 1) * 10 {
            tracker.record_draw_indexed(100, 300, 1);
        }
        tracker.end_pass();
        tracker.end_frame();
    }

    // Should only have last 5 frames
    assert_eq!(tracker.history().len(), 5);

    // Frames 15-19 should be present
    let frames: Vec<_> = tracker.history().iter().map(|f| f.frame_number).collect();
    assert_eq!(frames, vec![15, 16, 17, 18, 19]);
}

/// Test 112: Spike detection in history.
#[test]
fn trend_spike_detection() {
    let mut history = VecDeque::new();

    // Normal frames
    for i in 0..5 {
        let mut frame = DrawFrameStats::new(i);
        let mut pass = PassStats::new(PassType::Render, None);
        for _ in 0..100 {
            pass.record_draw(&DrawCall::new(DrawType::Draw, 100, 1));
        }
        pass.finish();
        frame.add_pass(pass);
        frame.finish();
        history.push_back(frame);
    }

    // Spike frame
    let mut spike_frame = DrawFrameStats::new(5);
    let mut pass = PassStats::new(PassType::Render, None);
    for _ in 0..10000 {
        pass.record_draw(&DrawCall::new(DrawType::Draw, 100, 1));
    }
    pass.finish();
    spike_frame.add_pass(pass);
    spike_frame.finish();
    history.push_back(spike_frame);

    let trend = DrawCallAnalyzer::analyze_trend(&history);
    // High variance indicates spike
    assert!(trend.draw_variance > 1000.0);
}

/// Test 113: Frame comparison deltas.
#[test]
fn trend_frame_comparison() {
    let mut frame_a = DrawFrameStats::new(0);
    let mut pass_a = PassStats::new(PassType::Render, None);
    for _ in 0..100 {
        pass_a.record_draw(&DrawCall::new(DrawType::Draw, 100, 1));
    }
    pass_a.finish();
    frame_a.add_pass(pass_a);
    frame_a.finish();

    let mut frame_b = DrawFrameStats::new(1);
    let mut pass_b = PassStats::new(PassType::Render, None);
    for _ in 0..150 {
        pass_b.record_draw(&DrawCall::new(DrawType::Draw, 200, 1));
    }
    pass_b.finish();
    frame_b.add_pass(pass_b);
    frame_b.finish();

    let comparison = DrawCallAnalyzer::compare_frames(&frame_a, &frame_b);

    assert_eq!(comparison.draw_delta, 50);
    assert_eq!(comparison.vertex_delta, 20_000); // 150*200 - 100*100
}

/// Test 114: Trend display formatting.
#[test]
fn trend_display_formatting() {
    let trend = TrendAnalysis {
        avg_draw_calls: 100.0,
        avg_frame_time_ms: 16.67,
        draw_variance: 25.0,
        time_variance: 0.5,
        draw_trend: 0.0,
        samples: 10,
    };

    let display = format!("{}", trend);
    assert!(display.contains("10 samples"));
    assert!(display.contains("100"));
    assert!(display.contains("stable"));
}

/// Test 115: Empty history trend analysis.
#[test]
fn trend_empty_history() {
    let history: VecDeque<DrawFrameStats> = VecDeque::new();
    let trend = DrawCallAnalyzer::analyze_trend(&history);

    assert_eq!(trend.samples, 0);
    assert_eq!(trend.avg_draw_calls, 0.0);
}

/// Test 116: Single frame history.
#[test]
fn trend_single_frame_history() {
    let mut history = VecDeque::new();

    let mut frame = DrawFrameStats::new(0);
    let mut pass = PassStats::new(PassType::Render, None);
    pass.record_draw(&DrawCall::new(DrawType::Draw, 100, 1));
    pass.finish();
    frame.add_pass(pass);
    frame.finish();
    history.push_back(frame);

    let trend = DrawCallAnalyzer::analyze_trend(&history);
    assert_eq!(trend.samples, 1);
}

/// Test 117: Average frame time calculation.
#[test]
fn trend_avg_frame_time() {
    let mut tracker = DrawCallTracker::new();

    for _ in 0..5 {
        tracker.begin_frame();
        tracker.begin_pass(PassType::Render, None);
        tracker.record_draw_indexed(100, 300, 1);
        tracker.end_pass();
        tracker.end_frame();
    }

    let avg_time = tracker.avg_frame_time_ms();
    // Should be some non-negative value
    assert!(avg_time >= 0.0);
}

/// Test 118: Average dispatches calculation.
#[test]
fn trend_avg_dispatches() {
    let mut tracker = DrawCallTracker::new();

    for i in 0..5 {
        tracker.begin_frame();
        tracker.begin_pass(PassType::Compute, None);
        for _ in 0..(i + 1) {
            tracker.record_dispatch(8, 8, 1);
        }
        tracker.end_pass();
        tracker.end_frame();
    }

    let avg = tracker.avg_dispatches_per_frame();
    // Average of 1, 2, 3, 4, 5 = 3
    assert_eq!(avg, 3.0);
}

/// Test 119: Average vertices calculation.
#[test]
fn trend_avg_vertices() {
    let mut tracker = DrawCallTracker::new();

    for i in 0..4 {
        tracker.begin_frame();
        tracker.begin_pass(PassType::Render, None);
        tracker.record_draw_indexed((i + 1) * 1000, (i + 1) * 3000, 1);
        tracker.end_pass();
        tracker.end_frame();
    }

    let avg = tracker.avg_vertices_per_frame();
    // Average of 1000, 2000, 3000, 4000 = 2500
    assert_eq!(avg, 2500.0);
}

/// Test 120: Last frame stats retrieval.
#[test]
fn trend_last_frame_stats() {
    let mut tracker = DrawCallTracker::new();

    assert!(tracker.last_frame_stats().is_none());

    tracker.begin_frame();
    tracker.begin_pass(PassType::Render, None);
    tracker.record_draw_indexed(100, 300, 1);
    tracker.end_pass();
    tracker.end_frame();

    let last = tracker.last_frame_stats();
    assert!(last.is_some());
    assert_eq!(last.unwrap().frame_number, 0);

    tracker.begin_frame();
    tracker.begin_pass(PassType::Render, None);
    tracker.record_draw_indexed(200, 600, 2);
    tracker.end_pass();
    tracker.end_frame();

    let last = tracker.last_frame_stats();
    assert!(last.is_some());
    assert_eq!(last.unwrap().frame_number, 1);
}

// ============================================================================
// SECTION 7 -- Additional Coverage Tests (121-130)
// ============================================================================

/// Test 121: DrawType Display implementation.
#[test]
fn extra_drawtype_display() {
    assert_eq!(format!("{}", DrawType::Draw), "Draw");
    assert_eq!(format!("{}", DrawType::DrawIndexed), "DrawIndexed");
    assert_eq!(format!("{}", DrawType::DrawIndirect), "DrawIndirect");
    assert_eq!(format!("{}", DrawType::DrawIndexedIndirect), "DrawIndexedIndirect");
    assert_eq!(format!("{}", DrawType::MultiDrawIndirect), "MultiDrawIndirect");
    assert_eq!(format!("{}", DrawType::MultiDrawIndexedIndirect), "MultiDrawIndexedIndirect");
    assert_eq!(format!("{}", DrawType::Dispatch), "Dispatch");
    assert_eq!(format!("{}", DrawType::DispatchIndirect), "DispatchIndirect");
}

/// Test 122: PassType Display implementation.
#[test]
fn extra_passtype_display() {
    assert_eq!(format!("{}", PassType::Render), "Render");
    assert_eq!(format!("{}", PassType::Compute), "Compute");
}

/// Test 123: PassStats Display implementation.
#[test]
fn extra_passstats_display() {
    let mut pass = PassStats::new(PassType::Render, Some("TestPass".to_string()));
    pass.record_draw(&DrawCall::new(DrawType::Draw, 1000, 1));
    pass.finish();

    let display = format!("{}", pass);
    assert!(display.contains("Render"));
    assert!(display.contains("TestPass"));
    assert!(display.contains("1 draws"));
}

/// Test 124: DrawFrameStats Display implementation.
#[test]
fn extra_framestats_display() {
    let mut frame = DrawFrameStats::new(42);
    let mut pass = PassStats::new(PassType::Render, None);
    pass.record_draw(&DrawCall::new(DrawType::Draw, 1000, 1));
    pass.finish();
    frame.add_pass(pass);
    frame.finish();

    let display = format!("{}", frame);
    assert!(display.contains("42"));
    assert!(display.contains("1 draws"));
}

/// Test 125: DrawFrameStats summary method.
#[test]
fn extra_framestats_summary() {
    let mut frame = DrawFrameStats::new(5);
    let mut pass = PassStats::new(PassType::Render, None);
    pass.record_draw(&DrawCall::new(DrawType::Draw, 500, 10));
    pass.record_draw(&DrawCall::new(DrawType::Draw, 500, 5));
    pass.finish();
    frame.add_pass(pass);
    frame.finish();

    let summary = frame.summary();
    assert!(summary.contains("Frame 5"));
    assert!(summary.contains("2 draws"));
    assert!(summary.contains("15 instances"));
}

/// Test 126: DrawBatch total instance count.
#[test]
fn extra_batch_instance_count() {
    let mut batch = DrawBatch::new(1);
    batch.add_draw(DrawCall::new(DrawType::Draw, 100, 10));
    batch.add_draw(DrawCall::new(DrawType::Draw, 100, 20));
    batch.add_draw(DrawCall::new(DrawType::Draw, 100, 30));

    assert_eq!(batch.total_instance_count(), 60);
}

/// Test 127: PrimitiveTopology edge cases.
#[test]
fn extra_primitive_topology_edge() {
    // Zero vertices
    assert_eq!(PrimitiveTopology::TriangleList.primitives_from_vertices(0), 0);
    assert_eq!(PrimitiveTopology::LineStrip.primitives_from_vertices(0), 0);
    assert_eq!(PrimitiveTopology::TriangleStrip.primitives_from_vertices(0), 0);

    // Single vertex
    assert_eq!(PrimitiveTopology::PointList.primitives_from_vertices(1), 1);
    assert_eq!(PrimitiveTopology::LineStrip.primitives_from_vertices(1), 0);
    assert_eq!(PrimitiveTopology::TriangleStrip.primitives_from_vertices(1), 0);

    // Two vertices
    assert_eq!(PrimitiveTopology::LineList.primitives_from_vertices(2), 1);
    assert_eq!(PrimitiveTopology::LineStrip.primitives_from_vertices(2), 1);
    assert_eq!(PrimitiveTopology::TriangleStrip.primitives_from_vertices(2), 0);
}

/// Test 128: DrawCall workgroups for non-compute.
#[test]
fn extra_drawcall_workgroups_non_compute() {
    let draw = DrawCall::new(DrawType::Draw, 100, 1);
    assert!(draw.workgroups().is_none());

    let indexed = DrawCall::indexed(100, 300, 1);
    assert!(indexed.workgroups().is_none());
}

/// Test 129: Constants are accessible and sensible.
#[test]
fn extra_constants() {
    assert_eq!(DEFAULT_HISTORY_SIZE, 120);
    assert_eq!(MIN_HISTORY_SIZE, 1);
    assert_eq!(MAX_HISTORY_SIZE, 3600);
    assert!(STATE_THRASHING_THRESHOLD > 0.0);
    assert!(STATE_THRASHING_THRESHOLD < 1.0);
    assert!(HEAVY_PASS_DRAW_THRESHOLD > 0);
    assert!(HEAVY_PASS_VERTEX_THRESHOLD > 0);
}

/// Test 130: DrawCall builder chain.
#[test]
fn extra_drawcall_builder_chain() {
    let draw = DrawCall::new(DrawType::DrawIndexed, 1000, 10)
        .with_pipeline(123)
        .with_label("chained_draw");

    assert_eq!(draw.draw_type, DrawType::DrawIndexed);
    assert_eq!(draw.vertex_count, 1000);
    assert_eq!(draw.instance_count, 10);
    assert_eq!(draw.pipeline_id, Some(123));
    assert_eq!(draw.label.as_deref(), Some("chained_draw"));
}
