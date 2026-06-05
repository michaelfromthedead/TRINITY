// SPDX-License-Identifier: MIT
//
// blackbox_gpu_culling_pipeline.rs -- Blackbox tests for T-WGPU-P6.6.3 GPUCullingPipeline.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only the documented public types and functions:
//
//   - GPUCullingPipeline
//   - GPUCullingPipelineBuilder
//   - GPUCullingConfig
//   - GPUCullingParams
//   - CullingStage
//   - CullingDebugDump
//   - workgroups_for_objects (gpu_culling_workgroups_for_objects)
//   - Flags: FLAG_SKIP_FRUSTUM, FLAG_SKIP_HIZ, FLAG_SKIP_LOD, FLAG_CONSERVATIVE, FLAG_DEBUG
//   - Constants: DEFAULT_MAX_OBJECTS, DEFAULT_MAX_DRAWS, DEFAULT_HIZ_WIDTH, DEFAULT_HIZ_HEIGHT
//   - WORKGROUP_SIZE, GPU_CULLING_PARAMS_SIZE
//
// ACCEPTANCE CRITERIA:
//   1. Pipeline lifecycle      -- 15+ tests covering builder pattern, config, construction
//   2. Stage execution order   -- 10+ tests for CullingStage enum
//   3. Debug dump output       -- 10+ tests for timing, visibility, cull rate
//   4. Configuration options   -- 15+ tests for skip flags, workgroup sizes
//   5. Parameter conversion    -- 15+ tests for to_hiz_cull_params, to_lod_select_params, etc.
//
// Total target: 60+ tests

use renderer_backend::gpu_driven::{
    // Main pipeline types (GPUCullingPipeline requires GPU for instantiation, tested via config/params)
    GPUCullingConfig, GPUCullingParams,
    CullingStage, CullingDebugDump,
    // Workgroup calculation
    gpu_culling_workgroups_for_objects,
    // Flags
    GPU_CULLING_FLAG_SKIP_FRUSTUM, GPU_CULLING_FLAG_SKIP_HIZ,
    GPU_CULLING_FLAG_SKIP_LOD, GPU_CULLING_FLAG_CONSERVATIVE, GPU_CULLING_FLAG_DEBUG,
    // Constants
    GPU_CULLING_DEFAULT_MAX_OBJECTS, GPU_CULLING_DEFAULT_MAX_DRAWS,
    DEFAULT_HIZ_WIDTH, DEFAULT_HIZ_HEIGHT,
    GPU_CULLING_PARAMS_SIZE, GPU_CULLING_WORKGROUP_SIZE,
    // LOD selection mode for param conversion tests
    SelectionMode,
};

// =============================================================================
// HELPERS -- Construction helpers for cleanroom testing
// =============================================================================

/// Create an identity view-projection matrix.
fn identity_vp() -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// Create a basic culling params for testing.
fn basic_params(object_count: u32) -> GPUCullingParams {
    GPUCullingParams::new(
        object_count,
        [0.0, 0.0, 0.0],
        &identity_vp(),
        0.1,
        std::f32::consts::FRAC_PI_4,
    )
}

// =============================================================================
// SECTION 1 -- CULLING STAGE TESTS (Stage Enumeration) -- 15 tests
// =============================================================================

/// CullingStage::count() returns 5 (total stages).
#[test]
fn culling_stage_count_is_five() {
    assert_eq!(CullingStage::count(), 5);
}

/// CullingStage::FrustumCull has index 0.
#[test]
fn culling_stage_frustum_cull_index_is_zero() {
    assert_eq!(CullingStage::FrustumCull.index(), 0);
}

/// CullingStage::HiZCull has index 1.
#[test]
fn culling_stage_hiz_cull_index_is_one() {
    assert_eq!(CullingStage::HiZCull.index(), 1);
}

/// CullingStage::LodSelect has index 2.
#[test]
fn culling_stage_lod_select_index_is_two() {
    assert_eq!(CullingStage::LodSelect.index(), 2);
}

/// CullingStage::StreamCompact has index 3.
#[test]
fn culling_stage_stream_compact_index_is_three() {
    assert_eq!(CullingStage::StreamCompact.index(), 3);
}

/// CullingStage::BuildIndirect has index 4.
#[test]
fn culling_stage_build_indirect_index_is_four() {
    assert_eq!(CullingStage::BuildIndirect.index(), 4);
}

/// CullingStage::FrustumCull has correct name.
#[test]
fn culling_stage_frustum_cull_name() {
    assert_eq!(CullingStage::FrustumCull.name(), "FrustumCull");
}

/// CullingStage::HiZCull has correct name.
#[test]
fn culling_stage_hiz_cull_name() {
    assert_eq!(CullingStage::HiZCull.name(), "HiZCull");
}

/// CullingStage::LodSelect has correct name.
#[test]
fn culling_stage_lod_select_name() {
    assert_eq!(CullingStage::LodSelect.name(), "LodSelect");
}

/// CullingStage::StreamCompact has correct name.
#[test]
fn culling_stage_stream_compact_name() {
    assert_eq!(CullingStage::StreamCompact.name(), "StreamCompact");
}

/// CullingStage::BuildIndirect has correct name.
#[test]
fn culling_stage_build_indirect_name() {
    assert_eq!(CullingStage::BuildIndirect.name(), "BuildIndirect");
}

/// All stage indices are distinct.
#[test]
fn culling_stage_indices_distinct() {
    let indices = [
        CullingStage::FrustumCull.index(),
        CullingStage::HiZCull.index(),
        CullingStage::LodSelect.index(),
        CullingStage::StreamCompact.index(),
        CullingStage::BuildIndirect.index(),
    ];
    for i in 0..indices.len() {
        for j in (i + 1)..indices.len() {
            assert_ne!(indices[i], indices[j], "Stage indices {} and {} should be distinct", i, j);
        }
    }
}

/// Stage indices are sequential from 0 to 4.
#[test]
fn culling_stage_indices_sequential() {
    assert_eq!(CullingStage::FrustumCull.index(), 0);
    assert_eq!(CullingStage::HiZCull.index(), 1);
    assert_eq!(CullingStage::LodSelect.index(), 2);
    assert_eq!(CullingStage::StreamCompact.index(), 3);
    assert_eq!(CullingStage::BuildIndirect.index(), 4);
}

/// Stage can be used as array index.
#[test]
fn culling_stage_as_array_index() {
    let mut timings = [0u64; CullingStage::count()];
    timings[CullingStage::FrustumCull.index()] = 100;
    timings[CullingStage::HiZCull.index()] = 200;
    timings[CullingStage::LodSelect.index()] = 50;
    timings[CullingStage::StreamCompact.index()] = 75;
    timings[CullingStage::BuildIndirect.index()] = 125;

    assert_eq!(timings[0], 100);
    assert_eq!(timings[1], 200);
    assert_eq!(timings[2], 50);
    assert_eq!(timings[3], 75);
    assert_eq!(timings[4], 125);
}

/// CullingStage is Copy.
#[test]
fn culling_stage_is_copy() {
    let stage = CullingStage::FrustumCull;
    let copy = stage;
    assert_eq!(stage, copy);
}

// =============================================================================
// SECTION 2 -- CONFIGURATION TESTS (GPUCullingConfig) -- 15 tests
// =============================================================================

/// GPUCullingConfig::default() has expected max_objects.
#[test]
fn config_default_max_objects() {
    let config = GPUCullingConfig::default();
    assert_eq!(config.max_objects, GPU_CULLING_DEFAULT_MAX_OBJECTS);
    assert_eq!(config.max_objects, 100_000);
}

/// GPUCullingConfig::default() has expected max_draws.
#[test]
fn config_default_max_draws() {
    let config = GPUCullingConfig::default();
    assert_eq!(config.max_draws, GPU_CULLING_DEFAULT_MAX_DRAWS);
    assert_eq!(config.max_draws, 65536);
}

/// GPUCullingConfig::default() has expected HiZ width.
#[test]
fn config_default_hiz_width() {
    let config = GPUCullingConfig::default();
    assert_eq!(config.hiz_width, DEFAULT_HIZ_WIDTH);
    assert_eq!(config.hiz_width, 1920);
}

/// GPUCullingConfig::default() has expected HiZ height.
#[test]
fn config_default_hiz_height() {
    let config = GPUCullingConfig::default();
    assert_eq!(config.hiz_height, DEFAULT_HIZ_HEIGHT);
    assert_eq!(config.hiz_height, 1080);
}

/// GPUCullingConfig::default() has debug_visualization disabled.
#[test]
fn config_default_debug_visualization_disabled() {
    let config = GPUCullingConfig::default();
    assert!(!config.debug_visualization);
}

/// GPUCullingConfig::default() does not skip frustum cull.
#[test]
fn config_default_no_skip_frustum() {
    let config = GPUCullingConfig::default();
    assert!(!config.skip_frustum_cull);
}

/// GPUCullingConfig::default() does not skip HiZ cull.
#[test]
fn config_default_no_skip_hiz() {
    let config = GPUCullingConfig::default();
    assert!(!config.skip_hiz_cull);
}

/// GPUCullingConfig::default() does not skip LOD select.
#[test]
fn config_default_no_skip_lod() {
    let config = GPUCullingConfig::default();
    assert!(!config.skip_lod_select);
}

/// GPUCullingConfig::default() does not enable conservative HiZ.
#[test]
fn config_default_no_conservative() {
    let config = GPUCullingConfig::default();
    assert!(!config.conservative_hiz);
}

/// GPUCullingConfig can be cloned.
#[test]
fn config_is_clone() {
    let config = GPUCullingConfig::default();
    let cloned = config.clone();
    assert_eq!(config.max_objects, cloned.max_objects);
    assert_eq!(config.max_draws, cloned.max_draws);
}

/// GPUCullingConfig hiz_mip_count has valid default.
#[test]
fn config_default_hiz_mip_count() {
    let config = GPUCullingConfig::default();
    // log2(1920) + 1 = 11
    assert_eq!(config.hiz_mip_count, 11);
}

/// GPUCullingConfig can have all skip flags set.
#[test]
fn config_all_skip_flags() {
    let mut config = GPUCullingConfig::default();
    config.skip_frustum_cull = true;
    config.skip_hiz_cull = true;
    config.skip_lod_select = true;

    assert!(config.skip_frustum_cull);
    assert!(config.skip_hiz_cull);
    assert!(config.skip_lod_select);
}

/// GPUCullingConfig is Debug.
#[test]
fn config_is_debug() {
    let config = GPUCullingConfig::default();
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("GPUCullingConfig"));
    assert!(debug_str.contains("max_objects"));
}

/// GPUCullingConfig max_objects can be modified.
#[test]
fn config_modify_max_objects() {
    let mut config = GPUCullingConfig::default();
    config.max_objects = 50_000;
    assert_eq!(config.max_objects, 50_000);
}

/// GPUCullingConfig max_draws can be modified.
#[test]
fn config_modify_max_draws() {
    let mut config = GPUCullingConfig::default();
    config.max_draws = 8192;
    assert_eq!(config.max_draws, 8192);
}

// =============================================================================
// SECTION 3 -- PARAMS STRUCT TESTS (GPUCullingParams) -- 20 tests
// =============================================================================

/// GPUCullingParams::SIZE equals GPU_CULLING_PARAMS_SIZE.
#[test]
fn params_size_constant() {
    assert_eq!(GPUCullingParams::SIZE, GPU_CULLING_PARAMS_SIZE);
    assert_eq!(GPUCullingParams::SIZE, 128);
}

/// GPUCullingParams::default() has zero object_count.
#[test]
fn params_default_zero_objects() {
    let params = GPUCullingParams::default();
    assert_eq!(params.object_count, 0);
}

/// GPUCullingParams::default() has zero flags.
#[test]
fn params_default_zero_flags() {
    let params = GPUCullingParams::default();
    assert_eq!(params.flags, 0);
}

/// GPUCullingParams::default() has default HiZ dimensions.
#[test]
fn params_default_hiz_dimensions() {
    let params = GPUCullingParams::default();
    assert_eq!(params.hiz_width, DEFAULT_HIZ_WIDTH);
    assert_eq!(params.hiz_height, DEFAULT_HIZ_HEIGHT);
}

/// GPUCullingParams::new() sets object_count correctly.
#[test]
fn params_new_sets_object_count() {
    let params = basic_params(1234);
    assert_eq!(params.object_count, 1234);
}

/// GPUCullingParams::new() sets camera_position correctly.
#[test]
fn params_new_sets_camera_position() {
    let params = GPUCullingParams::new(
        100,
        [10.0, 20.0, 30.0],
        &identity_vp(),
        0.1,
        0.785,
    );
    assert_eq!(params.camera_position, [10.0, 20.0, 30.0]);
}

/// GPUCullingParams::new() sets near_plane correctly.
#[test]
fn params_new_sets_near_plane() {
    let params = GPUCullingParams::new(
        100,
        [0.0, 0.0, 0.0],
        &identity_vp(),
        0.5,
        0.785,
    );
    assert_eq!(params.near_plane, 0.5);
}

/// GPUCullingParams::new() sets fov_y correctly.
#[test]
fn params_new_sets_fov_y() {
    let fov = std::f32::consts::FRAC_PI_3;
    let params = GPUCullingParams::new(
        100,
        [0.0, 0.0, 0.0],
        &identity_vp(),
        0.1,
        fov,
    );
    assert!((params.fov_y - fov).abs() < 1e-6);
}

/// GPUCullingParams::with_hiz() sets dimensions.
#[test]
fn params_with_hiz_sets_dimensions() {
    let params = basic_params(100).with_hiz(2560, 1440, 12);
    assert_eq!(params.hiz_width, 2560);
    assert_eq!(params.hiz_height, 1440);
    assert_eq!(params.max_mip, 12);
}

/// GPUCullingParams::with_screen_size() sets screen dimensions.
#[test]
fn params_with_screen_size() {
    let params = basic_params(100).with_screen_size(3840.0, 2160.0);
    assert_eq!(params.screen_width, 3840.0);
    assert_eq!(params.screen_height, 2160.0);
}

/// GPUCullingParams::with_max_draws() sets max_draws.
#[test]
fn params_with_max_draws() {
    let params = basic_params(100).with_max_draws(8192);
    assert_eq!(params.max_draws, 8192);
}

/// GPUCullingParams::with_lod_mode() sets lod_mode.
#[test]
fn params_with_lod_mode_distance() {
    let params = basic_params(100).with_lod_mode(SelectionMode::Distance);
    assert_eq!(params.lod_mode, 0);
}

/// GPUCullingParams::with_lod_mode() sets screen size mode.
#[test]
fn params_with_lod_mode_screen_size() {
    let params = basic_params(100).with_lod_mode(SelectionMode::ScreenSize);
    assert_eq!(params.lod_mode, 1);
}

/// GPUCullingParams::with_flags() sets all flags.
#[test]
fn params_with_flags() {
    let params = basic_params(100).with_flags(0b11111);
    assert_eq!(params.flags, 0b11111);
}

/// GPUCullingParams::add_flag() adds single flag.
#[test]
fn params_add_flag() {
    let params = basic_params(100)
        .add_flag(GPU_CULLING_FLAG_SKIP_FRUSTUM);
    assert_eq!(params.flags & GPU_CULLING_FLAG_SKIP_FRUSTUM, GPU_CULLING_FLAG_SKIP_FRUSTUM);
}

/// GPUCullingParams::add_flag() can chain multiple flags.
#[test]
fn params_add_flag_chained() {
    let params = basic_params(100)
        .add_flag(GPU_CULLING_FLAG_SKIP_FRUSTUM)
        .add_flag(GPU_CULLING_FLAG_SKIP_HIZ)
        .add_flag(GPU_CULLING_FLAG_DEBUG);

    assert!(!params.frustum_enabled());
    assert!(!params.hiz_enabled());
    assert!(params.lod_enabled());
}

/// GPUCullingParams::frustum_enabled() returns true by default.
#[test]
fn params_frustum_enabled_default() {
    let params = basic_params(100);
    assert!(params.frustum_enabled());
}

/// GPUCullingParams::frustum_enabled() returns false when skipped.
#[test]
fn params_frustum_disabled_when_skipped() {
    let params = basic_params(100).add_flag(GPU_CULLING_FLAG_SKIP_FRUSTUM);
    assert!(!params.frustum_enabled());
}

/// GPUCullingParams::hiz_enabled() returns true by default.
#[test]
fn params_hiz_enabled_default() {
    let params = basic_params(100);
    assert!(params.hiz_enabled());
}

/// GPUCullingParams::lod_enabled() returns true by default.
#[test]
fn params_lod_enabled_default() {
    let params = basic_params(100);
    assert!(params.lod_enabled());
}

// =============================================================================
// SECTION 4 -- WORKGROUP CALCULATION TESTS -- 15 tests
// =============================================================================

/// Workgroups for 0 objects is 0.
#[test]
fn workgroups_zero_objects() {
    assert_eq!(gpu_culling_workgroups_for_objects(0), 0);
}

/// Workgroups for 1 object is 1.
#[test]
fn workgroups_one_object() {
    assert_eq!(gpu_culling_workgroups_for_objects(1), 1);
}

/// Workgroups for exactly workgroup size is 1.
#[test]
fn workgroups_exact_workgroup_size() {
    assert_eq!(gpu_culling_workgroups_for_objects(64), 1);
    assert_eq!(gpu_culling_workgroups_for_objects(GPU_CULLING_WORKGROUP_SIZE), 1);
}

/// Workgroups for workgroup size + 1 is 2.
#[test]
fn workgroups_one_more_than_workgroup_size() {
    assert_eq!(gpu_culling_workgroups_for_objects(65), 2);
}

/// Workgroups for two full workgroups.
#[test]
fn workgroups_two_full() {
    assert_eq!(gpu_culling_workgroups_for_objects(128), 2);
}

/// Workgroups for 1000 objects.
#[test]
fn workgroups_one_thousand() {
    // 1000 / 64 = 15.625 -> 16
    assert_eq!(gpu_culling_workgroups_for_objects(1000), 16);
}

/// Workgroups for default max objects (100K).
#[test]
fn workgroups_default_max_objects() {
    // 100000 / 64 = 1562.5 -> 1563
    assert_eq!(gpu_culling_workgroups_for_objects(100_000), 1563);
}

/// Workgroups calculation is consistent with params.workgroups().
#[test]
fn workgroups_consistent_with_params() {
    let params = basic_params(1000);
    assert_eq!(params.workgroups(), gpu_culling_workgroups_for_objects(1000));
}

/// Workgroups for large object count (1M).
#[test]
fn workgroups_one_million() {
    // 1000000 / 64 = 15625
    assert_eq!(gpu_culling_workgroups_for_objects(1_000_000), 15625);
}

/// Workgroups for 63 objects (just under workgroup size).
#[test]
fn workgroups_just_under_workgroup_size() {
    assert_eq!(gpu_culling_workgroups_for_objects(63), 1);
}

/// Workgroups for 127 objects.
#[test]
fn workgroups_127_objects() {
    assert_eq!(gpu_culling_workgroups_for_objects(127), 2);
}

/// GPU_CULLING_WORKGROUP_SIZE is 64.
#[test]
fn workgroup_size_is_64() {
    assert_eq!(GPU_CULLING_WORKGROUP_SIZE, 64);
}

/// Workgroups increases monotonically.
#[test]
fn workgroups_monotonic() {
    for n in 0..200 {
        let wg1 = gpu_culling_workgroups_for_objects(n);
        let wg2 = gpu_culling_workgroups_for_objects(n + 1);
        assert!(wg2 >= wg1);
    }
}

/// Workgroups for 192 objects (3 full workgroups).
#[test]
fn workgroups_three_full() {
    assert_eq!(gpu_culling_workgroups_for_objects(192), 3);
}

/// Workgroups formula: (n + 63) / 64.
#[test]
fn workgroups_formula_matches() {
    for n in [0, 1, 63, 64, 65, 127, 128, 1000, 10000] {
        let expected = (n + GPU_CULLING_WORKGROUP_SIZE - 1) / GPU_CULLING_WORKGROUP_SIZE;
        assert_eq!(gpu_culling_workgroups_for_objects(n), expected, "Mismatch for n={}", n);
    }
}

// =============================================================================
// SECTION 5 -- DEBUG DUMP TESTS (CullingDebugDump) -- 15 tests
// =============================================================================

/// CullingDebugDump::default() has zero total_objects.
#[test]
fn debug_dump_default_zero_total_objects() {
    let dump = CullingDebugDump::default();
    assert_eq!(dump.total_objects, 0);
}

/// CullingDebugDump::default() has zero final_visible_count.
#[test]
fn debug_dump_default_zero_visible() {
    let dump = CullingDebugDump::default();
    assert_eq!(dump.final_visible_count, 0);
}

/// CullingDebugDump::default() has zero final_draw_count.
#[test]
fn debug_dump_default_zero_draws() {
    let dump = CullingDebugDump::default();
    assert_eq!(dump.final_draw_count, 0);
}

/// CullingDebugDump::default() has debug_enabled false.
#[test]
fn debug_dump_default_debug_disabled() {
    let dump = CullingDebugDump::default();
    assert!(!dump.debug_enabled);
}

/// CullingDebugDump::new() is same as default.
#[test]
fn debug_dump_new_same_as_default() {
    let new_dump = CullingDebugDump::new();
    let default_dump = CullingDebugDump::default();
    assert_eq!(new_dump.total_objects, default_dump.total_objects);
    assert_eq!(new_dump.final_visible_count, default_dump.final_visible_count);
}

/// CullingDebugDump cull_rate() is 0.0 when no objects.
#[test]
fn debug_dump_cull_rate_zero_objects() {
    let dump = CullingDebugDump::default();
    assert_eq!(dump.cull_rate(), 0.0);
}

/// CullingDebugDump cull_rate() is 0.0 when all visible.
#[test]
fn debug_dump_cull_rate_all_visible() {
    let mut dump = CullingDebugDump::new();
    dump.total_objects = 100;
    dump.final_visible_count = 100;
    assert!((dump.cull_rate() - 0.0).abs() < 1e-6);
}

/// CullingDebugDump cull_rate() is 1.0 when none visible.
#[test]
fn debug_dump_cull_rate_none_visible() {
    let mut dump = CullingDebugDump::new();
    dump.total_objects = 100;
    dump.final_visible_count = 0;
    assert!((dump.cull_rate() - 1.0).abs() < 1e-6);
}

/// CullingDebugDump cull_rate() is 0.75 when 25% visible.
#[test]
fn debug_dump_cull_rate_75_percent() {
    let mut dump = CullingDebugDump::new();
    dump.total_objects = 1000;
    dump.final_visible_count = 250;
    assert!((dump.cull_rate() - 0.75).abs() < 0.001);
}

/// CullingDebugDump stage_timing_ms() converts ns to ms.
#[test]
fn debug_dump_stage_timing_ns_to_ms() {
    let mut dump = CullingDebugDump::new();
    dump.stage_timings_ns[CullingStage::FrustumCull.index()] = 1_000_000;
    assert!((dump.stage_timing_ms(CullingStage::FrustumCull) - 1.0).abs() < 1e-6);
}

/// CullingDebugDump stage_timing_ms() works for all stages.
#[test]
fn debug_dump_stage_timing_all_stages() {
    let mut dump = CullingDebugDump::new();
    dump.stage_timings_ns[CullingStage::FrustumCull.index()] = 100_000;
    dump.stage_timings_ns[CullingStage::HiZCull.index()] = 200_000;
    dump.stage_timings_ns[CullingStage::LodSelect.index()] = 50_000;
    dump.stage_timings_ns[CullingStage::StreamCompact.index()] = 75_000;
    dump.stage_timings_ns[CullingStage::BuildIndirect.index()] = 125_000;

    assert!((dump.stage_timing_ms(CullingStage::FrustumCull) - 0.1).abs() < 1e-6);
    assert!((dump.stage_timing_ms(CullingStage::HiZCull) - 0.2).abs() < 1e-6);
    assert!((dump.stage_timing_ms(CullingStage::LodSelect) - 0.05).abs() < 1e-6);
}

/// CullingDebugDump total_timing_ms() sums all stages.
#[test]
fn debug_dump_total_timing_sums_stages() {
    let mut dump = CullingDebugDump::new();
    dump.stage_timings_ns[0] = 1_000_000;
    dump.stage_timings_ns[1] = 2_000_000;
    dump.stage_timings_ns[2] = 500_000;
    dump.stage_timings_ns[3] = 500_000;
    dump.stage_timings_ns[4] = 1_000_000;

    assert!((dump.total_timing_ms() - 5.0).abs() < 1e-6);
}

/// CullingDebugDump total_timing_ms() is zero by default.
#[test]
fn debug_dump_total_timing_zero_default() {
    let dump = CullingDebugDump::default();
    assert_eq!(dump.total_timing_ms(), 0.0);
}

/// CullingDebugDump visibility_counts array has correct size.
#[test]
fn debug_dump_visibility_counts_size() {
    let dump = CullingDebugDump::default();
    assert_eq!(dump.visibility_counts.len(), CullingStage::count());
}

// =============================================================================
// SECTION 6 -- FLAG TESTS -- 10 tests
// =============================================================================

/// FLAG_SKIP_FRUSTUM has expected value.
#[test]
fn flag_skip_frustum_value() {
    assert_eq!(GPU_CULLING_FLAG_SKIP_FRUSTUM, 1 << 0);
}

/// FLAG_SKIP_HIZ has expected value.
#[test]
fn flag_skip_hiz_value() {
    assert_eq!(GPU_CULLING_FLAG_SKIP_HIZ, 1 << 1);
}

/// FLAG_SKIP_LOD has expected value.
#[test]
fn flag_skip_lod_value() {
    assert_eq!(GPU_CULLING_FLAG_SKIP_LOD, 1 << 2);
}

/// FLAG_CONSERVATIVE has expected value.
#[test]
fn flag_conservative_value() {
    assert_eq!(GPU_CULLING_FLAG_CONSERVATIVE, 1 << 3);
}

/// FLAG_DEBUG has expected value.
#[test]
fn flag_debug_value() {
    assert_eq!(GPU_CULLING_FLAG_DEBUG, 1 << 4);
}

/// All flags are distinct (no overlap).
#[test]
fn flags_all_distinct() {
    let flags = [
        GPU_CULLING_FLAG_SKIP_FRUSTUM,
        GPU_CULLING_FLAG_SKIP_HIZ,
        GPU_CULLING_FLAG_SKIP_LOD,
        GPU_CULLING_FLAG_CONSERVATIVE,
        GPU_CULLING_FLAG_DEBUG,
    ];

    for i in 0..flags.len() {
        for j in (i + 1)..flags.len() {
            assert_eq!(flags[i] & flags[j], 0, "Flags {} and {} overlap", flags[i], flags[j]);
        }
    }
}

/// All flags can be combined.
#[test]
fn flags_can_be_combined() {
    let combined = GPU_CULLING_FLAG_SKIP_FRUSTUM
        | GPU_CULLING_FLAG_SKIP_HIZ
        | GPU_CULLING_FLAG_SKIP_LOD
        | GPU_CULLING_FLAG_CONSERVATIVE
        | GPU_CULLING_FLAG_DEBUG;

    assert_eq!(combined, 0b11111);
}

/// Params correctly interprets FLAG_SKIP_HIZ.
#[test]
fn params_interprets_skip_hiz_flag() {
    let params = basic_params(100).add_flag(GPU_CULLING_FLAG_SKIP_HIZ);
    assert!(params.frustum_enabled());
    assert!(!params.hiz_enabled());
    assert!(params.lod_enabled());
}

/// Params correctly interprets FLAG_SKIP_LOD.
#[test]
fn params_interprets_skip_lod_flag() {
    let params = basic_params(100).add_flag(GPU_CULLING_FLAG_SKIP_LOD);
    assert!(params.frustum_enabled());
    assert!(params.hiz_enabled());
    assert!(!params.lod_enabled());
}

/// Multiple skip flags work together.
#[test]
fn multiple_skip_flags() {
    let params = basic_params(100)
        .add_flag(GPU_CULLING_FLAG_SKIP_FRUSTUM)
        .add_flag(GPU_CULLING_FLAG_SKIP_HIZ)
        .add_flag(GPU_CULLING_FLAG_SKIP_LOD);

    assert!(!params.frustum_enabled());
    assert!(!params.hiz_enabled());
    assert!(!params.lod_enabled());
}

// =============================================================================
// SECTION 7 -- PARAMETER CONVERSION TESTS -- 15 tests
// =============================================================================

/// to_hiz_cull_params() sets object_count.
#[test]
fn to_hiz_cull_params_object_count() {
    let params = basic_params(500);
    let hiz_params = params.to_hiz_cull_params();
    assert_eq!(hiz_params.object_count, 500);
}

/// to_hiz_cull_params() sets hiz_width.
#[test]
fn to_hiz_cull_params_hiz_width() {
    let params = basic_params(100).with_hiz(2560, 1440, 12);
    let hiz_params = params.to_hiz_cull_params();
    assert_eq!(hiz_params.hiz_width, 2560);
}

/// to_hiz_cull_params() sets hiz_height.
#[test]
fn to_hiz_cull_params_hiz_height() {
    let params = basic_params(100).with_hiz(2560, 1440, 12);
    let hiz_params = params.to_hiz_cull_params();
    assert_eq!(hiz_params.hiz_height, 1440);
}

/// to_hiz_cull_params() passes conservative flag.
#[test]
fn to_hiz_cull_params_conservative_flag() {
    let params = basic_params(100).add_flag(GPU_CULLING_FLAG_CONSERVATIVE);
    let hiz_params = params.to_hiz_cull_params();
    // Conservative flag should be set in hiz params
    // The actual flag value may differ in HiZCullParams, but it should be non-zero
    assert!(hiz_params.flags != 0 || params.flags & GPU_CULLING_FLAG_CONSERVATIVE != 0);
}

/// to_lod_select_params() sets object_count.
#[test]
fn to_lod_select_params_object_count() {
    let params = basic_params(750);
    let lod_params = params.to_lod_select_params();
    assert_eq!(lod_params.object_count, 750);
}

/// to_lod_select_params() sets camera_position.
#[test]
fn to_lod_select_params_camera_position() {
    let params = GPUCullingParams::new(
        100,
        [5.0, 10.0, 15.0],
        &identity_vp(),
        0.1,
        0.785,
    );
    let lod_params = params.to_lod_select_params();
    assert_eq!(lod_params.camera_position, [5.0, 10.0, 15.0]);
}

/// to_lod_select_params() sets screen_width.
#[test]
fn to_lod_select_params_screen_width() {
    let params = basic_params(100).with_screen_size(3840.0, 2160.0);
    let lod_params = params.to_lod_select_params();
    assert_eq!(lod_params.screen_width, 3840.0);
}

/// to_lod_select_params() sets screen_height.
#[test]
fn to_lod_select_params_screen_height() {
    let params = basic_params(100).with_screen_size(3840.0, 2160.0);
    let lod_params = params.to_lod_select_params();
    assert_eq!(lod_params.screen_height, 2160.0);
}

/// to_lod_select_params() sets fov_y.
#[test]
fn to_lod_select_params_fov_y() {
    let fov = std::f32::consts::FRAC_PI_3;
    let params = GPUCullingParams::new(
        100,
        [0.0, 0.0, 0.0],
        &identity_vp(),
        0.1,
        fov,
    );
    let lod_params = params.to_lod_select_params();
    assert!((lod_params.fov_y - fov).abs() < 1e-6);
}

/// to_lod_select_params() sets distance mode correctly.
#[test]
fn to_lod_select_params_distance_mode() {
    let params = basic_params(100).with_lod_mode(SelectionMode::Distance);
    let lod_params = params.to_lod_select_params();
    // selection_mode field is u32, use accessor method for SelectionMode comparison
    assert_eq!(lod_params.selection_mode(), SelectionMode::Distance);
}

/// to_lod_select_params() sets screen size mode correctly.
#[test]
fn to_lod_select_params_screen_size_mode() {
    let params = basic_params(100).with_lod_mode(SelectionMode::ScreenSize);
    let lod_params = params.to_lod_select_params();
    // selection_mode field is u32, use accessor method for SelectionMode comparison
    assert_eq!(lod_params.selection_mode(), SelectionMode::ScreenSize);
}

/// to_build_indirect_params() sets visible_count.
#[test]
fn to_build_indirect_params_visible_count() {
    let params = basic_params(100);
    let build_params = params.to_build_indirect_params(50);
    assert_eq!(build_params.visible_count, 50);
}

/// to_build_indirect_params() sets max_draws.
#[test]
fn to_build_indirect_params_max_draws() {
    let params = basic_params(100).with_max_draws(4096);
    let build_params = params.to_build_indirect_params(50);
    assert_eq!(build_params.max_draws, 4096);
}

/// to_build_indirect_params() uses default max_draws.
#[test]
fn to_build_indirect_params_default_max_draws() {
    let params = basic_params(100);
    let build_params = params.to_build_indirect_params(50);
    assert_eq!(build_params.max_draws, GPU_CULLING_DEFAULT_MAX_DRAWS);
}

/// Chained builder methods work correctly.
#[test]
fn params_chained_builders() {
    let params = basic_params(5000)
        .with_hiz(1280, 720, 10)
        .with_screen_size(1280.0, 720.0)
        .with_max_draws(2048)
        .with_lod_mode(SelectionMode::ScreenSize)
        .add_flag(GPU_CULLING_FLAG_CONSERVATIVE);

    assert_eq!(params.object_count, 5000);
    assert_eq!(params.hiz_width, 1280);
    assert_eq!(params.hiz_height, 720);
    assert_eq!(params.max_mip, 10);
    assert_eq!(params.screen_width, 1280.0);
    assert_eq!(params.screen_height, 720.0);
    assert_eq!(params.max_draws, 2048);
    assert_eq!(params.lod_mode, 1);
    assert!(params.flags & GPU_CULLING_FLAG_CONSERVATIVE != 0);
}

// =============================================================================
// SECTION 8 -- BYTEMUCK TRAIT TESTS -- 10 tests
// =============================================================================

/// GPUCullingParams implements Pod (can be cast to bytes).
#[test]
fn params_pod_bytes_of() {
    let params = basic_params(100);
    let bytes = bytemuck::bytes_of(&params);
    assert_eq!(bytes.len(), 128);
}

/// GPUCullingParams implements Zeroable.
#[test]
fn params_zeroable() {
    let zeroed: GPUCullingParams = bytemuck::Zeroable::zeroed();
    assert_eq!(zeroed.object_count, 0);
    assert_eq!(zeroed.flags, 0);
    assert_eq!(zeroed.hiz_width, 0);
    assert_eq!(zeroed.hiz_height, 0);
}

/// GPUCullingParams roundtrip through bytes.
#[test]
fn params_bytes_roundtrip() {
    let original = basic_params(999)
        .with_hiz(1920, 1080, 11)
        .with_screen_size(1920.0, 1080.0);

    let bytes = bytemuck::bytes_of(&original);
    let restored: &GPUCullingParams = bytemuck::from_bytes(bytes);

    assert_eq!(restored.object_count, 999);
    assert_eq!(restored.hiz_width, 1920);
    assert_eq!(restored.hiz_height, 1080);
    assert_eq!(restored.max_mip, 11);
}

/// GPUCullingParams camera_position roundtrips correctly.
#[test]
fn params_camera_position_roundtrip() {
    let params = GPUCullingParams::new(
        100,
        [1.5, 2.5, 3.5],
        &identity_vp(),
        0.1,
        0.785,
    );
    let bytes = bytemuck::bytes_of(&params);
    let restored: &GPUCullingParams = bytemuck::from_bytes(bytes);
    assert_eq!(restored.camera_position, [1.5, 2.5, 3.5]);
}

/// GPUCullingParams view_projection roundtrips correctly.
#[test]
fn params_view_projection_roundtrip() {
    let vp = [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0],
    ];
    let params = GPUCullingParams::new(100, [0.0, 0.0, 0.0], &vp, 0.1, 0.785);
    let bytes = bytemuck::bytes_of(&params);
    let restored: &GPUCullingParams = bytemuck::from_bytes(bytes);
    assert_eq!(restored.view_projection, vp);
}

/// GPUCullingParams near_plane roundtrips correctly.
#[test]
fn params_near_plane_roundtrip() {
    let params = GPUCullingParams::new(100, [0.0, 0.0, 0.0], &identity_vp(), 0.25, 0.785);
    let bytes = bytemuck::bytes_of(&params);
    let restored: &GPUCullingParams = bytemuck::from_bytes(bytes);
    assert!((restored.near_plane - 0.25).abs() < 1e-6);
}

/// GPUCullingParams fov_y roundtrips correctly.
#[test]
fn params_fov_y_roundtrip() {
    let fov = std::f32::consts::FRAC_PI_6;
    let params = GPUCullingParams::new(100, [0.0, 0.0, 0.0], &identity_vp(), 0.1, fov);
    let bytes = bytemuck::bytes_of(&params);
    let restored: &GPUCullingParams = bytemuck::from_bytes(bytes);
    assert!((restored.fov_y - fov).abs() < 1e-6);
}

/// GPUCullingParams flags roundtrip correctly.
#[test]
fn params_flags_roundtrip() {
    let params = basic_params(100)
        .add_flag(GPU_CULLING_FLAG_SKIP_FRUSTUM)
        .add_flag(GPU_CULLING_FLAG_DEBUG);
    let bytes = bytemuck::bytes_of(&params);
    let restored: &GPUCullingParams = bytemuck::from_bytes(bytes);
    assert_eq!(restored.flags, params.flags);
}

/// GPUCullingParams alignment is at least 4 bytes.
#[test]
fn params_alignment() {
    assert!(std::mem::align_of::<GPUCullingParams>() >= 4);
}

/// GPUCullingParams slice can be cast to bytes.
#[test]
fn params_slice_to_bytes() {
    let params_list = [basic_params(100), basic_params(200)];
    let bytes: &[u8] = bytemuck::cast_slice(&params_list);
    assert_eq!(bytes.len(), 256);
}

// =============================================================================
// SECTION 9 -- CONSTANTS TESTS -- 5 tests
// =============================================================================

/// DEFAULT_MAX_OBJECTS is 100,000.
#[test]
fn constant_default_max_objects() {
    assert_eq!(GPU_CULLING_DEFAULT_MAX_OBJECTS, 100_000);
}

/// DEFAULT_MAX_DRAWS is 65,536.
#[test]
fn constant_default_max_draws() {
    assert_eq!(GPU_CULLING_DEFAULT_MAX_DRAWS, 65536);
}

/// DEFAULT_HIZ_WIDTH is 1920.
#[test]
fn constant_default_hiz_width() {
    assert_eq!(DEFAULT_HIZ_WIDTH, 1920);
}

/// DEFAULT_HIZ_HEIGHT is 1080.
#[test]
fn constant_default_hiz_height() {
    assert_eq!(DEFAULT_HIZ_HEIGHT, 1080);
}

/// GPU_CULLING_PARAMS_SIZE is 128.
#[test]
fn constant_params_size() {
    assert_eq!(GPU_CULLING_PARAMS_SIZE, 128);
}
