//! Whitebox structural tests for GPUCullingPipeline (T-WGPU-P6.6.3).
//!
//! This module validates the internal structure and behavior of the unified
//! GPU culling pipeline that integrates 5 compute stages:
//!   1. Frustum culling
//!   2. HiZ occlusion culling
//!   3. LOD selection
//!   4. Stream compaction
//!   5. Build indirect draw commands
//!
//! Tests cover:
//!   - CullingStage enum ordering and values
//!   - GPUCullingConfig defaults and flags
//!   - GPUCullingParams memory layout (128 bytes)
//!   - CullingDebugDump metrics and calculations
//!   - Builder pattern methods
//!   - Flag manipulation and checking
//!   - Workgroup calculation

use renderer_backend::gpu_driven::{
    // Main types
    GPUCullingConfig, GPUCullingParams,
    CullingStage, CullingDebugDump,
    // Helper functions
    gpu_culling_workgroups_for_objects,
    // Flags
    GPU_CULLING_FLAG_SKIP_FRUSTUM, GPU_CULLING_FLAG_SKIP_HIZ,
    GPU_CULLING_FLAG_SKIP_LOD, GPU_CULLING_FLAG_CONSERVATIVE, GPU_CULLING_FLAG_DEBUG,
    // Constants
    GPU_CULLING_DEFAULT_MAX_OBJECTS, GPU_CULLING_DEFAULT_MAX_DRAWS,
    DEFAULT_HIZ_WIDTH, DEFAULT_HIZ_HEIGHT,
    GPU_CULLING_PARAMS_SIZE, GPU_CULLING_WORKGROUP_SIZE,
    // LOD types for conversion tests
    SelectionMode,
};

// ============================================================================
// 1. CullingStage Enum Tests
// ============================================================================

mod culling_stage {
    use super::*;

    #[test]
    fn stage_count_is_five() {
        assert_eq!(CullingStage::count(), 5, "GPU culling pipeline must have exactly 5 stages");
    }

    #[test]
    fn stage_repr_values_are_sequential() {
        assert_eq!(CullingStage::FrustumCull as u8, 0);
        assert_eq!(CullingStage::HiZCull as u8, 1);
        assert_eq!(CullingStage::LodSelect as u8, 2);
        assert_eq!(CullingStage::StreamCompact as u8, 3);
        assert_eq!(CullingStage::BuildIndirect as u8, 4);
    }

    #[test]
    fn stage_index_matches_repr() {
        assert_eq!(CullingStage::FrustumCull.index(), 0);
        assert_eq!(CullingStage::HiZCull.index(), 1);
        assert_eq!(CullingStage::LodSelect.index(), 2);
        assert_eq!(CullingStage::StreamCompact.index(), 3);
        assert_eq!(CullingStage::BuildIndirect.index(), 4);
    }

    #[test]
    fn all_stages_have_non_empty_names() {
        let stages = [
            CullingStage::FrustumCull,
            CullingStage::HiZCull,
            CullingStage::LodSelect,
            CullingStage::StreamCompact,
            CullingStage::BuildIndirect,
        ];

        for stage in stages {
            let name = stage.name();
            assert!(!name.is_empty(), "Stage {:?} should have a non-empty name", stage);
            assert!(!name.contains(' '), "Stage name should not contain spaces");
        }
    }

    #[test]
    fn stage_names_are_descriptive() {
        assert_eq!(CullingStage::FrustumCull.name(), "FrustumCull");
        assert_eq!(CullingStage::HiZCull.name(), "HiZCull");
        assert_eq!(CullingStage::LodSelect.name(), "LodSelect");
        assert_eq!(CullingStage::StreamCompact.name(), "StreamCompact");
        assert_eq!(CullingStage::BuildIndirect.name(), "BuildIndirect");
    }

    #[test]
    fn stages_are_copy() {
        let stage = CullingStage::FrustumCull;
        let copy = stage;
        assert_eq!(stage, copy);
    }

    #[test]
    fn stages_are_clone() {
        let stage = CullingStage::HiZCull;
        let cloned = stage.clone();
        assert_eq!(stage, cloned);
    }

    #[test]
    fn stages_implement_debug() {
        let stage = CullingStage::LodSelect;
        let debug_str = format!("{:?}", stage);
        assert!(debug_str.contains("LodSelect"));
    }

    #[test]
    fn stages_are_hashable() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(CullingStage::FrustumCull);
        set.insert(CullingStage::HiZCull);
        set.insert(CullingStage::LodSelect);
        set.insert(CullingStage::StreamCompact);
        set.insert(CullingStage::BuildIndirect);
        assert_eq!(set.len(), 5);
    }

    #[test]
    fn stages_equality() {
        assert_eq!(CullingStage::FrustumCull, CullingStage::FrustumCull);
        assert_ne!(CullingStage::FrustumCull, CullingStage::HiZCull);
        assert_ne!(CullingStage::LodSelect, CullingStage::BuildIndirect);
    }
}

// ============================================================================
// 2. GPUCullingConfig Tests
// ============================================================================

mod gpu_culling_config {
    use super::*;

    #[test]
    fn default_max_objects() {
        let config = GPUCullingConfig::default();
        assert_eq!(config.max_objects, GPU_CULLING_DEFAULT_MAX_OBJECTS);
        assert_eq!(config.max_objects, 100_000);
    }

    #[test]
    fn default_max_draws() {
        let config = GPUCullingConfig::default();
        assert_eq!(config.max_draws, GPU_CULLING_DEFAULT_MAX_DRAWS);
        assert_eq!(config.max_draws, 65536);
    }

    #[test]
    fn default_hiz_dimensions() {
        let config = GPUCullingConfig::default();
        assert_eq!(config.hiz_width, DEFAULT_HIZ_WIDTH);
        assert_eq!(config.hiz_height, DEFAULT_HIZ_HEIGHT);
        assert_eq!(config.hiz_width, 1920);
        assert_eq!(config.hiz_height, 1080);
    }

    #[test]
    fn default_hiz_mip_count() {
        let config = GPUCullingConfig::default();
        // log2(1920) + 1 = 11
        assert_eq!(config.hiz_mip_count, 11);
    }

    #[test]
    fn default_flags_are_disabled() {
        let config = GPUCullingConfig::default();
        assert!(!config.debug_visualization);
        assert!(!config.skip_frustum_cull);
        assert!(!config.skip_hiz_cull);
        assert!(!config.skip_lod_select);
        assert!(!config.conservative_hiz);
    }

    #[test]
    fn config_is_clone() {
        let config = GPUCullingConfig::default();
        let cloned = config.clone();
        assert_eq!(cloned.max_objects, config.max_objects);
        assert_eq!(cloned.max_draws, config.max_draws);
    }

    #[test]
    fn config_is_debug() {
        let config = GPUCullingConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("GPUCullingConfig"));
        assert!(debug_str.contains("max_objects"));
    }
}

// ============================================================================
// 3. GPUCullingParams Memory Layout Tests
// ============================================================================

mod gpu_culling_params_layout {
    use super::*;

    #[test]
    fn params_size_is_128_bytes() {
        assert_eq!(std::mem::size_of::<GPUCullingParams>(), 128);
        assert_eq!(GPU_CULLING_PARAMS_SIZE, 128);
        assert_eq!(GPUCullingParams::SIZE, 128);
    }

    #[test]
    fn params_alignment_is_at_least_4() {
        assert!(std::mem::align_of::<GPUCullingParams>() >= 4);
    }

    #[test]
    fn params_is_pod() {
        let params = GPUCullingParams::default();
        let bytes = bytemuck::bytes_of(&params);
        assert_eq!(bytes.len(), 128);
    }

    #[test]
    fn params_is_zeroable() {
        let zeroed: GPUCullingParams = bytemuck::Zeroable::zeroed();
        assert_eq!(zeroed.object_count, 0);
        assert_eq!(zeroed.flags, 0);
        assert_eq!(zeroed.hiz_width, 0);
        assert_eq!(zeroed.hiz_height, 0);
    }

    #[test]
    fn bytemuck_roundtrip_preserves_data() {
        let vp = [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ];
        let original = GPUCullingParams::new(
            12345,
            [100.0, 200.0, 300.0],
            &vp,
            0.5,
            1.0,
        );

        let bytes: &[u8] = bytemuck::bytes_of(&original);
        let restored: &GPUCullingParams = bytemuck::from_bytes(bytes);

        assert_eq!(restored.object_count, 12345);
        assert_eq!(restored.camera_position, [100.0, 200.0, 300.0]);
        assert_eq!(restored.near_plane, 0.5);
        assert_eq!(restored.fov_y, 1.0);
        assert_eq!(restored.view_projection, vp);
    }

    #[test]
    fn default_values_are_sensible() {
        let params = GPUCullingParams::default();
        assert_eq!(params.object_count, 0);
        assert_eq!(params.flags, 0);
        assert_eq!(params.hiz_width, DEFAULT_HIZ_WIDTH);
        assert_eq!(params.hiz_height, DEFAULT_HIZ_HEIGHT);
        assert_eq!(params.camera_position, [0.0, 0.0, 0.0]);
        assert_eq!(params.near_plane, 0.1);
        // Identity matrix
        assert_eq!(params.view_projection[0][0], 1.0);
        assert_eq!(params.view_projection[1][1], 1.0);
        assert_eq!(params.view_projection[2][2], 1.0);
        assert_eq!(params.view_projection[3][3], 1.0);
        assert_eq!(params.screen_width, 1920.0);
        assert_eq!(params.screen_height, 1080.0);
        assert_eq!(params.max_draws, GPU_CULLING_DEFAULT_MAX_DRAWS);
        assert_eq!(params.max_mip, 10);
        assert_eq!(params.lod_mode, 0);
    }

    #[test]
    fn params_copy_and_clone() {
        let original = GPUCullingParams::default();
        let copied = original;
        let cloned = original.clone();

        assert_eq!(copied.object_count, original.object_count);
        assert_eq!(cloned.object_count, original.object_count);
    }

    #[test]
    fn params_debug_format() {
        let params = GPUCullingParams::default();
        let debug_str = format!("{:?}", params);
        assert!(debug_str.contains("GPUCullingParams"));
        assert!(debug_str.contains("object_count"));
    }

    #[test]
    fn params_partial_eq() {
        let a = GPUCullingParams::default();
        let b = GPUCullingParams::default();
        assert_eq!(a, b);
    }
}

// ============================================================================
// 4. GPUCullingParams Builder Pattern Tests
// ============================================================================

mod gpu_culling_params_builder {
    use super::*;

    #[test]
    fn new_sets_basic_fields() {
        let vp = [[1.0, 0.0, 0.0, 0.0]; 4];
        let params = GPUCullingParams::new(
            500,
            [10.0, 20.0, 30.0],
            &vp,
            0.1,
            0.785,
        );

        assert_eq!(params.object_count, 500);
        assert_eq!(params.camera_position, [10.0, 20.0, 30.0]);
        assert_eq!(params.near_plane, 0.1);
        assert_eq!(params.fov_y, 0.785);
    }

    #[test]
    fn with_hiz_sets_dimensions_and_mip() {
        let vp = [[1.0, 0.0, 0.0, 0.0]; 4];
        let params = GPUCullingParams::new(100, [0.0; 3], &vp, 0.1, 0.5)
            .with_hiz(2560, 1440, 12);

        assert_eq!(params.hiz_width, 2560);
        assert_eq!(params.hiz_height, 1440);
        assert_eq!(params.max_mip, 12);
    }

    #[test]
    fn with_screen_size_sets_dimensions() {
        let vp = [[1.0, 0.0, 0.0, 0.0]; 4];
        let params = GPUCullingParams::new(100, [0.0; 3], &vp, 0.1, 0.5)
            .with_screen_size(3840.0, 2160.0);

        assert_eq!(params.screen_width, 3840.0);
        assert_eq!(params.screen_height, 2160.0);
    }

    #[test]
    fn with_max_draws_sets_limit() {
        let vp = [[1.0, 0.0, 0.0, 0.0]; 4];
        let params = GPUCullingParams::new(100, [0.0; 3], &vp, 0.1, 0.5)
            .with_max_draws(8192);

        assert_eq!(params.max_draws, 8192);
    }

    #[test]
    fn with_lod_mode_sets_mode() {
        let vp = [[1.0, 0.0, 0.0, 0.0]; 4];

        let distance_mode = GPUCullingParams::new(100, [0.0; 3], &vp, 0.1, 0.5)
            .with_lod_mode(SelectionMode::Distance);
        assert_eq!(distance_mode.lod_mode, 0);

        let screen_mode = GPUCullingParams::new(100, [0.0; 3], &vp, 0.1, 0.5)
            .with_lod_mode(SelectionMode::ScreenSize);
        assert_eq!(screen_mode.lod_mode, 1);
    }

    #[test]
    fn with_flags_sets_flags() {
        let vp = [[1.0, 0.0, 0.0, 0.0]; 4];
        let params = GPUCullingParams::new(100, [0.0; 3], &vp, 0.1, 0.5)
            .with_flags(GPU_CULLING_FLAG_SKIP_FRUSTUM | GPU_CULLING_FLAG_DEBUG);

        assert_eq!(params.flags, GPU_CULLING_FLAG_SKIP_FRUSTUM | GPU_CULLING_FLAG_DEBUG);
    }

    #[test]
    fn add_flag_accumulates_flags() {
        let vp = [[1.0, 0.0, 0.0, 0.0]; 4];
        let params = GPUCullingParams::new(100, [0.0; 3], &vp, 0.1, 0.5)
            .add_flag(GPU_CULLING_FLAG_SKIP_FRUSTUM)
            .add_flag(GPU_CULLING_FLAG_SKIP_HIZ)
            .add_flag(GPU_CULLING_FLAG_DEBUG);

        assert!(params.flags & GPU_CULLING_FLAG_SKIP_FRUSTUM != 0);
        assert!(params.flags & GPU_CULLING_FLAG_SKIP_HIZ != 0);
        assert!(params.flags & GPU_CULLING_FLAG_DEBUG != 0);
        assert!(params.flags & GPU_CULLING_FLAG_SKIP_LOD == 0);
    }

    #[test]
    fn builder_methods_chain() {
        let vp = [[1.0, 0.0, 0.0, 0.0]; 4];
        let params = GPUCullingParams::new(100, [1.0, 2.0, 3.0], &vp, 0.1, 0.5)
            .with_hiz(1920, 1080, 10)
            .with_screen_size(1920.0, 1080.0)
            .with_max_draws(4096)
            .with_lod_mode(SelectionMode::ScreenSize)
            .add_flag(GPU_CULLING_FLAG_CONSERVATIVE);

        assert_eq!(params.object_count, 100);
        assert_eq!(params.hiz_width, 1920);
        assert_eq!(params.screen_width, 1920.0);
        assert_eq!(params.max_draws, 4096);
        assert_eq!(params.lod_mode, 1);
        assert!(params.flags & GPU_CULLING_FLAG_CONSERVATIVE != 0);
    }
}

// ============================================================================
// 5. Flag Constants and Checking Tests
// ============================================================================

mod flags {
    use super::*;

    #[test]
    fn flags_are_distinct_powers_of_two() {
        let flags = [
            GPU_CULLING_FLAG_SKIP_FRUSTUM,
            GPU_CULLING_FLAG_SKIP_HIZ,
            GPU_CULLING_FLAG_SKIP_LOD,
            GPU_CULLING_FLAG_CONSERVATIVE,
            GPU_CULLING_FLAG_DEBUG,
        ];

        for (i, &flag) in flags.iter().enumerate() {
            // Each flag is a power of two
            assert_eq!(flag.count_ones(), 1, "Flag {} is not a power of two", i);

            // No overlap with other flags
            for (j, &other) in flags.iter().enumerate() {
                if i != j {
                    assert_eq!(flag & other, 0, "Flags {} and {} overlap", i, j);
                }
            }
        }
    }

    #[test]
    fn flag_values_are_sequential_bits() {
        assert_eq!(GPU_CULLING_FLAG_SKIP_FRUSTUM, 1 << 0);
        assert_eq!(GPU_CULLING_FLAG_SKIP_HIZ, 1 << 1);
        assert_eq!(GPU_CULLING_FLAG_SKIP_LOD, 1 << 2);
        assert_eq!(GPU_CULLING_FLAG_CONSERVATIVE, 1 << 3);
        assert_eq!(GPU_CULLING_FLAG_DEBUG, 1 << 4);
    }

    #[test]
    fn frustum_enabled_default() {
        let params = GPUCullingParams::default();
        assert!(params.frustum_enabled());
    }

    #[test]
    fn frustum_disabled_with_flag() {
        let params = GPUCullingParams::default()
            .add_flag(GPU_CULLING_FLAG_SKIP_FRUSTUM);
        assert!(!params.frustum_enabled());
    }

    #[test]
    fn hiz_enabled_default() {
        let params = GPUCullingParams::default();
        assert!(params.hiz_enabled());
    }

    #[test]
    fn hiz_disabled_with_flag() {
        let params = GPUCullingParams::default()
            .add_flag(GPU_CULLING_FLAG_SKIP_HIZ);
        assert!(!params.hiz_enabled());
    }

    #[test]
    fn lod_enabled_default() {
        let params = GPUCullingParams::default();
        assert!(params.lod_enabled());
    }

    #[test]
    fn lod_disabled_with_flag() {
        let params = GPUCullingParams::default()
            .add_flag(GPU_CULLING_FLAG_SKIP_LOD);
        assert!(!params.lod_enabled());
    }

    #[test]
    fn multiple_stages_can_be_disabled() {
        let params = GPUCullingParams::default()
            .add_flag(GPU_CULLING_FLAG_SKIP_FRUSTUM)
            .add_flag(GPU_CULLING_FLAG_SKIP_HIZ)
            .add_flag(GPU_CULLING_FLAG_SKIP_LOD);

        assert!(!params.frustum_enabled());
        assert!(!params.hiz_enabled());
        assert!(!params.lod_enabled());
    }

    #[test]
    fn flags_combine_correctly() {
        let combined = GPU_CULLING_FLAG_SKIP_FRUSTUM
            | GPU_CULLING_FLAG_SKIP_HIZ
            | GPU_CULLING_FLAG_CONSERVATIVE;

        let params = GPUCullingParams::default()
            .with_flags(combined);

        assert!(!params.frustum_enabled());
        assert!(!params.hiz_enabled());
        assert!(params.lod_enabled());
    }
}

// ============================================================================
// 6. Workgroup Calculation Tests
// ============================================================================

mod workgroup_calculation {
    use super::*;

    #[test]
    fn workgroup_size_is_64() {
        assert_eq!(GPU_CULLING_WORKGROUP_SIZE, 64);
    }

    #[test]
    fn zero_objects_needs_zero_workgroups() {
        assert_eq!(gpu_culling_workgroups_for_objects(0), 0);
    }

    #[test]
    fn one_object_needs_one_workgroup() {
        assert_eq!(gpu_culling_workgroups_for_objects(1), 1);
    }

    #[test]
    fn exact_workgroup_multiple() {
        assert_eq!(gpu_culling_workgroups_for_objects(64), 1);
        assert_eq!(gpu_culling_workgroups_for_objects(128), 2);
        assert_eq!(gpu_culling_workgroups_for_objects(256), 4);
    }

    #[test]
    fn non_multiple_rounds_up() {
        assert_eq!(gpu_culling_workgroups_for_objects(65), 2);
        assert_eq!(gpu_culling_workgroups_for_objects(127), 2);
        assert_eq!(gpu_culling_workgroups_for_objects(129), 3);
    }

    #[test]
    fn large_object_counts() {
        // 100K objects / 64 = 1562.5 -> 1563 workgroups
        assert_eq!(gpu_culling_workgroups_for_objects(100_000), 1563);

        // 1M objects / 64 = 15625 workgroups
        assert_eq!(gpu_culling_workgroups_for_objects(1_000_000), 15625);
    }

    #[test]
    fn params_workgroups_method() {
        let vp = [[1.0, 0.0, 0.0, 0.0]; 4];
        let params = GPUCullingParams::new(1000, [0.0; 3], &vp, 0.1, 0.5);

        // 1000 / 64 = 15.625 -> 16 workgroups
        assert_eq!(params.workgroups(), 16);
    }

    #[test]
    fn params_workgroups_edge_cases() {
        let vp = [[1.0, 0.0, 0.0, 0.0]; 4];

        let params_zero = GPUCullingParams::new(0, [0.0; 3], &vp, 0.1, 0.5);
        assert_eq!(params_zero.workgroups(), 0);

        let params_one = GPUCullingParams::new(1, [0.0; 3], &vp, 0.1, 0.5);
        assert_eq!(params_one.workgroups(), 1);

        let params_64 = GPUCullingParams::new(64, [0.0; 3], &vp, 0.1, 0.5);
        assert_eq!(params_64.workgroups(), 1);
    }
}

// ============================================================================
// 7. CullingDebugDump Tests
// ============================================================================

mod culling_debug_dump {
    use super::*;

    #[test]
    fn default_is_zeroed() {
        let dump = CullingDebugDump::default();
        assert_eq!(dump.total_objects, 0);
        assert_eq!(dump.final_visible_count, 0);
        assert_eq!(dump.final_draw_count, 0);
        assert!(!dump.debug_enabled);
        assert_eq!(dump.stage_timings_ns, [0; 5]);
        assert_eq!(dump.visibility_counts, [0; 5]);
    }

    #[test]
    fn new_equals_default() {
        let new = CullingDebugDump::new();
        let default = CullingDebugDump::default();
        assert_eq!(new.total_objects, default.total_objects);
        assert_eq!(new.debug_enabled, default.debug_enabled);
    }

    #[test]
    fn stage_timing_ms_converts_ns_to_ms() {
        let mut dump = CullingDebugDump::new();
        dump.stage_timings_ns[CullingStage::FrustumCull.index()] = 1_000_000; // 1ms

        let timing = dump.stage_timing_ms(CullingStage::FrustumCull);
        assert!((timing - 1.0).abs() < 0.0001);
    }

    #[test]
    fn stage_timing_ms_for_all_stages() {
        let mut dump = CullingDebugDump::new();
        dump.stage_timings_ns[0] = 100_000;     // 0.1ms
        dump.stage_timings_ns[1] = 200_000;     // 0.2ms
        dump.stage_timings_ns[2] = 300_000;     // 0.3ms
        dump.stage_timings_ns[3] = 400_000;     // 0.4ms
        dump.stage_timings_ns[4] = 500_000;     // 0.5ms

        assert!((dump.stage_timing_ms(CullingStage::FrustumCull) - 0.1).abs() < 0.0001);
        assert!((dump.stage_timing_ms(CullingStage::HiZCull) - 0.2).abs() < 0.0001);
        assert!((dump.stage_timing_ms(CullingStage::LodSelect) - 0.3).abs() < 0.0001);
        assert!((dump.stage_timing_ms(CullingStage::StreamCompact) - 0.4).abs() < 0.0001);
        assert!((dump.stage_timing_ms(CullingStage::BuildIndirect) - 0.5).abs() < 0.0001);
    }

    #[test]
    fn total_timing_ms_sums_all_stages() {
        let mut dump = CullingDebugDump::new();
        dump.stage_timings_ns[0] = 100_000;
        dump.stage_timings_ns[1] = 200_000;
        dump.stage_timings_ns[2] = 100_000;
        dump.stage_timings_ns[3] = 200_000;
        dump.stage_timings_ns[4] = 100_000;
        // Total: 700_000 ns = 0.7ms

        let total = dump.total_timing_ms();
        assert!((total - 0.7).abs() < 0.0001);
    }

    #[test]
    fn cull_rate_zero_objects() {
        let dump = CullingDebugDump::new();
        assert_eq!(dump.cull_rate(), 0.0);
    }

    #[test]
    fn cull_rate_no_culling() {
        let mut dump = CullingDebugDump::new();
        dump.total_objects = 1000;
        dump.final_visible_count = 1000;
        assert_eq!(dump.cull_rate(), 0.0);
    }

    #[test]
    fn cull_rate_all_culled() {
        let mut dump = CullingDebugDump::new();
        dump.total_objects = 1000;
        dump.final_visible_count = 0;
        assert_eq!(dump.cull_rate(), 1.0);
    }

    #[test]
    fn cull_rate_partial_culling() {
        let mut dump = CullingDebugDump::new();
        dump.total_objects = 1000;
        dump.final_visible_count = 250;
        // Culled 750/1000 = 75%
        assert!((dump.cull_rate() - 0.75).abs() < 0.001);
    }

    #[test]
    fn debug_dump_is_clone() {
        let mut original = CullingDebugDump::new();
        original.total_objects = 5000;
        original.debug_enabled = true;

        let cloned = original.clone();
        assert_eq!(cloned.total_objects, 5000);
        assert!(cloned.debug_enabled);
    }

    #[test]
    fn debug_dump_is_debug() {
        let dump = CullingDebugDump::new();
        let debug_str = format!("{:?}", dump);
        assert!(debug_str.contains("CullingDebugDump"));
    }
}

// ============================================================================
// 8. Params Conversion Tests
// ============================================================================

mod params_conversion {
    use super::*;

    #[test]
    fn to_hiz_cull_params_basic() {
        let vp = [[1.0, 0.0, 0.0, 0.0]; 4];
        let params = GPUCullingParams::new(100, [0.0; 3], &vp, 0.1, 0.5)
            .with_hiz(1920, 1080, 10);

        let hiz_params = params.to_hiz_cull_params();
        assert_eq!(hiz_params.object_count, 100);
        assert_eq!(hiz_params.hiz_width, 1920);
        assert_eq!(hiz_params.hiz_height, 1080);
        // max_mip: GPUCullingParams passes (max_mip + 1) as num_mips to HiZCullParams::new,
        // which then computes max_mip = num_mips - 1, resulting in the original value
        assert_eq!(hiz_params.max_mip, 10);
    }

    #[test]
    fn to_hiz_cull_params_conservative_flag() {
        let vp = [[1.0, 0.0, 0.0, 0.0]; 4];
        let params = GPUCullingParams::new(100, [0.0; 3], &vp, 0.1, 0.5)
            .add_flag(GPU_CULLING_FLAG_CONSERVATIVE);

        let hiz_params = params.to_hiz_cull_params();
        // Verify conservative flag is passed through
        use renderer_backend::gpu_driven::FLAG_CONSERVATIVE;
        assert!(hiz_params.flags & FLAG_CONSERVATIVE != 0);
    }

    #[test]
    fn to_lod_select_params_basic() {
        let vp = [[1.0, 0.0, 0.0, 0.0]; 4];
        let params = GPUCullingParams::new(100, [10.0, 20.0, 30.0], &vp, 0.1, 0.785)
            .with_screen_size(1920.0, 1080.0);

        let lod_params = params.to_lod_select_params();
        assert_eq!(lod_params.object_count, 100);
        assert_eq!(lod_params.camera_position, [10.0, 20.0, 30.0]);
        assert_eq!(lod_params.screen_width, 1920.0);
        assert_eq!(lod_params.screen_height, 1080.0);
        assert_eq!(lod_params.fov_y, 0.785);
    }

    #[test]
    fn to_lod_select_params_distance_mode() {
        let vp = [[1.0, 0.0, 0.0, 0.0]; 4];
        let params = GPUCullingParams::new(100, [0.0; 3], &vp, 0.1, 0.5)
            .with_lod_mode(SelectionMode::Distance);

        let lod_params = params.to_lod_select_params();
        // selection_mode is stored as u32 (0 = Distance)
        assert_eq!(lod_params.selection_mode, 0);
    }

    #[test]
    fn to_lod_select_params_screen_size_mode() {
        let vp = [[1.0, 0.0, 0.0, 0.0]; 4];
        let params = GPUCullingParams::new(100, [0.0; 3], &vp, 0.1, 0.5)
            .with_lod_mode(SelectionMode::ScreenSize);

        let lod_params = params.to_lod_select_params();
        // selection_mode is stored as u32 (1 = ScreenSize)
        assert_eq!(lod_params.selection_mode, 1);
    }

    #[test]
    fn to_build_indirect_params() {
        let vp = [[1.0, 0.0, 0.0, 0.0]; 4];
        let params = GPUCullingParams::new(100, [0.0; 3], &vp, 0.1, 0.5)
            .with_max_draws(4096);

        let build_params = params.to_build_indirect_params(75);
        assert_eq!(build_params.visible_count, 75);
        assert_eq!(build_params.max_draws, 4096);
    }
}

// ============================================================================
// 9. Constants Tests
// ============================================================================

mod constants {
    use super::*;

    #[test]
    fn default_max_objects_is_100k() {
        assert_eq!(GPU_CULLING_DEFAULT_MAX_OBJECTS, 100_000);
    }

    #[test]
    fn default_max_draws_is_65536() {
        assert_eq!(GPU_CULLING_DEFAULT_MAX_DRAWS, 65536);
    }

    #[test]
    fn default_hiz_dimensions_are_1080p() {
        assert_eq!(DEFAULT_HIZ_WIDTH, 1920);
        assert_eq!(DEFAULT_HIZ_HEIGHT, 1080);
    }

    #[test]
    fn workgroup_size_is_64() {
        assert_eq!(GPU_CULLING_WORKGROUP_SIZE, 64);
    }

    #[test]
    fn params_size_is_128() {
        assert_eq!(GPU_CULLING_PARAMS_SIZE, 128);
    }
}

// ============================================================================
// 10. Integration with Other GPU Culling Types
// ============================================================================

mod integration {
    use super::*;
    use renderer_backend::gpu_driven::{
        // Stream compact types
        StreamCompactParams,
        // Build indirect types
        BuildIndirectParams,
        // HiZ types
        HiZCullParams,
    };

    #[test]
    fn hiz_cull_params_size_matches() {
        // Verify HiZCullParams can be created from GPUCullingParams
        let vp = [[1.0, 0.0, 0.0, 0.0]; 4];
        let params = GPUCullingParams::new(1000, [0.0; 3], &vp, 0.1, 0.5);
        let _hiz_params = params.to_hiz_cull_params();

        // HiZCullParams is 96 bytes (see HIZ_CULL_PARAMS_SIZE constant)
        assert_eq!(std::mem::size_of::<HiZCullParams>(), 96);
    }

    #[test]
    fn stream_compact_params_compatible() {
        let params = StreamCompactParams::new(1000);
        assert_eq!(params.object_count, 1000);
        // StreamCompactParams is 16 bytes
        assert_eq!(std::mem::size_of::<StreamCompactParams>(), 16);
    }

    #[test]
    fn build_indirect_params_compatible() {
        let params = BuildIndirectParams::new(500, 4096);
        assert_eq!(params.visible_count, 500);
        assert_eq!(params.max_draws, 4096);
        // BuildIndirectParams is 16 bytes
        assert_eq!(std::mem::size_of::<BuildIndirectParams>(), 16);
    }

    #[test]
    fn culling_stage_array_indexing() {
        // Verify stages can be used as array indices
        let timings: [u64; CullingStage::count()] = [100, 200, 300, 400, 500];

        assert_eq!(timings[CullingStage::FrustumCull.index()], 100);
        assert_eq!(timings[CullingStage::HiZCull.index()], 200);
        assert_eq!(timings[CullingStage::LodSelect.index()], 300);
        assert_eq!(timings[CullingStage::StreamCompact.index()], 400);
        assert_eq!(timings[CullingStage::BuildIndirect.index()], 500);
    }

    #[test]
    fn all_stages_fit_in_arrays() {
        let counts: [u32; 5] = [0; CullingStage::count()];

        for stage_idx in 0..CullingStage::count() {
            assert!(stage_idx < counts.len());
        }
    }
}

// ============================================================================
// 11. Edge Case Tests
// ============================================================================

mod edge_cases {
    use super::*;

    #[test]
    fn max_object_count() {
        let vp = [[1.0, 0.0, 0.0, 0.0]; 4];
        let params = GPUCullingParams::new(u32::MAX, [0.0; 3], &vp, 0.1, 0.5);
        assert_eq!(params.object_count, u32::MAX);
    }

    #[test]
    fn zero_near_plane() {
        let vp = [[1.0, 0.0, 0.0, 0.0]; 4];
        let params = GPUCullingParams::new(100, [0.0; 3], &vp, 0.0, 0.5);
        assert_eq!(params.near_plane, 0.0);
    }

    #[test]
    fn negative_camera_position() {
        let vp = [[1.0, 0.0, 0.0, 0.0]; 4];
        let params = GPUCullingParams::new(100, [-100.0, -200.0, -300.0], &vp, 0.1, 0.5);
        assert_eq!(params.camera_position, [-100.0, -200.0, -300.0]);
    }

    #[test]
    fn large_fov() {
        let vp = [[1.0, 0.0, 0.0, 0.0]; 4];
        let params = GPUCullingParams::new(100, [0.0; 3], &vp, 0.1, std::f32::consts::PI);
        assert!((params.fov_y - std::f32::consts::PI).abs() < 0.0001);
    }

    #[test]
    fn small_hiz_dimensions() {
        let vp = [[1.0, 0.0, 0.0, 0.0]; 4];
        let params = GPUCullingParams::new(100, [0.0; 3], &vp, 0.1, 0.5)
            .with_hiz(1, 1, 0);

        assert_eq!(params.hiz_width, 1);
        assert_eq!(params.hiz_height, 1);
        assert_eq!(params.max_mip, 0);
    }

    #[test]
    fn all_flags_set() {
        let all_flags = GPU_CULLING_FLAG_SKIP_FRUSTUM
            | GPU_CULLING_FLAG_SKIP_HIZ
            | GPU_CULLING_FLAG_SKIP_LOD
            | GPU_CULLING_FLAG_CONSERVATIVE
            | GPU_CULLING_FLAG_DEBUG;

        let params = GPUCullingParams::default().with_flags(all_flags);
        assert_eq!(params.flags, all_flags);
        assert!(!params.frustum_enabled());
        assert!(!params.hiz_enabled());
        assert!(!params.lod_enabled());
    }

    #[test]
    fn identity_view_projection() {
        let identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let params = GPUCullingParams::new(100, [0.0; 3], &identity, 0.1, 0.5);

        assert_eq!(params.view_projection, identity);
    }

    #[test]
    fn arbitrary_view_projection() {
        let vp = [
            [0.5, 0.1, 0.2, 0.3],
            [0.4, 0.6, 0.1, 0.2],
            [0.3, 0.2, 0.7, 0.1],
            [0.0, 0.0, -1.0, 1.0],
        ];
        let params = GPUCullingParams::new(100, [0.0; 3], &vp, 0.1, 0.5);

        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(params.view_projection[i][j], vp[i][j]);
            }
        }
    }

    #[test]
    fn cull_rate_precision() {
        let mut dump = CullingDebugDump::new();
        dump.total_objects = 3;
        dump.final_visible_count = 1;
        // Culled 2/3 = 66.666...%
        let rate = dump.cull_rate();
        assert!(rate > 0.666 && rate < 0.667);
    }

    #[test]
    fn workgroups_near_overflow() {
        // Max u32 / 64 should not overflow
        let workgroups = gpu_culling_workgroups_for_objects(u32::MAX);
        // (2^32 - 1) / 64 + 1 = 67108864
        assert!(workgroups > 0);
    }
}
