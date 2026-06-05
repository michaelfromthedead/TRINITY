// SPDX-License-Identifier: MIT
//
// WHITEBOX T-WGPU-P6.5.2: LOD Selection Shader Tests
//
// This file provides comprehensive whitebox tests for the LOD selection
// compute shader and its Rust bindings. Unlike blackbox tests, these
// tests have full access to implementation details.
//
// Test Categories:
//   1. Shader Source Validation (entry points, workgroup size, bind groups)
//   2. Struct Layout Tests (sizes, alignment, Pod/Zeroable)
//   3. Distance Selection Tests (LOD transitions, custom thresholds)
//   4. Screen Size Selection Tests (coverage-based LOD)
//   5. Blend Factor Tests (transitions, ranges)
//
// Implementation Under Test:
//   - Shader: crates/renderer-backend/shaders/lod_select.wgsl (415 lines)
//   - Rust: crates/renderer-backend/src/gpu_driven/lod_select.rs (1117 lines)

#![allow(unexpected_cfgs)]

use renderer_backend::gpu_driven::{
    // LodSelectParams and related types from lod_select.rs
    LodSelectParams, ObjectLodInput, LodSelectOutput,
    SelectionMode, object_lod_flags,
    // Renamed LodSelectBuffer from lod_select.rs
    LodSelectBuffer,
    // Constants
    LOD_SELECT_WORKGROUP_SIZE, LOD_SELECT_PARAMS_SIZE, OBJECT_LOD_INPUT_SIZE,
    LOD_SELECT_OUTPUT_SIZE, LOD_SELECT_MAX_LEVELS, DEFAULT_BLEND_RANGE,
    LOD_SELECT_SHADER,
    // Helper functions
    lod_select_workgroups_for_objects, lod_select_calculate_dispatch,
    // CPU reference functions
    cpu_distance_to_camera, cpu_screen_coverage,
    cpu_lod_select_by_distance, cpu_lod_select_by_coverage,
    cpu_select_lod, cpu_select_lod_batch,
    // From lod.rs (parent module)
    DEFAULT_LOD0_DISTANCE, DEFAULT_LOD1_DISTANCE, DEFAULT_LOD2_DISTANCE,
    COVERAGE_LOD0, COVERAGE_LOD1, COVERAGE_LOD2,
};

use bytemuck::{Pod, Zeroable};
use std::f32::consts::FRAC_PI_4;
use std::mem;

const EPSILON: f32 = 1e-5;

// =============================================================================
// Helper Functions
// =============================================================================

/// Assert two floats are approximately equal.
fn assert_approx_eq(a: f32, b: f32, msg: &str) {
    let diff = (a - b).abs();
    assert!(
        diff < EPSILON,
        "{}: expected {} but got {} (diff={})",
        msg, b, a, diff
    );
}

/// Assert float is within a range [min, max].
fn assert_in_range(val: f32, min: f32, max: f32, msg: &str) {
    assert!(
        val >= min && val <= max,
        "{}: {} not in [{}, {}]",
        msg, val, min, max
    );
}

/// Standard camera position for tests.
const CAMERA_POS: [f32; 3] = [0.0, 0.0, 0.0];

/// Standard screen dimensions.
const SCREEN_WIDTH: f32 = 1920.0;
const SCREEN_HEIGHT: f32 = 1080.0;

/// Standard thresholds for distance-based LOD.
const DEFAULT_THRESHOLDS: [f32; 3] = [10.0, 25.0, 50.0];

// =============================================================================
// Category 1: Shader Source Validation
// =============================================================================

mod shader_source_validation {
    use super::*;

    #[test]
    fn test_shader_source_not_empty() {
        assert!(
            !LOD_SELECT_SHADER.is_empty(),
            "LOD select shader source must not be empty"
        );
    }

    #[test]
    fn test_shader_contains_main_entry_point() {
        assert!(
            LOD_SELECT_SHADER.contains("fn lod_select_main"),
            "Shader must contain lod_select_main entry point"
        );
    }

    #[test]
    fn test_shader_contains_distance_only_entry_point() {
        assert!(
            LOD_SELECT_SHADER.contains("fn lod_select_distance_only"),
            "Shader must contain lod_select_distance_only entry point"
        );
    }

    #[test]
    fn test_shader_contains_screen_size_only_entry_point() {
        assert!(
            LOD_SELECT_SHADER.contains("fn lod_select_screen_size_only"),
            "Shader must contain lod_select_screen_size_only entry point"
        );
    }

    #[test]
    fn test_shader_workgroup_size_is_64() {
        // Shader must declare @workgroup_size(64, 1, 1)
        assert!(
            LOD_SELECT_SHADER.contains("@workgroup_size(64"),
            "Shader must use workgroup size 64"
        );
    }

    #[test]
    fn test_shader_has_compute_attribute() {
        // All entry points must be compute shaders
        let compute_count = LOD_SELECT_SHADER.matches("@compute").count();
        assert!(
            compute_count >= 3,
            "Shader must have at least 3 @compute entry points, found {}",
            compute_count
        );
    }

    #[test]
    fn test_shader_bind_group_0_uniform() {
        assert!(
            LOD_SELECT_SHADER.contains("@group(0) @binding(0)"),
            "Shader must have bind group 0, binding 0 for params"
        );
        assert!(
            LOD_SELECT_SHADER.contains("var<uniform>"),
            "Bind group 0 must be uniform buffer"
        );
    }

    #[test]
    fn test_shader_bind_group_1_storage() {
        assert!(
            LOD_SELECT_SHADER.contains("@group(1) @binding(0)"),
            "Shader must have bind group 1, binding 0 for object input"
        );
        assert!(
            LOD_SELECT_SHADER.contains("@group(1) @binding(1)"),
            "Shader must have bind group 1, binding 1 for output"
        );
    }

    #[test]
    fn test_shader_defines_lod_select_params_struct() {
        assert!(
            LOD_SELECT_SHADER.contains("struct LodSelectParams"),
            "Shader must define LodSelectParams struct"
        );
    }

    #[test]
    fn test_shader_defines_object_lod_input_struct() {
        assert!(
            LOD_SELECT_SHADER.contains("struct ObjectLodInput"),
            "Shader must define ObjectLodInput struct"
        );
    }

    #[test]
    fn test_shader_defines_lod_select_output_struct() {
        assert!(
            LOD_SELECT_SHADER.contains("struct LodSelectOutput"),
            "Shader must define LodSelectOutput struct"
        );
    }

    #[test]
    fn test_shader_defines_mode_constants() {
        assert!(
            LOD_SELECT_SHADER.contains("MODE_DISTANCE"),
            "Shader must define MODE_DISTANCE constant"
        );
        assert!(
            LOD_SELECT_SHADER.contains("MODE_SCREEN_SIZE"),
            "Shader must define MODE_SCREEN_SIZE constant"
        );
    }

    #[test]
    fn test_shader_defines_coverage_thresholds() {
        assert!(
            LOD_SELECT_SHADER.contains("COVERAGE_LOD0"),
            "Shader must define COVERAGE_LOD0 constant"
        );
        assert!(
            LOD_SELECT_SHADER.contains("COVERAGE_LOD1"),
            "Shader must define COVERAGE_LOD1 constant"
        );
        assert!(
            LOD_SELECT_SHADER.contains("COVERAGE_LOD2"),
            "Shader must define COVERAGE_LOD2 constant"
        );
    }

    #[test]
    fn test_shader_defines_flag_constants() {
        assert!(
            LOD_SELECT_SHADER.contains("FLAG_FORCE_LOD"),
            "Shader must define FLAG_FORCE_LOD"
        );
        assert!(
            LOD_SELECT_SHADER.contains("FLAG_ALWAYS_LOD0"),
            "Shader must define FLAG_ALWAYS_LOD0"
        );
        assert!(
            LOD_SELECT_SHADER.contains("FLAG_ALWAYS_LOD3"),
            "Shader must define FLAG_ALWAYS_LOD3"
        );
        assert!(
            LOD_SELECT_SHADER.contains("FLAG_DISABLE_BLEND"),
            "Shader must define FLAG_DISABLE_BLEND"
        );
    }

    #[test]
    fn test_shader_workgroup_constant() {
        // Note: Shader uses "WORKGROUP_SIZE", Rust re-exports as "LOD_SELECT_WORKGROUP_SIZE"
        assert!(
            LOD_SELECT_SHADER.contains("WORKGROUP_SIZE: u32 = 64"),
            "Shader must define WORKGROUP_SIZE constant as 64"
        );
    }

    #[test]
    fn test_shader_num_lod_levels() {
        assert!(
            LOD_SELECT_SHADER.contains("NUM_LOD_LEVELS: u32 = 4"),
            "Shader must define NUM_LOD_LEVELS as 4"
        );
    }
}

// =============================================================================
// Category 2: Struct Layout Tests
// =============================================================================

mod struct_layout_tests {
    use super::*;

    // -------------------------------------------------------------------------
    // LodSelectOutput Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_lod_select_output_size_is_8_bytes() {
        assert_eq!(
            mem::size_of::<LodSelectOutput>(),
            8,
            "LodSelectOutput must be exactly 8 bytes"
        );
    }

    #[test]
    fn test_lod_select_output_size_constant_matches() {
        assert_eq!(
            LOD_SELECT_OUTPUT_SIZE,
            mem::size_of::<LodSelectOutput>(),
            "LOD_SELECT_OUTPUT_SIZE constant must match actual size"
        );
        assert_eq!(LOD_SELECT_OUTPUT_SIZE, 8);
        assert_eq!(LodSelectOutput::SIZE, 8);
    }

    #[test]
    fn test_lod_select_output_implements_pod() {
        fn assert_pod<T: Pod>() {}
        assert_pod::<LodSelectOutput>();
    }

    #[test]
    fn test_lod_select_output_implements_zeroable() {
        fn assert_zeroable<T: Zeroable>() {}
        assert_zeroable::<LodSelectOutput>();
    }

    #[test]
    fn test_lod_select_output_implements_copy() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<LodSelectOutput>();
    }

    #[test]
    fn test_lod_select_output_implements_clone() {
        fn assert_clone<T: Clone>() {}
        assert_clone::<LodSelectOutput>();
    }

    #[test]
    fn test_lod_select_output_implements_debug() {
        fn assert_debug<T: std::fmt::Debug>() {}
        assert_debug::<LodSelectOutput>();
    }

    #[test]
    fn test_lod_select_output_implements_default() {
        let output = LodSelectOutput::default();
        assert_eq!(output.level, 0);
        assert_eq!(output.blend_factor, 0.0);
    }

    #[test]
    fn test_lod_select_output_field_offsets() {
        let output = LodSelectOutput::new(2, 0.75);
        let bytes = bytemuck::bytes_of(&output);

        // level at offset 0 (4 bytes)
        let level = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(level, 2);

        // blend_factor at offset 4 (4 bytes)
        let blend = f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        assert_approx_eq(blend, 0.75, "blend_factor field offset");
    }

    // -------------------------------------------------------------------------
    // LodSelectParams Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_lod_select_params_size_is_48_bytes() {
        assert_eq!(
            mem::size_of::<LodSelectParams>(),
            48,
            "LodSelectParams must be exactly 48 bytes"
        );
    }

    #[test]
    fn test_lod_select_params_size_constant_matches() {
        assert_eq!(
            LOD_SELECT_PARAMS_SIZE,
            mem::size_of::<LodSelectParams>(),
            "LOD_SELECT_PARAMS_SIZE constant must match actual size"
        );
        assert_eq!(LOD_SELECT_PARAMS_SIZE, 48);
        assert_eq!(LodSelectParams::SIZE, 48);
    }

    #[test]
    fn test_lod_select_params_implements_pod() {
        fn assert_pod<T: Pod>() {}
        assert_pod::<LodSelectParams>();
    }

    #[test]
    fn test_lod_select_params_implements_zeroable() {
        fn assert_zeroable<T: Zeroable>() {}
        assert_zeroable::<LodSelectParams>();
    }

    #[test]
    fn test_lod_select_params_implements_copy() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<LodSelectParams>();
    }

    #[test]
    fn test_lod_select_params_implements_clone() {
        fn assert_clone<T: Clone>() {}
        assert_clone::<LodSelectParams>();
    }

    #[test]
    fn test_lod_select_params_implements_debug() {
        fn assert_debug<T: std::fmt::Debug>() {}
        assert_debug::<LodSelectParams>();
    }

    #[test]
    fn test_lod_select_params_field_offsets() {
        let params = LodSelectParams::new(
            [1.0, 2.0, 3.0],
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
            FRAC_PI_4,
            SelectionMode::Distance,
            100,
        );
        let bytes = bytemuck::bytes_of(&params);

        // camera_position at offset 0 (12 bytes)
        let cam_x = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let cam_y = f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let cam_z = f32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        assert_eq!(cam_x, 1.0);
        assert_eq!(cam_y, 2.0);
        assert_eq!(cam_z, 3.0);

        // _pad0 at offset 12 (4 bytes) - should be 0
        let pad0 = f32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        assert_eq!(pad0, 0.0);

        // screen_width at offset 16
        let width = f32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
        assert_eq!(width, SCREEN_WIDTH);

        // screen_height at offset 20
        let height = f32::from_le_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);
        assert_eq!(height, SCREEN_HEIGHT);

        // fov_y at offset 24
        let fov = f32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]);
        assert_approx_eq(fov, FRAC_PI_4, "fov_y field");

        // selection_mode at offset 28
        let mode = u32::from_le_bytes([bytes[28], bytes[29], bytes[30], bytes[31]]);
        assert_eq!(mode, 0); // Distance mode

        // object_count at offset 32
        let count = u32::from_le_bytes([bytes[32], bytes[33], bytes[34], bytes[35]]);
        assert_eq!(count, 100);

        // enable_blend at offset 36
        let blend = u32::from_le_bytes([bytes[36], bytes[37], bytes[38], bytes[39]]);
        assert_eq!(blend, 0); // Disabled by default
    }

    // -------------------------------------------------------------------------
    // ObjectLodInput Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_object_lod_input_size_is_48_bytes() {
        assert_eq!(
            mem::size_of::<ObjectLodInput>(),
            48,
            "ObjectLodInput must be exactly 48 bytes"
        );
    }

    #[test]
    fn test_object_lod_input_size_constant_matches() {
        assert_eq!(
            OBJECT_LOD_INPUT_SIZE,
            mem::size_of::<ObjectLodInput>(),
            "OBJECT_LOD_INPUT_SIZE constant must match actual size"
        );
        assert_eq!(OBJECT_LOD_INPUT_SIZE, 48);
        assert_eq!(ObjectLodInput::SIZE, 48);
    }

    #[test]
    fn test_object_lod_input_implements_pod() {
        fn assert_pod<T: Pod>() {}
        assert_pod::<ObjectLodInput>();
    }

    #[test]
    fn test_object_lod_input_implements_zeroable() {
        fn assert_zeroable<T: Zeroable>() {}
        assert_zeroable::<ObjectLodInput>();
    }

    #[test]
    fn test_object_lod_input_field_offsets() {
        let obj = ObjectLodInput::new([10.0, 20.0, 30.0], 5.0, [15.0, 30.0, 60.0]);
        let bytes = bytemuck::bytes_of(&obj);

        // world_position at offset 0 (12 bytes)
        let x = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let y = f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let z = f32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        assert_eq!(x, 10.0);
        assert_eq!(y, 20.0);
        assert_eq!(z, 30.0);

        // bounding_radius at offset 12
        let radius = f32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        assert_eq!(radius, 5.0);

        // thresholds at offset 16 (12 bytes)
        let t0 = f32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
        let t1 = f32::from_le_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);
        let t2 = f32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]);
        assert_eq!(t0, 15.0);
        assert_eq!(t1, 30.0);
        assert_eq!(t2, 60.0);

        // flags at offset 32
        let flags = u32::from_le_bytes([bytes[32], bytes[33], bytes[34], bytes[35]]);
        assert_eq!(flags, 0); // Default flags

        // forced_lod at offset 36
        let forced = u32::from_le_bytes([bytes[36], bytes[37], bytes[38], bytes[39]]);
        assert_eq!(forced, 0);
    }

    // -------------------------------------------------------------------------
    // Constants Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_workgroup_size_constant() {
        assert_eq!(LOD_SELECT_WORKGROUP_SIZE, 64, "LOD_SELECT_WORKGROUP_SIZE must be 64");
    }

    #[test]
    fn test_max_lod_levels_constant() {
        assert_eq!(LOD_SELECT_MAX_LEVELS, 4, "LOD_SELECT_MAX_LEVELS must be 4");
    }

    #[test]
    fn test_default_blend_range_constant() {
        assert_approx_eq(DEFAULT_BLEND_RANGE, 0.2, "DEFAULT_BLEND_RANGE must be 0.2");
    }
}

// =============================================================================
// Category 3: Distance Selection Tests
// =============================================================================

mod distance_selection_tests {
    use super::*;

    #[test]
    fn test_lod_level_0_for_close_objects() {
        let params = LodSelectParams::distance_based(
            CAMERA_POS,
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
            FRAC_PI_4,
            1,
        );

        // Distance 5 < threshold 10 => LOD 0
        let obj = ObjectLodInput::new([5.0, 0.0, 0.0], 1.0, DEFAULT_THRESHOLDS);
        let result = cpu_select_lod(&params, &obj);
        assert_eq!(result.level, 0, "Distance 5 should be LOD 0");
    }

    #[test]
    fn test_lod_level_1_for_medium_objects() {
        let params = LodSelectParams::distance_based(
            CAMERA_POS,
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
            FRAC_PI_4,
            1,
        );

        // Distance 15 is between 10 and 25 => LOD 1
        let obj = ObjectLodInput::new([15.0, 0.0, 0.0], 1.0, DEFAULT_THRESHOLDS);
        let result = cpu_select_lod(&params, &obj);
        assert_eq!(result.level, 1, "Distance 15 should be LOD 1");
    }

    #[test]
    fn test_lod_level_2_for_far_objects() {
        let params = LodSelectParams::distance_based(
            CAMERA_POS,
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
            FRAC_PI_4,
            1,
        );

        // Distance 35 is between 25 and 50 => LOD 2
        let obj = ObjectLodInput::new([35.0, 0.0, 0.0], 1.0, DEFAULT_THRESHOLDS);
        let result = cpu_select_lod(&params, &obj);
        assert_eq!(result.level, 2, "Distance 35 should be LOD 2");
    }

    #[test]
    fn test_lod_level_3_for_very_far_objects() {
        let params = LodSelectParams::distance_based(
            CAMERA_POS,
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
            FRAC_PI_4,
            1,
        );

        // Distance 100 > threshold 50 => LOD 3
        let obj = ObjectLodInput::new([100.0, 0.0, 0.0], 1.0, DEFAULT_THRESHOLDS);
        let result = cpu_select_lod(&params, &obj);
        assert_eq!(result.level, 3, "Distance 100 should be LOD 3");
    }

    #[test]
    fn test_lod_transitions_at_exact_thresholds() {
        let params = LodSelectParams::distance_based(
            CAMERA_POS,
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
            FRAC_PI_4,
            1,
        );

        // At exactly threshold 0
        let obj = ObjectLodInput::new([10.0, 0.0, 0.0], 1.0, DEFAULT_THRESHOLDS);
        let result = cpu_select_lod(&params, &obj);
        // At threshold boundary, could be 0 or 1 depending on < vs <=
        assert!(result.level <= 1, "At LOD0 threshold should be LOD 0 or 1");

        // Just below threshold 1
        let obj = ObjectLodInput::new([24.9, 0.0, 0.0], 1.0, DEFAULT_THRESHOLDS);
        let result = cpu_select_lod(&params, &obj);
        assert_eq!(result.level, 1, "Just below 25 should be LOD 1");

        // At exactly threshold 1
        let obj = ObjectLodInput::new([25.0, 0.0, 0.0], 1.0, DEFAULT_THRESHOLDS);
        let result = cpu_select_lod(&params, &obj);
        assert!(result.level == 1 || result.level == 2, "At LOD1 threshold should be LOD 1 or 2");
    }

    #[test]
    fn test_per_object_custom_thresholds() {
        let params = LodSelectParams::distance_based(
            CAMERA_POS,
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
            FRAC_PI_4,
            1,
        );

        // Custom thresholds: much closer transitions
        let custom_thresholds = [5.0, 10.0, 15.0];
        let obj = ObjectLodInput::new([8.0, 0.0, 0.0], 1.0, custom_thresholds);
        let result = cpu_select_lod(&params, &obj);
        assert_eq!(result.level, 1, "Distance 8 with custom thresholds [5,10,15] should be LOD 1");

        // Same distance with default thresholds would be LOD 0
        let obj_default = ObjectLodInput::new([8.0, 0.0, 0.0], 1.0, DEFAULT_THRESHOLDS);
        let result_default = cpu_select_lod(&params, &obj_default);
        assert_eq!(result_default.level, 0, "Distance 8 with default thresholds should be LOD 0");
    }

    #[test]
    fn test_squared_distance_optimization() {
        // Verify the squared distance function produces correct results
        let dist = cpu_distance_to_camera([0.0, 0.0, 0.0], [3.0, 4.0, 0.0]);
        assert_approx_eq(dist, 5.0, "3-4-5 triangle distance");

        let dist_diag = cpu_distance_to_camera([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let expected = (3.0_f32).sqrt();
        assert_approx_eq(dist_diag, expected, "diagonal distance");
    }

    #[test]
    fn test_distance_calculation_symmetry() {
        let a = [10.0, 20.0, 30.0];
        let b = [40.0, 50.0, 60.0];

        let dist_ab = cpu_distance_to_camera(a, b);
        let dist_ba = cpu_distance_to_camera(b, a);

        assert_approx_eq(dist_ab, dist_ba, "Distance must be symmetric");
    }

    #[test]
    fn test_lod_selection_monotonic_with_distance() {
        let params = LodSelectParams::distance_based(
            CAMERA_POS,
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
            FRAC_PI_4,
            1,
        );

        let mut prev_lod = 0;
        let distances = [0.0, 5.0, 10.0, 15.0, 25.0, 35.0, 50.0, 75.0, 100.0];

        for dist in distances.iter() {
            let obj = ObjectLodInput::new([*dist, 0.0, 0.0], 1.0, DEFAULT_THRESHOLDS);
            let result = cpu_select_lod(&params, &obj);
            assert!(
                result.level >= prev_lod,
                "LOD should not decrease with distance: at {} got LOD {}, prev was {}",
                dist, result.level, prev_lod
            );
            prev_lod = result.level;
        }
    }

    #[test]
    fn test_cpu_lod_select_by_distance_direct() {
        // LOD 0: distance < 10
        assert_eq!(cpu_lod_select_by_distance(5.0, &DEFAULT_THRESHOLDS), 0);

        // LOD 1: 10 <= distance < 25
        assert_eq!(cpu_lod_select_by_distance(15.0, &DEFAULT_THRESHOLDS), 1);

        // LOD 2: 25 <= distance < 50
        assert_eq!(cpu_lod_select_by_distance(35.0, &DEFAULT_THRESHOLDS), 2);

        // LOD 3: distance >= 50
        assert_eq!(cpu_lod_select_by_distance(75.0, &DEFAULT_THRESHOLDS), 3);
    }
}

// =============================================================================
// Category 4: Screen Size Selection Tests
// =============================================================================

mod screen_size_selection_tests {
    use super::*;

    #[test]
    fn test_high_coverage_selects_lod_0() {
        // > 10% coverage => LOD 0
        assert_eq!(cpu_lod_select_by_coverage(0.5), 0, "50% coverage should be LOD 0");
        assert_eq!(cpu_lod_select_by_coverage(0.15), 0, "15% coverage should be LOD 0");
        assert_eq!(cpu_lod_select_by_coverage(0.10), 0, "10% coverage should be LOD 0");
    }

    #[test]
    fn test_medium_coverage_selects_lod_1() {
        // 4% <= coverage < 10% => LOD 1
        assert_eq!(cpu_lod_select_by_coverage(0.08), 1, "8% coverage should be LOD 1");
        assert_eq!(cpu_lod_select_by_coverage(0.05), 1, "5% coverage should be LOD 1");
        assert_eq!(cpu_lod_select_by_coverage(0.04), 1, "4% coverage should be LOD 1");
    }

    #[test]
    fn test_low_coverage_selects_lod_2() {
        // 1% <= coverage < 4% => LOD 2
        assert_eq!(cpu_lod_select_by_coverage(0.03), 2, "3% coverage should be LOD 2");
        assert_eq!(cpu_lod_select_by_coverage(0.02), 2, "2% coverage should be LOD 2");
        assert_eq!(cpu_lod_select_by_coverage(0.01), 2, "1% coverage should be LOD 2");
    }

    #[test]
    fn test_very_low_coverage_selects_lod_3() {
        // < 1% coverage => LOD 3
        assert_eq!(cpu_lod_select_by_coverage(0.009), 3, "0.9% coverage should be LOD 3");
        assert_eq!(cpu_lod_select_by_coverage(0.005), 3, "0.5% coverage should be LOD 3");
        assert_eq!(cpu_lod_select_by_coverage(0.001), 3, "0.1% coverage should be LOD 3");
    }

    #[test]
    fn test_screen_size_selection_mode() {
        let params = LodSelectParams::screen_size_based(
            CAMERA_POS,
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
            FRAC_PI_4,
            1,
        );

        // Very close object (high coverage) => LOD 0
        let obj = ObjectLodInput::new([2.0, 0.0, 0.0], 1.0, DEFAULT_THRESHOLDS);
        let result = cpu_select_lod(&params, &obj);
        assert_eq!(result.level, 0, "Very close object should be LOD 0");

        // Very far object (low coverage) => LOD 3
        let obj = ObjectLodInput::new([1000.0, 0.0, 0.0], 1.0, DEFAULT_THRESHOLDS);
        let result = cpu_select_lod(&params, &obj);
        assert_eq!(result.level, 3, "Very far object should be LOD 3");
    }

    #[test]
    fn test_screen_coverage_calculation() {
        // Object at distance 10 with radius 1
        let coverage = cpu_screen_coverage(CAMERA_POS, [10.0, 0.0, 0.0], 1.0, FRAC_PI_4);

        // Coverage should be positive and reasonable
        assert!(coverage > 0.0, "Coverage should be positive");
        assert!(coverage < 1.0, "Coverage should be less than 1 for small object");
    }

    #[test]
    fn test_fov_effects_on_coverage() {
        // Narrower FOV means same object covers more of screen
        let coverage_narrow = cpu_screen_coverage(
            CAMERA_POS,
            [10.0, 0.0, 0.0],
            1.0,
            0.3, // narrow FOV
        );
        let coverage_wide = cpu_screen_coverage(
            CAMERA_POS,
            [10.0, 0.0, 0.0],
            1.0,
            1.5, // wide FOV
        );

        assert!(
            coverage_narrow > coverage_wide,
            "Narrow FOV should give higher coverage: {} vs {}",
            coverage_narrow, coverage_wide
        );
    }

    #[test]
    fn test_larger_radius_increases_coverage() {
        let coverage_small = cpu_screen_coverage(CAMERA_POS, [10.0, 0.0, 0.0], 1.0, FRAC_PI_4);
        let coverage_large = cpu_screen_coverage(CAMERA_POS, [10.0, 0.0, 0.0], 5.0, FRAC_PI_4);

        assert!(
            coverage_large > coverage_small,
            "Larger radius should increase coverage: {} vs {}",
            coverage_large, coverage_small
        );
    }

    #[test]
    fn test_closer_distance_increases_coverage() {
        let coverage_far = cpu_screen_coverage(CAMERA_POS, [100.0, 0.0, 0.0], 1.0, FRAC_PI_4);
        let coverage_near = cpu_screen_coverage(CAMERA_POS, [10.0, 0.0, 0.0], 1.0, FRAC_PI_4);

        assert!(
            coverage_near > coverage_far,
            "Closer distance should increase coverage: {} vs {}",
            coverage_near, coverage_far
        );
    }

    #[test]
    fn test_coverage_at_camera_position() {
        // Edge case: object at camera position
        let coverage = cpu_screen_coverage(CAMERA_POS, CAMERA_POS, 1.0, FRAC_PI_4);

        // Should be clamped to 1.0 or handled gracefully
        assert!(!coverage.is_nan(), "Coverage at camera should not be NaN");
        assert!(coverage >= 0.0, "Coverage should be non-negative");
    }

    #[test]
    fn test_coverage_thresholds_match_constants() {
        assert_approx_eq(COVERAGE_LOD0, 0.10, "COVERAGE_LOD0 should be 0.10");
        assert_approx_eq(COVERAGE_LOD1, 0.04, "COVERAGE_LOD1 should be 0.04");
        assert_approx_eq(COVERAGE_LOD2, 0.01, "COVERAGE_LOD2 should be 0.01");
    }
}

// =============================================================================
// Category 5: Blend Factor Tests
// =============================================================================

mod blend_factor_tests {
    use super::*;

    #[test]
    fn test_blend_factor_default_is_zero() {
        let params = LodSelectParams::distance_based(
            CAMERA_POS,
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
            FRAC_PI_4,
            1,
        );

        let obj = ObjectLodInput::new([15.0, 0.0, 0.0], 1.0, DEFAULT_THRESHOLDS);
        let result = cpu_select_lod(&params, &obj);

        // CPU reference doesn't calculate blend factor
        assert_eq!(result.blend_factor, 0.0, "CPU reference returns 0 blend factor");
    }

    #[test]
    fn test_lod_output_blend_factor_range() {
        // Create outputs with various blend factors
        let outputs = [
            LodSelectOutput::new(0, 0.0),
            LodSelectOutput::new(1, 0.25),
            LodSelectOutput::new(2, 0.5),
            LodSelectOutput::new(3, 0.75),
            LodSelectOutput::new(0, 1.0),
        ];

        for output in outputs.iter() {
            assert_in_range(
                output.blend_factor,
                0.0,
                1.0,
                &format!("Blend factor for LOD {}", output.level)
            );
        }
    }

    #[test]
    fn test_lod_output_is_transitioning() {
        // Not transitioning when blend_factor is 0 or 1
        let output_0 = LodSelectOutput::new(1, 0.0);
        assert!(!output_0.is_transitioning(), "blend_factor 0.0 is not transitioning");

        let output_1 = LodSelectOutput::new(1, 1.0);
        assert!(!output_1.is_transitioning(), "blend_factor 1.0 is not transitioning");

        // Transitioning when blend_factor is between 0 and 1
        let output_mid = LodSelectOutput::new(1, 0.5);
        assert!(output_mid.is_transitioning(), "blend_factor 0.5 is transitioning");
    }

    #[test]
    fn test_lod_output_next_level() {
        assert_eq!(LodSelectOutput::new(0, 0.0).next_level(), 1);
        assert_eq!(LodSelectOutput::new(1, 0.0).next_level(), 2);
        assert_eq!(LodSelectOutput::new(2, 0.0).next_level(), 3);
        assert_eq!(LodSelectOutput::new(3, 0.0).next_level(), 3); // Clamped to max
    }

    #[test]
    fn test_lod_output_is_highest_detail() {
        assert!(LodSelectOutput::new(0, 0.0).is_highest_detail());
        assert!(!LodSelectOutput::new(1, 0.0).is_highest_detail());
        assert!(!LodSelectOutput::new(2, 0.0).is_highest_detail());
        assert!(!LodSelectOutput::new(3, 0.0).is_highest_detail());
    }

    #[test]
    fn test_lod_output_is_lowest_detail() {
        assert!(!LodSelectOutput::new(0, 0.0).is_lowest_detail());
        assert!(!LodSelectOutput::new(1, 0.0).is_lowest_detail());
        assert!(!LodSelectOutput::new(2, 0.0).is_lowest_detail());
        assert!(LodSelectOutput::new(3, 0.0).is_lowest_detail());
    }

    #[test]
    fn test_params_with_blend_enabled() {
        let params = LodSelectParams::new(
            CAMERA_POS,
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
            FRAC_PI_4,
            SelectionMode::Distance,
            100,
        )
        .with_blend(0.3);

        assert!(params.is_blend_enabled(), "Blend should be enabled");
        assert_approx_eq(params.blend_range, 0.3, "Blend range should be 0.3");
    }

    #[test]
    fn test_params_without_blend() {
        let params = LodSelectParams::new(
            CAMERA_POS,
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
            FRAC_PI_4,
            SelectionMode::Distance,
            100,
        )
        .with_blend(0.5)
        .without_blend();

        assert!(!params.is_blend_enabled(), "Blend should be disabled");
    }

    #[test]
    fn test_blend_range_clamped() {
        // Blend range should be clamped to [0.0, 1.0]
        let params_high = LodSelectParams::new(
            CAMERA_POS,
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
            FRAC_PI_4,
            SelectionMode::Distance,
            100,
        )
        .with_blend(2.0);

        assert_approx_eq(params_high.blend_range, 1.0, "Blend range should be clamped to 1.0");

        let params_low = LodSelectParams::new(
            CAMERA_POS,
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
            FRAC_PI_4,
            SelectionMode::Distance,
            100,
        )
        .with_blend(-0.5);

        assert_approx_eq(params_low.blend_range, 0.0, "Blend range should be clamped to 0.0");
    }

    #[test]
    fn test_object_disable_blend_flag() {
        let obj = ObjectLodInput::new([10.0, 0.0, 0.0], 1.0, DEFAULT_THRESHOLDS)
            .without_blend();

        assert!(
            obj.flags & object_lod_flags::DISABLE_BLEND != 0,
            "DISABLE_BLEND flag should be set"
        );
    }
}

// =============================================================================
// Object Flags Tests
// =============================================================================

mod object_flags_tests {
    use super::*;

    #[test]
    fn test_force_lod_flag() {
        let params = LodSelectParams::distance_based(
            CAMERA_POS,
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
            FRAC_PI_4,
            1,
        );

        // Force LOD 2 regardless of distance
        let obj = ObjectLodInput::new([5.0, 0.0, 0.0], 1.0, DEFAULT_THRESHOLDS)
            .with_forced_lod(2);

        let result = cpu_select_lod(&params, &obj);
        assert_eq!(result.level, 2, "Forced LOD should be 2");
    }

    #[test]
    fn test_always_lod0_flag() {
        let params = LodSelectParams::distance_based(
            CAMERA_POS,
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
            FRAC_PI_4,
            1,
        );

        // Always LOD 0 even at far distance
        let obj = ObjectLodInput::new([100.0, 0.0, 0.0], 1.0, DEFAULT_THRESHOLDS)
            .with_always_lod0();

        let result = cpu_select_lod(&params, &obj);
        assert_eq!(result.level, 0, "ALWAYS_LOD0 should force LOD 0");
    }

    #[test]
    fn test_always_lod3_flag() {
        let params = LodSelectParams::distance_based(
            CAMERA_POS,
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
            FRAC_PI_4,
            1,
        );

        // Always LOD 3 even at close distance
        let obj = ObjectLodInput::new([5.0, 0.0, 0.0], 1.0, DEFAULT_THRESHOLDS)
            .with_always_lod3();

        let result = cpu_select_lod(&params, &obj);
        assert_eq!(result.level, 3, "ALWAYS_LOD3 should force LOD 3");
    }

    #[test]
    fn test_forced_lod_clamped_to_max() {
        let obj = ObjectLodInput::new([5.0, 0.0, 0.0], 1.0, DEFAULT_THRESHOLDS)
            .with_forced_lod(10); // Request LOD 10

        // Should be clamped to LOD_SELECT_MAX_LEVELS - 1 = 3
        assert!(obj.forced_lod <= 3, "Forced LOD should be clamped to 3");
    }

    #[test]
    fn test_is_forced_check() {
        let obj_default = ObjectLodInput::new([5.0, 0.0, 0.0], 1.0, DEFAULT_THRESHOLDS);
        assert!(!obj_default.is_forced(), "Default object should not be forced");

        let obj_forced = obj_default.with_forced_lod(2);
        assert!(obj_forced.is_forced(), "Object with forced LOD should be forced");
    }

    #[test]
    fn test_flag_constants_are_distinct() {
        assert_eq!(object_lod_flags::FORCE_LOD, 1 << 0);
        assert_eq!(object_lod_flags::ALWAYS_LOD0, 1 << 1);
        assert_eq!(object_lod_flags::ALWAYS_LOD3, 1 << 2);
        assert_eq!(object_lod_flags::DISABLE_BLEND, 1 << 3);
        assert_eq!(object_lod_flags::DEFAULT, 0);

        // Verify no overlap
        let all_flags = object_lod_flags::FORCE_LOD
            | object_lod_flags::ALWAYS_LOD0
            | object_lod_flags::ALWAYS_LOD3
            | object_lod_flags::DISABLE_BLEND;

        assert_eq!(all_flags, 0b1111, "All flags should be distinct bits");
    }

    #[test]
    fn test_clear_flags() {
        let obj = ObjectLodInput::new([5.0, 0.0, 0.0], 1.0, DEFAULT_THRESHOLDS)
            .with_forced_lod(2)
            .with_always_lod0()
            .with_default_flags();

        assert_eq!(obj.flags, 0, "Flags should be cleared");
    }
}

// =============================================================================
// Dispatch and Buffer Tests
// =============================================================================

mod dispatch_tests {
    use super::*;

    #[test]
    fn test_workgroups_for_zero_objects() {
        assert_eq!(lod_select_workgroups_for_objects(0), 0);
    }

    #[test]
    fn test_workgroups_for_single_object() {
        assert_eq!(lod_select_workgroups_for_objects(1), 1);
    }

    #[test]
    fn test_workgroups_for_exactly_64_objects() {
        assert_eq!(lod_select_workgroups_for_objects(64), 1);
    }

    #[test]
    fn test_workgroups_for_65_objects() {
        assert_eq!(lod_select_workgroups_for_objects(65), 2);
    }

    #[test]
    fn test_workgroups_for_large_count() {
        assert_eq!(lod_select_workgroups_for_objects(1000), 16); // ceil(1000/64) = 16
        assert_eq!(lod_select_workgroups_for_objects(100000), 1563); // ceil(100000/64) = 1563
    }

    #[test]
    fn test_lod_select_calculate_dispatch() {
        let (x, y, z) = lod_select_calculate_dispatch(1000);
        assert_eq!(x, 16);
        assert_eq!(y, 1);
        assert_eq!(z, 1);
    }

    #[test]
    fn test_lod_buffer_capacity() {
        let buffer = LodSelectBuffer::new(1000);
        assert_eq!(buffer.capacity(), 1000);
    }

    #[test]
    fn test_lod_buffer_size() {
        let buffer = LodSelectBuffer::new(1000);
        assert_eq!(buffer.buffer_size(), 8000); // 1000 * 8 bytes
    }

    #[test]
    fn test_lod_buffer_size_for_objects() {
        assert_eq!(LodSelectBuffer::size_for_objects(1), 8);
        assert_eq!(LodSelectBuffer::size_for_objects(100), 800);
        assert_eq!(LodSelectBuffer::size_for_objects(1000), 8000);
    }
}

// =============================================================================
// Batch Processing Tests
// =============================================================================

mod batch_tests {
    use super::*;

    #[test]
    fn test_batch_processing_multiple_objects() {
        let params = LodSelectParams::distance_based(
            CAMERA_POS,
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
            FRAC_PI_4,
            3,
        );

        let objects = [
            ObjectLodInput::new([5.0, 0.0, 0.0], 1.0, DEFAULT_THRESHOLDS),  // LOD 0
            ObjectLodInput::new([15.0, 0.0, 0.0], 1.0, DEFAULT_THRESHOLDS), // LOD 1
            ObjectLodInput::new([100.0, 0.0, 0.0], 1.0, DEFAULT_THRESHOLDS), // LOD 3
        ];

        let mut output = [LodSelectOutput::default(); 3];
        cpu_select_lod_batch(&params, &objects, &mut output);

        assert_eq!(output[0].level, 0);
        assert_eq!(output[1].level, 1);
        assert_eq!(output[2].level, 3);
    }

    #[test]
    fn test_batch_processing_with_flags() {
        let params = LodSelectParams::distance_based(
            CAMERA_POS,
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
            FRAC_PI_4,
            3,
        );

        let objects = [
            ObjectLodInput::new([100.0, 0.0, 0.0], 1.0, DEFAULT_THRESHOLDS).with_always_lod0(),
            ObjectLodInput::new([5.0, 0.0, 0.0], 1.0, DEFAULT_THRESHOLDS).with_always_lod3(),
            ObjectLodInput::new([20.0, 0.0, 0.0], 1.0, DEFAULT_THRESHOLDS).with_forced_lod(2),
        ];

        let mut output = [LodSelectOutput::default(); 3];
        cpu_select_lod_batch(&params, &objects, &mut output);

        assert_eq!(output[0].level, 0, "ALWAYS_LOD0 should force LOD 0");
        assert_eq!(output[1].level, 3, "ALWAYS_LOD3 should force LOD 3");
        assert_eq!(output[2].level, 2, "FORCE_LOD(2) should force LOD 2");
    }

    #[test]
    fn test_batch_processing_empty() {
        let params = LodSelectParams::distance_based(
            CAMERA_POS,
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
            FRAC_PI_4,
            0,
        );

        let objects: [ObjectLodInput; 0] = [];
        let mut output: [LodSelectOutput; 0] = [];

        cpu_select_lod_batch(&params, &objects, &mut output);
        // Should not crash
    }
}

// =============================================================================
// Selection Mode Tests
// =============================================================================

mod selection_mode_tests {
    use super::*;

    #[test]
    fn test_selection_mode_distance() {
        assert_eq!(SelectionMode::Distance.as_u32(), 0);
    }

    #[test]
    fn test_selection_mode_screen_size() {
        assert_eq!(SelectionMode::ScreenSize.as_u32(), 1);
    }

    #[test]
    fn test_selection_mode_default() {
        let mode: SelectionMode = Default::default();
        assert_eq!(mode, SelectionMode::Distance);
    }

    #[test]
    fn test_params_selection_mode_getter() {
        let params_dist = LodSelectParams::distance_based(
            CAMERA_POS,
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
            FRAC_PI_4,
            1,
        );
        assert_eq!(params_dist.selection_mode(), SelectionMode::Distance);

        let params_screen = LodSelectParams::screen_size_based(
            CAMERA_POS,
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
            FRAC_PI_4,
            1,
        );
        assert_eq!(params_screen.selection_mode(), SelectionMode::ScreenSize);
    }
}

// =============================================================================
// Bytemuck Round-Trip Tests
// =============================================================================

mod bytemuck_tests {
    use super::*;

    #[test]
    fn test_lod_select_params_roundtrip() {
        let params = LodSelectParams::new(
            [1.0, 2.0, 3.0],
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
            FRAC_PI_4,
            SelectionMode::ScreenSize,
            100,
        )
        .with_blend(0.3);

        let bytes: &[u8] = bytemuck::bytes_of(&params);
        let roundtrip: &LodSelectParams = bytemuck::from_bytes(bytes);

        assert_eq!(*roundtrip, params);
    }

    #[test]
    fn test_object_lod_input_roundtrip() {
        let obj = ObjectLodInput::new([10.0, 20.0, 30.0], 5.0, [15.0, 30.0, 60.0])
            .with_forced_lod(2);

        let bytes: &[u8] = bytemuck::bytes_of(&obj);
        let roundtrip: &ObjectLodInput = bytemuck::from_bytes(bytes);

        assert_eq!(*roundtrip, obj);
    }

    #[test]
    fn test_lod_select_output_roundtrip() {
        let output = LodSelectOutput::new(2, 0.75);

        let bytes: &[u8] = bytemuck::bytes_of(&output);
        let roundtrip: &LodSelectOutput = bytemuck::from_bytes(bytes);

        assert_eq!(*roundtrip, output);
    }

    #[test]
    fn test_zeroed_structs() {
        let zeroed_params: LodSelectParams = bytemuck::Zeroable::zeroed();
        assert_eq!(zeroed_params.camera_position, [0.0, 0.0, 0.0]);
        assert_eq!(zeroed_params.object_count, 0);

        let zeroed_obj: ObjectLodInput = bytemuck::Zeroable::zeroed();
        assert_eq!(zeroed_obj.world_position, [0.0, 0.0, 0.0]);
        assert_eq!(zeroed_obj.flags, 0);

        let zeroed_output: LodSelectOutput = bytemuck::Zeroable::zeroed();
        assert_eq!(zeroed_output.level, 0);
        assert_eq!(zeroed_output.blend_factor, 0.0);
    }
}

// =============================================================================
// Builder Pattern Tests
// =============================================================================

mod builder_tests {
    use super::*;

    #[test]
    fn test_params_builder_chain() {
        let params = LodSelectParams::new(
            [0.0, 0.0, 0.0],
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
            FRAC_PI_4,
            SelectionMode::Distance,
            100,
        )
        .with_blend(0.3)
        .with_object_count(200)
        .with_camera_position([10.0, 20.0, 30.0]);

        assert!(params.is_blend_enabled());
        assert_approx_eq(params.blend_range, 0.3, "blend_range");
        assert_eq!(params.object_count, 200);
        assert_eq!(params.camera_position, [10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_object_input_builder_chain() {
        let obj = ObjectLodInput::with_default_thresholds([10.0, 0.0, 0.0], 5.0)
            .with_position([20.0, 0.0, 0.0])
            .with_radius(10.0)
            .with_forced_lod(2)
            .without_blend();

        assert_eq!(obj.world_position, [20.0, 0.0, 0.0]);
        assert_eq!(obj.bounding_radius, 10.0);
        assert!(obj.is_forced());
        assert_eq!(obj.forced_lod, 2);
        assert!(obj.flags & object_lod_flags::DISABLE_BLEND != 0);
    }

    #[test]
    fn test_object_input_with_default_thresholds() {
        let obj = ObjectLodInput::with_default_thresholds([10.0, 0.0, 0.0], 5.0);

        assert_eq!(obj.thresholds[0], DEFAULT_LOD0_DISTANCE);
        assert_eq!(obj.thresholds[1], DEFAULT_LOD1_DISTANCE);
        assert_eq!(obj.thresholds[2], DEFAULT_LOD2_DISTANCE);
    }
}

// =============================================================================
// Edge Cases and Stress Tests
// =============================================================================

mod edge_case_tests {
    use super::*;

    #[test]
    fn test_object_at_camera_position() {
        let params = LodSelectParams::distance_based(
            CAMERA_POS,
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
            FRAC_PI_4,
            1,
        );

        let obj = ObjectLodInput::new(CAMERA_POS, 1.0, DEFAULT_THRESHOLDS);
        let result = cpu_select_lod(&params, &obj);

        // At zero distance, should be LOD 0
        assert_eq!(result.level, 0, "Object at camera should be LOD 0");
    }

    #[test]
    fn test_very_large_distance() {
        let params = LodSelectParams::distance_based(
            CAMERA_POS,
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
            FRAC_PI_4,
            1,
        );

        let obj = ObjectLodInput::new([1e6, 0.0, 0.0], 1.0, DEFAULT_THRESHOLDS);
        let result = cpu_select_lod(&params, &obj);

        // Very far should be LOD 3
        assert_eq!(result.level, 3, "Very far object should be LOD 3");
    }

    #[test]
    fn test_very_small_radius() {
        let coverage = cpu_screen_coverage(CAMERA_POS, [10.0, 0.0, 0.0], 0.001, FRAC_PI_4);
        assert!(coverage >= 0.0, "Small radius should have non-negative coverage");
    }

    #[test]
    fn test_zero_radius() {
        let coverage = cpu_screen_coverage(CAMERA_POS, [10.0, 0.0, 0.0], 0.0, FRAC_PI_4);
        assert_approx_eq(coverage, 0.0, "Zero radius should have zero coverage");
    }

    #[test]
    fn test_consistency_across_iterations() {
        let params = LodSelectParams::distance_based(
            CAMERA_POS,
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
            FRAC_PI_4,
            1,
        );

        let obj = ObjectLodInput::new([20.0, 0.0, 0.0], 1.0, DEFAULT_THRESHOLDS);

        for _ in 0..100 {
            let result1 = cpu_select_lod(&params, &obj);
            let result2 = cpu_select_lod(&params, &obj);
            assert_eq!(result1.level, result2.level, "LOD selection must be deterministic");
        }
    }

    #[test]
    fn test_many_objects_batch() {
        let params = LodSelectParams::distance_based(
            CAMERA_POS,
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
            FRAC_PI_4,
            100,
        );

        let objects: Vec<ObjectLodInput> = (0..100)
            .map(|i| {
                let dist = (i as f32) * 10.0;
                ObjectLodInput::new([dist, 0.0, 0.0], 1.0, DEFAULT_THRESHOLDS)
            })
            .collect();

        let mut output = vec![LodSelectOutput::default(); 100];
        cpu_select_lod_batch(&params, &objects, &mut output);

        // Verify all outputs are valid
        for (i, out) in output.iter().enumerate() {
            assert!(out.level < 4, "Object {} should have valid LOD level", i);
        }
    }
}

// =============================================================================
// Summary Test
// =============================================================================

#[test]
fn test_all_exports_accessible() {
    // Types
    let _: LodSelectParams = LodSelectParams::default();
    let _: ObjectLodInput = ObjectLodInput::default();
    let _: LodSelectOutput = LodSelectOutput::default();
    let _: SelectionMode = SelectionMode::Distance;
    let _: LodSelectBuffer = LodSelectBuffer::new(100);

    // Constants
    assert_eq!(LOD_SELECT_WORKGROUP_SIZE, 64);
    assert_eq!(LOD_SELECT_PARAMS_SIZE, 48);
    assert_eq!(OBJECT_LOD_INPUT_SIZE, 48);
    assert_eq!(LOD_SELECT_OUTPUT_SIZE, 8);
    assert_eq!(LOD_SELECT_MAX_LEVELS, 4);
    assert!(!LOD_SELECT_SHADER.is_empty());

    // Functions
    let _ = lod_select_workgroups_for_objects(100);
    let _ = lod_select_calculate_dispatch(100);
    let _ = cpu_distance_to_camera([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
    let _ = cpu_screen_coverage([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1.0, FRAC_PI_4);
    let _ = cpu_lod_select_by_distance(10.0, &DEFAULT_THRESHOLDS);
    let _ = cpu_lod_select_by_coverage(0.5);

    // Flags
    assert!(object_lod_flags::FORCE_LOD > 0);
    assert!(object_lod_flags::ALWAYS_LOD0 > 0);
    assert!(object_lod_flags::ALWAYS_LOD3 > 0);
    assert!(object_lod_flags::DISABLE_BLEND > 0);
    assert_eq!(object_lod_flags::DEFAULT, 0);
}
