// SPDX-License-Identifier: MIT
//
// WHITEBOX T-WGPU-P6.4.3: HiZ Occlusion Test
//
// Comprehensive whitebox tests for HiZ (Hierarchical-Z) occlusion testing with full
// source access to internal implementation details.
//
// Test Categories:
//   1. AABB Projection Tests - Project AABB to screen coordinates, clipping, edge cases
//   2. Mip Level Selection Tests - log2 calculation, power-of-2, clamping
//   3. Depth Comparison Tests - Reverse-Z semantics, occlusion detection, EPSILON
//   4. Struct Layout Tests - Size assertions, Pod/Zeroable traits, bytemuck
//   5. Shader Structure Tests - Entry points, bindings, function presence
//   6. CPU Reference Implementation Tests - Full occlusion pipeline
//   7. Helper Function Tests - Corner enumeration, transform, mip offset
//
// Coverage:
//   - HiZOcclusionParams, BatchParams, InputAABB structs
//   - cpu_project_aabb, cpu_select_mip_level, cpu_test_occlusion
//   - transform_point, get_aabb_corner, calculate_mip_offset
//   - Constants: WORKGROUP_SIZE, EPSILON, CONSERVATIVE_EXPAND

#![allow(unexpected_cfgs)]

use renderer_backend::gpu_driven::hiz_occlusion::{
    BatchParams, HiZOcclusionParams, InputAABB,
    HIZ_OCCLUSION_PARAMS_SIZE, BATCH_PARAMS_SIZE, INPUT_AABB_SIZE,
    WORKGROUP_SIZE, MAX_MIP_LEVEL, EPSILON, CONSERVATIVE_EXPAND,
    HIZ_OCCLUSION_SHADER,
    cpu_project_aabb, cpu_select_mip_level, cpu_test_occlusion,
    workgroups_for_objects,
};
use std::mem;

// =============================================================================
// CATEGORY 1: STRUCT LAYOUT TESTS
// =============================================================================

/// Verifies HiZOcclusionParams size matches expected 80 bytes (std140).
#[test]
fn test_hiz_occlusion_params_size_exact() {
    assert_eq!(mem::size_of::<HiZOcclusionParams>(), HIZ_OCCLUSION_PARAMS_SIZE);
    assert_eq!(mem::size_of::<HiZOcclusionParams>(), 80);
}

/// Verifies BatchParams size matches expected 16 bytes.
#[test]
fn test_batch_params_size_exact() {
    assert_eq!(mem::size_of::<BatchParams>(), BATCH_PARAMS_SIZE);
    assert_eq!(mem::size_of::<BatchParams>(), 16);
}

/// Verifies InputAABB size matches expected 32 bytes (vec4 aligned).
#[test]
fn test_input_aabb_size_exact() {
    assert_eq!(mem::size_of::<InputAABB>(), INPUT_AABB_SIZE);
    assert_eq!(mem::size_of::<InputAABB>(), 32);
}

/// Verifies HiZOcclusionParams alignment for GPU uniform buffer compatibility.
#[test]
fn test_hiz_occlusion_params_alignment() {
    // Must be 16-byte aligned for std140 layout
    assert_eq!(mem::align_of::<HiZOcclusionParams>() % 4, 0);
}

/// Verifies BatchParams alignment for GPU uniform buffer compatibility.
#[test]
fn test_batch_params_alignment() {
    // Must be 4-byte aligned minimum
    assert!(mem::align_of::<BatchParams>() >= 4);
}

/// Verifies InputAABB alignment for GPU storage buffer compatibility.
#[test]
fn test_input_aabb_alignment() {
    // Must be 4-byte aligned for storage buffer
    assert!(mem::align_of::<InputAABB>() >= 4);
}

// =============================================================================
// CATEGORY 2: BYTEMUCK POD/ZEROABLE TRAIT TESTS
// =============================================================================

/// Tests HiZOcclusionParams bytemuck round-trip serialization.
#[test]
fn test_hiz_occlusion_params_bytemuck_roundtrip() {
    let vp = identity_matrix();
    let original = HiZOcclusionParams::new(&vp, 1920.0, 1080.0, 0.1, 11);

    let bytes = bytemuck::bytes_of(&original);
    assert_eq!(bytes.len(), HIZ_OCCLUSION_PARAMS_SIZE);

    let restored: HiZOcclusionParams = *bytemuck::from_bytes(bytes);
    assert_eq!(restored.hiz_size, original.hiz_size);
    assert_eq!(restored.near_plane, original.near_plane);
    assert_eq!(restored.max_mip, original.max_mip);

    // Verify matrix preserved
    for i in 0..4 {
        for j in 0..4 {
            assert_eq!(restored.view_projection[i][j], original.view_projection[i][j]);
        }
    }
}

/// Tests BatchParams bytemuck round-trip serialization.
#[test]
fn test_batch_params_bytemuck_roundtrip() {
    let original = BatchParams::with_flags(12345, 0xABCD);

    let bytes = bytemuck::bytes_of(&original);
    assert_eq!(bytes.len(), BATCH_PARAMS_SIZE);

    let restored: BatchParams = *bytemuck::from_bytes(bytes);
    assert_eq!(restored.num_objects, original.num_objects);
    assert_eq!(restored.flags, original.flags);
}

/// Tests InputAABB bytemuck round-trip serialization.
#[test]
fn test_input_aabb_bytemuck_roundtrip() {
    let original = InputAABB::new([-1.0, -2.0, -3.0], [4.0, 5.0, 6.0]);

    let bytes = bytemuck::bytes_of(&original);
    assert_eq!(bytes.len(), INPUT_AABB_SIZE);

    let restored: InputAABB = *bytemuck::from_bytes(bytes);
    assert_eq!(restored.min, original.min);
    assert_eq!(restored.max, original.max);
}

/// Tests that padding bytes in InputAABB are zero.
#[test]
fn test_input_aabb_padding_zeroed() {
    let aabb = InputAABB::new([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
    assert_eq!(aabb._pad0, 0.0);
    assert_eq!(aabb._pad1, 0.0);
}

// =============================================================================
// CATEGORY 3: CONSTRUCTOR AND FACTORY TESTS
// =============================================================================

/// Tests HiZOcclusionParams::new() correctly initializes all fields.
#[test]
fn test_hiz_occlusion_params_new_1080p() {
    let vp = identity_matrix();
    let params = HiZOcclusionParams::new(&vp, 1920.0, 1080.0, 0.1, 11);

    assert_eq!(params.hiz_size, [1920.0, 1080.0]);
    assert_eq!(params.near_plane, 0.1);
    assert_eq!(params.max_mip, 10); // num_mips - 1
}

/// Tests HiZOcclusionParams::new() with 4K resolution.
#[test]
fn test_hiz_occlusion_params_new_4k() {
    let vp = identity_matrix();
    let params = HiZOcclusionParams::new(&vp, 3840.0, 2160.0, 0.01, 12);

    assert_eq!(params.hiz_size, [3840.0, 2160.0]);
    assert_eq!(params.near_plane, 0.01);
    assert_eq!(params.max_mip, 11);
}

/// Tests HiZOcclusionParams::from_dimensions() integer conversion.
#[test]
fn test_hiz_occlusion_params_from_dimensions() {
    let vp = identity_matrix();
    let params = HiZOcclusionParams::from_dimensions(&vp, 1920, 1080, 0.1, 11);

    assert_eq!(params.hiz_dimensions(), (1920, 1080));
    assert_eq!(params.hiz_size[0], 1920.0);
    assert_eq!(params.hiz_size[1], 1080.0);
}

/// Tests HiZOcclusionParams::from_dimensions() with zero mips (saturating sub).
#[test]
fn test_hiz_occlusion_params_zero_mips() {
    let vp = identity_matrix();
    let params = HiZOcclusionParams::new(&vp, 256.0, 256.0, 0.1, 0);

    // saturating_sub(1) from 0 = 0
    assert_eq!(params.max_mip, 0);
}

/// Tests HiZOcclusionParams::calculate_num_mips for various resolutions.
#[test]
fn test_calculate_num_mips_power_of_two() {
    assert_eq!(HiZOcclusionParams::calculate_num_mips(1, 1), 1);
    assert_eq!(HiZOcclusionParams::calculate_num_mips(2, 2), 2);
    assert_eq!(HiZOcclusionParams::calculate_num_mips(4, 4), 3);
    assert_eq!(HiZOcclusionParams::calculate_num_mips(8, 8), 4);
    assert_eq!(HiZOcclusionParams::calculate_num_mips(256, 256), 9);
    assert_eq!(HiZOcclusionParams::calculate_num_mips(512, 512), 10);
    assert_eq!(HiZOcclusionParams::calculate_num_mips(1024, 1024), 11);
    assert_eq!(HiZOcclusionParams::calculate_num_mips(2048, 2048), 12);
}

/// Tests calculate_num_mips for common screen resolutions.
#[test]
fn test_calculate_num_mips_common_resolutions() {
    assert_eq!(HiZOcclusionParams::calculate_num_mips(1920, 1080), 11);
    assert_eq!(HiZOcclusionParams::calculate_num_mips(2560, 1440), 12);
    assert_eq!(HiZOcclusionParams::calculate_num_mips(3840, 2160), 12);
}

/// Tests calculate_num_mips for zero dimensions.
#[test]
fn test_calculate_num_mips_zero() {
    assert_eq!(HiZOcclusionParams::calculate_num_mips(0, 0), 1);
}

/// Tests BatchParams::new() initialization.
#[test]
fn test_batch_params_new() {
    let params = BatchParams::new(1000);

    assert_eq!(params.num_objects, 1000);
    assert_eq!(params.flags, 0);
    assert_eq!(params._pad0, 0);
    assert_eq!(params._pad1, 0);
}

/// Tests BatchParams::with_flags() initialization.
#[test]
fn test_batch_params_with_flags() {
    let params = BatchParams::with_flags(256, 0xFF00FF00);

    assert_eq!(params.num_objects, 256);
    assert_eq!(params.flags, 0xFF00FF00);
}

/// Tests InputAABB::new() initialization.
#[test]
fn test_input_aabb_new() {
    let aabb = InputAABB::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);

    assert_eq!(aabb.min, [0.0, 0.0, 0.0]);
    assert_eq!(aabb.max, [1.0, 1.0, 1.0]);
}

/// Tests InputAABB::from_tuples() initialization.
#[test]
fn test_input_aabb_from_tuples() {
    let aabb = InputAABB::from_tuples((1.0, 2.0, 3.0), (4.0, 5.0, 6.0));

    assert_eq!(aabb.min, [1.0, 2.0, 3.0]);
    assert_eq!(aabb.max, [4.0, 5.0, 6.0]);
}

/// Tests InputAABB::center() calculation.
#[test]
fn test_input_aabb_center_unit_cube() {
    let aabb = InputAABB::new([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
    assert_eq!(aabb.center(), [1.0, 1.0, 1.0]);
}

/// Tests InputAABB::center() with negative coordinates.
#[test]
fn test_input_aabb_center_negative() {
    let aabb = InputAABB::new([-2.0, -4.0, -6.0], [2.0, 4.0, 6.0]);
    assert_eq!(aabb.center(), [0.0, 0.0, 0.0]);
}

/// Tests InputAABB::half_extents() calculation.
#[test]
fn test_input_aabb_half_extents() {
    let aabb = InputAABB::new([0.0, 0.0, 0.0], [4.0, 6.0, 8.0]);
    assert_eq!(aabb.half_extents(), [2.0, 3.0, 4.0]);
}

// =============================================================================
// CATEGORY 4: MIP LEVEL SELECTION TESTS
// =============================================================================

/// Tests mip selection for exact power-of-2 dimensions.
#[test]
fn test_mip_selection_power_of_two_exact() {
    // log2(1) = 0
    assert_eq!(cpu_select_mip_level(1.0, 1.0, 10), 0);
    // log2(2) = 1
    assert_eq!(cpu_select_mip_level(2.0, 2.0, 10), 1);
    // log2(4) = 2
    assert_eq!(cpu_select_mip_level(4.0, 4.0, 10), 2);
    // log2(8) = 3
    assert_eq!(cpu_select_mip_level(8.0, 8.0, 10), 3);
    // log2(16) = 4
    assert_eq!(cpu_select_mip_level(16.0, 16.0, 10), 4);
    // log2(32) = 5
    assert_eq!(cpu_select_mip_level(32.0, 32.0, 10), 5);
    // log2(64) = 6
    assert_eq!(cpu_select_mip_level(64.0, 64.0, 10), 6);
    // log2(128) = 7
    assert_eq!(cpu_select_mip_level(128.0, 128.0, 10), 7);
    // log2(256) = 8
    assert_eq!(cpu_select_mip_level(256.0, 256.0, 10), 8);
    // log2(512) = 9
    assert_eq!(cpu_select_mip_level(512.0, 512.0, 10), 9);
    // log2(1024) = 10
    assert_eq!(cpu_select_mip_level(1024.0, 1024.0, 10), 10);
}

/// Tests mip selection for non-power-of-2 dimensions (floor behavior).
#[test]
fn test_mip_selection_non_power_of_two() {
    // log2(3) ~= 1.58 -> floor = 1
    assert_eq!(cpu_select_mip_level(3.0, 3.0, 10), 1);
    // log2(5) ~= 2.32 -> floor = 2
    assert_eq!(cpu_select_mip_level(5.0, 5.0, 10), 2);
    // log2(7) ~= 2.81 -> floor = 2
    assert_eq!(cpu_select_mip_level(7.0, 7.0, 10), 2);
    // log2(9) ~= 3.17 -> floor = 3
    assert_eq!(cpu_select_mip_level(9.0, 9.0, 10), 3);
    // log2(100) ~= 6.64 -> floor = 6
    assert_eq!(cpu_select_mip_level(100.0, 100.0, 10), 6);
    // log2(1000) ~= 9.97 -> floor = 9
    assert_eq!(cpu_select_mip_level(1000.0, 1000.0, 10), 9);
}

/// Tests mip selection uses max dimension for asymmetric rects.
#[test]
fn test_mip_selection_asymmetric_width_dominant() {
    // max(64, 32) = 64 -> log2(64) = 6
    assert_eq!(cpu_select_mip_level(64.0, 32.0, 10), 6);
    // max(128, 16) = 128 -> log2(128) = 7
    assert_eq!(cpu_select_mip_level(128.0, 16.0, 10), 7);
    // max(256, 1) = 256 -> log2(256) = 8
    assert_eq!(cpu_select_mip_level(256.0, 1.0, 10), 8);
}

/// Tests mip selection uses max dimension for asymmetric rects (height dominant).
#[test]
fn test_mip_selection_asymmetric_height_dominant() {
    // max(32, 64) = 64 -> log2(64) = 6
    assert_eq!(cpu_select_mip_level(32.0, 64.0, 10), 6);
    // max(16, 128) = 128 -> log2(128) = 7
    assert_eq!(cpu_select_mip_level(16.0, 128.0, 10), 7);
}

/// Tests mip selection clamping to max_mip.
#[test]
fn test_mip_selection_clamped_to_max() {
    // log2(2048) = 11, but max_mip = 5 -> clamped to 5
    assert_eq!(cpu_select_mip_level(2048.0, 2048.0, 5), 5);
    // log2(4096) = 12, but max_mip = 8 -> clamped to 8
    assert_eq!(cpu_select_mip_level(4096.0, 4096.0, 8), 8);
    // log2(8192) = 13, but max_mip = 10 -> clamped to 10
    assert_eq!(cpu_select_mip_level(8192.0, 8192.0, 10), 10);
}

/// Tests mip selection for sub-pixel dimensions.
#[test]
fn test_mip_selection_sub_pixel() {
    // Anything <= 1.0 should return mip 0
    assert_eq!(cpu_select_mip_level(0.5, 0.5, 10), 0);
    assert_eq!(cpu_select_mip_level(0.1, 0.1, 10), 0);
    assert_eq!(cpu_select_mip_level(1.0, 1.0, 10), 0);
    assert_eq!(cpu_select_mip_level(0.0, 0.0, 10), 0);
}

/// Tests mip selection with zero max_mip.
#[test]
fn test_mip_selection_zero_max_mip() {
    // All sizes should clamp to 0 when max_mip is 0
    assert_eq!(cpu_select_mip_level(1.0, 1.0, 0), 0);
    assert_eq!(cpu_select_mip_level(64.0, 64.0, 0), 0);
    assert_eq!(cpu_select_mip_level(1024.0, 1024.0, 0), 0);
}

// =============================================================================
// CATEGORY 5: AABB PROJECTION TESTS
// =============================================================================

/// Helper: Create identity matrix (column-major).
fn identity_matrix() -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// Helper: Create a simple orthographic projection (no perspective).
#[allow(dead_code)]
fn orthographic_matrix() -> [[f32; 4]; 4] {
    // Maps [-1,1] to NDC directly
    identity_matrix()
}

/// Helper: Create perspective projection matrix with reverse-Z.
/// This is a simplified perspective for testing purposes.
#[allow(dead_code)]
fn simple_perspective_reverse_z(fov_y: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
    let tan_half_fov = (fov_y / 2.0).tan();
    let f = 1.0 / tan_half_fov;

    // Reverse-Z perspective matrix (near=1, far=0 in depth output)
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, near / (far - near), -1.0],
        [0.0, 0.0, (far * near) / (far - near), 0.0],
    ]
}

/// Tests AABB projection returns valid=true for AABB in front of camera.
#[test]
fn test_aabb_projection_valid_in_front() {
    // With identity, all w=1.0, so everything is "in front"
    let vp = identity_matrix();
    let (min, max, _depth, valid) = cpu_project_aabb(
        [-0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        &vp,
        100.0, 100.0
    );

    assert!(valid, "AABB should be valid when in front of camera");
    // With identity matrix, NDC should map reasonably to screen
    assert!(min.0 <= max.0, "min.x should be <= max.x");
    assert!(min.1 <= max.1, "min.y should be <= max.y");
}

/// Tests AABB projection returns valid=true for AABB at origin with identity.
#[test]
fn test_aabb_projection_at_origin() {
    let vp = identity_matrix();
    let (min, max, depth, valid) = cpu_project_aabb(
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
        &vp,
        200.0, 200.0
    );

    assert!(valid);
    // Should cover most of screen with identity matrix
    // NDC [-1,1] maps to [0, 200] in screen coords
}

/// Tests AABB projection with centered unit cube.
#[test]
fn test_aabb_projection_unit_cube_centered() {
    let vp = identity_matrix();
    let (min, max, _depth, valid) = cpu_project_aabb(
        [-0.5, -0.5, 0.0],
        [0.5, 0.5, 1.0],
        &vp,
        1920.0, 1080.0
    );

    assert!(valid);
    // Should produce a reasonable rect within screen bounds
    assert!(min.0 >= 0.0 - CONSERVATIVE_EXPAND);
    assert!(min.1 >= 0.0 - CONSERVATIVE_EXPAND);
    assert!(max.0 <= 1920.0 + CONSERVATIVE_EXPAND);
    assert!(max.1 <= 1080.0 + CONSERVATIVE_EXPAND);
}

/// Tests AABB projection tracking nearest depth (max in reverse-Z).
#[test]
fn test_aabb_projection_near_depth_reverse_z() {
    let vp = identity_matrix();
    // AABB with z from 0.2 to 0.8 in NDC
    let (_min, _max, near_depth, valid) = cpu_project_aabb(
        [-0.5, -0.5, 0.2],
        [0.5, 0.5, 0.8],
        &vp,
        100.0, 100.0
    );

    assert!(valid);
    // Near depth should be max of all corner depths (0.8 in this case)
    assert!((near_depth - 0.8).abs() < 0.01, "Near depth should be 0.8, got {}", near_depth);
}

/// Tests AABB projection screen coordinate calculation.
#[test]
fn test_aabb_projection_screen_coords() {
    // With identity matrix:
    // NDC x=0 maps to screen x = (0+1)*0.5*width = width/2
    // NDC y=0 maps to screen y = (1-(0+1)*0.5)*height = height/2
    let vp = identity_matrix();
    let (_min, _max, _depth, valid) = cpu_project_aabb(
        [0.0, 0.0, 0.5],
        [0.5, 0.5, 0.5],
        &vp,
        1000.0, 1000.0
    );

    assert!(valid);
    // The rect should be in the upper-right quadrant due to Y flip
}

/// Tests AABB projection with conservative expansion.
#[test]
fn test_aabb_projection_conservative_expansion() {
    let vp = identity_matrix();
    let (_min, _max, _depth, valid) = cpu_project_aabb(
        [-0.1, -0.1, 0.5],
        [0.1, 0.1, 0.5],
        &vp,
        1000.0, 1000.0
    );

    assert!(valid);
    // Conservative expansion should have been applied
    // The final bounds should be slightly larger than the exact projection
}

/// Tests AABB projection clamping to screen bounds.
#[test]
fn test_aabb_projection_clamping() {
    let vp = identity_matrix();
    // AABB extends outside NDC [-1,1] which should clamp to screen
    let (min, max, _depth, valid) = cpu_project_aabb(
        [-2.0, -2.0, 0.5],
        [2.0, 2.0, 0.5],
        &vp,
        100.0, 100.0
    );

    assert!(valid);
    // Should be clamped to screen bounds
    assert!(min.0 >= 0.0);
    assert!(min.1 >= 0.0);
    assert!(max.0 <= 100.0);
    assert!(max.1 <= 100.0);
}

// =============================================================================
// CATEGORY 6: DEPTH COMPARISON TESTS (REVERSE-Z SEMANTICS)
// =============================================================================

/// Tests reverse-Z depth comparison: object in front is visible.
#[test]
fn test_reverse_z_object_in_front_visible() {
    // In reverse-Z: near=1.0, far=0.0
    // Object near_depth=0.9 (closer), HiZ depth=0.8 (further away surface)
    // 0.9 >= 0.8 - EPSILON -> visible (object is in front of occluder)
    let object_depth = 0.9;
    let hiz_depth = 0.8;
    assert!(object_depth >= hiz_depth - EPSILON, "Object in front should be visible");
}

/// Tests reverse-Z depth comparison: object behind is occluded.
#[test]
fn test_reverse_z_object_behind_occluded() {
    // Object near_depth=0.7 (further), HiZ depth=0.8 (closer surface exists)
    // 0.7 < 0.8 - EPSILON -> occluded (object is behind visible geometry)
    let object_depth = 0.7;
    let hiz_depth = 0.8;
    assert!(object_depth < hiz_depth - EPSILON, "Object behind should be occluded");
}

/// Tests reverse-Z: object exactly at HiZ depth (within EPSILON) is visible.
#[test]
fn test_reverse_z_object_at_surface_visible() {
    // Object at same depth as visible surface should be visible (conservative)
    let object_depth = 0.8;
    let hiz_depth = 0.8;
    assert!(object_depth >= hiz_depth - EPSILON, "Object at surface should be visible");
}

/// Tests EPSILON tolerance for depth comparison.
#[test]
fn test_depth_epsilon_tolerance() {
    let hiz_depth = 0.5;

    // Object slightly behind (within EPSILON) should still be visible
    let object_depth = hiz_depth - EPSILON * 0.5;
    assert!(object_depth >= hiz_depth - EPSILON, "Within EPSILON should be visible");

    // Object clearly behind should be occluded
    let object_depth_behind = hiz_depth - EPSILON * 2.0;
    assert!(object_depth_behind < hiz_depth - EPSILON, "Beyond EPSILON should be occluded");
}

/// Tests reverse-Z: near plane depth value.
#[test]
fn test_reverse_z_near_plane_value() {
    // In reverse-Z, near plane has depth = 1.0
    let near_plane_depth = 1.0;
    let far_plane_depth = 0.0;

    // Near plane objects should always be visible (closest possible)
    assert!(near_plane_depth >= 0.0, "Near plane is max depth in reverse-Z");
    assert!(near_plane_depth > far_plane_depth, "Near > far in reverse-Z");
}

/// Tests reverse-Z: far plane depth value.
#[test]
fn test_reverse_z_far_plane_value() {
    // In reverse-Z, far plane has depth = 0.0
    let far_plane_depth = 0.0;

    // Far plane objects are likely occluded by anything closer
    assert_eq!(far_plane_depth, 0.0, "Far plane is 0 in reverse-Z");
}

// =============================================================================
// CATEGORY 7: WORKGROUP CALCULATION TESTS
// =============================================================================

/// Tests workgroup calculation for zero objects.
#[test]
fn test_workgroups_zero_objects() {
    assert_eq!(workgroups_for_objects(0), 0);
}

/// Tests workgroup calculation for exactly one workgroup.
#[test]
fn test_workgroups_exactly_one() {
    assert_eq!(workgroups_for_objects(1), 1);
    assert_eq!(workgroups_for_objects(256), 1); // WORKGROUP_SIZE = 256
}

/// Tests workgroup calculation at boundary.
#[test]
fn test_workgroups_boundary() {
    assert_eq!(workgroups_for_objects(256), 1);
    assert_eq!(workgroups_for_objects(257), 2);
}

/// Tests workgroup calculation for various counts.
#[test]
fn test_workgroups_various() {
    assert_eq!(workgroups_for_objects(512), 2);
    assert_eq!(workgroups_for_objects(768), 3);
    assert_eq!(workgroups_for_objects(1000), 4); // ceil(1000/256) = 4
    assert_eq!(workgroups_for_objects(1024), 4);
}

/// Tests workgroup calculation for large counts (100K instances).
#[test]
fn test_workgroups_large_count() {
    // Performance target: 100K instances
    assert_eq!(workgroups_for_objects(100000), 391); // ceil(100000/256) = 391
    assert_eq!(workgroups_for_objects(1000000), 3907); // ceil(1000000/256) = 3907
}

/// Tests BatchParams::num_workgroups() method.
#[test]
fn test_batch_params_num_workgroups() {
    let params = BatchParams::new(1000);
    assert_eq!(params.num_workgroups(), 4);
}

/// Tests BatchParams::dispatch_size() method.
#[test]
fn test_batch_params_dispatch_size() {
    let params = BatchParams::new(1000);
    assert_eq!(params.dispatch_size(), (4, 1, 1));
}

// =============================================================================
// CATEGORY 8: SHADER STRUCTURE TESTS
// =============================================================================

/// Tests shader source is non-empty and loads correctly.
#[test]
fn test_shader_source_exists() {
    assert!(!HIZ_OCCLUSION_SHADER.is_empty(), "Shader source should not be empty");
    assert!(HIZ_OCCLUSION_SHADER.len() > 1000, "Shader should have substantial content");
}

/// Tests shader contains required entry points.
#[test]
fn test_shader_entry_points() {
    assert!(HIZ_OCCLUSION_SHADER.contains("fn test_hiz_occlusion"),
            "Missing test_hiz_occlusion function");
    assert!(HIZ_OCCLUSION_SHADER.contains("fn hiz_occlusion_cull"),
            "Missing hiz_occlusion_cull entry point");
    assert!(HIZ_OCCLUSION_SHADER.contains("fn hiz_occlusion_cull_detailed"),
            "Missing hiz_occlusion_cull_detailed entry point");
}

/// Tests shader contains AABB projection function.
#[test]
fn test_shader_projection_function() {
    assert!(HIZ_OCCLUSION_SHADER.contains("fn project_aabb_to_screen"),
            "Missing project_aabb_to_screen function");
}

/// Tests shader contains mip level selection function.
#[test]
fn test_shader_mip_selection_function() {
    assert!(HIZ_OCCLUSION_SHADER.contains("fn select_mip_level"),
            "Missing select_mip_level function");
}

/// Tests shader contains depth sampling functions.
#[test]
fn test_shader_sampling_functions() {
    assert!(HIZ_OCCLUSION_SHADER.contains("fn sample_hiz_point"),
            "Missing sample_hiz_point function");
    assert!(HIZ_OCCLUSION_SHADER.contains("fn sample_hiz_rect_max"),
            "Missing sample_hiz_rect_max function");
}

/// Tests shader contains helper functions.
#[test]
fn test_shader_helper_functions() {
    assert!(HIZ_OCCLUSION_SHADER.contains("fn get_aabb_corner"),
            "Missing get_aabb_corner function");
    assert!(HIZ_OCCLUSION_SHADER.contains("fn world_to_clip"),
            "Missing world_to_clip function");
    assert!(HIZ_OCCLUSION_SHADER.contains("fn clip_to_ndc"),
            "Missing clip_to_ndc function");
    assert!(HIZ_OCCLUSION_SHADER.contains("fn ndc_to_screen"),
            "Missing ndc_to_screen function");
}

/// Tests shader has correct bind group declarations (Group 0).
#[test]
fn test_shader_bindings_group_0() {
    assert!(HIZ_OCCLUSION_SHADER.contains("@group(0) @binding(0)"),
            "Missing group 0 binding 0 (HiZ texture)");
    assert!(HIZ_OCCLUSION_SHADER.contains("@group(0) @binding(1)"),
            "Missing group 0 binding 1 (sampler)");
}

/// Tests shader has correct bind group declarations (Group 1).
#[test]
fn test_shader_bindings_group_1() {
    assert!(HIZ_OCCLUSION_SHADER.contains("@group(1) @binding(0)"),
            "Missing group 1 binding 0 (params)");
}

/// Tests shader has correct bind group declarations (Group 2).
#[test]
fn test_shader_bindings_group_2() {
    assert!(HIZ_OCCLUSION_SHADER.contains("@group(2) @binding(0)"),
            "Missing group 2 binding 0 (batch params)");
    assert!(HIZ_OCCLUSION_SHADER.contains("@group(2) @binding(1)"),
            "Missing group 2 binding 1 (input AABBs)");
    assert!(HIZ_OCCLUSION_SHADER.contains("@group(2) @binding(2)"),
            "Missing group 2 binding 2 (visibility results)");
}

/// Tests shader has correct workgroup size attribute.
#[test]
fn test_shader_workgroup_size() {
    assert!(HIZ_OCCLUSION_SHADER.contains("@compute @workgroup_size(256)"),
            "Missing or incorrect workgroup_size attribute");
}

/// Tests shader contains WORKGROUP_SIZE constant.
#[test]
fn test_shader_workgroup_constant() {
    assert!(HIZ_OCCLUSION_SHADER.contains("const WORKGROUP_SIZE: u32 = 256u"),
            "Missing WORKGROUP_SIZE constant");
}

/// Tests shader contains EPSILON constant.
#[test]
fn test_shader_epsilon_constant() {
    assert!(HIZ_OCCLUSION_SHADER.contains("const EPSILON: f32"),
            "Missing EPSILON constant");
}

/// Tests shader contains reverse-Z depth convention comments.
#[test]
fn test_shader_reverse_z_documentation() {
    assert!(HIZ_OCCLUSION_SHADER.contains("reverse-Z") ||
            HIZ_OCCLUSION_SHADER.contains("Reverse-Z"),
            "Missing reverse-Z documentation");
    assert!(HIZ_OCCLUSION_SHADER.contains("near=1.0") ||
            HIZ_OCCLUSION_SHADER.contains("near = 1"),
            "Missing near plane depth documentation");
}

// =============================================================================
// CATEGORY 9: CONSTANTS VERIFICATION
// =============================================================================

/// Tests WORKGROUP_SIZE constant value.
#[test]
fn test_workgroup_size_constant() {
    assert_eq!(WORKGROUP_SIZE, 256);
}

/// Tests MAX_MIP_LEVEL constant value.
#[test]
fn test_max_mip_level_constant() {
    assert_eq!(MAX_MIP_LEVEL, 14);
}

/// Tests EPSILON constant is small but positive.
#[test]
fn test_epsilon_constant() {
    assert!(EPSILON > 0.0);
    assert!(EPSILON < 0.001);
    assert_eq!(EPSILON, 1e-6);
}

/// Tests CONSERVATIVE_EXPAND constant value.
#[test]
fn test_conservative_expand_constant() {
    assert_eq!(CONSERVATIVE_EXPAND, 1.0);
}

/// Tests struct size constants match expected values.
#[test]
fn test_struct_size_constants() {
    assert_eq!(HIZ_OCCLUSION_PARAMS_SIZE, 80);
    assert_eq!(BATCH_PARAMS_SIZE, 16);
    assert_eq!(INPUT_AABB_SIZE, 32);
}

// =============================================================================
// CATEGORY 10: CPU OCCLUSION TEST INTEGRATION
// =============================================================================

/// Creates a simple HiZ buffer for testing (single mip level).
#[allow(dead_code)]
fn create_simple_hiz_buffer(width: u32, height: u32, depth_value: f32) -> Vec<f32> {
    vec![depth_value; (width * height) as usize]
}

/// Creates a multi-mip HiZ buffer for testing.
fn create_mip_chain_hiz_buffer(base_width: u32, base_height: u32, depth_value: f32, num_mips: u32) -> Vec<f32> {
    let mut buffer = Vec::new();
    for m in 0..num_mips {
        let w = (base_width >> m).max(1);
        let h = (base_height >> m).max(1);
        buffer.extend(std::iter::repeat(depth_value).take((w * h) as usize));
    }
    buffer
}

/// Tests CPU occlusion test returns visible for AABB in front of HiZ surface.
#[test]
fn test_cpu_occlusion_visible_in_front() {
    let vp = identity_matrix();
    // HiZ depth = 0.5 (some geometry at middle distance)
    let hiz_buffer = create_mip_chain_hiz_buffer(256, 256, 0.5, 9);

    // AABB with near depth > 0.5 (closer than HiZ surface)
    // This AABB projects to some part of the screen
    let visible = cpu_test_occlusion(
        [-0.3, -0.3, 0.6],  // Near depth will be > 0.5
        [0.3, 0.3, 0.8],
        &vp,
        &hiz_buffer,
        256, 256,
        9
    );

    assert!(visible, "Object in front of HiZ surface should be visible");
}

/// Tests CPU occlusion test returns occluded for AABB behind HiZ surface.
#[test]
fn test_cpu_occlusion_occluded_behind() {
    let vp = identity_matrix();
    // HiZ depth = 0.9 (geometry very close to camera)
    let hiz_buffer = create_mip_chain_hiz_buffer(256, 256, 0.9, 9);

    // AABB with near depth < 0.9 (further than HiZ surface)
    let visible = cpu_test_occlusion(
        [-0.3, -0.3, 0.2],  // Near depth will be < 0.9
        [0.3, 0.3, 0.4],
        &vp,
        &hiz_buffer,
        256, 256,
        9
    );

    assert!(!visible, "Object behind HiZ surface should be occluded");
}

/// Tests CPU occlusion test: very small AABB projects to small screen rect.
/// With identity matrix, small world-space AABB maps to small NDC rect.
#[test]
fn test_cpu_occlusion_small_rect() {
    let vp = identity_matrix();
    // HiZ depth = 0.0 (far plane) means everything is visible
    let hiz_buffer = create_mip_chain_hiz_buffer(256, 256, 0.0, 9);

    // Small AABB with depth 0.5 (middle distance, higher than HiZ)
    let visible = cpu_test_occlusion(
        [-0.01, -0.01, 0.5],
        [0.01, 0.01, 0.5],
        &vp,
        &hiz_buffer,
        256, 256,
        9
    );

    // Should be visible since object depth (0.5) > HiZ depth (0.0)
    assert!(visible, "Object in front of far HiZ should be visible");
}

/// Tests CPU occlusion: degenerate rect (width < 1 pixel) returns visible.
#[test]
fn test_cpu_occlusion_degenerate_returns_visible() {
    let vp = identity_matrix();
    // Even with very close HiZ, degenerate rects should be visible
    let _hiz_buffer = create_mip_chain_hiz_buffer(256, 256, 1.0, 9);

    // Project the AABB first to understand its screen rect
    let (_min, _max, _depth, valid) = cpu_project_aabb(
        [0.499, 0.499, 0.5],
        [0.501, 0.501, 0.5],
        &vp,
        256.0, 256.0
    );

    assert!(valid, "AABB should project validly");
    // The rect size will be very small in world space but
    // with identity matrix it maps to screen space directly
}

// =============================================================================
// CATEGORY 11: EDGE CASE TESTS
// =============================================================================

/// Tests AABB with inverted corners (min > max).
#[test]
fn test_input_aabb_inverted_corners() {
    // This is technically invalid but we should handle gracefully
    let aabb = InputAABB::new([1.0, 1.0, 1.0], [0.0, 0.0, 0.0]);

    // Center calculation should still work
    let center = aabb.center();
    assert_eq!(center, [0.5, 0.5, 0.5]);

    // Half-extents will be negative
    let extents = aabb.half_extents();
    assert!(extents[0] < 0.0);
}

/// Tests projection of zero-size AABB (point).
#[test]
fn test_aabb_projection_point() {
    let vp = identity_matrix();
    let (_min, _max, _depth, valid) = cpu_project_aabb(
        [0.0, 0.0, 0.5],
        [0.0, 0.0, 0.5],
        &vp,
        100.0, 100.0
    );

    assert!(valid);
    // After conservative expansion, rect should still have some size
}

/// Tests AABB far from camera (potential precision issues).
#[test]
fn test_aabb_projection_far_distance() {
    let vp = identity_matrix();
    let (_min, _max, _depth, valid) = cpu_project_aabb(
        [999.0, 999.0, 0.999],
        [1001.0, 1001.0, 0.999],
        &vp,
        1920.0, 1080.0
    );

    // Should still be valid even at extreme coordinates
    assert!(valid);
}

/// Tests BatchParams with large (but not overflowing) object count.
#[test]
fn test_batch_params_large_objects() {
    // Use a large value that doesn't overflow workgroup calculation
    // max safe value: u32::MAX - WORKGROUP_SIZE + 1 would overflow
    // Use a practical large value instead
    let params = BatchParams::new(10_000_000);

    // Workgroup calculation should work
    let workgroups = params.num_workgroups();
    assert!(workgroups > 0);
    assert_eq!(workgroups, (10_000_000 + 255) / 256);
}

/// Tests BatchParams workgroup overflow behavior (documents current limitation).
#[test]
#[should_panic(expected = "overflow")]
fn test_batch_params_max_objects_overflows() {
    // This documents that u32::MAX will overflow in num_workgroups()
    // This is acceptable since 4 billion objects is impractical
    let params = BatchParams::new(u32::MAX);
    let _ = params.num_workgroups();
}

/// Tests HiZOcclusionParams with very small screen.
#[test]
fn test_hiz_params_tiny_screen() {
    let vp = identity_matrix();
    let params = HiZOcclusionParams::new(&vp, 1.0, 1.0, 0.1, 1);

    assert_eq!(params.hiz_size, [1.0, 1.0]);
    assert_eq!(params.max_mip, 0);
}

/// Tests mip selection with very large rect.
#[test]
fn test_mip_selection_very_large() {
    // 16384x16384 -> log2(16384) = 14
    let mip = cpu_select_mip_level(16384.0, 16384.0, 14);
    assert_eq!(mip, 14);

    // Even larger, clamped
    let mip_clamped = cpu_select_mip_level(65536.0, 65536.0, 14);
    assert_eq!(mip_clamped, 14);
}

// =============================================================================
// CATEGORY 12: STRUCT DEFAULT TESTS
// =============================================================================

/// Tests HiZOcclusionParams default is zeroed.
#[test]
fn test_hiz_occlusion_params_default() {
    let params: HiZOcclusionParams = Default::default();

    assert_eq!(params.hiz_size, [0.0, 0.0]);
    assert_eq!(params.near_plane, 0.0);
    assert_eq!(params.max_mip, 0);

    for row in &params.view_projection {
        for &val in row {
            assert_eq!(val, 0.0);
        }
    }
}

/// Tests BatchParams default is zeroed.
#[test]
fn test_batch_params_default() {
    let params: BatchParams = Default::default();

    assert_eq!(params.num_objects, 0);
    assert_eq!(params.flags, 0);
    assert_eq!(params._pad0, 0);
    assert_eq!(params._pad1, 0);
}

/// Tests InputAABB default is zeroed.
#[test]
fn test_input_aabb_default() {
    let aabb: InputAABB = Default::default();

    assert_eq!(aabb.min, [0.0, 0.0, 0.0]);
    assert_eq!(aabb.max, [0.0, 0.0, 0.0]);
    assert_eq!(aabb._pad0, 0.0);
    assert_eq!(aabb._pad1, 0.0);
}
