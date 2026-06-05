// SPDX-License-Identifier: MIT
//
// blackbox_vertex_attribute.rs -- Blackbox contract tests for T-WGPU-P3.2.2 (Vertex Attribute Formats).
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::render_pipeline::*` -- no internal fields,
// no private methods, no implementation details.
//
// Public API under test:
//   vertex_format_size(format)         -- Returns size in bytes for a VertexFormat
//   vertex_format_components(format)   -- Returns component count (1-4)
//   vertex_format_is_normalized(format)-- Returns true if format maps to [0,1] or [-1,1]
//   vertex_format_is_float(format)     -- Returns true for float formats
//   vertex_format_is_signed_int(format)-- Returns true for signed integer formats
//   vertex_format_is_unsigned_int(format)-- Returns true for unsigned integer formats
//   VertexFormatInfo                   -- Struct with size, components, normalized, is_float
//   calculate_stride(formats)          -- Calculate total stride for format list
//   calculate_offsets(formats)         -- Calculate offsets for each format
//   common::*                          -- Common format presets (POSITION, NORMAL, UV, etc.)
//   strides::*                         -- Pre-calculated strides (PBR, SKINNED, TERRAIN, etc.)
//   vertex_attr_array!                 -- Re-exported wgpu macro for attribute arrays
//
// Acceptance criteria (T-WGPU-P3.2.2):
//   1.  All 33 VertexFormat variants have correct sizes
//   2.  All formats have valid component counts (1-4)
//   3.  Common format presets map to correct VertexFormat values
//   4.  Pre-calculated strides match manual calculations
//   5.  calculate_stride produces correct results for various layouts
//   6.  calculate_offsets produces cumulative offsets
//   7.  VertexFormatInfo correctly captures all format properties
//   8.  vertex_attr_array! macro produces valid attribute arrays
//   9.  Type classification functions (is_float, is_normalized, etc.) are correct
//  10.  Edge cases: empty arrays, single attribute, large layouts

use renderer_backend::render_pipeline::{
    calculate_offsets, calculate_stride, vertex_attr_array, vertex_format_components,
    vertex_format_is_float, vertex_format_is_normalized, vertex_format_is_signed_int,
    vertex_format_is_unsigned_int, vertex_format_size, vertex_formats as common,
    vertex_strides as strides, VertexFormatInfo,
};
use wgpu::VertexFormat;

// =============================================================================
// SECTION 1: SIZE TESTS - 8-BIT FORMATS
// =============================================================================

#[test]
fn test_uint8x2_size() {
    assert_eq!(vertex_format_size(VertexFormat::Uint8x2), 2);
}

#[test]
fn test_uint8x4_size() {
    assert_eq!(vertex_format_size(VertexFormat::Uint8x4), 4);
}

#[test]
fn test_sint8x2_size() {
    assert_eq!(vertex_format_size(VertexFormat::Sint8x2), 2);
}

#[test]
fn test_sint8x4_size() {
    assert_eq!(vertex_format_size(VertexFormat::Sint8x4), 4);
}

#[test]
fn test_unorm8x2_size() {
    assert_eq!(vertex_format_size(VertexFormat::Unorm8x2), 2);
}

#[test]
fn test_unorm8x4_size() {
    assert_eq!(vertex_format_size(VertexFormat::Unorm8x4), 4);
}

#[test]
fn test_snorm8x2_size() {
    assert_eq!(vertex_format_size(VertexFormat::Snorm8x2), 2);
}

#[test]
fn test_snorm8x4_size() {
    assert_eq!(vertex_format_size(VertexFormat::Snorm8x4), 4);
}

// =============================================================================
// SECTION 2: SIZE TESTS - 16-BIT FORMATS
// =============================================================================

#[test]
fn test_uint16x2_size() {
    assert_eq!(vertex_format_size(VertexFormat::Uint16x2), 4);
}

#[test]
fn test_uint16x4_size() {
    assert_eq!(vertex_format_size(VertexFormat::Uint16x4), 8);
}

#[test]
fn test_sint16x2_size() {
    assert_eq!(vertex_format_size(VertexFormat::Sint16x2), 4);
}

#[test]
fn test_sint16x4_size() {
    assert_eq!(vertex_format_size(VertexFormat::Sint16x4), 8);
}

#[test]
fn test_unorm16x2_size() {
    assert_eq!(vertex_format_size(VertexFormat::Unorm16x2), 4);
}

#[test]
fn test_unorm16x4_size() {
    assert_eq!(vertex_format_size(VertexFormat::Unorm16x4), 8);
}

#[test]
fn test_snorm16x2_size() {
    assert_eq!(vertex_format_size(VertexFormat::Snorm16x2), 4);
}

#[test]
fn test_snorm16x4_size() {
    assert_eq!(vertex_format_size(VertexFormat::Snorm16x4), 8);
}

#[test]
fn test_float16x2_size() {
    assert_eq!(vertex_format_size(VertexFormat::Float16x2), 4);
}

#[test]
fn test_float16x4_size() {
    assert_eq!(vertex_format_size(VertexFormat::Float16x4), 8);
}

// =============================================================================
// SECTION 3: SIZE TESTS - 32-BIT FORMATS
// =============================================================================

#[test]
fn test_uint32_size() {
    assert_eq!(vertex_format_size(VertexFormat::Uint32), 4);
}

#[test]
fn test_uint32x2_size() {
    assert_eq!(vertex_format_size(VertexFormat::Uint32x2), 8);
}

#[test]
fn test_uint32x3_size() {
    assert_eq!(vertex_format_size(VertexFormat::Uint32x3), 12);
}

#[test]
fn test_uint32x4_size() {
    assert_eq!(vertex_format_size(VertexFormat::Uint32x4), 16);
}

#[test]
fn test_sint32_size() {
    assert_eq!(vertex_format_size(VertexFormat::Sint32), 4);
}

#[test]
fn test_sint32x2_size() {
    assert_eq!(vertex_format_size(VertexFormat::Sint32x2), 8);
}

#[test]
fn test_sint32x3_size() {
    assert_eq!(vertex_format_size(VertexFormat::Sint32x3), 12);
}

#[test]
fn test_sint32x4_size() {
    assert_eq!(vertex_format_size(VertexFormat::Sint32x4), 16);
}

#[test]
fn test_float32_size() {
    assert_eq!(vertex_format_size(VertexFormat::Float32), 4);
}

#[test]
fn test_float32x2_size() {
    assert_eq!(vertex_format_size(VertexFormat::Float32x2), 8);
}

#[test]
fn test_float32x3_size() {
    assert_eq!(vertex_format_size(VertexFormat::Float32x3), 12);
}

#[test]
fn test_float32x4_size() {
    assert_eq!(vertex_format_size(VertexFormat::Float32x4), 16);
}

// =============================================================================
// SECTION 4: SIZE TESTS - 64-BIT FORMATS
// =============================================================================

#[test]
fn test_float64_size() {
    assert_eq!(vertex_format_size(VertexFormat::Float64), 8);
}

#[test]
fn test_float64x2_size() {
    assert_eq!(vertex_format_size(VertexFormat::Float64x2), 16);
}

#[test]
fn test_float64x3_size() {
    assert_eq!(vertex_format_size(VertexFormat::Float64x3), 24);
}

#[test]
fn test_float64x4_size() {
    assert_eq!(vertex_format_size(VertexFormat::Float64x4), 32);
}

// =============================================================================
// SECTION 5: SIZE TESTS - PACKED FORMATS
// =============================================================================

#[test]
fn test_unorm10_10_10_2_size() {
    // Packed 10-10-10-2 format is 4 bytes total
    assert_eq!(vertex_format_size(VertexFormat::Unorm10_10_10_2), 4);
}

// =============================================================================
// SECTION 6: COMPONENT COUNT TESTS
// =============================================================================

#[test]
fn test_single_component_formats() {
    assert_eq!(vertex_format_components(VertexFormat::Uint32), 1);
    assert_eq!(vertex_format_components(VertexFormat::Sint32), 1);
    assert_eq!(vertex_format_components(VertexFormat::Float32), 1);
    assert_eq!(vertex_format_components(VertexFormat::Float64), 1);
}

#[test]
fn test_two_component_8bit_formats() {
    assert_eq!(vertex_format_components(VertexFormat::Uint8x2), 2);
    assert_eq!(vertex_format_components(VertexFormat::Sint8x2), 2);
    assert_eq!(vertex_format_components(VertexFormat::Unorm8x2), 2);
    assert_eq!(vertex_format_components(VertexFormat::Snorm8x2), 2);
}

#[test]
fn test_two_component_16bit_formats() {
    assert_eq!(vertex_format_components(VertexFormat::Uint16x2), 2);
    assert_eq!(vertex_format_components(VertexFormat::Sint16x2), 2);
    assert_eq!(vertex_format_components(VertexFormat::Unorm16x2), 2);
    assert_eq!(vertex_format_components(VertexFormat::Snorm16x2), 2);
    assert_eq!(vertex_format_components(VertexFormat::Float16x2), 2);
}

#[test]
fn test_two_component_32bit_formats() {
    assert_eq!(vertex_format_components(VertexFormat::Uint32x2), 2);
    assert_eq!(vertex_format_components(VertexFormat::Sint32x2), 2);
    assert_eq!(vertex_format_components(VertexFormat::Float32x2), 2);
    assert_eq!(vertex_format_components(VertexFormat::Float64x2), 2);
}

#[test]
fn test_three_component_formats() {
    assert_eq!(vertex_format_components(VertexFormat::Uint32x3), 3);
    assert_eq!(vertex_format_components(VertexFormat::Sint32x3), 3);
    assert_eq!(vertex_format_components(VertexFormat::Float32x3), 3);
    assert_eq!(vertex_format_components(VertexFormat::Float64x3), 3);
}

#[test]
fn test_four_component_8bit_formats() {
    assert_eq!(vertex_format_components(VertexFormat::Uint8x4), 4);
    assert_eq!(vertex_format_components(VertexFormat::Sint8x4), 4);
    assert_eq!(vertex_format_components(VertexFormat::Unorm8x4), 4);
    assert_eq!(vertex_format_components(VertexFormat::Snorm8x4), 4);
}

#[test]
fn test_four_component_16bit_formats() {
    assert_eq!(vertex_format_components(VertexFormat::Uint16x4), 4);
    assert_eq!(vertex_format_components(VertexFormat::Sint16x4), 4);
    assert_eq!(vertex_format_components(VertexFormat::Unorm16x4), 4);
    assert_eq!(vertex_format_components(VertexFormat::Snorm16x4), 4);
    assert_eq!(vertex_format_components(VertexFormat::Float16x4), 4);
}

#[test]
fn test_four_component_32bit_formats() {
    assert_eq!(vertex_format_components(VertexFormat::Uint32x4), 4);
    assert_eq!(vertex_format_components(VertexFormat::Sint32x4), 4);
    assert_eq!(vertex_format_components(VertexFormat::Float32x4), 4);
    assert_eq!(vertex_format_components(VertexFormat::Float64x4), 4);
}

#[test]
fn test_packed_format_components() {
    // Unorm10_10_10_2 has 4 components (R10 G10 B10 A2)
    assert_eq!(vertex_format_components(VertexFormat::Unorm10_10_10_2), 4);
}

// =============================================================================
// SECTION 7: TYPE CLASSIFICATION TESTS
// =============================================================================

#[test]
fn test_normalized_formats_are_normalized() {
    assert!(vertex_format_is_normalized(VertexFormat::Unorm8x2));
    assert!(vertex_format_is_normalized(VertexFormat::Unorm8x4));
    assert!(vertex_format_is_normalized(VertexFormat::Snorm8x2));
    assert!(vertex_format_is_normalized(VertexFormat::Snorm8x4));
    assert!(vertex_format_is_normalized(VertexFormat::Unorm16x2));
    assert!(vertex_format_is_normalized(VertexFormat::Unorm16x4));
    assert!(vertex_format_is_normalized(VertexFormat::Snorm16x2));
    assert!(vertex_format_is_normalized(VertexFormat::Snorm16x4));
    assert!(vertex_format_is_normalized(VertexFormat::Unorm10_10_10_2));
}

#[test]
fn test_non_normalized_formats_are_not_normalized() {
    assert!(!vertex_format_is_normalized(VertexFormat::Uint8x2));
    assert!(!vertex_format_is_normalized(VertexFormat::Sint8x2));
    assert!(!vertex_format_is_normalized(VertexFormat::Float32x3));
    assert!(!vertex_format_is_normalized(VertexFormat::Float16x4));
    assert!(!vertex_format_is_normalized(VertexFormat::Uint32));
}

#[test]
fn test_float_formats_are_float() {
    assert!(vertex_format_is_float(VertexFormat::Float16x2));
    assert!(vertex_format_is_float(VertexFormat::Float16x4));
    assert!(vertex_format_is_float(VertexFormat::Float32));
    assert!(vertex_format_is_float(VertexFormat::Float32x2));
    assert!(vertex_format_is_float(VertexFormat::Float32x3));
    assert!(vertex_format_is_float(VertexFormat::Float32x4));
    assert!(vertex_format_is_float(VertexFormat::Float64));
    assert!(vertex_format_is_float(VertexFormat::Float64x2));
    assert!(vertex_format_is_float(VertexFormat::Float64x3));
    assert!(vertex_format_is_float(VertexFormat::Float64x4));
}

#[test]
fn test_non_float_formats_are_not_float() {
    assert!(!vertex_format_is_float(VertexFormat::Uint8x4));
    assert!(!vertex_format_is_float(VertexFormat::Sint16x2));
    assert!(!vertex_format_is_float(VertexFormat::Unorm8x4));
    assert!(!vertex_format_is_float(VertexFormat::Snorm16x4));
    assert!(!vertex_format_is_float(VertexFormat::Uint32x4));
}

#[test]
fn test_signed_int_formats() {
    assert!(vertex_format_is_signed_int(VertexFormat::Sint8x2));
    assert!(vertex_format_is_signed_int(VertexFormat::Sint8x4));
    assert!(vertex_format_is_signed_int(VertexFormat::Sint16x2));
    assert!(vertex_format_is_signed_int(VertexFormat::Sint16x4));
    assert!(vertex_format_is_signed_int(VertexFormat::Sint32));
    assert!(vertex_format_is_signed_int(VertexFormat::Sint32x2));
    assert!(vertex_format_is_signed_int(VertexFormat::Sint32x3));
    assert!(vertex_format_is_signed_int(VertexFormat::Sint32x4));
}

#[test]
fn test_non_signed_int_formats() {
    assert!(!vertex_format_is_signed_int(VertexFormat::Uint8x2));
    assert!(!vertex_format_is_signed_int(VertexFormat::Float32x3));
    assert!(!vertex_format_is_signed_int(VertexFormat::Unorm8x4));
    assert!(!vertex_format_is_signed_int(VertexFormat::Snorm8x4));
}

#[test]
fn test_unsigned_int_formats() {
    assert!(vertex_format_is_unsigned_int(VertexFormat::Uint8x2));
    assert!(vertex_format_is_unsigned_int(VertexFormat::Uint8x4));
    assert!(vertex_format_is_unsigned_int(VertexFormat::Uint16x2));
    assert!(vertex_format_is_unsigned_int(VertexFormat::Uint16x4));
    assert!(vertex_format_is_unsigned_int(VertexFormat::Uint32));
    assert!(vertex_format_is_unsigned_int(VertexFormat::Uint32x2));
    assert!(vertex_format_is_unsigned_int(VertexFormat::Uint32x3));
    assert!(vertex_format_is_unsigned_int(VertexFormat::Uint32x4));
}

#[test]
fn test_non_unsigned_int_formats() {
    assert!(!vertex_format_is_unsigned_int(VertexFormat::Sint8x2));
    assert!(!vertex_format_is_unsigned_int(VertexFormat::Float32x3));
    assert!(!vertex_format_is_unsigned_int(VertexFormat::Unorm8x4));
}

// =============================================================================
// SECTION 8: COMMON FORMAT PRESETS TESTS
// =============================================================================

#[test]
fn test_common_position_format() {
    assert_eq!(common::POSITION, VertexFormat::Float32x3);
    assert_eq!(vertex_format_size(common::POSITION), 12);
}

#[test]
fn test_common_position_2d_format() {
    assert_eq!(common::POSITION_2D, VertexFormat::Float32x2);
    assert_eq!(vertex_format_size(common::POSITION_2D), 8);
}

#[test]
fn test_common_normal_format() {
    assert_eq!(common::NORMAL, VertexFormat::Float32x3);
    assert_eq!(vertex_format_size(common::NORMAL), 12);
}

#[test]
fn test_common_normal_compressed_format() {
    assert_eq!(common::NORMAL_COMPRESSED, VertexFormat::Snorm8x4);
    assert_eq!(vertex_format_size(common::NORMAL_COMPRESSED), 4);
}

#[test]
fn test_common_tangent_format() {
    assert_eq!(common::TANGENT, VertexFormat::Float32x4);
    assert_eq!(vertex_format_size(common::TANGENT), 16);
}

#[test]
fn test_common_tangent_compressed_format() {
    assert_eq!(common::TANGENT_COMPRESSED, VertexFormat::Snorm8x4);
    assert_eq!(vertex_format_size(common::TANGENT_COMPRESSED), 4);
}

#[test]
fn test_common_normal_packed_1010102_format() {
    assert_eq!(common::NORMAL_PACKED_1010102, VertexFormat::Unorm10_10_10_2);
    assert_eq!(vertex_format_size(common::NORMAL_PACKED_1010102), 4);
}

#[test]
fn test_common_uv_format() {
    assert_eq!(common::UV, VertexFormat::Float32x2);
    assert_eq!(vertex_format_size(common::UV), 8);
}

#[test]
fn test_common_uv_half_format() {
    assert_eq!(common::UV_HALF, VertexFormat::Float16x2);
    assert_eq!(vertex_format_size(common::UV_HALF), 4);
}

#[test]
fn test_common_uv_normalized_format() {
    assert_eq!(common::UV_NORMALIZED, VertexFormat::Unorm16x2);
    assert_eq!(vertex_format_size(common::UV_NORMALIZED), 4);
}

#[test]
fn test_common_color_format() {
    assert_eq!(common::COLOR, VertexFormat::Unorm8x4);
    assert_eq!(vertex_format_size(common::COLOR), 4);
    assert!(vertex_format_is_normalized(common::COLOR));
}

#[test]
fn test_common_color_hdr_format() {
    assert_eq!(common::COLOR_HDR, VertexFormat::Float32x4);
    assert_eq!(vertex_format_size(common::COLOR_HDR), 16);
    assert!(vertex_format_is_float(common::COLOR_HDR));
}

#[test]
fn test_common_color_hdr_half_format() {
    assert_eq!(common::COLOR_HDR_HALF, VertexFormat::Float16x4);
    assert_eq!(vertex_format_size(common::COLOR_HDR_HALF), 8);
}

#[test]
fn test_common_bone_indices_format() {
    assert_eq!(common::BONE_INDICES, VertexFormat::Uint8x4);
    assert_eq!(vertex_format_size(common::BONE_INDICES), 4);
}

#[test]
fn test_common_bone_indices_large_format() {
    assert_eq!(common::BONE_INDICES_LARGE, VertexFormat::Uint16x4);
    assert_eq!(vertex_format_size(common::BONE_INDICES_LARGE), 8);
}

#[test]
fn test_common_bone_weights_format() {
    assert_eq!(common::BONE_WEIGHTS, VertexFormat::Unorm8x4);
    assert_eq!(vertex_format_size(common::BONE_WEIGHTS), 4);
}

#[test]
fn test_common_bone_weights_float_format() {
    assert_eq!(common::BONE_WEIGHTS_FLOAT, VertexFormat::Float32x4);
    assert_eq!(vertex_format_size(common::BONE_WEIGHTS_FLOAT), 16);
}

#[test]
fn test_common_instance_matrix_row_format() {
    assert_eq!(common::INSTANCE_MATRIX_ROW, VertexFormat::Float32x4);
    assert_eq!(vertex_format_size(common::INSTANCE_MATRIX_ROW), 16);
}

#[test]
fn test_common_instance_id_format() {
    assert_eq!(common::INSTANCE_ID, VertexFormat::Uint32);
    assert_eq!(vertex_format_size(common::INSTANCE_ID), 4);
}

#[test]
fn test_common_particle_size_format() {
    assert_eq!(common::PARTICLE_SIZE, VertexFormat::Float32);
    assert_eq!(vertex_format_size(common::PARTICLE_SIZE), 4);
}

#[test]
fn test_common_particle_size_2d_format() {
    assert_eq!(common::PARTICLE_SIZE_2D, VertexFormat::Float32x2);
    assert_eq!(vertex_format_size(common::PARTICLE_SIZE_2D), 8);
}

#[test]
fn test_common_particle_rotation_format() {
    assert_eq!(common::PARTICLE_ROTATION, VertexFormat::Float32);
    assert_eq!(vertex_format_size(common::PARTICLE_ROTATION), 4);
}

#[test]
fn test_common_particle_life_format() {
    assert_eq!(common::PARTICLE_LIFE, VertexFormat::Float32);
    assert_eq!(vertex_format_size(common::PARTICLE_LIFE), 4);
}

#[test]
fn test_common_particle_velocity_format() {
    assert_eq!(common::PARTICLE_VELOCITY, VertexFormat::Float32x3);
    assert_eq!(vertex_format_size(common::PARTICLE_VELOCITY), 12);
}

// =============================================================================
// SECTION 9: STANDARD STRIDE TESTS
// =============================================================================

#[test]
fn test_stride_pbr() {
    // PBR: position(12) + normal(12) + uv(8) + tangent(16) = 48
    assert_eq!(strides::PBR, 48);
}

#[test]
fn test_stride_skinned() {
    // Skinned: PBR(48) + bone_indices(8) + bone_weights(16) = 72
    assert_eq!(strides::SKINNED, 72);
}

#[test]
fn test_stride_terrain() {
    // Terrain: position(12) + normal(12) + uv(8) = 32
    assert_eq!(strides::TERRAIN, 32);
}

#[test]
fn test_stride_particle() {
    // Particle: position(12) + color(16) + size_rotation(8) = 36
    assert_eq!(strides::PARTICLE, 36);
}

#[test]
fn test_stride_ui() {
    // UI: position_2d(8) + uv(8) + color(4) = 20
    assert_eq!(strides::UI, 20);
}

#[test]
fn test_stride_position_only() {
    // Position only: position(12) = 12
    assert_eq!(strides::POSITION_ONLY, 12);
}

#[test]
fn test_stride_shadow() {
    // Shadow vertex (same as position only): 12 bytes
    assert_eq!(strides::SHADOW, 12);
}

// =============================================================================
// SECTION 10: CALCULATE_STRIDE TESTS
// =============================================================================

#[test]
fn test_calculate_stride_pbr_layout() {
    let stride = calculate_stride(&[
        VertexFormat::Float32x3, // position: 12
        VertexFormat::Float32x3, // normal: 12
        VertexFormat::Float32x2, // uv: 8
        VertexFormat::Float32x4, // tangent: 16
    ]);
    assert_eq!(stride, 48);
    assert_eq!(stride, strides::PBR);
}

#[test]
fn test_calculate_stride_terrain_layout() {
    let stride = calculate_stride(&[
        VertexFormat::Float32x3, // position: 12
        VertexFormat::Float32x3, // normal: 12
        VertexFormat::Float32x2, // uv: 8
    ]);
    assert_eq!(stride, 32);
    assert_eq!(stride, strides::TERRAIN);
}

#[test]
fn test_calculate_stride_ui_layout() {
    let stride = calculate_stride(&[
        VertexFormat::Float32x2, // position_2d: 8
        VertexFormat::Float32x2, // uv: 8
        VertexFormat::Unorm8x4,  // color: 4
    ]);
    assert_eq!(stride, 20);
    assert_eq!(stride, strides::UI);
}

#[test]
fn test_calculate_stride_empty_layout() {
    let stride = calculate_stride(&[]);
    assert_eq!(stride, 0);
}

#[test]
fn test_calculate_stride_single_attribute() {
    let stride = calculate_stride(&[VertexFormat::Float32x3]);
    assert_eq!(stride, 12);
}

#[test]
fn test_calculate_stride_skinned_layout() {
    let stride = calculate_stride(&[
        VertexFormat::Float32x3, // position: 12
        VertexFormat::Float32x3, // normal: 12
        VertexFormat::Float32x2, // uv: 8
        VertexFormat::Float32x4, // tangent: 16
        VertexFormat::Uint16x4,  // bone_indices: 8
        VertexFormat::Float32x4, // bone_weights: 16
    ]);
    assert_eq!(stride, 72);
    assert_eq!(stride, strides::SKINNED);
}

#[test]
fn test_calculate_stride_mixed_sizes() {
    // Mix of different sized formats
    let stride = calculate_stride(&[
        VertexFormat::Uint8x4,   // 4 bytes
        VertexFormat::Float16x2, // 4 bytes
        VertexFormat::Float32x3, // 12 bytes
        VertexFormat::Float64x2, // 16 bytes
    ]);
    assert_eq!(stride, 4 + 4 + 12 + 16);
    assert_eq!(stride, 36);
}

#[test]
fn test_calculate_stride_all_8bit() {
    let stride = calculate_stride(&[
        VertexFormat::Uint8x2,
        VertexFormat::Uint8x4,
        VertexFormat::Sint8x2,
        VertexFormat::Sint8x4,
    ]);
    assert_eq!(stride, 2 + 4 + 2 + 4);
    assert_eq!(stride, 12);
}

#[test]
fn test_calculate_stride_instance_matrix() {
    // 4x4 matrix = 4 rows of Float32x4
    let stride = calculate_stride(&[
        VertexFormat::Float32x4,
        VertexFormat::Float32x4,
        VertexFormat::Float32x4,
        VertexFormat::Float32x4,
    ]);
    assert_eq!(stride, 64);
}

// =============================================================================
// SECTION 11: CALCULATE_OFFSETS TESTS
// =============================================================================

#[test]
fn test_calculate_offsets_pbr() {
    let formats = [
        VertexFormat::Float32x3, // position: offset 0
        VertexFormat::Float32x3, // normal: offset 12
        VertexFormat::Float32x2, // uv: offset 24
        VertexFormat::Float32x4, // tangent: offset 32
    ];
    let offsets = calculate_offsets(&formats);
    assert_eq!(offsets, [0, 12, 24, 32]);
}

#[test]
fn test_calculate_offsets_terrain() {
    let formats = [
        VertexFormat::Float32x3, // position: offset 0
        VertexFormat::Float32x3, // normal: offset 12
        VertexFormat::Float32x2, // uv: offset 24
    ];
    let offsets = calculate_offsets(&formats);
    assert_eq!(offsets, [0, 12, 24]);
}

#[test]
fn test_calculate_offsets_ui() {
    let formats = [
        VertexFormat::Float32x2, // position_2d: offset 0
        VertexFormat::Float32x2, // uv: offset 8
        VertexFormat::Unorm8x4,  // color: offset 16
    ];
    let offsets = calculate_offsets(&formats);
    assert_eq!(offsets, [0, 8, 16]);
}

#[test]
fn test_calculate_offsets_single() {
    let formats = [VertexFormat::Float32x3];
    let offsets = calculate_offsets(&formats);
    assert_eq!(offsets, [0]);
}

#[test]
fn test_calculate_offsets_mixed_sizes() {
    let formats = [
        VertexFormat::Uint8x4,   // offset 0, size 4
        VertexFormat::Float16x2, // offset 4, size 4
        VertexFormat::Float32x3, // offset 8, size 12
        VertexFormat::Float64x2, // offset 20, size 16
    ];
    let offsets = calculate_offsets(&formats);
    assert_eq!(offsets, [0, 4, 8, 20]);
}

#[test]
fn test_calculate_offsets_skinned() {
    let formats = [
        VertexFormat::Float32x3, // position: 0
        VertexFormat::Float32x3, // normal: 12
        VertexFormat::Float32x2, // uv: 24
        VertexFormat::Float32x4, // tangent: 32
        VertexFormat::Uint16x4,  // bone_indices: 48
        VertexFormat::Float32x4, // bone_weights: 56
    ];
    let offsets = calculate_offsets(&formats);
    assert_eq!(offsets, [0, 12, 24, 32, 48, 56]);
}

// =============================================================================
// SECTION 12: VERTEXFORMATINFO TESTS
// =============================================================================

#[test]
fn test_vertex_format_info_float32x3() {
    let info = VertexFormatInfo::new(VertexFormat::Float32x3);
    assert_eq!(info.format, VertexFormat::Float32x3);
    assert_eq!(info.size, 12);
    assert_eq!(info.components, 3);
    assert!(!info.normalized);
    assert!(info.is_float);
    assert_eq!(info.bytes_per_component, 4);
}

#[test]
fn test_vertex_format_info_unorm8x4() {
    let info = VertexFormatInfo::new(VertexFormat::Unorm8x4);
    assert_eq!(info.format, VertexFormat::Unorm8x4);
    assert_eq!(info.size, 4);
    assert_eq!(info.components, 4);
    assert!(info.normalized);
    assert!(!info.is_float);
    assert_eq!(info.bytes_per_component, 1);
}

#[test]
fn test_vertex_format_info_float16x4() {
    let info = VertexFormatInfo::new(VertexFormat::Float16x4);
    assert_eq!(info.format, VertexFormat::Float16x4);
    assert_eq!(info.size, 8);
    assert_eq!(info.components, 4);
    assert!(!info.normalized);
    assert!(info.is_float);
    assert_eq!(info.bytes_per_component, 2);
}

#[test]
fn test_vertex_format_info_float64x3() {
    let info = VertexFormatInfo::new(VertexFormat::Float64x3);
    assert_eq!(info.format, VertexFormat::Float64x3);
    assert_eq!(info.size, 24);
    assert_eq!(info.components, 3);
    assert!(!info.normalized);
    assert!(info.is_float);
    assert_eq!(info.bytes_per_component, 8);
}

#[test]
fn test_vertex_format_info_uint32() {
    let info = VertexFormatInfo::new(VertexFormat::Uint32);
    assert_eq!(info.format, VertexFormat::Uint32);
    assert_eq!(info.size, 4);
    assert_eq!(info.components, 1);
    assert!(!info.normalized);
    assert!(!info.is_float);
    assert_eq!(info.bytes_per_component, 4);
}

#[test]
fn test_vertex_format_info_snorm16x4() {
    let info = VertexFormatInfo::new(VertexFormat::Snorm16x4);
    assert_eq!(info.format, VertexFormat::Snorm16x4);
    assert_eq!(info.size, 8);
    assert_eq!(info.components, 4);
    assert!(info.normalized);
    assert!(!info.is_float);
    assert_eq!(info.bytes_per_component, 2);
}

#[test]
fn test_vertex_format_info_from_trait() {
    let info: VertexFormatInfo = VertexFormat::Float32x4.into();
    assert_eq!(info.format, VertexFormat::Float32x4);
    assert_eq!(info.size, 16);
    assert_eq!(info.components, 4);
}

#[test]
fn test_vertex_format_info_packed() {
    let info = VertexFormatInfo::new(VertexFormat::Unorm10_10_10_2);
    assert_eq!(info.size, 4);
    assert_eq!(info.components, 4);
    assert!(info.normalized);
    // Bytes per component is 4/4 = 1 (though the actual bit packing is different)
    assert_eq!(info.bytes_per_component, 1);
}

// =============================================================================
// SECTION 13: VERTEX_ATTR_ARRAY MACRO TESTS
// =============================================================================

#[test]
fn test_vertex_attr_array_simple() {
    let attrs = vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2];
    assert_eq!(attrs.len(), 3);

    // Check shader locations
    assert_eq!(attrs[0].shader_location, 0);
    assert_eq!(attrs[1].shader_location, 1);
    assert_eq!(attrs[2].shader_location, 2);

    // Check formats
    assert_eq!(attrs[0].format, VertexFormat::Float32x3);
    assert_eq!(attrs[1].format, VertexFormat::Float32x3);
    assert_eq!(attrs[2].format, VertexFormat::Float32x2);

    // Check offsets (cumulative)
    assert_eq!(attrs[0].offset, 0);
    assert_eq!(attrs[1].offset, 12); // 0 + 12
    assert_eq!(attrs[2].offset, 24); // 12 + 12
}

#[test]
fn test_vertex_attr_array_pbr() {
    let attrs = vertex_attr_array![
        0 => Float32x3,  // position
        1 => Float32x3,  // normal
        2 => Float32x2,  // uv
        3 => Float32x4   // tangent
    ];
    assert_eq!(attrs.len(), 4);

    // Offsets: 0, 12, 24, 32
    assert_eq!(attrs[0].offset, 0);
    assert_eq!(attrs[1].offset, 12);
    assert_eq!(attrs[2].offset, 24);
    assert_eq!(attrs[3].offset, 32);
}

#[test]
fn test_vertex_attr_array_single() {
    let attrs = vertex_attr_array![0 => Float32x3];
    assert_eq!(attrs.len(), 1);
    assert_eq!(attrs[0].shader_location, 0);
    assert_eq!(attrs[0].format, VertexFormat::Float32x3);
    assert_eq!(attrs[0].offset, 0);
}

#[test]
fn test_vertex_attr_array_mixed_formats() {
    let attrs = vertex_attr_array![
        0 => Unorm8x4,   // 4 bytes
        1 => Float16x2,  // 4 bytes
        2 => Float32x3   // 12 bytes
    ];

    assert_eq!(attrs[0].offset, 0);
    assert_eq!(attrs[1].offset, 4);
    assert_eq!(attrs[2].offset, 8);
}

#[test]
fn test_vertex_attr_array_ui_layout() {
    let attrs = vertex_attr_array![
        0 => Float32x2,  // position_2d: 8 bytes
        1 => Float32x2,  // uv: 8 bytes
        2 => Unorm8x4    // color: 4 bytes
    ];

    assert_eq!(attrs.len(), 3);
    assert_eq!(attrs[0].offset, 0);
    assert_eq!(attrs[1].offset, 8);
    assert_eq!(attrs[2].offset, 16);
}

#[test]
fn test_vertex_attr_array_skinned_layout() {
    let attrs = vertex_attr_array![
        0 => Float32x3,  // position
        1 => Float32x3,  // normal
        2 => Float32x2,  // uv
        3 => Float32x4,  // tangent
        4 => Uint16x4,   // bone_indices
        5 => Float32x4   // bone_weights
    ];

    assert_eq!(attrs.len(), 6);
    assert_eq!(attrs[0].offset, 0);
    assert_eq!(attrs[1].offset, 12);
    assert_eq!(attrs[2].offset, 24);
    assert_eq!(attrs[3].offset, 32);
    assert_eq!(attrs[4].offset, 48);
    assert_eq!(attrs[5].offset, 56);
}

// =============================================================================
// SECTION 14: EDGE CASES AND SPECIAL SCENARIOS
// =============================================================================

#[test]
fn test_core_33_formats_have_nonzero_size() {
    // Core 33 formats as documented in wgpu 22.x (32 base + 1 packed: Unorm10_10_10_2)
    let core_formats = [
        // 8-bit formats (8 variants)
        VertexFormat::Uint8x2, VertexFormat::Uint8x4,
        VertexFormat::Sint8x2, VertexFormat::Sint8x4,
        VertexFormat::Unorm8x2, VertexFormat::Unorm8x4,
        VertexFormat::Snorm8x2, VertexFormat::Snorm8x4,
        // 16-bit formats (10 variants)
        VertexFormat::Uint16x2, VertexFormat::Uint16x4,
        VertexFormat::Sint16x2, VertexFormat::Sint16x4,
        VertexFormat::Unorm16x2, VertexFormat::Unorm16x4,
        VertexFormat::Snorm16x2, VertexFormat::Snorm16x4,
        VertexFormat::Float16x2, VertexFormat::Float16x4,
        // 32-bit formats (12 variants)
        VertexFormat::Uint32, VertexFormat::Uint32x2, VertexFormat::Uint32x3, VertexFormat::Uint32x4,
        VertexFormat::Sint32, VertexFormat::Sint32x2, VertexFormat::Sint32x3, VertexFormat::Sint32x4,
        VertexFormat::Float32, VertexFormat::Float32x2, VertexFormat::Float32x3, VertexFormat::Float32x4,
        // 64-bit formats (4 variants)
        VertexFormat::Float64, VertexFormat::Float64x2, VertexFormat::Float64x3, VertexFormat::Float64x4,
        // Packed formats (1 variant)
        VertexFormat::Unorm10_10_10_2,
    ];

    for format in core_formats {
        assert!(vertex_format_size(format) > 0, "Format {:?} should have non-zero size", format);
    }

    // Core formats count: 8 + 10 + 12 + 4 + 1 = 35
    // (wgpu may have added Unorm8x4Bgra and Snorm10_10_10_2 in newer versions)
    assert_eq!(core_formats.len(), 35, "Should test 35 core formats");
}

#[test]
fn test_core_formats_have_valid_component_count() {
    let core_formats = [
        VertexFormat::Uint8x2, VertexFormat::Uint8x4,
        VertexFormat::Sint8x2, VertexFormat::Sint8x4,
        VertexFormat::Unorm8x2, VertexFormat::Unorm8x4,
        VertexFormat::Snorm8x2, VertexFormat::Snorm8x4,
        VertexFormat::Uint16x2, VertexFormat::Uint16x4,
        VertexFormat::Sint16x2, VertexFormat::Sint16x4,
        VertexFormat::Unorm16x2, VertexFormat::Unorm16x4,
        VertexFormat::Snorm16x2, VertexFormat::Snorm16x4,
        VertexFormat::Float16x2, VertexFormat::Float16x4,
        VertexFormat::Uint32, VertexFormat::Uint32x2, VertexFormat::Uint32x3, VertexFormat::Uint32x4,
        VertexFormat::Sint32, VertexFormat::Sint32x2, VertexFormat::Sint32x3, VertexFormat::Sint32x4,
        VertexFormat::Float32, VertexFormat::Float32x2, VertexFormat::Float32x3, VertexFormat::Float32x4,
        VertexFormat::Float64, VertexFormat::Float64x2, VertexFormat::Float64x3, VertexFormat::Float64x4,
        VertexFormat::Unorm10_10_10_2,
    ];

    for format in core_formats {
        let components = vertex_format_components(format);
        assert!(
            (1..=4).contains(&components),
            "Format {:?} has invalid component count: {}",
            format,
            components
        );
    }
}

#[test]
fn test_size_consistency_with_components() {
    // For standard formats, size should be components * bytes_per_component
    let test_cases = [
        (VertexFormat::Float32x3, 3, 4),  // 3 * 4 = 12
        (VertexFormat::Float32x4, 4, 4),  // 4 * 4 = 16
        (VertexFormat::Float64x2, 2, 8),  // 2 * 8 = 16
        (VertexFormat::Uint8x4, 4, 1),    // 4 * 1 = 4
        (VertexFormat::Float16x4, 4, 2),  // 4 * 2 = 8
    ];

    for (format, expected_components, expected_bytes_per_component) in test_cases {
        let info = VertexFormatInfo::new(format);
        assert_eq!(info.components, expected_components);
        assert_eq!(info.bytes_per_component, expected_bytes_per_component);
        assert_eq!(info.size, (expected_components * expected_bytes_per_component) as u64);
    }
}

#[test]
fn test_type_classification_mutual_exclusion() {
    // A format should only match one category (float, signed int, unsigned int, or normalized)
    // Note: normalized formats are neither float nor int
    let all_formats = [
        VertexFormat::Uint8x4, VertexFormat::Sint8x4,
        VertexFormat::Unorm8x4, VertexFormat::Snorm8x4,
        VertexFormat::Float32x3, VertexFormat::Float64x2,
        VertexFormat::Uint32, VertexFormat::Sint32,
    ];

    for format in all_formats {
        let is_float = vertex_format_is_float(format);
        let is_signed = vertex_format_is_signed_int(format);
        let is_unsigned = vertex_format_is_unsigned_int(format);
        let is_normalized = vertex_format_is_normalized(format);

        let categories = [is_float, is_signed, is_unsigned].iter().filter(|&&x| x).count();

        // Float, signed int, and unsigned int should be mutually exclusive
        assert!(
            categories <= 1,
            "Format {:?} matches multiple type categories: float={}, signed={}, unsigned={}",
            format, is_float, is_signed, is_unsigned
        );

        // Normalized formats should not be classified as int or float
        if is_normalized {
            assert!(
                !is_signed && !is_unsigned && !is_float,
                "Normalized format {:?} should not be classified as int or float",
                format
            );
        }
    }
}

#[test]
fn test_stride_matches_offset_sum() {
    // Stride should equal the sum of all format sizes
    let formats = [
        VertexFormat::Float32x3,
        VertexFormat::Float32x3,
        VertexFormat::Float32x2,
        VertexFormat::Float32x4,
    ];

    let stride = calculate_stride(&formats);
    let offsets = calculate_offsets(&formats);
    let last_offset = offsets[formats.len() - 1];
    let last_size = vertex_format_size(formats[formats.len() - 1]);

    // Stride should equal last offset + last format size
    assert_eq!(stride, last_offset + last_size);
}

#[test]
fn test_large_vertex_layout() {
    // Test a very large vertex layout (e.g., for complex effects)
    let formats = [
        VertexFormat::Float32x4, // 16
        VertexFormat::Float32x4, // 16
        VertexFormat::Float32x4, // 16
        VertexFormat::Float32x4, // 16
        VertexFormat::Float32x4, // 16
        VertexFormat::Float32x4, // 16
        VertexFormat::Float32x4, // 16
        VertexFormat::Float32x4, // 16
    ];

    let stride = calculate_stride(&formats);
    assert_eq!(stride, 128);

    let offsets = calculate_offsets(&formats);
    assert_eq!(offsets, [0, 16, 32, 48, 64, 80, 96, 112]);
}

#[test]
fn test_compressed_vertex_layout() {
    // Test a compressed layout using smaller formats
    let stride = calculate_stride(&[
        VertexFormat::Float16x4,  // position (16-bit): 8
        VertexFormat::Snorm8x4,   // normal (8-bit): 4
        VertexFormat::Unorm16x2,  // uv (16-bit): 4
        VertexFormat::Snorm8x4,   // tangent (8-bit): 4
    ]);
    assert_eq!(stride, 20);

    // Compare to uncompressed
    let uncompressed = calculate_stride(&[
        VertexFormat::Float32x3,  // position: 12
        VertexFormat::Float32x3,  // normal: 12
        VertexFormat::Float32x2,  // uv: 8
        VertexFormat::Float32x4,  // tangent: 16
    ]);
    assert_eq!(uncompressed, 48);

    // Compressed saves 28 bytes per vertex (48 - 20)
    assert_eq!(uncompressed - stride, 28);
}

#[test]
fn test_vertex_format_info_equality() {
    let info1 = VertexFormatInfo::new(VertexFormat::Float32x3);
    let info2 = VertexFormatInfo::new(VertexFormat::Float32x3);
    let info3 = VertexFormatInfo::new(VertexFormat::Float32x4);

    // Same format should produce equal info
    assert_eq!(info1, info2);

    // Different format should produce different info
    assert_ne!(info1, info3);
}

#[test]
fn test_vertex_format_info_copy_clone() {
    let info1 = VertexFormatInfo::new(VertexFormat::Float32x3);
    let info2 = info1; // Copy
    let info3 = info1.clone();

    assert_eq!(info1, info2);
    assert_eq!(info1, info3);
}

#[test]
fn test_vertex_format_info_debug() {
    let info = VertexFormatInfo::new(VertexFormat::Float32x3);
    let debug_str = format!("{:?}", info);

    // Debug output should contain the format name
    assert!(debug_str.contains("Float32x3"));
}

// =============================================================================
// SECTION 15: CONSISTENCY TESTS WITH STRIDES MODULE
// =============================================================================

#[test]
fn test_pbr_stride_consistency() {
    // Verify PBR stride constant matches calculated stride
    let calculated = calculate_stride(&[
        common::POSITION, // Float32x3: 12
        common::NORMAL,   // Float32x3: 12
        common::UV,       // Float32x2: 8
        common::TANGENT,  // Float32x4: 16
    ]);
    assert_eq!(calculated, strides::PBR);
}

#[test]
fn test_terrain_stride_consistency() {
    let calculated = calculate_stride(&[
        common::POSITION, // Float32x3: 12
        common::NORMAL,   // Float32x3: 12
        common::UV,       // Float32x2: 8
    ]);
    assert_eq!(calculated, strides::TERRAIN);
}

#[test]
fn test_ui_stride_consistency() {
    let calculated = calculate_stride(&[
        common::POSITION_2D, // Float32x2: 8
        common::UV,          // Float32x2: 8
        common::COLOR,       // Unorm8x4: 4
    ]);
    assert_eq!(calculated, strides::UI);
}

#[test]
fn test_position_only_stride_consistency() {
    let calculated = calculate_stride(&[common::POSITION]);
    assert_eq!(calculated, strides::POSITION_ONLY);
    assert_eq!(calculated, strides::SHADOW);
}
