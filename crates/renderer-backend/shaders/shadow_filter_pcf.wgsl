// SPDX-License-Identifier: MIT
//
// shadow_filter_pcf.wgsl - Percentage Closer Filtering (T-LIT-6.4)
//
// Implements PCF shadow filtering with 4 kernel sizes (2x2, 3x3, 5x5, 7x7)
// and an optimized Poisson disk variant for high-quality soft shadows.
//
// Quality Levels:
// - Low:    2x2 bilinear (1 sample, hardware-accelerated)
// - Medium: 3x3 kernel (9 samples)
// - High:   5x5 kernel (25 samples)
// - Ultra:  Poisson disk (16 samples with per-pixel rotation)
//
// Dependencies:
// - shadow_common.wgsl: ShadowTileInfo, ShadowConfig types
//
// Usage:
// If using a shader preprocessor, include shadow_common.wgsl first.
// Otherwise, this file includes ShadowTileInfo inline (must match shadow_common.wgsl).
//
// #include "shadow_common.wgsl"

// ============================================================================
// Shared Types (must match shadow_common.wgsl)
// ============================================================================

// Note: If using a preprocessor that handles #include, comment out this block.
// These definitions must stay synchronized with shadow_common.wgsl.

#ifndef SHADOW_COMMON_INCLUDED

/// Per-tile shadow atlas information.
/// Used for both tiled shadow atlases and CSM cascade selection.
struct ShadowTileInfo {
    /// Atlas UV offset for this tile (bottom-left corner in UV space).
    uv_offset: vec2<f32>,
    /// Atlas UV scale for this tile (tile size in UV space).
    uv_scale: vec2<f32>,
    /// World-to-light clip space transformation matrix.
    light_space_matrix: mat4x4<f32>,
    /// For CSM: which cascade this tile represents (0-3).
    /// For point/spot lights: light index.
    cascade_index: u32,
    /// PCF kernel size in texels (e.g., 3.0 for 3x3 kernel).
    filter_size: f32,
    /// Constant depth bias applied to all fragments.
    bias_constant: f32,
    /// Slope-scaled bias multiplier (multiplied by surface slope).
    bias_slope: f32,
}

#endif // SHADOW_COMMON_INCLUDED

// ============================================================================
// Constants
// ============================================================================

const PCF_EPSILON: f32 = 0.00001;
const PI: f32 = 3.14159265359;
const TAU: f32 = 6.28318530718;

// ============================================================================
// Kernel Offset Patterns
// ============================================================================

/// 2x2 PCF kernel offsets (4 samples).
/// Uses half-texel offsets for optimal bilinear interpolation.
const PCF_2X2_OFFSETS: array<vec2<f32>, 4> = array(
    vec2<f32>(-0.5, -0.5),
    vec2<f32>( 0.5, -0.5),
    vec2<f32>(-0.5,  0.5),
    vec2<f32>( 0.5,  0.5)
);

/// 3x3 PCF kernel offsets (9 samples).
/// Standard grid pattern centered on the sample point.
const PCF_3X3_OFFSETS: array<vec2<f32>, 9> = array(
    vec2<f32>(-1.0, -1.0), vec2<f32>( 0.0, -1.0), vec2<f32>( 1.0, -1.0),
    vec2<f32>(-1.0,  0.0), vec2<f32>( 0.0,  0.0), vec2<f32>( 1.0,  0.0),
    vec2<f32>(-1.0,  1.0), vec2<f32>( 0.0,  1.0), vec2<f32>( 1.0,  1.0)
);

/// 5x5 PCF kernel offsets (25 samples).
/// Full grid pattern for high-quality filtering.
const PCF_5X5_OFFSETS: array<vec2<f32>, 25> = array(
    vec2<f32>(-2.0, -2.0), vec2<f32>(-1.0, -2.0), vec2<f32>( 0.0, -2.0), vec2<f32>( 1.0, -2.0), vec2<f32>( 2.0, -2.0),
    vec2<f32>(-2.0, -1.0), vec2<f32>(-1.0, -1.0), vec2<f32>( 0.0, -1.0), vec2<f32>( 1.0, -1.0), vec2<f32>( 2.0, -1.0),
    vec2<f32>(-2.0,  0.0), vec2<f32>(-1.0,  0.0), vec2<f32>( 0.0,  0.0), vec2<f32>( 1.0,  0.0), vec2<f32>( 2.0,  0.0),
    vec2<f32>(-2.0,  1.0), vec2<f32>(-1.0,  1.0), vec2<f32>( 0.0,  1.0), vec2<f32>( 1.0,  1.0), vec2<f32>( 2.0,  1.0),
    vec2<f32>(-2.0,  2.0), vec2<f32>(-1.0,  2.0), vec2<f32>( 0.0,  2.0), vec2<f32>( 1.0,  2.0), vec2<f32>( 2.0,  2.0)
);

/// 7x7 PCF kernel offsets (49 samples).
/// Maximum quality grid pattern for ultra-smooth shadows.
const PCF_7X7_OFFSETS: array<vec2<f32>, 49> = array(
    // Row -3
    vec2<f32>(-3.0, -3.0), vec2<f32>(-2.0, -3.0), vec2<f32>(-1.0, -3.0), vec2<f32>( 0.0, -3.0),
    vec2<f32>( 1.0, -3.0), vec2<f32>( 2.0, -3.0), vec2<f32>( 3.0, -3.0),
    // Row -2
    vec2<f32>(-3.0, -2.0), vec2<f32>(-2.0, -2.0), vec2<f32>(-1.0, -2.0), vec2<f32>( 0.0, -2.0),
    vec2<f32>( 1.0, -2.0), vec2<f32>( 2.0, -2.0), vec2<f32>( 3.0, -2.0),
    // Row -1
    vec2<f32>(-3.0, -1.0), vec2<f32>(-2.0, -1.0), vec2<f32>(-1.0, -1.0), vec2<f32>( 0.0, -1.0),
    vec2<f32>( 1.0, -1.0), vec2<f32>( 2.0, -1.0), vec2<f32>( 3.0, -1.0),
    // Row 0
    vec2<f32>(-3.0,  0.0), vec2<f32>(-2.0,  0.0), vec2<f32>(-1.0,  0.0), vec2<f32>( 0.0,  0.0),
    vec2<f32>( 1.0,  0.0), vec2<f32>( 2.0,  0.0), vec2<f32>( 3.0,  0.0),
    // Row 1
    vec2<f32>(-3.0,  1.0), vec2<f32>(-2.0,  1.0), vec2<f32>(-1.0,  1.0), vec2<f32>( 0.0,  1.0),
    vec2<f32>( 1.0,  1.0), vec2<f32>( 2.0,  1.0), vec2<f32>( 3.0,  1.0),
    // Row 2
    vec2<f32>(-3.0,  2.0), vec2<f32>(-2.0,  2.0), vec2<f32>(-1.0,  2.0), vec2<f32>( 0.0,  2.0),
    vec2<f32>( 1.0,  2.0), vec2<f32>( 2.0,  2.0), vec2<f32>( 3.0,  2.0),
    // Row 3
    vec2<f32>(-3.0,  3.0), vec2<f32>(-2.0,  3.0), vec2<f32>(-1.0,  3.0), vec2<f32>( 0.0,  3.0),
    vec2<f32>( 1.0,  3.0), vec2<f32>( 2.0,  3.0), vec2<f32>( 3.0,  3.0)
);

/// 16-sample Poisson disk pattern for high-quality filtering.
/// Pre-computed blue-noise distribution for minimal banding artifacts.
const POISSON_DISK_16: array<vec2<f32>, 16> = array(
    vec2<f32>(-0.94201624, -0.39906216),
    vec2<f32>( 0.94558609, -0.76890725),
    vec2<f32>(-0.09418410, -0.92938870),
    vec2<f32>( 0.34495938,  0.29387760),
    vec2<f32>(-0.91588581,  0.45771432),
    vec2<f32>(-0.81544232, -0.87912464),
    vec2<f32>(-0.38277543,  0.27676845),
    vec2<f32>( 0.97484398,  0.75648379),
    vec2<f32>( 0.44323325, -0.97511554),
    vec2<f32>( 0.53742981, -0.47373420),
    vec2<f32>(-0.26496911, -0.41893023),
    vec2<f32>( 0.79197514,  0.19090188),
    vec2<f32>(-0.24188840,  0.99706507),
    vec2<f32>(-0.81409955,  0.91437590),
    vec2<f32>( 0.19984126,  0.78641367),
    vec2<f32>( 0.14383161, -0.14100790)
);

/// 32-sample Poisson disk pattern for ultra-high quality.
/// Extended pattern for very large penumbra sizes.
const POISSON_DISK_32: array<vec2<f32>, 32> = array(
    vec2<f32>(-0.94201624, -0.39906216),
    vec2<f32>( 0.94558609, -0.76890725),
    vec2<f32>(-0.09418410, -0.92938870),
    vec2<f32>( 0.34495938,  0.29387760),
    vec2<f32>(-0.91588581,  0.45771432),
    vec2<f32>(-0.81544232, -0.87912464),
    vec2<f32>(-0.38277543,  0.27676845),
    vec2<f32>( 0.97484398,  0.75648379),
    vec2<f32>( 0.44323325, -0.97511554),
    vec2<f32>( 0.53742981, -0.47373420),
    vec2<f32>(-0.26496911, -0.41893023),
    vec2<f32>( 0.79197514,  0.19090188),
    vec2<f32>(-0.24188840,  0.99706507),
    vec2<f32>(-0.81409955,  0.91437590),
    vec2<f32>( 0.19984126,  0.78641367),
    vec2<f32>( 0.14383161, -0.14100790),
    vec2<f32>(-0.40756750,  0.84127750),
    vec2<f32>( 0.60543710, -0.12348520),
    vec2<f32>(-0.65231420, -0.21548930),
    vec2<f32>( 0.12948760,  0.42857390),
    vec2<f32>(-0.23847520, -0.75839240),
    vec2<f32>( 0.83629450,  0.48573920),
    vec2<f32>(-0.72938470,  0.63847290),
    vec2<f32>( 0.47382910, -0.69284730),
    vec2<f32>(-0.58293740,  0.09283740),
    vec2<f32>( 0.29384750,  0.82937460),
    vec2<f32>(-0.84729370, -0.49283740),
    vec2<f32>( 0.67839240, -0.38472930),
    vec2<f32>(-0.19283740,  0.57382940),
    vec2<f32>( 0.38472930,  0.58293740),
    vec2<f32>(-0.49283740, -0.58293740),
    vec2<f32>( 0.82937460, -0.19283740)
);

// ============================================================================
// Helper Functions
// ============================================================================

/// Creates a 2D rotation matrix from an angle.
///
/// Parameters:
/// - angle: Rotation angle in radians.
///
/// Returns: 2x2 rotation matrix.
fn rotation_matrix_2d(angle: f32) -> mat2x2<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return mat2x2<f32>(
        vec2<f32>(c, -s),
        vec2<f32>(s,  c)
    );
}

/// Retrieves a Poisson disk sample with optional rotation.
///
/// Parameters:
/// - index: Sample index (0-15 for 16-sample disk).
/// - rotation: Per-pixel rotation angle in radians.
///
/// Returns: Rotated 2D sample offset in [-1, 1] range.
fn get_poisson_sample(index: u32, rotation: f32) -> vec2<f32> {
    let sample = POISSON_DISK_16[index % 16u];
    let rot_mat = rotation_matrix_2d(rotation);
    return rot_mat * sample;
}

/// Retrieves a Poisson disk sample from the 32-sample disk.
///
/// Parameters:
/// - index: Sample index (0-31).
/// - rotation: Per-pixel rotation angle in radians.
///
/// Returns: Rotated 2D sample offset.
fn get_poisson_sample_32(index: u32, rotation: f32) -> vec2<f32> {
    let sample = POISSON_DISK_32[index % 32u];
    let rot_mat = rotation_matrix_2d(rotation);
    return rot_mat * sample;
}

/// Generates interleaved gradient noise for per-pixel rotation.
/// Provides temporally stable noise that works well with TAA.
///
/// Parameters:
/// - screen_pos: Screen-space pixel position.
///
/// Returns: Noise value in [0, 1] suitable for rotation angle.
fn interleaved_gradient_noise_pcf(screen_pos: vec2<f32>) -> f32 {
    let magic = vec3<f32>(0.06711056, 0.00583715, 52.9829189);
    return fract(magic.z * fract(dot(screen_pos, magic.xy)));
}

/// Computes per-pixel rotation angle from screen position.
///
/// Parameters:
/// - screen_pos: Screen-space pixel position.
///
/// Returns: Rotation angle in radians [0, 2*PI).
fn compute_rotation_angle(screen_pos: vec2<f32>) -> f32 {
    return interleaved_gradient_noise_pcf(screen_pos) * TAU;
}

// ============================================================================
// Core PCF Functions
// ============================================================================

/// Optimized 2x2 PCF using hardware bilinear filtering.
/// Single sample leverages GPU's built-in bilinear interpolation.
///
/// Quality: Low (1 hardware sample, effectively 4 depth comparisons)
/// Performance: Best
///
/// Parameters:
/// - shadow_map: Depth comparison texture.
/// - shadow_sampler: Comparison sampler configured for depth testing.
/// - uv: Shadow map UV coordinates.
/// - depth: Reference depth for comparison.
///
/// Returns: Shadow factor in [0, 1]: 0 = fully shadowed, 1 = fully lit.
fn pcf_shadow_bilinear(
    shadow_map: texture_depth_2d,
    shadow_sampler: sampler_comparison,
    uv: vec2<f32>,
    depth: f32
) -> f32 {
    // Single sample with hardware bilinear interpolation.
    // The comparison sampler automatically performs 2x2 PCF
    // when the GPU supports shadow comparison with filtering.
    return textureSampleCompare(shadow_map, shadow_sampler, uv, depth);
}

/// 2x2 PCF kernel (4 explicit samples).
/// Manual implementation for consistency across hardware.
///
/// Quality: Low
/// Performance: Very Good
///
/// Parameters:
/// - shadow_map: Depth comparison texture.
/// - shadow_sampler: Comparison sampler.
/// - uv: Shadow map UV coordinates.
/// - depth: Reference depth for comparison.
/// - texel_size: Size of one texel in UV space.
///
/// Returns: Shadow factor in [0, 1].
fn pcf_shadow_2x2(
    shadow_map: texture_depth_2d,
    shadow_sampler: sampler_comparison,
    uv: vec2<f32>,
    depth: f32,
    texel_size: f32
) -> f32 {
    var shadow = 0.0;

    for (var i = 0u; i < 4u; i++) {
        let offset = PCF_2X2_OFFSETS[i] * texel_size;
        shadow += textureSampleCompare(shadow_map, shadow_sampler, uv + offset, depth);
    }

    return shadow * 0.25; // Divide by 4
}

/// 3x3 PCF kernel (9 samples).
/// Good balance of quality and performance.
///
/// Quality: Medium
/// Performance: Good
///
/// Parameters:
/// - shadow_map: Depth comparison texture.
/// - shadow_sampler: Comparison sampler.
/// - uv: Shadow map UV coordinates.
/// - depth: Reference depth for comparison.
/// - texel_size: Size of one texel in UV space.
///
/// Returns: Shadow factor in [0, 1].
fn pcf_shadow_3x3(
    shadow_map: texture_depth_2d,
    shadow_sampler: sampler_comparison,
    uv: vec2<f32>,
    depth: f32,
    texel_size: f32
) -> f32 {
    var shadow = 0.0;

    for (var i = 0u; i < 9u; i++) {
        let offset = PCF_3X3_OFFSETS[i] * texel_size;
        shadow += textureSampleCompare(shadow_map, shadow_sampler, uv + offset, depth);
    }

    return shadow / 9.0;
}

/// 5x5 PCF kernel (25 samples).
/// High-quality filtering for smooth shadow edges.
///
/// Quality: High
/// Performance: Moderate
///
/// Parameters:
/// - shadow_map: Depth comparison texture.
/// - shadow_sampler: Comparison sampler.
/// - uv: Shadow map UV coordinates.
/// - depth: Reference depth for comparison.
/// - texel_size: Size of one texel in UV space.
///
/// Returns: Shadow factor in [0, 1].
fn pcf_shadow_5x5(
    shadow_map: texture_depth_2d,
    shadow_sampler: sampler_comparison,
    uv: vec2<f32>,
    depth: f32,
    texel_size: f32
) -> f32 {
    var shadow = 0.0;

    for (var i = 0u; i < 25u; i++) {
        let offset = PCF_5X5_OFFSETS[i] * texel_size;
        shadow += textureSampleCompare(shadow_map, shadow_sampler, uv + offset, depth);
    }

    return shadow / 25.0;
}

/// 7x7 PCF kernel (49 samples).
/// Maximum quality grid-based filtering.
///
/// Quality: Very High
/// Performance: Lower
///
/// Parameters:
/// - shadow_map: Depth comparison texture.
/// - shadow_sampler: Comparison sampler.
/// - uv: Shadow map UV coordinates.
/// - depth: Reference depth for comparison.
/// - texel_size: Size of one texel in UV space.
///
/// Returns: Shadow factor in [0, 1].
fn pcf_shadow_7x7(
    shadow_map: texture_depth_2d,
    shadow_sampler: sampler_comparison,
    uv: vec2<f32>,
    depth: f32,
    texel_size: f32
) -> f32 {
    var shadow = 0.0;

    for (var i = 0u; i < 49u; i++) {
        let offset = PCF_7X7_OFFSETS[i] * texel_size;
        shadow += textureSampleCompare(shadow_map, shadow_sampler, uv + offset, depth);
    }

    return shadow / 49.0;
}

// ============================================================================
// Main PCF Function with Kernel Selection
// ============================================================================

/// Main PCF shadow sampling function with selectable kernel size.
///
/// Dispatches to the appropriate kernel implementation based on size parameter.
/// Use this for dynamic quality selection at runtime.
///
/// Parameters:
/// - shadow_map: Depth comparison texture.
/// - shadow_sampler: Comparison sampler configured for depth testing.
/// - uv: Shadow map UV coordinates in atlas space.
/// - depth: Reference depth for shadow comparison.
/// - tile: Shadow tile information containing filter parameters.
/// - kernel_size: Kernel dimension (2, 3, 5, or 7).
///
/// Returns: Shadow factor in [0, 1]: 0 = fully shadowed, 1 = fully lit.
fn pcf_shadow(
    shadow_map: texture_depth_2d,
    shadow_sampler: sampler_comparison,
    uv: vec2<f32>,
    depth: f32,
    tile: ShadowTileInfo,
    kernel_size: u32
) -> f32 {
    // Compute texel size based on tile filter size and shadow map dimensions.
    let shadow_map_size = vec2<f32>(textureDimensions(shadow_map));
    let texel_size = tile.filter_size / shadow_map_size.x;

    var shadow = 0.0;

    switch kernel_size {
        case 2u: {
            // 2x2 PCF: 4 samples
            for (var i = 0u; i < 4u; i++) {
                let offset = PCF_2X2_OFFSETS[i] * texel_size;
                shadow += textureSampleCompare(shadow_map, shadow_sampler, uv + offset, depth);
            }
            shadow /= 4.0;
        }
        case 3u: {
            // 3x3 PCF: 9 samples
            for (var i = 0u; i < 9u; i++) {
                let offset = PCF_3X3_OFFSETS[i] * texel_size;
                shadow += textureSampleCompare(shadow_map, shadow_sampler, uv + offset, depth);
            }
            shadow /= 9.0;
        }
        case 5u: {
            // 5x5 PCF: 25 samples
            for (var i = 0u; i < 25u; i++) {
                let offset = PCF_5X5_OFFSETS[i] * texel_size;
                shadow += textureSampleCompare(shadow_map, shadow_sampler, uv + offset, depth);
            }
            shadow /= 25.0;
        }
        case 7u: {
            // 7x7 PCF: 49 samples
            for (var i = 0u; i < 49u; i++) {
                let offset = PCF_7X7_OFFSETS[i] * texel_size;
                shadow += textureSampleCompare(shadow_map, shadow_sampler, uv + offset, depth);
            }
            shadow /= 49.0;
        }
        default: {
            // Fallback: single bilinear sample
            shadow = textureSampleCompare(shadow_map, shadow_sampler, uv, depth);
        }
    }

    return shadow;
}

// ============================================================================
// Poisson Disk PCF Variants
// ============================================================================

/// 16-sample Poisson disk PCF with per-pixel rotation.
/// Provides high-quality filtering with reduced banding artifacts.
///
/// Quality: Ultra
/// Performance: Good (16 samples, better distribution than grid)
///
/// Parameters:
/// - shadow_map: Depth comparison texture.
/// - shadow_sampler: Comparison sampler.
/// - uv: Shadow map UV coordinates.
/// - depth: Reference depth for comparison.
/// - tile: Shadow tile information.
/// - rotation: Per-pixel rotation angle in radians.
///
/// Returns: Shadow factor in [0, 1].
fn pcf_shadow_poisson(
    shadow_map: texture_depth_2d,
    shadow_sampler: sampler_comparison,
    uv: vec2<f32>,
    depth: f32,
    tile: ShadowTileInfo,
    rotation: f32
) -> f32 {
    let shadow_map_size = vec2<f32>(textureDimensions(shadow_map));
    let texel_size = tile.filter_size / shadow_map_size.x;
    let rot_mat = rotation_matrix_2d(rotation);

    var shadow = 0.0;

    for (var i = 0u; i < 16u; i++) {
        let rotated_offset = rot_mat * POISSON_DISK_16[i];
        let offset = rotated_offset * texel_size;
        shadow += textureSampleCompare(shadow_map, shadow_sampler, uv + offset, depth);
    }

    return shadow / 16.0;
}

/// 32-sample Poisson disk PCF for maximum quality.
/// Use for close-up shadows or when highest quality is required.
///
/// Quality: Ultra+
/// Performance: Lower (32 samples)
///
/// Parameters:
/// - shadow_map: Depth comparison texture.
/// - shadow_sampler: Comparison sampler.
/// - uv: Shadow map UV coordinates.
/// - depth: Reference depth for comparison.
/// - tile: Shadow tile information.
/// - rotation: Per-pixel rotation angle in radians.
///
/// Returns: Shadow factor in [0, 1].
fn pcf_shadow_poisson_32(
    shadow_map: texture_depth_2d,
    shadow_sampler: sampler_comparison,
    uv: vec2<f32>,
    depth: f32,
    tile: ShadowTileInfo,
    rotation: f32
) -> f32 {
    let shadow_map_size = vec2<f32>(textureDimensions(shadow_map));
    let texel_size = tile.filter_size / shadow_map_size.x;
    let rot_mat = rotation_matrix_2d(rotation);

    var shadow = 0.0;

    for (var i = 0u; i < 32u; i++) {
        let rotated_offset = rot_mat * POISSON_DISK_32[i];
        let offset = rotated_offset * texel_size;
        shadow += textureSampleCompare(shadow_map, shadow_sampler, uv + offset, depth);
    }

    return shadow / 32.0;
}

/// Convenience function: Poisson PCF with automatic rotation from screen position.
///
/// Parameters:
/// - shadow_map: Depth comparison texture.
/// - shadow_sampler: Comparison sampler.
/// - uv: Shadow map UV coordinates.
/// - depth: Reference depth for comparison.
/// - tile: Shadow tile information.
/// - screen_pos: Screen-space pixel position for rotation generation.
///
/// Returns: Shadow factor in [0, 1].
fn pcf_shadow_poisson_auto(
    shadow_map: texture_depth_2d,
    shadow_sampler: sampler_comparison,
    uv: vec2<f32>,
    depth: f32,
    tile: ShadowTileInfo,
    screen_pos: vec2<f32>
) -> f32 {
    let rotation = compute_rotation_angle(screen_pos);
    return pcf_shadow_poisson(shadow_map, shadow_sampler, uv, depth, tile, rotation);
}

// ============================================================================
// Optimized PCF Variants
// ============================================================================

/// Optimized 3x3 PCF using bilinear filtering to reduce samples.
/// Takes 4 bilinear samples instead of 9 point samples.
///
/// Quality: Medium (equivalent to 3x3)
/// Performance: Better than standard 3x3 (4 samples vs 9)
///
/// Parameters:
/// - shadow_map: Depth comparison texture.
/// - shadow_sampler: Comparison sampler.
/// - uv: Shadow map UV coordinates.
/// - depth: Reference depth for comparison.
/// - texel_size: Size of one texel in UV space.
///
/// Returns: Shadow factor in [0, 1].
fn pcf_shadow_3x3_optimized(
    shadow_map: texture_depth_2d,
    shadow_sampler: sampler_comparison,
    uv: vec2<f32>,
    depth: f32,
    texel_size: f32
) -> f32 {
    // Use 4 bilinear samples at strategic positions to cover 3x3 area.
    // Each sample covers a 2x2 region with bilinear weighting.
    let offsets = array<vec2<f32>, 4>(
        vec2<f32>(-0.5, -0.5),
        vec2<f32>( 0.5, -0.5),
        vec2<f32>(-0.5,  0.5),
        vec2<f32>( 0.5,  0.5)
    );

    var shadow = 0.0;

    for (var i = 0u; i < 4u; i++) {
        let offset = offsets[i] * texel_size;
        shadow += textureSampleCompare(shadow_map, shadow_sampler, uv + offset, depth);
    }

    return shadow * 0.25;
}

/// Optimized 5x5 PCF using bilinear filtering.
/// Takes 9 bilinear samples instead of 25 point samples.
///
/// Quality: High (equivalent to 5x5)
/// Performance: Better than standard 5x5 (9 samples vs 25)
///
/// Parameters:
/// - shadow_map: Depth comparison texture.
/// - shadow_sampler: Comparison sampler.
/// - uv: Shadow map UV coordinates.
/// - depth: Reference depth for comparison.
/// - texel_size: Size of one texel in UV space.
///
/// Returns: Shadow factor in [0, 1].
fn pcf_shadow_5x5_optimized(
    shadow_map: texture_depth_2d,
    shadow_sampler: sampler_comparison,
    uv: vec2<f32>,
    depth: f32,
    texel_size: f32
) -> f32 {
    // 3x3 grid of bilinear samples covering 5x5 texel area.
    let offsets = array<vec2<f32>, 9>(
        vec2<f32>(-1.5, -1.5), vec2<f32>( 0.0, -1.5), vec2<f32>( 1.5, -1.5),
        vec2<f32>(-1.5,  0.0), vec2<f32>( 0.0,  0.0), vec2<f32>( 1.5,  0.0),
        vec2<f32>(-1.5,  1.5), vec2<f32>( 0.0,  1.5), vec2<f32>( 1.5,  1.5)
    );

    // Weights for proper reconstruction of 5x5 kernel.
    let weights = array<f32, 9>(
        1.0, 2.0, 1.0,
        2.0, 4.0, 2.0,
        1.0, 2.0, 1.0
    );
    let total_weight = 16.0;

    var shadow = 0.0;

    for (var i = 0u; i < 9u; i++) {
        let offset = offsets[i] * texel_size;
        shadow += textureSampleCompare(shadow_map, shadow_sampler, uv + offset, depth) * weights[i];
    }

    return shadow / total_weight;
}

// ============================================================================
// Adaptive PCF
// ============================================================================

/// Adaptive PCF that selects kernel size based on shadow edge detection.
/// Reduces sample count in fully lit/shadowed areas.
///
/// Quality: Variable (adapts to shadow complexity)
/// Performance: Better average case than fixed high-quality
///
/// Parameters:
/// - shadow_map: Depth comparison texture.
/// - shadow_sampler: Comparison sampler.
/// - uv: Shadow map UV coordinates.
/// - depth: Reference depth for comparison.
/// - tile: Shadow tile information.
/// - screen_pos: Screen-space pixel position.
///
/// Returns: Shadow factor in [0, 1].
fn pcf_shadow_adaptive(
    shadow_map: texture_depth_2d,
    shadow_sampler: sampler_comparison,
    uv: vec2<f32>,
    depth: f32,
    tile: ShadowTileInfo,
    screen_pos: vec2<f32>
) -> f32 {
    let shadow_map_size = vec2<f32>(textureDimensions(shadow_map));
    let texel_size = tile.filter_size / shadow_map_size.x;

    // First pass: cheap 2x2 to detect if we're near a shadow edge.
    var coarse_shadow = 0.0;
    for (var i = 0u; i < 4u; i++) {
        let offset = PCF_2X2_OFFSETS[i] * texel_size;
        coarse_shadow += textureSampleCompare(shadow_map, shadow_sampler, uv + offset, depth);
    }
    coarse_shadow *= 0.25;

    // If fully lit or fully shadowed, return early.
    if (coarse_shadow < PCF_EPSILON || coarse_shadow > 1.0 - PCF_EPSILON) {
        return coarse_shadow;
    }

    // Near shadow edge: use higher quality Poisson sampling.
    let rotation = compute_rotation_angle(screen_pos);
    return pcf_shadow_poisson(shadow_map, shadow_sampler, uv, depth, tile, rotation);
}

// ============================================================================
// Quality Level Selection
// ============================================================================

/// PCF quality levels for easy selection.
const PCF_QUALITY_LOW: u32 = 0u;      // 2x2 bilinear (1 sample)
const PCF_QUALITY_MEDIUM: u32 = 1u;   // 3x3 (9 samples)
const PCF_QUALITY_HIGH: u32 = 2u;     // 5x5 (25 samples)
const PCF_QUALITY_ULTRA: u32 = 3u;    // Poisson disk (16 samples with rotation)

/// Samples shadow with quality level selection.
///
/// Parameters:
/// - shadow_map: Depth comparison texture.
/// - shadow_sampler: Comparison sampler.
/// - uv: Shadow map UV coordinates.
/// - depth: Reference depth for comparison.
/// - tile: Shadow tile information.
/// - quality: Quality level (0=Low, 1=Medium, 2=High, 3=Ultra).
/// - screen_pos: Screen-space pixel position (used for Ultra quality rotation).
///
/// Returns: Shadow factor in [0, 1].
fn pcf_shadow_quality(
    shadow_map: texture_depth_2d,
    shadow_sampler: sampler_comparison,
    uv: vec2<f32>,
    depth: f32,
    tile: ShadowTileInfo,
    quality: u32,
    screen_pos: vec2<f32>
) -> f32 {
    let shadow_map_size = vec2<f32>(textureDimensions(shadow_map));
    let texel_size = tile.filter_size / shadow_map_size.x;

    switch quality {
        case PCF_QUALITY_LOW: {
            // 2x2 bilinear: single hardware-accelerated sample
            return textureSampleCompare(shadow_map, shadow_sampler, uv, depth);
        }
        case PCF_QUALITY_MEDIUM: {
            // 3x3 kernel
            return pcf_shadow_3x3(shadow_map, shadow_sampler, uv, depth, texel_size);
        }
        case PCF_QUALITY_HIGH: {
            // 5x5 kernel
            return pcf_shadow_5x5(shadow_map, shadow_sampler, uv, depth, texel_size);
        }
        case PCF_QUALITY_ULTRA: {
            // 16-sample Poisson disk with per-pixel rotation
            let rotation = compute_rotation_angle(screen_pos);
            return pcf_shadow_poisson(shadow_map, shadow_sampler, uv, depth, tile, rotation);
        }
        default: {
            return textureSampleCompare(shadow_map, shadow_sampler, uv, depth);
        }
    }
}

// ============================================================================
// Cascaded Shadow Map PCF
// ============================================================================

/// PCF sampling for cascaded shadow maps with cascade blending.
///
/// Parameters:
/// - shadow_map: Depth comparison texture (atlas containing all cascades).
/// - shadow_sampler: Comparison sampler.
/// - uv: Shadow map UV coordinates in cascade's tile.
/// - depth: Reference depth for comparison.
/// - tile: Current cascade's tile information.
/// - next_tile: Next cascade's tile information (for blending).
/// - next_uv: UV coordinates in next cascade.
/// - next_depth: Depth in next cascade.
/// - blend_factor: Blend factor between cascades [0, 1].
/// - kernel_size: PCF kernel size.
///
/// Returns: Blended shadow factor in [0, 1].
fn pcf_shadow_csm_blend(
    shadow_map: texture_depth_2d,
    shadow_sampler: sampler_comparison,
    uv: vec2<f32>,
    depth: f32,
    tile: ShadowTileInfo,
    next_tile: ShadowTileInfo,
    next_uv: vec2<f32>,
    next_depth: f32,
    blend_factor: f32,
    kernel_size: u32
) -> f32 {
    let current_shadow = pcf_shadow(shadow_map, shadow_sampler, uv, depth, tile, kernel_size);

    // Early out if not blending
    if (blend_factor < PCF_EPSILON) {
        return current_shadow;
    }

    let next_shadow = pcf_shadow(shadow_map, shadow_sampler, next_uv, next_depth, next_tile, kernel_size);

    return mix(current_shadow, next_shadow, blend_factor);
}

// ============================================================================
// Debug Visualization
// ============================================================================

/// Returns a color representing the kernel size for debug visualization.
///
/// Parameters:
/// - kernel_size: PCF kernel dimension.
///
/// Returns: RGB color for visualization.
fn pcf_debug_kernel_color(kernel_size: u32) -> vec3<f32> {
    switch kernel_size {
        case 2u: { return vec3<f32>(0.0, 1.0, 0.0); }  // Green: 2x2
        case 3u: { return vec3<f32>(0.0, 0.0, 1.0); }  // Blue: 3x3
        case 5u: { return vec3<f32>(1.0, 1.0, 0.0); }  // Yellow: 5x5
        case 7u: { return vec3<f32>(1.0, 0.0, 0.0); }  // Red: 7x7
        default: { return vec3<f32>(1.0, 1.0, 1.0); } // White: bilinear
    }
}

/// Returns a color representing the quality level for debug visualization.
///
/// Parameters:
/// - quality: Quality level (0-3).
///
/// Returns: RGB color for visualization.
fn pcf_debug_quality_color(quality: u32) -> vec3<f32> {
    switch quality {
        case PCF_QUALITY_LOW:    { return vec3<f32>(0.0, 1.0, 0.0); } // Green
        case PCF_QUALITY_MEDIUM: { return vec3<f32>(1.0, 1.0, 0.0); } // Yellow
        case PCF_QUALITY_HIGH:   { return vec3<f32>(1.0, 0.5, 0.0); } // Orange
        case PCF_QUALITY_ULTRA:  { return vec3<f32>(1.0, 0.0, 0.0); } // Red
        default: { return vec3<f32>(1.0, 1.0, 1.0); }
    }
}
