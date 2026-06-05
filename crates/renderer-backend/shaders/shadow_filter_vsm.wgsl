// SPDX-License-Identifier: MIT
//
// shadow_filter_vsm.wgsl — Variance Shadow Maps with Chebyshev inequality (T-LIT-6.6).
//
// Implements VSM shadow filtering using statistical variance analysis for soft
// shadow edges. VSM stores depth moments (mean, mean of squared depth) and uses
// Chebyshev's inequality to compute upper bounds on shadow probability.
//
// Key features:
// - Chebyshev inequality for probabilistic shadow estimation
// - Light bleed reduction via rescaling and exponential clamping
// - Separable Gaussian blur for pre-filtering shadow maps
// - Support for both RG16F and RG32F moment formats
//
// This module is designed to be imported by:
// - pbr.frag.wgsl        (main PBR fragment shader)
// - lighting_pass.wgsl   (deferred lighting compute shader)
//
// Research Notes:
// - Recommended bleed_reduction_exponent: 32-64 for most scenes
//   - Lower values (8-16): Softer shadows, more light bleeding
//   - Higher values (64-128): Sharper shadows, may cause banding
// - Thin occluder behavior: VSM can exhibit light bleeding when thin
//   occluders cast shadows. Mitigation strategies:
//   1. Use EVSM (Exponential VSM) for better thin occluder handling
//   2. Increase bleed_reduction_exponent at cost of shadow softness
//   3. Combine with contact shadows for near-occluder precision
// - Pre-filtering: Gaussian blur should be applied BEFORE VSM sampling,
//   as filtering in moment space preserves variance properties.

// ============================================================================
// Constants
// ============================================================================

/// Minimum variance to prevent divide-by-zero in Chebyshev computation.
const VSM_MIN_VARIANCE: f32 = 0.00001;

/// Default light bleeding threshold for rescaling.
const VSM_BLEED_THRESHOLD: f32 = 0.2;

/// Maximum supported kernel size for Gaussian blur.
const VSM_MAX_KERNEL_SIZE: u32 = 9u;

// ============================================================================
// Data Structures
// ============================================================================

/// VSM-specific configuration parameters.
/// These should be tuned per-scene for optimal shadow quality.
struct VsmParams {
    /// Exponent for light bleed reduction (8-64 recommended).
    /// Higher values reduce bleeding but may cause shadow banding.
    /// Typical values: 32 for outdoor, 64 for indoor with thin objects.
    bleed_reduction_exponent: f32,

    /// Minimum variance threshold to prevent divide-by-zero.
    /// Should be very small (1e-5 to 1e-7). Larger values reduce
    /// shadow sharpness but improve numerical stability.
    min_variance: f32,

    /// Scale factor applied to depth values before moment storage.
    /// Use values < 1.0 to fit depth range in limited precision formats.
    /// Typically 1.0 for RG32F, 0.5-1.0 for RG16F.
    depth_scale: f32,

    /// Light bleeding threshold for linear rescaling [0, 1).
    /// Shadows with probability below this are clamped to 0.
    /// Typical values: 0.1-0.3. Higher = less bleeding, darker shadows.
    bleed_threshold: f32,
}

/// Creates default VSM parameters suitable for most scenes.
fn vsm_params_default() -> VsmParams {
    return VsmParams(
        32.0,           // bleed_reduction_exponent
        VSM_MIN_VARIANCE, // min_variance
        1.0,            // depth_scale
        VSM_BLEED_THRESHOLD, // bleed_threshold
    );
}

/// Blur configuration for separable Gaussian filter.
struct VsmBlurConfig {
    /// Kernel size: 3, 5, 7, or 9 (must be odd).
    kernel_size: u32,

    /// Direction: vec2(1, 0) for horizontal, vec2(0, 1) for vertical.
    direction: vec2<f32>,

    /// Texel size in UV space (1.0 / texture_dimension).
    texel_size: vec2<f32>,
}

// ============================================================================
// Gaussian Kernel Weights
// ============================================================================

/// Returns Gaussian weights for a 3x3 kernel (sigma ~= 0.85).
fn gaussian_weights_3() -> array<f32, 3> {
    return array<f32, 3>(
        0.27901,  // center
        0.36049,  // adjacent
        0.27901   // edge (symmetric, reuse center)
    );
}

/// Returns Gaussian weights for a 5x5 kernel (sigma ~= 1.4).
fn gaussian_weights_5() -> array<f32, 5> {
    return array<f32, 5>(
        0.06136,
        0.24477,
        0.38774,  // center
        0.24477,
        0.06136
    );
}

/// Returns Gaussian weights for a 7x7 kernel (sigma ~= 2.0).
fn gaussian_weights_7() -> array<f32, 7> {
    return array<f32, 7>(
        0.00598,
        0.060626,
        0.241843,
        0.383103,  // center
        0.241843,
        0.060626,
        0.00598
    );
}

/// Returns Gaussian weights for a 9x9 kernel (sigma ~= 2.5).
fn gaussian_weights_9() -> array<f32, 9> {
    return array<f32, 9>(
        0.000229,
        0.005977,
        0.060598,
        0.241732,
        0.382928,  // center
        0.241732,
        0.060598,
        0.005977,
        0.000229
    );
}

// ============================================================================
// Core VSM Functions
// ============================================================================

/// Computes shadow factor using VSM Chebyshev inequality.
///
/// This is the main VSM sampling function. It reads depth moments from the
/// shadow map and uses Chebyshev's inequality to compute the probability
/// that the fragment is in shadow.
///
/// Parameters:
/// - shadow_map: RG format texture where R=mean depth, G=mean depth squared.
/// - shadow_sampler: Linear sampler for moment filtering (NOT comparison sampler).
/// - uv: Shadow map UV coordinates [0, 1].
/// - receiver_depth: Depth of the receiving surface in light space [0, 1].
/// - params: VSM configuration parameters.
///
/// Returns: Shadow factor in [0, 1] where 1 = fully lit, 0 = fully shadowed.
fn vsm_shadow(
    shadow_map: texture_2d<f32>,
    shadow_sampler: sampler,
    uv: vec2<f32>,
    receiver_depth: f32,
    params: VsmParams
) -> f32 {
    // Sample the depth moments from the shadow map.
    // R channel = E[z] = mean depth
    // G channel = E[z^2] = mean of squared depth
    let moments = textureSample(shadow_map, shadow_sampler, uv).rg;

    // Scale receiver depth to match stored moments.
    let scaled_depth = receiver_depth * params.depth_scale;

    // Extract mean (first moment) and compute variance.
    // Variance = E[z^2] - E[z]^2
    let mean = moments.x;
    let mean_sq = moments.y;
    let variance = max(mean_sq - mean * mean, params.min_variance);

    // Compute depth difference (positive when receiver is behind mean).
    let depth_diff = scaled_depth - mean;

    // If receiver is in front of or at the mean depth, it's fully lit.
    // This is the "definitely not in shadow" case.
    if depth_diff <= 0.0 {
        return 1.0;
    }

    // Chebyshev's inequality gives an upper bound on the probability
    // that the depth is greater than or equal to receiver_depth:
    //
    //   P(z >= t) <= variance / (variance + (t - mean)^2)
    //
    // This is p_max: the maximum probability that the receiver is lit.
    let d_sq = depth_diff * depth_diff;
    let p_max = variance / (variance + d_sq);

    // Apply light bleed reduction using linear rescaling.
    // This clamps low-probability shadows to zero, reducing the
    // characteristic "glowing" artifacts around shadow edges.
    //
    // Rescale: shadow = (p_max - threshold) / (1 - threshold)
    let threshold = params.bleed_threshold;
    let shadow_linear = clamp((p_max - threshold) / (1.0 - threshold), 0.0, 1.0);

    // Apply exponential bleed reduction for additional sharpening.
    // Higher exponents make shadows sharper but can cause banding.
    return pow(shadow_linear, params.bleed_reduction_exponent);
}

/// VSM shadow sampling with default parameters.
fn vsm_shadow_default(
    shadow_map: texture_2d<f32>,
    shadow_sampler: sampler,
    uv: vec2<f32>,
    receiver_depth: f32
) -> f32 {
    return vsm_shadow(shadow_map, shadow_sampler, uv, receiver_depth, vsm_params_default());
}

/// Computes VSM shadow with soft falloff for smoother transitions.
///
/// Applies additional smoothing to reduce harsh shadow boundaries.
/// Useful for low-resolution shadow maps or artistic soft shadows.
///
/// Parameters:
/// - Same as vsm_shadow, plus:
/// - softness: Softness factor [0, 1]. 0 = standard VSM, 1 = very soft.
fn vsm_shadow_soft(
    shadow_map: texture_2d<f32>,
    shadow_sampler: sampler,
    uv: vec2<f32>,
    receiver_depth: f32,
    params: VsmParams,
    softness: f32
) -> f32 {
    let moments = textureSample(shadow_map, shadow_sampler, uv).rg;
    let scaled_depth = receiver_depth * params.depth_scale;

    let mean = moments.x;
    let mean_sq = moments.y;

    // Increase minimum variance based on softness for smoother gradients.
    let soft_min_variance = params.min_variance + softness * 0.01;
    let variance = max(mean_sq - mean * mean, soft_min_variance);

    let depth_diff = scaled_depth - mean;

    if depth_diff <= 0.0 {
        return 1.0;
    }

    let d_sq = depth_diff * depth_diff;
    let p_max = variance / (variance + d_sq);

    // Reduce threshold for softer shadows (more light bleeding allowed).
    let soft_threshold = params.bleed_threshold * (1.0 - softness * 0.5);
    let shadow_linear = clamp((p_max - soft_threshold) / (1.0 - soft_threshold), 0.0, 1.0);

    // Reduce exponent for softer shadows.
    let soft_exponent = params.bleed_reduction_exponent * (1.0 - softness * 0.75);
    return pow(shadow_linear, max(soft_exponent, 1.0));
}

// ============================================================================
// Moment Generation (Shadow Map Pass)
// ============================================================================

/// Computes depth moments for VSM shadow map storage.
///
/// Call this in the shadow pass fragment shader to output moments
/// instead of raw depth. Output to RG16F or RG32F render target.
///
/// Parameters:
/// - depth: Linear depth value [0, 1].
/// - depth_scale: Scale factor for depth compression.
///
/// Returns: vec2(mean, mean_squared) for storage in RG channels.
fn vsm_compute_moments(depth: f32, depth_scale: f32) -> vec2<f32> {
    let scaled = depth * depth_scale;
    return vec2<f32>(scaled, scaled * scaled);
}

/// Computes moments with depth derivative bias for improved precision.
///
/// Uses depth derivatives to add variance along polygon edges,
/// reducing shadow acne on sloped surfaces.
///
/// Parameters:
/// - depth: Linear depth value [0, 1].
/// - depth_scale: Scale factor for depth compression.
/// - dx: Partial derivative of depth with respect to screen x.
/// - dy: Partial derivative of depth with respect to screen y.
fn vsm_compute_moments_biased(
    depth: f32,
    depth_scale: f32,
    dx: f32,
    dy: f32
) -> vec2<f32> {
    let scaled = depth * depth_scale;
    let moment1 = scaled;

    // Add variance based on depth derivatives (receiver plane).
    // This biases the variance to account for surface orientation.
    let derivative_bias = 0.25 * (dx * dx + dy * dy);
    let moment2 = scaled * scaled + derivative_bias;

    return vec2<f32>(moment1, moment2);
}

// ============================================================================
// Gaussian Blur Compute Shaders
// ============================================================================

/// Horizontal Gaussian blur pass for VSM pre-filtering.
///
/// This should run as a compute shader on the shadow map AFTER depth
/// rendering but BEFORE VSM sampling. Filtering in moment space
/// preserves the statistical properties needed for Chebyshev.
///
/// Workgroup size: 256x1x1 (processes one row per workgroup).
fn vsm_blur_horizontal_sample(
    input: texture_2d<f32>,
    coord: vec2<i32>,
    kernel_size: u32
) -> vec2<f32> {
    var sum = vec2<f32>(0.0);
    let half_kernel = i32(kernel_size / 2u);

    // Select appropriate weights based on kernel size.
    if kernel_size == 3u {
        let weights = gaussian_weights_3();
        for (var i = 0; i < 3; i++) {
            let offset = i - 1;
            let sample_coord = coord + vec2<i32>(offset, 0);
            let sample = textureLoad(input, sample_coord, 0).rg;
            sum += sample * weights[i];
        }
    } else if kernel_size == 5u {
        let weights = gaussian_weights_5();
        for (var i = 0; i < 5; i++) {
            let offset = i - 2;
            let sample_coord = coord + vec2<i32>(offset, 0);
            let sample = textureLoad(input, sample_coord, 0).rg;
            sum += sample * weights[i];
        }
    } else if kernel_size == 7u {
        let weights = gaussian_weights_7();
        for (var i = 0; i < 7; i++) {
            let offset = i - 3;
            let sample_coord = coord + vec2<i32>(offset, 0);
            let sample = textureLoad(input, sample_coord, 0).rg;
            sum += sample * weights[i];
        }
    } else {
        // Default to 9x9.
        let weights = gaussian_weights_9();
        for (var i = 0; i < 9; i++) {
            let offset = i - 4;
            let sample_coord = coord + vec2<i32>(offset, 0);
            let sample = textureLoad(input, sample_coord, 0).rg;
            sum += sample * weights[i];
        }
    }

    return sum;
}

/// Vertical Gaussian blur pass for VSM pre-filtering.
///
/// Run after horizontal pass to complete separable Gaussian blur.
/// Workgroup size: 1x256x1 (processes one column per workgroup).
fn vsm_blur_vertical_sample(
    input: texture_2d<f32>,
    coord: vec2<i32>,
    kernel_size: u32
) -> vec2<f32> {
    var sum = vec2<f32>(0.0);

    if kernel_size == 3u {
        let weights = gaussian_weights_3();
        for (var i = 0; i < 3; i++) {
            let offset = i - 1;
            let sample_coord = coord + vec2<i32>(0, offset);
            let sample = textureLoad(input, sample_coord, 0).rg;
            sum += sample * weights[i];
        }
    } else if kernel_size == 5u {
        let weights = gaussian_weights_5();
        for (var i = 0; i < 5; i++) {
            let offset = i - 2;
            let sample_coord = coord + vec2<i32>(0, offset);
            let sample = textureLoad(input, sample_coord, 0).rg;
            sum += sample * weights[i];
        }
    } else if kernel_size == 7u {
        let weights = gaussian_weights_7();
        for (var i = 0; i < 7; i++) {
            let offset = i - 3;
            let sample_coord = coord + vec2<i32>(0, offset);
            let sample = textureLoad(input, sample_coord, 0).rg;
            sum += sample * weights[i];
        }
    } else {
        let weights = gaussian_weights_9();
        for (var i = 0; i < 9; i++) {
            let offset = i - 4;
            let sample_coord = coord + vec2<i32>(0, offset);
            let sample = textureLoad(input, sample_coord, 0).rg;
            sum += sample * weights[i];
        }
    }

    return sum;
}

// ============================================================================
// Compute Shader Entry Points
// ============================================================================

// Bindings for blur compute shaders.
@group(0) @binding(0) var vsm_input: texture_2d<f32>;
@group(0) @binding(1) var vsm_output: texture_storage_2d<rg32float, write>;

struct VsmBlurUniforms {
    kernel_size: u32,
    direction: u32, // 0 = horizontal, 1 = vertical
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(2) var<uniform> blur_uniforms: VsmBlurUniforms;

/// Horizontal blur compute shader.
/// Dispatched with ceil(width / 8) x ceil(height / 8) x 1 workgroups.
@compute @workgroup_size(8, 8, 1)
fn vsm_blur_horizontal_cs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(vsm_input);
    let coord = vec2<i32>(global_id.xy);

    // Bounds check.
    if coord.x >= i32(dims.x) || coord.y >= i32(dims.y) {
        return;
    }

    let result = vsm_blur_horizontal_sample(vsm_input, coord, blur_uniforms.kernel_size);
    textureStore(vsm_output, vec2<u32>(coord), vec4<f32>(result, 0.0, 1.0));
}

/// Vertical blur compute shader.
/// Dispatched with ceil(width / 8) x ceil(height / 8) x 1 workgroups.
@compute @workgroup_size(8, 8, 1)
fn vsm_blur_vertical_cs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(vsm_input);
    let coord = vec2<i32>(global_id.xy);

    if coord.x >= i32(dims.x) || coord.y >= i32(dims.y) {
        return;
    }

    let result = vsm_blur_vertical_sample(vsm_input, coord, blur_uniforms.kernel_size);
    textureStore(vsm_output, vec2<u32>(coord), vec4<f32>(result, 0.0, 1.0));
}

/// Combined bidirectional blur (single pass, less efficient but simpler).
/// Uses separable property internally but in one dispatch.
@compute @workgroup_size(8, 8, 1)
fn vsm_blur_bidirectional_cs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(vsm_input);
    let coord = vec2<i32>(global_id.xy);

    if coord.x >= i32(dims.x) || coord.y >= i32(dims.y) {
        return;
    }

    // For single-pass blur, we use a smaller kernel to maintain performance.
    // This is a 2D Gaussian (not separable) - less efficient but single dispatch.
    var sum = vec2<f32>(0.0);
    var weight_sum: f32 = 0.0;

    let kernel_size = min(blur_uniforms.kernel_size, 5u);
    let half_kernel = i32(kernel_size / 2u);

    // 2D Gaussian weights (approximation for small kernels).
    for (var y = -half_kernel; y <= half_kernel; y++) {
        for (var x = -half_kernel; x <= half_kernel; x++) {
            let sample_coord = coord + vec2<i32>(x, y);

            // Skip out-of-bounds samples.
            if sample_coord.x < 0 || sample_coord.x >= i32(dims.x) ||
               sample_coord.y < 0 || sample_coord.y >= i32(dims.y) {
                continue;
            }

            // Gaussian weight: exp(-(x^2 + y^2) / (2 * sigma^2)).
            // Using sigma = kernel_size / 4 for reasonable falloff.
            let sigma = f32(kernel_size) * 0.25;
            let dist_sq = f32(x * x + y * y);
            let weight = exp(-dist_sq / (2.0 * sigma * sigma));

            let sample = textureLoad(vsm_input, sample_coord, 0).rg;
            sum += sample * weight;
            weight_sum += weight;
        }
    }

    let result = sum / max(weight_sum, 0.0001);
    textureStore(vsm_output, vec2<u32>(coord), vec4<f32>(result, 0.0, 1.0));
}

// ============================================================================
// Integration Helpers
// ============================================================================

/// Converts shadow map UV and computes VSM shadow factor.
///
/// Convenience function that handles the full VSM pipeline for a single
/// shadow sample. Use this in fragment shaders.
///
/// Parameters:
/// - world_pos: World-space position of the fragment.
/// - light_view_proj: Light's view-projection matrix.
/// - shadow_map: Pre-filtered VSM shadow map (RG format).
/// - shadow_sampler: Linear sampler.
/// - params: VSM parameters.
///
/// Returns: Shadow factor in [0, 1].
fn vsm_shadow_world(
    world_pos: vec3<f32>,
    light_view_proj: mat4x4<f32>,
    shadow_map: texture_2d<f32>,
    shadow_sampler: sampler,
    params: VsmParams
) -> f32 {
    // Transform world position to light clip space.
    let light_clip = light_view_proj * vec4<f32>(world_pos, 1.0);

    // Perspective divide.
    if light_clip.w <= 0.0 {
        return 1.0; // Behind the light, fully lit.
    }

    let ndc = light_clip.xyz / light_clip.w;

    // Check bounds.
    if any(abs(ndc.xy) > vec2<f32>(1.0)) || ndc.z < 0.0 || ndc.z > 1.0 {
        return 1.0; // Outside shadow frustum, fully lit.
    }

    // Convert NDC to UV space.
    let uv = ndc.xy * 0.5 + 0.5;

    // Sample VSM.
    return vsm_shadow(shadow_map, shadow_sampler, uv, ndc.z, params);
}

/// VSM shadow sampling for texture arrays (multiple shadow maps).
fn vsm_shadow_array(
    shadow_map: texture_2d_array<f32>,
    shadow_sampler: sampler,
    uv: vec2<f32>,
    layer: i32,
    receiver_depth: f32,
    params: VsmParams
) -> f32 {
    let moments = textureSampleLevel(shadow_map, shadow_sampler, uv, layer, 0.0).rg;
    let scaled_depth = receiver_depth * params.depth_scale;

    let mean = moments.x;
    let mean_sq = moments.y;
    let variance = max(mean_sq - mean * mean, params.min_variance);

    let depth_diff = scaled_depth - mean;

    if depth_diff <= 0.0 {
        return 1.0;
    }

    let d_sq = depth_diff * depth_diff;
    let p_max = variance / (variance + d_sq);

    let threshold = params.bleed_threshold;
    let shadow_linear = clamp((p_max - threshold) / (1.0 - threshold), 0.0, 1.0);

    return pow(shadow_linear, params.bleed_reduction_exponent);
}

// ============================================================================
// Debug Visualization
// ============================================================================

/// Returns debug color based on VSM internal values.
///
/// Useful for diagnosing shadow quality issues:
/// - Red channel: Mean depth.
/// - Green channel: Variance (scaled for visibility).
/// - Blue channel: Raw Chebyshev probability.
fn vsm_debug_color(
    shadow_map: texture_2d<f32>,
    shadow_sampler: sampler,
    uv: vec2<f32>,
    receiver_depth: f32
) -> vec3<f32> {
    let moments = textureSample(shadow_map, shadow_sampler, uv).rg;

    let mean = moments.x;
    let mean_sq = moments.y;
    let variance = max(mean_sq - mean * mean, VSM_MIN_VARIANCE);

    let depth_diff = receiver_depth - mean;
    var p_max: f32 = 1.0;

    if depth_diff > 0.0 {
        let d_sq = depth_diff * depth_diff;
        p_max = variance / (variance + d_sq);
    }

    // Scale variance for visibility (typical variance is very small).
    let scaled_variance = saturate(variance * 1000.0);

    return vec3<f32>(mean, scaled_variance, p_max);
}
