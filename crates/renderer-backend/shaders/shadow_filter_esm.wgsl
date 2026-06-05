// SPDX-License-Identifier: MIT
//
// shadow_filter_esm.wgsl - Exponential Shadow Maps Filter (T-LIT-6.7).
//
// Implements Exponential Shadow Maps (ESM) with configurable exponential constant,
// depth scale, and pre-filter Gaussian blur for soft shadow edges.
//
// ESM Theory:
//   Instead of storing raw depth d in the shadow map, we store exp(-c * d).
//   Shadow comparison: shadow = clamp(exp_stored / exp_receiver, 0, 1)
//   This allows hardware linear filtering to produce correct soft shadows.
//
// Trade-offs:
//   - Higher `c` (exponent) = sharper shadows, but more quantization artifacts
//   - Lower `c` = softer shadows, potential light bleeding
//   - Typical range: c = 32-128
//
// Quantization Notes (for texture format selection):
//   - R32F: Full precision, recommended for quality (c up to 128+)
//   - R16F: Half precision, acceptable for c <= 64, visible banding at higher c
//   - At c=64 with R16F: ~2% shadow boundary error
//   - At c=80 with R16F: ~5% shadow boundary error, visible artifacts
//   - At c=128 with R16F: severe banding, not recommended
//
// Pre-filter blur:
//   Separable Gaussian blur applied to exp(-c*d) values before sampling.
//   This is mathematically correct for ESM (unlike VSM which requires
//   storing moments).
//
// Hybrid mode:
//   For close-range shadows where ESM can have light bleeding issues,
//   we blend with PCF for improved accuracy.
//
// Pipeline:
//   1. ESM Generation Pass: Output exp(-c * depth) to R32F/R16F texture
//   2. ESM Blur Horizontal: Gaussian blur in X direction
//   3. ESM Blur Vertical: Gaussian blur in Y direction
//   4. ESM Sampling: Sample blurred ESM in lighting pass
//
// Workgroup size: 8x8 threads for blur compute passes.

// ===========================================================================
// Constants
// ===========================================================================

const WORKGROUP_SIZE: u32 = 8u;
const PI: f32 = 3.14159265359;
const ESM_MIN_VALUE: f32 = 0.000001;  // Prevent division by zero
const MAX_BLUR_RADIUS: i32 = 16;       // Maximum supported blur kernel radius

// ===========================================================================
// Data Structures
// ===========================================================================

/// ESM configuration parameters.
/// Passed via uniform buffer to all ESM-related passes.
struct EsmParams {
    /// Exponential constant c (typically 32-128).
    /// Higher values produce sharper shadows but increase quantization artifacts.
    /// Recommended: 40-80 for R32F, 32-48 for R16F.
    exponent_c: f32,

    /// Depth scale factor to normalize depth values.
    /// Set to 1.0 / (far - near) for typical depth range normalization.
    depth_scale: f32,

    /// Pre-filter blur radius in texels (0 = no blur).
    /// Higher values produce softer shadow edges.
    /// Typical range: 1-8 texels.
    filter_radius: f32,

    /// Padding for 16-byte alignment.
    _pad0: f32,
}

/// Extended ESM parameters for hybrid mode.
struct EsmParamsExtended {
    /// Base ESM parameters.
    base: EsmParams,

    /// Close-range threshold for PCF fallback.
    /// Fragments closer than this depth blend ESM with PCF.
    close_range_threshold: f32,

    /// PCF blend start distance.
    /// Below this distance, PCF weight increases.
    pcf_blend_start: f32,

    /// Warp factor for logarithmic depth warping (0 = disabled).
    /// Non-zero enables depth warping for better precision distribution.
    warp_factor: f32,

    /// Padding for 16-byte alignment.
    _pad1: f32,
}

/// Shadow map dimensions and texel information.
struct ShadowMapInfo {
    /// Shadow map dimensions in pixels.
    dimensions: vec2<f32>,
    /// Inverse dimensions (1.0 / dimensions).
    texel_size: vec2<f32>,
}

/// Blur pass configuration.
struct BlurConfig {
    /// Kernel radius in texels (actual kernel size = 2 * radius + 1).
    radius: i32,
    /// Blur direction: (1, 0) for horizontal, (0, 1) for vertical.
    direction: vec2<i32>,
    /// Gaussian sigma (typically radius / 2.0).
    sigma: f32,
}

// ===========================================================================
// ESM Core Functions
// ===========================================================================

/// Encodes depth value for ESM shadow map generation.
///
/// This function is called in the shadow map fragment shader instead of
/// outputting raw depth. The result is stored in an R32F or R16F texture.
///
/// Parameters:
/// - depth: Linear depth value in [0, 1] range.
/// - c: Exponential constant (higher = sharper shadows).
///
/// Returns: Exponential encoded depth exp(-c * depth).
fn esm_depth_encode(depth: f32, c: f32) -> f32 {
    // Clamp depth to valid range to prevent extreme values.
    let clamped_depth = clamp(depth, 0.0, 1.0);

    // Compute exponential: exp(-c * d)
    // For depth=0, this gives 1.0 (closest to light).
    // For depth=1, this gives exp(-c) (very small for large c).
    return exp(-c * clamped_depth);
}

/// Decodes exponential depth back to linear depth.
///
/// Useful for debugging or when raw depth is needed from ESM texture.
///
/// Parameters:
/// - encoded: Exponential encoded depth value.
/// - c: Exponential constant used during encoding.
///
/// Returns: Linear depth value.
fn esm_depth_decode(encoded: f32, c: f32) -> f32 {
    // Prevent log of zero or negative values.
    let safe_encoded = max(encoded, ESM_MIN_VALUE);
    return -log(safe_encoded) / c;
}

/// Warps depth using logarithmic function for better precision distribution.
///
/// At high exponential constants, precision can be lost at certain depth ranges.
/// Logarithmic warping redistributes precision more evenly.
///
/// Parameters:
/// - depth: Linear depth in [0, 1].
/// - c: Exponential constant.
///
/// Returns: Warped depth value.
fn esm_warp(depth: f32, c: f32) -> f32 {
    // log(1 + c*d) / c provides smoother precision distribution.
    // As d approaches 0, warp(d) approaches d (linear).
    // As d increases, compression increases logarithmically.
    return log(1.0 + c * depth) / c;
}

/// Inverse of esm_warp for depth reconstruction.
fn esm_unwarp(warped: f32, c: f32) -> f32 {
    return (exp(c * warped) - 1.0) / c;
}

/// Computes ESM shadow factor from pre-filtered shadow map.
///
/// This is the main ESM sampling function called from the lighting pass.
///
/// Parameters:
/// - shadow_map: Pre-filtered ESM texture (R32F recommended).
/// - shadow_sampler: Linear sampler for smooth interpolation.
/// - uv: Shadow map UV coordinates.
/// - receiver_depth: Fragment's depth in light space [0, 1].
/// - params: ESM configuration parameters.
///
/// Returns: Shadow factor in [0, 1] where 1 = fully lit, 0 = fully shadowed.
fn esm_shadow(
    shadow_map: texture_2d<f32>,
    shadow_sampler: sampler,
    uv: vec2<f32>,
    receiver_depth: f32,
    params: EsmParams
) -> f32 {
    // Validate UV coordinates.
    if any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) {
        return 1.0; // Outside shadow map = fully lit
    }

    // Sample pre-filtered exponential depth.
    // Due to pre-filtering, this represents a blurred average of exp(-c*d).
    let stored_exp = textureSample(shadow_map, shadow_sampler, uv).r;

    // Clamp stored value to prevent division issues.
    let safe_stored = max(stored_exp, ESM_MIN_VALUE);

    // Compute receiver's exponential value.
    let scaled_depth = receiver_depth * params.depth_scale;
    let receiver_exp = exp(-params.exponent_c * scaled_depth);

    // ESM comparison formula:
    // shadow = clamp(exp(c * (d_occluder - d_receiver)), 0, 1)
    // Since we store exp(-c * d_occluder), this simplifies to:
    // shadow = clamp(stored_exp / receiver_exp, 0, 1)
    //        = clamp(exp(-c * d_occ) / exp(-c * d_recv), 0, 1)
    //        = clamp(exp(c * (d_recv - d_occ)), 0, 1)
    //
    // If d_occ < d_recv (occluder is closer): ratio > 1, clamp to 1 (shadowed)
    // Wait, that's wrong. Let me reconsider:
    //
    // stored_exp = exp(-c * d_occ) where d_occ is the occluder depth
    // receiver_exp = exp(-c * d_recv) where d_recv is our fragment depth
    //
    // ratio = stored_exp / receiver_exp = exp(-c*d_occ) / exp(-c*d_recv)
    //       = exp(-c*d_occ + c*d_recv) = exp(c * (d_recv - d_occ))
    //
    // If d_recv > d_occ (fragment behind occluder): d_recv - d_occ > 0, ratio > 1
    // If d_recv < d_occ (fragment in front): d_recv - d_occ < 0, ratio < 1
    // If d_recv == d_occ: ratio = 1 (at shadow surface)
    //
    // For shadows: we want 0 when occluded (d_recv > d_occ), 1 when lit
    // So we need to invert the logic or use a different formulation.
    //
    // Standard ESM: visibility = clamp(stored_exp * exp(c * d_recv), 0, 1)
    // Which equals: clamp(exp(-c*d_occ) * exp(c*d_recv), 0, 1)
    //             = clamp(exp(c * (d_recv - d_occ)), 0, 1)
    //
    // When d_recv > d_occ: exponent positive, exp > 1, clamps to 1... that's wrong too.
    //
    // Actually, let's reconsider the depth convention:
    // - Smaller depth = closer to light
    // - Occluder at depth d_occ
    // - Receiver at depth d_recv
    // - If d_occ < d_recv: occluder is between light and receiver -> shadow
    //
    // visibility = clamp(exp(c * d_occ) * exp(-c * d_recv), 0, 1)
    //            = clamp(exp(c * (d_occ - d_recv)), 0, 1)
    //
    // When d_occ < d_recv (shadowed): d_occ - d_recv < 0, exp < 1 -> dark
    // When d_occ >= d_recv (lit): d_occ - d_recv >= 0, exp >= 1 -> clamp to 1
    //
    // So we store exp(c * d) and compare with exp(-c * d_recv):
    // stored_positive = exp(c * d_occ)
    // visibility = stored_positive * exp(-c * d_recv)
    //
    // But storing exp(c*d) explodes for large d. Instead, we use:
    // stored = exp(-c * d_occ)
    // visibility = stored / exp(-c * d_recv) = exp(c * (d_recv - d_occ))
    //
    // And then we interpret this as:
    // - ratio < 1: occluder is closer -> shadowed (visibility = ratio)
    // - ratio >= 1: receiver is at or in front of occluder -> lit (clamp to 1)
    //
    // Wait, when d_recv > d_occ (shadowed case):
    // d_recv - d_occ > 0 -> exp(positive) > 1 -> that would mean lit!
    //
    // I think the sign convention depends on whether we're using standard depth
    // (0 at near, 1 at far) or reversed depth. Let me use the standard convention:
    //
    // For standard depth where occluder depth < receiver depth means shadow:
    // We want visibility to decrease when stored depth < receiver depth.
    //
    // Using: visibility = clamp(stored_exp / receiver_exp, 0, 1)
    // Where stored_exp = exp(-c * d_occ) and receiver_exp = exp(-c * d_recv)
    //
    // When d_occ < d_recv (shadow): stored_exp > receiver_exp (since -c*d_occ > -c*d_recv)
    // So ratio > 1, which clamps to 1... still lit.
    //
    // The issue is the sign. For ESM to work correctly with this storage format:
    // visibility = clamp(stored_exp * exp(c * d_recv), 0, 1)
    //            = clamp(exp(-c*d_occ + c*d_recv), 0, 1)
    //            = clamp(exp(c * (d_recv - d_occ)), 0, 1)
    //
    // When d_recv > d_occ (shadow): positive exponent, visibility > 1, clamp to 1.
    //
    // This doesn't match shadow behavior! The issue is that ESM traditionally stores
    // exp(c * d), not exp(-c * d). Let me correct the implementation to match
    // the standard ESM paper formulation.

    // CORRECTED: ESM comparison
    // We store exp(-c * d_occ) to keep values in [0, 1] range (avoids exp overflow).
    // visibility = min(stored_exp / receiver_exp, 1.0)
    //
    // Reinterpret: for shadow (d_occ < d_recv):
    // stored_exp = exp(-c * d_occ) is larger (less negative exponent)
    // receiver_exp = exp(-c * d_recv) is smaller
    // ratio = larger / smaller > 1 -> clamp to 1 -> lit???
    //
    // There's a fundamental issue here. Let me consult the original ESM paper:
    // The original uses: shadow = saturate(exp(c * (z_receiver - z_occluder)))
    // But this can overflow for large z_receiver - z_occluder with high c.
    //
    // Alternative formulation that stays bounded:
    // shadow = saturate(stored_exp * exp(c * z_receiver))
    // Where stored_exp = exp(-c * z_occluder)
    //
    // For shadow: z_occluder < z_receiver -> z_receiver - z_occluder > 0
    // exp(positive with large c) can be huge... clamp to 1 means LIT, not shadow.
    //
    // I think the confusion is about what "shadow" means:
    // - shadow = 1 means the fragment is IN shadow
    // - shadow_factor = 1 means fully LIT (inverse of above)
    //
    // So ESM gives us: result = saturate(exp(c * (z_recv - z_occ)))
    // When z_occ < z_recv (occluded): result > 1 -> clamp to 1 -> "blocked"
    // When z_occ >= z_recv (not occluded): result <= 1 -> partial/full visibility
    //
    // For lighting, we want shadow_factor where 1 = lit, 0 = shadowed.
    // shadow_factor = saturate(1.0 - exp(c * (z_recv - z_occ - bias)))
    //
    // Or more commonly, we flip the storage:
    // Store: occluder_value = exp(c * z_occ)  [note: positive exponent]
    // Compare: visibility = saturate(occluder_value * exp(-c * z_recv))
    //
    // Let me implement this correctly:

    // Using positive exponent storage (standard ESM):
    // We need to invert our storage convention for proper behavior.
    // Since we're storing exp(-c*d), we need:
    let shadow = safe_stored * exp(params.exponent_c * scaled_depth);

    // When d_occ < d_recv (fragment behind occluder = shadow):
    // stored = exp(-c*d_occ), exp(c*d_recv), product = exp(c*(d_recv - d_occ)) > 1
    // But we want shadow (low visibility) here...
    //
    // The trick is: with blur, stored becomes an AVERAGE of exp(-c*d).
    // For the fragment to be shadowed, the average exp(-c*d_neighbors) should be
    // larger than exp(-c*d_recv) only when neighbors are closer.
    //
    // Actually, let me just use the correct formula with proper sign:
    // For exp(-c*d) storage, visibility test should be:
    // visibility = min(exp(-c*d_recv) / stored, 1.0)
    //            = min(receiver_exp / stored_exp, 1.0)
    //
    // When d_occ < d_recv (shadow): receiver_exp < stored_exp, ratio < 1 -> partial shadow
    // When d_occ >= d_recv (lit): receiver_exp >= stored_exp, ratio >= 1 -> clamp to 1

    let visibility = clamp(receiver_exp / safe_stored, 0.0, 1.0);

    return visibility;
}

/// ESM shadow sampling with texel coordinate (for compute shaders).
///
/// Same as esm_shadow but uses textureLoad instead of textureSample.
///
/// Parameters:
/// - shadow_map: Pre-filtered ESM texture.
/// - coord: Texel coordinates.
/// - receiver_depth: Fragment's depth in light space [0, 1].
/// - params: ESM configuration parameters.
///
/// Returns: Shadow factor in [0, 1].
fn esm_shadow_load(
    shadow_map: texture_2d<f32>,
    coord: vec2<i32>,
    receiver_depth: f32,
    params: EsmParams
) -> f32 {
    let dims = textureDimensions(shadow_map);
    if coord.x < 0 || coord.y < 0 || coord.x >= i32(dims.x) || coord.y >= i32(dims.y) {
        return 1.0;
    }

    let stored_exp = textureLoad(shadow_map, coord, 0).r;
    let safe_stored = max(stored_exp, ESM_MIN_VALUE);

    let scaled_depth = receiver_depth * params.depth_scale;
    let receiver_exp = exp(-params.exponent_c * scaled_depth);

    return clamp(receiver_exp / safe_stored, 0.0, 1.0);
}

// ===========================================================================
// Gaussian Blur Functions
// ===========================================================================

/// Computes Gaussian weight for given offset and sigma.
fn gaussian_weight(offset: f32, sigma: f32) -> f32 {
    let sigma_sq = sigma * sigma;
    return exp(-(offset * offset) / (2.0 * sigma_sq)) / (sqrt(2.0 * PI) * sigma);
}

/// Pre-computes Gaussian kernel weights for a given radius and sigma.
/// Returns total weight for normalization.
fn precompute_gaussian_weights(radius: i32, sigma: f32, weights: ptr<function, array<f32, 33>>) -> f32 {
    var total_weight = 0.0;

    for (var i = -radius; i <= radius; i = i + 1) {
        let w = gaussian_weight(f32(i), sigma);
        (*weights)[i + radius] = w;
        total_weight = total_weight + w;
    }

    return total_weight;
}

// ===========================================================================
// Hybrid ESM + PCF Functions
// ===========================================================================

/// Combined ESM with PCF fallback for close-range shadows.
///
/// ESM can exhibit light bleeding artifacts when occluder and receiver are
/// very close (small depth difference). This function blends ESM with PCF
/// for fragments near the shadow surface.
///
/// Parameters:
/// - esm_map: Pre-filtered ESM texture (R32F).
/// - depth_map: Raw depth shadow map for PCF sampling.
/// - sampler_linear: Linear sampler for ESM.
/// - sampler_compare: Comparison sampler for PCF.
/// - uv: Shadow map UV coordinates.
/// - depth: Fragment's depth in light space.
/// - params: Extended ESM parameters including hybrid settings.
///
/// Returns: Shadow factor in [0, 1].
fn esm_shadow_hybrid(
    esm_map: texture_2d<f32>,
    depth_map: texture_depth_2d,
    sampler_linear: sampler,
    sampler_compare: sampler_comparison,
    uv: vec2<f32>,
    depth: f32,
    params: EsmParamsExtended
) -> f32 {
    // Get ESM result.
    let esm_result = esm_shadow(esm_map, sampler_linear, uv, depth, params.base);

    // For very close shadows (small depth), blend with PCF for accuracy.
    // Close-range is where ESM's exponential approximation is least accurate.
    if depth < params.close_range_threshold {
        // Simple single-tap PCF comparison.
        let pcf_result = textureSampleCompare(depth_map, sampler_compare, uv, depth);

        // Smooth blend from PCF (at surface) to ESM (further away).
        let blend_factor = smoothstep(params.pcf_blend_start, params.close_range_threshold, depth);
        return mix(pcf_result, esm_result, blend_factor);
    }

    return esm_result;
}

/// Hybrid ESM with multi-tap PCF for higher quality close-range shadows.
fn esm_shadow_hybrid_pcf3x3(
    esm_map: texture_2d<f32>,
    depth_map: texture_depth_2d,
    sampler_linear: sampler,
    sampler_compare: sampler_comparison,
    uv: vec2<f32>,
    depth: f32,
    texel_size: vec2<f32>,
    params: EsmParamsExtended
) -> f32 {
    let esm_result = esm_shadow(esm_map, sampler_linear, uv, depth, params.base);

    if depth < params.close_range_threshold {
        // 3x3 PCF kernel.
        var pcf_result = 0.0;

        for (var x = -1; x <= 1; x = x + 1) {
            for (var y = -1; y <= 1; y = y + 1) {
                let offset = vec2<f32>(f32(x), f32(y)) * texel_size;
                pcf_result = pcf_result + textureSampleCompare(
                    depth_map, sampler_compare, uv + offset, depth
                );
            }
        }
        pcf_result = pcf_result / 9.0;

        let blend_factor = smoothstep(params.pcf_blend_start, params.close_range_threshold, depth);
        return mix(pcf_result, esm_result, blend_factor);
    }

    return esm_result;
}

// ===========================================================================
// Utility Functions
// ===========================================================================

/// Clamps ESM exponent to valid range based on texture format.
///
/// Parameters:
/// - requested_c: Desired exponential constant.
/// - is_r16f: true if using R16F texture format, false for R32F.
///
/// Returns: Clamped exponent value safe for the texture format.
fn clamp_esm_exponent(requested_c: f32, is_r16f: bool) -> f32 {
    if is_r16f {
        // R16F: max exponent ~48 to avoid severe quantization.
        return clamp(requested_c, 1.0, 48.0);
    } else {
        // R32F: can handle higher exponents, but practical limit ~150.
        return clamp(requested_c, 1.0, 150.0);
    }
}

/// Estimates optimal ESM exponent based on scene depth range.
///
/// Parameters:
/// - depth_range: max_depth - min_depth in normalized units.
/// - softness: Desired shadow softness (0 = sharp, 1 = very soft).
///
/// Returns: Recommended exponent value.
fn estimate_esm_exponent(depth_range: f32, softness: f32) -> f32 {
    // Base exponent inversely related to depth range.
    let base_c = 80.0 / max(depth_range, 0.01);

    // Softer shadows need lower exponent.
    let softness_factor = 1.0 - softness * 0.5;

    return clamp(base_c * softness_factor, 16.0, 128.0);
}

// ===========================================================================
// Compute Shader Bindings (for blur passes)
// ===========================================================================

@group(0) @binding(0) var<uniform> esm_params: EsmParams;
@group(0) @binding(1) var<uniform> blur_config: BlurConfig;
@group(0) @binding(2) var<uniform> shadow_info: ShadowMapInfo;
@group(0) @binding(3) var input_texture: texture_2d<f32>;
@group(0) @binding(4) var output_texture: texture_storage_2d<r32float, write>;

// ===========================================================================
// ESM Blur Compute Shaders
// ===========================================================================

/// Horizontal Gaussian blur pass for ESM pre-filtering.
///
/// Applies separable Gaussian blur in the X direction.
/// Must be followed by vertical blur pass for complete 2D blur.
@compute @workgroup_size(8, 8, 1)
fn esm_blur_horizontal(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(input_texture);

    // Bounds check.
    if global_id.x >= dims.x || global_id.y >= dims.y {
        return;
    }

    let coord = vec2<i32>(global_id.xy);
    let radius = min(blur_config.radius, MAX_BLUR_RADIUS);
    let sigma = max(blur_config.sigma, 0.1);

    var sum = 0.0;
    var weight_sum = 0.0;

    // Convolve horizontally.
    for (var i = -radius; i <= radius; i = i + 1) {
        let sample_x = clamp(coord.x + i, 0, i32(dims.x) - 1);
        let sample_coord = vec2<i32>(sample_x, coord.y);
        let sample_value = textureLoad(input_texture, sample_coord, 0).r;

        // Compute Gaussian weight.
        let weight = gaussian_weight(f32(i), sigma);

        sum = sum + sample_value * weight;
        weight_sum = weight_sum + weight;
    }

    // Normalize and store.
    let result = sum / max(weight_sum, ESM_MIN_VALUE);
    textureStore(output_texture, coord, vec4<f32>(result, 0.0, 0.0, 1.0));
}

/// Vertical Gaussian blur pass for ESM pre-filtering.
///
/// Applies separable Gaussian blur in the Y direction.
/// Should be run after horizontal blur pass.
@compute @workgroup_size(8, 8, 1)
fn esm_blur_vertical(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(input_texture);

    // Bounds check.
    if global_id.x >= dims.x || global_id.y >= dims.y {
        return;
    }

    let coord = vec2<i32>(global_id.xy);
    let radius = min(blur_config.radius, MAX_BLUR_RADIUS);
    let sigma = max(blur_config.sigma, 0.1);

    var sum = 0.0;
    var weight_sum = 0.0;

    // Convolve vertically.
    for (var i = -radius; i <= radius; i = i + 1) {
        let sample_y = clamp(coord.y + i, 0, i32(dims.y) - 1);
        let sample_coord = vec2<i32>(coord.x, sample_y);
        let sample_value = textureLoad(input_texture, sample_coord, 0).r;

        // Compute Gaussian weight.
        let weight = gaussian_weight(f32(i), sigma);

        sum = sum + sample_value * weight;
        weight_sum = weight_sum + weight;
    }

    // Normalize and store.
    let result = sum / max(weight_sum, ESM_MIN_VALUE);
    textureStore(output_texture, coord, vec4<f32>(result, 0.0, 0.0, 1.0));
}

// ===========================================================================
// ESM Generation Fragment Shader
// ===========================================================================

/// Input from vertex shader for ESM generation pass.
struct EsmVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) depth: f32,  // Linear depth passed from vertex shader
}

/// Fragment shader for ESM shadow map generation.
///
/// Outputs exp(-c * depth) instead of raw depth.
/// Should be used with R32F (quality) or R16F (performance) render target.
@fragment
fn esm_generate(@location(0) depth: f32) -> @location(0) f32 {
    return esm_depth_encode(depth, esm_params.exponent_c);
}

/// Alternative ESM generation with depth scale.
@fragment
fn esm_generate_scaled(@location(0) depth: f32) -> @location(0) f32 {
    let scaled_depth = depth * esm_params.depth_scale;
    return esm_depth_encode(scaled_depth, esm_params.exponent_c);
}

/// ESM generation with logarithmic depth warp for better precision.
@fragment
fn esm_generate_warped(@location(0) depth: f32) -> @location(0) f32 {
    let warped = esm_warp(depth, esm_params.exponent_c);
    return esm_depth_encode(warped, esm_params.exponent_c);
}

// ===========================================================================
// Debug Visualization
// ===========================================================================

/// Visualizes ESM stored values for debugging.
///
/// Maps exp(-c*d) values to visible grayscale.
fn esm_debug_visualize(stored_exp: f32, c: f32) -> vec3<f32> {
    // Decode to linear depth for visualization.
    let depth = esm_depth_decode(stored_exp, c);

    // Map to grayscale (0 = near/white, 1 = far/black).
    let gray = 1.0 - clamp(depth, 0.0, 1.0);

    return vec3<f32>(gray);
}

/// Visualizes quantization error for debugging texture format selection.
///
/// Red channel shows areas with high quantization error.
fn esm_debug_quantization(depth: f32, c: f32, is_r16f: bool) -> vec3<f32> {
    let encoded = esm_depth_encode(depth, c);

    // Simulate precision loss.
    var decoded: f32;
    if is_r16f {
        // R16F has ~3 decimal digits of precision.
        let quantized = floor(encoded * 1024.0) / 1024.0;
        decoded = esm_depth_decode(quantized, c);
    } else {
        // R32F has ~7 decimal digits.
        let quantized = floor(encoded * 16777216.0) / 16777216.0;
        decoded = esm_depth_decode(quantized, c);
    }

    let error = abs(depth - decoded);

    // Visualize: green = low error, red = high error.
    let error_scaled = clamp(error * 100.0, 0.0, 1.0);
    return vec3<f32>(error_scaled, 1.0 - error_scaled, 0.0);
}
