// Subsurface Scattering Functions for TRINITY Material System
// T-MAT-4.1: Dual-Pass Screen-Space SSS Implementation
//
// This module provides subsurface scattering BRDF functions for PBR rendering:
//   - Burley normalized diffusion profile
//   - Separable 9-tap Gaussian blur for screen-space diffusion
//   - Pre-integrated diffusion LUT support
//   - Dual-pass screen-space SSS pipeline
//
// The SSS model simulates light transport within translucent materials:
//   - Human skin
//   - Wax/candles
//   - Jade/marble
//   - Milk/liquids
//
// References:
//   - Burley "Extending the Disney BRDF to a BSDF with Integrated Subsurface Scattering"
//   - Jimenez et al. "Separable Subsurface Scattering"
//   - d'Eon & Luebke "Advanced Techniques for Realistic Real-Time Skin Rendering"
//
// Pipeline:
//   Pass 1: Render diffuse to SSS target (gbuffer with SSS mask)
//   Pass 2: Horizontal separable blur with diffusion kernel
//   Pass 3: Vertical separable blur with diffusion kernel
//   Pass 4: Final composite with specular

// Mathematical constants (may be imported from brdf.wgsl in production)
const SSS_PI: f32 = 3.14159265359;
const SSS_TWO_PI: f32 = 6.28318530718;
const SSS_EPSILON: f32 = 0.0001;

// Number of blur taps (must be odd for symmetric kernel)
const SSS_KERNEL_SIZE: u32 = 9u;
const SSS_KERNEL_HALF: u32 = 4u;  // (SSS_KERNEL_SIZE - 1) / 2

// ============================================================================
// SSS Profile Parameters
// ============================================================================

/// Subsurface scattering profile parameters.
/// Defines how light scatters within a translucent material.
///
/// scatter_distance: Mean free path in world units (per-channel for colored scattering)
/// scatter_color: Subsurface tint color (absorption/scattering color)
/// blur_strength: Strength of the screen-space blur effect [0, 1]
/// falloff_color: Color at the edge of scattering
/// curvature_scale: How much surface curvature affects scattering [0, 2]
struct SSSProfile {
    scatter_distance: vec3<f32>,
    scatter_color: vec3<f32>,
    blur_strength: f32,
    falloff_color: vec3<f32>,
    curvature_scale: f32,
    transmittance_color: vec3<f32>,
    boundary_color_bleed: f32,
}

/// SSS material parameters for per-fragment evaluation.
struct SSSParams {
    profile: SSSProfile,
    subsurface_intensity: f32,
    enable_transmission: bool,
    transmission_tint: vec3<f32>,
}

// ============================================================================
// Quality Tier Gating
// ============================================================================

/// Quality tier control for SSS. When false, SSS is disabled
/// and evaluate_sss returns base diffuse only. This const is dead-code
/// eliminated by naga compiler in LOW quality variants.
const QUALITY_SSS_ENABLED: bool = true;

// ============================================================================
// Predefined SSS Profiles
// ============================================================================

/// Human skin SSS profile (Burley approximation)
fn sss_profile_skin() -> SSSProfile {
    var profile: SSSProfile;
    profile.scatter_distance = vec3<f32>(1.0, 0.4, 0.25);  // RGB scatter radii
    profile.scatter_color = vec3<f32>(0.48, 0.25, 0.17);   // Warm skin tones
    profile.blur_strength = 0.8;
    profile.falloff_color = vec3<f32>(1.0, 0.37, 0.3);
    profile.curvature_scale = 0.75;
    profile.transmittance_color = vec3<f32>(0.88, 0.23, 0.17);
    profile.boundary_color_bleed = 0.5;
    return profile;
}

/// Wax SSS profile (shorter scatter, more uniform)
fn sss_profile_wax() -> SSSProfile {
    var profile: SSSProfile;
    profile.scatter_distance = vec3<f32>(0.5, 0.5, 0.4);
    profile.scatter_color = vec3<f32>(0.9, 0.85, 0.7);
    profile.blur_strength = 0.6;
    profile.falloff_color = vec3<f32>(0.95, 0.9, 0.8);
    profile.curvature_scale = 0.5;
    profile.transmittance_color = vec3<f32>(0.95, 0.9, 0.85);
    profile.boundary_color_bleed = 0.3;
    return profile;
}

/// Jade SSS profile (green tint, medium scatter)
fn sss_profile_jade() -> SSSProfile {
    var profile: SSSProfile;
    profile.scatter_distance = vec3<f32>(0.25, 0.5, 0.25);
    profile.scatter_color = vec3<f32>(0.5, 0.9, 0.5);
    profile.blur_strength = 0.5;
    profile.falloff_color = vec3<f32>(0.3, 0.7, 0.3);
    profile.curvature_scale = 0.6;
    profile.transmittance_color = vec3<f32>(0.4, 0.8, 0.4);
    profile.boundary_color_bleed = 0.4;
    return profile;
}

/// Milk SSS profile (high scatter, nearly uniform)
fn sss_profile_milk() -> SSSProfile {
    var profile: SSSProfile;
    profile.scatter_distance = vec3<f32>(0.8, 0.8, 0.75);
    profile.scatter_color = vec3<f32>(0.95, 0.95, 0.9);
    profile.blur_strength = 0.9;
    profile.falloff_color = vec3<f32>(0.98, 0.98, 0.95);
    profile.curvature_scale = 0.3;
    profile.transmittance_color = vec3<f32>(0.98, 0.98, 0.95);
    profile.boundary_color_bleed = 0.6;
    return profile;
}

/// Default SSS profile (balanced, neutral)
fn sss_profile_default() -> SSSProfile {
    var profile: SSSProfile;
    profile.scatter_distance = vec3<f32>(1.0, 1.0, 1.0);
    profile.scatter_color = vec3<f32>(1.0, 0.2, 0.1);
    profile.blur_strength = 0.7;
    profile.falloff_color = vec3<f32>(1.0, 0.37, 0.3);
    profile.curvature_scale = 0.75;
    profile.transmittance_color = vec3<f32>(0.88, 0.23, 0.17);
    profile.boundary_color_bleed = 0.5;
    return profile;
}

// ============================================================================
// Burley Normalized Diffusion Profile
// ============================================================================

/// Burley normalized diffusion function.
/// Models subsurface scattering using a sum of two exponentials.
///
/// Formula: R(r) = A * exp(-r/d) + B * exp(-r/(3d))
/// where A = 1/(2*PI*d^2), B = 1/(6*PI*d^2) for proper normalization.
///
/// This form integrates to 1 over the plane and decreases monotonically.
///
/// @param r: Distance from sample point (in world units)
/// @param d: Mean free path / scatter distance
/// @returns: Diffusion weight at distance r
fn burley_diffusion(r: f32, d: f32) -> f32 {
    if d < SSS_EPSILON {
        return 0.0;
    }

    // Burley's normalized diffusion profile (Burley 2015):
    // R(r) = A * e^(-r/d) + B * e^(-r/(3d))
    // Coefficients chosen so that 2*PI * integral(R(r) * r dr) = 1
    // A = 1 / (2 * PI * d^2), B = 1 / (6 * PI * d^2)
    let d2 = d * d;
    let A = 1.0 / (SSS_TWO_PI * d2);
    let B = 1.0 / (6.0 * SSS_PI * d2);

    let exp1 = exp(-r / d);
    let exp2 = exp(-r / (3.0 * d));

    return A * exp1 + B * exp2;
}

/// Per-channel Burley diffusion for colored scattering.
/// Each RGB channel can have a different scatter distance.
///
/// @param r: Distance from sample point
/// @param d: Scatter distance per channel (vec3)
/// @returns: Diffusion weight per channel
fn burley_diffusion_rgb(r: f32, d: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        burley_diffusion(r, d.x),
        burley_diffusion(r, d.y),
        burley_diffusion(r, d.z)
    );
}

/// Evaluate Burley diffusion profile for a sample.
/// Used to build the diffusion kernel for screen-space blur.
///
/// @param r: Radial distance in texture space
/// @param profile: SSS profile with scatter parameters
/// @returns: Diffusion weight with color tinting
fn evaluate_diffusion_profile(r: f32, profile: SSSProfile) -> vec3<f32> {
    // Get per-channel diffusion weights
    let weights = burley_diffusion_rgb(r, profile.scatter_distance);

    // Apply scatter color tinting
    return weights * profile.scatter_color;
}

// ============================================================================
// Screen-Space Blur Kernel
// ============================================================================

/// Pre-computed 9-tap Gaussian weights for separable blur.
/// These approximate a Gaussian distribution for the separable passes.
/// Actual kernel is modulated by the diffusion profile at runtime.
fn sss_kernel_weights() -> array<f32, 9> {
    // Gaussian sigma ~ 2.0, normalized weights
    return array<f32, 9>(
        0.0162, 0.0540, 0.1216, 0.1945, 0.2274,
        0.1945, 0.1216, 0.0540, 0.0162
    );
}

/// Pre-computed kernel offsets (in pixels, centered at 0)
fn sss_kernel_offsets() -> array<f32, 9> {
    return array<f32, 9>(
        -4.0, -3.0, -2.0, -1.0, 0.0,
        1.0, 2.0, 3.0, 4.0
    );
}

/// Compute diffusion-weighted kernel for a given profile.
/// Returns weights that combine Gaussian shape with diffusion falloff.
///
/// @param profile: SSS profile with scatter parameters
/// @param pixel_size: Size of a pixel in world units (for distance conversion)
/// @returns: 9 kernel weights (vec3 per tap for colored scattering)
fn compute_sss_kernel(profile: SSSProfile, pixel_size: f32) -> array<vec3<f32>, 9> {
    var kernel: array<vec3<f32>, 9>;
    let base_weights = sss_kernel_weights();
    let offsets = sss_kernel_offsets();

    // Total weight for normalization
    var total_weight = vec3<f32>(0.0);

    for (var i = 0u; i < SSS_KERNEL_SIZE; i++) {
        // Convert pixel offset to world distance
        let pixel_offset = abs(offsets[i]);
        let world_distance = pixel_offset * pixel_size;

        // Evaluate diffusion profile at this distance
        let diffusion = evaluate_diffusion_profile(world_distance, profile);

        // Combine with base Gaussian weight
        kernel[i] = diffusion * base_weights[i] * profile.blur_strength;
        total_weight += kernel[i];
    }

    // Normalize kernel to preserve energy
    if total_weight.x > SSS_EPSILON {
        for (var i = 0u; i < SSS_KERNEL_SIZE; i++) {
            kernel[i] = kernel[i] / total_weight;
        }
    }

    return kernel;
}

// ============================================================================
// Screen-Space SSS Blur Passes
// ============================================================================

/// Horizontal blur pass for screen-space SSS.
/// First pass of the separable convolution.
///
/// @param uv: Current fragment UV coordinates
/// @param sss_texture: SSS buffer texture from previous pass
/// @param sss_sampler: Texture sampler
/// @param depth_texture: Depth buffer for depth-aware blur
/// @param depth_sampler: Depth buffer sampler
/// @param profile: SSS profile for diffusion kernel
/// @param pixel_size: Pixel size in UV space (1/width, 1/height)
/// @param world_to_pixel: World units to pixel conversion factor
/// @returns: Blurred color for this fragment
fn sss_blur_horizontal(
    uv: vec2<f32>,
    sss_texture: texture_2d<f32>,
    sss_sampler: sampler,
    depth_texture: texture_2d<f32>,
    depth_sampler: sampler,
    profile: SSSProfile,
    pixel_size: vec2<f32>,
    world_to_pixel: f32
) -> vec3<f32> {
    // Quality gate
    if !QUALITY_SSS_ENABLED {
        return textureSample(sss_texture, sss_sampler, uv).rgb;
    }

    // Skip if blur strength is zero
    if profile.blur_strength < SSS_EPSILON {
        return textureSample(sss_texture, sss_sampler, uv).rgb;
    }

    // Sample center depth for depth-aware blur
    let center_depth = textureSample(depth_texture, depth_sampler, uv).r;

    // Compute kernel for current pixel
    let kernel = compute_sss_kernel(profile, 1.0 / world_to_pixel);
    let offsets = sss_kernel_offsets();

    var result = vec3<f32>(0.0);
    var total_weight = vec3<f32>(0.0);

    // Horizontal blur (along X axis)
    for (var i = 0u; i < SSS_KERNEL_SIZE; i++) {
        let offset_uv = vec2<f32>(offsets[i] * pixel_size.x, 0.0);
        let sample_uv = uv + offset_uv;

        // Sample color and depth
        let sample_color = textureSample(sss_texture, sss_sampler, sample_uv).rgb;
        let sample_depth = textureSample(depth_texture, depth_sampler, sample_uv).r;

        // Depth-aware weight (reduce contribution of samples at different depths)
        let depth_diff = abs(sample_depth - center_depth);
        let depth_weight = exp(-depth_diff * 100.0);  // Depth sensitivity

        // Apply kernel weight with depth modulation
        let weight = kernel[i] * depth_weight;
        result += sample_color * weight;
        total_weight += weight;
    }

    // Normalize by actual total weight
    if total_weight.x > SSS_EPSILON {
        result = result / total_weight;
    }

    return result;
}

/// Vertical blur pass for screen-space SSS.
/// Second pass of the separable convolution.
///
/// @param uv: Current fragment UV coordinates
/// @param sss_texture: SSS buffer texture from horizontal pass
/// @param sss_sampler: Texture sampler
/// @param depth_texture: Depth buffer for depth-aware blur
/// @param depth_sampler: Depth buffer sampler
/// @param profile: SSS profile for diffusion kernel
/// @param pixel_size: Pixel size in UV space (1/width, 1/height)
/// @param world_to_pixel: World units to pixel conversion factor
/// @returns: Blurred color for this fragment
fn sss_blur_vertical(
    uv: vec2<f32>,
    sss_texture: texture_2d<f32>,
    sss_sampler: sampler,
    depth_texture: texture_2d<f32>,
    depth_sampler: sampler,
    profile: SSSProfile,
    pixel_size: vec2<f32>,
    world_to_pixel: f32
) -> vec3<f32> {
    // Quality gate
    if !QUALITY_SSS_ENABLED {
        return textureSample(sss_texture, sss_sampler, uv).rgb;
    }

    // Skip if blur strength is zero
    if profile.blur_strength < SSS_EPSILON {
        return textureSample(sss_texture, sss_sampler, uv).rgb;
    }

    // Sample center depth
    let center_depth = textureSample(depth_texture, depth_sampler, uv).r;

    // Compute kernel
    let kernel = compute_sss_kernel(profile, 1.0 / world_to_pixel);
    let offsets = sss_kernel_offsets();

    var result = vec3<f32>(0.0);
    var total_weight = vec3<f32>(0.0);

    // Vertical blur (along Y axis)
    for (var i = 0u; i < SSS_KERNEL_SIZE; i++) {
        let offset_uv = vec2<f32>(0.0, offsets[i] * pixel_size.y);
        let sample_uv = uv + offset_uv;

        // Sample color and depth
        let sample_color = textureSample(sss_texture, sss_sampler, sample_uv).rgb;
        let sample_depth = textureSample(depth_texture, depth_sampler, sample_uv).r;

        // Depth-aware weight
        let depth_diff = abs(sample_depth - center_depth);
        let depth_weight = exp(-depth_diff * 100.0);

        // Apply kernel weight with depth modulation
        let weight = kernel[i] * depth_weight;
        result += sample_color * weight;
        total_weight += weight;
    }

    // Normalize
    if total_weight.x > SSS_EPSILON {
        result = result / total_weight;
    }

    return result;
}

// ============================================================================
// SSS Application and Compositing
// ============================================================================

/// Apply subsurface scattering to base color.
/// Final compositing step that blends SSS with base material.
///
/// @param base_color: Original surface color (diffuse)
/// @param sss_buffer: Blurred SSS buffer result
/// @param profile: SSS profile for color tinting
/// @param sss_intensity: Subsurface intensity factor [0, 1]
/// @returns: Final color with SSS applied
fn apply_sss(
    base_color: vec3<f32>,
    sss_buffer: vec3<f32>,
    profile: SSSProfile,
    sss_intensity: f32
) -> vec3<f32> {
    // Quality gate
    if !QUALITY_SSS_ENABLED {
        return base_color;
    }

    // Skip if intensity is zero
    if sss_intensity < SSS_EPSILON {
        return base_color;
    }

    // Blend SSS contribution with base color
    // The SSS buffer already contains the blurred diffuse, so we
    // interpolate between direct diffuse and scattered diffuse
    let sss_contribution = sss_buffer * profile.scatter_color;

    // Mix based on intensity
    return mix(base_color, sss_contribution, sss_intensity * profile.blur_strength);
}

/// Apply SSS with boundary color bleed.
/// Enhanced version that handles color bleeding at light/shadow boundaries.
///
/// @param base_color: Original surface color
/// @param sss_buffer: Blurred SSS buffer result
/// @param shadow_mask: Shadow attenuation factor [0=shadow, 1=lit]
/// @param profile: SSS profile
/// @param sss_intensity: Subsurface intensity
/// @returns: Final color with boundary bleeding
fn apply_sss_with_bleeding(
    base_color: vec3<f32>,
    sss_buffer: vec3<f32>,
    shadow_mask: f32,
    profile: SSSProfile,
    sss_intensity: f32
) -> vec3<f32> {
    // Quality gate
    if !QUALITY_SSS_ENABLED {
        return base_color;
    }

    if sss_intensity < SSS_EPSILON {
        return base_color;
    }

    // Compute boundary factor (high near light/shadow transitions)
    // Derivative of shadow mask indicates boundary
    // Note: In practice, this would use ddx/ddy or pre-computed edge detection
    let boundary_factor = profile.boundary_color_bleed;

    // Bleed falloff color into shadowed regions
    let bleed_color = mix(base_color, profile.falloff_color, boundary_factor * (1.0 - shadow_mask));

    // Apply SSS blur
    let sss_contribution = sss_buffer * profile.scatter_color;
    let sss_result = mix(bleed_color, sss_contribution, sss_intensity * profile.blur_strength);

    return sss_result;
}

// ============================================================================
// Transmission (Back-Face Scattering)
// ============================================================================

/// Evaluate subsurface transmission (light through thin surfaces).
/// Models light entering from the back side of a surface.
///
/// @param N: Surface normal (front-facing)
/// @param L: Light direction (toward light)
/// @param V: View direction (toward camera)
/// @param thickness: Local surface thickness (0=thin, 1=thick)
/// @param profile: SSS profile
/// @returns: Transmission contribution
fn evaluate_sss_transmission(
    N: vec3<f32>,
    L: vec3<f32>,
    V: vec3<f32>,
    thickness: f32,
    profile: SSSProfile
) -> vec3<f32> {
    // Quality gate
    if !QUALITY_SSS_ENABLED {
        return vec3<f32>(0.0);
    }

    // Wrap lighting for transmission (light from behind surface)
    // -N.L > 0 means light is coming from behind
    let NoL_back = max(-dot(N, L), 0.0);

    // Distance light travels through material (approximation)
    let scatter_dist = thickness * profile.scatter_distance;

    // Compute transmission attenuation using Beer-Lambert law
    // Transmission decreases exponentially with distance
    let attenuation = exp(-scatter_dist * 2.0);

    // Forward scattering lobe (Henyey-Greenstein approximation)
    let VoL = dot(V, -L);
    let phase = 0.25 + 0.5 * pow(max(VoL, 0.0), 2.0);

    // Final transmission
    let transmission = NoL_back * attenuation * phase * profile.transmittance_color;

    return transmission;
}

// ============================================================================
// Pre-Integrated Diffusion LUT
// ============================================================================

/// Sample pre-integrated diffusion LUT.
/// The LUT encodes integrated diffusion profiles for efficient runtime lookup.
///
/// @param NoL: N dot L (can be negative for wrap lighting)
/// @param curvature: Surface curvature (high curvature = more scattering)
/// @param lut_texture: Pre-integrated LUT texture
/// @param lut_sampler: LUT sampler
/// @returns: Pre-integrated diffusion factor (RGB)
fn sample_diffusion_lut(
    NoL: f32,
    curvature: f32,
    lut_texture: texture_2d<f32>,
    lut_sampler: sampler
) -> vec3<f32> {
    // Map NoL from [-1, 1] to [0, 1] for LUT U coordinate
    // Negative NoL represents light from behind (for wrap/transmission)
    let u = NoL * 0.5 + 0.5;

    // Curvature as V coordinate
    // High curvature = more scattering visible
    let v = saturate(curvature);

    return textureSample(lut_texture, lut_sampler, vec2<f32>(u, v)).rgb;
}

/// Generate diffusion LUT value for baking.
/// Call this in a compute shader to pre-compute the LUT.
///
/// @param uv: LUT coordinate (u=NoL mapped, v=curvature)
/// @param profile: SSS profile
/// @returns: Integrated diffusion value
fn compute_diffusion_lut_value(uv: vec2<f32>, profile: SSSProfile) -> vec3<f32> {
    // Unmap U to NoL [-1, 1]
    let NoL = uv.x * 2.0 - 1.0;

    // V is curvature [0, 1]
    let curvature = uv.y;

    // Integrate diffusion profile weighted by curvature
    // Higher curvature means light scatters more visibly
    let scatter_scale = 1.0 + curvature * profile.curvature_scale;

    // Simulate integration over hemisphere
    var integrated = vec3<f32>(0.0);
    let num_samples = 16u;

    for (var i = 0u; i < num_samples; i++) {
        let t = f32(i) / f32(num_samples - 1u);
        let angle = t * SSS_PI;
        let sample_NoL = cos(angle);

        // Distance based on angle from direct lighting
        let angle_diff = abs(sample_NoL - NoL);
        let r = angle_diff * scatter_scale;

        // Sample diffusion profile
        let sample_weight = evaluate_diffusion_profile(r, profile);

        // Weight by solid angle approximation
        integrated += sample_weight * sin(angle);
    }

    // Normalize
    integrated = integrated / f32(num_samples);

    return integrated;
}

// ============================================================================
// Curvature Estimation
// ============================================================================

/// Estimate surface curvature from depth buffer.
/// Used to modulate SSS based on geometric complexity.
///
/// @param uv: Fragment UV
/// @param depth_texture: Depth buffer
/// @param depth_sampler: Depth sampler
/// @param pixel_size: Pixel size in UV space
/// @returns: Estimated curvature [0, 1]
fn estimate_curvature_from_depth(
    uv: vec2<f32>,
    depth_texture: texture_2d<f32>,
    depth_sampler: sampler,
    pixel_size: vec2<f32>
) -> f32 {
    // Sample depth in cross pattern
    let d_c = textureSample(depth_texture, depth_sampler, uv).r;
    let d_l = textureSample(depth_texture, depth_sampler, uv - vec2<f32>(pixel_size.x, 0.0)).r;
    let d_r = textureSample(depth_texture, depth_sampler, uv + vec2<f32>(pixel_size.x, 0.0)).r;
    let d_u = textureSample(depth_texture, depth_sampler, uv - vec2<f32>(0.0, pixel_size.y)).r;
    let d_d = textureSample(depth_texture, depth_sampler, uv + vec2<f32>(0.0, pixel_size.y)).r;

    // Compute second derivatives (Laplacian)
    let d2x = d_l - 2.0 * d_c + d_r;
    let d2y = d_u - 2.0 * d_c + d_d;

    // Curvature magnitude
    let curvature = abs(d2x) + abs(d2y);

    // Normalize to [0, 1] range
    return saturate(curvature * 100.0);
}

// ============================================================================
// Full SSS Evaluation for Direct Lighting
// ============================================================================

/// Evaluate complete SSS contribution for direct lighting.
/// For real-time use with pre-integrated LUT.
///
/// @param N: Surface normal
/// @param L: Light direction
/// @param V: View direction
/// @param base_diffuse: Base diffuse color (albedo * NdotL * light)
/// @param sss_params: SSS parameters
/// @param lut_texture: Pre-integrated diffusion LUT
/// @param lut_sampler: LUT sampler
/// @param curvature: Surface curvature estimate
/// @returns: Diffuse with SSS applied
fn evaluate_sss_direct(
    N: vec3<f32>,
    L: vec3<f32>,
    V: vec3<f32>,
    base_diffuse: vec3<f32>,
    sss_params: SSSParams,
    lut_texture: texture_2d<f32>,
    lut_sampler: sampler,
    curvature: f32
) -> vec3<f32> {
    // Quality gate
    if !QUALITY_SSS_ENABLED {
        return base_diffuse;
    }

    if sss_params.subsurface_intensity < SSS_EPSILON {
        return base_diffuse;
    }

    let NoL = dot(N, L);

    // Sample pre-integrated diffusion
    let diffusion = sample_diffusion_lut(
        NoL,
        curvature * sss_params.profile.curvature_scale,
        lut_texture,
        lut_sampler
    );

    // Apply SSS coloring
    let sss_diffuse = base_diffuse * diffusion * sss_params.profile.scatter_color;

    // Optionally add transmission
    var transmission = vec3<f32>(0.0);
    if sss_params.enable_transmission {
        transmission = evaluate_sss_transmission(
            N, L, V,
            0.5,  // Default thickness, could be from texture
            sss_params.profile
        ) * sss_params.transmission_tint;
    }

    // Blend SSS with base diffuse
    let result = mix(base_diffuse, sss_diffuse, sss_params.subsurface_intensity);

    return result + transmission * sss_params.subsurface_intensity;
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Create default SSS parameters.
fn sss_params_default() -> SSSParams {
    var params: SSSParams;
    params.profile = sss_profile_default();
    params.subsurface_intensity = 0.0;
    params.enable_transmission = false;
    params.transmission_tint = vec3<f32>(1.0, 0.0, 0.0);
    return params;
}

/// Create SSS parameters from profile.
fn sss_params_from_profile(
    profile: SSSProfile,
    intensity: f32,
    enable_transmission: bool
) -> SSSParams {
    var params: SSSParams;
    params.profile = profile;
    params.subsurface_intensity = saturate(intensity);
    params.enable_transmission = enable_transmission;
    params.transmission_tint = profile.transmittance_color;
    return params;
}

/// Get SSS mask value for G-buffer.
/// Returns value to store in G-buffer for SSS pass identification.
///
/// @param sss_params: SSS parameters
/// @returns: Mask value (0=no SSS, 1=full SSS)
fn get_sss_mask(sss_params: SSSParams) -> f32 {
    if !QUALITY_SSS_ENABLED {
        return 0.0;
    }
    return sss_params.subsurface_intensity * sss_params.profile.blur_strength;
}
