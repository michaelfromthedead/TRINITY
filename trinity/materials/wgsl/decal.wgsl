// Deferred Decal Domain - G-buffer modification via box projection
// T-MAT-5.3: Decal Domain Implementation
//
// This module provides deferred decal rendering that modifies existing G-buffer
// data without direct lighting evaluation. Decals are projected onto surfaces
// using box projection with configurable blend modes per channel.
//
// Features:
//   - Box projection: world_pos -> decal_uv via inverse projection matrix
//   - G-buffer modification: albedo, normal, roughness, metallic writes
//   - Blend modes: Alpha, Additive, Multiply per channel
//   - Normal fade: attenuate at glancing angles to avoid stretching
//   - Angle fade: fade based on projection angle
//
// References:
//   - Rendering Technology of "Lords of the Fallen" - decal projection
//   - UE4 Deferred Decals - blend mode architecture

// Mathematical constants
const DECAL_PI: f32 = 3.14159265359;
const DECAL_EPSILON: f32 = 0.0001;

// ============================================================================
// Blend Mode Enumeration
// ============================================================================

/// Blend modes for individual G-buffer channels.
/// Encoded as u32 for WGSL compatibility.
const BLEND_ALPHA: u32 = 0u;      // Standard alpha blend: src * alpha + dst * (1 - alpha)
const BLEND_ADDITIVE: u32 = 1u;   // Additive: src + dst
const BLEND_MULTIPLY: u32 = 2u;   // Multiplicative: src * dst

// ============================================================================
// Decal Parameter Structs
// ============================================================================

/// Decal projection and blending parameters.
/// Passed to decal projection functions.
struct DecalParams {
    /// Inverse projection matrix: transforms world position to decal-local space
    /// Decal-local space has XY in [-1,1] as texture coordinates, Z for depth
    projection_matrix: mat4x4<f32>,

    /// Decal bounding box extents in local space (half-width, half-height, half-depth)
    /// Used for clipping fragments outside the decal volume
    bounds: vec3<f32>,

    /// Normal fade parameters: (start_angle_cos, end_angle_cos, unused, unused)
    /// start_angle_cos: angle at which fade begins (e.g., cos(60deg) = 0.5)
    /// end_angle_cos: angle at which decal is fully faded (e.g., cos(85deg) = 0.087)
    normal_fade: vec4<f32>,

    /// Angle fade parameters: (enabled, strength, exponent, unused)
    /// Controls how the decal fades based on projection angle
    angle_fade: vec4<f32>,

    /// Blend modes per channel: (albedo_mode, normal_mode, roughness_mode, metallic_mode)
    /// Each component is one of BLEND_ALPHA, BLEND_ADDITIVE, BLEND_MULTIPLY
    blend_modes: vec4<u32>,

    /// Global opacity multiplier (0.0 - 1.0)
    opacity: f32,

    /// Normal blend intensity (0.0 - 1.0)
    /// Controls how much the decal normal affects the surface
    normal_intensity: f32,

    /// Padding for alignment
    _padding: vec2<f32>,
}

/// Decal texture samples.
/// Contains sampled values from decal texture atlas.
struct DecalSamples {
    /// Albedo/diffuse color with alpha
    albedo: vec4<f32>,
    /// Tangent-space normal (RGB packed)
    normal: vec3<f32>,
    /// Roughness value
    roughness: f32,
    /// Metallic value
    metallic: f32,
    /// Ambient occlusion (optional, defaults to 1.0)
    ao: f32,
}

/// Decal projection result.
/// Contains UV coordinates and validity information.
struct DecalProjection {
    /// UV coordinates for texture sampling (0-1 range)
    uv: vec2<f32>,
    /// Depth along projection axis (-1 to 1 in decal local space)
    depth: f32,
    /// True if the projection is valid (inside decal volume)
    valid: bool,
    /// Fade factor from angle-based attenuation
    angle_fade: f32,
}

/// G-buffer output for deferred decal rendering.
struct DecalGBufferOutput {
    /// Albedo color with blend factor in alpha
    albedo: vec4<f32>,
    /// World-space normal (octahedral or XYZ packed)
    normal: vec4<f32>,
    /// Material properties: metallic, roughness, AO, specular
    material: vec4<f32>,
}

// ============================================================================
// Decal Projection Functions
// ============================================================================

/// Project world position into decal UV space.
/// Returns DecalProjection with UV coordinates and validity.
///
/// @param world_pos: World-space position of the fragment
/// @param params: Decal parameters containing projection matrix and bounds
/// @returns: DecalProjection with UV, depth, and validity
fn project_decal(world_pos: vec3<f32>, params: DecalParams) -> DecalProjection {
    var result: DecalProjection;

    // Transform world position to decal-local space
    let local_pos = params.projection_matrix * vec4<f32>(world_pos, 1.0);
    let local_pos_3 = local_pos.xyz / local_pos.w;

    // Check if inside decal bounding box
    let inside_x = abs(local_pos_3.x) <= params.bounds.x;
    let inside_y = abs(local_pos_3.y) <= params.bounds.y;
    let inside_z = abs(local_pos_3.z) <= params.bounds.z;
    result.valid = inside_x && inside_y && inside_z;

    // Convert to UV coordinates (local XY from [-bounds, bounds] to [0, 1])
    result.uv = (local_pos_3.xy / params.bounds.xy) * 0.5 + 0.5;
    result.depth = local_pos_3.z / params.bounds.z;

    // Default angle fade
    result.angle_fade = 1.0;

    return result;
}

/// Compute angle-based fade factor for decal projection.
/// Prevents decals from appearing on surfaces nearly parallel to projection axis.
///
/// @param decal_dir: Normalized decal projection direction (typically decal Z-axis)
/// @param surface_normal: Normalized surface normal at the fragment
/// @param params: Decal parameters with angle fade settings
/// @returns: Fade factor (0.0 = fully faded, 1.0 = no fade)
fn compute_angle_fade(decal_dir: vec3<f32>, surface_normal: vec3<f32>, params: DecalParams) -> f32 {
    if (params.angle_fade.x < 0.5) {
        // Angle fade disabled
        return 1.0;
    }

    // Compute angle between decal direction and surface normal
    // Decal projects along -Z in its local space
    let cos_angle = abs(dot(decal_dir, surface_normal));

    // Apply fade based on angle
    let fade_strength = params.angle_fade.y;
    let fade_exponent = params.angle_fade.z;

    // Smooth step fade
    let fade = pow(cos_angle, fade_exponent) * fade_strength;
    return clamp(fade, 0.0, 1.0);
}

/// Compute normal fade factor to prevent stretching at glancing angles.
///
/// @param surface_normal: Normalized surface normal at the fragment
/// @param decal_normal: Decal projection direction (typically decal forward/Z)
/// @param params: Decal parameters with normal fade settings
/// @returns: Fade factor (0.0 = fully faded, 1.0 = no fade)
fn apply_normal_fade(surface_normal: vec3<f32>, decal_normal: vec3<f32>, params: DecalParams) -> f32 {
    let cos_angle = dot(surface_normal, decal_normal);

    let start_cos = params.normal_fade.x;
    let end_cos = params.normal_fade.y;

    // Smooth interpolation between start and end angles
    if (cos_angle >= start_cos) {
        return 1.0;
    } else if (cos_angle <= end_cos) {
        return 0.0;
    } else {
        // Smooth step between end and start
        let t = (cos_angle - end_cos) / (start_cos - end_cos);
        return t * t * (3.0 - 2.0 * t); // smoothstep
    }
}

// ============================================================================
// Decal Texture Sampling
// ============================================================================

/// Sample all decal textures at given UV coordinates.
/// This is a helper that samples albedo, normal, and material textures.
///
/// Note: Actual texture bindings should be provided by the material system.
/// This function shows the expected interface.
///
/// @param uv: UV coordinates from decal projection
/// @param albedo_tex: Decal albedo texture
/// @param albedo_samp: Sampler for albedo texture
/// @param normal_tex: Decal normal map texture
/// @param normal_samp: Sampler for normal texture
/// @param material_tex: Decal material properties texture (R=metallic, G=roughness, B=AO)
/// @param material_samp: Sampler for material texture
/// @returns: DecalSamples with all sampled values
fn sample_decal_textures(
    uv: vec2<f32>,
    albedo_tex: texture_2d<f32>,
    albedo_samp: sampler,
    normal_tex: texture_2d<f32>,
    normal_samp: sampler,
    material_tex: texture_2d<f32>,
    material_samp: sampler,
) -> DecalSamples {
    var samples: DecalSamples;

    // Sample albedo (RGBA)
    samples.albedo = textureSample(albedo_tex, albedo_samp, uv);

    // Sample normal map and unpack from [0,1] to [-1,1]
    let normal_sample = textureSample(normal_tex, normal_samp, uv);
    samples.normal = normalize(normal_sample.rgb * 2.0 - 1.0);

    // Sample material properties
    let material_sample = textureSample(material_tex, material_samp, uv);
    samples.metallic = material_sample.r;
    samples.roughness = material_sample.g;
    samples.ao = material_sample.b;

    return samples;
}

/// Create default decal samples for materials without all textures.
fn decal_samples_default() -> DecalSamples {
    var samples: DecalSamples;
    samples.albedo = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    samples.normal = vec3<f32>(0.0, 0.0, 1.0);
    samples.roughness = 0.5;
    samples.metallic = 0.0;
    samples.ao = 1.0;
    return samples;
}

// ============================================================================
// G-buffer Blending Functions
// ============================================================================

/// Blend a single channel using the specified blend mode.
///
/// @param original: Original G-buffer value
/// @param decal: Decal contribution
/// @param blend_mode: One of BLEND_ALPHA, BLEND_ADDITIVE, BLEND_MULTIPLY
/// @param alpha: Blend factor for alpha blending
/// @returns: Blended channel value
fn blend_channel(original: f32, decal: f32, blend_mode: u32, alpha: f32) -> f32 {
    switch (blend_mode) {
        case BLEND_ALPHA: {
            // Standard alpha blend
            return mix(original, decal, alpha);
        }
        case BLEND_ADDITIVE: {
            // Additive blend
            return original + decal * alpha;
        }
        case BLEND_MULTIPLY: {
            // Multiplicative blend
            return mix(original, original * decal, alpha);
        }
        default: {
            return original;
        }
    }
}

/// Blend RGB channels using the specified blend mode.
fn blend_channel_rgb(original: vec3<f32>, decal: vec3<f32>, blend_mode: u32, alpha: f32) -> vec3<f32> {
    switch (blend_mode) {
        case BLEND_ALPHA: {
            return mix(original, decal, alpha);
        }
        case BLEND_ADDITIVE: {
            return original + decal * alpha;
        }
        case BLEND_MULTIPLY: {
            return mix(original, original * decal, alpha);
        }
        default: {
            return original;
        }
    }
}

/// Blend G-buffer albedo channel.
fn blend_gbuffer_albedo(original: vec3<f32>, decal: vec4<f32>, params: DecalParams) -> vec3<f32> {
    let alpha = decal.a * params.opacity;
    return blend_channel_rgb(original, decal.rgb, params.blend_modes.x, alpha);
}

/// Blend G-buffer normal channel with reoriented normal mapping.
/// Uses the Reoriented Normal Mapping technique for correct blending.
fn blend_gbuffer_normal(
    original_normal: vec3<f32>,
    decal_normal: vec3<f32>,
    params: DecalParams,
    alpha: f32
) -> vec3<f32> {
    let blend_alpha = alpha * params.opacity * params.normal_intensity;

    // Reoriented Normal Mapping blend
    // This produces correct results when blending normal maps
    let t = original_normal + vec3<f32>(0.0, 0.0, 1.0);
    let u = decal_normal * vec3<f32>(-1.0, -1.0, 1.0);
    let blended = t * dot(t, u) - u * t.z;

    // Interpolate between original and blended based on alpha
    return normalize(mix(original_normal, blended, blend_alpha));
}

/// Blend G-buffer roughness channel.
fn blend_gbuffer_roughness(original: f32, decal: f32, params: DecalParams, alpha: f32) -> f32 {
    let blend_alpha = alpha * params.opacity;
    return blend_channel(original, decal, params.blend_modes.z, blend_alpha);
}

/// Blend G-buffer metallic channel.
fn blend_gbuffer_metallic(original: f32, decal: f32, params: DecalParams, alpha: f32) -> f32 {
    let blend_alpha = alpha * params.opacity;
    return blend_channel(original, decal, params.blend_modes.w, blend_alpha);
}

// ============================================================================
// Complete Decal Evaluation
// ============================================================================

/// Evaluate deferred decal and produce G-buffer modifications.
/// This is the main entry point for decal rendering.
///
/// @param world_pos: World-space position of the fragment
/// @param surface_normal: World-space surface normal
/// @param decal_samples: Sampled decal textures
/// @param params: Decal parameters
/// @param original_gbuffer: Original G-buffer values to modify
/// @returns: Modified G-buffer output
fn evaluate_decal(
    world_pos: vec3<f32>,
    surface_normal: vec3<f32>,
    decal_samples: DecalSamples,
    params: DecalParams,
    original_gbuffer: DecalGBufferOutput
) -> DecalGBufferOutput {
    var output = original_gbuffer;

    // Project world position to decal UV space
    let projection = project_decal(world_pos, params);

    // Early exit if outside decal volume
    if (!projection.valid) {
        return output;
    }

    // Extract decal projection direction from matrix (Z-axis)
    let decal_z = normalize(vec3<f32>(
        params.projection_matrix[2][0],
        params.projection_matrix[2][1],
        params.projection_matrix[2][2]
    ));

    // Compute fade factors
    let normal_fade = apply_normal_fade(surface_normal, decal_z, params);
    let angle_fade = compute_angle_fade(decal_z, surface_normal, params);
    let total_fade = normal_fade * angle_fade * params.opacity;

    // Skip if fully faded
    if (total_fade < DECAL_EPSILON) {
        return output;
    }

    // Compute effective alpha
    let alpha = decal_samples.albedo.a * total_fade;

    // Blend albedo
    output.albedo.rgb = blend_gbuffer_albedo(
        original_gbuffer.albedo.rgb,
        vec4<f32>(decal_samples.albedo.rgb, alpha),
        params
    );
    output.albedo.a = max(original_gbuffer.albedo.a, alpha);

    // Blend normal (transform decal tangent-space normal to world space first)
    // Note: For proper implementation, tangent/bitangent from surface should be used
    output.normal.xyz = blend_gbuffer_normal(
        original_gbuffer.normal.xyz,
        decal_samples.normal,
        params,
        alpha
    );

    // Blend material properties
    output.material.x = blend_gbuffer_metallic(
        original_gbuffer.material.x,
        decal_samples.metallic,
        params,
        alpha
    );
    output.material.y = blend_gbuffer_roughness(
        original_gbuffer.material.y,
        decal_samples.roughness,
        params,
        alpha
    );
    // AO uses alpha blend
    output.material.z = mix(original_gbuffer.material.z, decal_samples.ao, alpha);

    return output;
}

/// Create default decal parameters.
fn decal_params_default() -> DecalParams {
    var params: DecalParams;
    params.projection_matrix = mat4x4<f32>(
        vec4<f32>(1.0, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, 1.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 1.0, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0)
    );
    params.bounds = vec3<f32>(1.0, 1.0, 1.0);
    params.normal_fade = vec4<f32>(0.5, 0.087, 0.0, 0.0); // 60deg start, 85deg end
    params.angle_fade = vec4<f32>(1.0, 1.0, 1.0, 0.0);    // enabled, full strength
    params.blend_modes = vec4<u32>(BLEND_ALPHA, BLEND_ALPHA, BLEND_ALPHA, BLEND_ALPHA);
    params.opacity = 1.0;
    params.normal_intensity = 1.0;
    params._padding = vec2<f32>(0.0, 0.0);
    return params;
}

// ============================================================================
// Decal Pass Fragment Shader
// ============================================================================

/// Decal uniforms for rendering pass.
struct DecalUniforms {
    /// View-projection matrix
    view_proj: mat4x4<f32>,
    /// Inverse view-projection for depth reconstruction
    inv_view_proj: mat4x4<f32>,
    /// Camera position in world space
    camera_pos: vec3<f32>,
    /// Near plane distance
    near_plane: f32,
    /// Far plane distance
    far_plane: f32,
    /// Padding
    _padding: vec3<f32>,
}

/// Reconstruct world position from depth buffer.
///
/// @param uv: Screen-space UV coordinates
/// @param depth: Sampled depth value
/// @param inv_view_proj: Inverse view-projection matrix
/// @returns: World-space position
fn reconstruct_world_position(uv: vec2<f32>, depth: f32, inv_view_proj: mat4x4<f32>) -> vec3<f32> {
    // Convert UV to clip space (-1 to 1)
    let clip_xy = uv * 2.0 - 1.0;
    let clip_pos = vec4<f32>(clip_xy.x, -clip_xy.y, depth, 1.0);

    // Transform to world space
    let world_pos = inv_view_proj * clip_pos;
    return world_pos.xyz / world_pos.w;
}

/// Decode normal from G-buffer (assuming octahedral encoding or simple XYZ pack).
fn decode_gbuffer_normal(encoded: vec4<f32>) -> vec3<f32> {
    // Simple decode: stored as (N * 0.5 + 0.5)
    return normalize(encoded.xyz * 2.0 - 1.0);
}

/// Encode normal for G-buffer output.
fn encode_gbuffer_normal(normal: vec3<f32>) -> vec4<f32> {
    return vec4<f32>(normal * 0.5 + 0.5, 1.0);
}
