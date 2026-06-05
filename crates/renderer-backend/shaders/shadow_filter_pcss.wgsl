// SPDX-License-Identifier: MIT
//
// shadow_filter_pcss.wgsl - Percentage Closer Soft Shadows (T-LIT-6.5).
//
// Implements PCSS for physically-based soft shadows with distance-dependent
// penumbra. The algorithm models real-world shadow behavior where shadows
// become softer as the distance between occluder and receiver increases.
//
// 3-Step Algorithm:
//   1. Blocker Search: Find average depth of occluders in search region
//   2. Penumbra Estimation: Calculate penumbra width using similar triangles
//   3. Variable PCF: Apply PCF with kernel size proportional to penumbra
//
// Imports shadow_common.wgsl for shared types (ShadowTileInfo, ShadowSampleResult).

// ============================================================================
// Constants
// ============================================================================

const PCSS_EPSILON: f32 = 0.00001;
const PCSS_MIN_BLOCKERS: f32 = 0.5;

// ============================================================================
// PCSS Parameters
// ============================================================================

/// Configuration parameters for PCSS sampling.
struct PcssParams {
    /// Virtual light size in shadow map UV space.
    /// Larger values produce wider penumbras.
    /// Typical values: 0.02 - 0.1 for area lights.
    light_size: f32,

    /// Search radius for blocker detection (in texels).
    /// Should be large enough to find blockers at max penumbra width.
    /// Typical values: 8 - 32 texels.
    blocker_search_radius: f32,

    /// Minimum PCF filter radius (in texels).
    /// Prevents over-sharpening when blockers are very close.
    /// Typical values: 1 - 2 texels.
    min_filter_radius: f32,

    /// Maximum PCF filter radius (in texels).
    /// Limits blur for distant blockers to control performance.
    /// Typical values: 16 - 64 texels.
    max_filter_radius: f32,
}

// ============================================================================
// 32-Sample Poisson Disk
// ============================================================================

/// Returns a 32-sample Poisson disk offset for high-quality PCSS sampling.
/// Samples are distributed to minimize clustering and provide uniform coverage.
fn poisson_disk_32(index: u32) -> vec2<f32> {
    // Pre-computed 32-sample Poisson disk with blue noise distribution.
    // Generated using Mitchell's best-candidate algorithm.
    switch index {
        case 0u:  { return vec2<f32>(-0.9405873, -0.2987282); }
        case 1u:  { return vec2<f32>( 0.9358258, -0.3323381); }
        case 2u:  { return vec2<f32>(-0.0756175, -0.9597216); }
        case 3u:  { return vec2<f32>( 0.3865239,  0.9110471); }
        case 4u:  { return vec2<f32>(-0.8965247,  0.4369221); }
        case 5u:  { return vec2<f32>( 0.4547742, -0.8732817); }
        case 6u:  { return vec2<f32>(-0.4253765,  0.1428245); }
        case 7u:  { return vec2<f32>( 0.8932862,  0.4390281); }
        case 8u:  { return vec2<f32>(-0.1874152, -0.5623748); }
        case 9u:  { return vec2<f32>( 0.5428163, -0.3984527); }
        case 10u: { return vec2<f32>(-0.6473128, -0.7412846); }
        case 11u: { return vec2<f32>( 0.1892746,  0.4837921); }
        case 12u: { return vec2<f32>(-0.8172649,  0.0483627); }
        case 13u: { return vec2<f32>( 0.7628154,  0.1238475); }
        case 14u: { return vec2<f32>(-0.2931748,  0.9428163); }
        case 15u: { return vec2<f32>( 0.0742816, -0.2847163); }
        case 16u: { return vec2<f32>(-0.5127463,  0.5937281); }
        case 17u: { return vec2<f32>( 0.6237481, -0.0847162); }
        case 18u: { return vec2<f32>(-0.0384726, -0.7328461); }
        case 19u: { return vec2<f32>( 0.2847163,  0.1437281); }
        case 20u: { return vec2<f32>(-0.7428163, -0.3847216); }
        case 21u: { return vec2<f32>( 0.4738261,  0.5382716); }
        case 22u: { return vec2<f32>(-0.3847216,  0.3928471); }
        case 23u: { return vec2<f32>( 0.9127463, -0.0742816); }
        case 24u: { return vec2<f32>(-0.5847216, -0.1238471); }
        case 25u: { return vec2<f32>( 0.1847263,  0.7382716); }
        case 26u: { return vec2<f32>(-0.2738461, -0.1847263); }
        case 27u: { return vec2<f32>( 0.7328461,  0.6847216); }
        case 28u: { return vec2<f32>(-0.9837261,  0.1428376); }
        case 29u: { return vec2<f32>( 0.0847263, -0.5837216); }
        case 30u: { return vec2<f32>(-0.4627381,  0.8372816); }
        case 31u: { return vec2<f32>( 0.5627381, -0.6372816); }
        default: { return vec2<f32>(0.0, 0.0); }
    }
}

/// Returns a 16-sample Poisson disk offset for blocker search.
/// Fewer samples for the initial search pass to reduce cost.
fn poisson_disk_16(index: u32) -> vec2<f32> {
    switch index {
        case 0u:  { return vec2<f32>(-0.9420162, -0.3990622); }
        case 1u:  { return vec2<f32>( 0.9455861, -0.7689073); }
        case 2u:  { return vec2<f32>(-0.0941841, -0.9293887); }
        case 3u:  { return vec2<f32>( 0.3449594,  0.2938776); }
        case 4u:  { return vec2<f32>(-0.9158858,  0.4577143); }
        case 5u:  { return vec2<f32>(-0.8154423, -0.8791246); }
        case 6u:  { return vec2<f32>(-0.3827754,  0.2767685); }
        case 7u:  { return vec2<f32>( 0.9748440,  0.7564838); }
        case 8u:  { return vec2<f32>( 0.4432333, -0.9751155); }
        case 9u:  { return vec2<f32>( 0.5374298, -0.4737342); }
        case 10u: { return vec2<f32>(-0.2649691, -0.4189302); }
        case 11u: { return vec2<f32>( 0.7919751,  0.1909019); }
        case 12u: { return vec2<f32>(-0.2418884,  0.9970651); }
        case 13u: { return vec2<f32>(-0.8140996,  0.9143759); }
        case 14u: { return vec2<f32>( 0.1998413,  0.7864137); }
        case 15u: { return vec2<f32>( 0.1438316, -0.1410079); }
        default: { return vec2<f32>(0.0, 0.0); }
    }
}

// ============================================================================
// Rotated Sample Generation
// ============================================================================

/// Generates a rotation angle from screen coordinates for sample randomization.
/// This breaks up banding artifacts by rotating the Poisson disk per-pixel.
fn generate_rotation(screen_pos: vec2<f32>) -> f32 {
    // Interleaved gradient noise provides temporally stable randomization.
    let magic = vec3<f32>(0.06711056, 0.00583715, 52.9829189);
    return fract(magic.z * fract(dot(screen_pos, magic.xy))) * 6.283185307;
}

/// Applies rotation to a 2D sample offset.
fn rotate_sample(sample: vec2<f32>, angle: f32) -> vec2<f32> {
    let cos_a = cos(angle);
    let sin_a = sin(angle);
    return vec2<f32>(
        sample.x * cos_a - sample.y * sin_a,
        sample.x * sin_a + sample.y * cos_a
    );
}

// ============================================================================
// Step 1: Blocker Search
// ============================================================================

/// Searches for blockers in the shadow map within the search region.
///
/// Returns:
/// - x: Average blocker depth (only valid if y > 0)
/// - y: Number of blockers found (as float for interpolation)
///
/// Parameters:
/// - shadow_map: Depth texture for shadow sampling
/// - uv: Shadow map UV coordinates (atlas-transformed)
/// - receiver_depth: Depth of the receiving surface in light space
/// - search_radius: Search radius in texels
/// - texel_size: Size of one texel in UV space
/// - rotation: Per-pixel rotation angle for sample randomization
fn find_blocker_depth(
    shadow_map: texture_depth_2d,
    uv: vec2<f32>,
    receiver_depth: f32,
    search_radius: f32,
    texel_size: f32,
    rotation: f32
) -> vec2<f32> {
    var blocker_sum = 0.0;
    var blocker_count = 0.0;

    let search_scale = search_radius * texel_size;

    // Use 16 samples for blocker search (balance between quality and performance).
    for (var i = 0u; i < 16u; i = i + 1u) {
        let base_offset = poisson_disk_16(i);
        let rotated_offset = rotate_sample(base_offset, rotation);
        let sample_uv = uv + rotated_offset * search_scale;

        // Sample shadow map depth using textureLoad for blocker search.
        // We need the raw depth value, not a comparison result.
        let shadow_dims = textureDimensions(shadow_map, 0);
        let sample_coord = vec2<i32>(sample_uv * vec2<f32>(shadow_dims));

        // Bounds check to avoid sampling outside the texture.
        if sample_coord.x >= 0 && sample_coord.x < i32(shadow_dims.x) &&
           sample_coord.y >= 0 && sample_coord.y < i32(shadow_dims.y) {
            let sample_depth = textureLoad(shadow_map, sample_coord, 0);

            // If this sample is closer than the receiver, it's a blocker.
            if sample_depth < receiver_depth - PCSS_EPSILON {
                blocker_sum = blocker_sum + sample_depth;
                blocker_count = blocker_count + 1.0;
            }
        }
    }

    // Return average blocker depth and count.
    // Guard against division by zero.
    let avg_depth = select(0.0, blocker_sum / blocker_count, blocker_count > 0.0);
    return vec2<f32>(avg_depth, blocker_count);
}

// ============================================================================
// Step 2: Penumbra Estimation
// ============================================================================

/// Estimates penumbra width using similar triangles.
///
/// The penumbra width is derived from the geometric relationship:
///   penumbra = light_size * (d_receiver - d_blocker) / d_blocker
///
/// This models how a larger light source or greater blocker-receiver
/// distance produces softer shadows.
///
/// Parameters:
/// - receiver_depth: Depth of the receiving surface
/// - blocker_depth: Average depth of blockers
/// - light_size: Virtual light size (affects penumbra scaling)
///
/// Returns: Estimated penumbra width in light-space units.
fn estimate_penumbra(
    receiver_depth: f32,
    blocker_depth: f32,
    light_size: f32
) -> f32 {
    // Guard against division by zero and negative depths.
    let safe_blocker = max(blocker_depth, PCSS_EPSILON);
    let depth_diff = max(receiver_depth - safe_blocker, 0.0);

    // Similar triangles: penumbra width = light_size * depth_diff / blocker_depth
    return light_size * depth_diff / safe_blocker;
}

// ============================================================================
// Step 3: Variable-Radius PCF
// ============================================================================

/// Performs PCF filtering with a variable kernel radius.
///
/// Parameters:
/// - shadow_map: Depth texture for shadow sampling
/// - shadow_sampler: Comparison sampler for depth testing
/// - uv: Shadow map UV coordinates
/// - depth: Biased receiver depth for comparison
/// - filter_radius: PCF kernel radius in texels
/// - texel_size: Size of one texel in UV space
/// - rotation: Per-pixel rotation angle
///
/// Returns: Shadow factor in [0, 1] (1 = fully lit, 0 = fully shadowed).
fn pcf_variable_radius(
    shadow_map: texture_depth_2d,
    shadow_sampler: sampler_comparison,
    uv: vec2<f32>,
    depth: f32,
    filter_radius: f32,
    texel_size: f32,
    rotation: f32
) -> f32 {
    var shadow = 0.0;
    let filter_scale = filter_radius * texel_size;

    // Use 32 samples for final PCF (high quality).
    for (var i = 0u; i < 32u; i = i + 1u) {
        let base_offset = poisson_disk_32(i);
        let rotated_offset = rotate_sample(base_offset, rotation);
        let sample_uv = uv + rotated_offset * filter_scale;

        shadow = shadow + textureSampleCompare(
            shadow_map,
            shadow_sampler,
            sample_uv,
            depth
        );
    }

    return shadow / 32.0;
}

// ============================================================================
// Main PCSS Function
// ============================================================================

/// Result structure for PCSS sampling with debug information.
struct PcssSampleResult {
    /// Shadow factor in [0, 1]: 0 = fully shadowed, 1 = fully lit.
    factor: f32,
    /// Number of PCF samples taken.
    sample_count: u32,
    /// Average blocker depth found.
    avg_blocker_depth: f32,
    /// Estimated penumbra size.
    penumbra_size: f32,
    /// Number of blockers found in search.
    blocker_count: f32,
    /// Final filter radius used.
    filter_radius: f32,
}

/// Computes PCSS shadow factor using the 3-step algorithm.
///
/// Parameters:
/// - shadow_map: Depth texture for shadow sampling
/// - shadow_sampler: Comparison sampler for depth testing
/// - uv: Shadow map UV coordinates (in atlas space)
/// - depth: Receiver depth in light space [0, 1]
/// - params: PCSS configuration parameters
/// - screen_pos: Screen-space position for per-pixel rotation
///
/// Returns: Complete PCSS result with shadow factor and debug info.
fn pcss_shadow(
    shadow_map: texture_depth_2d,
    shadow_sampler: sampler_comparison,
    uv: vec2<f32>,
    depth: f32,
    params: PcssParams,
    screen_pos: vec2<f32>
) -> PcssSampleResult {
    var result: PcssSampleResult;
    result.sample_count = 0u;
    result.avg_blocker_depth = 0.0;
    result.penumbra_size = 0.0;
    result.blocker_count = 0.0;
    result.filter_radius = params.min_filter_radius;

    // Get texture dimensions for texel size calculation.
    let shadow_dims = textureDimensions(shadow_map, 0);
    let texel_size = 1.0 / f32(max(shadow_dims.x, shadow_dims.y));

    // Generate per-pixel rotation for sample randomization.
    let rotation = generate_rotation(screen_pos);

    // Step 1: Blocker search
    let blocker_result = find_blocker_depth(
        shadow_map,
        uv,
        depth,
        params.blocker_search_radius,
        texel_size,
        rotation
    );

    result.avg_blocker_depth = blocker_result.x;
    result.blocker_count = blocker_result.y;

    // No blockers found = fully lit
    if blocker_result.y < PCSS_MIN_BLOCKERS {
        result.factor = 1.0;
        result.sample_count = 16u; // Only blocker search samples
        return result;
    }

    // Step 2: Penumbra estimation
    let penumbra = estimate_penumbra(
        depth,
        blocker_result.x,
        params.light_size
    );

    result.penumbra_size = penumbra;

    // Convert penumbra to filter radius and clamp to valid range.
    let filter_radius = clamp(
        penumbra / texel_size,
        params.min_filter_radius,
        params.max_filter_radius
    );

    result.filter_radius = filter_radius;

    // Step 3: Variable-radius PCF
    let shadow_factor = pcf_variable_radius(
        shadow_map,
        shadow_sampler,
        uv,
        depth,
        filter_radius,
        texel_size,
        rotation
    );

    result.factor = shadow_factor;
    result.sample_count = 16u + 32u; // Blocker search + PCF

    return result;
}

/// Simplified PCSS function returning only the shadow factor.
///
/// Use this for production rendering where debug info is not needed.
fn pcss_shadow_simple(
    shadow_map: texture_depth_2d,
    shadow_sampler: sampler_comparison,
    uv: vec2<f32>,
    depth: f32,
    params: PcssParams,
    screen_pos: vec2<f32>
) -> f32 {
    return pcss_shadow(shadow_map, shadow_sampler, uv, depth, params, screen_pos).factor;
}

// ============================================================================
// Tiled Shadow Atlas Support
// ============================================================================

/// Shadow tile information for atlas-based shadow maps.
/// Re-exported for compatibility with shadow_common.wgsl types.
struct ShadowTileInfo {
    /// Atlas UV offset for this tile.
    uv_offset: vec2<f32>,
    /// Atlas UV scale for this tile.
    uv_scale: vec2<f32>,
    /// Light-space transformation matrix.
    light_space_matrix: mat4x4<f32>,
    /// Cascade or light index.
    cascade_index: u32,
    /// PCF kernel size hint.
    filter_size: f32,
    /// Constant depth bias.
    bias_constant: f32,
    /// Slope-scaled depth bias.
    bias_slope: f32,
}

/// PCSS shadow sampling with tile support for shadow atlases.
///
/// Parameters:
/// - shadow_map: Shadow atlas depth texture
/// - shadow_sampler: Comparison sampler
/// - world_pos: World-space position to shadow
/// - normal: World-space surface normal
/// - light_dir: Direction TO the light (normalized)
/// - tile: Shadow tile information
/// - params: PCSS parameters
/// - screen_pos: Screen position for rotation
///
/// Returns: PCSS result with shadow factor and debug info.
fn pcss_shadow_tiled(
    shadow_map: texture_depth_2d,
    shadow_sampler: sampler_comparison,
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    light_dir: vec3<f32>,
    tile: ShadowTileInfo,
    params: PcssParams,
    screen_pos: vec2<f32>
) -> PcssSampleResult {
    // Transform world position to light clip space.
    let light_clip = tile.light_space_matrix * vec4<f32>(world_pos, 1.0);

    // Perspective divide.
    let ndc = light_clip.xyz / light_clip.w;

    // Transform NDC [-1, 1] to UV [0, 1].
    let local_uv = ndc.xy * 0.5 + 0.5;

    // Apply atlas tile transform.
    let atlas_uv = local_uv * tile.uv_scale + tile.uv_offset;

    // Compute slope-scaled bias.
    let cos_theta = max(dot(normal, light_dir), PCSS_EPSILON);
    let sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    let slope = sin_theta / cos_theta;
    let bias = tile.bias_constant + tile.bias_slope * slope;

    // Apply bias to depth.
    let biased_depth = ndc.z - bias;

    // Clamp depth to valid range.
    let clamped_depth = clamp(biased_depth, 0.0, 1.0);

    // Sample PCSS.
    return pcss_shadow(
        shadow_map,
        shadow_sampler,
        atlas_uv,
        clamped_depth,
        params,
        screen_pos
    );
}

// ============================================================================
// Adaptive Quality PCSS
// ============================================================================

/// Adaptive PCSS that adjusts sample count based on penumbra size.
///
/// Uses fewer samples for small penumbras (sharp shadows) and more
/// samples for large penumbras (soft shadows) to optimize performance.
///
/// Parameters:
/// - shadow_map: Depth texture
/// - shadow_sampler: Comparison sampler
/// - uv: Shadow map UV coordinates
/// - depth: Receiver depth
/// - params: PCSS parameters
/// - screen_pos: Screen position for rotation
///
/// Returns: Shadow factor in [0, 1].
fn pcss_shadow_adaptive(
    shadow_map: texture_depth_2d,
    shadow_sampler: sampler_comparison,
    uv: vec2<f32>,
    depth: f32,
    params: PcssParams,
    screen_pos: vec2<f32>
) -> f32 {
    let shadow_dims = textureDimensions(shadow_map, 0);
    let texel_size = 1.0 / f32(max(shadow_dims.x, shadow_dims.y));
    let rotation = generate_rotation(screen_pos);

    // Blocker search with reduced samples.
    let blocker_result = find_blocker_depth(
        shadow_map,
        uv,
        depth,
        params.blocker_search_radius,
        texel_size,
        rotation
    );

    // No blockers = fully lit.
    if blocker_result.y < PCSS_MIN_BLOCKERS {
        return 1.0;
    }

    // Estimate penumbra.
    let penumbra = estimate_penumbra(depth, blocker_result.x, params.light_size);
    let filter_radius = clamp(
        penumbra / texel_size,
        params.min_filter_radius,
        params.max_filter_radius
    );

    // Determine sample count based on filter radius.
    // Small radius (< 4 texels): 8 samples
    // Medium radius (4-16 texels): 16 samples
    // Large radius (> 16 texels): 32 samples
    var shadow = 0.0;
    let filter_scale = filter_radius * texel_size;

    if filter_radius < 4.0 {
        // Low quality: 8 samples.
        for (var i = 0u; i < 8u; i = i + 1u) {
            let base_offset = poisson_disk_16(i * 2u);
            let rotated_offset = rotate_sample(base_offset, rotation);
            let sample_uv = uv + rotated_offset * filter_scale;
            shadow = shadow + textureSampleCompare(shadow_map, shadow_sampler, sample_uv, depth);
        }
        shadow = shadow / 8.0;
    } else if filter_radius < 16.0 {
        // Medium quality: 16 samples.
        for (var i = 0u; i < 16u; i = i + 1u) {
            let base_offset = poisson_disk_16(i);
            let rotated_offset = rotate_sample(base_offset, rotation);
            let sample_uv = uv + rotated_offset * filter_scale;
            shadow = shadow + textureSampleCompare(shadow_map, shadow_sampler, sample_uv, depth);
        }
        shadow = shadow / 16.0;
    } else {
        // High quality: 32 samples.
        for (var i = 0u; i < 32u; i = i + 1u) {
            let base_offset = poisson_disk_32(i);
            let rotated_offset = rotate_sample(base_offset, rotation);
            let sample_uv = uv + rotated_offset * filter_scale;
            shadow = shadow + textureSampleCompare(shadow_map, shadow_sampler, sample_uv, depth);
        }
        shadow = shadow / 32.0;
    }

    return shadow;
}

// ============================================================================
// Default Parameters
// ============================================================================

/// Returns default PCSS parameters for typical use cases.
fn pcss_default_params() -> PcssParams {
    var params: PcssParams;
    params.light_size = 0.04;
    params.blocker_search_radius = 16.0;
    params.min_filter_radius = 1.0;
    params.max_filter_radius = 32.0;
    return params;
}

/// Returns PCSS parameters optimized for directional lights (sun).
fn pcss_sun_params() -> PcssParams {
    var params: PcssParams;
    params.light_size = 0.02;           // Smaller = sharper sun shadows
    params.blocker_search_radius = 24.0;
    params.min_filter_radius = 0.5;
    params.max_filter_radius = 48.0;
    return params;
}

/// Returns PCSS parameters for area lights.
fn pcss_area_light_params(light_radius: f32) -> PcssParams {
    var params: PcssParams;
    params.light_size = light_radius;   // Match physical light size
    params.blocker_search_radius = 32.0;
    params.min_filter_radius = 2.0;
    params.max_filter_radius = 64.0;
    return params;
}
