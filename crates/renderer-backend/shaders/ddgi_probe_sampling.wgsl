// SPDX-License-Identifier: MIT
//
// ddgi_probe_sampling.wgsl -- DDGI probe irradiance sampling with L2 spherical harmonics.
//
// This shader samples the DDGI probe grid to compute indirect irradiance at shading points.
// Features:
//   - Trilinear interpolation between 8 surrounding probes
//   - Visibility-weighted sampling to avoid light leaking
//   - Parallax correction for off-center sampling
//   - L2 (9 coefficient) spherical harmonics for high-quality irradiance
//   - Infinite scrolling grid support via wrap-around indexing
//
// Entry points:
//   - ddgi_sample_irradiance: Compute shader for full-screen irradiance sampling
//
// Reference: McGuire et al., "Dynamic Diffuse Global Illumination with Ray-Traced
//            Irradiance Fields", JCGT 2017

// ============================================================================
// Constants
// ============================================================================

const PI: f32 = 3.14159265358979323846;
const EPSILON: f32 = 0.0001;

// Visibility thresholds
const VIS_BACKFACE_THRESHOLD: f32 = 0.1;
const VIS_CHEBYSHEV_BIAS: f32 = 0.25;
const VIS_MIN_WEIGHT: f32 = 0.0001;

// SH basis constants (matching spherical_harmonics.wgsl)
const SH_Y00: f32 = 0.28209479177387814;
const SH_Y1: f32 = 0.4886025119029199;
const SH_Y2_NEG2: f32 = 1.0925484305920792;
const SH_Y2_NEG1: f32 = 1.0925484305920792;
const SH_Y2_0: f32 = 0.31539156525252005;
const SH_Y2_POS1: f32 = 1.0925484305920792;
const SH_Y2_POS2: f32 = 0.5462742152960396;

// Cosine lobe convolution (for irradiance from radiance SH)
const SH_A0: f32 = 1.0;
const SH_A1: f32 = 0.6666666666666666;
const SH_A2: f32 = 0.25;

// ============================================================================
// Data Structures
// ============================================================================

/// GPU probe grid metadata (matches ProbeGridGpu in Rust).
struct ProbeGridGpu {
    origin: vec3<f32>,
    _pad0: f32,
    cell_size: vec3<f32>,
    _pad1: f32,
    dimensions: vec3<u32>,
    total_probes: u32,
    scroll_offset: vec3<i32>,
    frame_index: u32,
}

/// Per-probe SH irradiance storage (L2, 9 RGB coefficients).
/// Stored as vec4 array for GPU alignment (w component unused).
struct ProbeSH {
    coeffs: array<vec4<f32>, 9>,
    visibility: array<f32, 9>,
    _pad: array<f32, 3>,
}

/// Probe sampling weight and data for trilinear interpolation.
struct ProbeWeight {
    index: u32,
    position: vec3<f32>,
    trilinear_weight: f32,
}

/// Camera uniforms for view-dependent effects.
struct CameraUniforms {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    inv_projection: mat4x4<f32>,
    camera_position: vec3<f32>,
    _pad: f32,
}

// ============================================================================
// Bindings
// ============================================================================

@group(0) @binding(0) var<uniform> grid: ProbeGridGpu;
@group(0) @binding(1) var<storage, read> probes: array<ProbeSH>;
@group(0) @binding(2) var<uniform> camera: CameraUniforms;

// G-buffer inputs
@group(0) @binding(3) var world_position_texture: texture_2d<f32>;
@group(0) @binding(4) var world_normal_texture: texture_2d<f32>;
@group(0) @binding(5) var depth_texture: texture_depth_2d;

// Output
@group(0) @binding(6) var irradiance_output: texture_storage_2d<rgba16float, write>;

// Optional: distance texture for visibility testing
@group(0) @binding(7) var probe_distance_texture: texture_2d<f32>;
@group(0) @binding(8) var linear_sampler: sampler;

// ============================================================================
// SH Evaluation (L2)
// ============================================================================

/// Evaluate all 9 SH basis functions at a direction.
fn sh_basis_l2(dir: vec3<f32>) -> array<f32, 9> {
    let x = dir.x;
    let y = dir.y;
    let z = dir.z;

    var basis: array<f32, 9>;

    // L=0
    basis[0] = SH_Y00;

    // L=1
    basis[1] = SH_Y1 * y;
    basis[2] = SH_Y1 * z;
    basis[3] = SH_Y1 * x;

    // L=2
    basis[4] = SH_Y2_NEG2 * x * y;
    basis[5] = SH_Y2_NEG1 * y * z;
    basis[6] = SH_Y2_0 * (3.0 * z * z - 1.0);
    basis[7] = SH_Y2_POS1 * x * z;
    basis[8] = SH_Y2_POS2 * (x * x - y * y);

    return basis;
}

/// Evaluate SH irradiance at a direction using L2 coefficients.
/// Includes cosine lobe convolution for proper irradiance computation.
fn sh_evaluate_irradiance_l2(probe: ProbeSH, dir: vec3<f32>) -> vec3<f32> {
    let basis = sh_basis_l2(dir);

    var irradiance = vec3<f32>(0.0);

    // L0 band (1 coefficient)
    irradiance += probe.coeffs[0].xyz * basis[0] * SH_A0;

    // L1 band (3 coefficients)
    irradiance += probe.coeffs[1].xyz * basis[1] * SH_A1;
    irradiance += probe.coeffs[2].xyz * basis[2] * SH_A1;
    irradiance += probe.coeffs[3].xyz * basis[3] * SH_A1;

    // L2 band (5 coefficients)
    irradiance += probe.coeffs[4].xyz * basis[4] * SH_A2;
    irradiance += probe.coeffs[5].xyz * basis[5] * SH_A2;
    irradiance += probe.coeffs[6].xyz * basis[6] * SH_A2;
    irradiance += probe.coeffs[7].xyz * basis[7] * SH_A2;
    irradiance += probe.coeffs[8].xyz * basis[8] * SH_A2;

    return max(irradiance, vec3<f32>(0.0));
}

/// Evaluate raw SH (without irradiance convolution) - for debugging.
fn sh_evaluate_l2(probe: ProbeSH, dir: vec3<f32>) -> vec3<f32> {
    let basis = sh_basis_l2(dir);

    var result = vec3<f32>(0.0);
    for (var i = 0u; i < 9u; i++) {
        result += probe.coeffs[i].xyz * basis[i];
    }

    return result;
}

// ============================================================================
// Grid Indexing
// ============================================================================

/// Convert world position to grid coordinates (floating-point).
fn world_to_grid(world_pos: vec3<f32>) -> vec3<f32> {
    return (world_pos - grid.origin) / grid.cell_size;
}

/// Convert grid coordinates to world position.
fn grid_to_world(grid_idx: vec3<u32>) -> vec3<f32> {
    return grid.origin + vec3<f32>(grid_idx) * grid.cell_size;
}

/// Apply scroll offset with wraparound.
fn apply_scroll(grid_idx: vec3<i32>) -> vec3<u32> {
    let scrolled = grid_idx + grid.scroll_offset;
    let dims = vec3<i32>(grid.dimensions);

    // Euclidean modulo for proper negative handling
    let wrapped = vec3<i32>(
        ((scrolled.x % dims.x) + dims.x) % dims.x,
        ((scrolled.y % dims.y) + dims.y) % dims.y,
        ((scrolled.z % dims.z) + dims.z) % dims.z,
    );

    return vec3<u32>(wrapped);
}

/// Convert 3D grid index to linear buffer index.
fn grid_to_linear(grid_idx: vec3<u32>) -> u32 {
    return grid_idx.x + grid_idx.y * grid.dimensions.x +
           grid_idx.z * grid.dimensions.x * grid.dimensions.y;
}

/// Clamp grid index to valid range.
fn clamp_grid_index(idx: vec3<i32>) -> vec3<u32> {
    return vec3<u32>(
        clamp(idx.x, 0i, i32(grid.dimensions.x) - 1i),
        clamp(idx.y, 0i, i32(grid.dimensions.y) - 1i),
        clamp(idx.z, 0i, i32(grid.dimensions.z) - 1i),
    );
}

// ============================================================================
// Probe Sampling
// ============================================================================

/// Find 8 surrounding probes and compute trilinear weights.
/// Returns array of 8 ProbeWeight structures.
fn get_surrounding_probes(world_pos: vec3<f32>) -> array<ProbeWeight, 8> {
    let grid_pos = world_to_grid(world_pos);

    // Base cell (floor)
    let base_idx = vec3<i32>(floor(grid_pos));

    // Fractional position within cell [0, 1]
    let frac = clamp(grid_pos - vec3<f32>(base_idx), vec3<f32>(0.0), vec3<f32>(1.0));

    var weights: array<ProbeWeight, 8>;

    // Iterate over 8 corners of the cell
    for (var i = 0u; i < 8u; i++) {
        let offset = vec3<i32>(
            i32(i & 1u),
            i32((i >> 1u) & 1u),
            i32((i >> 2u) & 1u),
        );

        let corner_idx = base_idx + offset;

        // Clamp to grid bounds (or wrap with scroll)
        let clamped = clamp_grid_index(corner_idx);
        let scrolled = apply_scroll(vec3<i32>(clamped));

        // Compute trilinear weight
        let fx = select(1.0 - frac.x, frac.x, offset.x == 1);
        let fy = select(1.0 - frac.y, frac.y, offset.y == 1);
        let fz = select(1.0 - frac.z, frac.z, offset.z == 1);

        weights[i].index = grid_to_linear(scrolled);
        weights[i].position = grid_to_world(clamped);
        weights[i].trilinear_weight = fx * fy * fz;
    }

    return weights;
}

/// Compute visibility weight for a probe relative to shading point.
/// Uses multiple heuristics to reduce light leaking:
///   1. Backface rejection: reduce weight if probe is behind surface
///   2. Distance-based falloff: reduce weight for distant probes
///   3. Chebyshev visibility: use probe depth statistics if available
fn evaluate_visibility(
    probe_pos: vec3<f32>,
    shading_pos: vec3<f32>,
    normal: vec3<f32>,
) -> f32 {
    let to_probe = probe_pos - shading_pos;
    let probe_dist = length(to_probe);
    let probe_dir = to_probe / max(probe_dist, EPSILON);

    // Backface rejection: if probe is behind the surface, reduce weight
    let n_dot_d = dot(normal, probe_dir);
    let backface_weight = smoothstep(-VIS_BACKFACE_THRESHOLD, 0.0, n_dot_d);

    // Distance-based falloff (optional, based on cell size)
    let cell_diag = length(grid.cell_size);
    let dist_weight = saturate(1.0 - probe_dist / (cell_diag * 2.0));

    // Combine weights
    var weight = backface_weight;

    // Apply distance falloff only for probes far from expected position
    weight *= mix(1.0, dist_weight, 0.3);

    return max(weight, VIS_MIN_WEIGHT);
}

/// Chebyshev visibility test using probe distance statistics.
/// Returns visibility probability based on mean/variance of ray hit distances.
fn chebyshev_visibility(
    probe_idx: u32,
    direction: vec3<f32>,
    distance: f32,
) -> f32 {
    // TODO: Sample from distance texture octahedron map
    // For now, return full visibility
    return 1.0;
}

// ============================================================================
// Parallax Correction
// ============================================================================

/// Apply parallax correction for off-center sampling.
/// Adjusts the sampling direction to account for the offset between
/// probe position and shading point.
fn parallax_correct_direction(
    probe_pos: vec3<f32>,
    shading_pos: vec3<f32>,
    direction: vec3<f32>,
    proxy_distance: f32,
) -> vec3<f32> {
    // Vector from probe to shading point
    let offset = shading_pos - probe_pos;

    // Project offset onto the hemisphere
    // Use a spherical proxy centered at the probe
    let proj_dist = proxy_distance;

    // Compute corrected direction
    // The idea: trace from shading point in `direction` and find intersection
    // with probe's proxy sphere, then use direction from probe to that point.

    // Simplified: just offset the direction slightly
    // Full implementation would use ray-sphere intersection

    let corrected = normalize(direction * proj_dist + offset);

    return corrected;
}

/// Full parallax correction with ray-sphere intersection.
fn parallax_correct_direction_full(
    probe_pos: vec3<f32>,
    shading_pos: vec3<f32>,
    direction: vec3<f32>,
    proxy_radius: f32,
) -> vec3<f32> {
    // Ray: P = shading_pos + t * direction
    // Sphere: |P - probe_pos|^2 = proxy_radius^2
    //
    // Substituting and solving for t:
    // |shading_pos - probe_pos + t * direction|^2 = proxy_radius^2
    // Let d = shading_pos - probe_pos
    // |d + t * dir|^2 = r^2
    // t^2 + 2*t*(d.dir) + |d|^2 - r^2 = 0
    // t = -d.dir +/- sqrt((d.dir)^2 - |d|^2 + r^2)

    let d = shading_pos - probe_pos;
    let d_dot_dir = dot(d, direction);
    let d_sq = dot(d, d);
    let r_sq = proxy_radius * proxy_radius;

    let discriminant = d_dot_dir * d_dot_dir - d_sq + r_sq;

    if (discriminant < 0.0) {
        // No intersection, return original direction
        return direction;
    }

    // Take the positive root (intersection in front)
    let t = -d_dot_dir + sqrt(discriminant);

    if (t < 0.0) {
        return direction;
    }

    // Intersection point on sphere
    let hit_point = shading_pos + t * direction;

    // Direction from probe to hit point
    let corrected = normalize(hit_point - probe_pos);

    return corrected;
}

// ============================================================================
// Main Sampling Function
// ============================================================================

/// Sample DDGI irradiance at a shading point.
/// Uses trilinear interpolation with visibility weighting.
fn sample_ddgi_irradiance(
    world_pos: vec3<f32>,
    normal: vec3<f32>,
) -> vec3<f32> {
    // Get surrounding 8 probes with trilinear weights
    let probe_weights = get_surrounding_probes(world_pos);

    var irradiance = vec3<f32>(0.0);
    var total_weight = 0.0;

    // Accumulate weighted irradiance from all probes
    for (var i = 0u; i < 8u; i++) {
        let pw = probe_weights[i];

        // Skip if trilinear weight is negligible
        if (pw.trilinear_weight < EPSILON) {
            continue;
        }

        // Evaluate visibility weight
        let vis_weight = evaluate_visibility(pw.position, world_pos, normal);

        // Combined weight
        let weight = pw.trilinear_weight * vis_weight;

        if (weight < VIS_MIN_WEIGHT) {
            continue;
        }

        // Get probe data
        let probe = probes[pw.index];

        // Optional: apply parallax correction
        let proxy_radius = length(grid.cell_size) * 0.5;
        let corrected_normal = parallax_correct_direction_full(
            pw.position,
            world_pos,
            normal,
            proxy_radius,
        );

        // Evaluate SH irradiance in normal direction
        let probe_irradiance = sh_evaluate_irradiance_l2(probe, corrected_normal);

        irradiance += probe_irradiance * weight;
        total_weight += weight;
    }

    // Normalize by total weight
    if (total_weight > EPSILON) {
        irradiance /= total_weight;
    }

    return irradiance;
}

/// Sample DDGI irradiance without visibility weighting (faster, more light leaking).
fn sample_ddgi_irradiance_simple(
    world_pos: vec3<f32>,
    normal: vec3<f32>,
) -> vec3<f32> {
    let probe_weights = get_surrounding_probes(world_pos);

    var irradiance = vec3<f32>(0.0);
    var total_weight = 0.0;

    for (var i = 0u; i < 8u; i++) {
        let pw = probe_weights[i];

        if (pw.trilinear_weight < EPSILON) {
            continue;
        }

        let probe = probes[pw.index];
        let probe_irradiance = sh_evaluate_irradiance_l2(probe, normal);

        irradiance += probe_irradiance * pw.trilinear_weight;
        total_weight += pw.trilinear_weight;
    }

    return irradiance / max(total_weight, EPSILON);
}

// ============================================================================
// Compute Shader Entry Points
// ============================================================================

/// Main entry point: sample DDGI irradiance for each pixel.
@compute @workgroup_size(8, 8, 1)
fn ddgi_sample_irradiance(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;

    // Read world position from G-buffer
    let world_pos_sample = textureLoad(world_position_texture, vec2<i32>(i32(px), i32(py)), 0);
    let world_pos = world_pos_sample.xyz;

    // Early out for sky/background pixels
    if (world_pos_sample.w < 0.5) {
        textureStore(irradiance_output, vec2<i32>(i32(px), i32(py)), vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    // Read world normal from G-buffer
    let world_normal = normalize(textureLoad(world_normal_texture, vec2<i32>(i32(px), i32(py)), 0).xyz);

    // Sample DDGI irradiance
    let irradiance = sample_ddgi_irradiance(world_pos, world_normal);

    // Write result
    textureStore(irradiance_output, vec2<i32>(i32(px), i32(py)), vec4<f32>(irradiance, 1.0));
}

/// Fast path entry point: trilinear only, no visibility weighting.
@compute @workgroup_size(8, 8, 1)
fn ddgi_sample_irradiance_fast(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;

    let world_pos_sample = textureLoad(world_position_texture, vec2<i32>(i32(px), i32(py)), 0);
    let world_pos = world_pos_sample.xyz;

    if (world_pos_sample.w < 0.5) {
        textureStore(irradiance_output, vec2<i32>(i32(px), i32(py)), vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    let world_normal = normalize(textureLoad(world_normal_texture, vec2<i32>(i32(px), i32(py)), 0).xyz);
    let irradiance = sample_ddgi_irradiance_simple(world_pos, world_normal);

    textureStore(irradiance_output, vec2<i32>(i32(px), i32(py)), vec4<f32>(irradiance, 1.0));
}

// ============================================================================
// Debug Entry Points
// ============================================================================

/// Debug: visualize probe grid weights.
@compute @workgroup_size(8, 8, 1)
fn ddgi_debug_weights(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;

    let world_pos_sample = textureLoad(world_position_texture, vec2<i32>(i32(px), i32(py)), 0);
    let world_pos = world_pos_sample.xyz;

    if (world_pos_sample.w < 0.5) {
        textureStore(irradiance_output, vec2<i32>(i32(px), i32(py)), vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    let world_normal = normalize(textureLoad(world_normal_texture, vec2<i32>(i32(px), i32(py)), 0).xyz);
    let probe_weights = get_surrounding_probes(world_pos);

    // Sum trilinear weights (should be ~1.0)
    var total_trilinear = 0.0;
    var total_visibility = 0.0;

    for (var i = 0u; i < 8u; i++) {
        let pw = probe_weights[i];
        total_trilinear += pw.trilinear_weight;
        total_visibility += evaluate_visibility(pw.position, world_pos, world_normal) * pw.trilinear_weight;
    }

    // Output: R=trilinear sum, G=visibility weighted sum, B=probe count
    textureStore(
        irradiance_output,
        vec2<i32>(i32(px), i32(py)),
        vec4<f32>(total_trilinear, total_visibility, 8.0 / 8.0, 1.0)
    );
}

/// Debug: visualize which probe cell a pixel falls into.
@compute @workgroup_size(8, 8, 1)
fn ddgi_debug_grid_cell(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;

    let world_pos_sample = textureLoad(world_position_texture, vec2<i32>(i32(px), i32(py)), 0);
    let world_pos = world_pos_sample.xyz;

    if (world_pos_sample.w < 0.5) {
        textureStore(irradiance_output, vec2<i32>(i32(px), i32(py)), vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    let grid_pos = world_to_grid(world_pos);
    let cell_idx = vec3<i32>(floor(grid_pos));
    let frac = grid_pos - vec3<f32>(cell_idx);

    // Color by cell index (mod for visibility)
    let color = vec3<f32>(
        f32((cell_idx.x % 8) + 1) / 8.0,
        f32((cell_idx.y % 4) + 1) / 4.0,
        f32((cell_idx.z % 8) + 1) / 8.0,
    );

    textureStore(irradiance_output, vec2<i32>(i32(px), i32(py)), vec4<f32>(color, 1.0));
}
