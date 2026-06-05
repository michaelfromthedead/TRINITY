// SPDX-License-Identifier: MIT
//
// ddgi_rasterized_to_sh.comp.wgsl -- Convert rasterized cubemap atlas to SH coefficients.
//
// This shader reads 6 cubemap faces from a probe atlas texture and projects
// the captured radiance to L2 spherical harmonics coefficients. It's the
// compute pass that completes the rasterized DDGI fallback pipeline.
//
// Workgroup: 64 threads (one per probe in a batch)
// Dispatch: ceil(probe_count / 64) workgroups
//
// Atlas Layout:
//   - 8x8 probe grid (default), each probe = 3x2 faces
//   - Face order within probe cell:
//     Row 0: +X, -X, +Y
//     Row 1: -Y, +Z, -Z

// ============================================================================
// Constants
// ============================================================================

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;

// SH basis constants
const SH_Y00: f32 = 0.28209479177387814;      // sqrt(1/(4*PI))
const SH_Y1: f32 = 0.4886025119029199;         // sqrt(3/(4*PI))
const SH_Y2_NEG2: f32 = 1.0925484305920792;    // sqrt(15/(4*PI))
const SH_Y2_NEG1: f32 = 1.0925484305920792;    // sqrt(15/(4*PI))
const SH_Y2_0: f32 = 0.31539156525252005;      // sqrt(5/(16*PI))
const SH_Y2_POS1: f32 = 1.0925484305920792;    // sqrt(15/(4*PI))
const SH_Y2_POS2: f32 = 0.5462742152960396;    // sqrt(15/(16*PI))

// Irradiance convolution coefficients
const SH_A0: f32 = 1.0;
const SH_A1: f32 = 0.6666666666666666;
const SH_A2: f32 = 0.25;

// Atlas configuration
const FACES_PER_ROW: u32 = 3u;
const FACE_ROWS: u32 = 2u;
const NUM_FACES: u32 = 6u;

// ============================================================================
// Bindings
// ============================================================================

struct ProbeProjectionParams {
    // Atlas dimensions
    atlas_width: u32,
    atlas_height: u32,
    // Probe cell dimensions (face_resolution * 3, face_resolution * 2)
    probe_cell_width: u32,
    probe_cell_height: u32,
    // Face resolution
    face_resolution: u32,
    // Number of probes per atlas row
    probes_per_row: u32,
    // Batch info
    batch_start: u32,
    probe_count: u32,
    // Hysteresis blend factor (0 = full replace, 1 = keep old)
    hysteresis: f32,
    // Padding
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

// Per-probe SH output (matches ProbeSH in Rust)
struct ProbeSH {
    // L2 irradiance coefficients (9 RGB values as vec4 with padding)
    irradiance: array<vec4<f32>, 9>,
    // L2 visibility coefficients
    visibility: array<f32, 9>,
    // Padding
    _pad: array<f32, 3>,
}

@group(0) @binding(0) var<uniform> params: ProbeProjectionParams;
@group(0) @binding(1) var atlas_texture: texture_2d<f32>;
@group(0) @binding(2) var atlas_sampler: sampler;
@group(0) @binding(3) var<storage, read_write> probe_sh: array<ProbeSH>;

// Optional: depth atlas for visibility computation
@group(0) @binding(4) var depth_atlas: texture_2d<f32>;

// ============================================================================
// Helper Functions
// ============================================================================

/// Get UV coordinates for a pixel in a specific face of a specific probe.
fn get_face_pixel_uv(
    probe_idx: u32,
    face: u32,
    pixel_x: u32,
    pixel_y: u32
) -> vec2<f32> {
    // Probe position in atlas grid
    let probe_x = probe_idx % params.probes_per_row;
    let probe_y = probe_idx / params.probes_per_row;

    // Face position within probe cell
    let face_col = face % FACES_PER_ROW;
    let face_row = face / FACES_PER_ROW;

    // Pixel position in atlas
    let atlas_x = probe_x * params.probe_cell_width
                + face_col * params.face_resolution
                + pixel_x;
    let atlas_y = probe_y * params.probe_cell_height
                + face_row * params.face_resolution
                + pixel_y;

    // Convert to UV (add 0.5 for pixel center)
    return vec2<f32>(
        (f32(atlas_x) + 0.5) / f32(params.atlas_width),
        (f32(atlas_y) + 0.5) / f32(params.atlas_height)
    );
}

/// Convert face-local UV to world direction.
/// Face order: 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z
fn face_uv_to_direction(face: u32, u: f32, v: f32) -> vec3<f32> {
    // Convert UV to [-1, 1] range
    let uc = 2.0 * u - 1.0;
    let vc = 2.0 * v - 1.0;

    var dir: vec3<f32>;
    switch face {
        case 0u: { dir = vec3<f32>(1.0, -vc, -uc); }   // +X
        case 1u: { dir = vec3<f32>(-1.0, -vc, uc); }   // -X
        case 2u: { dir = vec3<f32>(uc, 1.0, vc); }     // +Y
        case 3u: { dir = vec3<f32>(uc, -1.0, -vc); }   // -Y
        case 4u: { dir = vec3<f32>(uc, -vc, 1.0); }    // +Z
        case 5u: { dir = vec3<f32>(-uc, -vc, -1.0); }  // -Z
        default: { dir = vec3<f32>(0.0, 0.0, 1.0); }
    }

    return normalize(dir);
}

/// Compute solid angle for a cubemap texel.
fn texel_solid_angle(u: f32, v: f32) -> f32 {
    // Convert to [-1, 1] centered coordinates
    let uc = 2.0 * u - 1.0;
    let vc = 2.0 * v - 1.0;

    // Texel size
    let texel_size = 2.0 / f32(params.face_resolution);

    // Area element approximation
    let d = 1.0 + uc * uc + vc * vc;
    return texel_size * texel_size / (d * sqrt(d));
}

/// Evaluate L2 SH basis functions at a direction.
fn sh_basis_l2(dir: vec3<f32>) -> array<f32, 9> {
    let x = dir.x;
    let y = dir.y;
    let z = dir.z;

    var basis: array<f32, 9>;
    basis[0] = SH_Y00;                        // L0
    basis[1] = SH_Y1 * y;                     // L1 m=-1
    basis[2] = SH_Y1 * z;                     // L1 m=0
    basis[3] = SH_Y1 * x;                     // L1 m=+1
    basis[4] = SH_Y2_NEG2 * x * y;            // L2 m=-2
    basis[5] = SH_Y2_NEG1 * y * z;            // L2 m=-1
    basis[6] = SH_Y2_0 * (3.0 * z * z - 1.0); // L2 m=0
    basis[7] = SH_Y2_POS1 * x * z;            // L2 m=+1
    basis[8] = SH_Y2_POS2 * (x * x - y * y);  // L2 m=+2

    return basis;
}

/// Accumulate a sample into SH coefficients.
fn sh_accumulate(
    coeffs: ptr<function, array<vec3<f32>, 9>>,
    dir: vec3<f32>,
    color: vec3<f32>,
    weight: f32
) {
    let basis = sh_basis_l2(dir);
    let weighted_color = color * weight;

    for (var i: u32 = 0u; i < 9u; i = i + 1u) {
        (*coeffs)[i] = (*coeffs)[i] + weighted_color * basis[i];
    }
}

/// Apply irradiance convolution to SH coefficients.
fn sh_convolve_irradiance(coeffs: array<vec3<f32>, 9>) -> array<vec3<f32>, 9> {
    var result: array<vec3<f32>, 9>;

    // L0
    result[0] = coeffs[0] * SH_A0;

    // L1
    result[1] = coeffs[1] * SH_A1;
    result[2] = coeffs[2] * SH_A1;
    result[3] = coeffs[3] * SH_A1;

    // L2
    result[4] = coeffs[4] * SH_A2;
    result[5] = coeffs[5] * SH_A2;
    result[6] = coeffs[6] * SH_A2;
    result[7] = coeffs[7] * SH_A2;
    result[8] = coeffs[8] * SH_A2;

    return result;
}

/// Linear interpolation between coefficient sets.
fn sh_lerp(a: array<vec3<f32>, 9>, b: array<vec3<f32>, 9>, t: f32) -> array<vec3<f32>, 9> {
    var result: array<vec3<f32>, 9>;
    for (var i: u32 = 0u; i < 9u; i = i + 1u) {
        result[i] = mix(a[i], b[i], t);
    }
    return result;
}

// ============================================================================
// Main Compute Kernel
// ============================================================================

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let local_probe_idx = global_id.x;

    // Bounds check
    if local_probe_idx >= params.probe_count {
        return;
    }

    let global_probe_idx = params.batch_start + local_probe_idx;

    // Initialize accumulators
    var sh_coeffs: array<vec3<f32>, 9>;
    for (var i: u32 = 0u; i < 9u; i = i + 1u) {
        sh_coeffs[i] = vec3<f32>(0.0);
    }
    var total_weight: f32 = 0.0;

    // Sample all 6 faces
    for (var face: u32 = 0u; face < NUM_FACES; face = face + 1u) {
        // Sample at multiple points within each face
        let res = params.face_resolution;
        let step = max(res / 8u, 1u); // Adaptive sampling based on resolution

        for (var py: u32 = 0u; py < res; py = py + step) {
            for (var px: u32 = 0u; px < res; px = px + step) {
                // Get UV for this pixel
                let atlas_uv = get_face_pixel_uv(local_probe_idx, face, px, py);

                // Sample atlas texture
                let radiance = textureSampleLevel(atlas_texture, atlas_sampler, atlas_uv, 0.0);

                // Compute face-local UV
                let face_u = (f32(px) + 0.5) / f32(res);
                let face_v = (f32(py) + 0.5) / f32(res);

                // Get world direction
                let dir = face_uv_to_direction(face, face_u, face_v);

                // Compute solid angle weight
                let solid_angle = texel_solid_angle(face_u, face_v);

                // Accumulate into SH
                sh_accumulate(&sh_coeffs, dir, radiance.rgb, solid_angle);
                total_weight = total_weight + solid_angle;
            }
        }
    }

    // Normalize by total solid angle (should be ~4*PI)
    if total_weight > 0.0 {
        let norm = (4.0 * PI) / total_weight;
        for (var i: u32 = 0u; i < 9u; i = i + 1u) {
            sh_coeffs[i] = sh_coeffs[i] * norm;
        }
    }

    // Apply irradiance convolution
    let irradiance_coeffs = sh_convolve_irradiance(sh_coeffs);

    // Apply hysteresis blending with previous values
    var final_coeffs: array<vec3<f32>, 9>;
    if params.hysteresis > 0.0 {
        // Read previous coefficients
        var prev_coeffs: array<vec3<f32>, 9>;
        for (var i: u32 = 0u; i < 9u; i = i + 1u) {
            prev_coeffs[i] = probe_sh[global_probe_idx].irradiance[i].rgb;
        }
        final_coeffs = sh_lerp(irradiance_coeffs, prev_coeffs, params.hysteresis);
    } else {
        final_coeffs = irradiance_coeffs;
    }

    // Write output
    for (var i: u32 = 0u; i < 9u; i = i + 1u) {
        probe_sh[global_probe_idx].irradiance[i] = vec4<f32>(final_coeffs[i], 0.0);
    }

    // Compute visibility coefficients from depth atlas (optional)
    // For now, set default full visibility
    for (var i: u32 = 0u; i < 9u; i = i + 1u) {
        probe_sh[global_probe_idx].visibility[i] = select(0.0, 1.0, i == 0u);
    }
}

// ============================================================================
// Alternative entry point for high-quality projection
// ============================================================================

// High-quality version that samples every texel
@compute @workgroup_size(8, 8, 1)
fn main_hq(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // This entry point processes one face at a time
    // global_id.x, global_id.y = texel within face
    // Requires dispatch per face, managed by CPU

    // Implementation would sample every texel for maximum quality
    // Left as placeholder for future enhancement
}

// ============================================================================
// Debug visualization entry point
// ============================================================================

struct DebugParams {
    probe_idx: u32,
    face_idx: u32,
    mode: u32,  // 0=radiance, 1=direction, 2=solid_angle
    _pad: u32,
}

@group(1) @binding(0) var<uniform> debug_params: DebugParams;
@group(1) @binding(1) var output_texture: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8, 1)
fn debug_visualize(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let px = global_id.x;
    let py = global_id.y;

    if px >= params.face_resolution || py >= params.face_resolution {
        return;
    }

    let atlas_uv = get_face_pixel_uv(debug_params.probe_idx, debug_params.face_idx, px, py);

    var color: vec4<f32>;

    switch debug_params.mode {
        case 0u: {
            // Radiance visualization
            color = textureSampleLevel(atlas_texture, atlas_sampler, atlas_uv, 0.0);
        }
        case 1u: {
            // Direction visualization
            let face_u = (f32(px) + 0.5) / f32(params.face_resolution);
            let face_v = (f32(py) + 0.5) / f32(params.face_resolution);
            let dir = face_uv_to_direction(debug_params.face_idx, face_u, face_v);
            color = vec4<f32>(dir * 0.5 + 0.5, 1.0);
        }
        case 2u: {
            // Solid angle visualization
            let face_u = (f32(px) + 0.5) / f32(params.face_resolution);
            let face_v = (f32(py) + 0.5) / f32(params.face_resolution);
            let sa = texel_solid_angle(face_u, face_v);
            // Normalize for visualization (center texels have ~0.01 solid angle)
            let normalized = clamp(sa * 100.0, 0.0, 1.0);
            color = vec4<f32>(normalized, normalized, normalized, 1.0);
        }
        default: {
            color = vec4<f32>(1.0, 0.0, 1.0, 1.0); // Magenta = error
        }
    }

    textureStore(output_texture, vec2<i32>(i32(px), i32(py)), color);
}
