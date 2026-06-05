// SPDX-License-Identifier: MIT
//
// spherical_harmonics.wgsl -- 3rd-order (L2) spherical harmonics library for DDGI.
//
// Provides complete SH basis functions through order 2 (9 coefficients per color channel):
//   - L0: 1 coefficient (DC term)
//   - L1: 3 coefficients (linear terms)
//   - L2: 5 coefficients (quadratic terms)
//
// Reference: "An Efficient Representation for Irradiance Environment Maps"
//            Ramamoorthi & Hanrahan, SIGGRAPH 2001
//
// Conventions:
//   - Real spherical harmonics with Condon-Shortley phase
//   - Direction vectors assumed normalized
//   - Coefficient ordering: Y_0^0, Y_1^-1, Y_1^0, Y_1^1, Y_2^-2, Y_2^-1, Y_2^0, Y_2^1, Y_2^2

// ============================================================================
// Constants
// ============================================================================

// Normalization constants for SH basis functions
// Y_l^m = K_l^m * P_l^|m|(cos(theta)) * {cos(m*phi) for m>=0, sin(|m|*phi) for m<0}

// L0 basis constant: sqrt(1/(4*PI))
const SH_Y00: f32 = 0.28209479177387814;

// L1 basis constant: sqrt(3/(4*PI))
const SH_Y1: f32 = 0.4886025119029199;

// L2 basis constants
const SH_Y2_NEG2: f32 = 1.0925484305920792;   // sqrt(15/(4*PI))     for xy
const SH_Y2_NEG1: f32 = 1.0925484305920792;   // sqrt(15/(4*PI))     for yz
const SH_Y2_0: f32 = 0.31539156525252005;     // sqrt(5/(16*PI))     for 3z^2-1
const SH_Y2_POS1: f32 = 1.0925484305920792;   // sqrt(15/(4*PI))     for xz
const SH_Y2_POS2: f32 = 0.5462742152960396;   // sqrt(15/(16*PI))    for x^2-y^2

// Cosine lobe convolution coefficients (Ramamoorthi & Hanrahan 2001)
// These scale SH coefficients to compute irradiance from radiance
const SH_A0: f32 = 1.0;                       // PI * (1/PI) = 1
const SH_A1: f32 = 0.6666666666666666;        // 2*PI/3 * (1/PI) = 2/3
const SH_A2: f32 = 0.25;                      // PI/4 * (1/PI) = 1/4

// Precomputed A_l * sqrt(4*PI / (2*l+1)) for irradiance reconstruction
const SH_IRRADIANCE_L0: f32 = 3.5449077018110318;  // A0 * sqrt(4*PI)
const SH_IRRADIANCE_L1: f32 = 2.0943951023931953;  // A1 * sqrt(4*PI/3)
const SH_IRRADIANCE_L2: f32 = 0.7853981633974483;  // A2 * sqrt(4*PI/5)

const PI: f32 = 3.14159265358979323846;

// ============================================================================
// SH Coefficient Storage (9 RGB coefficients = 27 floats)
// ============================================================================

// Note: WGSL doesn't allow array<vec3<f32>, 9> as a struct field cleanly,
// so we pack coefficients into 3 arrays per RGB channel, or use explicit fields.

struct SHCoeffsL2 {
    // Band 0 (L=0): 1 coefficient
    c0: vec3<f32>,
    // Band 1 (L=1): 3 coefficients
    c1: vec3<f32>,
    c2: vec3<f32>,
    c3: vec3<f32>,
    // Band 2 (L=2): 5 coefficients
    c4: vec3<f32>,
    c5: vec3<f32>,
    c6: vec3<f32>,
    c7: vec3<f32>,
    c8: vec3<f32>,
}

// ============================================================================
// Basis Function Evaluation
// ============================================================================

/// Evaluate all 9 SH basis functions at a direction.
/// Returns basis values: [Y_0^0, Y_1^-1, Y_1^0, Y_1^1, Y_2^-2, Y_2^-1, Y_2^0, Y_2^1, Y_2^2]
fn sh_basis_l2(dir: vec3<f32>) -> array<f32, 9> {
    let x = dir.x;
    let y = dir.y;
    let z = dir.z;

    var basis: array<f32, 9>;

    // L=0, m=0: Y_0^0 = 0.5 * sqrt(1/PI)
    basis[0] = SH_Y00;

    // L=1, m=-1: Y_1^-1 = sqrt(3/(4*PI)) * y
    basis[1] = SH_Y1 * y;

    // L=1, m=0: Y_1^0 = sqrt(3/(4*PI)) * z
    basis[2] = SH_Y1 * z;

    // L=1, m=1: Y_1^1 = sqrt(3/(4*PI)) * x
    basis[3] = SH_Y1 * x;

    // L=2, m=-2: Y_2^-2 = sqrt(15/(4*PI)) * x * y
    basis[4] = SH_Y2_NEG2 * x * y;

    // L=2, m=-1: Y_2^-1 = sqrt(15/(4*PI)) * y * z
    basis[5] = SH_Y2_NEG1 * y * z;

    // L=2, m=0: Y_2^0 = sqrt(5/(16*PI)) * (3*z^2 - 1)
    basis[6] = SH_Y2_0 * (3.0 * z * z - 1.0);

    // L=2, m=1: Y_2^1 = sqrt(15/(4*PI)) * x * z
    basis[7] = SH_Y2_POS1 * x * z;

    // L=2, m=2: Y_2^2 = sqrt(15/(16*PI)) * (x^2 - y^2)
    basis[8] = SH_Y2_POS2 * (x * x - y * y);

    return basis;
}

/// Evaluate individual L0 basis function.
fn sh_basis_l0(dir: vec3<f32>) -> f32 {
    return SH_Y00;
}

/// Evaluate L1 basis functions, returning [Y_1^-1, Y_1^0, Y_1^1].
fn sh_basis_l1(dir: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(SH_Y1 * dir.y, SH_Y1 * dir.z, SH_Y1 * dir.x);
}

/// Evaluate L2 basis functions, returning 5 values.
fn sh_basis_l2_band(dir: vec3<f32>) -> array<f32, 5> {
    let x = dir.x;
    let y = dir.y;
    let z = dir.z;

    var basis: array<f32, 5>;
    basis[0] = SH_Y2_NEG2 * x * y;
    basis[1] = SH_Y2_NEG1 * y * z;
    basis[2] = SH_Y2_0 * (3.0 * z * z - 1.0);
    basis[3] = SH_Y2_POS1 * x * z;
    basis[4] = SH_Y2_POS2 * (x * x - y * y);
    return basis;
}

// ============================================================================
// SH Evaluation (Coefficients -> Color at Direction)
// ============================================================================

/// Evaluate SH at a direction using L0+L1+L2 coefficients (9 vec3 coefficients).
/// This reconstructs the color stored in the SH representation at the given direction.
fn sh_evaluate_l2(coeffs: SHCoeffsL2, dir: vec3<f32>) -> vec3<f32> {
    let basis = sh_basis_l2(dir);

    return coeffs.c0 * basis[0]
         + coeffs.c1 * basis[1]
         + coeffs.c2 * basis[2]
         + coeffs.c3 * basis[3]
         + coeffs.c4 * basis[4]
         + coeffs.c5 * basis[5]
         + coeffs.c6 * basis[6]
         + coeffs.c7 * basis[7]
         + coeffs.c8 * basis[8];
}

/// Evaluate SH at a direction using array form (for compatibility).
fn sh_evaluate_l2_array(coeffs: array<vec3<f32>, 9>, dir: vec3<f32>) -> vec3<f32> {
    let basis = sh_basis_l2(dir);

    var result = vec3<f32>(0.0);
    for (var i: u32 = 0u; i < 9u; i = i + 1u) {
        result = result + coeffs[i] * basis[i];
    }
    return result;
}

/// Evaluate only L0+L1 terms (4 coefficients) for faster approximation.
fn sh_evaluate_l1(c0: vec3<f32>, c1: vec3<f32>, c2: vec3<f32>, c3: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    return c0 * SH_Y00
         + c1 * (SH_Y1 * dir.y)
         + c2 * (SH_Y1 * dir.z)
         + c3 * (SH_Y1 * dir.x);
}

// ============================================================================
// SH Projection (Color at Direction -> Coefficients)
// ============================================================================

/// Project a color sample at a direction into L2 SH coefficients.
/// This is the inverse of evaluation: integrate color*basis over the sphere.
/// For Monte Carlo integration, multiply result by 4*PI/num_samples.
fn sh_project_l2(dir: vec3<f32>, color: vec3<f32>) -> SHCoeffsL2 {
    let basis = sh_basis_l2(dir);

    var coeffs: SHCoeffsL2;
    coeffs.c0 = color * basis[0];
    coeffs.c1 = color * basis[1];
    coeffs.c2 = color * basis[2];
    coeffs.c3 = color * basis[3];
    coeffs.c4 = color * basis[4];
    coeffs.c5 = color * basis[5];
    coeffs.c6 = color * basis[6];
    coeffs.c7 = color * basis[7];
    coeffs.c8 = color * basis[8];
    return coeffs;
}

/// Project into array form.
fn sh_project_l2_array(dir: vec3<f32>, color: vec3<f32>) -> array<vec3<f32>, 9> {
    let basis = sh_basis_l2(dir);

    var coeffs: array<vec3<f32>, 9>;
    for (var i: u32 = 0u; i < 9u; i = i + 1u) {
        coeffs[i] = color * basis[i];
    }
    return coeffs;
}

/// Accumulate a projected sample into existing coefficients.
fn sh_accumulate_l2(coeffs: ptr<function, SHCoeffsL2>, dir: vec3<f32>, color: vec3<f32>) {
    let projected = sh_project_l2(dir, color);
    (*coeffs).c0 = (*coeffs).c0 + projected.c0;
    (*coeffs).c1 = (*coeffs).c1 + projected.c1;
    (*coeffs).c2 = (*coeffs).c2 + projected.c2;
    (*coeffs).c3 = (*coeffs).c3 + projected.c3;
    (*coeffs).c4 = (*coeffs).c4 + projected.c4;
    (*coeffs).c5 = (*coeffs).c5 + projected.c5;
    (*coeffs).c6 = (*coeffs).c6 + projected.c6;
    (*coeffs).c7 = (*coeffs).c7 + projected.c7;
    (*coeffs).c8 = (*coeffs).c8 + projected.c8;
}

// ============================================================================
// Irradiance Convolution (Cosine Lobe Kernel)
// ============================================================================

/// Convolve SH coefficients with cosine lobe to convert radiance to irradiance.
/// Applies the A_l factors from Ramamoorthi & Hanrahan 2001:
///   A_0 = 1.0 (PI/PI)
///   A_1 = 2/3 (2*PI/3 / PI)
///   A_2 = 1/4 (PI/4 / PI)
fn sh_convolve_irradiance(coeffs: SHCoeffsL2) -> SHCoeffsL2 {
    var result: SHCoeffsL2;

    // L=0 band: multiply by A_0
    result.c0 = coeffs.c0 * SH_A0;

    // L=1 band: multiply by A_1
    result.c1 = coeffs.c1 * SH_A1;
    result.c2 = coeffs.c2 * SH_A1;
    result.c3 = coeffs.c3 * SH_A1;

    // L=2 band: multiply by A_2
    result.c4 = coeffs.c4 * SH_A2;
    result.c5 = coeffs.c5 * SH_A2;
    result.c6 = coeffs.c6 * SH_A2;
    result.c7 = coeffs.c7 * SH_A2;
    result.c8 = coeffs.c8 * SH_A2;

    return result;
}

/// Convolve array form.
fn sh_convolve_irradiance_array(coeffs: array<vec3<f32>, 9>) -> array<vec3<f32>, 9> {
    var result: array<vec3<f32>, 9>;

    // L=0
    result[0] = coeffs[0] * SH_A0;

    // L=1
    result[1] = coeffs[1] * SH_A1;
    result[2] = coeffs[2] * SH_A1;
    result[3] = coeffs[3] * SH_A1;

    // L=2
    result[4] = coeffs[4] * SH_A2;
    result[5] = coeffs[5] * SH_A2;
    result[6] = coeffs[6] * SH_A2;
    result[7] = coeffs[7] * SH_A2;
    result[8] = coeffs[8] * SH_A2;

    return result;
}

// ============================================================================
// SH Rotation
// ============================================================================

/// Rotate L2 SH coefficients by a 3x3 rotation matrix.
///
/// L0 is rotationally invariant.
/// L1 transforms like a vector (apply R directly).
/// L2 transforms via a 5x5 matrix derived from R.
///
/// Reference: "Rotation Invariant Spherical Harmonic Representation of 3D Shape Descriptors"
fn sh_rotate_l2(coeffs: SHCoeffsL2, r: mat3x3<f32>) -> SHCoeffsL2 {
    var result: SHCoeffsL2;

    // L0: rotationally invariant
    result.c0 = coeffs.c0;

    // L1: transforms as a vector
    // [Y_1^-1, Y_1^0, Y_1^1] maps to [y, z, x] directions
    // So we rotate the "direction" formed by L1 coefficients
    let l1_vec_r = vec3<f32>(coeffs.c3.r, coeffs.c1.r, coeffs.c2.r);
    let l1_vec_g = vec3<f32>(coeffs.c3.g, coeffs.c1.g, coeffs.c2.g);
    let l1_vec_b = vec3<f32>(coeffs.c3.b, coeffs.c1.b, coeffs.c2.b);

    let rotated_r = r * l1_vec_r;
    let rotated_g = r * l1_vec_g;
    let rotated_b = r * l1_vec_b;

    // Unpack back: c1=Y_1^-1=y, c2=Y_1^0=z, c3=Y_1^1=x
    result.c1 = vec3<f32>(rotated_r.y, rotated_g.y, rotated_b.y);
    result.c2 = vec3<f32>(rotated_r.z, rotated_g.z, rotated_b.z);
    result.c3 = vec3<f32>(rotated_r.x, rotated_g.x, rotated_b.x);

    // L2: requires a 5x5 rotation matrix
    // Compute the L2 rotation matrix from the 3x3 rotation
    let m = sh_compute_l2_rotation_matrix(r);

    // Apply to each color channel
    for (var c: u32 = 0u; c < 3u; c = c + 1u) {
        var l2_in: array<f32, 5>;
        l2_in[0] = select(select(coeffs.c4.b, coeffs.c4.g, c == 1u), coeffs.c4.r, c == 0u);
        l2_in[1] = select(select(coeffs.c5.b, coeffs.c5.g, c == 1u), coeffs.c5.r, c == 0u);
        l2_in[2] = select(select(coeffs.c6.b, coeffs.c6.g, c == 1u), coeffs.c6.r, c == 0u);
        l2_in[3] = select(select(coeffs.c7.b, coeffs.c7.g, c == 1u), coeffs.c7.r, c == 0u);
        l2_in[4] = select(select(coeffs.c8.b, coeffs.c8.g, c == 1u), coeffs.c8.r, c == 0u);

        var l2_out: array<f32, 5>;
        for (var i: u32 = 0u; i < 5u; i = i + 1u) {
            var sum = 0.0;
            for (var j: u32 = 0u; j < 5u; j = j + 1u) {
                sum = sum + m[i * 5u + j] * l2_in[j];
            }
            l2_out[i] = sum;
        }

        if c == 0u {
            result.c4.r = l2_out[0];
            result.c5.r = l2_out[1];
            result.c6.r = l2_out[2];
            result.c7.r = l2_out[3];
            result.c8.r = l2_out[4];
        } else if c == 1u {
            result.c4.g = l2_out[0];
            result.c5.g = l2_out[1];
            result.c6.g = l2_out[2];
            result.c7.g = l2_out[3];
            result.c8.g = l2_out[4];
        } else {
            result.c4.b = l2_out[0];
            result.c5.b = l2_out[1];
            result.c6.b = l2_out[2];
            result.c7.b = l2_out[3];
            result.c8.b = l2_out[4];
        }
    }

    return result;
}

/// Compute the 5x5 rotation matrix for L=2 band from a 3x3 rotation matrix.
/// Returns a flat array of 25 elements (row-major).
fn sh_compute_l2_rotation_matrix(r: mat3x3<f32>) -> array<f32, 25> {
    // Elements of the rotation matrix
    let r00 = r[0][0]; let r01 = r[0][1]; let r02 = r[0][2];
    let r10 = r[1][0]; let r11 = r[1][1]; let r12 = r[1][2];
    let r20 = r[2][0]; let r21 = r[2][1]; let r22 = r[2][2];

    var m: array<f32, 25>;

    // The L2 rotation matrix is derived from products of rotation matrix elements.
    // Order: [Y_2^-2, Y_2^-1, Y_2^0, Y_2^1, Y_2^2]
    // Basis functions map to: [xy, yz, 3z^2-1, xz, x^2-y^2]

    // Row 0: Y_2^-2 (xy term)
    m[0] = r00 * r11 + r01 * r10;                                    // xy
    m[1] = r01 * r12 + r02 * r11;                                    // yz
    m[2] = r02 * r12 * 2.0;                                          // z^2 contribution
    m[3] = r00 * r12 + r02 * r10;                                    // xz
    m[4] = (r00 * r10 - r01 * r11);                                  // x^2-y^2

    // Row 1: Y_2^-1 (yz term)
    m[5] = r10 * r21 + r11 * r20;
    m[6] = r11 * r22 + r12 * r21;
    m[7] = r12 * r22 * 2.0;
    m[8] = r10 * r22 + r12 * r20;
    m[9] = (r10 * r20 - r11 * r21);

    // Row 2: Y_2^0 (3z^2-1 term)
    m[10] = r20 * r21 * 2.0;
    m[11] = r21 * r22 * 2.0;
    m[12] = 3.0 * r22 * r22 - 1.0;
    m[13] = r20 * r22 * 2.0;
    m[14] = (r20 * r20 - r21 * r21);

    // Row 3: Y_2^1 (xz term)
    m[15] = r00 * r21 + r01 * r20;
    m[16] = r01 * r22 + r02 * r21;
    m[17] = r02 * r22 * 2.0;
    m[18] = r00 * r22 + r02 * r20;
    m[19] = (r00 * r20 - r01 * r21);

    // Row 4: Y_2^2 (x^2-y^2 term)
    m[20] = (r00 * r01 - r10 * r11) * 2.0;
    m[21] = (r01 * r02 - r11 * r12) * 2.0;
    m[22] = (r02 * r02 - r12 * r12);
    m[23] = (r00 * r02 - r10 * r12) * 2.0;
    m[24] = (r00 * r00 - r01 * r01 - r10 * r10 + r11 * r11) * 0.5;

    return m;
}

/// Rotate array form.
fn sh_rotate_l2_array(coeffs: array<vec3<f32>, 9>, r: mat3x3<f32>) -> array<vec3<f32>, 9> {
    var sh: SHCoeffsL2;
    sh.c0 = coeffs[0];
    sh.c1 = coeffs[1];
    sh.c2 = coeffs[2];
    sh.c3 = coeffs[3];
    sh.c4 = coeffs[4];
    sh.c5 = coeffs[5];
    sh.c6 = coeffs[6];
    sh.c7 = coeffs[7];
    sh.c8 = coeffs[8];

    let rotated = sh_rotate_l2(sh, r);

    var result: array<vec3<f32>, 9>;
    result[0] = rotated.c0;
    result[1] = rotated.c1;
    result[2] = rotated.c2;
    result[3] = rotated.c3;
    result[4] = rotated.c4;
    result[5] = rotated.c5;
    result[6] = rotated.c6;
    result[7] = rotated.c7;
    result[8] = rotated.c8;
    return result;
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Initialize zero coefficients.
fn sh_zero_l2() -> SHCoeffsL2 {
    var coeffs: SHCoeffsL2;
    coeffs.c0 = vec3<f32>(0.0);
    coeffs.c1 = vec3<f32>(0.0);
    coeffs.c2 = vec3<f32>(0.0);
    coeffs.c3 = vec3<f32>(0.0);
    coeffs.c4 = vec3<f32>(0.0);
    coeffs.c5 = vec3<f32>(0.0);
    coeffs.c6 = vec3<f32>(0.0);
    coeffs.c7 = vec3<f32>(0.0);
    coeffs.c8 = vec3<f32>(0.0);
    return coeffs;
}

/// Scale coefficients.
fn sh_scale_l2(coeffs: SHCoeffsL2, scale: f32) -> SHCoeffsL2 {
    var result: SHCoeffsL2;
    result.c0 = coeffs.c0 * scale;
    result.c1 = coeffs.c1 * scale;
    result.c2 = coeffs.c2 * scale;
    result.c3 = coeffs.c3 * scale;
    result.c4 = coeffs.c4 * scale;
    result.c5 = coeffs.c5 * scale;
    result.c6 = coeffs.c6 * scale;
    result.c7 = coeffs.c7 * scale;
    result.c8 = coeffs.c8 * scale;
    return result;
}

/// Add two SH coefficient sets.
fn sh_add_l2(a: SHCoeffsL2, b: SHCoeffsL2) -> SHCoeffsL2 {
    var result: SHCoeffsL2;
    result.c0 = a.c0 + b.c0;
    result.c1 = a.c1 + b.c1;
    result.c2 = a.c2 + b.c2;
    result.c3 = a.c3 + b.c3;
    result.c4 = a.c4 + b.c4;
    result.c5 = a.c5 + b.c5;
    result.c6 = a.c6 + b.c6;
    result.c7 = a.c7 + b.c7;
    result.c8 = a.c8 + b.c8;
    return result;
}

/// Linear interpolation between two SH coefficient sets.
fn sh_lerp_l2(a: SHCoeffsL2, b: SHCoeffsL2, t: f32) -> SHCoeffsL2 {
    var result: SHCoeffsL2;
    result.c0 = mix(a.c0, b.c0, t);
    result.c1 = mix(a.c1, b.c1, t);
    result.c2 = mix(a.c2, b.c2, t);
    result.c3 = mix(a.c3, b.c3, t);
    result.c4 = mix(a.c4, b.c4, t);
    result.c5 = mix(a.c5, b.c5, t);
    result.c6 = mix(a.c6, b.c6, t);
    result.c7 = mix(a.c7, b.c7, t);
    result.c8 = mix(a.c8, b.c8, t);
    return result;
}

/// Compute approximate "energy" of SH coefficients (sum of squared magnitudes).
fn sh_energy_l2(coeffs: SHCoeffsL2) -> f32 {
    return dot(coeffs.c0, coeffs.c0)
         + dot(coeffs.c1, coeffs.c1)
         + dot(coeffs.c2, coeffs.c2)
         + dot(coeffs.c3, coeffs.c3)
         + dot(coeffs.c4, coeffs.c4)
         + dot(coeffs.c5, coeffs.c5)
         + dot(coeffs.c6, coeffs.c6)
         + dot(coeffs.c7, coeffs.c7)
         + dot(coeffs.c8, coeffs.c8);
}

/// Clamp negative values in coefficients (useful for lighting).
fn sh_clamp_negative_l2(coeffs: SHCoeffsL2) -> SHCoeffsL2 {
    var result: SHCoeffsL2;
    result.c0 = max(coeffs.c0, vec3<f32>(0.0));
    result.c1 = coeffs.c1;  // L1 can be negative (directional)
    result.c2 = coeffs.c2;
    result.c3 = coeffs.c3;
    result.c4 = coeffs.c4;
    result.c5 = coeffs.c5;
    result.c6 = coeffs.c6;
    result.c7 = coeffs.c7;
    result.c8 = coeffs.c8;
    return result;
}

// ============================================================================
// DDGI Integration Helpers
// ============================================================================

/// Convert L1 (4 coefficients per channel) to L2 format (padding with zeros).
fn sh_l1_to_l2(sh_r: vec4<f32>, sh_g: vec4<f32>, sh_b: vec4<f32>) -> SHCoeffsL2 {
    var coeffs: SHCoeffsL2;

    // L0: first element of each vec4
    coeffs.c0 = vec3<f32>(sh_r.x, sh_g.x, sh_b.x);

    // L1: remaining elements
    // Note: ddgi.wgsl uses [L0, L1.x, L1.y, L1.z] = [L0, Y_1^1, Y_1^-1, Y_1^0]
    // We need to map: Y_1^-1=y, Y_1^0=z, Y_1^1=x
    coeffs.c1 = vec3<f32>(sh_r.z, sh_g.z, sh_b.z);  // Y_1^-1 (y)
    coeffs.c2 = vec3<f32>(sh_r.w, sh_g.w, sh_b.w);  // Y_1^0 (z)
    coeffs.c3 = vec3<f32>(sh_r.y, sh_g.y, sh_b.y);  // Y_1^1 (x)

    // L2: zero (not present in L1)
    coeffs.c4 = vec3<f32>(0.0);
    coeffs.c5 = vec3<f32>(0.0);
    coeffs.c6 = vec3<f32>(0.0);
    coeffs.c7 = vec3<f32>(0.0);
    coeffs.c8 = vec3<f32>(0.0);

    return coeffs;
}

/// Extract L1 representation from L2 coefficients (truncation).
fn sh_l2_to_l1(coeffs: SHCoeffsL2) -> array<vec4<f32>, 3> {
    var result: array<vec4<f32>, 3>;

    // Red channel: [L0, Y_1^1(x), Y_1^-1(y), Y_1^0(z)]
    result[0] = vec4<f32>(coeffs.c0.r, coeffs.c3.r, coeffs.c1.r, coeffs.c2.r);

    // Green channel
    result[1] = vec4<f32>(coeffs.c0.g, coeffs.c3.g, coeffs.c1.g, coeffs.c2.g);

    // Blue channel
    result[2] = vec4<f32>(coeffs.c0.b, coeffs.c3.b, coeffs.c1.b, coeffs.c2.b);

    return result;
}
