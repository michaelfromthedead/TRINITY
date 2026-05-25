// SPDX-License-Identifier: MIT
//
// noise_value.wgsl -- Value noise 1D/2D/3D.
//
// Value noise generates a pseudo-random value at each integer grid point via
// the hash functions in noise_hash.wgsl, then smoothly interpolates between
// adjacent grid values. The result is a smooth, continuous noise function with
// controllable frequency.
//
// Properties:
//   - Deterministic: same input always produces same output
//   - Continuous: smooth interpolation via 6t^5 - 15t^4 + 10t^3 fade curve
//   - Range: output is in [-1, 1] for scalar hash values (remapped to [-1, 1])
//   - 1D/2D/3D: works in any dimension
//
// Dependencies:
//   - noise_hash.wgsl (hash11, hash21, hash31)
//
// Reference: Inigo Quilez -- Value Noise
// https://iquilezles.org/articles/value-noise/

// =============================================================================
// T-DEMO-1.29: Value Noise 1D
// =============================================================================

/// 1D value noise: maps a scalar to a smooth pseudo-random value in [-1, 1].
///   p        -- input coordinate
///   returns  -- noise value in [-1, 1]
fn value_noise_1d(p: f32) -> f32 {
    let i = floor(p);
    let f = p - i;

    // Smoothstep fade curve: 6t^5 - 15t^4 + 10t^3
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    // Hash values at adjacent integer grid points
    let a = hash11(i);
    let b = hash11(i + 1.0);

    // Remap from [0, 1) to [-1, 1]
    let va = a * 2.0 - 1.0;
    let vb = b * 2.0 - 1.0;

    // Smooth interpolation
    return va + u * (vb - va);
}

// =============================================================================
// T-DEMO-1.29: Value Noise 2D
// =============================================================================

/// 2D value noise: maps a 2D coordinate to a smooth pseudo-random value in [-1, 1].
///   p        -- 2D input coordinate
///   returns  -- noise value in [-1, 1]
fn value_noise_2d(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = p - i;

    // Smoothstep fade curve
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    // Hash values at 4 corners of the cell
    let a = hash21(i);
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));

    // Remap from [0, 1) to [-1, 1]
    let va = a * 2.0 - 1.0;
    let vb = b * 2.0 - 1.0;
    let vc = c * 2.0 - 1.0;
    let vd = d * 2.0 - 1.0;

    // Bilinear interpolation
    let vx0 = va + u.x * (vb - va);
    let vx1 = vc + u.x * (vd - vc);
    return vx0 + u.y * (vx1 - vx0);
}

// =============================================================================
// T-DEMO-1.29: Value Noise 3D
// =============================================================================

/// 3D value noise: maps a 3D coordinate to a smooth pseudo-random value in [-1, 1].
///   p        -- 3D input coordinate
///   returns  -- noise value in [-1, 1]
fn value_noise_3d(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = p - i;

    // Smoothstep fade curve
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    // Hash values at 8 corners of the cell
    let a = hash31(i);
    let b = hash31(i + vec3<f32>(1.0, 0.0, 0.0));
    let c = hash31(i + vec3<f32>(0.0, 1.0, 0.0));
    let d = hash31(i + vec3<f32>(1.0, 1.0, 0.0));
    let e = hash31(i + vec3<f32>(0.0, 0.0, 1.0));
    let f_ = hash31(i + vec3<f32>(1.0, 0.0, 1.0));
    let g = hash31(i + vec3<f32>(0.0, 1.0, 1.0));
    let h = hash31(i + vec3<f32>(1.0, 1.0, 1.0));

    // Remap from [0, 1) to [-1, 1]
    let va = a * 2.0 - 1.0;
    let vb = b * 2.0 - 1.0;
    let vc = c * 2.0 - 1.0;
    let vd = d * 2.0 - 1.0;
    let ve = e * 2.0 - 1.0;
    let vf = f_ * 2.0 - 1.0;
    let vg = g * 2.0 - 1.0;
    let vh = h * 2.0 - 1.0;

    // Trilinear interpolation
    let vx00 = va + u.x * (vb - va);
    let vx10 = vc + u.x * (vd - vc);
    let vx01 = ve + u.x * (vf - ve);
    let vx11 = vg + u.x * (vh - vg);

    let vy0 = vx00 + u.y * (vx10 - vx00);
    let vy1 = vx01 + u.y * (vx11 - vx01);

    return vy0 + u.z * (vy1 - vy0);
}
