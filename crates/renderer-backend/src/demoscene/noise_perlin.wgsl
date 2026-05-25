// SPDX-License-Identifier: MIT
//
// noise_perlin.wgsl -- Perlin noise 3D.
//
// Perlin noise generates a pseudo-random gradient vector at each integer grid
// point via the hash functions in noise_hash.wgsl, then takes the dot product
// of that gradient with the offset from the grid point. The dot products at
// the eight corners of a cell are smoothly interpolated (trilinearly with the
// 6t^5 - 15t^4 + 10t^3 fade curve) to produce the final value.
//
// Unlike value noise (which interpolates scalar hash values), Perlin noise
// interpolates gradient dot products. This produces a fundamentally smoother
// result with fewer grid artifacts and a zero-mean output distribution, since
// gradient vectors are symmetric around origin.
//
// Properties:
//   - Deterministic: same input always produces same output
//   - Continuous: smooth interpolation via fade curve, C1 at boundaries
//   - Zero mean: gradient dot-product approach yields mean ~0
//   - Range: output is in approximately [-sqrt(3), sqrt(3)] (~[-1.73, 1.73])
//     but typical amplitude is in [-1, 1]; gradient vectors are unit-length
//
// Gradient selection uses 12 edge-centered gradient vectors mapped from a
// hash-derived index. This follows the classic Perlin noise design.
//
// Dependencies:
//   - noise_hash.wgsl (hash31 from T-DEMO-1.28)
//
// Reference: Ken Perlin -- Improving Noise
// https://mrl.cs.nyu.edu/~perlin/paper445.pdf

// =============================================================================
// T-DEMO-1.30: Perlin Noise 3D
// =============================================================================

/// Table of 12 gradient vectors for 3D Perlin noise.
///
/// These are the edge-centered unit vectors of a cube, selected so that no
/// gradient aligns with a coordinate axis. Each vector has exactly two
/// non-zero components of equal magnitude (1/sqrt(2)), ensuring that the
/// dot product with a corner offset is equally sensitive to displacement
/// along each of the two relevant axes.
fn perlin_gradient(hash_value: f32, offset: vec3<f32>) -> f32 {
    // Map hash [0, 1) to integer index in 0..12
    let h = i32(floor(hash_value * 12.0));

    // 12 edge-centered gradient vectors:
    //   (+-1, +-1,  0) -- 4 combinations
    //   (+-1,  0, +-1) -- 4 combinations
    //   ( 0, +-1, +-1) -- 4 combinations
    var g: vec3<f32>;
    switch h {
        case 0 { g = vec3<f32>( 1.0,  1.0,  0.0); }
        case 1 { g = vec3<f32>(-1.0,  1.0,  0.0); }
        case 2 { g = vec3<f32>( 1.0, -1.0,  0.0); }
        case 3 { g = vec3<f32>(-1.0, -1.0,  0.0); }
        case 4 { g = vec3<f32>( 1.0,  0.0,  1.0); }
        case 5 { g = vec3<f32>(-1.0,  0.0,  1.0); }
        case 6 { g = vec3<f32>( 1.0,  0.0, -1.0); }
        case 7 { g = vec3<f32>(-1.0,  0.0, -1.0); }
        case 8 { g = vec3<f32>( 0.0,  1.0,  1.0); }
        case 9 { g = vec3<f32>( 0.0, -1.0,  1.0); }
        case 10 { g = vec3<f32>( 0.0,  1.0, -1.0); }
        case 11 { g = vec3<f32>( 0.0, -1.0, -1.0); }
        default { g = vec3<f32>( 1.0,  1.0,  0.0); }
    }

    // Normalize to unit length: each edge vector has magnitude sqrt(2)
    // so we divide by sqrt(2) = 0.7071067811865475
    g = g * 0.7071067811865475;

    return dot(g, offset);
}

/// 3D Perlin noise: maps a 3D coordinate to a smooth gradient-based noise
/// value with approximately zero mean.
///
///   p        -- 3D input coordinate
///   returns  -- noise value, approximately in [-1, 1], zero mean
fn perlin_noise_3d(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = p - i;

    // Smoothstep fade curve: 6t^5 - 15t^4 + 10t^3
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    // Eight corner offsets from the input point
    let o000 = f;
    let o100 = f - vec3<f32>(1.0, 0.0, 0.0);
    let o010 = f - vec3<f32>(0.0, 1.0, 0.0);
    let o110 = f - vec3<f32>(1.0, 1.0, 0.0);
    let o001 = f - vec3<f32>(0.0, 0.0, 1.0);
    let o101 = f - vec3<f32>(1.0, 0.0, 1.0);
    let o011 = f - vec3<f32>(0.0, 1.0, 1.0);
    let o111 = f - vec3<f32>(1.0, 1.0, 1.0);

    // Hash values at 8 corners -- used to select gradient vectors
    let h000 = hash31(i + vec3<f32>(0.0, 0.0, 0.0));
    let h100 = hash31(i + vec3<f32>(1.0, 0.0, 0.0));
    let h010 = hash31(i + vec3<f32>(0.0, 1.0, 0.0));
    let h110 = hash31(i + vec3<f32>(1.0, 1.0, 0.0));
    let h001 = hash31(i + vec3<f32>(0.0, 0.0, 1.0));
    let h101 = hash31(i + vec3<f32>(1.0, 0.0, 1.0));
    let h011 = hash31(i + vec3<f32>(0.0, 1.0, 1.0));
    let h111 = hash31(i + vec3<f32>(1.0, 1.0, 1.0));

    // Gradient dot products at each corner
    let g000 = perlin_gradient(h000, o000);
    let g100 = perlin_gradient(h100, o100);
    let g010 = perlin_gradient(h010, o010);
    let g110 = perlin_gradient(h110, o110);
    let g001 = perlin_gradient(h001, o001);
    let g101 = perlin_gradient(h101, o101);
    let g011 = perlin_gradient(h011, o011);
    let g111 = perlin_gradient(h111, o111);

    // Trilinear interpolation of gradient dot products
    let vx00 = g000 + u.x * (g100 - g000);
    let vx10 = g010 + u.x * (g110 - g010);
    let vx01 = g001 + u.x * (g101 - g001);
    let vx11 = g011 + u.x * (g111 - g011);

    let vy0 = vx00 + u.y * (vx10 - vx00);
    let vy1 = vx01 + u.y * (vx11 - vx01);

    return vy0 + u.z * (vy1 - vy0);
}
