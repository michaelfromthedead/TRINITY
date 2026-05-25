// SPDX-License-Identifier: MIT
//
// noise_hash.wgsl -- Hash functions for pseudo-random number generation.
//
// These functions take integer or float coordinates and return deterministic
// pseudo-random values in [0, 1). They are designed for noise generation,
// procedural variation, and seeding downstream PRNG sequences.
//
// Properties:
//   - Deterministic: same input always produces same output
//   - Uniform distribution: output values are uniformly distributed in [0, 1)
//   - No visible spatial patterns: adjacent inputs produce uncorrelated outputs
//   - Float-domain: works with f32 coordinates, no integer bit ops required
//
// All functions use the fract(sin(dot(...))) pattern with large irrational
// constants. The output range is [0, 1) for all return types.
//
// Reference: Inigo Quilez -- Hash functions
// https://iquilezles.org/articles/hash/

// =============================================================================
// T-DEMO-1.28: Scalar Hash Functions (1D-4D -> f32)
// =============================================================================

/// 1D hash: maps a scalar coordinate to a pseudo-random f32 in [0, 1).
///   p        -- scalar coordinate (typically an integer or integer + offset)
///   returns  -- pseudo-random value in [0, 1)
fn hash11(p: f32) -> f32 {
    var q = p;
    q = fract(q * 0.1031);
    q = q * (q + 33.33);
    q = q * (q + q);
    return fract(q);
}

/// 2D hash: maps a 2D coordinate to a pseudo-random f32 in [0, 1).
///   p        -- 2D coordinate (e.g. cell index, pixel position)
///   returns  -- pseudo-random value in [0, 1)
fn hash21(p: vec2<f32>) -> f32 {
    var q = p;
    q = fract(q * vec2<f32>(0.1031, 0.1030));
    q = q + dot(q, q + 33.33);
    return fract(q.x * q.y);
}

/// 3D hash: maps a 3D coordinate to a pseudo-random f32 in [0, 1).
///   p        -- 3D coordinate (e.g. grid position with time)
///   returns  -- pseudo-random value in [0, 1)
fn hash31(p: vec3<f32>) -> f32 {
    var q = p;
    q = fract(q * vec3<f32>(0.1031, 0.1030, 0.0973));
    q = q + dot(q, q + 33.33);
    return fract(q.x * q.y * q.z);
}

/// 4D hash: maps a 4D coordinate to a pseudo-random f32 in [0, 1).
///   p        -- 4D coordinate (e.g. 3D + time)
///   returns  -- pseudo-random value in [0, 1)
fn hash41(p: vec4<f32>) -> f32 {
    var q = p;
    q = fract(q * 0.1031);
    q = q + dot(q, q + 33.33);
    return fract(q.x * q.y * q.z * q.w);
}

// =============================================================================
// T-DEMO-1.28: Vector Hash Functions (2D-3D -> vec2/vec3)
// =============================================================================

/// 2D-to-2D hash: maps a 2D coordinate to two uncorrelated f32 values in [0, 1).
/// Use this for generating pairs of random values (e.g. 2D jitter, u,v
/// texture coordinates) from a single 2D input.
///   p        -- 2D coordinate
///   returns  -- vec2<f32> with both components in [0, 1)
fn hash22(p: vec2<f32>) -> vec2<f32> {
    var q = vec3<f32>(p.x, p.y, p.x);
    q = fract(q * vec3<f32>(0.1031, 0.1030, 0.0973));
    q = q + dot(q, q.yzx + 33.33);
    return fract(vec2<f32>(q.x + q.y, q.x + q.z) * vec2<f32>(q.z, q.y));
}

/// 3D-to-2D hash: maps a 3D coordinate to two uncorrelated f32 values in [0, 1).
/// Use this for generating random pairs from a 3D input (e.g. color + material
/// index from spatial position).
///   p        -- 3D coordinate
///   returns  -- vec2<f32> with both components in [0, 1)
fn hash32(p: vec3<f32>) -> vec2<f32> {
    var q = p;
    q = fract(q * vec3<f32>(0.1031, 0.1030, 0.0973));
    q = q + dot(q, q.yxz + 33.33);
    let sum = vec2<f32>(q.x + q.y, q.x + q.z);
    let zy = vec2<f32>(q.z, q.y);
    return fract(sum * zy);
}

/// 3D-to-3D hash: maps a 3D coordinate to three uncorrelated f32 values in [0, 1).
/// Use this for generating random triples from a 3D input (e.g. RGB color
/// from spatial position).
///   p        -- 3D coordinate
///   returns  -- vec3<f32> with all components in [0, 1)
fn hash33(p: vec3<f32>) -> vec3<f32> {
    var q = p;
    q = fract(q * vec3<f32>(0.1031, 0.1030, 0.0973));
    q = q + dot(q, q.yxz + 33.33);
    // (q.xxy + q.yxx) = (x+y, 2x, x+y)
    // multiplied by q.zyx = (z, y, x)
    // result: ((x+y)*z, 2x*y, (x+y)*x)
    return fract((q.xxy + q.yxx) * q.zyx);
}
