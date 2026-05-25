// SPDX-License-Identifier: MIT
//
// noise_ridged.wgsl -- Ridged noise 1D/2D/3D.
//
// T-DEMO-1.32: Ridged noise for terrain generation.
// Acceptance: 1.0 - abs(FBM) for terrain, sharp valleys smooth ridges.
//
// Ridged noise transforms a standard FBM signal with the formula
//
//   ridged(p) = 1.0 - abs(fbm(p))
//
// The absolute value reflects negative signal values into the positive
// domain, creating a V-shaped cusp wherever the original FBM crosses
// zero (sharp ridges). The subsequent 1 - inversion means that former
// FBM zero-crossings become sharp peaks (ridges) and former peaks near
// +/-1 become smooth valleys.
//
// Properties:
//   - Range: output is in [0, 1] (FBM in [-1, 1], abs -> [0, 1], 1 - abs -> [0, 1])
//   - Sharp ridges: V-shaped cusps where FBM crosses zero produce
//     sharp, well-defined ridge lines
//   - Smooth valleys: where FBM approaches +/-1, the abs and subsequent
//     subtract produce smooth, rounded valleys
//   - Deterministic: same input + parameters always produces same output
//   - 1D/2D/3D: works in any dimension via FBM base
//   - Perlin variant: uses gradient-based Perlin FBM for smoother base
//
// Dependencies:
//   - noise_fbm.wgsl (fbm_1d, fbm_2d, fbm_3d, fbm_perlin_3d from T-DEMO-1.31)
//   - noise_hash.wgsl (hash11, hash21, hash31 from T-DEMO-1.28)
//   - noise_value.wgsl (value_noise_1d, value_noise_2d, value_noise_3d from T-DEMO-1.29)
//   - noise_perlin.wgsl (perlin_noise_3d from T-DEMO-1.30)
//
// Reference: Inigo Quilez -- Ridged Noise
// https://iquilezles.org/articles/fbm/

// =============================================================================
// T-DEMO-1.32: Ridged Noise 1D (Value Noise FBM Base)
// =============================================================================

/// 1D ridged noise using value-noise-based FBM as the base signal.
///
/// Transforms fbm_1d via 1.0 - abs(x), creating sharp ridges at FBM
/// zero-crossings and smooth valleys at FBM extrema.
///
///   p           -- input coordinate
///   octaves     -- number of FBM octaves
///   lacunarity  -- frequency multiplier between octaves (e.g. 2.0)
///   gain        -- amplitude multiplier between octaves (e.g. 0.5)
///   returns     -- ridged noise value in [0, 1]
fn ridged_noise_1d(p: f32, octaves: u32, lacunarity: f32, gain: f32) -> f32 {
    let fbm_val = fbm_1d(p, octaves, lacunarity, gain);
    return 1.0 - abs(fbm_val);
}

// =============================================================================
// T-DEMO-1.32: Ridged Noise 2D (Value Noise FBM Base)
// =============================================================================

/// 2D ridged noise using value-noise-based FBM as the base signal.
///
/// Transforms fbm_2d via 1.0 - abs(x), creating sharp ridges at FBM
/// zero-crossings and smooth valleys at FBM extrema.
///
///   p           -- 2D input coordinate
///   octaves     -- number of FBM octaves
///   lacunarity  -- frequency multiplier between octaves (e.g. 2.0)
///   gain        -- amplitude multiplier between octaves (e.g. 0.5)
///   returns     -- ridged noise value in [0, 1]
fn ridged_noise_2d(p: vec2<f32>, octaves: u32, lacunarity: f32, gain: f32) -> f32 {
    let fbm_val = fbm_2d(p, octaves, lacunarity, gain);
    return 1.0 - abs(fbm_val);
}

// =============================================================================
// T-DEMO-1.32: Ridged Noise 3D (Value Noise FBM Base)
// =============================================================================

/// 3D ridged noise using value-noise-based FBM as the base signal.
///
/// Transforms fbm_3d via 1.0 - abs(x), creating sharp ridges at FBM
/// zero-crossings and smooth valleys at FBM extrema.
///
///   p           -- 3D input coordinate
///   octaves     -- number of FBM octaves
///   lacunarity  -- frequency multiplier between octaves (e.g. 2.0)
///   gain        -- amplitude multiplier between octaves (e.g. 0.5)
///   returns     -- ridged noise value in [0, 1]
fn ridged_noise_3d(p: vec3<f32>, octaves: u32, lacunarity: f32, gain: f32) -> f32 {
    let fbm_val = fbm_3d(p, octaves, lacunarity, gain);
    return 1.0 - abs(fbm_val);
}

// =============================================================================
// T-DEMO-1.32: Ridged Noise 3D (Perlin Noise FBM Base)
// =============================================================================

/// 3D ridged noise using Perlin-noise-based FBM as the base signal.
///
/// Transforms fbm_perlin_3d via 1.0 - abs(x). Uses the gradient-based
/// Perlin FBM for visually smoother results with fewer grid artifacts.
/// At integer grid positions, fbm_perlin_3d = 0, so ridged_perlin_3d = 1
/// (the sharpest possible ridge).
///
///   p           -- 3D input coordinate
///   octaves     -- number of FBM octaves
///   lacunarity  -- frequency multiplier between octaves (e.g. 2.0)
///   gain        -- amplitude multiplier between octaves (e.g. 0.5)
///   returns     -- ridged noise value in [0, 1]
fn ridged_perlin_3d(p: vec3<f32>, octaves: u32, lacunarity: f32, gain: f32) -> f32 {
    let fbm_val = fbm_perlin_3d(p, octaves, lacunarity, gain);
    return 1.0 - abs(fbm_val);
}
