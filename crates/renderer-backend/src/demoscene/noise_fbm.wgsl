// SPDX-License-Identifier: MIT
//
// noise_fbm.wgsl -- Fractal Brownian Motion noise 1D/2D/3D.
//
// Fractal Brownian Motion (fBM) layers multiple octaves of a base noise
// function at successively higher frequencies and lower amplitudes. Each
// octave contributes detail at a finer scale, creating a self-similar
// fractal signal with natural-looking variation across scales.
//
// The spectral composition is controlled by three parameters:
//   - octaves:    number of noise layers summed
//   - lacunarity: frequency multiplier between successive octaves (typ. 2.0)
//   - gain:       amplitude multiplier between successive octaves (typ. 0.5)
//
// Octave n has:
//   frequency_n = frequency_0 * lacunarity^n
//   amplitude_n = amplitude_0 * gain^n
//
// The result is normalized by the sum of amplitudes so that the output
// range is approximately [-1, 1] regardless of the number of octaves.
//
// Properties:
//   - Deterministic: same input + parameters always produces same output
//   - Self-similar: detail repeats across scales (statistical self-similarity)
//   - Configurable: octaves/lacunarity/gain give explicit spectral control
//   - Range: output is approximately in [-1, 1] (normalized by amplitude sum)
//   - 1D/2D/3D: works in any dimension with value noise base
//   - 3D Perlin variant: uses gradient-based Perlin noise for smoother results
//
// Dependencies:
//   - noise_hash.wgsl (hash11, hash21, hash31 from T-DEMO-1.28)
//   - noise_value.wgsl (value_noise_1d, value_noise_2d, value_noise_3d from T-DEMO-1.29)
//   - noise_perlin.wgsl (perlin_noise_3d from T-DEMO-1.30)
//
// Reference: Inigo Quilez -- Fractional Brownian Motion
// https://iquilezles.org/articles/fbm/

// =============================================================================
// T-DEMO-1.31: FBM Noise 1D (Value Noise Base)
// =============================================================================

/// 1D fractal Brownian motion noise using value noise as the base function.
///
/// Layers `octaves` copies of 1D value noise at increasing frequencies,
/// decaying amplitudes. The result is normalized by the sum of amplitudes
/// to keep the output in a consistent range.
///
///   p           -- input coordinate
///   octaves     -- number of noise octaves to sum
///   lacunarity  -- frequency multiplier between octaves (e.g. 2.0)
///   gain        -- amplitude multiplier between octaves (e.g. 0.5)
///   returns     -- fBM noise value, approximately in [-1, 1]
fn fbm_1d(p: f32, octaves: u32, lacunarity: f32, gain: f32) -> f32 {
    var value = 0.0;
    var amplitude = 1.0;
    var frequency = 1.0;
    var max_amplitude = 0.0;

    for (var i = 0u; i < octaves; i = i + 1u) {
        value += amplitude * value_noise_1d(p * frequency);
        max_amplitude += amplitude;
        frequency *= lacunarity;
        amplitude *= gain;
    }

    // Guard against division by zero (octaves == 0 || gain chain zero)
    return select(value / max_amplitude, 0.0, max_amplitude < 1e-8);
}

// =============================================================================
// T-DEMO-1.31: FBM Noise 2D (Value Noise Base)
// =============================================================================

/// 2D fractal Brownian motion noise using value noise as the base function.
///
/// Layers `octaves` copies of 2D value noise at increasing frequencies,
/// decaying amplitudes. The result is normalized by the sum of amplitudes.
///
///   p           -- 2D input coordinate
///   octaves     -- number of noise octaves to sum
///   lacunarity  -- frequency multiplier between octaves (e.g. 2.0)
///   gain        -- amplitude multiplier between octaves (e.g. 0.5)
///   returns     -- fBM noise value, approximately in [-1, 1]
fn fbm_2d(p: vec2<f32>, octaves: u32, lacunarity: f32, gain: f32) -> f32 {
    var value = 0.0;
    var amplitude = 1.0;
    var frequency = 1.0;
    var max_amplitude = 0.0;

    for (var i = 0u; i < octaves; i = i + 1u) {
        value += amplitude * value_noise_2d(p * frequency);
        max_amplitude += amplitude;
        frequency *= lacunarity;
        amplitude *= gain;
    }

    return select(value / max_amplitude, 0.0, max_amplitude < 1e-8);
}

// =============================================================================
// T-DEMO-1.31: FBM Noise 3D (Value Noise Base)
// =============================================================================

/// 3D fractal Brownian motion noise using value noise as the base function.
///
/// Layers `octaves` copies of 3D value noise at increasing frequencies,
/// decaying amplitudes. The result is normalized by the sum of amplitudes.
///
///   p           -- 3D input coordinate
///   octaves     -- number of noise octaves to sum
///   lacunarity  -- frequency multiplier between octaves (e.g. 2.0)
///   gain        -- amplitude multiplier between octaves (e.g. 0.5)
///   returns     -- fBM noise value, approximately in [-1, 1]
fn fbm_3d(p: vec3<f32>, octaves: u32, lacunarity: f32, gain: f32) -> f32 {
    var value = 0.0;
    var amplitude = 1.0;
    var frequency = 1.0;
    var max_amplitude = 0.0;

    for (var i = 0u; i < octaves; i = i + 1u) {
        value += amplitude * value_noise_3d(p * frequency);
        max_amplitude += amplitude;
        frequency *= lacunarity;
        amplitude *= gain;
    }

    return select(value / max_amplitude, 0.0, max_amplitude < 1e-8);
}

// =============================================================================
// T-DEMO-1.31: FBM Noise 3D (Perlin Noise Base)
// =============================================================================

/// 3D fractal Brownian motion noise using Perlin noise as the base function.
///
/// Identical spectral composition to `fbm_3d`, but uses gradient-based
/// Perlin noise (zero mean) instead of value noise. This produces visually
/// smoother results with fewer grid artifacts and a more natural appearance
/// for organic terrains and volumetric effects.
///
///   p           -- 3D input coordinate
///   octaves     -- number of noise octaves to sum
///   lacunarity  -- frequency multiplier between octaves (e.g. 2.0)
///   gain        -- amplitude multiplier between octaves (e.g. 0.5)
///   returns     -- fBM noise value, approximately in [-1, 1]
fn fbm_perlin_3d(p: vec3<f32>, octaves: u32, lacunarity: f32, gain: f32) -> f32 {
    var value = 0.0;
    var amplitude = 1.0;
    var frequency = 1.0;
    var max_amplitude = 0.0;

    for (var i = 0u; i < octaves; i = i + 1u) {
        value += amplitude * perlin_noise_3d(p * frequency);
        max_amplitude += amplitude;
        frequency *= lacunarity;
        amplitude *= gain;
    }

    return select(value / max_amplitude, 0.0, max_amplitude < 1e-8);
}
