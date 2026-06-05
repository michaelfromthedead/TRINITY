//! SDF Noise Functions for Demoscene Rendering (T-DEMO-1.28 through T-DEMO-1.33)
//!
//! This module provides procedural noise functions optimized for SDF-based demoscene
//! rendering, with both CPU-side evaluation and WGSL code generation.
//!
//! # Functions
//!
//! * **Hash Functions (T-DEMO-1.28)**: Pseudo-random hash functions for 1D/2D/3D input.
//! * **Value Noise (T-DEMO-1.29)**: Smooth interpolated noise with lattice-based hashing.
//! * **Perlin Noise (T-DEMO-1.30)**: Gradient-based noise with zero mean.
//! * **FBM (T-DEMO-1.31)**: Fractal Brownian motion with configurable octaves.
//! * **Ridged Noise (T-DEMO-1.32)**: 1.0 - abs(FBM) for sharp terrain ridges.
//! * **Domain Warp (T-DEMO-1.33)**: FBM-warped coordinates for organic patterns.
//!
//! # Range Conventions
//!
//! * Hash functions: output in [0, 1)
//! * Value noise: output in [-1, 1]
//! * Perlin noise: output approximately in [-1, 1], zero mean
//! * FBM: output in [-1, 1] (normalized)
//! * Ridged noise: output in [0, 1]
//! * Domain warp: output in [-1, 1] (inherited from FBM)
//!
//! # WGSL Code Generation
//!
//! Each noise function can generate its WGSL shader code via the `*_wgsl()` functions.
//! These are designed to be included in demoscene shader pipelines.
//!
//! # References
//!
//! * Inigo Quilez -- Hash functions: https://iquilezles.org/articles/hash/
//! * Ken Perlin -- Improving Noise: https://mrl.cs.nyu.edu/~perlin/paper445.pdf
//! * Inigo Quilez -- fBM: https://iquilezles.org/articles/fbm/
//! * Inigo Quilez -- Domain Warping: https://iquilezles.org/articles/warp/


// =============================================================================
// Constants
// =============================================================================

/// Default number of FBM octaves.
pub const DEFAULT_OCTAVES: u32 = 6;

/// Default lacunarity (frequency multiplier).
pub const DEFAULT_LACUNARITY: f32 = 2.0;

/// Default gain (amplitude multiplier).
pub const DEFAULT_GAIN: f32 = 0.5;

/// Default domain warp strength.
pub const DEFAULT_WARP_STRENGTH: f32 = 0.5;

/// Epsilon for floating-point comparisons.
const EPSILON: f32 = 1e-8;

// =============================================================================
// T-DEMO-1.28: Hash Functions
// =============================================================================

/// 1D hash: maps a scalar to pseudo-random f32 in [0, 1).
///
/// Uses the fract(sin(p * K) * L) pattern for fast GPU evaluation.
///
/// # Arguments
/// * `p` - Input scalar coordinate
///
/// # Returns
/// Pseudo-random value in [0, 1)
#[inline]
pub fn hash11(p: f32) -> f32 {
    let mut q = p;
    q = fract(q * 0.1031);
    q = q * (q + 33.33);
    q = q * (q + q);
    fract(q)
}

/// 2D hash: maps a 2D coordinate to pseudo-random f32 in [0, 1).
///
/// Uses dot product with irrational constants to decorrelate input axes.
///
/// # Arguments
/// * `p` - Input 2D coordinate [x, y]
///
/// # Returns
/// Pseudo-random value in [0, 1)
#[inline]
pub fn hash21(p: [f32; 2]) -> f32 {
    let mut q = [fract(p[0] * 0.1031), fract(p[1] * 0.1030)];
    let d = q[0] * (q[1] + 33.33) + q[1] * (q[0] + 33.33);
    q[0] += d;
    q[1] += d;
    fract(q[0] * q[1])
}

/// 3D hash: maps a 3D coordinate to pseudo-random f32 in [0, 1).
///
/// Uses three-axis decorrelation with distinct irrational constants.
///
/// # Arguments
/// * `p` - Input 3D coordinate [x, y, z]
///
/// # Returns
/// Pseudo-random value in [0, 1)
#[inline]
pub fn hash31(p: [f32; 3]) -> f32 {
    let mut q = [
        fract(p[0] * 0.1031),
        fract(p[1] * 0.1030),
        fract(p[2] * 0.0973),
    ];
    let d = q[0] * q[1] + q[1] * q[2] + q[2] * q[0] + 33.33;
    q[0] += d;
    q[1] += d;
    q[2] += d;
    fract(q[0] * q[1] * q[2])
}

/// 2D-to-2D hash: maps a 2D coordinate to two uncorrelated f32 values in [0, 1).
///
/// # Arguments
/// * `p` - Input 2D coordinate
///
/// # Returns
/// Two pseudo-random values [r0, r1], each in [0, 1)
#[inline]
pub fn hash22(p: [f32; 2]) -> [f32; 2] {
    let q = [
        fract(p[0] * 0.1031),
        fract(p[1] * 0.1030),
        fract(p[0] * 0.0973),
    ];
    let d = q[0] * (q[1] + 33.33) + q[1] * (q[2] + 33.33) + q[2] * (q[0] + 33.33);
    [
        fract((q[0] + d) * (q[2] + d)),
        fract((q[1] + d) * (q[0] + d)),
    ]
}

/// 3D-to-3D hash: maps a 3D coordinate to three uncorrelated f32 values in [0, 1).
///
/// # Arguments
/// * `p` - Input 3D coordinate
///
/// # Returns
/// Three pseudo-random values [r0, r1, r2], each in [0, 1)
#[inline]
pub fn hash33(p: [f32; 3]) -> [f32; 3] {
    let mut q = [
        fract(p[0] * 0.1031),
        fract(p[1] * 0.1030),
        fract(p[2] * 0.0973),
    ];
    let d = q[0] * q[1] + q[1] * q[2] + q[2] * q[0] + 33.33;
    q[0] = fract(q[0] + d);
    q[1] = fract(q[1] + d);
    q[2] = fract(q[2] + d);
    [
        fract((q[0] + q[1]) * q[2]),
        fract((q[1] + q[2]) * q[0]),
        fract((q[2] + q[0]) * q[1]),
    ]
}

// =============================================================================
// T-DEMO-1.29: Value Noise
// =============================================================================

/// Quintic smoothstep for C2 continuous interpolation.
///
/// Uses the polynomial 6t^5 - 15t^4 + 10t^3 (Ken Perlin's improved fade curve).
#[inline]
fn smoothstep_quintic(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// 1D value noise: maps a scalar to smooth pseudo-random value in [-1, 1].
///
/// Interpolates hash values at adjacent integer grid points using a quintic
/// smoothstep for C2 continuity.
///
/// # Arguments
/// * `p` - Input coordinate
///
/// # Returns
/// Smooth noise value in [-1, 1]
#[inline]
pub fn value_noise_1d(p: f32) -> f32 {
    let i = p.floor();
    let f = p - i;

    let u = smoothstep_quintic(f);

    let a = hash11(i) * 2.0 - 1.0;
    let b = hash11(i + 1.0) * 2.0 - 1.0;

    a + u * (b - a)
}

/// 2D value noise: maps a 2D coordinate to smooth pseudo-random value in [-1, 1].
///
/// Bilinearly interpolates hash values at the four corners of the grid cell
/// using quintic smoothstep for C2 continuity.
///
/// # Arguments
/// * `p` - Input 2D coordinate [x, y]
///
/// # Returns
/// Smooth noise value in [-1, 1]
#[inline]
pub fn value_noise_2d(p: [f32; 2]) -> f32 {
    let i = [p[0].floor(), p[1].floor()];
    let f = [p[0] - i[0], p[1] - i[1]];

    let u = [smoothstep_quintic(f[0]), smoothstep_quintic(f[1])];

    let a = hash21(i) * 2.0 - 1.0;
    let b = hash21([i[0] + 1.0, i[1]]) * 2.0 - 1.0;
    let c = hash21([i[0], i[1] + 1.0]) * 2.0 - 1.0;
    let d = hash21([i[0] + 1.0, i[1] + 1.0]) * 2.0 - 1.0;

    let x0 = a + u[0] * (b - a);
    let x1 = c + u[0] * (d - c);
    x0 + u[1] * (x1 - x0)
}

/// 3D value noise: maps a 3D coordinate to smooth pseudo-random value in [-1, 1].
///
/// Trilinearly interpolates hash values at the eight corners of the grid cell
/// using quintic smoothstep for C2 continuity.
///
/// # Arguments
/// * `p` - Input 3D coordinate [x, y, z]
///
/// # Returns
/// Smooth noise value in [-1, 1]
#[inline]
pub fn value_noise_3d(p: [f32; 3]) -> f32 {
    let i = [p[0].floor(), p[1].floor(), p[2].floor()];
    let f = [p[0] - i[0], p[1] - i[1], p[2] - i[2]];

    let u = [
        smoothstep_quintic(f[0]),
        smoothstep_quintic(f[1]),
        smoothstep_quintic(f[2]),
    ];

    // Eight corner hash values (remapped to [-1, 1])
    let h000 = hash31([i[0], i[1], i[2]]) * 2.0 - 1.0;
    let h100 = hash31([i[0] + 1.0, i[1], i[2]]) * 2.0 - 1.0;
    let h010 = hash31([i[0], i[1] + 1.0, i[2]]) * 2.0 - 1.0;
    let h110 = hash31([i[0] + 1.0, i[1] + 1.0, i[2]]) * 2.0 - 1.0;
    let h001 = hash31([i[0], i[1], i[2] + 1.0]) * 2.0 - 1.0;
    let h101 = hash31([i[0] + 1.0, i[1], i[2] + 1.0]) * 2.0 - 1.0;
    let h011 = hash31([i[0], i[1] + 1.0, i[2] + 1.0]) * 2.0 - 1.0;
    let h111 = hash31([i[0] + 1.0, i[1] + 1.0, i[2] + 1.0]) * 2.0 - 1.0;

    // Trilinear interpolation
    let x00 = h000 + u[0] * (h100 - h000);
    let x10 = h010 + u[0] * (h110 - h010);
    let x01 = h001 + u[0] * (h101 - h001);
    let x11 = h011 + u[0] * (h111 - h011);

    let y0 = x00 + u[1] * (x10 - x00);
    let y1 = x01 + u[1] * (x11 - x01);

    y0 + u[2] * (y1 - y0)
}

// =============================================================================
// T-DEMO-1.30: Perlin Noise
// =============================================================================

/// 12 gradient vectors for 3D Perlin noise (edge-centered unit vectors).
const PERLIN_GRADIENTS: [[f32; 3]; 12] = [
    [1.0, 1.0, 0.0],
    [-1.0, 1.0, 0.0],
    [1.0, -1.0, 0.0],
    [-1.0, -1.0, 0.0],
    [1.0, 0.0, 1.0],
    [-1.0, 0.0, 1.0],
    [1.0, 0.0, -1.0],
    [-1.0, 0.0, -1.0],
    [0.0, 1.0, 1.0],
    [0.0, -1.0, 1.0],
    [0.0, 1.0, -1.0],
    [0.0, -1.0, -1.0],
];

/// Normalization factor for edge-centered gradients (1/sqrt(2)).
const GRAD_NORM: f32 = 0.7071067811865475;

/// Compute dot product of gradient and offset for Perlin noise.
#[inline]
fn perlin_gradient(hash_value: f32, offset: [f32; 3]) -> f32 {
    let idx = ((hash_value * 12.0).floor() as usize).min(11);
    let g = PERLIN_GRADIENTS[idx];
    (g[0] * offset[0] + g[1] * offset[1] + g[2] * offset[2]) * GRAD_NORM
}

/// 3D Perlin noise: gradient-based noise with approximately zero mean.
///
/// Computes gradient vectors at each grid corner, takes dot products with
/// offset vectors, and interpolates using quintic smoothstep.
///
/// # Arguments
/// * `p` - Input 3D coordinate [x, y, z]
///
/// # Returns
/// Noise value approximately in [-1, 1], zero mean
#[inline]
pub fn perlin_noise_3d(p: [f32; 3]) -> f32 {
    let i = [p[0].floor(), p[1].floor(), p[2].floor()];
    let f = [p[0] - i[0], p[1] - i[1], p[2] - i[2]];

    let u = [
        smoothstep_quintic(f[0]),
        smoothstep_quintic(f[1]),
        smoothstep_quintic(f[2]),
    ];

    // Eight corner offsets
    let o000 = [f[0], f[1], f[2]];
    let o100 = [f[0] - 1.0, f[1], f[2]];
    let o010 = [f[0], f[1] - 1.0, f[2]];
    let o110 = [f[0] - 1.0, f[1] - 1.0, f[2]];
    let o001 = [f[0], f[1], f[2] - 1.0];
    let o101 = [f[0] - 1.0, f[1], f[2] - 1.0];
    let o011 = [f[0], f[1] - 1.0, f[2] - 1.0];
    let o111 = [f[0] - 1.0, f[1] - 1.0, f[2] - 1.0];

    // Hash values at eight corners
    let h000 = hash31([i[0], i[1], i[2]]);
    let h100 = hash31([i[0] + 1.0, i[1], i[2]]);
    let h010 = hash31([i[0], i[1] + 1.0, i[2]]);
    let h110 = hash31([i[0] + 1.0, i[1] + 1.0, i[2]]);
    let h001 = hash31([i[0], i[1], i[2] + 1.0]);
    let h101 = hash31([i[0] + 1.0, i[1], i[2] + 1.0]);
    let h011 = hash31([i[0], i[1] + 1.0, i[2] + 1.0]);
    let h111 = hash31([i[0] + 1.0, i[1] + 1.0, i[2] + 1.0]);

    // Gradient dot products
    let g000 = perlin_gradient(h000, o000);
    let g100 = perlin_gradient(h100, o100);
    let g010 = perlin_gradient(h010, o010);
    let g110 = perlin_gradient(h110, o110);
    let g001 = perlin_gradient(h001, o001);
    let g101 = perlin_gradient(h101, o101);
    let g011 = perlin_gradient(h011, o011);
    let g111 = perlin_gradient(h111, o111);

    // Trilinear interpolation of gradient dot products
    let x00 = g000 + u[0] * (g100 - g000);
    let x10 = g010 + u[0] * (g110 - g010);
    let x01 = g001 + u[0] * (g101 - g001);
    let x11 = g011 + u[0] * (g111 - g011);

    let y0 = x00 + u[1] * (x10 - x00);
    let y1 = x01 + u[1] * (x11 - x01);

    y0 + u[2] * (y1 - y0)
}

/// 2D Perlin noise for completeness (uses simplified 4-gradient approach).
///
/// # Arguments
/// * `p` - Input 2D coordinate [x, y]
///
/// # Returns
/// Noise value approximately in [-1, 1]
#[inline]
pub fn perlin_noise_2d(p: [f32; 2]) -> f32 {
    let i = [p[0].floor(), p[1].floor()];
    let f = [p[0] - i[0], p[1] - i[1]];

    let u = [smoothstep_quintic(f[0]), smoothstep_quintic(f[1])];

    // 4 gradient directions for 2D
    let gradients: [[f32; 2]; 4] = [[1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]];

    let h00 = hash21(i);
    let h10 = hash21([i[0] + 1.0, i[1]]);
    let h01 = hash21([i[0], i[1] + 1.0]);
    let h11 = hash21([i[0] + 1.0, i[1] + 1.0]);

    let g00 = gradients[(h00 * 4.0).floor() as usize % 4];
    let g10 = gradients[(h10 * 4.0).floor() as usize % 4];
    let g01 = gradients[(h01 * 4.0).floor() as usize % 4];
    let g11 = gradients[(h11 * 4.0).floor() as usize % 4];

    let d00 = g00[0] * f[0] + g00[1] * f[1];
    let d10 = g10[0] * (f[0] - 1.0) + g10[1] * f[1];
    let d01 = g01[0] * f[0] + g01[1] * (f[1] - 1.0);
    let d11 = g11[0] * (f[0] - 1.0) + g11[1] * (f[1] - 1.0);

    let x0 = d00 + u[0] * (d10 - d00);
    let x1 = d01 + u[0] * (d11 - d01);

    (x0 + u[1] * (x1 - x0)) * 0.7071067811865475 // Normalize
}

// =============================================================================
// T-DEMO-1.31: Fractal Brownian Motion (FBM)
// =============================================================================

/// FBM configuration for customizable fractal noise.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FbmConfig {
    /// Number of noise octaves to sum.
    pub octaves: u32,
    /// Frequency multiplier between octaves (typically 2.0).
    pub lacunarity: f32,
    /// Amplitude multiplier between octaves (typically 0.5).
    pub gain: f32,
}

impl Default for FbmConfig {
    fn default() -> Self {
        Self {
            octaves: DEFAULT_OCTAVES,
            lacunarity: DEFAULT_LACUNARITY,
            gain: DEFAULT_GAIN,
        }
    }
}

impl FbmConfig {
    /// Create a new FBM configuration.
    #[inline]
    pub fn new(octaves: u32, lacunarity: f32, gain: f32) -> Self {
        Self {
            octaves: octaves.max(1),
            lacunarity: lacunarity.max(1.0),
            gain: gain.clamp(0.0, 1.0),
        }
    }

    /// Calculate the theoretical maximum amplitude sum.
    #[inline]
    pub fn max_amplitude(&self) -> f32 {
        let mut sum = 0.0;
        let mut amp = 1.0;
        for _ in 0..self.octaves {
            sum += amp;
            amp *= self.gain;
        }
        sum
    }
}

/// 1D FBM using value noise base.
///
/// # Arguments
/// * `p` - Input coordinate
/// * `octaves` - Number of noise layers
/// * `lacunarity` - Frequency multiplier
/// * `gain` - Amplitude multiplier
///
/// # Returns
/// FBM noise value in [-1, 1]
#[inline]
pub fn fbm_1d(p: f32, octaves: u32, lacunarity: f32, gain: f32) -> f32 {
    let mut value = 0.0;
    let mut amplitude = 1.0;
    let mut frequency = 1.0;
    let mut max_amplitude = 0.0;

    for _ in 0..octaves {
        value += amplitude * value_noise_1d(p * frequency);
        max_amplitude += amplitude;
        frequency *= lacunarity;
        amplitude *= gain;
    }

    if max_amplitude < EPSILON {
        0.0
    } else {
        value / max_amplitude
    }
}

/// 2D FBM using value noise base.
///
/// # Arguments
/// * `p` - Input 2D coordinate
/// * `octaves` - Number of noise layers
/// * `lacunarity` - Frequency multiplier
/// * `gain` - Amplitude multiplier
///
/// # Returns
/// FBM noise value in [-1, 1]
#[inline]
pub fn fbm_2d(p: [f32; 2], octaves: u32, lacunarity: f32, gain: f32) -> f32 {
    let mut value = 0.0;
    let mut amplitude = 1.0;
    let mut frequency = 1.0;
    let mut max_amplitude = 0.0;

    for _ in 0..octaves {
        value += amplitude * value_noise_2d([p[0] * frequency, p[1] * frequency]);
        max_amplitude += amplitude;
        frequency *= lacunarity;
        amplitude *= gain;
    }

    if max_amplitude < EPSILON {
        0.0
    } else {
        value / max_amplitude
    }
}

/// 3D FBM using value noise base.
///
/// # Arguments
/// * `p` - Input 3D coordinate
/// * `octaves` - Number of noise layers
/// * `lacunarity` - Frequency multiplier
/// * `gain` - Amplitude multiplier
///
/// # Returns
/// FBM noise value in [-1, 1]
#[inline]
pub fn fbm_3d(p: [f32; 3], octaves: u32, lacunarity: f32, gain: f32) -> f32 {
    let mut value = 0.0;
    let mut amplitude = 1.0;
    let mut frequency = 1.0;
    let mut max_amplitude = 0.0;

    for _ in 0..octaves {
        value += amplitude * value_noise_3d([
            p[0] * frequency,
            p[1] * frequency,
            p[2] * frequency,
        ]);
        max_amplitude += amplitude;
        frequency *= lacunarity;
        amplitude *= gain;
    }

    if max_amplitude < EPSILON {
        0.0
    } else {
        value / max_amplitude
    }
}

/// 3D FBM using Perlin noise base for smoother results.
///
/// # Arguments
/// * `p` - Input 3D coordinate
/// * `octaves` - Number of noise layers
/// * `lacunarity` - Frequency multiplier
/// * `gain` - Amplitude multiplier
///
/// # Returns
/// FBM noise value approximately in [-1, 1]
#[inline]
pub fn fbm_perlin_3d(p: [f32; 3], octaves: u32, lacunarity: f32, gain: f32) -> f32 {
    let mut value = 0.0;
    let mut amplitude = 1.0;
    let mut frequency = 1.0;
    let mut max_amplitude = 0.0;

    for _ in 0..octaves {
        value += amplitude * perlin_noise_3d([
            p[0] * frequency,
            p[1] * frequency,
            p[2] * frequency,
        ]);
        max_amplitude += amplitude;
        frequency *= lacunarity;
        amplitude *= gain;
    }

    if max_amplitude < EPSILON {
        0.0
    } else {
        value / max_amplitude
    }
}

/// Convenience function: 3D FBM with default parameters.
#[inline]
pub fn fbm(p: [f32; 3]) -> f32 {
    fbm_3d(p, DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN)
}

// =============================================================================
// T-DEMO-1.32: Ridged Noise
// =============================================================================

/// 1D ridged noise: 1.0 - abs(FBM) for sharp ridges.
///
/// # Arguments
/// * `p` - Input coordinate
/// * `octaves` - Number of FBM octaves
/// * `lacunarity` - Frequency multiplier
/// * `gain` - Amplitude multiplier
///
/// # Returns
/// Ridged noise value in [0, 1]
#[inline]
pub fn ridged_noise_1d(p: f32, octaves: u32, lacunarity: f32, gain: f32) -> f32 {
    1.0 - fbm_1d(p, octaves, lacunarity, gain).abs()
}

/// 2D ridged noise: 1.0 - abs(FBM) for sharp ridges.
///
/// # Arguments
/// * `p` - Input 2D coordinate
/// * `octaves` - Number of FBM octaves
/// * `lacunarity` - Frequency multiplier
/// * `gain` - Amplitude multiplier
///
/// # Returns
/// Ridged noise value in [0, 1]
#[inline]
pub fn ridged_noise_2d(p: [f32; 2], octaves: u32, lacunarity: f32, gain: f32) -> f32 {
    1.0 - fbm_2d(p, octaves, lacunarity, gain).abs()
}

/// 3D ridged noise: 1.0 - abs(FBM) for sharp ridges.
///
/// # Arguments
/// * `p` - Input 3D coordinate
/// * `octaves` - Number of FBM octaves
/// * `lacunarity` - Frequency multiplier
/// * `gain` - Amplitude multiplier
///
/// # Returns
/// Ridged noise value in [0, 1]
#[inline]
pub fn ridged_noise_3d(p: [f32; 3], octaves: u32, lacunarity: f32, gain: f32) -> f32 {
    1.0 - fbm_3d(p, octaves, lacunarity, gain).abs()
}

/// 3D ridged Perlin noise for smoother terrain ridges.
///
/// # Arguments
/// * `p` - Input 3D coordinate
/// * `octaves` - Number of FBM octaves
/// * `lacunarity` - Frequency multiplier
/// * `gain` - Amplitude multiplier
///
/// # Returns
/// Ridged noise value in [0, 1]
#[inline]
pub fn ridged_perlin_3d(p: [f32; 3], octaves: u32, lacunarity: f32, gain: f32) -> f32 {
    1.0 - fbm_perlin_3d(p, octaves, lacunarity, gain).abs()
}

/// Convenience function: 3D ridged noise with default parameters.
#[inline]
pub fn ridged_noise(p: [f32; 3], octaves: u32) -> f32 {
    ridged_noise_3d(p, octaves, DEFAULT_LACUNARITY, DEFAULT_GAIN)
}

// =============================================================================
// T-DEMO-1.33: Domain Warp
// =============================================================================

/// Domain warp configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DomainWarpConfig {
    /// Warp displacement strength.
    pub strength: f32,
    /// Warp field FBM octaves.
    pub warp_octaves: u32,
    /// Warp field lacunarity.
    pub warp_lacunarity: f32,
    /// Warp field gain.
    pub warp_gain: f32,
    /// Base signal FBM octaves.
    pub base_octaves: u32,
    /// Base signal lacunarity.
    pub base_lacunarity: f32,
    /// Base signal gain.
    pub base_gain: f32,
}

impl Default for DomainWarpConfig {
    fn default() -> Self {
        Self {
            strength: DEFAULT_WARP_STRENGTH,
            warp_octaves: 4,
            warp_lacunarity: DEFAULT_LACUNARITY,
            warp_gain: DEFAULT_GAIN,
            base_octaves: 6,
            base_lacunarity: DEFAULT_LACUNARITY,
            base_gain: DEFAULT_GAIN,
        }
    }
}

/// 2D domain warp: FBM-warped FBM coordinates.
///
/// # Arguments
/// * `p` - Input 2D coordinate
/// * `strength` - Warp displacement magnitude
/// * `warp_octaves` - FBM octaves for warp field
/// * `warp_lacunarity` - Warp frequency multiplier
/// * `warp_gain` - Warp amplitude multiplier
/// * `base_octaves` - FBM octaves for base signal
/// * `base_lacunarity` - Base frequency multiplier
/// * `base_gain` - Base amplitude multiplier
///
/// # Returns
/// Domain-warped FBM noise in [-1, 1]
#[inline]
pub fn domain_warp_2d(
    p: [f32; 2],
    strength: f32,
    warp_octaves: u32,
    warp_lacunarity: f32,
    warp_gain: f32,
    base_octaves: u32,
    base_lacunarity: f32,
    base_gain: f32,
) -> f32 {
    // Warp displacement from two decorrelated FBM evaluations
    let warp_x = fbm_2d(p, warp_octaves, warp_lacunarity, warp_gain);
    let warp_y = fbm_2d(
        [p[0] + 100.0, p[1] + 100.0],
        warp_octaves,
        warp_lacunarity,
        warp_gain,
    );

    // Warped coordinates
    let warped = [p[0] + strength * warp_x, p[1] + strength * warp_y];

    // Evaluate base FBM at warped position
    fbm_2d(warped, base_octaves, base_lacunarity, base_gain)
}

/// 3D domain warp: FBM-warped FBM coordinates.
///
/// # Arguments
/// * `p` - Input 3D coordinate
/// * `strength` - Warp displacement magnitude
/// * `warp_octaves` - FBM octaves for warp field
/// * `warp_lacunarity` - Warp frequency multiplier
/// * `warp_gain` - Warp amplitude multiplier
/// * `base_octaves` - FBM octaves for base signal
/// * `base_lacunarity` - Base frequency multiplier
/// * `base_gain` - Base amplitude multiplier
///
/// # Returns
/// Domain-warped FBM noise in [-1, 1]
#[inline]
pub fn domain_warp_3d(
    p: [f32; 3],
    strength: f32,
    warp_octaves: u32,
    warp_lacunarity: f32,
    warp_gain: f32,
    base_octaves: u32,
    base_lacunarity: f32,
    base_gain: f32,
) -> f32 {
    // Warp displacement from three decorrelated FBM evaluations
    let warp_x = fbm_3d(p, warp_octaves, warp_lacunarity, warp_gain);
    let warp_y = fbm_3d(
        [p[0] + 100.0, p[1] + 100.0, p[2] + 100.0],
        warp_octaves,
        warp_lacunarity,
        warp_gain,
    );
    let warp_z = fbm_3d(
        [p[0] + 200.0, p[1] + 200.0, p[2] + 200.0],
        warp_octaves,
        warp_lacunarity,
        warp_gain,
    );

    // Warped coordinates
    let warped = [
        p[0] + strength * warp_x,
        p[1] + strength * warp_y,
        p[2] + strength * warp_z,
    ];

    // Evaluate base FBM at warped position
    fbm_3d(warped, base_octaves, base_lacunarity, base_gain)
}

/// 3D domain warp using Perlin noise for smoother results.
#[inline]
pub fn domain_warp_perlin_3d(
    p: [f32; 3],
    strength: f32,
    warp_octaves: u32,
    warp_lacunarity: f32,
    warp_gain: f32,
    base_octaves: u32,
    base_lacunarity: f32,
    base_gain: f32,
) -> f32 {
    let warp_x = fbm_perlin_3d(p, warp_octaves, warp_lacunarity, warp_gain);
    let warp_y = fbm_perlin_3d(
        [p[0] + 100.0, p[1] + 100.0, p[2] + 100.0],
        warp_octaves,
        warp_lacunarity,
        warp_gain,
    );
    let warp_z = fbm_perlin_3d(
        [p[0] + 200.0, p[1] + 200.0, p[2] + 200.0],
        warp_octaves,
        warp_lacunarity,
        warp_gain,
    );

    let warped = [
        p[0] + strength * warp_x,
        p[1] + strength * warp_y,
        p[2] + strength * warp_z,
    ];

    fbm_perlin_3d(warped, base_octaves, base_lacunarity, base_gain)
}

/// Convenience function: 3D domain warp with default parameters.
#[inline]
pub fn domain_warp(p: [f32; 3], warp_strength: f32) -> f32 {
    domain_warp_3d(
        p,
        warp_strength,
        4,
        DEFAULT_LACUNARITY,
        DEFAULT_GAIN,
        6,
        DEFAULT_LACUNARITY,
        DEFAULT_GAIN,
    )
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Fractional part of a float (equivalent to WGSL fract).
#[inline]
fn fract(x: f32) -> f32 {
    x - x.floor()
}

/// Dot product of two 2D vectors.
#[inline]
fn dot2(a: [f32; 2], b: [f32; 2]) -> f32 {
    a[0] * b[0] + a[1] * b[1]
}

/// Dot product of two 3D vectors.
#[inline]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

// =============================================================================
// WGSL Code Generation
// =============================================================================

/// Generate WGSL code for hash functions (T-DEMO-1.28).
pub fn hash_wgsl() -> &'static str {
    include_str!("demoscene/noise_hash.wgsl")
}

/// Generate WGSL code for value noise (T-DEMO-1.29).
pub fn value_noise_wgsl() -> &'static str {
    include_str!("demoscene/noise_value.wgsl")
}

/// Generate WGSL code for Perlin noise (T-DEMO-1.30).
pub fn perlin_noise_wgsl() -> &'static str {
    include_str!("demoscene/noise_perlin.wgsl")
}

/// Generate WGSL code for FBM (T-DEMO-1.31).
pub fn fbm_wgsl() -> &'static str {
    include_str!("demoscene/noise_fbm.wgsl")
}

/// Generate WGSL code for ridged noise (T-DEMO-1.32).
pub fn ridged_noise_wgsl() -> &'static str {
    include_str!("demoscene/noise_ridged.wgsl")
}

/// Generate WGSL code for domain warp (T-DEMO-1.33).
pub fn domain_warp_wgsl() -> &'static str {
    include_str!("demoscene/noise_domain_warp.wgsl")
}

/// Generate all noise function WGSL code concatenated.
pub fn all_noise_wgsl() -> String {
    format!(
        "{}\n\n{}\n\n{}\n\n{}\n\n{}\n\n{}",
        hash_wgsl(),
        value_noise_wgsl(),
        perlin_noise_wgsl(),
        fbm_wgsl(),
        ridged_noise_wgsl(),
        domain_warp_wgsl()
    )
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // T-DEMO-1.28: Hash Function Tests
    // ========================================================================

    #[test]
    fn test_hash11_deterministic() {
        let a = hash11(42.0);
        let b = hash11(42.0);
        assert_eq!(a, b, "hash11 should be deterministic");
    }

    #[test]
    fn test_hash11_range() {
        for i in 0..1000 {
            let h = hash11(i as f32 * 0.1);
            assert!(h >= 0.0 && h < 1.0, "hash11({}) = {} should be in [0, 1)", i, h);
        }
    }

    #[test]
    fn test_hash11_different_inputs_different_outputs() {
        let h1 = hash11(0.0);
        let h2 = hash11(1.0);
        assert_ne!(h1, h2, "Different inputs should produce different outputs");
    }

    #[test]
    fn test_hash21_deterministic() {
        let a = hash21([1.0, 2.0]);
        let b = hash21([1.0, 2.0]);
        assert_eq!(a, b, "hash21 should be deterministic");
    }

    #[test]
    fn test_hash21_range() {
        for i in 0..100 {
            for j in 0..100 {
                let h = hash21([i as f32, j as f32]);
                assert!(h >= 0.0 && h < 1.0, "hash21 should be in [0, 1)");
            }
        }
    }

    #[test]
    fn test_hash21_axis_independence() {
        let h1 = hash21([1.0, 0.0]);
        let h2 = hash21([0.0, 1.0]);
        assert_ne!(h1, h2, "Different axis positions should differ");
    }

    #[test]
    fn test_hash31_deterministic() {
        let a = hash31([1.0, 2.0, 3.0]);
        let b = hash31([1.0, 2.0, 3.0]);
        assert_eq!(a, b, "hash31 should be deterministic");
    }

    #[test]
    fn test_hash31_range() {
        for i in 0..50 {
            for j in 0..50 {
                let h = hash31([i as f32, j as f32, (i + j) as f32]);
                assert!(h >= 0.0 && h < 1.0, "hash31 should be in [0, 1)");
            }
        }
    }

    #[test]
    fn test_hash31_negative_inputs() {
        let h = hash31([-1.0, -2.0, -3.0]);
        assert!(h >= 0.0 && h < 1.0, "hash31 with negative inputs should still be in [0, 1)");
    }

    #[test]
    fn test_hash22_deterministic() {
        let a = hash22([1.0, 2.0]);
        let b = hash22([1.0, 2.0]);
        assert_eq!(a, b, "hash22 should be deterministic");
    }

    #[test]
    fn test_hash22_range() {
        for i in 0..100 {
            let h = hash22([i as f32 * 0.7, i as f32 * 1.3]);
            assert!(h[0] >= 0.0 && h[0] < 1.0, "hash22[0] should be in [0, 1)");
            assert!(h[1] >= 0.0 && h[1] < 1.0, "hash22[1] should be in [0, 1)");
        }
    }

    #[test]
    fn test_hash22_components_decorrelated() {
        // Statistical test: components should not be identical
        let mut same_count = 0;
        for i in 0..1000 {
            let h = hash22([i as f32, i as f32 + 0.5]);
            if (h[0] - h[1]).abs() < 0.01 {
                same_count += 1;
            }
        }
        assert!(same_count < 50, "hash22 components should be decorrelated");
    }

    #[test]
    fn test_hash33_deterministic() {
        let a = hash33([1.0, 2.0, 3.0]);
        let b = hash33([1.0, 2.0, 3.0]);
        assert_eq!(a, b, "hash33 should be deterministic");
    }

    #[test]
    fn test_hash33_range() {
        for i in 0..100 {
            let h = hash33([i as f32, i as f32 * 0.5, i as f32 * 0.25]);
            assert!(h[0] >= 0.0 && h[0] < 1.0, "hash33[0] should be in [0, 1)");
            assert!(h[1] >= 0.0 && h[1] < 1.0, "hash33[1] should be in [0, 1)");
            assert!(h[2] >= 0.0 && h[2] < 1.0, "hash33[2] should be in [0, 1)");
        }
    }

    #[test]
    fn test_hash_uniformity() {
        // Statistical test for uniform distribution
        let mut buckets = [0u32; 10];
        for i in 0..10000 {
            let h = hash11(i as f32 * 0.001);
            let bucket = (h * 10.0).floor() as usize;
            if bucket < 10 {
                buckets[bucket] += 1;
            }
        }

        // Each bucket should have roughly 1000 samples
        for (i, &count) in buckets.iter().enumerate() {
            assert!(
                count > 800 && count < 1200,
                "Bucket {} has {} samples, expected ~1000",
                i,
                count
            );
        }
    }

    #[test]
    fn test_hash_no_visible_patterns() {
        // Adjacent inputs should produce uncorrelated outputs
        let mut correlation = 0.0;
        for i in 0..1000 {
            let h1 = hash11(i as f32);
            let h2 = hash11((i + 1) as f32);
            correlation += (h1 - 0.5) * (h2 - 0.5);
        }
        correlation /= 1000.0;
        assert!(
            correlation.abs() < 0.05,
            "Adjacent hash values should be uncorrelated, got {}",
            correlation
        );
    }

    // ========================================================================
    // T-DEMO-1.29: Value Noise Tests
    // ========================================================================

    #[test]
    fn test_value_noise_1d_deterministic() {
        let a = value_noise_1d(3.14159);
        let b = value_noise_1d(3.14159);
        assert_eq!(a, b, "value_noise_1d should be deterministic");
    }

    #[test]
    fn test_value_noise_1d_range() {
        for i in 0..1000 {
            let n = value_noise_1d(i as f32 * 0.1);
            assert!(
                n >= -1.0 && n <= 1.0,
                "value_noise_1d({}) = {} should be in [-1, 1]",
                i,
                n
            );
        }
    }

    #[test]
    fn test_value_noise_1d_continuous() {
        // Check that small input changes produce small output changes
        let base = value_noise_1d(5.0);
        let nearby = value_noise_1d(5.001);
        assert!(
            (base - nearby).abs() < 0.1,
            "Value noise should be continuous"
        );
    }

    #[test]
    fn test_value_noise_1d_at_integers() {
        // At integers, noise is well-defined
        let n = value_noise_1d(10.0);
        assert!(!n.is_nan() && !n.is_infinite());
    }

    #[test]
    fn test_value_noise_2d_deterministic() {
        let a = value_noise_2d([1.5, 2.5]);
        let b = value_noise_2d([1.5, 2.5]);
        assert_eq!(a, b, "value_noise_2d should be deterministic");
    }

    #[test]
    fn test_value_noise_2d_range() {
        for i in 0..100 {
            for j in 0..100 {
                let n = value_noise_2d([i as f32 * 0.1, j as f32 * 0.1]);
                assert!(
                    n >= -1.0 && n <= 1.0,
                    "value_noise_2d should be in [-1, 1]"
                );
            }
        }
    }

    #[test]
    fn test_value_noise_2d_continuous() {
        let base = value_noise_2d([3.0, 4.0]);
        let nearby = value_noise_2d([3.001, 4.001]);
        assert!(
            (base - nearby).abs() < 0.1,
            "Value noise 2D should be continuous"
        );
    }

    #[test]
    fn test_value_noise_3d_deterministic() {
        let a = value_noise_3d([1.0, 2.0, 3.0]);
        let b = value_noise_3d([1.0, 2.0, 3.0]);
        assert_eq!(a, b, "value_noise_3d should be deterministic");
    }

    #[test]
    fn test_value_noise_3d_range() {
        for i in 0..50 {
            for j in 0..50 {
                let n = value_noise_3d([i as f32 * 0.1, j as f32 * 0.1, (i + j) as f32 * 0.05]);
                assert!(
                    n >= -1.0 && n <= 1.0,
                    "value_noise_3d should be in [-1, 1]"
                );
            }
        }
    }

    #[test]
    fn test_value_noise_3d_continuous() {
        let base = value_noise_3d([2.0, 3.0, 4.0]);
        let nearby = value_noise_3d([2.001, 3.001, 4.001]);
        assert!(
            (base - nearby).abs() < 0.1,
            "Value noise 3D should be continuous"
        );
    }

    #[test]
    fn test_value_noise_3d_negative_coords() {
        let n = value_noise_3d([-1.5, -2.5, -3.5]);
        assert!(!n.is_nan() && !n.is_infinite());
        assert!(n >= -1.0 && n <= 1.0);
    }

    #[test]
    fn test_smoothstep_quintic_endpoints() {
        assert_eq!(smoothstep_quintic(0.0), 0.0);
        assert_eq!(smoothstep_quintic(1.0), 1.0);
    }

    #[test]
    fn test_smoothstep_quintic_midpoint() {
        let mid = smoothstep_quintic(0.5);
        assert!((mid - 0.5).abs() < 1e-6, "Midpoint should be 0.5");
    }

    #[test]
    fn test_smoothstep_quintic_monotonic() {
        let mut prev = 0.0;
        for i in 1..=100 {
            let t = i as f32 / 100.0;
            let s = smoothstep_quintic(t);
            assert!(s >= prev, "Smoothstep should be monotonic");
            prev = s;
        }
    }

    // ========================================================================
    // T-DEMO-1.30: Perlin Noise Tests
    // ========================================================================

    #[test]
    fn test_perlin_noise_3d_deterministic() {
        let a = perlin_noise_3d([1.5, 2.5, 3.5]);
        let b = perlin_noise_3d([1.5, 2.5, 3.5]);
        assert_eq!(a, b, "perlin_noise_3d should be deterministic");
    }

    #[test]
    fn test_perlin_noise_3d_range() {
        let mut min_val = f32::MAX;
        let mut max_val = f32::MIN;
        for i in 0..50 {
            for j in 0..50 {
                let n = perlin_noise_3d([i as f32 * 0.1, j as f32 * 0.1, (i + j) as f32 * 0.05]);
                min_val = min_val.min(n);
                max_val = max_val.max(n);
            }
        }
        // Perlin noise typically stays within [-1, 1] for most samples
        assert!(
            min_val >= -1.5 && max_val <= 1.5,
            "Perlin noise should be approximately in [-1, 1], got [{}, {}]",
            min_val,
            max_val
        );
    }

    #[test]
    fn test_perlin_noise_3d_zero_mean() {
        // Statistical test: mean should be near zero
        let mut sum = 0.0;
        let count = 10000;
        for i in 0..count {
            let x = (i as f32 * 0.017) % 100.0;
            let y = (i as f32 * 0.031) % 100.0;
            let z = (i as f32 * 0.053) % 100.0;
            sum += perlin_noise_3d([x, y, z]);
        }
        let mean = sum / count as f32;
        assert!(
            mean.abs() < 0.1,
            "Perlin noise mean should be near zero, got {}",
            mean
        );
    }

    #[test]
    fn test_perlin_noise_3d_continuous() {
        let base = perlin_noise_3d([2.0, 3.0, 4.0]);
        let nearby = perlin_noise_3d([2.001, 3.001, 4.001]);
        assert!(
            (base - nearby).abs() < 0.1,
            "Perlin noise should be continuous"
        );
    }

    #[test]
    fn test_perlin_noise_3d_at_integers() {
        // At integer coordinates, gradient dot products with zero offset
        // should produce values near zero
        let n = perlin_noise_3d([1.0, 2.0, 3.0]);
        assert!(
            n.abs() < 0.5,
            "Perlin noise at integers should be near zero, got {}",
            n
        );
    }

    #[test]
    fn test_perlin_noise_2d_deterministic() {
        let a = perlin_noise_2d([1.5, 2.5]);
        let b = perlin_noise_2d([1.5, 2.5]);
        assert_eq!(a, b, "perlin_noise_2d should be deterministic");
    }

    #[test]
    fn test_perlin_noise_2d_range() {
        for i in 0..100 {
            for j in 0..100 {
                let n = perlin_noise_2d([i as f32 * 0.1, j as f32 * 0.1]);
                assert!(
                    n >= -2.0 && n <= 2.0,
                    "perlin_noise_2d should be approximately in [-1, 1]"
                );
            }
        }
    }

    #[test]
    fn test_perlin_gradient_all_indices() {
        for i in 0..12 {
            let h = i as f32 / 12.0;
            let g = perlin_gradient(h, [0.5, 0.5, 0.5]);
            assert!(!g.is_nan() && !g.is_infinite());
        }
    }

    // ========================================================================
    // T-DEMO-1.31: FBM Tests
    // ========================================================================

    #[test]
    fn test_fbm_1d_deterministic() {
        let a = fbm_1d(3.14, 4, 2.0, 0.5);
        let b = fbm_1d(3.14, 4, 2.0, 0.5);
        assert_eq!(a, b, "fbm_1d should be deterministic");
    }

    #[test]
    fn test_fbm_1d_range() {
        for i in 0..1000 {
            let n = fbm_1d(i as f32 * 0.1, 6, 2.0, 0.5);
            assert!(
                n >= -1.0 && n <= 1.0,
                "fbm_1d should be normalized to [-1, 1]"
            );
        }
    }

    #[test]
    fn test_fbm_1d_single_octave() {
        // Single octave FBM should equal value noise
        let fbm_val = fbm_1d(2.5, 1, 2.0, 0.5);
        let vn_val = value_noise_1d(2.5);
        assert!(
            (fbm_val - vn_val).abs() < 1e-5,
            "Single octave FBM should equal value noise"
        );
    }

    #[test]
    fn test_fbm_2d_deterministic() {
        let a = fbm_2d([1.0, 2.0], 4, 2.0, 0.5);
        let b = fbm_2d([1.0, 2.0], 4, 2.0, 0.5);
        assert_eq!(a, b, "fbm_2d should be deterministic");
    }

    #[test]
    fn test_fbm_2d_range() {
        for i in 0..100 {
            for j in 0..100 {
                let n = fbm_2d([i as f32 * 0.1, j as f32 * 0.1], 6, 2.0, 0.5);
                assert!(
                    n >= -1.0 && n <= 1.0,
                    "fbm_2d should be normalized to [-1, 1]"
                );
            }
        }
    }

    #[test]
    fn test_fbm_3d_deterministic() {
        let a = fbm_3d([1.0, 2.0, 3.0], 4, 2.0, 0.5);
        let b = fbm_3d([1.0, 2.0, 3.0], 4, 2.0, 0.5);
        assert_eq!(a, b, "fbm_3d should be deterministic");
    }

    #[test]
    fn test_fbm_3d_range() {
        for i in 0..50 {
            for j in 0..50 {
                let n = fbm_3d([i as f32 * 0.1, j as f32 * 0.1, (i + j) as f32 * 0.05], 6, 2.0, 0.5);
                assert!(
                    n >= -1.0 && n <= 1.0,
                    "fbm_3d should be normalized to [-1, 1]"
                );
            }
        }
    }

    #[test]
    fn test_fbm_3d_octave_composition() {
        // More octaves should add more detail (higher frequency variation)
        let low_oct = fbm_3d([2.5, 3.5, 4.5], 2, 2.0, 0.5);
        let high_oct = fbm_3d([2.5, 3.5, 4.5], 8, 2.0, 0.5);
        // Values should differ due to additional octaves
        assert!(
            (low_oct - high_oct).abs() > 1e-6 || low_oct == high_oct,
            "Different octave counts should generally produce different values"
        );
    }

    #[test]
    fn test_fbm_perlin_3d_deterministic() {
        let a = fbm_perlin_3d([1.0, 2.0, 3.0], 4, 2.0, 0.5);
        let b = fbm_perlin_3d([1.0, 2.0, 3.0], 4, 2.0, 0.5);
        assert_eq!(a, b, "fbm_perlin_3d should be deterministic");
    }

    #[test]
    fn test_fbm_perlin_3d_range() {
        for i in 0..50 {
            for j in 0..50 {
                let n = fbm_perlin_3d([i as f32 * 0.1, j as f32 * 0.1, (i + j) as f32 * 0.05], 4, 2.0, 0.5);
                assert!(
                    n >= -1.5 && n <= 1.5,
                    "fbm_perlin_3d should be approximately in [-1, 1]"
                );
            }
        }
    }

    #[test]
    fn test_fbm_zero_octaves() {
        let n = fbm_3d([1.0, 2.0, 3.0], 0, 2.0, 0.5);
        assert_eq!(n, 0.0, "Zero octaves should produce 0");
    }

    #[test]
    fn test_fbm_config_default() {
        let config = FbmConfig::default();
        assert_eq!(config.octaves, DEFAULT_OCTAVES);
        assert_eq!(config.lacunarity, DEFAULT_LACUNARITY);
        assert_eq!(config.gain, DEFAULT_GAIN);
    }

    #[test]
    fn test_fbm_config_max_amplitude() {
        let config = FbmConfig::new(4, 2.0, 0.5);
        let expected = 1.0 + 0.5 + 0.25 + 0.125; // 1.875
        assert!((config.max_amplitude() - expected).abs() < 1e-5);
    }

    #[test]
    fn test_fbm_convenience() {
        let a = fbm([1.0, 2.0, 3.0]);
        let b = fbm_3d([1.0, 2.0, 3.0], DEFAULT_OCTAVES, DEFAULT_LACUNARITY, DEFAULT_GAIN);
        assert_eq!(a, b, "Convenience function should match explicit call");
    }

    // ========================================================================
    // T-DEMO-1.32: Ridged Noise Tests
    // ========================================================================

    #[test]
    fn test_ridged_noise_1d_deterministic() {
        let a = ridged_noise_1d(3.14, 4, 2.0, 0.5);
        let b = ridged_noise_1d(3.14, 4, 2.0, 0.5);
        assert_eq!(a, b, "ridged_noise_1d should be deterministic");
    }

    #[test]
    fn test_ridged_noise_1d_range() {
        for i in 0..1000 {
            let n = ridged_noise_1d(i as f32 * 0.1, 4, 2.0, 0.5);
            assert!(
                n >= 0.0 && n <= 1.0,
                "ridged_noise_1d({}) = {} should be in [0, 1]",
                i,
                n
            );
        }
    }

    #[test]
    fn test_ridged_noise_2d_deterministic() {
        let a = ridged_noise_2d([1.0, 2.0], 4, 2.0, 0.5);
        let b = ridged_noise_2d([1.0, 2.0], 4, 2.0, 0.5);
        assert_eq!(a, b, "ridged_noise_2d should be deterministic");
    }

    #[test]
    fn test_ridged_noise_2d_range() {
        for i in 0..100 {
            for j in 0..100 {
                let n = ridged_noise_2d([i as f32 * 0.1, j as f32 * 0.1], 4, 2.0, 0.5);
                assert!(
                    n >= 0.0 && n <= 1.0,
                    "ridged_noise_2d should be in [0, 1]"
                );
            }
        }
    }

    #[test]
    fn test_ridged_noise_3d_deterministic() {
        let a = ridged_noise_3d([1.0, 2.0, 3.0], 4, 2.0, 0.5);
        let b = ridged_noise_3d([1.0, 2.0, 3.0], 4, 2.0, 0.5);
        assert_eq!(a, b, "ridged_noise_3d should be deterministic");
    }

    #[test]
    fn test_ridged_noise_3d_range() {
        for i in 0..50 {
            for j in 0..50 {
                let n = ridged_noise_3d([i as f32 * 0.1, j as f32 * 0.1, (i + j) as f32 * 0.05], 4, 2.0, 0.5);
                assert!(
                    n >= 0.0 && n <= 1.0,
                    "ridged_noise_3d should be in [0, 1]"
                );
            }
        }
    }

    #[test]
    fn test_ridged_noise_3d_sharp_valleys() {
        // Ridged noise should have sharp features where FBM crosses zero
        // Find a point where FBM is near zero and check that ridged is near 1
        let mut found_ridge = false;
        for i in 0..1000 {
            let p = [i as f32 * 0.01, i as f32 * 0.02, i as f32 * 0.03];
            let fbm_val = fbm_3d(p, 4, 2.0, 0.5);
            if fbm_val.abs() < 0.1 {
                let ridged = ridged_noise_3d(p, 4, 2.0, 0.5);
                if ridged > 0.9 {
                    found_ridge = true;
                    break;
                }
            }
        }
        assert!(found_ridge, "Should find at least one sharp ridge");
    }

    #[test]
    fn test_ridged_perlin_3d_deterministic() {
        let a = ridged_perlin_3d([1.0, 2.0, 3.0], 4, 2.0, 0.5);
        let b = ridged_perlin_3d([1.0, 2.0, 3.0], 4, 2.0, 0.5);
        assert_eq!(a, b, "ridged_perlin_3d should be deterministic");
    }

    #[test]
    fn test_ridged_perlin_3d_range() {
        for i in 0..50 {
            for j in 0..50 {
                let n = ridged_perlin_3d([i as f32 * 0.1, j as f32 * 0.1, (i + j) as f32 * 0.05], 4, 2.0, 0.5);
                assert!(
                    n >= 0.0 && n <= 1.0,
                    "ridged_perlin_3d should be in [0, 1]"
                );
            }
        }
    }

    #[test]
    fn test_ridged_noise_convenience() {
        let a = ridged_noise([1.0, 2.0, 3.0], 4);
        let b = ridged_noise_3d([1.0, 2.0, 3.0], 4, DEFAULT_LACUNARITY, DEFAULT_GAIN);
        assert_eq!(a, b, "Convenience function should match explicit call");
    }

    // ========================================================================
    // T-DEMO-1.33: Domain Warp Tests
    // ========================================================================

    #[test]
    fn test_domain_warp_2d_deterministic() {
        let a = domain_warp_2d([1.0, 2.0], 0.5, 3, 2.0, 0.5, 4, 2.0, 0.5);
        let b = domain_warp_2d([1.0, 2.0], 0.5, 3, 2.0, 0.5, 4, 2.0, 0.5);
        assert_eq!(a, b, "domain_warp_2d should be deterministic");
    }

    #[test]
    fn test_domain_warp_2d_range() {
        for i in 0..50 {
            for j in 0..50 {
                let n = domain_warp_2d([i as f32 * 0.1, j as f32 * 0.1], 0.5, 3, 2.0, 0.5, 4, 2.0, 0.5);
                assert!(
                    n >= -1.0 && n <= 1.0,
                    "domain_warp_2d should be in [-1, 1]"
                );
            }
        }
    }

    #[test]
    fn test_domain_warp_2d_zero_strength() {
        // Zero warp strength should equal regular FBM
        let warped = domain_warp_2d([1.0, 2.0], 0.0, 3, 2.0, 0.5, 4, 2.0, 0.5);
        let plain = fbm_2d([1.0, 2.0], 4, 2.0, 0.5);
        assert!(
            (warped - plain).abs() < 1e-5,
            "Zero warp strength should equal plain FBM"
        );
    }

    #[test]
    fn test_domain_warp_3d_deterministic() {
        let a = domain_warp_3d([1.0, 2.0, 3.0], 0.5, 3, 2.0, 0.5, 4, 2.0, 0.5);
        let b = domain_warp_3d([1.0, 2.0, 3.0], 0.5, 3, 2.0, 0.5, 4, 2.0, 0.5);
        assert_eq!(a, b, "domain_warp_3d should be deterministic");
    }

    #[test]
    fn test_domain_warp_3d_range() {
        for i in 0..30 {
            for j in 0..30 {
                let n = domain_warp_3d(
                    [i as f32 * 0.1, j as f32 * 0.1, (i + j) as f32 * 0.05],
                    0.5,
                    3,
                    2.0,
                    0.5,
                    4,
                    2.0,
                    0.5,
                );
                assert!(
                    n >= -1.0 && n <= 1.0,
                    "domain_warp_3d should be in [-1, 1]"
                );
            }
        }
    }

    #[test]
    fn test_domain_warp_3d_zero_strength() {
        let warped = domain_warp_3d([1.0, 2.0, 3.0], 0.0, 3, 2.0, 0.5, 4, 2.0, 0.5);
        let plain = fbm_3d([1.0, 2.0, 3.0], 4, 2.0, 0.5);
        assert!(
            (warped - plain).abs() < 1e-5,
            "Zero warp strength should equal plain FBM"
        );
    }

    #[test]
    fn test_domain_warp_perlin_3d_deterministic() {
        let a = domain_warp_perlin_3d([1.0, 2.0, 3.0], 0.5, 3, 2.0, 0.5, 4, 2.0, 0.5);
        let b = domain_warp_perlin_3d([1.0, 2.0, 3.0], 0.5, 3, 2.0, 0.5, 4, 2.0, 0.5);
        assert_eq!(a, b, "domain_warp_perlin_3d should be deterministic");
    }

    #[test]
    fn test_domain_warp_perlin_3d_range() {
        for i in 0..30 {
            for j in 0..30 {
                let n = domain_warp_perlin_3d(
                    [i as f32 * 0.1, j as f32 * 0.1, (i + j) as f32 * 0.05],
                    0.5,
                    3,
                    2.0,
                    0.5,
                    4,
                    2.0,
                    0.5,
                );
                assert!(
                    n >= -1.5 && n <= 1.5,
                    "domain_warp_perlin_3d should be approximately in [-1, 1]"
                );
            }
        }
    }

    #[test]
    fn test_domain_warp_convenience() {
        let a = domain_warp([1.0, 2.0, 3.0], 0.5);
        let b = domain_warp_3d([1.0, 2.0, 3.0], 0.5, 4, 2.0, 0.5, 6, 2.0, 0.5);
        assert_eq!(a, b, "Convenience function should match explicit call");
    }

    #[test]
    fn test_domain_warp_config_default() {
        let config = DomainWarpConfig::default();
        assert_eq!(config.strength, DEFAULT_WARP_STRENGTH);
        assert_eq!(config.warp_octaves, 4);
        assert_eq!(config.base_octaves, 6);
    }

    // ========================================================================
    // WGSL Code Generation Tests
    // ========================================================================

    #[test]
    fn test_hash_wgsl_not_empty() {
        let code = hash_wgsl();
        assert!(!code.is_empty(), "Hash WGSL should not be empty");
        assert!(code.contains("hash11"), "Should contain hash11 function");
        assert!(code.contains("hash21"), "Should contain hash21 function");
        assert!(code.contains("hash31"), "Should contain hash31 function");
    }

    #[test]
    fn test_value_noise_wgsl_not_empty() {
        let code = value_noise_wgsl();
        assert!(!code.is_empty(), "Value noise WGSL should not be empty");
        assert!(code.contains("value_noise"), "Should contain value_noise function");
    }

    #[test]
    fn test_perlin_noise_wgsl_not_empty() {
        let code = perlin_noise_wgsl();
        assert!(!code.is_empty(), "Perlin noise WGSL should not be empty");
        assert!(code.contains("perlin_noise_3d"), "Should contain perlin_noise_3d function");
    }

    #[test]
    fn test_fbm_wgsl_not_empty() {
        let code = fbm_wgsl();
        assert!(!code.is_empty(), "FBM WGSL should not be empty");
        assert!(code.contains("fbm"), "Should contain fbm function");
    }

    #[test]
    fn test_ridged_noise_wgsl_not_empty() {
        let code = ridged_noise_wgsl();
        assert!(!code.is_empty(), "Ridged noise WGSL should not be empty");
        assert!(code.contains("ridged"), "Should contain ridged function");
    }

    #[test]
    fn test_domain_warp_wgsl_not_empty() {
        let code = domain_warp_wgsl();
        assert!(!code.is_empty(), "Domain warp WGSL should not be empty");
        assert!(code.contains("domain_warp"), "Should contain domain_warp function");
    }

    #[test]
    fn test_all_noise_wgsl() {
        let code = all_noise_wgsl();
        assert!(code.contains("hash11"));
        assert!(code.contains("value_noise"));
        assert!(code.contains("perlin_noise_3d"));
        assert!(code.contains("fbm"));
        assert!(code.contains("ridged"));
        assert!(code.contains("domain_warp"));
    }

    #[test]
    fn test_wgsl_valid_syntax_hash() {
        let code = hash_wgsl();
        // Basic syntax checks
        assert!(code.contains("fn "), "Should contain function definitions");
        assert!(code.contains("-> f32"), "Should have return types");
    }

    #[test]
    fn test_wgsl_valid_syntax_value() {
        let code = value_noise_wgsl();
        assert!(code.contains("fn "), "Should contain function definitions");
        assert!(code.contains("floor("), "Should use floor function");
    }

    #[test]
    fn test_wgsl_valid_syntax_perlin() {
        let code = perlin_noise_wgsl();
        assert!(code.contains("fn "), "Should contain function definitions");
        assert!(code.contains("vec3<f32>"), "Should use vec3 type");
    }

    #[test]
    fn test_wgsl_valid_syntax_fbm() {
        let code = fbm_wgsl();
        assert!(code.contains("fn "), "Should contain function definitions");
        assert!(code.contains("octaves"), "Should reference octaves parameter");
    }

    #[test]
    fn test_wgsl_valid_syntax_ridged() {
        let code = ridged_noise_wgsl();
        assert!(code.contains("fn "), "Should contain function definitions");
        assert!(code.contains("abs("), "Should use abs function");
    }

    #[test]
    fn test_wgsl_valid_syntax_domain_warp() {
        let code = domain_warp_wgsl();
        assert!(code.contains("fn "), "Should contain function definitions");
        assert!(code.contains("strength"), "Should reference strength parameter");
    }

    // ========================================================================
    // Helper Function Tests
    // ========================================================================

    #[test]
    fn test_fract_positive() {
        assert!((fract(3.14) - 0.14).abs() < 1e-5);
        assert!((fract(0.5) - 0.5).abs() < 1e-5);
        assert!((fract(1.0) - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_fract_negative() {
        // fract(-0.5) = -0.5 - floor(-0.5) = -0.5 - (-1) = 0.5
        assert!((fract(-0.5) - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_dot2() {
        let a = [1.0, 2.0];
        let b = [3.0, 4.0];
        assert_eq!(dot2(a, b), 11.0); // 1*3 + 2*4 = 11
    }

    #[test]
    fn test_dot3() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        assert_eq!(dot3(a, b), 32.0); // 1*4 + 2*5 + 3*6 = 32
    }

    // ========================================================================
    // Edge Case Tests
    // ========================================================================

    #[test]
    fn test_noise_at_origin() {
        let h = hash31([0.0, 0.0, 0.0]);
        assert!(h >= 0.0 && h < 1.0);

        let v = value_noise_3d([0.0, 0.0, 0.0]);
        assert!(v >= -1.0 && v <= 1.0);

        let p = perlin_noise_3d([0.0, 0.0, 0.0]);
        assert!(p >= -1.5 && p <= 1.5);
    }

    #[test]
    fn test_noise_at_large_coords() {
        let h = hash31([10000.0, 20000.0, 30000.0]);
        assert!(h >= 0.0 && h < 1.0);

        let v = value_noise_3d([10000.0, 20000.0, 30000.0]);
        assert!(v >= -1.0 && v <= 1.0);

        let p = perlin_noise_3d([10000.0, 20000.0, 30000.0]);
        assert!(!p.is_nan() && !p.is_infinite());
    }

    #[test]
    fn test_noise_nan_propagation() {
        // Ensure NaN inputs don't cause panics (though output may be NaN)
        let h = hash31([f32::NAN, 1.0, 2.0]);
        // Just verify it doesn't panic
        let _ = h;
    }

    #[test]
    fn test_fbm_extreme_parameters() {
        // Very high gain (unrealistic but should not crash)
        let n = fbm_3d([1.0, 2.0, 3.0], 10, 2.0, 0.99);
        assert!(!n.is_nan() && !n.is_infinite());

        // Very low gain
        let n = fbm_3d([1.0, 2.0, 3.0], 10, 2.0, 0.01);
        assert!(!n.is_nan() && !n.is_infinite());

        // High lacunarity
        let n = fbm_3d([1.0, 2.0, 3.0], 4, 10.0, 0.5);
        assert!(!n.is_nan() && !n.is_infinite());
    }

    #[test]
    fn test_domain_warp_high_strength() {
        // High warp strength should still produce valid output
        let n = domain_warp_3d([1.0, 2.0, 3.0], 10.0, 3, 2.0, 0.5, 4, 2.0, 0.5);
        assert!(!n.is_nan() && !n.is_infinite());
        assert!(n >= -1.0 && n <= 1.0);
    }

    #[test]
    fn test_constants() {
        assert_eq!(DEFAULT_OCTAVES, 6);
        assert_eq!(DEFAULT_LACUNARITY, 2.0);
        assert_eq!(DEFAULT_GAIN, 0.5);
        assert_eq!(DEFAULT_WARP_STRENGTH, 0.5);
    }

    #[test]
    fn test_perlin_gradients_count() {
        assert_eq!(PERLIN_GRADIENTS.len(), 12);
    }

    #[test]
    fn test_perlin_gradients_edge_centered() {
        // Each gradient should have exactly two non-zero components
        for g in PERLIN_GRADIENTS.iter() {
            let non_zero = g.iter().filter(|&&v| v != 0.0).count();
            assert_eq!(non_zero, 2, "Gradients should have exactly 2 non-zero components");
        }
    }

    #[test]
    fn test_grad_norm_constant() {
        assert!((GRAD_NORM - 1.0 / 2.0_f32.sqrt()).abs() < 1e-10);
    }
}
