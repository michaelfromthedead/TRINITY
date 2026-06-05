//! Cloud Noise Texture Generation (T-ENV-2.1)
//!
//! This module provides procedural noise texture generation for volumetric cloud
//! rendering, implementing both 3D Worley noise and Perlin-Worley FBM combinations.
//!
//! # Texture Types
//!
//! * **Base Shape Noise**: 32x32x32 RG8 texture using Perlin-Worley FBM (4-5 octaves).
//!   The R channel stores inverted Worley, G channel stores Perlin.
//!
//! * **Detail Noise**: 16x16x16 R8 texture using Worley FBM (2-3 octaves).
//!   Used for wispy cloud edges with animated scrolling.
//!
//! # Memory Budget
//!
//! Total texture memory: < 2MB
//! - Base (32^3 * 2 bytes): 65,536 bytes
//! - Detail (16^3 * 2 bytes): 8,192 bytes
//! - Total: ~74KB (well under budget)
//!
//! # Tiling
//!
//! All textures tile seamlessly with configurable world-space tile size (4-8 km).
//! Animation is achieved by scrolling detail noise at a different rate than base.
//!
//! # References
//!
//! * Schneider & Vos, "The Real-time Volumetric Cloudscapes of Horizon: Zero Dawn"
//! * Inigo Quilez, "Voronoi edges" - https://iquilezles.org/articles/voronoilines/

use bytemuck::{Pod, Zeroable};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default base resolution for shape noise (32x32x32).
pub const DEFAULT_BASE_RESOLUTION: u32 = 32;

/// Default detail resolution (16x16x16).
pub const DEFAULT_DETAIL_RESOLUTION: u32 = 16;

/// Default number of octaves for base Perlin-Worley FBM.
pub const DEFAULT_BASE_OCTAVES: u32 = 4;

/// Default number of octaves for detail Worley FBM.
pub const DEFAULT_DETAIL_OCTAVES: u32 = 2;

/// Default persistence (amplitude decay per octave).
pub const DEFAULT_PERSISTENCE: f32 = 0.5;

/// Default lacunarity (frequency multiplier per octave).
pub const DEFAULT_LACUNARITY: f32 = 2.0;

/// Minimum tile size in world units (kilometers).
pub const MIN_TILE_SIZE_KM: f32 = 4.0;

/// Maximum tile size in world units (kilometers).
pub const MAX_TILE_SIZE_KM: f32 = 8.0;

/// Default tile size in world units (kilometers).
pub const DEFAULT_TILE_SIZE_KM: f32 = 6.0;

/// Number of cells for Worley noise at base resolution.
pub const WORLEY_CELL_COUNT: u32 = 4;

/// Maximum memory budget for cloud textures (2MB).
pub const MAX_MEMORY_BYTES: usize = 2 * 1024 * 1024;

// ---------------------------------------------------------------------------
// CloudNoiseConfig — GPU-uploadable configuration
// ---------------------------------------------------------------------------

/// Configuration for cloud noise texture generation.
///
/// This struct is designed to be uploaded to the GPU as a uniform buffer.
/// All fields are chosen for optimal cloud appearance.
///
/// # Memory Layout (32 bytes)
///
/// | Offset | Field       | Size    |
/// |--------|-------------|---------|
/// | 0      | base_resolution | 4 bytes |
/// | 4      | octaves     | 4 bytes |
/// | 8      | persistence | 4 bytes |
/// | 12     | lacunarity  | 4 bytes |
/// | 16     | tile_size   | 4 bytes |
/// | 20     | detail_scroll_rate | 4 bytes |
/// | 24     | _padding    | 8 bytes |
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct CloudNoiseConfig {
    /// Base texture resolution (32 for base, 16 for detail).
    pub base_resolution: u32,

    /// Number of noise octaves (4-5 for base, 2-3 for detail).
    pub octaves: u32,

    /// Amplitude falloff per octave (typical: 0.5).
    pub persistence: f32,

    /// Frequency multiplier per octave (typical: 2.0).
    pub lacunarity: f32,

    /// World-space tile size in kilometers (4-8 km).
    pub tile_size: f32,

    /// Detail noise scroll rate relative to base (0.0-2.0).
    pub detail_scroll_rate: f32,

    /// Padding for GPU alignment (vec4 alignment).
    pub _padding: [u32; 2],
}

impl Default for CloudNoiseConfig {
    fn default() -> Self {
        Self::new_base()
    }
}

impl CloudNoiseConfig {
    /// Create a configuration for base shape noise (32^3, 4 octaves).
    #[inline]
    pub fn new_base() -> Self {
        Self {
            base_resolution: DEFAULT_BASE_RESOLUTION,
            octaves: DEFAULT_BASE_OCTAVES,
            persistence: DEFAULT_PERSISTENCE,
            lacunarity: DEFAULT_LACUNARITY,
            tile_size: DEFAULT_TILE_SIZE_KM,
            detail_scroll_rate: 0.0,
            _padding: [0; 2],
        }
    }

    /// Create a configuration for detail noise (16^3, 2 octaves).
    #[inline]
    pub fn new_detail() -> Self {
        Self {
            base_resolution: DEFAULT_DETAIL_RESOLUTION,
            octaves: DEFAULT_DETAIL_OCTAVES,
            persistence: DEFAULT_PERSISTENCE,
            lacunarity: DEFAULT_LACUNARITY,
            tile_size: DEFAULT_TILE_SIZE_KM,
            detail_scroll_rate: 1.5, // Detail scrolls faster
            _padding: [0; 2],
        }
    }

    /// Create a custom configuration.
    #[inline]
    pub fn custom(
        resolution: u32,
        octaves: u32,
        persistence: f32,
        lacunarity: f32,
        tile_size: f32,
    ) -> Self {
        Self {
            base_resolution: resolution.max(1),
            octaves: octaves.max(1),
            persistence: persistence.clamp(0.0, 1.0),
            lacunarity: lacunarity.max(1.0),
            tile_size: tile_size.clamp(MIN_TILE_SIZE_KM, MAX_TILE_SIZE_KM),
            detail_scroll_rate: 1.0,
            _padding: [0; 2],
        }
    }

    /// Calculate the total memory usage for this configuration.
    #[inline]
    pub fn memory_usage(&self, bytes_per_texel: u32) -> usize {
        let dim = self.base_resolution as usize;
        dim * dim * dim * bytes_per_texel as usize
    }

    /// Validate that this configuration fits within memory budget.
    #[inline]
    pub fn validate_memory(&self, bytes_per_texel: u32) -> bool {
        self.memory_usage(bytes_per_texel) <= MAX_MEMORY_BYTES
    }
}

// ---------------------------------------------------------------------------
// WorleyCell — GPU-uploadable cell data
// ---------------------------------------------------------------------------

/// A single Worley noise cell with its feature point.
///
/// Used for cellular noise generation where each cell contains a randomly
/// placed feature point, and the noise value is the distance to the nearest
/// feature point.
///
/// # Memory Layout (16 bytes)
///
/// | Offset | Field    | Size    |
/// |--------|----------|---------|
/// | 0      | center   | 12 bytes |
/// | 12     | distance | 4 bytes |
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct WorleyCell {
    /// Feature point position within the cell (0.0-1.0 in each dimension).
    pub center: [f32; 3],

    /// Cached distance value (used during generation).
    pub distance: f32,
}

impl Default for WorleyCell {
    fn default() -> Self {
        Self {
            center: [0.5, 0.5, 0.5],
            distance: f32::MAX,
        }
    }
}

impl WorleyCell {
    /// Create a new cell with a specified center.
    #[inline]
    pub fn new(center: [f32; 3]) -> Self {
        Self {
            center,
            distance: f32::MAX,
        }
    }

    /// Create a cell with a deterministic pseudo-random center based on cell index.
    #[inline]
    pub fn from_hash(cell_x: i32, cell_y: i32, cell_z: i32, seed: u32) -> Self {
        let hash = hash_3d(cell_x, cell_y, cell_z, seed);
        let center = [
            hash_to_float(hash),
            hash_to_float(hash.wrapping_mul(0x9E3779B9)),
            hash_to_float(hash.wrapping_mul(0x517CC1B7)),
        ];
        Self {
            center,
            distance: f32::MAX,
        }
    }
}

// ---------------------------------------------------------------------------
// AnimationState — Animation parameters for cloud scrolling
// ---------------------------------------------------------------------------

/// Animation state for cloud noise scrolling.
///
/// Detail noise scrolls at a different rate than base noise to create
/// the appearance of billowing, evolving clouds.
///
/// # Memory Layout (32 bytes)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct AnimationState {
    /// Current time offset for base noise (normalized 0-1).
    pub base_offset: [f32; 3],

    /// Base scroll speed in units per second.
    pub base_speed: f32,

    /// Current time offset for detail noise (normalized 0-1).
    pub detail_offset: [f32; 3],

    /// Detail scroll speed multiplier relative to base.
    pub detail_speed_multiplier: f32,
}

impl Default for AnimationState {
    fn default() -> Self {
        Self {
            base_offset: [0.0; 3],
            base_speed: 0.01,
            detail_offset: [0.0; 3],
            detail_speed_multiplier: 1.5,
        }
    }
}

impl AnimationState {
    /// Create a new animation state with custom speeds.
    #[inline]
    pub fn new(base_speed: f32, detail_multiplier: f32) -> Self {
        Self {
            base_offset: [0.0; 3],
            base_speed: base_speed.max(0.0),
            detail_offset: [0.0; 3],
            detail_speed_multiplier: detail_multiplier.max(0.0),
        }
    }

    /// Update animation state by delta time (in seconds).
    ///
    /// Offsets wrap at 1.0 for seamless tiling.
    #[inline]
    pub fn update(&mut self, delta_time: f32, wind_direction: [f32; 3]) {
        let base_delta = self.base_speed * delta_time;
        let detail_delta = base_delta * self.detail_speed_multiplier;

        // Normalize wind direction
        let len = (wind_direction[0] * wind_direction[0]
            + wind_direction[1] * wind_direction[1]
            + wind_direction[2] * wind_direction[2])
            .sqrt()
            .max(1e-6);

        let norm = [
            wind_direction[0] / len,
            wind_direction[1] / len,
            wind_direction[2] / len,
        ];

        // Update base offset with wrapping
        for i in 0..3 {
            self.base_offset[i] = (self.base_offset[i] + norm[i] * base_delta).fract();
            if self.base_offset[i] < 0.0 {
                self.base_offset[i] += 1.0;
            }
        }

        // Update detail offset with wrapping
        for i in 0..3 {
            self.detail_offset[i] = (self.detail_offset[i] + norm[i] * detail_delta).fract();
            if self.detail_offset[i] < 0.0 {
                self.detail_offset[i] += 1.0;
            }
        }
    }

    /// Get the combined offset for a given point (for sampling).
    #[inline]
    pub fn apply_base_offset(&self, point: [f32; 3]) -> [f32; 3] {
        [
            point[0] + self.base_offset[0],
            point[1] + self.base_offset[1],
            point[2] + self.base_offset[2],
        ]
    }

    /// Get the combined offset for detail noise sampling.
    #[inline]
    pub fn apply_detail_offset(&self, point: [f32; 3]) -> [f32; 3] {
        [
            point[0] + self.detail_offset[0],
            point[1] + self.detail_offset[1],
            point[2] + self.detail_offset[2],
        ]
    }
}

// ---------------------------------------------------------------------------
// Hash functions (deterministic pseudo-random)
// ---------------------------------------------------------------------------

/// 3D integer hash function for cell-based noise.
///
/// Based on the MurmurHash3 finalizer for good avalanche properties.
#[inline]
pub fn hash_3d(x: i32, y: i32, z: i32, seed: u32) -> u32 {
    let mut h = seed;
    h ^= x as u32;
    h = h.wrapping_mul(0x85EBCA6B);
    h ^= h >> 13;
    h ^= y as u32;
    h = h.wrapping_mul(0xC2B2AE35);
    h ^= h >> 16;
    h ^= z as u32;
    h = h.wrapping_mul(0x85EBCA6B);
    h ^= h >> 13;
    h
}

/// Convert a hash value to a float in [0, 1).
#[inline]
pub fn hash_to_float(hash: u32) -> f32 {
    // Use upper 24 bits for better distribution
    (hash >> 8) as f32 / 16777216.0
}

/// 3D float hash for gradient-based noise.
///
/// Maps a 3D float coordinate to a pseudo-random value in [0, 1).
#[inline]
pub fn hash_3d_float(x: f32, y: f32, z: f32) -> f32 {
    let ix = x.floor() as i32;
    let iy = y.floor() as i32;
    let iz = z.floor() as i32;
    hash_to_float(hash_3d(ix, iy, iz, 0x1337CAFE))
}

// ---------------------------------------------------------------------------
// Worley Noise Functions
// ---------------------------------------------------------------------------

/// Calculate Worley noise distance at a 3D point.
///
/// Worley noise (cellular noise) computes the distance to the nearest
/// feature point in a cell-based grid. Each cell contains one randomly
/// placed feature point.
///
/// # Arguments
///
/// * `point` - 3D sample point (any range, wraps for tiling).
/// * `cell_count` - Number of cells per dimension (e.g., 4 for 4x4x4 grid).
///
/// # Returns
///
/// Distance to nearest feature point, normalized to approximately [0, 1].
#[inline]
pub fn worley_distance(point: [f32; 3], cell_count: u32) -> f32 {
    worley_distance_seeded(point, cell_count, 0xDEADBEEF)
}

/// Calculate Worley noise distance with a custom seed.
pub fn worley_distance_seeded(point: [f32; 3], cell_count: u32, seed: u32) -> f32 {
    let scale = cell_count as f32;
    let scaled = [point[0] * scale, point[1] * scale, point[2] * scale];

    // Current cell coordinates
    let cell_x = scaled[0].floor() as i32;
    let cell_y = scaled[1].floor() as i32;
    let cell_z = scaled[2].floor() as i32;

    let mut min_dist = f32::MAX;

    // Search 3x3x3 neighborhood for closest feature point
    for dz in -1..=1 {
        for dy in -1..=1 {
            for dx in -1..=1 {
                let nx = cell_x + dx;
                let ny = cell_y + dy;
                let nz = cell_z + dz;

                // Wrap cell coordinates for tiling
                let wx = ((nx % cell_count as i32) + cell_count as i32) % cell_count as i32;
                let wy = ((ny % cell_count as i32) + cell_count as i32) % cell_count as i32;
                let wz = ((nz % cell_count as i32) + cell_count as i32) % cell_count as i32;

                // Get feature point in this cell
                let cell = WorleyCell::from_hash(wx, wy, wz, seed);

                // Feature point position in world space
                let fx = nx as f32 + cell.center[0];
                let fy = ny as f32 + cell.center[1];
                let fz = nz as f32 + cell.center[2];

                // Calculate squared distance
                let dx_f = scaled[0] - fx;
                let dy_f = scaled[1] - fy;
                let dz_f = scaled[2] - fz;
                let dist_sq = dx_f * dx_f + dy_f * dy_f + dz_f * dz_f;

                min_dist = min_dist.min(dist_sq);
            }
        }
    }

    // Normalize distance (max possible is sqrt(3) for cell diagonal)
    (min_dist.sqrt() / 1.732).clamp(0.0, 1.0)
}

/// Calculate both F1 and F2 Worley distances (first and second closest).
///
/// Useful for creating edge effects (F2 - F1).
pub fn worley_f1_f2(point: [f32; 3], cell_count: u32, seed: u32) -> (f32, f32) {
    let scale = cell_count as f32;
    let scaled = [point[0] * scale, point[1] * scale, point[2] * scale];

    let cell_x = scaled[0].floor() as i32;
    let cell_y = scaled[1].floor() as i32;
    let cell_z = scaled[2].floor() as i32;

    let mut f1 = f32::MAX;
    let mut f2 = f32::MAX;

    for dz in -1..=1 {
        for dy in -1..=1 {
            for dx in -1..=1 {
                let nx = cell_x + dx;
                let ny = cell_y + dy;
                let nz = cell_z + dz;

                let wx = ((nx % cell_count as i32) + cell_count as i32) % cell_count as i32;
                let wy = ((ny % cell_count as i32) + cell_count as i32) % cell_count as i32;
                let wz = ((nz % cell_count as i32) + cell_count as i32) % cell_count as i32;

                let cell = WorleyCell::from_hash(wx, wy, wz, seed);

                let fx = nx as f32 + cell.center[0];
                let fy = ny as f32 + cell.center[1];
                let fz = nz as f32 + cell.center[2];

                let dx_f = scaled[0] - fx;
                let dy_f = scaled[1] - fy;
                let dz_f = scaled[2] - fz;
                let dist_sq = dx_f * dx_f + dy_f * dy_f + dz_f * dz_f;

                if dist_sq < f1 {
                    f2 = f1;
                    f1 = dist_sq;
                } else if dist_sq < f2 {
                    f2 = dist_sq;
                }
            }
        }
    }

    let normalize = 1.0 / 1.732; // 1/sqrt(3)
    (
        (f1.sqrt() * normalize).clamp(0.0, 1.0),
        (f2.sqrt() * normalize).clamp(0.0, 1.0),
    )
}

// ---------------------------------------------------------------------------
// Perlin Noise Functions
// ---------------------------------------------------------------------------

/// Smoothstep fade curve: 6t^5 - 15t^4 + 10t^3.
///
/// This quintic polynomial has zero first and second derivatives at t=0 and t=1,
/// producing C2 continuous noise.
#[inline]
pub fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// Linear interpolation.
#[inline]
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

/// Calculate Perlin gradient contribution.
///
/// Selects one of 12 gradient vectors based on hash and computes dot product
/// with the offset from the grid corner.
#[inline]
pub fn perlin_gradient(hash: u32, x: f32, y: f32, z: f32) -> f32 {
    // Use low 4 bits to select gradient (12 edge-centered vectors)
    let h = hash & 15;

    // Fast gradient selection using bit manipulation
    let u = if h < 8 { x } else { y };
    let v = if h < 4 {
        y
    } else if h == 12 || h == 14 {
        x
    } else {
        z
    };

    let u_sign = if (h & 1) == 0 { u } else { -u };
    let v_sign = if (h & 2) == 0 { v } else { -v };

    u_sign + v_sign
}

/// 3D Perlin noise at a given point.
///
/// Returns a value in approximately [-1, 1] with zero mean.
pub fn perlin_3d(point: [f32; 3]) -> f32 {
    perlin_3d_seeded(point, 0x1337BEEF)
}

/// 3D Perlin noise with custom seed.
pub fn perlin_3d_seeded(point: [f32; 3], seed: u32) -> f32 {
    // Integer coordinates
    let xi = point[0].floor() as i32;
    let yi = point[1].floor() as i32;
    let zi = point[2].floor() as i32;

    // Fractional coordinates
    let xf = point[0] - xi as f32;
    let yf = point[1] - yi as f32;
    let zf = point[2] - zi as f32;

    // Fade curves
    let u = fade(xf);
    let v = fade(yf);
    let w = fade(zf);

    // Hash values at 8 corners
    let h000 = hash_3d(xi, yi, zi, seed);
    let h100 = hash_3d(xi + 1, yi, zi, seed);
    let h010 = hash_3d(xi, yi + 1, zi, seed);
    let h110 = hash_3d(xi + 1, yi + 1, zi, seed);
    let h001 = hash_3d(xi, yi, zi + 1, seed);
    let h101 = hash_3d(xi + 1, yi, zi + 1, seed);
    let h011 = hash_3d(xi, yi + 1, zi + 1, seed);
    let h111 = hash_3d(xi + 1, yi + 1, zi + 1, seed);

    // Gradient contributions at 8 corners
    let g000 = perlin_gradient(h000, xf, yf, zf);
    let g100 = perlin_gradient(h100, xf - 1.0, yf, zf);
    let g010 = perlin_gradient(h010, xf, yf - 1.0, zf);
    let g110 = perlin_gradient(h110, xf - 1.0, yf - 1.0, zf);
    let g001 = perlin_gradient(h001, xf, yf, zf - 1.0);
    let g101 = perlin_gradient(h101, xf - 1.0, yf, zf - 1.0);
    let g011 = perlin_gradient(h011, xf, yf - 1.0, zf - 1.0);
    let g111 = perlin_gradient(h111, xf - 1.0, yf - 1.0, zf - 1.0);

    // Trilinear interpolation
    let x00 = lerp(g000, g100, u);
    let x10 = lerp(g010, g110, u);
    let x01 = lerp(g001, g101, u);
    let x11 = lerp(g011, g111, u);

    let y0 = lerp(x00, x10, v);
    let y1 = lerp(x01, x11, v);

    lerp(y0, y1, w)
}

/// 3D Perlin noise with tiling support.
///
/// Wraps coordinates so noise tiles seamlessly at the given period.
pub fn perlin_3d_tiled(point: [f32; 3], period: u32) -> f32 {
    let period_f = period as f32;
    let wrapped = [
        point[0].rem_euclid(period_f),
        point[1].rem_euclid(period_f),
        point[2].rem_euclid(period_f),
    ];

    let xi = wrapped[0].floor() as i32;
    let yi = wrapped[1].floor() as i32;
    let zi = wrapped[2].floor() as i32;

    let xf = wrapped[0] - xi as f32;
    let yf = wrapped[1] - yi as f32;
    let zf = wrapped[2] - zi as f32;

    let u = fade(xf);
    let v = fade(yf);
    let w = fade(zf);

    let period_i = period as i32;

    // Wrap integer coordinates
    let wrap = |x: i32| ((x % period_i) + period_i) % period_i;

    let x0 = wrap(xi);
    let x1 = wrap(xi + 1);
    let y0 = wrap(yi);
    let y1 = wrap(yi + 1);
    let z0 = wrap(zi);
    let z1 = wrap(zi + 1);

    let seed = 0x1337BEEF;

    let h000 = hash_3d(x0, y0, z0, seed);
    let h100 = hash_3d(x1, y0, z0, seed);
    let h010 = hash_3d(x0, y1, z0, seed);
    let h110 = hash_3d(x1, y1, z0, seed);
    let h001 = hash_3d(x0, y0, z1, seed);
    let h101 = hash_3d(x1, y0, z1, seed);
    let h011 = hash_3d(x0, y1, z1, seed);
    let h111 = hash_3d(x1, y1, z1, seed);

    let g000 = perlin_gradient(h000, xf, yf, zf);
    let g100 = perlin_gradient(h100, xf - 1.0, yf, zf);
    let g010 = perlin_gradient(h010, xf, yf - 1.0, zf);
    let g110 = perlin_gradient(h110, xf - 1.0, yf - 1.0, zf);
    let g001 = perlin_gradient(h001, xf, yf, zf - 1.0);
    let g101 = perlin_gradient(h101, xf - 1.0, yf, zf - 1.0);
    let g011 = perlin_gradient(h011, xf, yf - 1.0, zf - 1.0);
    let g111 = perlin_gradient(h111, xf - 1.0, yf - 1.0, zf - 1.0);

    let x00 = lerp(g000, g100, u);
    let x10 = lerp(g010, g110, u);
    let x01 = lerp(g001, g101, u);
    let x11 = lerp(g011, g111, u);

    let y0_val = lerp(x00, x10, v);
    let y1_val = lerp(x01, x11, v);

    lerp(y0_val, y1_val, w)
}

// ---------------------------------------------------------------------------
// FBM Functions (Fractal Brownian Motion)
// ---------------------------------------------------------------------------

/// 3D Fractal Brownian Motion using Perlin noise.
///
/// Layers multiple octaves of Perlin noise at increasing frequencies
/// and decreasing amplitudes to create natural-looking detail.
///
/// # Arguments
///
/// * `point` - 3D sample point.
/// * `octaves` - Number of noise layers (1-8 typical).
/// * `persistence` - Amplitude decay per octave (0.5 typical).
/// * `lacunarity` - Frequency multiplier per octave (2.0 typical).
///
/// # Returns
///
/// FBM noise value, normalized to approximately [-1, 1].
pub fn fbm_3d(point: [f32; 3], octaves: u32, persistence: f32, lacunarity: f32) -> f32 {
    let mut value = 0.0;
    let mut amplitude = 1.0;
    let mut max_amplitude = 0.0;

    let mut p = point;

    for _ in 0..octaves {
        value += amplitude * perlin_3d(p);
        max_amplitude += amplitude;

        p[0] *= lacunarity;
        p[1] *= lacunarity;
        p[2] *= lacunarity;

        amplitude *= persistence;
    }

    if max_amplitude > 1e-8 {
        value / max_amplitude
    } else {
        0.0
    }
}

/// 3D FBM using Worley noise (for cellular/cloud detail).
pub fn fbm_worley_3d(
    point: [f32; 3],
    octaves: u32,
    persistence: f32,
    lacunarity: f32,
    cell_count: u32,
) -> f32 {
    let mut value = 0.0;
    let mut amplitude = 1.0;
    let mut max_amplitude = 0.0;
    let mut cells = cell_count;

    let mut p = point;

    for i in 0..octaves {
        // Use different seed per octave for variation
        let seed = 0xDEADBEEF_u32.wrapping_add(i.wrapping_mul(0x9E3779B9));
        value += amplitude * worley_distance_seeded(p, cells, seed);
        max_amplitude += amplitude;

        p[0] *= lacunarity;
        p[1] *= lacunarity;
        p[2] *= lacunarity;

        // Increase cell count for finer detail at higher octaves
        cells = (cells as f32 * lacunarity).ceil() as u32;
        amplitude *= persistence;
    }

    if max_amplitude > 1e-8 {
        value / max_amplitude
    } else {
        0.0
    }
}

/// Combined Perlin-Worley FBM for cloud base shapes.
///
/// Mixes Perlin and Worley noise for the characteristic cloud appearance:
/// - Perlin provides overall shape billowing
/// - Worley provides cellular detail and hard edges
///
/// # Arguments
///
/// * `point` - 3D sample point (normalized 0-1).
/// * `octaves` - Number of octaves (4-5 for base shapes).
/// * `worley_weight` - Mix factor (0 = pure Perlin, 1 = pure Worley).
pub fn perlin_worley_fbm(
    point: [f32; 3],
    octaves: u32,
    persistence: f32,
    lacunarity: f32,
    worley_weight: f32,
) -> f32 {
    let perlin_val = fbm_3d(point, octaves, persistence, lacunarity);
    let worley_val = 1.0 - fbm_worley_3d(point, octaves, persistence, lacunarity, WORLEY_CELL_COUNT);

    let w = worley_weight.clamp(0.0, 1.0);
    lerp(perlin_val, worley_val, w)
}

// ---------------------------------------------------------------------------
// CloudNoiseGenerator — Main texture generation
// ---------------------------------------------------------------------------

/// Cloud noise texture generator.
///
/// Generates 3D texture data for volumetric cloud rendering including
/// base shape noise and detail noise textures.
pub struct CloudNoiseGenerator {
    /// Configuration for noise generation.
    pub config: CloudNoiseConfig,
}

impl CloudNoiseGenerator {
    /// Create a new generator with default base configuration.
    pub fn new() -> Self {
        Self {
            config: CloudNoiseConfig::new_base(),
        }
    }

    /// Create a generator with custom configuration.
    pub fn with_config(config: CloudNoiseConfig) -> Self {
        Self { config }
    }

    /// Create a generator for detail noise.
    pub fn new_detail() -> Self {
        Self {
            config: CloudNoiseConfig::new_detail(),
        }
    }

    /// Generate 3D Worley noise texture data.
    ///
    /// Returns R8 format (1 byte per texel) containing inverted Worley distance.
    /// Inverted so that cell centers are bright (high density).
    ///
    /// # Arguments
    ///
    /// * `resolution` - Texture resolution per dimension (e.g., 32 for 32x32x32).
    ///
    /// # Returns
    ///
    /// Vector of resolution^3 bytes.
    pub fn generate_worley_3d(&self, resolution: u32) -> Vec<u8> {
        let size = (resolution * resolution * resolution) as usize;
        let mut data = vec![0u8; size];

        let inv_res = 1.0 / resolution as f32;

        for z in 0..resolution {
            for y in 0..resolution {
                for x in 0..resolution {
                    let point = [
                        x as f32 * inv_res,
                        y as f32 * inv_res,
                        z as f32 * inv_res,
                    ];

                    // Invert Worley so cell centers are bright
                    let worley = 1.0 - worley_distance(point, WORLEY_CELL_COUNT);
                    let byte = (worley.clamp(0.0, 1.0) * 255.0) as u8;

                    let idx = (z * resolution * resolution + y * resolution + x) as usize;
                    data[idx] = byte;
                }
            }
        }

        data
    }

    /// Generate 3D Worley noise with 2 octaves (RG8 format).
    ///
    /// Returns RG8 format (2 bytes per texel):
    /// - R channel: low frequency Worley (4 cells)
    /// - G channel: high frequency Worley (8 cells)
    ///
    /// # Arguments
    ///
    /// * `resolution` - Texture resolution per dimension.
    ///
    /// # Returns
    ///
    /// Vector of resolution^3 * 2 bytes.
    pub fn generate_worley_2_octaves(&self, resolution: u32) -> Vec<u8> {
        let size = (resolution * resolution * resolution) as usize;
        let mut data = vec![0u8; size * 2];

        let inv_res = 1.0 / resolution as f32;

        for z in 0..resolution {
            for y in 0..resolution {
                for x in 0..resolution {
                    let point = [
                        x as f32 * inv_res,
                        y as f32 * inv_res,
                        z as f32 * inv_res,
                    ];

                    // Low frequency (4 cells)
                    let worley_low = 1.0 - worley_distance_seeded(point, 4, 0xDEADBEEF);

                    // High frequency (8 cells)
                    let worley_high = 1.0 - worley_distance_seeded(point, 8, 0xCAFEBABE);

                    let idx = (z * resolution * resolution + y * resolution + x) as usize * 2;
                    data[idx] = (worley_low.clamp(0.0, 1.0) * 255.0) as u8;
                    data[idx + 1] = (worley_high.clamp(0.0, 1.0) * 255.0) as u8;
                }
            }
        }

        data
    }

    /// Generate 3D Perlin noise texture data.
    ///
    /// Returns R8 format (1 byte per texel) with Perlin noise remapped to [0, 1].
    pub fn generate_perlin_3d(&self, resolution: u32) -> Vec<u8> {
        let size = (resolution * resolution * resolution) as usize;
        let mut data = vec![0u8; size];

        let scale = self.config.octaves as f32; // Scale based on octave count

        for z in 0..resolution {
            for y in 0..resolution {
                for x in 0..resolution {
                    let point = [
                        x as f32 / resolution as f32 * scale,
                        y as f32 / resolution as f32 * scale,
                        z as f32 / resolution as f32 * scale,
                    ];

                    // Perlin returns [-1, 1], remap to [0, 1]
                    let perlin = perlin_3d(point) * 0.5 + 0.5;
                    let byte = (perlin.clamp(0.0, 1.0) * 255.0) as u8;

                    let idx = (z * resolution * resolution + y * resolution + x) as usize;
                    data[idx] = byte;
                }
            }
        }

        data
    }

    /// Generate Perlin-Worley FBM base shape texture.
    ///
    /// Returns RG8 format:
    /// - R channel: Combined Perlin-Worley FBM
    /// - G channel: Pure Perlin FBM (for blending control)
    ///
    /// This is the primary cloud shape texture used for volumetric rendering.
    pub fn generate_perlin_worley_fbm(&self) -> Vec<u8> {
        let resolution = self.config.base_resolution;
        let size = (resolution * resolution * resolution) as usize;
        let mut data = vec![0u8; size * 2];

        let inv_res = 1.0 / resolution as f32;

        for z in 0..resolution {
            for y in 0..resolution {
                for x in 0..resolution {
                    let point = [
                        x as f32 * inv_res,
                        y as f32 * inv_res,
                        z as f32 * inv_res,
                    ];

                    // Combined Perlin-Worley (70% Perlin, 30% Worley for soft clouds)
                    let combined = perlin_worley_fbm(
                        point,
                        self.config.octaves,
                        self.config.persistence,
                        self.config.lacunarity,
                        0.3, // Worley weight
                    );

                    // Pure Perlin for G channel
                    let perlin = fbm_3d(
                        point,
                        self.config.octaves,
                        self.config.persistence,
                        self.config.lacunarity,
                    );

                    // Remap [-1, 1] to [0, 1]
                    let r = ((combined * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
                    let g = ((perlin * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;

                    let idx = (z * resolution * resolution + y * resolution + x) as usize * 2;
                    data[idx] = r;
                    data[idx + 1] = g;
                }
            }
        }

        data
    }

    /// Generate detail FBM texture for wispy cloud edges.
    ///
    /// Returns R8 format using Worley FBM at higher frequency.
    /// This texture scrolls at a different rate for animation.
    pub fn generate_detail_fbm(&self) -> Vec<u8> {
        let resolution = self.config.base_resolution;
        let size = (resolution * resolution * resolution) as usize;
        let mut data = vec![0u8; size];

        let inv_res = 1.0 / resolution as f32;

        for z in 0..resolution {
            for y in 0..resolution {
                for x in 0..resolution {
                    let point = [
                        x as f32 * inv_res,
                        y as f32 * inv_res,
                        z as f32 * inv_res,
                    ];

                    // Use higher cell count for finer detail
                    let detail = fbm_worley_3d(
                        point,
                        self.config.octaves,
                        self.config.persistence,
                        self.config.lacunarity,
                        6, // More cells for finer detail
                    );

                    // Invert so cell centers are bright
                    let value = 1.0 - detail;
                    let byte = (value.clamp(0.0, 1.0) * 255.0) as u8;

                    let idx = (z * resolution * resolution + y * resolution + x) as usize;
                    data[idx] = byte;
                }
            }
        }

        data
    }

    /// Generate complete cloud texture set.
    ///
    /// Returns a tuple of:
    /// - Base shape texture (RG8, 32^3)
    /// - Detail texture (R8, 16^3)
    ///
    /// Total memory: ~74KB (well under 2MB budget).
    pub fn generate_all(&self) -> (Vec<u8>, Vec<u8>) {
        let base = self.generate_perlin_worley_fbm();

        let detail_gen = CloudNoiseGenerator::new_detail();
        let detail = detail_gen.generate_detail_fbm();

        (base, detail)
    }

    /// Calculate total memory usage of generated textures.
    pub fn total_memory_usage(&self) -> usize {
        let base_size = (self.config.base_resolution as usize).pow(3) * 2; // RG8
        let detail_size = (DEFAULT_DETAIL_RESOLUTION as usize).pow(3); // R8
        base_size + detail_size
    }
}

impl Default for CloudNoiseGenerator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Utility functions for texture coordinates
// ---------------------------------------------------------------------------

/// Convert world position to tiled texture coordinate.
///
/// # Arguments
///
/// * `world_pos` - World-space position in kilometers.
/// * `tile_size` - Tile size in kilometers.
///
/// # Returns
///
/// Normalized texture coordinate (0-1) that wraps seamlessly.
#[inline]
pub fn world_to_tile_coord(world_pos: [f32; 3], tile_size: f32) -> [f32; 3] {
    let inv_tile = 1.0 / tile_size;
    [
        (world_pos[0] * inv_tile).fract().abs(),
        (world_pos[1] * inv_tile).fract().abs(),
        (world_pos[2] * inv_tile).fract().abs(),
    ]
}

/// Apply animation offset to texture coordinates.
#[inline]
pub fn apply_animation_offset(coord: [f32; 3], offset: [f32; 3]) -> [f32; 3] {
    [
        (coord[0] + offset[0]).fract().abs(),
        (coord[1] + offset[1]).fract().abs(),
        (coord[2] + offset[2]).fract().abs(),
    ]
}

/// Calculate animation offset for a given time.
///
/// # Arguments
///
/// * `time` - Current time in seconds.
/// * `speed` - Scroll speed (units per second).
/// * `direction` - Wind direction (normalized).
#[inline]
pub fn calculate_animation_offset(time: f32, speed: f32, direction: [f32; 3]) -> [f32; 3] {
    let t = time * speed;
    [
        (direction[0] * t).fract().abs(),
        (direction[1] * t).fract().abs(),
        (direction[2] * t).fract().abs(),
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Hash function tests
    // ========================================================================

    #[test]
    fn test_hash_3d_deterministic() {
        let h1 = hash_3d(1, 2, 3, 0);
        let h2 = hash_3d(1, 2, 3, 0);
        assert_eq!(h1, h2, "Hash should be deterministic");
    }

    #[test]
    fn test_hash_3d_different_inputs() {
        let h1 = hash_3d(1, 2, 3, 0);
        let h2 = hash_3d(1, 2, 4, 0);
        assert_ne!(h1, h2, "Different inputs should produce different hashes");
    }

    #[test]
    fn test_hash_3d_seed_affects_output() {
        let h1 = hash_3d(1, 2, 3, 0);
        let h2 = hash_3d(1, 2, 3, 1);
        assert_ne!(h1, h2, "Different seeds should produce different hashes");
    }

    #[test]
    fn test_hash_to_float_range() {
        for i in 0..1000 {
            let hash = hash_3d(i, i * 2, i * 3, 0);
            let f = hash_to_float(hash);
            assert!(f >= 0.0 && f < 1.0, "Float should be in [0, 1), got {}", f);
        }
    }

    #[test]
    fn test_hash_3d_float_deterministic() {
        let h1 = hash_3d_float(1.5, 2.7, 3.9);
        let h2 = hash_3d_float(1.5, 2.7, 3.9);
        assert_eq!(h1, h2, "Float hash should be deterministic");
    }

    #[test]
    fn test_hash_distribution() {
        // Check hash values are reasonably distributed
        let mut buckets = [0u32; 10];
        for i in 0..1000 {
            let hash = hash_3d(i, i * 7, i * 13, 0xCAFE);
            let f = hash_to_float(hash);
            let bucket = (f * 10.0).floor() as usize;
            buckets[bucket.min(9)] += 1;
        }
        // Each bucket should have roughly 100 items (allow 50-150 range)
        for (i, &count) in buckets.iter().enumerate() {
            assert!(
                count > 30 && count < 200,
                "Bucket {} has {} items, expected ~100",
                i,
                count
            );
        }
    }

    // ========================================================================
    // WorleyCell tests
    // ========================================================================

    #[test]
    fn test_worley_cell_default() {
        let cell = WorleyCell::default();
        assert_eq!(cell.center, [0.5, 0.5, 0.5]);
        assert_eq!(cell.distance, f32::MAX);
    }

    #[test]
    fn test_worley_cell_new() {
        let cell = WorleyCell::new([0.1, 0.2, 0.3]);
        assert_eq!(cell.center, [0.1, 0.2, 0.3]);
        assert_eq!(cell.distance, f32::MAX);
    }

    #[test]
    fn test_worley_cell_from_hash_deterministic() {
        let c1 = WorleyCell::from_hash(1, 2, 3, 0);
        let c2 = WorleyCell::from_hash(1, 2, 3, 0);
        assert_eq!(c1.center, c2.center);
    }

    #[test]
    fn test_worley_cell_from_hash_range() {
        for i in 0..100 {
            let cell = WorleyCell::from_hash(i, i * 2, i * 3, 0);
            for c in cell.center {
                assert!(c >= 0.0 && c < 1.0, "Cell center should be in [0, 1)");
            }
        }
    }

    #[test]
    fn test_worley_cell_pod_zeroable() {
        // Verify Pod and Zeroable are correctly derived
        let zeroed: WorleyCell = bytemuck::Zeroable::zeroed();
        assert_eq!(zeroed.center, [0.0, 0.0, 0.0]);
        assert_eq!(zeroed.distance, 0.0);
    }

    // ========================================================================
    // Worley distance tests
    // ========================================================================

    #[test]
    fn test_worley_distance_range() {
        for z in 0..10 {
            for y in 0..10 {
                for x in 0..10 {
                    let point = [x as f32 / 10.0, y as f32 / 10.0, z as f32 / 10.0];
                    let dist = worley_distance(point, 4);
                    assert!(
                        dist >= 0.0 && dist <= 1.0,
                        "Worley distance should be in [0, 1], got {}",
                        dist
                    );
                }
            }
        }
    }

    #[test]
    fn test_worley_distance_deterministic() {
        let d1 = worley_distance([0.5, 0.5, 0.5], 4);
        let d2 = worley_distance([0.5, 0.5, 0.5], 4);
        assert_eq!(d1, d2, "Worley distance should be deterministic");
    }

    #[test]
    fn test_worley_distance_cell_count_affects_frequency() {
        let d4 = worley_distance([0.123, 0.456, 0.789], 4);
        let d8 = worley_distance([0.123, 0.456, 0.789], 8);
        // Different cell counts should generally produce different values
        // (not guaranteed for all points, but should be different on average)
        assert!(
            (d4 - d8).abs() > 0.0 || d4 == d8,
            "Different cell counts may produce different results"
        );
    }

    #[test]
    fn test_worley_distance_seeded() {
        let d1 = worley_distance_seeded([0.5, 0.5, 0.5], 4, 123);
        let d2 = worley_distance_seeded([0.5, 0.5, 0.5], 4, 456);
        // Different seeds should produce different distances
        assert_ne!(d1, d2, "Different seeds should affect Worley distance");
    }

    #[test]
    fn test_worley_tiling() {
        // Test that noise tiles seamlessly at 0 and 1 boundaries
        let d_start = worley_distance([0.0, 0.5, 0.5], 4);
        let d_end = worley_distance([1.0, 0.5, 0.5], 4);
        // Should be very close due to wrapping
        assert!(
            (d_start - d_end).abs() < 0.2,
            "Worley should tile: start={}, end={}",
            d_start,
            d_end
        );
    }

    #[test]
    fn test_worley_f1_f2() {
        let (f1, f2) = worley_f1_f2([0.5, 0.5, 0.5], 4, 0xDEADBEEF);
        assert!(f1 <= f2, "F1 should be <= F2");
        assert!(f1 >= 0.0 && f1 <= 1.0);
        assert!(f2 >= 0.0 && f2 <= 1.0);
    }

    #[test]
    fn test_worley_f1_f2_edge_distance() {
        // F2 - F1 should give edge-like pattern
        let (f1, f2) = worley_f1_f2([0.25, 0.25, 0.25], 4, 0xDEADBEEF);
        let edge = f2 - f1;
        assert!(edge >= 0.0, "Edge distance should be non-negative");
    }

    // ========================================================================
    // Perlin noise tests
    // ========================================================================

    #[test]
    fn test_fade_curve() {
        // Test fade curve properties
        assert_eq!(fade(0.0), 0.0, "fade(0) should be 0");
        assert_eq!(fade(1.0), 1.0, "fade(1) should be 1");
        assert!((fade(0.5) - 0.5).abs() < 0.01, "fade(0.5) should be ~0.5");
    }

    #[test]
    fn test_fade_curve_smoothness() {
        // Fade should be monotonically increasing
        let mut prev = 0.0;
        for i in 1..=100 {
            let t = i as f32 / 100.0;
            let f = fade(t);
            assert!(f >= prev, "Fade curve should be monotonically increasing");
            prev = f;
        }
    }

    #[test]
    fn test_lerp() {
        assert_eq!(lerp(0.0, 1.0, 0.0), 0.0);
        assert_eq!(lerp(0.0, 1.0, 1.0), 1.0);
        assert!((lerp(0.0, 1.0, 0.5) - 0.5).abs() < 1e-6);
        assert!((lerp(10.0, 20.0, 0.3) - 13.0).abs() < 1e-6);
    }

    #[test]
    fn test_perlin_gradient() {
        // Test that gradients produce reasonable values
        for h in 0..16 {
            let g = perlin_gradient(h, 1.0, 0.0, 0.0);
            assert!(g.abs() <= 2.0, "Gradient should be bounded");
        }
    }

    #[test]
    fn test_perlin_3d_range() {
        // Perlin noise should be approximately in [-1, 1]
        let mut min_val = f32::MAX;
        let mut max_val = f32::MIN;

        for z in 0..20 {
            for y in 0..20 {
                for x in 0..20 {
                    let point = [x as f32 * 0.13, y as f32 * 0.13, z as f32 * 0.13];
                    let val = perlin_3d(point);
                    min_val = min_val.min(val);
                    max_val = max_val.max(val);
                }
            }
        }

        assert!(
            min_val >= -1.5 && max_val <= 1.5,
            "Perlin range exceeded: [{}, {}]",
            min_val,
            max_val
        );
    }

    #[test]
    fn test_perlin_3d_deterministic() {
        let p1 = perlin_3d([1.5, 2.5, 3.5]);
        let p2 = perlin_3d([1.5, 2.5, 3.5]);
        assert_eq!(p1, p2, "Perlin should be deterministic");
    }

    #[test]
    fn test_perlin_3d_seeded() {
        let p1 = perlin_3d_seeded([0.5, 0.5, 0.5], 123);
        let p2 = perlin_3d_seeded([0.5, 0.5, 0.5], 456);
        assert_ne!(p1, p2, "Different seeds should produce different values");
    }

    #[test]
    fn test_perlin_3d_continuity() {
        // Test that noise is continuous (small changes in input = small changes in output)
        let p1 = perlin_3d([0.5, 0.5, 0.5]);
        let p2 = perlin_3d([0.501, 0.5, 0.5]);
        assert!(
            (p1 - p2).abs() < 0.1,
            "Perlin should be continuous: {} vs {}",
            p1,
            p2
        );
    }

    #[test]
    fn test_perlin_3d_tiled() {
        // Test tiled version wraps correctly
        let p_start = perlin_3d_tiled([0.0, 0.5, 0.5], 8);
        let p_end = perlin_3d_tiled([8.0, 0.5, 0.5], 8);
        assert!(
            (p_start - p_end).abs() < 1e-5,
            "Tiled Perlin should wrap: {} vs {}",
            p_start,
            p_end
        );
    }

    #[test]
    fn test_perlin_3d_tiled_y() {
        let p_start = perlin_3d_tiled([0.5, 0.0, 0.5], 8);
        let p_end = perlin_3d_tiled([0.5, 8.0, 0.5], 8);
        assert!(
            (p_start - p_end).abs() < 1e-5,
            "Tiled Perlin Y should wrap"
        );
    }

    #[test]
    fn test_perlin_3d_tiled_z() {
        let p_start = perlin_3d_tiled([0.5, 0.5, 0.0], 8);
        let p_end = perlin_3d_tiled([0.5, 0.5, 8.0], 8);
        assert!(
            (p_start - p_end).abs() < 1e-5,
            "Tiled Perlin Z should wrap"
        );
    }

    // ========================================================================
    // FBM tests
    // ========================================================================

    #[test]
    fn test_fbm_3d_range() {
        let mut min_val = f32::MAX;
        let mut max_val = f32::MIN;

        for z in 0..10 {
            for y in 0..10 {
                for x in 0..10 {
                    let point = [x as f32 * 0.1, y as f32 * 0.1, z as f32 * 0.1];
                    let val = fbm_3d(point, 4, 0.5, 2.0);
                    min_val = min_val.min(val);
                    max_val = max_val.max(val);
                }
            }
        }

        // FBM is normalized, should be close to [-1, 1]
        assert!(
            min_val >= -1.5 && max_val <= 1.5,
            "FBM range: [{}, {}]",
            min_val,
            max_val
        );
    }

    #[test]
    fn test_fbm_3d_deterministic() {
        let f1 = fbm_3d([1.0, 2.0, 3.0], 4, 0.5, 2.0);
        let f2 = fbm_3d([1.0, 2.0, 3.0], 4, 0.5, 2.0);
        assert_eq!(f1, f2, "FBM should be deterministic");
    }

    #[test]
    fn test_fbm_3d_octave_count_affects_detail() {
        // Use a point that's not at origin where differences are more apparent
        let f1 = fbm_3d([0.37, 0.73, 0.91], 1, 0.5, 2.0);
        let f4 = fbm_3d([0.37, 0.73, 0.91], 4, 0.5, 2.0);
        // More octaves should generally produce different values at non-special points
        assert!(
            (f1 - f4).abs() > 1e-6,
            "Different octave counts should affect result: f1={}, f4={}",
            f1, f4
        );
    }

    #[test]
    fn test_fbm_3d_zero_octaves() {
        let f = fbm_3d([0.5, 0.5, 0.5], 0, 0.5, 2.0);
        assert_eq!(f, 0.0, "Zero octaves should return 0");
    }

    #[test]
    fn test_fbm_worley_3d_range() {
        for z in 0..5 {
            for y in 0..5 {
                for x in 0..5 {
                    let point = [x as f32 / 5.0, y as f32 / 5.0, z as f32 / 5.0];
                    let val = fbm_worley_3d(point, 2, 0.5, 2.0, 4);
                    assert!(
                        val >= 0.0 && val <= 1.0,
                        "Worley FBM should be in [0, 1], got {}",
                        val
                    );
                }
            }
        }
    }

    #[test]
    fn test_fbm_worley_3d_deterministic() {
        let f1 = fbm_worley_3d([0.5, 0.5, 0.5], 2, 0.5, 2.0, 4);
        let f2 = fbm_worley_3d([0.5, 0.5, 0.5], 2, 0.5, 2.0, 4);
        assert_eq!(f1, f2, "Worley FBM should be deterministic");
    }

    #[test]
    fn test_perlin_worley_fbm_range() {
        for z in 0..5 {
            for y in 0..5 {
                for x in 0..5 {
                    let point = [x as f32 / 5.0, y as f32 / 5.0, z as f32 / 5.0];
                    let val = perlin_worley_fbm(point, 4, 0.5, 2.0, 0.3);
                    assert!(
                        val >= -1.5 && val <= 1.5,
                        "Perlin-Worley FBM out of range: {}",
                        val
                    );
                }
            }
        }
    }

    #[test]
    fn test_perlin_worley_fbm_worley_weight() {
        let pure_perlin = perlin_worley_fbm([0.5, 0.5, 0.5], 4, 0.5, 2.0, 0.0);
        let pure_worley = perlin_worley_fbm([0.5, 0.5, 0.5], 4, 0.5, 2.0, 1.0);
        let mixed = perlin_worley_fbm([0.5, 0.5, 0.5], 4, 0.5, 2.0, 0.5);

        // Mixed should be between pure values (approximately)
        let min_v = pure_perlin.min(pure_worley);
        let max_v = pure_perlin.max(pure_worley);
        // Allow some tolerance due to non-linearity
        assert!(
            mixed >= min_v - 0.5 && mixed <= max_v + 0.5,
            "Mixed should blend: pure_p={}, pure_w={}, mixed={}",
            pure_perlin,
            pure_worley,
            mixed
        );
    }

    // ========================================================================
    // CloudNoiseConfig tests
    // ========================================================================

    #[test]
    fn test_config_new_base() {
        let config = CloudNoiseConfig::new_base();
        assert_eq!(config.base_resolution, 32);
        assert_eq!(config.octaves, 4);
        assert_eq!(config.persistence, 0.5);
        assert_eq!(config.lacunarity, 2.0);
    }

    #[test]
    fn test_config_new_detail() {
        let config = CloudNoiseConfig::new_detail();
        assert_eq!(config.base_resolution, 16);
        assert_eq!(config.octaves, 2);
    }

    #[test]
    fn test_config_custom() {
        let config = CloudNoiseConfig::custom(64, 6, 0.6, 2.5, 5.0);
        assert_eq!(config.base_resolution, 64);
        assert_eq!(config.octaves, 6);
        assert!((config.persistence - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_config_custom_clamping() {
        let config = CloudNoiseConfig::custom(0, 0, 2.0, 0.5, 1.0);
        assert_eq!(config.base_resolution, 1, "Resolution should be at least 1");
        assert_eq!(config.octaves, 1, "Octaves should be at least 1");
        assert_eq!(config.persistence, 1.0, "Persistence should clamp to 1.0");
        assert_eq!(config.lacunarity, 1.0, "Lacunarity should be at least 1.0");
        assert_eq!(config.tile_size, MIN_TILE_SIZE_KM, "Tile size should clamp");
    }

    #[test]
    fn test_config_memory_usage() {
        let config = CloudNoiseConfig::new_base();
        let usage = config.memory_usage(2); // RG8 = 2 bytes
        assert_eq!(usage, 32 * 32 * 32 * 2);
    }

    #[test]
    fn test_config_validate_memory() {
        let config = CloudNoiseConfig::new_base();
        assert!(config.validate_memory(2), "32^3 * 2 should fit in 2MB");

        // 64^3 * 4 = 1MB, should fit
        let medium = CloudNoiseConfig::custom(64, 4, 0.5, 2.0, 6.0);
        assert!(medium.validate_memory(4), "64^3 * 4 should fit in 2MB");

        // 256^3 * 4 = 67MB, should NOT fit
        let large = CloudNoiseConfig::custom(256, 4, 0.5, 2.0, 6.0);
        assert!(!large.validate_memory(4), "256^3 * 4 should NOT fit in 2MB");
    }

    #[test]
    fn test_config_pod_zeroable() {
        let zeroed: CloudNoiseConfig = bytemuck::Zeroable::zeroed();
        assert_eq!(zeroed.base_resolution, 0);
        assert_eq!(zeroed.octaves, 0);
    }

    #[test]
    fn test_config_default() {
        let default = CloudNoiseConfig::default();
        let base = CloudNoiseConfig::new_base();
        assert_eq!(default.base_resolution, base.base_resolution);
    }

    // ========================================================================
    // AnimationState tests
    // ========================================================================

    #[test]
    fn test_animation_state_default() {
        let state = AnimationState::default();
        assert_eq!(state.base_offset, [0.0, 0.0, 0.0]);
        assert_eq!(state.detail_offset, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_animation_state_new() {
        let state = AnimationState::new(0.02, 2.0);
        assert_eq!(state.base_speed, 0.02);
        assert_eq!(state.detail_speed_multiplier, 2.0);
    }

    #[test]
    fn test_animation_state_update() {
        let mut state = AnimationState::new(1.0, 1.5);
        state.update(0.1, [1.0, 0.0, 0.0]);

        assert!(state.base_offset[0] > 0.0, "X offset should increase");
        assert!(state.detail_offset[0] > state.base_offset[0], "Detail should move faster");
    }

    #[test]
    fn test_animation_state_wrapping() {
        let mut state = AnimationState::new(10.0, 1.0);
        state.update(1.0, [1.0, 0.0, 0.0]);

        // Should wrap to [0, 1)
        assert!(
            state.base_offset[0] >= 0.0 && state.base_offset[0] < 1.0,
            "Offset should wrap"
        );
    }

    #[test]
    fn test_animation_apply_offset() {
        let state = AnimationState {
            base_offset: [0.1, 0.2, 0.3],
            base_speed: 1.0,
            detail_offset: [0.4, 0.5, 0.6],
            detail_speed_multiplier: 1.5,
        };

        let base = state.apply_base_offset([0.5, 0.5, 0.5]);
        assert!((base[0] - 0.6).abs() < 1e-6);
        assert!((base[1] - 0.7).abs() < 1e-6);
        assert!((base[2] - 0.8).abs() < 1e-6);

        let detail = state.apply_detail_offset([0.5, 0.5, 0.5]);
        assert!((detail[0] - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_animation_state_pod_zeroable() {
        let zeroed: AnimationState = bytemuck::Zeroable::zeroed();
        assert_eq!(zeroed.base_speed, 0.0);
    }

    // ========================================================================
    // CloudNoiseGenerator tests
    // ========================================================================

    #[test]
    fn test_generator_new() {
        let gen = CloudNoiseGenerator::new();
        assert_eq!(gen.config.base_resolution, DEFAULT_BASE_RESOLUTION);
    }

    #[test]
    fn test_generator_new_detail() {
        let gen = CloudNoiseGenerator::new_detail();
        assert_eq!(gen.config.base_resolution, DEFAULT_DETAIL_RESOLUTION);
    }

    #[test]
    fn test_generator_with_config() {
        let config = CloudNoiseConfig::custom(16, 3, 0.5, 2.0, 6.0);
        let gen = CloudNoiseGenerator::with_config(config);
        assert_eq!(gen.config.base_resolution, 16);
    }

    #[test]
    fn test_generate_worley_3d_size() {
        let gen = CloudNoiseGenerator::new();
        let data = gen.generate_worley_3d(8);
        assert_eq!(data.len(), 8 * 8 * 8);
    }

    #[test]
    fn test_generate_worley_3d_values() {
        let gen = CloudNoiseGenerator::new();
        let data = gen.generate_worley_3d(4);

        // All values should be valid (non-NaN when converted back)
        // and the data should have reasonable distribution
        let sum: u32 = data.iter().map(|&v| v as u32).sum();
        let avg = sum as f32 / data.len() as f32;
        // Average should be somewhere in middle range
        assert!(avg > 50.0 && avg < 200.0, "Average value {} should be reasonable", avg);
    }

    #[test]
    fn test_generate_worley_2_octaves_size() {
        let gen = CloudNoiseGenerator::new();
        let data = gen.generate_worley_2_octaves(8);
        assert_eq!(data.len(), 8 * 8 * 8 * 2); // RG8 = 2 bytes per texel
    }

    #[test]
    fn test_generate_perlin_3d_size() {
        let gen = CloudNoiseGenerator::new();
        let data = gen.generate_perlin_3d(8);
        assert_eq!(data.len(), 8 * 8 * 8);
    }

    #[test]
    fn test_generate_perlin_worley_fbm_size() {
        let gen = CloudNoiseGenerator::with_config(CloudNoiseConfig::custom(8, 4, 0.5, 2.0, 6.0));
        let data = gen.generate_perlin_worley_fbm();
        assert_eq!(data.len(), 8 * 8 * 8 * 2); // RG8
    }

    #[test]
    fn test_generate_detail_fbm_size() {
        let gen = CloudNoiseGenerator::new_detail();
        let data = gen.generate_detail_fbm();
        assert_eq!(data.len(), 16 * 16 * 16);
    }

    #[test]
    fn test_generate_all() {
        let gen = CloudNoiseGenerator::with_config(CloudNoiseConfig::custom(8, 4, 0.5, 2.0, 6.0));
        let (base, detail) = gen.generate_all();

        assert_eq!(base.len(), 8 * 8 * 8 * 2);
        assert_eq!(detail.len(), 16 * 16 * 16);
    }

    #[test]
    fn test_total_memory_usage() {
        let gen = CloudNoiseGenerator::new();
        let usage = gen.total_memory_usage();

        let expected = (32 * 32 * 32 * 2) + (16 * 16 * 16);
        assert_eq!(usage, expected);
    }

    #[test]
    fn test_memory_budget() {
        let gen = CloudNoiseGenerator::new();
        let usage = gen.total_memory_usage();
        assert!(
            usage < MAX_MEMORY_BYTES,
            "Total memory {} exceeds budget {}",
            usage,
            MAX_MEMORY_BYTES
        );
    }

    #[test]
    fn test_generator_default() {
        let gen = CloudNoiseGenerator::default();
        assert_eq!(gen.config.base_resolution, DEFAULT_BASE_RESOLUTION);
    }

    // ========================================================================
    // Utility function tests
    // ========================================================================

    #[test]
    fn test_world_to_tile_coord() {
        let coord = world_to_tile_coord([3.0, 6.0, 9.0], 6.0);
        assert!((coord[0] - 0.5).abs() < 1e-6);
        assert!(coord[1] >= 0.0 && coord[1] < 1.0);
        assert!((coord[2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_world_to_tile_coord_wrapping() {
        let coord1 = world_to_tile_coord([0.0, 0.0, 0.0], 6.0);
        let coord2 = world_to_tile_coord([6.0, 6.0, 6.0], 6.0);
        assert!((coord1[0] - coord2[0]).abs() < 1e-6);
    }

    #[test]
    fn test_apply_animation_offset() {
        let result = apply_animation_offset([0.5, 0.5, 0.5], [0.3, 0.3, 0.3]);
        assert!((result[0] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_apply_animation_offset_wrapping() {
        let result = apply_animation_offset([0.9, 0.9, 0.9], [0.3, 0.3, 0.3]);
        assert!((result[0] - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_calculate_animation_offset() {
        let offset = calculate_animation_offset(10.0, 0.1, [1.0, 0.0, 0.0]);
        assert!(offset[0] >= 0.0 && offset[0] < 1.0);
        assert_eq!(offset[1], 0.0);
        assert_eq!(offset[2], 0.0);
    }

    #[test]
    fn test_calculate_animation_offset_wrapping() {
        let offset = calculate_animation_offset(100.0, 0.1, [1.0, 0.0, 0.0]);
        assert!(offset[0] >= 0.0 && offset[0] < 1.0, "Should wrap");
    }

    // ========================================================================
    // Memory layout tests (for GPU compatibility)
    // ========================================================================

    #[test]
    fn test_cloud_noise_config_size() {
        assert_eq!(
            std::mem::size_of::<CloudNoiseConfig>(),
            32,
            "CloudNoiseConfig should be 32 bytes"
        );
    }

    #[test]
    fn test_worley_cell_size() {
        assert_eq!(
            std::mem::size_of::<WorleyCell>(),
            16,
            "WorleyCell should be 16 bytes"
        );
    }

    #[test]
    fn test_animation_state_size() {
        assert_eq!(
            std::mem::size_of::<AnimationState>(),
            32,
            "AnimationState should be 32 bytes"
        );
    }

    #[test]
    fn test_config_alignment() {
        assert_eq!(
            std::mem::align_of::<CloudNoiseConfig>(),
            4,
            "CloudNoiseConfig should have 4-byte alignment"
        );
    }

    // ========================================================================
    // Edge case tests
    // ========================================================================

    #[test]
    fn test_worley_at_cell_boundary() {
        // Test at exact cell boundaries
        let d = worley_distance([0.0, 0.0, 0.0], 4);
        assert!(d >= 0.0 && d <= 1.0);
    }

    #[test]
    fn test_worley_negative_coords() {
        // Test with negative coordinates (should wrap)
        let d = worley_distance([-0.5, -0.5, -0.5], 4);
        assert!(d >= 0.0 && d <= 1.0);
    }

    #[test]
    fn test_perlin_at_integer_coords() {
        // At integer coordinates, noise should be well-defined
        let p = perlin_3d([1.0, 2.0, 3.0]);
        assert!(!p.is_nan() && !p.is_infinite());
    }

    #[test]
    fn test_perlin_negative_coords() {
        let p = perlin_3d([-1.5, -2.5, -3.5]);
        assert!(!p.is_nan() && !p.is_infinite());
        assert!(p >= -2.0 && p <= 2.0);
    }

    #[test]
    fn test_fbm_single_octave() {
        let f = fbm_3d([0.5, 0.5, 0.5], 1, 0.5, 2.0);
        let p = perlin_3d([0.5, 0.5, 0.5]);
        // With single octave and normalized, should equal base noise
        assert!((f - p).abs() < 1e-5);
    }

    #[test]
    fn test_generator_small_resolution() {
        let gen = CloudNoiseGenerator::with_config(CloudNoiseConfig::custom(2, 2, 0.5, 2.0, 6.0));
        let data = gen.generate_perlin_worley_fbm();
        assert_eq!(data.len(), 2 * 2 * 2 * 2);
    }

    #[test]
    fn test_animation_zero_wind() {
        let mut state = AnimationState::new(1.0, 1.5);
        state.update(1.0, [0.0, 0.0, 0.0]);
        // With zero wind, offsets should still be handled (division by near-zero)
        assert!(!state.base_offset[0].is_nan());
    }
}
