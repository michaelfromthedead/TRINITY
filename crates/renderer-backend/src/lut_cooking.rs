//! LUT Cooking Pipeline for Atmospheric Scattering.
//!
//! This module implements the cooking (precomputation) and caching pipeline for
//! atmospheric scattering lookup tables used by the Bruneton 2017 sky rendering
//! system. The LUTs encode:
//!
//! - **Transmittance LUT**: Beer-Lambert extinction along view rays at various
//!   altitudes and zenith angles.
//! - **Sky-View LUT**: Single-scattering sky radiance for different view
//!   directions and sun positions.
//! - **Aerial Perspective LUT**: In-scattering and transmittance for distant
//!   objects (3D volume texture).
//!
//! # Architecture
//!
//! The pipeline follows a cook-and-cache pattern:
//!
//! 1. **Config**: [`LutCookingConfig`] specifies dimensions, precision, and
//!    caching preferences.
//! 2. **Cooking**: [`LutCookingPipeline`] generates raw LUT data from
//!    atmosphere parameters.
//! 3. **Caching**: [`LutCache`] persists cooked LUTs to disk for fast reload.
//! 4. **Upload**: GPU helper functions create textures from cooked data.
//!
//! # Example
//!
//! ```rust,no_run
//! use renderer_backend::lut_cooking::{
//!     LutCookingConfig, LutCookingPipeline, AtmosphereParams, LutCache,
//! };
//! use std::path::Path;
//!
//! // Configure the pipeline
//! let config = LutCookingConfig::default();
//! let mut pipeline = LutCookingPipeline::new(config);
//!
//! // Cook all LUTs
//! let params = AtmosphereParams::earth();
//! let lut_set = pipeline.cook_all(&params);
//!
//! // Cache to disk
//! let cache = LutCache::new(Path::new("./cache"));
//! cache.put(lut_set.transmittance.hash, &lut_set.transmittance);
//! ```
//!
//! # References
//!
//! - Bruneton, E., & Neyret, F. (2008). Precomputed atmospheric scattering.
//! - <https://ebruneton.github.io/precomputed_atmospheric_scattering/>

use std::collections::hash_map::DefaultHasher;
use std::fs::{self, File};
use std::hash::{Hash, Hasher};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use bytemuck::{Pod, Zeroable};

// ---------------------------------------------------------------------------
// LutPrecision
// ---------------------------------------------------------------------------

/// Precision mode for LUT texel storage.
///
/// Determines whether LUT data is stored as half-precision (f16) or
/// full-precision (f32) floating point values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum LutPrecision {
    /// Half-precision floating point (16-bit). Uses less memory and bandwidth
    /// but may have visible banding in dark regions.
    #[default]
    Half = 0,

    /// Full-precision floating point (32-bit). Higher quality but doubles
    /// memory usage.
    Full = 1,
}

impl LutPrecision {
    /// Returns the number of bytes per texel channel for this precision.
    ///
    /// - `Half`: 2 bytes per channel
    /// - `Full`: 4 bytes per channel
    #[inline]
    pub const fn bytes_per_channel(&self) -> usize {
        match self {
            LutPrecision::Half => 2,
            LutPrecision::Full => 4,
        }
    }

    /// Returns the number of bytes per RGBA texel.
    ///
    /// - `Half`: 8 bytes (RGBA16F)
    /// - `Full`: 16 bytes (RGBA32F)
    #[inline]
    pub const fn bytes_per_texel(&self) -> usize {
        self.bytes_per_channel() * 4
    }

    /// Returns the wgpu texture format for RGBA data at this precision.
    pub const fn rgba_format(&self) -> wgpu::TextureFormat {
        match self {
            LutPrecision::Half => wgpu::TextureFormat::Rgba16Float,
            LutPrecision::Full => wgpu::TextureFormat::Rgba32Float,
        }
    }

    /// Returns the wgpu texture format for RGB data at this precision.
    ///
    /// Note: wgpu does not have RGB16Float, so we use RGBA16Float and ignore
    /// the alpha channel.
    pub const fn rgb_format(&self) -> wgpu::TextureFormat {
        match self {
            LutPrecision::Half => wgpu::TextureFormat::Rgba16Float,
            LutPrecision::Full => wgpu::TextureFormat::Rgba32Float,
        }
    }
}

// ---------------------------------------------------------------------------
// LutType
// ---------------------------------------------------------------------------

/// Type of atmospheric LUT.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum LutType {
    /// Transmittance LUT: 2D texture storing extinction along view rays.
    Transmittance = 0,

    /// Sky-View LUT: 2D texture storing single-scattering sky radiance.
    SkyView = 1,

    /// Aerial Perspective LUT: 3D texture storing in-scattering and
    /// transmittance for distant objects.
    AerialPerspective = 2,
}

impl LutType {
    /// Returns a human-readable name for this LUT type.
    pub const fn name(&self) -> &'static str {
        match self {
            LutType::Transmittance => "transmittance",
            LutType::SkyView => "sky_view",
            LutType::AerialPerspective => "aerial_perspective",
        }
    }
}

// ---------------------------------------------------------------------------
// LutCookingConfig
// ---------------------------------------------------------------------------

/// Configuration for the LUT cooking pipeline.
///
/// Specifies dimensions for each LUT type, precision mode, and caching
/// behavior.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct LutCookingConfig {
    /// Dimensions of the transmittance LUT [width, height].
    ///
    /// - Width: number of zenith angle samples
    /// - Height: number of altitude samples
    ///
    /// Default: [256, 64]
    pub transmittance_size: [u32; 2],

    /// Dimensions of the sky-view LUT [width, height].
    ///
    /// - Width: number of view zenith angle samples
    /// - Height: number of view azimuth angle samples
    ///
    /// Default: [256, 512]
    pub sky_view_size: [u32; 2],

    /// Dimensions of the aerial perspective LUT [width, height, depth].
    ///
    /// - Width: view azimuth samples
    /// - Height: view zenith samples
    /// - Depth: distance samples
    ///
    /// Default: [32, 32, 32]
    pub aerial_perspective_size: [u32; 3],

    /// Precision for LUT storage.
    pub precision: LutPrecision,

    /// Whether to enable disk caching of cooked LUTs.
    pub cache_enabled: bool,
}

// Safety: LutCookingConfig is repr(C) and contains only POD types.
unsafe impl Pod for LutCookingConfig {}
unsafe impl Zeroable for LutCookingConfig {}

impl Default for LutCookingConfig {
    fn default() -> Self {
        Self {
            transmittance_size: [256, 64],
            sky_view_size: [256, 512],
            aerial_perspective_size: [32, 32, 32],
            precision: LutPrecision::Half,
            cache_enabled: true,
        }
    }
}

impl LutCookingConfig {
    /// Creates a new configuration with the specified settings.
    pub const fn new(
        transmittance_size: [u32; 2],
        sky_view_size: [u32; 2],
        aerial_perspective_size: [u32; 3],
        precision: LutPrecision,
        cache_enabled: bool,
    ) -> Self {
        Self {
            transmittance_size,
            sky_view_size,
            aerial_perspective_size,
            precision,
            cache_enabled,
        }
    }

    /// Creates a high-quality configuration with larger LUT dimensions.
    pub const fn high_quality() -> Self {
        Self {
            transmittance_size: [512, 128],
            sky_view_size: [512, 1024],
            aerial_perspective_size: [64, 64, 64],
            precision: LutPrecision::Full,
            cache_enabled: true,
        }
    }

    /// Creates a low-quality configuration for mobile/low-end devices.
    pub const fn low_quality() -> Self {
        Self {
            transmittance_size: [128, 32],
            sky_view_size: [128, 256],
            aerial_perspective_size: [16, 16, 16],
            precision: LutPrecision::Half,
            cache_enabled: true,
        }
    }

    /// Validates the configuration, returning an error message if invalid.
    pub fn validate(&self) -> Result<(), &'static str> {
        // Transmittance dimensions
        if self.transmittance_size[0] == 0 || self.transmittance_size[1] == 0 {
            return Err("transmittance_size dimensions must be non-zero");
        }
        if self.transmittance_size[0] > 4096 || self.transmittance_size[1] > 4096 {
            return Err("transmittance_size dimensions exceed maximum (4096)");
        }

        // Sky-view dimensions
        if self.sky_view_size[0] == 0 || self.sky_view_size[1] == 0 {
            return Err("sky_view_size dimensions must be non-zero");
        }
        if self.sky_view_size[0] > 4096 || self.sky_view_size[1] > 4096 {
            return Err("sky_view_size dimensions exceed maximum (4096)");
        }

        // Aerial perspective dimensions
        if self.aerial_perspective_size[0] == 0
            || self.aerial_perspective_size[1] == 0
            || self.aerial_perspective_size[2] == 0
        {
            return Err("aerial_perspective_size dimensions must be non-zero");
        }
        if self.aerial_perspective_size[0] > 256
            || self.aerial_perspective_size[1] > 256
            || self.aerial_perspective_size[2] > 256
        {
            return Err("aerial_perspective_size dimensions exceed maximum (256)");
        }

        Ok(())
    }

    /// Returns the total memory footprint in bytes for all LUTs.
    pub fn total_memory_bytes(&self) -> usize {
        let bpt = self.precision.bytes_per_texel();

        let transmittance = (self.transmittance_size[0] as usize)
            * (self.transmittance_size[1] as usize)
            * bpt;

        let sky_view =
            (self.sky_view_size[0] as usize) * (self.sky_view_size[1] as usize) * bpt;

        let aerial = (self.aerial_perspective_size[0] as usize)
            * (self.aerial_perspective_size[1] as usize)
            * (self.aerial_perspective_size[2] as usize)
            * bpt;

        transmittance + sky_view + aerial
    }
}

// ---------------------------------------------------------------------------
// AtmosphereParams
// ---------------------------------------------------------------------------

/// Physical parameters for atmospheric scattering.
///
/// These parameters define the optical properties of the atmosphere and are
/// used during LUT cooking to compute transmittance and scattering.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtmosphereParams {
    /// Planet radius in meters (Earth: 6,371,000 m).
    pub planet_radius: f32,

    /// Height of the atmosphere above the planet surface in meters.
    pub atmosphere_height: f32,

    /// Scale height for Rayleigh scattering in meters (Earth: ~8,000 m).
    pub rayleigh_scale_height: f32,

    /// Scale height for Mie scattering in meters (Earth: ~1,200 m).
    pub mie_scale_height: f32,

    /// Rayleigh scattering coefficients at sea level [R, G, B] in m^-1.
    pub rayleigh_scattering: [f32; 3],

    /// Mie scattering coefficient at sea level in m^-1.
    pub mie_scattering: f32,

    /// Mie absorption coefficient at sea level in m^-1.
    pub mie_absorption: f32,

    /// Henyey-Greenstein asymmetry parameter for Mie scattering.
    /// Positive values indicate forward scattering.
    pub mie_asymmetry_g: f32,

    /// Ozone absorption coefficients [R, G, B] in m^-1.
    pub ozone_absorption: [f32; 3],

    /// Angular radius of the sun in radians.
    pub sun_angular_radius: f32,
}

impl Default for AtmosphereParams {
    fn default() -> Self {
        Self::earth()
    }
}

impl AtmosphereParams {
    /// Returns Earth's standard atmosphere parameters.
    pub const fn earth() -> Self {
        Self {
            planet_radius: 6_371_000.0,
            atmosphere_height: 80_000.0,
            rayleigh_scale_height: 8_000.0,
            mie_scale_height: 1_200.0,
            rayleigh_scattering: [5.5e-6, 13.0e-6, 22.4e-6],
            mie_scattering: 21e-6,
            mie_absorption: 4.4e-6,
            mie_asymmetry_g: 0.8,
            ozone_absorption: [0.65e-6, 1.88e-6, 0.085e-6],
            sun_angular_radius: 0.00467,
        }
    }

    /// Returns Mars-like atmosphere parameters.
    pub const fn mars() -> Self {
        Self {
            planet_radius: 3_389_500.0,
            atmosphere_height: 100_000.0,
            rayleigh_scale_height: 11_100.0,
            mie_scale_height: 1_000.0,
            rayleigh_scattering: [19.9e-6, 13.6e-6, 5.6e-6], // Red-shifted
            mie_scattering: 50e-6,                           // Dusty
            mie_absorption: 15e-6,
            mie_asymmetry_g: 0.76,
            ozone_absorption: [0.0, 0.0, 0.0], // No ozone
            sun_angular_radius: 0.00295,       // Sun appears smaller
        }
    }

    /// Returns the radius at the top of the atmosphere.
    #[inline]
    pub fn atmosphere_top_radius(&self) -> f32 {
        self.planet_radius + self.atmosphere_height
    }

    /// Returns the total Mie extinction coefficient.
    #[inline]
    pub fn mie_extinction(&self) -> f32 {
        self.mie_scattering + self.mie_absorption
    }

    /// Computes a deterministic hash of these parameters.
    pub fn compute_hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();

        // Convert floats to bits for deterministic hashing
        self.planet_radius.to_bits().hash(&mut hasher);
        self.atmosphere_height.to_bits().hash(&mut hasher);
        self.rayleigh_scale_height.to_bits().hash(&mut hasher);
        self.mie_scale_height.to_bits().hash(&mut hasher);

        for &v in &self.rayleigh_scattering {
            v.to_bits().hash(&mut hasher);
        }

        self.mie_scattering.to_bits().hash(&mut hasher);
        self.mie_absorption.to_bits().hash(&mut hasher);
        self.mie_asymmetry_g.to_bits().hash(&mut hasher);

        for &v in &self.ozone_absorption {
            v.to_bits().hash(&mut hasher);
        }

        self.sun_angular_radius.to_bits().hash(&mut hasher);

        hasher.finish()
    }
}

// ---------------------------------------------------------------------------
// FrustumParams
// ---------------------------------------------------------------------------

/// View frustum parameters for aerial perspective cooking.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrustumParams {
    /// Near plane distance in meters.
    pub near: f32,

    /// Far plane distance in meters.
    pub far: f32,

    /// Horizontal field of view in radians.
    pub fov_horizontal: f32,

    /// Vertical field of view in radians.
    pub fov_vertical: f32,
}

impl Default for FrustumParams {
    fn default() -> Self {
        Self {
            near: 0.1,
            far: 100_000.0,
            fov_horizontal: std::f32::consts::FRAC_PI_2, // 90 degrees
            fov_vertical: std::f32::consts::FRAC_PI_3,   // 60 degrees
        }
    }
}

impl FrustumParams {
    /// Computes a deterministic hash of these parameters.
    pub fn compute_hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.near.to_bits().hash(&mut hasher);
        self.far.to_bits().hash(&mut hasher);
        self.fov_horizontal.to_bits().hash(&mut hasher);
        self.fov_vertical.to_bits().hash(&mut hasher);
        hasher.finish()
    }
}

// ---------------------------------------------------------------------------
// CookedLut
// ---------------------------------------------------------------------------

/// A cooked (precomputed) lookup table ready for GPU upload.
#[derive(Debug, Clone)]
pub struct CookedLut {
    /// Type of this LUT.
    pub lut_type: LutType,

    /// Dimensions of the LUT [width, height, depth].
    /// For 2D LUTs, depth is 1.
    pub dimensions: [u32; 3],

    /// Raw texture data as bytes.
    pub data: Vec<u8>,

    /// Texture format for GPU upload.
    pub format: wgpu::TextureFormat,

    /// Content hash for cache lookup.
    pub hash: u64,
}

impl CookedLut {
    /// Creates a new cooked LUT.
    pub fn new(
        lut_type: LutType,
        dimensions: [u32; 3],
        data: Vec<u8>,
        format: wgpu::TextureFormat,
        hash: u64,
    ) -> Self {
        Self {
            lut_type,
            dimensions,
            data,
            format,
            hash,
        }
    }

    /// Returns the total size of the LUT data in bytes.
    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }

    /// Returns the number of texels in this LUT.
    #[inline]
    pub fn texel_count(&self) -> usize {
        (self.dimensions[0] as usize)
            * (self.dimensions[1] as usize)
            * (self.dimensions[2] as usize)
    }

    /// Returns whether this LUT is a 3D volume texture.
    #[inline]
    pub fn is_3d(&self) -> bool {
        self.dimensions[2] > 1
    }

    /// Returns the wgpu texture dimension for this LUT.
    pub fn texture_dimension(&self) -> wgpu::TextureDimension {
        if self.is_3d() {
            wgpu::TextureDimension::D3
        } else {
            wgpu::TextureDimension::D2
        }
    }
}

// ---------------------------------------------------------------------------
// LutSet
// ---------------------------------------------------------------------------

/// A complete set of atmospheric LUTs.
#[derive(Debug, Clone)]
pub struct LutSet {
    /// Transmittance LUT.
    pub transmittance: CookedLut,

    /// Sky-view LUT.
    pub sky_view: CookedLut,

    /// Aerial perspective LUT.
    pub aerial_perspective: CookedLut,
}

impl LutSet {
    /// Creates a new LUT set.
    pub fn new(transmittance: CookedLut, sky_view: CookedLut, aerial_perspective: CookedLut) -> Self {
        Self {
            transmittance,
            sky_view,
            aerial_perspective,
        }
    }

    /// Returns the total memory footprint of all LUTs in bytes.
    pub fn total_bytes(&self) -> usize {
        self.transmittance.size_bytes()
            + self.sky_view.size_bytes()
            + self.aerial_perspective.size_bytes()
    }

    /// Returns the combined hash of all LUTs.
    pub fn combined_hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.transmittance.hash.hash(&mut hasher);
        self.sky_view.hash.hash(&mut hasher);
        self.aerial_perspective.hash.hash(&mut hasher);
        hasher.finish()
    }
}

// ---------------------------------------------------------------------------
// LutCookingPipeline
// ---------------------------------------------------------------------------

/// Pipeline for cooking atmospheric scattering LUTs.
///
/// The pipeline generates lookup tables from atmosphere parameters, with
/// support for different precision modes and dimensions.
pub struct LutCookingPipeline {
    config: LutCookingConfig,
}

impl LutCookingPipeline {
    /// Creates a new LUT cooking pipeline with the given configuration.
    pub fn new(config: LutCookingConfig) -> Self {
        Self { config }
    }

    /// Returns a reference to the pipeline configuration.
    pub fn config(&self) -> &LutCookingConfig {
        &self.config
    }

    /// Cooks the transmittance LUT.
    ///
    /// The transmittance LUT stores the fraction of light that reaches a point
    /// at a given altitude when looking in a direction with a given zenith
    /// cosine.
    pub fn cook_transmittance(&self, params: &AtmosphereParams) -> CookedLut {
        let width = self.config.transmittance_size[0] as usize;
        let height = self.config.transmittance_size[1] as usize;
        let precision = self.config.precision;
        let bytes_per_texel = precision.bytes_per_texel();

        let mut data = vec![0u8; width * height * bytes_per_texel];

        for y in 0..height {
            for x in 0..width {
                // Map UV to altitude and zenith cosine
                let u = (x as f32 + 0.5) / width as f32;
                let v = (y as f32 + 0.5) / height as f32;

                let (altitude, zenith_cos) = self.map_transmittance_uv(u, v, params);

                // Compute transmittance along the ray
                let transmittance = self.compute_transmittance(altitude, zenith_cos, params);

                // Write texel
                let offset = (y * width + x) * bytes_per_texel;
                self.write_rgba_texel(&mut data[offset..], transmittance, 1.0, precision);
            }
        }

        let hash = self.compute_hash(params, LutType::Transmittance, 0.0, None);

        CookedLut::new(
            LutType::Transmittance,
            [self.config.transmittance_size[0], self.config.transmittance_size[1], 1],
            data,
            precision.rgba_format(),
            hash,
        )
    }

    /// Cooks the sky-view LUT for a specific sun elevation.
    ///
    /// The sky-view LUT stores single-scattering sky radiance for all view
    /// directions at ground level.
    pub fn cook_sky_view(&self, params: &AtmosphereParams, sun_elevation: f32) -> CookedLut {
        let width = self.config.sky_view_size[0] as usize;
        let height = self.config.sky_view_size[1] as usize;
        let precision = self.config.precision;
        let bytes_per_texel = precision.bytes_per_texel();

        let mut data = vec![0u8; width * height * bytes_per_texel];

        // Sun direction from elevation angle
        let sun_cos = sun_elevation.cos();
        let sun_sin = sun_elevation.sin();
        let sun_dir = [0.0, sun_sin, sun_cos];

        for y in 0..height {
            for x in 0..width {
                // Map UV to view direction
                let u = (x as f32 + 0.5) / width as f32;
                let v = (y as f32 + 0.5) / height as f32;

                let view_dir = self.map_sky_view_uv(u, v);

                // Compute single scattering
                let radiance = self.compute_sky_radiance(params, &view_dir, &sun_dir);

                // Write texel
                let offset = (y * width + x) * bytes_per_texel;
                self.write_rgba_texel(&mut data[offset..], radiance, 1.0, precision);
            }
        }

        let hash = self.compute_hash(params, LutType::SkyView, sun_elevation, None);

        CookedLut::new(
            LutType::SkyView,
            [self.config.sky_view_size[0], self.config.sky_view_size[1], 1],
            data,
            precision.rgba_format(),
            hash,
        )
    }

    /// Cooks the aerial perspective LUT.
    ///
    /// The aerial perspective LUT is a 3D texture storing in-scattering and
    /// transmittance for rendering distant objects with atmospheric haze.
    pub fn cook_aerial_perspective(
        &self,
        params: &AtmosphereParams,
        frustum: &FrustumParams,
    ) -> CookedLut {
        let width = self.config.aerial_perspective_size[0] as usize;
        let height = self.config.aerial_perspective_size[1] as usize;
        let depth = self.config.aerial_perspective_size[2] as usize;
        let precision = self.config.precision;
        let bytes_per_texel = precision.bytes_per_texel();

        let mut data = vec![0u8; width * height * depth * bytes_per_texel];

        // Default sun direction (can be parameterized)
        let sun_dir = [0.5_f32.sqrt(), 0.5_f32.sqrt(), 0.0];

        for z in 0..depth {
            for y in 0..height {
                for x in 0..width {
                    // Map UVW to view direction and distance
                    let u = (x as f32 + 0.5) / width as f32;
                    let v = (y as f32 + 0.5) / height as f32;
                    let w = (z as f32 + 0.5) / depth as f32;

                    let (view_dir, distance) =
                        self.map_aerial_perspective_uvw(u, v, w, frustum);

                    // Compute aerial perspective
                    let (inscatter, transmittance) =
                        self.compute_aerial_perspective(params, &view_dir, &sun_dir, distance);

                    // Blend inscatter and transmittance into RGBA
                    // RGB = inscatter, A = average transmittance
                    let avg_trans = (transmittance[0] + transmittance[1] + transmittance[2]) / 3.0;

                    // Write texel
                    let offset = ((z * height + y) * width + x) * bytes_per_texel;
                    self.write_rgba_texel(&mut data[offset..], inscatter, avg_trans, precision);
                }
            }
        }

        let hash = self.compute_hash(params, LutType::AerialPerspective, 0.0, Some(frustum));

        CookedLut::new(
            LutType::AerialPerspective,
            [
                self.config.aerial_perspective_size[0],
                self.config.aerial_perspective_size[1],
                self.config.aerial_perspective_size[2],
            ],
            data,
            precision.rgba_format(),
            hash,
        )
    }

    /// Cooks all atmospheric LUTs with default sun elevation.
    pub fn cook_all(&self, params: &AtmosphereParams) -> LutSet {
        self.cook_all_with_sun(params, std::f32::consts::FRAC_PI_4) // 45 degrees
    }

    /// Cooks all atmospheric LUTs with a specific sun elevation.
    pub fn cook_all_with_sun(&self, params: &AtmosphereParams, sun_elevation: f32) -> LutSet {
        let transmittance = self.cook_transmittance(params);
        let sky_view = self.cook_sky_view(params, sun_elevation);
        let aerial_perspective = self.cook_aerial_perspective(params, &FrustumParams::default());

        LutSet::new(transmittance, sky_view, aerial_perspective)
    }

    /// Validates a cooked LUT for correctness.
    ///
    /// Checks for NaN and infinity values which would cause rendering artifacts.
    pub fn validate_lut(&self, lut: &CookedLut) -> bool {
        let precision = self.config.precision;

        match precision {
            LutPrecision::Half => {
                // Check f16 values
                for chunk in lut.data.chunks_exact(2) {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    let value = half::f16::from_bits(bits).to_f32();
                    if value.is_nan() || value.is_infinite() {
                        return false;
                    }
                }
            }
            LutPrecision::Full => {
                // Check f32 values
                for chunk in lut.data.chunks_exact(4) {
                    let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    if value.is_nan() || value.is_infinite() {
                        return false;
                    }
                }
            }
        }

        true
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Maps UV coordinates to altitude and zenith cosine for transmittance LUT.
    fn map_transmittance_uv(&self, u: f32, v: f32, params: &AtmosphereParams) -> (f32, f32) {
        // Non-linear mapping to concentrate samples near ground and horizon
        let altitude = v * v * params.atmosphere_height;

        // Map u to zenith cosine with quadratic mapping for horizon detail
        let x = 2.0 * u - 1.0;
        let zenith_cos = x.abs().sqrt() * x.signum();

        (altitude, zenith_cos)
    }

    /// Maps UV coordinates to view direction for sky-view LUT.
    fn map_sky_view_uv(&self, u: f32, v: f32) -> [f32; 3] {
        // u -> view zenith cosine, v -> view azimuth
        let zenith_cos = 2.0 * u - 1.0;
        let zenith_cos = zenith_cos * zenith_cos.abs(); // Quadratic mapping
        let zenith_sin = (1.0 - zenith_cos * zenith_cos).max(0.0).sqrt();

        let azimuth = v * 2.0 * std::f32::consts::PI;

        [
            zenith_sin * azimuth.cos(),
            zenith_cos,
            zenith_sin * azimuth.sin(),
        ]
    }

    /// Maps UVW coordinates to view direction and distance for aerial perspective.
    fn map_aerial_perspective_uvw(
        &self,
        u: f32,
        v: f32,
        w: f32,
        frustum: &FrustumParams,
    ) -> ([f32; 3], f32) {
        // u -> azimuth, v -> zenith, w -> distance
        let azimuth = u * 2.0 * std::f32::consts::PI;
        let zenith_cos = 2.0 * v - 1.0;
        let zenith_sin = (1.0 - zenith_cos * zenith_cos).max(0.0).sqrt();

        let view_dir = [
            zenith_sin * azimuth.cos(),
            zenith_cos,
            zenith_sin * azimuth.sin(),
        ];

        // Non-linear distance mapping
        let distance = frustum.near + w * w * (frustum.far - frustum.near);

        (view_dir, distance)
    }

    /// Computes transmittance along a ray.
    fn compute_transmittance(
        &self,
        altitude: f32,
        zenith_cos: f32,
        params: &AtmosphereParams,
    ) -> [f32; 3] {
        const STEPS: usize = 64;

        let r = params.planet_radius + altitude;
        let position = [0.0, r, 0.0];

        let zenith_sin = (1.0 - zenith_cos * zenith_cos).max(0.0).sqrt();
        let view_dir = [zenith_sin, zenith_cos, 0.0];

        // Find ray length to atmosphere top
        let t_max = self.ray_atmosphere_intersection(&position, &view_dir, params);
        if t_max <= 0.0 {
            return [1.0, 1.0, 1.0];
        }

        let dt = t_max / STEPS as f32;
        let mut optical_depth = [0.0_f32; 3];

        for i in 0..STEPS {
            let t = (i as f32 + 0.5) * dt;
            let sample_pos = [
                position[0] + t * view_dir[0],
                position[1] + t * view_dir[1],
                position[2] + t * view_dir[2],
            ];

            let sample_r = (sample_pos[0] * sample_pos[0]
                + sample_pos[1] * sample_pos[1]
                + sample_pos[2] * sample_pos[2])
            .sqrt();
            let sample_alt = sample_r - params.planet_radius;

            if sample_alt < 0.0 || sample_alt > params.atmosphere_height {
                continue;
            }

            // Rayleigh density
            let rayleigh_density = (-sample_alt / params.rayleigh_scale_height).exp();

            // Mie density
            let mie_density = (-sample_alt / params.mie_scale_height).exp();

            // Ozone density (peak at 25km)
            let ozone_t = (sample_alt - 25_000.0) / 15_000.0;
            let ozone_density = (1.0 - ozone_t * ozone_t).max(0.0);

            // Accumulate optical depth
            for c in 0..3 {
                optical_depth[c] += (params.rayleigh_scattering[c] * rayleigh_density
                    + params.mie_extinction() * mie_density
                    + params.ozone_absorption[c] * ozone_density)
                    * dt;
            }
        }

        // Beer-Lambert law
        [
            (-optical_depth[0]).exp(),
            (-optical_depth[1]).exp(),
            (-optical_depth[2]).exp(),
        ]
    }

    /// Computes single-scattering sky radiance.
    fn compute_sky_radiance(
        &self,
        params: &AtmosphereParams,
        view_dir: &[f32; 3],
        sun_dir: &[f32; 3],
    ) -> [f32; 3] {
        const STEPS: usize = 16;

        let observer_r = params.planet_radius;
        let position = [0.0, observer_r, 0.0];

        // Find ray length
        let t_max = self.ray_atmosphere_intersection(&position, view_dir, params);
        if t_max <= 0.0 {
            return [0.0, 0.0, 0.0];
        }

        // Cosine of angle between view and sun
        let cos_theta = view_dir[0] * sun_dir[0] + view_dir[1] * sun_dir[1] + view_dir[2] * sun_dir[2];

        // Phase functions
        let rayleigh_phase = 3.0 / (16.0 * std::f32::consts::PI) * (1.0 + cos_theta * cos_theta);
        let mie_phase = self.cornette_shanks_phase(cos_theta, params.mie_asymmetry_g);

        let dt = t_max / STEPS as f32;
        let mut radiance = [0.0_f32; 3];
        let mut transmittance = [1.0_f32; 3];

        for i in 0..STEPS {
            let t = (i as f32 + 0.5) * dt;
            let sample_pos = [
                position[0] + t * view_dir[0],
                position[1] + t * view_dir[1],
                position[2] + t * view_dir[2],
            ];

            let sample_r = (sample_pos[0] * sample_pos[0]
                + sample_pos[1] * sample_pos[1]
                + sample_pos[2] * sample_pos[2])
            .sqrt();
            let sample_alt = sample_r - params.planet_radius;

            if sample_alt < 0.0 || sample_alt > params.atmosphere_height {
                continue;
            }

            // Densities
            let rayleigh_density = (-sample_alt / params.rayleigh_scale_height).exp();
            let mie_density = (-sample_alt / params.mie_scale_height).exp();

            // Scattering coefficients
            let rayleigh_scatter = [
                params.rayleigh_scattering[0] * rayleigh_density,
                params.rayleigh_scattering[1] * rayleigh_density,
                params.rayleigh_scattering[2] * rayleigh_density,
            ];
            let mie_scatter = params.mie_scattering * mie_density;

            // Sun transmittance (simplified)
            let sun_zenith_cos = (sample_pos[0] * sun_dir[0]
                + sample_pos[1] * sun_dir[1]
                + sample_pos[2] * sun_dir[2])
                / sample_r;
            let sun_trans = self.compute_transmittance(sample_alt, sun_zenith_cos, params);

            // In-scattering contribution
            for c in 0..3 {
                let inscatter = (rayleigh_scatter[c] * rayleigh_phase + mie_scatter * mie_phase)
                    * sun_trans[c];
                radiance[c] += transmittance[c] * inscatter * dt;
            }

            // Update transmittance along view ray
            for c in 0..3 {
                let extinction = rayleigh_scatter[c] + mie_density * params.mie_extinction();
                transmittance[c] *= (-extinction * dt).exp();
            }
        }

        radiance
    }

    /// Computes aerial perspective (in-scattering and transmittance to distance).
    fn compute_aerial_perspective(
        &self,
        params: &AtmosphereParams,
        view_dir: &[f32; 3],
        sun_dir: &[f32; 3],
        distance: f32,
    ) -> ([f32; 3], [f32; 3]) {
        const STEPS: usize = 8;

        let observer_r = params.planet_radius;
        let position = [0.0, observer_r, 0.0];

        let t_max = distance.min(params.atmosphere_height);

        // Cosine of angle between view and sun
        let cos_theta = view_dir[0] * sun_dir[0] + view_dir[1] * sun_dir[1] + view_dir[2] * sun_dir[2];

        // Phase functions
        let rayleigh_phase = 3.0 / (16.0 * std::f32::consts::PI) * (1.0 + cos_theta * cos_theta);
        let mie_phase = self.cornette_shanks_phase(cos_theta, params.mie_asymmetry_g);

        let dt = t_max / STEPS as f32;
        let mut inscatter = [0.0_f32; 3];
        let mut transmittance = [1.0_f32; 3];

        for i in 0..STEPS {
            let t = (i as f32 + 0.5) * dt;
            let sample_pos = [
                position[0] + t * view_dir[0],
                position[1] + t * view_dir[1],
                position[2] + t * view_dir[2],
            ];

            let sample_r = (sample_pos[0] * sample_pos[0]
                + sample_pos[1] * sample_pos[1]
                + sample_pos[2] * sample_pos[2])
            .sqrt();
            let sample_alt = sample_r - params.planet_radius;

            if sample_alt < 0.0 {
                break;
            }
            if sample_alt > params.atmosphere_height {
                continue;
            }

            // Densities
            let rayleigh_density = (-sample_alt / params.rayleigh_scale_height).exp();
            let mie_density = (-sample_alt / params.mie_scale_height).exp();

            // Scattering
            let rayleigh_scatter = [
                params.rayleigh_scattering[0] * rayleigh_density,
                params.rayleigh_scattering[1] * rayleigh_density,
                params.rayleigh_scattering[2] * rayleigh_density,
            ];
            let mie_scatter = params.mie_scattering * mie_density;

            // Sun transmittance
            let sun_zenith_cos = (sample_pos[0] * sun_dir[0]
                + sample_pos[1] * sun_dir[1]
                + sample_pos[2] * sun_dir[2])
                / sample_r;
            let sun_trans = self.compute_transmittance(sample_alt, sun_zenith_cos, params);

            // In-scattering
            for c in 0..3 {
                let scatter = (rayleigh_scatter[c] * rayleigh_phase + mie_scatter * mie_phase)
                    * sun_trans[c];
                inscatter[c] += transmittance[c] * scatter * dt;
            }

            // Update transmittance
            for c in 0..3 {
                let extinction = rayleigh_scatter[c] + mie_density * params.mie_extinction();
                transmittance[c] *= (-extinction * dt).exp();
            }
        }

        (inscatter, transmittance)
    }

    /// Cornette-Shanks phase function for Mie scattering.
    fn cornette_shanks_phase(&self, cos_theta: f32, g: f32) -> f32 {
        if g.abs() < 1e-6 {
            // Rayleigh limit
            return 3.0 / (16.0 * std::f32::consts::PI) * (1.0 + cos_theta * cos_theta);
        }

        let g2 = g * g;
        let k = 3.0 / (8.0 * std::f32::consts::PI) * (1.0 - g2) / (2.0 + g2);
        let denom = 1.0 + g2 - 2.0 * g * cos_theta;

        if denom < 1e-6 {
            return 0.0;
        }

        k * (1.0 + cos_theta * cos_theta) / denom.powf(1.5)
    }

    /// Computes ray-atmosphere intersection distance.
    fn ray_atmosphere_intersection(
        &self,
        origin: &[f32; 3],
        dir: &[f32; 3],
        params: &AtmosphereParams,
    ) -> f32 {
        let top_radius = params.atmosphere_top_radius();

        // Ray-sphere intersection with atmosphere top
        let oc = [origin[0], origin[1], origin[2]];
        let a = dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2];
        let b = 2.0 * (oc[0] * dir[0] + oc[1] * dir[1] + oc[2] * dir[2]);
        let c = oc[0] * oc[0] + oc[1] * oc[1] + oc[2] * oc[2] - top_radius * top_radius;

        let discriminant = b * b - 4.0 * a * c;

        if discriminant < 0.0 {
            return -1.0;
        }

        let sqrt_disc = discriminant.sqrt();
        let t1 = (-b - sqrt_disc) / (2.0 * a);
        let t2 = (-b + sqrt_disc) / (2.0 * a);

        // Return far intersection (we're inside the atmosphere)
        if t2 > 0.0 {
            t2
        } else if t1 > 0.0 {
            t1
        } else {
            -1.0
        }
    }

    /// Writes an RGBA texel to the data buffer.
    fn write_rgba_texel(&self, dest: &mut [u8], rgb: [f32; 3], a: f32, precision: LutPrecision) {
        match precision {
            LutPrecision::Half => {
                let r = half::f16::from_f32(rgb[0]).to_bits().to_le_bytes();
                let g = half::f16::from_f32(rgb[1]).to_bits().to_le_bytes();
                let b = half::f16::from_f32(rgb[2]).to_bits().to_le_bytes();
                let a = half::f16::from_f32(a).to_bits().to_le_bytes();

                dest[0..2].copy_from_slice(&r);
                dest[2..4].copy_from_slice(&g);
                dest[4..6].copy_from_slice(&b);
                dest[6..8].copy_from_slice(&a);
            }
            LutPrecision::Full => {
                dest[0..4].copy_from_slice(&rgb[0].to_le_bytes());
                dest[4..8].copy_from_slice(&rgb[1].to_le_bytes());
                dest[8..12].copy_from_slice(&rgb[2].to_le_bytes());
                dest[12..16].copy_from_slice(&a.to_le_bytes());
            }
        }
    }

    /// Computes a deterministic hash for cache lookup.
    fn compute_hash(
        &self,
        params: &AtmosphereParams,
        lut_type: LutType,
        sun_elevation: f32,
        frustum: Option<&FrustumParams>,
    ) -> u64 {
        let mut hasher = DefaultHasher::new();

        // Hash atmosphere params
        params.compute_hash().hash(&mut hasher);

        // Hash LUT type
        (lut_type as u8).hash(&mut hasher);

        // Hash config
        self.config.transmittance_size.hash(&mut hasher);
        self.config.sky_view_size.hash(&mut hasher);
        self.config.aerial_perspective_size.hash(&mut hasher);
        (self.config.precision as u8).hash(&mut hasher);

        // Hash sun elevation for sky view
        sun_elevation.to_bits().hash(&mut hasher);

        // Hash frustum for aerial perspective
        if let Some(f) = frustum {
            f.compute_hash().hash(&mut hasher);
        }

        hasher.finish()
    }
}

// ---------------------------------------------------------------------------
// LutCache
// ---------------------------------------------------------------------------

/// Disk cache for cooked LUTs.
///
/// Stores cooked LUTs as binary files indexed by their content hash.
pub struct LutCache {
    cache_dir: PathBuf,
}

impl LutCache {
    /// Creates a new LUT cache at the specified directory.
    ///
    /// The directory will be created if it doesn't exist.
    pub fn new(cache_dir: &Path) -> Self {
        if !cache_dir.exists() {
            let _ = fs::create_dir_all(cache_dir);
        }
        Self {
            cache_dir: cache_dir.to_path_buf(),
        }
    }

    /// Returns the cache directory path.
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Retrieves a cached LUT by hash.
    ///
    /// Returns `None` if the LUT is not in the cache or cannot be read.
    pub fn get(&self, hash: u64) -> Option<CookedLut> {
        let path = self.hash_to_path(hash);
        if !path.exists() {
            return None;
        }

        let mut file = File::open(&path).ok()?;
        let mut data = Vec::new();
        file.read_to_end(&mut data).ok()?;

        // Parse header (16 bytes)
        if data.len() < 16 {
            return None;
        }

        let lut_type = match data[0] {
            0 => LutType::Transmittance,
            1 => LutType::SkyView,
            2 => LutType::AerialPerspective,
            _ => return None,
        };

        let precision = match data[1] {
            0 => LutPrecision::Half,
            1 => LutPrecision::Full,
            _ => return None,
        };

        let dimensions = [
            u32::from_le_bytes([data[2], data[3], data[4], data[5]]),
            u32::from_le_bytes([data[6], data[7], data[8], data[9]]),
            u32::from_le_bytes([data[10], data[11], data[12], data[13]]),
        ];

        // Skip 2 padding bytes
        let payload = data[16..].to_vec();

        Some(CookedLut {
            lut_type,
            dimensions,
            data: payload,
            format: precision.rgba_format(),
            hash,
        })
    }

    /// Stores a LUT in the cache.
    pub fn put(&self, hash: u64, lut: &CookedLut) {
        let path = self.hash_to_path(hash);

        // Create parent directories if needed
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }

        let mut file = match File::create(&path) {
            Ok(f) => f,
            Err(_) => return,
        };

        // Write header (16 bytes)
        let mut header = [0u8; 16];
        header[0] = lut.lut_type as u8;
        header[1] = match lut.format {
            wgpu::TextureFormat::Rgba16Float => 0,
            wgpu::TextureFormat::Rgba32Float => 1,
            _ => 0,
        };
        header[2..6].copy_from_slice(&lut.dimensions[0].to_le_bytes());
        header[6..10].copy_from_slice(&lut.dimensions[1].to_le_bytes());
        header[10..14].copy_from_slice(&lut.dimensions[2].to_le_bytes());
        // 2 bytes padding at [14..16]

        let _ = file.write_all(&header);
        let _ = file.write_all(&lut.data);
    }

    /// Clears all cached LUTs.
    pub fn clear(&self) {
        if self.cache_dir.exists() {
            let _ = fs::remove_dir_all(&self.cache_dir);
            let _ = fs::create_dir_all(&self.cache_dir);
        }
    }

    /// Computes a deterministic hash for the given parameters and LUT type.
    pub fn compute_hash(params: &AtmosphereParams, lut_type: LutType) -> u64 {
        let mut hasher = DefaultHasher::new();
        params.compute_hash().hash(&mut hasher);
        (lut_type as u8).hash(&mut hasher);
        hasher.finish()
    }

    /// Returns the number of cached entries.
    pub fn entry_count(&self) -> usize {
        if !self.cache_dir.exists() {
            return 0;
        }

        fs::read_dir(&self.cache_dir)
            .map(|entries| entries.filter_map(|e| e.ok()).count())
            .unwrap_or(0)
    }

    /// Returns the total size of cached data in bytes.
    pub fn total_size_bytes(&self) -> u64 {
        if !self.cache_dir.exists() {
            return 0;
        }

        fs::read_dir(&self.cache_dir)
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .filter_map(|e| e.metadata().ok())
                    .map(|m| m.len())
                    .sum()
            })
            .unwrap_or(0)
    }

    fn hash_to_path(&self, hash: u64) -> PathBuf {
        self.cache_dir.join(format!("{:016x}.lut", hash))
    }
}

// ---------------------------------------------------------------------------
// GPU Upload
// ---------------------------------------------------------------------------

/// Handle to an uploaded LUT texture.
pub struct TextureHandle {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
}

/// Handle to a sampler.
pub struct SamplerHandle {
    pub sampler: wgpu::Sampler,
}

/// Uploads a cooked LUT to the GPU.
///
/// Creates a texture with the appropriate dimensions and format, then
/// copies the LUT data to the GPU.
pub fn upload_lut_to_gpu(lut: &CookedLut, device: &wgpu::Device, queue: &wgpu::Queue) -> TextureHandle {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(lut.lut_type.name()),
        size: wgpu::Extent3d {
            width: lut.dimensions[0],
            height: lut.dimensions[1],
            depth_or_array_layers: lut.dimensions[2],
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: lut.texture_dimension(),
        format: lut.format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    // Calculate bytes per row
    let bytes_per_texel = match lut.format {
        wgpu::TextureFormat::Rgba16Float => 8,
        wgpu::TextureFormat::Rgba32Float => 16,
        _ => 8,
    };
    let bytes_per_row = lut.dimensions[0] * bytes_per_texel;

    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &lut.data,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(bytes_per_row),
            rows_per_image: Some(lut.dimensions[1]),
        },
        wgpu::Extent3d {
            width: lut.dimensions[0],
            height: lut.dimensions[1],
            depth_or_array_layers: lut.dimensions[2],
        },
    );

    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some(&format!("{}_view", lut.lut_type.name())),
        ..Default::default()
    });

    TextureHandle { texture, view }
}

/// Creates a sampler suitable for LUT sampling.
///
/// Uses bilinear filtering and clamp-to-edge addressing.
pub fn create_lut_sampler(device: &wgpu::Device) -> SamplerHandle {
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("LUT Sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    SamplerHandle { sampler }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    // -----------------------------------------------------------------------
    // LutPrecision tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_lut_precision_bytes_per_channel_half() {
        assert_eq!(LutPrecision::Half.bytes_per_channel(), 2);
    }

    #[test]
    fn test_lut_precision_bytes_per_channel_full() {
        assert_eq!(LutPrecision::Full.bytes_per_channel(), 4);
    }

    #[test]
    fn test_lut_precision_bytes_per_texel_half() {
        assert_eq!(LutPrecision::Half.bytes_per_texel(), 8);
    }

    #[test]
    fn test_lut_precision_bytes_per_texel_full() {
        assert_eq!(LutPrecision::Full.bytes_per_texel(), 16);
    }

    #[test]
    fn test_lut_precision_rgba_format_half() {
        assert_eq!(LutPrecision::Half.rgba_format(), wgpu::TextureFormat::Rgba16Float);
    }

    #[test]
    fn test_lut_precision_rgba_format_full() {
        assert_eq!(LutPrecision::Full.rgba_format(), wgpu::TextureFormat::Rgba32Float);
    }

    #[test]
    fn test_lut_precision_default() {
        assert_eq!(LutPrecision::default(), LutPrecision::Half);
    }

    #[test]
    fn test_lut_precision_clone_eq() {
        let a = LutPrecision::Half;
        let b = a;
        assert_eq!(a, b);
    }

    // -----------------------------------------------------------------------
    // LutType tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_lut_type_name_transmittance() {
        assert_eq!(LutType::Transmittance.name(), "transmittance");
    }

    #[test]
    fn test_lut_type_name_sky_view() {
        assert_eq!(LutType::SkyView.name(), "sky_view");
    }

    #[test]
    fn test_lut_type_name_aerial_perspective() {
        assert_eq!(LutType::AerialPerspective.name(), "aerial_perspective");
    }

    #[test]
    fn test_lut_type_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(LutType::Transmittance);
        set.insert(LutType::SkyView);
        set.insert(LutType::AerialPerspective);
        assert_eq!(set.len(), 3);
    }

    // -----------------------------------------------------------------------
    // LutCookingConfig tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_default() {
        let config = LutCookingConfig::default();
        assert_eq!(config.transmittance_size, [256, 64]);
        assert_eq!(config.sky_view_size, [256, 512]);
        assert_eq!(config.aerial_perspective_size, [32, 32, 32]);
        assert_eq!(config.precision, LutPrecision::Half);
        assert!(config.cache_enabled);
    }

    #[test]
    fn test_config_high_quality() {
        let config = LutCookingConfig::high_quality();
        assert_eq!(config.transmittance_size, [512, 128]);
        assert_eq!(config.sky_view_size, [512, 1024]);
        assert_eq!(config.aerial_perspective_size, [64, 64, 64]);
        assert_eq!(config.precision, LutPrecision::Full);
    }

    #[test]
    fn test_config_low_quality() {
        let config = LutCookingConfig::low_quality();
        assert_eq!(config.transmittance_size, [128, 32]);
        assert_eq!(config.sky_view_size, [128, 256]);
        assert_eq!(config.aerial_perspective_size, [16, 16, 16]);
    }

    #[test]
    fn test_config_validate_default() {
        let config = LutCookingConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validate_zero_transmittance_width() {
        let mut config = LutCookingConfig::default();
        config.transmittance_size[0] = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_zero_transmittance_height() {
        let mut config = LutCookingConfig::default();
        config.transmittance_size[1] = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_zero_sky_view() {
        let mut config = LutCookingConfig::default();
        config.sky_view_size[0] = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_zero_aerial_perspective() {
        let mut config = LutCookingConfig::default();
        config.aerial_perspective_size[2] = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_oversized_transmittance() {
        let mut config = LutCookingConfig::default();
        config.transmittance_size[0] = 8192;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_oversized_aerial_perspective() {
        let mut config = LutCookingConfig::default();
        config.aerial_perspective_size[0] = 512;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_total_memory_bytes() {
        let config = LutCookingConfig::default();
        let bytes = config.total_memory_bytes();
        // 256*64*8 + 256*512*8 + 32*32*32*8 = 131072 + 1048576 + 262144 = 1441792
        assert!(bytes > 0);
        assert_eq!(bytes, 256 * 64 * 8 + 256 * 512 * 8 + 32 * 32 * 32 * 8);
    }

    #[test]
    fn test_config_new() {
        let config = LutCookingConfig::new(
            [64, 16],
            [64, 128],
            [8, 8, 8],
            LutPrecision::Full,
            false,
        );
        assert_eq!(config.transmittance_size, [64, 16]);
        assert_eq!(config.precision, LutPrecision::Full);
        assert!(!config.cache_enabled);
    }

    #[test]
    fn test_config_pod_zeroable() {
        let config = LutCookingConfig::zeroed();
        assert_eq!(config.transmittance_size, [0, 0]);
    }

    // -----------------------------------------------------------------------
    // AtmosphereParams tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_atmosphere_params_earth() {
        let params = AtmosphereParams::earth();
        assert!((params.planet_radius - 6_371_000.0).abs() < 1.0);
        assert!((params.atmosphere_height - 80_000.0).abs() < 1.0);
    }

    #[test]
    fn test_atmosphere_params_mars() {
        let params = AtmosphereParams::mars();
        assert!((params.planet_radius - 3_389_500.0).abs() < 1.0);
        assert!(params.ozone_absorption[0] == 0.0);
    }

    #[test]
    fn test_atmosphere_params_default() {
        let params = AtmosphereParams::default();
        assert_eq!(params.planet_radius, AtmosphereParams::earth().planet_radius);
    }

    #[test]
    fn test_atmosphere_top_radius() {
        let params = AtmosphereParams::earth();
        let expected = 6_371_000.0 + 80_000.0;
        assert!((params.atmosphere_top_radius() - expected).abs() < 1.0);
    }

    #[test]
    fn test_mie_extinction() {
        let params = AtmosphereParams::earth();
        let expected = 21e-6 + 4.4e-6;
        assert!((params.mie_extinction() - expected).abs() < 1e-9);
    }

    #[test]
    fn test_atmosphere_params_compute_hash() {
        let params1 = AtmosphereParams::earth();
        let params2 = AtmosphereParams::earth();
        assert_eq!(params1.compute_hash(), params2.compute_hash());
    }

    #[test]
    fn test_atmosphere_params_hash_differs() {
        let params1 = AtmosphereParams::earth();
        let params2 = AtmosphereParams::mars();
        assert_ne!(params1.compute_hash(), params2.compute_hash());
    }

    // -----------------------------------------------------------------------
    // FrustumParams tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_frustum_params_default() {
        let frustum = FrustumParams::default();
        assert!(frustum.near > 0.0);
        assert!(frustum.far > frustum.near);
    }

    #[test]
    fn test_frustum_params_compute_hash() {
        let f1 = FrustumParams::default();
        let f2 = FrustumParams::default();
        assert_eq!(f1.compute_hash(), f2.compute_hash());
    }

    #[test]
    fn test_frustum_params_hash_differs() {
        let f1 = FrustumParams::default();
        let f2 = FrustumParams {
            near: 1.0,
            ..FrustumParams::default()
        };
        assert_ne!(f1.compute_hash(), f2.compute_hash());
    }

    // -----------------------------------------------------------------------
    // CookedLut tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cooked_lut_new() {
        let lut = CookedLut::new(
            LutType::Transmittance,
            [256, 64, 1],
            vec![0u8; 256 * 64 * 8],
            wgpu::TextureFormat::Rgba16Float,
            12345,
        );
        assert_eq!(lut.lut_type, LutType::Transmittance);
        assert_eq!(lut.dimensions, [256, 64, 1]);
        assert_eq!(lut.hash, 12345);
    }

    #[test]
    fn test_cooked_lut_size_bytes() {
        let lut = CookedLut::new(
            LutType::Transmittance,
            [256, 64, 1],
            vec![0u8; 256 * 64 * 8],
            wgpu::TextureFormat::Rgba16Float,
            0,
        );
        assert_eq!(lut.size_bytes(), 256 * 64 * 8);
    }

    #[test]
    fn test_cooked_lut_texel_count() {
        let lut = CookedLut::new(
            LutType::AerialPerspective,
            [32, 32, 32],
            vec![0u8; 32 * 32 * 32 * 8],
            wgpu::TextureFormat::Rgba16Float,
            0,
        );
        assert_eq!(lut.texel_count(), 32 * 32 * 32);
    }

    #[test]
    fn test_cooked_lut_is_3d_false() {
        let lut = CookedLut::new(
            LutType::Transmittance,
            [256, 64, 1],
            vec![0u8; 8],
            wgpu::TextureFormat::Rgba16Float,
            0,
        );
        assert!(!lut.is_3d());
    }

    #[test]
    fn test_cooked_lut_is_3d_true() {
        let lut = CookedLut::new(
            LutType::AerialPerspective,
            [32, 32, 32],
            vec![0u8; 8],
            wgpu::TextureFormat::Rgba16Float,
            0,
        );
        assert!(lut.is_3d());
    }

    #[test]
    fn test_cooked_lut_texture_dimension_2d() {
        let lut = CookedLut::new(
            LutType::SkyView,
            [256, 512, 1],
            vec![0u8; 8],
            wgpu::TextureFormat::Rgba16Float,
            0,
        );
        assert_eq!(lut.texture_dimension(), wgpu::TextureDimension::D2);
    }

    #[test]
    fn test_cooked_lut_texture_dimension_3d() {
        let lut = CookedLut::new(
            LutType::AerialPerspective,
            [32, 32, 32],
            vec![0u8; 8],
            wgpu::TextureFormat::Rgba16Float,
            0,
        );
        assert_eq!(lut.texture_dimension(), wgpu::TextureDimension::D3);
    }

    // -----------------------------------------------------------------------
    // LutSet tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_lut_set_new() {
        let trans = CookedLut::new(LutType::Transmittance, [1, 1, 1], vec![0u8; 8], wgpu::TextureFormat::Rgba16Float, 1);
        let sky = CookedLut::new(LutType::SkyView, [1, 1, 1], vec![0u8; 8], wgpu::TextureFormat::Rgba16Float, 2);
        let aerial = CookedLut::new(LutType::AerialPerspective, [1, 1, 1], vec![0u8; 8], wgpu::TextureFormat::Rgba16Float, 3);

        let set = LutSet::new(trans, sky, aerial);
        assert_eq!(set.transmittance.hash, 1);
        assert_eq!(set.sky_view.hash, 2);
        assert_eq!(set.aerial_perspective.hash, 3);
    }

    #[test]
    fn test_lut_set_total_bytes() {
        let trans = CookedLut::new(LutType::Transmittance, [1, 1, 1], vec![0u8; 100], wgpu::TextureFormat::Rgba16Float, 1);
        let sky = CookedLut::new(LutType::SkyView, [1, 1, 1], vec![0u8; 200], wgpu::TextureFormat::Rgba16Float, 2);
        let aerial = CookedLut::new(LutType::AerialPerspective, [1, 1, 1], vec![0u8; 300], wgpu::TextureFormat::Rgba16Float, 3);

        let set = LutSet::new(trans, sky, aerial);
        assert_eq!(set.total_bytes(), 600);
    }

    #[test]
    fn test_lut_set_combined_hash() {
        let trans = CookedLut::new(LutType::Transmittance, [1, 1, 1], vec![0u8; 8], wgpu::TextureFormat::Rgba16Float, 1);
        let sky = CookedLut::new(LutType::SkyView, [1, 1, 1], vec![0u8; 8], wgpu::TextureFormat::Rgba16Float, 2);
        let aerial = CookedLut::new(LutType::AerialPerspective, [1, 1, 1], vec![0u8; 8], wgpu::TextureFormat::Rgba16Float, 3);

        let set1 = LutSet::new(trans.clone(), sky.clone(), aerial.clone());
        let set2 = LutSet::new(trans, sky, aerial);

        assert_eq!(set1.combined_hash(), set2.combined_hash());
    }

    // -----------------------------------------------------------------------
    // LutCookingPipeline tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_pipeline_new() {
        let config = LutCookingConfig::default();
        let pipeline = LutCookingPipeline::new(config);
        assert_eq!(pipeline.config().transmittance_size, [256, 64]);
    }

    #[test]
    fn test_pipeline_cook_transmittance_dimensions() {
        let config = LutCookingConfig::low_quality();
        let pipeline = LutCookingPipeline::new(config);
        let params = AtmosphereParams::earth();

        let lut = pipeline.cook_transmittance(&params);

        assert_eq!(lut.dimensions[0], 128);
        assert_eq!(lut.dimensions[1], 32);
        assert_eq!(lut.dimensions[2], 1);
    }

    #[test]
    fn test_pipeline_cook_transmittance_format() {
        let config = LutCookingConfig::low_quality();
        let pipeline = LutCookingPipeline::new(config);
        let params = AtmosphereParams::earth();

        let lut = pipeline.cook_transmittance(&params);

        assert_eq!(lut.format, wgpu::TextureFormat::Rgba16Float);
    }

    #[test]
    fn test_pipeline_cook_transmittance_format_full() {
        let mut config = LutCookingConfig::low_quality();
        config.precision = LutPrecision::Full;
        let pipeline = LutCookingPipeline::new(config);
        let params = AtmosphereParams::earth();

        let lut = pipeline.cook_transmittance(&params);

        assert_eq!(lut.format, wgpu::TextureFormat::Rgba32Float);
    }

    #[test]
    fn test_pipeline_cook_transmittance_data_size() {
        let config = LutCookingConfig::low_quality();
        let pipeline = LutCookingPipeline::new(config);
        let params = AtmosphereParams::earth();

        let lut = pipeline.cook_transmittance(&params);

        // 128 * 32 * 8 bytes per texel (RGBA16F)
        assert_eq!(lut.data.len(), 128 * 32 * 8);
    }

    #[test]
    fn test_pipeline_cook_transmittance_type() {
        let config = LutCookingConfig::low_quality();
        let pipeline = LutCookingPipeline::new(config);
        let params = AtmosphereParams::earth();

        let lut = pipeline.cook_transmittance(&params);

        assert_eq!(lut.lut_type, LutType::Transmittance);
    }

    #[test]
    fn test_pipeline_cook_sky_view_dimensions() {
        let config = LutCookingConfig::low_quality();
        let pipeline = LutCookingPipeline::new(config);
        let params = AtmosphereParams::earth();

        let lut = pipeline.cook_sky_view(&params, 0.5);

        assert_eq!(lut.dimensions[0], 128);
        assert_eq!(lut.dimensions[1], 256);
        assert_eq!(lut.dimensions[2], 1);
    }

    #[test]
    fn test_pipeline_cook_sky_view_sun_elevation_0() {
        let config = LutCookingConfig::low_quality();
        let pipeline = LutCookingPipeline::new(config);
        let params = AtmosphereParams::earth();

        let lut = pipeline.cook_sky_view(&params, 0.0);

        assert_eq!(lut.lut_type, LutType::SkyView);
    }

    #[test]
    fn test_pipeline_cook_sky_view_sun_elevation_90() {
        let config = LutCookingConfig::low_quality();
        let pipeline = LutCookingPipeline::new(config);
        let params = AtmosphereParams::earth();

        let lut = pipeline.cook_sky_view(&params, std::f32::consts::FRAC_PI_2);

        assert_eq!(lut.lut_type, LutType::SkyView);
    }

    #[test]
    fn test_pipeline_cook_sky_view_different_elevations_different_hashes() {
        let config = LutCookingConfig::low_quality();
        let pipeline = LutCookingPipeline::new(config);
        let params = AtmosphereParams::earth();

        let lut1 = pipeline.cook_sky_view(&params, 0.0);
        let lut2 = pipeline.cook_sky_view(&params, 0.5);

        assert_ne!(lut1.hash, lut2.hash);
    }

    #[test]
    fn test_pipeline_cook_aerial_perspective_dimensions() {
        let config = LutCookingConfig::low_quality();
        let pipeline = LutCookingPipeline::new(config);
        let params = AtmosphereParams::earth();
        let frustum = FrustumParams::default();

        let lut = pipeline.cook_aerial_perspective(&params, &frustum);

        assert_eq!(lut.dimensions[0], 16);
        assert_eq!(lut.dimensions[1], 16);
        assert_eq!(lut.dimensions[2], 16);
    }

    #[test]
    fn test_pipeline_cook_aerial_perspective_is_3d() {
        let config = LutCookingConfig::low_quality();
        let pipeline = LutCookingPipeline::new(config);
        let params = AtmosphereParams::earth();
        let frustum = FrustumParams::default();

        let lut = pipeline.cook_aerial_perspective(&params, &frustum);

        assert!(lut.is_3d());
    }

    #[test]
    fn test_pipeline_cook_aerial_perspective_type() {
        let config = LutCookingConfig::low_quality();
        let pipeline = LutCookingPipeline::new(config);
        let params = AtmosphereParams::earth();
        let frustum = FrustumParams::default();

        let lut = pipeline.cook_aerial_perspective(&params, &frustum);

        assert_eq!(lut.lut_type, LutType::AerialPerspective);
    }

    #[test]
    fn test_pipeline_cook_all() {
        let config = LutCookingConfig::low_quality();
        let pipeline = LutCookingPipeline::new(config);
        let params = AtmosphereParams::earth();

        let lut_set = pipeline.cook_all(&params);

        assert_eq!(lut_set.transmittance.lut_type, LutType::Transmittance);
        assert_eq!(lut_set.sky_view.lut_type, LutType::SkyView);
        assert_eq!(lut_set.aerial_perspective.lut_type, LutType::AerialPerspective);
    }

    #[test]
    fn test_pipeline_cook_all_total_bytes() {
        let config = LutCookingConfig::low_quality();
        let pipeline = LutCookingPipeline::new(config);
        let params = AtmosphereParams::earth();

        let lut_set = pipeline.cook_all(&params);

        // All three LUTs should have data
        assert!(lut_set.total_bytes() > 0);
    }

    #[test]
    fn test_pipeline_validate_lut_valid() {
        let config = LutCookingConfig::low_quality();
        let pipeline = LutCookingPipeline::new(config);
        let params = AtmosphereParams::earth();

        let lut = pipeline.cook_transmittance(&params);

        assert!(pipeline.validate_lut(&lut));
    }

    #[test]
    fn test_pipeline_validate_lut_with_nan() {
        let config = LutCookingConfig::low_quality();
        let pipeline = LutCookingPipeline::new(config);

        // Create LUT with NaN value
        let mut data = vec![0u8; 8];
        let nan_bits = half::f16::from_f32(f32::NAN).to_bits().to_le_bytes();
        data[0..2].copy_from_slice(&nan_bits);

        let lut = CookedLut::new(
            LutType::Transmittance,
            [1, 1, 1],
            data,
            wgpu::TextureFormat::Rgba16Float,
            0,
        );

        assert!(!pipeline.validate_lut(&lut));
    }

    #[test]
    fn test_pipeline_validate_lut_with_inf() {
        let config = LutCookingConfig::low_quality();
        let pipeline = LutCookingPipeline::new(config);

        // Create LUT with infinity value
        let mut data = vec![0u8; 8];
        let inf_bits = half::f16::from_f32(f32::INFINITY).to_bits().to_le_bytes();
        data[0..2].copy_from_slice(&inf_bits);

        let lut = CookedLut::new(
            LutType::Transmittance,
            [1, 1, 1],
            data,
            wgpu::TextureFormat::Rgba16Float,
            0,
        );

        assert!(!pipeline.validate_lut(&lut));
    }

    #[test]
    fn test_pipeline_validate_lut_full_precision() {
        let mut config = LutCookingConfig::low_quality();
        config.precision = LutPrecision::Full;
        let pipeline = LutCookingPipeline::new(config);
        let params = AtmosphereParams::earth();

        let lut = pipeline.cook_transmittance(&params);

        assert!(pipeline.validate_lut(&lut));
    }

    #[test]
    fn test_pipeline_hash_determinism() {
        let config = LutCookingConfig::low_quality();
        let pipeline = LutCookingPipeline::new(config);
        let params = AtmosphereParams::earth();

        let lut1 = pipeline.cook_transmittance(&params);
        let lut2 = pipeline.cook_transmittance(&params);

        assert_eq!(lut1.hash, lut2.hash);
    }

    #[test]
    fn test_pipeline_hash_different_params() {
        let config = LutCookingConfig::low_quality();
        let pipeline = LutCookingPipeline::new(config);

        let lut1 = pipeline.cook_transmittance(&AtmosphereParams::earth());
        let lut2 = pipeline.cook_transmittance(&AtmosphereParams::mars());

        assert_ne!(lut1.hash, lut2.hash);
    }

    // -----------------------------------------------------------------------
    // LutCache tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cache_new() {
        let temp_dir = TempDir::new().unwrap();
        let cache = LutCache::new(temp_dir.path());
        assert!(cache.cache_dir().exists());
    }

    #[test]
    fn test_cache_get_nonexistent() {
        let temp_dir = TempDir::new().unwrap();
        let cache = LutCache::new(temp_dir.path());

        let result = cache.get(12345);
        assert!(result.is_none());
    }

    #[test]
    fn test_cache_put_get() {
        let temp_dir = TempDir::new().unwrap();
        let cache = LutCache::new(temp_dir.path());

        let lut = CookedLut::new(
            LutType::Transmittance,
            [128, 32, 1],
            vec![0u8; 128 * 32 * 8],
            wgpu::TextureFormat::Rgba16Float,
            99999,
        );

        cache.put(99999, &lut);

        let retrieved = cache.get(99999);
        assert!(retrieved.is_some());

        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.lut_type, LutType::Transmittance);
        assert_eq!(retrieved.dimensions, [128, 32, 1]);
        assert_eq!(retrieved.hash, 99999);
    }

    #[test]
    fn test_cache_clear() {
        let temp_dir = TempDir::new().unwrap();
        let cache = LutCache::new(temp_dir.path());

        let lut = CookedLut::new(
            LutType::Transmittance,
            [1, 1, 1],
            vec![0u8; 8],
            wgpu::TextureFormat::Rgba16Float,
            11111,
        );

        cache.put(11111, &lut);
        assert!(cache.get(11111).is_some());

        cache.clear();
        assert!(cache.get(11111).is_none());
    }

    #[test]
    fn test_cache_entry_count() {
        let temp_dir = TempDir::new().unwrap();
        let cache = LutCache::new(temp_dir.path());

        assert_eq!(cache.entry_count(), 0);

        let lut1 = CookedLut::new(LutType::Transmittance, [1, 1, 1], vec![0u8; 8], wgpu::TextureFormat::Rgba16Float, 1);
        let lut2 = CookedLut::new(LutType::SkyView, [1, 1, 1], vec![0u8; 8], wgpu::TextureFormat::Rgba16Float, 2);

        cache.put(1, &lut1);
        cache.put(2, &lut2);

        assert_eq!(cache.entry_count(), 2);
    }

    #[test]
    fn test_cache_total_size_bytes() {
        let temp_dir = TempDir::new().unwrap();
        let cache = LutCache::new(temp_dir.path());

        assert_eq!(cache.total_size_bytes(), 0);

        let lut = CookedLut::new(
            LutType::Transmittance,
            [1, 1, 1],
            vec![0u8; 100],
            wgpu::TextureFormat::Rgba16Float,
            1,
        );

        cache.put(1, &lut);

        // Should be > 100 bytes (header + payload)
        assert!(cache.total_size_bytes() > 100);
    }

    #[test]
    fn test_cache_compute_hash() {
        let params = AtmosphereParams::earth();

        let hash1 = LutCache::compute_hash(&params, LutType::Transmittance);
        let hash2 = LutCache::compute_hash(&params, LutType::Transmittance);

        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_cache_compute_hash_different_types() {
        let params = AtmosphereParams::earth();

        let hash1 = LutCache::compute_hash(&params, LutType::Transmittance);
        let hash2 = LutCache::compute_hash(&params, LutType::SkyView);

        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_cache_preserves_lut_type() {
        let temp_dir = TempDir::new().unwrap();
        let cache = LutCache::new(temp_dir.path());

        let lut = CookedLut::new(
            LutType::AerialPerspective,
            [32, 32, 32],
            vec![0u8; 32 * 32 * 32 * 8],
            wgpu::TextureFormat::Rgba16Float,
            77777,
        );

        cache.put(77777, &lut);

        let retrieved = cache.get(77777).unwrap();
        assert_eq!(retrieved.lut_type, LutType::AerialPerspective);
    }

    #[test]
    fn test_cache_preserves_precision_half() {
        let temp_dir = TempDir::new().unwrap();
        let cache = LutCache::new(temp_dir.path());

        let lut = CookedLut::new(
            LutType::Transmittance,
            [1, 1, 1],
            vec![0u8; 8],
            wgpu::TextureFormat::Rgba16Float,
            88888,
        );

        cache.put(88888, &lut);

        let retrieved = cache.get(88888).unwrap();
        assert_eq!(retrieved.format, wgpu::TextureFormat::Rgba16Float);
    }

    #[test]
    fn test_cache_preserves_precision_full() {
        let temp_dir = TempDir::new().unwrap();
        let cache = LutCache::new(temp_dir.path());

        let lut = CookedLut::new(
            LutType::Transmittance,
            [1, 1, 1],
            vec![0u8; 16],
            wgpu::TextureFormat::Rgba32Float,
            99999,
        );

        cache.put(99999, &lut);

        let retrieved = cache.get(99999).unwrap();
        assert_eq!(retrieved.format, wgpu::TextureFormat::Rgba32Float);
    }

    // -----------------------------------------------------------------------
    // GPU Upload tests (require device)
    // -----------------------------------------------------------------------

    #[test]
    fn test_upload_lut_to_gpu() {
        let device = match crate::rhi_device::RhiDevice::try_new_headless() {
            Some(d) => d,
            None => {
                eprintln!("Skipping GPU test: no adapter available");
                return;
            }
        };

        let lut = CookedLut::new(
            LutType::Transmittance,
            [64, 16, 1],
            vec![0u8; 64 * 16 * 8],
            wgpu::TextureFormat::Rgba16Float,
            0,
        );

        let handle = upload_lut_to_gpu(&lut, &device.device, &device.queue);

        let size = handle.texture.size();
        assert_eq!(size.width, 64);
        assert_eq!(size.height, 16);
    }

    #[test]
    fn test_upload_lut_to_gpu_3d() {
        let device = match crate::rhi_device::RhiDevice::try_new_headless() {
            Some(d) => d,
            None => {
                eprintln!("Skipping GPU test: no adapter available");
                return;
            }
        };

        let lut = CookedLut::new(
            LutType::AerialPerspective,
            [8, 8, 8],
            vec![0u8; 8 * 8 * 8 * 8],
            wgpu::TextureFormat::Rgba16Float,
            0,
        );

        let handle = upload_lut_to_gpu(&lut, &device.device, &device.queue);

        let size = handle.texture.size();
        assert_eq!(size.width, 8);
        assert_eq!(size.height, 8);
        assert_eq!(size.depth_or_array_layers, 8);
    }

    #[test]
    fn test_create_lut_sampler() {
        let device = match crate::rhi_device::RhiDevice::try_new_headless() {
            Some(d) => d,
            None => {
                eprintln!("Skipping GPU test: no adapter available");
                return;
            }
        };

        let handle = create_lut_sampler(&device.device);
        // Sampler creation succeeded if we get here
        let _ = handle.sampler;
    }
}
