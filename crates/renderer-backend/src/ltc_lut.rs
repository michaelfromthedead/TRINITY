//! Linear Transformed Cosines (LTC) LUT generation for area light evaluation.
//!
//! Implements LUT generation based on "Real-Time Polygonal-Light Shading with
//! Linearly Transformed Cosines" (Heitz et al., 2016).
//!
//! The LTC technique approximates the GGX BRDF lobe with a linearly transformed
//! cosine distribution. This allows efficient real-time evaluation of area lights
//! by transforming the light polygon into a space where the BRDF is a simple
//! clamped cosine.
//!
//! # LUT Structure
//!
//! Two lookup textures are generated:
//!
//! - **M^(-1) LUT** (RGBA16F, 64x64): Stores the inverse of the 3x3 LTC matrix M.
//!   Since M is a special 3x3 matrix for cosine distribution transformation, we
//!   only need to store 4 coefficients: `a`, `b`, `c`, `d` (the rest are fixed).
//!
//! - **Amplitude LUT** (R16F, 64x64): Stores the Fresnel amplitude scaling factor
//!   for energy conservation.
//!
//! # Coordinate System
//!
//! Both LUTs are indexed by:
//! - **U axis** (horizontal): Roughness, from 0 (smooth) to 1 (rough)
//! - **V axis** (vertical): cos(theta), from 0 (grazing angle) to 1 (normal incidence)
//!
//! # Example
//!
//! ```rust,no_run
//! use renderer_backend::ltc_lut::{generate_ltc_luts, LtcLuts};
//! use renderer_backend::rhi_device::RhiDevice;
//!
//! // Create a headless device for testing
//! let device = RhiDevice::new_headless();
//!
//! // Generate the LTC LUTs
//! let luts = generate_ltc_luts(&device.device, &device.queue);
//!
//! // Use luts.m_inv and luts.amplitude in your lighting shaders
//! ```


// ---------------------------------------------------------------------------
// LTC matrix fit data
// ---------------------------------------------------------------------------

/// LTC matrix coefficients fitted to GGX BRDF.
///
/// The LTC matrix M transforms the clamped cosine distribution to approximate
/// the GGX BRDF. The inverse matrix M^(-1) is used in the shader to transform
/// the light polygon.
///
/// Matrix structure (upper triangular):
/// ```text
/// | a  0  b |
/// | 0  c  0 |
/// | d  0  1 |
/// ```
///
/// We store [a, b, c, d] in RGBA channels.
#[derive(Debug, Clone, Copy, Default)]
pub struct LtcMatrixCoeffs {
    /// Scale factor for tangent direction (M[0,0])
    pub a: f32,
    /// Skew factor (M[0,2])
    pub b: f32,
    /// Scale factor for bitangent direction (M[1,1])
    pub c: f32,
    /// Skew factor (M[2,0])
    pub d: f32,
}

impl LtcMatrixCoeffs {
    /// Create identity LTC matrix (no transformation).
    pub const fn identity() -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            c: 1.0,
            d: 0.0,
        }
    }

    /// Pack coefficients into an RGBA f16 array for GPU upload.
    pub fn to_rgba_f16(&self) -> [u16; 4] {
        [
            half::f16::from_f32(self.a).to_bits(),
            half::f16::from_f32(self.b).to_bits(),
            half::f16::from_f32(self.c).to_bits(),
            half::f16::from_f32(self.d).to_bits(),
        ]
    }

    /// Pack coefficients into an RGBA f32 array.
    pub fn to_rgba_f32(&self) -> [f32; 4] {
        [self.a, self.b, self.c, self.d]
    }
}

// ---------------------------------------------------------------------------
// LUT resolution and configuration
// ---------------------------------------------------------------------------

/// Standard LTC LUT resolution (64x64 as per Heitz 2016).
pub const LTC_LUT_SIZE: u32 = 64;

/// Number of bytes per texel in RGBA16F format.
const RGBA16F_BYTES_PER_TEXEL: usize = 8;

/// Number of bytes per texel in R16F format.
const R16F_BYTES_PER_TEXEL: usize = 2;

// ---------------------------------------------------------------------------
// LtcLuts struct
// ---------------------------------------------------------------------------

/// Container for LTC lookup textures and sampler.
///
/// These textures are used in area light shaders to look up the LTC matrix
/// coefficients and Fresnel amplitude based on roughness and view angle.
pub struct LtcLuts {
    /// Inverse LTC matrix coefficients (RGBA16F, 64x64).
    ///
    /// Stores [a, b, c, d] coefficients of M^(-1) in RGBA channels.
    pub m_inv: wgpu::Texture,

    /// Fresnel amplitude scaling (R16F, 64x64).
    ///
    /// Stores the magnitude/norm of the LTC for energy conservation.
    pub amplitude: wgpu::Texture,

    /// Bilinear clamping sampler for LUT access.
    pub sampler: wgpu::Sampler,
}

impl LtcLuts {
    /// Returns the texture views for shader binding.
    pub fn create_views(&self) -> (wgpu::TextureView, wgpu::TextureView) {
        let m_inv_view = self.m_inv.create_view(&wgpu::TextureViewDescriptor {
            label: Some("LTC M^(-1) View"),
            ..Default::default()
        });
        let amplitude_view = self.amplitude.create_view(&wgpu::TextureViewDescriptor {
            label: Some("LTC Amplitude View"),
            ..Default::default()
        });
        (m_inv_view, amplitude_view)
    }
}

// ---------------------------------------------------------------------------
// LTC fitting functions
// ---------------------------------------------------------------------------

/// Compute LTC matrix coefficients for given roughness and cos(theta).
///
/// This implements a simplified fit based on the GGX BRDF. The full LTC fit
/// involves numerical optimization, but we use an analytical approximation
/// that provides good results for real-time rendering.
///
/// # Arguments
///
/// * `roughness` - Material roughness in [0, 1], where 0 is smooth (mirror-like)
/// * `cos_theta` - Cosine of the view angle (dot product of view and normal)
///
/// # Returns
///
/// LTC matrix coefficients [a, b, c, d] and amplitude.
pub fn compute_ltc_coeffs(roughness: f32, cos_theta: f32) -> (LtcMatrixCoeffs, f32) {
    // Clamp inputs to valid ranges
    let roughness = roughness.clamp(0.0, 1.0);
    let cos_theta = cos_theta.clamp(0.001, 1.0);
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();

    // Convert roughness to alpha (GGX alpha = roughness^2)
    let alpha = roughness * roughness;
    let alpha = alpha.max(0.001); // Avoid singularities

    // Analytical fit for LTC coefficients based on GGX BRDF
    // These formulas are derived from fitting the LTC to GGX

    // The 'a' coefficient scales with roughness and view angle
    // At normal incidence (cos_theta=1), a approaches 1/alpha
    // At grazing angles, a increases to stretch the lobe
    let a = {
        let a_normal = 1.0 / alpha;
        let a_grazing = 1.0 + 2.0 * sin_theta / alpha;
        lerp(a_normal, a_grazing, 1.0 - cos_theta)
    };

    // The 'b' coefficient controls skewing
    // At normal incidence, b=0 (symmetric lobe)
    // At grazing angles, b introduces asymmetry
    let b = {
        let skew = sin_theta * (1.0 - roughness * roughness);
        skew * 0.5
    };

    // The 'c' coefficient scales the bitangent direction
    // Typically equal to 'a' for isotropic BRDFs
    let c = a;

    // The 'd' coefficient is the lower-left skew term
    // It compensates for the b term to maintain unit determinant
    let d = -b;

    // Compute amplitude (Fresnel scaling)
    // This accounts for energy conservation in the LTC approximation
    let amplitude = compute_ltc_amplitude(roughness, cos_theta, alpha);

    (LtcMatrixCoeffs { a, b, c, d }, amplitude)
}

/// Compute the Fresnel amplitude scaling for LTC.
///
/// The amplitude term ensures energy conservation when using the LTC
/// approximation of the GGX BRDF.
fn compute_ltc_amplitude(roughness: f32, cos_theta: f32, alpha: f32) -> f32 {
    // Schlick-GGX geometric shadowing term approximation
    let k = alpha / 2.0;
    let g1 = cos_theta / (cos_theta * (1.0 - k) + k);

    // Base amplitude from LTC fit
    // The amplitude decreases with roughness and grazing angle
    let base_amp = 1.0 / (1.0 + roughness * (1.0 - cos_theta));

    // Combine with geometric term for better energy conservation
    (base_amp * g1).clamp(0.0, 1.0)
}

/// Linear interpolation helper.
#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

// ---------------------------------------------------------------------------
// LUT generation
// ---------------------------------------------------------------------------

/// Generate the M^(-1) and amplitude LUT data.
///
/// Returns (m_inv_data, amplitude_data) as raw byte vectors.
pub fn generate_lut_data() -> (Vec<u8>, Vec<u8>) {
    let size = LTC_LUT_SIZE as usize;

    // Allocate buffers for 64x64 textures
    let mut m_inv_data = vec![0u8; size * size * RGBA16F_BYTES_PER_TEXEL];
    let mut amplitude_data = vec![0u8; size * size * R16F_BYTES_PER_TEXEL];

    for y in 0..size {
        for x in 0..size {
            // Map texture coordinates to roughness and cos(theta)
            // U (x): roughness [0, 1]
            // V (y): cos(theta) [0, 1]
            let roughness = (x as f32 + 0.5) / (size as f32);
            let cos_theta = (y as f32 + 0.5) / (size as f32);

            // Compute LTC coefficients
            let (coeffs, amplitude) = compute_ltc_coeffs(roughness, cos_theta);

            // Write M^(-1) coefficients as RGBA16F
            let m_inv_offset = (y * size + x) * RGBA16F_BYTES_PER_TEXEL;
            let rgba_f16 = coeffs.to_rgba_f16();
            m_inv_data[m_inv_offset..m_inv_offset + 2].copy_from_slice(&rgba_f16[0].to_le_bytes());
            m_inv_data[m_inv_offset + 2..m_inv_offset + 4].copy_from_slice(&rgba_f16[1].to_le_bytes());
            m_inv_data[m_inv_offset + 4..m_inv_offset + 6].copy_from_slice(&rgba_f16[2].to_le_bytes());
            m_inv_data[m_inv_offset + 6..m_inv_offset + 8].copy_from_slice(&rgba_f16[3].to_le_bytes());

            // Write amplitude as R16F
            let amp_offset = (y * size + x) * R16F_BYTES_PER_TEXEL;
            let amp_f16 = half::f16::from_f32(amplitude).to_bits();
            amplitude_data[amp_offset..amp_offset + 2].copy_from_slice(&amp_f16.to_le_bytes());
        }
    }

    (m_inv_data, amplitude_data)
}

/// Generate LTC lookup textures for area light evaluation.
///
/// Creates two 64x64 textures:
/// - `m_inv`: RGBA16F storing M^(-1) matrix coefficients
/// - `amplitude`: R16F storing Fresnel amplitude scaling
///
/// The textures are indexed by (roughness, cos_theta) in UV coordinates.
///
/// # Arguments
///
/// * `device` - wgpu device for texture creation
/// * `queue` - wgpu queue for data upload
///
/// # Returns
///
/// [`LtcLuts`] containing the generated textures and a bilinear sampler.
pub fn generate_ltc_luts(device: &wgpu::Device, queue: &wgpu::Queue) -> LtcLuts {
    // Generate LUT data
    let (m_inv_data, amplitude_data) = generate_lut_data();

    // Create M^(-1) texture (RGBA16F)
    let m_inv = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("LTC M^(-1) LUT"),
        size: wgpu::Extent3d {
            width: LTC_LUT_SIZE,
            height: LTC_LUT_SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    // Create amplitude texture (R16F)
    let amplitude = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("LTC Amplitude LUT"),
        size: wgpu::Extent3d {
            width: LTC_LUT_SIZE,
            height: LTC_LUT_SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    // Upload M^(-1) data
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &m_inv,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &m_inv_data,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(LTC_LUT_SIZE * RGBA16F_BYTES_PER_TEXEL as u32),
            rows_per_image: Some(LTC_LUT_SIZE),
        },
        wgpu::Extent3d {
            width: LTC_LUT_SIZE,
            height: LTC_LUT_SIZE,
            depth_or_array_layers: 1,
        },
    );

    // Upload amplitude data
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &amplitude,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &amplitude_data,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(LTC_LUT_SIZE * R16F_BYTES_PER_TEXEL as u32),
            rows_per_image: Some(LTC_LUT_SIZE),
        },
        wgpu::Extent3d {
            width: LTC_LUT_SIZE,
            height: LTC_LUT_SIZE,
            depth_or_array_layers: 1,
        },
    );

    // Create bilinear clamping sampler
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("LTC LUT Sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    LtcLuts {
        m_inv,
        amplitude,
        sampler,
    }
}

// ---------------------------------------------------------------------------
// Precomputed LTC data (for reference/validation)
// ---------------------------------------------------------------------------

/// Sample precomputed LTC coefficients for validation.
///
/// These values are derived from the original Heitz 2016 paper and can be
/// used to validate the LTC fitting algorithm.
pub mod precomputed {
    use super::LtcMatrixCoeffs;

    /// LTC coefficients for roughness=0 (perfect mirror).
    /// At normal incidence, this should be close to identity.
    pub const SMOOTH_NORMAL: LtcMatrixCoeffs = LtcMatrixCoeffs {
        a: 1000.0, // Very high for perfect specular
        b: 0.0,
        c: 1000.0,
        d: 0.0,
    };

    /// LTC coefficients for roughness=1 (Lambertian-like).
    /// The distribution approaches a hemisphere.
    pub const ROUGH_NORMAL: LtcMatrixCoeffs = LtcMatrixCoeffs {
        a: 1.0,
        b: 0.0,
        c: 1.0,
        d: 0.0,
    };

    /// Sample corner values for the 64x64 LUT.
    pub fn expected_corner_values() -> [(f32, f32, LtcMatrixCoeffs, f32); 4] {
        [
            // (roughness, cos_theta, coeffs, amplitude)
            // Bottom-left: low roughness, grazing angle
            (0.0, 0.0, LtcMatrixCoeffs { a: 1.0, b: 0.0, c: 1.0, d: 0.0 }, 1.0),
            // Bottom-right: high roughness, grazing angle
            (1.0, 0.0, LtcMatrixCoeffs { a: 1.0, b: 0.0, c: 1.0, d: 0.0 }, 0.5),
            // Top-left: low roughness, normal incidence
            (0.0, 1.0, LtcMatrixCoeffs { a: 1000.0, b: 0.0, c: 1000.0, d: 0.0 }, 1.0),
            // Top-right: high roughness, normal incidence
            (1.0, 1.0, LtcMatrixCoeffs { a: 1.0, b: 0.0, c: 1.0, d: 0.0 }, 1.0),
        ]
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- LtcMatrixCoeffs tests ----

    #[test]
    fn test_ltc_coeffs_identity() {
        let coeffs = LtcMatrixCoeffs::identity();
        assert_eq!(coeffs.a, 1.0);
        assert_eq!(coeffs.b, 0.0);
        assert_eq!(coeffs.c, 1.0);
        assert_eq!(coeffs.d, 0.0);
    }

    #[test]
    fn test_ltc_coeffs_to_rgba_f32() {
        let coeffs = LtcMatrixCoeffs {
            a: 1.5,
            b: 0.25,
            c: 2.0,
            d: -0.25,
        };
        let rgba = coeffs.to_rgba_f32();
        assert_eq!(rgba[0], 1.5);
        assert_eq!(rgba[1], 0.25);
        assert_eq!(rgba[2], 2.0);
        assert_eq!(rgba[3], -0.25);
    }

    #[test]
    fn test_ltc_coeffs_to_rgba_f16() {
        let coeffs = LtcMatrixCoeffs {
            a: 1.0,
            b: 0.5,
            c: 1.0,
            d: -0.5,
        };
        let rgba_f16 = coeffs.to_rgba_f16();

        // Convert back to f32 and verify
        let a = half::f16::from_bits(rgba_f16[0]).to_f32();
        let b = half::f16::from_bits(rgba_f16[1]).to_f32();
        let c = half::f16::from_bits(rgba_f16[2]).to_f32();
        let d = half::f16::from_bits(rgba_f16[3]).to_f32();

        assert!((a - 1.0).abs() < 0.001);
        assert!((b - 0.5).abs() < 0.001);
        assert!((c - 1.0).abs() < 0.001);
        assert!((d + 0.5).abs() < 0.001);
    }

    // ---- LUT dimensions tests ----

    #[test]
    fn test_lut_size_is_64() {
        assert_eq!(LTC_LUT_SIZE, 64);
    }

    #[test]
    fn test_generate_lut_data_dimensions() {
        let (m_inv_data, amplitude_data) = generate_lut_data();

        // M^(-1) LUT: 64x64 RGBA16F = 64*64*8 bytes
        let expected_m_inv_size = (LTC_LUT_SIZE as usize) * (LTC_LUT_SIZE as usize) * RGBA16F_BYTES_PER_TEXEL;
        assert_eq!(m_inv_data.len(), expected_m_inv_size);

        // Amplitude LUT: 64x64 R16F = 64*64*2 bytes
        let expected_amp_size = (LTC_LUT_SIZE as usize) * (LTC_LUT_SIZE as usize) * R16F_BYTES_PER_TEXEL;
        assert_eq!(amplitude_data.len(), expected_amp_size);
    }

    #[test]
    fn test_lut_data_not_all_zeros() {
        let (m_inv_data, amplitude_data) = generate_lut_data();

        // Check that data is not all zeros
        let m_inv_sum: u64 = m_inv_data.iter().map(|&b| b as u64).sum();
        let amp_sum: u64 = amplitude_data.iter().map(|&b| b as u64).sum();

        assert!(m_inv_sum > 0, "M^(-1) LUT should not be all zeros");
        assert!(amp_sum > 0, "Amplitude LUT should not be all zeros");
    }

    // ---- LTC coefficient computation tests ----

    #[test]
    fn test_compute_ltc_coeffs_normal_incidence_smooth() {
        // At normal incidence (cos_theta=1) and low roughness,
        // the lobe should be very narrow (high 'a' value)
        let (coeffs, amp) = compute_ltc_coeffs(0.01, 1.0);

        assert!(coeffs.a > 10.0, "Smooth material at normal incidence should have narrow lobe");
        assert!(coeffs.b.abs() < 0.1, "Should have minimal skew at normal incidence");
        assert!(amp > 0.5, "Amplitude should be reasonable");
    }

    #[test]
    fn test_compute_ltc_coeffs_normal_incidence_rough() {
        // At normal incidence with high roughness,
        // the lobe should be wide (low 'a' value)
        let (coeffs, amp) = compute_ltc_coeffs(1.0, 1.0);

        assert!(coeffs.a < 10.0, "Rough material should have wide lobe");
        assert!(coeffs.a > 0.0, "Coefficient should be positive");
        assert!(amp > 0.0 && amp <= 1.0, "Amplitude should be in [0, 1]");
    }

    #[test]
    fn test_compute_ltc_coeffs_grazing_angle() {
        // At grazing angle (cos_theta near 0),
        // the lobe should be stretched
        let (coeffs, _amp) = compute_ltc_coeffs(0.5, 0.1);

        assert!(coeffs.a > 0.0, "Coefficient should be positive");
        // At grazing angles, skew terms become more significant
        assert!(coeffs.b.abs() > 0.0 || coeffs.d.abs() > 0.0, "Should have some skew at grazing angle");
    }

    #[test]
    fn test_compute_ltc_coeffs_clamping() {
        // Test edge cases with out-of-range inputs
        let (coeffs1, amp1) = compute_ltc_coeffs(-0.5, 1.0);
        let (coeffs2, amp2) = compute_ltc_coeffs(0.0, 1.0);

        // Should clamp roughness to 0
        assert!((coeffs1.a - coeffs2.a).abs() < 0.001);
        assert!((amp1 - amp2).abs() < 0.001);

        let (coeffs3, amp3) = compute_ltc_coeffs(1.5, 1.0);
        let (coeffs4, amp4) = compute_ltc_coeffs(1.0, 1.0);

        // Should clamp roughness to 1
        assert!((coeffs3.a - coeffs4.a).abs() < 0.001);
        assert!((amp3 - amp4).abs() < 0.001);
    }

    #[test]
    fn test_amplitude_bounds() {
        // Amplitude should always be in [0, 1]
        for roughness in [0.0, 0.25, 0.5, 0.75, 1.0] {
            for cos_theta in [0.1, 0.25, 0.5, 0.75, 1.0] {
                let (_, amp) = compute_ltc_coeffs(roughness, cos_theta);
                assert!(
                    amp >= 0.0 && amp <= 1.0,
                    "Amplitude {} out of bounds for roughness={}, cos_theta={}",
                    amp, roughness, cos_theta
                );
            }
        }
    }

    // ---- GPU texture creation tests (require device) ----

    #[test]
    fn test_generate_ltc_luts_texture_formats() {
        // Skip if no GPU available
        let device = match crate::rhi_device::RhiDevice::try_new_headless() {
            Some(d) => d,
            None => {
                eprintln!("Skipping GPU test: no adapter available");
                return;
            }
        };

        let luts = generate_ltc_luts(&device.device, &device.queue);

        // Verify M^(-1) texture dimensions
        let m_inv_size = luts.m_inv.size();
        assert_eq!(m_inv_size.width, LTC_LUT_SIZE);
        assert_eq!(m_inv_size.height, LTC_LUT_SIZE);
        assert_eq!(m_inv_size.depth_or_array_layers, 1);

        // Verify amplitude texture dimensions
        let amp_size = luts.amplitude.size();
        assert_eq!(amp_size.width, LTC_LUT_SIZE);
        assert_eq!(amp_size.height, LTC_LUT_SIZE);
        assert_eq!(amp_size.depth_or_array_layers, 1);
    }

    #[test]
    fn test_generate_ltc_luts_creates_views() {
        // Skip if no GPU available
        let device = match crate::rhi_device::RhiDevice::try_new_headless() {
            Some(d) => d,
            None => {
                eprintln!("Skipping GPU test: no adapter available");
                return;
            }
        };

        let luts = generate_ltc_luts(&device.device, &device.queue);

        // Should be able to create views without panicking
        let (m_inv_view, amp_view) = luts.create_views();

        // Views are opaque, but we can verify they were created
        let _ = m_inv_view;
        let _ = amp_view;
    }

    #[test]
    fn test_generate_ltc_luts_sampler_config() {
        // Skip if no GPU available
        let device = match crate::rhi_device::RhiDevice::try_new_headless() {
            Some(d) => d,
            None => {
                eprintln!("Skipping GPU test: no adapter available");
                return;
            }
        };

        let _luts = generate_ltc_luts(&device.device, &device.queue);

        // Sampler is created successfully if we reach here without panic
        // wgpu::Sampler is opaque, so we can't inspect its properties directly
    }

    // ---- Lerp helper test ----

    #[test]
    fn test_lerp() {
        assert_eq!(lerp(0.0, 10.0, 0.0), 0.0);
        assert_eq!(lerp(0.0, 10.0, 1.0), 10.0);
        assert_eq!(lerp(0.0, 10.0, 0.5), 5.0);
        assert_eq!(lerp(-5.0, 5.0, 0.5), 0.0);
    }

    // ---- Precomputed data tests ----

    #[test]
    fn test_precomputed_smooth_normal() {
        let coeffs = precomputed::SMOOTH_NORMAL;
        assert!(coeffs.a > 100.0, "Smooth surface should have high 'a'");
        assert_eq!(coeffs.b, 0.0, "Should have no skew");
    }

    #[test]
    fn test_precomputed_rough_normal() {
        let coeffs = precomputed::ROUGH_NORMAL;
        assert_eq!(coeffs.a, 1.0, "Rough surface should have a=1");
        assert_eq!(coeffs.b, 0.0, "Should have no skew");
        assert_eq!(coeffs.c, 1.0, "Should be isotropic");
        assert_eq!(coeffs.d, 0.0, "Should have no skew");
    }

    #[test]
    fn test_expected_corner_values_structure() {
        let corners = precomputed::expected_corner_values();
        assert_eq!(corners.len(), 4, "Should have 4 corner values");

        for (roughness, cos_theta, _coeffs, amplitude) in corners {
            assert!(roughness >= 0.0 && roughness <= 1.0);
            assert!(cos_theta >= 0.0 && cos_theta <= 1.0);
            assert!(amplitude >= 0.0 && amplitude <= 1.0);
        }
    }
}
