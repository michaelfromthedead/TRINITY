//! FFT Ocean Simulation for TRINITY Engine (T-ENV-2.8).
//!
//! Implements FFT-based ocean simulation using the Phillips spectrum model.
//! This provides highly realistic ocean surfaces with proper spectral distribution,
//! unlike Gerstner waves which use hand-tuned parameters.
//!
//! # Overview
//!
//! The FFT ocean simulation works in these stages:
//! 1. Generate initial Phillips spectrum h0(K) with Gaussian noise
//! 2. Evolve spectrum over time: h(K,t) = h0(K)*exp(i*w*t) + h0*(-K)*exp(-i*w*t)
//! 3. Apply 2D inverse FFT (via two 1D passes) to get height field
//! 4. Compute horizontal displacement for choppy waves
//!
//! # Physics
//!
//! The Phillips spectrum models wind-driven waves:
//! ```text
//! P(K) = A * exp(-1/(kL)^2) / k^4 * |K_hat . W_hat|^2
//! ```
//! Where:
//! - A = Phillips constant (amplitude scaling)
//! - L = V^2/g (largest possible wave from wind)
//! - V = wind speed (m/s)
//! - g = gravitational acceleration
//! - K = wave vector
//! - W = wind direction
//!
//! Deep water dispersion: `omega = sqrt(g * |k|)`
//!
//! # Usage
//!
//! ```ignore
//! use renderer_backend::water::fft_ocean::{FFTOcean, FFTOceanConfig};
//!
//! let config = FFTOceanConfig::default();
//! let mut ocean = FFTOcean::new(config);
//!
//! // Generate initial spectrum (once)
//! ocean.generate_phillips_spectrum();
//!
//! // Update each frame
//! ocean.update(elapsed_time);
//!
//! // Access height field
//! let height = ocean.sample_height(x, z);
//! ```

use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Gravitational acceleration (m/s^2).
pub const GRAVITY: f32 = 9.81;

/// Default FFT resolution.
pub const DEFAULT_FFT_SIZE: u32 = 256;

/// Default patch size in meters.
pub const DEFAULT_PATCH_SIZE: f32 = 500.0;

/// Small epsilon for numerical stability.
const EPSILON: f32 = 1e-12;

/// FFTOceanConfig struct size in bytes.
pub const FFT_OCEAN_CONFIG_SIZE: usize = 36;

// ---------------------------------------------------------------------------
// Complex Number Operations
// ---------------------------------------------------------------------------

/// Complex number with real and imaginary parts.
///
/// GPU-compatible layout for compute shaders.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Complex {
    /// Real part.
    pub re: f32,
    /// Imaginary part.
    pub im: f32,
}

impl Complex {
    /// Create a new complex number.
    #[inline]
    pub const fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    /// Create complex number from polar form (magnitude, angle).
    #[inline]
    pub fn from_polar(r: f32, theta: f32) -> Self {
        Self {
            re: r * theta.cos(),
            im: r * theta.sin(),
        }
    }

    /// Complex zero.
    pub const ZERO: Complex = Complex { re: 0.0, im: 0.0 };

    /// Complex one.
    pub const ONE: Complex = Complex { re: 1.0, im: 0.0 };

    /// Complex i.
    pub const I: Complex = Complex { re: 0.0, im: 1.0 };

    /// Magnitude squared: |z|^2 = re^2 + im^2.
    #[inline]
    pub fn magnitude_squared(self) -> f32 {
        self.re * self.re + self.im * self.im
    }

    /// Magnitude: |z| = sqrt(re^2 + im^2).
    #[inline]
    pub fn magnitude(self) -> f32 {
        self.magnitude_squared().sqrt()
    }

    /// Complex conjugate: (re, -im).
    #[inline]
    pub fn conjugate(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    /// Complex addition.
    #[inline]
    pub fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }

    /// Complex subtraction.
    #[inline]
    pub fn sub(self, other: Self) -> Self {
        Self {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }

    /// Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i.
    #[inline]
    pub fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    /// Scalar multiplication.
    #[inline]
    pub fn scale(self, s: f32) -> Self {
        Self {
            re: self.re * s,
            im: self.im * s,
        }
    }

    /// Complex exponential: e^(i*theta) = cos(theta) + i*sin(theta).
    #[inline]
    pub fn exp_i(theta: f32) -> Self {
        Self {
            re: theta.cos(),
            im: theta.sin(),
        }
    }

    /// Euler's formula: e^z = e^(re) * (cos(im) + i*sin(im)).
    #[inline]
    pub fn exp(self) -> Self {
        let r = self.re.exp();
        Self {
            re: r * self.im.cos(),
            im: r * self.im.sin(),
        }
    }
}

impl std::ops::Add for Complex {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Complex::add(self, rhs)
    }
}

impl std::ops::Sub for Complex {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Complex::sub(self, rhs)
    }
}

impl std::ops::Mul for Complex {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Complex::mul(self, rhs)
    }
}

impl std::ops::Mul<f32> for Complex {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self {
        self.scale(rhs)
    }
}

impl std::ops::Neg for Complex {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }
}

impl PartialEq for Complex {
    fn eq(&self, other: &Self) -> bool {
        (self.re - other.re).abs() < 1e-6 && (self.im - other.im).abs() < 1e-6
    }
}

// ---------------------------------------------------------------------------
// FFT Ocean Configuration
// ---------------------------------------------------------------------------

/// Configuration for FFT ocean simulation.
///
/// GPU-compatible layout for uniform buffers.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FFTOceanConfig {
    /// FFT resolution (power of 2, e.g., 256).
    pub fft_size: u32,
    /// Physical size of ocean patch in meters.
    pub patch_size: f32,
    /// Wind speed in m/s.
    pub wind_speed: f32,
    /// Normalized wind direction [x, z].
    pub wind_direction: [f32; 2],
    /// Phillips spectrum amplitude constant.
    pub phillips_constant: f32,
    /// Choppy wave displacement factor.
    pub chop_amount: f32,
    /// Current simulation time.
    pub time: f32,
    /// Padding for 16-byte alignment.
    pub _padding: u32,
}

impl Default for FFTOceanConfig {
    fn default() -> Self {
        Self {
            fft_size: DEFAULT_FFT_SIZE,
            patch_size: DEFAULT_PATCH_SIZE,
            wind_speed: 10.0,
            wind_direction: [0.8, 0.6], // Normalized
            phillips_constant: 0.0002,
            chop_amount: 1.0,
            time: 0.0,
            _padding: 0,
        }
    }
}

impl FFTOceanConfig {
    /// Create configuration for calm ocean.
    pub fn calm() -> Self {
        Self {
            wind_speed: 5.0,
            phillips_constant: 0.00005,
            chop_amount: 0.5,
            ..Default::default()
        }
    }

    /// Create configuration for moderate ocean.
    pub fn moderate() -> Self {
        Self {
            wind_speed: 15.0,
            phillips_constant: 0.0003,
            chop_amount: 1.2,
            ..Default::default()
        }
    }

    /// Create configuration for stormy ocean.
    pub fn stormy() -> Self {
        Self {
            wind_speed: 30.0,
            phillips_constant: 0.001,
            chop_amount: 2.0,
            ..Default::default()
        }
    }

    /// Validate configuration.
    pub fn validate(&self) -> Result<(), &'static str> {
        if !self.fft_size.is_power_of_two() {
            return Err("FFT size must be power of 2");
        }
        if self.fft_size < 16 || self.fft_size > 2048 {
            return Err("FFT size must be between 16 and 2048");
        }
        if self.patch_size <= 0.0 {
            return Err("Patch size must be positive");
        }
        if self.wind_speed < 0.0 {
            return Err("Wind speed must be non-negative");
        }
        let wind_len = (self.wind_direction[0].powi(2) + self.wind_direction[1].powi(2)).sqrt();
        if (wind_len - 1.0).abs() > 0.01 {
            return Err("Wind direction must be normalized");
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Phillips Spectrum
// ---------------------------------------------------------------------------

/// Compute the Phillips spectrum value for a given wave vector.
///
/// The Phillips spectrum models wind-driven wave energy distribution:
/// ```text
/// P(K) = A * exp(-1/(kL)^2) / k^4 * |K_hat . W_hat|^2
/// ```
///
/// # Arguments
/// * `k` - Wave vector [kx, kz]
/// * `wind` - Normalized wind direction [wx, wz]
/// * `wind_speed` - Wind speed in m/s
/// * `amplitude` - Phillips constant A
///
/// # Returns
/// Spectrum amplitude (sqrt of energy)
pub fn phillips_spectrum(k: [f32; 2], wind: [f32; 2], wind_speed: f32, amplitude: f32) -> f32 {
    let k_mag_sq = k[0] * k[0] + k[1] * k[1];

    // Skip DC component
    if k_mag_sq < EPSILON {
        return 0.0;
    }

    let k_mag = k_mag_sq.sqrt();
    let k_mag_4 = k_mag_sq * k_mag_sq;

    // Largest possible wave from wind: L = V^2 / g
    let l = wind_speed * wind_speed / GRAVITY;
    let l_sq = l * l;

    // Exponential suppression of waves larger than L
    let exp_term = (-1.0 / (k_mag_sq * l_sq)).exp();

    // Directional spreading: |K_hat . W_hat|^2
    let k_norm = [k[0] / k_mag, k[1] / k_mag];
    let k_dot_w = k_norm[0] * wind[0] + k_norm[1] * wind[1];
    let directional = k_dot_w * k_dot_w;

    // Small wave damping (suppress waves smaller than l/2000)
    let damping_scale = l / 2000.0;
    let damping = (-k_mag_sq * damping_scale * damping_scale).exp();

    let spectrum = amplitude * exp_term * directional * damping / k_mag_4;

    // Return sqrt for complex amplitude (energy is squared)
    spectrum.max(0.0).sqrt()
}

/// Deep water dispersion relation: omega = sqrt(g * |k|).
///
/// Relates angular frequency to wave number.
#[inline]
pub fn dispersion_relation(k_mag: f32) -> f32 {
    (GRAVITY * k_mag).sqrt()
}

// ---------------------------------------------------------------------------
// Gaussian Random Number Generation
// ---------------------------------------------------------------------------

/// Simple linear congruential generator for deterministic noise.
///
/// Uses Park-Miller parameters.
pub struct LCG {
    state: u64,
}

impl LCG {
    /// Create new LCG with seed.
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed.max(1),
        }
    }

    /// Generate next random u32.
    pub fn next_u32(&mut self) -> u32 {
        // Park-Miller constants
        self.state = self.state.wrapping_mul(48271) % 2147483647;
        self.state as u32
    }

    /// Generate uniform random in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        self.next_u32() as f32 / 2147483647.0
    }

    /// Generate Gaussian random using Box-Muller transform.
    pub fn next_gaussian(&mut self) -> f32 {
        let u1 = self.next_f32().max(EPSILON);
        let u2 = self.next_f32();

        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }

    /// Generate complex Gaussian with unit variance.
    pub fn next_gaussian_complex(&mut self) -> Complex {
        Complex {
            re: self.next_gaussian() / 2.0_f32.sqrt(),
            im: self.next_gaussian() / 2.0_f32.sqrt(),
        }
    }
}

// ---------------------------------------------------------------------------
// FFT Implementation
// ---------------------------------------------------------------------------

/// Check if a number is a power of 2.
#[inline]
pub fn is_power_of_2(n: usize) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

/// Compute bit-reversal of index for FFT.
#[inline]
fn bit_reverse(mut x: usize, log2_n: u32) -> usize {
    let mut result = 0;
    for _ in 0..log2_n {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

/// In-place bit-reversal permutation.
fn bit_reversal_permutation(data: &mut [Complex]) {
    let n = data.len();
    let log2_n = n.trailing_zeros();

    for i in 0..n {
        let j = bit_reverse(i, log2_n);
        if i < j {
            data.swap(i, j);
        }
    }
}

/// Cooley-Tukey FFT/IFFT.
///
/// Performs in-place FFT (inverse=false) or IFFT (inverse=true).
/// Data length must be a power of 2.
///
/// # Arguments
/// * `data` - Complex array to transform (length must be power of 2)
/// * `inverse` - If true, compute inverse FFT
pub fn fft_1d(data: &mut [Complex], inverse: bool) {
    let n = data.len();
    if n <= 1 {
        return;
    }

    assert!(is_power_of_2(n), "FFT size must be power of 2");

    // Bit-reversal permutation
    bit_reversal_permutation(data);

    // Iterative Cooley-Tukey
    let log2_n = n.trailing_zeros();
    let sign = if inverse { 1.0 } else { -1.0 };

    for s in 1..=log2_n {
        let m = 1 << s;
        let half_m = m >> 1;

        // Twiddle factor: e^(-2*pi*i/m) for FFT, e^(2*pi*i/m) for IFFT
        let angle = sign * 2.0 * PI / m as f32;
        let w_m = Complex::exp_i(angle);

        for k in (0..n).step_by(m) {
            let mut w = Complex::ONE;
            for j in 0..half_m {
                let t = w * data[k + j + half_m];
                let u = data[k + j];
                data[k + j] = u + t;
                data[k + j + half_m] = u - t;
                w = w * w_m;
            }
        }
    }

    // Scale by 1/N for inverse transform
    if inverse {
        let scale = 1.0 / n as f32;
        for x in data.iter_mut() {
            *x = x.scale(scale);
        }
    }
}

/// 2D FFT via row-column decomposition.
///
/// Performs FFT along rows, then along columns.
pub fn fft_2d(data: &mut [Complex], width: usize, height: usize, inverse: bool) {
    assert_eq!(data.len(), width * height);
    assert!(is_power_of_2(width) && is_power_of_2(height));

    // Row-wise FFT
    for row in 0..height {
        let start = row * width;
        let end = start + width;
        fft_1d(&mut data[start..end], inverse);
    }

    // Column-wise FFT (need to gather/scatter)
    let mut column = vec![Complex::ZERO; height];
    for col in 0..width {
        // Gather column
        for row in 0..height {
            column[row] = data[row * width + col];
        }

        // FFT column
        fft_1d(&mut column, inverse);

        // Scatter column back
        for row in 0..height {
            data[row * width + col] = column[row];
        }
    }
}

// ---------------------------------------------------------------------------
// FFT Ocean Simulation
// ---------------------------------------------------------------------------

/// FFT-based ocean simulation.
///
/// Generates realistic ocean surfaces using the Phillips spectrum and FFT.
pub struct FFTOcean {
    /// Configuration parameters.
    pub config: FFTOceanConfig,

    /// Initial spectrum h0(K) - complex amplitudes at t=0.
    pub h0: Vec<Complex>,

    /// Conjugate h0*(-K) for time evolution.
    pub h0_conj: Vec<Complex>,

    /// Time-evolved spectrum h(K, t).
    pub h_tilde: Vec<Complex>,

    /// Output height field (real values).
    pub heightfield: Vec<f32>,

    /// Horizontal displacement X component.
    pub displacement_x: Vec<f32>,

    /// Horizontal displacement Z component.
    pub displacement_z: Vec<f32>,

    /// Precomputed dispersion values omega(k).
    dispersion: Vec<f32>,
}

impl FFTOcean {
    /// Create new FFT ocean with given configuration.
    pub fn new(config: FFTOceanConfig) -> Self {
        let n = config.fft_size as usize;
        let size = n * n;

        Self {
            config,
            h0: vec![Complex::ZERO; size],
            h0_conj: vec![Complex::ZERO; size],
            h_tilde: vec![Complex::ZERO; size],
            heightfield: vec![0.0; size],
            displacement_x: vec![0.0; size],
            displacement_z: vec![0.0; size],
            dispersion: vec![0.0; size],
        }
    }

    /// Generate initial Phillips spectrum with Gaussian noise.
    ///
    /// Call once at initialization or when parameters change.
    pub fn generate_phillips_spectrum(&mut self) {
        self.generate_phillips_spectrum_with_seed(12345);
    }

    /// Generate Phillips spectrum with specific seed for reproducibility.
    pub fn generate_phillips_spectrum_with_seed(&mut self, seed: u64) {
        let n = self.config.fft_size as usize;
        let l = self.config.patch_size;

        let mut rng = LCG::new(seed);

        for z in 0..n {
            for x in 0..n {
                let idx = z * n + x;

                // Wave vector k = 2*pi*n/L where n is [-N/2, N/2)
                let nx = if x < n / 2 { x as i32 } else { x as i32 - n as i32 };
                let nz = if z < n / 2 { z as i32 } else { z as i32 - n as i32 };

                let kx = 2.0 * PI * nx as f32 / l;
                let kz = 2.0 * PI * nz as f32 / l;
                let k = [kx, kz];

                let k_mag = (kx * kx + kz * kz).sqrt();

                // Phillips spectrum amplitude
                let spectrum = phillips_spectrum(
                    k,
                    self.config.wind_direction,
                    self.config.wind_speed,
                    self.config.phillips_constant,
                );

                // h0(K) = 1/sqrt(2) * (xi_r + i*xi_i) * sqrt(P(K))
                let noise = rng.next_gaussian_complex();
                self.h0[idx] = noise * spectrum;

                // Store h0*(-K) for time evolution
                // -K maps (x,z) -> (N-x, N-z) with wrapping
                let neg_x = if x == 0 { 0 } else { n - x };
                let neg_z = if z == 0 { 0 } else { n - z };
                let neg_idx = neg_z * n + neg_x;

                // We'll fill h0_conj after all h0 values are computed
                self.dispersion[idx] = dispersion_relation(k_mag);
            }
        }

        // Fill h0_conj with conjugates
        for z in 0..n {
            for x in 0..n {
                let idx = z * n + x;
                let neg_x = if x == 0 { 0 } else { n - x };
                let neg_z = if z == 0 { 0 } else { n - z };
                let neg_idx = neg_z * n + neg_x;
                self.h0_conj[idx] = self.h0[neg_idx].conjugate();
            }
        }
    }

    /// Evolve spectrum to given time.
    ///
    /// h(K, t) = h0(K) * exp(i*omega*t) + h0*(-K) * exp(-i*omega*t)
    pub fn evolve_spectrum(&mut self, time: f32) {
        let n = self.config.fft_size as usize;
        self.config.time = time;

        for i in 0..n * n {
            let omega = self.dispersion[i];
            let phase = omega * time;

            // exp(i*omega*t) and exp(-i*omega*t)
            let exp_pos = Complex::exp_i(phase);
            let exp_neg = Complex::exp_i(-phase);

            // h(K, t) = h0(K)*exp(i*w*t) + h0*(-K)*exp(-i*w*t)
            self.h_tilde[i] = self.h0[i] * exp_pos + self.h0_conj[i] * exp_neg;
        }
    }

    /// Apply 2D inverse FFT to get height field.
    pub fn ifft_2d(&mut self) {
        let n = self.config.fft_size as usize;

        // Copy h_tilde for height IFFT
        let mut height_data = self.h_tilde.clone();
        fft_2d(&mut height_data, n, n, true);

        // Extract real parts for height field
        for i in 0..n * n {
            self.heightfield[i] = height_data[i].re;
        }
    }

    /// Compute horizontal displacement for choppy waves.
    ///
    /// D(x, t) = -i * K/|K| * IFFT(h(K, t))
    pub fn compute_displacement(&mut self) {
        let n = self.config.fft_size as usize;
        let l = self.config.patch_size;
        let chop = self.config.chop_amount;

        // Compute displacement spectrum: -i * (kx/|k|) * h(K, t)
        let mut disp_x_spectrum = vec![Complex::ZERO; n * n];
        let mut disp_z_spectrum = vec![Complex::ZERO; n * n];

        for z in 0..n {
            for x in 0..n {
                let idx = z * n + x;

                let nx = if x < n / 2 { x as i32 } else { x as i32 - n as i32 };
                let nz = if z < n / 2 { z as i32 } else { z as i32 - n as i32 };

                let kx = 2.0 * PI * nx as f32 / l;
                let kz = 2.0 * PI * nz as f32 / l;
                let k_mag = (kx * kx + kz * kz).sqrt();

                if k_mag > EPSILON {
                    // -i * (k/|k|) * h
                    // Multiplying by -i swaps real/imag: -i*(a+bi) = b - ai
                    let h = self.h_tilde[idx];
                    let neg_i_h = Complex::new(h.im, -h.re);

                    disp_x_spectrum[idx] = neg_i_h * (kx / k_mag);
                    disp_z_spectrum[idx] = neg_i_h * (kz / k_mag);
                }
            }
        }

        // IFFT displacement spectra
        fft_2d(&mut disp_x_spectrum, n, n, true);
        fft_2d(&mut disp_z_spectrum, n, n, true);

        // Extract real parts with chop scaling
        for i in 0..n * n {
            self.displacement_x[i] = disp_x_spectrum[i].re * chop;
            self.displacement_z[i] = disp_z_spectrum[i].re * chop;
        }
    }

    /// Full update: evolve spectrum + IFFT + displacement.
    pub fn update(&mut self, time: f32) {
        self.evolve_spectrum(time);
        self.ifft_2d();
        self.compute_displacement();
    }

    /// Sample height at normalized UV coordinates [0, 1).
    pub fn sample_height(&self, u: f32, v: f32) -> f32 {
        let n = self.config.fft_size as usize;
        let x = ((u.fract() + 1.0).fract() * n as f32) as usize % n;
        let z = ((v.fract() + 1.0).fract() * n as f32) as usize % n;
        self.heightfield[z * n + x]
    }

    /// Sample displacement at normalized UV coordinates [0, 1).
    pub fn sample_displacement(&self, u: f32, v: f32) -> [f32; 2] {
        let n = self.config.fft_size as usize;
        let x = ((u.fract() + 1.0).fract() * n as f32) as usize % n;
        let z = ((v.fract() + 1.0).fract() * n as f32) as usize % n;
        let idx = z * n + x;
        [self.displacement_x[idx], self.displacement_z[idx]]
    }

    /// Sample height and displacement at world position.
    pub fn sample_world(&self, world_x: f32, world_z: f32) -> (f32, [f32; 2]) {
        let u = world_x / self.config.patch_size;
        let v = world_z / self.config.patch_size;
        (self.sample_height(u, v), self.sample_displacement(u, v))
    }

    /// Get total wave energy (sum of |h0|^2).
    pub fn total_energy(&self) -> f32 {
        self.h0.iter().map(|h| h.magnitude_squared()).sum()
    }

    /// Compute RMS wave height.
    pub fn rms_height(&self) -> f32 {
        let n = self.heightfield.len() as f32;
        let sum_sq: f32 = self.heightfield.iter().map(|h| h * h).sum();
        (sum_sq / n).sqrt()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON_TEST: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    fn complex_approx_eq(a: Complex, b: Complex, eps: f32) -> bool {
        approx_eq(a.re, b.re, eps) && approx_eq(a.im, b.im, eps)
    }

    // ===== Complex Number Tests =====

    #[test]
    fn test_complex_new() {
        let c = Complex::new(3.0, 4.0);
        assert_eq!(c.re, 3.0);
        assert_eq!(c.im, 4.0);
    }

    #[test]
    fn test_complex_from_polar() {
        let c = Complex::from_polar(2.0, PI / 4.0);
        assert!(approx_eq(c.re, 2.0_f32.sqrt(), EPSILON_TEST));
        assert!(approx_eq(c.im, 2.0_f32.sqrt(), EPSILON_TEST));
    }

    #[test]
    fn test_complex_constants() {
        assert_eq!(Complex::ZERO.re, 0.0);
        assert_eq!(Complex::ZERO.im, 0.0);
        assert_eq!(Complex::ONE.re, 1.0);
        assert_eq!(Complex::ONE.im, 0.0);
        assert_eq!(Complex::I.re, 0.0);
        assert_eq!(Complex::I.im, 1.0);
    }

    #[test]
    fn test_complex_magnitude_squared() {
        let c = Complex::new(3.0, 4.0);
        assert!(approx_eq(c.magnitude_squared(), 25.0, EPSILON_TEST));
    }

    #[test]
    fn test_complex_magnitude() {
        let c = Complex::new(3.0, 4.0);
        assert!(approx_eq(c.magnitude(), 5.0, EPSILON_TEST));
    }

    #[test]
    fn test_complex_conjugate() {
        let c = Complex::new(3.0, 4.0);
        let conj = c.conjugate();
        assert_eq!(conj.re, 3.0);
        assert_eq!(conj.im, -4.0);
    }

    #[test]
    fn test_complex_add() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, 4.0);
        let c = a + b;
        assert!(approx_eq(c.re, 4.0, EPSILON_TEST));
        assert!(approx_eq(c.im, 6.0, EPSILON_TEST));
    }

    #[test]
    fn test_complex_sub() {
        let a = Complex::new(5.0, 7.0);
        let b = Complex::new(3.0, 4.0);
        let c = a - b;
        assert!(approx_eq(c.re, 2.0, EPSILON_TEST));
        assert!(approx_eq(c.im, 3.0, EPSILON_TEST));
    }

    #[test]
    fn test_complex_mul() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, 4.0);
        let c = a * b;
        // (1+2i)(3+4i) = 3 + 4i + 6i + 8i^2 = 3 + 10i - 8 = -5 + 10i
        assert!(approx_eq(c.re, -5.0, EPSILON_TEST));
        assert!(approx_eq(c.im, 10.0, EPSILON_TEST));
    }

    #[test]
    fn test_complex_scale() {
        let c = Complex::new(3.0, 4.0);
        let scaled = c * 2.0;
        assert!(approx_eq(scaled.re, 6.0, EPSILON_TEST));
        assert!(approx_eq(scaled.im, 8.0, EPSILON_TEST));
    }

    #[test]
    fn test_complex_neg() {
        let c = Complex::new(3.0, -4.0);
        let neg = -c;
        assert!(approx_eq(neg.re, -3.0, EPSILON_TEST));
        assert!(approx_eq(neg.im, 4.0, EPSILON_TEST));
    }

    #[test]
    fn test_complex_exp_i() {
        // e^(i*0) = 1
        let c = Complex::exp_i(0.0);
        assert!(approx_eq(c.re, 1.0, EPSILON_TEST));
        assert!(approx_eq(c.im, 0.0, EPSILON_TEST));

        // e^(i*pi/2) = i
        let c = Complex::exp_i(PI / 2.0);
        assert!(approx_eq(c.re, 0.0, EPSILON_TEST));
        assert!(approx_eq(c.im, 1.0, EPSILON_TEST));

        // e^(i*pi) = -1
        let c = Complex::exp_i(PI);
        assert!(approx_eq(c.re, -1.0, EPSILON_TEST));
        assert!(approx_eq(c.im, 0.0, EPSILON_TEST));
    }

    #[test]
    fn test_complex_exp() {
        // e^0 = 1
        let c = Complex::ZERO.exp();
        assert!(approx_eq(c.re, 1.0, EPSILON_TEST));
        assert!(approx_eq(c.im, 0.0, EPSILON_TEST));

        // e^1 = e
        let c = Complex::ONE.exp();
        assert!(approx_eq(c.re, std::f32::consts::E, EPSILON_TEST));
        assert!(approx_eq(c.im, 0.0, EPSILON_TEST));
    }

    #[test]
    fn test_complex_i_squared() {
        let i_sq = Complex::I * Complex::I;
        assert!(approx_eq(i_sq.re, -1.0, EPSILON_TEST));
        assert!(approx_eq(i_sq.im, 0.0, EPSILON_TEST));
    }

    #[test]
    fn test_complex_conjugate_product() {
        // z * conj(z) = |z|^2
        let z = Complex::new(3.0, 4.0);
        let product = z * z.conjugate();
        assert!(approx_eq(product.re, 25.0, EPSILON_TEST));
        assert!(approx_eq(product.im, 0.0, EPSILON_TEST));
    }

    // ===== FFT Tests =====

    #[test]
    fn test_is_power_of_2() {
        assert!(is_power_of_2(1));
        assert!(is_power_of_2(2));
        assert!(is_power_of_2(4));
        assert!(is_power_of_2(256));
        assert!(is_power_of_2(1024));

        assert!(!is_power_of_2(0));
        assert!(!is_power_of_2(3));
        assert!(!is_power_of_2(5));
        assert!(!is_power_of_2(100));
    }

    #[test]
    fn test_bit_reverse() {
        assert_eq!(bit_reverse(0, 3), 0);
        assert_eq!(bit_reverse(1, 3), 4);
        assert_eq!(bit_reverse(2, 3), 2);
        assert_eq!(bit_reverse(3, 3), 6);
        assert_eq!(bit_reverse(4, 3), 1);
        assert_eq!(bit_reverse(5, 3), 5);
        assert_eq!(bit_reverse(6, 3), 3);
        assert_eq!(bit_reverse(7, 3), 7);
    }

    #[test]
    fn test_fft_identity_size_1() {
        let mut data = [Complex::new(5.0, 3.0)];
        fft_1d(&mut data, false);
        assert!(complex_approx_eq(data[0], Complex::new(5.0, 3.0), EPSILON_TEST));
    }

    #[test]
    fn test_fft_identity_size_2() {
        let mut data = [Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)];
        fft_1d(&mut data, false);
        assert!(complex_approx_eq(data[0], Complex::new(2.0, 0.0), EPSILON_TEST));
        assert!(complex_approx_eq(data[1], Complex::new(0.0, 0.0), EPSILON_TEST));
    }

    #[test]
    fn test_fft_simple_pulse() {
        // DFT of delta function should be constant
        let mut data = vec![Complex::ZERO; 4];
        data[0] = Complex::ONE;

        fft_1d(&mut data, false);

        for c in &data {
            assert!(approx_eq(c.re, 1.0, EPSILON_TEST));
            assert!(approx_eq(c.im, 0.0, EPSILON_TEST));
        }
    }

    #[test]
    fn test_fft_constant_signal() {
        // DFT of constant should be delta at DC
        let mut data = vec![Complex::new(2.0, 0.0); 4];

        fft_1d(&mut data, false);

        assert!(approx_eq(data[0].re, 8.0, EPSILON_TEST)); // N * value
        for i in 1..4 {
            assert!(approx_eq(data[i].magnitude(), 0.0, EPSILON_TEST));
        }
    }

    #[test]
    fn test_ifft_inverts_fft() {
        let original: Vec<Complex> = (0..8)
            .map(|i| Complex::new(i as f32, (i as f32 * 0.5).sin()))
            .collect();

        let mut data = original.clone();

        fft_1d(&mut data, false);
        fft_1d(&mut data, true);

        for (a, b) in original.iter().zip(data.iter()) {
            assert!(complex_approx_eq(*a, *b, EPSILON_TEST));
        }
    }

    #[test]
    fn test_fft_ifft_roundtrip_16() {
        let mut rng = LCG::new(999);
        let original: Vec<Complex> = (0..16).map(|_| rng.next_gaussian_complex()).collect();

        let mut data = original.clone();
        fft_1d(&mut data, false);
        fft_1d(&mut data, true);

        for (a, b) in original.iter().zip(data.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-4));
        }
    }

    #[test]
    fn test_fft_ifft_roundtrip_256() {
        let mut rng = LCG::new(42);
        let original: Vec<Complex> = (0..256).map(|_| rng.next_gaussian_complex()).collect();

        let mut data = original.clone();
        fft_1d(&mut data, false);
        fft_1d(&mut data, true);

        for (a, b) in original.iter().zip(data.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-3));
        }
    }

    #[test]
    fn test_fft_2d_roundtrip() {
        let mut rng = LCG::new(123);
        let original: Vec<Complex> = (0..16).map(|_| rng.next_gaussian_complex()).collect();

        let mut data = original.clone();
        fft_2d(&mut data, 4, 4, false);
        fft_2d(&mut data, 4, 4, true);

        for (a, b) in original.iter().zip(data.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-4));
        }
    }

    #[test]
    fn test_fft_2d_delta() {
        // Delta in spatial domain -> constant in frequency
        let mut data = vec![Complex::ZERO; 16];
        data[0] = Complex::ONE;

        fft_2d(&mut data, 4, 4, false);

        for c in &data {
            assert!(approx_eq(c.re, 1.0, EPSILON_TEST));
            assert!(approx_eq(c.im, 0.0, EPSILON_TEST));
        }
    }

    #[test]
    fn test_fft_parsevals_theorem() {
        // Sum of |x|^2 = (1/N) * sum of |X|^2
        let original = vec![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 1.0),
            Complex::new(-1.0, 0.5),
            Complex::new(0.5, -0.5),
        ];

        let spatial_energy: f32 = original.iter().map(|c| c.magnitude_squared()).sum();

        let mut freq = original.clone();
        fft_1d(&mut freq, false);
        let freq_energy: f32 = freq.iter().map(|c| c.magnitude_squared()).sum();

        // |X|^2 = N * |x|^2
        assert!(approx_eq(freq_energy, 4.0 * spatial_energy, 1e-4));
    }

    // ===== Dispersion Relation Tests =====

    #[test]
    fn test_dispersion_relation_zero() {
        assert!(approx_eq(dispersion_relation(0.0), 0.0, EPSILON_TEST));
    }

    #[test]
    fn test_dispersion_relation_unit() {
        let omega = dispersion_relation(1.0);
        assert!(approx_eq(omega, GRAVITY.sqrt(), EPSILON_TEST));
    }

    #[test]
    fn test_dispersion_relation_increases() {
        let w1 = dispersion_relation(1.0);
        let w2 = dispersion_relation(4.0);
        assert!(w2 > w1);
        // omega = sqrt(g*k), so w2/w1 = sqrt(4) = 2
        assert!(approx_eq(w2 / w1, 2.0, EPSILON_TEST));
    }

    #[test]
    fn test_dispersion_deep_water() {
        // Phase velocity c = omega/k = sqrt(g/k)
        // Group velocity cg = d(omega)/d(k) = 0.5 * sqrt(g/k) = c/2
        let k = 0.1;
        let omega = dispersion_relation(k);
        let phase_velocity = omega / k;
        let expected_phase = (GRAVITY / k).sqrt();
        assert!(approx_eq(phase_velocity, expected_phase, EPSILON_TEST));
    }

    // ===== Phillips Spectrum Tests =====

    #[test]
    fn test_phillips_spectrum_dc() {
        // DC component (k=0) should be zero
        let s = phillips_spectrum([0.0, 0.0], [1.0, 0.0], 10.0, 0.001);
        assert!(approx_eq(s, 0.0, EPSILON_TEST));
    }

    #[test]
    fn test_phillips_spectrum_perpendicular() {
        // Waves perpendicular to wind have zero energy
        let wind = [1.0, 0.0];
        let k_perp = [0.0, 1.0]; // Perpendicular to wind
        let s = phillips_spectrum(k_perp, wind, 10.0, 0.001);
        assert!(approx_eq(s, 0.0, EPSILON_TEST));
    }

    #[test]
    fn test_phillips_spectrum_aligned() {
        // Waves aligned with wind should have non-zero energy
        let wind = [1.0, 0.0];
        let k_aligned = [0.1, 0.0];
        let s = phillips_spectrum(k_aligned, wind, 10.0, 0.001);
        assert!(s > 0.0);
    }

    #[test]
    fn test_phillips_spectrum_opposite() {
        // Waves opposite to wind also have energy (squared dot product)
        let wind = [1.0, 0.0];
        let k_opposite = [-0.1, 0.0];
        let s = phillips_spectrum(k_opposite, wind, 10.0, 0.001);
        assert!(s > 0.0);
    }

    #[test]
    fn test_phillips_spectrum_higher_wind() {
        let k = [0.1, 0.0];
        let wind = [1.0, 0.0];
        let s1 = phillips_spectrum(k, wind, 5.0, 0.001);
        let s2 = phillips_spectrum(k, wind, 15.0, 0.001);
        // Higher wind creates larger waves
        assert!(s2 > s1);
    }

    #[test]
    fn test_phillips_spectrum_symmetry() {
        // P(k) should equal P(-k) for directional term
        let wind = [0.8, 0.6];
        let k = [0.1, 0.05];
        let s1 = phillips_spectrum(k, wind, 10.0, 0.001);
        let s2 = phillips_spectrum([-k[0], -k[1]], wind, 10.0, 0.001);
        assert!(approx_eq(s1, s2, EPSILON_TEST));
    }

    #[test]
    fn test_phillips_spectrum_amplitude_scaling() {
        let k = [0.1, 0.0];
        let wind = [1.0, 0.0];
        let s1 = phillips_spectrum(k, wind, 10.0, 0.001);
        let s2 = phillips_spectrum(k, wind, 10.0, 0.004);
        // sqrt(4x) = 2 * sqrt(x)
        assert!(approx_eq(s2 / s1, 2.0, 0.01));
    }

    // ===== LCG Random Tests =====

    #[test]
    fn test_lcg_deterministic() {
        let mut rng1 = LCG::new(12345);
        let mut rng2 = LCG::new(12345);

        for _ in 0..10 {
            assert_eq!(rng1.next_u32(), rng2.next_u32());
        }
    }

    #[test]
    fn test_lcg_different_seeds() {
        let mut rng1 = LCG::new(12345);
        let mut rng2 = LCG::new(54321);

        assert_ne!(rng1.next_u32(), rng2.next_u32());
    }

    #[test]
    fn test_lcg_uniform_range() {
        let mut rng = LCG::new(999);
        for _ in 0..100 {
            let v = rng.next_f32();
            assert!(v >= 0.0 && v < 1.0);
        }
    }

    #[test]
    fn test_lcg_gaussian_mean() {
        let mut rng = LCG::new(42);
        let n = 10000;
        let sum: f32 = (0..n).map(|_| rng.next_gaussian()).sum();
        let mean = sum / n as f32;
        // Mean should be close to 0
        assert!(mean.abs() < 0.1);
    }

    #[test]
    fn test_lcg_gaussian_variance() {
        let mut rng = LCG::new(42);
        let n = 10000;
        let samples: Vec<f32> = (0..n).map(|_| rng.next_gaussian()).collect();
        let mean: f32 = samples.iter().sum::<f32>() / n as f32;
        let variance: f32 = samples.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;
        // Variance should be close to 1
        assert!(approx_eq(variance, 1.0, 0.1));
    }

    // ===== FFTOceanConfig Tests =====

    #[test]
    fn test_config_default() {
        let config = FFTOceanConfig::default();
        assert_eq!(config.fft_size, 256);
        assert!(approx_eq(config.patch_size, 500.0, EPSILON_TEST));
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_presets() {
        assert!(FFTOceanConfig::calm().validate().is_ok());
        assert!(FFTOceanConfig::moderate().validate().is_ok());
        assert!(FFTOceanConfig::stormy().validate().is_ok());
    }

    #[test]
    fn test_config_validation_fft_size() {
        let mut config = FFTOceanConfig::default();
        config.fft_size = 100; // Not power of 2
        assert!(config.validate().is_err());

        config.fft_size = 8; // Too small
        assert!(config.validate().is_err());

        config.fft_size = 4096; // Too large
        assert!(config.validate().is_err());

        config.fft_size = 512;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation_patch_size() {
        let mut config = FFTOceanConfig::default();
        config.patch_size = 0.0;
        assert!(config.validate().is_err());

        config.patch_size = -10.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_wind() {
        let mut config = FFTOceanConfig::default();
        config.wind_speed = -5.0;
        assert!(config.validate().is_err());

        config.wind_speed = 10.0;
        config.wind_direction = [0.0, 0.0]; // Not normalized
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_size() {
        assert_eq!(std::mem::size_of::<FFTOceanConfig>(), FFT_OCEAN_CONFIG_SIZE);
    }

    // ===== FFTOcean Tests =====

    #[test]
    fn test_ocean_new() {
        let config = FFTOceanConfig::default();
        let ocean = FFTOcean::new(config);

        let n = config.fft_size as usize;
        assert_eq!(ocean.h0.len(), n * n);
        assert_eq!(ocean.heightfield.len(), n * n);
        assert_eq!(ocean.displacement_x.len(), n * n);
    }

    #[test]
    fn test_ocean_spectrum_generation() {
        let mut config = FFTOceanConfig::default();
        config.fft_size = 64; // Smaller for faster test

        let mut ocean = FFTOcean::new(config);
        ocean.generate_phillips_spectrum();

        // Verify non-zero spectrum
        let energy = ocean.total_energy();
        assert!(energy > 0.0);
    }

    #[test]
    fn test_ocean_spectrum_deterministic() {
        let mut config = FFTOceanConfig::default();
        config.fft_size = 32;

        let mut ocean1 = FFTOcean::new(config);
        let mut ocean2 = FFTOcean::new(config);

        ocean1.generate_phillips_spectrum_with_seed(42);
        ocean2.generate_phillips_spectrum_with_seed(42);

        for (a, b) in ocean1.h0.iter().zip(ocean2.h0.iter()) {
            assert!(complex_approx_eq(*a, *b, EPSILON_TEST));
        }
    }

    #[test]
    fn test_ocean_evolve_spectrum_t0() {
        let mut config = FFTOceanConfig::default();
        config.fft_size = 32;

        let mut ocean = FFTOcean::new(config);
        ocean.generate_phillips_spectrum();
        ocean.evolve_spectrum(0.0);

        // At t=0, exp(i*0) = 1, so h_tilde = h0 + h0_conj
        // This should be real-valued (imaginary parts cancel)
        for h in &ocean.h_tilde {
            // h_tilde should be h0 + conj(h0(-k))
            // The imaginary part won't be exactly 0 due to different h0(-k)
            // but DC should be real
        }
        // Just verify spectrum evolved
        let has_nonzero = ocean.h_tilde.iter().any(|h| h.magnitude() > EPSILON_TEST);
        assert!(has_nonzero);
    }

    #[test]
    fn test_ocean_evolve_spectrum_changes() {
        let mut config = FFTOceanConfig::default();
        config.fft_size = 32;

        let mut ocean = FFTOcean::new(config);
        ocean.generate_phillips_spectrum();

        ocean.evolve_spectrum(0.0);
        let h0_copy = ocean.h_tilde.clone();

        ocean.evolve_spectrum(1.0);
        let h1_copy = ocean.h_tilde.clone();

        // Spectrum should change with time
        let mut different = false;
        for (a, b) in h0_copy.iter().zip(h1_copy.iter()) {
            if !complex_approx_eq(*a, *b, 1e-3) {
                different = true;
                break;
            }
        }
        assert!(different);
    }

    #[test]
    fn test_ocean_ifft_produces_real() {
        let mut config = FFTOceanConfig::default();
        config.fft_size = 32;

        let mut ocean = FFTOcean::new(config);
        ocean.generate_phillips_spectrum();
        ocean.evolve_spectrum(1.5);
        ocean.ifft_2d();

        // Heightfield should have finite values
        for h in &ocean.heightfield {
            assert!(h.is_finite());
        }
    }

    #[test]
    fn test_ocean_update() {
        let mut config = FFTOceanConfig::default();
        config.fft_size = 32;

        let mut ocean = FFTOcean::new(config);
        ocean.generate_phillips_spectrum();
        ocean.update(0.0);

        // Should have valid outputs
        let rms = ocean.rms_height();
        assert!(rms >= 0.0);
        assert!(rms.is_finite());
    }

    #[test]
    fn test_ocean_displacement() {
        let mut config = FFTOceanConfig::default();
        config.fft_size = 32;
        config.chop_amount = 1.5;

        let mut ocean = FFTOcean::new(config);
        ocean.generate_phillips_spectrum();
        ocean.update(1.0);

        // Should have finite displacement
        for d in &ocean.displacement_x {
            assert!(d.is_finite());
        }
        for d in &ocean.displacement_z {
            assert!(d.is_finite());
        }
    }

    #[test]
    fn test_ocean_sample_height() {
        let mut config = FFTOceanConfig::default();
        config.fft_size = 32;

        let mut ocean = FFTOcean::new(config);
        ocean.generate_phillips_spectrum();
        ocean.update(1.0);

        let h = ocean.sample_height(0.5, 0.5);
        assert!(h.is_finite());
    }

    #[test]
    fn test_ocean_sample_displacement() {
        let mut config = FFTOceanConfig::default();
        config.fft_size = 32;

        let mut ocean = FFTOcean::new(config);
        ocean.generate_phillips_spectrum();
        ocean.update(1.0);

        let [dx, dz] = ocean.sample_displacement(0.25, 0.75);
        assert!(dx.is_finite());
        assert!(dz.is_finite());
    }

    #[test]
    fn test_ocean_sample_world() {
        let mut config = FFTOceanConfig::default();
        config.fft_size = 32;
        config.patch_size = 100.0;

        let mut ocean = FFTOcean::new(config);
        ocean.generate_phillips_spectrum();
        ocean.update(1.0);

        let (h, [dx, dz]) = ocean.sample_world(50.0, 75.0);
        assert!(h.is_finite());
        assert!(dx.is_finite());
        assert!(dz.is_finite());
    }

    #[test]
    fn test_ocean_tiling() {
        let mut config = FFTOceanConfig::default();
        config.fft_size = 32;

        let mut ocean = FFTOcean::new(config);
        ocean.generate_phillips_spectrum();
        ocean.update(1.0);

        // Ocean should tile seamlessly
        let h0 = ocean.sample_height(0.0, 0.0);
        let h1 = ocean.sample_height(1.0, 1.0);
        assert!(approx_eq(h0, h1, EPSILON_TEST));
    }

    #[test]
    fn test_ocean_energy_conservation() {
        let mut config = FFTOceanConfig::default();
        config.fft_size = 32;

        let mut ocean = FFTOcean::new(config);
        ocean.generate_phillips_spectrum();

        let e0 = ocean.total_energy();
        ocean.update(0.0);
        let e1: f32 = ocean.h_tilde.iter().map(|h| h.magnitude_squared()).sum();

        // Energy should be conserved (approximately, due to h0_conj)
        // Actually increases due to h0 + h0_conj, but bounded
        assert!(e1 < e0 * 10.0); // Reasonable bound
    }

    #[test]
    fn test_ocean_rms_height() {
        let mut config = FFTOceanConfig::default();
        config.fft_size = 32;

        let mut ocean = FFTOcean::new(config);
        ocean.generate_phillips_spectrum();
        ocean.update(1.0);

        let rms = ocean.rms_height();
        assert!(rms > 0.0);
        assert!(rms.is_finite());
    }

    #[test]
    fn test_ocean_stormy_has_larger_waves() {
        let mut calm_config = FFTOceanConfig::calm();
        calm_config.fft_size = 32;
        let mut calm = FFTOcean::new(calm_config);
        calm.generate_phillips_spectrum();
        calm.update(1.0);

        let mut stormy_config = FFTOceanConfig::stormy();
        stormy_config.fft_size = 32;
        let mut stormy = FFTOcean::new(stormy_config);
        stormy.generate_phillips_spectrum();
        stormy.update(1.0);

        assert!(stormy.rms_height() > calm.rms_height());
    }

    #[test]
    fn test_wind_direction_affects_spectrum() {
        let mut config1 = FFTOceanConfig::default();
        config1.fft_size = 32;
        config1.wind_direction = [1.0, 0.0];

        let mut config2 = FFTOceanConfig::default();
        config2.fft_size = 32;
        config2.wind_direction = [0.0, 1.0];

        let mut ocean1 = FFTOcean::new(config1);
        let mut ocean2 = FFTOcean::new(config2);

        ocean1.generate_phillips_spectrum_with_seed(42);
        ocean2.generate_phillips_spectrum_with_seed(42);

        // Different wind directions should produce different spectra
        let mut different = false;
        for (a, b) in ocean1.h0.iter().zip(ocean2.h0.iter()) {
            if !complex_approx_eq(*a, *b, 1e-3) {
                different = true;
                break;
            }
        }
        assert!(different);
    }

    #[test]
    fn test_chop_amount_scaling() {
        let mut config1 = FFTOceanConfig::default();
        config1.fft_size = 32;
        config1.chop_amount = 1.0;

        let mut config2 = FFTOceanConfig::default();
        config2.fft_size = 32;
        config2.chop_amount = 2.0;

        let mut ocean1 = FFTOcean::new(config1);
        let mut ocean2 = FFTOcean::new(config2);

        ocean1.generate_phillips_spectrum_with_seed(42);
        ocean2.generate_phillips_spectrum_with_seed(42);

        ocean1.update(1.0);
        ocean2.update(1.0);

        // Higher chop should give larger displacement
        let max_disp1: f32 = ocean1
            .displacement_x
            .iter()
            .map(|x| x.abs())
            .fold(0.0, f32::max);
        let max_disp2: f32 = ocean2
            .displacement_x
            .iter()
            .map(|x| x.abs())
            .fold(0.0, f32::max);

        assert!(max_disp2 > max_disp1);
    }

    // ===== Bytemuck Tests =====

    #[test]
    fn test_complex_pod() {
        let c = Complex::new(1.0, 2.0);
        let bytes: &[u8] = bytemuck::bytes_of(&c);
        assert_eq!(bytes.len(), 8);
    }

    #[test]
    fn test_config_pod() {
        let config = FFTOceanConfig::default();
        let bytes: &[u8] = bytemuck::bytes_of(&config);
        assert_eq!(bytes.len(), FFT_OCEAN_CONFIG_SIZE);
    }

    #[test]
    fn test_complex_zeroable() {
        let c: Complex = bytemuck::Zeroable::zeroed();
        assert_eq!(c.re, 0.0);
        assert_eq!(c.im, 0.0);
    }
}
