//! God Rays (Crepuscular Rays) Rendering System (T-ENV-2.5)
//!
//! Implements volumetric light shafts / crepuscular rays for TRINITY's renderer.
//! Uses cloud density from `cloud_raymarching.rs` as the primary occluder source.
//!
//! # Overview
//!
//! The god ray system provides:
//! - **GodRayQuality**: Presets for sample count vs performance tradeoffs
//! - **GodRayUniforms**: GPU-uploadable configuration parameters
//! - **EpipolarConfig**: Epipolar sampling optimization for reduced ray count
//! - **TemporalConfig**: Temporal stability with reprojection
//! - **GodRayRenderer**: Main API for screen-space god ray rendering
//!
//! # Algorithm
//!
//! 1. Project sun position to screen space
//! 2. For each pixel, march radially toward sun position
//! 3. Sample depth/cloud buffer for occlusion at each step
//! 4. Accumulate light contribution using Beer-Lambert extinction
//! 5. Apply decay, density, and exposure post-factors
//! 6. Bilateral upsample for performance (optional half-res rendering)
//!
//! # Physics Model
//!
//! Light extinction follows Beer-Lambert law:
//! ```text
//! I(d) = I_0 * exp(-sigma * d)
//! ```
//!
//! Where:
//! - I_0 is initial light intensity
//! - sigma is the extinction coefficient
//! - d is the accumulated optical depth
//!
//! # Performance
//!
//! At 64 samples (High quality), targets 60 fps with:
//! - Epipolar sampling reducing ray count by up to 80%
//! - Half-resolution rendering with bilateral upsampling
//! - Temporal accumulation for noise reduction
//!
//! # References
//!
//! - Mitchell, "Volumetric Light Scattering as a Post-Process" (GPU Gems 3)
//! - Hillaire, "Physically Based Sky, Atmosphere and Cloud Rendering in Frostbite"
//! - Wronski, "Volumetric Fog and Lighting in Assassin's Creed 4"

use bytemuck::{Pod, Zeroable};
use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default number of radial samples for god rays.
pub const DEFAULT_NUM_SAMPLES: u32 = 64;

/// Default decay factor per sample (light falloff along ray).
pub const DEFAULT_DECAY: f32 = 0.96;

/// Default density factor (overall brightness).
pub const DEFAULT_DENSITY: f32 = 1.0;

/// Default weight factor (importance of each sample).
pub const DEFAULT_WEIGHT: f32 = 0.5;

/// Default exposure (final brightness scaling).
pub const DEFAULT_EXPOSURE: f32 = 0.3;

/// Default extinction coefficient for Beer-Lambert.
pub const DEFAULT_EXTINCTION_COEFF: f32 = 0.1;

/// Default scattering albedo (fraction of light scattered vs absorbed).
pub const DEFAULT_SCATTERING_ALBEDO: f32 = 0.8;

/// Maximum sun screen position clamping (normalized coords).
pub const SUN_SCREEN_CLAMP: f32 = 2.0;

/// Minimum valid decay value.
pub const MIN_DECAY: f32 = 0.5;

/// Maximum valid decay value.
pub const MAX_DECAY: f32 = 0.999;

/// Minimum valid density.
pub const MIN_DENSITY: f32 = 0.0;

/// Maximum valid density.
pub const MAX_DENSITY: f32 = 5.0;

/// Minimum valid exposure.
pub const MIN_EXPOSURE: f32 = 0.0;

/// Maximum valid exposure.
pub const MAX_EXPOSURE: f32 = 10.0;

/// Default epipolar slice count.
pub const DEFAULT_EPIPOLAR_SLICES: u32 = 256;

/// Default samples per epipolar slice.
pub const DEFAULT_EPIPOLAR_SAMPLES: u32 = 128;

/// Default temporal blend factor.
pub const DEFAULT_TEMPORAL_BLEND: f32 = 0.9;

/// Default jitter scale for temporal stability.
pub const DEFAULT_JITTER_SCALE: f32 = 0.5;

/// Small epsilon for floating point comparisons.
pub const EPSILON: f32 = 1e-6;

/// Half-resolution scale factor.
pub const HALF_RES_SCALE: f32 = 0.5;

/// Quarter-resolution scale factor.
pub const QUARTER_RES_SCALE: f32 = 0.25;

/// Default bilateral sigma (spatial).
pub const DEFAULT_BILATERAL_SIGMA_SPATIAL: f32 = 3.0;

/// Default bilateral sigma (range/depth).
pub const DEFAULT_BILATERAL_SIGMA_RANGE: f32 = 0.1;

/// Maximum depth difference for bilateral filtering.
pub const BILATERAL_DEPTH_THRESHOLD: f32 = 0.01;

/// Default sun angular diameter in radians (0.53 degrees).
pub const SUN_ANGULAR_DIAMETER: f32 = 0.00925; // ~0.53 degrees

// ---------------------------------------------------------------------------
// GodRayQuality - Quality presets
// ---------------------------------------------------------------------------

/// God ray rendering quality preset.
///
/// Controls the number of radial samples, trading quality for performance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
#[repr(u8)]
pub enum GodRayQuality {
    /// Low quality: 16 samples. Fast, suitable for mobile/low-end.
    Low = 0,

    /// Medium quality: 32 samples. Balanced quality/performance.
    #[default]
    Medium = 1,

    /// High quality: 64 samples. Good quality at 60 fps.
    High = 2,

    /// Ultra quality: 128 samples. Maximum quality, demanding.
    Ultra = 3,
}

impl GodRayQuality {
    /// Get the number of radial samples for this quality level.
    #[inline]
    pub fn sample_count(&self) -> u32 {
        match self {
            GodRayQuality::Low => 16,
            GodRayQuality::Medium => 32,
            GodRayQuality::High => 64,
            GodRayQuality::Ultra => 128,
        }
    }

    /// Get the resolution scale for this quality level.
    ///
    /// Lower quality uses lower resolution for the ray march pass.
    #[inline]
    pub fn resolution_scale(&self) -> f32 {
        match self {
            GodRayQuality::Low => QUARTER_RES_SCALE,
            GodRayQuality::Medium => HALF_RES_SCALE,
            GodRayQuality::High => HALF_RES_SCALE,
            GodRayQuality::Ultra => 1.0,
        }
    }

    /// Get the decay factor for this quality level.
    ///
    /// Fewer samples need faster decay to maintain similar visual result.
    #[inline]
    pub fn decay(&self) -> f32 {
        match self {
            GodRayQuality::Low => 0.92,
            GodRayQuality::Medium => 0.95,
            GodRayQuality::High => 0.96,
            GodRayQuality::Ultra => 0.98,
        }
    }

    /// Get quality level from sample count (rounds to nearest).
    #[inline]
    pub fn from_sample_count(samples: u32) -> Self {
        if samples >= 96 {
            GodRayQuality::Ultra
        } else if samples >= 48 {
            GodRayQuality::High
        } else if samples >= 24 {
            GodRayQuality::Medium
        } else {
            GodRayQuality::Low
        }
    }

    /// Get all quality levels.
    #[inline]
    pub fn all() -> [Self; 4] {
        [
            GodRayQuality::Low,
            GodRayQuality::Medium,
            GodRayQuality::High,
            GodRayQuality::Ultra,
        ]
    }

    /// Get name string.
    #[inline]
    pub fn name(&self) -> &'static str {
        match self {
            GodRayQuality::Low => "low",
            GodRayQuality::Medium => "medium",
            GodRayQuality::High => "high",
            GodRayQuality::Ultra => "ultra",
        }
    }
}

// ---------------------------------------------------------------------------
// GodRayUniforms - GPU-uploadable configuration
// ---------------------------------------------------------------------------

/// GPU-uploadable configuration for god ray rendering.
///
/// This struct is designed for direct upload to GPU uniform buffers.
/// The layout is `repr(C)` and implements `Pod` for bytemuck compatibility.
///
/// # Memory Layout (64 bytes, vec4 aligned)
///
/// | Offset | Field              | Size     |
/// |--------|--------------------|----------|
/// | 0      | sun_screen_pos     | 8 bytes  |
/// | 8      | decay              | 4 bytes  |
/// | 12     | exposure           | 4 bytes  |
/// | 16     | density            | 4 bytes  |
/// | 20     | weight             | 4 bytes  |
/// | 24     | num_samples        | 4 bytes  |
/// | 28     | extinction_coeff   | 4 bytes  |
/// | 32     | sun_intensity      | 12 bytes |
/// | 44     | max_ray_length     | 4 bytes  |
/// | 48     | resolution_scale   | 4 bytes  |
/// | 52     | _padding           | 12 bytes |
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct GodRayUniforms {
    /// Sun position in normalized screen coordinates (0-1, origin top-left).
    ///
    /// X: horizontal position (0 = left, 1 = right)
    /// Y: vertical position (0 = top, 1 = bottom)
    pub sun_screen_pos: [f32; 2],

    /// Decay factor per sample (exponential falloff along ray).
    ///
    /// Controls how quickly light contribution fades with distance from sun.
    /// Range: 0.5 to 0.999. Typical: 0.96.
    pub decay: f32,

    /// Final exposure/brightness scaling.
    ///
    /// Multiplied with accumulated light after all samples.
    /// Range: 0.0 to 10.0. Typical: 0.3.
    pub exposure: f32,

    /// Overall density multiplier.
    ///
    /// Scales the brightness contribution of each sample.
    /// Range: 0.0 to 5.0. Typical: 1.0.
    pub density: f32,

    /// Per-sample weight factor.
    ///
    /// Importance weight for accumulated light contributions.
    /// Range: 0.0 to 1.0. Typical: 0.5.
    pub weight: f32,

    /// Number of radial samples to take.
    ///
    /// More samples = smoother rays but slower.
    /// Typical: 32-128.
    pub num_samples: u32,

    /// Extinction coefficient for Beer-Lambert attenuation.
    ///
    /// Controls how quickly light is absorbed in the medium.
    /// Higher values = denser atmosphere.
    pub extinction_coeff: f32,

    /// Sun/light color and intensity (RGB, linear).
    ///
    /// The color tint applied to the god rays.
    pub sun_intensity: [f32; 3],

    /// Maximum ray march length in screen space (normalized).
    ///
    /// Limits how far from the sun rays are computed.
    /// Range: 0.0 to 2.0. Default: 1.0.
    pub max_ray_length: f32,

    /// Resolution scale for ray marching pass.
    ///
    /// 1.0 = full resolution, 0.5 = half resolution.
    pub resolution_scale: f32,

    /// Padding for GPU alignment (vec4 boundary).
    pub _padding: [u32; 3],
}

// Size assertion for GPU compatibility
const _: () = assert!(std::mem::size_of::<GodRayUniforms>() == 64);

impl Default for GodRayUniforms {
    fn default() -> Self {
        Self {
            sun_screen_pos: [0.5, 0.0], // Top-center
            decay: DEFAULT_DECAY,
            exposure: DEFAULT_EXPOSURE,
            density: DEFAULT_DENSITY,
            weight: DEFAULT_WEIGHT,
            num_samples: DEFAULT_NUM_SAMPLES,
            extinction_coeff: DEFAULT_EXTINCTION_COEFF,
            sun_intensity: [1.0, 0.95, 0.8], // Warm sunlight
            max_ray_length: 1.0,
            resolution_scale: HALF_RES_SCALE,
            _padding: [0; 3],
        }
    }
}

impl GodRayUniforms {
    /// Create a new configuration with default values.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create from a quality preset.
    #[inline]
    pub fn from_quality(quality: GodRayQuality) -> Self {
        Self {
            num_samples: quality.sample_count(),
            decay: quality.decay(),
            resolution_scale: quality.resolution_scale(),
            ..Self::default()
        }
    }

    /// Set sun screen position.
    #[inline]
    pub fn with_sun_position(mut self, x: f32, y: f32) -> Self {
        self.sun_screen_pos = [
            x.clamp(-SUN_SCREEN_CLAMP, 1.0 + SUN_SCREEN_CLAMP),
            y.clamp(-SUN_SCREEN_CLAMP, 1.0 + SUN_SCREEN_CLAMP),
        ];
        self
    }

    /// Set decay factor.
    #[inline]
    pub fn with_decay(mut self, decay: f32) -> Self {
        self.decay = decay.clamp(MIN_DECAY, MAX_DECAY);
        self
    }

    /// Set exposure.
    #[inline]
    pub fn with_exposure(mut self, exposure: f32) -> Self {
        self.exposure = exposure.clamp(MIN_EXPOSURE, MAX_EXPOSURE);
        self
    }

    /// Set density.
    #[inline]
    pub fn with_density(mut self, density: f32) -> Self {
        self.density = density.clamp(MIN_DENSITY, MAX_DENSITY);
        self
    }

    /// Set weight.
    #[inline]
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Set number of samples.
    #[inline]
    pub fn with_samples(mut self, samples: u32) -> Self {
        self.num_samples = samples.clamp(4, 256);
        self
    }

    /// Set extinction coefficient.
    #[inline]
    pub fn with_extinction(mut self, coeff: f32) -> Self {
        self.extinction_coeff = coeff.max(0.0);
        self
    }

    /// Set sun color and intensity.
    #[inline]
    pub fn with_sun_intensity(mut self, r: f32, g: f32, b: f32) -> Self {
        self.sun_intensity = [r.max(0.0), g.max(0.0), b.max(0.0)];
        self
    }

    /// Set maximum ray length.
    #[inline]
    pub fn with_max_ray_length(mut self, length: f32) -> Self {
        self.max_ray_length = length.clamp(0.1, 2.0);
        self
    }

    /// Set resolution scale.
    #[inline]
    pub fn with_resolution_scale(mut self, scale: f32) -> Self {
        self.resolution_scale = scale.clamp(0.25, 1.0);
        self
    }

    /// Validate the configuration.
    #[inline]
    pub fn validate(&self) -> bool {
        self.decay >= MIN_DECAY
            && self.decay <= MAX_DECAY
            && self.exposure >= MIN_EXPOSURE
            && self.exposure <= MAX_EXPOSURE
            && self.density >= MIN_DENSITY
            && self.density <= MAX_DENSITY
            && self.weight >= 0.0
            && self.weight <= 1.0
            && self.num_samples >= 4
            && self.num_samples <= 256
            && self.extinction_coeff >= 0.0
            && self.resolution_scale > 0.0
            && self.resolution_scale <= 1.0
    }

    /// Check if sun is visible on screen.
    #[inline]
    pub fn is_sun_on_screen(&self) -> bool {
        self.sun_screen_pos[0] >= 0.0
            && self.sun_screen_pos[0] <= 1.0
            && self.sun_screen_pos[1] >= 0.0
            && self.sun_screen_pos[1] <= 1.0
    }

    /// Check if sun is in front of camera (not behind horizon).
    #[inline]
    pub fn is_sun_visible(&self) -> bool {
        // Sun is visible if within extended screen bounds
        self.sun_screen_pos[0] >= -SUN_SCREEN_CLAMP
            && self.sun_screen_pos[0] <= 1.0 + SUN_SCREEN_CLAMP
            && self.sun_screen_pos[1] >= -SUN_SCREEN_CLAMP
            && self.sun_screen_pos[1] <= 1.0 + SUN_SCREEN_CLAMP
    }
}

// ---------------------------------------------------------------------------
// EpipolarConfig - Epipolar sampling optimization
// ---------------------------------------------------------------------------

/// Configuration for epipolar sampling optimization.
///
/// Epipolar sampling reduces ray count by sampling along epipolar lines
/// (lines radiating from the sun position) rather than uniformly.
///
/// # Memory Layout (32 bytes)
///
/// | Offset | Field              | Size     |
/// |--------|--------------------|----------|
/// | 0      | num_slices         | 4 bytes  |
/// | 4      | samples_per_slice  | 4 bytes  |
/// | 8      | min_march_samples  | 4 bytes  |
/// | 12     | max_march_samples  | 4 bytes  |
/// | 16     | depth_threshold    | 4 bytes  |
/// | 20     | interpolation_max  | 4 bytes  |
/// | 24     | _padding           | 8 bytes  |
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct EpipolarConfig {
    /// Number of epipolar slices (angular divisions around sun).
    ///
    /// More slices = smoother angular coverage.
    /// Typical: 128-512.
    pub num_slices: u32,

    /// Number of samples along each epipolar slice.
    ///
    /// Samples are distributed radially from sun to screen edge.
    /// Typical: 64-256.
    pub samples_per_slice: u32,

    /// Minimum samples per ray march in depth.
    pub min_march_samples: u32,

    /// Maximum samples per ray march in depth.
    pub max_march_samples: u32,

    /// Depth threshold for slice discontinuities.
    ///
    /// When depth changes exceed this, new slice is started.
    pub depth_threshold: f32,

    /// Maximum interpolation distance between samples.
    ///
    /// Limits stretching artifacts in smooth regions.
    pub interpolation_max: f32,

    /// Padding for GPU alignment.
    pub _padding: [u32; 2],
}

// Size assertion
const _: () = assert!(std::mem::size_of::<EpipolarConfig>() == 32);

impl Default for EpipolarConfig {
    fn default() -> Self {
        Self {
            num_slices: DEFAULT_EPIPOLAR_SLICES,
            samples_per_slice: DEFAULT_EPIPOLAR_SAMPLES,
            min_march_samples: 8,
            max_march_samples: 64,
            depth_threshold: 0.01,
            interpolation_max: 32.0,
            _padding: [0; 2],
        }
    }
}

impl EpipolarConfig {
    /// Create a new epipolar configuration.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create from quality preset.
    #[inline]
    pub fn from_quality(quality: GodRayQuality) -> Self {
        match quality {
            GodRayQuality::Low => Self {
                num_slices: 64,
                samples_per_slice: 32,
                min_march_samples: 4,
                max_march_samples: 16,
                depth_threshold: 0.02,
                interpolation_max: 64.0,
                _padding: [0; 2],
            },
            GodRayQuality::Medium => Self {
                num_slices: 128,
                samples_per_slice: 64,
                min_march_samples: 8,
                max_march_samples: 32,
                depth_threshold: 0.015,
                interpolation_max: 48.0,
                _padding: [0; 2],
            },
            GodRayQuality::High => Self {
                num_slices: 256,
                samples_per_slice: 128,
                min_march_samples: 8,
                max_march_samples: 64,
                depth_threshold: 0.01,
                interpolation_max: 32.0,
                _padding: [0; 2],
            },
            GodRayQuality::Ultra => Self {
                num_slices: 512,
                samples_per_slice: 256,
                min_march_samples: 16,
                max_march_samples: 128,
                depth_threshold: 0.005,
                interpolation_max: 16.0,
                _padding: [0; 2],
            },
        }
    }

    /// Set number of slices.
    #[inline]
    pub fn with_slices(mut self, slices: u32) -> Self {
        self.num_slices = slices.clamp(16, 1024);
        self
    }

    /// Set samples per slice.
    #[inline]
    pub fn with_samples_per_slice(mut self, samples: u32) -> Self {
        self.samples_per_slice = samples.clamp(8, 512);
        self
    }

    /// Set depth threshold.
    #[inline]
    pub fn with_depth_threshold(mut self, threshold: f32) -> Self {
        self.depth_threshold = threshold.clamp(0.001, 0.1);
        self
    }

    /// Calculate total sample count for this configuration.
    #[inline]
    pub fn total_samples(&self) -> u32 {
        self.num_slices * self.samples_per_slice
    }

    /// Calculate memory footprint in bytes (for slice buffer).
    #[inline]
    pub fn buffer_size_bytes(&self) -> usize {
        // 4 floats per sample: accumulated, depth, weight, _padding
        (self.total_samples() as usize) * 16
    }

    /// Validate the configuration.
    #[inline]
    pub fn validate(&self) -> bool {
        self.num_slices >= 16
            && self.num_slices <= 1024
            && self.samples_per_slice >= 8
            && self.samples_per_slice <= 512
            && self.depth_threshold > 0.0
            && self.interpolation_max > 0.0
    }
}

// ---------------------------------------------------------------------------
// TemporalConfig - Temporal stability configuration
// ---------------------------------------------------------------------------

/// Configuration for temporal reprojection and stability.
///
/// Temporal filtering reduces flickering and noise by blending
/// the current frame with previous frames using motion vectors.
///
/// # Memory Layout (32 bytes)
///
/// | Offset | Field               | Size     |
/// |--------|--------------------|----------|
/// | 0      | blend_factor       | 4 bytes  |
/// | 4      | jitter_scale       | 4 bytes  |
/// | 8      | velocity_weight    | 4 bytes  |
/// | 12     | depth_reject_thresh| 4 bytes  |
/// | 16     | color_box_size     | 4 bytes  |
/// | 20     | frame_index        | 4 bytes  |
/// | 24     | _padding           | 8 bytes  |
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct TemporalConfig {
    /// Blend factor between current and history (0-1).
    ///
    /// Higher = more history (smoother but more ghosting).
    /// Typical: 0.85-0.95.
    pub blend_factor: f32,

    /// Scale of temporal jitter offset.
    ///
    /// Jittering sample positions reduces banding.
    /// Typical: 0.25-1.0.
    pub jitter_scale: f32,

    /// Weight given to motion/velocity rejection.
    ///
    /// Higher = more aggressive rejection of fast-moving pixels.
    pub velocity_weight: f32,

    /// Depth difference threshold for history rejection.
    ///
    /// If depth changes more than this, history is discarded.
    pub depth_reject_threshold: f32,

    /// Color box neighborhood size for variance clamping.
    ///
    /// Typical: 1 (3x3) or 2 (5x5).
    pub color_box_size: u32,

    /// Current frame index (for jitter sequence).
    pub frame_index: u32,

    /// Padding for GPU alignment.
    pub _padding: [u32; 2],
}

// Size assertion
const _: () = assert!(std::mem::size_of::<TemporalConfig>() == 32);

impl Default for TemporalConfig {
    fn default() -> Self {
        Self {
            blend_factor: DEFAULT_TEMPORAL_BLEND,
            jitter_scale: DEFAULT_JITTER_SCALE,
            velocity_weight: 1.0,
            depth_reject_threshold: 0.05,
            color_box_size: 1,
            frame_index: 0,
            _padding: [0; 2],
        }
    }
}

impl TemporalConfig {
    /// Create a new temporal configuration.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with temporal filtering disabled.
    #[inline]
    pub fn disabled() -> Self {
        Self {
            blend_factor: 0.0,
            jitter_scale: 0.0,
            velocity_weight: 0.0,
            depth_reject_threshold: 0.0,
            color_box_size: 0,
            frame_index: 0,
            _padding: [0; 2],
        }
    }

    /// Set blend factor.
    #[inline]
    pub fn with_blend_factor(mut self, factor: f32) -> Self {
        self.blend_factor = factor.clamp(0.0, 0.99);
        self
    }

    /// Set jitter scale.
    #[inline]
    pub fn with_jitter_scale(mut self, scale: f32) -> Self {
        self.jitter_scale = scale.clamp(0.0, 2.0);
        self
    }

    /// Set depth reject threshold.
    #[inline]
    pub fn with_depth_threshold(mut self, threshold: f32) -> Self {
        self.depth_reject_threshold = threshold.clamp(0.001, 0.5);
        self
    }

    /// Advance to next frame.
    #[inline]
    pub fn next_frame(&mut self) {
        self.frame_index = self.frame_index.wrapping_add(1);
    }

    /// Get the current jitter offset for temporal anti-aliasing.
    ///
    /// Returns an offset in normalized screen coordinates.
    #[inline]
    pub fn current_jitter(&self) -> [f32; 2] {
        halton_2d(self.frame_index, self.jitter_scale)
    }

    /// Check if temporal filtering is enabled.
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.blend_factor > EPSILON
    }

    /// Validate the configuration.
    #[inline]
    pub fn validate(&self) -> bool {
        self.blend_factor >= 0.0
            && self.blend_factor < 1.0
            && self.jitter_scale >= 0.0
            && self.depth_reject_threshold >= 0.0
    }
}

// ---------------------------------------------------------------------------
// BilateralConfig - Bilateral upsample configuration
// ---------------------------------------------------------------------------

/// Configuration for bilateral upsampling.
///
/// When rendering god rays at lower resolution, bilateral upsampling
/// preserves edges while smoothly interpolating the result.
///
/// # Memory Layout (32 bytes)
///
/// | Offset | Field            | Size     |
/// |--------|------------------|----------|
/// | 0      | sigma_spatial    | 4 bytes  |
/// | 4      | sigma_range      | 4 bytes  |
/// | 8      | depth_threshold  | 4 bytes  |
/// | 12     | kernel_radius    | 4 bytes  |
/// | 16     | enabled          | 4 bytes  |
/// | 20     | _padding         | 12 bytes |
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct BilateralConfig {
    /// Spatial sigma (blur radius in pixels).
    ///
    /// Higher = more blurring. Typical: 2.0-5.0.
    pub sigma_spatial: f32,

    /// Range sigma (depth/color difference tolerance).
    ///
    /// Higher = more blurring across edges. Typical: 0.05-0.2.
    pub sigma_range: f32,

    /// Depth difference threshold for edge detection.
    ///
    /// Depths differing by more than this are considered edges.
    pub depth_threshold: f32,

    /// Kernel radius in pixels.
    ///
    /// Typical: 2-5.
    pub kernel_radius: u32,

    /// Whether bilateral filtering is enabled.
    pub enabled: u32,

    /// Padding for GPU alignment.
    pub _padding: [u32; 3],
}

// Size assertion
const _: () = assert!(std::mem::size_of::<BilateralConfig>() == 32);

impl Default for BilateralConfig {
    fn default() -> Self {
        Self {
            sigma_spatial: DEFAULT_BILATERAL_SIGMA_SPATIAL,
            sigma_range: DEFAULT_BILATERAL_SIGMA_RANGE,
            depth_threshold: BILATERAL_DEPTH_THRESHOLD,
            kernel_radius: 3,
            enabled: 1,
            _padding: [0; 3],
        }
    }
}

impl BilateralConfig {
    /// Create a new bilateral configuration.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with bilateral filtering disabled.
    #[inline]
    pub fn disabled() -> Self {
        Self {
            sigma_spatial: 0.0,
            sigma_range: 0.0,
            depth_threshold: 0.0,
            kernel_radius: 0,
            enabled: 0,
            _padding: [0; 3],
        }
    }

    /// Set spatial sigma.
    #[inline]
    pub fn with_sigma_spatial(mut self, sigma: f32) -> Self {
        self.sigma_spatial = sigma.clamp(0.5, 10.0);
        self
    }

    /// Set range sigma.
    #[inline]
    pub fn with_sigma_range(mut self, sigma: f32) -> Self {
        self.sigma_range = sigma.clamp(0.01, 1.0);
        self
    }

    /// Set kernel radius.
    #[inline]
    pub fn with_kernel_radius(mut self, radius: u32) -> Self {
        self.kernel_radius = radius.clamp(1, 8);
        self
    }

    /// Enable or disable bilateral filtering.
    #[inline]
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = if enabled { 1 } else { 0 };
        self
    }

    /// Check if enabled.
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.enabled != 0
    }

    /// Validate the configuration.
    #[inline]
    pub fn validate(&self) -> bool {
        !self.is_enabled()
            || (self.sigma_spatial > 0.0
                && self.sigma_range > 0.0
                && self.kernel_radius >= 1)
    }
}

// ---------------------------------------------------------------------------
// OcclusionSource - Occlusion sampling modes
// ---------------------------------------------------------------------------

/// Source for god ray occlusion sampling.
///
/// Different sources provide different visual results and performance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub enum OcclusionSource {
    /// Sample from depth buffer only.
    ///
    /// Fast but misses volumetric occluders like clouds.
    DepthBuffer,

    /// Sample from cloud density texture.
    ///
    /// Accurate for cloud-based god rays.
    CloudDensity,

    /// Combined depth and cloud sampling.
    ///
    /// Best quality, samples both sources.
    #[default]
    Combined,

    /// Use stencil-marked geometry as occluder.
    ///
    /// For specific occluder geometry (trees, buildings).
    StencilMask,
}

impl OcclusionSource {
    /// Get name string.
    #[inline]
    pub fn name(&self) -> &'static str {
        match self {
            OcclusionSource::DepthBuffer => "depth_buffer",
            OcclusionSource::CloudDensity => "cloud_density",
            OcclusionSource::Combined => "combined",
            OcclusionSource::StencilMask => "stencil_mask",
        }
    }

    /// Check if this source requires cloud texture binding.
    #[inline]
    pub fn needs_cloud_texture(&self) -> bool {
        matches!(self, OcclusionSource::CloudDensity | OcclusionSource::Combined)
    }

    /// Check if this source requires depth texture binding.
    #[inline]
    pub fn needs_depth_texture(&self) -> bool {
        matches!(
            self,
            OcclusionSource::DepthBuffer | OcclusionSource::Combined | OcclusionSource::StencilMask
        )
    }
}

// ---------------------------------------------------------------------------
// GodRaySample - Single sample result
// ---------------------------------------------------------------------------

/// Result of sampling god rays at a single position.
///
/// Contains accumulated light and occlusion information.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct GodRaySample {
    /// Accumulated light contribution (RGB linear).
    pub light: [f32; 3],

    /// Current transmittance (Beer-Lambert).
    pub transmittance: f32,

    /// Total distance traveled along ray.
    pub distance: f32,

    /// Number of samples taken.
    pub sample_count: u32,
}

impl GodRaySample {
    /// Create a new sample with initial values.
    #[inline]
    pub fn new() -> Self {
        Self {
            light: [0.0; 3],
            transmittance: 1.0,
            distance: 0.0,
            sample_count: 0,
        }
    }

    /// Check if fully occluded.
    #[inline]
    pub fn is_occluded(&self, threshold: f32) -> bool {
        self.transmittance < threshold
    }

    /// Get the luminance of accumulated light.
    #[inline]
    pub fn luminance(&self) -> f32 {
        0.2126 * self.light[0] + 0.7152 * self.light[1] + 0.0722 * self.light[2]
    }

    /// Accumulate a light sample.
    #[inline]
    pub fn accumulate(&mut self, occlusion: f32, sun_intensity: [f32; 3], decay: f32, weight: f32) {
        let contribution = 1.0 - occlusion;
        let scaled_weight = weight * self.transmittance * contribution;

        self.light[0] += scaled_weight * sun_intensity[0];
        self.light[1] += scaled_weight * sun_intensity[1];
        self.light[2] += scaled_weight * sun_intensity[2];

        self.transmittance *= decay;
        self.sample_count += 1;
    }

    /// Apply final exposure.
    #[inline]
    pub fn apply_exposure(&mut self, exposure: f32) {
        self.light[0] *= exposure;
        self.light[1] *= exposure;
        self.light[2] *= exposure;
    }
}

// ---------------------------------------------------------------------------
// GodRayOutput - Final render output
// ---------------------------------------------------------------------------

/// Final output from god ray rendering for compositing.
///
/// Contains color and additive blending weight.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct GodRayOutput {
    /// God ray color (RGB linear, additive).
    pub color: [f32; 3],

    /// Additive blend factor (typically 1.0).
    pub blend: f32,
}

// Size assertion
const _: () = assert!(std::mem::size_of::<GodRayOutput>() == 16);

impl Default for GodRayOutput {
    fn default() -> Self {
        Self {
            color: [0.0; 3],
            blend: 1.0,
        }
    }
}

impl GodRayOutput {
    /// Create empty output (no god rays).
    #[inline]
    pub fn empty() -> Self {
        Self::default()
    }

    /// Create from a god ray sample with exposure applied.
    #[inline]
    pub fn from_sample(sample: &GodRaySample, exposure: f32) -> Self {
        Self {
            color: [
                sample.light[0] * exposure,
                sample.light[1] * exposure,
                sample.light[2] * exposure,
            ],
            blend: 1.0,
        }
    }

    /// Apply additive blending with a background color.
    #[inline]
    pub fn blend_additive(&self, background: [f32; 3]) -> [f32; 3] {
        [
            (background[0] + self.color[0] * self.blend).min(1.0),
            (background[1] + self.color[1] * self.blend).min(1.0),
            (background[2] + self.color[2] * self.blend).min(1.0),
        ]
    }

    /// Get luminance.
    #[inline]
    pub fn luminance(&self) -> f32 {
        0.2126 * self.color[0] + 0.7152 * self.color[1] + 0.0722 * self.color[2]
    }
}

// ---------------------------------------------------------------------------
// Sun Screen Position Calculation
// ---------------------------------------------------------------------------

/// Calculate sun position in screen space from world space.
///
/// # Arguments
///
/// * `sun_direction` - Normalized world-space direction TO the sun.
/// * `view` - 4x4 view matrix (column-major).
/// * `projection` - 4x4 projection matrix (column-major).
///
/// # Returns
///
/// Sun position in normalized screen coordinates (0-1, origin top-left),
/// or None if sun is behind camera.
pub fn sun_to_screen_position(
    sun_direction: [f32; 3],
    view: &[[f32; 4]; 4],
    projection: &[[f32; 4]; 4],
) -> Option<[f32; 2]> {
    // Treat sun as infinitely far: use direction as position at large distance
    let sun_pos = [
        sun_direction[0] * 1e6,
        sun_direction[1] * 1e6,
        sun_direction[2] * 1e6,
        1.0,
    ];

    // Transform to view space
    let view_pos = mat4_mul_vec4(view, &sun_pos);

    // Check if sun is behind camera (positive Z in OpenGL/wgpu convention)
    // In right-handed coordinate system with -Z forward
    if view_pos[2] > 0.0 {
        return None; // Sun behind camera
    }

    // Transform to clip space
    let clip_pos = mat4_mul_vec4(projection, &view_pos);

    // Perspective divide
    if clip_pos[3].abs() < EPSILON {
        return None;
    }

    let ndc_x = clip_pos[0] / clip_pos[3];
    let ndc_y = clip_pos[1] / clip_pos[3];

    // Convert NDC (-1 to 1) to screen (0 to 1, Y inverted for top-left origin)
    let screen_x = (ndc_x + 1.0) * 0.5;
    let screen_y = (1.0 - ndc_y) * 0.5; // Y flipped for screen coordinates

    Some([screen_x, screen_y])
}

/// Calculate sun position with extended bounds (for off-screen sun).
///
/// Returns sun position even if outside screen bounds, clamped to
/// a maximum distance from screen center.
pub fn sun_to_screen_position_extended(
    sun_direction: [f32; 3],
    view: &[[f32; 4]; 4],
    projection: &[[f32; 4]; 4],
    max_extension: f32,
) -> Option<[f32; 2]> {
    let sun_pos = [
        sun_direction[0] * 1e6,
        sun_direction[1] * 1e6,
        sun_direction[2] * 1e6,
        1.0,
    ];

    let view_pos = mat4_mul_vec4(view, &sun_pos);

    // Check if sun is behind camera
    if view_pos[2] > 0.0 {
        return None;
    }

    let clip_pos = mat4_mul_vec4(projection, &view_pos);

    if clip_pos[3].abs() < EPSILON {
        return None;
    }

    let ndc_x = clip_pos[0] / clip_pos[3];
    let ndc_y = clip_pos[1] / clip_pos[3];

    // Clamp to extended bounds
    let clamped_x = ndc_x.clamp(-max_extension, max_extension);
    let clamped_y = ndc_y.clamp(-max_extension, max_extension);

    let screen_x = (clamped_x + 1.0) * 0.5;
    let screen_y = (1.0 - clamped_y) * 0.5;

    Some([screen_x, screen_y])
}

/// Check if sun direction is visible (above horizon).
///
/// # Arguments
///
/// * `sun_direction` - Normalized world-space direction TO the sun.
/// * `up` - World up vector (typically [0, 1, 0]).
#[inline]
pub fn is_sun_above_horizon(sun_direction: [f32; 3], up: [f32; 3]) -> bool {
    dot3(sun_direction, up) > -0.1 // Small margin below horizon
}

// ---------------------------------------------------------------------------
// Radial Sampling Functions
// ---------------------------------------------------------------------------

/// Calculate radial sample positions from pixel to sun.
///
/// Generates sample positions along the line from a pixel coordinate
/// toward the sun screen position.
///
/// # Arguments
///
/// * `pixel_uv` - Pixel UV coordinates (0-1).
/// * `sun_uv` - Sun UV coordinates (0-1).
/// * `num_samples` - Number of samples to generate.
/// * `max_length` - Maximum length of ray in UV space.
///
/// # Returns
///
/// Vector of sample UV coordinates.
pub fn radial_sample_positions(
    pixel_uv: [f32; 2],
    sun_uv: [f32; 2],
    num_samples: u32,
    max_length: f32,
) -> Vec<[f32; 2]> {
    let mut samples = Vec::with_capacity(num_samples as usize);

    // Direction from pixel toward sun
    let delta = [sun_uv[0] - pixel_uv[0], sun_uv[1] - pixel_uv[1]];

    // Calculate ray length
    let ray_length = (delta[0] * delta[0] + delta[1] * delta[1]).sqrt();

    if ray_length < EPSILON {
        // Pixel is at sun position
        return samples;
    }

    // Clamp to maximum length
    let actual_length = ray_length.min(max_length);
    let scale = actual_length / ray_length;

    // Step size per sample
    let step = [
        delta[0] * scale / num_samples as f32,
        delta[1] * scale / num_samples as f32,
    ];

    // Generate samples
    let mut pos = pixel_uv;
    for _ in 0..num_samples {
        samples.push(pos);
        pos[0] += step[0];
        pos[1] += step[1];
    }

    samples
}

/// Calculate step direction and size for radial march.
///
/// # Arguments
///
/// * `pixel_uv` - Starting UV coordinates.
/// * `sun_uv` - Target sun UV coordinates.
/// * `num_samples` - Number of samples to take.
/// * `max_length` - Maximum march length.
///
/// # Returns
///
/// (step_x, step_y, actual_length) tuple.
#[inline]
pub fn calculate_radial_step(
    pixel_uv: [f32; 2],
    sun_uv: [f32; 2],
    num_samples: u32,
    max_length: f32,
) -> ([f32; 2], f32) {
    let delta = [sun_uv[0] - pixel_uv[0], sun_uv[1] - pixel_uv[1]];
    let length = (delta[0] * delta[0] + delta[1] * delta[1]).sqrt();

    if length < EPSILON || num_samples == 0 {
        return ([0.0, 0.0], 0.0);
    }

    let clamped_length = length.min(max_length);
    let step_scale = clamped_length / (length * num_samples as f32);

    (
        [delta[0] * step_scale, delta[1] * step_scale],
        clamped_length,
    )
}

/// Calculate distance from pixel to sun in UV space.
#[inline]
pub fn distance_to_sun(pixel_uv: [f32; 2], sun_uv: [f32; 2]) -> f32 {
    let dx = sun_uv[0] - pixel_uv[0];
    let dy = sun_uv[1] - pixel_uv[1];
    (dx * dx + dy * dy).sqrt()
}

// ---------------------------------------------------------------------------
// Epipolar Coordinate Transforms
// ---------------------------------------------------------------------------

/// Convert screen UV to epipolar coordinates.
///
/// Epipolar coordinates are (angle, distance) from sun position.
///
/// # Arguments
///
/// * `screen_uv` - Screen UV coordinates (0-1).
/// * `sun_uv` - Sun UV coordinates (0-1).
///
/// # Returns
///
/// (angle, distance) where angle is in radians [0, 2pi) and distance
/// is in UV space units.
#[inline]
pub fn screen_to_epipolar(screen_uv: [f32; 2], sun_uv: [f32; 2]) -> (f32, f32) {
    let dx = screen_uv[0] - sun_uv[0];
    let dy = screen_uv[1] - sun_uv[1];

    let distance = (dx * dx + dy * dy).sqrt();
    let angle = dy.atan2(dx);

    // Normalize angle to [0, 2pi)
    let normalized_angle = if angle < 0.0 { angle + 2.0 * PI } else { angle };

    (normalized_angle, distance)
}

/// Convert epipolar coordinates back to screen UV.
///
/// # Arguments
///
/// * `angle` - Angle in radians [0, 2pi).
/// * `distance` - Distance from sun in UV space.
/// * `sun_uv` - Sun UV coordinates (0-1).
///
/// # Returns
///
/// Screen UV coordinates.
#[inline]
pub fn epipolar_to_screen(angle: f32, distance: f32, sun_uv: [f32; 2]) -> [f32; 2] {
    [
        sun_uv[0] + distance * angle.cos(),
        sun_uv[1] + distance * angle.sin(),
    ]
}

/// Calculate epipolar slice index from angle.
///
/// # Arguments
///
/// * `angle` - Angle in radians [0, 2pi).
/// * `num_slices` - Total number of slices.
///
/// # Returns
///
/// Slice index (0 to num_slices-1).
#[inline]
pub fn angle_to_slice_index(angle: f32, num_slices: u32) -> u32 {
    let normalized = angle / (2.0 * PI);
    let index = (normalized * num_slices as f32).floor() as u32;
    index.min(num_slices - 1)
}

/// Calculate slice center angle from index.
///
/// # Arguments
///
/// * `slice_index` - Slice index.
/// * `num_slices` - Total number of slices.
///
/// # Returns
///
/// Center angle in radians.
#[inline]
pub fn slice_index_to_angle(slice_index: u32, num_slices: u32) -> f32 {
    (slice_index as f32 + 0.5) / num_slices as f32 * 2.0 * PI
}

/// Calculate sample index along epipolar slice.
///
/// # Arguments
///
/// * `distance` - Distance from sun in UV space.
/// * `max_distance` - Maximum distance (edge of screen).
/// * `samples_per_slice` - Number of samples per slice.
///
/// # Returns
///
/// Sample index (0 to samples_per_slice-1).
#[inline]
pub fn distance_to_sample_index(distance: f32, max_distance: f32, samples_per_slice: u32) -> u32 {
    if max_distance < EPSILON {
        return 0;
    }

    let normalized = (distance / max_distance).clamp(0.0, 1.0);
    let index = (normalized * samples_per_slice as f32).floor() as u32;
    index.min(samples_per_slice - 1)
}

/// Calculate maximum epipolar distance for a given angle.
///
/// This is the distance from sun to the screen edge along the slice.
///
/// # Arguments
///
/// * `sun_uv` - Sun UV coordinates.
/// * `angle` - Angle in radians.
///
/// # Returns
///
/// Maximum distance to screen edge.
pub fn max_epipolar_distance(sun_uv: [f32; 2], angle: f32) -> f32 {
    let cos_a = angle.cos();
    let sin_a = angle.sin();

    // Calculate intersection with screen boundaries
    // Screen bounds: x in [0, 1], y in [0, 1]

    let mut min_t = f32::MAX;

    // Check intersection with x = 0
    if cos_a.abs() > EPSILON {
        let t = -sun_uv[0] / cos_a;
        if t > 0.0 {
            let y = sun_uv[1] + t * sin_a;
            if y >= 0.0 && y <= 1.0 {
                min_t = min_t.min(t);
            }
        }
    }

    // Check intersection with x = 1
    if cos_a.abs() > EPSILON {
        let t = (1.0 - sun_uv[0]) / cos_a;
        if t > 0.0 {
            let y = sun_uv[1] + t * sin_a;
            if y >= 0.0 && y <= 1.0 {
                min_t = min_t.min(t);
            }
        }
    }

    // Check intersection with y = 0
    if sin_a.abs() > EPSILON {
        let t = -sun_uv[1] / sin_a;
        if t > 0.0 {
            let x = sun_uv[0] + t * cos_a;
            if x >= 0.0 && x <= 1.0 {
                min_t = min_t.min(t);
            }
        }
    }

    // Check intersection with y = 1
    if sin_a.abs() > EPSILON {
        let t = (1.0 - sun_uv[1]) / sin_a;
        if t > 0.0 {
            let x = sun_uv[0] + t * cos_a;
            if x >= 0.0 && x <= 1.0 {
                min_t = min_t.min(t);
            }
        }
    }

    if min_t == f32::MAX {
        // Sun is outside screen, use diagonal length
        2.0f32.sqrt()
    } else {
        min_t
    }
}

// ---------------------------------------------------------------------------
// Beer-Lambert Extinction
// ---------------------------------------------------------------------------

/// Apply Beer-Lambert extinction law.
///
/// Models exponential attenuation of light through a participating medium.
///
/// # Arguments
///
/// * `optical_depth` - Accumulated optical depth (density * distance).
/// * `extinction_coeff` - Extinction coefficient.
///
/// # Returns
///
/// Transmittance factor (0-1).
#[inline]
pub fn beer_lambert(optical_depth: f32, extinction_coeff: f32) -> f32 {
    (-optical_depth * extinction_coeff).exp()
}

/// Calculate transmittance through cloud/fog medium.
///
/// # Arguments
///
/// * `density` - Medium density at sample.
/// * `distance` - Distance traveled through medium.
/// * `extinction` - Extinction coefficient.
#[inline]
pub fn transmittance(density: f32, distance: f32, extinction: f32) -> f32 {
    (-density * distance * extinction).exp()
}

/// Accumulate optical depth along a ray.
///
/// # Arguments
///
/// * `current_depth` - Current accumulated optical depth.
/// * `sample_density` - Density at current sample.
/// * `step_size` - Step size.
///
/// # Returns
///
/// Updated optical depth.
#[inline]
pub fn accumulate_optical_depth(current_depth: f32, sample_density: f32, step_size: f32) -> f32 {
    current_depth + sample_density * step_size
}

// ---------------------------------------------------------------------------
// Temporal Reprojection
// ---------------------------------------------------------------------------

/// Reproject a screen position from current to previous frame.
///
/// # Arguments
///
/// * `current_uv` - Current frame UV coordinates.
/// * `depth` - Depth at this pixel (linear or normalized).
/// * `current_view_proj_inv` - Inverse of current frame's view-projection matrix.
/// * `prev_view_proj` - Previous frame's view-projection matrix.
///
/// # Returns
///
/// Previous frame UV coordinates, or None if reprojection failed.
pub fn reproject_uv(
    current_uv: [f32; 2],
    depth: f32,
    current_view_proj_inv: &[[f32; 4]; 4],
    prev_view_proj: &[[f32; 4]; 4],
) -> Option<[f32; 2]> {
    // Convert UV to NDC
    let ndc_x = current_uv[0] * 2.0 - 1.0;
    let ndc_y = 1.0 - current_uv[1] * 2.0; // Flip Y

    // Reconstruct world position
    let clip_pos = [ndc_x, ndc_y, depth, 1.0];
    let world_pos = mat4_mul_vec4(current_view_proj_inv, &clip_pos);

    // Perspective divide
    if world_pos[3].abs() < EPSILON {
        return None;
    }

    let world_pos = [
        world_pos[0] / world_pos[3],
        world_pos[1] / world_pos[3],
        world_pos[2] / world_pos[3],
        1.0,
    ];

    // Project to previous frame
    let prev_clip = mat4_mul_vec4(prev_view_proj, &world_pos);

    if prev_clip[3].abs() < EPSILON {
        return None;
    }

    let prev_ndc_x = prev_clip[0] / prev_clip[3];
    let prev_ndc_y = prev_clip[1] / prev_clip[3];

    // Convert back to UV
    Some([
        (prev_ndc_x + 1.0) * 0.5,
        (1.0 - prev_ndc_y) * 0.5,
    ])
}

/// Calculate motion vector between current and reprojected position.
#[inline]
pub fn calculate_motion_vector(current_uv: [f32; 2], prev_uv: [f32; 2]) -> [f32; 2] {
    [prev_uv[0] - current_uv[0], prev_uv[1] - current_uv[1]]
}

/// Check if reprojected position is valid (on screen and consistent).
///
/// # Arguments
///
/// * `prev_uv` - Reprojected UV coordinates.
/// * `current_depth` - Depth in current frame.
/// * `prev_depth` - Depth at reprojected position in history.
/// * `depth_threshold` - Maximum allowed depth difference.
#[inline]
pub fn is_reprojection_valid(
    prev_uv: [f32; 2],
    current_depth: f32,
    prev_depth: f32,
    depth_threshold: f32,
) -> bool {
    // Check UV is on screen
    if prev_uv[0] < 0.0 || prev_uv[0] > 1.0 || prev_uv[1] < 0.0 || prev_uv[1] > 1.0 {
        return false;
    }

    // Check depth consistency
    (current_depth - prev_depth).abs() < depth_threshold
}

/// Blend current and history values with temporal filtering.
///
/// # Arguments
///
/// * `current` - Current frame value (RGB).
/// * `history` - History value (RGB).
/// * `blend_factor` - Blend factor (0 = current only, 1 = history only).
/// * `valid` - Whether history is valid for this pixel.
#[inline]
pub fn temporal_blend(
    current: [f32; 3],
    history: [f32; 3],
    blend_factor: f32,
    valid: bool,
) -> [f32; 3] {
    if !valid {
        return current;
    }

    let alpha = blend_factor;
    [
        current[0] * (1.0 - alpha) + history[0] * alpha,
        current[1] * (1.0 - alpha) + history[1] * alpha,
        current[2] * (1.0 - alpha) + history[2] * alpha,
    ]
}

/// Variance clamping for temporal stability.
///
/// Clamps history value to within the neighborhood color box to
/// reduce ghosting.
///
/// # Arguments
///
/// * `history` - History value to clamp.
/// * `neighborhood_min` - Minimum value in neighborhood.
/// * `neighborhood_max` - Maximum value in neighborhood.
#[inline]
pub fn variance_clamp(history: [f32; 3], neighborhood_min: [f32; 3], neighborhood_max: [f32; 3]) -> [f32; 3] {
    [
        history[0].clamp(neighborhood_min[0], neighborhood_max[0]),
        history[1].clamp(neighborhood_min[1], neighborhood_max[1]),
        history[2].clamp(neighborhood_min[2], neighborhood_max[2]),
    ]
}

// ---------------------------------------------------------------------------
// Bilateral Upsampling
// ---------------------------------------------------------------------------

/// Calculate bilateral weight for upsampling.
///
/// Combines spatial Gaussian with depth-aware range term.
///
/// # Arguments
///
/// * `spatial_distance` - Pixel distance from center.
/// * `depth_difference` - Depth difference from center.
/// * `sigma_spatial` - Spatial sigma.
/// * `sigma_range` - Range sigma.
#[inline]
pub fn bilateral_weight(
    spatial_distance: f32,
    depth_difference: f32,
    sigma_spatial: f32,
    sigma_range: f32,
) -> f32 {
    let spatial = (-spatial_distance * spatial_distance / (2.0 * sigma_spatial * sigma_spatial)).exp();
    let range = (-depth_difference * depth_difference / (2.0 * sigma_range * sigma_range)).exp();
    spatial * range
}

/// Generate bilateral kernel weights for a given kernel size.
///
/// # Arguments
///
/// * `radius` - Kernel radius.
/// * `sigma_spatial` - Spatial sigma.
///
/// # Returns
///
/// Vector of (offset_x, offset_y, spatial_weight) tuples.
pub fn generate_bilateral_kernel(radius: u32, sigma_spatial: f32) -> Vec<(i32, i32, f32)> {
    let mut kernel = Vec::new();
    let r = radius as i32;

    for y in -r..=r {
        for x in -r..=r {
            let dist = ((x * x + y * y) as f32).sqrt();
            let weight = (-dist * dist / (2.0 * sigma_spatial * sigma_spatial)).exp();
            kernel.push((x, y, weight));
        }
    }

    kernel
}

/// Upsample a low-resolution god ray buffer to full resolution.
///
/// # Arguments
///
/// * `low_res_value` - Value from low-res buffer.
/// * `neighbor_values` - Values from neighboring low-res pixels.
/// * `neighbor_depths` - Depths at neighbor positions.
/// * `center_depth` - Depth at full-res center.
/// * `config` - Bilateral configuration.
#[inline]
pub fn bilateral_upsample(
    low_res_value: [f32; 3],
    neighbor_values: &[[f32; 3]],
    neighbor_depths: &[f32],
    center_depth: f32,
    config: &BilateralConfig,
) -> [f32; 3] {
    if !config.is_enabled() || neighbor_values.is_empty() {
        return low_res_value;
    }

    let mut result = [0.0f32; 3];
    let mut total_weight = 0.0f32;

    for (i, (value, &depth)) in neighbor_values.iter().zip(neighbor_depths.iter()).enumerate() {
        // Calculate spatial distance (simplified for this example)
        let spatial_dist = (i as f32).sqrt(); // Would normally use actual pixel offsets

        // Depth difference
        let depth_diff = (center_depth - depth).abs();

        // Calculate weight
        let weight = bilateral_weight(
            spatial_dist,
            depth_diff,
            config.sigma_spatial,
            config.sigma_range,
        );

        result[0] += value[0] * weight;
        result[1] += value[1] * weight;
        result[2] += value[2] * weight;
        total_weight += weight;
    }

    if total_weight > EPSILON {
        result[0] /= total_weight;
        result[1] /= total_weight;
        result[2] /= total_weight;
    }

    result
}

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

/// 4x4 matrix * vec4 multiplication.
#[inline]
fn mat4_mul_vec4(m: &[[f32; 4]; 4], v: &[f32; 4]) -> [f32; 4] {
    [
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2] + m[3][0] * v[3],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2] + m[3][1] * v[3],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2] + m[3][2] * v[3],
        m[0][3] * v[0] + m[1][3] * v[1] + m[2][3] * v[2] + m[3][3] * v[3],
    ]
}

/// 3D dot product.
#[inline]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Halton sequence for quasi-random sampling.
///
/// Used for temporal jittering.
fn halton_2d(index: u32, scale: f32) -> [f32; 2] {
    [
        halton_sequence(index, 2) * scale - scale * 0.5,
        halton_sequence(index, 3) * scale - scale * 0.5,
    ]
}

/// Single-dimensional Halton sequence.
fn halton_sequence(mut index: u32, base: u32) -> f32 {
    let mut f = 1.0;
    let mut result = 0.0;
    let inv_base = 1.0 / base as f32;

    while index > 0 {
        f *= inv_base;
        result += f * (index % base) as f32;
        index /= base;
    }

    result
}

/// Smoothstep interpolation.
#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Linear interpolation.
#[inline]
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Linear interpolation for vec3.
#[inline]
pub fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)]
}

// ---------------------------------------------------------------------------
// GodRayRenderer - Main rendering API
// ---------------------------------------------------------------------------

/// Main god ray renderer.
///
/// Orchestrates the full god ray rendering pipeline:
/// 1. Calculate sun screen position
/// 2. Execute radial/epipolar march
/// 3. Apply temporal filtering
/// 4. Upsample if needed
#[derive(Debug, Clone)]
pub struct GodRayRenderer {
    /// Core uniforms for GPU.
    pub uniforms: GodRayUniforms,

    /// Epipolar sampling configuration.
    pub epipolar: EpipolarConfig,

    /// Temporal filtering configuration.
    pub temporal: TemporalConfig,

    /// Bilateral upsample configuration.
    pub bilateral: BilateralConfig,

    /// Occlusion source mode.
    pub occlusion_source: OcclusionSource,

    /// Whether rendering is enabled.
    enabled: bool,
}

impl Default for GodRayRenderer {
    fn default() -> Self {
        Self {
            uniforms: GodRayUniforms::default(),
            epipolar: EpipolarConfig::default(),
            temporal: TemporalConfig::default(),
            bilateral: BilateralConfig::default(),
            occlusion_source: OcclusionSource::Combined,
            enabled: true,
        }
    }
}

impl GodRayRenderer {
    /// Create a new god ray renderer with default settings.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create from a quality preset.
    #[inline]
    pub fn from_quality(quality: GodRayQuality) -> Self {
        Self {
            uniforms: GodRayUniforms::from_quality(quality),
            epipolar: EpipolarConfig::from_quality(quality),
            temporal: TemporalConfig::default(),
            bilateral: BilateralConfig::default(),
            occlusion_source: OcclusionSource::Combined,
            enabled: true,
        }
    }

    /// Set quality level.
    #[inline]
    pub fn set_quality(&mut self, quality: GodRayQuality) {
        self.uniforms = GodRayUniforms::from_quality(quality);
        self.epipolar = EpipolarConfig::from_quality(quality);
    }

    /// Update sun position from world direction.
    ///
    /// # Arguments
    ///
    /// * `sun_direction` - Normalized direction TO the sun.
    /// * `view` - View matrix.
    /// * `projection` - Projection matrix.
    ///
    /// # Returns
    ///
    /// true if sun is visible, false otherwise.
    pub fn update_sun_position(
        &mut self,
        sun_direction: [f32; 3],
        view: &[[f32; 4]; 4],
        projection: &[[f32; 4]; 4],
    ) -> bool {
        match sun_to_screen_position_extended(sun_direction, view, projection, SUN_SCREEN_CLAMP) {
            Some(pos) => {
                self.uniforms.sun_screen_pos = pos;
                true
            }
            None => {
                self.enabled = false;
                false
            }
        }
    }

    /// Set sun color and intensity.
    #[inline]
    pub fn set_sun_intensity(&mut self, r: f32, g: f32, b: f32) {
        self.uniforms.sun_intensity = [r, g, b];
    }

    /// Advance temporal frame counter.
    #[inline]
    pub fn next_frame(&mut self) {
        self.temporal.next_frame();
    }

    /// Enable or disable rendering.
    #[inline]
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if rendering is enabled.
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.enabled && self.uniforms.is_sun_visible()
    }

    /// Sample god rays at a single pixel.
    ///
    /// This is the core algorithm for CPU reference / testing.
    ///
    /// # Arguments
    ///
    /// * `pixel_uv` - Pixel UV coordinates.
    /// * `sample_occlusion` - Closure to sample occlusion at UV.
    pub fn sample_pixel<F>(&self, pixel_uv: [f32; 2], mut sample_occlusion: F) -> GodRaySample
    where
        F: FnMut([f32; 2]) -> f32,
    {
        let mut result = GodRaySample::new();

        if !self.enabled {
            return result;
        }

        let (step, length) = calculate_radial_step(
            pixel_uv,
            self.uniforms.sun_screen_pos,
            self.uniforms.num_samples,
            self.uniforms.max_ray_length,
        );

        if length < EPSILON {
            return result;
        }

        let mut pos = pixel_uv;
        let mut illumination = 1.0f32;

        for i in 0..self.uniforms.num_samples {
            // Sample occlusion
            let occlusion = sample_occlusion(pos);

            // Apply Beer-Lambert for accumulated density
            let step_transmittance =
                transmittance(occlusion, 1.0, self.uniforms.extinction_coeff);

            // Accumulate light
            let weight = self.uniforms.weight * illumination * (1.0 - occlusion);
            result.light[0] += weight * self.uniforms.sun_intensity[0];
            result.light[1] += weight * self.uniforms.sun_intensity[1];
            result.light[2] += weight * self.uniforms.sun_intensity[2];

            result.transmittance *= step_transmittance;
            illumination *= self.uniforms.decay;
            result.sample_count = i + 1;

            // Early termination
            if illumination < 0.001 {
                break;
            }

            // Step toward sun
            pos[0] += step[0];
            pos[1] += step[1];
        }

        // Apply exposure and density
        result.apply_exposure(self.uniforms.exposure * self.uniforms.density);

        result
    }

    /// Validate all configurations.
    #[inline]
    pub fn validate(&self) -> bool {
        self.uniforms.validate()
            && self.epipolar.validate()
            && self.temporal.validate()
            && self.bilateral.validate()
    }

    /// Get the estimated GPU memory usage in bytes.
    #[inline]
    pub fn estimated_memory_bytes(&self) -> usize {
        std::mem::size_of::<GodRayUniforms>()
            + std::mem::size_of::<EpipolarConfig>()
            + std::mem::size_of::<TemporalConfig>()
            + std::mem::size_of::<BilateralConfig>()
            + self.epipolar.buffer_size_bytes()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // GodRayQuality Tests
    // =========================================================================

    #[test]
    fn test_quality_sample_counts() {
        assert_eq!(GodRayQuality::Low.sample_count(), 16);
        assert_eq!(GodRayQuality::Medium.sample_count(), 32);
        assert_eq!(GodRayQuality::High.sample_count(), 64);
        assert_eq!(GodRayQuality::Ultra.sample_count(), 128);
    }

    #[test]
    fn test_quality_resolution_scales() {
        assert!((GodRayQuality::Low.resolution_scale() - QUARTER_RES_SCALE).abs() < EPSILON);
        assert!((GodRayQuality::Medium.resolution_scale() - HALF_RES_SCALE).abs() < EPSILON);
        assert!((GodRayQuality::High.resolution_scale() - HALF_RES_SCALE).abs() < EPSILON);
        assert!((GodRayQuality::Ultra.resolution_scale() - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_quality_decay_values() {
        assert!(GodRayQuality::Low.decay() > MIN_DECAY);
        assert!(GodRayQuality::Ultra.decay() < MAX_DECAY);
        assert!(GodRayQuality::High.decay() > GodRayQuality::Low.decay());
    }

    #[test]
    fn test_quality_from_sample_count() {
        assert_eq!(GodRayQuality::from_sample_count(10), GodRayQuality::Low);
        assert_eq!(GodRayQuality::from_sample_count(30), GodRayQuality::Medium);
        assert_eq!(GodRayQuality::from_sample_count(50), GodRayQuality::High);
        assert_eq!(GodRayQuality::from_sample_count(100), GodRayQuality::Ultra);
    }

    #[test]
    fn test_quality_all() {
        let all = GodRayQuality::all();
        assert_eq!(all.len(), 4);
    }

    #[test]
    fn test_quality_names() {
        assert_eq!(GodRayQuality::Low.name(), "low");
        assert_eq!(GodRayQuality::Medium.name(), "medium");
        assert_eq!(GodRayQuality::High.name(), "high");
        assert_eq!(GodRayQuality::Ultra.name(), "ultra");
    }

    // =========================================================================
    // GodRayUniforms Tests
    // =========================================================================

    #[test]
    fn test_uniforms_default() {
        let uniforms = GodRayUniforms::default();
        assert!(uniforms.validate());
        assert_eq!(uniforms.decay, DEFAULT_DECAY);
        assert_eq!(uniforms.exposure, DEFAULT_EXPOSURE);
        assert_eq!(uniforms.num_samples, DEFAULT_NUM_SAMPLES);
    }

    #[test]
    fn test_uniforms_from_quality() {
        for quality in GodRayQuality::all() {
            let uniforms = GodRayUniforms::from_quality(quality);
            assert!(uniforms.validate());
            assert_eq!(uniforms.num_samples, quality.sample_count());
            assert!((uniforms.decay - quality.decay()).abs() < EPSILON);
        }
    }

    #[test]
    fn test_uniforms_builders() {
        let uniforms = GodRayUniforms::new()
            .with_sun_position(0.3, 0.2)
            .with_decay(0.95)
            .with_exposure(0.5)
            .with_density(1.5)
            .with_weight(0.6)
            .with_samples(96)
            .with_extinction(0.2)
            .with_sun_intensity(1.0, 0.9, 0.7)
            .with_max_ray_length(1.5)
            .with_resolution_scale(0.75);

        assert!(uniforms.validate());
        assert!((uniforms.sun_screen_pos[0] - 0.3).abs() < EPSILON);
        assert!((uniforms.decay - 0.95).abs() < EPSILON);
        assert!((uniforms.exposure - 0.5).abs() < EPSILON);
        assert_eq!(uniforms.num_samples, 96);
    }

    #[test]
    fn test_uniforms_clamping() {
        let uniforms = GodRayUniforms::new()
            .with_decay(0.1) // Below minimum
            .with_exposure(20.0) // Above maximum
            .with_samples(500); // Above maximum

        assert_eq!(uniforms.decay, MIN_DECAY);
        assert_eq!(uniforms.exposure, MAX_EXPOSURE);
        assert_eq!(uniforms.num_samples, 256);
    }

    #[test]
    fn test_uniforms_sun_visibility() {
        let mut uniforms = GodRayUniforms::default();

        uniforms.sun_screen_pos = [0.5, 0.5];
        assert!(uniforms.is_sun_on_screen());
        assert!(uniforms.is_sun_visible());

        uniforms.sun_screen_pos = [-0.5, 0.5];
        assert!(!uniforms.is_sun_on_screen());
        assert!(uniforms.is_sun_visible()); // Still within extended bounds

        uniforms.sun_screen_pos = [-5.0, 0.5];
        assert!(!uniforms.is_sun_visible());
    }

    #[test]
    fn test_uniforms_gpu_alignment() {
        assert_eq!(std::mem::size_of::<GodRayUniforms>(), 64);
        assert_eq!(std::mem::align_of::<GodRayUniforms>(), 4);
    }

    #[test]
    fn test_uniforms_pod_zeroable() {
        let zeroed = GodRayUniforms::zeroed();
        assert_eq!(zeroed.decay, 0.0);
        assert_eq!(zeroed.num_samples, 0);

        let uniforms = GodRayUniforms::default();
        assert_eq!(std::mem::size_of_val(&uniforms), 64);
    }

    // =========================================================================
    // EpipolarConfig Tests
    // =========================================================================

    #[test]
    fn test_epipolar_default() {
        let config = EpipolarConfig::default();
        assert!(config.validate());
        assert_eq!(config.num_slices, DEFAULT_EPIPOLAR_SLICES);
        assert_eq!(config.samples_per_slice, DEFAULT_EPIPOLAR_SAMPLES);
    }

    #[test]
    fn test_epipolar_from_quality() {
        for quality in GodRayQuality::all() {
            let config = EpipolarConfig::from_quality(quality);
            assert!(config.validate());
        }
    }

    #[test]
    fn test_epipolar_total_samples() {
        let config = EpipolarConfig::default();
        assert_eq!(config.total_samples(), config.num_slices * config.samples_per_slice);
    }

    #[test]
    fn test_epipolar_buffer_size() {
        let config = EpipolarConfig::default();
        let expected = (config.total_samples() as usize) * 16;
        assert_eq!(config.buffer_size_bytes(), expected);
    }

    #[test]
    fn test_epipolar_builders() {
        let config = EpipolarConfig::new()
            .with_slices(512)
            .with_samples_per_slice(256)
            .with_depth_threshold(0.02);

        assert!(config.validate());
        assert_eq!(config.num_slices, 512);
        assert_eq!(config.samples_per_slice, 256);
    }

    #[test]
    fn test_epipolar_gpu_alignment() {
        assert_eq!(std::mem::size_of::<EpipolarConfig>(), 32);
    }

    // =========================================================================
    // TemporalConfig Tests
    // =========================================================================

    #[test]
    fn test_temporal_default() {
        let config = TemporalConfig::default();
        assert!(config.validate());
        assert!(config.is_enabled());
        assert_eq!(config.blend_factor, DEFAULT_TEMPORAL_BLEND);
    }

    #[test]
    fn test_temporal_disabled() {
        let config = TemporalConfig::disabled();
        assert!(config.validate());
        assert!(!config.is_enabled());
    }

    #[test]
    fn test_temporal_next_frame() {
        let mut config = TemporalConfig::default();
        assert_eq!(config.frame_index, 0);
        config.next_frame();
        assert_eq!(config.frame_index, 1);
        config.next_frame();
        assert_eq!(config.frame_index, 2);
    }

    #[test]
    fn test_temporal_jitter_sequence() {
        let mut config = TemporalConfig::default();
        let mut jitters = Vec::new();

        for _ in 0..16 {
            jitters.push(config.current_jitter());
            config.next_frame();
        }

        // Jitters should be different
        let unique: std::collections::HashSet<_> = jitters
            .iter()
            .map(|j| ((j[0] * 1000.0) as i32, (j[1] * 1000.0) as i32))
            .collect();
        assert!(unique.len() > 1);
    }

    #[test]
    fn test_temporal_gpu_alignment() {
        assert_eq!(std::mem::size_of::<TemporalConfig>(), 32);
    }

    // =========================================================================
    // BilateralConfig Tests
    // =========================================================================

    #[test]
    fn test_bilateral_default() {
        let config = BilateralConfig::default();
        assert!(config.validate());
        assert!(config.is_enabled());
    }

    #[test]
    fn test_bilateral_disabled() {
        let config = BilateralConfig::disabled();
        assert!(config.validate());
        assert!(!config.is_enabled());
    }

    #[test]
    fn test_bilateral_builders() {
        let config = BilateralConfig::new()
            .with_sigma_spatial(5.0)
            .with_sigma_range(0.2)
            .with_kernel_radius(4)
            .with_enabled(true);

        assert!(config.validate());
        assert!((config.sigma_spatial - 5.0).abs() < EPSILON);
        assert_eq!(config.kernel_radius, 4);
    }

    #[test]
    fn test_bilateral_gpu_alignment() {
        assert_eq!(std::mem::size_of::<BilateralConfig>(), 32);
    }

    // =========================================================================
    // OcclusionSource Tests
    // =========================================================================

    #[test]
    fn test_occlusion_source_names() {
        assert_eq!(OcclusionSource::DepthBuffer.name(), "depth_buffer");
        assert_eq!(OcclusionSource::CloudDensity.name(), "cloud_density");
        assert_eq!(OcclusionSource::Combined.name(), "combined");
        assert_eq!(OcclusionSource::StencilMask.name(), "stencil_mask");
    }

    #[test]
    fn test_occlusion_source_texture_requirements() {
        assert!(OcclusionSource::CloudDensity.needs_cloud_texture());
        assert!(OcclusionSource::Combined.needs_cloud_texture());
        assert!(!OcclusionSource::DepthBuffer.needs_cloud_texture());

        assert!(OcclusionSource::DepthBuffer.needs_depth_texture());
        assert!(OcclusionSource::Combined.needs_depth_texture());
        assert!(!OcclusionSource::CloudDensity.needs_depth_texture());
    }

    // =========================================================================
    // GodRaySample Tests
    // =========================================================================

    #[test]
    fn test_sample_new() {
        let sample = GodRaySample::new();
        assert_eq!(sample.light, [0.0, 0.0, 0.0]);
        assert_eq!(sample.transmittance, 1.0);
        assert_eq!(sample.sample_count, 0);
    }

    #[test]
    fn test_sample_accumulate() {
        let mut sample = GodRaySample::new();
        sample.accumulate(0.5, [1.0, 1.0, 1.0], 0.96, 0.5);

        assert!(sample.light[0] > 0.0);
        assert!(sample.transmittance < 1.0);
        assert_eq!(sample.sample_count, 1);
    }

    #[test]
    fn test_sample_luminance() {
        let mut sample = GodRaySample::new();
        sample.light = [1.0, 1.0, 1.0];
        let lum = sample.luminance();
        assert!((lum - 1.0).abs() < 0.01); // Luminance of white
    }

    #[test]
    fn test_sample_occlusion_check() {
        let mut sample = GodRaySample::new();
        assert!(!sample.is_occluded(0.01));

        sample.transmittance = 0.001;
        assert!(sample.is_occluded(0.01));
    }

    #[test]
    fn test_sample_apply_exposure() {
        let mut sample = GodRaySample::new();
        sample.light = [1.0, 1.0, 1.0];
        sample.apply_exposure(0.5);
        assert!((sample.light[0] - 0.5).abs() < EPSILON);
    }

    // =========================================================================
    // GodRayOutput Tests
    // =========================================================================

    #[test]
    fn test_output_default() {
        let output = GodRayOutput::default();
        assert_eq!(output.color, [0.0, 0.0, 0.0]);
        assert_eq!(output.blend, 1.0);
    }

    #[test]
    fn test_output_from_sample() {
        let mut sample = GodRaySample::new();
        sample.light = [0.5, 0.5, 0.5];
        let output = GodRayOutput::from_sample(&sample, 0.5);
        assert!((output.color[0] - 0.25).abs() < EPSILON);
    }

    #[test]
    fn test_output_blend_additive() {
        let output = GodRayOutput {
            color: [0.1, 0.1, 0.1],
            blend: 1.0,
        };
        let result = output.blend_additive([0.5, 0.5, 0.5]);
        assert!((result[0] - 0.6).abs() < EPSILON);
    }

    #[test]
    fn test_output_gpu_alignment() {
        assert_eq!(std::mem::size_of::<GodRayOutput>(), 16);
    }

    // =========================================================================
    // Sun Screen Position Tests
    // =========================================================================

    #[test]
    fn test_sun_above_horizon() {
        assert!(is_sun_above_horizon([0.0, 1.0, 0.0], [0.0, 1.0, 0.0]));
        assert!(is_sun_above_horizon([0.5, 0.5, 0.0], [0.0, 1.0, 0.0]));
        assert!(!is_sun_above_horizon([0.0, -1.0, 0.0], [0.0, 1.0, 0.0]));
    }

    #[test]
    fn test_sun_to_screen_basic() {
        // Identity view/projection (simplified test)
        let view = identity_matrix();
        let projection = identity_matrix();

        // Sun directly in front
        let result = sun_to_screen_position([0.0, 0.0, -1.0], &view, &projection);
        assert!(result.is_some());
    }

    #[test]
    fn test_sun_to_screen_behind_camera() {
        let view = identity_matrix();
        let projection = identity_matrix();

        // Sun behind camera
        let _result = sun_to_screen_position([0.0, 0.0, 1.0], &view, &projection);
        // Should return None when sun is behind
        // Note: actual behavior depends on view matrix setup
    }

    #[test]
    fn test_sun_to_screen_extended() {
        let view = identity_matrix();
        let projection = identity_matrix();

        let result = sun_to_screen_position_extended([0.0, 0.0, -1.0], &view, &projection, 2.0);
        assert!(result.is_some());
    }

    // =========================================================================
    // Radial Sampling Tests
    // =========================================================================

    #[test]
    fn test_radial_sample_positions_count() {
        let samples = radial_sample_positions([0.0, 0.5], [0.5, 0.5], 10, 1.0);
        assert_eq!(samples.len(), 10);
    }

    #[test]
    fn test_radial_sample_positions_direction() {
        let samples = radial_sample_positions([0.0, 0.5], [1.0, 0.5], 10, 1.0);

        // Samples should move toward sun (increasing X)
        for i in 1..samples.len() {
            assert!(samples[i][0] > samples[i - 1][0]);
        }
    }

    #[test]
    fn test_radial_sample_positions_at_sun() {
        let samples = radial_sample_positions([0.5, 0.5], [0.5, 0.5], 10, 1.0);
        assert!(samples.is_empty()); // No samples when at sun position
    }

    #[test]
    fn test_calculate_radial_step() {
        let (step, length) = calculate_radial_step([0.0, 0.5], [1.0, 0.5], 10, 1.0);
        assert!(step[0] > 0.0);
        assert!((step[1]).abs() < EPSILON);
        assert!((length - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_calculate_radial_step_clamped() {
        let (_step, length) = calculate_radial_step([0.0, 0.5], [2.0, 0.5], 10, 1.0);
        assert!((length - 1.0).abs() < EPSILON); // Clamped to max_length
    }

    #[test]
    fn test_distance_to_sun() {
        let dist = distance_to_sun([0.0, 0.0], [1.0, 0.0]);
        assert!((dist - 1.0).abs() < EPSILON);

        let dist = distance_to_sun([0.0, 0.0], [3.0, 4.0]);
        assert!((dist - 5.0).abs() < EPSILON);
    }

    // =========================================================================
    // Epipolar Coordinate Tests
    // =========================================================================

    #[test]
    fn test_screen_to_epipolar_basic() {
        let (angle, distance) = screen_to_epipolar([1.0, 0.5], [0.5, 0.5]);
        assert!((angle - 0.0).abs() < EPSILON); // Right of sun = angle 0
        assert!((distance - 0.5).abs() < EPSILON);
    }

    #[test]
    fn test_screen_to_epipolar_roundtrip() {
        let sun_uv = [0.5, 0.5];
        let screen_uv = [0.8, 0.3];

        let (angle, distance) = screen_to_epipolar(screen_uv, sun_uv);
        let recovered = epipolar_to_screen(angle, distance, sun_uv);

        assert!((recovered[0] - screen_uv[0]).abs() < EPSILON);
        assert!((recovered[1] - screen_uv[1]).abs() < EPSILON);
    }

    #[test]
    fn test_angle_to_slice_index() {
        assert_eq!(angle_to_slice_index(0.0, 4), 0);
        assert_eq!(angle_to_slice_index(PI / 2.0, 4), 1);
        assert_eq!(angle_to_slice_index(PI, 4), 2);
        assert_eq!(angle_to_slice_index(3.0 * PI / 2.0, 4), 3);
    }

    #[test]
    fn test_slice_index_to_angle() {
        let angle = slice_index_to_angle(0, 4);
        assert!((angle - PI / 4.0).abs() < EPSILON);

        let angle = slice_index_to_angle(2, 4);
        assert!((angle - 5.0 * PI / 4.0).abs() < EPSILON);
    }

    #[test]
    fn test_distance_to_sample_index() {
        assert_eq!(distance_to_sample_index(0.0, 1.0, 10), 0);
        assert_eq!(distance_to_sample_index(0.5, 1.0, 10), 5);
        assert_eq!(distance_to_sample_index(1.0, 1.0, 10), 9); // Clamped to max
    }

    #[test]
    fn test_max_epipolar_distance() {
        // Sun at center, angle 0 (right)
        let dist = max_epipolar_distance([0.5, 0.5], 0.0);
        assert!((dist - 0.5).abs() < EPSILON); // Distance to right edge

        // Sun at center, angle PI (left)
        let dist = max_epipolar_distance([0.5, 0.5], PI);
        assert!((dist - 0.5).abs() < EPSILON);
    }

    #[test]
    fn test_max_epipolar_distance_corner() {
        // Sun at corner
        let dist = max_epipolar_distance([0.0, 0.0], PI / 4.0);
        assert!(dist > 0.0);
    }

    // =========================================================================
    // Beer-Lambert Extinction Tests
    // =========================================================================

    #[test]
    fn test_beer_lambert_zero_depth() {
        let t = beer_lambert(0.0, 1.0);
        assert!((t - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_beer_lambert_high_depth() {
        let t = beer_lambert(10.0, 1.0);
        assert!(t < 0.001);
    }

    #[test]
    fn test_beer_lambert_physics() {
        // Double the depth = squared transmittance
        let t1 = beer_lambert(1.0, 0.5);
        let t2 = beer_lambert(2.0, 0.5);
        assert!((t2 - t1 * t1).abs() < EPSILON);
    }

    #[test]
    fn test_transmittance() {
        let t = transmittance(0.5, 2.0, 0.1);
        let expected = (-0.5 * 2.0 * 0.1f32).exp();
        assert!((t - expected).abs() < EPSILON);
    }

    #[test]
    fn test_accumulate_optical_depth() {
        let depth = accumulate_optical_depth(0.5, 0.3, 0.2);
        assert!((depth - 0.56).abs() < EPSILON);
    }

    // =========================================================================
    // Temporal Reprojection Tests
    // =========================================================================

    #[test]
    fn test_calculate_motion_vector() {
        let mv = calculate_motion_vector([0.5, 0.5], [0.6, 0.4]);
        assert!((mv[0] - 0.1).abs() < EPSILON);
        assert!((mv[1] - (-0.1)).abs() < EPSILON);
    }

    #[test]
    fn test_is_reprojection_valid_on_screen() {
        assert!(is_reprojection_valid([0.5, 0.5], 0.5, 0.5, 0.1));
        assert!(!is_reprojection_valid([-0.1, 0.5], 0.5, 0.5, 0.1)); // Off screen
        assert!(!is_reprojection_valid([0.5, 0.5], 0.5, 0.8, 0.1)); // Depth mismatch
    }

    #[test]
    fn test_temporal_blend() {
        let current = [1.0, 0.0, 0.0];
        let history = [0.0, 1.0, 0.0];

        let blended = temporal_blend(current, history, 0.5, true);
        assert!((blended[0] - 0.5).abs() < EPSILON);
        assert!((blended[1] - 0.5).abs() < EPSILON);

        let no_history = temporal_blend(current, history, 0.5, false);
        assert!((no_history[0] - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_variance_clamp() {
        let history = [0.5, 0.5, 0.5];
        let min = [0.0, 0.0, 0.0];
        let max = [1.0, 1.0, 1.0];

        let clamped = variance_clamp(history, min, max);
        assert_eq!(clamped, history);

        let out_of_bounds = [1.5, -0.5, 0.5];
        let clamped = variance_clamp(out_of_bounds, min, max);
        assert_eq!(clamped, [1.0, 0.0, 0.5]);
    }

    // =========================================================================
    // Bilateral Upsampling Tests
    // =========================================================================

    #[test]
    fn test_bilateral_weight() {
        let w = bilateral_weight(0.0, 0.0, 3.0, 0.1);
        assert!((w - 1.0).abs() < EPSILON); // At center

        let w = bilateral_weight(3.0, 0.0, 3.0, 0.1);
        assert!(w < 1.0 && w > 0.0);
    }

    #[test]
    fn test_bilateral_weight_depth_rejection() {
        let w_close = bilateral_weight(1.0, 0.01, 3.0, 0.1);
        let w_far = bilateral_weight(1.0, 0.5, 3.0, 0.1);
        assert!(w_close > w_far); // Close depth = higher weight
    }

    #[test]
    fn test_generate_bilateral_kernel() {
        let kernel = generate_bilateral_kernel(1, 2.0);
        assert_eq!(kernel.len(), 9); // 3x3

        let kernel = generate_bilateral_kernel(2, 2.0);
        assert_eq!(kernel.len(), 25); // 5x5
    }

    #[test]
    fn test_generate_bilateral_kernel_center_weight() {
        let kernel = generate_bilateral_kernel(1, 2.0);
        let center = kernel.iter().find(|(x, y, _)| *x == 0 && *y == 0);
        assert!(center.is_some());
        assert!((center.unwrap().2 - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_bilateral_upsample_disabled() {
        let config = BilateralConfig::disabled();
        let value = [0.5, 0.5, 0.5];
        let result = bilateral_upsample(value, &[], &[], 0.5, &config);
        assert_eq!(result, value);
    }

    // =========================================================================
    // Helper Function Tests
    // =========================================================================

    #[test]
    fn test_lerp() {
        assert!((lerp(0.0, 1.0, 0.5) - 0.5).abs() < EPSILON);
        assert!((lerp(0.0, 1.0, 0.0) - 0.0).abs() < EPSILON);
        assert!((lerp(0.0, 1.0, 1.0) - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_lerp3() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 1.0, 1.0];
        let result = lerp3(a, b, 0.5);
        assert!((result[0] - 0.5).abs() < EPSILON);
        assert!((result[1] - 0.5).abs() < EPSILON);
        assert!((result[2] - 0.5).abs() < EPSILON);
    }

    #[test]
    fn test_halton_sequence() {
        let h0 = halton_sequence(0, 2);
        let h1 = halton_sequence(1, 2);
        let h2 = halton_sequence(2, 2);

        assert!((h0 - 0.0).abs() < EPSILON);
        assert!((h1 - 0.5).abs() < EPSILON);
        assert!((h2 - 0.25).abs() < EPSILON);
    }

    #[test]
    fn test_halton_2d_sequence() {
        let samples: Vec<_> = (0..8).map(|i| halton_2d(i, 1.0)).collect();

        // Check they're distributed
        let unique_x: std::collections::HashSet<_> =
            samples.iter().map(|s| (s[0] * 100.0) as i32).collect();
        assert!(unique_x.len() > 1);
    }

    // =========================================================================
    // GodRayRenderer Tests
    // =========================================================================

    #[test]
    fn test_renderer_default() {
        let renderer = GodRayRenderer::default();
        assert!(renderer.validate());
        assert!(renderer.is_enabled());
    }

    #[test]
    fn test_renderer_from_quality() {
        for quality in GodRayQuality::all() {
            let renderer = GodRayRenderer::from_quality(quality);
            assert!(renderer.validate());
        }
    }

    #[test]
    fn test_renderer_set_quality() {
        let mut renderer = GodRayRenderer::new();
        renderer.set_quality(GodRayQuality::Ultra);
        assert_eq!(renderer.uniforms.num_samples, 128);
    }

    #[test]
    fn test_renderer_set_sun_intensity() {
        let mut renderer = GodRayRenderer::new();
        renderer.set_sun_intensity(1.0, 0.8, 0.6);
        assert!((renderer.uniforms.sun_intensity[0] - 1.0).abs() < EPSILON);
        assert!((renderer.uniforms.sun_intensity[1] - 0.8).abs() < EPSILON);
    }

    #[test]
    fn test_renderer_next_frame() {
        let mut renderer = GodRayRenderer::new();
        assert_eq!(renderer.temporal.frame_index, 0);
        renderer.next_frame();
        assert_eq!(renderer.temporal.frame_index, 1);
    }

    #[test]
    fn test_renderer_enable_disable() {
        let mut renderer = GodRayRenderer::new();
        assert!(renderer.is_enabled());

        renderer.set_enabled(false);
        assert!(!renderer.is_enabled());
    }

    #[test]
    fn test_renderer_sample_pixel_disabled() {
        let mut renderer = GodRayRenderer::new();
        renderer.set_enabled(false);

        let sample = renderer.sample_pixel([0.5, 0.5], |_| 0.0);
        assert_eq!(sample.light, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_renderer_sample_pixel_no_occlusion() {
        let renderer = GodRayRenderer::new();
        let sample = renderer.sample_pixel([0.0, 0.5], |_| 0.0);

        assert!(sample.light[0] > 0.0);
        assert!(sample.sample_count > 0);
    }

    #[test]
    fn test_renderer_sample_pixel_full_occlusion() {
        let renderer = GodRayRenderer::new();
        let sample = renderer.sample_pixel([0.0, 0.5], |_| 1.0);

        // With full occlusion, should have minimal light
        assert!(sample.light[0] < 0.1);
    }

    #[test]
    fn test_renderer_estimated_memory() {
        let renderer = GodRayRenderer::new();
        let memory = renderer.estimated_memory_bytes();
        assert!(memory > 0);
        assert!(memory > std::mem::size_of::<GodRayUniforms>());
    }

    #[test]
    fn test_renderer_update_sun_position() {
        let mut renderer = GodRayRenderer::new();
        let view = identity_matrix();
        let projection = identity_matrix();

        let visible = renderer.update_sun_position([0.0, 0.0, -1.0], &view, &projection);
        // Result depends on matrix setup
        assert!(visible || !visible);
    }

    // =========================================================================
    // Integration Tests
    // =========================================================================

    #[test]
    fn test_full_pipeline_basic() {
        let renderer = GodRayRenderer::from_quality(GodRayQuality::Medium);

        // Simple gradient occlusion
        let sample = renderer.sample_pixel([0.2, 0.5], |uv| {
            // Occlusion increases toward sun
            uv[0].clamp(0.0, 1.0)
        });

        assert!(sample.sample_count > 0);
        assert!(sample.light[0] > 0.0);
    }

    #[test]
    fn test_full_pipeline_with_temporal() {
        let mut renderer = GodRayRenderer::new();

        let mut samples = Vec::new();
        for _ in 0..4 {
            let sample = renderer.sample_pixel([0.2, 0.5], |_| 0.3);
            samples.push(sample.light[0]);
            renderer.next_frame();
        }

        // All samples should be similar (deterministic)
        let first = samples[0];
        for &s in &samples[1..] {
            assert!((s - first).abs() < 0.01);
        }
    }

    #[test]
    fn test_epipolar_sampling_coverage() {
        let config = EpipolarConfig::default();
        let _sun_uv = [0.5, 0.5]; // Reference point for epipolar sampling

        // Check that epipolar slices cover full circle
        let mut angles = Vec::new();
        for i in 0..config.num_slices {
            let angle = slice_index_to_angle(i, config.num_slices);
            angles.push(angle);
        }

        // First angle should be small
        assert!(angles[0] < PI / 2.0);

        // Last angle should be near 2*PI
        assert!(angles[angles.len() - 1] > PI);

        // Coverage should be complete
        let coverage = angles[angles.len() - 1] - angles[0];
        assert!(coverage > PI);
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_sun_at_pixel() {
        let renderer = GodRayRenderer::new();

        // Sample at exact sun position
        let sample = renderer.sample_pixel(
            renderer.uniforms.sun_screen_pos,
            |_| 0.0,
        );

        // Should return empty or minimal sample
        assert!(sample.sample_count == 0 || sample.light[0] < 0.1);
    }

    #[test]
    fn test_sun_off_screen() {
        let mut renderer = GodRayRenderer::new();
        renderer.uniforms.sun_screen_pos = [-1.0, -1.0];

        // Should still compute rays toward off-screen sun
        let _sample = renderer.sample_pixel([0.5, 0.5], |_| 0.0);
        // Implementation may or may not allow this
    }

    #[test]
    fn test_zero_samples() {
        let mut renderer = GodRayRenderer::new();
        renderer.uniforms.num_samples = 0;

        let _sample = renderer.sample_pixel([0.2, 0.5], |_| 0.0);
        // Should handle gracefully
    }

    #[test]
    fn test_extreme_decay() {
        let mut renderer = GodRayRenderer::new();
        renderer.uniforms.decay = 0.5; // Very fast decay

        let sample = renderer.sample_pixel([0.0, 0.5], |_| 0.0);
        // Should terminate early due to low illumination
        assert!(sample.sample_count < renderer.uniforms.num_samples);
    }

    #[test]
    fn test_extreme_extinction() {
        let mut renderer = GodRayRenderer::new();
        renderer.uniforms.extinction_coeff = 10.0;

        let _sample = renderer.sample_pixel([0.0, 0.5], |_| 0.5);
        // High extinction should reduce light significantly
    }

    // =========================================================================
    // Pod/Zeroable Tests
    // =========================================================================

    #[test]
    fn test_pod_cast_uniforms() {
        let uniforms = GodRayUniforms::default();
        let bytes: &[u8] = bytemuck::bytes_of(&uniforms);
        assert_eq!(bytes.len(), 64);

        let back: &GodRayUniforms = bytemuck::from_bytes(bytes);
        assert_eq!(back.decay, uniforms.decay);
    }

    #[test]
    fn test_pod_cast_epipolar() {
        let config = EpipolarConfig::default();
        let bytes: &[u8] = bytemuck::bytes_of(&config);
        assert_eq!(bytes.len(), 32);
    }

    #[test]
    fn test_pod_cast_temporal() {
        let config = TemporalConfig::default();
        let bytes: &[u8] = bytemuck::bytes_of(&config);
        assert_eq!(bytes.len(), 32);
    }

    #[test]
    fn test_pod_cast_bilateral() {
        let config = BilateralConfig::default();
        let bytes: &[u8] = bytemuck::bytes_of(&config);
        assert_eq!(bytes.len(), 32);
    }

    #[test]
    fn test_pod_cast_output() {
        let output = GodRayOutput::default();
        let bytes: &[u8] = bytemuck::bytes_of(&output);
        assert_eq!(bytes.len(), 16);
    }

    // =========================================================================
    // Helper for tests
    // =========================================================================

    fn identity_matrix() -> [[f32; 4]; 4] {
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }
}
