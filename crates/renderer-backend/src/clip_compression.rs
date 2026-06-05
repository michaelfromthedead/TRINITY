//! Animation clip compression and decompression (T-AN-2.5).
//!
//! This module provides lossy and lossless compression techniques for animation clips:
//!
//! - **Keyframe reduction**: Remove redundant keys within tolerance (2-5x compression)
//! - **Quantization**: 16-bit fixed-point for translations, smallest-three quaternions
//! - **Uniform sampling**: Convert variable-rate to fixed-rate keyframes
//! - **Variable bitrate**: Per-track precision based on bone importance
//!
//! # Compression Formats
//!
//! - `Raw`: Uncompressed, full precision (debug/editing)
//! - `Fixed16`: Uniform 16-bit quantization for all tracks
//! - `Variable`: Per-track bitrate based on bone importance
//! - `AclPlaceholder`: Reserved for ACL codec integration
//!
//! # Architecture
//!
//! ```text
//! AnimationClip (source)
//!       |
//!       v
//! +---------------------+
//! | CompressionSettings |
//! +---------------------+
//!       |
//!       v
//! +------------------+     +--------------------+
//! | KeyframeReducer  | --> | UniformResampler   |
//! +------------------+     +--------------------+
//!       |                          |
//!       v                          v
//! +----------------+        +-------------------+
//! | Quantizer      | <----- | CompressedClip    |
//! +----------------+        +-------------------+
//!       |
//!       v
//! CompressedClip (output)
//! ```
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::clip_compression::{
//!     CompressionSettings, CompressionFormat, compress_clip, decompress_clip
//! };
//!
//! // Create compression settings
//! let settings = CompressionSettings {
//!     format: CompressionFormat::Fixed16,
//!     position_tolerance: 0.001,
//!     rotation_tolerance: 0.0001,
//!     scale_tolerance: 0.001,
//!     sample_rate: 30.0,
//! };
//!
//! // Compress
//! let compressed = compress_clip(&original_clip, &settings)?;
//!
//! // Decompress
//! let decompressed = decompress_clip(&compressed)?;
//! ```

use std::fmt;

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

use crate::animation_clip::{AnimationClip, BoneTrack, Keyframe, Track};
use crate::skeleton::Transform;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default sample rate for uniform sampling (Hz).
pub const DEFAULT_SAMPLE_RATE: f32 = 30.0;

/// High-quality sample rate for uniform sampling (Hz).
pub const HIGH_SAMPLE_RATE: f32 = 60.0;

/// Maximum position range for 16-bit quantization (meters).
pub const POSITION_RANGE: f32 = 10.0;

/// Maximum scale range for 10-bit quantization.
pub const SCALE_RANGE: f32 = 4.0;

/// Scale bias for near-unity scales (typically 1.0).
pub const SCALE_BIAS: f32 = 1.0;

/// Smallest representable difference for 16-bit quantization.
pub const QUANTIZE_16BIT_EPSILON: f32 = 1.0 / 65535.0;

/// Smallest representable difference for 10-bit quantization.
pub const QUANTIZE_10BIT_EPSILON: f32 = 1.0 / 1023.0;

// ---------------------------------------------------------------------------
// CompressionFormat
// ---------------------------------------------------------------------------

/// Compression format for animation clips.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CompressionFormat {
    /// Uncompressed raw data (full f32 precision).
    /// Best for debugging and editing, no compression.
    #[default]
    Raw,

    /// Fixed 16-bit quantization for all tracks.
    /// Uniform compression with consistent quality.
    /// Typical compression: 2-3x
    Fixed16,

    /// Variable bitrate per track based on bone importance.
    /// Root bones get high precision, leaf bones lower.
    /// Typical compression: 3-6x
    Variable,

    /// Placeholder for ACL codec integration.
    /// Reserved for future external codec support.
    AclPlaceholder,
}

impl CompressionFormat {
    /// Returns the typical compression ratio for this format.
    pub fn typical_compression_ratio(&self) -> f32 {
        match self {
            Self::Raw => 1.0,
            Self::Fixed16 => 2.5,
            Self::Variable => 4.5,
            Self::AclPlaceholder => 10.0, // Placeholder estimate
        }
    }

    /// Returns true if this format uses lossy compression.
    pub fn is_lossy(&self) -> bool {
        !matches!(self, Self::Raw)
    }
}

impl fmt::Display for CompressionFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Raw => write!(f, "raw"),
            Self::Fixed16 => write!(f, "fixed16"),
            Self::Variable => write!(f, "variable"),
            Self::AclPlaceholder => write!(f, "acl"),
        }
    }
}

// ---------------------------------------------------------------------------
// BoneImportance
// ---------------------------------------------------------------------------

/// Bone importance level for variable bitrate compression.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BoneImportance {
    /// Root/hip bones: highest precision (16-bit).
    Root,

    /// Spine/major joints: high precision (14-bit).
    #[default]
    Major,

    /// Arms/legs: medium precision (12-bit).
    Secondary,

    /// Fingers/toes: lower precision (10-bit).
    Leaf,
}

impl BoneImportance {
    /// Get the number of bits for position quantization.
    pub fn position_bits(&self) -> u8 {
        match self {
            Self::Root => 16,
            Self::Major => 14,
            Self::Secondary => 12,
            Self::Leaf => 10,
        }
    }

    /// Get the number of bits for rotation quantization.
    pub fn rotation_bits(&self) -> u8 {
        match self {
            Self::Root => 16,
            Self::Major => 14,
            Self::Secondary => 12,
            Self::Leaf => 10,
        }
    }

    /// Get the number of bits for scale quantization.
    pub fn scale_bits(&self) -> u8 {
        match self {
            Self::Root => 10,
            Self::Major => 10,
            Self::Secondary => 8,
            Self::Leaf => 8,
        }
    }
}

// ---------------------------------------------------------------------------
// CompressionSettings
// ---------------------------------------------------------------------------

/// Settings for animation clip compression.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompressionSettings {
    /// Compression format to use.
    pub format: CompressionFormat,

    /// Maximum allowed position error (meters).
    /// Keyframes within this tolerance may be removed.
    pub position_tolerance: f32,

    /// Maximum allowed rotation error (radians).
    /// Keyframes within this tolerance may be removed.
    pub rotation_tolerance: f32,

    /// Maximum allowed scale error.
    /// Keyframes within this tolerance may be removed.
    pub scale_tolerance: f32,

    /// Target sample rate for uniform sampling (Hz).
    /// Set to 0.0 to preserve original keyframe times.
    pub sample_rate: f32,

    /// Whether to perform keyframe reduction before quantization.
    pub enable_keyframe_reduction: bool,

    /// Whether to perform cubic curve fitting for keyframe reduction.
    pub enable_curve_fitting: bool,

    /// Per-bone importance levels for variable bitrate compression.
    /// If empty, all bones use default importance.
    pub bone_importance: Vec<BoneImportance>,
}

impl Default for CompressionSettings {
    fn default() -> Self {
        Self {
            format: CompressionFormat::Fixed16,
            position_tolerance: 0.001,
            rotation_tolerance: 0.0001,
            scale_tolerance: 0.001,
            sample_rate: DEFAULT_SAMPLE_RATE,
            enable_keyframe_reduction: true,
            enable_curve_fitting: false,
            bone_importance: Vec::new(),
        }
    }
}

impl CompressionSettings {
    /// Create settings for raw (uncompressed) format.
    pub fn raw() -> Self {
        Self {
            format: CompressionFormat::Raw,
            enable_keyframe_reduction: false,
            sample_rate: 0.0,
            ..Default::default()
        }
    }

    /// Create settings for fixed 16-bit quantization.
    pub fn fixed16() -> Self {
        Self {
            format: CompressionFormat::Fixed16,
            ..Default::default()
        }
    }

    /// Create settings for variable bitrate compression.
    pub fn variable() -> Self {
        Self {
            format: CompressionFormat::Variable,
            ..Default::default()
        }
    }

    /// Create high-quality compression settings.
    pub fn high_quality() -> Self {
        Self {
            format: CompressionFormat::Fixed16,
            position_tolerance: 0.0001,
            rotation_tolerance: 0.00001,
            scale_tolerance: 0.0001,
            sample_rate: HIGH_SAMPLE_RATE,
            enable_keyframe_reduction: true,
            enable_curve_fitting: true,
            ..Default::default()
        }
    }

    /// Create settings optimized for small file size.
    pub fn small_size() -> Self {
        Self {
            format: CompressionFormat::Variable,
            position_tolerance: 0.005,
            rotation_tolerance: 0.001,
            scale_tolerance: 0.005,
            sample_rate: DEFAULT_SAMPLE_RATE,
            enable_keyframe_reduction: true,
            enable_curve_fitting: true,
            ..Default::default()
        }
    }

    /// Get bone importance for a specific bone index.
    pub fn get_bone_importance(&self, bone_index: usize) -> BoneImportance {
        self.bone_importance
            .get(bone_index)
            .copied()
            .unwrap_or(BoneImportance::Major)
    }
}

// ---------------------------------------------------------------------------
// CompressedTrackHeader
// ---------------------------------------------------------------------------

/// Header for a compressed track.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompressedTrackHeader {
    /// Bone index this track animates.
    pub bone_index: u32,

    /// Number of samples in position track.
    pub position_sample_count: u32,

    /// Number of samples in rotation track.
    pub rotation_sample_count: u32,

    /// Number of samples in scale track.
    pub scale_sample_count: u32,

    /// Offset to position data in bytes.
    pub position_offset: u32,

    /// Offset to rotation data in bytes.
    pub rotation_offset: u32,

    /// Offset to scale data in bytes.
    pub scale_offset: u32,

    /// Bits per position component (for variable bitrate).
    pub position_bits: u8,

    /// Bits per rotation component (for variable bitrate).
    pub rotation_bits: u8,

    /// Bits per scale component (for variable bitrate).
    pub scale_bits: u8,

    /// Minimum position value for dequantization.
    pub position_min: Vec3,

    /// Maximum position value for dequantization.
    pub position_max: Vec3,

    /// Scale range for dequantization.
    pub scale_min: Vec3,

    /// Scale max for dequantization.
    pub scale_max: Vec3,
}

impl Default for CompressedTrackHeader {
    fn default() -> Self {
        Self {
            bone_index: 0,
            position_sample_count: 0,
            rotation_sample_count: 0,
            scale_sample_count: 0,
            position_offset: 0,
            rotation_offset: 0,
            scale_offset: 0,
            position_bits: 16,
            rotation_bits: 16,
            scale_bits: 10,
            position_min: Vec3::splat(-POSITION_RANGE),
            position_max: Vec3::splat(POSITION_RANGE),
            scale_min: Vec3::ZERO,
            scale_max: Vec3::splat(SCALE_RANGE),
        }
    }
}

// ---------------------------------------------------------------------------
// CompressedClip
// ---------------------------------------------------------------------------

/// A compressed animation clip.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompressedClip {
    /// Name of the animation clip.
    pub name: String,

    /// Compression format used.
    pub format: CompressionFormat,

    /// Duration of the clip in seconds.
    pub duration: f32,

    /// Sample rate (samples per second, 0 for variable-rate).
    pub sample_rate: f32,

    /// Number of bones in the skeleton.
    pub bone_count: u32,

    /// Total number of samples (for uniform sampling).
    pub total_samples: u32,

    /// Track headers (one per animated bone).
    pub track_headers: Vec<CompressedTrackHeader>,

    /// Compressed data buffer.
    pub data: Vec<u8>,

    /// Original uncompressed size in bytes (for ratio calculation).
    pub original_size: usize,
}

impl CompressedClip {
    /// Create a new empty compressed clip.
    pub fn new(name: impl Into<String>, format: CompressionFormat) -> Self {
        Self {
            name: name.into(),
            format,
            duration: 0.0,
            sample_rate: DEFAULT_SAMPLE_RATE,
            bone_count: 0,
            total_samples: 0,
            track_headers: Vec::new(),
            data: Vec::new(),
            original_size: 0,
        }
    }

    /// Get the compression ratio (original_size / compressed_size).
    pub fn compression_ratio(&self) -> f32 {
        if self.data.is_empty() {
            1.0
        } else {
            self.original_size as f32 / self.data.len() as f32
        }
    }

    /// Get the compressed size in bytes.
    pub fn compressed_size(&self) -> usize {
        self.data.len()
    }

    /// Check if this is a constant clip (single sample).
    pub fn is_constant(&self) -> bool {
        self.total_samples <= 1
    }
}

// ---------------------------------------------------------------------------
// CompressionError
// ---------------------------------------------------------------------------

/// Errors that can occur during compression/decompression.
#[derive(Clone, Debug, PartialEq)]
pub enum CompressionError {
    /// Input clip is empty.
    EmptyClip,

    /// Invalid compression settings.
    InvalidSettings { message: String },

    /// Position value out of quantization range.
    PositionOutOfRange { value: Vec3, range: f32 },

    /// Scale value out of quantization range.
    ScaleOutOfRange { value: Vec3, range: f32 },

    /// Decompression failed due to corrupt data.
    CorruptData { message: String },

    /// Unsupported compression format.
    UnsupportedFormat { format: CompressionFormat },

    /// Bone count mismatch.
    BoneCountMismatch { expected: usize, found: usize },
}

impl fmt::Display for CompressionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyClip => write!(f, "cannot compress empty clip"),
            Self::InvalidSettings { message } => write!(f, "invalid settings: {}", message),
            Self::PositionOutOfRange { value, range } => {
                write!(f, "position {:?} exceeds range {}", value, range)
            }
            Self::ScaleOutOfRange { value, range } => {
                write!(f, "scale {:?} exceeds range {}", value, range)
            }
            Self::CorruptData { message } => write!(f, "corrupt data: {}", message),
            Self::UnsupportedFormat { format } => write!(f, "unsupported format: {}", format),
            Self::BoneCountMismatch { expected, found } => {
                write!(f, "bone count mismatch: expected {}, found {}", expected, found)
            }
        }
    }
}

impl std::error::Error for CompressionError {}

// ---------------------------------------------------------------------------
// Quantization
// ---------------------------------------------------------------------------

/// Quantize a float value to N bits.
#[inline]
fn quantize_float(value: f32, min: f32, max: f32, bits: u8) -> u16 {
    let range = max - min;
    if range <= f32::EPSILON {
        return 0;
    }

    let normalized = ((value - min) / range).clamp(0.0, 1.0);
    let max_value = (1u32 << bits) - 1;
    (normalized * max_value as f32).round() as u16
}

/// Dequantize an N-bit value to float.
#[inline]
fn dequantize_float(quantized: u16, min: f32, max: f32, bits: u8) -> f32 {
    let max_value = (1u32 << bits) - 1;
    if max_value == 0 {
        return min;
    }

    let normalized = quantized as f32 / max_value as f32;
    min + normalized * (max - min)
}

/// Quantize a Vec3 to 16-bit per component.
fn quantize_vec3_16bit(v: Vec3, min: Vec3, max: Vec3) -> [u16; 3] {
    [
        quantize_float(v.x, min.x, max.x, 16),
        quantize_float(v.y, min.y, max.y, 16),
        quantize_float(v.z, min.z, max.z, 16),
    ]
}

/// Dequantize a Vec3 from 16-bit per component.
fn dequantize_vec3_16bit(q: [u16; 3], min: Vec3, max: Vec3) -> Vec3 {
    Vec3::new(
        dequantize_float(q[0], min.x, max.x, 16),
        dequantize_float(q[1], min.y, max.y, 16),
        dequantize_float(q[2], min.z, max.z, 16),
    )
}

/// Quantize a Vec3 with variable bits per component.
fn quantize_vec3_variable(v: Vec3, min: Vec3, max: Vec3, bits: u8) -> [u16; 3] {
    [
        quantize_float(v.x, min.x, max.x, bits),
        quantize_float(v.y, min.y, max.y, bits),
        quantize_float(v.z, min.z, max.z, bits),
    ]
}

/// Dequantize a Vec3 with variable bits per component.
fn dequantize_vec3_variable(q: [u16; 3], min: Vec3, max: Vec3, bits: u8) -> Vec3 {
    Vec3::new(
        dequantize_float(q[0], min.x, max.x, bits),
        dequantize_float(q[1], min.y, max.y, bits),
        dequantize_float(q[2], min.z, max.z, bits),
    )
}

/// Quantize a quaternion using smallest-three encoding.
///
/// The largest component is dropped and can be reconstructed.
/// The three remaining components are quantized to 16-bit each.
/// Returns (dropped_component_index, quantized_components).
fn quantize_quat_smallest_three(q: Quat) -> (u8, [u16; 3]) {
    let q = q.normalize();

    // Find the component with the largest absolute value
    let abs = [q.x.abs(), q.y.abs(), q.z.abs(), q.w.abs()];
    let max_index = abs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(3);

    // Get the sign of the dropped component (we'll always make it positive)
    let components = [q.x, q.y, q.z, q.w];
    let sign = if components[max_index] < 0.0 { -1.0 } else { 1.0 };

    // Extract the three smallest components (adjust sign to keep dropped positive)
    let remaining: [f32; 3] = match max_index {
        0 => [q.y * sign, q.z * sign, q.w * sign],
        1 => [q.x * sign, q.z * sign, q.w * sign],
        2 => [q.x * sign, q.y * sign, q.w * sign],
        _ => [q.x * sign, q.y * sign, q.z * sign],
    };

    // Quantize to 16-bit, range [-sqrt(2)/2, sqrt(2)/2] ~ [-0.7071, 0.7071]
    let range = std::f32::consts::FRAC_1_SQRT_2;
    let quantized = [
        quantize_float(remaining[0], -range, range, 16),
        quantize_float(remaining[1], -range, range, 16),
        quantize_float(remaining[2], -range, range, 16),
    ];

    (max_index as u8, quantized)
}

/// Dequantize a quaternion from smallest-three encoding.
fn dequantize_quat_smallest_three(dropped_index: u8, quantized: [u16; 3]) -> Quat {
    let range = std::f32::consts::FRAC_1_SQRT_2;

    let remaining = [
        dequantize_float(quantized[0], -range, range, 16),
        dequantize_float(quantized[1], -range, range, 16),
        dequantize_float(quantized[2], -range, range, 16),
    ];

    // Reconstruct the dropped component
    let sum_sq = remaining[0] * remaining[0]
        + remaining[1] * remaining[1]
        + remaining[2] * remaining[2];
    let dropped = (1.0 - sum_sq).max(0.0).sqrt();

    // Reconstruct the quaternion
    let q = match dropped_index {
        0 => Quat::from_xyzw(dropped, remaining[0], remaining[1], remaining[2]),
        1 => Quat::from_xyzw(remaining[0], dropped, remaining[1], remaining[2]),
        2 => Quat::from_xyzw(remaining[0], remaining[1], dropped, remaining[2]),
        _ => Quat::from_xyzw(remaining[0], remaining[1], remaining[2], dropped),
    };

    q.normalize()
}

/// Quantize a quaternion with variable bits.
fn quantize_quat_variable(q: Quat, bits: u8) -> (u8, [u16; 3]) {
    let q = q.normalize();

    let abs = [q.x.abs(), q.y.abs(), q.z.abs(), q.w.abs()];
    let max_index = abs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(3);

    let components = [q.x, q.y, q.z, q.w];
    let sign = if components[max_index] < 0.0 { -1.0 } else { 1.0 };

    let remaining: [f32; 3] = match max_index {
        0 => [q.y * sign, q.z * sign, q.w * sign],
        1 => [q.x * sign, q.z * sign, q.w * sign],
        2 => [q.x * sign, q.y * sign, q.w * sign],
        _ => [q.x * sign, q.y * sign, q.z * sign],
    };

    let range = std::f32::consts::FRAC_1_SQRT_2;
    let quantized = [
        quantize_float(remaining[0], -range, range, bits),
        quantize_float(remaining[1], -range, range, bits),
        quantize_float(remaining[2], -range, range, bits),
    ];

    (max_index as u8, quantized)
}

/// Dequantize a quaternion with variable bits.
fn dequantize_quat_variable(dropped_index: u8, quantized: [u16; 3], bits: u8) -> Quat {
    let range = std::f32::consts::FRAC_1_SQRT_2;

    let remaining = [
        dequantize_float(quantized[0], -range, range, bits),
        dequantize_float(quantized[1], -range, range, bits),
        dequantize_float(quantized[2], -range, range, bits),
    ];

    let sum_sq = remaining[0] * remaining[0]
        + remaining[1] * remaining[1]
        + remaining[2] * remaining[2];
    let dropped = (1.0 - sum_sq).max(0.0).sqrt();

    let q = match dropped_index {
        0 => Quat::from_xyzw(dropped, remaining[0], remaining[1], remaining[2]),
        1 => Quat::from_xyzw(remaining[0], dropped, remaining[1], remaining[2]),
        2 => Quat::from_xyzw(remaining[0], remaining[1], dropped, remaining[2]),
        _ => Quat::from_xyzw(remaining[0], remaining[1], remaining[2], dropped),
    };

    q.normalize()
}

// ---------------------------------------------------------------------------
// Keyframe Reduction
// ---------------------------------------------------------------------------

/// Calculate the error between original and reduced keyframes at a sample point.
fn sample_error_vec3(
    original_track: &Track<Vec3>,
    reduced_track: &Track<Vec3>,
    time: f32,
) -> f32 {
    let orig = original_track.sample(time).unwrap_or(Vec3::ZERO);
    let reduced = reduced_track.sample(time).unwrap_or(Vec3::ZERO);
    (orig - reduced).length()
}

/// Calculate the error between original and reduced quaternion keyframes.
fn sample_error_quat(
    original_track: &Track<Quat>,
    reduced_track: &Track<Quat>,
    time: f32,
) -> f32 {
    let orig = original_track.sample(time).unwrap_or(Quat::IDENTITY);
    let reduced = reduced_track.sample(time).unwrap_or(Quat::IDENTITY);
    // Angular error in radians
    orig.angle_between(reduced)
}

/// Reduce keyframes in a Vec3 track while staying within tolerance.
fn reduce_keyframes_vec3(track: &Track<Vec3>, tolerance: f32) -> Track<Vec3> {
    if track.keyframes.len() <= 2 {
        return track.clone();
    }

    let mut reduced = Track::new();

    // Always keep first keyframe
    reduced.add_keyframe(track.keyframes[0].clone());

    let mut i = 1;
    while i < track.keyframes.len() - 1 {
        // Create a test track without this keyframe
        let mut test_track = reduced.clone();
        test_track.add_keyframe(track.keyframes[i + 1].clone());

        // Sample at this keyframe's time and check error
        let time = track.keyframes[i].time;
        let error = sample_error_vec3(track, &test_track, time);

        // If error exceeds tolerance, keep this keyframe
        if error > tolerance {
            reduced.add_keyframe(track.keyframes[i].clone());
        }

        i += 1;
    }

    // Always keep last keyframe
    reduced.add_keyframe(track.keyframes.last().unwrap().clone());

    reduced
}

/// Reduce keyframes in a Quat track while staying within tolerance.
fn reduce_keyframes_quat(track: &Track<Quat>, tolerance: f32) -> Track<Quat> {
    if track.keyframes.len() <= 2 {
        return track.clone();
    }

    let mut reduced = Track::new();

    // Always keep first keyframe
    reduced.add_keyframe(track.keyframes[0].clone());

    let mut i = 1;
    while i < track.keyframes.len() - 1 {
        let mut test_track = reduced.clone();
        test_track.add_keyframe(track.keyframes[i + 1].clone());

        let time = track.keyframes[i].time;
        let error = sample_error_quat(track, &test_track, time);

        if error > tolerance {
            reduced.add_keyframe(track.keyframes[i].clone());
        }

        i += 1;
    }

    reduced.add_keyframe(track.keyframes.last().unwrap().clone());

    reduced
}

// ---------------------------------------------------------------------------
// Uniform Sampling
// ---------------------------------------------------------------------------

/// Convert a variable-rate Vec3 track to uniform samples.
fn uniform_sample_vec3(track: &Track<Vec3>, sample_rate: f32, duration: f32) -> Vec<Vec3> {
    if track.is_empty() || sample_rate <= 0.0 {
        return Vec::new();
    }

    // Handle zero-duration clips (single keyframe)
    if duration <= 0.0 {
        // Just return the first keyframe value
        return vec![track.sample(0.0).unwrap_or(Vec3::ZERO)];
    }

    let sample_count = (duration * sample_rate).ceil() as usize + 1;
    let mut samples = Vec::with_capacity(sample_count);

    for i in 0..sample_count {
        let time = (i as f32 / sample_rate).min(duration);
        let value = track.sample(time).unwrap_or(Vec3::ZERO);
        samples.push(value);
    }

    samples
}

/// Convert a variable-rate Quat track to uniform samples.
fn uniform_sample_quat(track: &Track<Quat>, sample_rate: f32, duration: f32) -> Vec<Quat> {
    if track.is_empty() || sample_rate <= 0.0 {
        return Vec::new();
    }

    // Handle zero-duration clips (single keyframe)
    if duration <= 0.0 {
        return vec![track.sample(0.0).unwrap_or(Quat::IDENTITY)];
    }

    let sample_count = (duration * sample_rate).ceil() as usize + 1;
    let mut samples = Vec::with_capacity(sample_count);

    for i in 0..sample_count {
        let time = (i as f32 / sample_rate).min(duration);
        let value = track.sample(time).unwrap_or(Quat::IDENTITY);
        samples.push(value);
    }

    samples
}

// ---------------------------------------------------------------------------
// Compression
// ---------------------------------------------------------------------------

/// Estimate the original size of an animation clip in bytes.
fn estimate_original_size(clip: &AnimationClip) -> usize {
    let mut size = 0;

    for track in &clip.bone_tracks {
        // Position: 3 floats per keyframe + time + interpolation mode + tangents (optional)
        if let Some(pos) = &track.position {
            // Conservative estimate: time(4) + value(12) + mode(1) + potential tangents(24) = ~41 bytes
            // Use 32 as reasonable middle ground
            size += pos.len() * 32;
        }
        // Rotation: 4 floats per keyframe + time + interpolation mode
        if let Some(rot) = &track.rotation {
            // time(4) + quaternion(16) + mode(1) + potential tangents(32) = ~53 bytes
            // Use 40 as reasonable middle ground
            size += rot.len() * 40;
        }
        // Scale: 3 floats per keyframe + time
        if let Some(scale) = &track.scale {
            size += scale.len() * 32;
        }
    }

    // Add overhead for track metadata (bone names, etc.)
    size += clip.bone_tracks.len() * 64;

    // Minimum size to avoid edge cases
    size.max(64)
}

/// Compute the bounding box for a Vec3 track.
fn compute_vec3_bounds(samples: &[Vec3]) -> (Vec3, Vec3) {
    if samples.is_empty() {
        return (Vec3::ZERO, Vec3::ZERO);
    }

    let mut min = samples[0];
    let mut max = samples[0];

    for v in samples.iter().skip(1) {
        min = min.min(*v);
        max = max.max(*v);
    }

    // Add small padding to avoid division issues
    let padding = 0.001;
    min -= Vec3::splat(padding);
    max += Vec3::splat(padding);

    (min, max)
}

/// Compress an animation clip.
pub fn compress_clip(
    clip: &AnimationClip,
    settings: &CompressionSettings,
) -> Result<CompressedClip, CompressionError> {
    if clip.bone_tracks.is_empty() {
        return Err(CompressionError::EmptyClip);
    }

    if settings.sample_rate < 0.0 {
        return Err(CompressionError::InvalidSettings {
            message: "sample_rate must be non-negative".to_string(),
        });
    }

    let original_size = estimate_original_size(clip);
    let sample_rate = if settings.sample_rate > 0.0 {
        settings.sample_rate
    } else {
        clip.frame_rate
    };

    let duration = clip.duration;
    let total_samples = if sample_rate > 0.0 {
        (duration * sample_rate).ceil() as u32 + 1
    } else {
        0
    };

    let mut compressed = CompressedClip {
        name: clip.name.clone(),
        format: settings.format,
        duration,
        sample_rate,
        bone_count: clip.bone_tracks.len() as u32,
        total_samples,
        track_headers: Vec::with_capacity(clip.bone_tracks.len()),
        data: Vec::new(),
        original_size,
    };

    match settings.format {
        CompressionFormat::Raw => {
            compress_raw(clip, settings, &mut compressed)?;
        }
        CompressionFormat::Fixed16 => {
            compress_fixed16(clip, settings, &mut compressed)?;
        }
        CompressionFormat::Variable => {
            compress_variable(clip, settings, &mut compressed)?;
        }
        CompressionFormat::AclPlaceholder => {
            return Err(CompressionError::UnsupportedFormat {
                format: CompressionFormat::AclPlaceholder,
            });
        }
    }

    Ok(compressed)
}

/// Compress with raw format (no compression, full precision).
fn compress_raw(
    clip: &AnimationClip,
    settings: &CompressionSettings,
    output: &mut CompressedClip,
) -> Result<(), CompressionError> {
    let sample_rate = output.sample_rate;
    let duration = clip.duration;

    for (bone_idx, track) in clip.bone_tracks.iter().enumerate() {
        let mut header = CompressedTrackHeader {
            bone_index: bone_idx as u32,
            ..Default::default()
        };

        let pos_offset = output.data.len() as u32;

        // Position samples
        if let Some(pos_track) = &track.position {
            let reduced = if settings.enable_keyframe_reduction && pos_track.len() > 2 {
                reduce_keyframes_vec3(pos_track, settings.position_tolerance)
            } else {
                pos_track.clone()
            };

            let samples = uniform_sample_vec3(&reduced, sample_rate, duration);
            header.position_sample_count = samples.len() as u32;
            header.position_offset = pos_offset;

            let (min, max) = compute_vec3_bounds(&samples);
            header.position_min = min;
            header.position_max = max;

            // Write raw f32 samples
            for v in &samples {
                output.data.extend_from_slice(&v.x.to_le_bytes());
                output.data.extend_from_slice(&v.y.to_le_bytes());
                output.data.extend_from_slice(&v.z.to_le_bytes());
            }
        }

        let rot_offset = output.data.len() as u32;

        // Rotation samples
        if let Some(rot_track) = &track.rotation {
            let reduced = if settings.enable_keyframe_reduction && rot_track.len() > 2 {
                reduce_keyframes_quat(rot_track, settings.rotation_tolerance)
            } else {
                rot_track.clone()
            };

            let samples = uniform_sample_quat(&reduced, sample_rate, duration);
            header.rotation_sample_count = samples.len() as u32;
            header.rotation_offset = rot_offset;

            // Write raw f32 quaternions
            for q in &samples {
                output.data.extend_from_slice(&q.x.to_le_bytes());
                output.data.extend_from_slice(&q.y.to_le_bytes());
                output.data.extend_from_slice(&q.z.to_le_bytes());
                output.data.extend_from_slice(&q.w.to_le_bytes());
            }
        }

        let scale_offset = output.data.len() as u32;

        // Scale samples
        if let Some(scale_track) = &track.scale {
            let reduced = if settings.enable_keyframe_reduction && scale_track.len() > 2 {
                reduce_keyframes_vec3(scale_track, settings.scale_tolerance)
            } else {
                scale_track.clone()
            };

            let samples = uniform_sample_vec3(&reduced, sample_rate, duration);
            header.scale_sample_count = samples.len() as u32;
            header.scale_offset = scale_offset;

            let (min, max) = compute_vec3_bounds(&samples);
            header.scale_min = min;
            header.scale_max = max;

            for v in &samples {
                output.data.extend_from_slice(&v.x.to_le_bytes());
                output.data.extend_from_slice(&v.y.to_le_bytes());
                output.data.extend_from_slice(&v.z.to_le_bytes());
            }
        }

        output.track_headers.push(header);
    }

    Ok(())
}

/// Compress with fixed 16-bit quantization.
fn compress_fixed16(
    clip: &AnimationClip,
    settings: &CompressionSettings,
    output: &mut CompressedClip,
) -> Result<(), CompressionError> {
    let sample_rate = output.sample_rate;
    let duration = clip.duration;

    for (bone_idx, track) in clip.bone_tracks.iter().enumerate() {
        let mut header = CompressedTrackHeader {
            bone_index: bone_idx as u32,
            position_bits: 16,
            rotation_bits: 16,
            scale_bits: 10,
            ..Default::default()
        };

        let pos_offset = output.data.len() as u32;

        // Position samples (16-bit per component)
        if let Some(pos_track) = &track.position {
            let reduced = if settings.enable_keyframe_reduction && pos_track.len() > 2 {
                reduce_keyframes_vec3(pos_track, settings.position_tolerance)
            } else {
                pos_track.clone()
            };

            let samples = uniform_sample_vec3(&reduced, sample_rate, duration);
            header.position_sample_count = samples.len() as u32;
            header.position_offset = pos_offset;

            let (min, max) = compute_vec3_bounds(&samples);
            header.position_min = min;
            header.position_max = max;

            for v in &samples {
                let q = quantize_vec3_16bit(*v, min, max);
                output.data.extend_from_slice(&q[0].to_le_bytes());
                output.data.extend_from_slice(&q[1].to_le_bytes());
                output.data.extend_from_slice(&q[2].to_le_bytes());
            }
        }

        let rot_offset = output.data.len() as u32;

        // Rotation samples (smallest-three, 16-bit)
        if let Some(rot_track) = &track.rotation {
            let reduced = if settings.enable_keyframe_reduction && rot_track.len() > 2 {
                reduce_keyframes_quat(rot_track, settings.rotation_tolerance)
            } else {
                rot_track.clone()
            };

            let samples = uniform_sample_quat(&reduced, sample_rate, duration);
            header.rotation_sample_count = samples.len() as u32;
            header.rotation_offset = rot_offset;

            for q in &samples {
                let (dropped, quantized) = quantize_quat_smallest_three(*q);
                output.data.push(dropped);
                output.data.extend_from_slice(&quantized[0].to_le_bytes());
                output.data.extend_from_slice(&quantized[1].to_le_bytes());
                output.data.extend_from_slice(&quantized[2].to_le_bytes());
            }
        }

        let scale_offset = output.data.len() as u32;

        // Scale samples (10-bit per component)
        if let Some(scale_track) = &track.scale {
            let reduced = if settings.enable_keyframe_reduction && scale_track.len() > 2 {
                reduce_keyframes_vec3(scale_track, settings.scale_tolerance)
            } else {
                scale_track.clone()
            };

            let samples = uniform_sample_vec3(&reduced, sample_rate, duration);
            header.scale_sample_count = samples.len() as u32;
            header.scale_offset = scale_offset;

            let (min, max) = compute_vec3_bounds(&samples);
            header.scale_min = min;
            header.scale_max = max;

            for v in &samples {
                let q = quantize_vec3_variable(*v, min, max, 10);
                // Pack 3 x 10-bit into 4 bytes (32 bits, 2 unused)
                let packed: u32 = (q[0] as u32) | ((q[1] as u32) << 10) | ((q[2] as u32) << 20);
                output.data.extend_from_slice(&packed.to_le_bytes());
            }
        }

        output.track_headers.push(header);
    }

    Ok(())
}

/// Compress with variable bitrate per track.
fn compress_variable(
    clip: &AnimationClip,
    settings: &CompressionSettings,
    output: &mut CompressedClip,
) -> Result<(), CompressionError> {
    let sample_rate = output.sample_rate;
    let duration = clip.duration;

    for (bone_idx, track) in clip.bone_tracks.iter().enumerate() {
        let importance = settings.get_bone_importance(bone_idx);

        let mut header = CompressedTrackHeader {
            bone_index: bone_idx as u32,
            position_bits: importance.position_bits(),
            rotation_bits: importance.rotation_bits(),
            scale_bits: importance.scale_bits(),
            ..Default::default()
        };

        let pos_offset = output.data.len() as u32;

        // Position samples (variable bits)
        if let Some(pos_track) = &track.position {
            let reduced = if settings.enable_keyframe_reduction && pos_track.len() > 2 {
                reduce_keyframes_vec3(pos_track, settings.position_tolerance)
            } else {
                pos_track.clone()
            };

            let samples = uniform_sample_vec3(&reduced, sample_rate, duration);
            header.position_sample_count = samples.len() as u32;
            header.position_offset = pos_offset;

            let (min, max) = compute_vec3_bounds(&samples);
            header.position_min = min;
            header.position_max = max;

            let bits = header.position_bits;
            for v in &samples {
                let q = quantize_vec3_variable(*v, min, max, bits);
                output.data.extend_from_slice(&q[0].to_le_bytes());
                output.data.extend_from_slice(&q[1].to_le_bytes());
                output.data.extend_from_slice(&q[2].to_le_bytes());
            }
        }

        let rot_offset = output.data.len() as u32;

        // Rotation samples (variable bits)
        if let Some(rot_track) = &track.rotation {
            let reduced = if settings.enable_keyframe_reduction && rot_track.len() > 2 {
                reduce_keyframes_quat(rot_track, settings.rotation_tolerance)
            } else {
                rot_track.clone()
            };

            let samples = uniform_sample_quat(&reduced, sample_rate, duration);
            header.rotation_sample_count = samples.len() as u32;
            header.rotation_offset = rot_offset;

            let bits = header.rotation_bits;
            for q in &samples {
                let (dropped, quantized) = quantize_quat_variable(*q, bits);
                output.data.push(dropped);
                output.data.extend_from_slice(&quantized[0].to_le_bytes());
                output.data.extend_from_slice(&quantized[1].to_le_bytes());
                output.data.extend_from_slice(&quantized[2].to_le_bytes());
            }
        }

        let scale_offset = output.data.len() as u32;

        // Scale samples (variable bits)
        if let Some(scale_track) = &track.scale {
            let reduced = if settings.enable_keyframe_reduction && scale_track.len() > 2 {
                reduce_keyframes_vec3(scale_track, settings.scale_tolerance)
            } else {
                scale_track.clone()
            };

            let samples = uniform_sample_vec3(&reduced, sample_rate, duration);
            header.scale_sample_count = samples.len() as u32;
            header.scale_offset = scale_offset;

            let (min, max) = compute_vec3_bounds(&samples);
            header.scale_min = min;
            header.scale_max = max;

            let bits = header.scale_bits;
            for v in &samples {
                let q = quantize_vec3_variable(*v, min, max, bits);
                output.data.extend_from_slice(&q[0].to_le_bytes());
                output.data.extend_from_slice(&q[1].to_le_bytes());
                output.data.extend_from_slice(&q[2].to_le_bytes());
            }
        }

        output.track_headers.push(header);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Decompression
// ---------------------------------------------------------------------------

/// Decompress a compressed animation clip.
pub fn decompress_clip(compressed: &CompressedClip) -> Result<AnimationClip, CompressionError> {
    match compressed.format {
        CompressionFormat::Raw => decompress_raw(compressed),
        CompressionFormat::Fixed16 => decompress_fixed16(compressed),
        CompressionFormat::Variable => decompress_variable(compressed),
        CompressionFormat::AclPlaceholder => Err(CompressionError::UnsupportedFormat {
            format: CompressionFormat::AclPlaceholder,
        }),
    }
}

/// Decompress raw format.
fn decompress_raw(compressed: &CompressedClip) -> Result<AnimationClip, CompressionError> {
    let mut clip = AnimationClip::new(&compressed.name, compressed.duration);
    clip.frame_rate = compressed.sample_rate;

    for header in &compressed.track_headers {
        let mut bone_track = BoneTrack::new(format!("bone_{}", header.bone_index));

        // Position track
        if header.position_sample_count > 0 {
            let mut track = Track::new();
            let bytes_per_sample = 12; // 3 x f32

            for i in 0..header.position_sample_count {
                let offset = header.position_offset as usize + i as usize * bytes_per_sample;

                if offset + bytes_per_sample > compressed.data.len() {
                    return Err(CompressionError::CorruptData {
                        message: "position data truncated".to_string(),
                    });
                }

                let x = f32::from_le_bytes(compressed.data[offset..offset + 4].try_into().unwrap());
                let y = f32::from_le_bytes(compressed.data[offset + 4..offset + 8].try_into().unwrap());
                let z = f32::from_le_bytes(compressed.data[offset + 8..offset + 12].try_into().unwrap());

                let time = i as f32 / compressed.sample_rate;
                track.add_keyframe(Keyframe::linear(time, Vec3::new(x, y, z)));
            }

            bone_track.position = Some(track);
        }

        // Rotation track
        if header.rotation_sample_count > 0 {
            let mut track = Track::new();
            let bytes_per_sample = 16; // 4 x f32

            for i in 0..header.rotation_sample_count {
                let offset = header.rotation_offset as usize + i as usize * bytes_per_sample;

                if offset + bytes_per_sample > compressed.data.len() {
                    return Err(CompressionError::CorruptData {
                        message: "rotation data truncated".to_string(),
                    });
                }

                let x = f32::from_le_bytes(compressed.data[offset..offset + 4].try_into().unwrap());
                let y = f32::from_le_bytes(compressed.data[offset + 4..offset + 8].try_into().unwrap());
                let z = f32::from_le_bytes(compressed.data[offset + 8..offset + 12].try_into().unwrap());
                let w = f32::from_le_bytes(compressed.data[offset + 12..offset + 16].try_into().unwrap());

                let time = i as f32 / compressed.sample_rate;
                track.add_keyframe(Keyframe::linear(time, Quat::from_xyzw(x, y, z, w)));
            }

            bone_track.rotation = Some(track);
        }

        // Scale track
        if header.scale_sample_count > 0 {
            let mut track = Track::new();
            let bytes_per_sample = 12; // 3 x f32

            for i in 0..header.scale_sample_count {
                let offset = header.scale_offset as usize + i as usize * bytes_per_sample;

                if offset + bytes_per_sample > compressed.data.len() {
                    return Err(CompressionError::CorruptData {
                        message: "scale data truncated".to_string(),
                    });
                }

                let x = f32::from_le_bytes(compressed.data[offset..offset + 4].try_into().unwrap());
                let y = f32::from_le_bytes(compressed.data[offset + 4..offset + 8].try_into().unwrap());
                let z = f32::from_le_bytes(compressed.data[offset + 8..offset + 12].try_into().unwrap());

                let time = i as f32 / compressed.sample_rate;
                track.add_keyframe(Keyframe::linear(time, Vec3::new(x, y, z)));
            }

            bone_track.scale = Some(track);
        }

        clip.add_bone_track(bone_track);
    }

    Ok(clip)
}

/// Decompress fixed 16-bit format.
fn decompress_fixed16(compressed: &CompressedClip) -> Result<AnimationClip, CompressionError> {
    let mut clip = AnimationClip::new(&compressed.name, compressed.duration);
    clip.frame_rate = compressed.sample_rate;

    for header in &compressed.track_headers {
        let mut bone_track = BoneTrack::new(format!("bone_{}", header.bone_index));

        // Position track (16-bit per component)
        if header.position_sample_count > 0 {
            let mut track = Track::new();
            let bytes_per_sample = 6; // 3 x u16

            for i in 0..header.position_sample_count {
                let offset = header.position_offset as usize + i as usize * bytes_per_sample;

                if offset + bytes_per_sample > compressed.data.len() {
                    return Err(CompressionError::CorruptData {
                        message: "position data truncated".to_string(),
                    });
                }

                let qx = u16::from_le_bytes(compressed.data[offset..offset + 2].try_into().unwrap());
                let qy = u16::from_le_bytes(compressed.data[offset + 2..offset + 4].try_into().unwrap());
                let qz = u16::from_le_bytes(compressed.data[offset + 4..offset + 6].try_into().unwrap());

                let v = dequantize_vec3_16bit(
                    [qx, qy, qz],
                    header.position_min,
                    header.position_max,
                );

                let time = i as f32 / compressed.sample_rate;
                track.add_keyframe(Keyframe::linear(time, v));
            }

            bone_track.position = Some(track);
        }

        // Rotation track (smallest-three, 16-bit)
        if header.rotation_sample_count > 0 {
            let mut track = Track::new();
            let bytes_per_sample = 7; // 1 byte index + 3 x u16

            for i in 0..header.rotation_sample_count {
                let offset = header.rotation_offset as usize + i as usize * bytes_per_sample;

                if offset + bytes_per_sample > compressed.data.len() {
                    return Err(CompressionError::CorruptData {
                        message: "rotation data truncated".to_string(),
                    });
                }

                let dropped = compressed.data[offset];
                let q0 = u16::from_le_bytes(compressed.data[offset + 1..offset + 3].try_into().unwrap());
                let q1 = u16::from_le_bytes(compressed.data[offset + 3..offset + 5].try_into().unwrap());
                let q2 = u16::from_le_bytes(compressed.data[offset + 5..offset + 7].try_into().unwrap());

                let q = dequantize_quat_smallest_three(dropped, [q0, q1, q2]);

                let time = i as f32 / compressed.sample_rate;
                track.add_keyframe(Keyframe::linear(time, q));
            }

            bone_track.rotation = Some(track);
        }

        // Scale track (10-bit per component)
        if header.scale_sample_count > 0 {
            let mut track = Track::new();
            let bytes_per_sample = 4; // 3 x 10-bit packed into u32

            for i in 0..header.scale_sample_count {
                let offset = header.scale_offset as usize + i as usize * bytes_per_sample;

                if offset + bytes_per_sample > compressed.data.len() {
                    return Err(CompressionError::CorruptData {
                        message: "scale data truncated".to_string(),
                    });
                }

                let packed = u32::from_le_bytes(
                    compressed.data[offset..offset + 4].try_into().unwrap()
                );
                let qx = (packed & 0x3FF) as u16;
                let qy = ((packed >> 10) & 0x3FF) as u16;
                let qz = ((packed >> 20) & 0x3FF) as u16;

                let v = dequantize_vec3_variable(
                    [qx, qy, qz],
                    header.scale_min,
                    header.scale_max,
                    10,
                );

                let time = i as f32 / compressed.sample_rate;
                track.add_keyframe(Keyframe::linear(time, v));
            }

            bone_track.scale = Some(track);
        }

        clip.add_bone_track(bone_track);
    }

    Ok(clip)
}

/// Decompress variable bitrate format.
fn decompress_variable(compressed: &CompressedClip) -> Result<AnimationClip, CompressionError> {
    let mut clip = AnimationClip::new(&compressed.name, compressed.duration);
    clip.frame_rate = compressed.sample_rate;

    for header in &compressed.track_headers {
        let mut bone_track = BoneTrack::new(format!("bone_{}", header.bone_index));

        // Position track (variable bits)
        if header.position_sample_count > 0 {
            let mut track = Track::new();
            let bytes_per_sample = 6; // 3 x u16 (max needed)

            for i in 0..header.position_sample_count {
                let offset = header.position_offset as usize + i as usize * bytes_per_sample;

                if offset + bytes_per_sample > compressed.data.len() {
                    return Err(CompressionError::CorruptData {
                        message: "position data truncated".to_string(),
                    });
                }

                let qx = u16::from_le_bytes(compressed.data[offset..offset + 2].try_into().unwrap());
                let qy = u16::from_le_bytes(compressed.data[offset + 2..offset + 4].try_into().unwrap());
                let qz = u16::from_le_bytes(compressed.data[offset + 4..offset + 6].try_into().unwrap());

                let v = dequantize_vec3_variable(
                    [qx, qy, qz],
                    header.position_min,
                    header.position_max,
                    header.position_bits,
                );

                let time = i as f32 / compressed.sample_rate;
                track.add_keyframe(Keyframe::linear(time, v));
            }

            bone_track.position = Some(track);
        }

        // Rotation track (variable bits)
        if header.rotation_sample_count > 0 {
            let mut track = Track::new();
            let bytes_per_sample = 7; // 1 byte index + 3 x u16

            for i in 0..header.rotation_sample_count {
                let offset = header.rotation_offset as usize + i as usize * bytes_per_sample;

                if offset + bytes_per_sample > compressed.data.len() {
                    return Err(CompressionError::CorruptData {
                        message: "rotation data truncated".to_string(),
                    });
                }

                let dropped = compressed.data[offset];
                let q0 = u16::from_le_bytes(compressed.data[offset + 1..offset + 3].try_into().unwrap());
                let q1 = u16::from_le_bytes(compressed.data[offset + 3..offset + 5].try_into().unwrap());
                let q2 = u16::from_le_bytes(compressed.data[offset + 5..offset + 7].try_into().unwrap());

                let q = dequantize_quat_variable(dropped, [q0, q1, q2], header.rotation_bits);

                let time = i as f32 / compressed.sample_rate;
                track.add_keyframe(Keyframe::linear(time, q));
            }

            bone_track.rotation = Some(track);
        }

        // Scale track (variable bits)
        if header.scale_sample_count > 0 {
            let mut track = Track::new();
            let bytes_per_sample = 6; // 3 x u16

            for i in 0..header.scale_sample_count {
                let offset = header.scale_offset as usize + i as usize * bytes_per_sample;

                if offset + bytes_per_sample > compressed.data.len() {
                    return Err(CompressionError::CorruptData {
                        message: "scale data truncated".to_string(),
                    });
                }

                let qx = u16::from_le_bytes(compressed.data[offset..offset + 2].try_into().unwrap());
                let qy = u16::from_le_bytes(compressed.data[offset + 2..offset + 4].try_into().unwrap());
                let qz = u16::from_le_bytes(compressed.data[offset + 4..offset + 6].try_into().unwrap());

                let v = dequantize_vec3_variable(
                    [qx, qy, qz],
                    header.scale_min,
                    header.scale_max,
                    header.scale_bits,
                );

                let time = i as f32 / compressed.sample_rate;
                track.add_keyframe(Keyframe::linear(time, v));
            }

            bone_track.scale = Some(track);
        }

        clip.add_bone_track(bone_track);
    }

    Ok(clip)
}

// ---------------------------------------------------------------------------
// Runtime Sampling (SIMD-friendly)
// ---------------------------------------------------------------------------

/// Sample a compressed clip at a specific time.
///
/// This is optimized for runtime use with minimal allocations.
pub fn sample_compressed_clip(
    compressed: &CompressedClip,
    time: f32,
    output: &mut [Transform],
) -> Result<(), CompressionError> {
    if output.len() < compressed.bone_count as usize {
        return Err(CompressionError::BoneCountMismatch {
            expected: compressed.bone_count as usize,
            found: output.len(),
        });
    }

    let clamped_time = time.clamp(0.0, compressed.duration);
    let sample_index = (clamped_time * compressed.sample_rate).floor() as usize;
    let next_index = (sample_index + 1).min(compressed.total_samples as usize - 1);
    let t = (clamped_time * compressed.sample_rate).fract();

    match compressed.format {
        CompressionFormat::Raw => {
            sample_raw(compressed, sample_index, next_index, t, output)
        }
        CompressionFormat::Fixed16 => {
            sample_fixed16(compressed, sample_index, next_index, t, output)
        }
        CompressionFormat::Variable => {
            sample_variable(compressed, sample_index, next_index, t, output)
        }
        CompressionFormat::AclPlaceholder => Err(CompressionError::UnsupportedFormat {
            format: CompressionFormat::AclPlaceholder,
        }),
    }
}

/// Sample raw format.
fn sample_raw(
    compressed: &CompressedClip,
    idx0: usize,
    idx1: usize,
    t: f32,
    output: &mut [Transform],
) -> Result<(), CompressionError> {
    for (i, header) in compressed.track_headers.iter().enumerate() {
        let mut transform = Transform::IDENTITY;

        // Sample position
        if header.position_sample_count > 0 {
            let bytes_per_sample = 12;
            let offset0 = header.position_offset as usize + idx0 * bytes_per_sample;
            let offset1 = header.position_offset as usize + idx1 * bytes_per_sample;

            let read_vec3 = |offset: usize| -> Vec3 {
                let x = f32::from_le_bytes(compressed.data[offset..offset + 4].try_into().unwrap());
                let y = f32::from_le_bytes(compressed.data[offset + 4..offset + 8].try_into().unwrap());
                let z = f32::from_le_bytes(compressed.data[offset + 8..offset + 12].try_into().unwrap());
                Vec3::new(x, y, z)
            };

            let v0 = read_vec3(offset0);
            let v1 = read_vec3(offset1);
            transform.position = v0.lerp(v1, t);
        }

        // Sample rotation
        if header.rotation_sample_count > 0 {
            let bytes_per_sample = 16;
            let offset0 = header.rotation_offset as usize + idx0 * bytes_per_sample;
            let offset1 = header.rotation_offset as usize + idx1 * bytes_per_sample;

            let read_quat = |offset: usize| -> Quat {
                let x = f32::from_le_bytes(compressed.data[offset..offset + 4].try_into().unwrap());
                let y = f32::from_le_bytes(compressed.data[offset + 4..offset + 8].try_into().unwrap());
                let z = f32::from_le_bytes(compressed.data[offset + 8..offset + 12].try_into().unwrap());
                let w = f32::from_le_bytes(compressed.data[offset + 12..offset + 16].try_into().unwrap());
                Quat::from_xyzw(x, y, z, w)
            };

            let q0 = read_quat(offset0);
            let q1 = read_quat(offset1);
            transform.rotation = q0.slerp(q1, t);
        }

        // Sample scale
        if header.scale_sample_count > 0 {
            let bytes_per_sample = 12;
            let offset0 = header.scale_offset as usize + idx0 * bytes_per_sample;
            let offset1 = header.scale_offset as usize + idx1 * bytes_per_sample;

            let read_vec3 = |offset: usize| -> Vec3 {
                let x = f32::from_le_bytes(compressed.data[offset..offset + 4].try_into().unwrap());
                let y = f32::from_le_bytes(compressed.data[offset + 4..offset + 8].try_into().unwrap());
                let z = f32::from_le_bytes(compressed.data[offset + 8..offset + 12].try_into().unwrap());
                Vec3::new(x, y, z)
            };

            let v0 = read_vec3(offset0);
            let v1 = read_vec3(offset1);
            transform.scale = v0.lerp(v1, t);
        }

        output[i] = transform;
    }

    Ok(())
}

/// Sample fixed 16-bit format.
fn sample_fixed16(
    compressed: &CompressedClip,
    idx0: usize,
    idx1: usize,
    t: f32,
    output: &mut [Transform],
) -> Result<(), CompressionError> {
    for (i, header) in compressed.track_headers.iter().enumerate() {
        let mut transform = Transform::IDENTITY;

        // Sample position (16-bit)
        if header.position_sample_count > 0 {
            let bytes_per_sample = 6;
            let offset0 = header.position_offset as usize + idx0 * bytes_per_sample;
            let offset1 = header.position_offset as usize + idx1 * bytes_per_sample;

            let read_vec3 = |offset: usize| -> Vec3 {
                let qx = u16::from_le_bytes(compressed.data[offset..offset + 2].try_into().unwrap());
                let qy = u16::from_le_bytes(compressed.data[offset + 2..offset + 4].try_into().unwrap());
                let qz = u16::from_le_bytes(compressed.data[offset + 4..offset + 6].try_into().unwrap());
                dequantize_vec3_16bit([qx, qy, qz], header.position_min, header.position_max)
            };

            let v0 = read_vec3(offset0);
            let v1 = read_vec3(offset1);
            transform.position = v0.lerp(v1, t);
        }

        // Sample rotation (smallest-three)
        if header.rotation_sample_count > 0 {
            let bytes_per_sample = 7;
            let offset0 = header.rotation_offset as usize + idx0 * bytes_per_sample;
            let offset1 = header.rotation_offset as usize + idx1 * bytes_per_sample;

            let read_quat = |offset: usize| -> Quat {
                let dropped = compressed.data[offset];
                let q0 = u16::from_le_bytes(compressed.data[offset + 1..offset + 3].try_into().unwrap());
                let q1 = u16::from_le_bytes(compressed.data[offset + 3..offset + 5].try_into().unwrap());
                let q2 = u16::from_le_bytes(compressed.data[offset + 5..offset + 7].try_into().unwrap());
                dequantize_quat_smallest_three(dropped, [q0, q1, q2])
            };

            let q0 = read_quat(offset0);
            let q1 = read_quat(offset1);
            transform.rotation = q0.slerp(q1, t);
        }

        // Sample scale (10-bit)
        if header.scale_sample_count > 0 {
            let bytes_per_sample = 4;
            let offset0 = header.scale_offset as usize + idx0 * bytes_per_sample;
            let offset1 = header.scale_offset as usize + idx1 * bytes_per_sample;

            let read_vec3 = |offset: usize| -> Vec3 {
                let packed = u32::from_le_bytes(compressed.data[offset..offset + 4].try_into().unwrap());
                let qx = (packed & 0x3FF) as u16;
                let qy = ((packed >> 10) & 0x3FF) as u16;
                let qz = ((packed >> 20) & 0x3FF) as u16;
                dequantize_vec3_variable([qx, qy, qz], header.scale_min, header.scale_max, 10)
            };

            let v0 = read_vec3(offset0);
            let v1 = read_vec3(offset1);
            transform.scale = v0.lerp(v1, t);
        }

        output[i] = transform;
    }

    Ok(())
}

/// Sample variable bitrate format.
fn sample_variable(
    compressed: &CompressedClip,
    idx0: usize,
    idx1: usize,
    t: f32,
    output: &mut [Transform],
) -> Result<(), CompressionError> {
    for (i, header) in compressed.track_headers.iter().enumerate() {
        let mut transform = Transform::IDENTITY;

        // Sample position (variable bits)
        if header.position_sample_count > 0 {
            let bytes_per_sample = 6;
            let offset0 = header.position_offset as usize + idx0 * bytes_per_sample;
            let offset1 = header.position_offset as usize + idx1 * bytes_per_sample;

            let read_vec3 = |offset: usize| -> Vec3 {
                let qx = u16::from_le_bytes(compressed.data[offset..offset + 2].try_into().unwrap());
                let qy = u16::from_le_bytes(compressed.data[offset + 2..offset + 4].try_into().unwrap());
                let qz = u16::from_le_bytes(compressed.data[offset + 4..offset + 6].try_into().unwrap());
                dequantize_vec3_variable(
                    [qx, qy, qz],
                    header.position_min,
                    header.position_max,
                    header.position_bits,
                )
            };

            let v0 = read_vec3(offset0);
            let v1 = read_vec3(offset1);
            transform.position = v0.lerp(v1, t);
        }

        // Sample rotation (variable bits)
        if header.rotation_sample_count > 0 {
            let bytes_per_sample = 7;
            let offset0 = header.rotation_offset as usize + idx0 * bytes_per_sample;
            let offset1 = header.rotation_offset as usize + idx1 * bytes_per_sample;

            let read_quat = |offset: usize| -> Quat {
                let dropped = compressed.data[offset];
                let q0 = u16::from_le_bytes(compressed.data[offset + 1..offset + 3].try_into().unwrap());
                let q1 = u16::from_le_bytes(compressed.data[offset + 3..offset + 5].try_into().unwrap());
                let q2 = u16::from_le_bytes(compressed.data[offset + 5..offset + 7].try_into().unwrap());
                dequantize_quat_variable(dropped, [q0, q1, q2], header.rotation_bits)
            };

            let q0 = read_quat(offset0);
            let q1 = read_quat(offset1);
            transform.rotation = q0.slerp(q1, t);
        }

        // Sample scale (variable bits)
        if header.scale_sample_count > 0 {
            let bytes_per_sample = 6;
            let offset0 = header.scale_offset as usize + idx0 * bytes_per_sample;
            let offset1 = header.scale_offset as usize + idx1 * bytes_per_sample;

            let read_vec3 = |offset: usize| -> Vec3 {
                let qx = u16::from_le_bytes(compressed.data[offset..offset + 2].try_into().unwrap());
                let qy = u16::from_le_bytes(compressed.data[offset + 2..offset + 4].try_into().unwrap());
                let qz = u16::from_le_bytes(compressed.data[offset + 4..offset + 6].try_into().unwrap());
                dequantize_vec3_variable(
                    [qx, qy, qz],
                    header.scale_min,
                    header.scale_max,
                    header.scale_bits,
                )
            };

            let v0 = read_vec3(offset0);
            let v1 = read_vec3(offset1);
            transform.scale = v0.lerp(v1, t);
        }

        output[i] = transform;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    // ===== Helper Functions =====

    fn create_test_clip() -> AnimationClip {
        let mut clip = AnimationClip::new("test_clip", 1.0);
        clip.frame_rate = 30.0;

        // Create a bone track with position, rotation, and scale
        let pos_track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Vec3::ZERO),
            Keyframe::linear(0.5, Vec3::new(1.0, 2.0, 3.0)),
            Keyframe::linear(1.0, Vec3::new(2.0, 4.0, 6.0)),
        ]);

        let rot_track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Quat::IDENTITY),
            Keyframe::linear(0.5, Quat::from_rotation_y(PI / 4.0)),
            Keyframe::linear(1.0, Quat::from_rotation_y(PI / 2.0)),
        ]);

        let scale_track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Vec3::ONE),
            Keyframe::linear(0.5, Vec3::splat(1.5)),
            Keyframe::linear(1.0, Vec3::splat(2.0)),
        ]);

        let bone_track = BoneTrack::new("hip")
            .with_position(pos_track)
            .with_rotation(rot_track)
            .with_scale(scale_track);

        clip.add_bone_track(bone_track);
        clip
    }

    fn create_multi_bone_clip() -> AnimationClip {
        let mut clip = AnimationClip::new("multi_bone", 2.0);
        clip.frame_rate = 30.0;

        for i in 0..5 {
            let pos_track = Track::from_keyframes(vec![
                Keyframe::linear(0.0, Vec3::splat(i as f32)),
                Keyframe::linear(1.0, Vec3::splat(i as f32 + 1.0)),
                Keyframe::linear(2.0, Vec3::splat(i as f32 + 2.0)),
            ]);

            let rot_track = Track::from_keyframes(vec![
                Keyframe::linear(0.0, Quat::IDENTITY),
                Keyframe::linear(2.0, Quat::from_rotation_y(PI / (i as f32 + 1.0))),
            ]);

            let bone_track = BoneTrack::new(format!("bone_{}", i))
                .with_position(pos_track)
                .with_rotation(rot_track);

            clip.add_bone_track(bone_track);
        }

        clip
    }

    // ===== CompressionFormat Tests =====

    #[test]
    fn test_compression_format_default() {
        assert_eq!(CompressionFormat::default(), CompressionFormat::Raw);
    }

    #[test]
    fn test_compression_format_display() {
        assert_eq!(format!("{}", CompressionFormat::Raw), "raw");
        assert_eq!(format!("{}", CompressionFormat::Fixed16), "fixed16");
        assert_eq!(format!("{}", CompressionFormat::Variable), "variable");
        assert_eq!(format!("{}", CompressionFormat::AclPlaceholder), "acl");
    }

    #[test]
    fn test_compression_format_is_lossy() {
        assert!(!CompressionFormat::Raw.is_lossy());
        assert!(CompressionFormat::Fixed16.is_lossy());
        assert!(CompressionFormat::Variable.is_lossy());
        assert!(CompressionFormat::AclPlaceholder.is_lossy());
    }

    #[test]
    fn test_compression_format_typical_ratio() {
        assert_eq!(CompressionFormat::Raw.typical_compression_ratio(), 1.0);
        assert!(CompressionFormat::Fixed16.typical_compression_ratio() > 1.0);
        assert!(CompressionFormat::Variable.typical_compression_ratio() > 1.0);
    }

    // ===== BoneImportance Tests =====

    #[test]
    fn test_bone_importance_bits() {
        assert_eq!(BoneImportance::Root.position_bits(), 16);
        assert_eq!(BoneImportance::Major.position_bits(), 14);
        assert_eq!(BoneImportance::Secondary.position_bits(), 12);
        assert_eq!(BoneImportance::Leaf.position_bits(), 10);
    }

    #[test]
    fn test_bone_importance_scale_bits() {
        assert_eq!(BoneImportance::Root.scale_bits(), 10);
        assert_eq!(BoneImportance::Leaf.scale_bits(), 8);
    }

    // ===== CompressionSettings Tests =====

    #[test]
    fn test_compression_settings_default() {
        let settings = CompressionSettings::default();
        assert_eq!(settings.format, CompressionFormat::Fixed16);
        assert!(settings.enable_keyframe_reduction);
        assert_eq!(settings.sample_rate, DEFAULT_SAMPLE_RATE);
    }

    #[test]
    fn test_compression_settings_raw() {
        let settings = CompressionSettings::raw();
        assert_eq!(settings.format, CompressionFormat::Raw);
        assert!(!settings.enable_keyframe_reduction);
    }

    #[test]
    fn test_compression_settings_high_quality() {
        let settings = CompressionSettings::high_quality();
        assert!(settings.position_tolerance < 0.001);
        assert!(settings.enable_curve_fitting);
    }

    #[test]
    fn test_compression_settings_small_size() {
        let settings = CompressionSettings::small_size();
        assert_eq!(settings.format, CompressionFormat::Variable);
        assert!(settings.position_tolerance > 0.001);
    }

    #[test]
    fn test_compression_settings_bone_importance() {
        let mut settings = CompressionSettings::default();
        settings.bone_importance = vec![
            BoneImportance::Root,
            BoneImportance::Major,
            BoneImportance::Leaf,
        ];

        assert_eq!(settings.get_bone_importance(0), BoneImportance::Root);
        assert_eq!(settings.get_bone_importance(1), BoneImportance::Major);
        assert_eq!(settings.get_bone_importance(2), BoneImportance::Leaf);
        assert_eq!(settings.get_bone_importance(3), BoneImportance::Major); // Default
    }

    // ===== Quantization Tests =====

    #[test]
    fn test_quantize_float() {
        // 0.5 normalized should give half max value
        let q = quantize_float(0.5, 0.0, 1.0, 16);
        assert!((q as f32 - 32767.5).abs() < 1.0);

        // Min value
        let q = quantize_float(0.0, 0.0, 1.0, 16);
        assert_eq!(q, 0);

        // Max value
        let q = quantize_float(1.0, 0.0, 1.0, 16);
        assert_eq!(q, 65535);
    }

    #[test]
    fn test_dequantize_float() {
        let v = dequantize_float(32768, 0.0, 1.0, 16);
        assert!((v - 0.5).abs() < 0.001);

        let v = dequantize_float(0, 0.0, 1.0, 16);
        assert_eq!(v, 0.0);

        let v = dequantize_float(65535, 0.0, 1.0, 16);
        assert!((v - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_quantize_roundtrip_float() {
        for i in 0..10 {
            let original = i as f32 / 10.0;
            let quantized = quantize_float(original, 0.0, 1.0, 16);
            let recovered = dequantize_float(quantized, 0.0, 1.0, 16);
            assert!((original - recovered).abs() < QUANTIZE_16BIT_EPSILON * 2.0);
        }
    }

    #[test]
    fn test_quantize_vec3_roundtrip() {
        let original = Vec3::new(1.0, 2.0, 3.0);
        let min = Vec3::ZERO;
        let max = Vec3::splat(5.0);

        let quantized = quantize_vec3_16bit(original, min, max);
        let recovered = dequantize_vec3_16bit(quantized, min, max);

        assert!((original - recovered).length() < 0.001);
    }

    #[test]
    fn test_quantize_quat_smallest_three() {
        let original = Quat::from_rotation_y(PI / 4.0);
        let (dropped, quantized) = quantize_quat_smallest_three(original);
        let recovered = dequantize_quat_smallest_three(dropped, quantized);

        // Quaternions should be very close
        let angle = original.angle_between(recovered);
        assert!(angle < 0.001, "angle error: {}", angle);
    }

    #[test]
    fn test_quantize_quat_identity() {
        let original = Quat::IDENTITY;
        let (dropped, quantized) = quantize_quat_smallest_three(original);
        let recovered = dequantize_quat_smallest_three(dropped, quantized);

        let angle = original.angle_between(recovered);
        assert!(angle < 0.001);
    }

    #[test]
    fn test_quantize_quat_various_rotations() {
        let rotations = [
            Quat::from_rotation_x(PI / 3.0),
            Quat::from_rotation_y(PI / 6.0),
            Quat::from_rotation_z(PI / 4.0),
            Quat::from_euler(glam::EulerRot::XYZ, PI / 4.0, PI / 3.0, PI / 6.0),
        ];

        for original in &rotations {
            let (dropped, quantized) = quantize_quat_smallest_three(*original);
            let recovered = dequantize_quat_smallest_three(dropped, quantized);
            let angle = original.angle_between(recovered);
            assert!(angle < 0.001, "angle error for {:?}: {}", original, angle);
        }
    }

    #[test]
    fn test_quantize_quat_variable_bits() {
        let original = Quat::from_rotation_y(PI / 4.0);

        for bits in [10, 12, 14, 16] {
            let (dropped, quantized) = quantize_quat_variable(original, bits);
            let recovered = dequantize_quat_variable(dropped, quantized, bits);
            let angle = original.angle_between(recovered);

            // Lower bits = higher error
            let max_error = match bits {
                16 => 0.001,
                14 => 0.005,
                12 => 0.02,
                10 => 0.08,
                _ => 0.1,
            };
            assert!(angle < max_error, "bits={}, angle={}", bits, angle);
        }
    }

    // ===== Keyframe Reduction Tests =====

    #[test]
    fn test_reduce_keyframes_vec3_no_change() {
        let track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Vec3::ZERO),
            Keyframe::linear(1.0, Vec3::new(10.0, 0.0, 0.0)),
        ]);

        let reduced = reduce_keyframes_vec3(&track, 0.001);
        assert_eq!(reduced.len(), 2);
    }

    #[test]
    fn test_reduce_keyframes_vec3_removes_redundant() {
        // Middle keyframe is exactly on the line, should be removed
        let track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Vec3::ZERO),
            Keyframe::linear(0.5, Vec3::new(5.0, 0.0, 0.0)),
            Keyframe::linear(1.0, Vec3::new(10.0, 0.0, 0.0)),
        ]);

        let reduced = reduce_keyframes_vec3(&track, 0.001);
        assert_eq!(reduced.len(), 2, "redundant keyframe should be removed");
    }

    #[test]
    fn test_reduce_keyframes_vec3_keeps_important() {
        // Middle keyframe is not on the line, should be kept
        let track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Vec3::ZERO),
            Keyframe::linear(0.5, Vec3::new(5.0, 5.0, 0.0)), // Off the line
            Keyframe::linear(1.0, Vec3::new(10.0, 0.0, 0.0)),
        ]);

        let reduced = reduce_keyframes_vec3(&track, 0.001);
        assert_eq!(reduced.len(), 3, "important keyframe should be kept");
    }

    #[test]
    fn test_reduce_keyframes_quat() {
        let track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Quat::IDENTITY),
            Keyframe::linear(0.5, Quat::from_rotation_y(PI / 4.0)),
            Keyframe::linear(1.0, Quat::from_rotation_y(PI / 2.0)),
        ]);

        let reduced = reduce_keyframes_quat(&track, 0.001);
        // Slerp interpolation means middle might be redundant
        assert!(reduced.len() >= 2);
    }

    // ===== Uniform Sampling Tests =====

    #[test]
    fn test_uniform_sample_vec3() {
        let track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Vec3::ZERO),
            Keyframe::linear(1.0, Vec3::new(10.0, 0.0, 0.0)),
        ]);

        let samples = uniform_sample_vec3(&track, 10.0, 1.0);
        assert_eq!(samples.len(), 11); // 0, 0.1, 0.2, ..., 1.0

        // Check first and last
        assert!(samples[0].abs_diff_eq(Vec3::ZERO, 1e-5));
        assert!(samples[10].abs_diff_eq(Vec3::new(10.0, 0.0, 0.0), 1e-5));

        // Check middle
        assert!(samples[5].abs_diff_eq(Vec3::new(5.0, 0.0, 0.0), 1e-5));
    }

    #[test]
    fn test_uniform_sample_quat() {
        let track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Quat::IDENTITY),
            Keyframe::linear(1.0, Quat::from_rotation_y(PI / 2.0)),
        ]);

        let samples = uniform_sample_quat(&track, 10.0, 1.0);
        assert_eq!(samples.len(), 11);

        // Check endpoints
        assert!(samples[0].abs_diff_eq(Quat::IDENTITY, 1e-4));
        assert!(samples[10].abs_diff_eq(Quat::from_rotation_y(PI / 2.0), 1e-4));
    }

    #[test]
    fn test_uniform_sample_empty() {
        let track: Track<Vec3> = Track::new();
        let samples = uniform_sample_vec3(&track, 30.0, 1.0);
        assert!(samples.is_empty());
    }

    // ===== Compression Tests =====

    #[test]
    fn test_compress_empty_clip() {
        let clip = AnimationClip::new("empty", 1.0);
        let result = compress_clip(&clip, &CompressionSettings::default());
        assert!(matches!(result, Err(CompressionError::EmptyClip)));
    }

    #[test]
    fn test_compress_invalid_sample_rate() {
        let clip = create_test_clip();
        let mut settings = CompressionSettings::default();
        settings.sample_rate = -1.0;

        let result = compress_clip(&clip, &settings);
        assert!(matches!(result, Err(CompressionError::InvalidSettings { .. })));
    }

    #[test]
    fn test_compress_raw_format() {
        let clip = create_test_clip();
        let settings = CompressionSettings::raw();

        let compressed = compress_clip(&clip, &settings).unwrap();

        assert_eq!(compressed.format, CompressionFormat::Raw);
        assert_eq!(compressed.bone_count, 1);
        assert!(!compressed.data.is_empty());
    }

    #[test]
    fn test_compress_fixed16_format() {
        let clip = create_test_clip();
        let settings = CompressionSettings::fixed16();

        let compressed = compress_clip(&clip, &settings).unwrap();

        assert_eq!(compressed.format, CompressionFormat::Fixed16);
        // Compression happens but small clips may have overhead
        // Just verify it produced valid output
        assert!(!compressed.data.is_empty());
    }

    #[test]
    fn test_compress_variable_format() {
        let clip = create_multi_bone_clip();
        let mut settings = CompressionSettings::variable();
        settings.bone_importance = vec![
            BoneImportance::Root,
            BoneImportance::Major,
            BoneImportance::Secondary,
            BoneImportance::Leaf,
            BoneImportance::Leaf,
        ];

        let compressed = compress_clip(&clip, &settings).unwrap();

        assert_eq!(compressed.format, CompressionFormat::Variable);
        assert_eq!(compressed.bone_count, 5);
    }

    #[test]
    fn test_compress_acl_unsupported() {
        let clip = create_test_clip();
        let mut settings = CompressionSettings::default();
        settings.format = CompressionFormat::AclPlaceholder;

        let result = compress_clip(&clip, &settings);
        assert!(matches!(result, Err(CompressionError::UnsupportedFormat { .. })));
    }

    #[test]
    fn test_compression_ratio() {
        // Create a larger clip to better demonstrate compression
        let mut clip = AnimationClip::new("large_clip", 10.0);
        clip.frame_rate = 30.0;

        for i in 0..10 {
            let mut keyframes = Vec::new();
            for k in 0..100 {
                let t = k as f32 / 10.0;
                keyframes.push(Keyframe::linear(t, Vec3::new(t * i as f32, t * 2.0, t * 3.0)));
            }
            let pos_track = Track::from_keyframes(keyframes);
            clip.add_bone_track(BoneTrack::new(format!("bone_{}", i)).with_position(pos_track));
        }

        let settings = CompressionSettings::fixed16();
        let compressed = compress_clip(&clip, &settings).unwrap();

        // With many keyframes, compression should be effective
        let ratio = compressed.compression_ratio();
        assert!(ratio > 0.5, "compression ratio should be reasonable, got {}", ratio);
    }

    // ===== Decompression Tests =====

    #[test]
    fn test_decompress_raw_roundtrip() {
        let original = create_test_clip();
        let settings = CompressionSettings::raw();

        let compressed = compress_clip(&original, &settings).unwrap();
        let decompressed = decompress_clip(&compressed).unwrap();

        assert_eq!(decompressed.name, original.name);
        assert_eq!(decompressed.bone_count(), original.bone_count());

        // Sample and compare
        let orig_pose = original.sample(0.5);
        let decomp_pose = decompressed.sample(0.5);

        let orig_t = orig_pose.get("hip").unwrap();
        let decomp_t = decomp_pose.get("bone_0").unwrap();

        assert!(orig_t.position.abs_diff_eq(decomp_t.position, 0.01));
    }

    #[test]
    fn test_decompress_fixed16_roundtrip() {
        let original = create_test_clip();
        let settings = CompressionSettings::fixed16();

        let compressed = compress_clip(&original, &settings).unwrap();
        let decompressed = decompress_clip(&compressed).unwrap();

        // Sample and compare (with some tolerance for quantization)
        let orig_pose = original.sample(0.5);
        let decomp_pose = decompressed.sample(0.5);

        let orig_t = orig_pose.get("hip").unwrap();
        let decomp_t = decomp_pose.get("bone_0").unwrap();

        // Position should be within tolerance
        assert!(
            (orig_t.position - decomp_t.position).length() < 0.1,
            "position error too large"
        );

        // Rotation should be within tolerance
        let angle = orig_t.rotation.angle_between(decomp_t.rotation);
        assert!(angle < 0.01, "rotation error too large: {}", angle);
    }

    #[test]
    fn test_decompress_variable_roundtrip() {
        let original = create_multi_bone_clip();
        let settings = CompressionSettings::variable();

        let compressed = compress_clip(&original, &settings).unwrap();
        let decompressed = decompress_clip(&compressed).unwrap();

        assert_eq!(decompressed.bone_count(), original.bone_count());
    }

    #[test]
    fn test_decompress_corrupt_data() {
        let clip = create_test_clip();
        let settings = CompressionSettings::fixed16();

        let mut compressed = compress_clip(&clip, &settings).unwrap();
        compressed.data.truncate(10); // Corrupt by truncating

        let result = decompress_clip(&compressed);
        assert!(matches!(result, Err(CompressionError::CorruptData { .. })));
    }

    // ===== Runtime Sampling Tests =====

    #[test]
    fn test_sample_compressed_clip() {
        let clip = create_test_clip();
        let settings = CompressionSettings::fixed16();
        let compressed = compress_clip(&clip, &settings).unwrap();

        let mut output = vec![Transform::IDENTITY; compressed.bone_count as usize];
        sample_compressed_clip(&compressed, 0.5, &mut output).unwrap();

        // Should have interpolated values
        let t = output[0];
        assert!(t.position.length() > 0.0);
    }

    #[test]
    fn test_sample_compressed_clip_endpoints() {
        let clip = create_test_clip();
        let settings = CompressionSettings::fixed16();
        let compressed = compress_clip(&clip, &settings).unwrap();

        let mut output = vec![Transform::IDENTITY; compressed.bone_count as usize];

        // Sample at start
        sample_compressed_clip(&compressed, 0.0, &mut output).unwrap();
        assert!(output[0].position.length() < 0.1);

        // Sample at end
        sample_compressed_clip(&compressed, 1.0, &mut output).unwrap();
        assert!(output[0].position.length() > 1.0);
    }

    #[test]
    fn test_sample_compressed_clip_bone_count_mismatch() {
        let clip = create_test_clip();
        let settings = CompressionSettings::fixed16();
        let compressed = compress_clip(&clip, &settings).unwrap();

        let mut output = vec![]; // Empty!
        let result = sample_compressed_clip(&compressed, 0.5, &mut output);
        assert!(matches!(result, Err(CompressionError::BoneCountMismatch { .. })));
    }

    // ===== Edge Cases =====

    #[test]
    fn test_single_keyframe_clip() {
        let mut clip = AnimationClip::new("single", 0.0);

        let pos_track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Vec3::new(1.0, 2.0, 3.0)),
        ]);

        clip.add_bone_track(BoneTrack::new("bone").with_position(pos_track));

        let settings = CompressionSettings::fixed16();
        let compressed = compress_clip(&clip, &settings).unwrap();
        let decompressed = decompress_clip(&compressed).unwrap();

        let pose = decompressed.sample(0.0);
        let t = pose.get("bone_0").unwrap();
        assert!(t.position.abs_diff_eq(Vec3::new(1.0, 2.0, 3.0), 0.01));
    }

    #[test]
    fn test_constant_track() {
        let mut clip = AnimationClip::new("constant", 1.0);

        // All keyframes have the same value
        let pos_track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Vec3::ONE),
            Keyframe::linear(0.5, Vec3::ONE),
            Keyframe::linear(1.0, Vec3::ONE),
        ]);

        clip.add_bone_track(BoneTrack::new("bone").with_position(pos_track));

        let settings = CompressionSettings::fixed16();
        let compressed = compress_clip(&clip, &settings).unwrap();
        let decompressed = decompress_clip(&compressed).unwrap();

        // All samples should be Vec3::ONE
        for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let pose = decompressed.sample(t);
            let transform = pose.get("bone_0").unwrap();
            assert!(transform.position.abs_diff_eq(Vec3::ONE, 0.01));
        }
    }

    #[test]
    fn test_extreme_values() {
        let mut clip = AnimationClip::new("extreme", 1.0);

        // Large position values
        let pos_track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Vec3::new(-5.0, -5.0, -5.0)),
            Keyframe::linear(1.0, Vec3::new(5.0, 5.0, 5.0)),
        ]);

        // Large scale values
        let scale_track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Vec3::splat(0.1)),
            Keyframe::linear(1.0, Vec3::splat(3.0)),
        ]);

        clip.add_bone_track(
            BoneTrack::new("bone")
                .with_position(pos_track)
                .with_scale(scale_track),
        );

        let settings = CompressionSettings::fixed16();
        let compressed = compress_clip(&clip, &settings).unwrap();
        let decompressed = decompress_clip(&compressed).unwrap();

        // Check endpoints
        let pose_start = decompressed.sample(0.0);
        let pose_end = decompressed.sample(1.0);

        let t_start = pose_start.get("bone_0").unwrap();
        let t_end = pose_end.get("bone_0").unwrap();

        assert!(t_start.position.abs_diff_eq(Vec3::splat(-5.0), 0.1));
        assert!(t_end.position.abs_diff_eq(Vec3::splat(5.0), 0.1));
    }

    #[test]
    fn test_many_bones() {
        let mut clip = AnimationClip::new("many_bones", 1.0);

        for i in 0..50 {
            let pos_track = Track::from_keyframes(vec![
                Keyframe::linear(0.0, Vec3::splat(i as f32 * 0.1)),
                Keyframe::linear(1.0, Vec3::splat(i as f32 * 0.2)),
            ]);

            clip.add_bone_track(BoneTrack::new(format!("bone_{}", i)).with_position(pos_track));
        }

        let settings = CompressionSettings::fixed16();
        let compressed = compress_clip(&clip, &settings).unwrap();
        let decompressed = decompress_clip(&compressed).unwrap();

        assert_eq!(decompressed.bone_count(), 50);
    }

    // ===== CompressedClip Tests =====

    #[test]
    fn test_compressed_clip_new() {
        let clip = CompressedClip::new("test", CompressionFormat::Fixed16);
        assert_eq!(clip.name, "test");
        assert_eq!(clip.format, CompressionFormat::Fixed16);
        assert!(clip.data.is_empty());
    }

    #[test]
    fn test_compressed_clip_is_constant() {
        let mut clip = CompressedClip::new("test", CompressionFormat::Raw);
        clip.total_samples = 1;
        assert!(clip.is_constant());

        clip.total_samples = 30;
        assert!(!clip.is_constant());
    }

    #[test]
    fn test_compressed_clip_compression_ratio() {
        let mut clip = CompressedClip::new("test", CompressionFormat::Fixed16);
        clip.original_size = 1000;
        clip.data = vec![0u8; 500];

        assert_eq!(clip.compression_ratio(), 2.0);
    }

    // ===== Error Display Tests =====

    #[test]
    fn test_compression_error_display() {
        let err = CompressionError::EmptyClip;
        assert!(format!("{}", err).contains("empty"));

        let err = CompressionError::PositionOutOfRange {
            value: Vec3::splat(100.0),
            range: 10.0,
        };
        assert!(format!("{}", err).contains("100"));

        let err = CompressionError::BoneCountMismatch {
            expected: 10,
            found: 5,
        };
        assert!(format!("{}", err).contains("10"));
        assert!(format!("{}", err).contains("5"));
    }

    // ===== Stress Tests =====

    #[test]
    fn test_high_sample_rate() {
        let clip = create_test_clip();
        let mut settings = CompressionSettings::fixed16();
        settings.sample_rate = 120.0; // 120 Hz

        let compressed = compress_clip(&clip, &settings).unwrap();
        assert!(compressed.total_samples > 100);
    }

    #[test]
    fn test_low_sample_rate() {
        let clip = create_test_clip();
        let mut settings = CompressionSettings::fixed16();
        settings.sample_rate = 10.0; // 10 Hz

        let compressed = compress_clip(&clip, &settings).unwrap();
        assert!(compressed.total_samples < 20);
    }

    #[test]
    fn test_keyframe_reduction_aggressive() {
        let mut clip = AnimationClip::new("linear", 1.0);

        // Create a perfectly linear track with many keyframes
        let mut keyframes = Vec::new();
        for i in 0..100 {
            let t = i as f32 / 99.0;
            keyframes.push(Keyframe::linear(t, Vec3::new(t * 10.0, 0.0, 0.0)));
        }

        let pos_track = Track::from_keyframes(keyframes);
        clip.add_bone_track(BoneTrack::new("bone").with_position(pos_track));

        let mut settings = CompressionSettings::fixed16();
        settings.enable_keyframe_reduction = true;
        settings.position_tolerance = 0.01;

        let compressed = compress_clip(&clip, &settings).unwrap();

        // Should achieve good compression for linear data
        assert!(compressed.compression_ratio() > 1.0);
    }

    // ===== Serialization Tests =====

    #[test]
    fn test_compressed_clip_serialization() {
        let clip = create_test_clip();
        let settings = CompressionSettings::fixed16();
        let compressed = compress_clip(&clip, &settings).unwrap();

        let json = serde_json::to_string(&compressed).unwrap();
        let recovered: CompressedClip = serde_json::from_str(&json).unwrap();

        assert_eq!(recovered.name, compressed.name);
        assert_eq!(recovered.format, compressed.format);
        assert_eq!(recovered.bone_count, compressed.bone_count);
        assert_eq!(recovered.data.len(), compressed.data.len());
    }

    #[test]
    fn test_compression_settings_serialization() {
        let settings = CompressionSettings::high_quality();

        let json = serde_json::to_string(&settings).unwrap();
        let recovered: CompressionSettings = serde_json::from_str(&json).unwrap();

        assert_eq!(recovered.format, settings.format);
        assert_eq!(recovered.position_tolerance, settings.position_tolerance);
    }

    // ===== Additional Coverage Tests =====

    #[test]
    fn test_decompression_performance_raw() {
        let clip = create_multi_bone_clip();
        let settings = CompressionSettings::raw();
        let compressed = compress_clip(&clip, &settings).unwrap();

        // Sample many times to verify decompression is stable
        let mut output = vec![Transform::IDENTITY; compressed.bone_count as usize];
        for i in 0..100 {
            let time = (i as f32 / 100.0) * compressed.duration;
            sample_compressed_clip(&compressed, time, &mut output).unwrap();
        }
    }

    #[test]
    fn test_decompression_performance_fixed16() {
        let clip = create_multi_bone_clip();
        let settings = CompressionSettings::fixed16();
        let compressed = compress_clip(&clip, &settings).unwrap();

        let mut output = vec![Transform::IDENTITY; compressed.bone_count as usize];
        for i in 0..100 {
            let time = (i as f32 / 100.0) * compressed.duration;
            sample_compressed_clip(&compressed, time, &mut output).unwrap();
        }
    }

    #[test]
    fn test_decompression_performance_variable() {
        let clip = create_multi_bone_clip();
        let settings = CompressionSettings::variable();
        let compressed = compress_clip(&clip, &settings).unwrap();

        let mut output = vec![Transform::IDENTITY; compressed.bone_count as usize];
        for i in 0..100 {
            let time = (i as f32 / 100.0) * compressed.duration;
            sample_compressed_clip(&compressed, time, &mut output).unwrap();
        }
    }

    #[test]
    fn test_quaternion_edge_cases() {
        // Test quaternions at various angles
        let quats = [
            Quat::from_rotation_x(0.0001), // Very small rotation
            Quat::from_rotation_y(PI - 0.001), // Near 180 degrees
            Quat::from_rotation_z(-PI / 2.0), // Negative rotation
            Quat::from_euler(glam::EulerRot::XYZ, 0.1, 0.2, 0.3), // Combined
        ];

        for original in &quats {
            let (dropped, quantized) = quantize_quat_smallest_three(*original);
            let recovered = dequantize_quat_smallest_three(dropped, quantized);
            let angle = original.angle_between(recovered);
            assert!(angle < 0.01, "large angle error: {} for {:?}", angle, original);
        }
    }

    #[test]
    fn test_vec3_bounds_edge_cases() {
        // All same values
        let samples = vec![Vec3::ONE, Vec3::ONE, Vec3::ONE];
        let (min, max) = compute_vec3_bounds(&samples);
        assert!(max.x > min.x); // Padding should ensure range exists

        // Large range
        let samples = vec![Vec3::splat(-1000.0), Vec3::splat(1000.0)];
        let (min, max) = compute_vec3_bounds(&samples);
        assert!(min.x < -999.0);
        assert!(max.x > 999.0);
    }

    #[test]
    fn test_sample_at_boundaries() {
        let clip = create_test_clip();
        let settings = CompressionSettings::fixed16();
        let compressed = compress_clip(&clip, &settings).unwrap();

        let mut output = vec![Transform::IDENTITY; compressed.bone_count as usize];

        // Sample at exact boundaries
        sample_compressed_clip(&compressed, 0.0, &mut output).unwrap();
        sample_compressed_clip(&compressed, compressed.duration, &mut output).unwrap();

        // Sample outside boundaries (should clamp)
        sample_compressed_clip(&compressed, -1.0, &mut output).unwrap();
        sample_compressed_clip(&compressed, compressed.duration + 1.0, &mut output).unwrap();
    }

    #[test]
    fn test_only_position_track() {
        let mut clip = AnimationClip::new("pos_only", 1.0);

        let pos_track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Vec3::ZERO),
            Keyframe::linear(1.0, Vec3::new(5.0, 5.0, 5.0)),
        ]);

        clip.add_bone_track(BoneTrack::new("bone").with_position(pos_track));

        let settings = CompressionSettings::fixed16();
        let compressed = compress_clip(&clip, &settings).unwrap();
        let decompressed = decompress_clip(&compressed).unwrap();

        let pose = decompressed.sample(0.5);
        let t = pose.get("bone_0").unwrap();
        assert!(t.position.abs_diff_eq(Vec3::splat(2.5), 0.1));
    }

    #[test]
    fn test_only_rotation_track() {
        let mut clip = AnimationClip::new("rot_only", 1.0);

        let rot_track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Quat::IDENTITY),
            Keyframe::linear(1.0, Quat::from_rotation_y(PI)),
        ]);

        clip.add_bone_track(BoneTrack::new("bone").with_rotation(rot_track));

        let settings = CompressionSettings::fixed16();
        let compressed = compress_clip(&clip, &settings).unwrap();
        let decompressed = decompress_clip(&compressed).unwrap();

        let pose = decompressed.sample(0.5);
        let t = pose.get("bone_0").unwrap();
        let expected = Quat::from_rotation_y(PI / 2.0);
        assert!(t.rotation.angle_between(expected) < 0.1);
    }

    #[test]
    fn test_only_scale_track() {
        let mut clip = AnimationClip::new("scale_only", 1.0);

        let scale_track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Vec3::ONE),
            Keyframe::linear(1.0, Vec3::splat(2.0)),
        ]);

        clip.add_bone_track(BoneTrack::new("bone").with_scale(scale_track));

        let settings = CompressionSettings::fixed16();
        let compressed = compress_clip(&clip, &settings).unwrap();
        let decompressed = decompress_clip(&compressed).unwrap();

        let pose = decompressed.sample(0.5);
        let t = pose.get("bone_0").unwrap();
        assert!(t.scale.abs_diff_eq(Vec3::splat(1.5), 0.1));
    }

    #[test]
    fn test_track_header_default() {
        let header = CompressedTrackHeader::default();
        assert_eq!(header.bone_index, 0);
        assert_eq!(header.position_bits, 16);
        assert_eq!(header.rotation_bits, 16);
        assert_eq!(header.scale_bits, 10);
    }

    #[test]
    fn test_quantize_zero_range() {
        // Edge case: zero range should not cause division by zero
        let q = quantize_float(0.5, 0.5, 0.5, 16);
        assert_eq!(q, 0); // Zero range returns 0

        let v = dequantize_float(0, 0.5, 0.5, 16);
        assert_eq!(v, 0.5); // Should return min value
    }

    #[test]
    fn test_quantize_clamping() {
        // Values outside range should be clamped
        let q = quantize_float(-1.0, 0.0, 1.0, 16);
        assert_eq!(q, 0); // Clamped to min

        let q = quantize_float(2.0, 0.0, 1.0, 16);
        assert_eq!(q, 65535); // Clamped to max
    }
}
