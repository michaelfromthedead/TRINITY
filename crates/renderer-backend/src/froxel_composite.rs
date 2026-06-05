//! Froxel Compositing Pass (T-ENV-1.6)
//!
//! This module provides infrastructure for compositing froxel-based volumetric
//! fog with the scene color buffer. It integrates:
//! - Froxel volume sampling from `froxel.rs`
//! - Volumetric fog density and scattering from `volumetric_fog.rs`
//! - Temporal reprojection for stable, low-noise fog
//! - Multiple blending modes for final compositing
//!
//! # Overview
//!
//! The compositing pipeline:
//! 1. **Accumulate Lighting**: Integrate inscattered light along view rays through froxels
//! 2. **Temporal Reproject**: Blend with previous frame data for temporal stability
//! 3. **Composite**: Blend accumulated fog with scene color using selected blend mode
//!
//! # Temporal Reprojection
//!
//! Temporal reprojection improves quality by:
//! - Reducing noise from stochastic sampling
//! - Providing smoother animations
//! - Allowing lower per-frame sample counts
//!
//! The algorithm:
//! 1. Reproject current froxel to previous frame's UV coordinates
//! 2. Sample previous frame's accumulated lighting
//! 3. Blend current and previous using configurable blend factor
//! 4. Apply history rejection for disoccluded regions
//!
//! # Blending Modes
//!
//! - **Additive**: fog_color + scene_color (for emissive fog)
//! - **AlphaBlend**: lerp(scene_color, fog_color, fog_alpha)
//! - **Premultiplied**: fog_color + scene_color * (1 - fog_alpha)
//! - **FogTransmittance**: scene_color * transmittance + inscatter

use bytemuck::{Pod, Zeroable};

use crate::froxel::{FroxelConfig, FroxelQuality};
use crate::volumetric_fog::VolumetricFog;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default temporal blend factor (higher = more history weight).
pub const DEFAULT_TEMPORAL_BLEND: f32 = 0.9;

/// Minimum temporal blend factor.
pub const MIN_TEMPORAL_BLEND: f32 = 0.0;

/// Maximum temporal blend factor.
pub const MAX_TEMPORAL_BLEND: f32 = 0.99;

/// Default history rejection threshold for disocclusion detection.
pub const DEFAULT_REJECTION_THRESHOLD: f32 = 0.1;

/// Maximum ray march steps for accumulation.
pub const MAX_MARCH_STEPS: u32 = 128;

/// Default jitter strength for temporal anti-aliasing.
pub const DEFAULT_JITTER_STRENGTH: f32 = 0.5;

// ---------------------------------------------------------------------------
// BlendMode — Compositing blend modes
// ---------------------------------------------------------------------------

/// Blend mode for compositing fog with scene color.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BlendMode {
    /// Additive: result = fog + scene
    ///
    /// Use for emissive fog or god rays.
    Additive,
    /// Alpha blend: result = lerp(scene, fog, alpha)
    ///
    /// Standard transparency blending.
    #[default]
    AlphaBlend,
    /// Premultiplied alpha: result = fog + scene * (1 - alpha)
    ///
    /// More efficient for GPU; fog color should be premultiplied.
    Premultiplied,
    /// Physically-based fog: result = scene * transmittance + inscatter
    ///
    /// Most accurate for volumetric rendering.
    FogTransmittance,
}

impl BlendMode {
    /// Get the blend mode from a string name.
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "additive" | "add" => Some(BlendMode::Additive),
            "alpha" | "alphablend" | "alpha_blend" => Some(BlendMode::AlphaBlend),
            "premultiplied" | "premult" => Some(BlendMode::Premultiplied),
            "transmittance" | "fog" | "fogtransmittance" => Some(BlendMode::FogTransmittance),
            _ => None,
        }
    }

    /// Get the name string for this blend mode.
    #[inline]
    pub fn name(&self) -> &'static str {
        match self {
            BlendMode::Additive => "additive",
            BlendMode::AlphaBlend => "alpha_blend",
            BlendMode::Premultiplied => "premultiplied",
            BlendMode::FogTransmittance => "fog_transmittance",
        }
    }

    /// Blend fog color with scene color using this mode.
    ///
    /// # Arguments
    ///
    /// * `scene` - Scene color RGB.
    /// * `fog` - Fog color RGB.
    /// * `alpha` - Fog opacity (0 = transparent, 1 = opaque).
    /// * `transmittance` - Per-channel transmittance (only for FogTransmittance mode).
    ///
    /// # Returns
    ///
    /// Blended RGBA color.
    #[inline]
    pub fn blend(
        &self,
        scene: [f32; 3],
        fog: [f32; 3],
        alpha: f32,
        transmittance: [f32; 3],
    ) -> [f32; 4] {
        match self {
            BlendMode::Additive => [
                scene[0] + fog[0],
                scene[1] + fog[1],
                scene[2] + fog[2],
                1.0,
            ],
            BlendMode::AlphaBlend => {
                let inv_alpha = 1.0 - alpha;
                [
                    scene[0] * inv_alpha + fog[0] * alpha,
                    scene[1] * inv_alpha + fog[1] * alpha,
                    scene[2] * inv_alpha + fog[2] * alpha,
                    1.0,
                ]
            }
            BlendMode::Premultiplied => {
                let inv_alpha = 1.0 - alpha;
                [
                    fog[0] + scene[0] * inv_alpha,
                    fog[1] + scene[1] * inv_alpha,
                    fog[2] + scene[2] * inv_alpha,
                    1.0,
                ]
            }
            BlendMode::FogTransmittance => [
                scene[0] * transmittance[0] + fog[0],
                scene[1] * transmittance[1] + fog[1],
                scene[2] * transmittance[2] + fog[2],
                1.0,
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// CompositeConfig — GPU-uploadable configuration
// ---------------------------------------------------------------------------

/// Configuration for the froxel compositing pass.
///
/// This struct is designed to be uploaded to the GPU as a uniform buffer.
/// The layout is `repr(C)` and implements `Pod` for bytemuck compatibility.
///
/// # Memory Layout (64 bytes)
///
/// | Offset | Field                  | Size    |
/// |--------|------------------------|---------|
/// | 0      | temporal_blend         | 4 bytes |
/// | 4      | rejection_threshold    | 4 bytes |
/// | 8      | jitter_strength        | 4 bytes |
/// | 12     | blend_mode             | 4 bytes |
/// | 16     | max_steps              | 4 bytes |
/// | 20     | step_size              | 4 bytes |
/// | 24     | density_scale          | 4 bytes |
/// | 28     | inscatter_scale        | 4 bytes |
/// | 32     | ambient_intensity      | 4 bytes |
/// | 36     | enabled_flags          | 4 bytes |
/// | 40     | frame_index            | 4 bytes |
/// | 44     | _padding               | 20 bytes |
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct CompositeConfig {
    /// Temporal blend factor (0 = no history, 1 = all history).
    ///
    /// Higher values = smoother but more ghosting.
    /// Typical range: 0.85-0.95
    pub temporal_blend: f32,

    /// Threshold for rejecting history samples.
    ///
    /// Lower values = more aggressive rejection.
    /// Typical range: 0.05-0.2
    pub rejection_threshold: f32,

    /// Jitter strength for temporal anti-aliasing.
    ///
    /// Range: 0.0-1.0
    pub jitter_strength: f32,

    /// Blend mode as u32 (0=Additive, 1=AlphaBlend, 2=Premult, 3=Transmittance).
    pub blend_mode: u32,

    /// Maximum ray march steps for accumulation.
    pub max_steps: u32,

    /// Step size for ray marching (in froxel units).
    pub step_size: f32,

    /// Global density multiplier.
    pub density_scale: f32,

    /// Inscatter intensity multiplier.
    pub inscatter_scale: f32,

    /// Ambient fog intensity (0-1).
    pub ambient_intensity: f32,

    /// Feature flags (bit 0 = temporal enabled, bit 1 = noise enabled).
    pub enabled_flags: u32,

    /// Current frame index for jitter sequence.
    pub frame_index: u32,

    /// Padding for 16-byte alignment.
    pub _padding: [u32; 5],
}

impl CompositeConfig {
    /// Bit flag for temporal reprojection enabled.
    pub const FLAG_TEMPORAL_ENABLED: u32 = 1 << 0;
    /// Bit flag for noise modulation enabled.
    pub const FLAG_NOISE_ENABLED: u32 = 1 << 1;
    /// Bit flag for history rejection enabled.
    pub const FLAG_REJECTION_ENABLED: u32 = 1 << 2;
    /// Bit flag for depth-aware sampling enabled.
    pub const FLAG_DEPTH_AWARE: u32 = 1 << 3;

    /// Create a new composite configuration.
    pub fn new(blend_mode: BlendMode) -> Self {
        Self {
            temporal_blend: DEFAULT_TEMPORAL_BLEND,
            rejection_threshold: DEFAULT_REJECTION_THRESHOLD,
            jitter_strength: DEFAULT_JITTER_STRENGTH,
            blend_mode: blend_mode as u32,
            max_steps: 64,
            step_size: 1.0,
            density_scale: 1.0,
            inscatter_scale: 1.0,
            ambient_intensity: 0.1,
            enabled_flags: Self::FLAG_TEMPORAL_ENABLED | Self::FLAG_REJECTION_ENABLED,
            frame_index: 0,
            _padding: [0; 5],
        }
    }

    /// Set the temporal blend factor.
    pub fn with_temporal_blend(mut self, blend: f32) -> Self {
        self.temporal_blend = blend.clamp(MIN_TEMPORAL_BLEND, MAX_TEMPORAL_BLEND);
        self
    }

    /// Set the rejection threshold.
    pub fn with_rejection_threshold(mut self, threshold: f32) -> Self {
        self.rejection_threshold = threshold.max(0.0);
        self
    }

    /// Set the jitter strength.
    pub fn with_jitter(mut self, strength: f32) -> Self {
        self.jitter_strength = strength.clamp(0.0, 1.0);
        self
    }

    /// Set the maximum ray march steps.
    pub fn with_max_steps(mut self, steps: u32) -> Self {
        self.max_steps = steps.clamp(1, MAX_MARCH_STEPS);
        self
    }

    /// Set the density scale.
    pub fn with_density_scale(mut self, scale: f32) -> Self {
        self.density_scale = scale.max(0.0);
        self
    }

    /// Enable or disable temporal reprojection.
    pub fn with_temporal(mut self, enabled: bool) -> Self {
        if enabled {
            self.enabled_flags |= Self::FLAG_TEMPORAL_ENABLED;
        } else {
            self.enabled_flags &= !Self::FLAG_TEMPORAL_ENABLED;
        }
        self
    }

    /// Enable or disable noise modulation.
    pub fn with_noise(mut self, enabled: bool) -> Self {
        if enabled {
            self.enabled_flags |= Self::FLAG_NOISE_ENABLED;
        } else {
            self.enabled_flags &= !Self::FLAG_NOISE_ENABLED;
        }
        self
    }

    /// Check if temporal reprojection is enabled.
    #[inline]
    pub fn temporal_enabled(&self) -> bool {
        (self.enabled_flags & Self::FLAG_TEMPORAL_ENABLED) != 0
    }

    /// Check if noise modulation is enabled.
    #[inline]
    pub fn noise_enabled(&self) -> bool {
        (self.enabled_flags & Self::FLAG_NOISE_ENABLED) != 0
    }

    /// Check if history rejection is enabled.
    #[inline]
    pub fn rejection_enabled(&self) -> bool {
        (self.enabled_flags & Self::FLAG_REJECTION_ENABLED) != 0
    }

    /// Get the blend mode enum.
    #[inline]
    pub fn get_blend_mode(&self) -> BlendMode {
        match self.blend_mode {
            0 => BlendMode::Additive,
            1 => BlendMode::AlphaBlend,
            2 => BlendMode::Premultiplied,
            3 => BlendMode::FogTransmittance,
            _ => BlendMode::AlphaBlend,
        }
    }

    /// Advance to the next frame.
    pub fn advance_frame(&mut self) {
        self.frame_index = self.frame_index.wrapping_add(1);
    }

    /// Validate the configuration.
    pub fn is_valid(&self) -> bool {
        self.temporal_blend >= MIN_TEMPORAL_BLEND
            && self.temporal_blend <= MAX_TEMPORAL_BLEND
            && self.rejection_threshold >= 0.0
            && self.jitter_strength >= 0.0
            && self.jitter_strength <= 1.0
            && self.max_steps >= 1
            && self.max_steps <= MAX_MARCH_STEPS
            && self.step_size > 0.0
            && self.density_scale >= 0.0
            && self.inscatter_scale >= 0.0
            && self.blend_mode <= 3
    }
}

impl Default for CompositeConfig {
    fn default() -> Self {
        Self::new(BlendMode::FogTransmittance)
    }
}

// ---------------------------------------------------------------------------
// TemporalData — Per-pixel temporal reprojection data
// ---------------------------------------------------------------------------

/// Temporal history data for a single pixel/froxel.
///
/// This struct is `repr(C)` and `Pod` for GPU transfer.
///
/// # Memory Layout (32 bytes)
///
/// | Offset | Field            | Size    |
/// |--------|------------------|---------|
/// | 0      | accumulated_rgb  | 12 bytes |
/// | 12     | accumulated_a    | 4 bytes |
/// | 16     | transmittance    | 12 bytes |
/// | 28     | frame_age        | 4 bytes |
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct TemporalData {
    /// Accumulated inscattered light RGB.
    pub accumulated_rgb: [f32; 3],
    /// Accumulated opacity/alpha.
    pub accumulated_a: f32,
    /// Accumulated transmittance RGB.
    pub transmittance: [f32; 3],
    /// Number of frames since last valid sample.
    pub frame_age: u32,
}

impl TemporalData {
    /// Create temporal data from current frame values.
    pub fn new(inscatter: [f32; 3], alpha: f32, transmittance: [f32; 3]) -> Self {
        Self {
            accumulated_rgb: inscatter,
            accumulated_a: alpha,
            transmittance,
            frame_age: 0,
        }
    }

    /// Create zeroed temporal data (no history).
    #[inline]
    pub fn empty() -> Self {
        Self {
            accumulated_rgb: [0.0; 3],
            accumulated_a: 0.0,
            transmittance: [1.0, 1.0, 1.0],
            frame_age: u32::MAX,
        }
    }

    /// Check if this data is valid for blending.
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.frame_age < u32::MAX
    }

    /// Blend current frame data with this history.
    ///
    /// # Arguments
    ///
    /// * `current` - Current frame's temporal data.
    /// * `blend_factor` - Weight for history (0 = use current, 1 = use history).
    ///
    /// # Returns
    ///
    /// Blended temporal data.
    pub fn blend(&self, current: &TemporalData, blend_factor: f32) -> TemporalData {
        let inv_blend = 1.0 - blend_factor;
        TemporalData {
            accumulated_rgb: [
                self.accumulated_rgb[0] * blend_factor + current.accumulated_rgb[0] * inv_blend,
                self.accumulated_rgb[1] * blend_factor + current.accumulated_rgb[1] * inv_blend,
                self.accumulated_rgb[2] * blend_factor + current.accumulated_rgb[2] * inv_blend,
            ],
            accumulated_a: self.accumulated_a * blend_factor + current.accumulated_a * inv_blend,
            transmittance: [
                self.transmittance[0] * blend_factor + current.transmittance[0] * inv_blend,
                self.transmittance[1] * blend_factor + current.transmittance[1] * inv_blend,
                self.transmittance[2] * blend_factor + current.transmittance[2] * inv_blend,
            ],
            frame_age: 0,
        }
    }

    /// Increment frame age (mark as stale).
    pub fn age(&mut self) {
        if self.frame_age < u32::MAX {
            self.frame_age = self.frame_age.saturating_add(1);
        }
    }
}

impl Default for TemporalData {
    fn default() -> Self {
        Self::empty()
    }
}

// ---------------------------------------------------------------------------
// AccumulationResult — Ray march accumulation output
// ---------------------------------------------------------------------------

/// Result of ray marching through the froxel volume.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AccumulationResult {
    /// Total inscattered light RGB.
    pub inscatter: [f32; 3],
    /// Per-channel transmittance (fraction of background visible).
    pub transmittance: [f32; 3],
    /// Total optical depth traversed.
    pub optical_depth: [f32; 3],
    /// Number of steps taken.
    pub steps_taken: u32,
    /// Total distance marched.
    pub distance_marched: f32,
}

impl AccumulationResult {
    /// Create an empty result (fully transparent, no inscatter).
    pub fn empty() -> Self {
        Self {
            inscatter: [0.0; 3],
            transmittance: [1.0, 1.0, 1.0],
            optical_depth: [0.0; 3],
            steps_taken: 0,
            distance_marched: 0.0,
        }
    }

    /// Create a result representing fully opaque fog.
    pub fn opaque(inscatter: [f32; 3]) -> Self {
        Self {
            inscatter,
            transmittance: [0.0, 0.0, 0.0],
            optical_depth: [f32::MAX, f32::MAX, f32::MAX],
            steps_taken: 1,
            distance_marched: 0.0,
        }
    }

    /// Calculate effective alpha from transmittance.
    #[inline]
    pub fn alpha(&self) -> f32 {
        let avg_trans = (self.transmittance[0] + self.transmittance[1] + self.transmittance[2]) / 3.0;
        1.0 - avg_trans
    }

    /// Check if the result is essentially fully transparent.
    #[inline]
    pub fn is_transparent(&self) -> bool {
        self.transmittance[0] > 0.999
            && self.transmittance[1] > 0.999
            && self.transmittance[2] > 0.999
    }

    /// Check if the result is essentially fully opaque.
    #[inline]
    pub fn is_opaque(&self) -> bool {
        self.transmittance[0] < 0.001
            && self.transmittance[1] < 0.001
            && self.transmittance[2] < 0.001
    }

    /// Accumulate a sample into this result.
    ///
    /// # Arguments
    ///
    /// * `sample_inscatter` - Inscattered light at sample point.
    /// * `sample_extinction` - Extinction coefficient at sample point.
    /// * `step_distance` - Distance of this step.
    pub fn accumulate(
        &mut self,
        sample_inscatter: [f32; 3],
        sample_extinction: [f32; 3],
        step_distance: f32,
    ) {
        // Beer-Lambert transmittance for this step
        let step_trans = [
            (-sample_extinction[0] * step_distance).exp(),
            (-sample_extinction[1] * step_distance).exp(),
            (-sample_extinction[2] * step_distance).exp(),
        ];

        // Inscattered light weighted by transmittance so far
        let integrated = [
            sample_inscatter[0] * self.transmittance[0] * (1.0 - step_trans[0]),
            sample_inscatter[1] * self.transmittance[1] * (1.0 - step_trans[1]),
            sample_inscatter[2] * self.transmittance[2] * (1.0 - step_trans[2]),
        ];

        // Accumulate inscatter
        self.inscatter[0] += integrated[0];
        self.inscatter[1] += integrated[1];
        self.inscatter[2] += integrated[2];

        // Update transmittance
        self.transmittance[0] *= step_trans[0];
        self.transmittance[1] *= step_trans[1];
        self.transmittance[2] *= step_trans[2];

        // Update optical depth
        self.optical_depth[0] += sample_extinction[0] * step_distance;
        self.optical_depth[1] += sample_extinction[1] * step_distance;
        self.optical_depth[2] += sample_extinction[2] * step_distance;

        self.steps_taken += 1;
        self.distance_marched += step_distance;
    }
}

impl Default for AccumulationResult {
    fn default() -> Self {
        Self::empty()
    }
}

// ---------------------------------------------------------------------------
// TemporalResult — Temporal reprojection output
// ---------------------------------------------------------------------------

/// Result of temporal reprojection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TemporalResult {
    /// Blended temporal data.
    pub data: TemporalData,
    /// Whether history was successfully sampled.
    pub history_valid: bool,
    /// Whether history was rejected due to disocclusion.
    pub was_rejected: bool,
    /// Confidence in the temporal blend (0-1).
    pub confidence: f32,
    /// UV offset from reprojection.
    pub uv_offset: [f32; 2],
}

impl TemporalResult {
    /// Create a result with no valid history.
    pub fn no_history(current: TemporalData) -> Self {
        Self {
            data: current,
            history_valid: false,
            was_rejected: false,
            confidence: 0.0,
            uv_offset: [0.0, 0.0],
        }
    }

    /// Create a result with rejected history.
    pub fn rejected(current: TemporalData, uv_offset: [f32; 2]) -> Self {
        Self {
            data: current,
            history_valid: true,
            was_rejected: true,
            confidence: 0.0,
            uv_offset,
        }
    }

    /// Create a successful temporal blend result.
    pub fn success(data: TemporalData, confidence: f32, uv_offset: [f32; 2]) -> Self {
        Self {
            data,
            history_valid: true,
            was_rejected: false,
            confidence: confidence.clamp(0.0, 1.0),
            uv_offset,
        }
    }
}

impl Default for TemporalResult {
    fn default() -> Self {
        Self::no_history(TemporalData::empty())
    }
}

// ---------------------------------------------------------------------------
// ReprojectionMatrix — Temporal reprojection transform
// ---------------------------------------------------------------------------

/// Matrix data for temporal reprojection.
///
/// GPU-uploadable structure containing current and previous frame matrices.
///
/// # Memory Layout (128 bytes for current, 128 bytes for previous)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct ReprojectionMatrix {
    /// Current frame's view-projection matrix (column-major, 4x4).
    pub current_view_proj: [f32; 16],
    /// Previous frame's view-projection matrix (column-major, 4x4).
    pub prev_view_proj: [f32; 16],
    /// Current frame's inverse view-projection matrix.
    pub current_view_proj_inv: [f32; 16],
    /// Previous frame's inverse view-projection matrix.
    pub prev_view_proj_inv: [f32; 16],
}

impl ReprojectionMatrix {
    /// Create identity reprojection matrices.
    pub fn identity() -> Self {
        let ident: [f32; 16] = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        Self {
            current_view_proj: ident,
            prev_view_proj: ident,
            current_view_proj_inv: ident,
            prev_view_proj_inv: ident,
        }
    }

    /// Set the current frame matrices.
    pub fn set_current(&mut self, view_proj: [f32; 16], view_proj_inv: [f32; 16]) {
        self.current_view_proj = view_proj;
        self.current_view_proj_inv = view_proj_inv;
    }

    /// Copy current to previous (call at end of frame).
    pub fn swap_history(&mut self) {
        self.prev_view_proj = self.current_view_proj;
        self.prev_view_proj_inv = self.current_view_proj_inv;
    }

    /// Reproject a position from current frame to previous frame UV.
    ///
    /// # Arguments
    ///
    /// * `current_ndc` - Position in current frame NDC space [x, y, z].
    ///
    /// # Returns
    ///
    /// Previous frame UV coordinates [u, v], or None if outside frustum.
    pub fn reproject(&self, current_ndc: [f32; 3]) -> Option<[f32; 2]> {
        // Transform NDC to world space using current inverse
        let world = Self::transform_point(&self.current_view_proj_inv, current_ndc);

        // Transform world to previous frame clip space
        let prev_clip = Self::transform_point_clip(&self.prev_view_proj, world);

        // Check if behind camera
        if prev_clip[3] <= 0.0 {
            return None;
        }

        // Perspective divide to NDC
        let inv_w = 1.0 / prev_clip[3];
        let prev_ndc = [
            prev_clip[0] * inv_w,
            prev_clip[1] * inv_w,
        ];

        // Convert NDC to UV (0-1 range)
        let uv = [
            prev_ndc[0] * 0.5 + 0.5,
            prev_ndc[1] * 0.5 + 0.5,
        ];

        // Check bounds
        if uv[0] < 0.0 || uv[0] > 1.0 || uv[1] < 0.0 || uv[1] > 1.0 {
            return None;
        }

        Some(uv)
    }

    /// Transform point by 4x4 matrix (assumes w=1).
    fn transform_point(matrix: &[f32; 16], point: [f32; 3]) -> [f32; 3] {
        let x = matrix[0] * point[0] + matrix[4] * point[1] + matrix[8] * point[2] + matrix[12];
        let y = matrix[1] * point[0] + matrix[5] * point[1] + matrix[9] * point[2] + matrix[13];
        let z = matrix[2] * point[0] + matrix[6] * point[1] + matrix[10] * point[2] + matrix[14];
        let w = matrix[3] * point[0] + matrix[7] * point[1] + matrix[11] * point[2] + matrix[15];

        if w.abs() > 1e-6 {
            [x / w, y / w, z / w]
        } else {
            [x, y, z]
        }
    }

    /// Transform point to clip space (returns [x, y, z, w]).
    fn transform_point_clip(matrix: &[f32; 16], point: [f32; 3]) -> [f32; 4] {
        let x = matrix[0] * point[0] + matrix[4] * point[1] + matrix[8] * point[2] + matrix[12];
        let y = matrix[1] * point[0] + matrix[5] * point[1] + matrix[9] * point[2] + matrix[13];
        let z = matrix[2] * point[0] + matrix[6] * point[1] + matrix[10] * point[2] + matrix[14];
        let w = matrix[3] * point[0] + matrix[7] * point[1] + matrix[11] * point[2] + matrix[15];
        [x, y, z, w]
    }
}

impl Default for ReprojectionMatrix {
    fn default() -> Self {
        Self::identity()
    }
}

// ---------------------------------------------------------------------------
// FroxelCompositor — Main compositor
// ---------------------------------------------------------------------------

/// Main froxel compositing pass manager.
///
/// Coordinates froxel volume sampling, volumetric fog evaluation,
/// temporal reprojection, and final compositing.
pub struct FroxelCompositor {
    /// Compositing configuration.
    config: CompositeConfig,

    /// Froxel volume configuration.
    froxel_config: FroxelConfig,

    /// Reprojection matrices.
    reprojection: ReprojectionMatrix,

    /// Current frame index.
    frame_index: u64,

    /// History buffer dimensions.
    history_width: u32,
    history_height: u32,

    /// Temporal history data (flattened 2D array).
    history_buffer: Vec<TemporalData>,
}

impl FroxelCompositor {
    /// Create a new froxel compositor.
    ///
    /// # Arguments
    ///
    /// * `quality` - Froxel volume quality preset.
    pub fn new(quality: FroxelQuality) -> Self {
        let froxel_config = FroxelConfig::from_quality(quality, 0.1, 100.0);
        let (w, h, _) = quality.dimensions();

        Self {
            config: CompositeConfig::default(),
            froxel_config,
            reprojection: ReprojectionMatrix::identity(),
            frame_index: 0,
            history_width: w,
            history_height: h,
            history_buffer: vec![TemporalData::empty(); (w * h) as usize],
        }
    }

    /// Create with custom configuration.
    pub fn with_config(quality: FroxelQuality, config: CompositeConfig) -> Self {
        let mut compositor = Self::new(quality);
        compositor.config = config;
        compositor
    }

    /// Get the current configuration.
    #[inline]
    pub fn config(&self) -> &CompositeConfig {
        &self.config
    }

    /// Get mutable configuration.
    #[inline]
    pub fn config_mut(&mut self) -> &mut CompositeConfig {
        &mut self.config
    }

    /// Get the froxel configuration.
    #[inline]
    pub fn froxel_config(&self) -> &FroxelConfig {
        &self.froxel_config
    }

    /// Get the reprojection matrices.
    #[inline]
    pub fn reprojection(&self) -> &ReprojectionMatrix {
        &self.reprojection
    }

    /// Get mutable reprojection matrices.
    #[inline]
    pub fn reprojection_mut(&mut self) -> &mut ReprojectionMatrix {
        &mut self.reprojection
    }

    /// Get the current frame index.
    #[inline]
    pub fn frame_index(&self) -> u64 {
        self.frame_index
    }

    /// Update the froxel configuration.
    pub fn set_froxel_config(&mut self, config: FroxelConfig) {
        // Resize history buffer if dimensions changed
        let new_size = (config.grid_width * config.grid_height) as usize;
        if new_size != self.history_buffer.len() {
            self.history_buffer = vec![TemporalData::empty(); new_size];
            self.history_width = config.grid_width;
            self.history_height = config.grid_height;
        }
        self.froxel_config = config;
    }

    /// Accumulate lighting along a view ray.
    ///
    /// # Arguments
    ///
    /// * `ray_origin` - World-space ray origin.
    /// * `ray_dir` - Normalized ray direction.
    /// * `max_distance` - Maximum ray distance.
    /// * `fog` - Volumetric fog parameters.
    /// * `light_dir` - Primary light direction (normalized).
    /// * `light_color` - Primary light color/intensity.
    /// * `time` - Current time for noise animation.
    ///
    /// # Returns
    ///
    /// Accumulation result with inscatter and transmittance.
    pub fn accumulate_lighting(
        &self,
        ray_origin: [f32; 3],
        ray_dir: [f32; 3],
        max_distance: f32,
        fog: &VolumetricFog,
        light_dir: [f32; 3],
        light_color: [f32; 3],
        time: f32,
    ) -> AccumulationResult {
        let mut result = AccumulationResult::empty();

        let max_steps = self.config.max_steps;
        let step_size = max_distance / max_steps as f32;

        // Early out for zero-length rays
        if max_distance <= 0.0 {
            return result;
        }

        for step in 0..max_steps {
            let t = (step as f32 + 0.5) * step_size;
            let pos = [
                ray_origin[0] + ray_dir[0] * t,
                ray_origin[1] + ray_dir[1] * t,
                ray_origin[2] + ray_dir[2] * t,
            ];

            // Sample density
            let density = if self.config.noise_enabled() {
                fog.sample_density_noisy(pos, time)
            } else {
                fog.sample_density(pos)
            };

            // Skip if no fog
            if density < 1e-6 {
                result.steps_taken += 1;
                result.distance_marched += step_size;
                continue;
            }

            // Scale density
            let scaled_density = density * self.config.density_scale;

            // Compute inscatter for this sample
            let inscatter = fog.compute_inscatter(
                [-ray_dir[0], -ray_dir[1], -ray_dir[2]], // View toward camera
                light_dir,
                light_color,
                scaled_density,
            );

            // Scale inscatter
            let scaled_inscatter = [
                inscatter[0] * self.config.inscatter_scale,
                inscatter[1] * self.config.inscatter_scale,
                inscatter[2] * self.config.inscatter_scale,
            ];

            // Get extinction from fog parameters
            let scattering = fog.scattering_params();
            let extinction = [
                scattering.extinction_rgb[0] * scaled_density,
                scattering.extinction_rgb[1] * scaled_density,
                scattering.extinction_rgb[2] * scaled_density,
            ];

            // Accumulate sample
            result.accumulate(scaled_inscatter, extinction, step_size);

            // Early out if fully opaque
            if result.is_opaque() {
                break;
            }
        }

        // Add ambient contribution
        if self.config.ambient_intensity > 0.0 {
            let ambient = self.config.ambient_intensity * (1.0 - result.alpha());
            result.inscatter[0] += ambient;
            result.inscatter[1] += ambient;
            result.inscatter[2] += ambient;
        }

        result
    }

    /// Perform temporal reprojection for a pixel.
    ///
    /// # Arguments
    ///
    /// * `current` - Current frame's accumulated data.
    /// * `pixel_x` - Pixel X coordinate (0 to width-1).
    /// * `pixel_y` - Pixel Y coordinate (0 to height-1).
    /// * `current_depth` - Depth at pixel (for disocclusion check).
    ///
    /// # Returns
    ///
    /// Temporal reprojection result.
    pub fn temporal_reproject(
        &self,
        current: &AccumulationResult,
        pixel_x: u32,
        pixel_y: u32,
        current_depth: f32,
    ) -> TemporalResult {
        // Create current temporal data
        let current_data = TemporalData::new(
            current.inscatter,
            current.alpha(),
            current.transmittance,
        );

        // If temporal disabled, return current only
        if !self.config.temporal_enabled() {
            return TemporalResult::no_history(current_data);
        }

        // Check bounds
        if pixel_x >= self.history_width || pixel_y >= self.history_height {
            return TemporalResult::no_history(current_data);
        }

        // Convert pixel to NDC
        let ndc_x = (pixel_x as f32 + 0.5) / self.history_width as f32 * 2.0 - 1.0;
        let ndc_y = (pixel_y as f32 + 0.5) / self.history_height as f32 * 2.0 - 1.0;
        let ndc_z = self.froxel_config.near_plane / current_depth.max(0.001);

        // Reproject to previous frame
        let prev_uv = match self.reprojection.reproject([ndc_x, ndc_y, ndc_z]) {
            Some(uv) => uv,
            None => return TemporalResult::no_history(current_data),
        };

        // Sample history at reprojected location (nearest neighbor for now)
        let hist_x = (prev_uv[0] * self.history_width as f32).floor() as i32;
        let hist_y = (prev_uv[1] * self.history_height as f32).floor() as i32;

        if hist_x < 0 || hist_x >= self.history_width as i32
            || hist_y < 0 || hist_y >= self.history_height as i32
        {
            return TemporalResult::no_history(current_data);
        }

        let hist_idx = (hist_y as u32 * self.history_width + hist_x as u32) as usize;
        let history = &self.history_buffer[hist_idx];

        // Check if history is valid
        if !history.is_valid() {
            return TemporalResult::no_history(current_data);
        }

        // UV offset for debugging
        let uv_offset = [
            prev_uv[0] - (pixel_x as f32 + 0.5) / self.history_width as f32,
            prev_uv[1] - (pixel_y as f32 + 0.5) / self.history_height as f32,
        ];

        // Check for disocclusion (history rejection)
        if self.config.rejection_enabled() {
            let color_diff = Self::color_difference(
                &history.accumulated_rgb,
                &current_data.accumulated_rgb,
            );

            if color_diff > self.config.rejection_threshold {
                return TemporalResult::rejected(current_data, uv_offset);
            }
        }

        // Blend history with current
        let blend_factor = self.config.temporal_blend;
        let blended = history.blend(&current_data, blend_factor);

        // Compute confidence based on history age and color difference
        let age_factor = 1.0 / (1.0 + history.frame_age as f32 * 0.1);
        let confidence = blend_factor * age_factor;

        TemporalResult::success(blended, confidence, uv_offset)
    }

    /// Calculate color difference for rejection test.
    fn color_difference(a: &[f32; 3], b: &[f32; 3]) -> f32 {
        let dr = a[0] - b[0];
        let dg = a[1] - b[1];
        let db = a[2] - b[2];
        (dr * dr + dg * dg + db * db).sqrt()
    }

    /// Composite accumulated fog with scene color.
    ///
    /// # Arguments
    ///
    /// * `scene_color` - Background scene color RGB.
    /// * `temporal_result` - Result from temporal reprojection.
    ///
    /// # Returns
    ///
    /// Final composited RGBA color.
    pub fn composite_to_scene(
        &self,
        scene_color: [f32; 3],
        temporal_result: &TemporalResult,
    ) -> [f32; 4] {
        let data = &temporal_result.data;
        let blend_mode = self.config.get_blend_mode();

        blend_mode.blend(
            scene_color,
            data.accumulated_rgb,
            data.accumulated_a,
            data.transmittance,
        )
    }

    /// Update history buffer with current frame data.
    ///
    /// # Arguments
    ///
    /// * `pixel_x` - Pixel X coordinate.
    /// * `pixel_y` - Pixel Y coordinate.
    /// * `data` - Temporal data to store.
    pub fn update_history(&mut self, pixel_x: u32, pixel_y: u32, data: TemporalData) {
        if pixel_x < self.history_width && pixel_y < self.history_height {
            let idx = (pixel_y * self.history_width + pixel_x) as usize;
            self.history_buffer[idx] = data;
        }
    }

    /// Age all history samples (call at start of frame).
    pub fn age_history(&mut self) {
        for data in &mut self.history_buffer {
            data.age();
        }
    }

    /// Clear all history (e.g., on camera cut).
    pub fn clear_history(&mut self) {
        for data in &mut self.history_buffer {
            *data = TemporalData::empty();
        }
    }

    /// Begin a new frame.
    ///
    /// Updates frame counter and ages history.
    pub fn begin_frame(&mut self) {
        self.frame_index += 1;
        self.config.advance_frame();
        self.age_history();
    }

    /// End frame and swap reprojection matrices.
    pub fn end_frame(&mut self) {
        self.reprojection.swap_history();
    }

    /// Get jitter offset for current frame.
    ///
    /// Returns sub-pixel jitter for temporal anti-aliasing.
    pub fn get_jitter_offset(&self) -> [f32; 2] {
        if self.config.jitter_strength <= 0.0 {
            return [0.0, 0.0];
        }

        // Halton sequence for low-discrepancy jitter
        let idx = (self.config.frame_index % 16) as usize;
        let halton_2 = Self::halton(idx, 2);
        let halton_3 = Self::halton(idx, 3);

        let strength = self.config.jitter_strength;
        [
            (halton_2 - 0.5) * strength,
            (halton_3 - 0.5) * strength,
        ]
    }

    /// Halton sequence for low-discrepancy sampling.
    fn halton(mut index: usize, base: usize) -> f32 {
        let mut result = 0.0;
        let mut f = 1.0;
        while index > 0 {
            f /= base as f32;
            result += f * (index % base) as f32;
            index /= base;
        }
        result
    }

    /// Calculate metrics for the current frame.
    pub fn compute_metrics(&self) -> CompositeMetrics {
        let mut valid_samples = 0u32;
        let mut total_age = 0u64;
        let mut min_transmittance = 1.0f32;
        let mut max_inscatter = 0.0f32;

        for data in &self.history_buffer {
            if data.is_valid() {
                valid_samples += 1;
                total_age += data.frame_age as u64;

                let avg_trans = (data.transmittance[0] + data.transmittance[1] + data.transmittance[2]) / 3.0;
                min_transmittance = min_transmittance.min(avg_trans);

                let lum = data.accumulated_rgb[0] * 0.299
                    + data.accumulated_rgb[1] * 0.587
                    + data.accumulated_rgb[2] * 0.114;
                max_inscatter = max_inscatter.max(lum);
            }
        }

        let total_samples = self.history_buffer.len() as u32;
        let coverage = valid_samples as f32 / total_samples.max(1) as f32;
        let avg_age = if valid_samples > 0 {
            total_age as f32 / valid_samples as f32
        } else {
            0.0
        };

        CompositeMetrics {
            frame_index: self.frame_index,
            valid_history_samples: valid_samples,
            total_history_samples: total_samples,
            history_coverage: coverage,
            average_history_age: avg_age,
            min_transmittance,
            max_inscatter_luminance: max_inscatter,
        }
    }
}

impl Default for FroxelCompositor {
    fn default() -> Self {
        Self::new(FroxelQuality::Medium)
    }
}

// ---------------------------------------------------------------------------
// CompositeMetrics — Performance metrics
// ---------------------------------------------------------------------------

/// Metrics for monitoring compositor performance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompositeMetrics {
    /// Current frame index.
    pub frame_index: u64,
    /// Number of valid history samples.
    pub valid_history_samples: u32,
    /// Total history buffer size.
    pub total_history_samples: u32,
    /// Fraction of history that is valid (0-1).
    pub history_coverage: f32,
    /// Average age of valid history samples.
    pub average_history_age: f32,
    /// Minimum transmittance in scene.
    pub min_transmittance: f32,
    /// Maximum inscatter luminance.
    pub max_inscatter_luminance: f32,
}

impl Default for CompositeMetrics {
    fn default() -> Self {
        Self {
            frame_index: 0,
            valid_history_samples: 0,
            total_history_samples: 0,
            history_coverage: 0.0,
            average_history_age: 0.0,
            min_transmittance: 1.0,
            max_inscatter_luminance: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::froxel::FroxelVolume;
    use crate::volumetric_fog::{FogDensityConfig, ScatteringParams};

    const EPSILON: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    fn approx_eq_eps(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    // -----------------------------------------------------------------------
    // BlendMode tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_blend_mode_from_name() {
        assert_eq!(BlendMode::from_name("additive"), Some(BlendMode::Additive));
        assert_eq!(BlendMode::from_name("ADD"), Some(BlendMode::Additive));
        assert_eq!(BlendMode::from_name("alpha"), Some(BlendMode::AlphaBlend));
        assert_eq!(BlendMode::from_name("alphablend"), Some(BlendMode::AlphaBlend));
        assert_eq!(BlendMode::from_name("premultiplied"), Some(BlendMode::Premultiplied));
        assert_eq!(BlendMode::from_name("premult"), Some(BlendMode::Premultiplied));
        assert_eq!(BlendMode::from_name("transmittance"), Some(BlendMode::FogTransmittance));
        assert_eq!(BlendMode::from_name("fog"), Some(BlendMode::FogTransmittance));
        assert_eq!(BlendMode::from_name("invalid"), None);
    }

    #[test]
    fn test_blend_mode_name() {
        assert_eq!(BlendMode::Additive.name(), "additive");
        assert_eq!(BlendMode::AlphaBlend.name(), "alpha_blend");
        assert_eq!(BlendMode::Premultiplied.name(), "premultiplied");
        assert_eq!(BlendMode::FogTransmittance.name(), "fog_transmittance");
    }

    #[test]
    fn test_blend_mode_default() {
        assert_eq!(BlendMode::default(), BlendMode::AlphaBlend);
    }

    #[test]
    fn test_blend_additive() {
        let scene = [0.5, 0.5, 0.5];
        let fog = [0.1, 0.2, 0.3];
        let result = BlendMode::Additive.blend(scene, fog, 0.5, [1.0; 3]);

        assert!(approx_eq(result[0], 0.6));
        assert!(approx_eq(result[1], 0.7));
        assert!(approx_eq(result[2], 0.8));
        assert!(approx_eq(result[3], 1.0));
    }

    #[test]
    fn test_blend_alpha_zero() {
        let scene = [1.0, 0.5, 0.0];
        let fog = [0.0, 0.0, 1.0];
        let result = BlendMode::AlphaBlend.blend(scene, fog, 0.0, [1.0; 3]);

        // Alpha = 0 means fully scene
        assert!(approx_eq(result[0], 1.0));
        assert!(approx_eq(result[1], 0.5));
        assert!(approx_eq(result[2], 0.0));
    }

    #[test]
    fn test_blend_alpha_one() {
        let scene = [1.0, 0.5, 0.0];
        let fog = [0.0, 0.0, 1.0];
        let result = BlendMode::AlphaBlend.blend(scene, fog, 1.0, [1.0; 3]);

        // Alpha = 1 means fully fog
        assert!(approx_eq(result[0], 0.0));
        assert!(approx_eq(result[1], 0.0));
        assert!(approx_eq(result[2], 1.0));
    }

    #[test]
    fn test_blend_alpha_half() {
        let scene = [1.0, 0.0, 0.0];
        let fog = [0.0, 0.0, 1.0];
        let result = BlendMode::AlphaBlend.blend(scene, fog, 0.5, [1.0; 3]);

        // 50% blend
        assert!(approx_eq(result[0], 0.5));
        assert!(approx_eq(result[1], 0.0));
        assert!(approx_eq(result[2], 0.5));
    }

    #[test]
    fn test_blend_premultiplied() {
        let scene = [1.0, 1.0, 1.0];
        let fog = [0.5, 0.5, 0.5]; // Premultiplied by alpha
        let alpha = 0.5;
        let result = BlendMode::Premultiplied.blend(scene, fog, alpha, [1.0; 3]);

        // fog + scene * (1 - alpha)
        assert!(approx_eq(result[0], 1.0));
        assert!(approx_eq(result[1], 1.0));
        assert!(approx_eq(result[2], 1.0));
    }

    #[test]
    fn test_blend_transmittance() {
        let scene = [1.0, 1.0, 1.0];
        let fog = [0.2, 0.2, 0.2];
        let transmittance = [0.5, 0.5, 0.5];
        let result = BlendMode::FogTransmittance.blend(scene, fog, 0.0, transmittance);

        // scene * trans + fog
        assert!(approx_eq(result[0], 0.7));
        assert!(approx_eq(result[1], 0.7));
        assert!(approx_eq(result[2], 0.7));
    }

    #[test]
    fn test_blend_transmittance_zero() {
        let scene = [1.0, 1.0, 1.0];
        let fog = [0.5, 0.5, 0.5];
        let transmittance = [0.0, 0.0, 0.0];
        let result = BlendMode::FogTransmittance.blend(scene, fog, 0.0, transmittance);

        // Fully opaque fog
        assert!(approx_eq(result[0], 0.5));
        assert!(approx_eq(result[1], 0.5));
        assert!(approx_eq(result[2], 0.5));
    }

    #[test]
    fn test_blend_transmittance_full() {
        let scene = [0.8, 0.6, 0.4];
        let fog = [0.0, 0.0, 0.0];
        let transmittance = [1.0, 1.0, 1.0];
        let result = BlendMode::FogTransmittance.blend(scene, fog, 0.0, transmittance);

        // Fully transparent fog
        assert!(approx_eq(result[0], 0.8));
        assert!(approx_eq(result[1], 0.6));
        assert!(approx_eq(result[2], 0.4));
    }

    // -----------------------------------------------------------------------
    // CompositeConfig tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_composite_config_default() {
        let config = CompositeConfig::default();
        assert!(config.is_valid());
        assert!(config.temporal_enabled());
        assert!(config.rejection_enabled());
        assert!(!config.noise_enabled());
        assert_eq!(config.get_blend_mode(), BlendMode::FogTransmittance);
    }

    #[test]
    fn test_composite_config_new() {
        let config = CompositeConfig::new(BlendMode::Additive);
        assert_eq!(config.get_blend_mode(), BlendMode::Additive);
        assert_eq!(config.blend_mode, 0);
    }

    #[test]
    fn test_composite_config_temporal_blend_clamping() {
        let config = CompositeConfig::default().with_temporal_blend(1.5);
        assert_eq!(config.temporal_blend, MAX_TEMPORAL_BLEND);

        let config = CompositeConfig::default().with_temporal_blend(-0.5);
        assert_eq!(config.temporal_blend, MIN_TEMPORAL_BLEND);
    }

    #[test]
    fn test_composite_config_jitter_clamping() {
        let config = CompositeConfig::default().with_jitter(2.0);
        assert_eq!(config.jitter_strength, 1.0);

        let config = CompositeConfig::default().with_jitter(-1.0);
        assert_eq!(config.jitter_strength, 0.0);
    }

    #[test]
    fn test_composite_config_max_steps_clamping() {
        let config = CompositeConfig::default().with_max_steps(0);
        assert_eq!(config.max_steps, 1);

        let config = CompositeConfig::default().with_max_steps(1000);
        assert_eq!(config.max_steps, MAX_MARCH_STEPS);
    }

    #[test]
    fn test_composite_config_flags() {
        let config = CompositeConfig::default()
            .with_temporal(false)
            .with_noise(true);

        assert!(!config.temporal_enabled());
        assert!(config.noise_enabled());

        let config = config.with_temporal(true).with_noise(false);
        assert!(config.temporal_enabled());
        assert!(!config.noise_enabled());
    }

    #[test]
    fn test_composite_config_advance_frame() {
        let mut config = CompositeConfig::default();
        assert_eq!(config.frame_index, 0);

        config.advance_frame();
        assert_eq!(config.frame_index, 1);

        config.advance_frame();
        assert_eq!(config.frame_index, 2);
    }

    #[test]
    fn test_composite_config_validation() {
        let valid = CompositeConfig::default();
        assert!(valid.is_valid());

        let mut invalid = valid;
        invalid.temporal_blend = 2.0;
        assert!(!invalid.is_valid());

        let mut invalid = valid;
        invalid.step_size = 0.0;
        assert!(!invalid.is_valid());

        let mut invalid = valid;
        invalid.blend_mode = 100;
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_composite_config_size() {
        assert_eq!(std::mem::size_of::<CompositeConfig>(), 64);
    }

    #[test]
    fn test_composite_config_pod() {
        let config = CompositeConfig::default();
        let bytes: &[u8] = bytemuck::bytes_of(&config);
        assert_eq!(bytes.len(), 64);
    }

    // -----------------------------------------------------------------------
    // TemporalData tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_temporal_data_new() {
        let data = TemporalData::new([0.1, 0.2, 0.3], 0.5, [0.9, 0.8, 0.7]);
        assert_eq!(data.accumulated_rgb, [0.1, 0.2, 0.3]);
        assert_eq!(data.accumulated_a, 0.5);
        assert_eq!(data.transmittance, [0.9, 0.8, 0.7]);
        assert_eq!(data.frame_age, 0);
    }

    #[test]
    fn test_temporal_data_empty() {
        let data = TemporalData::empty();
        assert_eq!(data.accumulated_rgb, [0.0, 0.0, 0.0]);
        assert_eq!(data.accumulated_a, 0.0);
        assert_eq!(data.transmittance, [1.0, 1.0, 1.0]);
        assert!(!data.is_valid());
    }

    #[test]
    fn test_temporal_data_validity() {
        let valid = TemporalData::new([0.1, 0.2, 0.3], 0.5, [0.9, 0.8, 0.7]);
        assert!(valid.is_valid());

        let invalid = TemporalData::empty();
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_temporal_data_blend() {
        let history = TemporalData::new([1.0, 1.0, 1.0], 1.0, [0.0, 0.0, 0.0]);
        let current = TemporalData::new([0.0, 0.0, 0.0], 0.0, [1.0, 1.0, 1.0]);

        let blended = history.blend(&current, 0.5);
        assert!(approx_eq(blended.accumulated_rgb[0], 0.5));
        assert!(approx_eq(blended.accumulated_a, 0.5));
        assert!(approx_eq(blended.transmittance[0], 0.5));
    }

    #[test]
    fn test_temporal_data_blend_all_history() {
        let history = TemporalData::new([1.0, 0.5, 0.25], 0.8, [0.2, 0.4, 0.6]);
        let current = TemporalData::new([0.0, 0.0, 0.0], 0.0, [1.0, 1.0, 1.0]);

        let blended = history.blend(&current, 1.0);
        assert!(approx_eq(blended.accumulated_rgb[0], 1.0));
        assert!(approx_eq(blended.accumulated_rgb[1], 0.5));
        assert!(approx_eq(blended.accumulated_a, 0.8));
    }

    #[test]
    fn test_temporal_data_blend_all_current() {
        let history = TemporalData::new([1.0, 1.0, 1.0], 1.0, [0.0, 0.0, 0.0]);
        let current = TemporalData::new([0.5, 0.25, 0.125], 0.3, [0.8, 0.9, 1.0]);

        let blended = history.blend(&current, 0.0);
        assert!(approx_eq(blended.accumulated_rgb[0], 0.5));
        assert!(approx_eq(blended.accumulated_rgb[1], 0.25));
        assert!(approx_eq(blended.accumulated_a, 0.3));
    }

    #[test]
    fn test_temporal_data_age() {
        let mut data = TemporalData::new([0.1, 0.2, 0.3], 0.5, [0.9, 0.8, 0.7]);
        assert_eq!(data.frame_age, 0);

        data.age();
        assert_eq!(data.frame_age, 1);

        data.age();
        assert_eq!(data.frame_age, 2);
    }

    #[test]
    fn test_temporal_data_age_saturates() {
        let mut data = TemporalData::empty();
        data.age(); // Should not overflow
        assert_eq!(data.frame_age, u32::MAX);
    }

    #[test]
    fn test_temporal_data_size() {
        assert_eq!(std::mem::size_of::<TemporalData>(), 32);
    }

    #[test]
    fn test_temporal_data_pod() {
        let data = TemporalData::default();
        let bytes: &[u8] = bytemuck::bytes_of(&data);
        assert_eq!(bytes.len(), 32);
    }

    // -----------------------------------------------------------------------
    // AccumulationResult tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_accumulation_result_empty() {
        let result = AccumulationResult::empty();
        assert_eq!(result.inscatter, [0.0, 0.0, 0.0]);
        assert_eq!(result.transmittance, [1.0, 1.0, 1.0]);
        assert_eq!(result.optical_depth, [0.0, 0.0, 0.0]);
        assert_eq!(result.steps_taken, 0);
        assert!(result.is_transparent());
        assert!(!result.is_opaque());
    }

    #[test]
    fn test_accumulation_result_opaque() {
        let result = AccumulationResult::opaque([1.0, 0.5, 0.25]);
        assert_eq!(result.inscatter, [1.0, 0.5, 0.25]);
        assert_eq!(result.transmittance, [0.0, 0.0, 0.0]);
        assert!(result.is_opaque());
        assert!(!result.is_transparent());
    }

    #[test]
    fn test_accumulation_result_alpha() {
        let result = AccumulationResult {
            transmittance: [0.5, 0.5, 0.5],
            ..AccumulationResult::empty()
        };
        assert!(approx_eq(result.alpha(), 0.5));

        let transparent = AccumulationResult::empty();
        assert!(approx_eq(transparent.alpha(), 0.0));

        let opaque = AccumulationResult::opaque([1.0, 1.0, 1.0]);
        assert!(approx_eq(opaque.alpha(), 1.0));
    }

    #[test]
    fn test_accumulation_result_accumulate_single() {
        let mut result = AccumulationResult::empty();

        result.accumulate([0.5, 0.5, 0.5], [0.1, 0.1, 0.1], 1.0);

        assert!(result.inscatter[0] > 0.0);
        assert!(result.transmittance[0] < 1.0);
        assert!(result.optical_depth[0] > 0.0);
        assert_eq!(result.steps_taken, 1);
    }

    #[test]
    fn test_accumulation_result_accumulate_multiple() {
        let mut result = AccumulationResult::empty();

        for _ in 0..5 {
            result.accumulate([0.1, 0.1, 0.1], [0.05, 0.05, 0.05], 1.0);
        }

        assert_eq!(result.steps_taken, 5);
        assert!(result.transmittance[0] < 1.0);
        assert!(result.inscatter[0] > 0.0);
    }

    #[test]
    fn test_accumulation_result_transmittance_decreases() {
        let mut result = AccumulationResult::empty();

        let initial_trans = result.transmittance[0];
        result.accumulate([0.1, 0.1, 0.1], [0.2, 0.2, 0.2], 1.0);
        let after_one = result.transmittance[0];
        result.accumulate([0.1, 0.1, 0.1], [0.2, 0.2, 0.2], 1.0);
        let after_two = result.transmittance[0];

        assert!(after_one < initial_trans);
        assert!(after_two < after_one);
    }

    #[test]
    fn test_accumulation_result_optical_depth_increases() {
        let mut result = AccumulationResult::empty();

        result.accumulate([0.1, 0.1, 0.1], [0.1, 0.1, 0.1], 1.0);
        let od1 = result.optical_depth[0];
        result.accumulate([0.1, 0.1, 0.1], [0.1, 0.1, 0.1], 1.0);
        let od2 = result.optical_depth[0];

        assert!(approx_eq(od1, 0.1));
        assert!(approx_eq(od2, 0.2));
    }

    #[test]
    fn test_accumulation_result_zero_extinction() {
        let mut result = AccumulationResult::empty();

        result.accumulate([0.5, 0.5, 0.5], [0.0, 0.0, 0.0], 1.0);

        // Zero extinction = fully transparent, no inscatter contribution
        assert_eq!(result.transmittance, [1.0, 1.0, 1.0]);
        assert_eq!(result.inscatter, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_accumulation_result_distance_marched() {
        let mut result = AccumulationResult::empty();

        result.accumulate([0.1, 0.1, 0.1], [0.1, 0.1, 0.1], 2.5);
        result.accumulate([0.1, 0.1, 0.1], [0.1, 0.1, 0.1], 3.5);

        assert!(approx_eq(result.distance_marched, 6.0));
    }

    // -----------------------------------------------------------------------
    // TemporalResult tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_temporal_result_no_history() {
        let current = TemporalData::new([0.1, 0.2, 0.3], 0.5, [0.9, 0.8, 0.7]);
        let result = TemporalResult::no_history(current);

        assert!(!result.history_valid);
        assert!(!result.was_rejected);
        assert!(approx_eq(result.confidence, 0.0));
    }

    #[test]
    fn test_temporal_result_rejected() {
        let current = TemporalData::new([0.1, 0.2, 0.3], 0.5, [0.9, 0.8, 0.7]);
        let result = TemporalResult::rejected(current, [0.1, 0.2]);

        assert!(result.history_valid);
        assert!(result.was_rejected);
        assert!(approx_eq(result.confidence, 0.0));
        assert_eq!(result.uv_offset, [0.1, 0.2]);
    }

    #[test]
    fn test_temporal_result_success() {
        let data = TemporalData::new([0.1, 0.2, 0.3], 0.5, [0.9, 0.8, 0.7]);
        let result = TemporalResult::success(data, 0.8, [0.05, 0.05]);

        assert!(result.history_valid);
        assert!(!result.was_rejected);
        assert!(approx_eq(result.confidence, 0.8));
    }

    #[test]
    fn test_temporal_result_confidence_clamped() {
        let data = TemporalData::empty();
        let result = TemporalResult::success(data, 1.5, [0.0, 0.0]);
        assert!(approx_eq(result.confidence, 1.0));

        let result = TemporalResult::success(data, -0.5, [0.0, 0.0]);
        assert!(approx_eq(result.confidence, 0.0));
    }

    // -----------------------------------------------------------------------
    // ReprojectionMatrix tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_reprojection_matrix_identity() {
        let matrix = ReprojectionMatrix::identity();
        let ident: [f32; 16] = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        assert_eq!(matrix.current_view_proj, ident);
        assert_eq!(matrix.prev_view_proj, ident);
    }

    #[test]
    fn test_reprojection_matrix_swap() {
        let mut matrix = ReprojectionMatrix::identity();

        let new_mat: [f32; 16] = [
            2.0, 0.0, 0.0, 0.0,
            0.0, 2.0, 0.0, 0.0,
            0.0, 0.0, 2.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        matrix.set_current(new_mat, new_mat);
        matrix.swap_history();

        assert_eq!(matrix.prev_view_proj, new_mat);
    }

    #[test]
    fn test_reprojection_identity_ndc() {
        let matrix = ReprojectionMatrix::identity();

        // Identity matrix: NDC should map to same UV
        let uv = matrix.reproject([0.0, 0.0, 0.5]);
        assert!(uv.is_some());
        let uv = uv.unwrap();
        assert!(approx_eq(uv[0], 0.5));
        assert!(approx_eq(uv[1], 0.5));
    }

    #[test]
    fn test_reprojection_corner_ndc() {
        let matrix = ReprojectionMatrix::identity();

        // Bottom-left corner
        let uv = matrix.reproject([-1.0, -1.0, 0.5]);
        assert!(uv.is_some());
        let uv = uv.unwrap();
        assert!(approx_eq(uv[0], 0.0));
        assert!(approx_eq(uv[1], 0.0));

        // Top-right corner
        let uv = matrix.reproject([1.0, 1.0, 0.5]);
        assert!(uv.is_some());
        let uv = uv.unwrap();
        assert!(approx_eq(uv[0], 1.0));
        assert!(approx_eq(uv[1], 1.0));
    }

    #[test]
    fn test_reprojection_size() {
        assert_eq!(std::mem::size_of::<ReprojectionMatrix>(), 256);
    }

    #[test]
    fn test_reprojection_pod() {
        let matrix = ReprojectionMatrix::default();
        let bytes: &[u8] = bytemuck::bytes_of(&matrix);
        assert_eq!(bytes.len(), 256);
    }

    // -----------------------------------------------------------------------
    // FroxelCompositor tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_compositor_new() {
        let compositor = FroxelCompositor::new(FroxelQuality::Medium);
        assert_eq!(compositor.frame_index(), 0);
        assert_eq!(compositor.history_width, 64);
        assert_eq!(compositor.history_height, 48);
    }

    #[test]
    fn test_compositor_with_config() {
        let config = CompositeConfig::new(BlendMode::Additive)
            .with_temporal_blend(0.8);
        let compositor = FroxelCompositor::with_config(FroxelQuality::Low, config);

        assert_eq!(compositor.config().get_blend_mode(), BlendMode::Additive);
        assert!(approx_eq(compositor.config().temporal_blend, 0.8));
    }

    #[test]
    fn test_compositor_begin_frame() {
        let mut compositor = FroxelCompositor::new(FroxelQuality::Low);
        assert_eq!(compositor.frame_index(), 0);

        compositor.begin_frame();
        assert_eq!(compositor.frame_index(), 1);

        compositor.begin_frame();
        assert_eq!(compositor.frame_index(), 2);
    }

    #[test]
    fn test_compositor_clear_history() {
        let mut compositor = FroxelCompositor::new(FroxelQuality::Low);

        // Update some history
        compositor.update_history(0, 0, TemporalData::new([1.0, 1.0, 1.0], 1.0, [0.0, 0.0, 0.0]));
        compositor.update_history(1, 1, TemporalData::new([0.5, 0.5, 0.5], 0.5, [0.5, 0.5, 0.5]));

        compositor.clear_history();

        // All should be invalid now
        let metrics = compositor.compute_metrics();
        assert_eq!(metrics.valid_history_samples, 0);
    }

    #[test]
    fn test_compositor_set_froxel_config() {
        let mut compositor = FroxelCompositor::new(FroxelQuality::Low);
        let initial_size = compositor.history_buffer.len();

        let new_config = FroxelConfig::from_quality(FroxelQuality::Medium, 0.1, 200.0);
        compositor.set_froxel_config(new_config);

        assert!(compositor.history_buffer.len() > initial_size);
        assert_eq!(compositor.history_width, 64);
        assert_eq!(compositor.history_height, 48);
    }

    #[test]
    fn test_compositor_accumulate_lighting_empty() {
        let compositor = FroxelCompositor::new(FroxelQuality::Low);
        let fog = VolumetricFog::new(
            FogDensityConfig::new(0.0, 0.1, 0.0), // Zero density
            ScatteringParams::default(),
        );

        let result = compositor.accumulate_lighting(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            100.0,
            &fog,
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            0.0,
        );

        assert!(result.is_transparent());
    }

    #[test]
    fn test_compositor_accumulate_lighting_with_fog() {
        let compositor = FroxelCompositor::new(FroxelQuality::Low);
        let fog = VolumetricFog::new(
            FogDensityConfig::new(0.5, 0.0, 0.0), // Uniform 0.5 density
            ScatteringParams::default(),
        );

        let result = compositor.accumulate_lighting(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            50.0,
            &fog,
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            0.0,
        );

        assert!(!result.is_transparent());
        assert!(result.inscatter[0] > 0.0);
    }

    #[test]
    fn test_compositor_accumulate_zero_distance() {
        let compositor = FroxelCompositor::new(FroxelQuality::Low);
        let fog = VolumetricFog::default();

        let result = compositor.accumulate_lighting(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            0.0, // Zero distance
            &fog,
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            0.0,
        );

        assert!(result.is_transparent());
        assert_eq!(result.steps_taken, 0);
    }

    #[test]
    fn test_compositor_temporal_disabled() {
        let config = CompositeConfig::default().with_temporal(false);
        let compositor = FroxelCompositor::with_config(FroxelQuality::Low, config);

        let accumulation = AccumulationResult::empty();
        let result = compositor.temporal_reproject(&accumulation, 0, 0, 10.0);

        assert!(!result.history_valid);
    }

    #[test]
    fn test_compositor_temporal_out_of_bounds() {
        let compositor = FroxelCompositor::new(FroxelQuality::Low);

        let accumulation = AccumulationResult::empty();
        let result = compositor.temporal_reproject(&accumulation, 1000, 1000, 10.0);

        assert!(!result.history_valid);
    }

    #[test]
    fn test_compositor_composite_to_scene() {
        let compositor = FroxelCompositor::new(FroxelQuality::Low);

        let temporal_result = TemporalResult::success(
            TemporalData::new([0.2, 0.2, 0.2], 0.5, [0.5, 0.5, 0.5]),
            0.9,
            [0.0, 0.0],
        );

        let scene = [1.0, 1.0, 1.0];
        let composited = compositor.composite_to_scene(scene, &temporal_result);

        // Should blend scene with fog
        assert!(composited[0] < 1.0);
        assert!(composited[0] > 0.0);
    }

    #[test]
    fn test_compositor_update_history() {
        let mut compositor = FroxelCompositor::new(FroxelQuality::Low);

        let data = TemporalData::new([0.5, 0.5, 0.5], 0.5, [0.5, 0.5, 0.5]);
        compositor.update_history(5, 5, data);

        let metrics = compositor.compute_metrics();
        assert!(metrics.valid_history_samples > 0);
    }

    #[test]
    fn test_compositor_jitter_offset() {
        let mut compositor = FroxelCompositor::new(FroxelQuality::Low);

        let jitter1 = compositor.get_jitter_offset();
        compositor.begin_frame();
        let jitter2 = compositor.get_jitter_offset();

        // Jitter should vary between frames
        // (Could be equal by chance, but unlikely for Halton sequence)
        assert!(jitter1[0].abs() <= 0.5);
        assert!(jitter1[1].abs() <= 0.5);
        assert!(jitter2[0].abs() <= 0.5);
        assert!(jitter2[1].abs() <= 0.5);
    }

    #[test]
    fn test_compositor_jitter_disabled() {
        let config = CompositeConfig::default().with_jitter(0.0);
        let compositor = FroxelCompositor::with_config(FroxelQuality::Low, config);

        let jitter = compositor.get_jitter_offset();
        assert_eq!(jitter, [0.0, 0.0]);
    }

    #[test]
    fn test_compositor_metrics() {
        let compositor = FroxelCompositor::new(FroxelQuality::Low);
        let metrics = compositor.compute_metrics();

        assert_eq!(metrics.frame_index, 0);
        assert_eq!(metrics.valid_history_samples, 0);
        assert_eq!(metrics.total_history_samples, 32 * 24);
        assert!(approx_eq(metrics.history_coverage, 0.0));
    }

    #[test]
    fn test_compositor_metrics_with_history() {
        let mut compositor = FroxelCompositor::new(FroxelQuality::Low);

        // Fill half the history
        let (w, h, _) = FroxelQuality::Low.dimensions();
        for y in 0..h / 2 {
            for x in 0..w {
                let data = TemporalData::new([0.1, 0.1, 0.1], 0.1, [0.9, 0.9, 0.9]);
                compositor.update_history(x, y, data);
            }
        }

        let metrics = compositor.compute_metrics();
        assert!(metrics.valid_history_samples > 0);
        assert!(metrics.history_coverage > 0.0);
    }

    // -----------------------------------------------------------------------
    // Frame-to-frame stability tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_temporal_stability_static_camera() {
        let mut compositor = FroxelCompositor::new(FroxelQuality::Low);
        let fog = VolumetricFog::new(
            FogDensityConfig::new(0.3, 0.1, 0.0),
            ScatteringParams::default(),
        );

        // Simulate multiple frames with static camera
        let mut prev_inscatter = [0.0f32; 3];

        for frame in 0..10 {
            compositor.begin_frame();

            let result = compositor.accumulate_lighting(
                [0.0, 5.0, 0.0],
                [0.0, 0.0, 1.0],
                50.0,
                &fog,
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
                0.0,
            );

            if frame > 0 {
                // Values should be stable across frames
                let diff = (result.inscatter[0] - prev_inscatter[0]).abs();
                assert!(diff < 0.01, "Frame {} instability: {}", frame, diff);
            }

            prev_inscatter = result.inscatter;
            compositor.end_frame();
        }
    }

    #[test]
    fn test_temporal_blend_convergence() {
        // Test that temporal blending converges over time
        let mut compositor = FroxelCompositor::new(FroxelQuality::Low);

        let history = TemporalData::new([1.0, 1.0, 1.0], 1.0, [0.0, 0.0, 0.0]);
        let current = TemporalData::new([0.0, 0.0, 0.0], 0.0, [1.0, 1.0, 1.0]);

        // With high blend factor, should slowly approach current
        let blend_factor = 0.9;
        let mut blended = history;

        for _ in 0..100 {
            blended = blended.blend(&current, blend_factor);
        }

        // After many iterations, should be close to current
        assert!(approx_eq_eps(blended.accumulated_rgb[0], 0.0, 0.01));
    }

    // -----------------------------------------------------------------------
    // Edge case tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_edge_case_zero_density() {
        // Disable ambient to test pure zero-density case
        let mut config = CompositeConfig::default();
        config.ambient_intensity = 0.0;
        let compositor = FroxelCompositor::with_config(FroxelQuality::Low, config);

        let fog = VolumetricFog::new(
            FogDensityConfig::new(0.0, 0.1, 0.0),
            ScatteringParams::default(),
        );

        let result = compositor.accumulate_lighting(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            100.0,
            &fog,
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            0.0,
        );

        assert!(result.is_transparent());
        assert_eq!(result.inscatter, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_edge_case_max_density() {
        let mut config = CompositeConfig::default();
        config.max_steps = 32; // Reduce for test speed
        let compositor = FroxelCompositor::with_config(FroxelQuality::Low, config);

        let fog = VolumetricFog::new(
            FogDensityConfig::new(1.0, 0.0, 0.0), // Max density everywhere
            ScatteringParams::uniform(1.0, 0.9, 0.8),
        );

        let result = compositor.accumulate_lighting(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            100.0,
            &fog,
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            0.0,
        );

        assert!(!result.is_transparent());
        assert!(result.transmittance[0] < 0.5); // Significant absorption
    }

    #[test]
    fn test_edge_case_single_step() {
        let config = CompositeConfig::default().with_max_steps(1);
        let compositor = FroxelCompositor::with_config(FroxelQuality::Low, config);

        let fog = VolumetricFog::default();

        let result = compositor.accumulate_lighting(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            10.0,
            &fog,
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            0.0,
        );

        assert_eq!(result.steps_taken, 1);
    }

    #[test]
    fn test_edge_case_history_age_overflow() {
        let mut data = TemporalData::empty();

        // Ensure aging doesn't overflow
        for _ in 0..1000 {
            data.age();
        }

        assert_eq!(data.frame_age, u32::MAX);
    }

    #[test]
    fn test_edge_case_small_froxel_grid() {
        let config = FroxelConfig::new(8, 8, 8, 0.1, 100.0, true);
        let mut compositor = FroxelCompositor::new(FroxelQuality::Low);
        compositor.set_froxel_config(config);

        assert_eq!(compositor.history_buffer.len(), 64);
    }

    #[test]
    fn test_edge_case_volume_integration() {
        // Test integration with FroxelVolume from froxel.rs
        let volume = FroxelVolume::from_quality(FroxelQuality::Medium, 0.1, 100.0);
        let compositor = FroxelCompositor::new(FroxelQuality::Medium);

        // Configurations should be compatible
        assert_eq!(volume.config().grid_width, compositor.froxel_config().grid_width);
        assert_eq!(volume.config().grid_height, compositor.froxel_config().grid_height);
    }

    #[test]
    fn test_edge_case_scattering_params_integration() {
        // Test integration with ScatteringParams from volumetric_fog.rs
        let scattering = ScatteringParams::uniform(0.1, 0.9, 0.7);
        let fog = VolumetricFog::new(FogDensityConfig::default(), scattering);

        let compositor = FroxelCompositor::new(FroxelQuality::Low);
        let result = compositor.accumulate_lighting(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            50.0,
            &fog,
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            0.0,
        );

        // Should produce valid result
        assert!(result.inscatter[0] >= 0.0);
        assert!(result.transmittance[0] >= 0.0);
        assert!(result.transmittance[0] <= 1.0);
    }
}
