//! Animation textures for GPU-based crowd rendering (T-AN-8.1).
//!
//! This module provides texture-based animation storage for efficient crowd rendering.
//! Instead of updating bone matrices per-instance every frame, bone transforms are
//! baked into textures that can be sampled in the vertex shader.
//!
//! # Architecture
//!
//! ```text
//! AnimationClip                AnimationTextureBaker
//!       |                              |
//!       v                              v
//! +-------------+     +--------------------------------+
//! | Bone Tracks | --> | Bake to RGBA32F texture       |
//! +-------------+     | - Row per bone (pos + rot)    |
//! | Frame 0..N  |     | - Column per frame            |
//! +-------------+     +--------------------------------+
//!                                      |
//!                     +----------------+----------------+
//!                     v                                 v
//!              AnimationTextureData           AnimationTextureAtlas
//!              (single clip)                  (multiple clips)
//!                     |                                 |
//!                     v                                 v
//!              AnimationTextureSampler --> GPU Texture (RGBA32F)
//!              (frame interpolation)
//! ```
//!
//! # Texture Layout
//!
//! Each animation clip is stored as follows:
//! - **Width**: Number of frames (sample_rate * duration)
//! - **Height**: Bones * 2 (position row + rotation row per bone)
//! - **Format**: RGBA32F (4 x f32 per texel)
//!
//! Position row: `[x, y, z, scale]` where scale is uniform scale factor
//! Rotation row: `[qx, qy, qz, qw]` quaternion
//!
//! # GPU Sampling
//!
//! In the vertex shader:
//! ```wgsl
//! fn sample_bone_transform(bone: u32, frame: f32) -> mat4x4<f32> {
//!     let frame0 = u32(frame);
//!     let frame1 = frame0 + 1;
//!     let t = fract(frame);
//!
//!     let pos_row = bone * 2u;
//!     let rot_row = bone * 2u + 1u;
//!
//!     let pos0 = textureLoad(anim_tex, vec2(frame0, pos_row), 0);
//!     let pos1 = textureLoad(anim_tex, vec2(frame1, pos_row), 0);
//!     let position = mix(pos0, pos1, t);
//!
//!     let rot0 = textureLoad(anim_tex, vec2(frame0, rot_row), 0);
//!     let rot1 = textureLoad(anim_tex, vec2(frame1, rot_row), 0);
//!     let rotation = normalize(mix(rot0, rot1, t)); // Nlerp for quaternions
//!
//!     return compose_transform(position.xyz, rotation, position.w);
//! }
//! ```
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::animation_textures::{AnimationTextureBaker, AnimationTextureSampler};
//!
//! // Bake a walk animation
//! let baker = AnimationTextureBaker::new(64, 30.0);  // 64 bones, 30 FPS
//! let texture_data = baker.bake_clip(&walk_clip);
//!
//! // Sample at runtime
//! let sampler = AnimationTextureSampler::new(texture_data.layout.clone());
//! let (pos, rot, scale) = sampler.sample_bone(&texture_data, 5, 15.5);
//!
//! // Get GPU-ready texture data
//! let (bytes, format) = prepare_gpu_texture(&texture_data);
//! ```

use std::fmt;

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

use crate::animation_clip::AnimationClip;
use crate::skeleton::Transform;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default sample rate for texture baking (Hz).
pub const DEFAULT_TEXTURE_SAMPLE_RATE: f32 = 30.0;

/// Maximum frames per animation texture (width limit).
pub const MAX_FRAMES_PER_TEXTURE: u32 = 4096;

/// Maximum bones per animation texture (height / 2).
pub const MAX_BONES_PER_TEXTURE: u32 = 256;

/// Bytes per texel in RGBA32F format.
pub const RGBA32F_TEXEL_SIZE: usize = 16;

// ---------------------------------------------------------------------------
// TextureFormat
// ---------------------------------------------------------------------------

/// GPU texture format for animation data.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AnimationTextureFormat {
    /// Full precision RGBA32F (4 x f32).
    /// Best quality, highest memory usage.
    #[default]
    Rgba32Float,

    /// Half precision RGBA16F (4 x f16).
    /// Good quality, half the memory of RGBA32F.
    Rgba16Float,

    /// Normalized RGBA8 for rotations (quaternions only).
    /// Lowest quality, lowest memory usage.
    Rgba8Unorm,
}

impl AnimationTextureFormat {
    /// Get bytes per texel for this format.
    pub fn bytes_per_texel(&self) -> usize {
        match self {
            Self::Rgba32Float => 16,
            Self::Rgba16Float => 8,
            Self::Rgba8Unorm => 4,
        }
    }

    /// Get wgpu texture format string (for shader generation).
    pub fn wgpu_format(&self) -> &'static str {
        match self {
            Self::Rgba32Float => "rgba32float",
            Self::Rgba16Float => "rgba16float",
            Self::Rgba8Unorm => "rgba8unorm",
        }
    }
}

impl fmt::Display for AnimationTextureFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rgba32Float => write!(f, "RGBA32F"),
            Self::Rgba16Float => write!(f, "RGBA16F"),
            Self::Rgba8Unorm => write!(f, "RGBA8"),
        }
    }
}

// ---------------------------------------------------------------------------
// AnimationTextureLayout
// ---------------------------------------------------------------------------

/// Layout information for an animation texture.
///
/// Describes how bone transforms are packed into the texture.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AnimationTextureLayout {
    /// Number of bones in the skeleton.
    pub bone_count: u32,

    /// Number of frames in the animation.
    pub frame_count: u32,

    /// Texture width (frames).
    pub texture_width: u32,

    /// Texture height (bones * 2 for position and rotation rows).
    pub texture_height: u32,

    /// Sample rate used for baking (frames per second).
    pub sample_rate: f32,

    /// Duration of the animation in seconds.
    pub duration: f32,
}

impl AnimationTextureLayout {
    /// Create a new layout from bone count and frame count.
    pub fn new(bone_count: u32, frame_count: u32, sample_rate: f32) -> Self {
        let texture_height = bone_count * 2; // Position row + rotation row per bone

        Self {
            bone_count,
            frame_count,
            texture_width: frame_count,
            texture_height,
            sample_rate,
            duration: if sample_rate > 0.0 {
                (frame_count.saturating_sub(1)) as f32 / sample_rate
            } else {
                0.0
            },
        }
    }

    /// Calculate the row index for a bone's position data.
    #[inline]
    pub fn position_row(&self, bone_index: u32) -> u32 {
        bone_index * 2
    }

    /// Calculate the row index for a bone's rotation data.
    #[inline]
    pub fn rotation_row(&self, bone_index: u32) -> u32 {
        bone_index * 2 + 1
    }

    /// Calculate the texel index for position data.
    #[inline]
    pub fn position_texel_index(&self, bone_index: u32, frame: u32) -> usize {
        (self.position_row(bone_index) * self.texture_width + frame) as usize
    }

    /// Calculate the texel index for rotation data.
    #[inline]
    pub fn rotation_texel_index(&self, bone_index: u32, frame: u32) -> usize {
        (self.rotation_row(bone_index) * self.texture_width + frame) as usize
    }

    /// Get the total number of texels.
    pub fn texel_count(&self) -> usize {
        (self.texture_width * self.texture_height) as usize
    }

    /// Calculate texture memory size in bytes for the given format.
    pub fn memory_size(&self, format: AnimationTextureFormat) -> usize {
        self.texel_count() * format.bytes_per_texel()
    }

    /// Convert a time value to a frame index (with fractional part for interpolation).
    #[inline]
    pub fn time_to_frame(&self, time: f32) -> f32 {
        if self.frame_count == 0 || self.sample_rate <= 0.0 {
            return 0.0;
        }

        let frame = time * self.sample_rate;
        frame.clamp(0.0, (self.frame_count - 1) as f32)
    }

    /// Convert a frame index to time in seconds.
    #[inline]
    pub fn frame_to_time(&self, frame: f32) -> f32 {
        if self.sample_rate <= 0.0 {
            return 0.0;
        }
        frame / self.sample_rate
    }

    /// Check if the layout is valid.
    pub fn is_valid(&self) -> bool {
        self.bone_count > 0
            && self.frame_count > 0
            && self.bone_count <= MAX_BONES_PER_TEXTURE
            && self.frame_count <= MAX_FRAMES_PER_TEXTURE
            && self.sample_rate > 0.0
    }
}

// ---------------------------------------------------------------------------
// AnimationTextureData
// ---------------------------------------------------------------------------

/// Baked animation data ready for GPU upload.
///
/// Contains position and rotation data packed into texture-friendly arrays.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnimationTextureData {
    /// Layout information.
    pub layout: AnimationTextureLayout,

    /// Position data: `[x, y, z, scale]` per texel.
    /// Indexed as: `position_data[bone * 2 * width + frame]`
    pub position_data: Vec<[f32; 4]>,

    /// Rotation data: `[qx, qy, qz, qw]` per texel.
    /// Indexed as: `rotation_data[bone * 2 * width + frame]`
    pub rotation_data: Vec<[f32; 4]>,

    /// Name of the source animation clip.
    pub clip_name: String,
}

impl AnimationTextureData {
    /// Create empty texture data with the given layout.
    pub fn new(layout: AnimationTextureLayout, clip_name: impl Into<String>) -> Self {
        let texel_count = layout.texel_count();

        Self {
            layout,
            position_data: vec![[0.0, 0.0, 0.0, 1.0]; texel_count],
            rotation_data: vec![[0.0, 0.0, 0.0, 1.0]; texel_count],
            clip_name: clip_name.into(),
        }
    }

    /// Set position data for a bone at a frame.
    #[inline]
    pub fn set_position(&mut self, bone_index: u32, frame: u32, position: Vec3, scale: f32) {
        let idx = self.layout.position_texel_index(bone_index, frame);
        if idx < self.position_data.len() {
            self.position_data[idx] = [position.x, position.y, position.z, scale];
        }
    }

    /// Set rotation data for a bone at a frame.
    #[inline]
    pub fn set_rotation(&mut self, bone_index: u32, frame: u32, rotation: Quat) {
        let idx = self.layout.rotation_texel_index(bone_index, frame);
        if idx < self.rotation_data.len() {
            self.rotation_data[idx] = [rotation.x, rotation.y, rotation.z, rotation.w];
        }
    }

    /// Get position and scale for a bone at a frame.
    #[inline]
    pub fn get_position(&self, bone_index: u32, frame: u32) -> (Vec3, f32) {
        let idx = self.layout.position_texel_index(bone_index, frame);
        if idx < self.position_data.len() {
            let data = self.position_data[idx];
            (Vec3::new(data[0], data[1], data[2]), data[3])
        } else {
            (Vec3::ZERO, 1.0)
        }
    }

    /// Get rotation for a bone at a frame.
    #[inline]
    pub fn get_rotation(&self, bone_index: u32, frame: u32) -> Quat {
        let idx = self.layout.rotation_texel_index(bone_index, frame);
        if idx < self.rotation_data.len() {
            let data = self.rotation_data[idx];
            Quat::from_xyzw(data[0], data[1], data[2], data[3])
        } else {
            Quat::IDENTITY
        }
    }

    /// Get memory size in bytes (for RGBA32F format).
    pub fn memory_size(&self) -> usize {
        (self.position_data.len() + self.rotation_data.len()) * 16
    }

    /// Check if the data is valid.
    pub fn is_valid(&self) -> bool {
        self.layout.is_valid()
            && self.position_data.len() == self.layout.texel_count()
            && self.rotation_data.len() == self.layout.texel_count()
    }
}

impl Default for AnimationTextureData {
    fn default() -> Self {
        Self {
            layout: AnimationTextureLayout::default(),
            position_data: Vec::new(),
            rotation_data: Vec::new(),
            clip_name: String::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// ClipRegion
// ---------------------------------------------------------------------------

/// Region within an animation texture atlas for a single clip.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ClipRegion {
    /// Name of the animation clip.
    pub name: String,

    /// Starting frame in the atlas.
    pub start_frame: u32,

    /// Number of frames in this clip.
    pub frame_count: u32,

    /// Duration of this clip in seconds.
    pub duration: f32,

    /// Sample rate of this clip.
    pub sample_rate: f32,
}

impl ClipRegion {
    /// Create a new clip region.
    pub fn new(name: impl Into<String>, start_frame: u32, frame_count: u32, sample_rate: f32) -> Self {
        Self {
            name: name.into(),
            start_frame,
            frame_count,
            duration: if sample_rate > 0.0 {
                frame_count.saturating_sub(1) as f32 / sample_rate
            } else {
                0.0
            },
            sample_rate,
        }
    }

    /// Convert a local time to global frame index.
    #[inline]
    pub fn time_to_global_frame(&self, local_time: f32) -> f32 {
        let local_frame = local_time * self.sample_rate;
        let clamped = local_frame.clamp(0.0, (self.frame_count.saturating_sub(1)) as f32);
        self.start_frame as f32 + clamped
    }

    /// Check if a global frame is within this region.
    #[inline]
    pub fn contains_frame(&self, global_frame: u32) -> bool {
        global_frame >= self.start_frame && global_frame < self.start_frame + self.frame_count
    }
}

// ---------------------------------------------------------------------------
// AnimationTextureAtlas
// ---------------------------------------------------------------------------

/// Atlas containing multiple animation clips in a single texture.
///
/// Multiple clips are packed horizontally (along the frame axis) to reduce
/// texture switching overhead during rendering.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnimationTextureAtlas {
    /// The combined texture data.
    pub textures: AnimationTextureData,

    /// Regions for each clip in the atlas.
    pub clip_regions: Vec<ClipRegion>,

    /// Total number of clips in the atlas.
    pub clip_count: u32,
}

impl AnimationTextureAtlas {
    /// Create an empty atlas.
    pub fn new() -> Self {
        Self {
            textures: AnimationTextureData::default(),
            clip_regions: Vec::new(),
            clip_count: 0,
        }
    }

    /// Get the region for a clip by name.
    pub fn get_region(&self, name: &str) -> Option<&ClipRegion> {
        self.clip_regions.iter().find(|r| r.name == name)
    }

    /// Get the region for a clip by index.
    pub fn get_region_by_index(&self, index: usize) -> Option<&ClipRegion> {
        self.clip_regions.get(index)
    }

    /// Get total memory size in bytes.
    pub fn memory_size(&self) -> usize {
        self.textures.memory_size()
    }

    /// Check if the atlas is empty.
    pub fn is_empty(&self) -> bool {
        self.clip_regions.is_empty()
    }
}

impl Default for AnimationTextureAtlas {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// AnimationTextureBaker
// ---------------------------------------------------------------------------

/// Bakes animation clips into GPU-friendly texture data.
#[derive(Clone, Debug)]
pub struct AnimationTextureBaker {
    /// Number of bones in the target skeleton.
    pub bone_count: u32,

    /// Sample rate for texture baking (frames per second).
    pub sample_rate: f32,
}

impl AnimationTextureBaker {
    /// Create a new baker with the given bone count and sample rate.
    pub fn new(bone_count: u32, sample_rate: f32) -> Self {
        Self {
            bone_count,
            sample_rate: sample_rate.max(1.0),
        }
    }

    /// Create a baker with default sample rate.
    pub fn with_default_sample_rate(bone_count: u32) -> Self {
        Self::new(bone_count, DEFAULT_TEXTURE_SAMPLE_RATE)
    }

    /// Bake a single animation clip into texture data.
    pub fn bake_clip(&self, clip: &AnimationClip) -> AnimationTextureData {
        // Calculate number of frames needed
        let frame_count = self.calculate_frame_count(clip.duration);

        // Create layout
        let layout = AnimationTextureLayout::new(self.bone_count, frame_count, self.sample_rate);

        // Create texture data
        let mut data = AnimationTextureData::new(layout, &clip.name);

        // Bake each frame
        for frame in 0..frame_count {
            let time = frame as f32 / self.sample_rate;
            self.bake_frame(clip, frame, time, &mut data);
        }

        data
    }

    /// Bake multiple clips into a single texture atlas.
    pub fn bake_clips(&self, clips: &[&AnimationClip]) -> AnimationTextureAtlas {
        if clips.is_empty() {
            return AnimationTextureAtlas::new();
        }

        // Calculate total frames needed
        let mut total_frames: u32 = 0;
        let mut clip_frame_counts = Vec::with_capacity(clips.len());

        for clip in clips {
            let frame_count = self.calculate_frame_count(clip.duration);
            clip_frame_counts.push(frame_count);
            total_frames += frame_count;
        }

        // Check limits
        if total_frames > MAX_FRAMES_PER_TEXTURE {
            // In a real implementation, we might split into multiple textures
            // For now, clamp to max
            total_frames = MAX_FRAMES_PER_TEXTURE;
        }

        // Create layout for combined texture
        let layout = AnimationTextureLayout::new(self.bone_count, total_frames, self.sample_rate);
        let mut data = AnimationTextureData::new(layout, "atlas");

        // Create clip regions and bake each clip
        let mut clip_regions = Vec::with_capacity(clips.len());
        let mut current_frame: u32 = 0;

        for (i, clip) in clips.iter().enumerate() {
            let frame_count = clip_frame_counts[i];

            // Skip if we've exceeded the limit
            if current_frame >= total_frames {
                break;
            }

            let available_frames = (total_frames - current_frame).min(frame_count);

            // Create region
            let region = ClipRegion::new(
                &clip.name,
                current_frame,
                available_frames,
                self.sample_rate,
            );
            clip_regions.push(region);

            // Bake frames for this clip
            for frame_offset in 0..available_frames {
                let time = frame_offset as f32 / self.sample_rate;
                let global_frame = current_frame + frame_offset;
                self.bake_frame(clip, global_frame, time, &mut data);
            }

            current_frame += available_frames;
        }

        AnimationTextureAtlas {
            textures: data,
            clip_regions,
            clip_count: clips.len() as u32,
        }
    }

    /// Calculate the number of frames needed for a given duration.
    fn calculate_frame_count(&self, duration: f32) -> u32 {
        if duration <= 0.0 {
            return 1; // At least one frame for zero-duration clips
        }

        let frames = (duration * self.sample_rate).ceil() as u32 + 1;
        frames.min(MAX_FRAMES_PER_TEXTURE)
    }

    /// Bake a single frame from a clip into the texture data.
    fn bake_frame(&self, clip: &AnimationClip, frame: u32, time: f32, data: &mut AnimationTextureData) {
        // Sample each bone track
        for (bone_index, bone_track) in clip.bone_tracks.iter().enumerate() {
            if bone_index as u32 >= self.bone_count {
                break;
            }

            let bone_idx = bone_index as u32;

            // Sample position
            let position = if let Some(pos_track) = &bone_track.position {
                pos_track.sample(time).unwrap_or(Vec3::ZERO)
            } else {
                Vec3::ZERO
            };

            // Sample rotation
            let rotation = if let Some(rot_track) = &bone_track.rotation {
                rot_track.sample(time).unwrap_or(Quat::IDENTITY)
            } else {
                Quat::IDENTITY
            };

            // Sample scale (use average for uniform scale)
            let scale = if let Some(scale_track) = &bone_track.scale {
                let scale_vec = scale_track.sample(time).unwrap_or(Vec3::ONE);
                (scale_vec.x + scale_vec.y + scale_vec.z) / 3.0
            } else {
                1.0
            };

            // Store in texture
            data.set_position(bone_idx, frame, position, scale);
            data.set_rotation(bone_idx, frame, rotation);
        }

        // Fill remaining bones with identity
        for bone_idx in clip.bone_tracks.len() as u32..self.bone_count {
            data.set_position(bone_idx, frame, Vec3::ZERO, 1.0);
            data.set_rotation(bone_idx, frame, Quat::IDENTITY);
        }
    }
}

impl Default for AnimationTextureBaker {
    fn default() -> Self {
        Self::new(64, DEFAULT_TEXTURE_SAMPLE_RATE)
    }
}

// ---------------------------------------------------------------------------
// AnimationTextureSampler
// ---------------------------------------------------------------------------

/// Samples bone transforms from animation texture data with interpolation.
#[derive(Clone, Debug)]
pub struct AnimationTextureSampler {
    /// Layout of the texture being sampled.
    pub layout: AnimationTextureLayout,
}

impl AnimationTextureSampler {
    /// Create a new sampler with the given layout.
    pub fn new(layout: AnimationTextureLayout) -> Self {
        Self { layout }
    }

    /// Sample a single bone at a fractional frame index.
    ///
    /// Returns (position, rotation, scale) tuple with linear/nlerp interpolation.
    pub fn sample_bone(&self, data: &AnimationTextureData, bone: u32, frame: f32) -> (Vec3, Quat, f32) {
        if bone >= self.layout.bone_count || self.layout.frame_count == 0 {
            return (Vec3::ZERO, Quat::IDENTITY, 1.0);
        }

        let frame_clamped = frame.clamp(0.0, (self.layout.frame_count - 1) as f32);
        let frame0 = frame_clamped.floor() as u32;
        let frame1 = (frame0 + 1).min(self.layout.frame_count - 1);
        let t = frame_clamped.fract();

        // Get position and scale at both frames
        let (pos0, scale0) = data.get_position(bone, frame0);
        let (pos1, scale1) = data.get_position(bone, frame1);

        // Get rotation at both frames
        let rot0 = data.get_rotation(bone, frame0);
        let rot1 = data.get_rotation(bone, frame1);

        // Interpolate position (linear)
        let position = pos0.lerp(pos1, t);

        // Interpolate rotation (nlerp for GPU-friendly behavior)
        let rotation = nlerp(rot0, rot1, t);

        // Interpolate scale (linear)
        let scale = scale0 + (scale1 - scale0) * t;

        (position, rotation, scale)
    }

    /// Sample a single bone at a time value in seconds.
    pub fn sample_bone_at_time(&self, data: &AnimationTextureData, bone: u32, time: f32) -> (Vec3, Quat, f32) {
        let frame = self.layout.time_to_frame(time);
        self.sample_bone(data, bone, frame)
    }

    /// Sample all bones at a fractional frame index.
    ///
    /// Returns a vector of (position, rotation, scale) tuples.
    pub fn sample_pose(&self, data: &AnimationTextureData, frame: f32) -> Vec<(Vec3, Quat, f32)> {
        let mut pose = Vec::with_capacity(self.layout.bone_count as usize);

        for bone in 0..self.layout.bone_count {
            pose.push(self.sample_bone(data, bone, frame));
        }

        pose
    }

    /// Sample all bones at a time value in seconds.
    pub fn sample_pose_at_time(&self, data: &AnimationTextureData, time: f32) -> Vec<(Vec3, Quat, f32)> {
        let frame = self.layout.time_to_frame(time);
        self.sample_pose(data, frame)
    }

    /// Blend between two animation samples.
    ///
    /// Samples both animations and blends the results using linear interpolation.
    pub fn blend_samples(
        &self,
        a: &AnimationTextureData,
        b: &AnimationTextureData,
        frame_a: f32,
        frame_b: f32,
        weight: f32,
    ) -> Vec<(Vec3, Quat, f32)> {
        let pose_a = self.sample_pose(a, frame_a);
        let pose_b = self.sample_pose(b, frame_b);

        let weight_clamped = weight.clamp(0.0, 1.0);

        pose_a
            .iter()
            .zip(pose_b.iter())
            .map(|((pos_a, rot_a, scale_a), (pos_b, rot_b, scale_b))| {
                let pos = pos_a.lerp(*pos_b, weight_clamped);
                let rot = nlerp(*rot_a, *rot_b, weight_clamped);
                let scale = scale_a + (scale_b - scale_a) * weight_clamped;
                (pos, rot, scale)
            })
            .collect()
    }

    /// Blend between two animations at time values (in seconds).
    pub fn blend_samples_at_time(
        &self,
        a: &AnimationTextureData,
        b: &AnimationTextureData,
        time_a: f32,
        time_b: f32,
        weight: f32,
    ) -> Vec<(Vec3, Quat, f32)> {
        let frame_a = self.layout.time_to_frame(time_a);
        let frame_b = self.layout.time_to_frame(time_b);
        self.blend_samples(a, b, frame_a, frame_b, weight)
    }

    /// Convert a sampled pose to Transform array.
    pub fn pose_to_transforms(pose: &[(Vec3, Quat, f32)]) -> Vec<Transform> {
        pose.iter()
            .map(|(pos, rot, scale)| Transform::new(*pos, *rot, Vec3::splat(*scale)))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// GPU Texture Preparation
// ---------------------------------------------------------------------------

/// Prepare GPU-ready texture data from animation texture data.
///
/// Returns raw bytes suitable for texture upload and the recommended texture format.
/// The data is interleaved: position row 0, rotation row 0, position row 1, etc.
pub fn prepare_gpu_texture(data: &AnimationTextureData) -> (Vec<u8>, AnimationTextureFormat) {
    let layout = &data.layout;
    let total_texels = layout.texel_count();
    let bytes_per_texel = RGBA32F_TEXEL_SIZE;

    // Allocate combined buffer
    let mut bytes = Vec::with_capacity(total_texels * bytes_per_texel * 2);

    // Interleave position and rotation rows
    for bone in 0..layout.bone_count {
        // Position row for this bone
        for frame in 0..layout.frame_count {
            let idx = layout.position_texel_index(bone, frame);
            if idx < data.position_data.len() {
                let texel = &data.position_data[idx];
                bytes.extend_from_slice(bytemuck::cast_slice(&[*texel]));
            } else {
                bytes.extend_from_slice(&[0u8; RGBA32F_TEXEL_SIZE]);
            }
        }

        // Rotation row for this bone
        for frame in 0..layout.frame_count {
            let idx = layout.rotation_texel_index(bone, frame);
            if idx < data.rotation_data.len() {
                let texel = &data.rotation_data[idx];
                bytes.extend_from_slice(bytemuck::cast_slice(&[*texel]));
            } else {
                bytes.extend_from_slice(&[0u8; RGBA32F_TEXEL_SIZE]);
            }
        }
    }

    (bytes, AnimationTextureFormat::Rgba32Float)
}

/// Prepare GPU texture data in half-precision format for reduced memory.
pub fn prepare_gpu_texture_f16(data: &AnimationTextureData) -> (Vec<u8>, AnimationTextureFormat) {
    let layout = &data.layout;

    // Allocate combined buffer (8 bytes per texel for f16)
    let bytes_per_texel = 8;
    let mut bytes = Vec::with_capacity(layout.texel_count() * bytes_per_texel * 2);

    // Interleave position and rotation rows
    for bone in 0..layout.bone_count {
        // Position row for this bone
        for frame in 0..layout.frame_count {
            let idx = layout.position_texel_index(bone, frame);
            if idx < data.position_data.len() {
                let texel = &data.position_data[idx];
                // Convert f32 to f16
                for &v in texel {
                    let half = half::f16::from_f32(v);
                    bytes.extend_from_slice(&half.to_le_bytes());
                }
            } else {
                bytes.extend_from_slice(&[0u8; 8]);
            }
        }

        // Rotation row for this bone
        for frame in 0..layout.frame_count {
            let idx = layout.rotation_texel_index(bone, frame);
            if idx < data.rotation_data.len() {
                let texel = &data.rotation_data[idx];
                for &v in texel {
                    let half = half::f16::from_f32(v);
                    bytes.extend_from_slice(&half.to_le_bytes());
                }
            } else {
                bytes.extend_from_slice(&[0u8; 8]);
            }
        }
    }

    (bytes, AnimationTextureFormat::Rgba16Float)
}

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

/// Normalized linear interpolation for quaternions (GPU-friendly).
///
/// Unlike slerp, nlerp is cheaper and works well for small angular differences.
#[inline]
fn nlerp(a: Quat, b: Quat, t: f32) -> Quat {
    // Handle antipodal quaternions (shortest path)
    let b_adjusted = if a.dot(b) < 0.0 { -b } else { b };

    // Linear interpolation
    let result = Quat::from_xyzw(
        a.x + (b_adjusted.x - a.x) * t,
        a.y + (b_adjusted.y - a.y) * t,
        a.z + (b_adjusted.z - a.z) * t,
        a.w + (b_adjusted.w - a.w) * t,
    );

    // Normalize
    result.normalize()
}

/// Generate mipmap data for LOD support.
///
/// Returns a vector of mipmap levels, each with reduced frame count.
pub fn generate_mipmaps(data: &AnimationTextureData, levels: u32) -> Vec<AnimationTextureData> {
    let mut mipmaps = Vec::with_capacity(levels as usize);
    let mut current = data.clone();

    for _ in 0..levels {
        let new_frame_count = (current.layout.frame_count / 2).max(1);

        if new_frame_count == current.layout.frame_count {
            break; // Can't reduce further
        }

        let new_layout = AnimationTextureLayout::new(
            current.layout.bone_count,
            new_frame_count,
            current.layout.sample_rate / 2.0,
        );

        let mut mip = AnimationTextureData::new(new_layout, &format!("{}_mip{}", data.clip_name, mipmaps.len() + 1));

        // Downsample by averaging adjacent frames
        for bone in 0..current.layout.bone_count {
            for frame in 0..new_frame_count {
                let src_frame0 = frame * 2;
                let src_frame1 = (src_frame0 + 1).min(current.layout.frame_count - 1);

                let (pos0, scale0) = current.get_position(bone, src_frame0);
                let (pos1, scale1) = current.get_position(bone, src_frame1);
                let avg_pos = (pos0 + pos1) * 0.5;
                let avg_scale = (scale0 + scale1) * 0.5;

                let rot0 = current.get_rotation(bone, src_frame0);
                let rot1 = current.get_rotation(bone, src_frame1);
                let avg_rot = nlerp(rot0, rot1, 0.5);

                mip.set_position(bone, frame, avg_pos, avg_scale);
                mip.set_rotation(bone, frame, avg_rot);
            }
        }

        current = mip.clone();
        mipmaps.push(mip);
    }

    mipmaps
}

// ---------------------------------------------------------------------------
// Error Types
// ---------------------------------------------------------------------------

/// Errors that can occur during animation texture operations.
#[derive(Clone, Debug, PartialEq)]
pub enum AnimationTextureError {
    /// Clip has no bone tracks.
    EmptyClip,

    /// Bone count exceeds maximum.
    TooManyBones { count: u32, max: u32 },

    /// Frame count exceeds maximum.
    TooManyFrames { count: u32, max: u32 },

    /// Invalid sample rate.
    InvalidSampleRate { rate: f32 },

    /// Layout mismatch between data and sampler.
    LayoutMismatch { expected: AnimationTextureLayout, found: AnimationTextureLayout },
}

impl fmt::Display for AnimationTextureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyClip => write!(f, "empty animation clip has no bone tracks"),
            Self::TooManyBones { count, max } => {
                write!(f, "bone count {} exceeds maximum {}", count, max)
            }
            Self::TooManyFrames { count, max } => {
                write!(f, "frame count {} exceeds maximum {}", count, max)
            }
            Self::InvalidSampleRate { rate } => {
                write!(f, "invalid sample rate: {}", rate)
            }
            Self::LayoutMismatch { expected, found } => {
                write!(f, "layout mismatch: expected {:?}, found {:?}", expected, found)
            }
        }
    }
}

impl std::error::Error for AnimationTextureError {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::animation_clip::{AnimationClip, BoneTrack, Track, Keyframe};
    use std::f32::consts::PI;

    // ===== Helper Functions =====

    fn create_simple_clip(name: &str, duration: f32, bone_count: usize) -> AnimationClip {
        let mut clip = AnimationClip::new(name, duration);
        clip.frame_rate = 30.0;

        for i in 0..bone_count {
            let pos_track = Track::from_keyframes(vec![
                Keyframe::linear(0.0, Vec3::new(i as f32, 0.0, 0.0)),
                Keyframe::linear(duration, Vec3::new(i as f32, duration, 0.0)),
            ]);

            let rot_track = Track::from_keyframes(vec![
                Keyframe::linear(0.0, Quat::IDENTITY),
                Keyframe::linear(duration, Quat::from_rotation_y(PI * (i as f32 + 1.0) / 4.0)),
            ]);

            let scale_track = Track::from_keyframes(vec![
                Keyframe::linear(0.0, Vec3::ONE),
                Keyframe::linear(duration, Vec3::splat(1.0 + 0.1 * i as f32)),
            ]);

            let bone_track = BoneTrack::new(format!("bone_{}", i))
                .with_position(pos_track)
                .with_rotation(rot_track)
                .with_scale(scale_track);

            clip.add_bone_track(bone_track);
        }

        clip
    }

    fn create_constant_clip(name: &str, position: Vec3, rotation: Quat) -> AnimationClip {
        let mut clip = AnimationClip::new(name, 0.0);
        clip.frame_rate = 30.0;

        let pos_track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, position),
        ]);

        let rot_track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, rotation),
        ]);

        let bone_track = BoneTrack::new("bone_0")
            .with_position(pos_track)
            .with_rotation(rot_track);

        clip.add_bone_track(bone_track);
        clip
    }

    // ===== AnimationTextureLayout Tests =====

    #[test]
    fn test_layout_new() {
        let layout = AnimationTextureLayout::new(64, 100, 30.0);

        assert_eq!(layout.bone_count, 64);
        assert_eq!(layout.frame_count, 100);
        assert_eq!(layout.texture_width, 100);
        assert_eq!(layout.texture_height, 128); // 64 * 2
        assert_eq!(layout.sample_rate, 30.0);
        assert!((layout.duration - 3.3).abs() < 0.1); // (100-1)/30
    }

    #[test]
    fn test_layout_row_indices() {
        let layout = AnimationTextureLayout::new(10, 50, 30.0);

        assert_eq!(layout.position_row(0), 0);
        assert_eq!(layout.rotation_row(0), 1);
        assert_eq!(layout.position_row(5), 10);
        assert_eq!(layout.rotation_row(5), 11);
        assert_eq!(layout.position_row(9), 18);
        assert_eq!(layout.rotation_row(9), 19);
    }

    #[test]
    fn test_layout_texel_indices() {
        let layout = AnimationTextureLayout::new(4, 10, 30.0);

        // First bone, first frame
        assert_eq!(layout.position_texel_index(0, 0), 0);
        assert_eq!(layout.rotation_texel_index(0, 0), 10);

        // First bone, last frame
        assert_eq!(layout.position_texel_index(0, 9), 9);
        assert_eq!(layout.rotation_texel_index(0, 9), 19);

        // Last bone, first frame
        assert_eq!(layout.position_texel_index(3, 0), 60);
        assert_eq!(layout.rotation_texel_index(3, 0), 70);
    }

    #[test]
    fn test_layout_texel_count() {
        let layout = AnimationTextureLayout::new(64, 100, 30.0);

        assert_eq!(layout.texel_count(), 100 * 128); // width * height
    }

    #[test]
    fn test_layout_memory_size() {
        let layout = AnimationTextureLayout::new(64, 100, 30.0);

        assert_eq!(
            layout.memory_size(AnimationTextureFormat::Rgba32Float),
            100 * 128 * 16
        );
        assert_eq!(
            layout.memory_size(AnimationTextureFormat::Rgba16Float),
            100 * 128 * 8
        );
    }

    #[test]
    fn test_layout_time_to_frame() {
        let layout = AnimationTextureLayout::new(10, 31, 30.0);

        assert_eq!(layout.time_to_frame(0.0), 0.0);
        assert!((layout.time_to_frame(0.5) - 15.0).abs() < 0.01);
        assert_eq!(layout.time_to_frame(1.0), 30.0);
        assert_eq!(layout.time_to_frame(2.0), 30.0); // Clamped
        assert_eq!(layout.time_to_frame(-1.0), 0.0); // Clamped
    }

    #[test]
    fn test_layout_frame_to_time() {
        let layout = AnimationTextureLayout::new(10, 31, 30.0);

        assert_eq!(layout.frame_to_time(0.0), 0.0);
        assert!((layout.frame_to_time(15.0) - 0.5).abs() < 0.01);
        assert!((layout.frame_to_time(30.0) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_layout_is_valid() {
        assert!(AnimationTextureLayout::new(64, 100, 30.0).is_valid());
        assert!(!AnimationTextureLayout::new(0, 100, 30.0).is_valid());
        assert!(!AnimationTextureLayout::new(64, 0, 30.0).is_valid());
        assert!(!AnimationTextureLayout::new(64, 100, 0.0).is_valid());
        assert!(!AnimationTextureLayout::new(300, 100, 30.0).is_valid()); // > MAX_BONES
    }

    // ===== AnimationTextureFormat Tests =====

    #[test]
    fn test_format_bytes_per_texel() {
        assert_eq!(AnimationTextureFormat::Rgba32Float.bytes_per_texel(), 16);
        assert_eq!(AnimationTextureFormat::Rgba16Float.bytes_per_texel(), 8);
        assert_eq!(AnimationTextureFormat::Rgba8Unorm.bytes_per_texel(), 4);
    }

    #[test]
    fn test_format_wgpu_format() {
        assert_eq!(AnimationTextureFormat::Rgba32Float.wgpu_format(), "rgba32float");
        assert_eq!(AnimationTextureFormat::Rgba16Float.wgpu_format(), "rgba16float");
        assert_eq!(AnimationTextureFormat::Rgba8Unorm.wgpu_format(), "rgba8unorm");
    }

    #[test]
    fn test_format_display() {
        assert_eq!(format!("{}", AnimationTextureFormat::Rgba32Float), "RGBA32F");
        assert_eq!(format!("{}", AnimationTextureFormat::Rgba16Float), "RGBA16F");
        assert_eq!(format!("{}", AnimationTextureFormat::Rgba8Unorm), "RGBA8");
    }

    // ===== AnimationTextureData Tests =====

    #[test]
    fn test_texture_data_new() {
        let layout = AnimationTextureLayout::new(4, 10, 30.0);
        let data = AnimationTextureData::new(layout.clone(), "test");

        assert_eq!(data.clip_name, "test");
        assert_eq!(data.position_data.len(), layout.texel_count());
        assert_eq!(data.rotation_data.len(), layout.texel_count());

        // Check default values
        assert_eq!(data.position_data[0], [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(data.rotation_data[0], [0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_texture_data_set_get_position() {
        let layout = AnimationTextureLayout::new(4, 10, 30.0);
        let mut data = AnimationTextureData::new(layout, "test");

        let pos = Vec3::new(1.0, 2.0, 3.0);
        let scale = 1.5;

        data.set_position(2, 5, pos, scale);
        let (retrieved_pos, retrieved_scale) = data.get_position(2, 5);

        assert!((retrieved_pos - pos).length() < 0.0001);
        assert!((retrieved_scale - scale).abs() < 0.0001);
    }

    #[test]
    fn test_texture_data_set_get_rotation() {
        let layout = AnimationTextureLayout::new(4, 10, 30.0);
        let mut data = AnimationTextureData::new(layout, "test");

        let rot = Quat::from_rotation_y(PI / 4.0);

        data.set_rotation(1, 3, rot);
        let retrieved_rot = data.get_rotation(1, 3);

        assert!(rot.angle_between(retrieved_rot) < 0.0001);
    }

    #[test]
    fn test_texture_data_out_of_bounds() {
        let layout = AnimationTextureLayout::new(4, 10, 30.0);
        let data = AnimationTextureData::new(layout, "test");

        // Out of bounds should return defaults
        let (pos, scale) = data.get_position(100, 0);
        assert_eq!(pos, Vec3::ZERO);
        assert_eq!(scale, 1.0);

        let rot = data.get_rotation(0, 100);
        assert_eq!(rot, Quat::IDENTITY);
    }

    #[test]
    fn test_texture_data_memory_size() {
        let layout = AnimationTextureLayout::new(64, 100, 30.0);
        let data = AnimationTextureData::new(layout, "test");

        // Each texel is 4 floats * 4 bytes = 16 bytes
        // pos_data + rot_data = 2 * texel_count * 16
        let expected_size = 2 * 100 * 128 * 16;
        assert_eq!(data.memory_size(), expected_size);
    }

    #[test]
    fn test_texture_data_is_valid() {
        let layout = AnimationTextureLayout::new(4, 10, 30.0);
        let data = AnimationTextureData::new(layout, "test");

        assert!(data.is_valid());

        // Invalid: empty data
        let invalid = AnimationTextureData::default();
        assert!(!invalid.is_valid());
    }

    // ===== ClipRegion Tests =====

    #[test]
    fn test_clip_region_new() {
        let region = ClipRegion::new("walk", 0, 31, 30.0);

        assert_eq!(region.name, "walk");
        assert_eq!(region.start_frame, 0);
        assert_eq!(region.frame_count, 31);
        assert!((region.duration - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_clip_region_time_to_global_frame() {
        let region = ClipRegion::new("run", 100, 31, 30.0);

        assert_eq!(region.time_to_global_frame(0.0), 100.0);
        assert!((region.time_to_global_frame(0.5) - 115.0).abs() < 0.01);
        assert_eq!(region.time_to_global_frame(1.0), 130.0);
        assert_eq!(region.time_to_global_frame(2.0), 130.0); // Clamped
    }

    #[test]
    fn test_clip_region_contains_frame() {
        let region = ClipRegion::new("idle", 50, 20, 30.0);

        assert!(!region.contains_frame(49));
        assert!(region.contains_frame(50));
        assert!(region.contains_frame(60));
        assert!(region.contains_frame(69));
        assert!(!region.contains_frame(70));
    }

    // ===== AnimationTextureBaker Tests =====

    #[test]
    fn test_baker_new() {
        let baker = AnimationTextureBaker::new(64, 30.0);

        assert_eq!(baker.bone_count, 64);
        assert_eq!(baker.sample_rate, 30.0);
    }

    #[test]
    fn test_baker_sample_rate_minimum() {
        let baker = AnimationTextureBaker::new(64, 0.0);

        assert_eq!(baker.sample_rate, 1.0); // Clamped to minimum
    }

    #[test]
    fn test_baker_bake_single_clip() {
        let clip = create_simple_clip("walk", 1.0, 4);
        let baker = AnimationTextureBaker::new(4, 30.0);

        let data = baker.bake_clip(&clip);

        assert_eq!(data.clip_name, "walk");
        assert_eq!(data.layout.bone_count, 4);
        assert!(data.layout.frame_count >= 31); // At least 30*1.0 + 1 frames
        assert!(data.is_valid());
    }

    #[test]
    fn test_baker_bake_single_frame() {
        let clip = create_constant_clip("idle", Vec3::new(1.0, 2.0, 3.0), Quat::from_rotation_y(PI / 2.0));
        let baker = AnimationTextureBaker::new(1, 30.0);

        let data = baker.bake_clip(&clip);

        assert_eq!(data.layout.frame_count, 1);

        let (pos, scale) = data.get_position(0, 0);
        assert!((pos - Vec3::new(1.0, 2.0, 3.0)).length() < 0.0001);
        assert!((scale - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_baker_bake_preserves_position() {
        let clip = create_simple_clip("test", 1.0, 2);
        let baker = AnimationTextureBaker::new(2, 30.0);

        let data = baker.bake_clip(&clip);

        // Check first frame
        let (pos0, _) = data.get_position(0, 0);
        assert!((pos0.x - 0.0).abs() < 0.01);

        let (pos1, _) = data.get_position(1, 0);
        assert!((pos1.x - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_baker_bake_preserves_rotation() {
        let clip = create_simple_clip("test", 1.0, 1);
        let baker = AnimationTextureBaker::new(1, 30.0);

        let data = baker.bake_clip(&clip);

        // First frame should be identity
        let rot0 = data.get_rotation(0, 0);
        assert!(rot0.angle_between(Quat::IDENTITY) < 0.01);

        // Last frame should have rotation
        let last_frame = data.layout.frame_count - 1;
        let rot_last = data.get_rotation(0, last_frame);
        let expected = Quat::from_rotation_y(PI / 4.0);
        assert!(rot_last.angle_between(expected) < 0.01);
    }

    #[test]
    fn test_baker_fills_missing_bones() {
        let clip = create_simple_clip("test", 1.0, 2);
        let baker = AnimationTextureBaker::new(5, 30.0); // More bones than clip

        let data = baker.bake_clip(&clip);

        // Bones 2-4 should have identity transforms
        for bone in 2..5 {
            let (pos, scale) = data.get_position(bone, 0);
            assert!((pos - Vec3::ZERO).length() < 0.0001);
            assert!((scale - 1.0).abs() < 0.0001);

            let rot = data.get_rotation(bone, 0);
            assert!(rot.angle_between(Quat::IDENTITY) < 0.0001);
        }
    }

    #[test]
    fn test_baker_bake_multi_clip_atlas() {
        let walk = create_simple_clip("walk", 1.0, 4);
        let run = create_simple_clip("run", 0.5, 4);
        let clips: Vec<&AnimationClip> = vec![&walk, &run];

        let baker = AnimationTextureBaker::new(4, 30.0);
        let atlas = baker.bake_clips(&clips);

        assert_eq!(atlas.clip_count, 2);
        assert_eq!(atlas.clip_regions.len(), 2);

        // Check walk region
        assert_eq!(atlas.clip_regions[0].name, "walk");
        assert_eq!(atlas.clip_regions[0].start_frame, 0);

        // Check run region starts after walk
        assert_eq!(atlas.clip_regions[1].name, "run");
        assert!(atlas.clip_regions[1].start_frame > 0);
    }

    #[test]
    fn test_baker_empty_clips() {
        let clips: Vec<&AnimationClip> = vec![];

        let baker = AnimationTextureBaker::new(4, 30.0);
        let atlas = baker.bake_clips(&clips);

        assert!(atlas.is_empty());
        assert_eq!(atlas.clip_count, 0);
    }

    // ===== AnimationTextureSampler Tests =====

    #[test]
    fn test_sampler_sample_bone_exact_frame() {
        let clip = create_simple_clip("test", 1.0, 2);
        let baker = AnimationTextureBaker::new(2, 30.0);
        let data = baker.bake_clip(&clip);

        let sampler = AnimationTextureSampler::new(data.layout.clone());

        // Sample at frame 0
        let (pos, rot, scale) = sampler.sample_bone(&data, 0, 0.0);
        assert!((pos.x - 0.0).abs() < 0.01);
        assert!(rot.angle_between(Quat::IDENTITY) < 0.01);
        assert!((scale - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_sampler_sample_bone_interpolation() {
        let clip = create_simple_clip("test", 1.0, 1);
        let baker = AnimationTextureBaker::new(1, 30.0);
        let data = baker.bake_clip(&clip);

        let sampler = AnimationTextureSampler::new(data.layout.clone());

        // Sample at middle frame
        let mid_frame = (data.layout.frame_count - 1) as f32 / 2.0;
        let (pos, _, _) = sampler.sample_bone(&data, 0, mid_frame);

        // Y should be interpolated (starts 0, ends duration=1.0)
        assert!(pos.y > 0.0 && pos.y < 1.0);
    }

    #[test]
    fn test_sampler_sample_bone_at_time() {
        let clip = create_simple_clip("test", 1.0, 1);
        let baker = AnimationTextureBaker::new(1, 30.0);
        let data = baker.bake_clip(&clip);

        let sampler = AnimationTextureSampler::new(data.layout.clone());

        let (pos, _, _) = sampler.sample_bone_at_time(&data, 0, 0.5);

        // Y should be around 0.5
        assert!((pos.y - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_sampler_sample_pose() {
        let clip = create_simple_clip("test", 1.0, 3);
        let baker = AnimationTextureBaker::new(3, 30.0);
        let data = baker.bake_clip(&clip);

        let sampler = AnimationTextureSampler::new(data.layout.clone());
        let pose = sampler.sample_pose(&data, 0.0);

        assert_eq!(pose.len(), 3);

        // Check bone indices match
        for (i, (pos, _, _)) in pose.iter().enumerate() {
            assert!((pos.x - i as f32).abs() < 0.01);
        }
    }

    #[test]
    fn test_sampler_blend_samples() {
        let clip_a = create_constant_clip("a", Vec3::ZERO, Quat::IDENTITY);
        let clip_b = create_constant_clip("b", Vec3::new(10.0, 0.0, 0.0), Quat::IDENTITY);

        let baker = AnimationTextureBaker::new(1, 30.0);
        let data_a = baker.bake_clip(&clip_a);
        let data_b = baker.bake_clip(&clip_b);

        let sampler = AnimationTextureSampler::new(data_a.layout.clone());

        // 50% blend
        let blended = sampler.blend_samples(&data_a, &data_b, 0.0, 0.0, 0.5);

        assert_eq!(blended.len(), 1);
        assert!((blended[0].0.x - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_sampler_blend_samples_weights() {
        let clip_a = create_constant_clip("a", Vec3::ZERO, Quat::IDENTITY);
        let clip_b = create_constant_clip("b", Vec3::new(10.0, 0.0, 0.0), Quat::IDENTITY);

        let baker = AnimationTextureBaker::new(1, 30.0);
        let data_a = baker.bake_clip(&clip_a);
        let data_b = baker.bake_clip(&clip_b);

        let sampler = AnimationTextureSampler::new(data_a.layout.clone());

        // 0% blend (all A)
        let blend_0 = sampler.blend_samples(&data_a, &data_b, 0.0, 0.0, 0.0);
        assert!((blend_0[0].0.x - 0.0).abs() < 0.01);

        // 100% blend (all B)
        let blend_100 = sampler.blend_samples(&data_a, &data_b, 0.0, 0.0, 1.0);
        assert!((blend_100[0].0.x - 10.0).abs() < 0.01);

        // Weight clamping
        let blend_over = sampler.blend_samples(&data_a, &data_b, 0.0, 0.0, 1.5);
        assert!((blend_over[0].0.x - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_sampler_pose_to_transforms() {
        let pose = vec![
            (Vec3::new(1.0, 2.0, 3.0), Quat::from_rotation_y(PI / 4.0), 1.5),
            (Vec3::ZERO, Quat::IDENTITY, 1.0),
        ];

        let transforms = AnimationTextureSampler::pose_to_transforms(&pose);

        assert_eq!(transforms.len(), 2);
        assert!((transforms[0].position - Vec3::new(1.0, 2.0, 3.0)).length() < 0.0001);
        assert!((transforms[0].scale.x - 1.5).abs() < 0.0001);
        assert!(transforms[1].rotation.angle_between(Quat::IDENTITY) < 0.0001);
    }

    #[test]
    fn test_sampler_out_of_bounds_bone() {
        let layout = AnimationTextureLayout::new(4, 10, 30.0);
        let data = AnimationTextureData::new(layout.clone(), "test");
        let sampler = AnimationTextureSampler::new(layout);

        // Out of bounds bone returns defaults
        let (pos, rot, scale) = sampler.sample_bone(&data, 100, 0.0);
        assert_eq!(pos, Vec3::ZERO);
        assert_eq!(rot, Quat::IDENTITY);
        assert_eq!(scale, 1.0);
    }

    #[test]
    fn test_sampler_frame_clamping() {
        let clip = create_simple_clip("test", 1.0, 1);
        let baker = AnimationTextureBaker::new(1, 30.0);
        let data = baker.bake_clip(&clip);

        let sampler = AnimationTextureSampler::new(data.layout.clone());

        // Negative frame should clamp to 0
        let (pos_neg, _, _) = sampler.sample_bone(&data, 0, -10.0);
        let (pos_zero, _, _) = sampler.sample_bone(&data, 0, 0.0);
        assert!((pos_neg - pos_zero).length() < 0.0001);

        // Frame beyond end should clamp to last
        let last_frame = data.layout.frame_count as f32;
        let (pos_over, _, _) = sampler.sample_bone(&data, 0, last_frame + 100.0);
        let (pos_last, _, _) = sampler.sample_bone(&data, 0, last_frame - 1.0);
        assert!((pos_over - pos_last).length() < 0.01);
    }

    // ===== GPU Texture Preparation Tests =====

    #[test]
    fn test_prepare_gpu_texture() {
        let layout = AnimationTextureLayout::new(2, 3, 30.0);
        let mut data = AnimationTextureData::new(layout, "test");

        data.set_position(0, 0, Vec3::new(1.0, 2.0, 3.0), 1.0);
        data.set_rotation(0, 0, Quat::from_xyzw(0.1, 0.2, 0.3, 0.9).normalize());

        let (bytes, format) = prepare_gpu_texture(&data);

        assert_eq!(format, AnimationTextureFormat::Rgba32Float);

        // Check size: 2 bones * 2 rows * 3 frames * 16 bytes
        let expected_size = 2 * 2 * 3 * 16;
        assert_eq!(bytes.len(), expected_size);
    }

    #[test]
    fn test_prepare_gpu_texture_f16() {
        let layout = AnimationTextureLayout::new(2, 3, 30.0);
        let data = AnimationTextureData::new(layout, "test");

        let (bytes, format) = prepare_gpu_texture_f16(&data);

        assert_eq!(format, AnimationTextureFormat::Rgba16Float);

        // Check size: 2 bones * 2 rows * 3 frames * 8 bytes
        let expected_size = 2 * 2 * 3 * 8;
        assert_eq!(bytes.len(), expected_size);
    }

    #[test]
    fn test_prepare_gpu_texture_data_integrity() {
        let layout = AnimationTextureLayout::new(1, 1, 30.0);
        let mut data = AnimationTextureData::new(layout, "test");

        let pos = Vec3::new(1.5, 2.5, 3.5);
        let scale = 1.25;
        data.set_position(0, 0, pos, scale);

        let (bytes, _) = prepare_gpu_texture(&data);

        // Read back first position texel (first 16 bytes)
        let floats: &[f32] = bytemuck::cast_slice(&bytes[0..16]);

        assert!((floats[0] - 1.5).abs() < 0.0001);
        assert!((floats[1] - 2.5).abs() < 0.0001);
        assert!((floats[2] - 3.5).abs() < 0.0001);
        assert!((floats[3] - 1.25).abs() < 0.0001);
    }

    // ===== Mipmap Generation Tests =====

    #[test]
    fn test_generate_mipmaps() {
        let clip = create_simple_clip("test", 1.0, 4);
        let baker = AnimationTextureBaker::new(4, 30.0);
        let data = baker.bake_clip(&clip);

        let mipmaps = generate_mipmaps(&data, 3);

        // Should generate some mipmaps
        assert!(!mipmaps.is_empty());

        // Each mipmap should have fewer frames
        for mip in &mipmaps {
            assert!(mip.layout.frame_count < data.layout.frame_count);
        }
    }

    #[test]
    fn test_generate_mipmaps_minimum_frames() {
        let clip = create_constant_clip("test", Vec3::ZERO, Quat::IDENTITY);
        let baker = AnimationTextureBaker::new(1, 30.0);
        let data = baker.bake_clip(&clip);

        // Single frame clip shouldn't generate mipmaps
        let mipmaps = generate_mipmaps(&data, 5);
        assert!(mipmaps.is_empty());
    }

    #[test]
    fn test_generate_mipmaps_preserves_bone_count() {
        let clip = create_simple_clip("test", 1.0, 8);
        let baker = AnimationTextureBaker::new(8, 30.0);
        let data = baker.bake_clip(&clip);

        let mipmaps = generate_mipmaps(&data, 2);

        for mip in &mipmaps {
            assert_eq!(mip.layout.bone_count, 8);
        }
    }

    // ===== AnimationTextureAtlas Tests =====

    #[test]
    fn test_atlas_get_region() {
        let walk = create_simple_clip("walk", 1.0, 4);
        let run = create_simple_clip("run", 0.5, 4);
        let clips: Vec<&AnimationClip> = vec![&walk, &run];

        let baker = AnimationTextureBaker::new(4, 30.0);
        let atlas = baker.bake_clips(&clips);

        let walk_region = atlas.get_region("walk");
        assert!(walk_region.is_some());
        assert_eq!(walk_region.unwrap().name, "walk");

        let run_region = atlas.get_region("run");
        assert!(run_region.is_some());
        assert_eq!(run_region.unwrap().name, "run");

        let missing = atlas.get_region("jump");
        assert!(missing.is_none());
    }

    #[test]
    fn test_atlas_get_region_by_index() {
        let walk = create_simple_clip("walk", 1.0, 4);
        let run = create_simple_clip("run", 0.5, 4);
        let clips: Vec<&AnimationClip> = vec![&walk, &run];

        let baker = AnimationTextureBaker::new(4, 30.0);
        let atlas = baker.bake_clips(&clips);

        assert_eq!(atlas.get_region_by_index(0).unwrap().name, "walk");
        assert_eq!(atlas.get_region_by_index(1).unwrap().name, "run");
        assert!(atlas.get_region_by_index(2).is_none());
    }

    #[test]
    fn test_atlas_memory_size() {
        let walk = create_simple_clip("walk", 1.0, 4);
        let run = create_simple_clip("run", 0.5, 4);
        let clips: Vec<&AnimationClip> = vec![&walk, &run];

        let baker = AnimationTextureBaker::new(4, 30.0);
        let atlas = baker.bake_clips(&clips);

        assert!(atlas.memory_size() > 0);
        assert_eq!(atlas.memory_size(), atlas.textures.memory_size());
    }

    // ===== Nlerp Tests =====

    #[test]
    fn test_nlerp_identity() {
        let q = Quat::from_rotation_y(PI / 4.0);

        let result = nlerp(q, q, 0.5);

        assert!(result.angle_between(q) < 0.0001);
    }

    #[test]
    fn test_nlerp_endpoints() {
        let a = Quat::IDENTITY;
        let b = Quat::from_rotation_y(PI / 2.0);

        let result_0 = nlerp(a, b, 0.0);
        let result_1 = nlerp(a, b, 1.0);

        assert!(result_0.angle_between(a) < 0.001);
        assert!(result_1.angle_between(b) < 0.001);
    }

    #[test]
    fn test_nlerp_antipodal() {
        // Test that nlerp handles antipodal quaternions (q and -q are the same rotation)
        let a = Quat::IDENTITY;
        let b = -Quat::from_rotation_y(PI / 4.0);

        // Should not flip unexpectedly
        let result = nlerp(a, b, 0.5);

        // Result should be normalized
        assert!((result.length() - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_nlerp_normalized() {
        let a = Quat::from_rotation_x(0.5);
        let b = Quat::from_rotation_y(0.7);

        for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let result = nlerp(a, b, t);
            assert!((result.length() - 1.0).abs() < 0.0001);
        }
    }

    // ===== Edge Case Tests =====

    #[test]
    fn test_single_bone_clip() {
        let clip = create_simple_clip("single", 1.0, 1);
        let baker = AnimationTextureBaker::new(1, 30.0);
        let data = baker.bake_clip(&clip);

        assert_eq!(data.layout.bone_count, 1);
        assert!(data.is_valid());
    }

    #[test]
    fn test_max_bones_clip() {
        let clip = create_simple_clip("max", 0.1, MAX_BONES_PER_TEXTURE as usize);
        let baker = AnimationTextureBaker::new(MAX_BONES_PER_TEXTURE, 30.0);
        let data = baker.bake_clip(&clip);

        assert_eq!(data.layout.bone_count, MAX_BONES_PER_TEXTURE);
        assert!(data.is_valid());
    }

    #[test]
    fn test_very_short_clip() {
        let mut clip = AnimationClip::new("short", 0.01); // 10ms
        clip.frame_rate = 30.0;

        let pos_track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Vec3::ZERO),
            Keyframe::linear(0.01, Vec3::ONE),
        ]);

        let bone_track = BoneTrack::new("bone_0").with_position(pos_track);
        clip.add_bone_track(bone_track);

        let baker = AnimationTextureBaker::new(1, 30.0);
        let data = baker.bake_clip(&clip);

        assert!(data.layout.frame_count >= 1);
        assert!(data.is_valid());
    }

    #[test]
    fn test_high_sample_rate() {
        let clip = create_simple_clip("high_rate", 1.0, 2);
        let baker = AnimationTextureBaker::new(2, 120.0); // 120 FPS
        let data = baker.bake_clip(&clip);

        assert!(data.layout.frame_count >= 120);
        assert!(data.is_valid());
    }

    #[test]
    fn test_reconstruction_accuracy() {
        let original_pos = Vec3::new(1.5, 2.5, 3.5);
        let original_rot = Quat::from_euler(glam::EulerRot::XYZ, 0.1, 0.2, 0.3);
        let original_scale = 1.5;

        let mut clip = AnimationClip::new("accuracy", 0.0);
        clip.frame_rate = 30.0;

        let pos_track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, original_pos),
        ]);

        let rot_track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, original_rot),
        ]);

        let scale_track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Vec3::splat(original_scale)),
        ]);

        let bone_track = BoneTrack::new("bone_0")
            .with_position(pos_track)
            .with_rotation(rot_track)
            .with_scale(scale_track);
        clip.add_bone_track(bone_track);

        let baker = AnimationTextureBaker::new(1, 30.0);
        let data = baker.bake_clip(&clip);

        let sampler = AnimationTextureSampler::new(data.layout.clone());
        let (pos, rot, scale) = sampler.sample_bone(&data, 0, 0.0);

        // Check position accuracy
        assert!((pos - original_pos).length() < 0.0001);

        // Check rotation accuracy
        assert!(rot.angle_between(original_rot) < 0.0001);

        // Check scale accuracy
        assert!((scale - original_scale).abs() < 0.001);
    }

    // ===== Error Type Tests =====

    #[test]
    fn test_error_display() {
        let err = AnimationTextureError::TooManyBones { count: 300, max: 256 };
        assert!(format!("{}", err).contains("300"));
        assert!(format!("{}", err).contains("256"));

        let err = AnimationTextureError::EmptyClip;
        assert!(format!("{}", err).contains("empty"));
    }

    // ===== Integration Tests =====

    #[test]
    fn test_full_pipeline() {
        // Create clips
        let walk = create_simple_clip("walk", 1.0, 4);
        let run = create_simple_clip("run", 0.5, 4);

        // Bake to atlas
        let baker = AnimationTextureBaker::new(4, 30.0);
        let atlas = baker.bake_clips(&[&walk, &run]);

        // Create sampler
        let sampler = AnimationTextureSampler::new(atlas.textures.layout.clone());

        // Sample walk at time 0.5
        let walk_region = atlas.get_region("walk").unwrap();
        let walk_frame = walk_region.time_to_global_frame(0.5);
        let walk_pose = sampler.sample_pose(&atlas.textures, walk_frame);

        // Sample run at time 0.25
        let run_region = atlas.get_region("run").unwrap();
        let run_frame = run_region.time_to_global_frame(0.25);
        let run_pose = sampler.sample_pose(&atlas.textures, run_frame);

        // Verify we got poses
        assert_eq!(walk_pose.len(), 4);
        assert_eq!(run_pose.len(), 4);

        // Prepare for GPU
        let (gpu_data, format) = prepare_gpu_texture(&atlas.textures);
        assert!(!gpu_data.is_empty());
        assert_eq!(format, AnimationTextureFormat::Rgba32Float);
    }

    #[test]
    fn test_animation_blending_full() {
        let idle = create_constant_clip("idle", Vec3::new(0.0, 1.0, 0.0), Quat::IDENTITY);
        let walk = create_constant_clip("walk", Vec3::new(0.0, 1.1, 0.5), Quat::from_rotation_y(0.1));

        let baker = AnimationTextureBaker::new(1, 30.0);
        let idle_data = baker.bake_clip(&idle);
        let walk_data = baker.bake_clip(&walk);

        let sampler = AnimationTextureSampler::new(idle_data.layout.clone());

        // Test various blend weights
        for weight in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let blended = sampler.blend_samples(&idle_data, &walk_data, 0.0, 0.0, weight);

            // Position should interpolate linearly
            let expected_z = 0.0 * (1.0 - weight) + 0.5 * weight;
            assert!((blended[0].0.z - expected_z).abs() < 0.01);
        }
    }
}
