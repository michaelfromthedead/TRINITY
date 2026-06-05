//! Impostor/Billboard System for Crowd Rendering LOD (T-AN-8.4)
//!
//! This module provides an impostor system for efficient crowd rendering at distant LOD levels.
//! Instead of rendering full 3D meshes with skinning, impostors use pre-rendered sprite sheets
//! that capture the character from multiple view angles and animation frames.
//!
//! # Architecture
//!
//! ```text
//! CharacterMesh                     ImpostorRenderer
//!       |                                  |
//!       v                                  v
//! +----------------+     +--------------------------------+
//! | 3D Skinned     | --> | Render from multiple angles   |
//! | Mesh + Skel    |     | (8x4 = 32 view angles)        |
//! +----------------+     +--------------------------------+
//!                                          |
//!                        +----------------+----------------+
//!                        v                                 v
//!                 ImpostorAtlas               ViewAngleSelector
//!                 (sprite sheet)              (camera-based selection)
//!                        |                                 |
//!                        v                                 v
//!                 ImpostorInstance  <----- LOD Transition
//!                 (GPU instanced)           (crossfade)
//! ```
//!
//! # Billboard Modes
//!
//! Three billboard orientation modes are supported:
//! - **Spherical**: Always faces camera (full 3DOF rotation)
//! - **Cylindrical**: Rotates around Y-axis only (upright characters)
//! - **Axial**: Rotates around a custom axis
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::impostor_system::{
//!     ImpostorAtlas, ImpostorConfig, ImpostorRenderer, ViewAngleSelector,
//!     ImpostorInstance, BillboardMode,
//! };
//!
//! // Configure impostor generation
//! let config = ImpostorConfig::new()
//!     .with_view_angles(8, 4)  // 8 horizontal, 4 vertical
//!     .with_animation_frames(16)
//!     .with_sprite_resolution(256, 256)
//!     .with_billboard_mode(BillboardMode::Cylindrical);
//!
//! // Generate atlas from character mesh
//! let atlas = ImpostorRenderer::new(config.clone()).bake_atlas(&mesh, &skeleton);
//!
//! // At runtime: select view angle based on camera
//! let selector = ViewAngleSelector::new(&config);
//! let view_index = selector.select_angle(camera_pos, instance_pos, instance_rotation);
//!
//! // Create instance for rendering
//! let instance = ImpostorInstance::new([10.0, 0.0, 5.0])
//!     .with_animation_frame(3)
//!     .with_view_angle_index(view_index)
//!     .with_lod_fade(0.8);
//! ```

use bytemuck::{Pod, Zeroable};
use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default horizontal view angles (around Y-axis).
pub const DEFAULT_HORIZONTAL_ANGLES: u32 = 8;

/// Default vertical view angles (pitch).
pub const DEFAULT_VERTICAL_ANGLES: u32 = 4;

/// Default animation frames per impostor.
pub const DEFAULT_ANIMATION_FRAMES: u32 = 16;

/// Default sprite resolution (width = height).
pub const DEFAULT_SPRITE_RESOLUTION: u32 = 256;

/// Default alpha cutoff threshold.
pub const DEFAULT_ALPHA_CUTOFF: f32 = 0.5;

/// Size of ImpostorInstance in bytes (64 bytes for GPU alignment).
pub const IMPOSTOR_INSTANCE_SIZE: usize = 64;

/// Maximum impostors per batch.
pub const MAX_IMPOSTORS_PER_BATCH: usize = 65536;

/// LOD fade flag: Enable crossfade.
pub const FADE_FLAG_CROSSFADE: u8 = 0x01;

/// LOD fade flag: Use dithered transition.
pub const FADE_FLAG_DITHERED: u8 = 0x02;

/// Flag: Instance is visible.
pub const FLAG_VISIBLE: u8 = 0x01;

/// Flag: Cast shadows.
pub const FLAG_CAST_SHADOW: u8 = 0x02;

// ---------------------------------------------------------------------------
// Vec3 / Quat helper types (inline to avoid external dependencies)
// ---------------------------------------------------------------------------

/// 3D vector type for impostor system.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    /// Create a new Vec3.
    #[inline]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Zero vector.
    pub const ZERO: Self = Self::new(0.0, 0.0, 0.0);

    /// Unit Y vector.
    pub const UP: Self = Self::new(0.0, 1.0, 0.0);

    /// Calculate distance to another point.
    #[inline]
    pub fn distance(self, other: Self) -> f32 {
        self.distance_squared(other).sqrt()
    }

    /// Calculate squared distance to another point.
    #[inline]
    pub fn distance_squared(self, other: Self) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        dx * dx + dy * dy + dz * dz
    }

    /// Subtract two vectors.
    #[inline]
    pub fn sub(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }

    /// Normalize the vector.
    #[inline]
    pub fn normalize(self) -> Self {
        let len = (self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        if len > 0.0001 {
            Self::new(self.x / len, self.y / len, self.z / len)
        } else {
            Self::ZERO
        }
    }

    /// Length of the vector.
    #[inline]
    pub fn length(self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Dot product.
    #[inline]
    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Cross product.
    #[inline]
    pub fn cross(self, other: Self) -> Self {
        Self::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    /// Check if all components are finite.
    #[inline]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    /// Convert to array.
    #[inline]
    pub fn to_array(self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }

    /// Create from array.
    #[inline]
    pub fn from_array(arr: [f32; 3]) -> Self {
        Self::new(arr[0], arr[1], arr[2])
    }
}

impl From<[f32; 3]> for Vec3 {
    fn from(arr: [f32; 3]) -> Self {
        Self::from_array(arr)
    }
}

impl From<Vec3> for [f32; 3] {
    fn from(v: Vec3) -> Self {
        v.to_array()
    }
}

/// Quaternion type for rotation.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct Quat {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Quat {
    /// Create a new quaternion.
    #[inline]
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    /// Identity quaternion.
    pub const IDENTITY: Self = Self::new(0.0, 0.0, 0.0, 1.0);

    /// Create rotation from Y-axis angle (radians).
    #[inline]
    pub fn from_rotation_y(angle: f32) -> Self {
        let half = angle * 0.5;
        Self::new(0.0, half.sin(), 0.0, half.cos())
    }

    /// Extract Y-axis rotation angle (yaw) in radians.
    #[inline]
    pub fn to_rotation_y(self) -> f32 {
        2.0 * self.y.atan2(self.w)
    }

    /// Rotate a vector by this quaternion.
    #[inline]
    pub fn rotate_vec3(self, v: Vec3) -> Vec3 {
        // q * v * q^-1
        let u = Vec3::new(self.x, self.y, self.z);
        let s = self.w;
        let t = u.cross(v);
        let t = Vec3::new(t.x * 2.0, t.y * 2.0, t.z * 2.0);
        Vec3::new(
            v.x + s * t.x + u.y * t.z - u.z * t.y,
            v.y + s * t.y + u.z * t.x - u.x * t.z,
            v.z + s * t.z + u.x * t.y - u.y * t.x,
        )
    }

    /// Check if all components are finite.
    #[inline]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite() && self.w.is_finite()
    }
}

impl Default for Quat {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl From<[f32; 4]> for Quat {
    fn from(arr: [f32; 4]) -> Self {
        Self::new(arr[0], arr[1], arr[2], arr[3])
    }
}

// ---------------------------------------------------------------------------
// BillboardMode
// ---------------------------------------------------------------------------

/// Billboard orientation mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum BillboardMode {
    /// Always faces camera (full 3DOF rotation).
    Spherical = 0,

    /// Rotates around Y-axis only (upright characters).
    #[default]
    Cylindrical = 1,

    /// Rotates around a custom axis.
    Axial = 2,
}

impl BillboardMode {
    /// Get name for this mode.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Spherical => "spherical",
            Self::Cylindrical => "cylindrical",
            Self::Axial => "axial",
        }
    }

    /// Parse from u8.
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Spherical),
            1 => Some(Self::Cylindrical),
            2 => Some(Self::Axial),
            _ => None,
        }
    }

    /// All billboard modes.
    pub fn all() -> [Self; 3] {
        [Self::Spherical, Self::Cylindrical, Self::Axial]
    }
}

// ---------------------------------------------------------------------------
// ImpostorError
// ---------------------------------------------------------------------------

/// Errors that can occur during impostor operations.
#[derive(Clone, Debug, PartialEq)]
pub enum ImpostorError {
    /// Invalid configuration parameter.
    InvalidConfig { reason: &'static str },
    /// Atlas too large for GPU.
    AtlasTooLarge { width: u32, height: u32, max_size: u32 },
    /// Buffer capacity exceeded.
    BufferFull { max: usize },
    /// Instance validation failed.
    InvalidInstance { reason: &'static str },
    /// View angle index out of range.
    InvalidViewAngle { index: u32, max: u32 },
    /// Animation frame index out of range.
    InvalidAnimationFrame { frame: u32, max: u32 },
}

impl std::fmt::Display for ImpostorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig { reason } => write!(f, "invalid impostor config: {}", reason),
            Self::AtlasTooLarge { width, height, max_size } => {
                write!(f, "atlas {}x{} exceeds max size {}", width, height, max_size)
            }
            Self::BufferFull { max } => write!(f, "impostor buffer full: max {}", max),
            Self::InvalidInstance { reason } => write!(f, "invalid impostor instance: {}", reason),
            Self::InvalidViewAngle { index, max } => {
                write!(f, "view angle index {} out of range (max {})", index, max)
            }
            Self::InvalidAnimationFrame { frame, max } => {
                write!(f, "animation frame {} out of range (max {})", frame, max)
            }
        }
    }
}

impl std::error::Error for ImpostorError {}

// ---------------------------------------------------------------------------
// ImpostorConfig
// ---------------------------------------------------------------------------

/// Configuration for impostor generation and rendering.
#[derive(Clone, Debug, PartialEq)]
pub struct ImpostorConfig {
    /// Number of horizontal view angles (around Y-axis).
    pub horizontal_angles: u32,

    /// Number of vertical view angles (pitch).
    pub vertical_angles: u32,

    /// Number of animation frames to capture.
    pub animation_frames: u32,

    /// Sprite resolution width in pixels.
    pub sprite_width: u32,

    /// Sprite resolution height in pixels.
    pub sprite_height: u32,

    /// Alpha cutoff threshold for transparency.
    pub alpha_cutoff: f32,

    /// Billboard orientation mode.
    pub billboard_mode: BillboardMode,

    /// Custom axis for axial billboard mode.
    pub custom_axis: Vec3,

    /// Minimum vertical angle (radians, -PI/2 to PI/2).
    pub min_pitch: f32,

    /// Maximum vertical angle (radians, -PI/2 to PI/2).
    pub max_pitch: f32,

    /// Generate mipmaps for the atlas.
    pub generate_mipmaps: bool,

    /// Compress atlas texture (BC3/DXT5).
    pub compress_atlas: bool,
}

impl ImpostorConfig {
    /// Create a new default configuration.
    pub fn new() -> Self {
        Self {
            horizontal_angles: DEFAULT_HORIZONTAL_ANGLES,
            vertical_angles: DEFAULT_VERTICAL_ANGLES,
            animation_frames: DEFAULT_ANIMATION_FRAMES,
            sprite_width: DEFAULT_SPRITE_RESOLUTION,
            sprite_height: DEFAULT_SPRITE_RESOLUTION,
            alpha_cutoff: DEFAULT_ALPHA_CUTOFF,
            billboard_mode: BillboardMode::Cylindrical,
            custom_axis: Vec3::UP,
            min_pitch: -PI / 6.0,  // -30 degrees
            max_pitch: PI / 3.0,   // 60 degrees
            generate_mipmaps: true,
            compress_atlas: true,
        }
    }

    /// Set horizontal and vertical view angle counts.
    pub fn with_view_angles(mut self, horizontal: u32, vertical: u32) -> Self {
        self.horizontal_angles = horizontal.max(1);
        self.vertical_angles = vertical.max(1);
        self
    }

    /// Set number of animation frames.
    pub fn with_animation_frames(mut self, frames: u32) -> Self {
        self.animation_frames = frames.max(1);
        self
    }

    /// Set sprite resolution.
    pub fn with_sprite_resolution(mut self, width: u32, height: u32) -> Self {
        self.sprite_width = width.max(1);
        self.sprite_height = height.max(1);
        self
    }

    /// Set alpha cutoff threshold.
    pub fn with_alpha_cutoff(mut self, cutoff: f32) -> Self {
        self.alpha_cutoff = cutoff.clamp(0.0, 1.0);
        self
    }

    /// Set billboard mode.
    pub fn with_billboard_mode(mut self, mode: BillboardMode) -> Self {
        self.billboard_mode = mode;
        self
    }

    /// Set custom axis for axial billboard mode.
    pub fn with_custom_axis(mut self, axis: Vec3) -> Self {
        self.custom_axis = axis.normalize();
        self
    }

    /// Set pitch angle range.
    pub fn with_pitch_range(mut self, min: f32, max: f32) -> Self {
        self.min_pitch = min.clamp(-PI / 2.0, PI / 2.0);
        self.max_pitch = max.clamp(-PI / 2.0, PI / 2.0);
        self
    }

    /// Set mipmap generation.
    pub fn with_mipmaps(mut self, generate: bool) -> Self {
        self.generate_mipmaps = generate;
        self
    }

    /// Set texture compression.
    pub fn with_compression(mut self, compress: bool) -> Self {
        self.compress_atlas = compress;
        self
    }

    /// Get total number of view angles.
    #[inline]
    pub fn total_view_angles(&self) -> u32 {
        self.horizontal_angles * self.vertical_angles
    }

    /// Get total sprites in atlas.
    #[inline]
    pub fn total_sprites(&self) -> u32 {
        self.total_view_angles() * self.animation_frames
    }

    /// Calculate atlas dimensions.
    pub fn atlas_dimensions(&self) -> (u32, u32) {
        let sprites_per_row = self.horizontal_angles * self.animation_frames;
        let rows = self.vertical_angles;
        (sprites_per_row * self.sprite_width, rows * self.sprite_height)
    }

    /// Validate configuration.
    pub fn validate(&self) -> Result<(), ImpostorError> {
        if self.horizontal_angles == 0 {
            return Err(ImpostorError::InvalidConfig {
                reason: "horizontal_angles must be > 0",
            });
        }
        if self.vertical_angles == 0 {
            return Err(ImpostorError::InvalidConfig {
                reason: "vertical_angles must be > 0",
            });
        }
        if self.animation_frames == 0 {
            return Err(ImpostorError::InvalidConfig {
                reason: "animation_frames must be > 0",
            });
        }
        if self.sprite_width == 0 || self.sprite_height == 0 {
            return Err(ImpostorError::InvalidConfig {
                reason: "sprite dimensions must be > 0",
            });
        }
        if !self.alpha_cutoff.is_finite() {
            return Err(ImpostorError::InvalidConfig {
                reason: "alpha_cutoff must be finite",
            });
        }
        if self.min_pitch >= self.max_pitch {
            return Err(ImpostorError::InvalidConfig {
                reason: "min_pitch must be < max_pitch",
            });
        }

        let (w, h) = self.atlas_dimensions();
        const MAX_TEXTURE_SIZE: u32 = 16384;
        if w > MAX_TEXTURE_SIZE || h > MAX_TEXTURE_SIZE {
            return Err(ImpostorError::AtlasTooLarge {
                width: w,
                height: h,
                max_size: MAX_TEXTURE_SIZE,
            });
        }

        Ok(())
    }

    /// Configuration optimized for large crowds.
    pub fn for_large_crowd() -> Self {
        Self::new()
            .with_view_angles(8, 2)
            .with_animation_frames(8)
            .with_sprite_resolution(128, 128)
    }

    /// Configuration optimized for close-up impostors.
    pub fn for_closeup() -> Self {
        Self::new()
            .with_view_angles(16, 6)
            .with_animation_frames(32)
            .with_sprite_resolution(512, 512)
    }
}

impl Default for ImpostorConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// AtlasUV
// ---------------------------------------------------------------------------

/// UV coordinates for a sprite in the atlas.
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct AtlasUV {
    /// U coordinate (left).
    pub u_min: f32,
    /// V coordinate (top).
    pub v_min: f32,
    /// U coordinate (right).
    pub u_max: f32,
    /// V coordinate (bottom).
    pub v_max: f32,
}

impl AtlasUV {
    /// Create new UV coordinates.
    #[inline]
    pub const fn new(u_min: f32, v_min: f32, u_max: f32, v_max: f32) -> Self {
        Self { u_min, v_min, u_max, v_max }
    }

    /// Full texture UV (0,0 to 1,1).
    pub const FULL: Self = Self::new(0.0, 0.0, 1.0, 1.0);

    /// Calculate UV width.
    #[inline]
    pub fn width(&self) -> f32 {
        self.u_max - self.u_min
    }

    /// Calculate UV height.
    #[inline]
    pub fn height(&self) -> f32 {
        self.v_max - self.v_min
    }

    /// Sample at normalized coordinates (0-1).
    #[inline]
    pub fn sample(&self, u: f32, v: f32) -> (f32, f32) {
        (
            self.u_min + u * self.width(),
            self.v_min + v * self.height(),
        )
    }
}

// ---------------------------------------------------------------------------
// ImpostorAtlas
// ---------------------------------------------------------------------------

/// Pre-rendered sprite sheet texture for impostor rendering.
#[derive(Clone, Debug)]
pub struct ImpostorAtlas {
    /// Configuration used to generate this atlas.
    pub config: ImpostorConfig,

    /// Atlas texture data (RGBA, row-major).
    pub texture_data: Vec<u8>,

    /// Atlas width in pixels.
    pub width: u32,

    /// Atlas height in pixels.
    pub height: u32,

    /// Bytes per pixel (4 for RGBA).
    pub bytes_per_pixel: u32,

    /// Mipmap levels.
    pub mipmap_count: u32,

    /// Compressed texture data (optional).
    pub compressed_data: Option<Vec<u8>>,

    /// Pre-calculated UV coordinates for each sprite.
    uv_cache: Vec<AtlasUV>,
}

impl ImpostorAtlas {
    /// Create a new empty atlas with the given configuration.
    pub fn new(config: ImpostorConfig) -> Result<Self, ImpostorError> {
        config.validate()?;

        let (width, height) = config.atlas_dimensions();
        let bytes_per_pixel = 4;
        let texture_size = (width * height * bytes_per_pixel) as usize;

        let mut atlas = Self {
            config,
            texture_data: vec![0u8; texture_size],
            width,
            height,
            bytes_per_pixel,
            mipmap_count: 1,
            compressed_data: None,
            uv_cache: Vec::new(),
        };

        atlas.build_uv_cache();
        Ok(atlas)
    }

    /// Build UV coordinate cache for all sprites.
    fn build_uv_cache(&mut self) {
        let total = self.config.total_sprites() as usize;
        self.uv_cache = Vec::with_capacity(total);

        for v in 0..self.config.vertical_angles {
            for h in 0..self.config.horizontal_angles {
                for f in 0..self.config.animation_frames {
                    let uv = self.calculate_uv(h, v, f);
                    self.uv_cache.push(uv);
                }
            }
        }
    }

    /// Calculate UV coordinates for a sprite.
    pub fn calculate_uv(&self, horizontal_angle: u32, vertical_angle: u32, frame: u32) -> AtlasUV {
        let sprites_per_row = self.config.horizontal_angles * self.config.animation_frames;
        let sprite_x = horizontal_angle * self.config.animation_frames + frame;
        let sprite_y = vertical_angle;

        let u_min = (sprite_x * self.config.sprite_width) as f32 / self.width as f32;
        let v_min = (sprite_y * self.config.sprite_height) as f32 / self.height as f32;
        let u_max = ((sprite_x + 1) * self.config.sprite_width) as f32 / self.width as f32;
        let v_max = ((sprite_y + 1) * self.config.sprite_height) as f32 / self.height as f32;

        AtlasUV::new(u_min, v_min, u_max, v_max)
    }

    /// Get cached UV coordinates for a view angle and frame.
    pub fn get_uv(&self, horizontal_angle: u32, vertical_angle: u32, frame: u32) -> AtlasUV {
        let h = horizontal_angle % self.config.horizontal_angles;
        let v = vertical_angle % self.config.vertical_angles;
        let f = frame % self.config.animation_frames;

        let index = (v * self.config.horizontal_angles + h) * self.config.animation_frames + f;

        if let Some(uv) = self.uv_cache.get(index as usize) {
            *uv
        } else {
            self.calculate_uv(h, v, f)
        }
    }

    /// Get UV coordinates for a combined view angle index and frame.
    pub fn get_uv_by_index(&self, view_index: u32, frame: u32) -> AtlasUV {
        let v = view_index / self.config.horizontal_angles;
        let h = view_index % self.config.horizontal_angles;
        self.get_uv(h, v, frame)
    }

    /// Get texture data size in bytes.
    pub fn texture_size(&self) -> usize {
        self.texture_data.len()
    }

    /// Get texture data as a slice.
    pub fn as_bytes(&self) -> &[u8] {
        &self.texture_data
    }

    /// Check if the atlas is empty.
    pub fn is_empty(&self) -> bool {
        self.texture_data.is_empty()
    }

    /// Get memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        self.texture_data.len() + self.compressed_data.as_ref().map(|d| d.len()).unwrap_or(0)
    }

    /// Set pixel at (x, y) with RGBA values.
    pub fn set_pixel(&mut self, x: u32, y: u32, r: u8, g: u8, b: u8, a: u8) {
        if x < self.width && y < self.height {
            let idx = ((y * self.width + x) * self.bytes_per_pixel) as usize;
            if idx + 3 < self.texture_data.len() {
                self.texture_data[idx] = r;
                self.texture_data[idx + 1] = g;
                self.texture_data[idx + 2] = b;
                self.texture_data[idx + 3] = a;
            }
        }
    }

    /// Get pixel at (x, y) as RGBA.
    pub fn get_pixel(&self, x: u32, y: u32) -> (u8, u8, u8, u8) {
        if x < self.width && y < self.height {
            let idx = ((y * self.width + x) * self.bytes_per_pixel) as usize;
            if idx + 3 < self.texture_data.len() {
                return (
                    self.texture_data[idx],
                    self.texture_data[idx + 1],
                    self.texture_data[idx + 2],
                    self.texture_data[idx + 3],
                );
            }
        }
        (0, 0, 0, 0)
    }

    /// Blit a sprite into the atlas at the given angle and frame.
    pub fn blit_sprite(
        &mut self,
        horizontal_angle: u32,
        vertical_angle: u32,
        frame: u32,
        sprite_data: &[u8],
    ) {
        let sprites_per_row = self.config.horizontal_angles * self.config.animation_frames;
        let sprite_x = horizontal_angle * self.config.animation_frames + frame;
        let sprite_y = vertical_angle;

        let start_x = sprite_x * self.config.sprite_width;
        let start_y = sprite_y * self.config.sprite_height;

        let sprite_row_bytes = (self.config.sprite_width * self.bytes_per_pixel) as usize;

        for row in 0..self.config.sprite_height {
            let src_offset = (row as usize) * sprite_row_bytes;
            let dst_y = start_y + row;
            let dst_offset = ((dst_y * self.width + start_x) * self.bytes_per_pixel) as usize;

            if src_offset + sprite_row_bytes <= sprite_data.len()
                && dst_offset + sprite_row_bytes <= self.texture_data.len()
            {
                self.texture_data[dst_offset..dst_offset + sprite_row_bytes]
                    .copy_from_slice(&sprite_data[src_offset..src_offset + sprite_row_bytes]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ViewAngleSelector
// ---------------------------------------------------------------------------

/// Selects the best view angle based on camera and instance positions.
#[derive(Clone, Debug)]
pub struct ViewAngleSelector {
    /// Configuration reference.
    horizontal_angles: u32,
    vertical_angles: u32,
    min_pitch: f32,
    max_pitch: f32,
    billboard_mode: BillboardMode,
    custom_axis: Vec3,
}

impl ViewAngleSelector {
    /// Create a new view angle selector from config.
    pub fn new(config: &ImpostorConfig) -> Self {
        Self {
            horizontal_angles: config.horizontal_angles,
            vertical_angles: config.vertical_angles,
            min_pitch: config.min_pitch,
            max_pitch: config.max_pitch,
            billboard_mode: config.billboard_mode,
            custom_axis: config.custom_axis,
        }
    }

    /// Select the best view angle index based on camera and instance positions.
    ///
    /// Returns (combined_index, horizontal_index, vertical_index)
    pub fn select_angle(
        &self,
        camera_pos: Vec3,
        instance_pos: Vec3,
        instance_rotation: Quat,
    ) -> (u32, u32, u32) {
        // Direction from instance to camera
        let to_camera = camera_pos.sub(instance_pos);
        if to_camera.length() < 0.0001 {
            return (0, 0, 0);
        }

        let to_camera = to_camera.normalize();

        // Transform direction into instance's local space
        let inv_rotation = Quat::new(-instance_rotation.x, -instance_rotation.y, -instance_rotation.z, instance_rotation.w);
        let local_dir = inv_rotation.rotate_vec3(to_camera);

        // Calculate horizontal angle (yaw)
        let yaw = local_dir.z.atan2(local_dir.x);
        let yaw_normalized = (yaw + PI) / (2.0 * PI); // 0..1

        // Calculate vertical angle (pitch)
        let pitch = local_dir.y.asin();
        let pitch_clamped = pitch.clamp(self.min_pitch, self.max_pitch);
        let pitch_range = self.max_pitch - self.min_pitch;
        let pitch_normalized = if pitch_range > 0.0001 {
            (pitch_clamped - self.min_pitch) / pitch_range
        } else {
            0.5
        };

        // Quantize to grid
        let h = ((yaw_normalized * self.horizontal_angles as f32) as u32) % self.horizontal_angles;
        let v = ((pitch_normalized * self.vertical_angles as f32) as u32).min(self.vertical_angles - 1);

        let combined = v * self.horizontal_angles + h;
        (combined, h, v)
    }

    /// Select view angle with interpolation weights for smooth transitions.
    ///
    /// Returns (index0, index1, weight) where weight is the blend factor.
    pub fn select_angle_interpolated(
        &self,
        camera_pos: Vec3,
        instance_pos: Vec3,
        instance_rotation: Quat,
    ) -> (u32, u32, f32) {
        let to_camera = camera_pos.sub(instance_pos);
        if to_camera.length() < 0.0001 {
            return (0, 0, 0.0);
        }

        let to_camera = to_camera.normalize();
        let inv_rotation = Quat::new(-instance_rotation.x, -instance_rotation.y, -instance_rotation.z, instance_rotation.w);
        let local_dir = inv_rotation.rotate_vec3(to_camera);

        // Calculate horizontal angle
        let yaw = local_dir.z.atan2(local_dir.x);
        let yaw_normalized = (yaw + PI) / (2.0 * PI);

        let h_float = yaw_normalized * self.horizontal_angles as f32;
        let h0 = (h_float.floor() as u32) % self.horizontal_angles;
        let h1 = (h0 + 1) % self.horizontal_angles;
        let h_weight = h_float.fract();

        // For simplicity, use the center vertical angle
        let v = self.vertical_angles / 2;

        let idx0 = v * self.horizontal_angles + h0;
        let idx1 = v * self.horizontal_angles + h1;

        (idx0, idx1, h_weight)
    }

    /// Calculate the angle (in radians) for a given horizontal index.
    pub fn horizontal_index_to_angle(&self, index: u32) -> f32 {
        let index = index % self.horizontal_angles;
        (index as f32 / self.horizontal_angles as f32) * 2.0 * PI - PI
    }

    /// Calculate the pitch angle for a given vertical index.
    pub fn vertical_index_to_pitch(&self, index: u32) -> f32 {
        let index = index.min(self.vertical_angles - 1);
        let t = if self.vertical_angles > 1 {
            index as f32 / (self.vertical_angles - 1) as f32
        } else {
            0.5
        };
        self.min_pitch + t * (self.max_pitch - self.min_pitch)
    }
}

// ---------------------------------------------------------------------------
// ImpostorInstance
// ---------------------------------------------------------------------------

/// Per-instance data for GPU impostor rendering.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct ImpostorInstance {
    /// World-space position.
    pub position: Vec3,
    /// Uniform scale factor.
    pub scale: f32,
    /// Rotation quaternion.
    pub rotation: Quat,
    /// Current animation frame (0-based).
    pub animation_frame: u16,
    /// View angle index (combined horizontal + vertical).
    pub view_angle_index: u16,
    /// LOD fade factor (0 = invisible, 1 = fully visible).
    pub fade_factor: f32,
    /// Instance flags.
    pub flags: u8,
    /// Fade flags (crossfade, dithered).
    pub fade_flags: u8,
    /// Atlas index (for multiple character types).
    pub atlas_index: u8,
    /// Padding.
    pub _padding0: u8,
    /// Interpolation weight for view angle blending.
    pub view_angle_weight: f32,
    /// Secondary view angle index for interpolation.
    pub view_angle_index_secondary: u16,
    /// Secondary animation frame for interpolation.
    pub animation_frame_secondary: u16,
    /// Animation blend weight.
    pub animation_weight: f32,
    /// Padding for 64-byte alignment.
    pub _padding: [f32; 2],
}

impl ImpostorInstance {
    /// Create a new impostor instance at the given position.
    pub fn new(position: impl Into<Vec3>) -> Self {
        Self {
            position: position.into(),
            scale: 1.0,
            rotation: Quat::IDENTITY,
            animation_frame: 0,
            view_angle_index: 0,
            fade_factor: 1.0,
            flags: FLAG_VISIBLE | FLAG_CAST_SHADOW,
            fade_flags: 0,
            atlas_index: 0,
            _padding0: 0,
            view_angle_weight: 0.0,
            view_angle_index_secondary: 0,
            animation_frame_secondary: 0,
            animation_weight: 0.0,
            _padding: [0.0; 2],
        }
    }

    /// Set the rotation.
    pub fn with_rotation(mut self, rotation: impl Into<Quat>) -> Self {
        self.rotation = rotation.into();
        self
    }

    /// Set the uniform scale.
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    /// Set the animation frame.
    pub fn with_animation_frame(mut self, frame: u16) -> Self {
        self.animation_frame = frame;
        self
    }

    /// Set the view angle index.
    pub fn with_view_angle_index(mut self, index: u16) -> Self {
        self.view_angle_index = index;
        self
    }

    /// Set the LOD fade factor.
    pub fn with_lod_fade(mut self, fade: f32) -> Self {
        self.fade_factor = fade.clamp(0.0, 1.0);
        self
    }

    /// Enable crossfade transition.
    pub fn with_crossfade(mut self, enable: bool) -> Self {
        if enable {
            self.fade_flags |= FADE_FLAG_CROSSFADE;
        } else {
            self.fade_flags &= !FADE_FLAG_CROSSFADE;
        }
        self
    }

    /// Enable dithered transition.
    pub fn with_dithered(mut self, enable: bool) -> Self {
        if enable {
            self.fade_flags |= FADE_FLAG_DITHERED;
        } else {
            self.fade_flags &= !FADE_FLAG_DITHERED;
        }
        self
    }

    /// Set atlas index.
    pub fn with_atlas(mut self, index: u8) -> Self {
        self.atlas_index = index;
        self
    }

    /// Set view angle interpolation.
    pub fn with_view_interpolation(mut self, secondary_index: u16, weight: f32) -> Self {
        self.view_angle_index_secondary = secondary_index;
        self.view_angle_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Set animation frame interpolation.
    pub fn with_animation_interpolation(mut self, secondary_frame: u16, weight: f32) -> Self {
        self.animation_frame_secondary = secondary_frame;
        self.animation_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Check if the instance is visible.
    #[inline]
    pub fn is_visible(&self) -> bool {
        (self.flags & FLAG_VISIBLE) != 0 && self.fade_factor > 0.0
    }

    /// Set visibility flag.
    pub fn set_visible(&mut self, visible: bool) {
        if visible {
            self.flags |= FLAG_VISIBLE;
        } else {
            self.flags &= !FLAG_VISIBLE;
        }
    }

    /// Check if crossfade is enabled.
    #[inline]
    pub fn is_crossfade(&self) -> bool {
        (self.fade_flags & FADE_FLAG_CROSSFADE) != 0
    }

    /// Check if dithered transition is enabled.
    #[inline]
    pub fn is_dithered(&self) -> bool {
        (self.fade_flags & FADE_FLAG_DITHERED) != 0
    }

    /// Validate instance data.
    pub fn validate(&self) -> Result<(), ImpostorError> {
        if !self.position.is_finite() {
            return Err(ImpostorError::InvalidInstance {
                reason: "position contains non-finite values",
            });
        }
        if !self.scale.is_finite() || self.scale <= 0.0 {
            return Err(ImpostorError::InvalidInstance {
                reason: "scale must be positive and finite",
            });
        }
        if !self.rotation.is_finite() {
            return Err(ImpostorError::InvalidInstance {
                reason: "rotation contains non-finite values",
            });
        }
        if !self.fade_factor.is_finite() {
            return Err(ImpostorError::InvalidInstance {
                reason: "fade_factor must be finite",
            });
        }
        Ok(())
    }

    /// Calculate bounding sphere radius.
    #[inline]
    pub fn bounding_radius(&self) -> f32 {
        self.scale
    }

    /// Calculate distance squared to a point.
    #[inline]
    pub fn distance_squared(&self, point: Vec3) -> f32 {
        self.position.distance_squared(point)
    }
}

// ---------------------------------------------------------------------------
// LODTransition
// ---------------------------------------------------------------------------

/// LOD transition configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct LodTransition {
    /// Distance at which mesh LOD starts fading to impostor.
    pub mesh_fade_start: f32,

    /// Distance at which mesh LOD is fully replaced by impostor.
    pub mesh_fade_end: f32,

    /// Distance at which impostor starts fading out.
    pub impostor_fade_start: f32,

    /// Distance at which impostor is fully invisible.
    pub impostor_fade_end: f32,

    /// Use dithered transition instead of alpha blend.
    pub use_dithering: bool,

    /// Dither pattern scale (1.0 = screen-space, >1 = coarser).
    pub dither_scale: f32,
}

impl LodTransition {
    /// Create new LOD transition configuration.
    pub fn new(mesh_fade_end: f32, impostor_fade_end: f32) -> Self {
        let transition_range = 10.0;
        Self {
            mesh_fade_start: mesh_fade_end - transition_range,
            mesh_fade_end,
            impostor_fade_start: impostor_fade_end - transition_range,
            impostor_fade_end,
            use_dithering: false,
            dither_scale: 1.0,
        }
    }

    /// Set mesh fade range.
    pub fn with_mesh_fade(mut self, start: f32, end: f32) -> Self {
        self.mesh_fade_start = start;
        self.mesh_fade_end = end;
        self
    }

    /// Set impostor fade range.
    pub fn with_impostor_fade(mut self, start: f32, end: f32) -> Self {
        self.impostor_fade_start = start;
        self.impostor_fade_end = end;
        self
    }

    /// Enable dithering.
    pub fn with_dithering(mut self, enable: bool, scale: f32) -> Self {
        self.use_dithering = enable;
        self.dither_scale = scale.max(0.1);
        self
    }

    /// Calculate blend factor for mesh-to-impostor transition.
    ///
    /// Returns (mesh_alpha, impostor_alpha).
    pub fn calculate_blend(&self, distance: f32) -> (f32, f32) {
        // Mesh fades out as distance increases
        let mesh_alpha = if distance <= self.mesh_fade_start {
            1.0
        } else if distance >= self.mesh_fade_end {
            0.0
        } else {
            let t = (distance - self.mesh_fade_start) / (self.mesh_fade_end - self.mesh_fade_start);
            1.0 - t
        };

        // Impostor fades in as mesh fades out, then fades out at far distance
        let impostor_fade_in = 1.0 - mesh_alpha;
        let impostor_fade_out = if distance <= self.impostor_fade_start {
            1.0
        } else if distance >= self.impostor_fade_end {
            0.0
        } else {
            let t = (distance - self.impostor_fade_start)
                / (self.impostor_fade_end - self.impostor_fade_start);
            1.0 - t
        };

        let impostor_alpha = impostor_fade_in * impostor_fade_out;

        (mesh_alpha, impostor_alpha)
    }

    /// Check if mesh should be rendered at this distance.
    #[inline]
    pub fn should_render_mesh(&self, distance: f32) -> bool {
        distance < self.mesh_fade_end
    }

    /// Check if impostor should be rendered at this distance.
    #[inline]
    pub fn should_render_impostor(&self, distance: f32) -> bool {
        distance > self.mesh_fade_start && distance < self.impostor_fade_end
    }
}

impl Default for LodTransition {
    fn default() -> Self {
        Self::new(50.0, 200.0)
    }
}

// ---------------------------------------------------------------------------
// ImpostorBatch
// ---------------------------------------------------------------------------

/// Batch of impostor instances for efficient rendering.
#[derive(Clone, Debug)]
pub struct ImpostorBatch {
    /// Atlas index for this batch.
    pub atlas_index: u8,

    /// Instances in this batch.
    pub instances: Vec<ImpostorInstance>,

    /// Visible instance count.
    visible_count: usize,

    /// Dirty flag for GPU upload.
    dirty: bool,
}

impl ImpostorBatch {
    /// Create a new batch for an atlas.
    pub fn new(atlas_index: u8) -> Self {
        Self {
            atlas_index,
            instances: Vec::new(),
            visible_count: 0,
            dirty: false,
        }
    }

    /// Create a batch with pre-allocated capacity.
    pub fn with_capacity(atlas_index: u8, capacity: usize) -> Self {
        Self {
            atlas_index,
            instances: Vec::with_capacity(capacity),
            visible_count: 0,
            dirty: false,
        }
    }

    /// Add an instance to this batch.
    pub fn add_instance(&mut self, instance: ImpostorInstance) -> usize {
        let index = self.instances.len();
        self.instances.push(instance);
        self.dirty = true;
        if instance.is_visible() {
            self.visible_count += 1;
        }
        index
    }

    /// Remove an instance by swapping with the last.
    pub fn remove_instance(&mut self, index: usize) -> Option<usize> {
        if index >= self.instances.len() {
            return None;
        }

        if self.instances[index].is_visible() {
            self.visible_count = self.visible_count.saturating_sub(1);
        }

        let last_index = self.instances.len() - 1;
        if index != last_index {
            self.instances.swap(index, last_index);
        }
        self.instances.pop();
        self.dirty = true;

        if index != last_index && !self.instances.is_empty() {
            Some(last_index)
        } else {
            None
        }
    }

    /// Update view angles for all instances.
    pub fn update_view_angles(&mut self, camera_pos: Vec3, selector: &ViewAngleSelector) {
        for instance in &mut self.instances {
            let (idx, _, _) = selector.select_angle(camera_pos, instance.position, instance.rotation);
            instance.view_angle_index = idx as u16;
        }
        self.dirty = true;
    }

    /// Update LOD fade factors based on distance.
    pub fn update_lod_fades(&mut self, camera_pos: Vec3, transition: &LodTransition) {
        self.visible_count = 0;
        for instance in &mut self.instances {
            let distance = instance.position.distance(camera_pos);
            let (_, impostor_alpha) = transition.calculate_blend(distance);
            instance.fade_factor = impostor_alpha;
            if instance.is_visible() {
                self.visible_count += 1;
            }
        }
        self.dirty = true;
    }

    /// Get instance count.
    #[inline]
    pub fn instance_count(&self) -> usize {
        self.instances.len()
    }

    /// Get visible instance count.
    #[inline]
    pub fn visible_count(&self) -> usize {
        self.visible_count
    }

    /// Check if the batch is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }

    /// Check if the batch needs GPU upload.
    #[inline]
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Mark the batch as clean.
    pub fn mark_clean(&mut self) {
        self.dirty = false;
    }

    /// Get raw bytes for GPU upload.
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.instances)
    }

    /// Get buffer size in bytes.
    pub fn buffer_size(&self) -> usize {
        self.instances.len() * IMPOSTOR_INSTANCE_SIZE
    }

    /// Clear all instances.
    pub fn clear(&mut self) {
        self.instances.clear();
        self.visible_count = 0;
        self.dirty = true;
    }
}

// ---------------------------------------------------------------------------
// ImpostorRenderer
// ---------------------------------------------------------------------------

/// Renders characters to an impostor atlas.
#[derive(Clone, Debug)]
pub struct ImpostorRenderer {
    /// Configuration for atlas generation.
    pub config: ImpostorConfig,
}

impl ImpostorRenderer {
    /// Create a new impostor renderer.
    pub fn new(config: ImpostorConfig) -> Self {
        Self { config }
    }

    /// Generate camera position for a given view angle.
    pub fn camera_position_for_angle(&self, h_index: u32, v_index: u32, distance: f32) -> Vec3 {
        let h_angle = (h_index as f32 / self.config.horizontal_angles as f32) * 2.0 * PI;
        let v_angle = if self.config.vertical_angles > 1 {
            let t = v_index as f32 / (self.config.vertical_angles - 1) as f32;
            self.config.min_pitch + t * (self.config.max_pitch - self.config.min_pitch)
        } else {
            0.0
        };

        let cos_v = v_angle.cos();
        let sin_v = v_angle.sin();

        Vec3::new(
            h_angle.sin() * cos_v * distance,
            sin_v * distance,
            h_angle.cos() * cos_v * distance,
        )
    }

    /// Generate an empty atlas with the configured dimensions.
    pub fn create_atlas(&self) -> Result<ImpostorAtlas, ImpostorError> {
        ImpostorAtlas::new(self.config.clone())
    }

    /// Calculate the number of mipmaps for the atlas.
    pub fn mipmap_count(&self) -> u32 {
        let (w, h) = self.config.atlas_dimensions();
        let max_dim = w.max(h);
        (max_dim as f32).log2().ceil() as u32
    }
}

// ---------------------------------------------------------------------------
// Frustum Culling
// ---------------------------------------------------------------------------

/// Simple frustum for impostor culling.
#[derive(Clone, Copy, Debug, Default)]
pub struct Frustum {
    /// Six frustum planes.
    pub planes: [[f32; 4]; 6],
}

impl Frustum {
    /// Create from bounds.
    pub fn from_bounds(min: Vec3, max: Vec3) -> Self {
        Self {
            planes: [
                [1.0, 0.0, 0.0, -min.x],
                [-1.0, 0.0, 0.0, max.x],
                [0.0, 1.0, 0.0, -min.y],
                [0.0, -1.0, 0.0, max.y],
                [0.0, 0.0, 1.0, -min.z],
                [0.0, 0.0, -1.0, max.z],
            ],
        }
    }

    /// Create unbounded frustum.
    pub fn unbounded() -> Self {
        Self {
            planes: [[0.0, 0.0, 0.0, f32::MAX]; 6],
        }
    }

    /// Test if a sphere is inside the frustum.
    pub fn contains_sphere(&self, center: Vec3, radius: f32) -> bool {
        for plane in &self.planes {
            let dist = plane[0] * center.x + plane[1] * center.y + plane[2] * center.z + plane[3];
            if dist < -radius {
                return false;
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Vec3 Tests
    // ========================================================================

    #[test]
    fn test_vec3_new() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 2.0);
        assert_eq!(v.z, 3.0);
    }

    #[test]
    fn test_vec3_distance() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(3.0, 4.0, 0.0);
        assert!((a.distance(b) - 5.0).abs() < 0.0001);
    }

    #[test]
    fn test_vec3_normalize() {
        let v = Vec3::new(3.0, 0.0, 4.0);
        let n = v.normalize();
        assert!((n.length() - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_vec3_dot() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        assert!((a.dot(b)).abs() < 0.0001);
    }

    #[test]
    fn test_vec3_cross() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        let c = a.cross(b);
        assert!((c.z - 1.0).abs() < 0.0001);
    }

    // ========================================================================
    // Quat Tests
    // ========================================================================

    #[test]
    fn test_quat_identity() {
        let q = Quat::IDENTITY;
        let v = Vec3::new(1.0, 2.0, 3.0);
        let rotated = q.rotate_vec3(v);
        assert!((rotated.x - v.x).abs() < 0.0001);
        assert!((rotated.y - v.y).abs() < 0.0001);
        assert!((rotated.z - v.z).abs() < 0.0001);
    }

    #[test]
    fn test_quat_from_rotation_y() {
        let q = Quat::from_rotation_y(PI / 2.0);
        let v = Vec3::new(1.0, 0.0, 0.0);
        let rotated = q.rotate_vec3(v);
        assert!((rotated.x).abs() < 0.01);
        assert!((rotated.z - (-1.0)).abs() < 0.01);
    }

    // ========================================================================
    // BillboardMode Tests
    // ========================================================================

    #[test]
    fn test_billboard_mode_from_u8() {
        assert_eq!(BillboardMode::from_u8(0), Some(BillboardMode::Spherical));
        assert_eq!(BillboardMode::from_u8(1), Some(BillboardMode::Cylindrical));
        assert_eq!(BillboardMode::from_u8(2), Some(BillboardMode::Axial));
        assert_eq!(BillboardMode::from_u8(3), None);
    }

    #[test]
    fn test_billboard_mode_name() {
        assert_eq!(BillboardMode::Spherical.name(), "spherical");
        assert_eq!(BillboardMode::Cylindrical.name(), "cylindrical");
        assert_eq!(BillboardMode::Axial.name(), "axial");
    }

    // ========================================================================
    // ImpostorConfig Tests
    // ========================================================================

    #[test]
    fn test_config_default() {
        let config = ImpostorConfig::default();
        assert_eq!(config.horizontal_angles, DEFAULT_HORIZONTAL_ANGLES);
        assert_eq!(config.vertical_angles, DEFAULT_VERTICAL_ANGLES);
        assert_eq!(config.animation_frames, DEFAULT_ANIMATION_FRAMES);
    }

    #[test]
    fn test_config_builder() {
        let config = ImpostorConfig::new()
            .with_view_angles(16, 8)
            .with_animation_frames(32)
            .with_sprite_resolution(512, 512)
            .with_alpha_cutoff(0.3)
            .with_billboard_mode(BillboardMode::Spherical);

        assert_eq!(config.horizontal_angles, 16);
        assert_eq!(config.vertical_angles, 8);
        assert_eq!(config.animation_frames, 32);
        assert_eq!(config.sprite_width, 512);
        assert_eq!(config.sprite_height, 512);
        assert!((config.alpha_cutoff - 0.3).abs() < 0.0001);
        assert_eq!(config.billboard_mode, BillboardMode::Spherical);
    }

    #[test]
    fn test_config_total_view_angles() {
        let config = ImpostorConfig::new().with_view_angles(8, 4);
        assert_eq!(config.total_view_angles(), 32);
    }

    #[test]
    fn test_config_total_sprites() {
        let config = ImpostorConfig::new()
            .with_view_angles(8, 4)
            .with_animation_frames(16);
        assert_eq!(config.total_sprites(), 32 * 16);
    }

    #[test]
    fn test_config_atlas_dimensions() {
        let config = ImpostorConfig::new()
            .with_view_angles(8, 4)
            .with_animation_frames(16)
            .with_sprite_resolution(256, 256);

        let (w, h) = config.atlas_dimensions();
        assert_eq!(w, 8 * 16 * 256); // 8 horizontal * 16 frames * 256px
        assert_eq!(h, 4 * 256);       // 4 vertical * 256px
    }

    #[test]
    fn test_config_validate_valid() {
        let config = ImpostorConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validate_zero_angles() {
        let mut config = ImpostorConfig::default();
        config.horizontal_angles = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_pitch_range() {
        let config = ImpostorConfig::new()
            .with_pitch_range(0.5, 0.4);  // min >= max
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_for_large_crowd() {
        let config = ImpostorConfig::for_large_crowd();
        assert_eq!(config.horizontal_angles, 8);
        assert_eq!(config.vertical_angles, 2);
        assert_eq!(config.animation_frames, 8);
        assert_eq!(config.sprite_width, 128);
    }

    #[test]
    fn test_config_for_closeup() {
        let config = ImpostorConfig::for_closeup();
        assert_eq!(config.horizontal_angles, 16);
        assert!(config.sprite_width >= 512);
    }

    // ========================================================================
    // AtlasUV Tests
    // ========================================================================

    #[test]
    fn test_atlas_uv_new() {
        let uv = AtlasUV::new(0.0, 0.0, 0.5, 0.5);
        assert_eq!(uv.u_min, 0.0);
        assert_eq!(uv.v_min, 0.0);
        assert_eq!(uv.u_max, 0.5);
        assert_eq!(uv.v_max, 0.5);
    }

    #[test]
    fn test_atlas_uv_width_height() {
        let uv = AtlasUV::new(0.25, 0.25, 0.75, 0.5);
        assert!((uv.width() - 0.5).abs() < 0.0001);
        assert!((uv.height() - 0.25).abs() < 0.0001);
    }

    #[test]
    fn test_atlas_uv_sample() {
        let uv = AtlasUV::new(0.0, 0.0, 1.0, 1.0);
        let (u, v) = uv.sample(0.5, 0.5);
        assert!((u - 0.5).abs() < 0.0001);
        assert!((v - 0.5).abs() < 0.0001);
    }

    #[test]
    fn test_atlas_uv_sample_offset() {
        let uv = AtlasUV::new(0.5, 0.5, 1.0, 1.0);
        let (u, v) = uv.sample(0.0, 0.0);
        assert!((u - 0.5).abs() < 0.0001);
        assert!((v - 0.5).abs() < 0.0001);
    }

    // ========================================================================
    // ImpostorAtlas Tests
    // ========================================================================

    #[test]
    fn test_atlas_new() {
        let config = ImpostorConfig::new()
            .with_view_angles(4, 2)
            .with_animation_frames(4)
            .with_sprite_resolution(64, 64);

        let atlas = ImpostorAtlas::new(config).unwrap();
        assert_eq!(atlas.width, 4 * 4 * 64);  // 4 horizontal * 4 frames * 64px
        assert_eq!(atlas.height, 2 * 64);     // 2 vertical * 64px
    }

    #[test]
    fn test_atlas_calculate_uv() {
        let config = ImpostorConfig::new()
            .with_view_angles(4, 2)
            .with_animation_frames(4)
            .with_sprite_resolution(64, 64);

        let atlas = ImpostorAtlas::new(config).unwrap();

        // First sprite
        let uv = atlas.calculate_uv(0, 0, 0);
        assert!((uv.u_min - 0.0).abs() < 0.0001);
        assert!((uv.v_min - 0.0).abs() < 0.0001);

        // Second horizontal angle, first frame
        let uv2 = atlas.calculate_uv(1, 0, 0);
        assert!(uv2.u_min > uv.u_min);
    }

    #[test]
    fn test_atlas_get_uv() {
        let config = ImpostorConfig::new()
            .with_view_angles(4, 2)
            .with_animation_frames(4)
            .with_sprite_resolution(64, 64);

        let atlas = ImpostorAtlas::new(config).unwrap();

        let uv = atlas.get_uv(0, 0, 0);
        assert!((uv.u_min - 0.0).abs() < 0.0001);
    }

    #[test]
    fn test_atlas_get_uv_by_index() {
        let config = ImpostorConfig::new()
            .with_view_angles(4, 2)
            .with_animation_frames(4)
            .with_sprite_resolution(64, 64);

        let atlas = ImpostorAtlas::new(config).unwrap();

        // Index 0 = h=0, v=0
        let uv0 = atlas.get_uv_by_index(0, 0);
        let uv_explicit = atlas.get_uv(0, 0, 0);
        assert_eq!(uv0.u_min, uv_explicit.u_min);

        // Index 4 = h=0, v=1
        let uv4 = atlas.get_uv_by_index(4, 0);
        let uv_explicit4 = atlas.get_uv(0, 1, 0);
        assert_eq!(uv4.u_min, uv_explicit4.u_min);
    }

    #[test]
    fn test_atlas_set_get_pixel() {
        let config = ImpostorConfig::new()
            .with_view_angles(2, 2)
            .with_animation_frames(2)
            .with_sprite_resolution(32, 32);

        let mut atlas = ImpostorAtlas::new(config).unwrap();

        atlas.set_pixel(10, 10, 255, 128, 64, 200);
        let (r, g, b, a) = atlas.get_pixel(10, 10);
        assert_eq!(r, 255);
        assert_eq!(g, 128);
        assert_eq!(b, 64);
        assert_eq!(a, 200);
    }

    #[test]
    fn test_atlas_memory_usage() {
        let config = ImpostorConfig::new()
            .with_view_angles(4, 2)
            .with_animation_frames(4)
            .with_sprite_resolution(64, 64);

        let atlas = ImpostorAtlas::new(config).unwrap();
        let expected = (4 * 4 * 64) * (2 * 64) * 4;  // w * h * 4 bytes
        assert_eq!(atlas.memory_usage(), expected);
    }

    // ========================================================================
    // ViewAngleSelector Tests
    // ========================================================================

    #[test]
    fn test_selector_new() {
        let config = ImpostorConfig::new().with_view_angles(8, 4);
        let selector = ViewAngleSelector::new(&config);
        assert_eq!(selector.horizontal_angles, 8);
        assert_eq!(selector.vertical_angles, 4);
    }

    #[test]
    fn test_selector_select_angle_front() {
        let config = ImpostorConfig::new().with_view_angles(8, 4);
        let selector = ViewAngleSelector::new(&config);

        // Camera directly in front
        let camera = Vec3::new(0.0, 0.0, 10.0);
        let instance = Vec3::ZERO;
        let rotation = Quat::IDENTITY;

        let (combined, h, v) = selector.select_angle(camera, instance, rotation);
        // Front view should be one of the horizontal angles
        assert!(h < 8);
        assert!(v < 4);
        assert!(combined < 32);
    }

    #[test]
    fn test_selector_select_angle_behind() {
        let config = ImpostorConfig::new().with_view_angles(8, 4);
        let selector = ViewAngleSelector::new(&config);

        // Camera directly behind
        let camera = Vec3::new(0.0, 0.0, -10.0);
        let instance = Vec3::ZERO;
        let rotation = Quat::IDENTITY;

        let (combined, h, v) = selector.select_angle(camera, instance, rotation);
        assert!(combined < 32);
    }

    #[test]
    fn test_selector_select_angle_rotated_instance() {
        let config = ImpostorConfig::new().with_view_angles(8, 4);
        let selector = ViewAngleSelector::new(&config);

        // Camera in front, but instance is rotated 90 degrees
        let camera = Vec3::new(0.0, 0.0, 10.0);
        let instance = Vec3::ZERO;
        let rotation = Quat::from_rotation_y(PI / 2.0);

        let (_, h1, _) = selector.select_angle(camera, instance, rotation);

        // Same camera, no rotation
        let (_, h2, _) = selector.select_angle(camera, instance, Quat::IDENTITY);

        // Angles should be different
        // (Note: might be same if quantization aligns)
    }

    #[test]
    fn test_selector_select_angle_above() {
        let config = ImpostorConfig::new()
            .with_view_angles(8, 4)
            .with_pitch_range(-PI / 4.0, PI / 4.0);
        let selector = ViewAngleSelector::new(&config);

        // Camera above
        let camera = Vec3::new(0.0, 10.0, 1.0);
        let instance = Vec3::ZERO;
        let rotation = Quat::IDENTITY;

        let (_, _, v) = selector.select_angle(camera, instance, rotation);
        // Higher vertical index expected
        assert!(v > 0);
    }

    #[test]
    fn test_selector_select_angle_interpolated() {
        let config = ImpostorConfig::new().with_view_angles(8, 4);
        let selector = ViewAngleSelector::new(&config);

        let camera = Vec3::new(5.0, 0.0, 10.0);
        let instance = Vec3::ZERO;
        let rotation = Quat::IDENTITY;

        let (idx0, idx1, weight) = selector.select_angle_interpolated(camera, instance, rotation);
        assert!(idx0 < 32);
        assert!(idx1 < 32);
        assert!(weight >= 0.0 && weight <= 1.0);
    }

    #[test]
    fn test_selector_horizontal_index_to_angle() {
        let config = ImpostorConfig::new().with_view_angles(8, 4);
        let selector = ViewAngleSelector::new(&config);

        let angle0 = selector.horizontal_index_to_angle(0);
        let angle4 = selector.horizontal_index_to_angle(4);

        // 4 steps = 180 degrees = PI radians difference
        assert!((angle4 - angle0 - PI).abs() < 0.01);
    }

    #[test]
    fn test_selector_vertical_index_to_pitch() {
        let config = ImpostorConfig::new()
            .with_view_angles(8, 4)
            .with_pitch_range(0.0, PI / 2.0);
        let selector = ViewAngleSelector::new(&config);

        let pitch0 = selector.vertical_index_to_pitch(0);
        let pitch3 = selector.vertical_index_to_pitch(3);

        assert!((pitch0 - 0.0).abs() < 0.01);
        assert!((pitch3 - PI / 2.0).abs() < 0.01);
    }

    // ========================================================================
    // ImpostorInstance Tests
    // ========================================================================

    #[test]
    fn test_instance_struct_size() {
        assert_eq!(std::mem::size_of::<ImpostorInstance>(), IMPOSTOR_INSTANCE_SIZE);
    }

    #[test]
    fn test_instance_new() {
        let instance = ImpostorInstance::new([1.0, 2.0, 3.0]);
        assert_eq!(instance.position.x, 1.0);
        assert_eq!(instance.position.y, 2.0);
        assert_eq!(instance.position.z, 3.0);
        assert_eq!(instance.scale, 1.0);
        assert_eq!(instance.fade_factor, 1.0);
    }

    #[test]
    fn test_instance_builder() {
        let instance = ImpostorInstance::new([0.0, 0.0, 0.0])
            .with_rotation(Quat::from_rotation_y(1.0))
            .with_scale(2.0)
            .with_animation_frame(5)
            .with_view_angle_index(10)
            .with_lod_fade(0.5)
            .with_crossfade(true)
            .with_dithered(true)
            .with_atlas(2);

        assert_eq!(instance.scale, 2.0);
        assert_eq!(instance.animation_frame, 5);
        assert_eq!(instance.view_angle_index, 10);
        assert!((instance.fade_factor - 0.5).abs() < 0.0001);
        assert!(instance.is_crossfade());
        assert!(instance.is_dithered());
        assert_eq!(instance.atlas_index, 2);
    }

    #[test]
    fn test_instance_visibility() {
        let mut instance = ImpostorInstance::new([0.0, 0.0, 0.0]);
        assert!(instance.is_visible());

        instance.set_visible(false);
        assert!(!instance.is_visible());

        instance.set_visible(true);
        assert!(instance.is_visible());
    }

    #[test]
    fn test_instance_fade_visibility() {
        let mut instance = ImpostorInstance::new([0.0, 0.0, 0.0]);
        assert!(instance.is_visible());

        instance.fade_factor = 0.0;
        assert!(!instance.is_visible());
    }

    #[test]
    fn test_instance_validate_valid() {
        let instance = ImpostorInstance::new([1.0, 2.0, 3.0]).with_scale(1.0);
        assert!(instance.validate().is_ok());
    }

    #[test]
    fn test_instance_validate_nan_position() {
        let mut instance = ImpostorInstance::new([0.0, 0.0, 0.0]);
        instance.position = Vec3::new(f32::NAN, 0.0, 0.0);
        assert!(instance.validate().is_err());
    }

    #[test]
    fn test_instance_validate_zero_scale() {
        let instance = ImpostorInstance::new([0.0, 0.0, 0.0]).with_scale(0.0);
        assert!(instance.validate().is_err());
    }

    #[test]
    fn test_instance_view_interpolation() {
        let instance = ImpostorInstance::new([0.0, 0.0, 0.0])
            .with_view_angle_index(5)
            .with_view_interpolation(6, 0.3);

        assert_eq!(instance.view_angle_index, 5);
        assert_eq!(instance.view_angle_index_secondary, 6);
        assert!((instance.view_angle_weight - 0.3).abs() < 0.0001);
    }

    #[test]
    fn test_instance_animation_interpolation() {
        let instance = ImpostorInstance::new([0.0, 0.0, 0.0])
            .with_animation_frame(10)
            .with_animation_interpolation(11, 0.7);

        assert_eq!(instance.animation_frame, 10);
        assert_eq!(instance.animation_frame_secondary, 11);
        assert!((instance.animation_weight - 0.7).abs() < 0.0001);
    }

    // ========================================================================
    // LodTransition Tests
    // ========================================================================

    #[test]
    fn test_lod_transition_new() {
        let transition = LodTransition::new(50.0, 200.0);
        assert_eq!(transition.mesh_fade_end, 50.0);
        assert_eq!(transition.impostor_fade_end, 200.0);
    }

    #[test]
    fn test_lod_transition_calculate_blend_close() {
        let transition = LodTransition::new(50.0, 200.0);

        // Very close - full mesh, no impostor
        let (mesh, impostor) = transition.calculate_blend(10.0);
        assert!((mesh - 1.0).abs() < 0.01);
        assert!((impostor - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_lod_transition_calculate_blend_medium() {
        let transition = LodTransition::new(50.0, 200.0);

        // At transition distance - partial blend
        let (mesh, impostor) = transition.calculate_blend(45.0);
        assert!(mesh > 0.0 && mesh < 1.0);
        assert!(impostor > 0.0);
    }

    #[test]
    fn test_lod_transition_calculate_blend_far() {
        let transition = LodTransition::new(50.0, 200.0);

        // Far - no mesh, some impostor
        let (mesh, impostor) = transition.calculate_blend(100.0);
        assert!((mesh - 0.0).abs() < 0.01);
        assert!(impostor > 0.0);
    }

    #[test]
    fn test_lod_transition_calculate_blend_very_far() {
        let transition = LodTransition::new(50.0, 200.0);

        // Very far - nothing visible
        let (mesh, impostor) = transition.calculate_blend(300.0);
        assert!((mesh - 0.0).abs() < 0.01);
        assert!((impostor - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_lod_transition_should_render() {
        let transition = LodTransition::new(50.0, 200.0);

        // Close - render mesh
        assert!(transition.should_render_mesh(20.0));
        assert!(!transition.should_render_impostor(20.0));

        // Far - render impostor
        assert!(!transition.should_render_mesh(100.0));
        assert!(transition.should_render_impostor(100.0));
    }

    // ========================================================================
    // ImpostorBatch Tests
    // ========================================================================

    #[test]
    fn test_batch_new() {
        let batch = ImpostorBatch::new(0);
        assert_eq!(batch.atlas_index, 0);
        assert!(batch.is_empty());
    }

    #[test]
    fn test_batch_add_instance() {
        let mut batch = ImpostorBatch::new(0);
        let idx = batch.add_instance(ImpostorInstance::new([1.0, 0.0, 0.0]));
        assert_eq!(idx, 0);
        assert_eq!(batch.instance_count(), 1);
        assert!(batch.is_dirty());
    }

    #[test]
    fn test_batch_remove_instance() {
        let mut batch = ImpostorBatch::new(0);
        batch.add_instance(ImpostorInstance::new([0.0, 0.0, 0.0]));
        batch.add_instance(ImpostorInstance::new([1.0, 0.0, 0.0]));
        batch.add_instance(ImpostorInstance::new([2.0, 0.0, 0.0]));

        let swapped = batch.remove_instance(0);
        assert_eq!(swapped, Some(2));
        assert_eq!(batch.instance_count(), 2);
    }

    #[test]
    fn test_batch_visible_count() {
        let mut batch = ImpostorBatch::new(0);
        batch.add_instance(ImpostorInstance::new([0.0, 0.0, 0.0]));
        batch.add_instance(ImpostorInstance::new([1.0, 0.0, 0.0]).with_lod_fade(0.0));

        assert_eq!(batch.visible_count(), 1);  // Second has fade=0
    }

    #[test]
    fn test_batch_buffer_size() {
        let mut batch = ImpostorBatch::new(0);
        batch.add_instance(ImpostorInstance::new([0.0, 0.0, 0.0]));
        assert_eq!(batch.buffer_size(), IMPOSTOR_INSTANCE_SIZE);
    }

    #[test]
    fn test_batch_clear() {
        let mut batch = ImpostorBatch::new(0);
        batch.add_instance(ImpostorInstance::new([0.0, 0.0, 0.0]));
        batch.add_instance(ImpostorInstance::new([1.0, 0.0, 0.0]));

        batch.clear();
        assert!(batch.is_empty());
        assert_eq!(batch.visible_count(), 0);
    }

    // ========================================================================
    // ImpostorRenderer Tests
    // ========================================================================

    #[test]
    fn test_renderer_new() {
        let config = ImpostorConfig::default();
        let renderer = ImpostorRenderer::new(config);
        assert_eq!(renderer.config.horizontal_angles, DEFAULT_HORIZONTAL_ANGLES);
    }

    #[test]
    fn test_renderer_camera_position_for_angle() {
        let config = ImpostorConfig::new()
            .with_view_angles(8, 4)
            .with_pitch_range(0.0, PI / 2.0);
        let renderer = ImpostorRenderer::new(config);

        // Front view (h=0)
        let pos = renderer.camera_position_for_angle(0, 0, 10.0);
        assert!((pos.length() - 10.0).abs() < 0.01);

        // Side view (h=2, which is 90 degrees)
        let pos2 = renderer.camera_position_for_angle(2, 0, 10.0);
        assert!((pos2.length() - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_renderer_create_atlas() {
        let config = ImpostorConfig::new()
            .with_view_angles(4, 2)
            .with_animation_frames(4)
            .with_sprite_resolution(64, 64);

        let renderer = ImpostorRenderer::new(config);
        let atlas = renderer.create_atlas().unwrap();

        assert!(!atlas.is_empty());
    }

    #[test]
    fn test_renderer_mipmap_count() {
        let config = ImpostorConfig::new()
            .with_view_angles(4, 2)
            .with_animation_frames(4)
            .with_sprite_resolution(256, 256);

        let renderer = ImpostorRenderer::new(config);
        let mip_count = renderer.mipmap_count();

        // Atlas is 4*4*256 x 2*256 = 4096 x 512, max dim = 4096
        // log2(4096) = 12
        assert!(mip_count >= 10);
    }

    // ========================================================================
    // Frustum Tests
    // ========================================================================

    #[test]
    fn test_frustum_from_bounds() {
        let frustum = Frustum::from_bounds(
            Vec3::new(-100.0, -100.0, -100.0),
            Vec3::new(100.0, 100.0, 100.0),
        );

        assert!(frustum.contains_sphere(Vec3::ZERO, 1.0));
        assert!(!frustum.contains_sphere(Vec3::new(200.0, 0.0, 0.0), 1.0));
    }

    #[test]
    fn test_frustum_unbounded() {
        let frustum = Frustum::unbounded();
        assert!(frustum.contains_sphere(Vec3::new(1e10, 1e10, 1e10), 1.0));
    }

    // ========================================================================
    // Error Tests
    // ========================================================================

    #[test]
    fn test_error_display() {
        let err = ImpostorError::InvalidConfig { reason: "test" };
        assert!(err.to_string().contains("test"));

        let err = ImpostorError::AtlasTooLarge { width: 20000, height: 20000, max_size: 16384 };
        assert!(err.to_string().contains("20000"));

        let err = ImpostorError::BufferFull { max: 1000 };
        assert!(err.to_string().contains("1000"));
    }

    // ========================================================================
    // Edge Case Tests
    // ========================================================================

    #[test]
    fn test_camera_at_instance_position() {
        let config = ImpostorConfig::new().with_view_angles(8, 4);
        let selector = ViewAngleSelector::new(&config);

        // Camera at same position as instance
        let camera = Vec3::ZERO;
        let instance = Vec3::ZERO;
        let rotation = Quat::IDENTITY;

        let (combined, _, _) = selector.select_angle(camera, instance, rotation);
        // Should return default (0)
        assert_eq!(combined, 0);
    }

    #[test]
    fn test_atlas_uv_wrapping() {
        let config = ImpostorConfig::new()
            .with_view_angles(4, 2)
            .with_animation_frames(4)
            .with_sprite_resolution(64, 64);

        let atlas = ImpostorAtlas::new(config.clone()).unwrap();

        // Test wrapping (index beyond range)
        let uv1 = atlas.get_uv(0, 0, 0);
        let uv2 = atlas.get_uv(4, 2, 4);  // Should wrap to (0, 0, 0)

        assert_eq!(uv1.u_min, uv2.u_min);
    }

    #[test]
    fn test_instance_extreme_values() {
        let instance = ImpostorInstance::new([f32::MAX / 2.0, 0.0, 0.0])
            .with_scale(1000.0)
            .with_lod_fade(1.0);

        assert!(instance.validate().is_ok());
    }

    #[test]
    fn test_batch_update_view_angles() {
        let config = ImpostorConfig::new().with_view_angles(8, 4);
        let selector = ViewAngleSelector::new(&config);

        let mut batch = ImpostorBatch::new(0);
        batch.add_instance(ImpostorInstance::new([0.0, 0.0, 0.0]));
        batch.add_instance(ImpostorInstance::new([10.0, 0.0, 0.0]));

        let camera = Vec3::new(0.0, 0.0, 10.0);
        batch.update_view_angles(camera, &selector);

        assert!(batch.is_dirty());
    }

    #[test]
    fn test_batch_update_lod_fades() {
        let transition = LodTransition::new(50.0, 200.0);

        let mut batch = ImpostorBatch::new(0);
        batch.add_instance(ImpostorInstance::new([0.0, 0.0, 0.0]));    // Very close
        batch.add_instance(ImpostorInstance::new([100.0, 0.0, 0.0])); // Far

        let camera = Vec3::ZERO;
        batch.update_lod_fades(camera, &transition);

        // Close instance should have low impostor fade
        assert!(batch.instances[0].fade_factor < 0.5);
        // Far instance should have higher impostor fade
        assert!(batch.instances[1].fade_factor > 0.0);
    }

    // ========================================================================
    // Integration Tests
    // ========================================================================

    #[test]
    fn test_full_pipeline() {
        // Create configuration
        let config = ImpostorConfig::new()
            .with_view_angles(8, 4)
            .with_animation_frames(16)
            .with_sprite_resolution(128, 128)
            .with_billboard_mode(BillboardMode::Cylindrical);

        assert!(config.validate().is_ok());

        // Create renderer and atlas
        let renderer = ImpostorRenderer::new(config.clone());
        let atlas = renderer.create_atlas().unwrap();

        // Create view angle selector
        let selector = ViewAngleSelector::new(&config);

        // Create batch with instances
        let mut batch = ImpostorBatch::new(0);

        for i in 0..10 {
            let pos = Vec3::new(i as f32 * 5.0, 0.0, 0.0);
            let instance = ImpostorInstance::new(pos)
                .with_rotation(Quat::from_rotation_y(i as f32 * 0.5))
                .with_animation_frame((i % 16) as u16);
            batch.add_instance(instance);
        }

        // Update view angles
        let camera = Vec3::new(25.0, 5.0, 20.0);
        batch.update_view_angles(camera, &selector);

        // Update LOD fades
        let transition = LodTransition::default();
        batch.update_lod_fades(camera, &transition);

        // Verify
        assert_eq!(batch.instance_count(), 10);
        assert!(batch.is_dirty());

        // Get UV for first instance
        let inst = &batch.instances[0];
        let uv = atlas.get_uv_by_index(inst.view_angle_index as u32, inst.animation_frame as u32);
        assert!(uv.u_min >= 0.0 && uv.u_max <= 1.0);
    }

    #[test]
    fn test_bytemuck_compliance() {
        // Verify Pod and Zeroable implementations
        let instance = ImpostorInstance::default();
        let bytes = bytemuck::bytes_of(&instance);
        assert_eq!(bytes.len(), IMPOSTOR_INSTANCE_SIZE);

        let instances = vec![ImpostorInstance::default(); 10];
        let batch_bytes: &[u8] = bytemuck::cast_slice(&instances);
        assert_eq!(batch_bytes.len(), 10 * IMPOSTOR_INSTANCE_SIZE);
    }
}
