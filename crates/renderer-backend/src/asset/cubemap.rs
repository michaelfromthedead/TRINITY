//! Cubemap and Texture Array Assembly (T-AS-2.5)
//!
//! Provides cubemap construction from cross layouts, individual face images, and KTX files,
//! as well as texture array assembly for efficient GPU batching.
//!
//! # Features
//!
//! - Cross layout detection (4:3 horizontal, 3:4 vertical)
//! - Face extraction from cross layout images
//! - 6 individual image assembly with format/size validation
//! - KTX native cubemap parsing (array layers as faces)
//! - GPU cubemap creation with VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT
//! - Seam-aware mip filtering across cubemap edges
//! - Texture array construction (256-2048 layers)
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::asset::cubemap::{
//!     CubemapBuilder, CubemapConfig, CubemapLayout, CubemapFace,
//!     TextureArrayBuilder, TextureArrayConfig,
//! };
//!
//! // Detect and extract from cross layout
//! if let Some(layout) = detect_cubemap_layout(image.width, image.height) {
//!     let faces = extract_faces_from_cross(&image, layout);
//!     let cubemap = assemble_cubemap(faces, &config);
//!     generate_seam_aware_mips(&mut cubemap);
//! }
//!
//! // Create texture array
//! let array = create_texture_array(textures, &array_config);
//! ```

use std::fmt;

use super::mipmap::{FilterType, MipLevel, MipmapConfig, generate_mipmaps};
use super::texture_importer::{GpuTextureFormat, TextureAsset, TextureMetadata, TextureState, SourceFormat};

// ---------------------------------------------------------------------------
// Error Types
// ---------------------------------------------------------------------------

/// Errors that can occur during cubemap or texture array operations.
#[derive(Debug, Clone)]
pub enum CubemapError {
    /// Invalid cubemap layout
    InvalidLayout { width: u32, height: u32 },
    /// Face dimensions don't match
    FaceSizeMismatch { expected: u32, actual: u32, face: CubemapFace },
    /// Face format doesn't match
    FormatMismatch { expected: GpuTextureFormat, actual: GpuTextureFormat, face: CubemapFace },
    /// Invalid face data size
    InvalidDataSize { expected: usize, actual: usize },
    /// Invalid face count (must be 6 for cubemap)
    InvalidFaceCount { count: usize },
    /// Texture array layer count exceeded
    LayerCountExceeded { max: u32, requested: u32 },
    /// Texture array format mismatch
    ArrayFormatMismatch { expected: GpuTextureFormat, actual: GpuTextureFormat, layer: usize },
    /// Texture array dimension mismatch
    ArrayDimensionMismatch { expected: (u32, u32), actual: (u32, u32), layer: usize },
    /// KTX cubemap parsing error
    KtxParseError(String),
    /// Mipmap generation error
    MipmapError(String),
}

impl fmt::Display for CubemapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CubemapError::InvalidLayout { width, height } => {
                write!(f, "invalid cubemap layout: {}x{} (expected 4:3 or 3:4 aspect ratio)", width, height)
            }
            CubemapError::FaceSizeMismatch { expected, actual, face } => {
                write!(f, "face {:?} size mismatch: expected {}x{}, got {}x{}", face, expected, expected, actual, actual)
            }
            CubemapError::FormatMismatch { expected, actual, face } => {
                write!(f, "face {:?} format mismatch: expected {:?}, got {:?}", face, expected, actual)
            }
            CubemapError::InvalidDataSize { expected, actual } => {
                write!(f, "invalid data size: expected {} bytes, got {}", expected, actual)
            }
            CubemapError::InvalidFaceCount { count } => {
                write!(f, "invalid face count: expected 6, got {}", count)
            }
            CubemapError::LayerCountExceeded { max, requested } => {
                write!(f, "layer count {} exceeds maximum {}", requested, max)
            }
            CubemapError::ArrayFormatMismatch { expected, actual, layer } => {
                write!(f, "layer {} format mismatch: expected {:?}, got {:?}", layer, expected, actual)
            }
            CubemapError::ArrayDimensionMismatch { expected, actual, layer } => {
                write!(f, "layer {} dimension mismatch: expected {:?}, got {:?}", layer, expected, actual)
            }
            CubemapError::KtxParseError(msg) => write!(f, "KTX cubemap parse error: {}", msg),
            CubemapError::MipmapError(msg) => write!(f, "mipmap error: {}", msg),
        }
    }
}

impl std::error::Error for CubemapError {}

// ---------------------------------------------------------------------------
// Cubemap Types
// ---------------------------------------------------------------------------

/// Layout format for cubemap source images.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CubemapLayout {
    /// Horizontal cross layout (4:3 aspect ratio)
    /// ```text
    ///     +Y
    /// -X  +Z  +X  -Z
    ///     -Y
    /// ```
    CrossHorizontal,

    /// Vertical cross layout (3:4 aspect ratio)
    /// ```text
    ///     +Y
    /// -X  +Z  +X
    ///     -Y
    ///     -Z
    /// ```
    CrossVertical,

    /// Six separate image files per face
    SixImages,

    /// Native KTX cubemap with array layers
    KtxCubemap,
}

impl CubemapLayout {
    /// Get the expected aspect ratio for this layout.
    pub fn aspect_ratio(&self) -> Option<(u32, u32)> {
        match self {
            CubemapLayout::CrossHorizontal => Some((4, 3)),
            CubemapLayout::CrossVertical => Some((3, 4)),
            CubemapLayout::SixImages => None,
            CubemapLayout::KtxCubemap => None,
        }
    }

    /// Get the name of this layout.
    pub const fn name(&self) -> &'static str {
        match self {
            CubemapLayout::CrossHorizontal => "CrossHorizontal",
            CubemapLayout::CrossVertical => "CrossVertical",
            CubemapLayout::SixImages => "SixImages",
            CubemapLayout::KtxCubemap => "KtxCubemap",
        }
    }
}

impl fmt::Display for CubemapLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Cubemap face identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CubemapFace {
    /// +X face (right)
    PositiveX = 0,
    /// -X face (left)
    NegativeX = 1,
    /// +Y face (top)
    PositiveY = 2,
    /// -Y face (bottom)
    NegativeY = 3,
    /// +Z face (front)
    PositiveZ = 4,
    /// -Z face (back)
    NegativeZ = 5,
}

impl CubemapFace {
    /// All faces in GPU/Vulkan ordering.
    pub const ALL: [CubemapFace; 6] = [
        CubemapFace::PositiveX,
        CubemapFace::NegativeX,
        CubemapFace::PositiveY,
        CubemapFace::NegativeY,
        CubemapFace::PositiveZ,
        CubemapFace::NegativeZ,
    ];

    /// Get face index (0-5).
    pub const fn index(&self) -> usize {
        *self as usize
    }

    /// Get face from index.
    pub fn from_index(index: usize) -> Option<CubemapFace> {
        match index {
            0 => Some(CubemapFace::PositiveX),
            1 => Some(CubemapFace::NegativeX),
            2 => Some(CubemapFace::PositiveY),
            3 => Some(CubemapFace::NegativeY),
            4 => Some(CubemapFace::PositiveZ),
            5 => Some(CubemapFace::NegativeZ),
            _ => None,
        }
    }

    /// Get the face name.
    pub const fn name(&self) -> &'static str {
        match self {
            CubemapFace::PositiveX => "+X",
            CubemapFace::NegativeX => "-X",
            CubemapFace::PositiveY => "+Y",
            CubemapFace::NegativeY => "-Y",
            CubemapFace::PositiveZ => "+Z",
            CubemapFace::NegativeZ => "-Z",
        }
    }

    /// Get the direction vector for this face.
    pub const fn direction(&self) -> [f32; 3] {
        match self {
            CubemapFace::PositiveX => [1.0, 0.0, 0.0],
            CubemapFace::NegativeX => [-1.0, 0.0, 0.0],
            CubemapFace::PositiveY => [0.0, 1.0, 0.0],
            CubemapFace::NegativeY => [0.0, -1.0, 0.0],
            CubemapFace::PositiveZ => [0.0, 0.0, 1.0],
            CubemapFace::NegativeZ => [0.0, 0.0, -1.0],
        }
    }

    /// Get adjacent faces and their edge directions for seam filtering.
    pub const fn adjacent_faces(&self) -> [(CubemapFace, EdgeDirection); 4] {
        match self {
            CubemapFace::PositiveX => [
                (CubemapFace::PositiveZ, EdgeDirection::Left),
                (CubemapFace::NegativeZ, EdgeDirection::Right),
                (CubemapFace::PositiveY, EdgeDirection::Bottom),
                (CubemapFace::NegativeY, EdgeDirection::Top),
            ],
            CubemapFace::NegativeX => [
                (CubemapFace::NegativeZ, EdgeDirection::Left),
                (CubemapFace::PositiveZ, EdgeDirection::Right),
                (CubemapFace::PositiveY, EdgeDirection::Top),
                (CubemapFace::NegativeY, EdgeDirection::Bottom),
            ],
            CubemapFace::PositiveY => [
                (CubemapFace::NegativeX, EdgeDirection::Left),
                (CubemapFace::PositiveX, EdgeDirection::Right),
                (CubemapFace::NegativeZ, EdgeDirection::Top),
                (CubemapFace::PositiveZ, EdgeDirection::Bottom),
            ],
            CubemapFace::NegativeY => [
                (CubemapFace::NegativeX, EdgeDirection::Left),
                (CubemapFace::PositiveX, EdgeDirection::Right),
                (CubemapFace::PositiveZ, EdgeDirection::Top),
                (CubemapFace::NegativeZ, EdgeDirection::Bottom),
            ],
            CubemapFace::PositiveZ => [
                (CubemapFace::NegativeX, EdgeDirection::Left),
                (CubemapFace::PositiveX, EdgeDirection::Right),
                (CubemapFace::PositiveY, EdgeDirection::Top),
                (CubemapFace::NegativeY, EdgeDirection::Bottom),
            ],
            CubemapFace::NegativeZ => [
                (CubemapFace::PositiveX, EdgeDirection::Left),
                (CubemapFace::NegativeX, EdgeDirection::Right),
                (CubemapFace::PositiveY, EdgeDirection::Bottom),
                (CubemapFace::NegativeY, EdgeDirection::Top),
            ],
        }
    }
}

impl fmt::Display for CubemapFace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Edge direction for seam-aware filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeDirection {
    Left,
    Right,
    Top,
    Bottom,
}

impl EdgeDirection {
    /// Get the opposite edge direction.
    pub const fn opposite(&self) -> EdgeDirection {
        match self {
            EdgeDirection::Left => EdgeDirection::Right,
            EdgeDirection::Right => EdgeDirection::Left,
            EdgeDirection::Top => EdgeDirection::Bottom,
            EdgeDirection::Bottom => EdgeDirection::Top,
        }
    }
}

// ---------------------------------------------------------------------------
// Texture Data
// ---------------------------------------------------------------------------

/// Raw texture data for cubemap face or array layer.
#[derive(Debug, Clone)]
pub struct TextureData {
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
    /// Pixel format
    pub format: GpuTextureFormat,
    /// Raw pixel data
    pub data: Vec<u8>,
    /// Is this sRGB data?
    pub is_srgb: bool,
}

impl TextureData {
    /// Create new texture data.
    pub fn new(width: u32, height: u32, format: GpuTextureFormat, data: Vec<u8>, is_srgb: bool) -> Self {
        Self {
            width,
            height,
            format,
            data,
            is_srgb,
        }
    }

    /// Check if data is valid.
    pub fn is_valid(&self) -> bool {
        let expected = self.width as usize * self.height as usize * self.format.bytes_per_pixel();
        self.data.len() == expected
    }

    /// Get pixel at (x, y).
    pub fn get_pixel(&self, x: u32, y: u32) -> [u8; 4] {
        if x >= self.width || y >= self.height {
            return [0, 0, 0, 255];
        }

        let bpp = self.format.bytes_per_pixel();
        let idx = (y as usize * self.width as usize + x as usize) * bpp;

        if idx + bpp > self.data.len() {
            return [0, 0, 0, 255];
        }

        match bpp {
            1 => [self.data[idx], self.data[idx], self.data[idx], 255],
            2 => [self.data[idx], self.data[idx], self.data[idx], self.data[idx + 1]],
            3 => [self.data[idx], self.data[idx + 1], self.data[idx + 2], 255],
            4 => [
                self.data[idx],
                self.data[idx + 1],
                self.data[idx + 2],
                self.data[idx + 3],
            ],
            _ => [0, 0, 0, 255],
        }
    }

    /// Set pixel at (x, y).
    pub fn set_pixel(&mut self, x: u32, y: u32, rgba: [u8; 4]) {
        if x >= self.width || y >= self.height {
            return;
        }

        let bpp = self.format.bytes_per_pixel();
        let idx = (y as usize * self.width as usize + x as usize) * bpp;

        if idx + bpp > self.data.len() {
            return;
        }

        match bpp {
            1 => self.data[idx] = rgba[0],
            2 => {
                self.data[idx] = rgba[0];
                self.data[idx + 1] = rgba[3];
            }
            3 => {
                self.data[idx] = rgba[0];
                self.data[idx + 1] = rgba[1];
                self.data[idx + 2] = rgba[2];
            }
            4 => {
                self.data[idx] = rgba[0];
                self.data[idx + 1] = rgba[1];
                self.data[idx + 2] = rgba[2];
                self.data[idx + 3] = rgba[3];
            }
            _ => {}
        }
    }

    /// Create from TextureAsset.
    pub fn from_asset(asset: &TextureAsset) -> Self {
        Self {
            width: asset.metadata.width,
            height: asset.metadata.height,
            format: asset.metadata.format,
            data: asset.data.clone(),
            is_srgb: asset.metadata.is_srgb,
        }
    }

    /// Convert to TextureAsset.
    pub fn to_asset(&self, id: u64) -> TextureAsset {
        TextureAsset {
            id,
            metadata: TextureMetadata {
                width: self.width,
                height: self.height,
                format: self.format,
                memory_size: self.data.len(),
                is_srgb: self.is_srgb,
                source_format: SourceFormat::Png, // Default
                source_bit_depth: 8,
                source_channels: self.format.channel_count() as u8,
            },
            data: self.data.clone(),
            state: TextureState::Pending,
        }
    }
}

// ---------------------------------------------------------------------------
// Cubemap Configuration
// ---------------------------------------------------------------------------

/// Configuration for cubemap assembly.
#[derive(Debug, Clone)]
pub struct CubemapConfig {
    /// Source layout
    pub layout: CubemapLayout,
    /// Generate seam-aware mipmaps
    pub seam_aware_mips: bool,
    /// Face size (width = height for square faces)
    pub face_size: u32,
    /// Filter type for mipmap generation
    pub mip_filter: FilterType,
    /// Generate mipmaps
    pub generate_mips: bool,
    /// Number of mip levels (0 = auto)
    pub mip_levels: u32,
}

impl Default for CubemapConfig {
    fn default() -> Self {
        Self {
            layout: CubemapLayout::SixImages,
            seam_aware_mips: true,
            face_size: 0, // Auto-detect
            mip_filter: FilterType::Lanczos,
            generate_mips: true,
            mip_levels: 0,
        }
    }
}

impl CubemapConfig {
    /// Create config for horizontal cross layout.
    pub fn horizontal_cross(face_size: u32) -> Self {
        Self {
            layout: CubemapLayout::CrossHorizontal,
            face_size,
            ..Default::default()
        }
    }

    /// Create config for vertical cross layout.
    pub fn vertical_cross(face_size: u32) -> Self {
        Self {
            layout: CubemapLayout::CrossVertical,
            face_size,
            ..Default::default()
        }
    }

    /// Create config for six separate images.
    pub fn six_images(face_size: u32) -> Self {
        Self {
            layout: CubemapLayout::SixImages,
            face_size,
            ..Default::default()
        }
    }

    /// Create config for KTX cubemap.
    pub fn ktx_cubemap(face_size: u32) -> Self {
        Self {
            layout: CubemapLayout::KtxCubemap,
            face_size,
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Cubemap Face Data
// ---------------------------------------------------------------------------

/// A single cubemap face with its data.
#[derive(Debug, Clone)]
pub struct CubemapFaceData {
    /// Which face this is
    pub face: CubemapFace,
    /// Face texture data
    pub data: TextureData,
    /// Mip levels (if generated)
    pub mip_levels: Vec<TextureData>,
}

impl CubemapFaceData {
    /// Create new face data.
    pub fn new(face: CubemapFace, data: TextureData) -> Self {
        Self {
            face,
            data,
            mip_levels: Vec::new(),
        }
    }

    /// Get face size.
    pub fn size(&self) -> u32 {
        self.data.width
    }

    /// Get mip count.
    pub fn mip_count(&self) -> usize {
        if self.mip_levels.is_empty() {
            1
        } else {
            self.mip_levels.len()
        }
    }
}

// ---------------------------------------------------------------------------
// Cubemap Texture
// ---------------------------------------------------------------------------

/// A complete cubemap texture with all 6 faces.
#[derive(Debug, Clone)]
pub struct CubemapTexture {
    /// Face size (width = height)
    pub face_size: u32,
    /// Pixel format
    pub format: GpuTextureFormat,
    /// Is sRGB?
    pub is_srgb: bool,
    /// The 6 faces in GPU order (+X, -X, +Y, -Y, +Z, -Z)
    pub faces: [CubemapFaceData; 6],
    /// Number of mip levels
    pub mip_count: u32,
    /// Was seam-aware filtering applied?
    pub seam_aware: bool,
    /// Total memory size
    pub memory_size: usize,
    /// GPU creation flags (includes VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT)
    pub gpu_flags: u32,
}

/// VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT value
pub const VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT: u32 = 0x00000010;

impl CubemapTexture {
    /// Get a face by index.
    pub fn get_face(&self, face: CubemapFace) -> &CubemapFaceData {
        &self.faces[face.index()]
    }

    /// Get mutable face by index.
    pub fn get_face_mut(&mut self, face: CubemapFace) -> &mut CubemapFaceData {
        &mut self.faces[face.index()]
    }

    /// Check if cubemap is valid.
    pub fn is_valid(&self) -> bool {
        // All faces must have same size
        self.faces.iter().all(|f| f.data.width == self.face_size && f.data.height == self.face_size)
    }

    /// Get layer count (always 6 for cubemap).
    pub const fn layer_count(&self) -> u32 {
        6
    }
}

// ---------------------------------------------------------------------------
// Layout Detection
// ---------------------------------------------------------------------------

/// Detect cubemap layout from image dimensions.
///
/// Returns `Some(CrossHorizontal)` for 4:3 aspect ratio,
/// `Some(CrossVertical)` for 3:4 aspect ratio,
/// `None` if neither matches.
pub fn detect_cubemap_layout(width: u32, height: u32) -> Option<CubemapLayout> {
    if width == 0 || height == 0 {
        return None;
    }

    // Check for 4:3 horizontal cross
    // Total size is 4*face_size x 3*face_size
    if width * 3 == height * 4 && width % 4 == 0 {
        return Some(CubemapLayout::CrossHorizontal);
    }

    // Check for 3:4 vertical cross
    // Total size is 3*face_size x 4*face_size
    if width * 4 == height * 3 && width % 3 == 0 {
        return Some(CubemapLayout::CrossVertical);
    }

    None
}

/// Get face positions within a cross layout image.
///
/// Returns (x, y) coordinates for the top-left corner of each face.
fn get_cross_face_positions(layout: CubemapLayout, face_size: u32) -> [(u32, u32); 6] {
    match layout {
        CubemapLayout::CrossHorizontal => {
            // Horizontal cross:
            //     +Y
            // -X  +Z  +X  -Z
            //     -Y
            [
                (face_size * 2, face_size),     // +X (right of center)
                (0, face_size),                  // -X (left)
                (face_size, 0),                  // +Y (top)
                (face_size, face_size * 2),      // -Y (bottom)
                (face_size, face_size),          // +Z (center)
                (face_size * 3, face_size),      // -Z (far right)
            ]
        }
        CubemapLayout::CrossVertical => {
            // Vertical cross:
            //     +Y
            // -X  +Z  +X
            //     -Y
            //     -Z
            [
                (face_size * 2, face_size),      // +X (right)
                (0, face_size),                  // -X (left)
                (face_size, 0),                  // +Y (top)
                (face_size, face_size * 2),      // -Y (middle-bottom)
                (face_size, face_size),          // +Z (center)
                (face_size, face_size * 3),      // -Z (bottom)
            ]
        }
        _ => [(0, 0); 6], // Not applicable for other layouts
    }
}

// ---------------------------------------------------------------------------
// Face Extraction
// ---------------------------------------------------------------------------

/// Extract all 6 faces from a cross layout image.
pub fn extract_faces_from_cross(image: &TextureData, layout: CubemapLayout) -> Result<[TextureData; 6], CubemapError> {
    let face_size = match layout {
        CubemapLayout::CrossHorizontal => image.width / 4,
        CubemapLayout::CrossVertical => image.width / 3,
        _ => return Err(CubemapError::InvalidLayout {
            width: image.width,
            height: image.height,
        }),
    };

    if face_size == 0 {
        return Err(CubemapError::InvalidLayout {
            width: image.width,
            height: image.height,
        });
    }

    let positions = get_cross_face_positions(layout, face_size);
    let bpp = image.format.bytes_per_pixel();

    let mut faces: [TextureData; 6] = std::array::from_fn(|_| TextureData {
        width: face_size,
        height: face_size,
        format: image.format,
        data: vec![0u8; (face_size as usize) * (face_size as usize) * bpp],
        is_srgb: image.is_srgb,
    });

    for (i, (sx, sy)) in positions.iter().enumerate() {
        let face = &mut faces[i];

        for y in 0..face_size {
            for x in 0..face_size {
                let pixel = image.get_pixel(sx + x, sy + y);
                face.set_pixel(x, y, pixel);
            }
        }
    }

    Ok(faces)
}

/// Extract a single face from a cross layout image.
pub fn extract_single_face(
    image: &TextureData,
    layout: CubemapLayout,
    face: CubemapFace,
) -> Result<TextureData, CubemapError> {
    let face_size = match layout {
        CubemapLayout::CrossHorizontal => image.width / 4,
        CubemapLayout::CrossVertical => image.width / 3,
        _ => return Err(CubemapError::InvalidLayout {
            width: image.width,
            height: image.height,
        }),
    };

    let positions = get_cross_face_positions(layout, face_size);
    let (sx, sy) = positions[face.index()];
    let bpp = image.format.bytes_per_pixel();

    let mut face_data = TextureData {
        width: face_size,
        height: face_size,
        format: image.format,
        data: vec![0u8; (face_size as usize) * (face_size as usize) * bpp],
        is_srgb: image.is_srgb,
    };

    for y in 0..face_size {
        for x in 0..face_size {
            let pixel = image.get_pixel(sx + x, sy + y);
            face_data.set_pixel(x, y, pixel);
        }
    }

    Ok(face_data)
}

// ---------------------------------------------------------------------------
// Cubemap Assembly
// ---------------------------------------------------------------------------

/// Validate that all faces have matching format and size.
fn validate_faces(faces: &[TextureData; 6]) -> Result<(u32, GpuTextureFormat, bool), CubemapError> {
    let first = &faces[0];
    let face_size = first.width;
    let format = first.format;
    let is_srgb = first.is_srgb;

    // Check first face is square
    if first.width != first.height {
        return Err(CubemapError::FaceSizeMismatch {
            expected: first.width,
            actual: first.height,
            face: CubemapFace::PositiveX,
        });
    }

    // Check all faces match
    for (i, face_data) in faces.iter().enumerate().skip(1) {
        let face = CubemapFace::from_index(i).unwrap();

        if face_data.width != face_size || face_data.height != face_size {
            return Err(CubemapError::FaceSizeMismatch {
                expected: face_size,
                actual: face_data.width,
                face,
            });
        }

        if face_data.format != format {
            return Err(CubemapError::FormatMismatch {
                expected: format,
                actual: face_data.format,
                face,
            });
        }
    }

    Ok((face_size, format, is_srgb))
}

/// Assemble a cubemap from 6 face images.
pub fn assemble_cubemap(
    faces: [TextureData; 6],
    config: &CubemapConfig,
) -> Result<CubemapTexture, CubemapError> {
    let (face_size, format, is_srgb) = validate_faces(&faces)?;

    // Calculate memory size
    let face_memory = face_size as usize * face_size as usize * format.bytes_per_pixel();
    let memory_size = face_memory * 6;

    // Create face data
    let face_data: [CubemapFaceData; 6] = std::array::from_fn(|i| {
        CubemapFaceData::new(CubemapFace::from_index(i).unwrap(), faces[i].clone())
    });

    Ok(CubemapTexture {
        face_size,
        format,
        is_srgb,
        faces: face_data,
        mip_count: 1,
        seam_aware: false,
        memory_size,
        gpu_flags: VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT,
    })
}

/// Reorder faces from one convention to GPU convention if needed.
///
/// Different sources may use different face orderings. This function
/// converts from common orderings to the Vulkan/GPU standard:
/// +X, -X, +Y, -Y, +Z, -Z
pub fn reorder_faces_to_gpu(
    faces: [TextureData; 6],
    source_order: [CubemapFace; 6],
) -> [TextureData; 6] {
    let mut reordered: [Option<TextureData>; 6] = std::array::from_fn(|_| None);

    for (i, face_data) in faces.into_iter().enumerate() {
        let target_face = source_order[i];
        reordered[target_face.index()] = Some(face_data);
    }

    std::array::from_fn(|i| reordered[i].take().unwrap())
}

// ---------------------------------------------------------------------------
// Seam-Aware Mip Generation
// ---------------------------------------------------------------------------

/// Generate seam-aware mipmaps for a cubemap.
///
/// Standard mipmap filtering can cause visible seams at cubemap face boundaries.
/// This function samples across face edges during downsampling to eliminate seams.
pub fn generate_seam_aware_mips(cubemap: &mut CubemapTexture) -> Result<(), CubemapError> {
    let face_size = cubemap.face_size;
    let format = cubemap.format;
    let bpp = format.bytes_per_pixel();

    // Calculate mip levels
    let mip_count = (face_size as f32).log2().floor() as u32 + 1;
    cubemap.mip_count = mip_count;

    // Generate mips for each face
    for face_idx in 0..6 {
        let mut mip_levels = Vec::with_capacity(mip_count as usize);

        // Level 0 is the original
        mip_levels.push(cubemap.faces[face_idx].data.clone());

        let mut current_size = face_size;

        for level in 1..mip_count {
            let new_size = (current_size / 2).max(1);
            let mut mip_data = TextureData {
                width: new_size,
                height: new_size,
                format,
                data: vec![0u8; (new_size as usize) * (new_size as usize) * bpp],
                is_srgb: cubemap.is_srgb,
            };

            // Downsample with seam-aware sampling
            let prev = &mip_levels[level as usize - 1];
            downsample_face_seam_aware(
                &mut mip_data,
                prev,
                CubemapFace::from_index(face_idx).unwrap(),
                &cubemap.faces,
            );

            mip_levels.push(mip_data);
            current_size = new_size;
        }

        cubemap.faces[face_idx].mip_levels = mip_levels;
    }

    // Update memory size
    let mut total_size = 0;
    for face in &cubemap.faces {
        for mip in &face.mip_levels {
            total_size += mip.data.len();
        }
    }
    cubemap.memory_size = total_size;
    cubemap.seam_aware = true;

    Ok(())
}

/// Downsample a face with seam-aware sampling from adjacent faces.
fn downsample_face_seam_aware(
    output: &mut TextureData,
    source: &TextureData,
    face: CubemapFace,
    all_faces: &[CubemapFaceData; 6],
) {
    let out_size = output.width;
    let src_size = source.width;
    let scale = src_size as f32 / out_size as f32;

    for oy in 0..out_size {
        for ox in 0..out_size {
            // Map to source coordinates
            let sx = (ox as f32 + 0.5) * scale;
            let sy = (oy as f32 + 0.5) * scale;

            // Bilinear sample with edge handling
            let pixel = sample_with_seam_handling(
                sx, sy, source, face, all_faces,
            );

            output.set_pixel(ox, oy, pixel);
        }
    }
}

/// Sample a pixel with seam handling for edge pixels.
fn sample_with_seam_handling(
    x: f32,
    y: f32,
    face_data: &TextureData,
    face: CubemapFace,
    all_faces: &[CubemapFaceData; 6],
) -> [u8; 4] {
    let size = face_data.width as f32;
    let x0 = (x - 0.5).floor() as i32;
    let y0 = (y - 0.5).floor() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let fx = x - 0.5 - x0 as f32;
    let fy = y - 0.5 - y0 as f32;

    // Get samples, crossing seams if necessary
    let s00 = get_sample_with_seam(x0, y0, face_data, face, all_faces);
    let s10 = get_sample_with_seam(x1, y0, face_data, face, all_faces);
    let s01 = get_sample_with_seam(x0, y1, face_data, face, all_faces);
    let s11 = get_sample_with_seam(x1, y1, face_data, face, all_faces);

    // Bilinear interpolation
    let lerp = |a: u8, b: u8, t: f32| -> u8 {
        ((a as f32 * (1.0 - t) + b as f32 * t) + 0.5) as u8
    };

    let top = [
        lerp(s00[0], s10[0], fx),
        lerp(s00[1], s10[1], fx),
        lerp(s00[2], s10[2], fx),
        lerp(s00[3], s10[3], fx),
    ];

    let bottom = [
        lerp(s01[0], s11[0], fx),
        lerp(s01[1], s11[1], fx),
        lerp(s01[2], s11[2], fx),
        lerp(s01[3], s11[3], fx),
    ];

    [
        lerp(top[0], bottom[0], fy),
        lerp(top[1], bottom[1], fy),
        lerp(top[2], bottom[2], fy),
        lerp(top[3], bottom[3], fy),
    ]
}

/// Get a sample, crossing to adjacent face if coordinates are out of bounds.
fn get_sample_with_seam(
    x: i32,
    y: i32,
    face_data: &TextureData,
    face: CubemapFace,
    all_faces: &[CubemapFaceData; 6],
) -> [u8; 4] {
    let size = face_data.width as i32;

    // If within bounds, sample normally
    if x >= 0 && x < size && y >= 0 && y < size {
        return face_data.get_pixel(x as u32, y as u32);
    }

    // Determine which edge we're crossing and get adjacent face
    let adjacent = face.adjacent_faces();

    let (adj_face, new_x, new_y) = if x < 0 {
        // Crossing left edge
        let (adj, _) = adjacent[0];
        let new_x = size - 1;
        let new_y = y.clamp(0, size - 1);
        (adj, new_x, new_y)
    } else if x >= size {
        // Crossing right edge
        let (adj, _) = adjacent[1];
        let new_x = 0;
        let new_y = y.clamp(0, size - 1);
        (adj, new_x, new_y)
    } else if y < 0 {
        // Crossing top edge
        let (adj, _) = adjacent[2];
        let new_x = x.clamp(0, size - 1);
        let new_y = size - 1;
        (adj, new_x, new_y)
    } else {
        // Crossing bottom edge
        let (adj, _) = adjacent[3];
        let new_x = x.clamp(0, size - 1);
        let new_y = 0;
        (adj, new_x, new_y)
    };

    all_faces[adj_face.index()].data.get_pixel(new_x as u32, new_y as u32)
}

// ---------------------------------------------------------------------------
// KTX Cubemap Parsing
// ---------------------------------------------------------------------------

/// Parse a KTX file with cubemap array layers.
///
/// KTX files can store cubemaps as a 6-layer texture array.
/// This function extracts the faces and assembles them into a CubemapTexture.
pub fn parse_ktx_cubemap(
    ktx_data: &[u8],
    config: &CubemapConfig,
) -> Result<CubemapTexture, CubemapError> {
    // KTX identifier
    const KTX_IDENTIFIER: [u8; 12] = [
        0xAB, 0x4B, 0x54, 0x58, 0x20, 0x31, 0x31, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A
    ];

    if ktx_data.len() < 64 {
        return Err(CubemapError::KtxParseError("file too small".to_string()));
    }

    // Check identifier
    if &ktx_data[0..12] != &KTX_IDENTIFIER {
        return Err(CubemapError::KtxParseError("invalid KTX identifier".to_string()));
    }

    // Parse header
    let endianness = u32::from_le_bytes([ktx_data[12], ktx_data[13], ktx_data[14], ktx_data[15]]);
    let is_little_endian = endianness == 0x04030201;

    let read_u32 = |offset: usize| -> u32 {
        let bytes = [ktx_data[offset], ktx_data[offset + 1], ktx_data[offset + 2], ktx_data[offset + 3]];
        if is_little_endian {
            u32::from_le_bytes(bytes)
        } else {
            u32::from_be_bytes(bytes)
        }
    };

    let pixel_width = read_u32(36);
    let pixel_height = read_u32(40);
    let number_of_faces = read_u32(52);
    let number_of_mipmap_levels = read_u32(56).max(1);

    if number_of_faces != 6 {
        return Err(CubemapError::InvalidFaceCount { count: number_of_faces as usize });
    }

    if pixel_width != pixel_height {
        return Err(CubemapError::KtxParseError("cubemap faces must be square".to_string()));
    }

    let face_size = pixel_width;
    let format = GpuTextureFormat::R8G8B8A8Unorm; // Simplified - real impl would parse gl_format
    let bpp = format.bytes_per_pixel();

    // Skip key-value data
    let bytes_of_key_value_data = read_u32(60) as usize;
    let mut offset = 64 + bytes_of_key_value_data;

    // Parse mip levels
    let mut faces: [Option<CubemapFaceData>; 6] = std::array::from_fn(|_| None);
    let mut all_mips: [Vec<TextureData>; 6] = std::array::from_fn(|_| Vec::new());

    for mip in 0..number_of_mipmap_levels {
        if offset + 4 > ktx_data.len() {
            break;
        }

        let image_size = read_u32(offset) as usize;
        offset += 4;

        let mip_size = (face_size >> mip).max(1);
        let face_bytes = (mip_size as usize) * (mip_size as usize) * bpp;

        for face_idx in 0..6 {
            if offset + face_bytes > ktx_data.len() {
                return Err(CubemapError::KtxParseError("unexpected end of data".to_string()));
            }

            let face_data = TextureData {
                width: mip_size,
                height: mip_size,
                format,
                data: ktx_data[offset..offset + face_bytes].to_vec(),
                is_srgb: false,
            };

            if mip == 0 {
                faces[face_idx] = Some(CubemapFaceData::new(
                    CubemapFace::from_index(face_idx).unwrap(),
                    face_data.clone(),
                ));
            }

            all_mips[face_idx].push(face_data);
            offset += face_bytes;

            // Padding
            let padding = (4 - (face_bytes % 4)) % 4;
            offset += padding;
        }
    }

    // Build cubemap
    let cubemap_faces: [CubemapFaceData; 6] = std::array::from_fn(|i| {
        let mut face = faces[i].take().unwrap();
        face.mip_levels = std::mem::take(&mut all_mips[i]);
        face
    });

    let mut total_size = 0;
    for face in &cubemap_faces {
        for mip in &face.mip_levels {
            total_size += mip.data.len();
        }
    }

    Ok(CubemapTexture {
        face_size,
        format,
        is_srgb: false,
        faces: cubemap_faces,
        mip_count: number_of_mipmap_levels,
        seam_aware: false,
        memory_size: total_size,
        gpu_flags: VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT,
    })
}

// ---------------------------------------------------------------------------
// Texture Array Types
// ---------------------------------------------------------------------------

/// Configuration for texture array construction.
#[derive(Debug, Clone)]
pub struct TextureArrayConfig {
    /// Required format (all textures must match)
    pub format: GpuTextureFormat,
    /// Required width
    pub width: u32,
    /// Required height
    pub height: u32,
    /// Number of mip levels (0 = no mips)
    pub mip_levels: u32,
    /// Maximum number of layers (256-2048)
    pub max_layers: u32,
}

impl Default for TextureArrayConfig {
    fn default() -> Self {
        Self {
            format: GpuTextureFormat::R8G8B8A8Unorm,
            width: 256,
            height: 256,
            mip_levels: 0,
            max_layers: 256,
        }
    }
}

impl TextureArrayConfig {
    /// Create config with specific dimensions.
    pub fn new(width: u32, height: u32, format: GpuTextureFormat) -> Self {
        Self {
            format,
            width,
            height,
            mip_levels: 0,
            max_layers: 256,
        }
    }

    /// Set maximum layer count.
    pub fn with_max_layers(mut self, max: u32) -> Self {
        self.max_layers = max.clamp(1, 2048);
        self
    }

    /// Enable mipmap generation.
    pub fn with_mips(mut self, levels: u32) -> Self {
        self.mip_levels = levels;
        self
    }
}

// ---------------------------------------------------------------------------
// Texture Array
// ---------------------------------------------------------------------------

/// A texture array containing multiple layers with identical format/size.
#[derive(Debug, Clone)]
pub struct TextureArray {
    /// Width of each layer
    pub width: u32,
    /// Height of each layer
    pub height: u32,
    /// Pixel format
    pub format: GpuTextureFormat,
    /// Layer data
    pub layers: Vec<TextureData>,
    /// Mip levels per layer
    pub mip_levels: Vec<Vec<TextureData>>,
    /// Number of mip levels
    pub mip_count: u32,
    /// Total memory size
    pub memory_size: usize,
}

impl TextureArray {
    /// Get layer count.
    pub fn layer_count(&self) -> u32 {
        self.layers.len() as u32
    }

    /// Get a specific layer.
    pub fn get_layer(&self, index: usize) -> Option<&TextureData> {
        self.layers.get(index)
    }

    /// Get mip level for a layer.
    pub fn get_layer_mip(&self, layer: usize, mip: usize) -> Option<&TextureData> {
        self.mip_levels.get(layer).and_then(|mips| mips.get(mip))
    }

    /// Check if array is valid.
    pub fn is_valid(&self) -> bool {
        self.layers.iter().all(|l| {
            l.width == self.width && l.height == self.height && l.format == self.format
        })
    }
}

// ---------------------------------------------------------------------------
// Texture Array Construction
// ---------------------------------------------------------------------------

/// Create a texture array from multiple textures.
///
/// All textures must have identical format, width, height, and mip count.
/// The array supports 256-2048 layers for efficient GPU batching.
///
/// # WGSL Sampling
/// ```wgsl
/// @group(0) @binding(0) var texture_array: texture_2d_array<f32>;
/// @group(0) @binding(1) var array_sampler: sampler;
///
/// fn sample_layer(uv: vec2<f32>, layer: u32) -> vec4<f32> {
///     return textureSample(texture_array, array_sampler, uv, layer);
/// }
/// ```
pub fn create_texture_array(
    textures: Vec<TextureData>,
    config: &TextureArrayConfig,
) -> Result<TextureArray, CubemapError> {
    if textures.is_empty() {
        return Ok(TextureArray {
            width: config.width,
            height: config.height,
            format: config.format,
            layers: Vec::new(),
            mip_levels: Vec::new(),
            mip_count: 0,
            memory_size: 0,
        });
    }

    if textures.len() > config.max_layers as usize {
        return Err(CubemapError::LayerCountExceeded {
            max: config.max_layers,
            requested: textures.len() as u32,
        });
    }

    // Validate all textures match
    for (i, tex) in textures.iter().enumerate() {
        if tex.width != config.width || tex.height != config.height {
            return Err(CubemapError::ArrayDimensionMismatch {
                expected: (config.width, config.height),
                actual: (tex.width, tex.height),
                layer: i,
            });
        }

        if tex.format != config.format {
            return Err(CubemapError::ArrayFormatMismatch {
                expected: config.format,
                actual: tex.format,
                layer: i,
            });
        }
    }

    // Calculate memory
    let layer_size = (config.width as usize) * (config.height as usize) * config.format.bytes_per_pixel();
    let memory_size = layer_size * textures.len();

    Ok(TextureArray {
        width: config.width,
        height: config.height,
        format: config.format,
        layers: textures,
        mip_levels: Vec::new(),
        mip_count: 1,
        memory_size,
    })
}

/// Add a layer to an existing texture array.
pub fn add_array_layer(
    array: &mut TextureArray,
    texture: TextureData,
    max_layers: u32,
) -> Result<usize, CubemapError> {
    if array.layer_count() >= max_layers {
        return Err(CubemapError::LayerCountExceeded {
            max: max_layers,
            requested: array.layer_count() + 1,
        });
    }

    if texture.width != array.width || texture.height != array.height {
        return Err(CubemapError::ArrayDimensionMismatch {
            expected: (array.width, array.height),
            actual: (texture.width, texture.height),
            layer: array.layers.len(),
        });
    }

    if texture.format != array.format {
        return Err(CubemapError::ArrayFormatMismatch {
            expected: array.format,
            actual: texture.format,
            layer: array.layers.len(),
        });
    }

    let layer_idx = array.layers.len();
    let layer_size = texture.data.len();
    array.layers.push(texture);
    array.memory_size += layer_size;

    Ok(layer_idx)
}

/// Generate mipmaps for all layers in a texture array.
pub fn generate_array_mips(array: &mut TextureArray) -> Result<(), CubemapError> {
    let mip_count = (array.width.max(array.height) as f32).log2().floor() as u32 + 1;

    let mip_config = MipmapConfig {
        filter: FilterType::Lanczos,
        generate_mips: true,
        max_mip_levels: 0,
        srgb_correct: true,
        ..Default::default()
    };

    let mut all_mips = Vec::with_capacity(array.layers.len());
    let mut total_size = 0;

    for layer in &array.layers {
        let asset = layer.to_asset(0);
        let mips = generate_mipmaps(&asset, &mip_config)
            .map_err(|e| CubemapError::MipmapError(e.to_string()))?;

        let layer_mips: Vec<TextureData> = mips.into_iter().map(|m| {
            total_size += m.data.len();
            TextureData {
                width: m.width,
                height: m.height,
                format: array.format,
                data: m.data,
                is_srgb: layer.is_srgb,
            }
        }).collect();

        all_mips.push(layer_mips);
    }

    array.mip_levels = all_mips;
    array.mip_count = mip_count;
    array.memory_size = total_size;

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create test texture data
    fn create_test_texture(width: u32, height: u32) -> TextureData {
        let bpp = 4;
        let data: Vec<u8> = (0..(width * height))
            .flat_map(|i| {
                let x = i % width;
                let y = i / width;
                let r = ((x * 255) / width.max(1)) as u8;
                let g = ((y * 255) / height.max(1)) as u8;
                vec![r, g, 128, 255]
            })
            .collect();

        TextureData {
            width,
            height,
            format: GpuTextureFormat::R8G8B8A8Unorm,
            data,
            is_srgb: false,
        }
    }

    fn create_cross_texture(face_size: u32, horizontal: bool) -> TextureData {
        let (width, height) = if horizontal {
            (face_size * 4, face_size * 3)
        } else {
            (face_size * 3, face_size * 4)
        };
        create_test_texture(width, height)
    }

    // ---------------------------------------------------------------------------
    // Cross Layout Detection Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_detect_horizontal_cross() {
        let layout = detect_cubemap_layout(256, 192); // 4:3 ratio, face_size=64
        assert_eq!(layout, Some(CubemapLayout::CrossHorizontal));
    }

    #[test]
    fn test_detect_vertical_cross() {
        let layout = detect_cubemap_layout(192, 256); // 3:4 ratio, face_size=64
        assert_eq!(layout, Some(CubemapLayout::CrossVertical));
    }

    #[test]
    fn test_detect_invalid_layout() {
        let layout = detect_cubemap_layout(256, 256); // 1:1 ratio
        assert_eq!(layout, None);
    }

    #[test]
    fn test_detect_layout_various_sizes() {
        // Horizontal cross at various sizes
        assert_eq!(detect_cubemap_layout(512, 384), Some(CubemapLayout::CrossHorizontal));
        assert_eq!(detect_cubemap_layout(1024, 768), Some(CubemapLayout::CrossHorizontal));
        assert_eq!(detect_cubemap_layout(2048, 1536), Some(CubemapLayout::CrossHorizontal));

        // Vertical cross at various sizes
        assert_eq!(detect_cubemap_layout(384, 512), Some(CubemapLayout::CrossVertical));
        assert_eq!(detect_cubemap_layout(768, 1024), Some(CubemapLayout::CrossVertical));
    }

    // ---------------------------------------------------------------------------
    // Face Extraction Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_extract_faces_horizontal_cross() {
        let image = create_cross_texture(64, true);
        let faces = extract_faces_from_cross(&image, CubemapLayout::CrossHorizontal).unwrap();

        assert_eq!(faces.len(), 6);
        for face in &faces {
            assert_eq!(face.width, 64);
            assert_eq!(face.height, 64);
            assert!(face.is_valid());
        }
    }

    #[test]
    fn test_extract_faces_vertical_cross() {
        let image = create_cross_texture(64, false);
        let faces = extract_faces_from_cross(&image, CubemapLayout::CrossVertical).unwrap();

        assert_eq!(faces.len(), 6);
        for face in &faces {
            assert_eq!(face.width, 64);
            assert_eq!(face.height, 64);
            assert!(face.is_valid());
        }
    }

    #[test]
    fn test_extract_single_face() {
        let image = create_cross_texture(64, true);

        for face in CubemapFace::ALL {
            let extracted = extract_single_face(&image, CubemapLayout::CrossHorizontal, face).unwrap();
            assert_eq!(extracted.width, 64);
            assert_eq!(extracted.height, 64);
        }
    }

    #[test]
    fn test_extract_faces_invalid_layout() {
        // Test that 256x256 is NOT a valid cross layout
        let layout = detect_cubemap_layout(256, 256);
        assert!(layout.is_none(), "256x256 should not be a valid cross layout");

        // Test that SixImages layout cannot be used with extract_faces_from_cross
        let image = create_test_texture(256, 192);
        let result = extract_faces_from_cross(&image, CubemapLayout::SixImages);
        assert!(result.is_err());
    }

    // ---------------------------------------------------------------------------
    // 6-Image Assembly Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_assemble_cubemap_basic() {
        let faces: [TextureData; 6] = std::array::from_fn(|_| create_test_texture(64, 64));
        let config = CubemapConfig::six_images(64);

        let cubemap = assemble_cubemap(faces, &config).unwrap();

        assert_eq!(cubemap.face_size, 64);
        assert_eq!(cubemap.format, GpuTextureFormat::R8G8B8A8Unorm);
        assert!(cubemap.is_valid());
        assert_eq!(cubemap.gpu_flags, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT);
    }

    #[test]
    fn test_assemble_cubemap_different_sizes() {
        let mut faces: [TextureData; 6] = std::array::from_fn(|_| create_test_texture(64, 64));
        faces[2] = create_test_texture(32, 32); // Mismatched size

        let config = CubemapConfig::six_images(64);
        let result = assemble_cubemap(faces, &config);

        assert!(matches!(result, Err(CubemapError::FaceSizeMismatch { .. })));
    }

    #[test]
    fn test_assemble_cubemap_format_mismatch() {
        let mut faces: [TextureData; 6] = std::array::from_fn(|_| create_test_texture(64, 64));
        faces[3].format = GpuTextureFormat::R8G8B8A8Srgb; // Different format

        let config = CubemapConfig::six_images(64);
        let result = assemble_cubemap(faces, &config);

        assert!(matches!(result, Err(CubemapError::FormatMismatch { .. })));
    }

    // ---------------------------------------------------------------------------
    // Format/Size Validation Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_validate_faces_all_match() {
        let faces: [TextureData; 6] = std::array::from_fn(|_| create_test_texture(128, 128));
        let result = validate_faces(&faces);

        assert!(result.is_ok());
        let (size, format, _) = result.unwrap();
        assert_eq!(size, 128);
        assert_eq!(format, GpuTextureFormat::R8G8B8A8Unorm);
    }

    #[test]
    fn test_validate_faces_non_square() {
        let mut faces: [TextureData; 6] = std::array::from_fn(|_| create_test_texture(64, 64));
        faces[0] = create_test_texture(64, 32); // Non-square

        let result = validate_faces(&faces);
        assert!(matches!(result, Err(CubemapError::FaceSizeMismatch { .. })));
    }

    #[test]
    fn test_cubemap_face_ordering() {
        // Verify GPU ordering is correct
        assert_eq!(CubemapFace::PositiveX.index(), 0);
        assert_eq!(CubemapFace::NegativeX.index(), 1);
        assert_eq!(CubemapFace::PositiveY.index(), 2);
        assert_eq!(CubemapFace::NegativeY.index(), 3);
        assert_eq!(CubemapFace::PositiveZ.index(), 4);
        assert_eq!(CubemapFace::NegativeZ.index(), 5);
    }

    // ---------------------------------------------------------------------------
    // Seam-Aware Mip Filtering Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_seam_aware_mips_basic() {
        let faces: [TextureData; 6] = std::array::from_fn(|_| create_test_texture(64, 64));
        let config = CubemapConfig::six_images(64);
        let mut cubemap = assemble_cubemap(faces, &config).unwrap();

        generate_seam_aware_mips(&mut cubemap).unwrap();

        assert!(cubemap.seam_aware);
        assert!(cubemap.mip_count > 1);

        // Check mip chain
        for face in &cubemap.faces {
            assert!(!face.mip_levels.is_empty());
            assert_eq!(face.mip_levels[0].width, 64);
            if face.mip_levels.len() > 1 {
                assert_eq!(face.mip_levels[1].width, 32);
            }
        }
    }

    #[test]
    fn test_seam_aware_mips_memory_size() {
        let faces: [TextureData; 6] = std::array::from_fn(|_| create_test_texture(64, 64));
        let config = CubemapConfig::six_images(64);
        let mut cubemap = assemble_cubemap(faces, &config).unwrap();

        let initial_size = cubemap.memory_size;
        generate_seam_aware_mips(&mut cubemap).unwrap();

        // Memory should increase with mips
        assert!(cubemap.memory_size > initial_size);
    }

    #[test]
    fn test_seam_aware_mip_levels() {
        let faces: [TextureData; 6] = std::array::from_fn(|_| create_test_texture(128, 128));
        let config = CubemapConfig::six_images(128);
        let mut cubemap = assemble_cubemap(faces, &config).unwrap();

        generate_seam_aware_mips(&mut cubemap).unwrap();

        // 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1 = 8 levels
        assert_eq!(cubemap.mip_count, 8);
    }

    // ---------------------------------------------------------------------------
    // Texture Array Construction Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_create_texture_array_basic() {
        let textures: Vec<TextureData> = (0..10)
            .map(|_| create_test_texture(64, 64))
            .collect();

        let config = TextureArrayConfig::new(64, 64, GpuTextureFormat::R8G8B8A8Unorm);
        let array = create_texture_array(textures, &config).unwrap();

        assert_eq!(array.layer_count(), 10);
        assert_eq!(array.width, 64);
        assert_eq!(array.height, 64);
        assert!(array.is_valid());
    }

    #[test]
    fn test_create_texture_array_empty() {
        let config = TextureArrayConfig::new(64, 64, GpuTextureFormat::R8G8B8A8Unorm);
        let array = create_texture_array(Vec::new(), &config).unwrap();

        assert_eq!(array.layer_count(), 0);
    }

    #[test]
    fn test_create_texture_array_max_layers() {
        let textures: Vec<TextureData> = (0..300)
            .map(|_| create_test_texture(64, 64))
            .collect();

        let config = TextureArrayConfig::new(64, 64, GpuTextureFormat::R8G8B8A8Unorm)
            .with_max_layers(256);
        let result = create_texture_array(textures, &config);

        assert!(matches!(result, Err(CubemapError::LayerCountExceeded { max: 256, requested: 300 })));
    }

    #[test]
    fn test_add_array_layer() {
        let initial: Vec<TextureData> = (0..5)
            .map(|_| create_test_texture(64, 64))
            .collect();

        let config = TextureArrayConfig::new(64, 64, GpuTextureFormat::R8G8B8A8Unorm);
        let mut array = create_texture_array(initial, &config).unwrap();

        let new_layer = create_test_texture(64, 64);
        let idx = add_array_layer(&mut array, new_layer, 256).unwrap();

        assert_eq!(idx, 5);
        assert_eq!(array.layer_count(), 6);
    }

    #[test]
    fn test_add_array_layer_dimension_mismatch() {
        let initial: Vec<TextureData> = (0..5)
            .map(|_| create_test_texture(64, 64))
            .collect();

        let config = TextureArrayConfig::new(64, 64, GpuTextureFormat::R8G8B8A8Unorm);
        let mut array = create_texture_array(initial, &config).unwrap();

        let bad_layer = create_test_texture(32, 32); // Wrong size
        let result = add_array_layer(&mut array, bad_layer, 256);

        assert!(matches!(result, Err(CubemapError::ArrayDimensionMismatch { .. })));
    }

    #[test]
    fn test_generate_array_mips() {
        let textures: Vec<TextureData> = (0..4)
            .map(|_| create_test_texture(64, 64))
            .collect();

        let config = TextureArrayConfig::new(64, 64, GpuTextureFormat::R8G8B8A8Unorm);
        let mut array = create_texture_array(textures, &config).unwrap();

        generate_array_mips(&mut array).unwrap();

        assert!(array.mip_count > 1);
        assert_eq!(array.mip_levels.len(), 4);

        for layer_mips in &array.mip_levels {
            assert!(!layer_mips.is_empty());
        }
    }

    // ---------------------------------------------------------------------------
    // CubemapFace Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_cubemap_face_from_index() {
        for i in 0..6 {
            let face = CubemapFace::from_index(i).unwrap();
            assert_eq!(face.index(), i);
        }

        assert!(CubemapFace::from_index(6).is_none());
    }

    #[test]
    fn test_cubemap_face_directions() {
        assert_eq!(CubemapFace::PositiveX.direction(), [1.0, 0.0, 0.0]);
        assert_eq!(CubemapFace::NegativeX.direction(), [-1.0, 0.0, 0.0]);
        assert_eq!(CubemapFace::PositiveY.direction(), [0.0, 1.0, 0.0]);
        assert_eq!(CubemapFace::NegativeY.direction(), [0.0, -1.0, 0.0]);
        assert_eq!(CubemapFace::PositiveZ.direction(), [0.0, 0.0, 1.0]);
        assert_eq!(CubemapFace::NegativeZ.direction(), [0.0, 0.0, -1.0]);
    }

    #[test]
    fn test_cubemap_face_adjacent() {
        // Each face should have 4 adjacent faces
        for face in CubemapFace::ALL {
            let adjacent = face.adjacent_faces();
            assert_eq!(adjacent.len(), 4);

            // No face should be adjacent to itself
            for (adj, _) in adjacent {
                assert_ne!(adj, face);
            }
        }
    }

    // ---------------------------------------------------------------------------
    // Edge Direction Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_edge_direction_opposite() {
        assert_eq!(EdgeDirection::Left.opposite(), EdgeDirection::Right);
        assert_eq!(EdgeDirection::Right.opposite(), EdgeDirection::Left);
        assert_eq!(EdgeDirection::Top.opposite(), EdgeDirection::Bottom);
        assert_eq!(EdgeDirection::Bottom.opposite(), EdgeDirection::Top);
    }

    // ---------------------------------------------------------------------------
    // TextureData Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_texture_data_get_set_pixel() {
        let mut data = create_test_texture(16, 16);

        let original = data.get_pixel(5, 5);
        data.set_pixel(5, 5, [255, 0, 0, 255]);
        let modified = data.get_pixel(5, 5);

        assert_ne!(original, modified);
        assert_eq!(modified, [255, 0, 0, 255]);
    }

    #[test]
    fn test_texture_data_out_of_bounds() {
        let data = create_test_texture(16, 16);

        // Should return default for out of bounds
        let pixel = data.get_pixel(100, 100);
        assert_eq!(pixel, [0, 0, 0, 255]);
    }

    #[test]
    fn test_texture_data_is_valid() {
        let data = create_test_texture(64, 64);
        assert!(data.is_valid());

        let invalid = TextureData {
            width: 64,
            height: 64,
            format: GpuTextureFormat::R8G8B8A8Unorm,
            data: vec![0; 100], // Wrong size
            is_srgb: false,
        };
        assert!(!invalid.is_valid());
    }

    // ---------------------------------------------------------------------------
    // GPU Flags Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_cubemap_has_cube_compatible_flag() {
        let faces: [TextureData; 6] = std::array::from_fn(|_| create_test_texture(64, 64));
        let config = CubemapConfig::six_images(64);
        let cubemap = assemble_cubemap(faces, &config).unwrap();

        assert!(cubemap.gpu_flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT != 0);
    }

    // ---------------------------------------------------------------------------
    // Reorder Faces Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_reorder_faces_identity() {
        let faces: [TextureData; 6] = std::array::from_fn(|i| {
            let mut tex = create_test_texture(32, 32);
            tex.data[0] = i as u8; // Mark each face
            tex
        });

        let order = CubemapFace::ALL;
        let reordered = reorder_faces_to_gpu(faces, order);

        for i in 0..6 {
            assert_eq!(reordered[i].data[0], i as u8);
        }
    }

    #[test]
    fn test_reorder_faces_reversed() {
        let faces: [TextureData; 6] = std::array::from_fn(|i| {
            let mut tex = create_test_texture(32, 32);
            tex.data[0] = i as u8;
            tex
        });

        // Reversed order
        let order = [
            CubemapFace::NegativeZ,
            CubemapFace::PositiveZ,
            CubemapFace::NegativeY,
            CubemapFace::PositiveY,
            CubemapFace::NegativeX,
            CubemapFace::PositiveX,
        ];

        let reordered = reorder_faces_to_gpu(faces, order);

        // After reordering, face data should be in GPU order
        assert_eq!(reordered[CubemapFace::PositiveX.index()].data[0], 5);
        assert_eq!(reordered[CubemapFace::NegativeZ.index()].data[0], 0);
    }

    // ---------------------------------------------------------------------------
    // CubemapLayout Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_cubemap_layout_aspect_ratio() {
        assert_eq!(CubemapLayout::CrossHorizontal.aspect_ratio(), Some((4, 3)));
        assert_eq!(CubemapLayout::CrossVertical.aspect_ratio(), Some((3, 4)));
        assert_eq!(CubemapLayout::SixImages.aspect_ratio(), None);
        assert_eq!(CubemapLayout::KtxCubemap.aspect_ratio(), None);
    }

    #[test]
    fn test_cubemap_layout_name() {
        assert_eq!(CubemapLayout::CrossHorizontal.name(), "CrossHorizontal");
        assert_eq!(CubemapLayout::CrossVertical.name(), "CrossVertical");
        assert_eq!(CubemapLayout::SixImages.name(), "SixImages");
        assert_eq!(CubemapLayout::KtxCubemap.name(), "KtxCubemap");
    }

    // ---------------------------------------------------------------------------
    // Memory Size Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_cubemap_memory_size() {
        let faces: [TextureData; 6] = std::array::from_fn(|_| create_test_texture(64, 64));
        let config = CubemapConfig::six_images(64);
        let cubemap = assemble_cubemap(faces, &config).unwrap();

        let expected = 64 * 64 * 4 * 6; // width * height * bpp * faces
        assert_eq!(cubemap.memory_size, expected);
    }

    #[test]
    fn test_texture_array_memory_size() {
        let textures: Vec<TextureData> = (0..8)
            .map(|_| create_test_texture(64, 64))
            .collect();

        let config = TextureArrayConfig::new(64, 64, GpuTextureFormat::R8G8B8A8Unorm);
        let array = create_texture_array(textures, &config).unwrap();

        let expected = 64 * 64 * 4 * 8; // width * height * bpp * layers
        assert_eq!(array.memory_size, expected);
    }
}
