//! Cubemap and Texture Array support.
//!
//! Provides import, manipulation, and cooking of cubemap textures and texture arrays
//! for environment mapping, reflection probes, and other GPU operations.
//!
//! # Cubemap Layouts
//!
//! Supports multiple input layouts for cubemap faces:
//!
//! - **CrossHorizontal** (4x3): Standard horizontal cross layout
//! - **CrossVertical** (3x4): Vertical cross layout
//! - **Strip** (6x1): Horizontal strip of 6 faces
//! - **Separate**: 6 individual images
//!
//! # Example
//!
//! ```
//! use renderer_backend::cubemap::{CubemapLayout, CubemapImporter, CubemapFace};
//! use renderer_backend::texture_import::{TextureData, TextureFormat};
//!
//! // Create a horizontal cross cubemap (4x3 aspect ratio)
//! let face_size = 64;
//! let width = face_size * 4;
//! let height = face_size * 3;
//! let data = vec![128u8; (width * height * 4) as usize];
//! let texture = TextureData::new(width, height, TextureFormat::Rgba8, data, 1);
//!
//! let importer = CubemapImporter::new();
//! let cubemap = importer.import(&texture, CubemapLayout::CrossHorizontal).unwrap();
//!
//! assert_eq!(cubemap.face_size, 64);
//! assert_eq!(cubemap.faces.len(), 6);
//! ```

use std::fmt;

use crate::texture_import::{
    box_filter_2x2, calculate_mip_levels, CookError, GpuTextureFormat,
    TextureCooker, TextureData, TextureFormat, TextureUsage,
};

// ---------------------------------------------------------------------------
// CubemapFace
// ---------------------------------------------------------------------------

/// Identifies a face of a cubemap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum CubemapFace {
    /// Positive X (+X) face.
    PositiveX = 0,
    /// Negative X (-X) face.
    NegativeX = 1,
    /// Positive Y (+Y) face.
    PositiveY = 2,
    /// Negative Y (-Y) face.
    NegativeY = 3,
    /// Positive Z (+Z) face.
    PositiveZ = 4,
    /// Negative Z (-Z) face.
    NegativeZ = 5,
}

impl CubemapFace {
    /// All faces in order.
    pub const ALL: [CubemapFace; 6] = [
        CubemapFace::PositiveX,
        CubemapFace::NegativeX,
        CubemapFace::PositiveY,
        CubemapFace::NegativeY,
        CubemapFace::PositiveZ,
        CubemapFace::NegativeZ,
    ];

    /// Returns the index of this face (0-5).
    #[inline]
    pub const fn index(&self) -> usize {
        *self as usize
    }

    /// Creates a face from an index (0-5).
    pub const fn from_index(index: usize) -> Option<CubemapFace> {
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

    /// Returns the short name for this face.
    pub const fn short_name(&self) -> &'static str {
        match self {
            CubemapFace::PositiveX => "+X",
            CubemapFace::NegativeX => "-X",
            CubemapFace::PositiveY => "+Y",
            CubemapFace::NegativeY => "-Y",
            CubemapFace::PositiveZ => "+Z",
            CubemapFace::NegativeZ => "-Z",
        }
    }
}

impl fmt::Display for CubemapFace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.short_name())
    }
}

// ---------------------------------------------------------------------------
// CubemapLayout
// ---------------------------------------------------------------------------

/// Layout of cubemap faces in a single image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CubemapLayout {
    /// Horizontal cross layout (4x3 grid).
    ///
    /// ```text
    ///     [+Y]
    /// [-X][+Z][+X][-Z]
    ///     [-Y]
    /// ```
    CrossHorizontal,

    /// Vertical cross layout (3x4 grid).
    ///
    /// ```text
    ///     [+Y]
    /// [-X][+Z][+X]
    ///     [-Y]
    ///     [-Z]
    /// ```
    CrossVertical,

    /// Horizontal strip (6x1).
    ///
    /// Faces in order: +X, -X, +Y, -Y, +Z, -Z
    Strip,

    /// Six separate images (provided individually).
    Separate,
}

impl CubemapLayout {
    /// Detect layout from image dimensions.
    ///
    /// Returns `None` if dimensions don't match any known layout.
    pub fn detect(width: u32, height: u32) -> Option<CubemapLayout> {
        if width == 0 || height == 0 {
            return None;
        }

        // Check horizontal cross (4:3 ratio)
        if width % 4 == 0 && height % 3 == 0 {
            let face_w = width / 4;
            let face_h = height / 3;
            if face_w == face_h {
                return Some(CubemapLayout::CrossHorizontal);
            }
        }

        // Check vertical cross (3:4 ratio)
        if width % 3 == 0 && height % 4 == 0 {
            let face_w = width / 3;
            let face_h = height / 4;
            if face_w == face_h {
                return Some(CubemapLayout::CrossVertical);
            }
        }

        // Check strip (6:1 ratio)
        if width % 6 == 0 {
            let face_w = width / 6;
            if face_w == height {
                return Some(CubemapLayout::Strip);
            }
        }

        // Check if it's a single square face (separate mode)
        if width == height {
            return Some(CubemapLayout::Separate);
        }

        None
    }

    /// Calculate the face size for this layout given image dimensions.
    pub fn face_size(&self, width: u32, height: u32) -> Option<u32> {
        match self {
            CubemapLayout::CrossHorizontal => {
                if width % 4 == 0 && height % 3 == 0 {
                    let face_w = width / 4;
                    let face_h = height / 3;
                    if face_w == face_h {
                        return Some(face_w);
                    }
                }
                None
            }
            CubemapLayout::CrossVertical => {
                if width % 3 == 0 && height % 4 == 0 {
                    let face_w = width / 3;
                    let face_h = height / 4;
                    if face_w == face_h {
                        return Some(face_w);
                    }
                }
                None
            }
            CubemapLayout::Strip => {
                if width % 6 == 0 {
                    let face_w = width / 6;
                    if face_w == height {
                        return Some(face_w);
                    }
                }
                None
            }
            CubemapLayout::Separate => {
                if width == height {
                    Some(width)
                } else {
                    None
                }
            }
        }
    }

    /// Returns the grid position (column, row) of a face in this layout.
    ///
    /// For `Separate`, returns (0, 0) as faces are individual images.
    pub fn face_position(&self, face: CubemapFace) -> (u32, u32) {
        match self {
            CubemapLayout::CrossHorizontal => {
                // Layout:
                //     [+Y]        (col=1, row=0)
                // [-X][+Z][+X][-Z]  (row=1)
                //     [-Y]        (col=1, row=2)
                match face {
                    CubemapFace::PositiveX => (2, 1),
                    CubemapFace::NegativeX => (0, 1),
                    CubemapFace::PositiveY => (1, 0),
                    CubemapFace::NegativeY => (1, 2),
                    CubemapFace::PositiveZ => (1, 1),
                    CubemapFace::NegativeZ => (3, 1),
                }
            }
            CubemapLayout::CrossVertical => {
                // Layout:
                //     [+Y]     (col=1, row=0)
                // [-X][+Z][+X] (row=1)
                //     [-Y]     (col=1, row=2)
                //     [-Z]     (col=1, row=3)
                match face {
                    CubemapFace::PositiveX => (2, 1),
                    CubemapFace::NegativeX => (0, 1),
                    CubemapFace::PositiveY => (1, 0),
                    CubemapFace::NegativeY => (1, 2),
                    CubemapFace::PositiveZ => (1, 1),
                    CubemapFace::NegativeZ => (1, 3),
                }
            }
            CubemapLayout::Strip => {
                // Order: +X, -X, +Y, -Y, +Z, -Z
                (face.index() as u32, 0)
            }
            CubemapLayout::Separate => (0, 0),
        }
    }
}

impl fmt::Display for CubemapLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CubemapLayout::CrossHorizontal => write!(f, "CrossHorizontal (4x3)"),
            CubemapLayout::CrossVertical => write!(f, "CrossVertical (3x4)"),
            CubemapLayout::Strip => write!(f, "Strip (6x1)"),
            CubemapLayout::Separate => write!(f, "Separate (6 images)"),
        }
    }
}

// ---------------------------------------------------------------------------
// CubemapError
// ---------------------------------------------------------------------------

/// Errors that can occur during cubemap operations.
#[derive(Debug, Clone)]
pub enum CubemapError {
    /// Invalid dimensions for the specified layout.
    InvalidDimensions {
        width: u32,
        height: u32,
        layout: CubemapLayout,
    },
    /// Layout could not be detected from dimensions.
    LayoutDetectionFailed { width: u32, height: u32 },
    /// Missing face data.
    MissingFace(CubemapFace),
    /// Face size mismatch between faces.
    FaceSizeMismatch { expected: u32, got: u32 },
    /// Format mismatch between faces.
    FormatMismatch {
        expected: TextureFormat,
        got: TextureFormat,
    },
    /// Invalid face count for texture array.
    InvalidFaceCount { expected: usize, got: usize },
    /// Cooking error.
    CookError(CookError),
    /// Invalid texture array layer count.
    InvalidLayerCount(usize),
}

impl fmt::Display for CubemapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CubemapError::InvalidDimensions {
                width,
                height,
                layout,
            } => {
                write!(
                    f,
                    "invalid dimensions {}x{} for layout {}",
                    width, height, layout
                )
            }
            CubemapError::LayoutDetectionFailed { width, height } => {
                write!(f, "cannot detect layout for {}x{}", width, height)
            }
            CubemapError::MissingFace(face) => {
                write!(f, "missing face: {}", face)
            }
            CubemapError::FaceSizeMismatch { expected, got } => {
                write!(f, "face size mismatch: expected {}, got {}", expected, got)
            }
            CubemapError::FormatMismatch { expected, got } => {
                write!(f, "format mismatch: expected {}, got {}", expected, got)
            }
            CubemapError::InvalidFaceCount { expected, got } => {
                write!(f, "invalid face count: expected {}, got {}", expected, got)
            }
            CubemapError::CookError(e) => write!(f, "cook error: {}", e),
            CubemapError::InvalidLayerCount(count) => {
                write!(f, "invalid layer count: {}", count)
            }
        }
    }
}

impl std::error::Error for CubemapError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CubemapError::CookError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<CookError> for CubemapError {
    fn from(err: CookError) -> Self {
        CubemapError::CookError(err)
    }
}

// ---------------------------------------------------------------------------
// Cubemap
// ---------------------------------------------------------------------------

/// A cubemap texture with 6 faces.
#[derive(Debug, Clone)]
pub struct Cubemap {
    /// Size of each face (faces are square).
    pub face_size: u32,
    /// Pixel format.
    pub format: TextureFormat,
    /// Face data in order: +X, -X, +Y, -Y, +Z, -Z.
    pub faces: [Vec<u8>; 6],
    /// Number of mip levels per face.
    pub mip_levels: u32,
}

impl Cubemap {
    /// Create an empty cubemap with the given parameters.
    pub fn new(face_size: u32, format: TextureFormat, mip_levels: u32) -> Self {
        let face_bytes = (face_size * face_size) as usize * format.bytes_per_pixel();
        Self {
            face_size,
            format,
            faces: [
                vec![0u8; face_bytes],
                vec![0u8; face_bytes],
                vec![0u8; face_bytes],
                vec![0u8; face_bytes],
                vec![0u8; face_bytes],
                vec![0u8; face_bytes],
            ],
            mip_levels,
        }
    }

    /// Get the data for a specific face.
    #[inline]
    pub fn face(&self, face: CubemapFace) -> &[u8] {
        &self.faces[face.index()]
    }

    /// Get mutable data for a specific face.
    #[inline]
    pub fn face_mut(&mut self, face: CubemapFace) -> &mut Vec<u8> {
        &mut self.faces[face.index()]
    }

    /// Set the data for a specific face.
    pub fn set_face(&mut self, face: CubemapFace, data: Vec<u8>) -> Result<(), CubemapError> {
        let expected_size = (self.face_size * self.face_size) as usize * self.format.bytes_per_pixel();
        if data.len() != expected_size {
            return Err(CubemapError::FaceSizeMismatch {
                expected: expected_size as u32,
                got: data.len() as u32,
            });
        }
        self.faces[face.index()] = data;
        Ok(())
    }

    /// Returns the expected byte size for one face.
    #[inline]
    pub fn face_byte_size(&self) -> usize {
        (self.face_size * self.face_size) as usize * self.format.bytes_per_pixel()
    }

    /// Returns total size of all face data in bytes.
    pub fn total_size(&self) -> usize {
        self.faces.iter().map(|f| f.len()).sum()
    }

    /// Validates that all faces have correct sizes.
    pub fn is_valid(&self) -> bool {
        let expected_size = self.face_byte_size();
        self.faces.iter().all(|f| f.len() == expected_size)
    }

    /// Convert this cubemap to a TextureData array (one per face).
    pub fn to_texture_data_array(&self) -> [TextureData; 6] {
        CubemapFace::ALL.map(|face| TextureData {
            width: self.face_size,
            height: self.face_size,
            format: self.format,
            data: self.faces[face.index()].clone(),
            mip_levels: self.mip_levels,
        })
    }
}

// ---------------------------------------------------------------------------
// CubemapImporter
// ---------------------------------------------------------------------------

/// Imports cubemap textures from various layouts.
#[derive(Debug, Clone)]
pub struct CubemapImporter {
    /// Whether to flip faces vertically during import.
    pub flip_vertical: bool,
}

impl CubemapImporter {
    /// Create a new cubemap importer with default settings.
    pub fn new() -> Self {
        Self {
            flip_vertical: false,
        }
    }

    /// Enable or disable vertical flipping of faces.
    pub fn with_flip_vertical(mut self, flip: bool) -> Self {
        self.flip_vertical = flip;
        self
    }

    /// Import a cubemap from a single image with the specified layout.
    pub fn import(
        &self,
        texture: &TextureData,
        layout: CubemapLayout,
    ) -> Result<Cubemap, CubemapError> {
        let face_size = layout
            .face_size(texture.width, texture.height)
            .ok_or(CubemapError::InvalidDimensions {
                width: texture.width,
                height: texture.height,
                layout,
            })?;

        let mut cubemap = Cubemap::new(face_size, texture.format, texture.mip_levels);

        for face in CubemapFace::ALL {
            let face_data = self.extract_face(texture, layout, face, face_size)?;
            cubemap.set_face(face, face_data)?;
        }

        Ok(cubemap)
    }

    /// Import a cubemap from 6 separate face images.
    ///
    /// Images must be provided in order: +X, -X, +Y, -Y, +Z, -Z.
    pub fn import_separate(&self, faces: &[TextureData; 6]) -> Result<Cubemap, CubemapError> {
        // Validate all faces have same dimensions and format
        let first = &faces[0];
        if first.width != first.height {
            return Err(CubemapError::InvalidDimensions {
                width: first.width,
                height: first.height,
                layout: CubemapLayout::Separate,
            });
        }

        let face_size = first.width;
        let format = first.format;

        for (_i, face) in faces.iter().enumerate().skip(1) {
            if face.width != face_size || face.height != face_size {
                return Err(CubemapError::FaceSizeMismatch {
                    expected: face_size,
                    got: face.width.max(face.height),
                });
            }
            if face.format != format {
                return Err(CubemapError::FormatMismatch {
                    expected: format,
                    got: face.format,
                });
            }
        }

        let mut cubemap = Cubemap::new(face_size, format, first.mip_levels);
        for (i, face_data) in faces.iter().enumerate() {
            let face = CubemapFace::from_index(i).unwrap();
            let data = if self.flip_vertical {
                self.flip_face_vertical(&face_data.data, face_size, format)
            } else {
                face_data.data.clone()
            };
            cubemap.set_face(face, data)?;
        }

        Ok(cubemap)
    }

    /// Auto-detect layout and import.
    pub fn import_auto(&self, texture: &TextureData) -> Result<Cubemap, CubemapError> {
        let layout = CubemapLayout::detect(texture.width, texture.height).ok_or(
            CubemapError::LayoutDetectionFailed {
                width: texture.width,
                height: texture.height,
            },
        )?;

        self.import(texture, layout)
    }

    /// Extract a single face from the source texture.
    fn extract_face(
        &self,
        texture: &TextureData,
        layout: CubemapLayout,
        face: CubemapFace,
        face_size: u32,
    ) -> Result<Vec<u8>, CubemapError> {
        let (col, row) = layout.face_position(face);
        let bpp = texture.format.bytes_per_pixel();
        let src_stride = texture.width as usize * bpp;
        let face_stride = face_size as usize * bpp;

        let start_x = (col * face_size) as usize;
        let start_y = (row * face_size) as usize;

        let mut face_data = vec![0u8; (face_size * face_size) as usize * bpp];

        for y in 0..face_size as usize {
            let src_y = if self.flip_vertical {
                start_y + (face_size as usize - 1 - y)
            } else {
                start_y + y
            };

            let src_offset = src_y * src_stride + start_x * bpp;
            let dst_offset = y * face_stride;

            if src_offset + face_stride <= texture.data.len() {
                face_data[dst_offset..dst_offset + face_stride]
                    .copy_from_slice(&texture.data[src_offset..src_offset + face_stride]);
            }
        }

        Ok(face_data)
    }

    /// Flip face data vertically.
    fn flip_face_vertical(&self, data: &[u8], face_size: u32, format: TextureFormat) -> Vec<u8> {
        let bpp = format.bytes_per_pixel();
        let stride = face_size as usize * bpp;
        let mut flipped = vec![0u8; data.len()];

        for y in 0..face_size as usize {
            let src_y = face_size as usize - 1 - y;
            let src_offset = src_y * stride;
            let dst_offset = y * stride;
            flipped[dst_offset..dst_offset + stride]
                .copy_from_slice(&data[src_offset..src_offset + stride]);
        }

        flipped
    }
}

impl Default for CubemapImporter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// TextureArray
// ---------------------------------------------------------------------------

/// A texture array with multiple layers of the same dimensions.
#[derive(Debug, Clone)]
pub struct TextureArray {
    /// Texture data for each layer.
    pub layers: Vec<TextureData>,
    /// Pixel format (must be same for all layers).
    pub format: TextureFormat,
    /// Width of each layer.
    pub width: u32,
    /// Height of each layer.
    pub height: u32,
}

impl TextureArray {
    /// Create a new empty texture array.
    pub fn new(width: u32, height: u32, format: TextureFormat) -> Self {
        Self {
            layers: Vec::new(),
            format,
            width,
            height,
        }
    }

    /// Create a texture array from a vector of textures.
    ///
    /// All textures must have the same dimensions and format.
    pub fn from_textures(textures: Vec<TextureData>) -> Result<Self, CubemapError> {
        if textures.is_empty() {
            return Err(CubemapError::InvalidLayerCount(0));
        }

        let first = &textures[0];
        let width = first.width;
        let height = first.height;
        let format = first.format;

        for (_i, tex) in textures.iter().enumerate().skip(1) {
            if tex.width != width || tex.height != height {
                return Err(CubemapError::FaceSizeMismatch {
                    expected: width.max(height),
                    got: tex.width.max(tex.height),
                });
            }
            if tex.format != format {
                return Err(CubemapError::FormatMismatch {
                    expected: format,
                    got: tex.format,
                });
            }
        }

        Ok(Self {
            layers: textures,
            format,
            width,
            height,
        })
    }

    /// Add a layer to the texture array.
    pub fn add_layer(&mut self, texture: TextureData) -> Result<(), CubemapError> {
        if texture.width != self.width || texture.height != self.height {
            return Err(CubemapError::FaceSizeMismatch {
                expected: self.width.max(self.height),
                got: texture.width.max(texture.height),
            });
        }
        if texture.format != self.format {
            return Err(CubemapError::FormatMismatch {
                expected: self.format,
                got: texture.format,
            });
        }
        self.layers.push(texture);
        Ok(())
    }

    /// Get a specific layer.
    pub fn layer(&self, index: usize) -> Option<&TextureData> {
        self.layers.get(index)
    }

    /// Get mutable reference to a specific layer.
    pub fn layer_mut(&mut self, index: usize) -> Option<&mut TextureData> {
        self.layers.get_mut(index)
    }

    /// Returns the number of layers.
    #[inline]
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Returns true if the array is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Returns total size of all layer data in bytes.
    pub fn total_size(&self) -> usize {
        self.layers.iter().map(|l| l.data.len()).sum()
    }

    /// Validates all layers have correct sizes.
    pub fn is_valid(&self) -> bool {
        let expected_size = (self.width * self.height) as usize * self.format.bytes_per_pixel();
        self.layers.iter().all(|l| {
            l.width == self.width
                && l.height == self.height
                && l.format == self.format
                && l.data.len() >= expected_size
        })
    }
}

// ---------------------------------------------------------------------------
// CubemapArray
// ---------------------------------------------------------------------------

/// An array of cubemaps for reflection probes.
#[derive(Debug, Clone)]
pub struct CubemapArray {
    /// Individual cubemaps in the array.
    pub cubemaps: Vec<Cubemap>,
    /// Face size (must be same for all cubemaps).
    pub face_size: u32,
    /// Pixel format (must be same for all cubemaps).
    pub format: TextureFormat,
}

impl CubemapArray {
    /// Create a new empty cubemap array.
    pub fn new(face_size: u32, format: TextureFormat) -> Self {
        Self {
            cubemaps: Vec::new(),
            face_size,
            format,
        }
    }

    /// Create a cubemap array from a vector of cubemaps.
    pub fn from_cubemaps(cubemaps: Vec<Cubemap>) -> Result<Self, CubemapError> {
        if cubemaps.is_empty() {
            return Err(CubemapError::InvalidLayerCount(0));
        }

        let first = &cubemaps[0];
        let face_size = first.face_size;
        let format = first.format;

        for cubemap in cubemaps.iter().skip(1) {
            if cubemap.face_size != face_size {
                return Err(CubemapError::FaceSizeMismatch {
                    expected: face_size,
                    got: cubemap.face_size,
                });
            }
            if cubemap.format != format {
                return Err(CubemapError::FormatMismatch {
                    expected: format,
                    got: cubemap.format,
                });
            }
        }

        Ok(Self {
            cubemaps,
            face_size,
            format,
        })
    }

    /// Add a cubemap to the array.
    pub fn add_cubemap(&mut self, cubemap: Cubemap) -> Result<(), CubemapError> {
        if cubemap.face_size != self.face_size {
            return Err(CubemapError::FaceSizeMismatch {
                expected: self.face_size,
                got: cubemap.face_size,
            });
        }
        if cubemap.format != self.format {
            return Err(CubemapError::FormatMismatch {
                expected: self.format,
                got: cubemap.format,
            });
        }
        self.cubemaps.push(cubemap);
        Ok(())
    }

    /// Get a specific cubemap.
    pub fn cubemap(&self, index: usize) -> Option<&Cubemap> {
        self.cubemaps.get(index)
    }

    /// Get a specific face from a specific cubemap.
    pub fn face(&self, cubemap_index: usize, face: CubemapFace) -> Option<&[u8]> {
        self.cubemaps.get(cubemap_index).map(|c| c.face(face))
    }

    /// Returns the number of cubemaps in the array.
    #[inline]
    pub fn len(&self) -> usize {
        self.cubemaps.len()
    }

    /// Returns true if the array is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.cubemaps.is_empty()
    }

    /// Returns total face count (cubemaps * 6).
    #[inline]
    pub fn total_face_count(&self) -> usize {
        self.cubemaps.len() * 6
    }

    /// Returns total size of all data in bytes.
    pub fn total_size(&self) -> usize {
        self.cubemaps.iter().map(|c| c.total_size()).sum()
    }

    /// Flatten to a texture array (each face becomes a layer).
    ///
    /// Order: cubemap0_face0, cubemap0_face1, ..., cubemap1_face0, ...
    pub fn to_texture_array(&self) -> TextureArray {
        let mut array = TextureArray::new(self.face_size, self.face_size, self.format);

        for cubemap in &self.cubemaps {
            for face in CubemapFace::ALL {
                let tex = TextureData {
                    width: self.face_size,
                    height: self.face_size,
                    format: self.format,
                    data: cubemap.face(face).to_vec(),
                    mip_levels: cubemap.mip_levels,
                };
                // Unwrap is safe here because we control the format
                let _ = array.add_layer(tex);
            }
        }

        array
    }
}

// ---------------------------------------------------------------------------
// CookedCubemap
// ---------------------------------------------------------------------------

/// A cooked cubemap ready for GPU upload.
#[derive(Debug, Clone)]
pub struct CookedCubemap {
    /// Face size at mip level 0.
    pub face_size: u32,
    /// GPU texture format.
    pub format: GpuTextureFormat,
    /// Mip data for each face [face_index][mip_level].
    pub face_mips: [[Vec<u8>; 16]; 6], // Max 16 mip levels
    /// Number of mip levels.
    pub mip_count: u32,
    /// Usage hint.
    pub usage: TextureUsage,
}

impl CookedCubemap {
    /// Get mip data for a specific face and level.
    pub fn mip_data(&self, face: CubemapFace, level: u32) -> Option<&[u8]> {
        if level >= self.mip_count {
            return None;
        }
        let data = &self.face_mips[face.index()][level as usize];
        if data.is_empty() {
            None
        } else {
            Some(data)
        }
    }

    /// Returns the dimensions of a specific mip level.
    pub fn mip_dimensions(&self, level: u32) -> Option<u32> {
        if level >= self.mip_count {
            return None;
        }
        let divisor = 1 << level;
        Some((self.face_size / divisor).max(1))
    }

    /// Returns total size of all mip data in bytes.
    pub fn total_size(&self) -> usize {
        self.face_mips
            .iter()
            .map(|face| face.iter().map(|m| m.len()).sum::<usize>())
            .sum()
    }
}

// ---------------------------------------------------------------------------
// CubemapCooker
// ---------------------------------------------------------------------------

/// Cooks cubemaps for GPU upload.
#[derive(Debug, Clone)]
pub struct CubemapCooker {
    /// Inner texture cooker.
    cooker: TextureCooker,
}

impl CubemapCooker {
    /// Create a new cubemap cooker.
    pub fn new() -> Self {
        Self {
            cooker: TextureCooker::new(),
        }
    }

    /// Enable or disable mip generation.
    pub fn with_mips(mut self, generate: bool) -> Self {
        self.cooker = self.cooker.with_mips(generate);
        self
    }

    /// Enable or disable block compression.
    pub fn with_compression(mut self, compress: bool) -> Self {
        self.cooker = self.cooker.with_compression(compress);
        self
    }

    /// Cook a cubemap for GPU upload.
    pub fn cook(&self, cubemap: &Cubemap, usage: TextureUsage) -> Result<CookedCubemap, CubemapError> {
        // Cook each face
        let mut face_mips: [[Vec<u8>; 16]; 6] = Default::default();
        let mut format = None;
        let mut mip_count = 0u32;

        for face in CubemapFace::ALL {
            let texture_data = TextureData {
                width: cubemap.face_size,
                height: cubemap.face_size,
                format: cubemap.format,
                data: cubemap.face(face).to_vec(),
                mip_levels: 1,
            };

            let cooked = self.cooker.cook(&texture_data, usage)?;

            if format.is_none() {
                format = Some(cooked.format);
                mip_count = cooked.mip_count();
            }

            for (level, mip_data) in cooked.mip_data.into_iter().enumerate() {
                if level < 16 {
                    face_mips[face.index()][level] = mip_data;
                }
            }
        }

        Ok(CookedCubemap {
            face_size: cubemap.face_size,
            format: format.unwrap_or(GpuTextureFormat::Rgba8Unorm),
            face_mips,
            mip_count,
            usage,
        })
    }
}

impl Default for CubemapCooker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// CookedTextureArray
// ---------------------------------------------------------------------------

/// A cooked texture array ready for GPU upload.
#[derive(Debug, Clone)]
pub struct CookedTextureArray {
    /// Width at mip level 0.
    pub width: u32,
    /// Height at mip level 0.
    pub height: u32,
    /// Number of layers.
    pub layer_count: usize,
    /// GPU texture format.
    pub format: GpuTextureFormat,
    /// Mip data for each layer [layer_index][mip_level].
    pub layer_mips: Vec<Vec<Vec<u8>>>,
    /// Number of mip levels.
    pub mip_count: u32,
    /// Usage hint.
    pub usage: TextureUsage,
}

impl CookedTextureArray {
    /// Get mip data for a specific layer and level.
    pub fn mip_data(&self, layer: usize, level: u32) -> Option<&[u8]> {
        if layer >= self.layer_count || level >= self.mip_count {
            return None;
        }
        self.layer_mips
            .get(layer)
            .and_then(|mips| mips.get(level as usize))
            .map(|v| v.as_slice())
    }

    /// Returns total size of all mip data in bytes.
    pub fn total_size(&self) -> usize {
        self.layer_mips
            .iter()
            .map(|layer| layer.iter().map(|m| m.len()).sum::<usize>())
            .sum()
    }
}

// ---------------------------------------------------------------------------
// TextureArrayCooker
// ---------------------------------------------------------------------------

/// Cooks texture arrays for GPU upload.
#[derive(Debug, Clone)]
pub struct TextureArrayCooker {
    /// Inner texture cooker.
    cooker: TextureCooker,
}

impl TextureArrayCooker {
    /// Create a new texture array cooker.
    pub fn new() -> Self {
        Self {
            cooker: TextureCooker::new(),
        }
    }

    /// Enable or disable mip generation.
    pub fn with_mips(mut self, generate: bool) -> Self {
        self.cooker = self.cooker.with_mips(generate);
        self
    }

    /// Enable or disable block compression.
    pub fn with_compression(mut self, compress: bool) -> Self {
        self.cooker = self.cooker.with_compression(compress);
        self
    }

    /// Cook a texture array for GPU upload.
    pub fn cook(
        &self,
        array: &TextureArray,
        usage: TextureUsage,
    ) -> Result<CookedTextureArray, CubemapError> {
        if array.is_empty() {
            return Err(CubemapError::InvalidLayerCount(0));
        }

        let mut layer_mips = Vec::with_capacity(array.layer_count());
        let mut format = None;
        let mut mip_count = 0u32;

        for layer in &array.layers {
            let cooked = self.cooker.cook(layer, usage)?;

            if format.is_none() {
                format = Some(cooked.format);
                mip_count = cooked.mip_count();
            }

            layer_mips.push(cooked.mip_data);
        }

        Ok(CookedTextureArray {
            width: array.width,
            height: array.height,
            layer_count: array.layer_count(),
            format: format.unwrap_or(GpuTextureFormat::Rgba8Unorm),
            layer_mips,
            mip_count,
            usage,
        })
    }
}

impl Default for TextureArrayCooker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Generate mip chain for a single cubemap face.
pub fn generate_face_mips(data: &[u8], size: u32, format: TextureFormat) -> Vec<Vec<u8>> {
    let bpp = format.bytes_per_pixel() as u32;
    let num_levels = calculate_mip_levels(size, size);
    let mut mips = Vec::with_capacity(num_levels as usize);

    // Level 0 is the original data
    mips.push(data.to_vec());

    let mut current_data = data.to_vec();
    let mut current_size = size;

    for _ in 1..num_levels {
        let new_size = (current_size / 2).max(1);
        let downsampled = box_filter_2x2(&current_data, current_size, current_size, bpp);
        mips.push(downsampled.clone());
        current_data = downsampled;
        current_size = new_size;
    }

    mips
}

/// Generate mip chain for all faces of a cubemap.
pub fn generate_cubemap_mips(cubemap: &Cubemap) -> Vec<[Vec<u8>; 6]> {
    let num_levels = calculate_mip_levels(cubemap.face_size, cubemap.face_size);
    let mut all_mips: Vec<[Vec<u8>; 6]> = Vec::with_capacity(num_levels as usize);

    // Initialize empty arrays for each mip level
    for _ in 0..num_levels {
        all_mips.push([
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ]);
    }

    // Generate mips for each face
    for face in CubemapFace::ALL {
        let face_mips = generate_face_mips(cubemap.face(face), cubemap.face_size, cubemap.format);
        for (level, mip_data) in face_mips.into_iter().enumerate() {
            all_mips[level][face.index()] = mip_data;
        }
    }

    all_mips
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== Layout Detection Tests ====================

    #[test]
    fn test_layout_detect_horizontal_cross() {
        // 4x3 aspect ratio with face size 64
        let layout = CubemapLayout::detect(256, 192);
        assert_eq!(layout, Some(CubemapLayout::CrossHorizontal));

        // 4x3 with face size 128
        let layout = CubemapLayout::detect(512, 384);
        assert_eq!(layout, Some(CubemapLayout::CrossHorizontal));
    }

    #[test]
    fn test_layout_detect_vertical_cross() {
        // 3x4 aspect ratio with face size 64
        let layout = CubemapLayout::detect(192, 256);
        assert_eq!(layout, Some(CubemapLayout::CrossVertical));

        // 3x4 with face size 128
        let layout = CubemapLayout::detect(384, 512);
        assert_eq!(layout, Some(CubemapLayout::CrossVertical));
    }

    #[test]
    fn test_layout_detect_strip() {
        // 6x1 aspect ratio with face size 64
        let layout = CubemapLayout::detect(384, 64);
        assert_eq!(layout, Some(CubemapLayout::Strip));

        // 6x1 with face size 128
        let layout = CubemapLayout::detect(768, 128);
        assert_eq!(layout, Some(CubemapLayout::Strip));
    }

    #[test]
    fn test_layout_detect_separate() {
        // Square image (single face)
        let layout = CubemapLayout::detect(256, 256);
        assert_eq!(layout, Some(CubemapLayout::Separate));
    }

    #[test]
    fn test_layout_detect_invalid() {
        // Invalid dimensions
        assert_eq!(CubemapLayout::detect(0, 0), None);
        assert_eq!(CubemapLayout::detect(100, 50), None);
    }

    // ==================== Face Extraction Tests ====================

    #[test]
    fn test_extract_face_horizontal_cross() {
        let face_size = 4u32;
        let width = face_size * 4;
        let height = face_size * 3;
        let bpp = 4;

        // Create test data with each face having distinct values
        let mut data = vec![0u8; (width * height * bpp) as usize];

        // Mark +Z face (center of cross at col=1, row=1) with value 128
        for y in face_size..(face_size * 2) {
            for x in face_size..(face_size * 2) {
                let idx = ((y * width + x) * bpp) as usize;
                data[idx] = 128;
            }
        }

        let texture = TextureData::new(width, height, TextureFormat::Rgba8, data, 1);
        let importer = CubemapImporter::new();
        let cubemap = importer.import(&texture, CubemapLayout::CrossHorizontal).unwrap();

        // Verify +Z face has value 128
        let pz_face = cubemap.face(CubemapFace::PositiveZ);
        assert_eq!(pz_face[0], 128);
    }

    #[test]
    fn test_extract_face_vertical_cross() {
        let face_size = 4u32;
        let width = face_size * 3;
        let height = face_size * 4;
        let bpp = 4;

        let mut data = vec![0u8; (width * height * bpp) as usize];

        // Mark -Z face (col=1, row=3) with value 200
        for y in (face_size * 3)..(face_size * 4) {
            for x in face_size..(face_size * 2) {
                let idx = ((y * width + x) * bpp) as usize;
                data[idx] = 200;
            }
        }

        let texture = TextureData::new(width, height, TextureFormat::Rgba8, data, 1);
        let importer = CubemapImporter::new();
        let cubemap = importer.import(&texture, CubemapLayout::CrossVertical).unwrap();

        // Verify -Z face has value 200
        let nz_face = cubemap.face(CubemapFace::NegativeZ);
        assert_eq!(nz_face[0], 200);
    }

    #[test]
    fn test_extract_face_strip() {
        let face_size = 4u32;
        let width = face_size * 6;
        let height = face_size;
        let bpp = 4;

        let mut data = vec![0u8; (width * height * bpp) as usize];

        // Mark each face with its index value
        for face_idx in 0..6 {
            let start_x = face_idx * face_size;
            for y in 0..face_size {
                for x in start_x..(start_x + face_size) {
                    let idx = ((y * width + x) * bpp) as usize;
                    data[idx] = (face_idx * 40) as u8;
                }
            }
        }

        let texture = TextureData::new(width, height, TextureFormat::Rgba8, data, 1);
        let importer = CubemapImporter::new();
        let cubemap = importer.import(&texture, CubemapLayout::Strip).unwrap();

        // Verify each face has its expected value
        for (i, face) in CubemapFace::ALL.iter().enumerate() {
            let face_data = cubemap.face(*face);
            assert_eq!(face_data[0], (i * 40) as u8, "Face {:?} has wrong value", face);
        }
    }

    #[test]
    fn test_extract_all_faces_different_values() {
        let face_size = 2u32;
        let width = face_size * 4;
        let height = face_size * 3;
        let bpp = 4;

        // Create data where each region has a unique identifier
        let data = vec![128u8; (width * height * bpp) as usize];
        let texture = TextureData::new(width, height, TextureFormat::Rgba8, data, 1);

        let importer = CubemapImporter::new();
        let cubemap = importer.import(&texture, CubemapLayout::CrossHorizontal).unwrap();

        // All faces should be extracted
        assert_eq!(cubemap.faces.len(), 6);
        for face in CubemapFace::ALL {
            assert_eq!(
                cubemap.face(face).len(),
                (face_size * face_size * bpp) as usize
            );
        }
    }

    // ==================== Mip Generation Tests ====================

    #[test]
    fn test_mip_generation_single_face() {
        let size = 16u32;
        let data = vec![255u8; (size * size * 4) as usize];
        let mips = generate_face_mips(&data, size, TextureFormat::Rgba8);

        // Should generate 5 mip levels: 16, 8, 4, 2, 1
        assert_eq!(mips.len(), 5);

        // Verify sizes
        assert_eq!(mips[0].len(), (16 * 16 * 4) as usize); // 16x16
        assert_eq!(mips[1].len(), (8 * 8 * 4) as usize);   // 8x8
        assert_eq!(mips[2].len(), (4 * 4 * 4) as usize);   // 4x4
        assert_eq!(mips[3].len(), (2 * 2 * 4) as usize);   // 2x2
        assert_eq!(mips[4].len(), (1 * 1 * 4) as usize);   // 1x1
    }

    #[test]
    fn test_mip_generation_preserves_average() {
        let size = 4u32;
        // Create a gradient: 0, 64, 128, 192 repeated
        let mut data = vec![0u8; (size * size * 4) as usize];
        for i in 0..(size * size) as usize {
            let val = ((i % 4) * 64) as u8;
            data[i * 4] = val;
            data[i * 4 + 1] = val;
            data[i * 4 + 2] = val;
            data[i * 4 + 3] = 255;
        }

        let mips = generate_face_mips(&data, size, TextureFormat::Rgba8);

        // Final 1x1 mip should be close to average (96)
        assert!(mips.last().unwrap()[0] > 80 && mips.last().unwrap()[0] < 120);
    }

    #[test]
    fn test_cubemap_mip_generation() {
        let face_size = 8u32;
        let cubemap = Cubemap::new(face_size, TextureFormat::Rgba8, 1);

        let all_mips = generate_cubemap_mips(&cubemap);

        // Should have 4 mip levels: 8, 4, 2, 1
        assert_eq!(all_mips.len(), 4);

        // Each level should have 6 faces
        for level_mips in &all_mips {
            assert_eq!(level_mips.len(), 6);
        }
    }

    // ==================== Texture Array Tests ====================

    #[test]
    fn test_texture_array_creation() {
        let width = 64u32;
        let height = 64u32;
        let format = TextureFormat::Rgba8;

        let tex1 = TextureData::new(width, height, format, vec![100u8; (width * height * 4) as usize], 1);
        let tex2 = TextureData::new(width, height, format, vec![150u8; (width * height * 4) as usize], 1);
        let tex3 = TextureData::new(width, height, format, vec![200u8; (width * height * 4) as usize], 1);

        let array = TextureArray::from_textures(vec![tex1, tex2, tex3]).unwrap();

        assert_eq!(array.layer_count(), 3);
        assert_eq!(array.width, 64);
        assert_eq!(array.height, 64);
        assert_eq!(array.format, TextureFormat::Rgba8);
    }

    #[test]
    fn test_texture_array_add_layer() {
        let mut array = TextureArray::new(32, 32, TextureFormat::Rgba8);
        assert!(array.is_empty());

        let tex = TextureData::new(32, 32, TextureFormat::Rgba8, vec![0u8; 32 * 32 * 4], 1);
        array.add_layer(tex).unwrap();

        assert_eq!(array.layer_count(), 1);
        assert!(!array.is_empty());
    }

    #[test]
    fn test_texture_array_dimension_mismatch() {
        let mut array = TextureArray::new(64, 64, TextureFormat::Rgba8);
        let tex = TextureData::new(32, 32, TextureFormat::Rgba8, vec![0u8; 32 * 32 * 4], 1);

        let result = array.add_layer(tex);
        assert!(result.is_err());
    }

    #[test]
    fn test_texture_array_format_mismatch() {
        let mut array = TextureArray::new(64, 64, TextureFormat::Rgba8);
        let tex = TextureData::new(64, 64, TextureFormat::R8, vec![0u8; 64 * 64], 1);

        let result = array.add_layer(tex);
        assert!(result.is_err());
    }

    // ==================== Cubemap Array Tests ====================

    #[test]
    fn test_cubemap_array_creation() {
        let face_size = 32u32;
        let format = TextureFormat::Rgba8;

        let cm1 = Cubemap::new(face_size, format, 1);
        let cm2 = Cubemap::new(face_size, format, 1);

        let array = CubemapArray::from_cubemaps(vec![cm1, cm2]).unwrap();

        assert_eq!(array.len(), 2);
        assert_eq!(array.face_size, 32);
        assert_eq!(array.total_face_count(), 12);
    }

    #[test]
    fn test_cubemap_array_indexing() {
        let face_size = 4u32;
        let format = TextureFormat::Rgba8;

        let mut cm = Cubemap::new(face_size, format, 1);
        cm.faces[0] = vec![100u8; (face_size * face_size * 4) as usize];

        let array = CubemapArray::from_cubemaps(vec![cm]).unwrap();

        let face_data = array.face(0, CubemapFace::PositiveX).unwrap();
        assert_eq!(face_data[0], 100);
    }

    #[test]
    fn test_cubemap_array_to_texture_array() {
        let face_size = 2u32;
        let format = TextureFormat::Rgba8;

        let cm = Cubemap::new(face_size, format, 1);
        let array = CubemapArray::from_cubemaps(vec![cm]).unwrap();

        let tex_array = array.to_texture_array();
        assert_eq!(tex_array.layer_count(), 6); // 1 cubemap * 6 faces
    }

    // ==================== Format Validation Tests ====================

    #[test]
    fn test_cubemap_format_validation() {
        let face_size = 4u32;
        let format = TextureFormat::Rgba8;
        let cubemap = Cubemap::new(face_size, format, 1);

        assert!(cubemap.is_valid());
        assert_eq!(cubemap.face_byte_size(), (4 * 4 * 4) as usize);
    }

    #[test]
    fn test_texture_array_validation() {
        let tex1 = TextureData::new(8, 8, TextureFormat::Rgba8, vec![0u8; 8 * 8 * 4], 1);
        let tex2 = TextureData::new(8, 8, TextureFormat::Rgba8, vec![0u8; 8 * 8 * 4], 1);

        let array = TextureArray::from_textures(vec![tex1, tex2]).unwrap();
        assert!(array.is_valid());
    }

    // ==================== Round-trip Cooking Tests ====================

    #[test]
    fn test_cubemap_cook_roundtrip() {
        let face_size = 8u32;
        let format = TextureFormat::Rgba8;
        let cubemap = Cubemap::new(face_size, format, 1);

        let cooker = CubemapCooker::new().with_mips(true);
        let cooked = cooker.cook(&cubemap, TextureUsage::BaseColor).unwrap();

        assert_eq!(cooked.face_size, 8);
        assert_eq!(cooked.mip_count, 4); // 8, 4, 2, 1
        assert!(cooked.total_size() > 0);
    }

    #[test]
    fn test_texture_array_cook_roundtrip() {
        let tex1 = TextureData::new(16, 16, TextureFormat::Rgba8, vec![128u8; 16 * 16 * 4], 1);
        let tex2 = TextureData::new(16, 16, TextureFormat::Rgba8, vec![64u8; 16 * 16 * 4], 1);

        let array = TextureArray::from_textures(vec![tex1, tex2]).unwrap();
        let cooker = TextureArrayCooker::new().with_mips(true);
        let cooked = cooker.cook(&array, TextureUsage::BaseColor).unwrap();

        assert_eq!(cooked.layer_count, 2);
        assert_eq!(cooked.mip_count, 5); // 16, 8, 4, 2, 1
    }

    #[test]
    fn test_cubemap_cook_with_compression() {
        let face_size = 8u32;
        let format = TextureFormat::Rgba8;
        let cubemap = Cubemap::new(face_size, format, 1);

        let cooker = CubemapCooker::new().with_mips(true).with_compression(true);
        let cooked = cooker.cook(&cubemap, TextureUsage::BaseColor).unwrap();

        // Should produce BC7 format for base color with compression
        assert_eq!(cooked.format, GpuTextureFormat::Bc7Srgb);
    }

    // ==================== Edge Case Tests ====================

    #[test]
    fn test_cubemap_face_from_index() {
        assert_eq!(CubemapFace::from_index(0), Some(CubemapFace::PositiveX));
        assert_eq!(CubemapFace::from_index(5), Some(CubemapFace::NegativeZ));
        assert_eq!(CubemapFace::from_index(6), None);
        assert_eq!(CubemapFace::from_index(100), None);
    }

    #[test]
    fn test_face_position_consistency() {
        // Verify each face gets a unique position
        let mut positions = Vec::new();
        for face in CubemapFace::ALL {
            let pos = CubemapLayout::CrossHorizontal.face_position(face);
            assert!(!positions.contains(&pos), "Duplicate position for {:?}", face);
            positions.push(pos);
        }
    }

    #[test]
    fn test_separate_layout_import() {
        let face_size = 4u32;
        let format = TextureFormat::Rgba8;

        let faces: [TextureData; 6] = CubemapFace::ALL.map(|face| {
            let value = (face.index() * 40) as u8;
            TextureData::new(
                face_size,
                face_size,
                format,
                vec![value; (face_size * face_size * 4) as usize],
                1,
            )
        });

        let importer = CubemapImporter::new();
        let cubemap = importer.import_separate(&faces).unwrap();

        for (i, face) in CubemapFace::ALL.iter().enumerate() {
            assert_eq!(cubemap.face(*face)[0], (i * 40) as u8);
        }
    }

    #[test]
    fn test_empty_texture_array_error() {
        let result = TextureArray::from_textures(vec![]);
        assert!(matches!(result, Err(CubemapError::InvalidLayerCount(0))));
    }

    #[test]
    fn test_empty_cubemap_array_error() {
        let result = CubemapArray::from_cubemaps(vec![]);
        assert!(matches!(result, Err(CubemapError::InvalidLayerCount(0))));
    }
}
