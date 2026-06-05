//! Vertex format conversion engine for TRINITY.
//!
//! Converts glTF vertex data to engine-specific formats with support for:
//! - Interleaved, split, and compressed layouts
//! - Position compression (16-bit float, 10-10-10-2 normalized)
//! - Normal/tangent compression (8-bit SNORM, octahedral encoding)
//! - UV compression (16-bit float, 16-bit normalized)
//! - Color compression (8-bit UNORM, 10-10-10-2 HDR)
//! - Axis conversion and scale transforms
//! - Mesh merging
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::asset::vertex_format::*;
//!
//! // Configure import settings
//! let settings = ImportSettings::default()
//!     .with_scale(0.01)
//!     .with_axis_conversion(AxisConversion::YUpToZUp);
//!
//! // Convert to interleaved layout
//! let converter = VertexFormatConverter::new(settings);
//! let buffer = converter.to_interleaved(&gltf_primitive)?;
//!
//! // Or compressed layout
//! let settings = ImportSettings::default()
//!     .with_compression(CompressionSettings::aggressive());
//! let converter = VertexFormatConverter::new(settings);
//! let buffer = converter.to_compressed(&gltf_primitive)?;
//! ```

use crate::gltf::{
    ComponentType, GltfMesh, GltfPrimitive, IndexBuffer, IndexFormat,
    VertexAttribute, VertexSemantic,
};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Vertex format conversion error.
#[derive(Debug, Clone)]
pub enum VertexFormatError {
    /// Missing required attribute.
    MissingAttribute(VertexSemantic),
    /// Invalid attribute data.
    InvalidData(String),
    /// Compression error.
    CompressionError(String),
    /// Unsupported format combination.
    UnsupportedFormat(String),
    /// Vertex count mismatch between attributes.
    VertexCountMismatch { expected: usize, got: usize },
}

impl std::fmt::Display for VertexFormatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingAttribute(sem) => write!(f, "missing attribute: {:?}", sem),
            Self::InvalidData(msg) => write!(f, "invalid data: {}", msg),
            Self::CompressionError(msg) => write!(f, "compression error: {}", msg),
            Self::UnsupportedFormat(msg) => write!(f, "unsupported format: {}", msg),
            Self::VertexCountMismatch { expected, got } => {
                write!(f, "vertex count mismatch: expected {}, got {}", expected, got)
            }
        }
    }
}

impl std::error::Error for VertexFormatError {}

// ---------------------------------------------------------------------------
// Axis conversion
// ---------------------------------------------------------------------------

/// Coordinate system axis conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AxisConversion {
    /// No conversion (keep glTF Y-up right-handed).
    #[default]
    None,
    /// Convert Y-up to Z-up (Blender, 3ds Max default).
    YUpToZUp,
    /// Convert Z-up to Y-up.
    ZUpToYUp,
    /// Flip X axis (left-handed to right-handed).
    FlipX,
    /// Flip Z axis.
    FlipZ,
}

impl AxisConversion {
    /// Apply axis conversion to a position.
    #[inline]
    pub fn apply_position(self, x: f32, y: f32, z: f32) -> [f32; 3] {
        match self {
            Self::None => [x, y, z],
            Self::YUpToZUp => [x, z, -y],
            Self::ZUpToYUp => [x, -z, y],
            Self::FlipX => [-x, y, z],
            Self::FlipZ => [x, y, -z],
        }
    }

    /// Apply axis conversion to a normal/direction.
    #[inline]
    pub fn apply_normal(self, x: f32, y: f32, z: f32) -> [f32; 3] {
        // Same as position for linear transforms
        self.apply_position(x, y, z)
    }

    /// Apply axis conversion to a tangent (with handedness in w).
    #[inline]
    pub fn apply_tangent(self, x: f32, y: f32, z: f32, w: f32) -> [f32; 4] {
        let [nx, ny, nz] = self.apply_normal(x, y, z);
        // Flip handedness for axis flips
        let new_w = match self {
            Self::FlipX | Self::FlipZ => -w,
            _ => w,
        };
        [nx, ny, nz, new_w]
    }
}

// ---------------------------------------------------------------------------
// Compression formats
// ---------------------------------------------------------------------------

/// Position compression format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PositionCompression {
    /// Full 32-bit float (no compression).
    #[default]
    None,
    /// 16-bit float (half precision).
    Float16,
    /// 10-10-10-2 normalized (relative to AABB).
    Norm10_10_10_2,
}

/// Normal/tangent compression format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NormalCompression {
    /// Full 32-bit float (no compression).
    #[default]
    None,
    /// 8-bit signed normalized.
    Snorm8,
    /// Octahedral encoding (2x 16-bit).
    Octahedral,
}

/// UV compression format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UvCompression {
    /// Full 32-bit float (no compression).
    #[default]
    None,
    /// 16-bit float.
    Float16,
    /// 16-bit normalized (0-1 range).
    Unorm16,
}

/// Color compression format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorCompression {
    /// Full 32-bit float RGBA.
    #[default]
    None,
    /// 8-bit UNORM RGBA.
    Unorm8,
    /// 10-10-10-2 HDR (RGB10A2).
    Rgb10A2,
}

/// Compression settings for all vertex attributes.
#[derive(Debug, Clone, Copy, Default)]
pub struct CompressionSettings {
    /// Position compression format.
    pub position: PositionCompression,
    /// Normal compression format.
    pub normal: NormalCompression,
    /// Tangent compression format (usually same as normal).
    pub tangent: NormalCompression,
    /// UV compression format.
    pub uv: UvCompression,
    /// Color compression format.
    pub color: ColorCompression,
}

impl CompressionSettings {
    /// No compression (full precision).
    pub const fn none() -> Self {
        Self {
            position: PositionCompression::None,
            normal: NormalCompression::None,
            tangent: NormalCompression::None,
            uv: UvCompression::None,
            color: ColorCompression::None,
        }
    }

    /// Balanced compression (good quality, moderate savings).
    pub const fn balanced() -> Self {
        Self {
            position: PositionCompression::Float16,
            normal: NormalCompression::Octahedral,
            tangent: NormalCompression::Octahedral,
            uv: UvCompression::Float16,
            color: ColorCompression::Unorm8,
        }
    }

    /// Aggressive compression (smaller size, some quality loss).
    pub const fn aggressive() -> Self {
        Self {
            position: PositionCompression::Norm10_10_10_2,
            normal: NormalCompression::Octahedral,
            tangent: NormalCompression::Octahedral,
            uv: UvCompression::Unorm16,
            color: ColorCompression::Rgb10A2,
        }
    }
}

// ---------------------------------------------------------------------------
// Import settings
// ---------------------------------------------------------------------------

/// Import settings for vertex format conversion.
#[derive(Debug, Clone, Default)]
pub struct ImportSettings {
    /// Uniform scale factor applied to positions.
    pub scale: f32,
    /// Axis conversion.
    pub axis_conversion: AxisConversion,
    /// Compression settings.
    pub compression: CompressionSettings,
    /// Merge all primitives into a single mesh.
    pub merge_meshes: bool,
    /// Position AABB for normalized compression (auto-calculated if None).
    pub position_aabb: Option<([f32; 3], [f32; 3])>,
}

impl ImportSettings {
    /// Create default settings with scale 1.0 and no conversion.
    pub fn new() -> Self {
        Self {
            scale: 1.0,
            axis_conversion: AxisConversion::None,
            compression: CompressionSettings::none(),
            merge_meshes: false,
            position_aabb: None,
        }
    }

    /// Set scale factor.
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    /// Set axis conversion.
    pub fn with_axis_conversion(mut self, axis: AxisConversion) -> Self {
        self.axis_conversion = axis;
        self
    }

    /// Set compression settings.
    pub fn with_compression(mut self, compression: CompressionSettings) -> Self {
        self.compression = compression;
        self
    }

    /// Enable mesh merging.
    pub fn with_merge_meshes(mut self, merge: bool) -> Self {
        self.merge_meshes = merge;
        self
    }

    /// Set explicit position AABB for normalized compression.
    pub fn with_position_aabb(mut self, min: [f32; 3], max: [f32; 3]) -> Self {
        self.position_aabb = Some((min, max));
        self
    }
}

// ---------------------------------------------------------------------------
// Output vertex layout
// ---------------------------------------------------------------------------

/// Describes a single attribute in the output vertex format.
#[derive(Debug, Clone)]
pub struct VertexAttributeDescriptor {
    /// Semantic meaning.
    pub semantic: VertexSemantic,
    /// Byte offset within vertex (for interleaved) or buffer (for split).
    pub offset: usize,
    /// Output format.
    pub format: OutputFormat,
    /// Size in bytes.
    pub size: usize,
}

/// Output data format for an attribute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    /// 32-bit float components.
    Float32(u8),
    /// 16-bit float components.
    Float16(u8),
    /// 8-bit unsigned normalized components.
    Unorm8(u8),
    /// 8-bit signed normalized components.
    Snorm8(u8),
    /// 16-bit unsigned normalized components.
    Unorm16(u8),
    /// 16-bit signed normalized components.
    Snorm16(u8),
    /// 10-10-10-2 packed format.
    Rgb10A2,
    /// 10-10-10-2 signed normalized.
    Rgb10A2Snorm,
    /// 16-bit unsigned integer components.
    Uint16(u8),
    /// 8-bit unsigned integer components.
    Uint8(u8),
}

impl OutputFormat {
    /// Size in bytes.
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::Float32(n) => n as usize * 4,
            Self::Float16(n) => n as usize * 2,
            Self::Unorm8(n) | Self::Snorm8(n) | Self::Uint8(n) => n as usize,
            Self::Unorm16(n) | Self::Snorm16(n) | Self::Uint16(n) => n as usize * 2,
            Self::Rgb10A2 | Self::Rgb10A2Snorm => 4,
        }
    }
}

/// Describes the complete output vertex layout.
#[derive(Debug, Clone)]
pub struct VertexLayout {
    /// Attribute descriptors.
    pub attributes: Vec<VertexAttributeDescriptor>,
    /// Total vertex stride in bytes.
    pub stride: usize,
}

impl VertexLayout {
    /// Create an empty layout.
    pub fn new() -> Self {
        Self {
            attributes: Vec::new(),
            stride: 0,
        }
    }

    /// Add an attribute to the layout.
    pub fn add_attribute(&mut self, semantic: VertexSemantic, format: OutputFormat) {
        let offset = self.stride;
        let size = format.size_bytes();
        self.attributes.push(VertexAttributeDescriptor {
            semantic,
            offset,
            format,
            size,
        });
        self.stride += size;
    }

    /// Find attribute by semantic.
    pub fn find_attribute(&self, semantic: VertexSemantic) -> Option<&VertexAttributeDescriptor> {
        self.attributes.iter().find(|a| a.semantic == semantic)
    }
}

impl Default for VertexLayout {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Output buffers
// ---------------------------------------------------------------------------

/// Interleaved vertex buffer output.
#[derive(Debug, Clone)]
pub struct InterleavedBuffer {
    /// Vertex data.
    pub data: Vec<u8>,
    /// Vertex layout.
    pub layout: VertexLayout,
    /// Number of vertices.
    pub vertex_count: usize,
}

/// Split vertex buffer output (one buffer per attribute).
#[derive(Debug, Clone)]
pub struct SplitBuffers {
    /// Per-attribute buffers keyed by semantic.
    pub buffers: HashMap<VertexSemantic, Vec<u8>>,
    /// Per-attribute formats.
    pub formats: HashMap<VertexSemantic, OutputFormat>,
    /// Number of vertices.
    pub vertex_count: usize,
}

/// Compressed vertex buffer output.
#[derive(Debug, Clone)]
pub struct CompressedBuffer {
    /// Compressed vertex data.
    pub data: Vec<u8>,
    /// Vertex layout.
    pub layout: VertexLayout,
    /// Number of vertices.
    pub vertex_count: usize,
    /// Position AABB (for decoding normalized positions).
    pub position_aabb: ([f32; 3], [f32; 3]),
}

/// Merged mesh output.
#[derive(Debug, Clone)]
pub struct MergedMesh {
    /// Merged vertex buffer.
    pub vertices: InterleavedBuffer,
    /// Merged index buffer.
    pub indices: Option<MergedIndices>,
    /// Per-primitive offsets in the merged buffer.
    pub primitive_offsets: Vec<PrimitiveOffset>,
}

/// Merged index buffer.
#[derive(Debug, Clone)]
pub struct MergedIndices {
    /// Index data (always U32 after merging).
    pub data: Vec<u8>,
    /// Number of indices.
    pub count: usize,
}

/// Offset info for a merged primitive.
#[derive(Debug, Clone)]
pub struct PrimitiveOffset {
    /// Vertex offset (first vertex index).
    pub vertex_offset: u32,
    /// Vertex count.
    pub vertex_count: u32,
    /// Index offset (if indexed).
    pub index_offset: Option<u32>,
    /// Index count (if indexed).
    pub index_count: Option<u32>,
    /// Original material index.
    pub material_index: Option<usize>,
}

// ---------------------------------------------------------------------------
// Compression utilities
// ---------------------------------------------------------------------------

/// Convert f32 to f16 (half precision float).
#[inline]
pub fn f32_to_f16(value: f32) -> u16 {
    // Handle special cases
    let bits = value.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = (bits >> 23) & 0xFF;
    let mantissa = bits & 0x7FFFFF;

    // Check for NaN or Infinity
    if exp == 0xFF {
        // NaN or Infinity
        let f16_mantissa = if mantissa != 0 { 0x200 } else { 0 };
        return ((sign << 15) | 0x7C00 | f16_mantissa) as u16;
    }

    // Denormalized or zero
    if exp == 0 {
        return (sign << 15) as u16;
    }

    // Compute f16 exponent
    let f16_exp = exp as i32 - 127 + 15;

    if f16_exp <= 0 {
        // Underflow to zero or denormalized
        if f16_exp < -10 {
            return (sign << 15) as u16;
        }
        // Denormalized
        let m = (mantissa | 0x800000) >> (14 - f16_exp);
        return ((sign << 15) | m) as u16;
    }

    if f16_exp >= 31 {
        // Overflow to infinity
        return ((sign << 15) | 0x7C00) as u16;
    }

    // Normal number
    let f16_mantissa = mantissa >> 13;
    ((sign << 15) | ((f16_exp as u32) << 10) | f16_mantissa) as u16
}

/// Convert f16 to f32.
#[inline]
pub fn f16_to_f32(value: u16) -> f32 {
    let sign = ((value >> 15) & 1) as u32;
    let exp = ((value >> 10) & 0x1F) as u32;
    let mantissa = (value & 0x3FF) as u32;

    if exp == 0 {
        if mantissa == 0 {
            // Zero
            return f32::from_bits(sign << 31);
        }
        // Denormalized - normalize it
        let mut e = 1i32;
        let mut m = mantissa;
        while (m & 0x400) == 0 {
            m <<= 1;
            e -= 1;
        }
        let f32_exp = (127 - 15 + e) as u32;
        let f32_mantissa = (m & 0x3FF) << 13;
        return f32::from_bits((sign << 31) | (f32_exp << 23) | f32_mantissa);
    }

    if exp == 31 {
        // NaN or Infinity
        let f32_mantissa = if mantissa != 0 { 0x400000 } else { 0 };
        return f32::from_bits((sign << 31) | 0x7F800000 | f32_mantissa);
    }

    // Normal number
    let f32_exp = (exp as i32 + 127 - 15) as u32;
    let f32_mantissa = mantissa << 13;
    f32::from_bits((sign << 31) | (f32_exp << 23) | f32_mantissa)
}

/// Encode a unit normal to octahedral representation (2x 16-bit signed).
///
/// Reference: "Survey of Efficient Representations for Independent Unit Vectors"
/// Cigolle et al., Journal of Computer Graphics Techniques, 2014
#[inline]
pub fn encode_octahedral(x: f32, y: f32, z: f32) -> [i16; 2] {
    // Project onto octahedron
    let l1_norm = x.abs() + y.abs() + z.abs();
    let mut oct_x = x / l1_norm;
    let mut oct_y = y / l1_norm;

    // Reflect the folds of the lower hemisphere over the diagonals
    if z < 0.0 {
        let temp_x = oct_x;
        oct_x = (1.0 - oct_y.abs()) * if temp_x >= 0.0 { 1.0 } else { -1.0 };
        oct_y = (1.0 - temp_x.abs()) * if oct_y >= 0.0 { 1.0 } else { -1.0 };
    }

    // Encode to 16-bit signed normalized
    let encode = |v: f32| -> i16 {
        (v.clamp(-1.0, 1.0) * 32767.0).round() as i16
    };

    [encode(oct_x), encode(oct_y)]
}

/// Decode octahedral representation to unit normal.
#[inline]
pub fn decode_octahedral(oct: [i16; 2]) -> [f32; 3] {
    let decode = |v: i16| -> f32 { v as f32 / 32767.0 };

    let oct_x = decode(oct[0]);
    let oct_y = decode(oct[1]);

    let mut x = oct_x;
    let mut y = oct_y;
    let z = 1.0 - x.abs() - y.abs();

    if z < 0.0 {
        let temp_x = x;
        x = (1.0 - y.abs()) * if temp_x >= 0.0 { 1.0 } else { -1.0 };
        y = (1.0 - temp_x.abs()) * if y >= 0.0 { 1.0 } else { -1.0 };
    }

    // Normalize
    let len = (x * x + y * y + z * z).sqrt();
    if len > 0.0 {
        [x / len, y / len, z / len]
    } else {
        [0.0, 0.0, 1.0]
    }
}

/// Pack a normalized vector to 10-10-10-2 format.
#[inline]
pub fn pack_10_10_10_2(x: f32, y: f32, z: f32, w: f32) -> u32 {
    let pack_10 = |v: f32| -> u32 {
        ((v.clamp(-1.0, 1.0) * 511.0 + 512.0).round() as u32) & 0x3FF
    };
    let pack_2 = |v: f32| -> u32 {
        ((v.clamp(-1.0, 1.0) * 1.0 + 2.0).round() as u32) & 0x3
    };

    pack_10(x) | (pack_10(y) << 10) | (pack_10(z) << 20) | (pack_2(w) << 30)
}

/// Unpack 10-10-10-2 format to normalized vector.
#[inline]
pub fn unpack_10_10_10_2(packed: u32) -> [f32; 4] {
    let unpack_10 = |v: u32| -> f32 {
        ((v & 0x3FF) as f32 - 512.0) / 511.0
    };
    let unpack_2 = |v: u32| -> f32 {
        ((v & 0x3) as f32 - 2.0) / 1.0
    };

    [
        unpack_10(packed),
        unpack_10(packed >> 10),
        unpack_10(packed >> 20),
        unpack_2(packed >> 30),
    ]
}

/// Pack a position to 10-10-10-2 normalized format (relative to AABB).
#[inline]
pub fn pack_position_10_10_10_2(
    x: f32,
    y: f32,
    z: f32,
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
) -> u32 {
    let normalize = |v: f32, min: f32, max: f32| -> f32 {
        if max - min > f32::EPSILON {
            (v - min) / (max - min) * 2.0 - 1.0
        } else {
            0.0
        }
    };

    let nx = normalize(x, aabb_min[0], aabb_max[0]);
    let ny = normalize(y, aabb_min[1], aabb_max[1]);
    let nz = normalize(z, aabb_min[2], aabb_max[2]);

    pack_10_10_10_2(nx, ny, nz, 0.0)
}

/// Unpack 10-10-10-2 position (relative to AABB).
#[inline]
pub fn unpack_position_10_10_10_2(
    packed: u32,
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
) -> [f32; 3] {
    let [nx, ny, nz, _] = unpack_10_10_10_2(packed);

    let denormalize = |v: f32, min: f32, max: f32| -> f32 {
        (v + 1.0) * 0.5 * (max - min) + min
    };

    [
        denormalize(nx, aabb_min[0], aabb_max[0]),
        denormalize(ny, aabb_min[1], aabb_max[1]),
        denormalize(nz, aabb_min[2], aabb_max[2]),
    ]
}

/// Convert f32 to 8-bit signed normalized.
#[inline]
pub fn f32_to_snorm8(value: f32) -> i8 {
    (value.clamp(-1.0, 1.0) * 127.0).round() as i8
}

/// Convert 8-bit signed normalized to f32.
#[inline]
pub fn snorm8_to_f32(value: i8) -> f32 {
    value as f32 / 127.0
}

/// Convert f32 to 8-bit unsigned normalized.
#[inline]
pub fn f32_to_unorm8(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// Convert 8-bit unsigned normalized to f32.
#[inline]
pub fn unorm8_to_f32(value: u8) -> f32 {
    value as f32 / 255.0
}

/// Convert f32 to 16-bit unsigned normalized.
#[inline]
pub fn f32_to_unorm16(value: f32) -> u16 {
    (value.clamp(0.0, 1.0) * 65535.0).round() as u16
}

/// Convert 16-bit unsigned normalized to f32.
#[inline]
pub fn unorm16_to_f32(value: u16) -> f32 {
    value as f32 / 65535.0
}

// ---------------------------------------------------------------------------
// Vertex data extraction
// ---------------------------------------------------------------------------

/// Extract f32 components from vertex attribute data.
fn extract_f32_components(
    attr: &VertexAttribute,
    vertex_idx: usize,
) -> Result<Vec<f32>, VertexFormatError> {
    let comp_count = attr.attribute_type.component_count();
    let elem_size = attr.component_type.size_bytes();
    let total_size = comp_count * elem_size;
    let offset = vertex_idx * total_size;

    if offset + total_size > attr.data.len() {
        return Err(VertexFormatError::InvalidData(format!(
            "vertex {} out of bounds",
            vertex_idx
        )));
    }

    let bytes = &attr.data[offset..offset + total_size];
    let mut result = Vec::with_capacity(comp_count);

    for i in 0..comp_count {
        let comp_offset = i * elem_size;
        let value = match attr.component_type {
            ComponentType::F32 => f32::from_le_bytes([
                bytes[comp_offset],
                bytes[comp_offset + 1],
                bytes[comp_offset + 2],
                bytes[comp_offset + 3],
            ]),
            ComponentType::I8 => bytes[comp_offset] as i8 as f32 / 127.0,
            ComponentType::U8 => bytes[comp_offset] as f32 / 255.0,
            ComponentType::I16 => {
                let v = i16::from_le_bytes([bytes[comp_offset], bytes[comp_offset + 1]]);
                v as f32 / 32767.0
            }
            ComponentType::U16 => {
                let v = u16::from_le_bytes([bytes[comp_offset], bytes[comp_offset + 1]]);
                v as f32 / 65535.0
            }
            ComponentType::U32 => {
                let v = u32::from_le_bytes([
                    bytes[comp_offset],
                    bytes[comp_offset + 1],
                    bytes[comp_offset + 2],
                    bytes[comp_offset + 3],
                ]);
                v as f32
            }
        };
        result.push(value);
    }

    Ok(result)
}

/// Extract u16 components from vertex attribute data (for joints).
fn extract_u16_components(
    attr: &VertexAttribute,
    vertex_idx: usize,
) -> Result<Vec<u16>, VertexFormatError> {
    let comp_count = attr.attribute_type.component_count();
    let elem_size = attr.component_type.size_bytes();
    let total_size = comp_count * elem_size;
    let offset = vertex_idx * total_size;

    if offset + total_size > attr.data.len() {
        return Err(VertexFormatError::InvalidData(format!(
            "vertex {} out of bounds",
            vertex_idx
        )));
    }

    let bytes = &attr.data[offset..offset + total_size];
    let mut result = Vec::with_capacity(comp_count);

    for i in 0..comp_count {
        let comp_offset = i * elem_size;
        let value = match attr.component_type {
            ComponentType::U8 => bytes[comp_offset] as u16,
            ComponentType::U16 => {
                u16::from_le_bytes([bytes[comp_offset], bytes[comp_offset + 1]])
            }
            ComponentType::U32 => {
                let v = u32::from_le_bytes([
                    bytes[comp_offset],
                    bytes[comp_offset + 1],
                    bytes[comp_offset + 2],
                    bytes[comp_offset + 3],
                ]);
                v as u16
            }
            _ => {
                return Err(VertexFormatError::UnsupportedFormat(
                    "joints must be integer type".into(),
                ))
            }
        };
        result.push(value);
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Vertex format converter
// ---------------------------------------------------------------------------

/// Vertex format converter.
pub struct VertexFormatConverter {
    settings: ImportSettings,
}

impl VertexFormatConverter {
    /// Create a new converter with the given settings.
    pub fn new(settings: ImportSettings) -> Self {
        Self { settings }
    }

    /// Create with default settings.
    pub fn with_defaults() -> Self {
        Self::new(ImportSettings::new())
    }

    /// Get the import settings.
    pub fn settings(&self) -> &ImportSettings {
        &self.settings
    }

    /// Calculate the AABB for positions in a primitive.
    pub fn calculate_aabb(&self, primitive: &GltfPrimitive) -> Result<([f32; 3], [f32; 3]), VertexFormatError> {
        let pos_attr = primitive
            .attributes
            .get(&VertexSemantic::Position)
            .ok_or(VertexFormatError::MissingAttribute(VertexSemantic::Position))?;

        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];

        for i in 0..pos_attr.count {
            let comps = extract_f32_components(pos_attr, i)?;
            if comps.len() >= 3 {
                let [x, y, z] = self.settings.axis_conversion.apply_position(
                    comps[0] * self.settings.scale,
                    comps[1] * self.settings.scale,
                    comps[2] * self.settings.scale,
                );
                min[0] = min[0].min(x);
                min[1] = min[1].min(y);
                min[2] = min[2].min(z);
                max[0] = max[0].max(x);
                max[1] = max[1].max(y);
                max[2] = max[2].max(z);
            }
        }

        Ok((min, max))
    }

    /// Convert a primitive to interleaved vertex format.
    pub fn to_interleaved(
        &self,
        primitive: &GltfPrimitive,
    ) -> Result<InterleavedBuffer, VertexFormatError> {
        let pos_attr = primitive
            .attributes
            .get(&VertexSemantic::Position)
            .ok_or(VertexFormatError::MissingAttribute(VertexSemantic::Position))?;

        let vertex_count = pos_attr.count;

        // Build layout based on available attributes
        let mut layout = VertexLayout::new();

        // Position is always present
        layout.add_attribute(VertexSemantic::Position, OutputFormat::Float32(3));

        // Add other attributes if present
        if primitive.attributes.contains_key(&VertexSemantic::Normal) {
            layout.add_attribute(VertexSemantic::Normal, OutputFormat::Float32(3));
        }
        if primitive.attributes.contains_key(&VertexSemantic::Tangent) {
            layout.add_attribute(VertexSemantic::Tangent, OutputFormat::Float32(4));
        }
        if primitive.attributes.contains_key(&VertexSemantic::TexCoord0) {
            layout.add_attribute(VertexSemantic::TexCoord0, OutputFormat::Float32(2));
        }
        if primitive.attributes.contains_key(&VertexSemantic::TexCoord1) {
            layout.add_attribute(VertexSemantic::TexCoord1, OutputFormat::Float32(2));
        }
        if primitive.attributes.contains_key(&VertexSemantic::Color0) {
            layout.add_attribute(VertexSemantic::Color0, OutputFormat::Float32(4));
        }
        if primitive.attributes.contains_key(&VertexSemantic::Joints0) {
            layout.add_attribute(VertexSemantic::Joints0, OutputFormat::Uint16(4));
        }
        if primitive.attributes.contains_key(&VertexSemantic::Weights0) {
            layout.add_attribute(VertexSemantic::Weights0, OutputFormat::Float32(4));
        }

        // Allocate buffer
        let mut data = vec![0u8; vertex_count * layout.stride];

        // Fill buffer
        for vertex_idx in 0..vertex_count {
            let vertex_offset = vertex_idx * layout.stride;

            for attr_desc in &layout.attributes {
                let attr_offset = vertex_offset + attr_desc.offset;

                match attr_desc.semantic {
                    VertexSemantic::Position => {
                        let comps = extract_f32_components(pos_attr, vertex_idx)?;
                        let [x, y, z] = self.settings.axis_conversion.apply_position(
                            comps[0] * self.settings.scale,
                            comps[1] * self.settings.scale,
                            comps[2] * self.settings.scale,
                        );
                        data[attr_offset..attr_offset + 4].copy_from_slice(&x.to_le_bytes());
                        data[attr_offset + 4..attr_offset + 8].copy_from_slice(&y.to_le_bytes());
                        data[attr_offset + 8..attr_offset + 12].copy_from_slice(&z.to_le_bytes());
                    }
                    VertexSemantic::Normal => {
                        if let Some(attr) = primitive.attributes.get(&VertexSemantic::Normal) {
                            let comps = extract_f32_components(attr, vertex_idx)?;
                            let [x, y, z] = self.settings.axis_conversion.apply_normal(
                                comps[0], comps[1], comps[2],
                            );
                            data[attr_offset..attr_offset + 4].copy_from_slice(&x.to_le_bytes());
                            data[attr_offset + 4..attr_offset + 8].copy_from_slice(&y.to_le_bytes());
                            data[attr_offset + 8..attr_offset + 12].copy_from_slice(&z.to_le_bytes());
                        }
                    }
                    VertexSemantic::Tangent => {
                        if let Some(attr) = primitive.attributes.get(&VertexSemantic::Tangent) {
                            let comps = extract_f32_components(attr, vertex_idx)?;
                            let w = if comps.len() > 3 { comps[3] } else { 1.0 };
                            let [x, y, z, w] = self.settings.axis_conversion.apply_tangent(
                                comps[0], comps[1], comps[2], w,
                            );
                            data[attr_offset..attr_offset + 4].copy_from_slice(&x.to_le_bytes());
                            data[attr_offset + 4..attr_offset + 8].copy_from_slice(&y.to_le_bytes());
                            data[attr_offset + 8..attr_offset + 12].copy_from_slice(&z.to_le_bytes());
                            data[attr_offset + 12..attr_offset + 16].copy_from_slice(&w.to_le_bytes());
                        }
                    }
                    VertexSemantic::TexCoord0 | VertexSemantic::TexCoord1 => {
                        if let Some(attr) = primitive.attributes.get(&attr_desc.semantic) {
                            let comps = extract_f32_components(attr, vertex_idx)?;
                            data[attr_offset..attr_offset + 4].copy_from_slice(&comps[0].to_le_bytes());
                            data[attr_offset + 4..attr_offset + 8].copy_from_slice(&comps[1].to_le_bytes());
                        }
                    }
                    VertexSemantic::Color0 => {
                        if let Some(attr) = primitive.attributes.get(&VertexSemantic::Color0) {
                            let comps = extract_f32_components(attr, vertex_idx)?;
                            let r = comps.first().copied().unwrap_or(1.0);
                            let g = comps.get(1).copied().unwrap_or(1.0);
                            let b = comps.get(2).copied().unwrap_or(1.0);
                            let a = comps.get(3).copied().unwrap_or(1.0);
                            data[attr_offset..attr_offset + 4].copy_from_slice(&r.to_le_bytes());
                            data[attr_offset + 4..attr_offset + 8].copy_from_slice(&g.to_le_bytes());
                            data[attr_offset + 8..attr_offset + 12].copy_from_slice(&b.to_le_bytes());
                            data[attr_offset + 12..attr_offset + 16].copy_from_slice(&a.to_le_bytes());
                        }
                    }
                    VertexSemantic::Joints0 => {
                        if let Some(attr) = primitive.attributes.get(&VertexSemantic::Joints0) {
                            let joints = extract_u16_components(attr, vertex_idx)?;
                            for (i, &j) in joints.iter().take(4).enumerate() {
                                let off = attr_offset + i * 2;
                                data[off..off + 2].copy_from_slice(&j.to_le_bytes());
                            }
                        }
                    }
                    VertexSemantic::Weights0 => {
                        if let Some(attr) = primitive.attributes.get(&VertexSemantic::Weights0) {
                            let comps = extract_f32_components(attr, vertex_idx)?;
                            for (i, &w) in comps.iter().take(4).enumerate() {
                                let off = attr_offset + i * 4;
                                data[off..off + 4].copy_from_slice(&w.to_le_bytes());
                            }
                        }
                    }
                }
            }
        }

        Ok(InterleavedBuffer {
            data,
            layout,
            vertex_count,
        })
    }

    /// Convert a primitive to split vertex buffers (one per attribute).
    pub fn to_split(&self, primitive: &GltfPrimitive) -> Result<SplitBuffers, VertexFormatError> {
        let pos_attr = primitive
            .attributes
            .get(&VertexSemantic::Position)
            .ok_or(VertexFormatError::MissingAttribute(VertexSemantic::Position))?;

        let vertex_count = pos_attr.count;
        let mut buffers = HashMap::new();
        let mut formats = HashMap::new();

        // Process each attribute
        for (&semantic, attr) in &primitive.attributes {
            if attr.count != vertex_count {
                return Err(VertexFormatError::VertexCountMismatch {
                    expected: vertex_count,
                    got: attr.count,
                });
            }

            let (format, data) = self.convert_attribute_split(semantic, attr)?;
            buffers.insert(semantic, data);
            formats.insert(semantic, format);
        }

        Ok(SplitBuffers {
            buffers,
            formats,
            vertex_count,
        })
    }

    /// Convert a single attribute for split layout.
    fn convert_attribute_split(
        &self,
        semantic: VertexSemantic,
        attr: &VertexAttribute,
    ) -> Result<(OutputFormat, Vec<u8>), VertexFormatError> {
        match semantic {
            VertexSemantic::Position => {
                let format = OutputFormat::Float32(3);
                let mut data = Vec::with_capacity(attr.count * 12);
                for i in 0..attr.count {
                    let comps = extract_f32_components(attr, i)?;
                    let [x, y, z] = self.settings.axis_conversion.apply_position(
                        comps[0] * self.settings.scale,
                        comps[1] * self.settings.scale,
                        comps[2] * self.settings.scale,
                    );
                    data.extend_from_slice(&x.to_le_bytes());
                    data.extend_from_slice(&y.to_le_bytes());
                    data.extend_from_slice(&z.to_le_bytes());
                }
                Ok((format, data))
            }
            VertexSemantic::Normal => {
                let format = OutputFormat::Float32(3);
                let mut data = Vec::with_capacity(attr.count * 12);
                for i in 0..attr.count {
                    let comps = extract_f32_components(attr, i)?;
                    let [x, y, z] = self.settings.axis_conversion.apply_normal(
                        comps[0], comps[1], comps[2],
                    );
                    data.extend_from_slice(&x.to_le_bytes());
                    data.extend_from_slice(&y.to_le_bytes());
                    data.extend_from_slice(&z.to_le_bytes());
                }
                Ok((format, data))
            }
            VertexSemantic::Tangent => {
                let format = OutputFormat::Float32(4);
                let mut data = Vec::with_capacity(attr.count * 16);
                for i in 0..attr.count {
                    let comps = extract_f32_components(attr, i)?;
                    let w = if comps.len() > 3 { comps[3] } else { 1.0 };
                    let [x, y, z, w] = self.settings.axis_conversion.apply_tangent(
                        comps[0], comps[1], comps[2], w,
                    );
                    data.extend_from_slice(&x.to_le_bytes());
                    data.extend_from_slice(&y.to_le_bytes());
                    data.extend_from_slice(&z.to_le_bytes());
                    data.extend_from_slice(&w.to_le_bytes());
                }
                Ok((format, data))
            }
            VertexSemantic::TexCoord0 | VertexSemantic::TexCoord1 => {
                let format = OutputFormat::Float32(2);
                let mut data = Vec::with_capacity(attr.count * 8);
                for i in 0..attr.count {
                    let comps = extract_f32_components(attr, i)?;
                    data.extend_from_slice(&comps[0].to_le_bytes());
                    data.extend_from_slice(&comps[1].to_le_bytes());
                }
                Ok((format, data))
            }
            VertexSemantic::Color0 => {
                let format = OutputFormat::Float32(4);
                let mut data = Vec::with_capacity(attr.count * 16);
                for i in 0..attr.count {
                    let comps = extract_f32_components(attr, i)?;
                    let r = comps.first().copied().unwrap_or(1.0);
                    let g = comps.get(1).copied().unwrap_or(1.0);
                    let b = comps.get(2).copied().unwrap_or(1.0);
                    let a = comps.get(3).copied().unwrap_or(1.0);
                    data.extend_from_slice(&r.to_le_bytes());
                    data.extend_from_slice(&g.to_le_bytes());
                    data.extend_from_slice(&b.to_le_bytes());
                    data.extend_from_slice(&a.to_le_bytes());
                }
                Ok((format, data))
            }
            VertexSemantic::Joints0 => {
                let format = OutputFormat::Uint16(4);
                let mut data = Vec::with_capacity(attr.count * 8);
                for i in 0..attr.count {
                    let joints = extract_u16_components(attr, i)?;
                    for j in joints.iter().take(4) {
                        data.extend_from_slice(&j.to_le_bytes());
                    }
                    // Pad if less than 4 joints
                    for _ in joints.len()..4 {
                        data.extend_from_slice(&0u16.to_le_bytes());
                    }
                }
                Ok((format, data))
            }
            VertexSemantic::Weights0 => {
                let format = OutputFormat::Float32(4);
                let mut data = Vec::with_capacity(attr.count * 16);
                for i in 0..attr.count {
                    let comps = extract_f32_components(attr, i)?;
                    for j in 0..4 {
                        let w = comps.get(j).copied().unwrap_or(0.0);
                        data.extend_from_slice(&w.to_le_bytes());
                    }
                }
                Ok((format, data))
            }
        }
    }

    /// Convert a primitive to compressed vertex format.
    pub fn to_compressed(
        &self,
        primitive: &GltfPrimitive,
    ) -> Result<CompressedBuffer, VertexFormatError> {
        let pos_attr = primitive
            .attributes
            .get(&VertexSemantic::Position)
            .ok_or(VertexFormatError::MissingAttribute(VertexSemantic::Position))?;

        let vertex_count = pos_attr.count;

        // Calculate AABB for position compression
        let position_aabb = if let Some(aabb) = self.settings.position_aabb {
            aabb
        } else {
            self.calculate_aabb(primitive)?
        };

        // Build compressed layout
        let mut layout = VertexLayout::new();
        let comp = &self.settings.compression;

        // Position format
        match comp.position {
            PositionCompression::None => {
                layout.add_attribute(VertexSemantic::Position, OutputFormat::Float32(3));
            }
            PositionCompression::Float16 => {
                layout.add_attribute(VertexSemantic::Position, OutputFormat::Float16(3));
            }
            PositionCompression::Norm10_10_10_2 => {
                layout.add_attribute(VertexSemantic::Position, OutputFormat::Rgb10A2);
            }
        }

        // Normal format
        if primitive.attributes.contains_key(&VertexSemantic::Normal) {
            match comp.normal {
                NormalCompression::None => {
                    layout.add_attribute(VertexSemantic::Normal, OutputFormat::Float32(3));
                }
                NormalCompression::Snorm8 => {
                    // Pad to 4 bytes for alignment
                    layout.add_attribute(VertexSemantic::Normal, OutputFormat::Snorm8(4));
                }
                NormalCompression::Octahedral => {
                    layout.add_attribute(VertexSemantic::Normal, OutputFormat::Snorm16(2));
                }
            }
        }

        // Tangent format
        if primitive.attributes.contains_key(&VertexSemantic::Tangent) {
            match comp.tangent {
                NormalCompression::None => {
                    layout.add_attribute(VertexSemantic::Tangent, OutputFormat::Float32(4));
                }
                NormalCompression::Snorm8 => {
                    layout.add_attribute(VertexSemantic::Tangent, OutputFormat::Snorm8(4));
                }
                NormalCompression::Octahedral => {
                    // Octahedral + handedness byte (4 bytes total)
                    layout.add_attribute(VertexSemantic::Tangent, OutputFormat::Snorm16(2));
                }
            }
        }

        // UV formats
        for sem in [VertexSemantic::TexCoord0, VertexSemantic::TexCoord1] {
            if primitive.attributes.contains_key(&sem) {
                match comp.uv {
                    UvCompression::None => {
                        layout.add_attribute(sem, OutputFormat::Float32(2));
                    }
                    UvCompression::Float16 => {
                        layout.add_attribute(sem, OutputFormat::Float16(2));
                    }
                    UvCompression::Unorm16 => {
                        layout.add_attribute(sem, OutputFormat::Unorm16(2));
                    }
                }
            }
        }

        // Color format
        if primitive.attributes.contains_key(&VertexSemantic::Color0) {
            match comp.color {
                ColorCompression::None => {
                    layout.add_attribute(VertexSemantic::Color0, OutputFormat::Float32(4));
                }
                ColorCompression::Unorm8 => {
                    layout.add_attribute(VertexSemantic::Color0, OutputFormat::Unorm8(4));
                }
                ColorCompression::Rgb10A2 => {
                    layout.add_attribute(VertexSemantic::Color0, OutputFormat::Rgb10A2);
                }
            }
        }

        // Joints/weights always use fixed formats
        if primitive.attributes.contains_key(&VertexSemantic::Joints0) {
            layout.add_attribute(VertexSemantic::Joints0, OutputFormat::Uint16(4));
        }
        if primitive.attributes.contains_key(&VertexSemantic::Weights0) {
            layout.add_attribute(VertexSemantic::Weights0, OutputFormat::Unorm16(4));
        }

        // Allocate buffer
        let mut data = vec![0u8; vertex_count * layout.stride];

        // Fill buffer with compressed data
        for vertex_idx in 0..vertex_count {
            let vertex_offset = vertex_idx * layout.stride;

            for attr_desc in &layout.attributes {
                let attr_offset = vertex_offset + attr_desc.offset;

                self.write_compressed_attribute(
                    primitive,
                    vertex_idx,
                    attr_desc,
                    &mut data[attr_offset..],
                    position_aabb,
                )?;
            }
        }

        Ok(CompressedBuffer {
            data,
            layout,
            vertex_count,
            position_aabb,
        })
    }

    /// Write a compressed attribute to the buffer.
    fn write_compressed_attribute(
        &self,
        primitive: &GltfPrimitive,
        vertex_idx: usize,
        attr_desc: &VertexAttributeDescriptor,
        dest: &mut [u8],
        position_aabb: ([f32; 3], [f32; 3]),
    ) -> Result<(), VertexFormatError> {
        let comp = &self.settings.compression;

        match attr_desc.semantic {
            VertexSemantic::Position => {
                let attr = &primitive.attributes[&VertexSemantic::Position];
                let comps = extract_f32_components(attr, vertex_idx)?;
                let [x, y, z] = self.settings.axis_conversion.apply_position(
                    comps[0] * self.settings.scale,
                    comps[1] * self.settings.scale,
                    comps[2] * self.settings.scale,
                );

                match comp.position {
                    PositionCompression::None => {
                        dest[0..4].copy_from_slice(&x.to_le_bytes());
                        dest[4..8].copy_from_slice(&y.to_le_bytes());
                        dest[8..12].copy_from_slice(&z.to_le_bytes());
                    }
                    PositionCompression::Float16 => {
                        dest[0..2].copy_from_slice(&f32_to_f16(x).to_le_bytes());
                        dest[2..4].copy_from_slice(&f32_to_f16(y).to_le_bytes());
                        dest[4..6].copy_from_slice(&f32_to_f16(z).to_le_bytes());
                    }
                    PositionCompression::Norm10_10_10_2 => {
                        let packed = pack_position_10_10_10_2(
                            x, y, z, position_aabb.0, position_aabb.1,
                        );
                        dest[0..4].copy_from_slice(&packed.to_le_bytes());
                    }
                }
            }
            VertexSemantic::Normal => {
                let attr = &primitive.attributes[&VertexSemantic::Normal];
                let comps = extract_f32_components(attr, vertex_idx)?;
                let [x, y, z] = self.settings.axis_conversion.apply_normal(
                    comps[0], comps[1], comps[2],
                );

                match comp.normal {
                    NormalCompression::None => {
                        dest[0..4].copy_from_slice(&x.to_le_bytes());
                        dest[4..8].copy_from_slice(&y.to_le_bytes());
                        dest[8..12].copy_from_slice(&z.to_le_bytes());
                    }
                    NormalCompression::Snorm8 => {
                        dest[0] = f32_to_snorm8(x) as u8;
                        dest[1] = f32_to_snorm8(y) as u8;
                        dest[2] = f32_to_snorm8(z) as u8;
                        dest[3] = 0; // Padding
                    }
                    NormalCompression::Octahedral => {
                        let [ox, oy] = encode_octahedral(x, y, z);
                        dest[0..2].copy_from_slice(&ox.to_le_bytes());
                        dest[2..4].copy_from_slice(&oy.to_le_bytes());
                    }
                }
            }
            VertexSemantic::Tangent => {
                let attr = &primitive.attributes[&VertexSemantic::Tangent];
                let comps = extract_f32_components(attr, vertex_idx)?;
                let w = if comps.len() > 3 { comps[3] } else { 1.0 };
                let [x, y, z, w] = self.settings.axis_conversion.apply_tangent(
                    comps[0], comps[1], comps[2], w,
                );

                match comp.tangent {
                    NormalCompression::None => {
                        dest[0..4].copy_from_slice(&x.to_le_bytes());
                        dest[4..8].copy_from_slice(&y.to_le_bytes());
                        dest[8..12].copy_from_slice(&z.to_le_bytes());
                        dest[12..16].copy_from_slice(&w.to_le_bytes());
                    }
                    NormalCompression::Snorm8 => {
                        dest[0] = f32_to_snorm8(x) as u8;
                        dest[1] = f32_to_snorm8(y) as u8;
                        dest[2] = f32_to_snorm8(z) as u8;
                        dest[3] = f32_to_snorm8(w) as u8;
                    }
                    NormalCompression::Octahedral => {
                        // For tangent, we encode the direction and store handedness separately
                        let [ox, oy] = encode_octahedral(x, y, z);
                        dest[0..2].copy_from_slice(&ox.to_le_bytes());
                        // Encode handedness in the upper byte of the second component
                        let oy_with_w = (oy as i32) | (if w < 0.0 { 0x8000 } else { 0 }) as i32;
                        dest[2..4].copy_from_slice(&(oy_with_w as i16).to_le_bytes());
                    }
                }
            }
            VertexSemantic::TexCoord0 | VertexSemantic::TexCoord1 => {
                let attr = &primitive.attributes[&attr_desc.semantic];
                let comps = extract_f32_components(attr, vertex_idx)?;
                let u = comps[0];
                let v = comps[1];

                match comp.uv {
                    UvCompression::None => {
                        dest[0..4].copy_from_slice(&u.to_le_bytes());
                        dest[4..8].copy_from_slice(&v.to_le_bytes());
                    }
                    UvCompression::Float16 => {
                        dest[0..2].copy_from_slice(&f32_to_f16(u).to_le_bytes());
                        dest[2..4].copy_from_slice(&f32_to_f16(v).to_le_bytes());
                    }
                    UvCompression::Unorm16 => {
                        // Wrap to 0-1 range for normalized encoding
                        let u_wrapped = u.fract();
                        let v_wrapped = v.fract();
                        let u_norm = if u_wrapped < 0.0 { u_wrapped + 1.0 } else { u_wrapped };
                        let v_norm = if v_wrapped < 0.0 { v_wrapped + 1.0 } else { v_wrapped };
                        dest[0..2].copy_from_slice(&f32_to_unorm16(u_norm).to_le_bytes());
                        dest[2..4].copy_from_slice(&f32_to_unorm16(v_norm).to_le_bytes());
                    }
                }
            }
            VertexSemantic::Color0 => {
                let attr = &primitive.attributes[&VertexSemantic::Color0];
                let comps = extract_f32_components(attr, vertex_idx)?;
                let r = comps.first().copied().unwrap_or(1.0);
                let g = comps.get(1).copied().unwrap_or(1.0);
                let b = comps.get(2).copied().unwrap_or(1.0);
                let a = comps.get(3).copied().unwrap_or(1.0);

                match comp.color {
                    ColorCompression::None => {
                        dest[0..4].copy_from_slice(&r.to_le_bytes());
                        dest[4..8].copy_from_slice(&g.to_le_bytes());
                        dest[8..12].copy_from_slice(&b.to_le_bytes());
                        dest[12..16].copy_from_slice(&a.to_le_bytes());
                    }
                    ColorCompression::Unorm8 => {
                        dest[0] = f32_to_unorm8(r);
                        dest[1] = f32_to_unorm8(g);
                        dest[2] = f32_to_unorm8(b);
                        dest[3] = f32_to_unorm8(a);
                    }
                    ColorCompression::Rgb10A2 => {
                        // Map 0-1 to signed range for pack function, then shift
                        let packed = pack_10_10_10_2(
                            r * 2.0 - 1.0,
                            g * 2.0 - 1.0,
                            b * 2.0 - 1.0,
                            a * 2.0 - 1.0,
                        );
                        dest[0..4].copy_from_slice(&packed.to_le_bytes());
                    }
                }
            }
            VertexSemantic::Joints0 => {
                let attr = &primitive.attributes[&VertexSemantic::Joints0];
                let joints = extract_u16_components(attr, vertex_idx)?;
                for (i, &j) in joints.iter().take(4).enumerate() {
                    dest[i * 2..(i + 1) * 2].copy_from_slice(&j.to_le_bytes());
                }
                for i in joints.len()..4 {
                    dest[i * 2..(i + 1) * 2].copy_from_slice(&0u16.to_le_bytes());
                }
            }
            VertexSemantic::Weights0 => {
                let attr = &primitive.attributes[&VertexSemantic::Weights0];
                let comps = extract_f32_components(attr, vertex_idx)?;
                for i in 0..4 {
                    let w = comps.get(i).copied().unwrap_or(0.0);
                    dest[i * 2..(i + 1) * 2].copy_from_slice(&f32_to_unorm16(w).to_le_bytes());
                }
            }
        }

        Ok(())
    }

    /// Merge multiple meshes into a single mesh.
    pub fn merge_meshes(&self, meshes: &[GltfMesh]) -> Result<MergedMesh, VertexFormatError> {
        if meshes.is_empty() {
            return Err(VertexFormatError::InvalidData("no meshes to merge".into()));
        }

        // Collect all primitives
        let mut all_primitives: Vec<(&GltfPrimitive, Option<usize>)> = Vec::new();
        for mesh in meshes {
            for prim in &mesh.primitives {
                all_primitives.push((prim, prim.material_index));
            }
        }

        if all_primitives.is_empty() {
            return Err(VertexFormatError::InvalidData("no primitives to merge".into()));
        }

        // Determine common attributes (intersection of all primitives)
        let first_prim = all_primitives[0].0;
        let mut common_attrs: Vec<VertexSemantic> = first_prim
            .attributes
            .keys()
            .copied()
            .collect();

        for (prim, _) in &all_primitives[1..] {
            common_attrs.retain(|sem| prim.attributes.contains_key(sem));
        }

        if !common_attrs.contains(&VertexSemantic::Position) {
            return Err(VertexFormatError::MissingAttribute(VertexSemantic::Position));
        }

        // Calculate total counts
        let mut total_vertices = 0usize;
        let mut total_indices = 0usize;
        let mut has_indices = false;

        for (prim, _) in &all_primitives {
            let pos_attr = prim.attributes.get(&VertexSemantic::Position).unwrap();
            total_vertices += pos_attr.count;
            if let Some(idx) = &prim.indices {
                total_indices += idx.count;
                has_indices = true;
            }
        }

        // Build merged layout
        let mut layout = VertexLayout::new();
        common_attrs.sort_by_key(|s| *s as u8); // Consistent ordering

        for sem in &common_attrs {
            match sem {
                VertexSemantic::Position => layout.add_attribute(*sem, OutputFormat::Float32(3)),
                VertexSemantic::Normal => layout.add_attribute(*sem, OutputFormat::Float32(3)),
                VertexSemantic::Tangent => layout.add_attribute(*sem, OutputFormat::Float32(4)),
                VertexSemantic::TexCoord0 | VertexSemantic::TexCoord1 => {
                    layout.add_attribute(*sem, OutputFormat::Float32(2))
                }
                VertexSemantic::Color0 => layout.add_attribute(*sem, OutputFormat::Float32(4)),
                VertexSemantic::Joints0 => layout.add_attribute(*sem, OutputFormat::Uint16(4)),
                VertexSemantic::Weights0 => layout.add_attribute(*sem, OutputFormat::Float32(4)),
            }
        }

        // Allocate merged buffers
        let mut vertex_data = vec![0u8; total_vertices * layout.stride];
        let mut index_data = if has_indices {
            Vec::with_capacity(total_indices * 4) // U32 indices
        } else {
            Vec::new()
        };

        let mut primitive_offsets = Vec::with_capacity(all_primitives.len());
        let mut vertex_offset = 0u32;
        let mut index_offset = 0u32;

        // Merge each primitive
        for (prim, material_index) in &all_primitives {
            let pos_attr = prim.attributes.get(&VertexSemantic::Position).unwrap();
            let prim_vertex_count = pos_attr.count as u32;

            // Write vertices
            for vertex_idx in 0..pos_attr.count {
                let dest_offset = (vertex_offset as usize + vertex_idx) * layout.stride;

                for attr_desc in &layout.attributes {
                    let attr_offset = dest_offset + attr_desc.offset;

                    if let Some(attr) = prim.attributes.get(&attr_desc.semantic) {
                        self.write_merged_attribute(
                            attr,
                            vertex_idx,
                            &attr_desc.semantic,
                            &mut vertex_data[attr_offset..],
                        )?;
                    }
                }
            }

            // Write indices (rebased to merged vertex offset)
            let prim_index_count;
            let prim_index_offset;

            if let Some(idx_buf) = &prim.indices {
                prim_index_offset = Some(index_offset);
                prim_index_count = Some(idx_buf.count as u32);

                for i in 0..idx_buf.count {
                    let original_idx = read_index(idx_buf, i);
                    let rebased_idx = original_idx + vertex_offset;
                    index_data.extend_from_slice(&rebased_idx.to_le_bytes());
                }

                index_offset += idx_buf.count as u32;
            } else {
                prim_index_offset = None;
                prim_index_count = None;
            }

            primitive_offsets.push(PrimitiveOffset {
                vertex_offset,
                vertex_count: prim_vertex_count,
                index_offset: prim_index_offset,
                index_count: prim_index_count,
                material_index: *material_index,
            });

            vertex_offset += prim_vertex_count;
        }

        let merged_indices = if has_indices {
            Some(MergedIndices {
                data: index_data,
                count: total_indices,
            })
        } else {
            None
        };

        Ok(MergedMesh {
            vertices: InterleavedBuffer {
                data: vertex_data,
                layout,
                vertex_count: total_vertices,
            },
            indices: merged_indices,
            primitive_offsets,
        })
    }

    /// Write a merged attribute value.
    fn write_merged_attribute(
        &self,
        attr: &VertexAttribute,
        vertex_idx: usize,
        semantic: &VertexSemantic,
        dest: &mut [u8],
    ) -> Result<(), VertexFormatError> {
        match semantic {
            VertexSemantic::Position => {
                let comps = extract_f32_components(attr, vertex_idx)?;
                let [x, y, z] = self.settings.axis_conversion.apply_position(
                    comps[0] * self.settings.scale,
                    comps[1] * self.settings.scale,
                    comps[2] * self.settings.scale,
                );
                dest[0..4].copy_from_slice(&x.to_le_bytes());
                dest[4..8].copy_from_slice(&y.to_le_bytes());
                dest[8..12].copy_from_slice(&z.to_le_bytes());
            }
            VertexSemantic::Normal => {
                let comps = extract_f32_components(attr, vertex_idx)?;
                let [x, y, z] = self.settings.axis_conversion.apply_normal(
                    comps[0], comps[1], comps[2],
                );
                dest[0..4].copy_from_slice(&x.to_le_bytes());
                dest[4..8].copy_from_slice(&y.to_le_bytes());
                dest[8..12].copy_from_slice(&z.to_le_bytes());
            }
            VertexSemantic::Tangent => {
                let comps = extract_f32_components(attr, vertex_idx)?;
                let w = if comps.len() > 3 { comps[3] } else { 1.0 };
                let [x, y, z, w] = self.settings.axis_conversion.apply_tangent(
                    comps[0], comps[1], comps[2], w,
                );
                dest[0..4].copy_from_slice(&x.to_le_bytes());
                dest[4..8].copy_from_slice(&y.to_le_bytes());
                dest[8..12].copy_from_slice(&z.to_le_bytes());
                dest[12..16].copy_from_slice(&w.to_le_bytes());
            }
            VertexSemantic::TexCoord0 | VertexSemantic::TexCoord1 => {
                let comps = extract_f32_components(attr, vertex_idx)?;
                dest[0..4].copy_from_slice(&comps[0].to_le_bytes());
                dest[4..8].copy_from_slice(&comps[1].to_le_bytes());
            }
            VertexSemantic::Color0 => {
                let comps = extract_f32_components(attr, vertex_idx)?;
                let r = comps.first().copied().unwrap_or(1.0);
                let g = comps.get(1).copied().unwrap_or(1.0);
                let b = comps.get(2).copied().unwrap_or(1.0);
                let a = comps.get(3).copied().unwrap_or(1.0);
                dest[0..4].copy_from_slice(&r.to_le_bytes());
                dest[4..8].copy_from_slice(&g.to_le_bytes());
                dest[8..12].copy_from_slice(&b.to_le_bytes());
                dest[12..16].copy_from_slice(&a.to_le_bytes());
            }
            VertexSemantic::Joints0 => {
                let joints = extract_u16_components(attr, vertex_idx)?;
                for (i, &j) in joints.iter().take(4).enumerate() {
                    dest[i * 2..(i + 1) * 2].copy_from_slice(&j.to_le_bytes());
                }
                for i in joints.len()..4 {
                    dest[i * 2..(i + 1) * 2].copy_from_slice(&0u16.to_le_bytes());
                }
            }
            VertexSemantic::Weights0 => {
                let comps = extract_f32_components(attr, vertex_idx)?;
                for i in 0..4 {
                    let w = comps.get(i).copied().unwrap_or(0.0);
                    dest[i * 4..(i + 1) * 4].copy_from_slice(&w.to_le_bytes());
                }
            }
        }
        Ok(())
    }
}

/// Read an index from an index buffer.
fn read_index(idx_buf: &IndexBuffer, idx: usize) -> u32 {
    match idx_buf.format {
        IndexFormat::U8 => idx_buf.data[idx] as u32,
        IndexFormat::U16 => {
            let offset = idx * 2;
            u16::from_le_bytes([idx_buf.data[offset], idx_buf.data[offset + 1]]) as u32
        }
        IndexFormat::U32 => {
            let offset = idx * 4;
            u32::from_le_bytes([
                idx_buf.data[offset],
                idx_buf.data[offset + 1],
                idx_buf.data[offset + 2],
                idx_buf.data[offset + 3],
            ])
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gltf::AttributeType;

    // Helper to create a simple position attribute
    fn make_position_attr(positions: &[[f32; 3]]) -> VertexAttribute {
        let mut data = Vec::with_capacity(positions.len() * 12);
        for pos in positions {
            data.extend_from_slice(&pos[0].to_le_bytes());
            data.extend_from_slice(&pos[1].to_le_bytes());
            data.extend_from_slice(&pos[2].to_le_bytes());
        }
        VertexAttribute {
            semantic: VertexSemantic::Position,
            component_type: ComponentType::F32,
            attribute_type: AttributeType::Vec3,
            offset: 0,
            stride: 12,
            count: positions.len(),
            data,
        }
    }

    // Helper to create a normal attribute
    fn make_normal_attr(normals: &[[f32; 3]]) -> VertexAttribute {
        let mut data = Vec::with_capacity(normals.len() * 12);
        for n in normals {
            data.extend_from_slice(&n[0].to_le_bytes());
            data.extend_from_slice(&n[1].to_le_bytes());
            data.extend_from_slice(&n[2].to_le_bytes());
        }
        VertexAttribute {
            semantic: VertexSemantic::Normal,
            component_type: ComponentType::F32,
            attribute_type: AttributeType::Vec3,
            offset: 0,
            stride: 12,
            count: normals.len(),
            data,
        }
    }

    // Helper to create a UV attribute
    fn make_uv_attr(uvs: &[[f32; 2]]) -> VertexAttribute {
        let mut data = Vec::with_capacity(uvs.len() * 8);
        for uv in uvs {
            data.extend_from_slice(&uv[0].to_le_bytes());
            data.extend_from_slice(&uv[1].to_le_bytes());
        }
        VertexAttribute {
            semantic: VertexSemantic::TexCoord0,
            component_type: ComponentType::F32,
            attribute_type: AttributeType::Vec2,
            offset: 0,
            stride: 8,
            count: uvs.len(),
            data,
        }
    }

    // Helper to create a color attribute
    fn make_color_attr(colors: &[[f32; 4]]) -> VertexAttribute {
        let mut data = Vec::with_capacity(colors.len() * 16);
        for c in colors {
            data.extend_from_slice(&c[0].to_le_bytes());
            data.extend_from_slice(&c[1].to_le_bytes());
            data.extend_from_slice(&c[2].to_le_bytes());
            data.extend_from_slice(&c[3].to_le_bytes());
        }
        VertexAttribute {
            semantic: VertexSemantic::Color0,
            component_type: ComponentType::F32,
            attribute_type: AttributeType::Vec4,
            offset: 0,
            stride: 16,
            count: colors.len(),
            data,
        }
    }

    // Helper to create a tangent attribute
    fn make_tangent_attr(tangents: &[[f32; 4]]) -> VertexAttribute {
        let mut data = Vec::with_capacity(tangents.len() * 16);
        for t in tangents {
            data.extend_from_slice(&t[0].to_le_bytes());
            data.extend_from_slice(&t[1].to_le_bytes());
            data.extend_from_slice(&t[2].to_le_bytes());
            data.extend_from_slice(&t[3].to_le_bytes());
        }
        VertexAttribute {
            semantic: VertexSemantic::Tangent,
            component_type: ComponentType::F32,
            attribute_type: AttributeType::Vec4,
            offset: 0,
            stride: 16,
            count: tangents.len(),
            data,
        }
    }

    // Helper to create joints attribute
    fn make_joints_attr(joints: &[[u16; 4]]) -> VertexAttribute {
        let mut data = Vec::with_capacity(joints.len() * 8);
        for j in joints {
            data.extend_from_slice(&j[0].to_le_bytes());
            data.extend_from_slice(&j[1].to_le_bytes());
            data.extend_from_slice(&j[2].to_le_bytes());
            data.extend_from_slice(&j[3].to_le_bytes());
        }
        VertexAttribute {
            semantic: VertexSemantic::Joints0,
            component_type: ComponentType::U16,
            attribute_type: AttributeType::Vec4,
            offset: 0,
            stride: 8,
            count: joints.len(),
            data,
        }
    }

    // Helper to create weights attribute
    fn make_weights_attr(weights: &[[f32; 4]]) -> VertexAttribute {
        let mut data = Vec::with_capacity(weights.len() * 16);
        for w in weights {
            data.extend_from_slice(&w[0].to_le_bytes());
            data.extend_from_slice(&w[1].to_le_bytes());
            data.extend_from_slice(&w[2].to_le_bytes());
            data.extend_from_slice(&w[3].to_le_bytes());
        }
        VertexAttribute {
            semantic: VertexSemantic::Weights0,
            component_type: ComponentType::F32,
            attribute_type: AttributeType::Vec4,
            offset: 0,
            stride: 16,
            count: weights.len(),
            data,
        }
    }

    // Helper to create a simple primitive
    fn make_primitive(positions: &[[f32; 3]]) -> GltfPrimitive {
        let mut attributes = HashMap::new();
        attributes.insert(VertexSemantic::Position, make_position_attr(positions));
        GltfPrimitive {
            attributes,
            indices: None,
            material_index: None,
        }
    }

    // Helper to create a primitive with all attributes
    fn make_full_primitive() -> GltfPrimitive {
        let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let normals = [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]];
        let tangents = [[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]];
        let uvs = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let colors = [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]];

        let mut attributes = HashMap::new();
        attributes.insert(VertexSemantic::Position, make_position_attr(&positions));
        attributes.insert(VertexSemantic::Normal, make_normal_attr(&normals));
        attributes.insert(VertexSemantic::Tangent, make_tangent_attr(&tangents));
        attributes.insert(VertexSemantic::TexCoord0, make_uv_attr(&uvs));
        attributes.insert(VertexSemantic::Color0, make_color_attr(&colors));

        GltfPrimitive {
            attributes,
            indices: Some(IndexBuffer {
                format: IndexFormat::U16,
                count: 3,
                data: vec![0, 0, 1, 0, 2, 0],
            }),
            material_index: Some(0),
        }
    }

    // ────────────────────────────────────────────────────────────────────────────
    // Interleaved layout tests
    // ────────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_interleaved_layout_positions_only() {
        let positions = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]];
        let primitive = make_primitive(&positions);

        let converter = VertexFormatConverter::with_defaults();
        let result = converter.to_interleaved(&primitive).unwrap();

        assert_eq!(result.vertex_count, 3);
        assert_eq!(result.layout.stride, 12); // 3 * f32

        // Verify first position
        let p0: [f32; 3] = [
            f32::from_le_bytes(result.data[0..4].try_into().unwrap()),
            f32::from_le_bytes(result.data[4..8].try_into().unwrap()),
            f32::from_le_bytes(result.data[8..12].try_into().unwrap()),
        ];
        assert_eq!(p0, [0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_interleaved_layout_with_normals() {
        let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let normals = [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]];

        let mut attributes = HashMap::new();
        attributes.insert(VertexSemantic::Position, make_position_attr(&positions));
        attributes.insert(VertexSemantic::Normal, make_normal_attr(&normals));

        let primitive = GltfPrimitive {
            attributes,
            indices: None,
            material_index: None,
        };

        let converter = VertexFormatConverter::with_defaults();
        let result = converter.to_interleaved(&primitive).unwrap();

        assert_eq!(result.vertex_count, 2);
        assert_eq!(result.layout.stride, 24); // pos(12) + normal(12)
        assert_eq!(result.layout.attributes.len(), 2);
    }

    #[test]
    fn test_interleaved_full_vertex() {
        let primitive = make_full_primitive();

        let converter = VertexFormatConverter::with_defaults();
        let result = converter.to_interleaved(&primitive).unwrap();

        assert_eq!(result.vertex_count, 3);
        // pos(12) + normal(12) + tangent(16) + uv(8) + color(16) = 64
        assert_eq!(result.layout.stride, 64);
        assert_eq!(result.layout.attributes.len(), 5);
    }

    // ────────────────────────────────────────────────────────────────────────────
    // Split layout tests
    // ────────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_split_layout_generation() {
        let positions = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]];
        let normals = [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]];

        let mut attributes = HashMap::new();
        attributes.insert(VertexSemantic::Position, make_position_attr(&positions));
        attributes.insert(VertexSemantic::Normal, make_normal_attr(&normals));

        let primitive = GltfPrimitive {
            attributes,
            indices: None,
            material_index: None,
        };

        let converter = VertexFormatConverter::with_defaults();
        let result = converter.to_split(&primitive).unwrap();

        assert_eq!(result.vertex_count, 2);
        assert!(result.buffers.contains_key(&VertexSemantic::Position));
        assert!(result.buffers.contains_key(&VertexSemantic::Normal));

        let pos_buf = &result.buffers[&VertexSemantic::Position];
        assert_eq!(pos_buf.len(), 24); // 2 vertices * 12 bytes

        let norm_buf = &result.buffers[&VertexSemantic::Normal];
        assert_eq!(norm_buf.len(), 24);
    }

    #[test]
    fn test_split_layout_with_uvs() {
        let positions = [[0.0, 0.0, 0.0]];
        let uvs = [[0.5, 0.5]];

        let mut attributes = HashMap::new();
        attributes.insert(VertexSemantic::Position, make_position_attr(&positions));
        attributes.insert(VertexSemantic::TexCoord0, make_uv_attr(&uvs));

        let primitive = GltfPrimitive {
            attributes,
            indices: None,
            material_index: None,
        };

        let converter = VertexFormatConverter::with_defaults();
        let result = converter.to_split(&primitive).unwrap();

        let uv_buf = &result.buffers[&VertexSemantic::TexCoord0];
        let u = f32::from_le_bytes(uv_buf[0..4].try_into().unwrap());
        let v = f32::from_le_bytes(uv_buf[4..8].try_into().unwrap());
        assert_eq!((u, v), (0.5, 0.5));
    }

    // ────────────────────────────────────────────────────────────────────────────
    // Position compression tests
    // ────────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_position_compression_float16() {
        let positions = [[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]];
        let primitive = make_primitive(&positions);

        let settings = ImportSettings::new()
            .with_compression(CompressionSettings {
                position: PositionCompression::Float16,
                ..CompressionSettings::none()
            });

        let converter = VertexFormatConverter::new(settings);
        let result = converter.to_compressed(&primitive).unwrap();

        // Position is 3 * f16 = 6 bytes
        assert_eq!(result.layout.find_attribute(VertexSemantic::Position).unwrap().size, 6);

        // Verify round-trip accuracy
        let p0_x = f16_to_f32(u16::from_le_bytes(result.data[0..2].try_into().unwrap()));
        let p0_y = f16_to_f32(u16::from_le_bytes(result.data[2..4].try_into().unwrap()));
        let p0_z = f16_to_f32(u16::from_le_bytes(result.data[4..6].try_into().unwrap()));

        assert!((p0_x - 1.0).abs() < 0.001);
        assert!((p0_y - 2.0).abs() < 0.001);
        assert!((p0_z - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_position_compression_10_10_10_2() {
        let positions = [[0.0, 0.5, 1.0], [0.25, 0.75, 0.5]];
        let primitive = make_primitive(&positions);

        let settings = ImportSettings::new()
            .with_compression(CompressionSettings {
                position: PositionCompression::Norm10_10_10_2,
                ..CompressionSettings::none()
            })
            .with_position_aabb([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);

        let converter = VertexFormatConverter::new(settings);
        let result = converter.to_compressed(&primitive).unwrap();

        // Position is packed into 4 bytes
        assert_eq!(result.layout.find_attribute(VertexSemantic::Position).unwrap().size, 4);

        // Verify round-trip
        let packed = u32::from_le_bytes(result.data[0..4].try_into().unwrap());
        let decoded = unpack_position_10_10_10_2(packed, result.position_aabb.0, result.position_aabb.1);

        assert!((decoded[0] - 0.0).abs() < 0.01);
        assert!((decoded[1] - 0.5).abs() < 0.01);
        assert!((decoded[2] - 1.0).abs() < 0.01);
    }

    // ────────────────────────────────────────────────────────────────────────────
    // Normal octahedral encoding tests
    // ────────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_normal_octahedral_encoding() {
        // Test various normals
        let test_normals: [[f32; 3]; 5] = [
            [0.0, 0.0, 1.0],   // +Z
            [0.0, 0.0, -1.0],  // -Z
            [1.0, 0.0, 0.0],   // +X
            [0.0, 1.0, 0.0],   // +Y
            [0.577, 0.577, 0.577], // Diagonal (normalized)
        ];

        for normal in &test_normals {
            let [nx, ny, nz] = *normal;
            let len = (nx * nx + ny * ny + nz * nz).sqrt();
            let nx = nx / len;
            let ny = ny / len;
            let nz = nz / len;

            let encoded = encode_octahedral(nx, ny, nz);
            let decoded = decode_octahedral(encoded);

            // Check accuracy (octahedral has good accuracy for unit vectors)
            assert!((decoded[0] - nx).abs() < 0.01, "X mismatch for {:?}", normal);
            assert!((decoded[1] - ny).abs() < 0.01, "Y mismatch for {:?}", normal);
            assert!((decoded[2] - nz).abs() < 0.01, "Z mismatch for {:?}", normal);

            // Verify it's still a unit vector
            let len2 = decoded[0] * decoded[0] + decoded[1] * decoded[1] + decoded[2] * decoded[2];
            assert!((len2 - 1.0).abs() < 0.01, "Not unit length for {:?}", normal);
        }
    }

    #[test]
    fn test_normal_compression_snorm8() {
        let normals = [[0.0, 0.0, 1.0]];
        let positions = [[0.0, 0.0, 0.0]];

        let mut attributes = HashMap::new();
        attributes.insert(VertexSemantic::Position, make_position_attr(&positions));
        attributes.insert(VertexSemantic::Normal, make_normal_attr(&normals));

        let primitive = GltfPrimitive {
            attributes,
            indices: None,
            material_index: None,
        };

        let settings = ImportSettings::new()
            .with_compression(CompressionSettings {
                normal: NormalCompression::Snorm8,
                ..CompressionSettings::none()
            });

        let converter = VertexFormatConverter::new(settings);
        let result = converter.to_compressed(&primitive).unwrap();

        // Normal is 4 bytes (3 + 1 padding)
        let normal_attr = result.layout.find_attribute(VertexSemantic::Normal).unwrap();
        assert_eq!(normal_attr.size, 4);

        // Position offset + normal
        let offset = normal_attr.offset;
        let nz = result.data[offset + 2] as i8;
        assert_eq!(nz, 127); // 1.0 -> 127 in SNORM8
    }

    // ────────────────────────────────────────────────────────────────────────────
    // UV compression tests
    // ────────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_uv_compression_float16() {
        let positions = [[0.0, 0.0, 0.0]];
        let uvs = [[0.5, 0.75]];

        let mut attributes = HashMap::new();
        attributes.insert(VertexSemantic::Position, make_position_attr(&positions));
        attributes.insert(VertexSemantic::TexCoord0, make_uv_attr(&uvs));

        let primitive = GltfPrimitive {
            attributes,
            indices: None,
            material_index: None,
        };

        let settings = ImportSettings::new()
            .with_compression(CompressionSettings {
                uv: UvCompression::Float16,
                ..CompressionSettings::none()
            });

        let converter = VertexFormatConverter::new(settings);
        let result = converter.to_compressed(&primitive).unwrap();

        let uv_attr = result.layout.find_attribute(VertexSemantic::TexCoord0).unwrap();
        assert_eq!(uv_attr.size, 4); // 2 * f16

        let offset = uv_attr.offset;
        let u = f16_to_f32(u16::from_le_bytes(result.data[offset..offset + 2].try_into().unwrap()));
        let v = f16_to_f32(u16::from_le_bytes(result.data[offset + 2..offset + 4].try_into().unwrap()));

        assert!((u - 0.5).abs() < 0.001);
        assert!((v - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_uv_compression_unorm16() {
        let positions = [[0.0, 0.0, 0.0]];
        let uvs = [[0.25, 0.5]];

        let mut attributes = HashMap::new();
        attributes.insert(VertexSemantic::Position, make_position_attr(&positions));
        attributes.insert(VertexSemantic::TexCoord0, make_uv_attr(&uvs));

        let primitive = GltfPrimitive {
            attributes,
            indices: None,
            material_index: None,
        };

        let settings = ImportSettings::new()
            .with_compression(CompressionSettings {
                uv: UvCompression::Unorm16,
                ..CompressionSettings::none()
            });

        let converter = VertexFormatConverter::new(settings);
        let result = converter.to_compressed(&primitive).unwrap();

        let uv_attr = result.layout.find_attribute(VertexSemantic::TexCoord0).unwrap();
        let offset = uv_attr.offset;

        let u = unorm16_to_f32(u16::from_le_bytes(result.data[offset..offset + 2].try_into().unwrap()));
        let v = unorm16_to_f32(u16::from_le_bytes(result.data[offset + 2..offset + 4].try_into().unwrap()));

        assert!((u - 0.25).abs() < 0.001);
        assert!((v - 0.5).abs() < 0.001);
    }

    // ────────────────────────────────────────────────────────────────────────────
    // Color compression tests
    // ────────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_color_compression_unorm8() {
        let positions = [[0.0, 0.0, 0.0]];
        let colors = [[1.0, 0.5, 0.25, 1.0]];

        let mut attributes = HashMap::new();
        attributes.insert(VertexSemantic::Position, make_position_attr(&positions));
        attributes.insert(VertexSemantic::Color0, make_color_attr(&colors));

        let primitive = GltfPrimitive {
            attributes,
            indices: None,
            material_index: None,
        };

        let settings = ImportSettings::new()
            .with_compression(CompressionSettings {
                color: ColorCompression::Unorm8,
                ..CompressionSettings::none()
            });

        let converter = VertexFormatConverter::new(settings);
        let result = converter.to_compressed(&primitive).unwrap();

        let color_attr = result.layout.find_attribute(VertexSemantic::Color0).unwrap();
        assert_eq!(color_attr.size, 4); // RGBA8

        let offset = color_attr.offset;
        let r = unorm8_to_f32(result.data[offset]);
        let g = unorm8_to_f32(result.data[offset + 1]);
        let b = unorm8_to_f32(result.data[offset + 2]);
        let a = unorm8_to_f32(result.data[offset + 3]);

        assert!((r - 1.0).abs() < 0.01);
        assert!((g - 0.5).abs() < 0.01);
        assert!((b - 0.25).abs() < 0.01);
        assert!((a - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_color_compression_rgb10a2() {
        let positions = [[0.0, 0.0, 0.0]];
        let colors = [[0.5, 0.5, 0.5, 1.0]];

        let mut attributes = HashMap::new();
        attributes.insert(VertexSemantic::Position, make_position_attr(&positions));
        attributes.insert(VertexSemantic::Color0, make_color_attr(&colors));

        let primitive = GltfPrimitive {
            attributes,
            indices: None,
            material_index: None,
        };

        let settings = ImportSettings::new()
            .with_compression(CompressionSettings {
                color: ColorCompression::Rgb10A2,
                ..CompressionSettings::none()
            });

        let converter = VertexFormatConverter::new(settings);
        let result = converter.to_compressed(&primitive).unwrap();

        let color_attr = result.layout.find_attribute(VertexSemantic::Color0).unwrap();
        assert_eq!(color_attr.size, 4); // RGB10A2 packed
    }

    // ────────────────────────────────────────────────────────────────────────────
    // Axis conversion tests
    // ────────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_axis_conversion_y_up_to_z_up() {
        let positions = [[1.0, 2.0, 3.0]];
        let primitive = make_primitive(&positions);

        let settings = ImportSettings::new()
            .with_axis_conversion(AxisConversion::YUpToZUp);

        let converter = VertexFormatConverter::new(settings);
        let result = converter.to_interleaved(&primitive).unwrap();

        let x = f32::from_le_bytes(result.data[0..4].try_into().unwrap());
        let y = f32::from_le_bytes(result.data[4..8].try_into().unwrap());
        let z = f32::from_le_bytes(result.data[8..12].try_into().unwrap());

        // Y-up to Z-up: (x, y, z) -> (x, z, -y)
        assert_eq!((x, y, z), (1.0, 3.0, -2.0));
    }

    #[test]
    fn test_axis_conversion_z_up_to_y_up() {
        let positions = [[1.0, 2.0, 3.0]];
        let primitive = make_primitive(&positions);

        let settings = ImportSettings::new()
            .with_axis_conversion(AxisConversion::ZUpToYUp);

        let converter = VertexFormatConverter::new(settings);
        let result = converter.to_interleaved(&primitive).unwrap();

        let x = f32::from_le_bytes(result.data[0..4].try_into().unwrap());
        let y = f32::from_le_bytes(result.data[4..8].try_into().unwrap());
        let z = f32::from_le_bytes(result.data[8..12].try_into().unwrap());

        // Z-up to Y-up: (x, y, z) -> (x, -z, y)
        assert_eq!((x, y, z), (1.0, -3.0, 2.0));
    }

    #[test]
    fn test_axis_conversion_flip_x() {
        let positions = [[1.0, 2.0, 3.0]];
        let primitive = make_primitive(&positions);

        let settings = ImportSettings::new()
            .with_axis_conversion(AxisConversion::FlipX);

        let converter = VertexFormatConverter::new(settings);
        let result = converter.to_interleaved(&primitive).unwrap();

        let x = f32::from_le_bytes(result.data[0..4].try_into().unwrap());
        assert_eq!(x, -1.0);
    }

    // ────────────────────────────────────────────────────────────────────────────
    // Scale transform tests
    // ────────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_scale_transform() {
        let positions = [[1.0, 2.0, 3.0]];
        let primitive = make_primitive(&positions);

        let settings = ImportSettings::new().with_scale(0.01);

        let converter = VertexFormatConverter::new(settings);
        let result = converter.to_interleaved(&primitive).unwrap();

        let x = f32::from_le_bytes(result.data[0..4].try_into().unwrap());
        let y = f32::from_le_bytes(result.data[4..8].try_into().unwrap());
        let z = f32::from_le_bytes(result.data[8..12].try_into().unwrap());

        assert!((x - 0.01).abs() < 0.0001);
        assert!((y - 0.02).abs() < 0.0001);
        assert!((z - 0.03).abs() < 0.0001);
    }

    #[test]
    fn test_scale_with_axis_conversion() {
        let positions = [[100.0, 200.0, 300.0]];
        let primitive = make_primitive(&positions);

        let settings = ImportSettings::new()
            .with_scale(0.01)
            .with_axis_conversion(AxisConversion::YUpToZUp);

        let converter = VertexFormatConverter::new(settings);
        let result = converter.to_interleaved(&primitive).unwrap();

        let x = f32::from_le_bytes(result.data[0..4].try_into().unwrap());
        let y = f32::from_le_bytes(result.data[4..8].try_into().unwrap());
        let z = f32::from_le_bytes(result.data[8..12].try_into().unwrap());

        // Scale first, then axis conversion: (1, 2, 3) -> (1, 3, -2)
        assert!((x - 1.0).abs() < 0.0001);
        assert!((y - 3.0).abs() < 0.0001);
        assert!((z - (-2.0)).abs() < 0.0001);
    }

    // ────────────────────────────────────────────────────────────────────────────
    // Merge meshes tests
    // ────────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_merge_meshes_basic() {
        let mesh1 = GltfMesh {
            name: Some("mesh1".into()),
            primitives: vec![make_primitive(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])],
        };
        let mesh2 = GltfMesh {
            name: Some("mesh2".into()),
            primitives: vec![make_primitive(&[[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])],
        };

        let converter = VertexFormatConverter::with_defaults();
        let merged = converter.merge_meshes(&[mesh1, mesh2]).unwrap();

        assert_eq!(merged.vertices.vertex_count, 4);
        assert_eq!(merged.primitive_offsets.len(), 2);
        assert_eq!(merged.primitive_offsets[0].vertex_offset, 0);
        assert_eq!(merged.primitive_offsets[0].vertex_count, 2);
        assert_eq!(merged.primitive_offsets[1].vertex_offset, 2);
        assert_eq!(merged.primitive_offsets[1].vertex_count, 2);
    }

    #[test]
    fn test_merge_meshes_with_indices() {
        let mut prim1 = make_primitive(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        prim1.indices = Some(IndexBuffer {
            format: IndexFormat::U16,
            count: 3,
            data: vec![0, 0, 1, 0, 2, 0],
        });

        let mut prim2 = make_primitive(&[[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 1.0, 0.0]]);
        prim2.indices = Some(IndexBuffer {
            format: IndexFormat::U16,
            count: 3,
            data: vec![0, 0, 1, 0, 2, 0],
        });

        let mesh1 = GltfMesh {
            name: Some("mesh1".into()),
            primitives: vec![prim1],
        };
        let mesh2 = GltfMesh {
            name: Some("mesh2".into()),
            primitives: vec![prim2],
        };

        let converter = VertexFormatConverter::with_defaults();
        let merged = converter.merge_meshes(&[mesh1, mesh2]).unwrap();

        assert!(merged.indices.is_some());
        let indices = merged.indices.as_ref().unwrap();
        assert_eq!(indices.count, 6);

        // Check rebased indices
        let idx0 = u32::from_le_bytes(indices.data[0..4].try_into().unwrap());
        let idx3 = u32::from_le_bytes(indices.data[12..16].try_into().unwrap()); // First index of second primitive

        assert_eq!(idx0, 0);
        assert_eq!(idx3, 3); // Rebased from 0 to 3 (vertex_offset of second primitive)
    }

    #[test]
    fn test_merge_preserves_material_indices() {
        let mut prim1 = make_primitive(&[[0.0, 0.0, 0.0]]);
        prim1.material_index = Some(0);

        let mut prim2 = make_primitive(&[[1.0, 0.0, 0.0]]);
        prim2.material_index = Some(1);

        let mesh = GltfMesh {
            name: Some("mesh".into()),
            primitives: vec![prim1, prim2],
        };

        let converter = VertexFormatConverter::with_defaults();
        let merged = converter.merge_meshes(&[mesh]).unwrap();

        assert_eq!(merged.primitive_offsets[0].material_index, Some(0));
        assert_eq!(merged.primitive_offsets[1].material_index, Some(1));
    }

    // ────────────────────────────────────────────────────────────────────────────
    // Round-trip accuracy tests
    // ────────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_roundtrip_f16() {
        let test_values = [0.0, 1.0, -1.0, 0.5, 0.001, 1000.0, -0.25, 3.14159];

        for &val in &test_values {
            let encoded = f32_to_f16(val);
            let decoded = f16_to_f32(encoded);

            // f16 has limited precision, allow some error
            let rel_error = if val.abs() > 0.0001 {
                (decoded - val).abs() / val.abs()
            } else {
                (decoded - val).abs()
            };

            assert!(
                rel_error < 0.01,
                "f16 roundtrip failed for {}: got {}, error {}",
                val, decoded, rel_error
            );
        }
    }

    #[test]
    fn test_roundtrip_10_10_10_2() {
        let test_vectors = [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0, -1.0],
            [0.5, -0.5, 0.25, 0.0],
        ];

        for vec in &test_vectors {
            let packed = pack_10_10_10_2(vec[0], vec[1], vec[2], vec[3]);
            let unpacked = unpack_10_10_10_2(packed);

            for i in 0..4 {
                let tolerance = if i < 3 { 0.01 } else { 0.5 }; // A component is only 2 bits
                assert!(
                    (unpacked[i] - vec[i]).abs() < tolerance,
                    "10-10-10-2 roundtrip failed for {:?}[{}]: got {}, expected {}",
                    vec, i, unpacked[i], vec[i]
                );
            }
        }
    }

    #[test]
    fn test_roundtrip_position_10_10_10_2() {
        let aabb_min = [-10.0, -10.0, -10.0];
        let aabb_max = [10.0, 10.0, 10.0];

        let test_positions = [
            [0.0, 0.0, 0.0],
            [10.0, 10.0, 10.0],
            [-10.0, -10.0, -10.0],
            [5.0, -5.0, 2.5],
        ];

        for pos in &test_positions {
            let packed = pack_position_10_10_10_2(pos[0], pos[1], pos[2], aabb_min, aabb_max);
            let unpacked = unpack_position_10_10_10_2(packed, aabb_min, aabb_max);

            for i in 0..3 {
                assert!(
                    (unpacked[i] - pos[i]).abs() < 0.1,
                    "Position 10-10-10-2 roundtrip failed for {:?}[{}]: got {}, expected {}",
                    pos, i, unpacked[i], pos[i]
                );
            }
        }
    }

    // ────────────────────────────────────────────────────────────────────────────
    // Edge case tests
    // ────────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_missing_position_attribute() {
        let primitive = GltfPrimitive {
            attributes: HashMap::new(),
            indices: None,
            material_index: None,
        };

        let converter = VertexFormatConverter::with_defaults();
        let result = converter.to_interleaved(&primitive);

        assert!(matches!(
            result,
            Err(VertexFormatError::MissingAttribute(VertexSemantic::Position))
        ));
    }

    #[test]
    fn test_empty_mesh_merge() {
        let converter = VertexFormatConverter::with_defaults();
        let result = converter.merge_meshes(&[]);

        assert!(matches!(result, Err(VertexFormatError::InvalidData(_))));
    }

    #[test]
    fn test_skinning_attributes() {
        let positions = [[0.0, 0.0, 0.0]];
        let joints = [[0, 1, 2, 3]];
        let weights = [[0.5, 0.3, 0.15, 0.05]];

        let mut attributes = HashMap::new();
        attributes.insert(VertexSemantic::Position, make_position_attr(&positions));
        attributes.insert(VertexSemantic::Joints0, make_joints_attr(&joints));
        attributes.insert(VertexSemantic::Weights0, make_weights_attr(&weights));

        let primitive = GltfPrimitive {
            attributes,
            indices: None,
            material_index: None,
        };

        let converter = VertexFormatConverter::with_defaults();
        let result = converter.to_interleaved(&primitive).unwrap();

        // Verify layout includes skinning
        assert!(result.layout.find_attribute(VertexSemantic::Joints0).is_some());
        assert!(result.layout.find_attribute(VertexSemantic::Weights0).is_some());
    }

    #[test]
    fn test_compressed_skinning() {
        let positions = [[0.0, 0.0, 0.0]];
        let joints = [[0, 1, 2, 3]];
        let weights = [[0.5, 0.3, 0.15, 0.05]];

        let mut attributes = HashMap::new();
        attributes.insert(VertexSemantic::Position, make_position_attr(&positions));
        attributes.insert(VertexSemantic::Joints0, make_joints_attr(&joints));
        attributes.insert(VertexSemantic::Weights0, make_weights_attr(&weights));

        let primitive = GltfPrimitive {
            attributes,
            indices: None,
            material_index: None,
        };

        let settings = ImportSettings::new()
            .with_compression(CompressionSettings::balanced());

        let converter = VertexFormatConverter::new(settings);
        let result = converter.to_compressed(&primitive).unwrap();

        // Weights should be UNORM16 in compressed mode
        let weights_attr = result.layout.find_attribute(VertexSemantic::Weights0).unwrap();
        assert_eq!(weights_attr.format, OutputFormat::Unorm16(4));
    }

    #[test]
    fn test_aabb_calculation() {
        let positions = [
            [-5.0, -3.0, -1.0],
            [5.0, 3.0, 1.0],
            [0.0, 0.0, 0.0],
        ];
        let primitive = make_primitive(&positions);

        let converter = VertexFormatConverter::with_defaults();
        let (min, max) = converter.calculate_aabb(&primitive).unwrap();

        assert_eq!(min, [-5.0, -3.0, -1.0]);
        assert_eq!(max, [5.0, 3.0, 1.0]);
    }

    #[test]
    fn test_aabb_with_scale() {
        let positions = [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]];
        let primitive = make_primitive(&positions);

        let settings = ImportSettings::new().with_scale(10.0);
        let converter = VertexFormatConverter::new(settings);
        let (min, max) = converter.calculate_aabb(&primitive).unwrap();

        assert_eq!(min, [-10.0, -10.0, -10.0]);
        assert_eq!(max, [10.0, 10.0, 10.0]);
    }

    #[test]
    fn test_compression_settings_presets() {
        let none = CompressionSettings::none();
        assert_eq!(none.position, PositionCompression::None);
        assert_eq!(none.normal, NormalCompression::None);

        let balanced = CompressionSettings::balanced();
        assert_eq!(balanced.position, PositionCompression::Float16);
        assert_eq!(balanced.normal, NormalCompression::Octahedral);

        let aggressive = CompressionSettings::aggressive();
        assert_eq!(aggressive.position, PositionCompression::Norm10_10_10_2);
        assert_eq!(aggressive.uv, UvCompression::Unorm16);
    }

    #[test]
    fn test_output_format_sizes() {
        assert_eq!(OutputFormat::Float32(3).size_bytes(), 12);
        assert_eq!(OutputFormat::Float16(3).size_bytes(), 6);
        assert_eq!(OutputFormat::Rgb10A2.size_bytes(), 4);
        assert_eq!(OutputFormat::Snorm8(4).size_bytes(), 4);
        assert_eq!(OutputFormat::Unorm16(2).size_bytes(), 4);
    }
}
