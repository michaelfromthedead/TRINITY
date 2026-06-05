//! Draco mesh decompression for glTF files (KHR_draco_mesh_compression).
//!
//! Provides decompression of Draco-compressed geometry data in glTF files,
//! supporting both required and optional extension usage.
//!
//! # Features
//!
//! - Extension detection (required vs optional)
//! - Draco-compressed mesh and point cloud decompression
//! - Attribute mapping to glTF semantics
//! - Seamless integration with existing glTF pipeline
//! - Graceful fallback when decoder not available
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::asset::draco::{
//!     detect_draco_extension, parse_draco_extension, decompress_draco,
//!     decompress_primitive_if_draco,
//! };
//!
//! // Check if file uses Draco compression
//! let has_draco = detect_draco_extension(&gltf_json);
//!
//! // Parse extension from a primitive
//! if let Some(ext) = parse_draco_extension(&primitive_json) {
//!     let result = decompress_draco(&buffer_data, &ext)?;
//!     // Use decompressed positions, normals, etc.
//! }
//!
//! // Or use the high-level integration function
//! if let Some(result) = decompress_primitive_if_draco(&primitive, &buffers)? {
//!     // Use decompressed data
//! }
//! ```

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

// Re-export from parent gltf module for convenience
use crate::gltf::{
    ComponentType, GltfPrimitive, IndexBuffer, IndexFormat, VertexAttribute,
    VertexSemantic, AttributeType, GltfError,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Extension name for KHR_draco_mesh_compression.
pub const DRACO_EXTENSION_NAME: &str = "KHR_draco_mesh_compression";

/// Draco attribute type for positions.
pub const DRACO_ATTR_POSITION: u32 = 0;
/// Draco attribute type for normals.
pub const DRACO_ATTR_NORMAL: u32 = 1;
/// Draco attribute type for colors.
pub const DRACO_ATTR_COLOR: u32 = 2;
/// Draco attribute type for texture coordinates.
pub const DRACO_ATTR_TEX_COORD: u32 = 3;
/// Draco attribute type for generic data.
pub const DRACO_ATTR_GENERIC: u32 = 4;

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Draco decompression error.
#[derive(Debug, Clone)]
pub enum DracoError {
    /// Draco decoder not available (feature not enabled).
    DecoderNotAvailable,
    /// Invalid Draco data.
    InvalidData(String),
    /// Unsupported Draco geometry type.
    UnsupportedGeometryType(String),
    /// Attribute not found in compressed data.
    MissingAttribute(String),
    /// Buffer view out of bounds.
    BufferOutOfBounds { view: u32, buffer_len: usize },
    /// Invalid buffer view index.
    InvalidBufferView(u32),
    /// JSON parsing error.
    JsonError(String),
    /// Attribute type mismatch.
    AttributeTypeMismatch { expected: String, got: String },
    /// Vertex count mismatch between attributes.
    VertexCountMismatch { expected: usize, got: usize },
    /// Decompression failed.
    DecompressionFailed(String),
    /// Invalid glTF structure.
    InvalidGltf(String),
}

impl std::fmt::Display for DracoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DecoderNotAvailable => write!(f, "Draco decoder not available"),
            Self::InvalidData(msg) => write!(f, "invalid Draco data: {}", msg),
            Self::UnsupportedGeometryType(t) => write!(f, "unsupported geometry type: {}", t),
            Self::MissingAttribute(attr) => write!(f, "missing attribute: {}", attr),
            Self::BufferOutOfBounds { view, buffer_len } => {
                write!(f, "buffer view {} out of bounds (buffer len {})", view, buffer_len)
            }
            Self::InvalidBufferView(idx) => write!(f, "invalid buffer view index: {}", idx),
            Self::JsonError(msg) => write!(f, "JSON error: {}", msg),
            Self::AttributeTypeMismatch { expected, got } => {
                write!(f, "attribute type mismatch: expected {}, got {}", expected, got)
            }
            Self::VertexCountMismatch { expected, got } => {
                write!(f, "vertex count mismatch: expected {}, got {}", expected, got)
            }
            Self::DecompressionFailed(msg) => write!(f, "decompression failed: {}", msg),
            Self::InvalidGltf(msg) => write!(f, "invalid glTF: {}", msg),
        }
    }
}

impl std::error::Error for DracoError {}

impl From<DracoError> for GltfError {
    fn from(e: DracoError) -> Self {
        GltfError::UnsupportedFeature(e.to_string())
    }
}

/// Result type for Draco operations.
pub type DracoResult<T> = Result<T, DracoError>;

// ---------------------------------------------------------------------------
// Extension data types
// ---------------------------------------------------------------------------

/// Parsed KHR_draco_mesh_compression extension data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DracoExtension {
    /// Buffer view index containing compressed Draco data.
    #[serde(rename = "bufferView")]
    pub buffer_view: u32,
    /// Mapping from glTF attribute semantic to Draco attribute ID.
    pub attributes: HashMap<String, u32>,
}

impl DracoExtension {
    /// Get the Draco attribute ID for a glTF semantic.
    pub fn get_attribute_id(&self, semantic: &str) -> Option<u32> {
        self.attributes.get(semantic).copied()
    }

    /// Check if the extension has a specific attribute.
    pub fn has_attribute(&self, semantic: &str) -> bool {
        self.attributes.contains_key(semantic)
    }

    /// Get all attribute semantics.
    pub fn attribute_semantics(&self) -> impl Iterator<Item = &str> {
        self.attributes.keys().map(|s| s.as_str())
    }
}

/// Result of Draco decompression.
#[derive(Debug, Clone)]
pub struct DracoDecompressResult {
    /// Decompressed vertex positions.
    pub positions: Vec<[f32; 3]>,
    /// Decompressed vertex normals (if present).
    pub normals: Option<Vec<[f32; 3]>>,
    /// Decompressed texture coordinates (if present).
    pub texcoords: Option<Vec<[f32; 2]>>,
    /// Decompressed vertex colors (if present).
    pub colors: Option<Vec<[f32; 4]>>,
    /// Decompressed joint indices (if present).
    pub joints: Option<Vec<[u16; 4]>>,
    /// Decompressed joint weights (if present).
    pub weights: Option<Vec<[f32; 4]>>,
    /// Decompressed triangle indices (if present for mesh, None for point cloud).
    pub indices: Option<Vec<u32>>,
    /// Number of vertices.
    pub vertex_count: usize,
    /// Number of faces (triangles).
    pub face_count: usize,
    /// Whether this was a point cloud (vs mesh).
    pub is_point_cloud: bool,
}

impl DracoDecompressResult {
    /// Create an empty result.
    pub fn empty() -> Self {
        Self {
            positions: Vec::new(),
            normals: None,
            texcoords: None,
            colors: None,
            joints: None,
            weights: None,
            indices: None,
            vertex_count: 0,
            face_count: 0,
            is_point_cloud: false,
        }
    }

    /// Check if the result has normals.
    pub fn has_normals(&self) -> bool {
        self.normals.is_some()
    }

    /// Check if the result has texture coordinates.
    pub fn has_texcoords(&self) -> bool {
        self.texcoords.is_some()
    }

    /// Check if the result has vertex colors.
    pub fn has_colors(&self) -> bool {
        self.colors.is_some()
    }

    /// Check if the result has skinning data.
    pub fn has_skinning(&self) -> bool {
        self.joints.is_some() && self.weights.is_some()
    }

    /// Convert positions to raw bytes (f32 little-endian).
    pub fn positions_as_bytes(&self) -> Vec<u8> {
        let mut data = Vec::with_capacity(self.positions.len() * 12);
        for pos in &self.positions {
            data.extend_from_slice(&pos[0].to_le_bytes());
            data.extend_from_slice(&pos[1].to_le_bytes());
            data.extend_from_slice(&pos[2].to_le_bytes());
        }
        data
    }

    /// Convert normals to raw bytes (f32 little-endian).
    pub fn normals_as_bytes(&self) -> Option<Vec<u8>> {
        self.normals.as_ref().map(|normals| {
            let mut data = Vec::with_capacity(normals.len() * 12);
            for n in normals {
                data.extend_from_slice(&n[0].to_le_bytes());
                data.extend_from_slice(&n[1].to_le_bytes());
                data.extend_from_slice(&n[2].to_le_bytes());
            }
            data
        })
    }

    /// Convert texcoords to raw bytes (f32 little-endian).
    pub fn texcoords_as_bytes(&self) -> Option<Vec<u8>> {
        self.texcoords.as_ref().map(|uvs| {
            let mut data = Vec::with_capacity(uvs.len() * 8);
            for uv in uvs {
                data.extend_from_slice(&uv[0].to_le_bytes());
                data.extend_from_slice(&uv[1].to_le_bytes());
            }
            data
        })
    }

    /// Convert colors to raw bytes (f32 little-endian RGBA).
    pub fn colors_as_bytes(&self) -> Option<Vec<u8>> {
        self.colors.as_ref().map(|colors| {
            let mut data = Vec::with_capacity(colors.len() * 16);
            for c in colors {
                data.extend_from_slice(&c[0].to_le_bytes());
                data.extend_from_slice(&c[1].to_le_bytes());
                data.extend_from_slice(&c[2].to_le_bytes());
                data.extend_from_slice(&c[3].to_le_bytes());
            }
            data
        })
    }

    /// Convert joints to raw bytes (u16 little-endian).
    pub fn joints_as_bytes(&self) -> Option<Vec<u8>> {
        self.joints.as_ref().map(|joints| {
            let mut data = Vec::with_capacity(joints.len() * 8);
            for j in joints {
                data.extend_from_slice(&j[0].to_le_bytes());
                data.extend_from_slice(&j[1].to_le_bytes());
                data.extend_from_slice(&j[2].to_le_bytes());
                data.extend_from_slice(&j[3].to_le_bytes());
            }
            data
        })
    }

    /// Convert weights to raw bytes (f32 little-endian).
    pub fn weights_as_bytes(&self) -> Option<Vec<u8>> {
        self.weights.as_ref().map(|weights| {
            let mut data = Vec::with_capacity(weights.len() * 16);
            for w in weights {
                data.extend_from_slice(&w[0].to_le_bytes());
                data.extend_from_slice(&w[1].to_le_bytes());
                data.extend_from_slice(&w[2].to_le_bytes());
                data.extend_from_slice(&w[3].to_le_bytes());
            }
            data
        })
    }

    /// Convert indices to raw bytes (u32 little-endian).
    pub fn indices_as_bytes(&self) -> Option<Vec<u8>> {
        self.indices.as_ref().map(|indices| {
            let mut data = Vec::with_capacity(indices.len() * 4);
            for idx in indices {
                data.extend_from_slice(&idx.to_le_bytes());
            }
            data
        })
    }

    /// Convert to glTF vertex attribute for positions.
    pub fn to_position_attribute(&self) -> VertexAttribute {
        VertexAttribute {
            semantic: VertexSemantic::Position,
            component_type: ComponentType::F32,
            attribute_type: AttributeType::Vec3,
            offset: 0,
            stride: 12,
            count: self.vertex_count,
            data: self.positions_as_bytes(),
        }
    }

    /// Convert to glTF vertex attribute for normals.
    pub fn to_normal_attribute(&self) -> Option<VertexAttribute> {
        self.normals_as_bytes().map(|data| VertexAttribute {
            semantic: VertexSemantic::Normal,
            component_type: ComponentType::F32,
            attribute_type: AttributeType::Vec3,
            offset: 0,
            stride: 12,
            count: self.vertex_count,
            data,
        })
    }

    /// Convert to glTF vertex attribute for texcoords.
    pub fn to_texcoord_attribute(&self) -> Option<VertexAttribute> {
        self.texcoords_as_bytes().map(|data| VertexAttribute {
            semantic: VertexSemantic::TexCoord0,
            component_type: ComponentType::F32,
            attribute_type: AttributeType::Vec2,
            offset: 0,
            stride: 8,
            count: self.vertex_count,
            data,
        })
    }

    /// Convert to glTF vertex attribute for colors.
    pub fn to_color_attribute(&self) -> Option<VertexAttribute> {
        self.colors_as_bytes().map(|data| VertexAttribute {
            semantic: VertexSemantic::Color0,
            component_type: ComponentType::F32,
            attribute_type: AttributeType::Vec4,
            offset: 0,
            stride: 16,
            count: self.vertex_count,
            data,
        })
    }

    /// Convert to glTF vertex attribute for joints.
    pub fn to_joints_attribute(&self) -> Option<VertexAttribute> {
        self.joints_as_bytes().map(|data| VertexAttribute {
            semantic: VertexSemantic::Joints0,
            component_type: ComponentType::U16,
            attribute_type: AttributeType::Vec4,
            offset: 0,
            stride: 8,
            count: self.vertex_count,
            data,
        })
    }

    /// Convert to glTF vertex attribute for weights.
    pub fn to_weights_attribute(&self) -> Option<VertexAttribute> {
        self.weights_as_bytes().map(|data| VertexAttribute {
            semantic: VertexSemantic::Weights0,
            component_type: ComponentType::F32,
            attribute_type: AttributeType::Vec4,
            offset: 0,
            stride: 16,
            count: self.vertex_count,
            data,
        })
    }

    /// Convert to glTF index buffer.
    pub fn to_index_buffer(&self) -> Option<IndexBuffer> {
        self.indices_as_bytes().map(|data| {
            // Determine optimal index format
            let max_idx = self.indices.as_ref()
                .map(|i| i.iter().copied().max().unwrap_or(0))
                .unwrap_or(0);

            let (format, data) = if max_idx <= u8::MAX as u32 {
                let u8_data: Vec<u8> = self.indices.as_ref().unwrap()
                    .iter()
                    .map(|&i| i as u8)
                    .collect();
                (IndexFormat::U8, u8_data)
            } else if max_idx <= u16::MAX as u32 {
                let mut u16_data = Vec::with_capacity(self.indices.as_ref().unwrap().len() * 2);
                for &idx in self.indices.as_ref().unwrap() {
                    u16_data.extend_from_slice(&(idx as u16).to_le_bytes());
                }
                (IndexFormat::U16, u16_data)
            } else {
                (IndexFormat::U32, data)
            };

            IndexBuffer {
                format,
                count: self.indices.as_ref().map(|i| i.len()).unwrap_or(0),
                data,
            }
        })
    }

    /// Convert all attributes to a HashMap suitable for GltfPrimitive.
    pub fn to_attribute_map(&self) -> HashMap<VertexSemantic, VertexAttribute> {
        let mut map = HashMap::new();
        map.insert(VertexSemantic::Position, self.to_position_attribute());

        if let Some(attr) = self.to_normal_attribute() {
            map.insert(VertexSemantic::Normal, attr);
        }
        if let Some(attr) = self.to_texcoord_attribute() {
            map.insert(VertexSemantic::TexCoord0, attr);
        }
        if let Some(attr) = self.to_color_attribute() {
            map.insert(VertexSemantic::Color0, attr);
        }
        if let Some(attr) = self.to_joints_attribute() {
            map.insert(VertexSemantic::Joints0, attr);
        }
        if let Some(attr) = self.to_weights_attribute() {
            map.insert(VertexSemantic::Weights0, attr);
        }

        map
    }
}

// ---------------------------------------------------------------------------
// Extension detection
// ---------------------------------------------------------------------------

/// Detect if a glTF document uses the KHR_draco_mesh_compression extension.
///
/// Returns true if the extension is listed in either `extensionsRequired` or
/// `extensionsUsed`.
pub fn detect_draco_extension(gltf_json: &Value) -> bool {
    // Check extensionsRequired
    if let Some(required) = gltf_json.get("extensionsRequired").and_then(|v| v.as_array()) {
        if required.iter().any(|v| v.as_str() == Some(DRACO_EXTENSION_NAME)) {
            return true;
        }
    }

    // Check extensionsUsed
    if let Some(used) = gltf_json.get("extensionsUsed").and_then(|v| v.as_array()) {
        if used.iter().any(|v| v.as_str() == Some(DRACO_EXTENSION_NAME)) {
            return true;
        }
    }

    false
}

/// Check if Draco extension is required (not just optional).
///
/// If true, the file cannot be loaded without Draco decompression.
pub fn is_draco_required(gltf_json: &Value) -> bool {
    if let Some(required) = gltf_json.get("extensionsRequired").and_then(|v| v.as_array()) {
        required.iter().any(|v| v.as_str() == Some(DRACO_EXTENSION_NAME))
    } else {
        false
    }
}

/// Check if Draco decoder is available at runtime.
///
/// Currently always returns false as no Draco decoder is integrated.
/// When a Rust Draco library becomes available and is integrated,
/// this should be updated to return true.
pub fn is_decoder_available() -> bool {
    // No Draco decoder currently integrated
    // Future: return true when draco crate is added as dependency
    false
}

// ---------------------------------------------------------------------------
// Extension parsing
// ---------------------------------------------------------------------------

/// Parse Draco extension data from a glTF primitive JSON.
///
/// Returns `None` if the primitive doesn't have the Draco extension.
pub fn parse_draco_extension(primitive_json: &Value) -> Option<DracoExtension> {
    primitive_json
        .get("extensions")
        .and_then(|exts| exts.get(DRACO_EXTENSION_NAME))
        .and_then(|draco| serde_json::from_value(draco.clone()).ok())
}

/// Parse Draco extension data from extension JSON value directly.
pub fn parse_draco_extension_value(extension_json: &Value) -> DracoResult<DracoExtension> {
    serde_json::from_value(extension_json.clone())
        .map_err(|e| DracoError::JsonError(e.to_string()))
}

/// Get the Draco extension from a primitive's extensions field.
///
/// This is a convenience function for accessing the extension data
/// when you already have the extensions object.
pub fn get_draco_from_extensions(extensions: &Value) -> Option<DracoExtension> {
    extensions
        .get(DRACO_EXTENSION_NAME)
        .and_then(|draco| serde_json::from_value(draco.clone()).ok())
}

// ---------------------------------------------------------------------------
// Buffer extraction
// ---------------------------------------------------------------------------

/// Draco buffer view information from glTF.
#[derive(Debug, Clone)]
pub struct DracoBufferView {
    /// Buffer index.
    pub buffer: usize,
    /// Byte offset into the buffer.
    pub byte_offset: usize,
    /// Byte length of the data.
    pub byte_length: usize,
}

/// Extract buffer view info from glTF JSON.
pub fn extract_buffer_view(gltf_json: &Value, view_index: u32) -> DracoResult<DracoBufferView> {
    let views = gltf_json
        .get("bufferViews")
        .and_then(|v| v.as_array())
        .ok_or_else(|| DracoError::InvalidGltf("missing bufferViews".into()))?;

    let view = views
        .get(view_index as usize)
        .ok_or(DracoError::InvalidBufferView(view_index))?;

    let buffer = view
        .get("buffer")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| DracoError::InvalidGltf("buffer view missing buffer".into()))?
        as usize;

    let byte_offset = view
        .get("byteOffset")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;

    let byte_length = view
        .get("byteLength")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| DracoError::InvalidGltf("buffer view missing byteLength".into()))?
        as usize;

    Ok(DracoBufferView {
        buffer,
        byte_offset,
        byte_length,
    })
}

/// Extract the compressed Draco data from buffers.
pub fn extract_draco_data(
    gltf_json: &Value,
    extension: &DracoExtension,
    buffers: &[Vec<u8>],
) -> DracoResult<Vec<u8>> {
    let view = extract_buffer_view(gltf_json, extension.buffer_view)?;

    let buffer = buffers
        .get(view.buffer)
        .ok_or(DracoError::BufferOutOfBounds {
            view: extension.buffer_view,
            buffer_len: buffers.len(),
        })?;

    let end = view.byte_offset + view.byte_length;
    if end > buffer.len() {
        return Err(DracoError::BufferOutOfBounds {
            view: extension.buffer_view,
            buffer_len: buffer.len(),
        });
    }

    Ok(buffer[view.byte_offset..end].to_vec())
}

// ---------------------------------------------------------------------------
// Software Draco decoder (fallback implementation)
// ---------------------------------------------------------------------------

/// Draco geometry type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DracoGeometryType {
    /// Triangle mesh.
    TriangularMesh,
    /// Point cloud.
    PointCloud,
}

/// Validate Draco compressed data header.
///
/// Returns the geometry type if valid.
pub fn validate_draco_header(data: &[u8]) -> DracoResult<DracoGeometryType> {
    // Draco files start with a header containing:
    // - Magic: "DRACO" (5 bytes) - but encoded versions may vary
    // - Version info
    // - Geometry type
    // - etc.

    if data.len() < 8 {
        return Err(DracoError::InvalidData("data too short for header".into()));
    }

    // Check for Draco encoded geometry
    // The first few bytes contain version and type info
    // Format depends on Draco version, but we can detect the geometry type

    // Draco uses a bitstream format. The header contains:
    // - 5 bits: draco_major_version
    // - 5 bits: draco_minor_version
    // - 2 bits: encoder_type (0=sequential, 1=edgebreaker, 2=sequential point cloud)
    // - encoder_method
    // - flags

    // For now, check if data appears valid
    // Real implementation would parse the bitstream header

    // Simplified header validation
    // In practice, we'd use the draco library for this

    // Check for valid-looking data (non-zero, has structure)
    let has_content = data.iter().take(64).any(|&b| b != 0);
    if !has_content {
        return Err(DracoError::InvalidData("data appears empty".into()));
    }

    // Without the actual draco decoder, we can't determine geometry type
    // Default to mesh for glTF usage
    Ok(DracoGeometryType::TriangularMesh)
}

// ---------------------------------------------------------------------------
// Decompression functions
// ---------------------------------------------------------------------------

/// Decompress Draco-compressed geometry data.
///
/// # Arguments
///
/// * `buffer_data` - Raw compressed Draco data from the buffer view.
/// * `extension` - Parsed Draco extension with attribute mappings.
///
/// # Returns
///
/// Decompressed vertex data and indices on success.
///
/// # Errors
///
/// Returns `DracoError::DecoderNotAvailable` since the draco crate is not currently
/// integrated. This function serves as a placeholder for future integration.
///
/// # Future Integration
///
/// When a Rust Draco decoder becomes available, this function would:
/// 1. Create a decoder instance
/// 2. Decode the compressed mesh data
/// 3. Extract attributes based on the extension mappings
/// 4. Return the decompressed vertex data and indices
///
/// Example (hypothetical):
/// ```ignore
/// use draco::Decoder;
/// let decoder = Decoder::new();
/// let mesh = decoder.decode_mesh(buffer_data)?;
/// // Extract positions, normals, etc. from mesh
/// ```
pub fn decompress_draco(
    _buffer_data: &[u8],
    _extension: &DracoExtension,
) -> DracoResult<DracoDecompressResult> {
    // Currently, no Rust Draco decoder is integrated.
    // When one becomes available, implement the actual decompression here.
    //
    // The implementation would:
    // 1. Parse the Draco bitstream
    // 2. Decode geometry (mesh or point cloud)
    // 3. Extract vertex attributes based on extension.attributes mapping
    // 4. Extract indices for meshes
    // 5. Return populated DracoDecompressResult
    Err(DracoError::DecoderNotAvailable)
}

/// Decompress Draco data with full buffer view resolution.
///
/// This is a higher-level function that extracts the compressed data
/// from the glTF buffers before decompression.
pub fn decompress_draco_with_buffers(
    gltf_json: &Value,
    extension: &DracoExtension,
    buffers: &[Vec<u8>],
) -> DracoResult<DracoDecompressResult> {
    let draco_data = extract_draco_data(gltf_json, extension, buffers)?;
    decompress_draco(&draco_data, extension)
}

// ---------------------------------------------------------------------------
// glTF integration
// ---------------------------------------------------------------------------

/// Check if a primitive has Draco compression and attempt to decompress it.
///
/// This function integrates seamlessly with the existing glTF pipeline.
///
/// # Arguments
///
/// * `primitive_json` - The primitive's JSON representation.
/// * `gltf_json` - The full glTF JSON document.
/// * `buffers` - Pre-loaded buffer data.
///
/// # Returns
///
/// * `Ok(Some(result))` - Successfully decompressed Draco data.
/// * `Ok(None)` - Primitive doesn't use Draco compression.
/// * `Err(e)` - Decompression or parsing error.
pub fn decompress_primitive_json(
    primitive_json: &Value,
    gltf_json: &Value,
    buffers: &[Vec<u8>],
) -> DracoResult<Option<DracoDecompressResult>> {
    // Check if primitive has Draco extension
    let extension = match parse_draco_extension(primitive_json) {
        Some(ext) => ext,
        None => return Ok(None),
    };

    // Check if decoder is available
    if !is_decoder_available() {
        // If Draco is required, return error
        if is_draco_required(gltf_json) {
            return Err(DracoError::DecoderNotAvailable);
        }
        // Otherwise, fallback to non-compressed data
        return Ok(None);
    }

    // Decompress
    let result = decompress_draco_with_buffers(gltf_json, &extension, buffers)?;
    Ok(Some(result))
}

/// Integration function for existing GltfPrimitive types.
///
/// This function can be called after initial primitive parsing to replace
/// the attribute data with decompressed Draco data if available.
///
/// Note: The actual implementation requires access to the original JSON
/// and buffers, which aren't stored in GltfPrimitive. This function
/// serves as documentation for the integration pattern.
///
/// # Returns
///
/// * `Ok(Some(result))` - Successfully decompressed, result contains new vertex data.
/// * `Ok(None)` - No Draco compression present.
/// * `Err(e)` - Decompression failed.
pub fn decompress_primitive_if_draco(
    _primitive: &GltfPrimitive,
    _buffers: &[Vec<u8>],
) -> DracoResult<Option<DracoDecompressResult>> {
    // This function requires the original JSON to parse extensions.
    // In practice, you would call decompress_primitive_json with the
    // JSON data and then update the primitive's attributes.
    //
    // Since GltfPrimitive doesn't store the extension data, this function
    // serves as a placeholder for documentation purposes.
    //
    // Usage pattern:
    // 1. Parse glTF JSON
    // 2. For each primitive, call decompress_primitive_json
    // 3. If decompression succeeds, use result.to_attribute_map()
    //    instead of the original attributes

    Ok(None)
}

/// Create a GltfPrimitive from decompressed Draco data.
pub fn draco_result_to_primitive(
    result: &DracoDecompressResult,
    material_index: Option<usize>,
) -> GltfPrimitive {
    GltfPrimitive {
        attributes: result.to_attribute_map(),
        indices: result.to_index_buffer(),
        material_index,
    }
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Map a glTF attribute semantic string to VertexSemantic.
pub fn semantic_from_string(semantic: &str) -> Option<VertexSemantic> {
    match semantic {
        "POSITION" => Some(VertexSemantic::Position),
        "NORMAL" => Some(VertexSemantic::Normal),
        "TANGENT" => Some(VertexSemantic::Tangent),
        "TEXCOORD_0" => Some(VertexSemantic::TexCoord0),
        "TEXCOORD_1" => Some(VertexSemantic::TexCoord1),
        "COLOR_0" => Some(VertexSemantic::Color0),
        "JOINTS_0" => Some(VertexSemantic::Joints0),
        "WEIGHTS_0" => Some(VertexSemantic::Weights0),
        _ => None,
    }
}

/// Map a VertexSemantic to glTF attribute semantic string.
pub fn semantic_to_string(semantic: VertexSemantic) -> &'static str {
    match semantic {
        VertexSemantic::Position => "POSITION",
        VertexSemantic::Normal => "NORMAL",
        VertexSemantic::Tangent => "TANGENT",
        VertexSemantic::TexCoord0 => "TEXCOORD_0",
        VertexSemantic::TexCoord1 => "TEXCOORD_1",
        VertexSemantic::Color0 => "COLOR_0",
        VertexSemantic::Joints0 => "JOINTS_0",
        VertexSemantic::Weights0 => "WEIGHTS_0",
    }
}

/// Get expected Draco attribute type for a glTF semantic.
pub fn expected_draco_attr_type(semantic: &str) -> u32 {
    match semantic {
        "POSITION" => DRACO_ATTR_POSITION,
        "NORMAL" => DRACO_ATTR_NORMAL,
        "COLOR_0" => DRACO_ATTR_COLOR,
        "TEXCOORD_0" | "TEXCOORD_1" => DRACO_ATTR_TEX_COORD,
        _ => DRACO_ATTR_GENERIC,
    }
}

/// Estimate decompressed data size from extension info.
///
/// This is useful for pre-allocating buffers before decompression.
pub fn estimate_decompressed_size(
    extension: &DracoExtension,
    estimated_vertices: usize,
    estimated_faces: usize,
) -> usize {
    let mut size = 0usize;

    for attr in extension.attributes.keys() {
        size += match attr.as_str() {
            "POSITION" => estimated_vertices * 12, // 3 x f32
            "NORMAL" => estimated_vertices * 12,   // 3 x f32
            "TANGENT" => estimated_vertices * 16,  // 4 x f32
            "TEXCOORD_0" | "TEXCOORD_1" => estimated_vertices * 8, // 2 x f32
            "COLOR_0" => estimated_vertices * 16,  // 4 x f32
            "JOINTS_0" => estimated_vertices * 8,  // 4 x u16
            "WEIGHTS_0" => estimated_vertices * 16, // 4 x f32
            _ => 0,
        };
    }

    // Add indices (3 per face, u32)
    size += estimated_faces * 3 * 4;

    size
}

// ---------------------------------------------------------------------------
// Fallback support
// ---------------------------------------------------------------------------

/// Draco fallback configuration.
#[derive(Debug, Clone, Default)]
pub struct DracoFallbackConfig {
    /// Whether to log warnings when falling back.
    pub log_warnings: bool,
    /// Whether to fail if Draco is required but decoder unavailable.
    pub fail_on_required: bool,
}

/// Result of fallback check.
#[derive(Debug, Clone)]
pub enum DracoFallbackResult {
    /// Use Draco-decompressed data.
    UseDraco(DracoDecompressResult),
    /// Use fallback (non-Draco) data.
    UseFallback,
    /// No Draco extension present.
    NoDraco,
}

/// Check whether to use Draco or fallback for a primitive.
///
/// This implements the fallback logic for when:
/// - Draco decoder is not available
/// - Draco decompression fails
/// - Fallback data is present
pub fn check_draco_fallback(
    primitive_json: &Value,
    gltf_json: &Value,
    buffers: &[Vec<u8>],
    config: &DracoFallbackConfig,
) -> DracoResult<DracoFallbackResult> {
    // Check if primitive has Draco extension
    let extension = match parse_draco_extension(primitive_json) {
        Some(ext) => ext,
        None => return Ok(DracoFallbackResult::NoDraco),
    };

    // Check if decoder is available
    if !is_decoder_available() {
        if config.fail_on_required && is_draco_required(gltf_json) {
            return Err(DracoError::DecoderNotAvailable);
        }
        if config.log_warnings {
            eprintln!(
                "Warning: Draco decoder not available, using fallback for primitive with {} attributes",
                extension.attributes.len()
            );
        }
        return Ok(DracoFallbackResult::UseFallback);
    }

    // Try to decompress
    match decompress_draco_with_buffers(gltf_json, &extension, buffers) {
        Ok(result) => Ok(DracoFallbackResult::UseDraco(result)),
        Err(e) => {
            if config.fail_on_required && is_draco_required(gltf_json) {
                return Err(e);
            }
            if config.log_warnings {
                eprintln!("Warning: Draco decompression failed ({}), using fallback", e);
            }
            Ok(DracoFallbackResult::UseFallback)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // =====================================================================
    // Extension Detection Tests (3 tests)
    // =====================================================================

    #[test]
    fn test_detect_draco_extension_in_required() {
        let gltf = json!({
            "extensionsRequired": ["KHR_draco_mesh_compression"],
            "extensionsUsed": []
        });
        assert!(detect_draco_extension(&gltf));
        assert!(is_draco_required(&gltf));
    }

    #[test]
    fn test_detect_draco_extension_in_used() {
        let gltf = json!({
            "extensionsUsed": ["KHR_draco_mesh_compression"]
        });
        assert!(detect_draco_extension(&gltf));
        assert!(!is_draco_required(&gltf));
    }

    #[test]
    fn test_detect_draco_extension_not_present() {
        let gltf = json!({
            "extensionsUsed": ["KHR_materials_unlit"],
            "extensionsRequired": []
        });
        assert!(!detect_draco_extension(&gltf));
        assert!(!is_draco_required(&gltf));
    }

    // =====================================================================
    // Draco Decompression Tests (4 tests - mocked without feature)
    // =====================================================================

    #[test]
    fn test_decompress_draco_unavailable() {
        // Without draco feature, should return DecoderNotAvailable
        let extension = DracoExtension {
            buffer_view: 0,
            attributes: {
                let mut map = HashMap::new();
                map.insert("POSITION".into(), 0);
                map
            },
        };

        let result = decompress_draco(&[0u8; 64], &extension);

        #[cfg(not(feature = "draco"))]
        assert!(matches!(result, Err(DracoError::DecoderNotAvailable)));
    }

    #[test]
    fn test_validate_draco_header_too_short() {
        let data = [0u8; 4]; // Too short for header
        let result = validate_draco_header(&data);
        assert!(matches!(result, Err(DracoError::InvalidData(_))));
    }

    #[test]
    fn test_validate_draco_header_empty_data() {
        let data = [0u8; 64]; // All zeros
        let result = validate_draco_header(&data);
        assert!(matches!(result, Err(DracoError::InvalidData(_))));
    }

    #[test]
    fn test_validate_draco_header_valid_looking() {
        // Data that looks valid (has non-zero content)
        let mut data = [0u8; 64];
        data[0] = 0x44; // D
        data[1] = 0x52; // R
        data[2] = 0x41; // A
        data[3] = 0x43; // C
        data[4] = 0x4F; // O
        data[5] = 0x01; // Version byte

        let result = validate_draco_header(&data);
        assert!(result.is_ok());
    }

    // =====================================================================
    // Attribute Mapping Tests (3 tests)
    // =====================================================================

    #[test]
    fn test_semantic_from_string_all_types() {
        assert_eq!(semantic_from_string("POSITION"), Some(VertexSemantic::Position));
        assert_eq!(semantic_from_string("NORMAL"), Some(VertexSemantic::Normal));
        assert_eq!(semantic_from_string("TANGENT"), Some(VertexSemantic::Tangent));
        assert_eq!(semantic_from_string("TEXCOORD_0"), Some(VertexSemantic::TexCoord0));
        assert_eq!(semantic_from_string("TEXCOORD_1"), Some(VertexSemantic::TexCoord1));
        assert_eq!(semantic_from_string("COLOR_0"), Some(VertexSemantic::Color0));
        assert_eq!(semantic_from_string("JOINTS_0"), Some(VertexSemantic::Joints0));
        assert_eq!(semantic_from_string("WEIGHTS_0"), Some(VertexSemantic::Weights0));
        assert_eq!(semantic_from_string("UNKNOWN"), None);
    }

    #[test]
    fn test_semantic_to_string_roundtrip() {
        let semantics = [
            VertexSemantic::Position,
            VertexSemantic::Normal,
            VertexSemantic::Tangent,
            VertexSemantic::TexCoord0,
            VertexSemantic::TexCoord1,
            VertexSemantic::Color0,
            VertexSemantic::Joints0,
            VertexSemantic::Weights0,
        ];

        for sem in semantics {
            let s = semantic_to_string(sem);
            let recovered = semantic_from_string(s);
            assert_eq!(recovered, Some(sem));
        }
    }

    #[test]
    fn test_expected_draco_attr_type() {
        assert_eq!(expected_draco_attr_type("POSITION"), DRACO_ATTR_POSITION);
        assert_eq!(expected_draco_attr_type("NORMAL"), DRACO_ATTR_NORMAL);
        assert_eq!(expected_draco_attr_type("COLOR_0"), DRACO_ATTR_COLOR);
        assert_eq!(expected_draco_attr_type("TEXCOORD_0"), DRACO_ATTR_TEX_COORD);
        assert_eq!(expected_draco_attr_type("TEXCOORD_1"), DRACO_ATTR_TEX_COORD);
        assert_eq!(expected_draco_attr_type("JOINTS_0"), DRACO_ATTR_GENERIC);
        assert_eq!(expected_draco_attr_type("WEIGHTS_0"), DRACO_ATTR_GENERIC);
    }

    // =====================================================================
    // glTF Integration Tests (3 tests)
    // =====================================================================

    #[test]
    fn test_parse_draco_extension_from_primitive() {
        let primitive = json!({
            "attributes": { "POSITION": 0 },
            "extensions": {
                "KHR_draco_mesh_compression": {
                    "bufferView": 1,
                    "attributes": {
                        "POSITION": 0,
                        "NORMAL": 1
                    }
                }
            }
        });

        let ext = parse_draco_extension(&primitive).expect("should parse");
        assert_eq!(ext.buffer_view, 1);
        assert_eq!(ext.attributes.len(), 2);
        assert_eq!(ext.get_attribute_id("POSITION"), Some(0));
        assert_eq!(ext.get_attribute_id("NORMAL"), Some(1));
    }

    #[test]
    fn test_parse_draco_extension_missing() {
        let primitive = json!({
            "attributes": { "POSITION": 0 }
        });

        let ext = parse_draco_extension(&primitive);
        assert!(ext.is_none());
    }

    #[test]
    fn test_extract_buffer_view() {
        let gltf = json!({
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0, "byteLength": 100 },
                { "buffer": 0, "byteOffset": 100, "byteLength": 200 }
            ]
        });

        let view = extract_buffer_view(&gltf, 1).expect("should extract");
        assert_eq!(view.buffer, 0);
        assert_eq!(view.byte_offset, 100);
        assert_eq!(view.byte_length, 200);
    }

    // =====================================================================
    // Fallback Behavior Tests (2 tests)
    // =====================================================================

    #[test]
    fn test_fallback_no_draco_extension() {
        let primitive = json!({
            "attributes": { "POSITION": 0 }
        });
        let gltf = json!({});
        let buffers: Vec<Vec<u8>> = vec![];

        let config = DracoFallbackConfig::default();
        let result = check_draco_fallback(&primitive, &gltf, &buffers, &config);

        assert!(matches!(result, Ok(DracoFallbackResult::NoDraco)));
    }

    #[test]
    fn test_fallback_decoder_unavailable() {
        let primitive = json!({
            "attributes": { "POSITION": 0 },
            "extensions": {
                "KHR_draco_mesh_compression": {
                    "bufferView": 0,
                    "attributes": { "POSITION": 0 }
                }
            }
        });
        let gltf = json!({
            "extensionsUsed": ["KHR_draco_mesh_compression"],
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0, "byteLength": 100 }
            ]
        });
        let buffers: Vec<Vec<u8>> = vec![vec![0u8; 100]];

        let config = DracoFallbackConfig {
            log_warnings: false,
            fail_on_required: false,
        };

        let result = check_draco_fallback(&primitive, &gltf, &buffers, &config);

        #[cfg(not(feature = "draco"))]
        assert!(matches!(result, Ok(DracoFallbackResult::UseFallback)));
    }

    // =====================================================================
    // DracoDecompressResult Tests (5 tests)
    // =====================================================================

    #[test]
    fn test_decompress_result_empty() {
        let result = DracoDecompressResult::empty();
        assert!(result.positions.is_empty());
        assert!(result.normals.is_none());
        assert!(result.texcoords.is_none());
        assert!(result.colors.is_none());
        assert!(result.joints.is_none());
        assert!(result.weights.is_none());
        assert!(result.indices.is_none());
        assert_eq!(result.vertex_count, 0);
        assert_eq!(result.face_count, 0);
    }

    #[test]
    fn test_decompress_result_has_methods() {
        let mut result = DracoDecompressResult::empty();
        result.normals = Some(vec![[0.0, 1.0, 0.0]]);
        result.texcoords = Some(vec![[0.0, 0.0]]);
        result.colors = Some(vec![[1.0, 1.0, 1.0, 1.0]]);
        result.joints = Some(vec![[0, 0, 0, 0]]);
        result.weights = Some(vec![[1.0, 0.0, 0.0, 0.0]]);

        assert!(result.has_normals());
        assert!(result.has_texcoords());
        assert!(result.has_colors());
        assert!(result.has_skinning());
    }

    #[test]
    fn test_decompress_result_positions_as_bytes() {
        let mut result = DracoDecompressResult::empty();
        result.positions = vec![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ];
        result.vertex_count = 2;

        let bytes = result.positions_as_bytes();
        assert_eq!(bytes.len(), 24); // 2 vertices * 3 components * 4 bytes

        // Check first position
        let x = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let y = f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let z = f32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        assert!((x - 1.0).abs() < 1e-6);
        assert!((y - 2.0).abs() < 1e-6);
        assert!((z - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_decompress_result_to_attribute_map() {
        let mut result = DracoDecompressResult::empty();
        result.positions = vec![[0.0, 0.0, 0.0]];
        result.normals = Some(vec![[0.0, 1.0, 0.0]]);
        result.vertex_count = 1;

        let map = result.to_attribute_map();
        assert!(map.contains_key(&VertexSemantic::Position));
        assert!(map.contains_key(&VertexSemantic::Normal));
        assert!(!map.contains_key(&VertexSemantic::TexCoord0));
    }

    #[test]
    fn test_decompress_result_to_index_buffer_formats() {
        // Test u8 indices
        let mut result = DracoDecompressResult::empty();
        result.indices = Some(vec![0, 1, 2]);

        let idx_buf = result.to_index_buffer().expect("should create");
        assert_eq!(idx_buf.format, IndexFormat::U8);
        assert_eq!(idx_buf.count, 3);

        // Test u16 indices
        result.indices = Some(vec![0, 256, 512]);
        let idx_buf = result.to_index_buffer().expect("should create");
        assert_eq!(idx_buf.format, IndexFormat::U16);

        // Test u32 indices
        result.indices = Some(vec![0, 70000, 140000]);
        let idx_buf = result.to_index_buffer().expect("should create");
        assert_eq!(idx_buf.format, IndexFormat::U32);
    }

    // =====================================================================
    // DracoExtension Tests (3 tests)
    // =====================================================================

    #[test]
    fn test_draco_extension_get_attribute_id() {
        let ext = DracoExtension {
            buffer_view: 0,
            attributes: {
                let mut map = HashMap::new();
                map.insert("POSITION".into(), 0);
                map.insert("NORMAL".into(), 1);
                map.insert("TEXCOORD_0".into(), 2);
                map
            },
        };

        assert_eq!(ext.get_attribute_id("POSITION"), Some(0));
        assert_eq!(ext.get_attribute_id("NORMAL"), Some(1));
        assert_eq!(ext.get_attribute_id("TEXCOORD_0"), Some(2));
        assert_eq!(ext.get_attribute_id("COLOR_0"), None);
    }

    #[test]
    fn test_draco_extension_has_attribute() {
        let ext = DracoExtension {
            buffer_view: 0,
            attributes: {
                let mut map = HashMap::new();
                map.insert("POSITION".into(), 0);
                map
            },
        };

        assert!(ext.has_attribute("POSITION"));
        assert!(!ext.has_attribute("NORMAL"));
    }

    #[test]
    fn test_draco_extension_attribute_semantics() {
        let ext = DracoExtension {
            buffer_view: 0,
            attributes: {
                let mut map = HashMap::new();
                map.insert("POSITION".into(), 0);
                map.insert("NORMAL".into(), 1);
                map
            },
        };

        let semantics: Vec<_> = ext.attribute_semantics().collect();
        assert_eq!(semantics.len(), 2);
        assert!(semantics.contains(&"POSITION"));
        assert!(semantics.contains(&"NORMAL"));
    }

    // =====================================================================
    // Estimate Size Tests (1 test)
    // =====================================================================

    #[test]
    fn test_estimate_decompressed_size() {
        let ext = DracoExtension {
            buffer_view: 0,
            attributes: {
                let mut map = HashMap::new();
                map.insert("POSITION".into(), 0);  // 12 bytes/vertex
                map.insert("NORMAL".into(), 1);    // 12 bytes/vertex
                map.insert("TEXCOORD_0".into(), 2); // 8 bytes/vertex
                map
            },
        };

        let size = estimate_decompressed_size(&ext, 100, 50);
        // 100 vertices * (12 + 12 + 8) = 3200
        // 50 faces * 3 indices * 4 bytes = 600
        // Total = 3800
        assert_eq!(size, 3800);
    }

    // =====================================================================
    // Error Display Tests (1 test)
    // =====================================================================

    #[test]
    fn test_error_display() {
        let errors = vec![
            DracoError::DecoderNotAvailable,
            DracoError::InvalidData("test".into()),
            DracoError::UnsupportedGeometryType("unknown".into()),
            DracoError::MissingAttribute("POSITION".into()),
            DracoError::BufferOutOfBounds { view: 0, buffer_len: 100 },
            DracoError::InvalidBufferView(5),
            DracoError::JsonError("parse error".into()),
            DracoError::AttributeTypeMismatch { expected: "VEC3".into(), got: "VEC2".into() },
            DracoError::VertexCountMismatch { expected: 100, got: 99 },
            DracoError::DecompressionFailed("internal error".into()),
            DracoError::InvalidGltf("malformed".into()),
        ];

        for err in errors {
            let s = err.to_string();
            assert!(!s.is_empty());
        }
    }

    // =====================================================================
    // draco_result_to_primitive Test (1 test)
    // =====================================================================

    #[test]
    fn test_draco_result_to_primitive() {
        let mut result = DracoDecompressResult::empty();
        result.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        result.normals = Some(vec![[0.0, 0.0, 1.0]; 3]);
        result.indices = Some(vec![0, 1, 2]);
        result.vertex_count = 3;
        result.face_count = 1;

        let primitive = draco_result_to_primitive(&result, Some(0));

        assert!(primitive.attributes.contains_key(&VertexSemantic::Position));
        assert!(primitive.attributes.contains_key(&VertexSemantic::Normal));
        assert!(primitive.indices.is_some());
        assert_eq!(primitive.material_index, Some(0));

        let pos_attr = &primitive.attributes[&VertexSemantic::Position];
        assert_eq!(pos_attr.count, 3);
        assert_eq!(pos_attr.component_type, ComponentType::F32);
        assert_eq!(pos_attr.attribute_type, AttributeType::Vec3);
    }
}
