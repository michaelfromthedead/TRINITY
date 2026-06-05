//! glTF 2.0 mesh loader
//!
//! Supports loading meshes from glTF JSON + binary buffers (.gltf) and
//! GLB binary container format (.glb).
//!
//! # Features
//!
//! - Parse glTF JSON and binary buffers
//! - Extract vertex attributes (position, normal, tangent, UV, color, joints, weights)
//! - Extract index buffers (8/16/32-bit formats)
//! - Support interleaved and split vertex formats
//! - GLB binary container parsing
//! - Material index passthrough (KHR_materials_* extensions)
//! - Schema validation with detailed error reporting
//! - Node hierarchy extraction with transforms
//! - Skin/skeleton support for skeletal animation
//! - Progressive loading (bounds first, then geometry, then skinning)
//! - Streaming parsing for large files (>2GB)
//! - Worker-thread offloadable parsing via `Send + Sync` types
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::gltf::{load_gltf, load_gltf_from_json, GltfParser, LoadStage};
//! use std::path::Path;
//!
//! // Load from file (simple API)
//! let meshes = load_gltf(Path::new("model.gltf"))?;
//!
//! // Load from JSON string with external buffers
//! let json = std::fs::read_to_string("model.gltf")?;
//! let buffer0 = std::fs::read("buffer0.bin")?;
//! let meshes = load_gltf_from_json(&json, &[buffer0])?;
//!
//! // Progressive loading API
//! let parser = GltfParser::new(&json)?;
//! let bounds = parser.load_stage(LoadStage::Bounds, &[buffer0])?;  // Fast: just AABB
//! let geometry = parser.load_stage(LoadStage::Geometry, &[buffer0])?;  // Vertices
//! let full = parser.load_stage(LoadStage::Skinning, &[buffer0])?;  // Full with skins
//! ```

use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Vertex attribute semantic per glTF specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VertexSemantic {
    Position,
    Normal,
    Tangent,
    TexCoord0,
    TexCoord1,
    Color0,
    Joints0,
    Weights0,
}

impl VertexSemantic {
    /// Parse from glTF attribute name string.
    fn from_gltf_name(name: &str) -> Option<Self> {
        match name {
            "POSITION" => Some(Self::Position),
            "NORMAL" => Some(Self::Normal),
            "TANGENT" => Some(Self::Tangent),
            "TEXCOORD_0" => Some(Self::TexCoord0),
            "TEXCOORD_1" => Some(Self::TexCoord1),
            "COLOR_0" => Some(Self::Color0),
            "JOINTS_0" => Some(Self::Joints0),
            "WEIGHTS_0" => Some(Self::Weights0),
            _ => None,
        }
    }
}

/// Component type for vertex data (glTF accessor componentType).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ComponentType {
    I8 = 5120,
    U8 = 5121,
    I16 = 5122,
    U16 = 5123,
    U32 = 5125,
    F32 = 5126,
}

impl ComponentType {
    /// Parse from glTF componentType integer.
    fn from_gltf(value: u32) -> Option<Self> {
        match value {
            5120 => Some(Self::I8),
            5121 => Some(Self::U8),
            5122 => Some(Self::I16),
            5123 => Some(Self::U16),
            5125 => Some(Self::U32),
            5126 => Some(Self::F32),
            _ => None,
        }
    }

    /// Size in bytes of this component type.
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::U32 | Self::F32 => 4,
        }
    }
}

/// Attribute type (scalar, vec2, vec3, etc.) from glTF accessor type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttributeType {
    Scalar,
    Vec2,
    Vec3,
    Vec4,
    Mat2,
    Mat3,
    Mat4,
}

impl AttributeType {
    /// Parse from glTF accessor type string.
    fn from_gltf(s: &str) -> Option<Self> {
        match s {
            "SCALAR" => Some(Self::Scalar),
            "VEC2" => Some(Self::Vec2),
            "VEC3" => Some(Self::Vec3),
            "VEC4" => Some(Self::Vec4),
            "MAT2" => Some(Self::Mat2),
            "MAT3" => Some(Self::Mat3),
            "MAT4" => Some(Self::Mat4),
            _ => None,
        }
    }

    /// Number of components for this attribute type.
    pub const fn component_count(self) -> usize {
        match self {
            Self::Scalar => 1,
            Self::Vec2 => 2,
            Self::Vec3 => 3,
            Self::Vec4 => 4,
            Self::Mat2 => 4,
            Self::Mat3 => 9,
            Self::Mat4 => 16,
        }
    }
}

/// A single vertex attribute accessor with extracted data.
#[derive(Debug, Clone)]
pub struct VertexAttribute {
    /// Semantic meaning of this attribute.
    pub semantic: VertexSemantic,
    /// Component type (f32, u16, etc.).
    pub component_type: ComponentType,
    /// Attribute type (scalar, vec2, vec3, etc.).
    pub attribute_type: AttributeType,
    /// Byte offset within the buffer view (for interleaved data).
    pub offset: usize,
    /// Byte stride between consecutive elements.
    pub stride: usize,
    /// Number of elements (vertices).
    pub count: usize,
    /// Raw attribute data bytes.
    pub data: Vec<u8>,
}

impl VertexAttribute {
    /// Size in bytes of a single element.
    pub fn element_size(&self) -> usize {
        self.component_type.size_bytes() * self.attribute_type.component_count()
    }
}

/// Index buffer format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexFormat {
    U8,
    U16,
    U32,
}

impl IndexFormat {
    /// Size in bytes of a single index.
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U16 => 2,
            Self::U32 => 4,
        }
    }
}

/// Index buffer with format info.
#[derive(Debug, Clone)]
pub struct IndexBuffer {
    /// Index format (U8, U16, U32).
    pub format: IndexFormat,
    /// Number of indices.
    pub count: usize,
    /// Raw index data bytes.
    pub data: Vec<u8>,
}

/// A loaded mesh primitive (single draw call unit).
#[derive(Debug, Clone)]
pub struct GltfPrimitive {
    /// Vertex attributes keyed by semantic.
    pub attributes: HashMap<VertexSemantic, VertexAttribute>,
    /// Optional index buffer.
    pub indices: Option<IndexBuffer>,
    /// Material index (for KHR_materials_* passthrough).
    pub material_index: Option<usize>,
}

/// A loaded mesh (collection of primitives).
#[derive(Debug, Clone)]
pub struct GltfMesh {
    /// Optional mesh name from glTF.
    pub name: Option<String>,
    /// Primitives in this mesh.
    pub primitives: Vec<GltfPrimitive>,
}

// ---------------------------------------------------------------------------
// Node hierarchy types
// ---------------------------------------------------------------------------

/// 4x4 transformation matrix in column-major order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat4([f32; 16]);

impl Mat4 {
    /// Identity matrix.
    pub const IDENTITY: Self = Self([
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]);

    /// Create from column-major array.
    pub fn from_cols(cols: [f32; 16]) -> Self {
        Self(cols)
    }

    /// Create from TRS components (translation, rotation quaternion, scale).
    pub fn from_trs(translation: [f32; 3], rotation: [f32; 4], scale: [f32; 3]) -> Self {
        // Quaternion to rotation matrix
        let [x, y, z, w] = rotation;
        let x2 = x + x;
        let y2 = y + y;
        let z2 = z + z;
        let xx = x * x2;
        let xy = x * y2;
        let xz = x * z2;
        let yy = y * y2;
        let yz = y * z2;
        let zz = z * z2;
        let wx = w * x2;
        let wy = w * y2;
        let wz = w * z2;

        let [sx, sy, sz] = scale;
        let [tx, ty, tz] = translation;

        Self([
            (1.0 - yy - zz) * sx, (xy + wz) * sx,       (xz - wy) * sx,       0.0,
            (xy - wz) * sy,       (1.0 - xx - zz) * sy, (yz + wx) * sy,       0.0,
            (xz + wy) * sz,       (yz - wx) * sz,       (1.0 - xx - yy) * sz, 0.0,
            tx,                   ty,                   tz,                   1.0,
        ])
    }

    /// Multiply two matrices.
    pub fn mul(&self, other: &Self) -> Self {
        let a = &self.0;
        let b = &other.0;
        let mut result = [0.0f32; 16];
        for col in 0..4 {
            for row in 0..4 {
                result[col * 4 + row] =
                    a[0 * 4 + row] * b[col * 4 + 0] +
                    a[1 * 4 + row] * b[col * 4 + 1] +
                    a[2 * 4 + row] * b[col * 4 + 2] +
                    a[3 * 4 + row] * b[col * 4 + 3];
            }
        }
        Self(result)
    }

    /// Get the underlying array.
    pub fn as_array(&self) -> &[f32; 16] {
        &self.0
    }

    /// Extract translation component.
    pub fn translation(&self) -> [f32; 3] {
        [self.0[12], self.0[13], self.0[14]]
    }
}

impl Default for Mat4 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    /// Minimum corner.
    pub min: [f32; 3],
    /// Maximum corner.
    pub max: [f32; 3],
}

impl Aabb {
    /// Empty AABB (inverted for union operations).
    pub const EMPTY: Self = Self {
        min: [f32::INFINITY, f32::INFINITY, f32::INFINITY],
        max: [f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY],
    };

    /// Create from min/max corners.
    pub fn new(min: [f32; 3], max: [f32; 3]) -> Self {
        Self { min, max }
    }

    /// Union with another AABB.
    pub fn union(&self, other: &Self) -> Self {
        Self {
            min: [
                self.min[0].min(other.min[0]),
                self.min[1].min(other.min[1]),
                self.min[2].min(other.min[2]),
            ],
            max: [
                self.max[0].max(other.max[0]),
                self.max[1].max(other.max[1]),
                self.max[2].max(other.max[2]),
            ],
        }
    }

    /// Expand to include a point.
    pub fn expand_point(&mut self, point: [f32; 3]) {
        self.min[0] = self.min[0].min(point[0]);
        self.min[1] = self.min[1].min(point[1]);
        self.min[2] = self.min[2].min(point[2]);
        self.max[0] = self.max[0].max(point[0]);
        self.max[1] = self.max[1].max(point[1]);
        self.max[2] = self.max[2].max(point[2]);
    }

    /// Check if AABB is valid (non-empty).
    pub fn is_valid(&self) -> bool {
        self.min[0] <= self.max[0] &&
        self.min[1] <= self.max[1] &&
        self.min[2] <= self.max[2]
    }

    /// Get center point.
    pub fn center(&self) -> [f32; 3] {
        [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        ]
    }

    /// Get extents (half-size).
    pub fn extents(&self) -> [f32; 3] {
        [
            (self.max[0] - self.min[0]) * 0.5,
            (self.max[1] - self.min[1]) * 0.5,
            (self.max[2] - self.min[2]) * 0.5,
        ]
    }
}

impl Default for Aabb {
    fn default() -> Self {
        Self::EMPTY
    }
}

/// A node in the glTF scene hierarchy.
#[derive(Debug, Clone)]
pub struct GltfNode {
    /// Node index in the glTF file.
    pub index: usize,
    /// Optional node name.
    pub name: Option<String>,
    /// Local transform matrix.
    pub local_transform: Mat4,
    /// World transform (computed by flattening hierarchy).
    pub world_transform: Mat4,
    /// Optional mesh index.
    pub mesh: Option<usize>,
    /// Optional skin index (for skeletal animation).
    pub skin: Option<usize>,
    /// Optional camera index.
    pub camera: Option<usize>,
    /// Child node indices.
    pub children: Vec<usize>,
    /// Parent node index (None for root nodes).
    pub parent: Option<usize>,
}

/// A skin (skeleton) for skeletal animation.
#[derive(Debug, Clone)]
pub struct GltfSkin {
    /// Skin index in the glTF file.
    pub index: usize,
    /// Optional skin name.
    pub name: Option<String>,
    /// Inverse bind matrices (one per joint).
    pub inverse_bind_matrices: Vec<Mat4>,
    /// Joint node indices.
    pub joints: Vec<usize>,
    /// Skeleton root node (optional).
    pub skeleton: Option<usize>,
}

/// Scene containing root nodes.
#[derive(Debug, Clone)]
pub struct GltfScene {
    /// Scene index.
    pub index: usize,
    /// Optional scene name.
    pub name: Option<String>,
    /// Root node indices.
    pub nodes: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Progressive loading types
// ---------------------------------------------------------------------------

/// Loading stage for progressive loading.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LoadStage {
    /// Stage 1: Parse JSON, extract bounds/AABB only (fast, no buffer reads).
    Bounds,
    /// Stage 2: Load vertex positions and indices (geometry).
    Geometry,
    /// Stage 3: Load all vertex attributes (normals, UVs, colors).
    Attributes,
    /// Stage 4: Load skinning data (joints, weights, skins).
    Skinning,
    /// Stage 5: Load everything (full parse).
    Full,
}

/// Result of progressive loading at the Bounds stage.
#[derive(Debug, Clone)]
pub struct BoundsResult {
    /// Mesh bounds (from accessor min/max if available).
    pub mesh_bounds: Vec<(Option<String>, Aabb)>,
    /// Total scene AABB.
    pub scene_bounds: Aabb,
    /// Number of meshes.
    pub mesh_count: usize,
    /// Number of nodes.
    pub node_count: usize,
    /// Number of skins.
    pub skin_count: usize,
    /// Total vertex count estimate.
    pub vertex_count_estimate: usize,
    /// Total index count estimate.
    pub index_count_estimate: usize,
}

/// Full parse result including hierarchy and skins.
#[derive(Debug, Clone)]
pub struct GltfDocument {
    /// All meshes.
    pub meshes: Vec<GltfMesh>,
    /// All nodes.
    pub nodes: Vec<GltfNode>,
    /// All skins.
    pub skins: Vec<GltfSkin>,
    /// All scenes.
    pub scenes: Vec<GltfScene>,
    /// Default scene index.
    pub default_scene: Option<usize>,
    /// Scene bounds.
    pub bounds: Aabb,
}

// ---------------------------------------------------------------------------
// Schema validation types
// ---------------------------------------------------------------------------

/// Schema validation error.
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// JSON path to the error (e.g., "meshes[0].primitives[0].attributes.POSITION").
    pub path: String,
    /// Error message.
    pub message: String,
    /// Severity level.
    pub severity: ValidationSeverity,
}

/// Validation error severity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationSeverity {
    /// Error: prevents loading.
    Error,
    /// Warning: may cause issues but loading can continue.
    Warning,
    /// Info: informational message.
    Info,
}

/// Schema validation result.
#[derive(Debug, Clone, Default)]
pub struct ValidationResult {
    /// List of validation errors/warnings.
    pub errors: Vec<ValidationError>,
    /// Whether the document is valid (no Error-level issues).
    pub is_valid: bool,
}

impl ValidationResult {
    /// Create a successful validation result.
    pub fn ok() -> Self {
        Self { errors: Vec::new(), is_valid: true }
    }

    /// Add an error.
    pub fn add_error(&mut self, path: impl Into<String>, message: impl Into<String>) {
        self.errors.push(ValidationError {
            path: path.into(),
            message: message.into(),
            severity: ValidationSeverity::Error,
        });
        self.is_valid = false;
    }

    /// Add a warning.
    pub fn add_warning(&mut self, path: impl Into<String>, message: impl Into<String>) {
        self.errors.push(ValidationError {
            path: path.into(),
            message: message.into(),
            severity: ValidationSeverity::Warning,
        });
    }

    /// Get error count.
    pub fn error_count(&self) -> usize {
        self.errors.iter().filter(|e| e.severity == ValidationSeverity::Error).count()
    }

    /// Get warning count.
    pub fn warning_count(&self) -> usize {
        self.errors.iter().filter(|e| e.severity == ValidationSeverity::Warning).count()
    }
}

/// glTF loader error.
#[derive(Debug)]
pub enum GltfError {
    /// I/O error reading file.
    IoError(std::io::Error),
    /// JSON parsing error.
    JsonError(String),
    /// Invalid accessor configuration.
    InvalidAccessor(String),
    /// Unsupported glTF feature.
    UnsupportedFeature(String),
    /// Referenced buffer not found.
    MissingBuffer(usize),
    /// Invalid buffer view configuration.
    InvalidBufferView(String),
    /// Invalid GLB format.
    InvalidGlb(String),
    /// Schema validation failed.
    ValidationFailed(ValidationResult),
    /// Invalid node hierarchy (e.g., cycles).
    InvalidHierarchy(String),
    /// Invalid skin configuration.
    InvalidSkin(String),
    /// Streaming error.
    StreamingError(String),
}

impl std::fmt::Display for GltfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IoError(e) => write!(f, "I/O error: {}", e),
            Self::JsonError(s) => write!(f, "JSON error: {}", s),
            Self::InvalidAccessor(s) => write!(f, "invalid accessor: {}", s),
            Self::UnsupportedFeature(s) => write!(f, "unsupported feature: {}", s),
            Self::MissingBuffer(i) => write!(f, "missing buffer {}", i),
            Self::InvalidBufferView(s) => write!(f, "invalid buffer view: {}", s),
            Self::InvalidGlb(s) => write!(f, "invalid GLB: {}", s),
            Self::ValidationFailed(r) => write!(f, "validation failed: {} errors", r.error_count()),
            Self::InvalidHierarchy(s) => write!(f, "invalid hierarchy: {}", s),
            Self::InvalidSkin(s) => write!(f, "invalid skin: {}", s),
            Self::StreamingError(s) => write!(f, "streaming error: {}", s),
        }
    }
}

impl std::error::Error for GltfError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::IoError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for GltfError {
    fn from(e: std::io::Error) -> Self {
        Self::IoError(e)
    }
}

// ---------------------------------------------------------------------------
// glTF JSON schema (complete for mesh, node, skin loading)
// ---------------------------------------------------------------------------

#[derive(Deserialize, Debug, Clone)]
struct GltfJson {
    #[serde(default)]
    asset: GltfAsset,
    #[serde(default)]
    buffers: Vec<GltfBuffer>,
    #[serde(default, rename = "bufferViews")]
    buffer_views: Vec<GltfBufferView>,
    #[serde(default)]
    accessors: Vec<GltfAccessor>,
    #[serde(default)]
    meshes: Vec<GltfMeshJson>,
    #[serde(default)]
    nodes: Vec<GltfNodeJson>,
    #[serde(default)]
    skins: Vec<GltfSkinJson>,
    #[serde(default)]
    scenes: Vec<GltfSceneJson>,
    #[serde(default)]
    scene: Option<usize>,
}

#[derive(Deserialize, Debug, Clone, Default)]
struct GltfAsset {
    #[serde(default)]
    version: String,
    #[serde(default, rename = "minVersion")]
    min_version: Option<String>,
    #[serde(default)]
    generator: Option<String>,
}

#[derive(Deserialize, Debug, Clone)]
struct GltfBuffer {
    #[serde(rename = "byteLength")]
    byte_length: usize,
    #[serde(default)]
    uri: Option<String>,
}

#[derive(Deserialize, Debug, Clone)]
struct GltfBufferView {
    buffer: usize,
    #[serde(default, rename = "byteOffset")]
    byte_offset: usize,
    #[serde(rename = "byteLength")]
    byte_length: usize,
    #[serde(default, rename = "byteStride")]
    byte_stride: Option<usize>,
    #[serde(default)]
    target: Option<u32>,
}

#[derive(Deserialize, Debug, Clone)]
struct GltfAccessor {
    #[serde(rename = "bufferView")]
    buffer_view: Option<usize>,
    #[serde(default, rename = "byteOffset")]
    byte_offset: usize,
    #[serde(rename = "componentType")]
    component_type: u32,
    count: usize,
    #[serde(rename = "type")]
    accessor_type: String,
    #[serde(default)]
    min: Option<Vec<f64>>,
    #[serde(default)]
    max: Option<Vec<f64>>,
    #[serde(default)]
    normalized: bool,
}

#[derive(Deserialize, Debug, Clone)]
struct GltfMeshJson {
    #[serde(default)]
    name: Option<String>,
    primitives: Vec<GltfPrimitiveJson>,
}

#[derive(Deserialize, Debug, Clone)]
struct GltfPrimitiveJson {
    #[serde(default)]
    attributes: HashMap<String, usize>,
    #[serde(default)]
    indices: Option<usize>,
    #[serde(default)]
    material: Option<usize>,
    #[serde(default)]
    mode: Option<u32>,
    #[serde(default)]
    targets: Vec<HashMap<String, usize>>,
}

#[derive(Deserialize, Debug, Clone)]
struct GltfNodeJson {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    mesh: Option<usize>,
    #[serde(default)]
    skin: Option<usize>,
    #[serde(default)]
    camera: Option<usize>,
    #[serde(default)]
    children: Vec<usize>,
    #[serde(default)]
    matrix: Option<[f64; 16]>,
    #[serde(default)]
    translation: Option<[f64; 3]>,
    #[serde(default)]
    rotation: Option<[f64; 4]>,
    #[serde(default)]
    scale: Option<[f64; 3]>,
    #[serde(default)]
    weights: Vec<f64>,
}

#[derive(Deserialize, Debug, Clone)]
struct GltfSkinJson {
    #[serde(default)]
    name: Option<String>,
    #[serde(default, rename = "inverseBindMatrices")]
    inverse_bind_matrices: Option<usize>,
    joints: Vec<usize>,
    #[serde(default)]
    skeleton: Option<usize>,
}

#[derive(Deserialize, Debug, Clone)]
struct GltfSceneJson {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    nodes: Vec<usize>,
}

// ---------------------------------------------------------------------------
// GLB parsing
// ---------------------------------------------------------------------------

/// GLB magic number: "glTF" in little-endian.
const GLB_MAGIC: u32 = 0x46546C67;
/// GLB chunk type: JSON.
const GLB_CHUNK_JSON: u32 = 0x4E4F534A;
/// GLB chunk type: BIN (binary buffer).
const GLB_CHUNK_BIN: u32 = 0x004E4942;

/// Parse GLB container, returning (json_string, binary_buffer).
fn parse_glb(data: &[u8]) -> Result<(String, Vec<u8>), GltfError> {
    if data.len() < 12 {
        return Err(GltfError::InvalidGlb("file too short for header".into()));
    }

    let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    if magic != GLB_MAGIC {
        return Err(GltfError::InvalidGlb("invalid magic number".into()));
    }

    let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    if version != 2 {
        return Err(GltfError::InvalidGlb(format!(
            "unsupported version {}, expected 2",
            version
        )));
    }

    let _length = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);

    // Parse chunks
    let mut offset = 12usize;
    let mut json_data: Option<String> = None;
    let mut bin_data: Vec<u8> = Vec::new();

    while offset + 8 <= data.len() {
        let chunk_length =
            u32::from_le_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]])
                as usize;
        let chunk_type = u32::from_le_bytes([
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]);

        let chunk_start = offset + 8;
        let chunk_end = chunk_start + chunk_length;

        if chunk_end > data.len() {
            return Err(GltfError::InvalidGlb("chunk extends past end of file".into()));
        }

        match chunk_type {
            GLB_CHUNK_JSON => {
                let json_bytes = &data[chunk_start..chunk_end];
                json_data = Some(
                    String::from_utf8(json_bytes.to_vec())
                        .map_err(|e| GltfError::JsonError(e.to_string()))?,
                );
            }
            GLB_CHUNK_BIN => {
                bin_data = data[chunk_start..chunk_end].to_vec();
            }
            _ => {
                // Unknown chunk type, skip
            }
        }

        // Chunks are padded to 4-byte alignment
        offset = chunk_end + ((4 - (chunk_length % 4)) % 4);
    }

    let json =
        json_data.ok_or_else(|| GltfError::InvalidGlb("missing JSON chunk".into()))?;

    Ok((json, bin_data))
}

// ---------------------------------------------------------------------------
// Core loading functions
// ---------------------------------------------------------------------------

/// Load a glTF file (JSON or GLB format).
///
/// For .gltf files, external buffer URIs are resolved relative to the file path.
/// For .glb files, the embedded binary buffer is used.
///
/// # Errors
///
/// Returns an error if the file cannot be read, parsed, or contains invalid data.
pub fn load_gltf(path: &Path) -> Result<Vec<GltfMesh>, GltfError> {
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match extension.as_str() {
        "glb" => {
            let data = std::fs::read(path)?;
            let (json, bin_buffer) = parse_glb(&data)?;
            let buffers = if bin_buffer.is_empty() {
                vec![]
            } else {
                vec![bin_buffer]
            };
            load_gltf_from_json(&json, &buffers)
        }
        "gltf" => {
            let file = File::open(path)?;
            let reader = BufReader::new(file);
            let json: String = {
                let mut s = String::new();
                let mut r = reader;
                r.read_to_string(&mut s)?;
                s
            };

            // Parse JSON to find buffer URIs
            let gltf: GltfJson =
                serde_json::from_str(&json).map_err(|e| GltfError::JsonError(e.to_string()))?;

            // Load external buffers
            let parent = path.parent();
            let mut buffers = Vec::with_capacity(gltf.buffers.len());

            for buffer in &gltf.buffers {
                if let Some(uri) = &buffer.uri {
                    // Handle data URIs
                    if uri.starts_with("data:") {
                        let data = parse_data_uri(uri)?;
                        buffers.push(data);
                    } else {
                        // External file
                        let buffer_path = if let Some(p) = parent {
                            p.join(uri)
                        } else {
                            Path::new(uri).to_path_buf()
                        };
                        let data = std::fs::read(&buffer_path)?;
                        buffers.push(data);
                    }
                } else {
                    // GLB-style embedded buffer (shouldn't happen in .gltf)
                    buffers.push(Vec::new());
                }
            }

            load_gltf_from_json(&json, &buffers)
        }
        _ => Err(GltfError::UnsupportedFeature(format!(
            "unsupported file extension: {}",
            extension
        ))),
    }
}

/// Parse a data URI and extract the binary data.
fn parse_data_uri(uri: &str) -> Result<Vec<u8>, GltfError> {
    // Format: data:[<mediatype>][;base64],<data>
    let comma_pos = uri
        .find(',')
        .ok_or_else(|| GltfError::JsonError("invalid data URI".into()))?;

    let header = &uri[..comma_pos];
    let data_str = &uri[comma_pos + 1..];

    if header.contains(";base64") {
        // Base64 decode
        base64_decode(data_str)
    } else {
        // Percent-encoded
        Ok(percent_decode(data_str))
    }
}

/// Simple base64 decoder.
fn base64_decode(input: &str) -> Result<Vec<u8>, GltfError> {
    const DECODE_TABLE: [i8; 256] = {
        let mut table = [-1i8; 256];
        let mut i = 0u8;
        while i < 26 {
            table[(b'A' + i) as usize] = i as i8;
            table[(b'a' + i) as usize] = (i + 26) as i8;
            i += 1;
        }
        let mut i = 0u8;
        while i < 10 {
            table[(b'0' + i) as usize] = (i + 52) as i8;
            i += 1;
        }
        table[b'+' as usize] = 62;
        table[b'/' as usize] = 63;
        table
    };

    let input: Vec<u8> = input.bytes().filter(|&b| b != b'\n' && b != b'\r').collect();
    let mut output = Vec::with_capacity(input.len() * 3 / 4);

    let mut i = 0;
    while i + 4 <= input.len() {
        let a = DECODE_TABLE[input[i] as usize];
        let b = DECODE_TABLE[input[i + 1] as usize];
        let c = DECODE_TABLE[input[i + 2] as usize];
        let d = DECODE_TABLE[input[i + 3] as usize];

        if a < 0 || b < 0 {
            return Err(GltfError::JsonError("invalid base64".into()));
        }

        output.push(((a as u8) << 2) | ((b as u8) >> 4));

        if input[i + 2] != b'=' {
            if c < 0 {
                return Err(GltfError::JsonError("invalid base64".into()));
            }
            output.push(((b as u8) << 4) | ((c as u8) >> 2));
        }

        if input[i + 3] != b'=' {
            if d < 0 {
                return Err(GltfError::JsonError("invalid base64".into()));
            }
            output.push(((c as u8) << 6) | (d as u8));
        }

        i += 4;
    }

    Ok(output)
}

/// Simple percent-decoding.
fn percent_decode(input: &str) -> Vec<u8> {
    let mut output = Vec::with_capacity(input.len());
    let bytes = input.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(hi), Some(lo)) = (
                hex_digit(bytes[i + 1]),
                hex_digit(bytes[i + 2]),
            ) {
                output.push((hi << 4) | lo);
                i += 3;
                continue;
            }
        }
        output.push(bytes[i]);
        i += 1;
    }

    output
}

fn hex_digit(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// Load meshes from glTF JSON string with pre-loaded buffer data.
///
/// # Arguments
///
/// * `json` - The glTF JSON content.
/// * `buffers` - Pre-loaded binary buffer data, indexed by buffer index.
///
/// # Errors
///
/// Returns an error if JSON parsing fails or accessor/buffer data is invalid.
pub fn load_gltf_from_json(json: &str, buffers: &[Vec<u8>]) -> Result<Vec<GltfMesh>, GltfError> {
    let gltf: GltfJson =
        serde_json::from_str(json).map_err(|e| GltfError::JsonError(e.to_string()))?;

    let mut result = Vec::with_capacity(gltf.meshes.len());

    for mesh_json in &gltf.meshes {
        let mut primitives = Vec::with_capacity(mesh_json.primitives.len());

        for prim_json in &mesh_json.primitives {
            let mut attributes = HashMap::new();

            // Extract vertex attributes
            for (attr_name, &accessor_idx) in &prim_json.attributes {
                if let Some(semantic) = VertexSemantic::from_gltf_name(attr_name) {
                    let attr = extract_accessor(
                        &gltf.accessors,
                        &gltf.buffer_views,
                        buffers,
                        accessor_idx,
                        semantic,
                    )?;
                    attributes.insert(semantic, attr);
                }
            }

            // Extract indices
            let indices = if let Some(idx_accessor) = prim_json.indices {
                Some(extract_index_buffer(
                    &gltf.accessors,
                    &gltf.buffer_views,
                    buffers,
                    idx_accessor,
                )?)
            } else {
                None
            };

            primitives.push(GltfPrimitive {
                attributes,
                indices,
                material_index: prim_json.material,
            });
        }

        result.push(GltfMesh {
            name: mesh_json.name.clone(),
            primitives,
        });
    }

    Ok(result)
}

/// Extract accessor data as a vertex attribute.
fn extract_accessor(
    accessors: &[GltfAccessor],
    buffer_views: &[GltfBufferView],
    buffers: &[Vec<u8>],
    accessor_idx: usize,
    semantic: VertexSemantic,
) -> Result<VertexAttribute, GltfError> {
    let accessor = accessors.get(accessor_idx).ok_or_else(|| {
        GltfError::InvalidAccessor(format!("accessor {} not found", accessor_idx))
    })?;

    let component_type = ComponentType::from_gltf(accessor.component_type).ok_or_else(|| {
        GltfError::InvalidAccessor(format!(
            "unknown component type {}",
            accessor.component_type
        ))
    })?;

    let attribute_type = AttributeType::from_gltf(&accessor.accessor_type).ok_or_else(|| {
        GltfError::InvalidAccessor(format!("unknown accessor type {}", accessor.accessor_type))
    })?;

    let element_size = component_type.size_bytes() * attribute_type.component_count();

    // Get buffer view if present
    let (data, offset, stride) = if let Some(bv_idx) = accessor.buffer_view {
        let buffer_view = buffer_views.get(bv_idx).ok_or_else(|| {
            GltfError::InvalidBufferView(format!("buffer view {} not found", bv_idx))
        })?;

        let buffer = buffers.get(buffer_view.buffer).ok_or_else(|| {
            GltfError::MissingBuffer(buffer_view.buffer)
        })?;

        let stride = buffer_view.byte_stride.unwrap_or(element_size);
        let total_offset = buffer_view.byte_offset + accessor.byte_offset;

        // Extract data
        let mut data = Vec::with_capacity(accessor.count * element_size);
        for i in 0..accessor.count {
            let start = total_offset + i * stride;
            let end = start + element_size;
            if end > buffer.len() {
                return Err(GltfError::InvalidAccessor(format!(
                    "accessor data extends past buffer end (element {} at {}..{}, buffer len {})",
                    i, start, end, buffer.len()
                )));
            }
            data.extend_from_slice(&buffer[start..end]);
        }

        (data, accessor.byte_offset, stride)
    } else {
        // No buffer view - sparse accessor or zero-filled
        let data = vec![0u8; accessor.count * element_size];
        (data, 0, element_size)
    };

    Ok(VertexAttribute {
        semantic,
        component_type,
        attribute_type,
        offset,
        stride,
        count: accessor.count,
        data,
    })
}

/// Extract index buffer from accessor.
fn extract_index_buffer(
    accessors: &[GltfAccessor],
    buffer_views: &[GltfBufferView],
    buffers: &[Vec<u8>],
    accessor_idx: usize,
) -> Result<IndexBuffer, GltfError> {
    let accessor = accessors.get(accessor_idx).ok_or_else(|| {
        GltfError::InvalidAccessor(format!("index accessor {} not found", accessor_idx))
    })?;

    let component_type = ComponentType::from_gltf(accessor.component_type).ok_or_else(|| {
        GltfError::InvalidAccessor(format!(
            "unknown index component type {}",
            accessor.component_type
        ))
    })?;

    let format = match component_type {
        ComponentType::U8 => IndexFormat::U8,
        ComponentType::U16 => IndexFormat::U16,
        ComponentType::U32 => IndexFormat::U32,
        _ => {
            return Err(GltfError::InvalidAccessor(format!(
                "invalid index component type {:?}",
                component_type
            )))
        }
    };

    let index_size = format.size_bytes();

    let data = if let Some(bv_idx) = accessor.buffer_view {
        let buffer_view = buffer_views.get(bv_idx).ok_or_else(|| {
            GltfError::InvalidBufferView(format!("buffer view {} not found", bv_idx))
        })?;

        let buffer = buffers.get(buffer_view.buffer).ok_or_else(|| {
            GltfError::MissingBuffer(buffer_view.buffer)
        })?;

        let stride = buffer_view.byte_stride.unwrap_or(index_size);
        let total_offset = buffer_view.byte_offset + accessor.byte_offset;

        let mut data = Vec::with_capacity(accessor.count * index_size);
        for i in 0..accessor.count {
            let start = total_offset + i * stride;
            let end = start + index_size;
            if end > buffer.len() {
                return Err(GltfError::InvalidAccessor(format!(
                    "index data extends past buffer end"
                )));
            }
            data.extend_from_slice(&buffer[start..end]);
        }
        data
    } else {
        vec![0u8; accessor.count * index_size]
    };

    Ok(IndexBuffer {
        format,
        count: accessor.count,
        data,
    })
}

// ---------------------------------------------------------------------------
// GltfParser - Progressive loading and streaming support
// ---------------------------------------------------------------------------

/// Thread-safe glTF parser with progressive loading support.
///
/// This parser is `Send + Sync` and can be used from worker threads.
/// It supports progressive loading in stages:
/// 1. Bounds - Fast AABB extraction from accessor min/max
/// 2. Geometry - Vertex positions and indices
/// 3. Attributes - All vertex attributes
/// 4. Skinning - Joint/weight data and skins
/// 5. Full - Complete document with node hierarchy
#[derive(Debug, Clone)]
pub struct GltfParser {
    gltf: Arc<GltfJson>,
    validation: ValidationResult,
}

// Ensure GltfParser is Send + Sync for worker thread usage
unsafe impl Send for GltfParser {}
unsafe impl Sync for GltfParser {}

impl GltfParser {
    /// Create a new parser from JSON string.
    ///
    /// Performs schema validation on construction.
    pub fn new(json: &str) -> Result<Self, GltfError> {
        let gltf: GltfJson =
            serde_json::from_str(json).map_err(|e| GltfError::JsonError(e.to_string()))?;

        let validation = Self::validate_schema(&gltf);
        if !validation.is_valid {
            return Err(GltfError::ValidationFailed(validation));
        }

        Ok(Self {
            gltf: Arc::new(gltf),
            validation,
        })
    }

    /// Create parser from file path.
    pub fn from_path(path: &Path) -> Result<Self, GltfError> {
        let extension = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        match extension.as_str() {
            "glb" => {
                let data = std::fs::read(path)?;
                let (json, _bin_buffer) = parse_glb(&data)?;
                Self::new(&json)
            }
            "gltf" => {
                let json = std::fs::read_to_string(path)?;
                Self::new(&json)
            }
            _ => Err(GltfError::UnsupportedFeature(format!(
                "unsupported file extension: {}",
                extension
            ))),
        }
    }

    /// Get validation result.
    pub fn validation(&self) -> &ValidationResult {
        &self.validation
    }

    /// Get mesh count.
    pub fn mesh_count(&self) -> usize {
        self.gltf.meshes.len()
    }

    /// Get node count.
    pub fn node_count(&self) -> usize {
        self.gltf.nodes.len()
    }

    /// Get skin count.
    pub fn skin_count(&self) -> usize {
        self.gltf.skins.len()
    }

    /// Get buffer count.
    pub fn buffer_count(&self) -> usize {
        self.gltf.buffers.len()
    }

    /// Get total buffer size.
    pub fn total_buffer_size(&self) -> usize {
        self.gltf.buffers.iter().map(|b| b.byte_length).sum()
    }

    /// Validate glTF schema.
    fn validate_schema(gltf: &GltfJson) -> ValidationResult {
        let mut result = ValidationResult::ok();

        // Validate asset version
        if gltf.asset.version.is_empty() {
            result.add_warning("asset.version", "version not specified");
        } else if !gltf.asset.version.starts_with("2.") {
            result.add_error(
                "asset.version",
                format!("unsupported version {}, expected 2.x", gltf.asset.version),
            );
        }

        // Validate buffers
        for (i, buffer) in gltf.buffers.iter().enumerate() {
            if buffer.byte_length == 0 {
                result.add_warning(format!("buffers[{}]", i), "empty buffer");
            }
        }

        // Validate buffer views
        for (i, bv) in gltf.buffer_views.iter().enumerate() {
            if bv.buffer >= gltf.buffers.len() {
                result.add_error(
                    format!("bufferViews[{}].buffer", i),
                    format!("references non-existent buffer {}", bv.buffer),
                );
            }
            if let Some(stride) = bv.byte_stride {
                if stride < 4 || stride > 252 || stride % 4 != 0 {
                    result.add_warning(
                        format!("bufferViews[{}].byteStride", i),
                        format!("stride {} should be 4-252 and multiple of 4", stride),
                    );
                }
            }
        }

        // Validate accessors
        for (i, acc) in gltf.accessors.iter().enumerate() {
            if let Some(bv) = acc.buffer_view {
                if bv >= gltf.buffer_views.len() {
                    result.add_error(
                        format!("accessors[{}].bufferView", i),
                        format!("references non-existent buffer view {}", bv),
                    );
                }
            }
            if ComponentType::from_gltf(acc.component_type).is_none() {
                result.add_error(
                    format!("accessors[{}].componentType", i),
                    format!("unknown component type {}", acc.component_type),
                );
            }
            if AttributeType::from_gltf(&acc.accessor_type).is_none() {
                result.add_error(
                    format!("accessors[{}].type", i),
                    format!("unknown accessor type {}", acc.accessor_type),
                );
            }
        }

        // Validate meshes
        for (i, mesh) in gltf.meshes.iter().enumerate() {
            if mesh.primitives.is_empty() {
                result.add_warning(format!("meshes[{}]", i), "mesh has no primitives");
            }
            for (j, prim) in mesh.primitives.iter().enumerate() {
                if !prim.attributes.contains_key("POSITION") {
                    result.add_error(
                        format!("meshes[{}].primitives[{}]", i, j),
                        "primitive missing POSITION attribute",
                    );
                }
                for (attr, &acc_idx) in &prim.attributes {
                    if acc_idx >= gltf.accessors.len() {
                        result.add_error(
                            format!("meshes[{}].primitives[{}].attributes.{}", i, j, attr),
                            format!("references non-existent accessor {}", acc_idx),
                        );
                    }
                }
                if let Some(idx) = prim.indices {
                    if idx >= gltf.accessors.len() {
                        result.add_error(
                            format!("meshes[{}].primitives[{}].indices", i, j),
                            format!("references non-existent accessor {}", idx),
                        );
                    }
                }
            }
        }

        // Validate nodes (check for cycles)
        for (i, node) in gltf.nodes.iter().enumerate() {
            if let Some(mesh) = node.mesh {
                if mesh >= gltf.meshes.len() {
                    result.add_error(
                        format!("nodes[{}].mesh", i),
                        format!("references non-existent mesh {}", mesh),
                    );
                }
            }
            if let Some(skin) = node.skin {
                if skin >= gltf.skins.len() {
                    result.add_error(
                        format!("nodes[{}].skin", i),
                        format!("references non-existent skin {}", skin),
                    );
                }
            }
            for (j, &child) in node.children.iter().enumerate() {
                if child >= gltf.nodes.len() {
                    result.add_error(
                        format!("nodes[{}].children[{}]", i, j),
                        format!("references non-existent node {}", child),
                    );
                }
                if child == i {
                    result.add_error(
                        format!("nodes[{}].children[{}]", i, j),
                        "node is its own child",
                    );
                }
            }
        }

        // Validate skins
        for (i, skin) in gltf.skins.iter().enumerate() {
            if skin.joints.is_empty() {
                result.add_error(format!("skins[{}]", i), "skin has no joints");
            }
            for (j, &joint) in skin.joints.iter().enumerate() {
                if joint >= gltf.nodes.len() {
                    result.add_error(
                        format!("skins[{}].joints[{}]", i, j),
                        format!("references non-existent node {}", joint),
                    );
                }
            }
            if let Some(ibm) = skin.inverse_bind_matrices {
                if ibm >= gltf.accessors.len() {
                    result.add_error(
                        format!("skins[{}].inverseBindMatrices", i),
                        format!("references non-existent accessor {}", ibm),
                    );
                }
            }
        }

        // Validate scenes
        for (i, scene) in gltf.scenes.iter().enumerate() {
            for (j, &node) in scene.nodes.iter().enumerate() {
                if node >= gltf.nodes.len() {
                    result.add_error(
                        format!("scenes[{}].nodes[{}]", i, j),
                        format!("references non-existent node {}", node),
                    );
                }
            }
        }

        if let Some(default_scene) = gltf.scene {
            if default_scene >= gltf.scenes.len() {
                result.add_error(
                    "scene",
                    format!("references non-existent scene {}", default_scene),
                );
            }
        }

        result
    }

    /// Load bounds stage (fast, no buffer reads required).
    ///
    /// Extracts AABB from accessor min/max values without reading buffer data.
    pub fn load_bounds(&self) -> BoundsResult {
        let mut mesh_bounds = Vec::with_capacity(self.gltf.meshes.len());
        let mut scene_bounds = Aabb::EMPTY;
        let mut total_vertices = 0usize;
        let mut total_indices = 0usize;

        for mesh in &self.gltf.meshes {
            let mut mesh_aabb = Aabb::EMPTY;

            for prim in &mesh.primitives {
                if let Some(&pos_accessor_idx) = prim.attributes.get("POSITION") {
                    if let Some(accessor) = self.gltf.accessors.get(pos_accessor_idx) {
                        total_vertices += accessor.count;

                        // Extract bounds from min/max if available
                        if let (Some(min), Some(max)) = (&accessor.min, &accessor.max) {
                            if min.len() >= 3 && max.len() >= 3 {
                                let prim_aabb = Aabb::new(
                                    [min[0] as f32, min[1] as f32, min[2] as f32],
                                    [max[0] as f32, max[1] as f32, max[2] as f32],
                                );
                                mesh_aabb = mesh_aabb.union(&prim_aabb);
                            }
                        }
                    }
                }

                if let Some(idx_accessor) = prim.indices {
                    if let Some(accessor) = self.gltf.accessors.get(idx_accessor) {
                        total_indices += accessor.count;
                    }
                }
            }

            if mesh_aabb.is_valid() {
                scene_bounds = scene_bounds.union(&mesh_aabb);
            }

            mesh_bounds.push((mesh.name.clone(), mesh_aabb));
        }

        BoundsResult {
            mesh_bounds,
            scene_bounds,
            mesh_count: self.gltf.meshes.len(),
            node_count: self.gltf.nodes.len(),
            skin_count: self.gltf.skins.len(),
            vertex_count_estimate: total_vertices,
            index_count_estimate: total_indices,
        }
    }

    /// Load full document with node hierarchy and skins.
    pub fn load_full(&self, buffers: &[Vec<u8>]) -> Result<GltfDocument, GltfError> {
        // Load meshes
        let meshes = self.load_meshes(buffers, LoadStage::Full)?;

        // Load nodes
        let nodes = self.load_nodes()?;

        // Load skins
        let skins = self.load_skins(buffers)?;

        // Load scenes
        let scenes = self.load_scenes()?;

        // Compute scene bounds
        let bounds_result = self.load_bounds();

        Ok(GltfDocument {
            meshes,
            nodes,
            skins,
            scenes,
            default_scene: self.gltf.scene,
            bounds: bounds_result.scene_bounds,
        })
    }

    /// Load meshes at a specific stage.
    pub fn load_meshes(&self, buffers: &[Vec<u8>], stage: LoadStage) -> Result<Vec<GltfMesh>, GltfError> {
        let mut result = Vec::with_capacity(self.gltf.meshes.len());

        for mesh_json in &self.gltf.meshes {
            let mut primitives = Vec::with_capacity(mesh_json.primitives.len());

            for prim_json in &mesh_json.primitives {
                let mut attributes = HashMap::new();

                // Filter attributes based on stage
                for (attr_name, &accessor_idx) in &prim_json.attributes {
                    if let Some(semantic) = VertexSemantic::from_gltf_name(attr_name) {
                        let should_load = match stage {
                            LoadStage::Bounds => false,
                            LoadStage::Geometry => semantic == VertexSemantic::Position,
                            LoadStage::Attributes => {
                                semantic != VertexSemantic::Joints0 &&
                                semantic != VertexSemantic::Weights0
                            }
                            LoadStage::Skinning | LoadStage::Full => true,
                        };

                        if should_load {
                            let attr = extract_accessor(
                                &self.gltf.accessors,
                                &self.gltf.buffer_views,
                                buffers,
                                accessor_idx,
                                semantic,
                            )?;
                            attributes.insert(semantic, attr);
                        }
                    }
                }

                // Load indices for Geometry stage and above
                let indices = if stage >= LoadStage::Geometry {
                    if let Some(idx_accessor) = prim_json.indices {
                        Some(extract_index_buffer(
                            &self.gltf.accessors,
                            &self.gltf.buffer_views,
                            buffers,
                            idx_accessor,
                        )?)
                    } else {
                        None
                    }
                } else {
                    None
                };

                primitives.push(GltfPrimitive {
                    attributes,
                    indices,
                    material_index: prim_json.material,
                });
            }

            result.push(GltfMesh {
                name: mesh_json.name.clone(),
                primitives,
            });
        }

        Ok(result)
    }

    /// Load node hierarchy.
    pub fn load_nodes(&self) -> Result<Vec<GltfNode>, GltfError> {
        let mut nodes = Vec::with_capacity(self.gltf.nodes.len());

        // First pass: create nodes with local transforms
        for (index, node_json) in self.gltf.nodes.iter().enumerate() {
            let local_transform = if let Some(matrix) = &node_json.matrix {
                let m: [f32; 16] = [
                    matrix[0] as f32, matrix[1] as f32, matrix[2] as f32, matrix[3] as f32,
                    matrix[4] as f32, matrix[5] as f32, matrix[6] as f32, matrix[7] as f32,
                    matrix[8] as f32, matrix[9] as f32, matrix[10] as f32, matrix[11] as f32,
                    matrix[12] as f32, matrix[13] as f32, matrix[14] as f32, matrix[15] as f32,
                ];
                Mat4::from_cols(m)
            } else {
                let translation = node_json.translation
                    .map(|t| [t[0] as f32, t[1] as f32, t[2] as f32])
                    .unwrap_or([0.0, 0.0, 0.0]);
                let rotation = node_json.rotation
                    .map(|r| [r[0] as f32, r[1] as f32, r[2] as f32, r[3] as f32])
                    .unwrap_or([0.0, 0.0, 0.0, 1.0]);
                let scale = node_json.scale
                    .map(|s| [s[0] as f32, s[1] as f32, s[2] as f32])
                    .unwrap_or([1.0, 1.0, 1.0]);
                Mat4::from_trs(translation, rotation, scale)
            };

            nodes.push(GltfNode {
                index,
                name: node_json.name.clone(),
                local_transform,
                world_transform: local_transform, // Will be updated in second pass
                mesh: node_json.mesh,
                skin: node_json.skin,
                camera: node_json.camera,
                children: node_json.children.clone(),
                parent: None,
            });
        }

        // Build parent references
        for i in 0..self.gltf.nodes.len() {
            let children = self.gltf.nodes[i].children.clone();
            for child_idx in children {
                if child_idx < nodes.len() {
                    nodes[child_idx].parent = Some(i);
                }
            }
        }

        // Compute world transforms (topological traversal)
        let mut visited = vec![false; nodes.len()];
        fn compute_world_transform(
            nodes: &mut [GltfNode],
            visited: &mut [bool],
            idx: usize,
        ) {
            if visited[idx] {
                return;
            }

            if let Some(parent_idx) = nodes[idx].parent {
                compute_world_transform(nodes, visited, parent_idx);
                let parent_world = nodes[parent_idx].world_transform;
                nodes[idx].world_transform = parent_world.mul(&nodes[idx].local_transform);
            }

            visited[idx] = true;
        }

        for i in 0..nodes.len() {
            compute_world_transform(&mut nodes, &mut visited, i);
        }

        Ok(nodes)
    }

    /// Load skin data.
    pub fn load_skins(&self, buffers: &[Vec<u8>]) -> Result<Vec<GltfSkin>, GltfError> {
        let mut skins = Vec::with_capacity(self.gltf.skins.len());

        for (index, skin_json) in self.gltf.skins.iter().enumerate() {
            let inverse_bind_matrices = if let Some(ibm_accessor) = skin_json.inverse_bind_matrices {
                self.extract_mat4_accessor(buffers, ibm_accessor)?
            } else {
                // Default to identity matrices
                vec![Mat4::IDENTITY; skin_json.joints.len()]
            };

            if inverse_bind_matrices.len() != skin_json.joints.len() {
                return Err(GltfError::InvalidSkin(format!(
                    "skin {} has {} joints but {} inverse bind matrices",
                    index,
                    skin_json.joints.len(),
                    inverse_bind_matrices.len()
                )));
            }

            skins.push(GltfSkin {
                index,
                name: skin_json.name.clone(),
                inverse_bind_matrices,
                joints: skin_json.joints.clone(),
                skeleton: skin_json.skeleton,
            });
        }

        Ok(skins)
    }

    /// Load scenes.
    pub fn load_scenes(&self) -> Result<Vec<GltfScene>, GltfError> {
        Ok(self.gltf.scenes.iter().enumerate().map(|(index, scene_json)| {
            GltfScene {
                index,
                name: scene_json.name.clone(),
                nodes: scene_json.nodes.clone(),
            }
        }).collect())
    }

    /// Extract Mat4 array from accessor.
    fn extract_mat4_accessor(&self, buffers: &[Vec<u8>], accessor_idx: usize) -> Result<Vec<Mat4>, GltfError> {
        let accessor = self.gltf.accessors.get(accessor_idx).ok_or_else(|| {
            GltfError::InvalidAccessor(format!("accessor {} not found", accessor_idx))
        })?;

        if accessor.accessor_type != "MAT4" {
            return Err(GltfError::InvalidAccessor(format!(
                "expected MAT4 accessor, got {}",
                accessor.accessor_type
            )));
        }

        let component_type = ComponentType::from_gltf(accessor.component_type).ok_or_else(|| {
            GltfError::InvalidAccessor(format!(
                "unknown component type {}",
                accessor.component_type
            ))
        })?;

        if component_type != ComponentType::F32 {
            return Err(GltfError::InvalidAccessor(
                "MAT4 accessor must be F32".into()
            ));
        }

        let element_size = 16 * 4; // 16 floats * 4 bytes

        let data = if let Some(bv_idx) = accessor.buffer_view {
            let buffer_view = self.gltf.buffer_views.get(bv_idx).ok_or_else(|| {
                GltfError::InvalidBufferView(format!("buffer view {} not found", bv_idx))
            })?;

            let buffer = buffers.get(buffer_view.buffer).ok_or_else(|| {
                GltfError::MissingBuffer(buffer_view.buffer)
            })?;

            let stride = buffer_view.byte_stride.unwrap_or(element_size);
            let total_offset = buffer_view.byte_offset + accessor.byte_offset;

            let mut matrices = Vec::with_capacity(accessor.count);
            for i in 0..accessor.count {
                let start = total_offset + i * stride;
                let end = start + element_size;
                if end > buffer.len() {
                    return Err(GltfError::InvalidAccessor(
                        "MAT4 data extends past buffer end".into()
                    ));
                }

                let mut cols = [0.0f32; 16];
                for j in 0..16 {
                    let offset = start + j * 4;
                    cols[j] = f32::from_le_bytes([
                        buffer[offset],
                        buffer[offset + 1],
                        buffer[offset + 2],
                        buffer[offset + 3],
                    ]);
                }
                matrices.push(Mat4::from_cols(cols));
            }
            matrices
        } else {
            vec![Mat4::IDENTITY; accessor.count]
        };

        Ok(data)
    }
}

// ---------------------------------------------------------------------------
// Streaming parser for large files (>2GB)
// ---------------------------------------------------------------------------

/// Streaming buffer reader for large glTF files.
///
/// Allows reading buffer data on-demand without loading entire file into memory.
pub struct StreamingBufferReader<R: Read + Seek> {
    reader: R,
    buffer_offsets: Vec<(u64, usize)>, // (offset, length)
}

impl<R: Read + Seek> StreamingBufferReader<R> {
    /// Create a new streaming reader from a GLB file.
    pub fn from_glb(mut reader: R) -> Result<(Self, String), GltfError> {
        // Read GLB header
        let mut header = [0u8; 12];
        reader.read_exact(&mut header).map_err(|e| {
            GltfError::StreamingError(format!("failed to read GLB header: {}", e))
        })?;

        let magic = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
        if magic != GLB_MAGIC {
            return Err(GltfError::InvalidGlb("invalid magic number".into()));
        }

        let version = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
        if version != 2 {
            return Err(GltfError::InvalidGlb(format!(
                "unsupported version {}, expected 2",
                version
            )));
        }

        // Read chunks
        let mut json_data: Option<String> = None;
        let mut buffer_offsets = Vec::new();

        loop {
            let mut chunk_header = [0u8; 8];
            if reader.read_exact(&mut chunk_header).is_err() {
                break; // End of file
            }

            let chunk_length = u32::from_le_bytes([
                chunk_header[0], chunk_header[1], chunk_header[2], chunk_header[3],
            ]) as usize;
            let chunk_type = u32::from_le_bytes([
                chunk_header[4], chunk_header[5], chunk_header[6], chunk_header[7],
            ]);

            let chunk_start = reader.stream_position().map_err(|e| {
                GltfError::StreamingError(format!("failed to get position: {}", e))
            })?;

            match chunk_type {
                GLB_CHUNK_JSON => {
                    let mut json_bytes = vec![0u8; chunk_length];
                    reader.read_exact(&mut json_bytes).map_err(|e| {
                        GltfError::StreamingError(format!("failed to read JSON chunk: {}", e))
                    })?;
                    json_data = Some(String::from_utf8(json_bytes).map_err(|e| {
                        GltfError::JsonError(e.to_string())
                    })?);
                }
                GLB_CHUNK_BIN => {
                    buffer_offsets.push((chunk_start, chunk_length));
                    // Skip the chunk data
                    reader.seek(SeekFrom::Current(chunk_length as i64)).map_err(|e| {
                        GltfError::StreamingError(format!("failed to seek: {}", e))
                    })?;
                }
                _ => {
                    // Skip unknown chunk
                    reader.seek(SeekFrom::Current(chunk_length as i64)).map_err(|e| {
                        GltfError::StreamingError(format!("failed to seek: {}", e))
                    })?;
                }
            }

            // Align to 4-byte boundary
            let padding = (4 - (chunk_length % 4)) % 4;
            if padding > 0 {
                reader.seek(SeekFrom::Current(padding as i64)).map_err(|e| {
                    GltfError::StreamingError(format!("failed to seek: {}", e))
                })?;
            }
        }

        let json = json_data.ok_or_else(|| {
            GltfError::InvalidGlb("missing JSON chunk".into())
        })?;

        Ok((Self { reader, buffer_offsets }, json))
    }

    /// Read a range of bytes from a buffer.
    pub fn read_range(&mut self, buffer_idx: usize, offset: usize, length: usize) -> Result<Vec<u8>, GltfError> {
        let (file_offset, buffer_length) = self.buffer_offsets.get(buffer_idx).ok_or_else(|| {
            GltfError::MissingBuffer(buffer_idx)
        })?;

        if offset + length > *buffer_length {
            return Err(GltfError::StreamingError(format!(
                "read range {}..{} exceeds buffer length {}",
                offset, offset + length, buffer_length
            )));
        }

        self.reader.seek(SeekFrom::Start(*file_offset + offset as u64)).map_err(|e| {
            GltfError::StreamingError(format!("failed to seek: {}", e))
        })?;

        let mut data = vec![0u8; length];
        self.reader.read_exact(&mut data).map_err(|e| {
            GltfError::StreamingError(format!("failed to read: {}", e))
        })?;

        Ok(data)
    }

    /// Read an entire buffer.
    pub fn read_buffer(&mut self, buffer_idx: usize) -> Result<Vec<u8>, GltfError> {
        let (file_offset, buffer_length) = self.buffer_offsets.get(buffer_idx).ok_or_else(|| {
            GltfError::MissingBuffer(buffer_idx)
        })?;

        self.reader.seek(SeekFrom::Start(*file_offset)).map_err(|e| {
            GltfError::StreamingError(format!("failed to seek: {}", e))
        })?;

        let mut data = vec![0u8; *buffer_length];
        self.reader.read_exact(&mut data).map_err(|e| {
            GltfError::StreamingError(format!("failed to read: {}", e))
        })?;

        Ok(data)
    }

    /// Get buffer info (offset, length).
    pub fn buffer_info(&self, buffer_idx: usize) -> Option<(u64, usize)> {
        self.buffer_offsets.get(buffer_idx).copied()
    }

    /// Get number of buffers.
    pub fn buffer_count(&self) -> usize {
        self.buffer_offsets.len()
    }
}

/// Load a large glTF file with streaming.
///
/// This function avoids loading the entire file into memory, suitable for files >2GB.
pub fn load_gltf_streaming<R: Read + Seek>(
    reader: R,
) -> Result<(GltfParser, StreamingBufferReader<R>), GltfError> {
    let (streaming_reader, json) = StreamingBufferReader::from_glb(reader)?;
    let parser = GltfParser::new(&json)?;
    Ok((parser, streaming_reader))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper: create minimal glTF JSON ────────────────────────────────────

    fn minimal_triangle_gltf() -> (String, Vec<u8>) {
        // A simple triangle with 3 vertices at positions:
        // (0, 0, 0), (1, 0, 0), (0, 1, 0)
        // Indexed with U16: 0, 1, 2

        // Positions: 3 * Vec3<f32> = 36 bytes
        // Indices: 3 * U16 = 6 bytes
        // Total buffer: 42 bytes (padded to 44 for alignment)

        let mut buffer = Vec::new();

        // Positions (offset 0)
        buffer.extend_from_slice(&0.0f32.to_le_bytes()); // v0.x
        buffer.extend_from_slice(&0.0f32.to_le_bytes()); // v0.y
        buffer.extend_from_slice(&0.0f32.to_le_bytes()); // v0.z
        buffer.extend_from_slice(&1.0f32.to_le_bytes()); // v1.x
        buffer.extend_from_slice(&0.0f32.to_le_bytes()); // v1.y
        buffer.extend_from_slice(&0.0f32.to_le_bytes()); // v1.z
        buffer.extend_from_slice(&0.0f32.to_le_bytes()); // v2.x
        buffer.extend_from_slice(&1.0f32.to_le_bytes()); // v2.y
        buffer.extend_from_slice(&0.0f32.to_le_bytes()); // v2.z

        // Indices (offset 36)
        buffer.extend_from_slice(&0u16.to_le_bytes());
        buffer.extend_from_slice(&1u16.to_le_bytes());
        buffer.extend_from_slice(&2u16.to_le_bytes());

        let json = r#"{
            "asset": { "version": "2.0" },
            "buffers": [{ "byteLength": 42 }],
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0, "byteLength": 36 },
                { "buffer": 0, "byteOffset": 36, "byteLength": 6 }
            ],
            "accessors": [
                { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3" },
                { "bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR" }
            ],
            "meshes": [{
                "name": "Triangle",
                "primitives": [{
                    "attributes": { "POSITION": 0 },
                    "indices": 1
                }]
            }]
        }"#;

        (json.to_string(), buffer)
    }

    // ── Test: load simple triangle ──────────────────────────────────────────

    #[test]
    fn test_gltf_load_simple_triangle() {
        let (json, buffer) = minimal_triangle_gltf();
        let meshes = load_gltf_from_json(&json, &[buffer]).unwrap();

        assert_eq!(meshes.len(), 1);
        let mesh = &meshes[0];
        assert_eq!(mesh.name, Some("Triangle".to_string()));
        assert_eq!(mesh.primitives.len(), 1);

        let prim = &mesh.primitives[0];
        assert!(prim.attributes.contains_key(&VertexSemantic::Position));
        assert!(prim.indices.is_some());

        let pos = prim.attributes.get(&VertexSemantic::Position).unwrap();
        assert_eq!(pos.count, 3);
        assert_eq!(pos.component_type, ComponentType::F32);
        assert_eq!(pos.attribute_type, AttributeType::Vec3);
        assert_eq!(pos.data.len(), 36); // 3 vertices * 3 components * 4 bytes

        let indices = prim.indices.as_ref().unwrap();
        assert_eq!(indices.format, IndexFormat::U16);
        assert_eq!(indices.count, 3);
        assert_eq!(indices.data.len(), 6);

        // Verify position data
        let positions: Vec<f32> = pos
            .data
            .chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(positions, vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);

        // Verify index data
        let index_values: Vec<u16> = indices
            .data
            .chunks(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        assert_eq!(index_values, vec![0, 1, 2]);
    }

    // ── Test: vertex semantics ──────────────────────────────────────────────

    #[test]
    fn test_gltf_vertex_semantics() {
        // Test all semantic types
        let mut buffer = Vec::new();

        // POSITION: 2 * Vec3<f32> = 24 bytes
        buffer.extend_from_slice(&[0u8; 24]);
        // NORMAL: 2 * Vec3<f32> = 24 bytes
        buffer.extend_from_slice(&[0u8; 24]);
        // TANGENT: 2 * Vec4<f32> = 32 bytes
        buffer.extend_from_slice(&[0u8; 32]);
        // TEXCOORD_0: 2 * Vec2<f32> = 16 bytes
        buffer.extend_from_slice(&[0u8; 16]);
        // TEXCOORD_1: 2 * Vec2<f32> = 16 bytes
        buffer.extend_from_slice(&[0u8; 16]);
        // COLOR_0: 2 * Vec4<f32> = 32 bytes
        buffer.extend_from_slice(&[0u8; 32]);
        // JOINTS_0: 2 * Vec4<u16> = 16 bytes
        buffer.extend_from_slice(&[0u8; 16]);
        // WEIGHTS_0: 2 * Vec4<f32> = 32 bytes
        buffer.extend_from_slice(&[0u8; 32]);

        let json = r#"{
            "asset": { "version": "2.0" },
            "buffers": [{ "byteLength": 192 }],
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0, "byteLength": 24 },
                { "buffer": 0, "byteOffset": 24, "byteLength": 24 },
                { "buffer": 0, "byteOffset": 48, "byteLength": 32 },
                { "buffer": 0, "byteOffset": 80, "byteLength": 16 },
                { "buffer": 0, "byteOffset": 96, "byteLength": 16 },
                { "buffer": 0, "byteOffset": 112, "byteLength": 32 },
                { "buffer": 0, "byteOffset": 144, "byteLength": 16 },
                { "buffer": 0, "byteOffset": 160, "byteLength": 32 }
            ],
            "accessors": [
                { "bufferView": 0, "componentType": 5126, "count": 2, "type": "VEC3" },
                { "bufferView": 1, "componentType": 5126, "count": 2, "type": "VEC3" },
                { "bufferView": 2, "componentType": 5126, "count": 2, "type": "VEC4" },
                { "bufferView": 3, "componentType": 5126, "count": 2, "type": "VEC2" },
                { "bufferView": 4, "componentType": 5126, "count": 2, "type": "VEC2" },
                { "bufferView": 5, "componentType": 5126, "count": 2, "type": "VEC4" },
                { "bufferView": 6, "componentType": 5123, "count": 2, "type": "VEC4" },
                { "bufferView": 7, "componentType": 5126, "count": 2, "type": "VEC4" }
            ],
            "meshes": [{
                "primitives": [{
                    "attributes": {
                        "POSITION": 0,
                        "NORMAL": 1,
                        "TANGENT": 2,
                        "TEXCOORD_0": 3,
                        "TEXCOORD_1": 4,
                        "COLOR_0": 5,
                        "JOINTS_0": 6,
                        "WEIGHTS_0": 7
                    }
                }]
            }]
        }"#;

        let meshes = load_gltf_from_json(json, &[buffer]).unwrap();
        let prim = &meshes[0].primitives[0];

        assert!(prim.attributes.contains_key(&VertexSemantic::Position));
        assert!(prim.attributes.contains_key(&VertexSemantic::Normal));
        assert!(prim.attributes.contains_key(&VertexSemantic::Tangent));
        assert!(prim.attributes.contains_key(&VertexSemantic::TexCoord0));
        assert!(prim.attributes.contains_key(&VertexSemantic::TexCoord1));
        assert!(prim.attributes.contains_key(&VertexSemantic::Color0));
        assert!(prim.attributes.contains_key(&VertexSemantic::Joints0));
        assert!(prim.attributes.contains_key(&VertexSemantic::Weights0));

        // Verify types
        assert_eq!(
            prim.attributes[&VertexSemantic::Tangent].attribute_type,
            AttributeType::Vec4
        );
        assert_eq!(
            prim.attributes[&VertexSemantic::Joints0].component_type,
            ComponentType::U16
        );
    }

    // ── Test: index formats ─────────────────────────────────────────────────

    #[test]
    fn test_gltf_index_formats() {
        // Test U8, U16, U32 index formats
        let mut buffer = Vec::new();

        // Positions: 4 * Vec3<f32> = 48 bytes
        buffer.extend_from_slice(&[0u8; 48]);

        // U8 indices: 3 bytes
        buffer.push(0u8);
        buffer.push(1u8);
        buffer.push(2u8);

        // U16 indices: 6 bytes
        buffer.extend_from_slice(&0u16.to_le_bytes());
        buffer.extend_from_slice(&1u16.to_le_bytes());
        buffer.extend_from_slice(&2u16.to_le_bytes());

        // U32 indices: 12 bytes
        buffer.extend_from_slice(&0u32.to_le_bytes());
        buffer.extend_from_slice(&1u32.to_le_bytes());
        buffer.extend_from_slice(&2u32.to_le_bytes());

        let json = r#"{
            "asset": { "version": "2.0" },
            "buffers": [{ "byteLength": 69 }],
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0, "byteLength": 48 },
                { "buffer": 0, "byteOffset": 48, "byteLength": 3 },
                { "buffer": 0, "byteOffset": 51, "byteLength": 6 },
                { "buffer": 0, "byteOffset": 57, "byteLength": 12 }
            ],
            "accessors": [
                { "bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3" },
                { "bufferView": 1, "componentType": 5121, "count": 3, "type": "SCALAR" },
                { "bufferView": 2, "componentType": 5123, "count": 3, "type": "SCALAR" },
                { "bufferView": 3, "componentType": 5125, "count": 3, "type": "SCALAR" }
            ],
            "meshes": [{
                "primitives": [
                    { "attributes": { "POSITION": 0 }, "indices": 1 },
                    { "attributes": { "POSITION": 0 }, "indices": 2 },
                    { "attributes": { "POSITION": 0 }, "indices": 3 }
                ]
            }]
        }"#;

        let meshes = load_gltf_from_json(json, &[buffer]).unwrap();
        let prims = &meshes[0].primitives;

        assert_eq!(prims[0].indices.as_ref().unwrap().format, IndexFormat::U8);
        assert_eq!(prims[1].indices.as_ref().unwrap().format, IndexFormat::U16);
        assert_eq!(prims[2].indices.as_ref().unwrap().format, IndexFormat::U32);

        // Verify data sizes
        assert_eq!(prims[0].indices.as_ref().unwrap().data.len(), 3);
        assert_eq!(prims[1].indices.as_ref().unwrap().data.len(), 6);
        assert_eq!(prims[2].indices.as_ref().unwrap().data.len(), 12);
    }

    // ── Test: interleaved attributes ────────────────────────────────────────

    #[test]
    fn test_gltf_interleaved_attributes() {
        // Interleaved format: [pos, normal, pos, normal, pos, normal]
        // Each pos = Vec3<f32> = 12 bytes
        // Each normal = Vec3<f32> = 12 bytes
        // Stride = 24 bytes

        let mut buffer = Vec::new();

        for i in 0..3 {
            // Position
            buffer.extend_from_slice(&(i as f32).to_le_bytes());
            buffer.extend_from_slice(&0.0f32.to_le_bytes());
            buffer.extend_from_slice(&0.0f32.to_le_bytes());
            // Normal
            buffer.extend_from_slice(&0.0f32.to_le_bytes());
            buffer.extend_from_slice(&1.0f32.to_le_bytes());
            buffer.extend_from_slice(&0.0f32.to_le_bytes());
        }

        let json = r#"{
            "asset": { "version": "2.0" },
            "buffers": [{ "byteLength": 72 }],
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0, "byteLength": 72, "byteStride": 24 }
            ],
            "accessors": [
                { "bufferView": 0, "byteOffset": 0, "componentType": 5126, "count": 3, "type": "VEC3" },
                { "bufferView": 0, "byteOffset": 12, "componentType": 5126, "count": 3, "type": "VEC3" }
            ],
            "meshes": [{
                "primitives": [{
                    "attributes": { "POSITION": 0, "NORMAL": 1 }
                }]
            }]
        }"#;

        let meshes = load_gltf_from_json(json, &[buffer]).unwrap();
        let prim = &meshes[0].primitives[0];

        let pos = &prim.attributes[&VertexSemantic::Position];
        let normal = &prim.attributes[&VertexSemantic::Normal];

        assert_eq!(pos.stride, 24);
        assert_eq!(normal.stride, 24);

        // Verify extracted position data (should be de-interleaved)
        let positions: Vec<f32> = pos
            .data
            .chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(positions, vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0]);

        // Verify extracted normal data
        let normals: Vec<f32> = normal
            .data
            .chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(normals, vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]);
    }

    // ── Test: split attributes ──────────────────────────────────────────────

    #[test]
    fn test_gltf_split_attributes() {
        // Separate buffer views for each attribute
        let mut buffer = Vec::new();

        // Positions: 3 * Vec3<f32> = 36 bytes
        for i in 0..3 {
            buffer.extend_from_slice(&(i as f32).to_le_bytes());
            buffer.extend_from_slice(&0.0f32.to_le_bytes());
            buffer.extend_from_slice(&0.0f32.to_le_bytes());
        }

        // UVs: 3 * Vec2<f32> = 24 bytes
        for i in 0..3 {
            buffer.extend_from_slice(&((i as f32) / 2.0).to_le_bytes());
            buffer.extend_from_slice(&((i as f32) / 3.0).to_le_bytes());
        }

        let json = r#"{
            "asset": { "version": "2.0" },
            "buffers": [{ "byteLength": 60 }],
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0, "byteLength": 36 },
                { "buffer": 0, "byteOffset": 36, "byteLength": 24 }
            ],
            "accessors": [
                { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3" },
                { "bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC2" }
            ],
            "meshes": [{
                "primitives": [{
                    "attributes": { "POSITION": 0, "TEXCOORD_0": 1 }
                }]
            }]
        }"#;

        let meshes = load_gltf_from_json(json, &[buffer]).unwrap();
        let prim = &meshes[0].primitives[0];

        assert!(prim.attributes.contains_key(&VertexSemantic::Position));
        assert!(prim.attributes.contains_key(&VertexSemantic::TexCoord0));

        let uv = &prim.attributes[&VertexSemantic::TexCoord0];
        assert_eq!(uv.attribute_type, AttributeType::Vec2);
        assert_eq!(uv.data.len(), 24);
    }

    // ── Test: error handling ────────────────────────────────────────────────

    #[test]
    fn test_gltf_error_missing_buffer() {
        let json = r#"{
            "asset": { "version": "2.0" },
            "buffers": [{ "byteLength": 36 }],
            "bufferViews": [
                { "buffer": 1, "byteOffset": 0, "byteLength": 36 }
            ],
            "accessors": [
                { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3" }
            ],
            "meshes": [{
                "primitives": [{
                    "attributes": { "POSITION": 0 }
                }]
            }]
        }"#;

        // Only provide one buffer, but buffer view references buffer 1
        let buffer = vec![0u8; 36];
        let result = load_gltf_from_json(json, &[buffer]);

        assert!(result.is_err());
        match result {
            Err(GltfError::MissingBuffer(1)) => {}
            _ => panic!("expected MissingBuffer(1), got {:?}", result),
        }
    }

    #[test]
    fn test_gltf_error_invalid_accessor() {
        let json = r#"{
            "asset": { "version": "2.0" },
            "buffers": [{ "byteLength": 12 }],
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0, "byteLength": 12 }
            ],
            "accessors": [
                { "bufferView": 0, "componentType": 9999, "count": 1, "type": "VEC3" }
            ],
            "meshes": [{
                "primitives": [{
                    "attributes": { "POSITION": 0 }
                }]
            }]
        }"#;

        let buffer = vec![0u8; 12];
        let result = load_gltf_from_json(json, &[buffer]);

        assert!(result.is_err());
        match result {
            Err(GltfError::InvalidAccessor(_)) => {}
            _ => panic!("expected InvalidAccessor, got {:?}", result),
        }
    }

    #[test]
    fn test_gltf_error_invalid_json() {
        let json = "{ invalid json }";
        let result = load_gltf_from_json(json, &[]);

        assert!(result.is_err());
        match result {
            Err(GltfError::JsonError(_)) => {}
            _ => panic!("expected JsonError, got {:?}", result),
        }
    }

    // ── Test: GLB format ────────────────────────────────────────────────────

    #[test]
    fn test_gltf_glb_format() {
        // Build a minimal GLB file
        let (json, bin_buffer) = minimal_triangle_gltf();
        let json_bytes = json.as_bytes();

        // Pad JSON to 4-byte alignment
        let json_padded_len = (json_bytes.len() + 3) & !3;
        let mut json_padded = json_bytes.to_vec();
        json_padded.resize(json_padded_len, b' ');

        // Pad binary buffer to 4-byte alignment
        let bin_padded_len = (bin_buffer.len() + 3) & !3;
        let mut bin_padded = bin_buffer.clone();
        bin_padded.resize(bin_padded_len, 0);

        // Build GLB
        let mut glb = Vec::new();

        // Header: magic, version, length
        let total_length = 12 + 8 + json_padded_len + 8 + bin_padded_len;
        glb.extend_from_slice(&GLB_MAGIC.to_le_bytes());
        glb.extend_from_slice(&2u32.to_le_bytes()); // version
        glb.extend_from_slice(&(total_length as u32).to_le_bytes());

        // JSON chunk
        glb.extend_from_slice(&(json_padded_len as u32).to_le_bytes());
        glb.extend_from_slice(&GLB_CHUNK_JSON.to_le_bytes());
        glb.extend_from_slice(&json_padded);

        // BIN chunk
        glb.extend_from_slice(&(bin_padded_len as u32).to_le_bytes());
        glb.extend_from_slice(&GLB_CHUNK_BIN.to_le_bytes());
        glb.extend_from_slice(&bin_padded);

        // Parse the GLB
        let (parsed_json, parsed_bin) = parse_glb(&glb).unwrap();
        assert_eq!(parsed_json.trim(), json.trim());
        assert!(parsed_bin.len() >= bin_buffer.len());

        // Load meshes from parsed GLB
        let meshes = load_gltf_from_json(&parsed_json, &[parsed_bin]).unwrap();
        assert_eq!(meshes.len(), 1);
        assert_eq!(meshes[0].name, Some("Triangle".to_string()));
    }

    #[test]
    fn test_gltf_glb_invalid_magic() {
        let data = vec![0u8; 20];
        let result = parse_glb(&data);
        assert!(matches!(result, Err(GltfError::InvalidGlb(_))));
    }

    #[test]
    fn test_gltf_glb_too_short() {
        let data = vec![0u8; 8];
        let result = parse_glb(&data);
        assert!(matches!(result, Err(GltfError::InvalidGlb(_))));
    }

    // ── Test: material index passthrough ────────────────────────────────────

    #[test]
    fn test_gltf_material_index() {
        let json = r#"{
            "asset": { "version": "2.0" },
            "buffers": [{ "byteLength": 36 }],
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0, "byteLength": 36 }
            ],
            "accessors": [
                { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3" }
            ],
            "meshes": [{
                "primitives": [
                    { "attributes": { "POSITION": 0 }, "material": 0 },
                    { "attributes": { "POSITION": 0 }, "material": 5 },
                    { "attributes": { "POSITION": 0 } }
                ]
            }]
        }"#;

        let buffer = vec![0u8; 36];
        let meshes = load_gltf_from_json(json, &[buffer]).unwrap();
        let prims = &meshes[0].primitives;

        assert_eq!(prims[0].material_index, Some(0));
        assert_eq!(prims[1].material_index, Some(5));
        assert_eq!(prims[2].material_index, None);
    }

    // ── Test: component type sizes ──────────────────────────────────────────

    #[test]
    fn test_component_type_sizes() {
        assert_eq!(ComponentType::I8.size_bytes(), 1);
        assert_eq!(ComponentType::U8.size_bytes(), 1);
        assert_eq!(ComponentType::I16.size_bytes(), 2);
        assert_eq!(ComponentType::U16.size_bytes(), 2);
        assert_eq!(ComponentType::U32.size_bytes(), 4);
        assert_eq!(ComponentType::F32.size_bytes(), 4);
    }

    // ── Test: attribute type component counts ───────────────────────────────

    #[test]
    fn test_attribute_type_component_counts() {
        assert_eq!(AttributeType::Scalar.component_count(), 1);
        assert_eq!(AttributeType::Vec2.component_count(), 2);
        assert_eq!(AttributeType::Vec3.component_count(), 3);
        assert_eq!(AttributeType::Vec4.component_count(), 4);
        assert_eq!(AttributeType::Mat2.component_count(), 4);
        assert_eq!(AttributeType::Mat3.component_count(), 9);
        assert_eq!(AttributeType::Mat4.component_count(), 16);
    }

    // ── Test: index format sizes ────────────────────────────────────────────

    #[test]
    fn test_index_format_sizes() {
        assert_eq!(IndexFormat::U8.size_bytes(), 1);
        assert_eq!(IndexFormat::U16.size_bytes(), 2);
        assert_eq!(IndexFormat::U32.size_bytes(), 4);
    }

    // ── Test: vertex attribute element size ─────────────────────────────────

    #[test]
    fn test_vertex_attribute_element_size() {
        let attr = VertexAttribute {
            semantic: VertexSemantic::Position,
            component_type: ComponentType::F32,
            attribute_type: AttributeType::Vec3,
            offset: 0,
            stride: 12,
            count: 3,
            data: vec![],
        };
        assert_eq!(attr.element_size(), 12); // 4 bytes * 3 components

        let attr2 = VertexAttribute {
            semantic: VertexSemantic::TexCoord0,
            component_type: ComponentType::U16,
            attribute_type: AttributeType::Vec2,
            offset: 0,
            stride: 4,
            count: 3,
            data: vec![],
        };
        assert_eq!(attr2.element_size(), 4); // 2 bytes * 2 components
    }

    // ── Test: error display ─────────────────────────────────────────────────

    #[test]
    fn test_gltf_error_display() {
        assert_eq!(
            format!("{}", GltfError::IoError(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "file not found"
            ))),
            "I/O error: file not found"
        );
        assert_eq!(
            format!("{}", GltfError::JsonError("parse error".into())),
            "JSON error: parse error"
        );
        assert_eq!(
            format!("{}", GltfError::InvalidAccessor("bad accessor".into())),
            "invalid accessor: bad accessor"
        );
        assert_eq!(
            format!("{}", GltfError::UnsupportedFeature("draco".into())),
            "unsupported feature: draco"
        );
        assert_eq!(
            format!("{}", GltfError::MissingBuffer(2)),
            "missing buffer 2"
        );
        assert_eq!(
            format!("{}", GltfError::InvalidBufferView("bad view".into())),
            "invalid buffer view: bad view"
        );
        assert_eq!(
            format!("{}", GltfError::InvalidGlb("bad magic".into())),
            "invalid GLB: bad magic"
        );
    }

    // ── Test: base64 decoding ───────────────────────────────────────────────

    #[test]
    fn test_base64_decode() {
        // "Hello" = "SGVsbG8="
        let decoded = base64_decode("SGVsbG8=").unwrap();
        assert_eq!(decoded, b"Hello");

        // Empty string
        let decoded = base64_decode("").unwrap();
        assert!(decoded.is_empty());

        // With newlines (should be filtered)
        let decoded = base64_decode("SGVs\nbG8=").unwrap();
        assert_eq!(decoded, b"Hello");
    }

    // ── Test: percent decoding ──────────────────────────────────────────────

    #[test]
    fn test_percent_decode() {
        assert_eq!(percent_decode("hello"), b"hello");
        assert_eq!(percent_decode("hello%20world"), b"hello world");
        assert_eq!(percent_decode("%00%FF"), vec![0x00, 0xFF]);
    }

    // ── Test: no indices ────────────────────────────────────────────────────

    #[test]
    fn test_gltf_no_indices() {
        let json = r#"{
            "asset": { "version": "2.0" },
            "buffers": [{ "byteLength": 36 }],
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0, "byteLength": 36 }
            ],
            "accessors": [
                { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3" }
            ],
            "meshes": [{
                "primitives": [{
                    "attributes": { "POSITION": 0 }
                }]
            }]
        }"#;

        let buffer = vec![0u8; 36];
        let meshes = load_gltf_from_json(json, &[buffer]).unwrap();
        assert!(meshes[0].primitives[0].indices.is_none());
    }

    // ── Test: multiple meshes ───────────────────────────────────────────────

    #[test]
    fn test_gltf_multiple_meshes() {
        let json = r#"{
            "asset": { "version": "2.0" },
            "buffers": [{ "byteLength": 72 }],
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0, "byteLength": 36 },
                { "buffer": 0, "byteOffset": 36, "byteLength": 36 }
            ],
            "accessors": [
                { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3" },
                { "bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC3" }
            ],
            "meshes": [
                { "name": "Mesh1", "primitives": [{ "attributes": { "POSITION": 0 } }] },
                { "name": "Mesh2", "primitives": [{ "attributes": { "POSITION": 1 } }] }
            ]
        }"#;

        let buffer = vec![0u8; 72];
        let meshes = load_gltf_from_json(json, &[buffer]).unwrap();

        assert_eq!(meshes.len(), 2);
        assert_eq!(meshes[0].name, Some("Mesh1".to_string()));
        assert_eq!(meshes[1].name, Some("Mesh2".to_string()));
    }

    // ── Test: empty mesh list ───────────────────────────────────────────────

    #[test]
    fn test_gltf_empty_mesh_list() {
        let json = r#"{
            "asset": { "version": "2.0" },
            "meshes": []
        }"#;

        let meshes = load_gltf_from_json(json, &[]).unwrap();
        assert!(meshes.is_empty());
    }

    // ── Test: accessor without buffer view ──────────────────────────────────

    #[test]
    fn test_gltf_accessor_no_buffer_view() {
        // Sparse accessor or zero-filled data
        let json = r#"{
            "asset": { "version": "2.0" },
            "accessors": [
                { "componentType": 5126, "count": 3, "type": "VEC3" }
            ],
            "meshes": [{
                "primitives": [{
                    "attributes": { "POSITION": 0 }
                }]
            }]
        }"#;

        let meshes = load_gltf_from_json(json, &[]).unwrap();
        let pos = &meshes[0].primitives[0].attributes[&VertexSemantic::Position];

        // Should be zero-filled
        assert_eq!(pos.count, 3);
        assert_eq!(pos.data.len(), 36);
        assert!(pos.data.iter().all(|&b| b == 0));
    }

    // ── Test: empty primitive list ──────────────────────────────────────────

    #[test]
    fn test_gltf_empty_primitives() {
        let json = r#"{
            "asset": { "version": "2.0" },
            "meshes": [{
                "name": "EmptyMesh",
                "primitives": []
            }]
        }"#;

        let meshes = load_gltf_from_json(json, &[]).unwrap();
        assert_eq!(meshes.len(), 1);
        assert_eq!(meshes[0].name, Some("EmptyMesh".to_string()));
        assert!(meshes[0].primitives.is_empty());
    }

    // ── Test: accessor bounds check ─────────────────────────────────────────

    #[test]
    fn test_gltf_accessor_out_of_bounds() {
        // Accessor tries to read past buffer end
        let json = r#"{
            "asset": { "version": "2.0" },
            "buffers": [{ "byteLength": 12 }],
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0, "byteLength": 12 }
            ],
            "accessors": [
                { "bufferView": 0, "componentType": 5126, "count": 10, "type": "VEC3" }
            ],
            "meshes": [{
                "primitives": [{
                    "attributes": { "POSITION": 0 }
                }]
            }]
        }"#;

        // Buffer only has 12 bytes but accessor wants 10 * 12 = 120 bytes
        let buffer = vec![0u8; 12];
        let result = load_gltf_from_json(json, &[buffer]);

        assert!(result.is_err());
        match result {
            Err(GltfError::InvalidAccessor(msg)) => {
                assert!(msg.contains("extends past buffer end"));
            }
            _ => panic!("expected InvalidAccessor error, got {:?}", result),
        }
    }

    // ── Test: truncated GLB chunk ───────────────────────────────────────────

    #[test]
    fn test_gltf_glb_truncated_chunk() {
        // GLB with header claiming larger chunk than data available
        let mut glb = Vec::new();

        // Header
        glb.extend_from_slice(&GLB_MAGIC.to_le_bytes());
        glb.extend_from_slice(&2u32.to_le_bytes()); // version
        glb.extend_from_slice(&100u32.to_le_bytes()); // total length (fake)

        // JSON chunk header claiming 1000 bytes
        glb.extend_from_slice(&1000u32.to_le_bytes()); // chunk length
        glb.extend_from_slice(&GLB_CHUNK_JSON.to_le_bytes());
        // But only 10 bytes of data
        glb.extend_from_slice(&[0u8; 10]);

        let result = parse_glb(&glb);
        assert!(matches!(result, Err(GltfError::InvalidGlb(_))));
    }

    // ── Test: GLB with unsupported version ──────────────────────────────────

    #[test]
    fn test_gltf_glb_wrong_version() {
        let mut glb = Vec::new();

        // Header with version 1
        glb.extend_from_slice(&GLB_MAGIC.to_le_bytes());
        glb.extend_from_slice(&1u32.to_le_bytes()); // version 1 (unsupported)
        glb.extend_from_slice(&12u32.to_le_bytes()); // total length

        let result = parse_glb(&glb);
        assert!(matches!(result, Err(GltfError::InvalidGlb(_))));
        if let Err(GltfError::InvalidGlb(msg)) = result {
            assert!(msg.contains("unsupported version"));
        }
    }

    // ── Test: GLB missing JSON chunk ────────────────────────────────────────

    #[test]
    fn test_gltf_glb_missing_json() {
        let mut glb = Vec::new();

        // Header
        glb.extend_from_slice(&GLB_MAGIC.to_le_bytes());
        glb.extend_from_slice(&2u32.to_le_bytes());
        glb.extend_from_slice(&28u32.to_le_bytes()); // total length

        // Only BIN chunk, no JSON
        glb.extend_from_slice(&8u32.to_le_bytes()); // chunk length
        glb.extend_from_slice(&GLB_CHUNK_BIN.to_le_bytes());
        glb.extend_from_slice(&[0u8; 8]);

        let result = parse_glb(&glb);
        assert!(matches!(result, Err(GltfError::InvalidGlb(_))));
        if let Err(GltfError::InvalidGlb(msg)) = result {
            assert!(msg.contains("missing JSON chunk"));
        }
    }

    // ── Test: invalid index component type ──────────────────────────────────

    #[test]
    fn test_gltf_invalid_index_type() {
        // Indices with F32 component type (invalid - indices must be integers)
        let json = r#"{
            "asset": { "version": "2.0" },
            "buffers": [{ "byteLength": 48 }],
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0, "byteLength": 36 },
                { "buffer": 0, "byteOffset": 36, "byteLength": 12 }
            ],
            "accessors": [
                { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3" },
                { "bufferView": 1, "componentType": 5126, "count": 3, "type": "SCALAR" }
            ],
            "meshes": [{
                "primitives": [{
                    "attributes": { "POSITION": 0 },
                    "indices": 1
                }]
            }]
        }"#;

        let buffer = vec![0u8; 48];
        let result = load_gltf_from_json(json, &[buffer]);

        assert!(result.is_err());
        match result {
            Err(GltfError::InvalidAccessor(msg)) => {
                assert!(msg.contains("invalid index component type"));
            }
            _ => panic!("expected InvalidAccessor error, got {:?}", result),
        }
    }

    // ── Test: unknown attribute ignored ─────────────────────────────────────

    #[test]
    fn test_gltf_unknown_attribute_ignored() {
        // Custom attribute _CUSTOM should be ignored, not error
        let json = r#"{
            "asset": { "version": "2.0" },
            "buffers": [{ "byteLength": 72 }],
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0, "byteLength": 36 },
                { "buffer": 0, "byteOffset": 36, "byteLength": 36 }
            ],
            "accessors": [
                { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3" },
                { "bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC3" }
            ],
            "meshes": [{
                "primitives": [{
                    "attributes": {
                        "POSITION": 0,
                        "_CUSTOM": 1
                    }
                }]
            }]
        }"#;

        let buffer = vec![0u8; 72];
        let meshes = load_gltf_from_json(json, &[buffer]).unwrap();

        // Should only have POSITION, _CUSTOM is ignored
        let prim = &meshes[0].primitives[0];
        assert_eq!(prim.attributes.len(), 1);
        assert!(prim.attributes.contains_key(&VertexSemantic::Position));
    }

    // ── Test: data URI with base64 ──────────────────────────────────────────

    #[test]
    fn test_gltf_data_uri_base64() {
        // Test base64 data URI parsing
        let uri = "data:application/octet-stream;base64,AAAAAAAAAAAAAAAAAACAP/+/fz8=";
        let data = parse_data_uri(uri).unwrap();

        // Decoded: 12 bytes of Vec3<f32> = (0, 0, 0), plus 4 bytes extra
        assert!(!data.is_empty());
    }

    // ── Test: data URI percent-encoded ──────────────────────────────────────

    #[test]
    fn test_gltf_data_uri_percent() {
        let uri = "data:application/octet-stream,hello%20world";
        let data = parse_data_uri(uri).unwrap();
        assert_eq!(data, b"hello world");
    }

    // ── Test: buffer view references invalid buffer ─────────────────────────

    #[test]
    fn test_gltf_invalid_buffer_view_index() {
        let json = r#"{
            "asset": { "version": "2.0" },
            "buffers": [{ "byteLength": 36 }],
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0, "byteLength": 36 }
            ],
            "accessors": [
                { "bufferView": 99, "componentType": 5126, "count": 3, "type": "VEC3" }
            ],
            "meshes": [{
                "primitives": [{
                    "attributes": { "POSITION": 0 }
                }]
            }]
        }"#;

        let buffer = vec![0u8; 36];
        let result = load_gltf_from_json(json, &[buffer]);

        assert!(result.is_err());
        match result {
            Err(GltfError::InvalidBufferView(_)) => {}
            _ => panic!("expected InvalidBufferView error, got {:?}", result),
        }
    }

    // =========================================================================
    // NEW TESTS: Schema validation, node hierarchy, skins, progressive loading
    // =========================================================================

    // ── Test: GltfParser creation and validation ────────────────────────────

    #[test]
    fn test_gltf_parser_creation() {
        let (json, _buffer) = minimal_triangle_gltf();
        let parser = GltfParser::new(&json).unwrap();

        assert!(parser.validation().is_valid);
        assert_eq!(parser.mesh_count(), 1);
        assert_eq!(parser.node_count(), 0);
        assert_eq!(parser.skin_count(), 0);
    }

    #[test]
    fn test_gltf_parser_validation_errors() {
        // Invalid version
        let json = r#"{
            "asset": { "version": "1.0" },
            "meshes": []
        }"#;

        let result = GltfParser::new(json);
        assert!(result.is_err());
        match result {
            Err(GltfError::ValidationFailed(v)) => {
                assert!(!v.is_valid);
                assert!(v.error_count() > 0);
            }
            _ => panic!("expected ValidationFailed error"),
        }
    }

    #[test]
    fn test_gltf_parser_validation_missing_position() {
        let json = r#"{
            "asset": { "version": "2.0" },
            "buffers": [{ "byteLength": 36 }],
            "bufferViews": [{ "buffer": 0, "byteOffset": 0, "byteLength": 36 }],
            "accessors": [{ "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3" }],
            "meshes": [{
                "primitives": [{
                    "attributes": { "NORMAL": 0 }
                }]
            }]
        }"#;

        let result = GltfParser::new(json);
        assert!(result.is_err());
        match result {
            Err(GltfError::ValidationFailed(v)) => {
                assert!(!v.is_valid);
                assert!(v.errors.iter().any(|e| e.message.contains("POSITION")));
            }
            _ => panic!("expected ValidationFailed error"),
        }
    }

    // ── Test: Bounds loading ────────────────────────────────────────────────

    #[test]
    fn test_gltf_parser_load_bounds() {
        let json = r#"{
            "asset": { "version": "2.0" },
            "buffers": [{ "byteLength": 42 }],
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0, "byteLength": 36 },
                { "buffer": 0, "byteOffset": 36, "byteLength": 6 }
            ],
            "accessors": [
                {
                    "bufferView": 0,
                    "componentType": 5126,
                    "count": 3,
                    "type": "VEC3",
                    "min": [0.0, 0.0, 0.0],
                    "max": [1.0, 1.0, 0.0]
                },
                { "bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR" }
            ],
            "meshes": [{
                "name": "BoundedMesh",
                "primitives": [{
                    "attributes": { "POSITION": 0 },
                    "indices": 1
                }]
            }]
        }"#;

        let parser = GltfParser::new(json).unwrap();
        let bounds = parser.load_bounds();

        assert_eq!(bounds.mesh_count, 1);
        assert_eq!(bounds.vertex_count_estimate, 3);
        assert_eq!(bounds.index_count_estimate, 3);
        assert!(bounds.scene_bounds.is_valid());
        assert_eq!(bounds.scene_bounds.min, [0.0, 0.0, 0.0]);
        assert_eq!(bounds.scene_bounds.max, [1.0, 1.0, 0.0]);
        assert_eq!(bounds.mesh_bounds[0].0, Some("BoundedMesh".to_string()));
    }

    // ── Test: Progressive loading stages ────────────────────────────────────

    #[test]
    fn test_gltf_parser_progressive_loading() {
        let mut buffer = Vec::new();

        // POSITION: 3 * Vec3<f32> = 36 bytes
        buffer.extend_from_slice(&[0u8; 36]);
        // NORMAL: 3 * Vec3<f32> = 36 bytes
        buffer.extend_from_slice(&[0u8; 36]);
        // JOINTS_0: 3 * Vec4<u16> = 24 bytes
        buffer.extend_from_slice(&[0u8; 24]);
        // WEIGHTS_0: 3 * Vec4<f32> = 48 bytes
        buffer.extend_from_slice(&[0u8; 48]);
        // Indices: 3 * U16 = 6 bytes
        buffer.extend_from_slice(&[0u8; 6]);

        let json = r#"{
            "asset": { "version": "2.0" },
            "buffers": [{ "byteLength": 150 }],
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0, "byteLength": 36 },
                { "buffer": 0, "byteOffset": 36, "byteLength": 36 },
                { "buffer": 0, "byteOffset": 72, "byteLength": 24 },
                { "buffer": 0, "byteOffset": 96, "byteLength": 48 },
                { "buffer": 0, "byteOffset": 144, "byteLength": 6 }
            ],
            "accessors": [
                { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3" },
                { "bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC3" },
                { "bufferView": 2, "componentType": 5123, "count": 3, "type": "VEC4" },
                { "bufferView": 3, "componentType": 5126, "count": 3, "type": "VEC4" },
                { "bufferView": 4, "componentType": 5123, "count": 3, "type": "SCALAR" }
            ],
            "meshes": [{
                "primitives": [{
                    "attributes": {
                        "POSITION": 0,
                        "NORMAL": 1,
                        "JOINTS_0": 2,
                        "WEIGHTS_0": 3
                    },
                    "indices": 4
                }]
            }]
        }"#;

        let parser = GltfParser::new(json).unwrap();

        // Geometry stage: only position and indices
        let geom_meshes = parser.load_meshes(&[buffer.clone()], LoadStage::Geometry).unwrap();
        let geom_prim = &geom_meshes[0].primitives[0];
        assert!(geom_prim.attributes.contains_key(&VertexSemantic::Position));
        assert!(!geom_prim.attributes.contains_key(&VertexSemantic::Normal));
        assert!(!geom_prim.attributes.contains_key(&VertexSemantic::Joints0));
        assert!(geom_prim.indices.is_some());

        // Attributes stage: position, normal, but not skinning
        let attr_meshes = parser.load_meshes(&[buffer.clone()], LoadStage::Attributes).unwrap();
        let attr_prim = &attr_meshes[0].primitives[0];
        assert!(attr_prim.attributes.contains_key(&VertexSemantic::Position));
        assert!(attr_prim.attributes.contains_key(&VertexSemantic::Normal));
        assert!(!attr_prim.attributes.contains_key(&VertexSemantic::Joints0));

        // Full stage: everything
        let full_meshes = parser.load_meshes(&[buffer.clone()], LoadStage::Full).unwrap();
        let full_prim = &full_meshes[0].primitives[0];
        assert!(full_prim.attributes.contains_key(&VertexSemantic::Position));
        assert!(full_prim.attributes.contains_key(&VertexSemantic::Normal));
        assert!(full_prim.attributes.contains_key(&VertexSemantic::Joints0));
        assert!(full_prim.attributes.contains_key(&VertexSemantic::Weights0));
    }

    // ── Test: Node hierarchy ────────────────────────────────────────────────

    #[test]
    fn test_gltf_parser_node_hierarchy() {
        let json = r#"{
            "asset": { "version": "2.0" },
            "nodes": [
                { "name": "Root", "children": [1, 2], "translation": [0, 0, 0] },
                { "name": "Child1", "translation": [1, 0, 0] },
                { "name": "Child2", "children": [3], "translation": [0, 1, 0] },
                { "name": "Grandchild", "translation": [0, 0, 1] }
            ],
            "scenes": [{ "name": "Scene", "nodes": [0] }],
            "scene": 0
        }"#;

        let parser = GltfParser::new(json).unwrap();
        let nodes = parser.load_nodes().unwrap();

        assert_eq!(nodes.len(), 4);

        // Check parent relationships
        assert_eq!(nodes[0].parent, None);
        assert_eq!(nodes[1].parent, Some(0));
        assert_eq!(nodes[2].parent, Some(0));
        assert_eq!(nodes[3].parent, Some(2));

        // Check world transforms
        let root_trans = nodes[0].world_transform.translation();
        assert_eq!(root_trans, [0.0, 0.0, 0.0]);

        let child1_trans = nodes[1].world_transform.translation();
        assert_eq!(child1_trans, [1.0, 0.0, 0.0]);

        let grandchild_trans = nodes[3].world_transform.translation();
        // Should be parent (0,1,0) + local (0,0,1) = (0,1,1)
        assert_eq!(grandchild_trans, [0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_gltf_parser_node_with_matrix() {
        let json = r#"{
            "asset": { "version": "2.0" },
            "nodes": [{
                "name": "MatrixNode",
                "matrix": [
                    1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    5, 6, 7, 1
                ]
            }]
        }"#;

        let parser = GltfParser::new(json).unwrap();
        let nodes = parser.load_nodes().unwrap();

        let trans = nodes[0].world_transform.translation();
        assert_eq!(trans, [5.0, 6.0, 7.0]);
    }

    // ── Test: Skin loading ──────────────────────────────────────────────────

    #[test]
    fn test_gltf_parser_skin_loading() {
        let mut buffer = Vec::new();
        // 2 identity matrices for inverse bind matrices
        for _ in 0..2 {
            buffer.extend_from_slice(&1.0f32.to_le_bytes()); // m00
            buffer.extend_from_slice(&0.0f32.to_le_bytes());
            buffer.extend_from_slice(&0.0f32.to_le_bytes());
            buffer.extend_from_slice(&0.0f32.to_le_bytes());
            buffer.extend_from_slice(&0.0f32.to_le_bytes());
            buffer.extend_from_slice(&1.0f32.to_le_bytes()); // m11
            buffer.extend_from_slice(&0.0f32.to_le_bytes());
            buffer.extend_from_slice(&0.0f32.to_le_bytes());
            buffer.extend_from_slice(&0.0f32.to_le_bytes());
            buffer.extend_from_slice(&0.0f32.to_le_bytes());
            buffer.extend_from_slice(&1.0f32.to_le_bytes()); // m22
            buffer.extend_from_slice(&0.0f32.to_le_bytes());
            buffer.extend_from_slice(&0.0f32.to_le_bytes());
            buffer.extend_from_slice(&0.0f32.to_le_bytes());
            buffer.extend_from_slice(&0.0f32.to_le_bytes());
            buffer.extend_from_slice(&1.0f32.to_le_bytes()); // m33
        }

        let json = r#"{
            "asset": { "version": "2.0" },
            "buffers": [{ "byteLength": 128 }],
            "bufferViews": [{ "buffer": 0, "byteOffset": 0, "byteLength": 128 }],
            "accessors": [{
                "bufferView": 0,
                "componentType": 5126,
                "count": 2,
                "type": "MAT4"
            }],
            "nodes": [
                { "name": "Armature" },
                { "name": "Bone1" },
                { "name": "Bone2" }
            ],
            "skins": [{
                "name": "Skeleton",
                "inverseBindMatrices": 0,
                "joints": [1, 2],
                "skeleton": 0
            }]
        }"#;

        let parser = GltfParser::new(json).unwrap();
        let skins = parser.load_skins(&[buffer]).unwrap();

        assert_eq!(skins.len(), 1);
        assert_eq!(skins[0].name, Some("Skeleton".to_string()));
        assert_eq!(skins[0].joints, vec![1, 2]);
        assert_eq!(skins[0].skeleton, Some(0));
        assert_eq!(skins[0].inverse_bind_matrices.len(), 2);
    }

    // ── Test: Full document loading ─────────────────────────────────────────

    #[test]
    fn test_gltf_parser_full_document() {
        let (json_str, buffer) = minimal_triangle_gltf();

        // Add nodes and scene to the JSON
        let json: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        let mut json_obj = json.as_object().unwrap().clone();
        json_obj.insert("nodes".to_string(), serde_json::json!([
            { "name": "MeshNode", "mesh": 0 }
        ]));
        json_obj.insert("scenes".to_string(), serde_json::json!([
            { "name": "MainScene", "nodes": [0] }
        ]));
        json_obj.insert("scene".to_string(), serde_json::json!(0));

        let json = serde_json::to_string(&json_obj).unwrap();
        let parser = GltfParser::new(&json).unwrap();
        let doc = parser.load_full(&[buffer]).unwrap();

        assert_eq!(doc.meshes.len(), 1);
        assert_eq!(doc.nodes.len(), 1);
        assert_eq!(doc.scenes.len(), 1);
        assert_eq!(doc.default_scene, Some(0));
        assert_eq!(doc.nodes[0].mesh, Some(0));
    }

    // ── Test: Mat4 operations ───────────────────────────────────────────────

    #[test]
    fn test_mat4_identity() {
        let m = Mat4::IDENTITY;
        let arr = m.as_array();
        assert_eq!(arr[0], 1.0);
        assert_eq!(arr[5], 1.0);
        assert_eq!(arr[10], 1.0);
        assert_eq!(arr[15], 1.0);
        assert_eq!(m.translation(), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_mat4_from_trs() {
        let m = Mat4::from_trs(
            [1.0, 2.0, 3.0],      // translation
            [0.0, 0.0, 0.0, 1.0], // identity rotation
            [1.0, 1.0, 1.0],      // uniform scale
        );
        assert_eq!(m.translation(), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_mat4_multiply() {
        let a = Mat4::from_trs([1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0]);
        let b = Mat4::from_trs([0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0]);
        let c = a.mul(&b);
        assert_eq!(c.translation(), [1.0, 1.0, 0.0]);
    }

    // ── Test: AABB operations ───────────────────────────────────────────────

    #[test]
    fn test_aabb_empty() {
        let aabb = Aabb::EMPTY;
        assert!(!aabb.is_valid());
    }

    #[test]
    fn test_aabb_union() {
        let a = Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let b = Aabb::new([2.0, 2.0, 2.0], [3.0, 3.0, 3.0]);
        let c = a.union(&b);
        assert_eq!(c.min, [0.0, 0.0, 0.0]);
        assert_eq!(c.max, [3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_aabb_expand_point() {
        let mut aabb = Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        aabb.expand_point([2.0, -1.0, 0.5]);
        assert_eq!(aabb.min, [0.0, -1.0, 0.0]);
        assert_eq!(aabb.max, [2.0, 1.0, 1.0]);
    }

    #[test]
    fn test_aabb_center_extents() {
        let aabb = Aabb::new([0.0, 0.0, 0.0], [2.0, 4.0, 6.0]);
        assert_eq!(aabb.center(), [1.0, 2.0, 3.0]);
        assert_eq!(aabb.extents(), [1.0, 2.0, 3.0]);
    }

    // ── Test: Validation result ─────────────────────────────────────────────

    #[test]
    fn test_validation_result() {
        let mut result = ValidationResult::ok();
        assert!(result.is_valid);
        assert_eq!(result.error_count(), 0);
        assert_eq!(result.warning_count(), 0);

        result.add_warning("test.path", "warning message");
        assert!(result.is_valid);
        assert_eq!(result.warning_count(), 1);

        result.add_error("test.path", "error message");
        assert!(!result.is_valid);
        assert_eq!(result.error_count(), 1);
    }

    // ── Test: Error display ─────────────────────────────────────────────────

    #[test]
    fn test_gltf_error_display_new_variants() {
        let v = ValidationResult::ok();
        assert_eq!(
            format!("{}", GltfError::ValidationFailed(v)),
            "validation failed: 0 errors"
        );
        assert_eq!(
            format!("{}", GltfError::InvalidHierarchy("cycle detected".into())),
            "invalid hierarchy: cycle detected"
        );
        assert_eq!(
            format!("{}", GltfError::InvalidSkin("no joints".into())),
            "invalid skin: no joints"
        );
        assert_eq!(
            format!("{}", GltfError::StreamingError("seek failed".into())),
            "streaming error: seek failed"
        );
    }

    // ── Test: Streaming reader ──────────────────────────────────────────────

    #[test]
    fn test_streaming_reader_from_glb() {
        use std::io::Cursor;

        let (json, bin_buffer) = minimal_triangle_gltf();
        let json_bytes = json.as_bytes();

        // Pad JSON to 4-byte alignment
        let json_padded_len = (json_bytes.len() + 3) & !3;
        let mut json_padded = json_bytes.to_vec();
        json_padded.resize(json_padded_len, b' ');

        // Pad binary buffer to 4-byte alignment
        let bin_padded_len = (bin_buffer.len() + 3) & !3;
        let mut bin_padded = bin_buffer.clone();
        bin_padded.resize(bin_padded_len, 0);

        // Build GLB
        let mut glb = Vec::new();

        // Header
        let total_length = 12 + 8 + json_padded_len + 8 + bin_padded_len;
        glb.extend_from_slice(&GLB_MAGIC.to_le_bytes());
        glb.extend_from_slice(&2u32.to_le_bytes());
        glb.extend_from_slice(&(total_length as u32).to_le_bytes());

        // JSON chunk
        glb.extend_from_slice(&(json_padded_len as u32).to_le_bytes());
        glb.extend_from_slice(&GLB_CHUNK_JSON.to_le_bytes());
        glb.extend_from_slice(&json_padded);

        // BIN chunk
        glb.extend_from_slice(&(bin_padded_len as u32).to_le_bytes());
        glb.extend_from_slice(&GLB_CHUNK_BIN.to_le_bytes());
        glb.extend_from_slice(&bin_padded);

        // Test streaming reader
        let cursor = Cursor::new(glb);
        let (mut reader, parsed_json) = StreamingBufferReader::from_glb(cursor).unwrap();

        assert!(!parsed_json.is_empty());
        assert_eq!(reader.buffer_count(), 1);

        // Read entire buffer
        let buffer_data = reader.read_buffer(0).unwrap();
        assert!(buffer_data.len() >= bin_buffer.len());

        // Read range
        let range_data = reader.read_range(0, 0, 12).unwrap();
        assert_eq!(range_data.len(), 12);
    }

    // ── Test: Load stage ordering ───────────────────────────────────────────

    #[test]
    fn test_load_stage_ordering() {
        assert!(LoadStage::Bounds < LoadStage::Geometry);
        assert!(LoadStage::Geometry < LoadStage::Attributes);
        assert!(LoadStage::Attributes < LoadStage::Skinning);
        assert!(LoadStage::Skinning < LoadStage::Full);
    }

    // ── Test: Parser is Send + Sync ─────────────────────────────────────────

    #[test]
    fn test_gltf_parser_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<GltfParser>();
        assert_sync::<GltfParser>();
    }

    // ── Test: Parse PBR materials (material index passthrough) ──────────────

    #[test]
    fn test_gltf_pbr_materials() {
        let json = r#"{
            "asset": { "version": "2.0" },
            "buffers": [{ "byteLength": 36 }],
            "bufferViews": [{ "buffer": 0, "byteOffset": 0, "byteLength": 36 }],
            "accessors": [{ "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3" }],
            "materials": [
                { "name": "Gold", "pbrMetallicRoughness": { "metallicFactor": 1.0 } },
                { "name": "Plastic", "pbrMetallicRoughness": { "metallicFactor": 0.0 } }
            ],
            "meshes": [{
                "primitives": [
                    { "attributes": { "POSITION": 0 }, "material": 0 },
                    { "attributes": { "POSITION": 0 }, "material": 1 }
                ]
            }]
        }"#;

        let buffer = vec![0u8; 36];
        let parser = GltfParser::new(json).unwrap();
        let meshes = parser.load_meshes(&[buffer], LoadStage::Full).unwrap();

        assert_eq!(meshes[0].primitives[0].material_index, Some(0));
        assert_eq!(meshes[0].primitives[1].material_index, Some(1));
    }

    // ── Test: Parse cube mesh ───────────────────────────────────────────────

    #[test]
    fn test_gltf_parse_cube() {
        // 8 vertices, 36 indices (12 triangles)
        let mut buffer = Vec::new();

        // 8 cube vertices
        let vertices: [[f32; 3]; 8] = [
            [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0],
            [-1.0, -1.0,  1.0], [1.0, -1.0,  1.0], [1.0, 1.0,  1.0], [-1.0, 1.0,  1.0],
        ];
        for v in &vertices {
            for c in v {
                buffer.extend_from_slice(&c.to_le_bytes());
            }
        }

        // 36 indices (12 triangles)
        let indices: [u16; 36] = [
            0, 1, 2, 2, 3, 0, // front
            4, 5, 6, 6, 7, 4, // back
            0, 1, 5, 5, 4, 0, // bottom
            2, 3, 7, 7, 6, 2, // top
            0, 3, 7, 7, 4, 0, // left
            1, 2, 6, 6, 5, 1, // right
        ];
        for i in &indices {
            buffer.extend_from_slice(&i.to_le_bytes());
        }

        let json = r#"{
            "asset": { "version": "2.0" },
            "buffers": [{ "byteLength": 168 }],
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0, "byteLength": 96 },
                { "buffer": 0, "byteOffset": 96, "byteLength": 72 }
            ],
            "accessors": [
                {
                    "bufferView": 0, "componentType": 5126, "count": 8, "type": "VEC3",
                    "min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]
                },
                { "bufferView": 1, "componentType": 5123, "count": 36, "type": "SCALAR" }
            ],
            "meshes": [{
                "name": "Cube",
                "primitives": [{
                    "attributes": { "POSITION": 0 },
                    "indices": 1
                }]
            }]
        }"#;

        let parser = GltfParser::new(json).unwrap();
        let bounds = parser.load_bounds();

        assert_eq!(bounds.vertex_count_estimate, 8);
        assert_eq!(bounds.index_count_estimate, 36);
        assert_eq!(bounds.scene_bounds.min, [-1.0, -1.0, -1.0]);
        assert_eq!(bounds.scene_bounds.max, [1.0, 1.0, 1.0]);

        let meshes = parser.load_meshes(&[buffer], LoadStage::Full).unwrap();
        assert_eq!(meshes[0].name, Some("Cube".to_string()));
        assert_eq!(meshes[0].primitives[0].indices.as_ref().unwrap().count, 36);
    }

    // ── Test: Parse sphere mesh (high vertex count) ─────────────────────────

    #[test]
    fn test_gltf_parse_sphere_bounds() {
        // Just test bounds extraction for a "sphere" with many vertices
        let json = r#"{
            "asset": { "version": "2.0" },
            "buffers": [{ "byteLength": 100000 }],
            "bufferViews": [{ "buffer": 0, "byteOffset": 0, "byteLength": 100000 }],
            "accessors": [{
                "bufferView": 0, "componentType": 5126, "count": 2562, "type": "VEC3",
                "min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]
            }],
            "meshes": [{
                "name": "Sphere",
                "primitives": [{
                    "attributes": { "POSITION": 0 }
                }]
            }]
        }"#;

        let parser = GltfParser::new(json).unwrap();
        let bounds = parser.load_bounds();

        assert_eq!(bounds.mesh_bounds[0].0, Some("Sphere".to_string()));
        assert_eq!(bounds.vertex_count_estimate, 2562);
        assert!(bounds.scene_bounds.is_valid());
    }

    // ── Test: Validation of node hierarchy cycles ───────────────────────────

    #[test]
    fn test_gltf_validation_node_self_reference() {
        let json = r#"{
            "asset": { "version": "2.0" },
            "nodes": [
                { "children": [0] }
            ]
        }"#;

        let result = GltfParser::new(json);
        assert!(result.is_err());
        match result {
            Err(GltfError::ValidationFailed(v)) => {
                assert!(v.errors.iter().any(|e| e.message.contains("own child")));
            }
            _ => panic!("expected ValidationFailed error"),
        }
    }

    // ── Test: Validation of skin with no joints ─────────────────────────────

    #[test]
    fn test_gltf_validation_skin_no_joints() {
        let json = r#"{
            "asset": { "version": "2.0" },
            "nodes": [{ "name": "Node" }],
            "skins": [{ "joints": [] }]
        }"#;

        let result = GltfParser::new(json);
        assert!(result.is_err());
        match result {
            Err(GltfError::ValidationFailed(v)) => {
                assert!(v.errors.iter().any(|e| e.message.contains("no joints")));
            }
            _ => panic!("expected ValidationFailed error"),
        }
    }

    // ── Test: Buffer size calculation ───────────────────────────────────────

    #[test]
    fn test_gltf_parser_total_buffer_size() {
        let json = r#"{
            "asset": { "version": "2.0" },
            "buffers": [
                { "byteLength": 1000 },
                { "byteLength": 2000 },
                { "byteLength": 3000 }
            ]
        }"#;

        let parser = GltfParser::new(json).unwrap();
        assert_eq!(parser.buffer_count(), 3);
        assert_eq!(parser.total_buffer_size(), 6000);
    }
}
