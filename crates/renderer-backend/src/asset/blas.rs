//! BLAS (Bottom-Level Acceleration Structure) baking pipeline for ray tracing.
//!
//! Provides BLAS build input preparation, compaction, serialization, and
//! integration with the `@ray_tracing` decorator for the TRINITY engine.
//!
//! # Features
//!
//! - Prepare vertex/index data for BLAS build
//! - Support indexed and non-indexed geometry
//! - Build flags configuration (compaction, fast trace, etc.)
//! - Post-build compaction queries and copy
//! - Serialization for PAK archive storage
//! - Skinned mesh support with ALLOW_UPDATE flag
//! - `@ray_tracing` decorator parsing
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::asset::blas::*;
//!
//! // Prepare BLAS input from mesh data
//! let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
//! let indices = vec![0, 1, 2];
//! let geometry = prepare_blas_input(&positions, Some(&indices), None);
//!
//! // Configure BLAS build
//! let config = BlasConfig {
//!     flags: BlasBuildFlags::ALLOW_COMPACTION | BlasBuildFlags::PREFER_FAST_TRACE,
//!     geometries: vec![geometry],
//!     is_skinned: false,
//! };
//!
//! // Compute build sizes
//! let sizes = compute_blas_sizes(&config);
//! println!("Scratch: {}, Result: {}", sizes.scratch_size, sizes.result_size);
//!
//! // Serialize for storage
//! let serialized = serialize_blas(&blas_data, true);
//! let restored = deserialize_blas(&serialized)?;
//! ```

use std::collections::HashMap;

use crate::asset::index_buffer::IndexType;

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// BLAS baking error.
#[derive(Debug, Clone)]
pub enum BlasError {
    /// Invalid vertex data.
    InvalidVertexData(String),
    /// Invalid index data.
    InvalidIndexData(String),
    /// Serialization error.
    SerializationError(String),
    /// Deserialization error.
    DeserializationError(String),
    /// Unsupported configuration.
    UnsupportedConfig(String),
    /// Version mismatch during deserialization.
    VersionMismatch { expected: u32, got: u32 },
}

impl std::fmt::Display for BlasError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidVertexData(msg) => write!(f, "invalid vertex data: {}", msg),
            Self::InvalidIndexData(msg) => write!(f, "invalid index data: {}", msg),
            Self::SerializationError(msg) => write!(f, "serialization error: {}", msg),
            Self::DeserializationError(msg) => write!(f, "deserialization error: {}", msg),
            Self::UnsupportedConfig(msg) => write!(f, "unsupported config: {}", msg),
            Self::VersionMismatch { expected, got } => {
                write!(f, "version mismatch: expected {}, got {}", expected, got)
            }
        }
    }
}

impl std::error::Error for BlasError {}

/// Result type for BLAS operations.
pub type BlasResult<T> = Result<T, BlasError>;

// ---------------------------------------------------------------------------
// Build flags
// ---------------------------------------------------------------------------

bitflags::bitflags! {
    /// BLAS build flags controlling optimization and features.
    ///
    /// These flags map to Vulkan/DXR acceleration structure build flags.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct BlasBuildFlags: u32 {
        /// Enable post-build compaction to reduce memory usage.
        /// Requires querying compacted size after build.
        const ALLOW_COMPACTION = 0x01;

        /// Optimize for faster ray traversal at the cost of slower build.
        /// Best for static geometry that's traced frequently.
        const PREFER_FAST_TRACE = 0x02;

        /// Optimize for faster build at the cost of slower traversal.
        /// Best for geometry that changes frequently or is traced rarely.
        const PREFER_FAST_BUILD = 0x04;

        /// Minimize memory footprint at the cost of performance.
        /// Useful for memory-constrained scenarios.
        const MINIMIZE_MEMORY = 0x08;

        /// Allow runtime updates (refitting) without full rebuild.
        /// Required for skinned/deforming meshes.
        const ALLOW_UPDATE = 0x10;
    }
}

impl Default for BlasBuildFlags {
    fn default() -> Self {
        Self::ALLOW_COMPACTION | Self::PREFER_FAST_TRACE
    }
}

// ---------------------------------------------------------------------------
// Geometry types
// ---------------------------------------------------------------------------

/// BLAS geometry input for a single mesh.
#[derive(Debug, Clone)]
pub struct BlasGeometry {
    /// Vertex position buffer (3x f32 per vertex, tightly packed).
    pub vertex_buffer: Vec<u8>,
    /// Stride between consecutive vertices in bytes.
    pub vertex_stride: u32,
    /// Number of vertices.
    pub vertex_count: u32,
    /// Optional index buffer (U16 or U32 indices).
    pub index_buffer: Option<Vec<u8>>,
    /// Index type (U16 or U32).
    pub index_type: IndexType,
    /// Optional 3x4 row-major transform matrix.
    pub transform: Option<[f32; 12]>,
    /// Geometry flags (opaque, no-duplicate-any-hit, etc.).
    pub geometry_flags: GeometryFlags,
}

impl Default for BlasGeometry {
    fn default() -> Self {
        Self {
            vertex_buffer: Vec::new(),
            vertex_stride: 12, // 3x f32
            vertex_count: 0,
            index_buffer: None,
            index_type: IndexType::U32,
            transform: None,
            geometry_flags: GeometryFlags::OPAQUE,
        }
    }
}

bitflags::bitflags! {
    /// Per-geometry flags for ray tracing behavior.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
    pub struct GeometryFlags: u32 {
        /// Geometry is fully opaque (no any-hit shader invocation).
        const OPAQUE = 0x01;
        /// Prevent duplicate any-hit shader invocations.
        const NO_DUPLICATE_ANY_HIT = 0x02;
    }
}

// ---------------------------------------------------------------------------
// BLAS configuration
// ---------------------------------------------------------------------------

/// Configuration for BLAS build.
#[derive(Debug, Clone)]
pub struct BlasConfig {
    /// Build flags.
    pub flags: BlasBuildFlags,
    /// Geometries to include in this BLAS.
    pub geometries: Vec<BlasGeometry>,
    /// Whether this BLAS contains skinned/deforming geometry.
    pub is_skinned: bool,
    /// Optional name for debugging.
    pub name: Option<String>,
}

impl Default for BlasConfig {
    fn default() -> Self {
        Self {
            flags: BlasBuildFlags::default(),
            geometries: Vec::new(),
            is_skinned: false,
            name: None,
        }
    }
}

impl BlasConfig {
    /// Create a new BLAS config with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a geometry to this BLAS.
    pub fn add_geometry(&mut self, geometry: BlasGeometry) -> &mut Self {
        self.geometries.push(geometry);
        self
    }

    /// Mark as skinned mesh (enables ALLOW_UPDATE).
    pub fn with_skinned(mut self, skinned: bool) -> Self {
        self.is_skinned = skinned;
        if skinned {
            self.flags |= BlasBuildFlags::ALLOW_UPDATE;
        }
        self
    }

    /// Set build flags.
    pub fn with_flags(mut self, flags: BlasBuildFlags) -> Self {
        self.flags = flags;
        self
    }

    /// Set name for debugging.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Get effective build flags (accounting for skinned mesh requirements).
    pub fn effective_flags(&self) -> BlasBuildFlags {
        let mut flags = self.flags;
        if self.is_skinned {
            flags |= BlasBuildFlags::ALLOW_UPDATE;
            // Remove incompatible flags
            flags.remove(BlasBuildFlags::ALLOW_COMPACTION);
        }
        flags
    }

    /// Calculate total vertex count across all geometries.
    pub fn total_vertex_count(&self) -> u64 {
        self.geometries.iter().map(|g| g.vertex_count as u64).sum()
    }

    /// Calculate total triangle count.
    pub fn total_triangle_count(&self) -> u64 {
        self.geometries
            .iter()
            .map(|g| {
                if let Some(ref idx) = g.index_buffer {
                    let index_count = match g.index_type {
                        IndexType::U8 => idx.len(),
                        IndexType::U16 => idx.len() / 2,
                        IndexType::U32 => idx.len() / 4,
                    };
                    (index_count / 3) as u64
                } else {
                    (g.vertex_count / 3) as u64
                }
            })
            .sum()
    }
}

// ---------------------------------------------------------------------------
// Build result
// ---------------------------------------------------------------------------

/// Result of computing BLAS build sizes.
#[derive(Debug, Clone, Copy, Default)]
pub struct BlasBuildResult {
    /// Size of scratch buffer needed for build.
    pub scratch_size: u64,
    /// Size of the result BLAS buffer.
    pub result_size: u64,
    /// Estimated compacted size (if ALLOW_COMPACTION is set).
    pub compacted_size: Option<u64>,
    /// Size of scratch buffer needed for update (if ALLOW_UPDATE is set).
    pub update_scratch_size: u64,
}

// ---------------------------------------------------------------------------
// Serialization types
// ---------------------------------------------------------------------------

/// Current serialization format version.
pub const BLAS_SERIALIZATION_VERSION: u32 = 1;

/// Magic number for BLAS serialization format.
pub const BLAS_MAGIC: [u8; 4] = *b"TBLS";

/// Serialized BLAS for storage.
#[derive(Debug, Clone)]
pub struct SerializedBlas {
    /// Serialized BLAS data (format: magic + version + metadata + data).
    pub data: Vec<u8>,
    /// Serialization format version.
    pub version: u32,
    /// Original (uncompacted) size.
    pub original_size: u64,
    /// Whether the BLAS data is compacted.
    pub compacted: bool,
    /// Build flags used to create this BLAS.
    pub build_flags: BlasBuildFlags,
    /// Number of geometries in the BLAS.
    pub geometry_count: u32,
    /// Total triangle count.
    pub triangle_count: u64,
}

/// Serialized BLAS header (on-disk format).
///
/// Layout (44 bytes total, no padding):
/// - magic:          4 bytes  (offset 0)
/// - version:        4 bytes  (offset 4)
/// - original_size:  8 bytes  (offset 8)
/// - compacted:      4 bytes  (offset 16)
/// - build_flags:    4 bytes  (offset 20)
/// - geometry_count: 4 bytes  (offset 24)
/// - triangle_count: 8 bytes  (offset 28)
/// - data_size:      8 bytes  (offset 36)
#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct BlasHeader {
    magic: [u8; 4],
    version: u32,
    original_size: u64,
    compacted: u32,
    build_flags: u32,
    geometry_count: u32,
    triangle_count: u64,
    data_size: u64,
}

impl BlasHeader {
    /// Header size in bytes (explicitly calculated to avoid padding issues).
    /// 4 + 4 + 8 + 4 + 4 + 4 + 8 + 8 = 44 bytes
    const SIZE: usize = 44;
}

// ---------------------------------------------------------------------------
// Ray tracing decorator types
// ---------------------------------------------------------------------------

/// Parsed ray tracing decorator parameters.
#[derive(Debug, Clone)]
pub struct RayTracingDecoratorParams {
    /// Whether the geometry is dynamic (skinned/deforming).
    pub dynamic: bool,
    /// Prefer fast trace over fast build.
    pub fast_trace: bool,
    /// Enable compaction.
    pub compaction: bool,
    /// LOD level (0 = highest detail).
    pub lod_level: u32,
    /// Custom geometry flags.
    pub geometry_flags: GeometryFlags,
    /// Custom name for the BLAS.
    pub name: Option<String>,
}

impl Default for RayTracingDecoratorParams {
    fn default() -> Self {
        Self {
            dynamic: false,
            fast_trace: true,
            compaction: true,
            lod_level: 0,
            geometry_flags: GeometryFlags::OPAQUE,
            name: None,
        }
    }
}

/// Generic value type for decorator parameters.
#[derive(Debug, Clone)]
pub enum DecoratorValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Array(Vec<DecoratorValue>),
}

impl DecoratorValue {
    /// Get as bool.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }

    /// Get as i64.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Self::Int(v) => Some(*v),
            _ => None,
        }
    }

    /// Get as f64.
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Self::Float(v) => Some(*v),
            Self::Int(v) => Some(*v as f64),
            _ => None,
        }
    }

    /// Get as string.
    pub fn as_string(&self) -> Option<&str> {
        match self {
            Self::String(v) => Some(v),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Core functions
// ---------------------------------------------------------------------------

/// Prepare BLAS input from position and index data.
///
/// # Arguments
///
/// * `positions` - Vertex positions as [x, y, z] arrays
/// * `indices` - Optional triangle indices
/// * `transform` - Optional 3x4 row-major transform matrix
///
/// # Returns
///
/// Prepared `BlasGeometry` ready for BLAS build.
///
/// # Example
///
/// ```ignore
/// let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
/// let indices = vec![0, 1, 2];
/// let geometry = prepare_blas_input(&positions, Some(&indices), None);
/// ```
pub fn prepare_blas_input(
    positions: &[[f32; 3]],
    indices: Option<&[u32]>,
    transform: Option<&[f32; 12]>,
) -> BlasGeometry {
    // Convert positions to bytes
    let mut vertex_buffer = Vec::with_capacity(positions.len() * 12);
    for pos in positions {
        vertex_buffer.extend_from_slice(&pos[0].to_le_bytes());
        vertex_buffer.extend_from_slice(&pos[1].to_le_bytes());
        vertex_buffer.extend_from_slice(&pos[2].to_le_bytes());
    }

    // Convert indices to bytes if present
    let (index_buffer, index_type) = if let Some(idx) = indices {
        let max_index = idx.iter().copied().max().unwrap_or(0);
        if max_index <= u16::MAX as u32 && positions.len() <= 65536 {
            // Use U16 indices
            let mut buffer = Vec::with_capacity(idx.len() * 2);
            for &i in idx {
                buffer.extend_from_slice(&(i as u16).to_le_bytes());
            }
            (Some(buffer), IndexType::U16)
        } else {
            // Use U32 indices
            let mut buffer = Vec::with_capacity(idx.len() * 4);
            for &i in idx {
                buffer.extend_from_slice(&i.to_le_bytes());
            }
            (Some(buffer), IndexType::U32)
        }
    } else {
        (None, IndexType::U32)
    };

    BlasGeometry {
        vertex_buffer,
        vertex_stride: 12,
        vertex_count: positions.len() as u32,
        index_buffer,
        index_type,
        transform: transform.copied(),
        geometry_flags: GeometryFlags::OPAQUE,
    }
}

/// Prepare BLAS input from raw vertex buffer with custom stride.
///
/// # Arguments
///
/// * `vertex_buffer` - Raw vertex position data
/// * `vertex_stride` - Stride between consecutive vertices in bytes
/// * `vertex_count` - Number of vertices
/// * `index_buffer` - Optional raw index data
/// * `index_type` - Index type (U16 or U32)
///
/// # Returns
///
/// Prepared `BlasGeometry` ready for BLAS build.
pub fn prepare_blas_input_raw(
    vertex_buffer: Vec<u8>,
    vertex_stride: u32,
    vertex_count: u32,
    index_buffer: Option<Vec<u8>>,
    index_type: IndexType,
) -> BlasGeometry {
    BlasGeometry {
        vertex_buffer,
        vertex_stride,
        vertex_count,
        index_buffer,
        index_type,
        transform: None,
        geometry_flags: GeometryFlags::OPAQUE,
    }
}

/// Compute BLAS build sizes based on configuration.
///
/// This function estimates the memory requirements for building a BLAS.
/// The actual sizes depend on the driver/hardware, but these estimates
/// are based on typical implementations.
///
/// # Arguments
///
/// * `config` - BLAS build configuration
///
/// # Returns
///
/// Build size requirements.
///
/// # Size Estimation
///
/// - Scratch: ~2x triangle data size
/// - Result: ~1.5x triangle data size (uncompacted)
/// - Compacted: ~0.7x uncompacted size
/// - Update scratch: ~0.5x scratch size
pub fn compute_blas_sizes(config: &BlasConfig) -> BlasBuildResult {
    let total_triangles = config.total_triangle_count();
    let total_vertices = config.total_vertex_count();

    // Base sizes (bytes per triangle/vertex)
    const BYTES_PER_TRIANGLE_BASE: u64 = 64; // BVH node overhead
    const BYTES_PER_VERTEX: u64 = 12; // Position only

    let triangle_data = total_triangles * BYTES_PER_TRIANGLE_BASE;
    let vertex_data = total_vertices * BYTES_PER_VERTEX;
    let total_data = triangle_data + vertex_data;

    // Scratch size: ~2x data for sorting/building
    let scratch_size = total_data * 2;

    // Result size depends on flags
    let effective_flags = config.effective_flags();
    let result_size = if effective_flags.contains(BlasBuildFlags::MINIMIZE_MEMORY) {
        (total_data as f64 * 1.2) as u64
    } else if effective_flags.contains(BlasBuildFlags::PREFER_FAST_TRACE) {
        (total_data as f64 * 1.8) as u64
    } else {
        (total_data as f64 * 1.5) as u64
    };

    // Compacted size estimate
    let compacted_size = if effective_flags.contains(BlasBuildFlags::ALLOW_COMPACTION) {
        Some((result_size as f64 * 0.7) as u64)
    } else {
        None
    };

    // Update scratch (smaller than build scratch)
    let update_scratch_size = if effective_flags.contains(BlasBuildFlags::ALLOW_UPDATE) {
        scratch_size / 2
    } else {
        0
    };

    // Ensure minimum sizes
    let scratch_size = scratch_size.max(256);
    let result_size = result_size.max(256);

    BlasBuildResult {
        scratch_size,
        result_size,
        compacted_size,
        update_scratch_size,
    }
}

/// Query the actual compacted size after BLAS build.
///
/// In a real implementation, this would query the GPU driver.
/// This function simulates the query based on typical compaction ratios.
///
/// # Arguments
///
/// * `uncompacted_size` - Size of the uncompacted BLAS
/// * `triangle_count` - Number of triangles in the BLAS
///
/// # Returns
///
/// Estimated compacted size in bytes.
pub fn query_compacted_size(uncompacted_size: u64, triangle_count: u64) -> u64 {
    // Compaction ratio varies based on geometry complexity
    // Typical ratios: 0.5-0.8 depending on BVH quality vs size tradeoff
    let ratio = if triangle_count < 1000 {
        0.8 // Small meshes compact less
    } else if triangle_count < 100000 {
        0.7 // Medium meshes
    } else {
        0.6 // Large meshes compact more
    };

    ((uncompacted_size as f64) * ratio) as u64
}

/// Serialize BLAS data for storage.
///
/// # Arguments
///
/// * `blas_data` - Raw BLAS data from GPU
/// * `compacted` - Whether the data is compacted
///
/// # Returns
///
/// Serialized BLAS ready for storage.
pub fn serialize_blas(blas_data: &[u8], compacted: bool) -> SerializedBlas {
    serialize_blas_with_metadata(blas_data, compacted, BlasBuildFlags::default(), 1, 0)
}

/// Serialize BLAS data with full metadata.
///
/// # Arguments
///
/// * `blas_data` - Raw BLAS data from GPU
/// * `compacted` - Whether the data is compacted
/// * `build_flags` - Build flags used to create the BLAS
/// * `geometry_count` - Number of geometries
/// * `triangle_count` - Total triangle count
///
/// # Returns
///
/// Serialized BLAS ready for storage.
pub fn serialize_blas_with_metadata(
    blas_data: &[u8],
    compacted: bool,
    build_flags: BlasBuildFlags,
    geometry_count: u32,
    triangle_count: u64,
) -> SerializedBlas {
    let original_size = if compacted {
        // Estimate original size from compacted
        ((blas_data.len() as f64) / 0.7) as u64
    } else {
        blas_data.len() as u64
    };

    // Build header
    let header = BlasHeader {
        magic: BLAS_MAGIC,
        version: BLAS_SERIALIZATION_VERSION,
        original_size,
        compacted: if compacted { 1 } else { 0 },
        build_flags: build_flags.bits(),
        geometry_count,
        triangle_count,
        data_size: blas_data.len() as u64,
    };

    // Serialize header + data
    let mut data = Vec::with_capacity(BlasHeader::SIZE + blas_data.len());

    // Write header fields
    data.extend_from_slice(&header.magic);
    data.extend_from_slice(&header.version.to_le_bytes());
    data.extend_from_slice(&header.original_size.to_le_bytes());
    data.extend_from_slice(&header.compacted.to_le_bytes());
    data.extend_from_slice(&header.build_flags.to_le_bytes());
    data.extend_from_slice(&header.geometry_count.to_le_bytes());
    data.extend_from_slice(&header.triangle_count.to_le_bytes());
    data.extend_from_slice(&header.data_size.to_le_bytes());

    // Write BLAS data
    data.extend_from_slice(blas_data);

    SerializedBlas {
        data,
        version: BLAS_SERIALIZATION_VERSION,
        original_size,
        compacted,
        build_flags,
        geometry_count,
        triangle_count,
    }
}

/// Deserialize BLAS data from storage.
///
/// # Arguments
///
/// * `serialized` - Serialized BLAS data
///
/// # Returns
///
/// Deserialized BLAS data, or error if format is invalid.
pub fn deserialize_blas(serialized: &SerializedBlas) -> BlasResult<Vec<u8>> {
    deserialize_blas_from_bytes(&serialized.data)
}

/// Deserialize BLAS data from raw bytes.
///
/// # Arguments
///
/// * `data` - Raw serialized BLAS bytes
///
/// # Returns
///
/// Deserialized BLAS data and metadata.
pub fn deserialize_blas_from_bytes(data: &[u8]) -> BlasResult<Vec<u8>> {
    if data.len() < BlasHeader::SIZE {
        return Err(BlasError::DeserializationError(
            "data too small for header".into(),
        ));
    }

    // Read magic
    let magic: [u8; 4] = data[0..4].try_into().unwrap();
    if magic != BLAS_MAGIC {
        return Err(BlasError::DeserializationError(format!(
            "invalid magic: expected {:?}, got {:?}",
            BLAS_MAGIC, magic
        )));
    }

    // Read version
    let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
    if version != BLAS_SERIALIZATION_VERSION {
        return Err(BlasError::VersionMismatch {
            expected: BLAS_SERIALIZATION_VERSION,
            got: version,
        });
    }

    // Read data size
    let data_size_offset = 4 + 4 + 8 + 4 + 4 + 4 + 8; // magic + version + original_size + compacted + flags + geom_count + tri_count
    let data_size = u64::from_le_bytes(
        data[data_size_offset..data_size_offset + 8]
            .try_into()
            .unwrap(),
    );

    // Extract BLAS data
    let data_start = BlasHeader::SIZE;
    let data_end = data_start + data_size as usize;

    if data.len() < data_end {
        return Err(BlasError::DeserializationError(format!(
            "data truncated: expected {} bytes, got {}",
            data_end,
            data.len()
        )));
    }

    Ok(data[data_start..data_end].to_vec())
}

/// Deserialize BLAS with full metadata.
pub fn deserialize_blas_with_metadata(data: &[u8]) -> BlasResult<SerializedBlas> {
    if data.len() < BlasHeader::SIZE {
        return Err(BlasError::DeserializationError(
            "data too small for header".into(),
        ));
    }

    // Read all header fields
    let magic: [u8; 4] = data[0..4].try_into().unwrap();
    if magic != BLAS_MAGIC {
        return Err(BlasError::DeserializationError("invalid magic".into()));
    }

    let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
    if version != BLAS_SERIALIZATION_VERSION {
        return Err(BlasError::VersionMismatch {
            expected: BLAS_SERIALIZATION_VERSION,
            got: version,
        });
    }

    let original_size = u64::from_le_bytes(data[8..16].try_into().unwrap());
    let compacted = u32::from_le_bytes(data[16..20].try_into().unwrap()) != 0;
    let build_flags =
        BlasBuildFlags::from_bits_truncate(u32::from_le_bytes(data[20..24].try_into().unwrap()));
    let geometry_count = u32::from_le_bytes(data[24..28].try_into().unwrap());
    let triangle_count = u64::from_le_bytes(data[28..36].try_into().unwrap());
    let data_size = u64::from_le_bytes(data[36..44].try_into().unwrap());

    let data_end = BlasHeader::SIZE + data_size as usize;
    if data.len() < data_end {
        return Err(BlasError::DeserializationError("data truncated".into()));
    }

    Ok(SerializedBlas {
        data: data.to_vec(),
        version,
        original_size,
        compacted,
        build_flags,
        geometry_count,
        triangle_count,
    })
}

// ---------------------------------------------------------------------------
// Skinned mesh support
// ---------------------------------------------------------------------------

/// Check if a BLAS config represents a skinned mesh.
pub fn is_skinned_mesh(config: &BlasConfig) -> bool {
    config.is_skinned
}

/// Configure BLAS for skinned mesh (enables refitting).
///
/// Skinned meshes require special handling:
/// - ALLOW_UPDATE flag is required for refitting
/// - ALLOW_COMPACTION is disabled (updates require stable memory)
/// - PREFER_FAST_BUILD may be preferred over PREFER_FAST_TRACE
pub fn configure_for_skinned(mut config: BlasConfig) -> BlasConfig {
    config.is_skinned = true;
    config.flags |= BlasBuildFlags::ALLOW_UPDATE;
    config.flags.remove(BlasBuildFlags::ALLOW_COMPACTION);
    config
}

/// Update BLAS vertex data for skinned mesh refitting.
///
/// # Arguments
///
/// * `geometry` - Geometry to update
/// * `new_positions` - New vertex positions after skinning
///
/// # Returns
///
/// Updated geometry, or error if position count doesn't match.
pub fn update_skinned_positions(
    mut geometry: BlasGeometry,
    new_positions: &[[f32; 3]],
) -> BlasResult<BlasGeometry> {
    if new_positions.len() as u32 != geometry.vertex_count {
        return Err(BlasError::InvalidVertexData(format!(
            "position count mismatch: expected {}, got {}",
            geometry.vertex_count,
            new_positions.len()
        )));
    }

    // Update vertex buffer
    geometry.vertex_buffer.clear();
    geometry
        .vertex_buffer
        .reserve(new_positions.len() * geometry.vertex_stride as usize);

    for pos in new_positions {
        geometry.vertex_buffer.extend_from_slice(&pos[0].to_le_bytes());
        geometry.vertex_buffer.extend_from_slice(&pos[1].to_le_bytes());
        geometry.vertex_buffer.extend_from_slice(&pos[2].to_le_bytes());
        // Pad to stride if necessary
        let padding = geometry.vertex_stride as usize - 12;
        if padding > 0 {
            geometry.vertex_buffer.extend(std::iter::repeat(0u8).take(padding));
        }
    }

    Ok(geometry)
}

// ---------------------------------------------------------------------------
// Ray tracing decorator parsing
// ---------------------------------------------------------------------------

/// Parse `@ray_tracing` decorator parameters into BLAS configuration.
///
/// # Arguments
///
/// * `params` - Decorator parameters as key-value pairs
///
/// # Returns
///
/// Parsed decorator parameters.
///
/// # Supported Parameters
///
/// - `dynamic` (bool): Whether geometry is dynamic/skinned
/// - `fast_trace` (bool): Prefer fast trace over fast build
/// - `compaction` (bool): Enable compaction
/// - `lod` (int): LOD level (0 = highest)
/// - `opaque` (bool): Whether geometry is opaque
/// - `name` (string): Custom BLAS name
///
/// # Example
///
/// ```ignore
/// let params = [
///     ("dynamic".to_string(), DecoratorValue::Bool(true)),
///     ("lod".to_string(), DecoratorValue::Int(0)),
/// ].into_iter().collect();
///
/// let parsed = parse_ray_tracing_decorator(&params);
/// assert!(parsed.dynamic);
/// ```
pub fn parse_ray_tracing_decorator(
    params: &HashMap<String, DecoratorValue>,
) -> RayTracingDecoratorParams {
    let mut result = RayTracingDecoratorParams::default();

    if let Some(v) = params.get("dynamic").and_then(|v| v.as_bool()) {
        result.dynamic = v;
    }

    if let Some(v) = params.get("fast_trace").and_then(|v| v.as_bool()) {
        result.fast_trace = v;
    }

    if let Some(v) = params.get("compaction").and_then(|v| v.as_bool()) {
        result.compaction = v;
    }

    if let Some(v) = params.get("lod").and_then(|v| v.as_int()) {
        result.lod_level = v.max(0) as u32;
    }

    if let Some(v) = params.get("opaque").and_then(|v| v.as_bool()) {
        if v {
            result.geometry_flags |= GeometryFlags::OPAQUE;
        } else {
            result.geometry_flags.remove(GeometryFlags::OPAQUE);
        }
    }

    if let Some(v) = params.get("no_duplicate_any_hit").and_then(|v| v.as_bool()) {
        if v {
            result.geometry_flags |= GeometryFlags::NO_DUPLICATE_ANY_HIT;
        }
    }

    if let Some(v) = params.get("name").and_then(|v| v.as_string()) {
        result.name = Some(v.to_string());
    }

    result
}

/// Convert parsed decorator parameters to BLAS config.
///
/// # Arguments
///
/// * `params` - Parsed decorator parameters
///
/// # Returns
///
/// BLAS configuration based on decorator parameters.
pub fn decorator_to_blas_config(params: &RayTracingDecoratorParams) -> BlasConfig {
    let mut flags = BlasBuildFlags::empty();

    if params.compaction && !params.dynamic {
        flags |= BlasBuildFlags::ALLOW_COMPACTION;
    }

    if params.fast_trace {
        flags |= BlasBuildFlags::PREFER_FAST_TRACE;
    } else {
        flags |= BlasBuildFlags::PREFER_FAST_BUILD;
    }

    if params.dynamic {
        flags |= BlasBuildFlags::ALLOW_UPDATE;
    }

    BlasConfig {
        flags,
        geometries: Vec::new(),
        is_skinned: params.dynamic,
        name: params.name.clone(),
    }
}

// ---------------------------------------------------------------------------
// Multi-LOD support
// ---------------------------------------------------------------------------

/// Configuration for multiple LOD levels of a mesh.
#[derive(Debug, Clone)]
pub struct BlasLodConfig {
    /// LOD level (0 = highest detail).
    pub lod_level: u32,
    /// Geometries for this LOD.
    pub geometries: Vec<BlasGeometry>,
    /// Distance threshold for switching to this LOD.
    pub distance_threshold: f32,
}

/// Create BLAS configs for multiple LOD levels.
///
/// # Arguments
///
/// * `lod_configs` - Per-LOD configurations
/// * `base_flags` - Base build flags to apply to all LODs
///
/// # Returns
///
/// Vector of BLAS configs, one per LOD level.
pub fn create_multi_lod_blas(
    lod_configs: &[BlasLodConfig],
    base_flags: BlasBuildFlags,
) -> Vec<BlasConfig> {
    lod_configs
        .iter()
        .map(|lod| BlasConfig {
            flags: base_flags,
            geometries: lod.geometries.clone(),
            is_skinned: false,
            name: Some(format!("LOD_{}", lod.lod_level)),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Validate BLAS geometry data.
///
/// # Arguments
///
/// * `geometry` - Geometry to validate
///
/// # Returns
///
/// Ok if valid, or error describing the issue.
pub fn validate_geometry(geometry: &BlasGeometry) -> BlasResult<()> {
    // Check vertex buffer size
    let expected_vertex_size = geometry.vertex_count as usize * geometry.vertex_stride as usize;
    if geometry.vertex_buffer.len() < expected_vertex_size {
        return Err(BlasError::InvalidVertexData(format!(
            "vertex buffer too small: expected {} bytes, got {}",
            expected_vertex_size,
            geometry.vertex_buffer.len()
        )));
    }

    // Check index buffer if present
    if let Some(ref idx_buf) = geometry.index_buffer {
        let index_size = match geometry.index_type {
            IndexType::U8 => 1,
            IndexType::U16 => 2,
            IndexType::U32 => 4,
        };

        if idx_buf.len() % index_size != 0 {
            return Err(BlasError::InvalidIndexData(format!(
                "index buffer size {} not aligned to index size {}",
                idx_buf.len(),
                index_size
            )));
        }

        let index_count = idx_buf.len() / index_size;
        if index_count % 3 != 0 {
            return Err(BlasError::InvalidIndexData(format!(
                "index count {} not divisible by 3 (not triangle list)",
                index_count
            )));
        }

        // Validate indices are in bounds
        for i in 0..(idx_buf.len() / index_size) {
            let index = match geometry.index_type {
                IndexType::U8 => idx_buf[i] as u32,
                IndexType::U16 => {
                    u16::from_le_bytes([idx_buf[i * 2], idx_buf[i * 2 + 1]]) as u32
                }
                IndexType::U32 => u32::from_le_bytes([
                    idx_buf[i * 4],
                    idx_buf[i * 4 + 1],
                    idx_buf[i * 4 + 2],
                    idx_buf[i * 4 + 3],
                ]),
            };

            if index >= geometry.vertex_count {
                return Err(BlasError::InvalidIndexData(format!(
                    "index {} out of bounds (vertex count: {})",
                    index, geometry.vertex_count
                )));
            }
        }
    } else {
        // Non-indexed geometry must have triangle-aligned vertex count
        if geometry.vertex_count % 3 != 0 {
            return Err(BlasError::InvalidVertexData(format!(
                "non-indexed vertex count {} not divisible by 3",
                geometry.vertex_count
            )));
        }
    }

    Ok(())
}

/// Validate BLAS config.
pub fn validate_config(config: &BlasConfig) -> BlasResult<()> {
    if config.geometries.is_empty() {
        return Err(BlasError::UnsupportedConfig("no geometries".into()));
    }

    for (i, geom) in config.geometries.iter().enumerate() {
        validate_geometry(geom).map_err(|e| {
            BlasError::InvalidVertexData(format!("geometry {}: {}", i, e))
        })?;
    }

    // Check for incompatible flags
    let flags = config.effective_flags();
    if flags.contains(BlasBuildFlags::PREFER_FAST_TRACE)
        && flags.contains(BlasBuildFlags::PREFER_FAST_BUILD)
    {
        return Err(BlasError::UnsupportedConfig(
            "cannot use both PREFER_FAST_TRACE and PREFER_FAST_BUILD".into(),
        ));
    }

    Ok(())
}

/// Calculate memory savings from compaction.
///
/// # Arguments
///
/// * `original_size` - Original BLAS size
/// * `compacted_size` - Compacted BLAS size
///
/// # Returns
///
/// Tuple of (bytes saved, percentage saved).
pub fn calculate_compaction_savings(original_size: u64, compacted_size: u64) -> (u64, f32) {
    let saved = original_size.saturating_sub(compacted_size);
    let percentage = if original_size > 0 {
        (saved as f64 / original_size as f64 * 100.0) as f32
    } else {
        0.0
    };
    (saved, percentage)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Build input preparation tests (4 tests)
    // ========================================================================

    #[test]
    fn test_prepare_blas_input_basic() {
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let geometry = prepare_blas_input(&positions, None, None);

        assert_eq!(geometry.vertex_count, 3);
        assert_eq!(geometry.vertex_stride, 12);
        assert_eq!(geometry.vertex_buffer.len(), 36); // 3 vertices * 12 bytes
        assert!(geometry.index_buffer.is_none());
    }

    #[test]
    fn test_prepare_blas_input_indexed() {
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        let indices = vec![0, 1, 2, 1, 3, 2];
        let geometry = prepare_blas_input(&positions, Some(&indices), None);

        assert_eq!(geometry.vertex_count, 4);
        assert!(geometry.index_buffer.is_some());
        // Should use U16 for small meshes
        assert_eq!(geometry.index_type, IndexType::U16);
        assert_eq!(geometry.index_buffer.as_ref().unwrap().len(), 12); // 6 indices * 2 bytes
    }

    #[test]
    fn test_prepare_blas_input_with_transform() {
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let transform: [f32; 12] = [
            1.0, 0.0, 0.0, 0.0, // Row 0
            0.0, 1.0, 0.0, 0.0, // Row 1
            0.0, 0.0, 1.0, 0.0, // Row 2 (no translation)
        ];
        let geometry = prepare_blas_input(&positions, None, Some(&transform));

        assert!(geometry.transform.is_some());
        assert_eq!(geometry.transform.unwrap()[0], 1.0);
    }

    #[test]
    fn test_prepare_blas_input_large_index() {
        // Create mesh large enough to require U32 indices
        let positions: Vec<[f32; 3]> = (0..70000).map(|i| [i as f32, 0.0, 0.0]).collect();
        let indices: Vec<u32> = (0..69999).collect();
        let geometry = prepare_blas_input(&positions, Some(&indices), None);

        assert_eq!(geometry.index_type, IndexType::U32);
    }

    // ========================================================================
    // Flags configuration tests (4 tests)
    // ========================================================================

    #[test]
    fn test_default_flags() {
        let flags = BlasBuildFlags::default();
        assert!(flags.contains(BlasBuildFlags::ALLOW_COMPACTION));
        assert!(flags.contains(BlasBuildFlags::PREFER_FAST_TRACE));
        assert!(!flags.contains(BlasBuildFlags::ALLOW_UPDATE));
    }

    #[test]
    fn test_flags_combination() {
        let flags =
            BlasBuildFlags::ALLOW_COMPACTION | BlasBuildFlags::PREFER_FAST_TRACE | BlasBuildFlags::MINIMIZE_MEMORY;

        assert!(flags.contains(BlasBuildFlags::ALLOW_COMPACTION));
        assert!(flags.contains(BlasBuildFlags::PREFER_FAST_TRACE));
        assert!(flags.contains(BlasBuildFlags::MINIMIZE_MEMORY));
        assert!(!flags.contains(BlasBuildFlags::ALLOW_UPDATE));
    }

    #[test]
    fn test_effective_flags_skinned() {
        let config = BlasConfig {
            flags: BlasBuildFlags::ALLOW_COMPACTION | BlasBuildFlags::PREFER_FAST_TRACE,
            geometries: Vec::new(),
            is_skinned: true,
            name: None,
        };

        let effective = config.effective_flags();
        assert!(effective.contains(BlasBuildFlags::ALLOW_UPDATE));
        assert!(!effective.contains(BlasBuildFlags::ALLOW_COMPACTION)); // Removed for skinned
    }

    #[test]
    fn test_flags_removal() {
        let mut flags = BlasBuildFlags::ALLOW_COMPACTION | BlasBuildFlags::PREFER_FAST_TRACE;
        flags.remove(BlasBuildFlags::ALLOW_COMPACTION);

        assert!(!flags.contains(BlasBuildFlags::ALLOW_COMPACTION));
        assert!(flags.contains(BlasBuildFlags::PREFER_FAST_TRACE));
    }

    // ========================================================================
    // Size computation tests (3 tests)
    // ========================================================================

    #[test]
    fn test_compute_sizes_empty() {
        let config = BlasConfig::default();
        let sizes = compute_blas_sizes(&config);

        // Should return minimum sizes
        assert!(sizes.scratch_size >= 256);
        assert!(sizes.result_size >= 256);
    }

    #[test]
    fn test_compute_sizes_basic() {
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let geometry = prepare_blas_input(&positions, None, None);

        let config = BlasConfig {
            flags: BlasBuildFlags::default(),
            geometries: vec![geometry],
            is_skinned: false,
            name: None,
        };

        let sizes = compute_blas_sizes(&config);

        assert!(sizes.scratch_size > 0);
        assert!(sizes.result_size > 0);
        assert!(sizes.compacted_size.is_some()); // ALLOW_COMPACTION is default
        assert!(sizes.compacted_size.unwrap() < sizes.result_size);
    }

    #[test]
    fn test_compute_sizes_skinned() {
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let geometry = prepare_blas_input(&positions, None, None);

        let config = BlasConfig {
            flags: BlasBuildFlags::default(),
            geometries: vec![geometry],
            is_skinned: true,
            name: None,
        };

        let sizes = compute_blas_sizes(&config);

        assert!(sizes.update_scratch_size > 0); // ALLOW_UPDATE enabled
        assert!(sizes.compacted_size.is_none()); // No compaction for skinned
    }

    // ========================================================================
    // Serialization round-trip tests (4 tests)
    // ========================================================================

    #[test]
    fn test_serialize_deserialize_basic() {
        let blas_data = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x12, 0x34, 0x56, 0x78];
        let serialized = serialize_blas(&blas_data, false);

        let deserialized = deserialize_blas(&serialized).unwrap();
        assert_eq!(deserialized, blas_data);
    }

    #[test]
    fn test_serialize_deserialize_compacted() {
        let blas_data = vec![0x11, 0x22, 0x33, 0x44];
        let serialized = serialize_blas(&blas_data, true);

        assert!(serialized.compacted);
        let deserialized = deserialize_blas(&serialized).unwrap();
        assert_eq!(deserialized, blas_data);
    }

    #[test]
    fn test_serialize_with_metadata() {
        let blas_data = vec![0xAA, 0xBB, 0xCC, 0xDD];
        let flags = BlasBuildFlags::ALLOW_COMPACTION | BlasBuildFlags::PREFER_FAST_TRACE;
        let serialized = serialize_blas_with_metadata(&blas_data, false, flags, 3, 100);

        assert_eq!(serialized.geometry_count, 3);
        assert_eq!(serialized.triangle_count, 100);
        assert_eq!(serialized.build_flags, flags);

        // Deserialize with metadata
        let restored = deserialize_blas_with_metadata(&serialized.data).unwrap();
        assert_eq!(restored.geometry_count, 3);
        assert_eq!(restored.triangle_count, 100);
    }

    #[test]
    fn test_deserialize_version_mismatch() {
        // Create a full header with wrong version (44 bytes minimum)
        let mut data = Vec::new();
        data.extend_from_slice(&BLAS_MAGIC);            // 4 bytes
        data.extend_from_slice(&999u32.to_le_bytes());  // 4 bytes - Invalid version
        data.extend_from_slice(&0u64.to_le_bytes());    // 8 bytes - original_size
        data.extend_from_slice(&0u32.to_le_bytes());    // 4 bytes - compacted
        data.extend_from_slice(&0u32.to_le_bytes());    // 4 bytes - build_flags
        data.extend_from_slice(&0u32.to_le_bytes());    // 4 bytes - geometry_count
        data.extend_from_slice(&0u64.to_le_bytes());    // 8 bytes - triangle_count
        data.extend_from_slice(&0u64.to_le_bytes());    // 8 bytes - data_size

        let result = deserialize_blas_from_bytes(&data);
        assert!(matches!(result, Err(BlasError::VersionMismatch { .. })));
    }

    // ========================================================================
    // Skinned mesh handling tests (3 tests)
    // ========================================================================

    #[test]
    fn test_configure_for_skinned() {
        let config = BlasConfig::default();
        let skinned_config = configure_for_skinned(config);

        assert!(skinned_config.is_skinned);
        assert!(skinned_config.flags.contains(BlasBuildFlags::ALLOW_UPDATE));
        assert!(!skinned_config.flags.contains(BlasBuildFlags::ALLOW_COMPACTION));
    }

    #[test]
    fn test_update_skinned_positions() {
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let geometry = prepare_blas_input(&positions, None, None);

        let new_positions = vec![[0.0, 0.0, 0.1], [1.0, 0.0, 0.1], [0.0, 1.0, 0.1]];
        let updated = update_skinned_positions(geometry, &new_positions).unwrap();

        // Verify the vertex buffer was updated
        let z_offset = 8; // Offset to Z component
        let z_bytes: [u8; 4] = updated.vertex_buffer[z_offset..z_offset + 4]
            .try_into()
            .unwrap();
        let z = f32::from_le_bytes(z_bytes);
        assert!((z - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_update_skinned_positions_mismatch() {
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let geometry = prepare_blas_input(&positions, None, None);

        let new_positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]; // Wrong count
        let result = update_skinned_positions(geometry, &new_positions);

        assert!(matches!(result, Err(BlasError::InvalidVertexData(_))));
    }

    // ========================================================================
    // Decorator parsing tests (2 tests)
    // ========================================================================

    #[test]
    fn test_parse_ray_tracing_decorator_default() {
        let params: HashMap<String, DecoratorValue> = HashMap::new();
        let parsed = parse_ray_tracing_decorator(&params);

        assert!(!parsed.dynamic);
        assert!(parsed.fast_trace);
        assert!(parsed.compaction);
        assert_eq!(parsed.lod_level, 0);
    }

    #[test]
    fn test_parse_ray_tracing_decorator_full() {
        let mut params = HashMap::new();
        params.insert("dynamic".to_string(), DecoratorValue::Bool(true));
        params.insert("fast_trace".to_string(), DecoratorValue::Bool(false));
        params.insert("compaction".to_string(), DecoratorValue::Bool(false));
        params.insert("lod".to_string(), DecoratorValue::Int(2));
        params.insert("opaque".to_string(), DecoratorValue::Bool(false));
        params.insert("name".to_string(), DecoratorValue::String("MyBLAS".to_string()));

        let parsed = parse_ray_tracing_decorator(&params);

        assert!(parsed.dynamic);
        assert!(!parsed.fast_trace);
        assert!(!parsed.compaction);
        assert_eq!(parsed.lod_level, 2);
        assert!(!parsed.geometry_flags.contains(GeometryFlags::OPAQUE));
        assert_eq!(parsed.name, Some("MyBLAS".to_string()));
    }

    // ========================================================================
    // Validation tests (4 tests)
    // ========================================================================

    #[test]
    fn test_validate_geometry_valid() {
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let geometry = prepare_blas_input(&positions, None, None);

        assert!(validate_geometry(&geometry).is_ok());
    }

    #[test]
    fn test_validate_geometry_invalid_vertex_count() {
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]; // Not divisible by 3
        let geometry = prepare_blas_input(&positions, None, None);

        assert!(matches!(
            validate_geometry(&geometry),
            Err(BlasError::InvalidVertexData(_))
        ));
    }

    #[test]
    fn test_validate_geometry_index_out_of_bounds() {
        let mut geometry = BlasGeometry {
            vertex_buffer: vec![0u8; 36], // 3 vertices
            vertex_stride: 12,
            vertex_count: 3,
            index_buffer: Some(vec![0, 0, 5, 0, 0, 0]), // Index 5 out of bounds (U16)
            index_type: IndexType::U16,
            transform: None,
            geometry_flags: GeometryFlags::OPAQUE,
        };

        assert!(matches!(
            validate_geometry(&geometry),
            Err(BlasError::InvalidIndexData(_))
        ));
    }

    #[test]
    fn test_validate_config_empty() {
        let config = BlasConfig::default();
        assert!(matches!(
            validate_config(&config),
            Err(BlasError::UnsupportedConfig(_))
        ));
    }

    // ========================================================================
    // Additional tests (6 more to reach 20+)
    // ========================================================================

    #[test]
    fn test_geometry_flags() {
        let flags = GeometryFlags::OPAQUE | GeometryFlags::NO_DUPLICATE_ANY_HIT;
        assert!(flags.contains(GeometryFlags::OPAQUE));
        assert!(flags.contains(GeometryFlags::NO_DUPLICATE_ANY_HIT));
    }

    #[test]
    fn test_blas_config_builder() {
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let geometry = prepare_blas_input(&positions, None, None);

        let config = BlasConfig::new()
            .with_flags(BlasBuildFlags::ALLOW_COMPACTION)
            .with_skinned(false)
            .with_name("TestBLAS");

        assert_eq!(config.name, Some("TestBLAS".to_string()));
        assert!(!config.is_skinned);
    }

    #[test]
    fn test_total_counts() {
        let positions1 = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let positions2 = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.5, 0.5, 1.0],
            [0.5, 1.5, 0.0],
        ];

        let geom1 = prepare_blas_input(&positions1, None, None);
        let geom2 = prepare_blas_input(&positions2, None, None);

        let config = BlasConfig {
            flags: BlasBuildFlags::default(),
            geometries: vec![geom1, geom2],
            is_skinned: false,
            name: None,
        };

        assert_eq!(config.total_vertex_count(), 9);
        assert_eq!(config.total_triangle_count(), 3); // 3 + 6/3 = 3 triangles
    }

    #[test]
    fn test_compaction_savings() {
        let (saved, percentage) = calculate_compaction_savings(1000, 700);
        assert_eq!(saved, 300);
        assert!((percentage - 30.0).abs() < 0.1);
    }

    #[test]
    fn test_query_compacted_size() {
        let size = query_compacted_size(1000000, 50000);
        assert!(size < 1000000);
        assert!(size > 500000); // Should be ~70% for medium meshes
    }

    #[test]
    fn test_decorator_to_blas_config() {
        let params = RayTracingDecoratorParams {
            dynamic: true,
            fast_trace: false,
            compaction: false,
            lod_level: 1,
            geometry_flags: GeometryFlags::empty(),
            name: Some("DynamicMesh".to_string()),
        };

        let config = decorator_to_blas_config(&params);

        assert!(config.is_skinned);
        assert!(config.flags.contains(BlasBuildFlags::ALLOW_UPDATE));
        assert!(config.flags.contains(BlasBuildFlags::PREFER_FAST_BUILD));
        assert!(!config.flags.contains(BlasBuildFlags::ALLOW_COMPACTION));
        assert_eq!(config.name, Some("DynamicMesh".to_string()));
    }

    #[test]
    fn test_multi_lod_blas() {
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];

        let lod_configs = vec![
            BlasLodConfig {
                lod_level: 0,
                geometries: vec![prepare_blas_input(&positions, None, None)],
                distance_threshold: 0.0,
            },
            BlasLodConfig {
                lod_level: 1,
                geometries: vec![prepare_blas_input(&positions, None, None)],
                distance_threshold: 100.0,
            },
        ];

        let configs = create_multi_lod_blas(&lod_configs, BlasBuildFlags::default());

        assert_eq!(configs.len(), 2);
        assert_eq!(configs[0].name, Some("LOD_0".to_string()));
        assert_eq!(configs[1].name, Some("LOD_1".to_string()));
    }

    #[test]
    fn test_prepare_blas_input_raw() {
        let vertex_buffer: Vec<u8> = vec![
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // Vertex 0
            0, 0, 128, 63, 0, 0, 0, 0, 0, 0, 0, 0, // Vertex 1 (1.0, 0.0, 0.0)
            0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 0, 0, // Vertex 2 (0.0, 1.0, 0.0)
        ];

        let geometry = prepare_blas_input_raw(vertex_buffer.clone(), 12, 3, None, IndexType::U32);

        assert_eq!(geometry.vertex_count, 3);
        assert_eq!(geometry.vertex_stride, 12);
        assert_eq!(geometry.vertex_buffer, vertex_buffer);
    }

    #[test]
    fn test_is_skinned_mesh() {
        let static_config = BlasConfig::default();
        let skinned_config = BlasConfig {
            is_skinned: true,
            ..Default::default()
        };

        assert!(!is_skinned_mesh(&static_config));
        assert!(is_skinned_mesh(&skinned_config));
    }

    #[test]
    fn test_decorator_value_conversions() {
        let bool_val = DecoratorValue::Bool(true);
        let int_val = DecoratorValue::Int(42);
        let float_val = DecoratorValue::Float(3.14);
        let string_val = DecoratorValue::String("test".to_string());

        assert_eq!(bool_val.as_bool(), Some(true));
        assert_eq!(int_val.as_int(), Some(42));
        assert_eq!(float_val.as_float(), Some(3.14));
        assert_eq!(string_val.as_string(), Some("test"));

        // Cross-type conversions
        assert_eq!(int_val.as_float(), Some(42.0));
        assert_eq!(float_val.as_int(), None);
    }
}
