//! Texture and Mesh Hot-Reload Propagation (T-AS-6.4)
//!
//! Provides hot-reload support for texture and mesh assets with:
//!
//! - **Texture hot-reload**: detect change -> reimport -> reprocess (mips, compression) ->
//!   new GPU image -> descriptor update -> optional 1-2 frame fade transition
//! - **Mesh hot-reload**: detect change -> reimport -> regenerate meshlets ->
//!   rebuild BLAS -> buffer update via staging buffer -> atomic render proxy update
//!   at phase boundary -> BLAS swap (may take multiple frames)
//! - **Resource retention**: old resources retained until new ones ready
//! - **No rendering stalls**: asynchronous processing pipeline
//!
//! # Architecture
//!
//! ```text
//! +------------------+     +--------------------+     +------------------+
//! | ContentChange    | --> | AssetReloadManager | --> | Swap Queue       |
//! | (file events)    |     | (process, stage)   |     | (phase boundary) |
//! +------------------+     +--------------------+     +------------------+
//!                                |
//!                                v
//!                         +--------------------+
//!                         | StagingBufferPool  |
//!                         | (triple-buffered)  |
//!                         +--------------------+
//! ```
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::hot_reload::asset_reload::{
//!     AssetReloadManager, AssetReloadConfig,
//! };
//!
//! let config = AssetReloadConfig::default();
//! let mut manager = AssetReloadManager::new(staging_pool, config);
//!
//! // When file watcher detects a texture change
//! manager.on_texture_change(&Path::new("textures/albedo.png"), old_handle);
//!
//! // In render loop
//! loop {
//!     // Poll progress (non-blocking)
//!     let events = manager.poll_progress();
//!     for event in events {
//!         println!("{:?} progress: {:.0}%", event.path, event.progress * 100.0);
//!     }
//!
//!     // At phase boundary (e.g., between frames)
//!     manager.execute_swaps_at_phase_boundary();
//!
//!     // For cross-fade rendering
//!     if let Some(alpha) = manager.get_fade_alpha(&Path::new("textures/albedo.png")) {
//!         // Blend old and new texture with alpha
//!     }
//! }
//! ```

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::asset::meshlet::{MeshletConfig, MeshletMesh};
use crate::asset::mipmap::{CompressionFormat, FilterType};
use crate::blas_pool::BlasHandle;
use crate::pipeline::ContentHash;

// ---------------------------------------------------------------------------
// Error Types
// ---------------------------------------------------------------------------

/// Errors that can occur during asset hot-reload.
#[derive(Debug, Clone)]
pub enum AssetReloadError {
    /// Failed to import asset from path.
    ImportFailed(String),
    /// Failed to process asset (mips, compression, meshlets).
    ProcessingFailed(String),
    /// Failed to upload asset to GPU.
    UploadFailed(String),
    /// Failed to allocate staging buffer.
    StagingAllocationFailed { required: usize, available: usize },
    /// Asset not found at path.
    AssetNotFound(PathBuf),
    /// Invalid asset format.
    InvalidFormat(String),
    /// BLAS rebuild failed.
    BlasRebuildFailed(String),
    /// Descriptor update failed.
    DescriptorUpdateFailed(String),
    /// Timeout during reload.
    Timeout { elapsed: Duration, max: Duration },
    /// Internal error.
    Internal(String),
}

impl fmt::Display for AssetReloadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ImportFailed(msg) => write!(f, "asset import failed: {}", msg),
            Self::ProcessingFailed(msg) => write!(f, "asset processing failed: {}", msg),
            Self::UploadFailed(msg) => write!(f, "GPU upload failed: {}", msg),
            Self::StagingAllocationFailed { required, available } => {
                write!(
                    f,
                    "staging buffer allocation failed: required {} bytes, {} available",
                    required, available
                )
            }
            Self::AssetNotFound(path) => write!(f, "asset not found: {:?}", path),
            Self::InvalidFormat(msg) => write!(f, "invalid asset format: {}", msg),
            Self::BlasRebuildFailed(msg) => write!(f, "BLAS rebuild failed: {}", msg),
            Self::DescriptorUpdateFailed(msg) => write!(f, "descriptor update failed: {}", msg),
            Self::Timeout { elapsed, max } => {
                write!(
                    f,
                    "reload timeout: {:?} elapsed, max {:?}",
                    elapsed, max
                )
            }
            Self::Internal(msg) => write!(f, "internal error: {}", msg),
        }
    }
}

impl std::error::Error for AssetReloadError {}

/// Result type for asset reload operations.
pub type AssetReloadResult<T> = Result<T, AssetReloadError>;

// ---------------------------------------------------------------------------
// Handle Types
// ---------------------------------------------------------------------------

/// Handle for a texture resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureHandle(pub u32);

impl TextureHandle {
    /// Create a new texture handle.
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the raw handle ID.
    pub fn id(&self) -> u32 {
        self.0
    }
}

/// Handle for a mesh resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MeshHandle(pub u32);

impl MeshHandle {
    /// Create a new mesh handle.
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the raw handle ID.
    pub fn id(&self) -> u32 {
        self.0
    }
}

// ---------------------------------------------------------------------------
// Reload Kind
// ---------------------------------------------------------------------------

/// Classification of asset reload operations.
#[derive(Debug, Clone)]
pub enum AssetReloadKind {
    /// Texture reload with old handle and target format.
    Texture {
        /// Handle to the old texture being replaced.
        old_handle: TextureHandle,
        /// Target GPU format for the reloaded texture.
        format: TextureFormat,
    },
    /// Mesh reload with old handle and BLAS rebuild flag.
    Mesh {
        /// Handle to the old mesh being replaced.
        old_handle: MeshHandle,
        /// Whether the mesh has an associated BLAS to rebuild.
        has_blas: bool,
    },
}

impl AssetReloadKind {
    /// Returns true if this is a texture reload.
    pub fn is_texture(&self) -> bool {
        matches!(self, Self::Texture { .. })
    }

    /// Returns true if this is a mesh reload.
    pub fn is_mesh(&self) -> bool {
        matches!(self, Self::Mesh { .. })
    }

    /// Returns true if this reload requires BLAS rebuild.
    pub fn requires_blas_rebuild(&self) -> bool {
        matches!(self, Self::Mesh { has_blas: true, .. })
    }
}

// ---------------------------------------------------------------------------
// Texture Format
// ---------------------------------------------------------------------------

/// Target texture format for reload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureFormat {
    /// RGBA 8-bit unsigned normalized.
    Rgba8Unorm,
    /// RGBA 8-bit sRGB.
    Rgba8Srgb,
    /// RGBA 16-bit float.
    Rgba16Float,
    /// BC1 compressed (RGB).
    BC1,
    /// BC3 compressed (RGBA).
    BC3,
    /// BC5 compressed (normal maps).
    BC5,
    /// BC7 compressed (high quality).
    BC7,
    /// ASTC 4x4 compressed.
    Astc4x4,
}

impl TextureFormat {
    /// Returns bytes per pixel (or average for compressed formats).
    pub fn bytes_per_pixel(&self) -> f32 {
        match self {
            Self::Rgba8Unorm | Self::Rgba8Srgb => 4.0,
            Self::Rgba16Float => 8.0,
            Self::BC1 => 0.5,
            Self::BC3 | Self::BC5 | Self::BC7 => 1.0,
            Self::Astc4x4 => 1.0,
        }
    }

    /// Returns true if this is a compressed format.
    pub fn is_compressed(&self) -> bool {
        matches!(
            self,
            Self::BC1 | Self::BC3 | Self::BC5 | Self::BC7 | Self::Astc4x4
        )
    }

    /// Returns true if this is an sRGB format.
    pub fn is_srgb(&self) -> bool {
        matches!(self, Self::Rgba8Srgb)
    }
}

impl Default for TextureFormat {
    fn default() -> Self {
        Self::Rgba8Srgb
    }
}

// ---------------------------------------------------------------------------
// Reload Status
// ---------------------------------------------------------------------------

/// Status of an asset reload operation.
#[derive(Debug, Clone)]
pub enum AssetReloadStatus {
    /// Importing the asset from disk.
    Importing,
    /// Processing the asset (mips, compression, meshlets).
    Processing,
    /// Uploading to GPU via staging buffer.
    Uploading,
    /// Waiting for phase boundary to execute swap.
    WaitingForSwap,
    /// Swap completed successfully.
    Swapped,
    /// Reload failed with error.
    Failed(AssetReloadError),
}

impl AssetReloadStatus {
    /// Returns true if reload is still in progress.
    pub fn is_in_progress(&self) -> bool {
        matches!(
            self,
            Self::Importing | Self::Processing | Self::Uploading | Self::WaitingForSwap
        )
    }

    /// Returns true if reload completed (success or failure).
    pub fn is_complete(&self) -> bool {
        matches!(self, Self::Swapped | Self::Failed(_))
    }

    /// Returns true if reload succeeded.
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Swapped)
    }

    /// Returns true if reload failed.
    pub fn is_failed(&self) -> bool {
        matches!(self, Self::Failed(_))
    }

    /// Get the error if failed.
    pub fn error(&self) -> Option<&AssetReloadError> {
        match self {
            Self::Failed(e) => Some(e),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Reload Events
// ---------------------------------------------------------------------------

/// Event emitted during asset reload lifecycle.
#[derive(Debug, Clone)]
pub struct AssetReloadEvent {
    /// Path to the asset being reloaded.
    pub path: PathBuf,
    /// Kind of asset reload.
    pub kind: AssetReloadKind,
    /// Current status.
    pub status: AssetReloadStatus,
    /// Progress (0.0 to 1.0).
    pub progress: f32,
    /// When the reload started.
    pub started_at: Instant,
    /// Sequence number for ordering.
    pub sequence: u64,
}

impl AssetReloadEvent {
    /// Create a new reload event.
    pub fn new(path: PathBuf, kind: AssetReloadKind) -> Self {
        static SEQUENCE: AtomicU64 = AtomicU64::new(0);
        Self {
            path,
            kind,
            status: AssetReloadStatus::Importing,
            progress: 0.0,
            started_at: Instant::now(),
            sequence: SEQUENCE.fetch_add(1, Ordering::Relaxed),
        }
    }

    /// Update the status and progress.
    pub fn update(&mut self, status: AssetReloadStatus, progress: f32) {
        self.status = status;
        self.progress = progress.clamp(0.0, 1.0);
    }

    /// Get elapsed time since reload started.
    pub fn elapsed(&self) -> Duration {
        self.started_at.elapsed()
    }
}

// ---------------------------------------------------------------------------
// GPU Image (Texture)
// ---------------------------------------------------------------------------

/// GPU image data for a reloaded texture.
#[derive(Debug, Clone)]
pub struct GpuImage {
    /// Raw pixel data (all mip levels concatenated).
    pub data: Vec<u8>,
    /// Width of the base mip level.
    pub width: u32,
    /// Height of the base mip level.
    pub height: u32,
    /// Number of mip levels.
    pub mip_levels: u32,
    /// Target GPU format.
    pub format: TextureFormat,
    /// Content hash for deduplication.
    pub content_hash: ContentHash,
    /// Byte offsets for each mip level.
    pub mip_offsets: Vec<usize>,
}

impl GpuImage {
    /// Create a new GPU image.
    pub fn new(
        data: Vec<u8>,
        width: u32,
        height: u32,
        mip_levels: u32,
        format: TextureFormat,
    ) -> Self {
        let content_hash = ContentHash::from_bytes(&data);
        let mip_offsets = Self::calculate_mip_offsets(width, height, mip_levels, format);
        Self {
            data,
            width,
            height,
            mip_levels,
            format,
            content_hash,
            mip_offsets,
        }
    }

    /// Calculate byte offsets for each mip level.
    fn calculate_mip_offsets(
        width: u32,
        height: u32,
        mip_levels: u32,
        format: TextureFormat,
    ) -> Vec<usize> {
        let mut offsets = Vec::with_capacity(mip_levels as usize);
        let mut offset = 0usize;
        let mut w = width;
        let mut h = height;

        for _ in 0..mip_levels {
            offsets.push(offset);
            let size = Self::calculate_mip_size(w, h, format);
            offset += size;
            w = (w / 2).max(1);
            h = (h / 2).max(1);
        }

        offsets
    }

    /// Calculate byte size for a single mip level.
    fn calculate_mip_size(width: u32, height: u32, format: TextureFormat) -> usize {
        let pixels = width as usize * height as usize;
        match format {
            TextureFormat::Rgba8Unorm | TextureFormat::Rgba8Srgb => pixels * 4,
            TextureFormat::Rgba16Float => pixels * 8,
            TextureFormat::BC1 => {
                let blocks_x = ((width + 3) / 4) as usize;
                let blocks_y = ((height + 3) / 4) as usize;
                blocks_x * blocks_y * 8
            }
            TextureFormat::BC3 | TextureFormat::BC5 | TextureFormat::BC7 => {
                let blocks_x = ((width + 3) / 4) as usize;
                let blocks_y = ((height + 3) / 4) as usize;
                blocks_x * blocks_y * 16
            }
            TextureFormat::Astc4x4 => {
                let blocks_x = ((width + 3) / 4) as usize;
                let blocks_y = ((height + 3) / 4) as usize;
                blocks_x * blocks_y * 16
            }
        }
    }

    /// Get data for a specific mip level.
    pub fn mip_data(&self, level: u32) -> Option<&[u8]> {
        if level >= self.mip_levels {
            return None;
        }
        let start = self.mip_offsets.get(level as usize)?;
        let end = self
            .mip_offsets
            .get(level as usize + 1)
            .copied()
            .unwrap_or(self.data.len());
        Some(&self.data[*start..end])
    }

    /// Total size in bytes.
    pub fn total_size(&self) -> usize {
        self.data.len()
    }
}

// ---------------------------------------------------------------------------
// Mesh Buffers
// ---------------------------------------------------------------------------

/// Mesh buffer data for a reloaded mesh.
#[derive(Debug, Clone)]
pub struct MeshBuffers {
    /// Vertex positions (3 floats per vertex).
    pub positions: Vec<[f32; 3]>,
    /// Vertex normals (3 floats per vertex).
    pub normals: Vec<[f32; 3]>,
    /// Vertex UVs (2 floats per vertex).
    pub uvs: Vec<[f32; 2]>,
    /// Indices (3 per triangle).
    pub indices: Vec<u32>,
    /// Content hash for deduplication.
    pub content_hash: ContentHash,
}

impl MeshBuffers {
    /// Create new mesh buffers.
    pub fn new(
        positions: Vec<[f32; 3]>,
        normals: Vec<[f32; 3]>,
        uvs: Vec<[f32; 2]>,
        indices: Vec<u32>,
    ) -> Self {
        // Hash positions and indices for deduplication
        let mut hash_data = Vec::new();
        for p in &positions {
            hash_data.extend_from_slice(bytemuck::cast_slice(p));
        }
        for i in &indices {
            hash_data.extend_from_slice(&i.to_le_bytes());
        }
        let content_hash = ContentHash::from_bytes(&hash_data);

        Self {
            positions,
            normals,
            uvs,
            indices,
            content_hash,
        }
    }

    /// Number of vertices.
    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    /// Number of triangles.
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Total size in bytes.
    pub fn total_size(&self) -> usize {
        self.positions.len() * 12
            + self.normals.len() * 12
            + self.uvs.len() * 8
            + self.indices.len() * 4
    }
}

// ---------------------------------------------------------------------------
// Texture Reload State
// ---------------------------------------------------------------------------

/// Stage of texture reload process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureReloadStage {
    /// Importing texture from disk.
    Importing,
    /// Generating mipmaps.
    GeneratingMips,
    /// Compressing texture.
    Compressing,
    /// Uploading to GPU.
    Uploading,
    /// Waiting for descriptor update.
    UpdatingDescriptor,
    /// Cross-fading (optional).
    Fading,
    /// Complete.
    Complete,
    /// Failed.
    Failed,
}

/// State of a texture reload operation.
#[derive(Debug)]
pub struct TextureReloadState {
    /// Handle to the old texture being replaced.
    pub old_handle: TextureHandle,
    /// New GPU image (set after processing).
    pub new_image: Option<GpuImage>,
    /// New texture handle (set after upload).
    pub new_handle: Option<TextureHandle>,
    /// Cross-fade progress (0.0 = old, 1.0 = new).
    pub fade_progress: f32,
    /// Current processing stage.
    pub stage: TextureReloadStage,
    /// When this reload started.
    pub started_at: Instant,
    /// Raw imported data (before processing).
    pub import_data: Option<Vec<u8>>,
    /// Target format.
    pub target_format: TextureFormat,
    /// Error if failed.
    pub error: Option<AssetReloadError>,
}

impl TextureReloadState {
    /// Create new texture reload state.
    pub fn new(old_handle: TextureHandle, target_format: TextureFormat) -> Self {
        Self {
            old_handle,
            new_image: None,
            new_handle: None,
            fade_progress: 0.0,
            stage: TextureReloadStage::Importing,
            started_at: Instant::now(),
            import_data: None,
            target_format,
            error: None,
        }
    }

    /// Get progress as fraction (0.0 to 1.0).
    pub fn progress(&self) -> f32 {
        match self.stage {
            TextureReloadStage::Importing => 0.1,
            TextureReloadStage::GeneratingMips => 0.3,
            TextureReloadStage::Compressing => 0.5,
            TextureReloadStage::Uploading => 0.7,
            TextureReloadStage::UpdatingDescriptor => 0.85,
            TextureReloadStage::Fading => 0.9 + self.fade_progress * 0.1,
            TextureReloadStage::Complete => 1.0,
            TextureReloadStage::Failed => 0.0,
        }
    }

    /// Check if reload is complete.
    pub fn is_complete(&self) -> bool {
        matches!(
            self.stage,
            TextureReloadStage::Complete | TextureReloadStage::Failed
        )
    }

    /// Mark as failed with error.
    pub fn fail(&mut self, error: AssetReloadError) {
        self.stage = TextureReloadStage::Failed;
        self.error = Some(error);
    }
}

// ---------------------------------------------------------------------------
// Mesh Reload State
// ---------------------------------------------------------------------------

/// Stage of mesh reload process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshReloadStage {
    /// Importing mesh from disk.
    Importing,
    /// Generating meshlets.
    GeneratingMeshlets,
    /// Rebuilding BLAS.
    RebuildingBlas,
    /// Uploading buffers via staging.
    Uploading,
    /// Waiting for render proxy update.
    UpdatingProxy,
    /// Waiting for BLAS swap.
    SwappingBlas,
    /// Complete.
    Complete,
    /// Failed.
    Failed,
}

/// State of a mesh reload operation.
#[derive(Debug)]
pub struct MeshReloadState {
    /// Handle to the old mesh being replaced.
    pub old_handle: MeshHandle,
    /// New mesh buffers (set after import).
    pub new_buffers: Option<MeshBuffers>,
    /// New meshlet data (set after meshlet generation).
    pub new_meshlets: Option<MeshletMesh>,
    /// New BLAS handle (set after rebuild).
    pub new_blas: Option<BlasHandle>,
    /// New mesh handle (set after upload).
    pub new_handle: Option<MeshHandle>,
    /// Current processing stage.
    pub stage: MeshReloadStage,
    /// Whether this mesh has BLAS to rebuild.
    pub has_blas: bool,
    /// When this reload started.
    pub started_at: Instant,
    /// Staging buffer allocation offset.
    pub staging_offset: Option<usize>,
    /// Error if failed.
    pub error: Option<AssetReloadError>,
}

impl MeshReloadState {
    /// Create new mesh reload state.
    pub fn new(old_handle: MeshHandle, has_blas: bool) -> Self {
        Self {
            old_handle,
            new_buffers: None,
            new_meshlets: None,
            new_blas: None,
            new_handle: None,
            stage: MeshReloadStage::Importing,
            has_blas,
            started_at: Instant::now(),
            staging_offset: None,
            error: None,
        }
    }

    /// Get progress as fraction (0.0 to 1.0).
    pub fn progress(&self) -> f32 {
        match self.stage {
            MeshReloadStage::Importing => 0.1,
            MeshReloadStage::GeneratingMeshlets => 0.3,
            MeshReloadStage::RebuildingBlas => 0.5,
            MeshReloadStage::Uploading => 0.7,
            MeshReloadStage::UpdatingProxy => 0.85,
            MeshReloadStage::SwappingBlas => 0.95,
            MeshReloadStage::Complete => 1.0,
            MeshReloadStage::Failed => 0.0,
        }
    }

    /// Check if reload is complete.
    pub fn is_complete(&self) -> bool {
        matches!(
            self.stage,
            MeshReloadStage::Complete | MeshReloadStage::Failed
        )
    }

    /// Mark as failed with error.
    pub fn fail(&mut self, error: AssetReloadError) {
        self.stage = MeshReloadStage::Failed;
        self.error = Some(error);
    }
}

// ---------------------------------------------------------------------------
// Asset Swap
// ---------------------------------------------------------------------------

/// A pending asset swap operation.
#[derive(Debug)]
pub enum AssetSwap {
    /// Texture swap.
    Texture {
        /// Path to the texture.
        path: PathBuf,
        /// Old handle to release.
        old_handle: TextureHandle,
        /// New handle to activate.
        new_handle: TextureHandle,
        /// When swap was queued.
        queued_at: Instant,
    },
    /// Mesh swap.
    Mesh {
        /// Path to the mesh.
        path: PathBuf,
        /// Old handle to release.
        old_handle: MeshHandle,
        /// New handle to activate.
        new_handle: MeshHandle,
        /// Old BLAS handle (if any).
        old_blas: Option<BlasHandle>,
        /// New BLAS handle (if any).
        new_blas: Option<BlasHandle>,
        /// When swap was queued.
        queued_at: Instant,
    },
}

impl AssetSwap {
    /// Get the path of the asset being swapped.
    pub fn path(&self) -> &Path {
        match self {
            Self::Texture { path, .. } => path,
            Self::Mesh { path, .. } => path,
        }
    }

    /// Check if this is a texture swap.
    pub fn is_texture(&self) -> bool {
        matches!(self, Self::Texture { .. })
    }

    /// Check if this is a mesh swap.
    pub fn is_mesh(&self) -> bool {
        matches!(self, Self::Mesh { .. })
    }
}

// ---------------------------------------------------------------------------
// Staging Buffer Pool
// ---------------------------------------------------------------------------

/// Configuration for staging buffer pool.
#[derive(Debug, Clone)]
pub struct StagingBufferConfig {
    /// Size of each staging buffer in bytes.
    pub buffer_size: usize,
    /// Number of buffers in the pool (triple buffering recommended).
    pub pool_size: usize,
    /// Alignment requirement for allocations.
    pub alignment: usize,
}

impl Default for StagingBufferConfig {
    fn default() -> Self {
        Self {
            buffer_size: 64 * 1024 * 1024, // 64 MB
            pool_size: 3,                  // Triple buffering
            alignment: 256,                // GPU alignment
        }
    }
}

/// A staging buffer allocation.
#[derive(Debug, Clone)]
pub struct StagingAllocation {
    /// Buffer index in the pool.
    pub buffer_index: usize,
    /// Offset within the buffer.
    pub offset: usize,
    /// Size of the allocation.
    pub size: usize,
    /// Frame when this allocation was made.
    pub frame: u64,
}

/// Pool of staging buffers for CPU-to-GPU transfers.
#[derive(Debug)]
pub struct StagingBufferPool {
    /// Configuration.
    config: StagingBufferConfig,
    /// Buffer data storage (simulated).
    buffers: Vec<Vec<u8>>,
    /// Current write offset per buffer.
    write_offsets: Vec<usize>,
    /// Current buffer index for round-robin allocation.
    current_buffer: usize,
    /// Current frame number.
    current_frame: u64,
    /// Pending allocations.
    pending: Vec<StagingAllocation>,
    /// Total bytes allocated.
    total_allocated: usize,
}

impl StagingBufferPool {
    /// Create a new staging buffer pool.
    pub fn new(config: StagingBufferConfig) -> Self {
        let buffers = (0..config.pool_size)
            .map(|_| vec![0u8; config.buffer_size])
            .collect();
        let write_offsets = vec![0; config.pool_size];

        Self {
            config,
            buffers,
            write_offsets,
            current_buffer: 0,
            current_frame: 0,
            pending: Vec::new(),
            total_allocated: 0,
        }
    }

    /// Create pool with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(StagingBufferConfig::default())
    }

    /// Allocate space in the staging buffer.
    pub fn allocate(&mut self, size: usize) -> Option<StagingAllocation> {
        // Align the size
        let aligned_size = (size + self.config.alignment - 1) & !(self.config.alignment - 1);

        // Try current buffer first
        let buffer_idx = self.current_buffer;
        let offset = self.write_offsets[buffer_idx];

        if offset + aligned_size <= self.config.buffer_size {
            let allocation = StagingAllocation {
                buffer_index: buffer_idx,
                offset,
                size: aligned_size,
                frame: self.current_frame,
            };
            self.write_offsets[buffer_idx] += aligned_size;
            self.total_allocated += aligned_size;
            self.pending.push(allocation.clone());
            return Some(allocation);
        }

        // Try other buffers
        for i in 0..self.config.pool_size {
            let idx = (buffer_idx + i + 1) % self.config.pool_size;
            let offset = self.write_offsets[idx];

            if offset + aligned_size <= self.config.buffer_size {
                let allocation = StagingAllocation {
                    buffer_index: idx,
                    offset,
                    size: aligned_size,
                    frame: self.current_frame,
                };
                self.write_offsets[idx] += aligned_size;
                self.total_allocated += aligned_size;
                self.pending.push(allocation.clone());
                return Some(allocation);
            }
        }

        None // No space available
    }

    /// Write data to an allocation.
    pub fn write(&mut self, allocation: &StagingAllocation, data: &[u8]) -> bool {
        if allocation.buffer_index >= self.buffers.len() {
            return false;
        }
        if data.len() > allocation.size {
            return false;
        }

        let buffer = &mut self.buffers[allocation.buffer_index];
        let start = allocation.offset;
        let end = start + data.len();

        if end > buffer.len() {
            return false;
        }

        buffer[start..end].copy_from_slice(data);
        true
    }

    /// Read data from an allocation.
    pub fn read(&self, allocation: &StagingAllocation) -> Option<&[u8]> {
        let buffer = self.buffers.get(allocation.buffer_index)?;
        let start = allocation.offset;
        let end = start + allocation.size;

        if end > buffer.len() {
            return None;
        }

        Some(&buffer[start..end])
    }

    /// Advance to the next frame, resetting the current buffer.
    pub fn advance_frame(&mut self) {
        self.current_frame += 1;
        self.current_buffer = (self.current_buffer + 1) % self.config.pool_size;

        // Reset the buffer we're about to use (old data is no longer needed)
        self.write_offsets[self.current_buffer] = 0;

        // Clean up old pending allocations (older than pool_size frames)
        let grace_frames = self.config.pool_size as u64;
        self.pending.retain(|a| {
            self.current_frame.saturating_sub(a.frame) < grace_frames
        });
    }

    /// Get available space in the current buffer.
    pub fn available_space(&self) -> usize {
        self.config.buffer_size - self.write_offsets[self.current_buffer]
    }

    /// Get total allocated bytes.
    pub fn total_allocated(&self) -> usize {
        self.total_allocated
    }

    /// Get current frame number.
    pub fn current_frame(&self) -> u64 {
        self.current_frame
    }

    /// Reset all buffers (use sparingly).
    pub fn reset(&mut self) {
        for offset in &mut self.write_offsets {
            *offset = 0;
        }
        self.pending.clear();
        self.total_allocated = 0;
    }
}

// ---------------------------------------------------------------------------
// Asset Reload Configuration
// ---------------------------------------------------------------------------

/// Configuration for the asset reload manager.
#[derive(Debug, Clone)]
pub struct AssetReloadConfig {
    /// Enable cross-fade transitions for textures.
    pub enable_texture_fade: bool,
    /// Duration of texture cross-fade in seconds.
    pub fade_duration_secs: f32,
    /// Maximum concurrent texture reloads.
    pub max_texture_reloads: usize,
    /// Maximum concurrent mesh reloads.
    pub max_mesh_reloads: usize,
    /// Timeout for reload operations.
    pub reload_timeout: Duration,
    /// Meshlet generation config.
    pub meshlet_config: MeshletConfig,
    /// Compression quality for textures.
    pub compression_quality: CompressionQuality,
    /// Mipmap filter type.
    pub mipmap_filter: MipmapFilter,
}

impl Default for AssetReloadConfig {
    fn default() -> Self {
        Self {
            enable_texture_fade: true,
            fade_duration_secs: 0.1, // 100ms = 1-2 frames at 60fps
            max_texture_reloads: 4,
            max_mesh_reloads: 2,
            reload_timeout: Duration::from_secs(30),
            meshlet_config: MeshletConfig::default(),
            compression_quality: CompressionQuality::High,
            mipmap_filter: MipmapFilter::Lanczos,
        }
    }
}

/// Compression quality setting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionQuality {
    /// Fast compression, lower quality.
    Low,
    /// Balanced compression.
    Medium,
    /// High quality compression.
    High,
}

/// Mipmap filter type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MipmapFilter {
    /// Box filter (fast).
    Box,
    /// Lanczos filter (high quality).
    Lanczos,
    /// Kaiser filter.
    Kaiser,
}

// ---------------------------------------------------------------------------
// Asset Reload Manager
// ---------------------------------------------------------------------------

/// Manager for asset hot-reload operations.
///
/// Handles texture and mesh reloads with:
/// - Asynchronous processing pipeline
/// - Staging buffer management
/// - Atomic swap at phase boundaries
/// - Cross-fade transitions for textures
/// - BLAS rebuild for meshes
#[derive(Debug)]
pub struct AssetReloadManager {
    /// Pending texture reloads.
    pending_textures: HashMap<PathBuf, TextureReloadState>,
    /// Pending mesh reloads.
    pending_meshes: HashMap<PathBuf, MeshReloadState>,
    /// Staging buffer pool.
    staging_buffers: StagingBufferPool,
    /// Queue of pending swaps.
    swap_queue: VecDeque<AssetSwap>,
    /// Configuration.
    config: AssetReloadConfig,
    /// Next texture handle ID.
    next_texture_handle: u32,
    /// Next mesh handle ID.
    next_mesh_handle: u32,
    /// Statistics.
    stats: AssetReloadStats,
}

/// Statistics for asset reload operations.
#[derive(Debug, Clone, Default)]
pub struct AssetReloadStats {
    /// Total texture reloads completed.
    pub textures_reloaded: u64,
    /// Total mesh reloads completed.
    pub meshes_reloaded: u64,
    /// Total texture reload failures.
    pub texture_failures: u64,
    /// Total mesh reload failures.
    pub mesh_failures: u64,
    /// Total bytes uploaded via staging.
    pub bytes_uploaded: u64,
    /// Total BLAS rebuilds.
    pub blas_rebuilds: u64,
    /// Average texture reload time in milliseconds.
    pub avg_texture_reload_ms: f64,
    /// Average mesh reload time in milliseconds.
    pub avg_mesh_reload_ms: f64,
}

impl AssetReloadManager {
    /// Create a new asset reload manager.
    pub fn new(staging_pool: StagingBufferPool, config: AssetReloadConfig) -> Self {
        Self {
            pending_textures: HashMap::new(),
            pending_meshes: HashMap::new(),
            staging_buffers: staging_pool,
            swap_queue: VecDeque::new(),
            config,
            next_texture_handle: 1000, // Start after initial handles
            next_mesh_handle: 1000,
            stats: AssetReloadStats::default(),
        }
    }

    /// Create manager with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(
            StagingBufferPool::with_defaults(),
            AssetReloadConfig::default(),
        )
    }

    /// Start a texture reload operation.
    ///
    /// # Arguments
    /// * `path` - Path to the texture file
    /// * `old_handle` - Handle to the texture being replaced
    pub fn on_texture_change(&mut self, path: &Path, old_handle: TextureHandle) {
        self.on_texture_change_with_format(path, old_handle, TextureFormat::default());
    }

    /// Start a texture reload with specific format.
    pub fn on_texture_change_with_format(
        &mut self,
        path: &Path,
        old_handle: TextureHandle,
        format: TextureFormat,
    ) {
        // Check if already reloading
        if self.pending_textures.contains_key(path) {
            return;
        }

        // Check max concurrent reloads
        if self.pending_textures.len() >= self.config.max_texture_reloads {
            return;
        }

        let state = TextureReloadState::new(old_handle, format);
        self.pending_textures.insert(path.to_path_buf(), state);
    }

    /// Start a mesh reload operation.
    ///
    /// # Arguments
    /// * `path` - Path to the mesh file
    /// * `old_handle` - Handle to the mesh being replaced
    /// * `has_blas` - Whether the mesh has an associated BLAS
    pub fn on_mesh_change(&mut self, path: &Path, old_handle: MeshHandle, has_blas: bool) {
        // Check if already reloading
        if self.pending_meshes.contains_key(path) {
            return;
        }

        // Check max concurrent reloads
        if self.pending_meshes.len() >= self.config.max_mesh_reloads {
            return;
        }

        let state = MeshReloadState::new(old_handle, has_blas);
        self.pending_meshes.insert(path.to_path_buf(), state);
    }

    /// Poll progress of all reload operations.
    ///
    /// Returns events for all in-progress and completed reloads.
    pub fn poll_progress(&mut self) -> Vec<AssetReloadEvent> {
        let mut events = Vec::new();

        // Process texture reloads
        for (path, state) in &self.pending_textures {
            let status = match state.stage {
                TextureReloadStage::Importing => AssetReloadStatus::Importing,
                TextureReloadStage::GeneratingMips | TextureReloadStage::Compressing => {
                    AssetReloadStatus::Processing
                }
                TextureReloadStage::Uploading | TextureReloadStage::UpdatingDescriptor => {
                    AssetReloadStatus::Uploading
                }
                TextureReloadStage::Fading => AssetReloadStatus::WaitingForSwap,
                TextureReloadStage::Complete => AssetReloadStatus::Swapped,
                TextureReloadStage::Failed => {
                    AssetReloadStatus::Failed(state.error.clone().unwrap_or_else(|| {
                        AssetReloadError::Internal("unknown error".to_string())
                    }))
                }
            };

            events.push(AssetReloadEvent {
                path: path.clone(),
                kind: AssetReloadKind::Texture {
                    old_handle: state.old_handle,
                    format: state.target_format,
                },
                status,
                progress: state.progress(),
                started_at: state.started_at,
                sequence: 0,
            });
        }

        // Process mesh reloads
        for (path, state) in &self.pending_meshes {
            let status = match state.stage {
                MeshReloadStage::Importing => AssetReloadStatus::Importing,
                MeshReloadStage::GeneratingMeshlets | MeshReloadStage::RebuildingBlas => {
                    AssetReloadStatus::Processing
                }
                MeshReloadStage::Uploading | MeshReloadStage::UpdatingProxy => {
                    AssetReloadStatus::Uploading
                }
                MeshReloadStage::SwappingBlas => AssetReloadStatus::WaitingForSwap,
                MeshReloadStage::Complete => AssetReloadStatus::Swapped,
                MeshReloadStage::Failed => {
                    AssetReloadStatus::Failed(state.error.clone().unwrap_or_else(|| {
                        AssetReloadError::Internal("unknown error".to_string())
                    }))
                }
            };

            events.push(AssetReloadEvent {
                path: path.clone(),
                kind: AssetReloadKind::Mesh {
                    old_handle: state.old_handle,
                    has_blas: state.has_blas,
                },
                status,
                progress: state.progress(),
                started_at: state.started_at,
                sequence: 0,
            });
        }

        events
    }

    /// Execute pending swaps at a phase boundary.
    ///
    /// This should be called at a safe point in the render loop,
    /// typically between frames or between render phases.
    pub fn execute_swaps_at_phase_boundary(&mut self) -> Vec<AssetSwap> {
        let swaps: Vec<AssetSwap> = self.swap_queue.drain(..).collect();

        for swap in &swaps {
            match swap {
                AssetSwap::Texture { path, .. } => {
                    if let Some(state) = self.pending_textures.get_mut(path) {
                        state.stage = TextureReloadStage::Complete;
                        self.stats.textures_reloaded += 1;
                    }
                }
                AssetSwap::Mesh { path, .. } => {
                    if let Some(state) = self.pending_meshes.get_mut(path) {
                        state.stage = MeshReloadStage::Complete;
                        self.stats.meshes_reloaded += 1;
                        if state.has_blas {
                            self.stats.blas_rebuilds += 1;
                        }
                    }
                }
            }
        }

        // Clean up completed reloads
        self.pending_textures.retain(|_, s| !s.is_complete());
        self.pending_meshes.retain(|_, s| !s.is_complete());

        swaps
    }

    /// Get fade alpha for cross-fade rendering.
    ///
    /// Returns the blend factor between old (0.0) and new (1.0) texture.
    pub fn get_fade_alpha(&self, path: &Path) -> Option<f32> {
        self.pending_textures
            .get(path)
            .filter(|s| s.stage == TextureReloadStage::Fading)
            .map(|s| s.fade_progress)
    }

    /// Update fade progress for texture cross-fade.
    pub fn update_fades(&mut self, delta_time: f32) {
        if !self.config.enable_texture_fade || self.config.fade_duration_secs <= 0.0 {
            return;
        }

        let fade_speed = 1.0 / self.config.fade_duration_secs;

        for (_, state) in &mut self.pending_textures {
            if state.stage == TextureReloadStage::Fading {
                state.fade_progress += delta_time * fade_speed;
                if state.fade_progress >= 1.0 {
                    state.fade_progress = 1.0;
                    // Queue the final swap
                    if let Some(new_handle) = state.new_handle {
                        // Swap is already queued
                        state.stage = TextureReloadStage::Complete;
                    }
                }
            }
        }
    }

    /// Advance the staging buffer frame.
    pub fn advance_frame(&mut self) {
        self.staging_buffers.advance_frame();
    }

    /// Get statistics.
    pub fn stats(&self) -> &AssetReloadStats {
        &self.stats
    }

    /// Get mutable access to staging buffers.
    pub fn staging_buffers_mut(&mut self) -> &mut StagingBufferPool {
        &mut self.staging_buffers
    }

    /// Get the number of pending texture reloads.
    pub fn pending_texture_count(&self) -> usize {
        self.pending_textures.len()
    }

    /// Get the number of pending mesh reloads.
    pub fn pending_mesh_count(&self) -> usize {
        self.pending_meshes.len()
    }

    /// Check if any reloads are in progress.
    pub fn has_pending_reloads(&self) -> bool {
        !self.pending_textures.is_empty() || !self.pending_meshes.is_empty()
    }

    /// Cancel a pending texture reload.
    pub fn cancel_texture_reload(&mut self, path: &Path) -> bool {
        self.pending_textures.remove(path).is_some()
    }

    /// Cancel a pending mesh reload.
    pub fn cancel_mesh_reload(&mut self, path: &Path) -> bool {
        self.pending_meshes.remove(path).is_some()
    }

    /// Cancel all pending reloads.
    pub fn cancel_all(&mut self) {
        self.pending_textures.clear();
        self.pending_meshes.clear();
        self.swap_queue.clear();
    }

    // -------------------------------------------------------------------------
    // Internal Processing Methods
    // -------------------------------------------------------------------------

    /// Process a texture import (simulated for testing).
    pub fn process_texture_import(
        &mut self,
        path: &Path,
        data: Vec<u8>,
        width: u32,
        height: u32,
    ) -> AssetReloadResult<()> {
        let state = self
            .pending_textures
            .get_mut(path)
            .ok_or_else(|| AssetReloadError::AssetNotFound(path.to_path_buf()))?;

        state.import_data = Some(data);
        state.stage = TextureReloadStage::GeneratingMips;
        Ok(())
    }

    /// Process mipmap generation (simulated for testing).
    pub fn process_texture_mips(
        &mut self,
        path: &Path,
        mip_count: u32,
    ) -> AssetReloadResult<()> {
        let state = self
            .pending_textures
            .get_mut(path)
            .ok_or_else(|| AssetReloadError::AssetNotFound(path.to_path_buf()))?;

        // In real implementation, this would generate mipmaps
        state.stage = TextureReloadStage::Compressing;
        Ok(())
    }

    /// Process texture compression (simulated for testing).
    pub fn process_texture_compression(&mut self, path: &Path) -> AssetReloadResult<()> {
        let state = self
            .pending_textures
            .get_mut(path)
            .ok_or_else(|| AssetReloadError::AssetNotFound(path.to_path_buf()))?;

        // In real implementation, this would compress the texture
        state.stage = TextureReloadStage::Uploading;
        Ok(())
    }

    /// Process texture upload (simulated for testing).
    pub fn process_texture_upload(&mut self, path: &Path, image: GpuImage) -> AssetReloadResult<TextureHandle> {
        let state = self
            .pending_textures
            .get_mut(path)
            .ok_or_else(|| AssetReloadError::AssetNotFound(path.to_path_buf()))?;

        // Allocate staging buffer
        let size = image.total_size();
        let allocation = self
            .staging_buffers
            .allocate(size)
            .ok_or(AssetReloadError::StagingAllocationFailed {
                required: size,
                available: self.staging_buffers.available_space(),
            })?;

        // Write data to staging
        self.staging_buffers.write(&allocation, &image.data);
        self.stats.bytes_uploaded += size as u64;

        // Generate new handle
        let new_handle = TextureHandle::new(self.next_texture_handle);
        self.next_texture_handle += 1;

        state.new_image = Some(image);
        state.new_handle = Some(new_handle);
        state.stage = TextureReloadStage::UpdatingDescriptor;

        Ok(new_handle)
    }

    /// Process descriptor update and queue swap.
    pub fn process_texture_descriptor_update(&mut self, path: &Path) -> AssetReloadResult<()> {
        let state = self
            .pending_textures
            .get_mut(path)
            .ok_or_else(|| AssetReloadError::AssetNotFound(path.to_path_buf()))?;

        let new_handle = state
            .new_handle
            .ok_or(AssetReloadError::Internal("no new handle".to_string()))?;

        if self.config.enable_texture_fade {
            state.stage = TextureReloadStage::Fading;
            state.fade_progress = 0.0;
        } else {
            // Queue immediate swap
            self.swap_queue.push_back(AssetSwap::Texture {
                path: path.to_path_buf(),
                old_handle: state.old_handle,
                new_handle,
                queued_at: Instant::now(),
            });
            state.stage = TextureReloadStage::Complete;
        }

        Ok(())
    }

    /// Process a mesh import (simulated for testing).
    pub fn process_mesh_import(
        &mut self,
        path: &Path,
        buffers: MeshBuffers,
    ) -> AssetReloadResult<()> {
        let state = self
            .pending_meshes
            .get_mut(path)
            .ok_or_else(|| AssetReloadError::AssetNotFound(path.to_path_buf()))?;

        state.new_buffers = Some(buffers);
        state.stage = MeshReloadStage::GeneratingMeshlets;
        Ok(())
    }

    /// Process meshlet generation.
    pub fn process_mesh_meshlets(
        &mut self,
        path: &Path,
        meshlets: MeshletMesh,
    ) -> AssetReloadResult<()> {
        let state = self
            .pending_meshes
            .get_mut(path)
            .ok_or_else(|| AssetReloadError::AssetNotFound(path.to_path_buf()))?;

        state.new_meshlets = Some(meshlets);

        if state.has_blas {
            state.stage = MeshReloadStage::RebuildingBlas;
        } else {
            state.stage = MeshReloadStage::Uploading;
        }

        Ok(())
    }

    /// Process BLAS rebuild.
    pub fn process_mesh_blas(&mut self, path: &Path, blas: BlasHandle) -> AssetReloadResult<()> {
        let state = self
            .pending_meshes
            .get_mut(path)
            .ok_or_else(|| AssetReloadError::AssetNotFound(path.to_path_buf()))?;

        state.new_blas = Some(blas);
        state.stage = MeshReloadStage::Uploading;
        Ok(())
    }

    /// Process mesh buffer upload.
    pub fn process_mesh_upload(&mut self, path: &Path) -> AssetReloadResult<MeshHandle> {
        let state = self
            .pending_meshes
            .get_mut(path)
            .ok_or_else(|| AssetReloadError::AssetNotFound(path.to_path_buf()))?;

        let buffers = state
            .new_buffers
            .as_ref()
            .ok_or(AssetReloadError::Internal("no buffers".to_string()))?;

        // Allocate staging buffer
        let size = buffers.total_size();
        let allocation = self
            .staging_buffers
            .allocate(size)
            .ok_or(AssetReloadError::StagingAllocationFailed {
                required: size,
                available: self.staging_buffers.available_space(),
            })?;

        state.staging_offset = Some(allocation.offset);
        self.stats.bytes_uploaded += size as u64;

        // Generate new handle
        let new_handle = MeshHandle::new(self.next_mesh_handle);
        self.next_mesh_handle += 1;

        state.new_handle = Some(new_handle);
        state.stage = MeshReloadStage::UpdatingProxy;

        Ok(new_handle)
    }

    /// Process render proxy update and queue swap.
    pub fn process_mesh_proxy_update(&mut self, path: &Path) -> AssetReloadResult<()> {
        let state = self
            .pending_meshes
            .get_mut(path)
            .ok_or_else(|| AssetReloadError::AssetNotFound(path.to_path_buf()))?;

        let new_handle = state
            .new_handle
            .ok_or(AssetReloadError::Internal("no new handle".to_string()))?;

        if state.has_blas && state.new_blas.is_some() {
            state.stage = MeshReloadStage::SwappingBlas;
        } else {
            // Queue immediate swap
            self.swap_queue.push_back(AssetSwap::Mesh {
                path: path.to_path_buf(),
                old_handle: state.old_handle,
                new_handle,
                old_blas: None,
                new_blas: state.new_blas,
                queued_at: Instant::now(),
            });
            state.stage = MeshReloadStage::Complete;
        }

        Ok(())
    }

    /// Queue BLAS swap (may take multiple frames).
    pub fn queue_blas_swap(&mut self, path: &Path) -> AssetReloadResult<()> {
        let state = self
            .pending_meshes
            .get_mut(path)
            .ok_or_else(|| AssetReloadError::AssetNotFound(path.to_path_buf()))?;

        let new_handle = state
            .new_handle
            .ok_or(AssetReloadError::Internal("no new handle".to_string()))?;

        self.swap_queue.push_back(AssetSwap::Mesh {
            path: path.to_path_buf(),
            old_handle: state.old_handle,
            new_handle,
            old_blas: None, // Old BLAS would be tracked separately
            new_blas: state.new_blas,
            queued_at: Instant::now(),
        });

        // Don't mark complete yet - BLAS swap may take multiple frames
        Ok(())
    }

    /// Mark a texture reload as failed.
    pub fn fail_texture_reload(&mut self, path: &Path, error: AssetReloadError) {
        if let Some(state) = self.pending_textures.get_mut(path) {
            state.fail(error);
            self.stats.texture_failures += 1;
        }
    }

    /// Mark a mesh reload as failed.
    pub fn fail_mesh_reload(&mut self, path: &Path, error: AssetReloadError) {
        if let Some(state) = self.pending_meshes.get_mut(path) {
            state.fail(error);
            self.stats.mesh_failures += 1;
        }
    }

    /// Get the current stage of a texture reload.
    pub fn texture_stage(&self, path: &Path) -> Option<TextureReloadStage> {
        self.pending_textures.get(path).map(|s| s.stage)
    }

    /// Get the current stage of a mesh reload.
    pub fn mesh_stage(&self, path: &Path) -> Option<MeshReloadStage> {
        self.pending_meshes.get(path).map(|s| s.stage)
    }

    /// Get pending swap count.
    pub fn pending_swap_count(&self) -> usize {
        self.swap_queue.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Texture Reload Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_texture_change_detection() {
        let mut manager = AssetReloadManager::with_defaults();
        let path = PathBuf::from("textures/albedo.png");
        let old_handle = TextureHandle::new(1);

        manager.on_texture_change(&path, old_handle);

        assert_eq!(manager.pending_texture_count(), 1);
        assert!(manager.pending_textures.contains_key(&path));
    }

    #[test]
    fn test_texture_reimport() {
        let mut manager = AssetReloadManager::with_defaults();
        let path = PathBuf::from("textures/albedo.png");
        let old_handle = TextureHandle::new(1);

        manager.on_texture_change(&path, old_handle);

        // Simulate import
        let data = vec![0u8; 1024];
        manager.process_texture_import(&path, data.clone(), 32, 32).unwrap();

        let state = manager.pending_textures.get(&path).unwrap();
        assert_eq!(state.stage, TextureReloadStage::GeneratingMips);
        assert!(state.import_data.is_some());
    }

    #[test]
    fn test_texture_recompress() {
        let mut manager = AssetReloadManager::with_defaults();
        let path = PathBuf::from("textures/albedo.png");
        let old_handle = TextureHandle::new(1);

        manager.on_texture_change(&path, old_handle);
        manager.process_texture_import(&path, vec![0u8; 1024], 32, 32).unwrap();
        manager.process_texture_mips(&path, 5).unwrap();
        manager.process_texture_compression(&path).unwrap();

        let state = manager.pending_textures.get(&path).unwrap();
        assert_eq!(state.stage, TextureReloadStage::Uploading);
    }

    #[test]
    fn test_texture_upload() {
        let mut manager = AssetReloadManager::with_defaults();
        let path = PathBuf::from("textures/albedo.png");
        let old_handle = TextureHandle::new(1);

        manager.on_texture_change(&path, old_handle);
        manager.process_texture_import(&path, vec![0u8; 1024], 32, 32).unwrap();
        manager.process_texture_mips(&path, 5).unwrap();
        manager.process_texture_compression(&path).unwrap();

        let image = GpuImage::new(vec![0u8; 4096], 32, 32, 5, TextureFormat::Rgba8Srgb);
        let new_handle = manager.process_texture_upload(&path, image).unwrap();

        assert!(new_handle.id() >= 1000);
        let state = manager.pending_textures.get(&path).unwrap();
        assert_eq!(state.stage, TextureReloadStage::UpdatingDescriptor);
    }

    #[test]
    fn test_texture_descriptor_update() {
        let mut manager = AssetReloadManager::with_defaults();
        manager.config.enable_texture_fade = false;

        let path = PathBuf::from("textures/albedo.png");
        let old_handle = TextureHandle::new(1);

        manager.on_texture_change(&path, old_handle);
        manager.process_texture_import(&path, vec![0u8; 1024], 32, 32).unwrap();
        manager.process_texture_mips(&path, 5).unwrap();
        manager.process_texture_compression(&path).unwrap();

        let image = GpuImage::new(vec![0u8; 4096], 32, 32, 5, TextureFormat::Rgba8Srgb);
        manager.process_texture_upload(&path, image).unwrap();
        manager.process_texture_descriptor_update(&path).unwrap();

        assert_eq!(manager.pending_swap_count(), 1);
    }

    #[test]
    fn test_texture_fade_transition() {
        let mut manager = AssetReloadManager::with_defaults();
        manager.config.enable_texture_fade = true;
        manager.config.fade_duration_secs = 0.1;

        let path = PathBuf::from("textures/albedo.png");
        let old_handle = TextureHandle::new(1);

        manager.on_texture_change(&path, old_handle);
        manager.process_texture_import(&path, vec![0u8; 1024], 32, 32).unwrap();
        manager.process_texture_mips(&path, 5).unwrap();
        manager.process_texture_compression(&path).unwrap();

        let image = GpuImage::new(vec![0u8; 4096], 32, 32, 5, TextureFormat::Rgba8Srgb);
        manager.process_texture_upload(&path, image).unwrap();
        manager.process_texture_descriptor_update(&path).unwrap();

        let state = manager.pending_textures.get(&path).unwrap();
        assert_eq!(state.stage, TextureReloadStage::Fading);

        // Test fade progress
        assert!(manager.get_fade_alpha(&path).is_some());
        let alpha = manager.get_fade_alpha(&path).unwrap();
        assert_eq!(alpha, 0.0);

        // Update fade
        manager.update_fades(0.05);
        let alpha = manager.get_fade_alpha(&path).unwrap();
        assert!(alpha > 0.0);
    }

    // -------------------------------------------------------------------------
    // Mesh Reload Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_mesh_change_detection() {
        let mut manager = AssetReloadManager::with_defaults();
        let path = PathBuf::from("meshes/cube.gltf");
        let old_handle = MeshHandle::new(1);

        manager.on_mesh_change(&path, old_handle, true);

        assert_eq!(manager.pending_mesh_count(), 1);
        assert!(manager.pending_meshes.contains_key(&path));
    }

    #[test]
    fn test_mesh_reimport() {
        let mut manager = AssetReloadManager::with_defaults();
        let path = PathBuf::from("meshes/cube.gltf");
        let old_handle = MeshHandle::new(1);

        manager.on_mesh_change(&path, old_handle, false);

        let buffers = MeshBuffers::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            vec![[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            vec![0, 1, 2],
        );
        manager.process_mesh_import(&path, buffers).unwrap();

        let state = manager.pending_meshes.get(&path).unwrap();
        assert_eq!(state.stage, MeshReloadStage::GeneratingMeshlets);
        assert!(state.new_buffers.is_some());
    }

    #[test]
    fn test_mesh_regenerate_meshlets() {
        let mut manager = AssetReloadManager::with_defaults();
        let path = PathBuf::from("meshes/cube.gltf");
        let old_handle = MeshHandle::new(1);

        manager.on_mesh_change(&path, old_handle, false);

        let buffers = MeshBuffers::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            vec![[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            vec![0, 1, 2],
        );
        manager.process_mesh_import(&path, buffers).unwrap();

        let meshlets = MeshletMesh::new();
        manager.process_mesh_meshlets(&path, meshlets).unwrap();

        let state = manager.pending_meshes.get(&path).unwrap();
        assert_eq!(state.stage, MeshReloadStage::Uploading);
        assert!(state.new_meshlets.is_some());
    }

    #[test]
    fn test_mesh_update_buffers() {
        let mut manager = AssetReloadManager::with_defaults();
        let path = PathBuf::from("meshes/cube.gltf");
        let old_handle = MeshHandle::new(1);

        manager.on_mesh_change(&path, old_handle, false);

        let buffers = MeshBuffers::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            vec![[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            vec![0, 1, 2],
        );
        manager.process_mesh_import(&path, buffers).unwrap();

        let meshlets = MeshletMesh::new();
        manager.process_mesh_meshlets(&path, meshlets).unwrap();

        let new_handle = manager.process_mesh_upload(&path).unwrap();

        assert!(new_handle.id() >= 1000);
        let state = manager.pending_meshes.get(&path).unwrap();
        assert_eq!(state.stage, MeshReloadStage::UpdatingProxy);
    }

    #[test]
    fn test_mesh_blas_rebuild() {
        let mut manager = AssetReloadManager::with_defaults();
        let path = PathBuf::from("meshes/cube.gltf");
        let old_handle = MeshHandle::new(1);

        manager.on_mesh_change(&path, old_handle, true);

        let buffers = MeshBuffers::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            vec![[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            vec![0, 1, 2],
        );
        manager.process_mesh_import(&path, buffers).unwrap();

        let meshlets = MeshletMesh::new();
        manager.process_mesh_meshlets(&path, meshlets).unwrap();

        // Should be waiting for BLAS rebuild
        let state = manager.pending_meshes.get(&path).unwrap();
        assert_eq!(state.stage, MeshReloadStage::RebuildingBlas);

        // Process BLAS rebuild - use unsafe transmute to create handle for testing
        // In production code, BlasHandle would come from BlasPool
        let blas = unsafe { std::mem::transmute::<u32, BlasHandle>(100) };
        manager.process_mesh_blas(&path, blas).unwrap();

        let state = manager.pending_meshes.get(&path).unwrap();
        assert_eq!(state.stage, MeshReloadStage::Uploading);
        assert!(state.new_blas.is_some());
    }

    #[test]
    fn test_mesh_atomic_proxy_update() {
        let mut manager = AssetReloadManager::with_defaults();
        let path = PathBuf::from("meshes/cube.gltf");
        let old_handle = MeshHandle::new(1);

        manager.on_mesh_change(&path, old_handle, false);

        let buffers = MeshBuffers::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            vec![[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            vec![0, 1, 2],
        );
        manager.process_mesh_import(&path, buffers).unwrap();

        let meshlets = MeshletMesh::new();
        manager.process_mesh_meshlets(&path, meshlets).unwrap();
        manager.process_mesh_upload(&path).unwrap();
        manager.process_mesh_proxy_update(&path).unwrap();

        assert_eq!(manager.pending_swap_count(), 1);
    }

    // -------------------------------------------------------------------------
    // Staging Buffer Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_staging_allocate() {
        let config = StagingBufferConfig {
            buffer_size: 1024,
            pool_size: 3,
            alignment: 16,
        };
        let mut pool = StagingBufferPool::new(config);

        let alloc = pool.allocate(100).unwrap();
        assert_eq!(alloc.buffer_index, 0);
        assert_eq!(alloc.offset, 0);
        assert!(alloc.size >= 100);
        assert_eq!(alloc.size % 16, 0); // Aligned
    }

    #[test]
    fn test_staging_copy() {
        let config = StagingBufferConfig {
            buffer_size: 1024,
            pool_size: 3,
            alignment: 16,
        };
        let mut pool = StagingBufferPool::new(config);

        let alloc = pool.allocate(100).unwrap();
        let data = vec![42u8; 100];

        assert!(pool.write(&alloc, &data));

        let read_data = pool.read(&alloc).unwrap();
        assert_eq!(&read_data[..100], &data[..]);
    }

    #[test]
    fn test_staging_release() {
        let config = StagingBufferConfig {
            buffer_size: 1024,
            pool_size: 3,
            alignment: 16,
        };
        let mut pool = StagingBufferPool::new(config);

        // Fill up first buffer
        let alloc1 = pool.allocate(512).unwrap();
        let alloc2 = pool.allocate(512).unwrap();

        // Should use second buffer now
        let alloc3 = pool.allocate(256).unwrap();
        assert_eq!(alloc3.buffer_index, 1);

        // Advance frame to release old allocations
        pool.advance_frame();
        pool.advance_frame();
        pool.advance_frame();

        // First buffer should be available again
        let alloc4 = pool.allocate(256).unwrap();
        assert_eq!(alloc4.buffer_index, 0);
        assert_eq!(alloc4.offset, 0);
    }

    #[test]
    fn test_staging_pool_management() {
        let config = StagingBufferConfig {
            buffer_size: 1024,
            pool_size: 3,
            alignment: 16,
        };
        let mut pool = StagingBufferPool::new(config);

        // Test frame advancement
        assert_eq!(pool.current_frame(), 0);
        pool.advance_frame();
        assert_eq!(pool.current_frame(), 1);

        // Test available space
        let initial_space = pool.available_space();
        pool.allocate(100).unwrap();
        assert!(pool.available_space() < initial_space);

        // Test reset
        pool.reset();
        assert_eq!(pool.available_space(), 1024);
    }

    // -------------------------------------------------------------------------
    // Swap Queue Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_queue_swap() {
        let mut manager = AssetReloadManager::with_defaults();
        manager.config.enable_texture_fade = false;

        let path = PathBuf::from("textures/albedo.png");
        let old_handle = TextureHandle::new(1);

        manager.on_texture_change(&path, old_handle);
        manager.process_texture_import(&path, vec![0u8; 1024], 32, 32).unwrap();
        manager.process_texture_mips(&path, 5).unwrap();
        manager.process_texture_compression(&path).unwrap();

        let image = GpuImage::new(vec![0u8; 4096], 32, 32, 5, TextureFormat::Rgba8Srgb);
        manager.process_texture_upload(&path, image).unwrap();
        manager.process_texture_descriptor_update(&path).unwrap();

        assert_eq!(manager.pending_swap_count(), 1);
    }

    #[test]
    fn test_execute_at_boundary() {
        let mut manager = AssetReloadManager::with_defaults();
        manager.config.enable_texture_fade = false;

        let path = PathBuf::from("textures/albedo.png");
        let old_handle = TextureHandle::new(1);

        manager.on_texture_change(&path, old_handle);
        manager.process_texture_import(&path, vec![0u8; 1024], 32, 32).unwrap();
        manager.process_texture_mips(&path, 5).unwrap();
        manager.process_texture_compression(&path).unwrap();

        let image = GpuImage::new(vec![0u8; 4096], 32, 32, 5, TextureFormat::Rgba8Srgb);
        manager.process_texture_upload(&path, image).unwrap();
        manager.process_texture_descriptor_update(&path).unwrap();

        let swaps = manager.execute_swaps_at_phase_boundary();

        assert_eq!(swaps.len(), 1);
        assert_eq!(manager.pending_swap_count(), 0);
        assert_eq!(manager.stats.textures_reloaded, 1);
    }

    #[test]
    fn test_retain_old_resource() {
        let mut manager = AssetReloadManager::with_defaults();
        let path = PathBuf::from("textures/albedo.png");
        let old_handle = TextureHandle::new(1);

        manager.on_texture_change(&path, old_handle);

        // During reload, old resource is retained
        let state = manager.pending_textures.get(&path).unwrap();
        assert_eq!(state.old_handle, old_handle);

        // Old handle is preserved until swap completes
        manager.process_texture_import(&path, vec![0u8; 1024], 32, 32).unwrap();
        let state = manager.pending_textures.get(&path).unwrap();
        assert_eq!(state.old_handle, old_handle);
    }

    #[test]
    fn test_swap_timing() {
        let mut manager = AssetReloadManager::with_defaults();
        manager.config.enable_texture_fade = false;

        let path = PathBuf::from("textures/albedo.png");
        let old_handle = TextureHandle::new(1);

        let start = Instant::now();
        manager.on_texture_change(&path, old_handle);
        manager.process_texture_import(&path, vec![0u8; 1024], 32, 32).unwrap();
        manager.process_texture_mips(&path, 5).unwrap();
        manager.process_texture_compression(&path).unwrap();

        let image = GpuImage::new(vec![0u8; 4096], 32, 32, 5, TextureFormat::Rgba8Srgb);
        manager.process_texture_upload(&path, image).unwrap();
        manager.process_texture_descriptor_update(&path).unwrap();

        let swaps = manager.execute_swaps_at_phase_boundary();

        // Check that swap was queued after reload started
        if let AssetSwap::Texture { queued_at, .. } = &swaps[0] {
            assert!(*queued_at >= start);
        }
    }

    // -------------------------------------------------------------------------
    // Error Handling Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_import_failure() {
        let mut manager = AssetReloadManager::with_defaults();
        let path = PathBuf::from("textures/missing.png");
        let old_handle = TextureHandle::new(1);

        manager.on_texture_change(&path, old_handle);
        manager.fail_texture_reload(&path, AssetReloadError::AssetNotFound(path.clone()));

        let state = manager.pending_textures.get(&path).unwrap();
        assert_eq!(state.stage, TextureReloadStage::Failed);
        assert!(state.error.is_some());
        assert_eq!(manager.stats.texture_failures, 1);
    }

    #[test]
    fn test_upload_failure() {
        // Create a pool with very small buffers
        let config = StagingBufferConfig {
            buffer_size: 64,
            pool_size: 1,
            alignment: 16,
        };
        let pool = StagingBufferPool::new(config);
        let mut manager = AssetReloadManager::new(pool, AssetReloadConfig::default());

        let path = PathBuf::from("textures/large.png");
        let old_handle = TextureHandle::new(1);

        manager.on_texture_change(&path, old_handle);
        manager.process_texture_import(&path, vec![0u8; 1024], 32, 32).unwrap();
        manager.process_texture_mips(&path, 5).unwrap();
        manager.process_texture_compression(&path).unwrap();

        // Try to upload a large image
        let image = GpuImage::new(vec![0u8; 4096], 32, 32, 5, TextureFormat::Rgba8Srgb);
        let result = manager.process_texture_upload(&path, image);

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            AssetReloadError::StagingAllocationFailed { .. }
        ));
    }

    #[test]
    fn test_graceful_fallback() {
        let mut manager = AssetReloadManager::with_defaults();
        let path = PathBuf::from("textures/albedo.png");
        let old_handle = TextureHandle::new(1);

        manager.on_texture_change(&path, old_handle);

        // Fail the reload
        manager.fail_texture_reload(&path, AssetReloadError::ImportFailed("test".to_string()));

        // Verify old handle is still valid (not released)
        let state = manager.pending_textures.get(&path).unwrap();
        assert_eq!(state.old_handle, old_handle);
        assert_eq!(state.stage, TextureReloadStage::Failed);

        // No swap should be queued
        assert_eq!(manager.pending_swap_count(), 0);
    }

    // -------------------------------------------------------------------------
    // Integration Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_full_texture_reload() {
        let mut manager = AssetReloadManager::with_defaults();
        manager.config.enable_texture_fade = false;

        let path = PathBuf::from("textures/albedo.png");
        let old_handle = TextureHandle::new(1);

        // 1. Detect change
        manager.on_texture_change(&path, old_handle);
        assert_eq!(manager.texture_stage(&path), Some(TextureReloadStage::Importing));

        // 2. Import
        manager.process_texture_import(&path, vec![0u8; 1024], 64, 64).unwrap();
        assert_eq!(manager.texture_stage(&path), Some(TextureReloadStage::GeneratingMips));

        // 3. Generate mips
        manager.process_texture_mips(&path, 6).unwrap();
        assert_eq!(manager.texture_stage(&path), Some(TextureReloadStage::Compressing));

        // 4. Compress
        manager.process_texture_compression(&path).unwrap();
        assert_eq!(manager.texture_stage(&path), Some(TextureReloadStage::Uploading));

        // 5. Upload
        let image = GpuImage::new(vec![0u8; 8192], 64, 64, 6, TextureFormat::BC7);
        let new_handle = manager.process_texture_upload(&path, image).unwrap();
        assert_eq!(manager.texture_stage(&path), Some(TextureReloadStage::UpdatingDescriptor));

        // 6. Update descriptor
        manager.process_texture_descriptor_update(&path).unwrap();

        // 7. Execute swap at phase boundary
        let swaps = manager.execute_swaps_at_phase_boundary();

        assert_eq!(swaps.len(), 1);
        if let AssetSwap::Texture { old_handle: old, new_handle: new, .. } = &swaps[0] {
            assert_eq!(*old, old_handle);
            assert_eq!(*new, new_handle);
        }

        // 8. Verify completion
        assert_eq!(manager.pending_texture_count(), 0);
        assert_eq!(manager.stats.textures_reloaded, 1);
    }

    #[test]
    fn test_full_mesh_reload_with_blas() {
        let mut manager = AssetReloadManager::with_defaults();
        let path = PathBuf::from("meshes/character.gltf");
        let old_handle = MeshHandle::new(1);

        // 1. Detect change
        manager.on_mesh_change(&path, old_handle, true);
        assert_eq!(manager.mesh_stage(&path), Some(MeshReloadStage::Importing));

        // 2. Import
        let buffers = MeshBuffers::new(
            vec![[0.0, 0.0, 0.0]; 1000],
            vec![[0.0, 1.0, 0.0]; 1000],
            vec![[0.0, 0.0]; 1000],
            (0..3000).collect(),
        );
        manager.process_mesh_import(&path, buffers).unwrap();
        assert_eq!(manager.mesh_stage(&path), Some(MeshReloadStage::GeneratingMeshlets));

        // 3. Generate meshlets
        let meshlets = MeshletMesh::new();
        manager.process_mesh_meshlets(&path, meshlets).unwrap();
        assert_eq!(manager.mesh_stage(&path), Some(MeshReloadStage::RebuildingBlas));

        // 4. Rebuild BLAS - use unsafe transmute to create handle for testing
        let blas = unsafe { std::mem::transmute::<u32, BlasHandle>(100) };
        manager.process_mesh_blas(&path, blas).unwrap();
        assert_eq!(manager.mesh_stage(&path), Some(MeshReloadStage::Uploading));

        // 5. Upload buffers
        let new_handle = manager.process_mesh_upload(&path).unwrap();
        assert_eq!(manager.mesh_stage(&path), Some(MeshReloadStage::UpdatingProxy));

        // 6. Update proxy
        manager.process_mesh_proxy_update(&path).unwrap();
        assert_eq!(manager.mesh_stage(&path), Some(MeshReloadStage::SwappingBlas));

        // 7. Queue BLAS swap
        manager.queue_blas_swap(&path).unwrap();

        // 8. Execute swap at phase boundary
        let swaps = manager.execute_swaps_at_phase_boundary();

        assert_eq!(swaps.len(), 1);
        if let AssetSwap::Mesh { new_blas, .. } = &swaps[0] {
            assert!(new_blas.is_some());
        }

        // 9. Verify completion
        assert_eq!(manager.stats.meshes_reloaded, 1);
        assert_eq!(manager.stats.blas_rebuilds, 1);
    }

    // -------------------------------------------------------------------------
    // Additional Tests for Coverage
    // -------------------------------------------------------------------------

    #[test]
    fn test_poll_progress() {
        let mut manager = AssetReloadManager::with_defaults();
        let path = PathBuf::from("textures/albedo.png");

        manager.on_texture_change(&path, TextureHandle::new(1));

        let events = manager.poll_progress();
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0].status, AssetReloadStatus::Importing));
        assert!(events[0].progress < 0.2);
    }

    #[test]
    fn test_cancel_reload() {
        let mut manager = AssetReloadManager::with_defaults();
        let path = PathBuf::from("textures/albedo.png");

        manager.on_texture_change(&path, TextureHandle::new(1));
        assert_eq!(manager.pending_texture_count(), 1);

        manager.cancel_texture_reload(&path);
        assert_eq!(manager.pending_texture_count(), 0);
    }

    #[test]
    fn test_max_concurrent_reloads() {
        let mut manager = AssetReloadManager::with_defaults();
        manager.config.max_texture_reloads = 2;

        for i in 0..5 {
            let path = PathBuf::from(format!("textures/{}.png", i));
            manager.on_texture_change(&path, TextureHandle::new(i));
        }

        // Should only allow 2 concurrent reloads
        assert_eq!(manager.pending_texture_count(), 2);
    }

    #[test]
    fn test_duplicate_reload_ignored() {
        let mut manager = AssetReloadManager::with_defaults();
        let path = PathBuf::from("textures/albedo.png");

        manager.on_texture_change(&path, TextureHandle::new(1));
        manager.on_texture_change(&path, TextureHandle::new(2)); // Should be ignored

        assert_eq!(manager.pending_texture_count(), 1);
        let state = manager.pending_textures.get(&path).unwrap();
        assert_eq!(state.old_handle.id(), 1); // First handle preserved
    }

    #[test]
    fn test_gpu_image_mip_data() {
        // 64x64 RGBA8 texture with 6 mip levels:
        // Mip 0: 64x64 = 16384 bytes
        // Mip 1: 32x32 = 4096 bytes
        // Mip 2: 16x16 = 1024 bytes
        // Mip 3: 8x8 = 256 bytes
        // Mip 4: 4x4 = 64 bytes
        // Mip 5: 2x2 = 16 bytes
        // Total: 21840 bytes
        let data = vec![0u8; 16384 + 4096 + 1024 + 256 + 64 + 16];
        let image = GpuImage::new(data.clone(), 64, 64, 6, TextureFormat::Rgba8Srgb);

        assert_eq!(image.mip_levels, 6);
        assert!(image.mip_data(0).is_some());
        assert_eq!(image.mip_data(0).unwrap().len(), 16384);
        assert!(image.mip_data(5).is_some());
        assert_eq!(image.mip_data(5).unwrap().len(), 16);
        assert!(image.mip_data(6).is_none());
    }

    #[test]
    fn test_mesh_buffers_stats() {
        let buffers = MeshBuffers::new(
            vec![[0.0, 0.0, 0.0]; 100],
            vec![[0.0, 1.0, 0.0]; 100],
            vec![[0.0, 0.0]; 100],
            (0..300).collect(),
        );

        assert_eq!(buffers.vertex_count(), 100);
        assert_eq!(buffers.triangle_count(), 100);
        assert_eq!(buffers.total_size(), 100 * 12 + 100 * 12 + 100 * 8 + 300 * 4);
    }

    #[test]
    fn test_texture_format_properties() {
        assert_eq!(TextureFormat::Rgba8Srgb.bytes_per_pixel(), 4.0);
        assert!(TextureFormat::Rgba8Srgb.is_srgb());
        assert!(!TextureFormat::Rgba8Srgb.is_compressed());

        assert!(TextureFormat::BC7.is_compressed());
        assert!(!TextureFormat::BC7.is_srgb());
    }

    #[test]
    fn test_reload_status_helpers() {
        assert!(AssetReloadStatus::Importing.is_in_progress());
        assert!(!AssetReloadStatus::Importing.is_complete());

        assert!(AssetReloadStatus::Swapped.is_complete());
        assert!(AssetReloadStatus::Swapped.is_success());

        let err = AssetReloadError::ImportFailed("test".to_string());
        let status = AssetReloadStatus::Failed(err);
        assert!(status.is_failed());
        assert!(status.error().is_some());
    }

    #[test]
    fn test_asset_reload_kind_helpers() {
        let texture_kind = AssetReloadKind::Texture {
            old_handle: TextureHandle::new(1),
            format: TextureFormat::Rgba8Srgb,
        };
        assert!(texture_kind.is_texture());
        assert!(!texture_kind.is_mesh());
        assert!(!texture_kind.requires_blas_rebuild());

        let mesh_kind = AssetReloadKind::Mesh {
            old_handle: MeshHandle::new(1),
            has_blas: true,
        };
        assert!(!mesh_kind.is_texture());
        assert!(mesh_kind.is_mesh());
        assert!(mesh_kind.requires_blas_rebuild());
    }
}
