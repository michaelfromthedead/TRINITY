//! Virtual Texture Streaming System (T-ENV-2.13)
//!
//! Implements a streaming system for virtual texturing that coordinates page
//! loading, eviction, and GPU upload. Integrates with the page table, physical
//! atlas, and feedback system to provide seamless texture streaming.
//!
//! # Architecture
//!
//! - **StreamingConfig**: GPU-compatible configuration for streaming parameters
//! - **StreamingState**: Tracks pending uploads, evictions, and frame budgets
//! - **VirtualTextureStreamer**: Main orchestrator for page streaming
//! - **PageData**: Container for tile pixel data ready for upload
//! - **AsyncLoadHandle**: Non-blocking load coordination
//!
//! # Workflow
//!
//! 1. GPU feedback pass identifies visible pages
//! 2. `process_feedback()` decodes feedback into page requests
//! 3. `prioritize_requests()` sorts by camera distance and mip level
//! 4. `upload_pages()` streams highest priority pages to GPU (within budget)
//! 5. `evict_lru_pages()` frees slots when atlas is full
//! 6. `update_page_table()` updates indirection table for shaders
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::virtual_texture_streaming::*;
//!
//! let config = StreamingConfig::default();
//! let mut streamer = VirtualTextureStreamer::new(config);
//!
//! // Each frame:
//! streamer.frame_begin();
//! streamer.process_feedback(&feedback_data);
//! streamer.prioritize_requests([camera_x, camera_y, camera_z]);
//! streamer.upload_pages(&page_data, &gpu_queue);
//! let stats = streamer.frame_end();
//! ```

use bytemuck::{Pod, Zeroable};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::virtual_texture_atlas::{
    AllocationResult, PhysicalAtlas, PhysicalAtlasConfig, TileCoordinate, TileSlot,
};
use crate::virtual_texture_feedback::{FeedbackEntry, FeedbackPass, PageRequest};
use crate::virtual_texture_page_table::{
    PageTableEntry, PageTableFlags, VirtualTextureConfig, VirtualTexturePageTable,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default maximum pages to upload per frame.
pub const DEFAULT_MAX_PAGES_PER_FRAME: u32 = 16;

/// Default prefetch distance in mip levels.
pub const DEFAULT_PREFETCH_DISTANCE: u32 = 2;

/// Default priority bias for camera proximity.
pub const DEFAULT_PRIORITY_BIAS: f32 = 1.0;

/// Default memory budget in bytes (256 MB).
pub const DEFAULT_BUDGET_BYTES: u32 = 256 * 1024 * 1024;

/// Size of a tile in pixels (128x128 RGBA8).
pub const TILE_PIXEL_SIZE: u32 = 128;

/// Bytes per pixel (RGBA8).
pub const BYTES_PER_PIXEL: u32 = 4;

/// Default tile data size in bytes.
pub const DEFAULT_TILE_BYTES: u32 = TILE_PIXEL_SIZE * TILE_PIXEL_SIZE * BYTES_PER_PIXEL;

// ---------------------------------------------------------------------------
// Texture Format
// ---------------------------------------------------------------------------

/// Supported texture formats for page data.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TextureFormat {
    /// RGBA 8-bit unsigned normalized.
    #[default]
    Rgba8Unorm = 0,
    /// RGBA 8-bit sRGB.
    Rgba8UnormSrgb = 1,
    /// BC1/DXT1 compressed (RGB).
    Bc1Unorm = 2,
    /// BC3/DXT5 compressed (RGBA).
    Bc3Unorm = 3,
    /// BC4 compressed (R).
    Bc4Unorm = 4,
    /// BC5 compressed (RG).
    Bc5Unorm = 5,
    /// BC7 compressed (RGBA).
    Bc7Unorm = 6,
}

impl TextureFormat {
    /// Get bytes per pixel (or per block for compressed formats).
    pub fn bytes_per_unit(&self) -> u32 {
        match self {
            Self::Rgba8Unorm | Self::Rgba8UnormSrgb => 4,
            Self::Bc1Unorm => 8,  // 8 bytes per 4x4 block
            Self::Bc3Unorm => 16, // 16 bytes per 4x4 block
            Self::Bc4Unorm => 8,
            Self::Bc5Unorm => 16,
            Self::Bc7Unorm => 16,
        }
    }

    /// Check if format is block-compressed.
    pub fn is_compressed(&self) -> bool {
        !matches!(self, Self::Rgba8Unorm | Self::Rgba8UnormSrgb)
    }

    /// Get block size (4 for compressed, 1 for uncompressed).
    pub fn block_size(&self) -> u32 {
        if self.is_compressed() {
            4
        } else {
            1
        }
    }

    /// Calculate data size for given dimensions.
    pub fn data_size(&self, width: u32, height: u32) -> u32 {
        if self.is_compressed() {
            let blocks_x = (width + 3) / 4;
            let blocks_y = (height + 3) / 4;
            blocks_x * blocks_y * self.bytes_per_unit()
        } else {
            width * height * self.bytes_per_unit()
        }
    }
}

// ---------------------------------------------------------------------------
// Streaming Configuration
// ---------------------------------------------------------------------------

/// Configuration for the virtual texture streaming system (GPU-compatible).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct StreamingConfig {
    /// Maximum number of pages to upload per frame.
    pub max_pages_per_frame: u32,
    /// Number of mip levels to prefetch ahead.
    pub prefetch_distance: u32,
    /// Weight for camera proximity in priority calculation.
    pub priority_bias: f32,
    /// Memory budget in bytes.
    pub budget_bytes: u32,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            max_pages_per_frame: DEFAULT_MAX_PAGES_PER_FRAME,
            prefetch_distance: DEFAULT_PREFETCH_DISTANCE,
            priority_bias: DEFAULT_PRIORITY_BIAS,
            budget_bytes: DEFAULT_BUDGET_BYTES,
        }
    }
}

impl StreamingConfig {
    /// Create a new streaming configuration.
    pub fn new(
        max_pages_per_frame: u32,
        prefetch_distance: u32,
        priority_bias: f32,
        budget_bytes: u32,
    ) -> Self {
        Self {
            max_pages_per_frame,
            prefetch_distance,
            priority_bias,
            budget_bytes,
        }
    }

    /// Create a minimal configuration for testing.
    pub fn minimal() -> Self {
        Self {
            max_pages_per_frame: 4,
            prefetch_distance: 1,
            priority_bias: 1.0,
            budget_bytes: 16 * 1024 * 1024, // 16 MB
        }
    }

    /// Create a high-performance configuration.
    pub fn high_performance() -> Self {
        Self {
            max_pages_per_frame: 64,
            prefetch_distance: 3,
            priority_bias: 1.5,
            budget_bytes: 512 * 1024 * 1024, // 512 MB
        }
    }

    /// Validate configuration values.
    pub fn validate(&self) -> bool {
        self.max_pages_per_frame > 0
            && self.prefetch_distance <= 16
            && self.priority_bias >= 0.0
            && self.budget_bytes > 0
    }
}

// ---------------------------------------------------------------------------
// Page Handle
// ---------------------------------------------------------------------------

/// Handle to a page in the streaming system.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Pod, Zeroable)]
pub struct PageHandle {
    /// Tile X coordinate.
    pub tile_x: u16,
    /// Tile Y coordinate.
    pub tile_y: u16,
    /// Mip level.
    pub mip_level: u8,
    /// Generation counter for validity checking.
    pub generation: u8,
    /// Reserved for future use.
    pub _reserved: u16,
}

impl PageHandle {
    /// Create a new page handle.
    pub fn new(tile_x: u16, tile_y: u16, mip_level: u8) -> Self {
        Self {
            tile_x,
            tile_y,
            mip_level,
            generation: 0,
            _reserved: 0,
        }
    }

    /// Create from TileCoordinate.
    pub fn from_coord(coord: &TileCoordinate) -> Self {
        Self::new(coord.x, coord.y, coord.mip)
    }

    /// Convert to TileCoordinate.
    pub fn to_coord(&self) -> TileCoordinate {
        TileCoordinate::new(self.tile_x, self.tile_y, self.mip_level)
    }

    /// Create an invalid handle.
    pub fn invalid() -> Self {
        Self {
            tile_x: u16::MAX,
            tile_y: u16::MAX,
            mip_level: u8::MAX,
            generation: 0,
            _reserved: 0,
        }
    }

    /// Check if handle is invalid.
    pub fn is_invalid(&self) -> bool {
        self.tile_x == u16::MAX && self.tile_y == u16::MAX
    }

    /// Pack into u64 for compact storage.
    pub fn pack(&self) -> u64 {
        ((self.tile_x as u64) << 48)
            | ((self.tile_y as u64) << 32)
            | ((self.mip_level as u64) << 24)
            | ((self.generation as u64) << 16)
            | (self._reserved as u64)
    }

    /// Unpack from u64.
    pub fn unpack(packed: u64) -> Self {
        Self {
            tile_x: ((packed >> 48) & 0xFFFF) as u16,
            tile_y: ((packed >> 32) & 0xFFFF) as u16,
            mip_level: ((packed >> 24) & 0xFF) as u8,
            generation: ((packed >> 16) & 0xFF) as u8,
            _reserved: (packed & 0xFFFF) as u16,
        }
    }
}

impl Default for PageHandle {
    fn default() -> Self {
        Self::invalid()
    }
}

// ---------------------------------------------------------------------------
// Page Upload
// ---------------------------------------------------------------------------

/// Represents a pending page upload operation.
#[derive(Debug, Clone)]
pub struct PageUpload {
    /// Handle to the page being uploaded.
    pub handle: PageHandle,
    /// Physical slot in the atlas.
    pub slot: TileSlot,
    /// Size of data in bytes.
    pub data_size: u32,
    /// Frame when upload was queued.
    pub frame_queued: u32,
}

impl PageUpload {
    /// Create a new page upload record.
    pub fn new(handle: PageHandle, slot: TileSlot, data_size: u32, frame_queued: u32) -> Self {
        Self {
            handle,
            slot,
            data_size,
            frame_queued,
        }
    }
}

// ---------------------------------------------------------------------------
// Page Data
// ---------------------------------------------------------------------------

/// Container for tile pixel data ready for GPU upload.
#[derive(Debug, Clone)]
pub struct PageData {
    /// Tile X coordinate.
    pub tile_x: u32,
    /// Tile Y coordinate.
    pub tile_y: u32,
    /// Mip level.
    pub mip_level: u32,
    /// Uncompressed pixel data.
    pub pixels: Vec<u8>,
    /// Compressed pixel data (if applicable).
    pub compressed: Vec<u8>,
    /// Texture format.
    pub format: TextureFormat,
}

impl PageData {
    /// Create a new page data container.
    pub fn new(tile_x: u32, tile_y: u32, mip_level: u32, pixels: Vec<u8>, format: TextureFormat) -> Self {
        Self {
            tile_x,
            tile_y,
            mip_level,
            pixels,
            compressed: Vec::new(),
            format,
        }
    }

    /// Create with compressed data.
    pub fn with_compressed(
        tile_x: u32,
        tile_y: u32,
        mip_level: u32,
        compressed: Vec<u8>,
        format: TextureFormat,
    ) -> Self {
        Self {
            tile_x,
            tile_y,
            mip_level,
            pixels: Vec::new(),
            compressed,
            format,
        }
    }

    /// Get the data to upload (compressed if available, otherwise uncompressed).
    pub fn data(&self) -> &[u8] {
        if !self.compressed.is_empty() {
            &self.compressed
        } else {
            &self.pixels
        }
    }

    /// Get data size in bytes.
    pub fn data_size(&self) -> usize {
        if !self.compressed.is_empty() {
            self.compressed.len()
        } else {
            self.pixels.len()
        }
    }

    /// Check if this page has valid data.
    pub fn is_valid(&self) -> bool {
        !self.pixels.is_empty() || !self.compressed.is_empty()
    }

    /// Create a handle for this page.
    pub fn handle(&self) -> PageHandle {
        PageHandle::new(self.tile_x as u16, self.tile_y as u16, self.mip_level as u8)
    }

    /// Create a TileCoordinate for this page.
    pub fn coord(&self) -> TileCoordinate {
        TileCoordinate::from_u32(self.tile_x, self.tile_y, self.mip_level as u8)
    }
}

// ---------------------------------------------------------------------------
// Streaming State
// ---------------------------------------------------------------------------

/// Tracks the state of the streaming system.
#[derive(Debug, Clone)]
pub struct StreamingState {
    /// Pages pending upload to GPU.
    pub pending_uploads: Vec<PageUpload>,
    /// Pages pending eviction.
    pub pending_evictions: Vec<PageHandle>,
    /// Bytes uploaded this frame.
    pub frame_budget_used: u32,
    /// Total pages streamed since start.
    pub total_pages_streamed: u64,
    /// Total bytes streamed since start.
    pub total_bytes_streamed: u64,
    /// Pages evicted this frame.
    pub frame_pages_evicted: u32,
    /// Current frame number.
    pub current_frame: u32,
}

impl StreamingState {
    /// Create a new streaming state.
    pub fn new() -> Self {
        Self {
            pending_uploads: Vec::new(),
            pending_evictions: Vec::new(),
            frame_budget_used: 0,
            total_pages_streamed: 0,
            total_bytes_streamed: 0,
            frame_pages_evicted: 0,
            current_frame: 0,
        }
    }

    /// Reset frame-local state.
    pub fn frame_reset(&mut self) {
        self.pending_uploads.clear();
        self.pending_evictions.clear();
        self.frame_budget_used = 0;
        self.frame_pages_evicted = 0;
    }

    /// Check if budget allows another upload of given size.
    pub fn can_upload(&self, size: u32, budget: u32) -> bool {
        self.frame_budget_used + size <= budget
    }

    /// Record an upload.
    pub fn record_upload(&mut self, upload: PageUpload) {
        let size = upload.data_size;
        self.pending_uploads.push(upload);
        self.frame_budget_used += size;
        self.total_pages_streamed += 1;
        self.total_bytes_streamed += size as u64;
    }

    /// Record an eviction.
    pub fn record_eviction(&mut self, handle: PageHandle) {
        self.pending_evictions.push(handle);
        self.frame_pages_evicted += 1;
    }

    /// Get number of pending uploads.
    pub fn pending_upload_count(&self) -> usize {
        self.pending_uploads.len()
    }

    /// Get number of pending evictions.
    pub fn pending_eviction_count(&self) -> usize {
        self.pending_evictions.len()
    }
}

impl Default for StreamingState {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Streaming Statistics
// ---------------------------------------------------------------------------

/// Statistics about the streaming system.
#[derive(Debug, Clone, Copy, Default)]
pub struct StreamingStats {
    /// Current frame number.
    pub frame: u32,
    /// Pages uploaded this frame.
    pub pages_uploaded: u32,
    /// Pages evicted this frame.
    pub pages_evicted: u32,
    /// Bytes uploaded this frame.
    pub bytes_uploaded: u32,
    /// Total pages streamed since start.
    pub total_pages_streamed: u64,
    /// Total bytes streamed since start.
    pub total_bytes_streamed: u64,
    /// Number of pending page requests.
    pub pending_requests: u32,
    /// Atlas occupancy percentage.
    pub atlas_occupancy: f32,
    /// Number of async loads in progress.
    pub async_loads_pending: u32,
}

// ---------------------------------------------------------------------------
// Async Load Handle
// ---------------------------------------------------------------------------

/// Handle for tracking asynchronous page loads.
#[derive(Debug)]
pub struct AsyncLoadHandle {
    /// Unique ID for this load operation.
    pub id: u64,
    /// Page being loaded.
    pub handle: PageHandle,
    /// Whether load is complete.
    pub completed: Arc<std::sync::atomic::AtomicBool>,
    /// Frame when load was scheduled.
    pub frame_scheduled: u32,
}

impl AsyncLoadHandle {
    /// Create a new async load handle.
    pub fn new(id: u64, handle: PageHandle, frame: u32) -> Self {
        Self {
            id,
            handle,
            completed: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            frame_scheduled: frame,
        }
    }

    /// Check if load is complete.
    pub fn is_complete(&self) -> bool {
        self.completed.load(Ordering::Acquire)
    }

    /// Mark load as complete.
    pub fn mark_complete(&self) {
        self.completed.store(true, Ordering::Release);
    }
}

// ---------------------------------------------------------------------------
// GPU Queue (Mock for API compatibility)
// ---------------------------------------------------------------------------

/// Mock GPU queue for upload operations.
/// In a real implementation, this would wrap wgpu::Queue or similar.
pub struct GpuQueue {
    /// Number of uploads performed.
    pub uploads_performed: u32,
    /// Total bytes uploaded.
    pub bytes_uploaded: u64,
}

impl GpuQueue {
    /// Create a new mock GPU queue.
    pub fn new() -> Self {
        Self {
            uploads_performed: 0,
            bytes_uploaded: 0,
        }
    }

    /// Simulate uploading data.
    pub fn upload(&mut self, _data: &[u8], size: usize) {
        self.uploads_performed += 1;
        self.bytes_uploaded += size as u64;
    }
}

impl Default for GpuQueue {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Virtual Texture Streamer
// ---------------------------------------------------------------------------

/// The main virtual texture streaming system.
///
/// Orchestrates page loading, eviction, and GPU upload operations.
pub struct VirtualTextureStreamer {
    /// Configuration.
    pub config: StreamingConfig,
    /// Current streaming state.
    pub state: StreamingState,
    /// Prioritized page requests.
    requests: Vec<PageRequest>,
    /// Pages currently being loaded asynchronously.
    async_loads: HashMap<u64, AsyncLoadHandle>,
    /// Completed async loads ready for upload.
    completed_loads: Vec<(u64, PageData)>,
    /// Next async load ID.
    next_load_id: AtomicU64,
    /// Resident page set (tile_x, tile_y, mip) for quick lookup.
    resident_pages: HashSet<(u32, u32, u32)>,
}

impl VirtualTextureStreamer {
    /// Create a new virtual texture streamer.
    pub fn new(config: StreamingConfig) -> Self {
        Self {
            config,
            state: StreamingState::new(),
            requests: Vec::with_capacity(256),
            async_loads: HashMap::new(),
            completed_loads: Vec::new(),
            next_load_id: AtomicU64::new(1),
            resident_pages: HashSet::new(),
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(StreamingConfig::default())
    }

    // -----------------------------------------------------------------------
    // Request Management
    // -----------------------------------------------------------------------

    /// Queue pages for streaming.
    pub fn request_pages(&mut self, requests: &[PageRequest]) {
        for request in requests {
            // Skip if already resident
            if self.resident_pages.contains(&request.key()) {
                continue;
            }
            self.requests.push(*request);
        }
    }

    /// Process feedback buffer to generate page requests.
    pub fn process_feedback(&mut self, feedback: &[u32]) {
        // Decode feedback entries
        for &value in feedback {
            if !FeedbackPass::is_valid_feedback(value) {
                continue;
            }

            let entry = FeedbackPass::decode_feedback(value);
            if entry.is_invalid() {
                continue;
            }

            let key = (entry.tile_x as u32, entry.tile_y as u32, entry.mip_level as u32);

            // Skip if already resident
            if self.resident_pages.contains(&key) {
                continue;
            }

            self.requests.push(PageRequest::from_entry(&entry));
        }
    }

    /// Prioritize requests based on camera position.
    pub fn prioritize_requests(&mut self, camera_pos: [f32; 3]) {
        // Compute priority for each request
        for request in &mut self.requests {
            let base_priority = request.priority;

            // Distance-based priority
            let tile_world_x = request.tile_x as f32 * TILE_PIXEL_SIZE as f32;
            let tile_world_z = request.tile_y as f32 * TILE_PIXEL_SIZE as f32;

            let dx = tile_world_x - camera_pos[0];
            let dz = tile_world_z - camera_pos[2];
            let distance = (dx * dx + dz * dz).sqrt();

            // Exponential falloff for distance
            let distance_factor = (-distance / 10000.0 * self.config.priority_bias).exp();

            // Mip level factor (lower mip = higher priority)
            let mip_factor = 1.0 - (request.mip_level as f32 / 16.0);

            request.priority = (base_priority * 0.3 + distance_factor * 0.5 + mip_factor * 0.2)
                .clamp(0.0, 1.0);
        }

        // Sort by priority (highest first)
        self.requests.sort();

        // Deduplicate by key (keep highest priority)
        let mut seen = HashSet::new();
        self.requests.retain(|r| seen.insert(r.key()));
    }

    /// Get current requests.
    pub fn requests(&self) -> &[PageRequest] {
        &self.requests
    }

    /// Clear all pending requests.
    pub fn clear_requests(&mut self) {
        self.requests.clear();
    }

    // -----------------------------------------------------------------------
    // Upload Operations
    // -----------------------------------------------------------------------

    /// Upload pages to GPU.
    ///
    /// Respects the per-frame budget and returns the number of pages uploaded.
    pub fn upload_pages(&mut self, pages: &[PageData], gpu_queue: &mut GpuQueue) -> u32 {
        let mut uploaded = 0;
        let budget = self.config.max_pages_per_frame as usize * DEFAULT_TILE_BYTES as usize;

        for page in pages {
            // Check frame budget
            if !self.state.can_upload(page.data_size() as u32, budget as u32) {
                break;
            }

            // Check per-frame limit
            if uploaded >= self.config.max_pages_per_frame {
                break;
            }

            // Simulate GPU upload
            gpu_queue.upload(page.data(), page.data_size());

            // Record the upload
            let upload = PageUpload::new(
                page.handle(),
                TileSlot::new(0, 0), // Would be provided by atlas
                page.data_size() as u32,
                self.state.current_frame,
            );
            self.state.record_upload(upload);

            // Mark as resident
            self.resident_pages.insert((page.tile_x, page.tile_y, page.mip_level));

            uploaded += 1;
        }

        uploaded
    }

    /// Upload pages with atlas integration.
    pub fn upload_pages_with_atlas(
        &mut self,
        pages: &[PageData],
        atlas: &mut PhysicalAtlas,
        page_table: &mut VirtualTexturePageTable,
        gpu_queue: &mut GpuQueue,
    ) -> u32 {
        let mut uploaded = 0;
        let budget = self.config.max_pages_per_frame as usize * DEFAULT_TILE_BYTES as usize;

        for page in pages {
            // Check frame budget
            if !self.state.can_upload(page.data_size() as u32, budget as u32) {
                break;
            }

            // Check per-frame limit
            if uploaded >= self.config.max_pages_per_frame {
                break;
            }

            // Allocate slot in atlas
            let coord = page.coord();
            let result = atlas.allocate_tile(coord);

            let slot = match result {
                AllocationResult::Allocated(slot) => slot,
                AllocationResult::AlreadyResident(slot) => slot,
                AllocationResult::Evicted { slot, evicted_coord } => {
                    // Mark evicted page as non-resident
                    self.resident_pages.remove(&(
                        evicted_coord.x as u32,
                        evicted_coord.y as u32,
                        evicted_coord.mip as u32,
                    ));
                    page_table.invalidate(
                        evicted_coord.x as u32,
                        evicted_coord.y as u32,
                        evicted_coord.mip as u32,
                    );
                    self.state.record_eviction(PageHandle::from_coord(&evicted_coord));
                    slot
                }
                AllocationResult::Failed => continue,
            };

            // Simulate GPU upload
            gpu_queue.upload(page.data(), page.data_size());

            // Update page table
            page_table.mark_resident(
                page.tile_x,
                page.tile_y,
                page.mip_level,
                slot.physical_x,
                slot.physical_y,
            );

            // Record the upload
            let upload = PageUpload::new(
                page.handle(),
                slot,
                page.data_size() as u32,
                self.state.current_frame,
            );
            self.state.record_upload(upload);

            // Mark as resident
            self.resident_pages.insert((page.tile_x, page.tile_y, page.mip_level));

            uploaded += 1;
        }

        uploaded
    }

    // -----------------------------------------------------------------------
    // Eviction
    // -----------------------------------------------------------------------

    /// Evict least-recently-used pages from the atlas.
    pub fn evict_lru_pages(&mut self, atlas: &mut PhysicalAtlas, page_table: &mut VirtualTexturePageTable, count: u32) -> u32 {
        let mut evicted = 0;

        for _ in 0..count {
            if let Some((_, coord)) = atlas.evict_lru() {
                // Update page table
                page_table.invalidate(coord.x as u32, coord.y as u32, coord.mip as u32);

                // Remove from resident set
                self.resident_pages.remove(&(coord.x as u32, coord.y as u32, coord.mip as u32));

                // Record eviction
                self.state.record_eviction(PageHandle::from_coord(&coord));

                evicted += 1;
            } else {
                break;
            }
        }

        evicted
    }

    // -----------------------------------------------------------------------
    // Page Table Updates
    // -----------------------------------------------------------------------

    /// Update page table entries for completed uploads.
    pub fn update_page_table(&self, page_table: &mut VirtualTexturePageTable, uploads: &[PageUpload]) {
        for upload in uploads {
            if upload.handle.is_invalid() {
                continue;
            }

            page_table.mark_resident(
                upload.handle.tile_x as u32,
                upload.handle.tile_y as u32,
                upload.handle.mip_level as u32,
                upload.slot.physical_x,
                upload.slot.physical_y,
            );
        }
    }

    // -----------------------------------------------------------------------
    // Async Loading
    // -----------------------------------------------------------------------

    /// Schedule an asynchronous page load.
    pub fn schedule_async_load(&mut self, handle: PageHandle) -> u64 {
        let id = self.next_load_id.fetch_add(1, Ordering::Relaxed);
        let async_handle = AsyncLoadHandle::new(id, handle, self.state.current_frame);
        self.async_loads.insert(id, async_handle);
        id
    }

    /// Poll for completed async loads.
    pub fn poll_completed(&mut self) -> Vec<u64> {
        let completed: Vec<u64> = self
            .async_loads
            .iter()
            .filter(|(_, h)| h.is_complete())
            .map(|(id, _)| *id)
            .collect();

        for id in &completed {
            self.async_loads.remove(id);
        }

        completed
    }

    /// Mark an async load as complete with data.
    pub fn complete_async_load(&mut self, id: u64, data: PageData) {
        if let Some(handle) = self.async_loads.get(&id) {
            handle.mark_complete();
        }
        self.completed_loads.push((id, data));
    }

    /// Get completed loads ready for upload.
    pub fn take_completed_loads(&mut self) -> Vec<(u64, PageData)> {
        std::mem::take(&mut self.completed_loads)
    }

    /// Get the number of pending async loads.
    pub fn async_load_count(&self) -> usize {
        self.async_loads.len()
    }

    // -----------------------------------------------------------------------
    // Frame Management
    // -----------------------------------------------------------------------

    /// Begin a new frame.
    pub fn frame_begin(&mut self) {
        self.state.frame_reset();
        self.requests.clear();
    }

    /// End the current frame.
    pub fn frame_end(&mut self) -> StreamingStats {
        let stats = self.get_stats();
        self.state.current_frame = self.state.current_frame.wrapping_add(1);
        stats
    }

    /// Get current statistics.
    pub fn get_stats(&self) -> StreamingStats {
        StreamingStats {
            frame: self.state.current_frame,
            pages_uploaded: self.state.pending_upload_count() as u32,
            pages_evicted: self.state.frame_pages_evicted,
            bytes_uploaded: self.state.frame_budget_used,
            total_pages_streamed: self.state.total_pages_streamed,
            total_bytes_streamed: self.state.total_bytes_streamed,
            pending_requests: self.requests.len() as u32,
            atlas_occupancy: 0.0, // Would be computed from atlas
            async_loads_pending: self.async_loads.len() as u32,
        }
    }

    // -----------------------------------------------------------------------
    // Resident Page Management
    // -----------------------------------------------------------------------

    /// Check if a page is resident.
    pub fn is_resident(&self, tile_x: u32, tile_y: u32, mip_level: u32) -> bool {
        self.resident_pages.contains(&(tile_x, tile_y, mip_level))
    }

    /// Mark a page as resident.
    pub fn mark_resident(&mut self, tile_x: u32, tile_y: u32, mip_level: u32) {
        self.resident_pages.insert((tile_x, tile_y, mip_level));
    }

    /// Mark a page as non-resident.
    pub fn mark_non_resident(&mut self, tile_x: u32, tile_y: u32, mip_level: u32) {
        self.resident_pages.remove(&(tile_x, tile_y, mip_level));
    }

    /// Get the number of resident pages.
    pub fn resident_count(&self) -> usize {
        self.resident_pages.len()
    }

    /// Clear all resident pages.
    pub fn clear_resident(&mut self) {
        self.resident_pages.clear();
    }

    /// Reset the streamer to initial state.
    pub fn reset(&mut self) {
        self.state = StreamingState::new();
        self.requests.clear();
        self.async_loads.clear();
        self.completed_loads.clear();
        self.resident_pages.clear();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // TextureFormat Tests
    // =========================================================================

    #[test]
    fn test_texture_format_default() {
        let format = TextureFormat::default();
        assert_eq!(format, TextureFormat::Rgba8Unorm);
    }

    #[test]
    fn test_texture_format_bytes_per_unit() {
        assert_eq!(TextureFormat::Rgba8Unorm.bytes_per_unit(), 4);
        assert_eq!(TextureFormat::Rgba8UnormSrgb.bytes_per_unit(), 4);
        assert_eq!(TextureFormat::Bc1Unorm.bytes_per_unit(), 8);
        assert_eq!(TextureFormat::Bc3Unorm.bytes_per_unit(), 16);
        assert_eq!(TextureFormat::Bc4Unorm.bytes_per_unit(), 8);
        assert_eq!(TextureFormat::Bc5Unorm.bytes_per_unit(), 16);
        assert_eq!(TextureFormat::Bc7Unorm.bytes_per_unit(), 16);
    }

    #[test]
    fn test_texture_format_is_compressed() {
        assert!(!TextureFormat::Rgba8Unorm.is_compressed());
        assert!(!TextureFormat::Rgba8UnormSrgb.is_compressed());
        assert!(TextureFormat::Bc1Unorm.is_compressed());
        assert!(TextureFormat::Bc3Unorm.is_compressed());
        assert!(TextureFormat::Bc7Unorm.is_compressed());
    }

    #[test]
    fn test_texture_format_block_size() {
        assert_eq!(TextureFormat::Rgba8Unorm.block_size(), 1);
        assert_eq!(TextureFormat::Bc1Unorm.block_size(), 4);
        assert_eq!(TextureFormat::Bc7Unorm.block_size(), 4);
    }

    #[test]
    fn test_texture_format_data_size_uncompressed() {
        let format = TextureFormat::Rgba8Unorm;
        assert_eq!(format.data_size(128, 128), 128 * 128 * 4);
        assert_eq!(format.data_size(64, 64), 64 * 64 * 4);
    }

    #[test]
    fn test_texture_format_data_size_compressed() {
        let format = TextureFormat::Bc1Unorm;
        // 128x128 = 32x32 blocks * 8 bytes = 8192
        assert_eq!(format.data_size(128, 128), 32 * 32 * 8);
    }

    #[test]
    fn test_texture_format_data_size_non_block_aligned() {
        let format = TextureFormat::Bc1Unorm;
        // 130x130 -> ceil(130/4) = 33 blocks each dimension
        assert_eq!(format.data_size(130, 130), 33 * 33 * 8);
    }

    // =========================================================================
    // StreamingConfig Tests
    // =========================================================================

    #[test]
    fn test_streaming_config_default() {
        let config = StreamingConfig::default();
        assert_eq!(config.max_pages_per_frame, DEFAULT_MAX_PAGES_PER_FRAME);
        assert_eq!(config.prefetch_distance, DEFAULT_PREFETCH_DISTANCE);
        assert!((config.priority_bias - DEFAULT_PRIORITY_BIAS).abs() < 0.001);
        assert_eq!(config.budget_bytes, DEFAULT_BUDGET_BYTES);
    }

    #[test]
    fn test_streaming_config_new() {
        let config = StreamingConfig::new(32, 4, 1.5, 128 * 1024 * 1024);
        assert_eq!(config.max_pages_per_frame, 32);
        assert_eq!(config.prefetch_distance, 4);
        assert!((config.priority_bias - 1.5).abs() < 0.001);
        assert_eq!(config.budget_bytes, 128 * 1024 * 1024);
    }

    #[test]
    fn test_streaming_config_minimal() {
        let config = StreamingConfig::minimal();
        assert_eq!(config.max_pages_per_frame, 4);
        assert_eq!(config.prefetch_distance, 1);
    }

    #[test]
    fn test_streaming_config_high_performance() {
        let config = StreamingConfig::high_performance();
        assert_eq!(config.max_pages_per_frame, 64);
        assert_eq!(config.prefetch_distance, 3);
    }

    #[test]
    fn test_streaming_config_validate_valid() {
        let config = StreamingConfig::default();
        assert!(config.validate());
    }

    #[test]
    fn test_streaming_config_validate_invalid_pages() {
        let config = StreamingConfig::new(0, 2, 1.0, 1024);
        assert!(!config.validate());
    }

    #[test]
    fn test_streaming_config_validate_invalid_prefetch() {
        let config = StreamingConfig::new(16, 20, 1.0, 1024);
        assert!(!config.validate());
    }

    #[test]
    fn test_streaming_config_validate_invalid_bias() {
        let config = StreamingConfig::new(16, 2, -1.0, 1024);
        assert!(!config.validate());
    }

    #[test]
    fn test_streaming_config_validate_invalid_budget() {
        let config = StreamingConfig::new(16, 2, 1.0, 0);
        assert!(!config.validate());
    }

    #[test]
    fn test_streaming_config_pod_zeroable() {
        let config: StreamingConfig = bytemuck::Zeroable::zeroed();
        assert_eq!(config.max_pages_per_frame, 0);
        assert_eq!(config.prefetch_distance, 0);
        assert_eq!(config.priority_bias, 0.0);
        assert_eq!(config.budget_bytes, 0);
    }

    #[test]
    fn test_streaming_config_memory_layout() {
        assert_eq!(std::mem::size_of::<StreamingConfig>(), 16);
        assert_eq!(std::mem::align_of::<StreamingConfig>(), 4);
    }

    // =========================================================================
    // PageHandle Tests
    // =========================================================================

    #[test]
    fn test_page_handle_new() {
        let handle = PageHandle::new(100, 200, 5);
        assert_eq!(handle.tile_x, 100);
        assert_eq!(handle.tile_y, 200);
        assert_eq!(handle.mip_level, 5);
        assert_eq!(handle.generation, 0);
    }

    #[test]
    fn test_page_handle_from_coord() {
        let coord = TileCoordinate::new(50, 75, 3);
        let handle = PageHandle::from_coord(&coord);
        assert_eq!(handle.tile_x, 50);
        assert_eq!(handle.tile_y, 75);
        assert_eq!(handle.mip_level, 3);
    }

    #[test]
    fn test_page_handle_to_coord() {
        let handle = PageHandle::new(50, 75, 3);
        let coord = handle.to_coord();
        assert_eq!(coord.x, 50);
        assert_eq!(coord.y, 75);
        assert_eq!(coord.mip, 3);
    }

    #[test]
    fn test_page_handle_invalid() {
        let handle = PageHandle::invalid();
        assert!(handle.is_invalid());
        assert_eq!(handle.tile_x, u16::MAX);
        assert_eq!(handle.tile_y, u16::MAX);
    }

    #[test]
    fn test_page_handle_default_is_invalid() {
        let handle = PageHandle::default();
        assert!(handle.is_invalid());
    }

    #[test]
    fn test_page_handle_is_invalid_valid_handle() {
        let handle = PageHandle::new(0, 0, 0);
        assert!(!handle.is_invalid());
    }

    #[test]
    fn test_page_handle_pack_unpack() {
        let original = PageHandle::new(1234, 5678, 7);
        let packed = original.pack();
        let unpacked = PageHandle::unpack(packed);

        assert_eq!(unpacked.tile_x, 1234);
        assert_eq!(unpacked.tile_y, 5678);
        assert_eq!(unpacked.mip_level, 7);
    }

    #[test]
    fn test_page_handle_pod_zeroable() {
        let handle: PageHandle = bytemuck::Zeroable::zeroed();
        assert_eq!(handle.tile_x, 0);
        assert_eq!(handle.tile_y, 0);
        assert_eq!(handle.mip_level, 0);
    }

    #[test]
    fn test_page_handle_memory_layout() {
        assert_eq!(std::mem::size_of::<PageHandle>(), 8);
    }

    #[test]
    fn test_page_handle_hash_eq() {
        use std::collections::HashSet;
        let mut set = HashSet::new();

        let h1 = PageHandle::new(10, 20, 3);
        let h2 = PageHandle::new(10, 20, 3);

        set.insert(h1);
        assert!(set.contains(&h2));
    }

    // =========================================================================
    // PageData Tests
    // =========================================================================

    #[test]
    fn test_page_data_new() {
        let pixels = vec![0u8; 1024];
        let data = PageData::new(10, 20, 3, pixels.clone(), TextureFormat::Rgba8Unorm);

        assert_eq!(data.tile_x, 10);
        assert_eq!(data.tile_y, 20);
        assert_eq!(data.mip_level, 3);
        assert_eq!(data.pixels.len(), 1024);
        assert!(data.compressed.is_empty());
        assert_eq!(data.format, TextureFormat::Rgba8Unorm);
    }

    #[test]
    fn test_page_data_with_compressed() {
        let compressed = vec![0u8; 512];
        let data = PageData::with_compressed(10, 20, 3, compressed.clone(), TextureFormat::Bc1Unorm);

        assert!(data.pixels.is_empty());
        assert_eq!(data.compressed.len(), 512);
        assert!(data.format.is_compressed());
    }

    #[test]
    fn test_page_data_data_prefers_compressed() {
        let mut data = PageData::new(0, 0, 0, vec![1u8; 100], TextureFormat::Rgba8Unorm);
        data.compressed = vec![2u8; 50];

        assert_eq!(data.data().len(), 50);
        assert_eq!(data.data()[0], 2);
    }

    #[test]
    fn test_page_data_data_falls_back_to_pixels() {
        let data = PageData::new(0, 0, 0, vec![1u8; 100], TextureFormat::Rgba8Unorm);

        assert_eq!(data.data().len(), 100);
        assert_eq!(data.data()[0], 1);
    }

    #[test]
    fn test_page_data_data_size() {
        let data = PageData::new(0, 0, 0, vec![0u8; 200], TextureFormat::Rgba8Unorm);
        assert_eq!(data.data_size(), 200);
    }

    #[test]
    fn test_page_data_is_valid() {
        let valid = PageData::new(0, 0, 0, vec![0u8; 100], TextureFormat::Rgba8Unorm);
        assert!(valid.is_valid());

        let invalid = PageData {
            tile_x: 0,
            tile_y: 0,
            mip_level: 0,
            pixels: Vec::new(),
            compressed: Vec::new(),
            format: TextureFormat::Rgba8Unorm,
        };
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_page_data_handle() {
        let data = PageData::new(100, 200, 5, vec![], TextureFormat::Rgba8Unorm);
        let handle = data.handle();

        assert_eq!(handle.tile_x, 100);
        assert_eq!(handle.tile_y, 200);
        assert_eq!(handle.mip_level, 5);
    }

    #[test]
    fn test_page_data_coord() {
        let data = PageData::new(100, 200, 5, vec![], TextureFormat::Rgba8Unorm);
        let coord = data.coord();

        assert_eq!(coord.x, 100);
        assert_eq!(coord.y, 200);
        assert_eq!(coord.mip, 5);
    }

    // =========================================================================
    // StreamingState Tests
    // =========================================================================

    #[test]
    fn test_streaming_state_new() {
        let state = StreamingState::new();
        assert!(state.pending_uploads.is_empty());
        assert!(state.pending_evictions.is_empty());
        assert_eq!(state.frame_budget_used, 0);
        assert_eq!(state.total_pages_streamed, 0);
        assert_eq!(state.current_frame, 0);
    }

    #[test]
    fn test_streaming_state_default() {
        let state = StreamingState::default();
        assert_eq!(state.total_pages_streamed, 0);
    }

    #[test]
    fn test_streaming_state_frame_reset() {
        let mut state = StreamingState::new();
        state.frame_budget_used = 1000;
        state.pending_uploads.push(PageUpload::new(
            PageHandle::new(0, 0, 0),
            TileSlot::new(0, 0),
            100,
            0,
        ));

        state.frame_reset();

        assert!(state.pending_uploads.is_empty());
        assert_eq!(state.frame_budget_used, 0);
    }

    #[test]
    fn test_streaming_state_can_upload() {
        let mut state = StreamingState::new();
        assert!(state.can_upload(100, 1000));

        state.frame_budget_used = 900;
        assert!(state.can_upload(100, 1000));
        assert!(!state.can_upload(101, 1000));
    }

    #[test]
    fn test_streaming_state_record_upload() {
        let mut state = StreamingState::new();
        let upload = PageUpload::new(PageHandle::new(0, 0, 0), TileSlot::new(0, 0), 500, 0);

        state.record_upload(upload);

        assert_eq!(state.pending_upload_count(), 1);
        assert_eq!(state.frame_budget_used, 500);
        assert_eq!(state.total_pages_streamed, 1);
        assert_eq!(state.total_bytes_streamed, 500);
    }

    #[test]
    fn test_streaming_state_record_eviction() {
        let mut state = StreamingState::new();
        state.record_eviction(PageHandle::new(0, 0, 0));

        assert_eq!(state.pending_eviction_count(), 1);
        assert_eq!(state.frame_pages_evicted, 1);
    }

    // =========================================================================
    // VirtualTextureStreamer Tests
    // =========================================================================

    #[test]
    fn test_streamer_new() {
        let config = StreamingConfig::default();
        let streamer = VirtualTextureStreamer::new(config);

        assert_eq!(streamer.config.max_pages_per_frame, DEFAULT_MAX_PAGES_PER_FRAME);
        assert!(streamer.requests.is_empty());
        assert_eq!(streamer.resident_count(), 0);
    }

    #[test]
    fn test_streamer_with_defaults() {
        let streamer = VirtualTextureStreamer::with_defaults();
        assert_eq!(streamer.config.max_pages_per_frame, DEFAULT_MAX_PAGES_PER_FRAME);
    }

    #[test]
    fn test_streamer_request_pages() {
        let mut streamer = VirtualTextureStreamer::with_defaults();
        let requests = vec![
            PageRequest::new(0, 0, 0, 1.0),
            PageRequest::new(1, 1, 1, 0.5),
        ];

        streamer.request_pages(&requests);

        assert_eq!(streamer.requests.len(), 2);
    }

    #[test]
    fn test_streamer_request_pages_skips_resident() {
        let mut streamer = VirtualTextureStreamer::with_defaults();
        streamer.mark_resident(0, 0, 0);

        let requests = vec![
            PageRequest::new(0, 0, 0, 1.0), // Already resident
            PageRequest::new(1, 1, 1, 0.5),
        ];

        streamer.request_pages(&requests);

        assert_eq!(streamer.requests.len(), 1);
        assert_eq!(streamer.requests[0].tile_x, 1);
    }

    #[test]
    fn test_streamer_process_feedback() {
        let mut streamer = VirtualTextureStreamer::with_defaults();

        let feedback = vec![
            FeedbackPass::encode_feedback(10, 20, 3),
            FeedbackPass::encode_feedback(30, 40, 1),
        ];

        streamer.process_feedback(&feedback);

        assert_eq!(streamer.requests.len(), 2);
    }

    #[test]
    fn test_streamer_process_feedback_skips_invalid() {
        let mut streamer = VirtualTextureStreamer::with_defaults();

        let feedback = vec![
            FeedbackPass::encode_feedback(10, 20, 3),
            crate::virtual_texture_feedback::INVALID_FEEDBACK,
            FeedbackPass::encode_feedback(30, 40, 1),
        ];

        streamer.process_feedback(&feedback);

        assert_eq!(streamer.requests.len(), 2);
    }

    #[test]
    fn test_streamer_process_feedback_skips_resident() {
        let mut streamer = VirtualTextureStreamer::with_defaults();
        streamer.mark_resident(10, 20, 3);

        let feedback = vec![
            FeedbackPass::encode_feedback(10, 20, 3), // Resident
            FeedbackPass::encode_feedback(30, 40, 1),
        ];

        streamer.process_feedback(&feedback);

        assert_eq!(streamer.requests.len(), 1);
    }

    #[test]
    fn test_streamer_prioritize_requests() {
        let mut streamer = VirtualTextureStreamer::with_defaults();
        let requests = vec![
            PageRequest::new(0, 0, 0, 0.5),
            PageRequest::new(100, 100, 5, 0.5),
        ];

        streamer.request_pages(&requests);
        streamer.prioritize_requests([0.0, 0.0, 0.0]); // Camera at origin

        // Closer tile (0,0,0) should have higher priority
        assert_eq!(streamer.requests[0].tile_x, 0);
    }

    #[test]
    fn test_streamer_prioritize_requests_deduplicates() {
        let mut streamer = VirtualTextureStreamer::with_defaults();
        let requests = vec![
            PageRequest::new(0, 0, 0, 0.5),
            PageRequest::new(0, 0, 0, 0.8), // Duplicate
            PageRequest::new(1, 1, 1, 0.5),
        ];

        streamer.request_pages(&requests);
        streamer.prioritize_requests([0.0, 0.0, 0.0]);

        assert_eq!(streamer.requests.len(), 2);
    }

    #[test]
    fn test_streamer_upload_pages() {
        let mut streamer = VirtualTextureStreamer::with_defaults();
        let mut gpu_queue = GpuQueue::new();

        let pages = vec![
            PageData::new(0, 0, 0, vec![0u8; 1000], TextureFormat::Rgba8Unorm),
            PageData::new(1, 1, 1, vec![0u8; 1000], TextureFormat::Rgba8Unorm),
        ];

        let uploaded = streamer.upload_pages(&pages, &mut gpu_queue);

        assert_eq!(uploaded, 2);
        assert_eq!(gpu_queue.uploads_performed, 2);
        assert!(streamer.is_resident(0, 0, 0));
        assert!(streamer.is_resident(1, 1, 1));
    }

    #[test]
    fn test_streamer_upload_pages_respects_limit() {
        let config = StreamingConfig::new(2, 2, 1.0, 1024 * 1024);
        let mut streamer = VirtualTextureStreamer::new(config);
        let mut gpu_queue = GpuQueue::new();

        let pages: Vec<_> = (0..5)
            .map(|i| PageData::new(i, 0, 0, vec![0u8; 100], TextureFormat::Rgba8Unorm))
            .collect();

        let uploaded = streamer.upload_pages(&pages, &mut gpu_queue);

        assert_eq!(uploaded, 2);
    }

    #[test]
    fn test_streamer_upload_pages_respects_budget() {
        let config = StreamingConfig::new(100, 2, 1.0, 1024 * 1024);
        let mut streamer = VirtualTextureStreamer::new(config);
        let mut gpu_queue = GpuQueue::new();

        // Create pages larger than budget allows
        let pages: Vec<_> = (0..10)
            .map(|i| PageData::new(i, 0, 0, vec![0u8; DEFAULT_TILE_BYTES as usize], TextureFormat::Rgba8Unorm))
            .collect();

        let uploaded = streamer.upload_pages(&pages, &mut gpu_queue);

        // Should stop when budget is exhausted
        assert!(uploaded <= config.max_pages_per_frame);
    }

    #[test]
    fn test_streamer_frame_begin_end() {
        let mut streamer = VirtualTextureStreamer::with_defaults();
        let requests = vec![PageRequest::new(0, 0, 0, 1.0)];
        streamer.request_pages(&requests);

        streamer.frame_begin();

        assert!(streamer.requests.is_empty());

        let stats = streamer.frame_end();
        assert_eq!(stats.frame, 0);
    }

    #[test]
    fn test_streamer_frame_counter() {
        let mut streamer = VirtualTextureStreamer::with_defaults();

        assert_eq!(streamer.state.current_frame, 0);
        streamer.frame_end();
        assert_eq!(streamer.state.current_frame, 1);
        streamer.frame_end();
        assert_eq!(streamer.state.current_frame, 2);
    }

    #[test]
    fn test_streamer_get_stats() {
        let mut streamer = VirtualTextureStreamer::with_defaults();
        let mut gpu_queue = GpuQueue::new();

        let pages = vec![
            PageData::new(0, 0, 0, vec![0u8; 1000], TextureFormat::Rgba8Unorm),
        ];
        streamer.upload_pages(&pages, &mut gpu_queue);

        let stats = streamer.get_stats();

        assert_eq!(stats.pages_uploaded, 1);
        assert_eq!(stats.bytes_uploaded, 1000);
        assert_eq!(stats.total_pages_streamed, 1);
    }

    #[test]
    fn test_streamer_resident_management() {
        let mut streamer = VirtualTextureStreamer::with_defaults();

        assert!(!streamer.is_resident(0, 0, 0));
        assert_eq!(streamer.resident_count(), 0);

        streamer.mark_resident(0, 0, 0);
        assert!(streamer.is_resident(0, 0, 0));
        assert_eq!(streamer.resident_count(), 1);

        streamer.mark_non_resident(0, 0, 0);
        assert!(!streamer.is_resident(0, 0, 0));
        assert_eq!(streamer.resident_count(), 0);
    }

    #[test]
    fn test_streamer_clear_resident() {
        let mut streamer = VirtualTextureStreamer::with_defaults();
        streamer.mark_resident(0, 0, 0);
        streamer.mark_resident(1, 1, 1);

        streamer.clear_resident();

        assert_eq!(streamer.resident_count(), 0);
    }

    #[test]
    fn test_streamer_reset() {
        let mut streamer = VirtualTextureStreamer::with_defaults();
        let mut gpu_queue = GpuQueue::new();

        let pages = vec![
            PageData::new(0, 0, 0, vec![0u8; 1000], TextureFormat::Rgba8Unorm),
        ];
        streamer.upload_pages(&pages, &mut gpu_queue);

        streamer.reset();

        assert_eq!(streamer.state.total_pages_streamed, 0);
        assert!(streamer.requests.is_empty());
        assert_eq!(streamer.resident_count(), 0);
    }

    // =========================================================================
    // Async Load Tests
    // =========================================================================

    #[test]
    fn test_async_load_schedule() {
        let mut streamer = VirtualTextureStreamer::with_defaults();
        let handle = PageHandle::new(0, 0, 0);

        let id = streamer.schedule_async_load(handle);

        assert!(id > 0);
        assert_eq!(streamer.async_load_count(), 1);
    }

    #[test]
    fn test_async_load_complete() {
        let mut streamer = VirtualTextureStreamer::with_defaults();
        let handle = PageHandle::new(0, 0, 0);
        let id = streamer.schedule_async_load(handle);

        let data = PageData::new(0, 0, 0, vec![0u8; 100], TextureFormat::Rgba8Unorm);
        streamer.complete_async_load(id, data);

        let completed = streamer.poll_completed();
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0], id);
    }

    #[test]
    fn test_async_load_take_completed() {
        let mut streamer = VirtualTextureStreamer::with_defaults();
        let handle = PageHandle::new(0, 0, 0);
        let id = streamer.schedule_async_load(handle);

        let data = PageData::new(0, 0, 0, vec![0u8; 100], TextureFormat::Rgba8Unorm);
        streamer.complete_async_load(id, data);

        let completed = streamer.take_completed_loads();
        assert_eq!(completed.len(), 1);

        // Should be empty after taking
        let empty = streamer.take_completed_loads();
        assert!(empty.is_empty());
    }

    #[test]
    fn test_async_load_poll_incomplete() {
        let mut streamer = VirtualTextureStreamer::with_defaults();
        let handle = PageHandle::new(0, 0, 0);
        streamer.schedule_async_load(handle);

        // Not yet complete
        let completed = streamer.poll_completed();
        assert!(completed.is_empty());
        assert_eq!(streamer.async_load_count(), 1);
    }

    // =========================================================================
    // Integration Tests
    // =========================================================================

    #[test]
    fn test_full_streaming_workflow() {
        let config = StreamingConfig::new(8, 2, 1.0, 1024 * 1024);
        let mut streamer = VirtualTextureStreamer::new(config);
        let mut gpu_queue = GpuQueue::new();

        // Frame 1: Request pages from feedback
        streamer.frame_begin();

        let feedback = vec![
            FeedbackPass::encode_feedback(0, 0, 0),
            FeedbackPass::encode_feedback(1, 0, 0),
            FeedbackPass::encode_feedback(2, 0, 0),
        ];
        streamer.process_feedback(&feedback);
        streamer.prioritize_requests([0.0, 0.0, 0.0]);

        assert_eq!(streamer.requests.len(), 3);

        // Upload pages
        let pages: Vec<_> = (0..3)
            .map(|i| PageData::new(i, 0, 0, vec![0u8; 1000], TextureFormat::Rgba8Unorm))
            .collect();
        let uploaded = streamer.upload_pages(&pages, &mut gpu_queue);

        assert_eq!(uploaded, 3);

        let stats = streamer.frame_end();
        assert_eq!(stats.pages_uploaded, 3);

        // Frame 2: Same pages should be skipped
        streamer.frame_begin();
        streamer.process_feedback(&feedback);

        // All pages already resident
        assert_eq!(streamer.requests.len(), 0);
    }

    #[test]
    fn test_streaming_with_eviction() {
        let atlas_config = PhysicalAtlasConfig::new(256, 64, 2);
        let mut atlas = PhysicalAtlas::new(atlas_config);
        let vt_config = VirtualTextureConfig::new(8192, 64, 128, 2);
        let mut page_table = VirtualTexturePageTable::new(vt_config);

        let config = StreamingConfig::new(16, 2, 1.0, 1024 * 1024);
        let mut streamer = VirtualTextureStreamer::new(config);
        let mut gpu_queue = GpuQueue::new();

        // Fill the atlas
        let total_slots = atlas.capacity();
        let pages: Vec<_> = (0..total_slots)
            .map(|i| PageData::new(i, 0, 0, vec![0u8; 100], TextureFormat::Rgba8Unorm))
            .collect();

        for page in &pages {
            let uploaded = streamer.upload_pages_with_atlas(
                std::slice::from_ref(page),
                &mut atlas,
                &mut page_table,
                &mut gpu_queue,
            );
            assert_eq!(uploaded, 1);
        }

        assert!(atlas.is_full());

        // Upload one more page - should evict
        let new_page = PageData::new(999, 0, 0, vec![0u8; 100], TextureFormat::Rgba8Unorm);
        let uploaded = streamer.upload_pages_with_atlas(
            std::slice::from_ref(&new_page),
            &mut atlas,
            &mut page_table,
            &mut gpu_queue,
        );

        assert_eq!(uploaded, 1);
        assert!(streamer.is_resident(999, 0, 0));
        assert_eq!(streamer.state.frame_pages_evicted, 1);
    }

    #[test]
    fn test_evict_lru_pages() {
        let atlas_config = PhysicalAtlasConfig::new(256, 64, 2);
        let mut atlas = PhysicalAtlas::new(atlas_config);
        let vt_config = VirtualTextureConfig::new(8192, 64, 128, 2);
        let mut page_table = VirtualTexturePageTable::new(vt_config);

        let mut streamer = VirtualTextureStreamer::with_defaults();

        // Add some pages to atlas
        for i in 0..5 {
            let coord = TileCoordinate::new(i, 0, 0);
            atlas.allocate_tile(coord);
            page_table.mark_resident(i as u32, 0, 0, i as u16, 0);
            streamer.mark_resident(i as u32, 0, 0);
        }

        // Evict 2 pages
        let evicted = streamer.evict_lru_pages(&mut atlas, &mut page_table, 2);

        assert_eq!(evicted, 2);
        assert_eq!(streamer.resident_count(), 3);
        assert_eq!(streamer.state.frame_pages_evicted, 2);
    }

    #[test]
    fn test_update_page_table() {
        // Use a larger page table to accommodate the tile coordinates
        let vt_config = VirtualTextureConfig::new(131072, 128, 1024, 2);
        let mut page_table = VirtualTexturePageTable::new(vt_config);
        let streamer = VirtualTextureStreamer::with_defaults();

        let uploads = vec![
            PageUpload::new(
                PageHandle::new(10, 20, 3),
                TileSlot::new(5, 6),
                1000,
                0,
            ),
            PageUpload::new(
                PageHandle::new(30, 40, 1),
                TileSlot::new(7, 8),
                1000,
                0,
            ),
        ];

        streamer.update_page_table(&mut page_table, &uploads);

        let entry1 = page_table.get_entry(10, 20, 3).unwrap();
        assert!(entry1.is_resident());
        assert_eq!(entry1.physical_x, 5);
        assert_eq!(entry1.physical_y, 6);

        let entry2 = page_table.get_entry(30, 40, 1).unwrap();
        assert!(entry2.is_resident());
        assert_eq!(entry2.physical_x, 7);
        assert_eq!(entry2.physical_y, 8);
    }

    #[test]
    fn test_update_page_table_skips_invalid() {
        let vt_config = VirtualTextureConfig::new(8192, 64, 128, 2);
        let mut page_table = VirtualTexturePageTable::new(vt_config);
        let streamer = VirtualTextureStreamer::with_defaults();

        let uploads = vec![
            PageUpload::new(
                PageHandle::invalid(),
                TileSlot::new(0, 0),
                1000,
                0,
            ),
        ];

        streamer.update_page_table(&mut page_table, &uploads);

        // Should not crash, and page table should be unchanged
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_empty_feedback() {
        let mut streamer = VirtualTextureStreamer::with_defaults();
        streamer.process_feedback(&[]);
        assert!(streamer.requests.is_empty());
    }

    #[test]
    fn test_empty_page_upload() {
        let mut streamer = VirtualTextureStreamer::with_defaults();
        let mut gpu_queue = GpuQueue::new();

        let uploaded = streamer.upload_pages(&[], &mut gpu_queue);

        assert_eq!(uploaded, 0);
        assert_eq!(gpu_queue.uploads_performed, 0);
    }

    #[test]
    fn test_clear_requests() {
        let mut streamer = VirtualTextureStreamer::with_defaults();
        let requests = vec![
            PageRequest::new(0, 0, 0, 1.0),
            PageRequest::new(1, 1, 1, 0.5),
        ];
        streamer.request_pages(&requests);

        streamer.clear_requests();

        assert!(streamer.requests.is_empty());
    }

    #[test]
    fn test_frame_counter_wrapping() {
        let mut streamer = VirtualTextureStreamer::with_defaults();
        streamer.state.current_frame = u32::MAX;

        streamer.frame_end();

        assert_eq!(streamer.state.current_frame, 0);
    }

    #[test]
    fn test_multiple_async_loads() {
        let mut streamer = VirtualTextureStreamer::with_defaults();

        let id1 = streamer.schedule_async_load(PageHandle::new(0, 0, 0));
        let id2 = streamer.schedule_async_load(PageHandle::new(1, 1, 1));
        let id3 = streamer.schedule_async_load(PageHandle::new(2, 2, 2));

        assert_eq!(streamer.async_load_count(), 3);
        assert!(id1 != id2 && id2 != id3);

        // Complete only one
        let data = PageData::new(0, 0, 0, vec![0u8; 100], TextureFormat::Rgba8Unorm);
        streamer.complete_async_load(id1, data);

        let completed = streamer.poll_completed();
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0], id1);

        // Two still pending
        assert_eq!(streamer.async_load_count(), 2);
    }

    // =========================================================================
    // GpuQueue Tests
    // =========================================================================

    #[test]
    fn test_gpu_queue_new() {
        let queue = GpuQueue::new();
        assert_eq!(queue.uploads_performed, 0);
        assert_eq!(queue.bytes_uploaded, 0);
    }

    #[test]
    fn test_gpu_queue_upload() {
        let mut queue = GpuQueue::new();
        let data = vec![0u8; 1000];

        queue.upload(&data, data.len());

        assert_eq!(queue.uploads_performed, 1);
        assert_eq!(queue.bytes_uploaded, 1000);
    }

    #[test]
    fn test_gpu_queue_multiple_uploads() {
        let mut queue = GpuQueue::new();

        for i in 0..5 {
            let data = vec![0u8; (i + 1) * 100];
            queue.upload(&data, data.len());
        }

        assert_eq!(queue.uploads_performed, 5);
        assert_eq!(queue.bytes_uploaded, 100 + 200 + 300 + 400 + 500);
    }

    // =========================================================================
    // PageUpload Tests
    // =========================================================================

    #[test]
    fn test_page_upload_new() {
        let handle = PageHandle::new(10, 20, 3);
        let slot = TileSlot::new(5, 6);
        let upload = PageUpload::new(handle, slot, 1000, 42);

        assert_eq!(upload.handle.tile_x, 10);
        assert_eq!(upload.slot.physical_x, 5);
        assert_eq!(upload.data_size, 1000);
        assert_eq!(upload.frame_queued, 42);
    }

    // =========================================================================
    // AsyncLoadHandle Tests
    // =========================================================================

    #[test]
    fn test_async_load_handle_new() {
        let handle = AsyncLoadHandle::new(42, PageHandle::new(0, 0, 0), 10);

        assert_eq!(handle.id, 42);
        assert_eq!(handle.frame_scheduled, 10);
        assert!(!handle.is_complete());
    }

    #[test]
    fn test_async_load_handle_complete() {
        let handle = AsyncLoadHandle::new(1, PageHandle::new(0, 0, 0), 0);

        assert!(!handle.is_complete());
        handle.mark_complete();
        assert!(handle.is_complete());
    }
}
