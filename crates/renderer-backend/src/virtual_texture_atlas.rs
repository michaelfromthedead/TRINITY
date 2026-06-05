//! Virtual Texturing Physical Atlas (T-ENV-2.11)
//!
//! Implements a physical texture atlas for virtual texturing. The atlas stores
//! tile data in a large GPU texture (16K x 16K by default) with tiles arranged
//! in a grid. Supports LRU eviction, free list management, and upload staging.
//!
//! # Architecture
//!
//! - **PhysicalAtlasConfig**: Configuration for atlas dimensions and tile sizes
//! - **TileSlot**: Represents a physical location in the atlas
//! - **TileCoordinate**: Virtual tile identifier (x, y, mip)
//! - **PhysicalAtlas**: Main atlas structure with LRU eviction
//! - **TileUploadRequest**: Staging buffer management for uploads
//!
//! # Memory Layout
//!
//! The atlas is a 16384 x 16384 texel texture divided into tiles:
//! - Tile size: 128 x 128 texels (content)
//! - Border size: 2 texels per edge (for filtering)
//! - Storage per tile: 132 x 132 texels
//! - Tiles per axis: 16384 / 132 = 124 tiles (rounded down)
//! - Total tile slots: 124 x 124 = 15,376 tiles
//!
//! # Mip Chain
//!
//! The atlas supports multiple mip levels stored in the same physical space.
//! Mip levels: log2(16384/4) = 12 levels (from 16384 down to 4 texels)
//!
//! # LRU Eviction
//!
//! Tiles are tracked by frame access time. When the atlas is full, the least
//! recently used tile is evicted to make room for new tiles. Eviction uses a
//! combination of free list and LRU queue for O(1) amortized allocation.

use bytemuck::{Pod, Zeroable};
use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default atlas size in texels (16K x 16K).
pub const DEFAULT_ATLAS_SIZE: u32 = 16384;

/// Default tile size in texels (not including border).
pub const DEFAULT_TILE_SIZE: u32 = 128;

/// Default border size in texels (per edge).
pub const DEFAULT_BORDER_SIZE: u32 = 2;

/// Maximum number of mip levels for 16K atlas.
pub const MAX_MIP_LEVELS: u32 = 13; // log2(16384/4) rounded up

/// Invalid slot index sentinel value.
pub const INVALID_SLOT: u32 = u32::MAX;

/// Maximum tiles that can be uploaded per frame (staging buffer limit).
pub const MAX_UPLOADS_PER_FRAME: usize = 64;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the physical texture atlas.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
pub struct PhysicalAtlasConfig {
    /// Atlas size in texels (e.g., 16384).
    pub atlas_size: u32,
    /// Tile content size in texels (e.g., 128).
    pub tile_size: u32,
    /// Border size per edge in texels (e.g., 2).
    pub border_size: u32,
    /// Number of tiles per axis (computed from other fields).
    pub tiles_per_axis: u32,
}

impl Default for PhysicalAtlasConfig {
    fn default() -> Self {
        Self::new(DEFAULT_ATLAS_SIZE, DEFAULT_TILE_SIZE, DEFAULT_BORDER_SIZE)
    }
}

impl PhysicalAtlasConfig {
    /// Create a new configuration with the given parameters.
    pub fn new(atlas_size: u32, tile_size: u32, border_size: u32) -> Self {
        let tile_size_with_border = tile_size + border_size * 2;
        let tiles_per_axis = if tile_size_with_border > 0 {
            atlas_size / tile_size_with_border
        } else {
            0
        };

        Self {
            atlas_size,
            tile_size,
            border_size,
            tiles_per_axis,
        }
    }

    /// Get the tile size including border texels.
    #[inline]
    pub fn tile_size_with_border(&self) -> u32 {
        self.tile_size + self.border_size * 2
    }

    /// Get the total number of tile slots in the atlas.
    #[inline]
    pub fn total_slots(&self) -> u32 {
        self.tiles_per_axis * self.tiles_per_axis
    }

    /// Get the number of mip levels supported.
    pub fn mip_levels(&self) -> u32 {
        if self.atlas_size == 0 {
            return 0;
        }
        // log2(atlas_size / 4) + 1, minimum of 1
        let min_size = 4u32;
        if self.atlas_size < min_size {
            return 1;
        }
        ((self.atlas_size / min_size) as f32).log2().floor() as u32 + 1
    }

    /// Calculate memory size in bytes for the atlas (RGBA8 format).
    pub fn memory_size_bytes(&self) -> usize {
        (self.atlas_size as usize) * (self.atlas_size as usize) * 4
    }

    /// Calculate memory size in bytes for a single tile with border.
    pub fn tile_memory_size_bytes(&self) -> usize {
        let size = self.tile_size_with_border() as usize;
        size * size * 4
    }

    /// Convert slot index to physical coordinates.
    #[inline]
    pub fn slot_to_coords(&self, slot_index: u32) -> (u32, u32) {
        if self.tiles_per_axis == 0 {
            return (0, 0);
        }
        let x = slot_index % self.tiles_per_axis;
        let y = slot_index / self.tiles_per_axis;
        (x, y)
    }

    /// Convert physical coordinates to slot index.
    #[inline]
    pub fn coords_to_slot(&self, x: u32, y: u32) -> Option<u32> {
        if x >= self.tiles_per_axis || y >= self.tiles_per_axis {
            return None;
        }
        Some(y * self.tiles_per_axis + x)
    }

    /// Convert slot coordinates to texel coordinates (top-left corner).
    #[inline]
    pub fn slot_to_texel(&self, slot_x: u32, slot_y: u32) -> (u32, u32) {
        let tile_with_border = self.tile_size_with_border();
        (slot_x * tile_with_border, slot_y * tile_with_border)
    }

    /// Convert slot coordinates to texel coordinates for content (excluding border).
    #[inline]
    pub fn slot_to_content_texel(&self, slot_x: u32, slot_y: u32) -> (u32, u32) {
        let (texel_x, texel_y) = self.slot_to_texel(slot_x, slot_y);
        (texel_x + self.border_size, texel_y + self.border_size)
    }

    /// Validate that a slot index is within bounds.
    #[inline]
    pub fn is_valid_slot(&self, slot_index: u32) -> bool {
        slot_index < self.total_slots()
    }
}

// ---------------------------------------------------------------------------
// Tile Slot
// ---------------------------------------------------------------------------

/// A single tile slot in the physical atlas.
///
/// Stores the physical location and access tracking information.
/// Packed as 8 bytes for efficient GPU upload.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
pub struct TileSlot {
    /// Physical X coordinate in tile units.
    pub physical_x: u16,
    /// Physical Y coordinate in tile units.
    pub physical_y: u16,
    /// Frame number when last accessed.
    pub frame_accessed: u32,
}

impl TileSlot {
    /// Create a new tile slot with the given coordinates.
    pub fn new(physical_x: u16, physical_y: u16) -> Self {
        Self {
            physical_x,
            physical_y,
            frame_accessed: 0,
        }
    }

    /// Create an invalid/empty tile slot.
    pub fn invalid() -> Self {
        Self {
            physical_x: u16::MAX,
            physical_y: u16::MAX,
            frame_accessed: 0,
        }
    }

    /// Check if this slot is invalid.
    #[inline]
    pub fn is_invalid(&self) -> bool {
        self.physical_x == u16::MAX && self.physical_y == u16::MAX
    }

    /// Update the frame accessed timestamp.
    #[inline]
    pub fn touch(&mut self, frame: u32) {
        self.frame_accessed = frame;
    }

    /// Get the slot index for the given configuration.
    #[inline]
    pub fn slot_index(&self, config: &PhysicalAtlasConfig) -> Option<u32> {
        config.coords_to_slot(self.physical_x as u32, self.physical_y as u32)
    }

    /// Convert to texel coordinates for the given configuration.
    #[inline]
    pub fn to_texel(&self, config: &PhysicalAtlasConfig) -> (u32, u32) {
        config.slot_to_texel(self.physical_x as u32, self.physical_y as u32)
    }

    /// Convert to content texel coordinates (excluding border).
    #[inline]
    pub fn to_content_texel(&self, config: &PhysicalAtlasConfig) -> (u32, u32) {
        config.slot_to_content_texel(self.physical_x as u32, self.physical_y as u32)
    }

    /// Pack the slot as a u64 for compact storage.
    pub fn pack(&self) -> u64 {
        ((self.physical_x as u64) << 48)
            | ((self.physical_y as u64) << 32)
            | (self.frame_accessed as u64)
    }

    /// Unpack from u64 representation.
    pub fn unpack(packed: u64) -> Self {
        Self {
            physical_x: ((packed >> 48) & 0xFFFF) as u16,
            physical_y: ((packed >> 32) & 0xFFFF) as u16,
            frame_accessed: (packed & 0xFFFFFFFF) as u32,
        }
    }
}

impl Default for TileSlot {
    fn default() -> Self {
        Self::invalid()
    }
}

// ---------------------------------------------------------------------------
// Tile Coordinate (Virtual Tile ID)
// ---------------------------------------------------------------------------

/// Identifies a virtual tile by its coordinates and mip level.
///
/// Used as a key to map virtual tiles to physical atlas slots.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Pod, Zeroable)]
pub struct TileCoordinate {
    /// Virtual X coordinate in tile units.
    pub x: u16,
    /// Virtual Y coordinate in tile units.
    pub y: u16,
    /// Mip level.
    pub mip: u8,
    /// Padding for alignment.
    pub _padding: [u8; 3],
}

impl TileCoordinate {
    /// Create a new tile coordinate.
    pub fn new(x: u16, y: u16, mip: u8) -> Self {
        Self {
            x,
            y,
            mip,
            _padding: [0; 3],
        }
    }

    /// Create from u32 coordinates (clamped to u16 range).
    pub fn from_u32(x: u32, y: u32, mip: u8) -> Self {
        Self::new(x.min(u16::MAX as u32) as u16, y.min(u16::MAX as u32) as u16, mip)
    }

    /// Pack into a u64 for compact storage.
    pub fn pack(&self) -> u64 {
        ((self.x as u64) << 24) | ((self.y as u64) << 8) | (self.mip as u64)
    }

    /// Unpack from u64 representation.
    pub fn unpack(packed: u64) -> Self {
        Self {
            x: ((packed >> 24) & 0xFFFF) as u16,
            y: ((packed >> 8) & 0xFFFF) as u16,
            mip: (packed & 0xFF) as u8,
            _padding: [0; 3],
        }
    }
}

impl Default for TileCoordinate {
    fn default() -> Self {
        Self::new(0, 0, 0)
    }
}

// ---------------------------------------------------------------------------
// Allocation Result
// ---------------------------------------------------------------------------

/// Result of a tile allocation operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationResult {
    /// Tile was allocated from the free list.
    Allocated(TileSlot),
    /// Tile was allocated by evicting an existing tile.
    Evicted {
        /// The newly allocated slot.
        slot: TileSlot,
        /// The coordinate of the evicted tile.
        evicted_coord: TileCoordinate,
    },
    /// Tile is already resident at this slot.
    AlreadyResident(TileSlot),
    /// Allocation failed (should not happen with LRU eviction).
    Failed,
}

impl AllocationResult {
    /// Get the allocated slot if successful.
    pub fn slot(&self) -> Option<TileSlot> {
        match self {
            Self::Allocated(slot) => Some(*slot),
            Self::Evicted { slot, .. } => Some(*slot),
            Self::AlreadyResident(slot) => Some(*slot),
            Self::Failed => None,
        }
    }

    /// Check if allocation was successful.
    pub fn is_success(&self) -> bool {
        !matches!(self, Self::Failed)
    }

    /// Check if a tile was evicted.
    pub fn was_evicted(&self) -> bool {
        matches!(self, Self::Evicted { .. })
    }

    /// Get the evicted coordinate if any.
    pub fn evicted_coordinate(&self) -> Option<TileCoordinate> {
        match self {
            Self::Evicted { evicted_coord, .. } => Some(*evicted_coord),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Physical Atlas
// ---------------------------------------------------------------------------

/// The physical texture atlas for virtual texturing.
///
/// Manages tile slots, tracks usage via LRU, and handles allocation/eviction.
#[derive(Debug)]
pub struct PhysicalAtlas {
    /// Configuration for the atlas.
    pub config: PhysicalAtlasConfig,
    /// All tile slots in the atlas.
    slots: Vec<TileSlot>,
    /// Free list of available slot indices.
    free_list: Vec<u32>,
    /// LRU queue of slot indices (front = least recently used).
    lru_queue: VecDeque<u32>,
    /// Mapping from virtual coordinate to slot index.
    coord_to_slot: HashMap<TileCoordinate, u32>,
    /// Mapping from slot index to virtual coordinate.
    slot_to_coord: HashMap<u32, TileCoordinate>,
    /// Current frame number for LRU tracking.
    current_frame: u32,
}

impl PhysicalAtlas {
    /// Create a new physical atlas with the given configuration.
    pub fn new(config: PhysicalAtlasConfig) -> Self {
        let total_slots = config.total_slots();

        // Initialize all slots
        let mut slots = Vec::with_capacity(total_slots as usize);
        for i in 0..total_slots {
            let (x, y) = config.slot_to_coords(i);
            slots.push(TileSlot::new(x as u16, y as u16));
        }

        // Initialize free list with all slots
        let free_list: Vec<u32> = (0..total_slots).collect();

        Self {
            config,
            slots,
            free_list,
            lru_queue: VecDeque::new(),
            coord_to_slot: HashMap::new(),
            slot_to_coord: HashMap::new(),
            current_frame: 0,
        }
    }

    /// Create an atlas with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(PhysicalAtlasConfig::default())
    }

    /// Get the total capacity in tile slots.
    #[inline]
    pub fn capacity(&self) -> u32 {
        self.config.total_slots()
    }

    /// Get the number of tiles currently in use.
    #[inline]
    pub fn used(&self) -> u32 {
        self.coord_to_slot.len() as u32
    }

    /// Get the number of free slots.
    #[inline]
    pub fn free_count(&self) -> u32 {
        self.free_list.len() as u32
    }

    /// Check if the atlas is full (no free slots).
    #[inline]
    pub fn is_full(&self) -> bool {
        self.free_list.is_empty()
    }

    /// Check if a tile coordinate is currently resident.
    #[inline]
    pub fn is_resident(&self, coord: &TileCoordinate) -> bool {
        self.coord_to_slot.contains_key(coord)
    }

    /// Get the slot for a tile coordinate if resident.
    pub fn get_slot(&self, coord: &TileCoordinate) -> Option<&TileSlot> {
        self.coord_to_slot
            .get(coord)
            .and_then(|&idx| self.slots.get(idx as usize))
    }

    /// Get the tile coordinate for a slot index.
    pub fn get_coordinate(&self, slot_index: u32) -> Option<&TileCoordinate> {
        self.slot_to_coord.get(&slot_index)
    }

    /// Get the current frame number.
    #[inline]
    pub fn current_frame(&self) -> u32 {
        self.current_frame
    }

    /// Advance to the next frame.
    pub fn advance_frame(&mut self) {
        self.current_frame = self.current_frame.wrapping_add(1);
    }

    /// Touch a tile to update its LRU status.
    ///
    /// Call this when a tile is accessed to prevent eviction.
    pub fn touch(&mut self, coord: &TileCoordinate) {
        if let Some(&slot_index) = self.coord_to_slot.get(coord) {
            if let Some(slot) = self.slots.get_mut(slot_index as usize) {
                slot.touch(self.current_frame);
            }

            // Move to back of LRU queue (most recently used)
            self.lru_queue.retain(|&idx| idx != slot_index);
            self.lru_queue.push_back(slot_index);
        }
    }

    /// Allocate a slot for a tile coordinate.
    ///
    /// If the tile is already resident, returns the existing slot.
    /// If the atlas is full, evicts the least recently used tile.
    pub fn allocate_tile(&mut self, coord: TileCoordinate) -> AllocationResult {
        // Check if already resident
        if let Some(&slot_index) = self.coord_to_slot.get(&coord) {
            let slot = self.slots[slot_index as usize];
            self.touch(&coord);
            return AllocationResult::AlreadyResident(slot);
        }

        // Try to get a free slot
        if let Some(slot_index) = self.free_list.pop() {
            let slot = &mut self.slots[slot_index as usize];
            slot.touch(self.current_frame);

            self.coord_to_slot.insert(coord, slot_index);
            self.slot_to_coord.insert(slot_index, coord);
            self.lru_queue.push_back(slot_index);

            return AllocationResult::Allocated(*slot);
        }

        // Need to evict LRU tile
        if let Some(evicted_index) = self.lru_queue.pop_front() {
            if let Some(evicted_coord) = self.slot_to_coord.remove(&evicted_index) {
                self.coord_to_slot.remove(&evicted_coord);

                let slot = &mut self.slots[evicted_index as usize];
                slot.touch(self.current_frame);

                self.coord_to_slot.insert(coord, evicted_index);
                self.slot_to_coord.insert(evicted_index, coord);
                self.lru_queue.push_back(evicted_index);

                return AllocationResult::Evicted {
                    slot: *slot,
                    evicted_coord,
                };
            }
        }

        AllocationResult::Failed
    }

    /// Free a tile slot, returning it to the free list.
    pub fn free_tile(&mut self, coord: &TileCoordinate) -> bool {
        if let Some(slot_index) = self.coord_to_slot.remove(coord) {
            self.slot_to_coord.remove(&slot_index);
            self.lru_queue.retain(|&idx| idx != slot_index);
            self.free_list.push(slot_index);
            return true;
        }
        false
    }

    /// Free a tile by slot index.
    pub fn free_tile_by_index(&mut self, slot_index: u32) -> bool {
        if let Some(coord) = self.slot_to_coord.remove(&slot_index) {
            self.coord_to_slot.remove(&coord);
            self.lru_queue.retain(|&idx| idx != slot_index);
            self.free_list.push(slot_index);
            return true;
        }
        false
    }

    /// Evict the least recently used tile.
    ///
    /// Returns the evicted slot and its coordinate.
    pub fn evict_lru(&mut self) -> Option<(TileSlot, TileCoordinate)> {
        let evicted_index = self.lru_queue.pop_front()?;
        let evicted_coord = self.slot_to_coord.remove(&evicted_index)?;
        self.coord_to_slot.remove(&evicted_coord);

        let slot = self.slots[evicted_index as usize];
        self.free_list.push(evicted_index);

        Some((slot, evicted_coord))
    }

    /// Convert a tile slot to texel coordinates.
    #[inline]
    pub fn tile_to_texel(&self, slot: &TileSlot) -> (u32, u32) {
        slot.to_texel(&self.config)
    }

    /// Get the tile size including border.
    #[inline]
    pub fn tile_size_with_border(&self) -> u32 {
        self.config.tile_size_with_border()
    }

    /// Clear all tiles from the atlas.
    pub fn clear(&mut self) {
        self.coord_to_slot.clear();
        self.slot_to_coord.clear();
        self.lru_queue.clear();

        // Refill free list
        self.free_list.clear();
        for i in 0..self.config.total_slots() {
            self.free_list.push(i);
        }

        // Reset frame counter
        self.current_frame = 0;
    }

    /// Get all resident tile coordinates.
    pub fn resident_tiles(&self) -> impl Iterator<Item = &TileCoordinate> {
        self.coord_to_slot.keys()
    }

    /// Get the number of tiles in the LRU queue.
    #[inline]
    pub fn lru_queue_len(&self) -> usize {
        self.lru_queue.len()
    }

    /// Iterate over slots in LRU order (front = oldest).
    pub fn iter_lru(&self) -> impl Iterator<Item = (u32, &TileSlot, Option<&TileCoordinate>)> {
        self.lru_queue.iter().map(|&idx| {
            let slot = &self.slots[idx as usize];
            let coord = self.slot_to_coord.get(&idx);
            (idx, slot, coord)
        })
    }

    /// Get statistics about the atlas.
    pub fn stats(&self) -> PhysicalAtlasStats {
        PhysicalAtlasStats {
            capacity: self.capacity(),
            used: self.used(),
            free: self.free_count(),
            current_frame: self.current_frame,
            lru_queue_len: self.lru_queue.len(),
            memory_bytes: self.config.memory_size_bytes(),
        }
    }
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Statistics about the physical atlas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhysicalAtlasStats {
    /// Total capacity in tile slots.
    pub capacity: u32,
    /// Number of tiles currently in use.
    pub used: u32,
    /// Number of free slots.
    pub free: u32,
    /// Current frame number.
    pub current_frame: u32,
    /// Length of the LRU queue.
    pub lru_queue_len: usize,
    /// Total memory in bytes.
    pub memory_bytes: usize,
}

// ---------------------------------------------------------------------------
// Upload Request
// ---------------------------------------------------------------------------

/// A request to upload tile data to the atlas.
#[derive(Debug, Clone)]
pub struct TileUploadRequest {
    /// Target slot in the atlas.
    pub slot: TileSlot,
    /// Virtual tile coordinate.
    pub coord: TileCoordinate,
    /// Byte offset in staging buffer.
    pub staging_offset: usize,
    /// Size of data in bytes.
    pub data_size: usize,
}

impl TileUploadRequest {
    /// Create a new upload request.
    pub fn new(slot: TileSlot, coord: TileCoordinate, staging_offset: usize, data_size: usize) -> Self {
        Self {
            slot,
            coord,
            staging_offset,
            data_size,
        }
    }
}

// ---------------------------------------------------------------------------
// Staging Buffer
// ---------------------------------------------------------------------------

/// Manages staging buffer for tile uploads.
///
/// Handles batching multiple tile uploads into a single buffer transfer.
#[derive(Debug)]
pub struct TileStagingBuffer {
    /// Staging data buffer.
    data: Vec<u8>,
    /// Maximum capacity in bytes.
    capacity: usize,
    /// Current write offset.
    write_offset: usize,
    /// Pending upload requests.
    requests: Vec<TileUploadRequest>,
    /// Tile size with border in bytes.
    tile_bytes: usize,
}

impl TileStagingBuffer {
    /// Create a new staging buffer for the given configuration.
    pub fn new(config: &PhysicalAtlasConfig, max_tiles: usize) -> Self {
        let tile_bytes = config.tile_memory_size_bytes();
        let capacity = tile_bytes * max_tiles;

        Self {
            data: vec![0u8; capacity],
            capacity,
            write_offset: 0,
            requests: Vec::with_capacity(max_tiles),
            tile_bytes,
        }
    }

    /// Create a staging buffer with default settings.
    pub fn with_defaults() -> Self {
        Self::new(&PhysicalAtlasConfig::default(), MAX_UPLOADS_PER_FRAME)
    }

    /// Get the capacity in tiles.
    pub fn tile_capacity(&self) -> usize {
        if self.tile_bytes > 0 {
            self.capacity / self.tile_bytes
        } else {
            0
        }
    }

    /// Get the number of pending upload requests.
    pub fn pending_count(&self) -> usize {
        self.requests.len()
    }

    /// Check if the staging buffer can accept another tile.
    pub fn can_stage(&self) -> bool {
        self.write_offset + self.tile_bytes <= self.capacity
    }

    /// Stage a tile for upload.
    ///
    /// Returns the staging offset if successful, or None if buffer is full.
    pub fn stage_tile(&mut self, slot: TileSlot, coord: TileCoordinate, data: &[u8]) -> Option<usize> {
        if !self.can_stage() {
            return None;
        }

        if data.len() > self.tile_bytes {
            return None;
        }

        let offset = self.write_offset;

        // Copy data to staging buffer
        let dest_end = offset + data.len();
        self.data[offset..dest_end].copy_from_slice(data);

        // Pad remaining space with zeros
        if data.len() < self.tile_bytes {
            let pad_start = dest_end;
            let pad_end = offset + self.tile_bytes;
            self.data[pad_start..pad_end].fill(0);
        }

        self.write_offset += self.tile_bytes;

        let request = TileUploadRequest::new(slot, coord, offset, data.len());
        self.requests.push(request);

        Some(offset)
    }

    /// Get the staging buffer data.
    pub fn data(&self) -> &[u8] {
        &self.data[..self.write_offset]
    }

    /// Get the staging buffer data as mutable.
    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data[..self.write_offset]
    }

    /// Get pending upload requests.
    pub fn requests(&self) -> &[TileUploadRequest] {
        &self.requests
    }

    /// Clear the staging buffer for the next frame.
    pub fn clear(&mut self) {
        self.write_offset = 0;
        self.requests.clear();
    }

    /// Get the current write offset.
    pub fn write_offset(&self) -> usize {
        self.write_offset
    }

    /// Get bytes per tile.
    pub fn tile_bytes(&self) -> usize {
        self.tile_bytes
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // PhysicalAtlasConfig Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_default() {
        let config = PhysicalAtlasConfig::default();
        assert_eq!(config.atlas_size, 16384);
        assert_eq!(config.tile_size, 128);
        assert_eq!(config.border_size, 2);
        // 16384 / 132 = 124
        assert_eq!(config.tiles_per_axis, 124);
    }

    #[test]
    fn test_config_new() {
        let config = PhysicalAtlasConfig::new(8192, 64, 4);
        assert_eq!(config.atlas_size, 8192);
        assert_eq!(config.tile_size, 64);
        assert_eq!(config.border_size, 4);
        // 8192 / 72 = 113
        assert_eq!(config.tiles_per_axis, 113);
    }

    #[test]
    fn test_config_tile_size_with_border() {
        let config = PhysicalAtlasConfig::default();
        assert_eq!(config.tile_size_with_border(), 132); // 128 + 2*2
    }

    #[test]
    fn test_config_total_slots() {
        let config = PhysicalAtlasConfig::default();
        assert_eq!(config.total_slots(), 124 * 124);
    }

    #[test]
    fn test_config_total_slots_small() {
        let config = PhysicalAtlasConfig::new(256, 64, 2);
        // 256 / 68 = 3
        assert_eq!(config.tiles_per_axis, 3);
        assert_eq!(config.total_slots(), 9);
    }

    #[test]
    fn test_config_mip_levels() {
        let config = PhysicalAtlasConfig::default();
        // log2(16384/4) + 1 = log2(4096) + 1 = 12 + 1 = 13
        assert_eq!(config.mip_levels(), 13);
    }

    #[test]
    fn test_config_mip_levels_small() {
        let config = PhysicalAtlasConfig::new(256, 64, 2);
        // log2(256/4) + 1 = log2(64) + 1 = 6 + 1 = 7
        assert_eq!(config.mip_levels(), 7);
    }

    #[test]
    fn test_config_mip_levels_zero() {
        let config = PhysicalAtlasConfig::new(0, 64, 2);
        assert_eq!(config.mip_levels(), 0);
    }

    #[test]
    fn test_config_mip_levels_very_small() {
        let config = PhysicalAtlasConfig::new(2, 64, 2);
        assert_eq!(config.mip_levels(), 1);
    }

    #[test]
    fn test_config_memory_size_bytes() {
        let config = PhysicalAtlasConfig::new(256, 64, 2);
        // 256 * 256 * 4 = 262144
        assert_eq!(config.memory_size_bytes(), 262144);
    }

    #[test]
    fn test_config_tile_memory_size_bytes() {
        let config = PhysicalAtlasConfig::default();
        // (128 + 4)^2 * 4 = 132^2 * 4 = 69696
        assert_eq!(config.tile_memory_size_bytes(), 69696);
    }

    #[test]
    fn test_config_slot_to_coords() {
        let config = PhysicalAtlasConfig::new(256, 64, 2);
        // tiles_per_axis = 3
        assert_eq!(config.slot_to_coords(0), (0, 0));
        assert_eq!(config.slot_to_coords(1), (1, 0));
        assert_eq!(config.slot_to_coords(2), (2, 0));
        assert_eq!(config.slot_to_coords(3), (0, 1));
        assert_eq!(config.slot_to_coords(8), (2, 2));
    }

    #[test]
    fn test_config_coords_to_slot() {
        let config = PhysicalAtlasConfig::new(256, 64, 2);
        assert_eq!(config.coords_to_slot(0, 0), Some(0));
        assert_eq!(config.coords_to_slot(1, 0), Some(1));
        assert_eq!(config.coords_to_slot(0, 1), Some(3));
        assert_eq!(config.coords_to_slot(2, 2), Some(8));
        assert_eq!(config.coords_to_slot(3, 0), None); // Out of bounds
        assert_eq!(config.coords_to_slot(0, 3), None);
    }

    #[test]
    fn test_config_slot_to_texel() {
        let config = PhysicalAtlasConfig::new(256, 64, 2);
        // tile with border = 68
        assert_eq!(config.slot_to_texel(0, 0), (0, 0));
        assert_eq!(config.slot_to_texel(1, 0), (68, 0));
        assert_eq!(config.slot_to_texel(0, 1), (0, 68));
        assert_eq!(config.slot_to_texel(1, 1), (68, 68));
    }

    #[test]
    fn test_config_slot_to_content_texel() {
        let config = PhysicalAtlasConfig::new(256, 64, 2);
        // border = 2
        assert_eq!(config.slot_to_content_texel(0, 0), (2, 2));
        assert_eq!(config.slot_to_content_texel(1, 0), (70, 2));
        assert_eq!(config.slot_to_content_texel(0, 1), (2, 70));
    }

    #[test]
    fn test_config_is_valid_slot() {
        let config = PhysicalAtlasConfig::new(256, 64, 2);
        assert!(config.is_valid_slot(0));
        assert!(config.is_valid_slot(8));
        assert!(!config.is_valid_slot(9));
        assert!(!config.is_valid_slot(100));
    }

    #[test]
    fn test_config_pod_zeroable() {
        let config: PhysicalAtlasConfig = bytemuck::Zeroable::zeroed();
        assert_eq!(config.atlas_size, 0);
        assert_eq!(config.tile_size, 0);
        assert_eq!(config.border_size, 0);
        assert_eq!(config.tiles_per_axis, 0);
    }

    #[test]
    fn test_config_memory_layout() {
        assert_eq!(std::mem::size_of::<PhysicalAtlasConfig>(), 16);
        assert_eq!(std::mem::align_of::<PhysicalAtlasConfig>(), 4);
    }

    // -----------------------------------------------------------------------
    // TileSlot Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_tile_slot_new() {
        let slot = TileSlot::new(10, 20);
        assert_eq!(slot.physical_x, 10);
        assert_eq!(slot.physical_y, 20);
        assert_eq!(slot.frame_accessed, 0);
    }

    #[test]
    fn test_tile_slot_invalid() {
        let slot = TileSlot::invalid();
        assert!(slot.is_invalid());
        assert_eq!(slot.physical_x, u16::MAX);
        assert_eq!(slot.physical_y, u16::MAX);
    }

    #[test]
    fn test_tile_slot_default() {
        let slot = TileSlot::default();
        assert!(slot.is_invalid());
    }

    #[test]
    fn test_tile_slot_is_invalid() {
        let valid = TileSlot::new(0, 0);
        assert!(!valid.is_invalid());

        let invalid = TileSlot::invalid();
        assert!(invalid.is_invalid());

        // Partially invalid
        let partial = TileSlot::new(u16::MAX, 0);
        assert!(!partial.is_invalid());
    }

    #[test]
    fn test_tile_slot_touch() {
        let mut slot = TileSlot::new(0, 0);
        assert_eq!(slot.frame_accessed, 0);

        slot.touch(42);
        assert_eq!(slot.frame_accessed, 42);

        slot.touch(100);
        assert_eq!(slot.frame_accessed, 100);
    }

    #[test]
    fn test_tile_slot_slot_index() {
        let config = PhysicalAtlasConfig::new(256, 64, 2);
        let slot = TileSlot::new(1, 2);
        // tiles_per_axis = 3
        // index = 2 * 3 + 1 = 7
        assert_eq!(slot.slot_index(&config), Some(7));
    }

    #[test]
    fn test_tile_slot_to_texel() {
        let config = PhysicalAtlasConfig::new(256, 64, 2);
        let slot = TileSlot::new(1, 1);
        let (tx, ty) = slot.to_texel(&config);
        assert_eq!(tx, 68);
        assert_eq!(ty, 68);
    }

    #[test]
    fn test_tile_slot_to_content_texel() {
        let config = PhysicalAtlasConfig::new(256, 64, 2);
        let slot = TileSlot::new(1, 1);
        let (tx, ty) = slot.to_content_texel(&config);
        assert_eq!(tx, 70);
        assert_eq!(ty, 70);
    }

    #[test]
    fn test_tile_slot_pack_unpack() {
        let original = TileSlot::new(1234, 5678);
        let mut slot = original;
        slot.frame_accessed = 987654321;

        let packed = slot.pack();
        let unpacked = TileSlot::unpack(packed);

        assert_eq!(unpacked.physical_x, 1234);
        assert_eq!(unpacked.physical_y, 5678);
        assert_eq!(unpacked.frame_accessed, 987654321);
    }

    #[test]
    fn test_tile_slot_pod_zeroable() {
        let slot: TileSlot = bytemuck::Zeroable::zeroed();
        assert_eq!(slot.physical_x, 0);
        assert_eq!(slot.physical_y, 0);
        assert_eq!(slot.frame_accessed, 0);
    }

    #[test]
    fn test_tile_slot_memory_layout() {
        assert_eq!(std::mem::size_of::<TileSlot>(), 8);
        assert_eq!(std::mem::align_of::<TileSlot>(), 4);
    }

    // -----------------------------------------------------------------------
    // TileCoordinate Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_tile_coordinate_new() {
        let coord = TileCoordinate::new(10, 20, 3);
        assert_eq!(coord.x, 10);
        assert_eq!(coord.y, 20);
        assert_eq!(coord.mip, 3);
    }

    #[test]
    fn test_tile_coordinate_from_u32() {
        let coord = TileCoordinate::from_u32(100000, 200000, 5);
        assert_eq!(coord.x, u16::MAX);
        assert_eq!(coord.y, u16::MAX);
        assert_eq!(coord.mip, 5);

        let coord2 = TileCoordinate::from_u32(100, 200, 2);
        assert_eq!(coord2.x, 100);
        assert_eq!(coord2.y, 200);
    }

    #[test]
    fn test_tile_coordinate_default() {
        let coord = TileCoordinate::default();
        assert_eq!(coord.x, 0);
        assert_eq!(coord.y, 0);
        assert_eq!(coord.mip, 0);
    }

    #[test]
    fn test_tile_coordinate_pack_unpack() {
        let original = TileCoordinate::new(0x1234, 0x5678, 0xAB);
        let packed = original.pack();
        let unpacked = TileCoordinate::unpack(packed);

        assert_eq!(unpacked.x, 0x1234);
        assert_eq!(unpacked.y, 0x5678);
        assert_eq!(unpacked.mip, 0xAB);
    }

    #[test]
    fn test_tile_coordinate_hash_eq() {
        let coord1 = TileCoordinate::new(1, 2, 3);
        let coord2 = TileCoordinate::new(1, 2, 3);
        let coord3 = TileCoordinate::new(1, 2, 4);

        assert_eq!(coord1, coord2);
        assert_ne!(coord1, coord3);

        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(coord1);
        assert!(set.contains(&coord2));
        assert!(!set.contains(&coord3));
    }

    #[test]
    fn test_tile_coordinate_pod_zeroable() {
        let coord: TileCoordinate = bytemuck::Zeroable::zeroed();
        assert_eq!(coord.x, 0);
        assert_eq!(coord.y, 0);
        assert_eq!(coord.mip, 0);
    }

    #[test]
    fn test_tile_coordinate_memory_layout() {
        assert_eq!(std::mem::size_of::<TileCoordinate>(), 8);
        assert_eq!(std::mem::align_of::<TileCoordinate>(), 2);
    }

    // -----------------------------------------------------------------------
    // AllocationResult Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_allocation_result_allocated() {
        let slot = TileSlot::new(1, 2);
        let result = AllocationResult::Allocated(slot);

        assert!(result.is_success());
        assert!(!result.was_evicted());
        assert_eq!(result.slot(), Some(slot));
        assert_eq!(result.evicted_coordinate(), None);
    }

    #[test]
    fn test_allocation_result_evicted() {
        let slot = TileSlot::new(1, 2);
        let coord = TileCoordinate::new(5, 6, 1);
        let result = AllocationResult::Evicted {
            slot,
            evicted_coord: coord,
        };

        assert!(result.is_success());
        assert!(result.was_evicted());
        assert_eq!(result.slot(), Some(slot));
        assert_eq!(result.evicted_coordinate(), Some(coord));
    }

    #[test]
    fn test_allocation_result_already_resident() {
        let slot = TileSlot::new(3, 4);
        let result = AllocationResult::AlreadyResident(slot);

        assert!(result.is_success());
        assert!(!result.was_evicted());
        assert_eq!(result.slot(), Some(slot));
    }

    #[test]
    fn test_allocation_result_failed() {
        let result = AllocationResult::Failed;

        assert!(!result.is_success());
        assert!(!result.was_evicted());
        assert_eq!(result.slot(), None);
        assert_eq!(result.evicted_coordinate(), None);
    }

    // -----------------------------------------------------------------------
    // PhysicalAtlas Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_atlas_new() {
        let config = PhysicalAtlasConfig::new(256, 64, 2);
        let atlas = PhysicalAtlas::new(config);

        assert_eq!(atlas.capacity(), 9);
        assert_eq!(atlas.used(), 0);
        assert_eq!(atlas.free_count(), 9);
        assert!(!atlas.is_full());
    }

    #[test]
    fn test_atlas_with_defaults() {
        let atlas = PhysicalAtlas::with_defaults();
        assert_eq!(atlas.config.atlas_size, 16384);
        assert_eq!(atlas.capacity(), 124 * 124);
    }

    #[test]
    fn test_atlas_allocate_single() {
        let config = PhysicalAtlasConfig::new(256, 64, 2);
        let mut atlas = PhysicalAtlas::new(config);

        let coord = TileCoordinate::new(0, 0, 0);
        let result = atlas.allocate_tile(coord);

        assert!(matches!(result, AllocationResult::Allocated(_)));
        assert_eq!(atlas.used(), 1);
        assert_eq!(atlas.free_count(), 8);
        assert!(atlas.is_resident(&coord));
    }

    #[test]
    fn test_atlas_allocate_multiple() {
        let config = PhysicalAtlasConfig::new(256, 64, 2);
        let mut atlas = PhysicalAtlas::new(config);

        for i in 0..5 {
            let coord = TileCoordinate::new(i, 0, 0);
            let result = atlas.allocate_tile(coord);
            assert!(result.is_success());
        }

        assert_eq!(atlas.used(), 5);
        assert_eq!(atlas.free_count(), 4);
    }

    #[test]
    fn test_atlas_allocate_already_resident() {
        let config = PhysicalAtlasConfig::new(256, 64, 2);
        let mut atlas = PhysicalAtlas::new(config);

        let coord = TileCoordinate::new(0, 0, 0);
        atlas.allocate_tile(coord);

        let result = atlas.allocate_tile(coord);
        assert!(matches!(result, AllocationResult::AlreadyResident(_)));
        assert_eq!(atlas.used(), 1); // Should not increase
    }

    #[test]
    fn test_atlas_allocate_full_eviction() {
        let config = PhysicalAtlasConfig::new(256, 64, 2);
        let mut atlas = PhysicalAtlas::new(config);

        // Fill the atlas
        for i in 0..9 {
            let coord = TileCoordinate::new(i, 0, 0);
            atlas.allocate_tile(coord);
            atlas.advance_frame();
        }

        assert!(atlas.is_full());

        // Allocate one more, should evict
        let new_coord = TileCoordinate::new(100, 100, 0);
        let result = atlas.allocate_tile(new_coord);

        assert!(matches!(result, AllocationResult::Evicted { .. }));
        assert_eq!(atlas.used(), 9);

        // First tile should have been evicted
        let evicted_coord = result.evicted_coordinate().unwrap();
        assert_eq!(evicted_coord.x, 0);
        assert!(!atlas.is_resident(&TileCoordinate::new(0, 0, 0)));
        assert!(atlas.is_resident(&new_coord));
    }

    #[test]
    fn test_atlas_touch() {
        let config = PhysicalAtlasConfig::new(256, 64, 2);
        let mut atlas = PhysicalAtlas::new(config);

        let coord1 = TileCoordinate::new(0, 0, 0);
        let coord2 = TileCoordinate::new(1, 0, 0);

        atlas.allocate_tile(coord1);
        atlas.advance_frame();
        atlas.allocate_tile(coord2);
        atlas.advance_frame();

        // Touch coord1 to make it more recent
        atlas.touch(&coord1);

        // Fill remaining slots
        for i in 2..9 {
            let coord = TileCoordinate::new(i, 0, 0);
            atlas.allocate_tile(coord);
            atlas.advance_frame();
        }

        // Allocate one more - should evict coord2 (not coord1)
        let new_coord = TileCoordinate::new(100, 0, 0);
        let result = atlas.allocate_tile(new_coord);

        let evicted = result.evicted_coordinate().unwrap();
        assert_eq!(evicted.x, 1); // coord2 was evicted
        assert!(atlas.is_resident(&coord1)); // coord1 is still there
    }

    #[test]
    fn test_atlas_free_tile() {
        let config = PhysicalAtlasConfig::new(256, 64, 2);
        let mut atlas = PhysicalAtlas::new(config);

        let coord = TileCoordinate::new(0, 0, 0);
        atlas.allocate_tile(coord);

        assert!(atlas.is_resident(&coord));
        assert_eq!(atlas.used(), 1);

        let freed = atlas.free_tile(&coord);
        assert!(freed);
        assert!(!atlas.is_resident(&coord));
        assert_eq!(atlas.used(), 0);
        assert_eq!(atlas.free_count(), 9);
    }

    #[test]
    fn test_atlas_free_tile_not_resident() {
        let config = PhysicalAtlasConfig::new(256, 64, 2);
        let mut atlas = PhysicalAtlas::new(config);

        let coord = TileCoordinate::new(0, 0, 0);
        let freed = atlas.free_tile(&coord);
        assert!(!freed);
    }

    #[test]
    fn test_atlas_free_tile_by_index() {
        let config = PhysicalAtlasConfig::new(256, 64, 2);
        let mut atlas = PhysicalAtlas::new(config);

        let coord = TileCoordinate::new(0, 0, 0);
        let result = atlas.allocate_tile(coord);
        let slot = result.slot().unwrap();
        let idx = slot.slot_index(&config).unwrap();

        assert!(atlas.free_tile_by_index(idx));
        assert!(!atlas.is_resident(&coord));
    }

    #[test]
    fn test_atlas_evict_lru() {
        let config = PhysicalAtlasConfig::new(256, 64, 2);
        let mut atlas = PhysicalAtlas::new(config);

        let coord1 = TileCoordinate::new(0, 0, 0);
        let coord2 = TileCoordinate::new(1, 0, 0);

        atlas.allocate_tile(coord1);
        atlas.advance_frame();
        atlas.allocate_tile(coord2);

        let evicted = atlas.evict_lru();
        assert!(evicted.is_some());
        let (_, evicted_coord) = evicted.unwrap();
        assert_eq!(evicted_coord.x, 0);
        assert!(!atlas.is_resident(&coord1));
        assert!(atlas.is_resident(&coord2));
    }

    #[test]
    fn test_atlas_evict_lru_empty() {
        let config = PhysicalAtlasConfig::new(256, 64, 2);
        let mut atlas = PhysicalAtlas::new(config);

        let result = atlas.evict_lru();
        assert!(result.is_none());
    }

    #[test]
    fn test_atlas_tile_to_texel() {
        let config = PhysicalAtlasConfig::new(256, 64, 2);
        let atlas = PhysicalAtlas::new(config);

        let slot = TileSlot::new(1, 2);
        let (tx, ty) = atlas.tile_to_texel(&slot);
        assert_eq!(tx, 68);
        assert_eq!(ty, 136);
    }

    #[test]
    fn test_atlas_tile_size_with_border() {
        let config = PhysicalAtlasConfig::new(256, 64, 2);
        let atlas = PhysicalAtlas::new(config);
        assert_eq!(atlas.tile_size_with_border(), 68);
    }

    #[test]
    fn test_atlas_clear() {
        let config = PhysicalAtlasConfig::new(256, 64, 2);
        let mut atlas = PhysicalAtlas::new(config);

        for i in 0..5 {
            atlas.allocate_tile(TileCoordinate::new(i, 0, 0));
        }

        atlas.clear();

        assert_eq!(atlas.used(), 0);
        assert_eq!(atlas.free_count(), 9);
        assert_eq!(atlas.current_frame(), 0);
    }

    #[test]
    fn test_atlas_get_slot() {
        let config = PhysicalAtlasConfig::new(256, 64, 2);
        let mut atlas = PhysicalAtlas::new(config);

        let coord = TileCoordinate::new(0, 0, 0);
        atlas.allocate_tile(coord);

        let slot = atlas.get_slot(&coord);
        assert!(slot.is_some());

        let non_existent = TileCoordinate::new(100, 100, 0);
        assert!(atlas.get_slot(&non_existent).is_none());
    }

    #[test]
    fn test_atlas_get_coordinate() {
        let config = PhysicalAtlasConfig::new(256, 64, 2);
        let mut atlas = PhysicalAtlas::new(config);

        let coord = TileCoordinate::new(0, 0, 0);
        let result = atlas.allocate_tile(coord);
        let slot = result.slot().unwrap();
        let idx = slot.slot_index(&config).unwrap();

        let retrieved = atlas.get_coordinate(idx);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap(), &coord);
    }

    #[test]
    fn test_atlas_resident_tiles() {
        let config = PhysicalAtlasConfig::new(256, 64, 2);
        let mut atlas = PhysicalAtlas::new(config);

        let coords: Vec<_> = (0..3).map(|i| TileCoordinate::new(i, 0, 0)).collect();
        for &coord in &coords {
            atlas.allocate_tile(coord);
        }

        let resident: Vec<_> = atlas.resident_tiles().collect();
        assert_eq!(resident.len(), 3);
        for coord in &coords {
            assert!(resident.contains(&coord));
        }
    }

    #[test]
    fn test_atlas_lru_queue_len() {
        let config = PhysicalAtlasConfig::new(256, 64, 2);
        let mut atlas = PhysicalAtlas::new(config);

        assert_eq!(atlas.lru_queue_len(), 0);

        atlas.allocate_tile(TileCoordinate::new(0, 0, 0));
        assert_eq!(atlas.lru_queue_len(), 1);

        atlas.allocate_tile(TileCoordinate::new(1, 0, 0));
        assert_eq!(atlas.lru_queue_len(), 2);
    }

    #[test]
    fn test_atlas_iter_lru() {
        let config = PhysicalAtlasConfig::new(256, 64, 2);
        let mut atlas = PhysicalAtlas::new(config);

        atlas.allocate_tile(TileCoordinate::new(0, 0, 0));
        atlas.allocate_tile(TileCoordinate::new(1, 0, 0));

        let lru_items: Vec<_> = atlas.iter_lru().collect();
        assert_eq!(lru_items.len(), 2);

        // First item should be the oldest (coord 0,0,0)
        let (_, _, coord) = lru_items[0];
        assert_eq!(coord.unwrap().x, 0);
    }

    #[test]
    fn test_atlas_stats() {
        let config = PhysicalAtlasConfig::new(256, 64, 2);
        let mut atlas = PhysicalAtlas::new(config);

        atlas.allocate_tile(TileCoordinate::new(0, 0, 0));
        atlas.allocate_tile(TileCoordinate::new(1, 0, 0));
        atlas.advance_frame();

        let stats = atlas.stats();
        assert_eq!(stats.capacity, 9);
        assert_eq!(stats.used, 2);
        assert_eq!(stats.free, 7);
        assert_eq!(stats.current_frame, 1);
        assert_eq!(stats.lru_queue_len, 2);
    }

    #[test]
    fn test_atlas_advance_frame() {
        let config = PhysicalAtlasConfig::new(256, 64, 2);
        let mut atlas = PhysicalAtlas::new(config);

        assert_eq!(atlas.current_frame(), 0);
        atlas.advance_frame();
        assert_eq!(atlas.current_frame(), 1);
        atlas.advance_frame();
        assert_eq!(atlas.current_frame(), 2);
    }

    #[test]
    fn test_atlas_frame_overflow() {
        let config = PhysicalAtlasConfig::new(256, 64, 2);
        let mut atlas = PhysicalAtlas::new(config);

        atlas.current_frame = u32::MAX;
        atlas.advance_frame();
        assert_eq!(atlas.current_frame(), 0); // Wraps around
    }

    // -----------------------------------------------------------------------
    // Staging Buffer Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_staging_buffer_new() {
        let config = PhysicalAtlasConfig::new(256, 64, 2);
        let staging = TileStagingBuffer::new(&config, 4);

        assert_eq!(staging.tile_capacity(), 4);
        assert_eq!(staging.pending_count(), 0);
        assert!(staging.can_stage());
    }

    #[test]
    fn test_staging_buffer_with_defaults() {
        let staging = TileStagingBuffer::with_defaults();
        assert_eq!(staging.tile_capacity(), MAX_UPLOADS_PER_FRAME);
    }

    #[test]
    fn test_staging_buffer_stage_tile() {
        let config = PhysicalAtlasConfig::new(256, 64, 2);
        let mut staging = TileStagingBuffer::new(&config, 4);

        let slot = TileSlot::new(0, 0);
        let coord = TileCoordinate::new(0, 0, 0);
        let data = vec![0u8; 1000];

        let offset = staging.stage_tile(slot, coord, &data);
        assert!(offset.is_some());
        assert_eq!(offset.unwrap(), 0);
        assert_eq!(staging.pending_count(), 1);
    }

    #[test]
    fn test_staging_buffer_stage_multiple() {
        let config = PhysicalAtlasConfig::new(256, 64, 2);
        let mut staging = TileStagingBuffer::new(&config, 4);

        let data = vec![0u8; 1000];

        for i in 0..4 {
            let slot = TileSlot::new(i, 0);
            let coord = TileCoordinate::new(i as u16, 0, 0);
            let offset = staging.stage_tile(slot, coord, &data);
            assert!(offset.is_some());
        }

        assert_eq!(staging.pending_count(), 4);
        assert!(!staging.can_stage()); // Buffer is now full
    }

    #[test]
    fn test_staging_buffer_stage_full() {
        let config = PhysicalAtlasConfig::new(256, 64, 2);
        let mut staging = TileStagingBuffer::new(&config, 2);

        let data = vec![0u8; 1000];

        // Fill the buffer
        for i in 0..2 {
            staging.stage_tile(TileSlot::new(i, 0), TileCoordinate::new(i as u16, 0, 0), &data);
        }

        // Try to stage one more
        let result = staging.stage_tile(TileSlot::new(2, 0), TileCoordinate::new(2, 0, 0), &data);
        assert!(result.is_none());
    }

    #[test]
    fn test_staging_buffer_data() {
        let config = PhysicalAtlasConfig::new(256, 64, 2);
        let mut staging = TileStagingBuffer::new(&config, 4);

        let data = vec![42u8; 1000];
        staging.stage_tile(TileSlot::new(0, 0), TileCoordinate::new(0, 0, 0), &data);

        let staged_data = staging.data();
        assert!(!staged_data.is_empty());
        assert_eq!(staged_data[0], 42);
    }

    #[test]
    fn test_staging_buffer_requests() {
        let config = PhysicalAtlasConfig::new(256, 64, 2);
        let mut staging = TileStagingBuffer::new(&config, 4);

        let slot = TileSlot::new(1, 2);
        let coord = TileCoordinate::new(5, 6, 1);
        let data = vec![0u8; 100];

        staging.stage_tile(slot, coord, &data);

        let requests = staging.requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].slot.physical_x, 1);
        assert_eq!(requests[0].slot.physical_y, 2);
        assert_eq!(requests[0].coord.x, 5);
        assert_eq!(requests[0].coord.y, 6);
        assert_eq!(requests[0].data_size, 100);
    }

    #[test]
    fn test_staging_buffer_clear() {
        let config = PhysicalAtlasConfig::new(256, 64, 2);
        let mut staging = TileStagingBuffer::new(&config, 4);

        let data = vec![0u8; 1000];
        staging.stage_tile(TileSlot::new(0, 0), TileCoordinate::new(0, 0, 0), &data);

        staging.clear();

        assert_eq!(staging.pending_count(), 0);
        assert_eq!(staging.write_offset(), 0);
        assert!(staging.can_stage());
    }

    #[test]
    fn test_staging_buffer_write_offset() {
        let config = PhysicalAtlasConfig::new(256, 64, 2);
        let mut staging = TileStagingBuffer::new(&config, 4);

        assert_eq!(staging.write_offset(), 0);

        let data = vec![0u8; 1000];
        staging.stage_tile(TileSlot::new(0, 0), TileCoordinate::new(0, 0, 0), &data);

        assert_eq!(staging.write_offset(), staging.tile_bytes());
    }

    #[test]
    fn test_staging_buffer_tile_bytes() {
        let config = PhysicalAtlasConfig::new(256, 64, 2);
        let staging = TileStagingBuffer::new(&config, 4);

        // (64 + 4)^2 * 4 = 68^2 * 4 = 18496
        assert_eq!(staging.tile_bytes(), 18496);
    }

    #[test]
    fn test_staging_buffer_data_mut() {
        let config = PhysicalAtlasConfig::new(256, 64, 2);
        let mut staging = TileStagingBuffer::new(&config, 4);

        let data = vec![0u8; 1000];
        staging.stage_tile(TileSlot::new(0, 0), TileCoordinate::new(0, 0, 0), &data);

        let staged_mut = staging.data_mut();
        staged_mut[0] = 123;

        assert_eq!(staging.data()[0], 123);
    }

    // -----------------------------------------------------------------------
    // TileUploadRequest Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_upload_request_new() {
        let slot = TileSlot::new(1, 2);
        let coord = TileCoordinate::new(3, 4, 1);
        let request = TileUploadRequest::new(slot, coord, 1000, 500);

        assert_eq!(request.slot.physical_x, 1);
        assert_eq!(request.slot.physical_y, 2);
        assert_eq!(request.coord.x, 3);
        assert_eq!(request.coord.y, 4);
        assert_eq!(request.staging_offset, 1000);
        assert_eq!(request.data_size, 500);
    }

    // -----------------------------------------------------------------------
    // Integration Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_full_allocation_cycle() {
        let config = PhysicalAtlasConfig::new(256, 64, 2);
        let mut atlas = PhysicalAtlas::new(config);

        // Fill the atlas
        for i in 0..9 {
            let coord = TileCoordinate::new(i, 0, 0);
            let result = atlas.allocate_tile(coord);
            assert!(result.is_success());
            atlas.advance_frame();
        }

        assert!(atlas.is_full());
        assert_eq!(atlas.used(), 9);

        // Allocate 5 more tiles (should evict 5)
        for i in 0..5 {
            let coord = TileCoordinate::new(100 + i, 0, 0);
            let result = atlas.allocate_tile(coord);
            assert!(result.was_evicted());
            atlas.advance_frame();
        }

        // Original tiles 0-4 should be evicted
        for i in 0..5 {
            assert!(!atlas.is_resident(&TileCoordinate::new(i, 0, 0)));
        }

        // New tiles should be resident
        for i in 0..5 {
            assert!(atlas.is_resident(&TileCoordinate::new(100 + i, 0, 0)));
        }

        // Original tiles 5-8 should still be resident
        for i in 5..9 {
            assert!(atlas.is_resident(&TileCoordinate::new(i, 0, 0)));
        }
    }

    #[test]
    fn test_lru_order_maintained() {
        let config = PhysicalAtlasConfig::new(256, 64, 2);
        let mut atlas = PhysicalAtlas::new(config);

        // Allocate tiles in order
        for i in 0..9 {
            atlas.allocate_tile(TileCoordinate::new(i, 0, 0));
            atlas.advance_frame();
        }

        // Touch tiles in reverse order
        for i in (0..9).rev() {
            atlas.touch(&TileCoordinate::new(i, 0, 0));
            atlas.advance_frame();
        }

        // Now tile 8 should be LRU (was touched first)
        let new_coord = TileCoordinate::new(100, 0, 0);
        let result = atlas.allocate_tile(new_coord);
        let evicted = result.evicted_coordinate().unwrap();

        assert_eq!(evicted.x, 8);
    }

    #[test]
    fn test_staging_to_atlas_workflow() {
        let config = PhysicalAtlasConfig::new(256, 64, 2);
        let mut atlas = PhysicalAtlas::new(config);
        let mut staging = TileStagingBuffer::new(&config, 4);

        // Allocate tiles and prepare uploads
        for i in 0..3 {
            let coord = TileCoordinate::new(i, 0, 0);
            let result = atlas.allocate_tile(coord);
            let slot = result.slot().unwrap();

            let data = vec![i as u8; 1000];
            staging.stage_tile(slot, coord, &data);
        }

        assert_eq!(atlas.used(), 3);
        assert_eq!(staging.pending_count(), 3);

        // Verify uploads would target correct positions
        for (i, request) in staging.requests().iter().enumerate() {
            assert_eq!(request.coord.x, i as u16);
        }

        // Clear staging for next frame
        staging.clear();
        assert_eq!(staging.pending_count(), 0);
    }

    #[test]
    fn test_border_calculation_accuracy() {
        let config = PhysicalAtlasConfig::new(1024, 128, 2);

        // Slot 0,0 content should start at (2,2)
        let (cx, cy) = config.slot_to_content_texel(0, 0);
        assert_eq!(cx, 2);
        assert_eq!(cy, 2);

        // Slot 1,1 content should start at (132+2, 132+2) = (134, 134)
        let (cx, cy) = config.slot_to_content_texel(1, 1);
        assert_eq!(cx, 134);
        assert_eq!(cy, 134);
    }

    #[test]
    fn test_memory_calculation() {
        let config = PhysicalAtlasConfig::default();

        // 16384^2 * 4 = 1GB
        assert_eq!(config.memory_size_bytes(), 16384 * 16384 * 4);

        // 132^2 * 4 = 69696 bytes per tile
        assert_eq!(config.tile_memory_size_bytes(), 132 * 132 * 4);
    }
}
