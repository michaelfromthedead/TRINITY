//! Virtual Texturing Page Table (T-ENV-2.10)
//!
//! Implements a hierarchical page table for virtual texturing. The page table
//! maps virtual texture coordinates to physical tile locations in a texture atlas.
//!
//! # Architecture
//!
//! - **VirtualTextureConfig**: Configuration for virtual texture dimensions
//! - **PageTableEntry**: Single entry mapping virtual to physical tile location
//! - **PageTableFlags**: Bit flags for entry status (resident, streaming, etc.)
//! - **VirtualTexturePageTable**: The main page table structure with mip chain
//!
//! # Memory Layout
//!
//! The page table is organized as a flat buffer with mip levels stored contiguously.
//! Each mip level has (page_table_size >> mip)^2 entries. For a 1024x1024 base:
//! - Mip 0: 1024x1024 = 1,048,576 entries
//! - Mip 1: 512x512 = 262,144 entries
//! - Mip 2: 256x256 = 65,536 entries
//! - ...and so on
//!
//! Total entries for all mip levels: ~1.33M entries (8 bytes each = ~10.7MB)
//!
//! # GPU Format
//!
//! The page table is uploaded as an RGBA16Uint texture where:
//! - R: physical_x (u16)
//! - G: physical_y (u16)
//! - B: mip_level (u8 in low byte), flags (u8 in high byte)
//! - A: padding/reserved

use bytemuck::{Pod, Zeroable};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default virtual texture size (128K x 128K pixels).
pub const DEFAULT_VIRTUAL_SIZE: u32 = 131072;

/// Default tile size in pixels.
pub const DEFAULT_TILE_SIZE: u32 = 128;

/// Default page table size (tiles per dimension at mip 0).
pub const DEFAULT_PAGE_TABLE_SIZE: u32 = 1024;

/// Default border size for filtering.
pub const DEFAULT_BORDER_SIZE: u32 = 2;

/// Maximum number of mip levels supported.
pub const MAX_MIP_LEVELS: u32 = 11; // log2(1024) + 1

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the virtual texture system.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
pub struct VirtualTextureConfig {
    /// Total virtual texture size in pixels (e.g., 131072 for 128K).
    pub virtual_size: u32,
    /// Size of each tile in pixels (e.g., 128).
    pub tile_size: u32,
    /// Page table size in tiles at mip 0 (e.g., 1024).
    pub page_table_size: u32,
    /// Border size for filtering (e.g., 2).
    pub border_size: u32,
}

impl Default for VirtualTextureConfig {
    fn default() -> Self {
        Self {
            virtual_size: DEFAULT_VIRTUAL_SIZE,
            tile_size: DEFAULT_TILE_SIZE,
            page_table_size: DEFAULT_PAGE_TABLE_SIZE,
            border_size: DEFAULT_BORDER_SIZE,
        }
    }
}

impl VirtualTextureConfig {
    /// Create a new configuration.
    pub fn new(virtual_size: u32, tile_size: u32, page_table_size: u32, border_size: u32) -> Self {
        Self {
            virtual_size,
            tile_size,
            page_table_size,
            border_size,
        }
    }

    /// Calculate the number of mip levels.
    pub fn mip_levels(&self) -> u32 {
        if self.page_table_size == 0 {
            return 0;
        }
        (self.page_table_size as f32).log2().floor() as u32 + 1
    }

    /// Calculate the page table size at a given mip level.
    pub fn mip_size(&self, mip: u32) -> u32 {
        if mip >= self.mip_levels() {
            return 0;
        }
        self.page_table_size >> mip
    }

    /// Calculate the total number of entries across all mip levels.
    pub fn total_entries(&self) -> usize {
        let mut total = 0usize;
        for mip in 0..self.mip_levels() {
            let size = self.mip_size(mip) as usize;
            total += size * size;
        }
        total
    }

    /// Calculate the tile size including borders.
    pub fn tile_size_with_border(&self) -> u32 {
        self.tile_size + self.border_size * 2
    }
}

// ---------------------------------------------------------------------------
// Page Table Entry
// ---------------------------------------------------------------------------

/// Flags for page table entries.
pub struct PageTableFlags;

impl PageTableFlags {
    /// Page is resident in the physical atlas.
    pub const RESIDENT: u8 = 1 << 0;
    /// Page has been requested for loading.
    pub const REQUESTED: u8 = 1 << 1;
    /// Page is currently streaming from disk.
    pub const STREAMING: u8 = 1 << 2;
    /// Page data is invalid/corrupted.
    pub const INVALID: u8 = 1 << 3;
    /// Page is locked and cannot be evicted.
    pub const LOCKED: u8 = 1 << 4;
    /// Page was accessed this frame.
    pub const ACCESSED: u8 = 1 << 5;
    /// Reserved for future use.
    pub const RESERVED_6: u8 = 1 << 6;
    /// Reserved for future use.
    pub const RESERVED_7: u8 = 1 << 7;
}

/// A single entry in the page table.
///
/// Maps a virtual tile to a physical location in the texture atlas.
/// Packed as 8 bytes for efficient GPU upload as RGBA16Uint.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
pub struct PageTableEntry {
    /// Physical X coordinate in the atlas (in tiles).
    pub physical_x: u16,
    /// Physical Y coordinate in the atlas (in tiles).
    pub physical_y: u16,
    /// Mip level this entry maps to.
    pub mip_level: u8,
    /// Status flags (see `PageTableFlags`).
    pub flags: u8,
    /// Padding for alignment.
    pub _padding: u16,
}

impl PageTableEntry {
    /// Create a new page table entry.
    pub fn new(physical_x: u16, physical_y: u16, mip_level: u8, flags: u8) -> Self {
        Self {
            physical_x,
            physical_y,
            mip_level,
            flags,
            _padding: 0,
        }
    }

    /// Create an invalid/empty entry.
    pub fn invalid() -> Self {
        Self {
            physical_x: 0,
            physical_y: 0,
            mip_level: 0,
            flags: PageTableFlags::INVALID,
            _padding: 0,
        }
    }

    /// Check if the page is resident in the physical atlas.
    #[inline]
    pub fn is_resident(&self) -> bool {
        (self.flags & PageTableFlags::RESIDENT) != 0
    }

    /// Check if the page has been requested.
    #[inline]
    pub fn is_requested(&self) -> bool {
        (self.flags & PageTableFlags::REQUESTED) != 0
    }

    /// Check if the page is currently streaming.
    #[inline]
    pub fn is_streaming(&self) -> bool {
        (self.flags & PageTableFlags::STREAMING) != 0
    }

    /// Check if the page data is invalid.
    #[inline]
    pub fn is_invalid(&self) -> bool {
        (self.flags & PageTableFlags::INVALID) != 0
    }

    /// Check if the page is locked.
    #[inline]
    pub fn is_locked(&self) -> bool {
        (self.flags & PageTableFlags::LOCKED) != 0
    }

    /// Check if the page was accessed this frame.
    #[inline]
    pub fn is_accessed(&self) -> bool {
        (self.flags & PageTableFlags::ACCESSED) != 0
    }

    /// Check if a specific flag is set.
    #[inline]
    pub fn has_flag(&self, flag: u8) -> bool {
        (self.flags & flag) != 0
    }

    /// Set a flag to a specific value.
    #[inline]
    pub fn set_flag(&mut self, flag: u8, value: bool) {
        if value {
            self.flags |= flag;
        } else {
            self.flags &= !flag;
        }
    }

    /// Set a flag.
    #[inline]
    pub fn add_flag(&mut self, flag: u8) {
        self.flags |= flag;
    }

    /// Clear a flag.
    #[inline]
    pub fn clear_flag(&mut self, flag: u8) {
        self.flags &= !flag;
    }

    /// Clear all flags.
    #[inline]
    pub fn clear_all_flags(&mut self) {
        self.flags = 0;
    }

    /// Set physical coordinates.
    #[inline]
    pub fn set_physical(&mut self, x: u16, y: u16) {
        self.physical_x = x;
        self.physical_y = y;
    }

    /// Pack the entry as RGBA16 for GPU upload.
    /// Returns [R, G, B, A] where each is u16.
    pub fn as_rgba16(&self) -> [u16; 4] {
        [
            self.physical_x,
            self.physical_y,
            (self.mip_level as u16) | ((self.flags as u16) << 8),
            self._padding,
        ]
    }

    /// Unpack from RGBA16 format.
    pub fn from_rgba16(rgba: [u16; 4]) -> Self {
        Self {
            physical_x: rgba[0],
            physical_y: rgba[1],
            mip_level: (rgba[2] & 0xFF) as u8,
            flags: ((rgba[2] >> 8) & 0xFF) as u8,
            _padding: rgba[3],
        }
    }
}

impl Default for PageTableEntry {
    fn default() -> Self {
        Self::invalid()
    }
}

// ---------------------------------------------------------------------------
// Virtual Texture Page Table
// ---------------------------------------------------------------------------

/// The virtual texture page table.
///
/// Maintains a hierarchical mapping from virtual tile coordinates to physical
/// atlas locations. Supports multiple mip levels with each level being half
/// the size of the previous.
#[derive(Debug, Clone)]
pub struct VirtualTexturePageTable {
    /// Configuration for the virtual texture.
    pub config: VirtualTextureConfig,
    /// Flat buffer of all page table entries across all mip levels.
    pub entries: Vec<PageTableEntry>,
    /// Byte offsets into `entries` for each mip level.
    pub mip_offsets: Vec<u32>,
}

impl VirtualTexturePageTable {
    /// Create a new page table with the given configuration.
    pub fn new(config: VirtualTextureConfig) -> Self {
        let mip_levels = config.mip_levels();
        let total_entries = config.total_entries();

        // Calculate mip offsets
        let mut mip_offsets = Vec::with_capacity(mip_levels as usize);
        let mut offset = 0u32;
        for mip in 0..mip_levels {
            mip_offsets.push(offset);
            let size = config.mip_size(mip);
            offset += size * size;
        }

        // Initialize all entries as invalid
        let entries = vec![PageTableEntry::invalid(); total_entries];

        Self {
            config,
            entries,
            mip_offsets,
        }
    }

    /// Create a page table with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(VirtualTextureConfig::default())
    }

    /// Get the number of mip levels.
    #[inline]
    pub fn mip_levels(&self) -> u32 {
        self.mip_offsets.len() as u32
    }

    /// Get the page table size at a given mip level.
    #[inline]
    pub fn mip_size(&self, mip: u32) -> u32 {
        self.config.mip_size(mip)
    }

    /// Convert virtual UV coordinates to tile coordinates.
    ///
    /// UV coordinates are in range [0, 1]. Returns (tile_x, tile_y) for the
    /// given mip level.
    pub fn virtual_to_tile(&self, vt_uv: [f32; 2], mip: u32) -> (u32, u32) {
        let mip_size = self.mip_size(mip);
        if mip_size == 0 {
            return (0, 0);
        }

        // Clamp UV to [0, 1)
        let u = vt_uv[0].clamp(0.0, 0.999999);
        let v = vt_uv[1].clamp(0.0, 0.999999);

        // Convert to tile coordinates
        let tile_x = (u * mip_size as f32) as u32;
        let tile_y = (v * mip_size as f32) as u32;

        (tile_x.min(mip_size - 1), tile_y.min(mip_size - 1))
    }

    /// Convert tile coordinates back to UV center.
    pub fn tile_to_virtual_uv(&self, tile_x: u32, tile_y: u32, mip: u32) -> [f32; 2] {
        let mip_size = self.mip_size(mip);
        if mip_size == 0 {
            return [0.0, 0.0];
        }

        let u = (tile_x as f32 + 0.5) / mip_size as f32;
        let v = (tile_y as f32 + 0.5) / mip_size as f32;

        [u, v]
    }

    /// Calculate the linear index for a tile at given coordinates and mip level.
    fn tile_index(&self, tile_x: u32, tile_y: u32, mip: u32) -> Option<usize> {
        if mip >= self.mip_levels() {
            return None;
        }

        let mip_size = self.mip_size(mip);
        if tile_x >= mip_size || tile_y >= mip_size {
            return None;
        }

        let offset = self.mip_offsets[mip as usize] as usize;
        Some(offset + (tile_y as usize * mip_size as usize) + tile_x as usize)
    }

    /// Get a reference to a page table entry.
    pub fn get_entry(&self, tile_x: u32, tile_y: u32, mip: u32) -> Option<&PageTableEntry> {
        self.tile_index(tile_x, tile_y, mip)
            .map(|idx| &self.entries[idx])
    }

    /// Get a mutable reference to a page table entry.
    pub fn get_entry_mut(
        &mut self,
        tile_x: u32,
        tile_y: u32,
        mip: u32,
    ) -> Option<&mut PageTableEntry> {
        self.tile_index(tile_x, tile_y, mip)
            .map(|idx| &mut self.entries[idx])
    }

    /// Set a page table entry.
    pub fn set_entry(&mut self, tile_x: u32, tile_y: u32, mip: u32, entry: PageTableEntry) -> bool {
        if let Some(idx) = self.tile_index(tile_x, tile_y, mip) {
            self.entries[idx] = entry;
            true
        } else {
            false
        }
    }

    /// Compute the mip level from screen-space derivatives.
    ///
    /// Uses the standard formula: mip = log2(max(|ddx|, |ddy|) * virtual_size / tile_size)
    ///
    /// # Arguments
    /// * `ddx` - Screen-space derivative in X direction (du/dx, dv/dx)
    /// * `ddy` - Screen-space derivative in Y direction (du/dy, dv/dy)
    pub fn compute_mip_level(&self, ddx: [f32; 2], ddy: [f32; 2]) -> u32 {
        // Calculate the length of the derivatives
        let ddx_len = (ddx[0] * ddx[0] + ddx[1] * ddx[1]).sqrt();
        let ddy_len = (ddy[0] * ddy[0] + ddy[1] * ddy[1]).sqrt();

        // Use the maximum derivative
        let max_derivative = ddx_len.max(ddy_len);

        if max_derivative <= 0.0 {
            return 0;
        }

        // Scale by virtual texture size to get pixel-space derivative
        let pixel_derivative = max_derivative * self.config.virtual_size as f32;

        // Compute mip level: log2(pixel_derivative / tile_size)
        let mip = (pixel_derivative / self.config.tile_size as f32).log2();

        // Clamp to valid range
        let max_mip = self.mip_levels().saturating_sub(1);
        (mip.max(0.0) as u32).min(max_mip)
    }

    /// Compute mip level from a single LOD value (like textureLod).
    pub fn compute_mip_from_lod(&self, lod: f32) -> u32 {
        let max_mip = self.mip_levels().saturating_sub(1);
        (lod.max(0.0) as u32).min(max_mip)
    }

    /// Mark a tile as resident with the given physical coordinates.
    pub fn mark_resident(&mut self, tile_x: u32, tile_y: u32, mip: u32, phys_x: u16, phys_y: u16) {
        if let Some(entry) = self.get_entry_mut(tile_x, tile_y, mip) {
            entry.physical_x = phys_x;
            entry.physical_y = phys_y;
            entry.mip_level = mip as u8;
            entry.flags = PageTableFlags::RESIDENT;
        }
    }

    /// Mark a tile as requested.
    pub fn mark_requested(&mut self, tile_x: u32, tile_y: u32, mip: u32) {
        if let Some(entry) = self.get_entry_mut(tile_x, tile_y, mip) {
            entry.add_flag(PageTableFlags::REQUESTED);
        }
    }

    /// Mark a tile as streaming.
    pub fn mark_streaming(&mut self, tile_x: u32, tile_y: u32, mip: u32) {
        if let Some(entry) = self.get_entry_mut(tile_x, tile_y, mip) {
            entry.clear_flag(PageTableFlags::REQUESTED);
            entry.add_flag(PageTableFlags::STREAMING);
        }
    }

    /// Invalidate a tile (mark as non-resident).
    pub fn invalidate(&mut self, tile_x: u32, tile_y: u32, mip: u32) {
        if let Some(entry) = self.get_entry_mut(tile_x, tile_y, mip) {
            *entry = PageTableEntry::invalid();
        }
    }

    /// Clear all entries (mark all as invalid).
    pub fn clear(&mut self) {
        for entry in &mut self.entries {
            *entry = PageTableEntry::invalid();
        }
    }

    /// Clear the accessed flag on all entries.
    pub fn clear_accessed_flags(&mut self) {
        for entry in &mut self.entries {
            entry.clear_flag(PageTableFlags::ACCESSED);
        }
    }

    /// Get all entries for a specific mip level as a slice.
    pub fn mip_entries(&self, mip: u32) -> Option<&[PageTableEntry]> {
        if mip >= self.mip_levels() {
            return None;
        }

        let offset = self.mip_offsets[mip as usize] as usize;
        let size = self.mip_size(mip) as usize;
        let count = size * size;

        Some(&self.entries[offset..offset + count])
    }

    /// Get all entries for a specific mip level as a mutable slice.
    pub fn mip_entries_mut(&mut self, mip: u32) -> Option<&mut [PageTableEntry]> {
        if mip >= self.mip_levels() {
            return None;
        }

        let offset = self.mip_offsets[mip as usize] as usize;
        let size = self.mip_size(mip) as usize;
        let count = size * size;

        Some(&mut self.entries[offset..offset + count])
    }

    /// Count resident pages at a given mip level.
    pub fn count_resident(&self, mip: u32) -> usize {
        self.mip_entries(mip)
            .map(|entries| entries.iter().filter(|e| e.is_resident()).count())
            .unwrap_or(0)
    }

    /// Count total resident pages across all mip levels.
    pub fn total_resident(&self) -> usize {
        self.entries.iter().filter(|e| e.is_resident()).count()
    }

    /// Get the raw entry data as bytes for GPU upload.
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.entries)
    }

    /// Get the memory size in bytes.
    pub fn memory_size(&self) -> usize {
        self.entries.len() * std::mem::size_of::<PageTableEntry>()
    }

    /// Find the fallback mip level for a non-resident tile.
    ///
    /// Searches higher (coarser) mip levels for a resident tile that covers
    /// the same virtual location.
    pub fn find_fallback_mip(
        &self,
        tile_x: u32,
        tile_y: u32,
        requested_mip: u32,
    ) -> Option<(u32, u32, u32)> {
        for mip in (requested_mip + 1)..self.mip_levels() {
            // Calculate the corresponding tile at this mip level
            let scale = 1u32 << (mip - requested_mip);
            let parent_x = tile_x / scale;
            let parent_y = tile_y / scale;

            if let Some(entry) = self.get_entry(parent_x, parent_y, mip) {
                if entry.is_resident() {
                    return Some((parent_x, parent_y, mip));
                }
            }
        }
        None
    }

    /// Iterate over all non-resident tiles at a given mip level.
    pub fn iter_non_resident(&self, mip: u32) -> impl Iterator<Item = (u32, u32)> + '_ {
        let mip_size = self.mip_size(mip);
        let offset = self.mip_offsets.get(mip as usize).copied().unwrap_or(0) as usize;

        (0..mip_size).flat_map(move |y| {
            (0..mip_size).filter_map(move |x| {
                let idx = offset + (y as usize * mip_size as usize) + x as usize;
                if !self.entries.get(idx).map(|e| e.is_resident()).unwrap_or(true) {
                    Some((x, y))
                } else {
                    None
                }
            })
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // VirtualTextureConfig Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_default() {
        let config = VirtualTextureConfig::default();
        assert_eq!(config.virtual_size, 131072);
        assert_eq!(config.tile_size, 128);
        assert_eq!(config.page_table_size, 1024);
        assert_eq!(config.border_size, 2);
    }

    #[test]
    fn test_config_new() {
        let config = VirtualTextureConfig::new(65536, 64, 512, 4);
        assert_eq!(config.virtual_size, 65536);
        assert_eq!(config.tile_size, 64);
        assert_eq!(config.page_table_size, 512);
        assert_eq!(config.border_size, 4);
    }

    #[test]
    fn test_config_mip_levels() {
        let config = VirtualTextureConfig::default();
        assert_eq!(config.mip_levels(), 11); // log2(1024) + 1

        let config = VirtualTextureConfig::new(8192, 128, 64, 2);
        assert_eq!(config.mip_levels(), 7); // log2(64) + 1
    }

    #[test]
    fn test_config_mip_levels_power_of_two() {
        let config = VirtualTextureConfig::new(16384, 128, 128, 2);
        assert_eq!(config.mip_levels(), 8); // log2(128) + 1
    }

    #[test]
    fn test_config_mip_levels_zero() {
        let config = VirtualTextureConfig::new(0, 128, 0, 2);
        assert_eq!(config.mip_levels(), 0);
    }

    #[test]
    fn test_config_mip_size() {
        let config = VirtualTextureConfig::default();
        assert_eq!(config.mip_size(0), 1024);
        assert_eq!(config.mip_size(1), 512);
        assert_eq!(config.mip_size(2), 256);
        assert_eq!(config.mip_size(10), 1);
        assert_eq!(config.mip_size(11), 0); // Out of bounds
    }

    #[test]
    fn test_config_total_entries() {
        let config = VirtualTextureConfig::new(8192, 128, 64, 2);
        // 64^2 + 32^2 + 16^2 + 8^2 + 4^2 + 2^2 + 1^2
        // = 4096 + 1024 + 256 + 64 + 16 + 4 + 1 = 5461
        assert_eq!(config.total_entries(), 5461);
    }

    #[test]
    fn test_config_total_entries_small() {
        let config = VirtualTextureConfig::new(256, 128, 2, 2);
        // 2^2 + 1^2 = 4 + 1 = 5
        assert_eq!(config.total_entries(), 5);
    }

    #[test]
    fn test_config_tile_size_with_border() {
        let config = VirtualTextureConfig::default();
        assert_eq!(config.tile_size_with_border(), 132); // 128 + 2*2
    }

    #[test]
    fn test_config_pod_zeroable() {
        let config: VirtualTextureConfig = bytemuck::Zeroable::zeroed();
        assert_eq!(config.virtual_size, 0);
        assert_eq!(config.tile_size, 0);
        assert_eq!(config.page_table_size, 0);
        assert_eq!(config.border_size, 0);
    }

    #[test]
    fn test_config_memory_layout() {
        assert_eq!(std::mem::size_of::<VirtualTextureConfig>(), 16);
        assert_eq!(std::mem::align_of::<VirtualTextureConfig>(), 4);
    }

    // -----------------------------------------------------------------------
    // PageTableFlags Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_flags_values() {
        assert_eq!(PageTableFlags::RESIDENT, 0b0000_0001);
        assert_eq!(PageTableFlags::REQUESTED, 0b0000_0010);
        assert_eq!(PageTableFlags::STREAMING, 0b0000_0100);
        assert_eq!(PageTableFlags::INVALID, 0b0000_1000);
        assert_eq!(PageTableFlags::LOCKED, 0b0001_0000);
        assert_eq!(PageTableFlags::ACCESSED, 0b0010_0000);
    }

    #[test]
    fn test_flags_no_overlap() {
        let all_flags = [
            PageTableFlags::RESIDENT,
            PageTableFlags::REQUESTED,
            PageTableFlags::STREAMING,
            PageTableFlags::INVALID,
            PageTableFlags::LOCKED,
            PageTableFlags::ACCESSED,
            PageTableFlags::RESERVED_6,
            PageTableFlags::RESERVED_7,
        ];

        for (i, &flag_a) in all_flags.iter().enumerate() {
            for (j, &flag_b) in all_flags.iter().enumerate() {
                if i != j {
                    assert_eq!(flag_a & flag_b, 0, "Flags {} and {} overlap", i, j);
                }
            }
        }
    }

    #[test]
    fn test_flags_cover_all_bits() {
        let all_flags = PageTableFlags::RESIDENT
            | PageTableFlags::REQUESTED
            | PageTableFlags::STREAMING
            | PageTableFlags::INVALID
            | PageTableFlags::LOCKED
            | PageTableFlags::ACCESSED
            | PageTableFlags::RESERVED_6
            | PageTableFlags::RESERVED_7;
        assert_eq!(all_flags, 0xFF);
    }

    // -----------------------------------------------------------------------
    // PageTableEntry Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_entry_new() {
        let entry = PageTableEntry::new(100, 200, 3, PageTableFlags::RESIDENT);
        assert_eq!(entry.physical_x, 100);
        assert_eq!(entry.physical_y, 200);
        assert_eq!(entry.mip_level, 3);
        assert_eq!(entry.flags, PageTableFlags::RESIDENT);
        assert_eq!(entry._padding, 0);
    }

    #[test]
    fn test_entry_invalid() {
        let entry = PageTableEntry::invalid();
        assert_eq!(entry.physical_x, 0);
        assert_eq!(entry.physical_y, 0);
        assert_eq!(entry.mip_level, 0);
        assert!(entry.is_invalid());
    }

    #[test]
    fn test_entry_default() {
        let entry = PageTableEntry::default();
        assert!(entry.is_invalid());
    }

    #[test]
    fn test_entry_is_resident() {
        let mut entry = PageTableEntry::new(0, 0, 0, PageTableFlags::RESIDENT);
        assert!(entry.is_resident());

        entry.flags = 0;
        assert!(!entry.is_resident());
    }

    #[test]
    fn test_entry_is_requested() {
        let entry = PageTableEntry::new(0, 0, 0, PageTableFlags::REQUESTED);
        assert!(entry.is_requested());
        assert!(!entry.is_resident());
    }

    #[test]
    fn test_entry_is_streaming() {
        let entry = PageTableEntry::new(0, 0, 0, PageTableFlags::STREAMING);
        assert!(entry.is_streaming());
    }

    #[test]
    fn test_entry_is_invalid() {
        let entry = PageTableEntry::new(0, 0, 0, PageTableFlags::INVALID);
        assert!(entry.is_invalid());
    }

    #[test]
    fn test_entry_is_locked() {
        let entry = PageTableEntry::new(0, 0, 0, PageTableFlags::LOCKED);
        assert!(entry.is_locked());
    }

    #[test]
    fn test_entry_is_accessed() {
        let entry = PageTableEntry::new(0, 0, 0, PageTableFlags::ACCESSED);
        assert!(entry.is_accessed());
    }

    #[test]
    fn test_entry_has_flag() {
        let entry = PageTableEntry::new(0, 0, 0, PageTableFlags::RESIDENT | PageTableFlags::LOCKED);
        assert!(entry.has_flag(PageTableFlags::RESIDENT));
        assert!(entry.has_flag(PageTableFlags::LOCKED));
        assert!(!entry.has_flag(PageTableFlags::STREAMING));
    }

    #[test]
    fn test_entry_set_flag_true() {
        let mut entry = PageTableEntry::new(0, 0, 0, 0);
        entry.set_flag(PageTableFlags::RESIDENT, true);
        assert!(entry.is_resident());
    }

    #[test]
    fn test_entry_set_flag_false() {
        let mut entry = PageTableEntry::new(0, 0, 0, PageTableFlags::RESIDENT);
        entry.set_flag(PageTableFlags::RESIDENT, false);
        assert!(!entry.is_resident());
    }

    #[test]
    fn test_entry_add_flag() {
        let mut entry = PageTableEntry::new(0, 0, 0, PageTableFlags::RESIDENT);
        entry.add_flag(PageTableFlags::LOCKED);
        assert!(entry.is_resident());
        assert!(entry.is_locked());
    }

    #[test]
    fn test_entry_clear_flag() {
        let mut entry =
            PageTableEntry::new(0, 0, 0, PageTableFlags::RESIDENT | PageTableFlags::LOCKED);
        entry.clear_flag(PageTableFlags::RESIDENT);
        assert!(!entry.is_resident());
        assert!(entry.is_locked());
    }

    #[test]
    fn test_entry_clear_all_flags() {
        let mut entry = PageTableEntry::new(
            0,
            0,
            0,
            PageTableFlags::RESIDENT | PageTableFlags::LOCKED | PageTableFlags::ACCESSED,
        );
        entry.clear_all_flags();
        assert_eq!(entry.flags, 0);
    }

    #[test]
    fn test_entry_set_physical() {
        let mut entry = PageTableEntry::new(0, 0, 0, 0);
        entry.set_physical(42, 99);
        assert_eq!(entry.physical_x, 42);
        assert_eq!(entry.physical_y, 99);
    }

    #[test]
    fn test_entry_as_rgba16() {
        let entry = PageTableEntry::new(100, 200, 5, PageTableFlags::RESIDENT);
        let rgba = entry.as_rgba16();
        assert_eq!(rgba[0], 100); // R = physical_x
        assert_eq!(rgba[1], 200); // G = physical_y
        assert_eq!(rgba[2], 5 | (1 << 8)); // B = mip_level | (flags << 8)
        assert_eq!(rgba[3], 0); // A = padding
    }

    #[test]
    fn test_entry_from_rgba16() {
        let rgba = [100u16, 200, 5 | (1 << 8), 0];
        let entry = PageTableEntry::from_rgba16(rgba);
        assert_eq!(entry.physical_x, 100);
        assert_eq!(entry.physical_y, 200);
        assert_eq!(entry.mip_level, 5);
        assert_eq!(entry.flags, PageTableFlags::RESIDENT);
    }

    #[test]
    fn test_entry_rgba16_roundtrip() {
        let original = PageTableEntry::new(
            1234,
            5678,
            7,
            PageTableFlags::RESIDENT | PageTableFlags::STREAMING,
        );
        let rgba = original.as_rgba16();
        let decoded = PageTableEntry::from_rgba16(rgba);
        assert_eq!(original.physical_x, decoded.physical_x);
        assert_eq!(original.physical_y, decoded.physical_y);
        assert_eq!(original.mip_level, decoded.mip_level);
        assert_eq!(original.flags, decoded.flags);
    }

    #[test]
    fn test_entry_pod_zeroable() {
        let entry: PageTableEntry = bytemuck::Zeroable::zeroed();
        assert_eq!(entry.physical_x, 0);
        assert_eq!(entry.physical_y, 0);
        assert_eq!(entry.mip_level, 0);
        assert_eq!(entry.flags, 0);
    }

    #[test]
    fn test_entry_memory_layout() {
        assert_eq!(std::mem::size_of::<PageTableEntry>(), 8);
        assert_eq!(std::mem::align_of::<PageTableEntry>(), 2);
    }

    #[test]
    fn test_entry_bytes_cast() {
        let entry = PageTableEntry::new(0x0102, 0x0304, 5, 6);
        let bytes: &[u8] = bytemuck::bytes_of(&entry);
        assert_eq!(bytes.len(), 8);
        // Little-endian: physical_x = 0x0102 -> [0x02, 0x01]
        assert_eq!(bytes[0], 0x02);
        assert_eq!(bytes[1], 0x01);
        assert_eq!(bytes[2], 0x04);
        assert_eq!(bytes[3], 0x03);
        assert_eq!(bytes[4], 5);
        assert_eq!(bytes[5], 6);
    }

    // -----------------------------------------------------------------------
    // VirtualTexturePageTable Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_page_table_new() {
        let config = VirtualTextureConfig::new(8192, 128, 64, 2);
        let table = VirtualTexturePageTable::new(config);
        assert_eq!(table.mip_levels(), 7);
        assert_eq!(table.entries.len(), 5461);
    }

    #[test]
    fn test_page_table_with_defaults() {
        let table = VirtualTexturePageTable::with_defaults();
        assert_eq!(table.config.virtual_size, 131072);
        assert_eq!(table.mip_levels(), 11);
    }

    #[test]
    fn test_page_table_mip_offsets() {
        let config = VirtualTextureConfig::new(8192, 128, 64, 2);
        let table = VirtualTexturePageTable::new(config);

        assert_eq!(table.mip_offsets[0], 0);
        assert_eq!(table.mip_offsets[1], 64 * 64); // 4096
        assert_eq!(table.mip_offsets[2], 64 * 64 + 32 * 32); // 5120
    }

    #[test]
    fn test_page_table_virtual_to_tile_center() {
        let config = VirtualTextureConfig::new(8192, 128, 64, 2);
        let table = VirtualTexturePageTable::new(config);

        // Center of UV space
        let (x, y) = table.virtual_to_tile([0.5, 0.5], 0);
        assert_eq!(x, 32);
        assert_eq!(y, 32);
    }

    #[test]
    fn test_page_table_virtual_to_tile_origin() {
        let config = VirtualTextureConfig::new(8192, 128, 64, 2);
        let table = VirtualTexturePageTable::new(config);

        let (x, y) = table.virtual_to_tile([0.0, 0.0], 0);
        assert_eq!(x, 0);
        assert_eq!(y, 0);
    }

    #[test]
    fn test_page_table_virtual_to_tile_max() {
        let config = VirtualTextureConfig::new(8192, 128, 64, 2);
        let table = VirtualTexturePageTable::new(config);

        let (x, y) = table.virtual_to_tile([1.0, 1.0], 0);
        assert_eq!(x, 63);
        assert_eq!(y, 63);
    }

    #[test]
    fn test_page_table_virtual_to_tile_clamp_negative() {
        let config = VirtualTextureConfig::new(8192, 128, 64, 2);
        let table = VirtualTexturePageTable::new(config);

        let (x, y) = table.virtual_to_tile([-0.5, -0.5], 0);
        assert_eq!(x, 0);
        assert_eq!(y, 0);
    }

    #[test]
    fn test_page_table_virtual_to_tile_clamp_over() {
        let config = VirtualTextureConfig::new(8192, 128, 64, 2);
        let table = VirtualTexturePageTable::new(config);

        let (x, y) = table.virtual_to_tile([1.5, 2.0], 0);
        assert_eq!(x, 63);
        assert_eq!(y, 63);
    }

    #[test]
    fn test_page_table_virtual_to_tile_mip1() {
        let config = VirtualTextureConfig::new(8192, 128, 64, 2);
        let table = VirtualTexturePageTable::new(config);

        // At mip 1, size is 32x32
        let (x, y) = table.virtual_to_tile([0.5, 0.5], 1);
        assert_eq!(x, 16);
        assert_eq!(y, 16);
    }

    #[test]
    fn test_page_table_tile_to_virtual_uv() {
        let config = VirtualTextureConfig::new(8192, 128, 64, 2);
        let table = VirtualTexturePageTable::new(config);

        let uv = table.tile_to_virtual_uv(32, 32, 0);
        // Center of tile 32 in 64x64 grid = (32.5/64, 32.5/64)
        assert!((uv[0] - 0.5078125).abs() < 0.001);
        assert!((uv[1] - 0.5078125).abs() < 0.001);
    }

    #[test]
    fn test_page_table_get_entry() {
        let config = VirtualTextureConfig::new(1024, 128, 8, 2);
        let table = VirtualTexturePageTable::new(config);

        let entry = table.get_entry(0, 0, 0);
        assert!(entry.is_some());
        assert!(entry.unwrap().is_invalid());
    }

    #[test]
    fn test_page_table_get_entry_out_of_bounds_x() {
        let config = VirtualTextureConfig::new(1024, 128, 8, 2);
        let table = VirtualTexturePageTable::new(config);

        assert!(table.get_entry(8, 0, 0).is_none());
    }

    #[test]
    fn test_page_table_get_entry_out_of_bounds_y() {
        let config = VirtualTextureConfig::new(1024, 128, 8, 2);
        let table = VirtualTexturePageTable::new(config);

        assert!(table.get_entry(0, 8, 0).is_none());
    }

    #[test]
    fn test_page_table_get_entry_out_of_bounds_mip() {
        let config = VirtualTextureConfig::new(1024, 128, 8, 2);
        let table = VirtualTexturePageTable::new(config);

        assert!(table.get_entry(0, 0, 100).is_none());
    }

    #[test]
    fn test_page_table_set_entry() {
        let config = VirtualTextureConfig::new(1024, 128, 8, 2);
        let mut table = VirtualTexturePageTable::new(config);

        let entry = PageTableEntry::new(10, 20, 0, PageTableFlags::RESIDENT);
        assert!(table.set_entry(3, 4, 0, entry));

        let retrieved = table.get_entry(3, 4, 0).unwrap();
        assert_eq!(retrieved.physical_x, 10);
        assert_eq!(retrieved.physical_y, 20);
        assert!(retrieved.is_resident());
    }

    #[test]
    fn test_page_table_set_entry_out_of_bounds() {
        let config = VirtualTextureConfig::new(1024, 128, 8, 2);
        let mut table = VirtualTexturePageTable::new(config);

        let entry = PageTableEntry::new(10, 20, 0, PageTableFlags::RESIDENT);
        assert!(!table.set_entry(100, 0, 0, entry));
    }

    #[test]
    fn test_page_table_compute_mip_level_zero() {
        let config = VirtualTextureConfig::default();
        let table = VirtualTexturePageTable::new(config);

        // Very small derivative -> mip 0
        let mip = table.compute_mip_level([0.000001, 0.0], [0.0, 0.000001]);
        assert_eq!(mip, 0);
    }

    #[test]
    fn test_page_table_compute_mip_level_high() {
        let config = VirtualTextureConfig::default();
        let table = VirtualTexturePageTable::new(config);

        // Large derivative -> high mip
        let mip = table.compute_mip_level([0.1, 0.0], [0.0, 0.1]);
        assert!(mip > 5);
    }

    #[test]
    fn test_page_table_compute_mip_level_clamped() {
        let config = VirtualTextureConfig::default();
        let table = VirtualTexturePageTable::new(config);

        // Huge derivative -> clamped to max mip
        let mip = table.compute_mip_level([100.0, 0.0], [0.0, 100.0]);
        assert_eq!(mip, table.mip_levels() - 1);
    }

    #[test]
    fn test_page_table_compute_mip_level_zero_derivative() {
        let config = VirtualTextureConfig::default();
        let table = VirtualTexturePageTable::new(config);

        let mip = table.compute_mip_level([0.0, 0.0], [0.0, 0.0]);
        assert_eq!(mip, 0);
    }

    #[test]
    fn test_page_table_compute_mip_from_lod() {
        let config = VirtualTextureConfig::default();
        let table = VirtualTexturePageTable::new(config);

        assert_eq!(table.compute_mip_from_lod(0.0), 0);
        assert_eq!(table.compute_mip_from_lod(5.5), 5);
        assert_eq!(table.compute_mip_from_lod(-1.0), 0);
        assert_eq!(table.compute_mip_from_lod(100.0), table.mip_levels() - 1);
    }

    #[test]
    fn test_page_table_mark_resident() {
        let config = VirtualTextureConfig::new(1024, 128, 8, 2);
        let mut table = VirtualTexturePageTable::new(config);

        table.mark_resident(2, 3, 0, 100, 200);

        let entry = table.get_entry(2, 3, 0).unwrap();
        assert!(entry.is_resident());
        assert_eq!(entry.physical_x, 100);
        assert_eq!(entry.physical_y, 200);
        assert_eq!(entry.mip_level, 0);
    }

    #[test]
    fn test_page_table_mark_requested() {
        let config = VirtualTextureConfig::new(1024, 128, 8, 2);
        let mut table = VirtualTexturePageTable::new(config);

        table.mark_requested(2, 3, 0);

        let entry = table.get_entry(2, 3, 0).unwrap();
        assert!(entry.is_requested());
    }

    #[test]
    fn test_page_table_mark_streaming() {
        let config = VirtualTextureConfig::new(1024, 128, 8, 2);
        let mut table = VirtualTexturePageTable::new(config);

        table.mark_requested(2, 3, 0);
        table.mark_streaming(2, 3, 0);

        let entry = table.get_entry(2, 3, 0).unwrap();
        assert!(entry.is_streaming());
        assert!(!entry.is_requested()); // Cleared when streaming starts
    }

    #[test]
    fn test_page_table_invalidate() {
        let config = VirtualTextureConfig::new(1024, 128, 8, 2);
        let mut table = VirtualTexturePageTable::new(config);

        table.mark_resident(2, 3, 0, 100, 200);
        table.invalidate(2, 3, 0);

        let entry = table.get_entry(2, 3, 0).unwrap();
        assert!(entry.is_invalid());
        assert!(!entry.is_resident());
    }

    #[test]
    fn test_page_table_clear() {
        let config = VirtualTextureConfig::new(1024, 128, 8, 2);
        let mut table = VirtualTexturePageTable::new(config);

        table.mark_resident(0, 0, 0, 1, 1);
        table.mark_resident(1, 1, 0, 2, 2);
        table.mark_resident(2, 2, 1, 3, 3);

        table.clear();

        assert!(table.get_entry(0, 0, 0).unwrap().is_invalid());
        assert!(table.get_entry(1, 1, 0).unwrap().is_invalid());
        assert!(table.get_entry(2, 2, 1).unwrap().is_invalid());
    }

    #[test]
    fn test_page_table_clear_accessed_flags() {
        let config = VirtualTextureConfig::new(1024, 128, 8, 2);
        let mut table = VirtualTexturePageTable::new(config);

        if let Some(entry) = table.get_entry_mut(0, 0, 0) {
            entry.add_flag(PageTableFlags::ACCESSED | PageTableFlags::RESIDENT);
        }
        if let Some(entry) = table.get_entry_mut(1, 1, 0) {
            entry.add_flag(PageTableFlags::ACCESSED);
        }

        table.clear_accessed_flags();

        assert!(!table.get_entry(0, 0, 0).unwrap().is_accessed());
        assert!(table.get_entry(0, 0, 0).unwrap().is_resident()); // Other flags preserved
        assert!(!table.get_entry(1, 1, 0).unwrap().is_accessed());
    }

    #[test]
    fn test_page_table_mip_entries() {
        let config = VirtualTextureConfig::new(1024, 128, 8, 2);
        let table = VirtualTexturePageTable::new(config);

        let mip0_entries = table.mip_entries(0).unwrap();
        assert_eq!(mip0_entries.len(), 64); // 8x8

        let mip1_entries = table.mip_entries(1).unwrap();
        assert_eq!(mip1_entries.len(), 16); // 4x4

        let mip2_entries = table.mip_entries(2).unwrap();
        assert_eq!(mip2_entries.len(), 4); // 2x2

        assert!(table.mip_entries(100).is_none());
    }

    #[test]
    fn test_page_table_count_resident() {
        let config = VirtualTextureConfig::new(1024, 128, 8, 2);
        let mut table = VirtualTexturePageTable::new(config);

        assert_eq!(table.count_resident(0), 0);

        table.mark_resident(0, 0, 0, 1, 1);
        table.mark_resident(1, 0, 0, 2, 1);
        table.mark_resident(0, 1, 0, 1, 2);

        assert_eq!(table.count_resident(0), 3);
        assert_eq!(table.count_resident(1), 0);
    }

    #[test]
    fn test_page_table_total_resident() {
        let config = VirtualTextureConfig::new(1024, 128, 8, 2);
        let mut table = VirtualTexturePageTable::new(config);

        table.mark_resident(0, 0, 0, 1, 1);
        table.mark_resident(1, 0, 0, 2, 1);
        table.mark_resident(0, 0, 1, 1, 1);

        assert_eq!(table.total_resident(), 3);
    }

    #[test]
    fn test_page_table_as_bytes() {
        let config = VirtualTextureConfig::new(256, 128, 2, 2);
        let table = VirtualTexturePageTable::new(config);

        let bytes = table.as_bytes();
        assert_eq!(bytes.len(), table.entries.len() * 8);
    }

    #[test]
    fn test_page_table_memory_size() {
        let config = VirtualTextureConfig::new(256, 128, 2, 2);
        let table = VirtualTexturePageTable::new(config);

        // 2^2 + 1^2 = 5 entries, 8 bytes each
        assert_eq!(table.memory_size(), 40);
    }

    #[test]
    fn test_page_table_find_fallback_mip() {
        let config = VirtualTextureConfig::new(1024, 128, 8, 2);
        let mut table = VirtualTexturePageTable::new(config);

        // Mark a tile resident at mip 2
        table.mark_resident(0, 0, 2, 10, 10);

        // Look for fallback from mip 0, tile (0,0)
        // At mip 0, tile (0,0) corresponds to tile (0,0) at mip 2 (scale by 4)
        let fallback = table.find_fallback_mip(0, 0, 0);
        assert!(fallback.is_some());
        let (x, y, mip) = fallback.unwrap();
        assert_eq!(mip, 2);
        assert_eq!(x, 0);
        assert_eq!(y, 0);
    }

    #[test]
    fn test_page_table_find_fallback_mip_none() {
        let config = VirtualTextureConfig::new(1024, 128, 8, 2);
        let table = VirtualTexturePageTable::new(config);

        // No fallback available
        let fallback = table.find_fallback_mip(0, 0, 0);
        assert!(fallback.is_none());
    }

    #[test]
    fn test_page_table_find_fallback_mip_scaled() {
        let config = VirtualTextureConfig::new(1024, 128, 8, 2);
        let mut table = VirtualTexturePageTable::new(config);

        // Mark tile (1, 1) at mip 1 (covers tiles 2-3, 2-3 at mip 0)
        table.mark_resident(1, 1, 1, 5, 5);

        let fallback = table.find_fallback_mip(2, 2, 0);
        assert!(fallback.is_some());
        let (x, y, mip) = fallback.unwrap();
        assert_eq!(mip, 1);
        assert_eq!(x, 1);
        assert_eq!(y, 1);
    }

    #[test]
    fn test_page_table_iter_non_resident() {
        let config = VirtualTextureConfig::new(256, 128, 2, 2);
        let mut table = VirtualTexturePageTable::new(config);

        // 2x2 at mip 0
        table.mark_resident(0, 0, 0, 1, 1);
        // Now 3 tiles are non-resident at mip 0

        let non_resident: Vec<_> = table.iter_non_resident(0).collect();
        assert_eq!(non_resident.len(), 3);
        assert!(non_resident.contains(&(1, 0)));
        assert!(non_resident.contains(&(0, 1)));
        assert!(non_resident.contains(&(1, 1)));
    }

    #[test]
    fn test_page_table_multiple_mip_operations() {
        let config = VirtualTextureConfig::new(1024, 128, 4, 2);
        let mut table = VirtualTexturePageTable::new(config);

        // Set entries at different mip levels
        table.mark_resident(0, 0, 0, 0, 0);
        table.mark_resident(3, 3, 0, 1, 1);
        table.mark_resident(1, 1, 1, 2, 2);
        table.mark_resident(0, 0, 2, 3, 3);

        assert_eq!(table.count_resident(0), 2);
        assert_eq!(table.count_resident(1), 1);
        assert_eq!(table.count_resident(2), 1);
        assert_eq!(table.total_resident(), 4);
    }

    #[test]
    fn test_page_table_entry_independence() {
        let config = VirtualTextureConfig::new(1024, 128, 4, 2);
        let mut table = VirtualTexturePageTable::new(config);

        // Set one entry
        table.mark_resident(1, 2, 0, 100, 200);

        // Verify adjacent entries are unaffected
        assert!(table.get_entry(0, 2, 0).unwrap().is_invalid());
        assert!(table.get_entry(2, 2, 0).unwrap().is_invalid());
        assert!(table.get_entry(1, 1, 0).unwrap().is_invalid());
        assert!(table.get_entry(1, 3, 0).unwrap().is_invalid());
    }

    #[test]
    fn test_page_table_get_entry_mut() {
        let config = VirtualTextureConfig::new(1024, 128, 4, 2);
        let mut table = VirtualTexturePageTable::new(config);

        if let Some(entry) = table.get_entry_mut(1, 1, 0) {
            entry.physical_x = 42;
            entry.physical_y = 99;
            entry.flags = PageTableFlags::RESIDENT;
        }

        let entry = table.get_entry(1, 1, 0).unwrap();
        assert_eq!(entry.physical_x, 42);
        assert_eq!(entry.physical_y, 99);
        assert!(entry.is_resident());
    }

    #[test]
    fn test_page_table_mip_entries_mut() {
        let config = VirtualTextureConfig::new(256, 128, 2, 2);
        let mut table = VirtualTexturePageTable::new(config);

        if let Some(entries) = table.mip_entries_mut(0) {
            for entry in entries.iter_mut() {
                entry.flags = PageTableFlags::RESIDENT;
            }
        }

        assert_eq!(table.count_resident(0), 4);
    }

    // -----------------------------------------------------------------------
    // Edge Cases and Stress Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_page_table_single_tile() {
        let config = VirtualTextureConfig::new(128, 128, 1, 2);
        let mut table = VirtualTexturePageTable::new(config);

        assert_eq!(table.mip_levels(), 1);
        assert_eq!(table.entries.len(), 1);

        table.mark_resident(0, 0, 0, 0, 0);
        assert_eq!(table.total_resident(), 1);
    }

    #[test]
    fn test_page_table_max_coordinates() {
        let config = VirtualTextureConfig::new(8192, 128, 64, 2);
        let mut table = VirtualTexturePageTable::new(config);

        // Set entry at max coordinates
        table.mark_resident(63, 63, 0, u16::MAX, u16::MAX);

        let entry = table.get_entry(63, 63, 0).unwrap();
        assert_eq!(entry.physical_x, u16::MAX);
        assert_eq!(entry.physical_y, u16::MAX);
    }

    #[test]
    fn test_config_equality() {
        let config1 = VirtualTextureConfig::default();
        let config2 = VirtualTextureConfig::default();
        assert_eq!(config1, config2);

        let config3 = VirtualTextureConfig::new(1024, 64, 16, 1);
        assert_ne!(config1, config3);
    }

    #[test]
    fn test_entry_equality() {
        let entry1 = PageTableEntry::new(10, 20, 3, PageTableFlags::RESIDENT);
        let entry2 = PageTableEntry::new(10, 20, 3, PageTableFlags::RESIDENT);
        assert_eq!(entry1, entry2);

        let entry3 = PageTableEntry::new(10, 20, 3, PageTableFlags::STREAMING);
        assert_ne!(entry1, entry3);
    }
}
