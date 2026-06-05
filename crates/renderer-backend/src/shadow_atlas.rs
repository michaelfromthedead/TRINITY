//! Shadow Atlas 2D Bin-Packing Allocator.
//!
//! Provides dynamic allocation of shadow map tiles within a single large atlas
//! texture. Supports multiple tile size tiers (256, 512, 1024, 2048) with
//! automatic fallback to smaller tiers when larger ones are exhausted.
//!
//! # Atlas Layout
//!
//! The atlas is divided into a grid based on tile size tiers:
//!
//! ```text
//! For a 4096x4096 atlas:
//! - 2048 tier: 2x2 = 4 tiles
//! - 1024 tier: 4x4 = 16 tiles
//! - 512 tier: 8x8 = 64 tiles
//! - 256 tier: 16x16 = 256 tiles
//! ```
//!
//! Each tier maintains a free-list for O(1) allocation/deallocation.
//!
//! # Usage
//!
//! ```ignore
//! let mut atlas = ShadowAtlas::new(4096);
//!
//! // Allocate a large tile for an important light
//! if let Some(tile) = atlas.allocate(TileSizeTier::Large1024, light_id, 100) {
//!     // Use tile.uv_offset and tile.uv_scale for shadow sampling
//! }
//!
//! // Deallocate when light is removed
//! atlas.deallocate(tile);
//! ```

use bytemuck::{Pod, Zeroable};
use std::collections::{BTreeMap, HashMap, VecDeque};

// ---------------------------------------------------------------------------
// Tile Size Tiers
// ---------------------------------------------------------------------------

/// Available shadow tile size tiers.
///
/// Larger tiers provide higher shadow quality but consume more atlas space.
/// The allocator will fall back to smaller tiers when requested tier is full.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TileSizeTier {
    /// 256x256 pixel tiles - lowest quality, highest capacity
    Small256 = 0,
    /// 512x512 pixel tiles - medium quality
    Medium512 = 1,
    /// 1024x1024 pixel tiles - high quality
    Large1024 = 2,
    /// 2048x2048 pixel tiles - highest quality, lowest capacity
    XLarge2048 = 3,
}

impl TileSizeTier {
    /// Get the pixel size for this tier.
    #[inline]
    pub const fn size_pixels(self) -> u32 {
        match self {
            TileSizeTier::Small256 => 256,
            TileSizeTier::Medium512 => 512,
            TileSizeTier::Large1024 => 1024,
            TileSizeTier::XLarge2048 => 2048,
        }
    }

    /// All tiers in ascending size order.
    pub const ALL: [TileSizeTier; 4] = [
        TileSizeTier::Small256,
        TileSizeTier::Medium512,
        TileSizeTier::Large1024,
        TileSizeTier::XLarge2048,
    ];

    /// Get the next smaller tier, if any.
    #[inline]
    pub const fn smaller(self) -> Option<TileSizeTier> {
        match self {
            TileSizeTier::Small256 => None,
            TileSizeTier::Medium512 => Some(TileSizeTier::Small256),
            TileSizeTier::Large1024 => Some(TileSizeTier::Medium512),
            TileSizeTier::XLarge2048 => Some(TileSizeTier::Large1024),
        }
    }

    /// Create from u8 value.
    pub const fn from_u8(value: u8) -> Option<TileSizeTier> {
        match value {
            0 => Some(TileSizeTier::Small256),
            1 => Some(TileSizeTier::Medium512),
            2 => Some(TileSizeTier::Large1024),
            3 => Some(TileSizeTier::XLarge2048),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Shadow Tile
// ---------------------------------------------------------------------------

/// A single allocated shadow tile within the atlas.
///
/// Contains both pixel coordinates (for render target setup) and normalized
/// UV coordinates (for shader sampling).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShadowTile {
    /// X offset in pixels from atlas origin.
    pub offset_x: u32,
    /// Y offset in pixels from atlas origin.
    pub offset_y: u32,
    /// Tile size in pixels (same for width and height).
    pub size: u32,
    /// Normalized UV scale factor (size / atlas_size).
    pub uv_scale: f32,
    /// Normalized UV offset (offset / atlas_size).
    pub uv_offset: [f32; 2],
    /// The size tier this tile was allocated from.
    pub tier: TileSizeTier,
    /// The light ID this tile is assigned to.
    pub light_id: u32,
    /// Priority for eviction (lower = evict first).
    pub priority: u32,
}

impl ShadowTile {
    /// Create a new shadow tile.
    fn new(
        offset_x: u32,
        offset_y: u32,
        size: u32,
        atlas_size: u32,
        tier: TileSizeTier,
        light_id: u32,
        priority: u32,
    ) -> Self {
        let atlas_size_f = atlas_size as f32;
        Self {
            offset_x,
            offset_y,
            size,
            uv_scale: size as f32 / atlas_size_f,
            uv_offset: [offset_x as f32 / atlas_size_f, offset_y as f32 / atlas_size_f],
            tier,
            light_id,
            priority,
        }
    }

    /// Convert to GPU-compatible struct for buffer upload.
    #[inline]
    pub fn to_gpu(&self) -> ShadowTileGpu {
        ShadowTileGpu {
            uv_offset: self.uv_offset,
            uv_scale: [self.uv_scale, self.uv_scale],
        }
    }
}

// ---------------------------------------------------------------------------
// GPU-Compatible Struct
// ---------------------------------------------------------------------------

/// GPU-compatible shadow tile data for buffer upload.
///
/// Contains only the UV transformation data needed for shadow sampling
/// in shaders. This struct is 16 bytes aligned for efficient GPU access.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
pub struct ShadowTileGpu {
    /// Normalized atlas UV offset [u, v].
    pub uv_offset: [f32; 2],
    /// Normalized atlas UV scale [u_scale, v_scale].
    pub uv_scale: [f32; 2],
}

// Compile-time size check
const _: () = assert!(std::mem::size_of::<ShadowTileGpu>() == 16);

// ---------------------------------------------------------------------------
// Tile Slot
// ---------------------------------------------------------------------------

/// Internal representation of a slot in the tile grid.
#[derive(Debug, Clone, Copy)]
struct TileSlot {
    /// Grid X coordinate within the tier.
    grid_x: u32,
    /// Grid Y coordinate within the tier.
    grid_y: u32,
}

impl TileSlot {
    fn new(grid_x: u32, grid_y: u32) -> Self {
        Self { grid_x, grid_y }
    }

    /// Convert to pixel offset given tile size.
    fn to_pixel_offset(self, tile_size: u32) -> (u32, u32) {
        (self.grid_x * tile_size, self.grid_y * tile_size)
    }
}

// ---------------------------------------------------------------------------
// Atlas Statistics
// ---------------------------------------------------------------------------

/// Statistics about shadow atlas usage.
#[derive(Debug, Clone, Default)]
pub struct ShadowAtlasStats {
    /// Total tiles currently in use.
    pub tiles_used: u32,
    /// Free tiles per tier: [Small256, Medium512, Large1024, XLarge2048].
    pub tiles_free_per_tier: [u32; 4],
    /// Total tiles per tier.
    pub tiles_total_per_tier: [u32; 4],
    /// Fragmentation ratio (0.0 = no fragmentation, 1.0 = fully fragmented).
    /// Calculated as 1 - (largest_contiguous_block / total_free).
    pub fragmentation_ratio: f32,
    /// Total allocation count since last reset.
    pub total_allocations: u64,
    /// Total deallocation count since last reset.
    pub total_deallocations: u64,
    /// Number of tier fallbacks that occurred.
    pub tier_fallbacks: u64,
    /// Number of evictions performed.
    pub evictions: u64,
}

// ---------------------------------------------------------------------------
// Shadow Atlas
// ---------------------------------------------------------------------------

/// 2D bin-packing shadow atlas allocator.
///
/// Manages allocation and deallocation of shadow map tiles within a single
/// large atlas texture. Supports multiple tile size tiers with automatic
/// fallback to smaller tiers when requested sizes are exhausted.
pub struct ShadowAtlas {
    /// Atlas texture size in pixels (square).
    atlas_size: u32,

    /// Free-list per tier (FIFO queue for locality).
    free_lists: [VecDeque<TileSlot>; 4],

    /// Active allocations: light_id -> ShadowTile.
    allocations: HashMap<u32, ShadowTile>,

    /// Priority index for eviction: priority -> light_ids.
    /// Using BTreeMap for ordered iteration (lowest priority first).
    priority_index: BTreeMap<u32, Vec<u32>>,

    /// Statistics tracking.
    stats: ShadowAtlasStats,
}

impl ShadowAtlas {
    /// Create a new shadow atlas with the given size.
    ///
    /// # Arguments
    ///
    /// * `atlas_size` - Atlas texture size in pixels (must be power of 2, >= 1024).
    ///
    /// # Panics
    ///
    /// Panics if `atlas_size` is not a power of 2 or is less than 1024.
    pub fn new(atlas_size: u32) -> Self {
        assert!(atlas_size.is_power_of_two(), "atlas_size must be power of 2");
        assert!(atlas_size >= 1024, "atlas_size must be at least 1024");

        let mut atlas = Self {
            atlas_size,
            free_lists: [
                VecDeque::new(),
                VecDeque::new(),
                VecDeque::new(),
                VecDeque::new(),
            ],
            allocations: HashMap::new(),
            priority_index: BTreeMap::new(),
            stats: ShadowAtlasStats::default(),
        };

        atlas.initialize_free_lists();
        atlas
    }

    /// Create a new shadow atlas with default size (4096x4096).
    pub fn default_size() -> Self {
        Self::new(4096)
    }

    /// Get the atlas texture size.
    #[inline]
    pub fn atlas_size(&self) -> u32 {
        self.atlas_size
    }

    /// Initialize free lists for all tiers.
    fn initialize_free_lists(&mut self) {
        for tier in TileSizeTier::ALL {
            let tile_size = tier.size_pixels();
            let tiles_per_side = self.atlas_size / tile_size;
            let tier_idx = tier as usize;

            self.stats.tiles_total_per_tier[tier_idx] = tiles_per_side * tiles_per_side;
            self.stats.tiles_free_per_tier[tier_idx] = tiles_per_side * tiles_per_side;

            // Initialize all slots as free
            self.free_lists[tier_idx].clear();
            for y in 0..tiles_per_side {
                for x in 0..tiles_per_side {
                    self.free_lists[tier_idx].push_back(TileSlot::new(x, y));
                }
            }
        }
    }

    /// Allocate a shadow tile for a light.
    ///
    /// # Arguments
    ///
    /// * `size_tier` - Desired tile size tier.
    /// * `light_id` - Unique identifier for the light.
    /// * `priority` - Eviction priority (higher = more important, evicted last).
    ///
    /// # Returns
    ///
    /// `Some(ShadowTile)` if allocation succeeded, `None` if atlas is full.
    /// When the requested tier is exhausted, automatically falls back to
    /// smaller tiers.
    pub fn allocate(
        &mut self,
        size_tier: TileSizeTier,
        light_id: u32,
        priority: u32,
    ) -> Option<ShadowTile> {
        // Check if light already has an allocation
        if self.allocations.contains_key(&light_id) {
            return None;
        }

        // Try requested tier first, then fall back to smaller tiers
        let mut current_tier = Some(size_tier);

        while let Some(tier) = current_tier {
            let tier_idx = tier as usize;

            if let Some(slot) = self.free_lists[tier_idx].pop_front() {
                let tile_size = tier.size_pixels();
                let (offset_x, offset_y) = slot.to_pixel_offset(tile_size);

                let tile = ShadowTile::new(
                    offset_x,
                    offset_y,
                    tile_size,
                    self.atlas_size,
                    tier,
                    light_id,
                    priority,
                );

                // Track allocation
                self.allocations.insert(light_id, tile);
                self.priority_index
                    .entry(priority)
                    .or_insert_with(Vec::new)
                    .push(light_id);

                // Update stats
                self.stats.tiles_used += 1;
                self.stats.tiles_free_per_tier[tier_idx] -= 1;
                self.stats.total_allocations += 1;

                if tier != size_tier {
                    self.stats.tier_fallbacks += 1;
                }

                return Some(tile);
            }

            // Try next smaller tier
            current_tier = tier.smaller();
        }

        // All tiers exhausted
        None
    }

    /// Deallocate a shadow tile, returning it to the free-list.
    ///
    /// # Arguments
    ///
    /// * `tile` - The tile to deallocate.
    ///
    /// # Returns
    ///
    /// `true` if the tile was successfully deallocated, `false` if it wasn't found.
    pub fn deallocate(&mut self, tile: ShadowTile) -> bool {
        self.deallocate_by_light_id(tile.light_id)
    }

    /// Deallocate a shadow tile by light ID.
    ///
    /// # Arguments
    ///
    /// * `light_id` - The light ID whose tile should be deallocated.
    ///
    /// # Returns
    ///
    /// `true` if the tile was successfully deallocated, `false` if not found.
    pub fn deallocate_by_light_id(&mut self, light_id: u32) -> bool {
        if let Some(tile) = self.allocations.remove(&light_id) {
            let tier_idx = tile.tier as usize;
            let tile_size = tile.tier.size_pixels();

            // Return slot to free list
            let slot = TileSlot::new(tile.offset_x / tile_size, tile.offset_y / tile_size);
            self.free_lists[tier_idx].push_back(slot);

            // Update priority index
            if let Some(light_ids) = self.priority_index.get_mut(&tile.priority) {
                light_ids.retain(|&id| id != light_id);
                if light_ids.is_empty() {
                    self.priority_index.remove(&tile.priority);
                }
            }

            // Update stats
            self.stats.tiles_used -= 1;
            self.stats.tiles_free_per_tier[tier_idx] += 1;
            self.stats.total_deallocations += 1;

            true
        } else {
            false
        }
    }

    /// Evict the lowest priority light's tile.
    ///
    /// # Returns
    ///
    /// The evicted tile if one was found, `None` if the atlas is empty.
    pub fn evict_lowest_priority(&mut self) -> Option<ShadowTile> {
        // Find lowest priority (first key in BTreeMap)
        let lowest_priority = *self.priority_index.keys().next()?;

        // Get first light at this priority
        let light_id = {
            let light_ids = self.priority_index.get(&lowest_priority)?;
            *light_ids.first()?
        };

        // Get the tile before deallocating
        let tile = *self.allocations.get(&light_id)?;

        // Deallocate and update stats
        if self.deallocate_by_light_id(light_id) {
            self.stats.evictions += 1;
            Some(tile)
        } else {
            None
        }
    }

    /// Get the tile for a specific light.
    pub fn get_tile(&self, light_id: u32) -> Option<&ShadowTile> {
        self.allocations.get(&light_id)
    }

    /// Check if a light has an allocated tile.
    pub fn has_tile(&self, light_id: u32) -> bool {
        self.allocations.contains_key(&light_id)
    }

    /// Get the number of currently allocated tiles.
    #[inline]
    pub fn tiles_used(&self) -> u32 {
        self.stats.tiles_used
    }

    /// Get the number of free tiles for a specific tier.
    #[inline]
    pub fn tiles_free(&self, tier: TileSizeTier) -> u32 {
        self.stats.tiles_free_per_tier[tier as usize]
    }

    /// Get the total capacity for a specific tier.
    #[inline]
    pub fn tiles_total(&self, tier: TileSizeTier) -> u32 {
        self.stats.tiles_total_per_tier[tier as usize]
    }

    /// Check if a tier has any free tiles.
    #[inline]
    pub fn tier_has_space(&self, tier: TileSizeTier) -> bool {
        !self.free_lists[tier as usize].is_empty()
    }

    /// Check if the atlas is completely full.
    pub fn is_full(&self) -> bool {
        self.free_lists.iter().all(|list| list.is_empty())
    }

    /// Check if the atlas has any allocations.
    pub fn is_empty(&self) -> bool {
        self.allocations.is_empty()
    }

    /// Reset the atlas, clearing all allocations.
    pub fn reset(&mut self) {
        self.allocations.clear();
        self.priority_index.clear();
        self.initialize_free_lists();

        // Reset usage stats but keep totals
        self.stats.tiles_used = 0;
        self.stats.total_allocations = 0;
        self.stats.total_deallocations = 0;
        self.stats.tier_fallbacks = 0;
        self.stats.evictions = 0;
        self.stats.fragmentation_ratio = 0.0;
    }

    /// Get atlas statistics.
    pub fn stats(&self) -> &ShadowAtlasStats {
        &self.stats
    }

    /// Calculate and update the fragmentation ratio.
    ///
    /// Fragmentation is measured as the ratio of total free slots across
    /// all tiers versus the theoretical maximum contiguous block.
    pub fn update_fragmentation(&mut self) {
        let total_free: u32 = self.stats.tiles_free_per_tier.iter().sum();
        let total_capacity: u32 = self.stats.tiles_total_per_tier.iter().sum();

        if total_free == 0 || total_capacity == 0 {
            self.stats.fragmentation_ratio = 0.0;
            return;
        }

        // Find the largest contiguous run in each tier
        // For simplicity, we consider fragmentation as the inverse of tier utilization uniformity
        let mut largest_free_tier = 0u32;
        for &free in &self.stats.tiles_free_per_tier {
            if free > largest_free_tier {
                largest_free_tier = free;
            }
        }

        // Fragmentation = 1 - (largest_contiguous / total_free)
        // A perfectly unfragmented atlas would have all free space in one tier
        self.stats.fragmentation_ratio = 1.0 - (largest_free_tier as f32 / total_free as f32);
    }

    /// Get all allocated tiles (for iteration).
    pub fn allocated_tiles(&self) -> impl Iterator<Item = &ShadowTile> {
        self.allocations.values()
    }

    /// Get all light IDs with allocations.
    pub fn allocated_light_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.allocations.keys().copied()
    }

    /// Update the priority of an existing allocation.
    ///
    /// # Returns
    ///
    /// `true` if the priority was updated, `false` if the light wasn't found.
    pub fn update_priority(&mut self, light_id: u32, new_priority: u32) -> bool {
        if let Some(tile) = self.allocations.get_mut(&light_id) {
            let old_priority = tile.priority;

            if old_priority == new_priority {
                return true;
            }

            // Remove from old priority group
            if let Some(light_ids) = self.priority_index.get_mut(&old_priority) {
                light_ids.retain(|&id| id != light_id);
                if light_ids.is_empty() {
                    self.priority_index.remove(&old_priority);
                }
            }

            // Add to new priority group
            tile.priority = new_priority;
            self.priority_index
                .entry(new_priority)
                .or_insert_with(Vec::new)
                .push(light_id);

            true
        } else {
            false
        }
    }

    /// Get GPU data for all allocated tiles.
    ///
    /// Returns a vector of GPU-compatible tile data ordered by light ID.
    pub fn get_gpu_data(&self) -> Vec<ShadowTileGpu> {
        let mut result: Vec<_> = self
            .allocations
            .iter()
            .map(|(&light_id, tile)| (light_id, tile.to_gpu()))
            .collect();

        result.sort_by_key(|(id, _)| *id);
        result.into_iter().map(|(_, gpu)| gpu).collect()
    }

    /// Get GPU data as a map from light ID to tile data.
    pub fn get_gpu_data_map(&self) -> HashMap<u32, ShadowTileGpu> {
        self.allocations
            .iter()
            .map(|(&light_id, tile)| (light_id, tile.to_gpu()))
            .collect()
    }
}

impl Default for ShadowAtlas {
    fn default() -> Self {
        Self::default_size()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_size_tier_values() {
        assert_eq!(TileSizeTier::Small256.size_pixels(), 256);
        assert_eq!(TileSizeTier::Medium512.size_pixels(), 512);
        assert_eq!(TileSizeTier::Large1024.size_pixels(), 1024);
        assert_eq!(TileSizeTier::XLarge2048.size_pixels(), 2048);
    }

    #[test]
    fn test_tile_size_tier_smaller() {
        assert_eq!(TileSizeTier::XLarge2048.smaller(), Some(TileSizeTier::Large1024));
        assert_eq!(TileSizeTier::Large1024.smaller(), Some(TileSizeTier::Medium512));
        assert_eq!(TileSizeTier::Medium512.smaller(), Some(TileSizeTier::Small256));
        assert_eq!(TileSizeTier::Small256.smaller(), None);
    }

    #[test]
    fn test_atlas_creation() {
        let atlas = ShadowAtlas::new(4096);
        assert_eq!(atlas.atlas_size(), 4096);
        assert_eq!(atlas.tiles_used(), 0);
        assert!(atlas.is_empty());
        assert!(!atlas.is_full());
    }

    #[test]
    fn test_atlas_default_size() {
        let atlas = ShadowAtlas::default_size();
        assert_eq!(atlas.atlas_size(), 4096);
    }

    #[test]
    fn test_atlas_tier_capacities() {
        let atlas = ShadowAtlas::new(4096);

        // 4096 / 2048 = 2x2 = 4 tiles
        assert_eq!(atlas.tiles_total(TileSizeTier::XLarge2048), 4);
        // 4096 / 1024 = 4x4 = 16 tiles
        assert_eq!(atlas.tiles_total(TileSizeTier::Large1024), 16);
        // 4096 / 512 = 8x8 = 64 tiles
        assert_eq!(atlas.tiles_total(TileSizeTier::Medium512), 64);
        // 4096 / 256 = 16x16 = 256 tiles
        assert_eq!(atlas.tiles_total(TileSizeTier::Small256), 256);
    }

    #[test]
    fn test_allocation_returns_valid_tile() {
        let mut atlas = ShadowAtlas::new(4096);

        let tile = atlas.allocate(TileSizeTier::Large1024, 1, 100).unwrap();

        assert_eq!(tile.size, 1024);
        assert_eq!(tile.light_id, 1);
        assert_eq!(tile.priority, 100);
        assert_eq!(tile.tier, TileSizeTier::Large1024);
        assert!(tile.offset_x < 4096);
        assert!(tile.offset_y < 4096);
        assert!((tile.uv_scale - 0.25).abs() < 0.0001); // 1024/4096 = 0.25
    }

    #[test]
    fn test_allocation_uv_offset_correct() {
        let mut atlas = ShadowAtlas::new(4096);

        let tile = atlas.allocate(TileSizeTier::Large1024, 1, 100).unwrap();

        let expected_u = tile.offset_x as f32 / 4096.0;
        let expected_v = tile.offset_y as f32 / 4096.0;

        assert!((tile.uv_offset[0] - expected_u).abs() < 0.0001);
        assert!((tile.uv_offset[1] - expected_v).abs() < 0.0001);
    }

    #[test]
    fn test_deallocation_returns_to_free_list() {
        let mut atlas = ShadowAtlas::new(4096);

        let initial_free = atlas.tiles_free(TileSizeTier::Large1024);
        let tile = atlas.allocate(TileSizeTier::Large1024, 1, 100).unwrap();
        assert_eq!(atlas.tiles_free(TileSizeTier::Large1024), initial_free - 1);

        assert!(atlas.deallocate(tile));
        assert_eq!(atlas.tiles_free(TileSizeTier::Large1024), initial_free);
    }

    #[test]
    fn test_deallocation_by_light_id() {
        let mut atlas = ShadowAtlas::new(4096);

        atlas.allocate(TileSizeTier::Large1024, 42, 100).unwrap();
        assert!(atlas.has_tile(42));

        assert!(atlas.deallocate_by_light_id(42));
        assert!(!atlas.has_tile(42));
    }

    #[test]
    fn test_multiple_allocations_no_overlap() {
        let mut atlas = ShadowAtlas::new(4096);

        let mut tiles = Vec::new();
        for i in 0..16 {
            let tile = atlas
                .allocate(TileSizeTier::Large1024, i, 100)
                .expect("Should allocate 16 1024x1024 tiles in 4096 atlas");
            tiles.push(tile);
        }

        // Verify no overlaps
        for (i, a) in tiles.iter().enumerate() {
            for (j, b) in tiles.iter().enumerate() {
                if i != j {
                    let a_right = a.offset_x + a.size;
                    let a_bottom = a.offset_y + a.size;
                    let b_right = b.offset_x + b.size;
                    let b_bottom = b.offset_y + b.size;

                    // Check for overlap
                    let overlaps_x = a.offset_x < b_right && a_right > b.offset_x;
                    let overlaps_y = a.offset_y < b_bottom && a_bottom > b.offset_y;

                    assert!(
                        !(overlaps_x && overlaps_y),
                        "Tiles {} and {} overlap: {:?} vs {:?}",
                        i,
                        j,
                        a,
                        b
                    );
                }
            }
        }
    }

    #[test]
    fn test_tier_fallback_when_exhausted() {
        let mut atlas = ShadowAtlas::new(4096);

        // Exhaust XLarge2048 tier (only 4 slots)
        for i in 0..4 {
            atlas
                .allocate(TileSizeTier::XLarge2048, i, 100)
                .expect("Should allocate 4 XLarge tiles");
        }

        // Next allocation should fall back to Large1024
        let tile = atlas
            .allocate(TileSizeTier::XLarge2048, 100, 100)
            .expect("Should fall back to smaller tier");

        assert_eq!(tile.tier, TileSizeTier::Large1024);
        assert!(atlas.stats().tier_fallbacks > 0);
    }

    #[test]
    fn test_allocation_fails_when_completely_full() {
        let mut atlas = ShadowAtlas::new(1024); // Small atlas for testing

        // 1024 atlas has:
        // - 1 XLarge2048 slots (doesn't fit)
        // - 1 Large1024 slot
        // - 4 Medium512 slots
        // - 16 Small256 slots

        // Fill all tiers
        let mut light_id = 0u32;

        // Large1024: 1 slot
        atlas.allocate(TileSizeTier::Large1024, light_id, 100);
        light_id += 1;

        // Medium512: 4 slots
        for _ in 0..4 {
            atlas.allocate(TileSizeTier::Medium512, light_id, 100);
            light_id += 1;
        }

        // Small256: 16 slots
        for _ in 0..16 {
            atlas.allocate(TileSizeTier::Small256, light_id, 100);
            light_id += 1;
        }

        // Atlas should be full now
        assert!(atlas.is_full());

        // Further allocations should fail
        assert!(atlas.allocate(TileSizeTier::Small256, 999, 100).is_none());
    }

    #[test]
    fn test_eviction_removes_lowest_priority() {
        let mut atlas = ShadowAtlas::new(4096);

        atlas.allocate(TileSizeTier::Large1024, 1, 100).unwrap();
        atlas.allocate(TileSizeTier::Large1024, 2, 50).unwrap(); // Lowest priority
        atlas.allocate(TileSizeTier::Large1024, 3, 200).unwrap();

        let evicted = atlas.evict_lowest_priority().expect("Should evict a tile");

        assert_eq!(evicted.light_id, 2);
        assert_eq!(evicted.priority, 50);
        assert!(!atlas.has_tile(2));
        assert!(atlas.has_tile(1));
        assert!(atlas.has_tile(3));
    }

    #[test]
    fn test_eviction_on_empty_atlas() {
        let mut atlas = ShadowAtlas::new(4096);
        assert!(atlas.evict_lowest_priority().is_none());
    }

    #[test]
    fn test_fragment_free_after_full_cycle() {
        let mut atlas = ShadowAtlas::new(4096);

        let initial_free_large = atlas.tiles_free(TileSizeTier::Large1024);

        // Allocate all Large1024 tiles
        for i in 0..16 {
            atlas.allocate(TileSizeTier::Large1024, i, 100).unwrap();
        }

        assert_eq!(atlas.tiles_free(TileSizeTier::Large1024), 0);

        // Deallocate all
        for i in 0..16 {
            atlas.deallocate_by_light_id(i);
        }

        assert_eq!(atlas.tiles_free(TileSizeTier::Large1024), initial_free_large);
        assert!(atlas.is_empty());
    }

    #[test]
    fn test_atlas_reset_clears_all() {
        let mut atlas = ShadowAtlas::new(4096);

        // Allocate some tiles
        for i in 0..10 {
            atlas.allocate(TileSizeTier::Large1024, i, 100);
        }

        assert!(!atlas.is_empty());

        atlas.reset();

        assert!(atlas.is_empty());
        assert_eq!(atlas.tiles_used(), 0);
        assert_eq!(atlas.tiles_free(TileSizeTier::Large1024), 16);
        assert_eq!(atlas.stats().total_allocations, 0);
    }

    #[test]
    fn test_duplicate_light_id_rejected() {
        let mut atlas = ShadowAtlas::new(4096);

        atlas.allocate(TileSizeTier::Large1024, 1, 100).unwrap();

        // Same light ID should fail
        assert!(atlas.allocate(TileSizeTier::Large1024, 1, 200).is_none());
    }

    #[test]
    fn test_get_tile_returns_correct_tile() {
        let mut atlas = ShadowAtlas::new(4096);

        let allocated = atlas.allocate(TileSizeTier::Large1024, 42, 100).unwrap();
        let retrieved = atlas.get_tile(42).unwrap();

        assert_eq!(allocated.offset_x, retrieved.offset_x);
        assert_eq!(allocated.offset_y, retrieved.offset_y);
        assert_eq!(allocated.light_id, retrieved.light_id);
    }

    #[test]
    fn test_gpu_struct_size() {
        assert_eq!(std::mem::size_of::<ShadowTileGpu>(), 16);
    }

    #[test]
    fn test_tile_to_gpu_conversion() {
        let mut atlas = ShadowAtlas::new(4096);

        let tile = atlas.allocate(TileSizeTier::Large1024, 1, 100).unwrap();
        let gpu = tile.to_gpu();

        assert_eq!(gpu.uv_offset, tile.uv_offset);
        assert_eq!(gpu.uv_scale, [tile.uv_scale, tile.uv_scale]);
    }

    #[test]
    fn test_update_priority() {
        let mut atlas = ShadowAtlas::new(4096);

        atlas.allocate(TileSizeTier::Large1024, 1, 100).unwrap();
        atlas.allocate(TileSizeTier::Large1024, 2, 200).unwrap();

        assert!(atlas.update_priority(1, 300));

        // Light 2 should now have lowest priority
        let evicted = atlas.evict_lowest_priority().unwrap();
        assert_eq!(evicted.light_id, 2);
    }

    #[test]
    fn test_update_priority_nonexistent() {
        let mut atlas = ShadowAtlas::new(4096);
        assert!(!atlas.update_priority(999, 100));
    }

    #[test]
    fn test_stats_tracking() {
        let mut atlas = ShadowAtlas::new(4096);

        atlas.allocate(TileSizeTier::Large1024, 1, 100);
        atlas.allocate(TileSizeTier::Large1024, 2, 100);

        assert_eq!(atlas.stats().total_allocations, 2);
        assert_eq!(atlas.stats().tiles_used, 2);

        atlas.deallocate_by_light_id(1);

        assert_eq!(atlas.stats().total_deallocations, 1);
        assert_eq!(atlas.stats().tiles_used, 1);
    }

    #[test]
    fn test_fragmentation_calculation() {
        let mut atlas = ShadowAtlas::new(4096);

        // Allocate some tiles from different tiers to create fragmentation
        atlas.allocate(TileSizeTier::Large1024, 0, 100);
        atlas.allocate(TileSizeTier::Medium512, 1, 100);
        atlas.allocate(TileSizeTier::Small256, 2, 100);

        atlas.update_fragmentation();
        // Some fragmentation expected when tiles are spread across tiers
        assert!(atlas.stats().fragmentation_ratio >= 0.0);
        assert!(atlas.stats().fragmentation_ratio <= 1.0);

        // After deallocating all, fragmentation should be minimal
        atlas.deallocate_by_light_id(0);
        atlas.deallocate_by_light_id(1);
        atlas.deallocate_by_light_id(2);

        atlas.update_fragmentation();
        // With all tiers fully free, fragmentation should be low
        // (calculated as 1 - largest_free_tier / total_free)
        assert!(atlas.stats().fragmentation_ratio >= 0.0);
        assert!(atlas.stats().fragmentation_ratio < 1.0);
    }

    #[test]
    fn test_get_gpu_data() {
        let mut atlas = ShadowAtlas::new(4096);

        atlas.allocate(TileSizeTier::Large1024, 3, 100);
        atlas.allocate(TileSizeTier::Large1024, 1, 100);
        atlas.allocate(TileSizeTier::Large1024, 2, 100);

        let gpu_data = atlas.get_gpu_data();

        // Should be sorted by light_id
        assert_eq!(gpu_data.len(), 3);
        // Verify data is present (exact values depend on allocation order)
        assert!(gpu_data.iter().all(|d| d.uv_scale[0] > 0.0));
    }

    #[test]
    fn test_get_gpu_data_map() {
        let mut atlas = ShadowAtlas::new(4096);

        atlas.allocate(TileSizeTier::Large1024, 1, 100);
        atlas.allocate(TileSizeTier::Large1024, 2, 100);

        let gpu_map = atlas.get_gpu_data_map();

        assert!(gpu_map.contains_key(&1));
        assert!(gpu_map.contains_key(&2));
        assert!(!gpu_map.contains_key(&3));
    }

    #[test]
    fn test_allocated_tiles_iterator() {
        let mut atlas = ShadowAtlas::new(4096);

        atlas.allocate(TileSizeTier::Large1024, 1, 100);
        atlas.allocate(TileSizeTier::Large1024, 2, 100);

        let tiles: Vec<_> = atlas.allocated_tiles().collect();
        assert_eq!(tiles.len(), 2);
    }

    #[test]
    fn test_allocated_light_ids_iterator() {
        let mut atlas = ShadowAtlas::new(4096);

        atlas.allocate(TileSizeTier::Large1024, 10, 100);
        atlas.allocate(TileSizeTier::Large1024, 20, 100);

        let ids: Vec<_> = atlas.allocated_light_ids().collect();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&10));
        assert!(ids.contains(&20));
    }

    #[test]
    fn test_tier_has_space() {
        let mut atlas = ShadowAtlas::new(4096);

        assert!(atlas.tier_has_space(TileSizeTier::XLarge2048));

        // Fill XLarge tier (4 slots)
        for i in 0..4 {
            atlas.allocate(TileSizeTier::XLarge2048, i, 100);
        }

        assert!(!atlas.tier_has_space(TileSizeTier::XLarge2048));
        assert!(atlas.tier_has_space(TileSizeTier::Large1024));
    }

    #[test]
    fn test_different_atlas_sizes() {
        // Test 2048 atlas
        let atlas = ShadowAtlas::new(2048);
        assert_eq!(atlas.tiles_total(TileSizeTier::XLarge2048), 1);
        assert_eq!(atlas.tiles_total(TileSizeTier::Large1024), 4);

        // Test 8192 atlas
        let atlas = ShadowAtlas::new(8192);
        assert_eq!(atlas.tiles_total(TileSizeTier::XLarge2048), 16);
        assert_eq!(atlas.tiles_total(TileSizeTier::Large1024), 64);
    }

    #[test]
    #[should_panic(expected = "atlas_size must be power of 2")]
    fn test_invalid_atlas_size_not_power_of_two() {
        ShadowAtlas::new(3000);
    }

    #[test]
    #[should_panic(expected = "atlas_size must be at least 1024")]
    fn test_invalid_atlas_size_too_small() {
        ShadowAtlas::new(512);
    }

    #[test]
    fn test_bytemuck_pod_zeroable() {
        // Verify ShadowTileGpu is Pod and Zeroable
        let zeroed: ShadowTileGpu = bytemuck::Zeroable::zeroed();
        assert_eq!(zeroed.uv_offset, [0.0, 0.0]);
        assert_eq!(zeroed.uv_scale, [0.0, 0.0]);

        // Verify we can cast to bytes
        let gpu = ShadowTileGpu {
            uv_offset: [0.5, 0.25],
            uv_scale: [0.25, 0.25],
        };
        let bytes: &[u8] = bytemuck::bytes_of(&gpu);
        assert_eq!(bytes.len(), 16);
    }
}
