//! Virtual Texturing System
//!
//! Implements a GPU-driven virtual texturing system for efficient streaming of
//! large textures. The system uses a page-based approach where texture data is
//! divided into fixed-size pages that are loaded on demand.
//!
//! # Architecture
//!
//! - **VirtualPage**: Logical page coordinates in the virtual texture
//! - **PhysicalPage**: Location in the physical texture atlas
//! - **PageTable**: GPU buffer mapping virtual pages to physical locations
//! - **FeedbackBuffer**: GPU writes page requests, CPU reads back for streaming
//! - **PageCache**: LRU cache managing physical page residency
//! - **VirtualTextureManager**: Coordinates the entire system
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::virtual_texture::{VirtualTextureManager, VirtualTextureConfig};
//!
//! let config = VirtualTextureConfig {
//!     virtual_size: 16384,
//!     atlas_size: 4096,
//!     page_size: 128,
//!     max_mip_levels: 8,
//! };
//!
//! let mut manager = VirtualTextureManager::new(config);
//! manager.begin_frame();
//! // ... render scene, shader writes to feedback buffer ...
//! let requests = manager.process_feedback();
//! for request in requests {
//!     manager.load_page(request.page, &page_data);
//! }
//! manager.end_frame();
//! ```

use std::collections::{HashMap, HashSet, VecDeque};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default page size in pixels (128x128 is common for virtual texturing).
pub const PAGE_SIZE: u32 = 128;

/// Default border size for filtering across page boundaries.
pub const PAGE_BORDER: u32 = 4;

/// Maximum number of pages that can be loaded per frame.
pub const MAX_PAGES_PER_FRAME: usize = 32;

/// Default feedback buffer size (entries).
pub const DEFAULT_FEEDBACK_SIZE: usize = 65536;

// ---------------------------------------------------------------------------
// Virtual Page
// ---------------------------------------------------------------------------

/// Represents a logical page in the virtual texture.
///
/// Pages are identified by their (x, y) coordinates and mip level.
/// Coordinates are in page units, not pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VirtualPage {
    /// Page X coordinate in page units.
    pub x: u32,
    /// Page Y coordinate in page units.
    pub y: u32,
    /// Mip level (0 = highest detail).
    pub mip: u8,
}

impl VirtualPage {
    /// Create a new virtual page.
    pub fn new(x: u32, y: u32, mip: u8) -> Self {
        Self { x, y, mip }
    }

    /// Convert pixel coordinates to page coordinates for a given mip level.
    pub fn from_pixel_coords(pixel_x: u32, pixel_y: u32, mip: u8, page_size: u32) -> Self {
        // At each mip level, the texture is half the size
        let mip_scale = 1u32 << mip;
        let scaled_x = pixel_x / mip_scale;
        let scaled_y = pixel_y / mip_scale;

        Self {
            x: scaled_x / page_size,
            y: scaled_y / page_size,
            mip,
        }
    }

    /// Convert page coordinates back to pixel coordinates (top-left corner).
    pub fn to_pixel_coords(&self, page_size: u32) -> (u32, u32) {
        let mip_scale = 1u32 << self.mip;
        let pixel_x = self.x * page_size * mip_scale;
        let pixel_y = self.y * page_size * mip_scale;
        (pixel_x, pixel_y)
    }

    /// Pack page coordinates into a single u32 for GPU storage.
    /// Format: [mip:8][y:12][x:12]
    pub fn pack(&self) -> u32 {
        ((self.mip as u32) << 24) | ((self.y & 0xFFF) << 12) | (self.x & 0xFFF)
    }

    /// Unpack a u32 into page coordinates.
    pub fn unpack(packed: u32) -> Self {
        Self {
            x: packed & 0xFFF,
            y: (packed >> 12) & 0xFFF,
            mip: ((packed >> 24) & 0xFF) as u8,
        }
    }
}

// ---------------------------------------------------------------------------
// Physical Page
// ---------------------------------------------------------------------------

/// Represents a page slot in the physical texture atlas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhysicalPage {
    /// X position in the atlas (in page units).
    pub atlas_x: u32,
    /// Y position in the atlas (in page units).
    pub atlas_y: u32,
    /// Whether this page slot currently holds valid data.
    pub resident: bool,
}

impl PhysicalPage {
    /// Create a new physical page slot.
    pub fn new(atlas_x: u32, atlas_y: u32) -> Self {
        Self {
            atlas_x,
            atlas_y,
            resident: false,
        }
    }

    /// Mark this page as resident (containing valid data).
    pub fn set_resident(&mut self, resident: bool) {
        self.resident = resident;
    }

    /// Convert atlas coordinates to pixel coordinates.
    pub fn to_pixel_coords(&self, page_size: u32) -> (u32, u32) {
        (self.atlas_x * page_size, self.atlas_y * page_size)
    }

    /// Pack physical page info for GPU page table.
    /// Format: [resident:1][unused:7][y:12][x:12]
    pub fn pack(&self) -> u32 {
        let resident_bit = if self.resident { 1u32 << 31 } else { 0 };
        resident_bit | ((self.atlas_y & 0xFFF) << 12) | (self.atlas_x & 0xFFF)
    }

    /// Unpack GPU page table entry.
    pub fn unpack(packed: u32) -> Self {
        Self {
            atlas_x: packed & 0xFFF,
            atlas_y: (packed >> 12) & 0xFFF,
            resident: (packed >> 31) != 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Page Table
// ---------------------------------------------------------------------------

/// GPU page table storing virtual-to-physical page mappings.
///
/// The page table is a 3D array indexed by (x, y, mip) that returns
/// the physical location in the atlas.
#[derive(Debug)]
pub struct PageTable {
    /// Width in pages at mip 0.
    width: u32,
    /// Height in pages at mip 0.
    height: u32,
    /// Maximum mip levels.
    max_mip: u8,
    /// Page table entries: vec[mip][y * width_at_mip + x]
    entries: Vec<Vec<u32>>,
    /// Dirty flag for GPU upload.
    dirty: bool,
    /// Set of dirty pages for partial updates.
    dirty_pages: HashSet<VirtualPage>,
}

impl PageTable {
    /// Create a new page table for the given virtual texture dimensions.
    ///
    /// # Arguments
    ///
    /// * `virtual_width` - Virtual texture width in pixels.
    /// * `virtual_height` - Virtual texture height in pixels.
    /// * `page_size` - Page size in pixels.
    /// * `max_mip` - Maximum mip level to support.
    pub fn new(virtual_width: u32, virtual_height: u32, page_size: u32, max_mip: u8) -> Self {
        let width = (virtual_width + page_size - 1) / page_size;
        let height = (virtual_height + page_size - 1) / page_size;

        let mut entries = Vec::with_capacity(max_mip as usize + 1);

        for mip in 0..=max_mip {
            let mip_width = (width >> mip).max(1);
            let mip_height = (height >> mip).max(1);
            let count = (mip_width * mip_height) as usize;
            // Initialize with invalid mapping (not resident)
            entries.push(vec![0u32; count]);
        }

        Self {
            width,
            height,
            max_mip,
            entries,
            dirty: false,
            dirty_pages: HashSet::new(),
        }
    }

    /// Get the page table entry for a virtual page.
    pub fn get(&self, page: &VirtualPage) -> Option<PhysicalPage> {
        if page.mip > self.max_mip {
            return None;
        }

        let mip_width = (self.width >> page.mip).max(1);
        let mip_height = (self.height >> page.mip).max(1);

        if page.x >= mip_width || page.y >= mip_height {
            return None;
        }

        let index = (page.y * mip_width + page.x) as usize;
        self.entries.get(page.mip as usize)
            .and_then(|mip_entries| mip_entries.get(index))
            .map(|&packed| PhysicalPage::unpack(packed))
    }

    /// Set the page table entry for a virtual page.
    pub fn set(&mut self, page: &VirtualPage, physical: &PhysicalPage) {
        if page.mip > self.max_mip {
            return;
        }

        let mip_width = (self.width >> page.mip).max(1);
        let mip_height = (self.height >> page.mip).max(1);

        if page.x >= mip_width || page.y >= mip_height {
            return;
        }

        let index = (page.y * mip_width + page.x) as usize;
        if let Some(mip_entries) = self.entries.get_mut(page.mip as usize) {
            if let Some(entry) = mip_entries.get_mut(index) {
                *entry = physical.pack();
                self.dirty = true;
                self.dirty_pages.insert(*page);
            }
        }
    }

    /// Clear the residency flag for a virtual page.
    pub fn clear_residency(&mut self, page: &VirtualPage) {
        if let Some(mut physical) = self.get(page) {
            physical.resident = false;
            self.set(page, &physical);
        }
    }

    /// Check if the page table has pending updates.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Clear the dirty flag after GPU upload.
    pub fn clear_dirty(&mut self) {
        self.dirty = false;
        self.dirty_pages.clear();
    }

    /// Get the set of dirty pages for partial updates.
    pub fn dirty_pages(&self) -> &HashSet<VirtualPage> {
        &self.dirty_pages
    }

    /// Get the flat buffer for GPU upload at a specific mip level.
    pub fn get_mip_buffer(&self, mip: u8) -> Option<&[u32]> {
        self.entries.get(mip as usize).map(|v| v.as_slice())
    }

    /// Get dimensions at a specific mip level.
    pub fn dimensions_at_mip(&self, mip: u8) -> (u32, u32) {
        let w = (self.width >> mip).max(1);
        let h = (self.height >> mip).max(1);
        (w, h)
    }

    /// Total number of entries across all mip levels.
    pub fn total_entries(&self) -> usize {
        self.entries.iter().map(|v| v.len()).sum()
    }
}

// ---------------------------------------------------------------------------
// Feedback Buffer
// ---------------------------------------------------------------------------

/// Entry in the feedback buffer written by the GPU shader.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FeedbackEntry {
    /// Virtual page that was accessed.
    pub page: VirtualPage,
    /// Priority hint (lower = more urgent).
    pub priority: u8,
}

impl FeedbackEntry {
    /// Pack feedback entry for GPU storage.
    /// Format: [priority:8][mip:8][y:8][x:8]
    pub fn pack(&self) -> u32 {
        ((self.priority as u32) << 24)
            | ((self.page.mip as u32) << 16)
            | ((self.page.y as u32 & 0xFF) << 8)
            | (self.page.x as u32 & 0xFF)
    }

    /// Unpack GPU feedback entry.
    pub fn unpack(packed: u32) -> Self {
        Self {
            page: VirtualPage {
                x: packed & 0xFF,
                y: (packed >> 8) & 0xFF,
                mip: ((packed >> 16) & 0xFF) as u8,
            },
            priority: ((packed >> 24) & 0xFF) as u8,
        }
    }
}

/// Double-buffered feedback buffer for async GPU readback.
///
/// Frame N renders to buffer A while CPU reads buffer B from frame N-1.
pub struct FeedbackBuffer {
    /// Buffer capacity in entries.
    capacity: usize,
    /// Current write buffer index (0 or 1).
    write_index: usize,
    /// Double-buffered storage.
    buffers: [Vec<u32>; 2],
    /// Number of valid entries in each buffer.
    counts: [usize; 2],
    /// Frame counter for synchronization.
    frame: u64,
}

impl FeedbackBuffer {
    /// Create a new feedback buffer with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            write_index: 0,
            buffers: [vec![0u32; capacity], vec![0u32; capacity]],
            counts: [0, 0],
            frame: 0,
        }
    }

    /// Get the write buffer for the current frame (GPU writes here).
    pub fn write_buffer(&mut self) -> &mut [u32] {
        &mut self.buffers[self.write_index]
    }

    /// Get the read buffer from the previous frame (CPU reads here).
    pub fn read_buffer(&self) -> &[u32] {
        let read_index = 1 - self.write_index;
        &self.buffers[read_index][..self.counts[read_index]]
    }

    /// Set the number of valid entries in the current write buffer.
    pub fn set_write_count(&mut self, count: usize) {
        self.counts[self.write_index] = count.min(self.capacity);
    }

    /// Swap buffers at frame boundary.
    pub fn swap(&mut self) {
        self.write_index = 1 - self.write_index;
        self.counts[self.write_index] = 0;
        self.frame += 1;
    }

    /// Process the read buffer and return unique page requests sorted by priority.
    pub fn process(&self) -> Vec<FeedbackEntry> {
        let mut seen = HashSet::new();
        let mut entries = Vec::new();

        for &packed in self.read_buffer() {
            if packed == 0 {
                continue;
            }
            let entry = FeedbackEntry::unpack(packed);
            if seen.insert(entry.page) {
                entries.push(entry);
            }
        }

        // Sort by priority (lower = more urgent)
        entries.sort_by_key(|e| e.priority);
        entries
    }

    /// Clear all buffers.
    pub fn clear(&mut self) {
        for buf in &mut self.buffers {
            buf.fill(0);
        }
        self.counts = [0, 0];
    }

    /// Get the current frame number.
    pub fn frame(&self) -> u64 {
        self.frame
    }

    /// Get buffer capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

// ---------------------------------------------------------------------------
// Page Cache (LRU)
// ---------------------------------------------------------------------------

/// LRU entry tracking page usage.
#[derive(Debug, Clone)]
struct CacheEntry {
    /// Virtual page mapped to this slot.
    page: VirtualPage,
    /// Physical location in the atlas.
    physical: PhysicalPage,
    /// Last frame this page was accessed.
    last_access: u64,
    /// Load priority (lower = higher priority).
    priority: u8,
}

/// LRU cache managing physical page allocation and eviction.
pub struct PageCache {
    /// Total number of physical page slots.
    capacity: usize,
    /// Atlas dimensions in pages.
    atlas_width: u32,
    atlas_height: u32,
    /// Map from virtual page to cache entry.
    entries: HashMap<VirtualPage, CacheEntry>,
    /// Free physical page slots.
    free_slots: VecDeque<PhysicalPage>,
    /// Current frame for LRU tracking.
    current_frame: u64,
    /// Load queue for pending page loads.
    load_queue: VecDeque<(VirtualPage, u8)>,
}

impl PageCache {
    /// Create a new page cache for the given atlas dimensions.
    pub fn new(atlas_width: u32, atlas_height: u32) -> Self {
        let capacity = (atlas_width * atlas_height) as usize;

        // Initialize free slots
        let mut free_slots = VecDeque::with_capacity(capacity);
        for y in 0..atlas_height {
            for x in 0..atlas_width {
                free_slots.push_back(PhysicalPage::new(x, y));
            }
        }

        Self {
            capacity,
            atlas_width,
            atlas_height,
            entries: HashMap::new(),
            free_slots,
            current_frame: 0,
            load_queue: VecDeque::new(),
        }
    }

    /// Check if a page is currently resident in the cache.
    pub fn is_resident(&self, page: &VirtualPage) -> bool {
        self.entries.get(page).map_or(false, |e| e.physical.resident)
    }

    /// Get the physical location of a resident page.
    pub fn get_physical(&self, page: &VirtualPage) -> Option<&PhysicalPage> {
        self.entries.get(page).map(|e| &e.physical)
    }

    /// Touch a page to update its LRU status.
    pub fn touch(&mut self, page: &VirtualPage) {
        if let Some(entry) = self.entries.get_mut(page) {
            entry.last_access = self.current_frame;
        }
    }

    /// Request a page to be loaded with the given priority.
    pub fn request_load(&mut self, page: VirtualPage, priority: u8) {
        if !self.entries.contains_key(&page) {
            self.load_queue.push_back((page, priority));
        }
    }

    /// Allocate a physical slot for a page, evicting if necessary.
    ///
    /// Returns the allocated physical page and optionally the evicted virtual page.
    pub fn allocate(&mut self, page: VirtualPage) -> Option<(PhysicalPage, Option<VirtualPage>)> {
        // Check if already allocated
        if let Some(entry) = self.entries.get(&page) {
            return Some((entry.physical, None));
        }

        // Try to get a free slot
        if let Some(mut physical) = self.free_slots.pop_front() {
            physical.resident = true;
            self.entries.insert(page, CacheEntry {
                page,
                physical,
                last_access: self.current_frame,
                priority: 0,
            });
            return Some((physical, None));
        }

        // Need to evict - find LRU entry
        let evict_page = self.find_lru_page()?;
        let evicted_entry = self.entries.remove(&evict_page)?;

        let mut physical = evicted_entry.physical;
        physical.resident = true;

        self.entries.insert(page, CacheEntry {
            page,
            physical,
            last_access: self.current_frame,
            priority: 0,
        });

        Some((physical, Some(evict_page)))
    }

    /// Find the least recently used page for eviction.
    fn find_lru_page(&self) -> Option<VirtualPage> {
        self.entries
            .iter()
            .min_by_key(|(_, entry)| entry.last_access)
            .map(|(page, _)| *page)
    }

    /// Evict a specific page from the cache.
    pub fn evict(&mut self, page: &VirtualPage) -> Option<PhysicalPage> {
        if let Some(entry) = self.entries.remove(page) {
            let mut physical = entry.physical;
            physical.resident = false;
            self.free_slots.push_back(physical);
            Some(physical)
        } else {
            None
        }
    }

    /// Get the next page from the load queue.
    pub fn pop_load_request(&mut self) -> Option<(VirtualPage, u8)> {
        self.load_queue.pop_front()
    }

    /// Check if there are pending load requests.
    pub fn has_pending_loads(&self) -> bool {
        !self.load_queue.is_empty()
    }

    /// Get the number of pending load requests.
    pub fn pending_load_count(&self) -> usize {
        self.load_queue.len()
    }

    /// Advance to the next frame for LRU tracking.
    pub fn advance_frame(&mut self) {
        self.current_frame += 1;
    }

    /// Get the current frame number.
    pub fn current_frame(&self) -> u64 {
        self.current_frame
    }

    /// Get cache statistics.
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            capacity: self.capacity,
            resident_count: self.entries.len(),
            free_count: self.free_slots.len(),
            pending_loads: self.load_queue.len(),
            current_frame: self.current_frame,
        }
    }

    /// Clear all entries and reset the cache.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.load_queue.clear();

        // Reinitialize free slots
        self.free_slots.clear();
        for y in 0..self.atlas_height {
            for x in 0..self.atlas_width {
                self.free_slots.push_back(PhysicalPage::new(x, y));
            }
        }
    }
}

/// Cache statistics snapshot.
#[derive(Debug, Clone, Copy)]
pub struct CacheStats {
    /// Total cache capacity in pages.
    pub capacity: usize,
    /// Number of currently resident pages.
    pub resident_count: usize,
    /// Number of free page slots.
    pub free_count: usize,
    /// Number of pending load requests.
    pub pending_loads: usize,
    /// Current frame number.
    pub current_frame: u64,
}

// ---------------------------------------------------------------------------
// Virtual Texture Manager
// ---------------------------------------------------------------------------

/// Configuration for the virtual texture system.
#[derive(Debug, Clone)]
pub struct VirtualTextureConfig {
    /// Virtual texture size in pixels (assumed square).
    pub virtual_size: u32,
    /// Physical atlas size in pixels (assumed square).
    pub atlas_size: u32,
    /// Page size in pixels.
    pub page_size: u32,
    /// Maximum mip levels.
    pub max_mip_levels: u8,
    /// Feedback buffer size.
    pub feedback_capacity: usize,
}

impl Default for VirtualTextureConfig {
    fn default() -> Self {
        Self {
            virtual_size: 16384,
            atlas_size: 4096,
            page_size: PAGE_SIZE,
            max_mip_levels: 8,
            feedback_capacity: DEFAULT_FEEDBACK_SIZE,
        }
    }
}

/// Page load request with data.
#[derive(Debug)]
pub struct PageLoadRequest {
    /// Virtual page to load.
    pub page: VirtualPage,
    /// Physical location in the atlas.
    pub physical: PhysicalPage,
    /// Load priority.
    pub priority: u8,
}

/// Coordinates the virtual texturing system.
pub struct VirtualTextureManager {
    /// System configuration.
    config: VirtualTextureConfig,
    /// Page table for virtual-to-physical mapping.
    page_table: PageTable,
    /// Feedback buffer for GPU requests.
    feedback: FeedbackBuffer,
    /// LRU page cache.
    cache: PageCache,
    /// Pages loaded this frame.
    pages_loaded_this_frame: usize,
    /// Frame counter.
    frame: u64,
}

impl VirtualTextureManager {
    /// Create a new virtual texture manager with the given configuration.
    pub fn new(config: VirtualTextureConfig) -> Self {
        let atlas_pages = config.atlas_size / config.page_size;

        Self {
            page_table: PageTable::new(
                config.virtual_size,
                config.virtual_size,
                config.page_size,
                config.max_mip_levels,
            ),
            feedback: FeedbackBuffer::new(config.feedback_capacity),
            cache: PageCache::new(atlas_pages, atlas_pages),
            pages_loaded_this_frame: 0,
            frame: 0,
            config,
        }
    }

    /// Begin a new frame.
    pub fn begin_frame(&mut self) {
        self.feedback.swap();
        self.cache.advance_frame();
        self.pages_loaded_this_frame = 0;
        self.frame += 1;
    }

    /// Process feedback buffer and return page load requests.
    ///
    /// Returns up to `MAX_PAGES_PER_FRAME` load requests. Excess requests are
    /// queued internally and will be processed in subsequent frames.
    pub fn process_feedback(&mut self) -> Vec<PageLoadRequest> {
        let entries = self.feedback.process();
        let mut requests = Vec::new();
        let mut requests_this_call = 0usize;

        for entry in entries {
            if requests_this_call >= MAX_PAGES_PER_FRAME {
                // Queue remaining for next frame
                self.cache.request_load(entry.page, entry.priority);
                continue;
            }

            if self.cache.is_resident(&entry.page) {
                // Already resident, just touch for LRU
                self.cache.touch(&entry.page);
                continue;
            }

            // Allocate physical slot
            if let Some((physical, evicted)) = self.cache.allocate(entry.page) {
                // Update page table for evicted page
                if let Some(evicted_page) = evicted {
                    self.page_table.clear_residency(&evicted_page);
                }

                requests.push(PageLoadRequest {
                    page: entry.page,
                    physical,
                    priority: entry.priority,
                });
                requests_this_call += 1;
            }
        }

        requests
    }

    /// Mark a page as loaded and update the page table.
    pub fn page_loaded(&mut self, page: &VirtualPage, physical: &PhysicalPage) {
        let mut phys = *physical;
        phys.resident = true;
        self.page_table.set(page, &phys);
        self.pages_loaded_this_frame += 1;
    }

    /// End the current frame.
    pub fn end_frame(&mut self) {
        // Process any remaining queued loads for next frame
        while self.cache.has_pending_loads() && self.pages_loaded_this_frame < MAX_PAGES_PER_FRAME {
            if let Some((page, _priority)) = self.cache.pop_load_request() {
                if let Some((physical, evicted)) = self.cache.allocate(page) {
                    if let Some(evicted_page) = evicted {
                        self.page_table.clear_residency(&evicted_page);
                    }
                    let mut phys = physical;
                    phys.resident = true;
                    self.page_table.set(&page, &phys);
                    self.pages_loaded_this_frame += 1;
                }
            } else {
                break;
            }
        }
    }

    /// Get the page table for GPU upload.
    pub fn page_table(&self) -> &PageTable {
        &self.page_table
    }

    /// Get mutable access to the page table.
    pub fn page_table_mut(&mut self) -> &mut PageTable {
        &mut self.page_table
    }

    /// Get the feedback buffer for GPU readback.
    pub fn feedback_buffer(&mut self) -> &mut FeedbackBuffer {
        &mut self.feedback
    }

    /// Get the page cache.
    pub fn cache(&self) -> &PageCache {
        &self.cache
    }

    /// Get mutable access to the page cache.
    pub fn cache_mut(&mut self) -> &mut PageCache {
        &mut self.cache
    }

    /// Get the configuration.
    pub fn config(&self) -> &VirtualTextureConfig {
        &self.config
    }

    /// Get the current frame number.
    pub fn frame(&self) -> u64 {
        self.frame
    }

    /// Get system statistics.
    pub fn stats(&self) -> VirtualTextureStats {
        let cache_stats = self.cache.stats();
        VirtualTextureStats {
            frame: self.frame,
            pages_loaded_this_frame: self.pages_loaded_this_frame,
            resident_pages: cache_stats.resident_count,
            free_pages: cache_stats.free_count,
            pending_loads: cache_stats.pending_loads,
            page_table_dirty: self.page_table.is_dirty(),
        }
    }

    /// Reset the entire system.
    pub fn reset(&mut self) {
        self.page_table = PageTable::new(
            self.config.virtual_size,
            self.config.virtual_size,
            self.config.page_size,
            self.config.max_mip_levels,
        );
        self.feedback.clear();
        self.cache.clear();
        self.pages_loaded_this_frame = 0;
        self.frame = 0;
    }
}

/// Virtual texture system statistics.
#[derive(Debug, Clone, Copy)]
pub struct VirtualTextureStats {
    /// Current frame number.
    pub frame: u64,
    /// Pages loaded this frame.
    pub pages_loaded_this_frame: usize,
    /// Currently resident pages.
    pub resident_pages: usize,
    /// Free page slots.
    pub free_pages: usize,
    /// Pending load requests.
    pub pending_loads: usize,
    /// Whether the page table needs GPU upload.
    pub page_table_dirty: bool,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // VirtualPage Tests
    // =========================================================================

    #[test]
    fn test_virtual_page_creation() {
        let page = VirtualPage::new(5, 10, 2);
        assert_eq!(page.x, 5);
        assert_eq!(page.y, 10);
        assert_eq!(page.mip, 2);
    }

    #[test]
    fn test_virtual_page_from_pixel_coords() {
        // At mip 0, 256 pixels / 128 page_size = page 2
        let page = VirtualPage::from_pixel_coords(256, 384, 0, PAGE_SIZE);
        assert_eq!(page.x, 2);
        assert_eq!(page.y, 3);
        assert_eq!(page.mip, 0);
    }

    #[test]
    fn test_virtual_page_from_pixel_coords_with_mip() {
        // At mip 1, coordinates are halved before division
        // pixel 512 at mip 1 -> scaled to 256 -> page 2
        let page = VirtualPage::from_pixel_coords(512, 256, 1, PAGE_SIZE);
        assert_eq!(page.x, 2);
        assert_eq!(page.y, 1);
        assert_eq!(page.mip, 1);
    }

    #[test]
    fn test_virtual_page_to_pixel_coords() {
        let page = VirtualPage::new(2, 3, 0);
        let (px, py) = page.to_pixel_coords(PAGE_SIZE);
        assert_eq!(px, 256);
        assert_eq!(py, 384);
    }

    #[test]
    fn test_virtual_page_to_pixel_coords_with_mip() {
        let page = VirtualPage::new(1, 2, 2);
        let (px, py) = page.to_pixel_coords(PAGE_SIZE);
        // mip 2 scales by 4x
        assert_eq!(px, 1 * PAGE_SIZE * 4);
        assert_eq!(py, 2 * PAGE_SIZE * 4);
    }

    #[test]
    fn test_virtual_page_pack_unpack() {
        let original = VirtualPage::new(0xABC, 0x123, 0x45);
        let packed = original.pack();
        let unpacked = VirtualPage::unpack(packed);

        assert_eq!(unpacked.x, original.x & 0xFFF);
        assert_eq!(unpacked.y, original.y & 0xFFF);
        assert_eq!(unpacked.mip, original.mip);
    }

    #[test]
    fn test_virtual_page_pack_format() {
        let page = VirtualPage::new(0x123, 0x456, 0x78);
        let packed = page.pack();

        // Format: [mip:8][y:12][x:12]
        assert_eq!(packed & 0xFFF, 0x123);
        assert_eq!((packed >> 12) & 0xFFF, 0x456);
        assert_eq!((packed >> 24) & 0xFF, 0x78);
    }

    // =========================================================================
    // PhysicalPage Tests
    // =========================================================================

    #[test]
    fn test_physical_page_creation() {
        let page = PhysicalPage::new(3, 7);
        assert_eq!(page.atlas_x, 3);
        assert_eq!(page.atlas_y, 7);
        assert!(!page.resident);
    }

    #[test]
    fn test_physical_page_set_resident() {
        let mut page = PhysicalPage::new(1, 2);
        assert!(!page.resident);

        page.set_resident(true);
        assert!(page.resident);

        page.set_resident(false);
        assert!(!page.resident);
    }

    #[test]
    fn test_physical_page_to_pixel_coords() {
        let page = PhysicalPage::new(2, 3);
        let (px, py) = page.to_pixel_coords(PAGE_SIZE);
        assert_eq!(px, 256);
        assert_eq!(py, 384);
    }

    #[test]
    fn test_physical_page_pack_unpack() {
        let mut original = PhysicalPage::new(0x123, 0x456);
        original.resident = true;

        let packed = original.pack();
        let unpacked = PhysicalPage::unpack(packed);

        assert_eq!(unpacked.atlas_x, original.atlas_x);
        assert_eq!(unpacked.atlas_y, original.atlas_y);
        assert_eq!(unpacked.resident, original.resident);
    }

    #[test]
    fn test_physical_page_pack_residency_bit() {
        let mut page = PhysicalPage::new(0, 0);
        page.resident = true;
        let packed = page.pack();
        assert!((packed >> 31) != 0, "resident bit should be set");

        page.resident = false;
        let packed = page.pack();
        assert!((packed >> 31) == 0, "resident bit should be clear");
    }

    // =========================================================================
    // PageTable Tests
    // =========================================================================

    #[test]
    fn test_page_table_creation() {
        let table = PageTable::new(2048, 2048, PAGE_SIZE, 4);
        let (w, h) = table.dimensions_at_mip(0);
        assert_eq!(w, 16); // 2048 / 128
        assert_eq!(h, 16);
    }

    #[test]
    fn test_page_table_get_set() {
        let mut table = PageTable::new(1024, 1024, PAGE_SIZE, 3);

        let vpage = VirtualPage::new(2, 3, 1);
        let mut ppage = PhysicalPage::new(5, 6);
        ppage.resident = true;

        table.set(&vpage, &ppage);

        let retrieved = table.get(&vpage).unwrap();
        assert_eq!(retrieved.atlas_x, 5);
        assert_eq!(retrieved.atlas_y, 6);
        assert!(retrieved.resident);
    }

    #[test]
    fn test_page_table_out_of_bounds() {
        let table = PageTable::new(1024, 1024, PAGE_SIZE, 3);

        // Out of bounds mip
        let page = VirtualPage::new(0, 0, 10);
        assert!(table.get(&page).is_none());

        // Out of bounds coordinates
        let page = VirtualPage::new(100, 0, 0);
        assert!(table.get(&page).is_none());
    }

    #[test]
    fn test_page_table_dirty_flag() {
        let mut table = PageTable::new(1024, 1024, PAGE_SIZE, 3);
        assert!(!table.is_dirty());

        let vpage = VirtualPage::new(0, 0, 0);
        let ppage = PhysicalPage::new(0, 0);
        table.set(&vpage, &ppage);

        assert!(table.is_dirty());
        assert!(table.dirty_pages().contains(&vpage));

        table.clear_dirty();
        assert!(!table.is_dirty());
        assert!(table.dirty_pages().is_empty());
    }

    #[test]
    fn test_page_table_clear_residency() {
        let mut table = PageTable::new(1024, 1024, PAGE_SIZE, 3);

        let vpage = VirtualPage::new(1, 1, 0);
        let mut ppage = PhysicalPage::new(2, 2);
        ppage.resident = true;

        table.set(&vpage, &ppage);
        assert!(table.get(&vpage).unwrap().resident);

        table.clear_residency(&vpage);
        assert!(!table.get(&vpage).unwrap().resident);
    }

    #[test]
    fn test_page_table_mip_dimensions() {
        let table = PageTable::new(2048, 2048, PAGE_SIZE, 4);

        let (w0, h0) = table.dimensions_at_mip(0);
        assert_eq!(w0, 16);
        assert_eq!(h0, 16);

        let (w1, h1) = table.dimensions_at_mip(1);
        assert_eq!(w1, 8);
        assert_eq!(h1, 8);

        let (w2, h2) = table.dimensions_at_mip(2);
        assert_eq!(w2, 4);
        assert_eq!(h2, 4);
    }

    #[test]
    fn test_page_table_get_mip_buffer() {
        let table = PageTable::new(256, 256, PAGE_SIZE, 2);

        let buf = table.get_mip_buffer(0).unwrap();
        assert_eq!(buf.len(), 4); // 256/128 = 2, 2*2 = 4

        let buf = table.get_mip_buffer(1).unwrap();
        assert_eq!(buf.len(), 1); // 1*1 = 1

        assert!(table.get_mip_buffer(10).is_none());
    }

    // =========================================================================
    // FeedbackBuffer Tests
    // =========================================================================

    #[test]
    fn test_feedback_buffer_creation() {
        let fb = FeedbackBuffer::new(1024);
        assert_eq!(fb.capacity(), 1024);
        assert_eq!(fb.frame(), 0);
    }

    #[test]
    fn test_feedback_buffer_write_read() {
        let mut fb = FeedbackBuffer::new(100);

        // Write to buffer
        let entry = FeedbackEntry {
            page: VirtualPage::new(1, 2, 3),
            priority: 5,
        };
        fb.write_buffer()[0] = entry.pack();
        fb.set_write_count(1);

        // Swap buffers
        fb.swap();

        // Read from previous buffer
        let entries = fb.process();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].page.x, 1);
        assert_eq!(entries[0].page.y, 2);
        assert_eq!(entries[0].page.mip, 3);
        assert_eq!(entries[0].priority, 5);
    }

    #[test]
    fn test_feedback_buffer_double_buffering() {
        let mut fb = FeedbackBuffer::new(10);

        // Frame 0: write to buffer A
        fb.write_buffer()[0] = 1;
        fb.set_write_count(1);

        // Frame 1: swap, write to buffer B, read from A
        fb.swap();
        assert_eq!(fb.frame(), 1);
        assert_eq!(fb.read_buffer().len(), 1);

        fb.write_buffer()[0] = 2;
        fb.set_write_count(1);

        // Frame 2: swap, read from B
        fb.swap();
        assert_eq!(fb.frame(), 2);
    }

    #[test]
    fn test_feedback_buffer_deduplication() {
        let mut fb = FeedbackBuffer::new(100);

        let page = VirtualPage::new(1, 1, 0);
        let entry1 = FeedbackEntry { page, priority: 5 };
        let entry2 = FeedbackEntry { page, priority: 3 }; // Same page, different priority

        let write_buf = fb.write_buffer();
        write_buf[0] = entry1.pack();
        write_buf[1] = entry2.pack();
        fb.set_write_count(2);

        fb.swap();
        let entries = fb.process();

        // Should deduplicate to single entry
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn test_feedback_buffer_priority_sort() {
        let mut fb = FeedbackBuffer::new(100);

        let entries_in = [
            FeedbackEntry { page: VirtualPage::new(0, 0, 0), priority: 10 },
            FeedbackEntry { page: VirtualPage::new(1, 1, 0), priority: 5 },
            FeedbackEntry { page: VirtualPage::new(2, 2, 0), priority: 1 },
        ];

        let write_buf = fb.write_buffer();
        for (i, e) in entries_in.iter().enumerate() {
            write_buf[i] = e.pack();
        }
        fb.set_write_count(3);

        fb.swap();
        let sorted = fb.process();

        assert_eq!(sorted.len(), 3);
        assert_eq!(sorted[0].priority, 1);  // Lowest priority first
        assert_eq!(sorted[1].priority, 5);
        assert_eq!(sorted[2].priority, 10);
    }

    #[test]
    fn test_feedback_buffer_clear() {
        let mut fb = FeedbackBuffer::new(100);

        fb.write_buffer()[0] = 123;
        fb.set_write_count(1);
        fb.swap();

        fb.clear();

        assert_eq!(fb.read_buffer().len(), 0);
    }

    // =========================================================================
    // PageCache Tests
    // =========================================================================

    #[test]
    fn test_page_cache_creation() {
        let cache = PageCache::new(4, 4);
        let stats = cache.stats();
        assert_eq!(stats.capacity, 16);
        assert_eq!(stats.free_count, 16);
        assert_eq!(stats.resident_count, 0);
    }

    #[test]
    fn test_page_cache_allocate() {
        let mut cache = PageCache::new(4, 4);

        let page = VirtualPage::new(0, 0, 0);
        let (physical, evicted) = cache.allocate(page).unwrap();

        assert!(physical.resident);
        assert!(evicted.is_none());
        assert!(cache.is_resident(&page));
    }

    #[test]
    fn test_page_cache_allocate_same_page() {
        let mut cache = PageCache::new(4, 4);

        let page = VirtualPage::new(0, 0, 0);
        let (phys1, _) = cache.allocate(page).unwrap();
        let (phys2, _) = cache.allocate(page).unwrap();

        // Should return same physical location
        assert_eq!(phys1.atlas_x, phys2.atlas_x);
        assert_eq!(phys1.atlas_y, phys2.atlas_y);
    }

    #[test]
    fn test_page_cache_lru_eviction() {
        let mut cache = PageCache::new(2, 2); // Only 4 slots

        // Fill cache
        for i in 0..4 {
            let page = VirtualPage::new(i, 0, 0);
            cache.allocate(page);
            cache.advance_frame();
        }

        // Allocate one more, should evict LRU (page 0)
        let new_page = VirtualPage::new(10, 0, 0);
        let (_, evicted) = cache.allocate(new_page).unwrap();

        assert!(evicted.is_some());
        let evicted_page = evicted.unwrap();
        assert_eq!(evicted_page.x, 0);
    }

    #[test]
    fn test_page_cache_touch_updates_lru() {
        let mut cache = PageCache::new(2, 2);

        let page0 = VirtualPage::new(0, 0, 0);
        let page1 = VirtualPage::new(1, 0, 0);
        let page2 = VirtualPage::new(2, 0, 0);
        let page3 = VirtualPage::new(3, 0, 0);

        cache.allocate(page0);
        cache.advance_frame();
        cache.allocate(page1);
        cache.advance_frame();
        cache.allocate(page2);
        cache.advance_frame();
        cache.allocate(page3);
        cache.advance_frame();

        // Touch page0 to make it recently used
        cache.touch(&page0);

        // Allocate new page, should evict page1 (now LRU)
        let new_page = VirtualPage::new(10, 0, 0);
        let (_, evicted) = cache.allocate(new_page).unwrap();

        assert_eq!(evicted.unwrap().x, 1);
    }

    #[test]
    fn test_page_cache_evict() {
        let mut cache = PageCache::new(4, 4);

        let page = VirtualPage::new(0, 0, 0);
        cache.allocate(page);

        assert!(cache.is_resident(&page));

        let evicted_physical = cache.evict(&page);
        assert!(evicted_physical.is_some());
        assert!(!cache.is_resident(&page));
    }

    #[test]
    fn test_page_cache_load_queue() {
        let mut cache = PageCache::new(4, 4);

        let page = VirtualPage::new(5, 5, 0);
        cache.request_load(page, 3);

        assert!(cache.has_pending_loads());
        assert_eq!(cache.pending_load_count(), 1);

        let (popped_page, priority) = cache.pop_load_request().unwrap();
        assert_eq!(popped_page.x, 5);
        assert_eq!(priority, 3);

        assert!(!cache.has_pending_loads());
    }

    #[test]
    fn test_page_cache_clear() {
        let mut cache = PageCache::new(4, 4);

        for i in 0..4 {
            cache.allocate(VirtualPage::new(i, 0, 0));
        }
        cache.request_load(VirtualPage::new(10, 0, 0), 1);

        cache.clear();

        let stats = cache.stats();
        assert_eq!(stats.resident_count, 0);
        assert_eq!(stats.free_count, 16);
        assert_eq!(stats.pending_loads, 0);
    }

    #[test]
    fn test_page_cache_get_physical() {
        let mut cache = PageCache::new(4, 4);

        let page = VirtualPage::new(1, 2, 0);
        cache.allocate(page);

        let physical = cache.get_physical(&page);
        assert!(physical.is_some());
        assert!(physical.unwrap().resident);

        let non_existent = VirtualPage::new(100, 100, 0);
        assert!(cache.get_physical(&non_existent).is_none());
    }

    // =========================================================================
    // VirtualTextureManager Tests
    // =========================================================================

    #[test]
    fn test_manager_creation() {
        let config = VirtualTextureConfig::default();
        let manager = VirtualTextureManager::new(config.clone());

        assert_eq!(manager.frame(), 0);
        assert_eq!(manager.config().virtual_size, config.virtual_size);
    }

    #[test]
    fn test_manager_begin_end_frame() {
        let config = VirtualTextureConfig {
            virtual_size: 1024,
            atlas_size: 512,
            page_size: PAGE_SIZE,
            max_mip_levels: 3,
            feedback_capacity: 100,
        };
        let mut manager = VirtualTextureManager::new(config);

        manager.begin_frame();
        assert_eq!(manager.frame(), 1);

        manager.end_frame();

        manager.begin_frame();
        assert_eq!(manager.frame(), 2);
    }

    #[test]
    fn test_manager_process_feedback() {
        let config = VirtualTextureConfig {
            virtual_size: 1024,
            atlas_size: 512,
            page_size: PAGE_SIZE,
            max_mip_levels: 3,
            feedback_capacity: 100,
        };
        let mut manager = VirtualTextureManager::new(config);

        // Write feedback entries
        let entry = FeedbackEntry {
            page: VirtualPage::new(1, 1, 0),
            priority: 5,
        };
        manager.feedback_buffer().write_buffer()[0] = entry.pack();
        manager.feedback_buffer().set_write_count(1);

        manager.begin_frame();

        let requests = manager.process_feedback();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].page.x, 1);
        assert_eq!(requests[0].page.y, 1);
    }

    #[test]
    fn test_manager_page_loaded() {
        let config = VirtualTextureConfig {
            virtual_size: 1024,
            atlas_size: 512,
            page_size: PAGE_SIZE,
            max_mip_levels: 3,
            feedback_capacity: 100,
        };
        let mut manager = VirtualTextureManager::new(config);

        let page = VirtualPage::new(0, 0, 0);
        let physical = PhysicalPage::new(0, 0);

        manager.page_loaded(&page, &physical);

        let table_entry = manager.page_table().get(&page).unwrap();
        assert!(table_entry.resident);
    }

    #[test]
    fn test_manager_stats() {
        let config = VirtualTextureConfig {
            virtual_size: 1024,
            atlas_size: 512,
            page_size: PAGE_SIZE,
            max_mip_levels: 3,
            feedback_capacity: 100,
        };
        let manager = VirtualTextureManager::new(config);

        let stats = manager.stats();
        assert_eq!(stats.frame, 0);
        assert_eq!(stats.pages_loaded_this_frame, 0);
        assert!(!stats.page_table_dirty);
    }

    #[test]
    fn test_manager_reset() {
        let config = VirtualTextureConfig {
            virtual_size: 1024,
            atlas_size: 512,
            page_size: PAGE_SIZE,
            max_mip_levels: 3,
            feedback_capacity: 100,
        };
        let mut manager = VirtualTextureManager::new(config);

        manager.begin_frame();
        manager.begin_frame();

        let page = VirtualPage::new(0, 0, 0);
        let physical = PhysicalPage::new(0, 0);
        manager.page_loaded(&page, &physical);

        manager.reset();

        assert_eq!(manager.frame(), 0);
        assert!(!manager.page_table().is_dirty());
    }

    // =========================================================================
    // FeedbackEntry Tests
    // =========================================================================

    #[test]
    fn test_feedback_entry_pack_unpack() {
        let entry = FeedbackEntry {
            page: VirtualPage::new(0x12, 0x34, 0x56),
            priority: 0x78,
        };

        let packed = entry.pack();
        let unpacked = FeedbackEntry::unpack(packed);

        assert_eq!(unpacked.page.x, entry.page.x & 0xFF);
        assert_eq!(unpacked.page.y, entry.page.y & 0xFF);
        assert_eq!(unpacked.page.mip, entry.page.mip);
        assert_eq!(unpacked.priority, entry.priority);
    }

    // =========================================================================
    // Integration Tests
    // =========================================================================

    #[test]
    fn test_full_streaming_workflow() {
        let config = VirtualTextureConfig {
            virtual_size: 512,
            atlas_size: 256,
            page_size: PAGE_SIZE,
            max_mip_levels: 2,
            feedback_capacity: 100,
        };
        let mut manager = VirtualTextureManager::new(config);

        // Simulate multiple frames
        for frame in 0..3 {
            // Write feedback simulating GPU page requests
            let entry = FeedbackEntry {
                page: VirtualPage::new(frame, 0, 0),
                priority: 1,
            };
            manager.feedback_buffer().write_buffer()[0] = entry.pack();
            manager.feedback_buffer().set_write_count(1);

            manager.begin_frame();

            // Process feedback and load pages
            let requests = manager.process_feedback();
            for req in &requests {
                manager.page_loaded(&req.page, &req.physical);
            }

            manager.end_frame();
        }

        // Verify pages are resident
        for i in 0..3 {
            let page = VirtualPage::new(i, 0, 0);
            assert!(manager.cache().is_resident(&page), "page {} should be resident", i);
        }
    }

    #[test]
    fn test_max_pages_per_frame_limit() {
        let config = VirtualTextureConfig {
            virtual_size: 4096,
            atlas_size: 1024,
            page_size: PAGE_SIZE,
            max_mip_levels: 4,
            feedback_capacity: 1000,
        };
        let mut manager = VirtualTextureManager::new(config);

        // Write many feedback entries (more than MAX_PAGES_PER_FRAME)
        let write_buf = manager.feedback_buffer().write_buffer();
        for i in 0..100u32 {
            let entry = FeedbackEntry {
                page: VirtualPage::new(i, 0, 0),
                priority: 1,
            };
            write_buf[i as usize] = entry.pack();
        }
        manager.feedback_buffer().set_write_count(100);

        manager.begin_frame();
        let requests = manager.process_feedback();

        // Should be limited by MAX_PAGES_PER_FRAME
        assert!(
            requests.len() <= MAX_PAGES_PER_FRAME,
            "expected at most {} requests, got {}",
            MAX_PAGES_PER_FRAME,
            requests.len()
        );

        // Verify excess requests were queued for later
        assert!(
            manager.cache().pending_load_count() > 0,
            "excess requests should be queued for next frame"
        );
    }

    #[test]
    fn test_already_resident_pages_not_reloaded() {
        let config = VirtualTextureConfig {
            virtual_size: 512,
            atlas_size: 256,
            page_size: PAGE_SIZE,
            max_mip_levels: 2,
            feedback_capacity: 100,
        };
        let mut manager = VirtualTextureManager::new(config);

        let page = VirtualPage::new(0, 0, 0);

        // First frame: load page
        {
            let entry = FeedbackEntry { page, priority: 1 };
            manager.feedback_buffer().write_buffer()[0] = entry.pack();
            manager.feedback_buffer().set_write_count(1);

            manager.begin_frame();
            let requests = manager.process_feedback();
            assert_eq!(requests.len(), 1);
            manager.page_loaded(&requests[0].page, &requests[0].physical);
            manager.end_frame();
        }

        // Second frame: same page should not be requested again
        {
            let entry = FeedbackEntry { page, priority: 1 };
            manager.feedback_buffer().write_buffer()[0] = entry.pack();
            manager.feedback_buffer().set_write_count(1);

            manager.begin_frame();
            let requests = manager.process_feedback();
            assert_eq!(requests.len(), 0, "already resident page should not generate load request");
            manager.end_frame();
        }
    }
}
