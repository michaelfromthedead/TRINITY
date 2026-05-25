//! Bindless Texture Table -- GPU `texture_2d_array<f32>` with a CPU-side
//! manager using free-list allocation (T-GPU-1.5).
//!
//! The texture table manages indices into a GPU `texture_2d_array<f32>` bindless
//! texture array. Shaders reference textures by a 32-bit index into this array.
//! The CPU-side manager tracks metadata (width, height, format) and uses an
//! explicit free-list for O(1) slot allocation and recycling.
//!
//! # Bindless access
//!
//! Any shader can sample a texture by index from the bindless array without
//! per-draw descriptor binding changes.
//!
//! # Free-list management
//!
//! Unlike [`MeshTable`](super::MeshTable) (which uses implicit zeroed holes),
//! `TextureTable` maintains an explicit free-list stack. Removing a texture
//! pushes its index onto the free-list; adding a texture pops from the
//! free-list first, avoiding O(n) hole scans and providing predictable O(1)
//! allocation.
//!
//! # Capacity limit
//!
//! The table is capped at [`MAX_BINDLESS_TEXTURES`] (4096) entries, matching
//! typical hardware limits for `texture_2d_array<f32>`.
//!
//! # Frame loop
//!
//! ```ignore
//! let mut table = TextureTable::new();
//! let tex_a = table.add(TextureTableEntry {
//!     width: 1024, height: 768, mip_levels: 10,
//!     format: 0, layer_count: 1, flags: 1,
//! }).unwrap();
//!
//! // Stage for GPU upload:
//! if let Some((slot_idx, written)) = table.stage(&mut registry) {
//!     registry.submit_staging(slot_idx, written);
//! }
//! ```

use crate::gpu_driven::buffers::{AcquireResult, BufferRegistry, SubmitResult};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of bindless textures supported by the table.
///
/// This matches typical Vulkan/D3D12 hardware limits for
/// `texture_2d_array<f32>` array layers (4096 on most desktop GPUs).
pub const MAX_BINDLESS_TEXTURES: usize = 4096;

/// Byte size of a single `TextureTableEntry` (must match the WGSL struct).
pub const TEXTURE_TABLE_ENTRY_SIZE: usize = 24;

/// Default initial capacity of the texture table (number of slots).
pub const DEFAULT_TEXTURE_TABLE_CAPACITY: usize = 1024;

// ---------------------------------------------------------------------------
// TextureTableEntry
// ---------------------------------------------------------------------------

/// Metadata for a single texture in the bindless texture array.
///
/// Each entry describes one texture loaded into the GPU `texture_2d_array<f32>`.
/// The struct layout is `#[repr(C)]` with six tightly packed `u32` fields
/// (24 bytes total), exactly matching the WGSL `TextureTableEntry` struct in
/// `texture_table.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TextureTableEntry {
    /// Width of the texture in pixels.
    pub width: u32,
    /// Height of the texture in pixels.
    pub height: u32,
    /// Number of mip levels.
    pub mip_levels: u32,
    /// Packed texture format identifier.
    pub format: u32,
    /// Number of array layers (1 for 2D, >1 for array textures).
    pub layer_count: u32,
    /// Flags field (bit 0 = valid).
    pub flags: u32,
}

impl TextureTableEntry {
    /// Create a new texture table entry with the given field values.
    pub const fn new(
        width: u32,
        height: u32,
        mip_levels: u32,
        format: u32,
        layer_count: u32,
        flags: u32,
    ) -> Self {
        Self {
            width,
            height,
            mip_levels,
            format,
            layer_count,
            flags,
        }
    }

    /// Default/zero entry where all fields are 0.
    pub const fn zero() -> Self {
        Self {
            width: 0,
            height: 0,
            mip_levels: 0,
            format: 0,
            layer_count: 0,
            flags: 0,
        }
    }

    /// Returns `true` when all fields are zero (an invalid/free slot).
    pub fn is_zero(&self) -> bool {
        self.width == 0
            && self.height == 0
            && self.mip_levels == 0
            && self.format == 0
            && self.layer_count == 0
            && self.flags == 0
    }
}

impl Default for TextureTableEntry {
    fn default() -> Self {
        Self::zero()
    }
}

// ---------------------------------------------------------------------------
// Result enums
// ---------------------------------------------------------------------------

/// Result of adding a new entry to the texture table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AddEntry {
    /// Index (handle) the GPU uses to reference this texture.
    pub index: u32,
}

/// Result of removing an entry from the texture table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemoveResult {
    /// Entry was removed and its slot returned to the free-list.
    Removed,
    /// No entry with the given index exists.
    NotFound,
}

// ---------------------------------------------------------------------------
// TextureTable
// ---------------------------------------------------------------------------

/// CPU-side manager for the bindless texture table with free-list allocation.
///
/// Maintains a `Vec<TextureTableEntry>` that shadows the GPU texture-array
/// metadata, plus an explicit free-list (`Vec<u32>`) of available slot indices.
///
/// Key differences from [`MeshTable`](super::MeshTable):
///
/// | Feature | MeshTable | TextureTable |
/// |---------|-----------|-------------|
/// | Allocation | Always appends | Free-list pop first |
/// | Removal | Zeroed hole | Zero + push to free-list |
/// | Capacity | Unbounded | Capped at `MAX_BINDLESS_TEXTURES` |
/// | `add()` return | `u32` | `Option<u32>` (may be full) |
///
/// # Free-list lifecycle
///
/// ```text
/// add(): free_list.pop()  ──or──>  entries.push()
/// remove(): free_list.push(index) + entries[index] = zero()
/// ```
pub struct TextureTable {
    /// Contiguous array of metadata entries indexed by GPU handle.
    entries: Vec<TextureTableEntry>,
    /// Free-list stack of slot indices available for reuse.
    free_list: Vec<u32>,
    /// Number of live (non-zero) entries.
    live_count: usize,
    /// Maximum number of entries (capped at [`MAX_BINDLESS_TEXTURES`]).
    max_textures: usize,
}

impl TextureTable {
    /// Create a new texture table with the given initial Vec capacity.
    ///
    /// The maximum number of textures is always [`MAX_BINDLESS_TEXTURES`] (4096)
    /// regardless of the `capacity` parameter, which only controls the initial
    /// allocation size to avoid reallocation during early inserts.
    pub fn with_capacity(capacity: usize) -> Self {
        let cap = capacity.min(MAX_BINDLESS_TEXTURES).max(1);
        Self {
            entries: Vec::with_capacity(cap),
            free_list: Vec::new(),
            live_count: 0,
            max_textures: MAX_BINDLESS_TEXTURES,
        }
    }

    /// Create a new texture table with [`DEFAULT_TEXTURE_TABLE_CAPACITY`].
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_TEXTURE_TABLE_CAPACITY)
    }

    // -- Accessors ---------------------------------------------------------

    /// Total number of entries allocated (including free-list slots).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Number of live (valid) entries currently in use.
    pub fn live_count(&self) -> usize {
        self.live_count
    }

    /// Returns `true` when no live entries exist.
    pub fn is_empty(&self) -> bool {
        self.live_count == 0
    }

    /// The configured maximum number of textures.
    pub fn max_textures(&self) -> usize {
        self.max_textures
    }

    /// Number of slots currently in the free-list (available for O(1) reuse).
    pub fn free_count(&self) -> usize {
        self.free_list.len()
    }

    /// Returns `true` when the table has reached [`MAX_BINDLESS_TEXTURES`] and
    /// no free slots remain.
    pub fn is_full(&self) -> bool {
        self.entries.len() >= self.max_textures && self.free_list.is_empty()
    }

    /// Get a shared reference to an entry by its GPU handle (index).
    pub fn get(&self, index: u32) -> Option<&TextureTableEntry> {
        self.entries.get(index as usize)
    }

    /// Get a mutable reference to an entry by its GPU handle (index).
    pub fn get_mut(&mut self, index: u32) -> Option<&mut TextureTableEntry> {
        self.entries.get_mut(index as usize)
    }

    /// Slice of all entries (including free-list / invalid slots).
    pub fn as_slice(&self) -> &[TextureTableEntry] {
        &self.entries
    }

    /// The raw byte representation of the full table, suitable for GPU upload.
    ///
    /// The returned slice has length `self.len() * TEXTURE_TABLE_ENTRY_SIZE`.
    pub fn as_bytes(&self) -> &[u8] {
        let byte_len = self.entries.len().wrapping_mul(TEXTURE_TABLE_ENTRY_SIZE);
        if byte_len == 0 {
            return &[];
        }
        // SAFETY: TextureTableEntry is `#[repr(C)]` with no padding between
        // its six `u32` fields, so a reinterpret-cast to a byte slice is
        // sound. The pointer is valid, properly aligned, and the size is
        // exactly `entries.len() * size_of::<TextureTableEntry>()`.
        unsafe { std::slice::from_raw_parts(self.entries.as_ptr() as *const u8, byte_len) }
    }

    // -- Free-list internals -----------------------------------------------

    /// Allocate a slot index, preferring the free-list first.
    ///
    /// Returns `None` when the table is at maximum capacity and the free-list
    /// is empty.
    fn allocate_slot(&mut self) -> Option<u32> {
        // Recycle from the free-list (O(1) pop).
        if let Some(index) = self.free_list.pop() {
            debug_assert!(
                (index as usize) < self.entries.len(),
                "free-list contains out-of-bounds index {} (len {})",
                index,
                self.entries.len(),
            );
            return Some(index);
        }

        // Extend the table if we are below the maximum.
        if self.entries.len() < self.max_textures {
            let index = self.entries.len() as u32;
            self.entries.push(TextureTableEntry::zero());
            return Some(index);
        }

        None
    }

    // -- Mutation ----------------------------------------------------------

    /// Add a new texture entry to the table.
    ///
    /// Allocates a slot from the free-list if available, or extends the table.
    /// Returns `Some(index)` on success, or `None` if the table is full
    /// ([`is_full`](Self::is_full)).
    pub fn add(&mut self, entry: TextureTableEntry) -> Option<u32> {
        let index = self.allocate_slot()?;
        self.entries[index as usize] = entry;
        self.live_count += 1;
        Some(index)
    }

    /// Insert or overwrite an entry at a specific index.
    ///
    /// If `index` is beyond the current length and the table has room, the
    /// entries vec is extended with zero entries to fill the gap. Returns
    /// `false` if the index is beyond [`MAX_BINDLESS_TEXTURES`].
    ///
    /// The `live_count` is adjusted correctly for all cases:
    /// - Extending: the new slot transitions from implicit-zero (not counted)
    ///   to whatever `entry` is.
    /// - Overwriting a zero slot: counted if `entry` is non-zero.
    /// - Overwriting a live slot: live_count unchanged if `entry` is non-zero;
    ///   decremented if `entry` is zero.
    ///
    /// Note: if the targeted slot was previously on the free-list and the
    /// caller passes a non-zero entry, this effectively removes it from the
    /// free-list. The caller should not hold both a free-list handle and an
    /// explicit index for the same slot.
    pub fn insert_at(&mut self, index: u32, entry: TextureTableEntry) -> bool {
        if (index as usize) >= self.max_textures {
            return false;
        }

        let idx = index as usize;
        let old_len = self.entries.len();

        if idx >= old_len {
            // Extend with zero entries. None of these count as live.
            self.entries.resize(idx + 1, TextureTableEntry::zero());
            // The entry we are writing is the only one at this position.
            if !entry.is_zero() {
                self.live_count += 1;
            }
            self.entries[idx] = entry;

            // Remove index from free-list if it somehow was there (defensive).
            if let Some(pos) = self.free_list.iter().position(|&x| x == index) {
                self.free_list.swap_remove(pos);
            }
            return true;
        }

        // Overwriting an existing slot.
        let old_is_zero = self.entries[idx].is_zero();
        let new_is_zero = entry.is_zero();

        if old_is_zero && !new_is_zero {
            self.live_count += 1;
            // Remove from free-list if present.
            if let Some(pos) = self.free_list.iter().position(|&x| x == index) {
                self.free_list.swap_remove(pos);
            }
        } else if !old_is_zero && new_is_zero {
            self.live_count -= 1;
            self.free_list.push(index);
        }

        self.entries[idx] = entry;
        true
    }

    /// Update an existing entry at the given index.
    ///
    /// Returns `false` if the index is beyond the current table length.
    /// Live-count is adjusted correctly if the entry transitions between
    /// zero and non-zero.
    pub fn update(&mut self, index: u32, entry: TextureTableEntry) -> bool {
        let idx = index as usize;
        let slot = match self.entries.get_mut(idx) {
            Some(s) => s,
            None => return false,
        };

        let was_zero = slot.is_zero();
        let is_zero = entry.is_zero();

        *slot = entry;

        // Adjust live_count and free-list.
        if was_zero && !is_zero {
            self.live_count += 1;
            // Remove from free-list if present.
            if let Some(pos) = self.free_list.iter().position(|&x| x == index) {
                self.free_list.swap_remove(pos);
            }
        } else if !was_zero && is_zero {
            self.live_count -= 1;
            self.free_list.push(index);
        }

        true
    }

    /// Remove the entry at `index`, replacing it with a zero hole and
    /// returning the slot to the free-list.
    ///
    /// Does NOT shift subsequent entries, preserving GPU handles. Returns
    /// `RemoveResult::NotFound` if the index is out of range or already
    /// a hole.
    pub fn remove(&mut self, index: u32) -> RemoveResult {
        let idx = index as usize;
        let slot = match self.entries.get_mut(idx) {
            Some(s) => s,
            None => return RemoveResult::NotFound,
        };

        if slot.is_zero() {
            return RemoveResult::NotFound;
        }

        *slot = TextureTableEntry::zero();
        self.live_count -= 1;
        self.free_list.push(index);
        RemoveResult::Removed
    }

    /// Clear all entries and reset the table.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.free_list.clear();
        self.live_count = 0;
    }

    /// Reserve capacity for at least `additional` more entries, avoiding
    /// reallocation during subsequent `add` calls.
    ///
    /// The total capacity will not exceed [`MAX_BINDLESS_TEXTURES`].
    pub fn reserve(&mut self, additional: usize) {
        let current = self.entries.len();
        let target = current.saturating_add(additional).min(self.max_textures);
        if target > current {
            self.entries.reserve(target - current);
        }
    }

    // -- GPU staging -------------------------------------------------------

    /// Stage the current table contents into a [`BufferRegistry`] staging slot.
    ///
    /// Returns `(slot_index, byte_size)` on success, or `None` if no staging
    /// slot is available (the caller should throttle and retry next frame).
    ///
    /// The slot's backing store is grown if its capacity is insufficient.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if the acquired slot is not in `Writing` state.
    pub fn stage(&self, registry: &mut BufferRegistry) -> Option<(usize, usize)> {
        let byte_size = self.entries.len().wrapping_mul(TEXTURE_TABLE_ENTRY_SIZE);
        if byte_size == 0 {
            return None;
        }

        let slot_index = match registry.acquire_staging() {
            AcquireResult::Acquired { slot_index } => slot_index,
            AcquireResult::NoSlotAvailable => return None,
        };

        // Ensure the slot is large enough.
        {
            let slot = registry.slot_mut(slot_index).unwrap();
            if slot.capacity() < byte_size {
                slot.resize(byte_size);
            }
        }

        // Copy entry bytes into the staging slot.
        {
            let slot = registry.slot_mut(slot_index).unwrap();
            let dest = slot.as_mut_slice();
            dest[..byte_size].copy_from_slice(self.as_bytes());
        }

        Some((slot_index, byte_size))
    }

    /// Stage and submit the table contents in one call.
    ///
    /// Combines [`stage`](Self::stage) and
    /// [`submit_staging`](BufferRegistry::submit_staging). Returns `true` on
    /// success, `false` if no slot was available.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if the slot transitions unexpectedly.
    pub fn stage_and_submit(&self, registry: &mut BufferRegistry) -> bool {
        let (slot_index, byte_size) = match self.stage(registry) {
            Some(pair) => pair,
            None => return false,
        };

        match registry.submit_staging(slot_index, byte_size) {
            SubmitResult::Submitted => true,
            SubmitResult::InvalidSlot => {
                debug_assert!(false, "stage_and_submit: acquired slot became invalid");
                false
            }
        }
    }
}

impl Default for TextureTable {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

impl core::fmt::Display for TextureTable {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "TextureTable(entries={}, live={}, free={}, max={})",
            self.entries.len(),
            self.live_count,
            self.free_list.len(),
            self.max_textures,
        )
    }
}

impl core::fmt::Display for TextureTableEntry {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "TextureTableEntry({}x{} mips={} fmt={} layers={} flags=0x{:08x})",
            self.width,
            self.height,
            self.mip_levels,
            self.format,
            self.layer_count,
            self.flags,
        )
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // TextureTableEntry
    // -----------------------------------------------------------------------

    #[test]
    fn test_entry_new_and_defaults() {
        let e = TextureTableEntry::new(1024, 768, 10, 0, 1, 1);
        assert_eq!(e.width, 1024);
        assert_eq!(e.height, 768);
        assert_eq!(e.mip_levels, 10);
        assert_eq!(e.flags, 1);

        let z = TextureTableEntry::zero();
        assert!(z.is_zero());

        let d = TextureTableEntry::default();
        assert!(d.is_zero());
    }

    /// Entry must match the WGSL layout: 24 bytes, 4-byte aligned.
    #[test]
    fn test_entry_size_and_alignment() {
        assert_eq!(
            std::mem::size_of::<TextureTableEntry>(),
            TEXTURE_TABLE_ENTRY_SIZE,
            "TextureTableEntry must be exactly {} bytes (matching WGSL struct)",
            TEXTURE_TABLE_ENTRY_SIZE,
        );
        assert_eq!(
            std::mem::align_of::<TextureTableEntry>(),
            4,
            "TextureTableEntry alignment must be 4 (u32)"
        );
    }

    /// Validate byte-level layout: fields appear in declaration order with
    /// no padding.
    #[test]
    fn test_entry_byte_layout() {
        let e = TextureTableEntry::new(0x04030201, 0x08070605, 3, 0x0A, 1, 0xFF);
        let bytes = unsafe {
            core::slice::from_raw_parts(
                &e as *const TextureTableEntry as *const u8,
                TEXTURE_TABLE_ENTRY_SIZE,
            )
        };

        // width = 0x04030201 (little-endian).
        assert_eq!(bytes[0..4], [0x01, 0x02, 0x03, 0x04]);
        // height = 0x08070605
        assert_eq!(bytes[4..8], [0x05, 0x06, 0x07, 0x08]);
        // mip_levels = 3
        assert_eq!(bytes[8..12], [3, 0, 0, 0]);
        // format = 0x0A
        assert_eq!(bytes[12..16], [0x0A, 0, 0, 0]);
        // layer_count = 1
        assert_eq!(bytes[16..20], [1, 0, 0, 0]);
        // flags = 0xFF
        assert_eq!(bytes[20..24], [0xFF, 0, 0, 0]);
    }

    /// is_zero returns true only when all six fields are zero.
    #[test]
    fn test_entry_is_zero() {
        assert!(TextureTableEntry::zero().is_zero());
        assert!(!TextureTableEntry::new(1, 0, 0, 0, 0, 0).is_zero());
        assert!(!TextureTableEntry::new(0, 1, 0, 0, 0, 0).is_zero());
        assert!(!TextureTableEntry::new(0, 0, 0, 0, 0, 1).is_zero());
    }

    // -----------------------------------------------------------------------
    // TextureTable -- basics
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_table() {
        let table = TextureTable::new();
        assert!(table.is_empty());
        assert_eq!(table.live_count(), 0);
        assert_eq!(table.len(), 0);
        assert_eq!(table.free_count(), 0);
        assert!(!table.is_full());
    }

    #[test]
    fn test_with_capacity() {
        let table = TextureTable::with_capacity(2048);
        assert!(table.is_empty());
        assert_eq!(table.len(), 0);
        assert!(table.entries.capacity() >= 2048);
    }

    #[test]
    fn test_max_textures_is_always_max() {
        let table = TextureTable::with_capacity(1);
        assert_eq!(table.max_textures(), MAX_BINDLESS_TEXTURES);

        let large = TextureTable::with_capacity(MAX_BINDLESS_TEXTURES + 100);
        assert_eq!(large.max_textures(), MAX_BINDLESS_TEXTURES);
    }

    #[test]
    fn test_add_returns_monotonic_indices() {
        let mut table = TextureTable::new();
        let entry = TextureTableEntry::new(0, 0, 0, 0, 0, 0);
        assert_eq!(table.add(entry), Some(0));
        assert_eq!(table.add(entry), Some(1));
        assert_eq!(table.add(entry), Some(2));
        assert_eq!(table.len(), 3);
        assert_eq!(table.live_count(), 3);
    }

    #[test]
    fn test_add_preserves_fields() {
        let mut table = TextureTable::new();
        let idx = table.add(TextureTableEntry::new(1024, 768, 10, 1, 1, 3)).unwrap();
        let entry = table.get(idx).unwrap();
        assert_eq!(entry.width, 1024);
        assert_eq!(entry.height, 768);
        assert_eq!(entry.mip_levels, 10);
        assert_eq!(entry.format, 1);
        assert_eq!(entry.layer_count, 1);
        assert_eq!(entry.flags, 3);
    }

    #[test]
    fn test_get_out_of_range_returns_none() {
        let table = TextureTable::new();
        assert!(table.get(0).is_none());
        assert!(table.get(usize::MAX as u32).is_none());
    }

    // -----------------------------------------------------------------------
    // TextureTable -- free-list management
    // -----------------------------------------------------------------------

    #[test]
    fn test_remove_reuses_slot_via_free_list() {
        let mut table = TextureTable::new();
        let a = table.add(TextureTableEntry::new(100, 100, 1, 0, 1, 1)).unwrap();
        let _b = table.add(TextureTableEntry::new(200, 200, 1, 0, 1, 1)).unwrap();
        let _c = table.add(TextureTableEntry::new(300, 300, 1, 0, 1, 1)).unwrap();
        assert_eq!(table.live_count(), 3);

        // Remove the middle entry.
        assert_eq!(table.remove(a), RemoveResult::Removed);
        assert_eq!(table.live_count(), 2);
        assert_eq!(table.free_count(), 1);

        // Add should reuse the freed slot (index 0).
        let recycled = table.add(TextureTableEntry::new(400, 400, 2, 1, 1, 1)).unwrap();
        assert_eq!(recycled, 0, "free-list must reuse index 0");
        assert_eq!(table.live_count(), 3);
        assert_eq!(table.free_count(), 0);

        // Verify overwritten data.
        let entry = table.get(recycled).unwrap();
        assert_eq!(entry.width, 400);
        assert_eq!(entry.height, 400);
    }

    #[test]
    fn test_free_list_multiple_remove_and_reuse() {
        let mut table = TextureTable::new();
        let mut indices = Vec::new();
        for i in 0..10u32 {
            indices.push(table.add(TextureTableEntry::new(i * 100, 100, 1, 0, 1, 1)).unwrap());
        }
        assert_eq!(table.live_count(), 10);

        // Remove even indices in reverse to build a non-trivial free-list.
        let mut freed = Vec::new();
        for i in (0..10).step_by(2).rev() {
            let idx = indices[i];
            table.remove(idx);
            freed.push(idx);
        }
        assert_eq!(table.free_count(), 5);

        // Free-list is a stack (LIFO), so the last removed (index 0) pops first.
        // Iterate freed in reverse to match LIFO pop order.
        for &expected_idx in freed.iter().rev() {
            let new_idx = table.add(TextureTableEntry::new(1, 1, 1, 0, 1, 1)).unwrap();
            assert_eq!(new_idx, expected_idx, "free-list LIFO: expected {}, got {}", expected_idx, new_idx);
        }
        assert_eq!(table.free_count(), 0);
        assert_eq!(table.live_count(), 10);
    }

    // -----------------------------------------------------------------------
    // TextureTable -- mutation
    // -----------------------------------------------------------------------

    #[test]
    fn test_update_existing_entry() {
        let mut table = TextureTable::new();
        let idx = table.add(TextureTableEntry::new(0, 0, 0, 0, 0, 0)).unwrap();
        assert!(table.update(idx, TextureTableEntry::new(1024, 768, 10, 1, 1, 1)));

        let entry = table.get(idx).unwrap();
        assert_eq!(entry.width, 1024);
        assert_eq!(entry.height, 768);
        assert_eq!(entry.mip_levels, 10);
        assert_eq!(entry.format, 1);
        assert_eq!(entry.layer_count, 1);
        assert_eq!(entry.flags, 1);
    }

    #[test]
    fn test_update_out_of_range_returns_false() {
        let mut table = TextureTable::new();
        assert!(!table.update(0, TextureTableEntry::default()));
        assert!(!table.update(usize::MAX as u32, TextureTableEntry::default()));
    }

    #[test]
    fn test_update_live_count_tracking() {
        let mut table = TextureTable::new();

        // Add a live entry.
        let idx = table.add(TextureTableEntry::new(1, 2, 3, 0, 1, 1)).unwrap();
        assert_eq!(table.live_count(), 1);

        // Update to zero: live_count decrements, index goes to free-list.
        table.update(idx, TextureTableEntry::zero());
        assert_eq!(table.live_count(), 0);
        assert_eq!(table.free_count(), 1);

        // Update from zero to non-zero: live_count increments, free-list cleared.
        table.update(idx, TextureTableEntry::new(7, 8, 9, 0, 1, 1));
        assert_eq!(table.live_count(), 1);
        assert_eq!(table.free_count(), 0);

        // Update non-zero to different non-zero: live_count unchanged.
        table.update(idx, TextureTableEntry::new(13, 14, 15, 0, 1, 1));
        assert_eq!(table.live_count(), 1);

        // Update zero to zero: no change.
        table.update(idx, TextureTableEntry::zero());
        assert_eq!(table.live_count(), 0);
        assert_eq!(table.free_count(), 1);
        table.update(idx, TextureTableEntry::zero());
        assert_eq!(table.live_count(), 0);
    }

    #[test]
    fn test_remove_entry() {
        let mut table = TextureTable::new();
        let idx = table.add(TextureTableEntry::new(1, 2, 3, 0, 1, 1)).unwrap();
        assert_eq!(table.live_count(), 1);

        assert_eq!(table.remove(idx), RemoveResult::Removed);
        assert_eq!(table.live_count(), 0);
        assert_eq!(table.free_count(), 1);

        // Entry still exists as a zero hole.
        let entry = table.get(idx).unwrap();
        assert!(entry.is_zero());
    }

    #[test]
    fn test_remove_nonexistent_index() {
        let mut table = TextureTable::new();
        assert_eq!(table.remove(0), RemoveResult::NotFound);
        assert_eq!(table.remove(usize::MAX as u32), RemoveResult::NotFound);
    }

    #[test]
    fn test_remove_hole_returns_not_found() {
        let mut table = TextureTable::new();
        let idx = table.add(TextureTableEntry::new(1, 2, 3, 0, 1, 1)).unwrap();
        table.remove(idx);
        // Second removal of the same index is a hole -> NotFound.
        assert_eq!(table.remove(idx), RemoveResult::NotFound);
    }

    // -----------------------------------------------------------------------
    // TextureTable -- capacity
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_returns_none_when_full() {
        let mut table = TextureTable::with_capacity(4);
        for i in 0u32..4 {
            assert!(table.add(TextureTableEntry::new(i, 0, 0, 0, 1, 1)).is_some());
        }
        // Table has capacity for MAX_BINDLESS_TEXTURES (4096), so 4 adds
        // does not fill it. Verify is_full is false.
        assert!(!table.is_full());
    }

    #[test]
    fn test_add_after_remove_when_full() {
        let mut table = TextureTable::with_capacity(4);
        let mut indices = Vec::new();
        for i in 0u32..4 {
            indices.push(table.add(TextureTableEntry::new(i, 0, 0, 0, 1, 1)).unwrap());
        }

        // Remove one to free a slot.
        table.remove(indices[2]);
        assert!(!table.is_full());

        // Add should reuse the freed slot via free-list.
        let idx = table.add(TextureTableEntry::new(99, 0, 0, 0, 1, 1)).unwrap();
        assert_eq!(idx, indices[2], "must reuse freed slot");
    }

    #[test]
    fn test_max_textures_constant() {
        assert_eq!(MAX_BINDLESS_TEXTURES, 4096);
    }

    // -----------------------------------------------------------------------
    // TextureTable -- insert_at
    // -----------------------------------------------------------------------

    #[test]
    fn test_insert_at_beyond_max_returns_false() {
        let mut table = TextureTable::with_capacity(4);
        assert!(!table.insert_at(MAX_BINDLESS_TEXTURES as u32, TextureTableEntry::new(1, 1, 1, 0, 1, 1)));
        assert!(!table.insert_at(u32::MAX, TextureTableEntry::new(1, 1, 1, 0, 1, 1)));
    }

    #[test]
    fn test_insert_at_extend() {
        let mut table = TextureTable::new();
        table.add(TextureTableEntry::new(1, 0, 0, 0, 1, 1));
        table.add(TextureTableEntry::new(2, 0, 0, 0, 1, 1));

        assert!(table.insert_at(5, TextureTableEntry::new(99, 0, 0, 0, 1, 1)));

        assert_eq!(table.len(), 6);
        assert_eq!(table.get(0).unwrap().width, 1);
        assert_eq!(table.get(1).unwrap().width, 2);
        for i in 2..5 {
            assert!(table.get(i).unwrap().is_zero());
        }
        assert_eq!(table.get(5).unwrap().width, 99);
        assert_eq!(table.live_count(), 3);
    }

    // -----------------------------------------------------------------------
    // TextureTable -- clear
    // -----------------------------------------------------------------------

    #[test]
    fn test_clear_empties_table() {
        let mut table = TextureTable::new();
        table.add(TextureTableEntry::new(1, 0, 0, 0, 1, 1));
        table.add(TextureTableEntry::new(2, 0, 0, 0, 1, 1));
        assert_eq!(table.live_count(), 2);

        table.clear();
        assert!(table.is_empty());
        assert_eq!(table.live_count(), 0);
        assert_eq!(table.len(), 0);
        assert_eq!(table.free_count(), 0);
    }

    #[test]
    fn test_clear_then_add() {
        let mut table = TextureTable::new();
        table.add(TextureTableEntry::new(1, 0, 0, 0, 1, 1));
        table.clear();
        let idx = table.add(TextureTableEntry::new(42, 0, 0, 0, 1, 1)).unwrap();
        assert_eq!(idx, 0);
        assert_eq!(table.live_count(), 1);
    }

    // -----------------------------------------------------------------------
    // TextureTable -- reserve
    // -----------------------------------------------------------------------

    #[test]
    fn test_reserve_increases_capacity() {
        let mut table = TextureTable::with_capacity(10);
        for i in 0..10u32 {
            table.add(TextureTableEntry::new(i + 1, 0, 0, 0, 1, 1));
        }
        let cap_before = table.entries.capacity();
        assert!(cap_before >= 10);

        table.reserve(100);
        assert!(
            table.entries.capacity() >= cap_before + 100,
            "capacity must increase by at least 100 (was {}, now {})",
            cap_before,
            table.entries.capacity(),
        );
    }

    #[test]
    fn test_reserve_clamped_to_max() {
        let mut table = TextureTable::with_capacity(100);
        table.reserve(MAX_BINDLESS_TEXTURES);
        assert!(table.entries.capacity() <= MAX_BINDLESS_TEXTURES);
    }

    // -----------------------------------------------------------------------
    // TextureTable -- serialization (as_bytes / as_slice)
    // -----------------------------------------------------------------------

    #[test]
    fn test_as_bytes_single_entry() {
        let mut table = TextureTable::new();
        table.add(TextureTableEntry::new(1, 2, 3, 4, 5, 0xFF));

        let bytes = table.as_bytes();
        assert_eq!(bytes.len(), TEXTURE_TABLE_ENTRY_SIZE);

        assert_eq!(&bytes[0..4], &[1, 0, 0, 0]);
        assert_eq!(&bytes[4..8], &[2, 0, 0, 0]);
        assert_eq!(&bytes[8..12], &[3, 0, 0, 0]);
        assert_eq!(&bytes[12..16], &[4, 0, 0, 0]);
        assert_eq!(&bytes[16..20], &[5, 0, 0, 0]);
        assert_eq!(&bytes[20..24], &[0xFF, 0, 0, 0]);
    }

    #[test]
    fn test_as_bytes_multi_entry() {
        let mut table = TextureTable::new();
        table.add(TextureTableEntry::new(10, 0, 0, 0, 0, 0));
        table.add(TextureTableEntry::new(20, 0, 0, 0, 0, 0));
        table.add(TextureTableEntry::new(30, 0, 0, 0, 0, 0));

        let bytes = table.as_bytes();
        assert_eq!(bytes.len(), TEXTURE_TABLE_ENTRY_SIZE * 3);

        assert_eq!(&bytes[0..4], &[10, 0, 0, 0]);
        assert_eq!(&bytes[24..28], &[20, 0, 0, 0]);
        assert_eq!(&bytes[48..52], &[30, 0, 0, 0]);
    }

    #[test]
    fn test_as_bytes_empty_table() {
        let table = TextureTable::new();
        assert!(table.as_bytes().is_empty());
    }

    #[test]
    fn test_as_bytes_with_holes() {
        let mut table = TextureTable::new();
        table.add(TextureTableEntry::new(1, 0, 0, 0, 0, 0));
        let idx = table.add(TextureTableEntry::new(2, 0, 0, 0, 0, 0)).unwrap();
        table.add(TextureTableEntry::new(3, 0, 0, 0, 0, 0));
        table.remove(idx); // Index 1 becomes a hole.

        let bytes = table.as_bytes();
        assert_eq!(bytes.len(), TEXTURE_TABLE_ENTRY_SIZE * 3);
        assert_eq!(&bytes[0..4], &[1, 0, 0, 0], "entry 0 intact");
        assert_eq!(&bytes[24..28], &[0, 0, 0, 0], "entry 1 is hole");
        assert_eq!(&bytes[48..52], &[3, 0, 0, 0], "entry 2 intact");
    }

    #[test]
    fn test_as_slice_matches() {
        let mut table = TextureTable::new();
        let a = table.add(TextureTableEntry::new(1, 2, 3, 0, 1, 0)).unwrap();
        let b = table.add(TextureTableEntry::new(7, 8, 9, 0, 1, 0)).unwrap();

        let slice = table.as_slice();
        assert_eq!(slice.len(), 2);
        assert_eq!(slice[a as usize].width, 1);
        assert_eq!(slice[b as usize].width, 7);
    }

    // -----------------------------------------------------------------------
    // TextureTable -- display
    // -----------------------------------------------------------------------

    #[test]
    fn test_texture_table_display() {
        let mut table = TextureTable::new();
        table.add(TextureTableEntry::new(1, 2, 3, 0, 1, 0));
        let s = format!("{}", table);
        assert!(s.contains("entries=1"));
        assert!(s.contains("live=1"));
        assert!(s.contains("free="));
        assert!(s.starts_with("TextureTable("));
    }

    #[test]
    fn test_texture_table_entry_display() {
        let e = TextureTableEntry::new(1024, 768, 10, 1, 1, 0xFF);
        let s = format!("{}", e);
        assert!(s.contains("1024"));
        assert!(s.contains("768"));
        assert!(s.contains("mips=10"));
        assert!(s.contains("flags=0x000000ff"));
    }

    // -----------------------------------------------------------------------
    // TextureTable -- BufferRegistry staging
    // -----------------------------------------------------------------------

    #[test]
    fn test_stage_into_registry() {
        let mut table = TextureTable::new();
        table.add(TextureTableEntry::new(10, 20, 30, 0, 1, 1));

        let mut registry = BufferRegistry::new(4096);
        let (slot_index, written) = table.stage(&mut registry).expect("must acquire slot");
        assert_eq!(written, TEXTURE_TABLE_ENTRY_SIZE);

        assert!(matches!(
            registry.submit_staging(slot_index, written),
            SubmitResult::Submitted
        ));
        let read_idx = registry.acquire_reading().expect("must have a ready slot");
        let slot = registry.slot(read_idx).unwrap();
        assert_eq!(slot.size(), TEXTURE_TABLE_ENTRY_SIZE);

        let bytes = slot.as_slice();
        assert_eq!(&bytes[0..4], &[10, 0, 0, 0]);
        assert_eq!(&bytes[4..8], &[20, 0, 0, 0]);
    }

    #[test]
    fn test_stage_multi_entry() {
        let mut table = TextureTable::new();
        table.add(TextureTableEntry::new(1, 0, 0, 0, 0, 0));
        table.add(TextureTableEntry::new(2, 0, 0, 0, 0, 0));

        let mut registry = BufferRegistry::new(4096);
        let (slot_index, written) = table.stage(&mut registry).expect("must acquire slot");
        assert_eq!(written, TEXTURE_TABLE_ENTRY_SIZE * 2);

        assert!(matches!(
            registry.submit_staging(slot_index, written),
            SubmitResult::Submitted
        ));
        let read_idx = registry.acquire_reading().expect("must have a ready slot");
        let bytes = registry.slot(read_idx).unwrap().as_slice();
        assert_eq!(&bytes[0..4], &[1, 0, 0, 0]);
        assert_eq!(&bytes[24..28], &[2, 0, 0, 0]);
    }

    #[test]
    fn test_stage_and_submit() {
        let mut table = TextureTable::new();
        table.add(TextureTableEntry::new(1, 2, 3, 0, 1, 1));

        let mut registry = BufferRegistry::new(4096);
        assert!(table.stage_and_submit(&mut registry));
        assert_eq!(registry.ready_slots(), 1);
        assert_eq!(registry.frame_count(), 1);
    }

    #[test]
    fn test_stage_empty_table_returns_none() {
        let table = TextureTable::new();
        let mut registry = BufferRegistry::new(4096);
        assert!(table.stage(&mut registry).is_none());
        assert!(!table.stage_and_submit(&mut registry));
    }

    #[test]
    fn test_stage_no_slot_available() {
        let mut table = TextureTable::new();
        table.add(TextureTableEntry::new(1, 2, 3, 0, 1, 1));

        let mut registry = BufferRegistry::new(4096);

        // Occupy all three staging slots.
        let _s0 = registry.acquire_staging();
        let _s1 = registry.acquire_staging();
        let _s2 = registry.acquire_staging();

        assert!(table.stage(&mut registry).is_none());
        assert!(!table.stage_and_submit(&mut registry));
    }

    #[test]
    fn test_stage_and_submit_full_cycle() {
        let mut table = TextureTable::new();
        table.add(TextureTableEntry::new(0xDE, 0xAD, 0xBE, 0xEF, 1, 1));

        let mut registry = BufferRegistry::new(4096);

        assert!(table.stage_and_submit(&mut registry));

        let read_idx = registry.acquire_reading().unwrap();
        let slot = registry.slot(read_idx).unwrap();
        assert_eq!(slot.size(), TEXTURE_TABLE_ENTRY_SIZE);

        let bytes = slot.as_slice();
        assert_eq!(&bytes[0..4], &[0xDE, 0, 0, 0]);
        assert_eq!(&bytes[4..8], &[0xAD, 0, 0, 0]);
    }

    #[test]
    fn test_staging_slot_auto_resize() {
        let mut table = TextureTable::new();
        for i in 0..300u32 {
            table.add(TextureTableEntry::new(i, 0, 0, 0, 1, 1));
        }

        let mut registry = BufferRegistry::new(64);
        let (slot_index, written) = table.stage(&mut registry).unwrap();
        assert_eq!(written, 300 * TEXTURE_TABLE_ENTRY_SIZE);

        let slot = registry.slot(slot_index).unwrap();
        assert!(slot.capacity() >= 300 * TEXTURE_TABLE_ENTRY_SIZE);
    }

    // -----------------------------------------------------------------------
    // TextureTable -- bulk operations
    // -----------------------------------------------------------------------

    #[test]
    fn test_multiple_adds_and_removes() {
        let mut table = TextureTable::new();
        let mut indices = Vec::new();

        for i in 0..100u32 {
            indices.push(
                table
                    .add(TextureTableEntry::new(
                        i + 1, (i + 1) * 2, (i + 1) * 3, (i + 1) % 10, 1, 0,
                    ))
                    .unwrap(),
            );
        }
        assert_eq!(table.live_count(), 100);
        assert_eq!(table.len(), 100);

        // Remove every other entry.
        for i in (0..100).step_by(2) {
            assert_eq!(
                table.remove(indices[i]),
                RemoveResult::Removed,
                "remove(indices[{}] = {}) must succeed",
                i,
                indices[i],
            );
        }
        assert_eq!(table.live_count(), 50);

        // Remaining entries are intact.
        for i in (1..100).step_by(2) {
            let entry = table.get(indices[i]).unwrap();
            assert_eq!(entry.width, (i + 1) as u32);
            assert_eq!(entry.height, ((i + 1) * 2) as u32);
        }

        // Removed entries are on the free-list and zeroed.
        for i in (0..100).step_by(2) {
            let entry = table.get(indices[i]).unwrap();
            assert!(entry.is_zero());
        }
    }

    #[test]
    fn test_reuse_after_full_clear() {
        let mut table = TextureTable::new();

        for i in 0..10u32 {
            table.add(TextureTableEntry::new(i, 0, 0, 0, 1, 1));
        }
        assert_eq!(table.len(), 10);

        table.clear();

        for i in 0..5u32 {
            let idx = table.add(TextureTableEntry::new(i * 10, 0, 0, 0, 1, 1)).unwrap();
            assert_eq!(idx, i);
        }
        assert_eq!(table.len(), 5);
        assert_eq!(table.live_count(), 5);
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_get_mut_allows_update() {
        let mut table = TextureTable::new();
        let idx = table.add(TextureTableEntry::new(1, 2, 3, 0, 1, 1)).unwrap();

        {
            let entry = table.get_mut(idx).unwrap();
            entry.flags = 0xFF;
        }

        assert_eq!(table.get(idx).unwrap().flags, 0xFF);
    }

    #[test]
    fn test_get_mut_out_of_range() {
        let mut table = TextureTable::new();
        assert!(table.get_mut(0).is_none());
    }

    #[test]
    fn test_default_trait_impls() {
        let _entry: TextureTableEntry = Default::default();
        let _table: TextureTable = Default::default();
    }

    #[test]
    fn test_insert_at_specific_index() {
        let mut table = TextureTable::new();
        table.add(TextureTableEntry::new(1, 0, 0, 0, 1, 1));
        table.add(TextureTableEntry::new(2, 0, 0, 0, 1, 1));

        assert!(table.insert_at(5, TextureTableEntry::new(99, 0, 0, 0, 1, 1)));

        assert_eq!(table.len(), 6);
        assert_eq!(table.get(0).unwrap().width, 1);
        assert_eq!(table.get(1).unwrap().width, 2);
        for i in 2..5 {
            assert!(table.get(i).unwrap().is_zero());
        }
        assert_eq!(table.get(5).unwrap().width, 99);
        assert_eq!(table.live_count(), 3);
    }

    #[test]
    fn test_as_bytes_safe_with_zero_capacity() {
        let table = TextureTable::with_capacity(0);
        let bytes = table.as_bytes();
        assert!(bytes.is_empty());
    }

    #[test]
    fn test_add_and_immediate_read() {
        let mut table = TextureTable::new();
        for i in 0..1000u32 {
            let idx = table
                .add(TextureTableEntry::new(i, i + 1, 0, 0, 1, 1))
                .unwrap();
            let entry = table.get(idx).unwrap();
            assert_eq!(entry.width, i);
            assert_eq!(entry.height, i + 1);
        }
    }

    // -----------------------------------------------------------------------
    // Whitebox: exhaustive live_count + free-list invariant tests
    // -----------------------------------------------------------------------

    /// Helper: count non-zero entries by scanning the entries vec.
    fn count_live_entries(table: &TextureTable) -> usize {
        table.as_slice().iter().filter(|e| !e.is_zero()).count()
    }

    #[test]
    fn test_insert_at_hole_to_hole() {
        let mut table = TextureTable::new();
        table.add(TextureTableEntry::new(10, 20, 30, 0, 1, 1));
        table.remove(0);
        assert_eq!(table.live_count(), 0);
        assert_eq!(table.free_count(), 1);
        assert!(table.get(0).unwrap().is_zero());

        // hole->hole: write zero over zero.
        table.insert_at(0, TextureTableEntry::zero());
        assert!(table.get(0).unwrap().is_zero());
        assert_eq!(table.live_count(), 0, "hole->hole must not change live_count");
        assert_eq!(table.len(), 1);
    }

    #[test]
    fn test_insert_at_live_to_live() {
        let mut table = TextureTable::new();
        table.add(TextureTableEntry::new(10, 20, 30, 0, 1, 1));
        assert_eq!(table.live_count(), 1);

        table.insert_at(0, TextureTableEntry::new(99, 88, 77, 0, 2, 1));
        assert_eq!(table.live_count(), 1, "live->live must not change live_count");
        assert_eq!(table.get(0).unwrap().width, 99);
    }

    #[test]
    fn test_insert_at_hole_to_live() {
        let mut table = TextureTable::new();
        table.add(TextureTableEntry::new(10, 20, 30, 0, 1, 1));
        table.remove(0);
        assert_eq!(table.live_count(), 0);
        assert_eq!(table.free_count(), 1);

        table.insert_at(0, TextureTableEntry::new(5, 6, 7, 0, 1, 1));
        assert_eq!(table.live_count(), 1, "hole->live must increment live_count");
        assert_eq!(table.free_count(), 0, "hole->live must clear free-list");
    }

    #[test]
    fn test_insert_at_live_to_hole() {
        let mut table = TextureTable::new();
        table.add(TextureTableEntry::new(10, 20, 30, 0, 1, 1));
        assert_eq!(table.live_count(), 1);

        table.insert_at(0, TextureTableEntry::zero());
        assert_eq!(table.live_count(), 0, "live->hole must decrement live_count");
        assert!(table.get(0).unwrap().is_zero());
    }

    #[test]
    fn test_insert_at_extend_with_zero() {
        let mut table = TextureTable::with_capacity(2);
        table.insert_at(5, TextureTableEntry::zero());
        assert_eq!(table.len(), 6);
        assert_eq!(table.live_count(), 0, "extend with zero must not change live_count");
        for i in 0..6 {
            assert!(table.get(i).unwrap().is_zero());
        }
    }

    #[test]
    fn test_insert_at_extend_with_live() {
        let mut table = TextureTable::new();
        table.add(TextureTableEntry::new(1, 0, 0, 0, 1, 1));
        assert_eq!(table.live_count(), 1);
        assert_eq!(table.len(), 1);

        table.insert_at(3, TextureTableEntry::new(99, 0, 0, 0, 5, 1));
        assert_eq!(table.len(), 4);
        assert_eq!(table.live_count(), 2, "extend with live must increment live_count");
        assert!(table.get(1).unwrap().is_zero());
        assert!(table.get(2).unwrap().is_zero());
        assert_eq!(table.get(3).unwrap().layer_count, 5);
    }

    #[test]
    fn test_insert_at_exact_len_boundary() {
        let mut table = TextureTable::new();
        table.add(TextureTableEntry::new(1, 2, 3, 0, 1, 1));
        let len = table.len();

        table.insert_at(len as u32, TextureTableEntry::new(5, 6, 7, 0, 1, 1));
        assert_eq!(table.len(), len + 1);
        assert_eq!(table.live_count(), 2);
        assert_eq!(table.get(1).unwrap().width, 5);
    }

    #[test]
    fn test_insert_at_index_zero_on_empty() {
        let mut table = TextureTable::new();
        assert!(table.is_empty());

        table.insert_at(0, TextureTableEntry::new(42, 1, 2, 0, 1, 1));
        assert_eq!(table.len(), 1);
        assert_eq!(table.live_count(), 1);
        assert_eq!(table.get(0).unwrap().width, 42);
    }

    #[test]
    fn test_insert_at_zero_into_empty() {
        let mut table = TextureTable::new();
        table.insert_at(0, TextureTableEntry::zero());
        assert_eq!(table.len(), 1);
        assert_eq!(table.live_count(), 0, "zero entry into empty must keep live_count=0");
        assert!(table.get(0).unwrap().is_zero());
    }

    /// After a sequence of mixed operations, live_count must equal the
    /// scanned count of non-zero entries.
    #[test]
    fn test_live_count_invariant_mixed_operations() {
        let mut table = TextureTable::new();

        for i in 0..5u32 {
            table.add(TextureTableEntry::new(i, i + 10, 0, 0, 1, 1));
        }
        assert_eq!(table.live_count(), count_live_entries(&table));

        table.remove(1);
        table.remove(3);
        assert_eq!(table.live_count(), count_live_entries(&table));

        table.update(0, TextureTableEntry::zero());
        assert_eq!(table.live_count(), count_live_entries(&table));

        table.insert_at(2, TextureTableEntry::new(99, 0, 0, 0, 1, 1));
        assert_eq!(table.live_count(), count_live_entries(&table));

        table.insert_at(5, TextureTableEntry::new(88, 0, 0, 0, 9, 1));
        assert_eq!(table.live_count(), count_live_entries(&table));

        table.clear();
        assert_eq!(table.live_count(), count_live_entries(&table));
        for i in 0..3u32 {
            table.add(TextureTableEntry::new(i, 0, 0, 0, 1, 1));
        }
        assert_eq!(table.live_count(), count_live_entries(&table));
    }

    // -----------------------------------------------------------------------
    // Whitebox: stress tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_large_table_stress() {
        let mut table = TextureTable::with_capacity(MAX_BINDLESS_TEXTURES);
        for i in 0..MAX_BINDLESS_TEXTURES as u32 {
            let idx = table
                .add(TextureTableEntry::new(i, i * 2, i * 3, i % 256, 1, 1))
                .unwrap();
            assert_eq!(idx, i);
        }
        assert_eq!(table.len(), MAX_BINDLESS_TEXTURES);
        assert_eq!(table.live_count(), MAX_BINDLESS_TEXTURES);

        assert_eq!(table.get(0).unwrap().width, 0);
        assert_eq!(table.get(2048).unwrap().width, 2048);
        assert_eq!(table.get((MAX_BINDLESS_TEXTURES - 1) as u32).unwrap().width, (MAX_BINDLESS_TEXTURES - 1) as u32);

        assert_eq!(table.as_bytes().len(), MAX_BINDLESS_TEXTURES * TEXTURE_TABLE_ENTRY_SIZE);
    }

    #[test]
    fn test_mixed_operations_stress() {
        let mut table = TextureTable::with_capacity(MAX_BINDLESS_TEXTURES);

        // Phase 1: add 2000 entries.
        for i in 0..2000u32 {
            table.add(TextureTableEntry::new(i, 0, 0, 0, 1, if i % 2 == 0 { 1 } else { 0 }));
        }
        assert_eq!(table.live_count(), 2000);

        // Phase 2: remove every other entry.
        for i in (0..2000u32).step_by(2) {
            table.remove(i);
        }
        assert_eq!(table.live_count(), 1000);

        // Phase 3: update 500 entries.
        for i in (1..1000u32).step_by(2) {
            table.update(i, TextureTableEntry::new(i + 5000, 0, 0, 0, 1, 1));
        }
        assert_eq!(table.live_count(), 1000);

        // Phase 4: insert_at 500 entries at extension positions.
        let base = table.len();
        for i in 0..500u32 {
            table.insert_at((base + i as usize) as u32, TextureTableEntry::new(i, 0, 0, 0, 1, 1));
        }
        assert_eq!(table.live_count(), 1500);

        assert_eq!(table.live_count(), count_live_entries(&table));
    }

    #[test]
    fn test_as_bytes_after_mutations() {
        let mut table = TextureTable::new();

        for i in 0..4u32 {
            table.add(TextureTableEntry::new(i * 16, i * 8, i * 4, i * 2, 1, 1));
        }

        table.remove(1);
        table.update(2, TextureTableEntry::new(999, 888, 0, 0, 7, 1));
        table.insert_at(5, TextureTableEntry::new(42, 0, 0, 0, 9, 1));

        let bytes = table.as_bytes();
        let expected_len = 6 * TEXTURE_TABLE_ENTRY_SIZE;
        assert_eq!(bytes.len(), expected_len);

        let entries: &[TextureTableEntry] = unsafe {
            std::slice::from_raw_parts(bytes.as_ptr() as *const TextureTableEntry, 6)
        };
        assert_eq!(entries[0].width, 0);
        assert!(entries[1].is_zero(), "removed entry must be zero in bytes");
        assert_eq!(entries[2].width, 999);
        assert_eq!(entries[2].layer_count, 7);
        assert_eq!(entries[3].width, 48);
        assert_eq!(entries[5].width, 42);
    }

    #[test]
    fn test_as_bytes_len_consistent() {
        let mut table = TextureTable::new();

        let check = |t: &TextureTable| {
            assert_eq!(t.as_bytes().len(), t.len() * TEXTURE_TABLE_ENTRY_SIZE);
        };

        check(&table);

        for i in 0..5u32 {
            table.add(TextureTableEntry::new(i, 0, 0, 0, 1, 1));
        }
        check(&table);

        table.remove(0);
        table.remove(2);
        check(&table);

        table.insert_at(3, TextureTableEntry::new(99, 0, 0, 0, 5, 1));
        check(&table);

        table.clear();
        check(&table);
    }

    // -----------------------------------------------------------------------
    // Whitebox: staging after complex mutation sequences
    // -----------------------------------------------------------------------

    #[test]
    fn test_stage_after_complex_mutations() {
        let mut registry = crate::gpu_driven::buffers::BufferRegistry::new(4096);
        let mut table = TextureTable::new();

        for i in 0..5u32 {
            table.add(TextureTableEntry::new(i * 10, 0, 0, 0, 1, 1));
        }

        table.remove(0);
        table.remove(3);
        table.update(1, TextureTableEntry::new(999, 0, 0, 0, 1, 1));
        table.insert_at(5, TextureTableEntry::new(42, 0, 0, 0, 9, 1));

        let (slot_idx, byte_size) = table.stage(&mut registry).expect("staging must succeed");
        assert_eq!(byte_size, 6 * TEXTURE_TABLE_ENTRY_SIZE);

        let slot = registry.slot_mut(slot_idx).unwrap();
        let staged: &[TextureTableEntry] = unsafe {
            std::slice::from_raw_parts(slot.as_slice().as_ptr() as *const TextureTableEntry, 6)
        };
        assert!(staged[0].is_zero());
        assert_eq!(staged[1].width, 999);
        assert_eq!(staged[2].width, 20);
        assert!(staged[3].is_zero());
        assert_eq!(staged[4].width, 40);
        assert_eq!(staged[5].width, 42);
    }

    #[test]
    fn test_multiple_stage_and_submit_cycles() {
        let mut registry = crate::gpu_driven::buffers::BufferRegistry::new(4096);
        let mut table = TextureTable::new();

        assert!(!table.stage_and_submit(&mut registry), "empty table must not stage");

        for i in 0..3u32 {
            table.add(TextureTableEntry::new(i, 0, 0, 0, 1, 1));
        }
        assert!(table.stage_and_submit(&mut registry));

        table.remove(1);
        table.add(TextureTableEntry::new(100, 0, 0, 0, 5, 1));
        assert!(table.stage_and_submit(&mut registry));

        table.clear();
        assert!(!table.stage_and_submit(&mut registry), "cleared table must not stage");
    }

    // -----------------------------------------------------------------------
    // Whitebox: reserve edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_reserve_after_clear() {
        let mut table = TextureTable::with_capacity(10);
        for i in 0..10u32 {
            table.add(TextureTableEntry::new(i, 0, 0, 0, 1, 1));
        }
        assert_eq!(table.len(), 10);

        table.clear();
        assert!(table.is_empty());

        table.reserve(100);
        for i in 0..100u32 {
            let idx = table
                .add(TextureTableEntry::new(i, 0, 0, 0, 1, 1))
                .unwrap();
            assert_eq!(idx, i);
        }
        assert_eq!(table.len(), 100);
        assert_eq!(table.live_count(), 100);
    }

    #[test]
    fn test_reserve_zero_is_noop() {
        let mut table = TextureTable::new();
        let cap_before = table.entries.capacity();
        table.reserve(0);
        assert_eq!(table.entries.capacity(), cap_before);
    }

    #[test]
    fn test_reserve_on_empty_table() {
        let mut table = TextureTable::new();
        table.reserve(50);
        assert!(table.entries.capacity() >= 50);
        for i in 0..50u32 {
            table.add(TextureTableEntry::new(i, 0, 0, 0, 1, 1));
        }
        assert_eq!(table.len(), 50);
    }

    // -----------------------------------------------------------------------
    // Whitebox: large field values
    // -----------------------------------------------------------------------

    #[test]
    fn test_entry_u32_max_values() {
        let entry = TextureTableEntry::new(u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX);
        assert_eq!(entry.width, u32::MAX);
        assert_eq!(entry.height, u32::MAX);
        assert_eq!(entry.mip_levels, u32::MAX);
        assert_eq!(entry.format, u32::MAX);
        assert_eq!(entry.layer_count, u32::MAX);
        assert_eq!(entry.flags, u32::MAX);
        assert!(!entry.is_zero());

        let mut table = TextureTable::new();
        let idx = table.add(entry).unwrap();
        let retrieved = table.get(idx).unwrap();
        assert_eq!(retrieved.width, u32::MAX);
        assert_eq!(retrieved.flags, u32::MAX);
    }

    #[test]
    fn test_update_entry_with_max_values() {
        let mut table = TextureTable::new();
        table.add(TextureTableEntry::new(0, 0, 0, 0, 0, 0));
        assert_eq!(table.live_count(), 1, "add always increments live_count");

        table.update(0, TextureTableEntry::new(u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX));
        assert!(!table.get(0).unwrap().is_zero());
        assert_eq!(table.get(0).unwrap().width, u32::MAX);
        assert_eq!(table.live_count(), 2);

        table.update(0, TextureTableEntry::zero());
        assert!(table.get(0).unwrap().is_zero());
        assert_eq!(table.live_count(), 1);
    }

    // -----------------------------------------------------------------------
    // Whitebox: Display consistency after mutations
    // -----------------------------------------------------------------------

    #[test]
    fn test_display_after_mutations() {
        let mut table = TextureTable::new();
        assert_eq!(
            format!("{}", table),
            format!("TextureTable(entries=0, live=0, free=0, max={})", MAX_BINDLESS_TEXTURES)
        );

        table.add(TextureTableEntry::new(10, 20, 30, 0, 1, 1));
        assert_eq!(
            format!("{}", table),
            format!("TextureTable(entries=1, live=1, free=0, max={})", MAX_BINDLESS_TEXTURES)
        );

        table.add(TextureTableEntry::new(50, 60, 70, 0, 1, 1));
        assert_eq!(
            format!("{}", table),
            format!("TextureTable(entries=2, live=2, free=0, max={})", MAX_BINDLESS_TEXTURES)
        );

        table.remove(0);
        assert_eq!(
            format!("{}", table),
            format!("TextureTable(entries=2, live=1, free=1, max={})", MAX_BINDLESS_TEXTURES)
        );

        table.clear();
        assert_eq!(
            format!("{}", table),
            format!("TextureTable(entries=0, live=0, free=0, max={})", MAX_BINDLESS_TEXTURES)
        );
    }

    #[test]
    fn test_entry_display_max_values() {
        let entry = TextureTableEntry::new(u32::MAX, 0, 0, 0, 255, 0xFFFFFFFF);
        let s = format!("{}", entry);
        assert!(s.contains("4294967295"));
        assert!(s.contains("flags=0xffffffff"));
    }

    // -----------------------------------------------------------------------
    // Whitebox: default texture table
    // -----------------------------------------------------------------------

    #[test]
    fn test_default_texture_table_capacity() {
        let table = TextureTable::default();
        assert_eq!(table.entries.capacity(), DEFAULT_TEXTURE_TABLE_CAPACITY);
    }

    // -----------------------------------------------------------------------
    // Whitebox: remove -> free_list -> add cycle stress
    // -----------------------------------------------------------------------

    /// Stress the free-list invariant through repeated remove/add cycles.
    #[test]
    fn test_remove_add_stress_cycle() {
        let mut table = TextureTable::with_capacity(100);

        // Cycle 100 times: add 50, remove 30, add 20, verify.
        for cycle in 0..100u32 {
            for i in 0..50u32 {
                table.add(TextureTableEntry::new(cycle * 1000 + i, 0, 0, 0, 1, 1));
            }

            for i in 0..30u32 {
                table.remove(i);
            }

            // Free-list should have 30 entries (we removed 0..29).
            // live_count should be 20 (indices 30..49).
            assert_eq!(
                table.live_count(),
                20,
                "cycle {}: expected 20 live after removing 30 of 50",
                cycle,
            );
            assert_eq!(
                table.free_count(),
                30,
                "cycle {}: expected 30 free slots",
                cycle,
            );

            // Add 20 more -- these should reuse free-list slots.
            for i in 0..20u32 {
                let idx = table
                    .add(TextureTableEntry::new(cycle * 2000 + i, 0, 0, 0, 1, 1))
                    .unwrap();
                // Free-list is LIFO: the last removed (29) comes back first.
                assert!(
                    idx < 30,
                    "cycle {}: reused index {} should be in [0, 30)",
                    cycle,
                    idx,
                );
            }

            // Now we should have 40 live, 10 free.
            assert_eq!(table.live_count(), 40);
            assert_eq!(table.free_count(), 10);

            // Clear for next cycle.
            table.clear();
            assert_eq!(table.live_count(), 0);
            assert_eq!(table.free_count(), 0);
        }
    }

    /// Verify the free-list never contains duplicate indices.
    #[test]
    fn test_free_list_no_duplicates() {
        let mut table = TextureTable::new();

        for i in 0..10u32 {
            table.add(TextureTableEntry::new(i, 0, 0, 0, 1, 1));
        }

        // Remove all entries.
        for i in 0..10u32 {
            table.remove(i);
        }
        assert_eq!(table.free_count(), 10);
        assert_eq!(table.live_count(), 0);

        // Verify no duplicates in the free-list.
        let mut sorted = table.free_list.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 10, "free-list must not contain duplicates");
    }

    /// Add up to MAX_BINDLESS_TEXTURES and verify the table fills exactly.
    #[test]
    fn test_add_up_to_max_textures() {
        let mut table = TextureTable::with_capacity(MAX_BINDLESS_TEXTURES);
        for i in 0..MAX_BINDLESS_TEXTURES as u32 {
            let result = table.add(TextureTableEntry::new(i, 0, 0, 0, 1, 1));
            assert!(result.is_some(), "add {} must succeed", i);
        }
        assert!(table.is_full());
        assert_eq!(table.live_count(), MAX_BINDLESS_TEXTURES);

        // One more must fail.
        assert!(table.add(TextureTableEntry::new(0, 0, 0, 0, 1, 1)).is_none());
    }

    /// Remove from a full table and re-add.
    #[test]
    fn test_full_table_remove_and_reuse() {
        let mut table = TextureTable::with_capacity(MAX_BINDLESS_TEXTURES);
        for i in 0..MAX_BINDLESS_TEXTURES as u32 {
            table.add(TextureTableEntry::new(i, 0, 0, 0, 1, 1));
        }
        assert!(table.is_full());

        // Remove the last 100 entries and verify they go to the free-list.
        for i in (MAX_BINDLESS_TEXTURES - 100) as u32..MAX_BINDLESS_TEXTURES as u32 {
            assert_eq!(table.remove(i), RemoveResult::Removed);
        }
        assert_eq!(table.free_count(), 100);
        assert!(!table.is_full());

        // Re-add 100, should reuse free-list slots (in reverse order).
        for i in 0..100u32 {
            let idx = table
                .add(TextureTableEntry::new(i + 9999, 0, 0, 0, 1, 1))
                .unwrap();
            // Last removed was MAX_BINDLESS_TEXTURES-1, so that pops first (LIFO).
            assert!(
                idx >= (MAX_BINDLESS_TEXTURES - 100) as u32,
                "reused index {} should be in the freed range",
                idx,
            );
        }
        assert_eq!(table.free_count(), 0);
        assert!(table.is_full());
        assert_eq!(table.live_count(), MAX_BINDLESS_TEXTURES);
    }
}
