//! Bindless Mesh Table -- GPU buffer (`array<MeshTableEntry>`) with a
//! CPU-side manager for mesh load-time population.
//!
//! The mesh table is the central indirection layer for GPU-driven rendering:
//! shaders reference meshes by a 32-bit index into this table rather than
//! binding individual vertex/index buffers per draw. This enables:
//!
//! - **Bindless access**: any shader can fetch mesh data by index from a
//!   single storage buffer binding.
//! - **Efficient culling**: the GPU can iterate the table and generate
//!   indirect draw commands without CPU round-trips.
//! - **Streaming**: the CPU-side manager can update entries while the GPU
//!   concurrently reads the staging slot (via `BufferRegistry`).
//!
//! # GPU layout
//!
//! The table is laid out as a tightly packed `array<MeshTableEntry>` in a
//! GPU storage buffer. Each entry is 24 bytes (six `u32` fields).
//!
//! # Frame loop
//!
//! ```ignore
//! let mut table = MeshTable::new();
//! let mesh_a = table.add(MeshTableEntry { ... });
//! let mesh_b = table.add(MeshTableEntry { ... });
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

/// Byte size of a single `MeshTableEntry` (must match the WGSL struct).
pub const MESH_TABLE_ENTRY_SIZE: usize = 24;

/// Default initial capacity of the mesh table (number of entries).
pub const DEFAULT_MESH_TABLE_CAPACITY: usize = 1024;

// ---------------------------------------------------------------------------
// MeshTableEntry
// ---------------------------------------------------------------------------

/// A single entry in the bindless mesh table.
///
/// Each entry describes one mesh and is referenced by index from GPU shaders.
/// The struct layout is `#[repr(C)]` with six tightly packed `u32` fields
/// (24 bytes total), exactly matching the WGSL `MeshTableEntry` struct in
/// `mesh_table.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MeshTableEntry {
    /// Byte offset of index data in the GPU index buffer.
    pub index_offset: u32,
    /// Byte offset of vertex data in the GPU vertex buffer.
    pub vertex_offset: u32,
    /// Number of indices to draw.
    pub index_count: u32,
    /// Number of vertices in this mesh.
    pub vertex_count: u32,
    /// Index into the material table.
    pub material_id: u32,
    /// Flags field (bit 0 = visible).
    pub flags: u32,
}

impl MeshTableEntry {
    /// Create a new mesh table entry with the given field values.
    pub const fn new(
        index_offset: u32,
        vertex_offset: u32,
        index_count: u32,
        vertex_count: u32,
        material_id: u32,
        flags: u32,
    ) -> Self {
        Self {
            index_offset,
            vertex_offset,
            index_count,
            vertex_count,
            material_id,
            flags,
        }
    }

    /// Default/zero entry where all fields are 0.
    pub const fn zero() -> Self {
        Self {
            index_offset: 0,
            vertex_offset: 0,
            index_count: 0,
            vertex_count: 0,
            material_id: 0,
            flags: 0,
        }
    }

    /// Returns `true` when all fields are zero (a hole entry).
    pub fn is_zero(&self) -> bool {
        self.index_offset == 0
            && self.vertex_offset == 0
            && self.index_count == 0
            && self.vertex_count == 0
            && self.material_id == 0
            && self.flags == 0
    }
}

impl Default for MeshTableEntry {
    fn default() -> Self {
        Self::zero()
    }
}

// ---------------------------------------------------------------------------
// Result enums
// ---------------------------------------------------------------------------

/// Result of adding a new entry to the mesh table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AddEntry {
    /// Index (handle) the GPU uses to reference this mesh.
    pub index: u32,
}

/// Result of removing an entry from the mesh table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemoveResult {
    /// Entry was removed and replaced with a hole.
    Removed,
    /// No entry with the given index exists.
    NotFound,
}

// ---------------------------------------------------------------------------
// MeshTable
// ---------------------------------------------------------------------------

/// CPU-side manager for the bindless mesh table.
///
/// Maintains a `Vec<MeshTableEntry>` that can be:
/// - Populated at mesh load time (`add`, `update`, `remove`)
/// - Serialized to raw bytes for GPU upload (`as_bytes`)
/// - Staged through `BufferRegistry` for triple-buffered transfer (`stage`)
///
/// The table supports **holes**: removing an entry replaces it with a zeroed
/// entry but does not shift subsequent entries, preserving existing GPU
/// handles. Zero entries are skipped by `live_count`.
pub struct MeshTable {
    /// Contiguous array of entries indexed by GPU handle.
    entries: Vec<MeshTableEntry>,
    /// Number of live (non-zero) entries.
    live_count: usize,
}

impl MeshTable {
    /// Create a new empty mesh table with the given initial capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
            live_count: 0,
        }
    }

    /// Create a new empty mesh table with [`DEFAULT_MESH_TABLE_CAPACITY`].
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_MESH_TABLE_CAPACITY)
    }

    // -- Accessors ---------------------------------------------------------

    /// Total number of entries allocated (including holes).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Number of live (non-hole) entries.
    pub fn live_count(&self) -> usize {
        self.live_count
    }

    /// Returns `true` when no live entries exist.
    pub fn is_empty(&self) -> bool {
        self.live_count == 0
    }

    /// Get a shared reference to an entry by its GPU handle (index).
    pub fn get(&self, index: u32) -> Option<&MeshTableEntry> {
        self.entries.get(index as usize)
    }

    /// Get a mutable reference to an entry by its GPU handle (index).
    pub fn get_mut(&mut self, index: u32) -> Option<&mut MeshTableEntry> {
        self.entries.get_mut(index as usize)
    }

    /// Slice of all entries (including holes).
    pub fn as_slice(&self) -> &[MeshTableEntry] {
        &self.entries
    }

    /// The raw byte representation of the full table, suitable for GPU upload.
    ///
    /// The returned slice has length `self.len() * MESH_TABLE_ENTRY_SIZE`.
    pub fn as_bytes(&self) -> &[u8] {
        let byte_len = self.entries.len().wrapping_mul(MESH_TABLE_ENTRY_SIZE);
        if byte_len == 0 {
            return &[];
        }
        // SAFETY: MeshTableEntry is `#[repr(C)]` with no padding between
        // its six `u32` fields, so a reinterpret-cast to a byte slice is
        // sound. The pointer is valid, properly aligned, and the size is
        // exactly `entries.len() * size_of::<MeshTableEntry>()`.
        unsafe { std::slice::from_raw_parts(self.entries.as_ptr() as *const u8, byte_len) }
    }

    // -- Mutation ----------------------------------------------------------

    /// Append a new entry to the end of the table.
    ///
    /// Returns the index (GPU handle) assigned to this entry.
    pub fn add(&mut self, entry: MeshTableEntry) -> u32 {
        let index = self.entries.len() as u32;
        self.entries.push(entry);
        self.live_count += 1;
        index
    }

    /// Insert or overwrite an entry at a specific index.
    ///
    /// If `index` is beyond the current length, the table is extended with
    /// zero entries to fill the gap. Prefer `add` for sequential population
    /// to avoid wasteful zero-fill.
    ///
    /// The `live_count` is adjusted correctly for all cases:
    /// - Extending the table: the new slot transitions from implicit-zero
    ///   (not counted) to whatever `entry` is.
    /// - Overwriting a hole (explicit zero): counted if `entry` is non-zero.
    /// - Overwriting a live entry: counted if `entry` is non-zero;
    ///   decremented if `entry` is zero.
    pub fn insert_at(&mut self, index: u32, entry: MeshTableEntry) {
        let idx = index as usize;
        let old_len = self.entries.len();

        if idx >= old_len {
            // Extend the table with zero entries. None of these count as live.
            self.entries.resize(idx + 1, MeshTableEntry::zero());
            // The entry we are about to write is the only one at this position.
            // If it is non-zero, count it.
            if !entry.is_zero() {
                self.live_count += 1;
            }
            self.entries[idx] = entry;
            return;
        }

        // Overwriting an existing slot.
        let old_is_zero = self.entries[idx].is_zero();
        let new_is_zero = entry.is_zero();

        if old_is_zero && !new_is_zero {
            self.live_count += 1;
        } else if !old_is_zero && new_is_zero {
            self.live_count -= 1;
        }

        self.entries[idx] = entry;
    }

    /// Update an existing entry at the given index.
    ///
    /// Returns `false` if the index is beyond the current table length.
    pub fn update(&mut self, index: u32, entry: MeshTableEntry) -> bool {
        let idx = index as usize;
        let slot = match self.entries.get_mut(idx) {
            Some(s) => s,
            None => return false,
        };

        let was_zero = slot.is_zero();
        let is_zero = entry.is_zero();

        *slot = entry;

        // Adjust live_count.
        if was_zero && !is_zero {
            self.live_count += 1;
        } else if !was_zero && is_zero {
            self.live_count -= 1;
        }
        // otherwise: (zero->zero) or (non-zero->non-zero): no change.

        true
    }

    /// Remove the entry at `index`, replacing it with a zero hole.
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

        *slot = MeshTableEntry::zero();
        self.live_count -= 1;
        RemoveResult::Removed
    }

    /// Clear all entries and reset the table.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.live_count = 0;
    }

    /// Reserve capacity for at least `additional` more entries, avoiding
    /// reallocation during subsequent `add` calls.
    pub fn reserve(&mut self, additional: usize) {
        self.entries.reserve(additional);
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
        let byte_size = self.entries.len().wrapping_mul(MESH_TABLE_ENTRY_SIZE);
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

impl Default for MeshTable {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

impl core::fmt::Display for MeshTable {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "MeshTable(entries={}, live={})",
            self.entries.len(),
            self.live_count,
        )
    }
}

impl core::fmt::Display for MeshTableEntry {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "MeshTableEntry(idx_off={}, vtx_off={}, idx_cnt={}, vtx_cnt={}, \
             mat={}, flags=0x{:08x})",
            self.index_offset,
            self.vertex_offset,
            self.index_count,
            self.vertex_count,
            self.material_id,
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
    // MeshTableEntry
    // -----------------------------------------------------------------------

    #[test]
    fn test_entry_new_and_defaults() {
        let e = MeshTableEntry::new(0, 0, 0, 0, 0, 0);
        assert_eq!(e.index_offset, 0);
        assert_eq!(e.flags, 0);

        let z = MeshTableEntry::zero();
        assert!(z.is_zero());

        let d = MeshTableEntry::default();
        assert!(d.is_zero());
    }

    /// Entry must match the WGSL layout: 24 bytes, 4-byte aligned.
    #[test]
    fn test_entry_size_and_alignment() {
        assert_eq!(
            std::mem::size_of::<MeshTableEntry>(),
            MESH_TABLE_ENTRY_SIZE,
            "MeshTableEntry must be exactly {} bytes (matching WGSL struct)",
            MESH_TABLE_ENTRY_SIZE,
        );
        assert_eq!(
            std::mem::align_of::<MeshTableEntry>(),
            4,
            "MeshTableEntry alignment must be 4 (u32)"
        );
    }

    /// Validate byte-level layout: fields appear in declaration order with
    /// no padding.
    #[test]
    fn test_entry_byte_layout() {
        let e = MeshTableEntry::new(0x01020304, 0x05060708, 3, 4, 5, 0xFF);
        let bytes = unsafe {
            core::slice::from_raw_parts(
                &e as *const MeshTableEntry as *const u8,
                MESH_TABLE_ENTRY_SIZE,
            )
        };

        // index_offset = 0x01020304 (little-endian).
        assert_eq!(bytes[0..4], [0x04, 0x03, 0x02, 0x01]);
        // vertex_offset = 0x05060708
        assert_eq!(bytes[4..8], [0x08, 0x07, 0x06, 0x05]);
        // index_count = 3
        assert_eq!(bytes[8..12], [3, 0, 0, 0]);
        // vertex_count = 4
        assert_eq!(bytes[12..16], [4, 0, 0, 0]);
        // material_id = 5
        assert_eq!(bytes[16..20], [5, 0, 0, 0]);
        // flags = 0xFF
        assert_eq!(bytes[20..24], [0xFF, 0, 0, 0]);
    }

    /// is_zero returns true only when all six fields are zero.
    #[test]
    fn test_entry_is_zero() {
        assert!(MeshTableEntry::zero().is_zero());
        assert!(!MeshTableEntry::new(1, 0, 0, 0, 0, 0).is_zero());
        assert!(!MeshTableEntry::new(0, 1, 0, 0, 0, 0).is_zero());
        assert!(!MeshTableEntry::new(0, 0, 0, 0, 0, 1).is_zero());
    }

    // -----------------------------------------------------------------------
    // MeshTable -- basics
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_table() {
        let table = MeshTable::new();
        assert!(table.is_empty());
        assert_eq!(table.live_count(), 0);
        assert_eq!(table.len(), 0);
    }

    #[test]
    fn test_with_capacity() {
        let table = MeshTable::with_capacity(2048);
        assert!(table.is_empty());
        assert_eq!(table.len(), 0);
        // Capacity is implementation-defined but must be >= 2048.
        assert!(table.entries.capacity() >= 2048);
    }

    #[test]
    fn test_add_returns_monotonic_indices() {
        let mut table = MeshTable::new();
        assert_eq!(table.add(MeshTableEntry::new(0, 0, 0, 0, 0, 0)), 0);
        assert_eq!(table.add(MeshTableEntry::new(0, 0, 0, 0, 0, 0)), 1);
        assert_eq!(table.add(MeshTableEntry::new(0, 0, 0, 0, 0, 0)), 2);
        assert_eq!(table.len(), 3);
        assert_eq!(table.live_count(), 3);
    }

    #[test]
    fn test_add_preserves_fields() {
        let mut table = MeshTable::new();
        let idx = table.add(MeshTableEntry::new(10, 20, 30, 40, 2, 3));
        let entry = table.get(idx).unwrap();
        assert_eq!(entry.index_offset, 10);
        assert_eq!(entry.vertex_offset, 20);
        assert_eq!(entry.index_count, 30);
        assert_eq!(entry.vertex_count, 40);
        assert_eq!(entry.material_id, 2);
        assert_eq!(entry.flags, 3);
    }

    #[test]
    fn test_get_out_of_range_returns_none() {
        let table = MeshTable::new();
        assert!(table.get(0).is_none());
        assert!(table.get(usize::MAX as u32).is_none());
    }

    // -----------------------------------------------------------------------
    // MeshTable -- mutation
    // -----------------------------------------------------------------------

    #[test]
    fn test_update_existing_entry() {
        let mut table = MeshTable::new();
        let idx = table.add(MeshTableEntry::new(0, 0, 0, 0, 0, 0));
        assert!(table.update(idx, MeshTableEntry::new(1, 2, 3, 4, 5, 6)));

        let entry = table.get(idx).unwrap();
        assert_eq!(entry.index_offset, 1);
        assert_eq!(entry.vertex_offset, 2);
        assert_eq!(entry.index_count, 3);
        assert_eq!(entry.vertex_count, 4);
        assert_eq!(entry.material_id, 5);
        assert_eq!(entry.flags, 6);
    }

    #[test]
    fn test_update_out_of_range_returns_false() {
        let mut table = MeshTable::new();
        assert!(!table.update(0, MeshTableEntry::default()));
        assert!(!table.update(usize::MAX as u32, MeshTableEntry::default()));
    }

    #[test]
    fn test_update_live_count_tracking() {
        let mut table = MeshTable::new();

        // Add a live entry.
        let idx = table.add(MeshTableEntry::new(1, 2, 3, 4, 5, 6));
        assert_eq!(table.live_count(), 1);

        // Update to zero: live_count decrements.
        table.update(idx, MeshTableEntry::zero());
        assert_eq!(table.live_count(), 0);

        // Update from zero to non-zero: live_count increments.
        table.update(idx, MeshTableEntry::new(7, 8, 9, 10, 11, 12));
        assert_eq!(table.live_count(), 1);

        // Update non-zero to different non-zero: live_count unchanged.
        table.update(idx, MeshTableEntry::new(13, 14, 15, 16, 17, 18));
        assert_eq!(table.live_count(), 1);

        // Update zero to zero: no change.
        table.update(idx, MeshTableEntry::zero());
        assert_eq!(table.live_count(), 0);
        table.update(idx, MeshTableEntry::zero());
        assert_eq!(table.live_count(), 0);
    }

    #[test]
    fn test_remove_entry() {
        let mut table = MeshTable::new();
        let idx = table.add(MeshTableEntry::new(1, 2, 3, 4, 5, 6));
        assert_eq!(table.live_count(), 1);

        assert_eq!(table.remove(idx), RemoveResult::Removed);
        assert_eq!(table.live_count(), 0);

        // Entry still exists as a zero hole.
        let entry = table.get(idx).unwrap();
        assert!(entry.is_zero());
    }

    #[test]
    fn test_remove_nonexistent_index() {
        let mut table = MeshTable::new();
        assert_eq!(table.remove(0), RemoveResult::NotFound);
        assert_eq!(table.remove(usize::MAX as u32), RemoveResult::NotFound);
    }

    #[test]
    fn test_remove_hole_returns_not_found() {
        let mut table = MeshTable::new();
        let idx = table.add(MeshTableEntry::new(1, 2, 3, 4, 5, 6));
        table.remove(idx);
        // Second removal of the same index is a hole -> NotFound.
        assert_eq!(table.remove(idx), RemoveResult::NotFound);
    }

    // -----------------------------------------------------------------------
    // MeshTable -- clear
    // -----------------------------------------------------------------------

    #[test]
    fn test_clear_empties_table() {
        let mut table = MeshTable::new();
        table.add(MeshTableEntry::new(1, 0, 0, 0, 0, 0));
        table.add(MeshTableEntry::new(2, 0, 0, 0, 0, 0));
        assert_eq!(table.live_count(), 2);

        table.clear();
        assert!(table.is_empty());
        assert_eq!(table.live_count(), 0);
        assert_eq!(table.len(), 0);
    }

    #[test]
    fn test_clear_then_add() {
        let mut table = MeshTable::new();
        table.add(MeshTableEntry::new(1, 0, 0, 0, 0, 0));
        table.clear();
        let idx = table.add(MeshTableEntry::new(42, 0, 0, 0, 0, 0));
        assert_eq!(idx, 0);
        assert_eq!(table.live_count(), 1);
    }

    // -----------------------------------------------------------------------
    // MeshTable -- reserve
    // -----------------------------------------------------------------------

    #[test]
    fn test_reserve_increases_capacity() {
        let mut table = MeshTable::with_capacity(10);
        // Fill to capacity so `reserve` must grow.
        for i in 0..10u32 {
            table.add(MeshTableEntry::new(i + 1, 0, 0, 0, 0, 0));
        }
        let cap_before = table.entries.capacity();
        assert!(
            cap_before >= 10,
            "initial capacity must be >= 10"
        );

        table.reserve(100);
        assert!(
            table.entries.capacity() >= cap_before + 100,
            "capacity must increase by at least 100 (was {}, now {})",
            cap_before,
            table.entries.capacity(),
        );
    }

    // -----------------------------------------------------------------------
    // MeshTable -- serialization (as_bytes / as_slice)
    // -----------------------------------------------------------------------

    #[test]
    fn test_as_bytes_single_entry() {
        let mut table = MeshTable::new();
        table.add(MeshTableEntry::new(1, 2, 3, 4, 5, 0xFF));

        let bytes = table.as_bytes();
        assert_eq!(bytes.len(), MESH_TABLE_ENTRY_SIZE);

        assert_eq!(&bytes[0..4], &[1, 0, 0, 0]);
        assert_eq!(&bytes[4..8], &[2, 0, 0, 0]);
        assert_eq!(&bytes[8..12], &[3, 0, 0, 0]);
        assert_eq!(&bytes[12..16], &[4, 0, 0, 0]);
        assert_eq!(&bytes[16..20], &[5, 0, 0, 0]);
        assert_eq!(&bytes[20..24], &[0xFF, 0, 0, 0]);
    }

    #[test]
    fn test_as_bytes_multi_entry() {
        let mut table = MeshTable::new();
        table.add(MeshTableEntry::new(10, 0, 0, 0, 0, 0));
        table.add(MeshTableEntry::new(20, 0, 0, 0, 0, 0));
        table.add(MeshTableEntry::new(30, 0, 0, 0, 0, 0));

        let bytes = table.as_bytes();
        assert_eq!(bytes.len(), MESH_TABLE_ENTRY_SIZE * 3);

        // Entry 0: index_offset = 10
        assert_eq!(&bytes[0..4], &[10, 0, 0, 0]);
        // Entry 1: index_offset = 20 (at offset 24)
        assert_eq!(&bytes[24..28], &[20, 0, 0, 0]);
        // Entry 2: index_offset = 30 (at offset 48)
        assert_eq!(&bytes[48..52], &[30, 0, 0, 0]);
    }

    #[test]
    fn test_as_bytes_empty_table() {
        let table = MeshTable::new();
        assert!(table.as_bytes().is_empty());
    }

    #[test]
    fn test_as_bytes_with_holes() {
        let mut table = MeshTable::new();
        table.add(MeshTableEntry::new(1, 0, 0, 0, 0, 0));
        let idx = table.add(MeshTableEntry::new(2, 0, 0, 0, 0, 0));
        table.add(MeshTableEntry::new(3, 0, 0, 0, 0, 0));
        table.remove(idx); // Index 1 becomes a hole.

        let bytes = table.as_bytes();
        assert_eq!(bytes.len(), MESH_TABLE_ENTRY_SIZE * 3);
        assert_eq!(&bytes[0..4], &[1, 0, 0, 0], "entry 0 intact");
        // Entry 1 is a hole (all zeros).
        assert_eq!(&bytes[24..28], &[0, 0, 0, 0], "entry 1 is hole");
        assert_eq!(&bytes[48..52], &[3, 0, 0, 0], "entry 2 intact");
    }

    #[test]
    fn test_as_slice_matches() {
        let mut table = MeshTable::new();
        let a = table.add(MeshTableEntry::new(1, 2, 3, 4, 5, 6));
        let b = table.add(MeshTableEntry::new(7, 8, 9, 10, 11, 12));

        let slice = table.as_slice();
        assert_eq!(slice.len(), 2);
        assert_eq!(slice[a as usize].index_offset, 1);
        assert_eq!(slice[b as usize].index_offset, 7);
    }

    // -----------------------------------------------------------------------
    // MeshTable -- display
    // -----------------------------------------------------------------------

    #[test]
    fn test_mesh_table_display() {
        let mut table = MeshTable::new();
        table.add(MeshTableEntry::new(1, 2, 3, 4, 5, 0));
        let s = format!("{}", table);
        assert!(s.contains("entries=1"));
        assert!(s.contains("live=1"));
        assert!(s.starts_with("MeshTable("));
    }

    #[test]
    fn test_mesh_table_entry_display() {
        let e = MeshTableEntry::new(1, 2, 3, 4, 5, 0xFF);
        let s = format!("{}", e);
        assert!(s.contains("idx_off=1"));
        assert!(s.contains("vtx_off=2"));
        assert!(s.contains("idx_cnt=3"));
        assert!(s.contains("vtx_cnt=4"));
        assert!(s.contains("mat=5"));
        assert!(s.contains("flags=0x000000ff"));
    }

    // -----------------------------------------------------------------------
    // MeshTable -- BufferRegistry staging
    // -----------------------------------------------------------------------

    #[test]
    fn test_stage_into_registry() {
        let mut table = MeshTable::new();
        table.add(MeshTableEntry::new(10, 20, 30, 40, 0, 1));

        let mut registry = BufferRegistry::new(4096);
        let (slot_index, written) = table.stage(&mut registry).expect("must acquire slot");
        assert_eq!(written, MESH_TABLE_ENTRY_SIZE);

        // Submit and read back to verify data integrity through the pipeline.
        assert!(matches!(
            registry.submit_staging(slot_index, written),
            SubmitResult::Submitted
        ));
        let read_idx = registry.acquire_reading().expect("must have a ready slot");
        let slot = registry.slot(read_idx).unwrap();
        assert_eq!(slot.size(), MESH_TABLE_ENTRY_SIZE);

        let bytes = slot.as_slice();
        assert_eq!(&bytes[0..4], &[10, 0, 0, 0]);
        assert_eq!(&bytes[4..8], &[20, 0, 0, 0]);
    }

    #[test]
    fn test_stage_multi_entry() {
        let mut table = MeshTable::new();
        table.add(MeshTableEntry::new(1, 0, 0, 0, 0, 0));
        table.add(MeshTableEntry::new(2, 0, 0, 0, 0, 0));

        let mut registry = BufferRegistry::new(4096);
        let (slot_index, written) = table.stage(&mut registry).expect("must acquire slot");
        assert_eq!(written, MESH_TABLE_ENTRY_SIZE * 2);

        // Submit and read back.
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
        let mut table = MeshTable::new();
        table.add(MeshTableEntry::new(1, 2, 3, 4, 5, 6));

        let mut registry = BufferRegistry::new(4096);
        assert!(table.stage_and_submit(&mut registry));
        assert_eq!(registry.ready_slots(), 1);
        assert_eq!(registry.frame_count(), 1);
    }

    #[test]
    fn test_stage_empty_table_returns_none() {
        let table = MeshTable::new();
        let mut registry = BufferRegistry::new(4096);
        assert!(table.stage(&mut registry).is_none());
        assert!(!table.stage_and_submit(&mut registry));
    }

    #[test]
    fn test_stage_no_slot_available() {
        let mut table = MeshTable::new();
        table.add(MeshTableEntry::new(1, 2, 3, 4, 5, 6));

        let mut registry = BufferRegistry::new(4096);

        // Occupy all three staging slots.
        let _s0 = registry.acquire_staging();
        let _s1 = registry.acquire_staging();
        let _s2 = registry.acquire_staging();

        // All slots occupied -- stage must return None.
        assert!(table.stage(&mut registry).is_none());
        assert!(!table.stage_and_submit(&mut registry));
    }

    #[test]
    fn test_stage_and_submit_full_cycle() {
        let mut table = MeshTable::new();
        table.add(MeshTableEntry::new(0xDE, 0xAD, 0xBE, 0xEF, 0, 1));

        let mut registry = BufferRegistry::new(4096);

        // Stage + submit.
        assert!(table.stage_and_submit(&mut registry));

        // GPU reads the submitted data.
        let read_idx = registry.acquire_reading().unwrap();
        let slot = registry.slot(read_idx).unwrap();
        assert_eq!(slot.size(), MESH_TABLE_ENTRY_SIZE);

        let bytes = slot.as_slice();
        assert_eq!(&bytes[0..4], &[0xDE, 0, 0, 0]);
        assert_eq!(&bytes[4..8], &[0xAD, 0, 0, 0]);
    }

    #[test]
    fn test_staging_slot_auto_resize() {
        let mut table = MeshTable::new();
        // Add enough entries to exceed a small staging slot.
        for i in 0..300u32 {
            table.add(MeshTableEntry::new(i, 0, 0, 0, 0, 0));
        }

        let mut registry = BufferRegistry::new(64); // Small initial capacity.
        let (slot_index, written) = table.stage(&mut registry).unwrap();
        assert_eq!(written, 300 * MESH_TABLE_ENTRY_SIZE);

        let slot = registry.slot(slot_index).unwrap();
        assert!(slot.capacity() >= 300 * MESH_TABLE_ENTRY_SIZE);
    }

    // -----------------------------------------------------------------------
    // MeshTable -- bulk operations
    // -----------------------------------------------------------------------

    #[test]
    fn test_multiple_adds_and_removes() {
        let mut table = MeshTable::new();
        let mut indices = Vec::new();

        for i in 0..100u32 {
            indices.push(table.add(MeshTableEntry::new(i + 1, (i + 1) * 2, (i + 1) * 3, (i + 1) * 4, (i + 1) % 10, 0)));
        }
        assert_eq!(table.live_count(), 100);
        assert_eq!(table.len(), 100);

        // Remove every other entry (even indices in the vector).
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

        // Remaining entries are intact (odd indices in the vector).
        for i in (1..100).step_by(2) {
            let entry = table.get(indices[i]).unwrap();
            assert_eq!(entry.index_offset, (i + 1) as u32);
            assert_eq!(entry.vertex_offset, ((i + 1) * 2) as u32);
            assert_eq!(entry.material_id, ((i + 1) % 10) as u32);
        }

        // Removed entries are holes (even indices in the vector).
        for i in (0..100).step_by(2) {
            let entry = table.get(indices[i]).unwrap();
            assert!(entry.is_zero());
        }
    }

    #[test]
    fn test_reuse_after_full_clear() {
        let mut table = MeshTable::new();

        // First batch.
        for i in 0..10u32 {
            table.add(MeshTableEntry::new(i, 0, 0, 0, 0, 0));
        }
        assert_eq!(table.len(), 10);

        table.clear();

        // Second batch -- indices restart.
        for i in 0..5u32 {
            let idx = table.add(MeshTableEntry::new(i * 10, 0, 0, 0, 0, 0));
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
        let mut table = MeshTable::new();
        let idx = table.add(MeshTableEntry::new(1, 2, 3, 4, 5, 6));

        {
            let entry = table.get_mut(idx).unwrap();
            entry.flags = 0xFF;
        }

        assert_eq!(table.get(idx).unwrap().flags, 0xFF);
    }

    #[test]
    fn test_get_mut_out_of_range() {
        let mut table = MeshTable::new();
        assert!(table.get_mut(0).is_none());
    }

    #[test]
    fn test_default_trait_impls() {
        let _entry: MeshTableEntry = Default::default();
        let _table: MeshTable = Default::default();
    }

    #[test]
    fn test_insert_at() {
        let mut table = MeshTable::new();
        table.add(MeshTableEntry::new(1, 0, 0, 0, 0, 0));
        table.add(MeshTableEntry::new(2, 0, 0, 0, 0, 0));

        // Insert at a specific index.
        table.insert_at(5, MeshTableEntry::new(99, 0, 0, 0, 0, 0));

        // Table should have 6 entries (indices 0..5).
        assert_eq!(table.len(), 6);
        assert_eq!(table.get(0).unwrap().index_offset, 1);
        assert_eq!(table.get(1).unwrap().index_offset, 2);
        // Indices 2..4 are zeros (from resize).
        for i in 2..5 {
            assert!(table.get(i).unwrap().is_zero());
        }
        assert_eq!(table.get(5).unwrap().index_offset, 99);

        // live_count: entry 0, 1, and 5 are live.
        assert_eq!(table.live_count(), 3);
    }

    /// When the table has 0 entries, as_bytes must return an empty slice
    /// without panicking or reading from a null pointer.
    #[test]
    fn test_as_bytes_safe_with_zero_capacity() {
        let table = MeshTable::with_capacity(0);
        let bytes = table.as_bytes();
        assert!(bytes.is_empty());
    }

    /// add() followed by immediate get() of the returned index must succeed.
    #[test]
    fn test_add_and_immediate_read() {
        let mut table = MeshTable::new();
        for i in 0..1000u32 {
            let idx = table.add(MeshTableEntry::new(i, i + 1, 0, 0, 0, 0));
            let entry = table.get(idx).unwrap();
            assert_eq!(entry.index_offset, i);
            assert_eq!(entry.vertex_offset, i + 1);
        }
    }

// =========================================================================
// Whitebox: insert_at -- exhaustive live_count transitions
// =========================================================================

/// insert_at: hole->hole -- overwriting a zero entry with another zero
/// entry must NOT change live_count.
#[test]
fn test_insert_at_hole_to_hole() {
    let mut table = MeshTable::new();
    // Add a live entry, then remove it -> hole at index 0.
    table.add(MeshTableEntry::new(10, 20, 30, 40, 0, 1));
    table.remove(0);
    assert_eq!(table.live_count(), 0);
    assert!(table.get(0).unwrap().is_zero());

    // hole->hole: write zero over zero.
    table.insert_at(0, MeshTableEntry::zero());
    assert!(table.get(0).unwrap().is_zero());
    assert_eq!(table.live_count(), 0, "hole->hole must not change live_count");
    assert_eq!(table.len(), 1);
}

/// insert_at: live->live -- overwriting a live entry with another live
/// entry must NOT change live_count.
#[test]
fn test_insert_at_live_to_live() {
    let mut table = MeshTable::new();
    table.add(MeshTableEntry::new(10, 20, 30, 40, 1, 1));
    assert_eq!(table.live_count(), 1);

    table.insert_at(0, MeshTableEntry::new(99, 88, 77, 66, 2, 0));
    assert_eq!(table.live_count(), 1, "live->live must not change live_count");
    let e = table.get(0).unwrap();
    assert_eq!(e.index_offset, 99);
    assert_eq!(e.material_id, 2);
}

/// insert_at: hole->live -- filling a hole with a live entry must
/// increment live_count.
#[test]
fn test_insert_at_hole_to_live() {
    let mut table = MeshTable::new();
    table.add(MeshTableEntry::new(10, 20, 30, 40, 0, 1));
    table.remove(0);
    assert_eq!(table.live_count(), 0);

    table.insert_at(0, MeshTableEntry::new(5, 6, 7, 8, 3, 1));
    assert_eq!(table.live_count(), 1, "hole->live must increment live_count");
}

/// insert_at: live->hole -- zeroing a live entry must decrement live_count.
#[test]
fn test_insert_at_live_to_hole() {
    let mut table = MeshTable::new();
    table.add(MeshTableEntry::new(10, 20, 30, 40, 0, 1));
    assert_eq!(table.live_count(), 1);

    table.insert_at(0, MeshTableEntry::zero());
    assert_eq!(table.live_count(), 0, "live->hole must decrement live_count");
    assert!(table.get(0).unwrap().is_zero());
}

/// insert_at: extend table with a zero entry -- extending with zero must
/// NOT increment live_count (the implicit zero state is also not counted).
#[test]
fn test_insert_at_extend_with_zero() {
    let mut table = MeshTable::with_capacity(2);
    // Table is empty. Insert zero at index 5 (extending).
    table.insert_at(5, MeshTableEntry::zero());
    assert_eq!(table.len(), 6);
    assert_eq!(table.live_count(), 0, "extend with zero must not change live_count");
    // All entries (including the one we wrote) are zero.
    for i in 0..6 {
        assert!(table.get(i).unwrap().is_zero());
    }
}

/// insert_at: extend table with a live entry -- extending beyond len with
/// a non-zero entry must increment live_count.
#[test]
fn test_insert_at_extend_with_live() {
    let mut table = MeshTable::new();
    // Add one entry at index 0.
    table.add(MeshTableEntry::new(1, 0, 0, 0, 0, 0));
    assert_eq!(table.live_count(), 1);
    assert_eq!(table.len(), 1);

    // Insert live at index 3 (extend by 3 zeros, then write live).
    table.insert_at(3, MeshTableEntry::new(99, 0, 0, 0, 5, 1));
    assert_eq!(table.len(), 4);
    // live_count: index 0 + index 3 = 2.
    assert_eq!(table.live_count(), 2, "extend with live must increment live_count");
    // Indices 1..2 are zeros (holes from resize).
    assert!(table.get(1).unwrap().is_zero());
    assert!(table.get(2).unwrap().is_zero());
    assert_eq!(table.get(3).unwrap().material_id, 5);
}

/// insert_at: write exactly at len() boundary (no resize needed, but
/// at the edge). This is the same as add but via insert_at.
#[test]
fn test_insert_at_exact_len_boundary() {
    let mut table = MeshTable::new();
    table.add(MeshTableEntry::new(1, 2, 3, 4, 0, 1));
    let len = table.len(); // 1

    // Insert at index == len -> extends by 1.
    table.insert_at(len as u32, MeshTableEntry::new(5, 6, 7, 8, 1, 0));
    assert_eq!(table.len(), len + 1);
    assert_eq!(table.live_count(), 2);
    assert_eq!(table.get(1).unwrap().index_offset, 5);
}

/// insert_at: index 0 on an empty table (first entry).
#[test]
fn test_insert_at_index_zero_on_empty() {
    let mut table = MeshTable::new();
    assert!(table.is_empty());

    table.insert_at(0, MeshTableEntry::new(42, 1, 2, 3, 7, 1));
    assert_eq!(table.len(), 1);
    assert_eq!(table.live_count(), 1);
    assert_eq!(table.get(0).unwrap().index_offset, 42);
}

/// insert_at: zero entry into an empty table at index 0 (should stay
/// at 0 live_count since the entry is zero).
#[test]
fn test_insert_at_zero_into_empty() {
    let mut table = MeshTable::new();
    table.insert_at(0, MeshTableEntry::zero());
    assert_eq!(table.len(), 1);
    assert_eq!(table.live_count(), 0, "zero entry into empty must keep live_count=0");
    assert!(table.get(0).unwrap().is_zero());
}

// =========================================================================
// Whitebox: invariant -- live_count === count_non_zero under stress
// =========================================================================

/// Helper: count non-zero entries by scanning the entries vec.
fn count_live_entries(table: &MeshTable) -> usize {
    table.as_slice().iter().filter(|e| !e.is_zero()).count()
}

/// After a sequence of mixed operations, live_count must equal the
/// scanned count of non-zero entries.
#[test]
fn test_live_count_invariant_mixed_operations() {
    let mut table = MeshTable::new();

    // Step 1: add 5 entries.
    for i in 0..5u32 {
        table.add(MeshTableEntry::new(i, i + 10, 0, 0, i, 1));
    }
    assert_eq!(table.live_count(), count_live_entries(&table));

    // Step 2: remove index 1 and 3.
    table.remove(1);
    table.remove(3);
    assert_eq!(table.live_count(), count_live_entries(&table));

    // Step 3: update index 0 to zero (create hole).
    table.update(0, MeshTableEntry::zero());
    assert_eq!(table.live_count(), count_live_entries(&table));

    // Step 4: insert_at index 2 with a live entry.
    table.insert_at(2, MeshTableEntry::new(99, 0, 0, 0, 0, 0));
    assert_eq!(table.live_count(), count_live_entries(&table));

    // Step 5: insert_at index 5 (extend) with a live entry.
    table.insert_at(5, MeshTableEntry::new(88, 0, 0, 0, 9, 1));
    assert_eq!(table.live_count(), count_live_entries(&table));

    // Step 6: clear and re-add.
    table.clear();
    assert_eq!(table.live_count(), count_live_entries(&table));
    for i in 0..3u32 {
        table.add(MeshTableEntry::new(i, 0, 0, 0, i, 1));
    }
    assert_eq!(table.live_count(), count_live_entries(&table));
}

/// Stress the invariant: 5,000 random-ish operations.
#[test]
fn test_live_count_invariant_stress() {
    let mut table = MeshTable::new();
    let mut expected_live = 0usize;

    for i in 0..5000u32 {
        let op = i % 8;
        match op {
            // add -- always increments live_count by 1
            0..=2 => {
                let entry = MeshTableEntry::new(i, i + 1, 0, 0, i % 16, 1);
                let _idx = table.add(entry);
                expected_live += 1;
            }
            // remove
            3..=4 => {
                if table.len() > 0 {
                    let idx = (i % table.len() as u32).max(0);
                    let was_live = !table.get(idx).map_or(true, |e| e.is_zero());
                    table.remove(idx);
                    if was_live {
                        expected_live = expected_live.saturating_sub(1);
                    }
                }
            }
            // update
            5..=6 => {
                if table.len() > 0 {
                    let idx = (i % table.len() as u32).max(0);
                    let was_zero = table.get(idx).map_or(true, |e| e.is_zero());
                    let new_entry = if i % 3 == 0 {
                        MeshTableEntry::zero()
                    } else {
                        MeshTableEntry::new(i * 2, 0, 0, 0, i % 16, 1)
                    };
                    let is_zero = new_entry.is_zero();
                    table.update(idx, new_entry);
                    if was_zero && !is_zero {
                        expected_live += 1;
                    } else if !was_zero && is_zero {
                        expected_live = expected_live.saturating_sub(1);
                    }
                }
            }
            // insert_at
            7 => {
                let idx = if table.len() < 10 {
                    table.len() as u32
                } else {
                    (i % table.len() as u32).max(0)
                };
                let entry = if i % 5 == 0 {
                    MeshTableEntry::zero()
                } else {
                    MeshTableEntry::new(i, 0, 0, 0, i % 16, 1)
                };
                let was_extend = (idx as usize) >= table.len();
                let old_was_zero = if was_extend {
                    true // implicit zero
                } else {
                    table.get(idx).map_or(true, |e| e.is_zero())
                };
                let new_is_zero = entry.is_zero();

                table.insert_at(idx, entry);

                if was_extend {
                    if !new_is_zero {
                        expected_live += 1;
                    }
                } else {
                    if old_was_zero && !new_is_zero {
                        expected_live += 1;
                    } else if !old_was_zero && new_is_zero {
                        expected_live = expected_live.saturating_sub(1);
                    }
                }
            }
            _ => {}
        }

        // Verify invariant every 500 ops.
        if i % 500 == 0 {
            assert_eq!(
                table.live_count(),
                count_live_entries(&table),
                "invariant broken at op {} (expected_live={})",
                i, expected_live
            );
        }
    }

    // Final invariant check.
    assert_eq!(table.live_count(), count_live_entries(&table));
    assert_eq!(table.live_count(), expected_live);
}

// =========================================================================
// Whitebox: complex interleaved mutation sequences
// =========================================================================

/// Interleave add, remove, update, insert_at across multiple indices.
#[test]
fn test_complex_mutation_interleaving() {
    let mut table = MeshTable::new();

    // Add 4 entries.
    for i in 0..4u32 {
        table.add(MeshTableEntry::new(i * 10, 0, 0, 0, i, 1));
    }

    // Remove index 1.
    table.remove(1);
    assert_eq!(table.live_count(), 3);

    // insert_at index 1 with a new live entry (fill hole).
    table.insert_at(1, MeshTableEntry::new(99, 0, 0, 0, 7, 0));
    assert_eq!(table.live_count(), 4);

    // Update index 3 to zero (create hole).
    table.update(3, MeshTableEntry::zero());
    assert_eq!(table.live_count(), 3);

    // Add more entries to extend the table.
    table.add(MeshTableEntry::new(200, 0, 0, 0, 8, 1));
    table.add(MeshTableEntry::new(300, 0, 0, 0, 9, 1));
    assert_eq!(table.live_count(), 5);

    // Verify the structure.
    assert_eq!(table.len(), 6);
    assert_eq!(table.get(0).unwrap().index_offset, 0);
    assert_eq!(table.get(1).unwrap().index_offset, 99);
    assert_eq!(table.get(2).unwrap().index_offset, 20);
    assert!(table.get(3).unwrap().is_zero());
    assert_eq!(table.get(4).unwrap().index_offset, 200);
    assert_eq!(table.get(5).unwrap().index_offset, 300);

    // Clear and verify.
    table.clear();
    assert!(table.is_empty());
    assert_eq!(table.live_count(), 0);
}

/// Fill holes via insert_at: remove entries then re-fill via insert_at.
#[test]
fn test_fill_holes_via_insert_at() {
    let mut table = MeshTable::new();
    for i in 0..5u32 {
        table.add(MeshTableEntry::new(i, 0, 0, 0, i, 1));
    }
    assert_eq!(table.live_count(), 5);

    // Remove indices 1 and 3.
    table.remove(1);
    table.remove(3);
    assert_eq!(table.live_count(), 3);

    // Fill holes via insert_at.
    table.insert_at(1, MeshTableEntry::new(111, 0, 0, 0, 1, 1));
    table.insert_at(3, MeshTableEntry::new(333, 0, 0, 0, 3, 1));
    assert_eq!(table.live_count(), 5);

    // Verify the filled entries.
    assert_eq!(table.get(1).unwrap().index_offset, 111);
    assert_eq!(table.get(3).unwrap().index_offset, 333);
    assert_eq!(table.get(0).unwrap().index_offset, 0);
    assert_eq!(table.get(2).unwrap().index_offset, 2);
    assert_eq!(table.get(4).unwrap().index_offset, 4);
}

/// Remove all entries sequentially from the front (0, 1, 2, ...).
#[test]
fn test_remove_all_entries_sequentially() {
    let mut table = MeshTable::new();
    for i in 0..10u32 {
        table.add(MeshTableEntry::new(i, 0, 0, 0, i, 1));
    }
    assert_eq!(table.live_count(), 10);

    for i in 0..10u32 {
        table.remove(i);
    }
    assert_eq!(table.live_count(), 0);
    // len is still 10 (holes), but is_empty() checks live_count.
    assert!(table.is_empty(), "no live entries => is_empty must be true");
    assert_eq!(table.len(), 10);

    // Verify all are holes.
    for i in 0..10 {
        assert!(table.get(i).unwrap().is_zero());
    }
}

/// Remove all entries in reverse order.
#[test]
fn test_remove_all_entries_reverse() {
    let mut table = MeshTable::new();
    for i in 0..10u32 {
        table.add(MeshTableEntry::new(i, 0, 0, 0, i, 1));
    }

    for i in (0..10u32).rev() {
        table.remove(i);
    }
    assert_eq!(table.live_count(), 0);
    assert_eq!(table.len(), 10);

    for i in 0..10 {
        assert!(table.get(i).unwrap().is_zero());
    }
}

// =========================================================================
// Whitebox: stress tests
// =========================================================================

/// Add 10,000 entries to verify the table scales.
#[test]
fn test_large_table_stress() {
    let mut table = MeshTable::with_capacity(10000);
    for i in 0..10000u32 {
        let idx = table.add(MeshTableEntry::new(i, i * 2, i * 3, i * 4, i % 256, 1));
        assert_eq!(idx, i);
    }
    assert_eq!(table.len(), 10000);
    assert_eq!(table.live_count(), 10000);

    // Verify a few entries.
    assert_eq!(table.get(0).unwrap().index_offset, 0);
    assert_eq!(table.get(5000).unwrap().index_offset, 5000);
    assert_eq!(table.get(9999).unwrap().index_offset, 9999);

    // Verify as_bytes length.
    assert_eq!(table.as_bytes().len(), 10000 * MESH_TABLE_ENTRY_SIZE);
}

/// Stress test with 5,000 mixed add/remove/update operations.
#[test]
fn test_mixed_operations_stress() {
    let mut table = MeshTable::new();

    // Phase 1: add 2000 entries.
    for i in 0..2000u32 {
        table.add(MeshTableEntry::new(i, 0, 0, 0, i % 64, if i % 2 == 0 { 1 } else { 0 }));
    }
    assert_eq!(table.live_count(), 2000);

    // Phase 2: remove every other entry (1000 removes).
    for i in (0..2000u32).step_by(2) {
        table.remove(i);
    }
    // live_count: 2000 - 1000 = 1000.
    assert_eq!(table.live_count(), 1000);

    // Phase 3: update 500 entries.
    for i in (1..1000u32).step_by(2) {
        table.update(i, MeshTableEntry::new(i + 5000, 0, 0, 0, i % 32, 1));
    }
    assert_eq!(table.live_count(), 1000);

    // Phase 4: insert_at 500 entries at extension positions.
    let base = table.len();
    for i in 0..500u32 {
        table.insert_at((base + i as usize) as u32, MeshTableEntry::new(i, 0, 0, 0, i % 16, 1));
    }
    assert_eq!(table.live_count(), 1500);

    // Verify invariant.
    assert_eq!(table.live_count(), count_live_entries(&table));
}

// =========================================================================
// Whitebox: as_bytes after complex mutations
// =========================================================================

/// as_bytes must reflect the correct byte representation after interleaved
/// add/remove/update/insert_at operations.
#[test]
fn test_as_bytes_after_mutations() {
    let mut table = MeshTable::new();

    // Add entries.
    for i in 0..4u32 {
        table.add(MeshTableEntry::new(i * 16, i * 8, i * 4, i * 2, i, 1));
    }

    // Remove index 1.
    table.remove(1);

    // Update index 2.
    table.update(2, MeshTableEntry::new(999, 888, 0, 0, 7, 1));

    // Insert at extension.
    table.insert_at(5, MeshTableEntry::new(42, 0, 0, 0, 9, 1));

    let bytes = table.as_bytes();
    let expected_len = 6 * MESH_TABLE_ENTRY_SIZE;
    assert_eq!(bytes.len(), expected_len);

    // Reinterpret back and verify.
    let entries: &[MeshTableEntry] = unsafe {
        std::slice::from_raw_parts(bytes.as_ptr() as *const MeshTableEntry, 6)
    };
    assert_eq!(entries[0].index_offset, 0);
    assert!(entries[1].is_zero(), "removed entry must be zero in bytes");
    assert_eq!(entries[2].index_offset, 999);
    assert_eq!(entries[2].material_id, 7);
    assert_eq!(entries[3].index_offset, 48); // 3 * 16
    assert_eq!(entries[5].index_offset, 42);
}

/// as_bytes length must always equal len() * MESH_TABLE_ENTRY_SIZE, even
/// after mutations that create holes.
#[test]
fn test_as_bytes_len_consistent() {
    let mut table = MeshTable::new();

    let check = |t: &MeshTable| {
        assert_eq!(t.as_bytes().len(), t.len() * MESH_TABLE_ENTRY_SIZE);
    };

    check(&table);

    // After adds.
    for i in 0..5u32 {
        table.add(MeshTableEntry::new(i, 0, 0, 0, i, 1));
    }
    check(&table);

    // After removes.
    table.remove(0);
    table.remove(2);
    check(&table);

    // After insert_at.
    table.insert_at(3, MeshTableEntry::new(99, 0, 0, 0, 5, 0));
    check(&table);

    // After clear.
    table.clear();
    check(&table);
}

// =========================================================================
// Whitebox: staging after complex mutation sequences
// =========================================================================

/// stage() must correctly serialize the table contents after complex
/// mutation sequences.
#[test]
fn test_stage_after_complex_mutations() {
    let mut registry = crate::gpu_driven::buffers::BufferRegistry::new(4096);
    let mut table = MeshTable::new();

    // Build up a table with some entries.
    for i in 0..5u32 {
        table.add(MeshTableEntry::new(i * 10, 0, 0, 0, i, 1));
    }

    // Mutate it.
    table.remove(0);
    table.remove(3);
    table.update(1, MeshTableEntry::new(999, 0, 0, 0, 1, 0));
    table.insert_at(5, MeshTableEntry::new(42, 0, 0, 0, 9, 1));

    // Stage and verify.
    let (slot_idx, byte_size) = table.stage(&mut registry).expect("staging must succeed");
    assert_eq!(byte_size, 6 * MESH_TABLE_ENTRY_SIZE);

    let slot = registry.slot_mut(slot_idx).unwrap();
    let staged: &[MeshTableEntry] = unsafe {
        std::slice::from_raw_parts(slot.as_slice().as_ptr() as *const MeshTableEntry, 6)
    };
    assert!(staged[0].is_zero());
    assert_eq!(staged[1].index_offset, 999);
    assert_eq!(staged[2].index_offset, 20);
    assert!(staged[3].is_zero());
    assert_eq!(staged[4].index_offset, 40);
    assert_eq!(staged[5].index_offset, 42);
}

/// Multiple stage_and_submit cycles across different table states.
#[test]
fn test_multiple_stage_and_submit_cycles() {
    let mut registry = crate::gpu_driven::buffers::BufferRegistry::new(4096);
    let mut table = MeshTable::new();

    // Cycle 1: empty table -- stage_and_submit should return false.
    assert!(!table.stage_and_submit(&mut registry), "empty table must not stage");

    // Cycle 2: add entries and stage.
    for i in 0..3u32 {
        table.add(MeshTableEntry::new(i, 0, 0, 0, i, 1));
    }
    assert!(table.stage_and_submit(&mut registry));

    // Cycle 3: mutate and stage again.
    table.remove(1);
    table.add(MeshTableEntry::new(100, 0, 0, 0, 5, 1));
    assert!(table.stage_and_submit(&mut registry));

    // Cycle 4: clear and stage (should return false since empty).
    table.clear();
    assert!(!table.stage_and_submit(&mut registry), "cleared table must not stage");
}

// =========================================================================
// Whitebox: reserve edge cases
// =========================================================================

/// Reserve after clear: ensure no stale capacity issues.
#[test]
fn test_reserve_after_clear() {
    let mut table = MeshTable::with_capacity(10);
    for i in 0..10u32 {
        table.add(MeshTableEntry::new(i, 0, 0, 0, i, 1));
    }
    assert_eq!(table.len(), 10);

    table.clear();
    assert!(table.is_empty());

    // Reserve additional capacity.
    table.reserve(100);
    // Adding entries should not reallocate.
    for i in 0..100u32 {
        let idx = table.add(MeshTableEntry::new(i, 0, 0, 0, i % 16, 1));
        assert_eq!(idx, i);
    }
    assert_eq!(table.len(), 100);
    assert_eq!(table.live_count(), 100);
}

/// Reserve(0) must be a no-op.
#[test]
fn test_reserve_zero_is_noop() {
    let mut table = MeshTable::new();
    let cap_before = table.entries.capacity();
    table.reserve(0);
    assert_eq!(table.entries.capacity(), cap_before);
}

/// Reserve on an empty fresh table must not panic and must allow adds.
#[test]
fn test_reserve_on_empty_table() {
    let mut table = MeshTable::new();
    table.reserve(50);
    assert!(table.entries.capacity() >= 50);

    for i in 0..50u32 {
        table.add(MeshTableEntry::new(i, 0, 0, 0, i, 1));
    }
    assert_eq!(table.len(), 50);
}

// =========================================================================
// Whitebox: large field values
// =========================================================================

/// MeshTableEntry must store u32::MAX values without overflow.
#[test]
fn test_entry_u32_max_values() {
    let entry = MeshTableEntry::new(u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX);
    assert_eq!(entry.index_offset, u32::MAX);
    assert_eq!(entry.vertex_offset, u32::MAX);
    assert_eq!(entry.index_count, u32::MAX);
    assert_eq!(entry.vertex_count, u32::MAX);
    assert_eq!(entry.material_id, u32::MAX);
    assert_eq!(entry.flags, u32::MAX);

    // Must NOT be is_zero.
    assert!(!entry.is_zero());

    // Must survive add/get round-trip.
    let mut table = MeshTable::new();
    let idx = table.add(entry);
    let retrieved = table.get(idx).unwrap();
    assert_eq!(retrieved.index_offset, u32::MAX);
    assert_eq!(retrieved.flags, u32::MAX);
}

/// Update entries with u32::MAX field values.
#[test]
fn test_update_entry_with_max_values() {
    let mut table = MeshTable::new();
    // Add a zero entry via add(). add() always increments live_count.
    table.add(MeshTableEntry::new(0, 0, 0, 0, 0, 0));
    // Enter is zero but add() unconditionally increments live_count.
    assert_eq!(table.live_count(), 1, "add always increments live_count");

    // Update with max values -- live_count goes from 1 to 2.
    table.update(0, MeshTableEntry::new(u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX));
    assert!(!table.get(0).unwrap().is_zero());
    assert_eq!(table.get(0).unwrap().index_offset, u32::MAX);
    assert_eq!(table.live_count(), 2);

    // Update back to zero -- live_count goes from 2 to 1.
    table.update(0, MeshTableEntry::zero());
    assert!(table.get(0).unwrap().is_zero());
    assert_eq!(table.live_count(), 1);
}

// =========================================================================
// Whitebox: Display consistency after mutations
// =========================================================================

/// Display output must reflect the current table state after mutations.
#[test]
fn test_display_after_mutations() {
    let mut table = MeshTable::new();
    assert_eq!(format!("{}", table), "MeshTable(entries=0, live=0)");

    table.add(MeshTableEntry::new(10, 20, 30, 40, 1, 1));
    assert_eq!(format!("{}", table), "MeshTable(entries=1, live=1)");

    table.add(MeshTableEntry::new(50, 60, 70, 80, 2, 0));
    assert_eq!(format!("{}", table), "MeshTable(entries=2, live=2)");

    table.remove(0);
    assert_eq!(format!("{}", table), "MeshTable(entries=2, live=1)");

    table.clear();
    assert_eq!(format!("{}", table), "MeshTable(entries=0, live=0)");
}

/// MeshTableEntry Display must handle all field values including
/// zero and u32::MAX.
#[test]
fn test_entry_display_max_values() {
    let entry = MeshTableEntry::new(u32::MAX, 0, 0, 0, 255, 0xFFFFFFFF);
    let s = format!("{}", entry);
    assert!(s.contains("idx_off=4294967295"));
    assert!(s.contains("mat=255"));
    assert!(s.contains("flags=0xffffffff"));
}

// =========================================================================
// Whitebox: default mesh table
// =========================================================================

/// Default must use DEFAULT_MESH_TABLE_CAPACITY initial capacity.
#[test]
fn test_default_mesh_table_capacity() {
    let table = MeshTable::default();
    assert_eq!(table.entries.capacity(), DEFAULT_MESH_TABLE_CAPACITY);
}


}
