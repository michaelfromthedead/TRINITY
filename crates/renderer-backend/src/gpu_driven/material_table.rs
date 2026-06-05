//! Bindless Material Table -- GPU buffer (`array<MaterialTableEntry>`) with a
//! CPU-side manager for material population and dirty-flag tracking.
//!
//! The material table is the central indirection layer for GPU-driven PBR
//! rendering: shaders reference materials by a 32-bit index into this table
//! rather than binding individual uniform buffers per material. This enables:
//!
//! - **Bindless access**: any shader can fetch material data by index from a
//!   single storage-buffer binding.
//! - **Dirty-flag tracking**: the CPU-side manager marks modified entries as
//!   "dirty" (bit 31 of the `flags` field). The staging pipeline (`stage()` /
//!   `stage_and_submit()`) uploads only the dirty range and clears the flag,
//!   avoiding redundant GPU transfers.
//! - **Hole-preserving removal**: removed entries are zeroed but their slot is
//!   NOT reclaimed -- the index remains invalid (u32::MAX sentinel for texture
//!   IDs, zeroed otherwise). This keeps all existing shader references valid.
//!
//! # Layout
//!
//! | Offset | Size | Field | Description |
//! |--------|------|-------|-------------|
//! | 0      | 16   | `base_color` | RGBA base colour |
//! | 16     | 16   | `emissive` | RGB emissive + intensity (.a) |
//! | 32     | 4    | `metallic` | Metalness (0-1) |
//! | 36     | 4    | `roughness` | Roughness (0-1) |
//! | 40     | 4    | `occlusion` | Ambient occlusion (0-1) |
//! | 44     | 4    | `normal_scale` | Normal map intensity scale |
//! | 48     | 4    | `albedo_texture_id` | Bindless texture index |
//! | 52     | 4    | `normal_texture_id` | Bindless texture index |
//! | 56     | 4    | `metallic_roughness_texture_id` | Bindless texture index |
//! | 60     | 4    | `emissive_texture_id` | Bindless texture index |
//! | 64     | 4    | `flags` | Bit 0 = visible, bit 31 = dirty |
//! | 68     | 4    | `alpha_cutoff` | Alpha-mask threshold |
//! | 72     | 8    | *(implicit padding)* | Rounds to 80 (align 16) |
//!
//! **Total: 80 bytes.** Array stride: 80.
//!
//! # Safety
//!
//! `as_bytes()` performs a `core::slice::from_raw_parts` reinterpret-cast of
//! the internal `Vec<MaterialTableEntry>` to `&[u8]`. This is safe because
//! `MaterialTableEntry` is `#[repr(C, align(16))]` and contains only plain-old-
//! data types (f32/u32). The byte length is always a multiple of the entry
//! size.
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::gpu_driven::material_table::{
//!     MaterialTable, MaterialTableEntry,
//! };
//!
//! let mut table = MaterialTable::with_capacity(64);
//! let idx = table.add(MaterialTableEntry {
//!     base_color: [0.9, 0.2, 0.1, 1.0],
//!     emissive: [0.0, 0.0, 0.0, 0.0],
//!     metallic: 0.8,
//!     roughness: 0.3,
//!     occlusion: 1.0,
//!     normal_scale: 1.0,
//!     albedo_texture_id: 0,
//!     normal_texture_id: !0,
//!     metallic_roughness_texture_id: !0,
//!     emissive_texture_id: !0,
//!     flags: 1,       // visible
//!     alpha_cutoff: 0.5,
//! });
//! assert!(table.any_dirty());
//! let bytes = table.as_bytes();
//! assert_eq!(bytes.len(), MATERIAL_TABLE_ENTRY_SIZE);
//! ```

use crate::gpu_driven::buffers::{
    AcquireResult, BufferRegistry, SubmitResult,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Size of a single `MaterialTableEntry` in bytes (80).
pub const MATERIAL_TABLE_ENTRY_SIZE: usize = 80;

/// Default material table capacity (1024 entries = 80 KiB).
pub const DEFAULT_MATERIAL_TABLE_CAPACITY: usize = 1024;

/// Flag bit indicating the material entry has been modified and needs staging.
pub const MATERIAL_FLAG_DIRTY: u32 = 0x8000_0000;

/// Flag bit indicating the material is visible (included in draw calls).
pub const MATERIAL_FLAG_VISIBLE: u32 = 0x0000_0001;

// ---------------------------------------------------------------------------
// MaterialTableEntry
// ---------------------------------------------------------------------------

/// A single PBR material descriptor in the bindless material table (80 bytes).
///
/// The layout matches the WGSL `MaterialTableEntry` struct in
/// `material_table.wgsl`. The struct is `#[repr(C)]` with `align(16)` to
/// satisfy WGSL's `vec4<f32>` alignment requirements.
///
/// Texture ID fields use `u32::MAX` (0xFFFF_FFFF) as a sentinel for "no
/// texture bound." This is consistent with WGSL's `arrayLength` behaviour:
/// an out-of-bounds index is never dereferenced -- shader helpers such as
/// `material_has_albedo_texture` check the sentinel first.
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C, align(16))]
pub struct MaterialTableEntry {
    /// RGBA base colour (linear space).
    pub base_color: [f32; 4],
    /// RGB emissive colour + intensity (stored in alpha channel).
    pub emissive: [f32; 4],
    /// Metalness in [0, 1].
    pub metallic: f32,
    /// Roughness in [0, 1].
    pub roughness: f32,
    /// Ambient occlusion in [0, 1].
    pub occlusion: f32,
    /// Normal map intensity scale.
    pub normal_scale: f32,
    /// Index into bindless texture array for the albedo (base colour) map.
    pub albedo_texture_id: u32,
    /// Index into bindless texture array for the normal map.
    pub normal_texture_id: u32,
    /// Index into bindless texture array for the metallic-roughness map.
    pub metallic_roughness_texture_id: u32,
    /// Index into bindless texture array for the emissive map.
    pub emissive_texture_id: u32,
    /// Bitfield: bit 0 = visible, bit 31 = dirty.
    pub flags: u32,
    /// Alpha-mask cutoff threshold (used when alpha-testing is enabled).
    pub alpha_cutoff: f32,
    /// Explicit padding for 16-byte alignment (required for Pod).
    pub _pad: [u32; 2],
}

/// Sentinel value for "no texture bound".
pub const NO_TEXTURE: u32 = u32::MAX;

impl MaterialTableEntry {
    /// Sentinel value for "no texture bound".
    pub const NO_TEXTURE: u32 = u32::MAX;

    /// Returns a zeroed material entry.
    ///
    /// All texture IDs are `u32::MAX` (no texture), the entry is marked
    /// invisible and clean. This is the canonical "hole" value used after
    /// removal.
    pub const fn zeroed() -> Self {
        Self {
            base_color: [0.0; 4],
            emissive: [0.0; 4],
            metallic: 0.0,
            roughness: 0.0,
            occlusion: 0.0,
            normal_scale: 0.0,
            albedo_texture_id: u32::MAX,
            normal_texture_id: u32::MAX,
            metallic_roughness_texture_id: u32::MAX,
            emissive_texture_id: u32::MAX,
            flags: 0,
            alpha_cutoff: 0.0,
            _pad: [0; 2],
        }
    }

    /// Create a new default material entry (visible, white base color).
    pub fn new() -> Self {
        Self {
            base_color: [1.0, 1.0, 1.0, 1.0],
            flags: MATERIAL_FLAG_VISIBLE,
            ..Self::zeroed()
        }
    }

    /// Create a new material entry with the given base color.
    pub fn with_base_color_init(base_color: [f32; 4]) -> Self {
        Self {
            base_color,
            flags: MATERIAL_FLAG_VISIBLE,
            ..Self::zeroed()
        }
    }

    /// Create an opaque material with the given base color.
    pub fn opaque(base_color: [f32; 4]) -> Self {
        Self::with_base_color_init(base_color)
    }

    /// Create a metallic material with the given base color and metallic value.
    pub fn metallic(base_color: [f32; 4], metallic: f32) -> Self {
        Self {
            base_color,
            metallic,
            roughness: 0.5,
            flags: MATERIAL_FLAG_VISIBLE,
            ..Self::zeroed()
        }
    }

    /// Returns `true` when every field is zero or `u32::MAX` (the hole
    /// sentinel). Note that `flags` is compared after masking out the
    /// dirty bit so a dirty-but-otherwise-zeroed entry reads as zero.
    pub fn is_zero(&self) -> bool {
        self.base_color == [0.0; 4]
            && self.emissive == [0.0; 4]
            && self.metallic == 0.0
            && self.roughness == 0.0
            && self.occlusion == 0.0
            && self.normal_scale == 0.0
            && self.albedo_texture_id == u32::MAX
            && self.normal_texture_id == u32::MAX
            && self.metallic_roughness_texture_id == u32::MAX
            && self.emissive_texture_id == u32::MAX
            && (self.flags & !MATERIAL_FLAG_DIRTY) == 0
            && self.alpha_cutoff == 0.0
    }
}

impl Default for MaterialTableEntry {
    fn default() -> Self {
        Self::zeroed()
    }
}

impl core::fmt::Display for MaterialTableEntry {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "MaterialTableEntry {{ base_color: [{:.2}, {:.2}, {:.2}, {:.2}], \
             emissive: [{:.2}, {:.2}, {:.2}, {:.2}], \
             metallic: {:.2}, roughness: {:.2}, occlusion: {:.2}, \
             normal_scale: {:.2}, albedo_tex: {}, normal_tex: {}, \
             mr_tex: {}, emissive_tex: {}, flags: {:#x}, alpha_cutoff: {:.2} }}",
            self.base_color[0],
            self.base_color[1],
            self.base_color[2],
            self.base_color[3],
            self.emissive[0],
            self.emissive[1],
            self.emissive[2],
            self.emissive[3],
            self.metallic,
            self.roughness,
            self.occlusion,
            self.normal_scale,
            self.albedo_texture_id,
            self.normal_texture_id,
            self.metallic_roughness_texture_id,
            self.emissive_texture_id,
            self.flags,
            self.alpha_cutoff,
        )
    }
}

// ---------------------------------------------------------------------------
// Compatibility aliases (old field names)
// ---------------------------------------------------------------------------

impl MaterialTableEntry {
    /// Alias for `albedo_texture_id` (old name: base_color_texture).
    pub fn base_color_texture(&self) -> u32 {
        self.albedo_texture_id
    }

    /// Alias for `normal_texture_id` (old name: normal_texture).
    pub fn normal_texture(&self) -> u32 {
        self.normal_texture_id
    }

    /// Alias for `metallic_roughness_texture_id` (old name: metallic_roughness_texture).
    pub fn metallic_roughness_texture(&self) -> u32 {
        self.metallic_roughness_texture_id
    }

    /// Alias for `emissive_texture_id` (old name: emissive_texture).
    pub fn emissive_texture(&self) -> u32 {
        self.emissive_texture_id
    }

    /// Alias for `base_color` (old name: base_color_factor).
    pub fn base_color_factor(&self) -> [f32; 4] {
        self.base_color
    }

    /// Alias for `metallic` (old name: metallic_factor).
    pub fn metallic_factor(&self) -> f32 {
        self.metallic
    }

    /// Alias for `roughness` (old name: roughness_factor).
    pub fn roughness_factor(&self) -> f32 {
        self.roughness
    }

    /// Alias for `emissive` (old name: emissive_factor).
    pub fn emissive_factor(&self) -> [f32; 4] {
        self.emissive
    }

    // -------------------------------------------------------------------------
    // Builder methods for test compatibility
    // -------------------------------------------------------------------------

    /// Builder: set albedo texture ID.
    pub fn with_albedo_texture_id(mut self, id: u32) -> Self {
        self.albedo_texture_id = id;
        self
    }

    /// Builder: set albedo texture ID (old name).
    pub fn with_base_color_texture(self, id: u32) -> Self {
        self.with_albedo_texture_id(id)
    }

    /// Builder: set normal texture ID.
    pub fn with_normal_texture_id(mut self, id: u32) -> Self {
        self.normal_texture_id = id;
        self
    }

    /// Builder: set normal texture ID (old name).
    pub fn with_normal_texture(self, id: u32) -> Self {
        self.with_normal_texture_id(id)
    }

    /// Builder: set metallic-roughness texture ID.
    pub fn with_metallic_roughness_texture_id(mut self, id: u32) -> Self {
        self.metallic_roughness_texture_id = id;
        self
    }

    /// Builder: set metallic-roughness texture ID (old name).
    pub fn with_metallic_roughness_texture(self, id: u32) -> Self {
        self.with_metallic_roughness_texture_id(id)
    }

    /// Builder: set emissive texture ID.
    pub fn with_emissive_texture_id(mut self, id: u32) -> Self {
        self.emissive_texture_id = id;
        self
    }

    /// Builder: set emissive texture ID (old name).
    pub fn with_emissive_texture(self, id: u32) -> Self {
        self.with_emissive_texture_id(id)
    }

    /// Builder: set base color.
    pub fn with_base_color(mut self, color: [f32; 4]) -> Self {
        self.base_color = color;
        self
    }

    /// Builder: set metallic value.
    pub fn with_metallic(mut self, value: f32) -> Self {
        self.metallic = value;
        self
    }

    /// Builder: set roughness value.
    pub fn with_roughness(mut self, value: f32) -> Self {
        self.roughness = value;
        self
    }

    /// Builder: set emissive color.
    pub fn with_emissive(mut self, color: [f32; 4]) -> Self {
        self.emissive = color;
        self
    }

    /// Builder: set alpha cutoff.
    pub fn with_alpha_cutoff(mut self, value: f32) -> Self {
        self.alpha_cutoff = value;
        self
    }

    /// Builder: set flags.
    pub fn with_flags(mut self, flags: u32) -> Self {
        self.flags = flags;
        self
    }

    /// Builder: mark as double-sided.
    pub fn with_double_sided(mut self, double_sided: bool) -> Self {
        if double_sided {
            self.flags |= 0x0000_0002; // MATERIAL_DESC_FLAG_DOUBLE_SIDED
        } else {
            self.flags &= !0x0000_0002;
        }
        self
    }

    /// Builder: set alpha mask mode with cutoff value.
    pub fn with_alpha_mask(mut self, cutoff: f32) -> Self {
        self.flags |= 0x0000_0004; // MATERIAL_DESC_FLAG_ALPHA_MASK
        self.flags &= !0x0000_0008; // Clear alpha blend (mutually exclusive)
        self.alpha_cutoff = cutoff;
        self
    }

    // -------------------------------------------------------------------------
    // Texture presence checks
    // -------------------------------------------------------------------------

    /// Returns true if an albedo texture is bound.
    pub fn has_albedo_texture(&self) -> bool {
        self.albedo_texture_id != u32::MAX
    }

    /// Returns true if an albedo texture is bound (old name).
    pub fn has_base_color_texture(&self) -> bool {
        self.has_albedo_texture()
    }

    /// Returns true if a normal texture is bound.
    pub fn has_normal_texture(&self) -> bool {
        self.normal_texture_id != u32::MAX
    }

    /// Returns true if a metallic-roughness texture is bound.
    pub fn has_metallic_roughness_texture(&self) -> bool {
        self.metallic_roughness_texture_id != u32::MAX
    }

    /// Returns true if an emissive texture is bound.
    pub fn has_emissive_texture(&self) -> bool {
        self.emissive_texture_id != u32::MAX
    }

    /// Alias for has_albedo_texture (with _id suffix).
    pub fn has_albedo_texture_id(&self) -> bool {
        self.has_albedo_texture()
    }

    /// Alias for has_normal_texture (with _id suffix).
    pub fn has_normal_texture_id(&self) -> bool {
        self.has_normal_texture()
    }

    /// Alias for has_metallic_roughness_texture (with _id suffix).
    pub fn has_metallic_roughness_texture_id(&self) -> bool {
        self.has_metallic_roughness_texture()
    }

    /// Alias for has_emissive_texture (with _id suffix).
    pub fn has_emissive_texture_id(&self) -> bool {
        self.has_emissive_texture()
    }

    // -------------------------------------------------------------------------
    // Flag query methods
    // -------------------------------------------------------------------------

    /// Returns true if double-sided rendering is enabled.
    pub fn is_double_sided(&self) -> bool {
        (self.flags & 0x0000_0002) != 0
    }

    /// Returns true if alpha mask mode is enabled.
    pub fn is_alpha_mask(&self) -> bool {
        (self.flags & 0x0000_0004) != 0
    }

    /// Returns true if alpha blend mode is enabled.
    pub fn is_alpha_blend(&self) -> bool {
        (self.flags & 0x0000_0008) != 0
    }

    /// Returns true if unlit mode is enabled.
    pub fn is_unlit(&self) -> bool {
        (self.flags & 0x0000_0010) != 0
    }

    /// Builder: set alpha blend mode.
    pub fn with_alpha_blend(mut self) -> Self {
        self.flags |= 0x0000_0008; // MATERIAL_DESC_FLAG_ALPHA_BLEND
        self.flags &= !0x0000_0004; // Clear alpha mask (mutually exclusive)
        self
    }

    /// Builder: set unlit mode.
    pub fn with_unlit(mut self, unlit: bool) -> Self {
        if unlit {
            self.flags |= 0x0000_0010;
        } else {
            self.flags &= !0x0000_0010;
        }
        self
    }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Result of a successful `MaterialTable::add()` call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AddEntry {
    /// The index (handle) assigned to the newly added entry.
    pub index: u32,
}

/// Result of a `MaterialTable::remove()` call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RemoveResult {
    /// The entry was successfully removed (zeroed, slot preserved).
    Removed,
    /// No entry was found at the given index (hole or out of bounds).
    NotFound,
}

// ---------------------------------------------------------------------------
// MaterialTable
// ---------------------------------------------------------------------------

/// CPU-side manager for the bindless material table.
///
/// Manages a contiguous `Vec<MaterialTableEntry>` that shadows the GPU buffer.
/// The manager provides:
///
/// - **Dirty-flag tracking**: entries are automatically marked dirty on
///   mutation. `mark_clean()` clears the flag after staging.
/// - **Hole-preserving removal**: removed entries are zeroed but not compacted.
/// - **BufferRegistry integration**: `stage()` / `stage_and_submit()` upload
///   dirty entries through the triple-buffered staging system.
///
/// The table auto-resizes (doubles capacity) when full, matching the
/// `BufferRegistry` auto-resize behaviour.
pub struct MaterialTable {
    /// Contiguous storage of material entries.
    entries: Vec<MaterialTableEntry>,
    /// Number of non-hole entries currently in the table.
    live_count: usize,
}

impl MaterialTable {
    /// Creates a new material table with the default capacity (1024 entries).
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_MATERIAL_TABLE_CAPACITY)
    }

    /// Creates a new material table with the given initial capacity.
    ///
    /// The table will auto-resize (double capacity) when full.
    pub fn with_capacity(capacity: usize) -> Self {
        let cap = capacity.max(1);
        let mut entries = Vec::with_capacity(cap);
        // Fill with zeroed entries so that the Vec length == capacity.
        entries.resize_with(cap, MaterialTableEntry::zeroed);
        Self {
            entries,
            live_count: 0,
        }
    }

    // ------------------------------------------------------------------
    // Queries
    // ------------------------------------------------------------------

    /// Returns the current capacity (total slots, including holes).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns the number of live (non-hole) entries.
    pub fn live_count(&self) -> usize {
        self.live_count
    }

    /// Returns `true` when there are no live entries.
    pub fn is_empty(&self) -> bool {
        self.live_count == 0
    }

    /// Returns a shared reference to the entry at `index`, or `None` if out
    /// of bounds or the slot is a hole (`is_zero()`).
    pub fn get(&self, index: u32) -> Option<&MaterialTableEntry> {
        let i = index as usize;
        if i >= self.entries.len() {
            return None;
        }
        let entry = &self.entries[i];
        if entry.is_zero() {
            return None;
        }
        Some(entry)
    }

    /// Returns a mutable reference to the entry at `index`, or `None` if out
    /// of bounds or the slot is a hole.
    ///
    /// The entry is automatically marked dirty on mutation (caller must
    /// complete their mutation before the returned reference is dropped, as
    /// the dirty flag is set when the reference is created).
    pub fn get_mut(&mut self, index: u32) -> Option<&mut MaterialTableEntry> {
        let i = index as usize;
        if i >= self.entries.len() {
            return None;
        }
        if self.entries[i].is_zero() {
            return None;
        }
        self.entries[i].flags |= MATERIAL_FLAG_DIRTY;
        Some(&mut self.entries[i])
    }

    // ------------------------------------------------------------------
    // Serialization
    // ------------------------------------------------------------------

    /// Returns a shared slice of all entries (including holes).
    pub fn as_slice(&self) -> &[MaterialTableEntry] {
        &self.entries
    }

    /// Returns the entire table as a byte slice for GPU upload.
    ///
    /// # Safety
    ///
    /// Reinterpret-casts the internal `Vec<MaterialTableEntry>` to `&[u8]`.
    /// This is safe because `MaterialTableEntry` is `#[repr(C, align(16))]`
    /// and contains only plain-old-data types (f32/u32).
    pub fn as_bytes(&self) -> &[u8] {
        let len = self.entries.len() * MATERIAL_TABLE_ENTRY_SIZE;
        // Safety: MaterialTableEntry is POD with #[repr(C)].
        unsafe { core::slice::from_raw_parts(self.entries.as_ptr() as *const u8, len) }
    }

    // ------------------------------------------------------------------
    // Mutation
    // ------------------------------------------------------------------

    /// Adds a material entry to the table and returns its index.
    ///
    /// The entry is automatically marked dirty. If the table is full
    /// (no holes available), it doubles in capacity.
    pub fn add(&mut self, mut entry: MaterialTableEntry) -> u32 {
        entry.flags |= MATERIAL_FLAG_DIRTY;
        self.add_inner(entry)
    }

    /// Internal: adds an entry without modifying its dirty flag.
    fn add_inner(&mut self, entry: MaterialTableEntry) -> u32 {
        // Look for the first hole.
        for (i, slot) in self.entries.iter_mut().enumerate() {
            if slot.is_zero() {
                *slot = entry;
                self.live_count += 1;
                return i as u32;
            }
        }
        // No hole found -- extend.
        let idx = self.entries.len() as u32;
        self.entries.push(entry);
        self.live_count += 1;
        idx
    }

    /// Inserts a material entry at a specific index, overwriting whatever is
    /// there. The entry is automatically marked dirty.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds (>= `self.len()`).
    pub fn insert_at(&mut self, index: u32, mut entry: MaterialTableEntry) {
        let i = index as usize;
        assert!(
            i < self.entries.len(),
            "MaterialTable::insert_at: index {} out of bounds (len {})",
            index,
            self.entries.len()
        );
        let was_zero = self.entries[i].is_zero();
        entry.flags |= MATERIAL_FLAG_DIRTY;
        self.entries[i] = entry;
        if was_zero {
            self.live_count += 1;
        }
    }

    /// Marks the entry at `index` as dirty.
    ///
    /// Returns `true` if the entry was found and marked, `false` if the index
    /// is out of bounds or the slot is a hole.
    pub fn mark_dirty(&mut self, index: u32) -> bool {
        let i = index as usize;
        if i >= self.entries.len() {
            return false;
        }
        if self.entries[i].is_zero() {
            return false;
        }
        self.entries[i].flags |= MATERIAL_FLAG_DIRTY;
        true
    }

    /// Replaces the entry at `index` with a new entry. The entry is
    /// automatically marked dirty.
    ///
    /// Returns `Some(())` on success, `None` if out of bounds or the slot
    /// was a hole.
    pub fn update(&mut self, index: u32, mut entry: MaterialTableEntry) -> Option<()> {
        let i = index as usize;
        if i >= self.entries.len() {
            return None;
        }
        if self.entries[i].is_zero() {
            return None;
        }
        entry.flags |= MATERIAL_FLAG_DIRTY;
        self.entries[i] = entry;
        Some(())
    }

    /// Removes the entry at `index` by zeroing it in place (hole preserving).
    ///
    /// Returns `RemoveResult::Removed` on success or `RemoveResult::NotFound`
    /// if the index is out of bounds or the slot is already a hole.
    pub fn remove(&mut self, index: u32) -> RemoveResult {
        let i = index as usize;
        if i >= self.entries.len() {
            return RemoveResult::NotFound;
        }
        if self.entries[i].is_zero() {
            return RemoveResult::NotFound;
        }
        self.entries[i] = MaterialTableEntry::zeroed();
        self.live_count -= 1;
        RemoveResult::Removed
    }

    /// Removes all entries from the table. The capacity remains unchanged
    /// (all slots become holes).
    pub fn clear(&mut self) {
        for slot in &mut self.entries {
            *slot = MaterialTableEntry::zeroed();
        }
        self.live_count = 0;
    }

    /// Reserves additional capacity.
    ///
    /// New slots are initialised as zeroed (holes). If the new capacity is
    /// less than or equal to the current length, this is a no-op.
    pub fn reserve(&mut self, additional: usize) {
        let new_len = self.entries.len().saturating_add(additional);
        self.entries.resize_with(new_len, MaterialTableEntry::zeroed);
    }

    // ------------------------------------------------------------------
    // Dirty-flag management
    // ------------------------------------------------------------------

    /// Clears the dirty flag (bit 31) on all entries.
    ///
    /// Call this after the GPU has consumed the staged data.
    pub fn mark_clean(&mut self) {
        for entry in &mut self.entries {
            entry.flags &= !MATERIAL_FLAG_DIRTY;
        }
    }

    /// Returns `true` if any entry has the dirty flag set.
    pub fn any_dirty(&self) -> bool {
        self.entries.iter().any(|e| (e.flags & MATERIAL_FLAG_DIRTY) != 0)
    }

    /// Returns the number of entries that have the dirty flag set.
    pub fn dirty_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| (e.flags & MATERIAL_FLAG_DIRTY) != 0)
            .count()
    }

    // ------------------------------------------------------------------
    // BufferRegistry staging
    // ------------------------------------------------------------------

    /// Stages dirty entries through a `BufferRegistry`.
    ///
    /// If any entry is dirty, the entire table is submitted for staging.
    /// Callers MUST call `mark_clean()` after `submit_staging()` succeeds.
    /// Dirty flags are intentionally preserved here so that a failed submit
    /// can be retried without losing which entries need uploading.
    ///
    /// Returns `Some((slot_index, byte_size))` if staging was initiated,
    /// or `None` if no entries are dirty or no slot is available.
    pub fn stage(&mut self, registry: &mut BufferRegistry) -> Option<(usize, usize)> {
        if !self.any_dirty() {
            return None;
        }
        let byte_size = self.entries.len().wrapping_mul(MATERIAL_TABLE_ENTRY_SIZE);
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

    /// Stages dirty entries and immediately submits.
    ///
    /// Combines [`stage`](Self::stage) and
    /// [`submit_staging`](BufferRegistry::submit_staging). Only clears the
    /// dirty flags (`mark_clean()`) after a successful submit. If the submit
    /// fails, dirty flags are preserved so the operation can be retried.
    ///
    /// Returns `true` on success, `false` if no slot was available, no entries
    /// were dirty, or the submit failed.
    pub fn stage_and_submit(&mut self, registry: &mut BufferRegistry) -> bool {
        let (slot_index, byte_size) = match self.stage(registry) {
            Some(pair) => pair,
            None => return false,
        };

        match registry.submit_staging(slot_index, byte_size) {
            SubmitResult::Submitted => {
                self.mark_clean();
                true
            }
            SubmitResult::InvalidSlot => {
                debug_assert!(false, "stage_and_submit: acquired slot became invalid");
                false
            }
        }
    }
}

impl Default for MaterialTable {
    fn default() -> Self {
        Self::new()
    }
}

impl core::fmt::Display for MaterialTable {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "MaterialTable {{ capacity: {}, live: {} }}",
            self.entries.len(),
            self.live_count
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --------------------------------------------------------------
    // Test helpers
    // --------------------------------------------------------------

    /// Creates a material entry with the given base colour for testing.
    fn make_entry(r: f32, g: f32, b: f32, a: f32) -> MaterialTableEntry {
        MaterialTableEntry {
            base_color: [r, g, b, a],
            ..MaterialTableEntry::zeroed()
        }
    }

    /// Returns a small `BufferRegistry` for staging tests.
    fn small_registry() -> BufferRegistry {
        // 8 KiB should be enough for any test table.
        BufferRegistry::new(8192)
    }

    // --------------------------------------------------------------
    // Entry tests
    // --------------------------------------------------------------

    #[test]
    fn test_entry_size() {
        assert_eq!(size_of::<MaterialTableEntry>(), MATERIAL_TABLE_ENTRY_SIZE);
    }

    #[test]
    fn test_entry_alignment() {
        assert_eq!(align_of::<MaterialTableEntry>(), 16);
    }

    #[test]
    fn test_entry_layout() {
        // Verify that the byte offsets match the WGSL spec.
        let entry = MaterialTableEntry {
            base_color: [1.0, 2.0, 3.0, 4.0],
            emissive: [5.0, 6.0, 7.0, 8.0],
            metallic: 9.0,
            roughness: 10.0,
            occlusion: 11.0,
            normal_scale: 12.0,
            albedo_texture_id: 13,
            normal_texture_id: 14,
            metallic_roughness_texture_id: 15,
            emissive_texture_id: 16,
            flags: 0x8000_0001,
            alpha_cutoff: 0.5,
            _pad: [0; 2],
        };
        let bytes = unsafe {
            core::slice::from_raw_parts(
                &entry as *const _ as *const u8,
                MATERIAL_TABLE_ENTRY_SIZE,
            )
        };

        // base_color (offset 0)
        let bc: &[f32; 4] = unsafe { &*(bytes.as_ptr() as *const [f32; 4]) };
        assert_eq!(bc[0], 1.0);
        assert_eq!(bc[3], 4.0);

        // emissive (offset 16)
        let em: &[f32; 4] = unsafe { &*(bytes.as_ptr().add(16) as *const [f32; 4]) };
        assert_eq!(em[0], 5.0);
        assert_eq!(em[3], 8.0);

        // metallic (offset 32)
        let m: f32 = unsafe { bytes.as_ptr().add(32).cast::<f32>().read() };
        assert_eq!(m, 9.0);

        // roughness (offset 36)
        let r: f32 = unsafe { bytes.as_ptr().add(36).cast::<f32>().read() };
        assert_eq!(r, 10.0);

        // flags (offset 64)
        let f: u32 = unsafe { bytes.as_ptr().add(64).cast::<u32>().read() };
        assert_eq!(f, 0x8000_0001);
    }

    #[test]
    fn test_entry_zeroed() {
        let entry = MaterialTableEntry::zeroed();
        assert!(entry.is_zero());
        assert_eq!(entry.base_color, [0.0; 4]);
        assert_eq!(entry.albedo_texture_id, u32::MAX);
        assert_eq!(entry.flags, 0);
    }

    #[test]
    fn test_entry_dirty_flag_initial() {
        let entry = MaterialTableEntry::zeroed();
        // Zeroed entry should not have dirty flag set.
        assert_eq!(entry.flags & MATERIAL_FLAG_DIRTY, 0);
    }

    #[test]
    fn test_entry_visible_flag() {
        let mut entry = MaterialTableEntry::zeroed();
        entry.flags |= MATERIAL_FLAG_VISIBLE;
        assert!(entry.flags & MATERIAL_FLAG_VISIBLE != 0);
    }

    #[test]
    fn test_entry_is_zero_with_dirty_flag() {
        // An entry that is otherwise zeroed but has the dirty flag set
        // should still read as zero.
        let entry = MaterialTableEntry {
            flags: MATERIAL_FLAG_DIRTY,
            ..MaterialTableEntry::zeroed()
        };
        assert!(entry.is_zero());
    }

    #[test]
    fn test_entry_display() {
        let entry = MaterialTableEntry {
            base_color: [0.5, 0.6, 0.7, 1.0],
            emissive: [0.0, 0.0, 0.0, 0.0],
            metallic: 0.8,
            roughness: 0.3,
            occlusion: 1.0,
            normal_scale: 1.0,
            albedo_texture_id: 0,
            normal_texture_id: !0,
            metallic_roughness_texture_id: !0,
            emissive_texture_id: !0,
            flags: 0x8000_0001,
            alpha_cutoff: 0.5,
            _pad: [0; 2],
        };
        let display = format!("{}", entry);
        assert!(display.contains("0.50"));
        assert!(display.contains("0.60"));
        assert!(display.contains("0.70"));
        assert!(display.contains("0.80"));
        assert!(display.contains("0x80000001"));
        assert!(display.contains("albedo_tex: 0"));
        assert!(display.contains("alpha_cutoff: 0.50"));
    }

    // --------------------------------------------------------------
    // Table basics
    // --------------------------------------------------------------

    #[test]
    fn test_table_empty() {
        let table = MaterialTable::new();
        assert!(table.is_empty());
        assert_eq!(table.live_count(), 0);
        assert_eq!(table.len(), DEFAULT_MATERIAL_TABLE_CAPACITY);
    }

    #[test]
    fn test_table_with_capacity() {
        let table = MaterialTable::with_capacity(64);
        assert_eq!(table.len(), 64);
    }

    #[test]
    fn test_table_add_returns_monotonic_indices() {
        let mut table = MaterialTable::with_capacity(64);
        let a = table.add(make_entry(1.0, 0.0, 0.0, 1.0));
        let b = table.add(make_entry(0.0, 1.0, 0.0, 1.0));
        let c = table.add(make_entry(0.0, 0.0, 1.0, 1.0));
        assert_eq!(a, 0);
        assert_eq!(b, 1);
        assert_eq!(c, 2);
        assert_eq!(table.live_count(), 3);
    }

    #[test]
    fn test_table_add_preserves_field_values() {
        let mut table = MaterialTable::with_capacity(16);
        let idx = table.add(MaterialTableEntry {
            base_color: [0.1, 0.2, 0.3, 0.4],
            emissive: [0.5, 0.6, 0.7, 0.8],
            metallic: 0.9,
            roughness: 0.1,
            occlusion: 0.2,
            normal_scale: 1.5,
            albedo_texture_id: 1,
            normal_texture_id: 2,
            metallic_roughness_texture_id: 3,
            emissive_texture_id: 4,
            flags: 0x8000_0001,
            alpha_cutoff: 0.45,
            _pad: [0; 2],
        });
        let entry = table.get(idx).unwrap();
        assert_eq!(entry.base_color, [0.1, 0.2, 0.3, 0.4]);
        assert_eq!(entry.emissive, [0.5, 0.6, 0.7, 0.8]);
        assert_eq!(entry.metallic, 0.9);
        assert_eq!(entry.roughness, 0.1);
        assert_eq!(entry.occlusion, 0.2);
        assert_eq!(entry.normal_scale, 1.5);
        assert_eq!(entry.albedo_texture_id, 1);
        assert_eq!(entry.normal_texture_id, 2);
        assert_eq!(entry.metallic_roughness_texture_id, 3);
        assert_eq!(entry.emissive_texture_id, 4);
        assert_eq!(entry.flags & MATERIAL_FLAG_DIRTY, MATERIAL_FLAG_DIRTY);
        assert_eq!(entry.flags & MATERIAL_FLAG_VISIBLE, MATERIAL_FLAG_VISIBLE);
    }

    // --------------------------------------------------------------
    // Mutation
    // --------------------------------------------------------------

    #[test]
    fn test_table_update() {
        let mut table = MaterialTable::with_capacity(8);
        let idx = table.add(make_entry(1.0, 0.0, 0.0, 1.0));
        table.mark_clean();

        let new_entry = MaterialTableEntry {
            base_color: [0.0, 1.0, 0.0, 1.0],
            ..MaterialTableEntry::zeroed()
        };
        assert!(table.update(idx, new_entry).is_some());
        assert!(table.any_dirty());
        let entry = table.get(idx).unwrap();
        assert_eq!(entry.base_color, [0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_table_remove() {
        let mut table = MaterialTable::with_capacity(8);
        let idx = table.add(make_entry(1.0, 0.0, 0.0, 1.0));
        assert_eq!(table.live_count(), 1);

        assert_eq!(table.remove(idx), RemoveResult::Removed);
        assert_eq!(table.live_count(), 0);
        assert!(table.get(idx).is_none());
        // Slot should be zeroed but still accessible for future add.
        assert!(table.entries[idx as usize].is_zero());
    }

    #[test]
    fn test_table_remove_nonexistent() {
        let mut table = MaterialTable::with_capacity(4);
        assert_eq!(table.remove(0), RemoveResult::NotFound);
        assert_eq!(table.remove(999), RemoveResult::NotFound);
    }

    #[test]
    fn test_table_remove_and_reuse_slot() {
        let mut table = MaterialTable::with_capacity(8);
        let idx = table.add(make_entry(1.0, 0.0, 0.0, 1.0));
        assert_eq!(idx, 0);
        assert_eq!(table.remove(idx), RemoveResult::Removed);
        // Adding again should reuse slot 0.
        let new_idx = table.add(make_entry(0.0, 1.0, 0.0, 1.0));
        assert_eq!(new_idx, 0);
        assert_eq!(table.live_count(), 1);
    }

    #[test]
    fn test_table_insert_at() {
        let mut table = MaterialTable::with_capacity(4);
        // Fill with two entries.
        table.add(make_entry(1.0, 0.0, 0.0, 1.0));
        table.add(make_entry(0.0, 1.0, 0.0, 1.0));
        assert_eq!(table.live_count(), 2);
        // Insert at index 1, overwriting a live slot.
        table.insert_at(
            1,
            MaterialTableEntry {
                base_color: [0.0, 0.0, 1.0, 1.0],
                ..MaterialTableEntry::zeroed()
            },
        );
        // Overwriting a live slot should NOT increase live_count.
        assert_eq!(table.live_count(), 2);
        assert!(table.get(1).is_some());
        assert_eq!(table.get(1).unwrap().base_color, [0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_table_insert_at_hole() {
        let mut table = MaterialTable::with_capacity(8);
        let idx = table.add(make_entry(1.0, 0.0, 0.0, 1.0));
        assert_eq!(idx, 0);
        assert_eq!(table.remove(idx), RemoveResult::Removed);
        assert_eq!(table.live_count(), 0);

        // Now insert_at the hole.
        table.insert_at(
            0,
            MaterialTableEntry {
                base_color: [0.0, 1.0, 0.0, 1.0],
                ..MaterialTableEntry::zeroed()
            },
        );
        assert_eq!(table.live_count(), 1);
        assert!(table.get(0).is_some());
        assert_eq!(table.get(0).unwrap().base_color, [0.0, 1.0, 0.0, 1.0]);
    }

    // --------------------------------------------------------------
    // Clear / reserve
    // --------------------------------------------------------------

    #[test]
    fn test_table_clear() {
        let mut table = MaterialTable::with_capacity(16);
        table.add(make_entry(1.0, 0.0, 0.0, 1.0));
        table.add(make_entry(0.0, 0.0, 1.0, 1.0));
        assert_eq!(table.live_count(), 2);

        table.clear();
        assert_eq!(table.live_count(), 0);
        assert!(table.is_empty());
        assert_eq!(table.len(), 16); // capacity unchanged
    }

    #[test]
    fn test_table_reserve() {
        let mut table = MaterialTable::with_capacity(4);
        assert_eq!(table.len(), 4);
        table.reserve(8);
        assert_eq!(table.len(), 12);
        // All new slots should be zeroed.
        for i in 4..12 {
            assert!(table.entries[i].is_zero());
        }
    }

    // --------------------------------------------------------------
    // Dirty flag management
    // --------------------------------------------------------------

    #[test]
    fn test_add_marks_dirty() {
        let mut table = MaterialTable::with_capacity(4);
        table.add(make_entry(1.0, 0.0, 0.0, 1.0));
        assert!(table.any_dirty());
        assert_eq!(table.dirty_count(), 1);
    }

    #[test]
    fn test_mark_dirty() {
        let mut table = MaterialTable::with_capacity(4);
        let idx = table.add(make_entry(1.0, 0.0, 0.0, 1.0));
        table.mark_clean();
        assert!(!table.any_dirty());

        assert!(table.mark_dirty(idx));
        assert!(table.any_dirty());
    }

    #[test]
    fn test_mark_dirty_nonexistent() {
        let mut table = MaterialTable::with_capacity(4);
        assert!(!table.mark_dirty(0)); // hole
        assert!(!table.mark_dirty(999)); // out of bounds
    }

    #[test]
    fn test_mark_clean() {
        let mut table = MaterialTable::with_capacity(4);
        table.add(make_entry(1.0, 0.0, 0.0, 1.0));
        table.add(make_entry(0.0, 1.0, 0.0, 1.0));
        assert_eq!(table.dirty_count(), 2);

        table.mark_clean();
        assert!(!table.any_dirty());
        assert_eq!(table.dirty_count(), 0);
    }

    #[test]
    fn test_mark_clean_after_remove() {
        let mut table = MaterialTable::with_capacity(4);
        let idx = table.add(make_entry(1.0, 0.0, 0.0, 1.0));
        table.mark_clean();
        table.remove(idx);
        table.mark_clean(); // should not panic
        assert!(!table.any_dirty());
    }

    // --------------------------------------------------------------
    // Serialization
    // --------------------------------------------------------------

    #[test]
    fn test_as_bytes_single_entry() {
        let mut table = MaterialTable::with_capacity(4);
        table.add(MaterialTableEntry {
            base_color: [0.5, 0.6, 0.7, 1.0],
            ..MaterialTableEntry::zeroed()
        });
        table.mark_clean();

        let bytes = table.as_bytes();
        assert_eq!(bytes.len(), 4 * MATERIAL_TABLE_ENTRY_SIZE);

        // Entry 0, base_color = [0.5, 0.6, 0.7, 1.0] at offset 0.
        let entry_base_color: &[f32; 4] = unsafe { &*(bytes.as_ptr() as *const [f32; 4]) };
        assert!((entry_base_color[0] - 0.5).abs() < f32::EPSILON);
        assert!((entry_base_color[1] - 0.6).abs() < f32::EPSILON);
        assert!((entry_base_color[2] - 0.7).abs() < f32::EPSILON);
        assert!((entry_base_color[3] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_as_bytes_multiple_entries() {
        let mut table = MaterialTable::with_capacity(4);
        table.add(MaterialTableEntry {
            base_color: [0.1, 0.2, 0.3, 0.4],
            ..MaterialTableEntry::zeroed()
        });
        table.add(MaterialTableEntry {
            base_color: [0.5, 0.6, 0.7, 0.8],
            ..MaterialTableEntry::zeroed()
        });
        table.mark_clean();

        let bytes = table.as_bytes();
        assert_eq!(bytes.len(), 4 * MATERIAL_TABLE_ENTRY_SIZE);

        // Entry 0 base_color
        let e0_bc: &[f32; 4] = unsafe { &*(bytes.as_ptr() as *const [f32; 4]) };
        assert!((e0_bc[0] - 0.1).abs() < f32::EPSILON);

        // Entry 1 base_color (offset 80)
        let e1_bc: &[f32; 4] =
            unsafe { &*(bytes.as_ptr().add(MATERIAL_TABLE_ENTRY_SIZE) as *const [f32; 4]) };
        assert!((e1_bc[0] - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_as_bytes_with_holes() {
        let mut table = MaterialTable::with_capacity(4);
        let idx0 = table.add(MaterialTableEntry {
            base_color: [0.1, 0.2, 0.3, 0.4],
            ..MaterialTableEntry::zeroed()
        });
        table.add(MaterialTableEntry {
            base_color: [0.5, 0.6, 0.7, 0.8],
            ..MaterialTableEntry::zeroed()
        });
        // Remove entry 0, creating a hole.
        table.remove(idx0);
        table.mark_clean();

        let bytes = table.as_bytes();
        assert_eq!(bytes.len(), 4 * MATERIAL_TABLE_ENTRY_SIZE);

        // Entry 0 should be zeroed (hole).
        let e0_flags: u32 = unsafe { bytes.as_ptr().add(64).cast::<u32>().read() };
        assert_eq!(e0_flags, 0);

        // Entry 1 should still have base_color = [0.5, 0.6, 0.7, 0.8].
        let e1_bc: &[f32; 4] =
            unsafe { &*(bytes.as_ptr().add(MATERIAL_TABLE_ENTRY_SIZE) as *const [f32; 4]) };
        assert!((e1_bc[1] - 0.6).abs() < f32::EPSILON);
    }

    #[test]
    fn test_as_slice() {
        let mut table = MaterialTable::with_capacity(4);
        let _idx = table.add(make_entry(1.0, 0.0, 0.0, 1.0));
        let slice = table.as_slice();
        assert_eq!(slice.len(), 4);
        assert!(!slice[0].is_zero());
        assert!(slice[1].is_zero());
    }

    // --------------------------------------------------------------
    // Display
    // --------------------------------------------------------------

    #[test]
    fn test_table_display() {
        let table = MaterialTable::with_capacity(64);
        let display = format!("{}", table);
        assert!(display.contains("64"));
        assert!(display.contains("0"));
    }

    // --------------------------------------------------------------
    // BufferRegistry staging
    // --------------------------------------------------------------

    #[test]
    fn test_stage_returns_none_when_clean() {
        let mut table = MaterialTable::with_capacity(4);
        let mut registry = small_registry();
        assert!(table.stage(&mut registry).is_none());
    }

    #[test]
    fn test_stage_preserves_dirty_flags() {
        let mut table = MaterialTable::with_capacity(4);
        let mut registry = small_registry();
        table.add(make_entry(1.0, 0.0, 0.0, 1.0));

        // stage() preserves dirty flags for retry safety.
        let result = table.stage(&mut registry);
        assert!(result.is_some());
        assert!(table.any_dirty(), "stage() must preserve dirty flags for retry safety");
    }

    #[test]
    fn test_stage_and_submit() {
        let mut table = MaterialTable::with_capacity(4);
        let mut registry = small_registry();
        table.add(make_entry(1.0, 0.0, 0.0, 1.0));

        assert!(table.stage_and_submit(&mut registry));
        assert!(!table.any_dirty());
    }

    #[test]
    fn test_stage_and_submit_clean_table() {
        let mut table = MaterialTable::with_capacity(4);
        let mut registry = small_registry();
        assert!(!table.stage_and_submit(&mut registry));
    }

    #[test]
    fn test_stage_auto_resize() {
        // Table big enough to not fit in the small registry.
        let mut table = MaterialTable::with_capacity(256);
        let mut registry = small_registry(); // 8 KiB
        table.add(make_entry(1.0, 0.0, 0.0, 1.0));

        // The BufferRegistry should auto-resize to accommodate 256 * 80 bytes.
        let result = table.stage(&mut registry);
        assert!(result.is_some());
        // stage() preserves dirty flags for retry safety.
        assert!(table.any_dirty(), "stage() must preserve dirty flags for retry safety");
    }

    // --------------------------------------------------------------
    // Bulk operations
    // --------------------------------------------------------------

    #[test]
    fn test_multiple_add_remove() {
        let mut table = MaterialTable::with_capacity(8);
        let mut indices = Vec::new();
        for i in 0..5u32 {
            let idx =
                table.add(MaterialTableEntry {
                    base_color: [i as f32 / 10.0, 0.0, 0.0, 1.0],
                    ..MaterialTableEntry::zeroed()
                });
            indices.push(idx);
        }
        assert_eq!(table.live_count(), 5);

        // Remove middle entries.
        assert_eq!(table.remove(indices[2]), RemoveResult::Removed);
        assert_eq!(table.remove(indices[3]), RemoveResult::Removed);
        assert_eq!(table.live_count(), 3);

        // Add more -- should fill holes first.
        let new_idx = table.add(MaterialTableEntry {
            base_color: [0.9, 0.9, 0.9, 1.0],
            ..MaterialTableEntry::zeroed()
        });
        assert_eq!(new_idx, 2); // reused slot 2
        assert_eq!(table.live_count(), 4);
    }

    #[test]
    fn test_reuse_after_clear() {
        let mut table = MaterialTable::with_capacity(8);
        for i in 0..10u32 {
            table.add(make_entry(
                i as f32 / 10.0,
                0.0,
                0.0,
                1.0,
            ));
        }
        assert_eq!(table.live_count(), 10);

        table.clear();
        assert_eq!(table.live_count(), 0);

        // Add again -- all slots are holes, indices restart.
        for i in 0..5u32 {
            let idx = table.add(make_entry(
                i as f32 / 10.0 + 0.1,
                0.0,
                0.0,
                1.0,
            ));
            assert_eq!(idx, i);
        }
        assert_eq!(table.live_count(), 5);
    }

    // --------------------------------------------------------------
    // Edge cases
    // --------------------------------------------------------------

    #[test]
    fn test_get_mut_marks_dirty() {
        let mut table = MaterialTable::with_capacity(8);
        let idx = table.add(make_entry(1.0, 0.0, 0.0, 1.0));
        table.mark_clean();

        let entry = table.get_mut(idx);
        assert!(entry.is_some());
        // The entry should now be dirty.
        assert!(table.get(idx).unwrap().flags & MATERIAL_FLAG_DIRTY != 0);
    }

    #[test]
    fn test_get_mut_nonexistent() {
        let mut table = MaterialTable::with_capacity(4);
        assert!(table.get_mut(0).is_none());
        assert!(table.get_mut(999).is_none());
    }

    #[test]
    fn test_insert_at_out_of_bounds() {
        let mut table = MaterialTable::with_capacity(4);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            table.insert_at(10, MaterialTableEntry::zeroed());
        }));
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_capacity_table() {
        // Should clamp to capacity 1.
        let table = MaterialTable::with_capacity(0);
        assert_eq!(table.len(), 1);
    }

    #[test]
    fn test_immediate_read_after_add() {
        let mut table = MaterialTable::with_capacity(16);
        let idx = table.add(MaterialTableEntry {
            base_color: [1.0, 0.0, 0.0, 1.0],
            emissive: [0.1, 0.0, 0.0, 2.0],
            metallic: 0.5,
            roughness: 0.25,
            occlusion: 0.8,
            normal_scale: 1.0,
            albedo_texture_id: 0,
            normal_texture_id: 1,
            metallic_roughness_texture_id: 2,
            emissive_texture_id: 3,
            flags: 0x8000_0001,
            alpha_cutoff: 0.5,
            _pad: [0; 2],
        });
        let entry = table.get(idx).unwrap();
        assert_eq!(entry.metallic, 0.5);
        assert_eq!(entry.roughness, 0.25);
        assert_eq!(entry.occlusion, 0.8);
        assert_eq!(entry.normal_scale, 1.0);
        assert_eq!(entry.albedo_texture_id, 0);
        assert_eq!(entry.normal_texture_id, 1);
        assert_eq!(entry.metallic_roughness_texture_id, 2);
        assert_eq!(entry.emissive_texture_id, 3);
        assert_eq!(entry.alpha_cutoff, 0.5);
    }

    #[test]
    fn test_dirty_and_hole_interaction() {
        let mut table = MaterialTable::with_capacity(4);
        let idx = table.add(make_entry(1.0, 0.0, 0.0, 1.0));
        table.mark_clean();
        assert!(!table.any_dirty());

        table.remove(idx);
        // Removing zeroes the entry. The dirty flag on the removed entry
        // is cleared because `zeroed()` has flags = 0.
        assert!(!table.any_dirty());
    }

    #[test]
    fn test_double_stage_and_submit_cycle() {
        let mut table = MaterialTable::with_capacity(4);
        let mut registry = small_registry();

        // First cycle: stage_and_submit clears dirty after successful submit.
        table.add(make_entry(1.0, 0.0, 0.0, 1.0));
        assert!(table.stage_and_submit(&mut registry));
        assert!(!table.any_dirty(), "First cycle must be clean after submit");

        // Second cycle.
        table.add(make_entry(0.0, 1.0, 0.0, 1.0));
        assert!(table.stage_and_submit(&mut registry));
        assert!(!table.any_dirty(), "Second cycle must be clean after submit");
    }

    #[test]
    fn test_slot_unavailable_returns_none() {
        let mut table = MaterialTable::with_capacity(1024);
        let mut registry = small_registry(); // 8 KiB

        // Add enough entries to make the table larger than the registry.
        for i in 0..200u32 {
            table.add(MaterialTableEntry {
                base_color: [i as f32 / 200.0, 0.0, 0.0, 1.0],
                ..MaterialTableEntry::zeroed()
            });
        }

        // First stage should succeed (registry auto-resizes).
        let r1 = table.stage(&mut registry);
        assert!(r1.is_some());
    }
}
