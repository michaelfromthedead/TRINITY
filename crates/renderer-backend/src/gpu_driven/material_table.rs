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
//! # Layout (MaterialTableEntry - 80 bytes)
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
//! # Layout (MaterialDescriptor - 64 bytes, T-WGPU-P6.8.4)
//!
//! A compact GPU-compatible material descriptor using bytemuck for safe
//! GPU buffer transfers. Used for bindless material tables in GPU-driven
//! rendering pipelines.
//!
//! | Offset | Size | Field | Description |
//! |--------|------|-------|-------------|
//! | 0      | 4    | `base_color_texture` | Bindless texture index (u32::MAX = none) |
//! | 4      | 4    | `normal_texture` | Bindless texture index |
//! | 8      | 4    | `metallic_roughness_texture` | Bindless texture index |
//! | 12     | 4    | `emissive_texture` | Bindless texture index |
//! | 16     | 16   | `base_color_factor` | RGBA base color factor |
//! | 32     | 4    | `metallic_factor` | Metallic factor (0-1) |
//! | 36     | 4    | `roughness_factor` | Roughness factor (0-1) |
//! | 40     | 12   | `emissive_factor` | RGB emissive factor |
//! | 52     | 4    | `alpha_cutoff` | Alpha cutoff for masked materials |
//! | 56     | 4    | `flags` | Material flags (double-sided, alpha mode) |
//! | 60     | 4    | `_pad` | Padding for 64-byte alignment |
//!
//! **Total: 64 bytes.** Array stride: 64.
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

use bytemuck::{Pod, Zeroable};
use wgpu::{Buffer, BufferDescriptor, BufferUsages, Device, Queue};

use crate::gpu_driven::buffers::{
    AcquireResult, BufferRegistry, SubmitResult,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Size of a single `MaterialTableEntry` in bytes (80).
pub const MATERIAL_TABLE_ENTRY_SIZE: usize = 80;

/// Size of a single `MaterialDescriptor` in bytes (64).
pub const MATERIAL_DESCRIPTOR_SIZE: usize = 64;

/// Default material table capacity (1024 entries = 80 KiB for MaterialTableEntry).
pub const DEFAULT_MATERIAL_TABLE_CAPACITY: usize = 1024;

/// Default GPU material table capacity (1024 entries = 64 KiB for MaterialDescriptor).
pub const DEFAULT_GPU_MATERIAL_TABLE_CAPACITY: u32 = 1024;

/// Flag bit indicating the material entry has been modified and needs staging.
pub const MATERIAL_FLAG_DIRTY: u32 = 0x8000_0000;

/// Flag bit indicating the material is visible (included in draw calls).
pub const MATERIAL_FLAG_VISIBLE: u32 = 0x0000_0001;

// ---------------------------------------------------------------------------
// MaterialDescriptor Flags (T-WGPU-P6.8.4)
// ---------------------------------------------------------------------------

/// Flag: Material is double-sided (no backface culling).
pub const MATERIAL_DESC_FLAG_DOUBLE_SIDED: u32 = 1 << 0;

/// Flag: Material uses alpha mask mode.
pub const MATERIAL_DESC_FLAG_ALPHA_MASK: u32 = 1 << 1;

/// Flag: Material uses alpha blend mode.
pub const MATERIAL_DESC_FLAG_ALPHA_BLEND: u32 = 1 << 2;

/// Flag: Material has unlit shading (no PBR).
pub const MATERIAL_DESC_FLAG_UNLIT: u32 = 1 << 3;

/// Sentinel value indicating no texture is bound.
pub const NO_TEXTURE: u32 = u32::MAX;

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

// ---------------------------------------------------------------------------
// MaterialDescriptor (T-WGPU-P6.8.4)
// ---------------------------------------------------------------------------

/// GPU-compatible material descriptor (64 bytes) for bindless rendering.
///
/// This struct is designed for direct GPU buffer transfer using bytemuck's
/// `Pod` and `Zeroable` traits. It stores texture indices into the bindless
/// texture registry and material parameters for PBR shading.
///
/// # WGSL Usage
///
/// ```wgsl
/// struct MaterialDescriptor {
///     base_color_texture: u32,
///     normal_texture: u32,
///     metallic_roughness_texture: u32,
///     emissive_texture: u32,
///     base_color_factor: vec4<f32>,
///     metallic_factor: f32,
///     roughness_factor: f32,
///     emissive_factor: vec3<f32>,
///     alpha_cutoff: f32,
///     flags: u32,
///     _pad: u32,
/// }
///
/// @group(0) @binding(0) var<storage, read> materials: array<MaterialDescriptor>;
/// ```
///
/// # Layout
///
/// The struct is 64 bytes with natural alignment. Texture indices use
/// `u32::MAX` as a sentinel for "no texture bound".
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct MaterialDescriptor {
    /// Base color texture index into bindless texture registry.
    /// Use `NO_TEXTURE` (u32::MAX) if no texture is bound.
    pub base_color_texture: u32,

    /// Normal map texture index into bindless texture registry.
    /// Use `NO_TEXTURE` (u32::MAX) if no texture is bound.
    pub normal_texture: u32,

    /// Metallic-roughness texture index into bindless texture registry.
    /// Red channel = metallic, green channel = roughness.
    /// Use `NO_TEXTURE` (u32::MAX) if no texture is bound.
    pub metallic_roughness_texture: u32,

    /// Emissive texture index into bindless texture registry.
    /// Use `NO_TEXTURE` (u32::MAX) if no texture is bound.
    pub emissive_texture: u32,

    /// Base color factor (RGBA, linear space).
    /// Multiplied with base color texture if present.
    pub base_color_factor: [f32; 4],

    /// Metallic factor (0.0 = dielectric, 1.0 = metal).
    /// Multiplied with metallic texture if present.
    pub metallic_factor: f32,

    /// Roughness factor (0.0 = smooth, 1.0 = rough).
    /// Multiplied with roughness texture if present.
    pub roughness_factor: f32,

    /// Emissive factor (RGB, linear space).
    /// Multiplied with emissive texture if present.
    pub emissive_factor: [f32; 3],

    /// Alpha cutoff threshold for alpha-mask materials.
    /// Fragments with alpha below this are discarded.
    pub alpha_cutoff: f32,

    /// Material flags (double-sided, alpha mode, etc.).
    /// See `MATERIAL_DESC_FLAG_*` constants.
    pub flags: u32,

    /// Padding for 64-byte alignment.
    pub _pad: u32,
}

impl MaterialDescriptor {
    /// Sentinel value for "no texture bound" (convenience re-export).
    pub const NO_TEXTURE: u32 = NO_TEXTURE;

    /// Creates a new material descriptor with default PBR values.
    ///
    /// - White base color
    /// - No textures bound
    /// - Metallic = 0 (dielectric)
    /// - Roughness = 0.5 (medium rough)
    /// - No emission
    /// - Alpha cutoff = 0.5
    /// - No flags set
    pub const fn new() -> Self {
        Self {
            base_color_texture: NO_TEXTURE,
            normal_texture: NO_TEXTURE,
            metallic_roughness_texture: NO_TEXTURE,
            emissive_texture: NO_TEXTURE,
            base_color_factor: [1.0, 1.0, 1.0, 1.0],
            metallic_factor: 0.0,
            roughness_factor: 0.5,
            emissive_factor: [0.0, 0.0, 0.0],
            alpha_cutoff: 0.5,
            flags: 0,
            _pad: 0,
        }
    }

    /// Creates a simple opaque material with the given base color.
    pub const fn opaque(r: f32, g: f32, b: f32) -> Self {
        Self {
            base_color_texture: NO_TEXTURE,
            normal_texture: NO_TEXTURE,
            metallic_roughness_texture: NO_TEXTURE,
            emissive_texture: NO_TEXTURE,
            base_color_factor: [r, g, b, 1.0],
            metallic_factor: 0.0,
            roughness_factor: 0.5,
            emissive_factor: [0.0, 0.0, 0.0],
            alpha_cutoff: 0.5,
            flags: 0,
            _pad: 0,
        }
    }

    /// Creates a metallic material with the given base color.
    pub const fn metallic(r: f32, g: f32, b: f32, metallic: f32, roughness: f32) -> Self {
        Self {
            base_color_texture: NO_TEXTURE,
            normal_texture: NO_TEXTURE,
            metallic_roughness_texture: NO_TEXTURE,
            emissive_texture: NO_TEXTURE,
            base_color_factor: [r, g, b, 1.0],
            metallic_factor: metallic,
            roughness_factor: roughness,
            emissive_factor: [0.0, 0.0, 0.0],
            alpha_cutoff: 0.5,
            flags: 0,
            _pad: 0,
        }
    }

    /// Sets the base color texture index.
    #[inline]
    pub const fn with_base_color_texture(mut self, index: u32) -> Self {
        self.base_color_texture = index;
        self
    }

    /// Sets the normal texture index.
    #[inline]
    pub const fn with_normal_texture(mut self, index: u32) -> Self {
        self.normal_texture = index;
        self
    }

    /// Sets the metallic-roughness texture index.
    #[inline]
    pub const fn with_metallic_roughness_texture(mut self, index: u32) -> Self {
        self.metallic_roughness_texture = index;
        self
    }

    /// Sets the emissive texture index.
    #[inline]
    pub const fn with_emissive_texture(mut self, index: u32) -> Self {
        self.emissive_texture = index;
        self
    }

    /// Sets the double-sided flag.
    #[inline]
    pub const fn with_double_sided(mut self, enabled: bool) -> Self {
        if enabled {
            self.flags |= MATERIAL_DESC_FLAG_DOUBLE_SIDED;
        } else {
            self.flags &= !MATERIAL_DESC_FLAG_DOUBLE_SIDED;
        }
        self
    }

    /// Sets the alpha mask mode.
    #[inline]
    pub const fn with_alpha_mask(mut self, cutoff: f32) -> Self {
        self.flags |= MATERIAL_DESC_FLAG_ALPHA_MASK;
        self.flags &= !MATERIAL_DESC_FLAG_ALPHA_BLEND;
        self.alpha_cutoff = cutoff;
        self
    }

    /// Sets the alpha blend mode.
    #[inline]
    pub const fn with_alpha_blend(mut self) -> Self {
        self.flags |= MATERIAL_DESC_FLAG_ALPHA_BLEND;
        self.flags &= !MATERIAL_DESC_FLAG_ALPHA_MASK;
        self
    }

    /// Returns true if the material is double-sided.
    #[inline]
    pub const fn is_double_sided(&self) -> bool {
        (self.flags & MATERIAL_DESC_FLAG_DOUBLE_SIDED) != 0
    }

    /// Returns true if the material uses alpha masking.
    #[inline]
    pub const fn is_alpha_mask(&self) -> bool {
        (self.flags & MATERIAL_DESC_FLAG_ALPHA_MASK) != 0
    }

    /// Returns true if the material uses alpha blending.
    #[inline]
    pub const fn is_alpha_blend(&self) -> bool {
        (self.flags & MATERIAL_DESC_FLAG_ALPHA_BLEND) != 0
    }

    /// Returns true if the material has a base color texture.
    #[inline]
    pub const fn has_base_color_texture(&self) -> bool {
        self.base_color_texture != NO_TEXTURE
    }

    /// Returns true if the material has a normal texture.
    #[inline]
    pub const fn has_normal_texture(&self) -> bool {
        self.normal_texture != NO_TEXTURE
    }

    /// Returns true if the material has a metallic-roughness texture.
    #[inline]
    pub const fn has_metallic_roughness_texture(&self) -> bool {
        self.metallic_roughness_texture != NO_TEXTURE
    }

    /// Returns true if the material has an emissive texture.
    #[inline]
    pub const fn has_emissive_texture(&self) -> bool {
        self.emissive_texture != NO_TEXTURE
    }
}

// ---------------------------------------------------------------------------
// GpuMaterialTable (T-WGPU-P6.8.4)
// ---------------------------------------------------------------------------

/// Bindless material table for GPU-driven rendering.
///
/// Manages a collection of `MaterialDescriptor` entries with efficient dirty
/// tracking and GPU buffer uploads. Materials are indexed by u32 handles that
/// can be passed directly to shaders.
///
/// # Usage
///
/// ```ignore
/// use renderer_backend::gpu_driven::material_table::{GpuMaterialTable, MaterialDescriptor};
///
/// // Create table with capacity for 256 materials
/// let mut table = GpuMaterialTable::new(256);
///
/// // Add materials
/// let metal_idx = table.add(MaterialDescriptor::metallic(0.8, 0.8, 0.8, 0.9, 0.2));
/// let plastic_idx = table.add(MaterialDescriptor::opaque(0.2, 0.4, 0.8));
///
/// // Upload to GPU (creates buffer if needed)
/// table.upload(&device, &queue);
///
/// // Use in render pass
/// if let Some(buffer) = table.buffer() {
///     render_pass.set_bind_group(0, material_bind_group, &[]);
/// }
/// ```
///
/// # Memory Layout
///
/// Materials are stored in a contiguous array for efficient GPU access.
/// The GPU buffer is created lazily on first `upload()` call and resized
/// as needed when materials are added.
pub struct GpuMaterialTable {
    /// Material descriptors (CPU-side).
    materials: Vec<MaterialDescriptor>,

    /// GPU buffer containing material data.
    buffer: Option<Buffer>,

    /// Dirty flag indicating GPU buffer needs update.
    dirty: bool,

    /// Maximum capacity (used for pre-allocation).
    capacity: u32,

    /// Free list for recycled material indices.
    free_indices: Vec<u32>,
}

impl GpuMaterialTable {
    /// Creates a new material table with the given capacity.
    ///
    /// The capacity determines the initial GPU buffer size. The table will
    /// automatically grow if more materials are added.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Initial capacity (number of materials).
    pub fn new(capacity: u32) -> Self {
        let cap = capacity.max(1) as usize;
        Self {
            materials: Vec::with_capacity(cap),
            buffer: None,
            dirty: false,
            capacity: capacity.max(1),
            free_indices: Vec::new(),
        }
    }

    /// Creates a material table with default capacity (1024 materials).
    pub fn with_default_capacity() -> Self {
        Self::new(DEFAULT_GPU_MATERIAL_TABLE_CAPACITY)
    }

    /// Adds a material and returns its index.
    ///
    /// If there are recycled indices available, one will be reused.
    /// Otherwise, a new index is allocated.
    ///
    /// # Arguments
    ///
    /// * `material` - The material descriptor to add.
    ///
    /// # Returns
    ///
    /// The material index (handle) for shader access.
    pub fn add(&mut self, material: MaterialDescriptor) -> u32 {
        self.dirty = true;

        // Reuse a recycled index if available
        if let Some(index) = self.free_indices.pop() {
            self.materials[index as usize] = material;
            return index;
        }

        // Allocate new index
        let index = self.materials.len() as u32;
        self.materials.push(material);
        index
    }

    /// Gets a material by index.
    ///
    /// # Arguments
    ///
    /// * `index` - The material index.
    ///
    /// # Returns
    ///
    /// A reference to the material, or `None` if the index is invalid.
    #[inline]
    pub fn get(&self, index: u32) -> Option<&MaterialDescriptor> {
        self.materials.get(index as usize)
    }

    /// Gets a mutable reference to a material by index.
    ///
    /// Marks the table as dirty for GPU upload.
    ///
    /// # Arguments
    ///
    /// * `index` - The material index.
    ///
    /// # Returns
    ///
    /// A mutable reference to the material, or `None` if the index is invalid.
    #[inline]
    pub fn get_mut(&mut self, index: u32) -> Option<&mut MaterialDescriptor> {
        if (index as usize) < self.materials.len() {
            self.dirty = true;
            self.materials.get_mut(index as usize)
        } else {
            None
        }
    }

    /// Updates a material at the given index.
    ///
    /// # Arguments
    ///
    /// * `index` - The material index.
    /// * `material` - The new material descriptor.
    ///
    /// # Returns
    ///
    /// `true` if the update succeeded, `false` if the index is invalid.
    pub fn update(&mut self, index: u32, material: MaterialDescriptor) -> bool {
        if (index as usize) >= self.materials.len() {
            return false;
        }
        self.materials[index as usize] = material;
        self.dirty = true;
        true
    }

    /// Removes a material at the given index.
    ///
    /// The index is added to the free list for reuse. The material data is
    /// zeroed to prevent stale shader reads.
    ///
    /// # Arguments
    ///
    /// * `index` - The material index to remove.
    ///
    /// # Returns
    ///
    /// `true` if the removal succeeded, `false` if the index is invalid.
    pub fn remove(&mut self, index: u32) -> bool {
        if (index as usize) >= self.materials.len() {
            return false;
        }

        // Zero out the material
        self.materials[index as usize] = MaterialDescriptor::zeroed();
        self.free_indices.push(index);
        self.dirty = true;
        true
    }

    /// Uploads dirty materials to the GPU buffer.
    ///
    /// Creates the buffer if it doesn't exist. Resizes the buffer if the
    /// material count exceeds the current capacity.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `queue` - The wgpu queue.
    pub fn upload(&mut self, device: &Device, queue: &Queue) {
        if self.materials.is_empty() {
            return;
        }

        let required_size = (self.materials.len() * MATERIAL_DESCRIPTOR_SIZE) as u64;

        // Create or resize buffer if needed
        let needs_new_buffer = match &self.buffer {
            None => true,
            Some(buf) => buf.size() < required_size,
        };

        if needs_new_buffer {
            // Round up to next power of 2 for growth efficiency
            let new_capacity = required_size.next_power_of_two().max(required_size);

            self.buffer = Some(device.create_buffer(&BufferDescriptor {
                label: Some("gpu_material_table"),
                size: new_capacity,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));

            self.dirty = true; // Force full upload after resize
        }

        // Upload if dirty
        if self.dirty {
            if let Some(buffer) = &self.buffer {
                let bytes = bytemuck::cast_slice(&self.materials);
                queue.write_buffer(buffer, 0, bytes);
                self.dirty = false;
            }
        }
    }

    /// Returns the GPU buffer, if it exists.
    ///
    /// The buffer is created on the first `upload()` call.
    #[inline]
    pub fn buffer(&self) -> Option<&Buffer> {
        self.buffer.as_ref()
    }

    /// Returns true if the table has pending changes for GPU upload.
    #[inline]
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Returns the number of materials in the table.
    #[inline]
    pub fn len(&self) -> u32 {
        self.materials.len() as u32
    }

    /// Returns true if the table is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.materials.is_empty()
    }

    /// Returns the current capacity.
    #[inline]
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Returns the number of active materials (excluding recycled slots).
    #[inline]
    pub fn active_count(&self) -> u32 {
        (self.materials.len() - self.free_indices.len()) as u32
    }

    /// Returns the number of recycled (free) slots.
    #[inline]
    pub fn free_count(&self) -> u32 {
        self.free_indices.len() as u32
    }

    /// Returns the raw material data as a byte slice.
    ///
    /// Useful for custom buffer uploads or debugging.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.materials)
    }

    /// Returns a slice of all materials.
    #[inline]
    pub fn as_slice(&self) -> &[MaterialDescriptor] {
        &self.materials
    }

    /// Clears all materials from the table.
    ///
    /// The GPU buffer is not deallocated, but will be fully zeroed on next upload.
    pub fn clear(&mut self) {
        self.materials.clear();
        self.free_indices.clear();
        self.dirty = true;
    }

    /// Marks the table as dirty, forcing a GPU upload on next `upload()` call.
    #[inline]
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }
}

impl std::fmt::Debug for GpuMaterialTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuMaterialTable")
            .field("material_count", &self.materials.len())
            .field("free_count", &self.free_indices.len())
            .field("capacity", &self.capacity)
            .field("dirty", &self.dirty)
            .field("has_buffer", &self.buffer.is_some())
            .finish()
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

    // --------------------------------------------------------------
    // MaterialDescriptor tests (T-WGPU-P6.8.4)
    // --------------------------------------------------------------

    #[test]
    fn test_material_descriptor_size() {
        assert_eq!(size_of::<MaterialDescriptor>(), MATERIAL_DESCRIPTOR_SIZE);
        assert_eq!(size_of::<MaterialDescriptor>(), 64);
    }

    #[test]
    fn test_material_descriptor_alignment() {
        // Should be 4-byte aligned for u32/f32 fields
        assert_eq!(align_of::<MaterialDescriptor>(), 4);
    }

    #[test]
    fn test_material_descriptor_is_pod() {
        // Verify bytemuck traits are implemented
        fn assert_pod<T: Pod>() {}
        fn assert_zeroable<T: Zeroable>() {}
        assert_pod::<MaterialDescriptor>();
        assert_zeroable::<MaterialDescriptor>();
    }

    #[test]
    fn test_material_descriptor_new() {
        let mat = MaterialDescriptor::new();
        assert_eq!(mat.base_color_texture, NO_TEXTURE);
        assert_eq!(mat.normal_texture, NO_TEXTURE);
        assert_eq!(mat.metallic_roughness_texture, NO_TEXTURE);
        assert_eq!(mat.emissive_texture, NO_TEXTURE);
        assert_eq!(mat.base_color_factor, [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(mat.metallic_factor, 0.0);
        assert_eq!(mat.roughness_factor, 0.5);
        assert_eq!(mat.emissive_factor, [0.0, 0.0, 0.0]);
        assert_eq!(mat.alpha_cutoff, 0.5);
        assert_eq!(mat.flags, 0);
    }

    #[test]
    fn test_material_descriptor_opaque() {
        let mat = MaterialDescriptor::opaque(0.5, 0.6, 0.7);
        assert_eq!(mat.base_color_factor, [0.5, 0.6, 0.7, 1.0]);
        assert_eq!(mat.metallic_factor, 0.0);
        assert_eq!(mat.roughness_factor, 0.5);
    }

    #[test]
    fn test_material_descriptor_metallic() {
        let mat = MaterialDescriptor::metallic(0.8, 0.8, 0.8, 0.9, 0.2);
        assert_eq!(mat.base_color_factor, [0.8, 0.8, 0.8, 1.0]);
        assert_eq!(mat.metallic_factor, 0.9);
        assert_eq!(mat.roughness_factor, 0.2);
    }

    #[test]
    fn test_material_descriptor_with_textures() {
        let mat = MaterialDescriptor::new()
            .with_base_color_texture(0)
            .with_normal_texture(1)
            .with_metallic_roughness_texture(2)
            .with_emissive_texture(3);

        assert_eq!(mat.base_color_texture, 0);
        assert_eq!(mat.normal_texture, 1);
        assert_eq!(mat.metallic_roughness_texture, 2);
        assert_eq!(mat.emissive_texture, 3);
        assert!(mat.has_base_color_texture());
        assert!(mat.has_normal_texture());
        assert!(mat.has_metallic_roughness_texture());
        assert!(mat.has_emissive_texture());
    }

    #[test]
    fn test_material_descriptor_no_textures() {
        let mat = MaterialDescriptor::new();
        assert!(!mat.has_base_color_texture());
        assert!(!mat.has_normal_texture());
        assert!(!mat.has_metallic_roughness_texture());
        assert!(!mat.has_emissive_texture());
    }

    #[test]
    fn test_material_descriptor_double_sided() {
        let mat = MaterialDescriptor::new().with_double_sided(true);
        assert!(mat.is_double_sided());
        assert_eq!(mat.flags & MATERIAL_DESC_FLAG_DOUBLE_SIDED, MATERIAL_DESC_FLAG_DOUBLE_SIDED);

        let mat2 = mat.with_double_sided(false);
        assert!(!mat2.is_double_sided());
    }

    #[test]
    fn test_material_descriptor_alpha_mask() {
        let mat = MaterialDescriptor::new().with_alpha_mask(0.75);
        assert!(mat.is_alpha_mask());
        assert!(!mat.is_alpha_blend());
        assert_eq!(mat.alpha_cutoff, 0.75);
        assert_eq!(mat.flags & MATERIAL_DESC_FLAG_ALPHA_MASK, MATERIAL_DESC_FLAG_ALPHA_MASK);
    }

    #[test]
    fn test_material_descriptor_alpha_blend() {
        let mat = MaterialDescriptor::new().with_alpha_blend();
        assert!(mat.is_alpha_blend());
        assert!(!mat.is_alpha_mask());
        assert_eq!(mat.flags & MATERIAL_DESC_FLAG_ALPHA_BLEND, MATERIAL_DESC_FLAG_ALPHA_BLEND);
    }

    #[test]
    fn test_material_descriptor_alpha_modes_exclusive() {
        // Setting alpha blend should clear alpha mask
        let mat = MaterialDescriptor::new()
            .with_alpha_mask(0.5)
            .with_alpha_blend();
        assert!(mat.is_alpha_blend());
        assert!(!mat.is_alpha_mask());

        // Setting alpha mask should clear alpha blend
        let mat2 = mat.with_alpha_mask(0.3);
        assert!(mat2.is_alpha_mask());
        assert!(!mat2.is_alpha_blend());
    }

    #[test]
    fn test_material_descriptor_bytemuck_cast() {
        let mat = MaterialDescriptor::metallic(0.5, 0.6, 0.7, 0.8, 0.2)
            .with_base_color_texture(42);

        let bytes: &[u8] = bytemuck::bytes_of(&mat);
        assert_eq!(bytes.len(), MATERIAL_DESCRIPTOR_SIZE);

        // Verify first field (base_color_texture = 42)
        let first_u32: u32 = u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(first_u32, 42);
    }

    #[test]
    fn test_material_descriptor_slice_cast() {
        let materials = vec![
            MaterialDescriptor::opaque(1.0, 0.0, 0.0),
            MaterialDescriptor::opaque(0.0, 1.0, 0.0),
            MaterialDescriptor::opaque(0.0, 0.0, 1.0),
        ];

        let bytes: &[u8] = bytemuck::cast_slice(&materials);
        assert_eq!(bytes.len(), 3 * MATERIAL_DESCRIPTOR_SIZE);
    }

    #[test]
    fn test_material_descriptor_zeroed() {
        let mat = MaterialDescriptor::zeroed();
        assert_eq!(mat.base_color_texture, 0);
        assert_eq!(mat.base_color_factor, [0.0, 0.0, 0.0, 0.0]);
        assert_eq!(mat.metallic_factor, 0.0);
        assert_eq!(mat.roughness_factor, 0.0);
        assert_eq!(mat.flags, 0);
    }

    #[test]
    fn test_material_descriptor_default() {
        let mat = MaterialDescriptor::default();
        let zeroed = MaterialDescriptor::zeroed();
        assert_eq!(mat, zeroed);
    }

    #[test]
    fn test_material_descriptor_debug() {
        let mat = MaterialDescriptor::opaque(0.5, 0.5, 0.5);
        let debug = format!("{:?}", mat);
        assert!(debug.contains("MaterialDescriptor"));
        assert!(debug.contains("base_color_factor"));
    }

    #[test]
    fn test_material_descriptor_layout() {
        // Verify byte offsets match documentation
        let mat = MaterialDescriptor {
            base_color_texture: 1,
            normal_texture: 2,
            metallic_roughness_texture: 3,
            emissive_texture: 4,
            base_color_factor: [0.1, 0.2, 0.3, 0.4],
            metallic_factor: 0.5,
            roughness_factor: 0.6,
            emissive_factor: [0.7, 0.8, 0.9],
            alpha_cutoff: 0.25,
            flags: 0x0000_0003,
            _pad: 0,
        };

        let bytes: &[u8] = bytemuck::bytes_of(&mat);

        // base_color_texture at offset 0
        let tex0: u32 = u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(tex0, 1);

        // normal_texture at offset 4
        let tex1: u32 = u32::from_ne_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        assert_eq!(tex1, 2);

        // metallic_roughness_texture at offset 8
        let tex2: u32 = u32::from_ne_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        assert_eq!(tex2, 3);

        // emissive_texture at offset 12
        let tex3: u32 = u32::from_ne_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        assert_eq!(tex3, 4);

        // base_color_factor at offset 16
        let bc0: f32 = f32::from_ne_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
        assert!((bc0 - 0.1).abs() < 0.001);

        // metallic_factor at offset 32
        let mf: f32 = f32::from_ne_bytes([bytes[32], bytes[33], bytes[34], bytes[35]]);
        assert!((mf - 0.5).abs() < 0.001);

        // roughness_factor at offset 36
        let rf: f32 = f32::from_ne_bytes([bytes[36], bytes[37], bytes[38], bytes[39]]);
        assert!((rf - 0.6).abs() < 0.001);

        // flags at offset 56
        let flags: u32 = u32::from_ne_bytes([bytes[56], bytes[57], bytes[58], bytes[59]]);
        assert_eq!(flags, 0x0000_0003);
    }

    // --------------------------------------------------------------
    // GpuMaterialTable tests (T-WGPU-P6.8.4)
    // --------------------------------------------------------------

    #[test]
    fn test_gpu_material_table_new() {
        let table = GpuMaterialTable::new(128);
        assert!(table.is_empty());
        assert_eq!(table.len(), 0);
        assert_eq!(table.capacity(), 128);
        assert!(!table.is_dirty());
        assert!(table.buffer().is_none());
    }

    #[test]
    fn test_gpu_material_table_with_default_capacity() {
        let table = GpuMaterialTable::with_default_capacity();
        assert_eq!(table.capacity(), DEFAULT_GPU_MATERIAL_TABLE_CAPACITY);
    }

    #[test]
    fn test_gpu_material_table_add() {
        let mut table = GpuMaterialTable::new(64);

        let idx0 = table.add(MaterialDescriptor::opaque(1.0, 0.0, 0.0));
        let idx1 = table.add(MaterialDescriptor::opaque(0.0, 1.0, 0.0));
        let idx2 = table.add(MaterialDescriptor::opaque(0.0, 0.0, 1.0));

        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
        assert_eq!(idx2, 2);
        assert_eq!(table.len(), 3);
        assert!(table.is_dirty());
    }

    #[test]
    fn test_gpu_material_table_get() {
        let mut table = GpuMaterialTable::new(64);
        let idx = table.add(MaterialDescriptor::metallic(0.5, 0.5, 0.5, 0.8, 0.3));

        let mat = table.get(idx).unwrap();
        assert_eq!(mat.base_color_factor, [0.5, 0.5, 0.5, 1.0]);
        assert_eq!(mat.metallic_factor, 0.8);
        assert_eq!(mat.roughness_factor, 0.3);

        assert!(table.get(999).is_none());
    }

    #[test]
    fn test_gpu_material_table_get_mut() {
        let mut table = GpuMaterialTable::new(64);
        let idx = table.add(MaterialDescriptor::opaque(1.0, 0.0, 0.0));

        // Clear dirty flag for testing
        table.dirty = false;

        {
            let mat = table.get_mut(idx).unwrap();
            mat.base_color_factor = [0.0, 1.0, 0.0, 1.0];
        }

        // get_mut should mark table dirty
        assert!(table.is_dirty());

        let mat = table.get(idx).unwrap();
        assert_eq!(mat.base_color_factor, [0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_gpu_material_table_update() {
        let mut table = GpuMaterialTable::new(64);
        let idx = table.add(MaterialDescriptor::opaque(1.0, 0.0, 0.0));
        table.dirty = false;

        let result = table.update(idx, MaterialDescriptor::metallic(0.8, 0.8, 0.8, 0.9, 0.1));
        assert!(result);
        assert!(table.is_dirty());

        let mat = table.get(idx).unwrap();
        assert_eq!(mat.metallic_factor, 0.9);

        // Update invalid index
        assert!(!table.update(999, MaterialDescriptor::new()));
    }

    #[test]
    fn test_gpu_material_table_remove() {
        let mut table = GpuMaterialTable::new(64);
        let idx0 = table.add(MaterialDescriptor::opaque(1.0, 0.0, 0.0));
        let idx1 = table.add(MaterialDescriptor::opaque(0.0, 1.0, 0.0));
        table.dirty = false;

        assert!(table.remove(idx0));
        assert!(table.is_dirty());
        assert_eq!(table.free_count(), 1);
        assert_eq!(table.active_count(), 1);

        // Material at idx0 should be zeroed
        let mat = table.get(idx0).unwrap();
        assert_eq!(mat, &MaterialDescriptor::zeroed());

        // Remove invalid index
        assert!(!table.remove(999));
    }

    #[test]
    fn test_gpu_material_table_index_reuse() {
        let mut table = GpuMaterialTable::new(64);

        let idx0 = table.add(MaterialDescriptor::opaque(1.0, 0.0, 0.0));
        let idx1 = table.add(MaterialDescriptor::opaque(0.0, 1.0, 0.0));

        // Remove first material
        table.remove(idx0);
        assert_eq!(table.free_count(), 1);

        // Add new material should reuse idx0
        let idx2 = table.add(MaterialDescriptor::opaque(0.0, 0.0, 1.0));
        assert_eq!(idx2, idx0);
        assert_eq!(table.free_count(), 0);
    }

    #[test]
    fn test_gpu_material_table_clear() {
        let mut table = GpuMaterialTable::new(64);
        table.add(MaterialDescriptor::opaque(1.0, 0.0, 0.0));
        table.add(MaterialDescriptor::opaque(0.0, 1.0, 0.0));
        table.dirty = false;

        table.clear();
        assert!(table.is_empty());
        assert_eq!(table.len(), 0);
        assert_eq!(table.free_count(), 0);
        assert!(table.is_dirty());
    }

    #[test]
    fn test_gpu_material_table_as_bytes() {
        let mut table = GpuMaterialTable::new(64);
        table.add(MaterialDescriptor::opaque(1.0, 0.0, 0.0));
        table.add(MaterialDescriptor::opaque(0.0, 1.0, 0.0));

        let bytes = table.as_bytes();
        assert_eq!(bytes.len(), 2 * MATERIAL_DESCRIPTOR_SIZE);
    }

    #[test]
    fn test_gpu_material_table_as_slice() {
        let mut table = GpuMaterialTable::new(64);
        table.add(MaterialDescriptor::opaque(1.0, 0.0, 0.0));
        table.add(MaterialDescriptor::opaque(0.0, 1.0, 0.0));

        let slice = table.as_slice();
        assert_eq!(slice.len(), 2);
        assert_eq!(slice[0].base_color_factor, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(slice[1].base_color_factor, [0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_gpu_material_table_mark_dirty() {
        let mut table = GpuMaterialTable::new(64);
        table.add(MaterialDescriptor::opaque(1.0, 0.0, 0.0));
        table.dirty = false;

        table.mark_dirty();
        assert!(table.is_dirty());
    }

    #[test]
    fn test_gpu_material_table_debug() {
        let mut table = GpuMaterialTable::new(128);
        table.add(MaterialDescriptor::opaque(1.0, 0.0, 0.0));
        table.add(MaterialDescriptor::opaque(0.0, 1.0, 0.0));

        let debug = format!("{:?}", table);
        assert!(debug.contains("GpuMaterialTable"));
        assert!(debug.contains("material_count"));
        assert!(debug.contains("2"));
    }

    #[test]
    fn test_gpu_material_table_capacity_minimum() {
        // Capacity should be at least 1
        let table = GpuMaterialTable::new(0);
        assert_eq!(table.capacity(), 1);
    }

    #[test]
    fn test_gpu_material_table_counts() {
        let mut table = GpuMaterialTable::new(64);

        // Empty table
        assert_eq!(table.active_count(), 0);
        assert_eq!(table.free_count(), 0);

        // Add materials
        let idx0 = table.add(MaterialDescriptor::new());
        let idx1 = table.add(MaterialDescriptor::new());
        let idx2 = table.add(MaterialDescriptor::new());
        assert_eq!(table.active_count(), 3);
        assert_eq!(table.free_count(), 0);

        // Remove one
        table.remove(idx1);
        assert_eq!(table.active_count(), 2);
        assert_eq!(table.free_count(), 1);

        // Remove another
        table.remove(idx0);
        assert_eq!(table.active_count(), 1);
        assert_eq!(table.free_count(), 2);
    }

    // --------------------------------------------------------------
    // Constants tests (T-WGPU-P6.8.4)
    // --------------------------------------------------------------

    #[test]
    fn test_material_descriptor_constants() {
        assert_eq!(MATERIAL_DESCRIPTOR_SIZE, 64);
        assert_eq!(DEFAULT_GPU_MATERIAL_TABLE_CAPACITY, 1024);
        assert_eq!(NO_TEXTURE, u32::MAX);
        assert_eq!(MaterialDescriptor::NO_TEXTURE, u32::MAX);
    }

    #[test]
    fn test_material_desc_flags() {
        assert_eq!(MATERIAL_DESC_FLAG_DOUBLE_SIDED, 1 << 0);
        assert_eq!(MATERIAL_DESC_FLAG_ALPHA_MASK, 1 << 1);
        assert_eq!(MATERIAL_DESC_FLAG_ALPHA_BLEND, 1 << 2);
        assert_eq!(MATERIAL_DESC_FLAG_UNLIT, 1 << 3);

        // Flags should not overlap
        let all_flags = MATERIAL_DESC_FLAG_DOUBLE_SIDED
            | MATERIAL_DESC_FLAG_ALPHA_MASK
            | MATERIAL_DESC_FLAG_ALPHA_BLEND
            | MATERIAL_DESC_FLAG_UNLIT;
        assert_eq!(all_flags, 0b1111);
    }
}
