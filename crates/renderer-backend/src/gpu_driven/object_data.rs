//! Per-object GPU data for GPU-driven rendering (T-WGPU-P6.2.1).
//!
//! This module provides the `ObjectData` struct that stores per-object
//! information for GPU-driven rendering. Each object in the scene has
//! an entry in a storage buffer that is accessed by culling compute
//! shaders and vertex shaders.
//!
//! # Overview
//!
//! In GPU-driven rendering, object data must be efficiently organized for:
//!
//! 1. Frustum and occlusion culling (uses transform, AABB)
//! 2. LOD selection (uses LOD distances, camera distance)
//! 3. Draw call generation (uses mesh/material indices)
//! 4. Instance rendering (uses transform, flags)
//!
//! # Data Layout
//!
//! The `ObjectData` struct is 144 bytes, aligned to 16 bytes:
//!
//! | Offset | Field           | Size | Description                      |
//! |--------|-----------------|------|----------------------------------|
//! | 0      | transform       | 64   | mat4x4 world transform           |
//! | 64     | aabb_min        | 12   | AABB minimum corner (world)      |
//! | 76     | _pad0           | 4    | Padding for alignment            |
//! | 80     | aabb_max        | 12   | AABB maximum corner (world)      |
//! | 92     | _pad1           | 4    | Padding for alignment            |
//! | 96     | mesh_index      | 4    | Index into mesh buffer           |
//! | 100    | material_index  | 4    | Index into material buffer       |
//! | 104    | lod_distances   | 16   | LOD switch distances (squared)   |
//! | 120    | flags           | 4    | Object flags bitfield            |
//! | 124    | _padding        | 20   | Padding for 144-byte size        |
//!
//! Note: WGSL shaders must use `array<f32, 4>` instead of `vec4<f32>` for
//! lod_distances to avoid WGSL's 16-byte alignment requirement for vec4.
//!
//! # Performance
//!
//! - Struct size: 144 bytes (divisible by 16 for GPU alignment)
//! - One object per storage buffer entry
//! - Optimal for GPU cache line access patterns
//!
//! # Usage
//!
//! ```ignore
//! use renderer_backend::gpu_driven::{ObjectData, object_flags};
//!
//! // Create a new visible object
//! let object = ObjectData::new()
//!     .with_transform(world_matrix)
//!     .with_aabb(aabb_min, aabb_max)
//!     .with_mesh(mesh_id)
//!     .with_material(material_id)
//!     .with_lod_distances([100.0, 400.0, 1600.0, 6400.0]);
//!
//! // Check visibility
//! if object.is_visible() {
//!     // Object will be processed by culling shaders
//! }
//!
//! // Access as bytes for GPU upload
//! let bytes: &[u8] = bytemuck::bytes_of(&object);
//! queue.write_buffer(&object_buffer, offset, bytes);
//! ```

use bytemuck::{Pod, Zeroable};

// =============================================================================
// CONSTANTS
// =============================================================================

/// Size of ObjectData struct in bytes.
pub const OBJECT_DATA_SIZE: usize = 144;

/// Maximum number of LOD levels supported.
pub const MAX_LOD_LEVELS: usize = 4;

/// Invalid mesh index marker.
pub const INVALID_MESH_INDEX: u32 = 0xFFFFFFFF;

/// Invalid material index marker.
pub const INVALID_MATERIAL_INDEX: u32 = 0xFFFFFFFF;

/// Default LOD distances (squared) for LOD 0-3.
/// LOD switches at 10, 25, 50, and 100 units respectively.
pub const DEFAULT_LOD_DISTANCES: [f32; 4] = [100.0, 625.0, 2500.0, 10000.0];

// =============================================================================
// OBJECT FLAGS
// =============================================================================

/// Object flags bitfield constants.
///
/// These flags control object behavior during culling and rendering.
pub mod object_flags {
    /// Object is visible and should be rendered.
    pub const VISIBLE: u32 = 1 << 0;

    /// Object casts shadows.
    pub const CASTS_SHADOW: u32 = 1 << 1;

    /// Object is static (no transform updates needed).
    pub const STATIC: u32 = 1 << 2;

    /// Object receives decals.
    pub const RECEIVES_DECALS: u32 = 1 << 3;

    /// Object receives shadows.
    pub const RECEIVES_SHADOW: u32 = 1 << 4;

    /// Object uses two-sided rendering.
    pub const TWO_SIDED: u32 = 1 << 5;

    /// Object uses alpha testing.
    pub const ALPHA_TEST: u32 = 1 << 6;

    /// Object uses alpha blending (requires sorting).
    pub const ALPHA_BLEND: u32 = 1 << 7;

    /// Object is selected (for editor highlighting).
    pub const SELECTED: u32 = 1 << 8;

    /// Object transform is dirty and needs update.
    pub const DIRTY: u32 = 1 << 9;

    /// Object has skinned animation.
    pub const SKINNED: u32 = 1 << 10;

    /// Object participates in motion blur.
    pub const MOTION_BLUR: u32 = 1 << 11;

    /// Default flags for a newly created object.
    pub const DEFAULT: u32 = VISIBLE | CASTS_SHADOW | RECEIVES_SHADOW | RECEIVES_DECALS;
}

// =============================================================================
// OBJECT DATA STRUCTURE
// =============================================================================

/// Per-object GPU data for GPU-driven rendering.
///
/// This struct is stored in a storage buffer and accessed by culling
/// compute shaders and vertex shaders. Each renderable object in the
/// scene has a corresponding `ObjectData` entry.
///
/// # Layout (144 bytes, 16-byte aligned)
///
/// ```text
/// +-------------------+--------+--------+----------------------------------+
/// | Field             | Offset | Size   | Description                      |
/// +-------------------+--------+--------+----------------------------------+
/// | transform         | 0      | 64     | mat4x4 world transform           |
/// | aabb_min          | 64     | 12     | AABB minimum corner              |
/// | _pad0             | 76     | 4      | Alignment padding                |
/// | aabb_max          | 80     | 12     | AABB maximum corner              |
/// | _pad1             | 92     | 4      | Alignment padding                |
/// | mesh_index        | 96     | 4      | Index into mesh table            |
/// | material_index    | 100    | 4      | Index into material table        |
/// | lod_distances     | 104    | 16     | LOD switch distances (squared)   |
/// | flags             | 120    | 4      | Object flags bitfield            |
/// | _padding          | 124    | 20     | Padding for 144-byte alignment   |
/// +-------------------+--------+--------+----------------------------------+
/// | Total             |        | 144    |                                  |
/// +-------------------+--------+--------+----------------------------------+
/// ```
///
/// # WGSL Binding
///
/// ```wgsl
/// struct ObjectData {
///     transform: mat4x4<f32>,
///     aabb_min: vec3<f32>,
///     _pad0: f32,
///     aabb_max: vec3<f32>,
///     _pad1: f32,
///     mesh_index: u32,
///     material_index: u32,
///     lod_distances: vec4<f32>,
///     flags: u32,
///     _padding: array<u32, 5>,
/// }
///
/// @group(0) @binding(0)
/// var<storage, read> objects: array<ObjectData>;
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct ObjectData {
    /// World transform matrix (column-major, 4x4).
    ///
    /// Layout: `[[col0], [col1], [col2], [col3]]`
    /// where each column is `[x, y, z, w]`.
    /// Column 3 contains the translation.
    pub transform: [[f32; 4]; 4],

    /// Axis-Aligned Bounding Box minimum corner (world space).
    pub aabb_min: [f32; 3],

    /// Padding for vec4 alignment.
    pub _pad0: f32,

    /// Axis-Aligned Bounding Box maximum corner (world space).
    pub aabb_max: [f32; 3],

    /// Padding for vec4 alignment.
    pub _pad1: f32,

    /// Index into the mesh table (bindless mesh buffer).
    pub mesh_index: u32,

    /// Index into the material table (bindless material buffer).
    pub material_index: u32,

    /// LOD switch distances (squared, in world units).
    ///
    /// Index 0 = distance² to switch from LOD0 to LOD1.
    /// Index 1 = distance² to switch from LOD1 to LOD2.
    /// Index 2 = distance² to switch from LOD2 to LOD3.
    /// Index 3 = distance² beyond which object is culled.
    pub lod_distances: [f32; 4],

    /// Object flags bitfield (see `object_flags` module).
    pub flags: u32,

    /// Padding for 144-byte alignment (5 x u32 = 20 bytes).
    pub _padding: [u32; 5],
}

impl Default for ObjectData {
    fn default() -> Self {
        Self::new()
    }
}

impl ObjectData {
    /// Size of this struct in bytes.
    pub const SIZE: usize = OBJECT_DATA_SIZE;

    /// Create a new ObjectData with identity transform and default flags.
    ///
    /// The object is visible by default with standard shadow and decal
    /// settings. Transform is identity, AABB is zeroed, and indices
    /// are set to invalid markers.
    #[inline]
    pub const fn new() -> Self {
        Self {
            transform: [
                [1.0, 0.0, 0.0, 0.0], // Column 0 (X axis)
                [0.0, 1.0, 0.0, 0.0], // Column 1 (Y axis)
                [0.0, 0.0, 1.0, 0.0], // Column 2 (Z axis)
                [0.0, 0.0, 0.0, 1.0], // Column 3 (Translation + W)
            ],
            aabb_min: [0.0, 0.0, 0.0],
            _pad0: 0.0,
            aabb_max: [0.0, 0.0, 0.0],
            _pad1: 0.0,
            mesh_index: INVALID_MESH_INDEX,
            material_index: INVALID_MATERIAL_INDEX,
            lod_distances: DEFAULT_LOD_DISTANCES,
            flags: object_flags::DEFAULT,
            _padding: [0; 5],
        }
    }

    /// Create a zeroed ObjectData (invisible, no mesh/material).
    #[inline]
    pub const fn zeroed() -> Self {
        Self {
            transform: [[0.0; 4]; 4],
            aabb_min: [0.0; 3],
            _pad0: 0.0,
            aabb_max: [0.0; 3],
            _pad1: 0.0,
            mesh_index: 0,
            material_index: 0,
            lod_distances: [0.0; 4],
            flags: 0,
            _padding: [0; 5],
        }
    }

    /// Set the world transform matrix.
    #[inline]
    pub const fn with_transform(mut self, transform: [[f32; 4]; 4]) -> Self {
        self.transform = transform;
        self
    }

    /// Set the AABB bounds (world space).
    #[inline]
    pub const fn with_aabb(mut self, min: [f32; 3], max: [f32; 3]) -> Self {
        self.aabb_min = min;
        self.aabb_max = max;
        self
    }

    /// Set the mesh index.
    #[inline]
    pub const fn with_mesh(mut self, mesh_index: u32) -> Self {
        self.mesh_index = mesh_index;
        self
    }

    /// Set the material index.
    #[inline]
    pub const fn with_material(mut self, material_index: u32) -> Self {
        self.material_index = material_index;
        self
    }

    /// Set the LOD switch distances (squared).
    #[inline]
    pub const fn with_lod_distances(mut self, distances: [f32; 4]) -> Self {
        self.lod_distances = distances;
        self
    }

    /// Set the object flags.
    #[inline]
    pub const fn with_flags(mut self, flags: u32) -> Self {
        self.flags = flags;
        self
    }

    /// Add flags to the current flags.
    #[inline]
    pub const fn with_flags_added(mut self, flags: u32) -> Self {
        self.flags |= flags;
        self
    }

    /// Remove flags from the current flags.
    #[inline]
    pub const fn with_flags_removed(mut self, flags: u32) -> Self {
        self.flags &= !flags;
        self
    }

    // -------------------------------------------------------------------------
    // Flag accessors
    // -------------------------------------------------------------------------

    /// Check if the object is visible.
    #[inline]
    pub const fn is_visible(&self) -> bool {
        (self.flags & object_flags::VISIBLE) != 0
    }

    /// Check if the object casts shadows.
    #[inline]
    pub const fn casts_shadow(&self) -> bool {
        (self.flags & object_flags::CASTS_SHADOW) != 0
    }

    /// Check if the object is static.
    #[inline]
    pub const fn is_static(&self) -> bool {
        (self.flags & object_flags::STATIC) != 0
    }

    /// Check if the object receives decals.
    #[inline]
    pub const fn receives_decals(&self) -> bool {
        (self.flags & object_flags::RECEIVES_DECALS) != 0
    }

    /// Check if the object receives shadows.
    #[inline]
    pub const fn receives_shadow(&self) -> bool {
        (self.flags & object_flags::RECEIVES_SHADOW) != 0
    }

    /// Check if the object uses two-sided rendering.
    #[inline]
    pub const fn is_two_sided(&self) -> bool {
        (self.flags & object_flags::TWO_SIDED) != 0
    }

    /// Check if the object uses alpha testing.
    #[inline]
    pub const fn has_alpha_test(&self) -> bool {
        (self.flags & object_flags::ALPHA_TEST) != 0
    }

    /// Check if the object uses alpha blending.
    #[inline]
    pub const fn has_alpha_blend(&self) -> bool {
        (self.flags & object_flags::ALPHA_BLEND) != 0
    }

    /// Check if the object is selected (editor).
    #[inline]
    pub const fn is_selected(&self) -> bool {
        (self.flags & object_flags::SELECTED) != 0
    }

    /// Check if the object transform is dirty.
    #[inline]
    pub const fn is_dirty(&self) -> bool {
        (self.flags & object_flags::DIRTY) != 0
    }

    /// Check if the object has skinned animation.
    #[inline]
    pub const fn is_skinned(&self) -> bool {
        (self.flags & object_flags::SKINNED) != 0
    }

    /// Check if the object participates in motion blur.
    #[inline]
    pub const fn has_motion_blur(&self) -> bool {
        (self.flags & object_flags::MOTION_BLUR) != 0
    }

    // -------------------------------------------------------------------------
    // Flag modifiers
    // -------------------------------------------------------------------------

    /// Set visibility flag.
    #[inline]
    pub fn set_visible(&mut self, visible: bool) {
        if visible {
            self.flags |= object_flags::VISIBLE;
        } else {
            self.flags &= !object_flags::VISIBLE;
        }
    }

    /// Set shadow casting flag.
    #[inline]
    pub fn set_casts_shadow(&mut self, casts: bool) {
        if casts {
            self.flags |= object_flags::CASTS_SHADOW;
        } else {
            self.flags &= !object_flags::CASTS_SHADOW;
        }
    }

    /// Set static flag.
    #[inline]
    pub fn set_static(&mut self, is_static: bool) {
        if is_static {
            self.flags |= object_flags::STATIC;
        } else {
            self.flags &= !object_flags::STATIC;
        }
    }

    /// Set dirty flag.
    #[inline]
    pub fn set_dirty(&mut self, dirty: bool) {
        if dirty {
            self.flags |= object_flags::DIRTY;
        } else {
            self.flags &= !object_flags::DIRTY;
        }
    }

    // -------------------------------------------------------------------------
    // Utility methods
    // -------------------------------------------------------------------------

    /// Check if mesh and material indices are valid.
    #[inline]
    pub const fn has_valid_resources(&self) -> bool {
        self.mesh_index != INVALID_MESH_INDEX && self.material_index != INVALID_MATERIAL_INDEX
    }

    /// Get the AABB center point.
    #[inline]
    pub fn aabb_center(&self) -> [f32; 3] {
        [
            (self.aabb_min[0] + self.aabb_max[0]) * 0.5,
            (self.aabb_min[1] + self.aabb_max[1]) * 0.5,
            (self.aabb_min[2] + self.aabb_max[2]) * 0.5,
        ]
    }

    /// Get the AABB extents (half-size).
    #[inline]
    pub fn aabb_extents(&self) -> [f32; 3] {
        [
            (self.aabb_max[0] - self.aabb_min[0]) * 0.5,
            (self.aabb_max[1] - self.aabb_min[1]) * 0.5,
            (self.aabb_max[2] - self.aabb_min[2]) * 0.5,
        ]
    }

    /// Get the AABB bounding sphere radius.
    #[inline]
    pub fn bounding_sphere_radius(&self) -> f32 {
        let extents = self.aabb_extents();
        (extents[0] * extents[0] + extents[1] * extents[1] + extents[2] * extents[2]).sqrt()
    }

    /// Get the translation from the transform matrix.
    #[inline]
    pub const fn translation(&self) -> [f32; 3] {
        [self.transform[3][0], self.transform[3][1], self.transform[3][2]]
    }

    /// Select LOD level based on squared distance from camera.
    ///
    /// Returns LOD level 0-3, or MAX_LOD_LEVELS if culled.
    #[inline]
    pub fn select_lod(&self, distance_squared: f32) -> usize {
        for (i, &lod_dist) in self.lod_distances.iter().enumerate() {
            if distance_squared < lod_dist {
                return i;
            }
        }
        MAX_LOD_LEVELS // Culled
    }
}

// =============================================================================
// OBJECT DATA BUFFER
// =============================================================================

/// A buffer of ObjectData entries for GPU-driven rendering.
///
/// This struct manages a collection of objects that can be uploaded
/// to the GPU for culling and rendering operations.
#[derive(Clone, Debug)]
pub struct ObjectDataBuffer {
    /// CPU-side object data.
    objects: Vec<ObjectData>,

    /// Capacity of the buffer.
    capacity: usize,

    /// Dirty flag indicating data needs upload.
    dirty: bool,
}

impl ObjectDataBuffer {
    /// Create a new ObjectDataBuffer with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            objects: Vec::with_capacity(capacity),
            capacity,
            dirty: false,
        }
    }

    /// Create a buffer with initial objects.
    pub fn with_objects(objects: Vec<ObjectData>) -> Self {
        let capacity = objects.capacity();
        Self {
            objects,
            capacity,
            dirty: true,
        }
    }

    /// Get the number of objects.
    #[inline]
    pub fn len(&self) -> usize {
        self.objects.len()
    }

    /// Check if the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }

    /// Get the capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Check if the buffer is dirty (needs upload).
    #[inline]
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Clear the dirty flag.
    #[inline]
    pub fn clear_dirty(&mut self) {
        self.dirty = false;
    }

    /// Add an object to the buffer.
    ///
    /// Returns the index of the added object.
    pub fn add(&mut self, object: ObjectData) -> usize {
        let index = self.objects.len();
        self.objects.push(object);
        self.dirty = true;
        index
    }

    /// Get an object by index.
    #[inline]
    pub fn get(&self, index: usize) -> Option<&ObjectData> {
        self.objects.get(index)
    }

    /// Get a mutable reference to an object by index.
    #[inline]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut ObjectData> {
        self.dirty = true;
        self.objects.get_mut(index)
    }

    /// Get the objects as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[ObjectData] {
        &self.objects
    }

    /// Get the objects as bytes for GPU upload.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.objects)
    }

    /// Get the buffer size in bytes.
    #[inline]
    pub fn byte_size(&self) -> usize {
        self.objects.len() * ObjectData::SIZE
    }

    /// Clear all objects.
    pub fn clear(&mut self) {
        self.objects.clear();
        self.dirty = true;
    }

    /// Update an object's transform.
    pub fn update_transform(&mut self, index: usize, transform: [[f32; 4]; 4]) {
        if let Some(obj) = self.objects.get_mut(index) {
            obj.transform = transform;
            obj.flags |= object_flags::DIRTY;
            self.dirty = true;
        }
    }

    /// Update an object's AABB.
    pub fn update_aabb(&mut self, index: usize, min: [f32; 3], max: [f32; 3]) {
        if let Some(obj) = self.objects.get_mut(index) {
            obj.aabb_min = min;
            obj.aabb_max = max;
            self.dirty = true;
        }
    }

    /// Update an object's visibility.
    pub fn set_visible(&mut self, index: usize, visible: bool) {
        if let Some(obj) = self.objects.get_mut(index) {
            obj.set_visible(visible);
            self.dirty = true;
        }
    }
}

impl Default for ObjectDataBuffer {
    fn default() -> Self {
        Self::new(1024)
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_object_data_size() {
        // Must be at least 144 bytes
        assert!(
            ObjectData::SIZE >= 144,
            "ObjectData::SIZE = {}, expected >= 144",
            ObjectData::SIZE
        );

        // Must be 16-byte aligned for GPU
        assert_eq!(
            ObjectData::SIZE % 16,
            0,
            "ObjectData::SIZE = {} is not 16-byte aligned",
            ObjectData::SIZE
        );

        // Verify actual struct size matches constant
        assert_eq!(
            std::mem::size_of::<ObjectData>(),
            ObjectData::SIZE,
            "ObjectData struct size mismatch"
        );
    }

    #[test]
    fn test_transform_identity() {
        let obj = ObjectData::new();

        // Check identity matrix
        assert_eq!(obj.transform[0][0], 1.0, "transform[0][0] should be 1.0");
        assert_eq!(obj.transform[1][1], 1.0, "transform[1][1] should be 1.0");
        assert_eq!(obj.transform[2][2], 1.0, "transform[2][2] should be 1.0");
        assert_eq!(obj.transform[3][3], 1.0, "transform[3][3] should be 1.0");

        // Check off-diagonal zeros
        assert_eq!(obj.transform[0][1], 0.0);
        assert_eq!(obj.transform[0][2], 0.0);
        assert_eq!(obj.transform[1][0], 0.0);
        assert_eq!(obj.transform[1][2], 0.0);
        assert_eq!(obj.transform[2][0], 0.0);
        assert_eq!(obj.transform[2][1], 0.0);

        // Check translation is zero
        assert_eq!(obj.translation(), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_bytemuck_pod_zeroable() {
        // Test Pod trait - can convert to bytes
        let obj = ObjectData::new();
        let bytes: &[u8] = bytemuck::bytes_of(&obj);
        assert_eq!(bytes.len(), ObjectData::SIZE);

        // Test roundtrip
        let obj2: &ObjectData = bytemuck::from_bytes(bytes);
        assert_eq!(obj.transform, obj2.transform);
        assert_eq!(obj.flags, obj2.flags);

        // Test Zeroable trait
        let zeroed: ObjectData = ObjectData::zeroed();
        assert_eq!(zeroed.flags, 0);
        assert_eq!(zeroed.transform[0][0], 0.0);
    }

    #[test]
    fn test_mesh_index_byte_offset() {
        // Verify mesh_index is at byte offset 96 (must match WGSL struct)
        let mut obj = ObjectData::zeroed();
        obj.mesh_index = 0xDEADBEEF;

        let bytes: &[u8] = bytemuck::bytes_of(&obj);

        // mesh_index should be at offset 96
        let mesh_bytes = &bytes[96..100];
        let mesh_val = u32::from_le_bytes([mesh_bytes[0], mesh_bytes[1], mesh_bytes[2], mesh_bytes[3]]);
        assert_eq!(
            mesh_val, 0xDEADBEEF,
            "mesh_index at wrong offset. Expected 0xDEADBEEF at bytes 96-99, got 0x{:08X}",
            mesh_val
        );
    }

    #[test]
    fn test_lod_distances_byte_offset() {
        // Verify lod_distances is at byte offset 104
        // WGSL shaders must use array<f32, 4> instead of vec4<f32> to match this layout
        let mut obj = ObjectData::zeroed();
        obj.lod_distances = [1.0, 2.0, 3.0, 4.0];

        let bytes: &[u8] = bytemuck::bytes_of(&obj);

        // lod_distances[0] should be at offset 104
        let lod0_bytes = &bytes[104..108];
        let lod0_val = f32::from_le_bytes([lod0_bytes[0], lod0_bytes[1], lod0_bytes[2], lod0_bytes[3]]);
        assert_eq!(
            lod0_val, 1.0,
            "lod_distances[0] at wrong offset. Expected 1.0 at bytes 104-107, got {}",
            lod0_val
        );
    }

    #[test]
    fn test_flags_byte_offset() {
        // Verify flags is at byte offset 120
        let mut obj = ObjectData::zeroed();
        obj.flags = 0xCAFEBABE;

        let bytes: &[u8] = bytemuck::bytes_of(&obj);

        let flags_bytes = &bytes[120..124];
        let flags_val = u32::from_le_bytes([flags_bytes[0], flags_bytes[1], flags_bytes[2], flags_bytes[3]]);
        assert_eq!(
            flags_val, 0xCAFEBABE,
            "flags at wrong offset. Expected 0xCAFEBABE at bytes 120-123, got 0x{:08X}",
            flags_val
        );
    }

    #[test]
    fn test_flags() {
        let obj = ObjectData::new();

        // Default flags should include VISIBLE
        assert!(obj.is_visible(), "New object should be visible");
        assert!(obj.casts_shadow(), "New object should cast shadow");
        assert!(obj.receives_shadow(), "New object should receive shadow");
        assert!(obj.receives_decals(), "New object should receive decals");

        // Default should NOT include these
        assert!(!obj.is_static(), "New object should not be static");
        assert!(!obj.is_selected(), "New object should not be selected");
        assert!(!obj.is_dirty(), "New object should not be dirty");
    }

    #[test]
    fn test_flag_modifiers() {
        let mut obj = ObjectData::new();

        // Test set_visible
        obj.set_visible(false);
        assert!(!obj.is_visible());
        obj.set_visible(true);
        assert!(obj.is_visible());

        // Test set_static
        obj.set_static(true);
        assert!(obj.is_static());
        obj.set_static(false);
        assert!(!obj.is_static());

        // Test set_dirty
        obj.set_dirty(true);
        assert!(obj.is_dirty());
        obj.set_dirty(false);
        assert!(!obj.is_dirty());
    }

    #[test]
    fn test_builder_pattern() {
        let transform = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [10.0, 20.0, 30.0, 1.0], // Translation
        ];

        let obj = ObjectData::new()
            .with_transform(transform)
            .with_aabb([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
            .with_mesh(42)
            .with_material(7)
            .with_lod_distances([100.0, 400.0, 1600.0, 6400.0]);

        assert_eq!(obj.transform, transform);
        assert_eq!(obj.aabb_min, [-1.0, -1.0, -1.0]);
        assert_eq!(obj.aabb_max, [1.0, 1.0, 1.0]);
        assert_eq!(obj.mesh_index, 42);
        assert_eq!(obj.material_index, 7);
        assert_eq!(obj.lod_distances, [100.0, 400.0, 1600.0, 6400.0]);
        assert_eq!(obj.translation(), [10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_aabb_methods() {
        let obj = ObjectData::new()
            .with_aabb([0.0, 0.0, 0.0], [2.0, 4.0, 6.0]);

        // Center should be midpoint
        let center = obj.aabb_center();
        assert_eq!(center, [1.0, 2.0, 3.0]);

        // Extents should be half-size
        let extents = obj.aabb_extents();
        assert_eq!(extents, [1.0, 2.0, 3.0]);

        // Bounding sphere radius
        let radius = obj.bounding_sphere_radius();
        let expected = (1.0_f32 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0).sqrt();
        assert!((radius - expected).abs() < 1e-6);
    }

    #[test]
    fn test_lod_selection() {
        let obj = ObjectData::new()
            .with_lod_distances([100.0, 400.0, 1600.0, 6400.0]);

        // LOD 0 (distance² < 100)
        assert_eq!(obj.select_lod(50.0), 0);
        assert_eq!(obj.select_lod(99.9), 0);

        // LOD 1 (100 <= distance² < 400)
        assert_eq!(obj.select_lod(100.0), 1);
        assert_eq!(obj.select_lod(399.9), 1);

        // LOD 2 (400 <= distance² < 1600)
        assert_eq!(obj.select_lod(400.0), 2);
        assert_eq!(obj.select_lod(1599.9), 2);

        // LOD 3 (1600 <= distance² < 6400)
        assert_eq!(obj.select_lod(1600.0), 3);
        assert_eq!(obj.select_lod(6399.9), 3);

        // Culled (distance² >= 6400)
        assert_eq!(obj.select_lod(6400.0), MAX_LOD_LEVELS);
        assert_eq!(obj.select_lod(10000.0), MAX_LOD_LEVELS);
    }

    #[test]
    fn test_valid_resources() {
        let obj = ObjectData::new();
        assert!(!obj.has_valid_resources(), "New object has invalid indices");

        let obj = ObjectData::new().with_mesh(0).with_material(0);
        assert!(obj.has_valid_resources(), "Object with index 0 is valid");

        let obj = ObjectData::new().with_mesh(INVALID_MESH_INDEX).with_material(0);
        assert!(!obj.has_valid_resources(), "Invalid mesh index");

        let obj = ObjectData::new().with_mesh(0).with_material(INVALID_MATERIAL_INDEX);
        assert!(!obj.has_valid_resources(), "Invalid material index");
    }

    #[test]
    fn test_object_data_buffer() {
        let mut buffer = ObjectDataBuffer::new(100);

        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);
        assert!(!buffer.is_dirty());

        // Add objects
        let idx0 = buffer.add(ObjectData::new().with_mesh(0));
        let idx1 = buffer.add(ObjectData::new().with_mesh(1));

        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
        assert_eq!(buffer.len(), 2);
        assert!(buffer.is_dirty());

        // Get object
        let obj = buffer.get(0).unwrap();
        assert_eq!(obj.mesh_index, 0);

        // Clear dirty and verify
        buffer.clear_dirty();
        assert!(!buffer.is_dirty());

        // Modify object
        buffer.update_transform(0, [[2.0, 0.0, 0.0, 0.0]; 4]);
        assert!(buffer.is_dirty());

        // Byte size check
        assert_eq!(buffer.byte_size(), 2 * ObjectData::SIZE);

        // Clear
        buffer.clear();
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_with_flags() {
        let obj = ObjectData::new()
            .with_flags(object_flags::VISIBLE | object_flags::STATIC);

        assert!(obj.is_visible());
        assert!(obj.is_static());
        assert!(!obj.casts_shadow()); // Removed by with_flags

        let obj = ObjectData::new()
            .with_flags_added(object_flags::STATIC | object_flags::SELECTED);

        assert!(obj.is_visible()); // Default still present
        assert!(obj.is_static());
        assert!(obj.is_selected());

        let obj = ObjectData::new()
            .with_flags_removed(object_flags::CASTS_SHADOW);

        assert!(obj.is_visible());
        assert!(!obj.casts_shadow());
    }
}
