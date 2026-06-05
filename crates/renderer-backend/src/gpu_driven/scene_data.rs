//! Scene Data Buffers for GPU-driven rendering (T-WGPU-P6.2.2).
//!
//! This module provides the `SceneDataBuffers` system that manages GPU storage
//! buffers containing per-object data for the entire scene. It supports dynamic
//! resizing and efficient partial uploads through dirty range tracking.
//!
//! # Overview
//!
//! In GPU-driven rendering, scene data must be efficiently organized for:
//!
//! 1. **GPU storage**: All object data in a single storage buffer
//! 2. **Dynamic updates**: CPU-side staging with dirty tracking
//! 3. **Automatic resizing**: Grows buffer when capacity is exceeded
//! 4. **Partial uploads**: Only uploads dirty ranges for efficiency
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                      SceneDataBuffers                          │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  staging: Vec<ObjectData>   ──────────────────────────────────┐ │
//! │  [Object0][Object1][Object2]...[ObjectN-1]                    │ │
//! │      ↓         ↓         ↓           ↓                        │ │
//! │  dirty_start=1  ←───── dirty range ─────→  dirty_end=N       │ │
//! │                                                               │ │
//! │  ┌─────────────────────────────────────────────────────────┐  │ │
//! │  │             GPU Storage Buffer (object_buffer)          │  │ │
//! │  │  [Object0][Object1][Object2]...[ObjectN-1][unused...]   │  │ │
//! │  └─────────────────────────────────────────────────────────┘  │ │
//! │                                                               │ │
//! │  count: N        capacity: M (M >= N)                         │ │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Performance
//!
//! - **Dirty tracking**: Only uploads modified ranges (O(dirty_count))
//! - **Batch updates**: Multiple modifications coalesced into single upload
//! - **Resize strategy**: 2x growth factor to amortize allocation cost
//! - **Memory**: sizeof(ObjectData) per object (144 bytes)
//!
//! # Usage
//!
//! ```ignore
//! use renderer_backend::gpu_driven::{SceneDataBuffers, ObjectData};
//!
//! // Create scene buffers with initial capacity
//! let mut scene = SceneDataBuffers::new(&device, 1024, Some("main_scene"));
//!
//! // Add objects to the scene
//! let idx = scene.add(ObjectData::new()
//!     .with_mesh(mesh_id)
//!     .with_material(material_id)
//!     .with_transform(world_matrix));
//!
//! // Modify an object
//! if let Some(obj) = scene.get_mut(idx) {
//!     obj.set_visible(false);
//! }
//!
//! // Upload dirty data to GPU (call once per frame)
//! scene.upload(&device, &queue);
//!
//! // Bind for rendering
//! let binding = wgpu::BindGroupEntry {
//!     binding: 0,
//!     resource: scene.object_buffer().as_entire_binding(),
//! };
//! ```

use super::object_data::ObjectData;

// =============================================================================
// CONSTANTS
// =============================================================================

/// Default initial capacity for scene data buffers.
pub const DEFAULT_SCENE_CAPACITY: usize = 4096;

/// Minimum buffer size to avoid degenerate cases.
pub const MIN_BUFFER_CAPACITY: usize = 16;

/// Growth factor when resizing buffers.
pub const GROWTH_FACTOR: usize = 2;

// =============================================================================
// SCENE DATA BUFFERS
// =============================================================================

/// Scene data buffer system for GPU-driven rendering.
///
/// Manages storage buffers containing per-object data for the entire scene.
/// Supports dynamic resizing and efficient partial uploads through dirty
/// range tracking.
///
/// # Design
///
/// - **Storage buffer**: Single `wgpu::Buffer` holding all `ObjectData` entries
/// - **CPU staging**: `Vec<ObjectData>` for modifications before upload
/// - **Dirty tracking**: Range-based tracking for partial GPU uploads
/// - **Dynamic resize**: Automatically grows when capacity is exceeded
///
/// # Thread Safety
///
/// This struct is not thread-safe. If multi-threaded access is needed,
/// wrap in appropriate synchronization primitives.
pub struct SceneDataBuffers {
    /// GPU storage buffer for ObjectData array.
    object_buffer: wgpu::Buffer,

    /// CPU-side staging buffer.
    staging: Vec<ObjectData>,

    /// Current capacity (number of objects the GPU buffer can hold).
    capacity: usize,

    /// Number of active objects.
    count: usize,

    /// Label for debugging.
    label: Option<String>,

    /// Start of dirty range (inclusive).
    /// Set to `usize::MAX` when no dirty range.
    dirty_start: usize,

    /// End of dirty range (exclusive).
    /// Set to `0` when no dirty range.
    dirty_end: usize,

    /// Track if buffer was resized since last upload.
    resized: bool,
}

impl SceneDataBuffers {
    /// Create new scene data buffers with initial capacity.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device for buffer creation
    /// * `capacity` - Initial capacity (number of objects)
    /// * `label` - Optional debug label for the buffer
    ///
    /// # Example
    ///
    /// ```ignore
    /// let scene = SceneDataBuffers::new(&device, 1024, Some("main_scene"));
    /// ```
    pub fn new(device: &wgpu::Device, capacity: usize, label: Option<&str>) -> Self {
        let capacity = capacity.max(MIN_BUFFER_CAPACITY);

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: label.map(|l| format!("{}_object_buffer", l)).as_deref(),
            size: (capacity * ObjectData::SIZE) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            object_buffer: buffer,
            staging: Vec::with_capacity(capacity),
            capacity,
            count: 0,
            label: label.map(String::from),
            dirty_start: usize::MAX,
            dirty_end: 0,
            resized: false,
        }
    }

    /// Create scene data buffers with default capacity.
    ///
    /// Uses `DEFAULT_SCENE_CAPACITY` (4096 objects).
    pub fn with_default_capacity(device: &wgpu::Device, label: Option<&str>) -> Self {
        Self::new(device, DEFAULT_SCENE_CAPACITY, label)
    }

    // -------------------------------------------------------------------------
    // Object Management
    // -------------------------------------------------------------------------

    /// Add an object to the scene, returning its index.
    ///
    /// The object will be uploaded on the next call to `upload()`.
    ///
    /// # Arguments
    ///
    /// * `object` - The ObjectData to add
    ///
    /// # Returns
    ///
    /// The index of the newly added object.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let idx = scene.add(ObjectData::new().with_mesh(0));
    /// ```
    pub fn add(&mut self, object: ObjectData) -> usize {
        let index = self.staging.len();
        self.staging.push(object);
        self.count = self.staging.len();
        self.mark_dirty(index);
        index
    }

    /// Add multiple objects to the scene, returning the starting index.
    ///
    /// More efficient than calling `add()` multiple times.
    ///
    /// # Arguments
    ///
    /// * `objects` - Iterator of ObjectData to add
    ///
    /// # Returns
    ///
    /// The starting index of the first added object.
    pub fn add_batch(&mut self, objects: impl IntoIterator<Item = ObjectData>) -> usize {
        let start_index = self.staging.len();
        self.staging.extend(objects);
        self.count = self.staging.len();

        if self.staging.len() > start_index {
            self.mark_dirty_range(start_index, self.staging.len());
        }

        start_index
    }

    /// Get immutable reference to object at index.
    #[inline]
    pub fn get(&self, index: usize) -> Option<&ObjectData> {
        self.staging.get(index)
    }

    /// Get mutable reference to object at index.
    ///
    /// Automatically marks the object as dirty for upload.
    #[inline]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut ObjectData> {
        if index < self.staging.len() {
            self.mark_dirty(index);
            self.staging.get_mut(index)
        } else {
            None
        }
    }

    /// Update object at index, returning the old value.
    ///
    /// # Arguments
    ///
    /// * `index` - Index of object to update
    /// * `object` - New object data
    ///
    /// # Returns
    ///
    /// `Some(old_object)` if index was valid, `None` otherwise.
    pub fn update(&mut self, index: usize, object: ObjectData) -> Option<ObjectData> {
        if index < self.staging.len() {
            self.mark_dirty(index);
            let old = std::mem::replace(&mut self.staging[index], object);
            Some(old)
        } else {
            None
        }
    }

    /// Remove object at index by swapping with last element.
    ///
    /// This is O(1) but changes the index of the last element.
    /// Returns the removed object, or None if index was invalid.
    ///
    /// # Note
    ///
    /// This invalidates any external references to indices >= the removed
    /// index. Use with caution in systems that track object indices.
    pub fn swap_remove(&mut self, index: usize) -> Option<ObjectData> {
        if index < self.staging.len() {
            let removed = self.staging.swap_remove(index);
            self.count = self.staging.len();

            // Mark both the removed position (now contains swapped element)
            // and adjust dirty range
            if index < self.staging.len() {
                self.mark_dirty(index);
            }
            // If dirty_end was pointing past the new length, adjust it
            if self.dirty_end > self.staging.len() {
                self.dirty_end = self.staging.len();
            }

            Some(removed)
        } else {
            None
        }
    }

    // -------------------------------------------------------------------------
    // Dirty Tracking
    // -------------------------------------------------------------------------

    /// Mark index as needing upload.
    #[inline]
    fn mark_dirty(&mut self, index: usize) {
        self.dirty_start = self.dirty_start.min(index);
        self.dirty_end = self.dirty_end.max(index + 1);
    }

    /// Mark range as needing upload.
    #[inline]
    fn mark_dirty_range(&mut self, start: usize, end: usize) {
        self.dirty_start = self.dirty_start.min(start);
        self.dirty_end = self.dirty_end.max(end);
    }

    /// Mark all objects as dirty (force full upload).
    pub fn mark_all_dirty(&mut self) {
        if !self.staging.is_empty() {
            self.dirty_start = 0;
            self.dirty_end = self.staging.len();
        }
    }

    /// Check if there are pending changes to upload.
    #[inline]
    pub fn is_dirty(&self) -> bool {
        self.dirty_start < self.dirty_end
    }

    /// Get the dirty range (start, end) if any.
    ///
    /// Returns `None` if no dirty data.
    #[inline]
    pub fn dirty_range(&self) -> Option<(usize, usize)> {
        if self.dirty_start < self.dirty_end {
            Some((self.dirty_start, self.dirty_end))
        } else {
            None
        }
    }

    // -------------------------------------------------------------------------
    // GPU Upload
    // -------------------------------------------------------------------------

    /// Upload dirty data to GPU.
    ///
    /// This method:
    /// 1. Resizes the GPU buffer if needed (2x growth)
    /// 2. Uploads only the dirty range to minimize transfer
    /// 3. Clears the dirty state
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device (needed for resize)
    /// * `queue` - The wgpu queue for buffer writes
    ///
    /// # Returns
    ///
    /// `true` if data was uploaded, `false` if nothing was dirty.
    pub fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> bool {
        // Check if resize is needed
        if self.staging.len() > self.capacity {
            let new_capacity = (self.staging.len() * GROWTH_FACTOR).max(MIN_BUFFER_CAPACITY);
            self.resize(device, new_capacity);
        }

        // Upload dirty range
        if self.dirty_start < self.dirty_end && self.dirty_end <= self.staging.len() {
            let offset = (self.dirty_start * ObjectData::SIZE) as u64;
            let data = &self.staging[self.dirty_start..self.dirty_end];
            queue.write_buffer(&self.object_buffer, offset, bytemuck::cast_slice(data));

            // Clear dirty state
            self.dirty_start = usize::MAX;
            self.dirty_end = 0;
            self.resized = false;

            return true;
        }

        false
    }

    /// Resize buffer to new capacity.
    ///
    /// Creates a new GPU buffer with the specified capacity.
    /// All data will be re-uploaded on the next `upload()` call.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device for buffer creation
    /// * `new_capacity` - New capacity (must be >= current count)
    pub fn resize(&mut self, device: &wgpu::Device, new_capacity: usize) {
        let new_capacity = new_capacity.max(self.staging.len()).max(MIN_BUFFER_CAPACITY);

        if new_capacity > self.capacity {
            let new_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: self
                    .label
                    .as_ref()
                    .map(|l| format!("{}_object_buffer", l))
                    .as_deref(),
                size: (new_capacity * ObjectData::SIZE) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.object_buffer = new_buffer;
            self.capacity = new_capacity;
            self.resized = true;

            // Mark all as dirty for re-upload
            if !self.staging.is_empty() {
                self.dirty_start = 0;
                self.dirty_end = self.staging.len();
            }
        }
    }

    /// Reserve capacity for additional objects.
    ///
    /// Ensures the buffer can hold at least `count() + additional` objects
    /// without reallocation.
    pub fn reserve(&mut self, device: &wgpu::Device, additional: usize) {
        let required = self.staging.len() + additional;
        if required > self.capacity {
            let new_capacity = required.max(self.capacity * GROWTH_FACTOR);
            self.resize(device, new_capacity);
        }
        self.staging.reserve(additional);
    }

    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------

    /// Get GPU buffer for binding.
    #[inline]
    pub fn object_buffer(&self) -> &wgpu::Buffer {
        &self.object_buffer
    }

    /// Get buffer binding resource for bind group creation.
    #[inline]
    pub fn buffer_binding(&self) -> wgpu::BufferBinding<'_> {
        self.object_buffer.as_entire_buffer_binding()
    }

    /// Get object count.
    #[inline]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Get capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Check if buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Get buffer size in bytes.
    #[inline]
    pub fn buffer_size(&self) -> u64 {
        (self.capacity * ObjectData::SIZE) as u64
    }

    /// Get used size in bytes.
    #[inline]
    pub fn used_size(&self) -> usize {
        self.count * ObjectData::SIZE
    }

    /// Get the debug label.
    #[inline]
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Check if buffer was resized since last upload.
    #[inline]
    pub fn was_resized(&self) -> bool {
        self.resized
    }

    /// Get objects as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[ObjectData] {
        &self.staging
    }

    // -------------------------------------------------------------------------
    // Iteration
    // -------------------------------------------------------------------------

    /// Iterate over all objects.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &ObjectData> {
        self.staging.iter()
    }

    /// Iterate over all objects with mutable access.
    ///
    /// Note: This marks ALL objects as dirty. For targeted updates,
    /// use `get_mut()` instead.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut ObjectData> {
        self.mark_all_dirty();
        self.staging.iter_mut()
    }

    /// Iterate over objects with their indices.
    #[inline]
    pub fn iter_indexed(&self) -> impl Iterator<Item = (usize, &ObjectData)> {
        self.staging.iter().enumerate()
    }

    // -------------------------------------------------------------------------
    // Bulk Operations
    // -------------------------------------------------------------------------

    /// Clear all objects.
    ///
    /// This removes all objects from the staging buffer but does not
    /// deallocate the GPU buffer. The buffer can be reused.
    pub fn clear(&mut self) {
        self.staging.clear();
        self.count = 0;
        self.dirty_start = usize::MAX;
        self.dirty_end = 0;
    }

    /// Retain only objects matching the predicate.
    ///
    /// Objects are compacted, which may change indices.
    /// Marks all retained objects as dirty.
    pub fn retain<F>(&mut self, predicate: F)
    where
        F: FnMut(&ObjectData) -> bool,
    {
        self.staging.retain(predicate);
        self.count = self.staging.len();
        self.mark_all_dirty();
    }

    /// Update all visible objects' transforms.
    ///
    /// Convenience method for common operation.
    pub fn update_transforms<F>(&mut self, mut transform_fn: F)
    where
        F: FnMut(usize, &mut [[f32; 4]; 4]),
    {
        for (i, obj) in self.staging.iter_mut().enumerate() {
            if obj.is_visible() {
                transform_fn(i, &mut obj.transform);
                self.dirty_start = self.dirty_start.min(i);
                self.dirty_end = self.dirty_end.max(i + 1);
            }
        }
    }
}

impl std::fmt::Debug for SceneDataBuffers {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SceneDataBuffers")
            .field("count", &self.count)
            .field("capacity", &self.capacity)
            .field("label", &self.label)
            .field(
                "dirty_range",
                &if self.dirty_start < self.dirty_end {
                    Some((self.dirty_start, self.dirty_end))
                } else {
                    None
                },
            )
            .field("resized", &self.resized)
            .finish_non_exhaustive()
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_driven::object_data::object_flags;

    // Note: Tests that require wgpu::Device/Queue use a mock or are integration tests.
    // Unit tests focus on CPU-side logic.

    #[test]
    fn test_constants() {
        assert!(DEFAULT_SCENE_CAPACITY >= MIN_BUFFER_CAPACITY);
        assert!(GROWTH_FACTOR >= 2);
    }

    #[test]
    fn test_staging_operations() {
        // Test CPU-side staging operations without GPU
        let mut staging: Vec<ObjectData> = Vec::new();

        // Add objects
        let obj = ObjectData::new().with_mesh(42);
        staging.push(obj);
        assert_eq!(staging.len(), 1);
        assert_eq!(staging[0].mesh_index, 42);

        // Add more
        staging.push(ObjectData::new().with_mesh(43));
        staging.push(ObjectData::new().with_mesh(44));
        assert_eq!(staging.len(), 3);

        // Modify
        staging[1].mesh_index = 100;
        assert_eq!(staging[1].mesh_index, 100);

        // Swap remove
        let removed = staging.swap_remove(0);
        assert_eq!(removed.mesh_index, 42);
        assert_eq!(staging.len(), 2);
        // Index 0 now contains what was index 2
        assert_eq!(staging[0].mesh_index, 44);
    }

    #[test]
    fn test_dirty_tracking_logic() {
        // Test dirty range logic without GPU
        let mut dirty_start = usize::MAX;
        let mut dirty_end = 0usize;

        // Mark dirty at index 5
        dirty_start = dirty_start.min(5);
        dirty_end = dirty_end.max(6);
        assert_eq!(dirty_start, 5);
        assert_eq!(dirty_end, 6);

        // Mark dirty at index 2 (expands range)
        dirty_start = dirty_start.min(2);
        dirty_end = dirty_end.max(3);
        assert_eq!(dirty_start, 2);
        assert_eq!(dirty_end, 6);

        // Mark dirty at index 10 (expands end)
        dirty_start = dirty_start.min(10);
        dirty_end = dirty_end.max(11);
        assert_eq!(dirty_start, 2);
        assert_eq!(dirty_end, 11);

        // Check is_dirty
        assert!(dirty_start < dirty_end);

        // Clear dirty
        dirty_start = usize::MAX;
        dirty_end = 0;
        assert!(dirty_start >= dirty_end);
    }

    #[test]
    fn test_capacity_calculations() {
        let count = 1000;
        let capacity = 512;

        // Growth factor calculation
        let new_capacity = (count * GROWTH_FACTOR).max(MIN_BUFFER_CAPACITY);
        assert_eq!(new_capacity, 2000);

        // Buffer size calculation
        let buffer_size = capacity * ObjectData::SIZE;
        assert_eq!(buffer_size, 512 * 144);
    }

    #[test]
    fn test_object_data_layout() {
        // Verify ObjectData is suitable for GPU upload
        assert_eq!(ObjectData::SIZE, 144);
        assert_eq!(ObjectData::SIZE % 16, 0); // 16-byte aligned

        // Verify bytemuck compatibility
        let obj = ObjectData::new();
        let bytes: &[u8] = bytemuck::bytes_of(&obj);
        assert_eq!(bytes.len(), ObjectData::SIZE);
    }

    #[test]
    fn test_batch_add_range() {
        let objects = vec![
            ObjectData::new().with_mesh(0),
            ObjectData::new().with_mesh(1),
            ObjectData::new().with_mesh(2),
        ];

        let start = 5; // Simulating existing objects
        let end = start + objects.len();

        assert_eq!(end, 8);
        assert_eq!(end - start, 3);
    }

    #[test]
    fn test_retain_logic() {
        let mut objects = vec![
            ObjectData::new().with_mesh(0).with_flags(object_flags::VISIBLE),
            ObjectData::new().with_mesh(1).with_flags(0), // invisible
            ObjectData::new().with_mesh(2).with_flags(object_flags::VISIBLE),
        ];

        objects.retain(|obj| obj.is_visible());
        assert_eq!(objects.len(), 2);
        assert_eq!(objects[0].mesh_index, 0);
        assert_eq!(objects[1].mesh_index, 2);
    }

    #[test]
    fn test_transform_update_iteration() {
        let mut objects = vec![
            ObjectData::new().with_mesh(0),
            ObjectData::new().with_mesh(1),
        ];

        for obj in objects.iter_mut() {
            obj.transform[3][0] = 10.0; // Set X translation
        }

        assert_eq!(objects[0].transform[3][0], 10.0);
        assert_eq!(objects[1].transform[3][0], 10.0);
    }

    #[test]
    fn test_indexed_iteration() {
        let objects = vec![
            ObjectData::new().with_mesh(10),
            ObjectData::new().with_mesh(20),
            ObjectData::new().with_mesh(30),
        ];

        let indexed: Vec<(usize, u32)> = objects
            .iter()
            .enumerate()
            .map(|(i, obj)| (i, obj.mesh_index))
            .collect();

        assert_eq!(indexed, vec![(0, 10), (1, 20), (2, 30)]);
    }
}
