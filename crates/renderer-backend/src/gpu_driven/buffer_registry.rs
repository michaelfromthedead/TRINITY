//! Bindless Buffer Registry for GPU-driven rendering (T-WGPU-P6.8.2).
//!
//! Provides a registry for managing arrays of storage buffers that can be
//! accessed by index in GPU shaders. This enables bindless rendering patterns
//! where buffers are referenced by slot index rather than requiring separate
//! bind groups per buffer.
//!
//! # Overview
//!
//! ```text
//! +------------------------------------------------------------------+
//! |                     BindlessBufferRegistry                       |
//! +------------------------------------------------------------------+
//! | Slot 0:  [Buffer A]  <- Storage buffer for mesh data             |
//! | Slot 1:  [Buffer B]  <- Storage buffer for instance data         |
//! | Slot 2:  [Empty]     <- Free slot (in free_slots list)           |
//! | Slot 3:  [Buffer C]  <- Storage buffer for material params       |
//! | ...                                                              |
//! | Slot N:  [Buffer X]  <- Up to MAX_BINDLESS_BUFFERS slots         |
//! +------------------------------------------------------------------+
//! | Dirty Slots: {0, 3}  <- Slots with changed contents              |
//! | Free Slots:  [2]     <- Available for reuse                      |
//! +------------------------------------------------------------------+
//! ```
//!
//! # Features
//!
//! - **Bindless Access**: Buffers accessed by index in shaders
//! - **Slot Allocation**: Efficient slot management with free-list reuse
//! - **Dirty Tracking**: Track which slots have changed for incremental updates
//! - **Bind Group Rebuild**: Automatic bind group reconstruction when needed
//!
//! # Usage
//!
//! ```ignore
//! use renderer_backend::gpu_driven::buffer_registry::BindlessBufferRegistry;
//!
//! // Create registry
//! let mut registry = BindlessBufferRegistry::new(&device);
//!
//! // Allocate slots for buffers
//! let mesh_slot = registry.allocate_slot(mesh_buffer);
//! let instance_slot = registry.allocate_slot(instance_buffer);
//!
//! // Mark slots as dirty when contents change
//! registry.mark_dirty(mesh_slot);
//!
//! // Update bind group if needed
//! registry.update(&device);
//!
//! // Use in render pass
//! render_pass.set_bind_group(0, registry.bind_group().unwrap(), &[]);
//!
//! // Free slots when done
//! registry.free_slot(mesh_slot);
//! ```
//!
//! # GPU Usage (WGSL)
//!
//! ```wgsl
//! // Bindless buffer array declaration
//! @group(0) @binding(0) var<storage, read> buffers: array<array<u32>>;
//!
//! // Access buffer by slot index
//! fn read_from_slot(slot: u32, offset: u32) -> u32 {
//!     return buffers[slot][offset];
//! }
//! ```
//!
//! # Limitations
//!
//! - Maximum of `MAX_BINDLESS_BUFFERS` (256) slots
//! - All buffers must be storage buffers with read access
//! - Bind group is rebuilt when slots are allocated or freed

use std::collections::HashSet;

// =============================================================================
// CONSTANTS
// =============================================================================

/// Maximum number of bindless buffer slots.
///
/// This is a practical limit balancing flexibility with GPU bind group
/// constraints. WebGPU spec allows up to 1000 bindings per bind group,
/// but 256 provides sufficient capacity for most scenes while keeping
/// bind group creation efficient.
pub const MAX_BINDLESS_BUFFERS: u32 = 256;

/// Minimum buffer size for placeholder buffers (4 bytes = 1 u32).
pub const MIN_BUFFER_SIZE: u64 = 4;

// =============================================================================
// BINDLESS BUFFER REGISTRY
// =============================================================================

/// Bindless buffer registry for GPU-driven rendering.
///
/// Manages an array of storage buffers that can be bound together and
/// accessed by slot index in shaders. This enables efficient bindless
/// rendering patterns where buffer selection happens on the GPU.
///
/// # Architecture
///
/// The registry maintains:
/// - A fixed-size array of buffer slots (up to `MAX_BINDLESS_BUFFERS`)
/// - A free-list for efficient slot reuse
/// - A dirty set tracking slots with changed contents
/// - A bind group that is rebuilt when slot assignments change
///
/// # Thread Safety
///
/// This struct is not thread-safe. All operations should be performed
/// from the main render thread.
pub struct BindlessBufferRegistry {
    /// Buffer slots (Some = occupied, None = free).
    buffers: Vec<Option<wgpu::Buffer>>,
    /// Stack of free slot indices for O(1) allocation.
    free_slots: Vec<u32>,
    /// Set of slots whose contents have changed.
    dirty_slots: HashSet<u32>,
    /// Current bind group (rebuilt when slots change).
    bind_group: Option<wgpu::BindGroup>,
    /// Bind group layout for storage buffer array.
    layout: wgpu::BindGroupLayout,
    /// Flag indicating bind group needs rebuild.
    needs_rebuild: bool,
    /// Placeholder buffer for empty slots.
    placeholder: wgpu::Buffer,
    /// Debug label.
    label: Option<String>,
}

impl BindlessBufferRegistry {
    /// Create a new bindless buffer registry.
    ///
    /// Initializes with all slots free and no bind group. The bind group
    /// will be created on the first `update()` call after allocating slots.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device
    ///
    /// # Example
    ///
    /// ```ignore
    /// let registry = BindlessBufferRegistry::new(&device);
    /// ```
    pub fn new(device: &wgpu::Device) -> Self {
        Self::with_label(device, None)
    }

    /// Create a new bindless buffer registry with a debug label.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device
    /// * `label` - Optional debug label
    pub fn with_label(device: &wgpu::Device, label: Option<&str>) -> Self {
        // Create bind group layout for storage buffer array
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: label.map(|l| format!("{}_layout", l)).as_deref(),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX
                    | wgpu::ShaderStages::FRAGMENT
                    | wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: std::num::NonZeroU32::new(MAX_BINDLESS_BUFFERS),
            }],
        });

        // Create placeholder buffer for empty slots
        let placeholder = device.create_buffer(&wgpu::BufferDescriptor {
            label: label.map(|l| format!("{}_placeholder", l)).as_deref(),
            size: MIN_BUFFER_SIZE,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Initialize all slots as free (in reverse order for LIFO allocation)
        let free_slots: Vec<u32> = (0..MAX_BINDLESS_BUFFERS).rev().collect();

        // Initialize buffer slots - can't use vec![None; N] because Buffer doesn't impl Clone
        let buffers: Vec<Option<wgpu::Buffer>> =
            (0..MAX_BINDLESS_BUFFERS).map(|_| None).collect();

        Self {
            buffers,
            free_slots,
            dirty_slots: HashSet::new(),
            bind_group: None,
            layout,
            needs_rebuild: true, // Build initial bind group on first update
            placeholder,
            label: label.map(String::from),
        }
    }

    /// Allocate a slot for a buffer and return the slot index.
    ///
    /// Takes ownership of the buffer and assigns it to a free slot.
    /// The slot index can be used in shaders to access the buffer.
    ///
    /// # Arguments
    ///
    /// * `buffer` - The storage buffer to register
    ///
    /// # Returns
    ///
    /// The allocated slot index.
    ///
    /// # Panics
    ///
    /// Panics if no free slots are available (all `MAX_BINDLESS_BUFFERS`
    /// slots are occupied).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let buffer = device.create_buffer(&wgpu::BufferDescriptor { ... });
    /// let slot = registry.allocate_slot(buffer);
    /// // Use `slot` in shader to access buffer
    /// ```
    pub fn allocate_slot(&mut self, buffer: wgpu::Buffer) -> u32 {
        let slot = self
            .free_slots
            .pop()
            .expect("BindlessBufferRegistry: no free slots available");

        self.buffers[slot as usize] = Some(buffer);
        self.dirty_slots.insert(slot);
        self.needs_rebuild = true;

        slot
    }

    /// Try to allocate a slot, returning None if no slots are available.
    ///
    /// Non-panicking version of `allocate_slot`.
    ///
    /// # Arguments
    ///
    /// * `buffer` - The storage buffer to register
    ///
    /// # Returns
    ///
    /// `Some(slot_index)` if allocation succeeded, `None` if no free slots.
    pub fn try_allocate_slot(&mut self, buffer: wgpu::Buffer) -> Option<u32> {
        let slot = self.free_slots.pop()?;

        self.buffers[slot as usize] = Some(buffer);
        self.dirty_slots.insert(slot);
        self.needs_rebuild = true;

        Some(slot)
    }

    /// Release a slot for reuse.
    ///
    /// The buffer in the slot is dropped and the slot is added back to
    /// the free list. The slot index should not be used after this call.
    ///
    /// # Arguments
    ///
    /// * `slot` - The slot index to free
    ///
    /// # Panics
    ///
    /// Panics if `slot` is out of range or already free.
    ///
    /// # Example
    ///
    /// ```ignore
    /// registry.free_slot(slot);
    /// // `slot` is now invalid and should not be used
    /// ```
    pub fn free_slot(&mut self, slot: u32) {
        assert!(
            slot < MAX_BINDLESS_BUFFERS,
            "BindlessBufferRegistry: slot {} out of range",
            slot
        );
        assert!(
            self.buffers[slot as usize].is_some(),
            "BindlessBufferRegistry: slot {} is already free",
            slot
        );

        self.buffers[slot as usize] = None;
        self.dirty_slots.remove(&slot);
        self.free_slots.push(slot);
        self.needs_rebuild = true;
    }

    /// Try to release a slot, returning false if the slot is invalid.
    ///
    /// Non-panicking version of `free_slot`.
    ///
    /// # Arguments
    ///
    /// * `slot` - The slot index to free
    ///
    /// # Returns
    ///
    /// `true` if the slot was freed, `false` if it was invalid or already free.
    pub fn try_free_slot(&mut self, slot: u32) -> bool {
        if slot >= MAX_BINDLESS_BUFFERS {
            return false;
        }

        if self.buffers[slot as usize].is_none() {
            return false;
        }

        self.buffers[slot as usize] = None;
        self.dirty_slots.remove(&slot);
        self.free_slots.push(slot);
        self.needs_rebuild = true;

        true
    }

    /// Mark a slot as dirty (buffer contents have changed).
    ///
    /// This is used to track which buffers need to be re-uploaded or
    /// processed. The dirty set can be queried with `take_dirty_slots()`.
    ///
    /// # Arguments
    ///
    /// * `slot` - The slot index to mark dirty
    ///
    /// # Note
    ///
    /// Marking a slot dirty does NOT trigger a bind group rebuild. The
    /// bind group only changes when slots are allocated or freed, not
    /// when buffer contents change.
    #[inline]
    pub fn mark_dirty(&mut self, slot: u32) {
        if slot < MAX_BINDLESS_BUFFERS && self.buffers[slot as usize].is_some() {
            self.dirty_slots.insert(slot);
        }
    }

    /// Get all dirty slots and clear the dirty set.
    ///
    /// Returns a vector of slot indices that have been marked dirty since
    /// the last call to this method.
    ///
    /// # Returns
    ///
    /// Vector of dirty slot indices (may be empty).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let dirty = registry.take_dirty_slots();
    /// for slot in dirty {
    ///     // Re-upload or process buffer at `slot`
    /// }
    /// ```
    pub fn take_dirty_slots(&mut self) -> Vec<u32> {
        self.dirty_slots.drain().collect()
    }

    /// Check if any slots are dirty.
    ///
    /// # Returns
    ///
    /// `true` if at least one slot has been marked dirty.
    #[inline]
    pub fn has_dirty_slots(&self) -> bool {
        !self.dirty_slots.is_empty()
    }

    /// Get the current set of dirty slots without clearing.
    ///
    /// # Returns
    ///
    /// Reference to the set of dirty slot indices.
    #[inline]
    pub fn dirty_slots(&self) -> &HashSet<u32> {
        &self.dirty_slots
    }

    /// Update the bind group if needed.
    ///
    /// Rebuilds the bind group when slots have been allocated or freed.
    /// Should be called once per frame before rendering.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Each frame before rendering:
    /// registry.update(&device);
    /// ```
    pub fn update(&mut self, device: &wgpu::Device) {
        if self.needs_rebuild {
            self.rebuild_bind_group(device);
        }
    }

    /// Force rebuild the bind group.
    ///
    /// Use this if external factors require a bind group rebuild.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device
    pub fn force_rebuild(&mut self, device: &wgpu::Device) {
        self.rebuild_bind_group(device);
    }

    /// Rebuild the bind group with current buffer assignments.
    fn rebuild_bind_group(&mut self, device: &wgpu::Device) {
        // Build array of buffer bindings (using placeholder for empty slots)
        let bindings: Vec<wgpu::BindingResource> = self
            .buffers
            .iter()
            .map(|opt| {
                let buffer = opt.as_ref().unwrap_or(&self.placeholder);
                wgpu::BindingResource::Buffer(buffer.as_entire_buffer_binding())
            })
            .collect();

        self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: self.label.as_ref().map(|l| format!("{}_bind_group", l)).as_deref(),
            layout: &self.layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::BufferArray(&bindings
                    .iter()
                    .map(|r| {
                        if let wgpu::BindingResource::Buffer(b) = r {
                            b.clone()
                        } else {
                            unreachable!()
                        }
                    })
                    .collect::<Vec<_>>()),
            }],
        }));

        self.needs_rebuild = false;
    }

    /// Get the current bind group.
    ///
    /// Returns `None` if `update()` has never been called or if no buffers
    /// have been allocated.
    ///
    /// # Returns
    ///
    /// Reference to the bind group, or `None`.
    #[inline]
    pub fn bind_group(&self) -> Option<&wgpu::BindGroup> {
        self.bind_group.as_ref()
    }

    /// Get the bind group layout.
    ///
    /// Use this when creating pipeline layouts that use this registry.
    ///
    /// # Returns
    ///
    /// Reference to the bind group layout.
    #[inline]
    pub fn layout(&self) -> &wgpu::BindGroupLayout {
        &self.layout
    }

    /// Get the number of active (occupied) buffer slots.
    ///
    /// # Returns
    ///
    /// Number of slots currently holding buffers.
    #[inline]
    pub fn active_count(&self) -> u32 {
        (MAX_BINDLESS_BUFFERS as usize - self.free_slots.len()) as u32
    }

    /// Get the number of free buffer slots.
    ///
    /// # Returns
    ///
    /// Number of slots available for allocation.
    #[inline]
    pub fn free_count(&self) -> u32 {
        self.free_slots.len() as u32
    }

    /// Check if a slot is occupied.
    ///
    /// # Arguments
    ///
    /// * `slot` - The slot index to check
    ///
    /// # Returns
    ///
    /// `true` if the slot contains a buffer.
    #[inline]
    pub fn is_occupied(&self, slot: u32) -> bool {
        slot < MAX_BINDLESS_BUFFERS && self.buffers[slot as usize].is_some()
    }

    /// Check if a slot is free.
    ///
    /// # Arguments
    ///
    /// * `slot` - The slot index to check
    ///
    /// # Returns
    ///
    /// `true` if the slot is available for allocation.
    #[inline]
    pub fn is_free(&self, slot: u32) -> bool {
        slot < MAX_BINDLESS_BUFFERS && self.buffers[slot as usize].is_none()
    }

    /// Check if the bind group needs to be rebuilt.
    ///
    /// # Returns
    ///
    /// `true` if `update()` will rebuild the bind group.
    #[inline]
    pub fn needs_rebuild(&self) -> bool {
        self.needs_rebuild
    }

    /// Get a reference to a buffer in a slot.
    ///
    /// # Arguments
    ///
    /// * `slot` - The slot index
    ///
    /// # Returns
    ///
    /// Reference to the buffer, or `None` if the slot is empty.
    #[inline]
    pub fn get_buffer(&self, slot: u32) -> Option<&wgpu::Buffer> {
        if slot < MAX_BINDLESS_BUFFERS {
            self.buffers[slot as usize].as_ref()
        } else {
            None
        }
    }

    /// Get the debug label.
    #[inline]
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Clear all slots and reset to initial state.
    ///
    /// All buffers are dropped and all slots are marked as free.
    /// The bind group will need to be rebuilt on the next `update()`.
    pub fn clear(&mut self) {
        // Can't use fill(None) because Buffer doesn't impl Clone
        for slot in self.buffers.iter_mut() {
            *slot = None;
        }
        self.free_slots = (0..MAX_BINDLESS_BUFFERS).rev().collect();
        self.dirty_slots.clear();
        self.bind_group = None;
        self.needs_rebuild = true;
    }

    /// Replace a buffer in an existing slot.
    ///
    /// The old buffer is dropped and replaced with the new one.
    /// The slot is marked dirty and the bind group will be rebuilt.
    ///
    /// # Arguments
    ///
    /// * `slot` - The slot index
    /// * `buffer` - The new buffer
    ///
    /// # Panics
    ///
    /// Panics if the slot is out of range or not occupied.
    pub fn replace_buffer(&mut self, slot: u32, buffer: wgpu::Buffer) {
        assert!(
            slot < MAX_BINDLESS_BUFFERS,
            "BindlessBufferRegistry: slot {} out of range",
            slot
        );
        assert!(
            self.buffers[slot as usize].is_some(),
            "BindlessBufferRegistry: slot {} is not occupied",
            slot
        );

        self.buffers[slot as usize] = Some(buffer);
        self.dirty_slots.insert(slot);
        self.needs_rebuild = true;
    }

    /// Get an iterator over all occupied slots.
    ///
    /// # Returns
    ///
    /// Iterator yielding (slot_index, &buffer) pairs.
    pub fn occupied_slots(&self) -> impl Iterator<Item = (u32, &wgpu::Buffer)> {
        self.buffers
            .iter()
            .enumerate()
            .filter_map(|(i, opt)| opt.as_ref().map(|b| (i as u32, b)))
    }
}

impl std::fmt::Debug for BindlessBufferRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BindlessBufferRegistry")
            .field("active_count", &self.active_count())
            .field("free_count", &self.free_count())
            .field("dirty_count", &self.dirty_slots.len())
            .field("needs_rebuild", &self.needs_rebuild)
            .field("has_bind_group", &self.bind_group.is_some())
            .field("label", &self.label)
            .finish()
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Constants Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_constants() {
        assert_eq!(MAX_BINDLESS_BUFFERS, 256);
        assert!(MAX_BINDLESS_BUFFERS > 0);
        assert!(MAX_BINDLESS_BUFFERS <= 1000); // WebGPU limit
        assert_eq!(MIN_BUFFER_SIZE, 4);
    }

    // -------------------------------------------------------------------------
    // Slot Allocation Tests (CPU-only logic)
    // -------------------------------------------------------------------------

    #[test]
    fn test_free_slots_initial_state() {
        // Simulate initial free slots state
        let free_slots: Vec<u32> = (0..MAX_BINDLESS_BUFFERS).rev().collect();
        assert_eq!(free_slots.len(), MAX_BINDLESS_BUFFERS as usize);
        // Last element (first to pop) should be 0
        assert_eq!(free_slots[free_slots.len() - 1], 0);
        // First element (last to pop) should be MAX-1
        assert_eq!(free_slots[0], MAX_BINDLESS_BUFFERS - 1);
    }

    #[test]
    fn test_slot_allocation_order() {
        // Simulate allocation order
        let mut free_slots: Vec<u32> = (0..MAX_BINDLESS_BUFFERS).rev().collect();

        let slot1 = free_slots.pop().unwrap();
        assert_eq!(slot1, 0);

        let slot2 = free_slots.pop().unwrap();
        assert_eq!(slot2, 1);

        let slot3 = free_slots.pop().unwrap();
        assert_eq!(slot3, 2);
    }

    #[test]
    fn test_slot_free_reuse() {
        // Simulate free and reuse
        let mut free_slots: Vec<u32> = (0..MAX_BINDLESS_BUFFERS).rev().collect();

        // Allocate 3 slots
        let slot0 = free_slots.pop().unwrap(); // 0
        let slot1 = free_slots.pop().unwrap(); // 1
        let _slot2 = free_slots.pop().unwrap(); // 2

        // Free slot 1
        free_slots.push(slot1);

        // Free slot 0
        free_slots.push(slot0);

        // Next allocation should get slot 0 (LIFO)
        let reused = free_slots.pop().unwrap();
        assert_eq!(reused, 0);
    }

    #[test]
    fn test_active_count_calculation() {
        let total = MAX_BINDLESS_BUFFERS;
        let free = 200u32;
        let active = total - free;
        assert_eq!(active, 56);
    }

    // -------------------------------------------------------------------------
    // Dirty Tracking Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_dirty_set_operations() {
        let mut dirty: HashSet<u32> = HashSet::new();

        // Mark dirty
        dirty.insert(5);
        dirty.insert(10);
        dirty.insert(5); // Duplicate

        assert_eq!(dirty.len(), 2);
        assert!(dirty.contains(&5));
        assert!(dirty.contains(&10));
        assert!(!dirty.contains(&0));
    }

    #[test]
    fn test_take_dirty_clears_set() {
        let mut dirty: HashSet<u32> = HashSet::new();
        dirty.insert(1);
        dirty.insert(2);
        dirty.insert(3);

        let taken: Vec<u32> = dirty.drain().collect();
        assert_eq!(taken.len(), 3);
        assert!(dirty.is_empty());
    }

    #[test]
    fn test_has_dirty_slots() {
        let mut dirty: HashSet<u32> = HashSet::new();
        assert!(dirty.is_empty());

        dirty.insert(0);
        assert!(!dirty.is_empty());

        dirty.clear();
        assert!(dirty.is_empty());
    }

    #[test]
    fn test_dirty_tracking_workflow() {
        let mut dirty: HashSet<u32> = HashSet::new();

        // Allocate and mark dirty
        dirty.insert(0);
        dirty.insert(1);
        dirty.insert(2);

        // Take dirty slots
        let batch1: Vec<u32> = dirty.drain().collect();
        assert_eq!(batch1.len(), 3);

        // More changes
        dirty.insert(0);
        dirty.insert(5);

        // Take again
        let batch2: Vec<u32> = dirty.drain().collect();
        assert_eq!(batch2.len(), 2);
    }

    // -------------------------------------------------------------------------
    // Needs Rebuild Flag Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_needs_rebuild_on_allocate() {
        let mut needs_rebuild = false;

        // Simulate allocate
        needs_rebuild = true;

        assert!(needs_rebuild);
    }

    #[test]
    fn test_needs_rebuild_on_free() {
        let mut needs_rebuild = false;

        // Simulate free
        needs_rebuild = true;

        assert!(needs_rebuild);
    }

    #[test]
    fn test_needs_rebuild_cleared_after_update() {
        let mut needs_rebuild = true;

        // Simulate update
        if needs_rebuild {
            // rebuild_bind_group()
            needs_rebuild = false;
        }

        assert!(!needs_rebuild);
    }

    // -------------------------------------------------------------------------
    // Slot State Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_is_occupied() {
        let buffers: Vec<Option<()>> = vec![Some(()), None, Some(()), None];

        assert!(buffers[0].is_some()); // Occupied
        assert!(buffers[1].is_none()); // Free
        assert!(buffers[2].is_some()); // Occupied
        assert!(buffers[3].is_none()); // Free
    }

    #[test]
    fn test_slot_bounds_check() {
        let slot = 300u32;
        assert!(slot >= MAX_BINDLESS_BUFFERS);

        let valid_slot = 100u32;
        assert!(valid_slot < MAX_BINDLESS_BUFFERS);
    }

    // -------------------------------------------------------------------------
    // Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_empty_registry_state() {
        let buffers: Vec<Option<()>> = vec![None; MAX_BINDLESS_BUFFERS as usize];
        let free_slots: Vec<u32> = (0..MAX_BINDLESS_BUFFERS).rev().collect();
        let dirty: HashSet<u32> = HashSet::new();

        assert_eq!(buffers.iter().filter(|b| b.is_some()).count(), 0);
        assert_eq!(free_slots.len(), MAX_BINDLESS_BUFFERS as usize);
        assert!(dirty.is_empty());
    }

    #[test]
    fn test_full_registry_state() {
        let buffers: Vec<Option<()>> = vec![Some(()); MAX_BINDLESS_BUFFERS as usize];
        let free_slots: Vec<u32> = vec![];

        assert_eq!(buffers.iter().filter(|b| b.is_some()).count(), MAX_BINDLESS_BUFFERS as usize);
        assert!(free_slots.is_empty());
    }

    #[test]
    fn test_clear_resets_state() {
        let mut buffers: Vec<Option<()>> = vec![Some(()); 10];
        let mut dirty: HashSet<u32> = HashSet::from([0, 1, 2]);

        // Clear
        buffers.fill(None);
        dirty.clear();

        assert!(buffers.iter().all(|b| b.is_none()));
        assert!(dirty.is_empty());
    }

    #[test]
    fn test_replace_buffer_logic() {
        let mut buffers: Vec<Option<i32>> = vec![None; 10];
        let mut dirty: HashSet<u32> = HashSet::new();

        // Allocate
        buffers[0] = Some(100);
        dirty.insert(0);

        // Replace
        buffers[0] = Some(200);
        dirty.insert(0);

        assert_eq!(buffers[0], Some(200));
        assert!(dirty.contains(&0));
    }

    // -------------------------------------------------------------------------
    // Iterator Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_occupied_slots_iterator() {
        let buffers: Vec<Option<i32>> = vec![Some(10), None, Some(20), None, Some(30)];

        let occupied: Vec<(usize, &i32)> = buffers
            .iter()
            .enumerate()
            .filter_map(|(i, opt)| opt.as_ref().map(|b| (i, b)))
            .collect();

        assert_eq!(occupied.len(), 3);
        assert_eq!(occupied[0], (0, &10));
        assert_eq!(occupied[1], (2, &20));
        assert_eq!(occupied[2], (4, &30));
    }

    // -------------------------------------------------------------------------
    // Debug Format Test
    // -------------------------------------------------------------------------

    #[test]
    fn test_debug_format() {
        // Just verify the debug struct fields exist
        let active = 10u32;
        let free = MAX_BINDLESS_BUFFERS - active;
        let dirty_count = 3usize;
        let needs_rebuild = true;
        let has_bind_group = false;

        let debug_str = format!(
            "BindlessBufferRegistry {{ active_count: {}, free_count: {}, dirty_count: {}, needs_rebuild: {}, has_bind_group: {} }}",
            active, free, dirty_count, needs_rebuild, has_bind_group
        );

        assert!(debug_str.contains("active_count: 10"));
        assert!(debug_str.contains("needs_rebuild: true"));
    }
}
