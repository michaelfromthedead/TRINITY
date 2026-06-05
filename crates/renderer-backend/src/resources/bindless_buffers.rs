//! Bindless buffer array management for TRINITY.
//!
//! This module provides a registry for bindless buffer rendering, allowing shaders
//! to access many storage buffers via index rather than individual bind groups.
//! Uses wgpu's `BUFFER_BINDING_ARRAY` feature for variable-count buffer bindings.
//!
//! # Overview
//!
//! Bindless buffer binding eliminates the need to switch bind groups when accessing
//! different buffers. Instead, all storage buffers are bound in a single large array,
//! and shaders access them by index:
//!
//! ```wgsl
//! @group(3) @binding(1) var<storage, read> buffers: binding_array<array<u32>>;
//!
//! @compute @workgroup_size(64)
//! fn main(@builtin(global_invocation_id) id: vec3<u32>) {
//!     let data = buffers[buffer_index][id.x];
//! }
//! ```
//!
//! # Architecture
//!
//! ```text
//! BufferRegistry
//!   +-- buffers: Vec<Option<Arc<Buffer>>>        // Sparse array of registered buffers
//!   +-- free_slots: Vec<u32>                     // Recycled indices (free list)
//!   +-- dirty_slots: HashSet<u32>                // Slots modified since last bind group
//!   +-- bind_group: Option<BindGroup>            // Cached, rebuilt when dirty
//!   +-- dirty: bool                              // Rebuild needed?
//!   +-- max_buffers: u32                         // Capacity limit
//! ```
//!
//! # Dirty Range Tracking
//!
//! Unlike textures, buffer contents can change frequently. The registry tracks which
//! slots have been modified via `mark_dirty(slot)`, allowing efficient partial updates:
//!
//! ```no_run
//! use renderer_backend::resources::bindless_buffers::BufferRegistry;
//!
//! # fn example() {
//! let mut registry = BufferRegistry::new(1024);
//! // ... register buffers ...
//! // registry.mark_dirty(slot); // After buffer content changes
//! // for dirty_slot in registry.dirty_slots() { ... }
//! // registry.clear_dirty();
//! # }
//! ```
//!
//! # Thread Safety
//!
//! The registry is `Send + Sync`. Registration and unregistration are thread-safe,
//! but bind group updates require exclusive access to ensure GPU synchronization.
//!
//! # Example
//!
//! ```no_run
//! use renderer_backend::resources::bindless_buffers::{BufferRegistry, supports_bindless_buffers};
//! use std::sync::Arc;
//!
//! # fn example(device: &wgpu::Device, buffer: wgpu::Buffer) {
//! // Check feature support
//! if !supports_bindless_buffers(device) {
//!     println!("Bindless buffers not supported, falling back to traditional binding");
//!     return;
//! }
//!
//! // Create registry with default capacity
//! let mut registry = BufferRegistry::new(1024);
//!
//! // Register buffers and get slot indices for shader access
//! let slot = registry.register(Arc::new(buffer)).unwrap();
//! println!("Buffer registered at slot {}", slot.index());
//!
//! // When done with a buffer, unregister to recycle the slot
//! registry.unregister(slot);
//! # }
//! ```
//!
//! # TRINITY Integration
//!
//! The bindless registry uses bind group index 3 (BINDLESS) with binding 1 for buffers
//! (binding 0 is used for textures). Use with `TrinityLayoutBuilder::bindless()` for
//! pipeline layout creation.
//!
//! # Feature Requirements
//!
//! - `BUFFER_BINDING_ARRAY` - Required for buffer array binding
//! - `STORAGE_RESOURCE_BINDING_ARRAY` - Optional, enables storage buffer arrays
//! - `PARTIALLY_BOUND_BINDING_ARRAY` - Optional, allows slots to be unbound

use log::{debug, trace, warn};
use std::collections::HashSet;
use std::num::NonZeroU32;
use std::sync::Arc;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferBindingType, Device,
    Features, Limits, ShaderStages,
};

// ============================================================================
// Constants
// ============================================================================

/// Default maximum number of buffers in the bindless registry.
pub const DEFAULT_MAX_BUFFERS: u32 = 1024;

/// Minimum buffers to allocate for bindless storage.
pub const MIN_BINDLESS_BUFFERS: u32 = 16;

/// Maximum buffers per shader stage (conservative limit).
/// Actual limit depends on device, use `max_bindless_buffers()` for runtime query.
pub const MAX_BINDLESS_BUFFERS_CONSERVATIVE: u32 = 16384;

/// Default bind group index for bindless resources (TRINITY convention).
pub const BINDLESS_BIND_GROUP_INDEX: u32 = 3;

/// Default binding index for buffers within the bindless bind group.
/// Textures use binding 0, buffers use binding 1.
pub const BINDLESS_BUFFER_BINDING: u32 = 1;

// ============================================================================
// Feature Detection
// ============================================================================

/// Checks if the device supports bindless buffer arrays.
///
/// Returns `true` if the device has the `BUFFER_BINDING_ARRAY` feature enabled,
/// which is required for variable-count buffer binding.
///
/// # Arguments
///
/// * `device` - The wgpu device to query
///
/// # Example
///
/// ```no_run
/// use renderer_backend::resources::bindless_buffers::supports_bindless_buffers;
///
/// # fn example(device: &wgpu::Device) {
/// if supports_bindless_buffers(device) {
///     println!("Bindless buffers supported!");
/// } else {
///     println!("Falling back to traditional buffer binding");
/// }
/// # }
/// ```
pub fn supports_bindless_buffers(device: &Device) -> bool {
    device.features().contains(Features::BUFFER_BINDING_ARRAY)
}

/// Checks if the device supports storage buffer arrays in bindless mode.
///
/// Storage buffer arrays allow accessing multiple storage buffers via indices.
/// This is an optional enhancement over basic buffer binding arrays.
///
/// # Arguments
///
/// * `device` - The wgpu device to query
pub fn supports_storage_buffer_array(device: &Device) -> bool {
    device
        .features()
        .contains(Features::STORAGE_RESOURCE_BINDING_ARRAY)
}

/// Checks if the device supports non-uniform indexing of buffer arrays.
///
/// Non-uniform indexing allows different shader invocations in the same workgroup
/// to access different buffer indices. Without this feature, all invocations must
/// access the same index.
///
/// # Arguments
///
/// * `device` - The wgpu device to query
pub fn supports_non_uniform_buffer_indexing(device: &Device) -> bool {
    device.features().contains(
        Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
    )
}

/// Checks if the device supports partially bound binding arrays.
///
/// When enabled, not all slots in the buffer array need to be bound. This allows
/// for more efficient memory usage when the actual buffer count is less than the
/// array capacity.
///
/// # Arguments
///
/// * `device` - The wgpu device to query
pub fn supports_partially_bound(device: &Device) -> bool {
    device
        .features()
        .contains(Features::PARTIALLY_BOUND_BINDING_ARRAY)
}

/// Returns the maximum number of buffers supported in a bindless array.
///
/// This queries the device's `max_storage_buffers_per_shader_stage` limit and
/// returns the minimum of that limit and the conservative maximum.
///
/// # Arguments
///
/// * `device` - The wgpu device to query
///
/// # Returns
///
/// Maximum number of storage buffers that can be bound in a single array.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::resources::bindless_buffers::max_bindless_buffers;
///
/// # fn example(device: &wgpu::Device) {
/// let max = max_bindless_buffers(device);
/// println!("Can bind up to {} buffers", max);
/// # }
/// ```
pub fn max_bindless_buffers(device: &Device) -> u32 {
    let limits = device.limits();
    limits
        .max_storage_buffers_per_shader_stage
        .min(MAX_BINDLESS_BUFFERS_CONSERVATIVE)
}

/// Returns the maximum number of buffers from device limits.
///
/// This is a lower-level query that takes limits directly, useful for
/// pre-creation validation.
///
/// # Arguments
///
/// * `limits` - The device limits to query
pub fn max_bindless_buffers_from_limits(limits: &Limits) -> u32 {
    limits
        .max_storage_buffers_per_shader_stage
        .min(MAX_BINDLESS_BUFFERS_CONSERVATIVE)
}

/// Bindless buffer feature requirements for device creation.
///
/// Returns the wgpu features needed for basic bindless buffer support.
pub fn bindless_buffer_required_features() -> Features {
    Features::BUFFER_BINDING_ARRAY
}

/// Bindless buffer feature requirements for optimal performance.
///
/// Returns the wgpu features needed for full bindless buffer support
/// including non-uniform indexing and partial binding.
pub fn bindless_buffer_optimal_features() -> Features {
    Features::BUFFER_BINDING_ARRAY
        | Features::STORAGE_RESOURCE_BINDING_ARRAY
        | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
        | Features::PARTIALLY_BOUND_BINDING_ARRAY
}

// ============================================================================
// BufferSlot
// ============================================================================

/// A handle to a registered buffer in the bindless registry.
///
/// This is a lightweight handle that can be passed to shaders as an index
/// for buffer array access. The slot remains valid until `unregister()` is called.
///
/// # Example
///
/// ```
/// use renderer_backend::resources::bindless_buffers::BufferSlot;
///
/// // Create a slot (normally obtained from BufferRegistry::register)
/// let slot = BufferSlot::new(42);
/// assert_eq!(slot.index(), 42);
///
/// // Convert to/from u32 for shader uniform passing
/// let index: u32 = slot.into();
/// let slot_again = BufferSlot::from(index);
/// assert_eq!(slot_again, slot);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferSlot(u32);

impl BufferSlot {
    /// Creates a new buffer slot with the given index.
    ///
    /// # Arguments
    ///
    /// * `index` - The slot index in the buffer array
    #[inline]
    pub const fn new(index: u32) -> Self {
        Self(index)
    }

    /// Returns the slot index for shader access.
    ///
    /// This index can be used directly in shaders to access the buffer:
    /// ```wgsl
    /// let data = buffers[slot_index][element];
    /// ```
    #[inline]
    pub const fn index(&self) -> u32 {
        self.0
    }

    /// Creates an invalid slot marker.
    ///
    /// This can be used as a sentinel value for "no buffer" scenarios.
    #[inline]
    pub const fn invalid() -> Self {
        Self(u32::MAX)
    }

    /// Checks if the slot is invalid (sentinel value).
    #[inline]
    pub const fn is_invalid(&self) -> bool {
        self.0 == u32::MAX
    }
}

impl From<u32> for BufferSlot {
    #[inline]
    fn from(index: u32) -> Self {
        Self(index)
    }
}

impl From<BufferSlot> for u32 {
    #[inline]
    fn from(slot: BufferSlot) -> Self {
        slot.0
    }
}

impl std::fmt::Display for BufferSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_invalid() {
            write!(f, "BufferSlot(INVALID)")
        } else {
            write!(f, "BufferSlot({})", self.0)
        }
    }
}

// ============================================================================
// BindlessBufferError
// ============================================================================

/// Errors that can occur during bindless buffer operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BindlessBufferError {
    /// The device does not support bindless buffers (BUFFER_BINDING_ARRAY feature missing).
    UnsupportedFeature,

    /// The buffer registry is full and cannot accept more buffers.
    RegistryFull {
        /// Current capacity of the registry.
        capacity: u32,
    },

    /// The specified buffer slot is invalid or has already been unregistered.
    InvalidSlot(BufferSlot),

    /// The requested buffer count exceeds the device limit.
    ExceedsDeviceLimit {
        /// Number of buffers requested.
        requested: u32,
        /// Maximum supported by the device.
        max: u32,
    },

    /// The bind group layout is incompatible with the registry.
    IncompatibleLayout,

    /// No buffers registered - cannot create empty bind group.
    EmptyRegistry,
}

impl std::fmt::Display for BindlessBufferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BindlessBufferError::UnsupportedFeature => {
                write!(
                    f,
                    "bindless buffers not supported: BUFFER_BINDING_ARRAY feature missing"
                )
            }
            BindlessBufferError::RegistryFull { capacity } => {
                write!(f, "bindless buffer registry full (capacity: {})", capacity)
            }
            BindlessBufferError::InvalidSlot(slot) => {
                write!(f, "invalid buffer slot: {}", slot)
            }
            BindlessBufferError::ExceedsDeviceLimit { requested, max } => {
                write!(
                    f,
                    "requested {} buffers exceeds device limit of {}",
                    requested, max
                )
            }
            BindlessBufferError::IncompatibleLayout => {
                write!(f, "bind group layout is incompatible with buffer registry")
            }
            BindlessBufferError::EmptyRegistry => {
                write!(f, "cannot create bind group for empty buffer registry")
            }
        }
    }
}

impl std::error::Error for BindlessBufferError {}

// ============================================================================
// BufferRegistry
// ============================================================================

/// A registry for managing bindless storage buffers.
///
/// The registry maintains a sparse array of buffers that can be bound
/// as a buffer array for shader access. It handles slot allocation, recycling,
/// dirty tracking, and bind group creation.
///
/// # Lifecycle
///
/// 1. Create registry with capacity: `BufferRegistry::new(1024)`
/// 2. Register buffers: `let slot = registry.register(buffer)?`
/// 3. Mark dirty when content changes: `registry.mark_dirty(slot)`
/// 4. Update bind group: `registry.update_bind_group(device, layout)`
/// 5. Use bind group in render/compute pass
/// 6. Unregister when done: `registry.unregister(slot)`
///
/// # Dirty Tracking
///
/// Unlike textures, buffer contents can change frequently. The registry tracks
/// which slots have been modified to allow efficient partial updates:
///
/// ```no_run
/// use renderer_backend::resources::bindless_buffers::BufferRegistry;
/// use std::sync::Arc;
///
/// # fn example(registry: &mut BufferRegistry, slot: renderer_backend::resources::bindless_buffers::BufferSlot) {
/// // After updating buffer contents
/// registry.mark_dirty(slot);
///
/// // Check which slots changed
/// for dirty_slot in registry.dirty_slots() {
///     println!("Slot {} was modified", dirty_slot.index());
/// }
///
/// // Clear dirty state after processing
/// registry.clear_dirty();
/// # }
/// ```
///
/// # Thread Safety
///
/// The registry is `Send + Sync`. Registration and unregistration are thread-safe,
/// but bind group updates require exclusive access to ensure GPU synchronization.
pub struct BufferRegistry {
    /// Sparse array of registered buffers.
    /// `None` indicates an empty/recycled slot.
    buffers: Vec<Option<Arc<Buffer>>>,

    /// Stack of free slot indices for efficient recycling.
    free_slots: Vec<u32>,

    /// Set of dirty slot indices (modified since last bind group update).
    dirty_slots: HashSet<u32>,

    /// Cached bind group, rebuilt when `dirty` is true.
    bind_group: Option<BindGroup>,

    /// True if buffer array changed since last bind group creation.
    dirty: bool,

    /// Maximum number of buffers this registry can hold.
    max_buffers: u32,

    /// Number of currently registered buffers.
    registered_count: u32,
}

impl BufferRegistry {
    /// Creates a new buffer registry with the specified capacity.
    ///
    /// # Arguments
    ///
    /// * `max_buffers` - Maximum number of buffers the registry can hold.
    ///   Clamped to `MIN_BINDLESS_BUFFERS..=MAX_BINDLESS_BUFFERS_CONSERVATIVE`.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::resources::bindless_buffers::BufferRegistry;
    ///
    /// let registry = BufferRegistry::new(1024);
    /// assert_eq!(registry.capacity(), 1024);
    /// assert_eq!(registry.count(), 0);
    /// ```
    pub fn new(max_buffers: u32) -> Self {
        let clamped = max_buffers
            .max(MIN_BINDLESS_BUFFERS)
            .min(MAX_BINDLESS_BUFFERS_CONSERVATIVE);

        if clamped != max_buffers {
            debug!(
                "BufferRegistry capacity clamped from {} to {}",
                max_buffers, clamped
            );
        }

        Self {
            buffers: Vec::with_capacity(clamped as usize),
            free_slots: Vec::new(),
            dirty_slots: HashSet::new(),
            bind_group: None,
            dirty: true,
            max_buffers: clamped,
            registered_count: 0,
        }
    }

    /// Creates a new buffer registry with validation against device limits.
    ///
    /// # Arguments
    ///
    /// * `max_buffers` - Desired maximum buffer count
    /// * `device` - Device to validate limits against
    ///
    /// # Errors
    ///
    /// Returns `BindlessBufferError::UnsupportedFeature` if bindless buffers are not supported.
    /// Returns `BindlessBufferError::ExceedsDeviceLimit` if `max_buffers` exceeds device limit.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::resources::bindless_buffers::BufferRegistry;
    ///
    /// # fn example(device: &wgpu::Device) {
    /// let registry = BufferRegistry::new_validated(1024, device)?;
    /// # Ok::<(), renderer_backend::resources::bindless_buffers::BindlessBufferError>(())
    /// # }
    /// ```
    pub fn new_validated(max_buffers: u32, device: &Device) -> Result<Self, BindlessBufferError> {
        if !supports_bindless_buffers(device) {
            return Err(BindlessBufferError::UnsupportedFeature);
        }

        let device_max = max_bindless_buffers(device);
        if max_buffers > device_max {
            return Err(BindlessBufferError::ExceedsDeviceLimit {
                requested: max_buffers,
                max: device_max,
            });
        }

        Ok(Self::new(max_buffers))
    }

    /// Registers a buffer and returns its slot index.
    ///
    /// The buffer is stored in the registry and will be included in the
    /// next bind group update. Use the returned slot index in shaders to access
    /// the buffer.
    ///
    /// # Arguments
    ///
    /// * `buffer` - The buffer to register. Must be wrapped in `Arc` for shared ownership.
    ///
    /// # Errors
    ///
    /// Returns `BindlessBufferError::RegistryFull` if the registry is at capacity.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::resources::bindless_buffers::BufferRegistry;
    /// use std::sync::Arc;
    ///
    /// # fn example(buffer: wgpu::Buffer) {
    /// let mut registry = BufferRegistry::new(1024);
    /// let slot = registry.register(Arc::new(buffer)).unwrap();
    /// println!("Buffer at index {}", slot.index());
    /// # }
    /// ```
    pub fn register(&mut self, buffer: Arc<Buffer>) -> Result<BufferSlot, BindlessBufferError> {
        // Check capacity
        if self.registered_count >= self.max_buffers {
            return Err(BindlessBufferError::RegistryFull {
                capacity: self.max_buffers,
            });
        }

        // Try to recycle a free slot first
        let index = if let Some(free_index) = self.free_slots.pop() {
            trace!("Recycling buffer slot {}", free_index);
            self.buffers[free_index as usize] = Some(buffer);
            free_index
        } else {
            // Allocate new slot at the end
            let index = self.buffers.len() as u32;
            trace!("Allocating new buffer slot {}", index);
            self.buffers.push(Some(buffer));
            index
        };

        self.registered_count += 1;
        self.dirty = true;
        self.dirty_slots.insert(index);

        Ok(BufferSlot::new(index))
    }

    /// Unregisters a buffer and recycles its slot for reuse.
    ///
    /// After unregistration, the slot index may be reused for future buffers.
    /// The bind group will be marked dirty and rebuilt on the next update.
    ///
    /// # Arguments
    ///
    /// * `slot` - The slot to unregister (obtained from `register()`).
    ///
    /// # Returns
    ///
    /// `true` if the slot was successfully unregistered, `false` if the slot
    /// was already empty or invalid.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::resources::bindless_buffers::{BufferRegistry, BufferSlot};
    /// use std::sync::Arc;
    ///
    /// # fn example(buffer: wgpu::Buffer) {
    /// let mut registry = BufferRegistry::new(1024);
    /// let slot = registry.register(Arc::new(buffer)).unwrap();
    ///
    /// // Later, when buffer is no longer needed
    /// let unregistered = registry.unregister(slot);
    /// assert!(unregistered);
    /// # }
    /// ```
    pub fn unregister(&mut self, slot: BufferSlot) -> bool {
        let index = slot.index() as usize;

        if index >= self.buffers.len() {
            warn!("Attempted to unregister invalid slot {}", slot);
            return false;
        }

        if self.buffers[index].is_none() {
            warn!("Attempted to unregister already-empty slot {}", slot);
            return false;
        }

        trace!("Unregistering buffer slot {}", slot.index());
        self.buffers[index] = None;
        self.free_slots.push(slot.index());
        self.dirty_slots.remove(&slot.index());
        self.registered_count -= 1;
        self.dirty = true;

        true
    }

    /// Marks a buffer slot as dirty (content modified).
    ///
    /// Call this after modifying the contents of a registered buffer.
    /// This allows the registry to track which buffers need attention.
    ///
    /// # Arguments
    ///
    /// * `slot` - The slot whose buffer content was modified.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::resources::bindless_buffers::{BufferRegistry, BufferSlot};
    ///
    /// # fn example(registry: &mut BufferRegistry, slot: BufferSlot) {
    /// // After writing new data to the buffer
    /// registry.mark_dirty(slot);
    /// # }
    /// ```
    pub fn mark_dirty(&mut self, slot: BufferSlot) {
        if !slot.is_invalid() && (slot.index() as usize) < self.buffers.len() {
            self.dirty_slots.insert(slot.index());
            trace!("Marked buffer slot {} as dirty", slot.index());
        }
    }

    /// Returns an iterator over all dirty buffer slots.
    ///
    /// This returns slots that have been marked dirty since the last
    /// call to `clear_dirty()`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::resources::bindless_buffers::BufferRegistry;
    ///
    /// # fn example(registry: &BufferRegistry) {
    /// for slot in registry.dirty_slots() {
    ///     println!("Slot {} was modified", slot.index());
    /// }
    /// # }
    /// ```
    pub fn dirty_slots(&self) -> impl Iterator<Item = BufferSlot> + '_ {
        self.dirty_slots.iter().map(|&idx| BufferSlot::new(idx))
    }

    /// Returns the dirty slot indices as a slice-compatible collection.
    ///
    /// This is useful for batch processing dirty slots.
    pub fn dirty_slot_indices(&self) -> impl Iterator<Item = u32> + '_ {
        self.dirty_slots.iter().copied()
    }

    /// Returns the number of dirty slots.
    pub fn dirty_count(&self) -> usize {
        self.dirty_slots.len()
    }

    /// Clears the dirty slot tracking.
    ///
    /// Call this after processing dirty slots (e.g., after updating
    /// a bind group or performing synchronization).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::resources::bindless_buffers::BufferRegistry;
    ///
    /// # fn example(registry: &mut BufferRegistry, device: &wgpu::Device, layout: &wgpu::BindGroupLayout) {
    /// // Process dirty slots
    /// for slot in registry.dirty_slots() {
    ///     // Handle dirty buffer...
    /// }
    ///
    /// // Clear after processing
    /// registry.clear_dirty();
    /// # }
    /// ```
    pub fn clear_dirty(&mut self) {
        self.dirty_slots.clear();
        trace!("Cleared dirty slot tracking");
    }

    /// Checks if a specific slot is marked dirty.
    pub fn is_slot_dirty(&self, slot: BufferSlot) -> bool {
        self.dirty_slots.contains(&slot.index())
    }

    /// Creates or updates the bind group for GPU access.
    ///
    /// This method must be called after registering/unregistering buffers and
    /// before using the bind group in a render/compute pass. It only rebuilds
    /// the bind group if the registry has been modified.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device for bind group creation
    /// * `layout` - The bind group layout (must match registry configuration)
    ///
    /// # Returns
    ///
    /// `true` if the bind group was rebuilt, `false` if it was already up to date.
    ///
    /// # Note
    ///
    /// If the registry is empty, no bind group is created and the existing one
    /// (if any) is cleared. Use `bind_group()` to check availability.
    pub fn update_bind_group(&mut self, device: &Device, layout: &BindGroupLayout) -> bool {
        if !self.dirty {
            return false;
        }

        // Collect non-None buffers
        let buffer_refs: Vec<&Buffer> = self
            .buffers
            .iter()
            .filter_map(|opt| opt.as_ref().map(|arc| arc.as_ref()))
            .collect();

        if buffer_refs.is_empty() {
            debug!("BufferRegistry: clearing bind group (no buffers registered)");
            self.bind_group = None;
            self.dirty = false;
            return true;
        }

        debug!(
            "BufferRegistry: creating bind group with {} buffers",
            buffer_refs.len()
        );

        // Build buffer binding array
        let buffer_bindings: Vec<wgpu::BufferBinding> = buffer_refs
            .iter()
            .map(|buf| wgpu::BufferBinding {
                buffer: *buf,
                offset: 0,
                size: None, // Entire buffer
            })
            .collect();

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("bindless_buffer_array"),
            layout,
            entries: &[BindGroupEntry {
                binding: BINDLESS_BUFFER_BINDING,
                resource: BindingResource::BufferArray(&buffer_bindings),
            }],
        });

        self.bind_group = Some(bind_group);
        self.dirty = false;
        true
    }

    /// Creates the bind group, returning an error if not possible.
    ///
    /// This is the fallible version of `update_bind_group()` that returns
    /// errors instead of silently handling edge cases.
    ///
    /// # Errors
    ///
    /// - `BindlessBufferError::EmptyRegistry` - No buffers registered
    pub fn create_bind_group(
        &mut self,
        device: &Device,
        layout: &BindGroupLayout,
    ) -> Result<&BindGroup, BindlessBufferError> {
        // Collect non-None buffers
        let buffer_refs: Vec<&Buffer> = self
            .buffers
            .iter()
            .filter_map(|opt| opt.as_ref().map(|arc| arc.as_ref()))
            .collect();

        if buffer_refs.is_empty() {
            return Err(BindlessBufferError::EmptyRegistry);
        }

        if self.dirty {
            debug!(
                "BufferRegistry: creating bind group with {} buffers",
                buffer_refs.len()
            );

            let buffer_bindings: Vec<wgpu::BufferBinding> = buffer_refs
                .iter()
                .map(|buf| wgpu::BufferBinding {
                    buffer: *buf,
                    offset: 0,
                    size: None,
                })
                .collect();

            let bind_group = device.create_bind_group(&BindGroupDescriptor {
                label: Some("bindless_buffer_array"),
                layout,
                entries: &[BindGroupEntry {
                    binding: BINDLESS_BUFFER_BINDING,
                    resource: BindingResource::BufferArray(&buffer_bindings),
                }],
            });

            self.bind_group = Some(bind_group);
            self.dirty = false;
        }

        self.bind_group
            .as_ref()
            .ok_or(BindlessBufferError::EmptyRegistry)
    }

    /// Returns a reference to the current bind group, if available.
    ///
    /// Returns `None` if no bind group has been created or if the registry is empty.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::resources::bindless_buffers::BufferRegistry;
    ///
    /// # fn example(registry: &BufferRegistry) {
    /// if let Some(bind_group) = registry.bind_group() {
    ///     // render_pass.set_bind_group(3, bind_group, &[]);
    /// }
    /// # }
    /// ```
    #[inline]
    pub fn bind_group(&self) -> Option<&BindGroup> {
        self.bind_group.as_ref()
    }

    /// Returns the number of currently registered buffers.
    #[inline]
    pub fn count(&self) -> u32 {
        self.registered_count
    }

    /// Returns the maximum capacity of the registry.
    #[inline]
    pub fn capacity(&self) -> u32 {
        self.max_buffers
    }

    /// Returns `true` if the registry is at capacity.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.registered_count >= self.max_buffers
    }

    /// Returns `true` if the registry has no registered buffers.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.registered_count == 0
    }

    /// Returns `true` if the bind group needs to be rebuilt.
    #[inline]
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Returns the number of free slots available for recycling.
    #[inline]
    pub fn free_slot_count(&self) -> usize {
        self.free_slots.len()
    }

    /// Clears all registered buffers.
    ///
    /// This removes all buffer registrations and marks the bind group as dirty.
    /// The capacity remains unchanged.
    pub fn clear(&mut self) {
        self.buffers.clear();
        self.free_slots.clear();
        self.dirty_slots.clear();
        self.bind_group = None;
        self.dirty = true;
        self.registered_count = 0;
    }

    /// Gets the buffer at a specific slot, if registered.
    ///
    /// Returns `None` if the slot is empty or out of bounds.
    pub fn get(&self, slot: BufferSlot) -> Option<&Arc<Buffer>> {
        let index = slot.index() as usize;
        self.buffers.get(index).and_then(|opt| opt.as_ref())
    }

    /// Checks if a slot is currently registered.
    pub fn is_registered(&self, slot: BufferSlot) -> bool {
        self.get(slot).is_some()
    }

    /// Returns an iterator over all registered (slot, buffer) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (BufferSlot, &Arc<Buffer>)> {
        self.buffers
            .iter()
            .enumerate()
            .filter_map(|(i, opt)| opt.as_ref().map(|v| (BufferSlot::new(i as u32), v)))
    }

    /// Returns metrics about the registry state.
    pub fn metrics(&self) -> BufferRegistryMetrics {
        BufferRegistryMetrics {
            registered_count: self.registered_count,
            capacity: self.max_buffers,
            free_slots: self.free_slots.len() as u32,
            allocated_slots: self.buffers.len() as u32,
            dirty_slots: self.dirty_slots.len() as u32,
            has_bind_group: self.bind_group.is_some(),
            is_dirty: self.dirty,
        }
    }
}

impl Default for BufferRegistry {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_BUFFERS)
    }
}

impl std::fmt::Debug for BufferRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferRegistry")
            .field("registered_count", &self.registered_count)
            .field("capacity", &self.max_buffers)
            .field("free_slots", &self.free_slots.len())
            .field("dirty_slots", &self.dirty_slots.len())
            .field("has_bind_group", &self.bind_group.is_some())
            .field("dirty", &self.dirty)
            .finish()
    }
}

// Safety: BufferRegistry contains only Send + Sync types
unsafe impl Send for BufferRegistry {}
unsafe impl Sync for BufferRegistry {}

// ============================================================================
// BufferRegistryMetrics
// ============================================================================

/// Metrics about buffer registry state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferRegistryMetrics {
    /// Number of currently registered buffers.
    pub registered_count: u32,
    /// Maximum capacity of the registry.
    pub capacity: u32,
    /// Number of free slots available for recycling.
    pub free_slots: u32,
    /// Total number of allocated slots (including free).
    pub allocated_slots: u32,
    /// Number of dirty slots (modified since last bind group update).
    pub dirty_slots: u32,
    /// Whether a bind group has been created.
    pub has_bind_group: bool,
    /// Whether the bind group needs rebuilding.
    pub is_dirty: bool,
}

impl BufferRegistryMetrics {
    /// Returns the utilization percentage (0.0 to 1.0).
    pub fn utilization(&self) -> f32 {
        if self.capacity == 0 {
            0.0
        } else {
            self.registered_count as f32 / self.capacity as f32
        }
    }

    /// Returns the fragmentation ratio (free slots / allocated slots).
    pub fn fragmentation(&self) -> f32 {
        if self.allocated_slots == 0 {
            0.0
        } else {
            self.free_slots as f32 / self.allocated_slots as f32
        }
    }

    /// Returns the dirty ratio (dirty slots / registered slots).
    pub fn dirty_ratio(&self) -> f32 {
        if self.registered_count == 0 {
            0.0
        } else {
            self.dirty_slots as f32 / self.registered_count as f32
        }
    }
}

// ============================================================================
// Bind Group Layout Helper
// ============================================================================

/// Creates a bind group layout entry for bindless storage buffer arrays.
///
/// This helper creates the layout entry with correct settings for variable-count
/// storage buffer binding.
///
/// # Arguments
///
/// * `binding` - Binding index (typically 1 for buffers in the bindless group)
/// * `count` - Number of buffers in the array
/// * `read_only` - Whether buffers are read-only (true) or read-write (false)
///
/// # Example
///
/// ```
/// use renderer_backend::resources::bindless_buffers::bindless_buffer_layout_entry;
///
/// let entry = bindless_buffer_layout_entry(1, 1024, true);
/// ```
pub fn bindless_buffer_layout_entry(binding: u32, count: u32, read_only: bool) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::VERTEX_FRAGMENT | ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: NonZeroU32::new(count),
    }
}

/// Creates a bind group layout entry for read-only storage buffer arrays.
///
/// This is a convenience wrapper for `bindless_buffer_layout_entry` with `read_only=true`.
pub fn bindless_buffer_layout_entry_readonly(binding: u32, count: u32) -> BindGroupLayoutEntry {
    bindless_buffer_layout_entry(binding, count, true)
}

/// Creates a bind group layout entry for read-write storage buffer arrays.
///
/// This is a convenience wrapper for `bindless_buffer_layout_entry` with `read_only=false`.
pub fn bindless_buffer_layout_entry_readwrite(binding: u32, count: u32) -> BindGroupLayoutEntry {
    bindless_buffer_layout_entry(binding, count, false)
}

/// Creates a bind group layout for bindless storage buffers.
///
/// This creates a standard layout for bindless buffer arrays using the
/// TRINITY convention (binding 1, read-only storage buffers).
///
/// # Arguments
///
/// * `device` - The wgpu device
/// * `count` - Number of buffers in the array
/// * `read_only` - Whether buffers are read-only
/// * `label` - Optional debug label
///
/// # Example
///
/// ```no_run
/// use renderer_backend::resources::bindless_buffers::create_bindless_buffer_layout;
///
/// # fn example(device: &wgpu::Device) {
/// let layout = create_bindless_buffer_layout(device, 1024, true, Some("bindless_buffers"));
/// # }
/// ```
pub fn create_bindless_buffer_layout(
    device: &Device,
    count: u32,
    read_only: bool,
    label: Option<&str>,
) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label,
        entries: &[bindless_buffer_layout_entry(
            BINDLESS_BUFFER_BINDING,
            count,
            read_only,
        )],
    })
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // BufferSlot Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_slot_new() {
        let slot = BufferSlot::new(42);
        assert_eq!(slot.index(), 42);
    }

    #[test]
    fn test_slot_from_u32() {
        let slot: BufferSlot = 123u32.into();
        assert_eq!(slot.index(), 123);
    }

    #[test]
    fn test_slot_into_u32() {
        let slot = BufferSlot::new(456);
        let index: u32 = slot.into();
        assert_eq!(index, 456);
    }

    #[test]
    fn test_slot_invalid() {
        let slot = BufferSlot::invalid();
        assert!(slot.is_invalid());
        assert_eq!(slot.index(), u32::MAX);
    }

    #[test]
    fn test_slot_valid_not_invalid() {
        let slot = BufferSlot::new(0);
        assert!(!slot.is_invalid());
    }

    #[test]
    fn test_slot_equality() {
        let slot1 = BufferSlot::new(10);
        let slot2 = BufferSlot::new(10);
        let slot3 = BufferSlot::new(20);

        assert_eq!(slot1, slot2);
        assert_ne!(slot1, slot3);
    }

    #[test]
    fn test_slot_hash() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(BufferSlot::new(1));
        set.insert(BufferSlot::new(2));
        set.insert(BufferSlot::new(1)); // Duplicate

        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_slot_display() {
        assert_eq!(format!("{}", BufferSlot::new(42)), "BufferSlot(42)");
        assert_eq!(format!("{}", BufferSlot::invalid()), "BufferSlot(INVALID)");
    }

    #[test]
    fn test_slot_debug() {
        let slot = BufferSlot::new(7);
        let debug = format!("{:?}", slot);
        assert!(debug.contains("7"));
    }

    #[test]
    fn test_slot_copy() {
        let slot1 = BufferSlot::new(5);
        let slot2 = slot1; // Copy
        assert_eq!(slot1, slot2);
    }

    #[test]
    fn test_slot_clone() {
        let slot1 = BufferSlot::new(8);
        let slot2 = slot1.clone();
        assert_eq!(slot1, slot2);
    }

    // -------------------------------------------------------------------------
    // BindlessBufferError Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_error_unsupported_feature_display() {
        let err = BindlessBufferError::UnsupportedFeature;
        let msg = err.to_string();
        assert!(msg.contains("BUFFER_BINDING_ARRAY"));
        assert!(msg.contains("not supported"));
    }

    #[test]
    fn test_error_registry_full_display() {
        let err = BindlessBufferError::RegistryFull { capacity: 1024 };
        let msg = err.to_string();
        assert!(msg.contains("1024"));
        assert!(msg.contains("full"));
    }

    #[test]
    fn test_error_invalid_slot_display() {
        let err = BindlessBufferError::InvalidSlot(BufferSlot::new(42));
        let msg = err.to_string();
        assert!(msg.contains("42"));
        assert!(msg.contains("invalid"));
    }

    #[test]
    fn test_error_exceeds_device_limit_display() {
        let err = BindlessBufferError::ExceedsDeviceLimit {
            requested: 2000,
            max: 1024,
        };
        let msg = err.to_string();
        assert!(msg.contains("2000"));
        assert!(msg.contains("1024"));
        assert!(msg.contains("exceeds"));
    }

    #[test]
    fn test_error_incompatible_layout_display() {
        let err = BindlessBufferError::IncompatibleLayout;
        assert!(err.to_string().contains("incompatible"));
    }

    #[test]
    fn test_error_empty_registry_display() {
        let err = BindlessBufferError::EmptyRegistry;
        assert!(err.to_string().contains("empty"));
    }

    #[test]
    fn test_error_equality() {
        let err1 = BindlessBufferError::RegistryFull { capacity: 100 };
        let err2 = BindlessBufferError::RegistryFull { capacity: 100 };
        let err3 = BindlessBufferError::RegistryFull { capacity: 200 };

        assert_eq!(err1, err2);
        assert_ne!(err1, err3);
    }

    #[test]
    fn test_error_is_std_error() {
        let err: Box<dyn std::error::Error> = Box::new(BindlessBufferError::UnsupportedFeature);
        assert!(err.to_string().contains("not supported"));
    }

    // -------------------------------------------------------------------------
    // BufferRegistry Construction Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_registry_new_default_capacity() {
        let registry = BufferRegistry::new(DEFAULT_MAX_BUFFERS);
        assert_eq!(registry.capacity(), DEFAULT_MAX_BUFFERS);
        assert_eq!(registry.count(), 0);
    }

    #[test]
    fn test_registry_new_custom_capacity() {
        let registry = BufferRegistry::new(512);
        assert_eq!(registry.capacity(), 512);
    }

    #[test]
    fn test_registry_new_clamps_minimum() {
        let registry = BufferRegistry::new(1);
        assert_eq!(registry.capacity(), MIN_BINDLESS_BUFFERS);
    }

    #[test]
    fn test_registry_new_clamps_maximum() {
        let registry = BufferRegistry::new(u32::MAX);
        assert_eq!(registry.capacity(), MAX_BINDLESS_BUFFERS_CONSERVATIVE);
    }

    #[test]
    fn test_registry_default() {
        let registry = BufferRegistry::default();
        assert_eq!(registry.capacity(), DEFAULT_MAX_BUFFERS);
    }

    #[test]
    fn test_registry_initial_state() {
        let registry = BufferRegistry::new(100);

        assert_eq!(registry.count(), 0);
        assert!(registry.is_empty());
        assert!(!registry.is_full());
        assert!(registry.is_dirty());
        assert!(registry.bind_group().is_none());
        assert_eq!(registry.free_slot_count(), 0);
        assert_eq!(registry.dirty_count(), 0);
    }

    // -------------------------------------------------------------------------
    // BufferRegistry State Tests (without device)
    // -------------------------------------------------------------------------

    #[test]
    fn test_registry_is_empty() {
        let registry = BufferRegistry::new(100);
        assert!(registry.is_empty());
    }

    #[test]
    fn test_registry_capacity_respects_limit() {
        let registry = BufferRegistry::new(50);
        assert_eq!(registry.capacity(), 50);
    }

    #[test]
    fn test_registry_clear() {
        let mut registry = BufferRegistry::new(100);
        // Simulate some registrations by manipulating internal state
        registry.registered_count = 5;
        registry.dirty = false;
        registry.dirty_slots.insert(1);
        registry.dirty_slots.insert(3);

        registry.clear();

        assert!(registry.is_empty());
        assert!(registry.is_dirty());
        assert_eq!(registry.free_slot_count(), 0);
        assert_eq!(registry.dirty_count(), 0);
    }

    #[test]
    fn test_registry_debug_format() {
        let registry = BufferRegistry::new(256);
        let debug = format!("{:?}", registry);

        assert!(debug.contains("BufferRegistry"));
        assert!(debug.contains("capacity"));
        assert!(debug.contains("256"));
    }

    // -------------------------------------------------------------------------
    // Dirty Tracking Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_dirty_tracking_initial_empty() {
        let registry = BufferRegistry::new(100);
        assert_eq!(registry.dirty_count(), 0);
    }

    #[test]
    fn test_dirty_tracking_mark_dirty() {
        let mut registry = BufferRegistry::new(100);
        registry.buffers.push(None); // Allocate slot 0

        registry.mark_dirty(BufferSlot::new(0));
        assert_eq!(registry.dirty_count(), 1);
        assert!(registry.is_slot_dirty(BufferSlot::new(0)));
    }

    #[test]
    fn test_dirty_tracking_clear() {
        let mut registry = BufferRegistry::new(100);
        registry.buffers.push(None);
        registry.buffers.push(None);

        registry.mark_dirty(BufferSlot::new(0));
        registry.mark_dirty(BufferSlot::new(1));
        assert_eq!(registry.dirty_count(), 2);

        registry.clear_dirty();
        assert_eq!(registry.dirty_count(), 0);
    }

    #[test]
    fn test_dirty_tracking_invalid_slot_ignored() {
        let mut registry = BufferRegistry::new(100);

        // Mark invalid slot
        registry.mark_dirty(BufferSlot::invalid());
        assert_eq!(registry.dirty_count(), 0);

        // Mark out of bounds slot
        registry.mark_dirty(BufferSlot::new(999));
        assert_eq!(registry.dirty_count(), 0);
    }

    #[test]
    fn test_dirty_slots_iterator() {
        let mut registry = BufferRegistry::new(100);
        registry.buffers.push(None);
        registry.buffers.push(None);
        registry.buffers.push(None);

        registry.mark_dirty(BufferSlot::new(0));
        registry.mark_dirty(BufferSlot::new(2));

        let dirty: Vec<_> = registry.dirty_slots().collect();
        assert_eq!(dirty.len(), 2);
        assert!(dirty.contains(&BufferSlot::new(0)));
        assert!(dirty.contains(&BufferSlot::new(2)));
    }

    #[test]
    fn test_dirty_slot_indices() {
        let mut registry = BufferRegistry::new(100);
        registry.buffers.push(None);
        registry.buffers.push(None);

        registry.mark_dirty(BufferSlot::new(0));
        registry.mark_dirty(BufferSlot::new(1));

        let indices: Vec<_> = registry.dirty_slot_indices().collect();
        assert_eq!(indices.len(), 2);
        assert!(indices.contains(&0));
        assert!(indices.contains(&1));
    }

    // -------------------------------------------------------------------------
    // BufferRegistry Metrics Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_metrics_empty_registry() {
        let registry = BufferRegistry::new(100);
        let metrics = registry.metrics();

        assert_eq!(metrics.registered_count, 0);
        assert_eq!(metrics.capacity, 100);
        assert_eq!(metrics.free_slots, 0);
        assert_eq!(metrics.allocated_slots, 0);
        assert_eq!(metrics.dirty_slots, 0);
        assert!(!metrics.has_bind_group);
        assert!(metrics.is_dirty);
    }

    #[test]
    fn test_metrics_utilization_empty() {
        let registry = BufferRegistry::new(100);
        assert_eq!(registry.metrics().utilization(), 0.0);
    }

    #[test]
    fn test_metrics_utilization_zero_capacity() {
        let metrics = BufferRegistryMetrics {
            registered_count: 0,
            capacity: 0,
            free_slots: 0,
            allocated_slots: 0,
            dirty_slots: 0,
            has_bind_group: false,
            is_dirty: true,
        };
        assert_eq!(metrics.utilization(), 0.0);
    }

    #[test]
    fn test_metrics_fragmentation_empty() {
        let metrics = BufferRegistryMetrics {
            registered_count: 0,
            capacity: 100,
            free_slots: 0,
            allocated_slots: 0,
            dirty_slots: 0,
            has_bind_group: false,
            is_dirty: true,
        };
        assert_eq!(metrics.fragmentation(), 0.0);
    }

    #[test]
    fn test_metrics_fragmentation_calculation() {
        let metrics = BufferRegistryMetrics {
            registered_count: 8,
            capacity: 100,
            free_slots: 2,
            allocated_slots: 10,
            dirty_slots: 1,
            has_bind_group: true,
            is_dirty: false,
        };
        // 2/10 = 0.2
        assert!((metrics.fragmentation() - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_metrics_utilization_calculation() {
        let metrics = BufferRegistryMetrics {
            registered_count: 25,
            capacity: 100,
            free_slots: 0,
            allocated_slots: 25,
            dirty_slots: 5,
            has_bind_group: true,
            is_dirty: false,
        };
        // 25/100 = 0.25
        assert!((metrics.utilization() - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_metrics_dirty_ratio_empty() {
        let metrics = BufferRegistryMetrics {
            registered_count: 0,
            capacity: 100,
            free_slots: 0,
            allocated_slots: 0,
            dirty_slots: 0,
            has_bind_group: false,
            is_dirty: true,
        };
        assert_eq!(metrics.dirty_ratio(), 0.0);
    }

    #[test]
    fn test_metrics_dirty_ratio_calculation() {
        let metrics = BufferRegistryMetrics {
            registered_count: 10,
            capacity: 100,
            free_slots: 0,
            allocated_slots: 10,
            dirty_slots: 5,
            has_bind_group: true,
            is_dirty: false,
        };
        // 5/10 = 0.5
        assert!((metrics.dirty_ratio() - 0.5).abs() < 0.001);
    }

    // -------------------------------------------------------------------------
    // Constants Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_constants_reasonable_values() {
        assert!(DEFAULT_MAX_BUFFERS >= MIN_BINDLESS_BUFFERS);
        assert!(DEFAULT_MAX_BUFFERS <= MAX_BINDLESS_BUFFERS_CONSERVATIVE);
        assert!(MIN_BINDLESS_BUFFERS >= 1);
    }

    #[test]
    fn test_bind_group_indices() {
        assert_eq!(BINDLESS_BIND_GROUP_INDEX, 3);
        assert_eq!(BINDLESS_BUFFER_BINDING, 1);
    }

    // -------------------------------------------------------------------------
    // Feature Functions Tests (without device)
    // -------------------------------------------------------------------------

    #[test]
    fn test_bindless_buffer_required_features() {
        let features = bindless_buffer_required_features();
        assert!(features.contains(Features::BUFFER_BINDING_ARRAY));
    }

    #[test]
    fn test_bindless_buffer_optimal_features() {
        let features = bindless_buffer_optimal_features();
        assert!(features.contains(Features::BUFFER_BINDING_ARRAY));
        assert!(features.contains(Features::STORAGE_RESOURCE_BINDING_ARRAY));
        assert!(features.contains(
            Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
        ));
        assert!(features.contains(Features::PARTIALLY_BOUND_BINDING_ARRAY));
    }

    #[test]
    fn test_max_bindless_buffers_from_limits() {
        let mut limits = Limits::default();
        limits.max_storage_buffers_per_shader_stage = 500;

        let max = max_bindless_buffers_from_limits(&limits);
        assert_eq!(max, 500);
    }

    #[test]
    fn test_max_bindless_buffers_from_limits_clamped() {
        let mut limits = Limits::default();
        limits.max_storage_buffers_per_shader_stage = u32::MAX;

        let max = max_bindless_buffers_from_limits(&limits);
        assert_eq!(max, MAX_BINDLESS_BUFFERS_CONSERVATIVE);
    }

    // -------------------------------------------------------------------------
    // Bind Group Layout Entry Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_bindless_buffer_layout_entry_readonly() {
        let entry = bindless_buffer_layout_entry(1, 1024, true);

        assert_eq!(entry.binding, 1);
        assert!(entry.visibility.contains(ShaderStages::VERTEX_FRAGMENT));
        assert!(entry.visibility.contains(ShaderStages::COMPUTE));
        assert_eq!(entry.count, NonZeroU32::new(1024));

        if let BindingType::Buffer {
            ty,
            has_dynamic_offset,
            min_binding_size,
        } = entry.ty
        {
            assert!(matches!(ty, BufferBindingType::Storage { read_only: true }));
            assert!(!has_dynamic_offset);
            assert!(min_binding_size.is_none());
        } else {
            panic!("Expected Buffer binding type");
        }
    }

    #[test]
    fn test_bindless_buffer_layout_entry_readwrite() {
        let entry = bindless_buffer_layout_entry(1, 512, false);

        assert_eq!(entry.binding, 1);
        assert_eq!(entry.count, NonZeroU32::new(512));

        if let BindingType::Buffer { ty, .. } = entry.ty {
            assert!(matches!(ty, BufferBindingType::Storage { read_only: false }));
        } else {
            panic!("Expected Buffer binding type");
        }
    }

    #[test]
    fn test_bindless_buffer_layout_entry_readonly_helper() {
        let entry = bindless_buffer_layout_entry_readonly(1, 256);

        if let BindingType::Buffer { ty, .. } = entry.ty {
            assert!(matches!(ty, BufferBindingType::Storage { read_only: true }));
        } else {
            panic!("Expected Buffer binding type");
        }
    }

    #[test]
    fn test_bindless_buffer_layout_entry_readwrite_helper() {
        let entry = bindless_buffer_layout_entry_readwrite(1, 256);

        if let BindingType::Buffer { ty, .. } = entry.ty {
            assert!(matches!(ty, BufferBindingType::Storage { read_only: false }));
        } else {
            panic!("Expected Buffer binding type");
        }
    }

    // -------------------------------------------------------------------------
    // Thread Safety Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_registry_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<BufferRegistry>();
    }

    #[test]
    fn test_registry_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<BufferRegistry>();
    }

    #[test]
    fn test_slot_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<BufferSlot>();
    }

    #[test]
    fn test_slot_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<BufferSlot>();
    }

    #[test]
    fn test_error_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<BindlessBufferError>();
    }

    #[test]
    fn test_error_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<BindlessBufferError>();
    }

    #[test]
    fn test_metrics_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<BufferRegistryMetrics>();
    }

    #[test]
    fn test_metrics_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<BufferRegistryMetrics>();
    }

    // -------------------------------------------------------------------------
    // Edge Case Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_slot_zero_valid() {
        let slot = BufferSlot::new(0);
        assert!(!slot.is_invalid());
        assert_eq!(slot.index(), 0);
    }

    #[test]
    fn test_slot_max_minus_one_valid() {
        let slot = BufferSlot::new(u32::MAX - 1);
        assert!(!slot.is_invalid());
        assert_eq!(slot.index(), u32::MAX - 1);
    }

    #[test]
    fn test_registry_min_capacity() {
        let registry = BufferRegistry::new(MIN_BINDLESS_BUFFERS);
        assert_eq!(registry.capacity(), MIN_BINDLESS_BUFFERS);
    }

    #[test]
    fn test_registry_max_capacity() {
        let registry = BufferRegistry::new(MAX_BINDLESS_BUFFERS_CONSERVATIVE);
        assert_eq!(registry.capacity(), MAX_BINDLESS_BUFFERS_CONSERVATIVE);
    }

    // -------------------------------------------------------------------------
    // Error Clone Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_error_clone() {
        let err1 = BindlessBufferError::RegistryFull { capacity: 500 };
        let err2 = err1.clone();
        assert_eq!(err1, err2);
    }

    #[test]
    fn test_error_debug() {
        let err = BindlessBufferError::InvalidSlot(BufferSlot::new(99));
        let debug = format!("{:?}", err);
        assert!(debug.contains("InvalidSlot"));
        assert!(debug.contains("99"));
    }

    // -------------------------------------------------------------------------
    // Metrics Copy/Clone Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_metrics_copy() {
        let m1 = BufferRegistryMetrics {
            registered_count: 10,
            capacity: 100,
            free_slots: 5,
            allocated_slots: 15,
            dirty_slots: 2,
            has_bind_group: true,
            is_dirty: false,
        };
        let m2 = m1; // Copy
        assert_eq!(m1, m2);
    }

    #[test]
    fn test_metrics_clone() {
        let m1 = BufferRegistryMetrics {
            registered_count: 10,
            capacity: 100,
            free_slots: 5,
            allocated_slots: 15,
            dirty_slots: 2,
            has_bind_group: true,
            is_dirty: false,
        };
        let m2 = m1.clone();
        assert_eq!(m1, m2);
    }

    #[test]
    fn test_metrics_equality() {
        let m1 = BufferRegistryMetrics {
            registered_count: 10,
            capacity: 100,
            free_slots: 5,
            allocated_slots: 15,
            dirty_slots: 2,
            has_bind_group: true,
            is_dirty: false,
        };
        let m2 = BufferRegistryMetrics {
            registered_count: 10,
            capacity: 100,
            free_slots: 5,
            allocated_slots: 15,
            dirty_slots: 2,
            has_bind_group: true,
            is_dirty: false,
        };
        let m3 = BufferRegistryMetrics {
            registered_count: 20, // Different
            capacity: 100,
            free_slots: 5,
            allocated_slots: 15,
            dirty_slots: 2,
            has_bind_group: true,
            is_dirty: false,
        };

        assert_eq!(m1, m2);
        assert_ne!(m1, m3);
    }

    #[test]
    fn test_metrics_debug() {
        let metrics = BufferRegistryMetrics {
            registered_count: 10,
            capacity: 100,
            free_slots: 5,
            allocated_slots: 15,
            dirty_slots: 2,
            has_bind_group: true,
            is_dirty: false,
        };
        let debug = format!("{:?}", metrics);
        assert!(debug.contains("BufferRegistryMetrics"));
        assert!(debug.contains("registered_count"));
        assert!(debug.contains("10"));
    }
}
