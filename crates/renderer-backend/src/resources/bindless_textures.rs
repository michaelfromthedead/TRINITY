//! Bindless texture array management for TRINITY.
//!
//! This module provides a registry for bindless texture rendering, allowing shaders
//! to access thousands of textures via index rather than individual bind groups.
//! Uses wgpu's `TEXTURE_BINDING_ARRAY` feature for variable-count texture bindings.
//!
//! # Overview
//!
//! Bindless texturing eliminates the need to switch bind groups when rendering
//! objects with different textures. Instead, all textures are bound in a single
//! large array, and shaders access them by index:
//!
//! ```wgsl
//! @group(3) @binding(0) var textures: binding_array<texture_2d<f32>>;
//!
//! @fragment
//! fn main(@location(0) tex_index: u32) -> @location(0) vec4<f32> {
//!     return textureSample(textures[tex_index], sampler, uv);
//! }
//! ```
//!
//! # Architecture
//!
//! ```text
//! TextureRegistry
//!   ├── texture_views: Vec<Option<Arc<TextureView>>>  // Sparse array
//!   ├── free_slots: Vec<u32>                          // Recycled indices
//!   ├── bind_group: Option<BindGroup>                 // Cached, rebuilt when dirty
//!   ├── dirty: bool                                   // Rebuild needed?
//!   └── max_textures: u32                             // Capacity limit
//! ```
//!
//! # Thread Safety
//!
//! The registry is `Send + Sync` and uses interior mutability via `RwLock` for
//! the internal state. However, bind group creation requires `&mut self` to ensure
//! proper synchronization with GPU resource updates.
//!
//! # Example
//!
//! ```no_run
//! use renderer_backend::resources::bindless_textures::{TextureRegistry, supports_bindless_textures};
//! use std::sync::Arc;
//!
//! # fn example(device: &wgpu::Device, texture_view: wgpu::TextureView) {
//! // Check feature support
//! if !supports_bindless_textures(device) {
//!     println!("Bindless textures not supported, falling back to traditional binding");
//!     return;
//! }
//!
//! // Create registry with default capacity
//! let mut registry = TextureRegistry::new(1024);
//!
//! // Register textures and get slot indices for shader access
//! let slot = registry.register(Arc::new(texture_view)).unwrap();
//! println!("Texture registered at slot {}", slot.index());
//!
//! // When done with a texture, unregister to recycle the slot
//! registry.unregister(slot);
//! # }
//! ```
//!
//! # TRINITY Integration
//!
//! The bindless registry uses bind group index 3 (BINDLESS) by default, matching
//! the TRINITY standard layout convention. Use with `TrinityLayoutBuilder::bindless()`
//! for pipeline layout creation.
//!
//! # Feature Requirements
//!
//! - `TEXTURE_BINDING_ARRAY` - Required for texture array binding
//! - `SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING` - Optional, enables
//!   non-uniform indexing in shaders (recommended for full bindless support)
//! - `PARTIALLY_BOUND_BINDING_ARRAY` - Optional, allows slots to be unbound

use log::{debug, trace, warn};
use std::num::NonZeroU32;
use std::sync::Arc;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, Device, Features, Limits, ShaderStages,
    TextureSampleType, TextureView, TextureViewDimension,
};

// ============================================================================
// Constants
// ============================================================================

/// Default maximum number of textures in the bindless registry.
pub const DEFAULT_MAX_TEXTURES: u32 = 1024;

/// Minimum textures to allocate for bindless rendering.
pub const MIN_BINDLESS_TEXTURES: u32 = 16;

/// Maximum textures per shader stage (conservative limit).
/// Actual limit depends on device, use `max_bindless_textures()` for runtime query.
pub const MAX_BINDLESS_TEXTURES_CONSERVATIVE: u32 = 16384;

/// Default bind group index for bindless textures (TRINITY convention).
pub const BINDLESS_BIND_GROUP_INDEX: u32 = 3;

/// Default binding index within the bindless bind group.
pub const BINDLESS_TEXTURE_BINDING: u32 = 0;

// ============================================================================
// Feature Detection
// ============================================================================

/// Checks if the device supports bindless texture arrays.
///
/// Returns `true` if the device has the `TEXTURE_BINDING_ARRAY` feature enabled,
/// which is required for variable-count texture binding.
///
/// # Arguments
///
/// * `device` - The wgpu device to query
///
/// # Example
///
/// ```no_run
/// use renderer_backend::resources::bindless_textures::supports_bindless_textures;
///
/// # fn example(device: &wgpu::Device) {
/// if supports_bindless_textures(device) {
///     println!("Bindless textures supported!");
/// } else {
///     println!("Falling back to traditional texture binding");
/// }
/// # }
/// ```
pub fn supports_bindless_textures(device: &Device) -> bool {
    device.features().contains(Features::TEXTURE_BINDING_ARRAY)
}

/// Checks if the device supports non-uniform indexing of texture arrays.
///
/// Non-uniform indexing allows different shader invocations in the same workgroup
/// to access different texture indices. Without this feature, all invocations must
/// access the same index.
///
/// # Arguments
///
/// * `device` - The wgpu device to query
pub fn supports_non_uniform_indexing(device: &Device) -> bool {
    device.features().contains(
        Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
    )
}

/// Checks if the device supports partially bound binding arrays.
///
/// When enabled, not all slots in the texture array need to be bound. This allows
/// for more efficient memory usage when the actual texture count is less than the
/// array capacity.
///
/// # Arguments
///
/// * `device` - The wgpu device to query
pub fn supports_partially_bound(device: &Device) -> bool {
    device.features().contains(Features::PARTIALLY_BOUND_BINDING_ARRAY)
}

/// Returns the maximum number of textures supported in a bindless array.
///
/// This queries the device's `max_sampled_textures_per_shader_stage` limit and
/// returns the minimum of that limit and the conservative maximum.
///
/// # Arguments
///
/// * `device` - The wgpu device to query
///
/// # Returns
///
/// Maximum number of textures that can be bound in a single array.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::resources::bindless_textures::max_bindless_textures;
///
/// # fn example(device: &wgpu::Device) {
/// let max = max_bindless_textures(device);
/// println!("Can bind up to {} textures", max);
/// # }
/// ```
pub fn max_bindless_textures(device: &Device) -> u32 {
    let limits = device.limits();
    limits
        .max_sampled_textures_per_shader_stage
        .min(MAX_BINDLESS_TEXTURES_CONSERVATIVE)
}

/// Returns the maximum number of textures from device limits.
///
/// This is a lower-level query that takes limits directly, useful for
/// pre-creation validation.
///
/// # Arguments
///
/// * `limits` - The device limits to query
pub fn max_bindless_textures_from_limits(limits: &Limits) -> u32 {
    limits
        .max_sampled_textures_per_shader_stage
        .min(MAX_BINDLESS_TEXTURES_CONSERVATIVE)
}

/// Bindless texture feature requirements for device creation.
///
/// Returns the wgpu features needed for full bindless texture support.
pub fn bindless_required_features() -> Features {
    Features::TEXTURE_BINDING_ARRAY
}

/// Bindless texture feature requirements for optimal performance.
///
/// Returns the wgpu features needed for full bindless texture support
/// including non-uniform indexing and partial binding.
pub fn bindless_optimal_features() -> Features {
    Features::TEXTURE_BINDING_ARRAY
        | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
        | Features::PARTIALLY_BOUND_BINDING_ARRAY
}

// ============================================================================
// TextureSlot
// ============================================================================

/// A handle to a registered texture in the bindless registry.
///
/// This is a lightweight handle that can be passed to shaders as an index
/// for texture array access. The slot remains valid until `unregister()` is called.
///
/// # Example
///
/// ```
/// use renderer_backend::resources::bindless_textures::TextureSlot;
///
/// // Create a slot (normally obtained from TextureRegistry::register)
/// let slot = TextureSlot::new(42);
/// assert_eq!(slot.index(), 42);
///
/// // Convert to/from u32 for shader uniform passing
/// let index: u32 = slot.into();
/// let slot_again = TextureSlot::from(index);
/// assert_eq!(slot_again, slot);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureSlot(u32);

impl TextureSlot {
    /// Creates a new texture slot with the given index.
    ///
    /// # Arguments
    ///
    /// * `index` - The slot index in the texture array
    #[inline]
    pub const fn new(index: u32) -> Self {
        Self(index)
    }

    /// Returns the slot index for shader access.
    ///
    /// This index can be used directly in shaders to access the texture:
    /// ```wgsl
    /// let color = textureSample(textures[slot_index], sampler, uv);
    /// ```
    #[inline]
    pub const fn index(&self) -> u32 {
        self.0
    }

    /// Creates an invalid slot marker.
    ///
    /// This can be used as a sentinel value for "no texture" scenarios.
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

impl From<u32> for TextureSlot {
    #[inline]
    fn from(index: u32) -> Self {
        Self(index)
    }
}

impl From<TextureSlot> for u32 {
    #[inline]
    fn from(slot: TextureSlot) -> Self {
        slot.0
    }
}

impl std::fmt::Display for TextureSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_invalid() {
            write!(f, "TextureSlot(INVALID)")
        } else {
            write!(f, "TextureSlot({})", self.0)
        }
    }
}

// ============================================================================
// BindlessError
// ============================================================================

/// Errors that can occur during bindless texture operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BindlessError {
    /// The device does not support bindless textures (TEXTURE_BINDING_ARRAY feature missing).
    UnsupportedFeature,

    /// The texture registry is full and cannot accept more textures.
    RegistryFull {
        /// Current capacity of the registry.
        capacity: u32,
    },

    /// The specified texture slot is invalid or has already been unregistered.
    InvalidSlot(TextureSlot),

    /// The requested texture count exceeds the device limit.
    ExceedsDeviceLimit {
        /// Number of textures requested.
        requested: u32,
        /// Maximum supported by the device.
        max: u32,
    },

    /// The bind group layout is incompatible with the registry.
    IncompatibleLayout,

    /// No textures registered - cannot create empty bind group.
    EmptyRegistry,
}

impl std::fmt::Display for BindlessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BindlessError::UnsupportedFeature => {
                write!(f, "bindless textures not supported: TEXTURE_BINDING_ARRAY feature missing")
            }
            BindlessError::RegistryFull { capacity } => {
                write!(f, "bindless texture registry full (capacity: {})", capacity)
            }
            BindlessError::InvalidSlot(slot) => {
                write!(f, "invalid texture slot: {}", slot)
            }
            BindlessError::ExceedsDeviceLimit { requested, max } => {
                write!(
                    f,
                    "requested {} textures exceeds device limit of {}",
                    requested, max
                )
            }
            BindlessError::IncompatibleLayout => {
                write!(f, "bind group layout is incompatible with texture registry")
            }
            BindlessError::EmptyRegistry => {
                write!(f, "cannot create bind group for empty texture registry")
            }
        }
    }
}

impl std::error::Error for BindlessError {}

// ============================================================================
// TextureRegistry
// ============================================================================

/// A registry for managing bindless textures.
///
/// The registry maintains a sparse array of texture views that can be bound
/// as a texture array for shader access. It handles slot allocation, recycling,
/// and bind group creation.
///
/// # Lifecycle
///
/// 1. Create registry with capacity: `TextureRegistry::new(1024)`
/// 2. Register textures: `let slot = registry.register(view)?`
/// 3. Update bind group: `registry.update_bind_group(device, layout)`
/// 4. Use bind group in render pass
/// 5. Unregister when done: `registry.unregister(slot)`
///
/// # Thread Safety
///
/// The registry is `Send + Sync`. Registration and unregistration are thread-safe,
/// but bind group updates require exclusive access to ensure GPU synchronization.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::resources::bindless_textures::TextureRegistry;
/// use std::sync::Arc;
///
/// # fn example(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, views: Vec<wgpu::TextureView>) {
/// let mut registry = TextureRegistry::new(1024);
///
/// // Register multiple textures
/// let mut slots = Vec::new();
/// for view in views {
///     let slot = registry.register(Arc::new(view)).unwrap();
///     slots.push(slot);
/// }
///
/// // Update bind group for GPU access
/// registry.update_bind_group(device, layout);
///
/// // In render pass: set_bind_group(3, registry.bind_group(), &[])
/// # }
/// ```
pub struct TextureRegistry {
    /// Sparse array of registered texture views.
    /// `None` indicates an empty/recycled slot.
    texture_views: Vec<Option<Arc<TextureView>>>,

    /// Stack of free slot indices for efficient recycling.
    free_slots: Vec<u32>,

    /// Cached bind group, rebuilt when `dirty` is true.
    bind_group: Option<BindGroup>,

    /// True if texture array changed since last bind group creation.
    dirty: bool,

    /// Maximum number of textures this registry can hold.
    max_textures: u32,

    /// Number of currently registered textures.
    registered_count: u32,
}

impl TextureRegistry {
    /// Creates a new texture registry with the specified capacity.
    ///
    /// # Arguments
    ///
    /// * `max_textures` - Maximum number of textures the registry can hold.
    ///   Clamped to `MIN_BINDLESS_TEXTURES..=MAX_BINDLESS_TEXTURES_CONSERVATIVE`.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::resources::bindless_textures::TextureRegistry;
    ///
    /// let registry = TextureRegistry::new(1024);
    /// assert_eq!(registry.capacity(), 1024);
    /// assert_eq!(registry.count(), 0);
    /// ```
    pub fn new(max_textures: u32) -> Self {
        let clamped = max_textures
            .max(MIN_BINDLESS_TEXTURES)
            .min(MAX_BINDLESS_TEXTURES_CONSERVATIVE);

        if clamped != max_textures {
            debug!(
                "TextureRegistry capacity clamped from {} to {}",
                max_textures, clamped
            );
        }

        Self {
            texture_views: Vec::with_capacity(clamped as usize),
            free_slots: Vec::new(),
            bind_group: None,
            dirty: true,
            max_textures: clamped,
            registered_count: 0,
        }
    }

    /// Creates a new texture registry with validation against device limits.
    ///
    /// # Arguments
    ///
    /// * `max_textures` - Desired maximum texture count
    /// * `device` - Device to validate limits against
    ///
    /// # Errors
    ///
    /// Returns `BindlessError::UnsupportedFeature` if bindless textures are not supported.
    /// Returns `BindlessError::ExceedsDeviceLimit` if `max_textures` exceeds device limit.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::resources::bindless_textures::TextureRegistry;
    ///
    /// # fn example(device: &wgpu::Device) {
    /// let registry = TextureRegistry::new_validated(1024, device)?;
    /// # Ok::<(), renderer_backend::resources::bindless_textures::BindlessError>(())
    /// # }
    /// ```
    pub fn new_validated(max_textures: u32, device: &Device) -> Result<Self, BindlessError> {
        if !supports_bindless_textures(device) {
            return Err(BindlessError::UnsupportedFeature);
        }

        let device_max = max_bindless_textures(device);
        if max_textures > device_max {
            return Err(BindlessError::ExceedsDeviceLimit {
                requested: max_textures,
                max: device_max,
            });
        }

        Ok(Self::new(max_textures))
    }

    /// Registers a texture view and returns its slot index.
    ///
    /// The texture view is stored in the registry and will be included in the
    /// next bind group update. Use the returned slot index in shaders to access
    /// the texture.
    ///
    /// # Arguments
    ///
    /// * `view` - The texture view to register. Must be wrapped in `Arc` for shared ownership.
    ///
    /// # Errors
    ///
    /// Returns `BindlessError::RegistryFull` if the registry is at capacity.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::resources::bindless_textures::TextureRegistry;
    /// use std::sync::Arc;
    ///
    /// # fn example(view: wgpu::TextureView) {
    /// let mut registry = TextureRegistry::new(1024);
    /// let slot = registry.register(Arc::new(view)).unwrap();
    /// println!("Texture at index {}", slot.index());
    /// # }
    /// ```
    pub fn register(&mut self, view: Arc<TextureView>) -> Result<TextureSlot, BindlessError> {
        // Check capacity
        if self.registered_count >= self.max_textures {
            return Err(BindlessError::RegistryFull {
                capacity: self.max_textures,
            });
        }

        // Try to recycle a free slot first
        let index = if let Some(free_index) = self.free_slots.pop() {
            trace!("Recycling texture slot {}", free_index);
            self.texture_views[free_index as usize] = Some(view);
            free_index
        } else {
            // Allocate new slot at the end
            let index = self.texture_views.len() as u32;
            trace!("Allocating new texture slot {}", index);
            self.texture_views.push(Some(view));
            index
        };

        self.registered_count += 1;
        self.dirty = true;

        Ok(TextureSlot::new(index))
    }

    /// Unregisters a texture and recycles its slot for reuse.
    ///
    /// After unregistration, the slot index may be reused for future textures.
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
    /// use renderer_backend::resources::bindless_textures::{TextureRegistry, TextureSlot};
    /// use std::sync::Arc;
    ///
    /// # fn example(view: wgpu::TextureView) {
    /// let mut registry = TextureRegistry::new(1024);
    /// let slot = registry.register(Arc::new(view)).unwrap();
    ///
    /// // Later, when texture is no longer needed
    /// let unregistered = registry.unregister(slot);
    /// assert!(unregistered);
    /// # }
    /// ```
    pub fn unregister(&mut self, slot: TextureSlot) -> bool {
        let index = slot.index() as usize;

        if index >= self.texture_views.len() {
            warn!("Attempted to unregister invalid slot {}", slot);
            return false;
        }

        if self.texture_views[index].is_none() {
            warn!("Attempted to unregister already-empty slot {}", slot);
            return false;
        }

        trace!("Unregistering texture slot {}", slot.index());
        self.texture_views[index] = None;
        self.free_slots.push(slot.index());
        self.registered_count -= 1;
        self.dirty = true;

        true
    }

    /// Creates or updates the bind group for GPU access.
    ///
    /// This method must be called after registering/unregistering textures and
    /// before using the bind group in a render pass. It only rebuilds the bind
    /// group if the registry has been modified.
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

        // Collect non-None texture views
        let views: Vec<&TextureView> = self
            .texture_views
            .iter()
            .filter_map(|opt| opt.as_ref().map(|arc| arc.as_ref()))
            .collect();

        if views.is_empty() {
            debug!("TextureRegistry: clearing bind group (no textures registered)");
            self.bind_group = None;
            self.dirty = false;
            return true;
        }

        debug!(
            "TextureRegistry: creating bind group with {} textures",
            views.len()
        );

        // Build texture view references for binding
        let texture_view_refs: Vec<&TextureView> = views;

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("bindless_texture_array"),
            layout,
            entries: &[BindGroupEntry {
                binding: BINDLESS_TEXTURE_BINDING,
                resource: BindingResource::TextureViewArray(&texture_view_refs),
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
    /// - `BindlessError::EmptyRegistry` - No textures registered
    pub fn create_bind_group(
        &mut self,
        device: &Device,
        layout: &BindGroupLayout,
    ) -> Result<&BindGroup, BindlessError> {
        // Collect non-None texture views
        let views: Vec<&TextureView> = self
            .texture_views
            .iter()
            .filter_map(|opt| opt.as_ref().map(|arc| arc.as_ref()))
            .collect();

        if views.is_empty() {
            return Err(BindlessError::EmptyRegistry);
        }

        if self.dirty {
            debug!(
                "TextureRegistry: creating bind group with {} textures",
                views.len()
            );

            let bind_group = device.create_bind_group(&BindGroupDescriptor {
                label: Some("bindless_texture_array"),
                layout,
                entries: &[BindGroupEntry {
                    binding: BINDLESS_TEXTURE_BINDING,
                    resource: BindingResource::TextureViewArray(&views),
                }],
            });

            self.bind_group = Some(bind_group);
            self.dirty = false;
        }

        self.bind_group.as_ref().ok_or(BindlessError::EmptyRegistry)
    }

    /// Returns a reference to the current bind group, if available.
    ///
    /// Returns `None` if no bind group has been created or if the registry is empty.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::resources::bindless_textures::TextureRegistry;
    ///
    /// # fn example(registry: &TextureRegistry) {
    /// if let Some(bind_group) = registry.bind_group() {
    ///     // render_pass.set_bind_group(3, bind_group, &[]);
    /// }
    /// # }
    /// ```
    #[inline]
    pub fn bind_group(&self) -> Option<&BindGroup> {
        self.bind_group.as_ref()
    }

    /// Returns the number of currently registered textures.
    #[inline]
    pub fn count(&self) -> u32 {
        self.registered_count
    }

    /// Returns the maximum capacity of the registry.
    #[inline]
    pub fn capacity(&self) -> u32 {
        self.max_textures
    }

    /// Returns `true` if the registry is at capacity.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.registered_count >= self.max_textures
    }

    /// Returns `true` if the registry has no registered textures.
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

    /// Clears all registered textures.
    ///
    /// This removes all texture registrations and marks the bind group as dirty.
    /// The capacity remains unchanged.
    pub fn clear(&mut self) {
        self.texture_views.clear();
        self.free_slots.clear();
        self.bind_group = None;
        self.dirty = true;
        self.registered_count = 0;
    }

    /// Gets the texture view at a specific slot, if registered.
    ///
    /// Returns `None` if the slot is empty or out of bounds.
    pub fn get(&self, slot: TextureSlot) -> Option<&Arc<TextureView>> {
        let index = slot.index() as usize;
        self.texture_views.get(index).and_then(|opt| opt.as_ref())
    }

    /// Checks if a slot is currently registered.
    pub fn is_registered(&self, slot: TextureSlot) -> bool {
        self.get(slot).is_some()
    }

    /// Returns an iterator over all registered (slot, view) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (TextureSlot, &Arc<TextureView>)> {
        self.texture_views
            .iter()
            .enumerate()
            .filter_map(|(i, opt)| opt.as_ref().map(|v| (TextureSlot::new(i as u32), v)))
    }

    /// Returns metrics about the registry state.
    pub fn metrics(&self) -> TextureRegistryMetrics {
        TextureRegistryMetrics {
            registered_count: self.registered_count,
            capacity: self.max_textures,
            free_slots: self.free_slots.len() as u32,
            allocated_slots: self.texture_views.len() as u32,
            has_bind_group: self.bind_group.is_some(),
            is_dirty: self.dirty,
        }
    }
}

impl Default for TextureRegistry {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_TEXTURES)
    }
}

impl std::fmt::Debug for TextureRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TextureRegistry")
            .field("registered_count", &self.registered_count)
            .field("capacity", &self.max_textures)
            .field("free_slots", &self.free_slots.len())
            .field("has_bind_group", &self.bind_group.is_some())
            .field("dirty", &self.dirty)
            .finish()
    }
}

// Safety: TextureRegistry contains only Send + Sync types
unsafe impl Send for TextureRegistry {}
unsafe impl Sync for TextureRegistry {}

// ============================================================================
// TextureRegistryMetrics
// ============================================================================

/// Metrics about texture registry state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureRegistryMetrics {
    /// Number of currently registered textures.
    pub registered_count: u32,
    /// Maximum capacity of the registry.
    pub capacity: u32,
    /// Number of free slots available for recycling.
    pub free_slots: u32,
    /// Total number of allocated slots (including free).
    pub allocated_slots: u32,
    /// Whether a bind group has been created.
    pub has_bind_group: bool,
    /// Whether the bind group needs rebuilding.
    pub is_dirty: bool,
}

impl TextureRegistryMetrics {
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
}

// ============================================================================
// Bind Group Layout Helper
// ============================================================================

/// Creates a bind group layout entry for bindless texture arrays.
///
/// This helper creates the layout entry with correct settings for variable-count
/// texture binding.
///
/// # Arguments
///
/// * `binding` - Binding index (typically 0 for textures in the bindless group)
/// * `count` - Number of textures in the array
/// * `sample_type` - Texture sample type (typically `Float { filterable: true }`)
/// * `dimension` - Texture view dimension (typically `D2`)
///
/// # Example
///
/// ```
/// use renderer_backend::resources::bindless_textures::bindless_texture_layout_entry;
/// use wgpu::{TextureSampleType, TextureViewDimension};
///
/// let entry = bindless_texture_layout_entry(
///     0,
///     1024,
///     TextureSampleType::Float { filterable: true },
///     TextureViewDimension::D2,
/// );
/// ```
pub fn bindless_texture_layout_entry(
    binding: u32,
    count: u32,
    sample_type: TextureSampleType,
    dimension: TextureViewDimension,
) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::VERTEX_FRAGMENT,
        ty: BindingType::Texture {
            sample_type,
            view_dimension: dimension,
            multisampled: false,
        },
        count: NonZeroU32::new(count),
    }
}

/// Creates a bind group layout for bindless 2D textures.
///
/// This creates a standard layout for bindless texture arrays using the
/// TRINITY convention (binding 0, filterable float textures, 2D).
///
/// # Arguments
///
/// * `device` - The wgpu device
/// * `count` - Number of textures in the array
/// * `label` - Optional debug label
///
/// # Example
///
/// ```no_run
/// use renderer_backend::resources::bindless_textures::create_bindless_layout;
///
/// # fn example(device: &wgpu::Device) {
/// let layout = create_bindless_layout(device, 1024, Some("bindless_textures"));
/// # }
/// ```
pub fn create_bindless_layout(
    device: &Device,
    count: u32,
    label: Option<&str>,
) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label,
        entries: &[bindless_texture_layout_entry(
            BINDLESS_TEXTURE_BINDING,
            count,
            TextureSampleType::Float { filterable: true },
            TextureViewDimension::D2,
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
    // TextureSlot Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_slot_new() {
        let slot = TextureSlot::new(42);
        assert_eq!(slot.index(), 42);
    }

    #[test]
    fn test_slot_from_u32() {
        let slot: TextureSlot = 123u32.into();
        assert_eq!(slot.index(), 123);
    }

    #[test]
    fn test_slot_into_u32() {
        let slot = TextureSlot::new(456);
        let index: u32 = slot.into();
        assert_eq!(index, 456);
    }

    #[test]
    fn test_slot_invalid() {
        let slot = TextureSlot::invalid();
        assert!(slot.is_invalid());
        assert_eq!(slot.index(), u32::MAX);
    }

    #[test]
    fn test_slot_valid_not_invalid() {
        let slot = TextureSlot::new(0);
        assert!(!slot.is_invalid());
    }

    #[test]
    fn test_slot_equality() {
        let slot1 = TextureSlot::new(10);
        let slot2 = TextureSlot::new(10);
        let slot3 = TextureSlot::new(20);

        assert_eq!(slot1, slot2);
        assert_ne!(slot1, slot3);
    }

    #[test]
    fn test_slot_hash() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(TextureSlot::new(1));
        set.insert(TextureSlot::new(2));
        set.insert(TextureSlot::new(1)); // Duplicate

        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_slot_display() {
        assert_eq!(format!("{}", TextureSlot::new(42)), "TextureSlot(42)");
        assert_eq!(format!("{}", TextureSlot::invalid()), "TextureSlot(INVALID)");
    }

    #[test]
    fn test_slot_debug() {
        let slot = TextureSlot::new(7);
        let debug = format!("{:?}", slot);
        assert!(debug.contains("7"));
    }

    #[test]
    fn test_slot_copy() {
        let slot1 = TextureSlot::new(5);
        let slot2 = slot1; // Copy
        assert_eq!(slot1, slot2);
    }

    #[test]
    fn test_slot_clone() {
        let slot1 = TextureSlot::new(8);
        let slot2 = slot1.clone();
        assert_eq!(slot1, slot2);
    }

    // -------------------------------------------------------------------------
    // BindlessError Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_error_unsupported_feature_display() {
        let err = BindlessError::UnsupportedFeature;
        let msg = err.to_string();
        assert!(msg.contains("TEXTURE_BINDING_ARRAY"));
        assert!(msg.contains("not supported"));
    }

    #[test]
    fn test_error_registry_full_display() {
        let err = BindlessError::RegistryFull { capacity: 1024 };
        let msg = err.to_string();
        assert!(msg.contains("1024"));
        assert!(msg.contains("full"));
    }

    #[test]
    fn test_error_invalid_slot_display() {
        let err = BindlessError::InvalidSlot(TextureSlot::new(42));
        let msg = err.to_string();
        assert!(msg.contains("42"));
        assert!(msg.contains("invalid"));
    }

    #[test]
    fn test_error_exceeds_device_limit_display() {
        let err = BindlessError::ExceedsDeviceLimit {
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
        let err = BindlessError::IncompatibleLayout;
        assert!(err.to_string().contains("incompatible"));
    }

    #[test]
    fn test_error_empty_registry_display() {
        let err = BindlessError::EmptyRegistry;
        assert!(err.to_string().contains("empty"));
    }

    #[test]
    fn test_error_equality() {
        let err1 = BindlessError::RegistryFull { capacity: 100 };
        let err2 = BindlessError::RegistryFull { capacity: 100 };
        let err3 = BindlessError::RegistryFull { capacity: 200 };

        assert_eq!(err1, err2);
        assert_ne!(err1, err3);
    }

    #[test]
    fn test_error_is_std_error() {
        let err: Box<dyn std::error::Error> = Box::new(BindlessError::UnsupportedFeature);
        assert!(err.to_string().contains("not supported"));
    }

    // -------------------------------------------------------------------------
    // TextureRegistry Construction Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_registry_new_default_capacity() {
        let registry = TextureRegistry::new(DEFAULT_MAX_TEXTURES);
        assert_eq!(registry.capacity(), DEFAULT_MAX_TEXTURES);
        assert_eq!(registry.count(), 0);
    }

    #[test]
    fn test_registry_new_custom_capacity() {
        let registry = TextureRegistry::new(512);
        assert_eq!(registry.capacity(), 512);
    }

    #[test]
    fn test_registry_new_clamps_minimum() {
        let registry = TextureRegistry::new(1);
        assert_eq!(registry.capacity(), MIN_BINDLESS_TEXTURES);
    }

    #[test]
    fn test_registry_new_clamps_maximum() {
        let registry = TextureRegistry::new(u32::MAX);
        assert_eq!(registry.capacity(), MAX_BINDLESS_TEXTURES_CONSERVATIVE);
    }

    #[test]
    fn test_registry_default() {
        let registry = TextureRegistry::default();
        assert_eq!(registry.capacity(), DEFAULT_MAX_TEXTURES);
    }

    #[test]
    fn test_registry_initial_state() {
        let registry = TextureRegistry::new(100);

        assert_eq!(registry.count(), 0);
        assert!(registry.is_empty());
        assert!(!registry.is_full());
        assert!(registry.is_dirty());
        assert!(registry.bind_group().is_none());
        assert_eq!(registry.free_slot_count(), 0);
    }

    // -------------------------------------------------------------------------
    // TextureRegistry State Tests (without device)
    // -------------------------------------------------------------------------

    #[test]
    fn test_registry_is_empty() {
        let registry = TextureRegistry::new(100);
        assert!(registry.is_empty());
    }

    #[test]
    fn test_registry_capacity_respects_limit() {
        let registry = TextureRegistry::new(50);
        assert_eq!(registry.capacity(), 50);
    }

    #[test]
    fn test_registry_clear() {
        let mut registry = TextureRegistry::new(100);
        // Simulate some registrations by manipulating internal state
        registry.registered_count = 5;
        registry.dirty = false;

        registry.clear();

        assert!(registry.is_empty());
        assert!(registry.is_dirty());
        assert_eq!(registry.free_slot_count(), 0);
    }

    #[test]
    fn test_registry_debug_format() {
        let registry = TextureRegistry::new(256);
        let debug = format!("{:?}", registry);

        assert!(debug.contains("TextureRegistry"));
        assert!(debug.contains("capacity"));
        assert!(debug.contains("256"));
    }

    // -------------------------------------------------------------------------
    // TextureRegistry Metrics Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_metrics_empty_registry() {
        let registry = TextureRegistry::new(100);
        let metrics = registry.metrics();

        assert_eq!(metrics.registered_count, 0);
        assert_eq!(metrics.capacity, 100);
        assert_eq!(metrics.free_slots, 0);
        assert_eq!(metrics.allocated_slots, 0);
        assert!(!metrics.has_bind_group);
        assert!(metrics.is_dirty);
    }

    #[test]
    fn test_metrics_utilization_empty() {
        let registry = TextureRegistry::new(100);
        assert_eq!(registry.metrics().utilization(), 0.0);
    }

    #[test]
    fn test_metrics_utilization_zero_capacity() {
        // This shouldn't happen due to clamping, but test the edge case
        let metrics = TextureRegistryMetrics {
            registered_count: 0,
            capacity: 0,
            free_slots: 0,
            allocated_slots: 0,
            has_bind_group: false,
            is_dirty: true,
        };
        assert_eq!(metrics.utilization(), 0.0);
    }

    #[test]
    fn test_metrics_fragmentation_empty() {
        let metrics = TextureRegistryMetrics {
            registered_count: 0,
            capacity: 100,
            free_slots: 0,
            allocated_slots: 0,
            has_bind_group: false,
            is_dirty: true,
        };
        assert_eq!(metrics.fragmentation(), 0.0);
    }

    #[test]
    fn test_metrics_fragmentation_calculation() {
        let metrics = TextureRegistryMetrics {
            registered_count: 8,
            capacity: 100,
            free_slots: 2,
            allocated_slots: 10,
            has_bind_group: true,
            is_dirty: false,
        };
        // 2/10 = 0.2
        assert!((metrics.fragmentation() - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_metrics_utilization_calculation() {
        let metrics = TextureRegistryMetrics {
            registered_count: 25,
            capacity: 100,
            free_slots: 0,
            allocated_slots: 25,
            has_bind_group: true,
            is_dirty: false,
        };
        // 25/100 = 0.25
        assert!((metrics.utilization() - 0.25).abs() < 0.001);
    }

    // -------------------------------------------------------------------------
    // Constants Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_constants_reasonable_values() {
        assert!(DEFAULT_MAX_TEXTURES >= MIN_BINDLESS_TEXTURES);
        assert!(DEFAULT_MAX_TEXTURES <= MAX_BINDLESS_TEXTURES_CONSERVATIVE);
        assert!(MIN_BINDLESS_TEXTURES >= 1);
    }

    #[test]
    fn test_bind_group_indices() {
        assert_eq!(BINDLESS_BIND_GROUP_INDEX, 3);
        assert_eq!(BINDLESS_TEXTURE_BINDING, 0);
    }

    // -------------------------------------------------------------------------
    // Feature Functions Tests (without device)
    // -------------------------------------------------------------------------

    #[test]
    fn test_bindless_required_features() {
        let features = bindless_required_features();
        assert!(features.contains(Features::TEXTURE_BINDING_ARRAY));
    }

    #[test]
    fn test_bindless_optimal_features() {
        let features = bindless_optimal_features();
        assert!(features.contains(Features::TEXTURE_BINDING_ARRAY));
        assert!(features.contains(
            Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
        ));
        assert!(features.contains(Features::PARTIALLY_BOUND_BINDING_ARRAY));
    }

    #[test]
    fn test_max_bindless_textures_from_limits() {
        let mut limits = Limits::default();
        limits.max_sampled_textures_per_shader_stage = 500;

        let max = max_bindless_textures_from_limits(&limits);
        assert_eq!(max, 500);
    }

    #[test]
    fn test_max_bindless_textures_from_limits_clamped() {
        let mut limits = Limits::default();
        limits.max_sampled_textures_per_shader_stage = u32::MAX;

        let max = max_bindless_textures_from_limits(&limits);
        assert_eq!(max, MAX_BINDLESS_TEXTURES_CONSERVATIVE);
    }

    // -------------------------------------------------------------------------
    // Bind Group Layout Entry Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_bindless_texture_layout_entry() {
        let entry = bindless_texture_layout_entry(
            0,
            1024,
            TextureSampleType::Float { filterable: true },
            TextureViewDimension::D2,
        );

        assert_eq!(entry.binding, 0);
        assert_eq!(entry.visibility, ShaderStages::VERTEX_FRAGMENT);
        assert_eq!(entry.count, NonZeroU32::new(1024));

        if let BindingType::Texture {
            sample_type,
            view_dimension,
            multisampled,
        } = entry.ty
        {
            assert_eq!(sample_type, TextureSampleType::Float { filterable: true });
            assert_eq!(view_dimension, TextureViewDimension::D2);
            assert!(!multisampled);
        } else {
            panic!("Expected Texture binding type");
        }
    }

    #[test]
    fn test_bindless_texture_layout_entry_custom() {
        let entry = bindless_texture_layout_entry(
            5,
            256,
            TextureSampleType::Uint,
            TextureViewDimension::D2Array,
        );

        assert_eq!(entry.binding, 5);
        assert_eq!(entry.count, NonZeroU32::new(256));

        if let BindingType::Texture {
            sample_type,
            view_dimension,
            ..
        } = entry.ty
        {
            assert_eq!(sample_type, TextureSampleType::Uint);
            assert_eq!(view_dimension, TextureViewDimension::D2Array);
        } else {
            panic!("Expected Texture binding type");
        }
    }

    // -------------------------------------------------------------------------
    // Thread Safety Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_registry_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<TextureRegistry>();
    }

    #[test]
    fn test_registry_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<TextureRegistry>();
    }

    #[test]
    fn test_slot_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<TextureSlot>();
    }

    #[test]
    fn test_slot_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<TextureSlot>();
    }

    #[test]
    fn test_error_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<BindlessError>();
    }

    #[test]
    fn test_error_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<BindlessError>();
    }

    // -------------------------------------------------------------------------
    // Edge Case Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_slot_zero_valid() {
        let slot = TextureSlot::new(0);
        assert!(!slot.is_invalid());
        assert_eq!(slot.index(), 0);
    }

    #[test]
    fn test_slot_max_minus_one_valid() {
        let slot = TextureSlot::new(u32::MAX - 1);
        assert!(!slot.is_invalid());
        assert_eq!(slot.index(), u32::MAX - 1);
    }

    #[test]
    fn test_registry_min_capacity() {
        let registry = TextureRegistry::new(MIN_BINDLESS_TEXTURES);
        assert_eq!(registry.capacity(), MIN_BINDLESS_TEXTURES);
    }

    #[test]
    fn test_registry_max_capacity() {
        let registry = TextureRegistry::new(MAX_BINDLESS_TEXTURES_CONSERVATIVE);
        assert_eq!(registry.capacity(), MAX_BINDLESS_TEXTURES_CONSERVATIVE);
    }

    // -------------------------------------------------------------------------
    // Error Clone Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_error_clone() {
        let err1 = BindlessError::RegistryFull { capacity: 500 };
        let err2 = err1.clone();
        assert_eq!(err1, err2);
    }

    #[test]
    fn test_error_debug() {
        let err = BindlessError::InvalidSlot(TextureSlot::new(99));
        let debug = format!("{:?}", err);
        assert!(debug.contains("InvalidSlot"));
        assert!(debug.contains("99"));
    }
}
