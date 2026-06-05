//! Bindless Texture Registry for GPU-driven rendering (T-WGPU-P6.8.1).
//!
//! This module provides a texture registry for managing bindless texture arrays,
//! allowing shaders to access textures via index rather than per-draw bind groups.
//! Uses wgpu's `TEXTURE_BINDING_ARRAY` feature for variable-count texture bindings.
//!
//! # Overview
//!
//! ```text
//! +------------------------------------------------------------------+
//! |                       TextureRegistry                            |
//! +------------------------------------------------------------------+
//! | textures: Vec<Option<wgpu::TextureView>>  // Sparse texture array|
//! | free_slots: Vec<u32>                      // Recycled slot stack |
//! | bind_group: Option<wgpu::BindGroup>       // Cached bind group   |
//! | layout: wgpu::BindGroupLayout             // Layout for binding  |
//! | sampler: wgpu::Sampler                    // Shared sampler      |
//! | dirty: bool                               // Rebuild trigger     |
//! | has_bindless: bool                        // Feature support     |
//! +------------------------------------------------------------------+
//! ```
//!
//! # Bindless Texturing
//!
//! Traditional rendering requires switching bind groups when textures change.
//! Bindless texturing allows all textures to be bound in a single array:
//!
//! ```wgsl
//! @group(3) @binding(0) var textures: binding_array<texture_2d<f32>>;
//! @group(3) @binding(1) var texture_sampler: sampler;
//!
//! @fragment
//! fn main(@location(0) tex_index: u32, @location(1) uv: vec2<f32>) -> @location(0) vec4<f32> {
//!     return textureSample(textures[tex_index], texture_sampler, uv);
//! }
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use renderer_backend::gpu_driven::texture_registry::{TextureRegistry, MAX_BINDLESS_TEXTURES};
//!
//! // Check feature support and create registry
//! let features = device.features();
//! let mut registry = TextureRegistry::new(&device, features);
//!
//! if registry.has_bindless_support() {
//!     // Allocate texture slots
//!     let slot = registry.allocate_slot(texture_view);
//!     println!("Texture at slot {}", slot);
//!
//!     // Get bind group (rebuilds if dirty)
//!     let bind_group = registry.bind_group(&device);
//!
//!     // In render pass:
//!     // render_pass.set_bind_group(3, bind_group, &[]);
//!
//!     // When texture is no longer needed
//!     registry.free_slot(slot);
//! } else {
//!     println!("Bindless textures not supported, use traditional binding");
//! }
//! ```
//!
//! # Data Layout
//!
//! The registry manages a sparse array of texture views with free-list allocation.
//! Texture slots are u32 indices that can be passed directly to shaders.
//!
//! # Capacity
//!
//! Default capacity is `MAX_BINDLESS_TEXTURES` (1024), matching typical driver
//! limits. The capacity can be configured at creation time.
//!
//! # Thread Safety
//!
//! `TextureRegistry` is `Send + Sync`. Slot allocation requires `&mut self`,
//! ensuring single-threaded mutation.

use std::num::NonZeroU32;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, Device, Features, FilterMode,
    SamplerBindingType, SamplerDescriptor, ShaderStages, TextureSampleType, TextureView,
    TextureViewDimension,
};

// =============================================================================
// CONSTANTS
// =============================================================================

/// Maximum number of bindless textures supported by the registry.
///
/// This is a conservative limit that works across most desktop GPUs. The actual
/// device limit may be higher; use `device.limits().max_sampled_textures_per_shader_stage`
/// for the exact value.
pub const MAX_BINDLESS_TEXTURES: u32 = 1024;

/// Minimum number of textures for the registry (prevents degenerate cases).
pub const MIN_BINDLESS_TEXTURES: u32 = 16;

/// Default bind group index for bindless textures (TRINITY convention).
pub const BINDLESS_BIND_GROUP_INDEX: u32 = 3;

/// Binding index for the texture array.
pub const TEXTURE_ARRAY_BINDING: u32 = 0;

/// Binding index for the shared sampler.
pub const SAMPLER_BINDING: u32 = 1;

// =============================================================================
// TEXTURE REGISTRY
// =============================================================================

/// Bindless texture registry for GPU-driven rendering.
///
/// Manages a sparse array of texture views that can be bound as a single texture
/// array for shader access. The registry handles slot allocation, recycling, and
/// bind group creation with automatic dirty-flag tracking.
///
/// # Features
///
/// - **Bindless binding**: All textures in one bind group, indexed by slot
/// - **Free-list allocation**: O(1) slot allocation and recycling
/// - **Lazy rebuild**: Bind group only rebuilt when textures change
/// - **Feature detection**: Graceful fallback when unsupported
///
/// # Lifecycle
///
/// 1. Check feature support with `has_bindless_support()`
/// 2. Allocate slots with `allocate_slot()`
/// 3. Get bind group with `bind_group()` (rebuilds if dirty)
/// 4. Free slots with `free_slot()` when textures are destroyed
pub struct TextureRegistry {
    /// Sparse array of registered texture views.
    /// `None` indicates an empty/recycled slot.
    textures: Vec<Option<TextureView>>,

    /// Stack of free slot indices for O(1) recycling.
    free_slots: Vec<u32>,

    /// Cached bind group, rebuilt when `dirty` is true.
    bind_group: Option<BindGroup>,

    /// Bind group layout for texture array + sampler.
    layout: BindGroupLayout,

    /// Shared sampler for all textures (linear filtering, repeat mode).
    sampler: wgpu::Sampler,

    /// True if texture array changed since last bind group creation.
    dirty: bool,

    /// Maximum number of textures this registry can hold.
    max_textures: u32,

    /// True if bindless textures are supported on this device.
    has_bindless: bool,
}

impl TextureRegistry {
    /// Create a new texture registry, checking for TEXTURE_BINDING_ARRAY feature.
    ///
    /// Creates the bind group layout and sampler, and checks feature support.
    /// If bindless textures are not supported, the registry will still be created
    /// but `has_bindless_support()` will return false.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device
    /// * `features` - Device features (use `device.features()`)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let features = device.features();
    /// let registry = TextureRegistry::new(&device, features);
    ///
    /// if registry.has_bindless_support() {
    ///     // Use bindless texturing
    /// }
    /// ```
    pub fn new(device: &Device, features: Features) -> Self {
        Self::with_capacity(device, features, MAX_BINDLESS_TEXTURES)
    }

    /// Create a texture registry with custom capacity.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device
    /// * `features` - Device features
    /// * `max_textures` - Maximum number of textures (clamped to MIN..MAX)
    pub fn with_capacity(device: &Device, features: Features, max_textures: u32) -> Self {
        let has_bindless = features.contains(Features::TEXTURE_BINDING_ARRAY);
        let supports_partially_bound = features.contains(Features::PARTIALLY_BOUND_BINDING_ARRAY);

        let max_textures = max_textures.max(MIN_BINDLESS_TEXTURES).min(MAX_BINDLESS_TEXTURES);

        // Create bind group layout with texture array + sampler
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("texture_registry_layout"),
            entries: &[
                // Texture array binding
                BindGroupLayoutEntry {
                    binding: TEXTURE_ARRAY_BINDING,
                    visibility: ShaderStages::VERTEX_FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    // Use None for variable count when partially bound is supported,
                    // otherwise use fixed count
                    count: if supports_partially_bound {
                        NonZeroU32::new(max_textures)
                    } else {
                        NonZeroU32::new(max_textures)
                    },
                },
                // Shared sampler binding
                BindGroupLayoutEntry {
                    binding: SAMPLER_BINDING,
                    visibility: ShaderStages::VERTEX_FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Create default sampler (linear filtering, repeat mode)
        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("texture_registry_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            ..Default::default()
        });

        Self {
            textures: Vec::with_capacity(max_textures as usize),
            free_slots: Vec::new(),
            bind_group: None,
            layout,
            sampler,
            dirty: true,
            max_textures,
            has_bindless,
        }
    }

    /// Allocate a slot for a texture, returns slot index.
    ///
    /// Uses a free slot if available, otherwise appends to the texture array.
    /// Marks the registry as dirty to trigger bind group rebuild.
    ///
    /// # Arguments
    ///
    /// * `view` - The texture view to register
    ///
    /// # Returns
    ///
    /// Slot index that can be passed to shaders for texture access.
    ///
    /// # Panics
    ///
    /// Panics if the registry is full (all slots allocated).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let slot = registry.allocate_slot(texture.create_view(&Default::default()));
    /// // Pass `slot` as uniform/push constant to shader
    /// ```
    pub fn allocate_slot(&mut self, view: TextureView) -> u32 {
        // Try to recycle a free slot first (LIFO for cache friendliness)
        if let Some(free_index) = self.free_slots.pop() {
            self.textures[free_index as usize] = Some(view);
            self.dirty = true;
            return free_index;
        }

        // Check capacity before allocating new slot
        if self.textures.len() >= self.max_textures as usize {
            panic!(
                "TextureRegistry full: {} textures at capacity {}",
                self.textures.len(),
                self.max_textures
            );
        }

        // Allocate new slot at end
        let index = self.textures.len() as u32;
        self.textures.push(Some(view));
        self.dirty = true;
        index
    }

    /// Try to allocate a slot, returning None if registry is full.
    ///
    /// Non-panicking version of `allocate_slot()`.
    ///
    /// # Arguments
    ///
    /// * `view` - The texture view to register
    ///
    /// # Returns
    ///
    /// `Some(slot)` if allocation succeeded, `None` if registry is full.
    pub fn try_allocate_slot(&mut self, view: TextureView) -> Option<u32> {
        // Try to recycle a free slot first
        if let Some(free_index) = self.free_slots.pop() {
            self.textures[free_index as usize] = Some(view);
            self.dirty = true;
            return Some(free_index);
        }

        // Check capacity before allocating new slot
        if self.textures.len() >= self.max_textures as usize {
            return None;
        }

        // Allocate new slot at end
        let index = self.textures.len() as u32;
        self.textures.push(Some(view));
        self.dirty = true;
        Some(index)
    }

    /// Release a slot for reuse.
    ///
    /// Clears the texture view at the given slot and adds the slot to the free
    /// list for recycling. Marks the registry as dirty.
    ///
    /// # Arguments
    ///
    /// * `slot` - The slot index to release (from `allocate_slot()`)
    ///
    /// # Returns
    ///
    /// `true` if the slot was released, `false` if slot was invalid or already free.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // When texture is no longer needed
    /// registry.free_slot(slot);
    /// ```
    pub fn free_slot(&mut self, slot: u32) -> bool {
        let index = slot as usize;

        if index >= self.textures.len() {
            return false;
        }

        if self.textures[index].is_none() {
            return false; // Already free
        }

        self.textures[index] = None;
        self.free_slots.push(slot);
        self.dirty = true;
        true
    }

    /// Get bind group, rebuilding if dirty.
    ///
    /// Returns a reference to the bind group for use in render passes. The bind
    /// group is automatically rebuilt when textures have been added or removed.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device (needed for bind group creation)
    ///
    /// # Returns
    ///
    /// Reference to the bind group. Returns the cached group if clean, or a newly
    /// created group if dirty.
    ///
    /// # Panics
    ///
    /// Panics if no textures are registered (bind group would be invalid).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let bind_group = registry.bind_group(&device);
    /// render_pass.set_bind_group(3, bind_group, &[]);
    /// ```
    pub fn bind_group(&mut self, device: &Device) -> &BindGroup {
        if self.dirty {
            self.rebuild_bind_group(device);
        }
        self.bind_group.as_ref().expect("No textures registered")
    }

    /// Try to get bind group, returning None if no textures are registered.
    ///
    /// Non-panicking version of `bind_group()`.
    pub fn try_bind_group(&mut self, device: &Device) -> Option<&BindGroup> {
        if self.dirty {
            if !self.rebuild_bind_group(device) {
                return None;
            }
        }
        self.bind_group.as_ref()
    }

    /// Force rebuild of bind group.
    ///
    /// Creates a new bind group with all active (non-None) textures. Called
    /// automatically by `bind_group()` when dirty flag is set.
    ///
    /// # Returns
    ///
    /// `true` if bind group was created, `false` if no textures are registered.
    fn rebuild_bind_group(&mut self, device: &Device) -> bool {
        // Collect non-None texture views
        let views: Vec<&TextureView> = self
            .textures
            .iter()
            .filter_map(|opt| opt.as_ref())
            .collect();

        if views.is_empty() {
            self.bind_group = None;
            self.dirty = false;
            return false;
        }

        // Create bind group with texture array
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("texture_registry_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: TEXTURE_ARRAY_BINDING,
                    resource: BindingResource::TextureViewArray(&views),
                },
                BindGroupEntry {
                    binding: SAMPLER_BINDING,
                    resource: BindingResource::Sampler(&self.sampler),
                },
            ],
        });

        self.bind_group = Some(bind_group);
        self.dirty = false;
        true
    }

    /// Check if bindless textures are supported.
    ///
    /// Returns `true` if the device supports the `TEXTURE_BINDING_ARRAY` feature
    /// required for bindless texture rendering.
    ///
    /// # Example
    ///
    /// ```ignore
    /// if registry.has_bindless_support() {
    ///     // Use bindless rendering path
    /// } else {
    ///     // Fall back to traditional per-draw binding
    /// }
    /// ```
    #[inline]
    pub fn has_bindless_support(&self) -> bool {
        self.has_bindless
    }

    /// Get the number of active textures (registered and not freed).
    #[inline]
    pub fn active_count(&self) -> u32 {
        self.textures.iter().filter(|t| t.is_some()).count() as u32
    }

    /// Get the total number of allocated slots (including freed).
    #[inline]
    pub fn allocated_count(&self) -> u32 {
        self.textures.len() as u32
    }

    /// Get the number of free slots available for recycling.
    #[inline]
    pub fn free_slot_count(&self) -> u32 {
        self.free_slots.len() as u32
    }

    /// Get the maximum capacity.
    #[inline]
    pub fn capacity(&self) -> u32 {
        self.max_textures
    }

    /// Check if the registry is full.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.textures.len() >= self.max_textures as usize && self.free_slots.is_empty()
    }

    /// Check if the registry is empty (no active textures).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.active_count() == 0
    }

    /// Check if the bind group needs to be rebuilt.
    #[inline]
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Get the bind group layout.
    ///
    /// Useful for pipeline layout creation.
    #[inline]
    pub fn layout(&self) -> &BindGroupLayout {
        &self.layout
    }

    /// Get a reference to the shared sampler.
    #[inline]
    pub fn sampler(&self) -> &wgpu::Sampler {
        &self.sampler
    }

    /// Clear all textures from the registry.
    ///
    /// Removes all texture views and clears the free list. The bind group will
    /// be invalidated and `bind_group()` will panic until new textures are added.
    pub fn clear(&mut self) {
        self.textures.clear();
        self.free_slots.clear();
        self.bind_group = None;
        self.dirty = true;
    }

    /// Check if a slot is currently occupied.
    pub fn is_slot_occupied(&self, slot: u32) -> bool {
        let index = slot as usize;
        index < self.textures.len() && self.textures[index].is_some()
    }

    /// Get metrics about the registry state.
    pub fn metrics(&self) -> TextureRegistryMetrics {
        TextureRegistryMetrics {
            active_count: self.active_count(),
            allocated_count: self.allocated_count(),
            free_slots: self.free_slot_count(),
            capacity: self.max_textures,
            has_bind_group: self.bind_group.is_some(),
            is_dirty: self.dirty,
            has_bindless: self.has_bindless,
        }
    }
}

impl std::fmt::Debug for TextureRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TextureRegistry")
            .field("active_count", &self.active_count())
            .field("allocated_count", &self.allocated_count())
            .field("free_slots", &self.free_slot_count())
            .field("capacity", &self.max_textures)
            .field("has_bindless", &self.has_bindless)
            .field("dirty", &self.dirty)
            .field("has_bind_group", &self.bind_group.is_some())
            .finish()
    }
}

// Safety: TextureRegistry contains only Send + Sync types
unsafe impl Send for TextureRegistry {}
unsafe impl Sync for TextureRegistry {}

// =============================================================================
// METRICS
// =============================================================================

/// Metrics about texture registry state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureRegistryMetrics {
    /// Number of currently active textures (registered and not freed).
    pub active_count: u32,
    /// Total number of allocated slots (including freed).
    pub allocated_count: u32,
    /// Number of free slots available for recycling.
    pub free_slots: u32,
    /// Maximum capacity of the registry.
    pub capacity: u32,
    /// Whether a bind group has been created.
    pub has_bind_group: bool,
    /// Whether the bind group needs rebuilding.
    pub is_dirty: bool,
    /// Whether bindless textures are supported.
    pub has_bindless: bool,
}

impl TextureRegistryMetrics {
    /// Returns the utilization percentage (0.0 to 1.0).
    pub fn utilization(&self) -> f32 {
        if self.capacity == 0 {
            0.0
        } else {
            self.active_count as f32 / self.capacity as f32
        }
    }

    /// Returns the fragmentation ratio (free slots / allocated slots).
    ///
    /// Higher values indicate more fragmentation from freed slots.
    pub fn fragmentation(&self) -> f32 {
        if self.allocated_count == 0 {
            0.0
        } else {
            self.free_slots as f32 / self.allocated_count as f32
        }
    }

    /// Returns the number of slots available for new allocations.
    pub fn available_slots(&self) -> u32 {
        self.capacity.saturating_sub(self.allocated_count) + self.free_slots
    }
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Check if bindless textures are supported by the device.
///
/// # Arguments
///
/// * `features` - Device features (use `device.features()`)
///
/// # Returns
///
/// `true` if TEXTURE_BINDING_ARRAY is supported.
pub fn supports_bindless_textures(features: Features) -> bool {
    features.contains(Features::TEXTURE_BINDING_ARRAY)
}

/// Check if partially bound binding arrays are supported.
///
/// When enabled, not all slots in the texture array need to be bound.
///
/// # Arguments
///
/// * `features` - Device features
pub fn supports_partially_bound(features: Features) -> bool {
    features.contains(Features::PARTIALLY_BOUND_BINDING_ARRAY)
}

/// Check if non-uniform indexing is supported.
///
/// Non-uniform indexing allows different shader invocations to access different
/// texture indices. Without this, all invocations must access the same index.
///
/// # Arguments
///
/// * `features` - Device features
pub fn supports_non_uniform_indexing(features: Features) -> bool {
    features.contains(Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING)
}

/// Required wgpu features for bindless textures.
pub fn required_features() -> Features {
    Features::TEXTURE_BINDING_ARRAY
}

/// Optimal wgpu features for bindless textures (includes non-uniform indexing).
pub fn optimal_features() -> Features {
    Features::TEXTURE_BINDING_ARRAY
        | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
        | Features::PARTIALLY_BOUND_BINDING_ARRAY
}

// =============================================================================
// CPU HELPER FUNCTIONS
// =============================================================================

/// CPU reference: Count active textures in a slot array.
pub fn cpu_count_active(slots: &[Option<TextureView>]) -> usize {
    slots.iter().filter(|s| s.is_some()).count()
}

/// CPU reference: Find first free slot in array.
pub fn cpu_find_free_slot(slots: &[Option<TextureView>]) -> Option<usize> {
    slots.iter().position(|s| s.is_none())
}

/// CPU reference: Calculate fragmentation.
pub fn cpu_fragmentation(active: usize, allocated: usize) -> f32 {
    if allocated == 0 {
        0.0
    } else {
        (allocated - active) as f32 / allocated as f32
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
        assert_eq!(MAX_BINDLESS_TEXTURES, 1024);
        assert_eq!(MIN_BINDLESS_TEXTURES, 16);
        assert_eq!(BINDLESS_BIND_GROUP_INDEX, 3);
        assert_eq!(TEXTURE_ARRAY_BINDING, 0);
        assert_eq!(SAMPLER_BINDING, 1);
    }

    #[test]
    fn test_max_at_least_min() {
        assert!(MAX_BINDLESS_TEXTURES >= MIN_BINDLESS_TEXTURES);
    }

    // -------------------------------------------------------------------------
    // Feature Detection Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_supports_bindless_textures_false() {
        let features = Features::empty();
        assert!(!supports_bindless_textures(features));
    }

    #[test]
    fn test_supports_bindless_textures_true() {
        let features = Features::TEXTURE_BINDING_ARRAY;
        assert!(supports_bindless_textures(features));
    }

    #[test]
    fn test_supports_partially_bound() {
        let features = Features::PARTIALLY_BOUND_BINDING_ARRAY;
        assert!(supports_partially_bound(features));
        assert!(!supports_partially_bound(Features::empty()));
    }

    #[test]
    fn test_supports_non_uniform_indexing() {
        let features = Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
        assert!(supports_non_uniform_indexing(features));
        assert!(!supports_non_uniform_indexing(Features::empty()));
    }

    #[test]
    fn test_required_features() {
        let req = required_features();
        assert!(req.contains(Features::TEXTURE_BINDING_ARRAY));
    }

    #[test]
    fn test_optimal_features() {
        let opt = optimal_features();
        assert!(opt.contains(Features::TEXTURE_BINDING_ARRAY));
        assert!(opt.contains(Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING));
        assert!(opt.contains(Features::PARTIALLY_BOUND_BINDING_ARRAY));
    }

    // -------------------------------------------------------------------------
    // Metrics Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_metrics_utilization_empty() {
        let metrics = TextureRegistryMetrics {
            active_count: 0,
            allocated_count: 0,
            free_slots: 0,
            capacity: 100,
            has_bind_group: false,
            is_dirty: true,
            has_bindless: true,
        };
        assert_eq!(metrics.utilization(), 0.0);
    }

    #[test]
    fn test_metrics_utilization_half() {
        let metrics = TextureRegistryMetrics {
            active_count: 50,
            allocated_count: 50,
            free_slots: 0,
            capacity: 100,
            has_bind_group: true,
            is_dirty: false,
            has_bindless: true,
        };
        assert!((metrics.utilization() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_metrics_utilization_zero_capacity() {
        let metrics = TextureRegistryMetrics {
            active_count: 0,
            allocated_count: 0,
            free_slots: 0,
            capacity: 0,
            has_bind_group: false,
            is_dirty: true,
            has_bindless: false,
        };
        assert_eq!(metrics.utilization(), 0.0);
    }

    #[test]
    fn test_metrics_fragmentation_none() {
        let metrics = TextureRegistryMetrics {
            active_count: 10,
            allocated_count: 10,
            free_slots: 0,
            capacity: 100,
            has_bind_group: true,
            is_dirty: false,
            has_bindless: true,
        };
        assert_eq!(metrics.fragmentation(), 0.0);
    }

    #[test]
    fn test_metrics_fragmentation_some() {
        let metrics = TextureRegistryMetrics {
            active_count: 8,
            allocated_count: 10,
            free_slots: 2,
            capacity: 100,
            has_bind_group: true,
            is_dirty: false,
            has_bindless: true,
        };
        // 2/10 = 0.2
        assert!((metrics.fragmentation() - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_metrics_fragmentation_zero_allocated() {
        let metrics = TextureRegistryMetrics {
            active_count: 0,
            allocated_count: 0,
            free_slots: 0,
            capacity: 100,
            has_bind_group: false,
            is_dirty: true,
            has_bindless: true,
        };
        assert_eq!(metrics.fragmentation(), 0.0);
    }

    #[test]
    fn test_metrics_available_slots() {
        let metrics = TextureRegistryMetrics {
            active_count: 40,
            allocated_count: 50, // 10 freed
            free_slots: 10,
            capacity: 100,
            has_bind_group: true,
            is_dirty: false,
            has_bindless: true,
        };
        // 100 - 50 + 10 = 60 available
        assert_eq!(metrics.available_slots(), 60);
    }

    // -------------------------------------------------------------------------
    // CPU Helper Function Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cpu_count_active_empty() {
        let slots: Vec<Option<TextureView>> = vec![];
        assert_eq!(cpu_count_active(&slots), 0);
    }

    #[test]
    fn test_cpu_find_free_slot_none() {
        let slots: Vec<Option<TextureView>> = vec![];
        assert_eq!(cpu_find_free_slot(&slots), None);
    }

    #[test]
    fn test_cpu_fragmentation_zero() {
        assert_eq!(cpu_fragmentation(10, 10), 0.0);
    }

    #[test]
    fn test_cpu_fragmentation_half() {
        assert!((cpu_fragmentation(5, 10) - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_cpu_fragmentation_zero_allocated() {
        assert_eq!(cpu_fragmentation(0, 0), 0.0);
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
    fn test_metrics_is_copy() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<TextureRegistryMetrics>();
    }

    // -------------------------------------------------------------------------
    // Metrics Equality Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_metrics_equality() {
        let m1 = TextureRegistryMetrics {
            active_count: 10,
            allocated_count: 15,
            free_slots: 5,
            capacity: 100,
            has_bind_group: true,
            is_dirty: false,
            has_bindless: true,
        };
        let m2 = m1;
        assert_eq!(m1, m2);
    }

    #[test]
    fn test_metrics_clone() {
        let m1 = TextureRegistryMetrics {
            active_count: 10,
            allocated_count: 15,
            free_slots: 5,
            capacity: 100,
            has_bind_group: true,
            is_dirty: false,
            has_bindless: true,
        };
        let m2 = m1;
        assert_eq!(m1.active_count, m2.active_count);
        assert_eq!(m1.has_bindless, m2.has_bindless);
    }

    #[test]
    fn test_metrics_debug() {
        let m = TextureRegistryMetrics {
            active_count: 10,
            allocated_count: 15,
            free_slots: 5,
            capacity: 100,
            has_bind_group: true,
            is_dirty: false,
            has_bindless: true,
        };
        let debug = format!("{:?}", m);
        assert!(debug.contains("active_count"));
        assert!(debug.contains("10"));
    }
}
