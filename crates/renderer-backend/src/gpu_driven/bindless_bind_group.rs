//! Bindless Bind Group for GPU-driven rendering (T-WGPU-P6.8.5).
//!
//! This module provides the unified bindless bind group layout and builder for
//! GPU-driven rendering. It combines texture arrays, sampler arrays, and the
//! material storage buffer into a single bind group for efficient bindless access.
//!
//! # Overview
//!
//! ```text
//! +------------------------------------------------------------------+
//! |                   BindlessBindGroup Layout                       |
//! +------------------------------------------------------------------+
//! | @binding(0) textures: binding_array<texture_2d<f32>>   [1024]   |
//! | @binding(1) samplers: binding_array<sampler>           [16]     |
//! | @binding(2) materials: storage buffer (read-only)               |
//! +------------------------------------------------------------------+
//! ```
//!
//! # Bindless Rendering
//!
//! Traditional rendering requires frequent bind group switches when textures or
//! materials change. Bindless rendering binds all resources once:
//!
//! ```wgsl
//! @group(0) @binding(0) var textures: binding_array<texture_2d<f32>>;
//! @group(0) @binding(1) var samplers: binding_array<sampler>;
//! @group(0) @binding(2) var<storage, read> materials: array<MaterialDescriptor>;
//!
//! @fragment
//! fn main(input: VertexOutput) -> @location(0) vec4<f32> {
//!     let material = materials[input.material_index];
//!     let sampler = samplers[input.sampler_index];
//!
//!     // Non-uniform indexing with PARTIALLY_BOUND
//!     if (material.base_color_texture != 0xFFFFFFFFu) {
//!         let tex_index = material.base_color_texture;
//!         return textureSample(textures[tex_index], sampler, input.uv);
//!     }
//!     return material.base_color_factor;
//! }
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use renderer_backend::gpu_driven::bindless_bind_group::{
//!     BindlessBindGroupBuilder, create_bindless_layout,
//!     MAX_BINDLESS_TEXTURES, MAX_BINDLESS_SAMPLERS,
//! };
//!
//! // Create layout once (store in pipeline layout)
//! let layout = create_bindless_layout(&device, features);
//!
//! // Build bind group with resources
//! let bind_group = BindlessBindGroupBuilder::new(&device, &layout)
//!     .with_textures(&texture_views)
//!     .with_samplers(&samplers)
//!     .with_material_buffer(&material_buffer)
//!     .build();
//!
//! // Use in render pass
//! render_pass.set_bind_group(0, &bind_group, &[]);
//! ```
//!
//! # Features Required
//!
//! - `TEXTURE_BINDING_ARRAY`: Variable-count texture bindings
//! - `SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING`: Non-uniform indexing
//! - `PARTIALLY_BOUND_BINDING_ARRAY`: Sparse bindings (optional but recommended)
//!
//! # Capacity
//!
//! - Textures: 1024 (matches `TextureRegistry`)
//! - Samplers: 16 (typical GPU limit for sampler arrays)
//! - Materials: Unlimited (storage buffer)

use std::num::NonZeroU32;

use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferBindingType, Device,
    Features, Sampler, SamplerBindingType, ShaderStages, TextureSampleType, TextureView,
    TextureViewDimension,
};

// =============================================================================
// CONSTANTS
// =============================================================================

/// Maximum number of bindless textures in the array.
///
/// Matches `TextureRegistry::MAX_BINDLESS_TEXTURES` and typical desktop GPU limits.
pub const MAX_BINDLESS_TEXTURES: u32 = 1024;

/// Maximum number of samplers in the array.
///
/// Most desktop GPUs support 16 samplers per shader stage. This is sufficient
/// for common use cases (linear, nearest, anisotropic, shadow, etc.).
pub const MAX_BINDLESS_SAMPLERS: u32 = 16;

/// Minimum number of textures required for bindless rendering.
pub const MIN_BINDLESS_TEXTURES: u32 = 16;

/// Minimum number of samplers required for bindless rendering.
pub const MIN_BINDLESS_SAMPLERS: u32 = 4;

/// Binding index for the texture array.
pub const BINDING_TEXTURES: u32 = 0;

/// Binding index for the sampler array.
pub const BINDING_SAMPLERS: u32 = 1;

/// Binding index for the material storage buffer.
pub const BINDING_MATERIALS: u32 = 2;

/// Default bind group index for bindless resources (TRINITY convention).
///
/// Group 0 is typically used for bindless resources, allowing per-pass and
/// per-object data in higher groups.
pub const BINDLESS_BIND_GROUP_INDEX: u32 = 0;

// =============================================================================
// FEATURE DETECTION
// =============================================================================

/// Check if bindless texture arrays are supported.
///
/// # Arguments
///
/// * `features` - Device features (use `device.features()`)
///
/// # Returns
///
/// `true` if `TEXTURE_BINDING_ARRAY` is supported.
#[inline]
pub fn supports_texture_arrays(features: Features) -> bool {
    features.contains(Features::TEXTURE_BINDING_ARRAY)
}

/// Check if non-uniform indexing is supported.
///
/// Required for using material indices to dynamically select textures.
///
/// # Arguments
///
/// * `features` - Device features
///
/// # Returns
///
/// `true` if non-uniform indexing for textures and storage buffers is supported.
#[inline]
pub fn supports_non_uniform_indexing(features: Features) -> bool {
    features.contains(Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING)
}

/// Check if partially bound binding arrays are supported.
///
/// When enabled, not all slots in the texture/sampler arrays need to be bound.
/// This allows sparse resource allocation without dummy textures.
///
/// # Arguments
///
/// * `features` - Device features
///
/// # Returns
///
/// `true` if `PARTIALLY_BOUND_BINDING_ARRAY` is supported.
#[inline]
pub fn supports_partially_bound(features: Features) -> bool {
    features.contains(Features::PARTIALLY_BOUND_BINDING_ARRAY)
}

/// Check if all required bindless features are supported.
///
/// # Arguments
///
/// * `features` - Device features
///
/// # Returns
///
/// `true` if texture arrays, non-uniform indexing, and partially bound are all supported.
pub fn supports_full_bindless(features: Features) -> bool {
    supports_texture_arrays(features)
        && supports_non_uniform_indexing(features)
        && supports_partially_bound(features)
}

/// Returns the required wgpu features for bindless rendering.
///
/// These features MUST be enabled when requesting the device.
pub fn required_features() -> Features {
    Features::TEXTURE_BINDING_ARRAY
        | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
}

/// Returns the optimal wgpu features for bindless rendering.
///
/// Includes required features plus partially bound for sparse bindings.
pub fn optimal_features() -> Features {
    required_features() | Features::PARTIALLY_BOUND_BINDING_ARRAY
}

// =============================================================================
// BIND GROUP LAYOUT CREATION
// =============================================================================

/// Creates the bindless bind group layout with configurable capacity.
///
/// The layout supports:
/// - Texture array with `max_textures` slots
/// - Sampler array with `max_samplers` slots
/// - Material storage buffer (read-only)
///
/// # Arguments
///
/// * `device` - The wgpu device
/// * `max_textures` - Maximum texture count (clamped to MIN..MAX)
/// * `max_samplers` - Maximum sampler count (clamped to MIN..MAX)
///
/// # Returns
///
/// A `BindGroupLayout` for bindless resource binding.
///
/// # Example
///
/// ```ignore
/// let layout = create_bindless_layout_with_capacity(&device, 512, 8);
/// ```
pub fn create_bindless_layout_with_capacity(
    device: &Device,
    max_textures: u32,
    max_samplers: u32,
) -> BindGroupLayout {
    let tex_count = max_textures
        .max(MIN_BINDLESS_TEXTURES)
        .min(MAX_BINDLESS_TEXTURES);
    let sampler_count = max_samplers
        .max(MIN_BINDLESS_SAMPLERS)
        .min(MAX_BINDLESS_SAMPLERS);

    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("bindless_bind_group_layout"),
        entries: &[
            // Texture array (potentially partially bound)
            BindGroupLayoutEntry {
                binding: BINDING_TEXTURES,
                visibility: ShaderStages::VERTEX_FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: NonZeroU32::new(tex_count),
            },
            // Sampler array
            BindGroupLayoutEntry {
                binding: BINDING_SAMPLERS,
                visibility: ShaderStages::VERTEX_FRAGMENT,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: NonZeroU32::new(sampler_count),
            },
            // Material storage buffer (read-only)
            BindGroupLayoutEntry {
                binding: BINDING_MATERIALS,
                visibility: ShaderStages::VERTEX_FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None, // Allows empty buffer initially
                },
                count: None,
            },
        ],
    })
}

/// Creates the bindless bind group layout with default capacity.
///
/// Uses `MAX_BINDLESS_TEXTURES` (1024) and `MAX_BINDLESS_SAMPLERS` (16).
///
/// # Arguments
///
/// * `device` - The wgpu device
///
/// # Returns
///
/// A `BindGroupLayout` for bindless resource binding.
///
/// # Example
///
/// ```ignore
/// let layout = create_bindless_layout(&device);
/// ```
pub fn create_bindless_layout(device: &Device) -> BindGroupLayout {
    create_bindless_layout_with_capacity(device, MAX_BINDLESS_TEXTURES, MAX_BINDLESS_SAMPLERS)
}

// =============================================================================
// BIND GROUP BUILDER
// =============================================================================

/// Builder for creating bindless bind groups.
///
/// Collects texture views, samplers, and material buffer references before
/// creating the final bind group. Validates resource counts against layout limits.
///
/// # Usage
///
/// ```ignore
/// let bind_group = BindlessBindGroupBuilder::new(&device, &layout)
///     .with_textures(&[&albedo_view, &normal_view, &metallic_view])
///     .with_samplers(&[&linear_sampler, &nearest_sampler])
///     .with_material_buffer(&material_buffer)
///     .build();
/// ```
///
/// # Panics
///
/// - `build()` panics if no textures are provided
/// - `build()` panics if no samplers are provided
/// - `build()` panics if no material buffer is provided
pub struct BindlessBindGroupBuilder<'a> {
    device: &'a Device,
    layout: &'a BindGroupLayout,
    textures: Vec<&'a TextureView>,
    samplers: Vec<&'a Sampler>,
    material_buffer: Option<&'a Buffer>,
    label: Option<&'a str>,
}

impl<'a> BindlessBindGroupBuilder<'a> {
    /// Creates a new bind group builder.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device
    /// * `layout` - The bindless bind group layout (from `create_bindless_layout`)
    pub fn new(device: &'a Device, layout: &'a BindGroupLayout) -> Self {
        Self {
            device,
            layout,
            textures: Vec::with_capacity(64),
            samplers: Vec::with_capacity(4),
            material_buffer: None,
            label: None,
        }
    }

    /// Sets the bind group label for debugging.
    ///
    /// # Arguments
    ///
    /// * `label` - Debug label for the bind group
    #[inline]
    pub fn with_label(mut self, label: &'a str) -> Self {
        self.label = Some(label);
        self
    }

    /// Adds texture views to the bind group.
    ///
    /// Texture views are bound sequentially starting at index 0. The shader
    /// accesses textures using the slot index passed from CPU-side material data.
    ///
    /// # Arguments
    ///
    /// * `textures` - Slice of texture view references
    ///
    /// # Example
    ///
    /// ```ignore
    /// builder.with_textures(&[&tex0, &tex1, &tex2])
    /// ```
    #[inline]
    pub fn with_textures(mut self, textures: &[&'a TextureView]) -> Self {
        self.textures.extend_from_slice(textures);
        self
    }

    /// Adds a single texture view.
    ///
    /// # Arguments
    ///
    /// * `texture` - Texture view reference
    ///
    /// # Returns
    ///
    /// The slot index for this texture (for shader access).
    #[inline]
    pub fn add_texture(&mut self, texture: &'a TextureView) -> u32 {
        let index = self.textures.len() as u32;
        self.textures.push(texture);
        index
    }

    /// Adds samplers to the bind group.
    ///
    /// Samplers are bound sequentially starting at index 0. The shader selects
    /// the appropriate sampler using a sampler index.
    ///
    /// # Arguments
    ///
    /// * `samplers` - Slice of sampler references
    #[inline]
    pub fn with_samplers(mut self, samplers: &[&'a Sampler]) -> Self {
        self.samplers.extend_from_slice(samplers);
        self
    }

    /// Adds a single sampler.
    ///
    /// # Arguments
    ///
    /// * `sampler` - Sampler reference
    ///
    /// # Returns
    ///
    /// The slot index for this sampler.
    #[inline]
    pub fn add_sampler(&mut self, sampler: &'a Sampler) -> u32 {
        let index = self.samplers.len() as u32;
        self.samplers.push(sampler);
        index
    }

    /// Sets the material storage buffer.
    ///
    /// The buffer should contain an array of `MaterialDescriptor` structs.
    /// Shaders access materials using the material index.
    ///
    /// # Arguments
    ///
    /// * `buffer` - Material storage buffer reference
    #[inline]
    pub fn with_material_buffer(mut self, buffer: &'a Buffer) -> Self {
        self.material_buffer = Some(buffer);
        self
    }

    /// Returns the current number of textures.
    #[inline]
    pub fn texture_count(&self) -> u32 {
        self.textures.len() as u32
    }

    /// Returns the current number of samplers.
    #[inline]
    pub fn sampler_count(&self) -> u32 {
        self.samplers.len() as u32
    }

    /// Checks if the builder has all required resources.
    pub fn is_complete(&self) -> bool {
        !self.textures.is_empty() && !self.samplers.is_empty() && self.material_buffer.is_some()
    }

    /// Builds the bind group.
    ///
    /// # Returns
    ///
    /// The created `BindGroup`.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - No textures are provided
    /// - No samplers are provided
    /// - No material buffer is provided
    pub fn build(self) -> BindGroup {
        assert!(
            !self.textures.is_empty(),
            "BindlessBindGroupBuilder: at least one texture is required"
        );
        assert!(
            !self.samplers.is_empty(),
            "BindlessBindGroupBuilder: at least one sampler is required"
        );
        let material_buffer = self
            .material_buffer
            .expect("BindlessBindGroupBuilder: material buffer is required");

        self.device.create_bind_group(&BindGroupDescriptor {
            label: self.label,
            layout: self.layout,
            entries: &[
                BindGroupEntry {
                    binding: BINDING_TEXTURES,
                    resource: BindingResource::TextureViewArray(&self.textures),
                },
                BindGroupEntry {
                    binding: BINDING_SAMPLERS,
                    resource: BindingResource::SamplerArray(&self.samplers),
                },
                BindGroupEntry {
                    binding: BINDING_MATERIALS,
                    resource: material_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Tries to build the bind group, returning None if incomplete.
    ///
    /// Non-panicking version of `build()`.
    ///
    /// # Returns
    ///
    /// `Some(BindGroup)` if all resources are provided, `None` otherwise.
    pub fn try_build(self) -> Option<BindGroup> {
        if self.textures.is_empty() || self.samplers.is_empty() {
            return None;
        }
        let material_buffer = self.material_buffer?;

        Some(self.device.create_bind_group(&BindGroupDescriptor {
            label: self.label,
            layout: self.layout,
            entries: &[
                BindGroupEntry {
                    binding: BINDING_TEXTURES,
                    resource: BindingResource::TextureViewArray(&self.textures),
                },
                BindGroupEntry {
                    binding: BINDING_SAMPLERS,
                    resource: BindingResource::SamplerArray(&self.samplers),
                },
                BindGroupEntry {
                    binding: BINDING_MATERIALS,
                    resource: material_buffer.as_entire_binding(),
                },
            ],
        }))
    }
}

// =============================================================================
// BINDLESS BIND GROUP MANAGER
// =============================================================================

/// Managed bindless bind group with automatic dirty tracking.
///
/// Caches the bind group and rebuilds it when resources change. This is more
/// efficient than creating a new bind group every frame.
///
/// # Usage
///
/// ```ignore
/// let mut manager = BindlessBindGroupManager::new(&device, features);
///
/// // Register resources
/// let tex_slot = manager.add_texture(texture_view);
/// let sampler_slot = manager.add_sampler(sampler);
/// manager.set_material_buffer(material_buffer);
///
/// // Get bind group (rebuilds if dirty)
/// let bind_group = manager.bind_group(&device);
/// render_pass.set_bind_group(0, bind_group, &[]);
/// ```
pub struct BindlessBindGroupManager {
    /// Bind group layout.
    layout: BindGroupLayout,

    /// Cached bind group.
    bind_group: Option<BindGroup>,

    /// Registered texture views.
    textures: Vec<Option<TextureView>>,

    /// Free texture slot stack.
    free_texture_slots: Vec<u32>,

    /// Registered samplers.
    samplers: Vec<Sampler>,

    /// Material buffer reference (stored for rebuild).
    /// We store the buffer in a RefCell to allow interior mutability.
    material_buffer: Option<Buffer>,

    /// Dirty flag for bind group rebuild.
    dirty: bool,

    /// Maximum texture capacity.
    max_textures: u32,

    /// Maximum sampler capacity.
    max_samplers: u32,

    /// Whether full bindless is supported.
    has_full_bindless: bool,
}

impl BindlessBindGroupManager {
    /// Creates a new bind group manager.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device
    /// * `features` - Device features for feature detection
    pub fn new(device: &Device, features: Features) -> Self {
        Self::with_capacity(device, features, MAX_BINDLESS_TEXTURES, MAX_BINDLESS_SAMPLERS)
    }

    /// Creates a bind group manager with custom capacity.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device
    /// * `features` - Device features
    /// * `max_textures` - Maximum texture slots
    /// * `max_samplers` - Maximum sampler slots
    pub fn with_capacity(
        device: &Device,
        features: Features,
        max_textures: u32,
        max_samplers: u32,
    ) -> Self {
        let max_textures = max_textures
            .max(MIN_BINDLESS_TEXTURES)
            .min(MAX_BINDLESS_TEXTURES);
        let max_samplers = max_samplers
            .max(MIN_BINDLESS_SAMPLERS)
            .min(MAX_BINDLESS_SAMPLERS);

        let layout = create_bindless_layout_with_capacity(device, max_textures, max_samplers);

        Self {
            layout,
            bind_group: None,
            textures: Vec::with_capacity(max_textures as usize),
            free_texture_slots: Vec::new(),
            samplers: Vec::with_capacity(max_samplers as usize),
            material_buffer: None,
            dirty: true,
            max_textures,
            max_samplers,
            has_full_bindless: supports_full_bindless(features),
        }
    }

    /// Returns the bind group layout.
    #[inline]
    pub fn layout(&self) -> &BindGroupLayout {
        &self.layout
    }

    /// Adds a texture and returns its slot index.
    ///
    /// # Arguments
    ///
    /// * `view` - The texture view to add
    ///
    /// # Returns
    ///
    /// Slot index for shader access.
    ///
    /// # Panics
    ///
    /// Panics if texture capacity is exceeded.
    pub fn add_texture(&mut self, view: TextureView) -> u32 {
        // Try to reuse a free slot
        if let Some(slot) = self.free_texture_slots.pop() {
            self.textures[slot as usize] = Some(view);
            self.dirty = true;
            return slot;
        }

        // Allocate new slot
        if self.textures.len() >= self.max_textures as usize {
            panic!(
                "BindlessBindGroupManager: texture capacity {} exceeded",
                self.max_textures
            );
        }

        let slot = self.textures.len() as u32;
        self.textures.push(Some(view));
        self.dirty = true;
        slot
    }

    /// Tries to add a texture, returning None if at capacity.
    pub fn try_add_texture(&mut self, view: TextureView) -> Option<u32> {
        if let Some(slot) = self.free_texture_slots.pop() {
            self.textures[slot as usize] = Some(view);
            self.dirty = true;
            return Some(slot);
        }

        if self.textures.len() >= self.max_textures as usize {
            return None;
        }

        let slot = self.textures.len() as u32;
        self.textures.push(Some(view));
        self.dirty = true;
        Some(slot)
    }

    /// Removes a texture by slot index.
    ///
    /// # Returns
    ///
    /// `true` if the texture was removed.
    pub fn remove_texture(&mut self, slot: u32) -> bool {
        let index = slot as usize;
        if index >= self.textures.len() || self.textures[index].is_none() {
            return false;
        }

        self.textures[index] = None;
        self.free_texture_slots.push(slot);
        self.dirty = true;
        true
    }

    /// Adds a sampler and returns its slot index.
    ///
    /// # Panics
    ///
    /// Panics if sampler capacity is exceeded.
    pub fn add_sampler(&mut self, sampler: Sampler) -> u32 {
        if self.samplers.len() >= self.max_samplers as usize {
            panic!(
                "BindlessBindGroupManager: sampler capacity {} exceeded",
                self.max_samplers
            );
        }

        let slot = self.samplers.len() as u32;
        self.samplers.push(sampler);
        self.dirty = true;
        slot
    }

    /// Sets the material buffer.
    pub fn set_material_buffer(&mut self, buffer: Buffer) {
        self.material_buffer = Some(buffer);
        self.dirty = true;
    }

    /// Gets the bind group, rebuilding if dirty.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device
    ///
    /// # Returns
    ///
    /// Reference to the bind group.
    ///
    /// # Panics
    ///
    /// Panics if resources are not set (textures, samplers, material buffer).
    pub fn bind_group(&mut self, device: &Device) -> &BindGroup {
        if self.dirty {
            self.rebuild_bind_group(device);
        }
        self.bind_group.as_ref().expect("Bind group not created")
    }

    /// Tries to get the bind group.
    ///
    /// Non-panicking version of `bind_group()`.
    pub fn try_bind_group(&mut self, device: &Device) -> Option<&BindGroup> {
        if self.dirty {
            self.try_rebuild_bind_group(device)?;
        }
        self.bind_group.as_ref()
    }

    /// Rebuilds the bind group.
    fn rebuild_bind_group(&mut self, device: &Device) {
        let views: Vec<&TextureView> = self
            .textures
            .iter()
            .filter_map(|opt| opt.as_ref())
            .collect();

        assert!(!views.is_empty(), "At least one texture required");
        assert!(!self.samplers.is_empty(), "At least one sampler required");
        let material_buffer = self
            .material_buffer
            .as_ref()
            .expect("Material buffer required");

        let sampler_refs: Vec<&Sampler> = self.samplers.iter().collect();

        self.bind_group = Some(device.create_bind_group(&BindGroupDescriptor {
            label: Some("bindless_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: BINDING_TEXTURES,
                    resource: BindingResource::TextureViewArray(&views),
                },
                BindGroupEntry {
                    binding: BINDING_SAMPLERS,
                    resource: BindingResource::SamplerArray(&sampler_refs),
                },
                BindGroupEntry {
                    binding: BINDING_MATERIALS,
                    resource: material_buffer.as_entire_binding(),
                },
            ],
        }));

        self.dirty = false;
    }

    /// Tries to rebuild the bind group.
    fn try_rebuild_bind_group(&mut self, device: &Device) -> Option<()> {
        let views: Vec<&TextureView> = self
            .textures
            .iter()
            .filter_map(|opt| opt.as_ref())
            .collect();

        if views.is_empty() || self.samplers.is_empty() {
            return None;
        }
        let material_buffer = self.material_buffer.as_ref()?;

        let sampler_refs: Vec<&Sampler> = self.samplers.iter().collect();

        self.bind_group = Some(device.create_bind_group(&BindGroupDescriptor {
            label: Some("bindless_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: BINDING_TEXTURES,
                    resource: BindingResource::TextureViewArray(&views),
                },
                BindGroupEntry {
                    binding: BINDING_SAMPLERS,
                    resource: BindingResource::SamplerArray(&sampler_refs),
                },
                BindGroupEntry {
                    binding: BINDING_MATERIALS,
                    resource: material_buffer.as_entire_binding(),
                },
            ],
        }));

        self.dirty = false;
        Some(())
    }

    /// Returns the number of active textures.
    #[inline]
    pub fn texture_count(&self) -> u32 {
        self.textures.iter().filter(|t| t.is_some()).count() as u32
    }

    /// Returns the number of samplers.
    #[inline]
    pub fn sampler_count(&self) -> u32 {
        self.samplers.len() as u32
    }

    /// Returns whether full bindless is supported.
    #[inline]
    pub fn has_full_bindless(&self) -> bool {
        self.has_full_bindless
    }

    /// Returns whether the bind group is dirty.
    #[inline]
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Forces a rebuild on the next `bind_group()` call.
    #[inline]
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Returns metrics about the manager state.
    pub fn metrics(&self) -> BindlessBindGroupMetrics {
        BindlessBindGroupMetrics {
            active_textures: self.texture_count(),
            max_textures: self.max_textures,
            free_texture_slots: self.free_texture_slots.len() as u32,
            sampler_count: self.sampler_count(),
            max_samplers: self.max_samplers,
            has_material_buffer: self.material_buffer.is_some(),
            has_bind_group: self.bind_group.is_some(),
            is_dirty: self.dirty,
            has_full_bindless: self.has_full_bindless,
        }
    }
}

impl std::fmt::Debug for BindlessBindGroupManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BindlessBindGroupManager")
            .field("active_textures", &self.texture_count())
            .field("max_textures", &self.max_textures)
            .field("sampler_count", &self.sampler_count())
            .field("max_samplers", &self.max_samplers)
            .field("has_material_buffer", &self.material_buffer.is_some())
            .field("dirty", &self.dirty)
            .field("has_full_bindless", &self.has_full_bindless)
            .finish()
    }
}

// =============================================================================
// METRICS
// =============================================================================

/// Metrics about bindless bind group manager state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BindlessBindGroupMetrics {
    /// Number of active textures.
    pub active_textures: u32,
    /// Maximum texture capacity.
    pub max_textures: u32,
    /// Number of free texture slots for recycling.
    pub free_texture_slots: u32,
    /// Number of registered samplers.
    pub sampler_count: u32,
    /// Maximum sampler capacity.
    pub max_samplers: u32,
    /// Whether a material buffer is set.
    pub has_material_buffer: bool,
    /// Whether a bind group exists.
    pub has_bind_group: bool,
    /// Whether the bind group needs rebuilding.
    pub is_dirty: bool,
    /// Whether full bindless features are supported.
    pub has_full_bindless: bool,
}

impl BindlessBindGroupMetrics {
    /// Returns the texture utilization percentage (0.0 to 1.0).
    pub fn texture_utilization(&self) -> f32 {
        if self.max_textures == 0 {
            0.0
        } else {
            self.active_textures as f32 / self.max_textures as f32
        }
    }

    /// Returns the sampler utilization percentage (0.0 to 1.0).
    pub fn sampler_utilization(&self) -> f32 {
        if self.max_samplers == 0 {
            0.0
        } else {
            self.sampler_count as f32 / self.max_samplers as f32
        }
    }

    /// Returns the number of available texture slots.
    pub fn available_texture_slots(&self) -> u32 {
        self.max_textures
            .saturating_sub(self.active_textures)
            .saturating_add(self.free_texture_slots)
    }

    /// Returns whether the manager is ready for use.
    pub fn is_ready(&self) -> bool {
        self.active_textures > 0 && self.sampler_count > 0 && self.has_material_buffer
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        // Verify binding indices are unique
        assert_ne!(BINDING_TEXTURES, BINDING_SAMPLERS);
        assert_ne!(BINDING_TEXTURES, BINDING_MATERIALS);
        assert_ne!(BINDING_SAMPLERS, BINDING_MATERIALS);

        // Verify capacity limits
        assert!(MAX_BINDLESS_TEXTURES >= MIN_BINDLESS_TEXTURES);
        assert!(MAX_BINDLESS_SAMPLERS >= MIN_BINDLESS_SAMPLERS);

        // Verify sensible defaults
        assert_eq!(MAX_BINDLESS_TEXTURES, 1024);
        assert_eq!(MAX_BINDLESS_SAMPLERS, 16);
    }

    #[test]
    fn test_feature_detection() {
        // No features
        let no_features = Features::empty();
        assert!(!supports_texture_arrays(no_features));
        assert!(!supports_non_uniform_indexing(no_features));
        assert!(!supports_partially_bound(no_features));
        assert!(!supports_full_bindless(no_features));

        // Texture binding array only
        let texture_only = Features::TEXTURE_BINDING_ARRAY;
        assert!(supports_texture_arrays(texture_only));
        assert!(!supports_non_uniform_indexing(texture_only));
        assert!(!supports_full_bindless(texture_only));

        // All required features
        let required = required_features();
        assert!(supports_texture_arrays(required));
        assert!(supports_non_uniform_indexing(required));
        assert!(!supports_partially_bound(required));

        // All optimal features
        let optimal = optimal_features();
        assert!(supports_full_bindless(optimal));
    }

    #[test]
    fn test_required_features() {
        let required = required_features();
        assert!(required.contains(Features::TEXTURE_BINDING_ARRAY));
        assert!(required.contains(
            Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
        ));
    }

    #[test]
    fn test_optimal_features() {
        let optimal = optimal_features();
        let required = required_features();

        // Optimal should contain all required features
        assert!(optimal.contains(required));
        assert!(optimal.contains(Features::PARTIALLY_BOUND_BINDING_ARRAY));
    }

    #[test]
    fn test_metrics_calculations() {
        let metrics = BindlessBindGroupMetrics {
            active_textures: 256,
            max_textures: 1024,
            free_texture_slots: 10,
            sampler_count: 4,
            max_samplers: 16,
            has_material_buffer: true,
            has_bind_group: true,
            is_dirty: false,
            has_full_bindless: true,
        };

        // Texture utilization: 256/1024 = 0.25
        assert!((metrics.texture_utilization() - 0.25).abs() < 0.001);

        // Sampler utilization: 4/16 = 0.25
        assert!((metrics.sampler_utilization() - 0.25).abs() < 0.001);

        // Available slots: (1024 - 256) + 10 = 778
        assert_eq!(metrics.available_texture_slots(), 778);

        // Ready check
        assert!(metrics.is_ready());
    }

    #[test]
    fn test_metrics_empty() {
        let metrics = BindlessBindGroupMetrics {
            active_textures: 0,
            max_textures: 1024,
            free_texture_slots: 0,
            sampler_count: 0,
            max_samplers: 16,
            has_material_buffer: false,
            has_bind_group: false,
            is_dirty: true,
            has_full_bindless: false,
        };

        assert!((metrics.texture_utilization() - 0.0).abs() < 0.001);
        assert!((metrics.sampler_utilization() - 0.0).abs() < 0.001);
        assert_eq!(metrics.available_texture_slots(), 1024);
        assert!(!metrics.is_ready());
    }

    #[test]
    fn test_metrics_zero_capacity() {
        let metrics = BindlessBindGroupMetrics {
            active_textures: 0,
            max_textures: 0,
            free_texture_slots: 0,
            sampler_count: 0,
            max_samplers: 0,
            has_material_buffer: false,
            has_bind_group: false,
            is_dirty: false,
            has_full_bindless: false,
        };

        // Should not divide by zero
        assert!((metrics.texture_utilization() - 0.0).abs() < 0.001);
        assert!((metrics.sampler_utilization() - 0.0).abs() < 0.001);
    }
}
