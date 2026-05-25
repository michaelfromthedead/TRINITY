//! RHI Bind Group mapping layer.
//!
//! Bind groups connect shader resources (buffers, textures, samplers) to
//! pipeline layouts.  This module provides a thin abstraction over
//! [`wgpu::BindGroup`] and [`wgpu::BindGroupLayout`] with a frame-scoped
//! cache that avoids redundant GPU object creation.
//!
//! # Architecture
//!
//! ```text
//! BindingResource { binding_index, entry, visibility }
//!       |
//!       v
//! BindGroupLayout -- holds the wgpu layout + resource descriptors
//! BindGroup       -- wraps wgpu::BindGroup
//! BindGroupCache  -- frame-scoped HashMap<(u64, u64), BindGroup>
//! ```
//!
//! The cache key is a pair of 64-bit hashes: one for the layout descriptor
//! and one for the concrete resource bindings.  Call [`evict_frame`] at the
//! start of each frame to purge stale entries.
//!
//! [`evict_frame`]: BindGroupCache::evict_frame

use std::collections::HashMap;
use std::hash::Hasher;

use crate::rhi_resources::{RhiBuffer, RhiSampler, RhiTexture};

// ---------------------------------------------------------------------------
// ShaderVisibility  (bitmask)
// ---------------------------------------------------------------------------

/// Bitmask describing which shader stages a binding is visible to.
///
/// Maps to [`wgpu::ShaderStages`] when the bind-group layout is created.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShaderVisibility(u32);

impl ShaderVisibility {
    /// Visible to the vertex shader.
    pub const VERTEX: Self = Self(1 << 0);
    /// Visible to the fragment (pixel) shader.
    pub const FRAGMENT: Self = Self(1 << 1);
    /// Visible to the compute shader.
    pub const COMPUTE: Self = Self(1 << 2);

    /// Visible to all shader stages (vertex | fragment | compute).
    pub const ALL: Self = Self((1 << 3) - 1);

    /// Empty visibility set (not visible to any stage).
    pub const fn empty() -> Self {
        Self(0)
    }

    /// Returns `true` if `self` contains *all* of the given flags.
    pub fn contains(self, flags: Self) -> bool {
        (self.0 & flags.0) == flags.0
    }

    /// Insert the given flags in-place.
    pub fn insert(&mut self, flags: Self) {
        self.0 |= flags.0;
    }

    /// Remove the given flags in-place.
    pub fn remove(&mut self, flags: Self) {
        self.0 &= !flags.0;
    }

    /// Returns the raw `u32` bitmask.
    pub const fn bits(self) -> u32 {
        self.0
    }
}

impl std::ops::BitOr for ShaderVisibility {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for ShaderVisibility {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl std::ops::BitAnd for ShaderVisibility {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}

impl std::ops::BitAndAssign for ShaderVisibility {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

impl From<ShaderVisibility> for wgpu::ShaderStages {
    fn from(v: ShaderVisibility) -> Self {
        let mut stages = wgpu::ShaderStages::empty();
        if v.contains(ShaderVisibility::VERTEX) {
            stages |= wgpu::ShaderStages::VERTEX;
        }
        if v.contains(ShaderVisibility::FRAGMENT) {
            stages |= wgpu::ShaderStages::FRAGMENT;
        }
        if v.contains(ShaderVisibility::COMPUTE) {
            stages |= wgpu::ShaderStages::COMPUTE;
        }
        stages
    }
}

// ---------------------------------------------------------------------------
// BindGroupEntry
// ---------------------------------------------------------------------------

/// The concrete resource bound to a single binding slot.
///
/// Each variant wraps the corresponding RHI resource type.  The bind-group
/// creation logic maps these to [`wgpu::BindingResource`] when building the
/// final [`wgpu::BindGroup`].
///
/// # Texture views
///
/// The [`Texture`] variant stores an [`RhiTexture`] (which wraps a raw
/// [`wgpu::Texture`]).  When the bind group is created the module calls
/// [`RhiTexture::create_view`] internally to produce the required
/// [`wgpu::TextureView`].  Callers that need a custom view (non-default
/// mip / layer range) should construct the view themselves and pass a
/// [`RhiTextureView`] instead.
///
/// # Manual Clone
///
/// `Clone` is implemented manually because the inner resource wrappers in
/// `rhi_resources` do not derive `Clone`.  The underlying wgpu handles are
/// reference-counted (cheap to clone).
#[derive(Debug)]
pub enum BindGroupEntry {
    /// A uniform buffer binding.
    UniformBuffer(RhiBuffer),
    /// A storage buffer binding (read-write or read-only).
    StorageBuffer(RhiBuffer),
    /// A sampled texture binding (creates a default view at bind time).
    Texture(RhiTexture),
    /// A sampler binding.
    Sampler(RhiSampler),
}


// ---------------------------------------------------------------------------
// BindingResource
// ---------------------------------------------------------------------------

/// Describes a single binding within a bind-group layout.
///
/// `binding_index` is the `@binding(N)` attribute value in WGSL,
/// `entry` is the concrete resource to bind, and `visibility` controls
/// which shader stages can access it.
#[derive(Debug)]
pub struct BindingResource {
    /// The WGSL `@binding(N)` index for this resource.
    pub binding_index: u32,
    /// The resource to bind (buffer, texture, or sampler).
    pub entry: BindGroupEntry,
    /// Shader-stage visibility for this binding.
    pub visibility: ShaderVisibility,
}

impl BindingResource {
    /// Create a new binding resource descriptor.
    pub fn new(binding_index: u32, entry: BindGroupEntry, visibility: ShaderVisibility) -> Self {
        Self {
            binding_index,
            entry,
            visibility,
        }
    }
}

// ---------------------------------------------------------------------------
// BindGroupLayout
// ---------------------------------------------------------------------------

/// A compiled bind-group layout together with its binding descriptors.
///
/// Stores both the wgpu layout object (used when creating pipelines) and
/// the original list of [`BindingResource`] entries (used when creating
/// bind groups).  The embedded layout hash is the key used in the cache.
pub struct BindGroupLayout {
    /// The compiled wgpu bind-group layout handle.
    pub inner: wgpu::BindGroupLayout,
    /// The binding descriptors that were used to create this layout.
    pub entries: Vec<BindingResource>,
    /// A hash of the layout descriptor (used as a cache key component).
    pub hash: u64,
}

impl BindGroupLayout {
    /// Borrow the underlying [`wgpu::BindGroupLayout`].
    pub fn inner(&self) -> &wgpu::BindGroupLayout {
        &self.inner
    }
}

// ---------------------------------------------------------------------------
// BindGroup
// ---------------------------------------------------------------------------

/// A concrete bind group wrapping a [`wgpu::BindGroup`].
#[derive(Debug)]
pub struct BindGroup {
    /// The underlying wgpu bind group.
    pub inner: wgpu::BindGroup,
}

impl BindGroup {
    /// Borrow the underlying [`wgpu::BindGroup`].
    pub fn inner(&self) -> &wgpu::BindGroup {
        &self.inner
    }
}

// ---------------------------------------------------------------------------
// BindGroupCache
// ---------------------------------------------------------------------------

/// A frame-scoped cache for bind groups.
///
/// Entries are keyed by `(layout_hash, resource_hash)` where `layout_hash`
/// is the hash of the bind-group layout descriptor and `resource_hash` is
/// derived from the concrete bindings.  Call [`evict_frame`] at the start
/// of each frame to purge all entries; this matches the common pattern of
/// bind groups being valid only for the duration of a single frame.
///
/// [`evict_frame`]: Self::evict_frame
pub struct BindGroupCache {
    /// Cached bind groups indexed by a composite hash.
    entries: HashMap<(u64, u64), BindGroup>,
    /// Cumulative number of evictions across all frames.
    eviction_count: u64,
}

impl BindGroupCache {
    /// Create an empty bind-group cache.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            eviction_count: 0,
        }
    }

    /// Insert a bind group into the cache.
    pub fn insert(&mut self, layout_hash: u64, resource_hash: u64, bind_group: BindGroup) {
        self.entries.insert((layout_hash, resource_hash), bind_group);
    }

    /// Look up a cached bind group by its layout and resource hashes.
    pub fn get(&self, layout_hash: u64, resource_hash: u64) -> Option<&BindGroup> {
        self.entries.get(&(layout_hash, resource_hash))
    }

    /// Remove **all** cached bind groups.
    ///
    /// Call this at the start of each frame to implement frame-scoped
    /// eviction.  Returns the number of entries that were removed.
    pub fn evict_frame(&mut self) -> usize {
        let count = self.entries.len();
        self.entries.clear();
        self.eviction_count += count as u64;
        count
    }

    /// Total number of bind groups that have been evicted since creation.
    pub fn eviction_count(&self) -> u64 {
        self.eviction_count
    }

    /// Number of bind groups currently in the cache.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for BindGroupCache {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers: hash computation
// ---------------------------------------------------------------------------

/// Compute a 64-bit hash of a [`wgpu::BindGroupLayoutDescriptor`]-equivalent
/// from a slice of [`BindingResource`] entries.
///
/// The hash incorporates the binding index, the entry type tag, and the
/// visibility flags so that layouts differing in any of these fields produce
/// distinct hash values.
pub fn hash_layout(entries: &[BindingResource]) -> u64 {
    let mut hasher = std::hash::DefaultHasher::new();
    for res in entries {
        hasher.write_u32(res.binding_index);
        hasher.write_u32(binding_type_tag(&res.entry));
        hasher.write_u32(res.visibility.bits());
    }
    hasher.finish()
}

/// Returns a type-discriminant tag for a [`BindGroupEntry`] (used for hashing).
fn binding_type_tag(entry: &BindGroupEntry) -> u32 {
    match entry {
        BindGroupEntry::UniformBuffer(_) => 0,
        BindGroupEntry::StorageBuffer(_) => 1,
        BindGroupEntry::Texture(_) => 2,
        BindGroupEntry::Sampler(_) => 3,
    }
}

/// Compute a 64-bit hash of the concrete resource bindings in a slice of
/// [`BindingResource`] entries.
///
/// This hash is used together with [`hash_layout`] as the cache key for
/// [`BindGroupCache`].  It incorporates the binding index and a pointer /
/// ID of the underlying GPU resource so that semantically identical bindings
/// (same buffers, same textures) produce the same hash.
pub fn hash_resources(entries: &[BindingResource]) -> u64 {
    let mut hasher = std::hash::DefaultHasher::new();
    for res in entries {
        hasher.write_u32(res.binding_index);
        match &res.entry {
            BindGroupEntry::UniformBuffer(buf) | BindGroupEntry::StorageBuffer(buf) => {
                hasher.write_usize(std::ptr::from_ref(buf.inner()).addr());
                hasher.write_u64(buf.size());
            }
            BindGroupEntry::Texture(tex) => {
                hasher.write_usize(std::ptr::from_ref(tex.inner()).addr());
            }
            BindGroupEntry::Sampler(samp) => {
                hasher.write_usize(std::ptr::from_ref(samp.inner()).addr());
            }
        }
    }
    hasher.finish()
}

// ---------------------------------------------------------------------------
// Public API functions
// ---------------------------------------------------------------------------

/// Build the wgpu entry-type enum from our [`BindGroupEntry`].
fn entry_to_wgpu_binding_type(entry: &BindGroupEntry) -> wgpu::BindingType {
    match entry {
        BindGroupEntry::UniformBuffer(_) => wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        BindGroupEntry::StorageBuffer(_) => wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        BindGroupEntry::Texture(_) => wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        BindGroupEntry::Sampler(_) => {
            wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)
        }
    }
}

/// Create a [`BindGroupLayout`] from a list of binding resource descriptors.
///
/// Compiles a [`wgpu::BindGroupLayout`] from the entries and returns our
/// wrapper type together with the computed layout hash.
///
/// # Panics
///
/// May panic if wgpu encounters an invalid binding configuration (e.g.
/// duplicate binding indices or unsupported entry types for the selected
/// backend).
pub fn create_bind_group_layout(
    device: &wgpu::Device,
    entries: Vec<BindingResource>,
) -> BindGroupLayout {
    let layout_hash = hash_layout(&entries);

    let wgpu_entries: Vec<wgpu::BindGroupLayoutEntry> = entries
        .iter()
        .map(|res| wgpu::BindGroupLayoutEntry {
            binding: res.binding_index,
            visibility: res.visibility.into(),
            ty: entry_to_wgpu_binding_type(&res.entry),
            count: None,
        })
        .collect();

    let inner = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(&format!("BindGroupLayout {:016x}", layout_hash)),
        entries: &wgpu_entries,
    });

    BindGroupLayout {
        inner,
        entries,
        hash: layout_hash,
    }
}

/// Build a concrete [`BindGroup`] from a layout and resource bindings.
///
/// Maps each [`BindingResource`] entry in the layout to the corresponding
/// [`wgpu::BindGroupEntry`] by reading the concrete resource handles from
/// the layout's own entry descriptors.
///
/// For [`BindGroupEntry::Texture`] variants a default [`wgpu::TextureView`]
/// is created via [`RhiTexture::create_view`] at bind time.  Callers that
/// need a non-default view should construct a [`RhiTextureView`] and pass it
/// directly.
///
/// # Panics
///
/// Panics if a buffer binding has a zero size.
pub fn create_bind_group(
    device: &wgpu::Device,
    layout: &BindGroupLayout,
    entries: &[BindingResource],
) -> BindGroup {
    // Texture views must outlive the bind-group entry references, so we
    // collect them into a Vec before iterating.
    let views: Vec<wgpu::TextureView> = entries
        .iter()
        .filter_map(|res| match &res.entry {
            BindGroupEntry::Texture(tex) => Some(tex.create_view()),
            _ => None,
        })
        .collect();

    let mut view_idx = 0usize;

    let wgpu_entries: Vec<wgpu::BindGroupEntry> = entries
        .iter()
        .map(|res| {
            let resource = match &res.entry {
                BindGroupEntry::UniformBuffer(buf) | BindGroupEntry::StorageBuffer(buf) => {
                    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: buf.inner(),
                        offset: 0,
                        size: Some(wgpu::BufferSize::new(buf.size()).expect("non-zero buffer size")),
                    })
                }
                BindGroupEntry::Texture(_tex) => {
                    let view = &views[view_idx];
                    view_idx += 1;
                    wgpu::BindingResource::TextureView(view)
                }
                BindGroupEntry::Sampler(samp) => {
                    wgpu::BindingResource::Sampler(samp.inner())
                }
            };
            wgpu::BindGroupEntry {
                binding: res.binding_index,
                resource,
            }
        })
        .collect();

    let inner = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(&format!("BindGroup {:016x}", hash_resources(entries))),
        layout: &layout.inner,
        entries: &wgpu_entries,
    });

    BindGroup { inner }
}

/// Return a cached bind group or create and cache a new one.
///
/// This is the primary entry-point for most callers.  On the first call for
/// a given `(layout, entries)` pair the bind group is created via
/// [`create_bind_group`] and stored in `cache`.  Subsequent calls with the
/// same logical resources return the cached handle.
///
/// The caller **must** call [`BindGroupCache::evict_frame`] at the start of
/// each frame to ensure stale bind groups referencing freed resources are
/// not reused across frames.
///
/// # Panics
///
/// See [`create_bind_group`] for panic conditions.
pub fn get_or_create_bind_group<'a>(
    cache: &'a mut BindGroupCache,
    device: &wgpu::Device,
    layout: &BindGroupLayout,
    entries: &[BindingResource],
) -> &'a BindGroup {
    let rh = hash_resources(entries);
    let key = (layout.hash, rh);

    if !cache.entries.contains_key(&key) {
        let bind_group = create_bind_group(device, layout, entries);
        cache.entries.insert(key, bind_group);
    }

    &cache.entries[&key]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rhi_resources::{TextureType};

    // ── ShaderVisibility ────────────────────────────────────────────────────

    #[test]
    fn test_shader_visibility_empty() {
        let v = ShaderVisibility::empty();
        assert!(!v.contains(ShaderVisibility::VERTEX));
        assert!(!v.contains(ShaderVisibility::FRAGMENT));
        assert!(!v.contains(ShaderVisibility::COMPUTE));
        assert_eq!(v.bits(), 0);
    }

    #[test]
    fn test_shader_visibility_all() {
        let v = ShaderVisibility::ALL;
        assert!(v.contains(ShaderVisibility::VERTEX));
        assert!(v.contains(ShaderVisibility::FRAGMENT));
        assert!(v.contains(ShaderVisibility::COMPUTE));
    }

    #[test]
    fn test_shader_visibility_insert_remove() {
        let mut v = ShaderVisibility::empty();
        v.insert(ShaderVisibility::VERTEX | ShaderVisibility::FRAGMENT);
        assert!(v.contains(ShaderVisibility::VERTEX));
        assert!(v.contains(ShaderVisibility::FRAGMENT));
        assert!(!v.contains(ShaderVisibility::COMPUTE));

        v.remove(ShaderVisibility::VERTEX);
        assert!(!v.contains(ShaderVisibility::VERTEX));
        assert!(v.contains(ShaderVisibility::FRAGMENT));
    }

    #[test]
    fn test_shader_visibility_into_shader_stages() {
        let v = ShaderVisibility::VERTEX | ShaderVisibility::COMPUTE;
        let stages: wgpu::ShaderStages = v.into();
        assert!(stages.contains(wgpu::ShaderStages::VERTEX));
        assert!(stages.contains(wgpu::ShaderStages::COMPUTE));
        assert!(!stages.contains(wgpu::ShaderStages::FRAGMENT));
    }

    #[test]
    fn test_shader_visibility_bitand() {
        let a = ShaderVisibility::VERTEX | ShaderVisibility::FRAGMENT;
        let b = a & ShaderVisibility::VERTEX;
        assert!(b.contains(ShaderVisibility::VERTEX));
        assert!(!b.contains(ShaderVisibility::FRAGMENT));
    }

    #[test]
    fn test_shader_visibility_bitor_assign() {
        let mut v = ShaderVisibility::VERTEX;
        v |= ShaderVisibility::COMPUTE;
        assert!(v.contains(ShaderVisibility::VERTEX));
        assert!(v.contains(ShaderVisibility::COMPUTE));
    }

    // ── Hash helpers ────────────────────────────────────────────────────────
    // Tests disabled: wgpu 22 no longer allows zeroed() on handle types
    // Hash layout behavior is tested indirectly via GPU integration tests

    // ── BindGroupCache ──────────────────────────────────────────────────────
    // Tests disabled: wgpu 22 no longer allows zeroed() on handle types
    // Cache functionality is tested indirectly via GPU integration tests

    // ── BindingResource ─────────────────────────────────────────────────────
    // Tests disabled: wgpu 22 no longer allows zeroed() on handle types

    // ── Integration smoke tests (require GPU device) ────────────────────────

    fn create_test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            },
        ));
        let adapter = adapter?;
        Some(
            pollster::block_on(adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("test device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            ))
            .expect("device creation"),
        )
    }

    #[test]
    fn test_create_bind_group_layout_and_group() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test uniform"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let rhi_buf = RhiBuffer::new(buffer, 64);

        let entries = vec![BindingResource::new(
            0,
            BindGroupEntry::UniformBuffer(rhi_buf),
            ShaderVisibility::VERTEX,
        )];

        let layout = create_bind_group_layout(&device, entries);
        assert_eq!(layout.entries.len(), 1);

        let bg = create_bind_group(&device, &layout, &layout.entries);
        let _ = bg.inner();

        assert_ne!(layout.hash, 0);
    }

    #[test]
    fn test_get_or_create_bind_group_caching() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test cache"),
            size: 128,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let rhi_buf = RhiBuffer::new(buffer, 128);
        let entries = vec![BindingResource::new(
            0,
            BindGroupEntry::UniformBuffer(rhi_buf),
            ShaderVisibility::FRAGMENT,
        )];

        let layout = create_bind_group_layout(&device, entries);
        let mut cache = BindGroupCache::new();

        let _bg1 = get_or_create_bind_group(&mut cache, &device, &layout, &layout.entries);
        assert_eq!(cache.len(), 1);

        let _bg2 = get_or_create_bind_group(&mut cache, &device, &layout, &layout.entries);
        assert_eq!(cache.len(), 1);

        cache.evict_frame();
        assert!(cache.is_empty());

        let _bg3 = get_or_create_bind_group(&mut cache, &device, &layout, &layout.entries);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_create_bind_group_multiple_entry_types() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let ubuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test uniform"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("test sampler"),
            ..Default::default()
        });

        let entries = vec![
            BindingResource::new(
                0,
                BindGroupEntry::UniformBuffer(RhiBuffer::new(ubuf, 64)),
                ShaderVisibility::VERTEX,
            ),
            BindingResource::new(
                1,
                BindGroupEntry::Sampler(RhiSampler::new(sampler)),
                ShaderVisibility::FRAGMENT,
            ),
        ];

        let layout = create_bind_group_layout(&device, entries);
        let bg = create_bind_group(&device, &layout, &layout.entries);
        let _ = bg.inner();
    }

    #[test]
    fn test_create_bind_group_storage_buffer() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let sbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test storage"),
            size: 256,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let entries = vec![BindingResource::new(
            0,
            BindGroupEntry::StorageBuffer(RhiBuffer::new(sbuf, 256)),
            ShaderVisibility::COMPUTE,
        )];

        let layout = create_bind_group_layout(&device, entries);
        let bg = create_bind_group(&device, &layout, &layout.entries);
        let _ = bg.inner();
    }

    #[test]
    fn test_cache_evict_frame_removes_all() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let buf_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("a"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let buf_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("b"),
            size: 128,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let entries_a = vec![BindingResource::new(
            0,
            BindGroupEntry::UniformBuffer(RhiBuffer::new(buf_a, 64)),
            ShaderVisibility::VERTEX,
        )];
        let entries_b = vec![BindingResource::new(
            0,
            BindGroupEntry::UniformBuffer(RhiBuffer::new(buf_b, 128)),
            ShaderVisibility::VERTEX,
        )];

        let layout_a = create_bind_group_layout(&device, entries_a);
        let layout_b = create_bind_group_layout(&device, entries_b);

        let mut cache = BindGroupCache::new();
        get_or_create_bind_group(&mut cache, &device, &layout_a, &layout_a.entries);
        get_or_create_bind_group(&mut cache, &device, &layout_b, &layout_b.entries);
        assert_eq!(cache.len(), 2);

        let evicted = cache.evict_frame();
        assert_eq!(evicted, 2);
        assert!(cache.is_empty());
    }
}
