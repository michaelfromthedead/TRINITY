//! Blackbox contract tests for T-WGPU-P7.5.2 Resource Declaration.
//!
//! CLEANROOM: No src/ access beyond the public API exported by the crate.
//! Tests use only `renderer_backend::frame_graph::resources::*` -- no internal
//! fields, no private methods, no implementation details.
//!
//! Acceptance criteria:
//!   - ResourceRegistry public interface
//!   - All descriptor constructors work as documented
//!   - TextureView and BufferSlice creation
//!   - ResourceHandle versioning
//!   - Real-world resource patterns (G-buffer, shadow maps, post-process, etc.)
//!   - Resource lifecycle management
//!   - Edge cases and error handling
//!
//! Coverage: 100+ tests across API contracts, real-world patterns, lifecycle,
//! descriptor builders, edge cases, and view/slice patterns.

use renderer_backend::frame_graph::resources::{
    BufferDescriptor, BufferSlice, ResourceDescriptor, ResourceHandle, ResourceId,
    ResourceRegistry, TextureDescriptor, TextureView,
};

// =============================================================================
// SECTION 1 -- ResourceId API Contract Tests (10 tests)
// =============================================================================

#[test]
fn resource_id_none_is_sentinel() {
    let id = ResourceId::NONE;
    assert!(id.is_none());
    assert!(!id.is_some());
}

#[test]
fn resource_id_none_raw_is_max() {
    assert_eq!(ResourceId::NONE.raw(), u32::MAX);
}

#[test]
fn resource_id_new_creates_valid_id() {
    let id = ResourceId::new(42);
    assert!(id.is_some());
    assert!(!id.is_none());
    assert_eq!(id.raw(), 42);
}

#[test]
fn resource_id_new_zero_is_valid() {
    let id = ResourceId::new(0);
    assert!(id.is_some());
    assert_eq!(id.raw(), 0);
}

#[test]
fn resource_id_equality() {
    let a = ResourceId::new(10);
    let b = ResourceId::new(10);
    let c = ResourceId::new(20);
    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn resource_id_ordering() {
    let a = ResourceId::new(5);
    let b = ResourceId::new(10);
    assert!(a < b);
    assert!(b > a);
}

#[test]
fn resource_id_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(ResourceId::new(1));
    set.insert(ResourceId::new(2));
    set.insert(ResourceId::new(1));
    assert_eq!(set.len(), 2);
}

#[test]
fn resource_id_display_valid() {
    let id = ResourceId::new(5);
    assert_eq!(format!("{}", id), "ResourceId(5)");
}

#[test]
fn resource_id_display_none() {
    assert_eq!(format!("{}", ResourceId::NONE), "ResourceId::NONE");
}

#[test]
fn resource_id_default_is_none() {
    let id: ResourceId = Default::default();
    assert!(id.is_none());
}

// =============================================================================
// SECTION 2 -- TextureDescriptor API Contract Tests (15 tests)
// =============================================================================

#[test]
fn texture_descriptor_new_2d_basic() {
    let desc = TextureDescriptor::new_2d(256, 256, wgpu::TextureFormat::Rgba8Unorm);
    assert_eq!(desc.width, 256);
    assert_eq!(desc.height, 256);
    assert_eq!(desc.depth_or_layers, 1);
    assert_eq!(desc.mip_levels, 1);
    assert_eq!(desc.sample_count, 1);
    assert_eq!(desc.dimension, wgpu::TextureDimension::D2);
}

#[test]
fn texture_descriptor_new_2d_usage_flags() {
    let desc = TextureDescriptor::new_2d(128, 128, wgpu::TextureFormat::Rgba8Unorm);
    assert!(desc.usage.contains(wgpu::TextureUsages::TEXTURE_BINDING));
    assert!(desc.usage.contains(wgpu::TextureUsages::COPY_DST));
}

#[test]
fn texture_descriptor_render_target_basic() {
    let desc = TextureDescriptor::new_render_target(1920, 1080, wgpu::TextureFormat::Rgba8Unorm);
    assert_eq!(desc.width, 1920);
    assert_eq!(desc.height, 1080);
}

#[test]
fn texture_descriptor_render_target_usage_flags() {
    let desc = TextureDescriptor::new_render_target(800, 600, wgpu::TextureFormat::Bgra8Unorm);
    assert!(desc.usage.contains(wgpu::TextureUsages::RENDER_ATTACHMENT));
    assert!(desc.usage.contains(wgpu::TextureUsages::TEXTURE_BINDING));
    assert!(desc.usage.contains(wgpu::TextureUsages::COPY_SRC));
}

#[test]
fn texture_descriptor_depth_format() {
    let desc = TextureDescriptor::new_depth(1024, 768);
    assert_eq!(desc.format, wgpu::TextureFormat::Depth32Float);
}

#[test]
fn texture_descriptor_depth_usage_flags() {
    let desc = TextureDescriptor::new_depth(512, 512);
    assert!(desc.usage.contains(wgpu::TextureUsages::RENDER_ATTACHMENT));
    assert!(desc.usage.contains(wgpu::TextureUsages::TEXTURE_BINDING));
}

#[test]
fn texture_descriptor_with_mips() {
    let desc = TextureDescriptor::new_2d(1024, 1024, wgpu::TextureFormat::Rgba8Unorm)
        .with_mips(10);
    assert_eq!(desc.mip_levels, 10);
}

#[test]
fn texture_descriptor_with_msaa() {
    let desc = TextureDescriptor::new_render_target(800, 600, wgpu::TextureFormat::Rgba8Unorm)
        .with_msaa(4);
    assert_eq!(desc.sample_count, 4);
}

#[test]
fn texture_descriptor_with_label() {
    let desc = TextureDescriptor::new_2d(64, 64, wgpu::TextureFormat::Rgba8Unorm)
        .with_label("my_texture");
    assert_eq!(desc.label, Some("my_texture".to_string()));
}

#[test]
fn texture_descriptor_builder_chaining() {
    let desc = TextureDescriptor::new_render_target(512, 512, wgpu::TextureFormat::Rgba16Float)
        .with_mips(5)
        .with_msaa(2)
        .with_label("chained");
    assert_eq!(desc.mip_levels, 5);
    assert_eq!(desc.sample_count, 2);
    assert_eq!(desc.label, Some("chained".to_string()));
}

#[test]
fn texture_descriptor_default() {
    let desc: TextureDescriptor = Default::default();
    assert_eq!(desc.width, 1);
    assert_eq!(desc.height, 1);
    assert_eq!(desc.format, wgpu::TextureFormat::Rgba8Unorm);
}

#[test]
fn texture_descriptor_display() {
    let desc = TextureDescriptor::new_2d(100, 200, wgpu::TextureFormat::R8Unorm);
    let s = format!("{}", desc);
    assert!(s.contains("100"));
    assert!(s.contains("200"));
}

#[test]
fn texture_descriptor_clone() {
    let desc = TextureDescriptor::new_depth(256, 256).with_label("cloned");
    let clone = desc.clone();
    assert_eq!(desc, clone);
}

#[test]
fn texture_descriptor_size_bytes_basic() {
    // 256x256 RGBA8 = 256 * 256 * 4 = 262144
    let desc = TextureDescriptor::new_2d(256, 256, wgpu::TextureFormat::Rgba8Unorm);
    assert_eq!(desc.size_bytes(), 262144);
}

#[test]
fn texture_descriptor_size_bytes_with_msaa() {
    // 256x256 RGBA8 * 4 samples = 1048576
    let desc = TextureDescriptor::new_2d(256, 256, wgpu::TextureFormat::Rgba8Unorm)
        .with_msaa(4);
    assert_eq!(desc.size_bytes(), 1048576);
}

// =============================================================================
// SECTION 3 -- BufferDescriptor API Contract Tests (12 tests)
// =============================================================================

#[test]
fn buffer_descriptor_new_basic() {
    let desc = BufferDescriptor::new(1024, wgpu::BufferUsages::COPY_DST);
    assert_eq!(desc.size, 1024);
    assert_eq!(desc.usage, wgpu::BufferUsages::COPY_DST);
    assert!(!desc.mapped_at_creation);
}

#[test]
fn buffer_descriptor_new_vertex() {
    let desc = BufferDescriptor::new_vertex(4096);
    assert_eq!(desc.size, 4096);
    assert!(desc.usage.contains(wgpu::BufferUsages::VERTEX));
    assert!(desc.usage.contains(wgpu::BufferUsages::COPY_DST));
}

#[test]
fn buffer_descriptor_new_index() {
    let desc = BufferDescriptor::new_index(2048);
    assert_eq!(desc.size, 2048);
    assert!(desc.usage.contains(wgpu::BufferUsages::INDEX));
    assert!(desc.usage.contains(wgpu::BufferUsages::COPY_DST));
}

#[test]
fn buffer_descriptor_new_uniform() {
    let desc = BufferDescriptor::new_uniform(256);
    assert_eq!(desc.size, 256);
    assert!(desc.usage.contains(wgpu::BufferUsages::UNIFORM));
    assert!(desc.usage.contains(wgpu::BufferUsages::COPY_DST));
}

#[test]
fn buffer_descriptor_new_storage() {
    let desc = BufferDescriptor::new_storage(65536);
    assert_eq!(desc.size, 65536);
    assert!(desc.usage.contains(wgpu::BufferUsages::STORAGE));
    assert!(desc.usage.contains(wgpu::BufferUsages::COPY_DST));
    assert!(desc.usage.contains(wgpu::BufferUsages::COPY_SRC));
}

#[test]
fn buffer_descriptor_new_staging_read() {
    let desc = BufferDescriptor::new_staging_read(1024);
    assert!(desc.usage.contains(wgpu::BufferUsages::MAP_READ));
    assert!(desc.usage.contains(wgpu::BufferUsages::COPY_DST));
    assert!(!desc.mapped_at_creation);
}

#[test]
fn buffer_descriptor_new_staging_write() {
    let desc = BufferDescriptor::new_staging_write(2048);
    assert!(desc.usage.contains(wgpu::BufferUsages::MAP_WRITE));
    assert!(desc.usage.contains(wgpu::BufferUsages::COPY_SRC));
    assert!(desc.mapped_at_creation);
}

#[test]
fn buffer_descriptor_with_label() {
    let desc = BufferDescriptor::new_uniform(128).with_label("camera_uniforms");
    assert_eq!(desc.label, Some("camera_uniforms".to_string()));
}

#[test]
fn buffer_descriptor_default() {
    let desc: BufferDescriptor = Default::default();
    assert_eq!(desc.size, 256);
    assert!(desc.usage.contains(wgpu::BufferUsages::COPY_DST));
}

#[test]
fn buffer_descriptor_display() {
    let desc = BufferDescriptor::new(512, wgpu::BufferUsages::VERTEX);
    let s = format!("{}", desc);
    assert!(s.contains("512"));
}

#[test]
fn buffer_descriptor_clone() {
    let desc = BufferDescriptor::new_storage(8192).with_label("cloned");
    let clone = desc.clone();
    assert_eq!(desc, clone);
}

#[test]
fn buffer_descriptor_equality() {
    let a = BufferDescriptor::new_uniform(256);
    let b = BufferDescriptor::new_uniform(256);
    let c = BufferDescriptor::new_uniform(512);
    assert_eq!(a, b);
    assert_ne!(a, c);
}

// =============================================================================
// SECTION 4 -- ResourceDescriptor API Contract Tests (10 tests)
// =============================================================================

#[test]
fn resource_descriptor_from_texture() {
    let tex = TextureDescriptor::new_2d(128, 128, wgpu::TextureFormat::Rgba8Unorm);
    let desc: ResourceDescriptor = tex.into();
    assert!(desc.is_texture());
    assert!(!desc.is_buffer());
}

#[test]
fn resource_descriptor_from_buffer() {
    let buf = BufferDescriptor::new_uniform(256);
    let desc: ResourceDescriptor = buf.into();
    assert!(desc.is_buffer());
    assert!(!desc.is_texture());
}

#[test]
fn resource_descriptor_as_texture() {
    let tex = TextureDescriptor::new_depth(512, 512);
    let desc: ResourceDescriptor = tex.clone().into();
    assert_eq!(desc.as_texture(), Some(&tex));
    assert_eq!(desc.as_buffer(), None);
}

#[test]
fn resource_descriptor_as_buffer() {
    let buf = BufferDescriptor::new_vertex(1024);
    let desc: ResourceDescriptor = buf.clone().into();
    assert_eq!(desc.as_buffer(), Some(&buf));
    assert_eq!(desc.as_texture(), None);
}

#[test]
fn resource_descriptor_size_bytes_texture() {
    let tex = TextureDescriptor::new_2d(100, 100, wgpu::TextureFormat::Rgba8Unorm);
    let desc: ResourceDescriptor = tex.into();
    assert_eq!(desc.size_bytes(), 40000); // 100*100*4
}

#[test]
fn resource_descriptor_size_bytes_buffer() {
    let buf = BufferDescriptor::new(1024, wgpu::BufferUsages::UNIFORM);
    let desc: ResourceDescriptor = buf.into();
    assert_eq!(desc.size_bytes(), 1024);
}

#[test]
fn resource_descriptor_label_texture() {
    let tex = TextureDescriptor::new_2d(64, 64, wgpu::TextureFormat::Rgba8Unorm)
        .with_label("labeled_tex");
    let desc: ResourceDescriptor = tex.into();
    assert_eq!(desc.label(), Some("labeled_tex"));
}

#[test]
fn resource_descriptor_label_buffer() {
    let buf = BufferDescriptor::new_uniform(64).with_label("labeled_buf");
    let desc: ResourceDescriptor = buf.into();
    assert_eq!(desc.label(), Some("labeled_buf"));
}

#[test]
fn resource_descriptor_label_none() {
    let tex = TextureDescriptor::new_2d(32, 32, wgpu::TextureFormat::R8Unorm);
    let desc: ResourceDescriptor = tex.into();
    assert_eq!(desc.label(), None);
}

#[test]
fn resource_descriptor_display() {
    let tex = TextureDescriptor::new_render_target(800, 600, wgpu::TextureFormat::Bgra8Unorm);
    let desc: ResourceDescriptor = tex.into();
    let s = format!("{}", desc);
    assert!(s.contains("800"));
    assert!(s.contains("600"));
}

// =============================================================================
// SECTION 5 -- ResourceHandle API Contract Tests (8 tests)
// =============================================================================

#[test]
fn resource_handle_new_texture() {
    let id = ResourceId::new(0);
    let desc = ResourceDescriptor::Texture(TextureDescriptor::new_depth(256, 256));
    let handle = ResourceHandle::new(id, desc);
    assert_eq!(handle.id, id);
    assert_eq!(handle.generation, 0);
    assert!(handle.is_texture());
}

#[test]
fn resource_handle_new_buffer() {
    let id = ResourceId::new(1);
    let desc = ResourceDescriptor::Buffer(BufferDescriptor::new_uniform(128));
    let handle = ResourceHandle::new(id, desc);
    assert!(handle.is_buffer());
    assert!(!handle.is_texture());
}

#[test]
fn resource_handle_with_generation() {
    let id = ResourceId::new(5);
    let desc = ResourceDescriptor::Buffer(BufferDescriptor::new_storage(1024));
    let handle = ResourceHandle::with_generation(id, desc, 42);
    assert_eq!(handle.generation, 42);
}

#[test]
fn resource_handle_generation_zero_default() {
    let handle = ResourceHandle::new(
        ResourceId::new(0),
        ResourceDescriptor::Texture(TextureDescriptor::default()),
    );
    assert_eq!(handle.generation, 0);
}

#[test]
fn resource_handle_display() {
    let handle = ResourceHandle::with_generation(
        ResourceId::new(3),
        ResourceDescriptor::Buffer(BufferDescriptor::new(512, wgpu::BufferUsages::VERTEX)),
        7,
    );
    let s = format!("{}", handle);
    assert!(s.contains("3"));
    assert!(s.contains("gen7"));
}

#[test]
fn resource_handle_clone() {
    let handle = ResourceHandle::with_generation(
        ResourceId::new(10),
        ResourceDescriptor::Texture(TextureDescriptor::new_depth(128, 128)),
        5,
    );
    let clone = handle.clone();
    assert_eq!(handle.id, clone.id);
    assert_eq!(handle.generation, clone.generation);
}

#[test]
fn resource_handle_equality() {
    let desc = ResourceDescriptor::Buffer(BufferDescriptor::new_uniform(64));
    let a = ResourceHandle::with_generation(ResourceId::new(1), desc.clone(), 0);
    let b = ResourceHandle::with_generation(ResourceId::new(1), desc.clone(), 0);
    let c = ResourceHandle::with_generation(ResourceId::new(1), desc, 1);
    assert_eq!(a, b);
    assert_ne!(a, c); // Different generation
}

#[test]
fn resource_handle_is_texture_vs_buffer() {
    let tex_handle = ResourceHandle::new(
        ResourceId::new(0),
        ResourceDescriptor::Texture(TextureDescriptor::default()),
    );
    let buf_handle = ResourceHandle::new(
        ResourceId::new(1),
        ResourceDescriptor::Buffer(BufferDescriptor::default()),
    );
    assert!(tex_handle.is_texture());
    assert!(!tex_handle.is_buffer());
    assert!(!buf_handle.is_texture());
    assert!(buf_handle.is_buffer());
}

// =============================================================================
// SECTION 6 -- TextureView API Contract Tests (10 tests)
// =============================================================================

#[test]
fn texture_view_new_full() {
    let id = ResourceId::new(5);
    let view = TextureView::new(id);
    assert_eq!(view.resource, id);
    assert_eq!(view.aspect, wgpu::TextureAspect::All);
    assert_eq!(view.base_mip, 0);
    assert_eq!(view.mip_count, None);
    assert_eq!(view.base_layer, 0);
    assert_eq!(view.layer_count, None);
}

#[test]
fn texture_view_with_mip_range() {
    let view = TextureView::new(ResourceId::new(0))
        .with_mip_range(2, Some(3));
    assert_eq!(view.base_mip, 2);
    assert_eq!(view.mip_count, Some(3));
}

#[test]
fn texture_view_with_mip_range_none_count() {
    let view = TextureView::new(ResourceId::new(0))
        .with_mip_range(1, None);
    assert_eq!(view.base_mip, 1);
    assert_eq!(view.mip_count, None);
}

#[test]
fn texture_view_with_layer_range() {
    let view = TextureView::new(ResourceId::new(0))
        .with_layer_range(3, Some(6));
    assert_eq!(view.base_layer, 3);
    assert_eq!(view.layer_count, Some(6));
}

#[test]
fn texture_view_with_layer_range_none_count() {
    let view = TextureView::new(ResourceId::new(0))
        .with_layer_range(5, None);
    assert_eq!(view.base_layer, 5);
    assert_eq!(view.layer_count, None);
}

#[test]
fn texture_view_depth_only() {
    let view = TextureView::depth_only(ResourceId::new(10));
    assert_eq!(view.aspect, wgpu::TextureAspect::DepthOnly);
    assert_eq!(view.resource, ResourceId::new(10));
}

#[test]
fn texture_view_stencil_only() {
    let view = TextureView::stencil_only(ResourceId::new(20));
    assert_eq!(view.aspect, wgpu::TextureAspect::StencilOnly);
    assert_eq!(view.resource, ResourceId::new(20));
}

#[test]
fn texture_view_chained_builders() {
    let view = TextureView::new(ResourceId::new(1))
        .with_mip_range(1, Some(4))
        .with_layer_range(0, Some(2));
    assert_eq!(view.base_mip, 1);
    assert_eq!(view.mip_count, Some(4));
    assert_eq!(view.base_layer, 0);
    assert_eq!(view.layer_count, Some(2));
}

#[test]
fn texture_view_display() {
    let view = TextureView::new(ResourceId::new(7));
    let s = format!("{}", view);
    assert!(s.contains("TextureView"));
    assert!(s.contains("7"));
}

#[test]
fn texture_view_clone() {
    let view = TextureView::depth_only(ResourceId::new(3));
    let clone = view.clone();
    assert_eq!(view, clone);
}

// =============================================================================
// SECTION 7 -- BufferSlice API Contract Tests (8 tests)
// =============================================================================

#[test]
fn buffer_slice_new_full() {
    let id = ResourceId::new(15);
    let slice = BufferSlice::new(id);
    assert_eq!(slice.resource, id);
    assert_eq!(slice.offset, 0);
    assert_eq!(slice.size, None);
}

#[test]
fn buffer_slice_with_range() {
    let slice = BufferSlice::new(ResourceId::new(0))
        .with_range(64, Some(256));
    assert_eq!(slice.offset, 64);
    assert_eq!(slice.size, Some(256));
}

#[test]
fn buffer_slice_with_range_to_end() {
    let slice = BufferSlice::new(ResourceId::new(0))
        .with_range(128, None);
    assert_eq!(slice.offset, 128);
    assert_eq!(slice.size, None);
}

#[test]
fn buffer_slice_zero_offset() {
    let slice = BufferSlice::new(ResourceId::new(0))
        .with_range(0, Some(512));
    assert_eq!(slice.offset, 0);
    assert_eq!(slice.size, Some(512));
}

#[test]
fn buffer_slice_display_with_size() {
    let slice = BufferSlice::new(ResourceId::new(5))
        .with_range(100, Some(200));
    let s = format!("{}", slice);
    assert!(s.contains("100"));
    assert!(s.contains("300")); // 100 + 200
}

#[test]
fn buffer_slice_display_to_end() {
    let slice = BufferSlice::new(ResourceId::new(3))
        .with_range(50, None);
    let s = format!("{}", slice);
    assert!(s.contains("50"));
    assert!(s.contains("end"));
}

#[test]
fn buffer_slice_clone() {
    let slice = BufferSlice::new(ResourceId::new(8))
        .with_range(32, Some(128));
    let clone = slice.clone();
    assert_eq!(slice, clone);
}

#[test]
fn buffer_slice_equality() {
    let a = BufferSlice::new(ResourceId::new(1)).with_range(0, Some(100));
    let b = BufferSlice::new(ResourceId::new(1)).with_range(0, Some(100));
    let c = BufferSlice::new(ResourceId::new(1)).with_range(0, Some(200));
    assert_eq!(a, b);
    assert_ne!(a, c);
}

// =============================================================================
// SECTION 8 -- ResourceRegistry API Contract Tests (15 tests)
// =============================================================================

#[test]
fn registry_new_is_empty() {
    let registry = ResourceRegistry::new();
    assert!(registry.is_empty());
    assert_eq!(registry.count(), 0);
}

#[test]
fn registry_default_is_empty() {
    let registry: ResourceRegistry = Default::default();
    assert!(registry.is_empty());
}

#[test]
fn registry_declare_texture() {
    let mut registry = ResourceRegistry::new();
    let id = registry.declare_texture(
        "color_buffer",
        TextureDescriptor::new_render_target(1920, 1080, wgpu::TextureFormat::Rgba8Unorm),
    );
    assert_eq!(registry.count(), 1);
    assert!(!registry.is_empty());
    assert!(registry.get(id).is_some());
}

#[test]
fn registry_declare_buffer() {
    let mut registry = ResourceRegistry::new();
    let id = registry.declare_buffer(
        "uniform_buf",
        BufferDescriptor::new_uniform(256),
    );
    assert!(registry.get(id).unwrap().is_buffer());
}

#[test]
fn registry_get_by_name() {
    let mut registry = ResourceRegistry::new();
    let id = registry.declare_texture("my_tex", TextureDescriptor::default());
    assert_eq!(registry.get_by_name("my_tex"), Some(id));
}

#[test]
fn registry_get_by_name_not_found() {
    let registry = ResourceRegistry::new();
    assert_eq!(registry.get_by_name("nonexistent"), None);
}

#[test]
fn registry_get_invalid_id() {
    let registry = ResourceRegistry::new();
    assert!(registry.get(ResourceId::new(999)).is_none());
}

#[test]
fn registry_remove() {
    let mut registry = ResourceRegistry::new();
    let id = registry.declare_buffer("temp", BufferDescriptor::default());
    let removed = registry.remove(id);
    assert!(removed.is_some());
    assert!(registry.get(id).is_none());
    assert!(registry.get_by_name("temp").is_none());
}

#[test]
fn registry_remove_updates_generation() {
    let mut registry = ResourceRegistry::new();
    let id = registry.declare_buffer("temp", BufferDescriptor::default());
    let gen_before = registry.generation();
    registry.remove(id);
    assert!(registry.generation() > gen_before);
}

#[test]
fn registry_clear() {
    let mut registry = ResourceRegistry::new();
    registry.declare_texture("t1", TextureDescriptor::default());
    registry.declare_buffer("b1", BufferDescriptor::default());
    assert_eq!(registry.count(), 2);
    registry.clear();
    assert!(registry.is_empty());
}

#[test]
fn registry_clear_updates_generation() {
    let mut registry = ResourceRegistry::new();
    registry.declare_texture("t", TextureDescriptor::default());
    let gen_before = registry.generation();
    registry.clear();
    assert!(registry.generation() > gen_before);
}

#[test]
fn registry_iter() {
    let mut registry = ResourceRegistry::new();
    registry.declare_texture("t1", TextureDescriptor::default());
    registry.declare_buffer("b1", BufferDescriptor::default());
    let items: Vec<_> = registry.iter().collect();
    assert_eq!(items.len(), 2);
}

#[test]
fn registry_total_size_bytes() {
    let mut registry = ResourceRegistry::new();
    // 100*100*4 = 40000
    registry.declare_texture(
        "tex",
        TextureDescriptor::new_2d(100, 100, wgpu::TextureFormat::Rgba8Unorm),
    );
    // 1024
    registry.declare_buffer("buf", BufferDescriptor::new(1024, wgpu::BufferUsages::UNIFORM));
    assert_eq!(registry.total_size_bytes(), 41024);
}

#[test]
fn registry_unique_ids() {
    let mut registry = ResourceRegistry::new();
    let id1 = registry.declare_texture("t1", TextureDescriptor::default());
    let id2 = registry.declare_texture("t2", TextureDescriptor::default());
    let id3 = registry.declare_buffer("b1", BufferDescriptor::default());
    assert_ne!(id1, id2);
    assert_ne!(id2, id3);
    assert_ne!(id1, id3);
}

#[test]
fn registry_ids_never_reused_after_clear() {
    let mut registry = ResourceRegistry::new();
    let id1 = registry.declare_texture("t1", TextureDescriptor::default());
    registry.clear();
    let id2 = registry.declare_texture("t2", TextureDescriptor::default());
    assert_ne!(id1, id2);
    assert!(id2.raw() > id1.raw());
}

// =============================================================================
// SECTION 9 -- Real-World Resource Patterns (30 tests)
// =============================================================================

#[test]
fn pattern_render_target_color_depth() {
    let mut registry = ResourceRegistry::new();
    let _color = registry.declare_texture(
        "main_color",
        TextureDescriptor::new_render_target(1920, 1080, wgpu::TextureFormat::Rgba8Unorm),
    );
    let _depth = registry.declare_texture(
        "main_depth",
        TextureDescriptor::new_depth(1920, 1080),
    );
    assert_eq!(registry.count(), 2);
}

#[test]
fn pattern_gbuffer_position() {
    let mut registry = ResourceRegistry::new();
    let id = registry.declare_texture(
        "gbuffer_position",
        TextureDescriptor::new_render_target(1920, 1080, wgpu::TextureFormat::Rgba32Float),
    );
    let handle = registry.get(id).unwrap();
    let tex = handle.descriptor.as_texture().unwrap();
    assert_eq!(tex.format, wgpu::TextureFormat::Rgba32Float);
}

#[test]
fn pattern_gbuffer_normal() {
    let mut registry = ResourceRegistry::new();
    let id = registry.declare_texture(
        "gbuffer_normal",
        TextureDescriptor::new_render_target(1920, 1080, wgpu::TextureFormat::Rgba16Float),
    );
    let handle = registry.get(id).unwrap();
    let tex = handle.descriptor.as_texture().unwrap();
    assert_eq!(tex.format, wgpu::TextureFormat::Rgba16Float);
}

#[test]
fn pattern_gbuffer_albedo() {
    let mut registry = ResourceRegistry::new();
    let id = registry.declare_texture(
        "gbuffer_albedo",
        TextureDescriptor::new_render_target(1920, 1080, wgpu::TextureFormat::Rgba8Unorm),
    );
    let handle = registry.get(id).unwrap();
    let tex = handle.descriptor.as_texture().unwrap();
    assert_eq!(tex.format, wgpu::TextureFormat::Rgba8Unorm);
}

#[test]
fn pattern_gbuffer_depth() {
    let mut registry = ResourceRegistry::new();
    let id = registry.declare_texture(
        "gbuffer_depth",
        TextureDescriptor::new_depth(1920, 1080),
    );
    let handle = registry.get(id).unwrap();
    let tex = handle.descriptor.as_texture().unwrap();
    assert_eq!(tex.format, wgpu::TextureFormat::Depth32Float);
}

#[test]
fn pattern_full_gbuffer() {
    let mut registry = ResourceRegistry::new();
    registry.declare_texture(
        "gbuffer_position",
        TextureDescriptor::new_render_target(1920, 1080, wgpu::TextureFormat::Rgba32Float),
    );
    registry.declare_texture(
        "gbuffer_normal",
        TextureDescriptor::new_render_target(1920, 1080, wgpu::TextureFormat::Rgba16Float),
    );
    registry.declare_texture(
        "gbuffer_albedo",
        TextureDescriptor::new_render_target(1920, 1080, wgpu::TextureFormat::Rgba8Unorm),
    );
    registry.declare_texture(
        "gbuffer_material",
        TextureDescriptor::new_render_target(1920, 1080, wgpu::TextureFormat::Rgba8Unorm),
    );
    registry.declare_texture(
        "gbuffer_depth",
        TextureDescriptor::new_depth(1920, 1080),
    );
    assert_eq!(registry.count(), 5);
}

#[test]
fn pattern_shadow_map_single() {
    let mut registry = ResourceRegistry::new();
    let id = registry.declare_texture(
        "shadow_map",
        TextureDescriptor::new_depth(2048, 2048),
    );
    let handle = registry.get(id).unwrap();
    let tex = handle.descriptor.as_texture().unwrap();
    assert_eq!(tex.width, 2048);
    assert_eq!(tex.height, 2048);
}

#[test]
fn pattern_shadow_cascade_4() {
    let mut registry = ResourceRegistry::new();
    for i in 0..4 {
        registry.declare_texture(
            format!("shadow_cascade_{}", i),
            TextureDescriptor::new_depth(2048, 2048),
        );
    }
    assert_eq!(registry.count(), 4);
}

#[test]
fn pattern_post_process_ping_pong() {
    let mut registry = ResourceRegistry::new();
    let _ping = registry.declare_texture(
        "post_ping",
        TextureDescriptor::new_render_target(1920, 1080, wgpu::TextureFormat::Rgba16Float),
    );
    let _pong = registry.declare_texture(
        "post_pong",
        TextureDescriptor::new_render_target(1920, 1080, wgpu::TextureFormat::Rgba16Float),
    );
    assert_eq!(registry.count(), 2);
}

#[test]
fn pattern_bloom_chain() {
    let mut registry = ResourceRegistry::new();
    let sizes = [(960, 540), (480, 270), (240, 135), (120, 68)];
    for (i, (w, h)) in sizes.iter().enumerate() {
        registry.declare_texture(
            format!("bloom_mip_{}", i),
            TextureDescriptor::new_render_target(*w, *h, wgpu::TextureFormat::Rgba16Float),
        );
    }
    assert_eq!(registry.count(), 4);
}

#[test]
fn pattern_uniform_buffer_camera() {
    let mut registry = ResourceRegistry::new();
    // Camera matrices: 2x mat4x4 (view, projection) = 2 * 64 = 128 bytes
    // Plus position, padding = 16 + 16 = 32 bytes
    // Total ~160 bytes, aligned to 256
    let id = registry.declare_buffer(
        "camera_uniforms",
        BufferDescriptor::new_uniform(256),
    );
    let handle = registry.get(id).unwrap();
    assert!(handle.is_buffer());
}

#[test]
fn pattern_uniform_buffer_lighting() {
    let mut registry = ResourceRegistry::new();
    // Light data: direction, color, intensity, etc.
    let id = registry.declare_buffer(
        "lighting_uniforms",
        BufferDescriptor::new_uniform(512),
    );
    let handle = registry.get(id).unwrap();
    let buf = handle.descriptor.as_buffer().unwrap();
    assert_eq!(buf.size, 512);
}

#[test]
fn pattern_vertex_index_pair() {
    let mut registry = ResourceRegistry::new();
    // 10000 vertices * 32 bytes each
    let _vb = registry.declare_buffer(
        "mesh_vertices",
        BufferDescriptor::new_vertex(320_000),
    );
    // 30000 indices * 4 bytes each
    let _ib = registry.declare_buffer(
        "mesh_indices",
        BufferDescriptor::new_index(120_000),
    );
    assert_eq!(registry.count(), 2);
}

#[test]
fn pattern_storage_buffer_compute() {
    let mut registry = ResourceRegistry::new();
    // 1M particles * 32 bytes each = 32MB
    let id = registry.declare_buffer(
        "particle_data",
        BufferDescriptor::new_storage(32 * 1024 * 1024),
    );
    let handle = registry.get(id).unwrap();
    let buf = handle.descriptor.as_buffer().unwrap();
    assert!(buf.usage.contains(wgpu::BufferUsages::STORAGE));
}

#[test]
fn pattern_staging_upload() {
    let mut registry = ResourceRegistry::new();
    let id = registry.declare_buffer(
        "texture_staging",
        BufferDescriptor::new_staging_write(4 * 1024 * 1024),
    );
    let handle = registry.get(id).unwrap();
    let buf = handle.descriptor.as_buffer().unwrap();
    assert!(buf.mapped_at_creation);
}

#[test]
fn pattern_staging_readback() {
    let mut registry = ResourceRegistry::new();
    let id = registry.declare_buffer(
        "readback_staging",
        BufferDescriptor::new_staging_read(1024 * 1024),
    );
    let handle = registry.get(id).unwrap();
    let buf = handle.descriptor.as_buffer().unwrap();
    assert!(buf.usage.contains(wgpu::BufferUsages::MAP_READ));
}

#[test]
fn pattern_hdr_render_target() {
    let mut registry = ResourceRegistry::new();
    let id = registry.declare_texture(
        "hdr_color",
        TextureDescriptor::new_render_target(1920, 1080, wgpu::TextureFormat::Rgba16Float),
    );
    let handle = registry.get(id).unwrap();
    let tex = handle.descriptor.as_texture().unwrap();
    assert_eq!(tex.format, wgpu::TextureFormat::Rgba16Float);
}

#[test]
fn pattern_msaa_color_resolve() {
    let mut registry = ResourceRegistry::new();
    let _msaa = registry.declare_texture(
        "msaa_color",
        TextureDescriptor::new_render_target(1920, 1080, wgpu::TextureFormat::Rgba8Unorm)
            .with_msaa(4),
    );
    let _resolve = registry.declare_texture(
        "resolved_color",
        TextureDescriptor::new_render_target(1920, 1080, wgpu::TextureFormat::Rgba8Unorm),
    );
    let msaa_handle = registry.get_by_name("msaa_color").unwrap();
    let msaa = registry.get(msaa_handle).unwrap();
    let tex = msaa.descriptor.as_texture().unwrap();
    assert_eq!(tex.sample_count, 4);
}

#[test]
fn pattern_cubemap_environment() {
    let mut registry = ResourceRegistry::new();
    // 6 faces of cubemap as texture array
    let mut desc = TextureDescriptor::new_2d(512, 512, wgpu::TextureFormat::Rgba16Float);
    desc.depth_or_layers = 6;
    desc.usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
    let id = registry.declare_texture("env_cubemap", desc);
    let handle = registry.get(id).unwrap();
    let tex = handle.descriptor.as_texture().unwrap();
    assert_eq!(tex.depth_or_layers, 6);
}

#[test]
fn pattern_terrain_texture_array() {
    let mut registry = ResourceRegistry::new();
    // 16 terrain textures in an array
    let mut desc = TextureDescriptor::new_2d(1024, 1024, wgpu::TextureFormat::Rgba8Unorm);
    desc.depth_or_layers = 16;
    desc.mip_levels = 10;
    let id = registry.declare_texture("terrain_array", desc);
    let handle = registry.get(id).unwrap();
    let tex = handle.descriptor.as_texture().unwrap();
    assert_eq!(tex.depth_or_layers, 16);
    assert_eq!(tex.mip_levels, 10);
}

#[test]
fn pattern_indirect_args_buffer() {
    let mut registry = ResourceRegistry::new();
    // DrawIndirect: 4 u32s = 16 bytes per draw, 1000 draws
    let id = registry.declare_buffer(
        "indirect_args",
        BufferDescriptor::new(16 * 1000, wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::STORAGE),
    );
    let handle = registry.get(id).unwrap();
    let buf = handle.descriptor.as_buffer().unwrap();
    assert!(buf.usage.contains(wgpu::BufferUsages::INDIRECT));
}

#[test]
fn pattern_instance_buffer() {
    let mut registry = ResourceRegistry::new();
    // 10000 instances * 64 bytes per instance (transform + data)
    let id = registry.declare_buffer(
        "instance_data",
        BufferDescriptor::new_vertex(640_000),
    );
    let handle = registry.get(id).unwrap();
    let buf = handle.descriptor.as_buffer().unwrap();
    assert!(buf.usage.contains(wgpu::BufferUsages::VERTEX));
}

#[test]
fn pattern_ssao_output() {
    let mut registry = ResourceRegistry::new();
    let id = registry.declare_texture(
        "ssao_output",
        TextureDescriptor::new_render_target(1920, 1080, wgpu::TextureFormat::R8Unorm),
    );
    let handle = registry.get(id).unwrap();
    let tex = handle.descriptor.as_texture().unwrap();
    assert_eq!(tex.format, wgpu::TextureFormat::R8Unorm);
}

#[test]
fn pattern_velocity_buffer() {
    let mut registry = ResourceRegistry::new();
    let id = registry.declare_texture(
        "velocity",
        TextureDescriptor::new_render_target(1920, 1080, wgpu::TextureFormat::Rg16Float),
    );
    let handle = registry.get(id).unwrap();
    let tex = handle.descriptor.as_texture().unwrap();
    assert_eq!(tex.format, wgpu::TextureFormat::Rg16Float);
}

#[test]
fn pattern_depth_prepass() {
    let mut registry = ResourceRegistry::new();
    let id = registry.declare_texture(
        "depth_prepass",
        TextureDescriptor::new_depth(1920, 1080),
    );
    let view = TextureView::depth_only(id);
    assert_eq!(view.aspect, wgpu::TextureAspect::DepthOnly);
}

#[test]
fn pattern_lut_3d_color_grading() {
    let mut registry = ResourceRegistry::new();
    // 64x64x64 3D LUT for color grading
    let mut desc = TextureDescriptor::new_2d(64, 64, wgpu::TextureFormat::Rgba16Float);
    desc.depth_or_layers = 64;
    desc.dimension = wgpu::TextureDimension::D3;
    let id = registry.declare_texture("color_lut", desc);
    let handle = registry.get(id).unwrap();
    let tex = handle.descriptor.as_texture().unwrap();
    assert_eq!(tex.dimension, wgpu::TextureDimension::D3);
}

#[test]
fn pattern_compute_reduction() {
    let mut registry = ResourceRegistry::new();
    // Histogram reduction: 256 bins * 4 bytes
    let id = registry.declare_buffer(
        "histogram",
        BufferDescriptor::new_storage(1024),
    );
    let handle = registry.get(id).unwrap();
    let buf = handle.descriptor.as_buffer().unwrap();
    assert!(buf.usage.contains(wgpu::BufferUsages::STORAGE));
}

#[test]
fn pattern_depth_pyramid() {
    let mut registry = ResourceRegistry::new();
    // Hierarchical Z buffer for occlusion culling
    let id = registry.declare_texture(
        "depth_pyramid",
        TextureDescriptor::new_2d(2048, 2048, wgpu::TextureFormat::R32Float)
            .with_mips(12),
    );
    let handle = registry.get(id).unwrap();
    let tex = handle.descriptor.as_texture().unwrap();
    assert_eq!(tex.mip_levels, 12);
}

#[test]
fn pattern_visibility_buffer() {
    let mut registry = ResourceRegistry::new();
    // Visibility buffer: instance ID + primitive ID
    let id = registry.declare_texture(
        "visibility",
        TextureDescriptor::new_render_target(1920, 1080, wgpu::TextureFormat::Rg32Uint),
    );
    let handle = registry.get(id).unwrap();
    let tex = handle.descriptor.as_texture().unwrap();
    assert_eq!(tex.format, wgpu::TextureFormat::Rg32Uint);
}

// =============================================================================
// SECTION 10 -- Resource Lifecycle Tests (15 tests)
// =============================================================================

#[test]
fn lifecycle_create_use_remove() {
    let mut registry = ResourceRegistry::new();
    let id = registry.declare_texture("temp", TextureDescriptor::default());
    assert!(registry.get(id).is_some());
    let removed = registry.remove(id);
    assert!(removed.is_some());
    assert!(registry.get(id).is_none());
}

#[test]
fn lifecycle_multiple_resources_same_type() {
    let mut registry = ResourceRegistry::new();
    let mut ids = Vec::new();
    for i in 0..10 {
        ids.push(registry.declare_texture(
            format!("tex_{}", i),
            TextureDescriptor::default(),
        ));
    }
    assert_eq!(registry.count(), 10);
    for id in &ids {
        assert!(registry.get(*id).is_some());
    }
}

#[test]
fn lifecycle_mixed_resource_types() {
    let mut registry = ResourceRegistry::new();
    let t1 = registry.declare_texture("tex1", TextureDescriptor::default());
    let b1 = registry.declare_buffer("buf1", BufferDescriptor::default());
    let t2 = registry.declare_texture("tex2", TextureDescriptor::default());
    let b2 = registry.declare_buffer("buf2", BufferDescriptor::default());

    assert!(registry.get(t1).unwrap().is_texture());
    assert!(registry.get(b1).unwrap().is_buffer());
    assert!(registry.get(t2).unwrap().is_texture());
    assert!(registry.get(b2).unwrap().is_buffer());
}

#[test]
fn lifecycle_generation_increases_on_remove() {
    let mut registry = ResourceRegistry::new();
    let id = registry.declare_buffer("temp", BufferDescriptor::default());
    let gen1 = registry.generation();
    registry.remove(id);
    let gen2 = registry.generation();
    assert!(gen2 > gen1);
}

#[test]
fn lifecycle_generation_increases_on_clear() {
    let mut registry = ResourceRegistry::new();
    registry.declare_texture("t", TextureDescriptor::default());
    let gen1 = registry.generation();
    registry.clear();
    let gen2 = registry.generation();
    assert!(gen2 > gen1);
}

#[test]
fn lifecycle_clear_and_reuse() {
    let mut registry = ResourceRegistry::new();
    registry.declare_texture("t1", TextureDescriptor::default());
    registry.declare_buffer("b1", BufferDescriptor::default());
    registry.clear();
    assert!(registry.is_empty());

    // Can declare new resources after clear
    let id = registry.declare_texture("t2", TextureDescriptor::default());
    assert!(registry.get(id).is_some());
}

#[test]
fn lifecycle_remove_preserves_other_resources() {
    let mut registry = ResourceRegistry::new();
    let id1 = registry.declare_texture("t1", TextureDescriptor::default());
    let id2 = registry.declare_texture("t2", TextureDescriptor::default());
    let id3 = registry.declare_buffer("b1", BufferDescriptor::default());

    registry.remove(id2);

    assert!(registry.get(id1).is_some());
    assert!(registry.get(id2).is_none());
    assert!(registry.get(id3).is_some());
    assert_eq!(registry.count(), 2);
}

#[test]
fn lifecycle_remove_clears_name_mapping() {
    let mut registry = ResourceRegistry::new();
    let id = registry.declare_texture("named_resource", TextureDescriptor::default());
    assert_eq!(registry.get_by_name("named_resource"), Some(id));
    registry.remove(id);
    assert_eq!(registry.get_by_name("named_resource"), None);
}

#[test]
fn lifecycle_handle_generation_matches_registry() {
    let mut registry = ResourceRegistry::new();
    let id = registry.declare_texture("t", TextureDescriptor::default());
    let handle = registry.get(id).unwrap();
    assert_eq!(handle.generation, registry.generation());
}

#[test]
fn lifecycle_sequential_declare_remove() {
    let mut registry = ResourceRegistry::new();
    for i in 0..5 {
        let id = registry.declare_texture(format!("t{}", i), TextureDescriptor::default());
        assert!(registry.get(id).is_some());
        registry.remove(id);
        assert!(registry.get(id).is_none());
    }
    assert!(registry.is_empty());
}

#[test]
fn lifecycle_iter_after_modifications() {
    let mut registry = ResourceRegistry::new();
    registry.declare_texture("t1", TextureDescriptor::default());
    registry.declare_buffer("b1", BufferDescriptor::default());
    let id = registry.declare_texture("t2", TextureDescriptor::default());
    registry.remove(id);

    let items: Vec<_> = registry.iter().collect();
    assert_eq!(items.len(), 2);
}

#[test]
fn lifecycle_total_size_after_remove() {
    let mut registry = ResourceRegistry::new();
    registry.declare_texture(
        "tex",
        TextureDescriptor::new_2d(100, 100, wgpu::TextureFormat::Rgba8Unorm),
    );
    let buf_id = registry.declare_buffer("buf", BufferDescriptor::new(1024, wgpu::BufferUsages::UNIFORM));

    let size_with_both = registry.total_size_bytes();
    registry.remove(buf_id);
    let size_after_remove = registry.total_size_bytes();

    assert_eq!(size_with_both, 41024);
    assert_eq!(size_after_remove, 40000);
}

#[test]
fn lifecycle_id_monotonic_increase() {
    let mut registry = ResourceRegistry::new();
    let mut last_raw = 0u32;
    for i in 0..10 {
        let id = registry.declare_buffer(format!("b{}", i), BufferDescriptor::default());
        if i > 0 {
            assert!(id.raw() > last_raw);
        }
        last_raw = id.raw();
    }
}

#[test]
fn lifecycle_id_monotonic_after_clear() {
    let mut registry = ResourceRegistry::new();
    let id1 = registry.declare_texture("t1", TextureDescriptor::default());
    registry.clear();
    let id2 = registry.declare_texture("t2", TextureDescriptor::default());
    assert!(id2.raw() > id1.raw());
}

#[test]
fn lifecycle_id_monotonic_after_remove() {
    let mut registry = ResourceRegistry::new();
    let id1 = registry.declare_texture("t1", TextureDescriptor::default());
    registry.remove(id1);
    let id2 = registry.declare_texture("t2", TextureDescriptor::default());
    assert!(id2.raw() > id1.raw());
}

// =============================================================================
// SECTION 11 -- Descriptor Builder Tests (10 tests)
// =============================================================================

#[test]
fn builder_texture_multiple_chained_calls() {
    let desc = TextureDescriptor::new_2d(512, 512, wgpu::TextureFormat::Rgba8Unorm)
        .with_mips(8)
        .with_msaa(4)
        .with_label("chained_texture");

    assert_eq!(desc.width, 512);
    assert_eq!(desc.height, 512);
    assert_eq!(desc.mip_levels, 8);
    assert_eq!(desc.sample_count, 4);
    assert_eq!(desc.label, Some("chained_texture".to_string()));
}

#[test]
fn builder_texture_order_independent() {
    let desc1 = TextureDescriptor::new_2d(256, 256, wgpu::TextureFormat::Rgba8Unorm)
        .with_label("test")
        .with_mips(4)
        .with_msaa(2);

    let desc2 = TextureDescriptor::new_2d(256, 256, wgpu::TextureFormat::Rgba8Unorm)
        .with_msaa(2)
        .with_mips(4)
        .with_label("test");

    assert_eq!(desc1.mip_levels, desc2.mip_levels);
    assert_eq!(desc1.sample_count, desc2.sample_count);
    assert_eq!(desc1.label, desc2.label);
}

#[test]
fn builder_texture_label_string_slice() {
    let desc = TextureDescriptor::new_2d(64, 64, wgpu::TextureFormat::R8Unorm)
        .with_label("static_str");
    assert_eq!(desc.label, Some("static_str".to_string()));
}

#[test]
fn builder_texture_label_owned_string() {
    let label = String::from("owned_string");
    let desc = TextureDescriptor::new_2d(64, 64, wgpu::TextureFormat::R8Unorm)
        .with_label(label);
    assert_eq!(desc.label, Some("owned_string".to_string()));
}

#[test]
fn builder_buffer_with_label() {
    let desc = BufferDescriptor::new_uniform(128)
        .with_label("uniform_buffer");
    assert_eq!(desc.label, Some("uniform_buffer".to_string()));
}

#[test]
fn builder_texture_view_chained() {
    let view = TextureView::new(ResourceId::new(0))
        .with_mip_range(2, Some(4))
        .with_layer_range(1, Some(3));

    assert_eq!(view.base_mip, 2);
    assert_eq!(view.mip_count, Some(4));
    assert_eq!(view.base_layer, 1);
    assert_eq!(view.layer_count, Some(3));
}

#[test]
fn builder_buffer_slice_chained() {
    let slice = BufferSlice::new(ResourceId::new(5))
        .with_range(256, Some(512));

    assert_eq!(slice.resource, ResourceId::new(5));
    assert_eq!(slice.offset, 256);
    assert_eq!(slice.size, Some(512));
}

#[test]
fn builder_msaa_sample_counts() {
    for samples in [1, 2, 4, 8] {
        let desc = TextureDescriptor::new_render_target(800, 600, wgpu::TextureFormat::Rgba8Unorm)
            .with_msaa(samples);
        assert_eq!(desc.sample_count, samples);
    }
}

#[test]
fn builder_mip_levels_calculation() {
    // For 1024x1024, max mips = log2(1024) + 1 = 11
    let desc = TextureDescriptor::new_2d(1024, 1024, wgpu::TextureFormat::Rgba8Unorm)
        .with_mips(11);
    assert_eq!(desc.mip_levels, 11);
}

#[test]
fn builder_depth_no_mips_default() {
    let desc = TextureDescriptor::new_depth(512, 512);
    assert_eq!(desc.mip_levels, 1);
}

// =============================================================================
// SECTION 12 -- Edge Case Tests (15 tests)
// =============================================================================

#[test]
fn edge_case_1x1_texture() {
    let desc = TextureDescriptor::new_2d(1, 1, wgpu::TextureFormat::Rgba8Unorm);
    assert_eq!(desc.width, 1);
    assert_eq!(desc.height, 1);
    assert_eq!(desc.size_bytes(), 4);
}

#[test]
fn edge_case_maximum_size_texture() {
    // 16384x16384 is typical max texture size
    let desc = TextureDescriptor::new_2d(16384, 16384, wgpu::TextureFormat::Rgba8Unorm);
    assert_eq!(desc.width, 16384);
    assert_eq!(desc.height, 16384);
    // 16384 * 16384 * 4 = 1073741824 (1GB)
    assert_eq!(desc.size_bytes(), 1073741824);
}

#[test]
fn edge_case_empty_registry_operations() {
    let registry = ResourceRegistry::new();
    assert!(registry.is_empty());
    assert_eq!(registry.count(), 0);
    assert_eq!(registry.total_size_bytes(), 0);
    assert!(registry.get(ResourceId::new(0)).is_none());
    assert!(registry.get_by_name("anything").is_none());
    assert_eq!(registry.iter().count(), 0);
}

#[test]
fn edge_case_get_none_id() {
    let registry = ResourceRegistry::new();
    assert!(registry.get(ResourceId::NONE).is_none());
}

#[test]
fn edge_case_get_nonexistent_id() {
    let mut registry = ResourceRegistry::new();
    registry.declare_texture("t", TextureDescriptor::default());
    assert!(registry.get(ResourceId::new(999)).is_none());
}

#[test]
fn edge_case_remove_nonexistent() {
    let mut registry = ResourceRegistry::new();
    let removed = registry.remove(ResourceId::new(100));
    assert!(removed.is_none());
}

#[test]
fn edge_case_remove_twice() {
    let mut registry = ResourceRegistry::new();
    let id = registry.declare_buffer("b", BufferDescriptor::default());
    let first = registry.remove(id);
    let second = registry.remove(id);
    assert!(first.is_some());
    assert!(second.is_none());
}

#[test]
fn edge_case_buffer_size_zero() {
    let desc = BufferDescriptor::new(0, wgpu::BufferUsages::UNIFORM);
    assert_eq!(desc.size, 0);
}

#[test]
fn edge_case_buffer_max_size() {
    let desc = BufferDescriptor::new(u64::MAX, wgpu::BufferUsages::STORAGE);
    assert_eq!(desc.size, u64::MAX);
}

#[test]
fn edge_case_texture_view_zero_ranges() {
    let view = TextureView::new(ResourceId::new(0))
        .with_mip_range(0, Some(0))
        .with_layer_range(0, Some(0));
    assert_eq!(view.base_mip, 0);
    assert_eq!(view.mip_count, Some(0));
    assert_eq!(view.base_layer, 0);
    assert_eq!(view.layer_count, Some(0));
}

#[test]
fn edge_case_buffer_slice_zero_size() {
    let slice = BufferSlice::new(ResourceId::new(0))
        .with_range(0, Some(0));
    assert_eq!(slice.offset, 0);
    assert_eq!(slice.size, Some(0));
}

#[test]
fn edge_case_buffer_slice_large_offset() {
    let slice = BufferSlice::new(ResourceId::new(0))
        .with_range(u64::MAX, None);
    assert_eq!(slice.offset, u64::MAX);
}

#[test]
fn edge_case_empty_label() {
    let desc = TextureDescriptor::new_2d(64, 64, wgpu::TextureFormat::R8Unorm)
        .with_label("");
    assert_eq!(desc.label, Some("".to_string()));
}

#[test]
fn edge_case_unicode_label() {
    let desc = TextureDescriptor::new_2d(64, 64, wgpu::TextureFormat::R8Unorm)
        .with_label("texture_alpha_diffuse_normal_spec");
    assert!(desc.label.as_ref().unwrap().contains("alpha"));
}

#[test]
fn edge_case_single_mip_level() {
    let desc = TextureDescriptor::new_2d(1024, 1024, wgpu::TextureFormat::Rgba8Unorm)
        .with_mips(1);
    assert_eq!(desc.mip_levels, 1);
}

// =============================================================================
// SECTION 13 -- View/Slice Pattern Tests (10 tests)
// =============================================================================

#[test]
fn view_full_texture_defaults() {
    let view = TextureView::new(ResourceId::new(10));
    assert_eq!(view.aspect, wgpu::TextureAspect::All);
    assert_eq!(view.base_mip, 0);
    assert_eq!(view.mip_count, None);
    assert_eq!(view.base_layer, 0);
    assert_eq!(view.layer_count, None);
}

#[test]
fn view_mip_level_selection() {
    let view = TextureView::new(ResourceId::new(0))
        .with_mip_range(3, Some(1)); // Single mip at level 3
    assert_eq!(view.base_mip, 3);
    assert_eq!(view.mip_count, Some(1));
}

#[test]
fn view_array_layer_selection() {
    let view = TextureView::new(ResourceId::new(0))
        .with_layer_range(2, Some(4)); // Layers 2-5
    assert_eq!(view.base_layer, 2);
    assert_eq!(view.layer_count, Some(4));
}

#[test]
fn view_depth_only_for_depth_stencil() {
    let view = TextureView::depth_only(ResourceId::new(0));
    assert_eq!(view.aspect, wgpu::TextureAspect::DepthOnly);
}

#[test]
fn view_stencil_only_for_depth_stencil() {
    let view = TextureView::stencil_only(ResourceId::new(0));
    assert_eq!(view.aspect, wgpu::TextureAspect::StencilOnly);
}

#[test]
fn slice_full_buffer() {
    let slice = BufferSlice::new(ResourceId::new(5));
    assert_eq!(slice.offset, 0);
    assert_eq!(slice.size, None);
}

#[test]
fn slice_partial_buffer_beginning() {
    let slice = BufferSlice::new(ResourceId::new(0))
        .with_range(0, Some(256));
    assert_eq!(slice.offset, 0);
    assert_eq!(slice.size, Some(256));
}

#[test]
fn slice_partial_buffer_middle() {
    let slice = BufferSlice::new(ResourceId::new(0))
        .with_range(512, Some(256));
    assert_eq!(slice.offset, 512);
    assert_eq!(slice.size, Some(256));
}

#[test]
fn slice_buffer_to_end() {
    let slice = BufferSlice::new(ResourceId::new(0))
        .with_range(1024, None);
    assert_eq!(slice.offset, 1024);
    assert_eq!(slice.size, None);
}

#[test]
fn slice_buffer_alignment() {
    // Typical uniform buffer alignment is 256 bytes
    let slice = BufferSlice::new(ResourceId::new(0))
        .with_range(256, Some(256));
    assert_eq!(slice.offset, 256);
    assert_eq!(slice.size, Some(256));
    // Verify offset is aligned to 256
    assert_eq!(slice.offset % 256, 0);
}

// =============================================================================
// SECTION 14 -- Format Size Calculation Tests (10 tests)
// =============================================================================

#[test]
fn format_size_r8() {
    let desc = TextureDescriptor::new_2d(100, 100, wgpu::TextureFormat::R8Unorm);
    // 100 * 100 * 1 = 10000
    assert_eq!(desc.size_bytes(), 10000);
}

#[test]
fn format_size_rg8() {
    let desc = TextureDescriptor::new_2d(100, 100, wgpu::TextureFormat::Rg8Unorm);
    // 100 * 100 * 2 = 20000
    assert_eq!(desc.size_bytes(), 20000);
}

#[test]
fn format_size_rgba8() {
    let desc = TextureDescriptor::new_2d(100, 100, wgpu::TextureFormat::Rgba8Unorm);
    // 100 * 100 * 4 = 40000
    assert_eq!(desc.size_bytes(), 40000);
}

#[test]
fn format_size_rgba16f() {
    let desc = TextureDescriptor::new_2d(100, 100, wgpu::TextureFormat::Rgba16Float);
    // 100 * 100 * 8 = 80000
    assert_eq!(desc.size_bytes(), 80000);
}

#[test]
fn format_size_rgba32f() {
    let desc = TextureDescriptor::new_2d(100, 100, wgpu::TextureFormat::Rgba32Float);
    // 100 * 100 * 16 = 160000
    assert_eq!(desc.size_bytes(), 160000);
}

#[test]
fn format_size_depth32f() {
    let desc = TextureDescriptor::new_depth(100, 100);
    // 100 * 100 * 4 = 40000
    assert_eq!(desc.size_bytes(), 40000);
}

#[test]
fn format_size_depth16() {
    let desc = TextureDescriptor::new_2d(100, 100, wgpu::TextureFormat::Depth16Unorm);
    // 100 * 100 * 2 = 20000
    assert_eq!(desc.size_bytes(), 20000);
}

#[test]
fn format_size_r32f() {
    let desc = TextureDescriptor::new_2d(100, 100, wgpu::TextureFormat::R32Float);
    // 100 * 100 * 4 = 40000
    assert_eq!(desc.size_bytes(), 40000);
}

#[test]
fn format_size_r32uint() {
    let desc = TextureDescriptor::new_2d(100, 100, wgpu::TextureFormat::R32Uint);
    // 100 * 100 * 4 = 40000
    assert_eq!(desc.size_bytes(), 40000);
}

#[test]
fn format_size_bgra8() {
    let desc = TextureDescriptor::new_2d(100, 100, wgpu::TextureFormat::Bgra8Unorm);
    // 100 * 100 * 4 = 40000
    assert_eq!(desc.size_bytes(), 40000);
}

// =============================================================================
// SECTION 15 -- Duplicate Name Panic Tests (2 tests)
// =============================================================================

#[test]
#[should_panic(expected = "already exists")]
fn registry_duplicate_texture_name_panics() {
    let mut registry = ResourceRegistry::new();
    registry.declare_texture("dup", TextureDescriptor::default());
    registry.declare_texture("dup", TextureDescriptor::default());
}

#[test]
#[should_panic(expected = "already exists")]
fn registry_duplicate_buffer_name_panics() {
    let mut registry = ResourceRegistry::new();
    registry.declare_buffer("dup", BufferDescriptor::default());
    registry.declare_buffer("dup", BufferDescriptor::default());
}

// =============================================================================
// SECTION 16 -- Additional Coverage Tests (5 tests)
// =============================================================================

#[test]
fn registry_generation_starts_at_zero() {
    let registry = ResourceRegistry::new();
    assert_eq!(registry.generation(), 0);
}

#[test]
fn registry_generation_unchanged_on_declare() {
    let mut registry = ResourceRegistry::new();
    let gen_before = registry.generation();
    registry.declare_texture("t", TextureDescriptor::default());
    // Generation doesn't change on declare, only on mutations (remove/clear)
    // But the handle gets the current generation
    assert_eq!(registry.generation(), gen_before);
}

#[test]
fn handle_preserves_descriptor() {
    let tex_desc = TextureDescriptor::new_render_target(800, 600, wgpu::TextureFormat::Rgba16Float)
        .with_mips(5)
        .with_label("preserved");

    let mut registry = ResourceRegistry::new();
    let id = registry.declare_texture("test", tex_desc.clone());
    let handle = registry.get(id).unwrap();

    let retrieved = handle.descriptor.as_texture().unwrap();
    assert_eq!(retrieved.width, 800);
    assert_eq!(retrieved.height, 600);
    assert_eq!(retrieved.mip_levels, 5);
    assert_eq!(retrieved.label, Some("preserved".to_string()));
}

#[test]
fn resource_descriptor_enum_variants() {
    let tex = ResourceDescriptor::Texture(TextureDescriptor::default());
    let buf = ResourceDescriptor::Buffer(BufferDescriptor::default());

    assert!(matches!(tex, ResourceDescriptor::Texture(_)));
    assert!(matches!(buf, ResourceDescriptor::Buffer(_)));
}

#[test]
fn complete_workflow_simulation() {
    let mut registry = ResourceRegistry::new();

    // Frame 1: Declare resources
    let color = registry.declare_texture(
        "color",
        TextureDescriptor::new_render_target(1920, 1080, wgpu::TextureFormat::Rgba8Unorm),
    );
    let depth = registry.declare_texture(
        "depth",
        TextureDescriptor::new_depth(1920, 1080),
    );
    let camera = registry.declare_buffer(
        "camera",
        BufferDescriptor::new_uniform(256),
    );

    // Use resources
    assert_eq!(registry.count(), 3);
    let color_view = TextureView::new(color);
    let depth_view = TextureView::depth_only(depth);
    let camera_slice = BufferSlice::new(camera);

    assert_eq!(color_view.resource, color);
    assert_eq!(depth_view.resource, depth);
    assert_eq!(camera_slice.resource, camera);

    // Clear for next frame
    let gen = registry.generation();
    registry.clear();
    assert!(registry.is_empty());
    assert!(registry.generation() > gen);
}
