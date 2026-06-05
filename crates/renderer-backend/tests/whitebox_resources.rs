//! Whitebox tests for frame graph resource declaration system.
//!
//! Tests internal implementation details of:
//! - ResourceId: Sentinel values, construction, predicates
//! - TextureDescriptor: Constructors, builders, size calculation
//! - BufferDescriptor: Constructors, usage flags, builder pattern
//! - ResourceDescriptor: Variant discrimination, delegation
//! - ResourceHandle: Versioned tracking, generation
//! - TextureView: View parameters, aspect selection
//! - BufferSlice: Range specification
//! - ResourceRegistry: CRUD operations, generation tracking

use renderer_backend::frame_graph::resources::*;
use std::collections::HashSet;
use wgpu::{BufferUsages, TextureAspect, TextureDimension, TextureFormat, TextureUsages};

// ===========================================================================
// ResourceId Tests (15+)
// ===========================================================================

mod resource_id_tests {
    use super::*;

    #[test]
    fn test_none_sentinel_value() {
        let none = ResourceId::NONE;
        assert_eq!(none.raw(), u32::MAX);
    }

    #[test]
    fn test_none_is_none_predicate() {
        assert!(ResourceId::NONE.is_none());
    }

    #[test]
    fn test_none_is_not_some() {
        assert!(!ResourceId::NONE.is_some());
    }

    #[test]
    fn test_new_creates_valid_id() {
        let id = ResourceId::new(0);
        assert!(id.is_some());
        assert!(!id.is_none());
    }

    #[test]
    fn test_new_with_various_values() {
        for val in [0, 1, 100, 1000, u32::MAX - 1] {
            let id = ResourceId::new(val);
            assert_eq!(id.raw(), val);
        }
    }

    #[test]
    fn test_raw_returns_inner_value() {
        let id = ResourceId::new(42);
        assert_eq!(id.raw(), 42);
    }

    #[test]
    fn test_raw_for_zero() {
        let id = ResourceId::new(0);
        assert_eq!(id.raw(), 0);
    }

    #[test]
    fn test_is_some_for_zero() {
        // Zero is a valid ID, not NONE
        let id = ResourceId::new(0);
        assert!(id.is_some());
    }

    #[test]
    fn test_is_none_only_for_max() {
        let id = ResourceId::new(u32::MAX);
        assert!(id.is_none());
    }

    #[test]
    fn test_equality() {
        let id1 = ResourceId::new(5);
        let id2 = ResourceId::new(5);
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_inequality() {
        let id1 = ResourceId::new(5);
        let id2 = ResourceId::new(6);
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_hash_consistency() {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let id1 = ResourceId::new(42);
        let id2 = ResourceId::new(42);

        let mut h1 = DefaultHasher::new();
        let mut h2 = DefaultHasher::new();
        id1.hash(&mut h1);
        id2.hash(&mut h2);

        assert_eq!(h1.finish(), h2.finish());
    }

    #[test]
    fn test_hash_uniqueness() {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let id1 = ResourceId::new(1);
        let id2 = ResourceId::new(2);

        let mut h1 = DefaultHasher::new();
        let mut h2 = DefaultHasher::new();
        id1.hash(&mut h1);
        id2.hash(&mut h2);

        assert_ne!(h1.finish(), h2.finish());
    }

    #[test]
    fn test_ordering() {
        let id1 = ResourceId::new(1);
        let id2 = ResourceId::new(2);
        assert!(id1 < id2);
    }

    #[test]
    fn test_default_is_none() {
        let id: ResourceId = Default::default();
        assert!(id.is_none());
    }

    #[test]
    fn test_display_none() {
        let s = format!("{}", ResourceId::NONE);
        assert_eq!(s, "ResourceId::NONE");
    }

    #[test]
    fn test_display_valid() {
        let s = format!("{}", ResourceId::new(123));
        assert_eq!(s, "ResourceId(123)");
    }

    #[test]
    fn test_clone() {
        let id = ResourceId::new(99);
        let cloned = id.clone();
        assert_eq!(id, cloned);
    }

    #[test]
    fn test_copy() {
        let id = ResourceId::new(88);
        let copied = id;
        assert_eq!(id, copied);
    }
}

// ===========================================================================
// TextureDescriptor Tests (30+)
// ===========================================================================

mod texture_descriptor_tests {
    use super::*;

    #[test]
    fn test_new_2d_constructor() {
        let desc = TextureDescriptor::new_2d(1920, 1080, TextureFormat::Rgba8Unorm);
        assert_eq!(desc.width, 1920);
        assert_eq!(desc.height, 1080);
        assert_eq!(desc.format, TextureFormat::Rgba8Unorm);
    }

    #[test]
    fn test_new_2d_depth_is_one() {
        let desc = TextureDescriptor::new_2d(100, 100, TextureFormat::Rgba8Unorm);
        assert_eq!(desc.depth_or_layers, 1);
    }

    #[test]
    fn test_new_2d_single_mip() {
        let desc = TextureDescriptor::new_2d(100, 100, TextureFormat::Rgba8Unorm);
        assert_eq!(desc.mip_levels, 1);
    }

    #[test]
    fn test_new_2d_single_sample() {
        let desc = TextureDescriptor::new_2d(100, 100, TextureFormat::Rgba8Unorm);
        assert_eq!(desc.sample_count, 1);
    }

    #[test]
    fn test_new_2d_dimension() {
        let desc = TextureDescriptor::new_2d(100, 100, TextureFormat::Rgba8Unorm);
        assert_eq!(desc.dimension, TextureDimension::D2);
    }

    #[test]
    fn test_new_2d_default_usage() {
        let desc = TextureDescriptor::new_2d(100, 100, TextureFormat::Rgba8Unorm);
        assert!(desc.usage.contains(TextureUsages::TEXTURE_BINDING));
        assert!(desc.usage.contains(TextureUsages::COPY_DST));
    }

    #[test]
    fn test_new_2d_no_label() {
        let desc = TextureDescriptor::new_2d(100, 100, TextureFormat::Rgba8Unorm);
        assert!(desc.label.is_none());
    }

    #[test]
    fn test_new_render_target_usage() {
        let desc = TextureDescriptor::new_render_target(800, 600, TextureFormat::Rgba8Unorm);
        assert!(desc.usage.contains(TextureUsages::RENDER_ATTACHMENT));
        assert!(desc.usage.contains(TextureUsages::TEXTURE_BINDING));
        assert!(desc.usage.contains(TextureUsages::COPY_SRC));
    }

    #[test]
    fn test_new_render_target_dimensions() {
        let desc = TextureDescriptor::new_render_target(1920, 1080, TextureFormat::Bgra8Unorm);
        assert_eq!(desc.width, 1920);
        assert_eq!(desc.height, 1080);
    }

    #[test]
    fn test_new_render_target_format() {
        let desc = TextureDescriptor::new_render_target(100, 100, TextureFormat::Bgra8UnormSrgb);
        assert_eq!(desc.format, TextureFormat::Bgra8UnormSrgb);
    }

    #[test]
    fn test_new_depth_format() {
        let desc = TextureDescriptor::new_depth(1024, 768);
        assert_eq!(desc.format, TextureFormat::Depth32Float);
    }

    #[test]
    fn test_new_depth_usage() {
        let desc = TextureDescriptor::new_depth(1024, 768);
        assert!(desc.usage.contains(TextureUsages::RENDER_ATTACHMENT));
        assert!(desc.usage.contains(TextureUsages::TEXTURE_BINDING));
    }

    #[test]
    fn test_new_depth_dimensions() {
        let desc = TextureDescriptor::new_depth(512, 512);
        assert_eq!(desc.width, 512);
        assert_eq!(desc.height, 512);
    }

    #[test]
    fn test_with_mips_builder() {
        let desc = TextureDescriptor::new_2d(512, 512, TextureFormat::Rgba8Unorm).with_mips(5);
        assert_eq!(desc.mip_levels, 5);
    }

    #[test]
    fn test_with_mips_preserves_other_fields() {
        let desc = TextureDescriptor::new_2d(512, 512, TextureFormat::Rgba8Unorm).with_mips(5);
        assert_eq!(desc.width, 512);
        assert_eq!(desc.format, TextureFormat::Rgba8Unorm);
    }

    #[test]
    fn test_with_msaa_builder() {
        let desc = TextureDescriptor::new_2d(512, 512, TextureFormat::Rgba8Unorm).with_msaa(4);
        assert_eq!(desc.sample_count, 4);
    }

    #[test]
    fn test_with_msaa_various_counts() {
        for samples in [1, 2, 4, 8, 16] {
            let desc =
                TextureDescriptor::new_2d(100, 100, TextureFormat::Rgba8Unorm).with_msaa(samples);
            assert_eq!(desc.sample_count, samples);
        }
    }

    #[test]
    fn test_with_label_builder() {
        let desc = TextureDescriptor::new_2d(100, 100, TextureFormat::Rgba8Unorm)
            .with_label("test_texture");
        assert_eq!(desc.label, Some("test_texture".to_string()));
    }

    #[test]
    fn test_with_label_from_string() {
        let desc = TextureDescriptor::new_2d(100, 100, TextureFormat::Rgba8Unorm)
            .with_label(String::from("owned_label"));
        assert_eq!(desc.label, Some("owned_label".to_string()));
    }

    #[test]
    fn test_chained_builders() {
        let desc = TextureDescriptor::new_2d(256, 256, TextureFormat::Rgba8Unorm)
            .with_mips(4)
            .with_msaa(4)
            .with_label("chained");
        assert_eq!(desc.mip_levels, 4);
        assert_eq!(desc.sample_count, 4);
        assert_eq!(desc.label, Some("chained".to_string()));
    }

    #[test]
    fn test_size_bytes_rgba8() {
        // 256 * 256 * 4 bytes = 262144
        let desc = TextureDescriptor::new_2d(256, 256, TextureFormat::Rgba8Unorm);
        assert_eq!(desc.size_bytes(), 262144);
    }

    #[test]
    fn test_size_bytes_r8() {
        // 256 * 256 * 1 byte = 65536
        let desc = TextureDescriptor::new_2d(256, 256, TextureFormat::R8Unorm);
        assert_eq!(desc.size_bytes(), 65536);
    }

    #[test]
    fn test_size_bytes_rg16() {
        // 256 * 256 * 2 bytes = 131072
        let desc = TextureDescriptor::new_2d(256, 256, TextureFormat::R16Float);
        assert_eq!(desc.size_bytes(), 131072);
    }

    #[test]
    fn test_size_bytes_rgba16f() {
        // 256 * 256 * 8 bytes = 524288
        let desc = TextureDescriptor::new_2d(256, 256, TextureFormat::Rgba16Float);
        assert_eq!(desc.size_bytes(), 524288);
    }

    #[test]
    fn test_size_bytes_rgba32f() {
        // 256 * 256 * 16 bytes = 1048576
        let desc = TextureDescriptor::new_2d(256, 256, TextureFormat::Rgba32Float);
        assert_eq!(desc.size_bytes(), 1048576);
    }

    #[test]
    fn test_size_bytes_depth32() {
        // 256 * 256 * 4 bytes = 262144
        let desc = TextureDescriptor::new_depth(256, 256);
        assert_eq!(desc.size_bytes(), 262144);
    }

    #[test]
    fn test_size_bytes_with_msaa() {
        // 256 * 256 * 4 * 4 samples = 1048576
        let desc = TextureDescriptor::new_2d(256, 256, TextureFormat::Rgba8Unorm).with_msaa(4);
        assert_eq!(desc.size_bytes(), 1048576);
    }

    #[test]
    fn test_size_bytes_with_mips() {
        // Base: 256*256*4 = 262144
        // Mip1: 128*128*4 = 65536
        // Mip2: 64*64*4   = 16384
        // Mip3: 32*32*4   = 4096
        // Total = 348160
        let desc = TextureDescriptor::new_2d(256, 256, TextureFormat::Rgba8Unorm).with_mips(4);
        assert_eq!(desc.size_bytes(), 348160);
    }

    #[test]
    fn test_size_bytes_1x1_texture() {
        // 1 * 1 * 4 = 4
        let desc = TextureDescriptor::new_2d(1, 1, TextureFormat::Rgba8Unorm);
        assert_eq!(desc.size_bytes(), 4);
    }

    #[test]
    fn test_size_bytes_non_square() {
        // 1920 * 1080 * 4 = 8294400
        let desc = TextureDescriptor::new_2d(1920, 1080, TextureFormat::Rgba8Unorm);
        assert_eq!(desc.size_bytes(), 8294400);
    }

    #[test]
    fn test_default() {
        let desc = TextureDescriptor::default();
        assert_eq!(desc.width, 1);
        assert_eq!(desc.height, 1);
        assert_eq!(desc.format, TextureFormat::Rgba8Unorm);
    }

    #[test]
    fn test_display() {
        let desc = TextureDescriptor::new_2d(100, 100, TextureFormat::Rgba8Unorm);
        let s = format!("{}", desc);
        assert!(s.contains("100x100"));
        assert!(s.contains("Rgba8Unorm"));
    }

    #[test]
    fn test_clone() {
        let desc = TextureDescriptor::new_2d(256, 256, TextureFormat::Rgba8Unorm).with_label("original");
        let cloned = desc.clone();
        assert_eq!(desc, cloned);
    }

    #[test]
    fn test_partial_eq() {
        let desc1 = TextureDescriptor::new_2d(256, 256, TextureFormat::Rgba8Unorm);
        let desc2 = TextureDescriptor::new_2d(256, 256, TextureFormat::Rgba8Unorm);
        assert_eq!(desc1, desc2);
    }

    #[test]
    fn test_partial_eq_different_size() {
        let desc1 = TextureDescriptor::new_2d(256, 256, TextureFormat::Rgba8Unorm);
        let desc2 = TextureDescriptor::new_2d(512, 512, TextureFormat::Rgba8Unorm);
        assert_ne!(desc1, desc2);
    }

    #[test]
    fn test_partial_eq_different_format() {
        let desc1 = TextureDescriptor::new_2d(256, 256, TextureFormat::Rgba8Unorm);
        let desc2 = TextureDescriptor::new_2d(256, 256, TextureFormat::Bgra8Unorm);
        assert_ne!(desc1, desc2);
    }
}

// ===========================================================================
// BufferDescriptor Tests (25+)
// ===========================================================================

mod buffer_descriptor_tests {
    use super::*;

    #[test]
    fn test_new_constructor() {
        let desc = BufferDescriptor::new(1024, BufferUsages::COPY_DST);
        assert_eq!(desc.size, 1024);
        assert_eq!(desc.usage, BufferUsages::COPY_DST);
    }

    #[test]
    fn test_new_not_mapped() {
        let desc = BufferDescriptor::new(1024, BufferUsages::COPY_DST);
        assert!(!desc.mapped_at_creation);
    }

    #[test]
    fn test_new_no_label() {
        let desc = BufferDescriptor::new(1024, BufferUsages::COPY_DST);
        assert!(desc.label.is_none());
    }

    #[test]
    fn test_new_vertex_usage() {
        let desc = BufferDescriptor::new_vertex(4096);
        assert!(desc.usage.contains(BufferUsages::VERTEX));
        assert!(desc.usage.contains(BufferUsages::COPY_DST));
    }

    #[test]
    fn test_new_vertex_size() {
        let desc = BufferDescriptor::new_vertex(4096);
        assert_eq!(desc.size, 4096);
    }

    #[test]
    fn test_new_index_usage() {
        let desc = BufferDescriptor::new_index(2048);
        assert!(desc.usage.contains(BufferUsages::INDEX));
        assert!(desc.usage.contains(BufferUsages::COPY_DST));
    }

    #[test]
    fn test_new_index_size() {
        let desc = BufferDescriptor::new_index(2048);
        assert_eq!(desc.size, 2048);
    }

    #[test]
    fn test_new_uniform_usage() {
        let desc = BufferDescriptor::new_uniform(256);
        assert!(desc.usage.contains(BufferUsages::UNIFORM));
        assert!(desc.usage.contains(BufferUsages::COPY_DST));
    }

    #[test]
    fn test_new_uniform_size() {
        let desc = BufferDescriptor::new_uniform(256);
        assert_eq!(desc.size, 256);
    }

    #[test]
    fn test_new_storage_usage() {
        let desc = BufferDescriptor::new_storage(65536);
        assert!(desc.usage.contains(BufferUsages::STORAGE));
        assert!(desc.usage.contains(BufferUsages::COPY_DST));
        assert!(desc.usage.contains(BufferUsages::COPY_SRC));
    }

    #[test]
    fn test_new_storage_size() {
        let desc = BufferDescriptor::new_storage(65536);
        assert_eq!(desc.size, 65536);
    }

    #[test]
    fn test_new_staging_read_usage() {
        let desc = BufferDescriptor::new_staging_read(1024);
        assert!(desc.usage.contains(BufferUsages::MAP_READ));
        assert!(desc.usage.contains(BufferUsages::COPY_DST));
    }

    #[test]
    fn test_new_staging_read_not_mapped() {
        let desc = BufferDescriptor::new_staging_read(1024);
        assert!(!desc.mapped_at_creation);
    }

    #[test]
    fn test_new_staging_write_usage() {
        let desc = BufferDescriptor::new_staging_write(1024);
        assert!(desc.usage.contains(BufferUsages::MAP_WRITE));
        assert!(desc.usage.contains(BufferUsages::COPY_SRC));
    }

    #[test]
    fn test_new_staging_write_mapped() {
        let desc = BufferDescriptor::new_staging_write(1024);
        assert!(desc.mapped_at_creation);
    }

    #[test]
    fn test_with_label_builder() {
        let desc = BufferDescriptor::new_uniform(128).with_label("camera_uniforms");
        assert_eq!(desc.label, Some("camera_uniforms".to_string()));
    }

    #[test]
    fn test_with_label_from_string() {
        let desc = BufferDescriptor::new_uniform(128).with_label(String::from("owned_label"));
        assert_eq!(desc.label, Some("owned_label".to_string()));
    }

    #[test]
    fn test_with_label_preserves_other_fields() {
        let desc = BufferDescriptor::new_uniform(512).with_label("test");
        assert_eq!(desc.size, 512);
        assert!(desc.usage.contains(BufferUsages::UNIFORM));
    }

    #[test]
    fn test_default() {
        let desc = BufferDescriptor::default();
        assert_eq!(desc.size, 256);
        assert_eq!(desc.usage, BufferUsages::COPY_DST);
    }

    #[test]
    fn test_display() {
        let desc = BufferDescriptor::new(1024, BufferUsages::VERTEX);
        let s = format!("{}", desc);
        assert!(s.contains("1024"));
        assert!(s.contains("bytes"));
    }

    #[test]
    fn test_clone() {
        let desc = BufferDescriptor::new_storage(4096).with_label("storage");
        let cloned = desc.clone();
        assert_eq!(desc, cloned);
    }

    #[test]
    fn test_eq() {
        let desc1 = BufferDescriptor::new_uniform(256);
        let desc2 = BufferDescriptor::new_uniform(256);
        assert_eq!(desc1, desc2);
    }

    #[test]
    fn test_ne_size() {
        let desc1 = BufferDescriptor::new_uniform(256);
        let desc2 = BufferDescriptor::new_uniform(512);
        assert_ne!(desc1, desc2);
    }

    #[test]
    fn test_ne_usage() {
        let desc1 = BufferDescriptor::new_uniform(256);
        let desc2 = BufferDescriptor::new_storage(256);
        assert_ne!(desc1, desc2);
    }

    #[test]
    fn test_zero_size() {
        let desc = BufferDescriptor::new(0, BufferUsages::COPY_DST);
        assert_eq!(desc.size, 0);
    }

    #[test]
    fn test_large_size() {
        let large_size = 1024 * 1024 * 1024; // 1 GB
        let desc = BufferDescriptor::new(large_size, BufferUsages::STORAGE);
        assert_eq!(desc.size, large_size);
    }
}

// ===========================================================================
// ResourceDescriptor Tests (20+)
// ===========================================================================

mod resource_descriptor_tests {
    use super::*;

    #[test]
    fn test_texture_variant_from() {
        let tex_desc = TextureDescriptor::new_2d(256, 256, TextureFormat::Rgba8Unorm);
        let desc: ResourceDescriptor = tex_desc.into();
        assert!(matches!(desc, ResourceDescriptor::Texture(_)));
    }

    #[test]
    fn test_buffer_variant_from() {
        let buf_desc = BufferDescriptor::new_uniform(256);
        let desc: ResourceDescriptor = buf_desc.into();
        assert!(matches!(desc, ResourceDescriptor::Buffer(_)));
    }

    #[test]
    fn test_is_texture_true() {
        let tex_desc = TextureDescriptor::new_2d(256, 256, TextureFormat::Rgba8Unorm);
        let desc = ResourceDescriptor::Texture(tex_desc);
        assert!(desc.is_texture());
    }

    #[test]
    fn test_is_texture_false() {
        let buf_desc = BufferDescriptor::new_uniform(256);
        let desc = ResourceDescriptor::Buffer(buf_desc);
        assert!(!desc.is_texture());
    }

    #[test]
    fn test_is_buffer_true() {
        let buf_desc = BufferDescriptor::new_uniform(256);
        let desc = ResourceDescriptor::Buffer(buf_desc);
        assert!(desc.is_buffer());
    }

    #[test]
    fn test_is_buffer_false() {
        let tex_desc = TextureDescriptor::new_2d(256, 256, TextureFormat::Rgba8Unorm);
        let desc = ResourceDescriptor::Texture(tex_desc);
        assert!(!desc.is_buffer());
    }

    #[test]
    fn test_size_bytes_texture() {
        let tex_desc = TextureDescriptor::new_2d(100, 100, TextureFormat::Rgba8Unorm);
        let desc = ResourceDescriptor::Texture(tex_desc);
        assert_eq!(desc.size_bytes(), 40000);
    }

    #[test]
    fn test_size_bytes_buffer() {
        let buf_desc = BufferDescriptor::new(512, BufferUsages::UNIFORM);
        let desc = ResourceDescriptor::Buffer(buf_desc);
        assert_eq!(desc.size_bytes(), 512);
    }

    #[test]
    fn test_label_texture_with_label() {
        let tex_desc =
            TextureDescriptor::new_2d(64, 64, TextureFormat::Rgba8Unorm).with_label("my_texture");
        let desc = ResourceDescriptor::Texture(tex_desc);
        assert_eq!(desc.label(), Some("my_texture"));
    }

    #[test]
    fn test_label_texture_without_label() {
        let tex_desc = TextureDescriptor::new_2d(64, 64, TextureFormat::Rgba8Unorm);
        let desc = ResourceDescriptor::Texture(tex_desc);
        assert_eq!(desc.label(), None);
    }

    #[test]
    fn test_label_buffer_with_label() {
        let buf_desc = BufferDescriptor::new_uniform(64).with_label("my_buffer");
        let desc = ResourceDescriptor::Buffer(buf_desc);
        assert_eq!(desc.label(), Some("my_buffer"));
    }

    #[test]
    fn test_label_buffer_without_label() {
        let buf_desc = BufferDescriptor::new_uniform(64);
        let desc = ResourceDescriptor::Buffer(buf_desc);
        assert_eq!(desc.label(), None);
    }

    #[test]
    fn test_as_texture_some() {
        let tex_desc = TextureDescriptor::new_2d(256, 256, TextureFormat::Rgba8Unorm);
        let desc = ResourceDescriptor::Texture(tex_desc.clone());
        assert_eq!(desc.as_texture(), Some(&tex_desc));
    }

    #[test]
    fn test_as_texture_none() {
        let buf_desc = BufferDescriptor::new_uniform(256);
        let desc = ResourceDescriptor::Buffer(buf_desc);
        assert_eq!(desc.as_texture(), None);
    }

    #[test]
    fn test_as_buffer_some() {
        let buf_desc = BufferDescriptor::new_uniform(256);
        let desc = ResourceDescriptor::Buffer(buf_desc.clone());
        assert_eq!(desc.as_buffer(), Some(&buf_desc));
    }

    #[test]
    fn test_as_buffer_none() {
        let tex_desc = TextureDescriptor::new_2d(256, 256, TextureFormat::Rgba8Unorm);
        let desc = ResourceDescriptor::Texture(tex_desc);
        assert_eq!(desc.as_buffer(), None);
    }

    #[test]
    fn test_display_texture() {
        let tex_desc = TextureDescriptor::new_2d(256, 256, TextureFormat::Rgba8Unorm);
        let desc = ResourceDescriptor::Texture(tex_desc);
        let s = format!("{}", desc);
        assert!(s.contains("256x256"));
    }

    #[test]
    fn test_display_buffer() {
        let buf_desc = BufferDescriptor::new(1024, BufferUsages::STORAGE);
        let desc = ResourceDescriptor::Buffer(buf_desc);
        let s = format!("{}", desc);
        assert!(s.contains("1024"));
    }

    #[test]
    fn test_clone() {
        let tex_desc = TextureDescriptor::new_2d(256, 256, TextureFormat::Rgba8Unorm);
        let desc = ResourceDescriptor::Texture(tex_desc);
        let cloned = desc.clone();
        assert_eq!(desc, cloned);
    }

    #[test]
    fn test_eq() {
        let desc1 = ResourceDescriptor::Texture(TextureDescriptor::new_2d(
            256,
            256,
            TextureFormat::Rgba8Unorm,
        ));
        let desc2 = ResourceDescriptor::Texture(TextureDescriptor::new_2d(
            256,
            256,
            TextureFormat::Rgba8Unorm,
        ));
        assert_eq!(desc1, desc2);
    }

    #[test]
    fn test_ne_different_variants() {
        let desc1 = ResourceDescriptor::Texture(TextureDescriptor::new_2d(
            256,
            256,
            TextureFormat::Rgba8Unorm,
        ));
        let desc2 = ResourceDescriptor::Buffer(BufferDescriptor::new_uniform(256));
        assert_ne!(desc1, desc2);
    }
}

// ===========================================================================
// ResourceHandle Tests (15+)
// ===========================================================================

mod resource_handle_tests {
    use super::*;

    #[test]
    fn test_new_construction() {
        let id = ResourceId::new(0);
        let desc = ResourceDescriptor::Texture(TextureDescriptor::new_depth(800, 600));
        let handle = ResourceHandle::new(id, desc.clone());
        assert_eq!(handle.id, id);
        assert_eq!(handle.descriptor, desc);
    }

    #[test]
    fn test_new_default_generation() {
        let id = ResourceId::new(0);
        let desc = ResourceDescriptor::Texture(TextureDescriptor::new_depth(800, 600));
        let handle = ResourceHandle::new(id, desc);
        assert_eq!(handle.generation, 0);
    }

    #[test]
    fn test_with_generation() {
        let id = ResourceId::new(5);
        let desc = ResourceDescriptor::Buffer(BufferDescriptor::new_uniform(128));
        let handle = ResourceHandle::with_generation(id, desc, 42);
        assert_eq!(handle.generation, 42);
    }

    #[test]
    fn test_with_generation_preserves_id() {
        let id = ResourceId::new(5);
        let desc = ResourceDescriptor::Buffer(BufferDescriptor::new_uniform(128));
        let handle = ResourceHandle::with_generation(id, desc, 42);
        assert_eq!(handle.id, id);
    }

    #[test]
    fn test_with_generation_preserves_descriptor() {
        let id = ResourceId::new(5);
        let desc = ResourceDescriptor::Buffer(BufferDescriptor::new_uniform(128));
        let handle = ResourceHandle::with_generation(id, desc.clone(), 42);
        assert_eq!(handle.descriptor, desc);
    }

    #[test]
    fn test_is_texture_true() {
        let id = ResourceId::new(0);
        let desc = ResourceDescriptor::Texture(TextureDescriptor::new_depth(800, 600));
        let handle = ResourceHandle::new(id, desc);
        assert!(handle.is_texture());
    }

    #[test]
    fn test_is_texture_false() {
        let id = ResourceId::new(0);
        let desc = ResourceDescriptor::Buffer(BufferDescriptor::new_uniform(128));
        let handle = ResourceHandle::new(id, desc);
        assert!(!handle.is_texture());
    }

    #[test]
    fn test_is_buffer_true() {
        let id = ResourceId::new(0);
        let desc = ResourceDescriptor::Buffer(BufferDescriptor::new_uniform(128));
        let handle = ResourceHandle::new(id, desc);
        assert!(handle.is_buffer());
    }

    #[test]
    fn test_is_buffer_false() {
        let id = ResourceId::new(0);
        let desc = ResourceDescriptor::Texture(TextureDescriptor::new_depth(800, 600));
        let handle = ResourceHandle::new(id, desc);
        assert!(!handle.is_buffer());
    }

    #[test]
    fn test_generation_tracking() {
        let id = ResourceId::new(1);
        let desc = ResourceDescriptor::Texture(TextureDescriptor::new_depth(100, 100));
        let handle1 = ResourceHandle::with_generation(id, desc.clone(), 0);
        let handle2 = ResourceHandle::with_generation(id, desc, 1);
        assert_ne!(handle1.generation, handle2.generation);
    }

    #[test]
    fn test_display() {
        let id = ResourceId::new(5);
        let desc = ResourceDescriptor::Texture(TextureDescriptor::new_2d(
            256,
            256,
            TextureFormat::Rgba8Unorm,
        ));
        let handle = ResourceHandle::with_generation(id, desc, 3);
        let s = format!("{}", handle);
        assert!(s.contains("gen3"));
        assert!(s.contains("ResourceId(5)"));
    }

    #[test]
    fn test_clone() {
        let id = ResourceId::new(5);
        let desc = ResourceDescriptor::Buffer(BufferDescriptor::new_uniform(128));
        let handle = ResourceHandle::with_generation(id, desc, 42);
        let cloned = handle.clone();
        assert_eq!(handle, cloned);
    }

    #[test]
    fn test_eq() {
        let id = ResourceId::new(1);
        let desc = ResourceDescriptor::Buffer(BufferDescriptor::new_uniform(128));
        let handle1 = ResourceHandle::with_generation(id, desc.clone(), 0);
        let handle2 = ResourceHandle::with_generation(id, desc, 0);
        assert_eq!(handle1, handle2);
    }

    #[test]
    fn test_ne_different_generation() {
        let id = ResourceId::new(1);
        let desc = ResourceDescriptor::Buffer(BufferDescriptor::new_uniform(128));
        let handle1 = ResourceHandle::with_generation(id, desc.clone(), 0);
        let handle2 = ResourceHandle::with_generation(id, desc, 1);
        assert_ne!(handle1, handle2);
    }

    #[test]
    fn test_ne_different_id() {
        let desc = ResourceDescriptor::Buffer(BufferDescriptor::new_uniform(128));
        let handle1 = ResourceHandle::with_generation(ResourceId::new(1), desc.clone(), 0);
        let handle2 = ResourceHandle::with_generation(ResourceId::new(2), desc, 0);
        assert_ne!(handle1, handle2);
    }
}

// ===========================================================================
// TextureView Tests (20+)
// ===========================================================================

mod texture_view_tests {
    use super::*;

    #[test]
    fn test_new_default_view() {
        let id = ResourceId::new(1);
        let view = TextureView::new(id);
        assert_eq!(view.resource, id);
    }

    #[test]
    fn test_new_default_aspect() {
        let view = TextureView::new(ResourceId::new(1));
        assert_eq!(view.aspect, TextureAspect::All);
    }

    #[test]
    fn test_new_default_mip_base() {
        let view = TextureView::new(ResourceId::new(1));
        assert_eq!(view.base_mip, 0);
    }

    #[test]
    fn test_new_default_mip_count() {
        let view = TextureView::new(ResourceId::new(1));
        assert_eq!(view.mip_count, None);
    }

    #[test]
    fn test_new_default_layer_base() {
        let view = TextureView::new(ResourceId::new(1));
        assert_eq!(view.base_layer, 0);
    }

    #[test]
    fn test_new_default_layer_count() {
        let view = TextureView::new(ResourceId::new(1));
        assert_eq!(view.layer_count, None);
    }

    #[test]
    fn test_with_mip_range_base() {
        let view = TextureView::new(ResourceId::new(0)).with_mip_range(2, Some(3));
        assert_eq!(view.base_mip, 2);
    }

    #[test]
    fn test_with_mip_range_count() {
        let view = TextureView::new(ResourceId::new(0)).with_mip_range(2, Some(3));
        assert_eq!(view.mip_count, Some(3));
    }

    #[test]
    fn test_with_mip_range_none_count() {
        let view = TextureView::new(ResourceId::new(0)).with_mip_range(1, None);
        assert_eq!(view.base_mip, 1);
        assert_eq!(view.mip_count, None);
    }

    #[test]
    fn test_with_layer_range_base() {
        let view = TextureView::new(ResourceId::new(0)).with_layer_range(5, Some(10));
        assert_eq!(view.base_layer, 5);
    }

    #[test]
    fn test_with_layer_range_count() {
        let view = TextureView::new(ResourceId::new(0)).with_layer_range(5, Some(10));
        assert_eq!(view.layer_count, Some(10));
    }

    #[test]
    fn test_with_layer_range_none_count() {
        let view = TextureView::new(ResourceId::new(0)).with_layer_range(3, None);
        assert_eq!(view.base_layer, 3);
        assert_eq!(view.layer_count, None);
    }

    #[test]
    fn test_depth_only_aspect() {
        let view = TextureView::depth_only(ResourceId::new(0));
        assert_eq!(view.aspect, TextureAspect::DepthOnly);
    }

    #[test]
    fn test_depth_only_default_mips() {
        let view = TextureView::depth_only(ResourceId::new(0));
        assert_eq!(view.base_mip, 0);
        assert_eq!(view.mip_count, None);
    }

    #[test]
    fn test_depth_only_default_layers() {
        let view = TextureView::depth_only(ResourceId::new(0));
        assert_eq!(view.base_layer, 0);
        assert_eq!(view.layer_count, None);
    }

    #[test]
    fn test_stencil_only_aspect() {
        let view = TextureView::stencil_only(ResourceId::new(0));
        assert_eq!(view.aspect, TextureAspect::StencilOnly);
    }

    #[test]
    fn test_stencil_only_default_mips() {
        let view = TextureView::stencil_only(ResourceId::new(0));
        assert_eq!(view.base_mip, 0);
        assert_eq!(view.mip_count, None);
    }

    #[test]
    fn test_stencil_only_default_layers() {
        let view = TextureView::stencil_only(ResourceId::new(0));
        assert_eq!(view.base_layer, 0);
        assert_eq!(view.layer_count, None);
    }

    #[test]
    fn test_chained_builders() {
        let view = TextureView::new(ResourceId::new(1))
            .with_mip_range(1, Some(4))
            .with_layer_range(2, Some(6));
        assert_eq!(view.base_mip, 1);
        assert_eq!(view.mip_count, Some(4));
        assert_eq!(view.base_layer, 2);
        assert_eq!(view.layer_count, Some(6));
    }

    #[test]
    fn test_display() {
        let view = TextureView::new(ResourceId::new(5)).with_mip_range(1, Some(3));
        let s = format!("{}", view);
        assert!(s.contains("TextureView"));
        assert!(s.contains("ResourceId(5)"));
    }

    #[test]
    fn test_clone() {
        let view = TextureView::new(ResourceId::new(5)).with_mip_range(1, Some(3));
        let cloned = view.clone();
        assert_eq!(view, cloned);
    }

    #[test]
    fn test_eq() {
        let view1 = TextureView::new(ResourceId::new(5)).with_mip_range(1, Some(3));
        let view2 = TextureView::new(ResourceId::new(5)).with_mip_range(1, Some(3));
        assert_eq!(view1, view2);
    }

    #[test]
    fn test_ne_different_resource() {
        let view1 = TextureView::new(ResourceId::new(5));
        let view2 = TextureView::new(ResourceId::new(6));
        assert_ne!(view1, view2);
    }
}

// ===========================================================================
// BufferSlice Tests (15+)
// ===========================================================================

mod buffer_slice_tests {
    use super::*;

    #[test]
    fn test_new_full_buffer() {
        let id = ResourceId::new(3);
        let slice = BufferSlice::new(id);
        assert_eq!(slice.resource, id);
    }

    #[test]
    fn test_new_default_offset() {
        let slice = BufferSlice::new(ResourceId::new(3));
        assert_eq!(slice.offset, 0);
    }

    #[test]
    fn test_new_default_size() {
        let slice = BufferSlice::new(ResourceId::new(3));
        assert_eq!(slice.size, None);
    }

    #[test]
    fn test_with_range_offset() {
        let slice = BufferSlice::new(ResourceId::new(0)).with_range(64, Some(256));
        assert_eq!(slice.offset, 64);
    }

    #[test]
    fn test_with_range_size() {
        let slice = BufferSlice::new(ResourceId::new(0)).with_range(64, Some(256));
        assert_eq!(slice.size, Some(256));
    }

    #[test]
    fn test_with_range_none_size() {
        let slice = BufferSlice::new(ResourceId::new(0)).with_range(128, None);
        assert_eq!(slice.offset, 128);
        assert_eq!(slice.size, None);
    }

    #[test]
    fn test_offset_handling_zero() {
        let slice = BufferSlice::new(ResourceId::new(0)).with_range(0, Some(100));
        assert_eq!(slice.offset, 0);
    }

    #[test]
    fn test_offset_handling_large() {
        let large_offset = 1024 * 1024 * 100; // 100 MB
        let slice = BufferSlice::new(ResourceId::new(0)).with_range(large_offset, Some(256));
        assert_eq!(slice.offset, large_offset);
    }

    #[test]
    fn test_size_some() {
        let slice = BufferSlice::new(ResourceId::new(0)).with_range(0, Some(1024));
        assert_eq!(slice.size, Some(1024));
    }

    #[test]
    fn test_size_none_means_to_end() {
        let slice = BufferSlice::new(ResourceId::new(0)).with_range(512, None);
        assert_eq!(slice.size, None);
    }

    #[test]
    fn test_display_with_size() {
        let slice = BufferSlice::new(ResourceId::new(5)).with_range(100, Some(200));
        let s = format!("{}", slice);
        assert!(s.contains("BufferSlice"));
        assert!(s.contains("100"));
        assert!(s.contains("300")); // offset + size
    }

    #[test]
    fn test_display_without_size() {
        let slice = BufferSlice::new(ResourceId::new(5)).with_range(100, None);
        let s = format!("{}", slice);
        assert!(s.contains("100..end"));
    }

    #[test]
    fn test_clone() {
        let slice = BufferSlice::new(ResourceId::new(5)).with_range(64, Some(128));
        let cloned = slice.clone();
        assert_eq!(slice, cloned);
    }

    #[test]
    fn test_eq() {
        let slice1 = BufferSlice::new(ResourceId::new(5)).with_range(64, Some(128));
        let slice2 = BufferSlice::new(ResourceId::new(5)).with_range(64, Some(128));
        assert_eq!(slice1, slice2);
    }

    #[test]
    fn test_ne_different_resource() {
        let slice1 = BufferSlice::new(ResourceId::new(5));
        let slice2 = BufferSlice::new(ResourceId::new(6));
        assert_ne!(slice1, slice2);
    }

    #[test]
    fn test_ne_different_offset() {
        let slice1 = BufferSlice::new(ResourceId::new(5)).with_range(0, Some(100));
        let slice2 = BufferSlice::new(ResourceId::new(5)).with_range(50, Some(100));
        assert_ne!(slice1, slice2);
    }
}

// ===========================================================================
// ResourceRegistry Tests (30+)
// ===========================================================================

mod resource_registry_tests {
    use super::*;

    #[test]
    fn test_new_initialization() {
        let registry = ResourceRegistry::new();
        assert_eq!(registry.count(), 0);
    }

    #[test]
    fn test_new_is_empty() {
        let registry = ResourceRegistry::new();
        assert!(registry.is_empty());
    }

    #[test]
    fn test_new_generation_zero() {
        let registry = ResourceRegistry::new();
        assert_eq!(registry.generation(), 0);
    }

    #[test]
    fn test_declare_texture_returns_id() {
        let mut registry = ResourceRegistry::new();
        let id = registry.declare_texture(
            "color_buffer",
            TextureDescriptor::new_render_target(1920, 1080, TextureFormat::Rgba8Unorm),
        );
        assert!(id.is_some());
    }

    #[test]
    fn test_declare_texture_increments_count() {
        let mut registry = ResourceRegistry::new();
        registry.declare_texture(
            "color_buffer",
            TextureDescriptor::new_render_target(1920, 1080, TextureFormat::Rgba8Unorm),
        );
        assert_eq!(registry.count(), 1);
    }

    #[test]
    fn test_declare_texture_unique_ids() {
        let mut registry = ResourceRegistry::new();
        let id1 = registry.declare_texture("t1", TextureDescriptor::default());
        let id2 = registry.declare_texture("t2", TextureDescriptor::default());
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_declare_buffer_returns_id() {
        let mut registry = ResourceRegistry::new();
        let id = registry.declare_buffer("uniform_buffer", BufferDescriptor::new_uniform(256));
        assert!(id.is_some());
    }

    #[test]
    fn test_declare_buffer_increments_count() {
        let mut registry = ResourceRegistry::new();
        registry.declare_buffer("uniform_buffer", BufferDescriptor::new_uniform(256));
        assert_eq!(registry.count(), 1);
    }

    #[test]
    fn test_declare_buffer_unique_ids() {
        let mut registry = ResourceRegistry::new();
        let id1 = registry.declare_buffer("b1", BufferDescriptor::default());
        let id2 = registry.declare_buffer("b2", BufferDescriptor::default());
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_get_retrieves_handle() {
        let mut registry = ResourceRegistry::new();
        let id = registry.declare_texture(
            "color_buffer",
            TextureDescriptor::new_render_target(1920, 1080, TextureFormat::Rgba8Unorm),
        );
        let handle = registry.get(id);
        assert!(handle.is_some());
        assert_eq!(handle.unwrap().id, id);
    }

    #[test]
    fn test_get_nonexistent() {
        let registry = ResourceRegistry::new();
        let handle = registry.get(ResourceId::new(999));
        assert!(handle.is_none());
    }

    #[test]
    fn test_get_by_name_retrieves_id() {
        let mut registry = ResourceRegistry::new();
        let id = registry.declare_texture(
            "color_buffer",
            TextureDescriptor::new_render_target(1920, 1080, TextureFormat::Rgba8Unorm),
        );
        let retrieved_id = registry.get_by_name("color_buffer");
        assert_eq!(retrieved_id, Some(id));
    }

    #[test]
    fn test_get_by_name_nonexistent() {
        let registry = ResourceRegistry::new();
        let id = registry.get_by_name("nonexistent");
        assert!(id.is_none());
    }

    #[test]
    fn test_remove_deletes_resource() {
        let mut registry = ResourceRegistry::new();
        let id = registry.declare_buffer("temp", BufferDescriptor::default());
        let removed = registry.remove(id);
        assert!(removed.is_some());
        assert!(registry.get(id).is_none());
    }

    #[test]
    fn test_remove_clears_name_mapping() {
        let mut registry = ResourceRegistry::new();
        let id = registry.declare_buffer("temp", BufferDescriptor::default());
        registry.remove(id);
        assert!(registry.get_by_name("temp").is_none());
    }

    #[test]
    fn test_remove_increments_generation() {
        let mut registry = ResourceRegistry::new();
        let id = registry.declare_buffer("temp", BufferDescriptor::default());
        let gen_before = registry.generation();
        registry.remove(id);
        assert!(registry.generation() > gen_before);
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut registry = ResourceRegistry::new();
        let removed = registry.remove(ResourceId::new(999));
        assert!(removed.is_none());
    }

    #[test]
    fn test_clear_empties_registry() {
        let mut registry = ResourceRegistry::new();
        registry.declare_texture("a", TextureDescriptor::default());
        registry.declare_buffer("b", BufferDescriptor::default());
        registry.clear();
        assert!(registry.is_empty());
    }

    #[test]
    fn test_clear_increments_generation() {
        let mut registry = ResourceRegistry::new();
        registry.declare_texture("a", TextureDescriptor::default());
        let gen_before = registry.generation();
        registry.clear();
        assert!(registry.generation() > gen_before);
    }

    #[test]
    fn test_clear_clears_name_mappings() {
        let mut registry = ResourceRegistry::new();
        registry.declare_texture("a", TextureDescriptor::default());
        registry.clear();
        assert!(registry.get_by_name("a").is_none());
    }

    #[test]
    fn test_iter_returns_all_resources() {
        let mut registry = ResourceRegistry::new();
        registry.declare_texture("t1", TextureDescriptor::default());
        registry.declare_buffer("b1", BufferDescriptor::default());
        let items: Vec<_> = registry.iter().collect();
        assert_eq!(items.len(), 2);
    }

    #[test]
    fn test_iter_empty_registry() {
        let registry = ResourceRegistry::new();
        let items: Vec<_> = registry.iter().collect();
        assert!(items.is_empty());
    }

    #[test]
    fn test_count_accuracy() {
        let mut registry = ResourceRegistry::new();
        assert_eq!(registry.count(), 0);
        registry.declare_texture("t1", TextureDescriptor::default());
        assert_eq!(registry.count(), 1);
        registry.declare_buffer("b1", BufferDescriptor::default());
        assert_eq!(registry.count(), 2);
    }

    #[test]
    fn test_is_empty_false_after_declaration() {
        let mut registry = ResourceRegistry::new();
        registry.declare_texture("t1", TextureDescriptor::default());
        assert!(!registry.is_empty());
    }

    #[test]
    fn test_generation_increments_on_remove() {
        let mut registry = ResourceRegistry::new();
        let id = registry.declare_texture("t1", TextureDescriptor::default());
        let gen_before = registry.generation();
        registry.remove(id);
        assert!(registry.generation() > gen_before);
    }

    #[test]
    fn test_generation_increments_on_clear() {
        let mut registry = ResourceRegistry::new();
        registry.declare_texture("t1", TextureDescriptor::default());
        let gen_before = registry.generation();
        registry.clear();
        assert!(registry.generation() > gen_before);
    }

    #[test]
    fn test_total_size_bytes_aggregation() {
        let mut registry = ResourceRegistry::new();
        // 100*100*4 = 40000
        registry.declare_texture(
            "tex",
            TextureDescriptor::new_2d(100, 100, TextureFormat::Rgba8Unorm),
        );
        // 1024 bytes
        registry.declare_buffer("buf", BufferDescriptor::new(1024, BufferUsages::UNIFORM));
        assert_eq!(registry.total_size_bytes(), 41024);
    }

    #[test]
    fn test_total_size_bytes_empty() {
        let registry = ResourceRegistry::new();
        assert_eq!(registry.total_size_bytes(), 0);
    }

    #[test]
    #[should_panic(expected = "already exists")]
    fn test_duplicate_name_panics() {
        let mut registry = ResourceRegistry::new();
        registry.declare_texture("dup", TextureDescriptor::default());
        registry.declare_texture("dup", TextureDescriptor::default());
    }

    #[test]
    #[should_panic(expected = "already exists")]
    fn test_duplicate_name_different_types_panics() {
        let mut registry = ResourceRegistry::new();
        registry.declare_texture("resource", TextureDescriptor::default());
        registry.declare_buffer("resource", BufferDescriptor::default());
    }

    #[test]
    fn test_id_uniqueness_across_types() {
        let mut registry = ResourceRegistry::new();
        let tex_id = registry.declare_texture("tex", TextureDescriptor::default());
        let buf_id = registry.declare_buffer("buf", BufferDescriptor::default());
        assert_ne!(tex_id, buf_id);
    }

    #[test]
    fn test_ids_never_reused_after_remove() {
        let mut registry = ResourceRegistry::new();
        let id1 = registry.declare_texture("t1", TextureDescriptor::default());
        registry.remove(id1);
        let id2 = registry.declare_texture("t2", TextureDescriptor::default());
        assert_ne!(id1, id2);
        assert!(id2.raw() > id1.raw());
    }

    #[test]
    fn test_ids_never_reused_after_clear() {
        let mut registry = ResourceRegistry::new();
        let id1 = registry.declare_texture("t1", TextureDescriptor::default());
        registry.clear();
        let id2 = registry.declare_texture("t2", TextureDescriptor::default());
        assert_ne!(id1, id2);
        assert!(id2.raw() > id1.raw());
    }

    #[test]
    fn test_many_resources() {
        let mut registry = ResourceRegistry::new();
        let mut ids = HashSet::new();
        for i in 0..100 {
            let id = registry.declare_texture(format!("tex_{}", i), TextureDescriptor::default());
            ids.insert(id);
        }
        assert_eq!(ids.len(), 100);
        assert_eq!(registry.count(), 100);
    }

    #[test]
    fn test_default() {
        let registry = ResourceRegistry::default();
        assert!(registry.is_empty());
        assert_eq!(registry.generation(), 0);
    }

    #[test]
    fn test_clone() {
        let mut registry = ResourceRegistry::new();
        registry.declare_texture("t1", TextureDescriptor::default());
        let cloned = registry.clone();
        assert_eq!(registry.count(), cloned.count());
    }
}

// ===========================================================================
// Additional Edge Cases and Integration Tests
// ===========================================================================

mod integration_tests {
    use super::*;

    #[test]
    fn test_registry_with_labeled_resources() {
        let mut registry = ResourceRegistry::new();
        let tex_id = registry.declare_texture(
            "labeled_tex",
            TextureDescriptor::new_2d(256, 256, TextureFormat::Rgba8Unorm)
                .with_label("internal_label"),
        );
        let handle = registry.get(tex_id).unwrap();
        assert_eq!(handle.descriptor.label(), Some("internal_label"));
    }

    #[test]
    fn test_registry_texture_handle_is_texture() {
        let mut registry = ResourceRegistry::new();
        let id = registry.declare_texture("tex", TextureDescriptor::default());
        let handle = registry.get(id).unwrap();
        assert!(handle.is_texture());
        assert!(!handle.is_buffer());
    }

    #[test]
    fn test_registry_buffer_handle_is_buffer() {
        let mut registry = ResourceRegistry::new();
        let id = registry.declare_buffer("buf", BufferDescriptor::default());
        let handle = registry.get(id).unwrap();
        assert!(handle.is_buffer());
        assert!(!handle.is_texture());
    }

    #[test]
    fn test_texture_view_from_registry_resource() {
        let mut registry = ResourceRegistry::new();
        let id = registry.declare_texture(
            "depth",
            TextureDescriptor::new_depth(1024, 768).with_mips(4),
        );
        let view = TextureView::new(id).with_mip_range(0, Some(1));
        assert_eq!(view.resource, id);
    }

    #[test]
    fn test_buffer_slice_from_registry_resource() {
        let mut registry = ResourceRegistry::new();
        let id = registry.declare_buffer("storage", BufferDescriptor::new_storage(65536));
        let slice = BufferSlice::new(id).with_range(0, Some(1024));
        assert_eq!(slice.resource, id);
    }

    #[test]
    fn test_size_calculation_for_common_formats() {
        // Test a variety of texture formats
        let formats_and_sizes = [
            (TextureFormat::R8Unorm, 1),
            (TextureFormat::Rg8Unorm, 2),
            (TextureFormat::Rgba8Unorm, 4),
            (TextureFormat::Rgba16Float, 8),
            (TextureFormat::Rgba32Float, 16),
        ];

        for (format, bytes_per_texel) in formats_and_sizes {
            let desc = TextureDescriptor::new_2d(64, 64, format);
            assert_eq!(
                desc.size_bytes(),
                64 * 64 * bytes_per_texel,
                "Failed for format {:?}",
                format
            );
        }
    }

    #[test]
    fn test_mipmap_chain_size_calculation() {
        // Full mip chain for 256x256 RGBA8: 256*256 + 128*128 + 64*64 + ... + 1*1
        // = 65536 + 16384 + 4096 + 1024 + 256 + 64 + 16 + 4 + 1 = 87381 * 4 bytes
        let desc = TextureDescriptor::new_2d(256, 256, TextureFormat::Rgba8Unorm).with_mips(9);
        // Calculation: sum of (2^(8-i))^2 for i=0..8 = 87381
        // With 4 bytes per texel = 349524
        let expected = (256u64 * 256 + 128 * 128 + 64 * 64 + 32 * 32 + 16 * 16 + 8 * 8 + 4 * 4 + 2 * 2 + 1 * 1) * 4;
        assert_eq!(desc.size_bytes(), expected);
    }

    #[test]
    fn test_msaa_multiplier() {
        let base = TextureDescriptor::new_2d(256, 256, TextureFormat::Rgba8Unorm);
        let msaa4 = TextureDescriptor::new_2d(256, 256, TextureFormat::Rgba8Unorm).with_msaa(4);
        assert_eq!(msaa4.size_bytes(), base.size_bytes() * 4);
    }

    #[test]
    fn test_resource_handle_generation_matches_registry() {
        let mut registry = ResourceRegistry::new();
        let id = registry.declare_texture("t", TextureDescriptor::default());
        let handle = registry.get(id).unwrap();
        assert_eq!(handle.generation, registry.generation());
    }

    #[test]
    fn test_iter_contains_correct_resources() {
        let mut registry = ResourceRegistry::new();
        let id1 = registry.declare_texture("t1", TextureDescriptor::default());
        let id2 = registry.declare_buffer("b1", BufferDescriptor::default());

        let ids: HashSet<ResourceId> = registry.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&id1));
        assert!(ids.contains(&id2));
    }

    #[test]
    fn test_depth_format_bytes() {
        let desc = TextureDescriptor::new_depth(100, 100);
        // Depth32Float = 4 bytes per texel
        assert_eq!(desc.size_bytes(), 100 * 100 * 4);
    }

    #[test]
    fn test_texture_aspect_coverage() {
        // Test all texture aspects
        let id = ResourceId::new(0);

        let all = TextureView::new(id);
        assert_eq!(all.aspect, TextureAspect::All);

        let depth = TextureView::depth_only(id);
        assert_eq!(depth.aspect, TextureAspect::DepthOnly);

        let stencil = TextureView::stencil_only(id);
        assert_eq!(stencil.aspect, TextureAspect::StencilOnly);
    }

    #[test]
    fn test_buffer_usage_combinations() {
        // Verify complex usage combinations work
        let desc = BufferDescriptor::new(
            1024,
            BufferUsages::VERTEX | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        );
        assert!(desc.usage.contains(BufferUsages::VERTEX));
        assert!(desc.usage.contains(BufferUsages::COPY_DST));
        assert!(desc.usage.contains(BufferUsages::COPY_SRC));
    }

    #[test]
    fn test_registry_stress_declare_remove() {
        let mut registry = ResourceRegistry::new();

        // Add and remove many resources
        for i in 0..50 {
            let id = registry.declare_texture(format!("tex_{}", i), TextureDescriptor::default());
            if i % 2 == 0 {
                registry.remove(id);
            }
        }

        // Should have 25 remaining
        assert_eq!(registry.count(), 25);
    }
}
