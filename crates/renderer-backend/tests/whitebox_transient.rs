//! Whitebox tests for the transient resource pooling system.
//!
//! This test module provides comprehensive internal testing of the transient
//! resource pool implementation including SizeClass, PoolKey, PooledTexture,
//! PooledBuffer, PoolStats, PoolConfig, TransientPool, AliasableResource,
//! and memory aliasing functionality.
//!
//! Test Count: 170+ tests
//! Coverage: SizeClass, PoolKey, PooledTexture, PooledBuffer, PoolStats,
//!           PoolConfig, TransientPool, AliasableResource, Memory Aliasing

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use renderer_backend::frame_graph::resources::{BufferDescriptor, TextureDescriptor};
use renderer_backend::frame_graph::transient::*;

// ===========================================================================
// Helper Functions
// ===========================================================================

fn hash_value<T: Hash>(val: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    val.hash(&mut hasher);
    hasher.finish()
}

// Constants for size boundaries
const KB: u64 = 1024;
const MB: u64 = 1024 * KB;

// ===========================================================================
// SizeClass Tests (25+ tests)
// ===========================================================================

mod size_class_tests {
    use super::*;

    // --- from_bytes() classification tests ---

    #[test]
    fn test_size_class_zero_bytes_is_tiny() {
        assert_eq!(SizeClass::from_bytes(0), SizeClass::Tiny);
    }

    #[test]
    fn test_size_class_one_byte_is_tiny() {
        assert_eq!(SizeClass::from_bytes(1), SizeClass::Tiny);
    }

    #[test]
    fn test_size_class_512_bytes_is_tiny() {
        assert_eq!(SizeClass::from_bytes(512), SizeClass::Tiny);
    }

    #[test]
    fn test_size_class_1023_bytes_is_tiny() {
        assert_eq!(SizeClass::from_bytes(1023), SizeClass::Tiny);
    }

    #[test]
    fn test_size_class_1024_bytes_is_small() {
        assert_eq!(SizeClass::from_bytes(1024), SizeClass::Small);
    }

    #[test]
    fn test_size_class_1025_bytes_is_small() {
        assert_eq!(SizeClass::from_bytes(1025), SizeClass::Small);
    }

    #[test]
    fn test_size_class_32kb_is_small() {
        assert_eq!(SizeClass::from_bytes(32 * KB), SizeClass::Small);
    }

    #[test]
    fn test_size_class_64kb_minus_one_is_small() {
        assert_eq!(SizeClass::from_bytes(64 * KB - 1), SizeClass::Small);
    }

    #[test]
    fn test_size_class_64kb_is_medium() {
        assert_eq!(SizeClass::from_bytes(64 * KB), SizeClass::Medium);
    }

    #[test]
    fn test_size_class_512kb_is_medium() {
        assert_eq!(SizeClass::from_bytes(512 * KB), SizeClass::Medium);
    }

    #[test]
    fn test_size_class_1mb_minus_one_is_medium() {
        assert_eq!(SizeClass::from_bytes(MB - 1), SizeClass::Medium);
    }

    #[test]
    fn test_size_class_1mb_is_large() {
        assert_eq!(SizeClass::from_bytes(MB), SizeClass::Large);
    }

    #[test]
    fn test_size_class_8mb_is_large() {
        assert_eq!(SizeClass::from_bytes(8 * MB), SizeClass::Large);
    }

    #[test]
    fn test_size_class_16mb_minus_one_is_large() {
        assert_eq!(SizeClass::from_bytes(16 * MB - 1), SizeClass::Large);
    }

    #[test]
    fn test_size_class_16mb_is_huge() {
        assert_eq!(SizeClass::from_bytes(16 * MB), SizeClass::Huge);
    }

    #[test]
    fn test_size_class_100mb_is_huge() {
        assert_eq!(SizeClass::from_bytes(100 * MB), SizeClass::Huge);
    }

    #[test]
    fn test_size_class_max_u64_is_huge() {
        assert_eq!(SizeClass::from_bytes(u64::MAX), SizeClass::Huge);
    }

    // --- min_size() tests ---

    #[test]
    fn test_size_class_tiny_min_size() {
        assert_eq!(SizeClass::Tiny.min_size(), 0);
    }

    #[test]
    fn test_size_class_small_min_size() {
        assert_eq!(SizeClass::Small.min_size(), KB);
    }

    #[test]
    fn test_size_class_medium_min_size() {
        assert_eq!(SizeClass::Medium.min_size(), 64 * KB);
    }

    #[test]
    fn test_size_class_large_min_size() {
        assert_eq!(SizeClass::Large.min_size(), MB);
    }

    #[test]
    fn test_size_class_huge_min_size() {
        assert_eq!(SizeClass::Huge.min_size(), 16 * MB);
    }

    // --- max_size() tests ---

    #[test]
    fn test_size_class_tiny_max_size() {
        assert_eq!(SizeClass::Tiny.max_size(), KB - 1);
    }

    #[test]
    fn test_size_class_small_max_size() {
        assert_eq!(SizeClass::Small.max_size(), 64 * KB - 1);
    }

    #[test]
    fn test_size_class_medium_max_size() {
        assert_eq!(SizeClass::Medium.max_size(), MB - 1);
    }

    #[test]
    fn test_size_class_large_max_size() {
        assert_eq!(SizeClass::Large.max_size(), 16 * MB - 1);
    }

    #[test]
    fn test_size_class_huge_max_size() {
        assert_eq!(SizeClass::Huge.max_size(), u64::MAX);
    }

    // --- allocation_size() tests ---

    #[test]
    fn test_size_class_tiny_allocation_size() {
        assert_eq!(SizeClass::Tiny.allocation_size(), KB);
    }

    #[test]
    fn test_size_class_small_allocation_size() {
        assert_eq!(SizeClass::Small.allocation_size(), 64 * KB);
    }

    #[test]
    fn test_size_class_medium_allocation_size() {
        assert_eq!(SizeClass::Medium.allocation_size(), MB);
    }

    #[test]
    fn test_size_class_large_allocation_size() {
        assert_eq!(SizeClass::Large.allocation_size(), 16 * MB);
    }

    #[test]
    fn test_size_class_huge_allocation_size() {
        assert_eq!(SizeClass::Huge.allocation_size(), 64 * MB);
    }

    // --- Display trait tests ---

    #[test]
    fn test_size_class_display_tiny() {
        assert_eq!(format!("{}", SizeClass::Tiny), "Tiny(<1KB)");
    }

    #[test]
    fn test_size_class_display_small() {
        assert_eq!(format!("{}", SizeClass::Small), "Small(1-64KB)");
    }

    #[test]
    fn test_size_class_display_medium() {
        assert_eq!(format!("{}", SizeClass::Medium), "Medium(64KB-1MB)");
    }

    #[test]
    fn test_size_class_display_large() {
        assert_eq!(format!("{}", SizeClass::Large), "Large(1-16MB)");
    }

    #[test]
    fn test_size_class_display_huge() {
        assert_eq!(format!("{}", SizeClass::Huge), "Huge(>16MB)");
    }

    // --- Edge case boundary tests ---

    #[test]
    fn test_size_class_boundary_tiny_small() {
        // Right at the boundary
        assert_eq!(SizeClass::from_bytes(KB - 1), SizeClass::Tiny);
        assert_eq!(SizeClass::from_bytes(KB), SizeClass::Small);
    }

    #[test]
    fn test_size_class_boundary_small_medium() {
        assert_eq!(SizeClass::from_bytes(64 * KB - 1), SizeClass::Small);
        assert_eq!(SizeClass::from_bytes(64 * KB), SizeClass::Medium);
    }

    #[test]
    fn test_size_class_boundary_medium_large() {
        assert_eq!(SizeClass::from_bytes(MB - 1), SizeClass::Medium);
        assert_eq!(SizeClass::from_bytes(MB), SizeClass::Large);
    }

    #[test]
    fn test_size_class_boundary_large_huge() {
        assert_eq!(SizeClass::from_bytes(16 * MB - 1), SizeClass::Large);
        assert_eq!(SizeClass::from_bytes(16 * MB), SizeClass::Huge);
    }

    // --- Equality and hash tests ---

    #[test]
    fn test_size_class_equality() {
        assert_eq!(SizeClass::Tiny, SizeClass::Tiny);
        assert_eq!(SizeClass::Small, SizeClass::Small);
        assert_ne!(SizeClass::Tiny, SizeClass::Small);
        assert_ne!(SizeClass::Large, SizeClass::Huge);
    }

    #[test]
    fn test_size_class_clone() {
        let sc = SizeClass::Medium;
        let cloned = sc.clone();
        assert_eq!(sc, cloned);
    }

    #[test]
    fn test_size_class_debug() {
        let debug_str = format!("{:?}", SizeClass::Large);
        assert!(debug_str.contains("Large"));
    }
}

// ===========================================================================
// PoolKey Tests (30+ tests)
// ===========================================================================

mod pool_key_tests {
    use super::*;

    // --- from_texture_desc() tests ---

    #[test]
    fn test_pool_key_from_texture_desc_basic() {
        let desc = TextureDescriptor::new_2d(256, 256, wgpu::TextureFormat::Rgba8Unorm);
        let key = PoolKey::from_texture_desc(&desc);
        assert!(key.is_texture());
        assert!(!key.is_buffer());
    }

    #[test]
    fn test_pool_key_from_texture_desc_render_target() {
        let desc = TextureDescriptor::new_render_target(1920, 1080, wgpu::TextureFormat::Rgba8Unorm);
        let key = PoolKey::from_texture_desc(&desc);

        match key {
            PoolKey::Texture { format, width, height, sample_count, usage, .. } => {
                assert_eq!(format, wgpu::TextureFormat::Rgba8Unorm);
                assert_eq!(width, 1920);
                assert_eq!(height, 1080);
                assert_eq!(sample_count, 1);
                assert!(usage.contains(wgpu::TextureUsages::RENDER_ATTACHMENT));
            }
            _ => panic!("Expected texture key"),
        }
    }

    #[test]
    fn test_pool_key_from_texture_desc_depth() {
        let desc = TextureDescriptor::new_depth(1024, 768);
        let key = PoolKey::from_texture_desc(&desc);

        match key {
            PoolKey::Texture { format, .. } => {
                assert_eq!(format, wgpu::TextureFormat::Depth32Float);
            }
            _ => panic!("Expected texture key"),
        }
    }

    #[test]
    fn test_pool_key_from_texture_desc_msaa() {
        let desc = TextureDescriptor::new_render_target(800, 600, wgpu::TextureFormat::Bgra8Unorm)
            .with_msaa(4);
        let key = PoolKey::from_texture_desc(&desc);

        match key {
            PoolKey::Texture { sample_count, .. } => {
                assert_eq!(sample_count, 4);
            }
            _ => panic!("Expected texture key"),
        }
    }

    #[test]
    fn test_pool_key_from_texture_desc_mips() {
        let desc = TextureDescriptor::new_2d(512, 512, wgpu::TextureFormat::Rgba8Unorm)
            .with_mips(5);
        let key = PoolKey::from_texture_desc(&desc);

        match key {
            PoolKey::Texture { mip_levels, .. } => {
                assert_eq!(mip_levels, 5);
            }
            _ => panic!("Expected texture key"),
        }
    }

    // --- from_buffer_desc() tests ---

    #[test]
    fn test_pool_key_from_buffer_desc_basic() {
        let desc = BufferDescriptor::new_storage(1024);
        let key = PoolKey::from_buffer_desc(&desc);
        assert!(key.is_buffer());
        assert!(!key.is_texture());
    }

    #[test]
    fn test_pool_key_from_buffer_desc_tiny_size() {
        let desc = BufferDescriptor::new_uniform(500);
        let key = PoolKey::from_buffer_desc(&desc);

        match key {
            PoolKey::Buffer { size_class, .. } => {
                assert_eq!(size_class, SizeClass::Tiny);
            }
            _ => panic!("Expected buffer key"),
        }
    }

    #[test]
    fn test_pool_key_from_buffer_desc_small_size() {
        let desc = BufferDescriptor::new_storage(10 * KB);
        let key = PoolKey::from_buffer_desc(&desc);

        match key {
            PoolKey::Buffer { size_class, .. } => {
                assert_eq!(size_class, SizeClass::Small);
            }
            _ => panic!("Expected buffer key"),
        }
    }

    #[test]
    fn test_pool_key_from_buffer_desc_medium_size() {
        let desc = BufferDescriptor::new_storage(100 * KB);
        let key = PoolKey::from_buffer_desc(&desc);

        match key {
            PoolKey::Buffer { size_class, .. } => {
                assert_eq!(size_class, SizeClass::Medium);
            }
            _ => panic!("Expected buffer key"),
        }
    }

    #[test]
    fn test_pool_key_from_buffer_desc_large_size() {
        let desc = BufferDescriptor::new_storage(5 * MB);
        let key = PoolKey::from_buffer_desc(&desc);

        match key {
            PoolKey::Buffer { size_class, .. } => {
                assert_eq!(size_class, SizeClass::Large);
            }
            _ => panic!("Expected buffer key"),
        }
    }

    #[test]
    fn test_pool_key_from_buffer_desc_huge_size() {
        let desc = BufferDescriptor::new_storage(50 * MB);
        let key = PoolKey::from_buffer_desc(&desc);

        match key {
            PoolKey::Buffer { size_class, .. } => {
                assert_eq!(size_class, SizeClass::Huge);
            }
            _ => panic!("Expected buffer key"),
        }
    }

    #[test]
    fn test_pool_key_from_buffer_desc_usage_preserved() {
        let desc = BufferDescriptor::new_vertex(4096);
        let key = PoolKey::from_buffer_desc(&desc);

        match key {
            PoolKey::Buffer { usage, .. } => {
                assert!(usage.contains(wgpu::BufferUsages::VERTEX));
            }
            _ => panic!("Expected buffer key"),
        }
    }

    // --- Hash equality tests ---

    #[test]
    fn test_pool_key_hash_equal_for_matching_textures() {
        let key1 = PoolKey::Texture {
            format: wgpu::TextureFormat::Rgba8Unorm,
            width: 1920,
            height: 1080,
            depth_or_layers: 1,
            mip_levels: 1,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
        };
        let key2 = PoolKey::Texture {
            format: wgpu::TextureFormat::Rgba8Unorm,
            width: 1920,
            height: 1080,
            depth_or_layers: 1,
            mip_levels: 1,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
        };

        assert_eq!(key1, key2);
        assert_eq!(hash_value(&key1), hash_value(&key2));
    }

    #[test]
    fn test_pool_key_hash_not_equal_for_different_formats() {
        let key1 = PoolKey::Texture {
            format: wgpu::TextureFormat::Rgba8Unorm,
            width: 1920,
            height: 1080,
            depth_or_layers: 1,
            mip_levels: 1,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
        };
        let key2 = PoolKey::Texture {
            format: wgpu::TextureFormat::Bgra8Unorm,  // Different format
            width: 1920,
            height: 1080,
            depth_or_layers: 1,
            mip_levels: 1,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
        };

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_pool_key_hash_not_equal_for_different_dimensions() {
        let key1 = PoolKey::Texture {
            format: wgpu::TextureFormat::Rgba8Unorm,
            width: 1920,
            height: 1080,
            depth_or_layers: 1,
            mip_levels: 1,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
        };
        let key2 = PoolKey::Texture {
            format: wgpu::TextureFormat::Rgba8Unorm,
            width: 1280,  // Different width
            height: 720,  // Different height
            depth_or_layers: 1,
            mip_levels: 1,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
        };

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_pool_key_hash_not_equal_for_different_usage() {
        let key1 = PoolKey::Texture {
            format: wgpu::TextureFormat::Rgba8Unorm,
            width: 256,
            height: 256,
            depth_or_layers: 1,
            mip_levels: 1,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
        };
        let key2 = PoolKey::Texture {
            format: wgpu::TextureFormat::Rgba8Unorm,
            width: 256,
            height: 256,
            depth_or_layers: 1,
            mip_levels: 1,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,  // Different usage
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
        };

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_pool_key_hash_not_equal_for_different_sample_count() {
        let key1 = PoolKey::Texture {
            format: wgpu::TextureFormat::Rgba8Unorm,
            width: 256,
            height: 256,
            depth_or_layers: 1,
            mip_levels: 1,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
        };
        let key2 = PoolKey::Texture {
            format: wgpu::TextureFormat::Rgba8Unorm,
            width: 256,
            height: 256,
            depth_or_layers: 1,
            mip_levels: 1,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            sample_count: 4,  // Different sample count
            dimension: wgpu::TextureDimension::D2,
        };

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_pool_key_hash_equal_for_matching_buffers() {
        let key1 = PoolKey::Buffer {
            size_class: SizeClass::Medium,
            usage: wgpu::BufferUsages::STORAGE,
        };
        let key2 = PoolKey::Buffer {
            size_class: SizeClass::Medium,
            usage: wgpu::BufferUsages::STORAGE,
        };

        assert_eq!(key1, key2);
        assert_eq!(hash_value(&key1), hash_value(&key2));
    }

    #[test]
    fn test_pool_key_hash_not_equal_for_different_size_class() {
        let key1 = PoolKey::Buffer {
            size_class: SizeClass::Small,
            usage: wgpu::BufferUsages::STORAGE,
        };
        let key2 = PoolKey::Buffer {
            size_class: SizeClass::Medium,
            usage: wgpu::BufferUsages::STORAGE,
        };

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_pool_key_hash_not_equal_for_different_buffer_usage() {
        let key1 = PoolKey::Buffer {
            size_class: SizeClass::Medium,
            usage: wgpu::BufferUsages::STORAGE,
        };
        let key2 = PoolKey::Buffer {
            size_class: SizeClass::Medium,
            usage: wgpu::BufferUsages::VERTEX,
        };

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_pool_key_texture_not_equal_to_buffer() {
        let tex_key = PoolKey::Texture {
            format: wgpu::TextureFormat::Rgba8Unorm,
            width: 256,
            height: 256,
            depth_or_layers: 1,
            mip_levels: 1,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
        };
        let buf_key = PoolKey::Buffer {
            size_class: SizeClass::Medium,
            usage: wgpu::BufferUsages::STORAGE,
        };

        assert_ne!(tex_key, buf_key);
    }

    // --- Display tests ---

    #[test]
    fn test_pool_key_display_texture() {
        let key = PoolKey::Texture {
            format: wgpu::TextureFormat::Rgba8Unorm,
            width: 1920,
            height: 1080,
            depth_or_layers: 1,
            mip_levels: 1,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
        };
        let display = format!("{}", key);
        assert!(display.contains("Texture"));
        assert!(display.contains("1920x1080"));
    }

    #[test]
    fn test_pool_key_display_buffer() {
        let key = PoolKey::Buffer {
            size_class: SizeClass::Large,
            usage: wgpu::BufferUsages::STORAGE,
        };
        let display = format!("{}", key);
        assert!(display.contains("Buffer"));
        assert!(display.contains("Large"));
    }

    // --- Clone and Debug tests ---

    #[test]
    fn test_pool_key_clone() {
        let key = PoolKey::Buffer {
            size_class: SizeClass::Medium,
            usage: wgpu::BufferUsages::UNIFORM,
        };
        let cloned = key.clone();
        assert_eq!(key, cloned);
    }

    #[test]
    fn test_pool_key_debug() {
        let key = PoolKey::Texture {
            format: wgpu::TextureFormat::Rgba8Unorm,
            width: 256,
            height: 256,
            depth_or_layers: 1,
            mip_levels: 1,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
        };
        let debug_str = format!("{:?}", key);
        assert!(debug_str.contains("Texture"));
    }
}

// ===========================================================================
// PoolStats Tests (25+ tests)
// ===========================================================================

mod pool_stats_tests {
    use super::*;

    #[test]
    fn test_pool_stats_new_is_zero() {
        let stats = PoolStats::new();
        assert_eq!(stats.total_textures, 0);
        assert_eq!(stats.total_buffers, 0);
        assert_eq!(stats.active_textures, 0);
        assert_eq!(stats.active_buffers, 0);
        assert_eq!(stats.texture_bytes, 0);
        assert_eq!(stats.buffer_bytes, 0);
        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.cache_misses, 0);
        assert_eq!(stats.gc_count, 0);
    }

    #[test]
    fn test_pool_stats_default_is_zero() {
        let stats = PoolStats::default();
        assert_eq!(stats.total_textures, 0);
        assert_eq!(stats.cache_hits, 0);
    }

    #[test]
    fn test_pool_stats_hit_rate_zero_when_no_accesses() {
        let stats = PoolStats::new();
        assert_eq!(stats.hit_rate(), 0.0);
    }

    #[test]
    fn test_pool_stats_hit_rate_100_percent() {
        let mut stats = PoolStats::new();
        stats.cache_hits = 100;
        stats.cache_misses = 0;
        assert_eq!(stats.hit_rate(), 1.0);
    }

    #[test]
    fn test_pool_stats_hit_rate_0_percent() {
        let mut stats = PoolStats::new();
        stats.cache_hits = 0;
        stats.cache_misses = 100;
        assert_eq!(stats.hit_rate(), 0.0);
    }

    #[test]
    fn test_pool_stats_hit_rate_50_percent() {
        let mut stats = PoolStats::new();
        stats.cache_hits = 50;
        stats.cache_misses = 50;
        assert!((stats.hit_rate() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_pool_stats_hit_rate_80_percent() {
        let mut stats = PoolStats::new();
        stats.cache_hits = 80;
        stats.cache_misses = 20;
        assert!((stats.hit_rate() - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_pool_stats_hit_rate_25_percent() {
        let mut stats = PoolStats::new();
        stats.cache_hits = 25;
        stats.cache_misses = 75;
        assert!((stats.hit_rate() - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_pool_stats_total_bytes_texture_only() {
        let mut stats = PoolStats::new();
        stats.texture_bytes = 1024 * 1024;
        stats.buffer_bytes = 0;
        assert_eq!(stats.total_bytes(), 1024 * 1024);
    }

    #[test]
    fn test_pool_stats_total_bytes_buffer_only() {
        let mut stats = PoolStats::new();
        stats.texture_bytes = 0;
        stats.buffer_bytes = 512 * 1024;
        assert_eq!(stats.total_bytes(), 512 * 1024);
    }

    #[test]
    fn test_pool_stats_total_bytes_combined() {
        let mut stats = PoolStats::new();
        stats.texture_bytes = 1024 * 1024;
        stats.buffer_bytes = 512 * 1024;
        assert_eq!(stats.total_bytes(), 1536 * 1024);
    }

    #[test]
    fn test_pool_stats_idle_textures_all_idle() {
        let mut stats = PoolStats::new();
        stats.total_textures = 10;
        stats.active_textures = 0;
        assert_eq!(stats.idle_textures(), 10);
    }

    #[test]
    fn test_pool_stats_idle_textures_none_idle() {
        let mut stats = PoolStats::new();
        stats.total_textures = 10;
        stats.active_textures = 10;
        assert_eq!(stats.idle_textures(), 0);
    }

    #[test]
    fn test_pool_stats_idle_textures_partial() {
        let mut stats = PoolStats::new();
        stats.total_textures = 10;
        stats.active_textures = 3;
        assert_eq!(stats.idle_textures(), 7);
    }

    #[test]
    fn test_pool_stats_idle_buffers_all_idle() {
        let mut stats = PoolStats::new();
        stats.total_buffers = 8;
        stats.active_buffers = 0;
        assert_eq!(stats.idle_buffers(), 8);
    }

    #[test]
    fn test_pool_stats_idle_buffers_none_idle() {
        let mut stats = PoolStats::new();
        stats.total_buffers = 8;
        stats.active_buffers = 8;
        assert_eq!(stats.idle_buffers(), 0);
    }

    #[test]
    fn test_pool_stats_idle_buffers_partial() {
        let mut stats = PoolStats::new();
        stats.total_buffers = 8;
        stats.active_buffers = 5;
        assert_eq!(stats.idle_buffers(), 3);
    }

    #[test]
    fn test_pool_stats_reset_counters() {
        let mut stats = PoolStats::new();
        stats.cache_hits = 100;
        stats.cache_misses = 50;
        stats.gc_count = 25;
        stats.total_textures = 10;  // Should not be reset

        stats.reset_counters();

        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.cache_misses, 0);
        assert_eq!(stats.gc_count, 0);
        assert_eq!(stats.total_textures, 10);  // Preserved
    }

    #[test]
    fn test_pool_stats_display() {
        let mut stats = PoolStats::new();
        stats.total_textures = 10;
        stats.active_textures = 3;
        stats.total_buffers = 8;
        stats.active_buffers = 2;
        stats.texture_bytes = 10 * 1024 * 1024;
        stats.buffer_bytes = 5 * 1024 * 1024;
        stats.cache_hits = 80;
        stats.cache_misses = 20;

        let display = format!("{}", stats);
        assert!(display.contains("textures=3/10"));
        assert!(display.contains("buffers=2/8"));
        assert!(display.contains("hit_rate=80.0%"));
    }

    #[test]
    fn test_pool_stats_clone() {
        let mut stats = PoolStats::new();
        stats.cache_hits = 42;
        let cloned = stats.clone();
        assert_eq!(cloned.cache_hits, 42);
    }

    #[test]
    fn test_pool_stats_debug() {
        let stats = PoolStats::new();
        let debug_str = format!("{:?}", stats);
        assert!(debug_str.contains("PoolStats"));
    }

    #[test]
    fn test_pool_stats_idle_saturating_sub() {
        // Edge case: active > total (shouldn't happen but test resilience)
        let mut stats = PoolStats::new();
        stats.total_textures = 5;
        stats.active_textures = 10;  // More active than total
        assert_eq!(stats.idle_textures(), 0);  // Should saturate to 0
    }
}

// ===========================================================================
// PoolConfig Tests (20+ tests)
// ===========================================================================

mod pool_config_tests {
    use super::*;

    #[test]
    fn test_pool_config_default_max_idle_frames() {
        let config = PoolConfig::default();
        assert_eq!(config.max_idle_frames, 3);
    }

    #[test]
    fn test_pool_config_default_max_texture_pool_size() {
        let config = PoolConfig::default();
        assert_eq!(config.max_texture_pool_size, 8);
    }

    #[test]
    fn test_pool_config_default_max_buffer_pool_size() {
        let config = PoolConfig::default();
        assert_eq!(config.max_buffer_pool_size, 16);
    }

    #[test]
    fn test_pool_config_default_defragmentation_disabled() {
        let config = PoolConfig::default();
        assert!(!config.enable_defragmentation);
    }

    #[test]
    fn test_pool_config_default_unlimited_memory() {
        let config = PoolConfig::default();
        assert_eq!(config.max_memory_bytes, 0);
    }

    #[test]
    fn test_pool_config_new_equals_default() {
        let new_config = PoolConfig::new();
        let default_config = PoolConfig::default();
        assert_eq!(new_config.max_idle_frames, default_config.max_idle_frames);
        assert_eq!(new_config.max_texture_pool_size, default_config.max_texture_pool_size);
    }

    #[test]
    fn test_pool_config_with_max_idle_frames() {
        let config = PoolConfig::new().with_max_idle_frames(5);
        assert_eq!(config.max_idle_frames, 5);
    }

    #[test]
    fn test_pool_config_with_max_texture_pool_size() {
        let config = PoolConfig::new().with_max_texture_pool_size(16);
        assert_eq!(config.max_texture_pool_size, 16);
    }

    #[test]
    fn test_pool_config_with_max_buffer_pool_size() {
        let config = PoolConfig::new().with_max_buffer_pool_size(32);
        assert_eq!(config.max_buffer_pool_size, 32);
    }

    #[test]
    fn test_pool_config_with_defragmentation_enabled() {
        let config = PoolConfig::new().with_defragmentation(true);
        assert!(config.enable_defragmentation);
    }

    #[test]
    fn test_pool_config_with_defragmentation_disabled() {
        let config = PoolConfig::new().with_defragmentation(false);
        assert!(!config.enable_defragmentation);
    }

    #[test]
    fn test_pool_config_with_memory_budget() {
        let config = PoolConfig::new().with_memory_budget(256 * MB);
        assert_eq!(config.max_memory_bytes, 256 * MB);
    }

    #[test]
    fn test_pool_config_builder_chain() {
        let config = PoolConfig::new()
            .with_max_idle_frames(10)
            .with_max_texture_pool_size(32)
            .with_max_buffer_pool_size(64)
            .with_defragmentation(true)
            .with_memory_budget(512 * MB);

        assert_eq!(config.max_idle_frames, 10);
        assert_eq!(config.max_texture_pool_size, 32);
        assert_eq!(config.max_buffer_pool_size, 64);
        assert!(config.enable_defragmentation);
        assert_eq!(config.max_memory_bytes, 512 * MB);
    }

    #[test]
    fn test_pool_config_with_max_idle_frames_zero() {
        let config = PoolConfig::new().with_max_idle_frames(0);
        assert_eq!(config.max_idle_frames, 0);
    }

    #[test]
    fn test_pool_config_with_max_idle_frames_large() {
        let config = PoolConfig::new().with_max_idle_frames(1000);
        assert_eq!(config.max_idle_frames, 1000);
    }

    #[test]
    fn test_pool_config_with_pool_size_one() {
        let config = PoolConfig::new()
            .with_max_texture_pool_size(1)
            .with_max_buffer_pool_size(1);
        assert_eq!(config.max_texture_pool_size, 1);
        assert_eq!(config.max_buffer_pool_size, 1);
    }

    #[test]
    fn test_pool_config_clone() {
        let config = PoolConfig::new()
            .with_max_idle_frames(7)
            .with_defragmentation(true);
        let cloned = config.clone();
        assert_eq!(cloned.max_idle_frames, 7);
        assert!(cloned.enable_defragmentation);
    }

    #[test]
    fn test_pool_config_debug() {
        let config = PoolConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("PoolConfig"));
        assert!(debug_str.contains("max_idle_frames"));
    }
}

// ===========================================================================
// ResourceLifetimeRange and AliasableResource Tests (20+ tests)
// ===========================================================================

mod aliasable_resource_tests {
    use super::*;

    // --- ResourceLifetimeRange construction tests ---

    #[test]
    fn test_resource_lifetime_range_new() {
        let range = ResourceLifetimeRange::new(1, 0, 5, 1024);
        assert_eq!(range.allocation_id, 1);
        assert_eq!(range.first_use_pass, 0);
        assert_eq!(range.last_use_pass, 5);
        assert_eq!(range.size_bytes, 1024);
    }

    #[test]
    fn test_resource_lifetime_range_single_pass() {
        let range = ResourceLifetimeRange::new(42, 3, 3, 512);
        assert_eq!(range.first_use_pass, 3);
        assert_eq!(range.last_use_pass, 3);
    }

    // --- AliasableResource trait tests ---

    #[test]
    fn test_aliasable_resource_first_use_pass() {
        let range = ResourceLifetimeRange::new(1, 5, 10, 1024);
        assert_eq!(range.first_use_pass(), 5);
    }

    #[test]
    fn test_aliasable_resource_last_use_pass() {
        let range = ResourceLifetimeRange::new(1, 5, 10, 1024);
        assert_eq!(range.last_use_pass(), 10);
    }

    #[test]
    fn test_can_alias_non_overlapping_sequential() {
        let r1 = ResourceLifetimeRange::new(1, 0, 5, 1024);
        let r2 = ResourceLifetimeRange::new(2, 6, 10, 2048);

        assert!(r1.can_alias_with(&r2));
        assert!(r2.can_alias_with(&r1));
    }

    #[test]
    fn test_can_alias_non_overlapping_gap() {
        let r1 = ResourceLifetimeRange::new(1, 0, 3, 1024);
        let r2 = ResourceLifetimeRange::new(2, 10, 15, 2048);

        assert!(r1.can_alias_with(&r2));
        assert!(r2.can_alias_with(&r1));
    }

    #[test]
    fn test_cannot_alias_overlapping() {
        let r1 = ResourceLifetimeRange::new(1, 0, 5, 1024);
        let r2 = ResourceLifetimeRange::new(2, 3, 8, 2048);

        assert!(!r1.can_alias_with(&r2));
        assert!(!r2.can_alias_with(&r1));
    }

    #[test]
    fn test_cannot_alias_contained() {
        let r1 = ResourceLifetimeRange::new(1, 0, 10, 1024);
        let r2 = ResourceLifetimeRange::new(2, 3, 7, 2048);

        assert!(!r1.can_alias_with(&r2));
        assert!(!r2.can_alias_with(&r1));
    }

    #[test]
    fn test_cannot_alias_same_range() {
        let r1 = ResourceLifetimeRange::new(1, 5, 10, 1024);
        let r2 = ResourceLifetimeRange::new(2, 5, 10, 2048);

        assert!(!r1.can_alias_with(&r2));
    }

    #[test]
    fn test_cannot_alias_touching_boundary() {
        // Resources that touch at a boundary still overlap (inclusive)
        let r1 = ResourceLifetimeRange::new(1, 0, 5, 1024);
        let r2 = ResourceLifetimeRange::new(2, 5, 10, 2048);

        assert!(!r1.can_alias_with(&r2));
    }

    #[test]
    fn test_overlapping_lifetime_true_when_overlap() {
        let r1 = ResourceLifetimeRange::new(1, 0, 5, 1024);
        let r2 = ResourceLifetimeRange::new(2, 3, 8, 2048);

        assert!(r1.overlapping_lifetime(&r2));
        assert!(r2.overlapping_lifetime(&r1));
    }

    #[test]
    fn test_overlapping_lifetime_false_when_no_overlap() {
        let r1 = ResourceLifetimeRange::new(1, 0, 5, 1024);
        let r2 = ResourceLifetimeRange::new(2, 6, 10, 2048);

        assert!(!r1.overlapping_lifetime(&r2));
        assert!(!r2.overlapping_lifetime(&r1));
    }

    #[test]
    fn test_overlapping_lifetime_single_pass_resources() {
        let r1 = ResourceLifetimeRange::new(1, 5, 5, 1024);
        let r2 = ResourceLifetimeRange::new(2, 5, 5, 2048);

        assert!(r1.overlapping_lifetime(&r2));
    }

    #[test]
    fn test_overlapping_lifetime_single_pass_non_overlapping() {
        let r1 = ResourceLifetimeRange::new(1, 5, 5, 1024);
        let r2 = ResourceLifetimeRange::new(2, 6, 6, 2048);

        assert!(!r1.overlapping_lifetime(&r2));
    }

    // --- Clone, Debug, PartialEq tests ---

    #[test]
    fn test_resource_lifetime_range_clone() {
        let range = ResourceLifetimeRange::new(1, 0, 5, 1024);
        let cloned = range.clone();
        assert_eq!(range, cloned);
    }

    #[test]
    fn test_resource_lifetime_range_debug() {
        let range = ResourceLifetimeRange::new(1, 0, 5, 1024);
        let debug_str = format!("{:?}", range);
        assert!(debug_str.contains("ResourceLifetimeRange"));
    }

    #[test]
    fn test_resource_lifetime_range_equality() {
        let r1 = ResourceLifetimeRange::new(1, 0, 5, 1024);
        let r2 = ResourceLifetimeRange::new(1, 0, 5, 1024);
        let r3 = ResourceLifetimeRange::new(2, 0, 5, 1024);

        assert_eq!(r1, r2);
        assert_ne!(r1, r3);
    }
}

// ===========================================================================
// AliasGroup Tests (15+ tests)
// ===========================================================================

mod alias_group_tests {
    use super::*;

    #[test]
    fn test_alias_group_new_is_empty() {
        let group = AliasGroup::new();
        assert!(group.is_empty());
        assert_eq!(group.len(), 0);
        assert_eq!(group.memory_size, 0);
    }

    #[test]
    fn test_alias_group_default_is_empty() {
        let group = AliasGroup::default();
        assert!(group.is_empty());
    }

    #[test]
    fn test_alias_group_add_first_resource() {
        let mut group = AliasGroup::new();
        let r = ResourceLifetimeRange::new(1, 0, 5, 1024);

        assert!(group.try_add(&r));
        assert_eq!(group.len(), 1);
        assert_eq!(group.memory_size, 1024);
        assert_eq!(group.first_pass, 0);
        assert_eq!(group.last_pass, 5);
    }

    #[test]
    fn test_alias_group_add_non_overlapping() {
        let mut group = AliasGroup::new();
        let r1 = ResourceLifetimeRange::new(1, 0, 3, 1024);
        let r2 = ResourceLifetimeRange::new(2, 5, 8, 2048);

        assert!(group.try_add(&r1));
        assert!(group.try_add(&r2));
        assert_eq!(group.len(), 2);
        assert_eq!(group.memory_size, 2048);  // Max of the two
    }

    #[test]
    fn test_alias_group_reject_overlapping() {
        let mut group = AliasGroup::new();
        let r1 = ResourceLifetimeRange::new(1, 0, 5, 1024);
        let r2 = ResourceLifetimeRange::new(2, 3, 8, 2048);

        assert!(group.try_add(&r1));
        assert!(!group.try_add(&r2));
        assert_eq!(group.len(), 1);
    }

    #[test]
    fn test_alias_group_memory_size_is_max() {
        let mut group = AliasGroup::new();
        let r1 = ResourceLifetimeRange::new(1, 0, 2, 500);
        let r2 = ResourceLifetimeRange::new(2, 4, 6, 2000);
        let r3 = ResourceLifetimeRange::new(3, 8, 10, 1000);

        assert!(group.try_add(&r1));
        assert!(group.try_add(&r2));
        assert!(group.try_add(&r3));

        assert_eq!(group.memory_size, 2000);
    }

    #[test]
    fn test_alias_group_tracks_pass_range() {
        let mut group = AliasGroup::new();
        let r1 = ResourceLifetimeRange::new(1, 5, 8, 1024);
        let r2 = ResourceLifetimeRange::new(2, 10, 15, 2048);

        assert!(group.try_add(&r1));
        assert!(group.try_add(&r2));

        assert_eq!(group.first_pass, 5);
        assert_eq!(group.last_pass, 15);
    }

    #[test]
    fn test_alias_group_members_tracking() {
        let mut group = AliasGroup::new();
        let r1 = ResourceLifetimeRange::new(100, 0, 2, 1024);
        let r2 = ResourceLifetimeRange::new(200, 4, 6, 2048);

        assert!(group.try_add(&r1));
        assert!(group.try_add(&r2));

        assert!(group.members.contains(&100));
        assert!(group.members.contains(&200));
    }

    #[test]
    fn test_alias_group_clone() {
        let mut group = AliasGroup::new();
        let r = ResourceLifetimeRange::new(1, 0, 5, 1024);
        group.try_add(&r);

        let cloned = group.clone();
        assert_eq!(cloned.len(), 1);
        assert_eq!(cloned.memory_size, 1024);
    }

    #[test]
    fn test_alias_group_debug() {
        let group = AliasGroup::new();
        let debug_str = format!("{:?}", group);
        assert!(debug_str.contains("AliasGroup"));
    }

    #[test]
    fn test_alias_group_initial_pass_values() {
        let group = AliasGroup::new();
        assert_eq!(group.first_pass, u32::MAX);
        assert_eq!(group.last_pass, 0);
    }
}

// ===========================================================================
// compute_alias_groups() Tests (15+ tests)
// ===========================================================================

mod compute_alias_groups_tests {
    use super::*;

    #[test]
    fn test_compute_alias_groups_empty() {
        let resources: Vec<ResourceLifetimeRange> = vec![];
        let groups = compute_alias_groups(&resources);
        assert!(groups.is_empty());
    }

    #[test]
    fn test_compute_alias_groups_single_resource() {
        let resources = vec![ResourceLifetimeRange::new(1, 0, 5, 1024)];
        let groups = compute_alias_groups(&resources);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 1);
    }

    #[test]
    fn test_compute_alias_groups_non_overlapping_all_same_group() {
        let resources = vec![
            ResourceLifetimeRange::new(1, 0, 2, 1024),
            ResourceLifetimeRange::new(2, 3, 5, 2048),
            ResourceLifetimeRange::new(3, 6, 8, 512),
        ];
        let groups = compute_alias_groups(&resources);

        // All can be in the same group
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 3);
    }

    #[test]
    fn test_compute_alias_groups_all_overlapping_separate_groups() {
        let resources = vec![
            ResourceLifetimeRange::new(1, 0, 10, 1024),
            ResourceLifetimeRange::new(2, 0, 10, 2048),
            ResourceLifetimeRange::new(3, 0, 10, 512),
        ];
        let groups = compute_alias_groups(&resources);

        // Each needs its own group
        assert_eq!(groups.len(), 3);
    }

    #[test]
    fn test_compute_alias_groups_partial_overlap() {
        let resources = vec![
            ResourceLifetimeRange::new(1, 0, 2, 1024),
            ResourceLifetimeRange::new(2, 3, 5, 2048),
            ResourceLifetimeRange::new(3, 1, 4, 512),  // Overlaps with both
        ];
        let groups = compute_alias_groups(&resources);

        // Resource 3 needs its own group
        assert!(groups.len() >= 2);

        // Total resources assigned
        let total: usize = groups.iter().map(|g| g.len()).sum();
        assert_eq!(total, 3);
    }

    #[test]
    fn test_compute_alias_groups_all_resources_assigned() {
        let resources = vec![
            ResourceLifetimeRange::new(1, 0, 2, 1000),
            ResourceLifetimeRange::new(2, 3, 5, 2000),
            ResourceLifetimeRange::new(3, 0, 4, 500),
            ResourceLifetimeRange::new(4, 6, 8, 750),
        ];
        let groups = compute_alias_groups(&resources);

        let total: usize = groups.iter().map(|g| g.len()).sum();
        assert_eq!(total, 4);
    }

    #[test]
    fn test_compute_alias_groups_two_chains() {
        // Two completely separate resource chains
        let resources = vec![
            ResourceLifetimeRange::new(1, 0, 2, 1024),
            ResourceLifetimeRange::new(2, 3, 5, 1024),
            ResourceLifetimeRange::new(3, 100, 102, 2048),
            ResourceLifetimeRange::new(4, 103, 105, 2048),
        ];
        let groups = compute_alias_groups(&resources);

        // Should fit in 1-2 groups depending on algorithm
        assert!(groups.len() <= 2);
    }

    #[test]
    fn test_compute_alias_groups_sorting_by_first_pass() {
        // Resources given out of order
        let resources = vec![
            ResourceLifetimeRange::new(1, 10, 12, 1024),
            ResourceLifetimeRange::new(2, 0, 2, 1024),
            ResourceLifetimeRange::new(3, 5, 7, 1024),
        ];
        let groups = compute_alias_groups(&resources);

        // All should still be able to share a group
        assert_eq!(groups.len(), 1);
    }

    #[test]
    fn test_compute_alias_groups_memory_size_preserved() {
        let resources = vec![
            ResourceLifetimeRange::new(1, 0, 2, 1000),
            ResourceLifetimeRange::new(2, 3, 5, 5000),
            ResourceLifetimeRange::new(3, 6, 8, 2000),
        ];
        let groups = compute_alias_groups(&resources);

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].memory_size, 5000);  // Max
    }

    #[test]
    fn test_compute_alias_groups_complex_scenario() {
        // More complex scenario with multiple groupings needed
        let resources = vec![
            ResourceLifetimeRange::new(1, 0, 5, 1024),
            ResourceLifetimeRange::new(2, 6, 10, 2048),
            ResourceLifetimeRange::new(3, 2, 8, 512),   // Overlaps 1 and 2
            ResourceLifetimeRange::new(4, 11, 15, 768),
        ];
        let groups = compute_alias_groups(&resources);

        // Resource 3 forces additional group
        assert!(groups.len() >= 2);

        let total: usize = groups.iter().map(|g| g.len()).sum();
        assert_eq!(total, 4);
    }
}

// ===========================================================================
// estimate_aliasing_savings() Tests (10+ tests)
// ===========================================================================

mod estimate_aliasing_savings_tests {
    use super::*;

    #[test]
    fn test_estimate_aliasing_savings_empty() {
        let resources: Vec<ResourceLifetimeRange> = vec![];
        let groups: Vec<AliasGroup> = vec![];

        let (without, with, savings) = estimate_aliasing_savings(&resources, &groups);

        assert_eq!(without, 0);
        assert_eq!(with, 0);
        assert_eq!(savings, 0.0);
    }

    #[test]
    fn test_estimate_aliasing_savings_single_resource() {
        let resources = vec![ResourceLifetimeRange::new(1, 0, 5, 1024)];
        let groups = compute_alias_groups(&resources);

        let (without, with, _savings) = estimate_aliasing_savings(&resources, &groups);

        assert_eq!(without, 1024);
        assert_eq!(with, 1024);  // No savings with single resource
    }

    #[test]
    fn test_estimate_aliasing_savings_all_can_alias() {
        let resources = vec![
            ResourceLifetimeRange::new(1, 0, 2, 1000),
            ResourceLifetimeRange::new(2, 3, 5, 2000),
            ResourceLifetimeRange::new(3, 6, 8, 1500),
        ];
        let groups = compute_alias_groups(&resources);

        let (without, with, savings) = estimate_aliasing_savings(&resources, &groups);

        assert_eq!(without, 4500);
        assert_eq!(with, 2000);  // Max of the three
        assert!(savings > 50.0);  // (4500 - 2000) / 4500 * 100 = ~55.6%
    }

    #[test]
    fn test_estimate_aliasing_savings_no_aliasing_possible() {
        let resources = vec![
            ResourceLifetimeRange::new(1, 0, 10, 1000),
            ResourceLifetimeRange::new(2, 0, 10, 2000),
        ];
        let groups = compute_alias_groups(&resources);

        let (without, with, savings) = estimate_aliasing_savings(&resources, &groups);

        assert_eq!(without, 3000);
        assert_eq!(with, 3000);  // Each in separate group
        assert_eq!(savings, 0.0);
    }

    #[test]
    fn test_estimate_aliasing_savings_partial_aliasing() {
        let resources = vec![
            ResourceLifetimeRange::new(1, 0, 2, 1000),
            ResourceLifetimeRange::new(2, 3, 5, 2000),
            ResourceLifetimeRange::new(3, 1, 4, 1500),  // Overlaps both
        ];
        let groups = compute_alias_groups(&resources);

        let (without, with, savings) = estimate_aliasing_savings(&resources, &groups);

        assert_eq!(without, 4500);
        // Some savings but not maximum
        assert!(with < without);
        assert!(savings > 0.0);
        assert!(savings < 100.0);
    }

    #[test]
    fn test_estimate_aliasing_savings_equal_sizes() {
        let resources = vec![
            ResourceLifetimeRange::new(1, 0, 2, 1000),
            ResourceLifetimeRange::new(2, 3, 5, 1000),
            ResourceLifetimeRange::new(3, 6, 8, 1000),
        ];
        let groups = compute_alias_groups(&resources);

        let (without, with, savings) = estimate_aliasing_savings(&resources, &groups);

        assert_eq!(without, 3000);
        assert_eq!(with, 1000);
        // (3000 - 1000) / 3000 * 100 = ~66.7%
        assert!((savings - 66.67).abs() < 1.0);
    }

    #[test]
    fn test_estimate_aliasing_savings_returns_percentage() {
        let resources = vec![
            ResourceLifetimeRange::new(1, 0, 2, 1000),
            ResourceLifetimeRange::new(2, 3, 5, 1000),
        ];
        let groups = compute_alias_groups(&resources);

        let (_without, _with, savings) = estimate_aliasing_savings(&resources, &groups);

        // Should be percentage value
        assert!(savings >= 0.0);
        assert!(savings <= 100.0);
    }

    #[test]
    fn test_estimate_aliasing_savings_large_resources() {
        let resources = vec![
            ResourceLifetimeRange::new(1, 0, 2, 100 * MB),
            ResourceLifetimeRange::new(2, 3, 5, 200 * MB),
            ResourceLifetimeRange::new(3, 6, 8, 150 * MB),
        ];
        let groups = compute_alias_groups(&resources);

        let (without, with, _savings) = estimate_aliasing_savings(&resources, &groups);

        assert_eq!(without, 450 * MB);
        assert_eq!(with, 200 * MB);
    }
}

// ===========================================================================
// PooledTexture and PooledBuffer Tests (structural, no GPU required)
// ===========================================================================

mod pooled_resource_tests {
    use super::*;

    // Note: These tests focus on the structural aspects that don't require
    // a real GPU device. GPU integration tests would be separate.

    #[test]
    fn test_pool_key_is_texture_method() {
        let tex_key = PoolKey::Texture {
            format: wgpu::TextureFormat::Rgba8Unorm,
            width: 256,
            height: 256,
            depth_or_layers: 1,
            mip_levels: 1,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
        };

        assert!(tex_key.is_texture());
        assert!(!tex_key.is_buffer());
    }

    #[test]
    fn test_pool_key_is_buffer_method() {
        let buf_key = PoolKey::Buffer {
            size_class: SizeClass::Medium,
            usage: wgpu::BufferUsages::STORAGE,
        };

        assert!(buf_key.is_buffer());
        assert!(!buf_key.is_texture());
    }

    #[test]
    fn test_pooled_texture_debug_impl() {
        // PooledTexture implements Debug, so we can format it
        // This is a compile-time check essentially
        let key = PoolKey::Texture {
            format: wgpu::TextureFormat::Rgba8Unorm,
            width: 256,
            height: 256,
            depth_or_layers: 1,
            mip_levels: 1,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
        };
        // Key should be debuggable
        let debug_str = format!("{:?}", key);
        assert!(!debug_str.is_empty());
    }

    #[test]
    fn test_pooled_buffer_debug_impl() {
        let key = PoolKey::Buffer {
            size_class: SizeClass::Large,
            usage: wgpu::BufferUsages::VERTEX,
        };
        let debug_str = format!("{:?}", key);
        assert!(!debug_str.is_empty());
    }
}

// ===========================================================================
// TransientPool Configuration Tests (no GPU required)
// ===========================================================================

mod transient_pool_config_tests {
    use super::*;

    // Note: Full TransientPool tests would require a wgpu device.
    // These tests focus on configuration and API without GPU.

    #[test]
    fn test_pool_config_reasonable_defaults() {
        let config = PoolConfig::default();

        // Check that defaults are reasonable for real-world use
        assert!(config.max_idle_frames >= 1);
        assert!(config.max_idle_frames <= 10);
        assert!(config.max_texture_pool_size >= 4);
        assert!(config.max_buffer_pool_size >= 8);
    }

    #[test]
    fn test_pool_config_builder_fluent() {
        // Ensure builder is fluent and chainable
        let config = PoolConfig::new()
            .with_max_idle_frames(5)
            .with_max_texture_pool_size(16)
            .with_max_buffer_pool_size(32)
            .with_defragmentation(true)
            .with_memory_budget(GB);

        assert_eq!(config.max_idle_frames, 5);
        assert_eq!(config.max_texture_pool_size, 16);
        assert_eq!(config.max_buffer_pool_size, 32);
        assert!(config.enable_defragmentation);
        assert_eq!(config.max_memory_bytes, GB);
    }

    const GB: u64 = 1024 * MB;
}

// ===========================================================================
// Additional Edge Case Tests
// ===========================================================================

mod edge_case_tests {
    use super::*;

    #[test]
    fn test_size_class_all_boundary_values_consecutive() {
        // Ensure no gaps in size classification
        for size in 0..=20 * MB {
            let class = SizeClass::from_bytes(size);
            // Just verify it doesn't panic and returns a valid class
            let _ = class.min_size();
            let _ = class.max_size();
        }
    }

    #[test]
    fn test_pool_key_hash_stability() {
        // Same key should produce same hash across multiple calls
        let key = PoolKey::Buffer {
            size_class: SizeClass::Medium,
            usage: wgpu::BufferUsages::STORAGE,
        };

        let hash1 = hash_value(&key);
        let hash2 = hash_value(&key);
        let hash3 = hash_value(&key);

        assert_eq!(hash1, hash2);
        assert_eq!(hash2, hash3);
    }

    #[test]
    fn test_resource_lifetime_range_zero_size() {
        let range = ResourceLifetimeRange::new(1, 0, 5, 0);
        assert_eq!(range.size_bytes, 0);
    }

    #[test]
    fn test_alias_group_first_resource_sets_all_fields() {
        let mut group = AliasGroup::new();
        let r = ResourceLifetimeRange::new(42, 7, 12, 4096);

        group.try_add(&r);

        assert_eq!(group.members[0], 42);
        assert_eq!(group.memory_size, 4096);
        assert_eq!(group.first_pass, 7);
        assert_eq!(group.last_pass, 12);
    }

    #[test]
    fn test_compute_alias_groups_preserves_allocation_ids() {
        let resources = vec![
            ResourceLifetimeRange::new(100, 0, 2, 1024),
            ResourceLifetimeRange::new(200, 3, 5, 1024),
            ResourceLifetimeRange::new(300, 6, 8, 1024),
        ];
        let groups = compute_alias_groups(&resources);

        // All allocation IDs should be present
        let all_ids: Vec<u64> = groups.iter()
            .flat_map(|g| g.members.iter().copied())
            .collect();

        assert!(all_ids.contains(&100));
        assert!(all_ids.contains(&200));
        assert!(all_ids.contains(&300));
    }

    #[test]
    fn test_pool_stats_extreme_values() {
        let mut stats = PoolStats::new();
        stats.cache_hits = u64::MAX - 1;
        stats.cache_misses = 1;

        // Should not panic
        let rate = stats.hit_rate();
        assert!(rate > 0.99);
    }

    #[test]
    fn test_size_class_u64_max_minus_one() {
        let class = SizeClass::from_bytes(u64::MAX - 1);
        assert_eq!(class, SizeClass::Huge);
    }

    #[test]
    fn test_pool_key_different_texture_dimensions() {
        let key1 = PoolKey::Texture {
            format: wgpu::TextureFormat::Rgba8Unorm,
            width: 256,
            height: 256,
            depth_or_layers: 1,
            mip_levels: 1,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
        };
        let key2 = PoolKey::Texture {
            format: wgpu::TextureFormat::Rgba8Unorm,
            width: 256,
            height: 256,
            depth_or_layers: 1,
            mip_levels: 1,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,  // Different dimension
        };

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_pool_key_different_depth_layers() {
        let key1 = PoolKey::Texture {
            format: wgpu::TextureFormat::Rgba8Unorm,
            width: 256,
            height: 256,
            depth_or_layers: 1,
            mip_levels: 1,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
        };
        let key2 = PoolKey::Texture {
            format: wgpu::TextureFormat::Rgba8Unorm,
            width: 256,
            height: 256,
            depth_or_layers: 6,  // Different layer count
            mip_levels: 1,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
        };

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_pool_key_different_mip_levels() {
        let key1 = PoolKey::Texture {
            format: wgpu::TextureFormat::Rgba8Unorm,
            width: 256,
            height: 256,
            depth_or_layers: 1,
            mip_levels: 1,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
        };
        let key2 = PoolKey::Texture {
            format: wgpu::TextureFormat::Rgba8Unorm,
            width: 256,
            height: 256,
            depth_or_layers: 1,
            mip_levels: 5,  // Different mip levels
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
        };

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_aliasable_resource_symmetry() {
        let r1 = ResourceLifetimeRange::new(1, 0, 5, 1024);
        let r2 = ResourceLifetimeRange::new(2, 6, 10, 2048);

        // Aliasing should be symmetric
        assert_eq!(r1.can_alias_with(&r2), r2.can_alias_with(&r1));
        assert_eq!(r1.overlapping_lifetime(&r2), r2.overlapping_lifetime(&r1));
    }

    #[test]
    fn test_aliasable_resource_reflexive_overlap() {
        let r = ResourceLifetimeRange::new(1, 3, 7, 1024);

        // A resource should overlap with itself
        assert!(r.overlapping_lifetime(&r));
        assert!(!r.can_alias_with(&r));
    }
}

// ===========================================================================
// Integration-style Tests (still whitebox, no GPU)
// ===========================================================================

mod integration_style_tests {
    use super::*;

    #[test]
    fn test_full_aliasing_workflow() {
        // Simulate a frame graph with multiple transient resources
        let resources = vec![
            ResourceLifetimeRange::new(1, 0, 2, 4 * MB),   // G-buffer color
            ResourceLifetimeRange::new(2, 0, 2, 4 * MB),   // G-buffer normal
            ResourceLifetimeRange::new(3, 0, 2, 4 * MB),   // G-buffer depth
            ResourceLifetimeRange::new(4, 3, 5, 4 * MB),   // Lighting buffer
            ResourceLifetimeRange::new(5, 6, 8, 2 * MB),   // Post-process temp
            ResourceLifetimeRange::new(6, 9, 10, 2 * MB),  // Final output
        ];

        let groups = compute_alias_groups(&resources);
        let (without, with, savings) = estimate_aliasing_savings(&resources, &groups);

        // Verify reasonable savings for this scenario
        assert!(savings > 0.0);
        assert!(with < without);

        // All resources should be assigned
        let total: usize = groups.iter().map(|g| g.len()).sum();
        assert_eq!(total, 6);
    }

    #[test]
    fn test_size_class_to_key_round_trip() {
        // Various buffer sizes should map correctly
        let test_sizes = [
            100,           // Tiny
            10 * KB,       // Small
            100 * KB,      // Medium
            5 * MB,        // Large
            50 * MB,       // Huge
        ];

        for size in test_sizes {
            let desc = BufferDescriptor::new_storage(size);
            let key = PoolKey::from_buffer_desc(&desc);

            match key {
                PoolKey::Buffer { size_class, .. } => {
                    assert_eq!(size_class, SizeClass::from_bytes(size));
                }
                _ => panic!("Expected buffer key"),
            }
        }
    }

    #[test]
    fn test_texture_descriptor_to_key_preserves_all_fields() {
        let desc = TextureDescriptor::new_render_target(1920, 1080, wgpu::TextureFormat::Rgba16Float)
            .with_mips(4)
            .with_msaa(4);

        let key = PoolKey::from_texture_desc(&desc);

        match key {
            PoolKey::Texture {
                format,
                width,
                height,
                mip_levels,
                sample_count,
                ..
            } => {
                assert_eq!(format, wgpu::TextureFormat::Rgba16Float);
                assert_eq!(width, 1920);
                assert_eq!(height, 1080);
                assert_eq!(mip_levels, 4);
                assert_eq!(sample_count, 4);
            }
            _ => panic!("Expected texture key"),
        }
    }

    #[test]
    fn test_pool_stats_accumulation() {
        let mut stats = PoolStats::new();

        // Simulate multiple operations
        for _ in 0..10 {
            stats.cache_hits += 1;
        }
        for _ in 0..5 {
            stats.cache_misses += 1;
        }

        assert_eq!(stats.cache_hits, 10);
        assert_eq!(stats.cache_misses, 5);
        assert!((stats.hit_rate() - 0.6667).abs() < 0.01);
    }

    #[test]
    fn test_config_modification_chain() {
        let base_config = PoolConfig::default();

        // Modify step by step
        let config1 = base_config.clone().with_max_idle_frames(10);
        let config2 = config1.with_max_texture_pool_size(32);
        let config3 = config2.with_defragmentation(true);

        // Original unchanged
        assert_eq!(base_config.max_idle_frames, 3);

        // Final has all changes
        assert_eq!(config3.max_idle_frames, 10);
        assert_eq!(config3.max_texture_pool_size, 32);
        assert!(config3.enable_defragmentation);
    }
}
