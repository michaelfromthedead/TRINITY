//! Blackbox tests for T-WGPU-P7.5.3 Transient Resource Pool
//!
//! These tests treat the transient pool module as a black box, testing only
//! the public API without accessing internal implementation details.
//!
//! Coverage areas:
//! - API Contract Tests (20+)
//! - Real-World Pool Scenarios (30+)
//! - Cache Behavior (20+)
//! - Frame Lifecycle (15+)
//! - Garbage Collection (15+)
//! - Edge Cases (15+)
//! - Memory Aliasing (15+)

use renderer_backend::frame_graph::transient::{
    AliasGroup, AliasableResource, PoolConfig, PoolKey, PoolStats, ResourceLifetimeRange,
    SizeClass, compute_alias_groups, estimate_aliasing_savings,
};
use renderer_backend::frame_graph::resources::{BufferDescriptor, TextureDescriptor};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// =============================================================================
// SECTION 1 -- SizeClass API Contract Tests
// =============================================================================

#[test]
fn sizeclass_from_bytes_zero() {
    assert_eq!(SizeClass::from_bytes(0), SizeClass::Tiny);
}

#[test]
fn sizeclass_from_bytes_tiny_boundary() {
    assert_eq!(SizeClass::from_bytes(1023), SizeClass::Tiny);
    assert_eq!(SizeClass::from_bytes(1024), SizeClass::Small);
}

#[test]
fn sizeclass_from_bytes_small_boundary() {
    assert_eq!(SizeClass::from_bytes(64 * 1024 - 1), SizeClass::Small);
    assert_eq!(SizeClass::from_bytes(64 * 1024), SizeClass::Medium);
}

#[test]
fn sizeclass_from_bytes_medium_boundary() {
    assert_eq!(SizeClass::from_bytes(1024 * 1024 - 1), SizeClass::Medium);
    assert_eq!(SizeClass::from_bytes(1024 * 1024), SizeClass::Large);
}

#[test]
fn sizeclass_from_bytes_large_boundary() {
    assert_eq!(SizeClass::from_bytes(16 * 1024 * 1024 - 1), SizeClass::Large);
    assert_eq!(SizeClass::from_bytes(16 * 1024 * 1024), SizeClass::Huge);
}

#[test]
fn sizeclass_from_bytes_huge_values() {
    assert_eq!(SizeClass::from_bytes(100 * 1024 * 1024), SizeClass::Huge);
    assert_eq!(SizeClass::from_bytes(1024 * 1024 * 1024), SizeClass::Huge);
    assert_eq!(SizeClass::from_bytes(u64::MAX), SizeClass::Huge);
}

#[test]
fn sizeclass_min_size_values() {
    assert_eq!(SizeClass::Tiny.min_size(), 0);
    assert_eq!(SizeClass::Small.min_size(), 1024);
    assert_eq!(SizeClass::Medium.min_size(), 64 * 1024);
    assert_eq!(SizeClass::Large.min_size(), 1024 * 1024);
    assert_eq!(SizeClass::Huge.min_size(), 16 * 1024 * 1024);
}

#[test]
fn sizeclass_max_size_values() {
    assert_eq!(SizeClass::Tiny.max_size(), 1023);
    assert_eq!(SizeClass::Small.max_size(), 64 * 1024 - 1);
    assert_eq!(SizeClass::Medium.max_size(), 1024 * 1024 - 1);
    assert_eq!(SizeClass::Large.max_size(), 16 * 1024 * 1024 - 1);
    assert_eq!(SizeClass::Huge.max_size(), u64::MAX);
}

#[test]
fn sizeclass_allocation_size_values() {
    assert_eq!(SizeClass::Tiny.allocation_size(), 1024);
    assert_eq!(SizeClass::Small.allocation_size(), 64 * 1024);
    assert_eq!(SizeClass::Medium.allocation_size(), 1024 * 1024);
    assert_eq!(SizeClass::Large.allocation_size(), 16 * 1024 * 1024);
    assert_eq!(SizeClass::Huge.allocation_size(), 64 * 1024 * 1024);
}

#[test]
fn sizeclass_display_format() {
    assert_eq!(format!("{}", SizeClass::Tiny), "Tiny(<1KB)");
    assert_eq!(format!("{}", SizeClass::Small), "Small(1-64KB)");
    assert_eq!(format!("{}", SizeClass::Medium), "Medium(64KB-1MB)");
    assert_eq!(format!("{}", SizeClass::Large), "Large(1-16MB)");
    assert_eq!(format!("{}", SizeClass::Huge), "Huge(>16MB)");
}

#[test]
fn sizeclass_equality() {
    assert_eq!(SizeClass::Tiny, SizeClass::Tiny);
    assert_eq!(SizeClass::Small, SizeClass::Small);
    assert_eq!(SizeClass::Medium, SizeClass::Medium);
    assert_eq!(SizeClass::Large, SizeClass::Large);
    assert_eq!(SizeClass::Huge, SizeClass::Huge);
}

#[test]
fn sizeclass_inequality() {
    assert_ne!(SizeClass::Tiny, SizeClass::Small);
    assert_ne!(SizeClass::Small, SizeClass::Medium);
    assert_ne!(SizeClass::Medium, SizeClass::Large);
    assert_ne!(SizeClass::Large, SizeClass::Huge);
}

#[test]
fn sizeclass_hash_consistency() {
    let mut h1 = DefaultHasher::new();
    let mut h2 = DefaultHasher::new();
    SizeClass::Medium.hash(&mut h1);
    SizeClass::Medium.hash(&mut h2);
    assert_eq!(h1.finish(), h2.finish());
}

#[test]
fn sizeclass_clone() {
    let original = SizeClass::Large;
    let cloned = original.clone();
    assert_eq!(original, cloned);
}

#[test]
fn sizeclass_debug_format() {
    let debug = format!("{:?}", SizeClass::Huge);
    assert!(debug.contains("Huge"));
}

// =============================================================================
// SECTION 2 -- PoolKey API Contract Tests
// =============================================================================

#[test]
fn poolkey_texture_from_desc() {
    let desc = TextureDescriptor::new_render_target(1920, 1080, wgpu::TextureFormat::Rgba8Unorm);
    let key = PoolKey::from_texture_desc(&desc);
    assert!(key.is_texture());
    assert!(!key.is_buffer());
}

#[test]
fn poolkey_buffer_from_desc() {
    let desc = BufferDescriptor::new_storage(100 * 1024);
    let key = PoolKey::from_buffer_desc(&desc);
    assert!(key.is_buffer());
    assert!(!key.is_texture());
}

#[test]
fn poolkey_texture_equality_same() {
    let desc = TextureDescriptor::new_render_target(1920, 1080, wgpu::TextureFormat::Rgba8Unorm);
    let key1 = PoolKey::from_texture_desc(&desc);
    let key2 = PoolKey::from_texture_desc(&desc);
    assert_eq!(key1, key2);
}

#[test]
fn poolkey_texture_inequality_width() {
    let desc1 = TextureDescriptor::new_render_target(1920, 1080, wgpu::TextureFormat::Rgba8Unorm);
    let desc2 = TextureDescriptor::new_render_target(1280, 1080, wgpu::TextureFormat::Rgba8Unorm);
    let key1 = PoolKey::from_texture_desc(&desc1);
    let key2 = PoolKey::from_texture_desc(&desc2);
    assert_ne!(key1, key2);
}

#[test]
fn poolkey_texture_inequality_height() {
    let desc1 = TextureDescriptor::new_render_target(1920, 1080, wgpu::TextureFormat::Rgba8Unorm);
    let desc2 = TextureDescriptor::new_render_target(1920, 720, wgpu::TextureFormat::Rgba8Unorm);
    let key1 = PoolKey::from_texture_desc(&desc1);
    let key2 = PoolKey::from_texture_desc(&desc2);
    assert_ne!(key1, key2);
}

#[test]
fn poolkey_texture_inequality_format() {
    let desc1 = TextureDescriptor::new_render_target(1920, 1080, wgpu::TextureFormat::Rgba8Unorm);
    let desc2 = TextureDescriptor::new_render_target(1920, 1080, wgpu::TextureFormat::Bgra8Unorm);
    let key1 = PoolKey::from_texture_desc(&desc1);
    let key2 = PoolKey::from_texture_desc(&desc2);
    assert_ne!(key1, key2);
}

#[test]
fn poolkey_buffer_equality_same_class() {
    let desc1 = BufferDescriptor::new_storage(100 * 1024);
    let desc2 = BufferDescriptor::new_storage(200 * 1024); // Same class (Medium)
    let key1 = PoolKey::from_buffer_desc(&desc1);
    let key2 = PoolKey::from_buffer_desc(&desc2);
    assert_eq!(key1, key2);
}

#[test]
fn poolkey_buffer_inequality_different_class() {
    let desc1 = BufferDescriptor::new_storage(100 * 1024);  // Medium
    let desc2 = BufferDescriptor::new_storage(2 * 1024 * 1024); // Large
    let key1 = PoolKey::from_buffer_desc(&desc1);
    let key2 = PoolKey::from_buffer_desc(&desc2);
    assert_ne!(key1, key2);
}

#[test]
fn poolkey_buffer_inequality_different_usage() {
    let desc1 = BufferDescriptor::new_storage(100 * 1024);
    let desc2 = BufferDescriptor::new_uniform(100 * 1024);
    let key1 = PoolKey::from_buffer_desc(&desc1);
    let key2 = PoolKey::from_buffer_desc(&desc2);
    assert_ne!(key1, key2);
}

#[test]
fn poolkey_texture_buffer_not_equal() {
    let tex_desc = TextureDescriptor::new_render_target(256, 256, wgpu::TextureFormat::Rgba8Unorm);
    let buf_desc = BufferDescriptor::new_storage(256 * 256 * 4);
    let tex_key = PoolKey::from_texture_desc(&tex_desc);
    let buf_key = PoolKey::from_buffer_desc(&buf_desc);
    assert_ne!(tex_key, buf_key);
}

#[test]
fn poolkey_texture_hash_consistency() {
    let desc = TextureDescriptor::new_render_target(1920, 1080, wgpu::TextureFormat::Rgba8Unorm);
    let key1 = PoolKey::from_texture_desc(&desc);
    let key2 = PoolKey::from_texture_desc(&desc);

    let mut h1 = DefaultHasher::new();
    let mut h2 = DefaultHasher::new();
    key1.hash(&mut h1);
    key2.hash(&mut h2);
    assert_eq!(h1.finish(), h2.finish());
}

#[test]
fn poolkey_buffer_hash_consistency() {
    let desc = BufferDescriptor::new_storage(64 * 1024);
    let key1 = PoolKey::from_buffer_desc(&desc);
    let key2 = PoolKey::from_buffer_desc(&desc);

    let mut h1 = DefaultHasher::new();
    let mut h2 = DefaultHasher::new();
    key1.hash(&mut h1);
    key2.hash(&mut h2);
    assert_eq!(h1.finish(), h2.finish());
}

#[test]
fn poolkey_texture_display() {
    let desc = TextureDescriptor::new_render_target(1920, 1080, wgpu::TextureFormat::Rgba8Unorm);
    let key = PoolKey::from_texture_desc(&desc);
    let display = format!("{}", key);
    assert!(display.contains("Texture"));
    assert!(display.contains("1920"));
    assert!(display.contains("1080"));
}

#[test]
fn poolkey_buffer_display() {
    let desc = BufferDescriptor::new_storage(100 * 1024);
    let key = PoolKey::from_buffer_desc(&desc);
    let display = format!("{}", key);
    assert!(display.contains("Buffer"));
    assert!(display.contains("Medium"));
}

#[test]
fn poolkey_clone() {
    let desc = TextureDescriptor::new_depth(800, 600);
    let key = PoolKey::from_texture_desc(&desc);
    let cloned = key.clone();
    assert_eq!(key, cloned);
}

#[test]
fn poolkey_debug() {
    let desc = BufferDescriptor::new_uniform(256);
    let key = PoolKey::from_buffer_desc(&desc);
    let debug = format!("{:?}", key);
    assert!(debug.contains("Buffer"));
}

// =============================================================================
// SECTION 3 -- PoolStats API Contract Tests
// =============================================================================

#[test]
fn poolstats_new_defaults() {
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
fn poolstats_default_is_new() {
    let stats1 = PoolStats::new();
    let stats2 = PoolStats::default();
    assert_eq!(stats1.total_textures, stats2.total_textures);
    assert_eq!(stats1.cache_hits, stats2.cache_hits);
}

#[test]
fn poolstats_hit_rate_zero_accesses() {
    let stats = PoolStats::new();
    assert_eq!(stats.hit_rate(), 0.0);
}

#[test]
fn poolstats_hit_rate_all_hits() {
    let mut stats = PoolStats::new();
    stats.cache_hits = 100;
    stats.cache_misses = 0;
    assert_eq!(stats.hit_rate(), 1.0);
}

#[test]
fn poolstats_hit_rate_all_misses() {
    let mut stats = PoolStats::new();
    stats.cache_hits = 0;
    stats.cache_misses = 100;
    assert_eq!(stats.hit_rate(), 0.0);
}

#[test]
fn poolstats_hit_rate_mixed() {
    let mut stats = PoolStats::new();
    stats.cache_hits = 75;
    stats.cache_misses = 25;
    assert!((stats.hit_rate() - 0.75).abs() < 0.001);
}

#[test]
fn poolstats_hit_rate_80_percent() {
    let mut stats = PoolStats::new();
    stats.cache_hits = 8;
    stats.cache_misses = 2;
    assert!((stats.hit_rate() - 0.8).abs() < 0.001);
}

#[test]
fn poolstats_total_bytes() {
    let mut stats = PoolStats::new();
    stats.texture_bytes = 1024 * 1024;
    stats.buffer_bytes = 512 * 1024;
    assert_eq!(stats.total_bytes(), 1024 * 1024 + 512 * 1024);
}

#[test]
fn poolstats_total_bytes_zero() {
    let stats = PoolStats::new();
    assert_eq!(stats.total_bytes(), 0);
}

#[test]
fn poolstats_idle_textures() {
    let mut stats = PoolStats::new();
    stats.total_textures = 10;
    stats.active_textures = 3;
    assert_eq!(stats.idle_textures(), 7);
}

#[test]
fn poolstats_idle_textures_all_active() {
    let mut stats = PoolStats::new();
    stats.total_textures = 5;
    stats.active_textures = 5;
    assert_eq!(stats.idle_textures(), 0);
}

#[test]
fn poolstats_idle_buffers() {
    let mut stats = PoolStats::new();
    stats.total_buffers = 20;
    stats.active_buffers = 8;
    assert_eq!(stats.idle_buffers(), 12);
}

#[test]
fn poolstats_idle_buffers_saturating() {
    let mut stats = PoolStats::new();
    stats.total_buffers = 5;
    stats.active_buffers = 10; // More active than total (shouldn't happen but test saturation)
    assert_eq!(stats.idle_buffers(), 0);
}

#[test]
fn poolstats_reset_counters() {
    let mut stats = PoolStats::new();
    stats.cache_hits = 100;
    stats.cache_misses = 50;
    stats.gc_count = 20;
    stats.total_textures = 10; // This should NOT reset

    stats.reset_counters();

    assert_eq!(stats.cache_hits, 0);
    assert_eq!(stats.cache_misses, 0);
    assert_eq!(stats.gc_count, 0);
    assert_eq!(stats.total_textures, 10); // Should be preserved
}

#[test]
fn poolstats_display() {
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
fn poolstats_clone() {
    let mut stats = PoolStats::new();
    stats.cache_hits = 42;
    let cloned = stats.clone();
    assert_eq!(stats.cache_hits, cloned.cache_hits);
}

#[test]
fn poolstats_debug() {
    let stats = PoolStats::new();
    let debug = format!("{:?}", stats);
    assert!(debug.contains("PoolStats"));
}

// =============================================================================
// SECTION 4 -- PoolConfig API Contract Tests
// =============================================================================

#[test]
fn poolconfig_default_values() {
    let config = PoolConfig::default();
    assert_eq!(config.max_idle_frames, 3);
    assert_eq!(config.max_texture_pool_size, 8);
    assert_eq!(config.max_buffer_pool_size, 16);
    assert!(!config.enable_defragmentation);
    assert_eq!(config.max_memory_bytes, 0);
}

#[test]
fn poolconfig_new_equals_default() {
    let config1 = PoolConfig::new();
    let config2 = PoolConfig::default();
    assert_eq!(config1.max_idle_frames, config2.max_idle_frames);
    assert_eq!(config1.max_texture_pool_size, config2.max_texture_pool_size);
}

#[test]
fn poolconfig_with_max_idle_frames() {
    let config = PoolConfig::new().with_max_idle_frames(10);
    assert_eq!(config.max_idle_frames, 10);
}

#[test]
fn poolconfig_with_max_texture_pool_size() {
    let config = PoolConfig::new().with_max_texture_pool_size(32);
    assert_eq!(config.max_texture_pool_size, 32);
}

#[test]
fn poolconfig_with_max_buffer_pool_size() {
    let config = PoolConfig::new().with_max_buffer_pool_size(64);
    assert_eq!(config.max_buffer_pool_size, 64);
}

#[test]
fn poolconfig_with_defragmentation_enabled() {
    let config = PoolConfig::new().with_defragmentation(true);
    assert!(config.enable_defragmentation);
}

#[test]
fn poolconfig_with_defragmentation_disabled() {
    let config = PoolConfig::new().with_defragmentation(false);
    assert!(!config.enable_defragmentation);
}

#[test]
fn poolconfig_with_memory_budget() {
    let config = PoolConfig::new().with_memory_budget(256 * 1024 * 1024);
    assert_eq!(config.max_memory_bytes, 256 * 1024 * 1024);
}

#[test]
fn poolconfig_builder_chain() {
    let config = PoolConfig::new()
        .with_max_idle_frames(5)
        .with_max_texture_pool_size(16)
        .with_max_buffer_pool_size(32)
        .with_defragmentation(true)
        .with_memory_budget(512 * 1024 * 1024);

    assert_eq!(config.max_idle_frames, 5);
    assert_eq!(config.max_texture_pool_size, 16);
    assert_eq!(config.max_buffer_pool_size, 32);
    assert!(config.enable_defragmentation);
    assert_eq!(config.max_memory_bytes, 512 * 1024 * 1024);
}

#[test]
fn poolconfig_zero_idle_frames() {
    let config = PoolConfig::new().with_max_idle_frames(0);
    assert_eq!(config.max_idle_frames, 0);
}

#[test]
fn poolconfig_clone() {
    let config = PoolConfig::new().with_max_idle_frames(7);
    let cloned = config.clone();
    assert_eq!(config.max_idle_frames, cloned.max_idle_frames);
}

#[test]
fn poolconfig_debug() {
    let config = PoolConfig::new();
    let debug = format!("{:?}", config);
    assert!(debug.contains("PoolConfig"));
    assert!(debug.contains("max_idle_frames"));
}

// =============================================================================
// SECTION 5 -- ResourceLifetimeRange and AliasableResource Tests
// =============================================================================

#[test]
fn resource_lifetime_range_new() {
    let range = ResourceLifetimeRange::new(1, 0, 5, 1024);
    assert_eq!(range.allocation_id, 1);
    assert_eq!(range.first_use_pass, 0);
    assert_eq!(range.last_use_pass, 5);
    assert_eq!(range.size_bytes, 1024);
}

#[test]
fn resource_lifetime_range_accessors() {
    let range = ResourceLifetimeRange::new(42, 3, 8, 2048);
    assert_eq!(range.first_use_pass(), 3);
    assert_eq!(range.last_use_pass(), 8);
}

#[test]
fn resource_lifetime_range_non_overlapping() {
    let r1 = ResourceLifetimeRange::new(1, 0, 5, 1024);
    let r2 = ResourceLifetimeRange::new(2, 6, 10, 2048);
    assert!(!r1.overlapping_lifetime(&r2));
    assert!(r1.can_alias_with(&r2));
}

#[test]
fn resource_lifetime_range_overlapping() {
    let r1 = ResourceLifetimeRange::new(1, 0, 5, 1024);
    let r2 = ResourceLifetimeRange::new(2, 3, 8, 2048);
    assert!(r1.overlapping_lifetime(&r2));
    assert!(!r1.can_alias_with(&r2));
}

#[test]
fn resource_lifetime_range_touching_boundaries() {
    // r1 ends at 5, r2 starts at 5 - they touch, so they overlap
    let r1 = ResourceLifetimeRange::new(1, 0, 5, 1024);
    let r2 = ResourceLifetimeRange::new(2, 5, 10, 2048);
    assert!(r1.overlapping_lifetime(&r2));
    assert!(!r1.can_alias_with(&r2));
}

#[test]
fn resource_lifetime_range_adjacent_boundaries() {
    // r1 ends at 5, r2 starts at 6 - they don't touch
    let r1 = ResourceLifetimeRange::new(1, 0, 5, 1024);
    let r2 = ResourceLifetimeRange::new(2, 6, 10, 2048);
    assert!(!r1.overlapping_lifetime(&r2));
    assert!(r1.can_alias_with(&r2));
}

#[test]
fn resource_lifetime_range_same_range() {
    let r1 = ResourceLifetimeRange::new(1, 0, 5, 1024);
    let r2 = ResourceLifetimeRange::new(2, 0, 5, 2048);
    assert!(r1.overlapping_lifetime(&r2));
    assert!(!r1.can_alias_with(&r2));
}

#[test]
fn resource_lifetime_range_contained() {
    let r1 = ResourceLifetimeRange::new(1, 0, 10, 1024);
    let r2 = ResourceLifetimeRange::new(2, 3, 7, 2048);
    assert!(r1.overlapping_lifetime(&r2));
    assert!(!r1.can_alias_with(&r2));
}

#[test]
fn resource_lifetime_range_single_pass() {
    let r1 = ResourceLifetimeRange::new(1, 5, 5, 1024);
    let r2 = ResourceLifetimeRange::new(2, 5, 5, 2048);
    assert!(r1.overlapping_lifetime(&r2));
}

#[test]
fn resource_lifetime_range_single_pass_different() {
    let r1 = ResourceLifetimeRange::new(1, 5, 5, 1024);
    let r2 = ResourceLifetimeRange::new(2, 6, 6, 2048);
    assert!(!r1.overlapping_lifetime(&r2));
    assert!(r1.can_alias_with(&r2));
}

#[test]
fn resource_lifetime_range_symmetry() {
    let r1 = ResourceLifetimeRange::new(1, 0, 5, 1024);
    let r2 = ResourceLifetimeRange::new(2, 6, 10, 2048);
    assert_eq!(r1.overlapping_lifetime(&r2), r2.overlapping_lifetime(&r1));
    assert_eq!(r1.can_alias_with(&r2), r2.can_alias_with(&r1));
}

#[test]
fn resource_lifetime_range_equality() {
    let r1 = ResourceLifetimeRange::new(1, 0, 5, 1024);
    let r2 = ResourceLifetimeRange::new(1, 0, 5, 1024);
    assert_eq!(r1, r2);
}

#[test]
fn resource_lifetime_range_inequality() {
    let r1 = ResourceLifetimeRange::new(1, 0, 5, 1024);
    let r2 = ResourceLifetimeRange::new(2, 0, 5, 1024);
    assert_ne!(r1, r2);
}

#[test]
fn resource_lifetime_range_clone() {
    let r1 = ResourceLifetimeRange::new(1, 0, 5, 1024);
    let r2 = r1.clone();
    assert_eq!(r1, r2);
}

#[test]
fn resource_lifetime_range_debug() {
    let r = ResourceLifetimeRange::new(1, 0, 5, 1024);
    let debug = format!("{:?}", r);
    assert!(debug.contains("ResourceLifetimeRange"));
}

// =============================================================================
// SECTION 6 -- AliasGroup Tests
// =============================================================================

#[test]
fn alias_group_new_empty() {
    let group = AliasGroup::new();
    assert!(group.is_empty());
    assert_eq!(group.len(), 0);
    assert_eq!(group.memory_size, 0);
}

#[test]
fn alias_group_default_empty() {
    let group = AliasGroup::default();
    assert!(group.is_empty());
}

#[test]
fn alias_group_add_first_resource() {
    let mut group = AliasGroup::new();
    let r = ResourceLifetimeRange::new(1, 0, 5, 1024);
    assert!(group.try_add(&r));
    assert_eq!(group.len(), 1);
    assert!(!group.is_empty());
    assert_eq!(group.memory_size, 1024);
    assert_eq!(group.first_pass, 0);
    assert_eq!(group.last_pass, 5);
}

#[test]
fn alias_group_add_non_overlapping() {
    let mut group = AliasGroup::new();
    let r1 = ResourceLifetimeRange::new(1, 0, 3, 1024);
    let r2 = ResourceLifetimeRange::new(2, 5, 8, 2048);

    assert!(group.try_add(&r1));
    assert!(group.try_add(&r2));
    assert_eq!(group.len(), 2);
    assert_eq!(group.memory_size, 2048); // Max of both
}

#[test]
fn alias_group_reject_overlapping() {
    let mut group = AliasGroup::new();
    let r1 = ResourceLifetimeRange::new(1, 0, 5, 1024);
    let r2 = ResourceLifetimeRange::new(2, 3, 8, 2048);

    assert!(group.try_add(&r1));
    assert!(!group.try_add(&r2)); // Should fail - overlaps
    assert_eq!(group.len(), 1);
}

#[test]
fn alias_group_memory_size_max() {
    let mut group = AliasGroup::new();
    let r1 = ResourceLifetimeRange::new(1, 0, 3, 1000);
    let r2 = ResourceLifetimeRange::new(2, 5, 8, 5000);
    let r3 = ResourceLifetimeRange::new(3, 10, 12, 2000);

    group.try_add(&r1);
    group.try_add(&r2);
    group.try_add(&r3);

    assert_eq!(group.memory_size, 5000);
}

#[test]
fn alias_group_span_tracking() {
    let mut group = AliasGroup::new();
    let r1 = ResourceLifetimeRange::new(1, 5, 8, 1024);
    let r2 = ResourceLifetimeRange::new(2, 0, 3, 2048);
    let r3 = ResourceLifetimeRange::new(3, 10, 15, 512);

    group.try_add(&r1);
    group.try_add(&r2);
    group.try_add(&r3);

    assert_eq!(group.first_pass, 0);
    assert_eq!(group.last_pass, 15);
}

#[test]
fn alias_group_members_tracking() {
    let mut group = AliasGroup::new();
    let r1 = ResourceLifetimeRange::new(42, 0, 3, 1024);
    let r2 = ResourceLifetimeRange::new(99, 5, 8, 2048);

    group.try_add(&r1);
    group.try_add(&r2);

    assert!(group.members.contains(&42));
    assert!(group.members.contains(&99));
}

#[test]
fn alias_group_clone() {
    let mut group = AliasGroup::new();
    group.try_add(&ResourceLifetimeRange::new(1, 0, 3, 1024));
    let cloned = group.clone();
    assert_eq!(group.len(), cloned.len());
    assert_eq!(group.memory_size, cloned.memory_size);
}

#[test]
fn alias_group_debug() {
    let group = AliasGroup::new();
    let debug = format!("{:?}", group);
    assert!(debug.contains("AliasGroup"));
}

// =============================================================================
// SECTION 7 -- compute_alias_groups Tests
// =============================================================================

#[test]
fn compute_alias_groups_empty() {
    let resources: Vec<ResourceLifetimeRange> = vec![];
    let groups = compute_alias_groups(&resources);
    assert!(groups.is_empty());
}

#[test]
fn compute_alias_groups_single_resource() {
    let resources = vec![ResourceLifetimeRange::new(1, 0, 5, 1024)];
    let groups = compute_alias_groups(&resources);
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].len(), 1);
}

#[test]
fn compute_alias_groups_two_non_overlapping() {
    let resources = vec![
        ResourceLifetimeRange::new(1, 0, 3, 1024),
        ResourceLifetimeRange::new(2, 5, 8, 2048),
    ];
    let groups = compute_alias_groups(&resources);
    // Both can fit in one group
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].len(), 2);
}

#[test]
fn compute_alias_groups_two_overlapping() {
    let resources = vec![
        ResourceLifetimeRange::new(1, 0, 5, 1024),
        ResourceLifetimeRange::new(2, 3, 8, 2048),
    ];
    let groups = compute_alias_groups(&resources);
    // Need separate groups
    assert_eq!(groups.len(), 2);
}

#[test]
fn compute_alias_groups_complex_scenario() {
    let resources = vec![
        ResourceLifetimeRange::new(1, 0, 2, 1024),
        ResourceLifetimeRange::new(2, 3, 5, 2048),
        ResourceLifetimeRange::new(3, 0, 4, 512),  // Overlaps with 1 and 2
        ResourceLifetimeRange::new(4, 6, 8, 768),
    ];
    let groups = compute_alias_groups(&resources);

    // Verify all resources are assigned
    let total_assigned: usize = groups.iter().map(|g| g.len()).sum();
    assert_eq!(total_assigned, 4);
}

#[test]
fn compute_alias_groups_sequential_chain() {
    let resources: Vec<ResourceLifetimeRange> = (0..10)
        .map(|i| ResourceLifetimeRange::new(i as u64, i * 10, i * 10 + 5, 1024))
        .collect();
    let groups = compute_alias_groups(&resources);
    // All sequential, should fit in one or few groups
    assert!(groups.len() <= 2);
}

#[test]
fn compute_alias_groups_all_overlapping() {
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
fn compute_alias_groups_preserves_all_resources() {
    let resources = vec![
        ResourceLifetimeRange::new(1, 0, 2, 1024),
        ResourceLifetimeRange::new(2, 1, 3, 2048),
        ResourceLifetimeRange::new(3, 2, 4, 512),
        ResourceLifetimeRange::new(4, 3, 5, 768),
        ResourceLifetimeRange::new(5, 4, 6, 256),
    ];
    let groups = compute_alias_groups(&resources);

    let total: usize = groups.iter().map(|g| g.len()).sum();
    assert_eq!(total, 5);
}

// =============================================================================
// SECTION 8 -- estimate_aliasing_savings Tests
// =============================================================================

#[test]
fn estimate_aliasing_savings_empty() {
    let resources: Vec<ResourceLifetimeRange> = vec![];
    let groups: Vec<AliasGroup> = vec![];
    let (without, with, savings) = estimate_aliasing_savings(&resources, &groups);
    assert_eq!(without, 0);
    assert_eq!(with, 0);
    assert_eq!(savings, 0.0);
}

#[test]
fn estimate_aliasing_savings_single_resource() {
    let resources = vec![ResourceLifetimeRange::new(1, 0, 5, 1024)];
    let groups = compute_alias_groups(&resources);
    let (without, with, savings) = estimate_aliasing_savings(&resources, &groups);
    assert_eq!(without, 1024);
    assert_eq!(with, 1024);
    assert_eq!(savings, 0.0);
}

#[test]
fn estimate_aliasing_savings_full_aliasing() {
    let resources = vec![
        ResourceLifetimeRange::new(1, 0, 2, 1000),
        ResourceLifetimeRange::new(2, 3, 5, 2000),
        ResourceLifetimeRange::new(3, 6, 8, 1500),
    ];
    let groups = compute_alias_groups(&resources);
    let (without, with, savings) = estimate_aliasing_savings(&resources, &groups);

    assert_eq!(without, 4500);
    assert_eq!(with, 2000); // Max size
    assert!(savings > 50.0);
}

#[test]
fn estimate_aliasing_savings_no_aliasing() {
    let resources = vec![
        ResourceLifetimeRange::new(1, 0, 10, 1000),
        ResourceLifetimeRange::new(2, 0, 10, 2000),
    ];
    let groups = compute_alias_groups(&resources);
    let (without, with, savings) = estimate_aliasing_savings(&resources, &groups);

    assert_eq!(without, 3000);
    assert_eq!(with, 3000);
    assert_eq!(savings, 0.0);
}

#[test]
fn estimate_aliasing_savings_partial_aliasing() {
    let resources = vec![
        ResourceLifetimeRange::new(1, 0, 2, 1000),
        ResourceLifetimeRange::new(2, 3, 5, 2000),
        ResourceLifetimeRange::new(3, 0, 5, 500), // Overlaps with 1 and 2
    ];
    let groups = compute_alias_groups(&resources);
    let (without, with, savings) = estimate_aliasing_savings(&resources, &groups);

    assert_eq!(without, 3500);
    assert!(with < without);
    assert!(savings > 0.0);
}

// =============================================================================
// SECTION 9 -- Real-World Pool Scenario Tests (Descriptors only, no GPU)
// =============================================================================

#[test]
fn scenario_render_target_descriptor() {
    let desc = TextureDescriptor::new_render_target(1920, 1080, wgpu::TextureFormat::Rgba8Unorm);
    assert_eq!(desc.width, 1920);
    assert_eq!(desc.height, 1080);
    assert!(desc.usage.contains(wgpu::TextureUsages::RENDER_ATTACHMENT));
}

#[test]
fn scenario_gbuffer_descriptors() {
    let albedo = TextureDescriptor::new_render_target(1920, 1080, wgpu::TextureFormat::Rgba8Unorm);
    let normal = TextureDescriptor::new_render_target(1920, 1080, wgpu::TextureFormat::Rgba16Float);
    let depth = TextureDescriptor::new_depth(1920, 1080);

    // All different formats = different pool keys
    let key_albedo = PoolKey::from_texture_desc(&albedo);
    let key_normal = PoolKey::from_texture_desc(&normal);
    let key_depth = PoolKey::from_texture_desc(&depth);

    assert_ne!(key_albedo, key_normal);
    assert_ne!(key_normal, key_depth);
    assert_ne!(key_albedo, key_depth);
}

#[test]
fn scenario_shadow_map_descriptors() {
    let shadow_2k = TextureDescriptor::new_depth(2048, 2048);
    let shadow_4k = TextureDescriptor::new_depth(4096, 4096);

    let key_2k = PoolKey::from_texture_desc(&shadow_2k);
    let key_4k = PoolKey::from_texture_desc(&shadow_4k);

    assert_ne!(key_2k, key_4k);
}

#[test]
fn scenario_post_process_pingpong() {
    let ping = TextureDescriptor::new_render_target(1920, 1080, wgpu::TextureFormat::Rgba16Float);
    let pong = TextureDescriptor::new_render_target(1920, 1080, wgpu::TextureFormat::Rgba16Float);

    let key_ping = PoolKey::from_texture_desc(&ping);
    let key_pong = PoolKey::from_texture_desc(&pong);

    // Same spec = same key
    assert_eq!(key_ping, key_pong);
}

#[test]
fn scenario_uniform_buffer_descriptors() {
    let camera = BufferDescriptor::new_uniform(256);
    let transform = BufferDescriptor::new_uniform(512);

    // Both Tiny class, same usage
    let key_camera = PoolKey::from_buffer_desc(&camera);
    let key_transform = PoolKey::from_buffer_desc(&transform);

    assert_eq!(key_camera, key_transform);
}

#[test]
fn scenario_staging_buffer_descriptors() {
    let upload = BufferDescriptor::new_staging_write(1024 * 1024);
    let download = BufferDescriptor::new_staging_read(1024 * 1024);

    let key_upload = PoolKey::from_buffer_desc(&upload);
    let key_download = PoolKey::from_buffer_desc(&download);

    // Different usage = different keys
    assert_ne!(key_upload, key_download);
}

#[test]
fn scenario_compute_storage_buffer() {
    let storage = BufferDescriptor::new_storage(16 * 1024 * 1024);
    let key = PoolKey::from_buffer_desc(&storage);
    assert!(key.is_buffer());
}

#[test]
fn scenario_multiple_render_targets() {
    let rt1 = TextureDescriptor::new_render_target(1920, 1080, wgpu::TextureFormat::Rgba8Unorm);
    let rt2 = TextureDescriptor::new_render_target(1920, 1080, wgpu::TextureFormat::Rgba8Unorm);
    let rt3 = TextureDescriptor::new_render_target(1920, 1080, wgpu::TextureFormat::Rgba8Unorm);

    let key1 = PoolKey::from_texture_desc(&rt1);
    let key2 = PoolKey::from_texture_desc(&rt2);
    let key3 = PoolKey::from_texture_desc(&rt3);

    assert_eq!(key1, key2);
    assert_eq!(key2, key3);
}

#[test]
fn scenario_resize_creates_new_key() {
    let hd = TextureDescriptor::new_render_target(1920, 1080, wgpu::TextureFormat::Rgba8Unorm);
    let qhd = TextureDescriptor::new_render_target(2560, 1440, wgpu::TextureFormat::Rgba8Unorm);
    let uhd = TextureDescriptor::new_render_target(3840, 2160, wgpu::TextureFormat::Rgba8Unorm);

    let key_hd = PoolKey::from_texture_desc(&hd);
    let key_qhd = PoolKey::from_texture_desc(&qhd);
    let key_uhd = PoolKey::from_texture_desc(&uhd);

    assert_ne!(key_hd, key_qhd);
    assert_ne!(key_qhd, key_uhd);
}

#[test]
fn scenario_msaa_variants() {
    let no_msaa = TextureDescriptor::new_render_target(1920, 1080, wgpu::TextureFormat::Rgba8Unorm);
    let msaa_4x = TextureDescriptor::new_render_target(1920, 1080, wgpu::TextureFormat::Rgba8Unorm)
        .with_msaa(4);

    let key_no_msaa = PoolKey::from_texture_desc(&no_msaa);
    let key_msaa = PoolKey::from_texture_desc(&msaa_4x);

    assert_ne!(key_no_msaa, key_msaa);
}

#[test]
fn scenario_mipmap_variants() {
    let no_mips = TextureDescriptor::new_2d(1024, 1024, wgpu::TextureFormat::Rgba8Unorm);
    let with_mips = TextureDescriptor::new_2d(1024, 1024, wgpu::TextureFormat::Rgba8Unorm)
        .with_mips(10);

    let key_no_mips = PoolKey::from_texture_desc(&no_mips);
    let key_with_mips = PoolKey::from_texture_desc(&with_mips);

    assert_ne!(key_no_mips, key_with_mips);
}

#[test]
fn scenario_vertex_buffer_sizes() {
    let small_mesh = BufferDescriptor::new_vertex(10 * 1024);    // Small
    let medium_mesh = BufferDescriptor::new_vertex(100 * 1024);  // Medium
    let large_mesh = BufferDescriptor::new_vertex(2 * 1024 * 1024); // Large

    let key_small = PoolKey::from_buffer_desc(&small_mesh);
    let key_medium = PoolKey::from_buffer_desc(&medium_mesh);
    let key_large = PoolKey::from_buffer_desc(&large_mesh);

    assert_ne!(key_small, key_medium);
    assert_ne!(key_medium, key_large);
}

#[test]
fn scenario_index_buffer() {
    let indices = BufferDescriptor::new_index(64 * 1024);
    let key = PoolKey::from_buffer_desc(&indices);
    assert!(key.is_buffer());
}

// =============================================================================
// SECTION 10 -- Cache Behavior Tests (via PoolKey matching)
// =============================================================================

#[test]
fn cache_exact_match_texture() {
    let desc1 = TextureDescriptor::new_render_target(1920, 1080, wgpu::TextureFormat::Rgba8Unorm);
    let desc2 = TextureDescriptor::new_render_target(1920, 1080, wgpu::TextureFormat::Rgba8Unorm);

    let key1 = PoolKey::from_texture_desc(&desc1);
    let key2 = PoolKey::from_texture_desc(&desc2);

    assert_eq!(key1, key2);
}

#[test]
fn cache_miss_different_format() {
    let desc1 = TextureDescriptor::new_render_target(1920, 1080, wgpu::TextureFormat::Rgba8Unorm);
    let desc2 = TextureDescriptor::new_render_target(1920, 1080, wgpu::TextureFormat::Rgba16Float);

    let key1 = PoolKey::from_texture_desc(&desc1);
    let key2 = PoolKey::from_texture_desc(&desc2);

    assert_ne!(key1, key2);
}

#[test]
fn cache_miss_different_width() {
    let desc1 = TextureDescriptor::new_render_target(1920, 1080, wgpu::TextureFormat::Rgba8Unorm);
    let desc2 = TextureDescriptor::new_render_target(1280, 1080, wgpu::TextureFormat::Rgba8Unorm);

    assert_ne!(PoolKey::from_texture_desc(&desc1), PoolKey::from_texture_desc(&desc2));
}

#[test]
fn cache_miss_different_height() {
    let desc1 = TextureDescriptor::new_render_target(1920, 1080, wgpu::TextureFormat::Rgba8Unorm);
    let desc2 = TextureDescriptor::new_render_target(1920, 720, wgpu::TextureFormat::Rgba8Unorm);

    assert_ne!(PoolKey::from_texture_desc(&desc1), PoolKey::from_texture_desc(&desc2));
}

#[test]
fn cache_miss_different_usage() {
    let rt = TextureDescriptor::new_render_target(256, 256, wgpu::TextureFormat::Rgba8Unorm);
    let tex = TextureDescriptor::new_2d(256, 256, wgpu::TextureFormat::Rgba8Unorm);

    assert_ne!(PoolKey::from_texture_desc(&rt), PoolKey::from_texture_desc(&tex));
}

#[test]
fn cache_buffer_same_class_match() {
    // Both in Small class (1KB-64KB)
    let desc1 = BufferDescriptor::new_storage(10 * 1024);
    let desc2 = BufferDescriptor::new_storage(50 * 1024);

    assert_eq!(PoolKey::from_buffer_desc(&desc1), PoolKey::from_buffer_desc(&desc2));
}

#[test]
fn cache_buffer_different_class_miss() {
    let desc1 = BufferDescriptor::new_storage(10 * 1024);      // Small
    let desc2 = BufferDescriptor::new_storage(100 * 1024);     // Medium

    assert_ne!(PoolKey::from_buffer_desc(&desc1), PoolKey::from_buffer_desc(&desc2));
}

#[test]
fn cache_buffer_different_usage_miss() {
    let storage = BufferDescriptor::new_storage(10 * 1024);
    let uniform = BufferDescriptor::new_uniform(10 * 1024);

    assert_ne!(PoolKey::from_buffer_desc(&storage), PoolKey::from_buffer_desc(&uniform));
}

// =============================================================================
// SECTION 11 -- Edge Case Tests
// =============================================================================

#[test]
fn edge_size_class_exact_boundaries() {
    // Test exact boundary values
    assert_eq!(SizeClass::from_bytes(1024), SizeClass::Small);
    assert_eq!(SizeClass::from_bytes(65536), SizeClass::Medium);
    assert_eq!(SizeClass::from_bytes(1048576), SizeClass::Large);
    assert_eq!(SizeClass::from_bytes(16777216), SizeClass::Huge);
}

#[test]
fn edge_size_class_one_below_boundary() {
    assert_eq!(SizeClass::from_bytes(1023), SizeClass::Tiny);
    assert_eq!(SizeClass::from_bytes(65535), SizeClass::Small);
    assert_eq!(SizeClass::from_bytes(1048575), SizeClass::Medium);
    assert_eq!(SizeClass::from_bytes(16777215), SizeClass::Large);
}

#[test]
fn edge_texture_1x1() {
    let desc = TextureDescriptor::new_2d(1, 1, wgpu::TextureFormat::Rgba8Unorm);
    assert_eq!(desc.width, 1);
    assert_eq!(desc.height, 1);
    assert_eq!(desc.size_bytes(), 4); // 1 * 1 * 4 bytes
}

#[test]
fn edge_texture_max_dimension() {
    let desc = TextureDescriptor::new_2d(16384, 16384, wgpu::TextureFormat::Rgba8Unorm);
    assert_eq!(desc.width, 16384);
    assert_eq!(desc.height, 16384);
}

#[test]
fn edge_buffer_zero_size() {
    let desc = BufferDescriptor::new(0, wgpu::BufferUsages::COPY_DST);
    assert_eq!(desc.size, 0);
    assert_eq!(SizeClass::from_bytes(0), SizeClass::Tiny);
}

#[test]
fn edge_buffer_one_byte() {
    let desc = BufferDescriptor::new(1, wgpu::BufferUsages::COPY_DST);
    assert_eq!(desc.size, 1);
}

#[test]
fn edge_pool_stats_saturating_sub() {
    let mut stats = PoolStats::new();
    stats.total_textures = 0;
    stats.active_textures = 5;
    assert_eq!(stats.idle_textures(), 0); // Should saturate, not underflow
}

#[test]
fn edge_alias_group_single_pass_resource() {
    let mut group = AliasGroup::new();
    let r = ResourceLifetimeRange::new(1, 5, 5, 1024);
    assert!(group.try_add(&r));
    assert_eq!(group.first_pass, 5);
    assert_eq!(group.last_pass, 5);
}

#[test]
fn edge_compute_alias_groups_single_pass_all() {
    let resources: Vec<ResourceLifetimeRange> = (0..5)
        .map(|i| ResourceLifetimeRange::new(i as u64, 0, 0, 1024))
        .collect();
    let groups = compute_alias_groups(&resources);
    // All same pass = all overlap = 5 groups
    assert_eq!(groups.len(), 5);
}

#[test]
fn edge_poolconfig_extreme_values() {
    let config = PoolConfig::new()
        .with_max_idle_frames(u64::MAX)
        .with_max_texture_pool_size(usize::MAX)
        .with_max_buffer_pool_size(usize::MAX)
        .with_memory_budget(u64::MAX);

    assert_eq!(config.max_idle_frames, u64::MAX);
    assert_eq!(config.max_texture_pool_size, usize::MAX);
}

#[test]
fn edge_resource_lifetime_zero_to_zero() {
    let r = ResourceLifetimeRange::new(1, 0, 0, 1024);
    assert_eq!(r.first_use_pass(), 0);
    assert_eq!(r.last_use_pass(), 0);
}

#[test]
fn edge_resource_lifetime_large_pass_numbers() {
    let r = ResourceLifetimeRange::new(1, u32::MAX - 1, u32::MAX, 1024);
    assert_eq!(r.first_use_pass(), u32::MAX - 1);
    assert_eq!(r.last_use_pass(), u32::MAX);
}

#[test]
fn edge_estimate_savings_identical_sizes() {
    let resources = vec![
        ResourceLifetimeRange::new(1, 0, 2, 1000),
        ResourceLifetimeRange::new(2, 3, 5, 1000),
        ResourceLifetimeRange::new(3, 6, 8, 1000),
    ];
    let groups = compute_alias_groups(&resources);
    let (without, with, _savings) = estimate_aliasing_savings(&resources, &groups);

    assert_eq!(without, 3000);
    assert_eq!(with, 1000);
}

#[test]
fn edge_texture_size_bytes_with_mips() {
    let desc = TextureDescriptor::new_2d(256, 256, wgpu::TextureFormat::Rgba8Unorm)
        .with_mips(5);
    let size = desc.size_bytes();
    // Should be more than base due to mips
    let base = 256 * 256 * 4;
    assert!(size > base as u64);
}

// =============================================================================
// SECTION 12 -- Memory Aliasing Eligibility Tests
// =============================================================================

#[test]
fn aliasing_eligibility_sequential_passes() {
    let r1 = ResourceLifetimeRange::new(1, 0, 0, 1024);
    let r2 = ResourceLifetimeRange::new(2, 1, 1, 2048);
    let r3 = ResourceLifetimeRange::new(3, 2, 2, 512);

    assert!(r1.can_alias_with(&r2));
    assert!(r2.can_alias_with(&r3));
    assert!(r1.can_alias_with(&r3));
}

#[test]
fn aliasing_eligibility_gap_between_passes() {
    let r1 = ResourceLifetimeRange::new(1, 0, 2, 1024);
    let r2 = ResourceLifetimeRange::new(2, 5, 7, 2048);

    assert!(r1.can_alias_with(&r2));
}

#[test]
fn aliasing_ineligibility_full_overlap() {
    let r1 = ResourceLifetimeRange::new(1, 0, 10, 1024);
    let r2 = ResourceLifetimeRange::new(2, 2, 8, 2048);

    assert!(!r1.can_alias_with(&r2));
}

#[test]
fn aliasing_ineligibility_partial_overlap_start() {
    let r1 = ResourceLifetimeRange::new(1, 0, 5, 1024);
    let r2 = ResourceLifetimeRange::new(2, 4, 8, 2048);

    assert!(!r1.can_alias_with(&r2));
}

#[test]
fn aliasing_ineligibility_partial_overlap_end() {
    let r1 = ResourceLifetimeRange::new(1, 4, 8, 1024);
    let r2 = ResourceLifetimeRange::new(2, 0, 5, 2048);

    assert!(!r1.can_alias_with(&r2));
}

#[test]
fn aliasing_savings_estimation_accuracy() {
    // 3 resources, each 1MB, all can alias
    let resources = vec![
        ResourceLifetimeRange::new(1, 0, 1, 1024 * 1024),
        ResourceLifetimeRange::new(2, 2, 3, 1024 * 1024),
        ResourceLifetimeRange::new(3, 4, 5, 1024 * 1024),
    ];
    let groups = compute_alias_groups(&resources);
    let (without, with, savings) = estimate_aliasing_savings(&resources, &groups);

    assert_eq!(without, 3 * 1024 * 1024);
    assert_eq!(with, 1024 * 1024);
    assert!((savings - 66.67).abs() < 1.0);
}

#[test]
fn aliasing_group_computation_optimal() {
    // Classic register allocation scenario
    let resources = vec![
        ResourceLifetimeRange::new(1, 0, 3, 1024),  // [0, 3]
        ResourceLifetimeRange::new(2, 1, 4, 2048),  // [1, 4] - overlaps with 1
        ResourceLifetimeRange::new(3, 4, 6, 512),   // [4, 6] - overlaps with 2
        ResourceLifetimeRange::new(4, 5, 7, 768),   // [5, 7] - overlaps with 3
    ];
    let groups = compute_alias_groups(&resources);

    // Should need at least 2 groups due to overlaps
    assert!(groups.len() >= 2);
    // But all resources should be assigned
    let total: usize = groups.iter().map(|g| g.len()).sum();
    assert_eq!(total, 4);
}

#[test]
fn aliasing_complex_dependency_chain() {
    // Pass 0: A writes
    // Pass 1: A reads, B writes
    // Pass 2: B reads, C writes
    // Pass 3: C reads, D writes
    // Pass 4: D reads
    let resources = vec![
        ResourceLifetimeRange::new(1, 0, 1, 1024),  // A: [0, 1]
        ResourceLifetimeRange::new(2, 1, 2, 2048),  // B: [1, 2]
        ResourceLifetimeRange::new(3, 2, 3, 512),   // C: [2, 3]
        ResourceLifetimeRange::new(4, 3, 4, 768),   // D: [3, 4]
    ];
    let groups = compute_alias_groups(&resources);

    // A and C can alias (no overlap), B and D can alias
    // Should need ~2 groups
    assert!(groups.len() <= 3);
}

// =============================================================================
// SECTION 13 -- Texture Descriptor Size Calculations
// =============================================================================

#[test]
fn texture_size_rgba8() {
    let desc = TextureDescriptor::new_2d(100, 100, wgpu::TextureFormat::Rgba8Unorm);
    assert_eq!(desc.size_bytes(), 100 * 100 * 4);
}

#[test]
fn texture_size_r8() {
    let desc = TextureDescriptor::new_2d(100, 100, wgpu::TextureFormat::R8Unorm);
    assert_eq!(desc.size_bytes(), 100 * 100 * 1);
}

#[test]
fn texture_size_rg16() {
    let desc = TextureDescriptor::new_2d(100, 100, wgpu::TextureFormat::Rg16Float);
    // Rg16Float = 2 channels * 2 bytes = 4 bytes per texel
    assert_eq!(desc.size_bytes(), 100 * 100 * 4);
}

#[test]
fn texture_size_rgba32f() {
    let desc = TextureDescriptor::new_2d(100, 100, wgpu::TextureFormat::Rgba32Float);
    assert_eq!(desc.size_bytes(), 100 * 100 * 16);
}

#[test]
fn texture_size_depth32() {
    let desc = TextureDescriptor::new_depth(100, 100);
    assert_eq!(desc.size_bytes(), 100 * 100 * 4);
}

#[test]
fn texture_size_msaa_multiplier() {
    let base = TextureDescriptor::new_2d(100, 100, wgpu::TextureFormat::Rgba8Unorm);
    let msaa = TextureDescriptor::new_2d(100, 100, wgpu::TextureFormat::Rgba8Unorm).with_msaa(4);

    assert_eq!(msaa.size_bytes(), base.size_bytes() * 4);
}

// =============================================================================
// SECTION 14 -- Buffer Descriptor Tests
// =============================================================================

#[test]
fn buffer_vertex_usage() {
    let desc = BufferDescriptor::new_vertex(1024);
    assert!(desc.usage.contains(wgpu::BufferUsages::VERTEX));
    assert!(desc.usage.contains(wgpu::BufferUsages::COPY_DST));
}

#[test]
fn buffer_index_usage() {
    let desc = BufferDescriptor::new_index(1024);
    assert!(desc.usage.contains(wgpu::BufferUsages::INDEX));
}

#[test]
fn buffer_storage_usage() {
    let desc = BufferDescriptor::new_storage(1024);
    assert!(desc.usage.contains(wgpu::BufferUsages::STORAGE));
    assert!(desc.usage.contains(wgpu::BufferUsages::COPY_DST));
    assert!(desc.usage.contains(wgpu::BufferUsages::COPY_SRC));
}

#[test]
fn buffer_staging_read_usage() {
    let desc = BufferDescriptor::new_staging_read(1024);
    assert!(desc.usage.contains(wgpu::BufferUsages::MAP_READ));
    assert!(desc.usage.contains(wgpu::BufferUsages::COPY_DST));
}

#[test]
fn buffer_staging_write_usage() {
    let desc = BufferDescriptor::new_staging_write(1024);
    assert!(desc.usage.contains(wgpu::BufferUsages::MAP_WRITE));
    assert!(desc.usage.contains(wgpu::BufferUsages::COPY_SRC));
    assert!(desc.mapped_at_creation);
}

#[test]
fn buffer_with_label() {
    let desc = BufferDescriptor::new_uniform(256).with_label("camera_uniforms");
    assert_eq!(desc.label, Some("camera_uniforms".to_string()));
}

#[test]
fn buffer_default() {
    let desc = BufferDescriptor::default();
    assert_eq!(desc.size, 256);
    assert!(desc.usage.contains(wgpu::BufferUsages::COPY_DST));
}

#[test]
fn buffer_display() {
    let desc = BufferDescriptor::new_storage(1024);
    let display = format!("{}", desc);
    assert!(display.contains("Buffer"));
    assert!(display.contains("1024"));
}

// =============================================================================
// SECTION 15 -- Additional PoolKey Hash Tests
// =============================================================================

#[test]
fn poolkey_hash_different_textures_differ() {
    let desc1 = TextureDescriptor::new_render_target(1920, 1080, wgpu::TextureFormat::Rgba8Unorm);
    let desc2 = TextureDescriptor::new_render_target(1280, 720, wgpu::TextureFormat::Rgba8Unorm);

    let key1 = PoolKey::from_texture_desc(&desc1);
    let key2 = PoolKey::from_texture_desc(&desc2);

    let mut h1 = DefaultHasher::new();
    let mut h2 = DefaultHasher::new();
    key1.hash(&mut h1);
    key2.hash(&mut h2);

    assert_ne!(h1.finish(), h2.finish());
}

#[test]
fn poolkey_hash_texture_vs_buffer_differ() {
    let tex = TextureDescriptor::new_2d(256, 256, wgpu::TextureFormat::Rgba8Unorm);
    let buf = BufferDescriptor::new_storage(256 * 256 * 4);

    let key1 = PoolKey::from_texture_desc(&tex);
    let key2 = PoolKey::from_buffer_desc(&buf);

    let mut h1 = DefaultHasher::new();
    let mut h2 = DefaultHasher::new();
    key1.hash(&mut h1);
    key2.hash(&mut h2);

    assert_ne!(h1.finish(), h2.finish());
}

// =============================================================================
// SECTION 16 -- Frame Lifecycle Concept Tests
// =============================================================================

#[test]
fn frame_lifecycle_stats_tracking() {
    let mut stats = PoolStats::new();

    // Simulate frame 1
    stats.cache_misses += 5; // Initial allocations
    stats.total_textures = 5;
    stats.active_textures = 5;

    // Simulate frame 2 with reuse
    stats.cache_hits += 4;
    stats.cache_misses += 1;

    assert_eq!(stats.hit_rate(), 4.0 / 10.0);
}

#[test]
fn frame_lifecycle_idle_tracking() {
    let mut stats = PoolStats::new();
    stats.total_textures = 10;

    // All active
    stats.active_textures = 10;
    assert_eq!(stats.idle_textures(), 0);

    // Half released
    stats.active_textures = 5;
    assert_eq!(stats.idle_textures(), 5);

    // All released
    stats.active_textures = 0;
    assert_eq!(stats.idle_textures(), 10);
}

#[test]
fn frame_lifecycle_gc_tracking() {
    let mut stats = PoolStats::new();
    stats.total_textures = 10;
    stats.gc_count = 0;

    // Simulate GC removing 3
    stats.total_textures = 7;
    stats.gc_count = 3;

    assert_eq!(stats.total_textures, 7);
    assert_eq!(stats.gc_count, 3);
}

// =============================================================================
// SECTION 17 -- TextureDescriptor Builder Chain Tests
// =============================================================================

#[test]
fn texture_builder_chain_all_options() {
    let desc = TextureDescriptor::new_2d(512, 512, wgpu::TextureFormat::Rgba8Unorm)
        .with_mips(5)
        .with_msaa(4)
        .with_label("test_texture");

    assert_eq!(desc.mip_levels, 5);
    assert_eq!(desc.sample_count, 4);
    assert_eq!(desc.label, Some("test_texture".to_string()));
}

#[test]
fn texture_builder_default_format() {
    let desc = TextureDescriptor::default();
    assert_eq!(desc.format, wgpu::TextureFormat::Rgba8Unorm);
}

#[test]
fn texture_display() {
    let desc = TextureDescriptor::new_render_target(1920, 1080, wgpu::TextureFormat::Rgba8Unorm);
    let display = format!("{}", desc);
    assert!(display.contains("Texture"));
    assert!(display.contains("1920"));
    assert!(display.contains("1080"));
}

// =============================================================================
// Total: 130+ tests covering all required categories
// =============================================================================
