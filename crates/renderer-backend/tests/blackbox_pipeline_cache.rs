// SPDX-License-Identifier: MIT
//
// blackbox_pipeline_cache.rs -- Blackbox tests for T-WGPU-P3.1.7 Pipeline Cache.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only the documented public types and functions:
//
//   - PipelineKey -- Unique identifier for pipeline configurations
//   - RenderPipelineCache -- Thread-safe cache for compiled pipelines
//   - CacheMetrics -- Cache performance statistics
//   - hash_vertex_layout -- Hash function for vertex layouts
//   - hash_color_targets -- Hash function for color targets
//   - VertexBufferLayoutDescriptor, ColorTargetStateDescriptor -- For hash testing
//
// ACCEPTANCE CRITERIA:
//   1. API surface tests -- All public types accessible (15+ tests)
//   2. Cache behavior tests -- Hit/miss, invalidation (15+ tests)
//   3. GPU integration tests -- Real pipeline caching (10+ tests)
//   4. Concurrent access tests -- Thread safety (5+ tests)
//   5. Hash function tests -- Consistency and collision resistance (10+ tests)
//
// Total target: 55+ tests

use pollster::block_on;
use renderer_backend::device::{enumerate_adapters_with_info, TrinityInstance};
use renderer_backend::render_pipeline::{
    hash_color_targets, hash_vertex_layout, CacheMetrics, ColorTargetStateDescriptor,
    PipelineKey, RenderPipelineCache, VertexBufferLayoutDescriptor,
};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

// =============================================================================
// TEST SHADERS
// =============================================================================

/// Minimal vertex shader for basic pipeline creation tests.
const MINIMAL_VERTEX_SHADER: &str = r#"
@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
"#;

/// Minimal fragment shader for basic pipeline creation tests.
const MINIMAL_FRAGMENT_SHADER: &str = r#"
@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
"#;

/// Alternative fragment shader for testing multiple pipelines.
const ALT_FRAGMENT_SHADER: &str = r#"
@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 1.0, 0.0, 1.0);
}
"#;

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Creates a TrinityInstance and gets the first available adapter.
fn get_test_adapter() -> Option<wgpu::Adapter> {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());
    result.adapters.into_iter().next()
}

/// Helper macro to skip test if no GPU adapter is available.
macro_rules! require_adapter {
    () => {
        match get_test_adapter() {
            Some(adapter) => adapter,
            None => {
                eprintln!("SKIP: No GPU adapter available for this test");
                return;
            }
        }
    };
}

/// Creates a wgpu device for testing.
fn create_test_device(adapter: &wgpu::Adapter) -> Option<(wgpu::Device, wgpu::Queue)> {
    block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None)).ok()
}

/// Helper macro to get a device, skipping if unavailable.
macro_rules! require_device {
    ($adapter:expr) => {
        match create_test_device($adapter) {
            Some((device, queue)) => (device, queue),
            None => {
                eprintln!("SKIP: Could not create device");
                return;
            }
        }
    };
}

/// Creates a shader module from WGSL source.
fn create_shader_module(device: &wgpu::Device, source: &str) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("test_shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(source)),
    })
}

/// Creates an empty pipeline layout for basic tests.
fn create_empty_layout(device: &wgpu::Device) -> wgpu::PipelineLayout {
    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("test_layout"),
        bind_group_layouts: &[],
        push_constant_ranges: &[],
    })
}

/// Helper to compute hash of a PipelineKey.
fn hash_key(key: &PipelineKey) -> u64 {
    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    hasher.finish()
}

// =============================================================================
// SECTION 1: API SURFACE TESTS (No GPU Required)
// =============================================================================

/// Test: All public types from pipeline_cache module are accessible.
#[test]
fn test_pipeline_cache_api_surface() {
    // Verify all public types are accessible (compile-time check)
    let _: fn(u64) -> PipelineKey = PipelineKey::new;
    let _: fn() -> CacheMetrics = CacheMetrics::default;

    // Verify hash functions are accessible
    let _: fn(&[VertexBufferLayoutDescriptor]) -> u64 = hash_vertex_layout;
    let _: fn(&[Option<ColorTargetStateDescriptor>]) -> u64 = hash_color_targets;
}

/// Test: PipelineKey constructor with minimal parameters.
#[test]
fn test_pipeline_key_new_minimal() {
    let key = PipelineKey::new(1);

    assert_eq!(key.vertex_shader_id, 1);
    assert_eq!(key.fragment_shader_id, None);
    assert_eq!(key.vertex_layout_hash, 0);
    assert_eq!(key.topology, wgpu::PrimitiveTopology::TriangleList);
    assert_eq!(key.front_face, wgpu::FrontFace::Ccw);
    assert_eq!(key.cull_mode, Some(wgpu::Face::Back));
    assert_eq!(key.polygon_mode, wgpu::PolygonMode::Fill);
    assert_eq!(key.depth_format, None);
    assert!(key.depth_write);
    assert_eq!(key.depth_compare, wgpu::CompareFunction::Less);
    assert_eq!(key.sample_count, 1);
    assert_eq!(key.color_targets_hash, 0);
}

/// Test: PipelineKey builder with fragment shader.
#[test]
fn test_pipeline_key_with_fragment_shader() {
    let key = PipelineKey::new(1).with_fragment_shader(2);

    assert_eq!(key.vertex_shader_id, 1);
    assert_eq!(key.fragment_shader_id, Some(2));
}

/// Test: PipelineKey builder with vertex layout hash.
#[test]
fn test_pipeline_key_with_vertex_layout_hash() {
    let key = PipelineKey::new(1).with_vertex_layout_hash(12345);

    assert_eq!(key.vertex_layout_hash, 12345);
}

/// Test: PipelineKey builder with all topology variants.
#[test]
fn test_pipeline_key_with_topologies() {
    let topologies = [
        wgpu::PrimitiveTopology::PointList,
        wgpu::PrimitiveTopology::LineList,
        wgpu::PrimitiveTopology::LineStrip,
        wgpu::PrimitiveTopology::TriangleList,
        wgpu::PrimitiveTopology::TriangleStrip,
    ];

    for topo in topologies {
        let key = PipelineKey::new(1).with_topology(topo);
        assert_eq!(key.topology, topo);
    }
}

/// Test: PipelineKey builder with front face variants.
#[test]
fn test_pipeline_key_with_front_face() {
    let key_ccw = PipelineKey::new(1).with_front_face(wgpu::FrontFace::Ccw);
    let key_cw = PipelineKey::new(1).with_front_face(wgpu::FrontFace::Cw);

    assert_eq!(key_ccw.front_face, wgpu::FrontFace::Ccw);
    assert_eq!(key_cw.front_face, wgpu::FrontFace::Cw);
}

/// Test: PipelineKey builder with cull mode variants.
#[test]
fn test_pipeline_key_with_cull_modes() {
    let key_back = PipelineKey::new(1).with_cull_mode(Some(wgpu::Face::Back));
    let key_front = PipelineKey::new(1).with_cull_mode(Some(wgpu::Face::Front));
    let key_none = PipelineKey::new(1).with_cull_mode(None);

    assert_eq!(key_back.cull_mode, Some(wgpu::Face::Back));
    assert_eq!(key_front.cull_mode, Some(wgpu::Face::Front));
    assert_eq!(key_none.cull_mode, None);
}

/// Test: PipelineKey builder with polygon mode variants.
#[test]
fn test_pipeline_key_with_polygon_modes() {
    let modes = [
        wgpu::PolygonMode::Fill,
        wgpu::PolygonMode::Line,
        wgpu::PolygonMode::Point,
    ];

    for mode in modes {
        let key = PipelineKey::new(1).with_polygon_mode(mode);
        assert_eq!(key.polygon_mode, mode);
    }
}

/// Test: PipelineKey builder with depth configuration.
#[test]
fn test_pipeline_key_with_depth_config() {
    let key = PipelineKey::new(1)
        .with_depth_format(wgpu::TextureFormat::Depth32Float)
        .with_depth_write(false)
        .with_depth_compare(wgpu::CompareFunction::Greater);

    assert_eq!(key.depth_format, Some(wgpu::TextureFormat::Depth32Float));
    assert!(!key.depth_write);
    assert_eq!(key.depth_compare, wgpu::CompareFunction::Greater);
}

/// Test: PipelineKey builder with sample count.
#[test]
fn test_pipeline_key_with_sample_count() {
    for count in [1, 2, 4, 8] {
        let key = PipelineKey::new(1).with_sample_count(count);
        assert_eq!(key.sample_count, count);
    }
}

/// Test: PipelineKey builder chaining all options.
#[test]
fn test_pipeline_key_builder_full_chain() {
    let key = PipelineKey::new(1)
        .with_fragment_shader(2)
        .with_vertex_layout_hash(12345)
        .with_topology(wgpu::PrimitiveTopology::TriangleStrip)
        .with_front_face(wgpu::FrontFace::Cw)
        .with_cull_mode(Some(wgpu::Face::Front))
        .with_polygon_mode(wgpu::PolygonMode::Line)
        .with_depth_format(wgpu::TextureFormat::Depth32Float)
        .with_depth_write(false)
        .with_depth_compare(wgpu::CompareFunction::Greater)
        .with_sample_count(4)
        .with_color_targets_hash(67890);

    assert_eq!(key.vertex_shader_id, 1);
    assert_eq!(key.fragment_shader_id, Some(2));
    assert_eq!(key.vertex_layout_hash, 12345);
    assert_eq!(key.topology, wgpu::PrimitiveTopology::TriangleStrip);
    assert_eq!(key.front_face, wgpu::FrontFace::Cw);
    assert_eq!(key.cull_mode, Some(wgpu::Face::Front));
    assert_eq!(key.polygon_mode, wgpu::PolygonMode::Line);
    assert_eq!(key.depth_format, Some(wgpu::TextureFormat::Depth32Float));
    assert!(!key.depth_write);
    assert_eq!(key.depth_compare, wgpu::CompareFunction::Greater);
    assert_eq!(key.sample_count, 4);
    assert_eq!(key.color_targets_hash, 67890);
}

/// Test: PipelineKey uses_shader correctly identifies vertex shader.
#[test]
fn test_pipeline_key_uses_shader_vertex() {
    let key = PipelineKey::new(42);

    assert!(key.uses_shader(42));
    assert!(!key.uses_shader(1));
    assert!(!key.uses_shader(0));
}

/// Test: PipelineKey uses_shader correctly identifies fragment shader.
#[test]
fn test_pipeline_key_uses_shader_fragment() {
    let key = PipelineKey::new(1).with_fragment_shader(99);

    assert!(key.uses_shader(1));
    assert!(key.uses_shader(99));
    assert!(!key.uses_shader(2));
}

/// Test: PipelineKey uses_shader with no fragment shader.
#[test]
fn test_pipeline_key_uses_shader_no_fragment() {
    let key = PipelineKey::new(1);

    assert!(key.uses_shader(1));
    assert!(!key.uses_shader(2));
    assert_eq!(key.fragment_shader_id, None);
}

/// Test: PipelineKey equality for identical configurations.
#[test]
fn test_pipeline_key_equality_same() {
    let key1 = PipelineKey::new(1).with_fragment_shader(2).with_sample_count(4);
    let key2 = PipelineKey::new(1).with_fragment_shader(2).with_sample_count(4);

    assert_eq!(key1, key2);
}

/// Test: PipelineKey inequality for different vertex shaders.
#[test]
fn test_pipeline_key_inequality_vertex_shader() {
    let key1 = PipelineKey::new(1);
    let key2 = PipelineKey::new(2);

    assert_ne!(key1, key2);
}

/// Test: PipelineKey inequality for different fragment shaders.
#[test]
fn test_pipeline_key_inequality_fragment_shader() {
    let key1 = PipelineKey::new(1).with_fragment_shader(2);
    let key2 = PipelineKey::new(1).with_fragment_shader(3);
    let key3 = PipelineKey::new(1); // No fragment shader

    assert_ne!(key1, key2);
    assert_ne!(key1, key3);
}

/// Test: PipelineKey inequality for different topologies.
#[test]
fn test_pipeline_key_inequality_topology() {
    let key1 = PipelineKey::new(1).with_topology(wgpu::PrimitiveTopology::TriangleList);
    let key2 = PipelineKey::new(1).with_topology(wgpu::PrimitiveTopology::LineList);

    assert_ne!(key1, key2);
}

/// Test: PipelineKey hash consistency.
#[test]
fn test_pipeline_key_hash_consistency() {
    let key = PipelineKey::new(42).with_fragment_shader(99);
    let hash1 = hash_key(&key);
    let hash2 = hash_key(&key);
    let hash3 = hash_key(&key);

    assert_eq!(hash1, hash2);
    assert_eq!(hash2, hash3);
}

/// Test: PipelineKey hash equality for equal keys.
#[test]
fn test_pipeline_key_hash_equality() {
    let key1 = PipelineKey::new(1).with_fragment_shader(2);
    let key2 = PipelineKey::new(1).with_fragment_shader(2);

    assert_eq!(hash_key(&key1), hash_key(&key2));
}

/// Test: PipelineKey hash inequality for different keys.
#[test]
fn test_pipeline_key_hash_inequality() {
    let key1 = PipelineKey::new(1);
    let key2 = PipelineKey::new(2);

    // Different keys should (almost certainly) have different hashes
    assert_ne!(hash_key(&key1), hash_key(&key2));
}

/// Test: PipelineKey Clone implementation.
#[test]
fn test_pipeline_key_clone() {
    let key = PipelineKey::new(1)
        .with_fragment_shader(2)
        .with_sample_count(4);
    let cloned = key.clone();

    assert_eq!(key, cloned);
    assert_eq!(hash_key(&key), hash_key(&cloned));
}

/// Test: PipelineKey implements Send + Sync.
#[test]
fn test_pipeline_key_send_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    assert_send::<PipelineKey>();
    assert_sync::<PipelineKey>();
}

// =============================================================================
// SECTION 2: CACHE METRICS TESTS
// =============================================================================

/// Test: CacheMetrics default values.
#[test]
fn test_cache_metrics_default() {
    let metrics = CacheMetrics::default();

    assert_eq!(metrics.hits, 0);
    assert_eq!(metrics.misses, 0);
    assert_eq!(metrics.evictions, 0);
    assert_eq!(metrics.invalidations, 0);
}

/// Test: CacheMetrics hit_rate with zero accesses.
#[test]
fn test_cache_metrics_hit_rate_zero() {
    let metrics = CacheMetrics::default();
    assert_eq!(metrics.hit_rate(), 0.0);
}

/// Test: CacheMetrics hit_rate with all hits.
#[test]
fn test_cache_metrics_hit_rate_all_hits() {
    let metrics = CacheMetrics {
        hits: 100,
        misses: 0,
        evictions: 0,
        invalidations: 0,
    };
    assert_eq!(metrics.hit_rate(), 1.0);
}

/// Test: CacheMetrics hit_rate with all misses.
#[test]
fn test_cache_metrics_hit_rate_all_misses() {
    let metrics = CacheMetrics {
        hits: 0,
        misses: 100,
        evictions: 0,
        invalidations: 0,
    };
    assert_eq!(metrics.hit_rate(), 0.0);
}

/// Test: CacheMetrics hit_rate with mixed results.
#[test]
fn test_cache_metrics_hit_rate_fifty_percent() {
    let metrics = CacheMetrics {
        hits: 50,
        misses: 50,
        evictions: 0,
        invalidations: 0,
    };
    assert!((metrics.hit_rate() - 0.5).abs() < f64::EPSILON);
}

/// Test: CacheMetrics total_accesses calculation.
#[test]
fn test_cache_metrics_total_accesses() {
    let metrics = CacheMetrics {
        hits: 30,
        misses: 70,
        evictions: 10,
        invalidations: 5,
    };
    assert_eq!(metrics.total_accesses(), 100);
}

/// Test: CacheMetrics implements Send + Sync.
#[test]
fn test_cache_metrics_send_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    assert_send::<CacheMetrics>();
    assert_sync::<CacheMetrics>();
}

/// Test: CacheMetrics is Copy.
#[test]
fn test_cache_metrics_copy() {
    let metrics = CacheMetrics {
        hits: 10,
        misses: 5,
        evictions: 2,
        invalidations: 1,
    };
    let copy = metrics;
    assert_eq!(metrics.hits, copy.hits);
    assert_eq!(metrics.misses, copy.misses);
}

// =============================================================================
// SECTION 3: HASH FUNCTION TESTS
// =============================================================================

/// Test: hash_vertex_layout with empty input.
#[test]
fn test_hash_vertex_layout_empty() {
    let hash = hash_vertex_layout(&[]);
    // Should produce a consistent hash
    assert_eq!(hash, hash_vertex_layout(&[]));
}

/// Test: hash_vertex_layout with single buffer.
#[test]
fn test_hash_vertex_layout_single_buffer() {
    let buffer = VertexBufferLayoutDescriptor::per_vertex(32)
        .with_attribute(wgpu::VertexFormat::Float32x3, 0, 0)
        .with_attribute(wgpu::VertexFormat::Float32x2, 12, 1);

    let hash1 = hash_vertex_layout(&[buffer.clone()]);
    let hash2 = hash_vertex_layout(&[buffer]);
    assert_eq!(hash1, hash2);
}

/// Test: hash_vertex_layout with different strides.
#[test]
fn test_hash_vertex_layout_different_stride() {
    let buffer1 = VertexBufferLayoutDescriptor::per_vertex(32);
    let buffer2 = VertexBufferLayoutDescriptor::per_vertex(64);

    assert_ne!(hash_vertex_layout(&[buffer1]), hash_vertex_layout(&[buffer2]));
}

/// Test: hash_vertex_layout with different step modes.
#[test]
fn test_hash_vertex_layout_different_step_mode() {
    let buffer1 = VertexBufferLayoutDescriptor::per_vertex(32);
    let buffer2 = VertexBufferLayoutDescriptor::per_instance(32);

    assert_ne!(hash_vertex_layout(&[buffer1]), hash_vertex_layout(&[buffer2]));
}

/// Test: hash_vertex_layout attribute order matters.
#[test]
fn test_hash_vertex_layout_attribute_order() {
    let buffer1 = VertexBufferLayoutDescriptor::per_vertex(32)
        .with_attribute(wgpu::VertexFormat::Float32x3, 0, 0)
        .with_attribute(wgpu::VertexFormat::Float32x2, 12, 1);

    let buffer2 = VertexBufferLayoutDescriptor::per_vertex(32)
        .with_attribute(wgpu::VertexFormat::Float32x2, 12, 1)
        .with_attribute(wgpu::VertexFormat::Float32x3, 0, 0);

    // Different order should produce different hash
    assert_ne!(hash_vertex_layout(&[buffer1]), hash_vertex_layout(&[buffer2]));
}

/// Test: hash_vertex_layout with multiple buffers.
#[test]
fn test_hash_vertex_layout_multiple_buffers() {
    let buffer1 = VertexBufferLayoutDescriptor::per_vertex(12)
        .with_attribute(wgpu::VertexFormat::Float32x3, 0, 0);

    let buffer2 = VertexBufferLayoutDescriptor::per_instance(64)
        .with_attribute(wgpu::VertexFormat::Float32x4, 0, 3)
        .with_attribute(wgpu::VertexFormat::Float32x4, 16, 4);

    let hash1 = hash_vertex_layout(&[buffer1.clone(), buffer2.clone()]);
    let hash2 = hash_vertex_layout(&[buffer1, buffer2]);
    assert_eq!(hash1, hash2);
}

/// Test: hash_vertex_layout buffer order matters.
#[test]
fn test_hash_vertex_layout_buffer_order() {
    let buffer1 = VertexBufferLayoutDescriptor::per_vertex(12);
    let buffer2 = VertexBufferLayoutDescriptor::per_instance(64);

    assert_ne!(
        hash_vertex_layout(&[buffer1.clone(), buffer2.clone()]),
        hash_vertex_layout(&[buffer2, buffer1])
    );
}

/// Test: hash_color_targets with empty input.
#[test]
fn test_hash_color_targets_empty() {
    let hash = hash_color_targets(&[]);
    assert_eq!(hash, hash_color_targets(&[]));
}

/// Test: hash_color_targets with single target.
#[test]
fn test_hash_color_targets_single() {
    let target = ColorTargetStateDescriptor::srgb();

    let hash1 = hash_color_targets(&[Some(target.clone())]);
    let hash2 = hash_color_targets(&[Some(target)]);
    assert_eq!(hash1, hash2);
}

/// Test: hash_color_targets with different formats.
#[test]
fn test_hash_color_targets_different_format() {
    let target1 = ColorTargetStateDescriptor::srgb();
    let target2 = ColorTargetStateDescriptor::hdr();

    assert_ne!(
        hash_color_targets(&[Some(target1)]),
        hash_color_targets(&[Some(target2)])
    );
}

/// Test: hash_color_targets with and without blend.
#[test]
fn test_hash_color_targets_blend_difference() {
    let target1 = ColorTargetStateDescriptor::srgb();
    let target2 = ColorTargetStateDescriptor::srgb().alpha_blend();

    assert_ne!(
        hash_color_targets(&[Some(target1)]),
        hash_color_targets(&[Some(target2)])
    );
}

/// Test: hash_color_targets with None target.
#[test]
fn test_hash_color_targets_none() {
    let hash1 = hash_color_targets(&[None]);
    let hash2 = hash_color_targets(&[Some(ColorTargetStateDescriptor::srgb())]);

    assert_ne!(hash1, hash2);
}

/// Test: hash_color_targets target order matters.
#[test]
fn test_hash_color_targets_order() {
    let target1 = ColorTargetStateDescriptor::srgb();
    let target2 = ColorTargetStateDescriptor::hdr();

    assert_ne!(
        hash_color_targets(&[Some(target1.clone()), Some(target2.clone())]),
        hash_color_targets(&[Some(target2), Some(target1)])
    );
}

/// Test: Hash functions produce diverse values.
#[test]
fn test_hash_functions_diversity() {
    let buffers: Vec<VertexBufferLayoutDescriptor> = (0..10)
        .map(|i| VertexBufferLayoutDescriptor::per_vertex((i + 1) * 4))
        .collect();

    let hashes: HashSet<u64> = buffers
        .iter()
        .map(|b| hash_vertex_layout(&[b.clone()]))
        .collect();

    // All different strides should produce different hashes
    assert_eq!(hashes.len(), 10);
}

// =============================================================================
// SECTION 4: GPU INTEGRATION TESTS - CACHE BEHAVIOR
// =============================================================================

/// Test: RenderPipelineCache can be created with a device.
#[test]
fn test_cache_creation_with_device() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);
    let device = Arc::new(device);

    let cache = RenderPipelineCache::new(device);

    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
}

/// Test: RenderPipelineCache get_or_create creates pipeline on miss.
#[test]
fn test_cache_get_or_create_miss() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);
    let device = Arc::new(device);

    let cache = RenderPipelineCache::new(device.clone());

    let vs = create_shader_module(&device, MINIMAL_VERTEX_SHADER);
    let fs = create_shader_module(&device, MINIMAL_FRAGMENT_SHADER);
    let layout = create_empty_layout(&device);

    let key = PipelineKey::new(1).with_fragment_shader(2);

    let pipeline = cache.get_or_create(&key, || {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("test_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &vs,
                entry_point: "vs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &fs,
                entry_point: "fs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Bgra8UnormSrgb,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        })
    });

    // Pipeline should be returned
    let _ = pipeline;

    // Cache should have one entry
    assert_eq!(cache.len(), 1);
    assert!(!cache.is_empty());

    // Metrics should show one miss
    let metrics = cache.metrics();
    assert_eq!(metrics.misses, 1);
    assert_eq!(metrics.hits, 0);
}

/// Test: RenderPipelineCache get_or_create returns cached pipeline on hit.
#[test]
fn test_cache_get_or_create_hit() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);
    let device = Arc::new(device);

    let cache = RenderPipelineCache::new(device.clone());

    let vs = create_shader_module(&device, MINIMAL_VERTEX_SHADER);
    let fs = create_shader_module(&device, MINIMAL_FRAGMENT_SHADER);
    let layout = create_empty_layout(&device);

    let key = PipelineKey::new(1).with_fragment_shader(2);

    let create_pipeline = || {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("test_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &vs,
                entry_point: "vs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &fs,
                entry_point: "fs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Bgra8UnormSrgb,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        })
    };

    // First call: miss
    let pipeline1 = cache.get_or_create(&key, create_pipeline);

    // Second call with same key: hit
    let mut create_count = 0;
    let pipeline2 = cache.get_or_create(&key, || {
        create_count += 1;
        create_pipeline()
    });

    // Same Arc should be returned
    assert!(Arc::ptr_eq(&pipeline1, &pipeline2));

    // Create function should not have been called
    assert_eq!(create_count, 0);

    // Metrics should show 1 miss and 1 hit
    let metrics = cache.metrics();
    assert_eq!(metrics.misses, 1);
    assert_eq!(metrics.hits, 1);
}

/// Test: RenderPipelineCache invalidate removes pipelines.
#[test]
fn test_cache_invalidate() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);
    let device = Arc::new(device);

    let cache = RenderPipelineCache::new(device.clone());

    let vs = create_shader_module(&device, MINIMAL_VERTEX_SHADER);
    let fs = create_shader_module(&device, MINIMAL_FRAGMENT_SHADER);
    let layout = create_empty_layout(&device);

    let key1 = PipelineKey::new(1).with_fragment_shader(10);
    let key2 = PipelineKey::new(2).with_fragment_shader(10);
    let key3 = PipelineKey::new(3).with_fragment_shader(20);

    let create_pipeline = || {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("test_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &vs,
                entry_point: "vs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &fs,
                entry_point: "fs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Bgra8UnormSrgb,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        })
    };

    // Add three pipelines
    cache.get_or_create(&key1, create_pipeline);
    cache.get_or_create(&key2, create_pipeline);
    cache.get_or_create(&key3, create_pipeline);
    assert_eq!(cache.len(), 3);

    // Invalidate fragment shader 10 (used by key1 and key2)
    let invalidated = cache.invalidate(10);

    assert_eq!(invalidated, 2);
    assert_eq!(cache.len(), 1);

    // key3 should still be in cache
    assert!(cache.contains(&key3));
    assert!(!cache.contains(&key1));
    assert!(!cache.contains(&key2));
}

/// Test: RenderPipelineCache invalidate tracks metrics.
#[test]
fn test_cache_invalidate_metrics() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);
    let device = Arc::new(device);

    let cache = RenderPipelineCache::new(device.clone());

    let vs = create_shader_module(&device, MINIMAL_VERTEX_SHADER);
    let fs = create_shader_module(&device, MINIMAL_FRAGMENT_SHADER);
    let layout = create_empty_layout(&device);

    let key = PipelineKey::new(1).with_fragment_shader(2);

    cache.get_or_create(&key, || {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("test_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &vs,
                entry_point: "vs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &fs,
                entry_point: "fs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Bgra8UnormSrgb,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        })
    });

    cache.invalidate(1);

    let metrics = cache.metrics();
    assert_eq!(metrics.invalidations, 1);
}

/// Test: RenderPipelineCache invalidate preserves unrelated pipelines.
#[test]
fn test_cache_invalidate_preserves_others() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);
    let device = Arc::new(device);

    let cache = RenderPipelineCache::new(device.clone());

    let vs = create_shader_module(&device, MINIMAL_VERTEX_SHADER);
    let fs = create_shader_module(&device, MINIMAL_FRAGMENT_SHADER);
    let layout = create_empty_layout(&device);

    let key1 = PipelineKey::new(100);
    let key2 = PipelineKey::new(200);

    let create_pipeline = || {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("test_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &vs,
                entry_point: "vs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &fs,
                entry_point: "fs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Bgra8UnormSrgb,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        })
    };

    cache.get_or_create(&key1, create_pipeline);
    cache.get_or_create(&key2, create_pipeline);

    // Invalidate shader that doesn't exist
    let invalidated = cache.invalidate(999);

    assert_eq!(invalidated, 0);
    assert_eq!(cache.len(), 2);
}

/// Test: RenderPipelineCache clear removes all pipelines.
#[test]
fn test_cache_clear() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);
    let device = Arc::new(device);

    let cache = RenderPipelineCache::new(device.clone());

    let vs = create_shader_module(&device, MINIMAL_VERTEX_SHADER);
    let fs = create_shader_module(&device, MINIMAL_FRAGMENT_SHADER);
    let layout = create_empty_layout(&device);

    for i in 0..5 {
        let key = PipelineKey::new(i);
        cache.get_or_create(&key, || {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("test_pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &vs,
                    entry_point: "vs_main",
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &fs,
                    entry_point: "fs_main",
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Bgra8UnormSrgb,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview: None,
                cache: None,
            })
        });
    }

    assert_eq!(cache.len(), 5);

    let cleared = cache.clear();
    assert_eq!(cleared, 5);
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());

    let metrics = cache.metrics();
    assert_eq!(metrics.evictions, 5);
}

/// Test: RenderPipelineCache remove single pipeline.
#[test]
fn test_cache_remove() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);
    let device = Arc::new(device);

    let cache = RenderPipelineCache::new(device.clone());

    let vs = create_shader_module(&device, MINIMAL_VERTEX_SHADER);
    let fs = create_shader_module(&device, MINIMAL_FRAGMENT_SHADER);
    let layout = create_empty_layout(&device);

    let key1 = PipelineKey::new(1);
    let key2 = PipelineKey::new(2);

    let create_pipeline = || {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("test_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &vs,
                entry_point: "vs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &fs,
                entry_point: "fs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Bgra8UnormSrgb,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        })
    };

    cache.get_or_create(&key1, create_pipeline);
    cache.get_or_create(&key2, create_pipeline);

    assert!(cache.remove(&key1));
    assert_eq!(cache.len(), 1);
    assert!(!cache.contains(&key1));
    assert!(cache.contains(&key2));

    // Remove again should return false
    assert!(!cache.remove(&key1));
}

/// Test: RenderPipelineCache metrics reset.
#[test]
fn test_cache_reset_metrics() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);
    let device = Arc::new(device);

    let cache = RenderPipelineCache::new(device.clone());

    let vs = create_shader_module(&device, MINIMAL_VERTEX_SHADER);
    let fs = create_shader_module(&device, MINIMAL_FRAGMENT_SHADER);
    let layout = create_empty_layout(&device);

    let key = PipelineKey::new(1);

    let create_pipeline = || {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("test_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &vs,
                entry_point: "vs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &fs,
                entry_point: "fs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Bgra8UnormSrgb,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        })
    };

    cache.get_or_create(&key, create_pipeline);
    cache.get_or_create(&key, create_pipeline);
    cache.clear();

    let metrics = cache.metrics();
    assert!(metrics.misses > 0 || metrics.hits > 0 || metrics.evictions > 0);

    cache.reset_metrics();

    let reset_metrics = cache.metrics();
    assert_eq!(reset_metrics.hits, 0);
    assert_eq!(reset_metrics.misses, 0);
    assert_eq!(reset_metrics.evictions, 0);
    assert_eq!(reset_metrics.invalidations, 0);
}

/// Test: RenderPipelineCache contains method.
#[test]
fn test_cache_contains() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);
    let device = Arc::new(device);

    let cache = RenderPipelineCache::new(device.clone());

    let vs = create_shader_module(&device, MINIMAL_VERTEX_SHADER);
    let fs = create_shader_module(&device, MINIMAL_FRAGMENT_SHADER);
    let layout = create_empty_layout(&device);

    let key = PipelineKey::new(42);
    let other_key = PipelineKey::new(99);

    assert!(!cache.contains(&key));

    cache.get_or_create(&key, || {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("test_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &vs,
                entry_point: "vs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &fs,
                entry_point: "fs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Bgra8UnormSrgb,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        })
    });

    assert!(cache.contains(&key));
    assert!(!cache.contains(&other_key));
}

/// Test: RenderPipelineCache implements Send + Sync.
#[test]
fn test_cache_send_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    assert_send::<RenderPipelineCache>();
    assert_sync::<RenderPipelineCache>();
}

// =============================================================================
// SECTION 5: CONCURRENT ACCESS TESTS
// =============================================================================

/// Test: Concurrent get_or_create with same key.
#[test]
fn test_cache_concurrent_same_key() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);
    let device = Arc::new(device);

    let vs = create_shader_module(&device, MINIMAL_VERTEX_SHADER);
    let fs = create_shader_module(&device, MINIMAL_FRAGMENT_SHADER);
    let layout = Arc::new(create_empty_layout(&device));

    let cache = Arc::new(RenderPipelineCache::new(device.clone()));
    let key = PipelineKey::new(1);

    let num_threads = 4;
    let mut handles = Vec::new();

    for _ in 0..num_threads {
        let cache = Arc::clone(&cache);
        let device = Arc::clone(&device);
        let layout = Arc::clone(&layout);
        let key = key.clone();
        let vs_source = MINIMAL_VERTEX_SHADER;
        let fs_source = MINIMAL_FRAGMENT_SHADER;

        handles.push(thread::spawn(move || {
            let vs = create_shader_module(&device, vs_source);
            let fs = create_shader_module(&device, fs_source);

            cache.get_or_create(&key, || {
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("test_pipeline"),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &vs,
                        entry_point: "vs_main",
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        buffers: &[],
                    },
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    fragment: Some(wgpu::FragmentState {
                        module: &fs,
                        entry_point: "fs_main",
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Bgra8UnormSrgb,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    multiview: None,
                    cache: None,
                })
            })
        }));
    }

    let pipelines: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    // All threads should get the same pipeline
    for pipeline in &pipelines[1..] {
        assert!(Arc::ptr_eq(&pipelines[0], pipeline));
    }

    // Only one pipeline should be created
    assert_eq!(cache.len(), 1);

    // Should have 1 miss and (num_threads - 1) hits
    let metrics = cache.metrics();
    assert_eq!(metrics.misses, 1);
    assert_eq!(metrics.hits as usize, num_threads - 1);
}

/// Test: Concurrent get_or_create with different keys.
#[test]
fn test_cache_concurrent_different_keys() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);
    let device = Arc::new(device);

    let layout = Arc::new(create_empty_layout(&device));
    let cache = Arc::new(RenderPipelineCache::new(device.clone()));

    let num_threads = 4;
    let mut handles = Vec::new();

    for i in 0..num_threads {
        let cache = Arc::clone(&cache);
        let device = Arc::clone(&device);
        let layout = Arc::clone(&layout);
        let vs_source = MINIMAL_VERTEX_SHADER;
        let fs_source = MINIMAL_FRAGMENT_SHADER;

        handles.push(thread::spawn(move || {
            let vs = create_shader_module(&device, vs_source);
            let fs = create_shader_module(&device, fs_source);
            let key = PipelineKey::new(i as u64);

            cache.get_or_create(&key, || {
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("test_pipeline"),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &vs,
                        entry_point: "vs_main",
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        buffers: &[],
                    },
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    fragment: Some(wgpu::FragmentState {
                        module: &fs,
                        entry_point: "fs_main",
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Bgra8UnormSrgb,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    multiview: None,
                    cache: None,
                })
            })
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // Each thread should create its own pipeline
    assert_eq!(cache.len(), num_threads);

    // All should be misses
    let metrics = cache.metrics();
    assert_eq!(metrics.misses as usize, num_threads);
    assert_eq!(metrics.hits, 0);
}

/// Test: Concurrent read access to cache.
#[test]
fn test_cache_concurrent_reads() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);
    let device = Arc::new(device);

    let vs = create_shader_module(&device, MINIMAL_VERTEX_SHADER);
    let fs = create_shader_module(&device, MINIMAL_FRAGMENT_SHADER);
    let layout = create_empty_layout(&device);

    let cache = Arc::new(RenderPipelineCache::new(device.clone()));
    let key = PipelineKey::new(1);

    // Pre-populate cache
    cache.get_or_create(&key, || {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("test_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &vs,
                entry_point: "vs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &fs,
                entry_point: "fs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Bgra8UnormSrgb,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        })
    });

    let num_threads = 8;
    let num_reads_per_thread = 100;
    let mut handles = Vec::new();

    for _ in 0..num_threads {
        let cache = Arc::clone(&cache);
        let key = key.clone();

        handles.push(thread::spawn(move || {
            for _ in 0..num_reads_per_thread {
                assert!(cache.contains(&key));
                assert_eq!(cache.len(), 1);
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // Cache should still have exactly one entry
    assert_eq!(cache.len(), 1);
}

/// Test: Concurrent invalidation safety.
#[test]
fn test_cache_concurrent_invalidation() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);
    let device = Arc::new(device);

    let layout = Arc::new(create_empty_layout(&device));
    let cache = Arc::new(RenderPipelineCache::new(device.clone()));

    // Pre-populate with multiple pipelines
    for i in 0..10 {
        let vs = create_shader_module(&device, MINIMAL_VERTEX_SHADER);
        let fs = create_shader_module(&device, MINIMAL_FRAGMENT_SHADER);
        let key = PipelineKey::new(i);

        cache.get_or_create(&key, || {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("test_pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &vs,
                    entry_point: "vs_main",
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &fs,
                    entry_point: "fs_main",
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Bgra8UnormSrgb,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview: None,
                cache: None,
            })
        });
    }

    assert_eq!(cache.len(), 10);

    // Concurrent invalidation
    let mut handles = Vec::new();
    for i in 0..5 {
        let cache = Arc::clone(&cache);
        handles.push(thread::spawn(move || {
            cache.invalidate(i);
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // Should have 5 pipelines remaining
    assert_eq!(cache.len(), 5);
}

/// Test: Concurrent metrics access.
#[test]
fn test_cache_concurrent_metrics() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);
    let device = Arc::new(device);

    let layout = Arc::new(create_empty_layout(&device));
    let cache = Arc::new(RenderPipelineCache::new(device.clone()));

    let num_threads = 4;
    let mut handles = Vec::new();

    for i in 0..num_threads {
        let cache = Arc::clone(&cache);
        let device = Arc::clone(&device);
        let layout = Arc::clone(&layout);
        let vs_source = MINIMAL_VERTEX_SHADER;
        let fs_source = MINIMAL_FRAGMENT_SHADER;

        handles.push(thread::spawn(move || {
            let vs = create_shader_module(&device, vs_source);
            let fs = create_shader_module(&device, fs_source);
            let key = PipelineKey::new(i as u64);

            cache.get_or_create(&key, || {
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("test_pipeline"),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &vs,
                        entry_point: "vs_main",
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        buffers: &[],
                    },
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    fragment: Some(wgpu::FragmentState {
                        module: &fs,
                        entry_point: "fs_main",
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Bgra8UnormSrgb,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    multiview: None,
                    cache: None,
                })
            });

            // Read metrics concurrently
            let metrics = cache.metrics();
            assert!(metrics.misses + metrics.hits >= 1);
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let final_metrics = cache.metrics();
    assert_eq!(final_metrics.total_accesses(), num_threads as u64);
}

// =============================================================================
// SECTION 6: PERFORMANCE TESTS
// =============================================================================

/// Test: Cache hit is faster than cache miss.
#[test]
fn test_cache_hit_performance() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);
    let device = Arc::new(device);

    let vs = create_shader_module(&device, MINIMAL_VERTEX_SHADER);
    let fs = create_shader_module(&device, MINIMAL_FRAGMENT_SHADER);
    let layout = create_empty_layout(&device);

    let cache = RenderPipelineCache::new(device.clone());
    let key = PipelineKey::new(1);

    let create_pipeline = || {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("test_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &vs,
                entry_point: "vs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &fs,
                entry_point: "fs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Bgra8UnormSrgb,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        })
    };

    // First call: miss (creates pipeline)
    let start_miss = Instant::now();
    cache.get_or_create(&key, create_pipeline);
    let miss_duration = start_miss.elapsed();

    // Second call: hit (returns cached)
    let start_hit = Instant::now();
    for _ in 0..100 {
        cache.get_or_create(&key, create_pipeline);
    }
    let hit_duration = start_hit.elapsed();

    // Cache hit should be much faster than miss
    let avg_hit = hit_duration / 100;
    assert!(
        avg_hit < miss_duration,
        "Cache hit ({:?}) should be faster than miss ({:?})",
        avg_hit,
        miss_duration
    );
}

/// Test: Pipeline creation through cache completes in reasonable time.
#[test]
fn test_cache_creation_performance() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);
    let device = Arc::new(device);

    let vs = create_shader_module(&device, MINIMAL_VERTEX_SHADER);
    let fs = create_shader_module(&device, MINIMAL_FRAGMENT_SHADER);
    let layout = create_empty_layout(&device);

    let cache = RenderPipelineCache::new(device.clone());

    let start = Instant::now();

    for i in 0..10 {
        let key = PipelineKey::new(i);
        cache.get_or_create(&key, || {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("test_pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &vs,
                    entry_point: "vs_main",
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &fs,
                    entry_point: "fs_main",
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Bgra8UnormSrgb,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview: None,
                cache: None,
            })
        });
    }

    let duration = start.elapsed();

    // 10 pipeline creations should complete in under 5 seconds
    assert!(
        duration.as_secs() < 5,
        "Creating 10 pipelines took too long: {:?}",
        duration
    );
}

// =============================================================================
// SECTION 7: EDGE CASES
// =============================================================================

/// Test: Cache with zero-ID shader.
#[test]
fn test_cache_zero_shader_id() {
    let key = PipelineKey::new(0);

    assert_eq!(key.vertex_shader_id, 0);
    assert!(key.uses_shader(0));
}

/// Test: Cache with maximum shader ID.
#[test]
fn test_cache_max_shader_id() {
    let key = PipelineKey::new(u64::MAX).with_fragment_shader(u64::MAX - 1);

    assert_eq!(key.vertex_shader_id, u64::MAX);
    assert_eq!(key.fragment_shader_id, Some(u64::MAX - 1));
    assert!(key.uses_shader(u64::MAX));
    assert!(key.uses_shader(u64::MAX - 1));
}

/// Test: Empty cache metrics.
#[test]
fn test_cache_empty_metrics() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);
    let device = Arc::new(device);

    let cache = RenderPipelineCache::new(device);

    let metrics = cache.metrics();
    assert_eq!(metrics.hits, 0);
    assert_eq!(metrics.misses, 0);
    assert_eq!(metrics.evictions, 0);
    assert_eq!(metrics.invalidations, 0);
    assert_eq!(metrics.hit_rate(), 0.0);
}

/// Test: Invalidate on empty cache.
#[test]
fn test_cache_invalidate_empty() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);
    let device = Arc::new(device);

    let cache = RenderPipelineCache::new(device);

    let invalidated = cache.invalidate(1);
    assert_eq!(invalidated, 0);
    assert_eq!(cache.len(), 0);
}

/// Test: Clear on empty cache.
#[test]
fn test_cache_clear_empty() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);
    let device = Arc::new(device);

    let cache = RenderPipelineCache::new(device);

    let cleared = cache.clear();
    assert_eq!(cleared, 0);
    assert_eq!(cache.len(), 0);
}

/// Test: Remove from empty cache.
#[test]
fn test_cache_remove_from_empty() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);
    let device = Arc::new(device);

    let cache = RenderPipelineCache::new(device);

    let key = PipelineKey::new(1);
    assert!(!cache.remove(&key));
}

/// Test: Multiple invalidations of same shader.
#[test]
fn test_cache_multiple_invalidations_same_shader() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);
    let device = Arc::new(device);

    let cache = RenderPipelineCache::new(device.clone());

    let vs = create_shader_module(&device, MINIMAL_VERTEX_SHADER);
    let fs = create_shader_module(&device, MINIMAL_FRAGMENT_SHADER);
    let layout = create_empty_layout(&device);

    let key = PipelineKey::new(1);

    cache.get_or_create(&key, || {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("test_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &vs,
                entry_point: "vs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &fs,
                entry_point: "fs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Bgra8UnormSrgb,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        })
    });

    // First invalidation
    let first = cache.invalidate(1);
    assert_eq!(first, 1);

    // Second invalidation of same shader
    let second = cache.invalidate(1);
    assert_eq!(second, 0);
}
