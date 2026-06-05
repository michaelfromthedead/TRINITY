// SPDX-License-Identifier: MIT
//
// blackbox_cache_warming.rs -- Blackbox tests for T-WGPU-P3.1.8 Cache Warming.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only the documented public types and functions:
//
//   - WarmingConfig -- Configuration for cache warming behavior
//   - WarmingProgress -- Progress information during warming
//   - WarmingResult -- Result of a warming operation
//   - WarmingHandle -- Handle for async warming operations
//   - ProgressCallback -- Callback type for progress updates
//   - common_pipelines -- Preset pipeline key generators
//   - RenderPipelineCache::warm_cache -- Synchronous warming
//   - RenderPipelineCache::warm_cache_async -- Asynchronous warming
//
// ACCEPTANCE CRITERIA:
//   1. API surface tests -- All public types accessible (10+ tests)
//   2. Real warming tests -- With wgpu device (10+ tests)
//   3. Progress callback tests -- Callback verification (5+ tests)
//   4. Background warming tests -- Async behavior (5+ tests)
//   5. Common pipelines integration -- Preset usage (5+ tests)
//
// Total target: 35+ tests

use pollster::block_on;
use renderer_backend::device::{enumerate_adapters_with_info, TrinityInstance};
use renderer_backend::render_pipeline::{
    common_pipelines, PipelineKey, ProgressCallback, RenderPipelineCache, WarmingConfig,
    WarmingHandle, WarmingProgress, WarmingResult,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

// =============================================================================
// TEST SHADERS
// =============================================================================

/// Minimal vertex shader for pipeline creation tests.
const MINIMAL_VERTEX_SHADER: &str = r#"
@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
"#;

/// Minimal fragment shader for pipeline creation tests.
const MINIMAL_FRAGMENT_SHADER: &str = r#"
@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
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

// =============================================================================
// SECTION 1: API SURFACE TESTS (No GPU Required)
// =============================================================================

/// Test: All cache warming public types are accessible.
#[test]
fn test_cache_warming_api_surface() {
    // Verify all public types are accessible (compile-time check)
    let _: fn() -> WarmingConfig = WarmingConfig::default;
    let _: fn() -> WarmingConfig = WarmingConfig::background;

    // Verify WarmingProgress fields are accessible
    let progress = WarmingProgress {
        completed: 0,
        total: 10,
        current_key: None,
    };
    let _ = progress.completed;
    let _ = progress.total;
    let _ = progress.current_key;

    // Verify WarmingResult fields are accessible
    let result = WarmingResult {
        warmed: 5,
        skipped: 3,
        failed: 0,
        duration: Duration::ZERO,
    };
    let _ = result.warmed;
    let _ = result.skipped;
    let _ = result.failed;
    let _ = result.duration;
}

/// Test: WarmingConfig default values.
#[test]
fn test_warming_config_default() {
    let config = WarmingConfig::default();

    assert!(!config.background);
    assert!(config.parallelism >= 1);
}

/// Test: WarmingConfig background constructor.
#[test]
fn test_warming_config_background() {
    let config = WarmingConfig::background();

    assert!(config.background);
    assert!(config.parallelism >= 1);
}

/// Test: WarmingConfig with_parallelism builder.
#[test]
fn test_warming_config_with_parallelism() {
    let config = WarmingConfig::default().with_parallelism(8);
    assert_eq!(config.parallelism, 8);

    // Zero should be clamped to 1
    let config_zero = WarmingConfig::default().with_parallelism(0);
    assert_eq!(config_zero.parallelism, 1);
}

/// Test: WarmingConfig parallelism builder chaining.
#[test]
fn test_warming_config_builder_chain() {
    let config = WarmingConfig::background().with_parallelism(4);

    assert!(config.background);
    assert_eq!(config.parallelism, 4);
}

/// Test: WarmingProgress fraction calculation.
#[test]
fn test_warming_progress_fraction() {
    // 0/10 = 0.0
    let p0 = WarmingProgress {
        completed: 0,
        total: 10,
        current_key: None,
    };
    assert_eq!(p0.fraction(), 0.0);

    // 5/10 = 0.5
    let p50 = WarmingProgress {
        completed: 5,
        total: 10,
        current_key: None,
    };
    assert!((p50.fraction() - 0.5).abs() < f64::EPSILON);

    // 10/10 = 1.0
    let p100 = WarmingProgress {
        completed: 10,
        total: 10,
        current_key: None,
    };
    assert_eq!(p100.fraction(), 1.0);

    // 0/0 = 1.0 (empty set is complete)
    let p_empty = WarmingProgress {
        completed: 0,
        total: 0,
        current_key: None,
    };
    assert_eq!(p_empty.fraction(), 1.0);
}

/// Test: WarmingProgress percent calculation.
#[test]
fn test_warming_progress_percent() {
    let progress = WarmingProgress {
        completed: 3,
        total: 4,
        current_key: None,
    };
    assert_eq!(progress.percent(), 75);

    let progress_full = WarmingProgress {
        completed: 10,
        total: 10,
        current_key: None,
    };
    assert_eq!(progress_full.percent(), 100);

    let progress_zero = WarmingProgress {
        completed: 0,
        total: 10,
        current_key: None,
    };
    assert_eq!(progress_zero.percent(), 0);
}

/// Test: WarmingProgress with current key.
#[test]
fn test_warming_progress_with_current_key() {
    let key = PipelineKey::new(42);
    let progress = WarmingProgress {
        completed: 1,
        total: 5,
        current_key: Some(key.clone()),
    };

    assert!(progress.current_key.is_some());
    assert_eq!(progress.current_key.unwrap().vertex_shader_id, 42);
}

/// Test: WarmingResult total calculation.
#[test]
fn test_warming_result_total() {
    let result = WarmingResult {
        warmed: 5,
        skipped: 3,
        failed: 2,
        duration: Duration::from_secs(1),
    };

    assert_eq!(result.total(), 10);
}

/// Test: WarmingResult is_success with no failures.
#[test]
fn test_warming_result_is_success() {
    let success = WarmingResult {
        warmed: 5,
        skipped: 5,
        failed: 0,
        duration: Duration::ZERO,
    };
    assert!(success.is_success());

    let failure = WarmingResult {
        warmed: 5,
        skipped: 3,
        failed: 2,
        duration: Duration::ZERO,
    };
    assert!(!failure.is_success());
}

/// Test: WarmingResult empty case.
#[test]
fn test_warming_result_empty() {
    let result = WarmingResult {
        warmed: 0,
        skipped: 0,
        failed: 0,
        duration: Duration::ZERO,
    };

    assert_eq!(result.total(), 0);
    assert!(result.is_success());
}

/// Test: WarmingConfig implements Send + Sync.
#[test]
fn test_warming_config_send_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    assert_send::<WarmingConfig>();
    assert_sync::<WarmingConfig>();
}

/// Test: WarmingProgress implements Send + Sync.
#[test]
fn test_warming_progress_send_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    assert_send::<WarmingProgress>();
    assert_sync::<WarmingProgress>();
}

/// Test: WarmingResult implements Send + Sync.
#[test]
fn test_warming_result_send_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    assert_send::<WarmingResult>();
    assert_sync::<WarmingResult>();
}

/// Test: WarmingHandle implements Send.
#[test]
fn test_warming_handle_send() {
    fn assert_send<T: Send>() {}

    assert_send::<WarmingHandle>();
}

/// Test: WarmingConfig Clone.
#[test]
fn test_warming_config_clone() {
    let config = WarmingConfig::background().with_parallelism(16);
    let cloned = config.clone();

    assert_eq!(config.background, cloned.background);
    assert_eq!(config.parallelism, cloned.parallelism);
}

/// Test: WarmingConfig Debug.
#[test]
fn test_warming_config_debug() {
    let config = WarmingConfig::default();
    let debug_str = format!("{:?}", config);

    assert!(debug_str.contains("WarmingConfig"));
    assert!(debug_str.contains("background"));
    assert!(debug_str.contains("parallelism"));
}

/// Test: WarmingProgress Clone.
#[test]
fn test_warming_progress_clone() {
    let key = PipelineKey::new(1);
    let progress = WarmingProgress {
        completed: 5,
        total: 10,
        current_key: Some(key),
    };
    let cloned = progress.clone();

    assert_eq!(progress.completed, cloned.completed);
    assert_eq!(progress.total, cloned.total);
    assert!(cloned.current_key.is_some());
}

/// Test: WarmingProgress Debug.
#[test]
fn test_warming_progress_debug() {
    let progress = WarmingProgress {
        completed: 3,
        total: 10,
        current_key: None,
    };
    let debug_str = format!("{:?}", progress);

    assert!(debug_str.contains("WarmingProgress"));
    assert!(debug_str.contains("completed"));
    assert!(debug_str.contains("total"));
}

/// Test: WarmingResult Clone.
#[test]
fn test_warming_result_clone() {
    let result = WarmingResult {
        warmed: 5,
        skipped: 3,
        failed: 1,
        duration: Duration::from_millis(100),
    };
    let cloned = result.clone();

    assert_eq!(result.warmed, cloned.warmed);
    assert_eq!(result.skipped, cloned.skipped);
    assert_eq!(result.failed, cloned.failed);
    assert_eq!(result.duration, cloned.duration);
}

/// Test: WarmingResult Debug.
#[test]
fn test_warming_result_debug() {
    let result = WarmingResult {
        warmed: 5,
        skipped: 3,
        failed: 0,
        duration: Duration::from_secs(1),
    };
    let debug_str = format!("{:?}", result);

    assert!(debug_str.contains("WarmingResult"));
    assert!(debug_str.contains("warmed"));
    assert!(debug_str.contains("skipped"));
}

// =============================================================================
// SECTION 2: COMMON PIPELINES TESTS (No GPU Required)
// =============================================================================

/// Test: common_pipelines::pbr_forward produces valid key.
#[test]
fn test_common_pipelines_pbr_forward() {
    let key = common_pipelines::pbr_forward(1, 2);

    assert_eq!(key.vertex_shader_id, 1);
    assert_eq!(key.fragment_shader_id, Some(2));
    assert_eq!(key.cull_mode, Some(wgpu::Face::Back));
    assert_eq!(key.depth_format, Some(wgpu::TextureFormat::Depth32Float));
    assert!(key.depth_write);
    assert_eq!(key.sample_count, 4);
}

/// Test: common_pipelines::shadow_map produces valid key.
#[test]
fn test_common_pipelines_shadow_map() {
    let key = common_pipelines::shadow_map(1);

    assert_eq!(key.vertex_shader_id, 1);
    assert_eq!(key.fragment_shader_id, None);
    assert_eq!(key.cull_mode, Some(wgpu::Face::Front));
    assert_eq!(key.depth_format, Some(wgpu::TextureFormat::Depth32Float));
    assert!(key.depth_write);
}

/// Test: common_pipelines::ui produces valid key.
#[test]
fn test_common_pipelines_ui() {
    let key = common_pipelines::ui(1, 2);

    assert_eq!(key.vertex_shader_id, 1);
    assert_eq!(key.fragment_shader_id, Some(2));
    assert_eq!(key.cull_mode, None);
    assert!(!key.depth_write);
}

/// Test: common_pipelines::skybox produces valid key.
#[test]
fn test_common_pipelines_skybox() {
    let key = common_pipelines::skybox(1, 2);

    assert_eq!(key.vertex_shader_id, 1);
    assert_eq!(key.fragment_shader_id, Some(2));
    assert_eq!(key.cull_mode, None);
    assert_eq!(key.depth_format, Some(wgpu::TextureFormat::Depth32Float));
    assert!(!key.depth_write);
    assert_eq!(key.depth_compare, wgpu::CompareFunction::LessEqual);
}

/// Test: common_pipelines::particle produces valid key.
#[test]
fn test_common_pipelines_particle() {
    let key = common_pipelines::particle(1, 2);

    assert_eq!(key.vertex_shader_id, 1);
    assert_eq!(key.fragment_shader_id, Some(2));
    assert_eq!(key.cull_mode, None);
    assert_eq!(key.depth_format, Some(wgpu::TextureFormat::Depth32Float));
    assert!(!key.depth_write);
}

/// Test: common_pipelines::fullscreen_quad produces valid key.
#[test]
fn test_common_pipelines_fullscreen_quad() {
    let key = common_pipelines::fullscreen_quad(1, 2);

    assert_eq!(key.vertex_shader_id, 1);
    assert_eq!(key.fragment_shader_id, Some(2));
    assert_eq!(key.cull_mode, None);
    assert!(!key.depth_write);
}

/// Test: All common_pipelines produce unique keys.
#[test]
fn test_common_pipelines_unique_keys() {
    let pbr = common_pipelines::pbr_forward(1, 2);
    let shadow = common_pipelines::shadow_map(1);
    let ui = common_pipelines::ui(1, 2);
    let skybox = common_pipelines::skybox(1, 2);
    let particle = common_pipelines::particle(1, 2);
    let fullscreen = common_pipelines::fullscreen_quad(1, 2);

    // All should be different from each other
    assert_ne!(pbr, shadow);
    assert_ne!(pbr, ui);
    assert_ne!(pbr, skybox);
    assert_ne!(pbr, particle);
    assert_ne!(pbr, fullscreen);
    assert_ne!(shadow, ui);
    assert_ne!(shadow, skybox);
    assert_ne!(ui, skybox);
    assert_ne!(particle, fullscreen);
}

/// Test: common_pipelines with different shader IDs produce different keys.
#[test]
fn test_common_pipelines_different_shader_ids() {
    let key1 = common_pipelines::pbr_forward(1, 2);
    let key2 = common_pipelines::pbr_forward(3, 4);
    let key3 = common_pipelines::pbr_forward(1, 3);

    assert_ne!(key1, key2);
    assert_ne!(key1, key3);
    assert_ne!(key2, key3);
}

// =============================================================================
// SECTION 3: WARMING HANDLE TESTS (via warm_cache_async - GPU Required)
// =============================================================================

// Note: WarmingHandle cannot be constructed directly (private field).
// These tests verify WarmingHandle behavior through the warm_cache_async API.

/// Test: WarmingHandle is_finished returns true after completion (via async warming).
#[test]
fn test_warming_handle_is_finished_via_async() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);
    let device = Arc::new(device);

    let cache = Arc::new(RenderPipelineCache::new(device.clone()));

    // Empty warming should complete quickly
    let handle = cache.warm_cache_async(
        vec![],
        |_| panic!("Should not be called"),
        None,
    );

    // Wait a bit for the thread to complete
    std::thread::sleep(Duration::from_millis(50));
    assert!(handle.is_finished());
}

/// Test: WarmingHandle join returns result (via async warming).
#[test]
fn test_warming_handle_join_via_async() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);
    let device = Arc::new(device);

    let vs = Arc::new(create_shader_module(&device, MINIMAL_VERTEX_SHADER));
    let fs = Arc::new(create_shader_module(&device, MINIMAL_FRAGMENT_SHADER));
    let layout = Arc::new(create_empty_layout(&device));

    let cache = Arc::new(RenderPipelineCache::new(device.clone()));

    let keys = vec![PipelineKey::new(1), PipelineKey::new(2)];

    let vs_clone = Arc::clone(&vs);
    let fs_clone = Arc::clone(&fs);
    let layout_clone = Arc::clone(&layout);
    let device_clone = Arc::clone(&device);

    let handle = cache.warm_cache_async(
        keys,
        move |_| {
            device_clone.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("test_pipeline"),
                layout: Some(&layout_clone),
                vertex: wgpu::VertexState {
                    module: &vs_clone,
                    entry_point: "vs_main",
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &fs_clone,
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
        },
        None,
    );

    let result = handle.join();

    assert_eq!(result.warmed, 2);
    assert_eq!(result.skipped, 0);
    assert!(result.is_success());
}

/// Test: WarmingHandle join returns default on empty keys.
#[test]
fn test_warming_handle_join_empty() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);
    let device = Arc::new(device);

    let cache = Arc::new(RenderPipelineCache::new(device.clone()));

    let handle = cache.warm_cache_async(
        vec![],
        |_| panic!("Should not be called"),
        None,
    );

    let result = handle.join();

    // Empty warming should return zeros
    assert_eq!(result.warmed, 0);
    assert_eq!(result.skipped, 0);
    assert_eq!(result.failed, 0);
    assert!(result.is_success());
}

// =============================================================================
// SECTION 4: REAL WARMING TESTS (GPU Required)
// =============================================================================

/// Test: warm_cache with empty array completes immediately.
#[test]
fn test_warm_cache_empty() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);
    let device = Arc::new(device);

    let cache = RenderPipelineCache::new(device.clone());

    let start = Instant::now();
    let result = cache.warm_cache(
        &[],
        |_key| panic!("Should not be called for empty array"),
        WarmingConfig::default(),
        None,
    );
    let elapsed = start.elapsed();

    assert_eq!(result.warmed, 0);
    assert_eq!(result.skipped, 0);
    assert_eq!(result.failed, 0);
    assert!(result.is_success());
    // Should be nearly instant
    assert!(elapsed < Duration::from_millis(100));
}

/// Test: warm_cache with single pipeline.
#[test]
fn test_warm_cache_single_pipeline() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);
    let device = Arc::new(device);

    let vs = create_shader_module(&device, MINIMAL_VERTEX_SHADER);
    let fs = create_shader_module(&device, MINIMAL_FRAGMENT_SHADER);
    let layout = create_empty_layout(&device);

    let cache = RenderPipelineCache::new(device.clone());

    let key = PipelineKey::new(1).with_fragment_shader(2);

    let result = cache.warm_cache(
        &[key.clone()],
        |_| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("warmed_pipeline"),
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
        },
        WarmingConfig::default(),
        None,
    );

    assert_eq!(result.warmed, 1);
    assert_eq!(result.skipped, 0);
    assert!(result.is_success());

    // Verify pipeline is in cache
    assert!(cache.contains(&key));
    assert_eq!(cache.len(), 1);
}

/// Test: warm_cache with multiple pipelines.
#[test]
fn test_warm_cache_multiple_pipelines() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);
    let device = Arc::new(device);

    let vs = create_shader_module(&device, MINIMAL_VERTEX_SHADER);
    let fs = create_shader_module(&device, MINIMAL_FRAGMENT_SHADER);
    let layout = create_empty_layout(&device);

    let cache = RenderPipelineCache::new(device.clone());

    let keys: Vec<PipelineKey> = (0..5).map(|i| PipelineKey::new(i)).collect();

    let result = cache.warm_cache(
        &keys,
        |_| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("warmed_pipeline"),
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
        },
        WarmingConfig::default(),
        None,
    );

    assert_eq!(result.warmed, 5);
    assert_eq!(result.skipped, 0);
    assert!(result.is_success());
    assert_eq!(cache.len(), 5);

    // Verify all are in cache
    for key in &keys {
        assert!(cache.contains(key));
    }
}

/// Test: warm_cache skips existing pipelines.
#[test]
fn test_warm_cache_skips_existing() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);
    let device = Arc::new(device);

    let vs = create_shader_module(&device, MINIMAL_VERTEX_SHADER);
    let fs = create_shader_module(&device, MINIMAL_FRAGMENT_SHADER);
    let layout = create_empty_layout(&device);

    let cache = RenderPipelineCache::new(device.clone());

    let key1 = PipelineKey::new(1);
    let key2 = PipelineKey::new(2);
    let key3 = PipelineKey::new(3);

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

    // Pre-populate cache with key1 and key2
    cache.get_or_create(&key1, create_pipeline);
    cache.get_or_create(&key2, create_pipeline);

    // Warm with all three keys
    let result = cache.warm_cache(
        &[key1.clone(), key2.clone(), key3.clone()],
        |_| create_pipeline(),
        WarmingConfig::default(),
        None,
    );

    // Should have warmed 1 and skipped 2
    assert_eq!(result.warmed, 1);
    assert_eq!(result.skipped, 2);
    assert_eq!(result.total(), 3);
    assert!(result.is_success());
}

/// Test: warm_cache tracks duration.
#[test]
fn test_warm_cache_duration() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);
    let device = Arc::new(device);

    let vs = create_shader_module(&device, MINIMAL_VERTEX_SHADER);
    let fs = create_shader_module(&device, MINIMAL_FRAGMENT_SHADER);
    let layout = create_empty_layout(&device);

    let cache = RenderPipelineCache::new(device.clone());

    let key = PipelineKey::new(1);

    let result = cache.warm_cache(
        &[key],
        |_| {
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
        },
        WarmingConfig::default(),
        None,
    );

    // Duration should be positive
    assert!(result.duration > Duration::ZERO);
}

// =============================================================================
// SECTION 5: PROGRESS CALLBACK TESTS (GPU Required)
// =============================================================================

/// Test: warm_cache calls progress callback.
#[test]
fn test_warm_cache_progress_callback() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);
    let device = Arc::new(device);

    let vs = create_shader_module(&device, MINIMAL_VERTEX_SHADER);
    let fs = create_shader_module(&device, MINIMAL_FRAGMENT_SHADER);
    let layout = create_empty_layout(&device);

    let cache = RenderPipelineCache::new(device.clone());

    let keys: Vec<PipelineKey> = (0..3).map(|i| PipelineKey::new(i)).collect();

    let callback_count = Arc::new(AtomicUsize::new(0));
    let callback_count_clone = Arc::clone(&callback_count);

    let progress_callback: ProgressCallback = Box::new(move |progress| {
        callback_count_clone.fetch_add(1, Ordering::SeqCst);
        assert!(progress.completed <= progress.total);
        assert_eq!(progress.total, 3);
    });

    let result = cache.warm_cache(
        &keys,
        |_| {
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
        },
        WarmingConfig::default(),
        Some(progress_callback),
    );

    assert!(result.is_success());
    // Callback should be called once per key + final completion = 4 times
    assert_eq!(callback_count.load(Ordering::SeqCst), 4);
}

/// Test: Progress callback receives correct progress values.
#[test]
fn test_warm_cache_progress_values() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);
    let device = Arc::new(device);

    let vs = create_shader_module(&device, MINIMAL_VERTEX_SHADER);
    let fs = create_shader_module(&device, MINIMAL_FRAGMENT_SHADER);
    let layout = create_empty_layout(&device);

    let cache = RenderPipelineCache::new(device.clone());

    let keys: Vec<PipelineKey> = (0..5).map(|i| PipelineKey::new(i)).collect();

    let progress_values = Arc::new(std::sync::Mutex::new(Vec::new()));
    let progress_values_clone = Arc::clone(&progress_values);

    let progress_callback: ProgressCallback = Box::new(move |progress| {
        progress_values_clone.lock().unwrap().push((
            progress.completed,
            progress.total,
            progress.current_key.is_some(),
        ));
    });

    cache.warm_cache(
        &keys,
        |_| {
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
        },
        WarmingConfig::default(),
        Some(progress_callback),
    );

    let values = progress_values.lock().unwrap();
    assert_eq!(values.len(), 6); // 5 keys + final

    // Check progression
    for i in 0..5 {
        assert_eq!(values[i].0, i); // completed
        assert_eq!(values[i].1, 5); // total
        assert!(values[i].2); // has current key
    }

    // Final callback
    assert_eq!(values[5].0, 5); // completed
    assert_eq!(values[5].1, 5); // total
    assert!(!values[5].2); // no current key
}

/// Test: Progress callback receives current key.
#[test]
fn test_warm_cache_progress_current_key() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);
    let device = Arc::new(device);

    let vs = create_shader_module(&device, MINIMAL_VERTEX_SHADER);
    let fs = create_shader_module(&device, MINIMAL_FRAGMENT_SHADER);
    let layout = create_empty_layout(&device);

    let cache = RenderPipelineCache::new(device.clone());

    let keys: Vec<PipelineKey> = (100..103).map(|i| PipelineKey::new(i)).collect();

    let seen_keys = Arc::new(std::sync::Mutex::new(Vec::new()));
    let seen_keys_clone = Arc::clone(&seen_keys);

    let progress_callback: ProgressCallback = Box::new(move |progress| {
        if let Some(key) = &progress.current_key {
            seen_keys_clone.lock().unwrap().push(key.vertex_shader_id);
        }
    });

    cache.warm_cache(
        &keys,
        |_| {
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
        },
        WarmingConfig::default(),
        Some(progress_callback),
    );

    let seen = seen_keys.lock().unwrap();
    assert_eq!(seen.len(), 3);
    assert_eq!(seen[0], 100);
    assert_eq!(seen[1], 101);
    assert_eq!(seen[2], 102);
}

// =============================================================================
// SECTION 6: ASYNC WARMING TESTS (GPU Required)
// =============================================================================

/// Test: warm_cache_async completes and returns result.
#[test]
fn test_warm_cache_async_completion() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);
    let device = Arc::new(device);

    let vs = Arc::new(create_shader_module(&device, MINIMAL_VERTEX_SHADER));
    let fs = Arc::new(create_shader_module(&device, MINIMAL_FRAGMENT_SHADER));
    let layout = Arc::new(create_empty_layout(&device));

    let cache = Arc::new(RenderPipelineCache::new(device.clone()));

    let keys: Vec<PipelineKey> = (0..3).map(|i| PipelineKey::new(i)).collect();

    let vs_clone = Arc::clone(&vs);
    let fs_clone = Arc::clone(&fs);
    let layout_clone = Arc::clone(&layout);
    let device_clone = Arc::clone(&device);

    let handle = cache.warm_cache_async(
        keys.clone(),
        move |_| {
            device_clone.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("async_warmed_pipeline"),
                layout: Some(&layout_clone),
                vertex: wgpu::VertexState {
                    module: &vs_clone,
                    entry_point: "vs_main",
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &fs_clone,
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
        },
        None,
    );

    let result = handle.join();

    assert_eq!(result.warmed, 3);
    assert_eq!(result.skipped, 0);
    assert!(result.is_success());

    // Verify all are in cache
    for key in &keys {
        assert!(cache.contains(key));
    }
}

/// Test: warm_cache_async is_finished behavior.
#[test]
fn test_warm_cache_async_is_finished() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);
    let device = Arc::new(device);

    let cache = Arc::new(RenderPipelineCache::new(device.clone()));

    // Empty warming should complete quickly
    let handle = cache.warm_cache_async(
        vec![],
        |_| panic!("Should not be called"),
        None,
    );

    // Wait a bit for completion
    std::thread::sleep(Duration::from_millis(50));
    assert!(handle.is_finished());

    let result = handle.join();
    assert_eq!(result.total(), 0);
}

/// Test: warm_cache_async with progress callback.
#[test]
fn test_warm_cache_async_with_progress() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);
    let device = Arc::new(device);

    let vs = Arc::new(create_shader_module(&device, MINIMAL_VERTEX_SHADER));
    let fs = Arc::new(create_shader_module(&device, MINIMAL_FRAGMENT_SHADER));
    let layout = Arc::new(create_empty_layout(&device));

    let cache = Arc::new(RenderPipelineCache::new(device.clone()));

    let keys: Vec<PipelineKey> = (0..5).map(|i| PipelineKey::new(i)).collect();

    let callback_count = Arc::new(AtomicUsize::new(0));
    let callback_count_clone = Arc::clone(&callback_count);

    let progress_callback: ProgressCallback = Box::new(move |progress| {
        callback_count_clone.fetch_add(1, Ordering::SeqCst);
        assert!(progress.completed <= progress.total);
    });

    let vs_clone = Arc::clone(&vs);
    let fs_clone = Arc::clone(&fs);
    let layout_clone = Arc::clone(&layout);
    let device_clone = Arc::clone(&device);

    let handle = cache.warm_cache_async(
        keys,
        move |_| {
            device_clone.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("async_pipeline"),
                layout: Some(&layout_clone),
                vertex: wgpu::VertexState {
                    module: &vs_clone,
                    entry_point: "vs_main",
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &fs_clone,
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
        },
        Some(progress_callback),
    );

    let result = handle.join();
    assert!(result.is_success());

    // Callback should be called multiple times
    assert!(callback_count.load(Ordering::SeqCst) > 0);
}

/// Test: warm_cache_async returns immediately without blocking.
#[test]
fn test_warm_cache_async_non_blocking() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);
    let device = Arc::new(device);

    let vs = Arc::new(create_shader_module(&device, MINIMAL_VERTEX_SHADER));
    let fs = Arc::new(create_shader_module(&device, MINIMAL_FRAGMENT_SHADER));
    let layout = Arc::new(create_empty_layout(&device));

    let cache = Arc::new(RenderPipelineCache::new(device.clone()));

    let keys: Vec<PipelineKey> = (0..10).map(|i| PipelineKey::new(i)).collect();

    let vs_clone = Arc::clone(&vs);
    let fs_clone = Arc::clone(&fs);
    let layout_clone = Arc::clone(&layout);
    let device_clone = Arc::clone(&device);

    let start = Instant::now();
    let _handle = cache.warm_cache_async(
        keys,
        move |_| {
            // Simulate slow compilation
            std::thread::sleep(Duration::from_millis(10));
            device_clone.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("slow_pipeline"),
                layout: Some(&layout_clone),
                vertex: wgpu::VertexState {
                    module: &vs_clone,
                    entry_point: "vs_main",
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &fs_clone,
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
        },
        None,
    );
    let elapsed = start.elapsed();

    // Should return almost immediately (< 50ms) even though warming takes ~100ms total
    assert!(
        elapsed < Duration::from_millis(50),
        "warm_cache_async took too long: {:?}",
        elapsed
    );
}

// =============================================================================
// SECTION 7: INTEGRATION WITH COMMON PIPELINES
// =============================================================================

/// Test: Warm common pipelines using presets.
#[test]
fn test_warm_common_pipelines() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);
    let device = Arc::new(device);

    let vs = create_shader_module(&device, MINIMAL_VERTEX_SHADER);
    let fs = create_shader_module(&device, MINIMAL_FRAGMENT_SHADER);
    let layout = create_empty_layout(&device);

    let cache = RenderPipelineCache::new(device.clone());

    // Build array of common pipeline keys
    let keys = vec![
        common_pipelines::pbr_forward(1, 2),
        common_pipelines::shadow_map(1),
        common_pipelines::ui(1, 2),
        common_pipelines::skybox(1, 2),
        common_pipelines::particle(1, 2),
        common_pipelines::fullscreen_quad(1, 2),
    ];

    let result = cache.warm_cache(
        &keys,
        |_| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("common_pipeline"),
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
        },
        WarmingConfig::default(),
        None,
    );

    assert_eq!(result.warmed, 6);
    assert!(result.is_success());
    assert_eq!(cache.len(), 6);
}

/// Test: Warm with mixed common and custom keys.
#[test]
fn test_warm_mixed_pipeline_keys() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);
    let device = Arc::new(device);

    let vs = create_shader_module(&device, MINIMAL_VERTEX_SHADER);
    let fs = create_shader_module(&device, MINIMAL_FRAGMENT_SHADER);
    let layout = create_empty_layout(&device);

    let cache = RenderPipelineCache::new(device.clone());

    let keys = vec![
        common_pipelines::pbr_forward(1, 2),
        PipelineKey::new(100).with_fragment_shader(200),
        common_pipelines::shadow_map(3),
        PipelineKey::new(300).with_sample_count(4),
    ];

    let result = cache.warm_cache(
        &keys,
        |_| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("mixed_pipeline"),
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
        },
        WarmingConfig::default(),
        None,
    );

    assert_eq!(result.warmed, 4);
    assert!(result.is_success());
}

// =============================================================================
// SECTION 8: EDGE CASES
// =============================================================================

/// Test: Warm with duplicate keys.
#[test]
fn test_warm_cache_duplicate_keys() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);
    let device = Arc::new(device);

    let vs = create_shader_module(&device, MINIMAL_VERTEX_SHADER);
    let fs = create_shader_module(&device, MINIMAL_FRAGMENT_SHADER);
    let layout = create_empty_layout(&device);

    let cache = RenderPipelineCache::new(device.clone());

    let key = PipelineKey::new(1);
    // Same key three times
    let keys = vec![key.clone(), key.clone(), key.clone()];

    let result = cache.warm_cache(
        &keys,
        |_| {
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
        },
        WarmingConfig::default(),
        None,
    );

    // First creates, subsequent skip
    assert_eq!(result.warmed, 1);
    assert_eq!(result.skipped, 2);
    assert_eq!(result.total(), 3);
    assert!(result.is_success());
    assert_eq!(cache.len(), 1);
}

/// Test: Warm updates cache metrics properly.
#[test]
fn test_warm_cache_metrics() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);
    let device = Arc::new(device);

    let vs = create_shader_module(&device, MINIMAL_VERTEX_SHADER);
    let fs = create_shader_module(&device, MINIMAL_FRAGMENT_SHADER);
    let layout = create_empty_layout(&device);

    let cache = RenderPipelineCache::new(device.clone());

    let keys: Vec<PipelineKey> = (0..5).map(|i| PipelineKey::new(i)).collect();

    cache.warm_cache(
        &keys,
        |_| {
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
        },
        WarmingConfig::default(),
        None,
    );

    // Get a cached pipeline to trigger a hit
    let _ = cache.get_or_create(&keys[0], || panic!("Should be cached"));

    let metrics = cache.metrics();
    // 5 misses from warming (via get_or_create) + 1 hit from get_or_create
    assert!(metrics.misses >= 5);
    assert!(metrics.hits >= 1);
}

/// Test: Concurrent warming and access.
#[test]
fn test_warm_cache_concurrent_access() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);
    let device = Arc::new(device);

    let vs = Arc::new(create_shader_module(&device, MINIMAL_VERTEX_SHADER));
    let fs = Arc::new(create_shader_module(&device, MINIMAL_FRAGMENT_SHADER));
    let layout = Arc::new(create_empty_layout(&device));

    let cache = Arc::new(RenderPipelineCache::new(device.clone()));

    // Start async warming
    let vs_clone = Arc::clone(&vs);
    let fs_clone = Arc::clone(&fs);
    let layout_clone = Arc::clone(&layout);
    let device_clone = Arc::clone(&device);

    let keys: Vec<PipelineKey> = (0..5).map(|i| PipelineKey::new(i)).collect();

    let handle = cache.warm_cache_async(
        keys.clone(),
        move |_| {
            device_clone.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("async_pipeline"),
                layout: Some(&layout_clone),
                vertex: wgpu::VertexState {
                    module: &vs_clone,
                    entry_point: "vs_main",
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &fs_clone,
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
        },
        None,
    );

    // While warming, spawn threads that also access the cache
    let mut handles = Vec::new();
    for i in 0..3 {
        let cache = Arc::clone(&cache);
        let device = Arc::clone(&device);
        let vs = Arc::clone(&vs);
        let fs = Arc::clone(&fs);
        let layout = Arc::clone(&layout);

        handles.push(thread::spawn(move || {
            let key = PipelineKey::new(100 + i as u64);
            cache.get_or_create(&key, || {
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("concurrent_pipeline"),
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

    // Wait for all threads
    for h in handles {
        h.join().unwrap();
    }

    // Wait for warming to complete
    let result = handle.join();
    assert!(result.is_success());

    // Should have 5 warmed + 3 concurrent = 8 pipelines
    assert_eq!(cache.len(), 8);
}

/// Test: Warming then invalidation.
#[test]
fn test_warm_then_invalidate() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);
    let device = Arc::new(device);

    let vs = create_shader_module(&device, MINIMAL_VERTEX_SHADER);
    let fs = create_shader_module(&device, MINIMAL_FRAGMENT_SHADER);
    let layout = create_empty_layout(&device);

    let cache = RenderPipelineCache::new(device.clone());

    // Use shader ID 42 for all
    let keys: Vec<PipelineKey> = (0..3)
        .map(|i| PipelineKey::new(42).with_fragment_shader(i))
        .collect();

    cache.warm_cache(
        &keys,
        |_| {
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
        },
        WarmingConfig::default(),
        None,
    );

    assert_eq!(cache.len(), 3);

    // Invalidate vertex shader 42
    let invalidated = cache.invalidate(42);
    assert_eq!(invalidated, 3);
    assert_eq!(cache.len(), 0);
}

/// Test: Warming after clear.
#[test]
fn test_warm_after_clear() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);
    let device = Arc::new(device);

    let vs = create_shader_module(&device, MINIMAL_VERTEX_SHADER);
    let fs = create_shader_module(&device, MINIMAL_FRAGMENT_SHADER);
    let layout = create_empty_layout(&device);

    let cache = RenderPipelineCache::new(device.clone());

    let keys: Vec<PipelineKey> = (0..3).map(|i| PipelineKey::new(i)).collect();

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

    // First warming
    let result1 = cache.warm_cache(
        &keys,
        |_| create_pipeline(),
        WarmingConfig::default(),
        None,
    );
    assert_eq!(result1.warmed, 3);

    // Clear
    cache.clear();
    assert_eq!(cache.len(), 0);

    // Second warming (should re-warm all)
    let result2 = cache.warm_cache(
        &keys,
        |_| create_pipeline(),
        WarmingConfig::default(),
        None,
    );
    assert_eq!(result2.warmed, 3);
    assert_eq!(result2.skipped, 0);
    assert_eq!(cache.len(), 3);
}

/// Test: ProgressCallback type compatibility.
#[test]
fn test_progress_callback_type() {
    // Verify ProgressCallback can be constructed with various closures
    let _callback1: ProgressCallback = Box::new(|_progress| {});

    let counter = Arc::new(AtomicUsize::new(0));
    let counter_clone = Arc::clone(&counter);
    let _callback2: ProgressCallback = Box::new(move |progress| {
        counter_clone.fetch_add(progress.completed, Ordering::SeqCst);
    });

    // Note: ProgressCallback = Box<dyn Fn(WarmingProgress) + Send + Sync>
    // This is implicitly Send + Sync due to the trait bounds
}

/// Test: Large number of keys.
#[test]
fn test_warm_cache_many_keys() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);
    let device = Arc::new(device);

    let vs = create_shader_module(&device, MINIMAL_VERTEX_SHADER);
    let fs = create_shader_module(&device, MINIMAL_FRAGMENT_SHADER);
    let layout = create_empty_layout(&device);

    let cache = RenderPipelineCache::new(device.clone());

    // Create 50 unique keys
    let keys: Vec<PipelineKey> = (0..50).map(|i| PipelineKey::new(i)).collect();

    let result = cache.warm_cache(
        &keys,
        |_| {
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
        },
        WarmingConfig::default(),
        None,
    );

    assert_eq!(result.warmed, 50);
    assert!(result.is_success());
    assert_eq!(cache.len(), 50);
}
