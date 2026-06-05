//! Blackbox tests for T-WGPU-P7.4.2 Memory Tracker.
//!
//! CLEANROOM: These tests use only the public API from
//! `renderer_backend::profiling::memory` without accessing internal implementation
//! details. The memory tracker is treated as a black box.
//!
//! Coverage:
//!   - API Contract Tests (20+): Public interface behavior
//!   - Real-World Scenarios (30+): Typical usage patterns
//!   - Budget Monitoring (15+): Budget tracking and warnings
//!   - Snapshot Comparison (15+): Memory state diffing
//!   - Leak Detection Patterns (15+): Finding resource leaks
//!   - Edge Cases (15+): Boundary conditions and error handling
//!   - Reporting (10+): Summary and formatting

use renderer_backend::profiling::memory::{
    // Enums
    MemoryType,
    ResourceType,
    // Core types
    AllocationInfo,
    AllocationStats,
    MemoryBudget,
    MemoryDiff,
    MemorySnapshot,
    MemoryTracker,
    // Utilities
    format_bytes,
};
// Use the enhanced LeakDetector from profiling::leaks (re-exported in profiling)
use renderer_backend::profiling::{LeakDetector, LeakThresholds};

use std::collections::HashMap;
use std::thread;
use std::time::Duration;

// =============================================================================
// SECTION 1 -- API Contract Tests (20+)
// =============================================================================

/// MemoryTracker can be created with default budget.
#[test]
fn api_tracker_with_default_budget_creates_empty_tracker() {
    let tracker = MemoryTracker::with_default_budget();
    assert_eq!(
        tracker.stats().current_allocations, 0,
        "new tracker should have no allocations"
    );
    assert_eq!(
        tracker.stats().current_bytes, 0,
        "new tracker should have 0 bytes used"
    );
    assert!(
        tracker.allocations().is_empty(),
        "allocations map should be empty"
    );
}

/// Track resource returns unique IDs.
#[test]
fn api_track_resource_returns_unique_ids() {
    let mut tracker = MemoryTracker::with_default_budget();

    let id1 = tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);
    let id2 = tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 2048, None);
    let id3 = tracker.track_resource(ResourceType::Texture, MemoryType::DeviceLocal, 4096, None);

    assert_ne!(id1, id2, "IDs should be unique");
    assert_ne!(id2, id3, "IDs should be unique");
    assert_ne!(id1, id3, "IDs should be unique");
}

/// Track resource increments allocation count.
#[test]
fn api_track_resource_increments_stats() {
    let mut tracker = MemoryTracker::with_default_budget();

    tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);
    assert_eq!(tracker.stats().current_allocations, 1);
    assert_eq!(tracker.stats().total_allocations, 1);
    assert_eq!(tracker.stats().current_bytes, 1024);

    tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 2048, None);
    assert_eq!(tracker.stats().current_allocations, 2);
    assert_eq!(tracker.stats().total_allocations, 2);
    assert_eq!(tracker.stats().current_bytes, 3072);
}

/// Untrack removes allocation and updates stats.
#[test]
fn api_untrack_removes_allocation() {
    let mut tracker = MemoryTracker::with_default_budget();

    let id = tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);
    assert_eq!(tracker.stats().current_allocations, 1);

    let removed = tracker.untrack(id);
    assert!(removed, "untrack should return true for existing allocation");
    assert_eq!(tracker.stats().current_allocations, 0);
    assert_eq!(tracker.stats().current_bytes, 0);
    assert_eq!(tracker.stats().total_deallocations, 1);
}

/// Untrack returns false for non-existent ID.
#[test]
fn api_untrack_nonexistent_returns_false() {
    let mut tracker = MemoryTracker::with_default_budget();

    let removed = tracker.untrack(999999);
    assert!(!removed, "untrack should return false for non-existent ID");
}

/// Get allocation returns info for tracked resource.
#[test]
fn api_get_allocation_returns_info() {
    let mut tracker = MemoryTracker::with_default_budget();

    let id = tracker.track_resource(
        ResourceType::Texture,
        MemoryType::DeviceLocal,
        4096,
        Some("Test Texture"),
    );

    let info = tracker.get_allocation(id);
    assert!(info.is_some(), "should find tracked allocation");

    let info = info.unwrap();
    assert_eq!(info.id, id);
    assert_eq!(info.resource_type, ResourceType::Texture);
    assert_eq!(info.memory_type, MemoryType::DeviceLocal);
    assert_eq!(info.size_bytes, 4096);
    assert_eq!(info.label, Some("Test Texture".to_string()));
}

/// Get allocation returns None for unknown ID.
#[test]
fn api_get_allocation_unknown_returns_none() {
    let tracker = MemoryTracker::with_default_budget();
    assert!(
        tracker.get_allocation(12345).is_none(),
        "unknown ID should return None"
    );
}

/// Track buffer with usage infers memory type.
#[test]
fn api_track_buffer_with_usage_infers_memory_type() {
    let mut tracker = MemoryTracker::with_default_budget();

    // MAP_READ implies HostCoherent
    let id = tracker.track_buffer_with_usage(
        1024,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        None,
    );
    let info = tracker.get_allocation(id).unwrap();
    assert_eq!(info.memory_type, MemoryType::HostCoherent);

    // MAP_WRITE implies HostCached
    let id2 = tracker.track_buffer_with_usage(
        2048,
        wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
        None,
    );
    let info2 = tracker.get_allocation(id2).unwrap();
    assert_eq!(info2.memory_type, MemoryType::HostCached);

    // No mapping implies DeviceLocal
    let id3 = tracker.track_buffer_with_usage(4096, wgpu::BufferUsages::VERTEX, None);
    let info3 = tracker.get_allocation(id3).unwrap();
    assert_eq!(info3.memory_type, MemoryType::DeviceLocal);
}

/// Track texture with usage infers memory type.
#[test]
fn api_track_texture_with_usage_infers_memory_type() {
    let mut tracker = MemoryTracker::with_default_budget();

    // COPY_SRC implies HostVisible
    let id = tracker.track_texture_with_usage(
        4096,
        wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::TEXTURE_BINDING,
        None,
    );
    let info = tracker.get_allocation(id).unwrap();
    assert_eq!(info.memory_type, MemoryType::HostVisible);

    // No COPY_SRC implies DeviceLocal
    let id2 =
        tracker.track_texture_with_usage(8192, wgpu::TextureUsages::RENDER_ATTACHMENT, None);
    let info2 = tracker.get_allocation(id2).unwrap();
    assert_eq!(info2.memory_type, MemoryType::DeviceLocal);
}

/// Track query set creates allocation.
#[test]
fn api_track_query_set() {
    let mut tracker = MemoryTracker::with_default_budget();

    let id = tracker.track_query_set(512, Some("Timestamp Queries"));
    let info = tracker.get_allocation(id).unwrap();

    assert_eq!(info.resource_type, ResourceType::QuerySet);
    assert_eq!(info.size_bytes, 512);
    assert_eq!(info.label, Some("Timestamp Queries".to_string()));
}

/// Track bind group creates allocation with nominal size.
#[test]
fn api_track_bind_group() {
    let mut tracker = MemoryTracker::with_default_budget();

    let id = tracker.track_bind_group(Some("Material Bindings"));
    let info = tracker.get_allocation(id).unwrap();

    assert_eq!(info.resource_type, ResourceType::BindGroup);
    assert_eq!(info.size_bytes, 256); // Nominal size for bind groups
}

/// Track pipeline creates allocation with nominal size.
#[test]
fn api_track_pipeline() {
    let mut tracker = MemoryTracker::with_default_budget();

    let id = tracker.track_pipeline(Some("PBR Pipeline"));
    let info = tracker.get_allocation(id).unwrap();

    assert_eq!(info.resource_type, ResourceType::Pipeline);
    assert_eq!(info.size_bytes, 4096); // Nominal size for pipelines
}

/// Clear removes all allocations and resets stats.
#[test]
fn api_clear_resets_tracker() {
    let mut tracker = MemoryTracker::with_default_budget();

    tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);
    tracker.track_resource(ResourceType::Texture, MemoryType::DeviceLocal, 4096, None);
    tracker.track_resource(ResourceType::Pipeline, MemoryType::DeviceLocal, 256, None);

    assert_eq!(tracker.stats().current_allocations, 3);

    tracker.clear();

    assert_eq!(tracker.stats().current_allocations, 0);
    assert_eq!(tracker.stats().current_bytes, 0);
    assert!(tracker.allocations().is_empty());
}

/// Stats tracks bytes by resource type.
#[test]
fn api_stats_tracks_bytes_by_type() {
    let mut tracker = MemoryTracker::with_default_budget();

    tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);
    tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 2048, None);
    tracker.track_resource(ResourceType::Texture, MemoryType::DeviceLocal, 4096, None);

    let stats = tracker.stats();
    assert_eq!(stats.bytes_by_type.get(&ResourceType::Buffer), Some(&3072));
    assert_eq!(
        stats.bytes_by_type.get(&ResourceType::Texture),
        Some(&4096)
    );
}

/// Peak tracking captures highest values.
#[test]
fn api_stats_peak_tracking() {
    let mut tracker = MemoryTracker::with_default_budget();

    let id1 = tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);
    let id2 = tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 2048, None);

    assert_eq!(tracker.stats().peak_allocations, 2);
    assert_eq!(tracker.stats().peak_bytes, 3072);

    tracker.untrack(id1);
    tracker.untrack(id2);

    // Peak should remain unchanged after deallocations
    assert_eq!(tracker.stats().peak_allocations, 2);
    assert_eq!(tracker.stats().peak_bytes, 3072);
    assert_eq!(tracker.stats().current_allocations, 0);
}

/// Stats can reset peak values.
#[test]
fn api_stats_reset_peak() {
    let mut tracker = MemoryTracker::with_default_budget();

    let id = tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 4096, None);
    tracker.untrack(id);

    assert_eq!(tracker.stats().peak_bytes, 4096);

    tracker.stats_mut().reset_peak();

    assert_eq!(tracker.stats().peak_bytes, 0);
    assert_eq!(tracker.stats().peak_allocations, 0);
}

/// Budget is accessible and has reasonable defaults.
#[test]
fn api_budget_accessible() {
    let tracker = MemoryTracker::with_default_budget();
    let budget = tracker.budget();

    assert!(budget.total_budget > 0, "total budget should be positive");
    assert!(
        budget.device_local_budget > 0,
        "device local budget should be positive"
    );
    assert_eq!(budget.device_local_used, 0, "initial usage should be zero");
    assert_eq!(
        budget.host_visible_used, 0,
        "initial host visible usage should be zero"
    );
}

/// Summary generates human-readable output.
#[test]
fn api_summary_generates_output() {
    let mut tracker = MemoryTracker::with_default_budget();

    tracker.track_resource(
        ResourceType::Buffer,
        MemoryType::DeviceLocal,
        1024 * 1024,
        None,
    );
    tracker.track_resource(
        ResourceType::Texture,
        MemoryType::DeviceLocal,
        4 * 1024 * 1024,
        None,
    );

    let summary = tracker.summary();

    assert!(
        summary.contains("GPU Memory Summary"),
        "summary should have title"
    );
    assert!(
        summary.contains("allocations"),
        "summary should mention allocations"
    );
    assert!(
        summary.contains("Budget"),
        "summary should mention budget"
    );
}

/// Debug output is available.
#[test]
fn api_tracker_debug_output() {
    let mut tracker = MemoryTracker::with_default_budget();
    tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);

    let debug = format!("{:?}", tracker);
    assert!(
        debug.contains("MemoryTracker"),
        "debug should contain type name"
    );
    assert!(
        debug.contains("allocations_count"),
        "debug should contain allocations count"
    );
}

// =============================================================================
// SECTION 2 -- Real-World Scenarios (30+)
// =============================================================================

/// Simulate render frame memory lifecycle.
#[test]
fn scenario_render_frame_lifecycle() {
    let mut tracker = MemoryTracker::with_default_budget();

    // Frame start: allocate per-frame resources
    let uniform_id = tracker.track_buffer_with_usage(
        256,
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        Some("Frame Uniforms"),
    );
    let instance_id = tracker.track_buffer_with_usage(
        64 * 1024,
        wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        Some("Instance Data"),
    );

    assert_eq!(tracker.stats().current_allocations, 2);

    // Mid-frame: add dynamic resources
    let staging_id = tracker.track_buffer_with_usage(
        32 * 1024,
        wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
        Some("Staging Buffer"),
    );

    assert_eq!(tracker.stats().current_allocations, 3);

    // Frame end: release per-frame resources
    tracker.untrack(staging_id);
    tracker.untrack(uniform_id);
    tracker.untrack(instance_id);

    assert_eq!(tracker.stats().current_allocations, 0);
    assert_eq!(tracker.stats().total_allocations, 3);
    assert_eq!(tracker.stats().total_deallocations, 3);
}

/// Simulate texture upload tracking.
#[test]
fn scenario_texture_upload_tracking() {
    let mut tracker = MemoryTracker::with_default_budget();

    // Create staging buffer for upload
    let staging_id = tracker.track_buffer_with_usage(
        1024 * 1024 * 4, // 1024x1024 RGBA8
        wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
        Some("Texture Staging"),
    );

    // Create GPU texture
    let texture_id = tracker.track_texture_with_usage(
        1024 * 1024 * 4,
        wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        Some("Albedo Texture"),
    );

    let peak_bytes = tracker.stats().peak_bytes;
    assert!(peak_bytes >= 8 * 1024 * 1024); // At least 8 MB peak

    // After upload, release staging
    tracker.untrack(staging_id);

    // Texture remains
    assert_eq!(tracker.stats().current_allocations, 1);
    assert_eq!(tracker.stats().current_bytes, 1024 * 1024 * 4);
}

/// Simulate buffer pool allocation pattern.
#[test]
fn scenario_buffer_pool_pattern() {
    let mut tracker = MemoryTracker::with_default_budget();
    let mut pool_ids = Vec::new();

    // Pre-allocate pool of uniform buffers
    for i in 0..8 {
        let id = tracker.track_buffer_with_usage(
            4096,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            Some(&format!("Pool Buffer {}", i)),
        );
        pool_ids.push(id);
    }

    assert_eq!(tracker.stats().current_allocations, 8);
    assert_eq!(tracker.stats().current_bytes, 8 * 4096);

    // "Use" and "return" buffers (simulated by checking they exist)
    for id in &pool_ids {
        let info = tracker.get_allocation(*id);
        assert!(info.is_some());
    }

    // Cleanup pool
    for id in pool_ids {
        tracker.untrack(id);
    }

    assert_eq!(tracker.stats().current_allocations, 0);
}

/// Simulate staging buffer lifecycle.
#[test]
fn scenario_staging_buffer_lifecycle() {
    let mut tracker = MemoryTracker::with_default_budget();

    // Multiple staging buffers for parallel uploads
    let staging1 = tracker.track_buffer_with_usage(
        64 * 1024,
        wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
        Some("Staging 1"),
    );
    let staging2 = tracker.track_buffer_with_usage(
        128 * 1024,
        wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
        Some("Staging 2"),
    );

    // All are host-cached (writable)
    assert_eq!(
        tracker.get_allocation(staging1).unwrap().memory_type,
        MemoryType::HostCached
    );
    assert_eq!(
        tracker.get_allocation(staging2).unwrap().memory_type,
        MemoryType::HostCached
    );

    // Release after copy completes
    tracker.untrack(staging1);
    tracker.untrack(staging2);

    assert_eq!(tracker.stats().current_allocations, 0);
}

/// Simulate render target creation and resize.
#[test]
fn scenario_render_target_resize() {
    let mut tracker = MemoryTracker::with_default_budget();

    // Initial render targets at 1920x1080
    let width = 1920;
    let height = 1080;
    let color_size = width * height * 4; // RGBA8
    let depth_size = width * height * 4; // Depth32Float

    let color_id = tracker.track_texture_with_usage(
        color_size,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        Some("Color Target"),
    );
    let depth_id = tracker.track_texture_with_usage(
        depth_size,
        wgpu::TextureUsages::RENDER_ATTACHMENT,
        Some("Depth Target"),
    );

    let initial_bytes = tracker.stats().current_bytes;

    // Resize to 2560x1440
    tracker.untrack(color_id);
    tracker.untrack(depth_id);

    let new_width = 2560;
    let new_height = 1440;
    let new_color_size = new_width * new_height * 4;
    let new_depth_size = new_width * new_height * 4;

    tracker.track_texture_with_usage(
        new_color_size,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        Some("Color Target"),
    );
    tracker.track_texture_with_usage(
        new_depth_size,
        wgpu::TextureUsages::RENDER_ATTACHMENT,
        Some("Depth Target"),
    );

    let resized_bytes = tracker.stats().current_bytes;
    assert!(resized_bytes > initial_bytes);
}

/// Simulate shader compilation memory.
#[test]
fn scenario_shader_compilation() {
    let mut tracker = MemoryTracker::with_default_budget();
    let mut pipeline_ids = Vec::new();

    // Compile multiple shader pipelines
    for i in 0..5 {
        let id = tracker.track_pipeline(Some(&format!("Pipeline {}", i)));
        pipeline_ids.push(id);
    }

    assert_eq!(tracker.stats().current_allocations, 5);
    assert_eq!(
        *tracker.stats().bytes_by_type.get(&ResourceType::Pipeline).unwrap_or(&0),
        5 * 4096
    );

    // Cleanup
    for id in pipeline_ids {
        tracker.untrack(id);
    }
}

/// Simulate bind group accumulation.
#[test]
fn scenario_bind_group_accumulation() {
    let mut tracker = MemoryTracker::with_default_budget();
    let mut bind_group_ids = Vec::new();

    // Create many bind groups (common in material systems)
    for i in 0..100 {
        let id = tracker.track_bind_group(Some(&format!("Material {}", i)));
        bind_group_ids.push(id);
    }

    assert_eq!(tracker.stats().current_allocations, 100);
    assert_eq!(
        *tracker.stats().bytes_by_type.get(&ResourceType::BindGroup).unwrap_or(&0),
        100 * 256
    );

    // Cleanup half
    for id in bind_group_ids.iter().take(50) {
        tracker.untrack(*id);
    }

    assert_eq!(tracker.stats().current_allocations, 50);
}

/// Simulate pipeline cache growth.
#[test]
fn scenario_pipeline_cache_growth() {
    let mut tracker = MemoryTracker::with_default_budget();

    // Simulate gradual pipeline cache buildup
    let mut ids = Vec::new();
    for frame in 0..10 {
        // Each frame adds 2 new pipeline variants
        for variant in 0..2 {
            let id = tracker.track_pipeline(Some(&format!("Frame {} Variant {}", frame, variant)));
            ids.push(id);
        }
    }

    assert_eq!(tracker.stats().current_allocations, 20);
    assert_eq!(tracker.stats().total_allocations, 20);

    // Verify memory grows linearly
    let expected_bytes = 20 * 4096;
    assert_eq!(
        *tracker.stats().bytes_by_type.get(&ResourceType::Pipeline).unwrap_or(&0),
        expected_bytes
    );
}

/// Simulate multi-frame memory pattern.
#[test]
fn scenario_multi_frame_pattern() {
    let mut tracker = MemoryTracker::with_default_budget();

    for frame in 0..5 {
        // Per-frame uniform buffer
        let uniform_id = tracker.track_buffer_with_usage(
            1024,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            Some(&format!("Frame {} Uniforms", frame)),
        );

        // Simulate work...

        // Release frame resources
        tracker.untrack(uniform_id);

        // Persistent resources should remain
        assert_eq!(tracker.stats().current_allocations, 0);
    }

    // Total should reflect all frames
    assert_eq!(tracker.stats().total_allocations, 5);
    assert_eq!(tracker.stats().total_deallocations, 5);
}

/// Simulate resource cleanup on scene change.
#[test]
fn scenario_scene_change_cleanup() {
    let mut tracker = MemoryTracker::with_default_budget();
    let mut scene_resources = Vec::new();

    // Load Scene A
    for i in 0..10 {
        let id = tracker.track_texture_with_usage(
            1024 * 1024,
            wgpu::TextureUsages::TEXTURE_BINDING,
            Some(&format!("Scene A Texture {}", i)),
        );
        scene_resources.push(id);
    }

    let scene_a_bytes = tracker.stats().current_bytes;

    // Cleanup Scene A
    for id in scene_resources.drain(..) {
        tracker.untrack(id);
    }

    assert_eq!(tracker.stats().current_bytes, 0);

    // Load Scene B (different size)
    for i in 0..5 {
        let id = tracker.track_texture_with_usage(
            2 * 1024 * 1024,
            wgpu::TextureUsages::TEXTURE_BINDING,
            Some(&format!("Scene B Texture {}", i)),
        );
        scene_resources.push(id);
    }

    let scene_b_bytes = tracker.stats().current_bytes;

    // Peak should reflect maximum of both scenes
    assert!(tracker.stats().peak_bytes >= scene_a_bytes.max(scene_b_bytes));
}

/// Simulate streaming asset loading.
#[test]
fn scenario_streaming_assets() {
    let mut tracker = MemoryTracker::with_default_budget();
    let mut loaded_textures = HashMap::new();

    // Stream in textures as needed
    for i in 0..20 {
        let id = tracker.track_texture_with_usage(
            512 * 512 * 4,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            Some(&format!("Streamed Texture {}", i)),
        );
        loaded_textures.insert(i, id);

        // Unload older textures to stay within budget (simulated LRU)
        if loaded_textures.len() > 10 {
            if let Some((&oldest_key, _)) = loaded_textures.iter().next() {
                let oldest_id = loaded_textures.remove(&oldest_key).unwrap();
                tracker.untrack(oldest_id);
            }
        }
    }

    // Should maintain roughly 10 textures
    assert!(tracker.stats().current_allocations <= 11);
}

/// Simulate GPU readback buffer pattern.
#[test]
fn scenario_gpu_readback() {
    let mut tracker = MemoryTracker::with_default_budget();

    // Create readback buffer
    let readback_id = tracker.track_buffer_with_usage(
        4 * 1024 * 1024,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        Some("Readback Buffer"),
    );

    let info = tracker.get_allocation(readback_id).unwrap();
    assert_eq!(info.memory_type, MemoryType::HostCoherent);

    // Simulate multiple readback cycles
    for _ in 0..3 {
        // Buffer stays mapped, no reallocation needed
        assert!(tracker.get_allocation(readback_id).is_some());
    }

    tracker.untrack(readback_id);
    assert_eq!(tracker.stats().current_allocations, 0);
}

/// Simulate mipmap chain allocation.
#[test]
fn scenario_mipmap_chain() {
    let mut tracker = MemoryTracker::with_default_budget();

    // Calculate mipmap sizes for 1024x1024 texture
    let base_size = 1024 * 1024 * 4;
    let mut total_mip_size = 0u64;
    let mut size = base_size;
    while size >= 4 {
        total_mip_size += size;
        size /= 4;
    }

    let texture_id = tracker.track_texture_with_usage(
        total_mip_size,
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        Some("Mipmapped Texture"),
    );

    let info = tracker.get_allocation(texture_id).unwrap();
    assert_eq!(info.size_bytes, total_mip_size);
}

/// Simulate indirect draw buffer usage.
#[test]
fn scenario_indirect_draw_buffers() {
    let mut tracker = MemoryTracker::with_default_budget();

    // Indirect argument buffer
    let indirect_id = tracker.track_buffer_with_usage(
        1024 * 20, // 1024 draw commands * 20 bytes each
        wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::STORAGE,
        Some("Indirect Args"),
    );

    // Count buffer
    let count_id = tracker.track_buffer_with_usage(
        4,
        wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::STORAGE,
        Some("Draw Count"),
    );

    assert_eq!(tracker.stats().current_allocations, 2);

    // Both should be device-local
    assert_eq!(
        tracker.get_allocation(indirect_id).unwrap().memory_type,
        MemoryType::DeviceLocal
    );
    assert_eq!(
        tracker.get_allocation(count_id).unwrap().memory_type,
        MemoryType::DeviceLocal
    );
}

/// Simulate compute shader storage buffers.
#[test]
fn scenario_compute_storage_buffers() {
    let mut tracker = MemoryTracker::with_default_budget();

    // Input buffer
    let input_id = tracker.track_buffer_with_usage(
        1024 * 1024,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        Some("Compute Input"),
    );

    // Output buffer
    let output_id = tracker.track_buffer_with_usage(
        1024 * 1024,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        Some("Compute Output"),
    );

    assert_eq!(tracker.stats().current_bytes, 2 * 1024 * 1024);

    tracker.untrack(input_id);
    tracker.untrack(output_id);
}

/// Simulate cubemap allocation.
#[test]
fn scenario_cubemap_allocation() {
    let mut tracker = MemoryTracker::with_default_budget();

    // 6 faces * resolution * format
    let face_size = 512 * 512 * 4;
    let cubemap_size = face_size * 6;

    let cubemap_id = tracker.track_texture_with_usage(
        cubemap_size,
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        Some("Environment Cubemap"),
    );

    let info = tracker.get_allocation(cubemap_id).unwrap();
    assert_eq!(info.size_bytes, cubemap_size);
    assert_eq!(info.resource_type, ResourceType::Texture);
}

/// Simulate shadow map cascade allocation.
#[test]
fn scenario_shadow_cascades() {
    let mut tracker = MemoryTracker::with_default_budget();
    let mut cascade_ids = Vec::new();

    // 4 cascades with different resolutions
    let resolutions = [2048, 1024, 512, 256];
    for (i, &res) in resolutions.iter().enumerate() {
        let size = (res * res * 4) as u64; // Depth32Float
        let id = tracker.track_texture_with_usage(
            size,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            Some(&format!("Shadow Cascade {}", i)),
        );
        cascade_ids.push(id);
    }

    assert_eq!(tracker.stats().current_allocations, 4);

    // Cleanup
    for id in cascade_ids {
        tracker.untrack(id);
    }
}

/// Simulate vertex/index buffer pair.
#[test]
fn scenario_mesh_buffers() {
    let mut tracker = MemoryTracker::with_default_budget();

    // Vertex buffer (1000 vertices * 32 bytes stride)
    let vertex_id = tracker.track_buffer_with_usage(
        1000 * 32,
        wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        Some("Mesh Vertices"),
    );

    // Index buffer (3000 indices * 2 bytes for u16)
    let index_id = tracker.track_buffer_with_usage(
        3000 * 2,
        wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        Some("Mesh Indices"),
    );

    assert_eq!(tracker.stats().current_allocations, 2);
    assert_eq!(tracker.stats().current_bytes, 32000 + 6000);

    // Both should be device-local
    assert_eq!(
        tracker.get_allocation(vertex_id).unwrap().memory_type,
        MemoryType::DeviceLocal
    );
    assert_eq!(
        tracker.get_allocation(index_id).unwrap().memory_type,
        MemoryType::DeviceLocal
    );
}

/// Simulate timestamp query allocation.
#[test]
fn scenario_timestamp_queries() {
    let mut tracker = MemoryTracker::with_default_budget();

    // Query set for GPU timestamps
    let query_id = tracker.track_query_set(64 * 8, Some("GPU Timestamps")); // 64 queries * 8 bytes

    let info = tracker.get_allocation(query_id).unwrap();
    assert_eq!(info.resource_type, ResourceType::QuerySet);

    // Readback buffer for results
    let readback_id = tracker.track_buffer_with_usage(
        64 * 8,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        Some("Query Results"),
    );

    assert_eq!(tracker.stats().current_allocations, 2);

    tracker.untrack(query_id);
    tracker.untrack(readback_id);
}

/// Simulate SSAO resource allocation.
#[test]
fn scenario_ssao_resources() {
    let mut tracker = MemoryTracker::with_default_budget();

    let width = 1920;
    let height = 1080;

    // Half-resolution AO texture
    let ao_id = tracker.track_texture_with_usage(
        (width / 2) * (height / 2),
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        Some("SSAO Result"),
    );

    // Noise texture (small)
    let noise_id = tracker.track_texture_with_usage(
        4 * 4 * 4 * 2, // 4x4 RG16F
        wgpu::TextureUsages::TEXTURE_BINDING,
        Some("SSAO Noise"),
    );

    // Kernel buffer
    let kernel_id = tracker.track_buffer_with_usage(
        64 * 16, // 64 samples * vec4
        wgpu::BufferUsages::UNIFORM,
        Some("SSAO Kernel"),
    );

    assert_eq!(tracker.stats().current_allocations, 3);

    tracker.untrack(ao_id);
    tracker.untrack(noise_id);
    tracker.untrack(kernel_id);
}

/// Simulate bloom chain allocation.
#[test]
fn scenario_bloom_chain() {
    let mut tracker = MemoryTracker::with_default_budget();
    let mut bloom_ids = Vec::new();

    // Progressively smaller bloom textures
    let mut width = 1920u64;
    let mut height = 1080u64;
    for i in 0..5 {
        width /= 2;
        height /= 2;
        let size = width * height * 8; // RGBA16F
        let id = tracker.track_texture_with_usage(
            size,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            Some(&format!("Bloom Mip {}", i)),
        );
        bloom_ids.push(id);
    }

    assert_eq!(tracker.stats().current_allocations, 5);

    for id in bloom_ids {
        tracker.untrack(id);
    }
}

/// Simulate GPU skinning buffers.
#[test]
fn scenario_skinning_buffers() {
    let mut tracker = MemoryTracker::with_default_budget();

    // Bone matrices buffer (128 bones * 64 bytes per mat4)
    let bones_id = tracker.track_buffer_with_usage(
        128 * 64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        Some("Bone Matrices"),
    );

    // Vertex weights (1000 vertices * 8 weights bytes)
    let weights_id = tracker.track_buffer_with_usage(
        1000 * 8,
        wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        Some("Skin Weights"),
    );

    // Output positions
    let output_id = tracker.track_buffer_with_usage(
        1000 * 12, // 1000 vertices * vec3
        wgpu::BufferUsages::STORAGE,
        Some("Skinned Positions"),
    );

    assert_eq!(tracker.stats().current_allocations, 3);

    tracker.untrack(bones_id);
    tracker.untrack(weights_id);
    tracker.untrack(output_id);
}

/// Simulate particle system buffers.
#[test]
fn scenario_particle_buffers() {
    let mut tracker = MemoryTracker::with_default_budget();

    let max_particles = 10000;

    // Position/velocity buffer
    let state_id = tracker.track_buffer_with_usage(
        max_particles * 32, // 2 * vec4 per particle
        wgpu::BufferUsages::STORAGE,
        Some("Particle State"),
    );

    // Indirect args
    let indirect_id = tracker.track_buffer_with_usage(
        16,
        wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::STORAGE,
        Some("Particle Indirect"),
    );

    assert_eq!(tracker.stats().current_allocations, 2);

    tracker.untrack(state_id);
    tracker.untrack(indirect_id);
}

/// Simulate instance culling buffers.
#[test]
fn scenario_instance_culling() {
    let mut tracker = MemoryTracker::with_default_budget();

    let max_instances = 50000;

    // Input transforms
    let input_id = tracker.track_buffer_with_usage(
        max_instances * 64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        Some("Instance Transforms"),
    );

    // Visibility flags
    let visibility_id = tracker.track_buffer_with_usage(
        max_instances / 8, // 1 bit per instance
        wgpu::BufferUsages::STORAGE,
        Some("Visibility Flags"),
    );

    // Compacted output
    let output_id = tracker.track_buffer_with_usage(
        max_instances * 64,
        wgpu::BufferUsages::STORAGE,
        Some("Culled Instances"),
    );

    assert_eq!(tracker.stats().current_allocations, 3);

    tracker.untrack(input_id);
    tracker.untrack(visibility_id);
    tracker.untrack(output_id);
}

/// Simulate deferred G-Buffer allocation.
#[test]
fn scenario_gbuffer_allocation() {
    let mut tracker = MemoryTracker::with_default_budget();

    let width = 1920u64;
    let height = 1080u64;

    // Albedo (RGBA8)
    let albedo_id = tracker.track_texture_with_usage(
        width * height * 4,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        Some("GBuffer Albedo"),
    );

    // Normal (RG16F)
    let normal_id = tracker.track_texture_with_usage(
        width * height * 4,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        Some("GBuffer Normal"),
    );

    // Material (RGBA8)
    let material_id = tracker.track_texture_with_usage(
        width * height * 4,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        Some("GBuffer Material"),
    );

    // Depth (D32F)
    let depth_id = tracker.track_texture_with_usage(
        width * height * 4,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        Some("GBuffer Depth"),
    );

    assert_eq!(tracker.stats().current_allocations, 4);
    assert_eq!(tracker.stats().current_bytes, width * height * 4 * 4);

    tracker.untrack(albedo_id);
    tracker.untrack(normal_id);
    tracker.untrack(material_id);
    tracker.untrack(depth_id);
}

// =============================================================================
// SECTION 3 -- Budget Monitoring (15+)
// =============================================================================

/// Budget starts at zero usage.
#[test]
fn budget_initial_zero_usage() {
    let tracker = MemoryTracker::with_default_budget();
    let budget = tracker.budget();

    assert_eq!(budget.device_local_used, 0);
    assert_eq!(budget.host_visible_used, 0);
}

/// Budget tracks device-local allocations.
#[test]
fn budget_tracks_device_local() {
    let mut tracker = MemoryTracker::with_default_budget();

    tracker.track_buffer_with_usage(
        1024 * 1024,
        wgpu::BufferUsages::VERTEX,
        None,
    );

    assert_eq!(tracker.budget().device_local_used, 1024 * 1024);
}

/// Budget tracks host-visible allocations.
#[test]
fn budget_tracks_host_visible() {
    let mut tracker = MemoryTracker::with_default_budget();

    tracker.track_buffer_with_usage(
        512 * 1024,
        wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
        None,
    );

    assert_eq!(tracker.budget().host_visible_used, 512 * 1024);
}

/// Under budget operation returns low utilization.
#[test]
fn budget_under_budget_low_utilization() {
    let mut tracker = MemoryTracker::with_default_budget();

    // Small allocation
    tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);

    let util = tracker.budget().utilization();
    assert!(util < 0.01, "utilization should be very low");
    assert!(!tracker.budget().is_over_budget());
}

/// Approaching budget warning threshold.
#[test]
fn budget_approaching_threshold() {
    let mut budget = MemoryBudget::default();
    let threshold = 0.8f32;

    // Set usage to 81% of device local budget
    budget.device_local_used = (budget.device_local_budget as f64 * 0.81) as u64;

    let util = budget.device_local_utilization();
    assert!(
        util > threshold,
        "utilization {} should exceed warning threshold {}",
        util,
        threshold
    );
}

/// Over budget detection works.
#[test]
fn budget_over_budget_detected() {
    let mut budget = MemoryBudget::default();

    assert!(!budget.is_over_budget());

    // Exceed device local budget
    budget.device_local_used = budget.device_local_budget + 1;
    assert!(budget.is_over_budget());

    // Reset and exceed host visible
    budget.device_local_used = 0;
    budget.host_visible_used = budget.host_visible_budget + 1;
    assert!(budget.is_over_budget());
}

/// Device-local vs host-visible breakdown.
#[test]
fn budget_memory_type_breakdown() {
    let mut tracker = MemoryTracker::with_default_budget();

    // Device local
    tracker.track_buffer_with_usage(1000, wgpu::BufferUsages::VERTEX, None);

    // Host visible
    tracker.track_buffer_with_usage(
        500,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        None,
    );

    let budget = tracker.budget();
    assert_eq!(budget.device_local_used, 1000);
    assert_eq!(budget.host_visible_used, 500);
}

/// Utilization percentage accuracy.
#[test]
fn budget_utilization_accuracy() {
    let mut budget = MemoryBudget::default();

    // Use exactly 50% of total
    let half = budget.total_budget / 2;
    budget.device_local_used = half;

    let util = budget.utilization();
    assert!(
        (util - 0.5).abs() < 0.01,
        "utilization {} should be close to 0.5",
        util
    );
}

/// Remaining capacity tracking.
#[test]
fn budget_remaining_capacity() {
    let mut budget = MemoryBudget::default();
    let initial_remaining = budget.remaining();

    budget.device_local_used = 1_000_000;
    budget.host_visible_used = 500_000;

    let remaining = budget.remaining();
    assert_eq!(remaining, initial_remaining - 1_500_000);
}

/// Record allocation updates budget.
#[test]
fn budget_record_allocation() {
    let mut budget = MemoryBudget::default();

    budget.record_allocation(MemoryType::DeviceLocal, 1000);
    assert_eq!(budget.device_local_used, 1000);

    budget.record_allocation(MemoryType::HostVisible, 500);
    assert_eq!(budget.host_visible_used, 500);

    budget.record_allocation(MemoryType::HostCoherent, 300);
    assert_eq!(budget.host_visible_used, 800);

    budget.record_allocation(MemoryType::HostCached, 200);
    assert_eq!(budget.host_visible_used, 1000);
}

/// Record deallocation updates budget.
#[test]
fn budget_record_deallocation() {
    let mut budget = MemoryBudget::default();
    budget.device_local_used = 2000;
    budget.host_visible_used = 1000;

    budget.record_deallocation(MemoryType::DeviceLocal, 500);
    assert_eq!(budget.device_local_used, 1500);

    budget.record_deallocation(MemoryType::HostVisible, 300);
    assert_eq!(budget.host_visible_used, 700);
}

/// Deallocation saturates at zero.
#[test]
fn budget_deallocation_saturates() {
    let mut budget = MemoryBudget::default();
    budget.device_local_used = 100;

    budget.record_deallocation(MemoryType::DeviceLocal, 500); // More than available
    assert_eq!(budget.device_local_used, 0);
}

/// Zero budget returns zero utilization.
#[test]
fn budget_zero_budget_handling() {
    let budget = MemoryBudget {
        device_local_budget: 0,
        host_visible_budget: 0,
        total_budget: 0,
        device_local_used: 0,
        host_visible_used: 0,
    };

    assert_eq!(budget.utilization(), 0.0);
    assert_eq!(budget.device_local_utilization(), 0.0);
    assert_eq!(budget.host_visible_utilization(), 0.0);
}

/// Budget remaining saturates at zero.
#[test]
fn budget_remaining_saturates() {
    let mut budget = MemoryBudget::default();
    budget.device_local_used = budget.total_budget + 1_000_000;

    assert_eq!(budget.remaining(), 0);
}

// =============================================================================
// SECTION 4 -- Snapshot Comparison (15+)
// =============================================================================

/// Snapshot captures current state.
#[test]
fn snapshot_captures_state() {
    let mut tracker = MemoryTracker::with_default_budget();
    tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);
    tracker.track_resource(ResourceType::Texture, MemoryType::DeviceLocal, 2048, None);

    let snapshot = tracker.snapshot();

    assert_eq!(snapshot.allocations.len(), 2);
    assert_eq!(snapshot.stats.current_bytes, 3072);
}

/// Empty diff for unchanged state.
#[test]
fn snapshot_diff_empty_no_changes() {
    let mut tracker = MemoryTracker::with_default_budget();
    tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);

    let snapshot1 = tracker.snapshot();
    let snapshot2 = tracker.snapshot();

    let diff = snapshot1.diff(&snapshot2);
    assert!(diff.is_empty());
    assert_eq!(diff.bytes_delta, 0);
}

/// Frame-to-frame diff detects additions.
#[test]
fn snapshot_diff_detects_additions() {
    let mut tracker = MemoryTracker::with_default_budget();

    let snapshot1 = tracker.snapshot();

    tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);

    let snapshot2 = tracker.snapshot();
    let diff = snapshot1.diff(&snapshot2);

    assert_eq!(diff.added.len(), 1);
    assert_eq!(diff.removed.len(), 0);
    assert_eq!(diff.bytes_delta, 1024);
}

/// Frame-to-frame diff detects removals.
#[test]
fn snapshot_diff_detects_removals() {
    let mut tracker = MemoryTracker::with_default_budget();
    let id = tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);

    let snapshot1 = tracker.snapshot();

    tracker.untrack(id);

    let snapshot2 = tracker.snapshot();
    let diff = snapshot1.diff(&snapshot2);

    assert_eq!(diff.added.len(), 0);
    assert_eq!(diff.removed.len(), 1);
    assert_eq!(diff.bytes_delta, -1024);
}

/// Scene load memory delta calculation.
#[test]
fn snapshot_scene_load_delta() {
    let mut tracker = MemoryTracker::with_default_budget();

    let before = tracker.snapshot();

    // Load scene resources
    for i in 0..5 {
        tracker.track_texture_with_usage(
            1024 * 1024,
            wgpu::TextureUsages::TEXTURE_BINDING,
            Some(&format!("Scene Texture {}", i)),
        );
    }

    let after = tracker.snapshot();
    let diff = before.diff(&after);

    assert_eq!(diff.added.len(), 5);
    assert_eq!(diff.bytes_delta, 5 * 1024 * 1024);
}

/// Resource cleanup verification via diff.
#[test]
fn snapshot_cleanup_verification() {
    let mut tracker = MemoryTracker::with_default_budget();

    // Create resources
    let ids: Vec<_> = (0..10)
        .map(|i| tracker.track_resource(
            ResourceType::Buffer,
            MemoryType::DeviceLocal,
            1024,
            Some(&format!("Temp {}", i)),
        ))
        .collect();

    let before_cleanup = tracker.snapshot();

    // Cleanup
    for id in ids {
        tracker.untrack(id);
    }

    let after_cleanup = tracker.snapshot();
    let diff = before_cleanup.diff(&after_cleanup);

    assert_eq!(diff.removed.len(), 10);
    assert_eq!(diff.bytes_delta, -10 * 1024);
}

/// Memory growth detection over frames.
#[test]
fn snapshot_growth_detection() {
    let mut tracker = MemoryTracker::with_default_budget();
    let mut prev_snapshot = tracker.snapshot();
    let mut total_growth = 0i64;

    // Simulate frames with gradual growth
    for frame in 0..5 {
        tracker.track_resource(
            ResourceType::Buffer,
            MemoryType::DeviceLocal,
            1024,
            Some(&format!("Frame {} Buffer", frame)),
        );

        let current_snapshot = tracker.snapshot();
        let diff = prev_snapshot.diff(&current_snapshot);

        total_growth += diff.bytes_delta;
        prev_snapshot = current_snapshot;
    }

    assert_eq!(total_growth, 5 * 1024);
}

/// Memory shrinkage detection.
#[test]
fn snapshot_shrinkage_detection() {
    let mut tracker = MemoryTracker::with_default_budget();

    let ids: Vec<_> = (0..5)
        .map(|_| tracker.track_resource(
            ResourceType::Buffer,
            MemoryType::DeviceLocal,
            2048,
            None,
        ))
        .collect();

    let peak = tracker.snapshot();

    // Release half
    for id in ids.iter().take(3) {
        tracker.untrack(*id);
    }

    let reduced = tracker.snapshot();
    let diff = peak.diff(&reduced);

    assert!(diff.bytes_delta < 0);
    assert_eq!(diff.bytes_delta, -(3 * 2048));
}

/// Stable memory pattern verification.
#[test]
fn snapshot_stable_pattern() {
    let mut tracker = MemoryTracker::with_default_budget();

    // Create stable resources
    tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 4096, None);
    tracker.track_resource(ResourceType::Texture, MemoryType::DeviceLocal, 8192, None);

    let snap1 = tracker.snapshot();

    // Simulate frames with no changes
    for _ in 0..5 {
        let snap = tracker.snapshot();
        let diff = snap1.diff(&snap);
        assert!(diff.is_empty(), "no changes should result in empty diff");
    }
}

/// Diff bytes_added calculation.
#[test]
fn snapshot_diff_bytes_added() {
    let mut tracker = MemoryTracker::with_default_budget();

    let snap1 = tracker.snapshot();

    tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1000, None);
    tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 2000, None);

    let snap2 = tracker.snapshot();
    let diff = snap1.diff(&snap2);

    assert_eq!(diff.bytes_added(), 3000);
}

/// Diff counts accessors.
#[test]
fn snapshot_diff_counts() {
    let diff = MemoryDiff {
        added: vec![
            AllocationInfo::new(1, ResourceType::Buffer, MemoryType::DeviceLocal, 100),
            AllocationInfo::new(2, ResourceType::Buffer, MemoryType::DeviceLocal, 200),
        ],
        removed: vec![10, 20, 30],
        bytes_delta: -100,
    };

    assert_eq!(diff.added_count(), 2);
    assert_eq!(diff.removed_count(), 3);
}

/// Diff Display formatting.
#[test]
fn snapshot_diff_display() {
    let diff = MemoryDiff {
        added: vec![AllocationInfo::new(1, ResourceType::Buffer, MemoryType::DeviceLocal, 1024)],
        removed: vec![2],
        bytes_delta: 512,
    };

    let display = format!("{}", diff);
    assert!(display.contains("+1 allocs"));
    assert!(display.contains("-1 allocs"));
}

/// Snapshot age tracking.
#[test]
fn snapshot_age_tracking() {
    let tracker = MemoryTracker::with_default_budget();
    let snapshot = tracker.snapshot();

    // Snapshot was just taken
    assert!(snapshot.age_secs() < 1.0);
}

/// Multiple consecutive snapshots differ.
#[test]
fn snapshot_consecutive_differ() {
    let mut tracker = MemoryTracker::with_default_budget();

    let snap1 = tracker.snapshot();
    tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 100, None);

    let snap2 = tracker.snapshot();
    tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 200, None);

    let snap3 = tracker.snapshot();

    let diff12 = snap1.diff(&snap2);
    let diff23 = snap2.diff(&snap3);

    assert_eq!(diff12.bytes_delta, 100);
    assert_eq!(diff23.bytes_delta, 200);
}

// =============================================================================
// SECTION 5 -- Leak Detection Patterns (15+)
// =============================================================================
//
// Note: The enhanced LeakDetector from profiling::leaks is used here.
// It has a different API than the simple LeakDetector in profiling::memory.
// The enhanced version tracks allocations internally and provides richer features.

/// Leak detector starts empty with default thresholds.
#[test]
fn leak_detector_starts_empty() {
    let detector = LeakDetector::with_default_thresholds();
    assert_eq!(detector.tracked_count(), 0);
}

/// Track allocation adds to detector.
#[test]
fn leak_detector_track_allocation() {
    let mut detector = LeakDetector::with_default_thresholds();

    detector.track_allocation(1, "Buffer 1", 1024);
    detector.track_allocation(2, "Buffer 2", 2048);
    detector.track_allocation(3, "Buffer 3", 4096);

    assert_eq!(detector.tracked_count(), 3);
}

/// Release allocation removes from detector.
#[test]
fn leak_detector_release_allocation() {
    let mut detector = LeakDetector::with_default_thresholds();

    detector.track_allocation(1, "Buffer 1", 1024);
    detector.track_allocation(2, "Buffer 2", 2048);

    let was_tracked = detector.release_allocation(1);
    assert!(was_tracked);
    assert_eq!(detector.tracked_count(), 1);

    let was_tracked = detector.release_allocation(1);
    assert!(!was_tracked, "already released");
}

/// Clear empties detector.
#[test]
fn leak_detector_clear() {
    let mut detector = LeakDetector::with_default_thresholds();

    detector.track_allocation(1, "Buffer 1", 1024);
    detector.track_allocation(2, "Buffer 2", 2048);

    detector.clear();

    assert_eq!(detector.tracked_count(), 0);
}

/// No leaks in empty detector.
#[test]
fn leak_detector_empty_no_leaks() {
    let mut detector = LeakDetector::with_default_thresholds();
    let leaks = detector.check();
    assert!(leaks.is_empty());
}

/// Expected allocations not reported as leaks.
#[test]
fn leak_detector_respects_expected() {
    // Use very short thresholds for testing
    let thresholds = LeakThresholds {
        warning_secs: 0,  // Immediate warning
        critical_secs: 1,
        min_size_bytes: 0,
    };
    let mut detector = LeakDetector::new(thresholds);

    detector.track_allocation(1, "Expected Resource", 1024);
    detector.mark_expected(1);

    // Even with 0 second threshold, expected allocations aren't leaks
    let leaks = detector.check();
    assert!(leaks.is_empty(), "expected allocations should not be leaks");
}

/// Unreleased buffer detected as leak with zero threshold.
#[test]
fn leak_detector_unreleased_buffer() {
    // Use very short thresholds for testing
    let thresholds = LeakThresholds {
        warning_secs: 0,  // Immediate warning
        critical_secs: 1,
        min_size_bytes: 0,
    };
    let mut detector = LeakDetector::new(thresholds);

    detector.track_allocation(1, "Leaked Buffer", 1024);

    // Check immediately with 0 second warning threshold
    let leaks = detector.check();
    assert!(!leaks.is_empty(), "should detect unreleased buffer");
}

/// Long-lived marked resources excluded from leak detection.
#[test]
fn leak_detector_excludes_marked_long_lived() {
    let thresholds = LeakThresholds {
        warning_secs: 0,
        critical_secs: 1,
        min_size_bytes: 0,
    };
    let mut detector = LeakDetector::new(thresholds);

    // Static mesh - expected to live forever
    detector.track_allocation(1, "Static Mesh", 1024 * 1024);
    detector.mark_expected(1);

    // Transient resource - should be detected
    detector.track_allocation(2, "Transient", 1024);

    let leaks = detector.check();
    assert_eq!(leaks.len(), 1);
    assert_eq!(leaks[0].label, Some("Transient".to_string()));
}

/// Accumulating allocations detected.
#[test]
fn leak_detector_accumulating_allocations() {
    let thresholds = LeakThresholds {
        warning_secs: 0,
        critical_secs: 1,
        min_size_bytes: 0,
    };
    let mut detector = LeakDetector::new(thresholds);

    // Simulate allocation accumulation over time
    for i in 0..10 {
        detector.track_allocation(i as u64, &format!("Accumulated {}", i), 1024 * 1024);
    }

    let leaks = detector.check();
    assert_eq!(leaks.len(), 10, "all accumulated allocations should be flagged");
}

/// Pipeline tracking with resource type.
#[test]
fn leak_detector_typed_allocations() {
    let thresholds = LeakThresholds {
        warning_secs: 0,
        critical_secs: 1,
        min_size_bytes: 0,
    };
    let mut detector = LeakDetector::new(thresholds);

    // Track with resource type
    detector.track_allocation_typed(1, "Pipeline 1", 4096, ResourceType::Pipeline);
    detector.track_allocation_typed(2, "Pipeline 2", 4096, ResourceType::Pipeline);

    let leaks = detector.check();
    assert_eq!(leaks.len(), 2);

    for leak in &leaks {
        assert_eq!(leak.resource_type, ResourceType::Pipeline);
    }
}

/// Bind group tracking.
#[test]
fn leak_detector_bind_group_tracking() {
    let thresholds = LeakThresholds {
        warning_secs: 0,
        critical_secs: 1,
        min_size_bytes: 0,
    };
    let mut detector = LeakDetector::new(thresholds);

    // Simulate bind group accumulation
    for i in 0..20 {
        detector.track_allocation_typed(
            i as u64,
            &format!("BindGroup {}", i),
            256,
            ResourceType::BindGroup,
        );
    }

    let leaks = detector.check();
    assert_eq!(leaks.len(), 20);
}

/// Query set cleanup verification.
#[test]
fn leak_detector_query_set_cleanup() {
    let thresholds = LeakThresholds {
        warning_secs: 0,
        critical_secs: 1,
        min_size_bytes: 0,
    };
    let mut detector = LeakDetector::new(thresholds);

    detector.track_allocation_typed(1, "Query Set", 512, ResourceType::QuerySet);

    // Before cleanup - should be leak
    let leaks = detector.check();
    assert_eq!(leaks.len(), 1);

    // After cleanup - no leaks
    detector.release_allocation(1);
    let leaks = detector.check();
    assert!(leaks.is_empty());
}

/// Check statistics tracking.
#[test]
fn leak_detector_stats() {
    let thresholds = LeakThresholds {
        warning_secs: 0,
        critical_secs: 1,
        min_size_bytes: 0,
    };
    let mut detector = LeakDetector::new(thresholds);

    detector.track_allocation(1, "Buffer 1", 1024);
    detector.track_allocation(2, "Buffer 2", 2048);
    detector.track_allocation(3, "Buffer 3", 4096);
    detector.release_allocation(1);

    let stats = detector.stats();
    assert_eq!(stats.total_tracked, 3);
    assert_eq!(stats.total_released, 1);
    assert_eq!(stats.current_tracked, 2);
}

/// Threshold configuration works.
#[test]
fn leak_detector_threshold_configuration() {
    // High threshold - no leaks detected immediately
    let high_thresholds = LeakThresholds {
        warning_secs: 60,
        critical_secs: 300,
        min_size_bytes: 0,
    };
    let mut detector = LeakDetector::new(high_thresholds);

    detector.track_allocation(1, "Buffer", 1024);

    let leaks = detector.check();
    assert!(leaks.is_empty(), "60 second threshold should not trigger immediately");
}

/// Minimum size threshold filters small allocations.
#[test]
fn leak_detector_min_size_filter() {
    let thresholds = LeakThresholds {
        warning_secs: 0,
        critical_secs: 1,
        min_size_bytes: 1000, // Only report leaks >= 1000 bytes
    };
    let mut detector = LeakDetector::new(thresholds);

    detector.track_allocation(1, "Small", 500);  // Below threshold
    detector.track_allocation(2, "Large", 2000); // Above threshold

    let leaks = detector.check();
    assert_eq!(leaks.len(), 1);
    assert_eq!(leaks[0].size_bytes, 2000);
}

/// Total bytes calculation.
#[test]
fn leak_detector_total_bytes() {
    let mut detector = LeakDetector::with_default_thresholds();

    detector.track_allocation(1, "Buffer 1", 1024);
    detector.track_allocation(2, "Buffer 2", 2048);
    detector.track_allocation(3, "Buffer 3", 4096);

    assert_eq!(detector.total_bytes(), 1024 + 2048 + 4096);
}

/// Critical leak detection.
#[test]
fn leak_detector_critical_only() {
    let thresholds = LeakThresholds {
        warning_secs: 0,
        critical_secs: 0, // Immediate critical
        min_size_bytes: 0,
    };
    let mut detector = LeakDetector::new(thresholds);

    detector.track_allocation(1, "Critical", 1024);

    let critical_leaks = detector.check_critical_only();
    assert_eq!(critical_leaks.len(), 1);
}

/// Temporary allocation marking (stricter checking).
#[test]
fn leak_detector_temporary_marking() {
    let thresholds = LeakThresholds {
        warning_secs: 10, // 10 second warning
        critical_secs: 60,
        min_size_bytes: 0,
    };
    let mut detector = LeakDetector::new(thresholds);

    detector.track_allocation(1, "Temporary", 1024);
    detector.mark_temporary(1);

    // Temporary allocations get stricter thresholds (warning_secs / 2)
    // But since we're checking immediately and threshold is 5 secs, still no leak
    let leaks = detector.check();
    assert!(leaks.is_empty());
}

// =============================================================================
// SECTION 6 -- Edge Cases (15+)
// =============================================================================

/// Zero size allocation.
#[test]
fn edge_zero_size_allocation() {
    let mut tracker = MemoryTracker::with_default_budget();

    let id = tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 0, None);

    let info = tracker.get_allocation(id).unwrap();
    assert_eq!(info.size_bytes, 0);
    assert_eq!(tracker.stats().current_bytes, 0);
}

/// Very large allocation.
#[test]
fn edge_very_large_allocation() {
    let mut tracker = MemoryTracker::with_default_budget();

    let large_size = 4 * 1024 * 1024 * 1024u64; // 4 GB
    let id = tracker.track_resource(
        ResourceType::Texture,
        MemoryType::DeviceLocal,
        large_size,
        Some("Huge Texture"),
    );

    let info = tracker.get_allocation(id).unwrap();
    assert_eq!(info.size_bytes, large_size);
}

/// Rapid alloc/dealloc cycles.
#[test]
fn edge_rapid_alloc_dealloc() {
    let mut tracker = MemoryTracker::with_default_budget();

    for _ in 0..1000 {
        let id = tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);
        tracker.untrack(id);
    }

    assert_eq!(tracker.stats().total_allocations, 1000);
    assert_eq!(tracker.stats().total_deallocations, 1000);
    assert_eq!(tracker.stats().current_allocations, 0);
}

/// Maximum allocation count stress test.
#[test]
fn edge_maximum_allocation_count() {
    let mut tracker = MemoryTracker::with_default_budget();
    let mut ids = Vec::new();

    for i in 0..10000 {
        let id = tracker.track_resource(
            ResourceType::Buffer,
            MemoryType::DeviceLocal,
            1,
            Some(&format!("{}", i)),
        );
        ids.push(id);
    }

    assert_eq!(tracker.stats().current_allocations, 10000);

    // Cleanup
    for id in ids {
        tracker.untrack(id);
    }

    assert_eq!(tracker.stats().current_allocations, 0);
}

/// ID values increment monotonically.
#[test]
fn edge_id_increment() {
    let mut tracker = MemoryTracker::with_default_budget();
    let mut prev_id = 0u64;

    for _ in 0..100 {
        let id = tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1, None);
        assert!(id > prev_id, "IDs should be monotonically increasing");
        prev_id = id;
    }
}

/// Empty tracker operations.
#[test]
fn edge_empty_tracker_operations() {
    let mut tracker = MemoryTracker::with_default_budget();

    // All these should be safe on empty tracker
    assert!(tracker.get_allocation(1).is_none());
    assert!(!tracker.untrack(1));
    assert!(tracker.allocations().is_empty());
    assert_eq!(tracker.stats().current_allocations, 0);

    let snapshot = tracker.snapshot();
    assert!(snapshot.allocations.is_empty());

    let summary = tracker.summary();
    assert!(!summary.is_empty());

    tracker.clear(); // Should be no-op
    assert_eq!(tracker.stats().current_allocations, 0);
}

/// Concurrent-style patterns (single-threaded simulation).
#[test]
fn edge_concurrent_style_pattern() {
    let mut tracker = MemoryTracker::with_default_budget();

    // Simulate interleaved operations from multiple "workers"
    let mut worker1_ids = Vec::new();
    let mut worker2_ids = Vec::new();

    for _ in 0..50 {
        // Worker 1 allocates
        let id1 = tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);
        worker1_ids.push(id1);

        // Worker 2 allocates
        let id2 = tracker.track_resource(ResourceType::Texture, MemoryType::DeviceLocal, 2048, None);
        worker2_ids.push(id2);
    }

    // Worker 1 releases all
    for id in worker1_ids {
        tracker.untrack(id);
    }

    // Worker 2 releases all
    for id in worker2_ids {
        tracker.untrack(id);
    }

    assert_eq!(tracker.stats().current_allocations, 0);
    assert_eq!(tracker.stats().total_allocations, 100);
}

/// Labels with special characters.
#[test]
fn edge_labels_special_chars() {
    let mut tracker = MemoryTracker::with_default_budget();

    let special_labels = [
        "Buffer with spaces",
        "Buffer-with-dashes",
        "Buffer_with_underscores",
        "Buffer.with.dots",
        "Buffer:with:colons",
        "Buffer<with>angle<brackets>",
        "",
    ];

    for label in &special_labels {
        let id = tracker.track_resource(
            ResourceType::Buffer,
            MemoryType::DeviceLocal,
            100,
            Some(*label),
        );
        let info = tracker.get_allocation(id).unwrap();
        assert_eq!(info.label, Some(label.to_string()));
    }
}

/// Untrack same ID multiple times.
#[test]
fn edge_untrack_multiple_times() {
    let mut tracker = MemoryTracker::with_default_budget();

    let id = tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);

    assert!(tracker.untrack(id), "first untrack should succeed");
    assert!(!tracker.untrack(id), "second untrack should fail");
    assert!(!tracker.untrack(id), "third untrack should fail");
}

/// Clear after heavy usage.
#[test]
fn edge_clear_after_heavy_usage() {
    let mut tracker = MemoryTracker::with_default_budget();

    // Heavy usage
    for _ in 0..1000 {
        tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);
    }

    assert_eq!(tracker.stats().current_allocations, 1000);

    tracker.clear();

    assert_eq!(tracker.stats().current_allocations, 0);
    assert_eq!(tracker.stats().current_bytes, 0);
    assert!(tracker.allocations().is_empty());
}

/// Stats deallocation saturates at zero.
#[test]
fn edge_stats_saturates() {
    let mut stats = AllocationStats::new();

    // Try to dealloc without prior alloc
    let info = AllocationInfo::new(1, ResourceType::Buffer, MemoryType::DeviceLocal, 1000);
    stats.record_deallocation(&info);

    assert_eq!(stats.current_allocations, 0);
    assert_eq!(stats.current_bytes, 0);
}

/// Mix of all resource types.
#[test]
fn edge_all_resource_types() {
    let mut tracker = MemoryTracker::with_default_budget();

    tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 100, None);
    tracker.track_resource(ResourceType::Texture, MemoryType::DeviceLocal, 200, None);
    tracker.track_resource(ResourceType::QuerySet, MemoryType::DeviceLocal, 300, None);
    tracker.track_resource(ResourceType::BindGroup, MemoryType::DeviceLocal, 400, None);
    tracker.track_resource(ResourceType::Pipeline, MemoryType::DeviceLocal, 500, None);
    tracker.track_resource(ResourceType::Other, MemoryType::DeviceLocal, 600, None);

    assert_eq!(tracker.stats().current_allocations, 6);

    let stats = tracker.stats();
    assert_eq!(stats.bytes_by_type.get(&ResourceType::Buffer), Some(&100));
    assert_eq!(stats.bytes_by_type.get(&ResourceType::Texture), Some(&200));
    assert_eq!(stats.bytes_by_type.get(&ResourceType::QuerySet), Some(&300));
    assert_eq!(stats.bytes_by_type.get(&ResourceType::BindGroup), Some(&400));
    assert_eq!(stats.bytes_by_type.get(&ResourceType::Pipeline), Some(&500));
    assert_eq!(stats.bytes_by_type.get(&ResourceType::Other), Some(&600));
}

/// Mix of all memory types.
#[test]
fn edge_all_memory_types() {
    let mut tracker = MemoryTracker::with_default_budget();

    tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 100, None);
    tracker.track_resource(ResourceType::Buffer, MemoryType::HostVisible, 200, None);
    tracker.track_resource(ResourceType::Buffer, MemoryType::HostCoherent, 300, None);
    tracker.track_resource(ResourceType::Buffer, MemoryType::HostCached, 400, None);

    let budget = tracker.budget();
    assert_eq!(budget.device_local_used, 100);
    assert_eq!(budget.host_visible_used, 200 + 300 + 400);
}

/// Net allocations calculation.
#[test]
fn edge_net_allocations() {
    let mut stats = AllocationStats::new();

    // 10 allocations
    for _ in 0..10 {
        let info = AllocationInfo::new(0, ResourceType::Buffer, MemoryType::DeviceLocal, 100);
        stats.record_allocation(&info);
    }

    // 3 deallocations
    for _ in 0..3 {
        let info = AllocationInfo::new(0, ResourceType::Buffer, MemoryType::DeviceLocal, 100);
        stats.record_deallocation(&info);
    }

    assert_eq!(stats.net_allocations(), 7);
}

// =============================================================================
// SECTION 7 -- Reporting (10+)
// =============================================================================

/// Format bytes for bytes.
#[test]
fn report_format_bytes_bytes() {
    assert_eq!(format_bytes(0), "0 B");
    assert_eq!(format_bytes(1), "1 B");
    assert_eq!(format_bytes(512), "512 B");
    assert_eq!(format_bytes(1023), "1023 B");
}

/// Format bytes for kilobytes.
#[test]
fn report_format_bytes_kb() {
    assert_eq!(format_bytes(1024), "1.00 KB");
    assert_eq!(format_bytes(1536), "1.50 KB");
    assert_eq!(format_bytes(10 * 1024), "10.00 KB");
}

/// Format bytes for megabytes.
#[test]
fn report_format_bytes_mb() {
    assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
    assert_eq!(format_bytes(512 * 1024 * 1024), "512.00 MB");
}

/// Format bytes for gigabytes.
#[test]
fn report_format_bytes_gb() {
    assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
    assert_eq!(format_bytes(4 * 1024 * 1024 * 1024u64), "4.00 GB");
}

/// Summary contains key sections.
#[test]
fn report_summary_sections() {
    let mut tracker = MemoryTracker::with_default_budget();
    tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);

    let summary = tracker.summary();

    assert!(summary.contains("GPU Memory Summary"));
    assert!(summary.contains("Current:"));
    assert!(summary.contains("Peak:"));
    assert!(summary.contains("Budget:"));
}

/// Summary contains resource type breakdown.
#[test]
fn report_summary_type_breakdown() {
    let mut tracker = MemoryTracker::with_default_budget();
    tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);
    tracker.track_resource(ResourceType::Texture, MemoryType::DeviceLocal, 4096, None);

    let summary = tracker.summary();

    assert!(summary.contains("Buffer"));
    assert!(summary.contains("Texture"));
}

/// Stats formatted methods.
#[test]
fn report_stats_formatted() {
    let mut stats = AllocationStats::new();
    let info = AllocationInfo::new(1, ResourceType::Buffer, MemoryType::DeviceLocal, 2 * 1024 * 1024);
    stats.record_allocation(&info);

    assert_eq!(stats.current_bytes_formatted(), "2.00 MB");
    assert_eq!(stats.peak_bytes_formatted(), "2.00 MB");
}

/// MemoryType display.
#[test]
fn report_memory_type_display() {
    assert_eq!(format!("{}", MemoryType::DeviceLocal), "Device Local");
    assert_eq!(format!("{}", MemoryType::HostVisible), "Host Visible");
    assert_eq!(format!("{}", MemoryType::HostCoherent), "Host Coherent");
    assert_eq!(format!("{}", MemoryType::HostCached), "Host Cached");
}

/// ResourceType display.
#[test]
fn report_resource_type_display() {
    assert_eq!(format!("{}", ResourceType::Buffer), "Buffer");
    assert_eq!(format!("{}", ResourceType::Texture), "Texture");
    assert_eq!(format!("{}", ResourceType::QuerySet), "Query Set");
    assert_eq!(format!("{}", ResourceType::BindGroup), "Bind Group");
    assert_eq!(format!("{}", ResourceType::Pipeline), "Pipeline");
    assert_eq!(format!("{}", ResourceType::Other), "Other");
}

/// ResourceType display_name.
#[test]
fn report_resource_type_display_name() {
    assert_eq!(ResourceType::Buffer.display_name(), "Buffer");
    assert_eq!(ResourceType::Texture.display_name(), "Texture");
    assert_eq!(ResourceType::QuerySet.display_name(), "Query Set");
    assert_eq!(ResourceType::BindGroup.display_name(), "Bind Group");
    assert_eq!(ResourceType::Pipeline.display_name(), "Pipeline");
    assert_eq!(ResourceType::Other.display_name(), "Other");
}

/// MemoryType is_mappable.
#[test]
fn report_memory_type_mappable() {
    assert!(!MemoryType::DeviceLocal.is_mappable());
    assert!(MemoryType::HostVisible.is_mappable());
    assert!(MemoryType::HostCoherent.is_mappable());
    assert!(MemoryType::HostCached.is_mappable());
}

/// AllocationInfo age tracking.
#[test]
fn report_allocation_age() {
    let info = AllocationInfo::new(1, ResourceType::Buffer, MemoryType::DeviceLocal, 1024);

    // Just created, age should be tiny
    assert!(info.age_secs() < 1.0);

    thread::sleep(Duration::from_millis(50));

    // After 50ms, age should be at least 0.05 seconds
    assert!(info.age_secs() >= 0.05);
}

// =============================================================================
// SECTION 8 -- Additional Coverage Tests
// =============================================================================

/// Snapshot budget is captured.
#[test]
fn snapshot_captures_budget() {
    let mut tracker = MemoryTracker::with_default_budget();
    tracker.track_resource(
        ResourceType::Buffer,
        MemoryType::DeviceLocal,
        1024 * 1024,
        None,
    );

    let snapshot = tracker.snapshot();

    assert!(snapshot.budget.device_local_used > 0);
    assert_eq!(snapshot.budget.device_local_used, 1024 * 1024);
}

/// Diff with negative bytes delta displays correctly.
#[test]
fn diff_negative_bytes_display() {
    let diff = MemoryDiff {
        added: vec![],
        removed: vec![1, 2, 3],
        bytes_delta: -3072,
    };

    let display = format!("{}", diff);
    assert!(display.contains("-3.00 KB") || display.contains("-3072"));
}

/// Multiple resource type tracking simultaneously.
#[test]
fn multi_type_simultaneous() {
    let mut tracker = MemoryTracker::with_default_budget();

    // Track many of each type
    for i in 0..100 {
        let resource_type = match i % 6 {
            0 => ResourceType::Buffer,
            1 => ResourceType::Texture,
            2 => ResourceType::QuerySet,
            3 => ResourceType::BindGroup,
            4 => ResourceType::Pipeline,
            _ => ResourceType::Other,
        };

        tracker.track_resource(resource_type, MemoryType::DeviceLocal, 100, None);
    }

    assert_eq!(tracker.stats().current_allocations, 100);

    // Verify each type has some allocations
    let stats = tracker.stats();
    for resource_type in [
        ResourceType::Buffer,
        ResourceType::Texture,
        ResourceType::QuerySet,
        ResourceType::BindGroup,
        ResourceType::Pipeline,
        ResourceType::Other,
    ] {
        assert!(
            stats.bytes_by_type.get(&resource_type).is_some(),
            "{:?} should have allocations",
            resource_type
        );
    }
}

/// Verify peak never decreases.
#[test]
fn peak_never_decreases() {
    let mut tracker = MemoryTracker::with_default_budget();

    let mut prev_peak_bytes = 0u64;
    let mut prev_peak_allocs = 0u64;

    for _ in 0..50 {
        // Add some
        let ids: Vec<_> = (0..5)
            .map(|_| tracker.track_resource(
                ResourceType::Buffer,
                MemoryType::DeviceLocal,
                1024,
                None,
            ))
            .collect();

        // Current peak should be >= previous
        assert!(tracker.stats().peak_bytes >= prev_peak_bytes);
        assert!(tracker.stats().peak_allocations >= prev_peak_allocs);

        prev_peak_bytes = tracker.stats().peak_bytes;
        prev_peak_allocs = tracker.stats().peak_allocations;

        // Remove some
        for id in ids.iter().take(3) {
            tracker.untrack(*id);
        }

        // Peak should still be >= previous
        assert!(tracker.stats().peak_bytes >= prev_peak_bytes);
        assert!(tracker.stats().peak_allocations >= prev_peak_allocs);
    }
}

/// Total bytes allocated accumulates.
#[test]
fn total_bytes_accumulates() {
    let mut tracker = MemoryTracker::with_default_budget();

    for _ in 0..100 {
        let id = tracker.track_resource(
            ResourceType::Buffer,
            MemoryType::DeviceLocal,
            1000,
            None,
        );
        tracker.untrack(id);
    }

    // Current should be 0
    assert_eq!(tracker.stats().current_bytes, 0);

    // Total allocated should be 100 * 1000
    assert_eq!(tracker.stats().total_bytes_allocated, 100_000);
}
