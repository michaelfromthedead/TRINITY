//! Whitebox tests for GPU Memory Tracking (T-WGPU-P7.4.2)
//!
//! Tests internal implementation details of the memory tracking module:
//! - MemoryType variants and inference logic
//! - ResourceType variants and display
//! - AllocationInfo construction and field access
//! - AllocationStats aggregation and peak tracking
//! - MemoryBudget calculations and utilization
//! - MemoryTracker lifecycle and resource management
//! - MemorySnapshot and MemoryDiff comparison
//! - LeakDetector threshold-based detection
//!
//! Coverage: 150+ tests across all components

use renderer_backend::profiling::memory::{
    format_bytes, AllocationInfo, AllocationStats, LeakDetector, MemoryBudget, MemoryDiff,
    MemorySnapshot, MemoryTracker, MemoryType, ResourceType,
};
use std::collections::HashMap;
use std::thread;
use std::time::Duration;

// =============================================================================
// MemoryType Tests (25 tests)
// =============================================================================

mod memory_type_tests {
    use super::*;

    #[test]
    fn test_device_local_variant() {
        let mt = MemoryType::DeviceLocal;
        assert_eq!(mt, MemoryType::DeviceLocal);
    }

    #[test]
    fn test_host_visible_variant() {
        let mt = MemoryType::HostVisible;
        assert_eq!(mt, MemoryType::HostVisible);
    }

    #[test]
    fn test_host_coherent_variant() {
        let mt = MemoryType::HostCoherent;
        assert_eq!(mt, MemoryType::HostCoherent);
    }

    #[test]
    fn test_host_cached_variant() {
        let mt = MemoryType::HostCached;
        assert_eq!(mt, MemoryType::HostCached);
    }

    #[test]
    fn test_from_buffer_usage_vertex_returns_device_local() {
        let usage = wgpu::BufferUsages::VERTEX;
        assert_eq!(MemoryType::from_buffer_usage(usage), MemoryType::DeviceLocal);
    }

    #[test]
    fn test_from_buffer_usage_index_returns_device_local() {
        let usage = wgpu::BufferUsages::INDEX;
        assert_eq!(MemoryType::from_buffer_usage(usage), MemoryType::DeviceLocal);
    }

    #[test]
    fn test_from_buffer_usage_uniform_returns_device_local() {
        let usage = wgpu::BufferUsages::UNIFORM;
        assert_eq!(MemoryType::from_buffer_usage(usage), MemoryType::DeviceLocal);
    }

    #[test]
    fn test_from_buffer_usage_storage_returns_device_local() {
        let usage = wgpu::BufferUsages::STORAGE;
        assert_eq!(MemoryType::from_buffer_usage(usage), MemoryType::DeviceLocal);
    }

    #[test]
    fn test_from_buffer_usage_map_read_returns_host_coherent() {
        let usage = wgpu::BufferUsages::MAP_READ;
        assert_eq!(
            MemoryType::from_buffer_usage(usage),
            MemoryType::HostCoherent
        );
    }

    #[test]
    fn test_from_buffer_usage_map_write_returns_host_cached() {
        let usage = wgpu::BufferUsages::MAP_WRITE;
        assert_eq!(MemoryType::from_buffer_usage(usage), MemoryType::HostCached);
    }

    #[test]
    fn test_from_buffer_usage_copy_dst_returns_device_local() {
        let usage = wgpu::BufferUsages::COPY_DST;
        assert_eq!(MemoryType::from_buffer_usage(usage), MemoryType::DeviceLocal);
    }

    #[test]
    fn test_from_buffer_usage_copy_src_returns_device_local() {
        let usage = wgpu::BufferUsages::COPY_SRC;
        assert_eq!(MemoryType::from_buffer_usage(usage), MemoryType::DeviceLocal);
    }

    #[test]
    fn test_from_buffer_usage_combined_vertex_copy_dst() {
        let usage = wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST;
        assert_eq!(MemoryType::from_buffer_usage(usage), MemoryType::DeviceLocal);
    }

    #[test]
    fn test_from_buffer_usage_combined_map_read_copy_dst() {
        let usage = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
        assert_eq!(
            MemoryType::from_buffer_usage(usage),
            MemoryType::HostCoherent
        );
    }

    #[test]
    fn test_from_buffer_usage_combined_map_write_copy_src() {
        let usage = wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC;
        assert_eq!(MemoryType::from_buffer_usage(usage), MemoryType::HostCached);
    }

    #[test]
    fn test_from_buffer_usage_empty_returns_device_local() {
        let usage = wgpu::BufferUsages::empty();
        assert_eq!(MemoryType::from_buffer_usage(usage), MemoryType::DeviceLocal);
    }

    #[test]
    fn test_from_texture_usage_render_attachment_returns_device_local() {
        let usage = wgpu::TextureUsages::RENDER_ATTACHMENT;
        assert_eq!(
            MemoryType::from_texture_usage(usage),
            MemoryType::DeviceLocal
        );
    }

    #[test]
    fn test_from_texture_usage_copy_src_returns_host_visible() {
        let usage = wgpu::TextureUsages::COPY_SRC;
        assert_eq!(
            MemoryType::from_texture_usage(usage),
            MemoryType::HostVisible
        );
    }

    #[test]
    fn test_from_texture_usage_copy_dst_returns_device_local() {
        let usage = wgpu::TextureUsages::COPY_DST;
        assert_eq!(
            MemoryType::from_texture_usage(usage),
            MemoryType::DeviceLocal
        );
    }

    #[test]
    fn test_from_texture_usage_texture_binding_returns_device_local() {
        let usage = wgpu::TextureUsages::TEXTURE_BINDING;
        assert_eq!(
            MemoryType::from_texture_usage(usage),
            MemoryType::DeviceLocal
        );
    }

    #[test]
    fn test_is_mappable_device_local_false() {
        assert!(!MemoryType::DeviceLocal.is_mappable());
    }

    #[test]
    fn test_is_mappable_host_visible_true() {
        assert!(MemoryType::HostVisible.is_mappable());
    }

    #[test]
    fn test_is_mappable_host_coherent_true() {
        assert!(MemoryType::HostCoherent.is_mappable());
    }

    #[test]
    fn test_is_mappable_host_cached_true() {
        assert!(MemoryType::HostCached.is_mappable());
    }

    #[test]
    fn test_memory_type_display_device_local() {
        assert_eq!(format!("{}", MemoryType::DeviceLocal), "Device Local");
    }

    #[test]
    fn test_memory_type_display_host_visible() {
        assert_eq!(format!("{}", MemoryType::HostVisible), "Host Visible");
    }

    #[test]
    fn test_memory_type_display_host_coherent() {
        assert_eq!(format!("{}", MemoryType::HostCoherent), "Host Coherent");
    }

    #[test]
    fn test_memory_type_display_host_cached() {
        assert_eq!(format!("{}", MemoryType::HostCached), "Host Cached");
    }

    #[test]
    fn test_memory_type_equality() {
        assert_eq!(MemoryType::DeviceLocal, MemoryType::DeviceLocal);
        assert_ne!(MemoryType::DeviceLocal, MemoryType::HostVisible);
    }

    #[test]
    fn test_memory_type_hash() {
        let mut map = HashMap::new();
        map.insert(MemoryType::DeviceLocal, 100);
        map.insert(MemoryType::HostVisible, 200);
        assert_eq!(map.get(&MemoryType::DeviceLocal), Some(&100));
        assert_eq!(map.get(&MemoryType::HostVisible), Some(&200));
    }
}

// =============================================================================
// ResourceType Tests (18 tests)
// =============================================================================

mod resource_type_tests {
    use super::*;

    #[test]
    fn test_buffer_variant() {
        let rt = ResourceType::Buffer;
        assert_eq!(rt, ResourceType::Buffer);
    }

    #[test]
    fn test_texture_variant() {
        let rt = ResourceType::Texture;
        assert_eq!(rt, ResourceType::Texture);
    }

    #[test]
    fn test_query_set_variant() {
        let rt = ResourceType::QuerySet;
        assert_eq!(rt, ResourceType::QuerySet);
    }

    #[test]
    fn test_bind_group_variant() {
        let rt = ResourceType::BindGroup;
        assert_eq!(rt, ResourceType::BindGroup);
    }

    #[test]
    fn test_pipeline_variant() {
        let rt = ResourceType::Pipeline;
        assert_eq!(rt, ResourceType::Pipeline);
    }

    #[test]
    fn test_other_variant() {
        let rt = ResourceType::Other;
        assert_eq!(rt, ResourceType::Other);
    }

    #[test]
    fn test_display_name_buffer() {
        assert_eq!(ResourceType::Buffer.display_name(), "Buffer");
    }

    #[test]
    fn test_display_name_texture() {
        assert_eq!(ResourceType::Texture.display_name(), "Texture");
    }

    #[test]
    fn test_display_name_query_set() {
        assert_eq!(ResourceType::QuerySet.display_name(), "Query Set");
    }

    #[test]
    fn test_display_name_bind_group() {
        assert_eq!(ResourceType::BindGroup.display_name(), "Bind Group");
    }

    #[test]
    fn test_display_name_pipeline() {
        assert_eq!(ResourceType::Pipeline.display_name(), "Pipeline");
    }

    #[test]
    fn test_display_name_other() {
        assert_eq!(ResourceType::Other.display_name(), "Other");
    }

    #[test]
    fn test_display_trait_buffer() {
        assert_eq!(format!("{}", ResourceType::Buffer), "Buffer");
    }

    #[test]
    fn test_display_trait_texture() {
        assert_eq!(format!("{}", ResourceType::Texture), "Texture");
    }

    #[test]
    fn test_resource_type_equality() {
        assert_eq!(ResourceType::Buffer, ResourceType::Buffer);
        assert_ne!(ResourceType::Buffer, ResourceType::Texture);
    }

    #[test]
    fn test_resource_type_hash_as_key() {
        let mut map: HashMap<ResourceType, u64> = HashMap::new();
        map.insert(ResourceType::Buffer, 1024);
        map.insert(ResourceType::Texture, 4096);
        map.insert(ResourceType::QuerySet, 256);
        assert_eq!(map.get(&ResourceType::Buffer), Some(&1024));
        assert_eq!(map.get(&ResourceType::Texture), Some(&4096));
        assert_eq!(map.get(&ResourceType::QuerySet), Some(&256));
    }

    #[test]
    fn test_resource_type_clone() {
        let rt = ResourceType::Pipeline;
        let rt_clone = rt;
        assert_eq!(rt, rt_clone);
    }

    #[test]
    fn test_resource_type_debug() {
        let rt = ResourceType::BindGroup;
        let debug_str = format!("{:?}", rt);
        assert!(debug_str.contains("BindGroup"));
    }
}

// =============================================================================
// AllocationInfo Tests (28 tests)
// =============================================================================

mod allocation_info_tests {
    use super::*;

    #[test]
    fn test_new_with_all_fields() {
        let info = AllocationInfo::new(1, ResourceType::Buffer, MemoryType::DeviceLocal, 1024);
        assert_eq!(info.id, 1);
        assert_eq!(info.resource_type, ResourceType::Buffer);
        assert_eq!(info.memory_type, MemoryType::DeviceLocal);
        assert_eq!(info.size_bytes, 1024);
        assert!(info.label.is_none());
    }

    #[test]
    fn test_with_label_all_fields() {
        let info =
            AllocationInfo::with_label(2, ResourceType::Texture, MemoryType::HostVisible, 4096, "Albedo");
        assert_eq!(info.id, 2);
        assert_eq!(info.resource_type, ResourceType::Texture);
        assert_eq!(info.memory_type, MemoryType::HostVisible);
        assert_eq!(info.size_bytes, 4096);
        assert_eq!(info.label, Some("Albedo".to_string()));
    }

    #[test]
    fn test_id_uniqueness_manual() {
        let info1 = AllocationInfo::new(1, ResourceType::Buffer, MemoryType::DeviceLocal, 1024);
        let info2 = AllocationInfo::new(2, ResourceType::Buffer, MemoryType::DeviceLocal, 1024);
        assert_ne!(info1.id, info2.id);
    }

    #[test]
    fn test_size_tracking_zero() {
        let info = AllocationInfo::new(1, ResourceType::Buffer, MemoryType::DeviceLocal, 0);
        assert_eq!(info.size_bytes, 0);
    }

    #[test]
    fn test_size_tracking_large() {
        let large_size = 4 * 1024 * 1024 * 1024u64; // 4 GB
        let info = AllocationInfo::new(1, ResourceType::Texture, MemoryType::DeviceLocal, large_size);
        assert_eq!(info.size_bytes, large_size);
    }

    #[test]
    fn test_timestamp_is_recent() {
        let info = AllocationInfo::new(1, ResourceType::Buffer, MemoryType::DeviceLocal, 1024);
        // Timestamp should be within the last second
        assert!(info.age_secs() < 1.0);
    }

    #[test]
    fn test_age_secs_increases() {
        let info = AllocationInfo::new(1, ResourceType::Buffer, MemoryType::DeviceLocal, 1024);
        thread::sleep(Duration::from_millis(10));
        assert!(info.age_secs() >= 0.01);
    }

    #[test]
    fn test_resource_type_buffer() {
        let info = AllocationInfo::new(1, ResourceType::Buffer, MemoryType::DeviceLocal, 1024);
        assert_eq!(info.resource_type, ResourceType::Buffer);
    }

    #[test]
    fn test_resource_type_texture() {
        let info = AllocationInfo::new(1, ResourceType::Texture, MemoryType::DeviceLocal, 4096);
        assert_eq!(info.resource_type, ResourceType::Texture);
    }

    #[test]
    fn test_resource_type_query_set() {
        let info = AllocationInfo::new(1, ResourceType::QuerySet, MemoryType::DeviceLocal, 256);
        assert_eq!(info.resource_type, ResourceType::QuerySet);
    }

    #[test]
    fn test_resource_type_bind_group() {
        let info = AllocationInfo::new(1, ResourceType::BindGroup, MemoryType::DeviceLocal, 128);
        assert_eq!(info.resource_type, ResourceType::BindGroup);
    }

    #[test]
    fn test_resource_type_pipeline() {
        let info = AllocationInfo::new(1, ResourceType::Pipeline, MemoryType::DeviceLocal, 2048);
        assert_eq!(info.resource_type, ResourceType::Pipeline);
    }

    #[test]
    fn test_resource_type_other() {
        let info = AllocationInfo::new(1, ResourceType::Other, MemoryType::DeviceLocal, 512);
        assert_eq!(info.resource_type, ResourceType::Other);
    }

    #[test]
    fn test_memory_type_device_local() {
        let info = AllocationInfo::new(1, ResourceType::Buffer, MemoryType::DeviceLocal, 1024);
        assert_eq!(info.memory_type, MemoryType::DeviceLocal);
    }

    #[test]
    fn test_memory_type_host_visible() {
        let info = AllocationInfo::new(1, ResourceType::Buffer, MemoryType::HostVisible, 1024);
        assert_eq!(info.memory_type, MemoryType::HostVisible);
    }

    #[test]
    fn test_memory_type_host_coherent() {
        let info = AllocationInfo::new(1, ResourceType::Buffer, MemoryType::HostCoherent, 1024);
        assert_eq!(info.memory_type, MemoryType::HostCoherent);
    }

    #[test]
    fn test_memory_type_host_cached() {
        let info = AllocationInfo::new(1, ResourceType::Buffer, MemoryType::HostCached, 1024);
        assert_eq!(info.memory_type, MemoryType::HostCached);
    }

    #[test]
    fn test_label_some() {
        let info = AllocationInfo::with_label(
            1,
            ResourceType::Buffer,
            MemoryType::DeviceLocal,
            1024,
            "Test Label",
        );
        assert_eq!(info.label, Some("Test Label".to_string()));
    }

    #[test]
    fn test_label_none() {
        let info = AllocationInfo::new(1, ResourceType::Buffer, MemoryType::DeviceLocal, 1024);
        assert!(info.label.is_none());
    }

    #[test]
    fn test_label_empty_string() {
        let info = AllocationInfo::with_label(1, ResourceType::Buffer, MemoryType::DeviceLocal, 1024, "");
        assert_eq!(info.label, Some("".to_string()));
    }

    #[test]
    fn test_label_unicode() {
        let info = AllocationInfo::with_label(
            1,
            ResourceType::Texture,
            MemoryType::DeviceLocal,
            1024,
            "Texture_2D_albedo",
        );
        assert_eq!(info.label, Some("Texture_2D_albedo".to_string()));
    }

    #[test]
    fn test_clone() {
        let info = AllocationInfo::with_label(
            1,
            ResourceType::Buffer,
            MemoryType::DeviceLocal,
            1024,
            "Original",
        );
        let cloned = info.clone();
        assert_eq!(info.id, cloned.id);
        assert_eq!(info.size_bytes, cloned.size_bytes);
        assert_eq!(info.label, cloned.label);
    }

    #[test]
    fn test_debug_format() {
        let info = AllocationInfo::new(1, ResourceType::Buffer, MemoryType::DeviceLocal, 1024);
        let debug_str = format!("{:?}", info);
        assert!(debug_str.contains("AllocationInfo"));
        assert!(debug_str.contains("1024"));
    }

    #[test]
    fn test_max_id_value() {
        let info = AllocationInfo::new(u64::MAX, ResourceType::Buffer, MemoryType::DeviceLocal, 1024);
        assert_eq!(info.id, u64::MAX);
    }

    #[test]
    fn test_with_label_string_conversion() {
        let label = String::from("Dynamic Label");
        let info = AllocationInfo::with_label(
            1,
            ResourceType::Buffer,
            MemoryType::DeviceLocal,
            1024,
            label,
        );
        assert_eq!(info.label, Some("Dynamic Label".to_string()));
    }

    #[test]
    fn test_with_label_str_slice() {
        let info = AllocationInfo::with_label(
            1,
            ResourceType::Buffer,
            MemoryType::DeviceLocal,
            1024,
            "Static Label",
        );
        assert_eq!(info.label, Some("Static Label".to_string()));
    }

    #[test]
    fn test_multiple_allocations_same_size() {
        let info1 = AllocationInfo::new(1, ResourceType::Buffer, MemoryType::DeviceLocal, 1024);
        let info2 = AllocationInfo::new(2, ResourceType::Buffer, MemoryType::DeviceLocal, 1024);
        assert_eq!(info1.size_bytes, info2.size_bytes);
        assert_ne!(info1.id, info2.id);
    }
}

// =============================================================================
// AllocationStats Tests (35 tests)
// =============================================================================

mod allocation_stats_tests {
    use super::*;

    #[test]
    fn test_initial_zero_state() {
        let stats = AllocationStats::new();
        assert_eq!(stats.total_allocations, 0);
        assert_eq!(stats.total_deallocations, 0);
        assert_eq!(stats.current_allocations, 0);
        assert_eq!(stats.peak_allocations, 0);
        assert_eq!(stats.total_bytes_allocated, 0);
        assert_eq!(stats.current_bytes, 0);
        assert_eq!(stats.peak_bytes, 0);
        assert!(stats.bytes_by_type.is_empty());
    }

    #[test]
    fn test_record_allocation_increments_total() {
        let mut stats = AllocationStats::new();
        let info = AllocationInfo::new(1, ResourceType::Buffer, MemoryType::DeviceLocal, 1024);
        stats.record_allocation(&info);
        assert_eq!(stats.total_allocations, 1);
    }

    #[test]
    fn test_record_allocation_increments_current() {
        let mut stats = AllocationStats::new();
        let info = AllocationInfo::new(1, ResourceType::Buffer, MemoryType::DeviceLocal, 1024);
        stats.record_allocation(&info);
        assert_eq!(stats.current_allocations, 1);
    }

    #[test]
    fn test_record_allocation_updates_peak() {
        let mut stats = AllocationStats::new();
        let info = AllocationInfo::new(1, ResourceType::Buffer, MemoryType::DeviceLocal, 1024);
        stats.record_allocation(&info);
        assert_eq!(stats.peak_allocations, 1);
    }

    #[test]
    fn test_record_allocation_accumulates_total_bytes() {
        let mut stats = AllocationStats::new();
        let info1 = AllocationInfo::new(1, ResourceType::Buffer, MemoryType::DeviceLocal, 1024);
        let info2 = AllocationInfo::new(2, ResourceType::Buffer, MemoryType::DeviceLocal, 2048);
        stats.record_allocation(&info1);
        stats.record_allocation(&info2);
        assert_eq!(stats.total_bytes_allocated, 3072);
    }

    #[test]
    fn test_record_allocation_updates_current_bytes() {
        let mut stats = AllocationStats::new();
        let info = AllocationInfo::new(1, ResourceType::Buffer, MemoryType::DeviceLocal, 1024);
        stats.record_allocation(&info);
        assert_eq!(stats.current_bytes, 1024);
    }

    #[test]
    fn test_record_allocation_updates_peak_bytes() {
        let mut stats = AllocationStats::new();
        let info = AllocationInfo::new(1, ResourceType::Buffer, MemoryType::DeviceLocal, 1024);
        stats.record_allocation(&info);
        assert_eq!(stats.peak_bytes, 1024);
    }

    #[test]
    fn test_record_allocation_tracks_bytes_by_type() {
        let mut stats = AllocationStats::new();
        let info = AllocationInfo::new(1, ResourceType::Buffer, MemoryType::DeviceLocal, 1024);
        stats.record_allocation(&info);
        assert_eq!(stats.bytes_by_type.get(&ResourceType::Buffer), Some(&1024));
    }

    #[test]
    fn test_record_deallocation_decrements_current() {
        let mut stats = AllocationStats::new();
        let info = AllocationInfo::new(1, ResourceType::Buffer, MemoryType::DeviceLocal, 1024);
        stats.record_allocation(&info);
        stats.record_deallocation(&info);
        assert_eq!(stats.current_allocations, 0);
    }

    #[test]
    fn test_record_deallocation_decreases_current_bytes() {
        let mut stats = AllocationStats::new();
        let info = AllocationInfo::new(1, ResourceType::Buffer, MemoryType::DeviceLocal, 1024);
        stats.record_allocation(&info);
        stats.record_deallocation(&info);
        assert_eq!(stats.current_bytes, 0);
    }

    #[test]
    fn test_record_deallocation_preserves_peak_allocations() {
        let mut stats = AllocationStats::new();
        let info = AllocationInfo::new(1, ResourceType::Buffer, MemoryType::DeviceLocal, 1024);
        stats.record_allocation(&info);
        stats.record_deallocation(&info);
        assert_eq!(stats.peak_allocations, 1);
    }

    #[test]
    fn test_record_deallocation_preserves_peak_bytes() {
        let mut stats = AllocationStats::new();
        let info = AllocationInfo::new(1, ResourceType::Buffer, MemoryType::DeviceLocal, 1024);
        stats.record_allocation(&info);
        stats.record_deallocation(&info);
        assert_eq!(stats.peak_bytes, 1024);
    }

    #[test]
    fn test_reset_peak_sets_to_current() {
        let mut stats = AllocationStats::new();
        let info1 = AllocationInfo::new(1, ResourceType::Buffer, MemoryType::DeviceLocal, 1024);
        let info2 = AllocationInfo::new(2, ResourceType::Buffer, MemoryType::DeviceLocal, 2048);
        stats.record_allocation(&info1);
        stats.record_allocation(&info2);
        stats.record_deallocation(&info1);
        stats.reset_peak();
        assert_eq!(stats.peak_allocations, 1);
        assert_eq!(stats.peak_bytes, 2048);
    }

    #[test]
    fn test_multiple_allocations_same_type() {
        let mut stats = AllocationStats::new();
        for i in 0..5 {
            let info = AllocationInfo::new(i, ResourceType::Buffer, MemoryType::DeviceLocal, 1024);
            stats.record_allocation(&info);
        }
        assert_eq!(stats.bytes_by_type.get(&ResourceType::Buffer), Some(&5120));
    }

    #[test]
    fn test_multiple_types_in_bytes_by_type() {
        let mut stats = AllocationStats::new();
        let buf = AllocationInfo::new(1, ResourceType::Buffer, MemoryType::DeviceLocal, 1024);
        let tex = AllocationInfo::new(2, ResourceType::Texture, MemoryType::DeviceLocal, 4096);
        stats.record_allocation(&buf);
        stats.record_allocation(&tex);
        assert_eq!(stats.bytes_by_type.get(&ResourceType::Buffer), Some(&1024));
        assert_eq!(stats.bytes_by_type.get(&ResourceType::Texture), Some(&4096));
    }

    #[test]
    fn test_deallocation_does_not_affect_total_allocations() {
        let mut stats = AllocationStats::new();
        let info = AllocationInfo::new(1, ResourceType::Buffer, MemoryType::DeviceLocal, 1024);
        stats.record_allocation(&info);
        stats.record_deallocation(&info);
        assert_eq!(stats.total_allocations, 1);
    }

    #[test]
    fn test_deallocation_does_not_affect_total_bytes() {
        let mut stats = AllocationStats::new();
        let info = AllocationInfo::new(1, ResourceType::Buffer, MemoryType::DeviceLocal, 1024);
        stats.record_allocation(&info);
        stats.record_deallocation(&info);
        assert_eq!(stats.total_bytes_allocated, 1024);
    }

    #[test]
    fn test_deallocation_increments_total_deallocations() {
        let mut stats = AllocationStats::new();
        let info = AllocationInfo::new(1, ResourceType::Buffer, MemoryType::DeviceLocal, 1024);
        stats.record_allocation(&info);
        stats.record_deallocation(&info);
        assert_eq!(stats.total_deallocations, 1);
    }

    #[test]
    fn test_net_allocations_positive() {
        let mut stats = AllocationStats::new();
        let info1 = AllocationInfo::new(1, ResourceType::Buffer, MemoryType::DeviceLocal, 1024);
        let info2 = AllocationInfo::new(2, ResourceType::Buffer, MemoryType::DeviceLocal, 1024);
        stats.record_allocation(&info1);
        stats.record_allocation(&info2);
        stats.record_deallocation(&info1);
        assert_eq!(stats.net_allocations(), 1);
    }

    #[test]
    fn test_net_allocations_zero() {
        let mut stats = AllocationStats::new();
        let info = AllocationInfo::new(1, ResourceType::Buffer, MemoryType::DeviceLocal, 1024);
        stats.record_allocation(&info);
        stats.record_deallocation(&info);
        assert_eq!(stats.net_allocations(), 0);
    }

    #[test]
    fn test_current_bytes_formatted() {
        let mut stats = AllocationStats::new();
        let info = AllocationInfo::new(1, ResourceType::Buffer, MemoryType::DeviceLocal, 1024);
        stats.record_allocation(&info);
        assert_eq!(stats.current_bytes_formatted(), "1.00 KB");
    }

    #[test]
    fn test_peak_bytes_formatted() {
        let mut stats = AllocationStats::new();
        let info = AllocationInfo::new(1, ResourceType::Buffer, MemoryType::DeviceLocal, 1024 * 1024);
        stats.record_allocation(&info);
        assert_eq!(stats.peak_bytes_formatted(), "1.00 MB");
    }

    #[test]
    fn test_bytes_by_type_updates_on_deallocation() {
        let mut stats = AllocationStats::new();
        let info = AllocationInfo::new(1, ResourceType::Buffer, MemoryType::DeviceLocal, 1024);
        stats.record_allocation(&info);
        stats.record_deallocation(&info);
        assert_eq!(stats.bytes_by_type.get(&ResourceType::Buffer), Some(&0));
    }

    #[test]
    fn test_saturating_deallocation_current_allocations() {
        let mut stats = AllocationStats::new();
        let info = AllocationInfo::new(1, ResourceType::Buffer, MemoryType::DeviceLocal, 1024);
        // Deallocate without prior allocation should not underflow
        stats.record_deallocation(&info);
        assert_eq!(stats.current_allocations, 0);
    }

    #[test]
    fn test_saturating_deallocation_current_bytes() {
        let mut stats = AllocationStats::new();
        let info = AllocationInfo::new(1, ResourceType::Buffer, MemoryType::DeviceLocal, 1024);
        stats.record_deallocation(&info);
        assert_eq!(stats.current_bytes, 0);
    }

    #[test]
    fn test_default_trait() {
        let stats = AllocationStats::default();
        assert_eq!(stats.total_allocations, 0);
    }

    #[test]
    fn test_clone() {
        let mut stats = AllocationStats::new();
        let info = AllocationInfo::new(1, ResourceType::Buffer, MemoryType::DeviceLocal, 1024);
        stats.record_allocation(&info);
        let cloned = stats.clone();
        assert_eq!(cloned.total_allocations, 1);
        assert_eq!(cloned.current_bytes, 1024);
    }

    #[test]
    fn test_peak_only_increases() {
        let mut stats = AllocationStats::new();
        for i in 0..10 {
            let info = AllocationInfo::new(i, ResourceType::Buffer, MemoryType::DeviceLocal, 100);
            stats.record_allocation(&info);
        }
        assert_eq!(stats.peak_allocations, 10);
        assert_eq!(stats.peak_bytes, 1000);

        // Deallocate some
        for i in 0..5 {
            let info = AllocationInfo::new(i, ResourceType::Buffer, MemoryType::DeviceLocal, 100);
            stats.record_deallocation(&info);
        }
        // Peak should remain at 10
        assert_eq!(stats.peak_allocations, 10);
        assert_eq!(stats.peak_bytes, 1000);
    }

    #[test]
    fn test_multiple_types_deallocation() {
        let mut stats = AllocationStats::new();
        let buf = AllocationInfo::new(1, ResourceType::Buffer, MemoryType::DeviceLocal, 1024);
        let tex = AllocationInfo::new(2, ResourceType::Texture, MemoryType::DeviceLocal, 4096);
        stats.record_allocation(&buf);
        stats.record_allocation(&tex);
        stats.record_deallocation(&buf);
        assert_eq!(stats.bytes_by_type.get(&ResourceType::Buffer), Some(&0));
        assert_eq!(stats.bytes_by_type.get(&ResourceType::Texture), Some(&4096));
    }

    #[test]
    fn test_debug_format() {
        let stats = AllocationStats::new();
        let debug_str = format!("{:?}", stats);
        assert!(debug_str.contains("AllocationStats"));
    }

    #[test]
    fn test_zero_size_allocation() {
        let mut stats = AllocationStats::new();
        let info = AllocationInfo::new(1, ResourceType::Buffer, MemoryType::DeviceLocal, 0);
        stats.record_allocation(&info);
        assert_eq!(stats.total_allocations, 1);
        assert_eq!(stats.current_bytes, 0);
    }

    #[test]
    fn test_large_allocation() {
        let mut stats = AllocationStats::new();
        let large = 8 * 1024 * 1024 * 1024u64; // 8 GB
        let info = AllocationInfo::new(1, ResourceType::Texture, MemoryType::DeviceLocal, large);
        stats.record_allocation(&info);
        assert_eq!(stats.current_bytes, large);
        assert_eq!(stats.peak_bytes, large);
    }

    #[test]
    fn test_reset_peak_after_decrease() {
        let mut stats = AllocationStats::new();
        let info1 = AllocationInfo::new(1, ResourceType::Buffer, MemoryType::DeviceLocal, 1000);
        let info2 = AllocationInfo::new(2, ResourceType::Buffer, MemoryType::DeviceLocal, 2000);
        stats.record_allocation(&info1);
        stats.record_allocation(&info2);
        stats.record_deallocation(&info2);
        stats.reset_peak();
        assert_eq!(stats.peak_bytes, 1000);
        assert_eq!(stats.peak_allocations, 1);
    }
}

// =============================================================================
// MemoryBudget Tests (30 tests)
// =============================================================================

mod memory_budget_tests {
    use super::*;

    #[test]
    fn test_default_device_local_budget() {
        let budget = MemoryBudget::default();
        assert_eq!(budget.device_local_budget, 2 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_default_host_visible_budget() {
        let budget = MemoryBudget::default();
        assert_eq!(budget.host_visible_budget, 512 * 1024 * 1024);
    }

    #[test]
    fn test_default_total_budget() {
        let budget = MemoryBudget::default();
        assert_eq!(
            budget.total_budget,
            2 * 1024 * 1024 * 1024 + 512 * 1024 * 1024
        );
    }

    #[test]
    fn test_default_device_local_used_zero() {
        let budget = MemoryBudget::default();
        assert_eq!(budget.device_local_used, 0);
    }

    #[test]
    fn test_default_host_visible_used_zero() {
        let budget = MemoryBudget::default();
        assert_eq!(budget.host_visible_used, 0);
    }

    #[test]
    fn test_utilization_at_zero() {
        let budget = MemoryBudget::default();
        assert_eq!(budget.utilization(), 0.0);
    }

    #[test]
    fn test_utilization_at_fifty_percent() {
        let mut budget = MemoryBudget::default();
        budget.device_local_used = budget.device_local_budget / 2;
        budget.host_visible_used = budget.host_visible_budget / 2;
        let util = budget.utilization();
        assert!((util - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_utilization_at_hundred_percent() {
        let mut budget = MemoryBudget::default();
        budget.device_local_used = budget.device_local_budget;
        budget.host_visible_used = budget.host_visible_budget;
        let util = budget.utilization();
        assert!((util - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_utilization_over_hundred_percent() {
        let mut budget = MemoryBudget::default();
        budget.device_local_used = budget.device_local_budget * 2;
        let util = budget.utilization();
        assert!(util > 1.0);
    }

    #[test]
    fn test_utilization_zero_budget() {
        let budget = MemoryBudget {
            device_local_budget: 0,
            host_visible_budget: 0,
            total_budget: 0,
            device_local_used: 0,
            host_visible_used: 0,
        };
        assert_eq!(budget.utilization(), 0.0);
    }

    #[test]
    fn test_remaining_with_no_usage() {
        let budget = MemoryBudget::default();
        assert_eq!(budget.remaining(), budget.total_budget);
    }

    #[test]
    fn test_remaining_with_partial_usage() {
        let mut budget = MemoryBudget::default();
        budget.device_local_used = 1000;
        assert_eq!(budget.remaining(), budget.total_budget - 1000);
    }

    #[test]
    fn test_remaining_at_capacity() {
        let mut budget = MemoryBudget::default();
        budget.device_local_used = budget.device_local_budget;
        budget.host_visible_used = budget.host_visible_budget;
        assert_eq!(budget.remaining(), 0);
    }

    #[test]
    fn test_remaining_over_capacity() {
        let mut budget = MemoryBudget::default();
        budget.device_local_used = budget.total_budget + 1000;
        assert_eq!(budget.remaining(), 0); // saturating_sub
    }

    #[test]
    fn test_is_over_budget_under() {
        let budget = MemoryBudget::default();
        assert!(!budget.is_over_budget());
    }

    #[test]
    fn test_is_over_budget_exactly_at() {
        let mut budget = MemoryBudget::default();
        budget.device_local_used = budget.device_local_budget;
        budget.host_visible_used = budget.host_visible_budget;
        assert!(!budget.is_over_budget());
    }

    #[test]
    fn test_is_over_budget_device_local_over() {
        let mut budget = MemoryBudget::default();
        budget.device_local_used = budget.device_local_budget + 1;
        assert!(budget.is_over_budget());
    }

    #[test]
    fn test_is_over_budget_host_visible_over() {
        let mut budget = MemoryBudget::default();
        budget.host_visible_used = budget.host_visible_budget + 1;
        assert!(budget.is_over_budget());
    }

    #[test]
    fn test_record_allocation_device_local() {
        let mut budget = MemoryBudget::default();
        budget.record_allocation(MemoryType::DeviceLocal, 1000);
        assert_eq!(budget.device_local_used, 1000);
        assert_eq!(budget.host_visible_used, 0);
    }

    #[test]
    fn test_record_allocation_host_visible() {
        let mut budget = MemoryBudget::default();
        budget.record_allocation(MemoryType::HostVisible, 500);
        assert_eq!(budget.device_local_used, 0);
        assert_eq!(budget.host_visible_used, 500);
    }

    #[test]
    fn test_record_allocation_host_coherent() {
        let mut budget = MemoryBudget::default();
        budget.record_allocation(MemoryType::HostCoherent, 500);
        assert_eq!(budget.host_visible_used, 500);
    }

    #[test]
    fn test_record_allocation_host_cached() {
        let mut budget = MemoryBudget::default();
        budget.record_allocation(MemoryType::HostCached, 500);
        assert_eq!(budget.host_visible_used, 500);
    }

    #[test]
    fn test_record_deallocation_device_local() {
        let mut budget = MemoryBudget::default();
        budget.device_local_used = 1000;
        budget.record_deallocation(MemoryType::DeviceLocal, 400);
        assert_eq!(budget.device_local_used, 600);
    }

    #[test]
    fn test_record_deallocation_host_visible() {
        let mut budget = MemoryBudget::default();
        budget.host_visible_used = 1000;
        budget.record_deallocation(MemoryType::HostVisible, 400);
        assert_eq!(budget.host_visible_used, 600);
    }

    #[test]
    fn test_record_deallocation_saturating() {
        let mut budget = MemoryBudget::default();
        budget.record_deallocation(MemoryType::DeviceLocal, 1000);
        assert_eq!(budget.device_local_used, 0);
    }

    #[test]
    fn test_device_local_utilization() {
        let mut budget = MemoryBudget::default();
        budget.device_local_used = budget.device_local_budget / 4;
        let util = budget.device_local_utilization();
        assert!((util - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_device_local_utilization_zero_budget() {
        let budget = MemoryBudget {
            device_local_budget: 0,
            host_visible_budget: 100,
            total_budget: 100,
            device_local_used: 0,
            host_visible_used: 0,
        };
        assert_eq!(budget.device_local_utilization(), 0.0);
    }

    #[test]
    fn test_host_visible_utilization() {
        let mut budget = MemoryBudget::default();
        budget.host_visible_used = budget.host_visible_budget / 2;
        let util = budget.host_visible_utilization();
        assert!((util - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_host_visible_utilization_zero_budget() {
        let budget = MemoryBudget {
            device_local_budget: 100,
            host_visible_budget: 0,
            total_budget: 100,
            device_local_used: 0,
            host_visible_used: 0,
        };
        assert_eq!(budget.host_visible_utilization(), 0.0);
    }

    #[test]
    fn test_clone() {
        let mut budget = MemoryBudget::default();
        budget.device_local_used = 1000;
        let cloned = budget.clone();
        assert_eq!(cloned.device_local_used, 1000);
    }

    #[test]
    fn test_debug_format() {
        let budget = MemoryBudget::default();
        let debug_str = format!("{:?}", budget);
        assert!(debug_str.contains("MemoryBudget"));
    }
}

// =============================================================================
// MemoryTracker Tests (40 tests)
// =============================================================================

mod memory_tracker_tests {
    use super::*;

    #[test]
    fn test_with_default_budget_initialization() {
        let tracker = MemoryTracker::with_default_budget();
        assert_eq!(tracker.stats().current_allocations, 0);
        assert_eq!(tracker.stats().current_bytes, 0);
    }

    #[test]
    fn test_track_resource_returns_unique_id() {
        let mut tracker = MemoryTracker::with_default_budget();
        let id1 = tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);
        let id2 = tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_track_resource_with_label() {
        let mut tracker = MemoryTracker::with_default_budget();
        let id = tracker.track_resource(
            ResourceType::Buffer,
            MemoryType::DeviceLocal,
            1024,
            Some("Test"),
        );
        let info = tracker.get_allocation(id).unwrap();
        assert_eq!(info.label, Some("Test".to_string()));
    }

    #[test]
    fn test_track_buffer_with_usage_vertex() {
        let mut tracker = MemoryTracker::with_default_budget();
        let id =
            tracker.track_buffer_with_usage(1024, wgpu::BufferUsages::VERTEX, Some("Vertex Buffer"));
        let info = tracker.get_allocation(id).unwrap();
        assert_eq!(info.memory_type, MemoryType::DeviceLocal);
        assert_eq!(info.resource_type, ResourceType::Buffer);
    }

    #[test]
    fn test_track_buffer_with_usage_map_read() {
        let mut tracker = MemoryTracker::with_default_budget();
        let id = tracker.track_buffer_with_usage(2048, wgpu::BufferUsages::MAP_READ, Some("Readback"));
        let info = tracker.get_allocation(id).unwrap();
        assert_eq!(info.memory_type, MemoryType::HostCoherent);
    }

    #[test]
    fn test_track_texture_with_usage_render_attachment() {
        let mut tracker = MemoryTracker::with_default_budget();
        let id = tracker.track_texture_with_usage(
            4096,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
            Some("GBuffer"),
        );
        let info = tracker.get_allocation(id).unwrap();
        assert_eq!(info.memory_type, MemoryType::DeviceLocal);
        assert_eq!(info.resource_type, ResourceType::Texture);
    }

    #[test]
    fn test_track_texture_with_usage_copy_src() {
        let mut tracker = MemoryTracker::with_default_budget();
        let id = tracker.track_texture_with_usage(
            4096,
            wgpu::TextureUsages::COPY_SRC,
            Some("Screenshot"),
        );
        let info = tracker.get_allocation(id).unwrap();
        assert_eq!(info.memory_type, MemoryType::HostVisible);
    }

    #[test]
    fn test_track_query_set() {
        let mut tracker = MemoryTracker::with_default_budget();
        let id = tracker.track_query_set(256, Some("Timestamp Queries"));
        let info = tracker.get_allocation(id).unwrap();
        assert_eq!(info.resource_type, ResourceType::QuerySet);
        assert_eq!(info.memory_type, MemoryType::DeviceLocal);
    }

    #[test]
    fn test_track_bind_group() {
        let mut tracker = MemoryTracker::with_default_budget();
        let id = tracker.track_bind_group(Some("Material Bindings"));
        let info = tracker.get_allocation(id).unwrap();
        assert_eq!(info.resource_type, ResourceType::BindGroup);
        assert_eq!(info.size_bytes, 256); // Default size
    }

    #[test]
    fn test_track_pipeline() {
        let mut tracker = MemoryTracker::with_default_budget();
        let id = tracker.track_pipeline(Some("PBR Pipeline"));
        let info = tracker.get_allocation(id).unwrap();
        assert_eq!(info.resource_type, ResourceType::Pipeline);
        assert_eq!(info.size_bytes, 4096); // Default size
    }

    #[test]
    fn test_untrack_removes_allocation() {
        let mut tracker = MemoryTracker::with_default_budget();
        let id = tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);
        assert!(tracker.untrack(id));
        assert!(tracker.get_allocation(id).is_none());
    }

    #[test]
    fn test_untrack_returns_false_for_unknown() {
        let mut tracker = MemoryTracker::with_default_budget();
        assert!(!tracker.untrack(999));
    }

    #[test]
    fn test_get_allocation_returns_info() {
        let mut tracker = MemoryTracker::with_default_budget();
        let id = tracker.track_resource(
            ResourceType::Buffer,
            MemoryType::DeviceLocal,
            1024,
            Some("Test"),
        );
        let info = tracker.get_allocation(id);
        assert!(info.is_some());
        assert_eq!(info.unwrap().size_bytes, 1024);
    }

    #[test]
    fn test_get_allocation_returns_none_for_unknown() {
        let tracker = MemoryTracker::with_default_budget();
        assert!(tracker.get_allocation(999).is_none());
    }

    #[test]
    fn test_stats_reflects_tracked_allocations() {
        let mut tracker = MemoryTracker::with_default_budget();
        tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);
        tracker.track_resource(ResourceType::Texture, MemoryType::DeviceLocal, 4096, None);
        assert_eq!(tracker.stats().current_allocations, 2);
        assert_eq!(tracker.stats().current_bytes, 5120);
    }

    #[test]
    fn test_budget_reflects_current_usage() {
        let mut tracker = MemoryTracker::with_default_budget();
        tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1000, None);
        assert_eq!(tracker.budget().device_local_used, 1000);
    }

    #[test]
    fn test_summary_contains_key_info() {
        let mut tracker = MemoryTracker::with_default_budget();
        tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);
        let summary = tracker.summary();
        assert!(summary.contains("GPU Memory Summary"));
        assert!(summary.contains("Current:"));
        assert!(summary.contains("Peak:"));
        assert!(summary.contains("Budget:"));
    }

    #[test]
    fn test_summary_shows_resource_types() {
        let mut tracker = MemoryTracker::with_default_budget();
        tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);
        tracker.track_resource(ResourceType::Texture, MemoryType::DeviceLocal, 4096, None);
        let summary = tracker.summary();
        assert!(summary.contains("Buffer"));
        assert!(summary.contains("Texture"));
    }

    #[test]
    fn test_snapshot_captures_allocations() {
        let mut tracker = MemoryTracker::with_default_budget();
        tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);
        let snapshot = tracker.snapshot();
        assert_eq!(snapshot.allocations.len(), 1);
    }

    #[test]
    fn test_snapshot_captures_stats() {
        let mut tracker = MemoryTracker::with_default_budget();
        tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);
        let snapshot = tracker.snapshot();
        assert_eq!(snapshot.stats.current_bytes, 1024);
    }

    #[test]
    fn test_snapshot_captures_budget() {
        let mut tracker = MemoryTracker::with_default_budget();
        tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1000, None);
        let snapshot = tracker.snapshot();
        assert_eq!(snapshot.budget.device_local_used, 1000);
    }

    #[test]
    fn test_clear_resets_tracker() {
        let mut tracker = MemoryTracker::with_default_budget();
        tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);
        tracker.clear();
        assert_eq!(tracker.stats().current_allocations, 0);
        assert_eq!(tracker.stats().current_bytes, 0);
        assert!(tracker.allocations().is_empty());
    }

    #[test]
    fn test_clear_resets_budget_usage() {
        let mut tracker = MemoryTracker::with_default_budget();
        tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1000, None);
        tracker.clear();
        assert_eq!(tracker.budget().device_local_used, 0);
        assert_eq!(tracker.budget().host_visible_used, 0);
    }

    #[test]
    fn test_multiple_allocations() {
        let mut tracker = MemoryTracker::with_default_budget();
        for i in 0..10 {
            tracker.track_resource(
                ResourceType::Buffer,
                MemoryType::DeviceLocal,
                1024,
                Some(&format!("Buffer {}", i)),
            );
        }
        assert_eq!(tracker.stats().current_allocations, 10);
        assert_eq!(tracker.stats().current_bytes, 10240);
    }

    #[test]
    fn test_id_sequence_incrementing() {
        let mut tracker = MemoryTracker::with_default_budget();
        let id1 = tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);
        let id2 = tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);
        let id3 = tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);
        assert!(id2 > id1);
        assert!(id3 > id2);
    }

    #[test]
    fn test_allocations_returns_hashmap() {
        let mut tracker = MemoryTracker::with_default_budget();
        let id = tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);
        let allocations = tracker.allocations();
        assert!(allocations.contains_key(&id));
    }

    #[test]
    fn test_stats_mut_allows_peak_reset() {
        let mut tracker = MemoryTracker::with_default_budget();
        tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);
        tracker.stats_mut().reset_peak();
        assert_eq!(tracker.stats().peak_bytes, 1024);
    }

    #[test]
    fn test_debug_format() {
        let tracker = MemoryTracker::with_default_budget();
        let debug_str = format!("{:?}", tracker);
        assert!(debug_str.contains("MemoryTracker"));
    }

    #[test]
    fn test_untrack_updates_stats() {
        let mut tracker = MemoryTracker::with_default_budget();
        let id = tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);
        tracker.untrack(id);
        assert_eq!(tracker.stats().current_allocations, 0);
        assert_eq!(tracker.stats().current_bytes, 0);
    }

    #[test]
    fn test_untrack_updates_budget() {
        let mut tracker = MemoryTracker::with_default_budget();
        let id = tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1000, None);
        tracker.untrack(id);
        assert_eq!(tracker.budget().device_local_used, 0);
    }

    #[test]
    fn test_mixed_memory_types() {
        let mut tracker = MemoryTracker::with_default_budget();
        tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1000, None);
        tracker.track_resource(ResourceType::Buffer, MemoryType::HostVisible, 500, None);
        assert_eq!(tracker.budget().device_local_used, 1000);
        assert_eq!(tracker.budget().host_visible_used, 500);
    }

    #[test]
    fn test_zero_size_allocation() {
        let mut tracker = MemoryTracker::with_default_budget();
        let id = tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 0, None);
        let info = tracker.get_allocation(id).unwrap();
        assert_eq!(info.size_bytes, 0);
    }

    #[test]
    fn test_large_allocation() {
        let mut tracker = MemoryTracker::with_default_budget();
        let large = 4 * 1024 * 1024 * 1024u64;
        let id = tracker.track_resource(ResourceType::Texture, MemoryType::DeviceLocal, large, None);
        let info = tracker.get_allocation(id).unwrap();
        assert_eq!(info.size_bytes, large);
    }

    #[test]
    fn test_track_without_label() {
        let mut tracker = MemoryTracker::with_default_budget();
        let id = tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);
        let info = tracker.get_allocation(id).unwrap();
        assert!(info.label.is_none());
    }

    #[test]
    fn test_track_with_empty_label() {
        let mut tracker = MemoryTracker::with_default_budget();
        let id = tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, Some(""));
        let info = tracker.get_allocation(id).unwrap();
        assert_eq!(info.label, Some("".to_string()));
    }

    #[test]
    fn test_peak_tracking_through_tracker() {
        let mut tracker = MemoryTracker::with_default_budget();
        let id1 = tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1000, None);
        let id2 = tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 2000, None);
        assert_eq!(tracker.stats().peak_bytes, 3000);
        tracker.untrack(id1);
        tracker.untrack(id2);
        assert_eq!(tracker.stats().peak_bytes, 3000);
        assert_eq!(tracker.stats().current_bytes, 0);
    }

    #[test]
    fn test_multiple_resource_types_tracking() {
        let mut tracker = MemoryTracker::with_default_budget();
        tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);
        tracker.track_resource(ResourceType::Texture, MemoryType::DeviceLocal, 4096, None);
        tracker.track_query_set(256, None);
        tracker.track_bind_group(None);
        tracker.track_pipeline(None);
        assert_eq!(tracker.stats().current_allocations, 5);
    }
}

// =============================================================================
// MemorySnapshot & MemoryDiff Tests (25 tests)
// =============================================================================

mod snapshot_diff_tests {
    use super::*;

    #[test]
    fn test_snapshot_allocations_captured() {
        let mut tracker = MemoryTracker::with_default_budget();
        tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);
        let snapshot = tracker.snapshot();
        assert_eq!(snapshot.allocations.len(), 1);
    }

    #[test]
    fn test_snapshot_stats_captured() {
        let mut tracker = MemoryTracker::with_default_budget();
        tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);
        let snapshot = tracker.snapshot();
        assert_eq!(snapshot.stats.current_bytes, 1024);
    }

    #[test]
    fn test_snapshot_budget_captured() {
        let mut tracker = MemoryTracker::with_default_budget();
        tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1000, None);
        let snapshot = tracker.snapshot();
        assert_eq!(snapshot.budget.device_local_used, 1000);
    }

    #[test]
    fn test_snapshot_timestamp_recent() {
        let tracker = MemoryTracker::with_default_budget();
        let snapshot = tracker.snapshot();
        assert!(snapshot.age_secs() < 1.0);
    }

    #[test]
    fn test_diff_finds_added_allocations() {
        let mut tracker = MemoryTracker::with_default_budget();
        let snapshot1 = tracker.snapshot();
        tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);
        let snapshot2 = tracker.snapshot();
        let diff = snapshot1.diff(&snapshot2);
        assert_eq!(diff.added.len(), 1);
    }

    #[test]
    fn test_diff_finds_removed_allocations() {
        let mut tracker = MemoryTracker::with_default_budget();
        let id = tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);
        let snapshot1 = tracker.snapshot();
        tracker.untrack(id);
        let snapshot2 = tracker.snapshot();
        let diff = snapshot1.diff(&snapshot2);
        assert_eq!(diff.removed.len(), 1);
        assert!(diff.removed.contains(&id));
    }

    #[test]
    fn test_diff_bytes_delta_positive() {
        let mut tracker = MemoryTracker::with_default_budget();
        let snapshot1 = tracker.snapshot();
        tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);
        let snapshot2 = tracker.snapshot();
        let diff = snapshot1.diff(&snapshot2);
        assert_eq!(diff.bytes_delta, 1024);
    }

    #[test]
    fn test_diff_bytes_delta_negative() {
        let mut tracker = MemoryTracker::with_default_budget();
        let id = tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);
        let snapshot1 = tracker.snapshot();
        tracker.untrack(id);
        let snapshot2 = tracker.snapshot();
        let diff = snapshot1.diff(&snapshot2);
        assert_eq!(diff.bytes_delta, -1024);
    }

    #[test]
    fn test_diff_is_empty_when_no_changes() {
        let tracker = MemoryTracker::with_default_budget();
        let snapshot1 = tracker.snapshot();
        let snapshot2 = tracker.snapshot();
        let diff = snapshot1.diff(&snapshot2);
        assert!(diff.is_empty());
    }

    #[test]
    fn test_diff_is_not_empty_with_additions() {
        let mut tracker = MemoryTracker::with_default_budget();
        let snapshot1 = tracker.snapshot();
        tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);
        let snapshot2 = tracker.snapshot();
        let diff = snapshot1.diff(&snapshot2);
        assert!(!diff.is_empty());
    }

    #[test]
    fn test_diff_between_empty_snapshots() {
        let tracker = MemoryTracker::with_default_budget();
        let snapshot1 = tracker.snapshot();
        let snapshot2 = tracker.snapshot();
        let diff = snapshot1.diff(&snapshot2);
        assert!(diff.is_empty());
        assert_eq!(diff.bytes_delta, 0);
    }

    #[test]
    fn test_diff_with_only_additions() {
        let mut tracker = MemoryTracker::with_default_budget();
        let snapshot1 = tracker.snapshot();
        tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);
        tracker.track_resource(ResourceType::Texture, MemoryType::DeviceLocal, 2048, None);
        let snapshot2 = tracker.snapshot();
        let diff = snapshot1.diff(&snapshot2);
        assert_eq!(diff.added.len(), 2);
        assert!(diff.removed.is_empty());
    }

    #[test]
    fn test_diff_with_only_removals() {
        let mut tracker = MemoryTracker::with_default_budget();
        let id1 = tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);
        let id2 = tracker.track_resource(ResourceType::Texture, MemoryType::DeviceLocal, 2048, None);
        let snapshot1 = tracker.snapshot();
        tracker.untrack(id1);
        tracker.untrack(id2);
        let snapshot2 = tracker.snapshot();
        let diff = snapshot1.diff(&snapshot2);
        assert!(diff.added.is_empty());
        assert_eq!(diff.removed.len(), 2);
    }

    #[test]
    fn test_diff_added_count() {
        let mut tracker = MemoryTracker::with_default_budget();
        let snapshot1 = tracker.snapshot();
        tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);
        tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);
        let snapshot2 = tracker.snapshot();
        let diff = snapshot1.diff(&snapshot2);
        assert_eq!(diff.added_count(), 2);
    }

    #[test]
    fn test_diff_removed_count() {
        let mut tracker = MemoryTracker::with_default_budget();
        let id1 = tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);
        let id2 = tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);
        let snapshot1 = tracker.snapshot();
        tracker.untrack(id1);
        tracker.untrack(id2);
        let snapshot2 = tracker.snapshot();
        let diff = snapshot1.diff(&snapshot2);
        assert_eq!(diff.removed_count(), 2);
    }

    #[test]
    fn test_diff_bytes_added() {
        let mut tracker = MemoryTracker::with_default_budget();
        let snapshot1 = tracker.snapshot();
        tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);
        tracker.track_resource(ResourceType::Texture, MemoryType::DeviceLocal, 2048, None);
        let snapshot2 = tracker.snapshot();
        let diff = snapshot1.diff(&snapshot2);
        assert_eq!(diff.bytes_added(), 3072);
    }

    #[test]
    fn test_diff_display() {
        let diff = MemoryDiff {
            added: vec![AllocationInfo::new(
                1,
                ResourceType::Buffer,
                MemoryType::DeviceLocal,
                1024,
            )],
            removed: vec![2],
            bytes_delta: 512,
        };
        let s = format!("{}", diff);
        assert!(s.contains("+1 allocs"));
        assert!(s.contains("-1 allocs"));
    }

    #[test]
    fn test_diff_display_negative() {
        let diff = MemoryDiff {
            added: vec![],
            removed: vec![1],
            bytes_delta: -1024,
        };
        let s = format!("{}", diff);
        assert!(s.contains("-1 allocs"));
        assert!(s.contains("-1.00 KB"));
    }

    #[test]
    fn test_snapshot_clone() {
        let mut tracker = MemoryTracker::with_default_budget();
        tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);
        let snapshot = tracker.snapshot();
        let cloned = snapshot.clone();
        assert_eq!(cloned.allocations.len(), 1);
    }

    #[test]
    fn test_diff_clone() {
        let diff = MemoryDiff {
            added: vec![AllocationInfo::new(
                1,
                ResourceType::Buffer,
                MemoryType::DeviceLocal,
                1024,
            )],
            removed: vec![2],
            bytes_delta: 512,
        };
        let cloned = diff.clone();
        assert_eq!(cloned.added.len(), 1);
        assert_eq!(cloned.removed.len(), 1);
    }

    #[test]
    fn test_snapshot_debug() {
        let tracker = MemoryTracker::with_default_budget();
        let snapshot = tracker.snapshot();
        let debug_str = format!("{:?}", snapshot);
        assert!(debug_str.contains("MemorySnapshot"));
    }

    #[test]
    fn test_diff_debug() {
        let diff = MemoryDiff {
            added: vec![],
            removed: vec![],
            bytes_delta: 0,
        };
        let debug_str = format!("{:?}", diff);
        assert!(debug_str.contains("MemoryDiff"));
    }

    #[test]
    fn test_multiple_snapshots_independent() {
        let mut tracker = MemoryTracker::with_default_budget();
        tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);
        let snapshot1 = tracker.snapshot();
        tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 2048, None);
        let snapshot2 = tracker.snapshot();
        assert_eq!(snapshot1.allocations.len(), 1);
        assert_eq!(snapshot2.allocations.len(), 2);
    }

    #[test]
    fn test_snapshot_age_increases() {
        let tracker = MemoryTracker::with_default_budget();
        let snapshot = tracker.snapshot();
        thread::sleep(Duration::from_millis(10));
        assert!(snapshot.age_secs() >= 0.01);
    }
}

// =============================================================================
// LeakDetector Tests (20 tests)
// =============================================================================

mod leak_detector_tests {
    use super::*;

    #[test]
    fn test_new_creates_empty() {
        let detector = LeakDetector::new();
        assert_eq!(detector.expected_count(), 0);
    }

    #[test]
    fn test_mark_expected_adds_id() {
        let mut detector = LeakDetector::new();
        detector.mark_expected(1);
        assert_eq!(detector.expected_count(), 1);
    }

    #[test]
    fn test_mark_expected_multiple() {
        let mut detector = LeakDetector::new();
        detector.mark_expected(1);
        detector.mark_expected(2);
        detector.mark_expected(3);
        assert_eq!(detector.expected_count(), 3);
    }

    #[test]
    fn test_unmark_expected_removes_id() {
        let mut detector = LeakDetector::new();
        detector.mark_expected(1);
        detector.unmark_expected(1);
        assert_eq!(detector.expected_count(), 0);
    }

    #[test]
    fn test_clear_expected_removes_all() {
        let mut detector = LeakDetector::new();
        detector.mark_expected(1);
        detector.mark_expected(2);
        detector.clear_expected();
        assert_eq!(detector.expected_count(), 0);
    }

    #[test]
    fn test_check_leaks_empty_tracker() {
        let tracker = MemoryTracker::with_default_budget();
        let detector = LeakDetector::new();
        let leaks = detector.check_leaks(&tracker, 0.0);
        assert!(leaks.is_empty());
    }

    #[test]
    fn test_check_leaks_respects_threshold() {
        let mut tracker = MemoryTracker::with_default_budget();
        tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);
        let detector = LeakDetector::new();
        // High threshold should find no leaks
        let leaks = detector.check_leaks(&tracker, 1000.0);
        assert!(leaks.is_empty());
    }

    #[test]
    fn test_check_leaks_finds_old_allocations() {
        let mut tracker = MemoryTracker::with_default_budget();
        tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);
        // Sleep briefly to ensure age > 0
        thread::sleep(Duration::from_millis(10));
        let detector = LeakDetector::new();
        let leaks = detector.check_leaks(&tracker, 0.001); // 1ms threshold
        assert_eq!(leaks.len(), 1);
    }

    #[test]
    fn test_check_leaks_excludes_expected() {
        let mut tracker = MemoryTracker::with_default_budget();
        let id = tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);
        let mut detector = LeakDetector::new();
        detector.mark_expected(id);
        let leaks = detector.check_leaks(&tracker, 0.0);
        assert!(leaks.is_empty());
    }

    #[test]
    fn test_check_leaks_by_type_filters_correctly() {
        let mut tracker = MemoryTracker::with_default_budget();
        tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);
        tracker.track_resource(ResourceType::Texture, MemoryType::DeviceLocal, 4096, None);
        thread::sleep(Duration::from_millis(10));
        let detector = LeakDetector::new();
        let buffer_leaks = detector.check_leaks_by_type(&tracker, ResourceType::Buffer, 0.001);
        assert_eq!(buffer_leaks.len(), 1);
        assert_eq!(buffer_leaks[0].resource_type, ResourceType::Buffer);
    }

    #[test]
    fn test_check_leaks_by_type_no_match() {
        let mut tracker = MemoryTracker::with_default_budget();
        tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);
        thread::sleep(Duration::from_millis(10));
        let detector = LeakDetector::new();
        let texture_leaks = detector.check_leaks_by_type(&tracker, ResourceType::Texture, 0.001);
        assert!(texture_leaks.is_empty());
    }

    #[test]
    fn test_no_false_positives_young_allocations() {
        let mut tracker = MemoryTracker::with_default_budget();
        tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);
        let detector = LeakDetector::new();
        // Use very high threshold
        let leaks = detector.check_leaks(&tracker, 60.0);
        assert!(leaks.is_empty());
    }

    #[test]
    fn test_default_trait() {
        let detector = LeakDetector::default();
        assert_eq!(detector.expected_count(), 0);
    }

    #[test]
    fn test_debug_format() {
        let detector = LeakDetector::new();
        let debug_str = format!("{:?}", detector);
        assert!(debug_str.contains("LeakDetector"));
    }

    #[test]
    fn test_mark_same_id_twice() {
        let mut detector = LeakDetector::new();
        detector.mark_expected(1);
        detector.mark_expected(1);
        assert_eq!(detector.expected_count(), 1); // HashSet deduplicates
    }

    #[test]
    fn test_unmark_nonexistent() {
        let mut detector = LeakDetector::new();
        detector.unmark_expected(999); // Should not panic
        assert_eq!(detector.expected_count(), 0);
    }

    #[test]
    fn test_check_leaks_returns_allocation_info() {
        let mut tracker = MemoryTracker::with_default_budget();
        let id = tracker.track_resource(
            ResourceType::Buffer,
            MemoryType::DeviceLocal,
            1024,
            Some("Leaky Buffer"),
        );
        thread::sleep(Duration::from_millis(10));
        let detector = LeakDetector::new();
        let leaks = detector.check_leaks(&tracker, 0.001);
        assert_eq!(leaks.len(), 1);
        assert_eq!(leaks[0].id, id);
        assert_eq!(leaks[0].size_bytes, 1024);
        assert_eq!(leaks[0].label, Some("Leaky Buffer".to_string()));
    }

    #[test]
    fn test_mixed_expected_and_leaks() {
        let mut tracker = MemoryTracker::with_default_budget();
        let id1 =
            tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, Some("Expected"));
        let _id2 = tracker.track_resource(
            ResourceType::Buffer,
            MemoryType::DeviceLocal,
            2048,
            Some("Leak"),
        );
        thread::sleep(Duration::from_millis(10));
        let mut detector = LeakDetector::new();
        detector.mark_expected(id1);
        let leaks = detector.check_leaks(&tracker, 0.001);
        assert_eq!(leaks.len(), 1);
        assert_eq!(leaks[0].label, Some("Leak".to_string()));
    }

    #[test]
    fn test_check_leaks_by_type_respects_expected() {
        let mut tracker = MemoryTracker::with_default_budget();
        let id = tracker.track_resource(ResourceType::Buffer, MemoryType::DeviceLocal, 1024, None);
        thread::sleep(Duration::from_millis(10));
        let mut detector = LeakDetector::new();
        detector.mark_expected(id);
        let leaks = detector.check_leaks_by_type(&tracker, ResourceType::Buffer, 0.001);
        assert!(leaks.is_empty());
    }

    #[test]
    fn test_expected_count_after_operations() {
        let mut detector = LeakDetector::new();
        detector.mark_expected(1);
        detector.mark_expected(2);
        detector.mark_expected(3);
        assert_eq!(detector.expected_count(), 3);
        detector.unmark_expected(2);
        assert_eq!(detector.expected_count(), 2);
        detector.clear_expected();
        assert_eq!(detector.expected_count(), 0);
    }
}

// =============================================================================
// Utility Function Tests (10 tests)
// =============================================================================

mod utility_tests {
    use super::*;

    #[test]
    fn test_format_bytes_zero() {
        assert_eq!(format_bytes(0), "0 B");
    }

    #[test]
    fn test_format_bytes_bytes() {
        assert_eq!(format_bytes(512), "512 B");
    }

    #[test]
    fn test_format_bytes_kilobytes() {
        assert_eq!(format_bytes(1024), "1.00 KB");
    }

    #[test]
    fn test_format_bytes_kilobytes_fractional() {
        assert_eq!(format_bytes(1536), "1.50 KB");
    }

    #[test]
    fn test_format_bytes_megabytes() {
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
    }

    #[test]
    fn test_format_bytes_megabytes_fractional() {
        assert_eq!(format_bytes(1024 * 1024 + 512 * 1024), "1.50 MB");
    }

    #[test]
    fn test_format_bytes_gigabytes() {
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
    }

    #[test]
    fn test_format_bytes_gigabytes_large() {
        assert_eq!(format_bytes(2 * 1024 * 1024 * 1024), "2.00 GB");
    }

    #[test]
    fn test_format_bytes_gigabytes_fractional() {
        assert_eq!(
            format_bytes(1024 * 1024 * 1024 + 512 * 1024 * 1024),
            "1.50 GB"
        );
    }

    #[test]
    fn test_format_bytes_near_boundary() {
        assert_eq!(format_bytes(1023), "1023 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
    }
}
