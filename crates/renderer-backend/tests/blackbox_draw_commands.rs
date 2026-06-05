//! Blackbox tests for draw commands API (T-WGPU-P3.8.4)
//!
//! Tests the public indirect draw command types and utilities without
//! reading implementation details. Focuses on API surface, constructors,
//! field access, and trait implementations.

use bytemuck::{Pod, Zeroable};
use std::mem;

// Import the public API from gpu_driven module
use renderer_backend::gpu_driven::{
    // Structs
    IndirectDrawIndexedArgs,
    IndirectDrawArgs,
    IndirectDispatchArgs,
    IndirectTier,
    DrawBatchBuilder,
    MultiIndirectConfig,
    // Constants
    DEFAULT_MAX_DRAWS,
    INDIRECT_DISPATCH_ARGS_SIZE,
    INDIRECT_DRAW_ARGS_SIZE,
    INDIRECT_DRAW_INDEXED_ARGS_SIZE,
};

// ============================================================================
// Test Category 1: API Surface Tests
// ============================================================================

mod api_surface {
    use super::*;

    #[test]
    fn indirect_draw_indexed_args_exists() {
        // Type exists and can be used in type position
        let _: Option<IndirectDrawIndexedArgs> = None;
    }

    #[test]
    fn indirect_draw_args_exists() {
        let _: Option<IndirectDrawArgs> = None;
    }

    #[test]
    fn indirect_dispatch_args_exists() {
        let _: Option<IndirectDispatchArgs> = None;
    }

    #[test]
    fn indirect_tier_exists() {
        let _: Option<IndirectTier> = None;
    }

    #[test]
    fn draw_batch_builder_exists() {
        let _: Option<DrawBatchBuilder> = None;
    }

    #[test]
    fn multi_indirect_config_exists() {
        let _: Option<MultiIndirectConfig> = None;
    }

    #[test]
    fn constants_are_accessible() {
        // Constants exist and have expected types
        let _: u32 = DEFAULT_MAX_DRAWS;
        let _: usize = INDIRECT_DISPATCH_ARGS_SIZE;
        let _: usize = INDIRECT_DRAW_ARGS_SIZE;
        let _: usize = INDIRECT_DRAW_INDEXED_ARGS_SIZE;
    }
}

// ============================================================================
// Test Category 2: IndirectDrawIndexedArgs Construction and Field Access
// ============================================================================

mod indirect_draw_indexed_args {
    use super::*;

    #[test]
    fn new_constructor() {
        let args = IndirectDrawIndexedArgs::new(
            100,   // index_count
            10,    // instance_count
            0,     // first_index
            0,     // base_vertex
            0,     // first_instance
        );

        // Verify construction succeeded
        assert!(args.is_visible());
    }

    #[test]
    fn new_constructor_with_all_params() {
        let args = IndirectDrawIndexedArgs::new(
            36,    // index_count (12 triangles)
            5,     // instance_count
            100,   // first_index
            50,    // base_vertex
            10,    // first_instance
        );

        assert!(args.is_visible());
    }

    #[test]
    fn single_constructor() {
        let args = IndirectDrawIndexedArgs::single(
            24,    // index_count
            0,     // first_index
            0,     // base_vertex
        );

        // Single implies instance_count = 1
        assert!(args.is_visible());
    }

    #[test]
    fn single_with_offset() {
        let args = IndirectDrawIndexedArgs::single(
            36,    // index_count
            200,   // first_index (offset into index buffer)
            100,   // base_vertex
        );

        assert!(args.is_visible());
    }

    #[test]
    fn zeroed_constructor() {
        let args = IndirectDrawIndexedArgs::zeroed();

        // Zeroed args should not be visible (index_count = 0)
        assert!(!args.is_visible());
    }

    #[test]
    fn is_visible_with_zero_instances() {
        let args = IndirectDrawIndexedArgs::new(
            100,   // index_count
            0,     // instance_count = 0 means not visible
            0,
            0,
            0,
        );

        assert!(!args.is_visible());
    }

    #[test]
    fn is_visible_with_zero_indices() {
        let args = IndirectDrawIndexedArgs::new(
            0,     // index_count = 0 means not visible
            5,
            0,
            0,
            0,
        );

        assert!(!args.is_visible());
    }

    #[test]
    fn total_vertices_calculation() {
        let args = IndirectDrawIndexedArgs::new(
            100,   // index_count
            5,     // instance_count
            0,
            0,
            0,
        );

        // total_vertices = index_count * instance_count
        assert_eq!(args.total_vertices(), 500);
    }

    #[test]
    fn total_vertices_single_instance() {
        let args = IndirectDrawIndexedArgs::single(36, 0, 0);
        assert_eq!(args.total_vertices(), 36);
    }

    #[test]
    fn total_vertices_zeroed() {
        let args = IndirectDrawIndexedArgs::zeroed();
        assert_eq!(args.total_vertices(), 0);
    }

    #[test]
    fn size_constant() {
        assert_eq!(IndirectDrawIndexedArgs::SIZE, INDIRECT_DRAW_INDEXED_ARGS_SIZE);
        assert_eq!(IndirectDrawIndexedArgs::SIZE, 20);
    }
}

// ============================================================================
// Test Category 3: IndirectDrawArgs Construction and Field Access
// ============================================================================

mod indirect_draw_args {
    use super::*;

    #[test]
    fn new_constructor() {
        let args = IndirectDrawArgs::new(
            100,   // vertex_count
            10,    // instance_count
            0,     // first_vertex
            0,     // first_instance
        );

        assert!(args.is_visible());
    }

    #[test]
    fn new_with_all_params() {
        let args = IndirectDrawArgs::new(
            256,   // vertex_count
            8,     // instance_count
            100,   // first_vertex
            4,     // first_instance
        );

        assert!(args.is_visible());
    }

    #[test]
    fn single_constructor() {
        let args = IndirectDrawArgs::single(
            64,    // vertex_count
            0,     // first_vertex
        );

        // Single implies instance_count = 1
        assert!(args.is_visible());
    }

    #[test]
    fn single_with_offset() {
        let args = IndirectDrawArgs::single(
            128,   // vertex_count
            256,   // first_vertex (offset)
        );

        assert!(args.is_visible());
    }

    #[test]
    fn zeroed_constructor() {
        let args = IndirectDrawArgs::zeroed();
        assert!(!args.is_visible());
    }

    #[test]
    fn is_visible_with_zero_instances() {
        let args = IndirectDrawArgs::new(100, 0, 0, 0);
        assert!(!args.is_visible());
    }

    #[test]
    fn is_visible_with_zero_vertices() {
        let args = IndirectDrawArgs::new(0, 5, 0, 0);
        assert!(!args.is_visible());
    }

    #[test]
    fn total_vertices_calculation() {
        let args = IndirectDrawArgs::new(100, 4, 0, 0);
        assert_eq!(args.total_vertices(), 400);
    }

    #[test]
    fn total_vertices_single() {
        let args = IndirectDrawArgs::single(64, 0);
        assert_eq!(args.total_vertices(), 64);
    }

    #[test]
    fn total_vertices_zeroed() {
        let args = IndirectDrawArgs::zeroed();
        assert_eq!(args.total_vertices(), 0);
    }

    #[test]
    fn size_constant() {
        assert_eq!(IndirectDrawArgs::SIZE, INDIRECT_DRAW_ARGS_SIZE);
        assert_eq!(IndirectDrawArgs::SIZE, 16);
    }
}

// ============================================================================
// Test Category 4: IndirectDispatchArgs Construction
// ============================================================================

mod indirect_dispatch_args {
    use super::*;

    #[test]
    fn new_constructor() {
        let args = IndirectDispatchArgs::new(4, 4, 4);
        assert!(args.is_active());
    }

    #[test]
    fn linear_constructor() {
        let args = IndirectDispatchArgs::linear(256);
        assert!(args.is_active());
    }

    #[test]
    fn grid_2d_constructor() {
        let args = IndirectDispatchArgs::grid_2d(16, 16);
        assert!(args.is_active());
    }

    #[test]
    fn zeroed_constructor() {
        let args = IndirectDispatchArgs::zeroed();
        assert!(!args.is_active());
    }

    #[test]
    fn is_active_with_zero_x() {
        let args = IndirectDispatchArgs::new(0, 4, 4);
        assert!(!args.is_active());
    }

    #[test]
    fn is_active_with_zero_y() {
        let args = IndirectDispatchArgs::new(4, 0, 4);
        assert!(!args.is_active());
    }

    #[test]
    fn is_active_with_zero_z() {
        let args = IndirectDispatchArgs::new(4, 4, 0);
        assert!(!args.is_active());
    }

    #[test]
    fn total_workgroups_calculation() {
        let args = IndirectDispatchArgs::new(2, 3, 4);
        assert_eq!(args.total_workgroups(), 24);
    }

    #[test]
    fn total_workgroups_linear() {
        let args = IndirectDispatchArgs::linear(100);
        assert_eq!(args.total_workgroups(), 100);
    }

    #[test]
    fn total_workgroups_grid_2d() {
        let args = IndirectDispatchArgs::grid_2d(8, 8);
        assert_eq!(args.total_workgroups(), 64);
    }

    #[test]
    fn total_workgroups_zeroed() {
        let args = IndirectDispatchArgs::zeroed();
        assert_eq!(args.total_workgroups(), 0);
    }

    #[test]
    fn for_elements_exact_fit() {
        // 256 elements with workgroup size 64 = 4 workgroups
        let args = IndirectDispatchArgs::for_elements(256, 64);
        assert_eq!(args.total_workgroups(), 4);
    }

    #[test]
    fn for_elements_with_remainder() {
        // 257 elements with workgroup size 64 = 5 workgroups (rounds up)
        let args = IndirectDispatchArgs::for_elements(257, 64);
        assert_eq!(args.total_workgroups(), 5);
    }

    #[test]
    fn for_elements_small_count() {
        // 10 elements with workgroup size 256 = 1 workgroup
        let args = IndirectDispatchArgs::for_elements(10, 256);
        assert_eq!(args.total_workgroups(), 1);
    }

    #[test]
    fn for_elements_zero() {
        // 0 elements should still result in at least 0 workgroups
        let args = IndirectDispatchArgs::for_elements(0, 64);
        // Could be 0 or 1 depending on implementation (division rounding)
        assert!(args.total_workgroups() <= 1);
    }

    #[test]
    fn size_constant() {
        assert_eq!(IndirectDispatchArgs::SIZE, INDIRECT_DISPATCH_ARGS_SIZE);
        assert_eq!(IndirectDispatchArgs::SIZE, 12);
    }
}

// ============================================================================
// Test Category 5: IndirectTier Enum Variants
// ============================================================================

mod indirect_tier {
    use super::*;

    #[test]
    fn tier_minimal_exists() {
        // Minimal tier: CPU-side batching, no multi-draw support
        let tier = IndirectTier::Minimal;
        assert!(!tier.supports_gpu_count());
        assert!(!tier.supports_multi_draw());
    }

    #[test]
    fn tier_partial_exists() {
        // Partial tier: draw_indexed_indirect without count
        let tier = IndirectTier::Partial;
        assert!(!tier.supports_gpu_count());
        assert!(tier.supports_multi_draw());
    }

    #[test]
    fn tier_full_exists() {
        // Full tier: draw_indexed_indirect_count
        let tier = IndirectTier::Full;
        assert!(tier.supports_gpu_count());
        assert!(tier.supports_multi_draw());
    }

    #[test]
    fn tier_description_minimal() {
        let tier = IndirectTier::Minimal;
        let desc = tier.description();
        assert!(!desc.is_empty());
    }

    #[test]
    fn tier_description_partial() {
        let tier = IndirectTier::Partial;
        let desc = tier.description();
        assert!(!desc.is_empty());
    }

    #[test]
    fn tier_description_full() {
        let tier = IndirectTier::Full;
        let desc = tier.description();
        assert!(!desc.is_empty());
    }

    #[test]
    fn tier_display_impl() {
        // IndirectTier should implement Display
        let tier = IndirectTier::Minimal;
        let s = format!("{}", tier);
        assert!(!s.is_empty());
    }

    #[test]
    fn tier_debug_impl() {
        // IndirectTier should implement Debug
        let tier = IndirectTier::Partial;
        let s = format!("{:?}", tier);
        assert!(!s.is_empty());
    }

    #[test]
    fn tier_clone() {
        let tier = IndirectTier::Full;
        let cloned = tier.clone();
        assert_eq!(tier, cloned);
    }

    #[test]
    fn tier_copy() {
        let tier = IndirectTier::Minimal;
        let copied: IndirectTier = tier;
        assert_eq!(tier, copied);
    }

    #[test]
    fn tier_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(IndirectTier::Minimal);
        set.insert(IndirectTier::Partial);
        set.insert(IndirectTier::Full);
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn tier_eq() {
        assert_eq!(IndirectTier::Minimal, IndirectTier::Minimal);
        assert_ne!(IndirectTier::Minimal, IndirectTier::Partial);
        assert_ne!(IndirectTier::Partial, IndirectTier::Full);
    }
}

// ============================================================================
// Test Category 6: Buffer Size Calculations and Constants
// ============================================================================

mod buffer_calculations {
    use super::*;

    #[test]
    fn default_max_draws_is_reasonable() {
        // DEFAULT_MAX_DRAWS should be a power of 2 and reasonable for batching
        assert!(DEFAULT_MAX_DRAWS >= 1024);
        assert!(DEFAULT_MAX_DRAWS <= 1_000_000);
    }

    #[test]
    fn indexed_args_size_is_20_bytes() {
        // DrawIndexedIndirect: 5 x u32 = 20 bytes
        // index_count, instance_count, first_index, base_vertex, first_instance
        assert_eq!(INDIRECT_DRAW_INDEXED_ARGS_SIZE, 20);
        assert_eq!(mem::size_of::<IndirectDrawIndexedArgs>(), 20);
    }

    #[test]
    fn draw_args_size_is_16_bytes() {
        // DrawIndirect: 4 x u32 = 16 bytes
        // vertex_count, instance_count, first_vertex, first_instance
        assert_eq!(INDIRECT_DRAW_ARGS_SIZE, 16);
        assert_eq!(mem::size_of::<IndirectDrawArgs>(), 16);
    }

    #[test]
    fn dispatch_args_size_is_12_bytes() {
        // DispatchIndirect: 3 x u32 = 12 bytes
        // x, y, z
        assert_eq!(INDIRECT_DISPATCH_ARGS_SIZE, 12);
        assert_eq!(mem::size_of::<IndirectDispatchArgs>(), 12);
    }

    #[test]
    fn indexed_buffer_size_for_draws() {
        // Buffer size for N draws = N * INDIRECT_DRAW_INDEXED_ARGS_SIZE
        let num_draws = 100;
        let expected_size = num_draws * INDIRECT_DRAW_INDEXED_ARGS_SIZE;
        assert_eq!(expected_size, 2000);
    }

    #[test]
    fn non_indexed_buffer_size_for_draws() {
        let num_draws = 100;
        let expected_size = num_draws * INDIRECT_DRAW_ARGS_SIZE;
        assert_eq!(expected_size, 1600);
    }

    #[test]
    fn dispatch_buffer_size_for_single() {
        assert_eq!(INDIRECT_DISPATCH_ARGS_SIZE, 12);
    }
}

// ============================================================================
// Test Category 7: Bytemuck Trait Verification (Pod, Zeroable)
// ============================================================================

mod bytemuck_traits {
    use super::*;

    #[test]
    fn indirect_draw_indexed_args_is_pod() {
        fn assert_pod<T: Pod>() {}
        assert_pod::<IndirectDrawIndexedArgs>();
    }

    #[test]
    fn indirect_draw_indexed_args_is_zeroable() {
        fn assert_zeroable<T: Zeroable>() {}
        assert_zeroable::<IndirectDrawIndexedArgs>();
    }

    #[test]
    fn indirect_draw_args_is_pod() {
        fn assert_pod<T: Pod>() {}
        assert_pod::<IndirectDrawArgs>();
    }

    #[test]
    fn indirect_draw_args_is_zeroable() {
        fn assert_zeroable<T: Zeroable>() {}
        assert_zeroable::<IndirectDrawArgs>();
    }

    #[test]
    fn indirect_dispatch_args_is_pod() {
        fn assert_pod<T: Pod>() {}
        assert_pod::<IndirectDispatchArgs>();
    }

    #[test]
    fn indirect_dispatch_args_is_zeroable() {
        fn assert_zeroable<T: Zeroable>() {}
        assert_zeroable::<IndirectDispatchArgs>();
    }

    #[test]
    fn can_cast_indexed_args_to_bytes() {
        let args = IndirectDrawIndexedArgs::new(100, 1, 0, 0, 0);
        let bytes: &[u8] = bytemuck::bytes_of(&args);
        assert_eq!(bytes.len(), 20);
    }

    #[test]
    fn can_cast_draw_args_to_bytes() {
        let args = IndirectDrawArgs::new(100, 1, 0, 0);
        let bytes: &[u8] = bytemuck::bytes_of(&args);
        assert_eq!(bytes.len(), 16);
    }

    #[test]
    fn can_cast_dispatch_args_to_bytes() {
        let args = IndirectDispatchArgs::new(4, 4, 4);
        let bytes: &[u8] = bytemuck::bytes_of(&args);
        assert_eq!(bytes.len(), 12);
    }

    #[test]
    fn can_cast_slice_to_bytes() {
        let commands = [
            IndirectDrawIndexedArgs::single(36, 0, 0),
            IndirectDrawIndexedArgs::single(24, 36, 100),
            IndirectDrawIndexedArgs::single(12, 60, 200),
        ];
        let bytes: &[u8] = bytemuck::cast_slice(&commands);
        assert_eq!(bytes.len(), 60); // 3 * 20
    }

    #[test]
    fn zeroed_is_all_zeros() {
        let args = IndirectDrawIndexedArgs::zeroed();
        let bytes: &[u8] = bytemuck::bytes_of(&args);
        assert!(bytes.iter().all(|&b| b == 0));
    }
}

// ============================================================================
// Test Category 8: Thread Safety (Send + Sync)
// ============================================================================

mod thread_safety {
    use super::*;

    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    #[test]
    fn indirect_draw_indexed_args_is_send() {
        assert_send::<IndirectDrawIndexedArgs>();
    }

    #[test]
    fn indirect_draw_indexed_args_is_sync() {
        assert_sync::<IndirectDrawIndexedArgs>();
    }

    #[test]
    fn indirect_draw_args_is_send() {
        assert_send::<IndirectDrawArgs>();
    }

    #[test]
    fn indirect_draw_args_is_sync() {
        assert_sync::<IndirectDrawArgs>();
    }

    #[test]
    fn indirect_dispatch_args_is_send() {
        assert_send::<IndirectDispatchArgs>();
    }

    #[test]
    fn indirect_dispatch_args_is_sync() {
        assert_sync::<IndirectDispatchArgs>();
    }

    #[test]
    fn indirect_tier_is_send() {
        assert_send::<IndirectTier>();
    }

    #[test]
    fn indirect_tier_is_sync() {
        assert_sync::<IndirectTier>();
    }

    #[test]
    fn draw_batch_builder_is_send() {
        assert_send::<DrawBatchBuilder>();
    }

    #[test]
    fn draw_batch_builder_is_sync() {
        assert_sync::<DrawBatchBuilder>();
    }

    #[test]
    fn multi_indirect_config_is_send() {
        assert_send::<MultiIndirectConfig>();
    }

    #[test]
    fn multi_indirect_config_is_sync() {
        assert_sync::<MultiIndirectConfig>();
    }

    #[test]
    fn can_share_args_across_threads() {
        use std::sync::Arc;
        use std::thread;

        let args = Arc::new(IndirectDrawIndexedArgs::new(100, 5, 0, 0, 0));

        let args_clone = Arc::clone(&args);
        let handle = thread::spawn(move || {
            assert!(args_clone.is_visible());
            args_clone.total_vertices()
        });

        let result = handle.join().unwrap();
        assert_eq!(result, 500);
    }
}

// ============================================================================
// Test Category 9: Real-World Usage Scenarios
// ============================================================================

mod usage_scenarios {
    use super::*;

    #[test]
    fn mesh_rendering_indexed() {
        // Typical mesh with index buffer
        let cube_args = IndirectDrawIndexedArgs::new(
            36,    // 12 triangles * 3 indices
            1,     // single instance
            0,     // start at first index
            0,     // no base vertex offset
            0,     // first instance
        );

        assert!(cube_args.is_visible());
        assert_eq!(cube_args.total_vertices(), 36);
    }

    #[test]
    fn instanced_mesh_rendering() {
        // Many instances of same mesh
        let tree_args = IndirectDrawIndexedArgs::new(
            1024,  // tree mesh indices
            500,   // 500 trees in scene
            0,
            0,
            0,
        );

        assert!(tree_args.is_visible());
        assert_eq!(tree_args.total_vertices(), 512_000);
    }

    #[test]
    fn submesh_rendering() {
        // Multiple submeshes with offsets
        let head_submesh = IndirectDrawIndexedArgs::single(1000, 0, 0);
        let body_submesh = IndirectDrawIndexedArgs::single(2000, 1000, 1000);
        let arms_submesh = IndirectDrawIndexedArgs::single(500, 3000, 3000);

        assert!(head_submesh.is_visible());
        assert!(body_submesh.is_visible());
        assert!(arms_submesh.is_visible());
    }

    #[test]
    fn procedural_geometry() {
        // Non-indexed procedural geometry
        let particles = IndirectDrawArgs::new(
            6,     // 2 triangles for billboard
            10000, // 10k particles
            0,
            0,
        );

        assert!(particles.is_visible());
        assert_eq!(particles.total_vertices(), 60_000);
    }

    #[test]
    fn point_cloud_rendering() {
        // Point cloud (1 vertex per point)
        let points = IndirectDrawArgs::new(
            1_000_000, // 1M points
            1,         // single "instance"
            0,
            0,
        );

        assert!(points.is_visible());
        assert_eq!(points.total_vertices(), 1_000_000);
    }

    #[test]
    fn compute_dispatch_for_culling() {
        // GPU culling compute dispatch
        let instance_count = 50_000u32;
        let workgroup_size = 256u32;

        let dispatch = IndirectDispatchArgs::for_elements(instance_count, workgroup_size);

        assert!(dispatch.is_active());
        // Should dispatch enough workgroups to cover all instances
        let total = dispatch.total_workgroups() as u32 * workgroup_size;
        assert!(total >= instance_count);
    }

    #[test]
    fn compute_dispatch_2d_image() {
        // Image processing compute (16x16 tiles)
        let image_width = 1920;
        let image_height = 1080;
        let tile_size = 16;

        let x_groups = (image_width + tile_size - 1) / tile_size;
        let y_groups = (image_height + tile_size - 1) / tile_size;

        let dispatch = IndirectDispatchArgs::grid_2d(x_groups, y_groups);

        assert!(dispatch.is_active());
    }

    #[test]
    fn batch_building() {
        let mut builder = DrawBatchBuilder::new();

        // Add various draw commands
        builder.add_indexed(IndirectDrawIndexedArgs::single(36, 0, 0));
        builder.add_indexed(IndirectDrawIndexedArgs::single(24, 36, 100));
        builder.add(IndirectDrawArgs::single(6, 0));

        assert_eq!(builder.indexed_count(), 2);
        assert_eq!(builder.count(), 1);
    }

    #[test]
    fn batch_building_with_capacity() {
        let builder = DrawBatchBuilder::with_capacity(100, 50);

        // Initially empty but with reserved capacity
        assert_eq!(builder.indexed_count(), 0);
        assert_eq!(builder.count(), 0);
    }

    #[test]
    fn batch_commands_access() {
        let mut builder = DrawBatchBuilder::new();

        builder.add_indexed(IndirectDrawIndexedArgs::new(100, 5, 0, 0, 0));
        builder.add_indexed(IndirectDrawIndexedArgs::new(200, 3, 100, 50, 5));

        let indexed = builder.indexed_commands();
        assert_eq!(indexed.len(), 2);
        assert!(indexed[0].is_visible());
        assert!(indexed[1].is_visible());
    }

    #[test]
    fn batch_clear() {
        let mut builder = DrawBatchBuilder::new();

        builder.add_indexed(IndirectDrawIndexedArgs::single(36, 0, 0));
        builder.add(IndirectDrawArgs::single(6, 0));

        builder.clear();

        assert_eq!(builder.indexed_count(), 0);
        assert_eq!(builder.count(), 0);
    }

    #[test]
    fn multi_indirect_config_indexed() {
        let config = MultiIndirectConfig::indexed(1000);

        // Config should be usable
        let _ = config;
    }

    #[test]
    fn multi_indirect_config_non_indexed() {
        let config = MultiIndirectConfig::non_indexed(1000);

        let _ = config;
    }

    #[test]
    fn multi_indirect_config_with_label() {
        let config = MultiIndirectConfig::indexed(500)
            .with_label("scene_draws");

        let _ = config;
    }
}

// ============================================================================
// Test Category 10: Constants and Size Assertions
// ============================================================================

mod constants_and_sizes {
    use super::*;

    #[test]
    fn struct_sizes_match_gpu_expectations() {
        // WebGPU/wgpu expects specific struct layouts for indirect commands

        // DrawIndexedIndirect: 5 u32s
        assert_eq!(mem::size_of::<IndirectDrawIndexedArgs>(), 5 * 4);

        // DrawIndirect: 4 u32s
        assert_eq!(mem::size_of::<IndirectDrawArgs>(), 4 * 4);

        // DispatchIndirect: 3 u32s
        assert_eq!(mem::size_of::<IndirectDispatchArgs>(), 3 * 4);
    }

    #[test]
    fn struct_alignments() {
        // All structs should be 4-byte aligned (u32 alignment)
        assert_eq!(mem::align_of::<IndirectDrawIndexedArgs>(), 4);
        assert_eq!(mem::align_of::<IndirectDrawArgs>(), 4);
        assert_eq!(mem::align_of::<IndirectDispatchArgs>(), 4);
    }

    #[test]
    fn size_constants_match_struct_sizes() {
        assert_eq!(INDIRECT_DRAW_INDEXED_ARGS_SIZE, mem::size_of::<IndirectDrawIndexedArgs>());
        assert_eq!(INDIRECT_DRAW_ARGS_SIZE, mem::size_of::<IndirectDrawArgs>());
        assert_eq!(INDIRECT_DISPATCH_ARGS_SIZE, mem::size_of::<IndirectDispatchArgs>());
    }

    #[test]
    fn default_max_draws_not_zero() {
        assert!(DEFAULT_MAX_DRAWS > 0);
    }

    #[test]
    fn constant_values_are_compile_time() {
        // These should all be const
        const _A: usize = INDIRECT_DRAW_INDEXED_ARGS_SIZE;
        const _B: usize = INDIRECT_DRAW_ARGS_SIZE;
        const _C: usize = INDIRECT_DISPATCH_ARGS_SIZE;
        const _D: u32 = DEFAULT_MAX_DRAWS;
    }

    #[test]
    fn structs_have_expected_fields_via_new() {
        // IndirectDrawIndexedArgs has 5 fields
        let _ = IndirectDrawIndexedArgs::new(1, 2, 3, 4, 5);

        // IndirectDrawArgs has 4 fields
        let _ = IndirectDrawArgs::new(1, 2, 3, 4);

        // IndirectDispatchArgs has 3 fields
        let _ = IndirectDispatchArgs::new(1, 2, 3);
    }
}

// ============================================================================
// Additional Edge Case Tests
// ============================================================================

mod edge_cases {
    use super::*;

    #[test]
    fn max_u32_values_indexed() {
        let args = IndirectDrawIndexedArgs::new(
            u32::MAX,
            u32::MAX,
            u32::MAX,
            i32::MAX,
            u32::MAX,
        );

        assert!(args.is_visible());
    }

    #[test]
    fn max_u32_values_non_indexed() {
        let args = IndirectDrawArgs::new(
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
        );

        assert!(args.is_visible());
    }

    #[test]
    fn max_u32_values_dispatch() {
        let args = IndirectDispatchArgs::new(
            u32::MAX,
            u32::MAX,
            u32::MAX,
        );

        assert!(args.is_active());
    }

    #[test]
    fn negative_base_vertex() {
        // base_vertex can be negative in indexed draws
        let args = IndirectDrawIndexedArgs::new(
            100,
            1,
            0,
            -50, // negative base vertex
            0,
        );

        assert!(args.is_visible());
    }

    #[test]
    fn clone_and_eq_indexed() {
        let args1 = IndirectDrawIndexedArgs::new(100, 5, 0, 0, 0);
        let args2 = args1.clone();

        assert_eq!(args1, args2);
    }

    #[test]
    fn clone_and_eq_non_indexed() {
        let args1 = IndirectDrawArgs::new(100, 5, 0, 0);
        let args2 = args1.clone();

        assert_eq!(args1, args2);
    }

    #[test]
    fn clone_and_eq_dispatch() {
        let args1 = IndirectDispatchArgs::new(4, 4, 4);
        let args2 = args1.clone();

        assert_eq!(args1, args2);
    }

    #[test]
    fn default_trait_indexed() {
        let args = IndirectDrawIndexedArgs::default();
        assert!(!args.is_visible());
        assert_eq!(args.total_vertices(), 0);
    }

    #[test]
    fn default_trait_non_indexed() {
        let args = IndirectDrawArgs::default();
        assert!(!args.is_visible());
        assert_eq!(args.total_vertices(), 0);
    }

    #[test]
    fn default_trait_dispatch() {
        let args = IndirectDispatchArgs::default();
        assert!(!args.is_active());
        assert_eq!(args.total_workgroups(), 0);
    }

    #[test]
    fn debug_output_indexed() {
        let args = IndirectDrawIndexedArgs::new(100, 5, 10, 20, 0);
        let debug = format!("{:?}", args);

        // Should contain struct name and field values
        assert!(debug.contains("IndirectDrawIndexedArgs"));
    }

    #[test]
    fn debug_output_non_indexed() {
        let args = IndirectDrawArgs::new(100, 5, 10, 0);
        let debug = format!("{:?}", args);

        assert!(debug.contains("IndirectDrawArgs"));
    }

    #[test]
    fn debug_output_dispatch() {
        let args = IndirectDispatchArgs::new(4, 4, 4);
        let debug = format!("{:?}", args);

        assert!(debug.contains("IndirectDispatchArgs"));
    }

    #[test]
    fn total_vertices_overflow_behavior() {
        // Large values that would overflow u32 but fit in u64
        let args = IndirectDrawIndexedArgs::new(
            1_000_000,
            1_000_000,
            0,
            0,
            0,
        );

        // total_vertices returns u64 to handle overflow
        let total = args.total_vertices();
        assert_eq!(total, 1_000_000_000_000u64);
    }

    #[test]
    fn total_workgroups_large_dispatch() {
        let args = IndirectDispatchArgs::new(1000, 1000, 1000);
        let total = args.total_workgroups();
        assert_eq!(total, 1_000_000_000u64);
    }
}

// ============================================================================
// Batch Builder Additional Tests
// ============================================================================

mod batch_builder_extended {
    use super::*;

    #[test]
    fn builder_new_is_empty() {
        let builder = DrawBatchBuilder::new();
        assert_eq!(builder.indexed_count(), 0);
        assert_eq!(builder.count(), 0);
    }

    #[test]
    fn builder_multiple_indexed_adds() {
        let mut builder = DrawBatchBuilder::new();

        for i in 0..10 {
            builder.add_indexed(IndirectDrawIndexedArgs::single(
                36 * (i + 1),
                i * 36,
                (i * 100) as i32,
            ));
        }

        assert_eq!(builder.indexed_count(), 10);
    }

    #[test]
    fn builder_multiple_non_indexed_adds() {
        let mut builder = DrawBatchBuilder::new();

        for i in 0..10 {
            builder.add(IndirectDrawArgs::single(
                100 + i,
                i * 100,
            ));
        }

        assert_eq!(builder.count(), 10);
    }

    #[test]
    fn builder_mixed_adds() {
        let mut builder = DrawBatchBuilder::new();

        builder.add_indexed(IndirectDrawIndexedArgs::single(36, 0, 0));
        builder.add(IndirectDrawArgs::single(6, 0));
        builder.add_indexed(IndirectDrawIndexedArgs::single(24, 36, 50));
        builder.add(IndirectDrawArgs::single(12, 6));

        assert_eq!(builder.indexed_count(), 2);
        assert_eq!(builder.count(), 2);
    }

    #[test]
    fn builder_commands_slice_access() {
        let mut builder = DrawBatchBuilder::new();

        builder.add_indexed(IndirectDrawIndexedArgs::new(100, 5, 0, 0, 0));

        let commands = builder.indexed_commands();
        assert_eq!(commands.len(), 1);
        assert_eq!(commands[0].total_vertices(), 500);
    }

    #[test]
    fn builder_non_indexed_slice_access() {
        let mut builder = DrawBatchBuilder::new();

        builder.add(IndirectDrawArgs::new(100, 3, 0, 0));

        let commands = builder.commands();
        assert_eq!(commands.len(), 1);
        assert_eq!(commands[0].total_vertices(), 300);
    }

    #[test]
    fn builder_clear_and_reuse() {
        let mut builder = DrawBatchBuilder::new();

        builder.add_indexed(IndirectDrawIndexedArgs::single(36, 0, 0));
        builder.add(IndirectDrawArgs::single(6, 0));

        builder.clear();

        // After clear, counts should be zero
        assert_eq!(builder.indexed_count(), 0);
        assert_eq!(builder.count(), 0);

        // Can add new commands after clear
        builder.add_indexed(IndirectDrawIndexedArgs::single(24, 0, 0));
        assert_eq!(builder.indexed_count(), 1);
    }
}

// ============================================================================
// MultiIndirectConfig Tests
// ============================================================================

mod multi_indirect_config_tests {
    use super::*;

    #[test]
    fn config_indexed_creation() {
        let config = MultiIndirectConfig::indexed(1000);
        let _ = config;
    }

    #[test]
    fn config_non_indexed_creation() {
        let config = MultiIndirectConfig::non_indexed(500);
        let _ = config;
    }

    #[test]
    fn config_with_label_chaining() {
        let config = MultiIndirectConfig::indexed(2000)
            .with_label("main_scene_draws");

        let _ = config;
    }

    #[test]
    fn config_default() {
        let config = MultiIndirectConfig::default();
        let _ = config;
    }

    #[test]
    fn config_clone() {
        let config = MultiIndirectConfig::indexed(1000);
        let cloned = config.clone();

        let _ = cloned;
    }

    #[test]
    fn config_debug() {
        let config = MultiIndirectConfig::indexed(500)
            .with_label("test");

        let debug = format!("{:?}", config);
        assert!(!debug.is_empty());
    }
}
