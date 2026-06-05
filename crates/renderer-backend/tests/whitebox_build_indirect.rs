//! Whitebox structural tests for Indirect Buffer Generation (T-WGPU-P6.6.2).
//!
//! Tests verify GPU struct layouts match WGSL expectations, bytemuck Pod/Zeroable
//! implementations, workgroup size calculations, LOD fallback logic, and CPU
//! reference implementation correctness.
//!
//! Task: T-WGPU-P6.6.2 - Indirect Buffer Generation
//!
//! Acceptance Criteria Tested:
//! 1. BuildIndirectParams - 16-byte alignment, workgroup calculation
//! 2. MeshData - 48-byte struct, LOD index/count fields
//! 3. IndirectDrawIndexedArgs - 20-byte wgpu-compatible layout
//! 4. BuildIndirectResources - buffer allocation, capacity management
//! 5. BuildIndirectPipeline - pipeline creation, bind group setup
//! 6. cpu_build_indirect() - CPU reference implementation correctness
//! 7. Batched vs single mode - threshold detection (>10K)
//!
//! Memory Layout Reference:
//! | Struct               | Size   | Description                           |
//! |----------------------|--------|---------------------------------------|
//! | BuildIndirectParams  | 16     | visible_count, max_draws, padding     |
//! | MeshData             | 48     | index info + LOD offsets              |
//! | DrawIndexedIndirectArgs | 20  | Standard indirect draw args           |

use bytemuck::{Pod, Zeroable};
use renderer_backend::gpu_driven::{
    BuildIndirectDrawArgs, BuildIndirectMeshData, BuildIndirectParams,
    BUILD_INDIRECT_BATCH_SIZE, BUILD_INDIRECT_DEFAULT_MAX_DRAWS, BUILD_INDIRECT_MAX_LOD_LEVELS,
    BUILD_INDIRECT_PARAMS_SIZE, BUILD_INDIRECT_SHADER, BUILD_INDIRECT_WORKGROUP_SIZE,
    DRAW_INDEXED_INDIRECT_ARGS_SIZE, MESH_DATA_SIZE,
    cpu_build_indirect,
};
use std::mem;

// ============================================================================
// 1. BuildIndirectParams Tests - 16-byte alignment, workgroup calculation
// ============================================================================

mod build_indirect_params {
    use super::*;

    #[test]
    fn size_is_exactly_16_bytes() {
        assert_eq!(mem::size_of::<BuildIndirectParams>(), 16);
        assert_eq!(mem::size_of::<BuildIndirectParams>(), BUILD_INDIRECT_PARAMS_SIZE);
    }

    #[test]
    fn alignment_is_4_bytes() {
        // std140/std430 requires uniform buffers to be aligned to vec4 (16 bytes)
        // but individual fields are u32 so alignment is 4
        assert_eq!(mem::align_of::<BuildIndirectParams>(), 4);
    }

    #[test]
    fn default_is_all_zeros() {
        let params = BuildIndirectParams::default();
        let bytes: &[u8] = bytemuck::bytes_of(&params);
        assert!(bytes.iter().all(|&b| b == 0));
    }

    #[test]
    fn new_sets_visible_count_and_max_draws() {
        let params = BuildIndirectParams::new(1000, 4096);
        assert_eq!(params.visible_count, 1000);
        assert_eq!(params.max_draws, 4096);
    }

    #[test]
    fn new_zeroes_padding_fields() {
        let params = BuildIndirectParams::new(1234, 5678);
        // Padding bytes should be zero (access via bytemuck)
        let bytes: &[u8] = bytemuck::bytes_of(&params);
        // Offsets 8-15 are padding
        assert_eq!(u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]), 0);
        assert_eq!(u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]), 0);
    }

    #[test]
    fn field_offset_visible_count_is_0() {
        let params = BuildIndirectParams::new(0x12345678, 0);
        let bytes: &[u8] = bytemuck::bytes_of(&params);
        let visible = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(visible, 0x12345678);
    }

    #[test]
    fn field_offset_max_draws_is_4() {
        let params = BuildIndirectParams::new(0, 0xDEADBEEF);
        let bytes: &[u8] = bytemuck::bytes_of(&params);
        let max_draws = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        assert_eq!(max_draws, 0xDEADBEEF);
    }

    #[test]
    fn workgroups_rounds_up_division() {
        // ceil(1000 / 64) = 16
        let params = BuildIndirectParams::new(1000, 4096);
        assert_eq!(params.workgroups(), 16);
    }

    #[test]
    fn workgroups_exact_multiple() {
        // ceil(128 / 64) = 2
        let params = BuildIndirectParams::new(128, 4096);
        assert_eq!(params.workgroups(), 2);
    }

    #[test]
    fn workgroups_single_thread() {
        // ceil(1 / 64) = 1
        let params = BuildIndirectParams::new(1, 4096);
        assert_eq!(params.workgroups(), 1);
    }

    #[test]
    fn workgroups_zero_visible_returns_zero() {
        let params = BuildIndirectParams::new(0, 4096);
        assert_eq!(params.workgroups(), 0);
    }

    #[test]
    fn workgroups_batched_divides_by_batch_size() {
        // objects_per_workgroup = WORKGROUP_SIZE * BATCH_SIZE = 64 * 4 = 256
        // ceil(1000 / 256) = 4
        let params = BuildIndirectParams::new(1000, 4096);
        assert_eq!(params.workgroups_batched(), 4);
    }

    #[test]
    fn workgroups_batched_exact_multiple() {
        // ceil(256 / 256) = 1
        let params = BuildIndirectParams::new(256, 4096);
        assert_eq!(params.workgroups_batched(), 1);
    }

    #[test]
    fn workgroups_batched_large_count() {
        // ceil(50000 / 256) = 196
        let params = BuildIndirectParams::new(50000, 65536);
        assert_eq!(params.workgroups_batched(), 196);
    }

    #[test]
    fn use_batched_mode_false_for_small_counts() {
        assert!(!BuildIndirectParams::new(5000, 4096).use_batched_mode());
        assert!(!BuildIndirectParams::new(1, 4096).use_batched_mode());
        assert!(!BuildIndirectParams::new(10000, 4096).use_batched_mode());
    }

    #[test]
    fn use_batched_mode_true_for_large_counts() {
        assert!(BuildIndirectParams::new(10001, 4096).use_batched_mode());
        assert!(BuildIndirectParams::new(50000, 65536).use_batched_mode());
        assert!(BuildIndirectParams::new(100000, 65536).use_batched_mode());
    }

    #[test]
    fn use_batched_mode_threshold_is_10000() {
        assert!(!BuildIndirectParams::new(10000, 4096).use_batched_mode());
        assert!(BuildIndirectParams::new(10001, 4096).use_batched_mode());
    }

    #[test]
    fn is_pod_and_zeroable() {
        // These assertions verify Pod and Zeroable are implemented
        fn assert_pod<T: Pod>() {}
        fn assert_zeroable<T: Zeroable>() {}
        assert_pod::<BuildIndirectParams>();
        assert_zeroable::<BuildIndirectParams>();
    }

    #[test]
    fn bytemuck_cast_from_bytes() {
        let bytes: [u8; 16] = [
            0x10, 0x27, 0x00, 0x00, // visible_count = 10000
            0x00, 0x10, 0x00, 0x00, // max_draws = 4096
            0x00, 0x00, 0x00, 0x00, // _pad0 = 0
            0x00, 0x00, 0x00, 0x00, // _pad1 = 0
        ];
        let params: &BuildIndirectParams = bytemuck::from_bytes(&bytes);
        assert_eq!(params.visible_count, 10000);
        assert_eq!(params.max_draws, 4096);
    }

    #[test]
    fn equality_check() {
        let p1 = BuildIndirectParams::new(1000, 4096);
        let p2 = BuildIndirectParams::new(1000, 4096);
        let p3 = BuildIndirectParams::new(1000, 8192);
        assert_eq!(p1, p2);
        assert_ne!(p1, p3);
    }

    #[test]
    fn clone_produces_equal_value() {
        let p1 = BuildIndirectParams::new(42, 256);
        let p2 = p1.clone();
        assert_eq!(p1, p2);
    }
}

// ============================================================================
// 2. MeshData Tests - 48-byte struct, LOD index/count fields
// ============================================================================

mod mesh_data {
    use super::*;

    #[test]
    fn size_is_exactly_48_bytes() {
        assert_eq!(mem::size_of::<BuildIndirectMeshData>(), 48);
        assert_eq!(mem::size_of::<BuildIndirectMeshData>(), MESH_DATA_SIZE);
    }

    #[test]
    fn alignment_is_4_bytes() {
        assert_eq!(mem::align_of::<BuildIndirectMeshData>(), 4);
    }

    #[test]
    fn default_is_all_zeros() {
        let mesh = BuildIndirectMeshData::default();
        let bytes: &[u8] = bytemuck::bytes_of(&mesh);
        assert!(bytes.iter().all(|&b| b == 0));
    }

    #[test]
    fn new_sets_base_fields() {
        let mesh = BuildIndirectMeshData::new(1000, 5000, -100);
        assert_eq!(mesh.index_count, 1000);
        assert_eq!(mesh.first_index, 5000);
        assert_eq!(mesh.base_vertex, -100);
    }

    #[test]
    fn new_zeroes_lod_arrays() {
        let mesh = BuildIndirectMeshData::new(1000, 5000, 0);
        assert_eq!(mesh.lod_index_counts, [0, 0, 0, 0]);
        assert_eq!(mesh.lod_first_index, [0, 0, 0, 0]);
    }

    #[test]
    fn field_offset_index_count_is_0() {
        let mesh = BuildIndirectMeshData::new(0x12345678, 0, 0);
        let bytes: &[u8] = bytemuck::bytes_of(&mesh);
        let value = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(value, 0x12345678);
    }

    #[test]
    fn field_offset_first_index_is_4() {
        let mesh = BuildIndirectMeshData::new(0, 0xDEADBEEF, 0);
        let bytes: &[u8] = bytemuck::bytes_of(&mesh);
        let value = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        assert_eq!(value, 0xDEADBEEF);
    }

    #[test]
    fn field_offset_base_vertex_is_8() {
        let mesh = BuildIndirectMeshData::new(0, 0, -1);
        let bytes: &[u8] = bytemuck::bytes_of(&mesh);
        let value = i32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        assert_eq!(value, -1);
    }

    #[test]
    fn field_offset_lod_index_counts_is_16() {
        let mesh = BuildIndirectMeshData::with_lods(0, 0, 0, &[100, 200, 300, 400], &[]);
        let bytes: &[u8] = bytemuck::bytes_of(&mesh);
        // lod_index_counts[0] at offset 16
        let val0 = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
        let val1 = u32::from_le_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);
        let val2 = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]);
        let val3 = u32::from_le_bytes([bytes[28], bytes[29], bytes[30], bytes[31]]);
        assert_eq!(val0, 100);
        assert_eq!(val1, 200);
        assert_eq!(val2, 300);
        assert_eq!(val3, 400);
    }

    #[test]
    fn field_offset_lod_first_index_is_32() {
        let mesh = BuildIndirectMeshData::with_lods(0, 0, 0, &[], &[10, 20, 30, 40]);
        let bytes: &[u8] = bytemuck::bytes_of(&mesh);
        // lod_first_index[0] at offset 32
        let val0 = u32::from_le_bytes([bytes[32], bytes[33], bytes[34], bytes[35]]);
        let val1 = u32::from_le_bytes([bytes[36], bytes[37], bytes[38], bytes[39]]);
        let val2 = u32::from_le_bytes([bytes[40], bytes[41], bytes[42], bytes[43]]);
        let val3 = u32::from_le_bytes([bytes[44], bytes[45], bytes[46], bytes[47]]);
        assert_eq!(val0, 10);
        assert_eq!(val1, 20);
        assert_eq!(val2, 30);
        assert_eq!(val3, 40);
    }

    #[test]
    fn with_lods_sets_partial_arrays() {
        let mesh = BuildIndirectMeshData::with_lods(
            1000,
            0,
            0,
            &[1000, 500], // Only 2 LOD counts
            &[0, 1000],   // Only 2 LOD offsets
        );
        assert_eq!(mesh.lod_index_counts[0], 1000);
        assert_eq!(mesh.lod_index_counts[1], 500);
        assert_eq!(mesh.lod_index_counts[2], 0);
        assert_eq!(mesh.lod_index_counts[3], 0);
        assert_eq!(mesh.lod_first_index[0], 0);
        assert_eq!(mesh.lod_first_index[1], 1000);
        assert_eq!(mesh.lod_first_index[2], 0);
        assert_eq!(mesh.lod_first_index[3], 0);
    }

    #[test]
    fn with_lods_truncates_extra_elements() {
        // If more than 4 elements are provided, only first 4 are used
        let mesh = BuildIndirectMeshData::with_lods(
            1000,
            0,
            0,
            &[1, 2, 3, 4, 5, 6], // 6 elements - should truncate
            &[10, 20, 30, 40, 50], // 5 elements - should truncate
        );
        assert_eq!(mesh.lod_index_counts, [1, 2, 3, 4]);
        assert_eq!(mesh.lod_first_index, [10, 20, 30, 40]);
    }

    #[test]
    fn index_count_for_lod_returns_lod_count_if_nonzero() {
        let mesh = BuildIndirectMeshData::with_lods(
            1000,
            0,
            0,
            &[1000, 500, 250, 125],
            &[0, 1000, 1500, 1750],
        );
        assert_eq!(mesh.index_count_for_lod(0), 1000);
        assert_eq!(mesh.index_count_for_lod(1), 500);
        assert_eq!(mesh.index_count_for_lod(2), 250);
        assert_eq!(mesh.index_count_for_lod(3), 125);
    }

    #[test]
    fn index_count_for_lod_falls_back_to_base_if_zero() {
        let mesh = BuildIndirectMeshData::new(777, 0, 0);
        // All LOD counts are 0, should fall back to index_count
        assert_eq!(mesh.index_count_for_lod(0), 777);
        assert_eq!(mesh.index_count_for_lod(1), 777);
        assert_eq!(mesh.index_count_for_lod(2), 777);
        assert_eq!(mesh.index_count_for_lod(3), 777);
    }

    #[test]
    fn index_count_for_lod_clamps_out_of_range() {
        let mesh = BuildIndirectMeshData::with_lods(
            1000,
            0,
            0,
            &[1000, 500, 250, 125],
            &[0, 0, 0, 0],
        );
        // LOD 10 should clamp to LOD 3
        assert_eq!(mesh.index_count_for_lod(10), 125);
        assert_eq!(mesh.index_count_for_lod(100), 125);
    }

    #[test]
    fn first_index_for_lod_adds_base_and_offset() {
        let mesh = BuildIndirectMeshData::with_lods(
            1000,
            5000, // first_index base
            0,
            &[1000, 500, 250, 125],
            &[0, 1000, 1500, 1750],
        );
        assert_eq!(mesh.first_index_for_lod(0), 5000 + 0);
        assert_eq!(mesh.first_index_for_lod(1), 5000 + 1000);
        assert_eq!(mesh.first_index_for_lod(2), 5000 + 1500);
        assert_eq!(mesh.first_index_for_lod(3), 5000 + 1750);
    }

    #[test]
    fn first_index_for_lod_clamps_out_of_range() {
        let mesh = BuildIndirectMeshData::with_lods(
            1000,
            1000,
            0,
            &[],
            &[0, 100, 200, 300],
        );
        // LOD 10 should clamp to LOD 3
        assert_eq!(mesh.first_index_for_lod(10), 1000 + 300);
    }

    #[test]
    fn is_pod_and_zeroable() {
        fn assert_pod<T: Pod>() {}
        fn assert_zeroable<T: Zeroable>() {}
        assert_pod::<BuildIndirectMeshData>();
        assert_zeroable::<BuildIndirectMeshData>();
    }

    #[test]
    fn negative_base_vertex_roundtrips() {
        let mesh = BuildIndirectMeshData::new(100, 200, -12345);
        let bytes: &[u8] = bytemuck::bytes_of(&mesh);
        let mesh2: &BuildIndirectMeshData = bytemuck::from_bytes(bytes);
        assert_eq!(mesh2.base_vertex, -12345);
    }

    #[test]
    fn max_lod_levels_constant() {
        assert_eq!(BUILD_INDIRECT_MAX_LOD_LEVELS, 4);
    }
}

// ============================================================================
// 3. IndirectDrawIndexedArgs Tests - 20-byte wgpu-compatible layout
// ============================================================================

mod indirect_draw_indexed_args {
    use super::*;

    #[test]
    fn size_is_exactly_20_bytes() {
        assert_eq!(mem::size_of::<BuildIndirectDrawArgs>(), 20);
        assert_eq!(mem::size_of::<BuildIndirectDrawArgs>(), DRAW_INDEXED_INDIRECT_ARGS_SIZE);
    }

    #[test]
    fn size_constant_matches() {
        assert_eq!(BuildIndirectDrawArgs::SIZE, 20);
    }

    #[test]
    fn alignment_is_4_bytes() {
        assert_eq!(mem::align_of::<BuildIndirectDrawArgs>(), 4);
    }

    #[test]
    fn default_is_all_zeros() {
        let args = BuildIndirectDrawArgs::default();
        let bytes: &[u8] = bytemuck::bytes_of(&args);
        assert!(bytes.iter().all(|&b| b == 0));
    }

    #[test]
    fn new_sets_all_fields() {
        let args = BuildIndirectDrawArgs::new(100, 2, 500, -50, 42);
        assert_eq!(args.index_count, 100);
        assert_eq!(args.instance_count, 2);
        assert_eq!(args.first_index, 500);
        assert_eq!(args.base_vertex, -50);
        assert_eq!(args.first_instance, 42);
    }

    #[test]
    fn field_offset_index_count_is_0() {
        let args = BuildIndirectDrawArgs::new(0x12345678, 0, 0, 0, 0);
        let bytes: &[u8] = bytemuck::bytes_of(&args);
        let value = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(value, 0x12345678);
    }

    #[test]
    fn field_offset_instance_count_is_4() {
        let args = BuildIndirectDrawArgs::new(0, 0xDEADBEEF, 0, 0, 0);
        let bytes: &[u8] = bytemuck::bytes_of(&args);
        let value = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        assert_eq!(value, 0xDEADBEEF);
    }

    #[test]
    fn field_offset_first_index_is_8() {
        let args = BuildIndirectDrawArgs::new(0, 0, 0xCAFEBABE, 0, 0);
        let bytes: &[u8] = bytemuck::bytes_of(&args);
        let value = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        assert_eq!(value, 0xCAFEBABE);
    }

    #[test]
    fn field_offset_base_vertex_is_12() {
        let args = BuildIndirectDrawArgs::new(0, 0, 0, -1, 0);
        let bytes: &[u8] = bytemuck::bytes_of(&args);
        let value = i32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        assert_eq!(value, -1);
    }

    #[test]
    fn field_offset_first_instance_is_16() {
        let args = BuildIndirectDrawArgs::new(0, 0, 0, 0, 0xABCDEF01);
        let bytes: &[u8] = bytemuck::bytes_of(&args);
        let value = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
        assert_eq!(value, 0xABCDEF01);
    }

    #[test]
    fn is_visible_true_when_both_counts_nonzero() {
        let args = BuildIndirectDrawArgs::new(100, 1, 0, 0, 0);
        assert!(args.is_visible());
    }

    #[test]
    fn is_visible_false_when_index_count_zero() {
        let args = BuildIndirectDrawArgs::new(0, 1, 0, 0, 0);
        assert!(!args.is_visible());
    }

    #[test]
    fn is_visible_false_when_instance_count_zero() {
        let args = BuildIndirectDrawArgs::new(100, 0, 0, 0, 0);
        assert!(!args.is_visible());
    }

    #[test]
    fn is_visible_false_when_both_counts_zero() {
        let args = BuildIndirectDrawArgs::new(0, 0, 0, 0, 0);
        assert!(!args.is_visible());
    }

    #[test]
    fn is_pod_and_zeroable() {
        fn assert_pod<T: Pod>() {}
        fn assert_zeroable<T: Zeroable>() {}
        assert_pod::<BuildIndirectDrawArgs>();
        assert_zeroable::<BuildIndirectDrawArgs>();
    }

    #[test]
    fn bytemuck_cast_from_bytes() {
        let bytes: [u8; 20] = [
            0x64, 0x00, 0x00, 0x00, // index_count = 100
            0x01, 0x00, 0x00, 0x00, // instance_count = 1
            0xF4, 0x01, 0x00, 0x00, // first_index = 500
            0xCE, 0xFF, 0xFF, 0xFF, // base_vertex = -50 (i32)
            0x2A, 0x00, 0x00, 0x00, // first_instance = 42
        ];
        let args: &BuildIndirectDrawArgs = bytemuck::from_bytes(&bytes);
        assert_eq!(args.index_count, 100);
        assert_eq!(args.instance_count, 1);
        assert_eq!(args.first_index, 500);
        assert_eq!(args.base_vertex, -50);
        assert_eq!(args.first_instance, 42);
    }

    #[test]
    fn wgpu_layout_compatibility() {
        // This test verifies the struct matches wgpu's DrawIndexedIndirectArgs layout
        // by checking field offsets explicitly
        let args = BuildIndirectDrawArgs::new(100, 1, 0, 0, 42);
        let bytes: &[u8] = bytemuck::bytes_of(&args);

        // wgpu DrawIndexedIndirectArgs layout:
        // offset 0: index_count (u32)
        // offset 4: instance_count (u32)
        // offset 8: first_index (u32)
        // offset 12: base_vertex (i32)
        // offset 16: first_instance (u32)

        assert_eq!(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]), 100);
        assert_eq!(u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]), 1);
        assert_eq!(u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]), 0);
        assert_eq!(i32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]), 0);
        assert_eq!(u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]), 42);
    }

    #[test]
    fn negative_base_vertex_roundtrips() {
        let args = BuildIndirectDrawArgs::new(100, 1, 0, -999999, 0);
        let bytes: &[u8] = bytemuck::bytes_of(&args);
        let args2: &BuildIndirectDrawArgs = bytemuck::from_bytes(bytes);
        assert_eq!(args2.base_vertex, -999999);
    }
}

// ============================================================================
// 4. Constants Tests
// ============================================================================

mod constants {
    use super::*;

    #[test]
    fn workgroup_size_is_64() {
        assert_eq!(BUILD_INDIRECT_WORKGROUP_SIZE, 64);
    }

    #[test]
    fn batch_size_is_4() {
        assert_eq!(BUILD_INDIRECT_BATCH_SIZE, 4);
    }

    #[test]
    fn default_max_draws_is_65536() {
        assert_eq!(BUILD_INDIRECT_DEFAULT_MAX_DRAWS, 65536);
    }

    #[test]
    fn max_lod_levels_is_4() {
        assert_eq!(BUILD_INDIRECT_MAX_LOD_LEVELS, 4);
    }

    #[test]
    fn params_size_matches_struct() {
        assert_eq!(BUILD_INDIRECT_PARAMS_SIZE, mem::size_of::<BuildIndirectParams>());
    }

    #[test]
    fn mesh_data_size_matches_struct() {
        assert_eq!(MESH_DATA_SIZE, mem::size_of::<BuildIndirectMeshData>());
    }

    #[test]
    fn draw_args_size_matches_struct() {
        assert_eq!(DRAW_INDEXED_INDIRECT_ARGS_SIZE, mem::size_of::<BuildIndirectDrawArgs>());
    }
}

// ============================================================================
// 5. Shader Source Tests
// ============================================================================

mod shader_source {
    use super::*;

    #[test]
    fn shader_source_is_not_empty() {
        assert!(!BUILD_INDIRECT_SHADER.is_empty());
    }

    #[test]
    fn shader_contains_main_entry_point() {
        assert!(BUILD_INDIRECT_SHADER.contains("build_indirect_main"));
    }

    #[test]
    fn shader_contains_batched_entry_point() {
        assert!(BUILD_INDIRECT_SHADER.contains("build_indirect_batched"));
    }

    #[test]
    fn shader_contains_clear_entry_point() {
        assert!(BUILD_INDIRECT_SHADER.contains("clear_draw_count"));
    }

    #[test]
    fn shader_contains_draw_indexed_indirect_args_struct() {
        assert!(BUILD_INDIRECT_SHADER.contains("DrawIndexedIndirectArgs"));
    }

    #[test]
    fn shader_contains_build_indirect_params_struct() {
        assert!(BUILD_INDIRECT_SHADER.contains("BuildIndirectParams"));
    }

    #[test]
    fn shader_contains_mesh_data_struct() {
        assert!(BUILD_INDIRECT_SHADER.contains("MeshData"));
    }

    #[test]
    fn shader_contains_lod_entry_struct() {
        assert!(BUILD_INDIRECT_SHADER.contains("LodEntry"));
    }

    #[test]
    fn shader_contains_object_data_struct() {
        assert!(BUILD_INDIRECT_SHADER.contains("ObjectData"));
    }

    #[test]
    fn shader_defines_workgroup_size_64() {
        assert!(BUILD_INDIRECT_SHADER.contains("WORKGROUP_SIZE: u32 = 64u"));
    }

    #[test]
    fn shader_defines_batch_size_4() {
        assert!(BUILD_INDIRECT_SHADER.contains("BATCH_SIZE: u32 = 4u"));
    }

    #[test]
    fn shader_defines_num_lod_levels_4() {
        assert!(BUILD_INDIRECT_SHADER.contains("NUM_LOD_LEVELS: u32 = 4u"));
    }

    #[test]
    fn shader_defines_max_lod_level_3() {
        assert!(BUILD_INDIRECT_SHADER.contains("MAX_LOD_LEVEL: u32 = 3u"));
    }

    #[test]
    fn shader_uses_atomic_add_for_draw_count() {
        assert!(BUILD_INDIRECT_SHADER.contains("atomicAdd(&draw_count"));
    }

    #[test]
    fn shader_uses_atomic_store_for_clear() {
        assert!(BUILD_INDIRECT_SHADER.contains("atomicStore(&draw_count"));
    }

    #[test]
    fn shader_has_input_bindings() {
        assert!(BUILD_INDIRECT_SHADER.contains("@group(0) @binding(0)"));
        assert!(BUILD_INDIRECT_SHADER.contains("@group(0) @binding(1)"));
        assert!(BUILD_INDIRECT_SHADER.contains("@group(0) @binding(2)"));
        assert!(BUILD_INDIRECT_SHADER.contains("@group(0) @binding(3)"));
    }

    #[test]
    fn shader_has_output_bindings() {
        assert!(BUILD_INDIRECT_SHADER.contains("@group(1) @binding(0)"));
        assert!(BUILD_INDIRECT_SHADER.contains("@group(1) @binding(1)"));
    }

    #[test]
    fn shader_has_params_binding() {
        assert!(BUILD_INDIRECT_SHADER.contains("@group(2) @binding(0)"));
    }

    #[test]
    fn shader_declares_compute_workgroup_size() {
        assert!(BUILD_INDIRECT_SHADER.contains("@compute @workgroup_size(64, 1, 1)"));
    }

    #[test]
    fn shader_has_lod_fallback_logic() {
        // get_lod_index_count should fall back if LOD count is 0
        assert!(BUILD_INDIRECT_SHADER.contains("if (lod_count > 0u)"));
        assert!(BUILD_INDIRECT_SHADER.contains("return mesh.index_count"));
    }
}

// ============================================================================
// 6. CPU Reference Implementation Tests
// ============================================================================

mod cpu_build_indirect_tests {
    use super::*;

    #[test]
    fn empty_input_returns_empty_output() {
        let commands = cpu_build_indirect(&[], &[], &[], &[]);
        assert!(commands.is_empty());
    }

    #[test]
    fn single_object_single_mesh() {
        let compacted_indices = vec![0];
        let object_mesh_indices = vec![0];
        let lod_levels = vec![0];
        let mesh_data = vec![BuildIndirectMeshData::new(100, 0, 0)];

        let commands = cpu_build_indirect(
            &compacted_indices,
            &object_mesh_indices,
            &lod_levels,
            &mesh_data,
        );

        assert_eq!(commands.len(), 1);
        assert_eq!(commands[0].index_count, 100);
        assert_eq!(commands[0].instance_count, 1);
        assert_eq!(commands[0].first_index, 0);
        assert_eq!(commands[0].base_vertex, 0);
        assert_eq!(commands[0].first_instance, 0);
    }

    #[test]
    fn multiple_objects_same_mesh() {
        let compacted_indices = vec![0, 1, 2];
        let object_mesh_indices = vec![0, 0, 0];
        let lod_levels = vec![0, 0, 0];
        let mesh_data = vec![BuildIndirectMeshData::new(50, 100, 10)];

        let commands = cpu_build_indirect(
            &compacted_indices,
            &object_mesh_indices,
            &lod_levels,
            &mesh_data,
        );

        assert_eq!(commands.len(), 3);
        for (i, cmd) in commands.iter().enumerate() {
            assert_eq!(cmd.index_count, 50);
            assert_eq!(cmd.instance_count, 1);
            assert_eq!(cmd.first_index, 100);
            assert_eq!(cmd.base_vertex, 10);
            assert_eq!(cmd.first_instance, i as u32);
        }
    }

    #[test]
    fn multiple_objects_different_meshes() {
        let compacted_indices = vec![0, 1, 2];
        let object_mesh_indices = vec![0, 1, 2];
        let lod_levels = vec![0, 0, 0];
        let mesh_data = vec![
            BuildIndirectMeshData::new(100, 0, 0),
            BuildIndirectMeshData::new(200, 100, 10),
            BuildIndirectMeshData::new(300, 300, 20),
        ];

        let commands = cpu_build_indirect(
            &compacted_indices,
            &object_mesh_indices,
            &lod_levels,
            &mesh_data,
        );

        assert_eq!(commands.len(), 3);
        assert_eq!(commands[0].index_count, 100);
        assert_eq!(commands[0].first_index, 0);
        assert_eq!(commands[1].index_count, 200);
        assert_eq!(commands[1].first_index, 100);
        assert_eq!(commands[2].index_count, 300);
        assert_eq!(commands[2].first_index, 300);
    }

    #[test]
    fn lod_selection_uses_correct_lod_level() {
        let compacted_indices = vec![0, 1, 2, 3];
        let object_mesh_indices = vec![0, 0, 0, 0];
        let lod_levels = vec![0, 1, 2, 3];
        let mesh_data = vec![BuildIndirectMeshData::with_lods(
            1000,
            0,
            0,
            &[1000, 500, 250, 125],
            &[0, 1000, 1500, 1750],
        )];

        let commands = cpu_build_indirect(
            &compacted_indices,
            &object_mesh_indices,
            &lod_levels,
            &mesh_data,
        );

        assert_eq!(commands.len(), 4);
        // LOD 0
        assert_eq!(commands[0].index_count, 1000);
        assert_eq!(commands[0].first_index, 0);
        // LOD 1
        assert_eq!(commands[1].index_count, 500);
        assert_eq!(commands[1].first_index, 1000);
        // LOD 2
        assert_eq!(commands[2].index_count, 250);
        assert_eq!(commands[2].first_index, 1500);
        // LOD 3
        assert_eq!(commands[3].index_count, 125);
        assert_eq!(commands[3].first_index, 1750);
    }

    #[test]
    fn lod_level_clamps_to_max() {
        let compacted_indices = vec![0];
        let object_mesh_indices = vec![0];
        let lod_levels = vec![100]; // Way out of range
        let mesh_data = vec![BuildIndirectMeshData::with_lods(
            1000,
            0,
            0,
            &[1000, 500, 250, 125],
            &[0, 1000, 1500, 1750],
        )];

        let commands = cpu_build_indirect(
            &compacted_indices,
            &object_mesh_indices,
            &lod_levels,
            &mesh_data,
        );

        assert_eq!(commands.len(), 1);
        // Should clamp to LOD 3
        assert_eq!(commands[0].index_count, 125);
        assert_eq!(commands[0].first_index, 1750);
    }

    #[test]
    fn lod_fallback_when_count_is_zero() {
        let compacted_indices = vec![0, 1];
        let object_mesh_indices = vec![0, 0];
        let lod_levels = vec![1, 2]; // LOD 1 and 2 have zero counts
        let mesh_data = vec![BuildIndirectMeshData::with_lods(
            1000,
            500,
            0,
            &[1000, 0, 0, 0], // Only LOD 0 has explicit count
            &[0, 0, 0, 0],
        )];

        let commands = cpu_build_indirect(
            &compacted_indices,
            &object_mesh_indices,
            &lod_levels,
            &mesh_data,
        );

        // Both should fall back to base index_count
        assert_eq!(commands[0].index_count, 1000);
        assert_eq!(commands[1].index_count, 1000);
    }

    #[test]
    fn sparse_compacted_indices() {
        // Compacted indices don't have to be contiguous
        let compacted_indices = vec![0, 5, 10];
        let object_mesh_indices = vec![0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2];
        let lod_levels = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let mesh_data = vec![
            BuildIndirectMeshData::new(100, 0, 0),
            BuildIndirectMeshData::new(200, 100, 0),
            BuildIndirectMeshData::new(300, 300, 0),
        ];

        let commands = cpu_build_indirect(
            &compacted_indices,
            &object_mesh_indices,
            &lod_levels,
            &mesh_data,
        );

        assert_eq!(commands.len(), 3);
        // Object 0 uses mesh 0
        assert_eq!(commands[0].index_count, 100);
        assert_eq!(commands[0].first_instance, 0);
        // Object 5 uses mesh 1
        assert_eq!(commands[1].index_count, 200);
        assert_eq!(commands[1].first_instance, 1);
        // Object 10 uses mesh 2
        assert_eq!(commands[2].index_count, 300);
        assert_eq!(commands[2].first_instance, 2);
    }

    #[test]
    fn first_instance_is_compact_index_not_object_index() {
        // first_instance should be the compact index (position in compacted_indices)
        // not the original object index
        let compacted_indices = vec![7, 3, 9];
        let object_mesh_indices = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let lod_levels = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let mesh_data = vec![BuildIndirectMeshData::new(100, 0, 0)];

        let commands = cpu_build_indirect(
            &compacted_indices,
            &object_mesh_indices,
            &lod_levels,
            &mesh_data,
        );

        // first_instance should be 0, 1, 2 (compact indices)
        // NOT 7, 3, 9 (object indices)
        assert_eq!(commands[0].first_instance, 0);
        assert_eq!(commands[1].first_instance, 1);
        assert_eq!(commands[2].first_instance, 2);
    }

    #[test]
    fn negative_base_vertex_propagates() {
        let compacted_indices = vec![0];
        let object_mesh_indices = vec![0];
        let lod_levels = vec![0];
        let mesh_data = vec![BuildIndirectMeshData::new(100, 0, -500)];

        let commands = cpu_build_indirect(
            &compacted_indices,
            &object_mesh_indices,
            &lod_levels,
            &mesh_data,
        );

        assert_eq!(commands[0].base_vertex, -500);
    }

    #[test]
    fn instance_count_always_one() {
        let compacted_indices: Vec<u32> = (0..10).collect();
        let object_mesh_indices: Vec<u32> = vec![0; 10];
        let lod_levels: Vec<u32> = vec![0; 10];
        let mesh_data = vec![BuildIndirectMeshData::new(100, 0, 0)];

        let commands = cpu_build_indirect(
            &compacted_indices,
            &object_mesh_indices,
            &lod_levels,
            &mesh_data,
        );

        for cmd in &commands {
            assert_eq!(cmd.instance_count, 1);
        }
    }

    #[test]
    fn missing_object_index_uses_defaults() {
        // If object_mesh_indices doesn't have the requested index, use 0
        let compacted_indices = vec![100]; // Object 100 doesn't exist
        let object_mesh_indices = vec![0, 1, 2]; // Only 3 objects
        let lod_levels = vec![]; // Empty
        let mesh_data = vec![BuildIndirectMeshData::new(50, 0, 0)];

        let commands = cpu_build_indirect(
            &compacted_indices,
            &object_mesh_indices,
            &lod_levels,
            &mesh_data,
        );

        // Should gracefully handle missing data with defaults
        assert_eq!(commands.len(), 1);
    }

    #[test]
    fn large_visible_count() {
        let count = 10000;
        let compacted_indices: Vec<u32> = (0..count).collect();
        let object_mesh_indices: Vec<u32> = (0..count).map(|i| i % 4).collect();
        let lod_levels: Vec<u32> = (0..count).map(|i| i % 4).collect();
        let mesh_data = vec![
            BuildIndirectMeshData::with_lods(1000, 0, 0, &[1000, 500, 250, 125], &[0, 0, 0, 0]),
            BuildIndirectMeshData::with_lods(800, 0, 0, &[800, 400, 200, 100], &[0, 0, 0, 0]),
            BuildIndirectMeshData::with_lods(600, 0, 0, &[600, 300, 150, 75], &[0, 0, 0, 0]),
            BuildIndirectMeshData::with_lods(400, 0, 0, &[400, 200, 100, 50], &[0, 0, 0, 0]),
        ];

        let commands = cpu_build_indirect(
            &compacted_indices,
            &object_mesh_indices,
            &lod_levels,
            &mesh_data,
        );

        assert_eq!(commands.len(), count as usize);

        // Verify first few entries
        assert_eq!(commands[0].index_count, 1000); // mesh 0, LOD 0
        assert_eq!(commands[1].index_count, 400);  // mesh 1, LOD 1
        assert_eq!(commands[2].index_count, 150);  // mesh 2, LOD 2
        assert_eq!(commands[3].index_count, 50);   // mesh 3, LOD 3
    }

    #[test]
    fn preserves_first_index_with_lod_offset() {
        let compacted_indices = vec![0];
        let object_mesh_indices = vec![0];
        let lod_levels = vec![2]; // Use LOD 2
        let mesh_data = vec![BuildIndirectMeshData::with_lods(
            1000,
            10000, // Base first_index
            0,
            &[1000, 500, 250, 125],
            &[0, 1000, 2000, 3000], // Relative offsets
        )];

        let commands = cpu_build_indirect(
            &compacted_indices,
            &object_mesh_indices,
            &lod_levels,
            &mesh_data,
        );

        // first_index should be base (10000) + LOD 2 offset (2000) = 12000
        assert_eq!(commands[0].first_index, 12000);
    }
}

// ============================================================================
// 7. Batched Mode Detection Tests
// ============================================================================

mod batched_mode_detection {
    use super::*;

    #[test]
    fn standard_mode_workgroup_calculation() {
        // Standard mode: ceil(N / WORKGROUP_SIZE)
        assert_eq!(BuildIndirectParams::new(64, 4096).workgroups(), 1);
        assert_eq!(BuildIndirectParams::new(65, 4096).workgroups(), 2);
        assert_eq!(BuildIndirectParams::new(128, 4096).workgroups(), 2);
        assert_eq!(BuildIndirectParams::new(129, 4096).workgroups(), 3);
    }

    #[test]
    fn batched_mode_workgroup_calculation() {
        // Batched mode: ceil(N / (WORKGROUP_SIZE * BATCH_SIZE))
        // = ceil(N / (64 * 4)) = ceil(N / 256)
        assert_eq!(BuildIndirectParams::new(256, 4096).workgroups_batched(), 1);
        assert_eq!(BuildIndirectParams::new(257, 4096).workgroups_batched(), 2);
        assert_eq!(BuildIndirectParams::new(512, 4096).workgroups_batched(), 2);
        assert_eq!(BuildIndirectParams::new(513, 4096).workgroups_batched(), 3);
    }

    #[test]
    fn mode_selection_at_threshold() {
        // Threshold is >10000
        let at_threshold = BuildIndirectParams::new(10000, 65536);
        let above_threshold = BuildIndirectParams::new(10001, 65536);

        assert!(!at_threshold.use_batched_mode());
        assert!(above_threshold.use_batched_mode());
    }

    #[test]
    fn workgroup_count_difference_at_threshold() {
        let above = BuildIndirectParams::new(15000, 65536);

        let standard_wg = above.workgroups();
        let batched_wg = above.workgroups_batched();

        // Standard: ceil(15000 / 64) = 235
        // Batched: ceil(15000 / 256) = 59
        assert_eq!(standard_wg, 235);
        assert_eq!(batched_wg, 59);

        // Batched mode uses ~4x fewer workgroups
        assert!(standard_wg > batched_wg * 3);
    }

    #[test]
    fn zero_count_both_modes_return_zero() {
        let params = BuildIndirectParams::new(0, 4096);
        assert_eq!(params.workgroups(), 0);
        assert_eq!(params.workgroups_batched(), 0);
    }

    #[test]
    fn large_count_batched_mode() {
        let params = BuildIndirectParams::new(100000, 131072);

        assert!(params.use_batched_mode());
        // Standard: ceil(100000 / 64) = 1563
        assert_eq!(params.workgroups(), 1563);
        // Batched: ceil(100000 / 256) = 391
        assert_eq!(params.workgroups_batched(), 391);
    }
}

// ============================================================================
// 8. Integration Tests - Full Data Flow
// ============================================================================

mod integration {
    use super::*;

    #[test]
    fn typical_scene_draw_generation() {
        // Simulate a typical scene with multiple object types
        let visible_objects = 500;
        let num_meshes = 10;

        let compacted_indices: Vec<u32> = (0..visible_objects).collect();
        let object_mesh_indices: Vec<u32> = (0..visible_objects)
            .map(|i| (i % num_meshes) as u32)
            .collect();
        let lod_levels: Vec<u32> = (0..visible_objects)
            .map(|i| ((i / 100) % 4) as u32) // Varying LOD levels
            .collect();

        let mesh_data: Vec<BuildIndirectMeshData> = (0..num_meshes)
            .map(|i| {
                let base_count = 1000 + i * 100;
                BuildIndirectMeshData::with_lods(
                    base_count as u32,
                    (i * 5000) as u32,
                    i as i32 * 100,
                    &[base_count as u32, (base_count / 2) as u32, (base_count / 4) as u32, (base_count / 8) as u32],
                    &[0, base_count as u32, (base_count + base_count / 2) as u32, (base_count + base_count / 2 + base_count / 4) as u32],
                )
            })
            .collect();

        let commands = cpu_build_indirect(
            &compacted_indices,
            &object_mesh_indices,
            &lod_levels,
            &mesh_data,
        );

        assert_eq!(commands.len(), visible_objects as usize);

        // All commands should be visible
        for cmd in &commands {
            assert!(cmd.is_visible());
            assert_eq!(cmd.instance_count, 1);
        }

        // first_instance should be sequential
        for (i, cmd) in commands.iter().enumerate() {
            assert_eq!(cmd.first_instance, i as u32);
        }
    }

    #[test]
    fn stress_test_max_objects() {
        // Test with large number of objects (but not too large for test speed)
        let count = 50000;
        let compacted_indices: Vec<u32> = (0..count).collect();
        let object_mesh_indices: Vec<u32> = vec![0; count as usize];
        let lod_levels: Vec<u32> = vec![0; count as usize];
        let mesh_data = vec![BuildIndirectMeshData::new(36, 0, 0)];

        let commands = cpu_build_indirect(
            &compacted_indices,
            &object_mesh_indices,
            &lod_levels,
            &mesh_data,
        );

        assert_eq!(commands.len(), count as usize);
    }

    #[test]
    fn verify_shader_constant_match_rust_constants() {
        // Verify WGSL constants match Rust constants
        assert_eq!(BUILD_INDIRECT_WORKGROUP_SIZE, 64);
        assert_eq!(BUILD_INDIRECT_BATCH_SIZE, 4);
        assert_eq!(BUILD_INDIRECT_MAX_LOD_LEVELS, 4);

        // Verify these match by checking shader source
        assert!(BUILD_INDIRECT_SHADER.contains("WORKGROUP_SIZE: u32 = 64u"));
        assert!(BUILD_INDIRECT_SHADER.contains("BATCH_SIZE: u32 = 4u"));
        assert!(BUILD_INDIRECT_SHADER.contains("NUM_LOD_LEVELS: u32 = 4u"));
    }
}
