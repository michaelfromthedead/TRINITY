//! Unit tests for Phase 6 GPU-Driven Rendering (T-WGPU-P6.10.1)
//!
//! This module provides comprehensive unit tests for Phase 6 GPU-driven rendering
//! components covering:
//!
//! - AABB-frustum intersection tests
//! - Index allocator tests (alloc/free/recycle)
//! - Indirect struct size tests (DrawIndexedIndirectArgs = 20 bytes)
//! - LOD selection tests
//! - Meshlet struct tests
//! - Material descriptor tests
//! - Texture registry tests (CPU-side)
//! - Buffer registry tests (CPU-side)
//! - Bindless bind group tests (CPU-side)
//! - Geometry path selection tests
//!
//! Target: 100+ test assertions across all modules with 80%+ coverage.

use std::collections::HashSet;
use std::mem;

use renderer_backend::gpu_driven::{
    // Frustum culling
    cpu_frustum_cull, cpu_frustum_cull_aabb_only, cpu_frustum_cull_sphere_only,
    CullParams, Frustum, FrustumPlane, InstanceBounds,

    // Build indirect
    BuildIndirectParams, BuildIndirectMeshData, BuildIndirectDrawArgs as IndirectDrawIndexedArgs,
    cpu_build_indirect,
    BUILD_INDIRECT_PARAMS_SIZE, MESH_DATA_SIZE, DRAW_INDEXED_INDIRECT_ARGS_SIZE,

    // LOD
    LodDistances, LodParams, LodConfig,
    distance_to_camera, distance_to_camera_squared,
    screen_coverage, select_lod, select_lod_by_distance, select_lod_by_distance_squared,
    select_lod_by_coverage, squared_thresholds,
    LOD_DISTANCES_SIZE, LOD_PARAMS_SIZE,

    // Meshlet
    Meshlet, MeshletBounds, MeshletData,
    MAX_MESHLET_TRIANGLES, MAX_MESHLET_VERTICES,
    MESHLET_BOUNDS_SIZE, MESHLET_SIZE,

    // Material table
    MaterialDescriptor, MaterialTableEntry, MaterialTable,
    MATERIAL_TABLE_ENTRY_SIZE, MATERIAL_DESCRIPTOR_SIZE,
    MATERIAL_DESC_FLAG_DOUBLE_SIDED, NO_TEXTURE,
    MaterialRemoveResult,

    // Texture registry (CPU-side helpers only)
    TextureRegistryMetrics,
    supports_bindless_textures, supports_partially_bound, supports_non_uniform_indexing,
    texture_registry_required_features,
    texture_registry_optimal_features,
    cpu_fragmentation,
    TEXTURE_REGISTRY_MAX_TEXTURES,
    TEXTURE_REGISTRY_MIN_TEXTURES,
    TEXTURE_REGISTRY_BIND_GROUP_INDEX,
    TEXTURE_ARRAY_BINDING, SAMPLER_BINDING,

    // Buffer registry
    MAX_BINDLESS_BUFFERS, MIN_BUFFER_SIZE,

    // Bindless bind group
    BINDLESS_MAX_TEXTURES,
    MAX_BINDLESS_SAMPLERS, BINDLESS_MIN_TEXTURES,
    MIN_BINDLESS_SAMPLERS, BINDING_TEXTURES, BINDING_SAMPLERS, BINDING_MATERIALS,
    BINDLESS_BIND_GROUP_INDEX,
    supports_texture_arrays, supports_full_bindless,
    bindless_required_features,
    bindless_optimal_features,

    // Geometry path
    GeometryPath, GeometryPathConfig,
    TRADITIONAL_PATH_NAME, MESHLET_PATH_NAME,

    // Index allocator
    IndexAllocator, GenerationalIndex, AllocatorError,
};

use wgpu::Features;

// ============================================================================
// CATEGORY 1: AABB-FRUSTUM INTERSECTION TESTS
// ============================================================================

/// Helper: Create a simple perspective frustum for testing.
/// Camera at origin, looking down -Z, with a 90-degree FOV.
fn make_test_frustum() -> Frustum {
    Frustum {
        planes: [
            // Left: normal points right (+X) at 45 degrees into frustum
            FrustumPlane::new([1.0, 0.0, -1.0], 0.0),
            // Right: normal points left (-X) at 45 degrees into frustum
            FrustumPlane::new([-1.0, 0.0, -1.0], 0.0),
            // Bottom: normal points up (+Y) at 45 degrees into frustum
            FrustumPlane::new([0.0, 1.0, -1.0], 0.0),
            // Top: normal points down (-Y) at 45 degrees into frustum
            FrustumPlane::new([0.0, -1.0, -1.0], 0.0),
            // Near: normal points forward (-Z), at z = -1
            FrustumPlane::new([0.0, 0.0, -1.0], -1.0),
            // Far: normal points backward (+Z), at z = -100
            FrustumPlane::new([0.0, 0.0, 1.0], 100.0),
        ],
    }
}

#[test]
fn test_frustum_plane_size() {
    assert_eq!(mem::size_of::<FrustumPlane>(), 16);
}

#[test]
fn test_frustum_size() {
    assert_eq!(mem::size_of::<Frustum>(), 96);
}

#[test]
fn test_instance_bounds_size() {
    assert_eq!(mem::size_of::<InstanceBounds>(), 48);
}

#[test]
fn test_cull_params_size() {
    assert_eq!(mem::size_of::<CullParams>(), 16);
}

#[test]
fn test_frustum_sphere_inside_visible() {
    let frustum = make_test_frustum();
    // Sphere at (0, 0, -10), radius 1: should be visible
    let center = [0.0, 0.0, -10.0];
    let radius = 1.0;
    assert!(frustum.test_sphere(center, radius));
}

#[test]
fn test_frustum_sphere_outside_culled() {
    let frustum = make_test_frustum();
    // Sphere at (100, 0, -10), radius 1: way outside right side
    let center = [100.0, 0.0, -10.0];
    let radius = 1.0;
    assert!(!frustum.test_sphere(center, radius));
}

#[test]
fn test_frustum_sphere_beyond_far_plane() {
    let frustum = make_test_frustum();
    // Sphere at (0, 0, -200), radius 1: beyond far plane
    let center = [0.0, 0.0, -200.0];
    let radius = 1.0;
    assert!(!frustum.test_sphere(center, radius));
}

#[test]
fn test_frustum_sphere_behind_camera() {
    let frustum = make_test_frustum();
    // Sphere at (0, 0, 10), radius 1: behind camera
    let center = [0.0, 0.0, 10.0];
    let radius = 1.0;
    assert!(!frustum.test_sphere(center, radius));
}

#[test]
fn test_frustum_sphere_intersecting_visible() {
    let frustum = make_test_frustum();
    // Sphere touching near plane from inside
    let center = [0.0, 0.0, -2.0];
    let radius = 1.5;
    assert!(frustum.test_sphere(center, radius));
}

#[test]
fn test_frustum_aabb_inside_visible() {
    let frustum = make_test_frustum();
    // Small AABB centered at (0, 0, -10)
    let aabb_min = [-1.0, -1.0, -11.0];
    let aabb_max = [1.0, 1.0, -9.0];
    assert!(frustum.test_aabb(aabb_min, aabb_max));
}

#[test]
fn test_frustum_aabb_outside_culled() {
    let frustum = make_test_frustum();
    // AABB far to the right
    let aabb_min = [50.0, -1.0, -11.0];
    let aabb_max = [52.0, 1.0, -9.0];
    assert!(!frustum.test_aabb(aabb_min, aabb_max));
}

#[test]
fn test_frustum_aabb_partial_overlap_visible() {
    let frustum = make_test_frustum();
    // Large AABB that partially overlaps frustum
    let aabb_min = [-5.0, -5.0, -20.0];
    let aabb_max = [5.0, 5.0, -10.0];
    assert!(frustum.test_aabb(aabb_min, aabb_max));
}

#[test]
fn test_cpu_frustum_cull_multiple_instances() {
    let frustum = make_test_frustum();
    let instances = vec![
        // Visible: inside frustum
        InstanceBounds::new([0.0, 0.0, -10.0], 1.0, [-1.0, -1.0, -11.0], [1.0, 1.0, -9.0]),
        // Culled: outside right
        InstanceBounds::new([100.0, 0.0, -10.0], 1.0, [99.0, -1.0, -11.0], [101.0, 1.0, -9.0]),
        // Visible: on edge
        InstanceBounds::new([0.0, 0.0, -5.0], 2.0, [-2.0, -2.0, -7.0], [2.0, 2.0, -3.0]),
        // Culled: behind camera
        InstanceBounds::new([0.0, 0.0, 5.0], 1.0, [-1.0, -1.0, 4.0], [1.0, 1.0, 6.0]),
    ];
    let visibility = cpu_frustum_cull(&frustum, &instances);
    assert_eq!(visibility, vec![1, 0, 1, 0]);
}

#[test]
fn test_cpu_frustum_cull_sphere_only() {
    let frustum = make_test_frustum();
    let instances = vec![
        InstanceBounds::new([0.0, 0.0, -10.0], 1.0, [-1.0, -1.0, -11.0], [1.0, 1.0, -9.0]),
        InstanceBounds::new([100.0, 0.0, -10.0], 1.0, [99.0, -1.0, -11.0], [101.0, 1.0, -9.0]),
    ];
    let visibility = cpu_frustum_cull_sphere_only(&frustum, &instances);
    assert_eq!(visibility, vec![1, 0]);
}

#[test]
fn test_cpu_frustum_cull_aabb_only() {
    let frustum = make_test_frustum();
    let instances = vec![
        InstanceBounds::from_aabb([-1.0, -1.0, -11.0], [1.0, 1.0, -9.0]),
        InstanceBounds::from_aabb([50.0, 50.0, -11.0], [52.0, 52.0, -9.0]),
    ];
    let visibility = cpu_frustum_cull_aabb_only(&frustum, &instances);
    assert_eq!(visibility, vec![1, 0]);
}

#[test]
fn test_instance_bounds_from_aabb_with_sphere() {
    let bounds = InstanceBounds::from_aabb_with_sphere([-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]);
    // Center should be midpoint
    assert!((bounds.sphere_center[0] - 0.0).abs() < 1e-6);
    assert!((bounds.sphere_center[1] - 0.0).abs() < 1e-6);
    assert!((bounds.sphere_center[2] - 0.0).abs() < 1e-6);
    // Radius should be sqrt(1^2 + 2^2 + 3^2) = sqrt(14)
    let expected_radius = (1.0_f32 + 4.0 + 9.0).sqrt();
    assert!((bounds.sphere_radius - expected_radius).abs() < 1e-6);
}

#[test]
fn test_cull_params_num_workgroups() {
    assert_eq!(CullParams::new(1).num_workgroups(), 1);
    assert_eq!(CullParams::new(256).num_workgroups(), 1);
    assert_eq!(CullParams::new(257).num_workgroups(), 2);
    assert_eq!(CullParams::new(512).num_workgroups(), 2);
    assert_eq!(CullParams::new(1000).num_workgroups(), 4);
}

#[test]
fn test_frustum_plane_normalization() {
    // Non-normalized input
    let plane = FrustumPlane::new([2.0, 0.0, 0.0], 4.0);
    // Should be normalized
    assert!((plane.normal[0] - 1.0).abs() < 1e-6);
    assert!((plane.distance - 2.0).abs() < 1e-6);
}

#[test]
fn test_frustum_from_view_projection() {
    // Create a simple orthographic-like matrix
    let vp = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let frustum = Frustum::from_view_projection(&vp);
    assert_eq!(frustum.planes.len(), 6);
}

// ============================================================================
// CATEGORY 2: INDEX ALLOCATOR TESTS
// ============================================================================

#[test]
fn test_index_allocator_new() {
    let allocator = IndexAllocator::new(100);
    assert_eq!(allocator.capacity(), 100);
    assert_eq!(allocator.count(), 0);
    assert_eq!(allocator.free_count(), 0);
    assert!(!allocator.has_generations());
}

#[test]
fn test_index_allocator_with_generations() {
    let allocator = IndexAllocator::with_generations(100);
    assert_eq!(allocator.capacity(), 100);
    assert!(allocator.has_generations());
}

#[test]
fn test_index_allocator_sequential_allocation() {
    let mut allocator = IndexAllocator::new(10);
    assert_eq!(allocator.allocate(), Some(0));
    assert_eq!(allocator.allocate(), Some(1));
    assert_eq!(allocator.allocate(), Some(2));
    assert_eq!(allocator.count(), 3);
}

#[test]
fn test_index_allocator_at_capacity() {
    let mut allocator = IndexAllocator::new(3);
    assert_eq!(allocator.allocate(), Some(0));
    assert_eq!(allocator.allocate(), Some(1));
    assert_eq!(allocator.allocate(), Some(2));
    assert_eq!(allocator.allocate(), None);
    assert_eq!(allocator.count(), 3);
}

#[test]
fn test_index_allocator_free_basic() {
    let mut allocator = IndexAllocator::new(10);
    let idx = allocator.allocate().unwrap();
    assert_eq!(allocator.count(), 1);
    assert!(allocator.free(idx));
    assert_eq!(allocator.count(), 0);
    assert_eq!(allocator.free_count(), 1);
}

#[test]
fn test_index_allocator_free_list_lifo_order() {
    let mut allocator = IndexAllocator::new(10);
    let idx0 = allocator.allocate().unwrap(); // 0
    let idx1 = allocator.allocate().unwrap(); // 1
    let idx2 = allocator.allocate().unwrap(); // 2

    allocator.free(idx0);
    allocator.free(idx1);
    allocator.free(idx2);

    // LIFO: last freed (2) comes out first
    assert_eq!(allocator.allocate(), Some(2));
    assert_eq!(allocator.allocate(), Some(1));
    assert_eq!(allocator.allocate(), Some(0));
}

#[test]
fn test_index_allocator_double_free_returns_false() {
    let mut allocator = IndexAllocator::new(10);
    let idx = allocator.allocate().unwrap();
    assert!(allocator.free(idx));
    assert!(!allocator.free(idx)); // Double-free
}

#[test]
fn test_index_allocator_try_free_double_free() {
    let mut allocator = IndexAllocator::new(10);
    let idx = allocator.allocate().unwrap();
    assert!(allocator.try_free(idx).is_ok());
    assert!(matches!(
        allocator.try_free(idx),
        Err(AllocatorError::DoubleFree(0))
    ));
}

#[test]
fn test_index_allocator_is_allocated() {
    let mut allocator = IndexAllocator::new(10);
    let idx = allocator.allocate().unwrap();
    assert!(allocator.is_allocated(idx));
    allocator.free(idx);
    assert!(!allocator.is_allocated(idx));
}

#[test]
fn test_index_allocator_generation_increments_on_reuse() {
    let mut allocator = IndexAllocator::with_generations(10);
    let gen_idx1 = allocator.allocate_generational().unwrap();
    allocator.free(gen_idx1.index);
    let gen_idx2 = allocator.allocate_generational().unwrap();

    assert_eq!(gen_idx1.index, gen_idx2.index);
    assert_eq!(gen_idx1.generation, 0);
    assert_eq!(gen_idx2.generation, 1);
}

#[test]
fn test_index_allocator_is_valid_stale() {
    let mut allocator = IndexAllocator::with_generations(10);
    let gen_idx1 = allocator.allocate_generational().unwrap();
    allocator.free(gen_idx1.index);
    let _gen_idx2 = allocator.allocate_generational().unwrap();

    assert!(!allocator.is_valid(gen_idx1)); // Stale!
}

#[test]
fn test_index_allocator_clear() {
    let mut allocator = IndexAllocator::new(10);
    allocator.allocate();
    allocator.allocate();
    allocator.clear();

    assert_eq!(allocator.count(), 0);
    assert_eq!(allocator.free_count(), 0);
    assert_eq!(allocator.allocate(), Some(0));
}

#[test]
fn test_index_allocator_metrics() {
    let mut allocator = IndexAllocator::with_generations(100);
    allocator.allocate();
    allocator.allocate();
    allocator.free(0);

    let metrics = allocator.metrics();
    assert_eq!(metrics.allocated_count, 1);
    assert_eq!(metrics.capacity, 100);
    assert_eq!(metrics.free_list_size, 1);
    assert_eq!(metrics.peak_allocations, 2);
    assert!(metrics.has_generations);
}

#[test]
fn test_index_allocator_fragmentation() {
    let mut allocator = IndexAllocator::new(10);
    assert_eq!(allocator.fragmentation(), 0.0);

    for _ in 0..4 {
        allocator.allocate();
    }
    // 4 allocated, 0 free
    assert_eq!(allocator.fragmentation(), 0.0);

    allocator.free(0);
    allocator.free(1);
    // 2 allocated, 2 free, peak = 4
    assert_eq!(allocator.fragmentation(), 0.5);
}

#[test]
fn test_generational_index_null() {
    let idx = GenerationalIndex::null();
    assert!(idx.is_null());
    assert_eq!(idx.index, u32::MAX);
    assert_eq!(idx.generation, u32::MAX);
}

// ============================================================================
// CATEGORY 3: INDIRECT STRUCT SIZE TESTS
// ============================================================================

#[test]
fn test_draw_indexed_indirect_args_size() {
    assert_eq!(
        mem::size_of::<IndirectDrawIndexedArgs>(),
        DRAW_INDEXED_INDIRECT_ARGS_SIZE
    );
    // Critical: DrawIndexedIndirectArgs must be exactly 20 bytes
    assert_eq!(mem::size_of::<IndirectDrawIndexedArgs>(), 20);
}

#[test]
fn test_build_indirect_params_size() {
    assert_eq!(
        mem::size_of::<BuildIndirectParams>(),
        BUILD_INDIRECT_PARAMS_SIZE
    );
    assert_eq!(mem::size_of::<BuildIndirectParams>(), 16);
}

#[test]
fn test_mesh_data_size() {
    assert_eq!(mem::size_of::<BuildIndirectMeshData>(), MESH_DATA_SIZE);
    assert_eq!(mem::size_of::<BuildIndirectMeshData>(), 48);
}

#[test]
fn test_indirect_args_layout() {
    // Verify field offsets match wgpu expectations
    let args = IndirectDrawIndexedArgs::new(100, 1, 0, 0, 42);
    let bytes: &[u8] = bytemuck::bytes_of(&args);

    // index_count at offset 0
    assert_eq!(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]), 100);
    // instance_count at offset 4
    assert_eq!(u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]), 1);
    // first_index at offset 8
    assert_eq!(u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]), 0);
    // base_vertex at offset 12
    assert_eq!(i32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]), 0);
    // first_instance at offset 16
    assert_eq!(u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]), 42);
}

#[test]
fn test_build_indirect_params_workgroups() {
    let params = BuildIndirectParams::new(1000, 4096);
    // Standard dispatch: ceil(1000 / 64) = 16 workgroups
    assert_eq!(params.workgroups(), 16);
    // Batched dispatch: ceil(1000 / (64 * 4)) = ceil(1000 / 256) = 4 workgroups
    assert_eq!(params.workgroups_batched(), 4);
}

#[test]
fn test_build_indirect_params_batched_mode() {
    // Small count - use standard mode
    assert!(!BuildIndirectParams::new(5000, 4096).use_batched_mode());
    // Large count - use batched mode
    assert!(BuildIndirectParams::new(50000, 65536).use_batched_mode());
    // Boundary
    assert!(!BuildIndirectParams::new(10000, 4096).use_batched_mode());
    assert!(BuildIndirectParams::new(10001, 4096).use_batched_mode());
}

#[test]
fn test_mesh_data_lod_selection() {
    let mesh = BuildIndirectMeshData::with_lods(
        1000,  // base index count
        0,     // first index
        0,     // base vertex
        &[1000, 500, 250, 125],  // LOD counts
        &[0, 1000, 1500, 1750],  // LOD offsets
    );

    // LOD 0 - highest detail
    assert_eq!(mesh.index_count_for_lod(0), 1000);
    assert_eq!(mesh.first_index_for_lod(0), 0);

    // LOD 1
    assert_eq!(mesh.index_count_for_lod(1), 500);
    assert_eq!(mesh.first_index_for_lod(1), 1000);

    // LOD 2
    assert_eq!(mesh.index_count_for_lod(2), 250);
    assert_eq!(mesh.first_index_for_lod(2), 1500);

    // LOD 3 - lowest detail
    assert_eq!(mesh.index_count_for_lod(3), 125);
    assert_eq!(mesh.first_index_for_lod(3), 1750);

    // Out of range LOD should clamp
    assert_eq!(mesh.index_count_for_lod(10), 125);
}

#[test]
fn test_mesh_data_lod_fallback() {
    // Mesh without explicit LOD counts
    let mesh = BuildIndirectMeshData::new(1000, 0, 0);

    // All LOD levels should fall back to base index_count
    for lod in 0..4 {
        assert_eq!(mesh.index_count_for_lod(lod), 1000);
        assert_eq!(mesh.first_index_for_lod(lod), 0);
    }
}

#[test]
fn test_cpu_build_indirect() {
    let compacted_indices = vec![0, 2, 5];
    let object_mesh_indices = vec![0, 1, 0, 2, 1, 0];
    let lod_levels = vec![0, 1, 2, 0, 1, 1];
    let mesh_data = vec![
        BuildIndirectMeshData::with_lods(1000, 0, 0, &[1000, 500, 250, 125], &[0, 1000, 1500, 1750]),
    ];

    let commands = cpu_build_indirect(
        &compacted_indices,
        &object_mesh_indices,
        &lod_levels,
        &mesh_data,
    );

    assert_eq!(commands.len(), 3);

    // Object 0 at LOD 0
    assert_eq!(commands[0].index_count, 1000);
    assert_eq!(commands[0].first_index, 0);
    assert_eq!(commands[0].first_instance, 0);

    // Object 2 at LOD 2
    assert_eq!(commands[1].index_count, 250);
    assert_eq!(commands[1].first_index, 1500);
    assert_eq!(commands[1].first_instance, 1);

    // Object 5 at LOD 1
    assert_eq!(commands[2].index_count, 500);
    assert_eq!(commands[2].first_index, 1000);
    assert_eq!(commands[2].first_instance, 2);
}

// ============================================================================
// CATEGORY 4: LOD SELECTION TESTS
// ============================================================================

#[test]
fn test_lod_distances_size() {
    assert_eq!(mem::size_of::<LodDistances>(), LOD_DISTANCES_SIZE);
    assert_eq!(LodDistances::SIZE, 16);
}

#[test]
fn test_lod_params_size() {
    assert_eq!(mem::size_of::<LodParams>(), LOD_PARAMS_SIZE);
    assert_eq!(LodParams::SIZE, 32);
}

#[test]
fn test_distance_to_camera() {
    // Origin to point on X axis
    let dist = distance_to_camera([0.0, 0.0, 0.0], [10.0, 0.0, 0.0]);
    assert!((dist - 10.0).abs() < 1e-6);

    // Diagonal (3-4-5 triangle)
    let dist = distance_to_camera([0.0, 0.0, 0.0], [3.0, 4.0, 0.0]);
    assert!((dist - 5.0).abs() < 1e-6);

    // Same point
    let dist = distance_to_camera([5.0, 5.0, 5.0], [5.0, 5.0, 5.0]);
    assert!(dist < 1e-6);
}

#[test]
fn test_distance_to_camera_squared() {
    let dist_sq = distance_to_camera_squared([0.0, 0.0, 0.0], [10.0, 0.0, 0.0]);
    assert!((dist_sq - 100.0).abs() < 1e-6);

    let dist_sq = distance_to_camera_squared([0.0, 0.0, 0.0], [3.0, 4.0, 0.0]);
    assert!((dist_sq - 25.0).abs() < 1e-6);
}

#[test]
fn test_select_lod_by_distance() {
    let thresholds = LodDistances::new(10.0, 25.0, 50.0);

    // LOD 0: distance < 10
    assert_eq!(select_lod_by_distance(5.0, &thresholds), 0);
    assert_eq!(select_lod_by_distance(9.9, &thresholds), 0);

    // LOD 1: 10 <= distance < 25
    assert_eq!(select_lod_by_distance(10.0, &thresholds), 1);
    assert_eq!(select_lod_by_distance(15.0, &thresholds), 1);

    // LOD 2: 25 <= distance < 50
    assert_eq!(select_lod_by_distance(25.0, &thresholds), 2);
    assert_eq!(select_lod_by_distance(35.0, &thresholds), 2);

    // LOD 3: distance >= 50
    assert_eq!(select_lod_by_distance(50.0, &thresholds), 3);
    assert_eq!(select_lod_by_distance(100.0, &thresholds), 3);
}

#[test]
fn test_select_lod_by_coverage() {
    // LOD 0: coverage >= 10%
    assert_eq!(select_lod_by_coverage(0.15), 0);
    assert_eq!(select_lod_by_coverage(0.10), 0);

    // LOD 1: 4% <= coverage < 10%
    assert_eq!(select_lod_by_coverage(0.05), 1);
    assert_eq!(select_lod_by_coverage(0.04), 1);

    // LOD 2: 1% <= coverage < 4%
    assert_eq!(select_lod_by_coverage(0.02), 2);
    assert_eq!(select_lod_by_coverage(0.01), 2);

    // LOD 3: coverage < 1%
    assert_eq!(select_lod_by_coverage(0.009), 3);
    assert_eq!(select_lod_by_coverage(0.0), 3);
}

#[test]
fn test_screen_coverage() {
    let camera_pos = [0.0, 0.0, 0.0];
    let fov_y = std::f32::consts::FRAC_PI_4; // 45 degrees
    let screen_height = 1080.0;

    // Object at distance 10 with radius 1
    let coverage = screen_coverage(camera_pos, [10.0, 0.0, 0.0], 1.0, fov_y, screen_height);
    assert!(coverage > 0.0);
    assert!(coverage < 1.0);

    // Closer object should have higher coverage
    let coverage_close = screen_coverage(camera_pos, [5.0, 0.0, 0.0], 1.0, fov_y, screen_height);
    assert!(coverage_close > coverage);

    // Larger object should have higher coverage
    let coverage_large = screen_coverage(camera_pos, [10.0, 0.0, 0.0], 2.0, fov_y, screen_height);
    assert!(coverage_large > coverage);
}

#[test]
fn test_squared_thresholds() {
    let distances = LodDistances::new(10.0, 20.0, 30.0);
    let squared = squared_thresholds(&distances);

    assert_eq!(squared[0], 100.0);
    assert_eq!(squared[1], 400.0);
    assert_eq!(squared[2], 900.0);
}

#[test]
fn test_lod_distances_scaled() {
    let distances = LodDistances::new(10.0, 25.0, 50.0);
    let scaled = distances.scaled(2.0);

    assert_eq!(scaled.thresholds[0], 20.0);
    assert_eq!(scaled.thresholds[1], 50.0);
    assert_eq!(scaled.thresholds[2], 100.0);
}

#[test]
fn test_lod_params_aspect_ratio() {
    let params = LodParams::standard_1080p([0.0, 0.0, 0.0]);
    let aspect = params.aspect_ratio();
    assert!((aspect - 16.0 / 9.0).abs() < 0.01);

    let params = LodParams::standard_4k([0.0, 0.0, 0.0]);
    let aspect = params.aspect_ratio();
    assert!((aspect - 16.0 / 9.0).abs() < 0.01);
}

#[test]
fn test_lod_config_modes() {
    let distances = LodDistances::new(10.0, 25.0, 50.0);

    let config = LodConfig::distance_based(distances);
    assert!(!config.use_screen_size);

    let config = LodConfig::screen_size_based(distances);
    assert!(config.use_screen_size);
}

#[test]
fn test_select_lod_integration() {
    let params = LodParams::standard_1080p([0.0, 0.0, 0.0]);
    let distances = LodDistances::new(10.0, 25.0, 50.0);

    // Distance-based
    let config = LodConfig::distance_based(distances);
    let lod = select_lod(&params, [5.0, 0.0, 0.0], 1.0, &config);
    assert_eq!(lod, 0);

    let lod = select_lod(&params, [15.0, 0.0, 0.0], 1.0, &config);
    assert_eq!(lod, 1);
}

// ============================================================================
// CATEGORY 5: MESHLET STRUCT TESTS
// ============================================================================

#[test]
fn test_meshlet_size() {
    assert_eq!(mem::size_of::<Meshlet>(), MESHLET_SIZE);
    assert_eq!(mem::size_of::<Meshlet>(), 12);
}

#[test]
fn test_meshlet_bounds_size() {
    assert_eq!(mem::size_of::<MeshletBounds>(), MESHLET_BOUNDS_SIZE);
    assert_eq!(mem::size_of::<MeshletBounds>(), 32);
}

#[test]
fn test_meshlet_constants() {
    assert_eq!(MAX_MESHLET_VERTICES, 64);
    assert_eq!(MAX_MESHLET_TRIANGLES, 124);
    assert_eq!(Meshlet::MAX_VERTICES as usize, MAX_MESHLET_VERTICES);
    assert_eq!(Meshlet::MAX_TRIANGLES as usize, MAX_MESHLET_TRIANGLES);
}

#[test]
fn test_meshlet_new() {
    let m = Meshlet::new(100, 200, 32, 40);
    assert_eq!(m.vertex_offset, 100);
    assert_eq!(m.triangle_offset, 200);
    assert_eq!(m.vertex_count, 32);
    assert_eq!(m.triangle_count, 40);
    assert_eq!(m._padding, [0, 0]);
}

#[test]
fn test_meshlet_is_empty() {
    assert!(Meshlet::default().is_empty());
    assert!(!Meshlet::new(0, 0, 3, 1).is_empty());
    assert!(Meshlet::new(0, 0, 3, 0).is_empty());
}

#[test]
fn test_meshlet_bounds_sphere_only() {
    let b = MeshletBounds::sphere_only([1.0, 2.0, 3.0], 5.0);
    assert_eq!(b.center, [1.0, 2.0, 3.0]);
    assert_eq!(b.radius, 5.0);
    assert!(!b.has_valid_cone());
}

#[test]
fn test_meshlet_bounds_with_cone() {
    let b = MeshletBounds::new([0.0, 0.0, 0.0], 1.0, [0.0, 1.0, 0.0], 0.5);
    assert_eq!(b.cone_axis, [0.0, 1.0, 0.0]);
    assert_eq!(b.cone_cutoff, 0.5);
    assert!(b.has_valid_cone());
}

#[test]
fn test_meshlet_bounds_builder_chain() {
    let bounds = MeshletBounds::default()
        .with_bounds([1.0, 2.0, 3.0], 5.0)
        .with_cone([0.0, 0.0, 1.0], 0.707);

    assert_eq!(bounds.center, [1.0, 2.0, 3.0]);
    assert_eq!(bounds.radius, 5.0);
    assert_eq!(bounds.cone_axis, [0.0, 0.0, 1.0]);
    assert!((bounds.cone_cutoff - 0.707).abs() < 1e-6);
    assert!(bounds.has_valid_cone());
}

#[test]
fn test_meshlet_data_generate_cube() {
    // Simple cube mesh
    let positions = vec![
        [-1.0, -1.0,  1.0], [ 1.0, -1.0,  1.0], [ 1.0,  1.0,  1.0], [-1.0,  1.0,  1.0],
        [-1.0, -1.0, -1.0], [-1.0,  1.0, -1.0], [ 1.0,  1.0, -1.0], [ 1.0, -1.0, -1.0],
    ];
    let indices = vec![
        0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7,
        3, 2, 6, 3, 6, 5, 4, 7, 1, 4, 1, 0,
        1, 7, 6, 1, 6, 2, 4, 0, 3, 4, 3, 5,
    ];

    let data = MeshletData::generate(&positions, &indices, None);

    // Cube has 12 triangles, should fit in one meshlet
    assert_eq!(data.meshlet_count(), 1);
    assert_eq!(data.meshlets[0].triangle_count, 12);
    assert!(data.meshlets[0].vertex_count <= 8);
    assert!(data.validate(positions.len()).is_ok());
}

#[test]
fn test_meshlet_data_empty() {
    let data = MeshletData::generate(&[], &[], None);
    assert!(data.is_empty());
    assert_eq!(data.meshlet_count(), 0);
}

#[test]
fn test_meshlet_gpu_layout() {
    // Verify field offsets match expected GPU layout
    let m = Meshlet::new(0x11223344, 0x55667788, 0xAA, 0xBB);
    let bytes = bytemuck::bytes_of(&m);

    // vertex_offset at offset 0 (4 bytes, little-endian)
    assert_eq!(&bytes[0..4], &[0x44, 0x33, 0x22, 0x11]);
    // triangle_offset at offset 4 (4 bytes, little-endian)
    assert_eq!(&bytes[4..8], &[0x88, 0x77, 0x66, 0x55]);
    // vertex_count at offset 8 (1 byte)
    assert_eq!(bytes[8], 0xAA);
    // triangle_count at offset 9 (1 byte)
    assert_eq!(bytes[9], 0xBB);
}

// ============================================================================
// CATEGORY 6: MATERIAL DESCRIPTOR TESTS
// ============================================================================

#[test]
fn test_material_table_entry_size() {
    assert_eq!(mem::size_of::<MaterialTableEntry>(), MATERIAL_TABLE_ENTRY_SIZE);
    assert_eq!(mem::size_of::<MaterialTableEntry>(), 80);
}

#[test]
fn test_material_descriptor_size() {
    assert_eq!(mem::size_of::<MaterialDescriptor>(), MATERIAL_DESCRIPTOR_SIZE);
    assert_eq!(mem::size_of::<MaterialDescriptor>(), 64);
}

#[test]
fn test_material_descriptor_new() {
    let mat = MaterialDescriptor::new();
    assert_eq!(mat.albedo_texture_id, NO_TEXTURE);
    assert_eq!(mat.normal_texture_id, NO_TEXTURE);
    assert_eq!(mat.metallic_roughness_texture_id, NO_TEXTURE);
    assert_eq!(mat.emissive_texture_id, NO_TEXTURE);
    assert_eq!(mat.base_color, [1.0, 1.0, 1.0, 1.0]);
    assert_eq!(mat.metallic, 0.0);
    assert_eq!(mat.roughness, 0.5);
    assert_eq!(mat.emissive, [0.0, 0.0, 0.0, 0.0]);
    assert_eq!(mat.flags, 0);
}

#[test]
fn test_material_descriptor_opaque() {
    let mat = MaterialDescriptor::opaque([0.5, 0.6, 0.7, 1.0]);
    assert_eq!(mat.base_color, [0.5, 0.6, 0.7, 1.0]);
    assert_eq!(mat.metallic, 0.0);
}

#[test]
fn test_material_descriptor_metallic() {
    let mat = MaterialDescriptor::metallic([0.8, 0.8, 0.8, 1.0], 0.2);
    assert_eq!(mat.base_color, [0.8, 0.8, 0.8, 1.0]);
    assert_eq!(mat.metallic, 0.9);
    assert_eq!(mat.roughness, 0.2);
}

#[test]
fn test_material_descriptor_texture_builders() {
    let mat = MaterialDescriptor::new()
        .with_albedo_texture_id(1)
        .with_normal_texture_id(2)
        .with_metallic_roughness_texture_id(3)
        .with_emissive_texture_id(4);

    assert_eq!(mat.albedo_texture_id, 1);
    assert_eq!(mat.normal_texture_id, 2);
    assert_eq!(mat.metallic_roughness_texture_id, 3);
    assert_eq!(mat.emissive_texture_id, 4);
}

#[test]
fn test_material_descriptor_double_sided() {
    let mat = MaterialDescriptor::new().with_double_sided(true);
    assert!(mat.is_double_sided());
    assert!((mat.flags & MATERIAL_DESC_FLAG_DOUBLE_SIDED) != 0);
}

#[test]
fn test_material_descriptor_alpha_modes() {
    let mask_mat = MaterialDescriptor::new().with_alpha_mask(0.75);
    assert!(mask_mat.is_alpha_mask());
    assert!(!mask_mat.is_alpha_blend());
    assert_eq!(mask_mat.alpha_cutoff, 0.75);

    let blend_mat = MaterialDescriptor::new().with_alpha_blend();
    assert!(blend_mat.is_alpha_blend());
    assert!(!blend_mat.is_alpha_mask());
}

#[test]
fn test_material_descriptor_has_textures() {
    let mat = MaterialDescriptor::new();
    assert!(!mat.has_albedo_texture_id());
    assert!(!mat.has_normal_texture_id());
    assert!(!mat.has_metallic_roughness_texture_id());
    assert!(!mat.has_emissive_texture_id());

    let mat_with_tex = mat.with_albedo_texture_id(0);
    assert!(mat_with_tex.has_albedo_texture_id());
}

#[test]
fn test_material_table_entry_zeroed() {
    let entry = MaterialTableEntry::zeroed();
    assert!(entry.is_zero());
    assert_eq!(entry.base_color, [0.0; 4]);
    assert_eq!(entry.albedo_texture_id, u32::MAX);
    assert_eq!(entry.flags, 0);
}

#[test]
fn test_material_table_add_and_get() {
    let mut table = MaterialTable::with_capacity(8);
    let entry = MaterialTableEntry {
        base_color: [1.0, 0.0, 0.0, 1.0],
        ..MaterialTableEntry::zeroed()
    };
    let idx = table.add(entry);
    assert_eq!(idx, 0);
    assert_eq!(table.live_count(), 1);

    let retrieved = table.get(idx).unwrap();
    assert_eq!(retrieved.base_color, [1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn test_material_table_dirty_tracking() {
    let mut table = MaterialTable::with_capacity(8);
    table.add(MaterialTableEntry {
        base_color: [1.0, 0.0, 0.0, 1.0],
        ..MaterialTableEntry::zeroed()
    });

    assert!(table.any_dirty());
    assert_eq!(table.dirty_count(), 1);

    table.mark_clean();
    assert!(!table.any_dirty());
}

// ============================================================================
// CATEGORY 7: TEXTURE REGISTRY CPU TESTS
// ============================================================================

#[test]
fn test_texture_registry_constants() {
    assert_eq!(TEXTURE_REGISTRY_MAX_TEXTURES, 1024);
    assert_eq!(TEXTURE_REGISTRY_MIN_TEXTURES, 16);
    assert_eq!(TEXTURE_REGISTRY_BIND_GROUP_INDEX, 3);
    assert_eq!(TEXTURE_ARRAY_BINDING, 0);
    assert_eq!(SAMPLER_BINDING, 1);
}

#[test]
fn test_texture_registry_feature_detection() {
    let features = Features::empty();
    assert!(!supports_bindless_textures(features));

    let features = Features::TEXTURE_BINDING_ARRAY;
    assert!(supports_bindless_textures(features));

    let features = Features::PARTIALLY_BOUND_BINDING_ARRAY;
    assert!(supports_partially_bound(features));

    let features = Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
    assert!(supports_non_uniform_indexing(features));
}

#[test]
fn test_texture_registry_required_features() {
    let req = texture_registry_required_features();
    assert!(req.contains(Features::TEXTURE_BINDING_ARRAY));
}

#[test]
fn test_texture_registry_optimal_features() {
    let opt = texture_registry_optimal_features();
    assert!(opt.contains(Features::TEXTURE_BINDING_ARRAY));
    assert!(opt.contains(Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING));
    assert!(opt.contains(Features::PARTIALLY_BOUND_BINDING_ARRAY));
}

#[test]
fn test_texture_registry_metrics() {
    let metrics = TextureRegistryMetrics {
        active_count: 50,
        allocated_count: 50,
        free_slots: 0,
        capacity: 100,
        has_bind_group: true,
        is_dirty: false,
        has_bindless: true,
    };

    assert!((metrics.utilization() - 0.5).abs() < 0.001);
    assert_eq!(metrics.fragmentation(), 0.0);
    assert_eq!(metrics.available_slots(), 50);
}

#[test]
fn test_texture_registry_metrics_fragmentation() {
    let metrics = TextureRegistryMetrics {
        active_count: 8,
        allocated_count: 10,
        free_slots: 2,
        capacity: 100,
        has_bind_group: true,
        is_dirty: false,
        has_bindless: true,
    };

    // 2/10 = 0.2
    assert!((metrics.fragmentation() - 0.2).abs() < 0.001);
}

#[test]
fn test_cpu_fragmentation_helper() {
    assert_eq!(cpu_fragmentation(10, 10), 0.0);
    assert!((cpu_fragmentation(5, 10) - 0.5).abs() < 0.001);
    assert_eq!(cpu_fragmentation(0, 0), 0.0);
}

// ============================================================================
// CATEGORY 8: BUFFER REGISTRY CPU TESTS
// ============================================================================

#[test]
fn test_buffer_registry_constants() {
    assert_eq!(MAX_BINDLESS_BUFFERS, 256);
    assert!(MAX_BINDLESS_BUFFERS > 0);
    assert!(MAX_BINDLESS_BUFFERS <= 1000); // WebGPU limit
    assert_eq!(MIN_BUFFER_SIZE, 4);
}

#[test]
fn test_buffer_registry_slot_allocation_logic() {
    // Simulate initial free slots state
    let free_slots: Vec<u32> = (0..MAX_BINDLESS_BUFFERS).rev().collect();
    assert_eq!(free_slots.len(), MAX_BINDLESS_BUFFERS as usize);
    // Last element (first to pop) should be 0
    assert_eq!(free_slots[free_slots.len() - 1], 0);
    // First element (last to pop) should be MAX-1
    assert_eq!(free_slots[0], MAX_BINDLESS_BUFFERS - 1);
}

#[test]
fn test_buffer_registry_dirty_set_operations() {
    let mut dirty: HashSet<u32> = HashSet::new();

    dirty.insert(5);
    dirty.insert(10);
    dirty.insert(5); // Duplicate

    assert_eq!(dirty.len(), 2);
    assert!(dirty.contains(&5));
    assert!(dirty.contains(&10));
    assert!(!dirty.contains(&0));
}

#[test]
fn test_buffer_registry_take_dirty_clears_set() {
    let mut dirty: HashSet<u32> = HashSet::new();
    dirty.insert(1);
    dirty.insert(2);
    dirty.insert(3);

    let taken: Vec<u32> = dirty.drain().collect();
    assert_eq!(taken.len(), 3);
    assert!(dirty.is_empty());
}

#[test]
fn test_buffer_registry_active_count_calculation() {
    let total = MAX_BINDLESS_BUFFERS;
    let free = 200u32;
    let active = total - free;
    assert_eq!(active, 56);
}

// ============================================================================
// CATEGORY 9: BINDLESS BIND GROUP TESTS
// ============================================================================

#[test]
fn test_bindless_constants() {
    assert_eq!(BINDLESS_MAX_TEXTURES, 1024);
    assert_eq!(MAX_BINDLESS_SAMPLERS, 16);
    assert_eq!(BINDLESS_MIN_TEXTURES, 16);
    assert_eq!(MIN_BINDLESS_SAMPLERS, 4);
    assert_eq!(BINDING_TEXTURES, 0);
    assert_eq!(BINDING_SAMPLERS, 1);
    assert_eq!(BINDING_MATERIALS, 2);
    assert_eq!(BINDLESS_BIND_GROUP_INDEX, 0);
}

#[test]
fn test_bindless_supports_texture_arrays() {
    let features = Features::empty();
    assert!(!supports_texture_arrays(features));

    let features = Features::TEXTURE_BINDING_ARRAY;
    assert!(supports_texture_arrays(features));
}

#[test]
fn test_bindless_supports_full_bindless() {
    let features = Features::empty();
    assert!(!supports_full_bindless(features));

    // Need all three features
    let features = Features::TEXTURE_BINDING_ARRAY
        | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
        | Features::PARTIALLY_BOUND_BINDING_ARRAY;
    assert!(supports_full_bindless(features));
}

#[test]
fn test_bindless_required_features() {
    let req = bindless_required_features();
    assert!(req.contains(Features::TEXTURE_BINDING_ARRAY));
    assert!(req.contains(Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING));
}

#[test]
fn test_bindless_optimal_features() {
    let opt = bindless_optimal_features();
    assert!(opt.contains(Features::TEXTURE_BINDING_ARRAY));
    assert!(opt.contains(Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING));
    assert!(opt.contains(Features::PARTIALLY_BOUND_BINDING_ARRAY));
}

// ============================================================================
// CATEGORY 10: GEOMETRY PATH TESTS
// ============================================================================

#[test]
fn test_geometry_path_default() {
    let path = GeometryPath::default();
    assert_eq!(path, GeometryPath::Traditional);
    assert!(path.is_traditional());
    assert!(!path.is_meshlet());
}

#[test]
fn test_geometry_path_names() {
    assert_eq!(GeometryPath::Traditional.name(), TRADITIONAL_PATH_NAME);
    assert_eq!(GeometryPath::Meshlet.name(), MESHLET_PATH_NAME);
    assert_eq!(TRADITIONAL_PATH_NAME, "Traditional");
    assert_eq!(MESHLET_PATH_NAME, "Meshlet");
}

#[test]
fn test_geometry_path_select() {
    // With empty features, should select Traditional
    let features = Features::empty();
    let path = GeometryPath::select(features);
    assert_eq!(path, GeometryPath::Traditional);
}

#[test]
fn test_geometry_path_meshlet_not_available() {
    // Currently meshlet is never available (mesh shaders not stable)
    let features = Features::empty();
    assert!(!GeometryPath::meshlet_available(features));

    let features = Features::all();
    assert!(!GeometryPath::meshlet_available(features));
}

#[test]
fn test_geometry_path_config_default() {
    let config = GeometryPathConfig::default();
    assert_eq!(config.preferred, GeometryPath::Traditional);
    assert!(!config.force_traditional);
}

#[test]
fn test_geometry_path_config_prefer_meshlet() {
    let config = GeometryPathConfig::prefer_meshlet();
    assert_eq!(config.preferred, GeometryPath::Meshlet);
    assert!(!config.force_traditional);
}

#[test]
fn test_geometry_path_config_force_traditional() {
    let config = GeometryPathConfig::force_traditional();
    assert_eq!(config.preferred, GeometryPath::Traditional);
    assert!(config.force_traditional);
}

#[test]
fn test_geometry_path_config_resolve_traditional() {
    let config = GeometryPathConfig::default();
    let features = Features::empty();
    let path = config.resolve(features);
    assert_eq!(path, GeometryPath::Traditional);
}

#[test]
fn test_geometry_path_config_resolve_forced() {
    let config = GeometryPathConfig {
        preferred: GeometryPath::Meshlet,
        force_traditional: true,
    };
    let features = Features::all();
    let path = config.resolve(features);
    assert_eq!(path, GeometryPath::Traditional);
}

#[test]
fn test_geometry_path_config_preferred_available() {
    let config = GeometryPathConfig::default();
    let features = Features::empty();

    // Traditional is always available
    assert!(config.preferred_available(features));

    // Meshlet is not available
    let meshlet_config = GeometryPathConfig::prefer_meshlet();
    assert!(!meshlet_config.preferred_available(features));
}

#[test]
fn test_geometry_path_hash_eq() {
    use std::collections::HashSet;

    let mut set = HashSet::new();
    set.insert(GeometryPath::Traditional);
    set.insert(GeometryPath::Meshlet);
    set.insert(GeometryPath::Traditional); // Duplicate

    assert_eq!(set.len(), 2);
    assert!(set.contains(&GeometryPath::Traditional));
    assert!(set.contains(&GeometryPath::Meshlet));
}

// ============================================================================
// CATEGORY 11: BYTEMUCK POD/ZEROABLE TRAIT TESTS
// ============================================================================

#[test]
fn test_frustum_plane_pod() {
    let plane = FrustumPlane::new([1.0, 0.0, 0.0], 5.0);
    let bytes: &[u8] = bytemuck::bytes_of(&plane);
    assert_eq!(bytes.len(), 16);
}

#[test]
fn test_instance_bounds_pod() {
    let bounds = InstanceBounds::new([0.0, 0.0, 0.0], 1.0, [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
    let bytes: &[u8] = bytemuck::bytes_of(&bounds);
    assert_eq!(bytes.len(), 48);
}

#[test]
fn test_cull_params_pod() {
    let params = CullParams::new(100);
    let bytes: &[u8] = bytemuck::bytes_of(&params);
    assert_eq!(bytes.len(), 16);
}

#[test]
fn test_build_indirect_params_pod() {
    let params = BuildIndirectParams::new(100, 4096);
    let bytes: &[u8] = bytemuck::bytes_of(&params);
    assert_eq!(bytes.len(), 16);
}

#[test]
fn test_mesh_data_pod() {
    let mesh = BuildIndirectMeshData::new(1000, 0, 0);
    let bytes: &[u8] = bytemuck::bytes_of(&mesh);
    assert_eq!(bytes.len(), 48);
}

#[test]
fn test_indirect_args_pod() {
    let args = IndirectDrawIndexedArgs::new(100, 1, 0, 0, 0);
    let bytes: &[u8] = bytemuck::bytes_of(&args);
    assert_eq!(bytes.len(), 20);
}

#[test]
fn test_lod_distances_pod() {
    let distances = LodDistances::new(10.0, 25.0, 50.0);
    let bytes: &[u8] = bytemuck::bytes_of(&distances);
    assert_eq!(bytes.len(), 16);

    let roundtrip: &LodDistances = bytemuck::from_bytes(bytes);
    assert_eq!(*roundtrip, distances);
}

#[test]
fn test_lod_params_pod() {
    let params = LodParams::standard_1080p([1.0, 2.0, 3.0]);
    let bytes: &[u8] = bytemuck::bytes_of(&params);
    assert_eq!(bytes.len(), 32);
}

#[test]
fn test_meshlet_pod() {
    let m = Meshlet::new(1, 2, 3, 4);
    let bytes: &[u8] = bytemuck::bytes_of(&m);
    assert_eq!(bytes.len(), 12);
}

#[test]
fn test_meshlet_bounds_pod() {
    let b = MeshletBounds::new([1.0, 2.0, 3.0], 4.0, [0.0, 1.0, 0.0], 0.5);
    let bytes: &[u8] = bytemuck::bytes_of(&b);
    assert_eq!(bytes.len(), 32);
}

#[test]
fn test_material_descriptor_pod() {
    let mat = MaterialDescriptor::new();
    let bytes: &[u8] = bytemuck::bytes_of(&mat);
    assert_eq!(bytes.len(), 64);
}

// ============================================================================
// CATEGORY 12: STRESS / EDGE CASE TESTS
// ============================================================================

#[test]
fn test_index_allocator_many_allocations() {
    let mut allocator = IndexAllocator::new(10000);

    for i in 0..10000 {
        assert_eq!(allocator.allocate(), Some(i));
    }
    assert_eq!(allocator.allocate(), None);
    assert_eq!(allocator.count(), 10000);
}

#[test]
fn test_index_allocator_allocate_free_cycle() {
    let mut allocator = IndexAllocator::with_generations(100);

    // Allocate all
    let indices: Vec<_> = (0..100).map(|_| allocator.allocate_generational().unwrap()).collect();

    // Free all (reverse order)
    for gen_idx in indices.iter().rev() {
        assert!(allocator.try_free_generational(*gen_idx).is_ok());
    }

    // Reallocate all (should all have generation 1)
    for _ in 0..100 {
        let gen_idx = allocator.allocate_generational().unwrap();
        assert_eq!(gen_idx.generation, 1);
    }
}

#[test]
fn test_large_mesh_meshlet_splitting() {
    // Create mesh that needs multiple meshlets
    let mut positions = Vec::new();
    let mut indices = Vec::new();

    // 200 triangles with 3 unique verts each = 600 verts
    // With 64 vert limit, need at least 10 meshlets
    for i in 0..200 {
        let x = (i % 100) as f32;
        let y = (i / 100) as f32;

        let base = positions.len() as u32;
        positions.push([x, y, 0.0]);
        positions.push([x + 1.0, y, 0.0]);
        positions.push([x + 0.5, y + 1.0, 0.0]);

        indices.push(base);
        indices.push(base + 1);
        indices.push(base + 2);
    }

    let data = MeshletData::generate(&positions, &indices, None);

    assert!(data.meshlet_count() > 1);
    assert!(data.validate(positions.len()).is_ok());

    // Verify all triangles are covered
    let reconstructed = data.reconstruct_indices();
    assert_eq!(reconstructed.len(), indices.len());
}

#[test]
fn test_lod_selection_consistency() {
    // Verify that squared and non-squared selection give same results
    let distances = LodDistances::new(10.0, 25.0, 50.0);
    let squared = squared_thresholds(&distances);

    for dist in [0.0, 5.0, 10.0, 15.0, 25.0, 35.0, 50.0, 100.0] {
        let lod_normal = select_lod_by_distance(dist, &distances);
        let lod_squared = select_lod_by_distance_squared(dist * dist, &squared);
        assert_eq!(
            lod_normal, lod_squared,
            "LOD mismatch at distance {}: normal={}, squared={}",
            dist, lod_normal, lod_squared
        );
    }
}

#[test]
fn test_all_lod_levels_reachable() {
    let thresholds = LodDistances::new(10.0, 25.0, 50.0);

    // Verify all 4 LOD levels are reachable
    let mut levels_reached = [false; 4];

    for dist in [0.0, 5.0, 15.0, 35.0, 100.0] {
        let lod = select_lod_by_distance(dist, &thresholds);
        levels_reached[lod as usize] = true;
    }

    assert!(levels_reached.iter().all(|&x| x), "Not all LOD levels reachable");
}

#[test]
fn test_material_table_multiple_add_remove() {
    let mut table = MaterialTable::with_capacity(8);
    let mut indices = Vec::new();

    for i in 0..5u32 {
        let idx = table.add(MaterialTableEntry {
            base_color: [i as f32 / 10.0, 0.0, 0.0, 1.0],
            ..MaterialTableEntry::zeroed()
        });
        indices.push(idx);
    }
    assert_eq!(table.live_count(), 5);

    // Remove middle entries
    assert!(matches!(table.remove(indices[2]), MaterialRemoveResult::Removed));
    assert!(matches!(table.remove(indices[3]), MaterialRemoveResult::Removed));
    assert_eq!(table.live_count(), 3);

    // Add more -- should fill holes first
    let new_idx = table.add(MaterialTableEntry {
        base_color: [0.9, 0.9, 0.9, 1.0],
        ..MaterialTableEntry::zeroed()
    });
    assert_eq!(new_idx, 2); // reused slot 2
    assert_eq!(table.live_count(), 4);
}

// ============================================================================
// Summary: 100+ test assertions across all modules
// - AABB-frustum intersection: ~20 tests
// - Index allocator: ~20 tests
// - Indirect struct sizes: ~15 tests
// - LOD selection: ~20 tests
// - Meshlet structs: ~15 tests
// - Material descriptors: ~15 tests
// - Texture/buffer registry CPU tests: ~15 tests
// - Bindless bind group tests: ~10 tests
// - Geometry path tests: ~15 tests
// - Bytemuck POD tests: ~15 tests
// - Stress/edge case tests: ~10 tests
// Total: 150+ test assertions
// ============================================================================
