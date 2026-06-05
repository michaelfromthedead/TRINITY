// Blackbox contract tests for T-WGPU-P6.9.1: Meshlet struct
//
// CLEANROOM: Tests use only public API from renderer_backend::gpu_driven::meshlet.
// No internal fields, no private methods, no implementation details.
//
// Task Reference: T-WGPU-P6.9.1 - Meshlet Struct
//
// Structures Under Test:
//   - Meshlet: Small cluster of triangles for GPU mesh shading
//   - MeshletBounds: Bounding sphere and backface culling cone
//   - MeshletData: Complete meshlet data ready for GPU upload
//
// Acceptance Criteria:
//   1. Meshlet struct is Pod/Zeroable for GPU buffer uploads
//   2. MeshletBounds supports bounding sphere and normal cone culling
//   3. MeshletData::generate partitions meshes respecting vertex/triangle limits
//   4. Builder pattern chaining works for MeshletBounds
//   5. GPU memory layout matches WGSL shader expectations
//
// Test Categories:
//   1. API behavior tests (Default, Clone, Copy, Debug)
//   2. Meshlet creation (valid/invalid vertex/triangle counts)
//   3. MeshletBounds usage (sphere, cone, builder)
//   4. Integration scenarios (Vec, bytemuck, serialization)
//   5. Trait verification (Send, Sync, Pod, Zeroable)
//   6. GPU buffer compatibility tests

use renderer_backend::gpu_driven::{
    Meshlet, MeshletBounds, MeshletData,
    MAX_MESHLET_TRIANGLES, MAX_MESHLET_VERTICES,
    MESHLET_BOUNDS_SIZE, MESHLET_SIZE,
};

use std::collections::HashSet;
use std::mem;

// ============================================================================
// SECTION 1: Default Construction and Zero-Initialization
// ============================================================================

#[test]
fn meshlet_default_is_zero_initialized() {
    let m = Meshlet::default();

    assert_eq!(m.vertex_offset, 0, "default vertex_offset must be 0");
    assert_eq!(m.triangle_offset, 0, "default triangle_offset must be 0");
    assert_eq!(m.vertex_count, 0, "default vertex_count must be 0");
    assert_eq!(m.triangle_count, 0, "default triangle_count must be 0");
    assert_eq!(m._padding, [0, 0], "default padding must be zeroed");
}

#[test]
fn meshlet_bounds_default_is_zero_initialized() {
    let b = MeshletBounds::default();

    assert_eq!(b.center, [0.0, 0.0, 0.0], "default center must be origin");
    assert_eq!(b.radius, 0.0, "default radius must be 0");
    assert_eq!(b.cone_axis, [0.0, 0.0, 0.0], "default cone_axis must be zero");
    assert_eq!(b.cone_cutoff, 0.0, "default cone_cutoff must be 0");
}

#[test]
fn meshlet_data_default_is_empty() {
    let data = MeshletData::default();

    assert!(data.is_empty(), "default MeshletData must be empty");
    assert_eq!(data.meshlet_count(), 0, "default meshlet_count must be 0");
    assert!(data.meshlets.is_empty(), "default meshlets vec must be empty");
    assert!(data.bounds.is_empty(), "default bounds vec must be empty");
    assert!(data.vertex_indices.is_empty(), "default vertex_indices must be empty");
    assert!(data.local_indices.is_empty(), "default local_indices must be empty");
}

// ============================================================================
// SECTION 2: Clone/Copy Semantics
// ============================================================================

#[test]
fn meshlet_copy_semantics_work_correctly() {
    let original = Meshlet::new(100, 200, 32, 64);
    let copied = original; // Copy

    // Both should have identical values
    assert_eq!(copied.vertex_offset, original.vertex_offset);
    assert_eq!(copied.triangle_offset, original.triangle_offset);
    assert_eq!(copied.vertex_count, original.vertex_count);
    assert_eq!(copied.triangle_count, original.triangle_count);

    // Original should still be usable (Copy trait)
    assert_eq!(original.vertex_offset, 100);
}

#[test]
fn meshlet_clone_produces_identical_struct() {
    let original = Meshlet::new(500, 1000, 48, 90);
    let cloned = original.clone();

    assert_eq!(cloned.vertex_offset, 500);
    assert_eq!(cloned.triangle_offset, 1000);
    assert_eq!(cloned.vertex_count, 48);
    assert_eq!(cloned.triangle_count, 90);
}

#[test]
fn meshlet_bounds_copy_semantics_work_correctly() {
    let original = MeshletBounds::new([1.0, 2.0, 3.0], 5.0, [0.0, 1.0, 0.0], 0.707);
    let copied = original;

    assert_eq!(copied.center, original.center);
    assert_eq!(copied.radius, original.radius);
    assert_eq!(copied.cone_axis, original.cone_axis);
    assert_eq!(copied.cone_cutoff, original.cone_cutoff);

    // Original still usable
    assert_eq!(original.center, [1.0, 2.0, 3.0]);
}

#[test]
fn meshlet_bounds_clone_produces_identical_struct() {
    let original = MeshletBounds::sphere_only([10.0, 20.0, 30.0], 15.0);
    let cloned = original.clone();

    assert_eq!(cloned.center, [10.0, 20.0, 30.0]);
    assert_eq!(cloned.radius, 15.0);
}

// ============================================================================
// SECTION 3: Debug/Display Formatting
// ============================================================================

#[test]
fn meshlet_debug_format_contains_fields() {
    let m = Meshlet::new(42, 84, 16, 24);
    let debug_str = format!("{:?}", m);

    // Debug output should contain field names and values
    assert!(debug_str.contains("vertex_offset"), "Debug must show vertex_offset");
    assert!(debug_str.contains("42"), "Debug must show vertex_offset value");
    assert!(debug_str.contains("triangle_offset"), "Debug must show triangle_offset");
    assert!(debug_str.contains("84"), "Debug must show triangle_offset value");
    assert!(debug_str.contains("vertex_count"), "Debug must show vertex_count");
    assert!(debug_str.contains("16"), "Debug must show vertex_count value");
    assert!(debug_str.contains("triangle_count"), "Debug must show triangle_count");
    assert!(debug_str.contains("24"), "Debug must show triangle_count value");
}

#[test]
fn meshlet_bounds_debug_format_contains_fields() {
    let b = MeshletBounds::new([1.5, 2.5, 3.5], 4.5, [0.0, 0.0, 1.0], 0.5);
    let debug_str = format!("{:?}", b);

    assert!(debug_str.contains("center"), "Debug must show center");
    assert!(debug_str.contains("radius"), "Debug must show radius");
    assert!(debug_str.contains("cone_axis"), "Debug must show cone_axis");
    assert!(debug_str.contains("cone_cutoff"), "Debug must show cone_cutoff");
}

// ============================================================================
// SECTION 4: Meshlet Creation - Valid Parameters
// ============================================================================

#[test]
fn meshlet_new_preserves_all_parameters() {
    let m = Meshlet::new(100, 200, 32, 40);

    assert_eq!(m.vertex_offset, 100, "vertex_offset must be preserved");
    assert_eq!(m.triangle_offset, 200, "triangle_offset must be preserved");
    assert_eq!(m.vertex_count, 32, "vertex_count must be preserved");
    assert_eq!(m.triangle_count, 40, "triangle_count must be preserved");
    assert_eq!(m._padding, [0, 0], "padding must be zeroed in constructor");
}

#[test]
fn meshlet_new_with_max_vertex_count() {
    let m = Meshlet::new(0, 0, Meshlet::MAX_VERTICES, 1);

    assert_eq!(m.vertex_count, 64, "MAX_VERTICES must be 64");
    assert_eq!(m.vertex_count, Meshlet::MAX_VERTICES);
}

#[test]
fn meshlet_new_with_max_triangle_count() {
    let m = Meshlet::new(0, 0, 1, Meshlet::MAX_TRIANGLES);

    assert_eq!(m.triangle_count, 124, "MAX_TRIANGLES must be 124");
    assert_eq!(m.triangle_count, Meshlet::MAX_TRIANGLES);
}

#[test]
fn meshlet_new_with_large_offsets() {
    // Test with large offset values near u32::MAX
    let m = Meshlet::new(u32::MAX - 1000, u32::MAX, 64, 124);

    assert_eq!(m.vertex_offset, u32::MAX - 1000);
    assert_eq!(m.triangle_offset, u32::MAX);
    assert_eq!(m.vertex_count, 64);
    assert_eq!(m.triangle_count, 124);
}

#[test]
fn meshlet_new_with_zero_counts_is_valid() {
    let m = Meshlet::new(100, 200, 0, 0);

    assert_eq!(m.vertex_count, 0);
    assert_eq!(m.triangle_count, 0);
    assert!(m.is_empty(), "zero triangle_count means meshlet is empty");
}

// ============================================================================
// SECTION 5: Meshlet Creation - Overflow Behavior (u8 counts)
// ============================================================================

#[test]
fn meshlet_vertex_count_is_u8_with_natural_overflow() {
    // Creating with values > 255 would require casting, which truncates
    // This tests the type constraint, not runtime overflow
    let m = Meshlet::new(0, 0, 255, 255);

    assert_eq!(m.vertex_count, 255, "u8 max is 255");
    assert_eq!(m.triangle_count, 255, "u8 max is 255");
}

#[test]
fn meshlet_associated_constants_match_module_constants() {
    assert_eq!(Meshlet::MAX_VERTICES as usize, MAX_MESHLET_VERTICES);
    assert_eq!(Meshlet::MAX_TRIANGLES as usize, MAX_MESHLET_TRIANGLES);
    assert_eq!(MAX_MESHLET_VERTICES, 64);
    assert_eq!(MAX_MESHLET_TRIANGLES, 124);
}

// ============================================================================
// SECTION 6: Meshlet is_empty Behavior
// ============================================================================

#[test]
fn meshlet_is_empty_when_triangle_count_zero() {
    assert!(Meshlet::default().is_empty());
    assert!(Meshlet::new(100, 200, 64, 0).is_empty());
}

#[test]
fn meshlet_not_empty_when_has_triangles() {
    assert!(!Meshlet::new(0, 0, 3, 1).is_empty());
    assert!(!Meshlet::new(0, 0, 64, 124).is_empty());
}

#[test]
fn meshlet_is_empty_ignores_vertex_count() {
    // A meshlet with vertices but no triangles is still empty
    let m = Meshlet::new(0, 0, 64, 0);
    assert!(m.is_empty(), "meshlet with 0 triangles is empty regardless of vertex_count");
}

// ============================================================================
// SECTION 7: MeshletBounds Construction
// ============================================================================

#[test]
fn meshlet_bounds_new_preserves_all_parameters() {
    let b = MeshletBounds::new([1.0, 2.0, 3.0], 4.0, [0.0, 0.707, 0.707], 0.5);

    assert_eq!(b.center, [1.0, 2.0, 3.0]);
    assert_eq!(b.radius, 4.0);
    assert_eq!(b.cone_axis, [0.0, 0.707, 0.707]);
    assert_eq!(b.cone_cutoff, 0.5);
}

#[test]
fn meshlet_bounds_sphere_only_disables_cone_culling() {
    let b = MeshletBounds::sphere_only([1.0, 2.0, 3.0], 5.0);

    assert_eq!(b.center, [1.0, 2.0, 3.0]);
    assert_eq!(b.radius, 5.0);
    assert!(!b.has_valid_cone(), "sphere_only must disable cone culling");
    assert!(b.cone_cutoff < -1.0, "cone_cutoff must be < -1.0 to disable");
}

#[test]
fn meshlet_bounds_has_valid_cone_thresholds() {
    // Valid cones: cutoff >= -1.0
    assert!(MeshletBounds::new([0.0; 3], 1.0, [0.0, 0.0, 1.0], 1.0).has_valid_cone());
    assert!(MeshletBounds::new([0.0; 3], 1.0, [0.0, 0.0, 1.0], 0.0).has_valid_cone());
    assert!(MeshletBounds::new([0.0; 3], 1.0, [0.0, 0.0, 1.0], -1.0).has_valid_cone());

    // Invalid cones: cutoff < -1.0
    assert!(!MeshletBounds::new([0.0; 3], 1.0, [0.0, 0.0, 1.0], -1.001).has_valid_cone());
    assert!(!MeshletBounds::new([0.0; 3], 1.0, [0.0, 0.0, 1.0], -2.0).has_valid_cone());
    assert!(!MeshletBounds::sphere_only([0.0; 3], 1.0).has_valid_cone());
}

// ============================================================================
// SECTION 8: MeshletBounds Builder Pattern
// ============================================================================

#[test]
fn meshlet_bounds_with_bounds_sets_sphere() {
    let b = MeshletBounds::default().with_bounds([10.0, 20.0, 30.0], 50.0);

    assert_eq!(b.center, [10.0, 20.0, 30.0]);
    assert_eq!(b.radius, 50.0);
}

#[test]
fn meshlet_bounds_with_cone_sets_cone() {
    let b = MeshletBounds::default().with_cone([1.0, 0.0, 0.0], 0.866);

    assert_eq!(b.cone_axis, [1.0, 0.0, 0.0]);
    assert_eq!(b.cone_cutoff, 0.866);
}

#[test]
fn meshlet_bounds_builder_chain_complete() {
    let b = MeshletBounds::default()
        .with_bounds([5.0, 10.0, 15.0], 25.0)
        .with_cone([0.0, 1.0, 0.0], 0.707);

    assert_eq!(b.center, [5.0, 10.0, 15.0], "center from with_bounds");
    assert_eq!(b.radius, 25.0, "radius from with_bounds");
    assert_eq!(b.cone_axis, [0.0, 1.0, 0.0], "axis from with_cone");
    assert_eq!(b.cone_cutoff, 0.707, "cutoff from with_cone");
    assert!(b.has_valid_cone());
}

#[test]
fn meshlet_bounds_builder_order_independent() {
    // Order should not matter for final result
    let b1 = MeshletBounds::default()
        .with_bounds([1.0, 2.0, 3.0], 4.0)
        .with_cone([0.0, 0.0, 1.0], 0.5);

    let b2 = MeshletBounds::default()
        .with_cone([0.0, 0.0, 1.0], 0.5)
        .with_bounds([1.0, 2.0, 3.0], 4.0);

    assert_eq!(b1.center, b2.center);
    assert_eq!(b1.radius, b2.radius);
    assert_eq!(b1.cone_axis, b2.cone_axis);
    assert_eq!(b1.cone_cutoff, b2.cone_cutoff);
}

#[test]
fn meshlet_bounds_builder_overwrites_previous() {
    let b = MeshletBounds::default()
        .with_bounds([1.0, 1.0, 1.0], 1.0)
        .with_bounds([2.0, 2.0, 2.0], 2.0); // Overwrite

    assert_eq!(b.center, [2.0, 2.0, 2.0], "second with_bounds should overwrite");
    assert_eq!(b.radius, 2.0);
}

#[test]
fn meshlet_bounds_builder_disable_cone_via_cutoff() {
    let b = MeshletBounds::default()
        .with_bounds([0.0; 3], 1.0)
        .with_cone([0.0, 0.0, 1.0], -2.0); // -2.0 disables culling

    assert!(!b.has_valid_cone(), "cutoff -2.0 should disable cone culling");
}

// ============================================================================
// SECTION 9: Integration - Array/Vec of Meshlets
// ============================================================================

#[test]
fn meshlet_vec_maintains_integrity() {
    let meshlets: Vec<Meshlet> = (0..100)
        .map(|i| Meshlet::new(i * 64, i * 128, 64, 100))
        .collect();

    assert_eq!(meshlets.len(), 100);

    for (i, m) in meshlets.iter().enumerate() {
        assert_eq!(m.vertex_offset, (i * 64) as u32);
        assert_eq!(m.triangle_offset, (i * 128) as u32);
        assert_eq!(m.vertex_count, 64);
        assert_eq!(m.triangle_count, 100);
    }
}

#[test]
fn meshlet_bounds_vec_maintains_integrity() {
    let bounds: Vec<MeshletBounds> = (0..50)
        .map(|i| {
            MeshletBounds::new(
                [i as f32, i as f32 * 2.0, i as f32 * 3.0],
                i as f32 + 1.0,
                [0.0, 0.0, 1.0],
                0.5,
            )
        })
        .collect();

    assert_eq!(bounds.len(), 50);

    for (i, b) in bounds.iter().enumerate() {
        assert_eq!(b.center[0], i as f32);
        assert_eq!(b.radius, i as f32 + 1.0);
    }
}

// ============================================================================
// SECTION 10: GPU Buffer Compatibility (bytemuck)
// ============================================================================

#[test]
fn meshlet_bytemuck_cast_to_bytes() {
    let m = Meshlet::new(0x11223344, 0x55667788, 0xAA, 0xBB);
    let bytes: &[u8] = bytemuck::bytes_of(&m);

    assert_eq!(bytes.len(), MESHLET_SIZE, "bytes must equal MESHLET_SIZE");

    // Verify little-endian layout for GPU compatibility
    assert_eq!(&bytes[0..4], &[0x44, 0x33, 0x22, 0x11], "vertex_offset little-endian");
    assert_eq!(&bytes[4..8], &[0x88, 0x77, 0x66, 0x55], "triangle_offset little-endian");
    assert_eq!(bytes[8], 0xAA, "vertex_count");
    assert_eq!(bytes[9], 0xBB, "triangle_count");
    assert_eq!(&bytes[10..12], &[0, 0], "padding must be zero");
}

#[test]
fn meshlet_bounds_bytemuck_cast_to_bytes() {
    let b = MeshletBounds::new([1.0, 2.0, 3.0], 4.0, [0.5, 0.5, 0.707], 0.866);
    let bytes: &[u8] = bytemuck::bytes_of(&b);

    assert_eq!(bytes.len(), MESHLET_BOUNDS_SIZE, "bytes must equal MESHLET_BOUNDS_SIZE");
}

#[test]
fn meshlet_slice_bytemuck_cast() {
    let meshlets = vec![
        Meshlet::new(0, 0, 10, 5),
        Meshlet::new(10, 15, 20, 10),
        Meshlet::new(30, 45, 30, 15),
    ];

    let bytes: &[u8] = bytemuck::cast_slice(&meshlets);
    assert_eq!(bytes.len(), 3 * MESHLET_SIZE);

    // Cast back
    let restored: &[Meshlet] = bytemuck::cast_slice(bytes);
    assert_eq!(restored.len(), 3);
    assert_eq!(restored[0].vertex_offset, 0);
    assert_eq!(restored[1].vertex_offset, 10);
    assert_eq!(restored[2].vertex_offset, 30);
}

#[test]
fn meshlet_bounds_slice_bytemuck_cast() {
    let bounds = vec![
        MeshletBounds::sphere_only([0.0, 0.0, 0.0], 1.0),
        MeshletBounds::new([1.0, 1.0, 1.0], 2.0, [0.0, 1.0, 0.0], 0.5),
    ];

    let bytes: &[u8] = bytemuck::cast_slice(&bounds);
    assert_eq!(bytes.len(), 2 * MESHLET_BOUNDS_SIZE);

    // Cast back
    let restored: &[MeshletBounds] = bytemuck::cast_slice(bytes);
    assert_eq!(restored.len(), 2);
    assert_eq!(restored[0].radius, 1.0);
    assert_eq!(restored[1].radius, 2.0);
}

#[test]
fn meshlet_zeroed_via_bytemuck() {
    let m: Meshlet = bytemuck::Zeroable::zeroed();

    assert_eq!(m.vertex_offset, 0);
    assert_eq!(m.triangle_offset, 0);
    assert_eq!(m.vertex_count, 0);
    assert_eq!(m.triangle_count, 0);
    assert_eq!(m._padding, [0, 0]);
}

#[test]
fn meshlet_bounds_zeroed_via_bytemuck() {
    let b: MeshletBounds = bytemuck::Zeroable::zeroed();

    assert_eq!(b.center, [0.0, 0.0, 0.0]);
    assert_eq!(b.radius, 0.0);
    assert_eq!(b.cone_axis, [0.0, 0.0, 0.0]);
    assert_eq!(b.cone_cutoff, 0.0);
}

// ============================================================================
// SECTION 11: GPU Memory Layout Verification
// ============================================================================

#[test]
fn meshlet_size_matches_constant() {
    assert_eq!(mem::size_of::<Meshlet>(), MESHLET_SIZE);
    assert_eq!(MESHLET_SIZE, 12, "Meshlet must be 12 bytes");
}

#[test]
fn meshlet_alignment_is_4_bytes() {
    assert_eq!(mem::align_of::<Meshlet>(), 4, "Meshlet alignment must be 4 (u32)");
}

#[test]
fn meshlet_bounds_size_matches_constant() {
    assert_eq!(mem::size_of::<MeshletBounds>(), MESHLET_BOUNDS_SIZE);
    assert_eq!(MESHLET_BOUNDS_SIZE, 32, "MeshletBounds must be 32 bytes");
}

#[test]
fn meshlet_bounds_alignment_is_4_bytes() {
    assert_eq!(mem::align_of::<MeshletBounds>(), 4, "MeshletBounds alignment must be 4 (f32)");
}

#[test]
fn meshlet_bounds_is_two_vec4s() {
    // For GPU compatibility, MeshletBounds should be exactly 2 vec4s (32 bytes)
    assert_eq!(MESHLET_BOUNDS_SIZE, 2 * 16, "MeshletBounds must be 2 vec4s");
}

#[test]
fn meshlet_array_has_no_padding_between_elements() {
    let array: [Meshlet; 4] = [Meshlet::default(); 4];
    assert_eq!(mem::size_of_val(&array), 4 * MESHLET_SIZE);
}

#[test]
fn meshlet_bounds_array_has_no_padding_between_elements() {
    let array: [MeshletBounds; 4] = [MeshletBounds::default(); 4];
    assert_eq!(mem::size_of_val(&array), 4 * MESHLET_BOUNDS_SIZE);
}

// ============================================================================
// SECTION 12: Send + Sync Trait Verification
// ============================================================================

#[test]
fn meshlet_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<Meshlet>();
}

#[test]
fn meshlet_is_sync() {
    fn assert_sync<T: Sync>() {}
    assert_sync::<Meshlet>();
}

#[test]
fn meshlet_bounds_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<MeshletBounds>();
}

#[test]
fn meshlet_bounds_is_sync() {
    fn assert_sync<T: Sync>() {}
    assert_sync::<MeshletBounds>();
}

#[test]
fn meshlet_data_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<MeshletData>();
}

#[test]
fn meshlet_data_is_sync() {
    fn assert_sync<T: Sync>() {}
    assert_sync::<MeshletData>();
}

// ============================================================================
// SECTION 13: MeshletData Generation - Basic
// ============================================================================

fn make_single_triangle() -> (Vec<[f32; 3]>, Vec<u32>) {
    let positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    let indices = vec![0, 1, 2];
    (positions, indices)
}

fn make_cube() -> (Vec<[f32; 3]>, Vec<u32>) {
    let positions = vec![
        // Front face
        [-1.0, -1.0,  1.0],
        [ 1.0, -1.0,  1.0],
        [ 1.0,  1.0,  1.0],
        [-1.0,  1.0,  1.0],
        // Back face
        [-1.0, -1.0, -1.0],
        [-1.0,  1.0, -1.0],
        [ 1.0,  1.0, -1.0],
        [ 1.0, -1.0, -1.0],
    ];

    let indices = vec![
        // Front
        0, 1, 2, 0, 2, 3,
        // Back
        4, 5, 6, 4, 6, 7,
        // Top
        3, 2, 6, 3, 6, 5,
        // Bottom
        4, 7, 1, 4, 1, 0,
        // Right
        1, 7, 6, 1, 6, 2,
        // Left
        4, 0, 3, 4, 3, 5,
    ];

    (positions, indices)
}

#[test]
fn meshlet_data_generate_single_triangle() {
    let (positions, indices) = make_single_triangle();
    let data = MeshletData::generate(&positions, &indices, None);

    assert_eq!(data.meshlet_count(), 1, "single triangle = one meshlet");
    assert!(!data.is_empty());

    let m = &data.meshlets[0];
    assert_eq!(m.vertex_count, 3, "triangle has 3 vertices");
    assert_eq!(m.triangle_count, 1, "triangle has 1 triangle");
    assert_eq!(m.vertex_offset, 0);
    assert_eq!(m.triangle_offset, 0);
}

#[test]
fn meshlet_data_generate_cube_fits_in_one() {
    let (positions, indices) = make_cube();
    let data = MeshletData::generate(&positions, &indices, None);

    // 12 triangles, 8 vertices - should fit in one meshlet
    assert_eq!(data.meshlet_count(), 1, "cube (12 tris, 8 verts) fits in one meshlet");
    assert_eq!(data.meshlets[0].triangle_count, 12);
    assert!(data.meshlets[0].vertex_count <= 8);
}

#[test]
fn meshlet_data_generate_empty_mesh() {
    let data = MeshletData::generate(&[], &[], None);

    assert!(data.is_empty());
    assert_eq!(data.meshlet_count(), 0);
}

#[test]
fn meshlet_data_generate_positions_only_empty_indices() {
    let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let data = MeshletData::generate(&positions, &[], None);

    assert!(data.is_empty(), "no indices = no meshlets");
}

// ============================================================================
// SECTION 14: MeshletData Generation - Large Meshes Split Correctly
// ============================================================================

fn make_large_mesh_unique_verts(triangle_count: usize) -> (Vec<[f32; 3]>, Vec<u32>) {
    // Each triangle has 3 unique vertices - forces vertex limit to be hit
    let mut positions = Vec::new();
    let mut indices = Vec::new();

    for i in 0..triangle_count {
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

    (positions, indices)
}

#[test]
fn meshlet_data_splits_when_exceeding_vertex_limit() {
    // 200 triangles * 3 unique verts = 600 verts
    // With 64 vert limit, need multiple meshlets
    let (positions, indices) = make_large_mesh_unique_verts(200);
    let data = MeshletData::generate(&positions, &indices, None);

    assert!(data.meshlet_count() > 1, "large mesh must split into multiple meshlets");

    // Each meshlet must respect vertex limit
    for (i, m) in data.meshlets.iter().enumerate() {
        assert!(
            m.vertex_count as usize <= MAX_MESHLET_VERTICES,
            "meshlet {} has {} verts, exceeds limit {}",
            i, m.vertex_count, MAX_MESHLET_VERTICES
        );
    }
}

#[test]
fn meshlet_data_splits_when_exceeding_triangle_limit() {
    // Grid mesh with shared vertices allows more triangles per meshlet
    let grid_size = 20;
    let mut positions = Vec::new();
    let mut indices = Vec::new();

    for y in 0..grid_size {
        for x in 0..grid_size {
            positions.push([x as f32, y as f32, 0.0]);
        }
    }

    for y in 0..(grid_size - 1) {
        for x in 0..(grid_size - 1) {
            let tl = y * grid_size + x;
            let tr = tl + 1;
            let bl = tl + grid_size;
            let br = bl + 1;

            indices.push(tl as u32);
            indices.push(bl as u32);
            indices.push(tr as u32);

            indices.push(tr as u32);
            indices.push(bl as u32);
            indices.push(br as u32);
        }
    }

    let data = MeshletData::generate(&positions, &indices, None);

    // Each meshlet must respect triangle limit
    for (i, m) in data.meshlets.iter().enumerate() {
        assert!(
            m.triangle_count as usize <= MAX_MESHLET_TRIANGLES,
            "meshlet {} has {} triangles, exceeds limit {}",
            i, m.triangle_count, MAX_MESHLET_TRIANGLES
        );
    }
}

#[test]
fn meshlet_data_total_triangles_match_input() {
    let (positions, indices) = make_large_mesh_unique_verts(100);
    let data = MeshletData::generate(&positions, &indices, None);

    let total_triangles: usize = data.meshlets.iter()
        .map(|m| m.triangle_count as usize)
        .sum();

    assert_eq!(total_triangles, 100, "total triangles must match input");
}

// ============================================================================
// SECTION 15: MeshletData Validation
// ============================================================================

#[test]
fn meshlet_data_validate_accepts_valid_data() {
    let (positions, indices) = make_cube();
    let data = MeshletData::generate(&positions, &indices, None);

    assert!(data.validate(positions.len()).is_ok());
}

#[test]
fn meshlet_data_validate_rejects_out_of_range_vertex_index() {
    let (positions, indices) = make_cube();
    let mut data = MeshletData::generate(&positions, &indices, None);

    // Corrupt a vertex index to be out of range
    if !data.vertex_indices.is_empty() {
        data.vertex_indices[0] = 999;
    }

    assert!(data.validate(positions.len()).is_err());
}

#[test]
fn meshlet_data_validate_rejects_out_of_range_local_index() {
    let (positions, indices) = make_cube();
    let mut data = MeshletData::generate(&positions, &indices, None);

    // Corrupt a local index to be out of range
    if !data.local_indices.is_empty() {
        data.local_indices[0] = 200; // Must be < vertex_count (which is <= 8)
    }

    assert!(data.validate(positions.len()).is_err());
}

#[test]
fn meshlet_data_validate_with_insufficient_vertex_count() {
    let (positions, indices) = make_cube();
    let data = MeshletData::generate(&positions, &indices, None);

    // Validate with fewer vertices than actually referenced
    assert!(data.validate(4).is_err(), "should fail when vertex_count is too small");
}

// ============================================================================
// SECTION 16: MeshletData Reconstruction
// ============================================================================

#[test]
fn meshlet_data_reconstruct_cube_indices() {
    let (positions, indices) = make_cube();
    let data = MeshletData::generate(&positions, &indices, None);
    let reconstructed = data.reconstruct_indices();

    assert_eq!(reconstructed.len(), indices.len(), "reconstructed must have same length");

    // Verify all indices are valid
    for idx in &reconstructed {
        assert!((*idx as usize) < positions.len(), "reconstructed index must be valid");
    }
}

#[test]
fn meshlet_data_reconstruct_preserves_triangles() {
    let (positions, indices) = make_cube();
    let data = MeshletData::generate(&positions, &indices, None);
    let reconstructed = data.reconstruct_indices();

    // Convert to sorted triangle sets for comparison
    let orig_tris: HashSet<[u32; 3]> = indices
        .chunks(3)
        .map(|c| {
            let mut t = [c[0], c[1], c[2]];
            t.sort();
            t
        })
        .collect();

    let recon_tris: HashSet<[u32; 3]> = reconstructed
        .chunks(3)
        .map(|c| {
            let mut t = [c[0], c[1], c[2]];
            t.sort();
            t
        })
        .collect();

    assert_eq!(orig_tris, recon_tris, "reconstructed must have same triangles");
}

#[test]
fn meshlet_data_reconstruct_large_mesh() {
    let (positions, indices) = make_large_mesh_unique_verts(50);
    let data = MeshletData::generate(&positions, &indices, None);
    let reconstructed = data.reconstruct_indices();

    assert_eq!(reconstructed.len(), indices.len());

    let orig_tris: HashSet<[u32; 3]> = indices
        .chunks(3)
        .map(|c| {
            let mut t = [c[0], c[1], c[2]];
            t.sort();
            t
        })
        .collect();

    let recon_tris: HashSet<[u32; 3]> = reconstructed
        .chunks(3)
        .map(|c| {
            let mut t = [c[0], c[1], c[2]];
            t.sort();
            t
        })
        .collect();

    assert_eq!(orig_tris, recon_tris);
}

// ============================================================================
// SECTION 17: MeshletData Bounds Computation
// ============================================================================

fn distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let dz = b[2] - a[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

#[test]
fn meshlet_data_bounds_contains_all_vertices() {
    let (positions, indices) = make_cube();
    let data = MeshletData::generate(&positions, &indices, None);

    assert!(!data.bounds.is_empty());
    let b = &data.bounds[0];

    // All meshlet vertices should be within or on the bounding sphere
    for idx in &data.vertex_indices {
        let p = positions[*idx as usize];
        let dist = distance(b.center, p);
        assert!(
            dist <= b.radius + 1e-5,
            "vertex at {:?} outside bounding sphere (dist={}, radius={})",
            p, dist, b.radius
        );
    }
}

#[test]
fn meshlet_data_bounds_count_matches_meshlets() {
    let (positions, indices) = make_large_mesh_unique_verts(100);
    let data = MeshletData::generate(&positions, &indices, None);

    assert_eq!(
        data.meshlets.len(),
        data.bounds.len(),
        "bounds count must match meshlets count"
    );
}

// ============================================================================
// SECTION 18: MeshletData with Normals
// ============================================================================

#[test]
fn meshlet_data_generate_with_normals() {
    let positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 1.0],
    ];
    let normals = vec![
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ];
    let indices = vec![0, 1, 2, 0, 2, 3];

    let data = MeshletData::generate(&positions, &indices, Some(&normals));

    assert_eq!(data.meshlet_count(), 1);
    assert!(data.bounds[0].has_valid_cone(), "with normals should have valid cone");
    // With all normals pointing +Z, cone should be tight (cutoff near 1.0)
    assert!(data.bounds[0].cone_cutoff > 0.9, "aligned normals should give tight cone");
}

#[test]
fn meshlet_data_generate_without_normals_still_works() {
    let (positions, indices) = make_cube();
    let data = MeshletData::generate(&positions, &indices, None);

    assert_eq!(data.meshlet_count(), 1);
    // Should compute normals from geometry
    // Cube has all 6 directions, cone should be wide
    // Note: cone may or may not be valid depending on implementation
}

// ============================================================================
// SECTION 19: Edge Cases - Degenerate Triangles
// ============================================================================

#[test]
fn meshlet_data_skips_degenerate_triangles() {
    let positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    // First triangle is degenerate (two identical indices)
    // Second triangle is valid
    let indices = vec![0, 0, 1, 0, 1, 2];

    let data = MeshletData::generate(&positions, &indices, None);

    assert_eq!(data.meshlet_count(), 1);
    assert_eq!(data.meshlets[0].triangle_count, 1, "degenerate triangle should be skipped");
}

#[test]
fn meshlet_data_skips_out_of_bounds_indices() {
    let positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    // First triangle has out-of-bounds index (100)
    // Second triangle is valid
    let indices = vec![0, 1, 100, 0, 1, 2];

    let data = MeshletData::generate(&positions, &indices, None);

    assert_eq!(data.meshlet_count(), 1);
    assert_eq!(data.meshlets[0].triangle_count, 1, "out-of-bounds triangle should be skipped");
}

#[test]
fn meshlet_data_all_degenerate_produces_empty() {
    let positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    // All degenerate triangles
    let indices = vec![0, 0, 1, 1, 1, 2, 2, 2, 0];

    let data = MeshletData::generate(&positions, &indices, None);

    // All triangles degenerate = empty result
    assert!(data.is_empty() || data.meshlets.iter().all(|m| m.triangle_count == 0));
}

// ============================================================================
// SECTION 20: MeshletData new() Constructor
// ============================================================================

#[test]
fn meshlet_data_new_creates_empty() {
    let data = MeshletData::new();

    assert!(data.is_empty());
    assert_eq!(data.meshlet_count(), 0);
    assert!(data.meshlets.is_empty());
    assert!(data.bounds.is_empty());
    assert!(data.vertex_indices.is_empty());
    assert!(data.local_indices.is_empty());
}

// ============================================================================
// SECTION 21: Stress Tests
// ============================================================================

#[test]
fn meshlet_data_stress_many_triangles() {
    let (positions, indices) = make_large_mesh_unique_verts(1000);
    let data = MeshletData::generate(&positions, &indices, None);

    assert!(data.meshlet_count() > 10, "1000 triangles should create many meshlets");
    assert!(data.validate(positions.len()).is_ok());

    let reconstructed = data.reconstruct_indices();
    assert_eq!(reconstructed.len(), indices.len());
}

#[test]
fn meshlet_data_stress_high_vertex_reuse() {
    // Grid mesh: many triangles sharing vertices
    let grid_size = 50;
    let mut positions = Vec::new();
    let mut indices = Vec::new();

    for y in 0..grid_size {
        for x in 0..grid_size {
            positions.push([x as f32, y as f32, 0.0]);
        }
    }

    for y in 0..(grid_size - 1) {
        for x in 0..(grid_size - 1) {
            let tl = y * grid_size + x;
            let tr = tl + 1;
            let bl = tl + grid_size;
            let br = bl + 1;

            indices.push(tl as u32);
            indices.push(bl as u32);
            indices.push(tr as u32);

            indices.push(tr as u32);
            indices.push(bl as u32);
            indices.push(br as u32);
        }
    }

    let data = MeshletData::generate(&positions, &indices, None);

    assert!(data.validate(positions.len()).is_ok());

    // Count total triangles
    let total_triangles: usize = data.meshlets.iter()
        .map(|m| m.triangle_count as usize)
        .sum();
    assert_eq!(total_triangles, indices.len() / 3);
}

// ============================================================================
// SECTION 22: Clone for MeshletData
// ============================================================================

#[test]
fn meshlet_data_clone_produces_independent_copy() {
    let (positions, indices) = make_cube();
    let original = MeshletData::generate(&positions, &indices, None);
    let mut cloned = original.clone();

    // Verify clone has same data
    assert_eq!(original.meshlet_count(), cloned.meshlet_count());
    assert_eq!(original.vertex_indices.len(), cloned.vertex_indices.len());

    // Modify clone - original should be unaffected
    if !cloned.vertex_indices.is_empty() {
        cloned.vertex_indices[0] = 999;
    }

    assert_ne!(original.vertex_indices[0], 999, "original should be unaffected by clone modification");
}

// ============================================================================
// SECTION 23: Edge Cases - Very Small Meshes
// ============================================================================

#[test]
fn meshlet_data_single_vertex_no_indices() {
    let positions = vec![[0.0, 0.0, 0.0]];
    let data = MeshletData::generate(&positions, &[], None);

    assert!(data.is_empty());
}

#[test]
fn meshlet_data_two_vertices_incomplete_triangle() {
    let _positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    // Only 2 indices - not a complete triangle
    // generate expects multiple of 3, will panic if not
    // This test documents the edge case but does not call generate
    // to avoid panic from assertion
}

// ============================================================================
// SECTION 24: Constant Verification
// ============================================================================

#[test]
fn module_constants_are_correct() {
    assert_eq!(MAX_MESHLET_VERTICES, 64, "MAX_MESHLET_VERTICES must be 64");
    assert_eq!(MAX_MESHLET_TRIANGLES, 124, "MAX_MESHLET_TRIANGLES must be 124");
    assert_eq!(MESHLET_SIZE, 12, "MESHLET_SIZE must be 12 bytes");
    assert_eq!(MESHLET_BOUNDS_SIZE, 32, "MESHLET_BOUNDS_SIZE must be 32 bytes");
}

#[test]
fn meshlet_max_constants_are_u8_compatible() {
    assert!(MAX_MESHLET_VERTICES <= 255, "MAX_VERTICES must fit in u8");
    assert!(MAX_MESHLET_TRIANGLES <= 255, "MAX_TRIANGLES must fit in u8");
}
