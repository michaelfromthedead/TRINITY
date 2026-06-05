//! Whitebox tests for T-WGPU-P6.9.1 - Meshlet Struct
//!
//! This module provides comprehensive whitebox testing for the meshlet
//! data structures used in GPU-driven rendering. Tests cover:
//!
//! - Struct layout and alignment verification
//! - bytemuck trait compliance (Pod, Zeroable)
//! - Associated constants
//! - Builder method functionality
//! - Edge cases and boundary conditions
//!
//! # Test Categories
//!
//! 1. **Struct Layout Tests** - Size and alignment assertions
//! 2. **Bytemuck Tests** - Pod/Zeroable trait verification
//! 3. **Constants Tests** - MAX_VERTICES, MAX_TRIANGLES
//! 4. **Builder Tests** - with_bounds(), with_cone()
//! 5. **Edge Case Tests** - Zero, max, negative values

use std::mem;

use bytemuck::{Pod, Zeroable};
use renderer_backend::gpu_driven::meshlet::{
    Meshlet, MeshletBounds, MeshletData,
    MAX_MESHLET_VERTICES, MAX_MESHLET_TRIANGLES,
    MESHLET_SIZE, MESHLET_BOUNDS_SIZE,
};

// ===========================================================================
// 1. STRUCT LAYOUT TESTS
// ===========================================================================

mod struct_layout {
    use super::*;

    // -----------------------------------------------------------------------
    // Meshlet size assertions
    // -----------------------------------------------------------------------

    #[test]
    fn meshlet_size_is_12_bytes() {
        assert_eq!(
            mem::size_of::<Meshlet>(),
            12,
            "Meshlet must be exactly 12 bytes for GPU buffer compatibility"
        );
    }

    #[test]
    fn meshlet_size_matches_constant() {
        assert_eq!(
            mem::size_of::<Meshlet>(),
            MESHLET_SIZE,
            "Meshlet size must match MESHLET_SIZE constant"
        );
    }

    #[test]
    fn meshlet_alignment_is_4_bytes() {
        assert_eq!(
            mem::align_of::<Meshlet>(),
            4,
            "Meshlet must be 4-byte aligned for u32 field access"
        );
    }

    // -----------------------------------------------------------------------
    // MeshletBounds size assertions
    // -----------------------------------------------------------------------

    #[test]
    fn meshlet_bounds_size_is_32_bytes() {
        assert_eq!(
            mem::size_of::<MeshletBounds>(),
            32,
            "MeshletBounds must be exactly 32 bytes (2x vec4)"
        );
    }

    #[test]
    fn meshlet_bounds_size_matches_constant() {
        assert_eq!(
            mem::size_of::<MeshletBounds>(),
            MESHLET_BOUNDS_SIZE,
            "MeshletBounds size must match MESHLET_BOUNDS_SIZE constant"
        );
    }

    #[test]
    fn meshlet_bounds_alignment_is_4_bytes() {
        assert_eq!(
            mem::align_of::<MeshletBounds>(),
            4,
            "MeshletBounds must be 4-byte aligned for f32 field access"
        );
    }

    #[test]
    fn meshlet_bounds_size_is_two_vec4s() {
        // GPU-friendly: 2x vec4 (16 bytes each)
        assert_eq!(
            mem::size_of::<MeshletBounds>(),
            2 * 16,
            "MeshletBounds should be 2x vec4 (32 bytes) for GPU alignment"
        );
    }

    // -----------------------------------------------------------------------
    // Field offset verification
    // -----------------------------------------------------------------------

    #[test]
    fn meshlet_field_offsets_match_layout() {
        // Create a meshlet with known values to verify byte layout
        let m = Meshlet::new(0x11223344, 0x55667788, 0xAA, 0xBB);
        let bytes = bytemuck::bytes_of(&m);

        // vertex_offset at offset 0 (4 bytes, little-endian on most platforms)
        let vertex_offset = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(vertex_offset, 0x11223344, "vertex_offset at offset 0");

        // triangle_offset at offset 4 (4 bytes)
        let triangle_offset = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        assert_eq!(triangle_offset, 0x55667788, "triangle_offset at offset 4");

        // vertex_count at offset 8 (1 byte)
        assert_eq!(bytes[8], 0xAA, "vertex_count at offset 8");

        // triangle_count at offset 9 (1 byte)
        assert_eq!(bytes[9], 0xBB, "triangle_count at offset 9");

        // padding at offset 10-11 (2 bytes)
        assert_eq!(bytes[10], 0, "padding[0] at offset 10");
        assert_eq!(bytes[11], 0, "padding[1] at offset 11");
    }

    #[test]
    fn meshlet_bounds_field_offsets_match_layout() {
        let b = MeshletBounds::new(
            [1.0_f32, 2.0, 3.0],  // center
            4.0,                   // radius
            [5.0, 6.0, 7.0],      // cone_axis
            8.0,                   // cone_cutoff
        );
        let bytes = bytemuck::bytes_of(&b);

        // center.x at offset 0
        let cx = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(cx, 1.0, "center.x at offset 0");

        // center.y at offset 4
        let cy = f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        assert_eq!(cy, 2.0, "center.y at offset 4");

        // center.z at offset 8
        let cz = f32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        assert_eq!(cz, 3.0, "center.z at offset 8");

        // radius at offset 12
        let r = f32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        assert_eq!(r, 4.0, "radius at offset 12");

        // cone_axis.x at offset 16
        let ax = f32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
        assert_eq!(ax, 5.0, "cone_axis.x at offset 16");

        // cone_axis.y at offset 20
        let ay = f32::from_le_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);
        assert_eq!(ay, 6.0, "cone_axis.y at offset 20");

        // cone_axis.z at offset 24
        let az = f32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]);
        assert_eq!(az, 7.0, "cone_axis.z at offset 24");

        // cone_cutoff at offset 28
        let cut = f32::from_le_bytes([bytes[28], bytes[29], bytes[30], bytes[31]]);
        assert_eq!(cut, 8.0, "cone_cutoff at offset 28");
    }

    // -----------------------------------------------------------------------
    // Array stride verification
    // -----------------------------------------------------------------------

    #[test]
    fn meshlet_array_has_no_padding() {
        let array: [Meshlet; 10] = [Meshlet::default(); 10];
        let expected_size = 10 * MESHLET_SIZE;
        assert_eq!(
            mem::size_of_val(&array),
            expected_size,
            "Meshlet array should have no inter-element padding"
        );
    }

    #[test]
    fn meshlet_bounds_array_has_no_padding() {
        let array: [MeshletBounds; 10] = [MeshletBounds::default(); 10];
        let expected_size = 10 * MESHLET_BOUNDS_SIZE;
        assert_eq!(
            mem::size_of_val(&array),
            expected_size,
            "MeshletBounds array should have no inter-element padding"
        );
    }

    #[test]
    fn vec_of_meshlets_has_correct_byte_length() {
        let meshlets = vec![Meshlet::default(); 100];
        let bytes: &[u8] = bytemuck::cast_slice(&meshlets);
        assert_eq!(bytes.len(), 100 * MESHLET_SIZE);
    }

    #[test]
    fn vec_of_bounds_has_correct_byte_length() {
        let bounds = vec![MeshletBounds::default(); 100];
        let bytes: &[u8] = bytemuck::cast_slice(&bounds);
        assert_eq!(bytes.len(), 100 * MESHLET_BOUNDS_SIZE);
    }
}

// ===========================================================================
// 2. BYTEMUCK TRAIT TESTS
// ===========================================================================

mod bytemuck_traits {
    use super::*;

    // -----------------------------------------------------------------------
    // Pod trait verification
    // -----------------------------------------------------------------------

    #[test]
    fn meshlet_implements_pod() {
        // Pod trait requires the type to be safe for byte-level manipulation
        fn assert_pod<T: Pod>() {}
        assert_pod::<Meshlet>();
    }

    #[test]
    fn meshlet_bounds_implements_pod() {
        fn assert_pod<T: Pod>() {}
        assert_pod::<MeshletBounds>();
    }

    // -----------------------------------------------------------------------
    // Zeroable trait verification
    // -----------------------------------------------------------------------

    #[test]
    fn meshlet_implements_zeroable() {
        fn assert_zeroable<T: Zeroable>() {}
        assert_zeroable::<Meshlet>();
    }

    #[test]
    fn meshlet_bounds_implements_zeroable() {
        fn assert_zeroable<T: Zeroable>() {}
        assert_zeroable::<MeshletBounds>();
    }

    #[test]
    fn meshlet_zeroed_has_all_zeros() {
        let m = Meshlet::zeroed();
        assert_eq!(m.vertex_offset, 0);
        assert_eq!(m.triangle_offset, 0);
        assert_eq!(m.vertex_count, 0);
        assert_eq!(m.triangle_count, 0);
        assert_eq!(m._padding, [0, 0]);
    }

    #[test]
    fn meshlet_bounds_zeroed_has_all_zeros() {
        let b = MeshletBounds::zeroed();
        assert_eq!(b.center, [0.0, 0.0, 0.0]);
        assert_eq!(b.radius, 0.0);
        assert_eq!(b.cone_axis, [0.0, 0.0, 0.0]);
        assert_eq!(b.cone_cutoff, 0.0);
    }

    // -----------------------------------------------------------------------
    // Bytes roundtrip (to_bytes, from_bytes)
    // -----------------------------------------------------------------------

    #[test]
    fn meshlet_bytes_roundtrip() {
        let original = Meshlet::new(12345, 67890, 42, 99);

        // Convert to bytes
        let bytes: &[u8] = bytemuck::bytes_of(&original);
        assert_eq!(bytes.len(), MESHLET_SIZE);

        // Convert back
        let restored: &Meshlet = bytemuck::from_bytes(bytes);

        assert_eq!(restored.vertex_offset, original.vertex_offset);
        assert_eq!(restored.triangle_offset, original.triangle_offset);
        assert_eq!(restored.vertex_count, original.vertex_count);
        assert_eq!(restored.triangle_count, original.triangle_count);
    }

    #[test]
    fn meshlet_bounds_bytes_roundtrip() {
        let original = MeshletBounds::new(
            [1.5, -2.5, 3.5],
            10.0,
            [0.0, 1.0, 0.0],
            0.707,
        );

        // Convert to bytes
        let bytes: &[u8] = bytemuck::bytes_of(&original);
        assert_eq!(bytes.len(), MESHLET_BOUNDS_SIZE);

        // Convert back
        let restored: &MeshletBounds = bytemuck::from_bytes(bytes);

        assert_eq!(restored.center, original.center);
        assert_eq!(restored.radius, original.radius);
        assert_eq!(restored.cone_axis, original.cone_axis);
        assert_eq!(restored.cone_cutoff, original.cone_cutoff);
    }

    #[test]
    fn meshlet_slice_bytes_roundtrip() {
        let originals = vec![
            Meshlet::new(0, 0, 10, 5),
            Meshlet::new(10, 15, 20, 10),
            Meshlet::new(30, 45, 15, 8),
        ];

        // Convert slice to bytes
        let bytes: &[u8] = bytemuck::cast_slice(&originals);
        assert_eq!(bytes.len(), 3 * MESHLET_SIZE);

        // Convert back to slice
        let restored: &[Meshlet] = bytemuck::cast_slice(bytes);
        assert_eq!(restored.len(), 3);

        for (i, (orig, rest)) in originals.iter().zip(restored.iter()).enumerate() {
            assert_eq!(
                orig.vertex_offset, rest.vertex_offset,
                "vertex_offset mismatch at index {}", i
            );
            assert_eq!(
                orig.triangle_offset, rest.triangle_offset,
                "triangle_offset mismatch at index {}", i
            );
            assert_eq!(
                orig.vertex_count, rest.vertex_count,
                "vertex_count mismatch at index {}", i
            );
            assert_eq!(
                orig.triangle_count, rest.triangle_count,
                "triangle_count mismatch at index {}", i
            );
        }
    }

    #[test]
    fn meshlet_bounds_slice_bytes_roundtrip() {
        let originals = vec![
            MeshletBounds::new([0.0, 0.0, 0.0], 1.0, [1.0, 0.0, 0.0], 0.5),
            MeshletBounds::new([5.0, 5.0, 5.0], 2.5, [0.0, 1.0, 0.0], 0.8),
        ];

        let bytes: &[u8] = bytemuck::cast_slice(&originals);
        assert_eq!(bytes.len(), 2 * MESHLET_BOUNDS_SIZE);

        let restored: &[MeshletBounds] = bytemuck::cast_slice(bytes);
        assert_eq!(restored.len(), 2);

        for (i, (orig, rest)) in originals.iter().zip(restored.iter()).enumerate() {
            assert_eq!(orig.center, rest.center, "center mismatch at index {}", i);
            assert_eq!(orig.radius, rest.radius, "radius mismatch at index {}", i);
            assert_eq!(orig.cone_axis, rest.cone_axis, "cone_axis mismatch at index {}", i);
            assert_eq!(orig.cone_cutoff, rest.cone_cutoff, "cone_cutoff mismatch at index {}", i);
        }
    }

    #[test]
    fn meshlet_try_from_bytes_invalid_size_fails() {
        let short_bytes = [0u8; 8]; // Too short
        let result: Result<&Meshlet, _> = bytemuck::try_from_bytes(&short_bytes);
        assert!(result.is_err());
    }

    #[test]
    fn meshlet_bounds_try_from_bytes_invalid_size_fails() {
        let short_bytes = [0u8; 16]; // Too short
        let result: Result<&MeshletBounds, _> = bytemuck::try_from_bytes(&short_bytes);
        assert!(result.is_err());
    }
}

// ===========================================================================
// 3. CONSTANTS TESTS
// ===========================================================================

mod constants {
    use super::*;

    #[test]
    fn max_vertices_is_64() {
        assert_eq!(
            MAX_MESHLET_VERTICES,
            64,
            "MAX_MESHLET_VERTICES must be 64 (GPU-friendly power of 2)"
        );
    }

    #[test]
    fn max_triangles_is_124() {
        assert_eq!(
            MAX_MESHLET_TRIANGLES,
            124,
            "MAX_MESHLET_TRIANGLES must be 124 (optimal for mesh shaders)"
        );
    }

    #[test]
    fn meshlet_associated_max_vertices() {
        assert_eq!(
            Meshlet::MAX_VERTICES,
            64,
            "Meshlet::MAX_VERTICES must be 64"
        );
    }

    #[test]
    fn meshlet_associated_max_triangles() {
        assert_eq!(
            Meshlet::MAX_TRIANGLES,
            124,
            "Meshlet::MAX_TRIANGLES must be 124"
        );
    }

    #[test]
    fn associated_constants_match_module_constants() {
        assert_eq!(
            Meshlet::MAX_VERTICES as usize,
            MAX_MESHLET_VERTICES,
            "Associated constant must match module constant"
        );
        assert_eq!(
            Meshlet::MAX_TRIANGLES as usize,
            MAX_MESHLET_TRIANGLES,
            "Associated constant must match module constant"
        );
    }

    #[test]
    fn meshlet_size_constant_is_12() {
        assert_eq!(MESHLET_SIZE, 12);
    }

    #[test]
    fn meshlet_bounds_size_constant_is_32() {
        assert_eq!(MESHLET_BOUNDS_SIZE, 32);
    }

    #[test]
    fn max_vertices_fits_in_u8() {
        // vertex_count is u8, so MAX_VERTICES must be <= 255
        assert!(MAX_MESHLET_VERTICES <= u8::MAX as usize);
    }

    #[test]
    fn max_triangles_fits_in_u8() {
        // triangle_count is u8, so MAX_TRIANGLES must be <= 255
        assert!(MAX_MESHLET_TRIANGLES <= u8::MAX as usize);
    }

    #[test]
    fn max_local_indices_fits_in_u8() {
        // Local indices reference vertices 0..vertex_count
        // So MAX_VERTICES - 1 must fit in u8
        assert!(MAX_MESHLET_VERTICES - 1 <= u8::MAX as usize);
    }
}

// ===========================================================================
// 4. BUILDER METHOD TESTS
// ===========================================================================

mod builder_methods {
    use super::*;

    // -----------------------------------------------------------------------
    // with_bounds() tests
    // -----------------------------------------------------------------------

    #[test]
    fn with_bounds_sets_center() {
        let bounds = MeshletBounds::default()
            .with_bounds([1.0, 2.0, 3.0], 5.0);

        assert_eq!(bounds.center, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn with_bounds_sets_radius() {
        let bounds = MeshletBounds::default()
            .with_bounds([0.0, 0.0, 0.0], 10.5);

        assert_eq!(bounds.radius, 10.5);
    }

    #[test]
    fn with_bounds_preserves_cone_values() {
        let bounds = MeshletBounds::new(
            [0.0, 0.0, 0.0],
            1.0,
            [0.0, 1.0, 0.0],
            0.5,
        ).with_bounds([5.0, 5.0, 5.0], 20.0);

        // Center and radius should be updated
        assert_eq!(bounds.center, [5.0, 5.0, 5.0]);
        assert_eq!(bounds.radius, 20.0);

        // Cone values should be preserved
        assert_eq!(bounds.cone_axis, [0.0, 1.0, 0.0]);
        assert_eq!(bounds.cone_cutoff, 0.5);
    }

    // -----------------------------------------------------------------------
    // with_cone() tests
    // -----------------------------------------------------------------------

    #[test]
    fn with_cone_sets_axis() {
        let bounds = MeshletBounds::default()
            .with_cone([0.0, 0.0, 1.0], 0.5);

        assert_eq!(bounds.cone_axis, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn with_cone_sets_cutoff() {
        let bounds = MeshletBounds::default()
            .with_cone([1.0, 0.0, 0.0], 0.707);

        assert_eq!(bounds.cone_cutoff, 0.707);
    }

    #[test]
    fn with_cone_preserves_bounds_values() {
        let bounds = MeshletBounds::new(
            [10.0, 20.0, 30.0],
            50.0,
            [0.0, 0.0, 1.0],
            -1.0,
        ).with_cone([1.0, 0.0, 0.0], 0.9);

        // Bounds should be preserved
        assert_eq!(bounds.center, [10.0, 20.0, 30.0]);
        assert_eq!(bounds.radius, 50.0);

        // Cone values should be updated
        assert_eq!(bounds.cone_axis, [1.0, 0.0, 0.0]);
        assert_eq!(bounds.cone_cutoff, 0.9);
    }

    // -----------------------------------------------------------------------
    // Builder chaining tests
    // -----------------------------------------------------------------------

    #[test]
    fn builder_chain_sets_all_values() {
        let bounds = MeshletBounds::default()
            .with_bounds([1.0, 2.0, 3.0], 4.0)
            .with_cone([0.577, 0.577, 0.577], 0.5);

        assert_eq!(bounds.center, [1.0, 2.0, 3.0]);
        assert_eq!(bounds.radius, 4.0);
        assert_eq!(bounds.cone_axis, [0.577, 0.577, 0.577]);
        assert_eq!(bounds.cone_cutoff, 0.5);
    }

    #[test]
    fn builder_chain_order_does_not_matter() {
        let bounds1 = MeshletBounds::default()
            .with_bounds([1.0, 2.0, 3.0], 4.0)
            .with_cone([0.0, 1.0, 0.0], 0.5);

        let bounds2 = MeshletBounds::default()
            .with_cone([0.0, 1.0, 0.0], 0.5)
            .with_bounds([1.0, 2.0, 3.0], 4.0);

        assert_eq!(bounds1.center, bounds2.center);
        assert_eq!(bounds1.radius, bounds2.radius);
        assert_eq!(bounds1.cone_axis, bounds2.cone_axis);
        assert_eq!(bounds1.cone_cutoff, bounds2.cone_cutoff);
    }

    #[test]
    fn builder_can_override_values() {
        let bounds = MeshletBounds::default()
            .with_bounds([1.0, 1.0, 1.0], 1.0)
            .with_bounds([2.0, 2.0, 2.0], 2.0)  // Override
            .with_cone([0.0, 1.0, 0.0], 0.5)
            .with_cone([1.0, 0.0, 0.0], 0.9);   // Override

        assert_eq!(bounds.center, [2.0, 2.0, 2.0]);
        assert_eq!(bounds.radius, 2.0);
        assert_eq!(bounds.cone_axis, [1.0, 0.0, 0.0]);
        assert_eq!(bounds.cone_cutoff, 0.9);
    }

    #[test]
    fn builder_with_bounds_returns_valid_cone_check() {
        let bounds = MeshletBounds::default()
            .with_bounds([0.0, 0.0, 0.0], 1.0)
            .with_cone([0.0, 1.0, 0.0], 0.5);

        assert!(bounds.has_valid_cone());
    }

    #[test]
    fn builder_can_disable_cone_culling() {
        let bounds = MeshletBounds::default()
            .with_bounds([0.0, 0.0, 0.0], 1.0)
            .with_cone([0.0, 0.0, 1.0], -2.0);  // -2.0 disables culling

        assert!(!bounds.has_valid_cone());
    }
}

// ===========================================================================
// 5. EDGE CASE TESTS
// ===========================================================================

mod edge_cases {
    use super::*;

    // -----------------------------------------------------------------------
    // Zero values
    // -----------------------------------------------------------------------

    #[test]
    fn meshlet_with_zero_values() {
        let m = Meshlet::new(0, 0, 0, 0);

        assert_eq!(m.vertex_offset, 0);
        assert_eq!(m.triangle_offset, 0);
        assert_eq!(m.vertex_count, 0);
        assert_eq!(m.triangle_count, 0);
        assert!(m.is_empty());
    }

    #[test]
    fn meshlet_bounds_with_zero_center() {
        let b = MeshletBounds::new([0.0, 0.0, 0.0], 1.0, [0.0, 1.0, 0.0], 0.5);

        assert_eq!(b.center, [0.0, 0.0, 0.0]);
        assert!(b.has_valid_cone());
    }

    #[test]
    fn meshlet_bounds_with_zero_radius() {
        let b = MeshletBounds::new([1.0, 2.0, 3.0], 0.0, [0.0, 1.0, 0.0], 0.5);

        assert_eq!(b.radius, 0.0);
        // Point sphere is valid
    }

    #[test]
    fn meshlet_bounds_with_zero_cone_axis() {
        // Zero axis is technically invalid but should not crash
        let b = MeshletBounds::new([0.0, 0.0, 0.0], 1.0, [0.0, 0.0, 0.0], 0.0);

        assert_eq!(b.cone_axis, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn meshlet_bounds_with_zero_cutoff() {
        // Cutoff of 0 means 90 degree cone (perpendicular)
        let b = MeshletBounds::new([0.0, 0.0, 0.0], 1.0, [0.0, 1.0, 0.0], 0.0);

        assert_eq!(b.cone_cutoff, 0.0);
        assert!(b.has_valid_cone());
    }

    // -----------------------------------------------------------------------
    // Max u32/u8 values
    // -----------------------------------------------------------------------

    #[test]
    fn meshlet_with_max_u32_offsets() {
        let m = Meshlet::new(u32::MAX, u32::MAX, 64, 124);

        assert_eq!(m.vertex_offset, u32::MAX);
        assert_eq!(m.triangle_offset, u32::MAX);
    }

    #[test]
    fn meshlet_with_max_u8_counts() {
        let m = Meshlet::new(0, 0, u8::MAX, u8::MAX);

        assert_eq!(m.vertex_count, 255);
        assert_eq!(m.triangle_count, 255);
    }

    #[test]
    fn meshlet_with_max_valid_counts() {
        let m = Meshlet::new(0, 0, Meshlet::MAX_VERTICES, Meshlet::MAX_TRIANGLES);

        assert_eq!(m.vertex_count, 64);
        assert_eq!(m.triangle_count, 124);
        assert!(!m.is_empty());
    }

    #[test]
    fn meshlet_bytes_roundtrip_with_max_values() {
        let original = Meshlet::new(u32::MAX, u32::MAX, u8::MAX, u8::MAX);
        let bytes = bytemuck::bytes_of(&original);
        let restored: &Meshlet = bytemuck::from_bytes(bytes);

        assert_eq!(restored.vertex_offset, u32::MAX);
        assert_eq!(restored.triangle_offset, u32::MAX);
        assert_eq!(restored.vertex_count, u8::MAX);
        assert_eq!(restored.triangle_count, u8::MAX);
    }

    // -----------------------------------------------------------------------
    // Negative coordinates (center, axis)
    // -----------------------------------------------------------------------

    #[test]
    fn meshlet_bounds_with_negative_center() {
        let b = MeshletBounds::new([-100.0, -200.0, -300.0], 50.0, [0.0, 1.0, 0.0], 0.5);

        assert_eq!(b.center, [-100.0, -200.0, -300.0]);
    }

    #[test]
    fn meshlet_bounds_with_negative_axis() {
        let b = MeshletBounds::new([0.0, 0.0, 0.0], 1.0, [-1.0, 0.0, 0.0], 0.5);

        assert_eq!(b.cone_axis, [-1.0, 0.0, 0.0]);
    }

    #[test]
    fn meshlet_bounds_with_negative_cutoff() {
        // Negative cutoff means cone spans more than 90 degrees
        let b = MeshletBounds::new([0.0, 0.0, 0.0], 1.0, [0.0, 1.0, 0.0], -0.5);

        assert_eq!(b.cone_cutoff, -0.5);
        assert!(b.has_valid_cone()); // -0.5 > -1.0, so still valid
    }

    #[test]
    fn meshlet_bounds_with_cutoff_negative_one() {
        // Cutoff of -1.0 means cone spans full hemisphere
        let b = MeshletBounds::new([0.0, 0.0, 0.0], 1.0, [0.0, 1.0, 0.0], -1.0);

        assert_eq!(b.cone_cutoff, -1.0);
        assert!(b.has_valid_cone()); // Exactly -1.0 is still valid
    }

    #[test]
    fn meshlet_bounds_with_cutoff_below_negative_one() {
        // Cutoff below -1.0 disables cone culling
        let b = MeshletBounds::new([0.0, 0.0, 0.0], 1.0, [0.0, 1.0, 0.0], -1.5);

        assert!(!b.has_valid_cone());
    }

    #[test]
    fn meshlet_bounds_bytes_roundtrip_with_negative_values() {
        let original = MeshletBounds::new(
            [-f32::MAX, -f32::MAX, -f32::MAX],
            f32::MAX,
            [-1.0, -1.0, -1.0],
            -1.0,
        );

        let bytes = bytemuck::bytes_of(&original);
        let restored: &MeshletBounds = bytemuck::from_bytes(bytes);

        assert_eq!(restored.center, original.center);
        assert_eq!(restored.radius, original.radius);
        assert_eq!(restored.cone_axis, original.cone_axis);
        assert_eq!(restored.cone_cutoff, original.cone_cutoff);
    }

    // -----------------------------------------------------------------------
    // Special float values
    // -----------------------------------------------------------------------

    #[test]
    fn meshlet_bounds_with_large_coordinates() {
        let big = 1e30_f32;
        let b = MeshletBounds::new([big, big, big], big, [0.0, 1.0, 0.0], 0.5);

        assert_eq!(b.center[0], big);
        assert_eq!(b.radius, big);
    }

    #[test]
    fn meshlet_bounds_with_small_coordinates() {
        let tiny = 1e-30_f32;
        let b = MeshletBounds::new([tiny, tiny, tiny], tiny, [0.0, 1.0, 0.0], 0.5);

        assert_eq!(b.center[0], tiny);
        assert_eq!(b.radius, tiny);
    }

    #[test]
    fn meshlet_bounds_with_subnormal_values() {
        // Subnormal (denormalized) float
        let subnormal = f32::MIN_POSITIVE / 2.0;
        let b = MeshletBounds::new([subnormal, 0.0, 0.0], subnormal, [0.0, 1.0, 0.0], 0.5);

        // Subnormal should survive roundtrip
        let bytes = bytemuck::bytes_of(&b);
        let restored: &MeshletBounds = bytemuck::from_bytes(bytes);
        assert_eq!(restored.center[0], subnormal);
    }

    // -----------------------------------------------------------------------
    // is_empty() edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn meshlet_is_empty_with_zero_triangles() {
        assert!(Meshlet::new(100, 200, 50, 0).is_empty());
    }

    #[test]
    fn meshlet_is_empty_with_nonzero_triangles() {
        assert!(!Meshlet::new(0, 0, 1, 1).is_empty());
    }

    #[test]
    fn meshlet_is_empty_with_vertices_but_no_triangles() {
        // Vertices but no triangles is considered empty
        assert!(Meshlet::new(0, 0, 64, 0).is_empty());
    }

    // -----------------------------------------------------------------------
    // has_valid_cone() edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn has_valid_cone_boundary_cases() {
        // Exactly -1.0 is valid
        assert!(MeshletBounds::new([0.0; 3], 1.0, [0.0, 0.0, 1.0], -1.0).has_valid_cone());

        // Just below -1.0 is invalid
        assert!(!MeshletBounds::new([0.0; 3], 1.0, [0.0, 0.0, 1.0], -1.0001).has_valid_cone());

        // Cutoff of 1.0 (single ray) is valid
        assert!(MeshletBounds::new([0.0; 3], 1.0, [0.0, 0.0, 1.0], 1.0).has_valid_cone());

        // Cutoff of 0.0 (90 degree cone) is valid
        assert!(MeshletBounds::new([0.0; 3], 1.0, [0.0, 0.0, 1.0], 0.0).has_valid_cone());
    }

    #[test]
    fn sphere_only_has_invalid_cone() {
        let b = MeshletBounds::sphere_only([1.0, 2.0, 3.0], 5.0);
        assert!(!b.has_valid_cone());
        assert_eq!(b.cone_cutoff, -2.0);
    }
}

// ===========================================================================
// 6. MESHLET DATA TESTS
// ===========================================================================

mod meshlet_data {
    use super::*;

    #[test]
    fn new_creates_empty_data() {
        let data = MeshletData::new();

        assert!(data.is_empty());
        assert_eq!(data.meshlet_count(), 0);
        assert!(data.meshlets.is_empty());
        assert!(data.bounds.is_empty());
        assert!(data.vertex_indices.is_empty());
        assert!(data.local_indices.is_empty());
    }

    #[test]
    fn default_creates_empty_data() {
        let data = MeshletData::default();

        assert!(data.is_empty());
        assert_eq!(data.meshlet_count(), 0);
    }

    #[test]
    fn generate_empty_positions_returns_empty() {
        let data = MeshletData::generate(&[], &[], None);
        assert!(data.is_empty());
    }

    #[test]
    fn generate_empty_indices_returns_empty() {
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let data = MeshletData::generate(&positions, &[], None);
        assert!(data.is_empty());
    }

    #[test]
    fn generate_single_triangle() {
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let indices = vec![0, 1, 2];

        let data = MeshletData::generate(&positions, &indices, None);

        assert_eq!(data.meshlet_count(), 1);
        assert_eq!(data.meshlets[0].vertex_count, 3);
        assert_eq!(data.meshlets[0].triangle_count, 1);
        assert!(data.validate(positions.len()).is_ok());
    }

    #[test]
    fn meshlet_data_bounds_match_meshlets() {
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let indices = vec![0, 1, 2];

        let data = MeshletData::generate(&positions, &indices, None);

        assert_eq!(data.meshlets.len(), data.bounds.len());
    }

    #[test]
    fn is_empty_returns_true_for_no_meshlets() {
        let data = MeshletData::new();
        assert!(data.is_empty());
    }

    #[test]
    fn is_empty_returns_false_for_meshlets() {
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let indices = vec![0, 1, 2];

        let data = MeshletData::generate(&positions, &indices, None);
        assert!(!data.is_empty());
    }
}

// ===========================================================================
// 7. CONSTRUCTOR TESTS
// ===========================================================================

mod constructors {
    use super::*;

    #[test]
    fn meshlet_new_sets_all_fields() {
        let m = Meshlet::new(100, 200, 32, 50);

        assert_eq!(m.vertex_offset, 100);
        assert_eq!(m.triangle_offset, 200);
        assert_eq!(m.vertex_count, 32);
        assert_eq!(m.triangle_count, 50);
        assert_eq!(m._padding, [0, 0]);
    }

    #[test]
    fn meshlet_default_is_zeroed() {
        let m = Meshlet::default();

        assert_eq!(m.vertex_offset, 0);
        assert_eq!(m.triangle_offset, 0);
        assert_eq!(m.vertex_count, 0);
        assert_eq!(m.triangle_count, 0);
        assert_eq!(m._padding, [0, 0]);
    }

    #[test]
    fn meshlet_bounds_new_sets_all_fields() {
        let b = MeshletBounds::new(
            [1.0, 2.0, 3.0],
            4.0,
            [0.0, 1.0, 0.0],
            0.5,
        );

        assert_eq!(b.center, [1.0, 2.0, 3.0]);
        assert_eq!(b.radius, 4.0);
        assert_eq!(b.cone_axis, [0.0, 1.0, 0.0]);
        assert_eq!(b.cone_cutoff, 0.5);
    }

    #[test]
    fn meshlet_bounds_sphere_only_sets_sphere_and_disables_cone() {
        let b = MeshletBounds::sphere_only([1.0, 2.0, 3.0], 5.0);

        assert_eq!(b.center, [1.0, 2.0, 3.0]);
        assert_eq!(b.radius, 5.0);
        assert_eq!(b.cone_axis, [0.0, 0.0, 1.0]); // Default axis
        assert_eq!(b.cone_cutoff, -2.0); // Disabled
        assert!(!b.has_valid_cone());
    }

    #[test]
    fn meshlet_bounds_default_is_zeroed() {
        let b = MeshletBounds::default();

        assert_eq!(b.center, [0.0, 0.0, 0.0]);
        assert_eq!(b.radius, 0.0);
        assert_eq!(b.cone_axis, [0.0, 0.0, 0.0]);
        assert_eq!(b.cone_cutoff, 0.0);
    }
}

// ===========================================================================
// 8. TRAIT IMPLEMENTATIONS
// ===========================================================================

mod traits {
    use super::*;

    #[test]
    fn meshlet_implements_clone() {
        let m = Meshlet::new(1, 2, 3, 4);
        let cloned = m.clone();

        assert_eq!(m.vertex_offset, cloned.vertex_offset);
        assert_eq!(m.triangle_offset, cloned.triangle_offset);
        assert_eq!(m.vertex_count, cloned.vertex_count);
        assert_eq!(m.triangle_count, cloned.triangle_count);
    }

    #[test]
    fn meshlet_implements_copy() {
        let m = Meshlet::new(1, 2, 3, 4);
        let copied = m; // Copy, not move

        // Original still usable
        assert_eq!(m.vertex_offset, 1);
        assert_eq!(copied.vertex_offset, 1);
    }

    #[test]
    fn meshlet_implements_debug() {
        let m = Meshlet::new(1, 2, 3, 4);
        let debug_str = format!("{:?}", m);

        assert!(debug_str.contains("Meshlet"));
        assert!(debug_str.contains("vertex_offset"));
    }

    #[test]
    fn meshlet_bounds_implements_clone() {
        let b = MeshletBounds::new([1.0, 2.0, 3.0], 4.0, [0.0, 1.0, 0.0], 0.5);
        let cloned = b.clone();

        assert_eq!(b.center, cloned.center);
        assert_eq!(b.radius, cloned.radius);
        assert_eq!(b.cone_axis, cloned.cone_axis);
        assert_eq!(b.cone_cutoff, cloned.cone_cutoff);
    }

    #[test]
    fn meshlet_bounds_implements_copy() {
        let b = MeshletBounds::new([1.0, 2.0, 3.0], 4.0, [0.0, 1.0, 0.0], 0.5);
        let copied = b;

        assert_eq!(b.center, [1.0, 2.0, 3.0]);
        assert_eq!(copied.center, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn meshlet_bounds_implements_debug() {
        let b = MeshletBounds::new([1.0, 2.0, 3.0], 4.0, [0.0, 1.0, 0.0], 0.5);
        let debug_str = format!("{:?}", b);

        assert!(debug_str.contains("MeshletBounds"));
        assert!(debug_str.contains("center"));
    }

    #[test]
    fn meshlet_data_implements_clone() {
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let indices = vec![0, 1, 2];
        let data = MeshletData::generate(&positions, &indices, None);

        let cloned = data.clone();

        assert_eq!(data.meshlet_count(), cloned.meshlet_count());
        assert_eq!(data.meshlets.len(), cloned.meshlets.len());
        assert_eq!(data.bounds.len(), cloned.bounds.len());
    }

    #[test]
    fn meshlet_data_implements_debug() {
        let data = MeshletData::new();
        let debug_str = format!("{:?}", data);

        assert!(debug_str.contains("MeshletData"));
    }

    #[test]
    fn meshlet_data_implements_default() {
        let data: MeshletData = Default::default();
        assert!(data.is_empty());
    }
}
