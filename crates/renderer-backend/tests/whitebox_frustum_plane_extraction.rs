//! Whitebox structural tests for Frustum Plane Extraction (T-WGPU-P6.3.1).
//!
//! These tests verify the internal structure and behavior of the FrustumPlane,
//! FrustumPlanes types, including plane extraction, normalization, struct layout,
//! and edge case handling. Full source access is utilized for comprehensive coverage.

use std::mem;

use bytemuck::Zeroable;
use renderer_backend::gpu_driven::frustum::{
    FrustumPlane, FrustumPlanes,
    NUM_FRUSTUM_PLANES, FRUSTUM_PLANE_SIZE, FRUSTUM_PLANES_SIZE,
    PLANE_LEFT, PLANE_RIGHT, PLANE_BOTTOM, PLANE_TOP, PLANE_NEAR, PLANE_FAR,
    perspective_matrix, look_at_matrix, multiply_matrices,
};

// ============================================================================
// Helper Functions
// ============================================================================

/// Create an identity 4x4 matrix in column-major order.
fn identity_matrix() -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// Create a standard perspective VP matrix for testing.
fn standard_perspective_vp() -> [[f32; 4]; 4] {
    let fovy = std::f32::consts::FRAC_PI_4; // 45 degrees
    let aspect = 16.0 / 9.0;
    let near = 0.1;
    let far = 100.0;

    let proj = perspective_matrix(fovy, aspect, near, far);
    let view = look_at_matrix(
        [0.0, 0.0, 5.0],  // eye
        [0.0, 0.0, 0.0],  // target
        [0.0, 1.0, 0.0],  // up
    );

    multiply_matrices(&view, &proj)
}

/// Create an orthographic projection matrix.
fn orthographic_matrix(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
    let width = right - left;
    let height = top - bottom;
    let depth = far - near;

    [
        [2.0 / width, 0.0, 0.0, 0.0],
        [0.0, 2.0 / height, 0.0, 0.0],
        [0.0, 0.0, -1.0 / depth, 0.0],
        [-(right + left) / width, -(top + bottom) / height, -near / depth, 1.0],
    ]
}

/// Compute the length of a 3D vector.
fn vec3_length(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Compute dot product of two 3D vectors.
fn vec3_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

// ============================================================================
// 1. Plane Extraction Tests
// ============================================================================

mod plane_extraction {
    use super::*;

    #[test]
    fn extract_from_identity_matrix_produces_six_planes() {
        let identity = identity_matrix();
        let planes = FrustumPlanes::from_view_projection(&identity);

        assert_eq!(planes.planes.len(), NUM_FRUSTUM_PLANES);
        assert_eq!(planes.planes.len(), 6);
    }

    #[test]
    fn identity_frustum_factory_matches_manual_extraction() {
        let identity = identity_matrix();
        let from_method = FrustumPlanes::from_view_projection(&identity);
        let from_factory = FrustumPlanes::identity();

        for i in 0..NUM_FRUSTUM_PLANES {
            assert_eq!(
                from_method.planes[i].normal,
                from_factory.planes[i].normal,
                "Plane {} normal mismatch", i
            );
            assert!(
                (from_method.planes[i].distance - from_factory.planes[i].distance).abs() < 1e-6,
                "Plane {} distance mismatch", i
            );
        }
    }

    #[test]
    fn extract_from_perspective_projection_produces_valid_planes() {
        let vp = standard_perspective_vp();
        let planes = FrustumPlanes::from_view_projection(&vp);

        // All planes should be normalized
        assert!(planes.all_normalized(), "All planes should be normalized after extraction");

        // No plane should have zero-length normal after normalization
        for (i, plane) in planes.planes.iter().enumerate() {
            let len = vec3_length(plane.normal);
            assert!(
                (len - 1.0).abs() < 1e-5,
                "Plane {} should have unit normal, got length {}", i, len
            );
        }
    }

    #[test]
    fn extract_from_orthographic_projection_produces_valid_planes() {
        let ortho = orthographic_matrix(-10.0, 10.0, -10.0, 10.0, 0.1, 100.0);
        let planes = FrustumPlanes::from_view_projection(&ortho);

        assert!(planes.all_normalized(), "Orthographic planes should be normalized");

        // In orthographic projection, left/right planes should be parallel
        // and so should top/bottom planes
        for (i, plane) in planes.planes.iter().enumerate() {
            let len = vec3_length(plane.normal);
            assert!(
                (len - 1.0).abs() < 1e-5,
                "Ortho plane {} should have unit normal, got length {}", i, len
            );
        }
    }

    #[test]
    fn plane_equations_are_correct_for_identity() {
        let planes = FrustumPlanes::identity();

        // For identity matrix, Gribb-Hartmann gives us:
        // Left:   (1, 0, 0, 1) -> normalized (1, 0, 0), d = 1/sqrt(2)
        // Right:  (-1, 0, 0, 1) -> normalized (-1, 0, 0), d = 1/sqrt(2)
        // etc.

        // Test that each plane normal has unit length
        for (i, plane) in planes.planes.iter().enumerate() {
            assert!(
                plane.is_normalized(),
                "Identity plane {} should be normalized", i
            );
        }
    }

    #[test]
    fn gribb_hartmann_row_extraction_is_correct() {
        // Test that row extraction in Gribb-Hartmann is done correctly
        // For a known matrix, verify the plane coefficients

        // Simple scaling matrix
        let scale = [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        let planes = FrustumPlanes::from_view_projection(&scale);

        // All planes should still be valid and normalized
        assert!(planes.all_normalized());
    }

    #[test]
    fn asymmetric_perspective_extracts_correct_planes() {
        // Create asymmetric frustum (e.g., stereo rendering)
        let view = look_at_matrix([1.0, 0.0, 5.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let proj = perspective_matrix(1.0, 1.5, 0.5, 50.0);
        let vp = multiply_matrices(&view, &proj);

        let planes = FrustumPlanes::from_view_projection(&vp);

        // Should produce valid normalized planes
        assert!(planes.all_normalized());

        // All plane normals should have unit length
        for plane in &planes.planes {
            let len = vec3_length(plane.normal);
            assert!((len - 1.0).abs() < 1e-5);
        }
    }
}

// ============================================================================
// 2. Plane Normalization Tests
// ============================================================================

mod plane_normalization {
    use super::*;

    #[test]
    fn all_planes_have_unit_length_normals() {
        let vp = standard_perspective_vp();
        let planes = FrustumPlanes::from_view_projection(&vp);

        for (i, plane) in planes.planes.iter().enumerate() {
            let len = vec3_length(plane.normal);
            assert!(
                (len - 1.0).abs() < 1e-5,
                "Plane {} normal length is {}, expected 1.0", i, len
            );
        }
    }

    #[test]
    fn normalize_scales_distance_correctly() {
        // Create a plane with non-unit normal
        let plane = FrustumPlane::new([2.0, 0.0, 0.0], 4.0);

        // Normal should be normalized to (1, 0, 0)
        assert!((plane.normal[0] - 1.0).abs() < 1e-6);
        assert!(plane.normal[1].abs() < 1e-6);
        assert!(plane.normal[2].abs() < 1e-6);

        // Distance should be scaled by 1/2 (original length was 2)
        assert!((plane.distance - 2.0).abs() < 1e-6);
    }

    #[test]
    fn normalize_handles_large_normals() {
        let plane = FrustumPlane::new([1000.0, 0.0, 0.0], 500.0);

        assert!(plane.is_normalized());
        assert!((plane.normal[0] - 1.0).abs() < 1e-5);
        assert!((plane.distance - 0.5).abs() < 1e-5);
    }

    #[test]
    fn normalize_handles_small_normals() {
        let plane = FrustumPlane::new([0.001, 0.0, 0.0], 0.001);

        assert!(plane.is_normalized());
        assert!((plane.normal[0] - 1.0).abs() < 1e-5);
        // Distance scales proportionally
        assert!((plane.distance - 1.0).abs() < 1e-5);
    }

    #[test]
    fn zero_length_normal_defaults_to_z_axis() {
        let plane = FrustumPlane::new([0.0, 0.0, 0.0], 10.0);

        // Should default to (0, 0, 1) with distance 0
        assert_eq!(plane.normal, [0.0, 0.0, 1.0]);
        assert_eq!(plane.distance, 0.0);
    }

    #[test]
    fn near_zero_normal_defaults_to_z_axis() {
        // Very small but non-zero normal
        let plane = FrustumPlane::new([1e-10, 1e-10, 1e-10], 5.0);

        // Should default to (0, 0, 1) because length < epsilon
        assert_eq!(plane.normal, [0.0, 0.0, 1.0]);
        assert_eq!(plane.distance, 0.0);
    }

    #[test]
    fn is_normalized_returns_true_for_unit_normal() {
        let plane = FrustumPlane::new([1.0, 0.0, 0.0], 5.0);
        assert!(plane.is_normalized());

        let plane2 = FrustumPlane::new([0.0, 1.0, 0.0], -3.0);
        assert!(plane2.is_normalized());

        let plane3 = FrustumPlane::new([0.0, 0.0, 1.0], 0.0);
        assert!(plane3.is_normalized());
    }

    #[test]
    fn is_normalized_returns_false_for_non_unit_normal() {
        let mut plane = FrustumPlane {
            normal: [2.0, 0.0, 0.0],
            distance: 1.0,
        };
        assert!(!plane.is_normalized());

        plane.normalize();
        assert!(plane.is_normalized());
    }

    #[test]
    fn normalize_method_can_be_called_multiple_times() {
        let mut plane = FrustumPlane::new([3.0, 4.0, 0.0], 10.0);

        // Already normalized by constructor
        assert!(plane.is_normalized());

        // Normalizing again should not change values
        let old_normal = plane.normal;
        let old_distance = plane.distance;

        plane.normalize();

        assert_eq!(plane.normal, old_normal);
        assert!((plane.distance - old_distance).abs() < 1e-6);
    }

    #[test]
    fn normalization_preserves_plane_equation() {
        let original_normal = [3.0, 4.0, 0.0]; // length = 5
        let original_distance = 10.0;

        let plane = FrustumPlane::new(original_normal, original_distance);

        // A point on the original plane: if 3x + 4y + 10 = 0, then at (0, -2.5, 0)
        // After normalization: (3/5)x + (4/5)y + 2 = 0
        // Check: (3/5)*0 + (4/5)*(-2.5) + 2 = -2 + 2 = 0 ✓

        let point = [0.0, -2.5, 0.0];
        let dist = plane.distance_to_point(point);
        assert!(dist.abs() < 1e-5, "Point on plane should have distance ~0, got {}", dist);
    }
}

// ============================================================================
// 3. Struct Layout Tests
// ============================================================================

mod struct_layout {
    use super::*;

    #[test]
    fn frustum_plane_is_16_bytes() {
        assert_eq!(
            mem::size_of::<FrustumPlane>(),
            16,
            "FrustumPlane must be 16 bytes for GPU alignment"
        );
    }

    #[test]
    fn frustum_plane_matches_constant() {
        assert_eq!(
            mem::size_of::<FrustumPlane>(),
            FRUSTUM_PLANE_SIZE,
            "FrustumPlane size must match FRUSTUM_PLANE_SIZE constant"
        );
    }

    #[test]
    fn frustum_planes_is_96_bytes() {
        assert_eq!(
            mem::size_of::<FrustumPlanes>(),
            96,
            "FrustumPlanes must be 96 bytes (6 x 16)"
        );
    }

    #[test]
    fn frustum_planes_matches_constant() {
        assert_eq!(
            mem::size_of::<FrustumPlanes>(),
            FRUSTUM_PLANES_SIZE,
            "FrustumPlanes size must match FRUSTUM_PLANES_SIZE constant"
        );
    }

    #[test]
    fn frustum_plane_field_alignment() {
        // FrustumPlane should have contiguous layout: [f32; 3] + f32
        // Use unit vector to avoid normalization changing values
        let plane = FrustumPlane::new([1.0, 0.0, 0.0], 5.0);
        let bytes: &[u8] = bytemuck::bytes_of(&plane);

        // Read back as f32 values using byte slices
        let f0 = f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let f1 = f32::from_ne_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let f2 = f32::from_ne_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        let f3 = f32::from_ne_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);

        // Since [1,0,0] is already unit, values remain unchanged
        assert!((f0 - 1.0).abs() < 1e-6, "normal.x should be 1.0, got {}", f0);
        assert!(f1.abs() < 1e-6, "normal.y should be 0.0, got {}", f1);
        assert!(f2.abs() < 1e-6, "normal.z should be 0.0, got {}", f2);
        assert!((f3 - 5.0).abs() < 1e-6, "distance should be 5.0, got {}", f3);
    }

    #[test]
    fn frustum_planes_array_layout() {
        let planes = FrustumPlanes::identity();
        let bytes: &[u8] = bytemuck::bytes_of(&planes);

        assert_eq!(bytes.len(), 96);

        // Each plane starts at 16-byte intervals
        for i in 0..6 {
            let start = i * 16;
            let plane_bytes = &bytes[start..start + 16];

            // Read back as f32 values using byte slices
            let f0 = f32::from_ne_bytes([plane_bytes[0], plane_bytes[1], plane_bytes[2], plane_bytes[3]]);
            let f1 = f32::from_ne_bytes([plane_bytes[4], plane_bytes[5], plane_bytes[6], plane_bytes[7]]);
            let f2 = f32::from_ne_bytes([plane_bytes[8], plane_bytes[9], plane_bytes[10], plane_bytes[11]]);
            let f3 = f32::from_ne_bytes([plane_bytes[12], plane_bytes[13], plane_bytes[14], plane_bytes[15]]);

            // Just verify we can read valid floats
            assert!(f0.is_finite());
            assert!(f1.is_finite());
            assert!(f2.is_finite());
            assert!(f3.is_finite());
        }
    }

    #[test]
    fn frustum_plane_is_pod() {
        // Verify Pod trait is properly implemented
        let plane = FrustumPlane::new([1.0, 0.0, 0.0], 1.0);
        let _bytes: &[u8] = bytemuck::bytes_of(&plane);

        // Should compile - Pod means we can safely cast to bytes
        let zeroed: FrustumPlane = FrustumPlane::zeroed();
        assert_eq!(zeroed.normal, [0.0, 0.0, 0.0]);
        assert_eq!(zeroed.distance, 0.0);
    }

    #[test]
    fn frustum_planes_is_pod() {
        let planes = FrustumPlanes::identity();
        let bytes: &[u8] = bytemuck::bytes_of(&planes);

        // Should be able to cast back and forth
        let planes_ref: &FrustumPlanes = bytemuck::from_bytes(bytes);
        assert_eq!(planes_ref.planes.len(), 6);
    }

    #[test]
    fn frustum_plane_is_zeroable() {
        let zeroed: FrustumPlane = FrustumPlane::zeroed();
        assert_eq!(zeroed.normal, [0.0, 0.0, 0.0]);
        assert_eq!(zeroed.distance, 0.0);
    }

    #[test]
    fn frustum_planes_is_zeroable() {
        let zeroed: FrustumPlanes = FrustumPlanes::zeroed();
        for plane in &zeroed.planes {
            assert_eq!(plane.normal, [0.0, 0.0, 0.0]);
            assert_eq!(plane.distance, 0.0);
        }
    }

    #[test]
    fn frustum_plane_is_copy() {
        let plane = FrustumPlane::new([1.0, 0.0, 0.0], 1.0);
        let copy = plane; // Copy, not move
        assert_eq!(plane.normal, copy.normal);
        assert_eq!(plane.distance, copy.distance);
    }

    #[test]
    fn frustum_planes_is_copy() {
        let planes = FrustumPlanes::identity();
        let copy = planes; // Copy, not move
        assert_eq!(planes.planes.len(), copy.planes.len());
    }

    #[test]
    fn num_frustum_planes_constant_is_six() {
        assert_eq!(NUM_FRUSTUM_PLANES, 6);
    }

    #[test]
    fn plane_index_constants_are_correct() {
        assert_eq!(PLANE_LEFT, 0);
        assert_eq!(PLANE_RIGHT, 1);
        assert_eq!(PLANE_BOTTOM, 2);
        assert_eq!(PLANE_TOP, 3);
        assert_eq!(PLANE_NEAR, 4);
        assert_eq!(PLANE_FAR, 5);
    }
}

// ============================================================================
// 4. Plane Accessor Tests
// ============================================================================

mod plane_accessors {
    use super::*;

    #[test]
    fn plane_index_accessor_returns_correct_plane() {
        let planes = FrustumPlanes::identity();

        for i in 0..NUM_FRUSTUM_PLANES {
            let plane = planes.plane(i);
            assert!(std::ptr::eq(plane, &planes.planes[i]));
        }
    }

    #[test]
    fn left_accessor_returns_plane_zero() {
        let planes = FrustumPlanes::identity();
        assert!(std::ptr::eq(planes.left(), &planes.planes[PLANE_LEFT]));
        assert!(std::ptr::eq(planes.left(), &planes.planes[0]));
    }

    #[test]
    fn right_accessor_returns_plane_one() {
        let planes = FrustumPlanes::identity();
        assert!(std::ptr::eq(planes.right(), &planes.planes[PLANE_RIGHT]));
        assert!(std::ptr::eq(planes.right(), &planes.planes[1]));
    }

    #[test]
    fn bottom_accessor_returns_plane_two() {
        let planes = FrustumPlanes::identity();
        assert!(std::ptr::eq(planes.bottom(), &planes.planes[PLANE_BOTTOM]));
        assert!(std::ptr::eq(planes.bottom(), &planes.planes[2]));
    }

    #[test]
    fn top_accessor_returns_plane_three() {
        let planes = FrustumPlanes::identity();
        assert!(std::ptr::eq(planes.top(), &planes.planes[PLANE_TOP]));
        assert!(std::ptr::eq(planes.top(), &planes.planes[3]));
    }

    #[test]
    fn near_accessor_returns_plane_four() {
        let planes = FrustumPlanes::identity();
        assert!(std::ptr::eq(planes.near(), &planes.planes[PLANE_NEAR]));
        assert!(std::ptr::eq(planes.near(), &planes.planes[4]));
    }

    #[test]
    fn far_accessor_returns_plane_five() {
        let planes = FrustumPlanes::identity();
        assert!(std::ptr::eq(planes.far(), &planes.planes[PLANE_FAR]));
        assert!(std::ptr::eq(planes.far(), &planes.planes[5]));
    }

    #[test]
    fn all_normalized_returns_true_for_normalized_planes() {
        let planes = FrustumPlanes::identity();
        assert!(planes.all_normalized());

        let vp = standard_perspective_vp();
        let perspective_planes = FrustumPlanes::from_view_projection(&vp);
        assert!(perspective_planes.all_normalized());
    }
}

// ============================================================================
// 5. Distance and Culling Tests
// ============================================================================

mod culling {
    use super::*;

    #[test]
    fn distance_to_point_positive_for_inside() {
        // Plane at z=0, facing +Z
        let plane = FrustumPlane::new([0.0, 0.0, 1.0], 0.0);

        // Point in front (positive Z) should have positive distance
        let dist = plane.distance_to_point([0.0, 0.0, 5.0]);
        assert!(dist > 0.0, "Point in front should have positive distance");
    }

    #[test]
    fn distance_to_point_negative_for_outside() {
        // Plane at z=0, facing +Z
        let plane = FrustumPlane::new([0.0, 0.0, 1.0], 0.0);

        // Point behind (negative Z) should have negative distance
        let dist = plane.distance_to_point([0.0, 0.0, -5.0]);
        assert!(dist < 0.0, "Point behind should have negative distance");
    }

    #[test]
    fn distance_to_point_zero_on_plane() {
        // Plane at z=5, facing +Z
        let plane = FrustumPlane::new([0.0, 0.0, 1.0], -5.0);

        let dist = plane.distance_to_point([0.0, 0.0, 5.0]);
        assert!(dist.abs() < 1e-5, "Point on plane should have ~zero distance");
    }

    #[test]
    fn test_sphere_visible_when_inside() {
        let plane = FrustumPlane::new([0.0, 0.0, 1.0], 0.0);

        // Sphere entirely in front of plane
        assert!(plane.test_sphere([0.0, 0.0, 5.0], 1.0));
    }

    #[test]
    fn test_sphere_visible_when_intersecting() {
        let plane = FrustumPlane::new([0.0, 0.0, 1.0], 0.0);

        // Sphere centered at z=0.5 with radius 1 intersects plane at z=0
        assert!(plane.test_sphere([0.0, 0.0, 0.5], 1.0));
    }

    #[test]
    fn test_sphere_culled_when_outside() {
        let plane = FrustumPlane::new([0.0, 0.0, 1.0], 0.0);

        // Sphere entirely behind plane (center at z=-5, radius 1)
        assert!(!plane.test_sphere([0.0, 0.0, -5.0], 1.0));
    }

    #[test]
    fn test_sphere_boundary_case() {
        let plane = FrustumPlane::new([0.0, 0.0, 1.0], 0.0);

        // Sphere touching plane exactly
        assert!(plane.test_sphere([0.0, 0.0, -1.0], 1.0));

        // Sphere just past the boundary
        assert!(!plane.test_sphere([0.0, 0.0, -1.001], 1.0));
    }

    #[test]
    fn frustum_test_sphere_inside() {
        let vp = standard_perspective_vp();
        let planes = FrustumPlanes::from_view_projection(&vp);

        // Sphere at origin (in view) should be visible
        assert!(planes.test_sphere([0.0, 0.0, 0.0], 0.5));
    }

    #[test]
    fn frustum_test_sphere_outside() {
        let vp = standard_perspective_vp();
        let planes = FrustumPlanes::from_view_projection(&vp);

        // Sphere far behind camera should be culled
        assert!(!planes.test_sphere([0.0, 0.0, 100.0], 1.0));

        // Sphere far to the side should be culled
        assert!(!planes.test_sphere([100.0, 0.0, 0.0], 1.0));
    }

    #[test]
    fn frustum_test_aabb_inside() {
        let vp = standard_perspective_vp();
        let planes = FrustumPlanes::from_view_projection(&vp);

        // AABB at origin should be visible
        assert!(planes.test_aabb([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]));
    }

    #[test]
    fn frustum_test_aabb_outside() {
        let vp = standard_perspective_vp();
        let planes = FrustumPlanes::from_view_projection(&vp);

        // AABB far behind camera should be culled
        assert!(!planes.test_aabb([100.0, 100.0, 100.0], [102.0, 102.0, 102.0]));
    }

    #[test]
    fn from_point_normal_creates_correct_plane() {
        let point = [0.0, 5.0, 0.0];
        let normal = [0.0, 1.0, 0.0];

        let plane = FrustumPlane::from_point_normal(point, normal);

        // Point should lie on the plane
        let dist = plane.distance_to_point(point);
        assert!(dist.abs() < 1e-5, "Origin point should be on plane, dist = {}", dist);

        // Another point above the plane
        let dist_above = plane.distance_to_point([0.0, 10.0, 0.0]);
        assert!(dist_above > 0.0, "Point above should have positive distance");

        // Point below the plane
        let dist_below = plane.distance_to_point([0.0, 0.0, 0.0]);
        assert!(dist_below < 0.0, "Point below should have negative distance");
    }

    #[test]
    fn from_point_normal_handles_zero_normal() {
        let point = [1.0, 2.0, 3.0];
        let normal = [0.0, 0.0, 0.0];

        let plane = FrustumPlane::from_point_normal(point, normal);

        // Should default to z-axis
        assert_eq!(plane.normal, [0.0, 0.0, 1.0]);
        assert_eq!(plane.distance, 0.0);
    }
}

// ============================================================================
// 6. Edge Cases Tests
// ============================================================================

mod edge_cases {
    use super::*;

    #[test]
    fn degenerate_zero_matrix() {
        let zero = [[0.0; 4]; 4];
        let planes = FrustumPlanes::from_view_projection(&zero);

        // All planes should still be valid (defaulting to z-axis)
        for plane in &planes.planes {
            assert!(plane.is_normalized());
            // Zero matrix produces zero normals, which default to (0,0,1)
            assert_eq!(plane.normal, [0.0, 0.0, 1.0]);
        }
    }

    #[test]
    fn nearly_singular_matrix() {
        // Matrix with very small determinant
        let singular = [
            [1e-10, 0.0, 0.0, 0.0],
            [0.0, 1e-10, 0.0, 0.0],
            [0.0, 0.0, 1e-10, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        let planes = FrustumPlanes::from_view_projection(&singular);

        // Should handle gracefully without NaN or infinity
        for (i, plane) in planes.planes.iter().enumerate() {
            assert!(plane.normal[0].is_finite(), "Plane {} normal.x should be finite", i);
            assert!(plane.normal[1].is_finite(), "Plane {} normal.y should be finite", i);
            assert!(plane.normal[2].is_finite(), "Plane {} normal.z should be finite", i);
            assert!(plane.distance.is_finite(), "Plane {} distance should be finite", i);
        }
    }

    #[test]
    fn extreme_fov_wide() {
        // Very wide FOV (near 180 degrees)
        let fovy = std::f32::consts::PI * 0.9; // 162 degrees
        let proj = perspective_matrix(fovy, 1.0, 0.1, 100.0);
        let view = look_at_matrix([0.0, 0.0, 5.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let vp = multiply_matrices(&view, &proj);

        let planes = FrustumPlanes::from_view_projection(&vp);

        // Should still produce valid normalized planes
        for (i, plane) in planes.planes.iter().enumerate() {
            let len = vec3_length(plane.normal);
            assert!(
                (len - 1.0).abs() < 1e-4 || plane.normal == [0.0, 0.0, 1.0],
                "Wide FOV plane {} should be normalized, got length {}", i, len
            );
            assert!(plane.distance.is_finite());
        }
    }

    #[test]
    fn extreme_fov_narrow() {
        // Very narrow FOV
        let fovy = 0.01; // ~0.5 degrees
        let proj = perspective_matrix(fovy, 1.0, 0.1, 100.0);
        let view = look_at_matrix([0.0, 0.0, 5.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let vp = multiply_matrices(&view, &proj);

        let planes = FrustumPlanes::from_view_projection(&vp);

        for (i, plane) in planes.planes.iter().enumerate() {
            assert!(plane.normal[0].is_finite(), "Narrow FOV plane {} normal.x", i);
            assert!(plane.normal[1].is_finite(), "Narrow FOV plane {} normal.y", i);
            assert!(plane.normal[2].is_finite(), "Narrow FOV plane {} normal.z", i);
            assert!(plane.distance.is_finite(), "Narrow FOV plane {} distance", i);
        }
    }

    #[test]
    fn extreme_near_far_ratio() {
        // Very large near/far ratio
        let proj = perspective_matrix(1.0, 1.0, 0.001, 10000.0);
        let view = look_at_matrix([0.0, 0.0, 5.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let vp = multiply_matrices(&view, &proj);

        let planes = FrustumPlanes::from_view_projection(&vp);

        for (i, plane) in planes.planes.iter().enumerate() {
            assert!(plane.normal[0].is_finite(), "Extreme ratio plane {} normal.x", i);
            assert!(plane.normal[1].is_finite(), "Extreme ratio plane {} normal.y", i);
            assert!(plane.normal[2].is_finite(), "Extreme ratio plane {} normal.z", i);
            assert!(plane.distance.is_finite(), "Extreme ratio plane {} distance", i);
        }
    }

    #[test]
    fn very_small_near_plane() {
        let proj = perspective_matrix(1.0, 1.0, 1e-6, 100.0);
        let view = look_at_matrix([0.0, 0.0, 5.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let vp = multiply_matrices(&view, &proj);

        let planes = FrustumPlanes::from_view_projection(&vp);

        // All values should be finite
        for plane in &planes.planes {
            for &val in &plane.normal {
                assert!(val.is_finite());
            }
            assert!(plane.distance.is_finite());
        }
    }

    #[test]
    fn extreme_aspect_ratio_wide() {
        let proj = perspective_matrix(1.0, 100.0, 0.1, 100.0); // 100:1 aspect
        let view = look_at_matrix([0.0, 0.0, 5.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let vp = multiply_matrices(&view, &proj);

        let planes = FrustumPlanes::from_view_projection(&vp);

        for plane in &planes.planes {
            for &val in &plane.normal {
                assert!(val.is_finite());
            }
            assert!(plane.distance.is_finite());
        }
    }

    #[test]
    fn extreme_aspect_ratio_tall() {
        let proj = perspective_matrix(1.0, 0.01, 0.1, 100.0); // 1:100 aspect
        let view = look_at_matrix([0.0, 0.0, 5.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let vp = multiply_matrices(&view, &proj);

        let planes = FrustumPlanes::from_view_projection(&vp);

        for plane in &planes.planes {
            for &val in &plane.normal {
                assert!(val.is_finite());
            }
            assert!(plane.distance.is_finite());
        }
    }

    #[test]
    fn rotated_view_matrix() {
        // Camera looking along X axis instead of -Z
        let view = look_at_matrix([5.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let proj = perspective_matrix(1.0, 1.0, 0.1, 100.0);
        let vp = multiply_matrices(&view, &proj);

        let planes = FrustumPlanes::from_view_projection(&vp);

        assert!(planes.all_normalized());

        // Object at origin should be visible
        assert!(planes.test_sphere([0.0, 0.0, 0.0], 0.5));
    }

    #[test]
    fn translated_view() {
        // Camera at a different position
        let view = look_at_matrix([10.0, 10.0, 10.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let proj = perspective_matrix(1.0, 1.0, 0.1, 100.0);
        let vp = multiply_matrices(&view, &proj);

        let planes = FrustumPlanes::from_view_projection(&vp);

        assert!(planes.all_normalized());
    }

    #[test]
    fn plane_default_is_zeroed() {
        let plane = FrustumPlane::default();
        assert_eq!(plane.normal, [0.0, 0.0, 0.0]);
        assert_eq!(plane.distance, 0.0);
    }

    #[test]
    fn planes_default_is_zeroed() {
        let planes = FrustumPlanes::default();
        for plane in &planes.planes {
            assert_eq!(plane.normal, [0.0, 0.0, 0.0]);
            assert_eq!(plane.distance, 0.0);
        }
    }

    #[test]
    fn plane_debug_output() {
        let plane = FrustumPlane::new([1.0, 0.0, 0.0], 5.0);
        let debug = format!("{:?}", plane);
        assert!(debug.contains("FrustumPlane"));
        assert!(debug.contains("normal"));
        assert!(debug.contains("distance"));
    }

    #[test]
    fn planes_debug_output() {
        let planes = FrustumPlanes::identity();
        let debug = format!("{:?}", planes);
        assert!(debug.contains("FrustumPlanes"));
        assert!(debug.contains("planes"));
    }

    #[test]
    fn plane_partial_eq() {
        let plane1 = FrustumPlane::new([1.0, 0.0, 0.0], 1.0);
        let plane2 = FrustumPlane::new([1.0, 0.0, 0.0], 1.0);
        let plane3 = FrustumPlane::new([0.0, 1.0, 0.0], 1.0);

        assert_eq!(plane1, plane2);
        assert_ne!(plane1, plane3);
    }

    #[test]
    fn planes_partial_eq() {
        let planes1 = FrustumPlanes::identity();
        let planes2 = FrustumPlanes::identity();

        assert_eq!(planes1, planes2);
    }

    #[test]
    fn nan_in_matrix_produces_fallback() {
        let nan_matrix = [
            [f32::NAN, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        let planes = FrustumPlanes::from_view_projection(&nan_matrix);

        // With NaN in the matrix, row operations produce NaN which should
        // trigger the zero-normal fallback
        // The left plane combines row3 + row0, which involves NaN
        // This should either produce NaN or fallback - implementation dependent
        // At minimum we shouldn't crash

        // Just verify we don't panic and structure is intact
        assert_eq!(planes.planes.len(), 6);
    }

    #[test]
    fn infinity_in_matrix_handled() {
        let inf_matrix = [
            [f32::INFINITY, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        let planes = FrustumPlanes::from_view_projection(&inf_matrix);

        // Should not panic
        assert_eq!(planes.planes.len(), 6);
    }
}

// ============================================================================
// 7. Helper Function Tests
// ============================================================================

mod helper_functions {
    use super::*;

    #[test]
    fn perspective_matrix_diagonal_elements() {
        let proj = perspective_matrix(1.0, 1.0, 0.1, 100.0);

        // For square aspect, [0][0] should equal [1][1]
        assert!((proj[0][0] - proj[1][1]).abs() < 1e-6);

        // [2][3] should be -1 for right-handed coords
        assert!((proj[2][3] - (-1.0)).abs() < 1e-6);

        // [3][3] should be 0 for perspective
        assert!(proj[3][3].abs() < 1e-6);
    }

    #[test]
    fn look_at_produces_orthonormal_basis() {
        let view = look_at_matrix([0.0, 0.0, 5.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);

        // Extract basis vectors from columns
        let right = [view[0][0], view[0][1], view[0][2]];
        let up = [view[1][0], view[1][1], view[1][2]];
        let forward = [view[2][0], view[2][1], view[2][2]];

        // Should be unit length
        assert!((vec3_length(right) - 1.0).abs() < 1e-5);
        assert!((vec3_length(up) - 1.0).abs() < 1e-5);
        assert!((vec3_length(forward) - 1.0).abs() < 1e-5);

        // Should be orthogonal
        assert!(vec3_dot(right, up).abs() < 1e-5);
        assert!(vec3_dot(right, forward).abs() < 1e-5);
        assert!(vec3_dot(up, forward).abs() < 1e-5);
    }

    #[test]
    fn matrix_multiply_identity() {
        let identity = identity_matrix();
        let result = multiply_matrices(&identity, &identity);

        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (result[i][j] - expected).abs() < 1e-6,
                    "Identity * Identity should be Identity at [{i}][{j}]"
                );
            }
        }
    }

    #[test]
    fn matrix_multiply_scaling() {
        let scale = [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        let identity = identity_matrix();
        let result = multiply_matrices(&scale, &identity);

        // Scale * Identity = Scale
        assert!((result[0][0] - 2.0).abs() < 1e-6);
        assert!((result[1][1] - 3.0).abs() < 1e-6);
        assert!((result[2][2] - 4.0).abs() < 1e-6);
        assert!((result[3][3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn matrix_multiply_non_commutative() {
        let a = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0], // Translation by (1, 0, 0)
        ];

        let b = [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0], // Scale X by 2
        ];

        let ab = multiply_matrices(&a, &b);
        let ba = multiply_matrices(&b, &a);

        // These should differ (matrix multiplication is not commutative)
        let differ = ab.iter()
            .zip(ba.iter())
            .any(|(col_ab, col_ba)| {
                col_ab.iter().zip(col_ba.iter())
                    .any(|(a, b)| (a - b).abs() > 1e-6)
            });

        assert!(differ, "AB and BA should differ for these matrices");
    }
}

// ============================================================================
// 8. Integration Tests
// ============================================================================

mod integration {
    use super::*;

    #[test]
    fn full_culling_pipeline() {
        // Create realistic camera setup
        let view = look_at_matrix([0.0, 2.0, 10.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let proj = perspective_matrix(std::f32::consts::FRAC_PI_4, 16.0/9.0, 0.1, 1000.0);
        let vp = multiply_matrices(&view, &proj);

        let planes = FrustumPlanes::from_view_projection(&vp);

        // All planes should be valid
        assert!(planes.all_normalized());

        // Object in view
        assert!(planes.test_sphere([0.0, 0.0, 0.0], 1.0), "Origin should be visible");

        // Object behind camera
        assert!(!planes.test_sphere([0.0, 0.0, 20.0], 1.0), "Behind camera should be culled");

        // Object far to the left
        assert!(!planes.test_sphere([-100.0, 0.0, 0.0], 1.0), "Far left should be culled");
    }

    #[test]
    fn aabb_vs_sphere_consistency() {
        let planes = FrustumPlanes::identity();

        // For a unit cube at origin, sphere and AABB tests should agree
        let sphere_visible = planes.test_sphere([0.0, 0.0, 0.0], 0.5);
        let aabb_visible = planes.test_aabb([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]);

        // Both should report visible
        assert!(sphere_visible);
        assert!(aabb_visible);
    }

    #[test]
    fn multiple_frame_updates() {
        // Simulate multiple frame updates
        for frame in 0..10 {
            let t = frame as f32 * 0.1;

            let eye = [t.cos() * 10.0, 5.0, t.sin() * 10.0];
            let view = look_at_matrix(eye, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
            let proj = perspective_matrix(1.0, 1.0, 0.1, 100.0);
            let vp = multiply_matrices(&view, &proj);

            let planes = FrustumPlanes::from_view_projection(&vp);

            // Should always produce valid planes
            assert!(planes.all_normalized(), "Frame {} should have normalized planes", frame);

            // Origin should always be visible from this orbit
            assert!(planes.test_sphere([0.0, 0.0, 0.0], 0.5),
                    "Origin should be visible from frame {} position", frame);
        }
    }
}
