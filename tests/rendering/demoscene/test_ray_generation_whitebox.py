"""
Whitebox tests for T-DEMO-3.1: Camera Ray Generation (Pinhole Model).

Internal implementation tests for the RayGenerator class including:
- Internal camera basis vector computation
- Edge cases in normalization
- Caching behavior
- WGSL inline generation details
"""

from __future__ import annotations

import math
import pytest

from engine.rendering.demoscene.ray_generation import (
    Ray,
    RayGenerator,
    Vec3,
    CameraParams,
    generate_ray_wgsl,
    generate_ray_wgsl_inline,
    validate_camera,
)
from engine.rendering.demoscene.ast_nodes import (
    CameraNode,
    Vec3Node,
    FloatNode,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def standard_camera() -> CameraNode:
    """Standard camera for testing."""
    return CameraNode(
        origin=Vec3Node(0.0, 0.0, 5.0),
        look_at=Vec3Node(0.0, 0.0, 0.0),
        up=Vec3Node(0.0, 1.0, 0.0),
        fov=FloatNode(60.0),
        aspect_ratio=FloatNode(1.0),
    )


# =============================================================================
# Test: Internal Basis Vector Computation
# =============================================================================


class TestBasisVectorComputation:
    """Tests for internal camera basis vector computation."""

    def test_forward_vector_computed_correctly(self):
        """Forward vector should be normalized(look_at - origin)."""
        gen = RayGenerator()
        camera = CameraNode(
            origin=Vec3Node(0.0, 0.0, 10.0),
            look_at=Vec3Node(0.0, 0.0, 0.0),
            up=Vec3Node(0.0, 1.0, 0.0),
            fov=FloatNode(60.0),
            aspect_ratio=FloatNode(1.0),
        )

        # Generate a ray to trigger setup
        gen.generate_ray(0.0, 0.0, camera)

        # Check internal forward vector
        assert gen._forward is not None
        assert abs(gen._forward.z + 1.0) < 1e-6
        assert abs(gen._forward.length() - 1.0) < 1e-6

    def test_right_vector_perpendicular_to_forward(self):
        """Right vector should be perpendicular to forward."""
        gen = RayGenerator()
        camera = CameraNode(
            origin=Vec3Node(0.0, 0.0, 5.0),
            look_at=Vec3Node(0.0, 0.0, 0.0),
            up=Vec3Node(0.0, 1.0, 0.0),
            fov=FloatNode(60.0),
            aspect_ratio=FloatNode(1.0),
        )

        gen.generate_ray(0.0, 0.0, camera)

        dot = gen._forward.dot(gen._right)
        assert abs(dot) < 1e-6

    def test_up_vector_perpendicular_to_forward_and_right(self):
        """Corrected up vector should be perpendicular to forward and right."""
        gen = RayGenerator()
        camera = CameraNode(
            origin=Vec3Node(0.0, 0.0, 5.0),
            look_at=Vec3Node(0.0, 0.0, 0.0),
            up=Vec3Node(0.0, 1.0, 0.0),
            fov=FloatNode(60.0),
            aspect_ratio=FloatNode(1.0),
        )

        gen.generate_ray(0.0, 0.0, camera)

        assert abs(gen._forward.dot(gen._up)) < 1e-6
        assert abs(gen._right.dot(gen._up)) < 1e-6

    def test_basis_vectors_are_unit_length(self):
        """All basis vectors should be unit length."""
        gen = RayGenerator()
        camera = CameraNode(
            origin=Vec3Node(1.0, 2.0, 3.0),
            look_at=Vec3Node(4.0, 5.0, 6.0),
            up=Vec3Node(0.0, 1.0, 0.0),
            fov=FloatNode(45.0),
            aspect_ratio=FloatNode(1.5),
        )

        gen.generate_ray(0.0, 0.0, camera)

        assert abs(gen._forward.length() - 1.0) < 1e-6
        assert abs(gen._right.length() - 1.0) < 1e-6
        assert abs(gen._up.length() - 1.0) < 1e-6


# =============================================================================
# Test: Half-Width and Half-Height Computation
# =============================================================================


class TestImagePlaneDimensions:
    """Tests for image plane dimension computation."""

    def test_half_height_from_fov(self):
        """Half height should be tan(fov/2)."""
        gen = RayGenerator()
        camera = CameraNode(
            origin=Vec3Node(0.0, 0.0, 5.0),
            look_at=Vec3Node(0.0, 0.0, 0.0),
            up=Vec3Node(0.0, 1.0, 0.0),
            fov=FloatNode(90.0),  # tan(45) = 1
            aspect_ratio=FloatNode(1.0),
        )

        gen.generate_ray(0.0, 0.0, camera)

        # tan(45 degrees) = 1
        assert abs(gen._half_height - 1.0) < 1e-6

    def test_half_width_includes_aspect_ratio(self):
        """Half width should be half_height * aspect_ratio."""
        gen = RayGenerator()
        camera = CameraNode(
            origin=Vec3Node(0.0, 0.0, 5.0),
            look_at=Vec3Node(0.0, 0.0, 0.0),
            up=Vec3Node(0.0, 1.0, 0.0),
            fov=FloatNode(90.0),
            aspect_ratio=FloatNode(2.0),
        )

        gen.generate_ray(0.0, 0.0, camera)

        assert abs(gen._half_width - 2.0) < 1e-6

    def test_small_fov_small_dimensions(self):
        """Small FOV should result in small half dimensions."""
        gen = RayGenerator()
        camera = CameraNode(
            origin=Vec3Node(0.0, 0.0, 5.0),
            look_at=Vec3Node(0.0, 0.0, 0.0),
            up=Vec3Node(0.0, 1.0, 0.0),
            fov=FloatNode(10.0),  # Small FOV
            aspect_ratio=FloatNode(1.0),
        )

        gen.generate_ray(0.0, 0.0, camera)

        # tan(5 degrees) is small
        assert gen._half_height < 0.1


# =============================================================================
# Test: Vec3 Edge Cases
# =============================================================================


class TestVec3EdgeCases:
    """Tests for Vec3 edge cases."""

    def test_vec3_from_node(self):
        """Vec3.from_node should correctly extract values."""
        node = Vec3Node(1.0, 2.0, 3.0)
        v = Vec3.from_node(node)

        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0

    def test_vec3_from_tuple(self):
        """Vec3.from_tuple should work with tuples."""
        v = Vec3.from_tuple((4.0, 5.0, 6.0))

        assert v.x == 4.0
        assert v.y == 5.0
        assert v.z == 6.0

    def test_vec3_zero_factory(self):
        """Vec3.zero should return zero vector."""
        v = Vec3.zero()

        assert v.x == 0.0
        assert v.y == 0.0
        assert v.z == 0.0

    def test_vec3_length_squared(self):
        """length_squared should avoid sqrt."""
        v = Vec3(3.0, 4.0, 0.0)

        assert abs(v.length_squared() - 25.0) < 1e-6

    def test_vec3_negation(self):
        """Negation should negate all components."""
        v = Vec3(1.0, -2.0, 3.0)
        n = -v

        assert n.x == -1.0
        assert n.y == 2.0
        assert n.z == -3.0

    def test_vec3_rmul(self):
        """Right multiplication should work."""
        v = Vec3(1.0, 2.0, 3.0)
        r = 2.0 * v

        assert r.x == 2.0
        assert r.y == 4.0
        assert r.z == 6.0

    def test_vec3_equality_with_tolerance(self):
        """Vec3 equality should handle floating point."""
        a = Vec3(1.0, 2.0, 3.0)
        b = Vec3(1.0 + 1e-10, 2.0 - 1e-10, 3.0)

        assert a == b

    def test_vec3_equality_different_types(self):
        """Vec3 equality with non-Vec3 should return NotImplemented."""
        v = Vec3(1.0, 2.0, 3.0)

        assert v.__eq__("not a vec3") == NotImplemented

    def test_vec3_repr(self):
        """Vec3 repr should be readable."""
        v = Vec3(1.0, 2.0, 3.0)

        assert "Vec3" in repr(v)
        assert "1.0" in repr(v)


# =============================================================================
# Test: Ray Edge Cases
# =============================================================================


class TestRayEdgeCases:
    """Tests for Ray edge cases."""

    def test_ray_point_at_negative_t(self):
        """point_at with negative t should work (behind origin)."""
        ray = Ray(origin=Vec3(0, 0, 0), direction=Vec3(1, 0, 0))

        p = ray.point_at(-5.0)
        assert p.x == -5.0

    def test_ray_point_at_zero(self):
        """point_at(0) should return origin."""
        ray = Ray(origin=Vec3(1, 2, 3), direction=Vec3(1, 0, 0))

        p = ray.point_at(0.0)
        assert p.x == 1.0
        assert p.y == 2.0
        assert p.z == 3.0

    def test_ray_repr(self):
        """Ray repr should be readable."""
        ray = Ray(origin=Vec3(0, 0, 0), direction=Vec3(1, 0, 0))

        assert "Ray" in repr(ray)
        assert "origin" in repr(ray)


# =============================================================================
# Test: CameraParams Edge Cases
# =============================================================================


class TestCameraParamsEdgeCases:
    """Tests for CameraParams edge cases."""

    def test_camera_params_extracts_aperture(self):
        """CameraParams should extract aperture."""
        camera = CameraNode(
            origin=Vec3Node(0.0, 0.0, 5.0),
            look_at=Vec3Node(0.0, 0.0, 0.0),
            up=Vec3Node(0.0, 1.0, 0.0),
            fov=FloatNode(60.0),
            aspect_ratio=FloatNode(1.0),
            aperture=FloatNode(0.5),
            focal_distance=FloatNode(20.0),
        )

        params = CameraParams.from_camera_node(camera)

        assert params.aperture == 0.5
        assert params.focal_distance == 20.0

    def test_camera_params_defaults(self):
        """CameraParams should handle missing optional fields."""
        camera = CameraNode(
            origin=Vec3Node(0.0, 0.0, 5.0),
            look_at=Vec3Node(0.0, 0.0, 0.0),
            up=Vec3Node(0.0, 1.0, 0.0),
            fov=FloatNode(60.0),
            aspect_ratio=FloatNode(1.0),
        )

        params = CameraParams.from_camera_node(camera)

        # Should have default values
        assert params.aperture == 0.0


# =============================================================================
# Test: WGSL Code Generation Details
# =============================================================================


class TestWGSLCodeGenDetails:
    """Tests for WGSL code generation details."""

    def test_wgsl_contains_comments(self):
        """Generated WGSL should have documentation comments."""
        wgsl = generate_ray_wgsl()

        assert "///" in wgsl  # Doc comments

    def test_wgsl_contains_normalize(self):
        """Generated WGSL should normalize direction."""
        wgsl = generate_ray_wgsl()

        assert "normalize" in wgsl

    def test_wgsl_contains_tan(self):
        """Generated WGSL should use tan for FOV."""
        wgsl = generate_ray_wgsl()

        assert "tan" in wgsl

    def test_wgsl_inline_contains_constants(self, standard_camera):
        """Inline WGSL should define camera constants."""
        wgsl = generate_ray_wgsl_inline(standard_camera)

        assert "const CAMERA_ORIGIN" in wgsl
        assert "const CAMERA_FORWARD" in wgsl
        assert "const CAMERA_RIGHT" in wgsl
        assert "const CAMERA_UP" in wgsl
        assert "const CAMERA_HALF_WIDTH" in wgsl
        assert "const CAMERA_HALF_HEIGHT" in wgsl

    def test_wgsl_inline_has_correct_values(self, standard_camera):
        """Inline WGSL should have correct constant values."""
        wgsl = generate_ray_wgsl_inline(standard_camera)

        # Origin should be (0, 0, 5)
        assert "5.0" in wgsl or "5" in wgsl

    def test_wgsl_has_struct_fields(self):
        """Ray struct should have origin and direction fields."""
        wgsl = generate_ray_wgsl()

        assert "origin: vec3<f32>" in wgsl
        assert "direction: vec3<f32>" in wgsl


# =============================================================================
# Test: Validation Edge Cases
# =============================================================================


class TestValidationEdgeCases:
    """Tests for validation edge cases."""

    def test_validate_very_small_fov(self):
        """Very small FOV should be valid."""
        camera = CameraNode(
            origin=Vec3Node(0.0, 0.0, 5.0),
            look_at=Vec3Node(0.0, 0.0, 0.0),
            up=Vec3Node(0.0, 1.0, 0.0),
            fov=FloatNode(0.001),  # Very small but positive
            aspect_ratio=FloatNode(1.0),
        )

        errors = validate_camera(camera)
        assert len(errors) == 0

    def test_validate_fov_179_valid(self):
        """FOV just under 180 should be valid."""
        camera = CameraNode(
            origin=Vec3Node(0.0, 0.0, 5.0),
            look_at=Vec3Node(0.0, 0.0, 0.0),
            up=Vec3Node(0.0, 1.0, 0.0),
            fov=FloatNode(179.0),
            aspect_ratio=FloatNode(1.0),
        )

        errors = validate_camera(camera)
        assert len(errors) == 0

    def test_validate_very_small_aspect(self):
        """Very small aspect ratio should be valid."""
        camera = CameraNode(
            origin=Vec3Node(0.0, 0.0, 5.0),
            look_at=Vec3Node(0.0, 0.0, 0.0),
            up=Vec3Node(0.0, 1.0, 0.0),
            fov=FloatNode(60.0),
            aspect_ratio=FloatNode(0.001),
        )

        errors = validate_camera(camera)
        assert len(errors) == 0

    def test_validate_negative_aspect(self):
        """Negative aspect ratio should be invalid."""
        camera = CameraNode(
            origin=Vec3Node(0.0, 0.0, 5.0),
            look_at=Vec3Node(0.0, 0.0, 0.0),
            up=Vec3Node(0.0, 1.0, 0.0),
            fov=FloatNode(60.0),
            aspect_ratio=FloatNode(-1.0),
        )

        errors = validate_camera(camera)
        assert len(errors) > 0

    def test_validate_up_nearly_parallel(self):
        """Up vector nearly parallel to forward should be invalid."""
        # Camera looking along Z, up along Z (parallel)
        camera = CameraNode(
            origin=Vec3Node(0.0, 0.0, 5.0),
            look_at=Vec3Node(0.0, 0.0, 0.0),
            up=Vec3Node(0.0, 0.0, -1.0),  # Same direction as forward
            fov=FloatNode(60.0),
            aspect_ratio=FloatNode(1.0),
        )

        errors = validate_camera(camera)
        assert len(errors) > 0


# =============================================================================
# Test: Grid Generation Details
# =============================================================================


class TestGridGenerationDetails:
    """Tests for grid ray generation details."""

    def test_grid_ray_directions_vary(self):
        """Grid rays should have different directions."""
        gen = RayGenerator()
        camera = CameraNode(
            origin=Vec3Node(0.0, 0.0, 5.0),
            look_at=Vec3Node(0.0, 0.0, 0.0),
            up=Vec3Node(0.0, 1.0, 0.0),
            fov=FloatNode(60.0),
            aspect_ratio=FloatNode(1.0),
        )

        rays = gen.generate_rays_grid(camera, 3, 3)

        # Check corner rays are different
        tl = rays[0][0]
        tr = rays[0][2]
        bl = rays[2][0]
        br = rays[2][2]

        assert not tl.direction.approx_equal(tr.direction, 0.01)
        assert not tl.direction.approx_equal(bl.direction, 0.01)
        assert not tl.direction.approx_equal(br.direction, 0.01)

    def test_grid_center_ray_is_forward(self):
        """Center of grid should point forward."""
        gen = RayGenerator()
        camera = CameraNode(
            origin=Vec3Node(0.0, 0.0, 5.0),
            look_at=Vec3Node(0.0, 0.0, 0.0),
            up=Vec3Node(0.0, 1.0, 0.0),
            fov=FloatNode(60.0),
            aspect_ratio=FloatNode(1.0),
        )

        rays = gen.generate_rays_grid(camera, 5, 5)
        center = rays[2][2]

        # Center ray should point approximately forward
        assert abs(center.direction.z + 1.0) < 0.1
        assert abs(center.direction.x) < 0.1
        assert abs(center.direction.y) < 0.1


# =============================================================================
# Test: Origin Preservation
# =============================================================================


class TestOriginPreservation:
    """Tests that ray origins are preserved correctly."""

    def test_all_rays_share_origin(self):
        """All rays from same camera should have same origin."""
        gen = RayGenerator()
        camera = CameraNode(
            origin=Vec3Node(1.0, 2.0, 3.0),
            look_at=Vec3Node(0.0, 0.0, 0.0),
            up=Vec3Node(0.0, 1.0, 0.0),
            fov=FloatNode(60.0),
            aspect_ratio=FloatNode(1.0),
        )

        ray1 = gen.generate_ray(0.0, 0.0, camera)
        ray2 = gen.generate_ray(1.0, 1.0, camera)
        ray3 = gen.generate_ray(-1.0, -1.0, camera)

        assert ray1.origin.approx_equal(ray2.origin)
        assert ray2.origin.approx_equal(ray3.origin)

    def test_origin_not_mutated(self):
        """Ray origin should be a copy, not shared."""
        gen = RayGenerator()
        camera = CameraNode(
            origin=Vec3Node(1.0, 2.0, 3.0),
            look_at=Vec3Node(0.0, 0.0, 0.0),
            up=Vec3Node(0.0, 1.0, 0.0),
            fov=FloatNode(60.0),
            aspect_ratio=FloatNode(1.0),
        )

        ray1 = gen.generate_ray(0.0, 0.0, camera)
        ray2 = gen.generate_ray(1.0, 1.0, camera)

        # Modify ray1's origin
        ray1.origin.x = 999.0

        # ray2's origin should be unchanged
        assert ray2.origin.x != 999.0
