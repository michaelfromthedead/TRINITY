"""
Blackbox tests for T-DEMO-3.1: Camera Ray Generation (Pinhole Model).

Tests the RayGenerator class and related functions for correct ray
generation from camera parameters and UV coordinates.

Coverage targets:
- RayGenerator.generate_ray() for various UV coordinates
- Pinhole camera model with FOV and aspect ratio
- Corner and edge cases
- WGSL code generation
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
def default_camera() -> CameraNode:
    """Default camera looking down -Z axis."""
    return CameraNode(
        origin=Vec3Node(0.0, 0.0, 5.0),
        look_at=Vec3Node(0.0, 0.0, 0.0),
        up=Vec3Node(0.0, 1.0, 0.0),
        fov=FloatNode(60.0),
        aspect_ratio=FloatNode(16.0 / 9.0),
    )


@pytest.fixture
def ray_generator() -> RayGenerator:
    """Create a RayGenerator instance."""
    return RayGenerator()


# =============================================================================
# Test: Ray Data Class
# =============================================================================


class TestRay:
    """Tests for the Ray dataclass."""

    def test_ray_creation(self):
        """Test Ray can be created with origin and direction."""
        origin = Vec3(0.0, 0.0, 5.0)
        direction = Vec3(0.0, 0.0, -1.0)
        ray = Ray(origin=origin, direction=direction)

        assert ray.origin.x == 0.0
        assert ray.origin.z == 5.0
        assert ray.direction.z == -1.0

    def test_ray_point_at(self):
        """Test ray.point_at() computes correct point along ray."""
        ray = Ray(origin=Vec3(0, 0, 0), direction=Vec3(0, 0, -1))

        p = ray.point_at(5.0)
        assert abs(p.x) < 1e-6
        assert abs(p.y) < 1e-6
        assert abs(p.z + 5.0) < 1e-6

    def test_ray_point_at_with_offset_origin(self):
        """Test point_at with non-zero origin."""
        ray = Ray(origin=Vec3(1, 2, 3), direction=Vec3(1, 0, 0))

        p = ray.point_at(10.0)
        assert abs(p.x - 11.0) < 1e-6
        assert abs(p.y - 2.0) < 1e-6
        assert abs(p.z - 3.0) < 1e-6


# =============================================================================
# Test: Vec3 Helper
# =============================================================================


class TestVec3:
    """Tests for the Vec3 helper class."""

    def test_vec3_creation(self):
        """Test Vec3 can be created with components."""
        v = Vec3(1.0, 2.0, 3.0)
        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0

    def test_vec3_length(self):
        """Test Vec3 length computation."""
        v = Vec3(3.0, 4.0, 0.0)
        assert abs(v.length() - 5.0) < 1e-6

    def test_vec3_normalized(self):
        """Test Vec3 normalization."""
        v = Vec3(3.0, 0.0, 0.0)
        n = v.normalized()
        assert abs(n.x - 1.0) < 1e-6
        assert abs(n.length() - 1.0) < 1e-6

    def test_vec3_normalized_zero_vector(self):
        """Test normalization of zero vector returns zero."""
        v = Vec3(0.0, 0.0, 0.0)
        n = v.normalized()
        assert abs(n.length()) < 1e-6

    def test_vec3_dot(self):
        """Test Vec3 dot product."""
        a = Vec3(1.0, 0.0, 0.0)
        b = Vec3(0.0, 1.0, 0.0)
        assert abs(a.dot(b)) < 1e-6  # Perpendicular

        c = Vec3(1.0, 2.0, 3.0)
        d = Vec3(1.0, 2.0, 3.0)
        assert abs(c.dot(d) - 14.0) < 1e-6

    def test_vec3_cross(self):
        """Test Vec3 cross product."""
        x = Vec3(1.0, 0.0, 0.0)
        y = Vec3(0.0, 1.0, 0.0)
        z = x.cross(y)
        assert abs(z.x) < 1e-6
        assert abs(z.y) < 1e-6
        assert abs(z.z - 1.0) < 1e-6

    def test_vec3_arithmetic(self):
        """Test Vec3 addition, subtraction, multiplication."""
        a = Vec3(1.0, 2.0, 3.0)
        b = Vec3(4.0, 5.0, 6.0)

        # Addition
        c = a + b
        assert abs(c.x - 5.0) < 1e-6
        assert abs(c.y - 7.0) < 1e-6
        assert abs(c.z - 9.0) < 1e-6

        # Subtraction
        d = b - a
        assert abs(d.x - 3.0) < 1e-6
        assert abs(d.y - 3.0) < 1e-6
        assert abs(d.z - 3.0) < 1e-6

        # Scalar multiplication
        e = a * 2.0
        assert abs(e.x - 2.0) < 1e-6
        assert abs(e.y - 4.0) < 1e-6
        assert abs(e.z - 6.0) < 1e-6

    def test_vec3_approx_equal(self):
        """Test Vec3 approximate equality."""
        a = Vec3(1.0, 2.0, 3.0)
        b = Vec3(1.000001, 2.000001, 3.000001)
        assert a.approx_equal(b, epsilon=1e-5)


# =============================================================================
# Test: RayGenerator Center Ray
# =============================================================================


class TestRayGeneratorCenterRay:
    """Tests for ray generation at screen center."""

    def test_center_ray_direction(self, ray_generator, default_camera):
        """Center ray should point straight ahead (normalized forward)."""
        ray = ray_generator.generate_ray(0.0, 0.0, default_camera)

        # Camera at (0,0,5) looking at (0,0,0): forward is (0,0,-1)
        assert abs(ray.direction.x) < 1e-6
        assert abs(ray.direction.y) < 1e-6
        assert abs(ray.direction.z + 1.0) < 1e-3

    def test_center_ray_origin(self, ray_generator, default_camera):
        """Ray origin should be at camera position."""
        ray = ray_generator.generate_ray(0.0, 0.0, default_camera)

        assert abs(ray.origin.x) < 1e-6
        assert abs(ray.origin.y) < 1e-6
        assert abs(ray.origin.z - 5.0) < 1e-6

    def test_center_ray_is_normalized(self, ray_generator, default_camera):
        """Ray direction should have unit length."""
        ray = ray_generator.generate_ray(0.0, 0.0, default_camera)
        assert abs(ray.direction.length() - 1.0) < 1e-6


# =============================================================================
# Test: RayGenerator Corner Rays
# =============================================================================


class TestRayGeneratorCornerRays:
    """Tests for ray generation at screen corners."""

    def test_top_left_corner(self, ray_generator, default_camera):
        """Top-left corner ray should point up and left."""
        ray = ray_generator.generate_ray(-1.0, 1.0, default_camera)

        # Should have negative X (left) and positive Y (up) components
        assert ray.direction.x < 0
        assert ray.direction.y > 0
        assert ray.direction.z < 0  # Still going forward

    def test_top_right_corner(self, ray_generator, default_camera):
        """Top-right corner ray should point up and right."""
        ray = ray_generator.generate_ray(1.0, 1.0, default_camera)

        assert ray.direction.x > 0  # Right
        assert ray.direction.y > 0  # Up
        assert ray.direction.z < 0

    def test_bottom_left_corner(self, ray_generator, default_camera):
        """Bottom-left corner ray should point down and left."""
        ray = ray_generator.generate_ray(-1.0, -1.0, default_camera)

        assert ray.direction.x < 0  # Left
        assert ray.direction.y < 0  # Down
        assert ray.direction.z < 0

    def test_bottom_right_corner(self, ray_generator, default_camera):
        """Bottom-right corner ray should point down and right."""
        ray = ray_generator.generate_ray(1.0, -1.0, default_camera)

        assert ray.direction.x > 0  # Right
        assert ray.direction.y < 0  # Down
        assert ray.direction.z < 0

    def test_corner_rays_symmetric(self, ray_generator, default_camera):
        """Corner rays should be symmetric about center."""
        tl = ray_generator.generate_ray(-1.0, 1.0, default_camera)
        tr = ray_generator.generate_ray(1.0, 1.0, default_camera)
        bl = ray_generator.generate_ray(-1.0, -1.0, default_camera)
        br = ray_generator.generate_ray(1.0, -1.0, default_camera)

        # X components should be symmetric
        assert abs(tl.direction.x + tr.direction.x) < 1e-6
        assert abs(bl.direction.x + br.direction.x) < 1e-6

        # Y components should be symmetric
        assert abs(tl.direction.y + bl.direction.y) < 1e-6
        assert abs(tr.direction.y + br.direction.y) < 1e-6


# =============================================================================
# Test: FOV Variations
# =============================================================================


class TestFOVVariations:
    """Tests for different field of view settings."""

    def test_narrow_fov(self, ray_generator):
        """Narrow FOV should produce rays closer to center."""
        camera = CameraNode(
            origin=Vec3Node(0.0, 0.0, 5.0),
            look_at=Vec3Node(0.0, 0.0, 0.0),
            up=Vec3Node(0.0, 1.0, 0.0),
            fov=FloatNode(30.0),  # Narrow FOV
            aspect_ratio=FloatNode(1.0),
        )

        ray = ray_generator.generate_ray(1.0, 1.0, camera)
        # With narrow FOV, corner ray should be closer to forward
        assert abs(ray.direction.z) > 0.9  # More forward-facing

    def test_wide_fov(self, ray_generator):
        """Wide FOV should spread rays more."""
        camera = CameraNode(
            origin=Vec3Node(0.0, 0.0, 5.0),
            look_at=Vec3Node(0.0, 0.0, 0.0),
            up=Vec3Node(0.0, 1.0, 0.0),
            fov=FloatNode(120.0),  # Wide FOV
            aspect_ratio=FloatNode(1.0),
        )

        ray = ray_generator.generate_ray(1.0, 1.0, camera)
        # With wide FOV, corner ray should spread more
        assert abs(ray.direction.z) < 0.7

    def test_fov_90_degrees(self, ray_generator):
        """90-degree FOV with aspect 1 should make corner at 45 degrees."""
        camera = CameraNode(
            origin=Vec3Node(0.0, 0.0, 5.0),
            look_at=Vec3Node(0.0, 0.0, 0.0),
            up=Vec3Node(0.0, 1.0, 0.0),
            fov=FloatNode(90.0),
            aspect_ratio=FloatNode(1.0),
        )

        # At edge (u=0, v=1), vertical angle should be 45 degrees
        ray = ray_generator.generate_ray(0.0, 1.0, camera)
        # tan(45) = 1, so y/z ratio should be ~1 at the edge
        ratio = abs(ray.direction.y / ray.direction.z)
        assert abs(ratio - 1.0) < 0.01


# =============================================================================
# Test: Aspect Ratio Variations
# =============================================================================


class TestAspectRatioVariations:
    """Tests for different aspect ratios."""

    def test_square_aspect(self, ray_generator):
        """Square aspect ratio should have symmetric spread."""
        camera = CameraNode(
            origin=Vec3Node(0.0, 0.0, 5.0),
            look_at=Vec3Node(0.0, 0.0, 0.0),
            up=Vec3Node(0.0, 1.0, 0.0),
            fov=FloatNode(60.0),
            aspect_ratio=FloatNode(1.0),
        )

        ray_h = ray_generator.generate_ray(1.0, 0.0, camera)  # Right edge
        ray_v = ray_generator.generate_ray(0.0, 1.0, camera)  # Top edge

        # For square aspect, horizontal and vertical spread should match
        assert abs(ray_h.direction.x - ray_v.direction.y) < 0.01

    def test_wide_aspect(self, ray_generator):
        """Wide aspect ratio should spread more horizontally."""
        camera = CameraNode(
            origin=Vec3Node(0.0, 0.0, 5.0),
            look_at=Vec3Node(0.0, 0.0, 0.0),
            up=Vec3Node(0.0, 1.0, 0.0),
            fov=FloatNode(60.0),
            aspect_ratio=FloatNode(2.0),  # 2:1 aspect
        )

        ray_h = ray_generator.generate_ray(1.0, 0.0, camera)
        ray_v = ray_generator.generate_ray(0.0, 1.0, camera)

        # Horizontal spread should be greater than vertical spread
        # With 2:1 aspect, x component at edge should be larger than y component
        assert ray_h.direction.x > ray_v.direction.y

    def test_tall_aspect(self, ray_generator):
        """Tall aspect ratio should spread more vertically."""
        camera = CameraNode(
            origin=Vec3Node(0.0, 0.0, 5.0),
            look_at=Vec3Node(0.0, 0.0, 0.0),
            up=Vec3Node(0.0, 1.0, 0.0),
            fov=FloatNode(60.0),
            aspect_ratio=FloatNode(0.5),  # 1:2 aspect
        )

        ray_h = ray_generator.generate_ray(1.0, 0.0, camera)
        ray_v = ray_generator.generate_ray(0.0, 1.0, camera)

        # Horizontal spread should be less than vertical spread
        assert ray_h.direction.x < ray_v.direction.y


# =============================================================================
# Test: Camera Position and Orientation
# =============================================================================


class TestCameraPositionOrientation:
    """Tests for different camera positions and orientations."""

    def test_camera_translated(self, ray_generator):
        """Camera at different position should still produce correct rays."""
        camera = CameraNode(
            origin=Vec3Node(10.0, 5.0, 3.0),
            look_at=Vec3Node(10.0, 5.0, 0.0),  # Looking along -Z
            up=Vec3Node(0.0, 1.0, 0.0),
            fov=FloatNode(60.0),
            aspect_ratio=FloatNode(1.0),
        )

        ray = ray_generator.generate_ray(0.0, 0.0, camera)

        # Origin should be camera position
        assert abs(ray.origin.x - 10.0) < 1e-6
        assert abs(ray.origin.y - 5.0) < 1e-6
        assert abs(ray.origin.z - 3.0) < 1e-6

        # Direction should point forward
        assert abs(ray.direction.z + 1.0) < 0.01

    def test_camera_looking_up(self, ray_generator):
        """Camera looking up should produce upward center ray."""
        camera = CameraNode(
            origin=Vec3Node(0.0, 0.0, 0.0),
            look_at=Vec3Node(0.0, 10.0, 0.0),  # Looking up +Y
            up=Vec3Node(0.0, 0.0, -1.0),  # Up is -Z when looking up
            fov=FloatNode(60.0),
            aspect_ratio=FloatNode(1.0),
        )

        ray = ray_generator.generate_ray(0.0, 0.0, camera)

        # Center ray should point up
        assert abs(ray.direction.y - 1.0) < 0.01
        assert abs(ray.direction.x) < 0.01
        assert abs(ray.direction.z) < 0.01

    def test_camera_looking_along_x(self, ray_generator):
        """Camera looking along +X should produce rightward center ray."""
        camera = CameraNode(
            origin=Vec3Node(0.0, 0.0, 0.0),
            look_at=Vec3Node(10.0, 0.0, 0.0),  # Looking along +X
            up=Vec3Node(0.0, 1.0, 0.0),
            fov=FloatNode(60.0),
            aspect_ratio=FloatNode(1.0),
        )

        ray = ray_generator.generate_ray(0.0, 0.0, camera)

        # Center ray should point along +X
        assert abs(ray.direction.x - 1.0) < 0.01
        assert abs(ray.direction.y) < 0.01
        assert abs(ray.direction.z) < 0.01


# =============================================================================
# Test: Pixel to UV Conversion
# =============================================================================


class TestPixelToUV:
    """Tests for pixel coordinate to UV conversion."""

    def test_center_pixel(self, ray_generator):
        """Center pixel should map to (0, 0) UV."""
        u, v = ray_generator.pixel_to_uv(100, 100, 200, 200)
        assert abs(u - 0.005) < 0.01  # Close to center
        assert abs(v - (-0.005)) < 0.01

    def test_corner_pixels(self, ray_generator):
        """Corner pixels should map to near (-1, -1) to (1, 1)."""
        # Top-left (0, 0) in pixel coordinates
        u_tl, v_tl = ray_generator.pixel_to_uv(0, 0, 100, 100)
        assert u_tl < -0.9  # Near -1
        assert v_tl > 0.9   # Near +1 (Y is flipped)

        # Bottom-right (99, 99)
        u_br, v_br = ray_generator.pixel_to_uv(99, 99, 100, 100)
        assert u_br > 0.9   # Near +1
        assert v_br < -0.9  # Near -1

    def test_pixel_center_sampling(self, ray_generator):
        """Pixel coordinates should sample pixel centers."""
        # First pixel center should be at 0.5/width
        u, v = ray_generator.pixel_to_uv(0, 0, 2, 2)
        # With 2 pixels, centers are at 0.25 and 0.75 in [0,1]
        # Mapped to [-1,1]: -0.5 and 0.5
        assert abs(u + 0.5) < 0.01
        assert abs(v - 0.5) < 0.01


# =============================================================================
# Test: WGSL Code Generation
# =============================================================================


class TestWGSLGeneration:
    """Tests for WGSL code generation."""

    def test_generate_ray_wgsl_contains_function(self):
        """Generated WGSL should contain generate_ray function."""
        wgsl = generate_ray_wgsl()
        assert "fn generate_ray(" in wgsl

    def test_generate_ray_wgsl_contains_struct(self):
        """Generated WGSL should contain Ray struct."""
        wgsl = generate_ray_wgsl()
        assert "struct Ray" in wgsl

    def test_generate_ray_wgsl_contains_create_ray(self):
        """Generated WGSL should contain create_ray helper."""
        wgsl = generate_ray_wgsl()
        assert "fn create_ray(" in wgsl

    def test_generate_ray_wgsl_inline(self, default_camera):
        """Inline WGSL should contain camera constants."""
        wgsl = generate_ray_wgsl_inline(default_camera)
        assert "CAMERA_ORIGIN" in wgsl
        assert "CAMERA_FORWARD" in wgsl
        assert "generate_ray_inline" in wgsl


# =============================================================================
# Test: Camera Validation
# =============================================================================


class TestCameraValidation:
    """Tests for camera parameter validation."""

    def test_valid_camera(self, default_camera):
        """Valid camera should have no errors."""
        errors = validate_camera(default_camera)
        assert len(errors) == 0

    def test_zero_fov_invalid(self):
        """Zero FOV should be invalid."""
        camera = CameraNode(
            origin=Vec3Node(0.0, 0.0, 5.0),
            look_at=Vec3Node(0.0, 0.0, 0.0),
            up=Vec3Node(0.0, 1.0, 0.0),
            fov=FloatNode(0.0),
            aspect_ratio=FloatNode(1.0),
        )
        errors = validate_camera(camera)
        assert len(errors) > 0
        assert any("FOV" in e for e in errors)

    def test_negative_fov_invalid(self):
        """Negative FOV should be invalid."""
        camera = CameraNode(
            origin=Vec3Node(0.0, 0.0, 5.0),
            look_at=Vec3Node(0.0, 0.0, 0.0),
            up=Vec3Node(0.0, 1.0, 0.0),
            fov=FloatNode(-60.0),
            aspect_ratio=FloatNode(1.0),
        )
        errors = validate_camera(camera)
        assert len(errors) > 0

    def test_fov_180_invalid(self):
        """FOV >= 180 should be invalid."""
        camera = CameraNode(
            origin=Vec3Node(0.0, 0.0, 5.0),
            look_at=Vec3Node(0.0, 0.0, 0.0),
            up=Vec3Node(0.0, 1.0, 0.0),
            fov=FloatNode(180.0),
            aspect_ratio=FloatNode(1.0),
        )
        errors = validate_camera(camera)
        assert len(errors) > 0

    def test_zero_aspect_invalid(self):
        """Zero aspect ratio should be invalid."""
        camera = CameraNode(
            origin=Vec3Node(0.0, 0.0, 5.0),
            look_at=Vec3Node(0.0, 0.0, 0.0),
            up=Vec3Node(0.0, 1.0, 0.0),
            fov=FloatNode(60.0),
            aspect_ratio=FloatNode(0.0),
        )
        errors = validate_camera(camera)
        assert len(errors) > 0

    def test_same_origin_lookat_invalid(self):
        """Same origin and look_at should be invalid."""
        camera = CameraNode(
            origin=Vec3Node(0.0, 0.0, 0.0),
            look_at=Vec3Node(0.0, 0.0, 0.0),
            up=Vec3Node(0.0, 1.0, 0.0),
            fov=FloatNode(60.0),
            aspect_ratio=FloatNode(1.0),
        )
        errors = validate_camera(camera)
        assert len(errors) > 0

    def test_zero_up_vector_invalid(self):
        """Zero up vector should be invalid."""
        camera = CameraNode(
            origin=Vec3Node(0.0, 0.0, 5.0),
            look_at=Vec3Node(0.0, 0.0, 0.0),
            up=Vec3Node(0.0, 0.0, 0.0),
            fov=FloatNode(60.0),
            aspect_ratio=FloatNode(1.0),
        )
        errors = validate_camera(camera)
        assert len(errors) > 0

    def test_up_parallel_to_forward_invalid(self):
        """Up vector parallel to forward should be invalid."""
        camera = CameraNode(
            origin=Vec3Node(0.0, 0.0, 5.0),
            look_at=Vec3Node(0.0, 0.0, 0.0),
            up=Vec3Node(0.0, 0.0, -1.0),  # Parallel to forward
            fov=FloatNode(60.0),
            aspect_ratio=FloatNode(1.0),
        )
        errors = validate_camera(camera)
        assert len(errors) > 0


# =============================================================================
# Test: CameraParams
# =============================================================================


class TestCameraParams:
    """Tests for CameraParams helper class."""

    def test_from_camera_node(self, default_camera):
        """CameraParams should extract values from CameraNode."""
        params = CameraParams.from_camera_node(default_camera)

        assert abs(params.origin.z - 5.0) < 1e-6
        assert abs(params.look_at.z) < 1e-6
        assert abs(params.up.y - 1.0) < 1e-6
        assert abs(params.fov - 60.0) < 1e-6
        assert abs(params.aspect_ratio - 16.0 / 9.0) < 1e-6


# =============================================================================
# Test: Grid Ray Generation
# =============================================================================


class TestGridRayGeneration:
    """Tests for generating rays in a grid."""

    def test_generate_rays_grid_dimensions(self, ray_generator, default_camera):
        """Grid should have correct dimensions."""
        rays = ray_generator.generate_rays_grid(default_camera, 4, 3)

        assert len(rays) == 3  # 3 rows
        assert len(rays[0]) == 4  # 4 columns

    def test_generate_rays_grid_all_normalized(self, ray_generator, default_camera):
        """All rays in grid should be normalized."""
        rays = ray_generator.generate_rays_grid(default_camera, 4, 4)

        for row in rays:
            for ray in row:
                assert abs(ray.direction.length() - 1.0) < 1e-6

    def test_generate_rays_grid_all_from_origin(self, ray_generator, default_camera):
        """All rays should originate from camera position."""
        rays = ray_generator.generate_rays_grid(default_camera, 4, 4)

        for row in rays:
            for ray in row:
                assert abs(ray.origin.z - 5.0) < 1e-6
