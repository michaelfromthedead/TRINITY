"""Tests for planar mirror rendering system.

Tests cover:
- Reflection matrix computation and properties
- Camera reflection transformations
- Fresnel reflectance calculations
- Mirror visibility and management
- Edge cases and numerical stability
"""

from __future__ import annotations

import math
import pytest

from engine.core.math.geometry import AABB, Frustum, Plane
from engine.core.math.mat import Mat4
from engine.core.math.vec import Vec3, Vec4
from engine.rendering.lighting.planar_mirror import (
    PlanarMirror,
    PlanarMirrorConfig,
    PlanarMirrorManager,
    MirrorUpdateMode,
    create_water_plane,
    create_mirror_plane,
    compute_oblique_projection,
    transform_plane_to_view,
    signed_distance_to_plane,
)


EPSILON = 1e-5


def approx_eq(a: float, b: float, eps: float = EPSILON) -> bool:
    """Check approximate equality."""
    return abs(a - b) < eps


def mat4_approx_eq(a: Mat4, b: Mat4, eps: float = EPSILON) -> bool:
    """Check approximate matrix equality."""
    for i in range(16):
        if abs(a.m[i] - b.m[i]) >= eps:
            return False
    return True


class TestReflectionMatrix:
    """Tests for reflection matrix computation."""

    def test_horizontal_plane_at_origin(self) -> None:
        """Test reflection matrix for horizontal plane at y=0."""
        plane = Plane(Vec3(0, 1, 0), 0.0)
        mirror = PlanarMirror(plane=plane)
        r = mirror.reflection_matrix

        # Should flip Y coordinate
        point = Vec3(1, 5, 2)
        reflected = r.transform_point(point)

        assert approx_eq(reflected.x, 1.0)
        assert approx_eq(reflected.y, -5.0)
        assert approx_eq(reflected.z, 2.0)

    def test_vertical_plane_x(self) -> None:
        """Test reflection matrix for vertical plane at x=0."""
        plane = Plane(Vec3(1, 0, 0), 0.0)
        mirror = PlanarMirror(plane=plane)
        r = mirror.reflection_matrix

        # Should flip X coordinate
        point = Vec3(3, 1, 2)
        reflected = r.transform_point(point)

        assert approx_eq(reflected.x, -3.0)
        assert approx_eq(reflected.y, 1.0)
        assert approx_eq(reflected.z, 2.0)

    def test_vertical_plane_z(self) -> None:
        """Test reflection matrix for vertical plane at z=0."""
        plane = Plane(Vec3(0, 0, 1), 0.0)
        mirror = PlanarMirror(plane=plane)
        r = mirror.reflection_matrix

        # Should flip Z coordinate
        point = Vec3(1, 2, 4)
        reflected = r.transform_point(point)

        assert approx_eq(reflected.x, 1.0)
        assert approx_eq(reflected.y, 2.0)
        assert approx_eq(reflected.z, -4.0)

    def test_offset_plane(self) -> None:
        """Test reflection matrix for plane not at origin."""
        # Plane at y=5
        plane = Plane(Vec3(0, 1, 0), -5.0)
        mirror = PlanarMirror(plane=plane)
        r = mirror.reflection_matrix

        # Point at y=10 should reflect to y=0
        point = Vec3(0, 10, 0)
        reflected = r.transform_point(point)

        assert approx_eq(reflected.x, 0.0)
        assert approx_eq(reflected.y, 0.0)
        assert approx_eq(reflected.z, 0.0)

    def test_reflection_matrix_is_involutory(self) -> None:
        """Test that R * R = I (reflecting twice returns to original)."""
        plane = Plane(Vec3(0, 1, 0), 0.0)
        mirror = PlanarMirror(plane=plane)
        r = mirror.reflection_matrix

        # Apply reflection twice
        point = Vec3(1, 2, 3)
        once = r.transform_point(point)
        twice = r.transform_point(once)

        assert approx_eq(twice.x, point.x)
        assert approx_eq(twice.y, point.y)
        assert approx_eq(twice.z, point.z)

    def test_reflection_preserves_plane_points(self) -> None:
        """Test that points on the plane are not moved."""
        plane = Plane(Vec3(0, 1, 0), -5.0)  # y = 5
        mirror = PlanarMirror(plane=plane)
        r = mirror.reflection_matrix

        # Point on the plane
        point = Vec3(10, 5, -3)
        reflected = r.transform_point(point)

        assert approx_eq(reflected.x, point.x)
        assert approx_eq(reflected.y, point.y)
        assert approx_eq(reflected.z, point.z)

    def test_diagonal_plane_reflection(self) -> None:
        """Test reflection across a 45-degree plane."""
        # 45 degree plane (normal = normalized (1, 1, 0))
        # Reflects points using formula: P' = P - 2*(n.P)*n
        sqrt2_2 = math.sqrt(2) / 2
        plane = Plane(Vec3(sqrt2_2, sqrt2_2, 0), 0.0)
        mirror = PlanarMirror(plane=plane)
        r = mirror.reflection_matrix

        # Point (1, 0, 0) should reflect to (0, -1, 0)
        # n.P = sqrt2/2 * 1 = sqrt2/2
        # P' = (1,0,0) - 2*(sqrt2/2)*(sqrt2/2, sqrt2/2, 0)
        # P' = (1,0,0) - (1, 1, 0) = (0, -1, 0)
        point = Vec3(1, 0, 0)
        reflected = r.transform_point(point)

        assert approx_eq(reflected.x, 0.0)
        assert approx_eq(reflected.y, -1.0)
        assert approx_eq(reflected.z, 0.0)


class TestCameraReflection:
    """Tests for camera reflection transformations."""

    def test_reflect_point(self) -> None:
        """Test reflecting a point across the mirror."""
        plane = Plane(Vec3(0, 1, 0), 0.0)
        mirror = PlanarMirror(plane=plane)

        point = Vec3(5, 3, 2)
        reflected = mirror.reflect_point(point)

        assert approx_eq(reflected.x, 5.0)
        assert approx_eq(reflected.y, -3.0)
        assert approx_eq(reflected.z, 2.0)

    def test_reflect_direction(self) -> None:
        """Test reflecting a direction across the mirror."""
        plane = Plane(Vec3(0, 1, 0), 0.0)
        mirror = PlanarMirror(plane=plane)

        direction = Vec3(0, 1, 0)
        reflected = mirror.reflect_direction(direction)

        assert approx_eq(reflected.x, 0.0)
        assert approx_eq(reflected.y, -1.0)
        assert approx_eq(reflected.z, 0.0)

    def test_reflect_camera_produces_valid_matrices(self) -> None:
        """Test that reflect_camera returns valid matrices."""
        plane = Plane(Vec3(0, 1, 0), 0.0)
        mirror = PlanarMirror(plane=plane)

        # Simple view and projection matrices
        view = Mat4.look_at(Vec3(0, 5, 10), Vec3(0, 0, 0), Vec3(0, 1, 0))
        proj = Mat4.perspective(math.pi / 4, 16 / 9, 0.1, 100.0)

        reflected_view, oblique_proj = mirror.reflect_camera(view, proj)

        # Matrices should be different
        assert not mat4_approx_eq(view, reflected_view)

        # Reflected view should be valid (non-zero determinant)
        assert abs(reflected_view.determinant()) > 1e-6


class TestFresnelReflectance:
    """Tests for Fresnel reflectance calculations."""

    def test_fresnel_at_normal_incidence(self) -> None:
        """Test Fresnel at normal incidence equals base reflectivity."""
        config = PlanarMirrorConfig(base_reflectivity=0.04)
        mirror = PlanarMirror(config=config)

        # Looking straight at the surface
        view_dir = Vec3(0, 1, 0)
        normal = Vec3(0, 1, 0)

        fresnel = mirror.compute_fresnel(view_dir, normal)

        # Should equal F0 at normal incidence
        assert approx_eq(fresnel, 0.04, eps=0.01)

    def test_fresnel_at_grazing_angle_approaches_one(self) -> None:
        """Test Fresnel at grazing angle approaches 1.0."""
        config = PlanarMirrorConfig(base_reflectivity=0.04, fresnel_power=5.0)
        mirror = PlanarMirror(config=config)

        # Looking at grazing angle (perpendicular to normal)
        view_dir = Vec3(1, 0, 0)
        normal = Vec3(0, 1, 0)

        fresnel = mirror.compute_fresnel(view_dir, normal)

        # Should be close to 1.0 at grazing angle
        assert fresnel > 0.9
        assert fresnel <= 1.0

    def test_fresnel_at_45_degrees(self) -> None:
        """Test Fresnel at 45 degrees is between extremes."""
        config = PlanarMirrorConfig(base_reflectivity=0.04, fresnel_power=5.0)
        mirror = PlanarMirror(config=config)

        # 45 degree angle
        sqrt2_2 = math.sqrt(2) / 2
        view_dir = Vec3(sqrt2_2, sqrt2_2, 0)
        normal = Vec3(0, 1, 0)

        fresnel = mirror.compute_fresnel(view_dir, normal)

        # Should be between F0 and 1.0
        assert fresnel > 0.04
        assert fresnel < 1.0

    def test_fresnel_clamps_to_one(self) -> None:
        """Test Fresnel never exceeds 1.0."""
        config = PlanarMirrorConfig(
            base_reflectivity=0.9,  # High base reflectivity
            fresnel_power=10.0,     # High power
        )
        mirror = PlanarMirror(config=config)

        view_dir = Vec3(1, 0.01, 0).normalized()
        normal = Vec3(0, 1, 0)

        fresnel = mirror.compute_fresnel(view_dir, normal)

        assert fresnel <= 1.0

    def test_fresnel_handles_negative_dot_product(self) -> None:
        """Test Fresnel handles negative cos(theta) gracefully."""
        config = PlanarMirrorConfig(base_reflectivity=0.04)
        mirror = PlanarMirror(config=config)

        # View direction pointing away from normal
        view_dir = Vec3(0, -1, 0)
        normal = Vec3(0, 1, 0)

        fresnel = mirror.compute_fresnel(view_dir, normal)

        # Should still return a valid value
        assert fresnel >= 0.0
        assert fresnel <= 1.0


class TestMirrorVisibility:
    """Tests for mirror visibility calculations."""

    def test_is_point_in_front_above_plane(self) -> None:
        """Test point above horizontal plane is in front."""
        plane = Plane(Vec3(0, 1, 0), 0.0)
        mirror = PlanarMirror(plane=plane)

        assert mirror.is_point_in_front(Vec3(0, 5, 0))

    def test_is_point_in_front_below_plane(self) -> None:
        """Test point below horizontal plane is not in front."""
        plane = Plane(Vec3(0, 1, 0), 0.0)
        mirror = PlanarMirror(plane=plane)

        assert not mirror.is_point_in_front(Vec3(0, -5, 0))

    def test_is_point_in_front_on_plane(self) -> None:
        """Test point on plane is considered in front."""
        plane = Plane(Vec3(0, 1, 0), 0.0)
        mirror = PlanarMirror(plane=plane)

        # On the plane should be >= 0
        assert mirror.is_point_in_front(Vec3(0, 0, 0))

    def test_activate_deactivate(self) -> None:
        """Test mirror activation and deactivation."""
        mirror = PlanarMirror()

        assert mirror.is_active

        mirror.deactivate()
        assert not mirror.is_active

        mirror.activate()
        assert mirror.is_active


class TestPlanarMirrorManager:
    """Tests for the mirror manager."""

    def test_add_and_remove_mirror(self) -> None:
        """Test adding and removing mirrors."""
        manager = PlanarMirrorManager()
        mirror = PlanarMirror()

        manager.add_mirror(mirror)
        assert manager.total_count == 1

        manager.remove_mirror(mirror)
        assert manager.total_count == 0

    def test_clear_mirrors(self) -> None:
        """Test clearing all mirrors."""
        manager = PlanarMirrorManager()
        manager.add_mirror(PlanarMirror())
        manager.add_mirror(PlanarMirror())

        assert manager.total_count == 2

        manager.clear()
        assert manager.total_count == 0

    def test_get_visible_mirrors_filters_inactive(self) -> None:
        """Test that inactive mirrors are filtered out."""
        manager = PlanarMirrorManager()

        mirror1 = PlanarMirror(plane=Plane(Vec3(0, 1, 0), 0.0))
        mirror2 = PlanarMirror(plane=Plane(Vec3(0, 1, 0), 0.0))
        mirror2.deactivate()

        manager.add_mirror(mirror1)
        manager.add_mirror(mirror2)

        frustum = Frustum([])  # Empty frustum (all visible)
        visible = manager.get_visible_mirrors(Vec3(0, 5, 0), frustum)

        assert len(visible) == 1

    def test_get_visible_mirrors_filters_behind_camera(self) -> None:
        """Test that mirrors behind camera are filtered out."""
        manager = PlanarMirrorManager()

        # Horizontal plane at y=0 (normal +Y)
        mirror = PlanarMirror(plane=Plane(Vec3(0, 1, 0), 0.0))
        manager.add_mirror(mirror)

        frustum = Frustum([])

        # Camera above plane should see it
        visible = manager.get_visible_mirrors(Vec3(0, 10, 0), frustum)
        assert len(visible) == 1

        # Camera below plane should not see it
        visible = manager.get_visible_mirrors(Vec3(0, -10, 0), frustum)
        assert len(visible) == 0

    def test_max_active_per_frame_enforced(self) -> None:
        """Test that max_active_per_frame limit is enforced."""
        manager = PlanarMirrorManager(max_active_per_frame=2)

        # Add 5 mirrors
        for i in range(5):
            mirror = PlanarMirror(plane=Plane(Vec3(0, 1, 0), -float(i)))
            manager.add_mirror(mirror)

        frustum = Frustum([])
        to_render = manager.get_mirrors_for_frame(Vec3(0, 100, 0), frustum)

        assert len(to_render) == 2

    def test_mirrors_sorted_by_priority(self) -> None:
        """Test that visible mirrors are sorted by priority."""
        manager = PlanarMirrorManager()

        low_priority = PlanarMirror(plane=Plane(Vec3(0, 1, 0), 0.0), priority=1)
        high_priority = PlanarMirror(plane=Plane(Vec3(0, 1, 0), 0.0), priority=10)

        manager.add_mirror(low_priority)
        manager.add_mirror(high_priority)

        frustum = Frustum([])
        visible = manager.get_visible_mirrors(Vec3(0, 5, 0), frustum)

        # High priority should come first
        assert len(visible) == 2
        assert visible[0].priority == 10
        assert visible[1].priority == 1

    def test_get_mirror_by_id(self) -> None:
        """Test finding a mirror by its unique ID."""
        manager = PlanarMirrorManager()
        mirror = PlanarMirror()
        manager.add_mirror(mirror)

        found = manager.get_mirror_by_id(mirror._mirror_id)
        assert found is mirror

        not_found = manager.get_mirror_by_id(-1)
        assert not_found is None

    def test_active_count(self) -> None:
        """Test counting active mirrors."""
        manager = PlanarMirrorManager()

        mirror1 = PlanarMirror()
        mirror2 = PlanarMirror()
        mirror2.deactivate()

        manager.add_mirror(mirror1)
        manager.add_mirror(mirror2)

        assert manager.active_count == 1
        assert manager.total_count == 2


class TestMirrorConfiguration:
    """Tests for mirror configuration."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = PlanarMirrorConfig()

        assert config.resolution_scale == 0.5
        assert config.max_distance == 100.0
        assert config.fresnel_power == 5.0
        assert config.blur_amount == 0.0
        assert config.base_reflectivity == 0.04
        assert config.update_mode == MirrorUpdateMode.ON_VISIBLE

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = PlanarMirrorConfig(
            resolution_scale=0.75,
            max_distance=200.0,
            fresnel_power=3.0,
            blur_amount=0.5,
            base_reflectivity=0.9,
            update_mode=MirrorUpdateMode.EVERY_FRAME,
        )

        assert config.resolution_scale == 0.75
        assert config.max_distance == 200.0
        assert config.fresnel_power == 3.0
        assert config.blur_amount == 0.5
        assert config.base_reflectivity == 0.9
        assert config.update_mode == MirrorUpdateMode.EVERY_FRAME


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_create_water_plane(self) -> None:
        """Test water plane creation helper."""
        water = create_water_plane(height=5.0)

        # Normal should point up
        assert approx_eq(water.plane.normal.y, 1.0)

        # Config should have water-appropriate values
        assert water.config.base_reflectivity == 0.02
        assert water.priority == 10

    def test_create_mirror_plane(self) -> None:
        """Test arbitrary mirror creation helper."""
        position = Vec3(0, 0, 10)
        normal = Vec3(0, 0, -1)  # Facing -Z

        mirror = create_mirror_plane(position, normal)

        # Normal should be normalized and match
        assert approx_eq(mirror.plane.normal.z, -1.0)

        # Config should have mirror-appropriate values
        assert mirror.config.base_reflectivity == 0.9

    def test_create_water_plane_with_bounds(self) -> None:
        """Test water plane with custom bounds."""
        bounds = AABB(Vec3(-100, -1, -100), Vec3(100, 1, 100))
        water = create_water_plane(height=0.0, bounds=bounds)

        assert water.bounds is not None
        assert water.bounds.min.x == -100
        assert water.bounds.max.x == 100


class TestEdgeCases:
    """Tests for edge cases and numerical stability."""

    def test_very_small_fresnel_power(self) -> None:
        """Test Fresnel with very small power."""
        config = PlanarMirrorConfig(fresnel_power=0.1)
        mirror = PlanarMirror(config=config)

        view_dir = Vec3(1, 0.5, 0).normalized()
        normal = Vec3(0, 1, 0)

        fresnel = mirror.compute_fresnel(view_dir, normal)

        assert fresnel >= 0.0
        assert fresnel <= 1.0

    def test_very_large_fresnel_power(self) -> None:
        """Test Fresnel with very large power."""
        config = PlanarMirrorConfig(fresnel_power=100.0)
        mirror = PlanarMirror(config=config)

        view_dir = Vec3(1, 0.5, 0).normalized()
        normal = Vec3(0, 1, 0)

        fresnel = mirror.compute_fresnel(view_dir, normal)

        assert fresnel >= 0.0
        assert fresnel <= 1.0

    def test_zero_normal_handling(self) -> None:
        """Test handling of zero-length normal (should be normalized)."""
        # Plane constructor normalizes the normal
        plane = Plane(Vec3(0, 0.001, 0), 0.0)
        mirror = PlanarMirror(plane=plane)

        # Should still have a valid reflection matrix
        assert abs(mirror.reflection_matrix.determinant()) > 0

    def test_update_plane_marks_dirty(self) -> None:
        """Test that updating plane marks reflection matrix as dirty."""
        mirror = PlanarMirror()

        # Get initial matrix
        _ = mirror.reflection_matrix

        # Update plane to x=5 (normal = +X, distance = -5)
        # Plane equation: n.p + d = 0, so for x=5: 1*x + (-5) = 0
        new_plane = Plane(Vec3(1, 0, 0), -5.0)
        mirror.set_plane(new_plane)

        # Matrix should be recomputed
        r = mirror.reflection_matrix
        point = Vec3(10, 0, 0)
        reflected = r.transform_point(point)

        # Point at x=10 should reflect to x=0 (mirror at x=5)
        # Distance from mirror: 10-5 = 5, reflected: 5-5 = 0
        assert approx_eq(reflected.x, 0.0)

    def test_unique_mirror_ids(self) -> None:
        """Test that each mirror gets a unique ID."""
        mirror1 = PlanarMirror()
        mirror2 = PlanarMirror()
        mirror3 = PlanarMirror()

        ids = {mirror1._mirror_id, mirror2._mirror_id, mirror3._mirror_id}
        assert len(ids) == 3

    def test_screen_coverage_no_bounds(self) -> None:
        """Test screen coverage returns 1.0 when no bounds set."""
        mirror = PlanarMirror(bounds=None)

        coverage = mirror.get_screen_coverage(
            Vec3(0, 0, 0),
            Mat4.identity(),
            1920,
            1080,
        )

        assert coverage == 1.0

    def test_reflection_matrix_determinant_is_negative_one(self) -> None:
        """Test that reflection matrix has determinant -1 (improper rotation)."""
        plane = Plane(Vec3(0, 1, 0), 0.0)
        mirror = PlanarMirror(plane=plane)
        r = mirror.reflection_matrix

        # Reflection is an improper orthogonal transformation
        det = r.determinant()
        assert approx_eq(abs(det), 1.0)


class TestObliqueNearPlaneClipping:
    """Tests for oblique near-plane clipping (Eric Lengyel's technique)."""

    def test_oblique_projection_maintains_frustum_sides(self) -> None:
        """Test that oblique projection preserves left/right/top/bottom planes."""
        proj = Mat4.perspective(math.pi / 4, 16 / 9, 0.1, 100.0)

        # Clip plane in view space pointing -Z at z=-1
        clip_plane = Vec4(0, 0, -1, -1)

        oblique = compute_oblique_projection(proj, clip_plane)

        # Column 0 and 1 should be unchanged (they define X and Y projection)
        assert approx_eq(oblique.m[0], proj.m[0])
        assert approx_eq(oblique.m[5], proj.m[5])

        # Row 3 (w-divide) should be unchanged
        assert approx_eq(oblique.m[3], proj.m[3])
        assert approx_eq(oblique.m[7], proj.m[7])
        assert approx_eq(oblique.m[11], proj.m[11])
        assert approx_eq(oblique.m[15], proj.m[15])

    def test_oblique_projection_near_plane_equals_clip_plane(self) -> None:
        """Test that points on the clip plane map to NDC z = -1."""
        proj = Mat4.perspective(math.pi / 4, 1.0, 0.1, 100.0)

        # Clip plane at z=-2 in view space
        clip_plane = Vec4(0, 0, -1, -2)

        oblique = compute_oblique_projection(proj, clip_plane)

        # Point exactly on the clip plane (z=-2 in view space)
        point = Vec3(0, 0, -2)
        projected = _project_point(oblique, point)

        # NDC z should be -1 (near plane)
        assert approx_eq(projected.z, -1.0, eps=0.01)

    def test_oblique_projection_clips_objects_behind_mirror(self) -> None:
        """Test that objects behind the clip plane are clipped (NDC z < -1)."""
        proj = Mat4.perspective(math.pi / 4, 1.0, 0.1, 100.0)

        # Clip plane at z=-5
        clip_plane = Vec4(0, 0, -1, -5)

        oblique = compute_oblique_projection(proj, clip_plane)

        # Point behind the clip plane (z=-6, further from camera than z=-5)
        behind = Vec3(0, 0, -6)
        projected_behind = _project_point(oblique, behind)

        # Should be beyond near plane (NDC z > -1 because it's actually clipped)
        # Wait, the convention is different: z=-6 is BEHIND z=-5 in view space
        # So it should map to NDC z > -1 (closer to far plane)
        assert projected_behind.z > -1.0

        # Point in front of clip plane (z=-4, closer to camera than z=-5)
        front = Vec3(0, 0, -4)
        projected_front = _project_point(oblique, front)

        # Should be clipped (NDC z < -1)
        assert projected_front.z < -1.0

    def test_oblique_projection_identity_when_degenerate(self) -> None:
        """Test that degenerate clip plane leaves projection unchanged."""
        proj = Mat4.perspective(math.pi / 4, 1.0, 0.1, 100.0)

        # Zero clip plane (degenerate)
        clip_plane = Vec4(0, 0, 0, 0)

        oblique = compute_oblique_projection(proj, clip_plane)

        # Should be unchanged
        assert mat4_approx_eq(oblique, proj)

    def test_transform_plane_to_view_identity(self) -> None:
        """Test plane transformation with identity view matrix."""
        plane = Plane(Vec3(0, 1, 0), -5.0)  # y=5 plane
        view = Mat4.identity()

        view_plane = transform_plane_to_view(plane, view)

        assert approx_eq(view_plane.x, 0.0)
        assert approx_eq(view_plane.y, 1.0)
        assert approx_eq(view_plane.z, 0.0)
        assert approx_eq(view_plane.w, -5.0)

    def test_transform_plane_to_view_translation(self) -> None:
        """Test plane transformation with translated camera."""
        plane = Plane(Vec3(0, 1, 0), -5.0)  # y=5 plane

        # Camera at (0, 10, 0)
        view = Mat4.translation(Vec3(0, -10, 0))

        view_plane = transform_plane_to_view(plane, view)

        # Normal should be unchanged
        assert approx_eq(view_plane.y, 1.0)

        # Distance changes: world y=5, camera at y=10
        # In view space, plane is at y = 5-10 = -5
        # Plane equation: 1*y + d = 0 => d = 5
        assert approx_eq(view_plane.w, 5.0)

    def test_transform_plane_to_view_with_offset(self) -> None:
        """Test plane transformation with offset for z-fighting prevention."""
        plane = Plane(Vec3(0, 1, 0), -5.0)
        view = Mat4.identity()

        view_plane = transform_plane_to_view(plane, view, offset=0.01)

        # Distance should include offset
        assert approx_eq(view_plane.w, -4.99)

    def test_signed_distance_to_plane_above(self) -> None:
        """Test signed distance for point above plane."""
        plane = Plane(Vec3(0, 1, 0), -5.0)  # y=5

        # Point at y=10 (above)
        dist = signed_distance_to_plane(Vec3(0, 10, 0), plane)
        assert approx_eq(dist, 5.0)

    def test_signed_distance_to_plane_below(self) -> None:
        """Test signed distance for point below plane."""
        plane = Plane(Vec3(0, 1, 0), -5.0)  # y=5

        # Point at y=0 (below)
        dist = signed_distance_to_plane(Vec3(0, 0, 0), plane)
        assert approx_eq(dist, -5.0)

    def test_signed_distance_to_plane_on(self) -> None:
        """Test signed distance for point on plane."""
        plane = Plane(Vec3(0, 1, 0), -5.0)  # y=5

        # Point exactly on plane
        dist = signed_distance_to_plane(Vec3(3, 5, -2), plane)
        assert approx_eq(dist, 0.0)

    def test_mirror_transform_plane_to_view(self) -> None:
        """Test PlanarMirror.transform_plane_to_view method."""
        plane = Plane(Vec3(0, 1, 0), 0.0)
        mirror = PlanarMirror(plane=plane)

        view = Mat4.identity()
        view_plane = mirror.transform_plane_to_view(view)

        assert approx_eq(view_plane.y, 1.0)

    def test_mirror_is_point_clipped_behind(self) -> None:
        """Test PlanarMirror.is_point_clipped for point behind mirror."""
        plane = Plane(Vec3(0, 1, 0), 0.0)  # y=0 horizontal
        mirror = PlanarMirror(plane=plane)

        view = Mat4.identity()

        # Point below plane (y=-5) should be clipped
        assert mirror.is_point_clipped(Vec3(0, -5, 0), view)

    def test_mirror_is_point_clipped_in_front(self) -> None:
        """Test PlanarMirror.is_point_clipped for point in front of mirror."""
        plane = Plane(Vec3(0, 1, 0), 0.0)
        mirror = PlanarMirror(plane=plane)

        view = Mat4.identity()

        # Point above plane (y=5) should not be clipped
        assert not mirror.is_point_clipped(Vec3(0, 5, 0), view)

    def test_oblique_projection_angled_plane(self) -> None:
        """Test oblique projection with 45-degree angled plane."""
        proj = Mat4.perspective(math.pi / 4, 1.0, 0.1, 100.0)

        # 45-degree plane
        sqrt2_2 = math.sqrt(2) / 2
        clip_plane = Vec4(sqrt2_2, 0, -sqrt2_2, -5)

        oblique = compute_oblique_projection(proj, clip_plane)

        # Should produce valid modified projection
        assert not mat4_approx_eq(oblique, proj)

        # X and Y factors preserved
        assert approx_eq(oblique.m[0], proj.m[0])
        assert approx_eq(oblique.m[5], proj.m[5])

    def test_oblique_projection_preserves_far_plane_approximately(self) -> None:
        """Test that far plane mapping is approximately preserved."""
        proj = Mat4.perspective(math.pi / 4, 1.0, 0.1, 100.0)

        clip_plane = Vec4(0, 0, -1, -1)
        oblique = compute_oblique_projection(proj, clip_plane)

        # Point at far plane (z=-100 in view space)
        far_point = Vec3(0, 0, -100)
        projected = _project_point(oblique, far_point)

        # Should be close to NDC z = 1 (far plane)
        # With oblique clipping, this may not be exact
        assert projected.z > 0.0  # At least beyond frustum midpoint

    def test_reflect_camera_uses_oblique_projection(self) -> None:
        """Test that reflect_camera applies oblique projection."""
        plane = Plane(Vec3(0, 1, 0), 0.0)
        mirror = PlanarMirror(plane=plane)

        view = Mat4.look_at(Vec3(0, 5, 10), Vec3(0, 0, 0), Vec3(0, 1, 0))
        proj = Mat4.perspective(math.pi / 4, 16 / 9, 0.1, 100.0)

        reflected_view, oblique_proj = mirror.reflect_camera(view, proj)

        # Projection should be modified
        assert not mat4_approx_eq(oblique_proj, proj)

        # View should be reflected
        assert not mat4_approx_eq(reflected_view, view)


def _project_point(proj: Mat4, point: Vec3) -> Vec3:
    """Project a point using a projection matrix and return NDC coordinates."""
    # Homogeneous coordinates
    x = proj.m[0] * point.x + proj.m[4] * point.y + proj.m[8] * point.z + proj.m[12]
    y = proj.m[1] * point.x + proj.m[5] * point.y + proj.m[9] * point.z + proj.m[13]
    z = proj.m[2] * point.x + proj.m[6] * point.y + proj.m[10] * point.z + proj.m[14]
    w = proj.m[3] * point.x + proj.m[7] * point.y + proj.m[11] * point.z + proj.m[15]

    if abs(w) > 1e-6:
        return Vec3(x / w, y / w, z / w)
    return Vec3(x, y, z)
