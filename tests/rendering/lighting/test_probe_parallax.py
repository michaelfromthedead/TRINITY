"""Tests for parallax correction for reflection probes.

Covers:
- ProbeBox AABB operations (contains, corners, transforms)
- Ray-box intersection math (slab method)
- Box projection direction correction (Lagarde/UE4)
- Inner/outer radius blending
- Edge cases (parallel rays, corner hits, degenerate cases)
- Integration with probe sampling
- Performance characteristics
"""

from __future__ import annotations

import math
import time
from typing import Optional

import pytest

from engine.core.math.geometry import AABB, Ray
from engine.core.math.vec import Vec3
from engine.rendering.lighting.probe_parallax import (
    ParallaxConstants,
    BoxFace,
    ProbeBox,
    RayBoxIntersection,
    ParallaxConfig,
    ParallaxCorrector,
    ParallaxProbeAdapter,
    compute_box_projection_direction,
    blend_directions,
)


# -----------------------------------------------------------------------------
# Test Utilities
# -----------------------------------------------------------------------------

def vec_approx(v1: Vec3, v2: Vec3, tol: float = 1e-5) -> bool:
    """Check if two vectors are approximately equal."""
    return (
        abs(v1.x - v2.x) < tol
        and abs(v1.y - v2.y) < tol
        and abs(v1.z - v2.z) < tol
    )


def direction_approx(v1: Vec3, v2: Vec3, tol: float = 1e-4) -> bool:
    """Check if two direction vectors are approximately equal (normalized)."""
    n1 = v1.normalized()
    n2 = v2.normalized()
    # Check if they point in the same direction
    dot = n1.dot(n2)
    return abs(dot - 1.0) < tol


# -----------------------------------------------------------------------------
# ProbeBox Tests
# -----------------------------------------------------------------------------

class TestProbeBox:
    """Tests for ProbeBox class."""

    def test_probe_box_creation(self) -> None:
        """Test creating a ProbeBox with center and extents."""
        box = ProbeBox(
            center=Vec3(0, 0, 0),
            extents=Vec3(1, 2, 3),
        )
        assert vec_approx(box.center, Vec3(0, 0, 0))
        assert vec_approx(box.extents, Vec3(1, 2, 3))

    def test_probe_box_min_max(self) -> None:
        """Test min/max point computation."""
        box = ProbeBox(
            center=Vec3(5, 10, 15),
            extents=Vec3(1, 2, 3),
        )
        assert vec_approx(box.min_point, Vec3(4, 8, 12))
        assert vec_approx(box.max_point, Vec3(6, 12, 18))

    def test_probe_box_size(self) -> None:
        """Test box size property."""
        box = ProbeBox(
            center=Vec3(0, 0, 0),
            extents=Vec3(2, 3, 4),
        )
        assert vec_approx(box.size, Vec3(4, 6, 8))

    def test_probe_box_volume(self) -> None:
        """Test box volume calculation."""
        box = ProbeBox(
            center=Vec3(0, 0, 0),
            extents=Vec3(1, 2, 3),
        )
        # Volume = 2*1 * 2*2 * 2*3 = 2 * 4 * 6 = 48
        assert box.volume == pytest.approx(48.0)

    def test_probe_box_contains_center(self) -> None:
        """Test that box contains its own center."""
        box = ProbeBox(
            center=Vec3(5, 5, 5),
            extents=Vec3(2, 2, 2),
        )
        assert box.contains(Vec3(5, 5, 5))

    def test_probe_box_contains_interior(self) -> None:
        """Test box contains interior points."""
        box = ProbeBox(
            center=Vec3(0, 0, 0),
            extents=Vec3(5, 5, 5),
        )
        assert box.contains(Vec3(1, 1, 1))
        assert box.contains(Vec3(-2, 3, -4))
        assert box.contains(Vec3(4.9, 4.9, 4.9))

    def test_probe_box_contains_boundary(self) -> None:
        """Test box contains boundary points."""
        box = ProbeBox(
            center=Vec3(0, 0, 0),
            extents=Vec3(1, 1, 1),
        )
        assert box.contains(Vec3(1, 0, 0))
        assert box.contains(Vec3(-1, -1, -1))
        assert box.contains(Vec3(0, 1, 0))

    def test_probe_box_not_contains_outside(self) -> None:
        """Test box does not contain exterior points."""
        box = ProbeBox(
            center=Vec3(0, 0, 0),
            extents=Vec3(1, 1, 1),
        )
        assert not box.contains(Vec3(2, 0, 0))
        assert not box.contains(Vec3(0, -2, 0))
        assert not box.contains(Vec3(1.1, 1.1, 1.1))

    def test_probe_box_contains_strict(self) -> None:
        """Test strict containment (not on boundary)."""
        box = ProbeBox(
            center=Vec3(0, 0, 0),
            extents=Vec3(1, 1, 1),
        )
        assert box.contains_strict(Vec3(0, 0, 0))
        assert box.contains_strict(Vec3(0.5, 0.5, 0.5))
        assert not box.contains_strict(Vec3(1, 0, 0))
        assert not box.contains_strict(Vec3(-1, -1, -1))

    def test_probe_box_get_corners(self) -> None:
        """Test getting all 8 corners."""
        box = ProbeBox(
            center=Vec3(0, 0, 0),
            extents=Vec3(1, 1, 1),
        )
        corners = box.get_corners()
        assert len(corners) == 8

        # Check corners include min and max
        assert any(vec_approx(c, Vec3(-1, -1, -1)) for c in corners)
        assert any(vec_approx(c, Vec3(1, 1, 1)) for c in corners)

    def test_probe_box_transform_to_local(self) -> None:
        """Test world to local space transform."""
        box = ProbeBox(
            center=Vec3(10, 20, 30),
            extents=Vec3(5, 5, 5),
        )
        local = box.transform_to_local(Vec3(15, 25, 35))
        assert vec_approx(local, Vec3(5, 5, 5))

    def test_probe_box_transform_to_world(self) -> None:
        """Test local to world space transform."""
        box = ProbeBox(
            center=Vec3(10, 20, 30),
            extents=Vec3(5, 5, 5),
        )
        world = box.transform_to_world(Vec3(5, 5, 5))
        assert vec_approx(world, Vec3(15, 25, 35))

    def test_probe_box_transform_round_trip(self) -> None:
        """Test transform round-trip."""
        box = ProbeBox(
            center=Vec3(7, 8, 9),
            extents=Vec3(3, 4, 5),
        )
        point = Vec3(5, 6, 7)
        local = box.transform_to_local(point)
        world = box.transform_to_world(local)
        assert vec_approx(world, point)

    def test_probe_box_closest_point_inside(self) -> None:
        """Test closest point for interior point."""
        box = ProbeBox(
            center=Vec3(0, 0, 0),
            extents=Vec3(5, 5, 5),
        )
        point = Vec3(2, 2, 2)
        closest = box.closest_point(point)
        # Inside: closest point is the point itself
        assert vec_approx(closest, point)

    def test_probe_box_closest_point_outside(self) -> None:
        """Test closest point for exterior point."""
        box = ProbeBox(
            center=Vec3(0, 0, 0),
            extents=Vec3(1, 1, 1),
        )
        point = Vec3(3, 0, 0)
        closest = box.closest_point(point)
        assert vec_approx(closest, Vec3(1, 0, 0))

    def test_probe_box_distance_inside(self) -> None:
        """Test distance is 0 for interior point."""
        box = ProbeBox(
            center=Vec3(0, 0, 0),
            extents=Vec3(5, 5, 5),
        )
        assert box.distance_to_point(Vec3(2, 2, 2)) == pytest.approx(0.0)

    def test_probe_box_distance_outside(self) -> None:
        """Test distance for exterior point."""
        box = ProbeBox(
            center=Vec3(0, 0, 0),
            extents=Vec3(1, 1, 1),
        )
        # Point (3, 0, 0) is 2 units from face at x=1
        assert box.distance_to_point(Vec3(3, 0, 0)) == pytest.approx(2.0)

    def test_probe_box_signed_distance_inside(self) -> None:
        """Test signed distance is negative inside."""
        box = ProbeBox(
            center=Vec3(0, 0, 0),
            extents=Vec3(2, 2, 2),
        )
        # At center, distance to nearest face is -2
        assert box.signed_distance(Vec3(0, 0, 0)) < 0

    def test_probe_box_signed_distance_outside(self) -> None:
        """Test signed distance is positive outside."""
        box = ProbeBox(
            center=Vec3(0, 0, 0),
            extents=Vec3(1, 1, 1),
        )
        assert box.signed_distance(Vec3(3, 0, 0)) > 0

    def test_probe_box_from_aabb(self) -> None:
        """Test creating ProbeBox from AABB."""
        aabb = AABB(Vec3(-1, -2, -3), Vec3(1, 2, 3))
        box = ProbeBox.from_aabb(aabb)
        assert vec_approx(box.center, Vec3(0, 0, 0))
        assert vec_approx(box.extents, Vec3(1, 2, 3))

    def test_probe_box_to_aabb(self) -> None:
        """Test converting ProbeBox to AABB."""
        box = ProbeBox(
            center=Vec3(5, 5, 5),
            extents=Vec3(2, 3, 4),
        )
        aabb = box.to_aabb()
        assert vec_approx(aabb.min, Vec3(3, 2, 1))
        assert vec_approx(aabb.max, Vec3(7, 8, 9))

    def test_probe_box_from_min_max(self) -> None:
        """Test creating ProbeBox from min/max corners."""
        box = ProbeBox.from_min_max(Vec3(-2, -2, -2), Vec3(4, 4, 4))
        assert vec_approx(box.center, Vec3(1, 1, 1))
        assert vec_approx(box.extents, Vec3(3, 3, 3))

    def test_probe_box_expand(self) -> None:
        """Test expanding a box."""
        box = ProbeBox(
            center=Vec3(0, 0, 0),
            extents=Vec3(1, 1, 1),
        )
        expanded = box.expand(0.5)
        assert vec_approx(expanded.extents, Vec3(1.5, 1.5, 1.5))
        assert vec_approx(expanded.center, Vec3(0, 0, 0))


# -----------------------------------------------------------------------------
# RayBoxIntersection Tests
# -----------------------------------------------------------------------------

class TestRayBoxIntersection:
    """Tests for ray-box intersection."""

    def test_ray_box_intersection_hit_front(self) -> None:
        """Test ray hitting box from front."""
        box = ProbeBox(
            center=Vec3(0, 0, 0),
            extents=Vec3(1, 1, 1),
        )
        intersection = RayBoxIntersection(
            ray_origin=Vec3(-5, 0, 0),
            ray_direction=Vec3(1, 0, 0),
            box=box,
        )
        assert intersection.intersect()
        assert intersection.get_t_entry() == pytest.approx(4.0)
        assert intersection.get_t_exit() == pytest.approx(6.0)

    def test_ray_box_intersection_hit_back(self) -> None:
        """Test ray hitting box from back."""
        box = ProbeBox(
            center=Vec3(0, 0, 0),
            extents=Vec3(1, 1, 1),
        )
        intersection = RayBoxIntersection(
            ray_origin=Vec3(5, 0, 0),
            ray_direction=Vec3(-1, 0, 0),
            box=box,
        )
        assert intersection.intersect()
        assert intersection.get_t_entry() == pytest.approx(4.0)

    def test_ray_box_intersection_miss_x(self) -> None:
        """Test ray missing box (parallel to X axis)."""
        box = ProbeBox(
            center=Vec3(0, 0, 0),
            extents=Vec3(1, 1, 1),
        )
        intersection = RayBoxIntersection(
            ray_origin=Vec3(-5, 3, 0),
            ray_direction=Vec3(1, 0, 0),
            box=box,
        )
        # Ray passes above the box
        assert not intersection.intersect()

    def test_ray_box_intersection_miss_y(self) -> None:
        """Test ray missing box (parallel to Y axis)."""
        box = ProbeBox(
            center=Vec3(0, 0, 0),
            extents=Vec3(1, 1, 1),
        )
        intersection = RayBoxIntersection(
            ray_origin=Vec3(3, -5, 0),
            ray_direction=Vec3(0, 1, 0),
            box=box,
        )
        assert not intersection.intersect()

    def test_ray_box_intersection_start_inside(self) -> None:
        """Test ray starting inside box."""
        box = ProbeBox(
            center=Vec3(0, 0, 0),
            extents=Vec3(2, 2, 2),
        )
        intersection = RayBoxIntersection(
            ray_origin=Vec3(0, 0, 0),
            ray_direction=Vec3(1, 0, 0),
            box=box,
        )
        assert intersection.intersect()
        assert intersection.is_ray_inside_box()
        # Entry is behind (negative)
        assert intersection.get_t_entry() < 0
        # Exit is in front
        assert intersection.get_t_exit() == pytest.approx(2.0)

    def test_ray_box_intersection_diagonal(self) -> None:
        """Test ray at diagonal angle."""
        box = ProbeBox(
            center=Vec3(5, 5, 5),
            extents=Vec3(1, 1, 1),
        )
        intersection = RayBoxIntersection(
            ray_origin=Vec3(0, 0, 0),
            ray_direction=Vec3(1, 1, 1).normalized(),
            box=box,
        )
        assert intersection.intersect()

    def test_ray_box_intersection_parallel_miss(self) -> None:
        """Test ray parallel to face and missing."""
        box = ProbeBox(
            center=Vec3(0, 0, 0),
            extents=Vec3(1, 1, 1),
        )
        intersection = RayBoxIntersection(
            ray_origin=Vec3(0, 5, 0),  # Above box
            ray_direction=Vec3(1, 0, 0),  # Parallel to X
            box=box,
        )
        assert not intersection.intersect()

    def test_ray_box_intersection_parallel_hit(self) -> None:
        """Test ray parallel to face and hitting."""
        box = ProbeBox(
            center=Vec3(0, 0, 0),
            extents=Vec3(2, 2, 2),
        )
        intersection = RayBoxIntersection(
            ray_origin=Vec3(-5, 0.5, 0.5),  # Inside Y-Z range
            ray_direction=Vec3(1, 0, 0),  # Parallel to X
            box=box,
        )
        assert intersection.intersect()

    def test_ray_box_intersection_get_point(self) -> None:
        """Test getting intersection point."""
        box = ProbeBox(
            center=Vec3(0, 0, 0),
            extents=Vec3(1, 1, 1),
        )
        intersection = RayBoxIntersection(
            ray_origin=Vec3(-5, 0, 0),
            ray_direction=Vec3(1, 0, 0),
            box=box,
        )
        point = intersection.get_intersection_point()
        assert point is not None
        assert vec_approx(point, Vec3(-1, 0, 0))

    def test_ray_box_intersection_exit_point(self) -> None:
        """Test getting exit point."""
        box = ProbeBox(
            center=Vec3(0, 0, 0),
            extents=Vec3(1, 1, 1),
        )
        intersection = RayBoxIntersection(
            ray_origin=Vec3(-5, 0, 0),
            ray_direction=Vec3(1, 0, 0),
            box=box,
        )
        point = intersection.get_exit_point()
        assert point is not None
        assert vec_approx(point, Vec3(1, 0, 0))

    def test_ray_box_intersection_normal(self) -> None:
        """Test getting intersection normal."""
        box = ProbeBox(
            center=Vec3(0, 0, 0),
            extents=Vec3(1, 1, 1),
        )
        intersection = RayBoxIntersection(
            ray_origin=Vec3(-5, 0, 0),
            ray_direction=Vec3(1, 0, 0),
            box=box,
        )
        normal = intersection.get_intersection_normal()
        assert normal is not None
        assert vec_approx(normal, Vec3(-1, 0, 0))

    def test_ray_box_intersection_exit_normal(self) -> None:
        """Test getting exit normal."""
        box = ProbeBox(
            center=Vec3(0, 0, 0),
            extents=Vec3(1, 1, 1),
        )
        intersection = RayBoxIntersection(
            ray_origin=Vec3(-5, 0, 0),
            ray_direction=Vec3(1, 0, 0),
            box=box,
        )
        normal = intersection.get_exit_normal()
        assert normal is not None
        assert vec_approx(normal, Vec3(1, 0, 0))

    def test_ray_box_intersection_forward_only(self) -> None:
        """Test getting forward intersection when starting inside."""
        box = ProbeBox(
            center=Vec3(0, 0, 0),
            extents=Vec3(2, 2, 2),
        )
        intersection = RayBoxIntersection(
            ray_origin=Vec3(0, 0, 0),
            ray_direction=Vec3(1, 0, 0),
            box=box,
        )
        point = intersection.get_forward_intersection_point()
        assert point is not None
        assert vec_approx(point, Vec3(2, 0, 0))

    def test_ray_box_intersection_corner_hit(self) -> None:
        """Test ray hitting corner of box."""
        box = ProbeBox(
            center=Vec3(0, 0, 0),
            extents=Vec3(1, 1, 1),
        )
        # Ray from (-2,-2,-2) toward corner (1,1,1)
        intersection = RayBoxIntersection(
            ray_origin=Vec3(-2, -2, -2),
            ray_direction=Vec3(1, 1, 1).normalized(),
            box=box,
        )
        assert intersection.intersect()

    def test_ray_box_intersection_edge_graze(self) -> None:
        """Test ray grazing edge of box."""
        box = ProbeBox(
            center=Vec3(0, 0, 0),
            extents=Vec3(1, 1, 1),
        )
        # Ray just barely touching edge
        intersection = RayBoxIntersection(
            ray_origin=Vec3(-2, 1, 0),
            ray_direction=Vec3(1, 0, 0),
            box=box,
        )
        assert intersection.intersect()

    def test_ray_box_static_helper(self) -> None:
        """Test static intersection helper."""
        box = ProbeBox(
            center=Vec3(0, 0, 0),
            extents=Vec3(1, 1, 1),
        )
        hit, t_entry, t_exit = RayBoxIntersection.intersect_ray_box(
            Vec3(-5, 0, 0),
            Vec3(1, 0, 0),
            box,
        )
        assert hit
        assert t_entry == pytest.approx(4.0)
        assert t_exit == pytest.approx(6.0)

    def test_ray_box_backward_ray_no_hit(self) -> None:
        """Test ray pointing away from box."""
        box = ProbeBox(
            center=Vec3(0, 0, 0),
            extents=Vec3(1, 1, 1),
        )
        intersection = RayBoxIntersection(
            ray_origin=Vec3(-5, 0, 0),
            ray_direction=Vec3(-1, 0, 0),  # Pointing away
            box=box,
        )
        # Ray misses (pointing wrong direction)
        point = intersection.get_forward_intersection_point()
        assert point is None


# -----------------------------------------------------------------------------
# ParallaxConfig Tests
# -----------------------------------------------------------------------------

class TestParallaxConfig:
    """Tests for ParallaxConfig."""

    def test_config_defaults(self) -> None:
        """Test default configuration values."""
        config = ParallaxConfig()
        assert config.use_box_projection is True
        assert config.inner_radius == pytest.approx(0.0)
        assert config.blend_distance > 0

    def test_config_validation(self) -> None:
        """Test configuration validation."""
        config = ParallaxConfig(
            blend_distance=-1.0,  # Invalid, should be clamped
            inner_radius=-5.0,  # Invalid, should be clamped
            outer_radius=-10.0,  # Invalid, should be at least inner_radius
        )
        assert config.blend_distance >= ParallaxConstants.MIN_BLEND_DISTANCE
        assert config.inner_radius >= 0.0
        assert config.outer_radius >= config.inner_radius

    def test_config_blend_factor_inner(self) -> None:
        """Test blend factor at inner radius."""
        config = ParallaxConfig(
            inner_radius=1.0,
            outer_radius=5.0,
        )
        assert config.get_blend_factor(0.5) == pytest.approx(0.0)
        assert config.get_blend_factor(1.0) == pytest.approx(0.0)

    def test_config_blend_factor_outer(self) -> None:
        """Test blend factor at outer radius."""
        config = ParallaxConfig(
            inner_radius=1.0,
            outer_radius=5.0,
        )
        assert config.get_blend_factor(5.0) == pytest.approx(1.0)
        assert config.get_blend_factor(10.0) == pytest.approx(1.0)

    def test_config_blend_factor_middle(self) -> None:
        """Test blend factor between radii."""
        config = ParallaxConfig(
            inner_radius=0.0,
            outer_radius=10.0,
            use_smooth_blending=False,  # Linear blending
        )
        assert config.get_blend_factor(5.0) == pytest.approx(0.5)

    def test_config_smooth_blending(self) -> None:
        """Test smooth (hermite) blending."""
        config = ParallaxConfig(
            inner_radius=0.0,
            outer_radius=10.0,
            use_smooth_blending=True,
        )
        # Smoothstep at 0.5 = 0.5^2 * (3 - 2*0.5) = 0.25 * 2 = 0.5
        # But not exactly 0.5 due to smoothstep curve
        blend = config.get_blend_factor(5.0)
        assert 0.4 < blend < 0.6


# -----------------------------------------------------------------------------
# ParallaxCorrector Tests
# -----------------------------------------------------------------------------

class TestParallaxCorrector:
    """Tests for parallax correction algorithm."""

    def test_corrector_creation(self) -> None:
        """Test creating a ParallaxCorrector."""
        corrector = ParallaxCorrector(
            probe_position=Vec3(0, 0, 0),
            probe_box=ProbeBox(Vec3(0, 0, 0), Vec3(5, 5, 5)),
        )
        assert vec_approx(corrector.probe_position, Vec3(0, 0, 0))

    def test_corrector_at_probe_position(self) -> None:
        """Test correction at probe position (no change)."""
        corrector = ParallaxCorrector(
            probe_position=Vec3(0, 0, 0),
            probe_box=ProbeBox(Vec3(0, 0, 0), Vec3(5, 5, 5)),
        )
        direction = Vec3(1, 0, 0)
        corrected = corrector.correct_direction(Vec3(0, 0, 0), direction)
        # At probe position, correction still happens but direction should be similar
        assert direction_approx(corrected, direction, 0.1)

    def test_corrector_offset_position(self) -> None:
        """Test correction at offset position."""
        # Probe at origin, box extends to +/- 5
        corrector = ParallaxCorrector(
            probe_position=Vec3(0, 0, 0),
            probe_box=ProbeBox(Vec3(0, 0, 0), Vec3(5, 5, 5)),
        )
        # Shading point offset in X, reflecting in +X direction
        world_pos = Vec3(3, 0, 0)
        reflection_dir = Vec3(1, 0, 0)

        corrected = corrector.correct_direction(world_pos, reflection_dir)

        # The ray from (3,0,0) in +X direction hits box at (5,0,0)
        # Corrected direction from probe (0,0,0) to (5,0,0) is (1,0,0)
        assert direction_approx(corrected, Vec3(1, 0, 0))

    def test_corrector_diagonal_reflection(self) -> None:
        """Test correction with diagonal reflection."""
        corrector = ParallaxCorrector(
            probe_position=Vec3(0, 0, 0),
            probe_box=ProbeBox(Vec3(0, 0, 0), Vec3(5, 5, 5)),
        )
        world_pos = Vec3(2, 2, 0)
        reflection_dir = Vec3(1, 1, 0).normalized()

        corrected = corrector.correct_direction(world_pos, reflection_dir)

        # Intersection should be at corner region
        assert corrected.length() == pytest.approx(1.0)  # Normalized

    def test_corrector_disabled(self) -> None:
        """Test that disabled correction returns original direction."""
        config = ParallaxConfig(use_box_projection=False)
        corrector = ParallaxCorrector(
            probe_position=Vec3(0, 0, 0),
            probe_box=ProbeBox(Vec3(0, 0, 0), Vec3(5, 5, 5)),
            config=config,
        )
        direction = Vec3(1, 2, 3).normalized()
        corrected = corrector.correct_direction(Vec3(3, 0, 0), direction)
        assert direction_approx(corrected, direction)

    def test_corrector_apply_box_projection(self) -> None:
        """Test apply_box_projection returns direction and blend."""
        config = ParallaxConfig(
            inner_radius=0.0,
            outer_radius=10.0,
        )
        corrector = ParallaxCorrector(
            probe_position=Vec3(0, 0, 0),
            probe_box=ProbeBox(Vec3(0, 0, 0), Vec3(5, 5, 5)),
            config=config,
        )
        direction, blend = corrector.apply_box_projection(
            Vec3(5, 0, 0),
            Vec3(1, 0, 0),
        )
        assert direction.length() == pytest.approx(1.0)
        assert 0.0 <= blend <= 1.0

    def test_corrector_apply_box_projection_inner_zone(self) -> None:
        """Test no correction in inner radius zone."""
        config = ParallaxConfig(
            inner_radius=2.0,
            outer_radius=10.0,
        )
        corrector = ParallaxCorrector(
            probe_position=Vec3(0, 0, 0),
            probe_box=ProbeBox(Vec3(0, 0, 0), Vec3(5, 5, 5)),
            config=config,
        )
        # Position within inner radius
        direction, blend = corrector.apply_box_projection(
            Vec3(1, 0, 0),
            Vec3(0, 1, 0),
        )
        assert blend == pytest.approx(0.0)

    def test_corrector_intersection_distance(self) -> None:
        """Test getting intersection distance."""
        corrector = ParallaxCorrector(
            probe_position=Vec3(0, 0, 0),
            probe_box=ProbeBox(Vec3(0, 0, 0), Vec3(5, 5, 5)),
        )
        distance = corrector.get_intersection_distance(
            Vec3(0, 0, 0),
            Vec3(1, 0, 0),
        )
        assert distance == pytest.approx(5.0)

    def test_corrector_setters(self) -> None:
        """Test property setters."""
        corrector = ParallaxCorrector(
            probe_position=Vec3(0, 0, 0),
            probe_box=ProbeBox(Vec3(0, 0, 0), Vec3(1, 1, 1)),
        )
        new_box = ProbeBox(Vec3(0, 0, 0), Vec3(10, 10, 10))
        corrector.probe_box = new_box
        corrector.probe_position = Vec3(5, 5, 5)

        assert vec_approx(corrector.probe_position, Vec3(5, 5, 5))
        assert corrector.probe_box.extents.x == pytest.approx(10.0)


# -----------------------------------------------------------------------------
# Box Projection Algorithm Tests
# -----------------------------------------------------------------------------

class TestBoxProjectionAlgorithm:
    """Tests for the core box projection algorithm."""

    def test_box_projection_x_axis(self) -> None:
        """Test box projection along X axis."""
        result = compute_box_projection_direction(
            world_position=Vec3(2, 0, 0),
            reflection_direction=Vec3(1, 0, 0),
            probe_position=Vec3(0, 0, 0),
            box_min=Vec3(-5, -5, -5),
            box_max=Vec3(5, 5, 5),
        )
        # Ray from (2,0,0) in +X hits box at (5,0,0)
        # Direction from (0,0,0) to (5,0,0) is (1,0,0)
        assert direction_approx(result, Vec3(1, 0, 0))

    def test_box_projection_y_axis(self) -> None:
        """Test box projection along Y axis."""
        result = compute_box_projection_direction(
            world_position=Vec3(0, 2, 0),
            reflection_direction=Vec3(0, 1, 0),
            probe_position=Vec3(0, 0, 0),
            box_min=Vec3(-5, -5, -5),
            box_max=Vec3(5, 5, 5),
        )
        assert direction_approx(result, Vec3(0, 1, 0))

    def test_box_projection_z_axis(self) -> None:
        """Test box projection along Z axis."""
        result = compute_box_projection_direction(
            world_position=Vec3(0, 0, 2),
            reflection_direction=Vec3(0, 0, 1),
            probe_position=Vec3(0, 0, 0),
            box_min=Vec3(-5, -5, -5),
            box_max=Vec3(5, 5, 5),
        )
        assert direction_approx(result, Vec3(0, 0, 1))

    def test_box_projection_diagonal(self) -> None:
        """Test box projection along diagonal."""
        result = compute_box_projection_direction(
            world_position=Vec3(0, 0, 0),
            reflection_direction=Vec3(1, 1, 1).normalized(),
            probe_position=Vec3(0, 0, 0),
            box_min=Vec3(-5, -5, -5),
            box_max=Vec3(5, 5, 5),
        )
        # At center, diagonal should remain diagonal
        expected = Vec3(1, 1, 1).normalized()
        assert direction_approx(result, expected)

    def test_box_projection_offset_probe(self) -> None:
        """Test box projection with offset probe position."""
        result = compute_box_projection_direction(
            world_position=Vec3(3, 0, 0),
            reflection_direction=Vec3(1, 0, 0),
            probe_position=Vec3(2, 0, 0),  # Probe offset
            box_min=Vec3(-5, -5, -5),
            box_max=Vec3(5, 5, 5),
        )
        # Ray hits at (5,0,0), direction from (2,0,0) to (5,0,0) is (1,0,0)
        assert direction_approx(result, Vec3(1, 0, 0))

    def test_box_projection_negative_direction(self) -> None:
        """Test box projection with negative direction."""
        result = compute_box_projection_direction(
            world_position=Vec3(2, 0, 0),
            reflection_direction=Vec3(-1, 0, 0),
            probe_position=Vec3(0, 0, 0),
            box_min=Vec3(-5, -5, -5),
            box_max=Vec3(5, 5, 5),
        )
        # Ray hits at (-5,0,0)
        assert direction_approx(result, Vec3(-1, 0, 0))


# -----------------------------------------------------------------------------
# ParallaxProbeAdapter Tests
# -----------------------------------------------------------------------------

class TestParallaxProbeAdapter:
    """Tests for ParallaxProbeAdapter integration."""

    def test_adapter_creation(self) -> None:
        """Test creating an adapter."""
        bounds = AABB(Vec3(-5, -5, -5), Vec3(5, 5, 5))
        adapter = ParallaxProbeAdapter(
            probe_position=Vec3(0, 0, 0),
            probe_bounds=bounds,
        )
        assert adapter.corrector is not None

    def test_adapter_get_corrected_direction(self) -> None:
        """Test getting corrected direction through adapter."""
        bounds = AABB(Vec3(-5, -5, -5), Vec3(5, 5, 5))
        adapter = ParallaxProbeAdapter(
            probe_position=Vec3(0, 0, 0),
            probe_bounds=bounds,
        )
        corrected = adapter.get_corrected_direction(
            Vec3(2, 0, 0),
            Vec3(1, 0, 0),
        )
        assert corrected.length() == pytest.approx(1.0)

    def test_adapter_sample_with_parallax(self) -> None:
        """Test sampling with parallax correction."""
        bounds = AABB(Vec3(-5, -5, -5), Vec3(5, 5, 5))
        adapter = ParallaxProbeAdapter(
            probe_position=Vec3(0, 0, 0),
            probe_bounds=bounds,
        )

        def mock_sample(direction: Vec3) -> Vec3:
            return Vec3(direction.x, direction.y, direction.z)

        result = adapter.sample_with_parallax(
            Vec3(2, 0, 0),
            Vec3(1, 0, 0),
            mock_sample,
        )
        # Should return sampled color
        assert isinstance(result, Vec3)

    def test_adapter_is_infinite_probe(self) -> None:
        """Test infinite probe detection."""
        config = ParallaxConfig(use_box_projection=False)
        bounds = AABB(Vec3(-5, -5, -5), Vec3(5, 5, 5))
        adapter = ParallaxProbeAdapter(
            probe_position=Vec3(0, 0, 0),
            probe_bounds=bounds,
            config=config,
        )
        assert adapter.is_infinite_probe()

    def test_adapter_set_box_from_aabb(self) -> None:
        """Test updating box from AABB."""
        bounds = AABB(Vec3(-5, -5, -5), Vec3(5, 5, 5))
        adapter = ParallaxProbeAdapter(
            probe_position=Vec3(0, 0, 0),
            probe_bounds=bounds,
        )
        new_bounds = AABB(Vec3(-10, -10, -10), Vec3(10, 10, 10))
        adapter.set_box_from_aabb(new_bounds)
        # Verify by checking corrector's box
        assert adapter.corrector.probe_box.extents.x == pytest.approx(10.0)

    def test_adapter_set_probe_position(self) -> None:
        """Test updating probe position."""
        bounds = AABB(Vec3(-5, -5, -5), Vec3(5, 5, 5))
        adapter = ParallaxProbeAdapter(
            probe_position=Vec3(0, 0, 0),
            probe_bounds=bounds,
        )
        adapter.set_probe_position(Vec3(1, 2, 3))
        assert vec_approx(adapter.corrector.probe_position, Vec3(1, 2, 3))


# -----------------------------------------------------------------------------
# Blend Directions Tests
# -----------------------------------------------------------------------------

class TestBlendDirections:
    """Tests for direction blending utility."""

    def test_blend_zero(self) -> None:
        """Test blending with factor 0."""
        original = Vec3(1, 0, 0)
        corrected = Vec3(0, 1, 0)
        result = blend_directions(original, corrected, 0.0)
        assert direction_approx(result, original)

    def test_blend_one(self) -> None:
        """Test blending with factor 1."""
        original = Vec3(1, 0, 0)
        corrected = Vec3(0, 1, 0)
        result = blend_directions(original, corrected, 1.0)
        assert direction_approx(result, corrected)

    def test_blend_half(self) -> None:
        """Test blending with factor 0.5."""
        original = Vec3(1, 0, 0)
        corrected = Vec3(0, 1, 0)
        result = blend_directions(original, corrected, 0.5)
        # Should be normalized blend of the two
        expected = Vec3(0.5, 0.5, 0).normalized()
        assert direction_approx(result, expected)

    def test_blend_same_direction(self) -> None:
        """Test blending same direction."""
        direction = Vec3(1, 2, 3).normalized()
        result = blend_directions(direction, direction, 0.5)
        assert direction_approx(result, direction)


# -----------------------------------------------------------------------------
# Edge Cases and Error Handling Tests
# -----------------------------------------------------------------------------

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_degenerate_box(self) -> None:
        """Test with very small box."""
        box = ProbeBox(
            center=Vec3(0, 0, 0),
            extents=Vec3(0.001, 0.001, 0.001),
        )
        intersection = RayBoxIntersection(
            ray_origin=Vec3(-1, 0, 0),
            ray_direction=Vec3(1, 0, 0),
            box=box,
        )
        # Should still work
        assert intersection.intersect()

    def test_zero_direction(self) -> None:
        """Test with zero-length direction (degenerate)."""
        corrector = ParallaxCorrector(
            probe_position=Vec3(0, 0, 0),
            probe_box=ProbeBox(Vec3(0, 0, 0), Vec3(5, 5, 5)),
        )
        # Zero direction should be handled gracefully
        result = corrector.correct_direction(
            Vec3(2, 0, 0),
            Vec3(0, 0, 0),
        )
        # Result should be a valid direction (possibly zero-ish)
        assert isinstance(result, Vec3)

    def test_very_large_box(self) -> None:
        """Test with very large box."""
        box = ProbeBox(
            center=Vec3(0, 0, 0),
            extents=Vec3(1000000, 1000000, 1000000),
        )
        intersection = RayBoxIntersection(
            ray_origin=Vec3(-2000000, 0, 0),
            ray_direction=Vec3(1, 0, 0),
            box=box,
        )
        assert intersection.intersect()

    def test_probe_at_box_corner(self) -> None:
        """Test with probe at box corner."""
        box = ProbeBox(
            center=Vec3(5, 5, 5),
            extents=Vec3(5, 5, 5),
        )
        corrector = ParallaxCorrector(
            probe_position=Vec3(0, 0, 0),  # At corner
            probe_box=box,
        )
        result = corrector.correct_direction(
            Vec3(5, 5, 5),
            Vec3(1, 0, 0),
        )
        assert result.length() == pytest.approx(1.0)

    def test_shading_point_outside_box(self) -> None:
        """Test with shading point outside probe box."""
        corrector = ParallaxCorrector(
            probe_position=Vec3(0, 0, 0),
            probe_box=ProbeBox(Vec3(0, 0, 0), Vec3(5, 5, 5)),
        )
        # Point way outside
        result = corrector.correct_direction(
            Vec3(100, 0, 0),
            Vec3(-1, 0, 0),  # Pointing toward box
        )
        # Should still compute valid direction
        assert result.length() == pytest.approx(1.0)

    def test_ray_exactly_on_face(self) -> None:
        """Test ray starting exactly on box face."""
        box = ProbeBox(
            center=Vec3(0, 0, 0),
            extents=Vec3(1, 1, 1),
        )
        intersection = RayBoxIntersection(
            ray_origin=Vec3(-1, 0, 0),  # On face
            ray_direction=Vec3(1, 0, 0),
            box=box,
        )
        assert intersection.intersect()


# -----------------------------------------------------------------------------
# Performance Tests
# -----------------------------------------------------------------------------

class TestPerformance:
    """Performance-related tests."""

    def test_many_intersections(self) -> None:
        """Test performance with many intersection calculations."""
        box = ProbeBox(
            center=Vec3(0, 0, 0),
            extents=Vec3(5, 5, 5),
        )

        start = time.perf_counter()
        count = 10000

        for i in range(count):
            x = (i % 100) / 10.0 - 5
            y = ((i // 100) % 100) / 10.0 - 5
            intersection = RayBoxIntersection(
                ray_origin=Vec3(x - 10, y, 0),
                ray_direction=Vec3(1, 0, 0),
                box=box,
            )
            intersection.intersect()

        elapsed = time.perf_counter() - start
        per_intersection_ms = (elapsed / count) * 1000

        # Should be well under 0.1ms per intersection
        assert per_intersection_ms < 0.1, f"Intersection too slow: {per_intersection_ms}ms"

    def test_many_corrections(self) -> None:
        """Test performance with many parallax corrections."""
        corrector = ParallaxCorrector(
            probe_position=Vec3(0, 0, 0),
            probe_box=ProbeBox(Vec3(0, 0, 0), Vec3(5, 5, 5)),
        )

        start = time.perf_counter()
        count = 10000

        for i in range(count):
            x = (i % 100) / 10.0 - 5
            y = ((i // 100) % 100) / 10.0 - 5
            corrector.correct_direction(
                Vec3(x, y, 0),
                Vec3(1, 0.1, 0.1).normalized(),
            )

        elapsed = time.perf_counter() - start
        per_correction_ms = (elapsed / count) * 1000

        # Should be well under 0.1ms per correction (budget allows <0.1ms overhead per probe)
        assert per_correction_ms < 0.1, f"Correction too slow: {per_correction_ms}ms"

    def test_cached_intersection(self) -> None:
        """Test that intersection results are cached."""
        box = ProbeBox(
            center=Vec3(0, 0, 0),
            extents=Vec3(5, 5, 5),
        )
        intersection = RayBoxIntersection(
            ray_origin=Vec3(-10, 0, 0),
            ray_direction=Vec3(1, 0, 0),
            box=box,
        )

        # First call computes
        intersection.intersect()

        # Subsequent calls should reuse cached result
        start = time.perf_counter()
        for _ in range(1000):
            intersection.get_t_entry()
            intersection.get_t_exit()
            intersection.get_intersection_point()
        elapsed = time.perf_counter() - start

        # Cached access should be very fast
        assert elapsed < 0.01  # Less than 10ms for 1000 iterations


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_probe_workflow(self) -> None:
        """Test complete parallax correction workflow."""
        # Create probe configuration
        probe_position = Vec3(0, 2, 0)
        probe_bounds = AABB(Vec3(-5, 0, -5), Vec3(5, 4, 5))

        # Create adapter
        adapter = ParallaxProbeAdapter(
            probe_position=probe_position,
            probe_bounds=probe_bounds,
        )

        # Simulate sampling at multiple positions
        test_positions = [
            Vec3(0, 2, 0),   # At probe
            Vec3(2, 1, 0),   # Floor level
            Vec3(-2, 3, -2), # Upper corner area
            Vec3(4, 2, 4),   # Near wall
        ]

        test_directions = [
            Vec3(1, 0, 0),
            Vec3(0, 1, 0),
            Vec3(0, -1, 0),
            Vec3(1, 1, 0).normalized(),
        ]

        for pos in test_positions:
            for dir in test_directions:
                corrected = adapter.get_corrected_direction(pos, dir)
                assert corrected.length() == pytest.approx(1.0)

    def test_interior_room_scenario(self) -> None:
        """Test realistic interior room scenario."""
        # Room: 10x3x8 meters
        room_center = Vec3(5, 1.5, 4)
        room_extents = Vec3(5, 1.5, 4)

        probe_position = room_center  # Probe at room center
        probe_box = ProbeBox(center=room_center, extents=room_extents)

        config = ParallaxConfig(
            use_box_projection=True,
            inner_radius=0.5,
            outer_radius=3.0,
        )

        corrector = ParallaxCorrector(
            probe_position=probe_position,
            probe_box=probe_box,
            config=config,
        )

        # Test reflection from floor looking at wall
        floor_pos = Vec3(5, 0.1, 4)
        wall_reflection = Vec3(1, 0, 0)  # Looking at +X wall

        corrected, blend = corrector.apply_box_projection(floor_pos, wall_reflection)

        # Corrected direction should point toward the intersection on +X wall
        assert corrected.x > 0  # Should still point toward +X
        assert corrected.length() == pytest.approx(1.0)

    def test_multiple_probes_scenario(self) -> None:
        """Test scenario with multiple adjacent probe boxes."""
        probes = [
            {
                "position": Vec3(-5, 0, 0),
                "box": ProbeBox(Vec3(-5, 0, 0), Vec3(5, 3, 5)),
            },
            {
                "position": Vec3(5, 0, 0),
                "box": ProbeBox(Vec3(5, 0, 0), Vec3(5, 3, 5)),
            },
        ]

        # Create correctors
        correctors = [
            ParallaxCorrector(p["position"], p["box"])
            for p in probes
        ]

        # Test point near boundary
        test_pos = Vec3(0, 1, 0)
        test_dir = Vec3(1, 0, 0)

        results = [c.correct_direction(test_pos, test_dir) for c in correctors]

        # Both should produce valid results
        for result in results:
            assert result.length() == pytest.approx(1.0)

    def test_probe_box_update_cycle(self) -> None:
        """Test updating probe box dynamically."""
        adapter = ParallaxProbeAdapter(
            probe_position=Vec3(0, 0, 0),
            probe_bounds=AABB(Vec3(-5, -5, -5), Vec3(5, 5, 5)),
        )

        # Get initial correction
        initial = adapter.get_corrected_direction(Vec3(2, 0, 0), Vec3(1, 0, 0))

        # Update box
        adapter.set_box_from_aabb(AABB(Vec3(-10, -10, -10), Vec3(10, 10, 10)))

        # Get new correction
        updated = adapter.get_corrected_direction(Vec3(2, 0, 0), Vec3(1, 0, 0))

        # Both should be valid
        assert initial.length() == pytest.approx(1.0)
        assert updated.length() == pytest.approx(1.0)
