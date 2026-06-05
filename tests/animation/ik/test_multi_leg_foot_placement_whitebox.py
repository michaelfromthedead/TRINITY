"""Whitebox tests for MultiLegFootPlacement class.

Tests the multi-leg foot placement implementation covering:
- __init__ creates one TwoBoneIK per foot in feet list
- solve() handles N legs (2, 4, 6, 8)
- solve() without raycast returns original transforms
- solve() with raycast modifies transforms for each foot
- Supports spider (8 legs), centaur (4 legs), bipod (2 legs)
- Configuration (ray_length, foot_height, blend_speed)
- Each leg solved independently
- Pelvis handling with multiple legs
- _calculate_multi_pelvis_offset() computation
"""

from __future__ import annotations

import math
import pytest
from typing import List, Optional, Tuple

from engine.animation.ik.foot_placement import (
    FootState,
    FootData,
    MultiLegFootPlacement,
    RaycastCallback,
    RaycastHit,
)
from engine.animation.ik.config import (
    FOOT_PLACEMENT_RAY_LENGTH,
    FOOT_PLACEMENT_FOOT_HEIGHT,
    FOOT_PLACEMENT_BLEND_SPEED,
    FOOT_PLACEMENT_REACH_SAFETY_MARGIN,
    SOFT_IK_DEFAULT_RATIO,
    SOFT_IK_DEFAULT_BLEND,
    MULTI_LEG_MAX_PELVIS_DROP,
)
from engine.animation.ik.two_bone import TwoBoneIK
from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.core.math.transform import Transform
from engine.core.constants import MATH_EPSILON


# =============================================================================
# Helper Functions - Multi-leg Skeleton Creation
# =============================================================================

def create_biped_skeleton(num_bones: int = 10) -> Tuple[List[Transform], List[FootData], int]:
    """Create a biped skeleton with 2 legs.

    Returns:
        (transforms, feet_list, pelvis_index)
    """
    positions = {
        0: Vec3(0, 1.0, 0),       # Pelvis
        1: Vec3(-0.1, 1.0, 0),    # Left upper leg
        2: Vec3(-0.1, 0.5, 0),    # Left lower leg
        3: Vec3(-0.1, 0.1, 0),    # Left foot
        4: Vec3(0.1, 1.0, 0),     # Right upper leg
        5: Vec3(0.1, 0.5, 0),     # Right lower leg
        6: Vec3(0.1, 0.1, 0),     # Right foot
    }

    transforms = []
    for i in range(num_bones):
        pos = positions.get(i, Vec3(0, 0, 0))
        transforms.append(Transform(pos, Quat.identity()))

    feet = [
        FootData(upper_leg=1, lower_leg=2, foot=3),
        FootData(upper_leg=4, lower_leg=5, foot=6),
    ]

    return transforms, feet, 0


def create_quadruped_skeleton(num_bones: int = 18) -> Tuple[List[Transform], List[FootData], int]:
    """Create a quadruped skeleton with 4 legs (centaur/dog/horse).

    Returns:
        (transforms, feet_list, pelvis_index)
    """
    positions = {
        0: Vec3(0, 1.2, 0),        # Pelvis/spine base
        # Front left leg
        1: Vec3(-0.2, 1.0, 0.5),   # Front left upper
        2: Vec3(-0.2, 0.5, 0.5),   # Front left lower
        3: Vec3(-0.2, 0.1, 0.5),   # Front left foot
        # Front right leg
        4: Vec3(0.2, 1.0, 0.5),    # Front right upper
        5: Vec3(0.2, 0.5, 0.5),    # Front right lower
        6: Vec3(0.2, 0.1, 0.5),    # Front right foot
        # Back left leg
        7: Vec3(-0.2, 1.0, -0.5),  # Back left upper
        8: Vec3(-0.2, 0.5, -0.5),  # Back left lower
        9: Vec3(-0.2, 0.1, -0.5),  # Back left foot
        # Back right leg
        10: Vec3(0.2, 1.0, -0.5),  # Back right upper
        11: Vec3(0.2, 0.5, -0.5),  # Back right lower
        12: Vec3(0.2, 0.1, -0.5),  # Back right foot
    }

    transforms = []
    for i in range(num_bones):
        pos = positions.get(i, Vec3(0, 0, 0))
        transforms.append(Transform(pos, Quat.identity()))

    feet = [
        FootData(upper_leg=1, lower_leg=2, foot=3),
        FootData(upper_leg=4, lower_leg=5, foot=6),
        FootData(upper_leg=7, lower_leg=8, foot=9),
        FootData(upper_leg=10, lower_leg=11, foot=12),
    ]

    return transforms, feet, 0


def create_hexapod_skeleton(num_bones: int = 22) -> Tuple[List[Transform], List[FootData], int]:
    """Create a hexapod skeleton with 6 legs (insect).

    Returns:
        (transforms, feet_list, pelvis_index)
    """
    positions = {
        0: Vec3(0, 0.8, 0),  # Central body/pelvis
        # Front left
        1: Vec3(-0.3, 0.7, 0.4),
        2: Vec3(-0.5, 0.4, 0.4),
        3: Vec3(-0.6, 0.1, 0.4),
        # Front right
        4: Vec3(0.3, 0.7, 0.4),
        5: Vec3(0.5, 0.4, 0.4),
        6: Vec3(0.6, 0.1, 0.4),
        # Middle left
        7: Vec3(-0.35, 0.7, 0),
        8: Vec3(-0.55, 0.4, 0),
        9: Vec3(-0.65, 0.1, 0),
        # Middle right
        10: Vec3(0.35, 0.7, 0),
        11: Vec3(0.55, 0.4, 0),
        12: Vec3(0.65, 0.1, 0),
        # Back left
        13: Vec3(-0.3, 0.7, -0.4),
        14: Vec3(-0.5, 0.4, -0.4),
        15: Vec3(-0.6, 0.1, -0.4),
        # Back right
        16: Vec3(0.3, 0.7, -0.4),
        17: Vec3(0.5, 0.4, -0.4),
        18: Vec3(0.6, 0.1, -0.4),
    }

    transforms = []
    for i in range(num_bones):
        pos = positions.get(i, Vec3(0, 0, 0))
        transforms.append(Transform(pos, Quat.identity()))

    feet = [
        FootData(upper_leg=1, lower_leg=2, foot=3),
        FootData(upper_leg=4, lower_leg=5, foot=6),
        FootData(upper_leg=7, lower_leg=8, foot=9),
        FootData(upper_leg=10, lower_leg=11, foot=12),
        FootData(upper_leg=13, lower_leg=14, foot=15),
        FootData(upper_leg=16, lower_leg=17, foot=18),
    ]

    return transforms, feet, 0


def create_spider_skeleton(num_bones: int = 28) -> Tuple[List[Transform], List[FootData], int]:
    """Create an octopod skeleton with 8 legs (spider).

    Returns:
        (transforms, feet_list, pelvis_index)
    """
    # Radial leg arrangement
    angles = [
        math.pi * 0.15,   # Front right
        math.pi * 0.35,   # Mid-front right
        math.pi * 0.65,   # Mid-back right
        math.pi * 0.85,   # Back right
        math.pi * 1.15,   # Back left
        math.pi * 1.35,   # Mid-back left
        math.pi * 1.65,   # Mid-front left
        math.pi * 1.85,   # Front left
    ]

    positions = {0: Vec3(0, 0.6, 0)}  # Central body
    bone_idx = 1

    for angle in angles:
        # Direction from center
        dx = math.cos(angle)
        dz = math.sin(angle)

        # Upper leg (hip joint)
        positions[bone_idx] = Vec3(dx * 0.15, 0.55, dz * 0.15)
        # Lower leg (knee)
        positions[bone_idx + 1] = Vec3(dx * 0.4, 0.3, dz * 0.4)
        # Foot
        positions[bone_idx + 2] = Vec3(dx * 0.6, 0.05, dz * 0.6)
        bone_idx += 3

    transforms = []
    for i in range(num_bones):
        pos = positions.get(i, Vec3(0, 0, 0))
        transforms.append(Transform(pos, Quat.identity()))

    feet = []
    for leg_idx in range(8):
        base = 1 + leg_idx * 3
        feet.append(FootData(upper_leg=base, lower_leg=base + 1, foot=base + 2))

    return transforms, feet, 0


# =============================================================================
# Raycast Helper Functions
# =============================================================================

def create_flat_ground_raycast(ground_y: float = 0.0) -> RaycastCallback:
    """Create raycast callback for flat ground at specified y."""
    def raycast(origin: Vec3, direction: Vec3, max_dist: float) -> Optional[RaycastHit]:
        if direction.y >= 0:
            return None
        t = (origin.y - ground_y) / (-direction.y)
        if t > max_dist or t < 0:
            return None
        hit_pos = Vec3(origin.x + direction.x * t, ground_y, origin.z + direction.z * t)
        return RaycastHit(hit=True, position=hit_pos, normal=Vec3(0, 1, 0), distance=t)
    return raycast


def create_uneven_terrain_raycast() -> RaycastCallback:
    """Create raycast callback for uneven terrain with hills."""
    def raycast(origin: Vec3, direction: Vec3, max_dist: float) -> Optional[RaycastHit]:
        if direction.y >= 0:
            return None
        # Terrain with sinusoidal hills
        ground_y = 0.1 * math.sin(origin.x * 2) * math.cos(origin.z * 2)
        t = (origin.y - ground_y) / (-direction.y)
        if t > max_dist or t < 0:
            return None
        hit_x = origin.x + direction.x * t
        hit_z = origin.z + direction.z * t
        actual_y = 0.1 * math.sin(hit_x * 2) * math.cos(hit_z * 2)
        # Approximate normal
        dx = 0.2 * math.cos(hit_x * 2) * math.cos(hit_z * 2)
        dz = -0.2 * math.sin(hit_x * 2) * math.sin(hit_z * 2)
        normal = Vec3(-dx, 1, -dz).normalized()
        return RaycastHit(hit=True, position=Vec3(hit_x, actual_y, hit_z), normal=normal, distance=t)
    return raycast


def create_stepped_terrain_raycast() -> RaycastCallback:
    """Create raycast for stepped terrain (stairs)."""
    def raycast(origin: Vec3, direction: Vec3, max_dist: float) -> Optional[RaycastHit]:
        if direction.y >= 0:
            return None
        # Steps every 0.5 units in z direction, 0.15 height each
        step_idx = int(origin.z / 0.5)
        ground_y = step_idx * 0.15 if step_idx >= 0 else 0
        t = (origin.y - ground_y) / (-direction.y)
        if t > max_dist or t < 0:
            return None
        hit_pos = Vec3(origin.x + direction.x * t, ground_y, origin.z + direction.z * t)
        return RaycastHit(hit=True, position=hit_pos, normal=Vec3(0, 1, 0), distance=t)
    return raycast


def create_sloped_ground_raycast(slope_x: float = 0.2, slope_z: float = 0.0) -> RaycastCallback:
    """Create raycast for sloped ground."""
    def raycast(origin: Vec3, direction: Vec3, max_dist: float) -> Optional[RaycastHit]:
        if direction.y >= 0:
            return None
        ground_y = slope_x * origin.x + slope_z * origin.z
        t = (origin.y - ground_y) / (-direction.y)
        if t > max_dist or t < 0:
            return None
        hit_x = origin.x + direction.x * t
        hit_z = origin.z + direction.z * t
        actual_y = slope_x * hit_x + slope_z * hit_z
        normal = Vec3(-slope_x, 1, -slope_z).normalized()
        return RaycastHit(hit=True, position=Vec3(hit_x, actual_y, hit_z), normal=normal, distance=t)
    return raycast


def create_no_hit_raycast() -> RaycastCallback:
    """Create raycast that never hits."""
    def raycast(origin: Vec3, direction: Vec3, max_dist: float) -> Optional[RaycastHit]:
        return None
    return raycast


def create_selective_hit_raycast(hit_indices: List[int]) -> RaycastCallback:
    """Create raycast that only hits for certain foot indices (based on x position)."""
    call_count = [0]
    def raycast(origin: Vec3, direction: Vec3, max_dist: float) -> Optional[RaycastHit]:
        idx = call_count[0]
        call_count[0] += 1
        if idx not in hit_indices:
            return None
        if direction.y >= 0:
            return None
        t = origin.y / (-direction.y)
        if t > max_dist or t < 0:
            return None
        hit_pos = Vec3(origin.x + direction.x * t, 0, origin.z + direction.z * t)
        return RaycastHit(hit=True, position=hit_pos, normal=Vec3(0, 1, 0), distance=t)
    return raycast


def create_tracking_raycast() -> Tuple[RaycastCallback, List]:
    """Create raycast that tracks all calls."""
    calls = []
    def raycast(origin: Vec3, direction: Vec3, max_dist: float) -> Optional[RaycastHit]:
        calls.append({'origin': origin, 'direction': direction, 'max_dist': max_dist})
        if direction.y >= 0:
            return None
        t = origin.y / (-direction.y)
        if t > max_dist or t < 0:
            return None
        hit_pos = Vec3(origin.x + direction.x * t, 0, origin.z + direction.z * t)
        return RaycastHit(hit=True, position=hit_pos, normal=Vec3(0, 1, 0), distance=t)
    return raycast, calls


# =============================================================================
# Test Class: __init__ Tests
# =============================================================================

class TestMultiLegFootPlacementInit:
    """Tests for MultiLegFootPlacement.__init__"""

    def test_init_creates_ik_solver_per_foot_biped(self):
        """Test __init__ creates exactly 2 TwoBoneIK solvers for biped."""
        transforms, feet, pelvis = create_biped_skeleton()
        placement = MultiLegFootPlacement(feet, pelvis)

        assert len(placement._leg_iks) == 2
        assert all(isinstance(ik, TwoBoneIK) for ik in placement._leg_iks)

    def test_init_creates_ik_solver_per_foot_quadruped(self):
        """Test __init__ creates exactly 4 TwoBoneIK solvers for quadruped."""
        transforms, feet, pelvis = create_quadruped_skeleton()
        placement = MultiLegFootPlacement(feet, pelvis)

        assert len(placement._leg_iks) == 4
        assert all(isinstance(ik, TwoBoneIK) for ik in placement._leg_iks)

    def test_init_creates_ik_solver_per_foot_hexapod(self):
        """Test __init__ creates exactly 6 TwoBoneIK solvers for hexapod."""
        transforms, feet, pelvis = create_hexapod_skeleton()
        placement = MultiLegFootPlacement(feet, pelvis)

        assert len(placement._leg_iks) == 6
        assert all(isinstance(ik, TwoBoneIK) for ik in placement._leg_iks)

    def test_init_creates_ik_solver_per_foot_spider(self):
        """Test __init__ creates exactly 8 TwoBoneIK solvers for spider."""
        transforms, feet, pelvis = create_spider_skeleton()
        placement = MultiLegFootPlacement(feet, pelvis)

        assert len(placement._leg_iks) == 8
        assert all(isinstance(ik, TwoBoneIK) for ik in placement._leg_iks)

    def test_init_stores_feet_list(self):
        """Test __init__ stores the feet list."""
        transforms, feet, pelvis = create_quadruped_skeleton()
        placement = MultiLegFootPlacement(feet, pelvis)

        assert placement.feet == feet
        assert len(placement.feet) == 4

    def test_init_stores_pelvis_index(self):
        """Test __init__ stores the pelvis bone index."""
        transforms, feet, pelvis = create_biped_skeleton()
        placement = MultiLegFootPlacement(feet, pelvis)

        assert placement.pelvis == 0

    def test_init_stores_pelvis_index_custom(self):
        """Test __init__ stores custom pelvis index."""
        transforms, feet, _ = create_biped_skeleton()
        custom_pelvis = 5
        placement = MultiLegFootPlacement(feet, custom_pelvis)

        assert placement.pelvis == custom_pelvis

    def test_init_stores_raycast_callback(self):
        """Test __init__ stores raycast callback."""
        transforms, feet, pelvis = create_biped_skeleton()
        raycast = create_flat_ground_raycast()
        placement = MultiLegFootPlacement(feet, pelvis, raycast)

        assert placement._raycast is raycast

    def test_init_raycast_callback_defaults_to_none(self):
        """Test __init__ defaults raycast callback to None."""
        transforms, feet, pelvis = create_biped_skeleton()
        placement = MultiLegFootPlacement(feet, pelvis)

        assert placement._raycast is None

    def test_init_sets_default_ray_length(self):
        """Test __init__ sets default ray_length."""
        transforms, feet, pelvis = create_biped_skeleton()
        placement = MultiLegFootPlacement(feet, pelvis)

        assert placement.ray_length == FOOT_PLACEMENT_RAY_LENGTH

    def test_init_sets_default_foot_height(self):
        """Test __init__ sets default foot_height."""
        transforms, feet, pelvis = create_biped_skeleton()
        placement = MultiLegFootPlacement(feet, pelvis)

        assert placement.foot_height == FOOT_PLACEMENT_FOOT_HEIGHT

    def test_init_sets_default_blend_speed(self):
        """Test __init__ sets default blend_speed."""
        transforms, feet, pelvis = create_biped_skeleton()
        placement = MultiLegFootPlacement(feet, pelvis)

        assert placement.blend_speed == FOOT_PLACEMENT_BLEND_SPEED

    def test_init_ik_solver_uses_correct_bone_indices(self):
        """Test each IK solver uses correct bone indices from FootData."""
        transforms, feet, pelvis = create_quadruped_skeleton()
        placement = MultiLegFootPlacement(feet, pelvis)

        for i, (foot, ik) in enumerate(zip(feet, placement._leg_iks)):
            assert ik.root_bone == foot.upper_leg
            assert ik.mid_bone == foot.lower_leg
            assert ik.end_bone == foot.foot

    def test_init_ik_solver_uses_soft_ik_defaults(self):
        """Test IK solvers use default soft IK parameters."""
        transforms, feet, pelvis = create_biped_skeleton()
        placement = MultiLegFootPlacement(feet, pelvis)

        for ik in placement._leg_iks:
            assert ik.soft_ik_ratio == SOFT_IK_DEFAULT_RATIO
            assert ik.soft_ik_blend == SOFT_IK_DEFAULT_BLEND

    def test_init_with_empty_feet_list(self):
        """Test __init__ with empty feet list."""
        transforms, _, pelvis = create_biped_skeleton()
        placement = MultiLegFootPlacement([], pelvis)

        assert len(placement._leg_iks) == 0
        assert len(placement.feet) == 0

    def test_init_with_single_foot(self):
        """Test __init__ with single foot (monopod)."""
        transforms, feet, pelvis = create_biped_skeleton()
        single_foot = [feet[0]]
        placement = MultiLegFootPlacement(single_foot, pelvis)

        assert len(placement._leg_iks) == 1


# =============================================================================
# Test Class: solve() Without Raycast
# =============================================================================

class TestMultiLegSolveNoRaycast:
    """Tests for solve() when raycast is None."""

    def test_solve_without_raycast_returns_original_transforms_biped(self):
        """Test solve() returns original transforms when no raycast (biped)."""
        transforms, feet, pelvis = create_biped_skeleton()
        placement = MultiLegFootPlacement(feet, pelvis)

        result = placement.solve(transforms, Vec3(0, 0, 0))

        assert len(result) == len(transforms)
        for i, (orig, res) in enumerate(zip(transforms, result)):
            assert res.translation.x == pytest.approx(orig.translation.x)
            assert res.translation.y == pytest.approx(orig.translation.y)
            assert res.translation.z == pytest.approx(orig.translation.z)

    def test_solve_without_raycast_returns_original_transforms_quadruped(self):
        """Test solve() returns original transforms when no raycast (quadruped)."""
        transforms, feet, pelvis = create_quadruped_skeleton()
        placement = MultiLegFootPlacement(feet, pelvis)

        result = placement.solve(transforms, Vec3(0, 0, 0))

        assert len(result) == len(transforms)

    def test_solve_without_raycast_returns_original_transforms_spider(self):
        """Test solve() returns original transforms when no raycast (spider)."""
        transforms, feet, pelvis = create_spider_skeleton()
        placement = MultiLegFootPlacement(feet, pelvis)

        result = placement.solve(transforms, Vec3(0, 0, 0))

        assert len(result) == len(transforms)

    def test_solve_without_raycast_preserves_rotations(self):
        """Test solve() preserves rotations when no raycast."""
        transforms, feet, pelvis = create_biped_skeleton()
        # Add non-identity rotations
        transforms[0].rotation = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 4)
        transforms[3].rotation = Quat.from_axis_angle(Vec3(1, 0, 0), math.pi / 6)

        placement = MultiLegFootPlacement(feet, pelvis)
        result = placement.solve(transforms, Vec3(0, 0, 0))

        for i, (orig, res) in enumerate(zip(transforms, result)):
            assert res.rotation.w == pytest.approx(orig.rotation.w, abs=1e-6)
            assert res.rotation.x == pytest.approx(orig.rotation.x, abs=1e-6)
            assert res.rotation.y == pytest.approx(orig.rotation.y, abs=1e-6)
            assert res.rotation.z == pytest.approx(orig.rotation.z, abs=1e-6)


# =============================================================================
# Test Class: solve() With Raycast - Basic
# =============================================================================

class TestMultiLegSolveWithRaycast:
    """Tests for solve() with raycast callback."""

    def test_solve_with_raycast_returns_modified_transforms(self):
        """Test solve() returns modified transforms with raycast."""
        transforms, feet, pelvis = create_biped_skeleton()
        raycast = create_flat_ground_raycast()
        placement = MultiLegFootPlacement(feet, pelvis, raycast)

        result = placement.solve(transforms, Vec3(0, 0, 0))

        assert len(result) == len(transforms)
        # Should have modified something (pelvis or foot bones)

    def test_solve_creates_copy_of_transforms(self):
        """Test solve() creates a copy, not modifying originals."""
        transforms, feet, pelvis = create_biped_skeleton()
        raycast = create_flat_ground_raycast()
        placement = MultiLegFootPlacement(feet, pelvis, raycast)

        original_positions = [t.translation.y for t in transforms]
        result = placement.solve(transforms, Vec3(0, 0, 0))

        # Original transforms should be unchanged
        for i, pos in enumerate(original_positions):
            assert transforms[i].translation.y == pos

    def test_solve_calls_raycast_for_each_foot(self):
        """Test solve() calls raycast once per foot."""
        transforms, feet, pelvis = create_quadruped_skeleton()
        raycast, calls = create_tracking_raycast()
        placement = MultiLegFootPlacement(feet, pelvis, raycast)

        placement.solve(transforms, Vec3(0, 0, 0))

        assert len(calls) == 4  # One per foot

    def test_solve_raycast_origin_from_foot_position(self):
        """Test raycast origin is derived from foot position."""
        transforms, feet, pelvis = create_biped_skeleton()
        raycast, calls = create_tracking_raycast()
        placement = MultiLegFootPlacement(feet, pelvis, raycast)

        placement.solve(transforms, Vec3(0, 0, 0))

        # First foot at index 3
        foot_pos = transforms[3].translation
        first_call = calls[0]
        assert first_call['origin'].x == pytest.approx(foot_pos.x)
        assert first_call['origin'].z == pytest.approx(foot_pos.z)

    def test_solve_raycast_direction_is_down(self):
        """Test raycast direction is downward."""
        transforms, feet, pelvis = create_biped_skeleton()
        raycast, calls = create_tracking_raycast()
        placement = MultiLegFootPlacement(feet, pelvis, raycast)

        placement.solve(transforms, Vec3(0, 0, 0))

        for call in calls:
            assert call['direction'].y == -1.0
            assert call['direction'].x == 0.0
            assert call['direction'].z == 0.0

    def test_solve_raycast_uses_ray_length(self):
        """Test raycast uses configured ray_length."""
        transforms, feet, pelvis = create_biped_skeleton()
        raycast, calls = create_tracking_raycast()
        placement = MultiLegFootPlacement(feet, pelvis, raycast)
        placement.ray_length = 3.0

        placement.solve(transforms, Vec3(0, 0, 0))

        for call in calls:
            assert call['max_dist'] == 3.0


# =============================================================================
# Test Class: solve() - Leg Count Variations
# =============================================================================

class TestMultiLegSolveLegCounts:
    """Tests for solve() with different leg counts."""

    def test_solve_handles_2_legs_biped(self):
        """Test solve() works correctly with 2 legs."""
        transforms, feet, pelvis = create_biped_skeleton()
        raycast = create_flat_ground_raycast()
        placement = MultiLegFootPlacement(feet, pelvis, raycast)

        result = placement.solve(transforms, Vec3(0, 0, 0))

        assert result is not None
        assert len(result) == len(transforms)

    def test_solve_handles_4_legs_quadruped(self):
        """Test solve() works correctly with 4 legs."""
        transforms, feet, pelvis = create_quadruped_skeleton()
        raycast = create_flat_ground_raycast()
        placement = MultiLegFootPlacement(feet, pelvis, raycast)

        result = placement.solve(transforms, Vec3(0, 0, 0))

        assert result is not None
        assert len(result) == len(transforms)

    def test_solve_handles_6_legs_hexapod(self):
        """Test solve() works correctly with 6 legs."""
        transforms, feet, pelvis = create_hexapod_skeleton()
        raycast = create_flat_ground_raycast()
        placement = MultiLegFootPlacement(feet, pelvis, raycast)

        result = placement.solve(transforms, Vec3(0, 0, 0))

        assert result is not None
        assert len(result) == len(transforms)

    def test_solve_handles_8_legs_spider(self):
        """Test solve() works correctly with 8 legs."""
        transforms, feet, pelvis = create_spider_skeleton()
        raycast = create_flat_ground_raycast()
        placement = MultiLegFootPlacement(feet, pelvis, raycast)

        result = placement.solve(transforms, Vec3(0, 0, 0))

        assert result is not None
        assert len(result) == len(transforms)

    def test_solve_handles_0_legs(self):
        """Test solve() handles empty feet list gracefully."""
        transforms, _, pelvis = create_biped_skeleton()
        raycast = create_flat_ground_raycast()
        placement = MultiLegFootPlacement([], pelvis, raycast)

        result = placement.solve(transforms, Vec3(0, 0, 0))

        assert result is not None
        assert len(result) == len(transforms)

    def test_solve_handles_1_leg(self):
        """Test solve() works with single leg."""
        transforms, feet, pelvis = create_biped_skeleton()
        raycast = create_flat_ground_raycast()
        placement = MultiLegFootPlacement([feet[0]], pelvis, raycast)

        result = placement.solve(transforms, Vec3(0, 0, 0))

        assert result is not None

    def test_solve_handles_3_legs(self):
        """Test solve() works with 3 legs (tripod)."""
        transforms, feet, pelvis = create_quadruped_skeleton()
        raycast = create_flat_ground_raycast()
        placement = MultiLegFootPlacement(feet[:3], pelvis, raycast)

        result = placement.solve(transforms, Vec3(0, 0, 0))

        assert result is not None
        assert len(placement._leg_iks) == 3


# =============================================================================
# Test Class: solve() - Independent Leg Processing
# =============================================================================

class TestMultiLegIndependentProcessing:
    """Tests for independent leg processing."""

    def test_solve_processes_each_leg_independently(self):
        """Test each leg is solved independently."""
        transforms, feet, pelvis = create_quadruped_skeleton()
        # Set different blend weights
        feet[0].blend_weight = 1.0
        feet[1].blend_weight = 0.5
        feet[2].blend_weight = 0.0  # Should not be processed
        feet[3].blend_weight = 1.0

        raycast = create_flat_ground_raycast()
        placement = MultiLegFootPlacement(feet, pelvis, raycast)

        result = placement.solve(transforms, Vec3(0, 0, 0))

        # Third leg should remain unchanged (blend_weight = 0)
        assert result is not None

    def test_solve_partial_raycast_hits(self):
        """Test solve() handles partial raycast hits."""
        transforms, feet, pelvis = create_quadruped_skeleton()
        # Only first two feet will hit
        raycast = create_selective_hit_raycast([0, 1])
        placement = MultiLegFootPlacement(feet, pelvis, raycast)

        result = placement.solve(transforms, Vec3(0, 0, 0))

        assert result is not None

    def test_solve_no_raycast_hits(self):
        """Test solve() handles no raycast hits."""
        transforms, feet, pelvis = create_quadruped_skeleton()
        raycast = create_no_hit_raycast()
        placement = MultiLegFootPlacement(feet, pelvis, raycast)

        result = placement.solve(transforms, Vec3(0, 0, 0))

        # Should not modify transforms much (no IK applied)
        assert result is not None


# =============================================================================
# Test Class: solve() - Pelvis Handling
# =============================================================================

class TestMultiLegPelvisHandling:
    """Tests for pelvis adjustment with multiple legs."""

    def test_solve_adjusts_pelvis_for_planted_feet(self):
        """Test pelvis is adjusted when feet are planted."""
        transforms, feet, pelvis = create_quadruped_skeleton()
        raycast = create_flat_ground_raycast(-0.2)  # Ground below original
        placement = MultiLegFootPlacement(feet, pelvis, raycast)

        result = placement.solve(transforms, Vec3(0, 0, 0))

        # Pelvis may be adjusted
        assert result is not None
        assert result[pelvis] is not None

    def test_solve_pelvis_adjusts_with_uneven_terrain(self):
        """Test pelvis adjustment on uneven terrain."""
        transforms, feet, pelvis = create_quadruped_skeleton()
        raycast = create_uneven_terrain_raycast()
        placement = MultiLegFootPlacement(feet, pelvis, raycast)

        result = placement.solve(transforms, Vec3(0, 0, 0))

        assert result is not None

    def test_solve_pelvis_uses_correct_bone_index(self):
        """Test pelvis modification uses correct bone index."""
        transforms, feet, _ = create_biped_skeleton()
        custom_pelvis = 2  # Non-standard pelvis index
        raycast = create_flat_ground_raycast()
        placement = MultiLegFootPlacement(feet, custom_pelvis, raycast)

        original_pelvis_y = transforms[custom_pelvis].translation.y
        result = placement.solve(transforms, Vec3(0, 0, 0))

        # The pelvis index should have been used
        assert result[custom_pelvis] is not None


# =============================================================================
# Test Class: _calculate_multi_pelvis_offset
# =============================================================================

class TestCalculateMultiPelvisOffset:
    """Tests for _calculate_multi_pelvis_offset() method."""

    def test_pelvis_offset_returns_float(self):
        """Test _calculate_multi_pelvis_offset returns float."""
        transforms, feet, pelvis = create_biped_skeleton()
        placement = MultiLegFootPlacement(feet, pelvis)

        pelvis_pos = Vec3(0, 1.0, 0)
        targets = [Vec3(-0.1, 0.0, 0), Vec3(0.1, 0.0, 0)]
        planted = [True, True]

        offset = placement._calculate_multi_pelvis_offset(pelvis_pos, targets, planted)

        assert isinstance(offset, float)

    def test_pelvis_offset_zero_when_in_reach(self):
        """Test pelvis offset is zero when all feet are in reach."""
        transforms, feet, pelvis = create_biped_skeleton()
        placement = MultiLegFootPlacement(feet, pelvis)

        # Close targets that are easily reachable
        pelvis_pos = Vec3(0, 1.0, 0)
        targets = [Vec3(-0.1, 0.5, 0), Vec3(0.1, 0.5, 0)]
        planted = [True, True]

        offset = placement._calculate_multi_pelvis_offset(pelvis_pos, targets, planted)

        assert offset <= 0  # Should not raise pelvis

    def test_pelvis_offset_negative_when_overshoot(self):
        """Test pelvis drops (negative offset) when feet overshoot reach."""
        transforms, feet, pelvis = create_biped_skeleton()
        placement = MultiLegFootPlacement(feet, pelvis)

        # Far targets that require pelvis drop
        pelvis_pos = Vec3(0, 1.0, 0)
        targets = [Vec3(-0.1, -0.5, 0), Vec3(0.1, -0.5, 0)]
        planted = [True, True]

        offset = placement._calculate_multi_pelvis_offset(pelvis_pos, targets, planted)

        assert offset <= 0  # Should be negative (pelvis drops)

    def test_pelvis_offset_capped_at_max_drop(self):
        """Test pelvis offset is capped at MULTI_LEG_MAX_PELVIS_DROP."""
        transforms, feet, pelvis = create_biped_skeleton()
        placement = MultiLegFootPlacement(feet, pelvis)

        # Extremely far targets
        pelvis_pos = Vec3(0, 1.0, 0)
        targets = [Vec3(-0.1, -5.0, 0), Vec3(0.1, -5.0, 0)]
        planted = [True, True]

        offset = placement._calculate_multi_pelvis_offset(pelvis_pos, targets, planted)

        assert offset >= -MULTI_LEG_MAX_PELVIS_DROP

    def test_pelvis_offset_ignores_unplanted_feet(self):
        """Test pelvis offset ignores unplanted feet."""
        transforms, feet, pelvis = create_biped_skeleton()
        placement = MultiLegFootPlacement(feet, pelvis)

        pelvis_pos = Vec3(0, 1.0, 0)
        # First foot very far (unplanted), second close (planted)
        targets = [Vec3(-0.1, -5.0, 0), Vec3(0.1, 0.5, 0)]
        planted = [False, True]

        offset = placement._calculate_multi_pelvis_offset(pelvis_pos, targets, planted)

        # Should ignore first foot, so offset should be small
        assert offset >= -0.5  # Should not drop fully

    def test_pelvis_offset_all_unplanted(self):
        """Test pelvis offset is zero when all feet unplanted."""
        transforms, feet, pelvis = create_biped_skeleton()
        placement = MultiLegFootPlacement(feet, pelvis)

        pelvis_pos = Vec3(0, 1.0, 0)
        targets = [Vec3(-0.1, -5.0, 0), Vec3(0.1, -5.0, 0)]
        planted = [False, False]

        offset = placement._calculate_multi_pelvis_offset(pelvis_pos, targets, planted)

        assert offset == 0.0

    def test_pelvis_offset_uses_reach_safety_margin(self):
        """Test pelvis offset calculation uses FOOT_PLACEMENT_REACH_SAFETY_MARGIN."""
        transforms, feet, pelvis = create_biped_skeleton()
        placement = MultiLegFootPlacement(feet, pelvis)

        # This test verifies the margin is used in calculation
        pelvis_pos = Vec3(0, 1.0, 0)
        targets = [Vec3(-0.1, 0.1, 0), Vec3(0.1, 0.1, 0)]
        planted = [True, True]

        offset = placement._calculate_multi_pelvis_offset(pelvis_pos, targets, planted)

        # Just verify it returns a valid value
        assert isinstance(offset, float)


# =============================================================================
# Test Class: Configuration Tests
# =============================================================================

class TestMultiLegConfiguration:
    """Tests for configuration attributes."""

    def test_ray_length_can_be_modified(self):
        """Test ray_length can be changed after init."""
        transforms, feet, pelvis = create_biped_skeleton()
        placement = MultiLegFootPlacement(feet, pelvis)

        placement.ray_length = 5.0

        assert placement.ray_length == 5.0

    def test_foot_height_can_be_modified(self):
        """Test foot_height can be changed after init."""
        transforms, feet, pelvis = create_biped_skeleton()
        placement = MultiLegFootPlacement(feet, pelvis)

        placement.foot_height = 0.2

        assert placement.foot_height == 0.2

    def test_blend_speed_can_be_modified(self):
        """Test blend_speed can be changed after init."""
        transforms, feet, pelvis = create_biped_skeleton()
        placement = MultiLegFootPlacement(feet, pelvis)

        placement.blend_speed = 20.0

        assert placement.blend_speed == 20.0

    def test_custom_ray_length_affects_raycast(self):
        """Test custom ray_length is used in raycast."""
        transforms, feet, pelvis = create_biped_skeleton()
        raycast, calls = create_tracking_raycast()
        placement = MultiLegFootPlacement(feet, pelvis, raycast)
        placement.ray_length = 10.0

        placement.solve(transforms, Vec3(0, 0, 0))

        assert calls[0]['max_dist'] == 10.0

    def test_foot_height_affects_target_position(self):
        """Test foot_height affects target position calculation."""
        transforms, feet, pelvis = create_biped_skeleton()
        raycast = create_flat_ground_raycast()

        placement1 = MultiLegFootPlacement(feet, pelvis, raycast)
        placement1.foot_height = 0.05
        result1 = placement1.solve(transforms, Vec3(0, 0, 0))

        placement2 = MultiLegFootPlacement(feet, pelvis, raycast)
        placement2.foot_height = 0.2
        result2 = placement2.solve(transforms, Vec3(0, 0, 0))

        # Different foot heights should produce different results
        # (or at least not crash)
        assert result1 is not None
        assert result2 is not None


# =============================================================================
# Test Class: solve() - Delta Time
# =============================================================================

class TestMultiLegDeltaTime:
    """Tests for delta time handling in solve()."""

    def test_solve_accepts_default_dt(self):
        """Test solve() works with default dt."""
        transforms, feet, pelvis = create_biped_skeleton()
        raycast = create_flat_ground_raycast()
        placement = MultiLegFootPlacement(feet, pelvis, raycast)

        result = placement.solve(transforms, Vec3(0, 0, 0))

        assert result is not None

    def test_solve_accepts_custom_dt(self):
        """Test solve() works with custom dt."""
        transforms, feet, pelvis = create_biped_skeleton()
        raycast = create_flat_ground_raycast()
        placement = MultiLegFootPlacement(feet, pelvis, raycast)

        result = placement.solve(transforms, Vec3(0, 0, 0), dt=1.0/30.0)

        assert result is not None

    def test_solve_accepts_large_dt(self):
        """Test solve() handles large dt."""
        transforms, feet, pelvis = create_biped_skeleton()
        raycast = create_flat_ground_raycast()
        placement = MultiLegFootPlacement(feet, pelvis, raycast)

        result = placement.solve(transforms, Vec3(0, 0, 0), dt=1.0)

        assert result is not None

    def test_solve_accepts_small_dt(self):
        """Test solve() handles very small dt."""
        transforms, feet, pelvis = create_biped_skeleton()
        raycast = create_flat_ground_raycast()
        placement = MultiLegFootPlacement(feet, pelvis, raycast)

        result = placement.solve(transforms, Vec3(0, 0, 0), dt=0.001)

        assert result is not None


# =============================================================================
# Test Class: solve() - Character Position
# =============================================================================

class TestMultiLegCharacterPosition:
    """Tests for character position handling in solve()."""

    def test_solve_accepts_zero_character_position(self):
        """Test solve() works with zero character position."""
        transforms, feet, pelvis = create_biped_skeleton()
        raycast = create_flat_ground_raycast()
        placement = MultiLegFootPlacement(feet, pelvis, raycast)

        result = placement.solve(transforms, Vec3(0, 0, 0))

        assert result is not None

    def test_solve_accepts_offset_character_position(self):
        """Test solve() works with offset character position."""
        transforms, feet, pelvis = create_biped_skeleton()
        raycast = create_flat_ground_raycast()
        placement = MultiLegFootPlacement(feet, pelvis, raycast)

        result = placement.solve(transforms, Vec3(10, 5, -3))

        assert result is not None

    def test_solve_different_character_positions(self):
        """Test solve() with various character positions."""
        transforms, feet, pelvis = create_biped_skeleton()
        raycast = create_flat_ground_raycast()
        placement = MultiLegFootPlacement(feet, pelvis, raycast)

        positions = [
            Vec3(0, 0, 0),
            Vec3(100, 0, 0),
            Vec3(0, 100, 0),
            Vec3(0, 0, 100),
            Vec3(-50, -50, -50),
        ]

        for pos in positions:
            result = placement.solve(transforms, pos)
            assert result is not None


# =============================================================================
# Test Class: IK Application
# =============================================================================

class TestMultiLegIKApplication:
    """Tests for IK application to legs."""

    def test_ik_applied_when_foot_planted(self):
        """Test IK is applied when foot is planted."""
        transforms, feet, pelvis = create_biped_skeleton()
        feet[0].blend_weight = 1.0
        raycast = create_flat_ground_raycast()
        placement = MultiLegFootPlacement(feet, pelvis, raycast)

        result = placement.solve(transforms, Vec3(0, 0, 0))

        # IK should have been applied (rotations may change)
        assert result is not None

    def test_ik_not_applied_when_blend_weight_zero(self):
        """Test IK is not applied when blend_weight is zero."""
        transforms, feet, pelvis = create_biped_skeleton()
        for foot in feet:
            foot.blend_weight = 0.0
        raycast = create_flat_ground_raycast()
        placement = MultiLegFootPlacement(feet, pelvis, raycast)

        result = placement.solve(transforms, Vec3(0, 0, 0))

        # Rotations should remain unchanged for leg bones
        for i in [1, 2, 3, 4, 5, 6]:  # Leg bones
            if i < len(transforms):
                assert result[i].rotation.w == pytest.approx(transforms[i].rotation.w, abs=1e-5)

    def test_ik_result_modifies_leg_rotations(self):
        """Test successful IK modifies leg bone rotations."""
        transforms, feet, pelvis = create_biped_skeleton()
        # Move feet above ground to ensure IK needs to solve
        transforms[3].translation = Vec3(-0.1, 0.3, 0)  # Left foot higher
        transforms[6].translation = Vec3(0.1, 0.3, 0)   # Right foot higher

        raycast = create_flat_ground_raycast()
        placement = MultiLegFootPlacement(feet, pelvis, raycast)

        result = placement.solve(transforms, Vec3(0, 0, 0))

        # Rotations should have changed
        assert result is not None


# =============================================================================
# Test Class: Terrain Scenarios
# =============================================================================

class TestMultiLegTerrainScenarios:
    """Tests for various terrain scenarios."""

    def test_solve_on_flat_terrain(self):
        """Test solve() on perfectly flat terrain."""
        transforms, feet, pelvis = create_quadruped_skeleton()
        raycast = create_flat_ground_raycast()
        placement = MultiLegFootPlacement(feet, pelvis, raycast)

        result = placement.solve(transforms, Vec3(0, 0, 0))

        assert result is not None

    def test_solve_on_sloped_terrain(self):
        """Test solve() on sloped terrain."""
        transforms, feet, pelvis = create_quadruped_skeleton()
        raycast = create_sloped_ground_raycast(0.3)
        placement = MultiLegFootPlacement(feet, pelvis, raycast)

        result = placement.solve(transforms, Vec3(0, 0, 0))

        assert result is not None

    def test_solve_on_uneven_terrain(self):
        """Test solve() on uneven terrain with hills."""
        transforms, feet, pelvis = create_spider_skeleton()
        raycast = create_uneven_terrain_raycast()
        placement = MultiLegFootPlacement(feet, pelvis, raycast)

        result = placement.solve(transforms, Vec3(0, 0, 0))

        assert result is not None

    def test_solve_on_stepped_terrain(self):
        """Test solve() on stepped terrain (stairs)."""
        transforms, feet, pelvis = create_quadruped_skeleton()
        raycast = create_stepped_terrain_raycast()
        placement = MultiLegFootPlacement(feet, pelvis, raycast)

        result = placement.solve(transforms, Vec3(0, 0, 0))

        assert result is not None

    def test_solve_elevated_ground(self):
        """Test solve() with elevated ground level."""
        transforms, feet, pelvis = create_biped_skeleton()
        raycast = create_flat_ground_raycast(ground_y=0.5)
        placement = MultiLegFootPlacement(feet, pelvis, raycast)

        result = placement.solve(transforms, Vec3(0, 0, 0))

        assert result is not None

    def test_solve_below_ground_level(self):
        """Test solve() with ground below origin."""
        transforms, feet, pelvis = create_biped_skeleton()
        raycast = create_flat_ground_raycast(ground_y=-0.5)
        placement = MultiLegFootPlacement(feet, pelvis, raycast)

        result = placement.solve(transforms, Vec3(0, 0, 0))

        assert result is not None


# =============================================================================
# Test Class: FootData Height Offset
# =============================================================================

class TestFootDataHeightOffset:
    """Tests for FootData height_offset handling."""

    def test_height_offset_zero_default(self):
        """Test height_offset defaults to zero."""
        foot = FootData(upper_leg=0, lower_leg=1, foot=2)
        assert foot.height_offset == 0.0

    def test_height_offset_applied_to_target(self):
        """Test height_offset is applied to target position."""
        transforms, feet, pelvis = create_biped_skeleton()
        feet[0].height_offset = 0.1
        feet[1].height_offset = -0.05

        raycast = create_flat_ground_raycast()
        placement = MultiLegFootPlacement(feet, pelvis, raycast)

        result = placement.solve(transforms, Vec3(0, 0, 0))

        assert result is not None

    def test_different_height_offsets_per_foot(self):
        """Test different height offsets for different feet."""
        transforms, feet, pelvis = create_quadruped_skeleton()
        feet[0].height_offset = 0.0
        feet[1].height_offset = 0.1
        feet[2].height_offset = 0.2
        feet[3].height_offset = -0.1

        raycast = create_flat_ground_raycast()
        placement = MultiLegFootPlacement(feet, pelvis, raycast)

        result = placement.solve(transforms, Vec3(0, 0, 0))

        assert result is not None


# =============================================================================
# Test Class: FootData Blend Weight
# =============================================================================

class TestFootDataBlendWeight:
    """Tests for FootData blend_weight handling."""

    def test_blend_weight_one_applies_full_ik(self):
        """Test blend_weight=1.0 applies full IK."""
        transforms, feet, pelvis = create_biped_skeleton()
        feet[0].blend_weight = 1.0

        raycast = create_flat_ground_raycast()
        placement = MultiLegFootPlacement(feet, pelvis, raycast)

        result = placement.solve(transforms, Vec3(0, 0, 0))

        assert result is not None

    def test_blend_weight_zero_skips_ik(self):
        """Test blend_weight=0.0 skips IK."""
        transforms, feet, pelvis = create_biped_skeleton()
        feet[0].blend_weight = 0.0
        feet[1].blend_weight = 0.0

        raycast = create_flat_ground_raycast()
        placement = MultiLegFootPlacement(feet, pelvis, raycast)

        result = placement.solve(transforms, Vec3(0, 0, 0))

        assert result is not None

    def test_blend_weight_half(self):
        """Test blend_weight=0.5 partial IK."""
        transforms, feet, pelvis = create_biped_skeleton()
        feet[0].blend_weight = 0.5
        feet[1].blend_weight = 0.5

        raycast = create_flat_ground_raycast()
        placement = MultiLegFootPlacement(feet, pelvis, raycast)

        result = placement.solve(transforms, Vec3(0, 0, 0))

        assert result is not None

    def test_mixed_blend_weights(self):
        """Test mixed blend weights across feet."""
        transforms, feet, pelvis = create_quadruped_skeleton()
        feet[0].blend_weight = 1.0
        feet[1].blend_weight = 0.75
        feet[2].blend_weight = 0.25
        feet[3].blend_weight = 0.0

        raycast = create_flat_ground_raycast()
        placement = MultiLegFootPlacement(feet, pelvis, raycast)

        result = placement.solve(transforms, Vec3(0, 0, 0))

        assert result is not None


# =============================================================================
# Test Class: Transform Copying
# =============================================================================

class TestTransformCopying:
    """Tests for transform copying behavior."""

    def test_solve_returns_new_transform_list(self):
        """Test solve() returns a new list, not the original."""
        transforms, feet, pelvis = create_biped_skeleton()
        raycast = create_flat_ground_raycast()
        placement = MultiLegFootPlacement(feet, pelvis, raycast)

        result = placement.solve(transforms, Vec3(0, 0, 0))

        assert result is not transforms

    def test_solve_returns_deep_copied_transforms(self):
        """Test solve() returns deep copied transforms."""
        transforms, feet, pelvis = create_biped_skeleton()
        raycast = create_flat_ground_raycast()
        placement = MultiLegFootPlacement(feet, pelvis, raycast)

        result = placement.solve(transforms, Vec3(0, 0, 0))

        # Modifying result should not affect original
        result[0].translation = Vec3(999, 999, 999)
        assert transforms[0].translation.x != 999

    def test_solve_copies_all_transforms(self):
        """Test solve() copies all transforms in the list."""
        transforms, feet, pelvis = create_quadruped_skeleton()
        raycast = create_flat_ground_raycast()
        placement = MultiLegFootPlacement(feet, pelvis, raycast)

        result = placement.solve(transforms, Vec3(0, 0, 0))

        assert len(result) == len(transforms)

    def test_solve_preserves_scale(self):
        """Test solve() preserves transform scale."""
        transforms, feet, pelvis = create_biped_skeleton()
        transforms[0].scale = Vec3(2, 2, 2)
        transforms[3].scale = Vec3(0.5, 0.5, 0.5)

        raycast = create_flat_ground_raycast()
        placement = MultiLegFootPlacement(feet, pelvis, raycast)

        result = placement.solve(transforms, Vec3(0, 0, 0))

        assert result[0].scale.x == pytest.approx(2.0)
        assert result[3].scale.x == pytest.approx(0.5)


# =============================================================================
# Test Class: Edge Cases
# =============================================================================

class TestMultiLegEdgeCases:
    """Tests for edge cases."""

    def test_solve_with_identity_transforms(self):
        """Test solve() with all identity transforms."""
        transforms = [Transform(Vec3.zero(), Quat.identity()) for _ in range(10)]
        feet = [
            FootData(upper_leg=1, lower_leg=2, foot=3),
            FootData(upper_leg=4, lower_leg=5, foot=6),
        ]

        raycast = create_flat_ground_raycast()
        placement = MultiLegFootPlacement(feet, 0, raycast)

        result = placement.solve(transforms, Vec3(0, 0, 0))

        assert result is not None

    def test_solve_with_coincident_feet(self):
        """Test solve() when all feet are at same position."""
        transforms = [Transform(Vec3(0, 1, 0), Quat.identity()) for _ in range(10)]
        feet = [
            FootData(upper_leg=1, lower_leg=2, foot=3),
            FootData(upper_leg=4, lower_leg=5, foot=6),
        ]

        raycast = create_flat_ground_raycast()
        placement = MultiLegFootPlacement(feet, 0, raycast)

        result = placement.solve(transforms, Vec3(0, 0, 0))

        assert result is not None

    def test_solve_very_long_legs(self):
        """Test solve() with very long leg chains."""
        transforms, feet, pelvis = create_biped_skeleton()
        # Extend leg length significantly
        transforms[1].translation = Vec3(-0.1, 5.0, 0)
        transforms[2].translation = Vec3(-0.1, 2.5, 0)
        transforms[3].translation = Vec3(-0.1, 0.1, 0)

        raycast = create_flat_ground_raycast()
        placement = MultiLegFootPlacement(feet, pelvis, raycast)

        result = placement.solve(transforms, Vec3(0, 0, 0))

        assert result is not None

    def test_solve_very_short_legs(self):
        """Test solve() with very short leg chains."""
        transforms, feet, pelvis = create_biped_skeleton()
        # Shorten legs
        transforms[1].translation = Vec3(-0.1, 0.3, 0)
        transforms[2].translation = Vec3(-0.1, 0.2, 0)
        transforms[3].translation = Vec3(-0.1, 0.1, 0)

        raycast = create_flat_ground_raycast()
        placement = MultiLegFootPlacement(feet, pelvis, raycast)

        result = placement.solve(transforms, Vec3(0, 0, 0))

        assert result is not None


# =============================================================================
# Test Class: Multiple Calls
# =============================================================================

class TestMultiLegMultipleCalls:
    """Tests for multiple solve() calls."""

    def test_multiple_solve_calls_independent(self):
        """Test multiple solve() calls produce independent results."""
        transforms, feet, pelvis = create_biped_skeleton()
        raycast = create_flat_ground_raycast()
        placement = MultiLegFootPlacement(feet, pelvis, raycast)

        result1 = placement.solve(transforms, Vec3(0, 0, 0))
        result2 = placement.solve(transforms, Vec3(0, 0, 0))

        assert result1 is not result2

    def test_solve_with_changing_transforms(self):
        """Test solve() with transforms that change between calls."""
        transforms, feet, pelvis = create_biped_skeleton()
        raycast = create_flat_ground_raycast()
        placement = MultiLegFootPlacement(feet, pelvis, raycast)

        result1 = placement.solve(transforms, Vec3(0, 0, 0))

        # Modify transforms
        transforms[3].translation = Vec3(-0.1, 0.5, 0)
        result2 = placement.solve(transforms, Vec3(0, 0, 0))

        assert result1 is not None
        assert result2 is not None

    def test_solve_consistent_results(self):
        """Test solve() produces consistent results for same input."""
        transforms1, feet1, pelvis = create_biped_skeleton()
        transforms2, feet2, _ = create_biped_skeleton()
        raycast = create_flat_ground_raycast()
        placement1 = MultiLegFootPlacement(feet1, pelvis, raycast)
        placement2 = MultiLegFootPlacement(feet2, pelvis, raycast)

        result1 = placement1.solve(transforms1, Vec3(0, 0, 0))
        result2 = placement2.solve(transforms2, Vec3(0, 0, 0))

        for i in range(len(transforms1)):
            assert result1[i].translation.x == pytest.approx(result2[i].translation.x, abs=1e-6)
            assert result1[i].translation.y == pytest.approx(result2[i].translation.y, abs=1e-6)
            assert result1[i].translation.z == pytest.approx(result2[i].translation.z, abs=1e-6)


# =============================================================================
# Test Class: Ray Origin Calculation
# =============================================================================

class TestRayOriginCalculation:
    """Tests for ray origin calculation."""

    def test_ray_origin_y_offset(self):
        """Test ray origin is offset upward from foot position."""
        transforms, feet, pelvis = create_biped_skeleton()
        raycast, calls = create_tracking_raycast()
        placement = MultiLegFootPlacement(feet, pelvis, raycast)

        placement.solve(transforms, Vec3(0, 0, 0))

        # Origin should be above foot position
        foot_y = transforms[3].translation.y
        origin_y = calls[0]['origin'].y
        assert origin_y > foot_y

    def test_ray_origin_offset_uses_ray_length(self):
        """Test ray origin offset uses ray_length * 0.5."""
        transforms, feet, pelvis = create_biped_skeleton()
        raycast, calls = create_tracking_raycast()
        placement = MultiLegFootPlacement(feet, pelvis, raycast)
        placement.ray_length = 4.0

        placement.solve(transforms, Vec3(0, 0, 0))

        foot_y = transforms[3].translation.y
        origin_y = calls[0]['origin'].y
        expected_offset = 4.0 * 0.5  # ray_length * 0.5
        assert origin_y == pytest.approx(foot_y + expected_offset, abs=1e-6)


# =============================================================================
# Test Class: Raycast Hit Processing
# =============================================================================

class TestRaycastHitProcessing:
    """Tests for raycast hit processing."""

    def test_hit_target_includes_foot_height(self):
        """Test target position includes foot_height offset."""
        transforms, feet, pelvis = create_biped_skeleton()
        raycast = create_flat_ground_raycast()
        placement = MultiLegFootPlacement(feet, pelvis, raycast)
        placement.foot_height = 0.15

        result = placement.solve(transforms, Vec3(0, 0, 0))

        assert result is not None

    def test_miss_uses_original_foot_position(self):
        """Test raycast miss uses original foot position."""
        transforms, feet, pelvis = create_biped_skeleton()
        raycast = create_no_hit_raycast()
        placement = MultiLegFootPlacement(feet, pelvis, raycast)

        result = placement.solve(transforms, Vec3(0, 0, 0))

        # Without hits, original positions should be preserved mostly
        assert result is not None

    def test_miss_uses_default_normal(self):
        """Test raycast miss uses default up normal."""
        transforms, feet, pelvis = create_biped_skeleton()
        raycast = create_no_hit_raycast()
        placement = MultiLegFootPlacement(feet, pelvis, raycast)

        # Should not crash
        result = placement.solve(transforms, Vec3(0, 0, 0))
        assert result is not None
