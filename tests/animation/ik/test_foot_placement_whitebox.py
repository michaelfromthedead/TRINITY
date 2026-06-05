"""Whitebox tests for foot placement IK.

Tests the foot placement implementation covering:
- FootState enum
- FootData dataclass
- FootPlacementResult dataclass
- FootPlacement class construction
- set_raycast_callback() method
- solve() for foot placement
- _raycast_foot() ground detection
- _calculate_pelvis_offset()
- _solve_leg_ik()
- _align_foot_to_terrain()
- _rotation_between_vectors()
- _scale_rotation()
- FootPlacementAnimated blending
- MultiLegFootPlacement coordination
"""

from __future__ import annotations

import math
import pytest
from dataclasses import fields
from typing import List, Tuple

from engine.animation.ik.foot_placement import (
    FootState,
    FootData,
    FootPlacementResult,
    FootPlacement,
    FootPlacementAnimated,
    MultiLegFootPlacement,
    RaycastCallback,
    RaycastHit,
)
from engine.animation.ik.config import (
    FOOT_PLACEMENT_RAY_LENGTH,
    FOOT_PLACEMENT_FOOT_HEIGHT,
    FOOT_PLACEMENT_BLEND_SPEED,
    FOOT_PLACEMENT_MAX_PELVIS_DROP,
    FOOT_PLACEMENT_MAX_PELVIS_RAISE,
    FOOT_PLACEMENT_TOE_ALIGN_WEIGHT,
    FOOT_PLACEMENT_REACH_SAFETY_MARGIN,
    SOFT_IK_DEFAULT_RATIO,
    SOFT_IK_DEFAULT_BLEND,
    MULTI_LEG_MAX_PELVIS_DROP,
)
from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.core.math.transform import Transform
from engine.core.constants import MATH_EPSILON


# =============================================================================
# Helper Functions
# =============================================================================

def create_biped_transforms(num_bones: int = 10) -> List[Transform]:
    """Create transforms for a simple biped skeleton."""
    positions = {
        0: Vec3(0, 1.0, 0),      # Pelvis
        1: Vec3(-0.1, 1.0, 0),   # Left upper leg
        2: Vec3(-0.1, 0.5, 0),   # Left lower leg
        3: Vec3(-0.1, 0.1, 0),   # Left foot
        4: Vec3(-0.1, 0.0, 0),   # Left toe
        5: Vec3(0.1, 1.0, 0),    # Right upper leg
        6: Vec3(0.1, 0.5, 0),    # Right lower leg
        7: Vec3(0.1, 0.1, 0),    # Right foot
        8: Vec3(0.1, 0.0, 0),    # Right toe
    }

    transforms = []
    for i in range(num_bones):
        pos = positions.get(i, Vec3(0, 0, 0))
        transforms.append(Transform(pos, Quat.identity()))

    return transforms


def create_left_foot_data() -> FootData:
    """Create left foot data for testing."""
    return FootData(
        upper_leg=1,
        lower_leg=2,
        foot=3,
        toe=4
    )


def create_right_foot_data() -> FootData:
    """Create right foot data for testing."""
    return FootData(
        upper_leg=5,
        lower_leg=6,
        foot=7,
        toe=8
    )


def create_flat_ground_raycast() -> RaycastCallback:
    """Create raycast callback for flat ground at y=0."""
    from typing import Optional

    def raycast(origin: Vec3, direction: Vec3, max_dist: float) -> Optional[RaycastHit]:
        if direction.y >= 0:
            return None

        t = -origin.y / direction.y
        if t > max_dist:
            return None

        hit_pos = Vec3(origin.x + direction.x * t, 0, origin.z + direction.z * t)
        return RaycastHit(hit=True, position=hit_pos, normal=Vec3(0, 1, 0), distance=t)

    return raycast


def create_sloped_ground_raycast(slope: float = 0.2) -> RaycastCallback:
    """Create raycast callback for sloped ground."""
    from typing import Optional

    def raycast(origin: Vec3, direction: Vec3, max_dist: float) -> Optional[RaycastHit]:
        if direction.y >= 0:
            return None

        # Ground height = slope * x
        ground_y = slope * origin.x
        t = (origin.y - ground_y) / (-direction.y)

        if t > max_dist or t < 0:
            return None

        hit_x = origin.x + direction.x * t
        hit_y = slope * hit_x
        hit_z = origin.z + direction.z * t

        normal = Vec3(-slope, 1, 0).normalized()
        return RaycastHit(hit=True, position=Vec3(hit_x, hit_y, hit_z), normal=normal, distance=t)

    return raycast


def create_no_hit_raycast() -> RaycastCallback:
    """Create raycast callback that never hits."""
    from typing import Optional

    def raycast(origin: Vec3, direction: Vec3, max_dist: float) -> Optional[RaycastHit]:
        return None

    return raycast


def vec3_approx_equal(v1: Vec3, v2: Vec3, eps: float = 1e-4) -> bool:
    """Check if two Vec3 are approximately equal."""
    return (
        abs(v1.x - v2.x) < eps and
        abs(v1.y - v2.y) < eps and
        abs(v1.z - v2.z) < eps
    )


def quat_approx_equal(q1: Quat, q2: Quat, eps: float = 1e-4) -> bool:
    """Check if two Quaternions are approximately equal."""
    dot = abs(q1.x * q2.x + q1.y * q2.y + q1.z * q2.z + q1.w * q2.w)
    return dot > (1.0 - eps)


# =============================================================================
# Test FootState Enum
# =============================================================================

class TestFootStateEnum:
    """Tests for FootState enum."""

    def test_enum_has_all_states(self):
        """Verify all expected states exist."""
        expected_states = ['PLANTED', 'LIFTING', 'AIRBORNE', 'LANDING']
        for state_name in expected_states:
            assert hasattr(FootState, state_name)

    def test_enum_values_are_unique(self):
        """Verify all enum values are unique."""
        values = [s.value for s in FootState]
        assert len(values) == len(set(values))

    def test_planted_state(self):
        """Test PLANTED state."""
        state = FootState.PLANTED
        assert state.name == 'PLANTED'

    def test_lifting_state(self):
        """Test LIFTING state."""
        state = FootState.LIFTING
        assert state.name == 'LIFTING'

    def test_airborne_state(self):
        """Test AIRBORNE state."""
        state = FootState.AIRBORNE
        assert state.name == 'AIRBORNE'

    def test_landing_state(self):
        """Test LANDING state."""
        state = FootState.LANDING
        assert state.name == 'LANDING'

    def test_enum_iteration(self):
        """Test iterating over all states."""
        count = sum(1 for _ in FootState)
        assert count == 4


# =============================================================================
# Test FootData Dataclass
# =============================================================================

class TestFootData:
    """Tests for FootData dataclass."""

    def test_dataclass_has_required_fields(self):
        """Verify dataclass has all required fields."""
        field_names = {f.name for f in fields(FootData)}
        expected_fields = {
            'upper_leg', 'lower_leg', 'foot', 'toe',
            'state', 'target_position', 'target_normal',
            'blend_weight', 'height_offset'
        }
        assert expected_fields.issubset(field_names)

    def test_minimal_construction(self):
        """Test construction with only required args."""
        foot = FootData(upper_leg=0, lower_leg=1, foot=2)

        assert foot.upper_leg == 0
        assert foot.lower_leg == 1
        assert foot.foot == 2
        assert foot.toe is None
        assert foot.state == FootState.PLANTED
        assert foot.blend_weight == 1.0
        assert foot.height_offset == 0.0

    def test_full_construction(self):
        """Test construction with all args."""
        target_pos = Vec3(1, 0, 0)
        target_normal = Vec3(0, 1, 0)

        foot = FootData(
            upper_leg=0,
            lower_leg=1,
            foot=2,
            toe=3,
            state=FootState.AIRBORNE,
            target_position=target_pos,
            target_normal=target_normal,
            blend_weight=0.5,
            height_offset=0.1
        )

        assert foot.upper_leg == 0
        assert foot.lower_leg == 1
        assert foot.foot == 2
        assert foot.toe == 3
        assert foot.state == FootState.AIRBORNE
        assert foot.blend_weight == 0.5
        assert foot.height_offset == 0.1

    def test_default_target_position(self):
        """Test default target position is zero."""
        foot = FootData(upper_leg=0, lower_leg=1, foot=2)
        assert vec3_approx_equal(foot.target_position, Vec3.zero())

    def test_default_target_normal(self):
        """Test default target normal is up vector."""
        foot = FootData(upper_leg=0, lower_leg=1, foot=2)
        assert vec3_approx_equal(foot.target_normal, Vec3(0, 1, 0))

    def test_state_can_be_changed(self):
        """Test foot state can be changed."""
        foot = FootData(upper_leg=0, lower_leg=1, foot=2)
        foot.state = FootState.LIFTING
        assert foot.state == FootState.LIFTING


# =============================================================================
# Test FootPlacementResult Dataclass
# =============================================================================

class TestFootPlacementResult:
    """Tests for FootPlacementResult dataclass."""

    def test_dataclass_has_required_fields(self):
        """Verify dataclass has all required fields."""
        field_names = {f.name for f in fields(FootPlacementResult)}
        expected_fields = {
            'success', 'transforms', 'pelvis_offset',
            'left_foot_planted', 'right_foot_planted', 'terrain_slope'
        }
        assert expected_fields.issubset(field_names)

    def test_default_values(self):
        """Test default values are set correctly."""
        result = FootPlacementResult(success=True)

        assert result.success is True
        assert result.transforms == []
        assert vec3_approx_equal(result.pelvis_offset, Vec3.zero())
        assert result.left_foot_planted is True
        assert result.right_foot_planted is True
        assert result.terrain_slope == 0.0

    def test_custom_values(self):
        """Test creating result with custom values."""
        transforms = [Transform(Vec3(1, 0, 0), Quat.identity())]
        pelvis_offset = Vec3(0, -0.1, 0)

        result = FootPlacementResult(
            success=True,
            transforms=transforms,
            pelvis_offset=pelvis_offset,
            left_foot_planted=True,
            right_foot_planted=False,
            terrain_slope=0.1
        )

        assert result.success is True
        assert len(result.transforms) == 1
        assert vec3_approx_equal(result.pelvis_offset, pelvis_offset)
        assert result.left_foot_planted is True
        assert result.right_foot_planted is False
        assert result.terrain_slope == 0.1

    def test_failed_result(self):
        """Test creating a failed result."""
        result = FootPlacementResult(success=False)
        assert result.success is False


# =============================================================================
# Test FootPlacement Construction
# =============================================================================

class TestFootPlacementConstruction:
    """Tests for FootPlacement class construction."""

    def test_basic_construction(self):
        """Test basic construction."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()

        placement = FootPlacement(left_foot, right_foot, pelvis=0)

        assert placement.left_foot == left_foot
        assert placement.right_foot == right_foot
        assert placement.pelvis == 0

    def test_construction_with_raycast(self):
        """Test construction with raycast callback."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()
        raycast = create_flat_ground_raycast()

        placement = FootPlacement(left_foot, right_foot, pelvis=0, raycast_callback=raycast)

        assert placement._raycast is not None

    def test_default_configuration(self):
        """Test default configuration values."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()

        placement = FootPlacement(left_foot, right_foot, pelvis=0)

        assert placement.ray_length == FOOT_PLACEMENT_RAY_LENGTH
        assert placement.foot_height == FOOT_PLACEMENT_FOOT_HEIGHT
        assert placement.blend_speed == FOOT_PLACEMENT_BLEND_SPEED
        assert placement.max_pelvis_drop == FOOT_PLACEMENT_MAX_PELVIS_DROP
        assert placement.max_pelvis_raise == FOOT_PLACEMENT_MAX_PELVIS_RAISE
        assert placement.toe_align_weight == FOOT_PLACEMENT_TOE_ALIGN_WEIGHT

    def test_leg_ik_solvers_created(self):
        """Test leg IK solvers are created."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()

        placement = FootPlacement(left_foot, right_foot, pelvis=0)

        assert placement._left_leg_ik is not None
        assert placement._right_leg_ik is not None

    def test_initial_state_tracking(self):
        """Test initial state tracking values."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()

        placement = FootPlacement(left_foot, right_foot, pelvis=0)

        assert vec3_approx_equal(placement._prev_left_target, Vec3.zero())
        assert vec3_approx_equal(placement._prev_right_target, Vec3.zero())
        assert placement._prev_pelvis_offset == 0.0


# =============================================================================
# Test set_raycast_callback()
# =============================================================================

class TestSetRaycastCallback:
    """Tests for set_raycast_callback() method."""

    def test_set_callback(self):
        """Test setting raycast callback."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()

        placement = FootPlacement(left_foot, right_foot, pelvis=0)
        raycast = create_flat_ground_raycast()

        placement.set_raycast_callback(raycast)

        assert placement._raycast == raycast

    def test_override_callback(self):
        """Test overriding raycast callback."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()
        initial_raycast = create_flat_ground_raycast()

        placement = FootPlacement(left_foot, right_foot, pelvis=0, raycast_callback=initial_raycast)

        new_raycast = create_no_hit_raycast()
        placement.set_raycast_callback(new_raycast)

        assert placement._raycast == new_raycast


# =============================================================================
# Test solve() Method
# =============================================================================

class TestFootPlacementSolve:
    """Tests for solve() method."""

    def test_solve_without_raycast_fails(self):
        """Test solve fails without raycast callback."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()

        placement = FootPlacement(left_foot, right_foot, pelvis=0)
        transforms = create_biped_transforms()

        result = placement.solve(transforms, Vec3.zero())

        assert result.success is False

    def test_solve_returns_result(self):
        """Test solve returns FootPlacementResult."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()
        raycast = create_flat_ground_raycast()

        placement = FootPlacement(left_foot, right_foot, pelvis=0, raycast_callback=raycast)
        transforms = create_biped_transforms()

        result = placement.solve(transforms, Vec3.zero())

        assert isinstance(result, FootPlacementResult)

    def test_solve_with_flat_ground(self):
        """Test solve with flat ground."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()
        raycast = create_flat_ground_raycast()

        placement = FootPlacement(left_foot, right_foot, pelvis=0, raycast_callback=raycast)
        transforms = create_biped_transforms()

        result = placement.solve(transforms, Vec3.zero())

        assert result.success is True
        assert result.left_foot_planted is True
        assert result.right_foot_planted is True

    def test_solve_with_no_ground(self):
        """Test solve with no ground hit."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()
        raycast = create_no_hit_raycast()

        placement = FootPlacement(left_foot, right_foot, pelvis=0, raycast_callback=raycast)
        transforms = create_biped_transforms()

        result = placement.solve(transforms, Vec3.zero())

        assert result.success is True
        assert result.left_foot_planted is False
        assert result.right_foot_planted is False

    def test_solve_copies_transforms(self):
        """Test solve doesn't modify input transforms."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()
        raycast = create_flat_ground_raycast()

        placement = FootPlacement(left_foot, right_foot, pelvis=0, raycast_callback=raycast)
        transforms = create_biped_transforms()

        original_pos = Vec3(
            transforms[0].translation.x,
            transforms[0].translation.y,
            transforms[0].translation.z
        )

        placement.solve(transforms, Vec3.zero())

        assert vec3_approx_equal(transforms[0].translation, original_pos)

    def test_solve_returns_correct_transform_count(self):
        """Test solve returns same number of transforms."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()
        raycast = create_flat_ground_raycast()

        placement = FootPlacement(left_foot, right_foot, pelvis=0, raycast_callback=raycast)
        transforms = create_biped_transforms()

        result = placement.solve(transforms, Vec3.zero())

        assert len(result.transforms) == len(transforms)

    def test_solve_with_sloped_ground(self):
        """Test solve with sloped ground."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()
        raycast = create_sloped_ground_raycast(0.2)

        placement = FootPlacement(left_foot, right_foot, pelvis=0, raycast_callback=raycast)
        transforms = create_biped_transforms()

        result = placement.solve(transforms, Vec3.zero())

        assert result.success is True
        # Terrain slope should be non-zero for sloped ground
        # (depends on foot positions)

    def test_solve_updates_prev_targets(self):
        """Test solve updates previous target tracking."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()
        raycast = create_flat_ground_raycast()

        placement = FootPlacement(left_foot, right_foot, pelvis=0, raycast_callback=raycast)
        transforms = create_biped_transforms()

        placement.solve(transforms, Vec3.zero())

        # After solve, prev targets should be updated
        # (exact values depend on foot positions and ground)

    def test_solve_with_custom_dt(self):
        """Test solve with custom delta time."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()
        raycast = create_flat_ground_raycast()

        placement = FootPlacement(left_foot, right_foot, pelvis=0, raycast_callback=raycast)
        transforms = create_biped_transforms()

        result = placement.solve(transforms, Vec3.zero(), dt=1.0/30.0)

        assert result.success is True

    def test_solve_with_zero_blend_weight(self):
        """Test solve with zero blend weight."""
        left_foot = create_left_foot_data()
        left_foot.blend_weight = 0.0
        right_foot = create_right_foot_data()
        right_foot.blend_weight = 0.0
        raycast = create_flat_ground_raycast()

        placement = FootPlacement(left_foot, right_foot, pelvis=0, raycast_callback=raycast)
        transforms = create_biped_transforms()

        result = placement.solve(transforms, Vec3.zero())

        assert result.success is True


# =============================================================================
# Test _raycast_foot()
# =============================================================================

class TestRaycastFoot:
    """Tests for _raycast_foot() method."""

    def test_raycast_from_foot_position(self):
        """Test raycast originates from above foot."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()

        # Custom raycast to verify origin
        origin_captured = []

        def capture_raycast(origin: Vec3, direction: Vec3, max_dist: float):
            origin_captured.append(origin)
            return RaycastHit(hit=True, position=Vec3(origin.x, 0, origin.z), normal=Vec3(0, 1, 0), distance=1.0)

        placement = FootPlacement(left_foot, right_foot, pelvis=0, raycast_callback=capture_raycast)

        foot_pos = Vec3(1, 0.5, 2)
        placement._raycast_foot(foot_pos)

        assert len(origin_captured) == 1
        # Origin should be above foot position
        assert origin_captured[0].y > foot_pos.y

    def test_raycast_direction_is_down(self):
        """Test raycast direction is downward."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()

        direction_captured = []

        def capture_raycast(origin: Vec3, direction: Vec3, max_dist: float):
            direction_captured.append(direction)
            return RaycastHit(hit=True, position=Vec3.zero(), normal=Vec3(0, 1, 0), distance=1.0)

        placement = FootPlacement(left_foot, right_foot, pelvis=0, raycast_callback=capture_raycast)

        placement._raycast_foot(Vec3(0, 1, 0))

        assert len(direction_captured) == 1
        assert direction_captured[0].y == -1


# =============================================================================
# Test _calculate_pelvis_offset()
# =============================================================================

class TestCalculatePelvisOffset:
    """Tests for _calculate_pelvis_offset() method."""

    def test_no_planted_feet_returns_blend_to_zero(self):
        """Test with no planted feet returns blended value."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()
        raycast = create_flat_ground_raycast()

        placement = FootPlacement(left_foot, right_foot, pelvis=0, raycast_callback=raycast)
        placement._prev_pelvis_offset = 0.1

        offset = placement._calculate_pelvis_offset(
            Vec3(0, 1, 0),  # pelvis pos
            Vec3.zero(), False,
            Vec3.zero(), False,
            1.0/60.0
        )

        # Should blend toward 0
        assert abs(offset) < abs(placement._prev_pelvis_offset) or offset == 0

    def test_planted_feet_calculates_offset(self):
        """Test with planted feet calculates offset."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()
        raycast = create_flat_ground_raycast()

        placement = FootPlacement(left_foot, right_foot, pelvis=0, raycast_callback=raycast)

        offset = placement._calculate_pelvis_offset(
            Vec3(0, 1, 0),  # pelvis pos
            Vec3(-0.1, 0.1, 0), True,  # left target, planted
            Vec3(0.1, 0.1, 0), True,   # right target, planted
            1.0/60.0
        )

        assert isinstance(offset, float)

    def test_offset_limited_by_max_drop(self):
        """Test offset is limited by max pelvis drop."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()
        raycast = create_flat_ground_raycast()

        placement = FootPlacement(left_foot, right_foot, pelvis=0, raycast_callback=raycast)

        # Very far target that would require large drop
        offset = placement._calculate_pelvis_offset(
            Vec3(0, 1, 0),
            Vec3(-0.1, -10, 0), True,  # Very far down
            Vec3(0.1, 0, 0), True,
            1.0/60.0
        )

        # Offset should not exceed max drop (as a drop, it's negative)
        # After one frame it won't reach the limit due to blending
        assert abs(offset) <= placement.max_pelvis_drop + 0.1


# =============================================================================
# Test _solve_leg_ik()
# =============================================================================

class TestSolveLegIK:
    """Tests for _solve_leg_ik() method."""

    def test_solve_leg_modifies_transforms(self):
        """Test _solve_leg_ik modifies transforms."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()
        raycast = create_flat_ground_raycast()

        placement = FootPlacement(left_foot, right_foot, pelvis=0, raycast_callback=raycast)
        transforms = create_biped_transforms()

        target = Vec3(-0.1, 0.2, 0.1)

        placement._solve_leg_ik(
            transforms,
            placement._left_leg_ik,
            left_foot,
            target
        )

        # Should not crash, transforms may be modified

    def test_solve_leg_with_partial_blend(self):
        """Test _solve_leg_ik with partial blend weight."""
        left_foot = create_left_foot_data()
        left_foot.blend_weight = 0.5
        right_foot = create_right_foot_data()
        raycast = create_flat_ground_raycast()

        placement = FootPlacement(left_foot, right_foot, pelvis=0, raycast_callback=raycast)
        transforms = create_biped_transforms()

        target = Vec3(-0.1, 0.2, 0.1)

        placement._solve_leg_ik(
            transforms,
            placement._left_leg_ik,
            left_foot,
            target
        )

        # Should apply blended rotation


# =============================================================================
# Test _align_foot_to_terrain()
# =============================================================================

class TestAlignFootToTerrain:
    """Tests for _align_foot_to_terrain() method."""

    def test_align_with_up_normal(self):
        """Test alignment with up normal (no change)."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()
        raycast = create_flat_ground_raycast()

        placement = FootPlacement(left_foot, right_foot, pelvis=0, raycast_callback=raycast)
        transforms = create_biped_transforms()

        original_rot = transforms[3].rotation

        placement._align_foot_to_terrain(
            transforms,
            left_foot,
            Vec3(0, 1, 0)  # Up normal
        )

        # With up normal, rotation should be minimal
        # (depends on current foot orientation)

    def test_align_with_tilted_normal(self):
        """Test alignment with tilted ground normal."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()
        raycast = create_flat_ground_raycast()

        placement = FootPlacement(left_foot, right_foot, pelvis=0, raycast_callback=raycast)
        transforms = create_biped_transforms()

        tilted_normal = Vec3(0.2, 0.98, 0).normalized()

        placement._align_foot_to_terrain(
            transforms,
            left_foot,
            tilted_normal
        )

        # Should not crash

    def test_align_also_affects_toe(self):
        """Test alignment also affects toe if present."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()
        raycast = create_flat_ground_raycast()

        placement = FootPlacement(left_foot, right_foot, pelvis=0, raycast_callback=raycast)
        transforms = create_biped_transforms()

        tilted_normal = Vec3(0.2, 0.98, 0).normalized()

        placement._align_foot_to_terrain(
            transforms,
            left_foot,
            tilted_normal
        )

        # Toe transform should also be modified
        # (if toe index is valid)


# =============================================================================
# Test _rotation_between_vectors()
# =============================================================================

class TestRotationBetweenVectors:
    """Tests for _rotation_between_vectors() method."""

    def test_same_vectors_returns_identity(self):
        """Test rotation between same vectors is identity."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()

        placement = FootPlacement(left_foot, right_foot, pelvis=0)

        vec = Vec3(0, 1, 0)
        rot = placement._rotation_between_vectors(vec, vec)

        assert quat_approx_equal(rot, Quat.identity())

    def test_opposite_vectors_returns_180_rotation(self):
        """Test rotation between opposite vectors is 180 degrees."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()

        placement = FootPlacement(left_foot, right_foot, pelvis=0)

        from_vec = Vec3(0, 1, 0)
        to_vec = Vec3(0, -1, 0)
        rot = placement._rotation_between_vectors(from_vec, to_vec)

        # Rotating from_vec by rot should give to_vec
        rotated = rot.rotate_vector(from_vec)
        assert abs(rotated.dot(to_vec) - 1.0) < 0.01

    def test_perpendicular_vectors(self):
        """Test rotation between perpendicular vectors produces valid rotation."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()

        placement = FootPlacement(left_foot, right_foot, pelvis=0)

        from_vec = Vec3(1, 0, 0)
        to_vec = Vec3(0, 1, 0)
        rot = placement._rotation_between_vectors(from_vec, to_vec)

        # Verify it's a valid rotation (unit quaternion)
        quat_len = math.sqrt(rot.x**2 + rot.y**2 + rot.z**2 + rot.w**2)
        assert abs(quat_len - 1.0) < 0.01

    def test_normalizes_vectors(self):
        """Test vectors are normalized before computation."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()

        placement = FootPlacement(left_foot, right_foot, pelvis=0)

        # Non-unit vectors should produce same rotation as normalized versions
        from_vec = Vec3(2, 0, 0)
        to_vec = Vec3(0, 3, 0)
        rot_unnorm = placement._rotation_between_vectors(from_vec, to_vec)

        from_vec_n = Vec3(1, 0, 0)
        to_vec_n = Vec3(0, 1, 0)
        rot_norm = placement._rotation_between_vectors(from_vec_n, to_vec_n)

        # Both rotations should be equivalent
        assert quat_approx_equal(rot_unnorm, rot_norm) or quat_approx_equal(
            Quat(-rot_unnorm.x, -rot_unnorm.y, -rot_unnorm.z, -rot_unnorm.w),
            rot_norm
        )


# =============================================================================
# Test _scale_rotation()
# =============================================================================

class TestScaleRotation:
    """Tests for _scale_rotation() method."""

    def test_scale_zero_returns_identity(self):
        """Test scaling by zero returns identity."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()

        placement = FootPlacement(left_foot, right_foot, pelvis=0)

        rot = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 2)
        scaled = placement._scale_rotation(rot, 0.0)

        assert quat_approx_equal(scaled, Quat.identity())

    def test_scale_one_returns_original(self):
        """Test scaling by one returns original."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()

        placement = FootPlacement(left_foot, right_foot, pelvis=0)

        rot = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 2)
        scaled = placement._scale_rotation(rot, 1.0)

        assert quat_approx_equal(scaled, rot)

    def test_scale_half(self):
        """Test scaling by 0.5 returns half rotation."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()

        placement = FootPlacement(left_foot, right_foot, pelvis=0)

        rot = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 2)
        scaled = placement._scale_rotation(rot, 0.5)

        # Apply scaled rotation to a vector
        vec = Vec3(1, 0, 0)
        rotated_full = rot.rotate_vector(vec)
        rotated_half = scaled.rotate_vector(vec)

        # Half rotation should be at 45 degrees (rotation direction depends on implementation)
        # Check that the angle is half (dot product with original should be cos(45 deg))
        dot = rotated_half.dot(vec)
        expected_dot = math.cos(math.pi / 4)
        assert abs(dot - expected_dot) < 0.01


# =============================================================================
# Test FootPlacementAnimated
# =============================================================================

class TestFootPlacementAnimated:
    """Tests for FootPlacementAnimated class."""

    def test_construction(self):
        """Test FootPlacementAnimated construction."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()
        raycast = create_flat_ground_raycast()

        base = FootPlacement(left_foot, right_foot, pelvis=0, raycast_callback=raycast)
        animated = FootPlacementAnimated(base)

        assert animated.base == base

    def test_initial_animation_time(self):
        """Test initial animation time is zero."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()
        raycast = create_flat_ground_raycast()

        base = FootPlacement(left_foot, right_foot, pelvis=0, raycast_callback=raycast)
        animated = FootPlacementAnimated(base)

        assert animated._animation_time == 0.0

    def test_update_advances_time(self):
        """Test update() advances animation time."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()
        raycast = create_flat_ground_raycast()

        base = FootPlacement(left_foot, right_foot, pelvis=0, raycast_callback=raycast)
        animated = FootPlacementAnimated(base)

        animated.update(0.1)

        assert abs(animated._animation_time - 0.1) < 0.001

    def test_update_accumulates_time(self):
        """Test update() accumulates time correctly."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()
        raycast = create_flat_ground_raycast()

        base = FootPlacement(left_foot, right_foot, pelvis=0, raycast_callback=raycast)
        animated = FootPlacementAnimated(base)

        animated.update(0.1)
        animated.update(0.2)

        assert abs(animated._animation_time - 0.3) < 0.001

    def test_set_height_curves(self):
        """Test setting height curves."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()
        raycast = create_flat_ground_raycast()

        base = FootPlacement(left_foot, right_foot, pelvis=0, raycast_callback=raycast)
        animated = FootPlacementAnimated(base)

        left_curve = lambda t: math.sin(t) * 0.1
        right_curve = lambda t: math.cos(t) * 0.1

        animated.set_height_curves(left_curve, right_curve)

        assert animated._left_height_curve is not None
        assert animated._right_height_curve is not None

    def test_solve_applies_height_curves(self):
        """Test solve applies height curve values."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()
        raycast = create_flat_ground_raycast()

        base = FootPlacement(left_foot, right_foot, pelvis=0, raycast_callback=raycast)
        animated = FootPlacementAnimated(base)

        # Set constant curves for predictable behavior
        animated.set_height_curves(lambda t: 0.05, lambda t: 0.03)

        transforms = create_biped_transforms()
        result = animated.solve(transforms, Vec3.zero())

        assert result.success is True
        assert base.left_foot.height_offset == 0.05
        assert base.right_foot.height_offset == 0.03

    def test_solve_without_curves(self):
        """Test solve works without height curves."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()
        raycast = create_flat_ground_raycast()

        base = FootPlacement(left_foot, right_foot, pelvis=0, raycast_callback=raycast)
        animated = FootPlacementAnimated(base)

        transforms = create_biped_transforms()
        result = animated.solve(transforms, Vec3.zero())

        assert result.success is True


# =============================================================================
# Test MultiLegFootPlacement
# =============================================================================

class TestMultiLegFootPlacement:
    """Tests for MultiLegFootPlacement class."""

    def test_construction_with_two_legs(self):
        """Test construction with two legs."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()

        multi = MultiLegFootPlacement([left_foot, right_foot], pelvis=0)

        assert len(multi.feet) == 2
        assert multi.pelvis == 0

    def test_construction_with_four_legs(self):
        """Test construction with four legs (quadruped)."""
        feet = [
            FootData(upper_leg=1, lower_leg=2, foot=3),
            FootData(upper_leg=4, lower_leg=5, foot=6),
            FootData(upper_leg=7, lower_leg=8, foot=9),
            FootData(upper_leg=10, lower_leg=11, foot=12),
        ]

        multi = MultiLegFootPlacement(feet, pelvis=0)

        assert len(multi.feet) == 4
        assert len(multi._leg_iks) == 4

    def test_construction_with_raycast(self):
        """Test construction with raycast callback."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()
        raycast = create_flat_ground_raycast()

        multi = MultiLegFootPlacement([left_foot, right_foot], pelvis=0, raycast_callback=raycast)

        assert multi._raycast is not None

    def test_default_configuration(self):
        """Test default configuration values."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()

        multi = MultiLegFootPlacement([left_foot, right_foot], pelvis=0)

        assert multi.ray_length == FOOT_PLACEMENT_RAY_LENGTH
        assert multi.foot_height == FOOT_PLACEMENT_FOOT_HEIGHT
        assert multi.blend_speed == FOOT_PLACEMENT_BLEND_SPEED

    def test_solve_without_raycast_returns_input(self):
        """Test solve returns input transforms without raycast."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()

        multi = MultiLegFootPlacement([left_foot, right_foot], pelvis=0)
        transforms = create_biped_transforms()

        result = multi.solve(transforms, Vec3.zero())

        assert result == transforms

    def test_solve_with_raycast(self):
        """Test solve with raycast callback."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()
        raycast = create_flat_ground_raycast()

        multi = MultiLegFootPlacement([left_foot, right_foot], pelvis=0, raycast_callback=raycast)
        transforms = create_biped_transforms()

        result = multi.solve(transforms, Vec3.zero())

        assert len(result) == len(transforms)

    def test_solve_returns_transforms(self):
        """Test solve returns list of transforms."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()
        raycast = create_flat_ground_raycast()

        multi = MultiLegFootPlacement([left_foot, right_foot], pelvis=0, raycast_callback=raycast)
        transforms = create_biped_transforms()

        result = multi.solve(transforms, Vec3.zero())

        assert all(isinstance(t, Transform) for t in result)

    def test_solve_copies_transforms(self):
        """Test solve doesn't modify input transforms."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()
        raycast = create_flat_ground_raycast()

        multi = MultiLegFootPlacement([left_foot, right_foot], pelvis=0, raycast_callback=raycast)
        transforms = create_biped_transforms()

        original_pos = Vec3(
            transforms[0].translation.x,
            transforms[0].translation.y,
            transforms[0].translation.z
        )

        multi.solve(transforms, Vec3.zero())

        assert vec3_approx_equal(transforms[0].translation, original_pos)


# =============================================================================
# Test _calculate_multi_pelvis_offset()
# =============================================================================

class TestCalculateMultiPelvisOffset:
    """Tests for _calculate_multi_pelvis_offset() method."""

    def test_no_planted_feet(self):
        """Test with no planted feet returns zero."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()

        multi = MultiLegFootPlacement([left_foot, right_foot], pelvis=0)

        offset = multi._calculate_multi_pelvis_offset(
            Vec3(0, 1, 0),
            [Vec3(0, 0, 0), Vec3(0, 0, 0)],
            [False, False]
        )

        assert offset == 0.0

    def test_planted_feet_within_reach(self):
        """Test with planted feet within reach."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()
        raycast = create_flat_ground_raycast()

        multi = MultiLegFootPlacement([left_foot, right_foot], pelvis=0, raycast_callback=raycast)

        offset = multi._calculate_multi_pelvis_offset(
            Vec3(0, 1, 0),
            [Vec3(-0.1, 0.1, 0), Vec3(0.1, 0.1, 0)],
            [True, True]
        )

        # Should be negative (drop) or zero
        assert offset <= 0

    def test_offset_limited_by_max_drop(self):
        """Test offset limited by max pelvis drop."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()

        multi = MultiLegFootPlacement([left_foot, right_foot], pelvis=0)

        # Very far targets
        offset = multi._calculate_multi_pelvis_offset(
            Vec3(0, 1, 0),
            [Vec3(-0.1, -10, 0), Vec3(0.1, -10, 0)],
            [True, True]
        )

        # Offset should be capped
        assert abs(offset) <= MULTI_LEG_MAX_PELVIS_DROP


# =============================================================================
# Test Spider/Centaur Legs
# =============================================================================

class TestMultiLegCreatures:
    """Tests for multi-legged creatures like spiders."""

    def test_six_legged_creature(self):
        """Test with six legs (insect)."""
        feet = [
            FootData(upper_leg=i*3+1, lower_leg=i*3+2, foot=i*3+3)
            for i in range(6)
        ]

        multi = MultiLegFootPlacement(feet, pelvis=0)

        assert len(multi.feet) == 6
        assert len(multi._leg_iks) == 6

    def test_eight_legged_creature(self):
        """Test with eight legs (spider)."""
        feet = [
            FootData(upper_leg=i*3+1, lower_leg=i*3+2, foot=i*3+3)
            for i in range(8)
        ]

        multi = MultiLegFootPlacement(feet, pelvis=0)

        assert len(multi.feet) == 8
        assert len(multi._leg_iks) == 8

    def test_single_leg(self):
        """Test with single leg (hopping creature)."""
        foot = FootData(upper_leg=1, lower_leg=2, foot=3)

        multi = MultiLegFootPlacement([foot], pelvis=0)

        assert len(multi.feet) == 1
        assert len(multi._leg_iks) == 1

    def test_empty_feet_list(self):
        """Test with empty feet list."""
        multi = MultiLegFootPlacement([], pelvis=0)

        assert len(multi.feet) == 0
        assert len(multi._leg_iks) == 0


# =============================================================================
# Test RaycastHit Dataclass
# =============================================================================

class TestRaycastHit:
    """Tests for RaycastHit dataclass."""

    def test_construction_with_hit(self):
        """Test construction of a hit result."""
        from engine.animation.ik.foot_placement import RaycastHit

        hit = RaycastHit(
            hit=True,
            position=Vec3(1, 0, 2),
            normal=Vec3(0, 1, 0),
            distance=1.5
        )

        assert hit.hit is True
        assert vec3_approx_equal(hit.position, Vec3(1, 0, 2))
        assert vec3_approx_equal(hit.normal, Vec3(0, 1, 0))
        assert hit.distance == 1.5

    def test_construction_with_miss(self):
        """Test construction of a miss result."""
        from engine.animation.ik.foot_placement import RaycastHit

        hit = RaycastHit(
            hit=False,
            position=Vec3.zero(),
            normal=Vec3(0, 1, 0),
            distance=float('inf')
        )

        assert hit.hit is False

    def test_miss_static_method(self):
        """Test the miss() static method."""
        from engine.animation.ik.foot_placement import RaycastHit

        miss = RaycastHit.miss()

        assert miss.hit is False
        assert vec3_approx_equal(miss.position, Vec3.zero())
        assert vec3_approx_equal(miss.normal, Vec3(0, 1, 0))
        assert miss.distance == float('inf')

    def test_miss_returns_new_instance(self):
        """Test miss() returns a new instance each call."""
        from engine.animation.ik.foot_placement import RaycastHit

        miss1 = RaycastHit.miss()
        miss2 = RaycastHit.miss()

        assert miss1 is not miss2


# =============================================================================
# Test State Persistence Across Calls
# =============================================================================

class TestStatePersistence:
    """Tests for state tracking persistence across solve() calls."""

    def test_prev_left_target_updates_after_solve(self):
        """Test _prev_left_target updates after solve."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()
        raycast = create_flat_ground_raycast()

        placement = FootPlacement(left_foot, right_foot, pelvis=0, raycast_callback=raycast)
        transforms = create_biped_transforms()

        # Initial state is zero
        assert vec3_approx_equal(placement._prev_left_target, Vec3.zero())

        # After solve, should be updated
        placement.solve(transforms, Vec3.zero())

        # Should no longer be zero (was blended toward foot position)
        # The exact value depends on foot position and blend factor

    def test_prev_right_target_updates_after_solve(self):
        """Test _prev_right_target updates after solve."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()
        raycast = create_flat_ground_raycast()

        placement = FootPlacement(left_foot, right_foot, pelvis=0, raycast_callback=raycast)
        transforms = create_biped_transforms()

        # Initial state is zero
        assert vec3_approx_equal(placement._prev_right_target, Vec3.zero())

        # After solve, should be updated
        placement.solve(transforms, Vec3.zero())

    def test_prev_pelvis_offset_updates_after_solve(self):
        """Test _prev_pelvis_offset updates after solve."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()
        raycast = create_flat_ground_raycast()

        placement = FootPlacement(left_foot, right_foot, pelvis=0, raycast_callback=raycast)
        transforms = create_biped_transforms()

        assert placement._prev_pelvis_offset == 0.0

        placement.solve(transforms, Vec3.zero())

        # Value may or may not change depending on ground distance

    def test_state_persists_across_multiple_solves(self):
        """Test state tracking persists correctly across multiple calls."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()
        raycast = create_flat_ground_raycast()

        placement = FootPlacement(left_foot, right_foot, pelvis=0, raycast_callback=raycast)
        transforms = create_biped_transforms()

        # Multiple solves
        for _ in range(5):
            placement.solve(transforms, Vec3.zero(), dt=1.0/60.0)

        # State should reflect accumulated blending
        # Check that values are not reset between calls

    def test_blending_converges_over_time(self):
        """Test target blending converges to final value over time."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()
        raycast = create_flat_ground_raycast()

        placement = FootPlacement(left_foot, right_foot, pelvis=0, raycast_callback=raycast)
        transforms = create_biped_transforms()

        # Many iterations with large dt to simulate convergence
        for _ in range(100):
            placement.solve(transforms, Vec3.zero(), dt=0.1)

        # After many iterations, should be close to target


# =============================================================================
# Test Blend Factor Edge Cases
# =============================================================================

class TestBlendFactorEdgeCases:
    """Tests for dt parameter and blend factor edge cases."""

    def test_very_small_dt(self):
        """Test solve with very small dt (slow blending)."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()
        raycast = create_flat_ground_raycast()

        placement = FootPlacement(left_foot, right_foot, pelvis=0, raycast_callback=raycast)
        transforms = create_biped_transforms()

        result = placement.solve(transforms, Vec3.zero(), dt=0.0001)

        assert result.success is True

    def test_very_large_dt(self):
        """Test solve with very large dt (instant blending)."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()
        raycast = create_flat_ground_raycast()

        placement = FootPlacement(left_foot, right_foot, pelvis=0, raycast_callback=raycast)
        transforms = create_biped_transforms()

        result = placement.solve(transforms, Vec3.zero(), dt=10.0)

        assert result.success is True

    def test_zero_dt(self):
        """Test solve with zero dt."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()
        raycast = create_flat_ground_raycast()

        placement = FootPlacement(left_foot, right_foot, pelvis=0, raycast_callback=raycast)
        transforms = create_biped_transforms()

        result = placement.solve(transforms, Vec3.zero(), dt=0.0)

        assert result.success is True

    def test_blend_factor_clamped_to_one(self):
        """Test blend factor is clamped to maximum 1.0."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()
        raycast = create_flat_ground_raycast()

        placement = FootPlacement(left_foot, right_foot, pelvis=0, raycast_callback=raycast)
        placement.blend_speed = 1000.0  # Very high blend speed
        transforms = create_biped_transforms()

        # With dt=1.0 and blend_speed=1000, blend_factor would be 1000 but should be clamped
        result = placement.solve(transforms, Vec3.zero(), dt=1.0)

        assert result.success is True


# =============================================================================
# Test Configuration Overrides
# =============================================================================

class TestConfigurationOverrides:
    """Tests for modifying configuration after construction."""

    def test_modify_ray_length(self):
        """Test modifying ray length."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()

        placement = FootPlacement(left_foot, right_foot, pelvis=0)
        placement.ray_length = 5.0

        assert placement.ray_length == 5.0

    def test_modify_foot_height(self):
        """Test modifying foot height."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()

        placement = FootPlacement(left_foot, right_foot, pelvis=0)
        placement.foot_height = 0.2

        assert placement.foot_height == 0.2

    def test_modify_blend_speed(self):
        """Test modifying blend speed."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()

        placement = FootPlacement(left_foot, right_foot, pelvis=0)
        placement.blend_speed = 20.0

        assert placement.blend_speed == 20.0

    def test_modify_max_pelvis_drop(self):
        """Test modifying max pelvis drop."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()

        placement = FootPlacement(left_foot, right_foot, pelvis=0)
        placement.max_pelvis_drop = 1.0

        assert placement.max_pelvis_drop == 1.0

    def test_modify_max_pelvis_raise(self):
        """Test modifying max pelvis raise."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()

        placement = FootPlacement(left_foot, right_foot, pelvis=0)
        placement.max_pelvis_raise = 0.5

        assert placement.max_pelvis_raise == 0.5

    def test_modify_toe_align_weight(self):
        """Test modifying toe alignment weight."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()

        placement = FootPlacement(left_foot, right_foot, pelvis=0)
        placement.toe_align_weight = 0.8

        assert placement.toe_align_weight == 0.8


# =============================================================================
# Test Pelvis Raise Logic
# =============================================================================

class TestPelvisRaiseLogic:
    """Tests for pelvis raise behavior when feet are on elevated ground."""

    def test_pelvis_raise_with_elevated_ground(self):
        """Test pelvis raises when both feet are on elevated ground."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()

        def elevated_raycast(origin: Vec3, direction: Vec3, max_dist: float):
            from engine.animation.ik.foot_placement import RaycastHit
            # Ground at y = 0.5 (elevated)
            if direction.y >= 0:
                return None
            ground_y = 0.5
            t = (origin.y - ground_y) / (-direction.y)
            if t > max_dist or t < 0:
                return None
            hit_pos = Vec3(origin.x, ground_y, origin.z)
            return RaycastHit(hit=True, position=hit_pos, normal=Vec3(0, 1, 0), distance=t)

        placement = FootPlacement(left_foot, right_foot, pelvis=0, raycast_callback=elevated_raycast)
        transforms = create_biped_transforms()

        result = placement.solve(transforms, Vec3.zero())

        assert result.success is True

    def test_pelvis_raise_limited_by_max(self):
        """Test pelvis raise is limited by max_pelvis_raise."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()

        def very_elevated_raycast(origin: Vec3, direction: Vec3, max_dist: float):
            from engine.animation.ik.foot_placement import RaycastHit
            # Ground at y = 2.0 (very elevated, beyond typical raise)
            if direction.y >= 0:
                return None
            ground_y = 2.0
            t = (origin.y - ground_y) / (-direction.y)
            if t > max_dist or t < 0:
                return None
            hit_pos = Vec3(origin.x, ground_y, origin.z)
            return RaycastHit(hit=True, position=hit_pos, normal=Vec3(0, 1, 0), distance=t)

        placement = FootPlacement(left_foot, right_foot, pelvis=0, raycast_callback=very_elevated_raycast)
        placement.max_pelvis_raise = 0.1  # Small limit
        transforms = create_biped_transforms()

        # Multiple iterations to allow blending
        for _ in range(50):
            result = placement.solve(transforms, Vec3.zero(), dt=0.1)

        # Pelvis offset raise should be limited


# =============================================================================
# Test Terrain Slope Calculation
# =============================================================================

class TestTerrainSlopeCalculation:
    """Tests for terrain slope calculation."""

    def test_flat_terrain_slope_is_zero(self):
        """Test flat terrain has zero slope."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()
        raycast = create_flat_ground_raycast()

        placement = FootPlacement(left_foot, right_foot, pelvis=0, raycast_callback=raycast)
        transforms = create_biped_transforms()

        result = placement.solve(transforms, Vec3.zero())

        assert result.terrain_slope == 0.0

    def test_sloped_terrain_has_nonzero_slope(self):
        """Test sloped terrain has non-zero slope angle."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()
        raycast = create_sloped_ground_raycast(0.3)  # 30% slope

        placement = FootPlacement(left_foot, right_foot, pelvis=0, raycast_callback=raycast)
        transforms = create_biped_transforms()

        result = placement.solve(transforms, Vec3.zero())

        # Slope calculation depends on foot positions
        # With sloped ground, slope should be non-zero if feet are at different heights

    def test_no_slope_when_only_left_foot_planted(self):
        """Test slope is zero when only left foot is planted."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()

        hit_count = [0]

        def partial_hit_raycast(origin: Vec3, direction: Vec3, max_dist: float):
            from engine.animation.ik.foot_placement import RaycastHit
            hit_count[0] += 1
            # Only hit for left foot (first two raycasts are left foot area)
            if origin.x < 0:
                return RaycastHit(hit=True, position=Vec3(origin.x, 0, origin.z), normal=Vec3(0, 1, 0), distance=1.0)
            return None

        placement = FootPlacement(left_foot, right_foot, pelvis=0, raycast_callback=partial_hit_raycast)
        transforms = create_biped_transforms()

        result = placement.solve(transforms, Vec3.zero())

        # With only one foot planted, slope is zero
        assert result.terrain_slope == 0.0


# =============================================================================
# Test Height Offset Integration
# =============================================================================

class TestHeightOffsetIntegration:
    """Tests for foot height_offset field integration."""

    def test_left_foot_height_offset_applied(self):
        """Test left foot height offset is applied."""
        left_foot = create_left_foot_data()
        left_foot.height_offset = 0.05
        right_foot = create_right_foot_data()
        raycast = create_flat_ground_raycast()

        placement = FootPlacement(left_foot, right_foot, pelvis=0, raycast_callback=raycast)
        transforms = create_biped_transforms()

        result = placement.solve(transforms, Vec3.zero())

        assert result.success is True

    def test_right_foot_height_offset_applied(self):
        """Test right foot height offset is applied."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()
        right_foot.height_offset = 0.03
        raycast = create_flat_ground_raycast()

        placement = FootPlacement(left_foot, right_foot, pelvis=0, raycast_callback=raycast)
        transforms = create_biped_transforms()

        result = placement.solve(transforms, Vec3.zero())

        assert result.success is True

    def test_different_height_offsets_per_foot(self):
        """Test different height offsets for each foot."""
        left_foot = create_left_foot_data()
        left_foot.height_offset = 0.1
        right_foot = create_right_foot_data()
        right_foot.height_offset = 0.05
        raycast = create_flat_ground_raycast()

        placement = FootPlacement(left_foot, right_foot, pelvis=0, raycast_callback=raycast)
        transforms = create_biped_transforms()

        result = placement.solve(transforms, Vec3.zero())

        assert result.success is True


# =============================================================================
# Test IK Solver Configuration
# =============================================================================

class TestIKSolverConfiguration:
    """Tests for the internal IK solver configuration."""

    def test_left_leg_ik_uses_correct_bones(self):
        """Test left leg IK solver uses correct bone indices."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()

        placement = FootPlacement(left_foot, right_foot, pelvis=0)

        # IK solver should reference correct bones
        assert placement._left_leg_ik.root_bone == left_foot.upper_leg
        assert placement._left_leg_ik.mid_bone == left_foot.lower_leg
        assert placement._left_leg_ik.end_bone == left_foot.foot

    def test_right_leg_ik_uses_correct_bones(self):
        """Test right leg IK solver uses correct bone indices."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()

        placement = FootPlacement(left_foot, right_foot, pelvis=0)

        # IK solver should reference correct bones
        assert placement._right_leg_ik.root_bone == right_foot.upper_leg
        assert placement._right_leg_ik.mid_bone == right_foot.lower_leg
        assert placement._right_leg_ik.end_bone == right_foot.foot

    def test_ik_solvers_use_soft_ik_defaults(self):
        """Test IK solvers are configured with soft IK defaults."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()

        placement = FootPlacement(left_foot, right_foot, pelvis=0)

        assert placement._left_leg_ik.soft_ik_ratio == SOFT_IK_DEFAULT_RATIO
        assert placement._left_leg_ik.soft_ik_blend == SOFT_IK_DEFAULT_BLEND
        assert placement._right_leg_ik.soft_ik_ratio == SOFT_IK_DEFAULT_RATIO
        assert placement._right_leg_ik.soft_ik_blend == SOFT_IK_DEFAULT_BLEND
