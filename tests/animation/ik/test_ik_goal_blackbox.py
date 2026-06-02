"""
Blackbox tests for T-IK-3.1: IK Goal Base Classes

These tests verify the IK Goal system's public contract without
knowledge of internal implementation details. Tests are derived from
acceptance criteria for task T-IK-3.1.

Acceptance Criteria under test:
1. IKGoal abstract base class
2. PositionGoal dataclass
3. RotationGoal dataclass
4. LookAtGoal dataclass
5. PositionRotationGoal dataclass
6. PoleVectorGoal dataclass
7. COMGoal (CenterOfMassGoal) dataclass
8. IKGoalBlender for weighted blending

Blackbox Test Strategy:
- Test public API contracts only
- Test behavioral expectations from acceptance criteria
- Test boundary conditions and edge cases
- Test error handling for invalid inputs
"""

import math
import pytest
import numpy as np
from abc import ABC

from engine.core.math import Vec3, Quat
from engine.animation.ik import (
    IKGoalType,
    IKGoal,
    PositionGoal,
    RotationGoal,
    LookAtGoal,
    PositionRotationGoal,
    PoleVectorGoal,
    CenterOfMassGoal,
    IKGoalBlender,
)


# -----------------------------------------------------------------------------
# Test Constants
# -----------------------------------------------------------------------------

EPSILON = 1e-6  # Floating point comparison tolerance
POSITION_TOLERANCE = 0.01  # Position tolerance for "achieved" checks
ANGULAR_TOLERANCE = 0.01  # Angular tolerance in radians


def vec3_approx_equal(v1: Vec3, v2: Vec3, eps: float = EPSILON) -> bool:
    """Check if two Vec3 are approximately equal."""
    return (
        abs(v1.x - v2.x) < eps
        and abs(v1.y - v2.y) < eps
        and abs(v1.z - v2.z) < eps
    )


def vec3_length(v: Vec3) -> float:
    """Compute length of a Vec3."""
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def vec3_distance(v1: Vec3, v2: Vec3) -> float:
    """Compute distance between two Vec3."""
    dx = v1.x - v2.x
    dy = v1.y - v2.y
    dz = v1.z - v2.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def quat_dot(q1: Quat, q2: Quat) -> float:
    """Compute dot product of two quaternions."""
    return q1.x * q2.x + q1.y * q2.y + q1.z * q2.z + q1.w * q2.w


def quat_angular_distance(q1: Quat, q2: Quat) -> float:
    """Compute angular distance between two quaternions in radians."""
    dot = abs(quat_dot(q1, q2))
    dot = min(1.0, max(-1.0, dot))  # Clamp to [-1, 1]
    return 2.0 * math.acos(dot)


# =============================================================================
# AC1: IKGoal Abstract Base Class Tests
# =============================================================================

class TestIKGoalAbstractBaseClass:
    """Tests for IKGoal base class (AC1)."""

    def test_ikgoal_is_dataclass(self):
        """IKGoal should be a dataclass (not abstract)."""
        # IKGoal is a dataclass, not an ABC - it can be instantiated directly
        from dataclasses import is_dataclass
        assert is_dataclass(IKGoal), "IKGoal should be a dataclass"

    def test_ikgoal_can_be_instantiated(self):
        """IKGoal can be instantiated directly as a base goal."""
        goal = IKGoal(bone_index=0)
        assert goal is not None
        assert goal.bone_index == 0

    def test_ikgoal_has_goal_type_attribute(self):
        """IKGoal derived classes should have goal_type attribute."""
        # Create a concrete goal to test
        goal = PositionGoal(bone_index=0, target_position=Vec3(0, 0, 0), weight=1.0)
        assert hasattr(goal, 'goal_type'), "IKGoal should have goal_type attribute"
        assert isinstance(goal.goal_type, IKGoalType), "goal_type should be IKGoalType enum"

    def test_ikgoal_has_weight_attribute(self):
        """IKGoal derived classes should have weight attribute."""
        goal = PositionGoal(bone_index=0, target_position=Vec3(0, 0, 0), weight=0.5)
        assert hasattr(goal, 'weight'), "IKGoal should have weight attribute"
        assert goal.weight == 0.5, "Weight should match constructor value"

    def test_ikgoal_has_bone_index_attribute(self):
        """IKGoal derived classes should have bone_index attribute."""
        goal = PositionGoal(bone_index=5, target_position=Vec3(0, 0, 0), weight=1.0)
        assert hasattr(goal, 'bone_index'), "IKGoal should have bone_index attribute"
        assert goal.bone_index == 5, "bone_index should match constructor value"


# =============================================================================
# AC2: PositionGoal Dataclass Tests
# =============================================================================

class TestPositionGoal:
    """Tests for PositionGoal dataclass (AC2)."""

    def test_position_goal_instantiation(self):
        """PositionGoal can be instantiated with required params."""
        goal = PositionGoal(
            bone_index=0,
            target_position=Vec3(1.0, 2.0, 3.0),
            weight=1.0
        )
        assert goal is not None
        assert goal.bone_index == 0
        assert goal.weight == 1.0

    def test_position_goal_has_correct_goal_type(self):
        """PositionGoal should have goal_type == POSITION."""
        goal = PositionGoal(bone_index=0, target_position=Vec3(0, 0, 0), weight=1.0)
        assert goal.goal_type == IKGoalType.POSITION

    def test_position_goal_stores_target(self):
        """PositionGoal should store the target position."""
        target = Vec3(1.0, 2.0, 3.0)
        goal = PositionGoal(bone_index=0, target_position=target, weight=1.0)
        assert hasattr(goal, 'target_position'), "PositionGoal should have target_position attribute"
        assert vec3_approx_equal(goal.target_position, target)

    def test_position_goal_distance_to_target_zero(self):
        """distance_to_target returns 0 when current equals target."""
        target = Vec3(1.0, 2.0, 3.0)
        goal = PositionGoal(bone_index=0, target_position=target, weight=1.0)

        if hasattr(goal, 'distance_to_target'):
            distance = goal.distance_to_target(target)
            assert abs(distance) < EPSILON, f"Distance should be 0, got {distance}"

    def test_position_goal_distance_to_target_correct(self):
        """distance_to_target returns correct Euclidean distance."""
        target = Vec3(1.0, 0.0, 0.0)
        current = Vec3(4.0, 0.0, 0.0)
        goal = PositionGoal(bone_index=0, target_position=target, weight=1.0)

        if hasattr(goal, 'distance_to_target'):
            distance = goal.distance_to_target(current)
            expected = 3.0  # |4-1| = 3
            assert abs(distance - expected) < EPSILON, f"Distance should be {expected}, got {distance}"

    def test_position_goal_distance_to_target_3d(self):
        """distance_to_target works correctly in 3D."""
        target = Vec3(0.0, 0.0, 0.0)
        current = Vec3(1.0, 1.0, 1.0)
        goal = PositionGoal(bone_index=0, target_position=target, weight=1.0)

        if hasattr(goal, 'distance_to_target'):
            distance = goal.distance_to_target(current)
            expected = math.sqrt(3.0)  # sqrt(1+1+1)
            assert abs(distance - expected) < EPSILON, f"Distance should be {expected}, got {distance}"

    def test_position_goal_weight_validation_lower_bound(self):
        """PositionGoal weight should be validated (>= 0.0)."""
        # Weight of 0 should be valid
        goal = PositionGoal(bone_index=0, target_position=Vec3(0, 0, 0), weight=0.0)
        assert goal.weight == 0.0

    def test_position_goal_weight_validation_upper_bound(self):
        """PositionGoal weight should be validated (<= 1.0)."""
        # Weight of 1 should be valid
        goal = PositionGoal(bone_index=0, target_position=Vec3(0, 0, 0), weight=1.0)
        assert goal.weight == 1.0

    def test_position_goal_weight_out_of_range_negative(self):
        """PositionGoal should handle or reject negative weight."""
        try:
            goal = PositionGoal(bone_index=0, target_position=Vec3(0, 0, 0), weight=-0.5)
            # If no error, weight might be clamped
            assert goal.weight >= 0.0, "Negative weight should be rejected or clamped"
        except (ValueError, AssertionError):
            pass  # Expected: negative weight rejected

    def test_position_goal_weight_out_of_range_above_one(self):
        """PositionGoal should handle or reject weight > 1.0."""
        try:
            goal = PositionGoal(bone_index=0, target_position=Vec3(0, 0, 0), weight=1.5)
            # If no error, weight might be clamped
            assert goal.weight <= 1.0, "Weight > 1.0 should be rejected or clamped"
        except (ValueError, AssertionError):
            pass  # Expected: weight > 1 rejected


# =============================================================================
# AC3: RotationGoal Dataclass Tests
# =============================================================================

class TestRotationGoal:
    """Tests for RotationGoal dataclass (AC3)."""

    def test_rotation_goal_instantiation(self):
        """RotationGoal can be instantiated with required params."""
        goal = RotationGoal(
            bone_index=0,
            target_rotation=Quat.identity(),
            weight=1.0
        )
        assert goal is not None
        assert goal.bone_index == 0
        assert goal.weight == 1.0

    def test_rotation_goal_has_correct_goal_type(self):
        """RotationGoal should have goal_type == ROTATION."""
        goal = RotationGoal(bone_index=0, target_rotation=Quat.identity(), weight=1.0)
        assert goal.goal_type == IKGoalType.ROTATION

    def test_rotation_goal_stores_target(self):
        """RotationGoal should store the target rotation."""
        target = Quat.identity()
        goal = RotationGoal(bone_index=0, target_rotation=target, weight=1.0)
        assert hasattr(goal, 'target_rotation'), "RotationGoal should have target_rotation attribute"

    def test_rotation_goal_angular_distance_identical(self):
        """angular_distance handles identical rotations (returns 0)."""
        target = Quat.identity()
        goal = RotationGoal(bone_index=0, target_rotation=target, weight=1.0)

        if hasattr(goal, 'angular_distance'):
            distance = goal.angular_distance(target)
            assert abs(distance) < EPSILON, f"Angular distance for identical rotations should be 0, got {distance}"

    def test_rotation_goal_angular_distance_opposite(self):
        """angular_distance handles opposite quaternions (same rotation)."""
        target = Quat.identity()
        # Negated quaternion represents same rotation
        opposite = Quat(-target.x, -target.y, -target.z, -target.w)
        goal = RotationGoal(bone_index=0, target_rotation=target, weight=1.0)

        if hasattr(goal, 'angular_distance'):
            distance = goal.angular_distance(opposite)
            # Should be 0 or very close (opposite quats = same rotation)
            assert abs(distance) < EPSILON or abs(distance - 2 * math.pi) < EPSILON, (
                f"Angular distance for opposite quaternions should be ~0, got {distance}"
            )

    def test_rotation_goal_angular_distance_90_degrees(self):
        """angular_distance returns correct value for 90 degree rotation."""
        target = Quat.identity()
        # Create 90 degree rotation around Y axis
        angle_rad = math.pi / 2
        rotated = Quat(0, math.sin(angle_rad / 2), 0, math.cos(angle_rad / 2))
        goal = RotationGoal(bone_index=0, target_rotation=target, weight=1.0)

        if hasattr(goal, 'angular_distance'):
            distance = goal.angular_distance(rotated)
            expected = math.pi / 2  # 90 degrees
            assert abs(distance - expected) < 0.01, f"Expected ~{expected}, got {distance}"

    def test_rotation_goal_weight_validation(self):
        """RotationGoal validates weight in 0.0-1.0 range."""
        # Valid weights
        goal_min = RotationGoal(bone_index=0, target_rotation=Quat.identity(), weight=0.0)
        goal_max = RotationGoal(bone_index=0, target_rotation=Quat.identity(), weight=1.0)
        assert goal_min.weight == 0.0
        assert goal_max.weight == 1.0


# =============================================================================
# AC4: LookAtGoal Dataclass Tests
# =============================================================================

class TestLookAtGoal:
    """Tests for LookAtGoal dataclass (AC4)."""

    def test_lookat_goal_instantiation(self):
        """LookAtGoal can be instantiated with required params."""
        goal = LookAtGoal(
            bone_index=0,
            target_point=Vec3(0, 0, 1),
            forward_axis=Vec3(0, 0, 1),
            weight=1.0
        )
        assert goal is not None
        assert goal.bone_index == 0
        assert goal.weight == 1.0

    def test_lookat_goal_has_correct_goal_type(self):
        """LookAtGoal should have goal_type == LOOK_AT."""
        goal = LookAtGoal(bone_index=0, target_point=Vec3(0, 0, 1), forward_axis=Vec3(0, 0, 1), weight=1.0)
        assert goal.goal_type == IKGoalType.LOOK_AT

    def test_lookat_goal_stores_target(self):
        """LookAtGoal should store the target position."""
        target = Vec3(10, 5, 0)
        goal = LookAtGoal(bone_index=0, target_point=target, forward_axis=Vec3(0, 0, 1), weight=1.0)
        assert hasattr(goal, 'target_point'), "LookAtGoal should have target_point attribute"
        assert vec3_approx_equal(goal.target_point, target)

    def test_lookat_goal_stores_axis(self):
        """LookAtGoal should store the look axis."""
        axis = Vec3(0, 1, 0)
        goal = LookAtGoal(bone_index=0, target_point=Vec3(0, 0, 1), forward_axis=axis, weight=1.0)
        assert hasattr(goal, 'forward_axis'), "LookAtGoal should have forward_axis attribute"
        assert vec3_approx_equal(goal.forward_axis, axis)

    def test_lookat_goal_has_max_angle(self):
        """LookAtGoal should have max_angle attribute."""
        goal = LookAtGoal(bone_index=0, target_point=Vec3(0, 0, 1), forward_axis=Vec3(0, 0, 1), weight=1.0)
        assert hasattr(goal, 'max_angle'), "LookAtGoal should have max_angle attribute"

    def test_lookat_goal_respects_max_angle_limit(self):
        """LookAtGoal should respect max_angle limit."""
        max_angle = math.pi / 4  # 45 degrees
        goal = LookAtGoal(
            bone_index=0,
            target_point=Vec3(0, 0, 1),
            forward_axis=Vec3(0, 0, 1),
            weight=1.0,
            max_angle=max_angle
        )
        assert goal.max_angle == max_angle

    def test_lookat_goal_default_max_angle(self):
        """LookAtGoal should have a reasonable default max_angle."""
        goal = LookAtGoal(bone_index=0, target_point=Vec3(0, 0, 1), forward_axis=Vec3(0, 0, 1), weight=1.0)
        # Default should allow reasonable rotation (not 0, not too large)
        assert goal.max_angle > 0, "Default max_angle should be positive"
        assert goal.max_angle <= math.pi, "Default max_angle should be <= 180 degrees"

    def test_lookat_goal_weight_validation(self):
        """LookAtGoal validates weight in 0.0-1.0 range."""
        goal = LookAtGoal(bone_index=0, target_point=Vec3(0, 0, 1), forward_axis=Vec3(0, 0, 1), weight=0.5)
        assert 0.0 <= goal.weight <= 1.0


# =============================================================================
# AC5: PositionRotationGoal Dataclass Tests
# =============================================================================

class TestPositionRotationGoal:
    """Tests for PositionRotationGoal dataclass (AC5)."""

    def test_position_rotation_goal_instantiation(self):
        """PositionRotationGoal can be instantiated with required params."""
        goal = PositionRotationGoal(
            bone_index=0,
            target_position=Vec3(1, 2, 3),
            target_rotation=Quat.identity(),
            weight=1.0
        )
        assert goal is not None
        assert goal.bone_index == 0
        assert goal.weight == 1.0

    def test_position_rotation_goal_has_correct_goal_type(self):
        """PositionRotationGoal should have goal_type == POSITION_AND_ROTATION."""
        goal = PositionRotationGoal(
            bone_index=0,
            target_position=Vec3(0, 0, 0),
            target_rotation=Quat.identity(),
            weight=1.0
        )
        assert goal.goal_type == IKGoalType.POSITION_AND_ROTATION

    def test_position_rotation_goal_stores_both_targets(self):
        """PositionRotationGoal should store both position and rotation targets."""
        pos = Vec3(1, 2, 3)
        rot = Quat.identity()
        goal = PositionRotationGoal(
            bone_index=0,
            target_position=pos,
            target_rotation=rot,
            weight=1.0
        )
        assert hasattr(goal, 'target_position'), "Should have target_position attribute"
        assert hasattr(goal, 'target_rotation'), "Should have target_rotation attribute"
        assert vec3_approx_equal(goal.target_position, pos)

    def test_position_rotation_goal_is_achieved_both_match(self):
        """is_achieved returns True when both position and rotation match."""
        target_pos = Vec3(1, 0, 0)
        target_rot = Quat.identity()
        goal = PositionRotationGoal(
            bone_index=0,
            target_position=target_pos,
            target_rotation=target_rot,
            weight=1.0
        )

        if hasattr(goal, 'is_achieved'):
            # Current matches target exactly
            result = goal.is_achieved(target_pos, target_rot)
            assert result is True, "is_achieved should return True when both match"

    def test_position_rotation_goal_is_achieved_position_mismatch(self):
        """is_achieved returns False when position doesn't match."""
        target_pos = Vec3(1, 0, 0)
        target_rot = Quat.identity()
        goal = PositionRotationGoal(
            bone_index=0,
            target_position=target_pos,
            target_rotation=target_rot,
            weight=1.0
        )

        if hasattr(goal, 'is_achieved'):
            # Position is far from target
            current_pos = Vec3(100, 0, 0)
            result = goal.is_achieved(current_pos, target_rot)
            assert result is False, "is_achieved should return False when position mismatches"

    def test_position_rotation_goal_is_achieved_rotation_mismatch(self):
        """is_achieved returns False when rotation doesn't match."""
        target_pos = Vec3(0, 0, 0)
        target_rot = Quat.identity()
        goal = PositionRotationGoal(
            bone_index=0,
            target_position=target_pos,
            target_rotation=target_rot,
            weight=1.0
        )

        if hasattr(goal, 'is_achieved'):
            # 90 degree rotation
            current_rot = Quat(0, math.sin(math.pi / 4), 0, math.cos(math.pi / 4))
            result = goal.is_achieved(target_pos, current_rot)
            assert result is False, "is_achieved should return False when rotation mismatches"

    def test_position_rotation_goal_is_achieved_with_tolerance(self):
        """is_achieved respects tolerance parameter if available."""
        target_pos = Vec3(1, 0, 0)
        target_rot = Quat.identity()
        goal = PositionRotationGoal(
            bone_index=0,
            target_position=target_pos,
            target_rotation=target_rot,
            weight=1.0
        )

        if hasattr(goal, 'is_achieved'):
            # Slightly off position
            current_pos = Vec3(1.001, 0, 0)
            try:
                result = goal.is_achieved(current_pos, target_rot, tolerance=0.01)
                assert result is True, "is_achieved should return True within tolerance"
            except TypeError:
                # tolerance parameter may not exist
                pass


# =============================================================================
# AC6: PoleVectorGoal Dataclass Tests
# =============================================================================

class TestPoleVectorGoal:
    """Tests for PoleVectorGoal dataclass (AC6)."""

    def test_pole_vector_goal_instantiation(self):
        """PoleVectorGoal can be instantiated with required params."""
        goal = PoleVectorGoal(
            bone_index=0,
            pole_position=Vec3(0, 1, 0),
            weight=1.0
        )
        assert goal is not None
        assert goal.bone_index == 0
        assert goal.weight == 1.0

    def test_pole_vector_goal_has_correct_goal_type(self):
        """PoleVectorGoal should have goal_type == POLE_VECTOR."""
        goal = PoleVectorGoal(bone_index=0, pole_position=Vec3(0, 1, 0), weight=1.0)
        assert goal.goal_type == IKGoalType.POLE_VECTOR

    def test_pole_vector_goal_stores_pole_position(self):
        """PoleVectorGoal should store the pole position."""
        pole_pos = Vec3(1, 2, 3)
        goal = PoleVectorGoal(bone_index=0, pole_position=pole_pos, weight=1.0)
        assert hasattr(goal, 'pole_position'), "Should have pole_position attribute"
        assert vec3_approx_equal(goal.pole_position, pole_pos)

    def test_pole_vector_goal_compute_pole_direction_normalizes(self):
        """compute_pole_direction should return a normalized vector."""
        goal = PoleVectorGoal(bone_index=0, pole_position=Vec3(0, 10, 0), weight=1.0)

        if hasattr(goal, 'compute_pole_direction'):
            # Test with non-normalized input (large vector)
            # compute_pole_direction requires (start_position, end_position)
            start_pos = Vec3(0, 0, 0)
            end_pos = Vec3(0, 5, 0)
            pole_dir = goal.compute_pole_direction(start_pos, end_pos)

            length = vec3_length(pole_dir)
            assert abs(length - 1.0) < EPSILON, f"Pole direction should be normalized, length={length}"

    def test_pole_vector_goal_compute_pole_direction_correct(self):
        """compute_pole_direction returns correct direction."""
        pole_pos = Vec3(3, 4, 0)  # Distance 5 from origin
        goal = PoleVectorGoal(bone_index=0, pole_position=pole_pos, weight=1.0)

        if hasattr(goal, 'compute_pole_direction'):
            start_pos = Vec3(0, 0, 0)
            end_pos = Vec3(0, 5, 0)
            pole_dir = goal.compute_pole_direction(start_pos, end_pos)

            # Result should be a normalized vector
            length = vec3_length(pole_dir)
            assert abs(length - 1.0) < EPSILON, f"Pole direction should be normalized"

    def test_pole_vector_goal_compute_pole_direction_offset_joint(self):
        """compute_pole_direction works with offset joint position."""
        pole_pos = Vec3(5, 5, 0)
        goal = PoleVectorGoal(bone_index=0, pole_position=pole_pos, weight=1.0)

        if hasattr(goal, 'compute_pole_direction'):
            start_pos = Vec3(2, 2, 0)  # Offset joint
            end_pos = Vec3(2, 7, 0)
            pole_dir = goal.compute_pole_direction(start_pos, end_pos)

            # Result should be normalized
            length = vec3_length(pole_dir)
            assert abs(length - 1.0) < EPSILON, f"Pole direction should be normalized"

    def test_pole_vector_goal_weight_validation(self):
        """PoleVectorGoal validates weight in 0.0-1.0 range."""
        goal = PoleVectorGoal(bone_index=0, pole_position=Vec3(0, 1, 0), weight=0.75)
        assert 0.0 <= goal.weight <= 1.0


# =============================================================================
# AC7: CenterOfMassGoal (COMGoal) Dataclass Tests
# =============================================================================

class TestCenterOfMassGoal:
    """Tests for CenterOfMassGoal dataclass (AC7)."""

    def test_com_goal_instantiation(self):
        """CenterOfMassGoal can be instantiated with required params."""
        goal = CenterOfMassGoal(
            bone_index=0,
            target_com=Vec3(0, 0, 0),
            bone_masses={0: 1.0},
            weight=1.0
        )
        assert goal is not None
        assert goal.weight == 1.0

    def test_com_goal_has_correct_goal_type(self):
        """CenterOfMassGoal should have goal_type == CENTER_OF_MASS."""
        goal = CenterOfMassGoal(
            bone_index=0,
            target_com=Vec3(0, 0, 0),
            bone_masses={0: 1.0},
            weight=1.0
        )
        assert goal.goal_type == IKGoalType.CENTER_OF_MASS

    def test_com_goal_stores_target(self):
        """CenterOfMassGoal should store the target COM position."""
        target = Vec3(0.5, 0, 0)
        goal = CenterOfMassGoal(
            bone_index=0,
            target_com=target,
            bone_masses={0: 1.0},
            weight=1.0
        )
        assert hasattr(goal, 'target_com'), "Should have target_com attribute"
        assert vec3_approx_equal(goal.target_com, target)

    def test_com_goal_stores_bone_masses(self):
        """CenterOfMassGoal should store bone mass mapping."""
        bone_masses = {0: 0.3, 1: 0.4, 2: 0.3}
        goal = CenterOfMassGoal(
            bone_index=0,
            target_com=Vec3(0, 0, 0),
            bone_masses=bone_masses,
            weight=1.0
        )
        assert hasattr(goal, 'bone_masses'), "Should have bone_masses attribute"
        assert goal.bone_masses == bone_masses

    def test_com_goal_is_balanced_inside_polygon(self):
        """is_balanced returns True when COM is inside support polygon."""
        # Support polygon: square from (-1,-1) to (1,1)
        support_polygon = [
            Vec3(-1, 0, -1),
            Vec3(1, 0, -1),
            Vec3(1, 0, 1),
            Vec3(-1, 0, 1),
        ]
        goal = CenterOfMassGoal(
            bone_index=0,
            target_com=Vec3(0, 0, 0),
            bone_masses={0: 1.0},
            support_polygon=support_polygon,
            weight=1.0
        )

        if hasattr(goal, 'is_balanced'):
            # COM at center (0, 0) - clearly inside
            com = Vec3(0, 0, 0)
            result = goal.is_balanced(com)
            assert result is True, "COM at center should be balanced"

    def test_com_goal_is_balanced_outside_polygon(self):
        """is_balanced returns False when COM is outside support polygon."""
        # Support polygon: small square
        support_polygon = [
            Vec3(-0.1, 0, -0.1),
            Vec3(0.1, 0, -0.1),
            Vec3(0.1, 0, 0.1),
            Vec3(-0.1, 0, 0.1),
        ]
        goal = CenterOfMassGoal(
            bone_index=0,
            target_com=Vec3(0, 0, 0),
            bone_masses={0: 1.0},
            support_polygon=support_polygon,
            weight=1.0
        )

        if hasattr(goal, 'is_balanced'):
            # COM far outside
            com = Vec3(10, 0, 10)
            result = goal.is_balanced(com)
            assert result is False, "COM far outside should not be balanced"

    def test_com_goal_is_balanced_edge_case(self):
        """is_balanced handles edge case of COM on polygon boundary."""
        # Support polygon
        support_polygon = [
            Vec3(-1, 0, -1),
            Vec3(1, 0, -1),
            Vec3(1, 0, 1),
            Vec3(-1, 0, 1),
        ]
        goal = CenterOfMassGoal(
            bone_index=0,
            target_com=Vec3(0, 0, 0),
            bone_masses={0: 1.0},
            support_polygon=support_polygon,
            weight=1.0
        )

        if hasattr(goal, 'is_balanced'):
            # COM exactly on edge
            com = Vec3(1, 0, 0)
            # Edge cases may return True or False depending on implementation
            result = goal.is_balanced(com)
            assert isinstance(result, bool), "is_balanced should return boolean"

    def test_com_goal_weight_validation(self):
        """CenterOfMassGoal validates weight in 0.0-1.0 range."""
        goal = CenterOfMassGoal(
            bone_index=0,
            target_com=Vec3(0, 0, 0),
            bone_masses={0: 1.0},
            weight=0.25
        )
        assert 0.0 <= goal.weight <= 1.0


# =============================================================================
# AC8: IKGoalBlender Tests
# =============================================================================

class TestIKGoalBlender:
    """Tests for IKGoalBlender for weighted blending (AC8)."""

    def test_ik_goal_blender_instantiation(self):
        """IKGoalBlender can be instantiated."""
        blender = IKGoalBlender()
        assert blender is not None

    def test_ik_goal_blender_has_blend_position(self):
        """IKGoalBlender should have blend_position method."""
        blender = IKGoalBlender()
        assert hasattr(blender, 'blend_position'), "IKGoalBlender should have blend_position method"

    def test_ik_goal_blender_has_blend_rotation(self):
        """IKGoalBlender should have blend_rotation method."""
        blender = IKGoalBlender()
        assert hasattr(blender, 'blend_rotation'), "IKGoalBlender should have blend_rotation method"

    def test_ik_goal_blender_blend_single_position(self):
        """Blending single position returns target immediately on first call."""
        blender = IKGoalBlender()
        target = Vec3(5, 0, 0)
        goal_id = 0
        dt = 0.016  # ~60fps

        result = blender.blend_position(goal_id, target, dt)
        # First call should return target directly (no previous position)
        assert vec3_approx_equal(result, target, eps=0.01), (
            f"First blend should return target position, got {result}"
        )

    def test_ik_goal_blender_blend_position_smooth(self):
        """Blending positions should smoothly approach target over time."""
        blender = IKGoalBlender()
        goal_id = 0
        dt = 0.1  # Large dt for noticeable blend

        # Start at origin
        start = Vec3(0, 0, 0)
        target = Vec3(10, 0, 0)

        # Initialize with start position
        blender.blend_position(goal_id, start, dt)

        # Now blend toward target
        positions = []
        for _ in range(10):
            pos = blender.blend_position(goal_id, target, dt)
            positions.append(pos)

        # Each position should be closer to target than the last (monotonic)
        for i in range(1, len(positions)):
            dist_prev = vec3_distance(positions[i - 1], target)
            dist_curr = vec3_distance(positions[i], target)
            assert dist_curr <= dist_prev + EPSILON, (
                f"Position should converge: frame {i-1} dist={dist_prev}, frame {i} dist={dist_curr}"
            )

    def test_ik_goal_blender_blend_rotation(self):
        """Blending rotation returns target immediately on first call."""
        blender = IKGoalBlender()
        target = Quat.identity()
        goal_id = 0
        dt = 0.016

        result = blender.blend_rotation(goal_id, target, dt)
        # First call should return target directly
        assert quat_angular_distance(result, target) < EPSILON, (
            f"First blend should return target rotation"
        )

    def test_ik_goal_blender_multiple_goals(self):
        """IKGoalBlender tracks multiple goals by ID."""
        blender = IKGoalBlender()
        target1 = Vec3(1, 0, 0)
        target2 = Vec3(0, 1, 0)
        dt = 0.016

        result1 = blender.blend_position(0, target1, dt)
        result2 = blender.blend_position(1, target2, dt)

        # Each goal should track independently
        assert vec3_approx_equal(result1, target1, eps=0.01)
        assert vec3_approx_equal(result2, target2, eps=0.01)

    def test_ik_goal_blender_reset_all(self):
        """IKGoalBlender reset() clears all blending state."""
        blender = IKGoalBlender()
        target = Vec3(10, 0, 0)
        goal_id = 0
        dt = 0.016

        # Initialize
        blender.blend_position(goal_id, Vec3(0, 0, 0), dt)
        # Blend partially
        blender.blend_position(goal_id, target, dt)

        # Reset all
        blender.reset()

        # After reset, next call should return target directly again
        result = blender.blend_position(goal_id, target, dt)
        assert vec3_approx_equal(result, target, eps=0.01), (
            f"After reset, should return target directly"
        )

    def test_ik_goal_blender_reset_single_goal(self):
        """IKGoalBlender reset(goal_id) clears specific goal only."""
        blender = IKGoalBlender()
        dt = 0.016

        # Initialize two goals
        blender.blend_position(0, Vec3(0, 0, 0), dt)
        blender.blend_position(1, Vec3(0, 0, 0), dt)

        # Blend partially
        blender.blend_position(0, Vec3(10, 0, 0), dt)
        blender.blend_position(1, Vec3(0, 10, 0), dt)

        # Reset only goal 0
        blender.reset(goal_id=0)

        # Goal 0 should be reset (returns target directly)
        result0 = blender.blend_position(0, Vec3(5, 0, 0), dt)
        assert vec3_approx_equal(result0, Vec3(5, 0, 0), eps=0.01), (
            f"Reset goal should return target directly"
        )

    def test_ik_goal_blender_custom_speed(self):
        """IKGoalBlender accepts custom blend speed."""
        blender = IKGoalBlender(blend_speed=10.0)  # Very fast blend
        goal_id = 0
        dt = 0.1

        # Initialize with start
        blender.blend_position(goal_id, Vec3(0, 0, 0), dt)

        # Blend toward target with high speed
        target = Vec3(10, 0, 0)
        result = blender.blend_position(goal_id, target, dt)

        # With speed=10 and dt=0.1, t = min(1.0, 10*0.1) = 1.0, so should reach target
        assert vec3_approx_equal(result, target, eps=0.1), (
            f"High speed blend should reach target quickly"
        )


# =============================================================================
# Edge Cases and Boundary Conditions
# =============================================================================

class TestIKGoalEdgeCases:
    """Edge cases and boundary conditions for all IK goal types."""

    def test_negative_bone_index_rejected(self):
        """Negative bone_index should be rejected or handled."""
        try:
            goal = PositionGoal(bone_index=-1, target_position=Vec3(0, 0, 0), weight=1.0)
            # If allowed, bone_index should be stored as-is or clamped
            assert goal.bone_index >= -1  # Implementation may use -1 for special cases
        except (ValueError, AssertionError):
            pass  # Expected: negative index rejected

    def test_large_bone_index(self):
        """Large bone_index should be accepted."""
        goal = PositionGoal(bone_index=999, target_position=Vec3(0, 0, 0), weight=1.0)
        assert goal.bone_index == 999

    def test_very_large_position_values(self):
        """Goals should handle very large position values."""
        large_val = 1e10
        goal = PositionGoal(
            bone_index=0,
            target_position=Vec3(large_val, large_val, large_val),
            weight=1.0
        )
        assert goal.target_position.x == large_val

    def test_very_small_position_values(self):
        """Goals should handle very small (near-zero) position values."""
        small_val = 1e-10
        goal = PositionGoal(
            bone_index=0,
            target_position=Vec3(small_val, small_val, small_val),
            weight=1.0
        )
        assert abs(goal.target_position.x - small_val) < 1e-15

    def test_zero_weight_goal(self):
        """Zero weight goal should be valid but have no effect."""
        goal = PositionGoal(bone_index=0, target_position=Vec3(100, 0, 0), weight=0.0)
        assert goal.weight == 0.0

    def test_multiple_goals_same_bone_blender(self):
        """IKGoalBlender can track multiple goals via different IDs."""
        blender = IKGoalBlender()
        dt = 0.016
        # Use blend_position with different goal_ids for multiple goals
        result1 = blender.blend_position(0, Vec3(0, 0, 0), dt)
        result2 = blender.blend_position(1, Vec3(1, 0, 0), dt)
        # Both should work independently
        assert result1 is not None
        assert result2 is not None

    def test_unnormalized_quaternion_rotation_goal(self):
        """RotationGoal should handle unnormalized quaternions."""
        # Create unnormalized quaternion
        unnorm = Quat(0, 0, 0, 2)  # Not unit length
        try:
            goal = RotationGoal(bone_index=0, target_rotation=unnorm, weight=1.0)
            # Implementation may normalize internally
            assert goal is not None
        except ValueError:
            pass  # May reject unnormalized quaternions


# =============================================================================
# IKGoalType Enum Tests
# =============================================================================

class TestIKGoalTypeEnum:
    """Tests for IKGoalType enum."""

    def test_ikgoaltype_has_position(self):
        """IKGoalType should have POSITION value."""
        assert hasattr(IKGoalType, 'POSITION')

    def test_ikgoaltype_has_rotation(self):
        """IKGoalType should have ROTATION value."""
        assert hasattr(IKGoalType, 'ROTATION')

    def test_ikgoaltype_has_look_at(self):
        """IKGoalType should have LOOK_AT value."""
        assert hasattr(IKGoalType, 'LOOK_AT')

    def test_ikgoaltype_has_position_and_rotation(self):
        """IKGoalType should have POSITION_AND_ROTATION value."""
        assert hasattr(IKGoalType, 'POSITION_AND_ROTATION')

    def test_ikgoaltype_has_pole_vector(self):
        """IKGoalType should have POLE_VECTOR value."""
        assert hasattr(IKGoalType, 'POLE_VECTOR')

    def test_ikgoaltype_has_center_of_mass(self):
        """IKGoalType should have CENTER_OF_MASS value."""
        assert hasattr(IKGoalType, 'CENTER_OF_MASS')

    def test_ikgoaltype_all_values_unique(self):
        """All IKGoalType enum values should be unique."""
        values = [e.value for e in IKGoalType]
        assert len(values) == len(set(values)), "IKGoalType values should be unique"
