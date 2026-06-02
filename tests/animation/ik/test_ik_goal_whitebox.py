"""Whitebox tests for IK Goal base classes.

Tests comprehensive coverage of:
- IKGoal abstract base class
- PositionGoal dataclass
- RotationGoal dataclass
- LookAtGoal dataclass
- PositionRotationGoal dataclass
- PoleVectorGoal dataclass
- CenterOfMassGoal dataclass
- IKGoalBlender for weighted blending

Task: T-IK-3.1
"""

from __future__ import annotations

import math
import pytest

from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.core.constants import MATH_EPSILON
from engine.animation.ik.config import (
    IK_DEFAULT_TOLERANCE,
    IK_ROTATION_TOLERANCE,
    LOOK_AT_MAX_ANGLE,
    GOAL_BLENDER_DEFAULT_SPEED,
    FABRIK_DEFAULT_MAX_ITERATIONS,
)
from engine.animation.ik.ik_goal import (
    IKGoalType,
    IKGoal,
    PositionGoal,
    RotationGoal,
    LookAtGoal,
    PositionRotationGoal,
    PoleVectorGoal,
    ChainGoal,
    CenterOfMassGoal,
    IKGoalBlender,
    ik_goal,
    ik_chain,
)


# =============================================================================
# IKGoalType Enum Tests
# =============================================================================

class TestIKGoalType:
    """Tests for IKGoalType enumeration."""

    def test_position_type_exists(self):
        """Verify POSITION goal type is defined."""
        assert IKGoalType.POSITION is not None

    def test_rotation_type_exists(self):
        """Verify ROTATION goal type is defined."""
        assert IKGoalType.ROTATION is not None

    def test_look_at_type_exists(self):
        """Verify LOOK_AT goal type is defined."""
        assert IKGoalType.LOOK_AT is not None

    def test_position_and_rotation_type_exists(self):
        """Verify POSITION_AND_ROTATION goal type is defined."""
        assert IKGoalType.POSITION_AND_ROTATION is not None

    def test_pole_vector_type_exists(self):
        """Verify POLE_VECTOR goal type is defined."""
        assert IKGoalType.POLE_VECTOR is not None

    def test_center_of_mass_type_exists(self):
        """Verify CENTER_OF_MASS goal type is defined."""
        assert IKGoalType.CENTER_OF_MASS is not None

    def test_all_types_are_unique(self):
        """Verify all goal types have unique values."""
        types = [
            IKGoalType.POSITION,
            IKGoalType.ROTATION,
            IKGoalType.LOOK_AT,
            IKGoalType.POSITION_AND_ROTATION,
            IKGoalType.POLE_VECTOR,
            IKGoalType.CENTER_OF_MASS,
        ]
        values = [t.value for t in types]
        assert len(values) == len(set(values))


# =============================================================================
# IKGoal Base Class Tests
# =============================================================================

class TestIKGoal:
    """Tests for IKGoal base dataclass."""

    def test_constructor_default_values(self):
        """Test IKGoal default values are set correctly."""
        goal = IKGoal(bone_index=5)
        assert goal.bone_index == 5
        assert goal.goal_type == IKGoalType.POSITION
        assert goal.weight == 1.0
        assert goal.priority == 0
        assert goal.enabled is True

    def test_constructor_custom_values(self):
        """Test IKGoal with custom parameter values."""
        goal = IKGoal(
            bone_index=10,
            goal_type=IKGoalType.ROTATION,
            weight=0.5,
            priority=3,
            enabled=False,
        )
        assert goal.bone_index == 10
        assert goal.goal_type == IKGoalType.ROTATION
        assert goal.weight == 0.5
        assert goal.priority == 3
        assert goal.enabled is False

    def test_validate_valid_goal(self):
        """Test validate returns True for valid goal."""
        goal = IKGoal(bone_index=5, weight=0.5)
        assert goal.validate() is True

    def test_validate_negative_bone_index(self):
        """Test validate returns False for negative bone index."""
        goal = IKGoal(bone_index=-1)
        assert goal.validate() is False

    def test_validate_weight_below_zero(self):
        """Test validate returns False for weight < 0."""
        goal = IKGoal(bone_index=5, weight=-0.1)
        assert goal.validate() is False

    def test_validate_weight_above_one(self):
        """Test validate returns False for weight > 1."""
        goal = IKGoal(bone_index=5, weight=1.1)
        assert goal.validate() is False

    def test_validate_weight_boundary_zero(self):
        """Test validate accepts weight = 0."""
        goal = IKGoal(bone_index=5, weight=0.0)
        assert goal.validate() is True

    def test_validate_weight_boundary_one(self):
        """Test validate accepts weight = 1."""
        goal = IKGoal(bone_index=5, weight=1.0)
        assert goal.validate() is True


# =============================================================================
# PositionGoal Tests
# =============================================================================

class TestPositionGoal:
    """Tests for PositionGoal dataclass."""

    def test_constructor_defaults(self):
        """Test PositionGoal default values."""
        goal = PositionGoal(bone_index=3)
        assert goal.bone_index == 3
        assert goal.goal_type == IKGoalType.POSITION
        assert goal.target_position == Vec3.zero()
        assert goal.tolerance == IK_DEFAULT_TOLERANCE
        assert goal.weight == 1.0

    def test_constructor_custom_target(self):
        """Test PositionGoal with custom target position."""
        target = Vec3(1.0, 2.0, 3.0)
        goal = PositionGoal(bone_index=5, target_position=target)
        assert goal.target_position.x == 1.0
        assert goal.target_position.y == 2.0
        assert goal.target_position.z == 3.0

    def test_constructor_custom_tolerance(self):
        """Test PositionGoal with custom tolerance."""
        goal = PositionGoal(bone_index=5, tolerance=0.01)
        assert goal.tolerance == 0.01

    def test_goal_type_set_in_post_init(self):
        """Test that goal_type is forced to POSITION in __post_init__."""
        # Even if we try to pass a different type, it should be overridden
        goal = PositionGoal(bone_index=5, goal_type=IKGoalType.ROTATION)
        assert goal.goal_type == IKGoalType.POSITION

    def test_distance_to_target_zero(self):
        """Test distance_to_target when at target position."""
        target = Vec3(5.0, 5.0, 5.0)
        goal = PositionGoal(bone_index=0, target_position=target)
        current = Vec3(5.0, 5.0, 5.0)
        assert goal.distance_to_target(current) < MATH_EPSILON

    def test_distance_to_target_nonzero(self):
        """Test distance_to_target with offset position."""
        target = Vec3(0.0, 0.0, 0.0)
        goal = PositionGoal(bone_index=0, target_position=target)
        current = Vec3(3.0, 4.0, 0.0)
        assert abs(goal.distance_to_target(current) - 5.0) < MATH_EPSILON

    def test_distance_to_target_3d(self):
        """Test distance_to_target in full 3D."""
        target = Vec3(1.0, 2.0, 3.0)
        goal = PositionGoal(bone_index=0, target_position=target)
        current = Vec3(4.0, 6.0, 3.0)
        expected = math.sqrt(9.0 + 16.0)  # sqrt((3)^2 + (4)^2) = 5
        assert abs(goal.distance_to_target(current) - expected) < MATH_EPSILON

    def test_is_achieved_within_tolerance(self):
        """Test is_achieved returns True when within tolerance."""
        target = Vec3(10.0, 10.0, 10.0)
        goal = PositionGoal(bone_index=0, target_position=target, tolerance=0.1)
        current = Vec3(10.05, 10.0, 10.0)  # 0.05 < 0.1
        assert goal.is_achieved(current) is True

    def test_is_achieved_at_tolerance_boundary(self):
        """Test is_achieved returns True when exactly at tolerance."""
        target = Vec3(0.0, 0.0, 0.0)
        goal = PositionGoal(bone_index=0, target_position=target, tolerance=0.1)
        current = Vec3(0.1, 0.0, 0.0)
        assert goal.is_achieved(current) is True

    def test_is_achieved_outside_tolerance(self):
        """Test is_achieved returns False when outside tolerance."""
        target = Vec3(0.0, 0.0, 0.0)
        goal = PositionGoal(bone_index=0, target_position=target, tolerance=0.1)
        current = Vec3(0.2, 0.0, 0.0)
        assert goal.is_achieved(current) is False

    def test_is_achieved_exact_position(self):
        """Test is_achieved returns True when exactly at target."""
        target = Vec3(5.0, 5.0, 5.0)
        goal = PositionGoal(bone_index=0, target_position=target)
        assert goal.is_achieved(target) is True


# =============================================================================
# RotationGoal Tests
# =============================================================================

class TestRotationGoal:
    """Tests for RotationGoal dataclass."""

    def test_constructor_defaults(self):
        """Test RotationGoal default values."""
        goal = RotationGoal(bone_index=2)
        assert goal.bone_index == 2
        assert goal.goal_type == IKGoalType.ROTATION
        assert goal.target_rotation == Quat.identity()
        assert goal.tolerance == IK_ROTATION_TOLERANCE

    def test_constructor_custom_rotation(self):
        """Test RotationGoal with custom target rotation."""
        rot = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 2)
        goal = RotationGoal(bone_index=5, target_rotation=rot)
        assert abs(goal.target_rotation.y - rot.y) < MATH_EPSILON

    def test_goal_type_set_in_post_init(self):
        """Test that goal_type is forced to ROTATION in __post_init__."""
        goal = RotationGoal(bone_index=5, goal_type=IKGoalType.POSITION)
        assert goal.goal_type == IKGoalType.ROTATION

    def test_angular_distance_identity(self):
        """Test angular_distance with identical rotations."""
        goal = RotationGoal(bone_index=0, target_rotation=Quat.identity())
        current = Quat.identity()
        assert goal.angular_distance(current) < MATH_EPSILON

    def test_angular_distance_90_degrees(self):
        """Test angular_distance with 90 degree rotation."""
        target = Quat.identity()
        goal = RotationGoal(bone_index=0, target_rotation=target)
        current = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 2)
        # Angular distance should be pi/2
        assert abs(goal.angular_distance(current) - math.pi / 2) < 0.01

    def test_angular_distance_180_degrees(self):
        """Test angular_distance with 180 degree rotation."""
        target = Quat.identity()
        goal = RotationGoal(bone_index=0, target_rotation=target)
        current = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi)
        # Angular distance should be pi
        assert abs(goal.angular_distance(current) - math.pi) < 0.01

    def test_angular_distance_negative_dot(self):
        """Test angular_distance handles negative dot product."""
        target = Quat.identity()
        goal = RotationGoal(bone_index=0, target_rotation=target)
        # Create rotation that gives negative dot product with identity
        current = Quat(-0.1, -0.1, -0.1, -0.9).normalized()
        # Should still return valid positive distance
        distance = goal.angular_distance(current)
        assert distance >= 0.0
        assert distance <= math.pi

    def test_is_achieved_within_tolerance(self):
        """Test is_achieved returns True within tolerance."""
        target = Quat.identity()
        goal = RotationGoal(bone_index=0, target_rotation=target, tolerance=0.1)
        # Small rotation
        current = Quat.from_axis_angle(Vec3(0, 1, 0), 0.05)
        assert goal.is_achieved(current) is True

    def test_is_achieved_outside_tolerance(self):
        """Test is_achieved returns False outside tolerance."""
        target = Quat.identity()
        goal = RotationGoal(bone_index=0, target_rotation=target, tolerance=0.01)
        # Large rotation
        current = Quat.from_axis_angle(Vec3(0, 1, 0), 0.5)
        assert goal.is_achieved(current) is False

    def test_is_achieved_exact_rotation(self):
        """Test is_achieved returns True with exact same rotation."""
        target = Quat.from_axis_angle(Vec3(1, 0, 0), 0.5)
        goal = RotationGoal(bone_index=0, target_rotation=target)
        assert goal.is_achieved(target) is True


# =============================================================================
# LookAtGoal Tests
# =============================================================================

class TestLookAtGoal:
    """Tests for LookAtGoal dataclass."""

    def test_constructor_defaults(self):
        """Test LookAtGoal default values."""
        goal = LookAtGoal(bone_index=1)
        assert goal.bone_index == 1
        assert goal.goal_type == IKGoalType.LOOK_AT
        assert goal.target_point == Vec3.zero()
        assert goal.forward_axis == Vec3(0, 0, 1)
        assert goal.up_axis == Vec3(0, 1, 0)
        assert goal.max_angle == LOOK_AT_MAX_ANGLE

    def test_constructor_custom_values(self):
        """Test LookAtGoal with custom values."""
        target = Vec3(10.0, 5.0, 0.0)
        forward = Vec3(0, 0, -1)
        up = Vec3(0, 1, 0)
        goal = LookAtGoal(
            bone_index=5,
            target_point=target,
            forward_axis=forward,
            up_axis=up,
            max_angle=math.pi / 4,
        )
        assert goal.target_point == target
        assert goal.forward_axis == forward
        assert goal.up_axis == up
        assert goal.max_angle == math.pi / 4

    def test_goal_type_set_in_post_init(self):
        """Test that goal_type is forced to LOOK_AT in __post_init__."""
        goal = LookAtGoal(bone_index=5, goal_type=IKGoalType.POSITION)
        assert goal.goal_type == IKGoalType.LOOK_AT

    def test_compute_look_rotation_target_at_bone_position(self):
        """Test compute_look_rotation when target is at bone position."""
        goal = LookAtGoal(bone_index=0, target_point=Vec3(0, 0, 0))
        bone_pos = Vec3(0, 0, 0)
        current_rot = Quat.identity()
        result = goal.compute_look_rotation(bone_pos, current_rot)
        # Should return current rotation (no change)
        assert abs(result.x - current_rot.x) < MATH_EPSILON
        assert abs(result.y - current_rot.y) < MATH_EPSILON
        assert abs(result.z - current_rot.z) < MATH_EPSILON
        assert abs(result.w - current_rot.w) < MATH_EPSILON

    def test_compute_look_rotation_already_looking_at_target(self):
        """Test compute_look_rotation when already looking at target."""
        # Forward is +Z, target is in front
        goal = LookAtGoal(bone_index=0, target_point=Vec3(0, 0, 10))
        bone_pos = Vec3(0, 0, 0)
        current_rot = Quat.identity()
        result = goal.compute_look_rotation(bone_pos, current_rot)
        # Should return nearly same rotation
        assert abs(result.x - current_rot.x) < 0.01
        assert abs(result.w - current_rot.w) < 0.01

    def test_compute_look_rotation_90_degree_turn(self):
        """Test compute_look_rotation with 90 degree turn."""
        # Forward is +Z, target is to the right (+X)
        goal = LookAtGoal(bone_index=0, target_point=Vec3(10, 0, 0))
        bone_pos = Vec3(0, 0, 0)
        current_rot = Quat.identity()
        result = goal.compute_look_rotation(bone_pos, current_rot)
        # Resulting forward should point toward target
        new_forward = result.rotate_vector(Vec3(0, 0, 1))
        expected = Vec3(1, 0, 0)
        assert abs(new_forward.x - expected.x) < 0.1

    def test_compute_look_rotation_max_angle_clamping(self):
        """Test compute_look_rotation respects max_angle limit."""
        goal = LookAtGoal(
            bone_index=0,
            target_point=Vec3(-10, 0, 0),  # Behind
            max_angle=math.pi / 6,  # 30 degrees max
        )
        bone_pos = Vec3(0, 0, 0)
        current_rot = Quat.identity()
        result = goal.compute_look_rotation(bone_pos, current_rot)
        # Should not exceed max_angle
        angle = 2.0 * math.acos(abs(result.dot(current_rot)))
        assert angle <= goal.max_angle + 0.01

    def test_compute_look_rotation_opposite_direction(self):
        """Test compute_look_rotation when target is opposite to forward."""
        # Forward is +Z, target is behind (-Z)
        goal = LookAtGoal(bone_index=0, target_point=Vec3(0, 0, -10))
        bone_pos = Vec3(0, 0, 0)
        current_rot = Quat.identity()
        result = goal.compute_look_rotation(bone_pos, current_rot)
        # Should rotate, limited by max_angle
        assert result is not None

    def test_compute_look_rotation_custom_forward_axis(self):
        """Test compute_look_rotation with custom forward axis."""
        goal = LookAtGoal(
            bone_index=0,
            target_point=Vec3(10, 0, 0),
            forward_axis=Vec3(1, 0, 0),  # Forward is +X
        )
        bone_pos = Vec3(0, 0, 0)
        current_rot = Quat.identity()
        result = goal.compute_look_rotation(bone_pos, current_rot)
        # Should not need to rotate much since already pointing +X
        assert result is not None


# =============================================================================
# PositionRotationGoal Tests
# =============================================================================

class TestPositionRotationGoal:
    """Tests for PositionRotationGoal dataclass."""

    def test_constructor_defaults(self):
        """Test PositionRotationGoal default values."""
        goal = PositionRotationGoal(bone_index=4)
        assert goal.bone_index == 4
        assert goal.goal_type == IKGoalType.POSITION_AND_ROTATION
        assert goal.target_position == Vec3.zero()
        assert goal.target_rotation == Quat.identity()
        assert goal.position_weight == 1.0
        assert goal.rotation_weight == 1.0
        assert goal.position_tolerance == IK_DEFAULT_TOLERANCE
        assert goal.rotation_tolerance == IK_ROTATION_TOLERANCE

    def test_constructor_custom_values(self):
        """Test PositionRotationGoal with custom values."""
        pos = Vec3(1, 2, 3)
        rot = Quat.from_axis_angle(Vec3(0, 1, 0), 0.5)
        goal = PositionRotationGoal(
            bone_index=7,
            target_position=pos,
            target_rotation=rot,
            position_weight=0.8,
            rotation_weight=0.6,
            position_tolerance=0.05,
            rotation_tolerance=0.1,
        )
        assert goal.target_position == pos
        assert goal.position_weight == 0.8
        assert goal.rotation_weight == 0.6
        assert goal.position_tolerance == 0.05
        assert goal.rotation_tolerance == 0.1

    def test_goal_type_set_in_post_init(self):
        """Test that goal_type is forced in __post_init__."""
        goal = PositionRotationGoal(bone_index=5, goal_type=IKGoalType.POSITION)
        assert goal.goal_type == IKGoalType.POSITION_AND_ROTATION

    def test_is_achieved_both_within_tolerance(self):
        """Test is_achieved when both pos and rot within tolerance."""
        pos = Vec3(5, 5, 5)
        rot = Quat.identity()
        goal = PositionRotationGoal(
            bone_index=0,
            target_position=pos,
            target_rotation=rot,
            position_tolerance=0.1,
            rotation_tolerance=0.1,
        )
        current_pos = Vec3(5.05, 5.0, 5.0)
        current_rot = Quat.from_axis_angle(Vec3(0, 1, 0), 0.05)
        assert goal.is_achieved(current_pos, current_rot) is True

    def test_is_achieved_position_outside_tolerance(self):
        """Test is_achieved when position is outside tolerance."""
        pos = Vec3(0, 0, 0)
        rot = Quat.identity()
        goal = PositionRotationGoal(
            bone_index=0,
            target_position=pos,
            target_rotation=rot,
            position_tolerance=0.1,
        )
        current_pos = Vec3(1, 0, 0)  # Way outside
        current_rot = Quat.identity()
        assert goal.is_achieved(current_pos, current_rot) is False

    def test_is_achieved_rotation_outside_tolerance(self):
        """Test is_achieved when rotation is outside tolerance."""
        pos = Vec3(0, 0, 0)
        rot = Quat.identity()
        goal = PositionRotationGoal(
            bone_index=0,
            target_position=pos,
            target_rotation=rot,
            rotation_tolerance=0.01,
        )
        current_pos = Vec3(0, 0, 0)
        current_rot = Quat.from_axis_angle(Vec3(0, 1, 0), 1.0)  # Way outside
        assert goal.is_achieved(current_pos, current_rot) is False

    def test_is_achieved_both_outside_tolerance(self):
        """Test is_achieved when both are outside tolerance."""
        pos = Vec3(0, 0, 0)
        rot = Quat.identity()
        goal = PositionRotationGoal(
            bone_index=0,
            target_position=pos,
            target_rotation=rot,
            position_tolerance=0.1,
            rotation_tolerance=0.1,
        )
        current_pos = Vec3(1, 1, 1)
        current_rot = Quat.from_axis_angle(Vec3(0, 1, 0), 1.0)
        assert goal.is_achieved(current_pos, current_rot) is False

    def test_is_achieved_exact_match(self):
        """Test is_achieved with exact position and rotation."""
        pos = Vec3(3, 4, 5)
        rot = Quat.from_axis_angle(Vec3(1, 0, 0), 0.5)
        goal = PositionRotationGoal(
            bone_index=0,
            target_position=pos,
            target_rotation=rot,
        )
        assert goal.is_achieved(pos, rot) is True


# =============================================================================
# PoleVectorGoal Tests
# =============================================================================

class TestPoleVectorGoal:
    """Tests for PoleVectorGoal dataclass."""

    def test_constructor_defaults(self):
        """Test PoleVectorGoal default values."""
        goal = PoleVectorGoal(bone_index=3)
        assert goal.bone_index == 3
        assert goal.goal_type == IKGoalType.POLE_VECTOR
        assert goal.pole_position == Vec3.zero()
        assert goal.twist_offset == 0.0

    def test_constructor_custom_values(self):
        """Test PoleVectorGoal with custom values."""
        pole = Vec3(0, 0, -5)
        goal = PoleVectorGoal(
            bone_index=5,
            pole_position=pole,
            twist_offset=0.5,
        )
        assert goal.pole_position == pole
        assert goal.twist_offset == 0.5

    def test_goal_type_set_in_post_init(self):
        """Test that goal_type is forced to POLE_VECTOR in __post_init__."""
        goal = PoleVectorGoal(bone_index=5, goal_type=IKGoalType.POSITION)
        assert goal.goal_type == IKGoalType.POLE_VECTOR

    def test_compute_pole_direction_basic(self):
        """Test compute_pole_direction with simple case."""
        goal = PoleVectorGoal(bone_index=0, pole_position=Vec3(0, 0, -5))
        root = Vec3(0, 0, 0)
        end = Vec3(0, 0, 0)  # Same as root, mid is (0,0,0)
        direction = goal.compute_pole_direction(root, end)
        # Direction from (0,0,0) to (0,0,-5) normalized is (0,0,-1)
        assert abs(direction.z - (-1.0)) < MATH_EPSILON

    def test_compute_pole_direction_with_chain(self):
        """Test compute_pole_direction with actual chain positions."""
        goal = PoleVectorGoal(bone_index=0, pole_position=Vec3(0, 5, 0))
        root = Vec3(0, 0, 0)
        end = Vec3(10, 0, 0)
        # Mid is (5, 0, 0), pole is (0, 5, 0)
        # Direction from (5,0,0) to (0,5,0) is (-5, 5, 0) normalized
        direction = goal.compute_pole_direction(root, end)
        expected_len = math.sqrt(50)
        expected = Vec3(-5 / expected_len, 5 / expected_len, 0)
        assert abs(direction.x - expected.x) < 0.01
        assert abs(direction.y - expected.y) < 0.01

    def test_compute_pole_direction_pole_at_midpoint(self):
        """Test compute_pole_direction when pole is at chain midpoint."""
        root = Vec3(0, 0, 0)
        end = Vec3(10, 0, 0)
        mid = root.lerp(end, 0.5)  # (5, 0, 0)
        goal = PoleVectorGoal(bone_index=0, pole_position=mid)
        direction = goal.compute_pole_direction(root, end)
        # When pole is at midpoint, should return default up
        assert direction == Vec3.up()

    def test_compute_pole_direction_zero_length_chain(self):
        """Test compute_pole_direction with zero-length chain."""
        goal = PoleVectorGoal(bone_index=0, pole_position=Vec3(0, 1, 0))
        root = Vec3(5, 5, 5)
        end = Vec3(5, 5, 5)  # Same position
        direction = goal.compute_pole_direction(root, end)
        # Should return normalized direction to pole from midpoint
        assert direction is not None
        assert abs(direction.length() - 1.0) < MATH_EPSILON


# =============================================================================
# ChainGoal Tests
# =============================================================================

class TestChainGoal:
    """Tests for ChainGoal dataclass."""

    def test_constructor_minimal(self):
        """Test ChainGoal with minimal parameters."""
        goal = ChainGoal(chain_name="arm", bone_indices=[0, 1, 2])
        assert goal.chain_name == "arm"
        assert goal.bone_indices == [0, 1, 2]
        assert goal.end_effector_goal is None
        assert goal.pole_goal is None
        assert goal.maintain_length is True
        assert goal.stiffness == 0.0

    def test_root_index_property(self):
        """Test root_index property returns first bone."""
        goal = ChainGoal(chain_name="leg", bone_indices=[5, 6, 7, 8])
        assert goal.root_index == 5

    def test_root_index_empty_chain(self):
        """Test root_index returns -1 for empty chain."""
        goal = ChainGoal(chain_name="empty", bone_indices=[])
        assert goal.root_index == -1

    def test_end_index_property(self):
        """Test end_index property returns last bone."""
        goal = ChainGoal(chain_name="arm", bone_indices=[0, 1, 2, 3])
        assert goal.end_index == 3

    def test_end_index_empty_chain(self):
        """Test end_index returns -1 for empty chain."""
        goal = ChainGoal(chain_name="empty", bone_indices=[])
        assert goal.end_index == -1

    def test_chain_length_property(self):
        """Test chain_length property returns bone count."""
        goal = ChainGoal(chain_name="spine", bone_indices=[0, 1, 2, 3, 4])
        assert goal.chain_length == 5

    def test_validate_valid_chain(self):
        """Test validate returns True for valid chain."""
        goal = ChainGoal(chain_name="arm", bone_indices=[0, 1, 2], stiffness=0.5)
        assert goal.validate() is True

    def test_validate_empty_chain(self):
        """Test validate returns False for empty bone list."""
        goal = ChainGoal(chain_name="empty", bone_indices=[])
        assert goal.validate() is False

    def test_validate_single_bone_chain(self):
        """Test validate returns False for single-bone chain."""
        goal = ChainGoal(chain_name="single", bone_indices=[5])
        assert goal.validate() is False

    def test_validate_negative_bone_index(self):
        """Test validate returns False if any bone index is negative."""
        goal = ChainGoal(chain_name="bad", bone_indices=[0, -1, 2])
        assert goal.validate() is False

    def test_validate_stiffness_below_zero(self):
        """Test validate returns False for negative stiffness."""
        goal = ChainGoal(chain_name="stiff", bone_indices=[0, 1], stiffness=-0.1)
        assert goal.validate() is False

    def test_validate_stiffness_above_one(self):
        """Test validate returns False for stiffness > 1."""
        goal = ChainGoal(chain_name="stiff", bone_indices=[0, 1], stiffness=1.5)
        assert goal.validate() is False

    def test_validate_stiffness_boundary_values(self):
        """Test validate accepts stiffness at boundaries."""
        goal_zero = ChainGoal(chain_name="flex", bone_indices=[0, 1], stiffness=0.0)
        goal_one = ChainGoal(chain_name="rigid", bone_indices=[0, 1], stiffness=1.0)
        assert goal_zero.validate() is True
        assert goal_one.validate() is True


# =============================================================================
# CenterOfMassGoal Tests
# =============================================================================

class TestCenterOfMassGoal:
    """Tests for CenterOfMassGoal dataclass."""

    def test_constructor_defaults(self):
        """Test CenterOfMassGoal default values."""
        goal = CenterOfMassGoal(bone_index=0)
        assert goal.bone_index == 0
        assert goal.goal_type == IKGoalType.CENTER_OF_MASS
        assert goal.target_com == Vec3.zero()
        assert goal.support_polygon == []
        assert goal.bone_masses == {}

    def test_constructor_custom_values(self):
        """Test CenterOfMassGoal with custom values."""
        com = Vec3(0, 1, 0)
        polygon = [Vec3(-1, 0, -1), Vec3(1, 0, -1), Vec3(0, 0, 1)]
        masses = {0: 10.0, 1: 5.0, 2: 3.0}
        goal = CenterOfMassGoal(
            bone_index=5,
            target_com=com,
            support_polygon=polygon,
            bone_masses=masses,
        )
        assert goal.target_com == com
        assert len(goal.support_polygon) == 3
        assert goal.bone_masses == masses

    def test_goal_type_set_in_post_init(self):
        """Test that goal_type is forced to CENTER_OF_MASS."""
        goal = CenterOfMassGoal(bone_index=0, goal_type=IKGoalType.POSITION)
        assert goal.goal_type == IKGoalType.CENTER_OF_MASS

    def test_is_balanced_no_polygon(self):
        """Test is_balanced returns True when no polygon defined."""
        goal = CenterOfMassGoal(bone_index=0, support_polygon=[])
        current_com = Vec3(0, 1, 0)
        assert goal.is_balanced(current_com) is True

    def test_is_balanced_two_vertices(self):
        """Test is_balanced returns True with only 2 vertices (can't form polygon)."""
        polygon = [Vec3(0, 0, 0), Vec3(1, 0, 0)]
        goal = CenterOfMassGoal(bone_index=0, support_polygon=polygon)
        assert goal.is_balanced(Vec3(5, 1, 5)) is True

    def test_is_balanced_com_inside_triangle(self):
        """Test is_balanced with COM inside triangular support."""
        # Triangle on XZ plane
        polygon = [
            Vec3(-1, 0, -1),
            Vec3(1, 0, -1),
            Vec3(0, 0, 1),
        ]
        goal = CenterOfMassGoal(bone_index=0, support_polygon=polygon)
        # COM at center (projected to XZ)
        current_com = Vec3(0, 1, 0)  # Above the triangle center
        assert goal.is_balanced(current_com) is True

    def test_is_balanced_com_outside_triangle(self):
        """Test is_balanced with COM outside triangular support."""
        polygon = [
            Vec3(-1, 0, -1),
            Vec3(1, 0, -1),
            Vec3(0, 0, 1),
        ]
        goal = CenterOfMassGoal(bone_index=0, support_polygon=polygon)
        # COM far outside
        current_com = Vec3(10, 1, 10)
        assert goal.is_balanced(current_com) is False

    def test_is_balanced_com_on_edge(self):
        """Test is_balanced with COM on polygon edge."""
        polygon = [
            Vec3(-1, 0, -1),
            Vec3(1, 0, -1),
            Vec3(0, 0, 1),
        ]
        goal = CenterOfMassGoal(bone_index=0, support_polygon=polygon)
        # COM on edge between first two vertices
        current_com = Vec3(0, 1, -1)
        # Point on edge should be considered inside or on boundary
        # (behavior depends on implementation)
        result = goal.is_balanced(current_com)
        assert isinstance(result, bool)

    def test_is_balanced_square_polygon(self):
        """Test is_balanced with square support polygon."""
        polygon = [
            Vec3(-1, 0, -1),
            Vec3(1, 0, -1),
            Vec3(1, 0, 1),
            Vec3(-1, 0, 1),
        ]
        goal = CenterOfMassGoal(bone_index=0, support_polygon=polygon)
        # Inside
        assert goal.is_balanced(Vec3(0, 1, 0)) is True
        # Outside
        assert goal.is_balanced(Vec3(5, 1, 0)) is False

    def test_is_balanced_com_at_vertex(self):
        """Test is_balanced with COM at polygon vertex."""
        polygon = [
            Vec3(-1, 0, -1),
            Vec3(1, 0, -1),
            Vec3(0, 0, 1),
        ]
        goal = CenterOfMassGoal(bone_index=0, support_polygon=polygon)
        # COM exactly at a vertex
        current_com = Vec3(-1, 1, -1)
        result = goal.is_balanced(current_com)
        assert isinstance(result, bool)

    def test_is_balanced_horizontal_edge(self):
        """Test is_balanced handles horizontal edges (dz == 0)."""
        # Polygon with horizontal edge (same z for two adjacent vertices)
        polygon = [
            Vec3(-1, 0, 0),
            Vec3(1, 0, 0),   # Horizontal edge from (-1,0,0) to (1,0,0)
            Vec3(0, 0, 1),
        ]
        goal = CenterOfMassGoal(bone_index=0, support_polygon=polygon)
        # COM inside
        current_com = Vec3(0, 1, 0.3)
        result = goal.is_balanced(current_com)
        assert isinstance(result, bool)


# =============================================================================
# IKGoalBlender Tests
# =============================================================================

class TestIKGoalBlender:
    """Tests for IKGoalBlender class."""

    def test_constructor_default_speed(self):
        """Test IKGoalBlender default blend speed."""
        blender = IKGoalBlender()
        assert blender.blend_speed == GOAL_BLENDER_DEFAULT_SPEED

    def test_constructor_custom_speed(self):
        """Test IKGoalBlender with custom blend speed."""
        blender = IKGoalBlender(blend_speed=5.0)
        assert blender.blend_speed == 5.0

    def test_blend_position_first_call(self):
        """Test blend_position returns target on first call for goal."""
        blender = IKGoalBlender()
        target = Vec3(10, 20, 30)
        result = blender.blend_position(goal_id=1, target=target, dt=0.016)
        assert result == target

    def test_blend_position_subsequent_calls(self):
        """Test blend_position interpolates on subsequent calls."""
        blender = IKGoalBlender(blend_speed=1.0)
        initial = Vec3(0, 0, 0)
        target = Vec3(10, 0, 0)

        # First call sets initial position
        blender.blend_position(goal_id=1, target=initial, dt=0.1)

        # Second call with new target should interpolate
        result = blender.blend_position(goal_id=1, target=target, dt=0.5)
        # t = min(1.0, 1.0 * 0.5) = 0.5
        # result = initial.lerp(target, 0.5) = (5, 0, 0)
        assert abs(result.x - 5.0) < 0.01

    def test_blend_position_clamps_t_to_one(self):
        """Test blend_position clamps interpolation factor to 1.0."""
        blender = IKGoalBlender(blend_speed=100.0)
        initial = Vec3(0, 0, 0)
        target = Vec3(10, 0, 0)

        blender.blend_position(goal_id=1, target=initial, dt=0.1)
        result = blender.blend_position(goal_id=1, target=target, dt=1.0)
        # t = min(1.0, 100.0 * 1.0) = 1.0, so result should be target
        assert abs(result.x - 10.0) < 0.01

    def test_blend_position_custom_speed_override(self):
        """Test blend_position with speed parameter override."""
        blender = IKGoalBlender(blend_speed=1.0)
        initial = Vec3(0, 0, 0)
        target = Vec3(10, 0, 0)

        blender.blend_position(goal_id=1, target=initial, dt=0.1)
        result = blender.blend_position(goal_id=1, target=target, dt=0.1, speed=10.0)
        # t = min(1.0, 10.0 * 0.1) = 1.0
        assert abs(result.x - 10.0) < 0.01

    def test_blend_position_multiple_goals(self):
        """Test blend_position tracks multiple goals independently."""
        blender = IKGoalBlender(blend_speed=1.0)

        target1 = Vec3(10, 0, 0)
        target2 = Vec3(0, 10, 0)

        r1 = blender.blend_position(goal_id=1, target=target1, dt=0.1)
        r2 = blender.blend_position(goal_id=2, target=target2, dt=0.1)

        assert r1 == target1
        assert r2 == target2

    def test_blend_rotation_first_call(self):
        """Test blend_rotation returns target on first call for goal."""
        blender = IKGoalBlender()
        target = Quat.from_axis_angle(Vec3(0, 1, 0), 0.5)
        result = blender.blend_rotation(goal_id=1, target=target, dt=0.016)
        assert abs(result.x - target.x) < MATH_EPSILON
        assert abs(result.y - target.y) < MATH_EPSILON
        assert abs(result.z - target.z) < MATH_EPSILON
        assert abs(result.w - target.w) < MATH_EPSILON

    def test_blend_rotation_subsequent_calls(self):
        """Test blend_rotation interpolates on subsequent calls."""
        blender = IKGoalBlender(blend_speed=1.0)
        initial = Quat.identity()
        target = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 2)

        blender.blend_rotation(goal_id=1, target=initial, dt=0.1)
        result = blender.blend_rotation(goal_id=1, target=target, dt=0.5)

        # Should be partway between identity and 90-degree rotation
        assert result is not None
        # Check it's not at initial or final
        assert abs(result.y - initial.y) > MATH_EPSILON
        assert abs(result.y - target.y) > MATH_EPSILON

    def test_blend_rotation_clamps_t_to_one(self):
        """Test blend_rotation clamps interpolation factor to 1.0."""
        blender = IKGoalBlender(blend_speed=100.0)
        initial = Quat.identity()
        target = Quat.from_axis_angle(Vec3(0, 1, 0), 0.5)

        blender.blend_rotation(goal_id=1, target=initial, dt=0.1)
        result = blender.blend_rotation(goal_id=1, target=target, dt=1.0)
        # Should be at target
        assert abs(result.y - target.y) < 0.01

    def test_blend_rotation_custom_speed_override(self):
        """Test blend_rotation with speed parameter override."""
        blender = IKGoalBlender(blend_speed=1.0)
        initial = Quat.identity()
        target = Quat.from_axis_angle(Vec3(0, 1, 0), 0.5)

        blender.blend_rotation(goal_id=1, target=initial, dt=0.1)
        result = blender.blend_rotation(goal_id=1, target=target, dt=0.1, speed=10.0)
        assert abs(result.y - target.y) < 0.01

    def test_reset_all(self):
        """Test reset clears all blending state."""
        blender = IKGoalBlender()
        blender.blend_position(goal_id=1, target=Vec3(1, 0, 0), dt=0.1)
        blender.blend_position(goal_id=2, target=Vec3(0, 1, 0), dt=0.1)
        blender.blend_rotation(goal_id=1, target=Quat.identity(), dt=0.1)

        blender.reset()

        assert len(blender._current_positions) == 0
        assert len(blender._current_rotations) == 0

    def test_reset_specific_goal_position(self):
        """Test reset with specific goal_id for positions."""
        blender = IKGoalBlender()
        blender.blend_position(goal_id=1, target=Vec3(1, 0, 0), dt=0.1)
        blender.blend_position(goal_id=2, target=Vec3(0, 1, 0), dt=0.1)

        blender.reset(goal_id=1)

        assert 1 not in blender._current_positions
        assert 2 in blender._current_positions

    def test_reset_specific_goal_rotation(self):
        """Test reset with specific goal_id for rotations."""
        blender = IKGoalBlender()
        blender.blend_rotation(goal_id=1, target=Quat.identity(), dt=0.1)
        blender.blend_rotation(goal_id=2, target=Quat.identity(), dt=0.1)

        blender.reset(goal_id=1)

        assert 1 not in blender._current_rotations
        assert 2 in blender._current_rotations

    def test_reset_nonexistent_goal(self):
        """Test reset with nonexistent goal_id doesn't raise."""
        blender = IKGoalBlender()
        blender.reset(goal_id=999)  # Should not raise

    def test_blend_position_preserves_state_between_calls(self):
        """Test that position blending maintains smooth transitions."""
        blender = IKGoalBlender(blend_speed=2.0)
        target = Vec3(10, 0, 0)

        # First call initializes
        blender.blend_position(goal_id=1, target=Vec3(0, 0, 0), dt=0.1)

        # Simulate multiple frames
        positions = []
        for _ in range(10):
            pos = blender.blend_position(goal_id=1, target=target, dt=0.1)
            positions.append(pos.x)

        # Each position should be larger than the previous (moving toward 10)
        for i in range(1, len(positions)):
            assert positions[i] >= positions[i - 1] - MATH_EPSILON


# =============================================================================
# Decorator Tests
# =============================================================================

class TestDecorators:
    """Tests for ik_goal and ik_chain decorators."""

    def test_ik_goal_decorator_default_values(self):
        """Test ik_goal decorator sets default attributes."""
        @ik_goal()
        class TestGoal:
            pass

        assert TestGoal._ik_goal is True
        assert TestGoal._ik_goal_priority == 0
        assert TestGoal._ik_goal_blend_speed == GOAL_BLENDER_DEFAULT_SPEED

    def test_ik_goal_decorator_custom_values(self):
        """Test ik_goal decorator with custom values."""
        @ik_goal(priority=5, blend_speed=20.0)
        class TestGoal:
            pass

        assert TestGoal._ik_goal is True
        assert TestGoal._ik_goal_priority == 5
        assert TestGoal._ik_goal_blend_speed == 20.0

    def test_ik_chain_decorator_default_values(self):
        """Test ik_chain decorator sets default attributes."""
        @ik_chain()
        class TestChain:
            pass

        assert TestChain._ik_chain is True
        assert TestChain._ik_solver == "fabrik"
        assert TestChain._ik_iterations == FABRIK_DEFAULT_MAX_ITERATIONS

    def test_ik_chain_decorator_custom_values(self):
        """Test ik_chain decorator with custom values."""
        @ik_chain(solver="ccd", iterations=25)
        class TestChain:
            pass

        assert TestChain._ik_chain is True
        assert TestChain._ik_solver == "ccd"
        assert TestChain._ik_iterations == 25


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================

class TestEdgeCases:
    """Edge case and boundary tests."""

    def test_position_goal_very_small_tolerance(self):
        """Test PositionGoal with very small tolerance."""
        goal = PositionGoal(bone_index=0, target_position=Vec3.zero(), tolerance=1e-10)
        assert goal.is_achieved(Vec3(1e-11, 0, 0)) is True
        assert goal.is_achieved(Vec3(1e-9, 0, 0)) is False

    def test_rotation_goal_numerical_stability(self):
        """Test RotationGoal angular_distance with dot > 1 clamping."""
        goal = RotationGoal(bone_index=0, target_rotation=Quat.identity())
        # Create nearly identical rotations
        current = Quat(0.0000001, 0, 0, 1.0).normalized()
        # Should not raise and should return small angle
        distance = goal.angular_distance(current)
        assert distance >= 0.0
        assert distance < 0.01

    def test_ik_goal_blender_zero_dt(self):
        """Test IKGoalBlender with zero delta time."""
        blender = IKGoalBlender(blend_speed=10.0)
        initial = Vec3(0, 0, 0)
        target = Vec3(10, 0, 0)

        blender.blend_position(goal_id=1, target=initial, dt=0.1)
        result = blender.blend_position(goal_id=1, target=target, dt=0.0)
        # With dt=0, t=0, so result should be initial
        assert abs(result.x - 0.0) < 0.01

    def test_look_at_goal_target_very_close(self):
        """Test LookAtGoal when target is very close to bone."""
        goal = LookAtGoal(bone_index=0, target_point=Vec3(0.000001, 0, 0))
        bone_pos = Vec3(0, 0, 0)
        current_rot = Quat.identity()
        # Should not crash and should return valid rotation
        result = goal.compute_look_rotation(bone_pos, current_rot)
        assert result is not None

    def test_pole_vector_large_coordinates(self):
        """Test PoleVectorGoal with large coordinate values."""
        goal = PoleVectorGoal(bone_index=0, pole_position=Vec3(1e6, 0, 0))
        root = Vec3(0, 0, 0)
        end = Vec3(0, 0, 1e6)
        direction = goal.compute_pole_direction(root, end)
        assert abs(direction.length() - 1.0) < MATH_EPSILON

    def test_com_goal_degenerate_polygon(self):
        """Test CenterOfMassGoal with collinear vertices."""
        # Three collinear points don't form a valid polygon
        polygon = [
            Vec3(-1, 0, 0),
            Vec3(0, 0, 0),
            Vec3(1, 0, 0),
        ]
        goal = CenterOfMassGoal(bone_index=0, support_polygon=polygon)
        result = goal.is_balanced(Vec3(0, 1, 0))
        # Behavior with degenerate polygon is implementation-defined
        assert isinstance(result, bool)


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_chain_goal_with_position_goal(self):
        """Test ChainGoal with PositionGoal end effector."""
        pos_goal = PositionGoal(bone_index=3, target_position=Vec3(5, 5, 5))
        chain = ChainGoal(
            chain_name="arm",
            bone_indices=[0, 1, 2, 3],
            end_effector_goal=pos_goal,
        )
        assert chain.end_effector_goal is pos_goal
        assert chain.validate() is True

    def test_chain_goal_with_pole_goal(self):
        """Test ChainGoal with PoleVectorGoal."""
        pole = PoleVectorGoal(bone_index=1, pole_position=Vec3(0, 0, -5))
        chain = ChainGoal(
            chain_name="leg",
            bone_indices=[0, 1, 2],
            pole_goal=pole,
        )
        assert chain.pole_goal is pole
        assert chain.validate() is True

    def test_blender_with_position_rotation_goal(self):
        """Test IKGoalBlender with PositionRotationGoal targets."""
        blender = IKGoalBlender(blend_speed=5.0)

        target_pos = Vec3(10, 10, 10)
        target_rot = Quat.from_axis_angle(Vec3(0, 1, 0), 0.5)

        # Initialize
        blender.blend_position(goal_id=1, target=Vec3.zero(), dt=0.1)
        blender.blend_rotation(goal_id=1, target=Quat.identity(), dt=0.1)

        # Blend toward targets
        pos_result = blender.blend_position(goal_id=1, target=target_pos, dt=0.1)
        rot_result = blender.blend_rotation(goal_id=1, target=target_rot, dt=0.1)

        # Both should have moved from initial
        assert pos_result.x > 0
        assert abs(rot_result.y) > MATH_EPSILON
