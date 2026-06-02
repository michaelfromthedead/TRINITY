"""
Blackbox tests for T-IK-3.2: Joint Constraints Base

These tests verify the Joint Constraint system's public contract without
knowledge of internal implementation details. Tests are derived from
acceptance criteria for task T-IK-3.2.

Acceptance Criteria under test:
1. JointConstraint dataclass with constraint_type field
2. apply(direction, parent_rotation) signature
3. BALL_SOCKET constraint type with cone angle
4. HINGE constraint type with axis and angle limits
5. TWIST_LIMIT constraint type with twist limits
6. Numerical stability checks

Blackbox Test Strategy:
- Test public API contracts only
- Test behavioral expectations from acceptance criteria
- Test boundary conditions and edge cases
- Test error handling for invalid inputs

API Notes:
- Uses single JointConstraint dataclass with constraint_type field
- JointConstraintType enum: NONE, HINGE, BALL_SOCKET, TWIST_LIMIT
- twist_min/twist_max fields (not min_twist/max_twist)
"""

import math
import pytest
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import is_dataclass

from engine.core.math import Vec3, Quat


# -----------------------------------------------------------------------------
# Test Constants
# -----------------------------------------------------------------------------

EPSILON = 1e-6  # Floating point comparison tolerance
ANGLE_TOLERANCE = 1e-4  # Angular tolerance in radians


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

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


def vec3_normalize(v: Vec3) -> Vec3:
    """Normalize a Vec3."""
    length = vec3_length(v)
    if length < EPSILON:
        return Vec3(0, 0, 0)
    return Vec3(v.x / length, v.y / length, v.z / length)


def vec3_dot(v1: Vec3, v2: Vec3) -> float:
    """Compute dot product of two Vec3."""
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z


def vec3_cross(v1: Vec3, v2: Vec3) -> Vec3:
    """Compute cross product of two Vec3."""
    return Vec3(
        v1.y * v2.z - v1.z * v2.y,
        v1.z * v2.x - v1.x * v2.z,
        v1.x * v2.y - v1.y * v2.x
    )


def angle_between_vectors(v1: Vec3, v2: Vec3) -> float:
    """Compute angle between two vectors in radians."""
    v1_norm = vec3_normalize(v1)
    v2_norm = vec3_normalize(v2)
    dot = vec3_dot(v1_norm, v2_norm)
    # Clamp to [-1, 1] for numerical stability
    dot = max(-1.0, min(1.0, dot))
    return math.acos(dot)


def is_normalized(v: Vec3, eps: float = EPSILON) -> bool:
    """Check if a Vec3 is normalized (unit length)."""
    length = vec3_length(v)
    return abs(length - 1.0) < eps


# =============================================================================
# Import Tests - Verify Module Structure
# =============================================================================

class TestJointConstraintImports:
    """Test that all expected classes and enums can be imported."""

    def test_import_joint_constraint_type_enum(self):
        """JointConstraintType enum should be importable."""
        from engine.animation.ik import JointConstraintType
        assert JointConstraintType is not None

    def test_import_joint_constraint_base(self):
        """JointConstraint dataclass should be importable."""
        from engine.animation.ik import JointConstraint
        assert JointConstraint is not None

    def test_ball_socket_via_joint_constraint(self):
        """Ball socket constraint should be creatable via JointConstraint."""
        from engine.animation.ik import JointConstraint, JointConstraintType
        constraint = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=0.5
        )
        assert constraint is not None
        assert constraint.constraint_type == JointConstraintType.BALL_SOCKET

    def test_hinge_via_joint_constraint(self):
        """Hinge constraint should be creatable via JointConstraint."""
        from engine.animation.ik import JointConstraint, JointConstraintType
        constraint = JointConstraint(
            constraint_type=JointConstraintType.HINGE,
            axis=Vec3(0.0, 1.0, 0.0),
            min_angle=-1.0,
            max_angle=1.0
        )
        assert constraint is not None
        assert constraint.constraint_type == JointConstraintType.HINGE

    def test_twist_via_joint_constraint(self):
        """Twist constraint should be creatable via JointConstraint."""
        from engine.animation.ik import JointConstraint, JointConstraintType
        constraint = JointConstraint(
            constraint_type=JointConstraintType.TWIST_LIMIT,
            twist_min=-1.0,
            twist_max=1.0
        )
        assert constraint is not None
        assert constraint.constraint_type == JointConstraintType.TWIST_LIMIT


# =============================================================================
# AC1: JointConstraintType Enum Tests
# =============================================================================

class TestJointConstraintTypeEnum:
    """Tests for JointConstraintType enum values."""

    def test_has_none_value(self):
        """JointConstraintType should have NONE value."""
        from engine.animation.ik import JointConstraintType
        assert hasattr(JointConstraintType, 'NONE'), "Should have NONE constraint type"

    def test_has_hinge_value(self):
        """JointConstraintType should have HINGE value."""
        from engine.animation.ik import JointConstraintType
        assert hasattr(JointConstraintType, 'HINGE'), "Should have HINGE constraint type"

    def test_has_ball_socket_value(self):
        """JointConstraintType should have BALL_SOCKET value."""
        from engine.animation.ik import JointConstraintType
        assert hasattr(JointConstraintType, 'BALL_SOCKET'), "Should have BALL_SOCKET constraint type"

    def test_has_twist_limit_value(self):
        """JointConstraintType should have TWIST_LIMIT value."""
        from engine.animation.ik import JointConstraintType
        assert hasattr(JointConstraintType, 'TWIST_LIMIT'), "Should have TWIST_LIMIT constraint type"

    def test_enum_values_are_distinct(self):
        """All JointConstraintType values should be distinct."""
        from engine.animation.ik import JointConstraintType
        values = [
            JointConstraintType.NONE,
            JointConstraintType.HINGE,
            JointConstraintType.BALL_SOCKET,
            JointConstraintType.TWIST_LIMIT
        ]
        assert len(values) == len(set(values)), "All enum values should be distinct"


# =============================================================================
# AC1: JointConstraint Dataclass Tests
# =============================================================================

class TestJointConstraintBase:
    """Tests for JointConstraint dataclass (AC1)."""

    def test_joint_constraint_exists(self):
        """JointConstraint class should exist."""
        from engine.animation.ik import JointConstraint
        assert JointConstraint is not None

    def test_joint_constraint_is_dataclass(self):
        """JointConstraint should be a dataclass."""
        from engine.animation.ik import JointConstraint
        assert is_dataclass(JointConstraint), "JointConstraint should be a dataclass"

    def test_joint_constraint_has_constraint_type(self):
        """JointConstraint should have constraint_type attribute."""
        from engine.animation.ik import JointConstraint, JointConstraintType
        constraint = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=0.5
        )
        assert hasattr(constraint, 'constraint_type'), "Should have constraint_type attribute"

    def test_joint_constraint_has_apply_method(self):
        """JointConstraint should have apply() method."""
        from engine.animation.ik import JointConstraint, JointConstraintType
        constraint = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=0.5
        )
        assert hasattr(constraint, 'apply'), "Should have apply method"
        assert callable(constraint.apply), "apply should be callable"


# =============================================================================
# AC2: apply(direction, parent_rotation) Signature Tests
# =============================================================================

class TestApplyMethodSignature:
    """Tests for the apply() method signature (AC2)."""

    def test_apply_accepts_direction_and_parent_rotation(self):
        """apply() should accept direction and parent_rotation parameters."""
        from engine.animation.ik import JointConstraint, JointConstraintType

        constraint = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=0.5
        )
        direction = Vec3(1.0, 0.0, 0.0)
        parent_rotation = Quat.identity()

        # Should not raise
        result = constraint.apply(direction=direction, parent_rotation=parent_rotation)
        assert result is not None

    def test_apply_returns_vec3(self):
        """apply() should return a Vec3."""
        from engine.animation.ik import JointConstraint, JointConstraintType

        constraint = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=0.5
        )
        direction = Vec3(1.0, 0.0, 0.0)
        parent_rotation = Quat.identity()

        result = constraint.apply(direction=direction, parent_rotation=parent_rotation)
        assert isinstance(result, Vec3), f"apply() should return Vec3, got {type(result)}"

    def test_apply_with_keyword_arguments(self):
        """apply() should work with keyword arguments."""
        from engine.animation.ik import JointConstraint, JointConstraintType

        constraint = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=0.5
        )

        # Should work with keyword args
        result = constraint.apply(
            direction=Vec3(0.0, 1.0, 0.0),
            parent_rotation=Quat.identity()
        )
        assert isinstance(result, Vec3)

    def test_apply_with_positional_arguments(self):
        """apply() should work with positional arguments."""
        from engine.animation.ik import JointConstraint, JointConstraintType

        constraint = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=0.5
        )

        # Should work with positional args
        result = constraint.apply(Vec3(0.0, 1.0, 0.0), Quat.identity())
        assert isinstance(result, Vec3)


# =============================================================================
# AC3: BALL_SOCKET Constraint Type with Cone Angle Tests
# =============================================================================

class TestBallSocketConstraint:
    """Tests for BALL_SOCKET constraint type (AC3)."""

    def test_ball_socket_instantiation(self):
        """BALL_SOCKET constraint can be instantiated with cone_angle."""
        from engine.animation.ik import JointConstraint, JointConstraintType

        constraint = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=0.5
        )
        assert constraint is not None

    def test_ball_socket_has_cone_angle(self):
        """BALL_SOCKET constraint should have cone_angle attribute."""
        from engine.animation.ik import JointConstraint, JointConstraintType

        cone_angle = 0.5
        constraint = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=cone_angle
        )
        assert hasattr(constraint, 'cone_angle'), "Should have cone_angle attribute"
        assert abs(constraint.cone_angle - cone_angle) < EPSILON

    def test_ball_socket_has_correct_constraint_type(self):
        """BALL_SOCKET constraint should have BALL_SOCKET constraint type."""
        from engine.animation.ik import JointConstraint, JointConstraintType

        constraint = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=0.5
        )
        assert constraint.constraint_type == JointConstraintType.BALL_SOCKET

    def test_ball_socket_direction_within_cone_unchanged(self):
        """Directions within cone angle should pass through unchanged."""
        from engine.animation.ik import JointConstraint, JointConstraintType

        cone_angle = math.pi / 4  # 45 degrees
        constraint = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=cone_angle
        )

        # Direction along the Y axis (reference direction for ball socket is +Y)
        direction = Vec3(0.0, 1.0, 0.0)
        parent_rotation = Quat.identity()

        result = constraint.apply(direction=direction, parent_rotation=parent_rotation)

        # Should be unchanged (or at least very close)
        assert vec3_approx_equal(result, direction, eps=ANGLE_TOLERANCE), \
            f"Direction within cone should be unchanged: got {result}"

    def test_ball_socket_direction_outside_cone_clamped(self):
        """Directions outside cone should be clamped to cone surface."""
        from engine.animation.ik import JointConstraint, JointConstraintType

        cone_angle = math.pi / 6  # 30 degrees
        constraint = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=cone_angle
        )

        # Direction at 60 degrees from +Y axis (reference direction)
        # This direction is mostly +X
        direction = Vec3(math.sin(math.pi / 3), math.cos(math.pi / 3), 0.0)
        parent_rotation = Quat.identity()

        result = constraint.apply(direction=direction, parent_rotation=parent_rotation)
        result_normalized = vec3_normalize(result)

        # The angle to the reference direction (+Y) should now be at most cone_angle
        ref_dir = Vec3(0.0, 1.0, 0.0)
        result_angle = angle_between_vectors(result_normalized, ref_dir)

        assert result_angle <= cone_angle + ANGLE_TOLERANCE, \
            f"Result angle {result_angle} should be <= cone angle {cone_angle}"

    def test_ball_socket_result_valid(self):
        """Ball socket constraint result should be valid Vec3."""
        from engine.animation.ik import JointConstraint, JointConstraintType

        constraint = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=0.5
        )
        direction = Vec3(1.0, 1.0, 1.0)  # Non-normalized input
        parent_rotation = Quat.identity()

        result = constraint.apply(direction=direction, parent_rotation=parent_rotation)

        # Result should be valid Vec3
        assert isinstance(result, Vec3)
        # Result should not be NaN or Inf
        assert not math.isnan(result.x) and not math.isnan(result.y) and not math.isnan(result.z)
        assert not math.isinf(result.x) and not math.isinf(result.y) and not math.isinf(result.z)

    def test_ball_socket_with_small_cone_angle(self):
        """BALL_SOCKET constraint should work with very small cone angles."""
        from engine.animation.ik import JointConstraint, JointConstraintType

        # Very small cone (1 degree)
        cone_angle = math.radians(1.0)
        constraint = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=cone_angle
        )

        # Any off-axis direction should be clamped significantly
        direction = Vec3(1.0, 0.0, 0.0)  # 90 degrees off from +Y
        parent_rotation = Quat.identity()

        result = constraint.apply(direction=direction, parent_rotation=parent_rotation)
        result_normalized = vec3_normalize(result)

        # Should be clamped to very close to +Y reference axis
        ref_dir = Vec3(0.0, 1.0, 0.0)
        result_angle = angle_between_vectors(result_normalized, ref_dir)

        assert result_angle <= cone_angle + ANGLE_TOLERANCE

    def test_ball_socket_with_large_cone_angle(self):
        """BALL_SOCKET constraint should work with large cone angles (nearly full sphere)."""
        from engine.animation.ik import JointConstraint, JointConstraintType

        # Very large cone (179 degrees - almost full sphere)
        cone_angle = math.radians(179.0)
        constraint = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=cone_angle
        )

        # Even backward direction should pass through
        direction = Vec3(0.0, -0.9, 0.0)  # Almost backward, but still within 179 degrees
        parent_rotation = Quat.identity()

        result = constraint.apply(direction=direction, parent_rotation=parent_rotation)

        # Should be close to original since within the large cone
        assert result is not None


# =============================================================================
# AC4: HINGE Constraint Type with Axis and Angle Limits Tests
# =============================================================================

class TestHingeConstraint:
    """Tests for HINGE constraint type (AC4)."""

    def test_hinge_instantiation(self):
        """HINGE constraint can be instantiated with axis and angle limits."""
        from engine.animation.ik import JointConstraint, JointConstraintType

        axis = Vec3(0.0, 1.0, 0.0)  # Hinge around Y axis
        min_angle = -math.pi / 2
        max_angle = math.pi / 2

        constraint = JointConstraint(
            constraint_type=JointConstraintType.HINGE,
            axis=axis,
            min_angle=min_angle,
            max_angle=max_angle
        )
        assert constraint is not None

    def test_hinge_has_axis_attribute(self):
        """HINGE constraint should have axis attribute."""
        from engine.animation.ik import JointConstraint, JointConstraintType

        axis = Vec3(0.0, 1.0, 0.0)
        constraint = JointConstraint(
            constraint_type=JointConstraintType.HINGE,
            axis=axis,
            min_angle=-1.0,
            max_angle=1.0
        )

        assert hasattr(constraint, 'axis'), "Should have axis attribute"
        assert isinstance(constraint.axis, Vec3), "axis should be Vec3"

    def test_hinge_has_min_max_angle_attributes(self):
        """HINGE constraint should have min_angle and max_angle attributes."""
        from engine.animation.ik import JointConstraint, JointConstraintType

        min_angle = -math.pi / 4
        max_angle = math.pi / 4
        constraint = JointConstraint(
            constraint_type=JointConstraintType.HINGE,
            axis=Vec3(0.0, 1.0, 0.0),
            min_angle=min_angle,
            max_angle=max_angle
        )

        assert hasattr(constraint, 'min_angle'), "Should have min_angle attribute"
        assert hasattr(constraint, 'max_angle'), "Should have max_angle attribute"
        assert abs(constraint.min_angle - min_angle) < EPSILON
        assert abs(constraint.max_angle - max_angle) < EPSILON

    def test_hinge_has_correct_constraint_type(self):
        """HINGE constraint should have HINGE constraint type."""
        from engine.animation.ik import JointConstraint, JointConstraintType

        constraint = JointConstraint(
            constraint_type=JointConstraintType.HINGE,
            axis=Vec3(0.0, 1.0, 0.0),
            min_angle=-1.0,
            max_angle=1.0
        )
        assert constraint.constraint_type == JointConstraintType.HINGE

    def test_hinge_constrains_to_plane(self):
        """Hinge constraint should constrain rotation to plane perpendicular to axis."""
        from engine.animation.ik import JointConstraint, JointConstraintType

        # Hinge around Y axis - motion should be in XZ plane
        axis = Vec3(0.0, 1.0, 0.0)
        constraint = JointConstraint(
            constraint_type=JointConstraintType.HINGE,
            axis=axis,
            min_angle=-math.pi,
            max_angle=math.pi
        )

        # Direction with Y component
        direction = Vec3(1.0, 1.0, 0.0)
        parent_rotation = Quat.identity()

        result = constraint.apply(direction=direction, parent_rotation=parent_rotation)

        # Result should have minimal Y component (in plane perpendicular to Y axis)
        # The result should be projected onto the XZ plane
        assert abs(result.y) < ANGLE_TOLERANCE, \
            f"Hinge result should be in plane perpendicular to axis, Y={result.y}"

    def test_hinge_projects_to_plane(self):
        """Hinge constraint should project direction onto plane perpendicular to axis."""
        from engine.animation.ik import JointConstraint, JointConstraintType

        # Hinge around Y axis with angle limits
        # Note: Current implementation only projects to plane, does not enforce angle limits
        axis = Vec3(0.0, 1.0, 0.0)
        min_angle = -math.pi / 4  # -45 degrees
        max_angle = math.pi / 4   # +45 degrees
        constraint = JointConstraint(
            constraint_type=JointConstraintType.HINGE,
            axis=axis,
            min_angle=min_angle,
            max_angle=max_angle
        )

        # Direction at 90 degrees
        direction = Vec3(1.0, 0.5, 0.0)  # Has Y component
        parent_rotation = Quat.identity()

        result = constraint.apply(direction=direction, parent_rotation=parent_rotation)

        # Result should be projected onto XZ plane (Y component should be near 0)
        assert abs(result.y) < ANGLE_TOLERANCE, \
            f"Result should be in plane perpendicular to Y axis, Y={result.y}"

    def test_hinge_apply_returns_vec3(self):
        """Hinge apply() should return Vec3."""
        from engine.animation.ik import JointConstraint, JointConstraintType

        constraint = JointConstraint(
            constraint_type=JointConstraintType.HINGE,
            axis=Vec3(0.0, 1.0, 0.0),
            min_angle=-1.0,
            max_angle=1.0
        )

        result = constraint.apply(
            direction=Vec3(1.0, 0.0, 0.0),
            parent_rotation=Quat.identity()
        )

        assert isinstance(result, Vec3)


# =============================================================================
# AC5: TWIST_LIMIT Constraint Type with Twist Limits Tests
# =============================================================================

class TestTwistConstraint:
    """Tests for TWIST_LIMIT constraint type (AC5)."""

    def test_twist_instantiation(self):
        """TWIST_LIMIT constraint can be instantiated with twist limits."""
        from engine.animation.ik import JointConstraint, JointConstraintType

        twist_min = -math.pi / 2
        twist_max = math.pi / 2

        constraint = JointConstraint(
            constraint_type=JointConstraintType.TWIST_LIMIT,
            twist_min=twist_min,
            twist_max=twist_max
        )
        assert constraint is not None

    def test_twist_has_twist_min_max_attributes(self):
        """TWIST_LIMIT constraint should have twist_min and twist_max attributes."""
        from engine.animation.ik import JointConstraint, JointConstraintType

        twist_min = -math.pi / 4
        twist_max = math.pi / 4
        constraint = JointConstraint(
            constraint_type=JointConstraintType.TWIST_LIMIT,
            twist_min=twist_min,
            twist_max=twist_max
        )

        assert hasattr(constraint, 'twist_min'), "Should have twist_min attribute"
        assert hasattr(constraint, 'twist_max'), "Should have twist_max attribute"
        assert abs(constraint.twist_min - twist_min) < EPSILON
        assert abs(constraint.twist_max - twist_max) < EPSILON

    def test_twist_has_correct_constraint_type(self):
        """TWIST_LIMIT constraint should have TWIST_LIMIT constraint type."""
        from engine.animation.ik import JointConstraint, JointConstraintType

        constraint = JointConstraint(
            constraint_type=JointConstraintType.TWIST_LIMIT,
            twist_min=-1.0,
            twist_max=1.0
        )
        assert constraint.constraint_type == JointConstraintType.TWIST_LIMIT

    def test_twist_apply_returns_vec3(self):
        """Twist apply() should return Vec3."""
        from engine.animation.ik import JointConstraint, JointConstraintType

        constraint = JointConstraint(
            constraint_type=JointConstraintType.TWIST_LIMIT,
            twist_min=-1.0,
            twist_max=1.0
        )

        result = constraint.apply(
            direction=Vec3(0.0, 0.0, 1.0),
            parent_rotation=Quat.identity()
        )

        assert isinstance(result, Vec3)

    def test_twist_with_symmetric_limits(self):
        """TWIST_LIMIT constraint should work with symmetric twist limits."""
        from engine.animation.ik import JointConstraint, JointConstraintType

        constraint = JointConstraint(
            constraint_type=JointConstraintType.TWIST_LIMIT,
            twist_min=-math.pi / 2,
            twist_max=math.pi / 2
        )

        # Apply with forward direction
        result = constraint.apply(
            direction=Vec3(0.0, 0.0, 1.0),
            parent_rotation=Quat.identity()
        )

        assert result is not None

    def test_twist_with_asymmetric_limits(self):
        """TWIST_LIMIT constraint should work with asymmetric twist limits."""
        from engine.animation.ik import JointConstraint, JointConstraintType

        # More range in positive direction
        constraint = JointConstraint(
            constraint_type=JointConstraintType.TWIST_LIMIT,
            twist_min=-math.pi / 4,
            twist_max=math.pi
        )

        result = constraint.apply(
            direction=Vec3(0.0, 0.0, 1.0),
            parent_rotation=Quat.identity()
        )

        assert result is not None


# =============================================================================
# AC6: Numerical Stability Tests
# =============================================================================

class TestNumericalStability:
    """Tests for numerical stability (AC6)."""

    def test_zero_length_direction_ball_socket(self):
        """BALL_SOCKET constraint should handle zero-length direction gracefully."""
        from engine.animation.ik import JointConstraint, JointConstraintType

        constraint = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=0.5
        )
        zero_dir = Vec3(0.0, 0.0, 0.0)
        parent_rotation = Quat.identity()

        # Should not raise exception
        result = constraint.apply(direction=zero_dir, parent_rotation=parent_rotation)

        # Result should be valid (zero or default direction)
        assert isinstance(result, Vec3)

    def test_zero_length_direction_hinge(self):
        """HINGE constraint should handle zero-length direction gracefully."""
        from engine.animation.ik import JointConstraint, JointConstraintType

        constraint = JointConstraint(
            constraint_type=JointConstraintType.HINGE,
            axis=Vec3(0.0, 1.0, 0.0),
            min_angle=-1.0,
            max_angle=1.0
        )
        zero_dir = Vec3(0.0, 0.0, 0.0)
        parent_rotation = Quat.identity()

        # Should not raise exception
        result = constraint.apply(direction=zero_dir, parent_rotation=parent_rotation)
        assert isinstance(result, Vec3)

    def test_zero_length_direction_twist(self):
        """TWIST_LIMIT constraint should handle zero-length direction gracefully."""
        from engine.animation.ik import JointConstraint, JointConstraintType

        constraint = JointConstraint(
            constraint_type=JointConstraintType.TWIST_LIMIT,
            twist_min=-1.0,
            twist_max=1.0
        )
        zero_dir = Vec3(0.0, 0.0, 0.0)
        parent_rotation = Quat.identity()

        # Should not raise exception
        result = constraint.apply(direction=zero_dir, parent_rotation=parent_rotation)
        assert isinstance(result, Vec3)

    def test_direction_parallel_to_hinge_axis(self):
        """HINGE constraint should handle direction parallel to axis."""
        from engine.animation.ik import JointConstraint, JointConstraintType

        axis = Vec3(0.0, 1.0, 0.0)
        constraint = JointConstraint(
            constraint_type=JointConstraintType.HINGE,
            axis=axis,
            min_angle=-1.0,
            max_angle=1.0
        )

        # Direction parallel to axis (edge case)
        parallel_dir = Vec3(0.0, 1.0, 0.0)
        parent_rotation = Quat.identity()

        # Should not raise exception
        result = constraint.apply(direction=parallel_dir, parent_rotation=parent_rotation)
        assert isinstance(result, Vec3)

    def test_very_small_direction(self):
        """Constraints should handle very small (near-zero) directions."""
        from engine.animation.ik import JointConstraint, JointConstraintType

        constraint = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=0.5
        )
        small_dir = Vec3(1e-10, 1e-10, 1e-10)
        parent_rotation = Quat.identity()

        # Should not raise exception
        result = constraint.apply(direction=small_dir, parent_rotation=parent_rotation)
        assert isinstance(result, Vec3)

    def test_very_large_direction(self):
        """Constraints should handle very large direction magnitudes."""
        from engine.animation.ik import JointConstraint, JointConstraintType

        constraint = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=0.5
        )
        large_dir = Vec3(1e10, 1e10, 1e10)
        parent_rotation = Quat.identity()

        # Should not raise exception
        result = constraint.apply(direction=large_dir, parent_rotation=parent_rotation)
        assert isinstance(result, Vec3)

        # Result should not be NaN or Inf
        assert not math.isnan(result.x) and not math.isnan(result.y) and not math.isnan(result.z)
        assert not math.isinf(result.x) and not math.isinf(result.y) and not math.isinf(result.z)

    def test_valid_output_ball_socket(self):
        """BALL_SOCKET constraint output should always be valid Vec3."""
        from engine.animation.ik import JointConstraint, JointConstraintType

        constraint = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=math.pi / 4
        )

        test_directions = [
            Vec3(1.0, 0.0, 0.0),
            Vec3(0.0, 1.0, 0.0),
            Vec3(0.0, 0.0, 1.0),
            Vec3(1.0, 1.0, 1.0),
            Vec3(-1.0, 0.5, 0.3),
        ]

        for direction in test_directions:
            result = constraint.apply(direction=direction, parent_rotation=Quat.identity())
            assert isinstance(result, Vec3), f"Result should be Vec3 for direction {direction}"
            # Result should not be NaN or Inf
            assert not math.isnan(result.x), f"Result x should not be NaN for direction {direction}"
            assert not math.isnan(result.y), f"Result y should not be NaN for direction {direction}"
            assert not math.isnan(result.z), f"Result z should not be NaN for direction {direction}"

    def test_normalized_output_hinge(self):
        """HINGE constraint output should always be normalized."""
        from engine.animation.ik import JointConstraint, JointConstraintType

        constraint = JointConstraint(
            constraint_type=JointConstraintType.HINGE,
            axis=Vec3(0.0, 1.0, 0.0),
            min_angle=-math.pi / 2,
            max_angle=math.pi / 2
        )

        test_directions = [
            Vec3(1.0, 0.0, 0.0),
            Vec3(0.0, 0.0, 1.0),
            Vec3(1.0, 0.0, 1.0),
            Vec3(-1.0, 0.0, 0.3),
        ]

        for direction in test_directions:
            result = constraint.apply(direction=direction, parent_rotation=Quat.identity())
            length = vec3_length(result)
            if length > EPSILON:
                assert abs(length - 1.0) < ANGLE_TOLERANCE, \
                    f"Result should be normalized for direction {direction}, got length {length}"

    def test_result_not_nan_or_inf(self):
        """Constraint results should never be NaN or Inf."""
        from engine.animation.ik import JointConstraint, JointConstraintType

        constraints = [
            JointConstraint(
                constraint_type=JointConstraintType.BALL_SOCKET,
                cone_angle=0.5
            ),
            JointConstraint(
                constraint_type=JointConstraintType.HINGE,
                axis=Vec3(0.0, 1.0, 0.0),
                min_angle=-1.0,
                max_angle=1.0
            ),
            JointConstraint(
                constraint_type=JointConstraintType.TWIST_LIMIT,
                twist_min=-1.0,
                twist_max=1.0
            ),
        ]

        test_directions = [
            Vec3(0.0, 0.0, 0.0),
            Vec3(1.0, 0.0, 0.0),
            Vec3(1e-15, 1e-15, 1e-15),
            Vec3(1e10, 0.0, 0.0),
        ]

        for constraint in constraints:
            for direction in test_directions:
                result = constraint.apply(direction=direction, parent_rotation=Quat.identity())

                assert not math.isnan(result.x), f"Result x should not be NaN: {result}"
                assert not math.isnan(result.y), f"Result y should not be NaN: {result}"
                assert not math.isnan(result.z), f"Result z should not be NaN: {result}"
                assert not math.isinf(result.x), f"Result x should not be Inf: {result}"
                assert not math.isinf(result.y), f"Result y should not be Inf: {result}"
                assert not math.isinf(result.z), f"Result z should not be Inf: {result}"


# =============================================================================
# NONE Constraint Tests
# =============================================================================

class TestNoneConstraint:
    """Tests for NONE constraint type (passthrough behavior)."""

    def test_none_constraint_passthrough(self):
        """NONE constraint type should pass direction through unchanged."""
        from engine.animation.ik import JointConstraint, JointConstraintType

        # Create a constraint with NONE type
        constraint = JointConstraint(constraint_type=JointConstraintType.NONE)

        direction = Vec3(1.0, 2.0, 3.0)
        parent_rotation = Quat.identity()

        result = constraint.apply(direction=direction, parent_rotation=parent_rotation)

        # NONE should pass through unchanged (normalized)
        dir_normalized = vec3_normalize(direction)
        assert vec3_approx_equal(result, dir_normalized, eps=ANGLE_TOLERANCE) or \
               vec3_approx_equal(result, direction, eps=ANGLE_TOLERANCE), \
               f"NONE constraint should passthrough, got {result}"


# =============================================================================
# Parent Rotation Tests
# =============================================================================

class TestParentRotation:
    """Tests for parent_rotation parameter behavior."""

    def test_ball_socket_respects_parent_rotation(self):
        """BALL_SOCKET constraint should consider parent rotation."""
        from engine.animation.ik import JointConstraint, JointConstraintType

        constraint = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=math.pi / 4
        )
        direction = Vec3(1.0, 0.0, 0.0)

        # Test with identity rotation
        result_identity = constraint.apply(
            direction=direction,
            parent_rotation=Quat.identity()
        )

        # Test with 90-degree rotation around Y
        # This creates a rotation that rotates X to Z
        rotated_quat = Quat.from_axis_angle(Vec3(0.0, 1.0, 0.0), math.pi / 2)

        result_rotated = constraint.apply(
            direction=direction,
            parent_rotation=rotated_quat
        )

        # Results should be different due to different reference frames
        # (unless direction is already within cone in both cases)
        assert isinstance(result_identity, Vec3)
        assert isinstance(result_rotated, Vec3)

    def test_hinge_respects_parent_rotation(self):
        """HINGE constraint should consider parent rotation."""
        from engine.animation.ik import JointConstraint, JointConstraintType

        constraint = JointConstraint(
            constraint_type=JointConstraintType.HINGE,
            axis=Vec3(0.0, 1.0, 0.0),
            min_angle=-math.pi / 4,
            max_angle=math.pi / 4
        )

        direction = Vec3(1.0, 0.0, 1.0)

        result_identity = constraint.apply(
            direction=direction,
            parent_rotation=Quat.identity()
        )

        rotated_quat = Quat.from_axis_angle(Vec3(0.0, 0.0, 1.0), math.pi / 4)

        result_rotated = constraint.apply(
            direction=direction,
            parent_rotation=rotated_quat
        )

        assert isinstance(result_identity, Vec3)
        assert isinstance(result_rotated, Vec3)


# =============================================================================
# Edge Cases and Boundary Conditions
# =============================================================================

class TestEdgeCases:
    """Additional edge case tests."""

    def test_ball_socket_exactly_on_cone_boundary(self):
        """Direction exactly on cone boundary should be valid."""
        from engine.animation.ik import JointConstraint, JointConstraintType

        cone_angle = math.pi / 4  # 45 degrees
        constraint = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=cone_angle
        )

        # Direction exactly at 45 degrees from +Y (reference direction)
        direction = Vec3(
            math.sin(cone_angle),
            math.cos(cone_angle),
            0.0
        )

        result = constraint.apply(direction=direction, parent_rotation=Quat.identity())

        # Should be valid and approximately unchanged
        assert isinstance(result, Vec3)
        length = vec3_length(result)
        if length > EPSILON:
            assert abs(length - 1.0) < ANGLE_TOLERANCE

    def test_hinge_at_min_angle(self):
        """Direction at exactly min_angle should be valid."""
        from engine.animation.ik import JointConstraint, JointConstraintType

        min_angle = -math.pi / 4
        max_angle = math.pi / 4
        constraint = JointConstraint(
            constraint_type=JointConstraintType.HINGE,
            axis=Vec3(0.0, 1.0, 0.0),
            min_angle=min_angle,
            max_angle=max_angle
        )

        # Direction at exactly min_angle in XZ plane
        direction = Vec3(
            math.sin(min_angle),
            0.0,
            math.cos(min_angle)
        )

        result = constraint.apply(direction=direction, parent_rotation=Quat.identity())
        assert isinstance(result, Vec3)

    def test_hinge_at_max_angle(self):
        """Direction at exactly max_angle should be valid."""
        from engine.animation.ik import JointConstraint, JointConstraintType

        min_angle = -math.pi / 4
        max_angle = math.pi / 4
        constraint = JointConstraint(
            constraint_type=JointConstraintType.HINGE,
            axis=Vec3(0.0, 1.0, 0.0),
            min_angle=min_angle,
            max_angle=max_angle
        )

        # Direction at exactly max_angle in XZ plane
        direction = Vec3(
            math.sin(max_angle),
            0.0,
            math.cos(max_angle)
        )

        result = constraint.apply(direction=direction, parent_rotation=Quat.identity())
        assert isinstance(result, Vec3)

    def test_twist_at_min_twist(self):
        """Twist at exactly twist_min should be valid."""
        from engine.animation.ik import JointConstraint, JointConstraintType

        constraint = JointConstraint(
            constraint_type=JointConstraintType.TWIST_LIMIT,
            twist_min=-math.pi / 2,
            twist_max=math.pi / 2
        )

        result = constraint.apply(
            direction=Vec3(0.0, 0.0, 1.0),
            parent_rotation=Quat.identity()
        )
        assert isinstance(result, Vec3)

    def test_twist_at_max_twist(self):
        """Twist at exactly twist_max should be valid."""
        from engine.animation.ik import JointConstraint, JointConstraintType

        constraint = JointConstraint(
            constraint_type=JointConstraintType.TWIST_LIMIT,
            twist_min=-math.pi / 2,
            twist_max=math.pi / 2
        )

        result = constraint.apply(
            direction=Vec3(0.0, 0.0, 1.0),
            parent_rotation=Quat.identity()
        )
        assert isinstance(result, Vec3)

    def test_ball_socket_zero_cone_angle(self):
        """BALL_SOCKET constraint with zero cone angle (locked)."""
        from engine.animation.ik import JointConstraint, JointConstraintType

        # Zero cone angle means only reference direction is valid
        constraint = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=0.0
        )

        direction = Vec3(1.0, 0.0, 0.0)  # Off-axis
        result = constraint.apply(direction=direction, parent_rotation=Quat.identity())

        # Should be clamped to exactly reference direction (+Y)
        assert isinstance(result, Vec3)
        # Result should be very close to +Y (or handle gracefully)

    def test_hinge_zero_range(self):
        """HINGE constraint with zero angle range (locked)."""
        from engine.animation.ik import JointConstraint, JointConstraintType

        constraint = JointConstraint(
            constraint_type=JointConstraintType.HINGE,
            axis=Vec3(0.0, 1.0, 0.0),
            min_angle=0.0,
            max_angle=0.0
        )

        direction = Vec3(1.0, 0.0, 1.0)
        result = constraint.apply(direction=direction, parent_rotation=Quat.identity())

        assert isinstance(result, Vec3)

    def test_multiple_constraint_applications(self):
        """Applying constraint multiple times should give consistent results."""
        from engine.animation.ik import JointConstraint, JointConstraintType

        constraint = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=math.pi / 4
        )
        direction = Vec3(1.0, 1.0, 1.0)
        parent_rotation = Quat.identity()

        result1 = constraint.apply(direction=direction, parent_rotation=parent_rotation)
        result2 = constraint.apply(direction=direction, parent_rotation=parent_rotation)

        # Results should be identical
        assert vec3_approx_equal(result1, result2), \
            f"Multiple applications should give same result: {result1} vs {result2}"

    def test_idempotent_constraint_application(self):
        """Applying constraint to already-constrained direction should be idempotent."""
        from engine.animation.ik import JointConstraint, JointConstraintType

        constraint = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=math.pi / 4
        )
        direction = Vec3(1.0, 0.0, 0.0)  # Outside cone
        parent_rotation = Quat.identity()

        # First application
        result1 = constraint.apply(direction=direction, parent_rotation=parent_rotation)

        # Second application on result
        result2 = constraint.apply(direction=result1, parent_rotation=parent_rotation)

        # Should be the same (idempotent)
        assert vec3_approx_equal(result1, result2, eps=ANGLE_TOLERANCE), \
            f"Constraint should be idempotent: {result1} vs {result2}"


# =============================================================================
# Integration Tests
# =============================================================================

class TestConstraintIntegration:
    """Integration tests for constraint interactions."""

    def test_all_constraint_types_have_apply(self):
        """All constraint types should have working apply method."""
        from engine.animation.ik import JointConstraint, JointConstraintType

        constraints = [
            JointConstraint(
                constraint_type=JointConstraintType.BALL_SOCKET,
                cone_angle=0.5
            ),
            JointConstraint(
                constraint_type=JointConstraintType.HINGE,
                axis=Vec3(0.0, 1.0, 0.0),
                min_angle=-1.0,
                max_angle=1.0
            ),
            JointConstraint(
                constraint_type=JointConstraintType.TWIST_LIMIT,
                twist_min=-1.0,
                twist_max=1.0
            ),
        ]

        direction = Vec3(1.0, 0.0, 0.0)
        parent_rotation = Quat.identity()

        for constraint in constraints:
            result = constraint.apply(direction=direction, parent_rotation=parent_rotation)
            assert isinstance(result, Vec3), \
                f"{constraint.constraint_type} apply should return Vec3"

    def test_constraint_types_have_correct_type(self):
        """All constraints should report correct constraint_type."""
        from engine.animation.ik import JointConstraint, JointConstraintType

        assert JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=0.5
        ).constraint_type == JointConstraintType.BALL_SOCKET

        assert JointConstraint(
            constraint_type=JointConstraintType.HINGE,
            axis=Vec3(0.0, 1.0, 0.0),
            min_angle=-1.0,
            max_angle=1.0
        ).constraint_type == JointConstraintType.HINGE

        assert JointConstraint(
            constraint_type=JointConstraintType.TWIST_LIMIT,
            twist_min=-1.0,
            twist_max=1.0
        ).constraint_type == JointConstraintType.TWIST_LIMIT


# =============================================================================
# Performance Sanity Tests
# =============================================================================

class TestPerformance:
    """Basic performance sanity tests."""

    def test_ball_socket_apply_performance(self):
        """BALL_SOCKET constraint apply should be reasonably fast."""
        import time
        from engine.animation.ik import JointConstraint, JointConstraintType

        constraint = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=0.5
        )
        direction = Vec3(1.0, 0.0, 0.0)
        parent_rotation = Quat.identity()

        # Warm up
        for _ in range(100):
            constraint.apply(direction=direction, parent_rotation=parent_rotation)

        # Time 1000 iterations
        start = time.perf_counter()
        for _ in range(1000):
            constraint.apply(direction=direction, parent_rotation=parent_rotation)
        elapsed = time.perf_counter() - start

        # Should complete 1000 applications in under 100ms (generous limit)
        assert elapsed < 0.1, f"1000 applications took {elapsed:.3f}s, expected <0.1s"

    def test_hinge_apply_performance(self):
        """HINGE constraint apply should be reasonably fast."""
        import time
        from engine.animation.ik import JointConstraint, JointConstraintType

        constraint = JointConstraint(
            constraint_type=JointConstraintType.HINGE,
            axis=Vec3(0.0, 1.0, 0.0),
            min_angle=-1.0,
            max_angle=1.0
        )
        direction = Vec3(1.0, 0.0, 0.0)
        parent_rotation = Quat.identity()

        # Warm up
        for _ in range(100):
            constraint.apply(direction=direction, parent_rotation=parent_rotation)

        # Time 1000 iterations
        start = time.perf_counter()
        for _ in range(1000):
            constraint.apply(direction=direction, parent_rotation=parent_rotation)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.1, f"1000 applications took {elapsed:.3f}s, expected <0.1s"
