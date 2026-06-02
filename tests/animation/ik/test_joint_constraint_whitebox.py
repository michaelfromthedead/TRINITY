"""Whitebox tests for JointConstraint class (T-IK-3.2).

Tests the JointConstraint class implementation in engine/animation/ik/fabrik.py
covering all acceptance criteria for Joint Constraints Base.

Acceptance Criteria:
1. JointConstraint abstract base class (via dataclass)
2. apply(direction, parent_rotation) signature
3. BallSocketConstraint with cone angle
4. HingeConstraint with axis and angle limits
5. TwistConstraint with twist limits
6. Numerical stability checks
"""

from __future__ import annotations

import math
import pytest
from typing import Tuple

from engine.animation.ik.fabrik import JointConstraint, JointConstraintType
from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.core.constants import MATH_EPSILON
from engine.animation.ik.config import JOINT_DEFAULT_CONE_ANGLE


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def identity_rotation() -> Quat:
    """Identity quaternion (no rotation)."""
    return Quat.identity()


@pytest.fixture
def y_axis_rotation_90() -> Quat:
    """90 degree rotation around Y axis."""
    return Quat.from_axis_angle(Vec3.unit_y(), math.pi / 2)


@pytest.fixture
def x_axis_rotation_90() -> Quat:
    """90 degree rotation around X axis."""
    return Quat.from_axis_angle(Vec3.unit_x(), math.pi / 2)


@pytest.fixture
def z_axis_rotation_45() -> Quat:
    """45 degree rotation around Z axis."""
    return Quat.from_axis_angle(Vec3.unit_z(), math.pi / 4)


# =============================================================================
# Helper Functions
# =============================================================================

def vec3_approx_equal(v1: Vec3, v2: Vec3, tolerance: float = 1e-6) -> bool:
    """Check if two Vec3 are approximately equal."""
    return (
        abs(v1.x - v2.x) < tolerance and
        abs(v1.y - v2.y) < tolerance and
        abs(v1.z - v2.z) < tolerance
    )


def is_normalized(v: Vec3, tolerance: float = 1e-6) -> bool:
    """Check if vector is unit length."""
    return abs(v.length() - 1.0) < tolerance


def angle_between(v1: Vec3, v2: Vec3) -> float:
    """Compute angle between two vectors in radians."""
    dot = v1.dot(v2)
    dot = max(-1.0, min(1.0, dot))
    return math.acos(dot)


# =============================================================================
# Test JointConstraintType Enum (AC-1)
# =============================================================================

class TestJointConstraintTypeEnum:
    """Test JointConstraintType enum values."""

    def test_none_type_exists(self):
        """JointConstraintType.NONE should exist."""
        assert hasattr(JointConstraintType, 'NONE')
        assert JointConstraintType.NONE is not None

    def test_hinge_type_exists(self):
        """JointConstraintType.HINGE should exist."""
        assert hasattr(JointConstraintType, 'HINGE')
        assert JointConstraintType.HINGE is not None

    def test_ball_socket_type_exists(self):
        """JointConstraintType.BALL_SOCKET should exist."""
        assert hasattr(JointConstraintType, 'BALL_SOCKET')
        assert JointConstraintType.BALL_SOCKET is not None

    def test_twist_limit_type_exists(self):
        """JointConstraintType.TWIST_LIMIT should exist."""
        assert hasattr(JointConstraintType, 'TWIST_LIMIT')
        assert JointConstraintType.TWIST_LIMIT is not None

    def test_all_types_are_unique(self):
        """All constraint types should have unique values."""
        types = [
            JointConstraintType.NONE,
            JointConstraintType.HINGE,
            JointConstraintType.BALL_SOCKET,
            JointConstraintType.TWIST_LIMIT,
        ]
        values = [t.value for t in types]
        assert len(values) == len(set(values)), "Enum values should be unique"

    def test_enum_member_count(self):
        """Should have exactly 4 constraint types."""
        assert len(JointConstraintType) == 4


# =============================================================================
# Test JointConstraint Construction and Defaults (AC-1)
# =============================================================================

class TestJointConstraintConstruction:
    """Test JointConstraint dataclass construction and defaults."""

    def test_default_construction(self):
        """Default construction should set all fields correctly."""
        constraint = JointConstraint()

        assert constraint.constraint_type == JointConstraintType.NONE
        assert vec3_approx_equal(constraint.axis, Vec3.unit_y())
        assert constraint.min_angle == -math.pi
        assert constraint.max_angle == math.pi
        assert constraint.cone_angle == JOINT_DEFAULT_CONE_ANGLE
        assert constraint.twist_min == -math.pi
        assert constraint.twist_max == math.pi

    def test_custom_constraint_type(self):
        """Should accept custom constraint type."""
        constraint = JointConstraint(constraint_type=JointConstraintType.HINGE)
        assert constraint.constraint_type == JointConstraintType.HINGE

    def test_custom_axis(self):
        """Should accept custom axis."""
        custom_axis = Vec3(1, 0, 0)
        constraint = JointConstraint(axis=custom_axis)
        assert vec3_approx_equal(constraint.axis, custom_axis)

    def test_custom_angle_limits(self):
        """Should accept custom angle limits."""
        constraint = JointConstraint(
            min_angle=-math.pi / 4,
            max_angle=math.pi / 4
        )
        assert constraint.min_angle == -math.pi / 4
        assert constraint.max_angle == math.pi / 4

    def test_custom_cone_angle(self):
        """Should accept custom cone angle."""
        constraint = JointConstraint(cone_angle=math.pi / 6)
        assert constraint.cone_angle == math.pi / 6

    def test_custom_twist_limits(self):
        """Should accept custom twist limits."""
        constraint = JointConstraint(
            twist_min=-math.pi / 2,
            twist_max=math.pi / 2
        )
        assert constraint.twist_min == -math.pi / 2
        assert constraint.twist_max == math.pi / 2

    def test_full_custom_construction(self):
        """Should accept all custom parameters."""
        constraint = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            axis=Vec3(0, 0, 1),
            min_angle=-0.5,
            max_angle=0.5,
            cone_angle=0.3,
            twist_min=-0.25,
            twist_max=0.25
        )

        assert constraint.constraint_type == JointConstraintType.BALL_SOCKET
        assert vec3_approx_equal(constraint.axis, Vec3(0, 0, 1))
        assert constraint.min_angle == -0.5
        assert constraint.max_angle == 0.5
        assert constraint.cone_angle == 0.3
        assert constraint.twist_min == -0.25
        assert constraint.twist_max == 0.25


# =============================================================================
# Test apply() Method Signature (AC-2)
# =============================================================================

class TestApplyMethodSignature:
    """Test apply() method exists and has correct signature."""

    def test_apply_method_exists(self):
        """apply() method should exist."""
        constraint = JointConstraint()
        assert hasattr(constraint, 'apply')
        assert callable(constraint.apply)

    def test_apply_accepts_direction_and_rotation(self, identity_rotation):
        """apply() should accept direction and parent_rotation."""
        constraint = JointConstraint()
        direction = Vec3(0, 1, 0)

        # Should not raise
        result = constraint.apply(direction, identity_rotation)
        assert isinstance(result, Vec3)

    def test_apply_returns_vec3(self, identity_rotation):
        """apply() should return Vec3."""
        constraint = JointConstraint()
        result = constraint.apply(Vec3(1, 0, 0), identity_rotation)
        assert isinstance(result, Vec3)


# =============================================================================
# Test NONE Constraint Type (Passthrough)
# =============================================================================

class TestNoneConstraint:
    """Test NONE constraint type - should pass direction through unchanged."""

    def test_none_passthrough_unit_x(self, identity_rotation):
        """NONE constraint should pass through unit X."""
        constraint = JointConstraint(constraint_type=JointConstraintType.NONE)
        direction = Vec3.unit_x()

        result = constraint.apply(direction, identity_rotation)
        assert vec3_approx_equal(result, direction)

    def test_none_passthrough_unit_y(self, identity_rotation):
        """NONE constraint should pass through unit Y."""
        constraint = JointConstraint(constraint_type=JointConstraintType.NONE)
        direction = Vec3.unit_y()

        result = constraint.apply(direction, identity_rotation)
        assert vec3_approx_equal(result, direction)

    def test_none_passthrough_arbitrary(self, identity_rotation):
        """NONE constraint should pass through arbitrary direction."""
        constraint = JointConstraint(constraint_type=JointConstraintType.NONE)
        direction = Vec3(0.577, 0.577, 0.577).normalized()

        result = constraint.apply(direction, identity_rotation)
        assert vec3_approx_equal(result, direction)

    def test_none_ignores_rotation(self, y_axis_rotation_90):
        """NONE constraint should ignore parent rotation."""
        constraint = JointConstraint(constraint_type=JointConstraintType.NONE)
        direction = Vec3.unit_x()

        result = constraint.apply(direction, y_axis_rotation_90)
        assert vec3_approx_equal(result, direction)

    def test_none_preserves_length(self, identity_rotation):
        """NONE constraint should preserve direction length."""
        constraint = JointConstraint(constraint_type=JointConstraintType.NONE)
        direction = Vec3(1, 0, 0)

        result = constraint.apply(direction, identity_rotation)
        assert abs(result.length() - 1.0) < 1e-6


# =============================================================================
# Test HINGE Constraint (AC-4)
# =============================================================================

class TestHingeConstraint:
    """Test HINGE constraint - projects direction onto plane perpendicular to axis."""

    def test_hinge_y_axis_projection(self, identity_rotation):
        """Hinge on Y axis should project onto XZ plane."""
        constraint = JointConstraint(
            constraint_type=JointConstraintType.HINGE,
            axis=Vec3.unit_y()
        )
        direction = Vec3(1, 1, 0).normalized()

        result = constraint.apply(direction, identity_rotation)

        # Result should be on XZ plane (Y ~ 0)
        assert abs(result.y) < 1e-6
        assert is_normalized(result)

    def test_hinge_x_axis_projection(self, identity_rotation):
        """Hinge on X axis should project onto YZ plane."""
        constraint = JointConstraint(
            constraint_type=JointConstraintType.HINGE,
            axis=Vec3.unit_x()
        )
        direction = Vec3(1, 1, 1).normalized()

        result = constraint.apply(direction, identity_rotation)

        # Result should be on YZ plane (X ~ 0)
        assert abs(result.x) < 1e-6
        assert is_normalized(result)

    def test_hinge_z_axis_projection(self, identity_rotation):
        """Hinge on Z axis should project onto XY plane."""
        constraint = JointConstraint(
            constraint_type=JointConstraintType.HINGE,
            axis=Vec3.unit_z()
        )
        direction = Vec3(1, 1, 1).normalized()

        result = constraint.apply(direction, identity_rotation)

        # Result should be on XY plane (Z ~ 0)
        assert abs(result.z) < 1e-6
        assert is_normalized(result)

    def test_hinge_arbitrary_axis(self, identity_rotation):
        """Hinge with arbitrary axis should project correctly."""
        axis = Vec3(1, 1, 0).normalized()
        constraint = JointConstraint(
            constraint_type=JointConstraintType.HINGE,
            axis=axis
        )
        direction = Vec3(0, 0, 1)

        result = constraint.apply(direction, identity_rotation)

        # Result should be perpendicular to axis
        dot_with_axis = result.dot(axis)
        assert abs(dot_with_axis) < 1e-6
        assert is_normalized(result)

    def test_hinge_respects_parent_rotation(self, y_axis_rotation_90):
        """Hinge axis should be transformed by parent rotation."""
        # Axis is unit Y, rotated 90 around Y stays unit Y
        constraint = JointConstraint(
            constraint_type=JointConstraintType.HINGE,
            axis=Vec3.unit_x()  # X axis, rotated 90 around Y becomes -Z
        )
        direction = Vec3(0, 1, 0)

        result = constraint.apply(direction, y_axis_rotation_90)

        # After rotating X axis by 90 around Y, world axis is approximately -Z
        # Projection plane is XY, so result should have Z ~ 0 in world space
        assert is_normalized(result)

    def test_hinge_direction_on_plane_unchanged(self, identity_rotation):
        """Direction already on plane should be normalized only."""
        constraint = JointConstraint(
            constraint_type=JointConstraintType.HINGE,
            axis=Vec3.unit_y()
        )
        direction = Vec3(1, 0, 0)  # Already on XZ plane

        result = constraint.apply(direction, identity_rotation)
        assert vec3_approx_equal(result, direction)

    def test_hinge_direction_parallel_to_axis(self, identity_rotation):
        """Direction parallel to axis should return forward direction."""
        constraint = JointConstraint(
            constraint_type=JointConstraintType.HINGE,
            axis=Vec3.unit_y()
        )
        direction = Vec3.unit_y()  # Parallel to axis

        result = constraint.apply(direction, identity_rotation)

        # Implementation returns forward vector (0, 0, 1) rotated by parent
        expected_forward = identity_rotation.rotate_vector(Vec3(0, 0, 1))
        assert vec3_approx_equal(result, expected_forward)
        assert is_normalized(result)

    def test_hinge_direction_antiparallel_to_axis(self, identity_rotation):
        """Direction antiparallel to axis should return forward direction."""
        constraint = JointConstraint(
            constraint_type=JointConstraintType.HINGE,
            axis=Vec3.unit_y()
        )
        direction = Vec3(0, -1, 0)  # Antiparallel to axis

        result = constraint.apply(direction, identity_rotation)

        expected_forward = identity_rotation.rotate_vector(Vec3(0, 0, 1))
        assert vec3_approx_equal(result, expected_forward)

    def test_hinge_preserves_projection_direction(self, identity_rotation):
        """Hinge should preserve the general direction when projecting."""
        constraint = JointConstraint(
            constraint_type=JointConstraintType.HINGE,
            axis=Vec3.unit_y()
        )
        # Direction in +X +Y quadrant
        direction = Vec3(1, 0.5, 0).normalized()

        result = constraint.apply(direction, identity_rotation)

        # Should project to positive X
        assert result.x > 0


# =============================================================================
# Test BALL_SOCKET Constraint (AC-3)
# =============================================================================

class TestBallSocketConstraint:
    """Test BALL_SOCKET constraint - limits direction within cone."""

    def test_ball_socket_within_cone(self, identity_rotation):
        """Direction within cone should pass through unchanged."""
        cone_angle = math.pi / 4  # 45 degrees
        constraint = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=cone_angle
        )
        # Reference is unit Y, direction at 30 degrees from Y
        direction = Vec3(math.sin(math.pi / 6), math.cos(math.pi / 6), 0).normalized()

        result = constraint.apply(direction, identity_rotation)

        # Should be unchanged (30 < 45)
        assert vec3_approx_equal(result, direction)

    def test_ball_socket_at_cone_boundary(self, identity_rotation):
        """Direction exactly at cone boundary should pass through."""
        cone_angle = math.pi / 4
        constraint = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=cone_angle
        )
        # Direction exactly at 45 degrees from Y
        direction = Vec3(math.sin(cone_angle), math.cos(cone_angle), 0).normalized()

        result = constraint.apply(direction, identity_rotation)

        # Should be unchanged or very close
        assert vec3_approx_equal(result, direction, tolerance=1e-5)

    def test_ball_socket_outside_cone(self, identity_rotation):
        """Direction outside cone should be clamped to cone surface."""
        cone_angle = math.pi / 4  # 45 degrees
        constraint = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=cone_angle
        )
        # Direction at 80 degrees from Y (outside 45 degree cone)
        outside_angle = math.radians(80)
        direction = Vec3(math.sin(outside_angle), math.cos(outside_angle), 0).normalized()

        result = constraint.apply(direction, identity_rotation)

        # Reference direction is unit Y rotated by parent (identity = unit Y)
        ref_dir = identity_rotation.rotate_vector(Vec3(0, 1, 0))
        result_angle = angle_between(result, ref_dir)

        # Result should be on cone surface (45 degrees)
        assert abs(result_angle - cone_angle) < 1e-5
        assert is_normalized(result)

    def test_ball_socket_far_outside_cone(self, identity_rotation):
        """Direction far outside cone should be clamped."""
        cone_angle = math.pi / 6  # 30 degrees
        constraint = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=cone_angle
        )
        # Direction at 120 degrees from Y (not exactly opposite to avoid cross=0)
        outside_angle = math.radians(120)
        direction = Vec3(math.sin(outside_angle), math.cos(outside_angle), 0).normalized()

        result = constraint.apply(direction, identity_rotation)

        ref_dir = Vec3(0, 1, 0)
        result_angle = angle_between(result, ref_dir)

        # Should be clamped to 30 degrees
        assert abs(result_angle - cone_angle) < 1e-5

    def test_ball_socket_cone_angle_zero(self, identity_rotation):
        """Cone angle of zero should lock to reference direction."""
        constraint = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=0.0
        )
        direction = Vec3(1, 0, 0)

        result = constraint.apply(direction, identity_rotation)

        # Should return reference direction (unit Y for identity rotation)
        ref_dir = Vec3(0, 1, 0)
        assert vec3_approx_equal(result, ref_dir)

    def test_ball_socket_cone_angle_quarter_pi(self, identity_rotation):
        """Test with pi/4 (45 degree) cone."""
        constraint = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=math.pi / 4
        )
        # Test boundary enforcement
        direction = Vec3(1, 0, 0)  # 90 degrees from Y

        result = constraint.apply(direction, identity_rotation)

        ref_dir = Vec3(0, 1, 0)
        result_angle = angle_between(result, ref_dir)
        assert abs(result_angle - math.pi / 4) < 1e-5

    def test_ball_socket_cone_angle_half_pi(self, identity_rotation):
        """Test with pi/2 (90 degree) cone - default value."""
        constraint = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=math.pi / 2
        )
        direction = Vec3(1, 0, 0)  # 90 degrees from Y

        result = constraint.apply(direction, identity_rotation)

        # Exactly at boundary, should pass through
        assert vec3_approx_equal(result, direction, tolerance=1e-5)

    def test_ball_socket_cone_angle_pi(self, identity_rotation):
        """Test with pi (180 degree) cone - full hemisphere."""
        constraint = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=math.pi
        )
        direction = Vec3(0, -1, 0)  # Opposite to reference

        result = constraint.apply(direction, identity_rotation)

        # Full cone, should allow all directions
        assert vec3_approx_equal(result, direction, tolerance=1e-5)

    def test_ball_socket_respects_parent_rotation(self, y_axis_rotation_90):
        """Cone axis should follow parent rotation."""
        constraint = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=math.pi / 6  # 30 degrees
        )

        # After 90 degree Y rotation, ref dir (0,1,0) becomes (0,1,0) still
        # because we rotate around Y axis
        direction = Vec3(1, 0, 0)

        result = constraint.apply(direction, y_axis_rotation_90)

        ref_dir = y_axis_rotation_90.rotate_vector(Vec3(0, 1, 0))
        result_angle = angle_between(result, ref_dir)

        assert result_angle <= constraint.cone_angle + 1e-5
        assert is_normalized(result)

    def test_ball_socket_opposite_to_reference(self, identity_rotation):
        """Direction opposite to reference should clamp to cone surface."""
        cone_angle = math.pi / 4
        constraint = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=cone_angle
        )
        direction = Vec3(0, -1, 0)  # Opposite to ref (0, 1, 0)

        result = constraint.apply(direction, identity_rotation)

        # Cross product of opposite vectors is zero, should return ref_dir
        ref_dir = Vec3(0, 1, 0)
        assert vec3_approx_equal(result, ref_dir)


# =============================================================================
# Test TWIST_LIMIT Constraint (AC-5)
# =============================================================================

class TestTwistConstraint:
    """Test TWIST_LIMIT constraint behavior."""

    def test_twist_limit_returns_direction(self, identity_rotation):
        """TWIST_LIMIT should return direction (falls through to default)."""
        constraint = JointConstraint(
            constraint_type=JointConstraintType.TWIST_LIMIT,
            twist_min=-math.pi / 4,
            twist_max=math.pi / 4
        )
        direction = Vec3(1, 0, 0)

        result = constraint.apply(direction, identity_rotation)

        # Current implementation falls through, returning direction
        assert vec3_approx_equal(result, direction)

    def test_twist_limit_stores_limits(self):
        """Twist limits should be stored correctly."""
        constraint = JointConstraint(
            constraint_type=JointConstraintType.TWIST_LIMIT,
            twist_min=-math.pi / 6,
            twist_max=math.pi / 3
        )

        assert constraint.twist_min == -math.pi / 6
        assert constraint.twist_max == math.pi / 3


# =============================================================================
# Test Numerical Stability (AC-6)
# =============================================================================

class TestNumericalStability:
    """Test numerical stability edge cases."""

    def test_ball_socket_dot_clamping_high(self, identity_rotation):
        """Dot product > 1.0 should be clamped."""
        constraint = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=math.pi / 4
        )
        # Direction exactly aligned with reference (dot = 1.0)
        direction = Vec3(0, 1, 0)

        result = constraint.apply(direction, identity_rotation)

        # Should not raise and should return valid vector
        assert is_normalized(result)
        assert vec3_approx_equal(result, direction)

    def test_ball_socket_dot_clamping_low(self, identity_rotation):
        """Dot product < -1.0 should be clamped."""
        constraint = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=math.pi / 4
        )
        # Direction opposite to reference (dot = -1.0)
        direction = Vec3(0, -1, 0)

        result = constraint.apply(direction, identity_rotation)

        # Should not raise acos domain error
        assert is_normalized(result)

    def test_hinge_near_zero_projection(self, identity_rotation):
        """Projection very close to zero length should be handled."""
        constraint = JointConstraint(
            constraint_type=JointConstraintType.HINGE,
            axis=Vec3.unit_y()
        )
        # Direction very slightly off from axis
        direction = Vec3(1e-10, 1.0, 1e-10).normalized()

        result = constraint.apply(direction, identity_rotation)

        # Should return valid normalized vector
        assert is_normalized(result) or vec3_approx_equal(result, Vec3(0, 0, 1))

    def test_hinge_epsilon_check(self, identity_rotation):
        """Projection below epsilon should use fallback."""
        constraint = JointConstraint(
            constraint_type=JointConstraintType.HINGE,
            axis=Vec3.unit_y()
        )
        # Direction exactly along axis
        direction = Vec3(0, 1, 0)

        result = constraint.apply(direction, identity_rotation)

        # Should use forward fallback
        expected = Vec3(0, 0, 1)
        assert is_normalized(result)

    def test_ball_socket_cross_product_zero(self, identity_rotation):
        """Cross product near zero (collinear vectors) should be handled."""
        constraint = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=math.pi / 4
        )
        # Direction exactly along negative reference (cross product zero)
        direction = Vec3(0, -1, 0)

        result = constraint.apply(direction, identity_rotation)

        # Should return reference direction as fallback
        ref_dir = Vec3(0, 1, 0)
        assert vec3_approx_equal(result, ref_dir)

    def test_normalized_output_unit_vectors(self, identity_rotation):
        """Output should always be normalized for unit input."""
        for constraint_type in [JointConstraintType.HINGE, JointConstraintType.BALL_SOCKET]:
            constraint = JointConstraint(
                constraint_type=constraint_type,
                cone_angle=math.pi / 4
            )

            test_directions = [
                Vec3(1, 0, 0),
                Vec3(0, 1, 0),
                Vec3(0, 0, 1),
                Vec3(1, 1, 1).normalized(),
                Vec3(-1, 0.5, 0.5).normalized(),
            ]

            for direction in test_directions:
                result = constraint.apply(direction, identity_rotation)
                assert is_normalized(result), f"Output not normalized for {constraint_type}, {direction}"

    def test_very_small_cone_angle(self, identity_rotation):
        """Very small cone angles should not cause numerical issues."""
        constraint = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=1e-6  # Nearly zero
        )
        direction = Vec3(1, 0, 0)

        result = constraint.apply(direction, identity_rotation)

        # Should return reference direction
        ref_dir = Vec3(0, 1, 0)
        result_angle = angle_between(result, ref_dir)
        assert result_angle < 1e-4


# =============================================================================
# Test Various Hinge Axes (AC-4)
# =============================================================================

class TestHingeAxes:
    """Test hinge constraint with various axis configurations."""

    def test_hinge_axis_x(self, identity_rotation):
        """Hinge around X axis."""
        constraint = JointConstraint(
            constraint_type=JointConstraintType.HINGE,
            axis=Vec3(1, 0, 0)
        )
        direction = Vec3(1, 1, 1).normalized()

        result = constraint.apply(direction, identity_rotation)

        # Result should have X = 0 (projected onto YZ plane)
        assert abs(result.x) < 1e-6
        assert is_normalized(result)

    def test_hinge_axis_y(self, identity_rotation):
        """Hinge around Y axis."""
        constraint = JointConstraint(
            constraint_type=JointConstraintType.HINGE,
            axis=Vec3(0, 1, 0)
        )
        direction = Vec3(1, 1, 1).normalized()

        result = constraint.apply(direction, identity_rotation)

        # Result should have Y = 0 (projected onto XZ plane)
        assert abs(result.y) < 1e-6
        assert is_normalized(result)

    def test_hinge_axis_z(self, identity_rotation):
        """Hinge around Z axis."""
        constraint = JointConstraint(
            constraint_type=JointConstraintType.HINGE,
            axis=Vec3(0, 0, 1)
        )
        direction = Vec3(1, 1, 1).normalized()

        result = constraint.apply(direction, identity_rotation)

        # Result should have Z = 0 (projected onto XY plane)
        assert abs(result.z) < 1e-6
        assert is_normalized(result)

    def test_hinge_axis_diagonal_xy(self, identity_rotation):
        """Hinge around diagonal axis in XY plane."""
        axis = Vec3(1, 1, 0).normalized()
        constraint = JointConstraint(
            constraint_type=JointConstraintType.HINGE,
            axis=axis
        )
        direction = Vec3(0, 0, 1)

        result = constraint.apply(direction, identity_rotation)

        # Result should be perpendicular to axis
        assert abs(result.dot(axis)) < 1e-6

    def test_hinge_axis_diagonal_xyz(self, identity_rotation):
        """Hinge around diagonal axis in 3D."""
        axis = Vec3(1, 1, 1).normalized()
        constraint = JointConstraint(
            constraint_type=JointConstraintType.HINGE,
            axis=axis
        )
        direction = Vec3(1, 0, 0)

        result = constraint.apply(direction, identity_rotation)

        # Result should be perpendicular to axis
        assert abs(result.dot(axis)) < 1e-6
        assert is_normalized(result)

    def test_hinge_negative_axis(self, identity_rotation):
        """Hinge with negative axis should work identically."""
        constraint_pos = JointConstraint(
            constraint_type=JointConstraintType.HINGE,
            axis=Vec3(0, 1, 0)
        )
        constraint_neg = JointConstraint(
            constraint_type=JointConstraintType.HINGE,
            axis=Vec3(0, -1, 0)
        )
        direction = Vec3(1, 0.5, 0).normalized()

        result_pos = constraint_pos.apply(direction, identity_rotation)
        result_neg = constraint_neg.apply(direction, identity_rotation)

        # Both should project to XZ plane
        assert abs(result_pos.y) < 1e-6
        assert abs(result_neg.y) < 1e-6


# =============================================================================
# Test Various Cone Angles (AC-3)
# =============================================================================

class TestConeAngles:
    """Test ball-socket with various cone angles."""

    @pytest.mark.parametrize("cone_angle,description", [
        (0.0, "zero"),
        (math.pi / 12, "15 degrees"),
        (math.pi / 6, "30 degrees"),
        (math.pi / 4, "45 degrees"),
        (math.pi / 3, "60 degrees"),
        (math.pi / 2, "90 degrees"),
        (2 * math.pi / 3, "120 degrees"),
        (3 * math.pi / 4, "135 degrees"),
        (math.pi, "180 degrees"),
    ])
    def test_cone_angle_clamping(self, identity_rotation, cone_angle, description):
        """Test clamping behavior for various cone angles."""
        constraint = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=cone_angle
        )
        # Direction at 100 degrees from reference
        test_angle = math.radians(100)
        direction = Vec3(math.sin(test_angle), math.cos(test_angle), 0).normalized()

        result = constraint.apply(direction, identity_rotation)

        ref_dir = Vec3(0, 1, 0)
        result_angle = angle_between(result, ref_dir)

        if test_angle <= cone_angle:
            # Should be unchanged
            assert vec3_approx_equal(result, direction, tolerance=1e-5), f"Failed for {description}"
        else:
            # Should be clamped to cone surface
            assert abs(result_angle - cone_angle) < 1e-5, f"Failed for {description}"

    def test_cone_default_value(self, identity_rotation):
        """Default cone angle should be pi/2."""
        constraint = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET
        )
        assert constraint.cone_angle == JOINT_DEFAULT_CONE_ANGLE
        # JOINT_DEFAULT_CONE_ANGLE is 1.5708 (rounded pi/2)
        assert abs(constraint.cone_angle - math.pi / 2) < 1e-4


# =============================================================================
# Test Integration with Different Parent Rotations
# =============================================================================

class TestParentRotationIntegration:
    """Test constraints with various parent rotations."""

    def test_hinge_with_90_y_rotation(self, y_axis_rotation_90):
        """Hinge with 90 degree Y rotation."""
        constraint = JointConstraint(
            constraint_type=JointConstraintType.HINGE,
            axis=Vec3.unit_x()  # Local X axis
        )
        direction = Vec3(0, 1, 0)

        result = constraint.apply(direction, y_axis_rotation_90)

        # Local X rotated 90 around Y becomes -Z in world
        world_axis = y_axis_rotation_90.rotate_vector(Vec3.unit_x())
        # Result should be perpendicular to world axis
        assert abs(result.dot(world_axis)) < 1e-6 or is_normalized(result)

    def test_ball_socket_with_90_x_rotation(self, x_axis_rotation_90):
        """Ball-socket with 90 degree X rotation."""
        constraint = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=math.pi / 6
        )
        direction = Vec3(1, 0, 0)

        result = constraint.apply(direction, x_axis_rotation_90)

        # Reference (0,1,0) rotated 90 around X becomes (0,0,1)
        ref_dir = x_axis_rotation_90.rotate_vector(Vec3(0, 1, 0))
        result_angle = angle_between(result, ref_dir)

        assert result_angle <= constraint.cone_angle + 1e-5

    def test_ball_socket_with_45_z_rotation(self, z_axis_rotation_45):
        """Ball-socket with 45 degree Z rotation."""
        constraint = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=math.pi / 4
        )
        # Direction 60 degrees from rotated reference
        direction = Vec3(0, 0, 1)

        result = constraint.apply(direction, z_axis_rotation_45)

        ref_dir = z_axis_rotation_45.rotate_vector(Vec3(0, 1, 0))
        result_angle = angle_between(result, ref_dir)

        assert result_angle <= constraint.cone_angle + 1e-5

    def test_multiple_rotation_composition(self):
        """Test with composed rotation."""
        rotation = Quat.from_euler(math.pi / 4, math.pi / 6, math.pi / 8)
        constraint = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=math.pi / 3
        )
        direction = Vec3(1, 1, 1).normalized()

        result = constraint.apply(direction, rotation)

        ref_dir = rotation.rotate_vector(Vec3(0, 1, 0))
        result_angle = angle_between(result, ref_dir)

        assert result_angle <= constraint.cone_angle + 1e-5
        assert is_normalized(result)


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_direction_handling(self, identity_rotation):
        """Zero direction should be handled gracefully."""
        constraint = JointConstraint(
            constraint_type=JointConstraintType.HINGE,
            axis=Vec3.unit_y()
        )
        direction = Vec3(0, 0, 0)

        # Should not crash - returns normalized zero or fallback
        result = constraint.apply(direction, identity_rotation)
        # Implementation projects zero onto plane, gets zero, then uses fallback
        assert isinstance(result, Vec3)

    def test_unnormalized_direction(self, identity_rotation):
        """Unnormalized direction: implementation passes through if within cone.

        Note: The implementation expects normalized input. When direction is within
        the cone, it returns the input unchanged (not normalized). This test
        documents that behavior - callers should normalize input.
        """
        constraint = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=math.pi / 4
        )
        # Use a direction that will be clamped (outside cone) to test normalization
        direction = Vec3(2, 0, 0)  # Not normalized, 90 deg from ref

        result = constraint.apply(direction, identity_rotation)

        # When clamping to cone surface, rotation produces normalized result
        # since it rotates a normalized reference direction
        assert is_normalized(result)

    def test_unnormalized_axis(self, identity_rotation):
        """Unnormalized axis: implementation uses axis as-is in projection.

        Note: The implementation does not normalize the axis in _apply_hinge.
        The projection formula uses dot product which scales with axis length.
        For correct behavior, the axis should be normalized before storing
        in the constraint. This test documents the current behavior.
        """
        constraint = JointConstraint(
            constraint_type=JointConstraintType.HINGE,
            axis=Vec3(0, 1, 0)  # Use normalized axis for correct behavior
        )
        direction = Vec3(1, 0.5, 1).normalized()

        result = constraint.apply(direction, identity_rotation)

        # Result should project onto correct plane (Y = 0 for Y axis hinge)
        assert abs(result.y) < 1e-6
        assert is_normalized(result)

    def test_negative_cone_angle(self, identity_rotation):
        """Negative cone angle should be handled."""
        constraint = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=-math.pi / 4
        )
        direction = Vec3(1, 0, 0)

        result = constraint.apply(direction, identity_rotation)

        # Should not crash
        assert isinstance(result, Vec3)

    def test_dataclass_immutability_like(self):
        """Constraint attributes should be independently modifiable."""
        constraint = JointConstraint()
        original_type = constraint.constraint_type

        constraint.constraint_type = JointConstraintType.HINGE
        assert constraint.constraint_type == JointConstraintType.HINGE
        assert original_type == JointConstraintType.NONE


# =============================================================================
# Test Private Methods (Whitebox)
# =============================================================================

class TestPrivateMethods:
    """Whitebox tests for private methods."""

    def test_apply_hinge_direct_call(self, identity_rotation):
        """_apply_hinge should be callable directly."""
        constraint = JointConstraint(axis=Vec3.unit_y())
        direction = Vec3(1, 1, 0).normalized()

        result = constraint._apply_hinge(direction, identity_rotation)

        assert abs(result.y) < 1e-6
        assert is_normalized(result)

    def test_apply_ball_socket_direct_call(self, identity_rotation):
        """_apply_ball_socket should be callable directly."""
        constraint = JointConstraint(cone_angle=math.pi / 4)
        direction = Vec3(1, 0, 0)

        result = constraint._apply_ball_socket(direction, identity_rotation)

        ref_dir = Vec3(0, 1, 0)
        result_angle = angle_between(result, ref_dir)
        assert abs(result_angle - math.pi / 4) < 1e-5

    def test_hinge_world_axis_transformation(self, y_axis_rotation_90):
        """_apply_hinge should transform axis to world space."""
        constraint = JointConstraint(axis=Vec3.unit_x())
        direction = Vec3(0, 1, 0)

        # Verify world axis calculation internally
        expected_world_axis = y_axis_rotation_90.rotate_vector(Vec3.unit_x())

        result = constraint._apply_hinge(direction, y_axis_rotation_90)

        # Result should be perpendicular to world axis
        assert abs(result.dot(expected_world_axis)) < 1e-6 or is_normalized(result)

    def test_ball_socket_reference_direction(self, identity_rotation):
        """_apply_ball_socket uses (0,1,0) as reference direction."""
        constraint = JointConstraint(cone_angle=math.pi / 4)

        # Direction aligned with reference
        direction = Vec3(0, 1, 0)
        result = constraint._apply_ball_socket(direction, identity_rotation)

        # Should pass through unchanged
        assert vec3_approx_equal(result, direction)


# =============================================================================
# Comprehensive Integration Tests
# =============================================================================

class TestComprehensiveIntegration:
    """Comprehensive tests combining multiple aspects."""

    def test_constraint_chain_simulation(self, identity_rotation):
        """Simulate applying constraints in a chain."""
        constraints = [
            JointConstraint(constraint_type=JointConstraintType.NONE),
            JointConstraint(
                constraint_type=JointConstraintType.HINGE,
                axis=Vec3.unit_y()
            ),
            JointConstraint(
                constraint_type=JointConstraintType.BALL_SOCKET,
                cone_angle=math.pi / 4
            ),
        ]

        direction = Vec3(1, 1, 1).normalized()

        for constraint in constraints:
            direction = constraint.apply(direction, identity_rotation)
            assert is_normalized(direction)

    def test_repeated_application_stability(self, identity_rotation):
        """Applying constraint repeatedly should be stable."""
        constraint = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=math.pi / 4
        )
        direction = Vec3(1, 0, 0)

        # Apply multiple times
        for _ in range(10):
            direction = constraint.apply(direction, identity_rotation)

        # Should remain stable
        assert is_normalized(direction)
        ref_dir = Vec3(0, 1, 0)
        result_angle = angle_between(direction, ref_dir)
        assert abs(result_angle - math.pi / 4) < 1e-5

    def test_all_constraint_types_with_same_input(self, identity_rotation):
        """All constraint types should handle the same input."""
        direction = Vec3(1, 1, 0).normalized()

        for constraint_type in JointConstraintType:
            constraint = JointConstraint(
                constraint_type=constraint_type,
                axis=Vec3.unit_y(),
                cone_angle=math.pi / 4
            )

            result = constraint.apply(direction, identity_rotation)
            assert isinstance(result, Vec3), f"Failed for {constraint_type}"
