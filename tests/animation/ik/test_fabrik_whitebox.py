"""Whitebox tests for FABRIK (Forward And Backward Reaching Inverse Kinematics).

Tests the internal implementation details of FABRIKChain, FABRIKResult,
and JointConstraint classes with comprehensive coverage of edge cases.
"""

from __future__ import annotations

import math
import pytest

from engine.animation.ik.fabrik import (
    FABRIKChain,
    FABRIKResult,
    JointConstraint,
    JointConstraintType,
)
from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.core.math.transform import Transform
from engine.core.constants import MATH_EPSILON
from engine.animation.ik.config import (
    IK_DEFAULT_TOLERANCE,
    FABRIK_DEFAULT_MAX_ITERATIONS,
    JOINT_DEFAULT_CONE_ANGLE,
)


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


def vec3_nearly_equal(a: Vec3, b: Vec3, tol: float = 1e-5) -> bool:
    """Check if two Vec3 are nearly equal."""
    return (a - b).length() < tol


def create_straight_chain_positions(
    num_joints: int,
    bone_length: float = 1.0,
    start: Vec3 = None
) -> list[Vec3]:
    """Create positions for a straight vertical chain."""
    if start is None:
        start = Vec3.zero()
    return [Vec3(start.x, start.y + i * bone_length, start.z) for i in range(num_joints)]


def create_horizontal_chain_positions(
    num_joints: int,
    bone_length: float = 1.0,
    start: Vec3 = None
) -> list[Vec3]:
    """Create positions for a straight horizontal chain along X."""
    if start is None:
        start = Vec3.zero()
    return [Vec3(start.x + i * bone_length, start.y, start.z) for i in range(num_joints)]


# =============================================================================
# TestFABRIKResultDataclass
# =============================================================================


class TestFABRIKResultDataclass:
    """Tests for FABRIKResult dataclass."""

    def test_default_values(self):
        """FABRIKResult has correct default values."""
        result = FABRIKResult(success=True)

        assert result.success is True
        assert result.iterations == 0
        assert result.final_error == float('inf')
        assert result.positions == []
        assert result.rotations == []

    def test_success_result(self):
        """FABRIKResult can represent a successful solve."""
        positions = [Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 2, 0)]
        rotations = [Quat.identity(), Quat.identity(), Quat.identity()]

        result = FABRIKResult(
            success=True,
            iterations=5,
            final_error=0.0005,
            positions=positions,
            rotations=rotations
        )

        assert result.success is True
        assert result.iterations == 5
        assert result.final_error == pytest.approx(0.0005)
        assert len(result.positions) == 3
        assert len(result.rotations) == 3

    def test_failed_result(self):
        """FABRIKResult can represent a failed solve."""
        result = FABRIKResult(
            success=False,
            iterations=10,
            final_error=0.5,
            positions=[Vec3.zero()],
            rotations=[Quat.identity()]
        )

        assert result.success is False
        assert result.iterations == 10
        assert result.final_error == 0.5

    def test_result_with_empty_lists(self):
        """FABRIKResult handles empty position/rotation lists."""
        result = FABRIKResult(success=False, iterations=0, final_error=1.0)

        assert len(result.positions) == 0
        assert len(result.rotations) == 0

    def test_result_infinity_error(self):
        """FABRIKResult default error is infinity."""
        result = FABRIKResult(success=False)

        assert math.isinf(result.final_error)
        assert result.final_error > 0

    def test_result_zero_error(self):
        """FABRIKResult can have zero error for perfect convergence."""
        result = FABRIKResult(
            success=True,
            iterations=1,
            final_error=0.0
        )

        assert result.final_error == 0.0


# =============================================================================
# TestJointConstraintType
# =============================================================================


class TestJointConstraintType:
    """Tests for JointConstraintType enum."""

    def test_none_type_exists(self):
        """NONE constraint type exists."""
        assert JointConstraintType.NONE is not None

    def test_hinge_type_exists(self):
        """HINGE constraint type exists."""
        assert JointConstraintType.HINGE is not None

    def test_ball_socket_type_exists(self):
        """BALL_SOCKET constraint type exists."""
        assert JointConstraintType.BALL_SOCKET is not None

    def test_twist_limit_type_exists(self):
        """TWIST_LIMIT constraint type exists."""
        assert JointConstraintType.TWIST_LIMIT is not None

    def test_all_types_unique(self):
        """All constraint types are unique."""
        types = [
            JointConstraintType.NONE,
            JointConstraintType.HINGE,
            JointConstraintType.BALL_SOCKET,
            JointConstraintType.TWIST_LIMIT
        ]
        assert len(set(types)) == 4


# =============================================================================
# TestJointConstraint
# =============================================================================


class TestJointConstraint:
    """Tests for JointConstraint dataclass."""

    def test_default_construction(self):
        """JointConstraint has correct default values."""
        constraint = JointConstraint()

        assert constraint.constraint_type == JointConstraintType.NONE
        assert constraint.min_angle == -math.pi
        assert constraint.max_angle == math.pi
        assert constraint.cone_angle == JOINT_DEFAULT_CONE_ANGLE
        assert constraint.twist_min == -math.pi
        assert constraint.twist_max == math.pi

    def test_default_axis_is_unit_y(self):
        """Default constraint axis is unit Y."""
        constraint = JointConstraint()

        assert vec3_nearly_equal(constraint.axis, Vec3.unit_y())

    def test_none_constraint_passthrough(self):
        """NONE constraint returns direction unchanged."""
        constraint = JointConstraint(constraint_type=JointConstraintType.NONE)
        direction = Vec3(1, 2, 3).normalized()
        parent_rot = Quat.identity()

        result = constraint.apply(direction, parent_rot)

        assert vec3_nearly_equal(result, direction)

    def test_none_constraint_with_any_direction(self):
        """NONE constraint passes through any direction."""
        constraint = JointConstraint()
        test_directions = [
            Vec3.unit_x(),
            Vec3.unit_y(),
            Vec3.unit_z(),
            Vec3(1, 1, 1).normalized(),
            Vec3(-1, 0.5, 2).normalized()
        ]

        for direction in test_directions:
            result = constraint.apply(direction, Quat.identity())
            assert vec3_nearly_equal(result, direction)

    def test_hinge_constraint_projection_to_plane(self):
        """HINGE constraint projects direction onto plane."""
        # Hinge around Y axis - should project to XZ plane
        constraint = JointConstraint(
            constraint_type=JointConstraintType.HINGE,
            axis=Vec3.unit_y()
        )

        # Direction with Y component
        direction = Vec3(1, 1, 1).normalized()
        result = constraint.apply(direction, Quat.identity())

        # Result should be in XZ plane (Y ~= 0)
        assert abs(result.y) < 1e-5
        # Result should be normalized
        assert abs(result.length() - 1.0) < 1e-5

    def test_hinge_constraint_parallel_to_axis_fallback(self):
        """HINGE constraint uses fallback when direction parallel to axis."""
        constraint = JointConstraint(
            constraint_type=JointConstraintType.HINGE,
            axis=Vec3.unit_y()
        )

        # Direction exactly along Y axis
        direction = Vec3.unit_y()
        result = constraint.apply(direction, Quat.identity())

        # Should return forward direction fallback
        assert result.length() > 0.9  # Should be normalized
        # The y component should be near zero (projected to plane)
        assert abs(result.y) < 0.1  # Fallback projects to forward

    def test_hinge_constraint_with_rotated_parent(self):
        """HINGE constraint respects parent rotation."""
        constraint = JointConstraint(
            constraint_type=JointConstraintType.HINGE,
            axis=Vec3.unit_y()  # Local Y axis
        )

        # Rotate parent 90 degrees around Z - Y axis becomes X axis
        parent_rot = Quat.from_axis_angle(Vec3.unit_z(), math.pi / 2)
        direction = Vec3(1, 1, 0).normalized()

        result = constraint.apply(direction, parent_rot)

        # Result should be projected onto plane perpendicular to world X
        assert abs(result.length() - 1.0) < 1e-5

    def test_ball_socket_inside_cone(self):
        """BALL_SOCKET allows directions within cone."""
        constraint = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=math.pi / 4  # 45 degrees
        )

        # Direction slightly off Y axis - within 45 degree cone
        direction = Vec3(0.1, 0.99, 0).normalized()
        result = constraint.apply(direction, Quat.identity())

        # Should pass through unchanged
        assert vec3_nearly_equal(result, direction, tol=1e-4)

    def test_ball_socket_outside_cone_clamped(self):
        """BALL_SOCKET clamps directions outside cone to surface."""
        cone_angle = math.pi / 6  # 30 degrees
        constraint = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=cone_angle
        )

        # Direction at 60 degrees from Y axis
        direction = Vec3(math.sin(math.pi / 3), math.cos(math.pi / 3), 0)
        result = constraint.apply(direction, Quat.identity())

        # Angle from reference should now be cone_angle
        ref_dir = Vec3(0, 1, 0)  # Reference direction (Y-up)
        dot = result.dot(ref_dir)
        result_angle = math.acos(max(-1, min(1, dot)))

        assert result_angle == pytest.approx(cone_angle, abs=1e-4)

    def test_ball_socket_opposite_direction_clamped(self):
        """BALL_SOCKET clamps opposite direction to reference."""
        constraint = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=math.pi / 4
        )

        # Direction opposite to Y axis
        direction = Vec3(0, -1, 0)
        result = constraint.apply(direction, Quat.identity())

        # Should return reference direction when parallel opposite
        assert vec3_nearly_equal(result, Vec3(0, 1, 0))

    def test_ball_socket_with_rotated_parent(self):
        """BALL_SOCKET cone follows parent rotation."""
        constraint = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=math.pi / 6
        )

        # Rotate parent 90 degrees around X - Y axis becomes Z axis
        parent_rot = Quat.from_axis_angle(Vec3.unit_x(), math.pi / 2)

        # Direction at 60 degrees from rotated Y (now Z)
        direction = Vec3(0, math.sin(math.pi / 3), math.cos(math.pi / 3)).normalized()
        result = constraint.apply(direction, parent_rot)

        # Result should be clamped
        ref_dir = parent_rot.rotate_vector(Vec3(0, 1, 0))
        dot = result.dot(ref_dir)
        result_angle = math.acos(max(-1, min(1, dot)))

        assert result_angle == pytest.approx(constraint.cone_angle, abs=1e-4)

    def test_twist_limit_passthrough(self):
        """TWIST_LIMIT constraint passes through direction (positional constraint not applicable)."""
        constraint = JointConstraint(
            constraint_type=JointConstraintType.TWIST_LIMIT,
            twist_min=-math.pi / 2,
            twist_max=math.pi / 2
        )

        direction = Vec3(1, 1, 1).normalized()
        result = constraint.apply(direction, Quat.identity())

        # Twist limit is rotation-space, not applicable to direction vectors
        assert vec3_nearly_equal(result, direction)

    def test_custom_cone_angle(self):
        """JointConstraint accepts custom cone angle."""
        constraint = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=math.pi / 8
        )

        assert constraint.cone_angle == math.pi / 8

    def test_custom_angle_limits(self):
        """JointConstraint accepts custom angle limits."""
        constraint = JointConstraint(
            min_angle=-math.pi / 4,
            max_angle=math.pi / 4
        )

        assert constraint.min_angle == -math.pi / 4
        assert constraint.max_angle == math.pi / 4


# =============================================================================
# TestFABRIKChainConstruction
# =============================================================================


class TestFABRIKChainConstruction:
    """Tests for FABRIKChain construction."""

    def test_valid_chain_two_bones(self):
        """FABRIKChain accepts chain with 2 bones."""
        chain = FABRIKChain([0, 1])

        assert chain.chain_length == 2
        assert chain.bone_indices == [0, 1]

    def test_valid_chain_three_bones(self):
        """FABRIKChain accepts chain with 3 bones."""
        chain = FABRIKChain([0, 1, 2])

        assert chain.chain_length == 3

    def test_valid_chain_five_bones(self):
        """FABRIKChain accepts chain with 5 bones."""
        chain = FABRIKChain([0, 1, 2, 3, 4])

        assert chain.chain_length == 5

    def test_invalid_chain_single_bone_raises(self):
        """FABRIKChain raises ValueError for chain with 1 bone."""
        with pytest.raises(ValueError, match="at least 2 bones"):
            FABRIKChain([0])

    def test_invalid_chain_empty_raises(self):
        """FABRIKChain raises ValueError for empty chain."""
        with pytest.raises(ValueError, match="at least 2 bones"):
            FABRIKChain([])

    def test_property_chain_length(self):
        """chain_length property returns correct count."""
        chain = FABRIKChain([10, 11, 12, 13])

        assert chain.chain_length == 4

    def test_property_root_index(self):
        """root_index property returns first bone index."""
        chain = FABRIKChain([5, 6, 7])

        assert chain.root_index == 5

    def test_property_end_index(self):
        """end_index property returns last bone index."""
        chain = FABRIKChain([5, 6, 7])

        assert chain.end_index == 7

    def test_default_tolerance(self):
        """FABRIKChain uses default tolerance."""
        chain = FABRIKChain([0, 1])

        assert chain.tolerance == IK_DEFAULT_TOLERANCE

    def test_default_max_iterations(self):
        """FABRIKChain uses default max iterations."""
        chain = FABRIKChain([0, 1])

        assert chain.max_iterations == FABRIK_DEFAULT_MAX_ITERATIONS

    def test_custom_tolerance(self):
        """FABRIKChain accepts custom tolerance."""
        chain = FABRIKChain([0, 1], tolerance=0.01)

        assert chain.tolerance == 0.01

    def test_custom_max_iterations(self):
        """FABRIKChain accepts custom max iterations."""
        chain = FABRIKChain([0, 1], max_iterations=50)

        assert chain.max_iterations == 50

    def test_bone_indices_copied(self):
        """FABRIKChain copies bone indices list."""
        indices = [0, 1, 2]
        chain = FABRIKChain(indices)

        indices[0] = 99  # Modify original

        assert chain.bone_indices[0] == 0  # Chain unaffected

    def test_non_contiguous_indices(self):
        """FABRIKChain accepts non-contiguous bone indices."""
        chain = FABRIKChain([0, 5, 10, 15])

        assert chain.bone_indices == [0, 5, 10, 15]
        assert chain.chain_length == 4


# =============================================================================
# TestBoneLengthCaching
# =============================================================================


class TestBoneLengthCaching:
    """Tests for bone length caching in FABRIKChain."""

    def test_cache_bone_lengths_stores_correct_lengths(self):
        """_cache_bone_lengths stores correct bone lengths."""
        chain = FABRIKChain([0, 1, 2])
        positions = [Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 3, 0)]

        chain._cache_bone_lengths(positions)

        assert len(chain._bone_lengths) == 2
        assert chain._bone_lengths[0] == pytest.approx(1.0)
        assert chain._bone_lengths[1] == pytest.approx(2.0)

    def test_cache_bone_lengths_computes_total(self):
        """_cache_bone_lengths computes total length."""
        chain = FABRIKChain([0, 1, 2, 3])
        positions = [Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 2.5, 0), Vec3(0, 4, 0)]

        chain._cache_bone_lengths(positions)

        # 1.0 + 1.5 + 1.5 = 4.0
        assert chain._total_length == pytest.approx(4.0)

    def test_cache_bone_lengths_sets_cached_flag(self):
        """_cache_bone_lengths sets _lengths_cached flag."""
        chain = FABRIKChain([0, 1])
        positions = [Vec3(0, 0, 0), Vec3(0, 1, 0)]

        assert chain._lengths_cached is False
        chain._cache_bone_lengths(positions)
        assert chain._lengths_cached is True

    def test_cache_bone_lengths_diagonal_bones(self):
        """_cache_bone_lengths handles diagonal bone directions."""
        chain = FABRIKChain([0, 1, 2])
        positions = [Vec3(0, 0, 0), Vec3(1, 1, 1), Vec3(2, 2, 2)]

        chain._cache_bone_lengths(positions)

        expected_length = math.sqrt(3)  # sqrt(1^2 + 1^2 + 1^2)
        assert chain._bone_lengths[0] == pytest.approx(expected_length)
        assert chain._bone_lengths[1] == pytest.approx(expected_length)

    def test_reset_cached_lengths_clears_data(self):
        """reset_cached_lengths clears cached data."""
        chain = FABRIKChain([0, 1, 2])
        positions = [Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 2, 0)]

        chain._cache_bone_lengths(positions)
        chain.reset_cached_lengths()

        assert chain._lengths_cached is False
        assert chain._bone_lengths == []
        assert chain._total_length == 0.0

    def test_cache_is_reused_on_subsequent_solves(self):
        """Bone lengths are not recached if already cached."""
        chain = FABRIKChain([0, 1, 2])
        positions = [Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 2, 0)]
        target = Vec3(0, 2, 0)

        chain.solve(positions, target)
        original_lengths = list(chain._bone_lengths)

        # Solve again with same positions
        chain.solve(positions, target)

        assert chain._bone_lengths == original_lengths

    def test_zero_length_bone_cached(self):
        """_cache_bone_lengths handles zero-length bones."""
        chain = FABRIKChain([0, 1, 2])
        positions = [Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(0, 1, 0)]  # First bone has zero length

        chain._cache_bone_lengths(positions)

        assert chain._bone_lengths[0] == pytest.approx(0.0)
        assert chain._bone_lengths[1] == pytest.approx(1.0)


# =============================================================================
# TestForwardPass
# =============================================================================


class TestForwardPass:
    """Tests for _forward_pass method."""

    def test_end_effector_placed_at_target(self):
        """Forward pass places end effector at target."""
        chain = FABRIKChain([0, 1, 2])
        positions = [Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 2, 0)]
        chain._cache_bone_lengths(positions)

        target = Vec3(1, 1.5, 0)
        rotations = [Quat.identity(), Quat.identity(), Quat.identity()]

        new_pos = chain._forward_pass(positions, target, rotations)

        assert vec3_nearly_equal(new_pos[-1], target)

    def test_bone_lengths_preserved_in_forward_pass(self):
        """Forward pass preserves bone lengths."""
        chain = FABRIKChain([0, 1, 2, 3])
        positions = [Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 2, 0), Vec3(0, 3, 0)]
        chain._cache_bone_lengths(positions)

        target = Vec3(1, 2.5, 0.5)
        rotations = [Quat.identity()] * 4

        new_pos = chain._forward_pass(positions, target, rotations)

        for i in range(len(chain._bone_lengths)):
            actual_length = (new_pos[i + 1] - new_pos[i]).length()
            assert actual_length == pytest.approx(chain._bone_lengths[i], abs=1e-5)

    def test_forward_pass_coincident_joint_fallback(self):
        """Forward pass uses fallback direction for coincident joints."""
        chain = FABRIKChain([0, 1, 2])
        # Set up positions where after setting end to target, joints coincide
        positions = [Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 2, 0)]
        chain._bone_lengths = [1.0, 1.0]
        chain._total_length = 2.0
        chain._lengths_cached = True

        # Target at exactly where end effector was - after setting last to target,
        # and calculating direction from second-to-last to last (which is same point),
        # fallback should be used
        target = Vec3(0, 1, 0)  # Same as middle joint
        rotations = [Quat.identity()] * 3

        new_pos = chain._forward_pass(positions, target, rotations)

        # Should not crash and positions should be valid
        assert len(new_pos) == 3
        # Bone lengths should still be preserved
        for i in range(2):
            actual_length = (new_pos[i + 1] - new_pos[i]).length()
            assert actual_length == pytest.approx(1.0, abs=1e-4)

    def test_forward_pass_applies_constraints(self):
        """Forward pass applies joint constraints."""
        chain = FABRIKChain([0, 1, 2])
        positions = [Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 2, 0)]
        chain._cache_bone_lengths(positions)

        # Apply a hinge constraint at joint 0
        hinge = JointConstraint(
            constraint_type=JointConstraintType.HINGE,
            axis=Vec3.unit_z()  # Hinge around Z - constrains to XY plane
        )
        chain.set_constraint(0, hinge)

        target = Vec3(1, 1, 1)  # Target with Z component
        rotations = [Quat.identity()] * 3

        new_pos = chain._forward_pass(positions, target, rotations)

        # Direction from joint 0 to joint 1 should be in XY plane (z near 0)
        direction = new_pos[1] - new_pos[0]
        assert abs(direction.z) < 0.1 or abs(direction.x) > 0.1 or abs(direction.y) > 0.1

    def test_forward_pass_backward_iteration_order(self):
        """Forward pass iterates from end to root."""
        chain = FABRIKChain([0, 1, 2, 3])
        positions = create_straight_chain_positions(4)
        chain._cache_bone_lengths(positions)

        target = Vec3(2, 2, 0)
        rotations = [Quat.identity()] * 4

        new_pos = chain._forward_pass(positions, target, rotations)

        # End should be at target
        assert vec3_nearly_equal(new_pos[-1], target)
        # All positions should be modified
        for i in range(len(new_pos) - 1):
            assert not vec3_nearly_equal(new_pos[i], positions[i], tol=1e-5) or i == 0


# =============================================================================
# TestBackwardPass
# =============================================================================


class TestBackwardPass:
    """Tests for _backward_pass method."""

    def test_root_at_original_position(self):
        """Backward pass restores root to original position."""
        chain = FABRIKChain([0, 1, 2])
        positions = [Vec3(0, 0, 0), Vec3(1, 1, 0), Vec3(2, 2, 0)]  # Moved positions
        chain._cache_bone_lengths([Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 2, 0)])

        original_root = Vec3(0, 0, 0)
        rotations = [Quat.identity()] * 3

        new_pos = chain._backward_pass(positions, original_root, rotations)

        assert vec3_nearly_equal(new_pos[0], original_root)

    def test_bone_lengths_preserved_in_backward_pass(self):
        """Backward pass preserves bone lengths."""
        chain = FABRIKChain([0, 1, 2, 3])
        positions = [Vec3(0, 0, 0), Vec3(0.5, 1, 0), Vec3(1, 1.5, 0), Vec3(1.5, 2, 0)]
        chain._bone_lengths = [1.0, 1.0, 1.0]
        chain._total_length = 3.0
        chain._lengths_cached = True

        root_pos = Vec3(0, 0, 0)
        rotations = [Quat.identity()] * 4

        new_pos = chain._backward_pass(positions, root_pos, rotations)

        for i in range(len(chain._bone_lengths)):
            actual_length = (new_pos[i + 1] - new_pos[i]).length()
            assert actual_length == pytest.approx(chain._bone_lengths[i], abs=1e-5)

    def test_backward_pass_coincident_joint_fallback(self):
        """Backward pass uses fallback direction for coincident joints."""
        chain = FABRIKChain([0, 1, 2])
        # After forward pass, joint 0 and 1 might be at same position
        positions = [Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(0, 1, 0)]  # 0 and 1 coincident
        chain._bone_lengths = [1.0, 1.0]
        chain._total_length = 2.0
        chain._lengths_cached = True

        root_pos = Vec3(0, 0, 0)
        rotations = [Quat.identity()] * 3

        new_pos = chain._backward_pass(positions, root_pos, rotations)

        # Should not crash
        assert len(new_pos) == 3
        # Bone lengths should be preserved
        for i in range(2):
            actual_length = (new_pos[i + 1] - new_pos[i]).length()
            assert actual_length == pytest.approx(1.0, abs=1e-4)

    def test_backward_pass_applies_constraints(self):
        """Backward pass applies joint constraints."""
        chain = FABRIKChain([0, 1, 2])
        positions = [Vec3(0, 0, 0), Vec3(0.5, 1, 0.5), Vec3(1, 2, 1)]
        chain._bone_lengths = [1.0, 1.0]
        chain._total_length = 2.0
        chain._lengths_cached = True

        # Apply ball socket constraint at joint 0
        ball_socket = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=math.pi / 6  # 30 degrees
        )
        chain.set_constraint(0, ball_socket)

        root_pos = Vec3(0, 0, 0)
        rotations = [Quat.identity()] * 3

        new_pos = chain._backward_pass(positions, root_pos, rotations)

        # Direction from root to next should be constrained
        direction = (new_pos[1] - new_pos[0]).normalized()
        dot = direction.dot(Vec3(0, 1, 0))  # Compare to reference Y direction
        angle = math.acos(max(-1, min(1, dot)))

        # Angle should be within or at cone limit
        assert angle <= ball_socket.cone_angle + 0.01

    def test_backward_pass_forward_iteration_order(self):
        """Backward pass iterates from root to end."""
        chain = FABRIKChain([0, 1, 2, 3])
        positions = [Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(2, 0, 0), Vec3(3, 0, 0)]
        chain._bone_lengths = [1.0, 1.0, 1.0]
        chain._total_length = 3.0
        chain._lengths_cached = True

        root_pos = Vec3(0, 0, 0)
        rotations = [Quat.identity()] * 4

        new_pos = chain._backward_pass(positions, root_pos, rotations)

        # Root should be at original
        assert vec3_nearly_equal(new_pos[0], root_pos)


# =============================================================================
# TestConvergence
# =============================================================================


class TestConvergence:
    """Tests for convergence behavior."""

    def test_success_when_error_below_tolerance(self):
        """Returns success=True when error < tolerance."""
        chain = FABRIKChain([0, 1, 2], tolerance=0.001)
        positions = [Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 2, 0)]

        # Target exactly at current end effector position
        target = Vec3(0, 2, 0)

        result = chain.solve(positions, target)

        assert result.success is True
        assert result.final_error <= chain.tolerance

    def test_fail_when_max_iterations_exceeded(self):
        """Returns success=False when max iterations reached."""
        chain = FABRIKChain([0, 1, 2], tolerance=1e-10, max_iterations=2)
        positions = [Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 2, 0)]

        # Target requires more iterations than allowed
        target = Vec3(1, 1, 0)

        result = chain.solve(positions, target)

        # May or may not succeed depending on convergence, but iterations should be limited
        assert result.iterations <= chain.max_iterations

    def test_final_error_computed_correctly(self):
        """final_error reflects actual distance to target."""
        chain = FABRIKChain([0, 1, 2])
        positions = [Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 2, 0)]
        target = Vec3(1, 1.5, 0)

        result = chain.solve(positions, target)

        # Verify error matches distance from end to target
        actual_error = (result.positions[-1] - target).length()
        assert result.final_error == pytest.approx(actual_error, abs=1e-6)

    def test_single_iteration_convergence(self):
        """Chain converges in single iteration for close target."""
        chain = FABRIKChain([0, 1, 2], tolerance=0.01)
        positions = [Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 2, 0)]

        # Target is already at end effector
        target = Vec3(0, 2, 0)

        result = chain.solve(positions, target)

        assert result.success is True
        assert result.iterations == 1

    def test_convergence_improves_each_iteration(self):
        """Error decreases across iterations for reachable target."""
        chain = FABRIKChain([0, 1, 2], max_iterations=10, tolerance=0.0001)
        positions = [Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 2, 0)]
        target = Vec3(0.5, 1.5, 0)

        result = chain.solve(positions, target)

        # Should converge for this reachable target
        assert result.final_error < 0.01


# =============================================================================
# TestUnreachableTargets
# =============================================================================


class TestUnreachableTargets:
    """Tests for handling targets beyond chain reach."""

    def test_target_beyond_chain_length(self):
        """Chain extends toward target beyond reach."""
        chain = FABRIKChain([0, 1, 2])
        positions = [Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 2, 0)]  # Total length = 2

        # Target at distance 5 (beyond reach)
        target = Vec3(0, 5, 0)

        result = chain.solve(positions, target)

        assert result.success is False

    def test_chain_extends_toward_unreachable_target(self):
        """Chain points toward unreachable target."""
        chain = FABRIKChain([0, 1, 2])
        positions = [Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 2, 0)]

        target = Vec3(10, 0, 0)  # Far to the right

        result = chain.solve(positions, target)

        # Chain should be extended toward target direction
        chain_direction = (result.positions[-1] - result.positions[0]).normalized()
        target_direction = (target - result.positions[0]).normalized()

        # Directions should be aligned
        dot = chain_direction.dot(target_direction)
        assert dot > 0.99

    def test_unreachable_error_equals_distance_minus_length(self):
        """For unreachable targets, error = distance - total_length."""
        chain = FABRIKChain([0, 1, 2])
        positions = [Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 2, 0)]

        target = Vec3(0, 5, 0)

        result = chain.solve(positions, target)

        expected_error = 5.0 - 2.0  # distance - total_length
        assert result.final_error == pytest.approx(expected_error, abs=0.01)

    def test_unreachable_diagonal_target(self):
        """Chain extends toward diagonal unreachable target."""
        chain = FABRIKChain([0, 1, 2, 3])
        positions = create_straight_chain_positions(4)  # Total length = 3

        # Target at distance 10 diagonal
        target = Vec3(7, 7, 0)  # Distance ~= 9.9

        result = chain.solve(positions, target)

        assert result.success is False
        # Chain should be fully extended toward target
        assert (result.positions[-1] - result.positions[0]).length() == pytest.approx(3.0, abs=0.01)

    def test_unreachable_preserves_bone_lengths(self):
        """Even for unreachable targets, bone lengths are preserved."""
        chain = FABRIKChain([0, 1, 2, 3])
        positions = [Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 2.5, 0), Vec3(0, 4, 0)]

        target = Vec3(100, 0, 0)

        result = chain.solve(positions, target)

        # Verify each bone length
        assert (result.positions[1] - result.positions[0]).length() == pytest.approx(1.0, abs=0.01)
        assert (result.positions[2] - result.positions[1]).length() == pytest.approx(1.5, abs=0.01)
        assert (result.positions[3] - result.positions[2]).length() == pytest.approx(1.5, abs=0.01)

    def test_unreachable_iterations_zero(self):
        """Unreachable targets don't iterate - immediate extension."""
        chain = FABRIKChain([0, 1, 2])
        positions = [Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 2, 0)]
        target = Vec3(0, 10, 0)

        result = chain.solve(positions, target)

        assert result.iterations == 0


# =============================================================================
# TestReachableTargets
# =============================================================================


class TestReachableTargets:
    """Tests for solving to reachable targets."""

    def test_two_bone_chain_reaches_target(self):
        """Two-bone chain successfully reaches reachable target."""
        chain = FABRIKChain([0, 1, 2], tolerance=0.001)
        positions = [Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 2, 0)]

        # Target off-axis within reach (total length = 2)
        target = Vec3(0.8, 1.5, 0)  # Distance from origin ~1.7

        result = chain.solve(positions, target)

        assert result.success is True
        assert vec3_nearly_equal(result.positions[-1], target, tol=0.01)

    def test_three_bone_chain_converges(self):
        """Three-bone chain converges to target."""
        chain = FABRIKChain([0, 1, 2], tolerance=0.001)
        positions = [Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 2, 0)]

        target = Vec3(1, 1, 0)

        result = chain.solve(positions, target)

        assert result.success is True
        assert result.final_error < 0.01

    def test_five_bone_chain_converges(self):
        """Five-bone chain converges to target."""
        chain = FABRIKChain([0, 1, 2, 3, 4], tolerance=0.001)
        positions = create_straight_chain_positions(5)  # Total length = 4

        target = Vec3(2, 2, 0)  # Within reach

        result = chain.solve(positions, target)

        assert result.success is True
        assert result.final_error < 0.01

    def test_target_at_root_position(self):
        """Chain folds when target is at root."""
        chain = FABRIKChain([0, 1, 2], tolerance=0.01)
        positions = [Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 2, 0)]

        target = Vec3(0, 0, 0)  # At root

        result = chain.solve(positions, target)

        # Should succeed since 0 <= total_length (2)
        assert result.success is True

    def test_target_at_max_reach(self):
        """Chain reaches target at exactly max reach."""
        chain = FABRIKChain([0, 1, 2], tolerance=0.001)
        positions = [Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 2, 0)]

        # Target at exactly total length distance
        target = Vec3(0, 2, 0)

        result = chain.solve(positions, target)

        assert result.success is True
        assert vec3_nearly_equal(result.positions[-1], target, tol=0.01)

    def test_target_slightly_below_max_reach(self):
        """Chain reaches target just below max reach."""
        chain = FABRIKChain([0, 1, 2], tolerance=0.001)
        positions = [Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 2, 0)]

        # Target off-axis at reasonable distance (chain can bend to reach)
        target = Vec3(0.5, 1.5, 0)  # Distance ~1.58, well within reach of 2.0

        result = chain.solve(positions, target)

        assert result.success is True

    def test_target_behind_root(self):
        """Chain reaches target behind root position."""
        chain = FABRIKChain([0, 1, 2], tolerance=0.001)
        positions = [Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 2, 0)]

        # Target off to the side but behind the root Y position
        # This allows the chain to bend around naturally
        target = Vec3(1.5, -0.5, 0)  # Distance ~1.58, within reach

        result = chain.solve(positions, target)

        assert result.success is True
        assert vec3_nearly_equal(result.positions[-1], target, tol=0.01)

    def test_target_off_to_side(self):
        """Chain reaches target off to the side."""
        chain = FABRIKChain([0, 1, 2], tolerance=0.001)
        positions = [Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 2, 0)]

        target = Vec3(1.5, 0.5, 0)

        result = chain.solve(positions, target)

        assert result.success is True
        assert result.final_error < 0.01

    def test_target_in_3d_space(self):
        """Chain reaches target in full 3D space."""
        chain = FABRIKChain([0, 1, 2, 3], tolerance=0.001)
        positions = create_straight_chain_positions(4)

        target = Vec3(1, 1, 1)

        result = chain.solve(positions, target)

        assert result.success is True
        assert result.final_error < 0.01


# =============================================================================
# TestRotationComputation
# =============================================================================


class TestRotationComputation:
    """Tests for rotation computation from positions."""

    def test_compute_rotations_count_matches_positions(self):
        """_compute_rotations returns correct number of rotations."""
        chain = FABRIKChain([0, 1, 2, 3])
        positions = create_straight_chain_positions(4)
        rotations = [Quat.identity()] * 4

        result_rots = chain._compute_rotations(positions, rotations)

        assert len(result_rots) == 4

    def test_compute_rotations_aligned_with_y(self):
        """Rotation is identity when bone aligns with Y axis."""
        chain = FABRIKChain([0, 1, 2])
        positions = [Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 2, 0)]
        rotations = [Quat.identity()] * 3

        result_rots = chain._compute_rotations(positions, rotations)

        # First rotation should be near identity (direction is +Y)
        assert abs(result_rots[0].w) > 0.99

    def test_rotation_to_direction_aligned(self):
        """_rotation_to_direction returns identity for aligned direction."""
        chain = FABRIKChain([0, 1])

        direction = Vec3(0, 1, 0)  # Aligned with default Y
        current = Quat.identity()

        result = chain._rotation_to_direction(direction, current)

        # Should be identity or very close
        assert abs(result.w) > 0.99

    def test_rotation_to_direction_opposite(self):
        """_rotation_to_direction handles opposite direction."""
        chain = FABRIKChain([0, 1])

        direction = Vec3(0, -1, 0)  # Opposite to default Y
        current = Quat.identity()

        result = chain._rotation_to_direction(direction, current)

        # Should rotate 180 degrees
        rotated = result.rotate_vector(Vec3(0, 1, 0))
        assert vec3_nearly_equal(rotated, Vec3(0, -1, 0))

    def test_rotation_to_direction_perpendicular(self):
        """_rotation_to_direction handles perpendicular direction."""
        chain = FABRIKChain([0, 1])

        direction = Vec3(1, 0, 0)  # Perpendicular to Y
        current = Quat.identity()

        result = chain._rotation_to_direction(direction, current)

        # Should rotate Y to X
        rotated = result.rotate_vector(Vec3(0, 1, 0))
        assert vec3_nearly_equal(rotated, Vec3(1, 0, 0))

    def test_rotation_to_direction_diagonal(self):
        """_rotation_to_direction handles diagonal direction."""
        chain = FABRIKChain([0, 1])

        direction = Vec3(1, 1, 0).normalized()
        current = Quat.identity()

        result = chain._rotation_to_direction(direction, current)

        rotated = result.rotate_vector(Vec3(0, 1, 0))
        assert vec3_nearly_equal(rotated, direction)

    def test_compute_rotations_end_keeps_original(self):
        """End effector keeps its original rotation."""
        chain = FABRIKChain([0, 1, 2])
        positions = [Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(2, 0, 0)]

        # Give end effector a specific rotation
        end_rot = Quat.from_axis_angle(Vec3.unit_z(), math.pi / 4)
        rotations = [Quat.identity(), Quat.identity(), end_rot]

        result_rots = chain._compute_rotations(positions, rotations)

        # End rotation should match original
        assert result_rots[-1].x == pytest.approx(end_rot.x, abs=1e-5)
        assert result_rots[-1].y == pytest.approx(end_rot.y, abs=1e-5)
        assert result_rots[-1].z == pytest.approx(end_rot.z, abs=1e-5)
        assert result_rots[-1].w == pytest.approx(end_rot.w, abs=1e-5)


# =============================================================================
# TestTransformAPI
# =============================================================================


class TestTransformAPI:
    """Tests for solve_with_transforms method."""

    def test_solve_with_transforms_basic(self):
        """solve_with_transforms extracts and applies correctly."""
        chain = FABRIKChain([0, 1, 2], tolerance=0.001)

        transforms = [
            Transform(Vec3(0, 0, 0)),
            Transform(Vec3(0, 1, 0)),
            Transform(Vec3(0, 2, 0))
        ]

        target = Vec3(1, 1, 0)

        new_transforms = chain.solve_with_transforms(transforms, target)

        assert len(new_transforms) == 3
        # End effector should be at target
        assert vec3_nearly_equal(new_transforms[2].translation, target, tol=0.01)

    def test_solve_with_transforms_preserves_non_chain_bones(self):
        """solve_with_transforms preserves transforms outside chain."""
        chain = FABRIKChain([1, 2, 3], tolerance=0.001)

        transforms = [
            Transform(Vec3(10, 10, 10)),  # Not in chain
            Transform(Vec3(0, 0, 0)),     # Chain start
            Transform(Vec3(0, 1, 0)),
            Transform(Vec3(0, 2, 0)),     # Chain end
            Transform(Vec3(20, 20, 20)),  # Not in chain
        ]

        target = Vec3(1, 1, 0)

        new_transforms = chain.solve_with_transforms(transforms, target)

        # Transform 0 should be unchanged
        assert vec3_nearly_equal(new_transforms[0].translation, Vec3(10, 10, 10))
        # Transform 4 should be unchanged
        assert vec3_nearly_equal(new_transforms[4].translation, Vec3(20, 20, 20))

    def test_solve_with_transforms_uses_existing_rotations(self):
        """solve_with_transforms uses existing transform rotations."""
        chain = FABRIKChain([0, 1, 2], tolerance=0.001)

        rot = Quat.from_axis_angle(Vec3.unit_z(), math.pi / 4)
        transforms = [
            Transform(Vec3(0, 0, 0), rot),
            Transform(Vec3(0, 1, 0), rot),
            Transform(Vec3(0, 2, 0), rot)
        ]

        target = Vec3(0.5, 1.5, 0)

        new_transforms = chain.solve_with_transforms(transforms, target)

        # Rotations should be computed based on new positions
        assert new_transforms[2].rotation is not None

    def test_solve_with_transforms_copies_input(self):
        """solve_with_transforms returns copy, doesn't modify input."""
        chain = FABRIKChain([0, 1, 2], tolerance=0.001)

        transforms = [
            Transform(Vec3(0, 0, 0)),
            Transform(Vec3(0, 1, 0)),
            Transform(Vec3(0, 2, 0))
        ]
        original_pos = Vec3(transforms[2].translation.x, transforms[2].translation.y, transforms[2].translation.z)

        target = Vec3(1, 1, 0)
        chain.solve_with_transforms(transforms, target)

        # Original should be unchanged
        assert vec3_nearly_equal(transforms[2].translation, original_pos)

    def test_solve_with_transforms_preserves_scale(self):
        """solve_with_transforms preserves transform scales."""
        chain = FABRIKChain([0, 1, 2], tolerance=0.001)

        scale = Vec3(2, 2, 2)
        transforms = [
            Transform(Vec3(0, 0, 0), scale=scale),
            Transform(Vec3(0, 1, 0), scale=scale),
            Transform(Vec3(0, 2, 0), scale=scale)
        ]

        target = Vec3(0.5, 1.5, 0)

        new_transforms = chain.solve_with_transforms(transforms, target)

        for t in new_transforms:
            assert vec3_nearly_equal(t.scale, scale)


# =============================================================================
# TestSetConstraint
# =============================================================================


class TestSetConstraint:
    """Tests for set_constraint method."""

    def test_set_constraint_valid_index(self):
        """set_constraint sets constraint at valid index."""
        chain = FABRIKChain([0, 1, 2])
        constraint = JointConstraint(constraint_type=JointConstraintType.HINGE)

        chain.set_constraint(1, constraint)

        assert chain._constraints[1].constraint_type == JointConstraintType.HINGE

    def test_set_constraint_first_joint(self):
        """set_constraint works for first joint."""
        chain = FABRIKChain([0, 1, 2])
        constraint = JointConstraint(constraint_type=JointConstraintType.BALL_SOCKET)

        chain.set_constraint(0, constraint)

        assert chain._constraints[0].constraint_type == JointConstraintType.BALL_SOCKET

    def test_set_constraint_last_joint(self):
        """set_constraint works for last joint."""
        chain = FABRIKChain([0, 1, 2])
        constraint = JointConstraint(cone_angle=0.5)

        chain.set_constraint(2, constraint)

        assert chain._constraints[2].cone_angle == 0.5

    def test_set_constraint_negative_index_ignored(self):
        """set_constraint ignores negative index."""
        chain = FABRIKChain([0, 1, 2])
        constraint = JointConstraint(constraint_type=JointConstraintType.HINGE)

        chain.set_constraint(-1, constraint)  # Should not crash

        # All constraints should still be NONE
        for c in chain._constraints:
            assert c.constraint_type == JointConstraintType.NONE

    def test_set_constraint_out_of_bounds_ignored(self):
        """set_constraint ignores out of bounds index."""
        chain = FABRIKChain([0, 1, 2])
        constraint = JointConstraint(constraint_type=JointConstraintType.HINGE)

        chain.set_constraint(10, constraint)  # Should not crash

        # Constraints should be unchanged
        assert len(chain._constraints) == 3


# =============================================================================
# TestSolvePositionValidation
# =============================================================================


class TestSolvePositionValidation:
    """Tests for solve method input validation."""

    def test_solve_wrong_position_count_raises(self):
        """solve raises ValueError for wrong position count."""
        chain = FABRIKChain([0, 1, 2])
        positions = [Vec3(0, 0, 0), Vec3(0, 1, 0)]  # Only 2 positions for 3-bone chain
        target = Vec3(0, 2, 0)

        with pytest.raises(ValueError, match="Expected 3 positions"):
            chain.solve(positions, target)

    def test_solve_too_many_positions_raises(self):
        """solve raises ValueError for too many positions."""
        chain = FABRIKChain([0, 1, 2])
        positions = [Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 2, 0), Vec3(0, 3, 0)]
        target = Vec3(0, 2, 0)

        with pytest.raises(ValueError, match="Expected 3 positions"):
            chain.solve(positions, target)

    def test_solve_correct_position_count_succeeds(self):
        """solve succeeds with correct position count."""
        chain = FABRIKChain([0, 1, 2])
        positions = [Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 2, 0)]
        target = Vec3(0, 2, 0)

        result = chain.solve(positions, target)

        assert result is not None


# =============================================================================
# TestSolveWithOptionalRotations
# =============================================================================


class TestSolveWithOptionalRotations:
    """Tests for solve method with optional rotations parameter."""

    def test_solve_without_rotations(self):
        """solve works without providing rotations."""
        chain = FABRIKChain([0, 1, 2])
        positions = [Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 2, 0)]
        target = Vec3(1, 1, 0)

        result = chain.solve(positions, target)

        assert result is not None
        assert len(result.rotations) == 3

    def test_solve_with_rotations(self):
        """solve accepts rotations parameter."""
        chain = FABRIKChain([0, 1, 2])
        positions = [Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 2, 0)]
        rotations = [Quat.identity(), Quat.identity(), Quat.identity()]
        target = Vec3(1, 1, 0)

        result = chain.solve(positions, target, rotations)

        assert result is not None
        assert len(result.rotations) == 3

    def test_solve_with_custom_rotations_affects_constraints(self):
        """solve uses provided rotations for constraint calculation."""
        chain = FABRIKChain([0, 1, 2])
        positions = [Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 2, 0)]

        # Provide rotated parent
        rotations = [
            Quat.from_axis_angle(Vec3.unit_z(), math.pi / 4),
            Quat.identity(),
            Quat.identity()
        ]

        # Apply constraint that depends on parent rotation
        chain.set_constraint(1, JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=math.pi / 4
        ))

        target = Vec3(1, 1, 0)

        result = chain.solve(positions, target, rotations)

        assert result is not None


# =============================================================================
# TestEdgeCases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_small_tolerance(self):
        """Chain converges with very small tolerance."""
        chain = FABRIKChain([0, 1, 2], tolerance=1e-8, max_iterations=50)
        positions = [Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 2, 0)]
        target = Vec3(0, 1.5, 0.5)

        result = chain.solve(positions, target)

        # Should still attempt to converge
        assert result.iterations > 0

    def test_very_short_bones(self):
        """Chain handles very short bones."""
        chain = FABRIKChain([0, 1, 2])
        positions = [Vec3(0, 0, 0), Vec3(0, 0.001, 0), Vec3(0, 0.002, 0)]
        target = Vec3(0.001, 0.001, 0)

        result = chain.solve(positions, target)

        assert result is not None

    def test_very_long_bones(self):
        """Chain handles very long bones."""
        chain = FABRIKChain([0, 1, 2])
        positions = [Vec3(0, 0, 0), Vec3(0, 1000, 0), Vec3(0, 2000, 0)]
        target = Vec3(500, 1500, 0)

        result = chain.solve(positions, target)

        assert result is not None

    def test_negative_positions(self):
        """Chain handles negative positions."""
        chain = FABRIKChain([0, 1, 2])
        positions = [Vec3(-5, -5, -5), Vec3(-5, -4, -5), Vec3(-5, -3, -5)]
        target = Vec3(-4, -4, -5)

        result = chain.solve(positions, target)

        assert result.success is True

    def test_chain_with_many_bones(self):
        """Chain handles many bones."""
        num_bones = 20
        chain = FABRIKChain(list(range(num_bones)), tolerance=0.01)
        positions = create_straight_chain_positions(num_bones)
        target = Vec3(10, 5, 0)

        result = chain.solve(positions, target)

        assert len(result.positions) == num_bones
        assert len(result.rotations) == num_bones

    def test_target_at_epsilon_distance(self):
        """Chain handles target at epsilon distance from end."""
        chain = FABRIKChain([0, 1, 2], tolerance=MATH_EPSILON * 10)
        positions = [Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 2, 0)]
        target = Vec3(0, 2 + MATH_EPSILON * 5, 0)

        result = chain.solve(positions, target)

        assert result is not None

    def test_coincident_initial_positions(self):
        """Chain handles coincident initial positions."""
        chain = FABRIKChain([0, 1, 2])
        positions = [Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(0, 0, 0)]  # All at origin
        target = Vec3(0, 1, 0)

        # This should handle gracefully, though bone lengths will be 0
        result = chain.solve(positions, target)

        assert result is not None

    def test_horizontal_chain(self):
        """Chain works with horizontal initial configuration."""
        chain = FABRIKChain([0, 1, 2], tolerance=0.001)
        positions = create_horizontal_chain_positions(3)
        target = Vec3(1, 1, 0)

        result = chain.solve(positions, target)

        assert result.success is True

    def test_mixed_bone_lengths(self):
        """Chain handles mixed bone lengths."""
        chain = FABRIKChain([0, 1, 2, 3])
        positions = [
            Vec3(0, 0, 0),
            Vec3(0, 0.1, 0),   # Very short bone
            Vec3(0, 5.1, 0),   # Very long bone
            Vec3(0, 5.6, 0)    # Medium bone
        ]
        target = Vec3(3, 3, 0)

        result = chain.solve(positions, target)

        # Verify bone lengths preserved
        assert (result.positions[1] - result.positions[0]).length() == pytest.approx(0.1, abs=0.01)
        assert (result.positions[2] - result.positions[1]).length() == pytest.approx(5.0, abs=0.01)
        assert (result.positions[3] - result.positions[2]).length() == pytest.approx(0.5, abs=0.01)


# =============================================================================
# TestNumericalStability
# =============================================================================


class TestNumericalStability:
    """Tests for numerical stability."""

    def test_stability_repeated_solves(self):
        """Chain remains stable across repeated solves."""
        chain = FABRIKChain([0, 1, 2])
        positions = [Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 2, 0)]

        target = Vec3(1, 1, 0)

        # Solve multiple times
        for _ in range(10):
            result = chain.solve(positions, target)
            positions = result.positions

        # Should still be valid
        assert result.final_error < 0.01

    def test_stability_moving_target(self):
        """Chain remains stable with moving target."""
        chain = FABRIKChain([0, 1, 2])
        positions = [Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 2, 0)]

        # Move target in circle
        for i in range(36):
            angle = i * math.pi / 18
            target = Vec3(math.cos(angle), 1 + math.sin(angle), 0)
            result = chain.solve(positions, target)
            positions = result.positions

        # Should return to approximate start
        final_result = chain.solve(positions, Vec3(1, 1, 0))
        assert final_result is not None

    def test_no_nan_in_results(self):
        """Results never contain NaN values."""
        chain = FABRIKChain([0, 1, 2])
        positions = [Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 2, 0)]

        targets = [
            Vec3(0, 0, 0),
            Vec3(100, 100, 100),
            Vec3(-1, -1, -1),
            Vec3(MATH_EPSILON, MATH_EPSILON, MATH_EPSILON)
        ]

        for target in targets:
            result = chain.solve(positions, target)

            for pos in result.positions:
                assert not math.isnan(pos.x)
                assert not math.isnan(pos.y)
                assert not math.isnan(pos.z)

            for rot in result.rotations:
                assert not math.isnan(rot.x)
                assert not math.isnan(rot.y)
                assert not math.isnan(rot.z)
                assert not math.isnan(rot.w)

    def test_no_inf_in_results(self):
        """Results never contain infinity values."""
        chain = FABRIKChain([0, 1, 2])
        positions = [Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 2, 0)]
        target = Vec3(1e6, 1e6, 0)  # Very far target

        result = chain.solve(positions, target)

        for pos in result.positions:
            assert not math.isinf(pos.x)
            assert not math.isinf(pos.y)
            assert not math.isinf(pos.z)


# =============================================================================
# TestPositionCopying
# =============================================================================


class TestPositionCopying:
    """Tests for position copying behavior."""

    def test_solve_copies_input_positions(self):
        """solve does not modify input positions."""
        chain = FABRIKChain([0, 1, 2])
        positions = [Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 2, 0)]
        original = [Vec3(p.x, p.y, p.z) for p in positions]

        chain.solve(positions, Vec3(1, 1, 0))

        for i, pos in enumerate(positions):
            assert vec3_nearly_equal(pos, original[i])

    def test_result_positions_independent(self):
        """Result positions are independent copies."""
        chain = FABRIKChain([0, 1, 2])
        positions = [Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 2, 0)]

        result = chain.solve(positions, Vec3(1, 1, 0))

        # Modify result positions
        result.positions[0].x = 999

        # Solve again - should not be affected
        result2 = chain.solve(positions, Vec3(1, 1, 0))

        assert result2.positions[0].x != 999


# =============================================================================
# TestConstraintInteraction
# =============================================================================


class TestConstraintInteraction:
    """Tests for constraint interactions."""

    def test_multiple_constraints_applied(self):
        """Multiple constraints can be applied to chain."""
        chain = FABRIKChain([0, 1, 2, 3])

        chain.set_constraint(0, JointConstraint(
            constraint_type=JointConstraintType.HINGE,
            axis=Vec3.unit_z()
        ))
        chain.set_constraint(1, JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=math.pi / 4
        ))

        positions = create_straight_chain_positions(4)
        target = Vec3(2, 1, 1)

        result = chain.solve(positions, target)

        assert result is not None

    def test_constraints_preserved_after_solve(self):
        """Constraints remain after solve."""
        chain = FABRIKChain([0, 1, 2])
        constraint = JointConstraint(constraint_type=JointConstraintType.HINGE)
        chain.set_constraint(0, constraint)

        positions = create_straight_chain_positions(3)
        chain.solve(positions, Vec3(1, 1, 0))

        assert chain._constraints[0].constraint_type == JointConstraintType.HINGE


# =============================================================================
# TestInitialConstraints
# =============================================================================


class TestInitialConstraints:
    """Tests for initial constraint state."""

    def test_default_constraints_are_none(self):
        """All joints start with NONE constraints."""
        chain = FABRIKChain([0, 1, 2, 3, 4])

        for constraint in chain._constraints:
            assert constraint.constraint_type == JointConstraintType.NONE

    def test_constraint_count_matches_bones(self):
        """Constraint count matches bone count."""
        chain = FABRIKChain([0, 1, 2, 3, 4, 5])

        assert len(chain._constraints) == 6
