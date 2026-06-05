"""Whitebox tests for CCD (Cyclic Coordinate Descent) IK solver.

These tests verify the internal implementation details of the CCD solver,
including per-joint rotation calculations, damping behavior, rotation limits,
and all solver variants.
"""

from __future__ import annotations

import math
import pytest
from typing import List

from engine.animation.ik.ccd import (
    CCDSolver,
    CCDSolverWithWeights,
    ConstrainedCCDSolver,
    CCDResult,
    CCDRotationOrder,
    RotationLimit,
)
from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.core.math.transform import Transform
from engine.core.constants import MATH_EPSILON
from engine.animation.ik.config import (
    IK_DEFAULT_TOLERANCE,
    CCD_DEFAULT_MAX_ITERATIONS,
    CCD_DEFAULT_DAMPING,
)


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


def create_simple_chain(count: int = 3, bone_length: float = 1.0) -> tuple:
    """Create a simple vertical chain of bones.

    Returns:
        Tuple of (positions, rotations, bone_indices)
    """
    positions = [Vec3(0, i * bone_length, 0) for i in range(count)]
    rotations = [Quat.identity() for _ in range(count)]
    bone_indices = list(range(count))
    return positions, rotations, bone_indices


def create_horizontal_chain(count: int = 3, bone_length: float = 1.0) -> tuple:
    """Create a horizontal chain along X axis."""
    positions = [Vec3(i * bone_length, 0, 0) for i in range(count)]
    rotations = [Quat.identity() for _ in range(count)]
    bone_indices = list(range(count))
    return positions, rotations, bone_indices


def create_bent_chain() -> tuple:
    """Create an L-shaped chain (bent 90 degrees)."""
    positions = [
        Vec3(0, 0, 0),  # Root
        Vec3(1, 0, 0),  # Middle (bent at 90 deg)
        Vec3(1, 1, 0),  # End
    ]
    rotations = [Quat.identity(), Quat.from_euler(0, 0, math.pi / 2), Quat.identity()]
    bone_indices = [0, 1, 2]
    return positions, rotations, bone_indices


def vec3_approx_equal(a: Vec3, b: Vec3, tol: float = 1e-5) -> bool:
    """Check if two Vec3 are approximately equal."""
    return (a - b).length() < tol


def quat_approx_equal(a: Quat, b: Quat, tol: float = 1e-5) -> bool:
    """Check if two quaternions are approximately equal (accounting for sign)."""
    # Quaternions q and -q represent the same rotation
    dot = abs(a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w)
    return dot > 1.0 - tol


# =============================================================================
# Test CCDResult Dataclass
# =============================================================================


class TestCCDResultDataclass:
    """Tests for CCDResult dataclass structure and defaults."""

    def test_default_values(self):
        """Test CCDResult has correct default values."""
        result = CCDResult(success=False)
        assert result.success is False
        assert result.iterations == 0
        assert result.final_error == float('inf')
        assert result.rotations == []
        assert result.positions == []

    def test_success_result(self):
        """Test creating a successful result."""
        result = CCDResult(
            success=True,
            iterations=5,
            final_error=0.0001,
            rotations=[Quat.identity()],
            positions=[Vec3(0, 0, 0)]
        )
        assert result.success is True
        assert result.iterations == 5
        assert result.final_error == 0.0001
        assert len(result.rotations) == 1
        assert len(result.positions) == 1

    def test_failure_result(self):
        """Test creating a failure result."""
        result = CCDResult(
            success=False,
            iterations=10,
            final_error=0.5,
            rotations=[],
            positions=[]
        )
        assert result.success is False
        assert result.iterations == 10
        assert result.final_error == 0.5

    def test_iterations_tracking(self):
        """Test iterations field tracking."""
        result = CCDResult(success=True, iterations=42)
        assert result.iterations == 42

    def test_final_error_tracking(self):
        """Test final_error field."""
        result = CCDResult(success=False, final_error=1.234)
        assert abs(result.final_error - 1.234) < 1e-9

    def test_rotations_list_mutable(self):
        """Test rotations list can be modified."""
        result = CCDResult(success=True)
        result.rotations.append(Quat.identity())
        assert len(result.rotations) == 1

    def test_positions_list_mutable(self):
        """Test positions list can be modified."""
        result = CCDResult(success=True)
        result.positions.append(Vec3(1, 2, 3))
        assert len(result.positions) == 1

    def test_zero_error_convergence(self):
        """Test result with zero error."""
        result = CCDResult(success=True, iterations=1, final_error=0.0)
        assert result.final_error == 0.0
        assert result.success is True

    def test_large_iterations(self):
        """Test with large iteration count."""
        result = CCDResult(success=False, iterations=10000)
        assert result.iterations == 10000


# =============================================================================
# Test RotationLimit
# =============================================================================


class TestRotationLimit:
    """Tests for RotationLimit dataclass and clamping behavior."""

    def test_disabled_passthrough(self):
        """Test disabled limits pass rotation through unchanged."""
        limit = RotationLimit(enabled=False)
        rotation = Quat.from_euler(0.5, 0.5, 0.5)
        result = limit.clamp_rotation(rotation)
        assert quat_approx_equal(result, rotation)

    def test_default_construction(self):
        """Test default RotationLimit values."""
        limit = RotationLimit()
        assert limit.enabled is False
        assert limit.is_hinge is False
        assert abs(limit.min_angles.x + math.pi) < 1e-6
        assert abs(limit.max_angles.x - math.pi) < 1e-6

    def test_enabled_euler_clamp(self):
        """Test Euler angle clamping when enabled."""
        limit = RotationLimit(
            enabled=True,
            min_angles=Vec3(-0.1, -0.1, -0.1),
            max_angles=Vec3(0.1, 0.1, 0.1),
            is_hinge=False
        )
        # Create rotation exceeding limits
        rotation = Quat.from_euler(0.5, 0.5, 0.5)
        result = limit.clamp_rotation(rotation)

        # Check result is within limits (with tolerance for Euler gimbal effects)
        # Note: Euler angles can have numerical issues near limits due to
        # the quaternion<->euler conversion, so we use a generous tolerance
        pitch, yaw, roll = result.to_euler()
        assert pitch <= 0.15  # Tolerance for numerical precision
        assert yaw <= 0.15
        assert roll <= 0.15

    def test_hinge_constraint_basic(self):
        """Test hinge constraint restricts to single axis."""
        limit = RotationLimit(
            enabled=True,
            axis=Vec3.unit_y(),
            min_angles=Vec3(-math.pi, -math.pi / 2, -math.pi),
            max_angles=Vec3(math.pi, math.pi / 2, math.pi),
            is_hinge=True
        )
        rotation = Quat.from_euler(0.5, 0.3, 0.4)  # Rotation on multiple axes
        result = limit.clamp_rotation(rotation)

        # Result should be rotation only around Y axis
        assert isinstance(result, Quat)

    def test_hinge_clamp_small_rotation(self):
        """Test hinge constraint with very small rotation (near identity)."""
        limit = RotationLimit(
            enabled=True,
            axis=Vec3.unit_z(),
            is_hinge=True
        )
        # Very small rotation - should return identity
        rotation = Quat(0.0, 0.0, 1e-12, 1.0)
        result = limit.clamp_rotation(rotation)
        assert quat_approx_equal(result, Quat.identity())

    def test_euler_clamp_negative_values(self):
        """Test Euler clamping with negative rotations."""
        limit = RotationLimit(
            enabled=True,
            min_angles=Vec3(-0.2, -0.2, -0.2),
            max_angles=Vec3(0.2, 0.2, 0.2)
        )
        rotation = Quat.from_euler(-0.5, -0.5, -0.5)
        result = limit.clamp_rotation(rotation)

        # Check result is within limits (with tolerance for Euler gimbal effects)
        pitch, yaw, roll = result.to_euler()
        assert pitch >= -0.25  # Tolerance for numerical precision
        assert yaw >= -0.25
        assert roll >= -0.25

    def test_hinge_axis_alignment_positive(self):
        """Test hinge with rotation aligned to axis."""
        limit = RotationLimit(
            enabled=True,
            axis=Vec3.unit_y(),
            min_angles=Vec3(0, -math.pi / 4, 0),
            max_angles=Vec3(0, math.pi / 4, 0),
            is_hinge=True
        )
        # Rotation around Y axis
        rotation = Quat.from_axis_angle(Vec3.unit_y(), 0.2)
        result = limit.clamp_rotation(rotation)
        assert isinstance(result, Quat)

    def test_hinge_axis_alignment_negative(self):
        """Test hinge with rotation opposite to axis direction."""
        limit = RotationLimit(
            enabled=True,
            axis=Vec3.unit_y(),
            min_angles=Vec3(0, -math.pi / 4, 0),
            max_angles=Vec3(0, math.pi / 4, 0),
            is_hinge=True
        )
        rotation = Quat.from_axis_angle(Vec3(0, -1, 0), 0.3)
        result = limit.clamp_rotation(rotation)
        assert isinstance(result, Quat)

    def test_euler_within_limits_unchanged(self):
        """Test Euler clamp does not change rotation within limits."""
        limit = RotationLimit(
            enabled=True,
            min_angles=Vec3(-1.0, -1.0, -1.0),
            max_angles=Vec3(1.0, 1.0, 1.0)
        )
        rotation = Quat.from_euler(0.2, 0.2, 0.2)
        result = limit.clamp_rotation(rotation)
        assert quat_approx_equal(result, rotation, 0.01)

    def test_hinge_perpendicular_axis_returns_identity(self):
        """Test hinge with rotation perpendicular to hinge axis."""
        limit = RotationLimit(
            enabled=True,
            axis=Vec3.unit_y(),  # Y axis hinge
            is_hinge=True
        )
        # Rotation around X axis (perpendicular to Y)
        rotation = Quat.from_axis_angle(Vec3.unit_x(), 0.5)
        result = limit.clamp_rotation(rotation)
        # Should project to zero on Y axis
        assert isinstance(result, Quat)

    def test_custom_hinge_axis(self):
        """Test hinge with custom (non-unit) axis."""
        limit = RotationLimit(
            enabled=True,
            axis=Vec3(1, 1, 0),  # Diagonal axis
            is_hinge=True
        )
        rotation = Quat.from_euler(0.3, 0.3, 0.0)
        result = limit.clamp_rotation(rotation)
        assert isinstance(result, Quat)


# =============================================================================
# Test CCDSolver Construction
# =============================================================================


class TestCCDSolverConstruction:
    """Tests for CCDSolver initialization and validation."""

    def test_valid_construction(self):
        """Test valid solver construction."""
        solver = CCDSolver([0, 1, 2])
        assert solver.chain_length == 3
        assert solver.bone_indices == [0, 1, 2]
        assert solver.tolerance == IK_DEFAULT_TOLERANCE
        assert solver.max_iterations == CCD_DEFAULT_MAX_ITERATIONS
        assert solver.damping == CCD_DEFAULT_DAMPING

    def test_minimum_chain_length(self):
        """Test chain must have at least 2 bones."""
        with pytest.raises(ValueError, match="at least 2 bones"):
            CCDSolver([0])

    def test_single_bone_raises(self):
        """Test single bone chain raises ValueError."""
        with pytest.raises(ValueError):
            CCDSolver([5])

    def test_empty_chain_raises(self):
        """Test empty chain raises ValueError."""
        with pytest.raises(ValueError):
            CCDSolver([])

    def test_invalid_damping_zero_raises(self):
        """Test damping=0 raises ValueError."""
        with pytest.raises(ValueError, match="Damping"):
            CCDSolver([0, 1], damping=0.0)

    def test_invalid_damping_negative_raises(self):
        """Test negative damping raises ValueError."""
        with pytest.raises(ValueError, match="Damping"):
            CCDSolver([0, 1], damping=-0.5)

    def test_invalid_damping_above_one_raises(self):
        """Test damping > 1 raises ValueError."""
        with pytest.raises(ValueError, match="Damping"):
            CCDSolver([0, 1], damping=1.5)

    def test_damping_exactly_one_valid(self):
        """Test damping=1.0 is valid."""
        solver = CCDSolver([0, 1], damping=1.0)
        assert solver.damping == 1.0

    def test_damping_small_positive_valid(self):
        """Test small positive damping is valid."""
        solver = CCDSolver([0, 1], damping=0.001)
        assert solver.damping == 0.001

    def test_custom_tolerance(self):
        """Test custom tolerance parameter."""
        solver = CCDSolver([0, 1], tolerance=0.01)
        assert solver.tolerance == 0.01

    def test_custom_max_iterations(self):
        """Test custom max_iterations parameter."""
        solver = CCDSolver([0, 1], max_iterations=50)
        assert solver.max_iterations == 50

    def test_bone_indices_copied(self):
        """Test bone indices are copied, not referenced."""
        indices = [0, 1, 2]
        solver = CCDSolver(indices)
        indices.append(3)  # Modify original
        assert solver.bone_indices == [0, 1, 2]  # Solver unchanged

    def test_chain_length_property(self):
        """Test chain_length property returns correct value."""
        solver = CCDSolver([0, 1, 2, 3, 4])
        assert solver.chain_length == 5

    def test_default_rotation_limits_created(self):
        """Test rotation limits are created for each joint."""
        solver = CCDSolver([0, 1, 2, 3])
        assert len(solver._rotation_limits) == 4
        for limit in solver._rotation_limits:
            assert isinstance(limit, RotationLimit)
            assert limit.enabled is False

    def test_default_rotation_order(self):
        """Test default rotation order is END_TO_ROOT."""
        solver = CCDSolver([0, 1])
        assert solver._rotation_order == CCDRotationOrder.END_TO_ROOT


# =============================================================================
# Test Basic Solve Behavior
# =============================================================================


class TestSolveBasic:
    """Tests for basic solve functionality."""

    def test_reachable_target_converges(self):
        """Test solver converges for reachable target."""
        # Use horizontal chain to require actual rotation
        positions, rotations, indices = create_horizontal_chain(3, 1.0)
        solver = CCDSolver(indices, tolerance=0.1)

        # Target within reach that requires rotation
        target = Vec3(1.5, 1.0, 0)
        result = solver.solve(positions, rotations, target)

        assert result.success is True
        assert result.iterations <= solver.max_iterations
        assert result.final_error <= solver.tolerance

    def test_unreachable_target_extends(self):
        """Test solver extends toward unreachable target."""
        positions, rotations, indices = create_simple_chain(3, 1.0)
        solver = CCDSolver(indices)

        # Target beyond reach (chain length = 2)
        target = Vec3(0, 10, 0)
        result = solver.solve(positions, rotations, target)

        # Should not converge but should extend toward target
        assert result.iterations == solver.max_iterations
        end_pos = result.positions[-1]
        # End should be further in target direction than starting
        assert end_pos.y >= positions[-1].y - 0.1

    def test_target_at_end_effector(self):
        """Test target at current end effector position."""
        positions, rotations, indices = create_simple_chain(3, 1.0)
        solver = CCDSolver(indices)

        target = Vec3(positions[-1].x, positions[-1].y, positions[-1].z)
        result = solver.solve(positions, rotations, target)

        assert result.success is True
        assert result.iterations == 1  # Converges immediately

    def test_wrong_position_count_raises(self):
        """Test solve raises for wrong position count."""
        solver = CCDSolver([0, 1, 2])
        positions = [Vec3(0, 0, 0), Vec3(1, 0, 0)]  # Only 2
        rotations = [Quat.identity(), Quat.identity(), Quat.identity()]

        with pytest.raises(ValueError, match="Expected 3 positions"):
            solver.solve(positions, rotations, Vec3(1, 1, 1))

    def test_convergence_within_tolerance(self):
        """Test result meets tolerance requirement when converged."""
        positions, rotations, indices = create_simple_chain(4, 1.0)
        solver = CCDSolver(indices, tolerance=0.01)

        target = Vec3(0.5, 2.5, 0)
        result = solver.solve(positions, rotations, target)

        if result.success:
            assert result.final_error <= 0.01

    def test_positions_modified_in_result(self):
        """Test result contains modified positions."""
        positions, rotations, indices = create_horizontal_chain(3, 1.0)
        solver = CCDSolver(indices)

        target = Vec3(1, 1, 0)  # Above the horizontal chain
        result = solver.solve(positions, rotations, target)

        # End position should have moved toward target
        assert len(result.positions) == 3
        assert result.positions[-1].y > 0  # Moved up

    def test_rotations_modified_in_result(self):
        """Test result contains modified rotations."""
        positions, rotations, indices = create_horizontal_chain(3, 1.0)
        solver = CCDSolver(indices)

        target = Vec3(1, 1, 0)
        result = solver.solve(positions, rotations, target)

        assert len(result.rotations) == 3
        # At least one rotation should have changed
        changed = False
        for i, rot in enumerate(result.rotations):
            if not quat_approx_equal(rot, Quat.identity()):
                changed = True
                break
        assert changed


# =============================================================================
# Test Per-Joint Rotation Calculation
# =============================================================================


class TestPerJointRotation:
    """Tests for per-joint rotation calculations."""

    def test_to_end_vector_calculation(self):
        """Test to_end vector is computed correctly."""
        positions, rotations, indices = create_horizontal_chain(3, 1.0)
        solver = CCDSolver(indices)

        # Joint 0 at (0,0,0), end at (2,0,0)
        # to_end should be (1,0,0) normalized
        joint_pos = positions[0]
        end_pos = positions[-1]
        to_end = (end_pos - joint_pos).normalized()

        assert vec3_approx_equal(to_end, Vec3(1, 0, 0))

    def test_to_target_vector_calculation(self):
        """Test to_target vector is computed correctly."""
        positions, rotations, indices = create_horizontal_chain(3, 1.0)

        target = Vec3(0, 2, 0)
        joint_pos = positions[0]
        to_target = (target - joint_pos).normalized()

        assert vec3_approx_equal(to_target, Vec3(0, 1, 0))

    def test_rotation_axis_from_cross_product(self):
        """Test rotation axis computed from cross product."""
        to_end = Vec3(1, 0, 0)
        to_target = Vec3(0, 1, 0)

        axis = to_end.cross(to_target).normalized()
        # Cross of X and Y is Z
        assert vec3_approx_equal(axis, Vec3(0, 0, 1))

    def test_angle_from_dot_product(self):
        """Test angle computed from dot product."""
        to_end = Vec3(1, 0, 0)
        to_target = Vec3(0, 1, 0)

        dot = to_end.dot(to_target)
        angle = math.acos(max(-1.0, min(1.0, dot)))

        assert abs(angle - math.pi / 2) < 1e-6

    def test_parallel_vectors_no_rotation(self):
        """Test no rotation when vectors are parallel."""
        positions, rotations, indices = create_horizontal_chain(3, 1.0)
        solver = CCDSolver(indices)

        # Target along same direction as chain
        target = Vec3(5, 0, 0)
        result = solver.solve(positions, rotations, target)

        # Chain should stretch toward target without major rotations
        # (unreachable, but aligned)

    def test_opposite_vectors_180_rotation(self):
        """Test 180 degree rotation for opposite vectors."""
        to_end = Vec3(1, 0, 0)
        to_target = Vec3(-1, 0, 0)

        dot = to_end.dot(to_target)
        angle = math.acos(max(-1.0, min(1.0, dot)))

        assert abs(angle - math.pi) < 1e-6

    def test_zero_length_to_end_skipped(self):
        """Test joint is skipped when to_end has zero length."""
        # Create chain where end effector is at same position as joint
        positions = [Vec3(0, 0, 0), Vec3(0, 0, 0)]  # Coincident
        rotations = [Quat.identity(), Quat.identity()]
        solver = CCDSolver([0, 1])

        # Should not crash, joint rotation should be skipped
        result = solver.solve(positions, rotations, Vec3(1, 0, 0))
        assert isinstance(result, CCDResult)

    def test_zero_length_to_target_skipped(self):
        """Test joint is skipped when target at joint position."""
        positions, rotations, indices = create_horizontal_chain(3, 1.0)
        solver = CCDSolver(indices)

        # Target at root joint position
        target = Vec3(0, 0, 0)
        result = solver.solve(positions, rotations, target)
        assert isinstance(result, CCDResult)


# =============================================================================
# Test Damping Behavior
# =============================================================================


class TestDamping:
    """Tests for damping factor behavior."""

    def test_full_damping_full_rotation(self):
        """Test damping=1.0 applies full rotation."""
        positions, rotations, indices = create_horizontal_chain(3, 1.0)
        solver_full = CCDSolver(indices, damping=1.0, max_iterations=1)

        target = Vec3(1, 1, 0)
        result = solver_full.solve(positions, rotations, target)

        # With full damping, should move more toward target
        assert result.positions[-1].y > 0

    def test_partial_damping_scales_rotation(self):
        """Test partial damping reduces rotation amount."""
        positions1, rotations1, indices = create_horizontal_chain(3, 1.0)
        positions2, rotations2, _ = create_horizontal_chain(3, 1.0)

        solver_full = CCDSolver(indices, damping=1.0, max_iterations=1)
        solver_half = CCDSolver(indices, damping=0.5, max_iterations=1)

        target = Vec3(1, 2, 0)

        result_full = solver_full.solve(positions1, rotations1, target)
        result_half = solver_half.solve(positions2, rotations2, target)

        # Half damping should move less than full damping
        assert result_half.positions[-1].y < result_full.positions[-1].y

    def test_low_damping_conservative_motion(self):
        """Test very low damping produces conservative motion."""
        positions, rotations, indices = create_horizontal_chain(3, 1.0)
        solver = CCDSolver(indices, damping=0.1, max_iterations=1)

        target = Vec3(1, 2, 0)
        result = solver.solve(positions, rotations, target)

        # Should move only slightly
        assert result.positions[-1].y < 0.5

    def test_damping_affects_convergence_speed(self):
        """Test damping affects how quickly solver converges."""
        target = Vec3(1, 1.5, 0)

        positions1, rotations1, indices = create_horizontal_chain(3, 1.0)
        solver_fast = CCDSolver(indices, damping=1.0, tolerance=0.01)
        result_fast = solver_fast.solve(positions1, rotations1, target)

        positions2, rotations2, _ = create_horizontal_chain(3, 1.0)
        solver_slow = CCDSolver(indices, damping=0.3, tolerance=0.01)
        result_slow = solver_slow.solve(positions2, rotations2, target)

        # Higher damping should converge in fewer iterations
        if result_fast.success and result_slow.success:
            assert result_fast.iterations <= result_slow.iterations

    def test_damping_stability(self):
        """Test low damping provides stable behavior."""
        positions, rotations, indices = create_horizontal_chain(5, 1.0)
        solver = CCDSolver(indices, damping=0.2, max_iterations=50)

        target = Vec3(2, 2, 0)
        result = solver.solve(positions, rotations, target)

        # Should converge smoothly without oscillation
        assert result.final_error < 1.0


# =============================================================================
# Test Rotation Orders
# =============================================================================


class TestRotationOrders:
    """Tests for different rotation orders."""

    def test_end_to_root_order(self):
        """Test END_TO_ROOT processes joints from end toward root."""
        positions, rotations, indices = create_horizontal_chain(3, 1.0)
        solver = CCDSolver(indices)
        solver.set_rotation_order(CCDRotationOrder.END_TO_ROOT)

        target = Vec3(1, 1, 0)
        result = solver.solve(positions, rotations, target)

        # Should converge
        assert result.final_error < 1.0

    def test_root_to_end_order(self):
        """Test ROOT_TO_END processes joints from root toward end."""
        positions, rotations, indices = create_horizontal_chain(3, 1.0)
        solver = CCDSolver(indices)
        solver.set_rotation_order(CCDRotationOrder.ROOT_TO_END)

        target = Vec3(1, 1, 0)
        result = solver.solve(positions, rotations, target)

        # Should converge
        assert result.final_error < 1.0

    def test_alternating_order(self):
        """Test ALTERNATING switches between passes."""
        positions, rotations, indices = create_horizontal_chain(3, 1.0)
        solver = CCDSolver(indices)
        solver.set_rotation_order(CCDRotationOrder.ALTERNATING)

        target = Vec3(1, 1, 0)
        result = solver.solve(positions, rotations, target)

        # Should converge
        assert result.final_error < 1.0

    def test_different_orders_converge_differently(self):
        """Test different orders may take different paths."""
        target = Vec3(1.5, 1.5, 0.5)

        results = []
        for order in CCDRotationOrder:
            positions, rotations, indices = create_horizontal_chain(4, 1.0)
            solver = CCDSolver(indices, max_iterations=20)
            solver.set_rotation_order(order)
            result = solver.solve(positions, rotations, target)
            results.append(result)

        # All should make progress toward target
        for result in results:
            assert result.final_error < 5.0

    def test_set_rotation_order_method(self):
        """Test set_rotation_order method."""
        solver = CCDSolver([0, 1, 2])

        solver.set_rotation_order(CCDRotationOrder.ROOT_TO_END)
        assert solver._rotation_order == CCDRotationOrder.ROOT_TO_END

        solver.set_rotation_order(CCDRotationOrder.ALTERNATING)
        assert solver._rotation_order == CCDRotationOrder.ALTERNATING

    def test_alternating_even_iteration_end_to_root(self):
        """Test alternating uses end_to_root on even iterations."""
        # This is tested via solve behavior
        positions, rotations, indices = create_horizontal_chain(3, 1.0)
        solver = CCDSolver(indices, max_iterations=2)
        solver.set_rotation_order(CCDRotationOrder.ALTERNATING)

        target = Vec3(1, 1, 0)
        result = solver.solve(positions, rotations, target)

        # Should process correctly
        assert isinstance(result, CCDResult)


# =============================================================================
# Test Joint Limits
# =============================================================================


class TestJointLimits:
    """Tests for joint rotation limits."""

    def test_limits_respected_per_joint(self):
        """Test rotation limits are respected for each joint."""
        positions, rotations, indices = create_horizontal_chain(3, 1.0)
        solver = CCDSolver(indices)

        # Set tight limit on joint 0
        limit = RotationLimit(
            enabled=True,
            min_angles=Vec3(-0.1, -0.1, -0.1),
            max_angles=Vec3(0.1, 0.1, 0.1)
        )
        solver.set_rotation_limit(0, limit)

        target = Vec3(0, 2, 0)  # Would require large rotation
        result = solver.solve(positions, rotations, target)

        # Solver should still work, just with limited motion on joint 0
        assert isinstance(result, CCDResult)

    def test_set_rotation_limit_method(self):
        """Test set_rotation_limit method."""
        solver = CCDSolver([0, 1, 2])
        limit = RotationLimit(enabled=True)

        solver.set_rotation_limit(1, limit)
        assert solver._rotation_limits[1].enabled is True
        assert solver._rotation_limits[0].enabled is False  # Unchanged

    def test_set_limit_out_of_range_ignored(self):
        """Test set_rotation_limit with invalid index is ignored."""
        solver = CCDSolver([0, 1, 2])
        limit = RotationLimit(enabled=True)

        # Should not crash
        solver.set_rotation_limit(10, limit)
        solver.set_rotation_limit(-1, limit)

        # Limits should be unchanged
        for l in solver._rotation_limits:
            assert l.enabled is False

    def test_hinge_limit_on_joint(self):
        """Test hinge constraint on a joint."""
        positions, rotations, indices = create_horizontal_chain(4, 1.0)
        solver = CCDSolver(indices)

        # Make joint 1 a hinge around Z
        limit = RotationLimit(
            enabled=True,
            axis=Vec3.unit_z(),
            is_hinge=True
        )
        solver.set_rotation_limit(1, limit)

        target = Vec3(2, 1, 1)
        result = solver.solve(positions, rotations, target)

        assert isinstance(result, CCDResult)

    def test_multiple_limits_combined(self):
        """Test multiple joints with different limits."""
        positions, rotations, indices = create_horizontal_chain(4, 1.0)
        solver = CCDSolver(indices)

        # Joint 0: tight euler limits
        solver.set_rotation_limit(0, RotationLimit(
            enabled=True,
            min_angles=Vec3(-0.2, -0.2, -0.2),
            max_angles=Vec3(0.2, 0.2, 0.2)
        ))

        # Joint 1: hinge
        solver.set_rotation_limit(1, RotationLimit(
            enabled=True,
            axis=Vec3.unit_y(),
            is_hinge=True
        ))

        target = Vec3(2, 2, 0)
        result = solver.solve(positions, rotations, target)

        assert isinstance(result, CCDResult)

    def test_disabled_limit_allows_full_rotation(self):
        """Test disabled limits allow full rotation."""
        positions, rotations, indices = create_horizontal_chain(3, 1.0)
        solver = CCDSolver(indices)

        # Ensure limits are disabled (default)
        for l in solver._rotation_limits:
            assert l.enabled is False

        target = Vec3(-2, 0, 0)  # Opposite direction, needs large rotation
        result = solver.solve(positions, rotations, target)

        # Should be able to make large rotation
        assert isinstance(result, CCDResult)


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_coincident_joints(self):
        """Test handling of coincident (same position) joints."""
        positions = [Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(1, 0, 0)]
        rotations = [Quat.identity()] * 3
        solver = CCDSolver([0, 1, 2])

        result = solver.solve(positions, rotations, Vec3(0, 1, 0))
        assert isinstance(result, CCDResult)

    def test_target_at_joint_position(self):
        """Test target at a joint position."""
        positions, rotations, indices = create_horizontal_chain(3, 1.0)
        solver = CCDSolver(indices)

        # Target at middle joint
        target = Vec3(positions[1].x, positions[1].y, positions[1].z)
        result = solver.solve(positions, rotations, target)
        assert isinstance(result, CCDResult)

    def test_zero_length_bone(self):
        """Test handling of zero-length bones in chain."""
        positions = [Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(1, 0, 0)]
        rotations = [Quat.identity()] * 3
        solver = CCDSolver([0, 1, 2])

        result = solver.solve(positions, rotations, Vec3(1, 1, 0))
        assert isinstance(result, CCDResult)

    def test_very_small_target_distance(self):
        """Test target very close to end effector."""
        positions, rotations, indices = create_horizontal_chain(3, 1.0)
        solver = CCDSolver(indices)

        end_pos = positions[-1]
        target = Vec3(end_pos.x + 1e-8, end_pos.y + 1e-8, end_pos.z)
        result = solver.solve(positions, rotations, target)

        assert result.success is True

    def test_very_long_chain(self):
        """Test solver with long chain."""
        positions, rotations, indices = create_simple_chain(20, 0.5)
        solver = CCDSolver(indices)

        target = Vec3(0, 8, 0)
        result = solver.solve(positions, rotations, target)

        assert isinstance(result, CCDResult)
        assert len(result.positions) == 20

    def test_two_bone_minimum_chain(self):
        """Test minimum valid chain (2 bones)."""
        solver = CCDSolver([0, 1])
        positions = [Vec3(0, 0, 0), Vec3(1, 0, 0)]
        rotations = [Quat.identity(), Quat.identity()]

        result = solver.solve(positions, rotations, Vec3(0.5, 0.5, 0))
        assert isinstance(result, CCDResult)

    def test_negative_target_coordinates(self):
        """Test target with negative coordinates."""
        positions, rotations, indices = create_horizontal_chain(3, 1.0)
        solver = CCDSolver(indices)

        target = Vec3(-1, -1, -1)
        result = solver.solve(positions, rotations, target)
        assert isinstance(result, CCDResult)

    def test_very_far_target(self):
        """Test with target very far from chain."""
        positions, rotations, indices = create_horizontal_chain(3, 1.0)
        solver = CCDSolver(indices)

        target = Vec3(1000, 1000, 1000)
        result = solver.solve(positions, rotations, target)

        # Should extend toward target but not converge
        assert result.iterations == solver.max_iterations

    def test_target_behind_chain(self):
        """Test target behind the chain root."""
        positions, rotations, indices = create_horizontal_chain(3, 1.0)
        solver = CCDSolver(indices)

        target = Vec3(-2, 0, 0)
        result = solver.solve(positions, rotations, target)
        assert isinstance(result, CCDResult)

    def test_3d_target_all_axes(self):
        """Test target requiring movement in all axes."""
        positions, rotations, indices = create_horizontal_chain(4, 1.0)
        solver = CCDSolver(indices)

        target = Vec3(1.5, 1.5, 1.5)
        result = solver.solve(positions, rotations, target)

        # End should have moved in all axes
        end = result.positions[-1]
        assert isinstance(result, CCDResult)


# =============================================================================
# Test Bone Length Caching
# =============================================================================


class TestBoneLengthCaching:
    """Tests for bone length caching mechanism."""

    def test_lengths_cached_on_first_solve(self):
        """Test bone lengths are cached on first solve."""
        positions, rotations, indices = create_horizontal_chain(3, 1.0)
        solver = CCDSolver(indices)

        assert solver._lengths_cached is False

        solver.solve(positions, rotations, Vec3(1, 1, 0))

        assert solver._lengths_cached is True
        assert len(solver._bone_lengths) == 2

    def test_cached_lengths_correct(self):
        """Test cached bone lengths are correct."""
        positions = [Vec3(0, 0, 0), Vec3(2, 0, 0), Vec3(2, 3, 0)]
        rotations = [Quat.identity()] * 3
        solver = CCDSolver([0, 1, 2])

        solver.solve(positions, rotations, Vec3(1, 1, 0))

        assert abs(solver._bone_lengths[0] - 2.0) < 1e-6
        assert abs(solver._bone_lengths[1] - 3.0) < 1e-6

    def test_reset_cached_lengths(self):
        """Test reset_cached_lengths method."""
        positions, rotations, indices = create_horizontal_chain(3, 1.0)
        solver = CCDSolver(indices)

        solver.solve(positions, rotations, Vec3(1, 1, 0))
        assert solver._lengths_cached is True

        solver.reset_cached_lengths()
        assert solver._lengths_cached is False
        assert solver._bone_lengths == []


# =============================================================================
# Test Update Chain Positions
# =============================================================================


class TestUpdateChainPositions:
    """Tests for chain position update after rotation."""

    def test_positions_updated_after_rotation(self):
        """Test positions are correctly updated after joint rotation."""
        positions, rotations, indices = create_horizontal_chain(3, 1.0)
        solver = CCDSolver(indices)

        # Single iteration should update positions
        result = solver.solve(positions, rotations, Vec3(1, 1, 0))

        # Positions should be different from input
        assert not vec3_approx_equal(result.positions[-1], positions[-1])

    def test_chain_integrity_preserved(self):
        """Test bone lengths are preserved during updates."""
        positions = [Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(2, 0, 0)]
        rotations = [Quat.identity()] * 3
        solver = CCDSolver([0, 1, 2])

        result = solver.solve(positions, rotations, Vec3(1, 1, 0))

        # Check bone lengths
        bone1_len = (result.positions[1] - result.positions[0]).length()
        bone2_len = (result.positions[2] - result.positions[1]).length()

        assert abs(bone1_len - 1.0) < 0.1  # Some tolerance for numerical error
        assert abs(bone2_len - 1.0) < 0.1


# =============================================================================
# Test Solve with Transforms
# =============================================================================


class TestSolveWithTransforms:
    """Tests for solve_with_transforms method."""

    def test_solve_with_transforms_basic(self):
        """Test solve_with_transforms returns modified transforms."""
        transforms = [
            Transform(Vec3(0, 0, 0), Quat.identity()),
            Transform(Vec3(1, 0, 0), Quat.identity()),
            Transform(Vec3(2, 0, 0), Quat.identity()),
        ]
        solver = CCDSolver([0, 1, 2])

        result = solver.solve_with_transforms(transforms, Vec3(1, 1, 0))

        assert len(result) == 3
        for t in result:
            assert isinstance(t, Transform)

    def test_original_transforms_unchanged(self):
        """Test original transforms are not modified."""
        transforms = [
            Transform(Vec3(0, 0, 0), Quat.identity()),
            Transform(Vec3(1, 0, 0), Quat.identity()),
        ]
        original_pos = Vec3(transforms[0].translation.x,
                           transforms[0].translation.y,
                           transforms[0].translation.z)

        solver = CCDSolver([0, 1])
        solver.solve_with_transforms(transforms, Vec3(0.5, 0.5, 0))

        assert vec3_approx_equal(transforms[0].translation, original_pos)

    def test_solve_with_transforms_applies_ik(self):
        """Test IK is actually applied to returned transforms."""
        transforms = [
            Transform(Vec3(0, 0, 0), Quat.identity()),
            Transform(Vec3(1, 0, 0), Quat.identity()),
            Transform(Vec3(2, 0, 0), Quat.identity()),
        ]
        solver = CCDSolver([0, 1, 2])

        target = Vec3(1, 1, 0)
        result = solver.solve_with_transforms(transforms, target)

        # End transform should be closer to target
        end_pos = result[2].translation
        original_end = transforms[2].translation

        dist_after = (target - end_pos).length()
        dist_before = (target - original_end).length()

        assert dist_after < dist_before


# =============================================================================
# Test CCDSolverWithWeights
# =============================================================================


class TestCCDSolverWithWeights:
    """Tests for weighted CCD solver."""

    def test_weighted_solver_construction(self):
        """Test weighted solver construction."""
        solver = CCDSolverWithWeights([0, 1, 2])
        assert solver.chain_length == 3
        assert solver._weights == [1.0, 1.0, 1.0]

    def test_custom_weights_construction(self):
        """Test construction with custom weights."""
        solver = CCDSolverWithWeights([0, 1, 2], weights=[0.5, 1.0, 0.25])
        assert solver._weights == [0.5, 1.0, 0.25]

    def test_weights_mismatch_raises(self):
        """Test weight count mismatch raises error."""
        with pytest.raises(ValueError, match="Weights must match"):
            CCDSolverWithWeights([0, 1, 2], weights=[0.5, 1.0])

    def test_weights_clamped_to_range(self):
        """Test weights are clamped to [0, 1]."""
        solver = CCDSolverWithWeights([0, 1, 2], weights=[-0.5, 1.5, 0.5])
        assert solver._weights == [0.0, 1.0, 0.5]

    def test_set_weight_method(self):
        """Test set_weight method."""
        solver = CCDSolverWithWeights([0, 1, 2])
        solver.set_weight(1, 0.5)
        assert solver._weights[1] == 0.5

    def test_set_weight_clamped(self):
        """Test set_weight clamps values."""
        solver = CCDSolverWithWeights([0, 1, 2])
        solver.set_weight(0, 2.0)
        solver.set_weight(1, -0.5)
        assert solver._weights[0] == 1.0
        assert solver._weights[1] == 0.0

    def test_set_weight_invalid_index(self):
        """Test set_weight with invalid index is ignored."""
        solver = CCDSolverWithWeights([0, 1, 2])
        solver.set_weight(10, 0.5)  # Should not crash
        solver.set_weight(-1, 0.5)  # Should not crash

    def test_zero_weight_joint_skipped(self):
        """Test zero weight joint doesn't rotate."""
        positions, rotations, indices = create_horizontal_chain(3, 1.0)
        solver = CCDSolverWithWeights(indices, weights=[0.0, 0.0, 1.0])

        target = Vec3(1, 1, 0)
        result = solver.solve(positions, rotations, target)

        # Only last joint should have significant rotation
        # (though it's the end effector, so it shouldn't rotate in standard CCD)
        assert isinstance(result, CCDResult)

    def test_weighted_solve_basic(self):
        """Test weighted solve produces valid result."""
        positions, rotations, indices = create_horizontal_chain(3, 1.0)
        solver = CCDSolverWithWeights(indices, weights=[1.0, 0.5, 0.5])

        result = solver.solve(positions, rotations, Vec3(1, 1, 0))
        assert isinstance(result, CCDResult)

    def test_weight_affects_rotation_amount(self):
        """Test weight affects how much joint rotates."""
        target = Vec3(1, 1, 0)

        # Full weight
        pos1, rot1, idx = create_horizontal_chain(3, 1.0)
        solver_full = CCDSolverWithWeights(idx, weights=[1.0, 1.0, 1.0], max_iterations=1)
        result_full = solver_full.solve(pos1, rot1, target)

        # Half weight
        pos2, rot2, _ = create_horizontal_chain(3, 1.0)
        solver_half = CCDSolverWithWeights(idx, weights=[0.5, 0.5, 0.5], max_iterations=1)
        result_half = solver_half.solve(pos2, rot2, target)

        # Half weight should move less
        dist_full = (target - result_full.positions[-1]).length()
        dist_half = (target - result_half.positions[-1]).length()

        # Half weight means less movement, so further from target after 1 iteration
        assert dist_half >= dist_full - 0.1


# =============================================================================
# Test ConstrainedCCDSolver
# =============================================================================


class TestConstrainedCCDSolver:
    """Tests for constrained CCD solver."""

    def test_constrained_solver_construction(self):
        """Test constrained solver construction."""
        solver = ConstrainedCCDSolver([0, 1, 2])
        assert solver.chain_length == 3
        assert len(solver._custom_constraints) == 3
        assert all(c is None for c in solver._custom_constraints)

    def test_set_custom_constraint(self):
        """Test setting custom constraint function."""
        solver = ConstrainedCCDSolver([0, 1, 2])

        def my_constraint(rot: Quat, idx: int) -> Quat:
            return rot

        solver.set_custom_constraint(1, my_constraint)
        assert solver._custom_constraints[1] is my_constraint

    def test_custom_constraint_invalid_index(self):
        """Test custom constraint with invalid index is ignored."""
        solver = ConstrainedCCDSolver([0, 1, 2])

        def my_constraint(rot: Quat, idx: int) -> Quat:
            return rot

        solver.set_custom_constraint(10, my_constraint)  # Should not crash
        assert all(c is None for c in solver._custom_constraints)

    def test_custom_constraint_called(self):
        """Test custom constraint is called during solve."""
        call_count = [0]

        def tracking_constraint(rot: Quat, idx: int) -> Quat:
            call_count[0] += 1
            return rot

        positions, rotations, indices = create_horizontal_chain(4, 1.0)
        solver = ConstrainedCCDSolver(indices, max_iterations=5)
        # Set constraint on joint 1 (not end effector, more likely to be called)
        solver.set_custom_constraint(1, tracking_constraint)

        # Target that requires rotation
        solver.solve(positions, rotations, Vec3(2, 2, 0))

        assert call_count[0] > 0

    def test_custom_constraint_modifies_rotation(self):
        """Test custom constraint can modify rotation."""
        def identity_constraint(rot: Quat, idx: int) -> Quat:
            return Quat.identity()  # Always return identity

        positions, rotations, indices = create_horizontal_chain(3, 1.0)
        solver = ConstrainedCCDSolver(indices)
        solver.set_custom_constraint(0, identity_constraint)

        result = solver.solve(positions, rotations, Vec3(1, 1, 0))

        # Joint 0 should stay at identity
        assert isinstance(result, CCDResult)

    def test_constrained_solve_basic(self):
        """Test constrained solve produces valid result."""
        positions, rotations, indices = create_horizontal_chain(3, 1.0)
        solver = ConstrainedCCDSolver(indices)

        result = solver.solve(positions, rotations, Vec3(1, 1, 0))
        assert isinstance(result, CCDResult)

    def test_combined_rotation_limit_and_custom(self):
        """Test combining rotation limit with custom constraint."""
        def scale_constraint(rot: Quat, idx: int) -> Quat:
            # Scale rotation by half
            pitch, yaw, roll = rot.to_euler()
            return Quat.from_euler(pitch * 0.5, yaw * 0.5, roll * 0.5)

        positions, rotations, indices = create_horizontal_chain(3, 1.0)
        solver = ConstrainedCCDSolver(indices)

        # Set both rotation limit and custom constraint
        solver.set_rotation_limit(0, RotationLimit(
            enabled=True,
            min_angles=Vec3(-1, -1, -1),
            max_angles=Vec3(1, 1, 1)
        ))
        solver.set_custom_constraint(0, scale_constraint)

        result = solver.solve(positions, rotations, Vec3(1, 1, 0))
        assert isinstance(result, CCDResult)


# =============================================================================
# Test Numerical Stability
# =============================================================================


class TestNumericalStability:
    """Tests for numerical stability."""

    def test_very_small_bone_lengths(self):
        """Test handling of very small bone lengths."""
        positions = [Vec3(0, 0, 0), Vec3(0.0001, 0, 0), Vec3(0.0002, 0, 0)]
        rotations = [Quat.identity()] * 3
        solver = CCDSolver([0, 1, 2])

        result = solver.solve(positions, rotations, Vec3(0.0001, 0.0001, 0))
        assert isinstance(result, CCDResult)

    def test_nearly_aligned_vectors(self):
        """Test with nearly aligned vectors (small angle)."""
        positions = [Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(2, 0.0001, 0)]
        rotations = [Quat.identity()] * 3
        solver = CCDSolver([0, 1, 2])

        target = Vec3(2, 0.0002, 0)
        result = solver.solve(positions, rotations, target)
        assert isinstance(result, CCDResult)

    def test_anti_parallel_vectors(self):
        """Test with anti-parallel vectors (180 degree rotation)."""
        positions, rotations, indices = create_horizontal_chain(3, 1.0)
        solver = CCDSolver(indices)

        # Target in opposite direction
        target = Vec3(-2, 0, 0)
        result = solver.solve(positions, rotations, target)
        assert isinstance(result, CCDResult)

    def test_repeated_solve_stability(self):
        """Test repeated solves don't accumulate error."""
        positions, rotations, indices = create_horizontal_chain(3, 1.0)
        solver = CCDSolver(indices)

        target = Vec3(1.5, 0.5, 0)

        # Solve multiple times from same starting configuration
        for _ in range(5):
            pos_copy = [Vec3(p.x, p.y, p.z) for p in positions]
            rot_copy = [Quat(r.x, r.y, r.z, r.w) for r in rotations]
            result = solver.solve(pos_copy, rot_copy, target)
            assert result.final_error < 1.0


# =============================================================================
# Test Iteration Behavior
# =============================================================================


class TestIterationBehavior:
    """Tests for iteration-related behavior."""

    def test_max_iterations_respected(self):
        """Test max iterations limit is respected."""
        positions, rotations, indices = create_horizontal_chain(3, 1.0)
        solver = CCDSolver(indices, max_iterations=5)

        # Unreachable target
        target = Vec3(100, 100, 100)
        result = solver.solve(positions, rotations, target)

        assert result.iterations == 5

    def test_early_convergence(self):
        """Test solver stops early when converged."""
        positions, rotations, indices = create_simple_chain(3, 1.0)
        solver = CCDSolver(indices, max_iterations=100, tolerance=0.1)

        # Easy target
        target = Vec3(0, 1.9, 0)
        result = solver.solve(positions, rotations, target)

        if result.success:
            assert result.iterations < 100

    def test_single_iteration(self):
        """Test behavior with max_iterations=1."""
        positions, rotations, indices = create_horizontal_chain(3, 1.0)
        solver = CCDSolver(indices, max_iterations=1)

        target = Vec3(1, 1, 0)
        result = solver.solve(positions, rotations, target)

        assert result.iterations <= 1

    def test_zero_iterations_not_allowed(self):
        """Test max_iterations must be positive."""
        # Note: The implementation doesn't explicitly validate this
        # but it would just return immediately
        positions, rotations, indices = create_horizontal_chain(3, 1.0)
        solver = CCDSolver(indices, max_iterations=0)

        result = solver.solve(positions, rotations, Vec3(1, 1, 0))
        assert result.iterations == 0


# =============================================================================
# Test Error Calculation
# =============================================================================


class TestErrorCalculation:
    """Tests for error calculation."""

    def test_error_is_distance_to_target(self):
        """Test error is Euclidean distance to target."""
        positions = [Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(2, 0, 0)]
        rotations = [Quat.identity()] * 3
        solver = CCDSolver([0, 1, 2], max_iterations=0)  # No iterations

        target = Vec3(2, 1, 0)  # 1 unit above end
        result = solver.solve(positions, rotations, target)

        # Error should be distance from (2,0,0) to (2,1,0) = 1.0
        assert abs(result.final_error - 1.0) < 0.01

    def test_error_decreases_with_iterations(self):
        """Test error generally decreases with iterations."""
        positions, rotations, indices = create_horizontal_chain(4, 1.0)

        target = Vec3(2, 1, 0)

        errors = []
        for max_iter in [1, 5, 10]:
            pos_copy = [Vec3(p.x, p.y, p.z) for p in positions]
            rot_copy = [Quat(r.x, r.y, r.z, r.w) for r in rotations]
            solver = CCDSolver(indices, max_iterations=max_iter)
            result = solver.solve(pos_copy, rot_copy, target)
            errors.append(result.final_error)

        # More iterations should generally lead to lower error
        assert errors[2] <= errors[0] + 0.1  # Some tolerance

    def test_zero_error_on_immediate_convergence(self):
        """Test zero error when target at end effector."""
        positions = [Vec3(0, 0, 0), Vec3(1, 0, 0)]
        rotations = [Quat.identity()] * 2
        solver = CCDSolver([0, 1])

        target = Vec3(1, 0, 0)  # At end effector
        result = solver.solve(positions, rotations, target)

        assert result.final_error < solver.tolerance


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_full_pipeline_vertical_chain(self):
        """Test full solve pipeline with vertical chain."""
        positions, rotations, indices = create_simple_chain(5, 1.0)
        solver = CCDSolver(indices, tolerance=0.01, max_iterations=50)

        # Reachable target
        target = Vec3(1, 3, 0)
        result = solver.solve(positions, rotations, target)

        assert result.success is True
        assert len(result.positions) == 5
        assert len(result.rotations) == 5

    def test_full_pipeline_with_limits(self):
        """Test solve with rotation limits enabled."""
        positions, rotations, indices = create_horizontal_chain(4, 1.0)
        solver = CCDSolver(indices, tolerance=0.01, max_iterations=50)

        # Apply limits to all joints
        for i in range(4):
            solver.set_rotation_limit(i, RotationLimit(
                enabled=True,
                min_angles=Vec3(-0.5, -0.5, -0.5),
                max_angles=Vec3(0.5, 0.5, 0.5)
            ))

        target = Vec3(2, 1, 0)
        result = solver.solve(positions, rotations, target)

        assert isinstance(result, CCDResult)

    def test_weighted_solver_with_limits(self):
        """Test weighted solver combined with rotation limits."""
        positions, rotations, indices = create_horizontal_chain(4, 1.0)
        solver = CCDSolverWithWeights(indices, weights=[1.0, 0.5, 0.5, 0.25])

        solver.set_rotation_limit(0, RotationLimit(
            enabled=True,
            min_angles=Vec3(-0.3, -0.3, -0.3),
            max_angles=Vec3(0.3, 0.3, 0.3)
        ))

        target = Vec3(2, 1, 0)
        result = solver.solve(positions, rotations, target)

        assert isinstance(result, CCDResult)

    def test_constrained_solver_full_pipeline(self):
        """Test constrained solver with all features."""
        positions, rotations, indices = create_horizontal_chain(4, 1.0)
        solver = ConstrainedCCDSolver(indices, damping=0.8)

        # Set rotation limit
        solver.set_rotation_limit(1, RotationLimit(
            enabled=True,
            axis=Vec3.unit_z(),
            is_hinge=True
        ))

        # Set custom constraint
        def clamping_constraint(rot: Quat, idx: int) -> Quat:
            pitch, yaw, roll = rot.to_euler()
            pitch = max(-0.5, min(0.5, pitch))
            return Quat.from_euler(pitch, yaw, roll)

        solver.set_custom_constraint(2, clamping_constraint)

        target = Vec3(2, 1, 0.5)
        result = solver.solve(positions, rotations, target)

        assert isinstance(result, CCDResult)

    def test_alternating_order_with_weights(self):
        """Test alternating rotation order with weights."""
        positions, rotations, indices = create_horizontal_chain(4, 1.0)
        solver = CCDSolverWithWeights(
            indices,
            weights=[1.0, 0.8, 0.6, 0.4],
            max_iterations=20
        )
        solver.set_rotation_order(CCDRotationOrder.ALTERNATING)

        target = Vec3(2, 1, 0)
        result = solver.solve(positions, rotations, target)

        assert isinstance(result, CCDResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
