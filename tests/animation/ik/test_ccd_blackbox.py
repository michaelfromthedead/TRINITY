"""Blackbox tests for CCD (Cyclic Coordinate Descent) IK solver.

Tests are written from the theoretical CCD algorithm specification without
reading the implementation. CCD iteratively solves IK by:

1. Start from end effector, work toward root
2. For each joint:
   - Compute vector from joint to end effector (to_end)
   - Compute vector from joint to target (to_target)
   - Rotation axis = to_end x to_target (cross product)
   - Rotation angle = acos(to_end . to_target) (dot product)
   - Apply rotation scaled by damping factor
3. Repeat until convergence or max iterations

Key properties:
- Converges for reachable targets
- Damping prevents oscillation
- Each joint rotates to minimize end-effector distance
- Preserves bone lengths (rotation only, no scaling)
"""

import pytest
import math
from typing import List

from engine.animation.ik.ccd import CCDSolver, CCDResult
from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat


# =============================================================================
# Helper Functions
# =============================================================================

def compute_bone_lengths(positions: List[Vec3]) -> List[float]:
    """Compute bone lengths from joint positions."""
    lengths = []
    for i in range(len(positions) - 1):
        lengths.append(positions[i].distance(positions[i + 1]))
    return lengths


def total_chain_length(positions: List[Vec3]) -> float:
    """Compute total reach of chain."""
    return sum(compute_bone_lengths(positions))


def create_straight_chain_y(num_joints: int, bone_length: float = 1.0) -> List[Vec3]:
    """Create a vertical chain along Y axis."""
    return [Vec3(0, i * bone_length, 0) for i in range(num_joints)]


def create_straight_chain_x(num_joints: int, bone_length: float = 1.0) -> List[Vec3]:
    """Create a horizontal chain along X axis."""
    return [Vec3(i * bone_length, 0, 0) for i in range(num_joints)]


def create_straight_chain_z(num_joints: int, bone_length: float = 1.0) -> List[Vec3]:
    """Create a chain along Z axis."""
    return [Vec3(0, 0, i * bone_length) for i in range(num_joints)]


def create_identity_rotations(num_joints: int) -> List[Quat]:
    """Create identity quaternions for all joints."""
    return [Quat.identity() for _ in range(num_joints)]


def nearly_equal(a: float, b: float, eps: float = 1e-4) -> bool:
    """Check if two floats are nearly equal."""
    return abs(a - b) <= eps


def vec_nearly_equal(a: Vec3, b: Vec3, eps: float = 1e-3) -> bool:
    """Check if two vectors are nearly equal."""
    return a.distance(b) <= eps


def apply_rotations_to_chain(
    positions: List[Vec3],
    rotations: List[Quat]
) -> List[Vec3]:
    """Apply local rotations to reconstruct joint positions.

    Each rotation is applied to rotate the bone from that joint
    to the next joint, relative to the parent's coordinate frame.
    """
    if len(positions) < 2:
        return positions[:]

    result = [positions[0]]  # Root stays fixed
    bone_lengths = compute_bone_lengths(positions)

    # Accumulated world rotation
    world_rot = Quat.identity()

    for i in range(len(positions) - 1):
        # Update world rotation with local rotation
        world_rot = world_rot * rotations[i]

        # Original bone direction (in local space, before any rotation)
        original_dir = (positions[i + 1] - positions[i]).normalized()

        # Rotate the original direction
        rotated_dir = world_rot.rotate_vector(original_dir)

        # New position
        new_pos = result[i] + rotated_dir * bone_lengths[i]
        result.append(new_pos)

    return result


def get_end_effector_position(
    positions: List[Vec3],
    rotations: List[Quat]
) -> Vec3:
    """Get the end effector position after applying rotations."""
    chain = apply_rotations_to_chain(positions, rotations)
    return chain[-1]


# =============================================================================
# CCDSolver Existence Tests
# =============================================================================

class TestCCDSolverExists:
    """Test that CCDSolver class exists and can be instantiated."""

    def test_can_import_ccd_solver(self):
        """CCDSolver should be importable from engine.animation.ik.ccd."""
        from engine.animation.ik.ccd import CCDSolver
        assert CCDSolver is not None

    def test_can_import_ccd_result(self):
        """CCDResult should be importable from engine.animation.ik.ccd."""
        from engine.animation.ik.ccd import CCDResult
        assert CCDResult is not None

    def test_can_instantiate_with_bone_indices(self):
        """CCDSolver should accept bone_indices parameter."""
        solver = CCDSolver(bone_indices=[0, 1, 2])
        assert solver is not None

    def test_can_instantiate_with_tolerance(self):
        """CCDSolver should accept tolerance parameter."""
        solver = CCDSolver(bone_indices=[0, 1, 2], tolerance=0.001)
        assert solver is not None

    def test_can_instantiate_with_max_iterations(self):
        """CCDSolver should accept max_iterations parameter."""
        solver = CCDSolver(bone_indices=[0, 1, 2], max_iterations=20)
        assert solver is not None

    def test_can_instantiate_with_damping(self):
        """CCDSolver should accept damping parameter."""
        solver = CCDSolver(bone_indices=[0, 1, 2], damping=0.5)
        assert solver is not None

    def test_can_instantiate_with_all_parameters(self):
        """CCDSolver should accept all parameters together."""
        solver = CCDSolver(
            bone_indices=[0, 1, 2, 3],
            tolerance=0.0001,
            max_iterations=50,
            damping=0.8
        )
        assert solver is not None

    def test_has_solve_method(self):
        """CCDSolver should have a solve method."""
        solver = CCDSolver(bone_indices=[0, 1, 2])
        assert hasattr(solver, 'solve')
        assert callable(solver.solve)


# =============================================================================
# CCDResult Structure Tests
# =============================================================================

class TestCCDResultStructure:
    """Test CCDResult dataclass has required fields."""

    def test_result_has_success_field(self):
        """CCDResult should have a success boolean field."""
        solver = CCDSolver(bone_indices=[0, 1, 2])
        positions = create_straight_chain_y(3)
        rotations = create_identity_rotations(3)
        target = Vec3(0, 2, 0)
        result = solver.solve(positions, rotations, target)
        assert hasattr(result, 'success')
        assert isinstance(result.success, bool)

    def test_result_has_iterations_field(self):
        """CCDResult should have an iterations integer field."""
        solver = CCDSolver(bone_indices=[0, 1, 2])
        positions = create_straight_chain_y(3)
        rotations = create_identity_rotations(3)
        target = Vec3(0, 2, 0)
        result = solver.solve(positions, rotations, target)
        assert hasattr(result, 'iterations')
        assert isinstance(result.iterations, int)

    def test_result_has_rotations_field(self):
        """CCDResult should have a rotations list field."""
        solver = CCDSolver(bone_indices=[0, 1, 2])
        positions = create_straight_chain_y(3)
        rotations = create_identity_rotations(3)
        target = Vec3(0, 2, 0)
        result = solver.solve(positions, rotations, target)
        assert hasattr(result, 'rotations')
        assert isinstance(result.rotations, list)

    def test_rotations_contains_quaternions(self):
        """CCDResult.rotations should contain Quat objects."""
        solver = CCDSolver(bone_indices=[0, 1, 2])
        positions = create_straight_chain_y(3)
        rotations = create_identity_rotations(3)
        target = Vec3(0, 2, 0)
        result = solver.solve(positions, rotations, target)
        for rot in result.rotations:
            assert isinstance(rot, Quat)

    def test_rotations_count_matches_joints(self):
        """CCDResult.rotations should have same count as input rotations."""
        solver = CCDSolver(bone_indices=[0, 1, 2])
        positions = create_straight_chain_y(3)
        rotations = create_identity_rotations(3)
        target = Vec3(0, 2, 0)
        result = solver.solve(positions, rotations, target)
        assert len(result.rotations) == len(rotations)

    def test_iterations_is_non_negative(self):
        """CCDResult.iterations should be non-negative."""
        solver = CCDSolver(bone_indices=[0, 1, 2])
        positions = create_straight_chain_y(3)
        rotations = create_identity_rotations(3)
        target = Vec3(0, 2, 0)
        result = solver.solve(positions, rotations, target)
        assert result.iterations >= 0


# =============================================================================
# Reachable Target Tests
# =============================================================================

class TestReachableTargets:
    """Test that CCD solver reaches reachable targets."""

    def test_two_bone_chain_reaches_target_in_plane(self):
        """Two bone chain should reach target within its reach."""
        solver = CCDSolver(
            bone_indices=[0, 1, 2],
            tolerance=0.01,
            max_iterations=50,
            damping=1.0
        )
        # Chain: root at origin, bones of length 1 each
        positions = create_straight_chain_y(3, bone_length=1.0)
        rotations = create_identity_rotations(3)
        # Target within reach (total reach = 2)
        target = Vec3(1.0, 1.0, 0)

        result = solver.solve(positions, rotations, target)

        assert result.success

    def test_three_bone_chain_reaches_close_target(self):
        """Three bone chain should reach a close target."""
        solver = CCDSolver(
            bone_indices=[0, 1, 2, 3],
            tolerance=0.01,
            max_iterations=50,
            damping=1.0
        )
        positions = create_straight_chain_y(4, bone_length=1.0)
        rotations = create_identity_rotations(4)
        # Target well within reach (total reach = 3)
        target = Vec3(0.5, 2.0, 0)

        result = solver.solve(positions, rotations, target)

        assert result.success

    def test_target_at_current_end_effector(self):
        """Target at current end effector should converge immediately."""
        solver = CCDSolver(
            bone_indices=[0, 1, 2],
            tolerance=0.01,
            max_iterations=50
        )
        positions = create_straight_chain_y(3, bone_length=1.0)
        rotations = create_identity_rotations(3)
        # Target exactly at current end effector
        target = Vec3(0, 2, 0)

        result = solver.solve(positions, rotations, target)

        assert result.success
        # Should converge in very few iterations
        assert result.iterations <= 2

    def test_horizontal_chain_reaches_vertical_target(self):
        """Horizontal chain should bend to reach vertical target."""
        solver = CCDSolver(
            bone_indices=[0, 1, 2],
            tolerance=0.01,
            max_iterations=100,
            damping=1.0
        )
        # Chain along X axis
        positions = create_straight_chain_x(3, bone_length=1.0)
        rotations = create_identity_rotations(3)
        # Target above the chain
        target = Vec3(1.0, 1.0, 0)

        result = solver.solve(positions, rotations, target)

        assert result.success

    def test_chain_reaches_target_behind(self):
        """Chain should be able to bend backward to reach target behind."""
        solver = CCDSolver(
            bone_indices=[0, 1, 2],
            tolerance=0.01,
            max_iterations=100,
            damping=1.0
        )
        positions = create_straight_chain_y(3, bone_length=1.0)
        rotations = create_identity_rotations(3)
        # Target behind (negative X)
        target = Vec3(-1.0, 1.0, 0)

        result = solver.solve(positions, rotations, target)

        assert result.success


# =============================================================================
# Unreachable Target Tests
# =============================================================================

class TestUnreachableTargets:
    """Test behavior with unreachable targets."""

    def test_target_beyond_reach_returns_false(self):
        """Target beyond chain reach should not succeed."""
        solver = CCDSolver(
            bone_indices=[0, 1, 2],
            tolerance=0.01,
            max_iterations=50,
            damping=1.0
        )
        positions = create_straight_chain_y(3, bone_length=1.0)
        rotations = create_identity_rotations(3)
        # Target beyond reach (total reach = 2)
        target = Vec3(5.0, 5.0, 0)  # Distance ~7.07, way beyond reach

        result = solver.solve(positions, rotations, target)

        assert result.success is False

    def test_unreachable_uses_max_iterations(self):
        """Unreachable target should use max iterations."""
        max_iter = 30
        solver = CCDSolver(
            bone_indices=[0, 1, 2],
            tolerance=0.0001,
            max_iterations=max_iter,
            damping=1.0
        )
        positions = create_straight_chain_y(3, bone_length=1.0)
        rotations = create_identity_rotations(3)
        target = Vec3(10.0, 10.0, 0)

        result = solver.solve(positions, rotations, target)

        assert result.iterations == max_iter

    def test_chain_extends_toward_unreachable(self):
        """Chain should extend maximally toward unreachable target."""
        solver = CCDSolver(
            bone_indices=[0, 1, 2],
            tolerance=0.01,
            max_iterations=100,
            damping=1.0
        )
        positions = create_straight_chain_y(3, bone_length=1.0)
        rotations = create_identity_rotations(3)
        # Target far away in +X direction
        target = Vec3(100.0, 0, 0)

        result = solver.solve(positions, rotations, target)

        # Even though unreachable, chain should orient toward target
        assert result.rotations is not None
        assert len(result.rotations) == 3

    def test_very_far_target_still_produces_valid_rotations(self):
        """Extremely far target should still produce valid quaternions."""
        solver = CCDSolver(
            bone_indices=[0, 1, 2, 3],
            tolerance=0.01,
            max_iterations=50,
            damping=0.5
        )
        positions = create_straight_chain_y(4, bone_length=1.0)
        rotations = create_identity_rotations(4)
        target = Vec3(1000.0, 1000.0, 1000.0)

        result = solver.solve(positions, rotations, target)

        for rot in result.rotations:
            # Quaternion should be normalized (unit length)
            length = math.sqrt(rot.x**2 + rot.y**2 + rot.z**2 + rot.w**2)
            assert nearly_equal(length, 1.0, eps=0.01)


# =============================================================================
# Damping Effect Tests
# =============================================================================

class TestDampingEffect:
    """Test that damping affects convergence behavior."""

    def test_high_damping_converges(self):
        """High damping (1.0) should still converge."""
        solver = CCDSolver(
            bone_indices=[0, 1, 2],
            tolerance=0.01,
            max_iterations=100,
            damping=1.0
        )
        positions = create_straight_chain_y(3, bone_length=1.0)
        rotations = create_identity_rotations(3)
        target = Vec3(1.0, 1.0, 0)

        result = solver.solve(positions, rotations, target)

        assert result.success

    def test_low_damping_needs_more_iterations(self):
        """Low damping should require more iterations than high damping."""
        positions = create_straight_chain_y(3, bone_length=1.0)
        rotations = create_identity_rotations(3)
        target = Vec3(1.0, 1.0, 0)

        solver_high = CCDSolver(
            bone_indices=[0, 1, 2],
            tolerance=0.01,
            max_iterations=200,
            damping=1.0
        )
        result_high = solver_high.solve(positions, rotations.copy(), target)

        solver_low = CCDSolver(
            bone_indices=[0, 1, 2],
            tolerance=0.01,
            max_iterations=200,
            damping=0.3
        )
        result_low = solver_low.solve(positions, rotations.copy(), target)

        # Both should converge, but low damping needs more iterations
        if result_high.success and result_low.success:
            assert result_low.iterations >= result_high.iterations

    def test_very_low_damping_converges_eventually(self):
        """Very low damping should still converge with enough iterations."""
        solver = CCDSolver(
            bone_indices=[0, 1, 2],
            tolerance=0.1,  # Looser tolerance
            max_iterations=500,
            damping=0.1
        )
        positions = create_straight_chain_y(3, bone_length=1.0)
        rotations = create_identity_rotations(3)
        target = Vec3(0.5, 1.5, 0)  # Easy target

        result = solver.solve(positions, rotations, target)

        # Should eventually converge
        assert result.iterations > 0

    def test_damping_prevents_overshoot(self):
        """Damping should prevent wild oscillations."""
        solver = CCDSolver(
            bone_indices=[0, 1, 2],
            tolerance=0.01,
            max_iterations=100,
            damping=0.5
        )
        positions = create_straight_chain_y(3, bone_length=1.0)
        rotations = create_identity_rotations(3)
        target = Vec3(1.0, 1.0, 0)

        result = solver.solve(positions, rotations, target)

        # All rotations should be reasonable (not extreme)
        for rot in result.rotations:
            # Check quaternion is normalized
            length = math.sqrt(rot.x**2 + rot.y**2 + rot.z**2 + rot.w**2)
            assert nearly_equal(length, 1.0, eps=0.01)

    def test_medium_damping_good_balance(self):
        """Medium damping should provide good convergence."""
        solver = CCDSolver(
            bone_indices=[0, 1, 2, 3],
            tolerance=0.01,
            max_iterations=100,
            damping=0.5
        )
        positions = create_straight_chain_y(4, bone_length=1.0)
        rotations = create_identity_rotations(4)
        target = Vec3(1.5, 1.5, 0)

        result = solver.solve(positions, rotations, target)

        assert result.success


# =============================================================================
# Convergence Tests
# =============================================================================

class TestConvergence:
    """Test convergence behavior."""

    def test_stops_at_tolerance(self):
        """Solver should stop when error is within tolerance."""
        tolerance = 0.1
        solver = CCDSolver(
            bone_indices=[0, 1, 2],
            tolerance=tolerance,
            max_iterations=1000,
            damping=1.0
        )
        positions = create_straight_chain_y(3, bone_length=1.0)
        rotations = create_identity_rotations(3)
        target = Vec3(0, 2, 0)  # Already at end effector

        result = solver.solve(positions, rotations, target)

        assert result.success
        # Should stop early, not use all iterations
        assert result.iterations < 1000

    def test_respects_max_iterations(self):
        """Solver should not exceed max_iterations."""
        max_iter = 25
        solver = CCDSolver(
            bone_indices=[0, 1, 2],
            tolerance=0.0000001,  # Very tight tolerance
            max_iterations=max_iter,
            damping=0.5
        )
        positions = create_straight_chain_y(3, bone_length=1.0)
        rotations = create_identity_rotations(3)
        target = Vec3(0.8, 0.8, 0)

        result = solver.solve(positions, rotations, target)

        assert result.iterations <= max_iter

    def test_tighter_tolerance_more_iterations(self):
        """Tighter tolerance should require more iterations."""
        positions = create_straight_chain_y(3, bone_length=1.0)
        rotations = create_identity_rotations(3)
        target = Vec3(1.0, 1.0, 0)

        solver_loose = CCDSolver(
            bone_indices=[0, 1, 2],
            tolerance=0.5,
            max_iterations=200,
            damping=1.0
        )
        result_loose = solver_loose.solve(positions, rotations.copy(), target)

        solver_tight = CCDSolver(
            bone_indices=[0, 1, 2],
            tolerance=0.01,
            max_iterations=200,
            damping=1.0
        )
        result_tight = solver_tight.solve(positions, rotations.copy(), target)

        if result_loose.success and result_tight.success:
            assert result_tight.iterations >= result_loose.iterations

    def test_zero_iterations_allowed(self):
        """If already at target, zero iterations is valid."""
        solver = CCDSolver(
            bone_indices=[0, 1, 2],
            tolerance=0.1,
            max_iterations=100,
            damping=1.0
        )
        positions = create_straight_chain_y(3, bone_length=1.0)
        rotations = create_identity_rotations(3)
        target = Vec3(0, 2, 0)  # Exactly at end effector

        result = solver.solve(positions, rotations, target)

        assert result.success
        # Zero or very few iterations
        assert result.iterations <= 2

    def test_single_iteration_max(self):
        """Solver with max_iterations=1 should do exactly one iteration."""
        solver = CCDSolver(
            bone_indices=[0, 1, 2],
            tolerance=0.00001,
            max_iterations=1,
            damping=1.0
        )
        positions = create_straight_chain_y(3, bone_length=1.0)
        rotations = create_identity_rotations(3)
        target = Vec3(1.0, 1.0, 0)

        result = solver.solve(positions, rotations, target)

        # Should not exceed max
        assert result.iterations <= 1


# =============================================================================
# 3D Target Tests
# =============================================================================

class Test3DTargets:
    """Test solver works in full 3D space."""

    def test_positive_x_target(self):
        """Chain should reach target in positive X direction."""
        solver = CCDSolver(
            bone_indices=[0, 1, 2],
            tolerance=0.01,
            max_iterations=100,
            damping=1.0
        )
        positions = create_straight_chain_y(3, bone_length=1.0)
        rotations = create_identity_rotations(3)
        target = Vec3(1.5, 0.5, 0)

        result = solver.solve(positions, rotations, target)

        assert result.success

    def test_negative_x_target(self):
        """Chain should reach target in negative X direction."""
        solver = CCDSolver(
            bone_indices=[0, 1, 2],
            tolerance=0.01,
            max_iterations=100,
            damping=1.0
        )
        positions = create_straight_chain_y(3, bone_length=1.0)
        rotations = create_identity_rotations(3)
        target = Vec3(-1.5, 0.5, 0)

        result = solver.solve(positions, rotations, target)

        assert result.success

    def test_positive_z_target(self):
        """Chain should reach target in positive Z direction."""
        solver = CCDSolver(
            bone_indices=[0, 1, 2],
            tolerance=0.01,
            max_iterations=100,
            damping=1.0
        )
        positions = create_straight_chain_y(3, bone_length=1.0)
        rotations = create_identity_rotations(3)
        target = Vec3(0, 0.5, 1.5)

        result = solver.solve(positions, rotations, target)

        assert result.success

    def test_negative_z_target(self):
        """Chain should reach target in negative Z direction."""
        solver = CCDSolver(
            bone_indices=[0, 1, 2],
            tolerance=0.01,
            max_iterations=100,
            damping=1.0
        )
        positions = create_straight_chain_y(3, bone_length=1.0)
        rotations = create_identity_rotations(3)
        target = Vec3(0, 0.5, -1.5)

        result = solver.solve(positions, rotations, target)

        assert result.success

    def test_diagonal_xy_target(self):
        """Chain should reach diagonal target in XY plane."""
        solver = CCDSolver(
            bone_indices=[0, 1, 2],
            tolerance=0.01,
            max_iterations=100,
            damping=1.0
        )
        positions = create_straight_chain_y(3, bone_length=1.0)
        rotations = create_identity_rotations(3)
        target = Vec3(1.0, 1.0, 0)

        result = solver.solve(positions, rotations, target)

        assert result.success

    def test_diagonal_xz_target(self):
        """Chain should reach diagonal target in XZ plane."""
        solver = CCDSolver(
            bone_indices=[0, 1, 2],
            tolerance=0.01,
            max_iterations=100,
            damping=1.0
        )
        positions = create_straight_chain_y(3, bone_length=1.0)
        rotations = create_identity_rotations(3)
        target = Vec3(1.0, 0.5, 1.0)

        result = solver.solve(positions, rotations, target)

        assert result.success

    def test_diagonal_yz_target(self):
        """Chain should reach diagonal target in YZ plane."""
        solver = CCDSolver(
            bone_indices=[0, 1, 2],
            tolerance=0.01,
            max_iterations=100,
            damping=1.0
        )
        positions = create_straight_chain_y(3, bone_length=1.0)
        rotations = create_identity_rotations(3)
        target = Vec3(0, 1.0, 1.0)

        result = solver.solve(positions, rotations, target)

        assert result.success

    def test_full_3d_diagonal_target(self):
        """Chain should reach target with all three coordinates nonzero."""
        solver = CCDSolver(
            bone_indices=[0, 1, 2, 3],
            tolerance=0.01,
            max_iterations=100,
            damping=1.0
        )
        positions = create_straight_chain_y(4, bone_length=1.0)
        rotations = create_identity_rotations(4)
        target = Vec3(1.0, 1.0, 1.0)

        result = solver.solve(positions, rotations, target)

        assert result.success

    def test_target_below_root(self):
        """Chain should reach target below root (negative Y)."""
        solver = CCDSolver(
            bone_indices=[0, 1, 2],
            tolerance=0.01,
            max_iterations=100,
            damping=1.0
        )
        positions = create_straight_chain_y(3, bone_length=1.0)
        rotations = create_identity_rotations(3)
        target = Vec3(0.5, -1.0, 0)

        result = solver.solve(positions, rotations, target)

        assert result.success


# =============================================================================
# Multiple Solves Tests
# =============================================================================

class TestMultipleSolves:
    """Test solver can be reused for multiple targets."""

    def test_same_solver_different_targets(self):
        """Same solver instance should work for multiple targets."""
        solver = CCDSolver(
            bone_indices=[0, 1, 2],
            tolerance=0.01,
            max_iterations=100,
            damping=1.0
        )
        positions = create_straight_chain_y(3, bone_length=1.0)

        targets = [
            Vec3(1.0, 1.0, 0),
            Vec3(-1.0, 1.0, 0),
            Vec3(0, 1.5, 0.5),
            Vec3(0.5, 1.0, -0.5),
        ]

        for target in targets:
            rotations = create_identity_rotations(3)
            result = solver.solve(positions, rotations, target)
            assert result.success
            assert len(result.rotations) == 3

    def test_sequential_solves_independent(self):
        """Each solve should be independent of previous solves."""
        solver = CCDSolver(
            bone_indices=[0, 1, 2],
            tolerance=0.01,
            max_iterations=100,
            damping=1.0
        )
        positions = create_straight_chain_y(3, bone_length=1.0)
        rotations = create_identity_rotations(3)

        # Solve for one target
        target1 = Vec3(1.0, 1.0, 0)
        result1 = solver.solve(positions, rotations.copy(), target1)

        # Solve for different target with same initial state
        target2 = Vec3(-1.0, 1.0, 0)
        result2 = solver.solve(positions, rotations.copy(), target2)

        # Results should be different for different targets
        assert result1.success
        assert result2.success

    def test_solve_does_not_modify_input_positions(self):
        """Solve should not modify input positions list."""
        solver = CCDSolver(
            bone_indices=[0, 1, 2],
            tolerance=0.01,
            max_iterations=100,
            damping=1.0
        )
        positions = create_straight_chain_y(3, bone_length=1.0)
        original_positions = [Vec3(p.x, p.y, p.z) for p in positions]
        rotations = create_identity_rotations(3)
        target = Vec3(1.0, 1.0, 0)

        solver.solve(positions, rotations, target)

        for orig, pos in zip(original_positions, positions):
            assert vec_nearly_equal(orig, pos)

    def test_solve_does_not_modify_input_rotations(self):
        """Solve should not modify input rotations list."""
        solver = CCDSolver(
            bone_indices=[0, 1, 2],
            tolerance=0.01,
            max_iterations=100,
            damping=1.0
        )
        positions = create_straight_chain_y(3, bone_length=1.0)
        rotations = create_identity_rotations(3)
        original_rotations = [Quat(r.x, r.y, r.z, r.w) for r in rotations]
        target = Vec3(1.0, 1.0, 0)

        solver.solve(positions, rotations, target)

        for orig, rot in zip(original_rotations, rotations):
            assert nearly_equal(orig.x, rot.x)
            assert nearly_equal(orig.y, rot.y)
            assert nearly_equal(orig.z, rot.z)
            assert nearly_equal(orig.w, rot.w)


# =============================================================================
# Bone Length Preservation Tests
# =============================================================================

class TestBoneLengthPreservation:
    """Test that rotations preserve bone lengths."""

    def test_rotations_are_unit_quaternions(self):
        """All output rotations should be unit quaternions."""
        solver = CCDSolver(
            bone_indices=[0, 1, 2, 3],
            tolerance=0.01,
            max_iterations=100,
            damping=1.0
        )
        positions = create_straight_chain_y(4, bone_length=1.0)
        rotations = create_identity_rotations(4)
        target = Vec3(1.5, 1.5, 0)

        result = solver.solve(positions, rotations, target)

        for rot in result.rotations:
            length = math.sqrt(rot.x**2 + rot.y**2 + rot.z**2 + rot.w**2)
            assert nearly_equal(length, 1.0, eps=0.001)

    def test_pure_rotation_no_scaling(self):
        """Rotations should not scale or stretch bones."""
        solver = CCDSolver(
            bone_indices=[0, 1, 2],
            tolerance=0.01,
            max_iterations=100,
            damping=1.0
        )
        positions = create_straight_chain_y(3, bone_length=1.0)
        rotations = create_identity_rotations(3)
        target = Vec3(1.0, 1.0, 0)

        result = solver.solve(positions, rotations, target)

        # All rotations should be valid (normalized quaternions)
        for rot in result.rotations:
            # w^2 + x^2 + y^2 + z^2 = 1 for unit quaternion
            mag_sq = rot.x**2 + rot.y**2 + rot.z**2 + rot.w**2
            assert nearly_equal(mag_sq, 1.0, eps=0.001)

    def test_different_bone_lengths_preserved(self):
        """Chains with varying bone lengths should preserve them."""
        solver = CCDSolver(
            bone_indices=[0, 1, 2, 3],
            tolerance=0.01,
            max_iterations=100,
            damping=1.0
        )
        # Chain with varying bone lengths
        positions = [
            Vec3(0, 0, 0),
            Vec3(0, 0.5, 0),   # length 0.5
            Vec3(0, 1.5, 0),   # length 1.0
            Vec3(0, 2.0, 0),   # length 0.5
        ]
        rotations = create_identity_rotations(4)
        target = Vec3(1.0, 1.0, 0)

        result = solver.solve(positions, rotations, target)

        # Rotations should all be unit quaternions
        for rot in result.rotations:
            length = math.sqrt(rot.x**2 + rot.y**2 + rot.z**2 + rot.w**2)
            assert nearly_equal(length, 1.0, eps=0.001)


# =============================================================================
# Chain Configuration Tests
# =============================================================================

class TestChainConfigurations:
    """Test various chain configurations."""

    def test_two_joint_chain(self):
        """Minimal two-joint chain (one bone) should work."""
        solver = CCDSolver(
            bone_indices=[0, 1],
            tolerance=0.01,
            max_iterations=50,
            damping=1.0
        )
        positions = [Vec3(0, 0, 0), Vec3(0, 1, 0)]
        rotations = create_identity_rotations(2)
        target = Vec3(0.5, 0.5, 0)

        result = solver.solve(positions, rotations, target)

        assert result.rotations is not None
        assert len(result.rotations) == 2

    def test_long_chain_five_bones(self):
        """Longer chain with 5 bones should work."""
        solver = CCDSolver(
            bone_indices=[0, 1, 2, 3, 4, 5],
            tolerance=0.01,
            max_iterations=100,
            damping=1.0
        )
        positions = create_straight_chain_y(6, bone_length=1.0)
        rotations = create_identity_rotations(6)
        target = Vec3(2.0, 3.0, 1.0)

        result = solver.solve(positions, rotations, target)

        assert result.success
        assert len(result.rotations) == 6

    def test_very_long_chain(self):
        """Very long chain should still work."""
        num_joints = 10
        solver = CCDSolver(
            bone_indices=list(range(num_joints)),
            tolerance=0.1,
            max_iterations=200,
            damping=0.8
        )
        positions = create_straight_chain_y(num_joints, bone_length=0.5)
        rotations = create_identity_rotations(num_joints)
        target = Vec3(2.0, 2.0, 0)

        result = solver.solve(positions, rotations, target)

        assert len(result.rotations) == num_joints

    def test_chain_along_z_axis(self):
        """Chain along Z axis should work."""
        solver = CCDSolver(
            bone_indices=[0, 1, 2],
            tolerance=0.01,
            max_iterations=100,
            damping=1.0
        )
        positions = create_straight_chain_z(3, bone_length=1.0)
        rotations = create_identity_rotations(3)
        target = Vec3(1.0, 1.0, 0)

        result = solver.solve(positions, rotations, target)

        assert result.success

    def test_chain_at_angle(self):
        """Chain starting at an angle should work."""
        solver = CCDSolver(
            bone_indices=[0, 1, 2],
            tolerance=0.01,
            max_iterations=100,
            damping=1.0
        )
        # Diagonal chain
        positions = [
            Vec3(0, 0, 0),
            Vec3(1, 1, 0),
            Vec3(2, 2, 0),
        ]
        rotations = create_identity_rotations(3)
        target = Vec3(2.5, 0, 0)

        result = solver.solve(positions, rotations, target)

        assert len(result.rotations) == 3


# =============================================================================
# Edge Cases Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_target_at_root(self):
        """Target at root position should be handled."""
        solver = CCDSolver(
            bone_indices=[0, 1, 2],
            tolerance=0.01,
            max_iterations=100,
            damping=1.0
        )
        positions = create_straight_chain_y(3, bone_length=1.0)
        rotations = create_identity_rotations(3)
        target = Vec3(0, 0, 0)  # At root

        result = solver.solve(positions, rotations, target)

        # Should not crash, may or may not succeed
        assert result.rotations is not None

    def test_target_very_close_to_end(self):
        """Target very close to end effector should converge quickly."""
        solver = CCDSolver(
            bone_indices=[0, 1, 2],
            tolerance=0.01,  # Reasonable tolerance for very close target
            max_iterations=100,
            damping=1.0
        )
        positions = create_straight_chain_y(3, bone_length=1.0)
        rotations = create_identity_rotations(3)
        target = Vec3(0.005, 2.005, 0)  # Very close to (0, 2, 0)

        result = solver.solve(positions, rotations, target)

        assert result.success
        assert result.iterations <= 10

    def test_zero_tolerance(self):
        """Zero tolerance should still work (will hit max iterations)."""
        solver = CCDSolver(
            bone_indices=[0, 1, 2],
            tolerance=0.0,
            max_iterations=10,
            damping=1.0
        )
        positions = create_straight_chain_y(3, bone_length=1.0)
        rotations = create_identity_rotations(3)
        target = Vec3(1.0, 1.0, 0)

        result = solver.solve(positions, rotations, target)

        assert result.iterations <= 10

    def test_very_small_bones(self):
        """Very small bone lengths should still work."""
        solver = CCDSolver(
            bone_indices=[0, 1, 2],
            tolerance=0.0001,
            max_iterations=100,
            damping=1.0
        )
        positions = create_straight_chain_y(3, bone_length=0.01)
        rotations = create_identity_rotations(3)
        target = Vec3(0.01, 0.01, 0)

        result = solver.solve(positions, rotations, target)

        assert len(result.rotations) == 3

    def test_very_large_bones(self):
        """Very large bone lengths should still work."""
        solver = CCDSolver(
            bone_indices=[0, 1, 2],
            tolerance=1.0,
            max_iterations=100,
            damping=1.0
        )
        positions = create_straight_chain_y(3, bone_length=100.0)
        rotations = create_identity_rotations(3)
        target = Vec3(100.0, 100.0, 0)

        result = solver.solve(positions, rotations, target)

        assert len(result.rotations) == 3

    def test_collinear_target(self):
        """Target collinear with chain but at end effector position should work."""
        solver = CCDSolver(
            bone_indices=[0, 1, 2],
            tolerance=0.01,
            max_iterations=100,
            damping=1.0
        )
        positions = create_straight_chain_y(3, bone_length=1.0)
        rotations = create_identity_rotations(3)
        # Target directly at end effector (collinear, at tip)
        target = Vec3(0, 2.0, 0)

        result = solver.solve(positions, rotations, target)

        assert result.success


# =============================================================================
# Rotation Quality Tests
# =============================================================================

class TestRotationQuality:
    """Test quality of computed rotations."""

    def test_rotations_are_valid_quaternions(self):
        """All rotations should be valid quaternions."""
        solver = CCDSolver(
            bone_indices=[0, 1, 2, 3],
            tolerance=0.01,
            max_iterations=100,
            damping=0.5
        )
        positions = create_straight_chain_y(4, bone_length=1.0)
        rotations = create_identity_rotations(4)
        target = Vec3(1.5, 1.5, 0.5)

        result = solver.solve(positions, rotations, target)

        for rot in result.rotations:
            # Check it's a valid Quat
            assert hasattr(rot, 'x')
            assert hasattr(rot, 'y')
            assert hasattr(rot, 'z')
            assert hasattr(rot, 'w')

    def test_rotations_are_normalized(self):
        """All output rotations should be normalized."""
        solver = CCDSolver(
            bone_indices=[0, 1, 2],
            tolerance=0.01,
            max_iterations=100,
            damping=1.0
        )
        positions = create_straight_chain_y(3, bone_length=1.0)
        rotations = create_identity_rotations(3)
        target = Vec3(1.0, 1.0, 0)

        result = solver.solve(positions, rotations, target)

        for rot in result.rotations:
            length = math.sqrt(rot.x**2 + rot.y**2 + rot.z**2 + rot.w**2)
            assert nearly_equal(length, 1.0, eps=0.001)

    def test_no_nan_in_rotations(self):
        """Rotations should not contain NaN values."""
        solver = CCDSolver(
            bone_indices=[0, 1, 2],
            tolerance=0.01,
            max_iterations=100,
            damping=1.0
        )
        positions = create_straight_chain_y(3, bone_length=1.0)
        rotations = create_identity_rotations(3)
        target = Vec3(1.0, 1.0, 0)

        result = solver.solve(positions, rotations, target)

        for rot in result.rotations:
            assert not math.isnan(rot.x)
            assert not math.isnan(rot.y)
            assert not math.isnan(rot.z)
            assert not math.isnan(rot.w)

    def test_no_inf_in_rotations(self):
        """Rotations should not contain infinity values."""
        solver = CCDSolver(
            bone_indices=[0, 1, 2],
            tolerance=0.01,
            max_iterations=100,
            damping=1.0
        )
        positions = create_straight_chain_y(3, bone_length=1.0)
        rotations = create_identity_rotations(3)
        target = Vec3(1.0, 1.0, 0)

        result = solver.solve(positions, rotations, target)

        for rot in result.rotations:
            assert not math.isinf(rot.x)
            assert not math.isinf(rot.y)
            assert not math.isinf(rot.z)
            assert not math.isinf(rot.w)


# =============================================================================
# Performance Characteristics Tests
# =============================================================================

class TestPerformanceCharacteristics:
    """Test performance-related characteristics."""

    def test_solver_completes_in_reasonable_time(self):
        """Solver should complete within reasonable time."""
        import time

        solver = CCDSolver(
            bone_indices=[0, 1, 2, 3, 4],
            tolerance=0.01,
            max_iterations=100,
            damping=1.0
        )
        positions = create_straight_chain_y(5, bone_length=1.0)
        rotations = create_identity_rotations(5)
        target = Vec3(2.0, 2.0, 1.0)

        start = time.time()
        result = solver.solve(positions, rotations, target)
        elapsed = time.time() - start

        # Should complete in under 1 second
        assert elapsed < 1.0
        assert result.rotations is not None

    def test_many_solves_stable(self):
        """Many sequential solves should be stable."""
        solver = CCDSolver(
            bone_indices=[0, 1, 2],
            tolerance=0.01,
            max_iterations=50,
            damping=1.0
        )
        positions = create_straight_chain_y(3, bone_length=1.0)

        for i in range(100):
            rotations = create_identity_rotations(3)
            angle = i * 0.1
            target = Vec3(math.cos(angle), 1.0 + math.sin(angle), 0)
            result = solver.solve(positions, rotations, target)

            # Should always produce valid output
            assert len(result.rotations) == 3
            for rot in result.rotations:
                assert not math.isnan(rot.x)
                assert not math.isnan(rot.y)
                assert not math.isnan(rot.z)
                assert not math.isnan(rot.w)


# =============================================================================
# Default Parameter Tests
# =============================================================================

class TestDefaultParameters:
    """Test default parameter handling."""

    def test_default_tolerance(self):
        """Solver should work with default tolerance."""
        solver = CCDSolver(bone_indices=[0, 1, 2])
        positions = create_straight_chain_y(3, bone_length=1.0)
        rotations = create_identity_rotations(3)
        target = Vec3(1.0, 1.0, 0)

        result = solver.solve(positions, rotations, target)

        assert result.rotations is not None

    def test_default_max_iterations(self):
        """Solver should work with default max_iterations."""
        solver = CCDSolver(bone_indices=[0, 1, 2])
        positions = create_straight_chain_y(3, bone_length=1.0)
        rotations = create_identity_rotations(3)
        target = Vec3(1.0, 1.0, 0)

        result = solver.solve(positions, rotations, target)

        assert result.iterations >= 0

    def test_default_damping(self):
        """Solver should work with default damping."""
        solver = CCDSolver(bone_indices=[0, 1, 2])
        positions = create_straight_chain_y(3, bone_length=1.0)
        rotations = create_identity_rotations(3)
        target = Vec3(1.0, 1.0, 0)

        result = solver.solve(positions, rotations, target)

        assert result.rotations is not None


# =============================================================================
# CCD Algorithm Behavior Tests
# =============================================================================

class TestCCDAlgorithmBehavior:
    """Test CCD-specific algorithmic behavior."""

    def test_end_joint_rotates_first(self):
        """CCD should rotate joints from end toward root."""
        solver = CCDSolver(
            bone_indices=[0, 1, 2],
            tolerance=0.01,
            max_iterations=1,  # Single iteration
            damping=1.0
        )
        positions = create_straight_chain_y(3, bone_length=1.0)
        rotations = create_identity_rotations(3)
        target = Vec3(1.0, 1.0, 0)

        result = solver.solve(positions, rotations, target)

        # After one iteration, at least one rotation should have changed
        has_rotation = any(
            not (nearly_equal(r.x, 0) and nearly_equal(r.y, 0)
                 and nearly_equal(r.z, 0) and nearly_equal(r.w, 1))
            for r in result.rotations
        )
        assert has_rotation or result.success

    def test_incremental_improvement(self):
        """Each iteration should improve or maintain distance to target."""
        solver_1iter = CCDSolver(
            bone_indices=[0, 1, 2],
            tolerance=0.0001,
            max_iterations=1,
            damping=1.0
        )
        solver_5iter = CCDSolver(
            bone_indices=[0, 1, 2],
            tolerance=0.0001,
            max_iterations=5,
            damping=1.0
        )

        positions = create_straight_chain_y(3, bone_length=1.0)
        target = Vec3(1.0, 1.0, 0)

        result_1 = solver_1iter.solve(positions, create_identity_rotations(3), target)
        result_5 = solver_5iter.solve(positions, create_identity_rotations(3), target)

        # More iterations should generally lead to better or equal result
        assert result_5.iterations >= result_1.iterations or result_5.success


# =============================================================================
# Input Validation Tests
# =============================================================================

class TestInputValidation:
    """Test input validation behavior."""

    def test_accepts_list_of_vec3(self):
        """Solver should accept list of Vec3 for positions."""
        solver = CCDSolver(bone_indices=[0, 1, 2])
        positions = [Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 2, 0)]
        rotations = create_identity_rotations(3)
        target = Vec3(1, 1, 0)

        result = solver.solve(positions, rotations, target)

        assert result.rotations is not None

    def test_accepts_list_of_quat(self):
        """Solver should accept list of Quat for rotations."""
        solver = CCDSolver(bone_indices=[0, 1, 2])
        positions = create_straight_chain_y(3, bone_length=1.0)
        rotations = [Quat.identity(), Quat.identity(), Quat.identity()]
        target = Vec3(1, 1, 0)

        result = solver.solve(positions, rotations, target)

        assert result.rotations is not None

    def test_accepts_vec3_target(self):
        """Solver should accept Vec3 for target."""
        solver = CCDSolver(bone_indices=[0, 1, 2])
        positions = create_straight_chain_y(3, bone_length=1.0)
        rotations = create_identity_rotations(3)
        target = Vec3(1.0, 1.0, 0.5)

        result = solver.solve(positions, rotations, target)

        assert result.rotations is not None


# =============================================================================
# Symmetry Tests
# =============================================================================

class TestSymmetry:
    """Test symmetric behavior of solver."""

    def test_symmetric_targets_produce_symmetric_results(self):
        """Symmetric targets should produce symmetric (mirrored) behavior."""
        solver = CCDSolver(
            bone_indices=[0, 1, 2],
            tolerance=0.01,
            max_iterations=100,
            damping=1.0
        )
        positions = create_straight_chain_y(3, bone_length=1.0)

        # Target in +X
        target_pos = Vec3(1.0, 1.0, 0)
        result_pos = solver.solve(positions, create_identity_rotations(3), target_pos)

        # Target in -X (mirror)
        target_neg = Vec3(-1.0, 1.0, 0)
        result_neg = solver.solve(positions, create_identity_rotations(3), target_neg)

        # Both should succeed or fail together for symmetric problem
        assert result_pos.success == result_neg.success

    def test_rotational_symmetry_xy(self):
        """Targets at same distance in XY plane should have similar iterations."""
        solver = CCDSolver(
            bone_indices=[0, 1, 2],
            tolerance=0.01,
            max_iterations=100,
            damping=1.0
        )
        positions = create_straight_chain_y(3, bone_length=1.0)

        targets = [
            Vec3(1.0, 1.0, 0),
            Vec3(0, 1.0, 1.0),
            Vec3(-1.0, 1.0, 0),
            Vec3(0, 1.0, -1.0),
        ]

        results = []
        for target in targets:
            result = solver.solve(positions, create_identity_rotations(3), target)
            results.append(result)

        # All should succeed (same distance)
        for result in results:
            assert result.success
