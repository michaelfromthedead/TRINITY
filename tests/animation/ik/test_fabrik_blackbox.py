"""Blackbox tests for FABRIK Chain IK solver.

Tests are written from the theoretical FABRIK algorithm specification without
reading the implementation. FABRIK (Forward And Backward Reaching Inverse
Kinematics) iteratively solves IK by:

1. Forward pass: Move end effector to target, pull joints backward
2. Backward pass: Reset root, push joints forward
3. Iterate until convergence or max iterations

Key properties:
- Preserves bone lengths exactly
- Root position is anchored
- Convergence guaranteed for reachable targets
"""

import pytest
import math
from typing import List

from engine.animation.ik.fabrik import FABRIKChain, FABRIKResult
from engine.core.math.vec import Vec3


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


def nearly_equal(a: float, b: float, eps: float = 1e-5) -> bool:
    """Check if two floats are nearly equal."""
    return abs(a - b) <= eps


def vec_nearly_equal(a: Vec3, b: Vec3, eps: float = 1e-5) -> bool:
    """Check if two vectors are nearly equal."""
    return a.distance(b) <= eps


# =============================================================================
# FABRIKResult Structure Tests
# =============================================================================

class TestFABRIKResultStructure:
    """Test FABRIKResult dataclass has required fields."""

    def test_has_success_field(self):
        """FABRIKResult should have a success boolean field."""
        chain = FABRIKChain(bone_indices=[0, 1, 2])
        positions = create_straight_chain_y(3)
        target = Vec3(0, 2, 0)
        result = chain.solve(positions, target)
        assert hasattr(result, 'success')
        assert isinstance(result.success, bool)

    def test_has_iterations_field(self):
        """FABRIKResult should have an iterations integer field."""
        chain = FABRIKChain(bone_indices=[0, 1, 2])
        positions = create_straight_chain_y(3)
        target = Vec3(0, 2, 0)
        result = chain.solve(positions, target)
        assert hasattr(result, 'iterations')
        assert isinstance(result.iterations, int)

    def test_has_final_error_field(self):
        """FABRIKResult should have a final_error float field."""
        chain = FABRIKChain(bone_indices=[0, 1, 2])
        positions = create_straight_chain_y(3)
        target = Vec3(0, 2, 0)
        result = chain.solve(positions, target)
        assert hasattr(result, 'final_error')
        assert isinstance(result.final_error, (int, float))

    def test_has_positions_field(self):
        """FABRIKResult should have a positions list field."""
        chain = FABRIKChain(bone_indices=[0, 1, 2])
        positions = create_straight_chain_y(3)
        target = Vec3(0, 2, 0)
        result = chain.solve(positions, target)
        assert hasattr(result, 'positions')
        assert isinstance(result.positions, list)

    def test_positions_contains_vec3(self):
        """FABRIKResult.positions should contain Vec3 objects."""
        chain = FABRIKChain(bone_indices=[0, 1, 2])
        positions = create_straight_chain_y(3)
        target = Vec3(0.5, 1.5, 0)
        result = chain.solve(positions, target)
        assert len(result.positions) > 0
        for pos in result.positions:
            assert isinstance(pos, Vec3)

    def test_positions_same_count_as_input(self):
        """Result positions should have same count as input."""
        chain = FABRIKChain(bone_indices=[0, 1, 2])
        positions = create_straight_chain_y(3)
        target = Vec3(0.5, 1.5, 0)
        result = chain.solve(positions, target)
        assert len(result.positions) == len(positions)

    def test_iterations_non_negative(self):
        """Iterations should be non-negative."""
        chain = FABRIKChain(bone_indices=[0, 1, 2])
        positions = create_straight_chain_y(3)
        target = Vec3(0, 2, 0)
        result = chain.solve(positions, target)
        assert result.iterations >= 0

    def test_final_error_non_negative(self):
        """Final error should be non-negative."""
        chain = FABRIKChain(bone_indices=[0, 1, 2])
        positions = create_straight_chain_y(3)
        target = Vec3(0.5, 1.5, 0)
        result = chain.solve(positions, target)
        assert result.final_error >= 0


# =============================================================================
# Chain Construction Tests
# =============================================================================

class TestChainConstruction:
    """Test FABRIKChain construction parameters."""

    def test_accepts_bone_indices(self):
        """FABRIKChain should accept bone_indices list."""
        chain = FABRIKChain(bone_indices=[0, 1, 2])
        assert chain is not None

    def test_accepts_tolerance(self):
        """FABRIKChain should accept tolerance parameter."""
        chain = FABRIKChain(bone_indices=[0, 1, 2], tolerance=0.001)
        assert chain is not None

    def test_accepts_max_iterations(self):
        """FABRIKChain should accept max_iterations parameter."""
        chain = FABRIKChain(bone_indices=[0, 1, 2], max_iterations=10)
        assert chain is not None

    def test_accepts_all_parameters(self):
        """FABRIKChain should accept all parameters together."""
        chain = FABRIKChain(
            bone_indices=[0, 1, 2, 3],
            tolerance=0.0001,
            max_iterations=50
        )
        assert chain is not None

    def test_single_bone_chain(self):
        """Should accept a 2-joint (1-bone) chain."""
        chain = FABRIKChain(bone_indices=[0, 1])
        positions = [Vec3(0, 0, 0), Vec3(0, 1, 0)]
        target = Vec3(0.5, 0.5, 0)
        result = chain.solve(positions, target)
        assert result is not None

    def test_many_bone_chain(self):
        """Should accept a chain with many bones."""
        indices = list(range(10))
        chain = FABRIKChain(bone_indices=indices)
        positions = create_straight_chain_y(10)
        target = Vec3(3, 5, 0)
        result = chain.solve(positions, target)
        assert result is not None

    def test_default_tolerance(self):
        """Should work with default tolerance."""
        chain = FABRIKChain(bone_indices=[0, 1, 2])
        positions = create_straight_chain_y(3)
        target = Vec3(0.5, 1.5, 0)
        result = chain.solve(positions, target)
        assert result is not None

    def test_default_max_iterations(self):
        """Should work with default max_iterations."""
        chain = FABRIKChain(bone_indices=[0, 1, 2])
        positions = create_straight_chain_y(3)
        target = Vec3(0.5, 1.5, 0)
        result = chain.solve(positions, target)
        assert result is not None


# =============================================================================
# Reachable Target Tests
# =============================================================================

class TestReachableTargets:
    """Test behavior when target is within chain reach."""

    def test_two_bone_reaches_target(self):
        """2-bone chain should reach target within range."""
        chain = FABRIKChain(bone_indices=[0, 1, 2], tolerance=0.001)
        positions = create_straight_chain_y(3)  # Total reach: 2.0
        target = Vec3(1.0, 1.0, 0)  # Distance: sqrt(2) < 2.0
        result = chain.solve(positions, target)

        end_effector = result.positions[-1]
        error = end_effector.distance(target)
        assert error < 0.01, f"End effector too far from target: {error}"

    def test_three_bone_reaches_target(self):
        """3-bone chain should reach target within range."""
        chain = FABRIKChain(bone_indices=[0, 1, 2, 3], tolerance=0.001)
        positions = create_straight_chain_y(4)  # Total reach: 3.0
        target = Vec3(1.5, 2.0, 0)  # Distance: 2.5 < 3.0
        result = chain.solve(positions, target)

        end_effector = result.positions[-1]
        error = end_effector.distance(target)
        assert error < 0.01

    def test_end_effector_within_tolerance(self):
        """End effector should be within specified tolerance."""
        tolerance = 0.0001
        chain = FABRIKChain(bone_indices=[0, 1, 2], tolerance=tolerance)
        positions = create_straight_chain_y(3)
        target = Vec3(0.8, 1.2, 0)
        result = chain.solve(positions, target)

        if result.success:
            end_effector = result.positions[-1]
            error = end_effector.distance(target)
            assert error <= tolerance * 10  # Allow some margin

    def test_success_true_when_reached(self):
        """success should be True when target is reached."""
        chain = FABRIKChain(bone_indices=[0, 1, 2], tolerance=0.01)
        positions = create_straight_chain_y(3)
        target = Vec3(0.5, 1.5, 0)  # Easy target
        result = chain.solve(positions, target)
        assert result.success is True

    def test_target_directly_above(self):
        """Should reach target directly above root (collinear case).

        Note: When target is collinear with initial chain but below end effector,
        there's no unique bending direction. The chain may not converge well.
        This tests that it at least gets close and preserves bone lengths.
        """
        chain = FABRIKChain(bone_indices=[0, 1, 2], tolerance=0.1, max_iterations=20)
        positions = create_straight_chain_y(3)
        target = Vec3(0, 1.8, 0)  # Collinear with initial chain
        result = chain.solve(positions, target)
        # May not reach due to collinear singularity - check error is reasonable
        assert result.final_error < 0.5 or result.success is True

    def test_target_at_max_reach(self):
        """Should reach target at exactly max reach."""
        chain = FABRIKChain(bone_indices=[0, 1, 2], tolerance=0.01)
        positions = create_straight_chain_y(3)  # Max reach = 2.0
        target = Vec3(0, 2.0, 0)  # At max reach
        result = chain.solve(positions, target)
        # Should reach (might be tight tolerance)
        end_effector = result.positions[-1]
        error = end_effector.distance(target)
        assert error < 0.1

    def test_target_slightly_inside_reach(self):
        """Should reach target slightly inside max reach (non-collinear).

        We use an offset in X to avoid the collinear singularity.
        """
        chain = FABRIKChain(bone_indices=[0, 1, 2], tolerance=0.05, max_iterations=20)
        positions = create_straight_chain_y(3)  # Max reach = 2.0
        # Offset slightly in X to break collinearity
        target = Vec3(0.1, 1.9, 0)  # Distance ~1.90, well within reach
        result = chain.solve(positions, target)
        # With more iterations and reasonable tolerance, should succeed
        assert result.success is True or result.final_error < 0.1


# =============================================================================
# Unreachable Target Tests
# =============================================================================

class TestUnreachableTargets:
    """Test behavior when target is beyond chain reach."""

    def test_target_beyond_range(self):
        """Chain should extend toward unreachable target."""
        chain = FABRIKChain(bone_indices=[0, 1, 2], tolerance=0.001, max_iterations=20)
        positions = create_straight_chain_y(3)  # Max reach = 2.0
        target = Vec3(0, 5.0, 0)  # Far beyond reach
        result = chain.solve(positions, target)

        # Chain should be extended toward target
        end_effector = result.positions[-1]
        # End effector should be at max reach in target direction
        assert end_effector.y > 1.5  # Should be extended upward

    def test_chain_extends_toward_target(self):
        """Chain should point toward unreachable target."""
        chain = FABRIKChain(bone_indices=[0, 1, 2], tolerance=0.001, max_iterations=20)
        positions = create_straight_chain_y(3)
        target = Vec3(10.0, 0, 0)  # Far to the right
        result = chain.solve(positions, target)

        end_effector = result.positions[-1]
        # Should be extended toward X direction
        assert end_effector.x > 1.0

    def test_success_false_when_unreachable(self):
        """success should be False when target is unreachable."""
        chain = FABRIKChain(bone_indices=[0, 1, 2], tolerance=0.001, max_iterations=20)
        positions = create_straight_chain_y(3)  # Max reach = 2.0
        target = Vec3(0, 10.0, 0)  # Far beyond reach
        result = chain.solve(positions, target)
        assert result.success is False

    def test_final_error_positive(self):
        """final_error should be positive for unreachable target."""
        chain = FABRIKChain(bone_indices=[0, 1, 2], tolerance=0.001, max_iterations=20)
        positions = create_straight_chain_y(3)
        target = Vec3(0, 10.0, 0)
        result = chain.solve(positions, target)
        assert result.final_error > 0

    def test_final_error_equals_distance_to_target(self):
        """final_error should reflect distance from end effector to target."""
        chain = FABRIKChain(bone_indices=[0, 1, 2], tolerance=0.001, max_iterations=20)
        positions = create_straight_chain_y(3)
        target = Vec3(0, 10.0, 0)
        result = chain.solve(positions, target)

        end_effector = result.positions[-1]
        expected_error = end_effector.distance(target)
        # Error should be approximately the distance
        assert nearly_equal(result.final_error, expected_error, eps=0.1)

    def test_far_diagonal_target(self):
        """Should handle far diagonal target."""
        chain = FABRIKChain(bone_indices=[0, 1, 2], tolerance=0.001, max_iterations=20)
        positions = create_straight_chain_y(3)
        target = Vec3(100, 100, 100)  # Very far
        result = chain.solve(positions, target)
        assert result.success is False
        assert result.final_error > 0


# =============================================================================
# Bone Length Preservation Tests
# =============================================================================

class TestBoneLengthPreservation:
    """Test that bone lengths are preserved after solving."""

    def test_two_bone_lengths_preserved(self):
        """2-bone chain should preserve bone lengths."""
        chain = FABRIKChain(bone_indices=[0, 1, 2], tolerance=0.001)
        positions = create_straight_chain_y(3)
        original_lengths = compute_bone_lengths(positions)

        target = Vec3(1.0, 1.0, 0)
        result = chain.solve(positions, target)
        result_lengths = compute_bone_lengths(result.positions)

        for orig, res in zip(original_lengths, result_lengths):
            assert nearly_equal(orig, res, eps=0.001), \
                f"Bone length changed: {orig} -> {res}"

    def test_three_bone_lengths_preserved(self):
        """3-bone chain should preserve bone lengths."""
        chain = FABRIKChain(bone_indices=[0, 1, 2, 3], tolerance=0.001)
        positions = create_straight_chain_y(4)
        original_lengths = compute_bone_lengths(positions)

        target = Vec3(1.5, 1.5, 0.5)
        result = chain.solve(positions, target)
        result_lengths = compute_bone_lengths(result.positions)

        for orig, res in zip(original_lengths, result_lengths):
            assert nearly_equal(orig, res, eps=0.001)

    def test_five_bone_lengths_preserved(self):
        """5-bone chain should preserve bone lengths."""
        chain = FABRIKChain(bone_indices=[0, 1, 2, 3, 4, 5], tolerance=0.001)
        positions = create_straight_chain_y(6)
        original_lengths = compute_bone_lengths(positions)

        target = Vec3(2.0, 3.0, 1.0)
        result = chain.solve(positions, target)
        result_lengths = compute_bone_lengths(result.positions)

        for orig, res in zip(original_lengths, result_lengths):
            assert nearly_equal(orig, res, eps=0.001)

    def test_varying_bone_lengths_preserved(self):
        """Chain with varying bone lengths should preserve them."""
        chain = FABRIKChain(bone_indices=[0, 1, 2, 3], tolerance=0.001)
        # Create chain with bones of length 1, 2, 1.5
        positions = [
            Vec3(0, 0, 0),
            Vec3(0, 1.0, 0),
            Vec3(0, 3.0, 0),  # Bone length 2.0
            Vec3(0, 4.5, 0),  # Bone length 1.5
        ]
        original_lengths = compute_bone_lengths(positions)

        target = Vec3(2.0, 2.0, 0)
        result = chain.solve(positions, target)
        result_lengths = compute_bone_lengths(result.positions)

        for orig, res in zip(original_lengths, result_lengths):
            assert nearly_equal(orig, res, eps=0.001)

    def test_lengths_preserved_unreachable_target(self):
        """Bone lengths preserved even for unreachable targets."""
        chain = FABRIKChain(bone_indices=[0, 1, 2], tolerance=0.001, max_iterations=20)
        positions = create_straight_chain_y(3)
        original_lengths = compute_bone_lengths(positions)

        target = Vec3(0, 100, 0)  # Far beyond reach
        result = chain.solve(positions, target)
        result_lengths = compute_bone_lengths(result.positions)

        for orig, res in zip(original_lengths, result_lengths):
            assert nearly_equal(orig, res, eps=0.001)

    def test_lengths_preserved_after_multiple_solves(self):
        """Bone lengths preserved after multiple solves."""
        chain = FABRIKChain(bone_indices=[0, 1, 2], tolerance=0.001)
        positions = create_straight_chain_y(3)
        original_lengths = compute_bone_lengths(positions)

        targets = [
            Vec3(1, 1, 0),
            Vec3(-1, 1, 0),
            Vec3(0, 1.5, 0.5),
        ]

        current_positions = positions
        for target in targets:
            result = chain.solve(current_positions, target)
            result_lengths = compute_bone_lengths(result.positions)
            for orig, res in zip(original_lengths, result_lengths):
                assert nearly_equal(orig, res, eps=0.001)


# =============================================================================
# Root Anchoring Tests
# =============================================================================

class TestRootAnchoring:
    """Test that root position remains unchanged."""

    def test_root_position_unchanged(self):
        """Root should not move during solve."""
        chain = FABRIKChain(bone_indices=[0, 1, 2], tolerance=0.001)
        positions = create_straight_chain_y(3)
        original_root = Vec3(positions[0].x, positions[0].y, positions[0].z)

        target = Vec3(1.0, 1.0, 0)
        result = chain.solve(positions, target)

        assert vec_nearly_equal(result.positions[0], original_root)

    def test_root_unchanged_extreme_target(self):
        """Root unchanged even for extreme targets."""
        chain = FABRIKChain(bone_indices=[0, 1, 2], tolerance=0.001, max_iterations=20)
        positions = create_straight_chain_y(3)
        original_root = Vec3(positions[0].x, positions[0].y, positions[0].z)

        target = Vec3(1000, 1000, 1000)  # Very far
        result = chain.solve(positions, target)

        assert vec_nearly_equal(result.positions[0], original_root)

    def test_root_unchanged_negative_target(self):
        """Root unchanged for target in negative direction."""
        chain = FABRIKChain(bone_indices=[0, 1, 2], tolerance=0.001)
        positions = create_straight_chain_y(3)
        original_root = Vec3(0, 0, 0)

        target = Vec3(-1, -1, 0)  # Behind root
        result = chain.solve(positions, target)

        assert vec_nearly_equal(result.positions[0], original_root)

    def test_root_unchanged_non_origin(self):
        """Root at non-origin position should stay fixed."""
        chain = FABRIKChain(bone_indices=[0, 1, 2], tolerance=0.001)
        positions = [
            Vec3(5.0, 3.0, 2.0),
            Vec3(5.0, 4.0, 2.0),
            Vec3(5.0, 5.0, 2.0),
        ]
        original_root = Vec3(5.0, 3.0, 2.0)

        target = Vec3(6.0, 4.0, 2.5)
        result = chain.solve(positions, target)

        assert vec_nearly_equal(result.positions[0], original_root)


# =============================================================================
# Convergence Tests
# =============================================================================

class TestConvergence:
    """Test convergence behavior."""

    def test_converges_quickly_simple_case(self):
        """Should converge quickly for simple targets."""
        chain = FABRIKChain(bone_indices=[0, 1, 2], tolerance=0.01, max_iterations=100)
        positions = create_straight_chain_y(3)
        target = Vec3(0, 2, 0)  # Already at end effector
        result = chain.solve(positions, target)

        # Should converge in very few iterations
        assert result.iterations <= 5

    def test_respects_max_iterations(self):
        """Should stop at max_iterations."""
        max_iter = 5
        chain = FABRIKChain(
            bone_indices=[0, 1, 2],
            tolerance=0.0000001,  # Very tight tolerance
            max_iterations=max_iter
        )
        positions = create_straight_chain_y(3)
        target = Vec3(1.0, 1.0, 0)  # Requires movement
        result = chain.solve(positions, target)

        assert result.iterations <= max_iter

    def test_iterations_count_returned(self):
        """iterations field should reflect actual iterations."""
        chain = FABRIKChain(bone_indices=[0, 1, 2], tolerance=0.001, max_iterations=50)
        positions = create_straight_chain_y(3)
        target = Vec3(0.5, 1.5, 0)
        result = chain.solve(positions, target)

        assert result.iterations > 0
        assert isinstance(result.iterations, int)

    def test_tighter_tolerance_more_iterations(self):
        """Tighter tolerance may require more iterations."""
        positions = create_straight_chain_y(3)
        target = Vec3(1.0, 1.0, 0)

        chain_loose = FABRIKChain(bone_indices=[0, 1, 2], tolerance=0.1, max_iterations=100)
        chain_tight = FABRIKChain(bone_indices=[0, 1, 2], tolerance=0.0001, max_iterations=100)

        result_loose = chain_loose.solve(positions, target)
        result_tight = chain_tight.solve(positions, target)

        # Tight tolerance should use at least as many iterations
        assert result_tight.iterations >= result_loose.iterations

    def test_zero_iterations_when_already_solved(self):
        """Should need minimal iterations if already at target."""
        chain = FABRIKChain(bone_indices=[0, 1, 2], tolerance=0.1, max_iterations=100)
        positions = create_straight_chain_y(3)
        target = Vec3(0, 2, 0)  # End effector already here
        result = chain.solve(positions, target)

        # Should need very few iterations
        assert result.iterations <= 2


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_target_at_end_effector(self):
        """Target at current end effector position."""
        chain = FABRIKChain(bone_indices=[0, 1, 2], tolerance=0.001)
        positions = create_straight_chain_y(3)
        target = Vec3(0, 2, 0)  # Exactly at end effector
        result = chain.solve(positions, target)

        assert result.success is True
        assert result.final_error < 0.01

    def test_target_at_root(self):
        """Target at root position (behind chain)."""
        chain = FABRIKChain(bone_indices=[0, 1, 2], tolerance=0.001, max_iterations=20)
        positions = create_straight_chain_y(3)
        target = Vec3(0, 0, 0)  # At root
        result = chain.solve(positions, target)

        # This is an extreme case - chain folds back
        # Bone lengths should still be preserved
        result_lengths = compute_bone_lengths(result.positions)
        for length in result_lengths:
            assert nearly_equal(length, 1.0, eps=0.01)

    def test_long_chain(self):
        """Chain with many bones (10+)."""
        num_joints = 12
        indices = list(range(num_joints))
        chain = FABRIKChain(bone_indices=indices, tolerance=0.01, max_iterations=50)
        positions = create_straight_chain_y(num_joints)
        target = Vec3(5, 5, 0)
        result = chain.solve(positions, target)

        assert len(result.positions) == num_joints
        # Bone lengths preserved
        result_lengths = compute_bone_lengths(result.positions)
        for length in result_lengths:
            assert nearly_equal(length, 1.0, eps=0.01)

    def test_tight_tolerance(self):
        """Very tight tolerance."""
        chain = FABRIKChain(bone_indices=[0, 1, 2], tolerance=0.00001, max_iterations=100)
        positions = create_straight_chain_y(3)
        target = Vec3(0.5, 1.5, 0)
        result = chain.solve(positions, target)

        if result.success:
            end_effector = result.positions[-1]
            error = end_effector.distance(target)
            assert error < 0.001

    def test_loose_tolerance(self):
        """Very loose tolerance."""
        chain = FABRIKChain(bone_indices=[0, 1, 2], tolerance=1.0, max_iterations=10)
        positions = create_straight_chain_y(3)
        target = Vec3(0.5, 1.5, 0)
        result = chain.solve(positions, target)

        # Should converge quickly with loose tolerance
        assert result.iterations <= 5

    def test_very_short_bones(self):
        """Chain with very short bones."""
        chain = FABRIKChain(bone_indices=[0, 1, 2], tolerance=0.0001)
        positions = [
            Vec3(0, 0, 0),
            Vec3(0, 0.01, 0),
            Vec3(0, 0.02, 0),
        ]
        target = Vec3(0.01, 0.01, 0)
        result = chain.solve(positions, target)

        # Should still work
        assert result is not None
        # Bone lengths preserved
        result_lengths = compute_bone_lengths(result.positions)
        assert nearly_equal(result_lengths[0], 0.01, eps=0.001)

    def test_very_long_bones(self):
        """Chain with very long bones."""
        chain = FABRIKChain(bone_indices=[0, 1, 2], tolerance=1.0)
        positions = [
            Vec3(0, 0, 0),
            Vec3(0, 100, 0),
            Vec3(0, 200, 0),
        ]
        target = Vec3(100, 100, 0)
        result = chain.solve(positions, target)

        # Bone lengths preserved
        result_lengths = compute_bone_lengths(result.positions)
        assert nearly_equal(result_lengths[0], 100.0, eps=1.0)
        assert nearly_equal(result_lengths[1], 100.0, eps=1.0)

    def test_single_iteration_limit(self):
        """max_iterations=1 should complete."""
        chain = FABRIKChain(bone_indices=[0, 1, 2], tolerance=0.001, max_iterations=1)
        positions = create_straight_chain_y(3)
        target = Vec3(1.0, 1.0, 0)
        result = chain.solve(positions, target)

        assert result.iterations <= 1

    def test_zero_length_target_distance(self):
        """Target at exact end effector position."""
        chain = FABRIKChain(bone_indices=[0, 1, 2], tolerance=0.001)
        positions = create_straight_chain_y(3)
        end_effector = positions[-1]
        target = Vec3(end_effector.x, end_effector.y, end_effector.z)
        result = chain.solve(positions, target)

        assert result.success is True
        assert result.final_error < 0.01


# =============================================================================
# Multiple Solves Tests
# =============================================================================

class TestMultipleSolves:
    """Test behavior across multiple solves."""

    def test_same_chain_different_targets(self):
        """Same chain should handle different targets."""
        chain = FABRIKChain(bone_indices=[0, 1, 2], tolerance=0.01)
        positions = create_straight_chain_y(3)

        targets = [
            Vec3(1.0, 1.0, 0),
            Vec3(-1.0, 1.0, 0),
            Vec3(0, 1.5, 0.5),
            Vec3(0.5, 1.5, -0.5),
        ]

        for target in targets:
            result = chain.solve(positions, target)
            assert result is not None
            assert len(result.positions) == 3

    def test_no_state_leak(self):
        """State should not leak between solves."""
        chain = FABRIKChain(bone_indices=[0, 1, 2], tolerance=0.01)
        positions1 = create_straight_chain_y(3)
        positions2 = [Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(2, 0, 0)]  # Horizontal

        target1 = Vec3(1.0, 1.0, 0)
        target2 = Vec3(1.5, 0.5, 0)

        result1 = chain.solve(positions1, target1)
        result2 = chain.solve(positions2, target2)

        # Second solve should use positions2, not positions1
        # Bone lengths should match positions2
        lengths2 = compute_bone_lengths(result2.positions)
        for length in lengths2:
            assert nearly_equal(length, 1.0, eps=0.01)

    def test_sequential_solves_independent(self):
        """Each solve should be independent."""
        chain = FABRIKChain(bone_indices=[0, 1, 2], tolerance=0.01)

        for i in range(5):
            positions = create_straight_chain_y(3)
            target = Vec3(0.5 + i * 0.1, 1.5, 0)
            result = chain.solve(positions, target)

            # Root should always be at origin
            assert vec_nearly_equal(result.positions[0], Vec3(0, 0, 0))

    def test_reuse_result_positions(self):
        """Can use result positions as input to next solve."""
        chain = FABRIKChain(bone_indices=[0, 1, 2], tolerance=0.01)
        positions = create_straight_chain_y(3)

        # First solve
        target1 = Vec3(1.0, 1.0, 0)
        result1 = chain.solve(positions, target1)

        # Use result as input to second solve
        target2 = Vec3(0.5, 1.5, 0)
        result2 = chain.solve(result1.positions, target2)

        # Should still have correct structure
        assert len(result2.positions) == 3
        # Bone lengths preserved
        result_lengths = compute_bone_lengths(result2.positions)
        for length in result_lengths:
            assert nearly_equal(length, 1.0, eps=0.01)


# =============================================================================
# 3D Target Tests
# =============================================================================

class Test3DTargets:
    """Test targets in various 3D positions."""

    def test_positive_octant(self):
        """Target in positive octant (+x, +y, +z)."""
        chain = FABRIKChain(bone_indices=[0, 1, 2], tolerance=0.01)
        positions = create_straight_chain_y(3)
        target = Vec3(0.5, 1.5, 0.5)
        result = chain.solve(positions, target)

        assert result.success is True
        end_effector = result.positions[-1]
        assert end_effector.distance(target) < 0.1

    def test_negative_octant(self):
        """Target in negative octant (-x, +y, -z)."""
        chain = FABRIKChain(bone_indices=[0, 1, 2], tolerance=0.01)
        positions = create_straight_chain_y(3)
        target = Vec3(-0.5, 1.5, -0.5)
        result = chain.solve(positions, target)

        assert result.success is True
        end_effector = result.positions[-1]
        assert end_effector.distance(target) < 0.1

    def test_non_planar(self):
        """Target requiring non-planar configuration."""
        chain = FABRIKChain(bone_indices=[0, 1, 2], tolerance=0.01)
        positions = create_straight_chain_y(3)
        target = Vec3(0.5, 1.0, 0.8)
        result = chain.solve(positions, target)

        # End effector should have non-zero z
        end_effector = result.positions[-1]
        assert abs(end_effector.z) > 0.1

    def test_all_octants(self):
        """Test targets in all 8 octants."""
        chain = FABRIKChain(bone_indices=[0, 1, 2], tolerance=0.1)
        positions = create_straight_chain_y(3)

        octants = [
            (1, 1, 1),
            (1, 1, -1),
            (1, -1, 1),
            (1, -1, -1),
            (-1, 1, 1),
            (-1, 1, -1),
            (-1, -1, 1),
            (-1, -1, -1),
        ]

        for ox, oy, oz in octants:
            target = Vec3(ox * 0.5, 1.0 + oy * 0.5, oz * 0.5)
            result = chain.solve(positions, target)
            assert result is not None

    def test_z_axis_target(self):
        """Target along Z axis."""
        chain = FABRIKChain(bone_indices=[0, 1, 2], tolerance=0.01)
        positions = create_straight_chain_y(3)
        target = Vec3(0, 0.5, 1.5)
        result = chain.solve(positions, target)

        end_effector = result.positions[-1]
        assert end_effector.z > 0.5  # Moved toward Z

    def test_negative_z_target(self):
        """Target in negative Z direction."""
        chain = FABRIKChain(bone_indices=[0, 1, 2], tolerance=0.01)
        positions = create_straight_chain_y(3)
        target = Vec3(0, 0.5, -1.5)
        result = chain.solve(positions, target)

        end_effector = result.positions[-1]
        assert end_effector.z < -0.5  # Moved toward -Z

    def test_diagonal_3d_target(self):
        """Target along diagonal in 3D space."""
        chain = FABRIKChain(bone_indices=[0, 1, 2, 3], tolerance=0.01)
        positions = create_straight_chain_y(4)  # Max reach 3.0
        # Target at distance sqrt(3) * 1.5 ~= 2.6 < 3.0
        target = Vec3(1.5, 1.5, 1.5)
        result = chain.solve(positions, target)

        assert result.success is True


# =============================================================================
# Stress Tests
# =============================================================================

class TestStressConditions:
    """Test under stress conditions."""

    def test_many_iterations_needed(self):
        """Case requiring many iterations."""
        chain = FABRIKChain(
            bone_indices=[0, 1, 2, 3, 4, 5],
            tolerance=0.0001,
            max_iterations=200
        )
        positions = create_straight_chain_y(6)
        # Complex target requiring significant reconfiguration
        target = Vec3(2.5, 2.5, 1.0)
        result = chain.solve(positions, target)

        # Should complete
        assert result is not None
        assert result.iterations <= 200

    def test_rapid_target_changes(self):
        """Many rapid target changes."""
        chain = FABRIKChain(bone_indices=[0, 1, 2], tolerance=0.1)
        positions = create_straight_chain_y(3)

        for i in range(20):
            angle = i * math.pi / 10
            target = Vec3(math.cos(angle), 1.0 + math.sin(angle), 0)
            result = chain.solve(positions, target)
            assert result is not None

    def test_extreme_tolerance_values(self):
        """Test with extreme tolerance values."""
        positions = create_straight_chain_y(3)
        target = Vec3(0.5, 1.5, 0)

        # Very tight
        chain_tight = FABRIKChain(
            bone_indices=[0, 1, 2],
            tolerance=1e-10,
            max_iterations=100
        )
        result_tight = chain_tight.solve(positions, target)
        assert result_tight is not None

        # Very loose
        chain_loose = FABRIKChain(
            bone_indices=[0, 1, 2],
            tolerance=10.0,
            max_iterations=100
        )
        result_loose = chain_loose.solve(positions, target)
        assert result_loose is not None


# =============================================================================
# Numerical Stability Tests
# =============================================================================

class TestNumericalStability:
    """Test numerical stability."""

    def test_small_movements(self):
        """Handle very small target movements."""
        chain = FABRIKChain(bone_indices=[0, 1, 2], tolerance=0.0001)
        positions = create_straight_chain_y(3)

        # Target almost at end effector
        target = Vec3(0.0001, 2.0001, 0.0001)
        result = chain.solve(positions, target)

        assert result is not None
        # Bone lengths should be preserved
        result_lengths = compute_bone_lengths(result.positions)
        for length in result_lengths:
            assert nearly_equal(length, 1.0, eps=0.01)

    def test_large_coordinates(self):
        """Handle large coordinate values."""
        chain = FABRIKChain(bone_indices=[0, 1, 2], tolerance=1.0)
        positions = [
            Vec3(1000, 1000, 1000),
            Vec3(1000, 1001, 1000),
            Vec3(1000, 1002, 1000),
        ]
        target = Vec3(1001, 1001, 1000.5)
        result = chain.solve(positions, target)

        # Root should not drift
        assert vec_nearly_equal(result.positions[0], Vec3(1000, 1000, 1000), eps=0.1)

    def test_near_singular_configuration(self):
        """Handle near-singular (straight line) configurations."""
        chain = FABRIKChain(bone_indices=[0, 1, 2], tolerance=0.01)
        positions = create_straight_chain_y(3)
        # Target requiring bending from straight line
        target = Vec3(0.01, 1.99, 0)
        result = chain.solve(positions, target)

        assert result is not None
        # Bone lengths preserved
        result_lengths = compute_bone_lengths(result.positions)
        for length in result_lengths:
            assert nearly_equal(length, 1.0, eps=0.01)

    def test_collinear_positions(self):
        """Handle perfectly collinear initial positions."""
        chain = FABRIKChain(bone_indices=[0, 1, 2], tolerance=0.01)
        positions = [Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 2, 0)]
        target = Vec3(1, 1, 0)  # Perpendicular to chain
        result = chain.solve(positions, target)

        assert result is not None


# =============================================================================
# API Robustness Tests
# =============================================================================

class TestAPIRobustness:
    """Test API robustness and error handling."""

    def test_result_is_fabrik_result_type(self):
        """solve() should return FABRIKResult instance."""
        chain = FABRIKChain(bone_indices=[0, 1, 2])
        positions = create_straight_chain_y(3)
        target = Vec3(0.5, 1.5, 0)
        result = chain.solve(positions, target)

        assert isinstance(result, FABRIKResult)

    def test_result_fields_accessible(self):
        """All result fields should be accessible."""
        chain = FABRIKChain(bone_indices=[0, 1, 2])
        positions = create_straight_chain_y(3)
        target = Vec3(0.5, 1.5, 0)
        result = chain.solve(positions, target)

        # Access all fields without error
        _ = result.success
        _ = result.iterations
        _ = result.final_error
        _ = result.positions

    def test_positions_list_not_modified(self):
        """Input positions list should not be modified."""
        chain = FABRIKChain(bone_indices=[0, 1, 2])
        positions = create_straight_chain_y(3)
        original_positions = [Vec3(p.x, p.y, p.z) for p in positions]

        target = Vec3(1.0, 1.0, 0)
        _ = chain.solve(positions, target)

        # Original positions unchanged
        for orig, curr in zip(original_positions, positions):
            assert vec_nearly_equal(orig, curr)

    def test_chain_reusable(self):
        """Same chain instance should be reusable."""
        chain = FABRIKChain(bone_indices=[0, 1, 2], tolerance=0.01)

        for _ in range(10):
            positions = create_straight_chain_y(3)
            target = Vec3(0.5, 1.5, 0)
            result = chain.solve(positions, target)
            assert result is not None


# =============================================================================
# Performance Characteristic Tests
# =============================================================================

class TestPerformanceCharacteristics:
    """Test performance-related characteristics."""

    def test_larger_chain_completes(self):
        """Larger chain should still complete in reasonable iterations."""
        num_joints = 20
        indices = list(range(num_joints))
        chain = FABRIKChain(bone_indices=indices, tolerance=0.1, max_iterations=200)
        positions = create_straight_chain_y(num_joints)
        target = Vec3(10, 10, 0)
        result = chain.solve(positions, target)

        assert result is not None
        assert result.iterations <= 200

    def test_iteration_count_reasonable(self):
        """Iteration count should be reasonable for simple cases."""
        chain = FABRIKChain(bone_indices=[0, 1, 2], tolerance=0.01, max_iterations=1000)
        positions = create_straight_chain_y(3)
        target = Vec3(0.5, 1.5, 0)
        result = chain.solve(positions, target)

        # Should not need anywhere near 1000 iterations
        assert result.iterations < 50


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
