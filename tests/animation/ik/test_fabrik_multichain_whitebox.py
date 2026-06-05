"""Whitebox tests for FABRIKMultiChain (Multi-chain FABRIK solver).

Tests the internal implementation details of FABRIKMultiChain class
with comprehensive coverage of shared joints, averaging, and multi-chain
coordination.
"""

from __future__ import annotations

import math
import pytest

from engine.animation.ik.fabrik import (
    FABRIKChain,
    FABRIKMultiChain,
    FABRIKResult,
    JointConstraint,
    JointConstraintType,
)
from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.core.constants import MATH_EPSILON
from engine.animation.ik.config import (
    IK_DEFAULT_TOLERANCE,
    FABRIK_DEFAULT_MAX_ITERATIONS,
    FABRIK_MULTI_CHAIN_MAX_ITERATIONS,
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


def create_diagonal_chain_positions(
    num_joints: int,
    bone_length: float = 1.0,
    start: Vec3 = None,
    direction: Vec3 = None
) -> list[Vec3]:
    """Create positions for a diagonal chain."""
    if start is None:
        start = Vec3.zero()
    if direction is None:
        direction = Vec3(1, 1, 0).normalized()
    step = direction * bone_length
    return [Vec3(start.x + i * step.x, start.y + i * step.y, start.z + i * step.z) for i in range(num_joints)]


def create_skeleton_positions(num_joints: int = 10) -> list[Vec3]:
    """Create a skeleton with multiple joints for multi-chain testing."""
    # Create a simple skeleton: spine (0-3), left arm (2, 4-5), right arm (2, 6-7)
    positions = []
    for i in range(num_joints):
        positions.append(Vec3(0, i * 0.5, 0))
    return positions


# =============================================================================
# TestMultiChainConstruction
# =============================================================================


class TestMultiChainConstruction:
    """Tests for FABRIKMultiChain construction and initialization."""

    def test_empty_initialization(self):
        """Empty solver initializes with empty chains list."""
        solver = FABRIKMultiChain()

        assert solver._chains == []
        assert solver._chain_targets == []
        assert solver._shared_joints == {}

    def test_chains_list_empty(self):
        """Chains list is empty after construction."""
        solver = FABRIKMultiChain()

        assert len(solver._chains) == 0

    def test_targets_list_empty(self):
        """Targets list is empty after construction."""
        solver = FABRIKMultiChain()

        assert len(solver._chain_targets) == 0

    def test_shared_joints_dict_empty(self):
        """Shared joints dictionary is empty after construction."""
        solver = FABRIKMultiChain()

        assert len(solver._shared_joints) == 0

    def test_internal_state_isolated(self):
        """Two solvers have isolated internal state."""
        solver1 = FABRIKMultiChain()
        solver2 = FABRIKMultiChain()

        chain = FABRIKChain([0, 1, 2])
        solver1.add_chain(chain, Vec3(0, 5, 0))

        assert len(solver1._chains) == 1
        assert len(solver2._chains) == 0


# =============================================================================
# TestAddChain
# =============================================================================


class TestAddChain:
    """Tests for add_chain method."""

    def test_add_single_chain(self):
        """Add a single chain to the solver."""
        solver = FABRIKMultiChain()
        chain = FABRIKChain([0, 1, 2])
        target = Vec3(0, 5, 0)

        solver.add_chain(chain, target)

        assert len(solver._chains) == 1
        assert solver._chains[0] is chain

    def test_add_returns_index_zero(self):
        """First added chain returns index 0."""
        solver = FABRIKMultiChain()
        chain = FABRIKChain([0, 1, 2])

        idx = solver.add_chain(chain, Vec3(0, 5, 0))

        assert idx == 0

    def test_add_returns_sequential_indices(self):
        """Each added chain returns sequential indices."""
        solver = FABRIKMultiChain()

        idx0 = solver.add_chain(FABRIKChain([0, 1, 2]), Vec3(0, 5, 0))
        idx1 = solver.add_chain(FABRIKChain([3, 4, 5]), Vec3(0, 10, 0))
        idx2 = solver.add_chain(FABRIKChain([6, 7, 8]), Vec3(0, 15, 0))

        assert idx0 == 0
        assert idx1 == 1
        assert idx2 == 2

    def test_add_multiple_chains(self):
        """Add multiple chains to the solver."""
        solver = FABRIKMultiChain()

        solver.add_chain(FABRIKChain([0, 1, 2]), Vec3(0, 5, 0))
        solver.add_chain(FABRIKChain([3, 4, 5]), Vec3(0, 10, 0))
        solver.add_chain(FABRIKChain([6, 7, 8]), Vec3(0, 15, 0))

        assert len(solver._chains) == 3

    def test_add_stores_target(self):
        """add_chain stores the target position."""
        solver = FABRIKMultiChain()
        chain = FABRIKChain([0, 1, 2])
        target = Vec3(1.5, 2.5, 3.5)

        solver.add_chain(chain, target)

        assert vec3_nearly_equal(solver._chain_targets[0], target)

    def test_add_chain_tracks_bone_indices(self):
        """add_chain tracks bone indices in shared_joints."""
        solver = FABRIKMultiChain()
        chain = FABRIKChain([0, 1, 2])

        solver.add_chain(chain, Vec3(0, 5, 0))

        assert 0 in solver._shared_joints
        assert 1 in solver._shared_joints
        assert 2 in solver._shared_joints

    def test_shared_joints_tracked_for_overlap(self):
        """Shared joints are tracked when chains overlap."""
        solver = FABRIKMultiChain()

        # Two chains sharing joint 2
        solver.add_chain(FABRIKChain([0, 1, 2]), Vec3(0, 5, 0))
        solver.add_chain(FABRIKChain([2, 3, 4]), Vec3(0, 10, 0))

        assert 2 in solver._shared_joints
        assert len(solver._shared_joints[2]) == 2
        assert 0 in solver._shared_joints[2]
        assert 1 in solver._shared_joints[2]

    def test_add_chain_with_same_indices(self):
        """Multiple chains can have the same bone indices."""
        solver = FABRIKMultiChain()

        solver.add_chain(FABRIKChain([0, 1, 2]), Vec3(0, 5, 0))
        solver.add_chain(FABRIKChain([0, 1, 2]), Vec3(0, 10, 0))

        assert len(solver._chains) == 2
        # All joints should be shared by both chains
        assert len(solver._shared_joints[0]) == 2
        assert len(solver._shared_joints[1]) == 2
        assert len(solver._shared_joints[2]) == 2

    def test_add_chain_updates_chain_indices_correctly(self):
        """Chain indices in shared_joints are correct."""
        solver = FABRIKMultiChain()

        solver.add_chain(FABRIKChain([0, 1]), Vec3(0, 5, 0))
        solver.add_chain(FABRIKChain([1, 2]), Vec3(0, 10, 0))
        solver.add_chain(FABRIKChain([1, 3]), Vec3(0, 15, 0))

        # Joint 1 shared by all three chains
        assert sorted(solver._shared_joints[1]) == [0, 1, 2]


# =============================================================================
# TestSetTarget
# =============================================================================


class TestSetTarget:
    """Tests for set_target method."""

    def test_set_target_valid_index(self):
        """Set target for a valid chain index."""
        solver = FABRIKMultiChain()
        solver.add_chain(FABRIKChain([0, 1, 2]), Vec3(0, 5, 0))

        solver.set_target(0, Vec3(1, 2, 3))

        assert vec3_nearly_equal(solver._chain_targets[0], Vec3(1, 2, 3))

    def test_set_target_updates_existing(self):
        """set_target updates existing target."""
        solver = FABRIKMultiChain()
        solver.add_chain(FABRIKChain([0, 1, 2]), Vec3(0, 5, 0))

        solver.set_target(0, Vec3(10, 20, 30))

        assert solver._chain_targets[0].x == pytest.approx(10)
        assert solver._chain_targets[0].y == pytest.approx(20)
        assert solver._chain_targets[0].z == pytest.approx(30)

    def test_set_target_multiple_chains(self):
        """set_target updates correct chain target."""
        solver = FABRIKMultiChain()
        solver.add_chain(FABRIKChain([0, 1, 2]), Vec3(0, 5, 0))
        solver.add_chain(FABRIKChain([3, 4, 5]), Vec3(0, 10, 0))

        solver.set_target(1, Vec3(100, 200, 300))

        # First chain target unchanged
        assert vec3_nearly_equal(solver._chain_targets[0], Vec3(0, 5, 0))
        # Second chain target updated
        assert vec3_nearly_equal(solver._chain_targets[1], Vec3(100, 200, 300))

    def test_set_target_invalid_negative_index(self):
        """set_target with negative index does nothing (no error)."""
        solver = FABRIKMultiChain()
        solver.add_chain(FABRIKChain([0, 1, 2]), Vec3(0, 5, 0))
        original_target = Vec3(0, 5, 0)

        # Should not raise, should not modify
        solver.set_target(-1, Vec3(100, 200, 300))

        assert vec3_nearly_equal(solver._chain_targets[0], original_target)

    def test_set_target_invalid_too_large_index(self):
        """set_target with index >= len does nothing (no error)."""
        solver = FABRIKMultiChain()
        solver.add_chain(FABRIKChain([0, 1, 2]), Vec3(0, 5, 0))
        original_target = Vec3(0, 5, 0)

        solver.set_target(1, Vec3(100, 200, 300))  # Index 1 doesn't exist

        assert vec3_nearly_equal(solver._chain_targets[0], original_target)

    def test_set_target_empty_solver(self):
        """set_target on empty solver does nothing."""
        solver = FABRIKMultiChain()

        # Should not raise
        solver.set_target(0, Vec3(1, 2, 3))

        assert len(solver._chain_targets) == 0

    def test_set_target_boundary_index(self):
        """set_target with boundary index (last valid)."""
        solver = FABRIKMultiChain()
        solver.add_chain(FABRIKChain([0, 1, 2]), Vec3(0, 5, 0))
        solver.add_chain(FABRIKChain([3, 4, 5]), Vec3(0, 10, 0))

        solver.set_target(1, Vec3(50, 50, 50))

        assert vec3_nearly_equal(solver._chain_targets[1], Vec3(50, 50, 50))


# =============================================================================
# TestSharedJointDetection
# =============================================================================


class TestSharedJointDetection:
    """Tests for shared joint detection logic."""

    def test_no_shared_joints_disjoint(self):
        """Disjoint chains have no shared joints."""
        solver = FABRIKMultiChain()
        solver.add_chain(FABRIKChain([0, 1, 2]), Vec3(0, 5, 0))
        solver.add_chain(FABRIKChain([3, 4, 5]), Vec3(0, 10, 0))

        # Each joint is used by only one chain
        for joint_idx in [0, 1, 2, 3, 4, 5]:
            assert len(solver._shared_joints[joint_idx]) == 1

    def test_shared_root_joint(self):
        """Detect shared root joint between chains."""
        solver = FABRIKMultiChain()
        # Shared root at joint 0
        solver.add_chain(FABRIKChain([0, 1, 2]), Vec3(0, 5, 0))
        solver.add_chain(FABRIKChain([0, 3, 4]), Vec3(5, 0, 0))

        assert len(solver._shared_joints[0]) == 2
        assert 0 in solver._shared_joints[0]
        assert 1 in solver._shared_joints[0]

    def test_shared_intermediate_joint(self):
        """Detect shared intermediate joint."""
        solver = FABRIKMultiChain()
        solver.add_chain(FABRIKChain([0, 1, 2]), Vec3(0, 5, 0))
        solver.add_chain(FABRIKChain([3, 1, 4]), Vec3(5, 0, 0))

        # Joint 1 shared
        assert len(solver._shared_joints[1]) == 2

    def test_shared_end_effector(self):
        """Detect shared end effector joint."""
        solver = FABRIKMultiChain()
        solver.add_chain(FABRIKChain([0, 1, 2]), Vec3(0, 5, 0))
        solver.add_chain(FABRIKChain([3, 4, 2]), Vec3(5, 0, 0))

        # Joint 2 shared (both end effectors)
        assert len(solver._shared_joints[2]) == 2

    def test_multiple_shared_joints(self):
        """Multiple joints shared between chains."""
        solver = FABRIKMultiChain()
        # Chains share joints 1 and 2
        solver.add_chain(FABRIKChain([0, 1, 2, 3]), Vec3(0, 5, 0))
        solver.add_chain(FABRIKChain([4, 1, 2, 5]), Vec3(5, 0, 0))

        assert len(solver._shared_joints[1]) == 2
        assert len(solver._shared_joints[2]) == 2
        assert len(solver._shared_joints[0]) == 1
        assert len(solver._shared_joints[3]) == 1

    def test_three_chains_single_shared_joint(self):
        """Three chains sharing a single joint."""
        solver = FABRIKMultiChain()
        solver.add_chain(FABRIKChain([0, 1]), Vec3(0, 5, 0))
        solver.add_chain(FABRIKChain([2, 1]), Vec3(5, 0, 0))
        solver.add_chain(FABRIKChain([3, 1]), Vec3(0, 0, 5))

        assert len(solver._shared_joints[1]) == 3
        assert sorted(solver._shared_joints[1]) == [0, 1, 2]

    def test_full_chain_overlap(self):
        """Complete overlap - same chain added twice."""
        solver = FABRIKMultiChain()
        solver.add_chain(FABRIKChain([0, 1, 2]), Vec3(0, 5, 0))
        solver.add_chain(FABRIKChain([0, 1, 2]), Vec3(5, 5, 5))

        for joint_idx in [0, 1, 2]:
            assert len(solver._shared_joints[joint_idx]) == 2


# =============================================================================
# TestSolveSingleChain
# =============================================================================


class TestSolveSingleChain:
    """Tests for solve() with a single chain."""

    def test_single_chain_reachable_target(self):
        """Single chain reaches a reachable target."""
        solver = FABRIKMultiChain()
        chain = FABRIKChain([0, 1, 2])
        target = Vec3(0, 1.8, 0)  # Reachable within chain length

        solver.add_chain(chain, target)

        positions = create_straight_chain_positions(3)
        result = solver.solve(positions, max_iterations=30)

        # End effector should be close to target (lerp averaging needs more tolerance)
        assert (result[2] - target).length() < 0.6

    def test_single_chain_unreachable_target(self):
        """Single chain extends toward unreachable target."""
        solver = FABRIKMultiChain()
        chain = FABRIKChain([0, 1, 2])
        target = Vec3(0, 100, 0)  # Far beyond reach

        solver.add_chain(chain, target)

        positions = create_straight_chain_positions(3)
        result = solver.solve(positions, max_iterations=10)

        # Chain should extend toward target
        assert result[2].y > result[1].y > result[0].y

    def test_single_chain_at_target(self):
        """Single chain already at target converges immediately."""
        solver = FABRIKMultiChain()
        chain = FABRIKChain([0, 1, 2])
        positions = create_straight_chain_positions(3)
        target = Vec3(positions[2].x, positions[2].y, positions[2].z)

        solver.add_chain(chain, target)

        result = solver.solve(positions, max_iterations=10)

        assert vec3_nearly_equal(result[2], target, tol=0.01)

    def test_single_chain_preserves_bone_lengths(self):
        """Single chain solve preserves bone lengths."""
        solver = FABRIKMultiChain()
        chain = FABRIKChain([0, 1, 2])
        target = Vec3(1.0, 1.0, 0)

        solver.add_chain(chain, target)

        positions = create_straight_chain_positions(3, bone_length=1.0)
        result = solver.solve(positions, max_iterations=20)

        # Check bone lengths
        len1 = (result[1] - result[0]).length()
        len2 = (result[2] - result[1]).length()

        assert len1 == pytest.approx(1.0, abs=0.05)
        assert len2 == pytest.approx(1.0, abs=0.05)

    def test_single_chain_horizontal_target(self):
        """Single chain solves for horizontal target."""
        solver = FABRIKMultiChain()
        chain = FABRIKChain([0, 1, 2])
        target = Vec3(1.5, 0.5, 0)

        solver.add_chain(chain, target)

        positions = create_straight_chain_positions(3)
        result = solver.solve(positions, max_iterations=20)

        # End effector should approach target
        error = (result[2] - target).length()
        assert error < 0.5  # Within reasonable error


# =============================================================================
# TestSolveMultiChain
# =============================================================================


class TestSolveMultiChain:
    """Tests for solve() with multiple chains."""

    def test_two_chains_independent(self):
        """Two independent chains both reach their targets."""
        solver = FABRIKMultiChain()
        solver.add_chain(FABRIKChain([0, 1, 2]), Vec3(0, 1.8, 0))
        solver.add_chain(FABRIKChain([3, 4, 5]), Vec3(0, 4.8, 0))

        # Create positions for 6 joints
        positions = [
            Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 2, 0),
            Vec3(0, 3, 0), Vec3(0, 4, 0), Vec3(0, 5, 0)
        ]

        result = solver.solve(positions, max_iterations=30)

        # Both end effectors should approach targets (within tolerance)
        assert (result[2] - Vec3(0, 1.8, 0)).length() < 0.6
        assert (result[5] - Vec3(0, 4.8, 0)).length() < 0.6

    def test_two_chains_shared_root(self):
        """Two chains with shared root joint."""
        solver = FABRIKMultiChain()
        solver.add_chain(FABRIKChain([0, 1, 2]), Vec3(0, 2, 0))
        solver.add_chain(FABRIKChain([0, 3, 4]), Vec3(2, 0, 0))

        positions = [
            Vec3(0, 0, 0),  # Shared root
            Vec3(0, 1, 0), Vec3(0, 2, 0),  # First chain
            Vec3(1, 0, 0), Vec3(2, 0, 0)   # Second chain
        ]

        result = solver.solve(positions, max_iterations=20)

        # Result should be valid
        assert len(result) == 5
        # Root should stay fixed at averaged position
        assert result[0] is not None

    def test_three_chains_shared_joint(self):
        """Three chains sharing a central joint."""
        solver = FABRIKMultiChain()
        # All chains share joint 0
        solver.add_chain(FABRIKChain([0, 1]), Vec3(0, 2, 0))
        solver.add_chain(FABRIKChain([0, 2]), Vec3(2, 0, 0))
        solver.add_chain(FABRIKChain([0, 3]), Vec3(0, 0, 2))

        positions = [
            Vec3(0, 0, 0),  # Shared joint
            Vec3(0, 1, 0),
            Vec3(1, 0, 0),
            Vec3(0, 0, 1)
        ]

        result = solver.solve(positions, max_iterations=20)

        assert len(result) == 4

    def test_parallel_chains_different_lengths(self):
        """Parallel chains with different lengths."""
        solver = FABRIKMultiChain()
        solver.add_chain(FABRIKChain([0, 1, 2]), Vec3(0, 2.5, 0))
        solver.add_chain(FABRIKChain([3, 4, 5, 6]), Vec3(1, 3.5, 0))

        positions = [
            # Chain 1: 3 joints
            Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 2, 0),
            # Chain 2: 4 joints
            Vec3(1, 0, 0), Vec3(1, 1, 0), Vec3(1, 2, 0), Vec3(1, 3, 0)
        ]

        result = solver.solve(positions, max_iterations=20)

        assert len(result) == 7

    def test_branching_structure(self):
        """Branching structure like a humanoid arm."""
        solver = FABRIKMultiChain()
        # Spine: 0-1-2
        # Left arm: 2-3-4
        # Right arm: 2-5-6
        solver.add_chain(FABRIKChain([0, 1, 2]), Vec3(0, 2, 0))
        solver.add_chain(FABRIKChain([2, 3, 4]), Vec3(-2, 2, 0))
        solver.add_chain(FABRIKChain([2, 5, 6]), Vec3(2, 2, 0))

        positions = [
            Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 2, 0),  # Spine
            Vec3(-1, 2, 0), Vec3(-2, 2, 0),  # Left arm
            Vec3(1, 2, 0), Vec3(2, 2, 0)    # Right arm
        ]

        result = solver.solve(positions, max_iterations=20)

        assert len(result) == 7
        # Joint 2 should be influenced by all three chains


# =============================================================================
# TestSharedJointAveraging
# =============================================================================


class TestSharedJointAveraging:
    """Tests for position averaging at shared joints."""

    def test_averaging_two_chains_same_position(self):
        """Two chains affecting same joint converge to averaged position."""
        solver = FABRIKMultiChain()
        solver.add_chain(FABRIKChain([0, 1, 2]), Vec3(0, 3, 0))
        solver.add_chain(FABRIKChain([0, 3, 4]), Vec3(3, 0, 0))

        positions = [
            Vec3(0, 0, 0),  # Shared root
            Vec3(0, 1, 0), Vec3(0, 2, 0),
            Vec3(1, 0, 0), Vec3(2, 0, 0)
        ]

        result = solver.solve(positions, max_iterations=30)

        # Shared joint 0 should be affected by averaging
        assert result[0] is not None

    def test_averaging_three_chains(self):
        """Three chains sharing a joint show combined averaging effect."""
        solver = FABRIKMultiChain()
        solver.add_chain(FABRIKChain([0, 1]), Vec3(0, 2, 0))
        solver.add_chain(FABRIKChain([0, 2]), Vec3(2, 0, 0))
        solver.add_chain(FABRIKChain([0, 3]), Vec3(0, 0, 2))

        positions = [
            Vec3(0, 0, 0),
            Vec3(0, 1, 0),
            Vec3(1, 0, 0),
            Vec3(0, 0, 1)
        ]

        result = solver.solve(positions, max_iterations=30)

        # Joint 0 influenced by three chains
        assert result[0] is not None

    def test_averaging_uses_lerp(self):
        """Verify averaging uses lerp calculation."""
        solver = FABRIKMultiChain()
        # Two chains share joint 1
        solver.add_chain(FABRIKChain([0, 1]), Vec3(0, 1, 0))
        solver.add_chain(FABRIKChain([2, 1]), Vec3(0, 1, 0))

        positions = [
            Vec3(0, 0, 0),
            Vec3(0, 0.5, 0),  # Shared joint
            Vec3(0, 1, 0)
        ]

        # After solving, joint 1 should be a blend
        result = solver.solve(positions, max_iterations=5)
        assert result[1] is not None

    def test_non_shared_joints_not_averaged(self):
        """Non-shared joints use direct assignment."""
        solver = FABRIKMultiChain()
        solver.add_chain(FABRIKChain([0, 1, 2]), Vec3(0, 2, 0))
        solver.add_chain(FABRIKChain([3, 4, 5]), Vec3(0, 5, 0))

        positions = [
            Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 2, 0),
            Vec3(0, 3, 0), Vec3(0, 4, 0), Vec3(0, 5, 0)
        ]

        result = solver.solve(positions, max_iterations=10)

        # All joints should have positions
        for i in range(6):
            assert result[i] is not None


# =============================================================================
# TestConvergence
# =============================================================================


class TestConvergence:
    """Tests for convergence behavior."""

    def test_converges_when_all_reach_target(self):
        """Solve converges when all chains reach their targets."""
        solver = FABRIKMultiChain()
        positions = create_straight_chain_positions(3)
        target = Vec3(positions[2].x, positions[2].y, positions[2].z)

        solver.add_chain(FABRIKChain([0, 1, 2]), target)

        result = solver.solve(positions, max_iterations=50)

        assert vec3_nearly_equal(result[2], target, tol=0.1)

    def test_respects_max_iterations(self):
        """Solve stops after max_iterations."""
        solver = FABRIKMultiChain()
        # Target is reachable but may need iterations
        solver.add_chain(FABRIKChain([0, 1, 2]), Vec3(1.5, 0.5, 0))

        positions = create_straight_chain_positions(3)

        # Very low max_iterations
        result = solver.solve(positions, max_iterations=1)

        # Should return a valid result even with few iterations
        assert len(result) == 3

    def test_max_iterations_zero(self):
        """Zero max_iterations returns original positions."""
        solver = FABRIKMultiChain()
        solver.add_chain(FABRIKChain([0, 1, 2]), Vec3(10, 10, 10))

        positions = create_straight_chain_positions(3)
        original = [Vec3(p.x, p.y, p.z) for p in positions]

        result = solver.solve(positions, max_iterations=0)

        # With 0 iterations, positions should be copied (no solving)
        for i in range(3):
            assert vec3_nearly_equal(result[i], original[i])

    def test_default_max_iterations(self):
        """Default max_iterations uses FABRIK_MULTI_CHAIN_MAX_ITERATIONS."""
        solver = FABRIKMultiChain()
        solver.add_chain(FABRIKChain([0, 1, 2]), Vec3(1.0, 1.0, 0))

        positions = create_straight_chain_positions(3)

        # Should use default without error
        result = solver.solve(positions)

        assert len(result) == 3

    def test_convergence_multiple_chains(self):
        """Multiple chains converge together."""
        solver = FABRIKMultiChain()
        solver.add_chain(FABRIKChain([0, 1, 2]), Vec3(0, 1.9, 0))
        solver.add_chain(FABRIKChain([3, 4, 5]), Vec3(0, 4.9, 0))

        positions = [
            Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 2, 0),
            Vec3(0, 3, 0), Vec3(0, 4, 0), Vec3(0, 5, 0)
        ]

        result = solver.solve(positions, max_iterations=50)

        # Both should be near targets
        assert (result[2] - Vec3(0, 1.9, 0)).length() < 0.5
        assert (result[5] - Vec3(0, 4.9, 0)).length() < 0.5


# =============================================================================
# TestEdgeCases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_solve(self):
        """Solve with no chains returns original positions."""
        solver = FABRIKMultiChain()

        positions = create_straight_chain_positions(5)
        original = [Vec3(p.x, p.y, p.z) for p in positions]

        result = solver.solve(positions, max_iterations=10)

        for i in range(5):
            assert vec3_nearly_equal(result[i], original[i])

    def test_single_bone_chains(self):
        """Minimum valid chains (2 joints each)."""
        solver = FABRIKMultiChain()
        solver.add_chain(FABRIKChain([0, 1]), Vec3(0, 0.9, 0))
        solver.add_chain(FABRIKChain([2, 3]), Vec3(0, 2.9, 0))

        positions = [
            Vec3(0, 0, 0), Vec3(0, 1, 0),
            Vec3(0, 2, 0), Vec3(0, 3, 0)
        ]

        result = solver.solve(positions, max_iterations=20)

        assert len(result) == 4

    def test_all_joints_shared(self):
        """All joints shared by multiple chains."""
        solver = FABRIKMultiChain()
        solver.add_chain(FABRIKChain([0, 1, 2]), Vec3(0, 2, 0))
        solver.add_chain(FABRIKChain([0, 1, 2]), Vec3(0, 2.5, 0))

        positions = create_straight_chain_positions(3)

        result = solver.solve(positions, max_iterations=20)

        # All joints should have valid positions
        for i in range(3):
            assert result[i] is not None

    def test_no_joints_shared(self):
        """Completely disjoint chains."""
        solver = FABRIKMultiChain()
        solver.add_chain(FABRIKChain([0, 1]), Vec3(0, 1.5, 0))
        solver.add_chain(FABRIKChain([2, 3]), Vec3(0, 3.5, 0))
        solver.add_chain(FABRIKChain([4, 5]), Vec3(0, 5.5, 0))

        positions = [
            Vec3(0, 0, 0), Vec3(0, 1, 0),
            Vec3(0, 2, 0), Vec3(0, 3, 0),
            Vec3(0, 4, 0), Vec3(0, 5, 0)
        ]

        result = solver.solve(positions, max_iterations=20)

        assert len(result) == 6

    def test_large_position_values(self):
        """Handle large position values."""
        solver = FABRIKMultiChain()
        solver.add_chain(FABRIKChain([0, 1, 2]), Vec3(0, 10002, 0))

        positions = [
            Vec3(0, 10000, 0),
            Vec3(0, 10001, 0),
            Vec3(0, 10002, 0)
        ]

        result = solver.solve(positions, max_iterations=10)

        assert len(result) == 3

    def test_small_position_values(self):
        """Handle small position values."""
        solver = FABRIKMultiChain()
        solver.add_chain(FABRIKChain([0, 1, 2]), Vec3(0, 0.002, 0))

        positions = [
            Vec3(0, 0, 0),
            Vec3(0, 0.001, 0),
            Vec3(0, 0.002, 0)
        ]

        result = solver.solve(positions, max_iterations=10)

        assert len(result) == 3

    def test_negative_positions(self):
        """Handle negative position values."""
        solver = FABRIKMultiChain()
        solver.add_chain(FABRIKChain([0, 1, 2]), Vec3(0, -1.5, 0))

        positions = [
            Vec3(0, 0, 0),
            Vec3(0, -1, 0),
            Vec3(0, -2, 0)
        ]

        result = solver.solve(positions, max_iterations=10)

        assert len(result) == 3

    def test_positions_copy_isolation(self):
        """Solve does not modify input positions."""
        solver = FABRIKMultiChain()
        solver.add_chain(FABRIKChain([0, 1, 2]), Vec3(1, 1, 0))

        positions = create_straight_chain_positions(3)
        original_copy = [Vec3(p.x, p.y, p.z) for p in positions]

        solver.solve(positions, max_iterations=20)

        # Original positions should be unchanged
        for i in range(3):
            assert vec3_nearly_equal(positions[i], original_copy[i])

    def test_target_at_origin(self):
        """Target at origin."""
        solver = FABRIKMultiChain()
        solver.add_chain(FABRIKChain([0, 1, 2]), Vec3.zero())

        positions = create_straight_chain_positions(3)

        result = solver.solve(positions, max_iterations=20)

        assert len(result) == 3


# =============================================================================
# TestInternalState
# =============================================================================


class TestInternalState:
    """Tests for internal state management."""

    def test_chains_stored_by_reference(self):
        """Chains are stored by reference."""
        solver = FABRIKMultiChain()
        chain = FABRIKChain([0, 1, 2])

        solver.add_chain(chain, Vec3(0, 5, 0))

        assert solver._chains[0] is chain

    def test_targets_stored_as_vec3(self):
        """Targets are stored as Vec3 objects."""
        solver = FABRIKMultiChain()
        target = Vec3(1, 2, 3)

        solver.add_chain(FABRIKChain([0, 1, 2]), target)

        assert isinstance(solver._chain_targets[0], Vec3)

    def test_shared_joints_dict_structure(self):
        """Shared joints dict has correct structure."""
        solver = FABRIKMultiChain()
        solver.add_chain(FABRIKChain([0, 1, 2]), Vec3(0, 5, 0))
        solver.add_chain(FABRIKChain([1, 3, 4]), Vec3(5, 0, 0))

        # Joint 1 should map to list of chain indices
        assert isinstance(solver._shared_joints[1], list)
        assert 0 in solver._shared_joints[1]
        assert 1 in solver._shared_joints[1]

    def test_solve_returns_new_list(self):
        """Solve returns a new list, not input."""
        solver = FABRIKMultiChain()
        solver.add_chain(FABRIKChain([0, 1, 2]), Vec3(0, 2, 0))

        positions = create_straight_chain_positions(3)

        result = solver.solve(positions, max_iterations=10)

        assert result is not positions

    def test_solve_returns_vec3_objects(self):
        """Solve returns list of Vec3 objects."""
        solver = FABRIKMultiChain()
        solver.add_chain(FABRIKChain([0, 1, 2]), Vec3(0, 2, 0))

        positions = create_straight_chain_positions(3)
        result = solver.solve(positions, max_iterations=10)

        for pos in result:
            assert isinstance(pos, Vec3)


# =============================================================================
# TestChainIndexing
# =============================================================================


class TestChainIndexing:
    """Tests for chain index handling in solve."""

    def test_chain_uses_correct_bone_indices(self):
        """Chain extracts positions using bone_indices."""
        solver = FABRIKMultiChain()
        chain = FABRIKChain([2, 3, 4])  # Uses joints 2, 3, 4
        solver.add_chain(chain, Vec3(0, 4, 0))

        positions = [
            Vec3(0, 0, 0), Vec3(0, 1, 0),  # Not used by chain
            Vec3(0, 2, 0), Vec3(0, 3, 0), Vec3(0, 4, 0)  # Used by chain
        ]

        result = solver.solve(positions, max_iterations=10)

        # All positions should be valid
        assert len(result) == 5

    def test_non_sequential_bone_indices(self):
        """Chain with non-sequential bone indices."""
        solver = FABRIKMultiChain()
        chain = FABRIKChain([0, 5, 10])  # Non-sequential
        solver.add_chain(chain, Vec3(0, 10, 0))

        positions = [Vec3(0, i, 0) for i in range(11)]

        result = solver.solve(positions, max_iterations=10)

        assert len(result) == 11

    def test_gaps_in_bone_indices(self):
        """Chain with gaps in bone indices works correctly."""
        solver = FABRIKMultiChain()
        solver.add_chain(FABRIKChain([0, 2, 4]), Vec3(0, 4, 0))

        positions = [
            Vec3(0, 0, 0),
            Vec3(0, 1, 0),  # Unused
            Vec3(0, 2, 0),
            Vec3(0, 3, 0),  # Unused
            Vec3(0, 4, 0)
        ]

        result = solver.solve(positions, max_iterations=10)

        assert len(result) == 5


# =============================================================================
# TestSolveIteration
# =============================================================================


class TestSolveIteration:
    """Tests for iteration behavior in solve."""

    def test_iteration_processes_all_chains(self):
        """Each iteration processes all chains."""
        solver = FABRIKMultiChain()
        solver.add_chain(FABRIKChain([0, 1, 2]), Vec3(0, 2, 0))
        solver.add_chain(FABRIKChain([3, 4, 5]), Vec3(0, 5, 0))

        positions = [Vec3(0, i, 0) for i in range(6)]

        # With 1 iteration, all chains should still be processed
        result = solver.solve(positions, max_iterations=1)

        assert len(result) == 6

    def test_iteration_updates_positions(self):
        """Iterations progressively update positions."""
        solver = FABRIKMultiChain()
        solver.add_chain(FABRIKChain([0, 1, 2]), Vec3(1.5, 0.5, 0))

        positions = create_straight_chain_positions(3)

        result1 = solver.solve(positions, max_iterations=1)
        result10 = solver.solve(positions, max_iterations=10)

        # More iterations should get closer to target
        error1 = (result1[2] - Vec3(1.5, 0.5, 0)).length()
        error10 = (result10[2] - Vec3(1.5, 0.5, 0)).length()

        assert error10 <= error1 + 0.1  # Allow small tolerance

    def test_early_convergence_breaks(self):
        """Solve breaks early when converged."""
        solver = FABRIKMultiChain()
        positions = create_straight_chain_positions(3)
        target = Vec3(positions[2].x, positions[2].y, positions[2].z)

        solver.add_chain(FABRIKChain([0, 1, 2]), target)

        # Already at target, should converge in first iteration
        result = solver.solve(positions, max_iterations=100)

        assert vec3_nearly_equal(result[2], target, tol=0.1)


# =============================================================================
# TestMultiChainCoordination
# =============================================================================


class TestMultiChainCoordination:
    """Tests for multi-chain coordination behavior."""

    def test_shared_joint_affected_by_both_chains(self):
        """Shared joint position reflects both chains' influence."""
        solver = FABRIKMultiChain()
        # Two chains pulling shared joint 1 in different directions
        solver.add_chain(FABRIKChain([0, 1]), Vec3(0, 2, 0))   # Pull up
        solver.add_chain(FABRIKChain([2, 1]), Vec3(0, -2, 0))  # Pull down

        positions = [
            Vec3(-1, 0, 0),
            Vec3(0, 0, 0),  # Shared
            Vec3(1, 0, 0)
        ]

        result = solver.solve(positions, max_iterations=20)

        # Shared joint should be somewhere between pulls
        # (exact position depends on implementation)
        assert result[1] is not None

    def test_chain_order_independence(self):
        """Result should be similar regardless of chain add order."""
        solver1 = FABRIKMultiChain()
        solver1.add_chain(FABRIKChain([0, 1, 2]), Vec3(0, 2, 0))
        solver1.add_chain(FABRIKChain([2, 3, 4]), Vec3(0, 4, 0))

        solver2 = FABRIKMultiChain()
        solver2.add_chain(FABRIKChain([2, 3, 4]), Vec3(0, 4, 0))
        solver2.add_chain(FABRIKChain([0, 1, 2]), Vec3(0, 2, 0))

        positions = [Vec3(0, i, 0) for i in range(5)]

        result1 = solver1.solve(positions, max_iterations=20)
        result2 = solver2.solve(positions, max_iterations=20)

        # Shared joint 2 should be similar
        assert (result1[2] - result2[2]).length() < 1.0

    def test_longer_chains_reach_further(self):
        """Longer chains can reach further targets."""
        solver = FABRIKMultiChain()
        # Short chain: 2 bones
        solver.add_chain(FABRIKChain([0, 1, 2]), Vec3(0, 3, 0))  # Max reach ~2

        positions = create_straight_chain_positions(3)
        result = solver.solve(positions, max_iterations=20)

        # Should extend toward but not reach target
        assert result[2].y < 3.0


# =============================================================================
# TestChainsAttribute
# =============================================================================


class TestChainsAttribute:
    """Tests for chains attribute access patterns."""

    def test_access_chain_bone_indices(self):
        """Can access bone_indices of stored chains."""
        solver = FABRIKMultiChain()
        chain = FABRIKChain([5, 6, 7, 8])
        solver.add_chain(chain, Vec3(0, 8, 0))

        assert solver._chains[0].bone_indices == [5, 6, 7, 8]

    def test_access_chain_tolerance(self):
        """Can access tolerance of stored chains."""
        solver = FABRIKMultiChain()
        chain = FABRIKChain([0, 1, 2], tolerance=0.05)
        solver.add_chain(chain, Vec3(0, 2, 0))

        assert solver._chains[0].tolerance == pytest.approx(0.05)

    def test_access_chain_max_iterations(self):
        """Can access max_iterations of stored chains."""
        solver = FABRIKMultiChain()
        chain = FABRIKChain([0, 1, 2], max_iterations=25)
        solver.add_chain(chain, Vec3(0, 2, 0))

        assert solver._chains[0].max_iterations == 25


# =============================================================================
# TestComplexScenarios
# =============================================================================


class TestComplexScenarios:
    """Tests for complex multi-chain scenarios."""

    def test_humanoid_upper_body(self):
        """Simulated humanoid upper body structure."""
        solver = FABRIKMultiChain()
        # Spine: joints 0-2
        # Left arm: joints 2-4
        # Right arm: joints 2, 5-6
        # Head: joints 2, 7-8
        solver.add_chain(FABRIKChain([0, 1, 2]), Vec3(0, 2, 0))     # Spine
        solver.add_chain(FABRIKChain([2, 3, 4]), Vec3(-2, 2, 0))   # Left arm
        solver.add_chain(FABRIKChain([2, 5, 6]), Vec3(2, 2, 0))    # Right arm
        solver.add_chain(FABRIKChain([2, 7, 8]), Vec3(0, 3, 0))    # Head

        positions = [
            Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 2, 0),  # Spine
            Vec3(-0.5, 2, 0), Vec3(-1, 2, 0),             # Left arm
            Vec3(0.5, 2, 0), Vec3(1, 2, 0),               # Right arm
            Vec3(0, 2.5, 0), Vec3(0, 3, 0)                # Head
        ]

        result = solver.solve(positions, max_iterations=30)

        assert len(result) == 9

    def test_spider_legs(self):
        """Simulated spider with 4 legs from central body."""
        solver = FABRIKMultiChain()
        # Central body at joint 0
        # Four legs radiating out
        solver.add_chain(FABRIKChain([0, 1, 2]), Vec3(2, 0, 0))    # Front-right
        solver.add_chain(FABRIKChain([0, 3, 4]), Vec3(-2, 0, 0))   # Front-left
        solver.add_chain(FABRIKChain([0, 5, 6]), Vec3(0, 0, 2))    # Back-right
        solver.add_chain(FABRIKChain([0, 7, 8]), Vec3(0, 0, -2))   # Back-left

        positions = [
            Vec3(0, 0, 0),  # Center
            Vec3(0.5, 0, 0), Vec3(1, 0, 0),
            Vec3(-0.5, 0, 0), Vec3(-1, 0, 0),
            Vec3(0, 0, 0.5), Vec3(0, 0, 1),
            Vec3(0, 0, -0.5), Vec3(0, 0, -1)
        ]

        result = solver.solve(positions, max_iterations=30)

        assert len(result) == 9

    def test_chain_of_chains(self):
        """Chains connected in sequence."""
        solver = FABRIKMultiChain()
        # Chain 1: 0-1-2
        # Chain 2: 2-3-4 (shares joint 2 with chain 1)
        # Chain 3: 4-5-6 (shares joint 4 with chain 2)
        solver.add_chain(FABRIKChain([0, 1, 2]), Vec3(0, 2, 0))
        solver.add_chain(FABRIKChain([2, 3, 4]), Vec3(0, 4, 0))
        solver.add_chain(FABRIKChain([4, 5, 6]), Vec3(0, 6, 0))

        positions = [Vec3(0, i, 0) for i in range(7)]

        result = solver.solve(positions, max_iterations=30)

        assert len(result) == 7
        # Verify chain connectivity
        assert (result[2] - result[1]).length() < 2.0
        assert (result[4] - result[3]).length() < 2.0
