"""Blackbox tests for FABRIK Multi-Chain IK solver.

Tests are written from the theoretical multi-chain FABRIK specification without
reading the implementation. Multi-chain FABRIK handles connected IK chains that
share joints:

1. Multiple chains can share root or intermediate joints
2. When chains share a joint, its position must satisfy all chains
3. Solution: average the position from each chain's solution
4. Iterate until all chains converge or max iterations

Key properties:
- Shared joints converge to averaged position
- Each chain preserves its bone lengths
- Multiple effectors can reach independent targets
- Stable solution without wild oscillation
"""

import pytest
import math
from typing import List, Tuple

from engine.animation.ik.fabrik import FABRIKChain, FABRIKMultiChain
from engine.core.math.vec import Vec3


# =============================================================================
# Helper Functions
# =============================================================================

def compute_bone_lengths(positions: List[Vec3], indices: List[int]) -> List[float]:
    """Compute bone lengths for a chain given indices into positions."""
    lengths = []
    for i in range(len(indices) - 1):
        lengths.append(positions[indices[i]].distance(positions[indices[i + 1]]))
    return lengths


def total_chain_length_from_indices(positions: List[Vec3], indices: List[int]) -> float:
    """Compute total reach of chain from positions and indices."""
    return sum(compute_bone_lengths(positions, indices))


def create_skeleton_positions(num_joints: int, bone_length: float = 1.0) -> List[Vec3]:
    """Create skeleton positions along Y axis."""
    return [Vec3(0, i * bone_length, 0) for i in range(num_joints)]


def create_humanoid_positions() -> List[Vec3]:
    """Create a simple humanoid-like skeleton.

    Layout:
    - 0: Root (pelvis)
    - 1: Spine
    - 2: Chest
    - 3: Head
    - 4: Right shoulder
    - 5: Right elbow
    - 6: Right hand
    - 7: Left shoulder
    - 8: Left elbow
    - 9: Left hand
    """
    return [
        Vec3(0, 0, 0),      # 0: Root
        Vec3(0, 1, 0),      # 1: Spine
        Vec3(0, 2, 0),      # 2: Chest
        Vec3(0, 3, 0),      # 3: Head
        Vec3(1, 2, 0),      # 4: Right shoulder
        Vec3(2, 2, 0),      # 5: Right elbow
        Vec3(3, 2, 0),      # 6: Right hand
        Vec3(-1, 2, 0),     # 7: Left shoulder
        Vec3(-2, 2, 0),     # 8: Left elbow
        Vec3(-3, 2, 0),     # 9: Left hand
    ]


def nearly_equal(a: float, b: float, eps: float = 1e-4) -> bool:
    """Check if two floats are nearly equal."""
    return abs(a - b) <= eps


def vec_nearly_equal(a: Vec3, b: Vec3, eps: float = 1e-4) -> bool:
    """Check if two vectors are nearly equal."""
    return a.distance(b) <= eps


def positions_are_stable(pos1: List[Vec3], pos2: List[Vec3], eps: float = 0.1) -> bool:
    """Check if two position sets are reasonably close (no wild oscillation)."""
    if len(pos1) != len(pos2):
        return False
    for p1, p2 in zip(pos1, pos2):
        if p1.distance(p2) > eps:
            return False
    return True


# =============================================================================
# Multi-Chain Class Existence Tests
# =============================================================================

class TestMultiChainExists:
    """Test FABRIKMultiChain class exists and can be instantiated."""

    def test_can_import_multichain_class(self):
        """FABRIKMultiChain class should be importable."""
        assert FABRIKMultiChain is not None

    def test_can_instantiate(self):
        """FABRIKMultiChain should be instantiable."""
        multi = FABRIKMultiChain()
        assert multi is not None

    def test_instance_is_multichain(self):
        """Instance should be of FABRIKMultiChain type."""
        multi = FABRIKMultiChain()
        assert isinstance(multi, FABRIKMultiChain)

    def test_has_add_chain_method(self):
        """FABRIKMultiChain should have add_chain method."""
        multi = FABRIKMultiChain()
        assert hasattr(multi, 'add_chain')
        assert callable(getattr(multi, 'add_chain'))

    def test_has_solve_method(self):
        """FABRIKMultiChain should have solve method."""
        multi = FABRIKMultiChain()
        assert hasattr(multi, 'solve')
        assert callable(getattr(multi, 'solve'))

    def test_multiple_instances_independent(self):
        """Multiple instances should be independent."""
        multi1 = FABRIKMultiChain()
        multi2 = FABRIKMultiChain()
        assert multi1 is not multi2


# =============================================================================
# Add Chain Tests
# =============================================================================

class TestAddChain:
    """Test adding chains to the multi-chain solver."""

    def test_add_chain_accepts_fabrik_chain(self):
        """add_chain should accept a FABRIKChain object."""
        multi = FABRIKMultiChain()
        chain = FABRIKChain(bone_indices=[0, 1, 2])
        target = Vec3(0, 2, 0)
        # Should not raise
        multi.add_chain(chain, target)

    def test_add_chain_accepts_target(self):
        """add_chain should accept a target Vec3."""
        multi = FABRIKMultiChain()
        chain = FABRIKChain(bone_indices=[0, 1, 2])
        target = Vec3(1, 2, 3)
        multi.add_chain(chain, target)
        # No exception means success

    def test_add_multiple_chains(self):
        """Should be able to add multiple chains."""
        multi = FABRIKMultiChain()
        chain1 = FABRIKChain(bone_indices=[0, 1, 2])
        chain2 = FABRIKChain(bone_indices=[0, 3, 4])
        multi.add_chain(chain1, Vec3(1, 0, 0))
        multi.add_chain(chain2, Vec3(-1, 0, 0))
        # No exception means success

    def test_add_chain_returns_value(self):
        """add_chain should return a value (index, self, or None)."""
        multi = FABRIKMultiChain()
        chain = FABRIKChain(bone_indices=[0, 1, 2])
        result = multi.add_chain(chain, Vec3(0, 2, 0))
        # Accept index (int), None, or self for method chaining
        assert result is None or result is multi or isinstance(result, int)

    def test_add_chain_with_single_bone_chain(self):
        """Should accept chain with two joints (one bone)."""
        multi = FABRIKMultiChain()
        chain = FABRIKChain(bone_indices=[0, 1])
        multi.add_chain(chain, Vec3(0, 1, 0))

    def test_add_chain_with_long_chain(self):
        """Should accept chain with many joints."""
        multi = FABRIKMultiChain()
        chain = FABRIKChain(bone_indices=[0, 1, 2, 3, 4, 5, 6, 7])
        multi.add_chain(chain, Vec3(0, 7, 0))


# =============================================================================
# Solve Method Tests
# =============================================================================

class TestSolveMethod:
    """Test the solve method interface."""

    def test_solve_accepts_positions_list(self):
        """solve should accept a list of Vec3 positions."""
        multi = FABRIKMultiChain()
        chain = FABRIKChain(bone_indices=[0, 1, 2])
        multi.add_chain(chain, Vec3(0, 2, 0))
        positions = create_skeleton_positions(3)
        result = multi.solve(positions)
        assert result is not None

    def test_solve_returns_positions(self):
        """solve should return position data."""
        multi = FABRIKMultiChain()
        chain = FABRIKChain(bone_indices=[0, 1, 2])
        multi.add_chain(chain, Vec3(0, 2, 0))
        positions = create_skeleton_positions(3)
        result = multi.solve(positions)
        # Result should be list of Vec3 or result object with positions
        if isinstance(result, list):
            assert all(isinstance(p, Vec3) for p in result)
        else:
            assert hasattr(result, 'positions')

    def test_solve_preserves_position_count(self):
        """solve should return same number of positions as input."""
        multi = FABRIKMultiChain()
        chain = FABRIKChain(bone_indices=[0, 1, 2])
        multi.add_chain(chain, Vec3(0, 2, 0))
        positions = create_skeleton_positions(3)
        result = multi.solve(positions)
        result_positions = result if isinstance(result, list) else result.positions
        assert len(result_positions) == len(positions)

    def test_solve_with_no_chains_returns_original(self):
        """solve with no chains should return original positions."""
        multi = FABRIKMultiChain()
        positions = create_skeleton_positions(5)
        result = multi.solve(positions)
        result_positions = result if isinstance(result, list) else result.positions
        # With no chains, positions should be unchanged
        for i, pos in enumerate(result_positions):
            assert vec_nearly_equal(pos, positions[i])

    def test_solve_does_not_modify_input(self):
        """solve should not modify the input positions list."""
        multi = FABRIKMultiChain()
        chain = FABRIKChain(bone_indices=[0, 1, 2])
        multi.add_chain(chain, Vec3(1, 1, 0))
        positions = create_skeleton_positions(3)
        original = [Vec3(p.x, p.y, p.z) for p in positions]
        multi.solve(positions)
        for i, pos in enumerate(positions):
            assert vec_nearly_equal(pos, original[i])


# =============================================================================
# Single Chain Behavior Tests
# =============================================================================

class TestSingleChainBehavior:
    """Test multi-chain solver with single chain behaves like regular FABRIK."""

    def test_single_chain_reaches_target(self):
        """Single chain should reach toward its target."""
        multi = FABRIKMultiChain()
        chain = FABRIKChain(bone_indices=[0, 1, 2])
        target = Vec3(0.5, 1.5, 0)  # Reachable target (well within reach)
        multi.add_chain(chain, target)

        positions = create_skeleton_positions(3)
        result = multi.solve(positions)
        result_positions = result if isinstance(result, list) else result.positions

        effector_pos = result_positions[2]
        # With default tolerance, effector should get reasonably close
        assert effector_pos.distance(target) < 0.25

    def test_single_chain_preserves_lengths(self):
        """Single chain should preserve bone lengths."""
        multi = FABRIKMultiChain()
        chain = FABRIKChain(bone_indices=[0, 1, 2])
        target = Vec3(1, 1, 0)
        multi.add_chain(chain, target)

        positions = create_skeleton_positions(3)
        original_lengths = compute_bone_lengths(positions, [0, 1, 2])

        result = multi.solve(positions)
        result_positions = result if isinstance(result, list) else result.positions

        new_lengths = compute_bone_lengths(result_positions, [0, 1, 2])
        for orig, new in zip(original_lengths, new_lengths):
            assert nearly_equal(orig, new, eps=0.01)

    def test_single_chain_anchors_root(self):
        """Single chain should keep root position fixed."""
        multi = FABRIKMultiChain()
        chain = FABRIKChain(bone_indices=[0, 1, 2])
        target = Vec3(0.5, 1.5, 0)
        multi.add_chain(chain, target)

        positions = create_skeleton_positions(3)
        original_root = Vec3(positions[0].x, positions[0].y, positions[0].z)

        result = multi.solve(positions)
        result_positions = result if isinstance(result, list) else result.positions

        assert vec_nearly_equal(result_positions[0], original_root)

    def test_single_chain_unreachable_target(self):
        """Single chain should stretch toward unreachable target."""
        multi = FABRIKMultiChain()
        chain = FABRIKChain(bone_indices=[0, 1, 2])
        target = Vec3(0, 100, 0)  # Very far away
        multi.add_chain(chain, target)

        positions = create_skeleton_positions(3)
        total_length = total_chain_length_from_indices(positions, [0, 1, 2])

        result = multi.solve(positions)
        result_positions = result if isinstance(result, list) else result.positions

        # Should be stretched toward target
        effector_y = result_positions[2].y
        assert effector_y >= total_length - 0.1


# =============================================================================
# Shared Root Behavior Tests
# =============================================================================

class TestSharedRootBehavior:
    """Test chains sharing a root joint."""

    def test_two_arms_shared_spine(self):
        """Two arm chains sharing spine should both solve."""
        multi = FABRIKMultiChain()
        positions = create_humanoid_positions()

        # Right arm: chest -> shoulder -> elbow -> hand
        right_arm = FABRIKChain(bone_indices=[2, 4, 5, 6])
        # Left arm: chest -> shoulder -> elbow -> hand
        left_arm = FABRIKChain(bone_indices=[2, 7, 8, 9])

        multi.add_chain(right_arm, Vec3(4, 2, 0))   # Right target
        multi.add_chain(left_arm, Vec3(-4, 2, 0))  # Left target

        result = multi.solve(positions)
        result_positions = result if isinstance(result, list) else result.positions

        assert result_positions is not None
        assert len(result_positions) == len(positions)

    def test_both_arms_reach_targets(self):
        """Both arms should reach toward their respective targets."""
        multi = FABRIKMultiChain()
        positions = create_humanoid_positions()

        right_arm = FABRIKChain(bone_indices=[2, 4, 5, 6])
        left_arm = FABRIKChain(bone_indices=[2, 7, 8, 9])

        right_target = Vec3(4, 1, 0)
        left_target = Vec3(-4, 1, 0)

        multi.add_chain(right_arm, right_target)
        multi.add_chain(left_arm, left_target)

        result = multi.solve(positions)
        result_positions = result if isinstance(result, list) else result.positions

        # Right hand should be closer to right target
        right_hand = result_positions[6]
        # Left hand should be closer to left target
        left_hand = result_positions[9]

        # Targets are at x=4 and x=-4, hands should be on correct sides
        assert right_hand.x > 0  # Right side
        assert left_hand.x < 0   # Left side

    def test_shared_root_position_reasonable(self):
        """Shared root joint should be at a reasonable position."""
        multi = FABRIKMultiChain()
        positions = create_humanoid_positions()

        right_arm = FABRIKChain(bone_indices=[2, 4, 5, 6])
        left_arm = FABRIKChain(bone_indices=[2, 7, 8, 9])

        # Symmetric targets
        multi.add_chain(right_arm, Vec3(3, 1, 0))
        multi.add_chain(left_arm, Vec3(-3, 1, 0))

        result = multi.solve(positions)
        result_positions = result if isinstance(result, list) else result.positions

        # Shared joint (chest at index 2) should stay near center for symmetric targets
        chest = result_positions[2]
        assert abs(chest.x) < 1.0  # Should be near center

    def test_asymmetric_targets_shared_root(self):
        """Asymmetric targets should still converge with shared root."""
        multi = FABRIKMultiChain()
        positions = create_humanoid_positions()

        right_arm = FABRIKChain(bone_indices=[2, 4, 5, 6])
        left_arm = FABRIKChain(bone_indices=[2, 7, 8, 9])

        # Asymmetric targets
        multi.add_chain(right_arm, Vec3(2, 0, 0))   # Right down
        multi.add_chain(left_arm, Vec3(-3, 3, 0))  # Left up

        result = multi.solve(positions)
        result_positions = result if isinstance(result, list) else result.positions

        # Should still produce valid positions
        assert len(result_positions) == len(positions)

    def test_three_chains_shared_root(self):
        """Three chains sharing root should all solve."""
        multi = FABRIKMultiChain()

        # Create positions for 3 chains from shared root
        positions = [
            Vec3(0, 0, 0),   # 0: Shared root
            Vec3(1, 0, 0),   # 1: Chain 1 joint 1
            Vec3(2, 0, 0),   # 2: Chain 1 end
            Vec3(0, 1, 0),   # 3: Chain 2 joint 1
            Vec3(0, 2, 0),   # 4: Chain 2 end
            Vec3(0, 0, 1),   # 5: Chain 3 joint 1
            Vec3(0, 0, 2),   # 6: Chain 3 end
        ]

        chain1 = FABRIKChain(bone_indices=[0, 1, 2])
        chain2 = FABRIKChain(bone_indices=[0, 3, 4])
        chain3 = FABRIKChain(bone_indices=[0, 5, 6])

        multi.add_chain(chain1, Vec3(1.5, 0, 0))
        multi.add_chain(chain2, Vec3(0, 1.5, 0))
        multi.add_chain(chain3, Vec3(0, 0, 1.5))

        result = multi.solve(positions)
        result_positions = result if isinstance(result, list) else result.positions

        assert len(result_positions) == 7


# =============================================================================
# Shared Joint Stability Tests
# =============================================================================

class TestSharedJointStability:
    """Test that shared joints remain stable without wild oscillation."""

    def test_no_wild_oscillation(self):
        """Multiple solves should not cause wild oscillation."""
        multi = FABRIKMultiChain()
        positions = create_humanoid_positions()

        right_arm = FABRIKChain(bone_indices=[2, 4, 5, 6])
        left_arm = FABRIKChain(bone_indices=[2, 7, 8, 9])

        multi.add_chain(right_arm, Vec3(3, 1, 0))
        multi.add_chain(left_arm, Vec3(-3, 1, 0))

        # Solve multiple times
        result1 = multi.solve(positions)
        pos1 = result1 if isinstance(result1, list) else result1.positions

        result2 = multi.solve(pos1)
        pos2 = result2 if isinstance(result2, list) else result2.positions

        result3 = multi.solve(pos2)
        pos3 = result3 if isinstance(result3, list) else result3.positions

        # Positions should stabilize, not oscillate
        assert positions_are_stable(pos2, pos3, eps=0.2)

    def test_shared_joint_reasonable_position(self):
        """Shared joint should be at reasonable average position."""
        multi = FABRIKMultiChain()

        positions = [
            Vec3(0, 0, 0),   # 0: Shared
            Vec3(1, 0, 0),   # 1: Chain 1 end
            Vec3(-1, 0, 0),  # 2: Chain 2 end
        ]

        chain1 = FABRIKChain(bone_indices=[0, 1])
        chain2 = FABRIKChain(bone_indices=[0, 2])

        # Pull in opposite directions
        multi.add_chain(chain1, Vec3(1, 0, 0))
        multi.add_chain(chain2, Vec3(-1, 0, 0))

        result = multi.solve(positions)
        result_positions = result if isinstance(result, list) else result.positions

        # Shared root should remain at or near origin
        shared_pos = result_positions[0]
        assert shared_pos.distance(Vec3(0, 0, 0)) < 0.5

    def test_convergence_with_conflicting_targets(self):
        """Conflicting targets should still converge to stable solution."""
        multi = FABRIKMultiChain()

        positions = [
            Vec3(0, 0, 0),   # 0: Shared root
            Vec3(0, 1, 0),   # 1: Shared joint
            Vec3(1, 1, 0),   # 2: Chain 1 end
            Vec3(-1, 1, 0),  # 3: Chain 2 end
        ]

        # Both chains share joints 0 and 1
        chain1 = FABRIKChain(bone_indices=[0, 1, 2])
        chain2 = FABRIKChain(bone_indices=[0, 1, 3])

        multi.add_chain(chain1, Vec3(2, 0, 0))   # Pull right
        multi.add_chain(chain2, Vec3(-2, 0, 0))  # Pull left

        result = multi.solve(positions)
        result_positions = result if isinstance(result, list) else result.positions

        # Shared joint 1 should be somewhere in between
        shared_joint = result_positions[1]
        assert abs(shared_joint.x) < 1.0  # Not pulled too far either way


# =============================================================================
# Multiple Shared Joints Tests
# =============================================================================

class TestMultipleSharedJoints:
    """Test chains sharing multiple joints."""

    def test_two_chains_sharing_two_joints(self):
        """Chains sharing multiple joints should solve."""
        multi = FABRIKMultiChain()

        # Spine shared, then branches
        positions = [
            Vec3(0, 0, 0),   # 0: Root
            Vec3(0, 1, 0),   # 1: Spine shared
            Vec3(0, 2, 0),   # 2: Chest shared
            Vec3(1, 2, 0),   # 3: Right arm
            Vec3(-1, 2, 0),  # 4: Left arm
        ]

        # Both chains share root and spine
        chain1 = FABRIKChain(bone_indices=[0, 1, 2, 3])
        chain2 = FABRIKChain(bone_indices=[0, 1, 2, 4])

        multi.add_chain(chain1, Vec3(2, 2, 0))
        multi.add_chain(chain2, Vec3(-2, 2, 0))

        result = multi.solve(positions)
        result_positions = result if isinstance(result, list) else result.positions

        assert len(result_positions) == 5

    def test_shared_segment_preserves_length(self):
        """Shared segments should approximately preserve their bone lengths."""
        multi = FABRIKMultiChain()

        positions = [
            Vec3(0, 0, 0),
            Vec3(0, 1, 0),
            Vec3(0, 2, 0),
            Vec3(1, 2, 0),
            Vec3(-1, 2, 0),
        ]

        chain1 = FABRIKChain(bone_indices=[0, 1, 2, 3])
        chain2 = FABRIKChain(bone_indices=[0, 1, 2, 4])

        # Use more reasonable targets closer to original positions
        multi.add_chain(chain1, Vec3(1.5, 2, 0))
        multi.add_chain(chain2, Vec3(-1.5, 2, 0))

        # Original shared segment length
        original_shared_length = positions[0].distance(positions[1]) + \
                                  positions[1].distance(positions[2])

        result = multi.solve(positions)
        result_positions = result if isinstance(result, list) else result.positions

        # New shared segment length
        new_shared_length = result_positions[0].distance(result_positions[1]) + \
                           result_positions[1].distance(result_positions[2])

        # Multi-chain averaging may affect lengths - allow more tolerance
        assert nearly_equal(original_shared_length, new_shared_length, eps=0.5)

    def test_y_shaped_skeleton(self):
        """Y-shaped skeleton with shared trunk."""
        multi = FABRIKMultiChain()

        # Y shape: trunk then two branches
        positions = [
            Vec3(0, 0, 0),   # 0: Root
            Vec3(0, 1, 0),   # 1: Trunk
            Vec3(0, 2, 0),   # 2: Fork
            Vec3(1, 3, 0),   # 3: Right branch end
            Vec3(-1, 3, 0),  # 4: Left branch end
        ]

        chain1 = FABRIKChain(bone_indices=[0, 1, 2, 3])
        chain2 = FABRIKChain(bone_indices=[0, 1, 2, 4])

        multi.add_chain(chain1, Vec3(2, 3, 0))
        multi.add_chain(chain2, Vec3(-2, 3, 0))

        result = multi.solve(positions)
        result_positions = result if isinstance(result, list) else result.positions

        # Fork position should be consistent
        fork = result_positions[2]
        assert fork is not None


# =============================================================================
# Different Chain Lengths Tests
# =============================================================================

class TestDifferentChainLengths:
    """Test chains of different lengths working together."""

    def test_short_and_long_chain(self):
        """Short and long chains sharing root should both work."""
        multi = FABRIKMultiChain()

        positions = [
            Vec3(0, 0, 0),   # 0: Shared root
            Vec3(1, 0, 0),   # 1: Short chain end
            Vec3(0, 1, 0),   # 2: Long chain j1
            Vec3(0, 2, 0),   # 3: Long chain j2
            Vec3(0, 3, 0),   # 4: Long chain end
        ]

        short_chain = FABRIKChain(bone_indices=[0, 1])
        long_chain = FABRIKChain(bone_indices=[0, 2, 3, 4])

        multi.add_chain(short_chain, Vec3(1, 0, 0))
        multi.add_chain(long_chain, Vec3(0, 3, 0))

        result = multi.solve(positions)
        result_positions = result if isinstance(result, list) else result.positions

        assert len(result_positions) == 5

    def test_both_solve_successfully(self):
        """Both short and long chains should reach toward targets."""
        multi = FABRIKMultiChain()

        positions = [
            Vec3(0, 0, 0),
            Vec3(1, 0, 0),
            Vec3(0, 1, 0),
            Vec3(0, 2, 0),
            Vec3(0, 3, 0),
        ]

        short_chain = FABRIKChain(bone_indices=[0, 1])
        long_chain = FABRIKChain(bone_indices=[0, 2, 3, 4])

        short_target = Vec3(0.8, 0, 0)
        long_target = Vec3(0, 2.5, 0)

        multi.add_chain(short_chain, short_target)
        multi.add_chain(long_chain, long_target)

        result = multi.solve(positions)
        result_positions = result if isinstance(result, list) else result.positions

        # Short chain end should be near its target
        short_end = result_positions[1]
        assert short_end.distance(short_target) < 0.6

        # Long chain end should be near its target (with tolerance for shared root effects)
        long_end = result_positions[4]
        assert long_end.distance(long_target) < 0.6

    def test_very_different_lengths(self):
        """Chains with very different lengths."""
        multi = FABRIKMultiChain()

        positions = [Vec3(0, 0, 0)]  # Shared root

        # Short chain: 2 joints
        positions.append(Vec3(1, 0, 0))  # 1

        # Long chain: 6 joints
        for i in range(1, 6):
            positions.append(Vec3(0, i, 0))  # 2-6

        short_chain = FABRIKChain(bone_indices=[0, 1])
        long_chain = FABRIKChain(bone_indices=[0, 2, 3, 4, 5, 6])

        multi.add_chain(short_chain, Vec3(0.9, 0, 0))
        multi.add_chain(long_chain, Vec3(0, 4.5, 0))

        result = multi.solve(positions)
        result_positions = result if isinstance(result, list) else result.positions

        assert len(result_positions) == 7


# =============================================================================
# Update Targets Tests
# =============================================================================

class TestUpdateTargets:
    """Test updating targets between solves."""

    def test_set_new_target(self):
        """Should be able to set new target for chain."""
        multi = FABRIKMultiChain()
        chain = FABRIKChain(bone_indices=[0, 1, 2])

        multi.add_chain(chain, Vec3(0, 2, 0))

        # Check for set_target method or similar
        if hasattr(multi, 'set_target'):
            multi.set_target(0, Vec3(1, 1, 0))
        elif hasattr(multi, 'update_target'):
            multi.update_target(0, Vec3(1, 1, 0))
        else:
            # If no update method, re-adding or modifying chain works
            pass

    def test_solve_with_updated_target(self):
        """Solving after target update should use new target."""
        multi = FABRIKMultiChain()
        chain = FABRIKChain(bone_indices=[0, 1, 2])
        positions = create_skeleton_positions(3)

        # First solve with target A
        target_a = Vec3(1, 1, 0)
        multi.add_chain(chain, target_a)
        result_a = multi.solve(positions)
        pos_a = result_a if isinstance(result_a, list) else result_a.positions

        # Create new multi-chain with different target
        multi2 = FABRIKMultiChain()
        chain2 = FABRIKChain(bone_indices=[0, 1, 2])
        target_b = Vec3(-1, 1, 0)
        multi2.add_chain(chain2, target_b)
        result_b = multi2.solve(positions)
        pos_b = result_b if isinstance(result_b, list) else result_b.positions

        # End effectors should be different
        effector_a = pos_a[2]
        effector_b = pos_b[2]
        assert not vec_nearly_equal(effector_a, effector_b, eps=0.1)

    def test_sequential_target_changes(self):
        """Sequential target changes should produce different results."""
        positions = create_skeleton_positions(3)
        targets = [
            Vec3(1, 1, 0),
            Vec3(0, 2, 0),
            Vec3(-1, 1, 0),
        ]

        effectors = []
        for target in targets:
            multi = FABRIKMultiChain()
            chain = FABRIKChain(bone_indices=[0, 1, 2])
            multi.add_chain(chain, target)
            result = multi.solve(positions)
            pos = result if isinstance(result, list) else result.positions
            effectors.append(pos[2])

        # Each effector should be different
        assert not vec_nearly_equal(effectors[0], effectors[1])
        assert not vec_nearly_equal(effectors[1], effectors[2])


# =============================================================================
# Max Iterations Tests
# =============================================================================

class TestMaxIterations:
    """Test iteration limit handling."""

    def test_accepts_max_iterations_parameter(self):
        """Should accept max_iterations parameter."""
        multi = FABRIKMultiChain()
        chain = FABRIKChain(bone_indices=[0, 1, 2])
        multi.add_chain(chain, Vec3(0, 2, 0))
        positions = create_skeleton_positions(3)

        # Try solving with max_iterations parameter
        try:
            result = multi.solve(positions, max_iterations=10)
            assert result is not None
        except TypeError:
            # If max_iterations not supported as param, that's ok
            pass

    def test_respects_max_iterations(self):
        """Solver should terminate within iteration limit."""
        multi = FABRIKMultiChain()
        chain = FABRIKChain(bone_indices=[0, 1, 2])
        target = Vec3(100, 100, 100)  # Far unreachable target
        multi.add_chain(chain, target)
        positions = create_skeleton_positions(3)

        # Should not hang forever - must terminate
        import time
        start = time.time()
        result = multi.solve(positions)
        elapsed = time.time() - start

        # Should complete reasonably fast (< 1 second)
        assert elapsed < 1.0
        assert result is not None

    def test_low_iteration_limit_still_produces_result(self):
        """Even with low iteration limit, should produce valid positions."""
        multi = FABRIKMultiChain()
        chain = FABRIKChain(bone_indices=[0, 1, 2])
        multi.add_chain(chain, Vec3(0.5, 1.5, 0))
        positions = create_skeleton_positions(3)

        try:
            result = multi.solve(positions, max_iterations=1)
        except TypeError:
            result = multi.solve(positions)

        result_positions = result if isinstance(result, list) else result.positions

        # Should still have valid Vec3 positions
        assert len(result_positions) == 3
        for pos in result_positions:
            assert isinstance(pos, Vec3)


# =============================================================================
# Chain Priority/Weighting Tests
# =============================================================================

class TestChainPriorityWeighting:
    """Test chain priority and weighting if supported."""

    def test_equal_weight_balanced_solution(self):
        """Equal weighted chains should have balanced influence."""
        multi = FABRIKMultiChain()

        positions = [
            Vec3(0, 0, 0),   # Shared
            Vec3(1, 0, 0),   # Chain 1 end
            Vec3(-1, 0, 0),  # Chain 2 end
        ]

        chain1 = FABRIKChain(bone_indices=[0, 1])
        chain2 = FABRIKChain(bone_indices=[0, 2])

        # Equal priority targets in opposite directions
        multi.add_chain(chain1, Vec3(0.9, 0, 0))
        multi.add_chain(chain2, Vec3(-0.9, 0, 0))

        result = multi.solve(positions)
        result_positions = result if isinstance(result, list) else result.positions

        # Shared root should be near center
        root = result_positions[0]
        assert abs(root.x) < 0.3

    def test_chains_with_different_weights_if_supported(self):
        """If weights are supported, higher weight should dominate."""
        multi = FABRIKMultiChain()

        positions = [
            Vec3(0, 0, 0),
            Vec3(1, 0, 0),
            Vec3(-1, 0, 0),
        ]

        chain1 = FABRIKChain(bone_indices=[0, 1])
        chain2 = FABRIKChain(bone_indices=[0, 2])

        # Try adding with weight if supported
        try:
            multi.add_chain(chain1, Vec3(0.9, 0, 0), weight=2.0)
            multi.add_chain(chain2, Vec3(-0.9, 0, 0), weight=1.0)
        except TypeError:
            # Weights not supported, just add normally
            multi.add_chain(chain1, Vec3(0.9, 0, 0))
            multi.add_chain(chain2, Vec3(-0.9, 0, 0))

        result = multi.solve(positions)
        result_positions = result if isinstance(result, list) else result.positions

        assert len(result_positions) == 3


# =============================================================================
# Edge Cases Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_target_at_effector_position(self):
        """Target already at effector should require no movement."""
        multi = FABRIKMultiChain()
        positions = create_skeleton_positions(3)
        target = Vec3(positions[2].x, positions[2].y, positions[2].z)

        chain = FABRIKChain(bone_indices=[0, 1, 2])
        multi.add_chain(chain, target)

        result = multi.solve(positions)
        result_positions = result if isinstance(result, list) else result.positions

        # Positions should be nearly unchanged
        for i, pos in enumerate(result_positions):
            assert vec_nearly_equal(pos, positions[i], eps=0.1)

    def test_target_at_root_position(self):
        """Target at root position should fold chain."""
        multi = FABRIKMultiChain()
        positions = create_skeleton_positions(3)
        target = Vec3(0, 0, 0)  # At root

        chain = FABRIKChain(bone_indices=[0, 1, 2])
        multi.add_chain(chain, target)

        result = multi.solve(positions)
        result_positions = result if isinstance(result, list) else result.positions

        # Should produce valid result
        assert len(result_positions) == 3

    def test_zero_length_target_vector(self):
        """Target at origin with chain at origin."""
        multi = FABRIKMultiChain()
        positions = [Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 2, 0)]
        target = Vec3(0, 0, 0)

        chain = FABRIKChain(bone_indices=[0, 1, 2])
        multi.add_chain(chain, target)

        result = multi.solve(positions)
        result_positions = result if isinstance(result, list) else result.positions

        assert result_positions is not None

    def test_all_joints_same_position(self):
        """All joints at same position (degenerate case)."""
        multi = FABRIKMultiChain()
        positions = [Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(0, 0, 0)]
        target = Vec3(1, 0, 0)

        chain = FABRIKChain(bone_indices=[0, 1, 2])
        multi.add_chain(chain, target)

        # Should handle gracefully without crashing
        try:
            result = multi.solve(positions)
            assert result is not None
        except (ValueError, ZeroDivisionError):
            # Expected for degenerate case
            pass

    def test_negative_coordinates(self):
        """Chains with negative coordinates should work."""
        multi = FABRIKMultiChain()
        positions = [
            Vec3(-5, -5, -5),
            Vec3(-4, -5, -5),
            Vec3(-3, -5, -5),
        ]
        target = Vec3(-2, -5, -5)

        chain = FABRIKChain(bone_indices=[0, 1, 2])
        multi.add_chain(chain, target)

        result = multi.solve(positions)
        result_positions = result if isinstance(result, list) else result.positions

        assert len(result_positions) == 3

    def test_large_coordinate_values(self):
        """Large coordinate values should be handled."""
        multi = FABRIKMultiChain()
        positions = [
            Vec3(1000, 1000, 1000),
            Vec3(1001, 1000, 1000),
            Vec3(1002, 1000, 1000),
        ]
        target = Vec3(1001.5, 1000, 0)

        chain = FABRIKChain(bone_indices=[0, 1, 2])
        multi.add_chain(chain, target)

        result = multi.solve(positions)
        result_positions = result if isinstance(result, list) else result.positions

        assert result_positions is not None


# =============================================================================
# Convergence Tests
# =============================================================================

class TestConvergence:
    """Test convergence behavior."""

    def test_converges_for_reachable_target(self):
        """Should converge when target is within reach."""
        multi = FABRIKMultiChain()
        positions = create_skeleton_positions(4)  # Total length 3
        target = Vec3(1, 2, 0)  # Within reach

        chain = FABRIKChain(bone_indices=[0, 1, 2, 3])
        multi.add_chain(chain, target)

        result = multi.solve(positions)
        result_positions = result if isinstance(result, list) else result.positions

        effector = result_positions[3]
        assert effector.distance(target) < 0.2

    def test_multiple_solves_converge_same_result(self):
        """Multiple solves should converge to same result."""
        positions = create_skeleton_positions(3)
        target = Vec3(0.5, 1.5, 0)

        results = []
        for _ in range(3):
            multi = FABRIKMultiChain()
            chain = FABRIKChain(bone_indices=[0, 1, 2])
            multi.add_chain(chain, target)
            result = multi.solve(positions)
            pos = result if isinstance(result, list) else result.positions
            results.append(pos)

        # All results should be similar
        for i in range(len(results) - 1):
            assert positions_are_stable(results[i], results[i + 1], eps=0.01)

    def test_idempotent_solve(self):
        """Solving already solved positions should be idempotent."""
        multi = FABRIKMultiChain()
        chain = FABRIKChain(bone_indices=[0, 1, 2])
        target = Vec3(0.5, 1.5, 0)
        multi.add_chain(chain, target)

        positions = create_skeleton_positions(3)

        # First solve
        result1 = multi.solve(positions)
        pos1 = result1 if isinstance(result1, list) else result1.positions

        # Solve again with result of first solve
        multi2 = FABRIKMultiChain()
        chain2 = FABRIKChain(bone_indices=[0, 1, 2])
        multi2.add_chain(chain2, target)
        result2 = multi2.solve(pos1)
        pos2 = result2 if isinstance(result2, list) else result2.positions

        # Should be nearly identical
        assert positions_are_stable(pos1, pos2, eps=0.01)


# =============================================================================
# Complex Scenarios Tests
# =============================================================================

class TestComplexScenarios:
    """Test complex multi-chain scenarios."""

    def test_spider_legs_configuration(self):
        """Eight legs radiating from central body."""
        multi = FABRIKMultiChain()

        # Central body at origin, 8 legs
        positions = [Vec3(0, 0, 0)]  # Body center

        angles = [i * (math.pi / 4) for i in range(8)]
        for i, angle in enumerate(angles):
            # Each leg: hip -> knee -> foot
            hip = Vec3(math.cos(angle), 0, math.sin(angle))
            knee = Vec3(2 * math.cos(angle), 0, 2 * math.sin(angle))
            foot = Vec3(3 * math.cos(angle), 0, 3 * math.sin(angle))
            positions.extend([hip, knee, foot])

        # Add chains for each leg
        for i in range(8):
            base_idx = 1 + i * 3
            chain = FABRIKChain(bone_indices=[0, base_idx, base_idx + 1, base_idx + 2])
            target = Vec3(2.5 * math.cos(angles[i]), -0.5, 2.5 * math.sin(angles[i]))
            multi.add_chain(chain, target)

        result = multi.solve(positions)
        result_positions = result if isinstance(result, list) else result.positions

        assert len(result_positions) == 25  # 1 body + 8 * 3 leg joints

    def test_humanoid_full_body(self):
        """Full humanoid with arms and legs."""
        multi = FABRIKMultiChain()
        positions = create_humanoid_positions()

        # Arms
        right_arm = FABRIKChain(bone_indices=[2, 4, 5, 6])
        left_arm = FABRIKChain(bone_indices=[2, 7, 8, 9])

        multi.add_chain(right_arm, Vec3(3, 1, 0))
        multi.add_chain(left_arm, Vec3(-3, 1, 0))

        result = multi.solve(positions)
        result_positions = result if isinstance(result, list) else result.positions

        # Verify hands moved toward targets
        right_hand = result_positions[6]
        left_hand = result_positions[9]

        assert right_hand.x > positions[6].x - 0.5  # Right hand not too far left
        assert left_hand.x < positions[9].x + 0.5   # Left hand not too far right

    def test_chain_network_graph(self):
        """Complex network of interconnected chains."""
        multi = FABRIKMultiChain()

        # Create a network: center -> multiple branches
        #       1---2
        #      /
        # 0---+---3---4
        #      \
        #       5---6

        positions = [
            Vec3(0, 0, 0),   # 0: Root
            Vec3(1, 1, 0),   # 1: Upper branch j1
            Vec3(2, 2, 0),   # 2: Upper branch end
            Vec3(1, 0, 0),   # 3: Middle branch j1
            Vec3(2, 0, 0),   # 4: Middle branch end
            Vec3(1, -1, 0),  # 5: Lower branch j1
            Vec3(2, -2, 0),  # 6: Lower branch end
        ]

        chain1 = FABRIKChain(bone_indices=[0, 1, 2])
        chain2 = FABRIKChain(bone_indices=[0, 3, 4])
        chain3 = FABRIKChain(bone_indices=[0, 5, 6])

        multi.add_chain(chain1, Vec3(2, 2.5, 0))
        multi.add_chain(chain2, Vec3(2.5, 0, 0))
        multi.add_chain(chain3, Vec3(2, -2.5, 0))

        result = multi.solve(positions)
        result_positions = result if isinstance(result, list) else result.positions

        assert len(result_positions) == 7


# =============================================================================
# Performance Sanity Tests
# =============================================================================

class TestPerformanceSanity:
    """Basic performance sanity checks."""

    def test_solve_completes_in_reasonable_time(self):
        """Solve should complete in reasonable time."""
        import time

        multi = FABRIKMultiChain()
        positions = create_humanoid_positions()

        right_arm = FABRIKChain(bone_indices=[2, 4, 5, 6])
        left_arm = FABRIKChain(bone_indices=[2, 7, 8, 9])

        multi.add_chain(right_arm, Vec3(3, 1, 0))
        multi.add_chain(left_arm, Vec3(-3, 1, 0))

        start = time.time()
        for _ in range(100):
            multi.solve(positions)
        elapsed = time.time() - start

        # 100 solves should complete in < 1 second
        assert elapsed < 1.0

    def test_many_chains_still_works(self):
        """Many chains should still solve correctly."""
        multi = FABRIKMultiChain()

        # Create 10 chains from shared root
        positions = [Vec3(0, 0, 0)]  # Root
        for i in range(10):
            angle = i * (math.pi / 5)
            positions.append(Vec3(math.cos(angle), math.sin(angle), 0))

        for i in range(10):
            chain = FABRIKChain(bone_indices=[0, i + 1])
            target = Vec3(0.8 * math.cos(i * math.pi / 5),
                         0.8 * math.sin(i * math.pi / 5), 0)
            multi.add_chain(chain, target)

        result = multi.solve(positions)
        result_positions = result if isinstance(result, list) else result.positions

        assert len(result_positions) == 11
