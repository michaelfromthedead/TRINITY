"""Whitebox tests for TwoBoneIK solver.

Tests the analytical two-bone IK implementation covering:
- TwoBoneIK class construction
- TwoBoneIKResult dataclass
- Law of cosines angle calculation
- Numerical stability clamping
- solve() method with various scenarios
- Pole vector handling
- Soft IK behavior
- Edge cases and bone length caching
"""

from __future__ import annotations

import math
import pytest
from dataclasses import fields
from typing import List

from engine.animation.ik.two_bone import TwoBoneIK, TwoBoneIKResult, TwoBoneIKConstraint
from engine.animation.ik.config import (
    SOFT_IK_DEFAULT_RATIO,
    SOFT_IK_DEFAULT_BLEND,
    SOFT_IK_FALLOFF_RATE,
    JOINT_MIN_BEND_ANGLE,
    JOINT_MAX_BEND_ANGLE,
)
from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.core.math.transform import Transform
from engine.core.constants import MATH_EPSILON


# =============================================================================
# Helper Functions
# =============================================================================

def create_arm_chain(
    root_pos: Vec3 = None,
    upper_length: float = 1.0,
    lower_length: float = 1.0
) -> tuple[Transform, Transform, Transform]:
    """Create a simple arm-like chain for testing.

    Args:
        root_pos: Position of root bone
        upper_length: Length of upper bone
        lower_length: Length of lower bone

    Returns:
        Tuple of (root_transform, mid_transform, end_transform)
    """
    if root_pos is None:
        root_pos = Vec3(0, 0, 0)

    # Chain extends along positive Y axis
    mid_pos = Vec3(root_pos.x, root_pos.y + upper_length, root_pos.z)
    end_pos = Vec3(mid_pos.x, mid_pos.y + lower_length, mid_pos.z)

    root_transform = Transform(root_pos, Quat.identity())
    mid_transform = Transform(mid_pos, Quat.identity())
    end_transform = Transform(end_pos, Quat.identity())

    return root_transform, mid_transform, end_transform


def vec3_approx_equal(v1: Vec3, v2: Vec3, eps: float = 1e-5) -> bool:
    """Check if two Vec3 are approximately equal."""
    return (
        abs(v1.x - v2.x) < eps and
        abs(v1.y - v2.y) < eps and
        abs(v1.z - v2.z) < eps
    )


def quat_approx_equal(q1: Quat, q2: Quat, eps: float = 1e-5) -> bool:
    """Check if two Quaternions are approximately equal (accounting for sign)."""
    dot = abs(q1.x * q2.x + q1.y * q2.y + q1.z * q2.z + q1.w * q2.w)
    return dot > (1.0 - eps)


# =============================================================================
# Test TwoBoneIKResult Dataclass
# =============================================================================

class TestTwoBoneIKResult:
    """Tests for TwoBoneIKResult dataclass structure and fields."""

    def test_result_dataclass_has_required_fields(self):
        """Verify TwoBoneIKResult has all required fields."""
        field_names = {f.name for f in fields(TwoBoneIKResult)}
        expected_fields = {
            'success', 'root_rotation', 'mid_rotation',
            'end_rotation', 'target_reached', 'extension_ratio'
        }
        assert expected_fields.issubset(field_names), \
            f"Missing fields: {expected_fields - field_names}"

    def test_result_default_values(self):
        """Test that default values are set correctly."""
        result = TwoBoneIKResult(success=True)

        assert result.success is True
        assert isinstance(result.root_rotation, Quat)
        assert isinstance(result.mid_rotation, Quat)
        assert isinstance(result.end_rotation, Quat)
        assert result.target_reached is False
        assert result.extension_ratio == 0.0

    def test_result_with_custom_values(self):
        """Test creating result with custom values."""
        rot = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 4)
        result = TwoBoneIKResult(
            success=True,
            root_rotation=rot,
            mid_rotation=rot,
            end_rotation=rot,
            target_reached=True,
            extension_ratio=0.75
        )

        assert result.success is True
        assert result.root_rotation == rot
        assert result.mid_rotation == rot
        assert result.end_rotation == rot
        assert result.target_reached is True
        assert result.extension_ratio == 0.75

    def test_result_identity_rotations_by_default(self):
        """Verify default rotations are identity quaternions."""
        result = TwoBoneIKResult(success=False)

        identity = Quat.identity()
        assert quat_approx_equal(result.root_rotation, identity)
        assert quat_approx_equal(result.mid_rotation, identity)
        assert quat_approx_equal(result.end_rotation, identity)


# =============================================================================
# Test TwoBoneIK Class Construction
# =============================================================================

class TestTwoBoneIKConstruction:
    """Tests for TwoBoneIK class initialization."""

    def test_valid_construction_with_positive_indices(self):
        """Test construction with valid positive bone indices."""
        solver = TwoBoneIK(root_bone=0, mid_bone=1, end_bone=2)

        assert solver.root_bone == 0
        assert solver.mid_bone == 1
        assert solver.end_bone == 2

    def test_construction_with_non_consecutive_indices(self):
        """Test construction with non-consecutive bone indices."""
        solver = TwoBoneIK(root_bone=5, mid_bone=10, end_bone=15)

        assert solver.root_bone == 5
        assert solver.mid_bone == 10
        assert solver.end_bone == 15

    def test_construction_raises_on_negative_root_bone(self):
        """Test that negative root_bone raises ValueError."""
        with pytest.raises(ValueError, match="Bone indices must be non-negative"):
            TwoBoneIK(root_bone=-1, mid_bone=1, end_bone=2)

    def test_construction_raises_on_negative_mid_bone(self):
        """Test that negative mid_bone raises ValueError."""
        with pytest.raises(ValueError, match="Bone indices must be non-negative"):
            TwoBoneIK(root_bone=0, mid_bone=-1, end_bone=2)

    def test_construction_raises_on_negative_end_bone(self):
        """Test that negative end_bone raises ValueError."""
        with pytest.raises(ValueError, match="Bone indices must be non-negative"):
            TwoBoneIK(root_bone=0, mid_bone=1, end_bone=-1)

    def test_construction_with_zero_indices(self):
        """Test construction with all zero indices (valid edge case)."""
        solver = TwoBoneIK(root_bone=0, mid_bone=0, end_bone=0)

        assert solver.root_bone == 0
        assert solver.mid_bone == 0
        assert solver.end_bone == 0

    def test_soft_ik_ratio_default(self):
        """Test default soft IK ratio is 0."""
        solver = TwoBoneIK(0, 1, 2)
        assert solver.soft_ik_ratio == 0.0

    def test_soft_ik_blend_default(self):
        """Test default soft IK blend is 1."""
        solver = TwoBoneIK(0, 1, 2)
        assert solver.soft_ik_blend == 1.0

    def test_soft_ik_ratio_clamped_below_zero(self):
        """Test soft IK ratio is clamped to 0 when negative."""
        solver = TwoBoneIK(0, 1, 2, soft_ik_ratio=-0.5)
        assert solver.soft_ik_ratio == 0.0

    def test_soft_ik_ratio_clamped_above_one(self):
        """Test soft IK ratio is clamped to 1 when above 1."""
        solver = TwoBoneIK(0, 1, 2, soft_ik_ratio=1.5)
        assert solver.soft_ik_ratio == 1.0

    def test_soft_ik_blend_clamped_below_zero(self):
        """Test soft IK blend is clamped to 0 when negative."""
        solver = TwoBoneIK(0, 1, 2, soft_ik_blend=-0.5)
        assert solver.soft_ik_blend == 0.0

    def test_soft_ik_blend_clamped_above_one(self):
        """Test soft IK blend is clamped to 1 when above 1."""
        solver = TwoBoneIK(0, 1, 2, soft_ik_blend=1.5)
        assert solver.soft_ik_blend == 1.0

    def test_bone_lengths_not_cached_initially(self):
        """Test that bone lengths are not cached on construction."""
        solver = TwoBoneIK(0, 1, 2)

        assert solver._lengths_cached is False
        assert solver._upper_length == 0.0
        assert solver._lower_length == 0.0
        assert solver._total_length == 0.0


# =============================================================================
# Test Bone Length Caching
# =============================================================================

class TestBoneLengthCaching:
    """Tests for bone length caching mechanism."""

    def test_lengths_cached_after_first_solve(self):
        """Test that bone lengths are cached after first solve."""
        solver = TwoBoneIK(0, 1, 2)
        root, mid, end = create_arm_chain(upper_length=1.0, lower_length=1.0)

        solver.solve(root, mid, end, Vec3(0, 1, 0))

        assert solver._lengths_cached is True
        assert abs(solver._upper_length - 1.0) < MATH_EPSILON
        assert abs(solver._lower_length - 1.0) < MATH_EPSILON
        assert abs(solver._total_length - 2.0) < MATH_EPSILON

    def test_reset_cached_lengths(self):
        """Test that reset_cached_lengths clears cached values."""
        solver = TwoBoneIK(0, 1, 2)
        root, mid, end = create_arm_chain()

        solver.solve(root, mid, end, Vec3(0, 1, 0))
        assert solver._lengths_cached is True

        solver.reset_cached_lengths()

        assert solver._lengths_cached is False
        assert solver._upper_length == 0.0
        assert solver._lower_length == 0.0
        assert solver._total_length == 0.0

    def test_max_reach_property(self):
        """Test max_reach returns total chain length."""
        solver = TwoBoneIK(0, 1, 2)
        root, mid, end = create_arm_chain(upper_length=1.5, lower_length=2.0)

        solver.solve(root, mid, end, Vec3(0, 1, 0))

        assert abs(solver.max_reach - 3.5) < MATH_EPSILON

    def test_min_reach_property(self):
        """Test min_reach returns difference of bone lengths."""
        solver = TwoBoneIK(0, 1, 2)
        root, mid, end = create_arm_chain(upper_length=2.0, lower_length=1.0)

        solver.solve(root, mid, end, Vec3(0, 1, 0))

        assert abs(solver.min_reach - 1.0) < MATH_EPSILON

    def test_min_reach_with_equal_bones(self):
        """Test min_reach is 0 with equal bone lengths."""
        solver = TwoBoneIK(0, 1, 2)
        root, mid, end = create_arm_chain(upper_length=1.5, lower_length=1.5)

        solver.solve(root, mid, end, Vec3(0, 1, 0))

        assert abs(solver.min_reach) < MATH_EPSILON

    def test_cache_bone_lengths_directly(self):
        """Test _cache_bone_lengths internal method."""
        solver = TwoBoneIK(0, 1, 2)

        root_pos = Vec3(0, 0, 0)
        mid_pos = Vec3(0, 3, 0)
        end_pos = Vec3(0, 5, 0)

        solver._cache_bone_lengths(root_pos, mid_pos, end_pos)

        assert solver._lengths_cached is True
        assert abs(solver._upper_length - 3.0) < MATH_EPSILON
        assert abs(solver._lower_length - 2.0) < MATH_EPSILON
        assert abs(solver._total_length - 5.0) < MATH_EPSILON


# =============================================================================
# Test Law of Cosines Angle Calculation
# =============================================================================

class TestLawOfCosinesCalculation:
    """Tests for law of cosines angle calculation in solve().

    The formula: cos_mid = (a^2 + b^2 - c^2) / (2ab)
    where a = upper_len, b = lower_len, c = target_dist
    """

    def test_equilateral_triangle_configuration(self):
        """Test with equilateral triangle (all sides equal)."""
        # upper = lower = target_dist = 1.0
        # cos(angle) = (1 + 1 - 1) / (2*1*1) = 0.5
        # angle = acos(0.5) = pi/3 = 60 degrees
        solver = TwoBoneIK(0, 1, 2)
        root, mid, end = create_arm_chain(upper_length=1.0, lower_length=1.0)

        # Target at distance 1.0 from root
        target = Vec3(0, 1, 0)
        result = solver.solve(root, mid, end, target)

        assert result.success is True
        # For equilateral, mid joint should be bent to 120 degrees (pi - pi/3)
        # Check extension_ratio is roughly 0.5 (since target is at 1.0, max is 2.0)
        assert 0.4 < result.extension_ratio < 0.6

    def test_fully_extended_chain(self):
        """Test with target at maximum reach (chain fully extended)."""
        solver = TwoBoneIK(0, 1, 2)
        root, mid, end = create_arm_chain(upper_length=1.0, lower_length=1.0)

        # Target at max reach (2.0)
        target = Vec3(0, 2.0, 0)
        result = solver.solve(root, mid, end, target)

        assert result.success is True
        assert result.target_reached is True
        # Extension ratio should be close to 1.0
        assert result.extension_ratio > 0.95

    def test_folded_chain_target_close_to_root(self):
        """Test with target very close to root (chain folded)."""
        solver = TwoBoneIK(0, 1, 2)
        root, mid, end = create_arm_chain(upper_length=1.0, lower_length=1.0)

        # Target very close to root
        target = Vec3(0, 0.1, 0)
        result = solver.solve(root, mid, end, target)

        assert result.success is True
        # Extension ratio should be very low
        assert result.extension_ratio < 0.1

    def test_right_angle_configuration(self):
        """Test configuration forming right angle at mid joint."""
        # For right angle (90 degrees): cos(90) = 0
        # 0 = (a^2 + b^2 - c^2) / (2ab)
        # c^2 = a^2 + b^2 (Pythagorean theorem)
        solver = TwoBoneIK(0, 1, 2)
        root, mid, end = create_arm_chain(upper_length=1.0, lower_length=1.0)

        # Target distance = sqrt(2) for 90-degree bend with equal bones
        target = Vec3(0, math.sqrt(2), 0)
        result = solver.solve(root, mid, end, target)

        assert result.success is True
        # Mid should be bent approximately 90 degrees
        expected_ratio = (math.sqrt(2)) / 2.0  # normalized position
        assert abs(result.extension_ratio - expected_ratio) < 0.1

    def test_isoceles_triangle_configuration(self):
        """Test with isoceles triangle (upper = lower != target)."""
        solver = TwoBoneIK(0, 1, 2)
        root, mid, end = create_arm_chain(upper_length=1.5, lower_length=1.5)

        # Target at 2.0 (between min 0 and max 3.0)
        target = Vec3(0, 2.0, 0)
        result = solver.solve(root, mid, end, target)

        assert result.success is True
        expected_ratio = 2.0 / 3.0
        assert abs(result.extension_ratio - expected_ratio) < 0.1

    def test_asymmetric_bone_lengths(self):
        """Test with asymmetric bone lengths."""
        solver = TwoBoneIK(0, 1, 2)
        root, mid, end = create_arm_chain(upper_length=2.0, lower_length=1.0)

        # Target at 2.5 (between min 1.0 and max 3.0)
        target = Vec3(0, 2.5, 0)
        result = solver.solve(root, mid, end, target)

        assert result.success is True
        # Extension ratio = (2.5 - 1.0) / (3.0 - 1.0) = 0.75
        assert abs(result.extension_ratio - 0.75) < 0.1


# =============================================================================
# Test Numerical Stability Clamping
# =============================================================================

class TestNumericalStabilityClamping:
    """Tests for cos clamping to [-1, 1] for numerical stability."""

    def test_clamp_cos_above_one(self):
        """Test that cos values slightly above 1 are handled."""
        solver = TwoBoneIK(0, 1, 2)
        root, mid, end = create_arm_chain(upper_length=1.0, lower_length=1.0)

        # Target at exact max reach - should handle floating point edge case
        target = Vec3(0, 2.0 - MATH_EPSILON * 10, 0)
        result = solver.solve(root, mid, end, target)

        assert result.success is True
        # Should not raise math domain error

    def test_clamp_cos_below_minus_one(self):
        """Test that cos values slightly below -1 are handled."""
        solver = TwoBoneIK(0, 1, 2)
        root, mid, end = create_arm_chain(upper_length=1.0, lower_length=1.0)

        # Target beyond max reach - tests clamping
        target = Vec3(0, 2.5, 0)  # Beyond max reach of 2.0
        result = solver.solve(root, mid, end, target)

        assert result.success is True
        # Should not crash, target_reached should be False
        assert result.target_reached is False

    def test_very_small_bone_lengths(self):
        """Test with very small bone lengths for numerical stability."""
        solver = TwoBoneIK(0, 1, 2)
        root, mid, end = create_arm_chain(upper_length=0.001, lower_length=0.001)

        target = Vec3(0, 0.001, 0)
        result = solver.solve(root, mid, end, target)

        assert result.success is True

    def test_degenerate_zero_length_bone_fails(self):
        """Test that zero-length bones return failure."""
        solver = TwoBoneIK(0, 1, 2)

        # Create chain with zero-length upper bone
        root = Transform(Vec3(0, 0, 0), Quat.identity())
        mid = Transform(Vec3(0, 0, 0), Quat.identity())  # Same as root
        end = Transform(Vec3(0, 1, 0), Quat.identity())

        target = Vec3(0, 0.5, 0)
        result = solver.solve(root, mid, end, target)

        assert result.success is False

    def test_degenerate_zero_length_lower_bone_fails(self):
        """Test that zero-length lower bone returns failure."""
        solver = TwoBoneIK(0, 1, 2)

        root = Transform(Vec3(0, 0, 0), Quat.identity())
        mid = Transform(Vec3(0, 1, 0), Quat.identity())
        end = Transform(Vec3(0, 1, 0), Quat.identity())  # Same as mid

        target = Vec3(0, 0.5, 0)
        result = solver.solve(root, mid, end, target)

        assert result.success is False


# =============================================================================
# Test solve() Method - Reachable Targets
# =============================================================================

class TestSolveReachableTargets:
    """Tests for solve() with targets within reach."""

    def test_solve_target_straight_ahead(self):
        """Test solving for target directly ahead (no rotation needed)."""
        solver = TwoBoneIK(0, 1, 2)
        root, mid, end = create_arm_chain(upper_length=1.0, lower_length=1.0)

        # Target straight ahead at mid-distance
        target = Vec3(0, 1.5, 0)
        result = solver.solve(root, mid, end, target)

        assert result.success is True
        assert result.target_reached is True

    def test_solve_target_to_side(self):
        """Test solving for target to the side."""
        solver = TwoBoneIK(0, 1, 2)
        root, mid, end = create_arm_chain(upper_length=1.0, lower_length=1.0)

        # Target to the right side
        target = Vec3(1.5, 0, 0)
        result = solver.solve(root, mid, end, target)

        assert result.success is True
        assert result.target_reached is True

    def test_solve_target_behind(self):
        """Test solving for target behind the chain."""
        solver = TwoBoneIK(0, 1, 2)
        root, mid, end = create_arm_chain(upper_length=1.0, lower_length=1.0)

        # Target behind (negative Y)
        target = Vec3(0, -1.5, 0)
        result = solver.solve(root, mid, end, target)

        assert result.success is True
        assert result.target_reached is True

    def test_solve_target_diagonal(self):
        """Test solving for diagonal target."""
        solver = TwoBoneIK(0, 1, 2)
        root, mid, end = create_arm_chain(upper_length=1.0, lower_length=1.0)

        # Diagonal target within reach
        target = Vec3(1.0, 1.0, 0)  # distance = sqrt(2) < 2.0
        result = solver.solve(root, mid, end, target)

        assert result.success is True
        assert result.target_reached is True

    def test_solve_target_3d(self):
        """Test solving for 3D target (all axes non-zero)."""
        solver = TwoBoneIK(0, 1, 2)
        root, mid, end = create_arm_chain(upper_length=1.0, lower_length=1.0)

        # 3D target within reach
        target = Vec3(0.5, 0.5, 0.5)  # distance = sqrt(0.75) < 2.0
        result = solver.solve(root, mid, end, target)

        assert result.success is True
        assert result.target_reached is True

    def test_solve_returns_valid_rotations(self):
        """Test that solve returns valid (normalized) quaternions."""
        solver = TwoBoneIK(0, 1, 2)
        root, mid, end = create_arm_chain(upper_length=1.0, lower_length=1.0)

        target = Vec3(1.0, 1.0, 0)
        result = solver.solve(root, mid, end, target)

        assert result.success is True

        # Check quaternions are valid (unit length)
        assert abs(result.root_rotation.length() - 1.0) < 0.01
        assert abs(result.mid_rotation.length() - 1.0) < 0.01
        assert abs(result.end_rotation.length() - 1.0) < 0.01


# =============================================================================
# Test solve() Method - Unreachable Targets
# =============================================================================

class TestSolveUnreachableTargets:
    """Tests for solve() with targets beyond chain reach."""

    def test_solve_target_beyond_max_reach(self):
        """Test solving for target beyond maximum reach."""
        solver = TwoBoneIK(0, 1, 2)
        root, mid, end = create_arm_chain(upper_length=1.0, lower_length=1.0)

        # Target beyond max reach of 2.0
        target = Vec3(0, 3.0, 0)
        result = solver.solve(root, mid, end, target)

        assert result.success is True  # Still solves, just clamped
        assert result.target_reached is False
        # Should be fully extended
        assert result.extension_ratio > 0.9

    def test_solve_target_way_beyond_reach(self):
        """Test solving for target way beyond reach."""
        solver = TwoBoneIK(0, 1, 2)
        root, mid, end = create_arm_chain(upper_length=1.0, lower_length=1.0)

        # Target way beyond reach
        target = Vec3(0, 100.0, 0)
        result = solver.solve(root, mid, end, target)

        assert result.success is True
        assert result.target_reached is False
        assert result.extension_ratio > 0.9

    def test_solve_target_at_exact_max_reach(self):
        """Test solving for target at exact maximum reach."""
        solver = TwoBoneIK(0, 1, 2)
        root, mid, end = create_arm_chain(upper_length=1.0, lower_length=1.0)

        # Target at exact max reach
        target = Vec3(0, 2.0, 0)
        result = solver.solve(root, mid, end, target)

        assert result.success is True
        assert result.target_reached is True

    def test_solve_unreachable_clamps_to_max_extension(self):
        """Test that unreachable targets result in max extension direction."""
        solver = TwoBoneIK(0, 1, 2)
        root, mid, end = create_arm_chain(upper_length=1.0, lower_length=1.0)

        # Target very far away
        target = Vec3(0, 50.0, 0)
        result = solver.solve(root, mid, end, target)

        assert result.success is True
        # Chain should point toward target
        # Extension should be maximal
        assert result.extension_ratio > 0.99


# =============================================================================
# Test solve() Method - Edge Cases
# =============================================================================

class TestSolveEdgeCases:
    """Tests for solve() edge cases."""

    def test_solve_target_at_root_position(self):
        """Test solving when target is at root position."""
        solver = TwoBoneIK(0, 1, 2)
        root, mid, end = create_arm_chain(upper_length=1.0, lower_length=1.0)

        # Target at root
        target = Vec3(0, 0, 0)
        result = solver.solve(root, mid, end, target)

        assert result.success is True
        assert result.target_reached is True  # Special case
        assert result.extension_ratio == 0.0

    def test_solve_target_very_close_to_root(self):
        """Test solving with target extremely close to root."""
        solver = TwoBoneIK(0, 1, 2)
        root, mid, end = create_arm_chain(upper_length=1.0, lower_length=1.0)

        # Target very close to root (less than epsilon)
        target = Vec3(MATH_EPSILON / 10, 0, 0)
        result = solver.solve(root, mid, end, target)

        assert result.success is True

    def test_solve_target_at_minimum_reach(self):
        """Test solving for target at minimum reach (fully bent)."""
        solver = TwoBoneIK(0, 1, 2)
        root, mid, end = create_arm_chain(upper_length=2.0, lower_length=1.0)

        # Min reach = |2.0 - 1.0| = 1.0
        target = Vec3(0, 1.0, 0)
        result = solver.solve(root, mid, end, target)

        assert result.success is True
        assert result.target_reached is True
        # Extension should be near 0 (fully bent)
        assert result.extension_ratio < 0.1

    def test_solve_with_equal_bone_lengths(self):
        """Test solving with equal bone lengths."""
        solver = TwoBoneIK(0, 1, 2)
        root, mid, end = create_arm_chain(upper_length=1.5, lower_length=1.5)

        # Min reach = 0 with equal bones
        target = Vec3(0, 0.1, 0)  # Very close to root
        result = solver.solve(root, mid, end, target)

        assert result.success is True

    def test_solve_with_large_bone_lengths(self):
        """Test solving with large bone lengths."""
        solver = TwoBoneIK(0, 1, 2)
        root, mid, end = create_arm_chain(upper_length=100.0, lower_length=100.0)

        target = Vec3(0, 150.0, 0)
        result = solver.solve(root, mid, end, target)

        assert result.success is True
        assert result.target_reached is True

    def test_solve_with_offset_root_position(self):
        """Test solving with root not at origin."""
        solver = TwoBoneIK(0, 1, 2)
        root_pos = Vec3(10, 20, 30)
        root, mid, end = create_arm_chain(root_pos, upper_length=1.0, lower_length=1.0)

        # Target relative to offset root
        target = Vec3(10, 21.5, 30)
        result = solver.solve(root, mid, end, target)

        assert result.success is True
        assert result.target_reached is True


# =============================================================================
# Test Pole Vector Handling
# =============================================================================

class TestPoleVectorHandling:
    """Tests for pole vector (bend direction) control."""

    def test_solve_without_pole_vector(self):
        """Test that solve works without pole vector."""
        solver = TwoBoneIK(0, 1, 2)
        root, mid, end = create_arm_chain(upper_length=1.0, lower_length=1.0)

        target = Vec3(1.0, 1.0, 0)
        result = solver.solve(root, mid, end, target)

        assert result.success is True

    def test_solve_with_pole_vector_forward(self):
        """Test solve with pole vector pointing forward."""
        solver = TwoBoneIK(0, 1, 2)
        root, mid, end = create_arm_chain(upper_length=1.0, lower_length=1.0)

        target = Vec3(1.0, 1.0, 0)
        pole = Vec3(0, 1, 1)  # Forward pole
        result = solver.solve(root, mid, end, target, pole_vector=pole)

        assert result.success is True

    def test_solve_with_pole_vector_backward(self):
        """Test solve with pole vector pointing backward."""
        solver = TwoBoneIK(0, 1, 2)
        root, mid, end = create_arm_chain(upper_length=1.0, lower_length=1.0)

        target = Vec3(1.0, 1.0, 0)
        pole = Vec3(0, 1, -1)  # Backward pole
        result = solver.solve(root, mid, end, target, pole_vector=pole)

        assert result.success is True

    def test_solve_with_pole_vector_left(self):
        """Test solve with pole vector pointing left."""
        solver = TwoBoneIK(0, 1, 2)
        root, mid, end = create_arm_chain(upper_length=1.0, lower_length=1.0)

        target = Vec3(0, 1.5, 0)
        pole = Vec3(-1, 1, 0)  # Left pole
        result = solver.solve(root, mid, end, target, pole_vector=pole)

        assert result.success is True

    def test_solve_with_pole_vector_right(self):
        """Test solve with pole vector pointing right."""
        solver = TwoBoneIK(0, 1, 2)
        root, mid, end = create_arm_chain(upper_length=1.0, lower_length=1.0)

        target = Vec3(0, 1.5, 0)
        pole = Vec3(1, 1, 0)  # Right pole
        result = solver.solve(root, mid, end, target, pole_vector=pole)

        assert result.success is True

    def test_pole_vector_at_root_position(self):
        """Test pole vector at root position (edge case)."""
        solver = TwoBoneIK(0, 1, 2)
        root, mid, end = create_arm_chain(upper_length=1.0, lower_length=1.0)

        target = Vec3(0, 1.5, 0)
        pole = Vec3(0, 0, 0)  # At root
        result = solver.solve(root, mid, end, target, pole_vector=pole)

        assert result.success is True

    def test_pole_vector_on_target_line(self):
        """Test pole vector directly on line to target (degenerate)."""
        solver = TwoBoneIK(0, 1, 2)
        root, mid, end = create_arm_chain(upper_length=1.0, lower_length=1.0)

        target = Vec3(0, 1.5, 0)
        pole = Vec3(0, 0.5, 0)  # On line to target
        result = solver.solve(root, mid, end, target, pole_vector=pole)

        # Should still work, using fallback
        assert result.success is True


# =============================================================================
# Test Target Rotation
# =============================================================================

class TestTargetRotation:
    """Tests for optional end effector rotation."""

    def test_solve_without_target_rotation(self):
        """Test that end rotation comes from original when no target rotation."""
        solver = TwoBoneIK(0, 1, 2)
        root, mid, end = create_arm_chain(upper_length=1.0, lower_length=1.0)

        target = Vec3(1.0, 1.0, 0)
        result = solver.solve(root, mid, end, target)

        assert result.success is True
        # End rotation should match original end_transform rotation
        assert quat_approx_equal(result.end_rotation, end.rotation)

    def test_solve_with_target_rotation(self):
        """Test that end rotation is set to target rotation when provided."""
        solver = TwoBoneIK(0, 1, 2)
        root, mid, end = create_arm_chain(upper_length=1.0, lower_length=1.0)

        target = Vec3(1.0, 1.0, 0)
        target_rot = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 2)
        result = solver.solve(root, mid, end, target, target_rotation=target_rot)

        assert result.success is True
        assert quat_approx_equal(result.end_rotation, target_rot)

    def test_solve_target_rotation_identity(self):
        """Test with identity target rotation."""
        solver = TwoBoneIK(0, 1, 2)
        root, mid, end = create_arm_chain(upper_length=1.0, lower_length=1.0)

        target = Vec3(1.0, 1.0, 0)
        target_rot = Quat.identity()
        result = solver.solve(root, mid, end, target, target_rotation=target_rot)

        assert result.success is True
        assert quat_approx_equal(result.end_rotation, Quat.identity())


# =============================================================================
# Test Soft IK Behavior
# =============================================================================

class TestSoftIKBehavior:
    """Tests for soft IK falloff functionality."""

    def test_soft_ik_disabled_by_default(self):
        """Test that soft IK is disabled with default parameters."""
        solver = TwoBoneIK(0, 1, 2)  # soft_ik_ratio = 0.0
        root, mid, end = create_arm_chain(upper_length=1.0, lower_length=1.0)

        # Target beyond reach
        target = Vec3(0, 2.5, 0)
        result = solver.solve(root, mid, end, target)

        assert result.success is True
        assert result.target_reached is False

    def test_soft_ik_enabled(self):
        """Test that soft IK softens reach limits when enabled."""
        solver = TwoBoneIK(0, 1, 2, soft_ik_ratio=0.9, soft_ik_blend=1.0)
        root, mid, end = create_arm_chain(upper_length=1.0, lower_length=1.0)

        # Target at 95% of max reach (in soft zone)
        target = Vec3(0, 1.9, 0)
        result = solver.solve(root, mid, end, target)

        assert result.success is True

    def test_soft_ik_ratio_boundary(self):
        """Test soft IK starts softening at specified ratio."""
        solver = TwoBoneIK(0, 1, 2, soft_ik_ratio=0.8, soft_ik_blend=1.0)
        root, mid, end = create_arm_chain(upper_length=1.0, lower_length=1.0)

        # Force cache
        solver.solve(root, mid, end, Vec3(0, 1, 0))

        # Below soft start (80% of 2.0 = 1.6)
        dist_below = 1.5
        result_below = solver._apply_soft_ik(dist_below)
        assert abs(result_below - dist_below) < MATH_EPSILON  # No softening

        # Above soft start - the soft IK asymptotically approaches max
        # For distances above soft_start, the result is bounded by total_length
        dist_above = 2.5  # Beyond max reach
        result_above = solver._apply_soft_ik(dist_above)
        # Result should be less than total_length and different from input
        assert result_above < solver._total_length
        assert result_above != dist_above

    def test_soft_ik_blend_zero(self):
        """Test that blend=0 disables soft IK effect."""
        solver = TwoBoneIK(0, 1, 2, soft_ik_ratio=0.8, soft_ik_blend=0.0)
        root, mid, end = create_arm_chain(upper_length=1.0, lower_length=1.0)

        solver.solve(root, mid, end, Vec3(0, 1, 0))  # Cache lengths

        dist = 1.9
        result = solver._apply_soft_ik(dist)
        assert abs(result - dist) < MATH_EPSILON  # No softening

    def test_soft_ik_blend_partial(self):
        """Test partial blend between hard and soft IK."""
        solver = TwoBoneIK(0, 1, 2, soft_ik_ratio=0.8, soft_ik_blend=0.5)
        root, mid, end = create_arm_chain(upper_length=1.0, lower_length=1.0)

        solver.solve(root, mid, end, Vec3(0, 1, 0))  # Cache lengths

        # Use a distance beyond max reach to see the effect
        dist = 2.5
        result = solver._apply_soft_ik(dist)

        # Result should be between hard (dist) and full soft
        full_soft_solver = TwoBoneIK(0, 1, 2, soft_ik_ratio=0.8, soft_ik_blend=1.0)
        full_soft_solver._total_length = 2.0
        full_soft_solver._lengths_cached = True
        full_soft = full_soft_solver._apply_soft_ik(dist)

        # With 0.5 blend: result = dist * 0.5 + soft_dist * 0.5
        # So result should be between the soft value and the original dist
        assert result >= full_soft
        assert result <= dist

    def test_apply_soft_ik_direct_method(self):
        """Test _apply_soft_ik internal method directly."""
        solver = TwoBoneIK(0, 1, 2, soft_ik_ratio=0.9, soft_ik_blend=1.0)
        solver._total_length = 2.0
        solver._lengths_cached = True

        # Below soft start
        result = solver._apply_soft_ik(1.5)
        assert abs(result - 1.5) < MATH_EPSILON

        # At soft start
        result = solver._apply_soft_ik(1.8)
        assert result == 1.8

        # Above soft start
        result = solver._apply_soft_ik(2.1)
        assert result < 2.1
        assert result >= 1.8

    def test_soft_ik_with_zero_total_length(self):
        """Test soft IK handles zero total length gracefully."""
        solver = TwoBoneIK(0, 1, 2, soft_ik_ratio=0.9, soft_ik_blend=1.0)
        solver._total_length = 0.0
        solver._lengths_cached = True

        result = solver._apply_soft_ik(1.0)
        assert result == 1.0  # Returns unchanged


# =============================================================================
# Test solve_with_pose()
# =============================================================================

class TestSolveWithPose:
    """Tests for solve_with_pose convenience method."""

    def test_solve_with_pose_basic(self):
        """Test solve_with_pose returns modified transforms."""
        solver = TwoBoneIK(0, 1, 2)
        transforms = [
            Transform(Vec3(0, 0, 0), Quat.identity()),
            Transform(Vec3(0, 1, 0), Quat.identity()),
            Transform(Vec3(0, 2, 0), Quat.identity()),
        ]

        target = Vec3(1.0, 1.0, 0)
        result_transforms = solver.solve_with_pose(transforms, target)

        assert len(result_transforms) == 3
        # Transforms should be different from input
        assert result_transforms is not transforms

    def test_solve_with_pose_preserves_extra_bones(self):
        """Test that extra bones in list are preserved."""
        solver = TwoBoneIK(0, 1, 2)
        transforms = [
            Transform(Vec3(0, 0, 0), Quat.identity()),
            Transform(Vec3(0, 1, 0), Quat.identity()),
            Transform(Vec3(0, 2, 0), Quat.identity()),
            Transform(Vec3(0, 3, 0), Quat.identity()),  # Extra bone
            Transform(Vec3(0, 4, 0), Quat.identity()),  # Extra bone
        ]

        target = Vec3(1.0, 1.0, 0)
        result_transforms = solver.solve_with_pose(transforms, target)

        assert len(result_transforms) == 5

    def test_solve_with_pose_with_pole_vector(self):
        """Test solve_with_pose with pole vector."""
        solver = TwoBoneIK(0, 1, 2)
        transforms = [
            Transform(Vec3(0, 0, 0), Quat.identity()),
            Transform(Vec3(0, 1, 0), Quat.identity()),
            Transform(Vec3(0, 2, 0), Quat.identity()),
        ]

        target = Vec3(1.0, 1.0, 0)
        pole = Vec3(0, 1, 1)
        result_transforms = solver.solve_with_pose(transforms, target, pole_vector=pole)

        assert len(result_transforms) == 3

    def test_solve_with_pose_raises_on_insufficient_bones(self):
        """Test that insufficient bones raises ValueError."""
        solver = TwoBoneIK(0, 1, 5)  # end_bone = 5
        transforms = [
            Transform(Vec3(0, 0, 0), Quat.identity()),
            Transform(Vec3(0, 1, 0), Quat.identity()),
            Transform(Vec3(0, 2, 0), Quat.identity()),
        ]  # Only 3 bones, need 6

        with pytest.raises(ValueError, match="Not enough bones"):
            solver.solve_with_pose(transforms, Vec3(0, 1, 0))

    def test_solve_with_pose_failed_solve_returns_original(self):
        """Test that failed solve returns original transforms."""
        solver = TwoBoneIK(0, 1, 2)
        # Create degenerate chain
        transforms = [
            Transform(Vec3(0, 0, 0), Quat.identity()),
            Transform(Vec3(0, 0, 0), Quat.identity()),  # Zero-length upper
            Transform(Vec3(0, 1, 0), Quat.identity()),
        ]

        target = Vec3(0, 0.5, 0)
        result = solver.solve_with_pose(transforms, target)

        # Should return same object (unchanged) when solve fails
        assert result == transforms


# =============================================================================
# Test _rotation_between_vectors()
# =============================================================================

class TestRotationBetweenVectors:
    """Tests for _rotation_between_vectors helper method."""

    def test_rotation_between_same_vectors(self):
        """Test rotation between identical vectors is identity."""
        solver = TwoBoneIK(0, 1, 2)

        v = Vec3(0, 1, 0).normalized()
        rot = solver._rotation_between_vectors(v, v)

        assert quat_approx_equal(rot, Quat.identity())

    def test_rotation_between_parallel_vectors(self):
        """Test rotation between parallel vectors is identity."""
        solver = TwoBoneIK(0, 1, 2)

        v1 = Vec3(0, 1, 0).normalized()
        v2 = Vec3(0, 2, 0).normalized()  # Same direction
        rot = solver._rotation_between_vectors(v1, v2)

        assert quat_approx_equal(rot, Quat.identity())

    def test_rotation_between_opposite_vectors(self):
        """Test rotation between opposite vectors is 180 degrees."""
        solver = TwoBoneIK(0, 1, 2)

        v1 = Vec3(0, 1, 0).normalized()
        v2 = Vec3(0, -1, 0).normalized()
        rot = solver._rotation_between_vectors(v1, v2)

        # Apply rotation to v1, should get v2
        rotated = rot.rotate_vector(v1)
        assert vec3_approx_equal(rotated, v2, eps=1e-4)

    def test_rotation_between_perpendicular_vectors(self):
        """Test rotation between perpendicular vectors is 90 degrees."""
        solver = TwoBoneIK(0, 1, 2)

        v1 = Vec3(1, 0, 0).normalized()
        v2 = Vec3(0, 1, 0).normalized()
        rot = solver._rotation_between_vectors(v1, v2)

        # Apply rotation to v1, should get v2
        rotated = rot.rotate_vector(v1)
        assert vec3_approx_equal(rotated, v2, eps=1e-4)

    def test_rotation_between_arbitrary_vectors(self):
        """Test rotation between arbitrary vectors."""
        solver = TwoBoneIK(0, 1, 2)

        v1 = Vec3(1, 1, 0).normalized()
        v2 = Vec3(0, 1, 1).normalized()
        rot = solver._rotation_between_vectors(v1, v2)

        # Apply rotation to v1, should get v2
        rotated = rot.rotate_vector(v1)
        assert vec3_approx_equal(rotated, v2, eps=1e-4)


# =============================================================================
# Test TwoBoneIKConstraint
# =============================================================================

class TestTwoBoneIKConstraint:
    """Tests for TwoBoneIKConstraint wrapper."""

    def test_constraint_construction(self):
        """Test constraint construction with default values."""
        solver = TwoBoneIK(0, 1, 2)
        constraint = TwoBoneIKConstraint(solver)

        assert constraint.solver is solver
        assert constraint.min_bend_angle == JOINT_MIN_BEND_ANGLE
        assert constraint.max_bend_angle == JOINT_MAX_BEND_ANGLE
        assert vec3_approx_equal(constraint.twist_axis, Vec3.unit_y())
        assert constraint.min_twist == -math.pi
        assert constraint.max_twist == math.pi

    def test_constraint_custom_values(self):
        """Test constraint with custom values."""
        solver = TwoBoneIK(0, 1, 2)
        constraint = TwoBoneIKConstraint(
            solver,
            min_bend_angle=0.2,
            max_bend_angle=2.5,
            twist_axis=Vec3(1, 0, 0),
            min_twist=-1.0,
            max_twist=1.0
        )

        assert constraint.min_bend_angle == 0.2
        assert constraint.max_bend_angle == 2.5
        assert vec3_approx_equal(constraint.twist_axis, Vec3(1, 0, 0).normalized())
        assert constraint.min_twist == -1.0
        assert constraint.max_twist == 1.0

    def test_apply_constraints_passthrough(self):
        """Test that apply_constraints passes through result."""
        solver = TwoBoneIK(0, 1, 2)
        constraint = TwoBoneIKConstraint(solver)

        result = TwoBoneIKResult(
            success=True,
            extension_ratio=0.5,
            target_reached=True
        )

        constrained = constraint.apply_constraints(result)

        assert constrained.success is True

    def test_apply_constraints_failed_result_passthrough(self):
        """Test that failed results pass through unchanged."""
        solver = TwoBoneIK(0, 1, 2)
        constraint = TwoBoneIKConstraint(solver)

        result = TwoBoneIKResult(success=False)
        constrained = constraint.apply_constraints(result)

        assert constrained.success is False


# =============================================================================
# Test Extension Ratio Calculation
# =============================================================================

class TestExtensionRatioCalculation:
    """Tests for extension_ratio calculation in results."""

    def test_extension_ratio_at_max_reach(self):
        """Test extension ratio is ~1.0 at max reach."""
        solver = TwoBoneIK(0, 1, 2)
        root, mid, end = create_arm_chain(upper_length=1.0, lower_length=1.0)

        # Target at max reach
        target = Vec3(0, 2.0, 0)
        result = solver.solve(root, mid, end, target)

        assert result.extension_ratio > 0.95

    def test_extension_ratio_at_min_reach(self):
        """Test extension ratio is ~0.0 at min reach."""
        solver = TwoBoneIK(0, 1, 2)
        root, mid, end = create_arm_chain(upper_length=2.0, lower_length=1.0)

        # Target at min reach (1.0 for these bone lengths)
        target = Vec3(0, 1.0, 0)
        result = solver.solve(root, mid, end, target)

        assert result.extension_ratio < 0.1

    def test_extension_ratio_at_mid_reach(self):
        """Test extension ratio is ~0.5 at mid reach."""
        solver = TwoBoneIK(0, 1, 2)
        root, mid, end = create_arm_chain(upper_length=1.0, lower_length=1.0)

        # Mid reach = (0 + 2) / 2 = 1.0
        target = Vec3(0, 1.0, 0)
        result = solver.solve(root, mid, end, target)

        assert 0.4 < result.extension_ratio < 0.6

    def test_extension_ratio_with_equal_bones(self):
        """Test extension ratio with equal bone lengths (min=0)."""
        solver = TwoBoneIK(0, 1, 2)
        root, mid, end = create_arm_chain(upper_length=1.0, lower_length=1.0)

        # With equal bones, min_dist = 0, max_dist = 2
        # At distance 1.0, ratio should be 0.5
        target = Vec3(0, 1.0, 0)
        result = solver.solve(root, mid, end, target)

        assert abs(result.extension_ratio - 0.5) < 0.1


# =============================================================================
# Test Chain Position Updates
# =============================================================================

class TestChainPositionUpdates:
    """Tests for _update_chain_positions method."""

    def test_update_chain_positions(self):
        """Test that chain positions are updated after rotation changes."""
        solver = TwoBoneIK(0, 1, 2)

        # Cache lengths first
        root, mid, end = create_arm_chain(upper_length=1.0, lower_length=1.0)
        solver.solve(root, mid, end, Vec3(0, 1, 0))

        # Create transforms with rotations
        transforms = [
            Transform(Vec3(0, 0, 0), Quat.identity()),
            Transform(Vec3(0, 1, 0), Quat.identity()),
            Transform(Vec3(0, 2, 0), Quat.identity()),
        ]

        # Apply a rotation and update
        transforms[0].rotation = Quat.from_axis_angle(Vec3(0, 0, 1), math.pi / 4)
        solver._update_chain_positions(transforms)

        # Mid position should have moved
        assert not vec3_approx_equal(transforms[1].translation, Vec3(0, 1, 0))


# =============================================================================
# Integration Tests
# =============================================================================

class TestTwoBoneIKIntegration:
    """Integration tests for complete IK solve scenarios."""

    def test_arm_reaching_forward(self):
        """Test arm IK reaching forward."""
        solver = TwoBoneIK(0, 1, 2)
        root, mid, end = create_arm_chain(upper_length=0.3, lower_length=0.25)

        # Reach forward
        target = Vec3(0.4, 0.2, 0)
        result = solver.solve(root, mid, end, target)

        assert result.success is True
        assert result.target_reached is True

    def test_leg_ik_ground_contact(self):
        """Test leg IK for ground contact."""
        solver = TwoBoneIK(0, 1, 2)
        # Typical leg proportions
        root, mid, end = create_arm_chain(
            root_pos=Vec3(0, 1.0, 0),  # Hip at height 1
            upper_length=0.5,  # Thigh
            lower_length=0.5   # Shin
        )

        # Foot target on ground
        target = Vec3(0.2, 0.0, 0.1)
        pole = Vec3(0, 1, 1)  # Knee forward
        result = solver.solve(root, mid, end, target, pole_vector=pole)

        assert result.success is True

    def test_multiple_solves_same_solver(self):
        """Test multiple consecutive solves with same solver."""
        solver = TwoBoneIK(0, 1, 2)
        root, mid, end = create_arm_chain(upper_length=1.0, lower_length=1.0)

        targets = [
            Vec3(0, 1.5, 0),
            Vec3(1.0, 1.0, 0),
            Vec3(-1.0, 0.5, 0),
            Vec3(0, 0.5, 1.0),
        ]

        for target in targets:
            result = solver.solve(root, mid, end, target)
            assert result.success is True

    def test_varying_bone_positions_same_solver(self):
        """Test same solver with varying bone positions."""
        solver = TwoBoneIK(0, 1, 2)

        # First solve caches lengths
        root1, mid1, end1 = create_arm_chain(upper_length=1.0, lower_length=1.0)
        result1 = solver.solve(root1, mid1, end1, Vec3(0, 1.5, 0))
        assert result1.success is True

        # Reset and solve with different lengths
        solver.reset_cached_lengths()
        root2, mid2, end2 = create_arm_chain(upper_length=2.0, lower_length=1.5)
        result2 = solver.solve(root2, mid2, end2, Vec3(0, 3.0, 0))
        assert result2.success is True

        # Verify new lengths were cached
        assert abs(solver._upper_length - 2.0) < MATH_EPSILON
        assert abs(solver._lower_length - 1.5) < MATH_EPSILON


# =============================================================================
# Performance/Stress Tests
# =============================================================================

class TestTwoBoneIKPerformance:
    """Performance and stress tests."""

    def test_many_consecutive_solves(self):
        """Test many consecutive solves for stability."""
        solver = TwoBoneIK(0, 1, 2)
        root, mid, end = create_arm_chain(upper_length=1.0, lower_length=1.0)

        for i in range(100):
            angle = (i / 100) * 2 * math.pi
            radius = 1.5
            target = Vec3(
                radius * math.cos(angle),
                radius * math.sin(angle),
                0
            )
            result = solver.solve(root, mid, end, target)
            assert result.success is True

    def test_extreme_target_distances(self):
        """Test with extreme target distances."""
        solver = TwoBoneIK(0, 1, 2)
        root, mid, end = create_arm_chain(upper_length=1.0, lower_length=1.0)

        # Very far
        result = solver.solve(root, mid, end, Vec3(0, 1000000, 0))
        assert result.success is True

        # Very close
        result = solver.solve(root, mid, end, Vec3(0, 0.00001, 0))
        assert result.success is True
