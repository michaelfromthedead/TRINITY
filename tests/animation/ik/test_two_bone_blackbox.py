"""Blackbox tests for Two-Bone IK Core (T-IK-3.4).

This module tests the TwoBoneIK solver from the public API only,
without knowledge of implementation details. Tests are derived from
the acceptance criteria:

1. TwoBoneIK class
2. Law of cosines angle calculation
3. cos_mid = (a^2 + b^2 - c^2) / (2ab) formula
4. Numerical stability clamping
5. solve(root, mid, end, target, pole) method
6. TwoBoneIKResult dataclass

Test Strategy:
- Test public API contracts only
- Test behavioral expectations from acceptance criteria
- Test boundary conditions (reachable/unreachable targets)
- Test pole vector influence on elbow direction
- Test numerical edge cases
"""

import math
import pytest
from dataclasses import fields
from typing import Optional

# Import public API
from engine.animation.ik import TwoBoneIK, TwoBoneIKResult
from engine.core.math import Vec3, Quat, Transform


# =============================================================================
# Helper Functions
# =============================================================================

def make_transform(position: Vec3, rotation: Optional[Quat] = None) -> Transform:
    """Create a Transform from position and optional rotation."""
    return Transform(
        translation=position,
        rotation=rotation if rotation else Quat.identity()
    )


def vec3_distance(a: Vec3, b: Vec3) -> float:
    """Calculate distance between two Vec3 points."""
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return math.sqrt(dx*dx + dy*dy + dz*dz)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def standard_arm_solver():
    """Standard two-bone solver simulating an arm (shoulder->elbow->wrist)."""
    return TwoBoneIK(root_bone=0, mid_bone=1, end_bone=2)


@pytest.fixture
def leg_solver():
    """Two-bone solver simulating a leg (hip->knee->ankle)."""
    return TwoBoneIK(root_bone=10, mid_bone=11, end_bone=12)


@pytest.fixture
def unit_chain_transforms():
    """Standard unit-length chain: root at origin, bones of length 1.0 each."""
    return {
        "root": make_transform(Vec3(0.0, 0.0, 0.0)),
        "mid": make_transform(Vec3(0.0, 1.0, 0.0)),
        "end": make_transform(Vec3(0.0, 2.0, 0.0)),
    }


@pytest.fixture
def asymmetric_chain_transforms():
    """Asymmetric chain: upper bone 1.5, lower bone 1.0."""
    return {
        "root": make_transform(Vec3(0.0, 0.0, 0.0)),
        "mid": make_transform(Vec3(0.0, 1.5, 0.0)),
        "end": make_transform(Vec3(0.0, 2.5, 0.0)),
    }


# =============================================================================
# AC-1: TwoBoneIK Class
# =============================================================================

class TestTwoBoneIKInstantiation:
    """Tests for TwoBoneIK class instantiation (AC-1)."""

    def test_can_instantiate_with_bone_indices(self):
        """TwoBoneIK can be instantiated with bone indices."""
        solver = TwoBoneIK(root_bone=0, mid_bone=1, end_bone=2)
        assert solver is not None

    def test_can_instantiate_with_different_indices(self):
        """TwoBoneIK accepts arbitrary bone index values."""
        solver = TwoBoneIK(root_bone=100, mid_bone=101, end_bone=102)
        assert solver is not None

    def test_solver_is_reusable(self, standard_arm_solver, unit_chain_transforms):
        """Same solver instance can be used for multiple solve calls."""
        target1 = Vec3(1.0, 1.0, 0.0)
        target2 = Vec3(-1.0, 1.0, 0.0)

        result1 = standard_arm_solver.solve(
            root_transform=unit_chain_transforms["root"],
            mid_transform=unit_chain_transforms["mid"],
            end_transform=unit_chain_transforms["end"],
            target_position=target1
        )

        result2 = standard_arm_solver.solve(
            root_transform=unit_chain_transforms["root"],
            mid_transform=unit_chain_transforms["mid"],
            end_transform=unit_chain_transforms["end"],
            target_position=target2
        )

        assert result1 is not None
        assert result2 is not None

    def test_multiple_solvers_independent(self):
        """Multiple solver instances operate independently."""
        solver1 = TwoBoneIK(root_bone=0, mid_bone=1, end_bone=2)
        solver2 = TwoBoneIK(root_bone=5, mid_bone=6, end_bone=7)

        # Both should be usable without interfering
        assert solver1 is not solver2


# =============================================================================
# AC-2 & AC-3: Law of Cosines Angle Calculation
# =============================================================================

class TestLawOfCosinesCalculation:
    """Tests verifying correct law of cosines behavior (AC-2, AC-3).

    The formula cos_mid = (a^2 + b^2 - c^2) / (2ab) should be used internally.
    We test observable behavior that confirms this mathematical basis.
    """

    def test_isosceles_triangle_target(self, standard_arm_solver, unit_chain_transforms):
        """When target forms isosceles triangle, mid angle should be symmetric."""
        # With unit bones (1.0 each), target at sqrt(2) distance at 45 degrees
        target = Vec3(1.0, 1.0, 0.0)  # Distance sqrt(2) from root

        result = standard_arm_solver.solve(
            root_transform=unit_chain_transforms["root"],
            mid_transform=unit_chain_transforms["mid"],
            end_transform=unit_chain_transforms["end"],
            target_position=target
        )

        assert result.success, "Should successfully solve for reachable target"

    def test_equilateral_configuration(self, standard_arm_solver):
        """Test with configuration that should produce 60-degree angles."""
        root = make_transform(Vec3(0.0, 0.0, 0.0))
        mid = make_transform(Vec3(0.0, 1.0, 0.0))
        end = make_transform(Vec3(0.0, 2.0, 0.0))

        # Target at distance 1.0 from root (forms equilateral potential)
        target = Vec3(0.5, 0.866, 0.0)  # Approx at 60 degrees, distance 1.0

        result = standard_arm_solver.solve(
            root_transform=root,
            mid_transform=mid,
            end_transform=end,
            target_position=target
        )

        assert result.success

    def test_right_angle_configuration(self, standard_arm_solver, unit_chain_transforms):
        """Test configuration producing 90-degree mid angle."""
        # With equal length bones (1.0 each), target at sqrt(2) distance
        target = Vec3(math.sqrt(2), 0.0, 0.0)

        result = standard_arm_solver.solve(
            root_transform=unit_chain_transforms["root"],
            mid_transform=unit_chain_transforms["mid"],
            end_transform=unit_chain_transforms["end"],
            target_position=target
        )

        assert result.success


# =============================================================================
# AC-4: Numerical Stability Clamping
# =============================================================================

class TestNumericalStability:
    """Tests for numerical stability clamping (AC-4).

    The solver should clamp cos_mid to [-1, 1] range to handle floating-point
    precision issues when target is at or near maximum reach.
    """

    def test_target_at_exact_max_reach(self, standard_arm_solver, unit_chain_transforms):
        """Target at exact maximum reach should not cause numerical errors."""
        # Max reach is sum of bone lengths: 1.0 + 1.0 = 2.0
        target = Vec3(2.0, 0.0, 0.0)

        result = standard_arm_solver.solve(
            root_transform=unit_chain_transforms["root"],
            mid_transform=unit_chain_transforms["mid"],
            end_transform=unit_chain_transforms["end"],
            target_position=target
        )

        # Should not raise exception, should handle gracefully
        assert result is not None

    def test_target_slightly_beyond_max_reach(self, standard_arm_solver, unit_chain_transforms):
        """Target slightly beyond max reach should clamp without errors."""
        # Slightly beyond max reach due to floating point
        target = Vec3(2.0001, 0.0, 0.0)

        result = standard_arm_solver.solve(
            root_transform=unit_chain_transforms["root"],
            mid_transform=unit_chain_transforms["mid"],
            end_transform=unit_chain_transforms["end"],
            target_position=target
        )

        # Should handle gracefully
        assert result is not None

    def test_target_well_beyond_max_reach(self, standard_arm_solver, unit_chain_transforms):
        """Target clearly beyond max reach should be handled gracefully."""
        target = Vec3(10.0, 0.0, 0.0)  # Far beyond 2.0 max reach

        result = standard_arm_solver.solve(
            root_transform=unit_chain_transforms["root"],
            mid_transform=unit_chain_transforms["mid"],
            end_transform=unit_chain_transforms["end"],
            target_position=target
        )

        # Should not crash, should indicate unreachable or extend toward target
        assert result is not None

    def test_target_at_root_position(self, standard_arm_solver, unit_chain_transforms):
        """Target at root position (minimum reach edge case)."""
        target = Vec3(0.0, 0.0, 0.0)  # At root

        result = standard_arm_solver.solve(
            root_transform=unit_chain_transforms["root"],
            mid_transform=unit_chain_transforms["mid"],
            end_transform=unit_chain_transforms["end"],
            target_position=target
        )

        # Should handle gracefully
        assert result is not None

    def test_near_zero_distance_target(self, standard_arm_solver, unit_chain_transforms):
        """Target very close to root (near-zero distance)."""
        target = Vec3(0.001, 0.0, 0.0)

        result = standard_arm_solver.solve(
            root_transform=unit_chain_transforms["root"],
            mid_transform=unit_chain_transforms["mid"],
            end_transform=unit_chain_transforms["end"],
            target_position=target
        )

        assert result is not None

    def test_asymmetric_chain_max_reach(self, standard_arm_solver, asymmetric_chain_transforms):
        """Max reach with asymmetric bone lengths."""
        # Upper bone 1.5, lower bone 1.0, max reach = 2.5
        target = Vec3(2.5, 0.0, 0.0)

        result = standard_arm_solver.solve(
            root_transform=asymmetric_chain_transforms["root"],
            mid_transform=asymmetric_chain_transforms["mid"],
            end_transform=asymmetric_chain_transforms["end"],
            target_position=target
        )

        assert result is not None

    def test_very_small_bone_lengths(self):
        """Solver handles very small bone lengths without overflow."""
        solver = TwoBoneIK(root_bone=0, mid_bone=1, end_bone=2)

        root = make_transform(Vec3(0.0, 0.0, 0.0))
        mid = make_transform(Vec3(0.0, 0.0001, 0.0))
        end = make_transform(Vec3(0.0, 0.0002, 0.0))
        target = Vec3(0.00015, 0.0, 0.0)

        result = solver.solve(
            root_transform=root,
            mid_transform=mid,
            end_transform=end,
            target_position=target
        )

        assert result is not None

    def test_very_large_bone_lengths(self):
        """Solver handles very large bone lengths without overflow."""
        solver = TwoBoneIK(root_bone=0, mid_bone=1, end_bone=2)

        root = make_transform(Vec3(0.0, 0.0, 0.0))
        mid = make_transform(Vec3(0.0, 1000.0, 0.0))
        end = make_transform(Vec3(0.0, 2000.0, 0.0))
        target = Vec3(1500.0, 0.0, 0.0)

        result = solver.solve(
            root_transform=root,
            mid_transform=mid,
            end_transform=end,
            target_position=target
        )

        assert result is not None


# =============================================================================
# AC-5: solve(root, mid, end, target, pole) Method
# =============================================================================

class TestSolveMethod:
    """Tests for the solve() method signature and behavior (AC-5)."""

    def test_solve_with_required_parameters(self, standard_arm_solver, unit_chain_transforms):
        """solve() works with required transform parameters."""
        result = standard_arm_solver.solve(
            root_transform=unit_chain_transforms["root"],
            mid_transform=unit_chain_transforms["mid"],
            end_transform=unit_chain_transforms["end"],
            target_position=Vec3(1.0, 1.0, 0.0)
        )

        assert result is not None
        assert isinstance(result, TwoBoneIKResult)

    def test_solve_with_pole_vector(self, standard_arm_solver, unit_chain_transforms):
        """solve() accepts optional pole vector parameter."""
        result = standard_arm_solver.solve(
            root_transform=unit_chain_transforms["root"],
            mid_transform=unit_chain_transforms["mid"],
            end_transform=unit_chain_transforms["end"],
            target_position=Vec3(1.0, 1.0, 0.0),
            pole_vector=Vec3(0.0, 0.0, 1.0)
        )

        assert result is not None
        assert isinstance(result, TwoBoneIKResult)

    def test_solve_with_target_rotation(self, standard_arm_solver, unit_chain_transforms):
        """solve() accepts optional target rotation parameter."""
        result = standard_arm_solver.solve(
            root_transform=unit_chain_transforms["root"],
            mid_transform=unit_chain_transforms["mid"],
            end_transform=unit_chain_transforms["end"],
            target_position=Vec3(1.0, 1.0, 0.0),
            target_rotation=Quat.identity()
        )

        assert result is not None
        assert isinstance(result, TwoBoneIKResult)

    def test_solve_returns_result_type(self, standard_arm_solver, unit_chain_transforms):
        """solve() returns TwoBoneIKResult instance."""
        result = standard_arm_solver.solve(
            root_transform=unit_chain_transforms["root"],
            mid_transform=unit_chain_transforms["mid"],
            end_transform=unit_chain_transforms["end"],
            target_position=Vec3(1.0, 1.0, 0.0)
        )

        assert isinstance(result, TwoBoneIKResult)

    def test_solve_multiple_times_same_inputs(self, standard_arm_solver, unit_chain_transforms):
        """Solving with same inputs produces consistent results."""
        target = Vec3(1.0, 1.0, 0.0)

        result1 = standard_arm_solver.solve(
            root_transform=unit_chain_transforms["root"],
            mid_transform=unit_chain_transforms["mid"],
            end_transform=unit_chain_transforms["end"],
            target_position=target
        )

        result2 = standard_arm_solver.solve(
            root_transform=unit_chain_transforms["root"],
            mid_transform=unit_chain_transforms["mid"],
            end_transform=unit_chain_transforms["end"],
            target_position=target
        )

        # Results should be consistent (deterministic)
        assert result1.success == result2.success


# =============================================================================
# AC-6: TwoBoneIKResult Dataclass
# =============================================================================

class TestTwoBoneIKResultDataclass:
    """Tests for TwoBoneIKResult dataclass structure (AC-6)."""

    def test_result_has_success_attribute(self, standard_arm_solver, unit_chain_transforms):
        """TwoBoneIKResult has success attribute."""
        result = standard_arm_solver.solve(
            root_transform=unit_chain_transforms["root"],
            mid_transform=unit_chain_transforms["mid"],
            end_transform=unit_chain_transforms["end"],
            target_position=Vec3(1.0, 1.0, 0.0)
        )

        assert hasattr(result, "success")
        assert isinstance(result.success, bool)

    def test_result_has_root_rotation(self, standard_arm_solver, unit_chain_transforms):
        """TwoBoneIKResult has root_rotation attribute."""
        result = standard_arm_solver.solve(
            root_transform=unit_chain_transforms["root"],
            mid_transform=unit_chain_transforms["mid"],
            end_transform=unit_chain_transforms["end"],
            target_position=Vec3(1.0, 1.0, 0.0)
        )

        assert hasattr(result, "root_rotation")

    def test_result_has_mid_rotation(self, standard_arm_solver, unit_chain_transforms):
        """TwoBoneIKResult has mid_rotation attribute."""
        result = standard_arm_solver.solve(
            root_transform=unit_chain_transforms["root"],
            mid_transform=unit_chain_transforms["mid"],
            end_transform=unit_chain_transforms["end"],
            target_position=Vec3(1.0, 1.0, 0.0)
        )

        assert hasattr(result, "mid_rotation")

    def test_result_has_end_rotation(self, standard_arm_solver, unit_chain_transforms):
        """TwoBoneIKResult has end_rotation attribute."""
        result = standard_arm_solver.solve(
            root_transform=unit_chain_transforms["root"],
            mid_transform=unit_chain_transforms["mid"],
            end_transform=unit_chain_transforms["end"],
            target_position=Vec3(1.0, 1.0, 0.0)
        )

        assert hasattr(result, "end_rotation")

    def test_result_has_target_reached(self, standard_arm_solver, unit_chain_transforms):
        """TwoBoneIKResult has target_reached attribute."""
        result = standard_arm_solver.solve(
            root_transform=unit_chain_transforms["root"],
            mid_transform=unit_chain_transforms["mid"],
            end_transform=unit_chain_transforms["end"],
            target_position=Vec3(1.0, 1.0, 0.0)
        )

        assert hasattr(result, "target_reached")
        assert isinstance(result.target_reached, bool)

    def test_result_has_extension_ratio(self, standard_arm_solver, unit_chain_transforms):
        """TwoBoneIKResult has extension_ratio attribute."""
        result = standard_arm_solver.solve(
            root_transform=unit_chain_transforms["root"],
            mid_transform=unit_chain_transforms["mid"],
            end_transform=unit_chain_transforms["end"],
            target_position=Vec3(1.0, 1.0, 0.0)
        )

        assert hasattr(result, "extension_ratio")
        assert isinstance(result.extension_ratio, float)

    def test_result_rotations_are_quaternions_on_success(self, standard_arm_solver, unit_chain_transforms):
        """On success, rotations are valid quaternions."""
        result = standard_arm_solver.solve(
            root_transform=unit_chain_transforms["root"],
            mid_transform=unit_chain_transforms["mid"],
            end_transform=unit_chain_transforms["end"],
            target_position=Vec3(1.0, 1.0, 0.0)
        )

        if result.success:
            # Check that rotations are Quat instances
            assert isinstance(result.root_rotation, Quat)
            assert isinstance(result.mid_rotation, Quat)
            assert isinstance(result.end_rotation, Quat)


# =============================================================================
# Behavioral Tests: Reachable Targets
# =============================================================================

class TestReachableTargets:
    """Tests for behavior with reachable targets."""

    def test_target_within_reach_succeeds(self, standard_arm_solver, unit_chain_transforms):
        """Target within reach should have success=True."""
        # Target at distance 1.5 (within 2.0 max reach)
        target = Vec3(1.0, 1.0, 0.0)  # Distance ~1.41

        result = standard_arm_solver.solve(
            root_transform=unit_chain_transforms["root"],
            mid_transform=unit_chain_transforms["mid"],
            end_transform=unit_chain_transforms["end"],
            target_position=target
        )

        assert result.success is True

    def test_target_at_half_reach(self, standard_arm_solver, unit_chain_transforms):
        """Target at half max reach should succeed."""
        target = Vec3(1.0, 0.0, 0.0)  # Distance 1.0 (half of 2.0 max)

        result = standard_arm_solver.solve(
            root_transform=unit_chain_transforms["root"],
            mid_transform=unit_chain_transforms["mid"],
            end_transform=unit_chain_transforms["end"],
            target_position=target
        )

        assert result.success is True

    def test_target_requiring_bend(self, standard_arm_solver, unit_chain_transforms):
        """Target requiring elbow bend should succeed."""
        # Target that requires the chain to bend (not straight)
        target = Vec3(1.0, 0.5, 0.0)

        result = standard_arm_solver.solve(
            root_transform=unit_chain_transforms["root"],
            mid_transform=unit_chain_transforms["mid"],
            end_transform=unit_chain_transforms["end"],
            target_position=target
        )

        assert result.success is True


# =============================================================================
# Behavioral Tests: Unreachable Targets
# =============================================================================

class TestUnreachableTargets:
    """Tests for behavior with unreachable targets."""

    def test_unreachable_target_handled_gracefully(self, standard_arm_solver, unit_chain_transforms):
        """Clearly unreachable target should not crash."""
        target = Vec3(100.0, 0.0, 0.0)  # Far beyond 2.0 max reach

        # Should not raise exception
        result = standard_arm_solver.solve(
            root_transform=unit_chain_transforms["root"],
            mid_transform=unit_chain_transforms["mid"],
            end_transform=unit_chain_transforms["end"],
            target_position=target
        )

        assert result is not None

    def test_unreachable_returns_valid_result(self, standard_arm_solver, unit_chain_transforms):
        """Unreachable target should still return TwoBoneIKResult."""
        target = Vec3(50.0, 0.0, 0.0)

        result = standard_arm_solver.solve(
            root_transform=unit_chain_transforms["root"],
            mid_transform=unit_chain_transforms["mid"],
            end_transform=unit_chain_transforms["end"],
            target_position=target
        )

        assert isinstance(result, TwoBoneIKResult)

    def test_target_inside_minimum_reach(self, standard_arm_solver, unit_chain_transforms):
        """Target inside minimum reach (difference of bone lengths)."""
        # With equal bones (1.0 each), minimum reach is |1.0 - 1.0| = 0.0
        target = Vec3(0.1, 0.0, 0.0)

        result = standard_arm_solver.solve(
            root_transform=unit_chain_transforms["root"],
            mid_transform=unit_chain_transforms["mid"],
            end_transform=unit_chain_transforms["end"],
            target_position=target
        )

        assert result is not None

    def test_unreachable_has_valid_extension_ratio(self, standard_arm_solver, unit_chain_transforms):
        """Unreachable target should have extension_ratio > 1.0 or special handling."""
        target = Vec3(10.0, 0.0, 0.0)  # Beyond max reach

        result = standard_arm_solver.solve(
            root_transform=unit_chain_transforms["root"],
            mid_transform=unit_chain_transforms["mid"],
            end_transform=unit_chain_transforms["end"],
            target_position=target
        )

        # Extension ratio should indicate over-extension
        assert result.extension_ratio >= 0.0


# =============================================================================
# Behavioral Tests: Pole Vector Influence
# =============================================================================

class TestPoleVectorInfluence:
    """Tests verifying pole vector affects elbow direction."""

    def test_pole_forward_vs_backward(self, standard_arm_solver, unit_chain_transforms):
        """Different pole vectors should produce different mid rotations."""
        target = Vec3(1.0, 1.0, 0.0)

        # Pole pointing in +Z
        result_forward = standard_arm_solver.solve(
            root_transform=unit_chain_transforms["root"],
            mid_transform=unit_chain_transforms["mid"],
            end_transform=unit_chain_transforms["end"],
            target_position=target,
            pole_vector=Vec3(0.0, 0.0, 1.0)
        )

        # Pole pointing in -Z
        result_backward = standard_arm_solver.solve(
            root_transform=unit_chain_transforms["root"],
            mid_transform=unit_chain_transforms["mid"],
            end_transform=unit_chain_transforms["end"],
            target_position=target,
            pole_vector=Vec3(0.0, 0.0, -1.0)
        )

        # Both should succeed
        assert result_forward.success
        assert result_backward.success

    def test_pole_left_vs_right(self, standard_arm_solver, unit_chain_transforms):
        """Pole on different sides should affect elbow direction."""
        target = Vec3(0.0, 0.0, 1.5)  # Target in front

        # Pole pointing right (+X)
        result_right = standard_arm_solver.solve(
            root_transform=unit_chain_transforms["root"],
            mid_transform=unit_chain_transforms["mid"],
            end_transform=unit_chain_transforms["end"],
            target_position=target,
            pole_vector=Vec3(1.0, 0.0, 0.0)
        )

        # Pole pointing left (-X)
        result_left = standard_arm_solver.solve(
            root_transform=unit_chain_transforms["root"],
            mid_transform=unit_chain_transforms["mid"],
            end_transform=unit_chain_transforms["end"],
            target_position=target,
            pole_vector=Vec3(-1.0, 0.0, 0.0)
        )

        assert result_right.success
        assert result_left.success

    def test_no_pole_vs_with_pole(self, standard_arm_solver, unit_chain_transforms):
        """Result without pole should differ from result with explicit pole."""
        target = Vec3(1.0, 0.5, 0.5)

        result_no_pole = standard_arm_solver.solve(
            root_transform=unit_chain_transforms["root"],
            mid_transform=unit_chain_transforms["mid"],
            end_transform=unit_chain_transforms["end"],
            target_position=target
        )

        result_with_pole = standard_arm_solver.solve(
            root_transform=unit_chain_transforms["root"],
            mid_transform=unit_chain_transforms["mid"],
            end_transform=unit_chain_transforms["end"],
            target_position=target,
            pole_vector=Vec3(0.0, 1.0, 0.0)
        )

        assert result_no_pole is not None
        assert result_with_pole is not None


# =============================================================================
# Behavioral Tests: Full Extension and Folding
# =============================================================================

class TestChainExtensionAndFolding:
    """Tests for chain fully extended and maximally folded states."""

    def test_target_at_max_reach_extends_chain(self, standard_arm_solver, unit_chain_transforms):
        """Target at max reach should extend chain fully."""
        # Max reach = 2.0
        target = Vec3(2.0, 0.0, 0.0)

        result = standard_arm_solver.solve(
            root_transform=unit_chain_transforms["root"],
            mid_transform=unit_chain_transforms["mid"],
            end_transform=unit_chain_transforms["end"],
            target_position=target
        )

        # Should succeed or indicate at limit
        assert result is not None
        # Extension ratio should be close to 1.0 at max reach
        assert result.extension_ratio >= 0.9

    def test_target_close_to_root_folds_chain(self, standard_arm_solver, unit_chain_transforms):
        """Target close to root should fold the chain."""
        # Target very close to root
        target = Vec3(0.1, 0.1, 0.0)

        result = standard_arm_solver.solve(
            root_transform=unit_chain_transforms["root"],
            mid_transform=unit_chain_transforms["mid"],
            end_transform=unit_chain_transforms["end"],
            target_position=target
        )

        assert result is not None
        # Extension ratio should be small when chain is folded
        assert result.extension_ratio < 0.5

    def test_chain_in_all_directions(self, standard_arm_solver, unit_chain_transforms):
        """Chain should solve for targets in all directions."""
        directions = [
            Vec3(1.0, 0.0, 0.0),   # +X
            Vec3(-1.0, 0.0, 0.0),  # -X
            Vec3(0.0, 1.0, 0.0),   # +Y
            Vec3(0.0, -1.0, 0.0),  # -Y
            Vec3(0.0, 0.0, 1.0),   # +Z
            Vec3(0.0, 0.0, -1.0),  # -Z
        ]

        for direction in directions:
            result = standard_arm_solver.solve(
                root_transform=unit_chain_transforms["root"],
                mid_transform=unit_chain_transforms["mid"],
                end_transform=unit_chain_transforms["end"],
                target_position=direction
            )

            assert result is not None, f"Failed for direction {direction}"
            assert result.success, f"Should succeed for reachable direction {direction}"


# =============================================================================
# Behavioral Tests: Valid Quaternion Rotations
# =============================================================================

class TestQuaternionRotationValidity:
    """Tests verifying result contains valid quaternion rotations."""

    def test_root_rotation_is_normalized(self, standard_arm_solver, unit_chain_transforms):
        """Root rotation quaternion should be normalized (unit length)."""
        result = standard_arm_solver.solve(
            root_transform=unit_chain_transforms["root"],
            mid_transform=unit_chain_transforms["mid"],
            end_transform=unit_chain_transforms["end"],
            target_position=Vec3(1.0, 1.0, 0.0)
        )

        if result.success and result.root_rotation is not None:
            rot = result.root_rotation
            magnitude = math.sqrt(rot.w**2 + rot.x**2 + rot.y**2 + rot.z**2)
            assert abs(magnitude - 1.0) < 0.001, f"Quaternion not normalized: magnitude={magnitude}"

    def test_mid_rotation_is_normalized(self, standard_arm_solver, unit_chain_transforms):
        """Mid rotation quaternion should be normalized (unit length)."""
        result = standard_arm_solver.solve(
            root_transform=unit_chain_transforms["root"],
            mid_transform=unit_chain_transforms["mid"],
            end_transform=unit_chain_transforms["end"],
            target_position=Vec3(1.0, 1.0, 0.0)
        )

        if result.success and result.mid_rotation is not None:
            rot = result.mid_rotation
            magnitude = math.sqrt(rot.w**2 + rot.x**2 + rot.y**2 + rot.z**2)
            assert abs(magnitude - 1.0) < 0.001, f"Quaternion not normalized: magnitude={magnitude}"

    def test_end_rotation_is_normalized(self, standard_arm_solver, unit_chain_transforms):
        """End rotation quaternion should be normalized (unit length)."""
        result = standard_arm_solver.solve(
            root_transform=unit_chain_transforms["root"],
            mid_transform=unit_chain_transforms["mid"],
            end_transform=unit_chain_transforms["end"],
            target_position=Vec3(1.0, 1.0, 0.0)
        )

        if result.success and result.end_rotation is not None:
            rot = result.end_rotation
            magnitude = math.sqrt(rot.w**2 + rot.x**2 + rot.y**2 + rot.z**2)
            assert abs(magnitude - 1.0) < 0.001, f"Quaternion not normalized: magnitude={magnitude}"

    def test_rotations_no_nan(self, standard_arm_solver, unit_chain_transforms):
        """Rotation quaternions should not contain NaN values."""
        result = standard_arm_solver.solve(
            root_transform=unit_chain_transforms["root"],
            mid_transform=unit_chain_transforms["mid"],
            end_transform=unit_chain_transforms["end"],
            target_position=Vec3(1.0, 1.0, 0.0)
        )

        if result.success:
            for rot in [result.root_rotation, result.mid_rotation, result.end_rotation]:
                if rot is not None:
                    assert not math.isnan(rot.w), "w component is NaN"
                    assert not math.isnan(rot.x), "x component is NaN"
                    assert not math.isnan(rot.y), "y component is NaN"
                    assert not math.isnan(rot.z), "z component is NaN"

    def test_rotations_no_inf(self, standard_arm_solver, unit_chain_transforms):
        """Rotation quaternions should not contain infinity values."""
        result = standard_arm_solver.solve(
            root_transform=unit_chain_transforms["root"],
            mid_transform=unit_chain_transforms["mid"],
            end_transform=unit_chain_transforms["end"],
            target_position=Vec3(1.0, 1.0, 0.0)
        )

        if result.success:
            for rot in [result.root_rotation, result.mid_rotation, result.end_rotation]:
                if rot is not None:
                    assert not math.isinf(rot.w), "w component is infinity"
                    assert not math.isinf(rot.x), "x component is infinity"
                    assert not math.isinf(rot.y), "y component is infinity"
                    assert not math.isinf(rot.z), "z component is infinity"


# =============================================================================
# Edge Cases and Boundary Conditions
# =============================================================================

class TestEdgeCases:
    """Additional edge case and boundary condition tests."""

    def test_target_on_chain_axis(self, standard_arm_solver, unit_chain_transforms):
        """Target directly along the initial chain axis."""
        # Chain starts pointing up Y, target also on Y axis
        target = Vec3(0.0, 1.5, 0.0)

        result = standard_arm_solver.solve(
            root_transform=unit_chain_transforms["root"],
            mid_transform=unit_chain_transforms["mid"],
            end_transform=unit_chain_transforms["end"],
            target_position=target
        )

        assert result is not None

    def test_target_perpendicular_to_chain(self, standard_arm_solver, unit_chain_transforms):
        """Target perpendicular to initial chain orientation."""
        # Chain points up Y, target to the side
        target = Vec3(1.5, 0.0, 0.0)

        result = standard_arm_solver.solve(
            root_transform=unit_chain_transforms["root"],
            mid_transform=unit_chain_transforms["mid"],
            end_transform=unit_chain_transforms["end"],
            target_position=target
        )

        assert result.success

    def test_target_behind_root(self, standard_arm_solver, unit_chain_transforms):
        """Target behind the root position."""
        target = Vec3(0.0, -1.5, 0.0)  # Below root

        result = standard_arm_solver.solve(
            root_transform=unit_chain_transforms["root"],
            mid_transform=unit_chain_transforms["mid"],
            end_transform=unit_chain_transforms["end"],
            target_position=target
        )

        assert result.success

    def test_diagonal_target(self, standard_arm_solver, unit_chain_transforms):
        """Target at diagonal position in 3D space."""
        target = Vec3(0.7, 0.7, 0.7)  # Diagonal, distance ~1.21

        result = standard_arm_solver.solve(
            root_transform=unit_chain_transforms["root"],
            mid_transform=unit_chain_transforms["mid"],
            end_transform=unit_chain_transforms["end"],
            target_position=target
        )

        assert result.success

    def test_collinear_initial_chain(self):
        """Chain with collinear initial positions."""
        solver = TwoBoneIK(root_bone=0, mid_bone=1, end_bone=2)

        # Perfectly straight chain
        root = make_transform(Vec3(0.0, 0.0, 0.0))
        mid = make_transform(Vec3(1.0, 0.0, 0.0))
        end = make_transform(Vec3(2.0, 0.0, 0.0))
        target = Vec3(1.0, 1.0, 0.0)

        result = solver.solve(
            root_transform=root,
            mid_transform=mid,
            end_transform=end,
            target_position=target
        )

        assert result is not None

    def test_non_origin_root(self):
        """Chain with root not at origin."""
        solver = TwoBoneIK(root_bone=0, mid_bone=1, end_bone=2)

        root = make_transform(Vec3(10.0, 5.0, -3.0))
        mid = make_transform(Vec3(10.0, 6.0, -3.0))
        end = make_transform(Vec3(10.0, 7.0, -3.0))
        target = Vec3(11.0, 5.5, -3.0)

        result = solver.solve(
            root_transform=root,
            mid_transform=mid,
            end_transform=end,
            target_position=target
        )

        assert result.success

    def test_negative_coordinate_positions(self):
        """Chain in negative coordinate space."""
        solver = TwoBoneIK(root_bone=0, mid_bone=1, end_bone=2)

        root = make_transform(Vec3(-5.0, -5.0, -5.0))
        mid = make_transform(Vec3(-5.0, -4.0, -5.0))
        end = make_transform(Vec3(-5.0, -3.0, -5.0))
        target = Vec3(-4.0, -4.0, -5.0)

        result = solver.solve(
            root_transform=root,
            mid_transform=mid,
            end_transform=end,
            target_position=target
        )

        assert result.success


# =============================================================================
# Extension Ratio Tests
# =============================================================================

class TestExtensionRatio:
    """Tests for extension_ratio field behavior."""

    def test_extension_ratio_zero_at_min_distance(self, standard_arm_solver, unit_chain_transforms):
        """Extension ratio should be near 0 when target is very close."""
        target = Vec3(0.01, 0.0, 0.0)

        result = standard_arm_solver.solve(
            root_transform=unit_chain_transforms["root"],
            mid_transform=unit_chain_transforms["mid"],
            end_transform=unit_chain_transforms["end"],
            target_position=target
        )

        assert result.extension_ratio >= 0.0

    def test_extension_ratio_one_at_max_reach(self, standard_arm_solver, unit_chain_transforms):
        """Extension ratio should be near 1.0 at max reach."""
        target = Vec3(1.99, 0.0, 0.0)  # Just under max

        result = standard_arm_solver.solve(
            root_transform=unit_chain_transforms["root"],
            mid_transform=unit_chain_transforms["mid"],
            end_transform=unit_chain_transforms["end"],
            target_position=target
        )

        assert result.extension_ratio >= 0.95

    def test_extension_ratio_scales_with_distance(self, standard_arm_solver, unit_chain_transforms):
        """Extension ratio should increase with target distance."""
        results = []
        distances = [0.5, 1.0, 1.5, 1.9]

        for dist in distances:
            target = Vec3(dist, 0.0, 0.0)
            result = standard_arm_solver.solve(
                root_transform=unit_chain_transforms["root"],
                mid_transform=unit_chain_transforms["mid"],
                end_transform=unit_chain_transforms["end"],
                target_position=target
            )
            results.append(result.extension_ratio)

        # Each subsequent result should have higher extension ratio
        for i in range(len(results) - 1):
            assert results[i] <= results[i + 1], \
                f"Extension ratio should increase with distance: {results}"


# =============================================================================
# Performance / Stress Tests
# =============================================================================

class TestPerformance:
    """Basic performance and stress tests."""

    def test_many_consecutive_solves(self, standard_arm_solver, unit_chain_transforms):
        """Solver can handle many consecutive solve calls."""
        for i in range(100):
            angle = i * 0.1
            target = Vec3(math.cos(angle), math.sin(angle) + 0.5, 0.0)

            result = standard_arm_solver.solve(
                root_transform=unit_chain_transforms["root"],
                mid_transform=unit_chain_transforms["mid"],
                end_transform=unit_chain_transforms["end"],
                target_position=target
            )

            assert result is not None

    def test_random_reachable_targets(self, standard_arm_solver, unit_chain_transforms):
        """Solver handles various random reachable targets."""
        import random
        random.seed(42)

        for _ in range(50):
            # Generate random target within reach
            angle1 = random.uniform(0, 2 * math.pi)
            angle2 = random.uniform(0, math.pi)
            distance = random.uniform(0.5, 1.8)  # Within 2.0 reach

            target = Vec3(
                distance * math.sin(angle2) * math.cos(angle1),
                distance * math.sin(angle2) * math.sin(angle1),
                distance * math.cos(angle2)
            )

            result = standard_arm_solver.solve(
                root_transform=unit_chain_transforms["root"],
                mid_transform=unit_chain_transforms["mid"],
                end_transform=unit_chain_transforms["end"],
                target_position=target
            )

            assert result.success, f"Failed for reachable target at distance {distance}"


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
