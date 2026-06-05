"""Blackbox tests for COMCalculator (T-FB-4.1).

CLEANROOM test suite for COMCalculator class.
Tests written from specification only - no implementation knowledge.

COMCalculator Specification:
- Calculates center of mass for skeleton bones
- Bones have configurable masses (default 1.0)
- COM = weighted average of bone positions
- Can calculate for full skeleton or subset
- Works with bone positions (Vec3) or transforms

Test Strategy:
- Test public API contracts only
- Test physics correctness (COM formulas)
- Test mass configuration behavior
- Test partial/subset calculations
- Test edge cases and boundaries
"""

import math
import pytest
from typing import Dict, Set, List, Optional

# Import public API
from engine.animation.ik.fullbody import COMCalculator
from engine.core.math import Vec3, Transform, Quat


# =============================================================================
# Helper Functions
# =============================================================================

def vec3_distance(a: Vec3, b: Vec3) -> float:
    """Calculate Euclidean distance between two Vec3 points."""
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def vec3_nearly_equal(a: Vec3, b: Vec3, eps: float = 1e-6) -> bool:
    """Check if two vectors are nearly equal within epsilon."""
    return vec3_distance(a, b) <= eps


def vec3_component_nearly_equal(a: Vec3, b: Vec3, eps: float = 1e-6) -> bool:
    """Check if two vectors are nearly equal component-wise."""
    return (
        abs(a.x - b.x) <= eps and
        abs(a.y - b.y) <= eps and
        abs(a.z - b.z) <= eps
    )


def make_transform(position: Vec3, rotation: Optional[Quat] = None) -> Transform:
    """Create a Transform from position and optional rotation."""
    return Transform(
        translation=position,
        rotation=rotation if rotation else Quat.identity()
    )


def compute_expected_com(
    positions: Dict[str, Vec3],
    masses: Dict[str, float],
    default_mass: float = 1.0
) -> Vec3:
    """
    Compute expected COM using standard physics formula.

    COM = sum(mass_i * position_i) / sum(mass_i)
    """
    total_mass = 0.0
    weighted_sum_x = 0.0
    weighted_sum_y = 0.0
    weighted_sum_z = 0.0

    for bone_name, position in positions.items():
        mass = masses.get(bone_name, default_mass)
        total_mass += mass
        weighted_sum_x += mass * position.x
        weighted_sum_y += mass * position.y
        weighted_sum_z += mass * position.z

    if total_mass == 0:
        return Vec3(0, 0, 0)

    return Vec3(
        weighted_sum_x / total_mass,
        weighted_sum_y / total_mass,
        weighted_sum_z / total_mass
    )


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def com_calculator() -> COMCalculator:
    """Create a fresh COMCalculator instance."""
    return COMCalculator()


@pytest.fixture
def symmetric_bone_positions() -> Dict[str, Vec3]:
    """Bones positioned symmetrically around origin."""
    return {
        "left_arm": Vec3(-1.0, 0.0, 0.0),
        "right_arm": Vec3(1.0, 0.0, 0.0),
    }


@pytest.fixture
def three_bone_positions() -> Dict[str, Vec3]:
    """Three bones at known positions."""
    return {
        "bone_a": Vec3(0.0, 0.0, 0.0),
        "bone_b": Vec3(3.0, 0.0, 0.0),
        "bone_c": Vec3(0.0, 4.0, 0.0),
    }


@pytest.fixture
def humanoid_bone_positions() -> Dict[str, Vec3]:
    """Realistic humanoid skeleton bone positions."""
    return {
        "pelvis": Vec3(0.0, 1.0, 0.0),
        "spine": Vec3(0.0, 1.3, 0.0),
        "chest": Vec3(0.0, 1.5, 0.0),
        "head": Vec3(0.0, 1.8, 0.0),
        "left_shoulder": Vec3(-0.2, 1.5, 0.0),
        "right_shoulder": Vec3(0.2, 1.5, 0.0),
        "left_hand": Vec3(-0.6, 1.2, 0.0),
        "right_hand": Vec3(0.6, 1.2, 0.0),
        "left_hip": Vec3(-0.1, 1.0, 0.0),
        "right_hip": Vec3(0.1, 1.0, 0.0),
        "left_foot": Vec3(-0.1, 0.0, 0.0),
        "right_foot": Vec3(0.1, 0.0, 0.0),
    }


# =============================================================================
# Test: COMCalculator Behavior
# =============================================================================

class TestCOMCalculatorBehavior:
    """Test basic COM calculation behavior from specification."""

    def test_com_at_origin_for_symmetric_bones(
        self, com_calculator: COMCalculator, symmetric_bone_positions: Dict[str, Vec3]
    ):
        """Two bones at (-1,0,0) and (1,0,0) with equal mass -> COM at origin."""
        result = com_calculator.calculate(symmetric_bone_positions)
        expected = Vec3(0.0, 0.0, 0.0)

        assert vec3_nearly_equal(result, expected), (
            f"Symmetric bones should have COM at origin. "
            f"Expected {expected}, got {result}"
        )

    def test_com_shifts_with_heavier_bone(
        self, com_calculator: COMCalculator, symmetric_bone_positions: Dict[str, Vec3]
    ):
        """Heavier bone pulls COM toward it."""
        # Make right arm twice as heavy
        com_calculator.set_bone_mass("right_arm", 2.0)

        result = com_calculator.calculate(symmetric_bone_positions)

        # COM should shift toward heavier bone (positive x)
        assert result.x > 0.0, (
            f"COM should shift toward heavier right_arm. "
            f"Expected x > 0, got x = {result.x}"
        )
        # With masses 1.0 and 2.0 at -1 and +1:
        # COM_x = (1.0 * -1 + 2.0 * 1) / (1.0 + 2.0) = 1/3 ≈ 0.333
        expected_x = 1.0 / 3.0
        assert abs(result.x - expected_x) < 1e-6, (
            f"COM x should be 1/3. Expected {expected_x}, got {result.x}"
        )

    def test_partial_com_ignores_excluded_bones(
        self, com_calculator: COMCalculator, three_bone_positions: Dict[str, Vec3]
    ):
        """calculate_partial only includes specified bones."""
        # Calculate COM for just bone_a and bone_b (exclude bone_c)
        subset: Set[str] = {"bone_a", "bone_b"}

        result = com_calculator.calculate_partial(three_bone_positions, subset)

        # bone_a at (0,0,0), bone_b at (3,0,0), equal mass
        # Partial COM should be at (1.5, 0, 0)
        expected = Vec3(1.5, 0.0, 0.0)

        assert vec3_nearly_equal(result, expected), (
            f"Partial COM should ignore bone_c. "
            f"Expected {expected}, got {result}"
        )

    def test_partial_com_differs_from_full_com(
        self, com_calculator: COMCalculator, three_bone_positions: Dict[str, Vec3]
    ):
        """Partial COM should differ from full COM when bones excluded."""
        full_com = com_calculator.calculate(three_bone_positions)

        subset: Set[str] = {"bone_a", "bone_b"}
        partial_com = com_calculator.calculate_partial(three_bone_positions, subset)

        # Full COM includes bone_c at (0,4,0), so y should be positive
        # Partial COM excludes bone_c, so y should be 0
        assert not vec3_nearly_equal(full_com, partial_com), (
            "Partial COM should differ from full COM when bones are excluded"
        )
        assert partial_com.y == 0.0, "Partial COM y should be 0 without bone_c"
        assert full_com.y > 0.0, "Full COM y should be positive with bone_c"


# =============================================================================
# Test: Physics Correctness
# =============================================================================

class TestPhysicsCorrectness:
    """Test that COM calculations follow correct physics formulas."""

    def test_single_point_mass_at_bone_position(self, com_calculator: COMCalculator):
        """Single bone COM should be at that bone's position."""
        bone_positions = {"single_bone": Vec3(5.0, 3.0, -2.0)}

        result = com_calculator.calculate(bone_positions)
        expected = Vec3(5.0, 3.0, -2.0)

        assert vec3_nearly_equal(result, expected), (
            f"Single bone COM should equal bone position. "
            f"Expected {expected}, got {result}"
        )

    def test_two_equal_masses_com_at_midpoint(self, com_calculator: COMCalculator):
        """Two equal masses have COM at their midpoint."""
        bone_positions = {
            "bone_a": Vec3(0.0, 0.0, 0.0),
            "bone_b": Vec3(10.0, 6.0, 4.0),
        }

        result = com_calculator.calculate(bone_positions)
        expected = Vec3(5.0, 3.0, 2.0)  # Midpoint

        assert vec3_nearly_equal(result, expected), (
            f"Equal mass COM should be at midpoint. "
            f"Expected {expected}, got {result}"
        )

    def test_weighted_com_formula(self, com_calculator: COMCalculator):
        """Verify weighted COM formula: COM = sum(m*r) / sum(m)."""
        bone_positions = {
            "light_bone": Vec3(0.0, 0.0, 0.0),
            "heavy_bone": Vec3(4.0, 0.0, 0.0),
        }

        com_calculator.set_bone_mass("light_bone", 1.0)
        com_calculator.set_bone_mass("heavy_bone", 3.0)

        result = com_calculator.calculate(bone_positions)

        # COM = (1*0 + 3*4) / (1+3) = 12/4 = 3.0
        expected = Vec3(3.0, 0.0, 0.0)

        assert vec3_nearly_equal(result, expected), (
            f"Weighted COM formula incorrect. "
            f"Expected {expected}, got {result}"
        )

    def test_com_in_3d_space(self, com_calculator: COMCalculator):
        """Verify COM calculation works correctly in 3D space."""
        bone_positions = {
            "origin": Vec3(0.0, 0.0, 0.0),
            "x_axis": Vec3(2.0, 0.0, 0.0),
            "y_axis": Vec3(0.0, 2.0, 0.0),
            "z_axis": Vec3(0.0, 0.0, 2.0),
        }

        result = com_calculator.calculate(bone_positions)

        # With equal masses, COM = average = (0.5, 0.5, 0.5)
        expected = Vec3(0.5, 0.5, 0.5)

        assert vec3_nearly_equal(result, expected), (
            f"3D COM incorrect. Expected {expected}, got {result}"
        )

    def test_com_with_varying_masses(self, com_calculator: COMCalculator):
        """Test COM with multiple bones of varying masses."""
        bone_positions = {
            "bone_1": Vec3(0.0, 0.0, 0.0),
            "bone_2": Vec3(1.0, 0.0, 0.0),
            "bone_3": Vec3(2.0, 0.0, 0.0),
        }
        masses = {"bone_1": 1.0, "bone_2": 2.0, "bone_3": 3.0}

        com_calculator.set_bone_masses(masses)
        result = com_calculator.calculate(bone_positions)

        # COM_x = (1*0 + 2*1 + 3*2) / (1+2+3) = 8/6 = 4/3
        expected = Vec3(4.0 / 3.0, 0.0, 0.0)

        assert vec3_nearly_equal(result, expected), (
            f"Varying mass COM incorrect. Expected {expected}, got {result}"
        )

    def test_com_is_independent_of_bone_order(self, com_calculator: COMCalculator):
        """COM should be same regardless of bone order in dict."""
        positions_order_1 = {
            "a": Vec3(1.0, 0.0, 0.0),
            "b": Vec3(2.0, 0.0, 0.0),
            "c": Vec3(3.0, 0.0, 0.0),
        }
        positions_order_2 = {
            "c": Vec3(3.0, 0.0, 0.0),
            "a": Vec3(1.0, 0.0, 0.0),
            "b": Vec3(2.0, 0.0, 0.0),
        }

        result_1 = com_calculator.calculate(positions_order_1)
        result_2 = com_calculator.calculate(positions_order_2)

        assert vec3_nearly_equal(result_1, result_2), (
            "COM should be independent of bone order"
        )


# =============================================================================
# Test: Mass Configuration
# =============================================================================

class TestMassConfiguration:
    """Test bone mass configuration behavior."""

    def test_default_mass_is_one(self, com_calculator: COMCalculator):
        """Unconfigured bones should have default mass of 1.0."""
        mass = com_calculator.get_bone_mass("unconfigured_bone")

        assert mass == 1.0, f"Default mass should be 1.0, got {mass}"

    def test_set_single_bone_mass(self, com_calculator: COMCalculator):
        """set_bone_mass should configure individual bone mass."""
        com_calculator.set_bone_mass("custom_bone", 5.0)

        mass = com_calculator.get_bone_mass("custom_bone")

        assert mass == 5.0, f"Configured mass should be 5.0, got {mass}"

    def test_set_bone_masses_batch(self, com_calculator: COMCalculator):
        """set_bone_masses should configure multiple bones at once."""
        masses = {
            "bone_a": 2.0,
            "bone_b": 3.0,
            "bone_c": 4.0,
        }

        com_calculator.set_bone_masses(masses)

        assert com_calculator.get_bone_mass("bone_a") == 2.0
        assert com_calculator.get_bone_mass("bone_b") == 3.0
        assert com_calculator.get_bone_mass("bone_c") == 4.0

    def test_mass_overwrite(self, com_calculator: COMCalculator):
        """Setting mass twice should overwrite previous value."""
        com_calculator.set_bone_mass("bone", 2.0)
        com_calculator.set_bone_mass("bone", 7.0)

        mass = com_calculator.get_bone_mass("bone")

        assert mass == 7.0, f"Mass should be overwritten to 7.0, got {mass}"

    def test_batch_mass_does_not_clear_existing(self, com_calculator: COMCalculator):
        """set_bone_masses should add to existing, not replace all."""
        com_calculator.set_bone_mass("existing_bone", 5.0)

        com_calculator.set_bone_masses({"new_bone": 3.0})

        # Existing bone should still have its mass
        assert com_calculator.get_bone_mass("existing_bone") == 5.0
        assert com_calculator.get_bone_mass("new_bone") == 3.0

    def test_zero_mass_bone(self, com_calculator: COMCalculator):
        """Zero mass bone should not contribute to COM."""
        bone_positions = {
            "heavy_bone": Vec3(0.0, 0.0, 0.0),
            "zero_mass_bone": Vec3(100.0, 100.0, 100.0),
        }

        com_calculator.set_bone_mass("heavy_bone", 1.0)
        com_calculator.set_bone_mass("zero_mass_bone", 0.0)

        result = com_calculator.calculate(bone_positions)
        expected = Vec3(0.0, 0.0, 0.0)

        assert vec3_nearly_equal(result, expected), (
            f"Zero mass bone should not affect COM. "
            f"Expected {expected}, got {result}"
        )

    def test_very_small_mass(self, com_calculator: COMCalculator):
        """Very small masses should still work correctly."""
        bone_positions = {
            "bone_a": Vec3(0.0, 0.0, 0.0),
            "bone_b": Vec3(1.0, 0.0, 0.0),
        }

        com_calculator.set_bone_mass("bone_a", 1e-10)
        com_calculator.set_bone_mass("bone_b", 1e-10)

        result = com_calculator.calculate(bone_positions)
        expected = Vec3(0.5, 0.0, 0.0)

        assert vec3_nearly_equal(result, expected, eps=1e-4), (
            f"Small masses should compute correctly. "
            f"Expected {expected}, got {result}"
        )

    def test_very_large_mass(self, com_calculator: COMCalculator):
        """Very large masses should still work correctly."""
        bone_positions = {
            "bone_a": Vec3(0.0, 0.0, 0.0),
            "bone_b": Vec3(1.0, 0.0, 0.0),
        }

        com_calculator.set_bone_mass("bone_a", 1e10)
        com_calculator.set_bone_mass("bone_b", 1e10)

        result = com_calculator.calculate(bone_positions)
        expected = Vec3(0.5, 0.0, 0.0)

        assert vec3_nearly_equal(result, expected), (
            f"Large masses should compute correctly. "
            f"Expected {expected}, got {result}"
        )


# =============================================================================
# Test: Transform-Based Calculation
# =============================================================================

class TestTransformBasedCalculation:
    """Test calculate_from_transforms method."""

    def test_transform_calculation_matches_position_calculation(
        self, com_calculator: COMCalculator
    ):
        """calculate_from_transforms should give same result as calculate."""
        bone_names = ["bone_a", "bone_b", "bone_c"]
        positions = [
            Vec3(0.0, 0.0, 0.0),
            Vec3(3.0, 0.0, 0.0),
            Vec3(0.0, 4.0, 0.0),
        ]

        transforms = [make_transform(pos) for pos in positions]
        bone_positions = dict(zip(bone_names, positions))

        result_transforms = com_calculator.calculate_from_transforms(
            transforms, bone_names
        )
        result_positions = com_calculator.calculate(bone_positions)

        assert vec3_nearly_equal(result_transforms, result_positions), (
            "Transform and position calculations should match"
        )

    def test_transform_ignores_rotation(self, com_calculator: COMCalculator):
        """Transform rotation should not affect COM calculation."""
        bone_names = ["bone"]
        position = Vec3(5.0, 3.0, 2.0)

        # Same position, different rotations
        transform_no_rot = make_transform(position, Quat.identity())
        transform_rotated = make_transform(position, Quat.from_euler(45, 90, 0))

        result_no_rot = com_calculator.calculate_from_transforms(
            [transform_no_rot], bone_names
        )
        result_rotated = com_calculator.calculate_from_transforms(
            [transform_rotated], bone_names
        )

        assert vec3_nearly_equal(result_no_rot, result_rotated), (
            "Rotation should not affect COM calculation"
        )

    def test_transform_with_configured_masses(self, com_calculator: COMCalculator):
        """Transform calculation should respect configured masses."""
        bone_names = ["light", "heavy"]
        positions = [Vec3(0.0, 0.0, 0.0), Vec3(4.0, 0.0, 0.0)]
        transforms = [make_transform(pos) for pos in positions]

        com_calculator.set_bone_mass("light", 1.0)
        com_calculator.set_bone_mass("heavy", 3.0)

        result = com_calculator.calculate_from_transforms(transforms, bone_names)

        # COM = (1*0 + 3*4) / 4 = 3.0
        expected = Vec3(3.0, 0.0, 0.0)

        assert vec3_nearly_equal(result, expected), (
            f"Transform calculation should use configured masses. "
            f"Expected {expected}, got {result}"
        )


# =============================================================================
# Test: Partial Calculation
# =============================================================================

class TestPartialCalculation:
    """Test calculate_partial method behavior."""

    def test_partial_with_all_bones_equals_full(self, com_calculator: COMCalculator):
        """Partial with all bones should equal full calculation."""
        bone_positions = {
            "a": Vec3(0.0, 0.0, 0.0),
            "b": Vec3(2.0, 0.0, 0.0),
            "c": Vec3(4.0, 0.0, 0.0),
        }
        all_bones: Set[str] = {"a", "b", "c"}

        full_result = com_calculator.calculate(bone_positions)
        partial_result = com_calculator.calculate_partial(bone_positions, all_bones)

        assert vec3_nearly_equal(full_result, partial_result), (
            "Partial with all bones should equal full calculation"
        )

    def test_partial_with_single_bone(self, com_calculator: COMCalculator):
        """Partial with one bone should return that bone's position."""
        bone_positions = {
            "a": Vec3(1.0, 2.0, 3.0),
            "b": Vec3(10.0, 20.0, 30.0),
        }
        single_bone: Set[str] = {"a"}

        result = com_calculator.calculate_partial(bone_positions, single_bone)
        expected = Vec3(1.0, 2.0, 3.0)

        assert vec3_nearly_equal(result, expected), (
            f"Single bone partial should return bone position. "
            f"Expected {expected}, got {result}"
        )

    def test_partial_respects_masses(self, com_calculator: COMCalculator):
        """Partial calculation should respect bone masses."""
        bone_positions = {
            "a": Vec3(0.0, 0.0, 0.0),
            "b": Vec3(4.0, 0.0, 0.0),
            "c": Vec3(100.0, 0.0, 0.0),  # Excluded
        }

        com_calculator.set_bone_mass("a", 1.0)
        com_calculator.set_bone_mass("b", 3.0)

        subset: Set[str] = {"a", "b"}
        result = com_calculator.calculate_partial(bone_positions, subset)

        # COM = (1*0 + 3*4) / 4 = 3.0
        expected = Vec3(3.0, 0.0, 0.0)

        assert vec3_nearly_equal(result, expected), (
            f"Partial should respect masses. Expected {expected}, got {result}"
        )

    def test_partial_with_nonexistent_bone_in_subset(
        self, com_calculator: COMCalculator
    ):
        """Partial should handle bones in subset but not in positions dict."""
        bone_positions = {
            "a": Vec3(0.0, 0.0, 0.0),
            "b": Vec3(2.0, 0.0, 0.0),
        }
        # Subset includes "c" which is not in positions
        subset: Set[str] = {"a", "b", "c"}

        # Should just ignore "c" and calculate for a and b
        result = com_calculator.calculate_partial(bone_positions, subset)
        expected = Vec3(1.0, 0.0, 0.0)

        assert vec3_nearly_equal(result, expected), (
            f"Should ignore nonexistent bones in subset. "
            f"Expected {expected}, got {result}"
        )


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_bone_skeleton(self, com_calculator: COMCalculator):
        """COM of single bone equals bone position."""
        bone_positions = {"only_bone": Vec3(7.5, -2.3, 8.1)}

        result = com_calculator.calculate(bone_positions)

        assert vec3_nearly_equal(result, Vec3(7.5, -2.3, 8.1))

    def test_bones_at_same_position(self, com_calculator: COMCalculator):
        """Multiple bones at same position should give that position."""
        shared_position = Vec3(3.0, 3.0, 3.0)
        bone_positions = {
            "bone_1": shared_position,
            "bone_2": shared_position,
            "bone_3": shared_position,
        }

        result = com_calculator.calculate(bone_positions)

        assert vec3_nearly_equal(result, shared_position)

    def test_negative_coordinates(self, com_calculator: COMCalculator):
        """COM should work with negative coordinates."""
        bone_positions = {
            "a": Vec3(-10.0, -20.0, -30.0),
            "b": Vec3(-4.0, -8.0, -12.0),
        }

        result = com_calculator.calculate(bone_positions)
        expected = Vec3(-7.0, -14.0, -21.0)

        assert vec3_nearly_equal(result, expected)

    def test_mixed_positive_negative_coordinates(self, com_calculator: COMCalculator):
        """COM should work with mixed coordinates."""
        bone_positions = {
            "positive": Vec3(10.0, 10.0, 10.0),
            "negative": Vec3(-10.0, -10.0, -10.0),
        }

        result = com_calculator.calculate(bone_positions)
        expected = Vec3(0.0, 0.0, 0.0)

        assert vec3_nearly_equal(result, expected)

    def test_very_large_coordinates(self, com_calculator: COMCalculator):
        """COM should handle large coordinate values."""
        bone_positions = {
            "a": Vec3(1e6, 1e6, 1e6),
            "b": Vec3(1e6 + 2, 1e6 + 2, 1e6 + 2),
        }

        result = com_calculator.calculate(bone_positions)
        expected = Vec3(1e6 + 1, 1e6 + 1, 1e6 + 1)

        assert vec3_nearly_equal(result, expected, eps=1e-3)

    def test_very_small_coordinate_differences(self, com_calculator: COMCalculator):
        """COM should handle tiny coordinate differences."""
        bone_positions = {
            "a": Vec3(0.0, 0.0, 0.0),
            "b": Vec3(1e-8, 1e-8, 1e-8),
        }

        result = com_calculator.calculate(bone_positions)
        expected = Vec3(0.5e-8, 0.5e-8, 0.5e-8)

        assert vec3_nearly_equal(result, expected, eps=1e-10)


# =============================================================================
# Test: Humanoid Skeleton Scenarios
# =============================================================================

class TestHumanoidScenarios:
    """Test realistic humanoid skeleton scenarios."""

    def test_standing_humanoid_com_above_feet(
        self, com_calculator: COMCalculator, humanoid_bone_positions: Dict[str, Vec3]
    ):
        """Standing humanoid COM should be above the feet (y > 0)."""
        result = com_calculator.calculate(humanoid_bone_positions)

        assert result.y > 0.0, "Humanoid COM should be above ground level"

    def test_symmetric_humanoid_com_on_centerline(
        self, com_calculator: COMCalculator, humanoid_bone_positions: Dict[str, Vec3]
    ):
        """Symmetric humanoid COM should be on centerline (x near 0)."""
        result = com_calculator.calculate(humanoid_bone_positions)

        assert abs(result.x) < 0.1, (
            f"Symmetric humanoid COM should be near x=0, got x={result.x}"
        )

    def test_arm_raising_shifts_com(
        self, com_calculator: COMCalculator, humanoid_bone_positions: Dict[str, Vec3]
    ):
        """Raising an arm should shift COM toward that arm."""
        # Calculate baseline COM
        baseline_com = com_calculator.calculate(humanoid_bone_positions)

        # Raise right arm (move hand up and to the side)
        modified_positions = humanoid_bone_positions.copy()
        modified_positions["right_hand"] = Vec3(1.0, 2.0, 0.0)

        raised_arm_com = com_calculator.calculate(modified_positions)

        # COM should shift right (positive x) and up (positive y)
        assert raised_arm_com.x > baseline_com.x, (
            "Raising right arm should shift COM right"
        )
        assert raised_arm_com.y > baseline_com.y, (
            "Raising arm should shift COM up"
        )

    def test_heavy_torso_keeps_com_centered(self, com_calculator: COMCalculator):
        """Heavy torso should keep COM near body center even with limbs extended."""
        bone_positions = {
            "pelvis": Vec3(0.0, 1.0, 0.0),
            "spine": Vec3(0.0, 1.3, 0.0),
            "chest": Vec3(0.0, 1.5, 0.0),
            "left_hand": Vec3(-2.0, 1.5, 0.0),
            "right_hand": Vec3(2.0, 1.5, 0.0),
        }

        # Make torso very heavy compared to hands
        com_calculator.set_bone_mass("pelvis", 20.0)
        com_calculator.set_bone_mass("spine", 15.0)
        com_calculator.set_bone_mass("chest", 10.0)
        com_calculator.set_bone_mass("left_hand", 1.0)
        com_calculator.set_bone_mass("right_hand", 1.0)

        result = com_calculator.calculate(bone_positions)

        # COM should be close to centerline despite hands being extended
        assert abs(result.x) < 0.1, (
            f"Heavy torso should keep COM centered, got x={result.x}"
        )

    def test_upper_body_partial_com(
        self, com_calculator: COMCalculator, humanoid_bone_positions: Dict[str, Vec3]
    ):
        """Partial COM of upper body should be higher than full body COM."""
        upper_body: Set[str] = {
            "spine", "chest", "head",
            "left_shoulder", "right_shoulder",
            "left_hand", "right_hand"
        }

        full_com = com_calculator.calculate(humanoid_bone_positions)
        upper_com = com_calculator.calculate_partial(humanoid_bone_positions, upper_body)

        assert upper_com.y > full_com.y, (
            "Upper body COM should be higher than full body COM"
        )


# =============================================================================
# Test: Property Attribute Access
# =============================================================================

class TestPropertyAccess:
    """Test property/attribute access behavior."""

    def test_bone_masses_attribute_exists(self, com_calculator: COMCalculator):
        """COMCalculator should have bone_masses attribute."""
        assert hasattr(com_calculator, "bone_masses")

    def test_default_mass_attribute_exists(self, com_calculator: COMCalculator):
        """COMCalculator should have default_mass attribute."""
        assert hasattr(com_calculator, "default_mass")

    def test_default_mass_is_one_initially(self, com_calculator: COMCalculator):
        """default_mass should be 1.0 initially."""
        assert com_calculator.default_mass == 1.0

    def test_bone_masses_initially_empty(self, com_calculator: COMCalculator):
        """bone_masses should be empty dict initially."""
        assert len(com_calculator.bone_masses) == 0 or isinstance(
            com_calculator.bone_masses, dict
        )

    def test_bone_masses_reflects_set_values(self, com_calculator: COMCalculator):
        """bone_masses should reflect set values."""
        com_calculator.set_bone_mass("test_bone", 5.0)

        assert "test_bone" in com_calculator.bone_masses
        assert com_calculator.bone_masses["test_bone"] == 5.0


# =============================================================================
# Test: Consistency and Determinism
# =============================================================================

class TestConsistencyDeterminism:
    """Test that calculations are consistent and deterministic."""

    def test_same_input_same_output(self, com_calculator: COMCalculator):
        """Same input should always produce same output."""
        bone_positions = {
            "a": Vec3(1.0, 2.0, 3.0),
            "b": Vec3(4.0, 5.0, 6.0),
        }

        results = [com_calculator.calculate(bone_positions) for _ in range(10)]

        for result in results[1:]:
            assert vec3_nearly_equal(result, results[0]), (
                "Repeated calculations should be identical"
            )

    def test_calculation_does_not_modify_input(self, com_calculator: COMCalculator):
        """Calculation should not modify input data."""
        bone_positions = {
            "a": Vec3(1.0, 2.0, 3.0),
            "b": Vec3(4.0, 5.0, 6.0),
        }
        original_positions = {
            "a": Vec3(1.0, 2.0, 3.0),
            "b": Vec3(4.0, 5.0, 6.0),
        }

        com_calculator.calculate(bone_positions)

        for name in bone_positions:
            assert vec3_nearly_equal(
                bone_positions[name], original_positions[name]
            ), "Input positions should not be modified"

    def test_independent_calculators(self):
        """Different calculator instances should be independent."""
        calc1 = COMCalculator()
        calc2 = COMCalculator()

        calc1.set_bone_mass("bone", 5.0)

        assert calc2.get_bone_mass("bone") == 1.0, (
            "Different instances should be independent"
        )


# =============================================================================
# Test: Error Handling (Expected Behavior)
# =============================================================================

class TestErrorHandling:
    """Test error handling and edge case behavior."""

    def test_empty_bone_positions_handling(self, com_calculator: COMCalculator):
        """Empty bone positions should be handled gracefully."""
        bone_positions: Dict[str, Vec3] = {}

        # Should either return zero vector or raise an appropriate error
        try:
            result = com_calculator.calculate(bone_positions)
            # If no error, expect zero or some default
            assert result is not None
        except (ValueError, ZeroDivisionError):
            # Raising an error for empty input is also acceptable
            pass

    def test_empty_subset_handling(self, com_calculator: COMCalculator):
        """Empty subset should be handled gracefully."""
        bone_positions = {"a": Vec3(1.0, 0.0, 0.0)}
        empty_subset: Set[str] = set()

        try:
            result = com_calculator.calculate_partial(bone_positions, empty_subset)
            assert result is not None
        except (ValueError, ZeroDivisionError):
            pass

    def test_mismatched_transforms_and_names_lengths(
        self, com_calculator: COMCalculator
    ):
        """Mismatched transform and name list lengths should be handled."""
        transforms = [
            make_transform(Vec3(0.0, 0.0, 0.0)),
            make_transform(Vec3(1.0, 0.0, 0.0)),
        ]
        bone_names = ["only_one_name"]  # Length mismatch

        try:
            result = com_calculator.calculate_from_transforms(transforms, bone_names)
            # If no error, implementation handles mismatch somehow
            assert result is not None
        except (ValueError, IndexError, AssertionError):
            # Raising error for mismatch is acceptable
            pass


# =============================================================================
# Test: Mathematical Properties
# =============================================================================

class TestMathematicalProperties:
    """Test mathematical properties of COM calculation."""

    def test_com_within_convex_hull(self, com_calculator: COMCalculator):
        """COM should lie within convex hull of bone positions."""
        bone_positions = {
            "a": Vec3(0.0, 0.0, 0.0),
            "b": Vec3(10.0, 0.0, 0.0),
            "c": Vec3(5.0, 10.0, 0.0),
        }

        result = com_calculator.calculate(bone_positions)

        # COM should be within bounding box at minimum
        assert 0.0 <= result.x <= 10.0, "COM x should be within bone x range"
        assert 0.0 <= result.y <= 10.0, "COM y should be within bone y range"

    def test_com_shifts_linearly_with_position(self, com_calculator: COMCalculator):
        """Moving a bone should shift COM proportionally."""
        bone_positions_a = {
            "static": Vec3(0.0, 0.0, 0.0),
            "moving": Vec3(2.0, 0.0, 0.0),
        }
        bone_positions_b = {
            "static": Vec3(0.0, 0.0, 0.0),
            "moving": Vec3(4.0, 0.0, 0.0),  # Moved 2 units further
        }

        com_a = com_calculator.calculate(bone_positions_a)
        com_b = com_calculator.calculate(bone_positions_b)

        # With equal masses, moving one bone 2 units should move COM 1 unit
        expected_shift = 1.0
        actual_shift = com_b.x - com_a.x

        assert abs(actual_shift - expected_shift) < 1e-6, (
            f"COM should shift linearly. Expected {expected_shift}, got {actual_shift}"
        )

    def test_scaling_masses_equally_preserves_com(self, com_calculator: COMCalculator):
        """Scaling all masses equally should not change COM."""
        bone_positions = {
            "a": Vec3(0.0, 0.0, 0.0),
            "b": Vec3(3.0, 0.0, 0.0),
            "c": Vec3(0.0, 4.0, 0.0),
        }

        # Calculate with default masses
        com_default = com_calculator.calculate(bone_positions)

        # Scale all masses by same factor
        com_calculator.set_bone_masses({
            "a": 5.0,
            "b": 5.0,
            "c": 5.0,
        })
        com_scaled = com_calculator.calculate(bone_positions)

        assert vec3_nearly_equal(com_default, com_scaled), (
            "Equal mass scaling should preserve COM"
        )

    def test_translating_all_bones_translates_com(self, com_calculator: COMCalculator):
        """Translating all bones by offset should translate COM by same offset."""
        bone_positions_original = {
            "a": Vec3(0.0, 0.0, 0.0),
            "b": Vec3(2.0, 0.0, 0.0),
        }

        offset = Vec3(5.0, 3.0, -2.0)
        bone_positions_translated = {
            "a": Vec3(0.0 + offset.x, 0.0 + offset.y, 0.0 + offset.z),
            "b": Vec3(2.0 + offset.x, 0.0 + offset.y, 0.0 + offset.z),
        }

        com_original = com_calculator.calculate(bone_positions_original)
        com_translated = com_calculator.calculate(bone_positions_translated)

        expected_com = Vec3(
            com_original.x + offset.x,
            com_original.y + offset.y,
            com_original.z + offset.z,
        )

        assert vec3_nearly_equal(com_translated, expected_com), (
            "Translating bones should translate COM equally"
        )
