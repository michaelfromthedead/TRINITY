"""Whitebox tests for COMCalculator class.

Tests internal implementation details of the COMCalculator class,
including per-bone mass configuration, weighted average computation,
and edge case handling.
"""

from __future__ import annotations

import pytest
import math

from engine.animation.ik.fullbody import COMCalculator
from engine.core.math.vec import Vec3
from engine.core.math.transform import Transform
from engine.core.math.quat import Quat


class TestCOMCalculatorInit:
    """Tests for COMCalculator initialization."""

    def test_default_initialization(self) -> None:
        """Default init creates empty bone_masses dict and default_mass=1.0."""
        calc = COMCalculator()
        assert calc.bone_masses == {}
        assert calc.default_mass == 1.0

    def test_with_bone_masses(self) -> None:
        """Init with bone_masses dictionary."""
        masses = {"hip": 10.0, "spine": 5.0, "head": 2.0}
        calc = COMCalculator(bone_masses=masses)
        assert calc.bone_masses == masses
        assert calc.default_mass == 1.0

    def test_with_custom_default_mass(self) -> None:
        """Init with custom default_mass value."""
        calc = COMCalculator(default_mass=2.5)
        assert calc.bone_masses == {}
        assert calc.default_mass == 2.5

    def test_with_both_parameters(self) -> None:
        """Init with both bone_masses and default_mass."""
        masses = {"pelvis": 15.0}
        calc = COMCalculator(bone_masses=masses, default_mass=0.5)
        assert calc.bone_masses == masses
        assert calc.default_mass == 0.5

    def test_bone_masses_is_mutable(self) -> None:
        """Verify bone_masses can be modified after init."""
        calc = COMCalculator()
        calc.bone_masses["new_bone"] = 3.0
        assert "new_bone" in calc.bone_masses
        assert calc.bone_masses["new_bone"] == 3.0


class TestBoneMassConfig:
    """Tests for bone mass configuration methods."""

    def test_set_bone_mass(self) -> None:
        """set_bone_mass adds entry to bone_masses dict."""
        calc = COMCalculator()
        calc.set_bone_mass("hip", 10.0)
        assert calc.bone_masses["hip"] == 10.0

    def test_set_bone_mass_overwrites(self) -> None:
        """set_bone_mass overwrites existing entry."""
        calc = COMCalculator(bone_masses={"hip": 5.0})
        calc.set_bone_mass("hip", 12.0)
        assert calc.bone_masses["hip"] == 12.0

    def test_set_bone_mass_zero(self) -> None:
        """Zero mass is allowed (makes bone weightless)."""
        calc = COMCalculator()
        calc.set_bone_mass("finger", 0.0)
        assert calc.bone_masses["finger"] == 0.0

    def test_set_bone_mass_small_positive(self) -> None:
        """Very small positive mass is allowed."""
        calc = COMCalculator()
        calc.set_bone_mass("eyelid", 0.001)
        assert calc.bone_masses["eyelid"] == 0.001

    def test_set_bone_mass_large(self) -> None:
        """Large mass values are allowed."""
        calc = COMCalculator()
        calc.set_bone_mass("torso", 100000.0)
        assert calc.bone_masses["torso"] == 100000.0

    def test_negative_mass_raises(self) -> None:
        """Negative mass raises ValueError."""
        calc = COMCalculator()
        with pytest.raises(ValueError, match="Mass cannot be negative"):
            calc.set_bone_mass("bone", -1.0)

    def test_negative_mass_raises_small(self) -> None:
        """Small negative mass also raises ValueError."""
        calc = COMCalculator()
        with pytest.raises(ValueError, match="Mass cannot be negative"):
            calc.set_bone_mass("bone", -0.001)

    def test_set_bone_masses_batch(self) -> None:
        """set_bone_masses adds multiple entries at once."""
        calc = COMCalculator()
        masses = {"hip": 10.0, "spine": 5.0, "head": 2.0, "neck": 1.5}
        calc.set_bone_masses(masses)
        assert calc.bone_masses == masses

    def test_set_bone_masses_partial_overwrite(self) -> None:
        """set_bone_masses overwrites only specified entries."""
        calc = COMCalculator(bone_masses={"hip": 1.0, "spine": 1.0})
        calc.set_bone_masses({"hip": 15.0, "head": 3.0})
        assert calc.bone_masses["hip"] == 15.0
        assert calc.bone_masses["spine"] == 1.0
        assert calc.bone_masses["head"] == 3.0

    def test_set_bone_masses_empty_dict(self) -> None:
        """set_bone_masses with empty dict does nothing."""
        calc = COMCalculator(bone_masses={"hip": 5.0})
        calc.set_bone_masses({})
        assert calc.bone_masses == {"hip": 5.0}

    def test_set_bone_masses_negative_raises(self) -> None:
        """set_bone_masses raises on first negative mass."""
        calc = COMCalculator()
        with pytest.raises(ValueError, match="Mass cannot be negative"):
            calc.set_bone_masses({"hip": 5.0, "bad_bone": -1.0, "head": 2.0})

    def test_set_bone_masses_rollback_not_guaranteed(self) -> None:
        """When set_bone_masses fails, prior valid entries may be set."""
        calc = COMCalculator()
        try:
            # Depending on dict iteration order, "good_bone" might be set
            calc.set_bone_masses({"good_bone": 5.0, "bad_bone": -1.0})
        except ValueError:
            pass
        # We don't guarantee atomicity - some entries may have been set

    def test_get_bone_mass_configured(self) -> None:
        """get_bone_mass returns configured mass."""
        calc = COMCalculator(bone_masses={"hip": 12.0})
        assert calc.get_bone_mass("hip") == 12.0

    def test_get_bone_mass_default(self) -> None:
        """get_bone_mass returns default_mass for unconfigured bones."""
        calc = COMCalculator(default_mass=3.0)
        assert calc.get_bone_mass("unknown_bone") == 3.0

    def test_get_bone_mass_zero_configured(self) -> None:
        """get_bone_mass returns 0.0 if configured as zero."""
        calc = COMCalculator(bone_masses={"weightless": 0.0}, default_mass=5.0)
        assert calc.get_bone_mass("weightless") == 0.0


class TestCalculate:
    """Tests for the calculate() method."""

    def test_single_bone(self) -> None:
        """COM of single bone equals its position."""
        calc = COMCalculator()
        positions = {"bone": Vec3(1.0, 2.0, 3.0)}
        result = calc.calculate(positions)
        assert result.x == pytest.approx(1.0)
        assert result.y == pytest.approx(2.0)
        assert result.z == pytest.approx(3.0)

    def test_single_bone_with_mass(self) -> None:
        """COM of single bone with configured mass equals its position."""
        calc = COMCalculator(bone_masses={"bone": 100.0})
        positions = {"bone": Vec3(5.0, -3.0, 7.0)}
        result = calc.calculate(positions)
        assert result.x == pytest.approx(5.0)
        assert result.y == pytest.approx(-3.0)
        assert result.z == pytest.approx(7.0)

    def test_multiple_bones_equal_mass(self) -> None:
        """COM of equal-mass bones is simple average."""
        calc = COMCalculator()  # default_mass = 1.0 for all
        positions = {
            "a": Vec3(0.0, 0.0, 0.0),
            "b": Vec3(2.0, 0.0, 0.0),
            "c": Vec3(0.0, 2.0, 0.0),
        }
        result = calc.calculate(positions)
        # Expected: (0+2+0)/3, (0+0+2)/3, (0+0+0)/3 = (2/3, 2/3, 0)
        assert result.x == pytest.approx(2.0 / 3.0)
        assert result.y == pytest.approx(2.0 / 3.0)
        assert result.z == pytest.approx(0.0)

    def test_multiple_bones_weighted(self) -> None:
        """COM calculation uses weighted average."""
        calc = COMCalculator(bone_masses={"heavy": 9.0, "light": 1.0})
        positions = {
            "heavy": Vec3(0.0, 0.0, 0.0),
            "light": Vec3(10.0, 0.0, 0.0),
        }
        result = calc.calculate(positions)
        # Expected: (9*0 + 1*10) / (9+1) = 1.0
        assert result.x == pytest.approx(1.0)
        assert result.y == pytest.approx(0.0)
        assert result.z == pytest.approx(0.0)

    def test_weighted_average_formula(self) -> None:
        """Verify COM = sum(mass_i * pos_i) / sum(mass_i) formula."""
        calc = COMCalculator(
            bone_masses={"a": 2.0, "b": 3.0, "c": 5.0},
            default_mass=0.0  # Only configured bones count
        )
        positions = {
            "a": Vec3(1.0, 0.0, 0.0),
            "b": Vec3(0.0, 1.0, 0.0),
            "c": Vec3(0.0, 0.0, 1.0),
        }
        result = calc.calculate(positions)
        # total_mass = 2+3+5 = 10
        # weighted_sum = 2*(1,0,0) + 3*(0,1,0) + 5*(0,0,1) = (2, 3, 5)
        # COM = (2,3,5)/10 = (0.2, 0.3, 0.5)
        assert result.x == pytest.approx(0.2)
        assert result.y == pytest.approx(0.3)
        assert result.z == pytest.approx(0.5)

    def test_empty_positions(self) -> None:
        """Empty positions dict returns zero vector (total_mass=0)."""
        calc = COMCalculator()
        result = calc.calculate({})
        assert result.x == pytest.approx(0.0)
        assert result.y == pytest.approx(0.0)
        assert result.z == pytest.approx(0.0)

    def test_all_zero_mass(self) -> None:
        """All bones with zero mass returns zero vector."""
        calc = COMCalculator(
            bone_masses={"a": 0.0, "b": 0.0},
            default_mass=0.0
        )
        positions = {
            "a": Vec3(5.0, 5.0, 5.0),
            "b": Vec3(-5.0, -5.0, -5.0),
        }
        result = calc.calculate(positions)
        assert result.x == pytest.approx(0.0)
        assert result.y == pytest.approx(0.0)
        assert result.z == pytest.approx(0.0)

    def test_mixed_zero_and_positive_mass(self) -> None:
        """Zero-mass bones don't contribute to COM."""
        calc = COMCalculator(bone_masses={"zero": 0.0, "one": 1.0})
        positions = {
            "zero": Vec3(100.0, 100.0, 100.0),
            "one": Vec3(5.0, -3.0, 2.0),
        }
        result = calc.calculate(positions)
        # Only "one" contributes
        assert result.x == pytest.approx(5.0)
        assert result.y == pytest.approx(-3.0)
        assert result.z == pytest.approx(2.0)

    def test_unconfigured_bones_use_default(self) -> None:
        """Bones not in bone_masses use default_mass."""
        calc = COMCalculator(bone_masses={"configured": 1.0}, default_mass=1.0)
        positions = {
            "configured": Vec3(0.0, 0.0, 0.0),
            "unconfigured": Vec3(10.0, 0.0, 0.0),
        }
        result = calc.calculate(positions)
        # Both have mass 1.0, so simple average
        assert result.x == pytest.approx(5.0)
        assert result.y == pytest.approx(0.0)

    def test_negative_positions(self) -> None:
        """Handle negative position coordinates correctly."""
        calc = COMCalculator()
        positions = {
            "a": Vec3(-10.0, -20.0, -30.0),
            "b": Vec3(10.0, 20.0, 30.0),
        }
        result = calc.calculate(positions)
        # Average = (0, 0, 0)
        assert result.x == pytest.approx(0.0)
        assert result.y == pytest.approx(0.0)
        assert result.z == pytest.approx(0.0)

    def test_large_number_of_bones(self) -> None:
        """Handle many bones efficiently."""
        calc = COMCalculator(default_mass=1.0)
        # 100 bones evenly distributed on x-axis from 0 to 99
        positions = {f"bone_{i}": Vec3(float(i), 0.0, 0.0) for i in range(100)}
        result = calc.calculate(positions)
        # Expected x = (0+1+...+99)/100 = 4950/100 = 49.5
        assert result.x == pytest.approx(49.5)
        assert result.y == pytest.approx(0.0)
        assert result.z == pytest.approx(0.0)


class TestCalculatePartial:
    """Tests for calculate_partial() method."""

    def test_subset_of_bones(self) -> None:
        """Calculate COM for subset of bones only."""
        calc = COMCalculator(default_mass=1.0)
        positions = {
            "arm_l": Vec3(0.0, 0.0, 0.0),
            "arm_r": Vec3(10.0, 0.0, 0.0),
            "leg_l": Vec3(0.0, -10.0, 0.0),
            "leg_r": Vec3(10.0, -10.0, 0.0),
        }
        # Only calculate for arms
        result = calc.calculate_partial(positions, {"arm_l", "arm_r"})
        # Average of arm positions
        assert result.x == pytest.approx(5.0)
        assert result.y == pytest.approx(0.0)
        assert result.z == pytest.approx(0.0)

    def test_missing_bones_ignored(self) -> None:
        """Bones in subset but not in positions are ignored."""
        calc = COMCalculator(default_mass=1.0)
        positions = {
            "present": Vec3(5.0, 5.0, 5.0),
        }
        # "missing" is in subset but not in positions
        result = calc.calculate_partial(positions, {"present", "missing"})
        # Only "present" contributes
        assert result.x == pytest.approx(5.0)
        assert result.y == pytest.approx(5.0)
        assert result.z == pytest.approx(5.0)

    def test_empty_subset(self) -> None:
        """Empty subset returns zero vector."""
        calc = COMCalculator()
        positions = {"bone": Vec3(10.0, 10.0, 10.0)}
        result = calc.calculate_partial(positions, set())
        assert result.x == pytest.approx(0.0)
        assert result.y == pytest.approx(0.0)
        assert result.z == pytest.approx(0.0)

    def test_empty_positions(self) -> None:
        """Empty positions with non-empty subset returns zero."""
        calc = COMCalculator()
        result = calc.calculate_partial({}, {"bone_a", "bone_b"})
        assert result.x == pytest.approx(0.0)
        assert result.y == pytest.approx(0.0)
        assert result.z == pytest.approx(0.0)

    def test_weighted_partial(self) -> None:
        """Partial calculation uses correct weights."""
        calc = COMCalculator(bone_masses={"heavy": 9.0, "medium": 3.0, "light": 1.0})
        positions = {
            "heavy": Vec3(0.0, 0.0, 0.0),
            "medium": Vec3(10.0, 0.0, 0.0),
            "light": Vec3(100.0, 0.0, 0.0),
        }
        # Only heavy and medium
        result = calc.calculate_partial(positions, {"heavy", "medium"})
        # (9*0 + 3*10) / (9+3) = 30/12 = 2.5
        assert result.x == pytest.approx(2.5)

    def test_subset_with_zero_mass(self) -> None:
        """Subset bones with zero mass don't contribute."""
        calc = COMCalculator(bone_masses={"zero": 0.0, "one": 1.0})
        positions = {
            "zero": Vec3(100.0, 0.0, 0.0),
            "one": Vec3(5.0, 0.0, 0.0),
        }
        result = calc.calculate_partial(positions, {"zero", "one"})
        assert result.x == pytest.approx(5.0)

    def test_all_subset_bones_zero_mass(self) -> None:
        """If all subset bones have zero mass, return zero vector."""
        calc = COMCalculator(bone_masses={"a": 0.0, "b": 0.0})
        positions = {
            "a": Vec3(10.0, 20.0, 30.0),
            "b": Vec3(-10.0, -20.0, -30.0),
        }
        result = calc.calculate_partial(positions, {"a", "b"})
        assert result.x == pytest.approx(0.0)
        assert result.y == pytest.approx(0.0)
        assert result.z == pytest.approx(0.0)

    def test_partial_uses_default_mass(self) -> None:
        """Unconfigured bones in subset use default_mass."""
        calc = COMCalculator(bone_masses={"configured": 2.0}, default_mass=2.0)
        positions = {
            "configured": Vec3(0.0, 0.0, 0.0),
            "unconfigured": Vec3(10.0, 0.0, 0.0),
        }
        result = calc.calculate_partial(positions, {"configured", "unconfigured"})
        # Both have mass 2.0, so average
        assert result.x == pytest.approx(5.0)


class TestCalculateFromTransforms:
    """Tests for calculate_from_transforms() method."""

    def test_transform_positions_extracted(self) -> None:
        """Extracts translation from Transform objects."""
        calc = COMCalculator(default_mass=1.0)
        transforms = [
            Transform(translation=Vec3(0.0, 0.0, 0.0)),
            Transform(translation=Vec3(10.0, 0.0, 0.0)),
        ]
        bone_names = ["bone_a", "bone_b"]
        result = calc.calculate_from_transforms(transforms, bone_names)
        assert result.x == pytest.approx(5.0)
        assert result.y == pytest.approx(0.0)
        assert result.z == pytest.approx(0.0)

    def test_transform_uses_bone_masses(self) -> None:
        """Uses configured bone_masses for weighting."""
        calc = COMCalculator(bone_masses={"heavy": 3.0, "light": 1.0})
        transforms = [
            Transform(translation=Vec3(0.0, 0.0, 0.0)),
            Transform(translation=Vec3(4.0, 0.0, 0.0)),
        ]
        bone_names = ["heavy", "light"]
        result = calc.calculate_from_transforms(transforms, bone_names)
        # (3*0 + 1*4) / 4 = 1.0
        assert result.x == pytest.approx(1.0)

    def test_transform_ignores_rotation_scale(self) -> None:
        """Only translation is used, rotation and scale are ignored."""
        calc = COMCalculator()
        transforms = [
            Transform(
                translation=Vec3(5.0, 0.0, 0.0),
                rotation=Quat(0.0, 0.707, 0.0, 0.707),  # 90 degree rotation
                scale=Vec3(2.0, 2.0, 2.0),
            ),
        ]
        bone_names = ["bone"]
        result = calc.calculate_from_transforms(transforms, bone_names)
        # Should only use translation
        assert result.x == pytest.approx(5.0)
        assert result.y == pytest.approx(0.0)
        assert result.z == pytest.approx(0.0)

    def test_mismatched_lengths_raises(self) -> None:
        """Raises ValueError when transforms and bone_names differ in length."""
        calc = COMCalculator()
        transforms = [Transform(), Transform()]
        bone_names = ["only_one"]
        with pytest.raises(ValueError, match="must have the same length"):
            calc.calculate_from_transforms(transforms, bone_names)

    def test_more_names_than_transforms_raises(self) -> None:
        """Raises ValueError when more names than transforms."""
        calc = COMCalculator()
        transforms = [Transform()]
        bone_names = ["a", "b", "c"]
        with pytest.raises(ValueError, match="must have the same length"):
            calc.calculate_from_transforms(transforms, bone_names)

    def test_empty_lists(self) -> None:
        """Empty lists return zero vector."""
        calc = COMCalculator()
        result = calc.calculate_from_transforms([], [])
        assert result.x == pytest.approx(0.0)
        assert result.y == pytest.approx(0.0)
        assert result.z == pytest.approx(0.0)

    def test_single_transform(self) -> None:
        """Single transform returns its translation."""
        calc = COMCalculator()
        transforms = [Transform(translation=Vec3(7.0, -3.0, 2.0))]
        bone_names = ["single"]
        result = calc.calculate_from_transforms(transforms, bone_names)
        assert result.x == pytest.approx(7.0)
        assert result.y == pytest.approx(-3.0)
        assert result.z == pytest.approx(2.0)

    def test_delegates_to_calculate(self) -> None:
        """Verify calculate_from_transforms delegates to calculate()."""
        calc = COMCalculator(bone_masses={"a": 1.0, "b": 3.0})
        transforms = [
            Transform(translation=Vec3(0.0, 0.0, 0.0)),
            Transform(translation=Vec3(8.0, 0.0, 0.0)),
        ]
        bone_names = ["a", "b"]

        # Calculate via transforms
        result_transforms = calc.calculate_from_transforms(transforms, bone_names)

        # Calculate directly
        positions = {name: t.translation for name, t in zip(bone_names, transforms)}
        result_direct = calc.calculate(positions)

        assert result_transforms.x == pytest.approx(result_direct.x)
        assert result_transforms.y == pytest.approx(result_direct.y)
        assert result_transforms.z == pytest.approx(result_direct.z)


class TestTotalMass:
    """Tests for total_mass() method."""

    def test_total_mass_all_configured(self) -> None:
        """total_mass with no args sums all configured bone masses."""
        calc = COMCalculator(bone_masses={"a": 1.0, "b": 2.0, "c": 3.0})
        assert calc.total_mass() == pytest.approx(6.0)

    def test_total_mass_no_configured(self) -> None:
        """total_mass with no configured bones returns 0."""
        calc = COMCalculator()
        assert calc.total_mass() == pytest.approx(0.0)

    def test_total_mass_with_bone_names(self) -> None:
        """total_mass for specific bones uses default for unconfigured."""
        calc = COMCalculator(
            bone_masses={"configured": 5.0},
            default_mass=2.0
        )
        total = calc.total_mass({"configured", "unconfigured"})
        # 5.0 + 2.0 = 7.0
        assert total == pytest.approx(7.0)

    def test_total_mass_empty_bone_set(self) -> None:
        """total_mass with empty set returns 0."""
        calc = COMCalculator(bone_masses={"a": 10.0})
        assert calc.total_mass(set()) == pytest.approx(0.0)

    def test_total_mass_subset(self) -> None:
        """total_mass for subset of configured bones."""
        calc = COMCalculator(bone_masses={"a": 1.0, "b": 2.0, "c": 3.0})
        assert calc.total_mass({"a", "c"}) == pytest.approx(4.0)

    def test_total_mass_with_zero_mass(self) -> None:
        """Zero-mass bones contribute zero to total."""
        calc = COMCalculator(bone_masses={"a": 5.0, "b": 0.0})
        assert calc.total_mass({"a", "b"}) == pytest.approx(5.0)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_small_mass(self) -> None:
        """Very small mass values work correctly."""
        calc = COMCalculator(bone_masses={"tiny": 1e-10, "normal": 1.0})
        positions = {
            "tiny": Vec3(1000.0, 0.0, 0.0),
            "normal": Vec3(0.0, 0.0, 0.0),
        }
        result = calc.calculate(positions)
        # Tiny mass barely affects result
        assert result.x == pytest.approx(0.0, abs=1e-6)

    def test_very_large_mass(self) -> None:
        """Very large mass values work correctly."""
        calc = COMCalculator(bone_masses={"huge": 1e10, "normal": 1.0})
        positions = {
            "huge": Vec3(5.0, 0.0, 0.0),
            "normal": Vec3(1000.0, 0.0, 0.0),
        }
        result = calc.calculate(positions)
        # Huge mass dominates
        assert result.x == pytest.approx(5.0, abs=1e-4)

    def test_extreme_positions(self) -> None:
        """Handle extreme position values."""
        calc = COMCalculator()
        positions = {
            "far": Vec3(1e8, 1e8, 1e8),
            "near": Vec3(-1e8, -1e8, -1e8),
        }
        result = calc.calculate(positions)
        # Should average to zero
        assert result.x == pytest.approx(0.0)
        assert result.y == pytest.approx(0.0)
        assert result.z == pytest.approx(0.0)

    def test_bone_name_special_characters(self) -> None:
        """Bone names with special characters work."""
        calc = COMCalculator()
        calc.set_bone_mass("bone.with.dots", 1.0)
        calc.set_bone_mass("bone/with/slashes", 2.0)
        calc.set_bone_mass("bone-with-dashes", 3.0)
        calc.set_bone_mass("bone_with_underscores", 4.0)
        assert calc.get_bone_mass("bone.with.dots") == 1.0
        assert calc.get_bone_mass("bone/with/slashes") == 2.0
        assert calc.get_bone_mass("bone-with-dashes") == 3.0
        assert calc.get_bone_mass("bone_with_underscores") == 4.0

    def test_empty_bone_name(self) -> None:
        """Empty string as bone name works."""
        calc = COMCalculator()
        calc.set_bone_mass("", 5.0)
        assert calc.get_bone_mass("") == 5.0

    def test_unicode_bone_names(self) -> None:
        """Unicode bone names work."""
        calc = COMCalculator()
        calc.set_bone_mass("bone_中文", 1.0)  # Chinese characters
        calc.set_bone_mass("bone_αβγ", 2.0)  # Greek letters
        assert calc.get_bone_mass("bone_中文") == 1.0
        assert calc.get_bone_mass("bone_αβγ") == 2.0

    def test_default_mass_zero(self) -> None:
        """default_mass=0 makes unconfigured bones weightless."""
        calc = COMCalculator(
            bone_masses={"configured": 1.0},
            default_mass=0.0
        )
        positions = {
            "configured": Vec3(5.0, 0.0, 0.0),
            "unconfigured": Vec3(100.0, 100.0, 100.0),
        }
        result = calc.calculate(positions)
        # Only configured bone contributes
        assert result.x == pytest.approx(5.0)
        assert result.y == pytest.approx(0.0)
        assert result.z == pytest.approx(0.0)

    def test_collinear_bones(self) -> None:
        """COM of collinear bones is on the line."""
        calc = COMCalculator(bone_masses={"a": 1.0, "b": 1.0, "c": 1.0})
        positions = {
            "a": Vec3(0.0, 0.0, 0.0),
            "b": Vec3(5.0, 5.0, 5.0),
            "c": Vec3(10.0, 10.0, 10.0),
        }
        result = calc.calculate(positions)
        # Average is (5, 5, 5)
        assert result.x == pytest.approx(5.0)
        assert result.y == pytest.approx(5.0)
        assert result.z == pytest.approx(5.0)


class TestIntegration:
    """Integration tests combining multiple methods."""

    def test_configure_then_calculate(self) -> None:
        """Full workflow: configure masses, then calculate COM."""
        calc = COMCalculator(default_mass=0.5)

        # Configure skeleton masses
        calc.set_bone_masses({
            "pelvis": 15.0,
            "spine": 10.0,
            "chest": 12.0,
            "head": 5.0,
            "arm_l": 3.0,
            "arm_r": 3.0,
            "leg_l": 8.0,
            "leg_r": 8.0,
        })

        positions = {
            "pelvis": Vec3(0.0, 1.0, 0.0),
            "spine": Vec3(0.0, 1.2, 0.0),
            "chest": Vec3(0.0, 1.5, 0.0),
            "head": Vec3(0.0, 1.8, 0.0),
            "arm_l": Vec3(-0.5, 1.3, 0.0),
            "arm_r": Vec3(0.5, 1.3, 0.0),
            "leg_l": Vec3(-0.2, 0.5, 0.0),
            "leg_r": Vec3(0.2, 0.5, 0.0),
        }

        result = calc.calculate(positions)

        # Verify reasonable result (COM should be near torso)
        assert -0.5 < result.x < 0.5
        assert 0.8 < result.y < 1.5
        assert -0.5 < result.z < 0.5

    def test_partial_vs_full_calculation(self) -> None:
        """Partial calculation excludes specified bones."""
        calc = COMCalculator(default_mass=1.0)
        positions = {
            "upper": Vec3(0.0, 10.0, 0.0),
            "lower": Vec3(0.0, 0.0, 0.0),
        }

        full_com = calc.calculate(positions)
        upper_com = calc.calculate_partial(positions, {"upper"})
        lower_com = calc.calculate_partial(positions, {"lower"})

        # Full COM should be between partial COMs
        assert full_com.y == pytest.approx(5.0)  # Average
        assert upper_com.y == pytest.approx(10.0)
        assert lower_com.y == pytest.approx(0.0)

    def test_transforms_matches_positions(self) -> None:
        """calculate_from_transforms produces same result as calculate."""
        calc = COMCalculator(bone_masses={"a": 2.0, "b": 3.0, "c": 5.0})

        positions = {
            "a": Vec3(1.0, 2.0, 3.0),
            "b": Vec3(4.0, 5.0, 6.0),
            "c": Vec3(7.0, 8.0, 9.0),
        }

        transforms = [
            Transform(translation=positions["a"]),
            Transform(translation=positions["b"]),
            Transform(translation=positions["c"]),
        ]
        bone_names = ["a", "b", "c"]

        result_positions = calc.calculate(positions)
        result_transforms = calc.calculate_from_transforms(transforms, bone_names)

        assert result_positions.x == pytest.approx(result_transforms.x)
        assert result_positions.y == pytest.approx(result_transforms.y)
        assert result_positions.z == pytest.approx(result_transforms.z)

    def test_modify_masses_recalculates(self) -> None:
        """Modifying masses affects subsequent calculations."""
        calc = COMCalculator()
        positions = {
            "a": Vec3(0.0, 0.0, 0.0),
            "b": Vec3(10.0, 0.0, 0.0),
        }

        # First calculation with default masses
        result1 = calc.calculate(positions)
        assert result1.x == pytest.approx(5.0)  # Average

        # Change mass of 'a'
        calc.set_bone_mass("a", 9.0)
        result2 = calc.calculate(positions)
        # (9*0 + 1*10) / 10 = 1.0
        assert result2.x == pytest.approx(1.0)


class TestDataclassBehavior:
    """Tests for dataclass-specific behavior."""

    def test_default_factory_independence(self) -> None:
        """Each instance has independent bone_masses dict."""
        calc1 = COMCalculator()
        calc2 = COMCalculator()

        calc1.set_bone_mass("bone", 5.0)

        assert "bone" in calc1.bone_masses
        assert "bone" not in calc2.bone_masses

    def test_explicit_bone_masses_shared(self) -> None:
        """Explicit bone_masses dict is shared if passed by reference."""
        shared_masses = {"bone": 1.0}
        calc1 = COMCalculator(bone_masses=shared_masses)
        calc2 = COMCalculator(bone_masses=shared_masses)

        calc1.set_bone_mass("new_bone", 2.0)

        # Both calculators share the same dict
        assert "new_bone" in calc2.bone_masses

    def test_copy_behavior(self) -> None:
        """Copy bone_masses to avoid shared state if needed."""
        original_masses = {"bone": 1.0}
        calc = COMCalculator(bone_masses=original_masses.copy())

        calc.set_bone_mass("new_bone", 2.0)

        # Original dict is unaffected
        assert "new_bone" not in original_masses


class TestNumericalStability:
    """Tests for numerical stability."""

    def test_many_small_masses(self) -> None:
        """Sum of many small masses doesn't lose precision."""
        calc = COMCalculator()
        n = 1000
        for i in range(n):
            calc.set_bone_mass(f"bone_{i}", 0.001)

        positions = {f"bone_{i}": Vec3(float(i), 0.0, 0.0) for i in range(n)}
        result = calc.calculate(positions)

        # Expected x = (0+1+...+999)/1000 = 499.5
        assert result.x == pytest.approx(499.5, rel=1e-6)

    def test_mixed_mass_magnitudes(self) -> None:
        """Handle bones with vastly different mass magnitudes."""
        calc = COMCalculator(bone_masses={
            "massive": 1e6,
            "tiny": 1e-6,
        })
        positions = {
            "massive": Vec3(0.0, 0.0, 0.0),
            "tiny": Vec3(1e6, 0.0, 0.0),
        }
        result = calc.calculate(positions)
        # Massive bone dominates, result should be near (0, 0, 0)
        assert result.x == pytest.approx(1.0, abs=1.0)
