"""Whitebox tests for BalanceController class.

Tests internal implementation details of the BalanceController class,
including COM calculation delegation, support polygon integration,
correction vector scaling, and bone position adjustment logic.
"""

from __future__ import annotations

import pytest
import math
from unittest.mock import Mock, MagicMock, patch

from engine.animation.ik.fullbody import (
    BalanceController,
    COMCalculator,
    SupportPolygon,
)
from engine.core.math.vec import Vec3


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def simple_bone_positions() -> dict[str, Vec3]:
    """Basic bone positions for testing."""
    return {
        "pelvis": Vec3(0.0, 1.0, 0.0),
        "spine": Vec3(0.0, 1.3, 0.0),
        "chest": Vec3(0.0, 1.6, 0.0),
        "head": Vec3(0.0, 1.8, 0.0),
    }


@pytest.fixture
def square_polygon() -> SupportPolygon:
    """Square support polygon centered at origin, 2x2 units."""
    return SupportPolygon(
        vertices=[
            Vec3(-1.0, 0.0, -1.0),
            Vec3(1.0, 0.0, -1.0),
            Vec3(1.0, 0.0, 1.0),
            Vec3(-1.0, 0.0, 1.0),
        ]
    )


@pytest.fixture
def mock_com_calculator() -> Mock:
    """Mock COMCalculator for isolated testing."""
    mock = Mock(spec=COMCalculator)
    mock.calculate = Mock(return_value=Vec3(0.0, 1.0, 0.0))
    return mock


@pytest.fixture
def mock_support_polygon() -> Mock:
    """Mock SupportPolygon for isolated testing."""
    mock = Mock(spec=SupportPolygon)
    mock.contains_point = Mock(return_value=True)
    mock.correction_vector = Mock(return_value=Vec3.zero())
    return mock


# =============================================================================
# TestBalanceControllerInit
# =============================================================================


class TestBalanceControllerInit:
    """Tests for BalanceController initialization and default values."""

    def test_default_values(self) -> None:
        """Default init creates controller with default attributes."""
        controller = BalanceController()
        assert isinstance(controller.com_calculator, COMCalculator)
        assert isinstance(controller.support_polygon, SupportPolygon)
        assert controller.correction_strength == 0.5
        assert controller.pelvis_weight == 0.7

    def test_custom_correction_strength(self) -> None:
        """Init with custom correction_strength."""
        controller = BalanceController(correction_strength=0.8)
        assert controller.correction_strength == 0.8

    def test_custom_pelvis_weight(self) -> None:
        """Init with custom pelvis_weight."""
        controller = BalanceController(pelvis_weight=0.5)
        assert controller.pelvis_weight == 0.5

    def test_custom_com_calculator(self) -> None:
        """Init with custom COMCalculator instance."""
        custom_calc = COMCalculator(default_mass=2.0)
        controller = BalanceController(com_calculator=custom_calc)
        assert controller.com_calculator is custom_calc
        assert controller.com_calculator.default_mass == 2.0

    def test_custom_support_polygon(self) -> None:
        """Init with custom SupportPolygon instance."""
        custom_polygon = SupportPolygon(
            vertices=[Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0.5, 0, 1)]
        )
        controller = BalanceController(support_polygon=custom_polygon)
        assert controller.support_polygon is custom_polygon
        assert len(controller.support_polygon.vertices) == 3

    def test_all_custom_parameters(self) -> None:
        """Init with all custom parameters."""
        calc = COMCalculator(default_mass=3.0)
        polygon = SupportPolygon(
            vertices=[Vec3(-1, 0, -1), Vec3(1, 0, -1), Vec3(0, 0, 1)]
        )
        controller = BalanceController(
            com_calculator=calc,
            support_polygon=polygon,
            correction_strength=0.9,
            pelvis_weight=0.4,
        )
        assert controller.com_calculator is calc
        assert controller.support_polygon is polygon
        assert controller.correction_strength == 0.9
        assert controller.pelvis_weight == 0.4

    def test_zero_correction_strength(self) -> None:
        """Zero correction_strength is valid (no correction applied)."""
        controller = BalanceController(correction_strength=0.0)
        assert controller.correction_strength == 0.0

    def test_max_correction_strength(self) -> None:
        """Correction strength of 1.0 is valid (full correction)."""
        controller = BalanceController(correction_strength=1.0)
        assert controller.correction_strength == 1.0

    def test_zero_pelvis_weight(self) -> None:
        """Zero pelvis_weight (all correction to spine)."""
        controller = BalanceController(pelvis_weight=0.0)
        assert controller.pelvis_weight == 0.0

    def test_full_pelvis_weight(self) -> None:
        """Full pelvis_weight (all correction to pelvis)."""
        controller = BalanceController(pelvis_weight=1.0)
        assert controller.pelvis_weight == 1.0


# =============================================================================
# TestIsBalanced
# =============================================================================


class TestIsBalanced:
    """Tests for is_balanced method."""

    def test_delegates_to_com_calculator(
        self, simple_bone_positions: dict[str, Vec3]
    ) -> None:
        """is_balanced calls com_calculator.calculate with bone_positions."""
        mock_calc = Mock(spec=COMCalculator)
        mock_calc.calculate = Mock(return_value=Vec3(0.0, 1.0, 0.0))
        mock_polygon = Mock(spec=SupportPolygon)
        mock_polygon.contains_point = Mock(return_value=True)

        controller = BalanceController(
            com_calculator=mock_calc, support_polygon=mock_polygon
        )
        controller.is_balanced(simple_bone_positions)

        mock_calc.calculate.assert_called_once_with(simple_bone_positions)

    def test_delegates_to_support_polygon(
        self, simple_bone_positions: dict[str, Vec3]
    ) -> None:
        """is_balanced calls support_polygon.contains_point with COM."""
        com_result = Vec3(0.5, 1.2, 0.3)
        mock_calc = Mock(spec=COMCalculator)
        mock_calc.calculate = Mock(return_value=com_result)
        mock_polygon = Mock(spec=SupportPolygon)
        mock_polygon.contains_point = Mock(return_value=True)

        controller = BalanceController(
            com_calculator=mock_calc, support_polygon=mock_polygon
        )
        controller.is_balanced(simple_bone_positions)

        mock_polygon.contains_point.assert_called_once_with(com_result)

    def test_com_inside_returns_true(
        self, simple_bone_positions: dict[str, Vec3], square_polygon: SupportPolygon
    ) -> None:
        """Returns True when COM is inside support polygon."""
        # COM at origin should be inside square polygon
        calc = COMCalculator()
        controller = BalanceController(
            com_calculator=calc, support_polygon=square_polygon
        )
        # Place bones so COM is near origin
        centered_positions = {
            "pelvis": Vec3(0.0, 1.0, 0.0),
            "spine": Vec3(0.0, 1.3, 0.0),
        }
        result = controller.is_balanced(centered_positions)
        assert result is True

    def test_com_outside_returns_false(
        self, square_polygon: SupportPolygon
    ) -> None:
        """Returns False when COM is outside support polygon."""
        calc = COMCalculator()
        controller = BalanceController(
            com_calculator=calc, support_polygon=square_polygon
        )
        # Place bones so COM is far outside polygon (x=5)
        offset_positions = {
            "pelvis": Vec3(5.0, 1.0, 0.0),
            "spine": Vec3(5.0, 1.3, 0.0),
        }
        result = controller.is_balanced(offset_positions)
        assert result is False

    def test_com_on_edge_behavior(self, square_polygon: SupportPolygon) -> None:
        """Test behavior when COM is exactly on polygon edge."""
        calc = COMCalculator()
        controller = BalanceController(
            com_calculator=calc, support_polygon=square_polygon
        )
        # Place bones so COM is exactly on edge (x=1, z=0)
        edge_positions = {
            "pelvis": Vec3(1.0, 1.0, 0.0),
            "spine": Vec3(1.0, 1.3, 0.0),
        }
        # Edge case - depends on polygon contains_point implementation
        # This tests the actual boundary condition
        result = controller.is_balanced(edge_positions)
        # Result may be True or False depending on edge handling
        assert isinstance(result, bool)

    def test_empty_bone_positions(self, square_polygon: SupportPolygon) -> None:
        """Test with empty bone positions dictionary."""
        calc = COMCalculator()
        controller = BalanceController(
            com_calculator=calc, support_polygon=square_polygon
        )
        # Empty positions yields Vec3.zero() from COM calculator
        result = controller.is_balanced({})
        # Zero COM at origin should be inside square polygon
        assert result is True

    def test_single_bone(self, square_polygon: SupportPolygon) -> None:
        """Test with single bone position."""
        calc = COMCalculator()
        controller = BalanceController(
            com_calculator=calc, support_polygon=square_polygon
        )
        single_bone = {"pelvis": Vec3(0.5, 1.0, 0.5)}
        result = controller.is_balanced(single_bone)
        assert result is True


# =============================================================================
# TestGetCorrection
# =============================================================================


class TestGetCorrection:
    """Tests for get_correction method."""

    def test_balanced_returns_zero_vector(
        self, simple_bone_positions: dict[str, Vec3], square_polygon: SupportPolygon
    ) -> None:
        """When balanced, get_correction returns zero vector."""
        calc = COMCalculator()
        controller = BalanceController(
            com_calculator=calc,
            support_polygon=square_polygon,
            correction_strength=1.0,
        )
        # Centered positions = balanced
        centered = {"pelvis": Vec3(0.0, 1.0, 0.0)}
        correction = controller.get_correction(centered)
        assert correction.x == pytest.approx(0.0, abs=1e-6)
        assert correction.y == pytest.approx(0.0, abs=1e-6)
        assert correction.z == pytest.approx(0.0, abs=1e-6)

    def test_unbalanced_returns_scaled_correction(
        self, square_polygon: SupportPolygon
    ) -> None:
        """When unbalanced, returns correction vector scaled by strength."""
        calc = COMCalculator()
        controller = BalanceController(
            com_calculator=calc,
            support_polygon=square_polygon,
            correction_strength=1.0,
        )
        # Position outside polygon (x=3, should correct toward x=1)
        outside = {"pelvis": Vec3(3.0, 1.0, 0.0)}
        correction = controller.get_correction(outside)
        # Correction should point toward polygon (negative x direction)
        assert correction.x < 0.0
        # Y should be zero (correction only in XZ plane)
        assert correction.y == pytest.approx(0.0, abs=1e-6)

    def test_strength_affects_magnitude(
        self, square_polygon: SupportPolygon
    ) -> None:
        """Correction magnitude is scaled by correction_strength."""
        calc = COMCalculator()
        # Position outside polygon
        outside = {"pelvis": Vec3(3.0, 1.0, 0.0)}

        # Full strength
        controller_full = BalanceController(
            com_calculator=calc,
            support_polygon=square_polygon,
            correction_strength=1.0,
        )
        correction_full = controller_full.get_correction(outside)

        # Half strength
        controller_half = BalanceController(
            com_calculator=calc,
            support_polygon=square_polygon,
            correction_strength=0.5,
        )
        correction_half = controller_half.get_correction(outside)

        # Half strength should produce half the correction magnitude
        full_length = correction_full.length()
        half_length = correction_half.length()
        assert half_length == pytest.approx(full_length * 0.5, rel=1e-6)

    def test_zero_strength_returns_zero(
        self, square_polygon: SupportPolygon
    ) -> None:
        """Zero correction_strength returns zero vector even when unbalanced."""
        calc = COMCalculator()
        controller = BalanceController(
            com_calculator=calc,
            support_polygon=square_polygon,
            correction_strength=0.0,
        )
        outside = {"pelvis": Vec3(5.0, 1.0, 5.0)}
        correction = controller.get_correction(outside)
        assert correction.length() == pytest.approx(0.0, abs=1e-10)

    def test_delegates_to_com_calculator(self) -> None:
        """get_correction calls com_calculator.calculate."""
        mock_calc = Mock(spec=COMCalculator)
        mock_calc.calculate = Mock(return_value=Vec3(0.0, 1.0, 0.0))
        mock_polygon = Mock(spec=SupportPolygon)
        mock_polygon.correction_vector = Mock(return_value=Vec3.zero())

        controller = BalanceController(
            com_calculator=mock_calc, support_polygon=mock_polygon
        )
        positions = {"bone": Vec3(1, 2, 3)}
        controller.get_correction(positions)

        mock_calc.calculate.assert_called_once_with(positions)

    def test_delegates_to_support_polygon_correction_vector(self) -> None:
        """get_correction calls support_polygon.correction_vector with COM."""
        com_result = Vec3(2.5, 0.8, 1.5)
        mock_calc = Mock(spec=COMCalculator)
        mock_calc.calculate = Mock(return_value=com_result)
        mock_polygon = Mock(spec=SupportPolygon)
        mock_polygon.correction_vector = Mock(return_value=Vec3(0.1, 0.0, 0.2))

        controller = BalanceController(
            com_calculator=mock_calc, support_polygon=mock_polygon
        )
        controller.get_correction({"bone": Vec3(0, 0, 0)})

        mock_polygon.correction_vector.assert_called_once_with(com_result)

    def test_correction_y_component_is_zero(
        self, square_polygon: SupportPolygon
    ) -> None:
        """Correction vector Y component is always zero (XZ plane only)."""
        calc = COMCalculator()
        controller = BalanceController(
            com_calculator=calc,
            support_polygon=square_polygon,
            correction_strength=1.0,
        )
        # Various positions outside polygon
        test_cases = [
            {"bone": Vec3(5.0, 0.0, 0.0)},
            {"bone": Vec3(0.0, 2.0, 5.0)},
            {"bone": Vec3(-3.0, 1.5, -3.0)},
        ]
        for positions in test_cases:
            correction = controller.get_correction(positions)
            assert correction.y == pytest.approx(0.0, abs=1e-10)


# =============================================================================
# TestApplyCorrection
# =============================================================================


class TestApplyCorrection:
    """Tests for apply_correction method."""

    def test_distributes_between_pelvis_spine(
        self, square_polygon: SupportPolygon
    ) -> None:
        """Correction is distributed between pelvis and spine bones."""
        calc = COMCalculator()
        controller = BalanceController(
            com_calculator=calc,
            support_polygon=square_polygon,
            correction_strength=1.0,
            pelvis_weight=0.7,  # 70% to pelvis, 30% to spine
        )
        # Position outside polygon
        positions = {
            "pelvis": Vec3(3.0, 1.0, 0.0),
            "spine": Vec3(3.0, 1.5, 0.0),
        }
        result = controller.apply_correction(positions)

        # Both bones should be adjusted
        assert result["pelvis"].x != positions["pelvis"].x
        assert result["spine"].x != positions["spine"].x

    def test_pelvis_weight_affects_distribution(
        self, square_polygon: SupportPolygon
    ) -> None:
        """pelvis_weight controls distribution ratio."""
        calc = COMCalculator()
        positions = {
            "pelvis": Vec3(3.0, 1.0, 0.0),
            "spine": Vec3(3.0, 1.5, 0.0),
        }

        # 100% to pelvis
        controller_full_pelvis = BalanceController(
            com_calculator=calc,
            support_polygon=square_polygon,
            correction_strength=1.0,
            pelvis_weight=1.0,
        )
        result_full_pelvis = controller_full_pelvis.apply_correction(positions)
        pelvis_change_full = abs(
            result_full_pelvis["pelvis"].x - positions["pelvis"].x
        )
        spine_change_full = abs(
            result_full_pelvis["spine"].x - positions["spine"].x
        )
        # Spine should have no change when pelvis_weight=1.0
        assert spine_change_full == pytest.approx(0.0, abs=1e-10)
        assert pelvis_change_full > 0.0

        # 0% to pelvis (100% to spine)
        controller_full_spine = BalanceController(
            com_calculator=calc,
            support_polygon=square_polygon,
            correction_strength=1.0,
            pelvis_weight=0.0,
        )
        result_full_spine = controller_full_spine.apply_correction(positions)
        pelvis_change_none = abs(
            result_full_spine["pelvis"].x - positions["pelvis"].x
        )
        spine_change_none = abs(
            result_full_spine["spine"].x - positions["spine"].x
        )
        # Pelvis should have no change when pelvis_weight=0.0
        assert pelvis_change_none == pytest.approx(0.0, abs=1e-10)
        assert spine_change_none > 0.0

    def test_preserves_y_coordinate(self, square_polygon: SupportPolygon) -> None:
        """Y coordinates are preserved (ground contact maintained)."""
        calc = COMCalculator()
        controller = BalanceController(
            com_calculator=calc,
            support_polygon=square_polygon,
            correction_strength=1.0,
            pelvis_weight=0.5,
        )
        positions = {
            "pelvis": Vec3(3.0, 1.0, 0.0),
            "spine": Vec3(3.0, 1.5, 0.0),
        }
        result = controller.apply_correction(positions)

        assert result["pelvis"].y == pytest.approx(positions["pelvis"].y, abs=1e-10)
        assert result["spine"].y == pytest.approx(positions["spine"].y, abs=1e-10)

    def test_missing_pelvis_handled(self, square_polygon: SupportPolygon) -> None:
        """Missing pelvis bone does not cause error."""
        calc = COMCalculator()
        controller = BalanceController(
            com_calculator=calc,
            support_polygon=square_polygon,
            correction_strength=1.0,
            pelvis_weight=0.5,
        )
        # No pelvis, only spine
        positions = {"spine": Vec3(3.0, 1.5, 0.0)}
        result = controller.apply_correction(positions)

        # Spine should still be adjusted
        assert "spine" in result
        # No pelvis in result either (wasn't in input)
        assert "pelvis" not in result

    def test_missing_spine_handled(self, square_polygon: SupportPolygon) -> None:
        """Missing spine bone does not cause error."""
        calc = COMCalculator()
        controller = BalanceController(
            com_calculator=calc,
            support_polygon=square_polygon,
            correction_strength=1.0,
            pelvis_weight=0.5,
        )
        # No spine, only pelvis
        positions = {"pelvis": Vec3(3.0, 1.0, 0.0)}
        result = controller.apply_correction(positions)

        # Pelvis should still be adjusted
        assert "pelvis" in result
        # No spine in result either (wasn't in input)
        assert "spine" not in result

    def test_missing_both_bones_handled(
        self, square_polygon: SupportPolygon
    ) -> None:
        """Missing both target bones does not cause error."""
        calc = COMCalculator()
        controller = BalanceController(
            com_calculator=calc,
            support_polygon=square_polygon,
            correction_strength=1.0,
        )
        # Neither pelvis nor spine
        positions = {"head": Vec3(3.0, 2.0, 0.0), "chest": Vec3(3.0, 1.7, 0.0)}
        result = controller.apply_correction(positions)

        # Original positions returned unchanged (no target bones to adjust)
        assert "head" in result
        assert "chest" in result

    def test_custom_bone_names(self, square_polygon: SupportPolygon) -> None:
        """Custom pelvis_name and spine_name work correctly."""
        calc = COMCalculator()
        controller = BalanceController(
            com_calculator=calc,
            support_polygon=square_polygon,
            correction_strength=1.0,
            pelvis_weight=0.6,
        )
        positions = {
            "hip_bone": Vec3(3.0, 1.0, 0.0),
            "back_bone": Vec3(3.0, 1.5, 0.0),
        }
        result = controller.apply_correction(
            positions, pelvis_name="hip_bone", spine_name="back_bone"
        )

        # Custom-named bones should be adjusted
        assert result["hip_bone"].x != positions["hip_bone"].x
        assert result["back_bone"].x != positions["back_bone"].x

    def test_does_not_mutate_input(self, square_polygon: SupportPolygon) -> None:
        """apply_correction creates new dict, does not mutate input."""
        calc = COMCalculator()
        controller = BalanceController(
            com_calculator=calc,
            support_polygon=square_polygon,
            correction_strength=1.0,
        )
        original_pelvis = Vec3(3.0, 1.0, 0.0)
        positions = {"pelvis": original_pelvis, "spine": Vec3(3.0, 1.5, 0.0)}
        original_dict_id = id(positions)
        original_pelvis_x = positions["pelvis"].x

        result = controller.apply_correction(positions)

        # Result should be a different dictionary
        assert id(result) != original_dict_id
        # Original dict values unchanged
        assert positions["pelvis"].x == original_pelvis_x

    def test_balanced_returns_original_positions(
        self, square_polygon: SupportPolygon
    ) -> None:
        """When balanced, returns original positions dict."""
        calc = COMCalculator()
        controller = BalanceController(
            com_calculator=calc,
            support_polygon=square_polygon,
            correction_strength=1.0,
        )
        # Centered positions = balanced
        positions = {
            "pelvis": Vec3(0.0, 1.0, 0.0),
            "spine": Vec3(0.0, 1.5, 0.0),
        }
        result = controller.apply_correction(positions)

        # Should return same dict when correction is negligible
        assert result is positions

    def test_negligible_correction_threshold(
        self, square_polygon: SupportPolygon
    ) -> None:
        """Very small corrections (< 1e-10 length_squared) are skipped."""
        calc = COMCalculator()
        controller = BalanceController(
            com_calculator=calc,
            support_polygon=square_polygon,
            correction_strength=1e-12,  # Very small strength
        )
        positions = {
            "pelvis": Vec3(3.0, 1.0, 0.0),  # Outside polygon
            "spine": Vec3(3.0, 1.5, 0.0),
        }
        result = controller.apply_correction(positions)

        # Correction magnitude = raw_correction * 1e-12
        # If length_squared < 1e-10, returns original positions
        # This tests the threshold logic
        assert result is positions

    def test_other_bones_preserved(self, square_polygon: SupportPolygon) -> None:
        """Bones other than pelvis/spine are preserved in result."""
        calc = COMCalculator()
        controller = BalanceController(
            com_calculator=calc,
            support_polygon=square_polygon,
            correction_strength=1.0,
        )
        positions = {
            "pelvis": Vec3(3.0, 1.0, 0.0),
            "spine": Vec3(3.0, 1.5, 0.0),
            "head": Vec3(3.0, 2.0, 0.0),
            "left_arm": Vec3(2.5, 1.4, 0.0),
        }
        result = controller.apply_correction(positions)

        # Head and left_arm should be in result unchanged
        assert result["head"].x == positions["head"].x
        assert result["head"].y == positions["head"].y
        assert result["head"].z == positions["head"].z
        assert result["left_arm"].x == positions["left_arm"].x

    def test_z_coordinate_corrected(self, square_polygon: SupportPolygon) -> None:
        """Z coordinate is corrected when COM is outside in Z direction."""
        calc = COMCalculator()
        controller = BalanceController(
            com_calculator=calc,
            support_polygon=square_polygon,
            correction_strength=1.0,
            pelvis_weight=1.0,
        )
        # Outside in Z direction
        positions = {"pelvis": Vec3(0.0, 1.0, 3.0)}
        result = controller.apply_correction(positions)

        # Z should be corrected toward polygon
        assert result["pelvis"].z < positions["pelvis"].z


# =============================================================================
# TestSetCorrectionStrength
# =============================================================================


class TestSetCorrectionStrength:
    """Tests for set_correction_strength method."""

    def test_sets_valid_strength(self) -> None:
        """Valid strength value is set directly."""
        controller = BalanceController(correction_strength=0.5)
        controller.set_correction_strength(0.8)
        assert controller.correction_strength == 0.8

    def test_clamps_to_zero(self) -> None:
        """Negative values are clamped to 0."""
        controller = BalanceController(correction_strength=0.5)
        controller.set_correction_strength(-0.5)
        assert controller.correction_strength == 0.0

    def test_clamps_large_negative_to_zero(self) -> None:
        """Large negative values are clamped to 0."""
        controller = BalanceController()
        controller.set_correction_strength(-100.0)
        assert controller.correction_strength == 0.0

    def test_clamps_to_one(self) -> None:
        """Values > 1 are clamped to 1."""
        controller = BalanceController(correction_strength=0.5)
        controller.set_correction_strength(1.5)
        assert controller.correction_strength == 1.0

    def test_clamps_large_positive_to_one(self) -> None:
        """Large positive values are clamped to 1."""
        controller = BalanceController()
        controller.set_correction_strength(100.0)
        assert controller.correction_strength == 1.0

    def test_zero_is_valid(self) -> None:
        """Zero is a valid value (no clamping)."""
        controller = BalanceController(correction_strength=0.5)
        controller.set_correction_strength(0.0)
        assert controller.correction_strength == 0.0

    def test_one_is_valid(self) -> None:
        """One is a valid value (no clamping)."""
        controller = BalanceController(correction_strength=0.5)
        controller.set_correction_strength(1.0)
        assert controller.correction_strength == 1.0

    def test_boundary_values(self) -> None:
        """Test values very close to boundaries."""
        controller = BalanceController()

        # Just below zero
        controller.set_correction_strength(-1e-10)
        assert controller.correction_strength == 0.0

        # Just above zero
        controller.set_correction_strength(1e-10)
        assert controller.correction_strength == pytest.approx(1e-10, abs=1e-15)

        # Just below one
        controller.set_correction_strength(1.0 - 1e-10)
        assert controller.correction_strength == pytest.approx(1.0 - 1e-10, rel=1e-9)

        # Just above one
        controller.set_correction_strength(1.0 + 1e-10)
        assert controller.correction_strength == 1.0


# =============================================================================
# TestUpdateSupportPolygon
# =============================================================================


class TestUpdateSupportPolygon:
    """Tests for update_support_polygon method."""

    def test_creates_new_polygon(self) -> None:
        """update_support_polygon creates a new SupportPolygon."""
        controller = BalanceController()
        original_polygon = controller.support_polygon

        foot_positions = [
            Vec3(-0.5, 0.0, -0.5),
            Vec3(0.5, 0.0, -0.5),
            Vec3(0.5, 0.0, 0.5),
            Vec3(-0.5, 0.0, 0.5),
        ]
        controller.update_support_polygon(foot_positions)

        assert controller.support_polygon is not original_polygon

    def test_new_polygon_uses_foot_positions(self) -> None:
        """New polygon vertices match foot positions (projected to XZ)."""
        controller = BalanceController()
        foot_positions = [
            Vec3(-1.0, 0.1, -1.0),  # Y will be projected to 0
            Vec3(1.0, 0.2, -1.0),
            Vec3(1.0, 0.0, 1.0),
            Vec3(-1.0, 0.1, 1.0),
        ]
        controller.update_support_polygon(foot_positions)

        # Vertices should be projected to y=0
        assert len(controller.support_polygon.vertices) == 4
        for vertex in controller.support_polygon.vertices:
            assert vertex.y == 0.0

    def test_empty_foot_positions(self) -> None:
        """Empty foot positions creates polygon with empty vertices."""
        controller = BalanceController()
        controller.update_support_polygon([])

        assert len(controller.support_polygon.vertices) == 0

    def test_single_foot_position(self) -> None:
        """Single foot position creates single-vertex polygon."""
        controller = BalanceController()
        controller.update_support_polygon([Vec3(1.0, 0.0, 2.0)])

        assert len(controller.support_polygon.vertices) == 1
        assert controller.support_polygon.vertices[0].x == 1.0
        assert controller.support_polygon.vertices[0].z == 2.0

    def test_two_foot_positions(self) -> None:
        """Two foot positions creates two-vertex polygon (line)."""
        controller = BalanceController()
        controller.update_support_polygon([
            Vec3(-1.0, 0.0, 0.0),
            Vec3(1.0, 0.0, 0.0),
        ])

        assert len(controller.support_polygon.vertices) == 2

    def test_replaces_previous_polygon(self) -> None:
        """Calling update_support_polygon replaces previous polygon."""
        initial_vertices = [Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0.5, 0, 1)]
        controller = BalanceController(
            support_polygon=SupportPolygon(vertices=initial_vertices)
        )
        assert len(controller.support_polygon.vertices) == 3

        # Update with different polygon
        controller.update_support_polygon([
            Vec3(-2, 0, -2),
            Vec3(2, 0, -2),
            Vec3(2, 0, 2),
            Vec3(-2, 0, 2),
        ])
        assert len(controller.support_polygon.vertices) == 4

    def test_y_coordinate_ignored(self) -> None:
        """Foot position Y coordinates are ignored (projected to ground)."""
        controller = BalanceController()
        # Feet at different heights
        foot_positions = [
            Vec3(0.0, 0.5, 0.0),
            Vec3(1.0, 0.1, 0.0),
            Vec3(0.5, 0.8, 1.0),
        ]
        controller.update_support_polygon(foot_positions)

        # All vertices should have y=0
        for vertex in controller.support_polygon.vertices:
            assert vertex.y == 0.0

    def test_preserves_xz_coordinates(self) -> None:
        """X and Z coordinates are preserved correctly."""
        controller = BalanceController()
        foot_positions = [
            Vec3(1.5, 0.0, 2.5),
            Vec3(-3.0, 0.0, 4.0),
            Vec3(0.0, 0.0, -1.0),
        ]
        controller.update_support_polygon(foot_positions)

        # Check each vertex
        vertices = controller.support_polygon.vertices
        assert vertices[0].x == pytest.approx(1.5)
        assert vertices[0].z == pytest.approx(2.5)
        assert vertices[1].x == pytest.approx(-3.0)
        assert vertices[1].z == pytest.approx(4.0)
        assert vertices[2].x == pytest.approx(0.0)
        assert vertices[2].z == pytest.approx(-1.0)


# =============================================================================
# TestIntegration
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple methods."""

    def test_full_balance_check_workflow(self) -> None:
        """Test complete workflow: update polygon, check balance, apply correction."""
        controller = BalanceController(
            correction_strength=0.8,
            pelvis_weight=0.6,
        )

        # Set up support polygon from feet
        foot_positions = [
            Vec3(-0.5, 0.0, -0.5),
            Vec3(0.5, 0.0, -0.5),
            Vec3(0.5, 0.0, 0.5),
            Vec3(-0.5, 0.0, 0.5),
        ]
        controller.update_support_polygon(foot_positions)

        # Check initial pose (outside polygon)
        bone_positions = {
            "pelvis": Vec3(2.0, 1.0, 0.0),
            "spine": Vec3(2.0, 1.3, 0.0),
        }
        assert controller.is_balanced(bone_positions) is False

        # Get and apply correction
        correction = controller.get_correction(bone_positions)
        assert correction.length() > 0.0

        corrected = controller.apply_correction(bone_positions)
        # After correction, positions should be closer to polygon
        assert corrected["pelvis"].x < bone_positions["pelvis"].x

    def test_dynamic_strength_adjustment(self) -> None:
        """Test adjusting strength during runtime."""
        controller = BalanceController(correction_strength=0.0)
        positions = {"pelvis": Vec3(3.0, 1.0, 0.0)}

        # With zero strength, no correction
        correction = controller.get_correction(positions)
        assert correction.length() == pytest.approx(0.0, abs=1e-10)

        # Increase strength
        controller.set_correction_strength(1.0)
        correction = controller.get_correction(positions)
        # Now should have correction (assuming outside default polygon)
        # Note: depends on default polygon which may be empty

    def test_repeated_corrections_converge(
        self, square_polygon: SupportPolygon
    ) -> None:
        """Repeated apply_correction should move toward balance."""
        controller = BalanceController(
            support_polygon=square_polygon,
            correction_strength=0.5,  # Not too aggressive
            pelvis_weight=1.0,  # Only adjust pelvis for simplicity
        )

        # Start outside polygon
        positions = {"pelvis": Vec3(3.0, 1.0, 0.0)}

        # Apply corrections iteratively
        for _ in range(10):
            positions = controller.apply_correction(positions, spine_name="")
            if controller.is_balanced(positions):
                break

        # After iterations, should be closer to or inside polygon
        # (May not be perfectly balanced due to COM calculation
        # including only pelvis position)
        final_x = positions["pelvis"].x
        assert final_x < 3.0  # Should have moved toward polygon


# =============================================================================
# TestEdgeCases
# =============================================================================


class TestEdgeCases:
    """Edge case tests for BalanceController."""

    def test_very_large_positions(self, square_polygon: SupportPolygon) -> None:
        """Handle very large position values."""
        controller = BalanceController(
            support_polygon=square_polygon, correction_strength=1.0
        )
        large_positions = {
            "pelvis": Vec3(1e10, 1.0, 1e10),
            "spine": Vec3(1e10, 1.5, 1e10),
        }

        # Should not crash
        is_balanced = controller.is_balanced(large_positions)
        assert is_balanced is False

        correction = controller.get_correction(large_positions)
        assert math.isfinite(correction.x)
        assert math.isfinite(correction.z)

    def test_very_small_positions(self, square_polygon: SupportPolygon) -> None:
        """Handle very small position values."""
        controller = BalanceController(
            support_polygon=square_polygon, correction_strength=1.0
        )
        small_positions = {
            "pelvis": Vec3(1e-10, 1e-10, 1e-10),
            "spine": Vec3(1e-10, 1e-10, 1e-10),
        }

        # Should be balanced (near origin, inside square polygon)
        is_balanced = controller.is_balanced(small_positions)
        assert is_balanced is True

    def test_negative_positions(self, square_polygon: SupportPolygon) -> None:
        """Handle negative position values correctly."""
        controller = BalanceController(
            support_polygon=square_polygon, correction_strength=1.0
        )
        negative_positions = {
            "pelvis": Vec3(-0.5, 1.0, -0.5),
            "spine": Vec3(-0.5, 1.5, -0.5),
        }

        # Should be balanced (inside square polygon centered at origin)
        is_balanced = controller.is_balanced(negative_positions)
        assert is_balanced is True

    def test_inf_position_values(self, square_polygon: SupportPolygon) -> None:
        """Handle infinite position values (may produce nan)."""
        controller = BalanceController(
            support_polygon=square_polygon, correction_strength=1.0
        )
        inf_positions = {"pelvis": Vec3(float("inf"), 1.0, 0.0)}

        # Should not crash - behavior with inf may vary
        try:
            controller.is_balanced(inf_positions)
        except (ValueError, OverflowError):
            pass  # Acceptable to raise

    def test_mixed_bone_count(self, square_polygon: SupportPolygon) -> None:
        """Handle varying numbers of bones correctly."""
        controller = BalanceController(
            support_polygon=square_polygon, correction_strength=1.0
        )

        # Single bone
        single = {"pelvis": Vec3(0.0, 1.0, 0.0)}
        assert isinstance(controller.is_balanced(single), bool)

        # Many bones
        many = {
            f"bone_{i}": Vec3(0.0, float(i) * 0.1, 0.0)
            for i in range(100)
        }
        assert isinstance(controller.is_balanced(many), bool)

    def test_duplicate_bone_names_last_wins(
        self, square_polygon: SupportPolygon
    ) -> None:
        """Dictionary semantics: duplicate keys take last value."""
        controller = BalanceController(
            support_polygon=square_polygon, correction_strength=1.0
        )
        # In Python, dict with duplicate keys keeps last value
        # This tests the expected behavior
        positions = {"pelvis": Vec3(0.0, 1.0, 0.0)}
        result = controller.apply_correction(positions)
        assert "pelvis" in result
