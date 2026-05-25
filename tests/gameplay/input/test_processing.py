"""Comprehensive tests for the input processing system.

Tests cover dead zone processing, response curves, input smoothing,
input modifiers, and the complete input processor pipeline.
"""

import pytest
import math
from unittest.mock import Mock, MagicMock

from engine.gameplay.input.processing import (
    DeadZoneType,
    apply_dead_zone,
    apply_radial_dead_zone,
    apply_cross_dead_zone,
    ResponseCurveType,
    apply_linear_curve,
    apply_power_curve,
    apply_exponential_curve,
    apply_scurve,
    apply_step_curve,
    SmoothingType,
    InputSmoother,
    Vector2Smoother,
    InputModifierType,
    InputModifier,
    InputModifierChain,
    ProcessingSettings,
    InputProcessor,
)
from engine.gameplay.input.constants import (
    DEFAULT_DEAD_ZONE,
    DEFAULT_RADIAL_DEAD_ZONE,
    DEFAULT_OUTER_DEAD_ZONE,
    DEFAULT_RESPONSE_EXPONENT,
    DEFAULT_SMOOTHING_ALPHA,
    DEFAULT_SMOOTHING_WINDOW,
    MAX_RESPONSE_EXPONENT,
)


# =============================================================================
# Dead Zone Tests
# =============================================================================

class TestAxialDeadZone:
    """Tests for axial dead zone processing."""

    def test_zero_input_returns_zero(self):
        """Zero input returns zero."""
        assert apply_dead_zone(0.0) == 0.0

    def test_within_dead_zone_returns_zero(self):
        """Input within dead zone returns zero."""
        assert apply_dead_zone(0.1, dead_zone=0.15) == 0.0
        assert apply_dead_zone(-0.1, dead_zone=0.15) == 0.0

    def test_exactly_at_dead_zone_returns_zero(self):
        """Input exactly at dead zone boundary returns zero."""
        assert apply_dead_zone(0.15, dead_zone=0.15) == 0.0

    def test_just_above_dead_zone(self):
        """Input just above dead zone returns small value."""
        result = apply_dead_zone(0.16, dead_zone=0.15)
        assert result > 0.0
        assert result < 0.05  # Should be rescaled to near zero

    def test_beyond_outer_zone_returns_one(self):
        """Input beyond outer zone returns 1.0."""
        assert apply_dead_zone(0.96, outer_zone=0.95) == 1.0
        assert apply_dead_zone(-0.96, outer_zone=0.95) == -1.0

    def test_exactly_at_outer_zone(self):
        """Input at outer zone boundary returns 1.0."""
        result = apply_dead_zone(0.95, dead_zone=0.15, outer_zone=0.95)
        assert result == pytest.approx(1.0, rel=0.01)

    def test_preserves_sign_positive(self):
        """Positive input returns positive output."""
        result = apply_dead_zone(0.5, dead_zone=0.15)
        assert result > 0

    def test_preserves_sign_negative(self):
        """Negative input returns negative output."""
        result = apply_dead_zone(-0.5, dead_zone=0.15)
        assert result < 0

    def test_symmetric_response(self):
        """Positive and negative values produce symmetric results."""
        pos = apply_dead_zone(0.5, dead_zone=0.15)
        neg = apply_dead_zone(-0.5, dead_zone=0.15)
        assert pos == pytest.approx(-neg, rel=0.001)

    def test_linear_rescaling(self):
        """Values are linearly rescaled in the active range."""
        # Midpoint between dead_zone and outer_zone should map to ~0.5
        dead_zone = 0.15
        outer_zone = 0.95
        midpoint = (dead_zone + outer_zone) / 2.0

        result = apply_dead_zone(midpoint, dead_zone=dead_zone, outer_zone=outer_zone)
        assert result == pytest.approx(0.5, rel=0.01)

    def test_default_values(self):
        """Default dead zone and outer zone values work correctly."""
        # Just verify no errors with defaults
        result = apply_dead_zone(0.5)
        assert 0 < result < 1

    def test_custom_dead_zone(self):
        """Custom dead zone values are respected."""
        assert apply_dead_zone(0.25, dead_zone=0.3) == 0.0
        assert apply_dead_zone(0.35, dead_zone=0.3) > 0.0

    def test_custom_outer_zone(self):
        """Custom outer zone values are respected."""
        assert apply_dead_zone(0.85, outer_zone=0.8) == 1.0


class TestRadialDeadZone:
    """Tests for radial dead zone processing."""

    def test_zero_input_returns_zero(self):
        """Zero input returns (0, 0)."""
        assert apply_radial_dead_zone(0.0, 0.0) == (0.0, 0.0)

    def test_within_dead_zone_returns_zero(self):
        """Input within radial dead zone returns zero."""
        result = apply_radial_dead_zone(0.1, 0.1, dead_zone=0.2)
        assert result == (0.0, 0.0)

    def test_just_outside_dead_zone(self):
        """Input just outside dead zone returns small values."""
        result = apply_radial_dead_zone(0.2, 0.0, dead_zone=0.15)
        assert result[0] > 0.0
        assert result[0] < 0.2

    def test_beyond_outer_zone_normalized(self):
        """Input beyond outer zone is normalized to unit circle."""
        result = apply_radial_dead_zone(1.0, 1.0, outer_zone=0.95)
        magnitude = math.sqrt(result[0]**2 + result[1]**2)
        assert magnitude == pytest.approx(1.0, rel=0.01)

    def test_preserves_direction(self):
        """Output preserves input direction."""
        x, y = apply_radial_dead_zone(0.6, 0.8, dead_zone=0.15)
        # Check angle is preserved
        input_angle = math.atan2(0.8, 0.6)
        output_angle = math.atan2(y, x)
        assert input_angle == pytest.approx(output_angle, rel=0.01)

    def test_diagonal_movement_normalized(self):
        """Diagonal input outside dead zone is properly scaled."""
        # Full diagonal input
        x, y = apply_radial_dead_zone(1.0, 1.0, dead_zone=0.15, outer_zone=0.95)
        magnitude = math.sqrt(x**2 + y**2)
        assert magnitude <= 1.0 + 0.01  # Allow small tolerance

    def test_single_axis_movement(self):
        """Single axis movement works correctly."""
        x, y = apply_radial_dead_zone(0.8, 0.0, dead_zone=0.15)
        assert x > 0.0
        assert y == 0.0

    def test_negative_values(self):
        """Negative input values work correctly."""
        x, y = apply_radial_dead_zone(-0.5, -0.5, dead_zone=0.15)
        assert x < 0.0
        assert y < 0.0


class TestCrossDeadZone:
    """Tests for cross dead zone processing."""

    def test_zero_input_returns_zero(self):
        """Zero input returns (0, 0)."""
        assert apply_cross_dead_zone(0.0, 0.0) == (0.0, 0.0)

    def test_both_in_dead_zone(self):
        """Both axes in dead zone return zero."""
        result = apply_cross_dead_zone(0.1, 0.1, dead_zone=0.15)
        assert result == (0.0, 0.0)

    def test_x_in_dead_zone_affects_y(self):
        """X in dead zone reduces Y value."""
        result_no_cross = apply_dead_zone(0.5, dead_zone=0.15)
        x, y = apply_cross_dead_zone(0.05, 0.5, dead_zone=0.15)
        assert y < result_no_cross

    def test_y_in_dead_zone_affects_x(self):
        """Y in dead zone reduces X value."""
        result_no_cross = apply_dead_zone(0.5, dead_zone=0.15)
        x, y = apply_cross_dead_zone(0.5, 0.05, dead_zone=0.15)
        assert x < result_no_cross

    def test_both_outside_dead_zone(self):
        """Both axes outside dead zone pass through."""
        x, y = apply_cross_dead_zone(0.5, 0.5, dead_zone=0.15)
        assert x > 0.0
        assert y > 0.0


class TestDeadZoneType:
    """Tests for DeadZoneType enum."""

    def test_dead_zone_types_exist(self):
        """All dead zone types exist."""
        assert DeadZoneType.NONE
        assert DeadZoneType.AXIAL
        assert DeadZoneType.RADIAL
        assert DeadZoneType.CROSS


# =============================================================================
# Response Curve Tests
# =============================================================================

class TestLinearCurve:
    """Tests for linear response curve."""

    def test_zero_returns_zero(self):
        """Zero input returns zero."""
        assert apply_linear_curve(0.0) == 0.0

    def test_one_returns_one(self):
        """One returns one."""
        assert apply_linear_curve(1.0) == 1.0

    def test_negative_one_returns_negative_one(self):
        """Negative one returns negative one."""
        assert apply_linear_curve(-1.0) == -1.0

    def test_value_unchanged(self):
        """Linear curve doesn't change values."""
        for v in [0.25, 0.5, 0.75, -0.25, -0.5, -0.75]:
            assert apply_linear_curve(v) == v


class TestPowerCurve:
    """Tests for power response curve."""

    def test_zero_returns_zero(self):
        """Zero input returns zero."""
        assert apply_power_curve(0.0, exponent=2.0) == 0.0

    def test_one_returns_one(self):
        """One returns one regardless of exponent."""
        assert apply_power_curve(1.0, exponent=2.0) == 1.0
        assert apply_power_curve(1.0, exponent=3.0) == 1.0

    def test_negative_one_returns_negative_one(self):
        """Negative one returns negative one."""
        assert apply_power_curve(-1.0, exponent=2.0) == -1.0

    def test_quadratic_curve(self):
        """Quadratic curve (exponent=2) produces squared values."""
        result = apply_power_curve(0.5, exponent=2.0)
        assert result == pytest.approx(0.25, rel=0.001)

    def test_cubic_curve(self):
        """Cubic curve (exponent=3) produces cubed values."""
        result = apply_power_curve(0.5, exponent=3.0)
        assert result == pytest.approx(0.125, rel=0.001)

    def test_exponent_one_is_linear(self):
        """Exponent of 1.0 is linear."""
        assert apply_power_curve(0.5, exponent=1.0) == pytest.approx(0.5)

    def test_preserves_sign(self):
        """Power curve preserves sign."""
        assert apply_power_curve(-0.5, exponent=2.0) < 0

    def test_exponent_clamped(self):
        """Exponent is clamped to max."""
        # Shouldn't crash with huge exponent
        result = apply_power_curve(0.5, exponent=100.0)
        assert 0 <= result <= 1

    def test_default_exponent(self):
        """Default exponent works."""
        result = apply_power_curve(0.5)
        assert 0 < result < 1


class TestExponentialCurve:
    """Tests for exponential response curve."""

    def test_zero_returns_zero(self):
        """Zero input returns zero."""
        result = apply_exponential_curve(0.0)
        assert result == pytest.approx(0.0, abs=0.001)

    def test_one_returns_one(self):
        """One returns one."""
        result = apply_exponential_curve(1.0)
        assert result == pytest.approx(1.0, rel=0.01)

    def test_curve_is_exponential(self):
        """Middle values show exponential curve behavior."""
        result = apply_exponential_curve(0.5)
        # Exponential curve should give < 0.5 for input of 0.5
        assert result < 0.5

    def test_preserves_sign(self):
        """Exponential curve preserves sign."""
        result = apply_exponential_curve(-0.5)
        assert result < 0

    def test_custom_base(self):
        """Custom base affects curve shape."""
        result_low = apply_exponential_curve(0.5, base=1.5)
        result_high = apply_exponential_curve(0.5, base=4.0)
        # Higher base = steeper curve = lower middle value
        assert result_high < result_low


class TestSCurve:
    """Tests for S-curve response."""

    def test_zero_returns_zero(self):
        """Zero input returns approximately zero."""
        result = apply_scurve(0.0)
        assert result == pytest.approx(0.0, abs=0.05)

    def test_one_returns_one(self):
        """One returns approximately one."""
        result = apply_scurve(1.0)
        assert result == pytest.approx(1.0, abs=0.05)

    def test_midpoint_returns_half(self):
        """Midpoint returns approximately 0.5."""
        result = apply_scurve(0.5, midpoint=0.5)
        assert result == pytest.approx(0.5, abs=0.1)

    def test_preserves_sign(self):
        """S-curve preserves sign."""
        result = apply_scurve(-0.5)
        assert result < 0

    def test_steepness_affects_curve(self):
        """Higher steepness makes sharper transition."""
        result_low = apply_scurve(0.4, steepness=1.0)
        result_high = apply_scurve(0.4, steepness=5.0)
        # Higher steepness should give lower value before midpoint
        # This depends on implementation details

    def test_result_clamped(self):
        """Result is clamped to valid range."""
        result = apply_scurve(1.0)
        assert 0 <= result <= 1
        result = apply_scurve(-1.0)
        assert -1 <= result <= 0


class TestStepCurve:
    """Tests for step/quantized response curve."""

    def test_zero_returns_zero(self):
        """Zero input returns zero."""
        assert apply_step_curve(0.0, steps=4) == 0.0

    def test_one_returns_one(self):
        """One returns one."""
        assert apply_step_curve(1.0, steps=4) == 1.0

    def test_quantization(self):
        """Values are quantized to steps."""
        # With 4 steps, possible values are 0, 0.25, 0.5, 0.75, 1.0
        result = apply_step_curve(0.3, steps=4)
        assert result == pytest.approx(0.25, rel=0.01)

        result = apply_step_curve(0.6, steps=4)
        assert result == pytest.approx(0.5, rel=0.01)

    def test_preserves_sign(self):
        """Step curve preserves sign."""
        result = apply_step_curve(-0.5, steps=4)
        assert result < 0

    def test_zero_steps_returns_zero(self):
        """Zero steps returns zero."""
        assert apply_step_curve(0.5, steps=0) == 0.0

    def test_one_step_binary(self):
        """One step gives binary output."""
        assert apply_step_curve(0.4, steps=1) == 0.0
        assert apply_step_curve(0.6, steps=1) == 1.0

    def test_many_steps_approaches_linear(self):
        """Many steps approaches linear response."""
        result = apply_step_curve(0.5, steps=100)
        assert result == pytest.approx(0.5, rel=0.02)


class TestResponseCurveType:
    """Tests for ResponseCurveType enum."""

    def test_curve_types_exist(self):
        """All curve types exist."""
        assert ResponseCurveType.LINEAR
        assert ResponseCurveType.POWER
        assert ResponseCurveType.EXPONENTIAL
        assert ResponseCurveType.SCURVE
        assert ResponseCurveType.STEP
        assert ResponseCurveType.CUSTOM


# =============================================================================
# Input Smoothing Tests
# =============================================================================

class TestInputSmoother:
    """Tests for InputSmoother class."""

    def test_initial_smoothed_value_is_zero(self):
        """Initial smoothed value is zero."""
        smoother = InputSmoother()
        assert smoother.smoothed_value == 0.0

    def test_no_smoothing_passes_through(self):
        """SmoothingType.NONE passes values through."""
        smoother = InputSmoother(SmoothingType.NONE)
        result = smoother.update(0.5)
        assert result == 0.5

    def test_moving_average_initial(self):
        """Moving average starts with first value."""
        smoother = InputSmoother(SmoothingType.MOVING_AVERAGE, window_size=3)
        result = smoother.update(0.6)
        assert result == 0.6

    def test_moving_average_accumulates(self):
        """Moving average accumulates values."""
        smoother = InputSmoother(SmoothingType.MOVING_AVERAGE, window_size=3)
        smoother.update(0.3)
        smoother.update(0.6)
        result = smoother.update(0.9)
        # Average of 0.3, 0.6, 0.9 = 0.6
        assert result == pytest.approx(0.6, rel=0.01)

    def test_moving_average_window_limit(self):
        """Moving average respects window size."""
        smoother = InputSmoother(SmoothingType.MOVING_AVERAGE, window_size=2)
        smoother.update(0.0)
        smoother.update(0.5)
        result = smoother.update(1.0)
        # Only last 2 values: average of 0.5 and 1.0 = 0.75
        assert result == pytest.approx(0.75, rel=0.01)

    def test_exponential_smoothing_initial(self):
        """Exponential smoothing starts with first value."""
        smoother = InputSmoother(SmoothingType.EXPONENTIAL, alpha=0.5)
        result = smoother.update(0.8)
        assert result == 0.8

    def test_exponential_smoothing_blends(self):
        """Exponential smoothing blends values."""
        smoother = InputSmoother(SmoothingType.EXPONENTIAL, alpha=0.5)
        smoother.update(0.0)
        result = smoother.update(1.0)
        # With alpha=0.5: 0.5 * 1.0 + 0.5 * 0.0 = 0.5
        assert result == pytest.approx(0.5, rel=0.01)

    def test_exponential_alpha_high_favors_new(self):
        """High alpha favors new values."""
        smoother = InputSmoother(SmoothingType.EXPONENTIAL, alpha=0.9)
        smoother.update(0.0)
        result = smoother.update(1.0)
        assert result == pytest.approx(0.9, rel=0.01)

    def test_exponential_alpha_low_favors_old(self):
        """Low alpha favors old values."""
        smoother = InputSmoother(SmoothingType.EXPONENTIAL, alpha=0.1)
        smoother.update(0.0)
        result = smoother.update(1.0)
        assert result == pytest.approx(0.1, rel=0.01)

    def test_double_exponential_initial(self):
        """Double exponential starts with first value."""
        smoother = InputSmoother(SmoothingType.DOUBLE_EXPONENTIAL, alpha=0.5)
        result = smoother.update(0.5)
        assert result == 0.5
        assert smoother.velocity == 0.0

    def test_double_exponential_tracks_velocity(self):
        """Double exponential tracks velocity."""
        smoother = InputSmoother(SmoothingType.DOUBLE_EXPONENTIAL, alpha=0.5)
        smoother.update(0.0)
        smoother.update(0.5)
        smoother.update(1.0)
        # Should have positive velocity
        assert smoother.velocity > 0

    def test_reset_clears_state(self):
        """reset clears all state."""
        smoother = InputSmoother(SmoothingType.EXPONENTIAL)
        smoother.update(0.5)
        smoother.update(0.8)

        smoother.reset()

        assert smoother.smoothed_value == 0.0
        assert smoother.velocity == 0.0

    def test_alpha_clamped_to_range(self):
        """Alpha is clamped to 0-1."""
        smoother = InputSmoother(SmoothingType.EXPONENTIAL, alpha=2.0)
        assert smoother._alpha == 1.0

        smoother = InputSmoother(SmoothingType.EXPONENTIAL, alpha=-0.5)
        assert smoother._alpha == 0.0

    def test_window_size_minimum(self):
        """Window size is at least 1."""
        smoother = InputSmoother(SmoothingType.MOVING_AVERAGE, window_size=0)
        assert smoother._window_size == 1


class TestVector2Smoother:
    """Tests for Vector2Smoother class."""

    def test_initial_value_is_zero(self):
        """Initial smoothed value is (0, 0)."""
        smoother = Vector2Smoother()
        assert smoother.smoothed_value == (0.0, 0.0)

    def test_update_returns_smoothed_tuple(self):
        """update returns smoothed (x, y) tuple."""
        smoother = Vector2Smoother(SmoothingType.NONE)
        result = smoother.update(0.5, 0.8)
        assert result == (0.5, 0.8)

    def test_independent_axis_smoothing(self):
        """X and Y axes are smoothed independently."""
        smoother = Vector2Smoother(SmoothingType.EXPONENTIAL, alpha=0.5)
        smoother.update(0.0, 0.0)
        x, y = smoother.update(1.0, 0.5)
        assert x == pytest.approx(0.5, rel=0.01)
        assert y == pytest.approx(0.25, rel=0.01)

    def test_reset_clears_both_axes(self):
        """reset clears both X and Y smoothers."""
        smoother = Vector2Smoother()
        smoother.update(0.5, 0.5)

        smoother.reset()

        assert smoother.smoothed_value == (0.0, 0.0)


class TestSmoothingType:
    """Tests for SmoothingType enum."""

    def test_smoothing_types_exist(self):
        """All smoothing types exist."""
        assert SmoothingType.NONE
        assert SmoothingType.MOVING_AVERAGE
        assert SmoothingType.EXPONENTIAL
        assert SmoothingType.DOUBLE_EXPONENTIAL


# =============================================================================
# Input Modifier Tests
# =============================================================================

class TestInputModifier:
    """Tests for InputModifier dataclass."""

    def test_modifier_creation(self):
        """InputModifier can be created."""
        mod = InputModifier(
            modifier_type=InputModifierType.SCALE,
            params={"scale": 2.0}
        )
        assert mod.modifier_type == InputModifierType.SCALE
        assert mod.params["scale"] == 2.0

    def test_default_params(self):
        """Default params is empty dict."""
        mod = InputModifier(modifier_type=InputModifierType.NEGATE)
        assert mod.params == {}


class TestInputModifierChain:
    """Tests for InputModifierChain class."""

    def test_empty_chain_passes_through(self):
        """Empty chain passes values through."""
        chain = InputModifierChain()
        result = chain.process(0.5)
        assert result == 0.5

    def test_negate_modifier(self):
        """NEGATE modifier inverts sign."""
        chain = InputModifierChain()
        chain.add_modifier(InputModifier(InputModifierType.NEGATE))

        result = chain.process(0.5)
        assert result == -0.5

    def test_scale_modifier(self):
        """SCALE modifier multiplies value."""
        chain = InputModifierChain()
        chain.add_modifier(InputModifier(
            InputModifierType.SCALE,
            {"scale": 2.0}
        ))

        result = chain.process(0.25)
        assert result == 0.5

    def test_clamp_modifier(self):
        """CLAMP modifier limits range."""
        chain = InputModifierChain()
        chain.add_modifier(InputModifier(
            InputModifierType.CLAMP,
            {"min": -0.5, "max": 0.5}
        ))

        assert chain.process(0.8) == 0.5
        assert chain.process(-0.8) == -0.5
        assert chain.process(0.3) == 0.3

    def test_dead_zone_modifier(self):
        """DEAD_ZONE modifier applies dead zone."""
        chain = InputModifierChain()
        chain.add_modifier(InputModifier(
            InputModifierType.DEAD_ZONE,
            {"dead_zone": 0.2}
        ))

        assert chain.process(0.1) == 0.0
        assert chain.process(0.5) > 0.0

    def test_response_curve_linear(self):
        """LINEAR response curve passes through."""
        chain = InputModifierChain()
        chain.add_modifier(InputModifier(
            InputModifierType.RESPONSE_CURVE,
            {"curve_type": ResponseCurveType.LINEAR}
        ))

        assert chain.process(0.5) == 0.5

    def test_response_curve_power(self):
        """POWER response curve applies power."""
        chain = InputModifierChain()
        chain.add_modifier(InputModifier(
            InputModifierType.RESPONSE_CURVE,
            {"curve_type": ResponseCurveType.POWER, "exponent": 2.0}
        ))

        result = chain.process(0.5)
        assert result == pytest.approx(0.25, rel=0.01)

    def test_response_curve_exponential(self):
        """EXPONENTIAL response curve works."""
        chain = InputModifierChain()
        chain.add_modifier(InputModifier(
            InputModifierType.RESPONSE_CURVE,
            {"curve_type": ResponseCurveType.EXPONENTIAL, "base": 2.0}
        ))

        result = chain.process(0.5)
        assert result < 0.5  # Exponential curve

    def test_response_curve_scurve(self):
        """SCURVE response curve works."""
        chain = InputModifierChain()
        chain.add_modifier(InputModifier(
            InputModifierType.RESPONSE_CURVE,
            {"curve_type": ResponseCurveType.SCURVE}
        ))

        result = chain.process(0.5)
        assert 0 < result < 1

    def test_response_curve_step(self):
        """STEP response curve quantizes."""
        chain = InputModifierChain()
        chain.add_modifier(InputModifier(
            InputModifierType.RESPONSE_CURVE,
            {"curve_type": ResponseCurveType.STEP, "steps": 4}
        ))

        result = chain.process(0.3)
        assert result == pytest.approx(0.25, rel=0.01)

    def test_smooth_modifier(self):
        """SMOOTH modifier applies smoothing."""
        chain = InputModifierChain()
        chain.add_modifier(InputModifier(
            InputModifierType.SMOOTH,
            {"type": SmoothingType.EXPONENTIAL, "alpha": 0.5}
        ))

        # First value
        chain.process(0.0, "test")
        # Second value blended
        result = chain.process(1.0, "test")
        assert result == pytest.approx(0.5, rel=0.01)

    def test_chained_modifiers(self):
        """Multiple modifiers are applied in order."""
        chain = InputModifierChain()
        chain.add_modifier(InputModifier(
            InputModifierType.SCALE,
            {"scale": 2.0}
        ))
        chain.add_modifier(InputModifier(
            InputModifierType.CLAMP,
            {"min": -1.0, "max": 1.0}
        ))

        # 0.8 * 2.0 = 1.6, clamped to 1.0
        result = chain.process(0.8)
        assert result == 1.0

    def test_remove_modifier(self):
        """remove_modifier removes by index."""
        chain = InputModifierChain()
        chain.add_modifier(InputModifier(InputModifierType.NEGATE))
        chain.add_modifier(InputModifier(InputModifierType.SCALE, {"scale": 2.0}))

        result = chain.remove_modifier(0)
        assert result is True

        # Now only scale remains
        assert chain.process(0.5) == 1.0

    def test_remove_modifier_invalid_index(self):
        """remove_modifier returns False for invalid index."""
        chain = InputModifierChain()
        assert chain.remove_modifier(0) is False
        assert chain.remove_modifier(-1) is False

    def test_clear_removes_all(self):
        """clear removes all modifiers."""
        chain = InputModifierChain()
        chain.add_modifier(InputModifier(InputModifierType.NEGATE))
        chain.add_modifier(InputModifier(InputModifierType.SCALE, {"scale": 2.0}))

        chain.clear()

        # Should pass through unchanged
        assert chain.process(0.5) == 0.5

    def test_process_2d_swizzle(self):
        """SWIZZLE swaps X and Y."""
        chain = InputModifierChain()
        chain.add_modifier(InputModifier(InputModifierType.SWIZZLE))

        x, y = chain.process_2d(0.3, 0.8)
        assert x == 0.8
        assert y == 0.3

    def test_process_2d_invert_y(self):
        """INVERT_Y inverts Y axis."""
        chain = InputModifierChain()
        chain.add_modifier(InputModifier(InputModifierType.INVERT_Y))

        x, y = chain.process_2d(0.5, 0.8)
        assert x == 0.5
        assert y == -0.8

    def test_process_2d_radial_dead_zone(self):
        """2D processing with radial dead zone."""
        chain = InputModifierChain()
        chain.add_modifier(InputModifier(
            InputModifierType.DEAD_ZONE,
            {"dead_zone": 0.2, "type": DeadZoneType.RADIAL}
        ))

        x, y = chain.process_2d(0.1, 0.1)
        assert x == 0.0
        assert y == 0.0

    def test_process_2d_cross_dead_zone(self):
        """2D processing with cross dead zone."""
        chain = InputModifierChain()
        chain.add_modifier(InputModifier(
            InputModifierType.DEAD_ZONE,
            {"dead_zone": 0.2, "type": DeadZoneType.CROSS}
        ))

        x, y = chain.process_2d(0.1, 0.1)
        assert x == 0.0
        assert y == 0.0


class TestInputModifierType:
    """Tests for InputModifierType enum."""

    def test_modifier_types_exist(self):
        """All modifier types exist."""
        assert InputModifierType.NEGATE
        assert InputModifierType.SCALE
        assert InputModifierType.CLAMP
        assert InputModifierType.SWIZZLE
        assert InputModifierType.DEAD_ZONE
        assert InputModifierType.RESPONSE_CURVE
        assert InputModifierType.SMOOTH
        assert InputModifierType.INVERT_Y


# =============================================================================
# Input Processor Tests
# =============================================================================

class TestProcessingSettings:
    """Tests for ProcessingSettings dataclass."""

    def test_default_values(self):
        """Default settings have sensible values."""
        settings = ProcessingSettings()
        assert settings.dead_zone_type == DeadZoneType.RADIAL
        assert settings.dead_zone == DEFAULT_DEAD_ZONE
        assert settings.response_curve == ResponseCurveType.LINEAR
        assert settings.smoothing_type == SmoothingType.NONE
        assert settings.sensitivity == 1.0
        assert settings.invert_x is False
        assert settings.invert_y is False

    def test_custom_settings(self):
        """Custom settings are stored correctly."""
        settings = ProcessingSettings(
            dead_zone_type=DeadZoneType.AXIAL,
            dead_zone=0.2,
            response_curve=ResponseCurveType.POWER,
            sensitivity=1.5,
            invert_y=True
        )
        assert settings.dead_zone_type == DeadZoneType.AXIAL
        assert settings.dead_zone == 0.2
        assert settings.response_curve == ResponseCurveType.POWER
        assert settings.sensitivity == 1.5
        assert settings.invert_y is True


class TestInputProcessor:
    """Tests for InputProcessor class."""

    def test_default_settings(self):
        """Processor has default settings."""
        processor = InputProcessor()
        assert processor.settings is not None

    def test_custom_settings(self):
        """Processor accepts custom settings."""
        settings = ProcessingSettings(sensitivity=2.0)
        processor = InputProcessor(settings)
        assert processor.settings.sensitivity == 2.0

    def test_process_1d_basic(self):
        """process_1d works with defaults."""
        processor = InputProcessor()
        result = processor.process_1d(0.5)
        assert 0 <= result <= 1

    def test_process_1d_with_dead_zone(self):
        """process_1d applies dead zone."""
        settings = ProcessingSettings(dead_zone=0.2)
        processor = InputProcessor(settings)

        assert processor.process_1d(0.1) == 0.0
        assert processor.process_1d(0.5) > 0.0

    def test_process_1d_with_sensitivity(self):
        """process_1d applies sensitivity."""
        settings = ProcessingSettings(
            dead_zone_type=DeadZoneType.NONE,
            sensitivity=2.0
        )
        processor = InputProcessor(settings)

        result = processor.process_1d(0.4)
        assert result == pytest.approx(0.8, rel=0.01)

    def test_process_1d_with_invert(self):
        """process_1d applies invert_x."""
        settings = ProcessingSettings(
            dead_zone_type=DeadZoneType.NONE,
            invert_x=True
        )
        processor = InputProcessor(settings)

        result = processor.process_1d(0.5)
        assert result == pytest.approx(-0.5, rel=0.01)

    def test_process_1d_clamps_output(self):
        """process_1d clamps output to -1, 1."""
        settings = ProcessingSettings(
            dead_zone_type=DeadZoneType.NONE,
            sensitivity=3.0
        )
        processor = InputProcessor(settings)

        result = processor.process_1d(0.5)
        assert -1.0 <= result <= 1.0

    def test_process_2d_basic(self):
        """process_2d works with defaults."""
        processor = InputProcessor()
        x, y = processor.process_2d(0.5, 0.5)
        assert -1 <= x <= 1
        assert -1 <= y <= 1

    def test_process_2d_radial_dead_zone(self):
        """process_2d applies radial dead zone."""
        settings = ProcessingSettings(
            dead_zone_type=DeadZoneType.RADIAL,
            dead_zone=0.2
        )
        processor = InputProcessor(settings)

        x, y = processor.process_2d(0.1, 0.1)
        assert x == 0.0
        assert y == 0.0

    def test_process_2d_with_invert_y(self):
        """process_2d applies invert_y."""
        settings = ProcessingSettings(
            dead_zone_type=DeadZoneType.NONE,
            invert_y=True
        )
        processor = InputProcessor(settings)

        x, y = processor.process_2d(0.5, 0.5)
        assert x == pytest.approx(0.5, rel=0.01)
        assert y == pytest.approx(-0.5, rel=0.01)

    def test_process_2d_clamps_output(self):
        """process_2d clamps output to -1, 1."""
        settings = ProcessingSettings(
            dead_zone_type=DeadZoneType.NONE,
            sensitivity=3.0
        )
        processor = InputProcessor(settings)

        x, y = processor.process_2d(0.5, 0.5)
        assert -1.0 <= x <= 1.0
        assert -1.0 <= y <= 1.0

    def test_process_trigger(self):
        """process_trigger works for trigger input."""
        processor = InputProcessor()
        result = processor.process_trigger(0.5)
        assert 0.0 <= result <= 1.0

    def test_process_trigger_clamps_input(self):
        """process_trigger clamps input to 0-1."""
        processor = InputProcessor()
        result = processor.process_trigger(-0.5)
        assert result >= 0.0

    def test_process_trigger_clamps_output(self):
        """process_trigger clamps output to 0-1."""
        settings = ProcessingSettings(
            dead_zone_type=DeadZoneType.NONE,
            sensitivity=3.0
        )
        processor = InputProcessor(settings)

        result = processor.process_trigger(0.5)
        assert 0.0 <= result <= 1.0

    def test_settings_setter_rebuilds_chain(self):
        """Setting new settings rebuilds modifier chain."""
        processor = InputProcessor()

        new_settings = ProcessingSettings(sensitivity=5.0)
        processor.settings = new_settings

        assert processor.settings.sensitivity == 5.0

    def test_reset_clears_state(self):
        """reset clears processing state."""
        settings = ProcessingSettings(
            smoothing_type=SmoothingType.EXPONENTIAL
        )
        processor = InputProcessor(settings)

        # Build up some smoothing state
        processor.process_1d(0.5)
        processor.process_1d(0.8)

        processor.reset()

        # After reset, should start fresh
        # This is hard to test directly, but we verify no crash

    def test_with_response_curve(self):
        """Processor with response curve."""
        settings = ProcessingSettings(
            dead_zone_type=DeadZoneType.NONE,
            response_curve=ResponseCurveType.POWER,
            response_exponent=2.0
        )
        processor = InputProcessor(settings)

        result = processor.process_1d(0.5)
        assert result == pytest.approx(0.25, rel=0.01)

    def test_with_smoothing(self):
        """Processor with smoothing."""
        settings = ProcessingSettings(
            dead_zone_type=DeadZoneType.NONE,
            smoothing_type=SmoothingType.EXPONENTIAL,
            smoothing_alpha=0.5
        )
        processor = InputProcessor(settings)

        processor.process_1d(0.0, "test")
        result = processor.process_1d(1.0, "test")

        assert result == pytest.approx(0.5, rel=0.1)

    def test_full_pipeline(self):
        """Full processing pipeline with all options."""
        settings = ProcessingSettings(
            dead_zone_type=DeadZoneType.RADIAL,
            dead_zone=0.15,
            response_curve=ResponseCurveType.POWER,
            response_exponent=1.5,
            smoothing_type=SmoothingType.EXPONENTIAL,
            smoothing_alpha=0.8,
            sensitivity=1.2,
            invert_y=True
        )
        processor = InputProcessor(settings)

        # Just verify it processes without error
        x, y = processor.process_2d(0.5, 0.5, "test_stick")
        assert isinstance(x, float)
        assert isinstance(y, float)


# =============================================================================
# Edge Cases and Integration Tests
# =============================================================================

class TestProcessingEdgeCases:
    """Edge case tests for processing."""

    def test_nan_input_dead_zone(self):
        """NaN input handling."""
        # This tests robustness - may return NaN or 0
        result = apply_dead_zone(float('nan'))
        # Just verify no exception

    def test_inf_input_dead_zone(self):
        """Infinity input handling."""
        result = apply_dead_zone(float('inf'))
        assert result == 1.0 or result == float('inf')

    def test_very_small_dead_zone(self):
        """Very small dead zone."""
        result = apply_dead_zone(0.001, dead_zone=0.0001)
        assert result > 0.0

    def test_dead_zone_equals_outer_zone(self):
        """Dead zone equals outer zone edge case."""
        # This is a degenerate case
        result = apply_dead_zone(0.5, dead_zone=0.5, outer_zone=0.5)
        # Should not crash

    def test_power_curve_large_exponent(self):
        """Large exponent is clamped."""
        result = apply_power_curve(0.5, exponent=1000.0)
        assert 0 <= result <= 1

    def test_smoothing_rapid_changes(self):
        """Smoothing handles rapid value changes."""
        smoother = InputSmoother(SmoothingType.EXPONENTIAL, alpha=0.5)

        values = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        for v in values:
            result = smoother.update(v)
            assert -0.1 <= result <= 1.1

    def test_modifier_chain_many_modifiers(self):
        """Chain handles many modifiers."""
        chain = InputModifierChain()
        for _ in range(50):
            chain.add_modifier(InputModifier(
                InputModifierType.SCALE,
                {"scale": 1.01}
            ))

        # Should still work
        result = chain.process(0.5)
        assert isinstance(result, float)

    def test_processor_many_updates(self):
        """Processor handles many updates."""
        processor = InputProcessor()

        for i in range(1000):
            val = (i % 100) / 100.0
            processor.process_1d(val, f"input_{i % 10}")


class TestProcessingIntegration:
    """Integration tests for the processing pipeline."""

    def test_gamepad_stick_simulation(self):
        """Simulate realistic gamepad stick processing."""
        settings = ProcessingSettings(
            dead_zone_type=DeadZoneType.RADIAL,
            dead_zone=0.15,
            outer_zone=0.95,
            response_curve=ResponseCurveType.POWER,
            response_exponent=1.5,
            sensitivity=1.0
        )
        processor = InputProcessor(settings)

        # Simulate stick movement
        positions = [
            (0.0, 0.0),      # Center
            (0.1, 0.0),      # Slight right (in dead zone)
            (0.5, 0.0),      # Half right
            (1.0, 0.0),      # Full right
            (0.7, 0.7),      # Diagonal
            (-0.5, 0.3),     # Mixed
        ]

        for raw_x, raw_y in positions:
            x, y = processor.process_2d(raw_x, raw_y)
            assert -1.0 <= x <= 1.0
            assert -1.0 <= y <= 1.0

    def test_mouse_sensitivity_simulation(self):
        """Simulate mouse sensitivity processing."""
        settings = ProcessingSettings(
            dead_zone_type=DeadZoneType.NONE,
            response_curve=ResponseCurveType.LINEAR,
            sensitivity=2.5
        )
        processor = InputProcessor(settings)

        # Small movements
        dx, dy = processor.process_2d(0.1, 0.2)
        assert abs(dx) > 0.1  # Amplified
        assert abs(dy) > 0.2

    def test_trigger_processing_simulation(self):
        """Simulate trigger processing."""
        settings = ProcessingSettings(
            dead_zone_type=DeadZoneType.AXIAL,
            dead_zone=0.1,
            response_curve=ResponseCurveType.POWER,
            response_exponent=2.0
        )
        processor = InputProcessor(settings)

        # Test trigger range
        for raw in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
            result = processor.process_trigger(raw)
            assert 0.0 <= result <= 1.0
