"""
Comprehensive tests for easing functions.

Tests cover:
- All standard easing functions (linear, quad, cubic, quart, quint)
- Trigonometric easings (sine)
- Exponential and circular easings (expo, circ)
- Special effect easings (elastic, back, bounce)
- Cubic bezier curves
- Mathematical correctness verification
- Boundary conditions
- Registry and lookup functionality
"""

from __future__ import annotations

import math
import pytest
from typing import Callable

from engine.ui.animation.easing import (
    EasingType,
    EasingFunction,
    # Linear
    linear,
    # Quad
    quad_in,
    quad_out,
    quad_in_out,
    # Cubic
    cubic_in,
    cubic_out,
    cubic_in_out,
    # Quart
    quart_in,
    quart_out,
    quart_in_out,
    # Quint
    quint_in,
    quint_out,
    quint_in_out,
    # Sine
    sine_in,
    sine_out,
    sine_in_out,
    # Expo
    expo_in,
    expo_out,
    expo_in_out,
    # Circ
    circ_in,
    circ_out,
    circ_in_out,
    # Elastic
    elastic_in,
    elastic_out,
    elastic_in_out,
    # Back
    back_in,
    back_out,
    back_in_out,
    # Bounce
    bounce_in,
    bounce_out,
    bounce_in_out,
    # Bezier
    CubicBezier,
    create_bezier,
    EASE,
    EASE_IN,
    EASE_OUT,
    EASE_IN_OUT,
    # Registry
    get_easing,
    # Utils
    clamp,
    lerp,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================


# Collect all easing functions for parametrized tests
ALL_EASING_FUNCTIONS = [
    linear,
    quad_in, quad_out, quad_in_out,
    cubic_in, cubic_out, cubic_in_out,
    quart_in, quart_out, quart_in_out,
    quint_in, quint_out, quint_in_out,
    sine_in, sine_out, sine_in_out,
    expo_in, expo_out, expo_in_out,
    circ_in, circ_out, circ_in_out,
    elastic_in, elastic_out, elastic_in_out,
    back_in, back_out, back_in_out,
    bounce_in, bounce_out, bounce_in_out,
]


# =============================================================================
# GENERAL EASING PROPERTIES TESTS
# =============================================================================


class TestEasingGeneralProperties:
    """Tests for properties that should hold for all easing functions."""

    @pytest.mark.parametrize("easing_func", ALL_EASING_FUNCTIONS)
    def test_easing_at_zero_is_zero(self, easing_func: EasingFunction) -> None:
        """All easing functions should return 0 at t=0."""
        result = easing_func(0.0)
        assert result == pytest.approx(0.0, abs=0.01)

    @pytest.mark.parametrize("easing_func", ALL_EASING_FUNCTIONS)
    def test_easing_at_one_is_one(self, easing_func: EasingFunction) -> None:
        """All easing functions should return 1 at t=1."""
        result = easing_func(1.0)
        assert result == pytest.approx(1.0, abs=0.01)

    @pytest.mark.parametrize("easing_func", ALL_EASING_FUNCTIONS)
    def test_easing_handles_intermediate_values(self, easing_func: EasingFunction) -> None:
        """All easing functions should handle values between 0 and 1."""
        # Test at several points
        for t in [0.1, 0.25, 0.5, 0.75, 0.9]:
            result = easing_func(t)
            # Result should be finite
            assert math.isfinite(result)


# =============================================================================
# LINEAR TESTS
# =============================================================================


class TestLinear:
    """Tests for linear easing."""

    def test_linear_at_zero(self) -> None:
        """Linear should return 0 at t=0."""
        assert linear(0.0) == 0.0

    def test_linear_at_one(self) -> None:
        """Linear should return 1 at t=1."""
        assert linear(1.0) == 1.0

    def test_linear_at_half(self) -> None:
        """Linear should return 0.5 at t=0.5."""
        assert linear(0.5) == 0.5

    def test_linear_identity(self) -> None:
        """Linear should return input unchanged."""
        for t in [0.1, 0.25, 0.33, 0.67, 0.9]:
            assert linear(t) == t


# =============================================================================
# QUAD TESTS
# =============================================================================


class TestQuad:
    """Tests for quadratic easing functions."""

    def test_quad_in_slower_than_linear(self) -> None:
        """quad_in should be slower than linear at start."""
        assert quad_in(0.5) < 0.5

    def test_quad_in_formula(self) -> None:
        """quad_in should follow t^2."""
        for t in [0.2, 0.4, 0.6, 0.8]:
            expected = t * t
            assert quad_in(t) == pytest.approx(expected)

    def test_quad_out_faster_than_linear(self) -> None:
        """quad_out should be faster than linear at start."""
        assert quad_out(0.5) > 0.5

    def test_quad_out_formula(self) -> None:
        """quad_out should follow t*(2-t)."""
        for t in [0.2, 0.4, 0.6, 0.8]:
            expected = t * (2 - t)
            assert quad_out(t) == pytest.approx(expected)

    def test_quad_in_out_slow_fast_slow(self) -> None:
        """quad_in_out should be slow-fast-slow."""
        # First half should be slower
        assert quad_in_out(0.25) < 0.25
        # Second half should be faster
        assert quad_in_out(0.75) > 0.75

    def test_quad_in_out_symmetric(self) -> None:
        """quad_in_out should be symmetric around 0.5."""
        # f(0.5) should be 0.5
        assert quad_in_out(0.5) == pytest.approx(0.5)


# =============================================================================
# CUBIC TESTS
# =============================================================================


class TestCubic:
    """Tests for cubic easing functions."""

    def test_cubic_in_formula(self) -> None:
        """cubic_in should follow t^3."""
        for t in [0.2, 0.5, 0.8]:
            expected = t * t * t
            assert cubic_in(t) == pytest.approx(expected)

    def test_cubic_out_formula(self) -> None:
        """cubic_out should follow (t-1)^3 + 1."""
        for t in [0.2, 0.5, 0.8]:
            t1 = t - 1
            expected = t1 * t1 * t1 + 1
            assert cubic_out(t) == pytest.approx(expected)

    def test_cubic_steeper_than_quad(self) -> None:
        """cubic should be steeper than quad at extremes."""
        # cubic_in should be slower than quad_in at start
        assert cubic_in(0.25) < quad_in(0.25)
        # cubic_out should be faster than quad_out at start
        assert cubic_out(0.25) > quad_out(0.25)


# =============================================================================
# QUART TESTS
# =============================================================================


class TestQuart:
    """Tests for quartic easing functions."""

    def test_quart_in_formula(self) -> None:
        """quart_in should follow t^4."""
        for t in [0.2, 0.5, 0.8]:
            expected = t ** 4
            assert quart_in(t) == pytest.approx(expected)

    def test_quart_out_formula(self) -> None:
        """quart_out should follow 1 - (t-1)^4."""
        for t in [0.2, 0.5, 0.8]:
            t1 = t - 1
            expected = 1 - t1 ** 4
            assert quart_out(t) == pytest.approx(expected)


# =============================================================================
# QUINT TESTS
# =============================================================================


class TestQuint:
    """Tests for quintic easing functions."""

    def test_quint_in_formula(self) -> None:
        """quint_in should follow t^5."""
        for t in [0.2, 0.5, 0.8]:
            expected = t ** 5
            assert quint_in(t) == pytest.approx(expected)

    def test_quint_out_formula(self) -> None:
        """quint_out should follow (t-1)^5 + 1."""
        for t in [0.2, 0.5, 0.8]:
            t1 = t - 1
            expected = t1 ** 5 + 1
            assert quint_out(t) == pytest.approx(expected)

    def test_quint_steepest_polynomial(self) -> None:
        """quint should be steeper than quart."""
        assert quint_in(0.25) < quart_in(0.25)
        assert quint_out(0.25) > quart_out(0.25)


# =============================================================================
# SINE TESTS
# =============================================================================


class TestSine:
    """Tests for sinusoidal easing functions."""

    def test_sine_in_formula(self) -> None:
        """sine_in should follow 1 - cos(t * pi/2)."""
        for t in [0.2, 0.5, 0.8]:
            expected = 1 - math.cos(t * math.pi / 2)
            assert sine_in(t) == pytest.approx(expected)

    def test_sine_out_formula(self) -> None:
        """sine_out should follow sin(t * pi/2)."""
        for t in [0.2, 0.5, 0.8]:
            expected = math.sin(t * math.pi / 2)
            assert sine_out(t) == pytest.approx(expected)

    def test_sine_in_out_formula(self) -> None:
        """sine_in_out should follow 0.5 * (1 - cos(pi*t))."""
        for t in [0.2, 0.5, 0.8]:
            expected = 0.5 * (1 - math.cos(math.pi * t))
            assert sine_in_out(t) == pytest.approx(expected)

    def test_sine_smoother_than_quad(self) -> None:
        """Sine should be smoother than quad (less extreme)."""
        # sine_in at 0.25 should be closer to linear than quad_in
        sine_diff = abs(sine_in(0.25) - 0.25)
        quad_diff = abs(quad_in(0.25) - 0.25)
        assert sine_diff < quad_diff


# =============================================================================
# EXPO TESTS
# =============================================================================


class TestExpo:
    """Tests for exponential easing functions."""

    def test_expo_in_at_zero(self) -> None:
        """expo_in should return 0 at t=0."""
        assert expo_in(0.0) == 0.0

    def test_expo_out_at_one(self) -> None:
        """expo_out should return 1 at t=1."""
        assert expo_out(1.0) == 1.0

    def test_expo_in_very_slow_at_start(self) -> None:
        """expo_in should be very slow at the start."""
        assert expo_in(0.1) < 0.01

    def test_expo_out_very_fast_at_start(self) -> None:
        """expo_out should be very fast at the start."""
        assert expo_out(0.1) >= 0.5  # At 10% progress, already at 50% output

    def test_expo_in_out_boundaries(self) -> None:
        """expo_in_out should handle boundaries correctly."""
        assert expo_in_out(0.0) == 0.0
        assert expo_in_out(1.0) == 1.0


# =============================================================================
# CIRC TESTS
# =============================================================================


class TestCirc:
    """Tests for circular easing functions."""

    def test_circ_in_formula(self) -> None:
        """circ_in should follow 1 - sqrt(1 - t^2)."""
        for t in [0.2, 0.5, 0.8]:
            expected = 1 - math.sqrt(1 - t * t)
            assert circ_in(t) == pytest.approx(expected)

    def test_circ_out_formula(self) -> None:
        """circ_out should follow sqrt(1 - (t-1)^2)."""
        for t in [0.2, 0.5, 0.8]:
            t1 = t - 1
            expected = math.sqrt(1 - t1 * t1)
            assert circ_out(t) == pytest.approx(expected)

    def test_circ_produces_circular_curve(self) -> None:
        """Points should lie on a circular arc."""
        # For circ_in, (t, 1-f(t)) should be on unit circle
        for t in [0.2, 0.4, 0.6, 0.8]:
            y = 1 - circ_in(t)
            # t^2 + y^2 should equal 1
            assert t * t + y * y == pytest.approx(1.0)


# =============================================================================
# ELASTIC TESTS
# =============================================================================


class TestElastic:
    """Tests for elastic easing functions."""

    def test_elastic_in_boundaries(self) -> None:
        """elastic_in should handle boundaries."""
        assert elastic_in(0.0) == 0.0
        assert elastic_in(1.0) == 1.0

    def test_elastic_out_boundaries(self) -> None:
        """elastic_out should handle boundaries."""
        assert elastic_out(0.0) == 0.0
        assert elastic_out(1.0) == 1.0

    def test_elastic_in_overshoots_negative(self) -> None:
        """elastic_in should go negative near start."""
        # Check various points - elastic goes negative before settling
        found_negative = False
        for t in [i * 0.01 for i in range(1, 100)]:
            if elastic_in(t) < 0:
                found_negative = True
                break
        assert found_negative, "elastic_in should have negative values"

    def test_elastic_out_overshoots_positive(self) -> None:
        """elastic_out should exceed 1 before settling."""
        found_over_one = False
        for t in [i * 0.01 for i in range(1, 100)]:
            if elastic_out(t) > 1:
                found_over_one = True
                break
        assert found_over_one, "elastic_out should exceed 1"

    def test_elastic_in_out_boundaries(self) -> None:
        """elastic_in_out should handle boundaries."""
        assert elastic_in_out(0.0) == 0.0
        assert elastic_in_out(1.0) == 1.0


# =============================================================================
# BACK TESTS
# =============================================================================


class TestBack:
    """Tests for back (overshoot) easing functions."""

    def test_back_in_goes_negative(self) -> None:
        """back_in should go negative at start."""
        # At some point early, should be negative
        assert back_in(0.3) < 0

    def test_back_out_exceeds_one(self) -> None:
        """back_out should exceed 1 before settling."""
        # At some point late, should exceed 1
        assert back_out(0.7) > 1

    def test_back_in_out_goes_negative_first(self) -> None:
        """back_in_out should go negative in first half."""
        assert back_in_out(0.2) < 0

    def test_back_in_out_exceeds_one_second_half(self) -> None:
        """back_in_out should exceed 1 in second half."""
        assert back_in_out(0.8) > 1

    def test_back_uses_standard_overshoot(self) -> None:
        """Back should use standard overshoot constant (~1.70158)."""
        # Verify the formula by checking a known value
        s = 1.70158
        t = 0.5
        expected = t * t * ((s + 1) * t - s)
        assert back_in(0.5) == pytest.approx(expected)


# =============================================================================
# BOUNCE TESTS
# =============================================================================


class TestBounce:
    """Tests for bounce easing functions."""

    def test_bounce_out_stays_in_bounds(self) -> None:
        """bounce_out should stay between 0 and 1."""
        for t in [i * 0.01 for i in range(101)]:
            result = bounce_out(t)
            assert 0.0 <= result <= 1.0

    def test_bounce_in_stays_in_bounds(self) -> None:
        """bounce_in should stay between 0 and 1."""
        for t in [i * 0.01 for i in range(101)]:
            result = bounce_in(t)
            assert 0.0 <= result <= 1.0

    def test_bounce_out_has_bounces(self) -> None:
        """bounce_out should have characteristic bounce pattern."""
        # The function should have local maxima (bounces)
        prev = 0.0
        direction_changes = 0
        going_up = True

        for i in range(1, 101):
            t = i * 0.01
            curr = bounce_out(t)
            if going_up and curr < prev:
                going_up = False
                direction_changes += 1
            elif not going_up and curr > prev:
                going_up = True
            prev = curr

        # Should have multiple direction changes (bounces)
        assert direction_changes >= 2

    def test_bounce_in_out_boundaries(self) -> None:
        """bounce_in_out should handle boundaries."""
        assert bounce_in_out(0.0) == pytest.approx(0.0)
        assert bounce_in_out(1.0) == pytest.approx(1.0)


# =============================================================================
# CUBIC BEZIER TESTS
# =============================================================================


class TestCubicBezier:
    """Tests for CubicBezier class."""

    def test_bezier_creation(self) -> None:
        """Should create bezier with control points."""
        bezier = CubicBezier(0.25, 0.1, 0.25, 1.0)
        assert bezier.x1 == 0.25
        assert bezier.y1 == 0.1
        assert bezier.x2 == 0.25
        assert bezier.y2 == 1.0

    def test_bezier_x1_out_of_bounds_raises(self) -> None:
        """x1 must be in [0, 1]."""
        with pytest.raises(ValueError):
            CubicBezier(-0.1, 0.0, 0.5, 1.0)
        with pytest.raises(ValueError):
            CubicBezier(1.1, 0.0, 0.5, 1.0)

    def test_bezier_x2_out_of_bounds_raises(self) -> None:
        """x2 must be in [0, 1]."""
        with pytest.raises(ValueError):
            CubicBezier(0.5, 0.0, -0.1, 1.0)
        with pytest.raises(ValueError):
            CubicBezier(0.5, 0.0, 1.1, 1.0)

    def test_bezier_y_can_be_outside_bounds(self) -> None:
        """y1 and y2 can be outside [0, 1] for overshoot."""
        bezier = CubicBezier(0.5, -0.5, 0.5, 1.5)
        assert bezier.y1 == -0.5
        assert bezier.y2 == 1.5

    def test_bezier_at_zero(self) -> None:
        """Bezier should return 0 at t=0."""
        bezier = CubicBezier(0.25, 0.1, 0.25, 1.0)
        assert bezier(0.0) == 0.0

    def test_bezier_at_one(self) -> None:
        """Bezier should return 1 at t=1."""
        bezier = CubicBezier(0.25, 0.1, 0.25, 1.0)
        assert bezier(1.0) == 1.0

    def test_bezier_callable(self) -> None:
        """Bezier should be callable as a function."""
        bezier = CubicBezier(0.25, 0.1, 0.25, 1.0)
        result = bezier(0.5)
        assert isinstance(result, float)
        assert 0.0 < result < 1.0

    def test_bezier_handles_edge_cases(self) -> None:
        """Bezier should handle edge cases gracefully."""
        bezier = CubicBezier(0.25, 0.1, 0.25, 1.0)
        # Values just outside [0, 1]
        assert bezier(-0.001) == 0.0
        assert bezier(1.001) == 1.0

    def test_create_bezier_factory(self) -> None:
        """create_bezier should create CubicBezier."""
        bezier = create_bezier(0.4, 0.0, 0.2, 1.0)
        assert isinstance(bezier, CubicBezier)

    def test_predefined_ease(self) -> None:
        """EASE should be the standard CSS ease curve."""
        assert EASE.x1 == 0.25
        assert EASE.y1 == 0.1
        assert EASE.x2 == 0.25
        assert EASE.y2 == 1.0

    def test_predefined_ease_in(self) -> None:
        """EASE_IN should be the standard CSS ease-in curve."""
        assert EASE_IN.x1 == 0.42
        assert EASE_IN.y1 == 0.0

    def test_predefined_ease_out(self) -> None:
        """EASE_OUT should be the standard CSS ease-out curve."""
        assert EASE_OUT.x2 == 0.58
        assert EASE_OUT.y2 == 1.0


# =============================================================================
# REGISTRY TESTS
# =============================================================================


class TestEasingRegistry:
    """Tests for the easing registry and get_easing function."""

    def test_get_easing_by_type(self) -> None:
        """Should get easing by EasingType enum."""
        func = get_easing(EasingType.LINEAR)
        assert func(0.5) == 0.5

    def test_get_easing_by_name(self) -> None:
        """Should get easing by string name."""
        func = get_easing("linear")
        assert func(0.5) == 0.5

    def test_get_easing_case_insensitive(self) -> None:
        """Should be case insensitive for string names."""
        func1 = get_easing("Linear")
        func2 = get_easing("LINEAR")
        func3 = get_easing("linear")
        assert func1(0.5) == func2(0.5) == func3(0.5)

    def test_get_easing_all_types(self) -> None:
        """Should be able to get all EasingType values."""
        for easing_type in EasingType:
            func = get_easing(easing_type)
            assert callable(func)

    def test_get_easing_unknown_raises(self) -> None:
        """Unknown easing name should raise ValueError."""
        with pytest.raises(ValueError):
            get_easing("unknown_easing")

    def test_get_easing_css_aliases(self) -> None:
        """Should support CSS-style aliases."""
        ease = get_easing("ease")
        ease_in = get_easing("ease_in")
        ease_out = get_easing("ease_out")
        ease_in_out = get_easing("ease_in_out")

        assert callable(ease)
        assert callable(ease_in)
        assert callable(ease_out)
        assert callable(ease_in_out)


# =============================================================================
# UTILITY FUNCTION TESTS
# =============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_clamp_within_range(self) -> None:
        """clamp should return value when within range."""
        assert clamp(0.5) == 0.5
        assert clamp(0.5, 0.0, 1.0) == 0.5

    def test_clamp_below_minimum(self) -> None:
        """clamp should return minimum when below range."""
        assert clamp(-0.5) == 0.0
        assert clamp(-0.5, 0.0, 1.0) == 0.0

    def test_clamp_above_maximum(self) -> None:
        """clamp should return maximum when above range."""
        assert clamp(1.5) == 1.0
        assert clamp(1.5, 0.0, 1.0) == 1.0

    def test_clamp_custom_range(self) -> None:
        """clamp should work with custom range."""
        assert clamp(5.0, 0.0, 10.0) == 5.0
        assert clamp(-5.0, 0.0, 10.0) == 0.0
        assert clamp(15.0, 0.0, 10.0) == 10.0

    def test_lerp_at_zero(self) -> None:
        """lerp at t=0 should return a."""
        assert lerp(0.0, 10.0, 0.0) == 0.0

    def test_lerp_at_one(self) -> None:
        """lerp at t=1 should return b."""
        assert lerp(0.0, 10.0, 1.0) == 10.0

    def test_lerp_at_half(self) -> None:
        """lerp at t=0.5 should return midpoint."""
        assert lerp(0.0, 10.0, 0.5) == 5.0

    def test_lerp_negative_values(self) -> None:
        """lerp should handle negative values."""
        assert lerp(-10.0, 10.0, 0.5) == 0.0

    def test_lerp_reverse(self) -> None:
        """lerp should work when b < a."""
        assert lerp(10.0, 0.0, 0.5) == 5.0
