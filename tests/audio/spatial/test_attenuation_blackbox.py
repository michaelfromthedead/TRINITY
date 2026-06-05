"""
Blackbox tests for Attenuation component.

Tests PUBLIC behavior only - no internal state inspection.
Covers: linear, inverse, logarithmic attenuation models, cone attenuation.
"""

import pytest
import math

from engine.audio.spatial import (
    AttenuationModel,
    AttenuationCurve,
    LinearAttenuation,
    LogarithmicAttenuation,
    InverseAttenuation,
    InverseSquaredAttenuation,
    NoAttenuation,
    CustomCurveAttenuation,
    CurvePoint,
    ConeAttenuation,
    create_attenuation,
    MIN_ATTENUATION_DISTANCE,
    MAX_ATTENUATION_DISTANCE,
    DEFAULT_ROLLOFF,
    CONE_INNER_ANGLE,
    CONE_OUTER_ANGLE,
    CONE_OUTER_GAIN,
)


class TestAttenuationModels:
    """Test different attenuation model types."""

    def test_linear_model_exists(self):
        """Linear attenuation model exists."""
        assert AttenuationModel.LINEAR is not None

    def test_inverse_model_exists(self):
        """Inverse attenuation model exists."""
        assert AttenuationModel.INVERSE is not None

    def test_inverse_squared_model_exists(self):
        """Inverse squared attenuation model exists."""
        assert AttenuationModel.INVERSE_SQUARED is not None

    def test_logarithmic_model_exists(self):
        """Logarithmic attenuation model exists."""
        assert AttenuationModel.LOGARITHMIC is not None

    def test_custom_model_exists(self):
        """Custom attenuation model exists."""
        assert AttenuationModel.CUSTOM is not None

    def test_none_model_exists(self):
        """None (no attenuation) model exists."""
        assert AttenuationModel.NONE is not None


class TestLinearAttenuation:
    """Test linear distance attenuation."""

    def test_create_linear_attenuation(self):
        """Linear attenuation can be created."""
        atten = LinearAttenuation(min_distance=1.0, max_distance=100.0)
        assert atten is not None

    def test_linear_at_min_distance_is_unity(self):
        """At min distance, gain is 1.0."""
        atten = LinearAttenuation(min_distance=1.0, max_distance=100.0)
        gain = atten.calculate(distance=1.0)
        assert gain == pytest.approx(1.0)

    def test_linear_at_max_distance_is_zero(self):
        """At max distance, gain is 0.0."""
        atten = LinearAttenuation(min_distance=1.0, max_distance=100.0)
        gain = atten.calculate(distance=100.0)
        assert gain == pytest.approx(0.0)

    def test_linear_below_min_is_unity(self):
        """Below min distance, gain is clamped to 1.0."""
        atten = LinearAttenuation(min_distance=10.0, max_distance=100.0)
        gain = atten.calculate(distance=5.0)
        assert gain == pytest.approx(1.0)

    def test_linear_beyond_max_is_zero(self):
        """Beyond max distance, gain is clamped to 0.0."""
        atten = LinearAttenuation(min_distance=1.0, max_distance=100.0)
        gain = atten.calculate(distance=150.0)
        assert gain == pytest.approx(0.0)

    def test_linear_monotonically_decreasing(self):
        """Gain decreases monotonically with distance."""
        atten = LinearAttenuation(min_distance=1.0, max_distance=100.0)
        prev_gain = 1.0
        for d in range(1, 101, 5):
            gain = atten.calculate(distance=float(d))
            assert gain <= prev_gain
            prev_gain = gain

    def test_linear_with_rolloff(self):
        """Rolloff affects linear attenuation rate."""
        atten_r1 = LinearAttenuation(min_distance=1.0, max_distance=100.0, rolloff=1.0)
        atten_r05 = LinearAttenuation(min_distance=1.0, max_distance=100.0, rolloff=0.5)

        gain_r1 = atten_r1.calculate(distance=50.0)
        gain_r05 = atten_r05.calculate(distance=50.0)

        # Lower rolloff = slower attenuation
        assert gain_r05 > gain_r1


class TestInverseAttenuation:
    """Test inverse distance attenuation (1/d)."""

    def test_create_inverse_attenuation(self):
        """Inverse attenuation can be created."""
        atten = InverseAttenuation(min_distance=1.0, max_distance=100.0)
        assert atten is not None

    def test_inverse_at_min_distance_is_unity(self):
        """At min distance, gain is 1.0."""
        atten = InverseAttenuation(min_distance=1.0, max_distance=100.0)
        gain = atten.calculate(distance=1.0)
        assert gain == pytest.approx(1.0)

    def test_inverse_decreases_with_distance(self):
        """Gain decreases with distance."""
        atten = InverseAttenuation(min_distance=1.0, max_distance=100.0, rolloff=1.0)
        gain_near = atten.calculate(distance=2.0)
        gain_far = atten.calculate(distance=10.0)
        assert gain_far < gain_near

    def test_inverse_with_rolloff(self):
        """Rolloff factor affects attenuation rate."""
        atten_r1 = InverseAttenuation(min_distance=1.0, max_distance=100.0, rolloff=1.0)
        atten_r2 = InverseAttenuation(min_distance=1.0, max_distance=100.0, rolloff=2.0)

        gain_r1 = atten_r1.calculate(distance=5.0)
        gain_r2 = atten_r2.calculate(distance=5.0)

        # Higher rolloff = faster attenuation
        assert gain_r2 < gain_r1

    def test_inverse_never_exceeds_unity(self):
        """Gain never exceeds 1.0 even at close distances."""
        atten = InverseAttenuation(min_distance=10.0, max_distance=100.0)
        gain = atten.calculate(distance=1.0)
        assert gain <= 1.0


class TestInverseSquaredAttenuation:
    """Test inverse squared distance attenuation (1/d^2)."""

    def test_create_inverse_squared_attenuation(self):
        """Inverse squared attenuation can be created."""
        atten = InverseSquaredAttenuation(min_distance=1.0, max_distance=100.0)
        assert atten is not None

    def test_inverse_squared_at_min_is_unity(self):
        """At min distance, gain is 1.0."""
        atten = InverseSquaredAttenuation(min_distance=1.0, max_distance=100.0)
        gain = atten.calculate(distance=1.0)
        assert gain == pytest.approx(1.0)

    def test_inverse_squared_decreases_with_distance(self):
        """Gain decreases with distance."""
        atten = InverseSquaredAttenuation(min_distance=1.0, max_distance=100.0, rolloff=1.0)
        gain_near = atten.calculate(distance=2.0)
        gain_far = atten.calculate(distance=10.0)
        assert gain_far < gain_near

    def test_inverse_squared_falls_faster_than_inverse(self):
        """Inverse squared falls faster than inverse."""
        inv = InverseAttenuation(min_distance=1.0, max_distance=100.0, rolloff=1.0)
        inv_sq = InverseSquaredAttenuation(min_distance=1.0, max_distance=100.0, rolloff=1.0)

        gain_inv = inv.calculate(distance=10.0)
        gain_inv_sq = inv_sq.calculate(distance=10.0)

        assert gain_inv_sq < gain_inv


class TestLogarithmicAttenuation:
    """Test logarithmic distance attenuation."""

    def test_create_logarithmic_attenuation(self):
        """Logarithmic attenuation can be created."""
        atten = LogarithmicAttenuation(min_distance=1.0, max_distance=100.0)
        assert atten is not None

    def test_logarithmic_at_min_is_unity(self):
        """At min distance, gain is 1.0."""
        atten = LogarithmicAttenuation(min_distance=1.0, max_distance=100.0)
        gain = atten.calculate(distance=1.0)
        assert gain == pytest.approx(1.0)

    def test_logarithmic_decreases_with_distance(self):
        """Gain decreases with distance."""
        atten = LogarithmicAttenuation(min_distance=1.0, max_distance=100.0)
        gain_near = atten.calculate(distance=2.0)
        gain_far = atten.calculate(distance=50.0)
        assert gain_far < gain_near


class TestNoAttenuation:
    """Test no attenuation (constant gain)."""

    def test_create_no_attenuation(self):
        """No attenuation can be created."""
        atten = NoAttenuation()
        assert atten is not None

    def test_no_attenuation_always_unity(self):
        """No attenuation always returns 1.0."""
        atten = NoAttenuation()

        assert atten.calculate(distance=0.0) == pytest.approx(1.0)
        assert atten.calculate(distance=1.0) == pytest.approx(1.0)
        assert atten.calculate(distance=100.0) == pytest.approx(1.0)
        assert atten.calculate(distance=10000.0) == pytest.approx(1.0)


class TestCustomCurveAttenuation:
    """Test custom curve attenuation."""

    def test_create_custom_curve(self):
        """Custom curve can be created from points."""
        points = [
            CurvePoint(distance=0.0, gain=1.0),
            CurvePoint(distance=50.0, gain=0.5),
            CurvePoint(distance=100.0, gain=0.0),
        ]
        atten = CustomCurveAttenuation(points=points)
        assert atten is not None

    def test_custom_curve_at_defined_points(self):
        """Custom curve returns exact values at defined points."""
        points = [
            CurvePoint(distance=0.0, gain=1.0),
            CurvePoint(distance=50.0, gain=0.5),
            CurvePoint(distance=100.0, gain=0.0),
        ]
        atten = CustomCurveAttenuation(points=points)

        assert atten.calculate(distance=0.0) == pytest.approx(1.0)
        assert atten.calculate(distance=50.0) == pytest.approx(0.5)
        assert atten.calculate(distance=100.0) == pytest.approx(0.0)

    def test_custom_curve_interpolation(self):
        """Custom curve interpolates between points."""
        points = [
            CurvePoint(distance=0.0, gain=1.0),
            CurvePoint(distance=100.0, gain=0.0),
        ]
        atten = CustomCurveAttenuation(points=points)

        # Linear interpolation: at 50, should be around 0.5
        gain = atten.calculate(distance=50.0)
        assert 0.3 <= gain <= 0.7

    def test_custom_curve_nonlinear(self):
        """Custom curve can be nonlinear."""
        points = [
            CurvePoint(distance=0.0, gain=1.0),
            CurvePoint(distance=25.0, gain=0.9),
            CurvePoint(distance=75.0, gain=0.3),
            CurvePoint(distance=100.0, gain=0.0),
        ]
        atten = CustomCurveAttenuation(points=points)

        # Should follow the nonlinear curve
        gain_25 = atten.calculate(distance=25.0)
        gain_75 = atten.calculate(distance=75.0)

        assert gain_25 == pytest.approx(0.9)
        assert gain_75 == pytest.approx(0.3)


class TestConeAttenuation:
    """Test directional cone attenuation."""

    def test_create_cone_attenuation(self):
        """Cone attenuation can be created."""
        cone = ConeAttenuation(
            inner_angle=45.0,
            outer_angle=90.0,
            outer_gain=0.0
        )
        assert cone is not None

    def test_cone_inside_inner_is_unity(self):
        """Inside inner cone, gain is 1.0."""
        cone = ConeAttenuation(inner_angle=90.0, outer_angle=180.0, outer_gain=0.0)
        # Forward direction: source facing listener directly
        source_direction = (0, 0, 1)
        to_listener = (0, 0, 1)
        gain = cone.calculate(source_direction=source_direction, to_listener=to_listener)
        assert gain == pytest.approx(1.0)

    def test_cone_outside_outer_is_outer_gain(self):
        """Outside outer cone, gain is outer_gain."""
        cone = ConeAttenuation(inner_angle=45.0, outer_angle=90.0, outer_gain=0.3)
        # Source facing opposite direction
        source_direction = (0, 0, 1)
        to_listener = (0, 0, -1)  # Behind the source
        gain = cone.calculate(source_direction=source_direction, to_listener=to_listener)
        assert gain == pytest.approx(0.3)

    def test_cone_between_angles_interpolates(self):
        """Between inner and outer, gain interpolates."""
        cone = ConeAttenuation(inner_angle=0.0, outer_angle=180.0, outer_gain=0.0)
        source_direction = (1, 0, 0)
        to_listener = (0, 1, 0)  # 90 degrees off-axis
        gain = cone.calculate(source_direction=source_direction, to_listener=to_listener)
        assert 0.0 < gain < 1.0


class TestCreateAttenuation:
    """Test factory function for creating attenuation."""

    def test_create_linear_via_factory(self):
        """Factory creates linear attenuation."""
        atten = create_attenuation(
            model=AttenuationModel.LINEAR,
            min_distance=1.0,
            max_distance=100.0
        )
        assert isinstance(atten, LinearAttenuation)

    def test_create_inverse_via_factory(self):
        """Factory creates inverse attenuation."""
        atten = create_attenuation(
            model=AttenuationModel.INVERSE,
            min_distance=1.0,
            max_distance=100.0
        )
        assert isinstance(atten, InverseAttenuation)

    def test_create_inverse_squared_via_factory(self):
        """Factory creates inverse squared attenuation."""
        atten = create_attenuation(
            model=AttenuationModel.INVERSE_SQUARED,
            min_distance=1.0,
            max_distance=100.0
        )
        assert isinstance(atten, InverseSquaredAttenuation)

    def test_create_none_via_factory(self):
        """Factory creates no attenuation."""
        atten = create_attenuation(model=AttenuationModel.NONE)
        assert isinstance(atten, NoAttenuation)


class TestAttenuationConstants:
    """Test attenuation constants."""

    def test_min_distance_is_positive(self):
        """Min distance constant is positive."""
        assert MIN_ATTENUATION_DISTANCE > 0

    def test_max_distance_greater_than_min(self):
        """Max distance is greater than min distance."""
        assert MAX_ATTENUATION_DISTANCE > MIN_ATTENUATION_DISTANCE

    def test_default_rolloff_is_positive(self):
        """Default rolloff is positive."""
        assert DEFAULT_ROLLOFF > 0

    def test_cone_inner_angle_is_valid(self):
        """Cone inner angle is valid."""
        assert 0 <= CONE_INNER_ANGLE <= 360

    def test_cone_outer_angle_is_valid(self):
        """Cone outer angle is valid."""
        assert 0 <= CONE_OUTER_ANGLE <= 360

    def test_cone_outer_angle_gte_inner(self):
        """Cone outer angle >= inner angle."""
        assert CONE_OUTER_ANGLE >= CONE_INNER_ANGLE

    def test_cone_outer_gain_is_valid(self):
        """Cone outer gain is in valid range."""
        assert 0 <= CONE_OUTER_GAIN <= 1


class TestAttenuationEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_attenuation_at_zero_distance(self):
        """Attenuation at zero distance is unity."""
        atten = LinearAttenuation(min_distance=0.0, max_distance=100.0)
        gain = atten.calculate(distance=0.0)
        assert gain == pytest.approx(1.0)

    def test_attenuation_at_negative_distance(self):
        """Attenuation handles negative distance."""
        atten = LinearAttenuation(min_distance=1.0, max_distance=100.0)
        # Should clamp or handle gracefully
        gain = atten.calculate(distance=-5.0)
        assert gain >= 0.0

    def test_equal_min_max_distance(self):
        """Equal min and max distance is handled."""
        # This is a degenerate case
        atten = LinearAttenuation(min_distance=10.0, max_distance=10.0)
        gain = atten.calculate(distance=10.0)
        assert 0.0 <= gain <= 1.0


class TestAttenuationMath:
    """Test mathematical properties of attenuation."""

    def test_linear_is_continuous(self):
        """Linear attenuation is continuous."""
        atten = LinearAttenuation(min_distance=0.0, max_distance=100.0)

        for d in range(100):
            gain1 = atten.calculate(distance=float(d))
            gain2 = atten.calculate(distance=float(d + 1))
            delta = abs(gain1 - gain2)
            assert delta < 0.05  # Small change between adjacent distances

    def test_gain_never_negative(self):
        """Gain is never negative for any model."""
        models = [
            LinearAttenuation(min_distance=1.0, max_distance=100.0),
            InverseAttenuation(min_distance=1.0, max_distance=100.0),
            InverseSquaredAttenuation(min_distance=1.0, max_distance=100.0),
            LogarithmicAttenuation(min_distance=1.0, max_distance=100.0),
            NoAttenuation(),
        ]

        for atten in models:
            for d in [0, 0.1, 1, 10, 100, 1000]:
                gain = atten.calculate(distance=float(d))
                assert gain >= 0.0

    def test_gain_never_exceeds_unity(self):
        """Gain never exceeds 1.0 for any distance."""
        models = [
            LinearAttenuation(min_distance=1.0, max_distance=100.0),
            InverseAttenuation(min_distance=1.0, max_distance=100.0),
            InverseSquaredAttenuation(min_distance=1.0, max_distance=100.0),
            LogarithmicAttenuation(min_distance=1.0, max_distance=100.0),
            NoAttenuation(),
        ]

        for atten in models:
            for d in [0, 0.1, 1, 10, 100, 1000]:
                gain = atten.calculate(distance=float(d))
                assert gain <= 1.0

    def test_monotonic_decrease(self):
        """Gain decreases monotonically with distance (except NoAttenuation)."""
        models = [
            LinearAttenuation(min_distance=1.0, max_distance=100.0, rolloff=1.0),
            InverseAttenuation(min_distance=1.0, max_distance=100.0, rolloff=1.0),
            InverseSquaredAttenuation(min_distance=1.0, max_distance=100.0, rolloff=1.0),
            LogarithmicAttenuation(min_distance=1.0, max_distance=100.0, rolloff=1.0),
        ]

        for atten in models:
            prev_gain = atten.calculate(distance=1.0)
            for d in [2, 5, 10, 20, 50, 100]:
                gain = atten.calculate(distance=float(d))
                assert gain <= prev_gain
                prev_gain = gain
