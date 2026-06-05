"""Whitebox tests for Distance Attenuation Models.

Tests internal implementation of:
- Linear attenuation curve
- Logarithmic attenuation curve
- Inverse distance attenuation
- Inverse squared (physically accurate) attenuation
- Custom curve with interpolation
- Cone directional attenuation
- AttenuationVolume shape calculations
- dB/linear conversions
"""

import math
from typing import List

import pytest

from engine.audio.spatial.attenuation import (
    ATTENUATION_PRESETS,
    AttenuationCurve,
    AttenuationVolume,
    ConeAttenuation,
    CurvePoint,
    CustomCurveAttenuation,
    InverseAttenuation,
    InverseSquaredAttenuation,
    LinearAttenuation,
    LogarithmicAttenuation,
    NoAttenuation,
    create_attenuation,
    db_to_linear,
    get_preset,
    linear_to_db,
)
from engine.audio.spatial.config import (
    CONE_INNER_ANGLE,
    CONE_OUTER_ANGLE,
    CONE_OUTER_GAIN,
    DEFAULT_ROLLOFF,
    MAX_ATTENUATION_DISTANCE,
    MIN_ATTENUATION_DISTANCE,
    AttenuationModel,
    AttenuationShape,
)
from engine.core.math.vec import Vec3


# =============================================================================
# dB/Linear Conversion Tests
# =============================================================================


class TestDbLinearConversion:
    """Test attenuation dB/linear conversions."""

    def test_db_to_linear_0db(self):
        """0 dB equals 1.0 linear."""
        assert db_to_linear(0.0) == pytest.approx(1.0, rel=1e-6)

    def test_db_to_linear_minus_6db(self):
        """-6 dB is approximately 0.5."""
        assert db_to_linear(-6.0206) == pytest.approx(0.5, rel=1e-3)

    def test_db_to_linear_plus_6db(self):
        """+6 dB is approximately 2.0."""
        assert db_to_linear(6.0206) == pytest.approx(2.0, rel=1e-3)

    def test_linear_to_db_1(self):
        """1.0 linear equals 0 dB."""
        assert linear_to_db(1.0) == pytest.approx(0.0, rel=1e-6)

    def test_linear_to_db_0_5(self):
        """0.5 linear is approximately -6 dB."""
        assert linear_to_db(0.5) == pytest.approx(-6.0206, rel=1e-3)

    def test_linear_to_db_zero(self):
        """Zero linear returns -96 dB."""
        assert linear_to_db(0.0) == -96.0

    def test_linear_to_db_negative(self):
        """Negative linear returns -96 dB."""
        assert linear_to_db(-0.5) == -96.0


# =============================================================================
# LinearAttenuation Tests
# =============================================================================


class TestLinearAttenuation:
    """Test LinearAttenuation curve."""

    def test_model_type(self):
        """Model type is LINEAR."""
        curve = LinearAttenuation()
        assert curve.model == AttenuationModel.LINEAR

    def test_default_values(self):
        """Default constructor values."""
        curve = LinearAttenuation()
        assert curve.min_distance == MIN_ATTENUATION_DISTANCE
        assert curve.max_distance == MAX_ATTENUATION_DISTANCE
        assert curve.rolloff == DEFAULT_ROLLOFF

    def test_custom_values(self):
        """Custom constructor values."""
        curve = LinearAttenuation(min_distance=2.0, max_distance=50.0, rolloff=1.5)
        assert curve.min_distance == 2.0
        assert curve.max_distance == 50.0
        assert curve.rolloff == 1.5

    def test_calculate_at_min_distance(self):
        """Full volume at or below min distance."""
        curve = LinearAttenuation(min_distance=5.0, max_distance=100.0)
        assert curve.calculate(5.0) == 1.0
        assert curve.calculate(3.0) == 1.0
        assert curve.calculate(0.0) == 1.0

    def test_calculate_at_max_distance(self):
        """Zero volume at or beyond max distance."""
        curve = LinearAttenuation(min_distance=1.0, max_distance=100.0)
        assert curve.calculate(100.0) == 0.0
        assert curve.calculate(150.0) == 0.0

    def test_calculate_midpoint(self):
        """Midpoint attenuation."""
        curve = LinearAttenuation(min_distance=1.0, max_distance=100.0, rolloff=1.0)
        # At midpoint between min and max
        midpoint = (1.0 + 100.0) / 2  # 50.5
        result = curve.calculate(midpoint)
        # Should be close to 0.5 (halfway)
        assert 0.4 < result < 0.6

    def test_calculate_linear_progression(self):
        """Linear progression across range."""
        curve = LinearAttenuation(min_distance=1.0, max_distance=100.0, rolloff=1.0)

        # Check that attenuation decreases with distance
        at_20 = curve.calculate(20.0)
        at_50 = curve.calculate(50.0)
        at_80 = curve.calculate(80.0)

        # Closer = louder
        assert at_20 > at_50
        assert at_50 > at_80
        assert at_80 > 0.0

    def test_calculate_with_rolloff(self):
        """Rolloff factor affects attenuation rate."""
        curve = LinearAttenuation(min_distance=0.0, max_distance=100.0, rolloff=2.0)

        # With rolloff=2, midpoint = 1 - 2 * 0.5 = 0, but clamped
        result = curve.calculate(25.0)
        assert result == pytest.approx(0.5, rel=1e-2)

    def test_calculate_db(self):
        """calculate_db converts to decibels."""
        curve = LinearAttenuation(min_distance=0.0, max_distance=100.0, rolloff=1.0)

        # At 50, linear = 0.5, dB = -6.02
        assert curve.calculate_db(50.0) == pytest.approx(-6.0206, rel=1e-2)


# =============================================================================
# LogarithmicAttenuation Tests
# =============================================================================


class TestLogarithmicAttenuation:
    """Test LogarithmicAttenuation curve."""

    def test_model_type(self):
        """Model type is LOGARITHMIC."""
        curve = LogarithmicAttenuation()
        assert curve.model == AttenuationModel.LOGARITHMIC

    def test_calculate_at_min_distance(self):
        """Full volume at or below min distance."""
        curve = LogarithmicAttenuation(min_distance=5.0)
        assert curve.calculate(5.0) == 1.0
        assert curve.calculate(3.0) == 1.0

    def test_calculate_at_max_distance(self):
        """Zero volume at or beyond max distance."""
        curve = LogarithmicAttenuation(max_distance=100.0)
        assert curve.calculate(100.0) == 0.0
        assert curve.calculate(150.0) == 0.0

    def test_calculate_falloff_shape(self):
        """Logarithmic falloff is less aggressive at start."""
        curve = LogarithmicAttenuation(min_distance=1.0, max_distance=100.0)

        # Logarithmic should have gentler falloff near min
        at_2x = curve.calculate(2.0)
        at_4x = curve.calculate(4.0)

        # Should both be positive
        assert at_2x > 0.0
        assert at_4x > 0.0
        assert at_2x > at_4x  # Further = quieter

    def test_calculate_smooth_to_zero(self):
        """Smoothly fades to zero near max distance."""
        curve = LogarithmicAttenuation(min_distance=1.0, max_distance=100.0)

        # Near the 80% mark, should start fading
        at_80 = curve.calculate(80.0)
        at_90 = curve.calculate(90.0)

        assert at_80 > at_90
        assert at_90 > 0.0


# =============================================================================
# InverseAttenuation Tests
# =============================================================================


class TestInverseAttenuation:
    """Test InverseAttenuation curve."""

    def test_model_type(self):
        """Model type is INVERSE."""
        curve = InverseAttenuation()
        assert curve.model == AttenuationModel.INVERSE

    def test_calculate_at_min_distance(self):
        """Full volume at or below min distance."""
        curve = InverseAttenuation(min_distance=5.0)
        assert curve.calculate(5.0) == 1.0
        assert curve.calculate(3.0) == 1.0

    def test_calculate_at_max_distance(self):
        """Zero volume at or beyond max distance."""
        curve = InverseAttenuation(max_distance=100.0)
        assert curve.calculate(100.0) == 0.0

    def test_calculate_inverse_formula(self):
        """Follows inverse distance formula."""
        curve = InverseAttenuation(min_distance=1.0, max_distance=1000.0, rolloff=1.0)

        # At distance 2, formula: 1 / (1 + 1*(2-1)) = 1/2 = 0.5
        # But may have fade factor
        result = curve.calculate(2.0)
        assert 0.3 < result < 0.7

    def test_calculate_with_rolloff(self):
        """Rolloff affects inverse rate."""
        curve_normal = InverseAttenuation(min_distance=1.0, max_distance=100.0, rolloff=1.0)
        curve_fast = InverseAttenuation(min_distance=1.0, max_distance=100.0, rolloff=2.0)

        at_10_normal = curve_normal.calculate(10.0)
        at_10_fast = curve_fast.calculate(10.0)

        # Higher rolloff = faster falloff = lower value
        assert at_10_fast < at_10_normal


# =============================================================================
# InverseSquaredAttenuation Tests
# =============================================================================


class TestInverseSquaredAttenuation:
    """Test InverseSquaredAttenuation (physically accurate)."""

    def test_model_type(self):
        """Model type is INVERSE_SQUARED."""
        curve = InverseSquaredAttenuation()
        assert curve.model == AttenuationModel.INVERSE_SQUARED

    def test_calculate_at_min_distance(self):
        """Full volume at or below min distance."""
        curve = InverseSquaredAttenuation(min_distance=5.0)
        assert curve.calculate(5.0) == 1.0
        assert curve.calculate(3.0) == 1.0

    def test_calculate_at_max_distance(self):
        """Zero volume at or beyond max distance."""
        curve = InverseSquaredAttenuation(max_distance=100.0)
        assert curve.calculate(100.0) == 0.0

    def test_calculate_inverse_square_law(self):
        """Follows inverse square law."""
        curve = InverseSquaredAttenuation(min_distance=1.0, max_distance=1000.0, rolloff=1.0)

        # At distance 2, formula: (1/2)^2 = 0.25
        result = curve.calculate(2.0)
        assert 0.1 < result < 0.4

        # At distance 4, should be ~0.0625
        result_4 = curve.calculate(4.0)
        assert result_4 < result

    def test_calculate_faster_falloff_than_inverse(self):
        """Falls off faster than linear inverse."""
        inv_curve = InverseAttenuation(min_distance=1.0, max_distance=100.0, rolloff=1.0)
        inv_sq_curve = InverseSquaredAttenuation(min_distance=1.0, max_distance=100.0, rolloff=1.0)

        at_20_inv = inv_curve.calculate(20.0)
        at_20_inv_sq = inv_sq_curve.calculate(20.0)

        # Inverse squared falls faster
        assert at_20_inv_sq < at_20_inv


# =============================================================================
# NoAttenuation Tests
# =============================================================================


class TestNoAttenuation:
    """Test NoAttenuation curve."""

    def test_model_type(self):
        """Model type is NONE."""
        curve = NoAttenuation()
        assert curve.model == AttenuationModel.NONE

    def test_calculate_constant_volume(self):
        """Constant volume within range."""
        curve = NoAttenuation(max_distance=100.0)

        assert curve.calculate(0.0) == 1.0
        assert curve.calculate(50.0) == 1.0
        assert curve.calculate(99.0) == 1.0

    def test_calculate_culled_at_max(self):
        """Zero at and beyond max distance."""
        curve = NoAttenuation(max_distance=100.0)

        assert curve.calculate(100.0) == 0.0
        assert curve.calculate(150.0) == 0.0


# =============================================================================
# CustomCurveAttenuation Tests
# =============================================================================


class TestCustomCurveAttenuation:
    """Test CustomCurveAttenuation with designer points."""

    def test_model_type(self):
        """Model type is CUSTOM."""
        curve = CustomCurveAttenuation(
            points=[CurvePoint(0.0, 1.0), CurvePoint(100.0, 0.0)]
        )
        assert curve.model == AttenuationModel.CUSTOM

    def test_minimum_points(self):
        """Ensures at least 2 points."""
        curve = CustomCurveAttenuation(points=[CurvePoint(0.0, 1.0)])

        assert len(curve.points) >= 2

    def test_points_sorted(self):
        """Points are sorted by distance."""
        curve = CustomCurveAttenuation(
            points=[
                CurvePoint(50.0, 0.5),
                CurvePoint(0.0, 1.0),
                CurvePoint(100.0, 0.0),
            ]
        )

        distances = [p.distance for p in curve.points]
        assert distances == sorted(distances)

    def test_calculate_at_points(self):
        """Calculate returns exact values at defined points."""
        curve = CustomCurveAttenuation(
            points=[
                CurvePoint(0.0, 1.0),
                CurvePoint(50.0, 0.5),
                CurvePoint(100.0, 0.0),
            ],
            smooth=False,
        )

        assert curve.calculate(0.0) == pytest.approx(1.0, rel=1e-3)
        assert curve.calculate(50.0) == pytest.approx(0.5, rel=1e-3)
        assert curve.calculate(100.0) == pytest.approx(0.0, rel=1e-3)

    def test_calculate_interpolated_linear(self):
        """Linear interpolation between points."""
        curve = CustomCurveAttenuation(
            points=[
                CurvePoint(0.0, 1.0),
                CurvePoint(100.0, 0.0),
            ],
            smooth=False,
        )

        assert curve.calculate(25.0) == pytest.approx(0.75, rel=1e-2)
        assert curve.calculate(75.0) == pytest.approx(0.25, rel=1e-2)

    def test_calculate_interpolated_smooth(self):
        """Smooth (Hermite) interpolation between points."""
        curve = CustomCurveAttenuation(
            points=[
                CurvePoint(0.0, 1.0),
                CurvePoint(100.0, 0.0),
            ],
            smooth=True,
        )

        # Smooth interpolation at midpoint
        result = curve.calculate(50.0)
        assert 0.4 < result < 0.6

    def test_add_point(self):
        """Add point to curve."""
        curve = CustomCurveAttenuation(
            points=[CurvePoint(0.0, 1.0), CurvePoint(100.0, 0.0)]
        )

        curve.add_point(50.0, 0.5)

        assert len(curve.points) == 3
        # Points should still be sorted
        distances = [p.distance for p in curve.points]
        assert distances == sorted(distances)

    def test_remove_point(self):
        """Remove point from curve."""
        curve = CustomCurveAttenuation(
            points=[
                CurvePoint(0.0, 1.0),
                CurvePoint(50.0, 0.5),
                CurvePoint(100.0, 0.0),
            ]
        )

        result = curve.remove_point(1)

        assert result is True
        assert len(curve.points) == 2

    def test_remove_point_minimum_preserved(self):
        """Cannot remove below 2 points."""
        curve = CustomCurveAttenuation(
            points=[CurvePoint(0.0, 1.0), CurvePoint(100.0, 0.0)]
        )

        result = curve.remove_point(0)

        assert result is False
        assert len(curve.points) == 2

    def test_remove_point_invalid_index(self):
        """Remove with invalid index returns False."""
        curve = CustomCurveAttenuation(
            points=[
                CurvePoint(0.0, 1.0),
                CurvePoint(50.0, 0.5),
                CurvePoint(100.0, 0.0),
            ]
        )

        result = curve.remove_point(10)
        assert result is False


# =============================================================================
# CurvePoint Tests
# =============================================================================


class TestCurvePoint:
    """Test CurvePoint dataclass."""

    def test_values_clamped(self):
        """Distance and value are clamped."""
        point = CurvePoint(distance=-10.0, value=2.0)

        assert point.distance == 0.0
        assert point.value == 1.0

        point2 = CurvePoint(distance=50.0, value=-0.5)
        assert point2.distance == 50.0
        assert point2.value == 0.0


# =============================================================================
# ConeAttenuation Tests
# =============================================================================


class TestConeAttenuation:
    """Test directional cone attenuation."""

    def test_default_values(self):
        """Default cone values."""
        cone = ConeAttenuation()
        assert cone.inner_angle == CONE_INNER_ANGLE
        assert cone.outer_angle == CONE_OUTER_ANGLE
        assert cone.outer_gain == CONE_OUTER_GAIN

    def test_angles_clamped(self):
        """Angles are clamped to valid range."""
        cone = ConeAttenuation(inner_angle=-10.0, outer_angle=400.0, outer_gain=2.0)

        assert cone.inner_angle == 0.0
        assert cone.outer_angle == 360.0
        assert cone.outer_gain == 1.0

    def test_calculate_within_inner(self):
        """Full volume within inner cone."""
        cone = ConeAttenuation(inner_angle=60.0, outer_angle=120.0)

        # Source facing forward, listener in front
        source_dir = Vec3.forward()
        to_listener = Vec3.forward()

        result = cone.calculate(source_dir, to_listener)
        assert result == 1.0

    def test_calculate_beyond_outer(self):
        """Outer gain beyond outer cone."""
        cone = ConeAttenuation(inner_angle=60.0, outer_angle=120.0, outer_gain=0.3)

        # Listener behind source (180 degrees from forward)
        source_dir = Vec3(0, 0, 1)  # Forward
        to_listener = Vec3(0, 0, -1).normalized()  # Behind, normalized

        result = cone.calculate(source_dir, to_listener)
        # Beyond outer cone should be at outer_gain
        assert result == pytest.approx(0.3, rel=0.05)

    def test_calculate_in_transition(self):
        """Smooth transition between inner and outer."""
        cone = ConeAttenuation(inner_angle=60.0, outer_angle=120.0, outer_gain=0.0)

        source_dir = Vec3(0, 0, 1)  # Forward

        # Within inner cone (30 degrees from center, inner half-angle is 30)
        angle_rad = math.radians(20)  # Well within inner
        to_listener_inner = Vec3(math.sin(angle_rad), 0, math.cos(angle_rad)).normalized()
        result_inner = cone.calculate(source_dir, to_listener_inner)
        assert result_inner == pytest.approx(1.0, rel=0.05)

        # Between inner and outer (45 degrees, between 30 and 60 half-angles)
        angle_rad = math.radians(45)
        to_listener_mid = Vec3(math.sin(angle_rad), 0, math.cos(angle_rad)).normalized()
        result_mid = cone.calculate(source_dir, to_listener_mid)
        # Should be between outer_gain (0) and 1.0
        assert 0.0 < result_mid < 1.0

    def test_calculate_360_degree_cone(self):
        """360 degree cone is omnidirectional."""
        cone = ConeAttenuation(inner_angle=360.0)

        source_dir = Vec3.forward()
        to_listener = Vec3(0, 0, -1)  # Behind

        result = cone.calculate(source_dir, to_listener)
        assert result == 1.0


# =============================================================================
# AttenuationVolume Tests
# =============================================================================


class TestAttenuationVolume:
    """Test AttenuationVolume shape calculations."""

    def test_sphere_distance(self):
        """Sphere shape calculates distance from center."""
        volume = AttenuationVolume(
            shape=AttenuationShape.SPHERE,
            center=Vec3(10, 0, 0),
        )

        listener = Vec3(15, 0, 0)
        distance = volume.get_distance(listener)

        assert distance == pytest.approx(5.0, rel=1e-6)

    def test_box_distance_inside(self):
        """Box distance is 0 when listener inside."""
        volume = AttenuationVolume(
            shape=AttenuationShape.BOX,
            center=Vec3(0, 0, 0),
            half_extents=Vec3(5, 5, 5),
        )

        listener = Vec3(2, 2, 2)
        distance = volume.get_distance(listener)

        assert distance == pytest.approx(0.0, rel=1e-6)

    def test_box_distance_outside(self):
        """Box distance to nearest point when outside."""
        volume = AttenuationVolume(
            shape=AttenuationShape.BOX,
            center=Vec3(0, 0, 0),
            half_extents=Vec3(5, 5, 5),
        )

        listener = Vec3(10, 0, 0)  # 5 units past edge
        distance = volume.get_distance(listener)

        assert distance == pytest.approx(5.0, rel=1e-6)

    def test_capsule_distance(self):
        """Capsule shape distance calculation."""
        volume = AttenuationVolume(
            shape=AttenuationShape.CAPSULE,
            center=Vec3(0, 0, 0),
            half_extents=Vec3(2, 10, 2),  # Radius 2, height 20
        )

        # Listener on axis
        listener = Vec3(0, 5, 0)
        distance = volume.get_distance(listener)

        # Should be 0 as listener is inside capsule body
        assert distance < 0.1

    def test_calculate_with_curve(self):
        """calculate combines distance and curve."""
        volume = AttenuationVolume(
            shape=AttenuationShape.SPHERE,
            center=Vec3(0, 0, 0),
            curve=LinearAttenuation(min_distance=0.0, max_distance=100.0),
        )

        listener = Vec3(50, 0, 0)
        attenuation = volume.calculate(listener)

        assert attenuation == pytest.approx(0.5, rel=1e-2)

    def test_calculate_with_cone(self):
        """calculate applies cone attenuation."""
        volume = AttenuationVolume(
            shape=AttenuationShape.SPHERE,
            center=Vec3(0, 0, 0),
            direction=Vec3(0, 0, 1),  # Forward
            curve=NoAttenuation(max_distance=100.0),
            cone=ConeAttenuation(inner_angle=60.0, outer_angle=120.0, outer_gain=0.0),
        )

        # Listener in front (full volume)
        listener_front = Vec3(0, 0, 10)
        result_front = volume.calculate(listener_front)
        assert result_front == pytest.approx(1.0, rel=0.05)

        # Listener behind (outer gain)
        listener_back = Vec3(0, 0, -10)
        result_back = volume.calculate(listener_back)
        # Should be at or near outer_gain (0.0)
        assert result_back < 0.5


# =============================================================================
# create_attenuation Factory Tests
# =============================================================================


class TestCreateAttenuation:
    """Test create_attenuation factory function."""

    def test_create_linear(self):
        """Create linear attenuation."""
        curve = create_attenuation(AttenuationModel.LINEAR, min_distance=2.0, max_distance=50.0)

        assert isinstance(curve, LinearAttenuation)
        assert curve.min_distance == 2.0
        assert curve.max_distance == 50.0

    def test_create_logarithmic(self):
        """Create logarithmic attenuation."""
        curve = create_attenuation(AttenuationModel.LOGARITHMIC)
        assert isinstance(curve, LogarithmicAttenuation)

    def test_create_inverse(self):
        """Create inverse attenuation."""
        curve = create_attenuation(AttenuationModel.INVERSE)
        assert isinstance(curve, InverseAttenuation)

    def test_create_inverse_squared(self):
        """Create inverse squared attenuation."""
        curve = create_attenuation(AttenuationModel.INVERSE_SQUARED)
        assert isinstance(curve, InverseSquaredAttenuation)

    def test_create_none(self):
        """Create no attenuation."""
        curve = create_attenuation(AttenuationModel.NONE)
        assert isinstance(curve, NoAttenuation)

    def test_create_custom(self):
        """Create custom curve attenuation."""
        points = [CurvePoint(0.0, 1.0), CurvePoint(100.0, 0.0)]
        curve = create_attenuation(
            AttenuationModel.CUSTOM,
            points=points,
            smooth=True,
        )

        assert isinstance(curve, CustomCurveAttenuation)
        assert len(curve.points) == 2

    def test_create_unknown_raises(self):
        """Unknown model raises ValueError."""
        with pytest.raises(ValueError):
            create_attenuation("invalid_model")


# =============================================================================
# Preset Tests
# =============================================================================


class TestAttenuationPresets:
    """Test attenuation presets."""

    def test_all_presets_exist(self):
        """All presets can be retrieved."""
        preset_names = ["realistic", "linear", "ambient", "dialog", "explosion", "whisper", "global"]

        for name in preset_names:
            preset = get_preset(name)
            assert preset is not None
            assert isinstance(preset, AttenuationCurve)

    def test_preset_not_found(self):
        """Unknown preset returns None."""
        result = get_preset("nonexistent")
        assert result is None

    def test_realistic_preset(self):
        """Realistic preset is inverse squared."""
        preset = get_preset("realistic")
        assert isinstance(preset, InverseSquaredAttenuation)

    def test_global_preset(self):
        """Global preset has no attenuation."""
        preset = get_preset("global")
        assert isinstance(preset, NoAttenuation)


# =============================================================================
# Property Setter Tests
# =============================================================================


class TestAttenuationPropertySetters:
    """Test attenuation curve property setters."""

    def test_min_distance_setter(self):
        """min_distance setter clamps and updates max."""
        curve = LinearAttenuation(min_distance=10.0, max_distance=100.0)

        curve.min_distance = 5.0
        assert curve.min_distance == 5.0

        # Minimum enforced
        curve.min_distance = -10.0
        assert curve.min_distance >= 0.001

    def test_max_distance_setter(self):
        """max_distance setter clamps to min."""
        curve = LinearAttenuation(min_distance=10.0, max_distance=100.0)

        curve.max_distance = 50.0
        assert curve.max_distance == 50.0

        # Cannot go below min
        curve.max_distance = 5.0
        assert curve.max_distance >= curve.min_distance

    def test_rolloff_setter(self):
        """rolloff setter clamps negative."""
        curve = LinearAttenuation()

        curve.rolloff = 2.0
        assert curve.rolloff == 2.0

        curve.rolloff = -1.0
        assert curve.rolloff >= 0.0
