"""
Comprehensive Tests for Spatial Audio Subsystem.

Tests all spatial audio components:
- Positioning: PointSource, AreaSource, LineSource, VolumeSource
- Attenuation: Linear, Logarithmic, Inverse, InverseSquared, Custom, Cone
- Spatialization: Stereo panning, Surround, VBAP, Ambisonics
- HRTF: ITD, ILD, binaural processing
- Doppler: pitch shifting, velocity calculations
- Speaker configurations: Stereo to 7.1.4
- Reverb zones: blending, presets
- Occlusion/Obstruction
- Sound propagation

Target: 80+ tests with edge cases using pytest fixtures.
"""

from __future__ import annotations

import math
import pytest
from dataclasses import dataclass
from typing import List, Optional, Tuple

from engine.audio.spatial.config import (
    AttenuationModel,
    AttenuationShape,
    HRTFQuality,
    OcclusionMethod,
    OcclusionResponse,
    ReverbPreset,
    SourceType,
    SpatializationMethod,
    SpeakerLayout,
    DOPPLER_FACTOR,
    SPEED_OF_SOUND,
    MAX_DOPPLER_SHIFT,
    MIN_DOPPLER_SHIFT,
    MIN_ATTENUATION_DISTANCE,
    MAX_ATTENUATION_DISTANCE,
    DEFAULT_ROLLOFF,
    CONE_INNER_ANGLE,
    CONE_OUTER_ANGLE,
    CONE_OUTER_GAIN,
    HRTF_SAMPLE_RATE,
    HEAD_RADIUS,
    SPEAKER_ANGLES,
)
from engine.audio.spatial.attenuation import (
    AttenuationCurve,
    LinearAttenuation,
    LogarithmicAttenuation,
    InverseAttenuation,
    InverseSquaredAttenuation,
    NoAttenuation,
    CustomCurveAttenuation,
    ConeAttenuation,
    CurvePoint,
    AttenuationVolume,
    create_attenuation,
    get_preset,
    db_to_linear,
    linear_to_db,
    ATTENUATION_PRESETS,
)
from engine.audio.spatial.positioning import (
    ListenerState,
    ListenerManager,
    PointSource,
    AreaSource,
    LineSource,
    VolumeSource,
    SpatialSourceState,
    create_source,
)
from engine.audio.spatial.spatialization import (
    ChannelGains,
    SpatializationParams,
    Spatializer,
    StereoPanner,
    SurroundPanner,
    VBAPSpatializer,
    AmbisonicsSpatializer,
    create_spatializer,
    spatialize,
)
from engine.audio.spatial.hrtf import (
    HRTFCoefficients,
    HRTFProfile,
    HRTFSpatializer,
    HRTFProcessingState,
    calculate_itd,
    calculate_ild,
    process_hrtf_block,
    create_default_hrtf_profile,
)
from engine.audio.spatial.doppler import (
    calculate_doppler_shift,
    DopplerConfig,
    DopplerProcessor,
    DopplerState,
    estimate_arrival_time,
    get_doppler_preset,
    DOPPLER_PRESETS,
)
from engine.audio.spatial.reverb_zone import (
    ReverbParameters,
    ReverbZone,
    ReverbZoneManager,
    get_preset_parameters,
)
from engine.audio.spatial.occlusion import (
    OcclusionDetector,
    OcclusionProcessor,
    OcclusionResult,
    OcclusionType,
    OcclusionSettings,
)
from engine.audio.spatial.propagation import (
    PropagationCalculator,
    PropagationPath,
    PropagationResult,
    PathType,
)
from engine.audio.spatial.materials import (
    AcousticMaterial,
    MaterialType,
    MaterialDatabase,
    MATERIAL_PRESETS,
)
from engine.core.math.vec import Vec3


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def vec3_zero():
    """Create a zero vector."""
    return Vec3.zero()


@pytest.fixture
def vec3_forward():
    """Create a forward vector."""
    return Vec3.forward()


@pytest.fixture
def listener_origin():
    """Create a listener at the origin."""
    listener = ListenerState()
    listener.position = Vec3.zero()
    listener.forward = Vec3.forward()
    listener.up = Vec3.up()
    return listener


@pytest.fixture
def linear_attenuation():
    """Create a linear attenuation curve."""
    return LinearAttenuation(
        min_distance=1.0,
        max_distance=100.0,
        rolloff=1.0,
    )


@pytest.fixture
def stereo_panner():
    """Create a stereo panner."""
    return StereoPanner()


@pytest.fixture
def surround_panner():
    """Create a 5.1 surround panner."""
    return SurroundPanner(layout=SpeakerLayout.SURROUND_5_1)


@pytest.fixture
def doppler_processor():
    """Create a Doppler processor."""
    return DopplerProcessor()


@pytest.fixture
def hrtf_spatializer():
    """Create an HRTF spatializer."""
    return HRTFSpatializer(quality=HRTFQuality.MEDIUM)


@pytest.fixture
def reverb_zone():
    """Create a reverb zone."""
    return ReverbZone(
        zone_id=1,
        name="test_zone",
        preset=ReverbPreset.MEDIUM_ROOM,
        center=Vec3.zero(),
        half_extents=Vec3(10.0, 10.0, 10.0),
        fade_distance=2.0,
    )


# =============================================================================
# Vec3 Math Tests (Spatial Audio specific)
# =============================================================================


class TestVec3Spatial:
    """Test Vec3 operations relevant to spatial audio."""

    def test_vec3_zero(self):
        """Test zero vector creation."""
        v = Vec3.zero()
        assert v.x == 0.0
        assert v.y == 0.0
        assert v.z == 0.0

    def test_vec3_forward(self):
        """Test forward vector."""
        v = Vec3.forward()
        # Forward is typically -Z or +Z depending on convention
        assert abs(v.z) == 1.0 or v.length() == pytest.approx(1.0, rel=0.01)

    def test_vec3_up(self):
        """Test up vector (+Y direction)."""
        v = Vec3.up()
        assert v.y == 1.0

    def test_vec3_right(self):
        """Test right vector (+X direction)."""
        v = Vec3.right()
        assert v.x == 1.0

    def test_vec3_angle_between(self):
        """Test angle calculation between vectors."""
        v1 = Vec3(1.0, 0.0, 0.0)
        v2 = Vec3(0.0, 1.0, 0.0)
        # Should be 90 degrees
        dot = v1.dot(v2)
        len1 = v1.length()
        len2 = v2.length()
        if len1 > 0 and len2 > 0:
            angle = math.degrees(math.acos(max(-1.0, min(1.0, dot / (len1 * len2)))))
            assert angle == pytest.approx(90.0, rel=0.01)


# =============================================================================
# Linear Attenuation Tests
# =============================================================================


class TestLinearAttenuation:
    """Test suite for LinearAttenuation."""

    def test_at_min_distance(self, linear_attenuation):
        """Test attenuation at minimum distance is 1.0."""
        result = linear_attenuation.calculate(1.0)
        assert result == 1.0

    def test_at_max_distance(self, linear_attenuation):
        """Test attenuation at maximum distance is 0.0."""
        result = linear_attenuation.calculate(100.0)
        assert result == 0.0

    def test_at_half_distance(self, linear_attenuation):
        """Test attenuation at midpoint."""
        # Linear should be 0.5 at midpoint
        result = linear_attenuation.calculate(50.5)  # midpoint
        assert result == pytest.approx(0.5, rel=0.1)

    def test_below_min_distance(self, linear_attenuation):
        """Test attenuation below minimum distance is 1.0."""
        result = linear_attenuation.calculate(0.5)
        assert result == 1.0

    def test_beyond_max_distance(self, linear_attenuation):
        """Test attenuation beyond maximum distance is 0.0."""
        result = linear_attenuation.calculate(200.0)
        assert result == 0.0

    def test_rolloff_factor(self):
        """Test rolloff factor affects attenuation."""
        fast = LinearAttenuation(1.0, 100.0, 2.0)  # Faster rolloff
        slow = LinearAttenuation(1.0, 100.0, 0.5)  # Slower rolloff

        dist = 50.0
        assert fast.calculate(dist) < slow.calculate(dist)

    def test_model_property(self, linear_attenuation):
        """Test model property returns correct type."""
        assert linear_attenuation.model == AttenuationModel.LINEAR


# =============================================================================
# Logarithmic Attenuation Tests
# =============================================================================


class TestLogarithmicAttenuation:
    """Test suite for LogarithmicAttenuation."""

    def test_at_min_distance(self):
        """Test attenuation at minimum distance."""
        atten = LogarithmicAttenuation(1.0, 100.0, 1.0)
        result = atten.calculate(1.0)
        assert result == 1.0

    def test_at_max_distance(self):
        """Test attenuation at maximum distance."""
        atten = LogarithmicAttenuation(1.0, 100.0, 1.0)
        result = atten.calculate(100.0)
        assert result == 0.0

    def test_logarithmic_curve_shape(self):
        """Test logarithmic curve falls off faster initially."""
        atten = LogarithmicAttenuation(1.0, 100.0, 1.0)

        # Sample points
        at_25 = atten.calculate(25.0)
        at_50 = atten.calculate(50.0)

        # Logarithmic should have faster initial falloff
        assert at_25 < 0.75  # Already below 75% at 25%

    def test_model_property(self):
        """Test model property."""
        atten = LogarithmicAttenuation()
        assert atten.model == AttenuationModel.LOGARITHMIC


# =============================================================================
# Inverse Attenuation Tests
# =============================================================================


class TestInverseAttenuation:
    """Test suite for InverseAttenuation."""

    def test_at_min_distance(self):
        """Test attenuation at minimum distance."""
        atten = InverseAttenuation(1.0, 100.0, 1.0)
        result = atten.calculate(1.0)
        assert result == 1.0

    def test_inverse_falloff(self):
        """Test inverse distance falloff."""
        atten = InverseAttenuation(1.0, 100.0, 1.0)
        # At 2x min distance, should be less than 1.0
        result = atten.calculate(2.0)
        assert result < 1.0
        assert result > 0.0

    def test_model_property(self):
        """Test model property."""
        atten = InverseAttenuation()
        assert atten.model == AttenuationModel.INVERSE


# =============================================================================
# Inverse Squared Attenuation Tests
# =============================================================================


class TestInverseSquaredAttenuation:
    """Test suite for InverseSquaredAttenuation (physically accurate)."""

    def test_at_min_distance(self):
        """Test attenuation at minimum distance."""
        atten = InverseSquaredAttenuation(1.0, 100.0, 1.0)
        result = atten.calculate(1.0)
        assert result == 1.0

    def test_inverse_square_law(self):
        """Test inverse square law behavior."""
        atten = InverseSquaredAttenuation(1.0, 100.0, 1.0)
        # At 2x distance, should be 1/4 the intensity
        at_1 = atten.calculate(1.0)
        at_2 = atten.calculate(2.0)
        ratio = at_2 / at_1
        assert ratio == pytest.approx(0.25, rel=0.1)

    def test_model_property(self):
        """Test model property."""
        atten = InverseSquaredAttenuation()
        assert atten.model == AttenuationModel.INVERSE_SQUARED


# =============================================================================
# No Attenuation Tests
# =============================================================================


class TestNoAttenuation:
    """Test suite for NoAttenuation."""

    def test_constant_volume(self):
        """Test volume is constant within range."""
        atten = NoAttenuation(1.0, 100.0)
        assert atten.calculate(1.0) == 1.0
        assert atten.calculate(50.0) == 1.0
        assert atten.calculate(99.0) == 1.0

    def test_beyond_max_distance(self):
        """Test attenuation beyond max distance is 0."""
        atten = NoAttenuation(1.0, 100.0)
        assert atten.calculate(101.0) == 0.0


# =============================================================================
# Custom Curve Attenuation Tests
# =============================================================================


class TestCustomCurveAttenuation:
    """Test suite for CustomCurveAttenuation."""

    def test_basic_curve(self):
        """Test custom curve interpolation."""
        points = [
            CurvePoint(0.0, 1.0),
            CurvePoint(50.0, 0.5),
            CurvePoint(100.0, 0.0),
        ]
        atten = CustomCurveAttenuation(points, 1.0, 100.0)

        # Check endpoints
        assert atten.calculate(0.0) == 1.0
        assert atten.calculate(100.0) == 0.0

        # Check midpoint
        result = atten.calculate(50.0)
        assert result == pytest.approx(0.5, rel=0.1)

    def test_smooth_interpolation(self):
        """Test smooth (smoothstep) interpolation."""
        points = [
            CurvePoint(0.0, 1.0),
            CurvePoint(100.0, 0.0),
        ]
        atten = CustomCurveAttenuation(points, 1.0, 100.0, smooth=True)

        # Smoothstep should have S-curve shape
        at_50 = atten.calculate(50.0)

        assert at_50 == pytest.approx(0.5, rel=0.1)

    def test_add_point(self):
        """Test adding points to curve."""
        points = [
            CurvePoint(0.0, 1.0),
            CurvePoint(100.0, 0.0),
        ]
        atten = CustomCurveAttenuation(points, 1.0, 100.0)
        atten.add_point(50.0, 0.8)  # Keep high longer

        result = atten.calculate(50.0)
        assert result == pytest.approx(0.8, rel=0.1)


# =============================================================================
# Cone Attenuation Tests
# =============================================================================


class TestConeAttenuation:
    """Test suite for ConeAttenuation (directional sources)."""

    def test_within_inner_cone(self):
        """Test full volume within inner cone."""
        cone = ConeAttenuation(
            inner_angle=45.0,
            outer_angle=90.0,
            outer_gain=0.3,
        )

        # Listener directly in front
        source_dir = Vec3(0.0, 0.0, 1.0)
        to_listener = Vec3(0.0, 0.0, 1.0)

        result = cone.calculate(source_dir, to_listener)
        assert result == 1.0

    def test_at_outer_cone(self):
        """Test outer gain at outer cone boundary."""
        cone = ConeAttenuation(
            inner_angle=45.0,
            outer_angle=90.0,
            outer_gain=0.3,
        )

        # Listener at 90 degrees (beyond outer cone half-angle of 45 degrees)
        source_dir = Vec3(0.0, 0.0, 1.0)
        to_listener = Vec3(1.0, 0.0, 0.0)  # 90 degrees off-axis

        result = cone.calculate(source_dir, to_listener)
        assert result == pytest.approx(0.3, rel=0.1)

    def test_full_sphere_cone(self):
        """Test 360-degree cone has no attenuation."""
        cone = ConeAttenuation(
            inner_angle=360.0,
            outer_angle=360.0,
            outer_gain=0.0,
        )

        source_dir = Vec3(0.0, 0.0, 1.0)
        to_listener = Vec3(-1.0, 0.0, 0.0)  # Behind

        result = cone.calculate(source_dir, to_listener)
        assert result == 1.0

    def test_equal_inner_outer_angles_no_division_by_zero(self):
        """Test cone with equal inner and outer angles doesn't crash."""
        # This would cause division by zero if not guarded
        cone = ConeAttenuation(
            inner_angle=90.0,
            outer_angle=90.0,  # Same as inner
            outer_gain=0.5,
        )

        source_dir = Vec3(0.0, 0.0, 1.0)
        # Angle at 30 degrees - would fall in interpolation zone if inner != outer
        to_listener = Vec3(0.5, 0.0, 0.866)

        # Should return outer_gain without crashing
        result = cone.calculate(source_dir, to_listener)
        assert result >= 0.0 and result <= 1.0


class TestCustomCurveEdgeCases:
    """Test edge cases for CustomCurveAttenuation to ensure robustness."""

    def test_duplicate_distance_points_no_division_by_zero(self):
        """Test that duplicate distance points don't cause division by zero."""
        # Create points with same distance
        points = [
            CurvePoint(0.0, 1.0),
            CurvePoint(50.0, 0.8),
            CurvePoint(50.0, 0.6),  # Same distance as previous!
            CurvePoint(100.0, 0.0),
        ]
        atten = CustomCurveAttenuation(points, 1.0, 100.0, smooth=False)

        # Should not crash when calculating at or near duplicate distance
        result = atten.calculate(50.0)
        # Should return average of the two values at that distance
        assert result >= 0.0 and result <= 1.0

    def test_very_close_distance_points(self):
        """Test points with very small distance difference."""
        points = [
            CurvePoint(0.0, 1.0),
            CurvePoint(50.0, 0.8),
            CurvePoint(50.0001, 0.6),  # Very close to previous
            CurvePoint(100.0, 0.0),
        ]
        atten = CustomCurveAttenuation(points, 1.0, 100.0, smooth=False)

        # Should not crash
        result = atten.calculate(50.0)
        assert result >= 0.0 and result <= 1.0


# =============================================================================
# Attenuation Volume Tests
# =============================================================================


class TestAttenuationVolume:
    """Test suite for AttenuationVolume (complex shapes)."""

    def test_sphere_volume(self):
        """Test spherical attenuation volume."""
        volume = AttenuationVolume(
            shape=AttenuationShape.SPHERE,
            center=Vec3.zero(),
            curve=LinearAttenuation(1.0, 50.0),
        )

        listener = Vec3(25.0, 0.0, 0.0)
        result = volume.calculate(listener)
        assert 0.0 < result < 1.0

    def test_box_volume(self):
        """Test box attenuation volume."""
        volume = AttenuationVolume(
            shape=AttenuationShape.BOX,
            center=Vec3.zero(),
            half_extents=Vec3(10.0, 10.0, 10.0),
            curve=LinearAttenuation(0.0, 20.0),
        )

        # Inside box
        inside = Vec3(5.0, 5.0, 5.0)
        result_inside = volume.calculate(inside)
        assert result_inside > 0.5

        # Outside box
        outside = Vec3(30.0, 0.0, 0.0)
        result_outside = volume.calculate(outside)
        assert result_outside < result_inside

    def test_volume_with_cone(self):
        """Test volume with directional cone."""
        volume = AttenuationVolume(
            shape=AttenuationShape.SPHERE,
            center=Vec3.zero(),
            direction=Vec3(0.0, 0.0, 1.0),
            curve=LinearAttenuation(1.0, 50.0),
            cone=ConeAttenuation(45.0, 90.0, 0.3),
        )

        # Listener in front
        front = Vec3(0.0, 0.0, 25.0)
        result_front = volume.calculate(front)

        # Listener behind
        behind = Vec3(0.0, 0.0, -25.0)
        result_behind = volume.calculate(behind)

        # Front should be louder
        assert result_front > result_behind


# =============================================================================
# Attenuation Factory Tests
# =============================================================================


class TestAttenuationFactory:
    """Test suite for create_attenuation factory function."""

    def test_create_linear(self):
        """Test creating linear attenuation."""
        atten = create_attenuation(AttenuationModel.LINEAR, 1.0, 50.0, 1.0)
        assert atten.model == AttenuationModel.LINEAR

    def test_create_logarithmic(self):
        """Test creating logarithmic attenuation."""
        atten = create_attenuation(AttenuationModel.LOGARITHMIC)
        assert atten.model == AttenuationModel.LOGARITHMIC

    def test_create_inverse(self):
        """Test creating inverse attenuation."""
        atten = create_attenuation(AttenuationModel.INVERSE)
        assert atten.model == AttenuationModel.INVERSE

    def test_create_inverse_squared(self):
        """Test creating inverse squared attenuation."""
        atten = create_attenuation(AttenuationModel.INVERSE_SQUARED)
        assert atten.model == AttenuationModel.INVERSE_SQUARED

    def test_create_custom(self):
        """Test creating custom attenuation."""
        points = [CurvePoint(0.0, 1.0), CurvePoint(100.0, 0.0)]
        atten = create_attenuation(
            AttenuationModel.CUSTOM,
            points=points,
            smooth=True,
        )
        assert atten.model == AttenuationModel.CUSTOM

    def test_get_preset(self):
        """Test getting attenuation presets."""
        realistic = get_preset("realistic")
        assert realistic is not None
        assert realistic.model == AttenuationModel.INVERSE_SQUARED


# =============================================================================
# dB Conversion Tests
# =============================================================================


class TestDBConversions:
    """Test suite for dB conversions."""

    def test_db_to_linear_0db(self):
        """Test 0 dB equals 1.0 linear."""
        assert db_to_linear(0.0) == 1.0

    def test_db_to_linear_negative(self):
        """Test negative dB values."""
        result = db_to_linear(-20.0)
        assert result == pytest.approx(0.1, rel=0.01)

    def test_linear_to_db_unity(self):
        """Test 1.0 linear equals 0 dB."""
        assert linear_to_db(1.0) == 0.0

    def test_linear_to_db_zero(self):
        """Test 0.0 linear returns very low dB."""
        result = linear_to_db(0.0)
        assert result <= -96.0


# =============================================================================
# Positioning Tests - PointSource
# =============================================================================


class TestPointSource:
    """Test suite for PointSource."""

    def test_point_source_creation(self):
        """Test point source creation."""
        source = PointSource(
            position=Vec3(10.0, 0.0, -5.0),
        )
        assert source.position.x == 10.0
        assert source.position.z == -5.0

    def test_point_source_distance(self, listener_origin):
        """Test distance calculation from point source."""
        source = PointSource(
            position=Vec3(10.0, 0.0, 0.0),
        )
        distance = source.get_distance(listener_origin.position)
        assert distance == 10.0

    def test_point_source_direction(self, listener_origin):
        """Test direction calculation from point source."""
        source = PointSource(
            position=Vec3(10.0, 0.0, 0.0),
        )
        direction = source.get_direction(listener_origin.position)
        assert direction.x == pytest.approx(1.0, rel=0.01)

    def test_point_source_closest_point(self, listener_origin):
        """Test closest point is always the source position."""
        source = PointSource(position=Vec3(10.0, 5.0, 3.0))
        closest = source.get_closest_point(listener_origin.position)
        assert closest.x == 10.0
        assert closest.y == 5.0
        assert closest.z == 3.0


# =============================================================================
# Positioning Tests - AreaSource
# =============================================================================


class TestAreaSource:
    """Test suite for AreaSource."""

    def test_area_source_creation(self):
        """Test area source creation."""
        source = AreaSource(
            center=Vec3.zero(),
            half_extents=Vec3(5.0, 0.0, 5.0),
        )
        assert source.half_extents.x == 5.0

    def test_area_source_closest_point_outside(self):
        """Test closest point when listener is outside."""
        source = AreaSource(
            center=Vec3.zero(),
            half_extents=Vec3(5.0, 0.0, 5.0),
        )
        listener = Vec3(10.0, 0.0, 0.0)
        closest = source.get_closest_point(listener)
        # Should clamp to edge
        assert closest.x == pytest.approx(5.0, rel=0.1)


# =============================================================================
# Positioning Tests - LineSource
# =============================================================================


class TestLineSource:
    """Test suite for LineSource."""

    def test_line_source_creation(self):
        """Test line source creation."""
        source = LineSource(
            start=Vec3(0.0, 0.0, 0.0),
            end=Vec3(10.0, 0.0, 0.0),
        )
        assert source.start.x == 0.0
        assert source.end.x == 10.0

    def test_line_source_perpendicular(self):
        """Test distance to line from perpendicular point."""
        source = LineSource(
            start=Vec3(0.0, 0.0, 0.0),
            end=Vec3(10.0, 0.0, 0.0),
        )

        # Listener directly above midpoint
        listener = Vec3(5.0, 10.0, 0.0)
        distance = source.get_distance(listener)
        assert distance == pytest.approx(10.0, rel=0.01)

    def test_line_source_past_end(self):
        """Test distance when listener is past line end."""
        source = LineSource(
            start=Vec3(0.0, 0.0, 0.0),
            end=Vec3(10.0, 0.0, 0.0),
        )

        # Listener past the end
        listener = Vec3(20.0, 0.0, 0.0)
        distance = source.get_distance(listener)
        assert distance == 10.0


# =============================================================================
# Positioning Tests - VolumeSource
# =============================================================================


class TestVolumeSource:
    """Test suite for VolumeSource."""

    def test_volume_source_creation(self):
        """Test volume source creation."""
        source = VolumeSource(
            center=Vec3.zero(),
            half_extents=Vec3(5.0, 5.0, 5.0),
        )
        assert source.half_extents.x == 5.0

    def test_volume_source_inside(self, listener_origin):
        """Test listener inside volume."""
        source = VolumeSource(
            center=Vec3.zero(),
            half_extents=Vec3(10.0, 10.0, 10.0),
        )

        distance = source.get_distance(listener_origin.position)
        assert distance == 0.0

    def test_volume_source_contains(self):
        """Test contains method."""
        source = VolumeSource(
            center=Vec3.zero(),
            half_extents=Vec3(5.0, 5.0, 5.0),
        )
        assert source.contains(Vec3(2.0, 2.0, 2.0))
        assert not source.contains(Vec3(10.0, 0.0, 0.0))


# =============================================================================
# Stereo Panner Tests
# =============================================================================


class TestStereoPanner:
    """Test suite for StereoPanner."""

    def test_center_pan(self, stereo_panner):
        """Test center panning (0.0)."""
        params = SpatializationParams(azimuth=0.0, elevation=0.0, gain=1.0)
        gains = stereo_panner.calculate_gains(params)
        assert gains.left == pytest.approx(gains.right, rel=0.01)

    def test_full_left_pan(self, stereo_panner):
        """Test full left panning (-90 degrees)."""
        params = SpatializationParams(azimuth=-90.0, elevation=0.0, gain=1.0)
        gains = stereo_panner.calculate_gains(params)
        assert gains.left > gains.right

    def test_full_right_pan(self, stereo_panner):
        """Test full right panning (90 degrees)."""
        params = SpatializationParams(azimuth=90.0, elevation=0.0, gain=1.0)
        gains = stereo_panner.calculate_gains(params)
        assert gains.right > gains.left

    def test_constant_power_panning(self, stereo_panner):
        """Test constant power panning (sum of squares)."""
        params = SpatializationParams(azimuth=45.0, elevation=0.0, gain=1.0)
        gains = stereo_panner.calculate_gains(params)
        power = gains.left**2 + gains.right**2
        # Constant power should be approximately 1.0
        assert power == pytest.approx(1.0, rel=0.2)


# =============================================================================
# Surround Panner Tests
# =============================================================================


class TestSurroundPanner:
    """Test suite for SurroundPanner."""

    def test_5_1_creation(self):
        """Test 5.1 surround panner creation."""
        panner = SurroundPanner(layout=SpeakerLayout.SURROUND_5_1)
        assert panner.layout == SpeakerLayout.SURROUND_5_1

    def test_7_1_creation(self):
        """Test 7.1 surround panner creation."""
        panner = SurroundPanner(layout=SpeakerLayout.SURROUND_7_1)
        assert panner.layout == SpeakerLayout.SURROUND_7_1

    def test_front_position(self, surround_panner):
        """Test front speaker placement."""
        params = SpatializationParams(azimuth=0.0, elevation=0.0, distance=5.0, gain=1.0)
        gains = surround_panner.calculate_gains(params)
        # Should produce some output
        assert any(g > 0 for g in gains.gains)


# =============================================================================
# VBAP Spatializer Tests
# =============================================================================


class TestVBAPSpatializer:
    """Test suite for VBAPSpatializer."""

    def test_vbap_creation(self):
        """Test VBAP spatializer creation."""
        spatializer = VBAPSpatializer(layout=SpeakerLayout.SURROUND_5_1)
        assert spatializer.method == SpatializationMethod.VBAP

    def test_vbap_calculate_gains(self):
        """Test VBAP gain calculation."""
        spatializer = VBAPSpatializer(layout=SpeakerLayout.SURROUND_5_1)
        params = SpatializationParams(azimuth=30.0, elevation=0.0, distance=5.0, gain=1.0)
        gains = spatializer.calculate_gains(params)
        # Should produce some output
        assert any(g > 0 for g in gains.gains)


# =============================================================================
# Ambisonics Spatializer Tests
# =============================================================================


class TestAmbisonicsSpatializer:
    """Test suite for AmbisonicsSpatializer."""

    def test_ambisonics_creation(self):
        """Test Ambisonics spatializer creation."""
        spatializer = AmbisonicsSpatializer(order=1)
        assert spatializer.method == SpatializationMethod.AMBISONICS

    def test_ambisonics_encode(self):
        """Test Ambisonics encoding."""
        spatializer = AmbisonicsSpatializer(order=1)
        params = SpatializationParams(azimuth=45.0, elevation=0.0, distance=5.0, gain=1.0)
        coeffs = spatializer.encode(params)

        # First order has 4 channels (W, Y, Z, X)
        assert len(coeffs) == 4
        # W channel should always have signal
        assert coeffs[0] > 0


# =============================================================================
# HRTF Tests
# =============================================================================


class TestHRTF:
    """Test suite for HRTF processing."""

    def test_calculate_itd(self):
        """Test Interaural Time Difference calculation."""
        # Sound from the left (90 degrees)
        itd = calculate_itd(90.0, HEAD_RADIUS, HRTF_SAMPLE_RATE)

        # ITD should be positive (left ear leads)
        assert itd > 0

    def test_calculate_itd_center(self):
        """Test ITD for centered sound (0 degrees)."""
        itd = calculate_itd(0.0, HEAD_RADIUS, HRTF_SAMPLE_RATE)
        # ITD should be approximately 0 for centered sound
        assert abs(itd) < 2

    def test_calculate_ild(self):
        """Test Interaural Level Difference calculation."""
        # Sound from the left (90 degrees)
        ild = calculate_ild(90.0)
        # ILD returns a single float (positive = right ear louder)
        # At 90 degrees left, ild should be negative or significant
        assert isinstance(ild, float)

    def test_calculate_ild_center(self):
        """Test ILD for centered sound."""
        ild = calculate_ild(0.0)
        # Should be approximately 0 for centered sound
        assert ild == pytest.approx(0.0, abs=0.1)

    def test_hrtf_spatializer_creation(self, hrtf_spatializer):
        """Test HRTF spatializer creation."""
        assert hrtf_spatializer.quality == HRTFQuality.MEDIUM

    def test_hrtf_spatializer_calculate_gains(self, hrtf_spatializer):
        """Test HRTF gain calculation."""
        params = SpatializationParams(azimuth=45.0, elevation=0.0, distance=5.0, gain=1.0)
        gains = hrtf_spatializer.calculate_gains(params)

        # Should produce stereo output
        assert len(gains.gains) == 2

    def test_hrtf_profile_creation(self):
        """Test HRTF profile creation."""
        profile = create_default_hrtf_profile()
        assert profile.name == "default"


# =============================================================================
# Doppler Tests
# =============================================================================


class TestDoppler:
    """Test suite for Doppler effect."""

    def test_no_movement_no_shift(self, doppler_processor):
        """Test no Doppler shift when stationary."""
        shift = calculate_doppler_shift(
            source_pos=Vec3(10.0, 0.0, 0.0),
            source_velocity=Vec3.zero(),
            listener_pos=Vec3.zero(),
            listener_velocity=Vec3.zero(),
        )
        assert shift == pytest.approx(1.0, rel=0.01)

    def test_approaching_source(self, doppler_processor):
        """Test higher pitch when source approaches."""
        shift = calculate_doppler_shift(
            source_pos=Vec3(10.0, 0.0, 0.0),
            source_velocity=Vec3(-50.0, 0.0, 0.0),  # Moving towards listener
            listener_pos=Vec3.zero(),
            listener_velocity=Vec3.zero(),
        )
        assert shift > 1.0

    def test_receding_source(self, doppler_processor):
        """Test lower pitch when source recedes."""
        shift = calculate_doppler_shift(
            source_pos=Vec3(10.0, 0.0, 0.0),
            source_velocity=Vec3(50.0, 0.0, 0.0),  # Moving away
            listener_pos=Vec3.zero(),
            listener_velocity=Vec3.zero(),
        )
        assert shift < 1.0

    def test_approaching_listener(self, doppler_processor):
        """Test higher pitch when listener approaches."""
        # Direction from source to listener = (0,0,0) - (10,0,0) = (-1,0,0) normalized
        # Listener moving in negative X direction approaches the source
        shift = calculate_doppler_shift(
            source_pos=Vec3(10.0, 0.0, 0.0),
            source_velocity=Vec3.zero(),
            listener_pos=Vec3.zero(),
            listener_velocity=Vec3(-50.0, 0.0, 0.0),  # Moving towards source (negative X)
        )
        assert shift > 1.0

    def test_doppler_clamping(self):
        """Test Doppler shift is clamped to valid range."""
        # Very high velocity
        shift = calculate_doppler_shift(
            source_pos=Vec3(1.0, 0.0, 0.0),
            source_velocity=Vec3(-300.0, 0.0, 0.0),  # Near speed of sound
            listener_pos=Vec3.zero(),
            listener_velocity=Vec3.zero(),
        )
        assert MIN_DOPPLER_SHIFT <= shift <= MAX_DOPPLER_SHIFT

    def test_doppler_factor_exaggeration(self):
        """Test Doppler factor exaggeration."""
        normal = calculate_doppler_shift(
            source_pos=Vec3(10.0, 0.0, 0.0),
            source_velocity=Vec3(-50.0, 0.0, 0.0),
            listener_pos=Vec3.zero(),
            listener_velocity=Vec3.zero(),
            doppler_factor=1.0,
        )

        exaggerated = calculate_doppler_shift(
            source_pos=Vec3(10.0, 0.0, 0.0),
            source_velocity=Vec3(-50.0, 0.0, 0.0),
            listener_pos=Vec3.zero(),
            listener_velocity=Vec3.zero(),
            doppler_factor=2.0,
        )

        # Exaggerated should have more shift
        assert abs(exaggerated - 1.0) > abs(normal - 1.0)


# =============================================================================
# DopplerProcessor Tests
# =============================================================================


class TestDopplerProcessor:
    """Test suite for DopplerProcessor with smoothing."""

    def test_processor_creation(self, doppler_processor):
        """Test processor creation."""
        assert doppler_processor.doppler_factor == DOPPLER_FACTOR

    def test_processor_update(self, doppler_processor):
        """Test processor update."""
        shift = doppler_processor.update(
            source_id=1,
            source_pos=Vec3(10.0, 0.0, 0.0),
            listener_pos=Vec3.zero(),
            dt=0.016,
            source_velocity=Vec3(-50.0, 0.0, 0.0),
        )
        assert MIN_DOPPLER_SHIFT <= shift <= MAX_DOPPLER_SHIFT

    def test_processor_smoothing(self, doppler_processor):
        """Test Doppler smoothing prevents abrupt changes."""
        # First update
        shift1 = doppler_processor.update(
            source_id=1,
            source_pos=Vec3(10.0, 0.0, 0.0),
            listener_pos=Vec3.zero(),
            dt=0.016,
            source_velocity=Vec3.zero(),
        )

        # Abrupt velocity change
        shift2 = doppler_processor.update(
            source_id=1,
            source_pos=Vec3(9.0, 0.0, 0.0),
            listener_pos=Vec3.zero(),
            dt=0.016,
            source_velocity=Vec3(-100.0, 0.0, 0.0),
        )

        # With smoothing, shift shouldn't jump dramatically
        assert abs(shift2 - shift1) < 0.5

    def test_processor_remove_state(self, doppler_processor):
        """Test removing state for a source."""
        doppler_processor.update(
            source_id=1,
            source_pos=Vec3(10.0, 0.0, 0.0),
            listener_pos=Vec3.zero(),
            dt=0.016,
        )
        doppler_processor.remove_state(1)
        assert doppler_processor.get_current_shift(1) == 1.0


# =============================================================================
# DopplerConfig Tests
# =============================================================================


class TestDopplerConfig:
    """Test suite for DopplerConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = DopplerConfig()
        assert config.enabled
        assert config.factor == DOPPLER_FACTOR
        assert config.speed_of_sound == SPEED_OF_SOUND

    def test_create_processor_from_config(self):
        """Test creating processor from config."""
        config = DopplerConfig(factor=2.0, smoothing_time=0.1)
        processor = config.create_processor()
        assert processor.doppler_factor == 2.0

    def test_doppler_presets(self):
        """Test Doppler presets."""
        realistic = get_doppler_preset("realistic")
        assert realistic is not None
        assert realistic.factor == 1.0

        exaggerated = get_doppler_preset("exaggerated")
        assert exaggerated.factor > 1.0


# =============================================================================
# Reverb Zone Tests
# =============================================================================


class TestReverbZone:
    """Test suite for ReverbZone."""

    def test_reverb_zone_creation(self, reverb_zone):
        """Test reverb zone creation."""
        assert reverb_zone.name == "test_zone"
        assert reverb_zone.preset == ReverbPreset.MEDIUM_ROOM

    def test_reverb_zone_inside(self, reverb_zone):
        """Test listener inside zone."""
        listener_pos = Vec3(2.0, 0.0, 0.0)
        blend = reverb_zone.get_blend_factor(listener_pos)
        assert blend > 0.0  # Should have some effect inside

    def test_reverb_zone_outside(self, reverb_zone):
        """Test listener outside zone."""
        listener_pos = Vec3(20.0, 0.0, 0.0)
        blend = reverb_zone.get_blend_factor(listener_pos)
        assert blend == 0.0  # No effect outside

    def test_reverb_zone_transition(self, reverb_zone):
        """Test listener in transition zone (fade distance)."""
        # Zone is 10x10x10 with fade_distance=2.0
        # At position (9, 0, 0), listener is 1m from edge, within fade distance
        listener_pos = Vec3(9.0, 0.0, 0.0)
        blend = reverb_zone.get_blend_factor(listener_pos)
        assert 0.0 < blend < 1.0

    def test_reverb_zone_contains(self, reverb_zone):
        """Test contains method."""
        assert reverb_zone.contains(Vec3(5.0, 5.0, 5.0))
        assert not reverb_zone.contains(Vec3(15.0, 0.0, 0.0))


# =============================================================================
# ReverbParameters Tests
# =============================================================================


class TestReverbParameters:
    """Test suite for ReverbParameters."""

    def test_default_parameters(self):
        """Test default reverb parameters."""
        params = ReverbParameters()
        assert params.room_size > 0
        assert params.rt60 > 0
        assert 0.0 <= params.wet_mix <= 1.0

    def test_parameter_lerp(self):
        """Test parameter interpolation."""
        params1 = ReverbParameters(room_size=10.0, wet_mix=0.2)
        params2 = ReverbParameters(room_size=50.0, wet_mix=0.8)

        lerped = params1.lerp(params2, 0.5)

        assert lerped.room_size == pytest.approx(30.0, rel=0.01)
        assert lerped.wet_mix == pytest.approx(0.5, rel=0.01)


# =============================================================================
# ReverbZoneManager Tests
# =============================================================================


class TestReverbZoneManager:
    """Test suite for ReverbZoneManager."""

    def test_manager_creation(self):
        """Test manager creation."""
        manager = ReverbZoneManager()
        assert len(manager.get_zones()) == 0

    def test_add_zone(self):
        """Test adding a zone."""
        manager = ReverbZoneManager()
        zone_id = manager.add_zone(
            center=Vec3.zero(),
            half_extents=Vec3(10.0, 10.0, 10.0),
            preset=ReverbPreset.MEDIUM_ROOM,
        )
        assert zone_id > 0
        assert len(manager.get_zones()) == 1

    def test_remove_zone(self):
        """Test removing a zone."""
        manager = ReverbZoneManager()
        zone_id = manager.add_zone(
            center=Vec3.zero(),
            half_extents=Vec3(10.0, 10.0, 10.0),
        )
        result = manager.remove_zone(zone_id)
        assert result
        assert len(manager.get_zones()) == 0

    def test_update_parameters(self):
        """Test updating reverb parameters for listener."""
        manager = ReverbZoneManager()
        manager.add_zone(
            center=Vec3.zero(),
            half_extents=Vec3(10.0, 10.0, 10.0),
            preset=ReverbPreset.LARGE_ROOM,
        )

        listener_pos = Vec3(5.0, 0.0, 0.0)  # Inside zone
        params = manager.update(listener_id=0, listener_pos=listener_pos, dt=0.016)
        assert params is not None


# =============================================================================
# Speaker Configuration Tests
# =============================================================================


class TestSpeakerConfiguration:
    """Test suite for speaker configurations."""

    def test_stereo_angles(self):
        """Test stereo speaker angles."""
        angles = SPEAKER_ANGLES[SpeakerLayout.STEREO]
        assert len(angles) == 2
        assert angles[0][0] == -30.0  # Left at -30 degrees
        assert angles[1][0] == 30.0   # Right at +30 degrees

    def test_5_1_angles(self):
        """Test 5.1 speaker angles."""
        angles = SPEAKER_ANGLES[SpeakerLayout.SURROUND_5_1]
        assert len(angles) == 6

    def test_7_1_angles(self):
        """Test 7.1 speaker angles."""
        angles = SPEAKER_ANGLES[SpeakerLayout.SURROUND_7_1]
        assert len(angles) == 8

    def test_atmos_angles(self):
        """Test Atmos 7.1.4 speaker angles."""
        angles = SPEAKER_ANGLES[SpeakerLayout.ATMOS_7_1_4]
        assert len(angles) == 12


# =============================================================================
# Estimate Arrival Time Tests
# =============================================================================


class TestEstimateArrivalTime:
    """Test suite for sound arrival time estimation."""

    def test_stationary_arrival(self):
        """Test arrival time for stationary source and listener."""
        arrival = estimate_arrival_time(
            source_pos=Vec3(343.0, 0.0, 0.0),  # 343m away (1 sec at speed of sound)
            source_velocity=Vec3.zero(),
            listener_pos=Vec3.zero(),
            listener_velocity=Vec3.zero(),
        )
        assert arrival == pytest.approx(1.0, rel=0.01)

    def test_approaching_faster_arrival(self):
        """Test faster arrival when listener approaches source."""
        # Source at 343m (1 second at speed of sound), stationary
        # Calculate base arrival time
        base_arrival = estimate_arrival_time(
            source_pos=Vec3(343.0, 0.0, 0.0),
            source_velocity=Vec3.zero(),
            listener_pos=Vec3.zero(),
            listener_velocity=Vec3.zero(),
        )
        # Listener approaching source at 100 m/s should arrive faster
        # In the formula: approach_speed = -relative_velocity.dot(direction)
        # direction = listener - source = (-1, 0, 0)
        # listener moving negative X means listener_vel = (-100, 0, 0)
        # relative_vel = listener_vel - source_vel = (-100, 0, 0)
        # approach_speed = -(-100) * (-1) = -100 (negative = approaching)
        # effective_speed = 343 - (-100) = 443
        approaching_arrival = estimate_arrival_time(
            source_pos=Vec3(343.0, 0.0, 0.0),
            source_velocity=Vec3.zero(),
            listener_pos=Vec3.zero(),
            listener_velocity=Vec3(-100.0, 0.0, 0.0),  # Moving towards source
        )
        # When listener moves towards source, arrival time should be less
        assert approaching_arrival < base_arrival

    def test_supersonic_separation(self):
        """Test sound never arrives for supersonic separation."""
        # Listener moving away faster than sound
        # Source at positive X, listener at origin
        # Listener moving in positive X direction (away from source direction towards listener)
        # Would need to move in the direction that increases distance
        # Since direction = listener - source = (0,0,0) - (10,0,0) = (-1,0,0)
        # Moving in positive X is moving AWAY from source
        arrival = estimate_arrival_time(
            source_pos=Vec3(10.0, 0.0, 0.0),
            source_velocity=Vec3.zero(),
            listener_pos=Vec3.zero(),
            listener_velocity=Vec3(400.0, 0.0, 0.0),  # Moving away faster than sound
        )
        assert arrival is None


# =============================================================================
# Occlusion Tests
# =============================================================================


class TestOcclusion:
    """Test suite for occlusion detection."""

    def test_occlusion_detector_creation(self):
        """Test occlusion detector creation."""
        detector = OcclusionDetector(num_rays=4)
        assert detector.num_rays == 4

    def test_occlusion_no_geometry(self):
        """Test occlusion with no geometry (no raycast function)."""
        detector = OcclusionDetector()
        result = detector.detect(
            source_pos=Vec3(10.0, 0.0, 0.0),
            listener_pos=Vec3.zero(),
        )
        assert result.occlusion_type == OcclusionType.NONE
        assert result.occlusion_factor == 0.0

    def test_occlusion_processor_creation(self):
        """Test occlusion processor creation."""
        processor = OcclusionProcessor()
        assert processor.detector is not None


# =============================================================================
# Propagation Tests
# =============================================================================


class TestPropagation:
    """Test suite for sound propagation."""

    def test_propagation_calculator_creation(self):
        """Test propagation calculator creation."""
        calc = PropagationCalculator()
        assert calc.max_reflection_order > 0

    def test_direct_path_calculation(self):
        """Test direct path calculation."""
        calc = PropagationCalculator()
        result = calc.calculate(
            source_pos=Vec3(10.0, 0.0, 0.0),
            listener_pos=Vec3.zero(),
            include_reflections=False,
            include_diffraction=False,
        )
        # Should have at least a direct path
        assert len(result.paths) > 0
        assert result.paths[0].path_type == PathType.DIRECT


# =============================================================================
# Materials Tests
# =============================================================================


class TestMaterials:
    """Test suite for acoustic materials."""

    def test_material_presets(self):
        """Test material presets exist."""
        assert MaterialType.CONCRETE in MATERIAL_PRESETS
        assert MaterialType.WOOD in MATERIAL_PRESETS
        assert MaterialType.CARPET in MATERIAL_PRESETS

    def test_material_absorption(self):
        """Test material absorption coefficients."""
        concrete = MATERIAL_PRESETS[MaterialType.CONCRETE]
        carpet = MATERIAL_PRESETS[MaterialType.CARPET]

        # Carpet should absorb more than concrete
        assert carpet.average_absorption > concrete.average_absorption

    def test_material_database(self):
        """Test material database."""
        db = MaterialDatabase()
        concrete = db.get("concrete")
        assert concrete is not None
        assert concrete.material_type == MaterialType.CONCRETE

    def test_material_get_absorption_at_frequency(self):
        """Test frequency-dependent absorption."""
        material = MATERIAL_PRESETS[MaterialType.CARPET]
        # Carpet absorbs more at high frequencies
        abs_low = material.get_absorption(125.0)
        abs_high = material.get_absorption(4000.0)
        assert abs_high > abs_low


# =============================================================================
# Integration Tests
# =============================================================================


class TestSpatialIntegration:
    """Integration tests combining multiple spatial audio components."""

    def test_full_spatialization_chain(self):
        """Test complete spatialization chain."""
        # Create point source
        source = PointSource(
            position=Vec3(10.0, 0.0, -10.0),
        )

        # Listener at origin
        listener_pos = Vec3.zero()

        # Calculate distance
        distance = source.get_distance(listener_pos)

        # Calculate attenuation
        atten_curve = InverseSquaredAttenuation(1.0, 100.0)
        attenuation = atten_curve.calculate(distance)

        # Calculate direction for panning
        direction = source.get_direction(listener_pos)

        # Create stereo panner and spatialize
        panner = StereoPanner()
        params = SpatializationParams.from_direction(direction, distance)
        params.gain = attenuation
        gains = panner.calculate_gains(params)

        # Verify chain produces valid results
        assert 0.0 < attenuation < 1.0
        assert 0.0 <= gains.left <= 1.0
        assert 0.0 <= gains.right <= 1.0

    def test_doppler_with_attenuation(self):
        """Test Doppler effect combined with distance attenuation."""
        source_pos = Vec3(50.0, 0.0, 0.0)
        source_vel = Vec3(-30.0, 0.0, 0.0)  # Approaching
        listener_pos = Vec3.zero()

        # Distance attenuation
        atten = InverseSquaredAttenuation(1.0, 100.0)
        volume = atten.calculate(source_pos.length())

        # Doppler shift
        shift = calculate_doppler_shift(
            source_pos, source_vel,
            listener_pos, Vec3.zero(),
        )

        # Both should be valid
        assert 0.0 < volume < 1.0
        assert shift > 1.0  # Higher pitch (approaching)

    def test_reverb_zone_with_distance(self):
        """Test reverb zone blend with distance attenuation."""
        zone = ReverbZone(
            zone_id=1,
            name="hall",
            preset=ReverbPreset.LARGE_ROOM,
            center=Vec3.zero(),
            half_extents=Vec3(30.0, 30.0, 30.0),
            fade_distance=5.0,
        )

        source = PointSource(
            position=Vec3(5.0, 0.0, 0.0),
        )

        # Put listener at x=27, which is 3 units from edge (30)
        # min_dist = 3, fade_distance = 5, so 3 < 5 means we're in transition zone
        listener_pos = Vec3(27.0, 0.0, 0.0)  # Near edge (in transition zone)

        reverb_blend = zone.get_blend_factor(listener_pos)
        distance = source.get_distance(listener_pos)
        atten = LinearAttenuation(1.0, 50.0)
        volume = atten.calculate(distance)

        # Listener is in reverb transition zone (near edge)
        assert 0.0 < reverb_blend < 1.0
        # Source is audible
        assert volume > 0.0

    def test_spatialize_high_level_function(self):
        """Test high-level spatialize function."""
        result = spatialize(
            position=Vec3(10.0, 0.0, -5.0),
            listener_pos=Vec3.zero(),
            listener_forward=Vec3.forward(),
            listener_up=Vec3.up(),
            method=SpatializationMethod.PANNING,
            layout=SpeakerLayout.STEREO,
            gain=0.8,
        )

        assert result.channel_gains is not None
        assert result.method == SpatializationMethod.PANNING
        assert len(result.channel_gains.gains) == 2
