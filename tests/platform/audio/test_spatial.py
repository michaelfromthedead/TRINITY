"""
Comprehensive tests for spatial audio API.

Tests spatial source management, listener updates, reverb presets,
and 3D audio calculations.
"""

import pytest
import sys
import math

sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from engine.platform.audio import (
    SpatialAudioEngine,
    SpatialAudioAPI,
    SpatialSource,
    SpatialListener,
    ReverbPreset,
    Vec3,
)


@pytest.fixture
def engine():
    """Provide a fresh spatial audio engine for each test."""
    return SpatialAudioEngine()


class TestVec3:
    """Tests for Vec3 vector math."""

    def test_vec3_creation(self):
        """Verify Vec3 creation."""
        v = Vec3(1.0, 2.0, 3.0)
        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0

    def test_vec3_default(self):
        """Verify Vec3 default values."""
        v = Vec3()
        assert v.x == 0.0
        assert v.y == 0.0
        assert v.z == 0.0

    def test_vec3_length(self):
        """Verify vector length calculation."""
        v = Vec3(3.0, 4.0, 0.0)
        assert v.length() == 5.0

    def test_vec3_normalize(self):
        """Verify vector normalization."""
        v = Vec3(3.0, 4.0, 0.0)
        normalized = v.normalize()

        assert abs(normalized.length() - 1.0) < 0.0001

    def test_vec3_normalize_zero(self):
        """Verify normalizing zero vector."""
        v = Vec3(0.0, 0.0, 0.0)
        normalized = v.normalize()

        assert normalized.x == 0.0
        assert normalized.y == 0.0
        assert normalized.z == 0.0

    def test_vec3_dot_product(self):
        """Verify dot product calculation."""
        v1 = Vec3(1.0, 0.0, 0.0)
        v2 = Vec3(0.0, 1.0, 0.0)

        assert v1.dot(v2) == 0.0  # Perpendicular

        v3 = Vec3(1.0, 0.0, 0.0)
        assert v1.dot(v3) == 1.0  # Parallel

    def test_vec3_addition(self):
        """Verify vector addition."""
        v1 = Vec3(1.0, 2.0, 3.0)
        v2 = Vec3(4.0, 5.0, 6.0)
        result = v1 + v2

        assert result.x == 5.0
        assert result.y == 7.0
        assert result.z == 9.0

    def test_vec3_subtraction(self):
        """Verify vector subtraction."""
        v1 = Vec3(4.0, 5.0, 6.0)
        v2 = Vec3(1.0, 2.0, 3.0)
        result = v1 - v2

        assert result.x == 3.0
        assert result.y == 3.0
        assert result.z == 3.0


class TestSpatialSource:
    """Tests for SpatialSource configuration."""

    def test_spatial_source_creation(self):
        """Verify SpatialSource creation with defaults."""
        source = SpatialSource()

        assert isinstance(source.position, Vec3)
        assert isinstance(source.velocity, Vec3)
        assert isinstance(source.direction, Vec3)
        assert source.cone_inner_angle == 360.0
        assert source.cone_outer_angle == 360.0
        assert source.min_distance == 1.0
        assert source.max_distance == 100.0
        assert source.rolloff == 1.0

    def test_spatial_source_custom_values(self):
        """Verify SpatialSource creation with custom values."""
        source = SpatialSource(
            position=Vec3(10.0, 5.0, 0.0),
            velocity=Vec3(1.0, 0.0, 0.0),
            min_distance=2.0,
            max_distance=50.0,
            rolloff=2.0
        )

        assert source.position.x == 10.0
        assert source.position.y == 5.0
        assert source.min_distance == 2.0
        assert source.max_distance == 50.0
        assert source.rolloff == 2.0


class TestSpatialListener:
    """Tests for SpatialListener configuration."""

    def test_spatial_listener_creation(self):
        """Verify SpatialListener creation with defaults."""
        listener = SpatialListener()

        assert isinstance(listener.position, Vec3)
        assert isinstance(listener.forward, Vec3)
        assert isinstance(listener.up, Vec3)
        assert isinstance(listener.velocity, Vec3)

    def test_spatial_listener_custom_values(self):
        """Verify SpatialListener creation with custom values."""
        listener = SpatialListener(
            position=Vec3(0.0, 1.7, 0.0),  # Head height
            forward=Vec3(1.0, 0.0, 0.0),
            up=Vec3(0.0, 1.0, 0.0)
        )

        assert listener.position.y == 1.7
        assert listener.forward.x == 1.0


class TestSpatialAudioEngine:
    """Tests for SpatialAudioEngine functionality."""

    def test_engine_creation(self, engine):
        """Verify engine creation."""
        assert isinstance(engine, SpatialAudioEngine)

    def test_is_available(self):
        """Verify is_available returns consistent state."""
        is_avail = SpatialAudioEngine.is_available()
        assert isinstance(is_avail, bool)
        assert is_avail is True

        # Verify creating an engine works when available
        engine = SpatialAudioEngine()
        assert engine is not None
        handle = engine.create_source()
        assert handle >= 1

    def test_current_api(self):
        """Verify current_api class method."""
        api = SpatialAudioEngine.current_api()
        assert isinstance(api, SpatialAudioAPI)

    def test_create_source(self, engine):
        """Verify creating a spatial source."""
        handle = engine.create_source()

        assert isinstance(handle, int)
        assert handle >= 1

    def test_create_source_with_config(self, engine):
        """Verify creating source with custom configuration."""
        config = SpatialSource(position=Vec3(10.0, 0.0, 0.0))
        handle = engine.create_source(config)

        source = engine.get_source(handle)
        assert source is not None
        assert source.position.x == 10.0

    def test_create_multiple_sources(self, engine):
        """Verify creating multiple sources."""
        handle1 = engine.create_source()
        handle2 = engine.create_source()
        handle3 = engine.create_source()

        # Each should have unique handle
        assert handle1 != handle2
        assert handle2 != handle3
        assert handle1 != handle3

    def test_update_source(self, engine):
        """Verify updating a source."""
        handle = engine.create_source()

        new_config = SpatialSource(position=Vec3(20.0, 10.0, 5.0))
        engine.update_source(handle, new_config)

        source = engine.get_source(handle)
        assert source.position.x == 20.0
        assert source.position.y == 10.0
        assert source.position.z == 5.0

    def test_update_invalid_source_raises_error(self, engine):
        """Verify updating invalid source raises error."""
        with pytest.raises(KeyError):
            engine.update_source(999, SpatialSource())

    def test_remove_source(self, engine):
        """Verify removing a source."""
        handle = engine.create_source()

        engine.remove_source(handle)

        # Source should no longer exist
        assert engine.get_source(handle) is None

    def test_remove_invalid_source_raises_error(self, engine):
        """Verify removing invalid source raises error."""
        with pytest.raises(KeyError):
            engine.remove_source(999)

    def test_update_listener(self, engine):
        """Verify updating the listener."""
        listener = SpatialListener(
            position=Vec3(5.0, 1.7, 3.0),
            forward=Vec3(1.0, 0.0, 0.0)
        )

        engine.update_listener(listener)

        current = engine.get_listener()
        assert current.position.x == 5.0
        assert current.position.y == 1.7
        assert current.forward.x == 1.0

    def test_get_listener_default(self, engine):
        """Verify getting default listener."""
        listener = engine.get_listener()

        assert isinstance(listener, SpatialListener)
        assert listener.position.x == 0.0
        assert listener.position.y == 0.0

    def test_get_all_sources(self, engine):
        """Verify getting all sources."""
        handle1 = engine.create_source(SpatialSource(position=Vec3(1.0, 0.0, 0.0)))
        handle2 = engine.create_source(SpatialSource(position=Vec3(2.0, 0.0, 0.0)))

        sources = engine.get_all_sources()

        assert len(sources) == 2
        assert handle1 in sources
        assert handle2 in sources


class TestReverbPresets:
    """Tests for reverb preset functionality."""

    def test_set_reverb_preset(self, engine):
        """Verify setting reverb preset."""
        engine.set_reverb(ReverbPreset.LARGE_HALL)

        assert engine.get_reverb() == ReverbPreset.LARGE_HALL

    def test_reverb_preset_default(self, engine):
        """Verify default reverb preset."""
        assert engine.get_reverb() == ReverbPreset.NONE

    def test_all_reverb_presets(self, engine):
        """Verify all reverb presets can be set."""
        presets = [
            ReverbPreset.NONE,
            ReverbPreset.SMALL_ROOM,
            ReverbPreset.LARGE_HALL,
            ReverbPreset.OUTDOOR,
            ReverbPreset.CAVE,
            ReverbPreset.UNDERWATER,
        ]

        for preset in presets:
            engine.set_reverb(preset)
            assert engine.get_reverb() == preset


class TestAttenuationCalculation:
    """Tests for distance-based attenuation calculation."""

    def test_attenuation_at_min_distance(self, engine):
        """Verify attenuation is 1.0 within min distance."""
        config = SpatialSource(
            position=Vec3(0.5, 0.0, 0.0),
            min_distance=1.0,
            max_distance=10.0
        )
        handle = engine.create_source(config)

        # Listener at origin, source at 0.5 units (< min_distance)
        attenuation = engine.calculate_attenuation(handle)
        assert attenuation == 1.0

    def test_attenuation_at_max_distance(self, engine):
        """Verify attenuation is 0.0 beyond max distance."""
        config = SpatialSource(
            position=Vec3(15.0, 0.0, 0.0),
            min_distance=1.0,
            max_distance=10.0
        )
        handle = engine.create_source(config)

        # Source at 15 units (> max_distance)
        attenuation = engine.calculate_attenuation(handle)
        assert attenuation == 0.0

    def test_attenuation_linear_rolloff(self, engine):
        """Verify linear attenuation rolloff."""
        config = SpatialSource(
            position=Vec3(5.5, 0.0, 0.0),
            min_distance=1.0,
            max_distance=10.0,
            rolloff=1.0
        )
        handle = engine.create_source(config)

        # Distance = 5.5, range = 1.0 to 10.0
        # Normalized distance = (5.5 - 1.0) / (10.0 - 1.0) = 0.5
        # Attenuation = 1.0 - 0.5^1.0 = 0.5
        attenuation = engine.calculate_attenuation(handle)
        assert abs(attenuation - 0.5) < 0.01

    def test_attenuation_with_listener_position(self, engine):
        """Verify attenuation considers listener position."""
        # Place source at (10, 0, 0)
        config = SpatialSource(
            position=Vec3(10.0, 0.0, 0.0),
            min_distance=1.0,
            max_distance=10.0
        )
        handle = engine.create_source(config)

        # Move listener to (5, 0, 0)
        listener = SpatialListener(position=Vec3(5.0, 0.0, 0.0))
        engine.update_listener(listener)

        # Distance is now 5.0
        attenuation = engine.calculate_attenuation(handle)
        assert attenuation > 0.0
        assert attenuation < 1.0

    def test_attenuation_invalid_handle(self, engine):
        """Verify attenuation calculation with invalid handle raises error."""
        with pytest.raises(KeyError):
            engine.calculate_attenuation(999)


class TestPanningCalculation:
    """Tests for stereo panning calculation."""

    def test_pan_source_at_center(self, engine):
        """Verify centered source has equal panning."""
        config = SpatialSource(position=Vec3(0.0, 0.0, 0.0))
        handle = engine.create_source(config)

        left, right = engine.calculate_pan(handle)

        # Source at listener position should be centered
        assert abs(left - 0.5) < 0.1
        assert abs(right - 0.5) < 0.1

    def test_pan_source_to_right(self, engine):
        """Verify source to the right pans right."""
        # Default listener faces +Z, with +X to the right
        config = SpatialSource(position=Vec3(10.0, 0.0, 0.0))
        handle = engine.create_source(config)

        left, right = engine.calculate_pan(handle)

        # Source to the right should favor right channel significantly
        assert right > left
        # Verify the magnitude is meaningful (not just epsilon difference)
        assert right > 0.6

    def test_pan_source_to_left(self, engine):
        """Verify source to the left pans left."""
        # Default listener faces +Z, with -X to the left
        config = SpatialSource(position=Vec3(-10.0, 0.0, 0.0))
        handle = engine.create_source(config)

        left, right = engine.calculate_pan(handle)

        # Source to the left should favor left channel
        assert left > right

    def test_pan_source_in_front(self, engine):
        """Verify source in front is centered."""
        # Default listener faces +Z
        config = SpatialSource(position=Vec3(0.0, 0.0, 10.0))
        handle = engine.create_source(config)

        left, right = engine.calculate_pan(handle)

        # Source in front should be roughly centered
        assert abs(left - right) < 0.2

    def test_pan_with_rotated_listener(self, engine):
        """Verify panning considers listener orientation."""
        # Listener facing +X
        listener = SpatialListener(
            position=Vec3(0.0, 0.0, 0.0),
            forward=Vec3(1.0, 0.0, 0.0),
            up=Vec3(0.0, 1.0, 0.0)
        )
        engine.update_listener(listener)

        # Source at +Z (which is now to the right)
        config = SpatialSource(position=Vec3(0.0, 0.0, 10.0))
        handle = engine.create_source(config)

        left, right = engine.calculate_pan(handle)

        # Should favor right channel
        assert right > left

    def test_pan_invalid_handle(self, engine):
        """Verify panning calculation with invalid handle raises error."""
        with pytest.raises(KeyError):
            engine.calculate_pan(999)


class TestEngineClear:
    """Tests for engine clear functionality."""

    def test_clear_removes_all_sources(self, engine):
        """Verify clear removes all sources."""
        engine.create_source()
        engine.create_source()
        engine.create_source()

        assert len(engine.get_all_sources()) == 3

        engine.clear()

        assert len(engine.get_all_sources()) == 0

    def test_clear_resets_listener(self, engine):
        """Verify clear resets listener to default."""
        listener = SpatialListener(position=Vec3(10.0, 10.0, 10.0))
        engine.update_listener(listener)

        engine.clear()

        default_listener = engine.get_listener()
        assert default_listener.position.x == 0.0
        assert default_listener.position.y == 0.0
        assert default_listener.position.z == 0.0

    def test_clear_resets_reverb(self, engine):
        """Verify clear resets reverb to NONE."""
        engine.set_reverb(ReverbPreset.LARGE_HALL)

        engine.clear()

        assert engine.get_reverb() == ReverbPreset.NONE

    def test_clear_resets_handle_counter(self, engine):
        """Verify clear resets handle counter."""
        handle1 = engine.create_source()
        engine.clear()
        handle2 = engine.create_source()

        # After clear, handles should start from 1 again
        assert handle2 == 1


class TestIntegration:
    """Integration tests for complete spatial audio scenarios."""

    def test_multiple_sources_with_listener_movement(self, engine):
        """Test multiple sources as listener moves."""
        # Create sources at different positions
        source1 = engine.create_source(SpatialSource(position=Vec3(10.0, 0.0, 0.0)))
        source2 = engine.create_source(SpatialSource(position=Vec3(-10.0, 0.0, 0.0)))
        source3 = engine.create_source(SpatialSource(position=Vec3(0.0, 0.0, 10.0)))

        # Listener at origin
        attenuation1 = engine.calculate_attenuation(source1)

        # Move listener closer to source1
        engine.update_listener(SpatialListener(position=Vec3(5.0, 0.0, 0.0)))
        attenuation2 = engine.calculate_attenuation(source1)

        # Attenuation should increase (less distance)
        assert attenuation2 > attenuation1

    def test_full_spatial_audio_scene(self, engine):
        """Test a complete spatial audio scene."""
        # Set up environment
        engine.set_reverb(ReverbPreset.LARGE_HALL)

        # Create listener (player)
        listener = SpatialListener(
            position=Vec3(0.0, 1.7, 0.0),  # Head height
            forward=Vec3(0.0, 0.0, -1.0),  # Standard forward direction (-Z)
            up=Vec3(0.0, 1.0, 0.0)
        )
        engine.update_listener(listener)

        # Create sources (sound emitters)
        footsteps = engine.create_source(SpatialSource(
            position=Vec3(5.0, 0.0, 0.0),
            min_distance=1.0,
            max_distance=20.0
        ))

        music = engine.create_source(SpatialSource(
            position=Vec3(0.0, 0.0, 10.0),
            min_distance=2.0,
            max_distance=50.0
        ))

        # Verify everything works together
        assert engine.get_reverb() == ReverbPreset.LARGE_HALL
        assert len(engine.get_all_sources()) == 2

        # Calculate audio properties
        footsteps_attenuation = engine.calculate_attenuation(footsteps)
        footsteps_pan = engine.calculate_pan(footsteps)

        music_attenuation = engine.calculate_attenuation(music)
        music_pan = engine.calculate_pan(music)

        # Footsteps to the right should pan right
        assert footsteps_pan[1] > footsteps_pan[0]

        # Music in front should be centered
        assert abs(music_pan[0] - music_pan[1]) < 0.2
