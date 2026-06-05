"""
Whitebox tests for the audio subsystem.

Tests audio device enumeration, stream management, spatial audio,
backend registry, and thread safety.
"""

import pytest
import sys
import numpy as np
import threading
import time
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, '/home/user/dev/USER/PROJECTS_VOID/TRINITY')

from engine.platform.audio import (
    AudioDevice,
    AudioDeviceInfo,
    AudioDeviceType,
    AudioFormat,
    AudioStream,
    NullAudioBackend,
    set_backend,
    SpatialAudioEngine,
    SpatialAudioAPI,
    SpatialSource,
    SpatialListener,
    ReverbPreset,
    Vec3,
    register_backend,
    get_backend,
    get_default_backend,
    list_backends,
    create_backend,
)
from engine.platform.audio.backends import BackendRegistry
from engine.platform.constants import (
    DEFAULT_AUDIO_SAMPLE_RATE,
    FALLBACK_AUDIO_SAMPLE_RATE,
    DEFAULT_AUDIO_CHANNELS,
    DEFAULT_AUDIO_BUFFER_SIZE,
    SPATIAL_DEFAULT_MIN_DISTANCE,
    SPATIAL_DEFAULT_MAX_DISTANCE,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def null_backend():
    """Provide fresh null backend for each test."""
    backend = NullAudioBackend()
    set_backend(backend)
    yield backend
    set_backend(NullAudioBackend())


@pytest.fixture
def spatial_engine():
    """Provide fresh spatial audio engine for each test."""
    return SpatialAudioEngine()


# ============================================================================
# Vec3 Tests
# ============================================================================

class TestVec3:
    """Tests for Vec3 3D vector class."""

    def test_default_construction(self):
        """Test default Vec3 is origin."""
        v = Vec3()
        assert v.x == 0.0
        assert v.y == 0.0
        assert v.z == 0.0

    def test_construction_with_values(self):
        """Test Vec3 construction with values."""
        v = Vec3(1.0, 2.0, 3.0)
        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0

    def test_length_unit_vectors(self):
        """Test length of unit vectors."""
        assert Vec3(1, 0, 0).length() == 1.0
        assert Vec3(0, 1, 0).length() == 1.0
        assert Vec3(0, 0, 1).length() == 1.0

    def test_length_zero_vector(self):
        """Test length of zero vector."""
        assert Vec3().length() == 0.0

    def test_length_arbitrary(self):
        """Test length of arbitrary vector."""
        v = Vec3(3, 4, 0)
        assert v.length() == 5.0

    def test_normalize_unit_vector(self):
        """Test normalizing unit vector."""
        v = Vec3(1, 0, 0).normalize()
        assert abs(v.x - 1.0) < 1e-10
        assert abs(v.y) < 1e-10
        assert abs(v.z) < 1e-10

    def test_normalize_arbitrary(self):
        """Test normalizing arbitrary vector."""
        v = Vec3(3, 4, 0).normalize()
        assert abs(v.length() - 1.0) < 1e-10
        assert abs(v.x - 0.6) < 1e-10
        assert abs(v.y - 0.8) < 1e-10

    def test_normalize_zero_vector(self):
        """Test normalizing zero vector returns zero."""
        v = Vec3().normalize()
        assert v.x == 0.0
        assert v.y == 0.0
        assert v.z == 0.0

    def test_dot_product_perpendicular(self):
        """Test dot product of perpendicular vectors is zero."""
        v1 = Vec3(1, 0, 0)
        v2 = Vec3(0, 1, 0)
        assert v1.dot(v2) == 0.0

    def test_dot_product_parallel(self):
        """Test dot product of parallel vectors."""
        v1 = Vec3(1, 0, 0)
        v2 = Vec3(2, 0, 0)
        assert v1.dot(v2) == 2.0

    def test_dot_product_opposite(self):
        """Test dot product of opposite vectors."""
        v1 = Vec3(1, 0, 0)
        v2 = Vec3(-1, 0, 0)
        assert v1.dot(v2) == -1.0

    def test_subtraction(self):
        """Test vector subtraction."""
        v1 = Vec3(5, 3, 1)
        v2 = Vec3(2, 1, 1)
        result = v1 - v2
        assert result.x == 3.0
        assert result.y == 2.0
        assert result.z == 0.0

    def test_addition(self):
        """Test vector addition."""
        v1 = Vec3(1, 2, 3)
        v2 = Vec3(4, 5, 6)
        result = v1 + v2
        assert result.x == 5.0
        assert result.y == 7.0
        assert result.z == 9.0


# ============================================================================
# AudioDeviceInfo Tests
# ============================================================================

class TestAudioDeviceInfo:
    """Tests for AudioDeviceInfo dataclass."""

    def test_device_info_creation(self):
        """Test creating device info."""
        info = AudioDeviceInfo(
            name="Test Device",
            device_type=AudioDeviceType.PLAYBACK,
            channel_count=2,
            sample_rate=48000,
            format=AudioFormat.F32,
            device_id="test-0"
        )
        assert info.name == "Test Device"
        assert info.device_type == AudioDeviceType.PLAYBACK
        assert info.channel_count == 2
        assert info.sample_rate == 48000
        assert info.format == AudioFormat.F32
        assert info.device_id == "test-0"


# ============================================================================
# NullAudioBackend Tests
# ============================================================================

class TestNullAudioBackend:
    """Tests for null audio backend implementation."""

    def test_enumerate_playback_returns_devices(self, null_backend):
        """Verify playback enumeration returns devices."""
        devices = null_backend.enumerate_devices(AudioDeviceType.PLAYBACK)
        assert len(devices) >= 1

    def test_enumerate_capture_returns_devices(self, null_backend):
        """Verify capture enumeration returns devices."""
        devices = null_backend.enumerate_devices(AudioDeviceType.CAPTURE)
        assert len(devices) >= 1

    def test_enumerate_device_types_correct(self, null_backend):
        """Verify enumerated devices have correct type."""
        playback = null_backend.enumerate_devices(AudioDeviceType.PLAYBACK)
        capture = null_backend.enumerate_devices(AudioDeviceType.CAPTURE)

        assert all(d.device_type == AudioDeviceType.PLAYBACK for d in playback)
        assert all(d.device_type == AudioDeviceType.CAPTURE for d in capture)

    def test_default_device_returns_first(self, null_backend):
        """Verify default device is first enumerated."""
        devices = null_backend.enumerate_devices(AudioDeviceType.PLAYBACK)
        default = null_backend.get_default_device(AudioDeviceType.PLAYBACK)
        assert default.device_id == devices[0].device_id

    def test_set_device_count(self, null_backend):
        """Verify setting device count."""
        null_backend.set_device_count(AudioDeviceType.PLAYBACK, 5)
        devices = null_backend.enumerate_devices(AudioDeviceType.PLAYBACK)
        assert len(devices) == 5

    def test_set_device_count_minimum(self, null_backend):
        """Verify device count validation."""
        with pytest.raises(ValueError):
            null_backend.set_device_count(AudioDeviceType.PLAYBACK, 0)

    def test_devices_have_unique_ids(self, null_backend):
        """Verify enumerated devices have unique IDs."""
        null_backend.set_device_count(AudioDeviceType.PLAYBACK, 10)
        devices = null_backend.enumerate_devices(AudioDeviceType.PLAYBACK)
        ids = [d.device_id for d in devices]
        assert len(ids) == len(set(ids))

    def test_devices_have_varied_properties(self, null_backend):
        """Verify enumerated devices have varied properties."""
        null_backend.set_device_count(AudioDeviceType.PLAYBACK, 3)
        devices = null_backend.enumerate_devices(AudioDeviceType.PLAYBACK)

        # Should have variation in channel count
        channel_counts = set(d.channel_count for d in devices)
        assert len(channel_counts) > 1

    def test_active_stream_count_initial(self, null_backend):
        """Verify initial stream count is zero."""
        assert null_backend.active_stream_count() == 0

    def test_open_stream_tracks_count(self, null_backend):
        """Verify opening stream increments count."""
        device = null_backend.get_default_device(AudioDeviceType.PLAYBACK)

        def callback(input_buffer, frame_count):
            return None

        stream = null_backend.open_stream(device, callback, 1024)
        assert null_backend.active_stream_count() == 1

    def test_close_stream_decrements_count(self, null_backend):
        """Verify closing stream decrements count."""
        device = null_backend.get_default_device(AudioDeviceType.PLAYBACK)

        def callback(input_buffer, frame_count):
            return None

        stream = null_backend.open_stream(device, callback, 1024)
        stream.start()
        stream.stop()
        assert null_backend.active_stream_count() == 0


# ============================================================================
# AudioStream Tests
# ============================================================================

class TestAudioStream:
    """Tests for AudioStream class."""

    def test_stream_initial_state(self, null_backend):
        """Verify stream initial state."""
        device = AudioDevice.default_device(AudioDeviceType.PLAYBACK)

        def callback(input_buffer, frame_count):
            return None

        stream = AudioDevice.open(device, callback)
        assert not stream.is_running()
        assert stream.buffer_size == DEFAULT_AUDIO_BUFFER_SIZE
        assert stream.device_info == device

    def test_stream_start(self, null_backend):
        """Verify stream start."""
        device = AudioDevice.default_device(AudioDeviceType.PLAYBACK)

        def callback(input_buffer, frame_count):
            return np.zeros((frame_count, 2), dtype=np.float32)

        stream = AudioDevice.open(device, callback)
        stream.start()
        assert stream.is_running()
        stream.stop()

    def test_stream_stop(self, null_backend):
        """Verify stream stop."""
        device = AudioDevice.default_device(AudioDeviceType.PLAYBACK)

        def callback(input_buffer, frame_count):
            return np.zeros((frame_count, 2), dtype=np.float32)

        stream = AudioDevice.open(device, callback)
        stream.start()
        stream.stop()
        assert not stream.is_running()

    def test_stream_latency_frames(self, null_backend):
        """Verify latency frames property."""
        device = AudioDevice.default_device(AudioDeviceType.PLAYBACK)

        def callback(input_buffer, frame_count):
            return None

        stream = AudioDevice.open(device, callback, buffer_size=512)
        # Null backend returns buffer size as latency
        assert stream.latency_frames == 512

    def test_stream_context_manager(self, null_backend):
        """Verify stream context manager."""
        device = AudioDevice.default_device(AudioDeviceType.PLAYBACK)

        def callback(input_buffer, frame_count):
            return np.zeros((frame_count, 2), dtype=np.float32)

        stream = AudioDevice.open(device, callback)

        with stream:
            assert stream.is_running()

        assert not stream.is_running()

    def test_stream_callback_invoked(self, null_backend):
        """Verify callback is invoked."""
        device = AudioDevice.default_device(AudioDeviceType.PLAYBACK)
        call_count = [0]

        def callback(input_buffer, frame_count):
            call_count[0] += 1
            return np.zeros((frame_count, 2), dtype=np.float32)

        stream = AudioDevice.open(device, callback, buffer_size=1024)
        stream.start()
        time.sleep(0.15)
        stream.stop()

        assert call_count[0] > 0

    def test_stream_callback_frame_count(self, null_backend):
        """Verify callback receives correct frame count."""
        device = AudioDevice.default_device(AudioDeviceType.PLAYBACK)
        received_counts = []

        def callback(input_buffer, frame_count):
            received_counts.append(frame_count)
            return np.zeros((frame_count, 2), dtype=np.float32)

        stream = AudioDevice.open(device, callback, buffer_size=256)
        stream.start()
        time.sleep(0.1)
        stream.stop()

        assert all(c == 256 for c in received_counts)

    def test_capture_stream_receives_input(self, null_backend):
        """Verify capture stream receives input buffer."""
        device = AudioDevice.default_device(AudioDeviceType.CAPTURE)
        received_inputs = []

        def callback(input_buffer, frame_count):
            received_inputs.append(input_buffer is not None)
            return None

        stream = AudioDevice.open(device, callback)
        stream.start()
        time.sleep(0.1)
        stream.stop()

        assert any(received_inputs)

    def test_playback_stream_no_input(self, null_backend):
        """Verify playback stream receives no input."""
        device = AudioDevice.default_device(AudioDeviceType.PLAYBACK)
        received_inputs = []

        def callback(input_buffer, frame_count):
            received_inputs.append(input_buffer)
            return np.zeros((frame_count, 2), dtype=np.float32)

        stream = AudioDevice.open(device, callback)
        stream.start()
        time.sleep(0.1)
        stream.stop()

        assert all(inp is None for inp in received_inputs)


# ============================================================================
# SpatialAudioEngine Tests
# ============================================================================

class TestSpatialAudioEngine:
    """Tests for SpatialAudioEngine class."""

    def test_initial_state(self, spatial_engine):
        """Verify initial engine state."""
        assert len(spatial_engine.get_all_sources()) == 0
        assert spatial_engine.get_reverb() == ReverbPreset.NONE

    def test_is_available(self):
        """Verify availability check."""
        assert SpatialAudioEngine.is_available()

    def test_current_api(self):
        """Verify current API returns value."""
        api = SpatialAudioEngine.current_api()
        assert isinstance(api, SpatialAudioAPI)

    def test_create_source_default(self, spatial_engine):
        """Verify creating source with defaults."""
        handle = spatial_engine.create_source()
        assert handle == 1
        source = spatial_engine.get_source(handle)
        assert source is not None

    def test_create_source_with_config(self, spatial_engine):
        """Verify creating source with config."""
        config = SpatialSource(position=Vec3(10, 0, 0))
        handle = spatial_engine.create_source(config)
        source = spatial_engine.get_source(handle)
        assert source.position.x == 10

    def test_create_multiple_sources(self, spatial_engine):
        """Verify creating multiple sources."""
        handles = [spatial_engine.create_source() for _ in range(5)]
        assert len(set(handles)) == 5  # All unique
        assert len(spatial_engine.get_all_sources()) == 5

    def test_update_source(self, spatial_engine):
        """Verify updating source."""
        handle = spatial_engine.create_source()
        new_config = SpatialSource(position=Vec3(20, 10, 5))
        spatial_engine.update_source(handle, new_config)
        source = spatial_engine.get_source(handle)
        assert source.position.x == 20
        assert source.position.y == 10
        assert source.position.z == 5

    def test_update_invalid_source(self, spatial_engine):
        """Verify updating invalid source raises."""
        with pytest.raises(KeyError):
            spatial_engine.update_source(999, SpatialSource())

    def test_remove_source(self, spatial_engine):
        """Verify removing source."""
        handle = spatial_engine.create_source()
        spatial_engine.remove_source(handle)
        assert spatial_engine.get_source(handle) is None

    def test_remove_invalid_source(self, spatial_engine):
        """Verify removing invalid source raises."""
        with pytest.raises(KeyError):
            spatial_engine.remove_source(999)

    def test_update_listener(self, spatial_engine):
        """Verify updating listener."""
        listener = SpatialListener(position=Vec3(5, 1.7, 0))
        spatial_engine.update_listener(listener)
        assert spatial_engine.get_listener().position.x == 5

    def test_set_reverb(self, spatial_engine):
        """Verify setting reverb."""
        spatial_engine.set_reverb(ReverbPreset.LARGE_HALL)
        assert spatial_engine.get_reverb() == ReverbPreset.LARGE_HALL

    def test_clear(self, spatial_engine):
        """Verify clearing engine."""
        spatial_engine.create_source()
        spatial_engine.create_source()
        spatial_engine.set_reverb(ReverbPreset.CAVE)
        spatial_engine.clear()
        assert len(spatial_engine.get_all_sources()) == 0
        assert spatial_engine.get_reverb() == ReverbPreset.NONE


# ============================================================================
# Spatial Audio Attenuation Tests
# ============================================================================

class TestSpatialAttenuation:
    """Tests for spatial audio attenuation calculations."""

    def test_attenuation_at_min_distance(self, spatial_engine):
        """Verify no attenuation at min distance."""
        source = SpatialSource(
            position=Vec3(1, 0, 0),
            min_distance=1.0,
            max_distance=100.0
        )
        handle = spatial_engine.create_source(source)
        attenuation = spatial_engine.calculate_attenuation(handle)
        assert attenuation == 1.0

    def test_attenuation_within_min_distance(self, spatial_engine):
        """Verify full volume within min distance."""
        source = SpatialSource(
            position=Vec3(0.5, 0, 0),
            min_distance=1.0,
            max_distance=100.0
        )
        handle = spatial_engine.create_source(source)
        attenuation = spatial_engine.calculate_attenuation(handle)
        assert attenuation == 1.0

    def test_attenuation_at_max_distance(self, spatial_engine):
        """Verify full attenuation at max distance."""
        source = SpatialSource(
            position=Vec3(100, 0, 0),
            min_distance=1.0,
            max_distance=100.0
        )
        handle = spatial_engine.create_source(source)
        attenuation = spatial_engine.calculate_attenuation(handle)
        assert attenuation == 0.0

    def test_attenuation_beyond_max_distance(self, spatial_engine):
        """Verify full attenuation beyond max distance."""
        source = SpatialSource(
            position=Vec3(200, 0, 0),
            min_distance=1.0,
            max_distance=100.0
        )
        handle = spatial_engine.create_source(source)
        attenuation = spatial_engine.calculate_attenuation(handle)
        assert attenuation == 0.0

    def test_attenuation_midpoint(self, spatial_engine):
        """Verify partial attenuation at midpoint."""
        source = SpatialSource(
            position=Vec3(50, 0, 0),
            min_distance=0.0,
            max_distance=100.0,
            rolloff=1.0  # Linear
        )
        handle = spatial_engine.create_source(source)
        attenuation = spatial_engine.calculate_attenuation(handle)
        assert 0.0 < attenuation < 1.0

    def test_attenuation_invalid_handle(self, spatial_engine):
        """Verify attenuation raises for invalid handle."""
        with pytest.raises(KeyError):
            spatial_engine.calculate_attenuation(999)


# ============================================================================
# Spatial Audio Panning Tests
# ============================================================================

class TestSpatialPanning:
    """Tests for spatial audio panning calculations."""

    def test_pan_centered(self, spatial_engine):
        """Verify centered source gives equal pan."""
        source = SpatialSource(position=Vec3(0, 0, -10))
        handle = spatial_engine.create_source(source)
        left, right = spatial_engine.calculate_pan(handle)
        assert abs(left - right) < 0.01

    def test_pan_left(self, spatial_engine):
        """Verify left source gives left bias."""
        source = SpatialSource(position=Vec3(-10, 0, 0))
        handle = spatial_engine.create_source(source)
        left, right = spatial_engine.calculate_pan(handle)
        assert left > right

    def test_pan_right(self, spatial_engine):
        """Verify right source gives right bias."""
        source = SpatialSource(position=Vec3(10, 0, 0))
        handle = spatial_engine.create_source(source)
        left, right = spatial_engine.calculate_pan(handle)
        assert right > left

    def test_pan_at_listener(self, spatial_engine):
        """Verify source at listener is centered."""
        source = SpatialSource(position=Vec3(0, 0, 0))
        handle = spatial_engine.create_source(source)
        left, right = spatial_engine.calculate_pan(handle)
        assert abs(left - 0.5) < 0.01
        assert abs(right - 0.5) < 0.01

    def test_pan_sum_to_one(self, spatial_engine):
        """Verify pan values sum to approximately one."""
        source = SpatialSource(position=Vec3(5, 3, -2))
        handle = spatial_engine.create_source(source)
        left, right = spatial_engine.calculate_pan(handle)
        assert abs(left + right - 1.0) < 0.01

    def test_pan_invalid_handle(self, spatial_engine):
        """Verify pan raises for invalid handle."""
        with pytest.raises(KeyError):
            spatial_engine.calculate_pan(999)


# ============================================================================
# SpatialSource Tests
# ============================================================================

class TestSpatialSource:
    """Tests for SpatialSource dataclass."""

    def test_default_source(self):
        """Verify default source configuration."""
        source = SpatialSource()
        assert source.position.x == 0.0
        assert source.min_distance == SPATIAL_DEFAULT_MIN_DISTANCE
        assert source.max_distance == SPATIAL_DEFAULT_MAX_DISTANCE

    def test_custom_source(self):
        """Verify custom source configuration."""
        source = SpatialSource(
            position=Vec3(10, 5, 0),
            velocity=Vec3(1, 0, 0),
            min_distance=2.0,
            max_distance=50.0,
            rolloff=2.0
        )
        assert source.position.x == 10
        assert source.velocity.x == 1
        assert source.min_distance == 2.0
        assert source.max_distance == 50.0
        assert source.rolloff == 2.0


# ============================================================================
# SpatialListener Tests
# ============================================================================

class TestSpatialListener:
    """Tests for SpatialListener dataclass."""

    def test_default_listener(self):
        """Verify default listener configuration."""
        listener = SpatialListener()
        assert listener.position.x == 0.0
        assert listener.forward.z == -1.0
        assert listener.up.y == 1.0

    def test_custom_listener(self):
        """Verify custom listener configuration."""
        listener = SpatialListener(
            position=Vec3(5, 1.7, 0),
            forward=Vec3(1, 0, 0),
            up=Vec3(0, 1, 0)
        )
        assert listener.position.x == 5
        assert listener.forward.x == 1.0


# ============================================================================
# Audio Backend Registry Tests
# ============================================================================

class TestAudioBackendRegistry:
    """Tests for audio backend registry."""

    def test_null_backend_registered(self):
        """Verify null backend is registered."""
        backends = list_backends()
        assert "null" in backends

    def test_get_null_backend(self):
        """Verify getting null backend."""
        backend_class = get_backend("null")
        assert backend_class is not None

    def test_get_default_backend(self):
        """Verify getting default backend."""
        backend_class = get_default_backend()
        assert backend_class is not None

    def test_create_backend(self):
        """Verify creating backend instance."""
        backend = create_backend("null")
        assert backend is not None
        # Check by class name due to potential import path differences
        assert type(backend).__name__ == "NullAudioBackend"

    def test_create_default_backend(self):
        """Verify creating default backend."""
        backend = create_backend()
        assert backend is not None


# ============================================================================
# Thread Safety Tests
# ============================================================================

class TestAudioThreadSafety:
    """Tests for audio subsystem thread safety."""

    def test_concurrent_stream_creation(self, null_backend):
        """Verify concurrent stream creation is safe."""
        device = AudioDevice.default_device(AudioDeviceType.PLAYBACK)
        streams = []
        lock = threading.Lock()

        def create_stream():
            def callback(input_buffer, frame_count):
                return np.zeros((frame_count, 2), dtype=np.float32)

            stream = AudioDevice.open(device, callback)
            with lock:
                streams.append(stream)

        threads = [threading.Thread(target=create_stream) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(streams) == 10

    def test_concurrent_spatial_operations(self, spatial_engine):
        """Verify concurrent spatial operations are safe."""
        errors = []

        def create_sources():
            try:
                for _ in range(20):
                    spatial_engine.create_source(
                        SpatialSource(position=Vec3(1, 0, 0))
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create_sources) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestAudioEdgeCases:
    """Tests for audio edge cases."""

    def test_very_small_buffer_size(self, null_backend):
        """Verify handling very small buffer sizes."""
        device = AudioDevice.default_device(AudioDeviceType.PLAYBACK)

        def callback(input_buffer, frame_count):
            return np.zeros((frame_count, 2), dtype=np.float32)

        stream = AudioDevice.open(device, callback, buffer_size=16)
        assert stream.buffer_size == 16

    def test_large_buffer_size(self, null_backend):
        """Verify handling large buffer sizes."""
        device = AudioDevice.default_device(AudioDeviceType.PLAYBACK)

        def callback(input_buffer, frame_count):
            return np.zeros((frame_count, 2), dtype=np.float32)

        stream = AudioDevice.open(device, callback, buffer_size=8192)
        assert stream.buffer_size == 8192

    def test_callback_exception_handling(self, null_backend):
        """Verify callback exception handling."""
        device = AudioDevice.default_device(AudioDeviceType.PLAYBACK)
        error_count = [0]

        def callback(input_buffer, frame_count):
            error_count[0] += 1
            if error_count[0] == 1:
                raise ValueError("Test error")
            return np.zeros((frame_count, 2), dtype=np.float32)

        stream = AudioDevice.open(device, callback)
        stream.start()
        time.sleep(0.1)
        stream.stop()

        # Should have handled the exception gracefully

    def test_many_spatial_sources(self, spatial_engine):
        """Verify handling many spatial sources."""
        handles = []
        for i in range(100):
            handle = spatial_engine.create_source(
                SpatialSource(position=Vec3(i, 0, 0))
            )
            handles.append(handle)

        assert len(spatial_engine.get_all_sources()) == 100

        # Remove all
        for handle in handles:
            spatial_engine.remove_source(handle)

        assert len(spatial_engine.get_all_sources()) == 0


# ============================================================================
# Audio Format Tests
# ============================================================================

class TestAudioFormat:
    """Tests for AudioFormat enum."""

    def test_format_values(self):
        """Verify format enum values."""
        assert AudioFormat.F32.value == "float32"
        assert AudioFormat.I16.value == "int16"
        assert AudioFormat.I24.value == "int24"
        assert AudioFormat.I32.value == "int32"


# ============================================================================
# Reverb Preset Tests
# ============================================================================

class TestReverbPreset:
    """Tests for ReverbPreset enum."""

    def test_reverb_values(self):
        """Verify reverb preset values."""
        assert ReverbPreset.NONE.value == "none"
        assert ReverbPreset.SMALL_ROOM.value == "small_room"
        assert ReverbPreset.LARGE_HALL.value == "large_hall"
        assert ReverbPreset.OUTDOOR.value == "outdoor"
        assert ReverbPreset.CAVE.value == "cave"
        assert ReverbPreset.UNDERWATER.value == "underwater"

    def test_all_presets_settable(self, spatial_engine):
        """Verify all presets can be set."""
        for preset in ReverbPreset:
            spatial_engine.set_reverb(preset)
            assert spatial_engine.get_reverb() == preset


# ============================================================================
# AudioDeviceType Tests
# ============================================================================

class TestAudioDeviceType:
    """Tests for AudioDeviceType enum."""

    def test_device_type_values(self):
        """Verify device type values."""
        assert AudioDeviceType.PLAYBACK.value == "playback"
        assert AudioDeviceType.CAPTURE.value == "capture"


# ============================================================================
# SpatialAudioAPI Tests
# ============================================================================

class TestSpatialAudioAPI:
    """Tests for SpatialAudioAPI enum."""

    def test_api_values(self):
        """Verify spatial audio API values."""
        assert SpatialAudioAPI.NONE.value == "none"
        assert SpatialAudioAPI.WINDOWS_SONIC.value == "windows_sonic"
        assert SpatialAudioAPI.TEMPEST_3D.value == "tempest_3d"
        assert SpatialAudioAPI.APPLE_SPATIAL.value == "apple_spatial"
