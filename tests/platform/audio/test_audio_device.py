"""
Comprehensive tests for audio device API.

Tests device enumeration, stream management, callbacks, and backend functionality.
"""

import pytest
import sys
import numpy as np
import time
import threading

sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from engine.platform.audio import (
    AudioDevice,
    AudioDeviceInfo,
    AudioDeviceType,
    AudioFormat,
    AudioStream,
    AudioCallback,
    NullAudioBackend,
    set_backend,
)


@pytest.fixture
def null_backend():
    """Provide a fresh null backend for each test."""
    backend = NullAudioBackend()
    set_backend(backend)
    yield backend
    # Reset to default null backend
    set_backend(NullAudioBackend())


class TestDeviceEnumeration:
    """Tests for device enumeration functionality."""

    def test_enumerate_playback_devices(self, null_backend):
        """Verify playback device enumeration returns devices."""
        devices = AudioDevice.enumerate(AudioDeviceType.PLAYBACK)
        assert len(devices) > 0
        assert all(d.device_type == AudioDeviceType.PLAYBACK for d in devices)

    def test_enumerate_capture_devices(self, null_backend):
        """Verify capture device enumeration returns devices."""
        devices = AudioDevice.enumerate(AudioDeviceType.CAPTURE)
        assert len(devices) > 0
        assert all(d.device_type == AudioDeviceType.CAPTURE for d in devices)

    def test_device_info_properties(self, null_backend):
        """Verify device info contains expected properties."""
        devices = AudioDevice.enumerate(AudioDeviceType.PLAYBACK)
        device = devices[0]

        assert isinstance(device.name, str)
        assert len(device.name) > 0
        assert isinstance(device.device_type, AudioDeviceType)
        assert device.channel_count > 0
        assert device.sample_rate > 0
        assert isinstance(device.format, AudioFormat)
        assert isinstance(device.device_id, str)

    def test_enumerate_returns_multiple_devices(self, null_backend):
        """Verify enumeration can return multiple devices."""
        null_backend.set_device_count(AudioDeviceType.PLAYBACK, 3)
        devices = AudioDevice.enumerate(AudioDeviceType.PLAYBACK)
        assert len(devices) == 3

        # Each device should have unique ID
        device_ids = [d.device_id for d in devices]
        assert len(device_ids) == len(set(device_ids))


class TestDefaultDevice:
    """Tests for default device functionality."""

    def test_get_default_playback_device(self, null_backend):
        """Verify getting default playback device."""
        device = AudioDevice.default_device(AudioDeviceType.PLAYBACK)
        assert device is not None
        assert device.device_type == AudioDeviceType.PLAYBACK

    def test_get_default_capture_device(self, null_backend):
        """Verify getting default capture device."""
        device = AudioDevice.default_device(AudioDeviceType.CAPTURE)
        assert device is not None
        assert device.device_type == AudioDeviceType.CAPTURE

    def test_default_device_is_in_enumeration(self, null_backend):
        """Verify default device appears in enumeration with expected properties."""
        default = AudioDevice.default_device(AudioDeviceType.PLAYBACK)
        devices = AudioDevice.enumerate(AudioDeviceType.PLAYBACK)

        # Default should be in the enumeration
        device_ids = [d.device_id for d in devices]
        assert default.device_id in device_ids

        # Verify default device has expected properties
        assert len(default.name) > 0
        assert default.sample_rate > 0
        assert default.channel_count > 0
        assert isinstance(default.format, AudioFormat)


class TestStreamManagement:
    """Tests for audio stream management."""

    def test_open_stream(self, null_backend):
        """Verify opening an audio stream."""
        device = AudioDevice.default_device(AudioDeviceType.PLAYBACK)

        def callback(input_buffer, frame_count):
            return np.zeros((frame_count, device.channel_count), dtype=np.float32)

        stream = AudioDevice.open(device, callback)
        assert isinstance(stream, AudioStream)
        assert stream.device_info == device
        assert stream.buffer_size == 1024  # Default

    def test_open_stream_with_custom_buffer_size(self, null_backend):
        """Verify opening stream with custom buffer size."""
        device = AudioDevice.default_device(AudioDeviceType.PLAYBACK)

        def callback(input_buffer, frame_count):
            return np.zeros((frame_count, device.channel_count), dtype=np.float32)

        stream = AudioDevice.open(device, callback, buffer_size=512)
        assert stream.buffer_size == 512

    def test_stream_start_stop(self, null_backend):
        """Verify starting and stopping a stream."""
        device = AudioDevice.default_device(AudioDeviceType.PLAYBACK)

        def callback(input_buffer, frame_count):
            return np.zeros((frame_count, device.channel_count), dtype=np.float32)

        stream = AudioDevice.open(device, callback)

        assert not stream.is_running()

        stream.start()
        assert stream.is_running()

        time.sleep(0.1)  # Let it run briefly

        stream.stop()
        assert not stream.is_running()

    def test_stream_start_idempotent(self, null_backend):
        """Verify calling start multiple times is safe."""
        device = AudioDevice.default_device(AudioDeviceType.PLAYBACK)

        def callback(input_buffer, frame_count):
            return np.zeros((frame_count, device.channel_count), dtype=np.float32)

        stream = AudioDevice.open(device, callback)

        stream.start()
        stream.start()  # Should not raise
        assert stream.is_running()

        stream.stop()

    def test_stream_stop_idempotent(self, null_backend):
        """Verify calling stop multiple times is safe."""
        device = AudioDevice.default_device(AudioDeviceType.PLAYBACK)

        def callback(input_buffer, frame_count):
            return np.zeros((frame_count, device.channel_count), dtype=np.float32)

        stream = AudioDevice.open(device, callback)

        stream.start()
        stream.stop()
        stream.stop()  # Should not raise
        assert not stream.is_running()

    def test_stream_context_manager(self, null_backend):
        """Verify stream works as context manager."""
        device = AudioDevice.default_device(AudioDeviceType.PLAYBACK)

        def callback(input_buffer, frame_count):
            return np.zeros((frame_count, device.channel_count), dtype=np.float32)

        stream = AudioDevice.open(device, callback)

        with stream:
            assert stream.is_running()
            time.sleep(0.05)

        assert not stream.is_running()


class TestAudioCallback:
    """Tests for audio callback functionality."""

    def test_callback_invoked(self, null_backend):
        """Verify callback is invoked when stream runs."""
        device = AudioDevice.default_device(AudioDeviceType.PLAYBACK)
        callback_count = [0]

        def callback(input_buffer, frame_count):
            callback_count[0] += 1
            return np.zeros((frame_count, device.channel_count), dtype=np.float32)

        stream = AudioDevice.open(device, callback, buffer_size=1024)
        stream.start()

        # Wait for several callbacks
        time.sleep(0.2)
        stream.stop()

        # Should have been called multiple times
        assert callback_count[0] > 0

    def test_callback_receives_correct_frame_count(self, null_backend):
        """Verify callback receives correct frame count."""
        device = AudioDevice.default_device(AudioDeviceType.PLAYBACK)
        buffer_size = 512
        received_counts = []

        def callback(input_buffer, frame_count):
            received_counts.append(frame_count)
            return np.zeros((frame_count, device.channel_count), dtype=np.float32)

        stream = AudioDevice.open(device, callback, buffer_size=buffer_size)
        stream.start()
        time.sleep(0.1)
        stream.stop()

        # All callbacks should receive the buffer size
        assert all(count == buffer_size for count in received_counts)

    def test_callback_receives_input_for_capture(self, null_backend):
        """Verify capture stream callback receives input buffer."""
        device = AudioDevice.default_device(AudioDeviceType.CAPTURE)
        received_inputs = []

        def callback(input_buffer, frame_count):
            if input_buffer is not None:
                received_inputs.append(input_buffer.shape)
            return None

        stream = AudioDevice.open(device, callback)
        stream.start()
        time.sleep(0.1)
        stream.stop()

        # Should have received input buffers
        assert len(received_inputs) > 0
        # Each should be (buffer_size, channels)
        for shape in received_inputs:
            assert shape == (stream.buffer_size, device.channel_count)

    def test_callback_no_input_for_playback(self, null_backend):
        """Verify playback stream callback receives None for input."""
        device = AudioDevice.default_device(AudioDeviceType.PLAYBACK)
        received_inputs = []

        def callback(input_buffer, frame_count):
            received_inputs.append(input_buffer)
            return np.zeros((frame_count, device.channel_count), dtype=np.float32)

        stream = AudioDevice.open(device, callback)
        stream.start()
        time.sleep(0.1)
        stream.stop()

        # All inputs should be None for playback
        assert all(inp is None for inp in received_inputs)

    def test_callback_generates_output(self, null_backend):
        """Verify callback can generate audio output."""
        device = AudioDevice.default_device(AudioDeviceType.PLAYBACK)

        def callback(input_buffer, frame_count):
            # Generate simple sine wave
            t = np.linspace(0, 1, frame_count)
            sine = np.sin(2 * np.pi * 440 * t)
            return np.column_stack([sine] * device.channel_count).astype(np.float32)

        stream = AudioDevice.open(device, callback)
        stream.start()
        time.sleep(0.1)
        stream.stop()

        # Should run without errors


class TestStreamProperties:
    """Tests for stream property access."""

    def test_latency_frames(self, null_backend):
        """Verify latency_frames property."""
        device = AudioDevice.default_device(AudioDeviceType.PLAYBACK)

        def callback(input_buffer, frame_count):
            return np.zeros((frame_count, device.channel_count), dtype=np.float32)

        stream = AudioDevice.open(device, callback, buffer_size=512)
        latency = stream.latency_frames

        assert isinstance(latency, int)
        assert latency > 0

    def test_buffer_size_property(self, null_backend):
        """Verify buffer_size property is accessible."""
        device = AudioDevice.default_device(AudioDeviceType.PLAYBACK)

        def callback(input_buffer, frame_count):
            return np.zeros((frame_count, device.channel_count), dtype=np.float32)

        buffer_size = 768
        stream = AudioDevice.open(device, callback, buffer_size=buffer_size)

        assert stream.buffer_size == buffer_size

    def test_device_info_property(self, null_backend):
        """Verify device_info property is accessible."""
        device = AudioDevice.default_device(AudioDeviceType.PLAYBACK)

        def callback(input_buffer, frame_count):
            return np.zeros((frame_count, device.channel_count), dtype=np.float32)

        stream = AudioDevice.open(device, callback)

        assert stream.device_info == device
        assert stream.device_info.device_type == AudioDeviceType.PLAYBACK


class TestNullBackend:
    """Tests for null backend implementation."""

    def test_null_backend_active_stream_count(self, null_backend):
        """Verify backend tracks active streams."""
        device = AudioDevice.default_device(AudioDeviceType.PLAYBACK)

        def callback(input_buffer, frame_count):
            return np.zeros((frame_count, device.channel_count), dtype=np.float32)

        assert null_backend.active_stream_count() == 0

        stream1 = AudioDevice.open(device, callback)
        stream2 = AudioDevice.open(device, callback)

        stream1.start()
        stream2.start()

        # Both streams should be tracked
        assert null_backend.active_stream_count() == 2

        stream1.stop()
        assert null_backend.active_stream_count() == 1

        stream2.stop()
        assert null_backend.active_stream_count() == 0

    def test_null_backend_set_device_count(self, null_backend):
        """Verify backend can configure device count."""
        null_backend.set_device_count(AudioDeviceType.PLAYBACK, 5)
        devices = AudioDevice.enumerate(AudioDeviceType.PLAYBACK)
        assert len(devices) == 5

    def test_null_backend_device_count_validation(self, null_backend):
        """Verify device count validation."""
        with pytest.raises(ValueError):
            null_backend.set_device_count(AudioDeviceType.PLAYBACK, 0)


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_stream_creation(self, null_backend):
        """Verify concurrent stream creation is safe."""
        device = AudioDevice.default_device(AudioDeviceType.PLAYBACK)
        streams = []

        def create_stream():
            def callback(input_buffer, frame_count):
                return np.zeros((frame_count, device.channel_count), dtype=np.float32)

            stream = AudioDevice.open(device, callback)
            streams.append(stream)

        threads = [threading.Thread(target=create_stream) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(streams) == 5

        # Clean up
        for stream in streams:
            stream.stop()

    def test_concurrent_stream_start_stop(self, null_backend):
        """Verify concurrent start/stop is safe."""
        device = AudioDevice.default_device(AudioDeviceType.PLAYBACK)

        def callback(input_buffer, frame_count):
            return np.zeros((frame_count, device.channel_count), dtype=np.float32)

        stream = AudioDevice.open(device, callback)

        def start_stop_loop():
            for _ in range(10):
                stream.start()
                time.sleep(0.01)
                stream.stop()
                time.sleep(0.01)

        threads = [threading.Thread(target=start_stop_loop) for _ in range(3)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should complete without errors


class TestBackendSwitching:
    """Tests for backend switching."""

    def test_set_backend(self):
        """Verify backend can be switched."""
        backend1 = NullAudioBackend()
        backend2 = NullAudioBackend()

        set_backend(backend1)
        device1 = AudioDevice.default_device(AudioDeviceType.PLAYBACK)

        set_backend(backend2)
        device2 = AudioDevice.default_device(AudioDeviceType.PLAYBACK)

        # Both should work
        assert device1 is not None
        assert device2 is not None

        # Reset
        set_backend(NullAudioBackend())
