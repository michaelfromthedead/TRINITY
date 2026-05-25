"""
Audio device abstraction for cross-platform audio I/O.

Provides device enumeration, stream management, and audio callbacks
with a working null backend for testing without audio hardware.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional
import logging
import numpy as np
import threading
import time

from ..constants import (
    DEFAULT_AUDIO_SAMPLE_RATE, FALLBACK_AUDIO_SAMPLE_RATE,
    DEFAULT_AUDIO_CHANNELS, DEFAULT_AUDIO_BUFFER_SIZE,
    AUDIO_THREAD_SLEEP_FACTOR
)

logger = logging.getLogger(__name__)


class AudioDeviceType(Enum):
    """Type of audio device."""
    PLAYBACK = "playback"
    CAPTURE = "capture"


class AudioFormat(Enum):
    """Audio sample format."""
    F32 = "float32"  # 32-bit float
    I16 = "int16"    # 16-bit signed integer
    I24 = "int24"    # 24-bit signed integer
    I32 = "int32"    # 32-bit signed integer


@dataclass
class AudioDeviceInfo:
    """Information about an audio device.

    Attributes:
        name: Human-readable device name
        device_type: PLAYBACK or CAPTURE
        channel_count: Number of audio channels
        sample_rate: Sample rate in Hz
        format: Audio sample format
        device_id: Unique device identifier
    """
    name: str
    device_type: AudioDeviceType
    channel_count: int
    sample_rate: int
    format: AudioFormat
    device_id: str


# Audio callback type: receives input buffer (or None) and frame count,
# returns output buffer (or None)
AudioCallback = Callable[[Optional[np.ndarray], int], Optional[np.ndarray]]


class AudioStream:
    """Active audio stream with playback/capture capability.

    Manages stream lifecycle (start, stop) and provides stream status.
    """

    def __init__(
        self,
        device_info: AudioDeviceInfo,
        callback: AudioCallback,
        buffer_size: int,
        backend: 'AudioBackend'
    ):
        """Initialize audio stream.

        Args:
            device_info: Device configuration
            callback: Audio processing callback
            buffer_size: Buffer size in frames
            backend: Backend implementation
        """
        self.device_info = device_info
        self.callback = callback
        self.buffer_size = buffer_size
        self._backend = backend
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start the audio stream."""
        if self._running:
            return

        self._running = True
        self._stop_event.clear()
        self._backend._start_stream(self)

    def stop(self) -> None:
        """Stop the audio stream."""
        if not self._running:
            return

        self._running = False
        self._stop_event.set()
        self._backend._stop_stream(self)

    def is_running(self) -> bool:
        """Check if stream is currently running.

        Returns:
            True if stream is active
        """
        return self._running

    @property
    def latency_frames(self) -> int:
        """Get stream latency in frames.

        Returns:
            Latency in frames
        """
        return self._backend._get_latency_frames(self)

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


class AudioBackend(ABC):
    """Abstract base class for audio backend implementations."""

    @abstractmethod
    def enumerate_devices(
        self,
        device_type: AudioDeviceType
    ) -> list[AudioDeviceInfo]:
        """Enumerate available audio devices.

        Args:
            device_type: Type of devices to enumerate

        Returns:
            List of available devices
        """
        pass

    @abstractmethod
    def get_default_device(
        self,
        device_type: AudioDeviceType
    ) -> AudioDeviceInfo:
        """Get default audio device.

        Args:
            device_type: Type of device

        Returns:
            Default device info
        """
        pass

    @abstractmethod
    def open_stream(
        self,
        device_info: AudioDeviceInfo,
        callback: AudioCallback,
        buffer_size: int
    ) -> AudioStream:
        """Open an audio stream.

        Args:
            device_info: Device to open
            callback: Audio processing callback
            buffer_size: Buffer size in frames

        Returns:
            Opened audio stream
        """
        pass

    @abstractmethod
    def _start_stream(self, stream: AudioStream) -> None:
        """Backend-specific stream start implementation."""
        pass

    @abstractmethod
    def _stop_stream(self, stream: AudioStream) -> None:
        """Backend-specific stream stop implementation."""
        pass

    @abstractmethod
    def _get_latency_frames(self, stream: AudioStream) -> int:
        """Get stream latency in frames."""
        pass


class NullAudioBackend(AudioBackend):
    """Null audio backend for testing without real audio hardware.

    This backend provides sensible defaults and produces/consumes silence.
    Useful for testing, CI/CD, and headless environments.
    """

    def __init__(self):
        """Initialize null audio backend."""
        self._streams: set[AudioStream] = set()
        self._device_count = {
            AudioDeviceType.PLAYBACK: 2,
            AudioDeviceType.CAPTURE: 1
        }

    def enumerate_devices(
        self,
        device_type: AudioDeviceType
    ) -> list[AudioDeviceInfo]:
        """Enumerate null audio devices.

        Returns multiple dummy devices for testing enumeration.
        """
        devices = []
        count = self._device_count[device_type]

        for i in range(count):
            # Vary properties for more realistic testing
            channel_count = DEFAULT_AUDIO_CHANNELS if i == 0 else (1 if i == 1 else 8)
            sample_rate = DEFAULT_AUDIO_SAMPLE_RATE if i == 0 else FALLBACK_AUDIO_SAMPLE_RATE

            devices.append(
                AudioDeviceInfo(
                    name=f"Null {device_type.value.capitalize()} Device {i}",
                    device_type=device_type,
                    channel_count=channel_count,
                    sample_rate=sample_rate,
                    format=AudioFormat.F32,
                    device_id=f"null-{device_type.value}-{i}"
                )
            )

        return devices

    def get_default_device(
        self,
        device_type: AudioDeviceType
    ) -> AudioDeviceInfo:
        """Get default null audio device."""
        return self.enumerate_devices(device_type)[0]

    def open_stream(
        self,
        device_info: AudioDeviceInfo,
        callback: AudioCallback,
        buffer_size: int = 1024
    ) -> AudioStream:
        """Open a null audio stream.

        Stream will call callback at appropriate intervals but won't
        produce actual audio output.
        """
        stream = AudioStream(device_info, callback, buffer_size, self)
        self._streams.add(stream)
        return stream

    def _start_stream(self, stream: AudioStream) -> None:
        """Start null stream processing thread."""
        def stream_thread():
            """Stream processing thread."""
            sample_rate = stream.device_info.sample_rate
            buffer_size = stream.buffer_size
            channels = stream.device_info.channel_count

            # Calculate sleep time to simulate real-time audio
            sleep_time = buffer_size / sample_rate * AUDIO_THREAD_SLEEP_FACTOR

            while not stream._stop_event.is_set():
                # Create silent input buffer for capture devices
                input_buffer = None
                if stream.device_info.device_type == AudioDeviceType.CAPTURE:
                    input_buffer = np.zeros(
                        (buffer_size, channels),
                        dtype=np.float32
                    )

                # Call user callback
                try:
                    output_buffer = stream.callback(input_buffer, buffer_size)
                except Exception as e:
                    logger.error(f"Audio callback error: {e}", exc_info=True)
                    break

                # Discard output (null backend)
                if output_buffer is not None:
                    pass  # Would be sent to audio hardware in real backend

                # Sleep to simulate real-time processing
                time.sleep(sleep_time)

        stream._thread = threading.Thread(target=stream_thread, daemon=True)
        stream._thread.start()

    def _stop_stream(self, stream: AudioStream) -> None:
        """Stop null stream processing thread."""
        if stream._thread and stream._thread.is_alive():
            stream._stop_event.set()
            stream._thread.join(timeout=1.0)

        if stream in self._streams:
            self._streams.remove(stream)

    def _get_latency_frames(self, stream: AudioStream) -> int:
        """Get null stream latency (returns buffer size)."""
        return stream.buffer_size

    def active_stream_count(self) -> int:
        """Get number of active streams.

        Returns:
            Number of active streams
        """
        return len(self._streams)

    def set_device_count(
        self,
        device_type: AudioDeviceType,
        count: int
    ) -> None:
        """Set number of devices to enumerate (for testing).

        Args:
            device_type: Device type
            count: Number of devices to enumerate

        Raises:
            ValueError: If count is less than 1
        """
        if count < 1:
            raise ValueError("Device count must be at least 1")
        self._device_count[device_type] = count


# Global backend registry
_backend: Optional[AudioBackend] = None


def _get_backend() -> AudioBackend:
    """Get current audio backend (creates null backend if none set)."""
    global _backend
    if _backend is None:
        _backend = NullAudioBackend()
    return _backend


def set_backend(backend: AudioBackend) -> None:
    """Set the global audio backend.

    Args:
        backend: Backend implementation to use
    """
    global _backend
    _backend = backend


class AudioDevice:
    """Main API for audio device access.

    Provides static methods for device enumeration and stream creation.
    Uses the configured backend (defaults to NullAudioBackend).
    """

    @classmethod
    def enumerate(
        cls,
        device_type: AudioDeviceType
    ) -> list[AudioDeviceInfo]:
        """Enumerate available audio devices.

        Args:
            device_type: Type of devices to enumerate (PLAYBACK or CAPTURE)

        Returns:
            List of available audio devices

        Example:
            >>> playback_devices = AudioDevice.enumerate(AudioDeviceType.PLAYBACK)
            >>> for device in playback_devices:
            ...     print(f"{device.name}: {device.channel_count}ch @ {device.sample_rate}Hz")
        """
        return _get_backend().enumerate_devices(device_type)

    @classmethod
    def default_device(
        cls,
        device_type: AudioDeviceType
    ) -> AudioDeviceInfo:
        """Get the default audio device.

        Args:
            device_type: Type of device (PLAYBACK or CAPTURE)

        Returns:
            Default device info

        Example:
            >>> default_output = AudioDevice.default_device(AudioDeviceType.PLAYBACK)
            >>> print(f"Default output: {default_output.name}")
        """
        return _get_backend().get_default_device(device_type)

    @classmethod
    def open(
        cls,
        device_info: AudioDeviceInfo,
        callback: AudioCallback,
        buffer_size: int = DEFAULT_AUDIO_BUFFER_SIZE
    ) -> AudioStream:
        """Open an audio stream on the specified device.

        Args:
            device_info: Device to open
            callback: Audio processing callback function
            buffer_size: Buffer size in frames (default: 1024)

        Returns:
            Opened audio stream

        Example:
            >>> def audio_callback(input_buffer, frame_count):
            ...     # Generate sine wave
            ...     t = np.linspace(0, 1, frame_count)
            ...     return np.sin(2 * np.pi * 440 * t).reshape(-1, 1)
            >>>
            >>> device = AudioDevice.default_device(AudioDeviceType.PLAYBACK)
            >>> stream = AudioDevice.open(device, audio_callback)
            >>> stream.start()
        """
        return _get_backend().open_stream(device_info, callback, buffer_size)
