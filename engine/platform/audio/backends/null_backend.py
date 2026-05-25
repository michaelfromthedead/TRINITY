"""
Null audio backend implementation.

Provides a working audio backend that runs without real hardware.
Perfect for testing, CI/CD, and headless environments.
"""

from ..audio_device import (
    AudioBackend,
    AudioDeviceInfo,
    AudioDeviceType,
    AudioFormat,
    AudioStream,
    AudioCallback
)
import logging
import numpy as np
import threading
import time
from typing import Optional

from ...constants import (
    DEFAULT_AUDIO_SAMPLE_RATE, FALLBACK_AUDIO_SAMPLE_RATE,
    DEFAULT_AUDIO_CHANNELS, DEFAULT_AUDIO_BUFFER_SIZE,
    AUDIO_THREAD_SLEEP_FACTOR
)

logger = logging.getLogger(__name__)


class NullAudioBackend(AudioBackend):
    """Null audio backend for testing without real audio hardware.

    This backend provides sensible defaults and produces/consumes silence.
    Streams run in real-time but don't interact with actual audio devices.
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

        Args:
            device_type: Type of devices to enumerate

        Returns:
            List of dummy audio devices
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
        """Get default null audio device.

        Args:
            device_type: Type of device

        Returns:
            Default device (first enumerated device)
        """
        return self.enumerate_devices(device_type)[0]

    def open_stream(
        self,
        device_info: AudioDeviceInfo,
        callback: AudioCallback,
        buffer_size: int = DEFAULT_AUDIO_BUFFER_SIZE
    ) -> AudioStream:
        """Open a null audio stream.

        Stream will call callback at appropriate intervals but won't
        produce actual audio output.

        Args:
            device_info: Device configuration
            callback: Audio processing callback
            buffer_size: Buffer size in frames

        Returns:
            Opened audio stream
        """
        stream = AudioStream(device_info, callback, buffer_size, self)
        self._streams.add(stream)
        return stream

    def _start_stream(self, stream: AudioStream) -> None:
        """Start null stream processing thread.

        Args:
            stream: Stream to start
        """
        def stream_thread():
            """Stream processing thread that simulates real-time audio."""
            sample_rate = stream.device_info.sample_rate
            buffer_size = stream.buffer_size
            channels = stream.device_info.channel_count

            # Calculate sleep time to simulate real-time audio
            # Add small margin for processing overhead
            sleep_time = (buffer_size / sample_rate) * AUDIO_THREAD_SLEEP_FACTOR

            while not stream._stop_event.is_set():
                start_time = time.time()

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

                    # Validate output if provided
                    if output_buffer is not None:
                        if not isinstance(output_buffer, np.ndarray):
                            raise TypeError(
                                f"Callback must return np.ndarray or None, "
                                f"got {type(output_buffer)}"
                            )

                        expected_shape = (buffer_size, channels)
                        if output_buffer.shape != expected_shape:
                            raise ValueError(
                                f"Output buffer shape mismatch: "
                                f"expected {expected_shape}, got {output_buffer.shape}"
                            )

                except Exception as e:
                    logger.error(f"Audio callback error: {e}", exc_info=True)
                    break

                # Discard output (null backend doesn't play audio)
                # In a real backend, this would be sent to audio hardware

                # Sleep to maintain real-time pace
                elapsed = time.time() - start_time
                remaining = sleep_time - elapsed
                if remaining > 0:
                    time.sleep(remaining)

        stream._thread = threading.Thread(target=stream_thread, daemon=True)
        stream._thread.start()

    def _stop_stream(self, stream: AudioStream) -> None:
        """Stop null stream processing thread.

        Args:
            stream: Stream to stop
        """
        if stream._thread and stream._thread.is_alive():
            stream._stop_event.set()
            stream._thread.join(timeout=1.0)

        if stream in self._streams:
            self._streams.remove(stream)

    def _get_latency_frames(self, stream: AudioStream) -> int:
        """Get null stream latency (returns buffer size).

        Args:
            stream: Stream to query

        Returns:
            Latency in frames
        """
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
        """
        if count < 1:
            raise ValueError("Device count must be at least 1")
        self._device_count[device_type] = count
