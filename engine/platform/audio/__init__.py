"""
Platform audio API for cross-platform audio I/O and spatial audio.

This module provides:
- Audio device enumeration and management
- Audio stream I/O with callback-based processing
- Spatial 3D audio with positional sources and reverb
- Multiple backend support (null backend included)

Example usage:
    >>> from engine.platform.audio import (
    ...     AudioDevice, AudioDeviceType, AudioStream,
    ...     SpatialAudioEngine, SpatialSource, Vec3
    ... )
    >>>
    >>> # List audio devices
    >>> devices = AudioDevice.enumerate(AudioDeviceType.PLAYBACK)
    >>> for device in devices:
    ...     print(device.name)
    >>>
    >>> # Open audio stream
    >>> def callback(input_buffer, frame_count):
    ...     # Generate audio
    ...     return np.zeros((frame_count, 2))
    >>>
    >>> device = AudioDevice.default_device(AudioDeviceType.PLAYBACK)
    >>> stream = AudioDevice.open(device, callback)
    >>> stream.start()
    >>>
    >>> # Spatial audio
    >>> engine = SpatialAudioEngine()
    >>> source = SpatialSource(position=Vec3(10, 0, 0))
    >>> handle = engine.create_source(source)
"""

# Audio device API
from .audio_device import (
    AudioDevice,
    AudioDeviceInfo,
    AudioDeviceType,
    AudioFormat,
    AudioStream,
    AudioCallback,
    AudioBackend,
    NullAudioBackend,
    set_backend,
)

# Spatial audio API
from .spatial import (
    SpatialAudioEngine,
    SpatialAudioAPI,
    SpatialSource,
    SpatialListener,
    ReverbPreset,
    Vec3,
)

# Backend registry
from .backends import (
    register_backend,
    get_backend,
    get_default_backend,
    list_backends,
    create_backend,
)

__all__ = [
    # Audio device
    "AudioDevice",
    "AudioDeviceInfo",
    "AudioDeviceType",
    "AudioFormat",
    "AudioStream",
    "AudioCallback",
    "AudioBackend",
    "NullAudioBackend",
    "set_backend",
    # Spatial audio
    "SpatialAudioEngine",
    "SpatialAudioAPI",
    "SpatialSource",
    "SpatialListener",
    "ReverbPreset",
    "Vec3",
    # Backend registry
    "register_backend",
    "get_backend",
    "get_default_backend",
    "list_backends",
    "create_backend",
]
