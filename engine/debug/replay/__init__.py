"""Replay and recording system for debug and analysis.

This module provides comprehensive replay functionality including:
- Input recording for deterministic replay
- State recording for snapshot-based replay
- Rolling buffer recording for crash replay
- Playback with speed control, seeking, and reverse
- Multiple camera modes for viewing replays
- Efficient storage with compression and delta encoding
- Frame capture for screenshots, video, and GIFs

Example:
    # Recording inputs
    from engine.debug.replay import InputRecorder, RecordingMode

    recorder = InputRecorder(mode=RecordingMode.CONTINUOUS)
    recorder.start()
    recorder.record_input("keyboard", {"key": "W", "action": "press"})
    recorder.save("replay.json")

    # Playback
    from engine.debug.replay import ReplayPlayer, PlaybackState

    player = ReplayPlayer()
    player.load("replay.json")
    player.play()
    player.set_speed(0.5)  # Slow motion

    # Camera control
    from engine.debug.replay import ReplayCamera, ReplayCameraMode

    camera = ReplayCamera()
    camera.set_mode(ReplayCameraMode.ORBIT)
    camera.set_target(player_entity_id)
"""

from engine.debug.replay.recorder import (
    RecordingMode,
    InputRecord,
    StateSnapshot,
    RecorderBase,
    InputRecorder,
    StateRecorder,
    RollingRecorder,
)

from engine.debug.replay.player import (
    PlaybackState,
    PlaybackInfo,
    ReplayPlayer,
)

from engine.debug.replay.camera import (
    Vec3,
    Mat4,
    ReplayCameraMode,
    EntityProvider,
    CameraSettings,
    ReplayCamera,
)

from engine.debug.replay.storage import (
    CompressionLevel,
    DeltaData,
    DeltaEncoder,
    ReplayStorage,
    ContentAddressedStorage,
)

from engine.debug.replay.capture import (
    CaptureFormat,
    FrameData,
    FrameProvider,
    ImageEncoder,
    PNGEncoder,
    JPEGEncoder,
    VideoEncoder,
    RawVideoEncoder,
    GIFEncoder,
    FrameCapture,
)


__all__ = [
    # recorder.py
    "RecordingMode",
    "InputRecord",
    "StateSnapshot",
    "RecorderBase",
    "InputRecorder",
    "StateRecorder",
    "RollingRecorder",
    # player.py
    "PlaybackState",
    "PlaybackInfo",
    "ReplayPlayer",
    # camera.py
    "Vec3",
    "Mat4",
    "ReplayCameraMode",
    "EntityProvider",
    "CameraSettings",
    "ReplayCamera",
    # storage.py
    "CompressionLevel",
    "DeltaData",
    "DeltaEncoder",
    "ReplayStorage",
    "ContentAddressedStorage",
    # capture.py
    "CaptureFormat",
    "FrameData",
    "FrameProvider",
    "ImageEncoder",
    "PNGEncoder",
    "JPEGEncoder",
    "VideoEncoder",
    "RawVideoEncoder",
    "GIFEncoder",
    "FrameCapture",
]
