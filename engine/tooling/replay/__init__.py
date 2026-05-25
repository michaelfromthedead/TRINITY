"""
Replay System - Record, playback, and analyze game sessions.

This module provides comprehensive replay functionality including:
- Input recording with precise timestamps
- State snapshots for seeking and determinism verification
- Variable speed playback with frame stepping
- Ghost replay system for racing/speedrun comparisons
- Replay file format with compression and metadata
- Timeline visualization with markers and events
- Export to video/GIF formats
"""

from .input_recorder import (
    InputRecorder,
    RecordedInput,
    InputRecordingConfig,
    InputRecordingStats,
)
from .state_recorder import (
    StateRecorder,
    StateSnapshot,
    StateDelta,
    StateRecordingConfig,
    CompressionMethod,
)
from .replay_playback import (
    ReplayPlayback,
    PlaybackState,
    PlaybackConfig,
    PlaybackSpeed,
    SeekMode,
)
from .replay_file import (
    ReplayFile,
    ReplayHeader,
    ReplayMetadata,
    ReplayFileFormat,
    ReplayFileError,
)
from .determinism_checker import (
    DeterminismChecker,
    DeterminismResult,
    DriftReport,
    DriftSeverity,
    StateComparisonConfig,
)
from .replay_browser import (
    ReplayBrowser,
    ReplayEntry,
    ReplayFilter,
    ReplaySortOrder,
    ReplaySearchResult,
)
from .replay_timeline import (
    ReplayTimeline,
    TimelineMarker,
    TimelineEvent,
    TimelineSegment,
    MarkerType,
)
from .replay_export import (
    ReplayExporter,
    ExportFormat,
    ExportConfig,
    ExportProgress,
    VideoCodec,
    GifConfig,
)
from .ghost_system import (
    GhostSystem,
    Ghost,
    GhostConfig,
    GhostRenderMode,
    GhostComparison,
)

__all__ = [
    # Input Recording
    "InputRecorder",
    "RecordedInput",
    "InputRecordingConfig",
    "InputRecordingStats",
    # State Recording
    "StateRecorder",
    "StateSnapshot",
    "StateDelta",
    "StateRecordingConfig",
    "CompressionMethod",
    # Playback
    "ReplayPlayback",
    "PlaybackState",
    "PlaybackConfig",
    "PlaybackSpeed",
    "SeekMode",
    # File Format
    "ReplayFile",
    "ReplayHeader",
    "ReplayMetadata",
    "ReplayFileFormat",
    "ReplayFileError",
    # Determinism
    "DeterminismChecker",
    "DeterminismResult",
    "DriftReport",
    "DriftSeverity",
    "StateComparisonConfig",
    # Browser
    "ReplayBrowser",
    "ReplayEntry",
    "ReplayFilter",
    "ReplaySortOrder",
    "ReplaySearchResult",
    # Timeline
    "ReplayTimeline",
    "TimelineMarker",
    "TimelineEvent",
    "TimelineSegment",
    "MarkerType",
    # Export
    "ReplayExporter",
    "ExportFormat",
    "ExportConfig",
    "ExportProgress",
    "VideoCodec",
    "GifConfig",
    # Ghost System
    "GhostSystem",
    "Ghost",
    "GhostConfig",
    "GhostRenderMode",
    "GhostComparison",
]

__version__ = "1.0.0"
