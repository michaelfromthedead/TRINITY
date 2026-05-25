"""
Replay System Configuration - Centralized constants and settings.

All magic numbers and configuration constants for the replay system
are defined here for maintainability and consistency.
"""

from dataclasses import dataclass
from typing import Final


# =============================================================================
# File Format Constants
# =============================================================================

# Replay file magic number (identifies file type)
REPLAY_FILE_MAGIC: Final[bytes] = b'RPLY'

# Replay file format version
REPLAY_FILE_VERSION: Final[int] = 1

# Header size in bytes (fixed structure)
REPLAY_HEADER_SIZE: Final[int] = 82  # 10 + 16 + 16 + 8 + 32 bytes


# =============================================================================
# Replay Buffer and Memory Settings
# =============================================================================

# Maximum inputs to buffer before auto-flush
MAX_INPUT_BUFFER_SIZE: Final[int] = 100000

# Auto-flush interval in seconds
INPUT_FLUSH_INTERVAL: Final[float] = 5.0

# Maximum state size in bytes (10 MB)
MAX_STATE_SIZE: Final[int] = 10 * 1024 * 1024

# Maximum number of snapshots to store
MAX_SNAPSHOTS: Final[int] = 10000

# Maximum number of drifts to collect before stopping
MAX_DRIFT_REPORTS: Final[int] = 1000


# =============================================================================
# Compression Settings
# =============================================================================

# Default zlib compression level (0-9, 6 is balanced)
ZLIB_DEFAULT_LEVEL: Final[int] = 6

# Fast compression level
ZLIB_FAST_LEVEL: Final[int] = 1

# Best compression level
ZLIB_BEST_LEVEL: Final[int] = 9


# =============================================================================
# State Snapshot Intervals
# =============================================================================

# Frames between full keyframe snapshots
DEFAULT_KEYFRAME_INTERVAL: Final[int] = 60

# Frames between delta snapshots
DEFAULT_DELTA_INTERVAL: Final[int] = 1

# Minimum change ratio to store a delta (0.0-1.0)
DELTA_THRESHOLD: Final[float] = 0.1


# =============================================================================
# Playback Speed Settings
# =============================================================================

# Default playback speed multiplier
DEFAULT_PLAYBACK_SPEED: Final[float] = 1.0

# Minimum allowed playback speed
MIN_PLAYBACK_SPEED: Final[float] = 0.1

# Maximum allowed playback speed
MAX_PLAYBACK_SPEED: Final[float] = 10.0

# Preset playback speeds
PLAYBACK_SPEED_QUARTER: Final[float] = 0.25
PLAYBACK_SPEED_HALF: Final[float] = 0.5
PLAYBACK_SPEED_NORMAL: Final[float] = 1.0
PLAYBACK_SPEED_DOUBLE: Final[float] = 2.0
PLAYBACK_SPEED_QUADRUPLE: Final[float] = 4.0


# =============================================================================
# Ghost System Settings
# =============================================================================

# Default ghost opacity (0.0-1.0)
DEFAULT_GHOST_OPACITY: Final[float] = 0.5

# Default ghost color (RGB)
DEFAULT_GHOST_COLOR: Final[tuple[int, int, int]] = (100, 100, 255)

# Default ghost outline color (RGB)
DEFAULT_GHOST_OUTLINE_COLOR: Final[tuple[int, int, int]] = (255, 255, 255)

# Default outline width in pixels
DEFAULT_GHOST_OUTLINE_WIDTH: Final[float] = 2.0

# Trail length in frames for TRAIL render mode
DEFAULT_GHOST_TRAIL_LENGTH: Final[int] = 30

# Maximum visibility distance for ghost rendering
DEFAULT_GHOST_VISIBLE_DISTANCE: Final[float] = 100.0

# Distance at which ghost starts to fade
DEFAULT_GHOST_FADE_DISTANCE: Final[float] = 80.0


# =============================================================================
# Determinism Checker Settings
# =============================================================================

# Float comparison tolerance (absolute)
FLOAT_TOLERANCE: Final[float] = 1e-6

# Float comparison tolerance (relative)
FLOAT_RELATIVE_TOLERANCE: Final[float] = 1e-5

# Position comparison tolerance
POSITION_TOLERANCE: Final[float] = 0.001

# Rotation comparison tolerance (radians)
ROTATION_TOLERANCE: Final[float] = 0.001

# Drift severity thresholds
MINOR_DRIFT_THRESHOLD: Final[float] = 0.01
MODERATE_DRIFT_THRESHOLD: Final[float] = 0.1
MAJOR_DRIFT_THRESHOLD: Final[float] = 1.0


# =============================================================================
# Export Settings
# =============================================================================

# Default video export FPS
DEFAULT_EXPORT_FPS: Final[int] = 30

# Default video dimensions
DEFAULT_EXPORT_WIDTH: Final[int] = 1920
DEFAULT_EXPORT_HEIGHT: Final[int] = 1080

# Default video bitrate
DEFAULT_VIDEO_BITRATE: Final[str] = "8M"

# Default Constant Rate Factor for video quality (0-51, lower = better)
DEFAULT_VIDEO_CRF: Final[int] = 23

# Default audio bitrate
DEFAULT_AUDIO_BITRATE: Final[str] = "192k"

# GIF export defaults
DEFAULT_GIF_FPS: Final[int] = 15
DEFAULT_GIF_WIDTH: Final[int] = 480
DEFAULT_GIF_HEIGHT: Final[int] = 270
DEFAULT_GIF_COLORS: Final[int] = 256
DEFAULT_GIF_QUALITY: Final[int] = 85

# Estimated bytes per second for GIF
GIF_ESTIMATED_BYTES_PER_SECOND: Final[int] = 50 * 1024


# =============================================================================
# Timeline Settings
# =============================================================================

# Default frames per second for timeline
DEFAULT_TIMELINE_FPS: Final[float] = 60.0

# Default track height in pixels
DEFAULT_TRACK_HEIGHT: Final[int] = 24


# =============================================================================
# Input Recording Settings
# =============================================================================

# Mouse move deduplication threshold in seconds
MOUSE_MOVE_THRESHOLD: Final[float] = 0.001


# =============================================================================
# Timestamp Limits (for overflow prevention)
# =============================================================================

# Maximum safe timestamp value (to prevent 32-bit overflow)
# This represents ~68 years worth of nanoseconds in a 64-bit integer
MAX_TIMESTAMP_SECONDS: Final[float] = 2**52  # Safe for float precision

# Maximum safe frame number (to prevent 32-bit overflow)
MAX_FRAME_NUMBER: Final[int] = 2**31 - 1  # Max signed 32-bit int


# =============================================================================
# Browser Settings
# =============================================================================

# Default file extensions for replay files
DEFAULT_REPLAY_EXTENSIONS: Final[list[str]] = ['.replay', '.rpy', '.rep']

# Default page size for search results
DEFAULT_PAGE_SIZE: Final[int] = 50


# =============================================================================
# Quaternion Interpolation Settings
# =============================================================================

# Threshold for linear vs spherical interpolation
SLERP_THRESHOLD: Final[float] = 0.9995


@dataclass(frozen=True)
class ReplaySystemLimits:
    """Hard limits for the replay system to prevent memory/overflow issues."""

    # Maximum replay duration (24 hours in seconds)
    max_replay_duration: float = 86400.0

    # Maximum number of inputs per replay
    max_inputs: int = 10_000_000

    # Maximum number of ghosts that can be active simultaneously
    max_active_ghosts: int = 10

    # Maximum number of markers on a timeline
    max_timeline_markers: int = 10000

    # Maximum number of segments on a timeline
    max_timeline_segments: int = 1000


# Global instance for easy access
LIMITS = ReplaySystemLimits()
