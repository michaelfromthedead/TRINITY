"""
Dialogue System Configuration Constants.

All configuration values for the dialogue and voice-over system including
priorities, cooldowns, streaming settings, processing parameters, and more.
"""

from enum import Enum, IntEnum
from typing import Final

# =============================================================================
# Priority Levels
# =============================================================================

class VOPriority(IntEnum):
    """Voice-over priority levels for queue management."""
    CRITICAL = 100    # Never interrupted (story critical)
    HIGH = 75         # Story dialogue
    NORMAL = 50       # Standard VO
    LOW = 40          # Low priority dialogue
    BARK = 25         # Combat barks (lower than LOW)
    AMBIENT = 10      # Background ambient


# Priority constants for direct access
PRIORITY_CRITICAL: Final[int] = VOPriority.CRITICAL
PRIORITY_HIGH: Final[int] = VOPriority.HIGH
PRIORITY_NORMAL: Final[int] = VOPriority.NORMAL
PRIORITY_LOW: Final[int] = VOPriority.LOW
PRIORITY_BARK: Final[int] = VOPriority.BARK
PRIORITY_AMBIENT: Final[int] = VOPriority.AMBIENT

# Default interrupt threshold - lines with priority >= this can interrupt
DEFAULT_INTERRUPT_PRIORITY: Final[int] = 60


# =============================================================================
# Queue Configuration
# =============================================================================

MAX_QUEUE_SIZE: Final[int] = 256
QUEUE_TIMEOUT_MS: Final[float] = 5000.0
QUEUE_PROCESS_INTERVAL_MS: Final[float] = 16.0  # ~60fps


# =============================================================================
# Cooldown Configuration (in milliseconds)
# =============================================================================

BARK_COOLDOWN_MS: Final[float] = 3000.0
SAME_LINE_COOLDOWN_MS: Final[float] = 30000.0
SAME_SPEAKER_COOLDOWN_MS: Final[float] = 1000.0
CONVERSATION_LINE_COOLDOWN_MS: Final[float] = 500.0


# =============================================================================
# Overlap Configuration
# =============================================================================

MAX_SIMULTANEOUS_VO: Final[int] = 2
OVERLAP_DUCK_DB: Final[float] = -6.0  # Ducking amount for overlapped VO
CONVERSATION_GAP_MS: Final[float] = 200.0  # Gap between conversation lines


# =============================================================================
# Streaming Configuration
# =============================================================================

VO_PRELOAD_TIME_MS: Final[float] = 500.0
VO_CACHE_SIZE_MB: Final[int] = 32
VO_STREAM_BUFFER_MS: Final[float] = 100.0
VO_MAX_PRELOAD_COUNT: Final[int] = 8
VO_CACHE_EVICTION_THRESHOLD: Final[float] = 0.9  # Evict when 90% full


# =============================================================================
# Audio Processing Configuration
# =============================================================================

# Radio/communication effect
RADIO_BAND_LOW: Final[float] = 300.0
RADIO_BAND_HIGH: Final[float] = 3400.0
RADIO_DISTORTION: Final[float] = 0.3
RADIO_NOISE_LEVEL: Final[float] = 0.05

# Distance filtering
DISTANCE_FILTER_START: Final[float] = 10.0  # Start filtering at this distance
DISTANCE_FILTER_MAX: Final[float] = 50.0    # Maximum distance for filtering
DISTANCE_FILTER_CUTOFF_MIN: Final[float] = 500.0  # Hz cutoff at max distance

# Reverb
VO_REVERB_SEND_DEFAULT: Final[float] = 0.2
VO_REVERB_SEND_MAX: Final[float] = 0.8

# Spatialization
VO_SPATIAL_BLEND_DEFAULT: Final[float] = 1.0  # 0=2D, 1=3D
VO_SPATIAL_MIN_DISTANCE: Final[float] = 1.0
VO_SPATIAL_MAX_DISTANCE: Final[float] = 50.0


# =============================================================================
# Subtitle Configuration
# =============================================================================

SUBTITLE_FADE_TIME_MS: Final[float] = 200.0
SUBTITLE_MIN_DISPLAY_MS: Final[float] = 1500.0
SUBTITLE_CHARS_PER_SECOND: Final[int] = 15
MAX_SUBTITLE_LINES: Final[int] = 3
SUBTITLE_LINE_HEIGHT: Final[int] = 24
SUBTITLE_MAX_WIDTH_PERCENT: Final[float] = 0.8  # 80% of screen width


# =============================================================================
# Localization Configuration
# =============================================================================

DEFAULT_LANGUAGE: Final[str] = "en"
SUPPORTED_LANGUAGES: Final[tuple[str, ...]] = (
    "en",  # English
    "es",  # Spanish
    "fr",  # French
    "de",  # German
    "it",  # Italian
    "ja",  # Japanese
    "ko",  # Korean
    "zh",  # Chinese
    "pt",  # Portuguese
    "ru",  # Russian
)


# =============================================================================
# Context Types
# =============================================================================

class ContextType(str, Enum):
    """Types of contextual dialogue."""
    BARK = "bark"
    CONVERSATION = "conversation"
    AMBIENT = "ambient"
    NARRATION = "narration"
    TUTORIAL = "tutorial"
    COMBAT = "combat"
    EXPLORATION = "exploration"


CONTEXT_BARK: Final[str] = ContextType.BARK.value
CONTEXT_CONVERSATION: Final[str] = ContextType.CONVERSATION.value
CONTEXT_AMBIENT: Final[str] = ContextType.AMBIENT.value
CONTEXT_NARRATION: Final[str] = ContextType.NARRATION.value
CONTEXT_TUTORIAL: Final[str] = ContextType.TUTORIAL.value
CONTEXT_COMBAT: Final[str] = ContextType.COMBAT.value
CONTEXT_EXPLORATION: Final[str] = ContextType.EXPLORATION.value


# =============================================================================
# Line Selection Modes
# =============================================================================

class SelectionMode(str, Enum):
    """Modes for selecting lines from pools."""
    RANDOM = "random"
    SEQUENTIAL = "sequential"
    WEIGHTED = "weighted"
    CONDITIONAL = "conditional"
    SHUFFLE = "shuffle"


SELECTION_RANDOM: Final[str] = SelectionMode.RANDOM.value
SELECTION_SEQUENTIAL: Final[str] = SelectionMode.SEQUENTIAL.value
SELECTION_WEIGHTED: Final[str] = SelectionMode.WEIGHTED.value
SELECTION_CONDITIONAL: Final[str] = SelectionMode.CONDITIONAL.value
SELECTION_SHUFFLE: Final[str] = SelectionMode.SHUFFLE.value


# =============================================================================
# Dialogue State
# =============================================================================

class DialogueState(str, Enum):
    """States for the dialogue manager."""
    IDLE = "idle"
    PENDING = "pending"
    ACTIVE = "active"
    PLAYING = "playing"
    PAUSED = "paused"
    COMPLETE = "complete"
    TRANSITIONING = "transitioning"
    INTERRUPTED = "interrupted"


# =============================================================================
# Audio Bus Configuration
# =============================================================================

VO_BUS_NAME: Final[str] = "voice"
VO_BUS_VOLUME_DEFAULT: Final[float] = 1.0
VO_DUCKING_TIME_MS: Final[float] = 200.0
VO_DUCKING_RELEASE_MS: Final[float] = 500.0


# =============================================================================
# Lip Sync Configuration
# =============================================================================

LIPSYNC_PHONEME_RATE: Final[float] = 30.0  # Updates per second
LIPSYNC_BLEND_TIME_MS: Final[float] = 50.0
LIPSYNC_VISEME_COUNT: Final[int] = 15


# =============================================================================
# Performance Limits
# =============================================================================

MAX_ACTIVE_CONVERSATIONS: Final[int] = 4
MAX_CONCURRENT_STREAMS: Final[int] = 8
MAX_PENDING_PRELOADS: Final[int] = 16

# =============================================================================
# Ambient VO Configuration
# =============================================================================

AMBIENT_MIN_INTERVAL_MS: Final[float] = 10000.0  # Minimum time between ambient lines
AMBIENT_MAX_INTERVAL_MS: Final[float] = 30000.0  # Maximum time between ambient lines


# =============================================================================
# Event Names
# =============================================================================

EVENT_LINE_STARTED: Final[str] = "vo.line.started"
EVENT_LINE_ENDED: Final[str] = "vo.line.ended"
EVENT_LINE_INTERRUPTED: Final[str] = "vo.line.interrupted"
EVENT_CONVERSATION_STARTED: Final[str] = "vo.conversation.started"
EVENT_CONVERSATION_ENDED: Final[str] = "vo.conversation.ended"
EVENT_SUBTITLE_SHOW: Final[str] = "subtitle.show"
EVENT_SUBTITLE_HIDE: Final[str] = "subtitle.hide"
EVENT_LANGUAGE_CHANGED: Final[str] = "localization.language.changed"
