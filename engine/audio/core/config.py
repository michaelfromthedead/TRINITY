"""
Audio Core Configuration Constants

All audio-related constants are centralized here.
NO magic numbers should appear in other audio modules.
"""

from enum import IntEnum, auto
from typing import Final

# =============================================================================
# Audio Limits
# =============================================================================

MAX_VOICES: Final[int] = 64
MAX_SFX_VOICES: Final[int] = 48
MAX_MUSIC_VOICES: Final[int] = 8
MAX_VO_VOICES: Final[int] = 8
MAX_AMBIENT_VOICES: Final[int] = 16
MAX_INSTANCES_PER_SOUND: Final[int] = 4

# =============================================================================
# Buffer Sizes
# =============================================================================

AUDIO_BUFFER_SIZE: Final[int] = 4096
STREAM_BUFFER_SIZE: Final[int] = 16384
RING_BUFFER_SIZE: Final[int] = 65536
DECODE_BUFFER_SIZE: Final[int] = 8192

# =============================================================================
# Sample Rates
# =============================================================================

SAMPLE_RATE_44100: Final[int] = 44100
SAMPLE_RATE_48000: Final[int] = 48000
SAMPLE_RATE_96000: Final[int] = 96000
DEFAULT_SAMPLE_RATE: Final[int] = SAMPLE_RATE_48000

SUPPORTED_SAMPLE_RATES: Final[tuple[int, ...]] = (
    SAMPLE_RATE_44100,
    SAMPLE_RATE_48000,
    SAMPLE_RATE_96000,
)

# =============================================================================
# Memory Budgets (bytes)
# =============================================================================

AUDIO_MEMORY_BUDGET: Final[int] = 256 * 1024 * 1024  # 256MB total
SFX_MEMORY_BUDGET: Final[int] = 128 * 1024 * 1024    # 128MB
MUSIC_MEMORY_BUDGET: Final[int] = 64 * 1024 * 1024   # 64MB
VO_MEMORY_BUDGET: Final[int] = 64 * 1024 * 1024      # 64MB
AMBIENT_MEMORY_BUDGET: Final[int] = 32 * 1024 * 1024 # 32MB

# Memory pool sizes
RESIDENT_POOL_SIZE: Final[int] = 64 * 1024 * 1024    # 64MB
STREAMING_POOL_SIZE: Final[int] = 32 * 1024 * 1024   # 32MB
TEMPORARY_POOL_SIZE: Final[int] = 16 * 1024 * 1024   # 16MB

# =============================================================================
# Streaming Configuration
# =============================================================================

STREAM_PREFETCH_SIZE: Final[int] = 32768
STREAM_CHUNK_SIZE: Final[int] = 8192
STREAM_BUFFER_COUNT: Final[int] = 4
STREAM_LOW_WATERMARK: Final[int] = STREAM_CHUNK_SIZE * 2
STREAM_HIGH_WATERMARK: Final[int] = STREAM_BUFFER_SIZE - STREAM_CHUNK_SIZE

# =============================================================================
# Voice Priority Levels
# =============================================================================

PRIORITY_CRITICAL: Final[int] = 100
PRIORITY_HIGH: Final[int] = 75
PRIORITY_NORMAL: Final[int] = 50
PRIORITY_LOW: Final[int] = 25
PRIORITY_BACKGROUND: Final[int] = 10

# =============================================================================
# Variation Ranges
# =============================================================================

PITCH_VARIATION_RANGE: Final[float] = 0.1      # ±10% pitch variation
VOLUME_VARIATION_DB: Final[float] = 3.0        # ±3dB volume variation
MAX_START_OFFSET_RATIO: Final[float] = 0.1     # Max 10% start offset

# =============================================================================
# Spatial Audio
# =============================================================================

DEFAULT_MIN_DISTANCE: Final[float] = 1.0
DEFAULT_MAX_DISTANCE: Final[float] = 100.0
DEFAULT_ROLLOFF_FACTOR: Final[float] = 1.0
DOPPLER_SCALE: Final[float] = 1.0
SPEED_OF_SOUND: Final[float] = 343.0  # m/s at 20°C

# =============================================================================
# Threading Configuration
# =============================================================================

AUDIO_THREAD_PRIORITY: Final[int] = 2  # Higher than normal
STREAM_THREAD_PRIORITY: Final[int] = 1
DECODE_THREAD_PRIORITY: Final[int] = 1
AUDIO_TICK_RATE_MS: Final[int] = 5     # 200Hz update rate
STREAM_TICK_RATE_MS: Final[int] = 10   # 100Hz for streaming

# =============================================================================
# Fade Configuration
# =============================================================================

DEFAULT_FADE_IN_MS: Final[int] = 10
DEFAULT_FADE_OUT_MS: Final[int] = 50
VOICE_STEAL_FADE_MS: Final[int] = 20

# =============================================================================
# Enumerations
# =============================================================================


class AudioFormat(IntEnum):
    """Audio sample formats."""
    PCM_INT16 = auto()
    PCM_INT24 = auto()
    PCM_FLOAT32 = auto()
    ADPCM = auto()
    VORBIS = auto()
    OPUS = auto()
    MP3 = auto()
    AAC = auto()


class ChannelLayout(IntEnum):
    """Audio channel configurations."""
    MONO = 1
    STEREO = 2
    SURROUND_5_1 = 6
    SURROUND_7_1 = 8


class AudioCategory(IntEnum):
    """Audio categories for voice management and mixing."""
    MASTER = 0
    SFX = auto()
    MUSIC = auto()
    VOICE_OVER = auto()
    AMBIENT = auto()
    UI = auto()


class VoiceState(IntEnum):
    """Voice playback states."""
    STOPPED = 0
    PLAYING = auto()
    PAUSED = auto()
    STOPPING = auto()
    VIRTUAL = auto()


class VoiceStealStrategy(IntEnum):
    """Strategies for voice stealing when limit is reached."""
    OLDEST = 0
    QUIETEST = auto()
    FARTHEST = auto()
    LOWEST_PRIORITY = auto()
    NONE = auto()  # Don't steal, fail allocation


class SoundCueType(IntEnum):
    """Types of sound cue playback."""
    SIMPLE = 0
    RANDOM = auto()
    SEQUENCE = auto()
    SWITCH = auto()
    SHUFFLE = auto()


class SourceType(IntEnum):
    """Types of audio sources."""
    ONE_SHOT = 0
    LOOPING = auto()
    STREAMING = auto()


class MemoryPoolType(IntEnum):
    """Memory pool types for audio data."""
    RESIDENT = 0    # Always loaded, frequently used
    STREAMING = auto()  # Streamed from disk
    TEMPORARY = auto()  # One-shot, can be evicted


class AttenuationModel(IntEnum):
    """Distance attenuation models."""
    NONE = 0
    LINEAR = auto()
    INVERSE = auto()
    INVERSE_CLAMPED = auto()
    EXPONENTIAL = auto()


# =============================================================================
# Category Limits Mapping
# =============================================================================

CATEGORY_VOICE_LIMITS: dict[AudioCategory, int] = {
    AudioCategory.MASTER: MAX_VOICES,
    AudioCategory.SFX: MAX_SFX_VOICES,
    AudioCategory.MUSIC: MAX_MUSIC_VOICES,
    AudioCategory.VOICE_OVER: MAX_VO_VOICES,
    AudioCategory.AMBIENT: MAX_AMBIENT_VOICES,
    AudioCategory.UI: 16,
}

CATEGORY_MEMORY_BUDGETS: dict[AudioCategory, int] = {
    AudioCategory.MASTER: AUDIO_MEMORY_BUDGET,
    AudioCategory.SFX: SFX_MEMORY_BUDGET,
    AudioCategory.MUSIC: MUSIC_MEMORY_BUDGET,
    AudioCategory.VOICE_OVER: VO_MEMORY_BUDGET,
    AudioCategory.AMBIENT: AMBIENT_MEMORY_BUDGET,
    AudioCategory.UI: 8 * 1024 * 1024,
}

# =============================================================================
# Threshold Constants
# =============================================================================

VOLUME_COMPARISON_THRESHOLD: Final[float] = 0.01  # Volume equality threshold
DISTANCE_COMPARISON_THRESHOLD: Final[float] = 0.1  # Distance equality threshold
MINIMUM_DB_VALUE: Final[float] = -96.0  # Minimum dB for silent audio
PAN_ANGLE_MULTIPLIER: Final[float] = 0.25  # Pan angle calculation: pan * multiplier * pi

# =============================================================================
# Virtual Voice Configuration
# =============================================================================

VIRTUAL_VOICE_MAX_TIME_SECONDS: Final[float] = 10.0
VIRTUAL_VOICE_URGENCY_PRIORITY_WEIGHT: Final[float] = 2.0
VIRTUAL_VOICE_URGENCY_TIME_WEIGHT: Final[float] = 1.0
VIRTUAL_VOICE_URGENCY_RISE_WEIGHT: Final[float] = 1.5
VIRTUAL_VOICE_PROMOTION_COOLDOWN_MS: Final[int] = 100
VIRTUAL_VOICE_FORCE_PROMOTE_GRACE_MS: Final[int] = 500

# =============================================================================
# Pool Sizes
# =============================================================================

SOURCE_POOL_INITIAL_SIZE: Final[int] = 32  # Initial audio source pool size
STREAM_BUFFER_MAX_MULTIPLIER: Final[int] = 2  # Max stream buffers = count * multiplier

# =============================================================================
# Virtual Voice Configuration
# =============================================================================

VIRTUAL_VOICE_MAX_TIME_SECONDS: Final[float] = 10.0
VIRTUAL_VOICE_URGENCY_PRIORITY_WEIGHT: Final[float] = 2.0
VIRTUAL_VOICE_URGENCY_TIME_WEIGHT: Final[float] = 1.0
VIRTUAL_VOICE_URGENCY_RISE_WEIGHT: Final[float] = 1.5
VIRTUAL_VOICE_PROMOTION_COOLDOWN_MS: Final[int] = 100
VIRTUAL_VOICE_FORCE_PROMOTE_GRACE_MS: Final[int] = 500

# =============================================================================
# Format Properties
# =============================================================================

FORMAT_BITS_PER_SAMPLE: dict[AudioFormat, int] = {
    AudioFormat.PCM_INT16: 16,
    AudioFormat.PCM_INT24: 24,
    AudioFormat.PCM_FLOAT32: 32,
    AudioFormat.ADPCM: 4,  # Compressed ratio
    AudioFormat.VORBIS: 0,  # Variable
    AudioFormat.OPUS: 0,
    AudioFormat.MP3: 0,
    AudioFormat.AAC: 0,
}

COMPRESSED_FORMATS: frozenset[AudioFormat] = frozenset({
    AudioFormat.ADPCM,
    AudioFormat.VORBIS,
    AudioFormat.OPUS,
    AudioFormat.MP3,
    AudioFormat.AAC,
})

UNCOMPRESSED_FORMATS: frozenset[AudioFormat] = frozenset({
    AudioFormat.PCM_INT16,
    AudioFormat.PCM_INT24,
    AudioFormat.PCM_FLOAT32,
})

# =============================================================================
# Virtual Voice Configuration
# =============================================================================

VIRTUAL_VOICE_MAX_TIME_SECONDS: Final[float] = 10.0
VIRTUAL_VOICE_URGENCY_PRIORITY_WEIGHT: Final[float] = 2.0
VIRTUAL_VOICE_URGENCY_TIME_WEIGHT: Final[float] = 1.0
VIRTUAL_VOICE_URGENCY_RISE_WEIGHT: Final[float] = 1.5
VIRTUAL_VOICE_PROMOTION_COOLDOWN_MS: Final[int] = 100
VIRTUAL_VOICE_FORCE_PROMOTE_GRACE_MS: Final[int] = 500
