"""
Configuration constants for the audio mixing subsystem.

This module contains all configurable parameters for:
- Bus defaults and limits
- Filter settings
- Ducking parameters
- Sidechain compression
- Mix snapshots
- HDR audio
- Category names
"""

from enum import Enum
from typing import Final

# =============================================================================
# Bus Defaults
# =============================================================================

MASTER_VOLUME: Final[float] = 1.0
DEFAULT_BUS_VOLUME: Final[float] = 1.0
MIN_VOLUME_DB: Final[float] = -80.0
MAX_VOLUME_DB: Final[float] = 12.0

# Volume conversion constants
DB_SILENCE_THRESHOLD: Final[float] = -80.0
LINEAR_SILENCE_THRESHOLD: Final[float] = 0.0001

# Default pitch (1.0 = normal playback rate)
DEFAULT_PITCH: Final[float] = 1.0
MIN_PITCH: Final[float] = 0.1
MAX_PITCH: Final[float] = 4.0

# =============================================================================
# Filter Settings
# =============================================================================

DEFAULT_LOW_PASS: Final[float] = 20000.0  # Hz - high frequency cutoff
DEFAULT_HIGH_PASS: Final[float] = 20.0     # Hz - low frequency cutoff
FILTER_Q: Final[float] = 0.707             # Butterworth Q factor
MIN_FILTER_FREQ: Final[float] = 20.0       # Hz
MAX_FILTER_FREQ: Final[float] = 20000.0    # Hz

# =============================================================================
# Ducking Parameters
# =============================================================================

DIALOGUE_DUCK_AMOUNT_DB: Final[float] = -12.0
EVENT_DUCK_AMOUNT_DB: Final[float] = -6.0
FOCUS_DUCK_AMOUNT_DB: Final[float] = -9.0

DUCK_ATTACK_MS: Final[float] = 50.0
DUCK_RELEASE_MS: Final[float] = 500.0
DUCK_THRESHOLD_DB: Final[float] = -20.0

DUCK_HOLD_MS: Final[float] = 100.0  # Hold time before release starts

# Event duck specific parameters (fast attack for sudden events)
EVENT_DUCK_ATTACK_MS: Final[float] = 10.0
EVENT_DUCK_HOLD_MS: Final[float] = 50.0

# Focus duck specific parameters (slower for smooth attention shifts)
FOCUS_DUCK_ATTACK_MS: Final[float] = 100.0
FOCUS_DUCK_HOLD_MS: Final[float] = 0.0  # No hold for focus
FOCUS_DUCK_RELEASE_MS: Final[float] = 800.0

# =============================================================================
# Sidechain Compression
# =============================================================================

SIDECHAIN_RATIO: Final[float] = 4.0
SIDECHAIN_THRESHOLD_DB: Final[float] = -20.0
SIDECHAIN_ATTACK_MS: Final[float] = 10.0
SIDECHAIN_RELEASE_MS: Final[float] = 100.0
SIDECHAIN_KNEE_DB: Final[float] = 6.0  # Soft knee width
SIDECHAIN_MAKEUP_GAIN_DB: Final[float] = 0.0

# =============================================================================
# Mix Snapshots
# =============================================================================

SNAPSHOT_BLEND_TIME: Final[float] = 1.0  # seconds
MAX_ACTIVE_SNAPSHOTS: Final[int] = 4
DEFAULT_SNAPSHOT_PRIORITY: Final[int] = 100

# Default bus snapshot values (for when capturing state)
DEFAULT_SNAPSHOT_VOLUME: Final[float] = 1.0
DEFAULT_SNAPSHOT_PITCH: Final[float] = 1.0

# Interpolation curve types
class InterpolationCurve(Enum):
    LINEAR = "linear"
    EASE_IN = "ease_in"
    EASE_OUT = "ease_out"
    EASE_IN_OUT = "ease_in_out"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"

DEFAULT_INTERPOLATION_CURVE: Final[InterpolationCurve] = InterpolationCurve.EASE_IN_OUT

# =============================================================================
# HDR Audio
# =============================================================================

HDR_WINDOW_DB: Final[float] = 24.0      # Audible dynamic range window
HDR_ADAPTATION_SPEED: Final[float] = 0.5  # seconds to shift window
HDR_CEILING_DB: Final[float] = 0.0       # Maximum output level
HDR_FLOOR_DB: Final[float] = -60.0       # Minimum audible level
LOUDNESS_UPDATE_RATE: Final[float] = 60.0  # Hz - how often to update loudness
HDR_WINDOW_MIN_DB: Final[float] = 6.0    # Minimum window size
HDR_WINDOW_MAX_DB: Final[float] = 60.0   # Maximum window size
HDR_DEFAULT_CENTER_DB: Final[float] = -30.0  # Default window center
HDR_ACTIVE_THRESHOLD_DB: Final[float] = -70.0  # Level above which source is active

# HDR priority tiers
HDR_PRIORITY_CRITICAL: Final[int] = 100  # Always audible (dialogue, alerts)
HDR_PRIORITY_HIGH: Final[int] = 75       # Usually audible (important SFX)
HDR_PRIORITY_NORMAL: Final[int] = 50     # Standard sounds
HDR_PRIORITY_LOW: Final[int] = 25        # Ambient, background

# =============================================================================
# Tick Pipeline
# =============================================================================

MIXER_BUFFER_SIZE: Final[int] = 512
MIXER_NUM_CHANNELS: Final[int] = 2
LOUDNESS_ANALYSIS_SMOOTHING: Final[float] = 0.1

# AudioCategory enum name to bus name mapping
CATEGORY_TO_BUS: Final[dict[str, str]] = {
    "MASTER": "master",
    "SFX": "sfx",
    "MUSIC": "music",
    "VOICE_OVER": "vo",
    "AMBIENT": "ambient",
    "UI": "ui",
}

# =============================================================================
# Category Names
# =============================================================================

CATEGORY_MASTER: Final[str] = "master"
CATEGORY_SFX: Final[str] = "sfx"
CATEGORY_MUSIC: Final[str] = "music"
CATEGORY_VO: Final[str] = "vo"
CATEGORY_AMBIENT: Final[str] = "ambient"
CATEGORY_UI: Final[str] = "ui"

# Default bus categories (used for hierarchy)
DEFAULT_CATEGORIES: Final[tuple[str, ...]] = (
    CATEGORY_MASTER,
    CATEGORY_SFX,
    CATEGORY_MUSIC,
    CATEGORY_VO,
    CATEGORY_AMBIENT,
    CATEGORY_UI,
)

# Sub-category mappings
SUBCATEGORIES: Final[dict[str, tuple[str, ...]]] = {
    CATEGORY_SFX: ("footsteps", "weapons", "impacts", "foley"),
    CATEGORY_MUSIC: ("combat", "exploration", "menu", "stingers"),
    CATEGORY_VO: ("dialogue", "barks", "narration"),
    CATEGORY_AMBIENT: ("wind", "water", "wildlife", "machinery"),
    CATEGORY_UI: ("buttons", "notifications", "feedback"),
}

# =============================================================================
# Routing
# =============================================================================

MAX_AUX_SENDS: Final[int] = 8
DEFAULT_SEND_LEVEL: Final[float] = 0.0  # dB
MAX_SEND_LEVEL: Final[float] = 6.0      # dB

# =============================================================================
# Thread Safety
# =============================================================================

LOCK_TIMEOUT: Final[float] = 0.1  # seconds
MAX_PENDING_OPERATIONS: Final[int] = 1000

# =============================================================================
# Sample Rate / Buffer
# =============================================================================

DEFAULT_SAMPLE_RATE: Final[int] = 48000
DEFAULT_BUFFER_SIZE: Final[int] = 512
MAX_CHANNELS: Final[int] = 8  # Up to 7.1 surround

# =============================================================================
# Utility Functions
# =============================================================================

import math


def db_to_linear(db: float) -> float:
    """Convert decibels to linear amplitude (0.0 to ~3.98 for -80 to +12 dB)."""
    if db <= DB_SILENCE_THRESHOLD:
        return 0.0
    return 10.0 ** (db / 20.0)


def linear_to_db(linear: float) -> float:
    """Convert linear amplitude to decibels."""
    if linear <= LINEAR_SILENCE_THRESHOLD:
        return MIN_VOLUME_DB
    return 20.0 * math.log10(linear)


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max."""
    return max(min_val, min(max_val, value))


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b by t (0.0 to 1.0)."""
    return a + (b - a) * clamp(t, 0.0, 1.0)


def ease_in(t: float) -> float:
    """Quadratic ease-in curve."""
    return t * t


def ease_out(t: float) -> float:
    """Quadratic ease-out curve."""
    return 1.0 - (1.0 - t) * (1.0 - t)


def ease_in_out(t: float) -> float:
    """Quadratic ease-in-out curve."""
    if t < 0.5:
        return 2.0 * t * t
    return 1.0 - (-2.0 * t + 2.0) ** 2 / 2.0


def apply_curve(t: float, curve: InterpolationCurve) -> float:
    """Apply an interpolation curve to a normalized time value."""
    t = clamp(t, 0.0, 1.0)
    if curve == InterpolationCurve.LINEAR:
        return t
    elif curve == InterpolationCurve.EASE_IN:
        return ease_in(t)
    elif curve == InterpolationCurve.EASE_OUT:
        return ease_out(t)
    elif curve == InterpolationCurve.EASE_IN_OUT:
        return ease_in_out(t)
    elif curve == InterpolationCurve.EXPONENTIAL:
        return t ** 2.5
    elif curve == InterpolationCurve.LOGARITHMIC:
        return math.log1p(t * (math.e - 1)) / math.log(math.e)
    return t
