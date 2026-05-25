"""
Music system configuration constants.

All magic numbers for the adaptive music system are centralized here.
"""

# =============================================================================
# Playback Constants
# =============================================================================

DEFAULT_BPM: float = 120.0
MIN_BPM: float = 30.0
MAX_BPM: float = 300.0
DEFAULT_TIME_SIGNATURE: tuple[int, int] = (4, 4)  # (beats_per_bar, beat_unit)
DEFAULT_VOLUME: float = 1.0
MIN_VOLUME: float = 0.0
MAX_VOLUME: float = 1.0

# =============================================================================
# Transition Constants
# =============================================================================

CROSSFADE_DEFAULT_DURATION: float = 2.0  # seconds
CROSSFADE_MIN_DURATION: float = 0.1  # seconds
CROSSFADE_MAX_DURATION: float = 10.0  # seconds
BEAT_SYNC_TOLERANCE_MS: float = 50.0  # milliseconds
BAR_LOOKAHEAD_BEATS: int = 4  # how many beats ahead to start preparing transition
STINGER_OVERLAP_BEATS: int = 2  # beats of overlap for stinger transitions
TRANSITION_QUEUE_SIZE: int = 8  # max queued transitions

# =============================================================================
# Stem/Layer Constants
# =============================================================================

MAX_STEMS: int = 8
STEM_FADE_TIME: float = 1.0  # seconds
STEM_VOLUME_SMOOTHING: float = 0.1  # smoothing factor (0-1)
STEM_CROSSFADE_CURVE: str = "equal_power"  # "linear", "equal_power", "s_curve"

# Layer names
LAYER_DRUMS: str = "drums"
LAYER_BASS: str = "bass"
LAYER_MELODY: str = "melody"
LAYER_PADS: str = "pads"
LAYER_STRINGS: str = "strings"
LAYER_PERCUSSION: str = "percussion"
LAYER_VOCALS: str = "vocals"
LAYER_FX: str = "fx"

DEFAULT_LAYERS: tuple[str, ...] = (
    LAYER_DRUMS,
    LAYER_BASS,
    LAYER_MELODY,
    LAYER_PADS,
)

# =============================================================================
# Adaptive Music Constants
# =============================================================================

INTENSITY_SMOOTHING: float = 0.5  # smoothing factor for intensity changes
INTENSITY_SMOOTHING_RATE: float = 10.0  # multiplier for intensity smoothing calculations
INTENSITY_MIN: float = 0.0
INTENSITY_MAX: float = 1.0

DANGER_MIN: float = 0.0
DANGER_MAX: float = 1.0
DANGER_THRESHOLD_LOW: float = 0.3
DANGER_THRESHOLD_HIGH: float = 0.7

STATE_TRANSITION_TIME: float = 2.0  # seconds to transition between states

# Vertical remixing thresholds (intensity levels that trigger layer changes)
VERTICAL_THRESHOLD_LOW: float = 0.25
VERTICAL_THRESHOLD_MED: float = 0.50
VERTICAL_THRESHOLD_HIGH: float = 0.75

# Horizontal branching
HORIZONTAL_BRANCH_PROBABILITY: float = 0.3
HORIZONTAL_MIN_SECTION_LENGTH: int = 4  # bars
HORIZONTAL_MAX_SECTION_LENGTH: int = 16  # bars

# =============================================================================
# Music State Constants
# =============================================================================

STATE_EXPLORATION: str = "exploration"
STATE_COMBAT: str = "combat"
STATE_STEALTH: str = "stealth"
STATE_VICTORY: str = "victory"
STATE_DEFEAT: str = "defeat"
STATE_BOSS: str = "boss"
STATE_MENU: str = "menu"
STATE_CUTSCENE: str = "cutscene"
STATE_AMBIENT: str = "ambient"
STATE_TENSION: str = "tension"

DEFAULT_STATE: str = STATE_EXPLORATION

VALID_STATES: frozenset[str] = frozenset({
    STATE_EXPLORATION,
    STATE_COMBAT,
    STATE_STEALTH,
    STATE_VICTORY,
    STATE_DEFEAT,
    STATE_BOSS,
    STATE_MENU,
    STATE_CUTSCENE,
    STATE_AMBIENT,
    STATE_TENSION,
})

# State priority (higher = takes precedence)
STATE_PRIORITY: dict[str, int] = {
    STATE_MENU: 0,
    STATE_AMBIENT: 1,
    STATE_EXPLORATION: 2,
    STATE_CUTSCENE: 3,
    STATE_STEALTH: 4,
    STATE_TENSION: 5,
    STATE_COMBAT: 6,
    STATE_BOSS: 7,
    STATE_VICTORY: 8,
    STATE_DEFEAT: 8,
}

# =============================================================================
# Timing Constants
# =============================================================================

BEAT_CALLBACK_PRECISION_MS: float = 5.0  # target precision for beat callbacks
CALLBACK_LOOKAHEAD_MS: float = 100.0  # how far ahead to look for callbacks
GRID_SUBDIVISIONS: int = 16  # subdivisions per beat (16 = 16th notes)
SYNC_POINT_TOLERANCE_MS: float = 10.0  # tolerance for sync point matching

# Update loop intervals (milliseconds)
UPDATE_INTERVAL_DEFAULT_MS: float = 10.0  # default update loop interval
UPDATE_INTERVAL_CALLBACK_MS: float = 5.0  # callback manager update interval
UPDATE_INTERVAL_ADAPTIVE_MS: float = 16.0  # adaptive music (~60fps)
UPDATE_INTERVAL_STATE_MS: float = 50.0  # state manager update interval

# Common time signatures
TIME_SIGNATURE_4_4: tuple[int, int] = (4, 4)
TIME_SIGNATURE_3_4: tuple[int, int] = (3, 4)
TIME_SIGNATURE_6_8: tuple[int, int] = (6, 8)
TIME_SIGNATURE_2_4: tuple[int, int] = (2, 4)
TIME_SIGNATURE_5_4: tuple[int, int] = (5, 4)
TIME_SIGNATURE_7_8: tuple[int, int] = (7, 8)

# =============================================================================
# Interactive Parameter Constants
# =============================================================================

PARAM_INTENSITY: str = "intensity"
PARAM_DANGER: str = "danger"
PARAM_EMOTION: str = "emotion"
PARAM_LOCATION: str = "location"
PARAM_TENSION: str = "tension"
PARAM_ENERGY: str = "energy"

DEFAULT_PARAMETERS: dict[str, float] = {
    PARAM_INTENSITY: 0.5,
    PARAM_DANGER: 0.0,
    PARAM_TENSION: 0.0,
    PARAM_ENERGY: 0.5,
}

# =============================================================================
# Emotion Constants
# =============================================================================

EMOTION_NEUTRAL: str = "neutral"
EMOTION_HAPPY: str = "happy"
EMOTION_SAD: str = "sad"
EMOTION_ANGRY: str = "angry"
EMOTION_FEARFUL: str = "fearful"
EMOTION_EXCITED: str = "excited"
EMOTION_PEACEFUL: str = "peaceful"
EMOTION_MYSTERIOUS: str = "mysterious"

VALID_EMOTIONS: frozenset[str] = frozenset({
    EMOTION_NEUTRAL,
    EMOTION_HAPPY,
    EMOTION_SAD,
    EMOTION_ANGRY,
    EMOTION_FEARFUL,
    EMOTION_EXCITED,
    EMOTION_PEACEFUL,
    EMOTION_MYSTERIOUS,
})

# =============================================================================
# Stinger Constants
# =============================================================================

STINGER_MAX_DURATION: float = 5.0  # seconds
STINGER_MIN_DURATION: float = 0.1  # seconds
STINGER_DEFAULT_VOLUME: float = 1.0
STINGER_FADE_OUT_TIME: float = 0.5  # seconds

# Stinger types
STINGER_TYPE_IMPACT: str = "impact"
STINGER_TYPE_TRANSITION: str = "transition"
STINGER_TYPE_ACCENT: str = "accent"
STINGER_TYPE_TAIL: str = "tail"

VALID_STINGER_TYPES: frozenset[str] = frozenset({
    STINGER_TYPE_IMPACT,
    STINGER_TYPE_TRANSITION,
    STINGER_TYPE_ACCENT,
    STINGER_TYPE_TAIL,
})

# =============================================================================
# Playback Mode Constants
# =============================================================================

PLAYBACK_MODE_LINEAR: str = "linear"
PLAYBACK_MODE_LOOP: str = "loop"
PLAYBACK_MODE_SHUFFLE: str = "shuffle"
PLAYBACK_MODE_ADAPTIVE: str = "adaptive"

VALID_PLAYBACK_MODES: frozenset[str] = frozenset({
    PLAYBACK_MODE_LINEAR,
    PLAYBACK_MODE_LOOP,
    PLAYBACK_MODE_SHUFFLE,
    PLAYBACK_MODE_ADAPTIVE,
})

# =============================================================================
# Transition Type Constants
# =============================================================================

TRANSITION_CROSSFADE: str = "crossfade"
TRANSITION_BEAT_SYNC: str = "beat_sync"
TRANSITION_BAR_SYNC: str = "bar_sync"
TRANSITION_STINGER: str = "stinger"
TRANSITION_IMMEDIATE: str = "immediate"
TRANSITION_EXIT_CUE: str = "exit_cue"

VALID_TRANSITION_TYPES: frozenset[str] = frozenset({
    TRANSITION_CROSSFADE,
    TRANSITION_BEAT_SYNC,
    TRANSITION_BAR_SYNC,
    TRANSITION_STINGER,
    TRANSITION_IMMEDIATE,
    TRANSITION_EXIT_CUE,
})

# =============================================================================
# Fade Curve Constants
# =============================================================================

FADE_CURVE_LINEAR: str = "linear"
FADE_CURVE_EQUAL_POWER: str = "equal_power"
FADE_CURVE_S_CURVE: str = "s_curve"
FADE_CURVE_EXPONENTIAL: str = "exponential"
FADE_CURVE_LOGARITHMIC: str = "logarithmic"

# Exponential curve steepness factor (higher = steeper curve)
EXPONENTIAL_CURVE_FACTOR: float = 3.0

VALID_FADE_CURVES: frozenset[str] = frozenset({
    FADE_CURVE_LINEAR,
    FADE_CURVE_EQUAL_POWER,
    FADE_CURVE_S_CURVE,
    FADE_CURVE_EXPONENTIAL,
    FADE_CURVE_LOGARITHMIC,
})

# =============================================================================
# Callback Event Types
# =============================================================================

CALLBACK_BEAT: str = "beat"
CALLBACK_BAR: str = "bar"
CALLBACK_MARKER: str = "marker"
CALLBACK_TRACK_END: str = "track_end"
CALLBACK_LOOP_START: str = "loop_start"
CALLBACK_LOOP_END: str = "loop_end"
CALLBACK_SYNC_POINT: str = "sync_point"
CALLBACK_STATE_CHANGE: str = "state_change"

VALID_CALLBACK_TYPES: frozenset[str] = frozenset({
    CALLBACK_BEAT,
    CALLBACK_BAR,
    CALLBACK_MARKER,
    CALLBACK_TRACK_END,
    CALLBACK_LOOP_START,
    CALLBACK_LOOP_END,
    CALLBACK_SYNC_POINT,
    CALLBACK_STATE_CHANGE,
})
