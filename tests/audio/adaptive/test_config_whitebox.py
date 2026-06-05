"""
Whitebox tests for config.py - Configuration constants.
"""

import pytest
from engine.audio.adaptive.config import (
    # Playback constants
    DEFAULT_BPM,
    MIN_BPM,
    MAX_BPM,
    DEFAULT_TIME_SIGNATURE,
    DEFAULT_VOLUME,
    MIN_VOLUME,
    MAX_VOLUME,
    # Transition constants
    CROSSFADE_DEFAULT_DURATION,
    CROSSFADE_MIN_DURATION,
    CROSSFADE_MAX_DURATION,
    BEAT_SYNC_TOLERANCE_MS,
    BAR_LOOKAHEAD_BEATS,
    STINGER_OVERLAP_BEATS,
    TRANSITION_QUEUE_SIZE,
    # Stem/Layer constants
    MAX_STEMS,
    STEM_FADE_TIME,
    STEM_VOLUME_SMOOTHING,
    STEM_CROSSFADE_CURVE,
    LAYER_DRUMS,
    LAYER_BASS,
    LAYER_MELODY,
    LAYER_PADS,
    LAYER_STRINGS,
    LAYER_PERCUSSION,
    LAYER_VOCALS,
    LAYER_FX,
    DEFAULT_LAYERS,
    # Adaptive music constants
    INTENSITY_SMOOTHING,
    INTENSITY_MIN,
    INTENSITY_MAX,
    DANGER_MIN,
    DANGER_MAX,
    DANGER_THRESHOLD_LOW,
    DANGER_THRESHOLD_HIGH,
    STATE_TRANSITION_TIME,
    VERTICAL_THRESHOLD_LOW,
    VERTICAL_THRESHOLD_MED,
    VERTICAL_THRESHOLD_HIGH,
    HORIZONTAL_BRANCH_PROBABILITY,
    HORIZONTAL_MIN_SECTION_LENGTH,
    HORIZONTAL_MAX_SECTION_LENGTH,
    # Music state constants
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
    DEFAULT_STATE,
    VALID_STATES,
    STATE_PRIORITY,
    # Timing constants
    BEAT_CALLBACK_PRECISION_MS,
    CALLBACK_LOOKAHEAD_MS,
    GRID_SUBDIVISIONS,
    SYNC_POINT_TOLERANCE_MS,
    UPDATE_INTERVAL_DEFAULT_MS,
    UPDATE_INTERVAL_CALLBACK_MS,
    UPDATE_INTERVAL_ADAPTIVE_MS,
    UPDATE_INTERVAL_STATE_MS,
    TIME_SIGNATURE_4_4,
    TIME_SIGNATURE_3_4,
    TIME_SIGNATURE_6_8,
    # Parameter constants
    PARAM_INTENSITY,
    PARAM_DANGER,
    PARAM_EMOTION,
    PARAM_LOCATION,
    PARAM_TENSION,
    PARAM_ENERGY,
    DEFAULT_PARAMETERS,
    # Emotion constants
    EMOTION_NEUTRAL,
    EMOTION_HAPPY,
    EMOTION_SAD,
    EMOTION_ANGRY,
    EMOTION_FEARFUL,
    EMOTION_EXCITED,
    EMOTION_PEACEFUL,
    EMOTION_MYSTERIOUS,
    VALID_EMOTIONS,
    # Stinger constants
    STINGER_MAX_DURATION,
    STINGER_MIN_DURATION,
    STINGER_DEFAULT_VOLUME,
    STINGER_FADE_OUT_TIME,
    STINGER_TYPE_IMPACT,
    STINGER_TYPE_TRANSITION,
    STINGER_TYPE_ACCENT,
    STINGER_TYPE_TAIL,
    VALID_STINGER_TYPES,
    # Playback mode constants
    PLAYBACK_MODE_LINEAR,
    PLAYBACK_MODE_LOOP,
    PLAYBACK_MODE_SHUFFLE,
    PLAYBACK_MODE_ADAPTIVE,
    VALID_PLAYBACK_MODES,
    # Transition type constants
    TRANSITION_CROSSFADE,
    TRANSITION_BEAT_SYNC,
    TRANSITION_BAR_SYNC,
    TRANSITION_STINGER,
    TRANSITION_IMMEDIATE,
    TRANSITION_EXIT_CUE,
    VALID_TRANSITION_TYPES,
    # Fade curve constants
    FADE_CURVE_LINEAR,
    FADE_CURVE_EQUAL_POWER,
    FADE_CURVE_S_CURVE,
    FADE_CURVE_EXPONENTIAL,
    FADE_CURVE_LOGARITHMIC,
    EXPONENTIAL_CURVE_FACTOR,
    VALID_FADE_CURVES,
    # Callback event types
    CALLBACK_BEAT,
    CALLBACK_BAR,
    CALLBACK_MARKER,
    CALLBACK_TRACK_END,
    CALLBACK_LOOP_START,
    CALLBACK_LOOP_END,
    CALLBACK_SYNC_POINT,
    CALLBACK_STATE_CHANGE,
    VALID_CALLBACK_TYPES,
)


class TestPlaybackConstants:
    """Tests for playback configuration constants."""

    def test_default_bpm_reasonable_value(self):
        """DEFAULT_BPM should be a standard tempo."""
        assert DEFAULT_BPM == 120.0

    def test_bpm_range_valid(self):
        """BPM range should be musically reasonable."""
        assert MIN_BPM == 30.0
        assert MAX_BPM == 300.0
        assert MIN_BPM < DEFAULT_BPM < MAX_BPM

    def test_default_time_signature(self):
        """DEFAULT_TIME_SIGNATURE should be 4/4."""
        assert DEFAULT_TIME_SIGNATURE == (4, 4)

    def test_volume_range(self):
        """Volume range should be 0.0-1.0."""
        assert MIN_VOLUME == 0.0
        assert MAX_VOLUME == 1.0
        assert MIN_VOLUME < DEFAULT_VOLUME <= MAX_VOLUME


class TestTransitionConstants:
    """Tests for transition configuration constants."""

    def test_crossfade_duration_valid(self):
        """Crossfade duration should be reasonable."""
        assert CROSSFADE_MIN_DURATION > 0
        assert CROSSFADE_DEFAULT_DURATION >= CROSSFADE_MIN_DURATION
        assert CROSSFADE_MAX_DURATION >= CROSSFADE_DEFAULT_DURATION

    def test_beat_sync_tolerance(self):
        """Beat sync tolerance should be tight for precision."""
        assert BEAT_SYNC_TOLERANCE_MS == 50.0
        assert BEAT_SYNC_TOLERANCE_MS > 0

    def test_bar_lookahead_beats(self):
        """Bar lookahead should be one bar or less."""
        assert BAR_LOOKAHEAD_BEATS == 4

    def test_stinger_overlap_beats(self):
        """Stinger overlap should be a small number of beats."""
        assert STINGER_OVERLAP_BEATS == 2

    def test_transition_queue_size(self):
        """Transition queue should have reasonable capacity."""
        assert TRANSITION_QUEUE_SIZE == 8
        assert TRANSITION_QUEUE_SIZE > 0


class TestStemLayerConstants:
    """Tests for stem/layer configuration constants."""

    def test_max_stems(self):
        """MAX_STEMS should support 8 stem types."""
        assert MAX_STEMS == 8

    def test_stem_fade_time(self):
        """Stem fade time should be positive."""
        assert STEM_FADE_TIME == 1.0
        assert STEM_FADE_TIME > 0

    def test_stem_volume_smoothing(self):
        """Stem volume smoothing should be in valid range."""
        assert 0 < STEM_VOLUME_SMOOTHING <= 1

    def test_stem_crossfade_curve_valid(self):
        """Default stem crossfade curve should be valid."""
        assert STEM_CROSSFADE_CURVE in VALID_FADE_CURVES

    def test_layer_names_defined(self):
        """All 8 layer types should be defined."""
        assert LAYER_DRUMS == "drums"
        assert LAYER_BASS == "bass"
        assert LAYER_MELODY == "melody"
        assert LAYER_PADS == "pads"
        assert LAYER_STRINGS == "strings"
        assert LAYER_PERCUSSION == "percussion"
        assert LAYER_VOCALS == "vocals"
        assert LAYER_FX == "fx"

    def test_default_layers_subset(self):
        """DEFAULT_LAYERS should contain the core layers."""
        assert LAYER_DRUMS in DEFAULT_LAYERS
        assert LAYER_BASS in DEFAULT_LAYERS
        assert LAYER_MELODY in DEFAULT_LAYERS
        assert LAYER_PADS in DEFAULT_LAYERS


class TestAdaptiveMusicConstants:
    """Tests for adaptive music configuration constants."""

    def test_intensity_range(self):
        """Intensity range should be 0.0-1.0."""
        assert INTENSITY_MIN == 0.0
        assert INTENSITY_MAX == 1.0

    def test_danger_range(self):
        """Danger range should be 0.0-1.0."""
        assert DANGER_MIN == 0.0
        assert DANGER_MAX == 1.0

    def test_danger_thresholds_ordered(self):
        """Danger thresholds should be in increasing order."""
        assert DANGER_THRESHOLD_LOW < DANGER_THRESHOLD_HIGH
        assert 0 <= DANGER_THRESHOLD_LOW <= 1
        assert 0 <= DANGER_THRESHOLD_HIGH <= 1

    def test_vertical_thresholds_ordered(self):
        """Vertical thresholds should be in increasing order."""
        assert VERTICAL_THRESHOLD_LOW < VERTICAL_THRESHOLD_MED < VERTICAL_THRESHOLD_HIGH
        assert 0 <= VERTICAL_THRESHOLD_LOW <= 1
        assert 0 <= VERTICAL_THRESHOLD_HIGH <= 1

    def test_horizontal_section_length_valid(self):
        """Horizontal section length should be reasonable."""
        assert HORIZONTAL_MIN_SECTION_LENGTH < HORIZONTAL_MAX_SECTION_LENGTH
        assert HORIZONTAL_MIN_SECTION_LENGTH >= 1


class TestMusicStateConstants:
    """Tests for music state configuration constants."""

    def test_all_states_in_valid_states(self):
        """All individual state constants should be in VALID_STATES."""
        assert STATE_EXPLORATION in VALID_STATES
        assert STATE_COMBAT in VALID_STATES
        assert STATE_STEALTH in VALID_STATES
        assert STATE_VICTORY in VALID_STATES
        assert STATE_DEFEAT in VALID_STATES
        assert STATE_BOSS in VALID_STATES
        assert STATE_MENU in VALID_STATES
        assert STATE_CUTSCENE in VALID_STATES
        assert STATE_AMBIENT in VALID_STATES
        assert STATE_TENSION in VALID_STATES

    def test_valid_states_is_frozen(self):
        """VALID_STATES should be a frozenset (immutable)."""
        assert isinstance(VALID_STATES, frozenset)
        assert len(VALID_STATES) == 10

    def test_default_state_valid(self):
        """DEFAULT_STATE should be a valid state."""
        assert DEFAULT_STATE in VALID_STATES
        assert DEFAULT_STATE == STATE_EXPLORATION

    def test_state_priority_covers_all_states(self):
        """STATE_PRIORITY should cover all states."""
        for state in VALID_STATES:
            assert state in STATE_PRIORITY

    def test_state_priority_values_reasonable(self):
        """State priorities should reflect gameplay importance."""
        assert STATE_PRIORITY[STATE_MENU] < STATE_PRIORITY[STATE_EXPLORATION]
        assert STATE_PRIORITY[STATE_COMBAT] > STATE_PRIORITY[STATE_EXPLORATION]
        assert STATE_PRIORITY[STATE_BOSS] >= STATE_PRIORITY[STATE_COMBAT]


class TestTimingConstants:
    """Tests for timing configuration constants."""

    def test_beat_callback_precision(self):
        """Beat callback precision should be tight (5ms target)."""
        assert BEAT_CALLBACK_PRECISION_MS == 5.0
        assert BEAT_CALLBACK_PRECISION_MS > 0

    def test_callback_lookahead(self):
        """Callback lookahead should be reasonable."""
        assert CALLBACK_LOOKAHEAD_MS == 100.0
        assert CALLBACK_LOOKAHEAD_MS > BEAT_CALLBACK_PRECISION_MS

    def test_grid_subdivisions(self):
        """Grid subdivisions should be 16 (16th notes)."""
        assert GRID_SUBDIVISIONS == 16

    def test_update_intervals_positive(self):
        """All update intervals should be positive."""
        assert UPDATE_INTERVAL_DEFAULT_MS > 0
        assert UPDATE_INTERVAL_CALLBACK_MS > 0
        assert UPDATE_INTERVAL_ADAPTIVE_MS > 0
        assert UPDATE_INTERVAL_STATE_MS > 0

    def test_common_time_signatures(self):
        """Common time signatures should be correctly defined."""
        assert TIME_SIGNATURE_4_4 == (4, 4)
        assert TIME_SIGNATURE_3_4 == (3, 4)
        assert TIME_SIGNATURE_6_8 == (6, 8)


class TestParameterConstants:
    """Tests for parameter configuration constants."""

    def test_parameter_names_defined(self):
        """All parameter names should be defined."""
        assert PARAM_INTENSITY == "intensity"
        assert PARAM_DANGER == "danger"
        assert PARAM_EMOTION == "emotion"
        assert PARAM_LOCATION == "location"
        assert PARAM_TENSION == "tension"
        assert PARAM_ENERGY == "energy"

    def test_default_parameters_values(self):
        """Default parameters should have reasonable values."""
        assert PARAM_INTENSITY in DEFAULT_PARAMETERS
        assert 0 <= DEFAULT_PARAMETERS[PARAM_INTENSITY] <= 1


class TestEmotionConstants:
    """Tests for emotion configuration constants."""

    def test_all_emotions_in_valid_emotions(self):
        """All individual emotion constants should be in VALID_EMOTIONS."""
        assert EMOTION_NEUTRAL in VALID_EMOTIONS
        assert EMOTION_HAPPY in VALID_EMOTIONS
        assert EMOTION_SAD in VALID_EMOTIONS
        assert EMOTION_ANGRY in VALID_EMOTIONS
        assert EMOTION_FEARFUL in VALID_EMOTIONS
        assert EMOTION_EXCITED in VALID_EMOTIONS
        assert EMOTION_PEACEFUL in VALID_EMOTIONS
        assert EMOTION_MYSTERIOUS in VALID_EMOTIONS

    def test_valid_emotions_is_frozen(self):
        """VALID_EMOTIONS should be a frozenset (immutable)."""
        assert isinstance(VALID_EMOTIONS, frozenset)


class TestStingerConstants:
    """Tests for stinger configuration constants."""

    def test_stinger_duration_range(self):
        """Stinger duration range should be reasonable."""
        assert STINGER_MIN_DURATION < STINGER_MAX_DURATION
        assert STINGER_MIN_DURATION > 0

    def test_stinger_default_volume(self):
        """Stinger default volume should be full."""
        assert STINGER_DEFAULT_VOLUME == 1.0

    def test_stinger_types_valid(self):
        """All stinger types should be in VALID_STINGER_TYPES."""
        assert STINGER_TYPE_IMPACT in VALID_STINGER_TYPES
        assert STINGER_TYPE_TRANSITION in VALID_STINGER_TYPES
        assert STINGER_TYPE_ACCENT in VALID_STINGER_TYPES
        assert STINGER_TYPE_TAIL in VALID_STINGER_TYPES


class TestPlaybackModeConstants:
    """Tests for playback mode configuration constants."""

    def test_all_modes_in_valid_modes(self):
        """All playback modes should be in VALID_PLAYBACK_MODES."""
        assert PLAYBACK_MODE_LINEAR in VALID_PLAYBACK_MODES
        assert PLAYBACK_MODE_LOOP in VALID_PLAYBACK_MODES
        assert PLAYBACK_MODE_SHUFFLE in VALID_PLAYBACK_MODES
        assert PLAYBACK_MODE_ADAPTIVE in VALID_PLAYBACK_MODES


class TestTransitionTypeConstants:
    """Tests for transition type configuration constants."""

    def test_all_transition_types_valid(self):
        """All transition types should be in VALID_TRANSITION_TYPES."""
        assert TRANSITION_CROSSFADE in VALID_TRANSITION_TYPES
        assert TRANSITION_BEAT_SYNC in VALID_TRANSITION_TYPES
        assert TRANSITION_BAR_SYNC in VALID_TRANSITION_TYPES
        assert TRANSITION_STINGER in VALID_TRANSITION_TYPES
        assert TRANSITION_IMMEDIATE in VALID_TRANSITION_TYPES
        assert TRANSITION_EXIT_CUE in VALID_TRANSITION_TYPES


class TestFadeCurveConstants:
    """Tests for fade curve configuration constants."""

    def test_all_fade_curves_valid(self):
        """All fade curves should be in VALID_FADE_CURVES."""
        assert FADE_CURVE_LINEAR in VALID_FADE_CURVES
        assert FADE_CURVE_EQUAL_POWER in VALID_FADE_CURVES
        assert FADE_CURVE_S_CURVE in VALID_FADE_CURVES
        assert FADE_CURVE_EXPONENTIAL in VALID_FADE_CURVES
        assert FADE_CURVE_LOGARITHMIC in VALID_FADE_CURVES

    def test_exponential_curve_factor(self):
        """Exponential curve factor should be positive."""
        assert EXPONENTIAL_CURVE_FACTOR > 0


class TestCallbackTypeConstants:
    """Tests for callback type configuration constants."""

    def test_all_callback_types_valid(self):
        """All callback types should be in VALID_CALLBACK_TYPES."""
        assert CALLBACK_BEAT in VALID_CALLBACK_TYPES
        assert CALLBACK_BAR in VALID_CALLBACK_TYPES
        assert CALLBACK_MARKER in VALID_CALLBACK_TYPES
        assert CALLBACK_TRACK_END in VALID_CALLBACK_TYPES
        assert CALLBACK_LOOP_START in VALID_CALLBACK_TYPES
        assert CALLBACK_LOOP_END in VALID_CALLBACK_TYPES
        assert CALLBACK_SYNC_POINT in VALID_CALLBACK_TYPES
        assert CALLBACK_STATE_CHANGE in VALID_CALLBACK_TYPES
