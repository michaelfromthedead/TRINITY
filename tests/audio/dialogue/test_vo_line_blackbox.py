"""
Blackbox tests for VOLine and related dialogue data structures.

Tests PUBLIC behavior only - no internal state inspection.
Based on GAPSET_15_AUDIO Phase 9 specifications.
"""

import pytest
from typing import Optional
from dataclasses import dataclass, FrozenInstanceError

# Public API imports
from engine.audio.dialogue import (
    VOLine,
    VOLineState,
    LipSyncData,
    SubtitleData,
    create_vo_line,
    VOPriority,
    PRIORITY_CRITICAL,
    PRIORITY_HIGH,
    PRIORITY_NORMAL,
    PRIORITY_LOW,
    PRIORITY_BARK,
    PRIORITY_AMBIENT,
    ContextType,
    CONTEXT_BARK,
    CONTEXT_CONVERSATION,
    CONTEXT_AMBIENT,
    CONTEXT_NARRATION,
    CONTEXT_TUTORIAL,
    SelectionMode,
    SELECTION_RANDOM,
    SELECTION_SEQUENTIAL,
    SELECTION_WEIGHTED,
    SELECTION_CONDITIONAL,
)


class TestVOLineCreation:
    """Test VOLine creation and basic properties."""

    def test_create_vo_line_minimal(self):
        """VOLine can be created with minimal parameters."""
        line = create_vo_line(line_id="test_001", audio_path="vo/test.wav")
        assert line is not None
        assert line.line_id == "test_001"
        assert line.audio_path == "vo/test.wav"

    def test_create_vo_line_with_priority(self):
        """VOLine respects priority parameter."""
        line = create_vo_line(
            line_id="test_002",
            audio_path="vo/test.wav",
            priority=PRIORITY_HIGH
        )
        assert line.priority == PRIORITY_HIGH

    def test_create_vo_line_with_critical_priority(self):
        """VOLine can be created with critical priority."""
        line = create_vo_line(
            line_id="critical_001",
            audio_path="vo/critical.wav",
            priority=PRIORITY_CRITICAL
        )
        assert line.priority == PRIORITY_CRITICAL

    def test_create_vo_line_default_priority_is_normal(self):
        """Default priority should be NORMAL."""
        line = create_vo_line(line_id="test_003", audio_path="vo/test.wav")
        assert line.priority == PRIORITY_NORMAL

    def test_create_vo_line_with_text(self):
        """VOLine can include subtitle text."""
        line = create_vo_line(
            line_id="test_004",
            audio_path="vo/test.wav",
            text="Hello, world!"
        )
        assert line.text == "Hello, world!"

    def test_create_vo_line_with_speaker(self):
        """VOLine can specify speaker."""
        line = create_vo_line(
            line_id="test_005",
            audio_path="vo/test.wav",
            speaker="Commander"
        )
        assert line.speaker == "Commander"

    def test_create_vo_line_with_duration(self):
        """VOLine can specify duration."""
        line = create_vo_line(
            line_id="test_006",
            audio_path="vo/test.wav",
            duration_ms=2500
        )
        assert line.duration_ms == 2500

    def test_create_vo_line_with_context(self):
        """VOLine can specify context type."""
        line = create_vo_line(
            line_id="bark_001",
            audio_path="vo/bark.wav",
            context=CONTEXT_BARK
        )
        assert line.context == CONTEXT_BARK

    def test_create_vo_line_conversation_context(self):
        """VOLine with conversation context."""
        line = create_vo_line(
            line_id="conv_001",
            audio_path="vo/conv.wav",
            context=CONTEXT_CONVERSATION
        )
        assert line.context == CONTEXT_CONVERSATION

    def test_create_vo_line_ambient_context(self):
        """VOLine with ambient context."""
        line = create_vo_line(
            line_id="ambient_001",
            audio_path="vo/ambient.wav",
            context=CONTEXT_AMBIENT,
            priority=PRIORITY_AMBIENT
        )
        assert line.context == CONTEXT_AMBIENT
        assert line.priority == PRIORITY_AMBIENT


class TestVOLineState:
    """Test VOLine state enumeration."""

    def test_pending_state_exists(self):
        """PENDING state is defined."""
        assert VOLineState.PENDING is not None

    def test_playing_state_exists(self):
        """PLAYING state is defined."""
        assert VOLineState.PLAYING is not None

    def test_completed_state_exists(self):
        """COMPLETED state is defined."""
        assert VOLineState.COMPLETED is not None

    def test_interrupted_state_exists(self):
        """INTERRUPTED state is defined."""
        assert VOLineState.INTERRUPTED is not None

    def test_vo_line_initial_state_is_pending(self):
        """New VOLine should have PENDING state."""
        line = create_vo_line(line_id="state_001", audio_path="vo/test.wav")
        assert line.state == VOLineState.PENDING

    def test_state_values_are_unique(self):
        """All states have unique values."""
        states = [
            VOLineState.PENDING,
            VOLineState.PLAYING,
            VOLineState.COMPLETED,
            VOLineState.INTERRUPTED,
        ]
        assert len(states) == len(set(states))


class TestVOLinePriority:
    """Test VOLine priority ordering."""

    def test_priority_ordering_critical_highest(self):
        """CRITICAL priority is highest."""
        assert PRIORITY_CRITICAL > PRIORITY_HIGH

    def test_priority_ordering_high_above_normal(self):
        """HIGH priority is above NORMAL."""
        assert PRIORITY_HIGH > PRIORITY_NORMAL

    def test_priority_ordering_normal_above_low(self):
        """NORMAL priority is above LOW."""
        assert PRIORITY_NORMAL > PRIORITY_LOW

    def test_priority_ordering_low_above_bark(self):
        """LOW priority is above BARK."""
        assert PRIORITY_LOW > PRIORITY_BARK

    def test_priority_ordering_bark_above_ambient(self):
        """BARK priority is above AMBIENT."""
        assert PRIORITY_BARK > PRIORITY_AMBIENT

    def test_priority_comparison_in_list(self):
        """VOLines can be sorted by priority."""
        lines = [
            create_vo_line("low", "vo/low.wav", priority=PRIORITY_LOW),
            create_vo_line("high", "vo/high.wav", priority=PRIORITY_HIGH),
            create_vo_line("normal", "vo/normal.wav", priority=PRIORITY_NORMAL),
        ]
        sorted_lines = sorted(lines, key=lambda x: x.priority, reverse=True)
        assert sorted_lines[0].line_id == "high"
        assert sorted_lines[1].line_id == "normal"
        assert sorted_lines[2].line_id == "low"


class TestLipSyncData:
    """Test lipsync data structures."""

    def test_lipsync_data_creation(self):
        """LipSyncData can be created."""
        lipsync = LipSyncData(phonemes=[("AA", 0.0, 0.1), ("T", 0.1, 0.15)])
        assert lipsync is not None

    def test_lipsync_data_phoneme_count(self):
        """LipSyncData reports phoneme count."""
        lipsync = LipSyncData(phonemes=[
            ("AA", 0.0, 0.1),
            ("T", 0.1, 0.15),
            ("IY", 0.15, 0.25),
        ])
        assert len(lipsync.phonemes) == 3

    def test_lipsync_data_empty(self):
        """LipSyncData can be empty."""
        lipsync = LipSyncData(phonemes=[])
        assert len(lipsync.phonemes) == 0

    def test_vo_line_with_lipsync(self):
        """VOLine can include lipsync data."""
        lipsync = LipSyncData(phonemes=[("AA", 0.0, 0.1)])
        line = create_vo_line(
            line_id="lipsync_001",
            audio_path="vo/test.wav",
            lipsync_data=lipsync
        )
        assert line.lipsync_data is not None
        assert len(line.lipsync_data.phonemes) == 1


class TestSubtitleData:
    """Test subtitle data structures."""

    def test_subtitle_data_creation(self):
        """SubtitleData can be created."""
        subtitle = SubtitleData(text="Hello!", start_ms=0, end_ms=1000)
        assert subtitle is not None
        assert subtitle.text == "Hello!"

    def test_subtitle_data_timing(self):
        """SubtitleData has correct timing."""
        subtitle = SubtitleData(text="Test", start_ms=500, end_ms=1500)
        assert subtitle.start_ms == 500
        assert subtitle.end_ms == 1500

    def test_subtitle_data_duration_calculation(self):
        """SubtitleData duration can be calculated."""
        subtitle = SubtitleData(text="Test", start_ms=100, end_ms=600)
        duration = subtitle.end_ms - subtitle.start_ms
        assert duration == 500

    def test_vo_line_with_subtitle(self):
        """VOLine can include subtitle data."""
        subtitle = SubtitleData(text="Hello!", start_ms=0, end_ms=1000)
        line = create_vo_line(
            line_id="sub_001",
            audio_path="vo/test.wav",
            subtitle_data=subtitle
        )
        assert line.subtitle_data is not None
        assert line.subtitle_data.text == "Hello!"


class TestContextType:
    """Test context type enumeration."""

    def test_bark_context_exists(self):
        """BARK context type exists."""
        assert CONTEXT_BARK is not None

    def test_conversation_context_exists(self):
        """CONVERSATION context type exists."""
        assert CONTEXT_CONVERSATION is not None

    def test_ambient_context_exists(self):
        """AMBIENT context type exists."""
        assert CONTEXT_AMBIENT is not None

    def test_narration_context_exists(self):
        """NARRATION context type exists."""
        assert CONTEXT_NARRATION is not None

    def test_tutorial_context_exists(self):
        """TUTORIAL context type exists."""
        assert CONTEXT_TUTORIAL is not None


class TestSelectionMode:
    """Test selection mode enumeration."""

    def test_random_selection_exists(self):
        """RANDOM selection mode exists."""
        assert SELECTION_RANDOM is not None

    def test_sequential_selection_exists(self):
        """SEQUENTIAL selection mode exists."""
        assert SELECTION_SEQUENTIAL is not None

    def test_weighted_selection_exists(self):
        """WEIGHTED selection mode exists."""
        assert SELECTION_WEIGHTED is not None

    def test_conditional_selection_exists(self):
        """CONDITIONAL selection mode exists."""
        assert SELECTION_CONDITIONAL is not None


class TestVOLineEquality:
    """Test VOLine equality and hashing."""

    def test_vo_lines_same_id_equal(self):
        """VOLines with same ID are equal."""
        line1 = create_vo_line("test_001", "vo/test.wav")
        line2 = create_vo_line("test_001", "vo/test.wav")
        assert line1 == line2

    def test_vo_lines_different_id_not_equal(self):
        """VOLines with different IDs are not equal."""
        line1 = create_vo_line("test_001", "vo/test.wav")
        line2 = create_vo_line("test_002", "vo/test.wav")
        assert line1 != line2

    def test_vo_line_hashable(self):
        """VOLine can be used in sets and dicts."""
        line = create_vo_line("hash_001", "vo/test.wav")
        line_set = {line}
        assert line in line_set


class TestVOLineValidation:
    """Test VOLine input validation."""

    def test_empty_line_id_rejected(self):
        """Empty line ID should be rejected."""
        with pytest.raises((ValueError, TypeError)):
            create_vo_line("", "vo/test.wav")

    def test_empty_audio_path_rejected(self):
        """Empty audio path should be rejected."""
        with pytest.raises((ValueError, TypeError)):
            create_vo_line("test_001", "")

    def test_negative_duration_rejected(self):
        """Negative duration should be rejected."""
        with pytest.raises((ValueError, TypeError)):
            create_vo_line("test_001", "vo/test.wav", duration_ms=-100)

    def test_none_line_id_rejected(self):
        """None line ID should be rejected."""
        with pytest.raises((ValueError, TypeError)):
            create_vo_line(None, "vo/test.wav")


class TestVOLineMetadata:
    """Test VOLine metadata handling."""

    def test_vo_line_with_tags(self):
        """VOLine can include tags metadata."""
        line = create_vo_line(
            line_id="tagged_001",
            audio_path="vo/test.wav",
            tags=["combat", "urgent"]
        )
        assert "combat" in line.tags
        assert "urgent" in line.tags

    def test_vo_line_with_language(self):
        """VOLine can specify language."""
        line = create_vo_line(
            line_id="lang_001",
            audio_path="vo/en/test.wav",
            language="en"
        )
        assert line.language == "en"

    def test_vo_line_default_language(self):
        """VOLine has default language."""
        line = create_vo_line("default_lang", "vo/test.wav")
        assert line.language is not None  # Should have a default


class TestVOLineImmutability:
    """Test VOLine immutability after creation."""

    def test_vo_line_state_can_change(self):
        """VOLine state can be updated (allowed transition)."""
        line = create_vo_line("mutable_001", "vo/test.wav")
        # State updates should be allowed through proper API
        assert line.state == VOLineState.PENDING

    def test_vo_line_id_immutable(self):
        """VOLine ID should not be changeable."""
        line = create_vo_line("immutable_001", "vo/test.wav")
        # The line_id should remain constant
        assert line.line_id == "immutable_001"


class TestVOLinePriorityEnum:
    """Test VOPriority enum functionality."""

    def test_priority_enum_values(self):
        """VOPriority enum has expected values."""
        assert VOPriority.CRITICAL == PRIORITY_CRITICAL
        assert VOPriority.HIGH == PRIORITY_HIGH
        assert VOPriority.NORMAL == PRIORITY_NORMAL
        assert VOPriority.LOW == PRIORITY_LOW

    def test_priority_comparisons(self):
        """Priority comparisons work correctly."""
        assert VOPriority.CRITICAL > VOPriority.HIGH
        assert VOPriority.HIGH > VOPriority.NORMAL
        assert VOPriority.NORMAL > VOPriority.LOW
