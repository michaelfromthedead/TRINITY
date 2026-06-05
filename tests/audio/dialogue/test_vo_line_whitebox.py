"""
Whitebox tests for VOLine module.

Tests VOLine dataclass, LipSyncData, SubtitleData, playback states,
callbacks, serialization, and factory functions.
"""

import pytest
import uuid
from unittest.mock import MagicMock, patch

from engine.audio.dialogue.vo_line import (
    VOLine,
    VOLineState,
    LipSyncData,
    SubtitleData,
    create_vo_line,
)
from engine.audio.dialogue.config import (
    PRIORITY_NORMAL,
    PRIORITY_HIGH,
    PRIORITY_CRITICAL,
    VOPriority,
    ContextType,
)


# =============================================================================
# LipSyncData Tests
# =============================================================================


class TestLipSyncData:
    """Tests for LipSyncData dataclass."""

    def test_default_initialization(self):
        """Test LipSyncData initializes with empty lists."""
        lip_sync = LipSyncData()
        assert lip_sync.phonemes == []
        assert lip_sync.visemes == []
        assert lip_sync.blend_shapes == {}

    def test_phoneme_initialization(self):
        """Test phoneme data initialization."""
        phonemes = [(0.0, "AA"), (100.0, "B"), (200.0, "M")]
        lip_sync = LipSyncData(phonemes=phonemes)
        assert len(lip_sync.phonemes) == 3
        assert lip_sync.phonemes[0] == (0.0, "AA")

    def test_viseme_initialization(self):
        """Test viseme data initialization."""
        visemes = [(0.0, 1), (50.0, 2), (100.0, 3)]
        lip_sync = LipSyncData(visemes=visemes)
        assert len(lip_sync.visemes) == 3
        assert lip_sync.visemes[1] == (50.0, 2)

    def test_get_phoneme_at_time_middle(self):
        """Test getting phoneme at middle of timeline."""
        phonemes = [(0.0, "AA"), (100.0, "B"), (200.0, "M")]
        lip_sync = LipSyncData(phonemes=phonemes)

        assert lip_sync.get_phoneme_at(50.0) == "AA"
        assert lip_sync.get_phoneme_at(100.0) == "B"
        assert lip_sync.get_phoneme_at(150.0) == "B"

    def test_get_phoneme_at_time_last(self):
        """Test getting phoneme at end of timeline returns last."""
        phonemes = [(0.0, "AA"), (100.0, "B")]
        lip_sync = LipSyncData(phonemes=phonemes)

        # After last phoneme time, should return last phoneme
        assert lip_sync.get_phoneme_at(150.0) == "B"
        assert lip_sync.get_phoneme_at(1000.0) == "B"

    def test_get_phoneme_at_time_before_first(self):
        """Test getting phoneme before first timestamp."""
        phonemes = [(100.0, "AA"), (200.0, "B")]
        lip_sync = LipSyncData(phonemes=phonemes)

        # Before first timestamp returns None
        assert lip_sync.get_phoneme_at(50.0) is None

    def test_get_phoneme_empty_list(self):
        """Test getting phoneme from empty list."""
        lip_sync = LipSyncData()
        assert lip_sync.get_phoneme_at(100.0) is None

    def test_get_viseme_at_time_middle(self):
        """Test getting viseme at middle of timeline."""
        visemes = [(0.0, 1), (100.0, 2), (200.0, 3)]
        lip_sync = LipSyncData(visemes=visemes)

        assert lip_sync.get_viseme_at(50.0) == 1
        assert lip_sync.get_viseme_at(100.0) == 2
        assert lip_sync.get_viseme_at(150.0) == 2

    def test_get_viseme_at_time_last(self):
        """Test getting viseme at end returns last."""
        visemes = [(0.0, 1), (100.0, 2)]
        lip_sync = LipSyncData(visemes=visemes)

        assert lip_sync.get_viseme_at(150.0) == 2
        assert lip_sync.get_viseme_at(1000.0) == 2

    def test_get_viseme_empty_list(self):
        """Test getting viseme from empty list."""
        lip_sync = LipSyncData()
        assert lip_sync.get_viseme_at(100.0) is None

    def test_blend_shapes_initialization(self):
        """Test blend shapes dictionary initialization."""
        blend_shapes = {
            "mouth_open": [(0.0, 0.0), (50.0, 1.0), (100.0, 0.0)],
            "smile": [(0.0, 0.5), (100.0, 0.8)],
        }
        lip_sync = LipSyncData(blend_shapes=blend_shapes)

        assert len(lip_sync.blend_shapes) == 2
        assert "mouth_open" in lip_sync.blend_shapes
        assert len(lip_sync.blend_shapes["mouth_open"]) == 3


# =============================================================================
# SubtitleData Tests
# =============================================================================


class TestSubtitleData:
    """Tests for SubtitleData dataclass."""

    def test_default_initialization(self):
        """Test SubtitleData initializes with defaults."""
        subtitle = SubtitleData(text="Hello World")

        assert subtitle.text == "Hello World"
        assert subtitle.speaker_name == ""
        assert subtitle.speaker_color == "#FFFFFF"
        assert subtitle.start_time_ms == 0.0
        assert subtitle.end_time_ms == 0.0
        assert subtitle.position == (0.5, 0.9)
        assert subtitle.alignment == "center"
        assert subtitle.font_size == 24

    def test_full_initialization(self):
        """Test SubtitleData with all fields."""
        subtitle = SubtitleData(
            text="Test Line",
            speaker_name="Character",
            speaker_color="#FF0000",
            start_time_ms=100.0,
            end_time_ms=2000.0,
            position=(0.1, 0.8),
            alignment="left",
            font_size=32,
        )

        assert subtitle.speaker_name == "Character"
        assert subtitle.speaker_color == "#FF0000"
        assert subtitle.position == (0.1, 0.8)
        assert subtitle.font_size == 32

    def test_duration_property(self):
        """Test duration_ms property calculation."""
        subtitle = SubtitleData(
            text="Test",
            start_time_ms=100.0,
            end_time_ms=2100.0,
        )

        assert subtitle.duration_ms == 2000.0

    def test_duration_zero(self):
        """Test duration when start equals end."""
        subtitle = SubtitleData(
            text="Test",
            start_time_ms=100.0,
            end_time_ms=100.0,
        )

        assert subtitle.duration_ms == 0.0

    def test_duration_negative(self):
        """Test duration when end is before start."""
        subtitle = SubtitleData(
            text="Test",
            start_time_ms=200.0,
            end_time_ms=100.0,
        )

        # Duration can be negative in edge case
        assert subtitle.duration_ms == -100.0


# =============================================================================
# VOLine State Tests
# =============================================================================


class TestVOLineState:
    """Tests for VOLineState enum."""

    def test_all_states_exist(self):
        """Test all required states are defined."""
        assert VOLineState.PENDING.value == "pending"
        assert VOLineState.LOADING.value == "loading"
        assert VOLineState.READY.value == "ready"
        assert VOLineState.PLAYING.value == "playing"
        assert VOLineState.PAUSED.value == "paused"
        assert VOLineState.COMPLETED.value == "completed"
        assert VOLineState.INTERRUPTED.value == "interrupted"
        assert VOLineState.FAILED.value == "failed"

    def test_state_count(self):
        """Test correct number of states."""
        assert len(VOLineState) == 8


# =============================================================================
# VOLine Basic Tests
# =============================================================================


class TestVOLineBasic:
    """Basic tests for VOLine dataclass."""

    def test_default_initialization(self):
        """Test VOLine initializes with defaults."""
        line = VOLine()

        assert line.line_id is not None
        assert line.audio_asset == ""
        assert line.text == ""
        assert line.speaker_id == ""
        assert line.duration_ms == 0.0
        assert line.priority == PRIORITY_NORMAL
        assert line.interruptible is True
        assert line.context_type == ContextType.BARK.value
        assert isinstance(line.tags, set)
        assert line.language == "en"
        assert line.weight == 1.0
        assert line.cooldown_ms == 0.0

    def test_line_id_auto_generated(self):
        """Test that line_id is auto-generated UUID."""
        line1 = VOLine()
        line2 = VOLine()

        # Should be valid UUIDs and unique
        assert line1.line_id != line2.line_id
        uuid.UUID(line1.line_id)  # Should not raise
        uuid.UUID(line2.line_id)

    def test_explicit_line_id(self):
        """Test explicit line_id is used."""
        line = VOLine(line_id="test-id-123")
        assert line.line_id == "test-id-123"

    def test_full_initialization(self):
        """Test VOLine with all fields."""
        line = VOLine(
            audio_asset="audio/test.wav",
            text="Hello World",
            speaker_id="npc_01",
            duration_ms=1500.0,
            priority=PRIORITY_HIGH,
            interruptible=False,
            context_type="conversation",
            tags={"greeting", "intro"},
            language="es",
            weight=0.8,
            cooldown_ms=5000.0,
        )

        assert line.audio_asset == "audio/test.wav"
        assert line.text == "Hello World"
        assert line.speaker_id == "npc_01"
        assert line.duration_ms == 1500.0
        assert line.priority == PRIORITY_HIGH
        assert line.interruptible is False
        assert line.context_type == "conversation"
        assert "greeting" in line.tags
        assert line.language == "es"
        assert line.weight == 0.8
        assert line.cooldown_ms == 5000.0

    def test_post_init_tags_list_to_set(self):
        """Test __post_init__ converts tags list to set."""
        line = VOLine(tags=["tag1", "tag2", "tag3"])
        assert isinstance(line.tags, set)
        assert line.tags == {"tag1", "tag2", "tag3"}

    def test_post_init_priority_enum_to_int(self):
        """Test __post_init__ converts VOPriority enum to int."""
        line = VOLine(priority=VOPriority.HIGH)
        assert isinstance(line.priority, int)
        assert line.priority == 75


# =============================================================================
# VOLine State Property Tests
# =============================================================================


class TestVOLineStateProperties:
    """Tests for VOLine state properties."""

    def test_initial_state_pending(self):
        """Test initial state is PENDING."""
        line = VOLine()
        assert line.state == VOLineState.PENDING

    def test_state_setter(self):
        """Test state setter."""
        line = VOLine()
        line.state = VOLineState.PLAYING
        assert line.state == VOLineState.PLAYING

    def test_playback_position_initial(self):
        """Test initial playback position is zero."""
        line = VOLine(duration_ms=1000.0)
        assert line.playback_position_ms == 0.0

    def test_playback_position_setter(self):
        """Test playback position setter."""
        line = VOLine(duration_ms=1000.0)
        line.playback_position_ms = 500.0
        assert line.playback_position_ms == 500.0

    def test_playback_position_clamped_min(self):
        """Test playback position clamped to zero."""
        line = VOLine(duration_ms=1000.0)
        line.playback_position_ms = -100.0
        assert line.playback_position_ms == 0.0

    def test_playback_position_clamped_max(self):
        """Test playback position clamped to duration."""
        line = VOLine(duration_ms=1000.0)
        line.playback_position_ms = 2000.0
        assert line.playback_position_ms == 1000.0

    def test_progress_property(self):
        """Test progress property calculation."""
        line = VOLine(duration_ms=1000.0)
        line.playback_position_ms = 250.0
        assert line.progress == 0.25

    def test_progress_zero_duration(self):
        """Test progress returns 0 when duration is zero."""
        line = VOLine(duration_ms=0.0)
        assert line.progress == 0.0

    def test_remaining_ms_property(self):
        """Test remaining_ms property."""
        line = VOLine(duration_ms=1000.0)
        line.playback_position_ms = 300.0
        assert line.remaining_ms == 700.0

    def test_remaining_ms_clamped(self):
        """Test remaining_ms clamped to zero."""
        line = VOLine(duration_ms=1000.0)
        line.playback_position_ms = 1000.0
        assert line.remaining_ms == 0.0


# =============================================================================
# VOLine State Query Tests
# =============================================================================


class TestVOLineStateQueries:
    """Tests for VOLine state query methods."""

    def test_is_playing_true(self):
        """Test is_playing returns True when PLAYING."""
        line = VOLine()
        line.state = VOLineState.PLAYING
        assert line.is_playing is True

    def test_is_playing_false(self):
        """Test is_playing returns False for other states."""
        line = VOLine()
        for state in [VOLineState.PENDING, VOLineState.PAUSED, VOLineState.COMPLETED]:
            line.state = state
            assert line.is_playing is False

    def test_is_completed_completed(self):
        """Test is_completed for COMPLETED state."""
        line = VOLine()
        line.state = VOLineState.COMPLETED
        assert line.is_completed is True

    def test_is_completed_interrupted(self):
        """Test is_completed for INTERRUPTED state."""
        line = VOLine()
        line.state = VOLineState.INTERRUPTED
        assert line.is_completed is True

    def test_is_completed_false(self):
        """Test is_completed returns False for other states."""
        line = VOLine()
        for state in [VOLineState.PENDING, VOLineState.PLAYING, VOLineState.PAUSED]:
            line.state = state
            assert line.is_completed is False

    def test_can_interrupt_true(self):
        """Test can_interrupt when interruptible and playing."""
        line = VOLine(interruptible=True)
        line.state = VOLineState.PLAYING
        assert line.can_interrupt is True

    def test_can_interrupt_not_interruptible(self):
        """Test can_interrupt when not interruptible."""
        line = VOLine(interruptible=False)
        line.state = VOLineState.PLAYING
        assert line.can_interrupt is False

    def test_can_interrupt_not_playing(self):
        """Test can_interrupt when not playing."""
        line = VOLine(interruptible=True)
        line.state = VOLineState.PENDING
        assert line.can_interrupt is False

    def test_can_be_interrupted_by_higher_priority(self):
        """Test can_be_interrupted_by with higher priority."""
        line = VOLine(priority=PRIORITY_NORMAL, interruptible=True)
        assert line.can_be_interrupted_by(PRIORITY_HIGH) is True

    def test_can_be_interrupted_by_lower_priority(self):
        """Test can_be_interrupted_by with lower priority."""
        line = VOLine(priority=PRIORITY_HIGH, interruptible=True)
        assert line.can_be_interrupted_by(PRIORITY_NORMAL) is False

    def test_can_be_interrupted_by_not_interruptible(self):
        """Test can_be_interrupted_by when not interruptible."""
        line = VOLine(priority=PRIORITY_NORMAL, interruptible=False)
        assert line.can_be_interrupted_by(PRIORITY_CRITICAL) is False


# =============================================================================
# VOLine Playback Control Tests
# =============================================================================


class TestVOLinePlaybackControl:
    """Tests for VOLine playback control methods."""

    def test_start_playback(self):
        """Test start_playback method."""
        line = VOLine(duration_ms=1000.0)
        line.start_playback(100.0)

        assert line.state == VOLineState.PLAYING
        assert line.playback_position_ms == 0.0
        assert line._last_played_time == 100.0
        assert line._play_count == 1

    def test_start_playback_increments_play_count(self):
        """Test start_playback increments play count."""
        line = VOLine(duration_ms=1000.0)
        line.start_playback(100.0)
        line.state = VOLineState.PENDING
        line.start_playback(200.0)

        assert line._play_count == 2

    def test_start_playback_calls_on_start(self):
        """Test start_playback calls on_start callback."""
        callback = MagicMock()
        line = VOLine(duration_ms=1000.0, on_start=callback)
        line.start_playback(100.0)

        callback.assert_called_once_with(line)

    def test_update_playback(self):
        """Test update_playback advances position."""
        line = VOLine(duration_ms=1000.0)
        line.state = VOLineState.PLAYING
        line.update_playback(100.0)

        assert line.playback_position_ms == 100.0

    def test_update_playback_not_playing(self):
        """Test update_playback does nothing when not playing."""
        line = VOLine(duration_ms=1000.0)
        line.update_playback(100.0)

        assert line.playback_position_ms == 0.0

    def test_update_playback_completes_at_duration(self):
        """Test update_playback completes when reaching duration."""
        line = VOLine(duration_ms=1000.0)
        line.state = VOLineState.PLAYING
        line.update_playback(1500.0)

        assert line.state == VOLineState.COMPLETED

    def test_complete_playback_normal(self):
        """Test complete_playback with normal completion."""
        callback = MagicMock()
        line = VOLine(on_end=callback)
        line.complete_playback(interrupted=False)

        assert line.state == VOLineState.COMPLETED
        callback.assert_called_once_with(line, False)

    def test_complete_playback_interrupted(self):
        """Test complete_playback with interruption."""
        callback = MagicMock()
        line = VOLine(on_end=callback)
        line.complete_playback(interrupted=True)

        assert line.state == VOLineState.INTERRUPTED
        callback.assert_called_once_with(line, True)

    def test_pause_from_playing(self):
        """Test pause from PLAYING state."""
        line = VOLine()
        line.state = VOLineState.PLAYING
        line.pause()

        assert line.state == VOLineState.PAUSED

    def test_pause_not_playing(self):
        """Test pause when not playing does nothing."""
        line = VOLine()
        line.pause()

        assert line.state == VOLineState.PENDING

    def test_resume_from_paused(self):
        """Test resume from PAUSED state."""
        line = VOLine()
        line.state = VOLineState.PAUSED
        line.resume()

        assert line.state == VOLineState.PLAYING

    def test_resume_not_paused(self):
        """Test resume when not paused does nothing."""
        line = VOLine()
        line.state = VOLineState.PENDING
        line.resume()

        assert line.state == VOLineState.PENDING

    def test_reset(self):
        """Test reset method."""
        line = VOLine(duration_ms=1000.0)
        line.state = VOLineState.PLAYING
        line.playback_position_ms = 500.0
        line.reset()

        assert line.state == VOLineState.PENDING
        assert line.playback_position_ms == 0.0


# =============================================================================
# VOLine Cooldown Tests
# =============================================================================


class TestVOLineCooldown:
    """Tests for VOLine cooldown functionality."""

    def test_is_on_cooldown_never_played(self):
        """Test is_on_cooldown when never played."""
        line = VOLine()
        assert line.is_on_cooldown(100.0, 1000.0) is False

    def test_is_on_cooldown_within_cooldown(self):
        """Test is_on_cooldown within cooldown period."""
        line = VOLine()
        line._last_played_time = 100.0

        # 500ms elapsed, 1000ms cooldown
        assert line.is_on_cooldown(100.5, 1000.0) is True

    def test_is_on_cooldown_after_cooldown(self):
        """Test is_on_cooldown after cooldown period."""
        line = VOLine()
        line._last_played_time = 100.0

        # 2000ms elapsed, 1000ms cooldown
        assert line.is_on_cooldown(102.0, 1000.0) is False

    def test_is_on_cooldown_uses_line_cooldown(self):
        """Test is_on_cooldown uses line's own cooldown if set."""
        line = VOLine(cooldown_ms=5000.0)
        line._last_played_time = 100.0

        # Line has 5000ms cooldown, should override default 1000ms
        assert line.is_on_cooldown(102.0, 1000.0) is True
        assert line.is_on_cooldown(106.0, 1000.0) is False


# =============================================================================
# VOLine Condition Matching Tests
# =============================================================================


class TestVOLineConditions:
    """Tests for VOLine condition matching."""

    def test_matches_conditions_empty(self):
        """Test matches_conditions with no conditions."""
        line = VOLine()
        assert line.matches_conditions({}) is True
        assert line.matches_conditions({"key": "value"}) is True

    def test_matches_conditions_simple_match(self):
        """Test matches_conditions with simple value matching."""
        line = VOLine(conditions={"level": 5, "name": "test"})

        assert line.matches_conditions({"level": 5, "name": "test"}) is True
        assert line.matches_conditions({"level": 5, "name": "test", "extra": 1}) is True

    def test_matches_conditions_mismatch(self):
        """Test matches_conditions with mismatched value."""
        line = VOLine(conditions={"level": 5})

        assert line.matches_conditions({"level": 10}) is False

    def test_matches_conditions_missing_key(self):
        """Test matches_conditions with missing key in state."""
        line = VOLine(conditions={"required_key": True})

        assert line.matches_conditions({}) is False

    def test_matches_conditions_callable(self):
        """Test matches_conditions with callable condition."""
        line = VOLine(conditions={"level": lambda x: x > 10})

        assert line.matches_conditions({"level": 15}) is True
        assert line.matches_conditions({"level": 5}) is False

    def test_matches_conditions_callable_none(self):
        """Test matches_conditions with callable receiving None."""
        line = VOLine(conditions={"key": lambda x: x is not None})

        assert line.matches_conditions({}) is False


# =============================================================================
# VOLine Tag Tests
# =============================================================================


class TestVOLineTags:
    """Tests for VOLine tag functionality."""

    def test_has_tag_true(self):
        """Test has_tag returns True when tag exists."""
        line = VOLine(tags={"combat", "greeting"})
        assert line.has_tag("combat") is True

    def test_has_tag_false(self):
        """Test has_tag returns False when tag missing."""
        line = VOLine(tags={"combat", "greeting"})
        assert line.has_tag("missing") is False

    def test_has_tag_empty_tags(self):
        """Test has_tag with empty tag set."""
        line = VOLine()
        assert line.has_tag("any") is False

    def test_has_all_tags_true(self):
        """Test has_all_tags returns True when all tags exist."""
        line = VOLine(tags={"a", "b", "c"})
        assert line.has_all_tags({"a", "b"}) is True

    def test_has_all_tags_false(self):
        """Test has_all_tags returns False when some missing."""
        line = VOLine(tags={"a", "b"})
        assert line.has_all_tags({"a", "c"}) is False

    def test_has_all_tags_empty(self):
        """Test has_all_tags with empty query."""
        line = VOLine(tags={"a"})
        assert line.has_all_tags(set()) is True

    def test_has_any_tag_true(self):
        """Test has_any_tag returns True when any tag exists."""
        line = VOLine(tags={"a", "b"})
        assert line.has_any_tag({"b", "c"}) is True

    def test_has_any_tag_false(self):
        """Test has_any_tag returns False when no tags match."""
        line = VOLine(tags={"a", "b"})
        assert line.has_any_tag({"c", "d"}) is False

    def test_has_any_tag_empty(self):
        """Test has_any_tag with empty query."""
        line = VOLine(tags={"a"})
        assert line.has_any_tag(set()) is False


# =============================================================================
# VOLine Clone Tests
# =============================================================================


class TestVOLineClone:
    """Tests for VOLine clone method."""

    def test_clone_creates_new_id(self):
        """Test clone creates new line_id."""
        original = VOLine(line_id="original-id")
        clone = original.clone()

        assert clone.line_id != original.line_id
        uuid.UUID(clone.line_id)  # Should be valid UUID

    def test_clone_copies_properties(self):
        """Test clone copies all properties."""
        original = VOLine(
            audio_asset="test.wav",
            text="Hello",
            speaker_id="npc",
            duration_ms=1000.0,
            priority=PRIORITY_HIGH,
            interruptible=False,
            context_type="conversation",
            tags={"tag1"},
            language="es",
            weight=0.5,
            cooldown_ms=2000.0,
        )
        clone = original.clone()

        assert clone.audio_asset == original.audio_asset
        assert clone.text == original.text
        assert clone.speaker_id == original.speaker_id
        assert clone.duration_ms == original.duration_ms
        assert clone.priority == original.priority
        assert clone.interruptible == original.interruptible
        assert clone.context_type == original.context_type
        assert clone.language == original.language
        assert clone.weight == original.weight
        assert clone.cooldown_ms == original.cooldown_ms

    def test_clone_independent_tags(self):
        """Test clone creates independent tags set."""
        original = VOLine(tags={"tag1"})
        clone = original.clone()

        clone.tags.add("tag2")
        assert "tag2" not in original.tags

    def test_clone_independent_conditions(self):
        """Test clone creates independent conditions dict."""
        original = VOLine(conditions={"key": "value"})
        clone = original.clone()

        clone.conditions["new_key"] = "new_value"
        assert "new_key" not in original.conditions

    def test_clone_copies_callbacks(self):
        """Test clone copies callback references."""
        callback = MagicMock()
        original = VOLine(on_start=callback, on_end=callback)
        clone = original.clone()

        assert clone.on_start is original.on_start
        assert clone.on_end is original.on_end


# =============================================================================
# VOLine Serialization Tests
# =============================================================================


class TestVOLineSerialization:
    """Tests for VOLine serialization."""

    def test_to_dict_basic(self):
        """Test to_dict includes all serializable fields."""
        line = VOLine(
            line_id="test-id",
            audio_asset="test.wav",
            text="Hello",
            speaker_id="npc",
            duration_ms=1000.0,
            priority=PRIORITY_HIGH,
            interruptible=False,
            context_type="conversation",
            tags={"tag1", "tag2"},
            language="es",
            weight=0.5,
            cooldown_ms=2000.0,
        )
        data = line.to_dict()

        assert data["line_id"] == "test-id"
        assert data["audio_asset"] == "test.wav"
        assert data["text"] == "Hello"
        assert data["speaker_id"] == "npc"
        assert data["duration_ms"] == 1000.0
        assert data["priority"] == PRIORITY_HIGH
        assert data["interruptible"] is False
        assert data["context_type"] == "conversation"
        assert set(data["tags"]) == {"tag1", "tag2"}
        assert data["language"] == "es"
        assert data["weight"] == 0.5
        assert data["cooldown_ms"] == 2000.0

    def test_to_dict_tags_as_list(self):
        """Test to_dict converts tags to list."""
        line = VOLine(tags={"a", "b"})
        data = line.to_dict()

        assert isinstance(data["tags"], list)

    def test_from_dict_basic(self):
        """Test from_dict creates VOLine correctly."""
        data = {
            "line_id": "test-id",
            "audio_asset": "test.wav",
            "text": "Hello",
            "speaker_id": "npc",
            "duration_ms": 1000.0,
            "priority": PRIORITY_HIGH,
            "interruptible": False,
            "context_type": "conversation",
            "tags": ["tag1", "tag2"],
            "language": "es",
            "weight": 0.5,
            "cooldown_ms": 2000.0,
        }
        line = VOLine.from_dict(data)

        assert line.line_id == "test-id"
        assert line.audio_asset == "test.wav"
        assert line.text == "Hello"
        assert line.speaker_id == "npc"
        assert line.duration_ms == 1000.0
        assert line.priority == PRIORITY_HIGH
        assert line.interruptible is False
        assert line.tags == {"tag1", "tag2"}

    def test_from_dict_defaults(self):
        """Test from_dict uses defaults for missing fields."""
        data = {}
        line = VOLine.from_dict(data)

        assert line.audio_asset == ""
        assert line.text == ""
        assert line.priority == PRIORITY_NORMAL
        assert line.interruptible is True
        assert line.language == "en"
        assert line.weight == 1.0

    def test_from_dict_generates_id(self):
        """Test from_dict generates line_id if missing."""
        data = {"audio_asset": "test.wav"}
        line = VOLine.from_dict(data)

        assert line.line_id is not None
        uuid.UUID(line.line_id)

    def test_roundtrip_serialization(self):
        """Test serialization roundtrip preserves data."""
        original = VOLine(
            audio_asset="test.wav",
            text="Hello",
            speaker_id="npc",
            duration_ms=1000.0,
            priority=PRIORITY_HIGH,
            tags={"a", "b"},
        )
        data = original.to_dict()
        restored = VOLine.from_dict(data)

        assert restored.line_id == original.line_id
        assert restored.audio_asset == original.audio_asset
        assert restored.text == original.text
        assert restored.priority == original.priority
        assert restored.tags == original.tags


# =============================================================================
# create_vo_line Factory Tests
# =============================================================================


class TestCreateVOLine:
    """Tests for create_vo_line factory function."""

    def test_create_basic(self):
        """Test create_vo_line with required fields."""
        line = create_vo_line(
            audio_asset="test.wav",
            text="Hello",
        )

        assert line.audio_asset == "test.wav"
        assert line.text == "Hello"
        assert line.speaker_id == ""
        assert line.priority == PRIORITY_NORMAL

    def test_create_with_options(self):
        """Test create_vo_line with all options."""
        line = create_vo_line(
            audio_asset="test.wav",
            text="Hello",
            speaker_id="npc",
            duration_ms=1000.0,
            priority=PRIORITY_HIGH,
            interruptible=False,
            context_type="conversation",
            tags={"tag1"},
            weight=0.5,
        )

        assert line.speaker_id == "npc"
        assert line.duration_ms == 1000.0
        assert line.priority == PRIORITY_HIGH
        assert line.interruptible is False
        assert line.weight == 0.5

    def test_create_with_none_tags(self):
        """Test create_vo_line with None tags."""
        line = create_vo_line(
            audio_asset="test.wav",
            text="Hello",
            tags=None,
        )

        assert line.tags == set()

    def test_create_with_kwargs(self):
        """Test create_vo_line with extra kwargs."""
        line = create_vo_line(
            audio_asset="test.wav",
            text="Hello",
            language="es",
            cooldown_ms=5000.0,
        )

        assert line.language == "es"
        assert line.cooldown_ms == 5000.0


# =============================================================================
# VOLine Lip Sync and Subtitle Integration
# =============================================================================


class TestVOLineLipSyncSubtitle:
    """Tests for VOLine lip sync and subtitle integration."""

    def test_lip_sync_attached(self):
        """Test VOLine with attached lip sync data."""
        lip_sync = LipSyncData(
            phonemes=[(0.0, "AA"), (100.0, "B")],
            visemes=[(0.0, 1), (100.0, 2)],
        )
        line = VOLine(lip_sync=lip_sync)

        assert line.lip_sync is lip_sync
        assert line.lip_sync.get_phoneme_at(50.0) == "AA"

    def test_subtitle_attached(self):
        """Test VOLine with attached subtitle data."""
        subtitle = SubtitleData(
            text="Hello World",
            speaker_name="NPC",
            start_time_ms=0.0,
            end_time_ms=1000.0,
        )
        line = VOLine(subtitle=subtitle)

        assert line.subtitle is subtitle
        assert line.subtitle.text == "Hello World"

    def test_lip_sync_and_subtitle_both(self):
        """Test VOLine with both lip sync and subtitle."""
        lip_sync = LipSyncData(phonemes=[(0.0, "AA")])
        subtitle = SubtitleData(text="Test")

        line = VOLine(lip_sync=lip_sync, subtitle=subtitle)

        assert line.lip_sync is lip_sync
        assert line.subtitle is subtitle
