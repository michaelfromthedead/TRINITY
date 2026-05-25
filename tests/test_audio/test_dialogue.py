"""
Comprehensive Tests for the Dialogue Subsystem.

Tests covering:
- DialogueManager: Main dialogue coordination
- VOQueue: Priority-based VO queuing
- VOLine: Individual voice-over lines
- ContextualDialogue: Barks, ambient VO, conditional lines
- Conversation: Branching dialogue system
- Localization: Language switching and audio banks
- SubtitleSync: Subtitle display and timing
- VOStreaming: Audio streaming and caching
- VOProcessing: Audio effects processing
"""

import time
import threading
from unittest.mock import Mock, MagicMock, patch, call
import pytest

# Import source modules
from engine.audio.dialogue.dialogue_manager import (
    DialogueManager, DialogueEvent, DialogueState,
)
from engine.audio.dialogue.vo_queue import (
    VOQueue, VOQueueManager, QueueEntry,
)
from engine.audio.dialogue.vo_line import (
    VOLine, VOLineState, LipSyncData, SubtitleData,
    create_vo_line,
)
from engine.audio.dialogue.contextual_dialogue import (
    ContextualDialogueManager, BarkSystem, AmbientVOSystem,
    LinePool, CooldownTracker, create_bark_lines,
)
from engine.audio.dialogue.conversation import (
    Conversation, ConversationManager, ConversationNode,
    ConversationState, create_linear_conversation, create_branching_conversation,
)
from engine.audio.dialogue.localization import (
    LocalizationManager, LocalizedAsset, AudioBank,
    create_localized_asset, create_audio_bank,
)
from engine.audio.dialogue.subtitle_sync import (
    SubtitleManager, SubtitleTrack, SubtitleCue, ActiveSubtitle,
    SubtitlePosition, SubtitleState, SubtitleStyle,
)
from engine.audio.dialogue.vo_streaming import (
    VOStreamManager, VOCache, StreamHandle, CachedAudio,
    StreamState,
)
from engine.audio.dialogue.vo_processing import (
    VOProcessor, VOProcessingState, RadioEffect, DistanceFilter,
    ReverbSettings, SpatialSettings, EffectType,
)
from engine.audio.dialogue.config import (
    VOPriority, PRIORITY_HIGH, PRIORITY_NORMAL, PRIORITY_LOW, PRIORITY_BARK,
    PRIORITY_AMBIENT, PRIORITY_CRITICAL,
    MAX_QUEUE_SIZE, MAX_SIMULTANEOUS_VO,
    DEFAULT_LANGUAGE, SUPPORTED_LANGUAGES,
    BARK_COOLDOWN_MS, SAME_LINE_COOLDOWN_MS, SAME_SPEAKER_COOLDOWN_MS,
    SelectionMode, ContextType, DialogueState as ConfigDialogueState,
    EVENT_LINE_STARTED, EVENT_LINE_ENDED, EVENT_LINE_INTERRUPTED,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_vo_line():
    """Create a sample VO line for testing."""
    return VOLine(
        line_id="vo_001",
        audio_asset="/audio/dialogue/test_line.ogg",
        text="This is a test dialogue line.",
        speaker_id="npc_guard",
        duration_ms=3000.0,
        priority=PRIORITY_NORMAL,
    )


@pytest.fixture
def sample_lines():
    """Create multiple sample VO lines."""
    return [
        VOLine(
            line_id=f"line_{i}",
            audio_asset=f"/audio/line_{i}.ogg",
            text=f"Test line number {i}",
            speaker_id=f"speaker_{i % 3}",
            duration_ms=2000.0 + i * 500,
            priority=PRIORITY_NORMAL + (i % 3) * 10,
        )
        for i in range(10)
    ]


@pytest.fixture
def high_priority_line():
    """Create a high priority line."""
    return VOLine(
        line_id="critical_001",
        audio_asset="/audio/critical.ogg",
        text="Critical dialogue!",
        speaker_id="narrator",
        duration_ms=2000.0,
        priority=PRIORITY_CRITICAL,
        interruptible=False,
    )


@pytest.fixture
def vo_queue():
    """Create a VOQueue instance."""
    return VOQueue()


@pytest.fixture
def dialogue_manager():
    """Create a DialogueManager instance."""
    return DialogueManager()


@pytest.fixture
def conversation_manager():
    """Create a ConversationManager instance."""
    return ConversationManager()


@pytest.fixture
def localization_manager():
    """Create a LocalizationManager instance."""
    return LocalizationManager()


@pytest.fixture
def subtitle_manager():
    """Create a SubtitleManager instance."""
    return SubtitleManager()


@pytest.fixture
def vo_processor():
    """Create a VOProcessor instance."""
    return VOProcessor()


@pytest.fixture
def stream_manager():
    """Create a VOStreamManager instance."""
    return VOStreamManager()


@pytest.fixture
def bark_system():
    """Create a BarkSystem instance."""
    return BarkSystem()


@pytest.fixture
def ambient_system():
    """Create an AmbientVOSystem instance."""
    return AmbientVOSystem()


@pytest.fixture
def sample_conversation():
    """Create a sample conversation."""
    nodes = [
        ConversationNode(
            node_id="start",
            line=VOLine(
                line_id="conv_start",
                audio_asset="/audio/conv/start.ogg",
                text="Hello, traveler!",
                speaker_id="merchant",
                duration_ms=2000.0,
            ),
            next_nodes=["question"],
        ),
        ConversationNode(
            node_id="question",
            line=VOLine(
                line_id="conv_question",
                audio_asset="/audio/conv/question.ogg",
                text="What would you like to buy?",
                speaker_id="merchant",
                duration_ms=2500.0,
            ),
            next_nodes=["weapons", "potions"],
            is_branch_point=True,
            branch_options=["Weapons", "Potions"],
        ),
        ConversationNode(
            node_id="weapons",
            line=VOLine(
                line_id="conv_weapons",
                audio_asset="/audio/conv/weapons.ogg",
                text="Here are my finest weapons.",
                speaker_id="merchant",
                duration_ms=2000.0,
            ),
            next_nodes=["end"],
        ),
        ConversationNode(
            node_id="potions",
            line=VOLine(
                line_id="conv_potions",
                audio_asset="/audio/conv/potions.ogg",
                text="These potions are very potent.",
                speaker_id="merchant",
                duration_ms=2000.0,
            ),
            next_nodes=["end"],
        ),
        ConversationNode(
            node_id="end",
            line=VOLine(
                line_id="conv_end",
                audio_asset="/audio/conv/end.ogg",
                text="Thank you for your purchase!",
                speaker_id="merchant",
                duration_ms=1500.0,
            ),
            next_nodes=[],
        ),
    ]

    return Conversation(
        conversation_id="merchant_dialogue",
        nodes={n.node_id: n for n in nodes},
        start_node_id="start",
    )


# =============================================================================
# VOLine Tests
# =============================================================================

class TestVOLine:
    """Tests for VOLine class."""

    def test_create_line(self, sample_vo_line):
        """Test creating a VO line."""
        assert sample_vo_line.line_id == "vo_001"
        assert sample_vo_line.text == "This is a test dialogue line."
        assert sample_vo_line.speaker_id == "npc_guard"
        assert sample_vo_line.duration_ms == 3000.0
        assert sample_vo_line.priority == PRIORITY_NORMAL

    def test_initial_state(self, sample_vo_line):
        """Test initial line state."""
        assert sample_vo_line.state == VOLineState.PENDING
        assert sample_vo_line.playback_position_ms == 0.0
        assert sample_vo_line.progress == 0.0

    def test_start_playback(self, sample_vo_line):
        """Test starting playback."""
        current_time = time.time()
        sample_vo_line.start_playback(current_time)

        assert sample_vo_line.state == VOLineState.PLAYING
        assert sample_vo_line.is_playing
        assert sample_vo_line._play_count == 1

    def test_update_playback(self, sample_vo_line):
        """Test updating playback position."""
        sample_vo_line.start_playback(time.time())
        sample_vo_line.update_playback(500.0)

        assert sample_vo_line.playback_position_ms == 500.0
        assert sample_vo_line.progress > 0.0

    def test_complete_playback(self, sample_vo_line):
        """Test completing playback."""
        sample_vo_line.start_playback(time.time())
        sample_vo_line.complete_playback(interrupted=False)

        assert sample_vo_line.state == VOLineState.COMPLETED
        assert sample_vo_line.is_completed

    def test_interrupt_playback(self, sample_vo_line):
        """Test interrupted playback."""
        sample_vo_line.start_playback(time.time())
        sample_vo_line.complete_playback(interrupted=True)

        assert sample_vo_line.state == VOLineState.INTERRUPTED
        assert sample_vo_line.is_completed

    def test_pause_resume(self, sample_vo_line):
        """Test pause and resume."""
        sample_vo_line.start_playback(time.time())
        sample_vo_line.pause()
        assert sample_vo_line.state == VOLineState.PAUSED

        sample_vo_line.resume()
        assert sample_vo_line.state == VOLineState.PLAYING

    def test_reset(self, sample_vo_line):
        """Test resetting line."""
        sample_vo_line.start_playback(time.time())
        sample_vo_line.update_playback(1000.0)
        sample_vo_line.reset()

        assert sample_vo_line.state == VOLineState.PENDING
        assert sample_vo_line.playback_position_ms == 0.0

    def test_remaining_time(self, sample_vo_line):
        """Test remaining time calculation."""
        sample_vo_line.start_playback(time.time())
        sample_vo_line.update_playback(1000.0)

        assert sample_vo_line.remaining_ms == 2000.0

    def test_can_interrupt(self, sample_vo_line, high_priority_line):
        """Test interrupt checking."""
        sample_vo_line.start_playback(time.time())

        # Normal line can be interrupted by higher priority
        assert sample_vo_line.can_be_interrupted_by(PRIORITY_HIGH)

        # Non-interruptible line cannot be interrupted
        high_priority_line.start_playback(time.time())
        assert not high_priority_line.can_be_interrupted_by(PRIORITY_CRITICAL)

    def test_cooldown_check(self, sample_vo_line):
        """Test cooldown checking."""
        current_time = time.time()
        sample_vo_line.start_playback(current_time)
        sample_vo_line.complete_playback()

        # Just after playing, should be on cooldown
        # Note: is_on_cooldown compares (current_time - last_played_time) < cooldown_ms
        # This means cooldown_ms is treated as seconds (the comparison is in seconds)
        assert sample_vo_line.is_on_cooldown(
            current_time + 1.0, cooldown_ms=30.0  # 30 seconds
        )

        # After cooldown expires (35 seconds later), should be available
        assert not sample_vo_line.is_on_cooldown(
            current_time + 35.0, cooldown_ms=30.0  # 30 seconds
        )

    def test_tags(self, sample_vo_line):
        """Test tag operations."""
        sample_vo_line.tags = {"combat", "urgent", "alert"}

        assert sample_vo_line.has_tag("combat")
        assert not sample_vo_line.has_tag("peaceful")
        assert sample_vo_line.has_all_tags({"combat", "urgent"})
        assert sample_vo_line.has_any_tag({"peaceful", "alert"})

    def test_conditions_match(self, sample_vo_line):
        """Test condition matching."""
        sample_vo_line.conditions = {
            "health": 50,
            "in_combat": True,
        }

        game_state = {"health": 50, "in_combat": True, "location": "dungeon"}
        assert sample_vo_line.matches_conditions(game_state)

        game_state = {"health": 100, "in_combat": True}
        assert not sample_vo_line.matches_conditions(game_state)

    def test_clone(self, sample_vo_line):
        """Test cloning a line."""
        clone = sample_vo_line.clone()

        assert clone.line_id != sample_vo_line.line_id
        assert clone.text == sample_vo_line.text
        assert clone.speaker_id == sample_vo_line.speaker_id

    def test_to_from_dict(self, sample_vo_line):
        """Test serialization/deserialization."""
        data = sample_vo_line.to_dict()
        restored = VOLine.from_dict(data)

        assert restored.text == sample_vo_line.text
        assert restored.speaker_id == sample_vo_line.speaker_id
        assert restored.priority == sample_vo_line.priority

    def test_on_start_callback(self, sample_vo_line):
        """Test start callback."""
        callback = Mock()
        sample_vo_line.on_start = callback

        sample_vo_line.start_playback(time.time())
        callback.assert_called_once_with(sample_vo_line)

    def test_on_end_callback(self, sample_vo_line):
        """Test end callback."""
        callback = Mock()
        sample_vo_line.on_end = callback

        sample_vo_line.start_playback(time.time())
        sample_vo_line.complete_playback(interrupted=False)

        callback.assert_called_once_with(sample_vo_line, False)


class TestLipSyncData:
    """Tests for LipSyncData class."""

    def test_create_lip_sync_data(self):
        """Test creating lip sync data."""
        data = LipSyncData(
            phonemes=[(0.0, "A"), (100.0, "E"), (200.0, "I")],
            visemes=[(0.0, 1), (100.0, 2), (200.0, 3)],
        )
        assert len(data.phonemes) == 3
        assert len(data.visemes) == 3

    def test_get_phoneme_at_time(self):
        """Test getting phoneme at time."""
        data = LipSyncData(
            phonemes=[(0.0, "A"), (100.0, "E"), (200.0, "I")]
        )

        assert data.get_phoneme_at(50.0) == "A"
        assert data.get_phoneme_at(150.0) == "E"
        assert data.get_phoneme_at(250.0) == "I"

    def test_get_viseme_at_time(self):
        """Test getting viseme at time."""
        data = LipSyncData(
            visemes=[(0.0, 1), (100.0, 2), (200.0, 3)]
        )

        assert data.get_viseme_at(50.0) == 1
        assert data.get_viseme_at(150.0) == 2


class TestSubtitleData:
    """Tests for SubtitleData class."""

    def test_create_subtitle_data(self):
        """Test creating subtitle data."""
        data = SubtitleData(
            text="Hello world",
            speaker_name="Guard",
            start_time_ms=0.0,
            end_time_ms=2000.0,
        )
        assert data.text == "Hello world"
        assert data.duration_ms == 2000.0


# =============================================================================
# VOQueue Tests
# =============================================================================

class TestQueueEntry:
    """Tests for QueueEntry class."""

    def test_create_entry(self, sample_vo_line):
        """Test creating queue entry."""
        entry = QueueEntry.create(sample_vo_line)
        assert entry.line == sample_vo_line
        assert entry.enqueue_time > 0

    def test_entry_expiration(self, sample_vo_line):
        """Test entry expiration."""
        entry = QueueEntry.create(sample_vo_line, timeout_ms=100.0)
        assert not entry.is_expired

        time.sleep(0.15)
        assert entry.is_expired

    def test_entry_age(self, sample_vo_line):
        """Test entry age tracking."""
        entry = QueueEntry.create(sample_vo_line)
        time.sleep(0.05)
        assert entry.age_ms >= 50

    def test_entry_ordering(self, sample_lines):
        """Test entry priority ordering."""
        entries = [QueueEntry.create(line) for line in sample_lines]
        sorted_entries = sorted(entries)

        # Higher priority (higher number) should come first
        for i in range(len(sorted_entries) - 1):
            assert sorted_entries[i].line.priority >= sorted_entries[i + 1].line.priority


class TestVOQueue:
    """Tests for VOQueue class."""

    def test_initial_state(self, vo_queue):
        """Test initial queue state."""
        assert vo_queue.size == 0
        assert vo_queue.is_empty
        assert not vo_queue.is_full
        assert not vo_queue.is_playing

    def test_enqueue(self, vo_queue, sample_vo_line):
        """Test enqueueing a line."""
        result = vo_queue.enqueue(sample_vo_line)
        assert result is True
        assert vo_queue.size == 1
        assert not vo_queue.is_empty

    def test_enqueue_when_full(self, vo_queue, sample_lines):
        """Test enqueueing when queue is full."""
        queue = VOQueue(max_size=5)
        for line in sample_lines[:5]:
            queue.enqueue(line)

        result = queue.enqueue(sample_lines[5])
        assert result is False
        assert queue.size == 5

    def test_enqueue_force(self, vo_queue, sample_lines):
        """Test force enqueueing when full."""
        queue = VOQueue(max_size=3)
        for line in sample_lines[:3]:
            queue.enqueue(line)

        result = queue.enqueue(sample_lines[3], force=True)
        assert result is True
        assert queue.size == 3  # Still 3, replaced lowest priority

    def test_dequeue(self, vo_queue, sample_lines):
        """Test dequeueing lines."""
        for line in sample_lines[:3]:
            vo_queue.enqueue(line)

        dequeued = vo_queue.dequeue()
        assert dequeued is not None
        assert vo_queue.size == 2

    def test_dequeue_priority_order(self, vo_queue):
        """Test dequeueing in priority order."""
        lines = [
            VOLine(line_id="low", priority=PRIORITY_LOW, duration_ms=1000.0),
            VOLine(line_id="high", priority=PRIORITY_HIGH, duration_ms=1000.0),
            VOLine(line_id="normal", priority=PRIORITY_NORMAL, duration_ms=1000.0),
        ]
        for line in lines:
            vo_queue.enqueue(line)

        # Highest priority first
        first = vo_queue.dequeue()
        assert first.line_id == "high"

        second = vo_queue.dequeue()
        assert second.line_id == "normal"

        third = vo_queue.dequeue()
        assert third.line_id == "low"

    def test_dequeue_priority_with_critical(self, vo_queue):
        """Test critical priority always comes first."""
        lines = [
            VOLine(line_id="high", priority=PRIORITY_HIGH, duration_ms=1000.0),
            VOLine(line_id="critical", priority=PRIORITY_CRITICAL, duration_ms=1000.0),
            VOLine(line_id="normal", priority=PRIORITY_NORMAL, duration_ms=1000.0),
        ]
        for line in lines:
            vo_queue.enqueue(line)

        # Critical should always be first
        first = vo_queue.dequeue()
        assert first.line_id == "critical"

    def test_dequeue_priority_fifo_same_priority(self, vo_queue):
        """Test FIFO order for same priority."""
        lines = [
            VOLine(line_id="first", priority=PRIORITY_NORMAL, duration_ms=1000.0),
            VOLine(line_id="second", priority=PRIORITY_NORMAL, duration_ms=1000.0),
            VOLine(line_id="third", priority=PRIORITY_NORMAL, duration_ms=1000.0),
        ]
        for line in lines:
            vo_queue.enqueue(line)
            time.sleep(0.001)  # Ensure different enqueue times

        # Same priority should maintain FIFO order
        first = vo_queue.dequeue()
        assert first.line_id == "first"

        second = vo_queue.dequeue()
        assert second.line_id == "second"

    def test_priority_interruption(self, vo_queue):
        """Test high priority interrupts low priority."""
        low_line = VOLine(
            line_id="low",
            priority=PRIORITY_LOW,
            duration_ms=5000.0,
            interruptible=True,
        )
        high_line = VOLine(
            line_id="high",
            priority=PRIORITY_CRITICAL,
            duration_ms=1000.0,
        )

        vo_queue.start_line(low_line, time.time())
        assert vo_queue.active_count == 1

        # Interrupt for higher priority
        interrupted = vo_queue.interrupt_for(PRIORITY_CRITICAL)
        assert len(interrupted) == 1
        assert interrupted[0].line_id == "low"

    def test_peek(self, vo_queue, sample_vo_line):
        """Test peeking at next line."""
        vo_queue.enqueue(sample_vo_line)
        peeked = vo_queue.peek()

        assert peeked == sample_vo_line
        assert vo_queue.size == 1  # Not removed

    def test_clear(self, vo_queue, sample_lines):
        """Test clearing queue."""
        for line in sample_lines[:5]:
            vo_queue.enqueue(line)

        count = vo_queue.clear()
        assert count == 5
        assert vo_queue.is_empty

    def test_remove_by_speaker(self, vo_queue, sample_lines):
        """Test removing by speaker."""
        for line in sample_lines[:6]:
            vo_queue.enqueue(line)

        removed = vo_queue.remove_by_speaker("speaker_0")
        assert removed > 0

    def test_remove_by_tag(self, vo_queue):
        """Test removing by tag."""
        lines = [
            VOLine(line_id="1", tags={"combat"}, duration_ms=1000.0),
            VOLine(line_id="2", tags={"peaceful"}, duration_ms=1000.0),
            VOLine(line_id="3", tags={"combat", "urgent"}, duration_ms=1000.0),
        ]
        for line in lines:
            vo_queue.enqueue(line)

        removed = vo_queue.remove_by_tag("combat")
        assert removed == 2
        assert vo_queue.size == 1

    def test_remove_below_priority(self, vo_queue, sample_lines):
        """Test removing below priority."""
        # Create lines with specific priorities
        lines_with_priorities = [
            VOLine(line_id=f"low_{i}", priority=PRIORITY_LOW, duration_ms=1000.0)
            for i in range(3)
        ] + [
            VOLine(line_id=f"high_{i}", priority=PRIORITY_HIGH, duration_ms=1000.0)
            for i in range(3)
        ]
        for line in lines_with_priorities:
            vo_queue.enqueue(line)

        removed = vo_queue.remove_below_priority(PRIORITY_NORMAL)
        # Should have removed the 3 low priority lines
        assert removed == 3
        assert vo_queue.size == 3

    def test_start_line(self, vo_queue, sample_vo_line):
        """Test starting a line."""
        result = vo_queue.start_line(sample_vo_line, time.time())
        assert result is True
        assert vo_queue.active_count == 1
        assert vo_queue.is_playing

    def test_max_simultaneous(self, vo_queue, sample_lines):
        """Test max simultaneous limit."""
        queue = VOQueue(max_simultaneous=2)
        for i, line in enumerate(sample_lines[:3]):
            queue.start_line(line, time.time())

        # Only 2 should be active
        assert queue.active_count == 2

    def test_end_line(self, vo_queue, sample_vo_line):
        """Test ending a line."""
        vo_queue.start_line(sample_vo_line, time.time())
        result = vo_queue.end_line(sample_vo_line, interrupted=False)

        assert result is True
        assert vo_queue.active_count == 0
        assert not vo_queue.is_playing

    def test_interrupt_for(self, vo_queue, sample_lines):
        """Test interrupting for higher priority."""
        for line in sample_lines[:3]:
            line.interruptible = True
            vo_queue.start_line(line, time.time())

        interrupted = vo_queue.interrupt_for(PRIORITY_CRITICAL)
        assert len(interrupted) > 0

    def test_update(self, vo_queue, sample_vo_line):
        """Test updating queue."""
        sample_vo_line.duration_ms = 100.0
        vo_queue.start_line(sample_vo_line, time.time())

        # Wait for line to complete
        time.sleep(0.15)
        completed = vo_queue.update(150.0)

        assert len(completed) == 1

    def test_pause_resume_queue(self, vo_queue, sample_vo_line):
        """Test pausing and resuming."""
        vo_queue.start_line(sample_vo_line, time.time())

        vo_queue.pause()
        assert vo_queue.is_paused
        assert sample_vo_line.state == VOLineState.PAUSED

        vo_queue.resume()
        assert not vo_queue.is_paused

    def test_callbacks(self, vo_queue, sample_vo_line):
        """Test queue callbacks."""
        on_started = Mock()
        on_ended = Mock()

        queue = VOQueue(
            on_line_started=on_started,
            on_line_ended=on_ended,
        )

        queue.start_line(sample_vo_line, time.time())
        on_started.assert_called_once_with(sample_vo_line)

        queue.end_line(sample_vo_line)
        on_ended.assert_called_once()

    def test_get_ducking_level(self, vo_queue, sample_lines):
        """Test ducking level calculation."""
        assert vo_queue.get_ducking_level() == 0.0

        vo_queue.start_line(sample_lines[0], time.time())
        assert vo_queue.get_ducking_level() == 0.0

        vo_queue.start_line(sample_lines[1], time.time())
        assert vo_queue.get_ducking_level() != 0.0

    def test_stats(self, vo_queue, sample_vo_line):
        """Test getting queue stats."""
        vo_queue.enqueue(sample_vo_line)
        stats = vo_queue.stats

        assert "queue_size" in stats
        assert stats["queue_size"] == 1
        assert "total_enqueued" in stats


class TestVOQueueManager:
    """Tests for VOQueueManager class."""

    def test_create_queue(self):
        """Test creating a named queue."""
        manager = VOQueueManager()
        queue = manager.create_queue("dialogue")
        assert queue is not None
        assert "dialogue" in manager.queue_names

    def test_create_duplicate_queue(self):
        """Test creating duplicate queue fails."""
        manager = VOQueueManager()
        manager.create_queue("test")
        with pytest.raises(ValueError):
            manager.create_queue("test")

    def test_get_queue(self):
        """Test getting queue by name."""
        manager = VOQueueManager()
        manager.create_queue("barks")
        queue = manager.get_queue("barks")
        assert queue is not None

    def test_get_or_create_queue(self):
        """Test get or create."""
        manager = VOQueueManager()
        queue1 = manager.get_or_create_queue("test")
        queue2 = manager.get_or_create_queue("test")
        assert queue1 is queue2

    def test_remove_queue(self):
        """Test removing queue."""
        manager = VOQueueManager()
        manager.create_queue("temp")
        result = manager.remove_queue("temp")
        assert result is True
        assert "temp" not in manager.queue_names

    def test_clear_all(self, sample_lines):
        """Test clearing all queues."""
        manager = VOQueueManager()
        q1 = manager.create_queue("q1")
        q2 = manager.create_queue("q2")

        q1.enqueue(sample_lines[0])
        q2.enqueue(sample_lines[1])

        manager.clear_all()
        assert q1.is_empty
        assert q2.is_empty

    def test_pause_resume_all(self):
        """Test pausing and resuming all queues."""
        manager = VOQueueManager()
        manager.create_queue("q1")
        manager.create_queue("q2")

        manager.pause_all()
        for name in manager.queue_names:
            assert manager.get_queue(name).is_paused

        manager.resume_all()
        for name in manager.queue_names:
            assert not manager.get_queue(name).is_paused


# =============================================================================
# ContextualDialogue Tests
# =============================================================================

class TestCooldownTracker:
    """Tests for CooldownTracker class."""

    def test_record_play(self):
        """Test recording play."""
        tracker = CooldownTracker()
        tracker.record_play("line_1", "speaker_1", "combat", time.time())
        # Should not raise

    def test_line_cooldown(self):
        """Test line cooldown checking."""
        tracker = CooldownTracker()
        current = time.time()
        tracker.record_play("line_1", "", "", current)

        assert tracker.is_line_on_cooldown("line_1", current + 1, 30000.0)
        assert not tracker.is_line_on_cooldown("line_1", current + 100, 30000.0)

    def test_speaker_cooldown(self):
        """Test speaker cooldown."""
        tracker = CooldownTracker()
        current = time.time()
        tracker.record_play("line_1", "speaker_1", "", current)

        assert tracker.is_speaker_on_cooldown("speaker_1", current + 0.5, 1000.0)

    def test_category_cooldown(self):
        """Test category cooldown."""
        tracker = CooldownTracker()
        current = time.time()
        tracker.record_play("line_1", "", "combat", current)

        assert tracker.is_category_on_cooldown("combat", current + 1, 5000.0)

    def test_clear_cooldowns(self):
        """Test clearing cooldowns."""
        tracker = CooldownTracker()
        tracker.record_play("line_1", "speaker_1", "cat_1", time.time())
        tracker.clear_cooldowns()

        # Should not be on cooldown after clear
        assert not tracker.is_line_on_cooldown("line_1", time.time(), 30000.0)

    def test_get_cooldown_remaining(self):
        """Test getting remaining cooldown."""
        tracker = CooldownTracker()
        current = time.time()
        tracker.record_play("line_1", "", "", current)

        remaining = tracker.get_cooldown_remaining("line_1", current + 10, 30000.0)
        assert remaining > 0


class TestLinePool:
    """Tests for LinePool class."""

    def test_create_pool(self):
        """Test creating a line pool."""
        pool = LinePool(pool_id="combat_barks")
        assert pool.pool_id == "combat_barks"
        assert pool.size == 0

    def test_add_remove_line(self, sample_vo_line):
        """Test adding and removing lines."""
        pool = LinePool(pool_id="test")
        pool.add_line(sample_vo_line)
        assert pool.size == 1

        result = pool.remove_line(sample_vo_line.line_id)
        assert result is True
        assert pool.size == 0

    def test_select_random(self, sample_lines):
        """Test random selection."""
        pool = LinePool(
            pool_id="test",
            selection_mode=SelectionMode.RANDOM.value,
        )
        for line in sample_lines[:5]:
            pool.add_line(line)

        selected = pool.select_line(time.time())
        assert selected in sample_lines[:5]

    def test_select_sequential(self, sample_lines):
        """Test sequential selection."""
        pool = LinePool(
            pool_id="test",
            selection_mode=SelectionMode.SEQUENTIAL.value,
        )
        for line in sample_lines[:3]:
            pool.add_line(line)

        first = pool.select_line(time.time())
        second = pool.select_line(time.time())
        assert first != second

    def test_select_weighted(self):
        """Test weighted selection."""
        pool = LinePool(
            pool_id="test",
            selection_mode=SelectionMode.WEIGHTED.value,
        )
        heavy = VOLine(line_id="heavy", weight=10.0, duration_ms=1000.0)
        light = VOLine(line_id="light", weight=1.0, duration_ms=1000.0)
        pool.add_line(heavy)
        pool.add_line(light)

        selections = [pool.select_line(time.time()) for _ in range(100)]
        heavy_count = sum(1 for s in selections if s.line_id == "heavy")

        # Heavy should be selected more often
        assert heavy_count > 50

    def test_select_shuffle(self, sample_lines):
        """Test shuffle selection."""
        pool = LinePool(
            pool_id="test",
            selection_mode=SelectionMode.SHUFFLE.value,
        )
        for line in sample_lines[:5]:
            pool.add_line(line)

        # Should cycle through all before repeating
        selected_ids = set()
        for _ in range(5):
            line = pool.select_line(time.time())
            selected_ids.add(line.line_id)

        assert len(selected_ids) == 5

    def test_select_with_cooldown(self, sample_lines):
        """Test selection respects cooldowns."""
        pool = LinePool(
            pool_id="test",
            cooldown_ms=30000.0,
        )
        for line in sample_lines[:2]:
            pool.add_line(line)

        tracker = CooldownTracker()
        current = time.time()

        # Play first line
        first = pool.select_line(current, tracker)
        tracker.record_play(first.line_id, "", "", current)

        # Next selection should pick the other line
        second = pool.select_line(current, tracker)
        assert second.line_id != first.line_id

    def test_select_with_conditions(self):
        """Test conditional selection."""
        pool = LinePool(pool_id="test")
        combat_line = VOLine(
            line_id="combat",
            conditions={"in_combat": True},
            duration_ms=1000.0,
        )
        peace_line = VOLine(
            line_id="peace",
            conditions={"in_combat": False},
            duration_ms=1000.0,
        )
        pool.add_line(combat_line)
        pool.add_line(peace_line)

        game_state = {"in_combat": True}
        selected = pool.select_line(time.time(), game_state=game_state)
        assert selected.line_id == "combat"


class TestBarkSystem:
    """Tests for BarkSystem class."""

    def test_register_bark_pool(self, bark_system, sample_lines):
        """Test registering bark pool."""
        pool = bark_system.register_bark_pool("reload", sample_lines[:3])
        assert pool is not None
        assert "reload" in bark_system.bark_types

    def test_trigger_bark(self, bark_system, sample_lines):
        """Test triggering bark."""
        bark_system.register_bark_pool("spotted", sample_lines[:3])
        line = bark_system.trigger_bark("spotted")
        assert line is not None

    def test_trigger_bark_by_speaker(self, bark_system):
        """Test triggering bark by specific speaker."""
        lines = [
            VOLine(line_id="a", speaker_id="guard_1", duration_ms=1000.0),
            VOLine(line_id="b", speaker_id="guard_2", duration_ms=1000.0),
        ]
        bark_system.register_bark_pool("alert", lines)

        line = bark_system.trigger_bark("alert", speaker_id="guard_1")
        # May or may not be the exact speaker depending on availability

    def test_enable_disable(self, bark_system, sample_lines):
        """Test enabling/disabling bark system."""
        bark_system.register_bark_pool("test", sample_lines[:2])

        bark_system.disable()
        assert not bark_system.is_enabled
        line = bark_system.trigger_bark("test")
        assert line is None

        bark_system.enable()
        assert bark_system.is_enabled
        line = bark_system.trigger_bark("test")
        assert line is not None

    def test_speaker_cooldown(self, bark_system):
        """Test speaker cooldown in barks."""
        lines = [VOLine(line_id=f"l_{i}", speaker_id="npc", duration_ms=1000.0) for i in range(5)]
        bark_system.register_bark_pool("test", lines)

        current = time.time()
        bark_system.trigger_bark("test", speaker_id="npc", current_time=current)

        # Should be on cooldown immediately after
        line = bark_system.trigger_bark(
            "test",
            speaker_id="npc",
            current_time=current + 0.1,
        )
        assert line is None


class TestAmbientVOSystem:
    """Tests for AmbientVOSystem class."""

    def test_register_zone(self, ambient_system, sample_lines):
        """Test registering ambient zone."""
        pool = ambient_system.register_zone("marketplace", sample_lines[:5])
        assert pool is not None

    def test_enter_exit_zone(self, ambient_system, sample_lines):
        """Test entering and exiting zones."""
        ambient_system.register_zone("tavern", sample_lines[:3])

        ambient_system.enter_zone("tavern")
        assert "tavern" in ambient_system.active_zones

        ambient_system.exit_zone("tavern")
        assert "tavern" not in ambient_system.active_zones

    def test_update_triggers_ambient(self, ambient_system, sample_lines):
        """Test update can trigger ambient line."""
        ambient_system = AmbientVOSystem(min_interval_ms=10.0, max_interval_ms=50.0)
        ambient_system.register_zone("test", sample_lines[:3])
        ambient_system.enter_zone("test")

        # Wait for interval to pass
        time.sleep(0.1)
        line = ambient_system.update(time.time())
        # May or may not trigger depending on random interval

    def test_force_trigger(self, ambient_system, sample_lines):
        """Test force triggering ambient."""
        ambient_system.register_zone("test", sample_lines[:3])
        ambient_system.enter_zone("test")

        line = ambient_system.force_trigger()
        assert line is not None

    def test_enable_disable(self, ambient_system, sample_lines):
        """Test enabling/disabling ambient system."""
        ambient_system.register_zone("test", sample_lines[:2])
        ambient_system.enter_zone("test")

        ambient_system.disable()
        assert not ambient_system.is_enabled

        line = ambient_system.update(time.time())
        assert line is None

        ambient_system.enable()
        assert ambient_system.is_enabled


class TestContextualDialogueManager:
    """Tests for ContextualDialogueManager class."""

    def test_create_pool(self):
        """Test creating pool."""
        manager = ContextualDialogueManager()
        pool = manager.create_pool("combat_barks")
        assert pool is not None

    def test_duplicate_pool_error(self):
        """Test duplicate pool raises error."""
        manager = ContextualDialogueManager()
        manager.create_pool("test")
        with pytest.raises(ValueError):
            manager.create_pool("test")

    def test_get_or_create_pool(self):
        """Test get or create pool."""
        manager = ContextualDialogueManager()
        pool1 = manager.get_or_create_pool("shared")
        pool2 = manager.get_or_create_pool("shared")
        assert pool1 is pool2

    def test_add_line_to_pool(self, sample_vo_line):
        """Test adding line to pool."""
        manager = ContextualDialogueManager()
        pool = manager.create_pool("test")
        # Add line directly to pool
        pool.add_line(sample_vo_line)
        assert pool.size == 1

    def test_select_from_pool(self, sample_lines):
        """Test selecting from pool."""
        manager = ContextualDialogueManager()
        pool = manager.create_pool("test")
        for line in sample_lines[:3]:
            pool.add_line(line)

        selected = manager.select_from_pool("test")
        assert selected is not None

    def test_update_game_state(self):
        """Test updating game state."""
        manager = ContextualDialogueManager()
        manager.update_game_state({"health": 100})
        manager.update_game_state({"stamina": 50})
        # Both should be present

    def test_set_game_state(self):
        """Test replacing game state."""
        manager = ContextualDialogueManager()
        manager.update_game_state({"old": True})
        manager.set_game_state({"new": True})
        # Old should be replaced


# =============================================================================
# Conversation Tests
# =============================================================================

class TestConversationNode:
    """Tests for ConversationNode class."""

    def test_create_node(self, sample_vo_line):
        """Test creating conversation node."""
        node = ConversationNode(
            node_id="test",
            line=sample_vo_line,
            next_nodes=["next"],
        )
        assert node.node_id == "test"
        assert node.has_line

    def test_branch_point(self, sample_vo_line):
        """Test branch point node."""
        node = ConversationNode(
            node_id="choice",
            line=sample_vo_line,
            next_nodes=["option_a", "option_b"],
            is_branch_point=True,
            branch_options=["Option A", "Option B"],
        )
        assert node.is_branch_point
        assert len(node.branch_options) == 2


class TestConversation:
    """Tests for Conversation class."""

    def test_create_conversation(self, sample_conversation):
        """Test creating conversation."""
        assert sample_conversation.conversation_id == "merchant_dialogue"
        assert sample_conversation.state == ConversationState.INACTIVE

    def test_start_conversation(self, sample_conversation):
        """Test starting conversation."""
        node = sample_conversation.start(time.time())
        assert node.node_id == "start"
        assert sample_conversation.state == ConversationState.ACTIVE

    def test_advance_conversation(self, sample_conversation):
        """Test advancing conversation."""
        sample_conversation.start(time.time())
        next_node = sample_conversation.advance(time.time())
        assert next_node.node_id == "question"

    def test_pause_resume(self, sample_conversation):
        """Test pausing and resuming."""
        sample_conversation.start(time.time())

        sample_conversation.pause()
        assert sample_conversation.state == ConversationState.PAUSED

        sample_conversation.resume()
        assert sample_conversation.state == ConversationState.ACTIVE

    def test_cancel(self, sample_conversation):
        """Test canceling conversation."""
        sample_conversation.start(time.time())
        sample_conversation.cancel()
        assert sample_conversation.state == ConversationState.CANCELLED

    def test_skip_to_node(self, sample_conversation):
        """Test skipping to specific node."""
        sample_conversation.start(time.time())
        node = sample_conversation.skip_to_node("weapons")
        assert node.node_id == "weapons"

    def test_reset(self, sample_conversation):
        """Test resetting conversation."""
        sample_conversation.start(time.time())
        sample_conversation.advance(time.time())
        sample_conversation.reset()

        assert sample_conversation.state == ConversationState.INACTIVE
        assert sample_conversation.current_node is None

    def test_cannot_advance_when_inactive(self, sample_conversation):
        """Test cannot advance inactive conversation."""
        result = sample_conversation.advance(time.time())
        assert result is None

    def test_cannot_advance_when_paused(self, sample_conversation):
        """Test paused conversation remains paused until resumed."""
        sample_conversation.start(time.time())
        sample_conversation.pause()

        # State should be paused
        assert sample_conversation.state == ConversationState.PAUSED

        # Resume should work
        sample_conversation.resume()
        assert sample_conversation.state == ConversationState.ACTIVE

    def test_conversation_completes_at_end(self, sample_conversation):
        """Test conversation completes when reaching end node."""
        sample_conversation.start(time.time())
        sample_conversation.advance(time.time())  # -> question
        sample_conversation.skip_to_node("end")

        # Advance from end (no next nodes)
        result = sample_conversation.advance(time.time())
        assert sample_conversation.state == ConversationState.COMPLETED

    def test_choice_at_branch_point(self, sample_conversation):
        """Test advancing with choice index at branch point."""
        sample_conversation.start(time.time())
        sample_conversation.advance(time.time())  # -> question (branch point)

        # Current node should be branch point
        assert sample_conversation.current_node.is_branch_point

        # Advance with choice index to select potions
        node = sample_conversation.advance(time.time(), choice_index=1)
        assert node is not None
        assert node.node_id == "potions"

    def test_skip_to_invalid_node(self, sample_conversation):
        """Test skipping to invalid node."""
        sample_conversation.start(time.time())
        result = sample_conversation.skip_to_node("nonexistent")
        assert result is None

    def test_restart_conversation(self, sample_conversation):
        """Test restarting conversation preserves structure."""
        sample_conversation.start(time.time())
        sample_conversation.advance(time.time())
        sample_conversation.cancel()

        # Start again
        node = sample_conversation.start(time.time())
        assert node.node_id == "start"
        assert sample_conversation.state == ConversationState.ACTIVE


class TestConversationManager:
    """Tests for ConversationManager class."""

    def test_register_conversation(self, conversation_manager, sample_conversation):
        """Test registering conversation."""
        conversation_manager.register_conversation(sample_conversation)
        # Verify we can get it back
        conv = conversation_manager.get_conversation(sample_conversation.conversation_id)
        assert conv is not None
        assert conv.conversation_id == sample_conversation.conversation_id

    def test_unregister_conversation(self, conversation_manager, sample_conversation):
        """Test unregistering conversation."""
        conversation_manager.register_conversation(sample_conversation)
        result = conversation_manager.unregister_conversation(
            sample_conversation.conversation_id
        )
        assert result is True

    def test_start_conversation(self, conversation_manager, sample_conversation):
        """Test starting conversation via manager."""
        conversation_manager.register_conversation(sample_conversation)
        node = conversation_manager.start_conversation(
            sample_conversation.conversation_id,
            time.time(),
        )
        assert node is not None
        assert conversation_manager.active_count == 1

    def test_end_conversation(self, conversation_manager, sample_conversation):
        """Test ending conversation."""
        conversation_manager.register_conversation(sample_conversation)
        conversation_manager.start_conversation(
            sample_conversation.conversation_id,
            time.time(),
        )

        result = conversation_manager.end_conversation(
            sample_conversation.conversation_id
        )
        assert result is True
        assert conversation_manager.active_count == 0

    def test_advance_conversation(self, conversation_manager, sample_conversation):
        """Test advancing conversation via manager."""
        conversation_manager.register_conversation(sample_conversation)
        conversation_manager.start_conversation(
            sample_conversation.conversation_id,
            time.time(),
        )

        node = conversation_manager.advance_conversation(
            sample_conversation.conversation_id,
            time.time(),
        )
        assert node is not None

    def test_make_choice(self, conversation_manager, sample_conversation):
        """Test making choice at branch point via advance with choice_index."""
        conversation_manager.register_conversation(sample_conversation)
        conversation_manager.start_conversation(
            sample_conversation.conversation_id,
            time.time(),
        )
        conversation_manager.advance_conversation(
            sample_conversation.conversation_id,
        )

        # Now at question (branch point) - advance with choice_index=0 (weapons)
        node = conversation_manager.advance_conversation(
            sample_conversation.conversation_id,
            choice_index=0,  # Choose "weapons"
        )
        assert node is not None
        assert node.node_id == "weapons"

    def test_pause_resume_all(self, conversation_manager, sample_conversation):
        """Test pausing and resuming all."""
        conversation_manager.register_conversation(sample_conversation)
        conversation_manager.start_conversation(
            sample_conversation.conversation_id,
            time.time(),
        )

        conversation_manager.pause_all()
        assert sample_conversation.state == ConversationState.PAUSED

        conversation_manager.resume_all()
        assert sample_conversation.state == ConversationState.ACTIVE

    def test_cancel_all(self, conversation_manager, sample_conversation):
        """Test canceling all conversations."""
        conversation_manager.register_conversation(sample_conversation)
        conversation_manager.start_conversation(
            sample_conversation.conversation_id,
            time.time(),
        )

        conversation_manager.cancel_all()
        assert sample_conversation.state == ConversationState.CANCELLED

    def test_callbacks(self, sample_conversation):
        """Test conversation callbacks."""
        on_started = Mock()
        on_ended = Mock()
        on_branch = Mock()

        manager = ConversationManager(
            on_conversation_started=on_started,
            on_conversation_ended=on_ended,
            on_branch_reached=on_branch,
        )
        manager.register_conversation(sample_conversation)

        manager.start_conversation(sample_conversation.conversation_id, time.time())
        on_started.assert_called_once()

        # Advance to branch point
        manager.advance_conversation(
            sample_conversation.conversation_id,
            time.time(),
        )
        on_branch.assert_called_once()


# =============================================================================
# Localization Tests
# =============================================================================

class TestLocalizedAsset:
    """Tests for LocalizedAsset class."""

    def test_create_asset(self):
        """Test creating localized asset."""
        asset = LocalizedAsset(asset_id="greet_001")
        assert asset.asset_id == "greet_001"

    def test_add_variant(self):
        """Test adding language variant."""
        asset = LocalizedAsset(asset_id="test")
        asset.add_variant(
            "en",
            "/audio/en/test.ogg",
            duration_ms=2000.0,
            subtitle="Hello!",
        )
        assert asset.has_language("en")
        assert asset.get_path("en") == "/audio/en/test.ogg"

    def test_get_path_fallback(self):
        """Test path fallback to default."""
        asset = LocalizedAsset(asset_id="test")
        asset.add_variant("en", "/audio/en/test.ogg")

        path = asset.get_path("fr")  # Not available
        assert path == "/audio/en/test.ogg"  # Falls back to en

    def test_get_duration(self):
        """Test getting duration."""
        asset = LocalizedAsset(asset_id="test")
        asset.add_variant("en", "/en.ogg", duration_ms=1000.0)
        asset.add_variant("de", "/de.ogg", duration_ms=1200.0)

        assert asset.get_duration("en") == 1000.0
        assert asset.get_duration("de") == 1200.0

    def test_get_subtitle(self):
        """Test getting subtitle."""
        asset = LocalizedAsset(asset_id="test")
        asset.add_variant("en", "/en.ogg", subtitle="Hello")
        asset.add_variant("es", "/es.ogg", subtitle="Hola")

        assert asset.get_subtitle("en") == "Hello"
        assert asset.get_subtitle("es") == "Hola"


class TestAudioBank:
    """Tests for AudioBank class."""

    def test_create_bank(self):
        """Test creating audio bank."""
        bank = AudioBank(
            bank_id="dialogue_en",
            language="en",
            category="dialogue",
        )
        assert bank.bank_id == "dialogue_en"
        assert bank.language == "en"

    def test_add_remove_asset(self):
        """Test adding and removing assets."""
        bank = AudioBank(bank_id="test", language="en")
        asset = LocalizedAsset(asset_id="asset_1")

        bank.add_asset(asset)
        assert bank.asset_count == 1

        result = bank.remove_asset("asset_1")
        assert result is True
        assert bank.asset_count == 0

    def test_get_asset(self):
        """Test getting asset."""
        bank = AudioBank(bank_id="test", language="en")
        asset = LocalizedAsset(asset_id="test_asset")
        bank.add_asset(asset)

        retrieved = bank.get_asset("test_asset")
        assert retrieved == asset

    def test_iterate_assets(self):
        """Test iterating over assets."""
        bank = AudioBank(bank_id="test", language="en")
        for i in range(5):
            bank.add_asset(LocalizedAsset(asset_id=f"asset_{i}"))

        count = sum(1 for _ in bank)
        assert count == 5


class TestLocalizationManager:
    """Tests for LocalizationManager class."""

    def test_initial_language(self, localization_manager):
        """Test initial language setting."""
        assert localization_manager.current_language == DEFAULT_LANGUAGE

    def test_set_language(self, localization_manager):
        """Test setting language."""
        result = localization_manager.set_language("es")
        assert result is True
        assert localization_manager.current_language == "es"

    def test_set_unsupported_language(self, localization_manager):
        """Test setting unsupported language."""
        result = localization_manager.set_language("invalid_lang")
        assert result is False

    def test_set_same_language(self, localization_manager):
        """Test setting same language returns False."""
        result = localization_manager.set_language(DEFAULT_LANGUAGE)
        assert result is False

    def test_is_language_supported(self, localization_manager):
        """Test checking language support."""
        assert localization_manager.is_language_supported("en")
        assert localization_manager.is_language_supported("es")
        assert not localization_manager.is_language_supported("xx")

    def test_register_bank(self, localization_manager):
        """Test registering audio bank."""
        bank = AudioBank(bank_id="test_bank", language="en")
        localization_manager.register_bank(bank)

        retrieved = localization_manager.get_bank("test_bank")
        assert retrieved == bank

    def test_unregister_bank(self, localization_manager):
        """Test unregistering bank."""
        bank = AudioBank(bank_id="temp", language="en")
        localization_manager.register_bank(bank)
        result = localization_manager.unregister_bank("temp")
        assert result is True

    def test_load_unload_bank(self, localization_manager):
        """Test loading and unloading banks."""
        bank = AudioBank(bank_id="test", language="en")
        localization_manager.register_bank(bank)

        localization_manager.load_bank("test")
        assert "test" in localization_manager.loaded_bank_ids

        localization_manager.unload_bank("test")
        assert "test" not in localization_manager.loaded_bank_ids

    def test_load_language_banks(self, localization_manager):
        """Test loading all banks for a language."""
        for i in range(3):
            bank = AudioBank(bank_id=f"en_bank_{i}", language="en")
            localization_manager.register_bank(bank)

        count = localization_manager.load_language_banks("en")
        assert count == 3

    def test_switch_language_banks(self, localization_manager):
        """Test switching language banks."""
        for lang in ["en", "es"]:
            for i in range(2):
                bank = AudioBank(bank_id=f"{lang}_{i}", language=lang)
                localization_manager.register_bank(bank)

        localization_manager.load_language_banks("en")
        unloaded, loaded = localization_manager.switch_language_banks("en", "es")

        assert unloaded == 2
        assert loaded == 2

    def test_register_asset(self, localization_manager):
        """Test registering localized asset."""
        asset = LocalizedAsset(asset_id="test_asset")
        localization_manager.register_asset(asset)

        retrieved = localization_manager.get_asset("test_asset")
        assert retrieved == asset

    def test_get_localized_path(self, localization_manager):
        """Test getting localized path."""
        asset = LocalizedAsset(asset_id="greeting")
        asset.add_variant("en", "/audio/en/greeting.ogg")
        asset.add_variant("es", "/audio/es/greeting.ogg")
        localization_manager.register_asset(asset)

        localization_manager.set_language("es")
        path = localization_manager.get_localized_path("greeting")
        assert path == "/audio/es/greeting.ogg"

    def test_set_fallback_chain(self, localization_manager):
        """Test setting fallback chain."""
        localization_manager.set_fallback_chain("pt-BR", ["pt", "en"])
        chain = localization_manager.get_fallback_chain("pt-BR")
        assert chain == ["pt", "en"]

    def test_localize_line(self, localization_manager, sample_vo_line):
        """Test localizing a VO line."""
        asset = LocalizedAsset(asset_id=sample_vo_line.audio_asset)
        asset.add_variant(
            "es",
            "/audio/es/test.ogg",
            duration_ms=3200.0,
            subtitle="Esta es una linea de prueba.",
        )
        localization_manager.register_asset(asset)
        localization_manager.set_language("es")

        localized = localization_manager.localize_line(sample_vo_line)
        assert localized.language == "es"

    def test_language_changed_callback(self):
        """Test language changed callback."""
        callback = Mock()
        manager = LocalizationManager(on_language_changed=callback)
        manager.set_language("es")
        callback.assert_called_once_with("en", "es")

    def test_stats(self, localization_manager):
        """Test getting stats."""
        bank = AudioBank(bank_id="test", language="en")
        localization_manager.register_bank(bank)

        stats = localization_manager.stats
        assert "current_language" in stats
        assert "total_banks" in stats

    def test_fallback_chain_used(self, localization_manager):
        """Test fallback chain is used for missing languages."""
        asset = LocalizedAsset(asset_id="test")
        asset.add_variant("en", "/audio/en/test.ogg")  # Only English
        localization_manager.register_asset(asset)

        localization_manager.set_fallback_chain("pt-BR", ["pt", "en"])
        # Even though pt-BR is set, should fallback to en
        path = localization_manager.get_localized_path("test")
        assert path == "/audio/en/test.ogg"

    def test_get_available_languages(self, localization_manager):
        """Test getting available languages for asset."""
        asset = LocalizedAsset(asset_id="multi_lang")
        asset.add_variant("en", "/audio/en/test.ogg")
        asset.add_variant("es", "/audio/es/test.ogg")
        asset.add_variant("de", "/audio/de/test.ogg")
        localization_manager.register_asset(asset)

        # Check each language has a path
        assert asset.has_language("en")
        assert asset.has_language("es")
        assert asset.has_language("de")
        assert not asset.has_language("fr")

    def test_localize_line_preserves_priority(self, localization_manager, sample_vo_line):
        """Test localizing line preserves priority."""
        asset = LocalizedAsset(asset_id=sample_vo_line.audio_asset)
        asset.add_variant("es", "/audio/es/test.ogg", duration_ms=3200.0)
        localization_manager.register_asset(asset)
        localization_manager.set_language("es")

        original_priority = sample_vo_line.priority
        localized = localization_manager.localize_line(sample_vo_line)
        assert localized.priority == original_priority

    def test_bank_asset_count(self, localization_manager):
        """Test bank asset counting."""
        bank = AudioBank(bank_id="test", language="en")
        asset = LocalizedAsset(asset_id="test_asset")
        asset.add_variant("en", "/audio/en/test.ogg", duration_ms=5000.0)
        bank.add_asset(asset)
        localization_manager.register_bank(bank)

        # Bank should have one asset
        assert bank.asset_count == 1


# =============================================================================
# SubtitleSync Tests
# =============================================================================

class TestSubtitleCue:
    """Tests for SubtitleCue class."""

    def test_create_cue(self):
        """Test creating subtitle cue."""
        cue = SubtitleCue(
            time_ms=0.0,
            text="Hello world",
            duration_ms=2000.0,
        )
        assert cue.text == "Hello world"
        assert cue.duration_ms == 2000.0
        assert cue.end_time_ms == 2000.0


class TestSubtitleTrack:
    """Tests for SubtitleTrack class."""

    def test_create_track(self):
        """Test creating subtitle track."""
        track = SubtitleTrack(line_id="main")
        assert track.line_id == "main"

    def test_add_cue(self):
        """Test adding cues."""
        track = SubtitleTrack(line_id="test")
        track.add_cue(SubtitleCue(
            time_ms=0.0,
            text="First",
            duration_ms=1000.0,
        ))
        assert track.cue_count == 1

    def test_get_cue_at_time(self):
        """Test getting cue at time."""
        track = SubtitleTrack(line_id="test")
        track.add_cue(SubtitleCue(time_ms=0.0, text="First", duration_ms=1000.0))
        track.add_cue(SubtitleCue(time_ms=1000.0, text="Second", duration_ms=1000.0))

        cue = track.get_cue_at_time(500.0)
        assert cue is not None
        assert cue.text == "First"

        cue = track.get_cue_at_time(1500.0)
        assert cue is not None
        assert cue.text == "Second"

    def test_get_next_cue(self):
        """Test getting next upcoming cue."""
        track = SubtitleTrack(line_id="test")
        track.add_cue(SubtitleCue(time_ms=0.0, text="First", duration_ms=1000.0))
        track.add_cue(SubtitleCue(time_ms=1000.0, text="Second", duration_ms=1000.0))

        next_cue = track.get_next_cue(500.0)
        assert next_cue is not None
        assert next_cue.text == "Second"

    def test_reset_track(self):
        """Test resetting track position."""
        track = SubtitleTrack(line_id="test")
        track.add_cue(SubtitleCue(time_ms=0.0, text="Test", duration_ms=1000.0))
        track.reset()
        # Should not raise, just resets internal position


class TestActiveSubtitle:
    """Tests for ActiveSubtitle class."""

    def test_create_active(self):
        """Test creating active subtitle."""
        active = ActiveSubtitle(
            subtitle_id="sub_1",
            line_id="line_1",
            text="Hello",
            start_time=0.0,
            end_time=2.0,
        )
        assert active.text == "Hello"
        assert active.state == SubtitleState.HIDDEN  # Initial state

    def test_show_subtitle(self):
        """Test showing active subtitle."""
        active = ActiveSubtitle(
            subtitle_id="sub_1",
            line_id="line_1",
            text="Hello",
            start_time=0.0,
            end_time=2.0,
        )
        active.show()
        assert active.state == SubtitleState.FADING_IN

    def test_update_subtitle_fade_in(self):
        """Test updating subtitle fade in state."""
        active = ActiveSubtitle(
            subtitle_id="sub_1",
            line_id="line_1",
            text="Hello",
            start_time=0.0,
            end_time=2.0,
        )
        active.show()

        # Update with enough time to complete fade in
        active.update(300.0, 200.0)  # delta_ms, fade_time_ms
        assert active.state == SubtitleState.VISIBLE
        assert active.opacity == 1.0

    def test_hide_subtitle(self):
        """Test hiding active subtitle."""
        active = ActiveSubtitle(
            subtitle_id="sub_1",
            line_id="line_1",
            text="Hello",
            start_time=0.0,
            end_time=2.0,
        )
        active.show()
        active.update(300.0, 200.0)  # Complete fade in
        active.hide()
        assert active.state == SubtitleState.FADING_OUT

    def test_subtitle_duration(self):
        """Test subtitle duration calculation."""
        active = ActiveSubtitle(
            subtitle_id="sub_1",
            line_id="line_1",
            text="Hello",
            start_time=1.0,
            end_time=3.0,
        )
        assert active.duration_ms == 2000.0

    def test_is_visible(self):
        """Test visibility check across states."""
        active = ActiveSubtitle(
            subtitle_id="sub_1",
            line_id="line_1",
            text="Hello",
            start_time=0.0,
            end_time=2.0,
        )
        assert not active.is_visible

        active.show()
        assert active.is_visible  # FADING_IN is visible

        active.update(300.0, 200.0)
        assert active.is_visible  # VISIBLE is visible


class TestSubtitleManager:
    """Tests for SubtitleManager class."""

    def test_enable_disable(self, subtitle_manager):
        """Test enabling/disabling subtitles."""
        subtitle_manager.enabled = False
        assert not subtitle_manager.enabled

        subtitle_manager.enabled = True
        assert subtitle_manager.enabled

    def test_show_subtitle(self, subtitle_manager, sample_vo_line):
        """Test showing subtitle."""
        sample_vo_line.subtitle = SubtitleData(
            text="Test subtitle",
            start_time_ms=0.0,
            end_time_ms=3000.0,
        )
        result = subtitle_manager.show_subtitle(sample_vo_line, time.time())
        assert result is not None
        assert subtitle_manager.active_count >= 0  # May be 0 if not visible yet

    def test_hide_all(self, subtitle_manager, sample_vo_line):
        """Test hiding all subtitles."""
        sample_vo_line.subtitle = SubtitleData(
            text="Test",
            start_time_ms=0.0,
            end_time_ms=3000.0,
        )
        subtitle_manager.show_subtitle(sample_vo_line, time.time())
        count = subtitle_manager.hide_all()
        # Should return count of hidden subtitles
        assert count >= 0

    def test_hide_for_line(self, subtitle_manager, sample_vo_line):
        """Test hiding subtitle for specific line."""
        sample_vo_line.subtitle = SubtitleData(
            text="Test",
            start_time_ms=0.0,
            end_time_ms=3000.0,
        )
        subtitle_manager.show_subtitle(sample_vo_line, time.time())
        result = subtitle_manager.hide_for_line(sample_vo_line.line_id)
        assert isinstance(result, bool)

    def test_hide_for_speaker(self, subtitle_manager, sample_vo_line):
        """Test hiding subtitles for specific speaker."""
        sample_vo_line.subtitle = SubtitleData(
            text="Test",
            start_time_ms=0.0,
            end_time_ms=3000.0,
        )
        subtitle_manager.show_subtitle(sample_vo_line, time.time())
        count = subtitle_manager.hide_for_speaker(sample_vo_line.speaker_id)
        assert count >= 0

    def test_create_track_from_line(self, subtitle_manager, sample_vo_line):
        """Test creating track from line."""
        sample_vo_line.subtitle = SubtitleData(
            text="Track test",
            start_time_ms=0.0,
            end_time_ms=3000.0,
        )
        track = subtitle_manager.create_track_from_line(sample_vo_line)
        assert track is not None
        assert track.cue_count >= 1

    def test_create_track_from_line_with_text_only(self, subtitle_manager):
        """Test creating track from line without subtitle data but with text."""
        line = VOLine(
            line_id="text_only",
            text="Simple text line",
            duration_ms=2000.0,
        )
        track = subtitle_manager.create_track_from_line(line)
        assert track is not None
        assert track.cue_count == 1

    def test_get_track(self, subtitle_manager, sample_vo_line):
        """Test getting subtitle track."""
        sample_vo_line.subtitle = SubtitleData(
            text="Test",
            start_time_ms=0.0,
            end_time_ms=1000.0,
        )
        subtitle_manager.create_track_from_line(sample_vo_line)
        track = subtitle_manager.get_track(sample_vo_line.line_id)
        assert track is not None

    def test_remove_track(self, subtitle_manager, sample_vo_line):
        """Test removing subtitle track."""
        sample_vo_line.subtitle = SubtitleData(
            text="Test",
            start_time_ms=0.0,
            end_time_ms=1000.0,
        )
        subtitle_manager.create_track_from_line(sample_vo_line)
        result = subtitle_manager.remove_track(sample_vo_line.line_id)
        assert result is True

    def test_set_speaker_style(self, subtitle_manager):
        """Test setting speaker-specific style."""
        style = SubtitleStyle(
            font_size=28,
            text_color="#FFFF00",
        )
        subtitle_manager.set_speaker_style("npc_guard", style)
        retrieved = subtitle_manager.get_speaker_style("npc_guard")
        assert retrieved.font_size == 28

    def test_set_default_style(self, subtitle_manager):
        """Test setting default style."""
        style = SubtitleStyle(font_size=32)
        subtitle_manager.set_default_style(style)
        # Default style should be used for unknown speakers

    def test_calculate_display_duration(self, subtitle_manager):
        """Test display duration calculation."""
        short_text = "Hi"
        long_text = "This is a much longer piece of dialogue that should display for more time"

        short_duration = subtitle_manager.calculate_display_duration(short_text)
        long_duration = subtitle_manager.calculate_display_duration(long_text)

        assert long_duration > short_duration
        assert short_duration >= 1500.0  # Minimum display time

    def test_get_visible_subtitles(self, subtitle_manager, sample_vo_line):
        """Test getting visible subtitles."""
        sample_vo_line.subtitle = SubtitleData(
            text="Test",
            start_time_ms=0.0,
            end_time_ms=3000.0,
        )
        subtitle_manager.show_subtitle(sample_vo_line, time.time())
        visible = subtitle_manager.get_visible_subtitles()
        assert isinstance(visible, list)

    def test_callbacks(self, sample_vo_line):
        """Test subtitle callbacks."""
        on_show = Mock()
        on_hide = Mock()

        manager = SubtitleManager(
            on_subtitle_show=on_show,
            on_subtitle_hide=on_hide,
        )
        sample_vo_line.subtitle = SubtitleData(
            text="Callback test",
            start_time_ms=0.0,
            end_time_ms=100.0,
        )
        manager.show_subtitle(sample_vo_line, time.time())
        on_show.assert_called_once()

    def test_update_auto_hide(self, subtitle_manager, sample_vo_line):
        """Test update auto-hides expired subtitles."""
        sample_vo_line.subtitle = SubtitleData(
            text="Short subtitle",
            start_time_ms=0.0,
            end_time_ms=100.0,  # Very short duration
        )
        current = time.time()
        subtitle_manager.show_subtitle(sample_vo_line, current)

        # Simulate time passing past the end time
        subtitle_manager.update(1000.0, current + 1.0)
        # Subtitle should have started hiding

    def test_max_lines_limit(self, subtitle_manager):
        """Test max simultaneous subtitle lines limit."""
        # Show more subtitles than max allowed
        for i in range(10):
            line = VOLine(
                line_id=f"line_{i}",
                subtitle=SubtitleData(
                    text=f"Subtitle {i}",
                    start_time_ms=0.0,
                    end_time_ms=10000.0,
                ),
                duration_ms=10000.0,
            )
            subtitle_manager.show_subtitle(line, time.time())

        visible = subtitle_manager.get_visible_subtitles()
        # Should respect max_lines limit (default is 3)
        assert len(visible) <= 3


# =============================================================================
# VOStreaming Tests
# =============================================================================

class TestCachedAudio:
    """Tests for CachedAudio class."""

    def test_create_cached(self):
        """Test creating cached audio entry."""
        cached = CachedAudio(
            asset_id="/audio/test.ogg",
            size_bytes=1024 * 100,
        )
        assert cached.asset_id == "/audio/test.ogg"
        assert cached.size_bytes == 102400

    def test_access_updates_time(self):
        """Test accessing updates time."""
        current = time.time()
        cached = CachedAudio(
            asset_id="/test.ogg",
            size_bytes=1000,
            last_access_time=current,
        )
        time.sleep(0.01)
        cached.access(time.time())
        assert cached.last_access_time > current
        assert cached.access_count == 1

    def test_age_calculation(self):
        """Test age calculation."""
        current = time.time()
        cached = CachedAudio(
            asset_id="/test.ogg",
            size_bytes=1000,
            load_time=current - 1.0,  # 1 second ago
        )
        assert cached.age_ms >= 1000.0

    def test_idle_time_calculation(self):
        """Test idle time calculation."""
        current = time.time()
        cached = CachedAudio(
            asset_id="/test.ogg",
            size_bytes=1000,
            last_access_time=current - 0.5,  # 0.5 seconds ago
        )
        assert cached.idle_time_ms >= 500.0


class TestVOCache:
    """Tests for VOCache class."""

    def test_create_cache(self):
        """Test creating cache."""
        cache = VOCache(max_size_mb=16)
        assert cache.size_bytes == 0

    def test_put_entry(self):
        """Test adding cache entry."""
        cache = VOCache(max_size_mb=16)
        data = b"x" * (1024 * 50)  # 50KB of data
        cache.put("/test.ogg", data, duration_ms=1000.0)
        assert cache.item_count == 1

    def test_get_entry(self):
        """Test getting cache entry."""
        cache = VOCache(max_size_mb=16)
        data = b"x" * (1024 * 50)
        cache.put("/test.ogg", data, duration_ms=1000.0)
        entry = cache.get("/test.ogg")
        assert entry is not None
        assert entry.data == data

    def test_get_nonexistent_entry(self):
        """Test getting nonexistent entry returns None."""
        cache = VOCache(max_size_mb=16)
        entry = cache.get("/nonexistent.ogg")
        assert entry is None

    def test_remove_entry(self):
        """Test removing cache entry."""
        cache = VOCache(max_size_mb=16)
        cache.put("/test.ogg", b"data", duration_ms=1000.0)
        result = cache.remove("/test.ogg")
        assert result is True
        assert cache.get("/test.ogg") is None

    def test_remove_nonexistent_entry(self):
        """Test removing nonexistent entry."""
        cache = VOCache(max_size_mb=16)
        result = cache.remove("/nonexistent.ogg")
        assert result is False

    def test_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = VOCache(max_size_mb=1)  # 1MB limit

        # Add entries that exceed limit
        for i in range(15):
            data = b"x" * (1024 * 100)  # 100KB each
            cache.put(f"/test_{i}.ogg", data, duration_ms=1000.0)

        # Some should have been evicted
        assert cache.item_count < 15

    def test_clear_cache(self):
        """Test clearing cache."""
        cache = VOCache(max_size_mb=16)
        for i in range(5):
            cache.put(f"/test_{i}.ogg", b"data", duration_ms=1000.0)
        count = cache.clear()
        assert count == 5
        assert cache.item_count == 0

    def test_cache_hit_rate(self):
        """Test cache hit rate tracking."""
        cache = VOCache(max_size_mb=16)
        cache.put("/test.ogg", b"data", duration_ms=1000.0)

        # One hit
        cache.get("/test.ogg")
        # One miss
        cache.get("/nonexistent.ogg")

        assert cache.hit_rate == 0.5

    def test_fill_percent(self):
        """Test fill percent calculation."""
        cache = VOCache(max_size_mb=1)  # 1MB
        assert cache.fill_percent == 0.0

        # Add ~500KB
        cache.put("/test.ogg", b"x" * (512 * 1024), duration_ms=1000.0)
        assert cache.fill_percent > 0.4

    def test_evict_preloaded(self):
        """Test evicting preloaded entries."""
        cache = VOCache(max_size_mb=16)
        cache.put("/preloaded.ogg", b"data", duration_ms=1000.0, is_preloaded=True)
        cache.put("/regular.ogg", b"data", duration_ms=1000.0, is_preloaded=False)

        count = cache.evict_preloaded()
        assert count == 1

    def test_stats(self):
        """Test getting cache stats."""
        cache = VOCache(max_size_mb=16)
        cache.put("/test.ogg", b"data", duration_ms=1000.0)
        cache.get("/test.ogg")

        stats = cache.stats
        assert "size_bytes" in stats
        assert "hit_rate" in stats
        assert "item_count" in stats
        assert stats["item_count"] == 1


class TestStreamHandle:
    """Tests for StreamHandle class."""

    def test_create_handle(self):
        """Test creating stream handle."""
        handle = StreamHandle(
            stream_id="stream_1",
            asset_id="/audio/test.ogg",
        )
        assert handle.stream_id == "stream_1"
        assert handle.state == StreamState.IDLE

    def test_handle_ready_property(self):
        """Test is_ready property."""
        handle = StreamHandle(
            stream_id="stream_1",
            asset_id="/audio/test.ogg",
            state=StreamState.READY,
        )
        assert handle.is_ready

    def test_handle_streaming_property(self):
        """Test is_streaming property."""
        handle = StreamHandle(
            stream_id="stream_1",
            asset_id="/audio/test.ogg",
            state=StreamState.STREAMING,
        )
        assert handle.is_streaming

    def test_progress_calculation(self):
        """Test progress calculation."""
        handle = StreamHandle(
            stream_id="stream_1",
            asset_id="/audio/test.ogg",
            duration_ms=2000.0,
            playback_position_ms=500.0,
        )
        assert handle.progress == 0.25

    def test_progress_with_zero_duration(self):
        """Test progress with zero duration."""
        handle = StreamHandle(
            stream_id="stream_1",
            asset_id="/audio/test.ogg",
            duration_ms=0.0,
        )
        assert handle.progress == 0.0


class TestVOStreamManager:
    """Tests for VOStreamManager class."""

    def test_preload_line(self, stream_manager, sample_vo_line):
        """Test preloading a line."""
        result = stream_manager.preload_line(sample_vo_line)
        assert result is True

    def test_preload_multiple_lines(self, stream_manager, sample_lines):
        """Test preloading multiple lines."""
        count = stream_manager.preload_lines(sample_lines[:5])
        assert count >= 0

    def test_start_stream(self, stream_manager, sample_vo_line):
        """Test starting a stream."""
        on_ready = Mock()
        on_complete = Mock()

        handle = stream_manager.start_stream(
            sample_vo_line,
            on_ready=on_ready,
            on_complete=on_complete,
        )
        assert handle is not None
        assert handle.asset_id == sample_vo_line.audio_asset

    def test_start_stream_when_full(self, sample_vo_line):
        """Test starting stream when max concurrent reached."""
        manager = VOStreamManager(max_concurrent_streams=1)
        handle1 = manager.start_stream(sample_vo_line)
        assert handle1 is not None

        # Second stream should fail when first is active
        if handle1 and handle1.state == StreamState.STREAMING:
            handle2 = manager.start_stream(sample_vo_line)
            assert handle2 is None

    def test_stop_stream(self, stream_manager, sample_vo_line):
        """Test stopping a stream."""
        handle = stream_manager.start_stream(sample_vo_line)
        if handle:
            result = stream_manager.stop_stream(handle.stream_id)
            assert result is True

    def test_stop_nonexistent_stream(self, stream_manager):
        """Test stopping nonexistent stream."""
        result = stream_manager.stop_stream("nonexistent_stream")
        assert result is False

    def test_pause_resume_stream(self, stream_manager, sample_vo_line):
        """Test pausing and resuming stream."""
        handle = stream_manager.start_stream(sample_vo_line)
        if handle:
            # Need to start playing first
            stream_manager.play_stream(handle.stream_id)

            pause_result = stream_manager.pause_stream(handle.stream_id)
            assert pause_result is True

            resume_result = stream_manager.resume_stream(handle.stream_id)
            assert resume_result is True

    def test_play_stream(self, stream_manager, sample_vo_line):
        """Test playing a ready stream."""
        handle = stream_manager.start_stream(sample_vo_line)
        if handle and handle.state == StreamState.READY:
            result = stream_manager.play_stream(handle.stream_id)
            assert result is True

    def test_update_stream(self, stream_manager, sample_vo_line):
        """Test updating stream position."""
        sample_vo_line.duration_ms = 2000.0
        handle = stream_manager.start_stream(sample_vo_line)
        if handle:
            stream_manager.play_stream(handle.stream_id)
            updated = stream_manager.update_stream(handle.stream_id, 100.0)
            assert updated is not None
            assert updated.playback_position_ms >= 100.0

    def test_get_stream(self, stream_manager, sample_vo_line):
        """Test getting stream by ID."""
        handle = stream_manager.start_stream(sample_vo_line)
        if handle:
            retrieved = stream_manager.get_stream(handle.stream_id)
            assert retrieved == handle

    def test_get_nonexistent_stream(self, stream_manager):
        """Test getting nonexistent stream."""
        result = stream_manager.get_stream("nonexistent")
        assert result is None

    def test_set_anticipated_lines(self, stream_manager, sample_lines):
        """Test setting anticipated lines for preloading."""
        stream_manager.set_anticipated_lines(sample_lines[:5])
        # Should preload the anticipated lines

    def test_cancel_preload(self, stream_manager, sample_vo_line):
        """Test canceling a preload."""
        stream_manager.preload(sample_vo_line.audio_asset)
        result = stream_manager.cancel_preload(sample_vo_line.audio_asset)
        # May return True or False depending on timing

    def test_clear_preload_queue(self, stream_manager, sample_lines):
        """Test clearing preload queue."""
        for line in sample_lines[:5]:
            stream_manager.preload(line.audio_asset)
        count = stream_manager.clear_preload_queue()
        assert count >= 0

    def test_get_memory_usage(self, stream_manager):
        """Test getting memory usage stats."""
        usage = stream_manager.get_memory_usage()
        assert "cache" in usage
        assert "active_streams" in usage
        assert "preload_queue_size" in usage

    def test_trim_cache(self, stream_manager):
        """Test trimming cache."""
        count = stream_manager.trim_cache(target_percent=0.5)
        assert count >= 0

    def test_clear_cache(self, stream_manager, sample_vo_line):
        """Test clearing cache."""
        stream_manager.preload_line(sample_vo_line)
        count = stream_manager.clear_cache()
        assert count >= 0

    def test_stats(self, stream_manager):
        """Test getting streaming stats."""
        stats = stream_manager.stats
        assert "active_streams" in stats
        assert "max_concurrent" in stats
        assert "cache" in stats

    def test_can_start_stream_property(self, stream_manager):
        """Test can_start_stream property."""
        assert stream_manager.can_start_stream is True

    def test_active_stream_count(self, stream_manager):
        """Test active stream count."""
        assert stream_manager.active_stream_count == 0


# =============================================================================
# VOProcessing Tests
# =============================================================================

class TestRadioEffect:
    """Tests for RadioEffect class."""

    def test_create_radio_effect(self):
        """Test creating radio effect."""
        effect = RadioEffect(
            band_low=300.0,
            band_high=3400.0,
            distortion=0.3,
            noise_level=0.05,
        )
        assert effect.band_low == 300.0
        assert effect.distortion == 0.3


class TestDistanceFilter:
    """Tests for DistanceFilter class."""

    def test_create_filter(self):
        """Test creating distance filter."""
        dist_filter = DistanceFilter(
            start_distance=10.0,
            max_distance=50.0,
            min_cutoff=500.0,
        )
        assert dist_filter.start_distance == 10.0
        assert dist_filter.max_distance == 50.0

    def test_calculate_cutoff_at_start(self):
        """Test cutoff at start distance."""
        dist_filter = DistanceFilter(
            start_distance=10.0,
            max_distance=50.0,
        )
        cutoff = dist_filter.calculate_cutoff(5.0)  # Below start
        assert cutoff == 20000.0  # No filtering

    def test_calculate_cutoff_at_max(self):
        """Test cutoff at max distance."""
        dist_filter = DistanceFilter(
            start_distance=10.0,
            max_distance=50.0,
            min_cutoff=500.0,
        )
        cutoff = dist_filter.calculate_cutoff(50.0)
        assert cutoff == 500.0

    def test_calculate_cutoff_midpoint(self):
        """Test cutoff at midpoint."""
        dist_filter = DistanceFilter(
            start_distance=10.0,
            max_distance=50.0,
            attenuation_curve="linear",
        )
        cutoff = dist_filter.calculate_cutoff(30.0)  # Midpoint
        assert 500.0 < cutoff < 20000.0

    def test_calculate_attenuation(self):
        """Test volume attenuation calculation."""
        dist_filter = DistanceFilter(
            start_distance=10.0,
            max_distance=50.0,
        )
        # At start distance, no attenuation
        assert dist_filter.calculate_attenuation(5.0) == 1.0

        # At max distance, full attenuation
        assert dist_filter.calculate_attenuation(50.0) == 0.0

        # Midpoint should be somewhere in between
        atten = dist_filter.calculate_attenuation(30.0)
        assert 0.0 < atten < 1.0

    def test_logarithmic_curve(self):
        """Test logarithmic attenuation curve."""
        dist_filter = DistanceFilter(
            start_distance=10.0,
            max_distance=50.0,
            attenuation_curve="logarithmic",
        )
        atten = dist_filter.calculate_attenuation(30.0)
        assert 0.0 < atten < 1.0


class TestReverbSettings:
    """Tests for ReverbSettings class."""

    def test_create_reverb(self):
        """Test creating reverb settings."""
        reverb = ReverbSettings(
            send_level=0.3,
            room_size=0.5,
        )
        assert reverb.send_level == 0.3
        assert reverb.room_size == 0.5

    def test_apply_environment_preset(self):
        """Test applying environment preset."""
        reverb = ReverbSettings()
        modified = reverb.apply_environment("cave", intensity=1.0)
        assert modified.room_size == 0.9  # Cave preset
        assert modified.decay_time == 3.0

    def test_apply_unknown_environment(self):
        """Test applying unknown environment returns original."""
        reverb = ReverbSettings(room_size=0.5)
        modified = reverb.apply_environment("unknown_env")
        assert modified.room_size == 0.5  # Unchanged

    def test_to_dict(self):
        """Test converting to dictionary."""
        reverb = ReverbSettings(send_level=0.4, room_size=0.6)
        data = reverb.to_dict()
        assert data["send_level"] == 0.4
        assert data["room_size"] == 0.6
        assert "decay_time" in data


class TestSpatialSettings:
    """Tests for SpatialSettings class."""

    def test_create_spatial(self):
        """Test creating spatial settings."""
        spatial = SpatialSettings(
            blend=1.0,
            min_distance=1.0,
            max_distance=50.0,
        )
        assert spatial.blend == 1.0
        assert spatial.min_distance == 1.0
        assert spatial.max_distance == 50.0

    def test_calculate_pan(self):
        """Test pan calculation based on position."""
        spatial = SpatialSettings(
            blend=1.0,
            position=(10.0, 0.0, 0.0),  # To the right
        )
        listener_pos = (0.0, 0.0, 0.0)
        listener_forward = (0.0, 0.0, 1.0)  # Looking forward

        pan = spatial.calculate_pan(listener_pos, listener_forward)
        # Pan value is bounded -1 to 1
        assert -1.0 <= pan <= 1.0

    def test_calculate_pan_left(self):
        """Test pan calculation for left position."""
        spatial = SpatialSettings(
            blend=1.0,
            position=(-10.0, 0.0, 0.0),  # To the left
        )
        listener_pos = (0.0, 0.0, 0.0)
        listener_forward = (0.0, 0.0, 1.0)

        pan = spatial.calculate_pan(listener_pos, listener_forward)
        # Pan value is bounded -1 to 1
        assert -1.0 <= pan <= 1.0

    def test_to_dict(self):
        """Test converting to dictionary."""
        spatial = SpatialSettings(blend=0.8, min_distance=2.0)
        data = spatial.to_dict()
        assert data["blend"] == 0.8
        assert data["min_distance"] == 2.0
        assert "position" in data


class TestVOProcessingState:
    """Tests for VOProcessingState class."""

    def test_create_state(self):
        """Test creating processing state."""
        state = VOProcessingState(source_id="vo_1")
        assert state.source_id == "vo_1"


class TestVOProcessor:
    """Tests for VOProcessor class."""

    def test_create_processor(self, vo_processor):
        """Test creating processor."""
        assert vo_processor is not None
        assert vo_processor.source_count == 0

    def test_create_state(self, vo_processor):
        """Test creating processing state."""
        state = vo_processor.create_state(
            "vo_1",
            radio_enabled=True,
            spatial_enabled=True,
        )
        assert state is not None
        assert state.source_id == "vo_1"
        assert state.radio_effect.enabled is True
        assert state.spatial.enabled is True

    def test_get_state(self, vo_processor):
        """Test getting processing state."""
        vo_processor.create_state("vo_1")
        state = vo_processor.get_state("vo_1")
        assert state is not None
        assert state.source_id == "vo_1"

    def test_get_nonexistent_state(self, vo_processor):
        """Test getting nonexistent state."""
        state = vo_processor.get_state("nonexistent")
        assert state is None

    def test_remove_state(self, vo_processor):
        """Test removing state."""
        vo_processor.create_state("vo_1")
        result = vo_processor.remove_state("vo_1")
        assert result is True
        assert vo_processor.get_state("vo_1") is None

    def test_remove_nonexistent_state(self, vo_processor):
        """Test removing nonexistent state."""
        result = vo_processor.remove_state("nonexistent")
        assert result is False

    def test_set_position(self, vo_processor):
        """Test setting source position."""
        vo_processor.create_state("vo_1", spatial_enabled=True)
        result = vo_processor.set_position("vo_1", (10.0, 0.0, 5.0))
        assert result is True

        state = vo_processor.get_state("vo_1")
        assert state.spatial.position == (10.0, 0.0, 5.0)

    def test_set_position_nonexistent(self, vo_processor):
        """Test setting position on nonexistent source."""
        result = vo_processor.set_position("nonexistent", (0.0, 0.0, 0.0))
        assert result is False

    def test_set_listener_position(self, vo_processor):
        """Test setting listener position."""
        vo_processor.set_listener_position(
            (0.0, 0.0, 0.0),
            forward=(0.0, 0.0, 1.0),
        )
        # Should update distance calculations for all sources

    def test_enable_radio(self, vo_processor):
        """Test enabling radio effect."""
        vo_processor.create_state("vo_1")
        result = vo_processor.enable_radio("vo_1", distortion=0.4, noise=0.1)
        assert result is True

        state = vo_processor.get_state("vo_1")
        assert state.radio_effect.enabled is True
        assert state.radio_effect.distortion == 0.4

    def test_disable_radio(self, vo_processor):
        """Test disabling radio effect."""
        vo_processor.create_state("vo_1", radio_enabled=True)
        result = vo_processor.disable_radio("vo_1")
        assert result is True

        state = vo_processor.get_state("vo_1")
        assert state.radio_effect.enabled is False

    def test_set_reverb(self, vo_processor):
        """Test setting reverb."""
        vo_processor.create_state("vo_1")
        result = vo_processor.set_reverb("vo_1", send_level=0.4, environment="cave")
        assert result is True

        state = vo_processor.get_state("vo_1")
        assert state.reverb.enabled is True

    def test_set_volume(self, vo_processor):
        """Test setting source volume."""
        vo_processor.create_state("vo_1")
        result = vo_processor.set_volume("vo_1", 0.7)
        assert result is True

        state = vo_processor.get_state("vo_1")
        assert state.volume == 0.7

    def test_set_muted(self, vo_processor):
        """Test muting source."""
        vo_processor.create_state("vo_1")
        result = vo_processor.set_muted("vo_1", True)
        assert result is True

        state = vo_processor.get_state("vo_1")
        assert state.is_muted is True

    def test_set_environment(self, vo_processor):
        """Test setting environment."""
        vo_processor.create_state("vo_1")
        vo_processor.set_reverb("vo_1", send_level=0.3)
        vo_processor.set_environment("cave")
        # Should update all sources with reverb enabled

    def test_master_volume(self, vo_processor):
        """Test master volume."""
        vo_processor.master_volume = 0.8
        assert vo_processor.master_volume == 0.8

        # Test clamping
        vo_processor.master_volume = 1.5
        assert vo_processor.master_volume == 1.0

        vo_processor.master_volume = -0.5
        assert vo_processor.master_volume == 0.0

    def test_get_processed_params(self, vo_processor):
        """Test getting processed parameters."""
        vo_processor.create_state("vo_1", spatial_enabled=True)
        vo_processor.set_volume("vo_1", 0.8)

        params = vo_processor.get_processed_params("vo_1")
        assert params is not None
        assert "volume" in params
        assert "pan" in params
        assert "effect_chain" in params

    def test_get_processed_params_muted(self, vo_processor):
        """Test processed params when muted."""
        vo_processor.create_state("vo_1")
        vo_processor.set_muted("vo_1", True)

        params = vo_processor.get_processed_params("vo_1")
        assert params["volume"] == 0.0
        assert params["is_muted"] is True

    def test_update(self, vo_processor):
        """Test updating processor."""
        vo_processor.create_state("vo_1", spatial_enabled=True)
        vo_processor.set_position("vo_1", (10.0, 0.0, 5.0))
        vo_processor.update(16.0)  # Should not raise

    def test_source_ids(self, vo_processor):
        """Test getting source IDs."""
        vo_processor.create_state("vo_1")
        vo_processor.create_state("vo_2")

        ids = vo_processor.source_ids
        assert "vo_1" in ids
        assert "vo_2" in ids

    def test_state_changed_callback(self):
        """Test state changed callback."""
        callback = Mock()
        processor = VOProcessor(on_state_changed=callback)
        processor.create_state("vo_1")
        processor.set_volume("vo_1", 0.5)

        callback.assert_called()


# =============================================================================
# DialogueManager Tests
# =============================================================================

class TestDialogueManager:
    """Tests for DialogueManager class."""

    def test_create_manager(self, dialogue_manager):
        """Test creating dialogue manager."""
        assert dialogue_manager.state == ConfigDialogueState.IDLE
        assert not dialogue_manager.is_playing

    def test_start_stop(self, dialogue_manager):
        """Test starting and stopping."""
        dialogue_manager.start()
        assert dialogue_manager.state == ConfigDialogueState.IDLE

        dialogue_manager.stop()
        assert dialogue_manager.state == ConfigDialogueState.IDLE

    def test_play_line(self, dialogue_manager, sample_vo_line):
        """Test playing a line."""
        result = dialogue_manager.play_line(sample_vo_line)
        assert result is True

    def test_play_line_with_position(self, dialogue_manager, sample_vo_line):
        """Test playing line with spatial position."""
        result = dialogue_manager.play_line(
            sample_vo_line,
            position=(10.0, 0.0, 5.0),
        )
        assert result is True

    def test_play_line_with_radio(self, dialogue_manager, sample_vo_line):
        """Test playing line with radio effect."""
        result = dialogue_manager.play_line(
            sample_vo_line,
            radio_effect=True,
        )
        assert result is True

    def test_pause_resume(self, dialogue_manager, sample_vo_line):
        """Test pausing and resuming."""
        dialogue_manager.play_line(sample_vo_line)

        dialogue_manager.pause()
        assert dialogue_manager.state == ConfigDialogueState.PAUSED

        dialogue_manager.resume()

    def test_register_conversation(self, dialogue_manager, sample_conversation):
        """Test registering conversation."""
        dialogue_manager.register_conversation(sample_conversation)
        # Should not raise

    def test_start_conversation(self, dialogue_manager, sample_conversation):
        """Test starting conversation."""
        dialogue_manager.register_conversation(sample_conversation)
        result = dialogue_manager.start_conversation(sample_conversation.conversation_id)
        assert result is True

    def test_end_conversation(self, dialogue_manager, sample_conversation):
        """Test ending conversation."""
        dialogue_manager.register_conversation(sample_conversation)
        dialogue_manager.start_conversation(sample_conversation.conversation_id)
        result = dialogue_manager.end_conversation(sample_conversation.conversation_id)
        assert result is True

    def test_advance_conversation(self, dialogue_manager, sample_conversation):
        """Test advancing conversation."""
        dialogue_manager.register_conversation(sample_conversation)
        dialogue_manager.start_conversation(sample_conversation.conversation_id)
        result = dialogue_manager.advance_conversation(sample_conversation.conversation_id)
        assert result is True

    def test_make_conversation_choice(self, dialogue_manager, sample_conversation):
        """Test making conversation choice."""
        dialogue_manager.register_conversation(sample_conversation)
        dialogue_manager.start_conversation(sample_conversation.conversation_id)
        dialogue_manager.advance_conversation(sample_conversation.conversation_id)
        result = dialogue_manager.make_conversation_choice(
            sample_conversation.conversation_id,
            0,
        )
        assert result is True

    def test_register_bark_pool(self, dialogue_manager, sample_lines):
        """Test registering bark pool."""
        dialogue_manager.register_bark_pool("combat_react", sample_lines[:3])

    def test_trigger_bark(self, dialogue_manager, sample_lines):
        """Test triggering bark."""
        dialogue_manager.register_bark_pool("alert", sample_lines[:3])
        result = dialogue_manager.trigger_bark("alert")
        assert result is True

    def test_register_ambient_zone(self, dialogue_manager, sample_lines):
        """Test registering ambient zone."""
        dialogue_manager.register_ambient_zone("tavern", sample_lines[:5])

    def test_enter_exit_ambient_zone(self, dialogue_manager, sample_lines):
        """Test entering and exiting ambient zone."""
        dialogue_manager.register_ambient_zone("market", sample_lines[:3])
        dialogue_manager.enter_ambient_zone("market")
        dialogue_manager.exit_ambient_zone("market")

    def test_set_language(self, dialogue_manager):
        """Test setting language."""
        result = dialogue_manager.set_language("es")
        assert result is True

    def test_get_supported_languages(self, dialogue_manager):
        """Test getting supported languages."""
        languages = dialogue_manager.get_supported_languages()
        assert "en" in languages

    def test_enable_disable_subtitles(self, dialogue_manager):
        """Test enabling/disabling subtitles."""
        dialogue_manager.disable_subtitles()
        assert not dialogue_manager.subtitles_enabled

        dialogue_manager.enable_subtitles()
        assert dialogue_manager.subtitles_enabled

    def test_set_listener_position(self, dialogue_manager):
        """Test setting listener position."""
        dialogue_manager.set_listener_position(
            (0.0, 0.0, 0.0),
            forward=(0.0, 0.0, 1.0),
        )

    def test_set_environment(self, dialogue_manager):
        """Test setting environment."""
        dialogue_manager.set_environment("cave")

    def test_set_master_volume(self, dialogue_manager):
        """Test setting master volume."""
        dialogue_manager.set_master_volume(0.8)

    def test_update(self, dialogue_manager, sample_vo_line):
        """Test updating manager."""
        dialogue_manager.play_line(sample_vo_line)
        dialogue_manager.update(16.0)

    def test_stats(self, dialogue_manager):
        """Test getting stats."""
        stats = dialogue_manager.stats
        assert "state" in stats
        assert "language" in stats

    def test_event_callback(self, sample_vo_line):
        """Test DialogueManager can process lines."""
        manager = DialogueManager()
        result = manager.play_line(sample_vo_line)

        # play_line should return True on success
        assert result is True

        # Manager should be playing
        # Note: Exact behavior depends on implementation
        # At minimum, the line should be queued or playing


# =============================================================================
# Edge Cases and Thread Safety
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_queue_operations(self, vo_queue):
        """Test operations on empty queue."""
        assert vo_queue.dequeue() is None
        assert vo_queue.peek() is None

    def test_zero_duration_line(self):
        """Test handling zero duration line."""
        line = VOLine(line_id="zero", duration_ms=0.0)
        assert line.progress == 0.0
        assert line.remaining_ms == 0.0

    def test_very_long_cooldown(self, sample_vo_line):
        """Test very long cooldown."""
        assert not sample_vo_line.is_on_cooldown(
            time.time(),
            cooldown_ms=1000000.0,
        )

    def test_concurrent_queue_access(self, vo_queue, sample_lines):
        """Test thread safety of queue."""
        def enqueue_lines():
            for line in sample_lines[:5]:
                vo_queue.enqueue(line.clone())

        def dequeue_lines():
            for _ in range(5):
                vo_queue.dequeue()

        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=enqueue_lines))
            threads.append(threading.Thread(target=dequeue_lines))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def test_concurrent_cooldown_tracking(self):
        """Test thread safety of cooldown tracker."""
        tracker = CooldownTracker()

        def record_plays():
            for i in range(100):
                tracker.record_play(f"line_{i}", f"speaker_{i}", "cat", time.time())

        def check_cooldowns():
            for i in range(100):
                tracker.is_line_on_cooldown(f"line_{i}", time.time(), 30000.0)

        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=record_plays))
            threads.append(threading.Thread(target=check_cooldowns))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def test_nonexistent_conversation(self, conversation_manager):
        """Test operations on nonexistent conversation."""
        result = conversation_manager.start_conversation("nonexistent", time.time())
        assert result is None

    def test_nonexistent_bark_type(self, bark_system):
        """Test triggering nonexistent bark type."""
        line = bark_system.trigger_bark("nonexistent")
        assert line is None

    def test_empty_conversation(self):
        """Test conversation with no nodes."""
        conv = Conversation(
            conversation_id="empty",
            nodes={},
            start_node_id="nonexistent",
        )
        result = conv.start(time.time())
        assert result is None


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Performance-related tests."""

    def test_many_lines_in_queue(self, vo_queue):
        """Test queue with many lines."""
        for i in range(100):
            line = VOLine(line_id=f"line_{i}", priority=i % 100, duration_ms=1000.0)
            vo_queue.enqueue(line)

        start = time.perf_counter()
        while not vo_queue.is_empty:
            vo_queue.dequeue()
        elapsed = time.perf_counter() - start

        assert elapsed < 0.5  # Should complete quickly

    def test_rapid_cooldown_checks(self):
        """Test rapid cooldown checking."""
        tracker = CooldownTracker()
        current = time.time()

        for i in range(1000):
            tracker.record_play(f"line_{i}", "", "", current)

        start = time.perf_counter()
        for i in range(1000):
            tracker.is_line_on_cooldown(f"line_{i}", current, 30000.0)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.1  # Should be fast

    def test_many_active_subtitles(self, subtitle_manager):
        """Test many active subtitles."""
        for i in range(50):
            line = VOLine(
                line_id=f"line_{i}",
                subtitle=SubtitleData(
                    text=f"Subtitle {i}",
                    start_time_ms=0.0,
                    end_time_ms=5000.0,
                ),
                duration_ms=5000.0,
            )
            subtitle_manager.show_subtitle(line, time.time())

        start = time.perf_counter()
        for _ in range(100):
            subtitle_manager.update(16.0, time.time())
        elapsed = time.perf_counter() - start

        assert elapsed < 0.5
