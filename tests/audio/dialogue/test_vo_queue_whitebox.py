"""
Whitebox tests for VOQueue module.

Tests QueueEntry, VOQueue priority ordering, thread safety,
interrupt handling, and VOQueueManager.
"""

import pytest
import threading
import time
from unittest.mock import MagicMock, patch

from engine.audio.dialogue.vo_queue import (
    QueueEntry,
    VOQueue,
    VOQueueManager,
)
from engine.audio.dialogue.vo_line import VOLine, VOLineState
from engine.audio.dialogue.config import (
    PRIORITY_NORMAL,
    PRIORITY_HIGH,
    PRIORITY_CRITICAL,
    PRIORITY_LOW,
    PRIORITY_AMBIENT,
    MAX_QUEUE_SIZE,
    MAX_SIMULTANEOUS_VO,
    QUEUE_TIMEOUT_MS,
)


# =============================================================================
# QueueEntry Tests
# =============================================================================


class TestQueueEntry:
    """Tests for QueueEntry dataclass."""

    def test_create_basic(self):
        """Test QueueEntry.create() factory method."""
        line = VOLine(priority=PRIORITY_NORMAL)
        entry = QueueEntry.create(line)

        assert entry.line is line
        assert entry.enqueue_time > 0
        assert entry.timeout_ms == QUEUE_TIMEOUT_MS

    def test_create_custom_timeout(self):
        """Test QueueEntry.create() with custom timeout."""
        line = VOLine()
        entry = QueueEntry.create(line, timeout_ms=1000.0)

        assert entry.timeout_ms == 1000.0

    def test_sort_key_priority_ordering(self):
        """Test sort key gives correct priority ordering."""
        low_line = VOLine(priority=PRIORITY_LOW)
        high_line = VOLine(priority=PRIORITY_HIGH)

        low_entry = QueueEntry.create(low_line)
        high_entry = QueueEntry.create(high_line)

        # Higher priority should have lower sort_key (negated)
        assert high_entry.sort_key < low_entry.sort_key

    def test_sort_key_time_ordering(self):
        """Test sort key uses time for same priority."""
        line1 = VOLine(priority=PRIORITY_NORMAL)
        entry1 = QueueEntry.create(line1)

        time.sleep(0.01)

        line2 = VOLine(priority=PRIORITY_NORMAL)
        entry2 = QueueEntry.create(line2)

        # Earlier enqueue should come first (lower sort_key)
        assert entry1.sort_key < entry2.sort_key

    def test_is_expired_false(self):
        """Test is_expired returns False when not expired."""
        line = VOLine()
        entry = QueueEntry.create(line, timeout_ms=10000.0)

        assert entry.is_expired is False

    def test_is_expired_true(self):
        """Test is_expired returns True when expired."""
        line = VOLine()
        # Very short timeout
        with patch('engine.audio.dialogue.vo_queue.time.time') as mock_time:
            mock_time.return_value = 100.0
            entry = QueueEntry.create(line, timeout_ms=1.0)

            mock_time.return_value = 101.0  # 1 second later = 1000ms
            assert entry.is_expired is True

    def test_age_ms_calculation(self):
        """Test age_ms property."""
        with patch('engine.audio.dialogue.vo_queue.time.time') as mock_time:
            mock_time.return_value = 100.0
            line = VOLine()
            entry = QueueEntry.create(line)

            mock_time.return_value = 100.5  # 500ms later
            assert abs(entry.age_ms - 500.0) < 1.0

    def test_comparison_operators(self):
        """Test QueueEntry comparison for heap ordering."""
        line1 = VOLine(priority=PRIORITY_HIGH)
        line2 = VOLine(priority=PRIORITY_LOW)

        entry1 = QueueEntry.create(line1)
        entry2 = QueueEntry.create(line2)

        # entry1 (high priority) should be less than entry2 (low priority)
        assert entry1 < entry2


# =============================================================================
# VOQueue Basic Tests
# =============================================================================


class TestVOQueueBasic:
    """Basic tests for VOQueue."""

    def test_initialization(self):
        """Test VOQueue initializes correctly."""
        queue = VOQueue()

        assert queue.size == 0
        assert queue.is_empty is True
        assert queue.is_full is False
        assert queue.active_count == 0

    def test_custom_initialization(self):
        """Test VOQueue with custom parameters."""
        queue = VOQueue(max_size=10, max_simultaneous=3)

        assert queue._max_size == 10
        assert queue._max_simultaneous == 3

    def test_enqueue_basic(self):
        """Test basic enqueue operation."""
        queue = VOQueue()
        line = VOLine()

        result = queue.enqueue(line)

        assert result is True
        assert queue.size == 1
        assert queue.is_empty is False

    def test_enqueue_multiple(self):
        """Test enqueueing multiple lines."""
        queue = VOQueue()

        for i in range(5):
            queue.enqueue(VOLine())

        assert queue.size == 5

    def test_enqueue_full_queue(self):
        """Test enqueue fails when queue is full."""
        queue = VOQueue(max_size=2)
        queue.enqueue(VOLine())
        queue.enqueue(VOLine())

        result = queue.enqueue(VOLine())

        assert result is False
        assert queue.size == 2

    def test_enqueue_force_drops_lowest(self):
        """Test enqueue with force drops lowest priority."""
        queue = VOQueue(max_size=2)

        # Add low and normal priority
        low_line = VOLine(priority=PRIORITY_LOW)
        normal_line = VOLine(priority=PRIORITY_NORMAL)
        queue.enqueue(low_line)
        queue.enqueue(normal_line)

        # Force add high priority
        high_line = VOLine(priority=PRIORITY_HIGH)
        result = queue.enqueue(high_line, force=True)

        assert result is True
        assert queue.size == 2

        # Verify low priority was dropped
        lines = list(queue)
        priorities = [l.priority for l in lines]
        assert PRIORITY_LOW not in priorities

    def test_dequeue_basic(self):
        """Test basic dequeue operation."""
        queue = VOQueue()
        line = VOLine()
        queue.enqueue(line)

        result = queue.dequeue()

        assert result is line
        assert queue.size == 0

    def test_dequeue_empty_queue(self):
        """Test dequeue from empty queue."""
        queue = VOQueue()

        result = queue.dequeue()

        assert result is None

    def test_dequeue_priority_order(self):
        """Test dequeue returns highest priority first."""
        queue = VOQueue()

        low = VOLine(priority=PRIORITY_LOW)
        high = VOLine(priority=PRIORITY_HIGH)
        normal = VOLine(priority=PRIORITY_NORMAL)

        queue.enqueue(low)
        queue.enqueue(high)
        queue.enqueue(normal)

        assert queue.dequeue().priority == PRIORITY_HIGH
        assert queue.dequeue().priority == PRIORITY_NORMAL
        assert queue.dequeue().priority == PRIORITY_LOW

    def test_peek_basic(self):
        """Test peek returns highest priority without removal."""
        queue = VOQueue()
        high = VOLine(priority=PRIORITY_HIGH)
        low = VOLine(priority=PRIORITY_LOW)

        queue.enqueue(low)
        queue.enqueue(high)

        result = queue.peek()

        assert result.priority == PRIORITY_HIGH
        assert queue.size == 2

    def test_peek_empty_queue(self):
        """Test peek from empty queue."""
        queue = VOQueue()

        assert queue.peek() is None

    def test_clear(self):
        """Test clear removes all entries."""
        queue = VOQueue()
        for _ in range(5):
            queue.enqueue(VOLine())

        count = queue.clear()

        assert count == 5
        assert queue.size == 0
        assert queue.is_empty is True


# =============================================================================
# VOQueue Properties Tests
# =============================================================================


class TestVOQueueProperties:
    """Tests for VOQueue properties."""

    def test_is_full(self):
        """Test is_full property."""
        queue = VOQueue(max_size=2)

        assert queue.is_full is False
        queue.enqueue(VOLine())
        assert queue.is_full is False
        queue.enqueue(VOLine())
        assert queue.is_full is True

    def test_can_play_more_true(self):
        """Test can_play_more when under limit."""
        queue = VOQueue(max_simultaneous=2)

        assert queue.can_play_more is True

    def test_can_play_more_false(self):
        """Test can_play_more when at limit."""
        queue = VOQueue(max_simultaneous=1)
        line = VOLine()
        queue.start_line(line, 0.0)

        assert queue.can_play_more is False

    def test_is_playing(self):
        """Test is_playing property."""
        queue = VOQueue()

        assert queue.is_playing is False

        line = VOLine()
        queue.start_line(line, 0.0)

        assert queue.is_playing is True

    def test_is_paused(self):
        """Test is_paused property."""
        queue = VOQueue()

        assert queue.is_paused is False

        queue.pause()

        assert queue.is_paused is True

    def test_stats(self):
        """Test stats property."""
        queue = VOQueue()
        queue.enqueue(VOLine())
        queue.enqueue(VOLine())

        stats = queue.stats

        assert stats["queue_size"] == 2
        assert stats["active_count"] == 0
        assert stats["total_enqueued"] == 2
        assert stats["total_played"] == 0
        assert "total_dropped" in stats
        assert "is_paused" in stats


# =============================================================================
# VOQueue Removal Tests
# =============================================================================


class TestVOQueueRemoval:
    """Tests for VOQueue removal methods."""

    def test_remove_by_speaker(self):
        """Test remove_by_speaker removes correct lines."""
        queue = VOQueue()

        queue.enqueue(VOLine(speaker_id="npc1"))
        queue.enqueue(VOLine(speaker_id="npc2"))
        queue.enqueue(VOLine(speaker_id="npc1"))

        count = queue.remove_by_speaker("npc1")

        assert count == 2
        assert queue.size == 1

        remaining = queue.dequeue()
        assert remaining.speaker_id == "npc2"

    def test_remove_by_speaker_none_found(self):
        """Test remove_by_speaker when none found."""
        queue = VOQueue()
        queue.enqueue(VOLine(speaker_id="npc1"))

        count = queue.remove_by_speaker("npc2")

        assert count == 0
        assert queue.size == 1

    def test_remove_by_tag(self):
        """Test remove_by_tag removes correct lines."""
        queue = VOQueue()

        queue.enqueue(VOLine(tags={"combat", "urgent"}))
        queue.enqueue(VOLine(tags={"ambient"}))
        queue.enqueue(VOLine(tags={"combat"}))

        count = queue.remove_by_tag("combat")

        assert count == 2
        assert queue.size == 1

    def test_remove_below_priority(self):
        """Test remove_below_priority removes low priority lines."""
        queue = VOQueue()

        queue.enqueue(VOLine(priority=PRIORITY_LOW))
        queue.enqueue(VOLine(priority=PRIORITY_NORMAL))
        queue.enqueue(VOLine(priority=PRIORITY_HIGH))

        count = queue.remove_below_priority(PRIORITY_NORMAL)

        assert count == 1
        assert queue.size == 2


# =============================================================================
# VOQueue Active Lines Tests
# =============================================================================


class TestVOQueueActiveLines:
    """Tests for VOQueue active line management."""

    def test_start_line(self):
        """Test start_line adds to active list."""
        queue = VOQueue()
        line = VOLine(duration_ms=1000.0)

        result = queue.start_line(line, 100.0)

        assert result is True
        assert queue.active_count == 1
        assert line.state == VOLineState.PLAYING

    def test_start_line_at_max(self):
        """Test start_line fails when at max simultaneous."""
        queue = VOQueue(max_simultaneous=1)
        line1 = VOLine()
        line2 = VOLine()

        queue.start_line(line1, 100.0)
        result = queue.start_line(line2, 100.0)

        assert result is False
        assert queue.active_count == 1

    def test_start_line_callback(self):
        """Test start_line triggers callback."""
        callback = MagicMock()
        queue = VOQueue(on_line_started=callback)
        line = VOLine()

        queue.start_line(line, 100.0)

        callback.assert_called_once_with(line)

    def test_end_line(self):
        """Test end_line removes from active list."""
        queue = VOQueue()
        line = VOLine()
        queue.start_line(line, 100.0)

        result = queue.end_line(line)

        assert result is True
        assert queue.active_count == 0
        assert line.state == VOLineState.COMPLETED

    def test_end_line_interrupted(self):
        """Test end_line with interrupted flag."""
        callback = MagicMock()
        queue = VOQueue(on_line_ended=callback)
        line = VOLine()
        queue.start_line(line, 100.0)

        queue.end_line(line, interrupted=True)

        assert line.state == VOLineState.INTERRUPTED
        callback.assert_called_with(line, True)

    def test_end_line_not_active(self):
        """Test end_line returns False for inactive line."""
        queue = VOQueue()
        line = VOLine()

        result = queue.end_line(line)

        assert result is False

    def test_get_active_lines(self):
        """Test get_active_lines returns copy."""
        queue = VOQueue(max_simultaneous=3)
        lines = [VOLine(), VOLine()]

        for line in lines:
            queue.start_line(line, 100.0)

        active = queue.get_active_lines()

        assert len(active) == 2
        # Should be a copy
        active.clear()
        assert queue.active_count == 2


# =============================================================================
# VOQueue Interrupt Tests
# =============================================================================


class TestVOQueueInterrupt:
    """Tests for VOQueue interrupt functionality."""

    def test_interrupt_for_interrupts_lower_priority(self):
        """Test interrupt_for interrupts lower priority lines."""
        callback = MagicMock()
        queue = VOQueue(max_simultaneous=2, on_line_ended=callback)

        low = VOLine(priority=PRIORITY_LOW, interruptible=True)
        normal = VOLine(priority=PRIORITY_NORMAL, interruptible=True)

        queue.start_line(low, 100.0)
        queue.start_line(normal, 100.0)

        interrupted = queue.interrupt_for(PRIORITY_HIGH)

        assert len(interrupted) == 2
        assert queue.active_count == 0

    def test_interrupt_for_respects_interruptible(self):
        """Test interrupt_for respects interruptible flag."""
        queue = VOQueue(max_simultaneous=2)

        interruptible = VOLine(priority=PRIORITY_LOW, interruptible=True)
        non_interruptible = VOLine(priority=PRIORITY_LOW, interruptible=False)

        queue.start_line(interruptible, 100.0)
        queue.start_line(non_interruptible, 100.0)

        interrupted = queue.interrupt_for(PRIORITY_HIGH)

        assert len(interrupted) == 1
        assert queue.active_count == 1

    def test_interrupt_for_no_interrupts(self):
        """Test interrupt_for when nothing to interrupt."""
        queue = VOQueue()

        high = VOLine(priority=PRIORITY_HIGH, interruptible=True)
        queue.start_line(high, 100.0)

        interrupted = queue.interrupt_for(PRIORITY_NORMAL)

        assert len(interrupted) == 0
        assert queue.active_count == 1


# =============================================================================
# VOQueue Update Tests
# =============================================================================


class TestVOQueueUpdate:
    """Tests for VOQueue update functionality."""

    def test_update_advances_playback(self):
        """Test update advances playback position."""
        queue = VOQueue()
        line = VOLine(duration_ms=1000.0)
        queue.start_line(line, 100.0)

        queue.update(100.0)

        assert line.playback_position_ms == 100.0

    def test_update_completes_finished_lines(self):
        """Test update removes completed lines."""
        callback = MagicMock()
        queue = VOQueue(on_line_ended=callback)
        line = VOLine(duration_ms=100.0)
        queue.start_line(line, 0.0)

        completed = queue.update(200.0)

        assert len(completed) == 1
        assert completed[0] is line
        assert queue.active_count == 0
        callback.assert_called()

    def test_update_multiple_lines(self):
        """Test update handles multiple active lines."""
        queue = VOQueue(max_simultaneous=3)

        short = VOLine(duration_ms=100.0)
        long = VOLine(duration_ms=1000.0)

        queue.start_line(short, 0.0)
        queue.start_line(long, 0.0)

        completed = queue.update(200.0)

        assert len(completed) == 1
        assert short in completed
        assert queue.active_count == 1


# =============================================================================
# VOQueue Pause/Resume Tests
# =============================================================================


class TestVOQueuePauseResume:
    """Tests for VOQueue pause/resume functionality."""

    def test_pause(self):
        """Test pause pauses all active lines."""
        queue = VOQueue(max_simultaneous=2)

        line1 = VOLine()
        line2 = VOLine()
        queue.start_line(line1, 100.0)
        queue.start_line(line2, 100.0)

        queue.pause()

        assert queue.is_paused is True
        assert line1.state == VOLineState.PAUSED
        assert line2.state == VOLineState.PAUSED

    def test_resume(self):
        """Test resume resumes all active lines."""
        queue = VOQueue(max_simultaneous=2)

        line1 = VOLine()
        line2 = VOLine()
        queue.start_line(line1, 100.0)
        queue.start_line(line2, 100.0)
        queue.pause()

        queue.resume()

        assert queue.is_paused is False
        assert line1.state == VOLineState.PLAYING
        assert line2.state == VOLineState.PLAYING

    def test_pause_speaker(self):
        """Test pause_speaker pauses specific speaker."""
        queue = VOQueue(max_simultaneous=2)

        line1 = VOLine(speaker_id="npc1")
        line2 = VOLine(speaker_id="npc2")
        queue.start_line(line1, 100.0)
        queue.start_line(line2, 100.0)

        queue.pause_speaker("npc1")

        assert line1.state == VOLineState.PAUSED
        assert line2.state == VOLineState.PLAYING

    def test_resume_speaker(self):
        """Test resume_speaker resumes specific speaker."""
        queue = VOQueue(max_simultaneous=2)

        line1 = VOLine(speaker_id="npc1")
        line2 = VOLine(speaker_id="npc2")
        queue.start_line(line1, 100.0)
        queue.start_line(line2, 100.0)
        line1.pause()

        queue.resume_speaker("npc1")

        assert line1.state == VOLineState.PLAYING


# =============================================================================
# VOQueue Ducking Tests
# =============================================================================


class TestVOQueueDucking:
    """Tests for VOQueue ducking functionality."""

    def test_ducking_level_no_active(self):
        """Test ducking level with no active lines."""
        queue = VOQueue()

        assert queue.get_ducking_level() == 0.0

    def test_ducking_level_one_active(self):
        """Test ducking level with one active line."""
        queue = VOQueue()
        queue.start_line(VOLine(), 100.0)

        assert queue.get_ducking_level() == 0.0

    def test_ducking_level_multiple_active(self):
        """Test ducking level with multiple active lines."""
        queue = VOQueue(max_simultaneous=3)
        queue.start_line(VOLine(), 100.0)
        queue.start_line(VOLine(), 100.0)

        assert queue.get_ducking_level() < 0  # Should be negative dB


# =============================================================================
# VOQueue Expiration Tests
# =============================================================================


class TestVOQueueExpiration:
    """Tests for VOQueue entry expiration."""

    def test_dequeue_skips_expired(self):
        """Test dequeue skips expired entries."""
        queue = VOQueue()

        with patch('engine.audio.dialogue.vo_queue.time.time') as mock_time:
            mock_time.return_value = 100.0
            queue.enqueue(VOLine(), timeout_ms=100.0)  # Will expire

            mock_time.return_value = 200.0  # 100 seconds later
            result = queue.dequeue()

            assert result is None

    def test_clean_expired_removes_old_entries(self):
        """Test _clean_expired removes old entries."""
        queue = VOQueue()

        with patch('engine.audio.dialogue.vo_queue.time.time') as mock_time:
            mock_time.return_value = 100.0
            queue.enqueue(VOLine(), timeout_ms=1.0)
            queue.enqueue(VOLine(), timeout_ms=10000.0)

            mock_time.return_value = 100.01  # Just past first timeout
            queue._clean_expired()

            assert queue.size == 1


# =============================================================================
# VOQueue Iterator Tests
# =============================================================================


class TestVOQueueIterator:
    """Tests for VOQueue iteration."""

    def test_iterate_in_priority_order(self):
        """Test iteration returns lines in priority order."""
        queue = VOQueue()

        low = VOLine(priority=PRIORITY_LOW)
        high = VOLine(priority=PRIORITY_HIGH)
        normal = VOLine(priority=PRIORITY_NORMAL)

        queue.enqueue(low)
        queue.enqueue(high)
        queue.enqueue(normal)

        priorities = [line.priority for line in queue]

        assert priorities == [PRIORITY_HIGH, PRIORITY_NORMAL, PRIORITY_LOW]

    def test_len(self):
        """Test __len__ returns queue size."""
        queue = VOQueue()
        queue.enqueue(VOLine())
        queue.enqueue(VOLine())

        assert len(queue) == 2

    def test_bool_empty(self):
        """Test __bool__ for empty queue."""
        queue = VOQueue()

        assert bool(queue) is False

    def test_bool_non_empty(self):
        """Test __bool__ for non-empty queue."""
        queue = VOQueue()
        queue.enqueue(VOLine())

        assert bool(queue) is True


# =============================================================================
# VOQueue Thread Safety Tests
# =============================================================================


class TestVOQueueThreadSafety:
    """Thread safety tests for VOQueue."""

    def test_concurrent_enqueue(self):
        """Test concurrent enqueue operations."""
        queue = VOQueue(max_size=100)
        results = []

        def enqueue_lines():
            for _ in range(20):
                results.append(queue.enqueue(VOLine()))
                time.sleep(0.001)

        threads = [threading.Thread(target=enqueue_lines) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All enqueues should succeed
        assert all(results)
        assert queue.size == 100

    def test_concurrent_dequeue(self):
        """Test concurrent dequeue operations."""
        queue = VOQueue()
        for _ in range(100):
            queue.enqueue(VOLine())

        results = []

        def dequeue_lines():
            for _ in range(20):
                result = queue.dequeue()
                results.append(result)
                time.sleep(0.001)

        threads = [threading.Thread(target=dequeue_lines) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should get 100 non-None results
        non_none = [r for r in results if r is not None]
        assert len(non_none) == 100

    def test_concurrent_start_end(self):
        """Test concurrent start/end operations."""
        queue = VOQueue(max_simultaneous=50)
        lines = [VOLine() for _ in range(100)]

        def start_lines():
            for line in lines[:50]:
                queue.start_line(line, time.time())
                time.sleep(0.001)

        def end_lines():
            for line in lines[:50]:
                queue.end_line(line)
                time.sleep(0.001)

        t1 = threading.Thread(target=start_lines)
        t2 = threading.Thread(target=end_lines)

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Should complete without deadlock


# =============================================================================
# VOQueueManager Tests
# =============================================================================


class TestVOQueueManager:
    """Tests for VOQueueManager."""

    def test_initialization(self):
        """Test VOQueueManager initializes empty."""
        manager = VOQueueManager()

        assert manager.queue_names == []

    def test_create_queue(self):
        """Test create_queue creates named queue."""
        manager = VOQueueManager()

        queue = manager.create_queue("dialogue")

        assert isinstance(queue, VOQueue)
        assert "dialogue" in manager.queue_names

    def test_create_queue_duplicate(self):
        """Test create_queue raises on duplicate name."""
        manager = VOQueueManager()
        manager.create_queue("dialogue")

        with pytest.raises(ValueError):
            manager.create_queue("dialogue")

    def test_get_queue(self):
        """Test get_queue retrieves queue."""
        manager = VOQueueManager()
        created = manager.create_queue("dialogue")

        retrieved = manager.get_queue("dialogue")

        assert retrieved is created

    def test_get_queue_not_found(self):
        """Test get_queue returns None for missing queue."""
        manager = VOQueueManager()

        result = manager.get_queue("missing")

        assert result is None

    def test_get_or_create_queue_creates(self):
        """Test get_or_create_queue creates new queue."""
        manager = VOQueueManager()

        queue = manager.get_or_create_queue("dialogue")

        assert isinstance(queue, VOQueue)
        assert "dialogue" in manager.queue_names

    def test_get_or_create_queue_gets_existing(self):
        """Test get_or_create_queue returns existing queue."""
        manager = VOQueueManager()
        created = manager.create_queue("dialogue")

        retrieved = manager.get_or_create_queue("dialogue")

        assert retrieved is created

    def test_remove_queue(self):
        """Test remove_queue removes queue."""
        manager = VOQueueManager()
        manager.create_queue("dialogue")

        result = manager.remove_queue("dialogue")

        assert result is True
        assert "dialogue" not in manager.queue_names

    def test_remove_queue_not_found(self):
        """Test remove_queue returns False for missing queue."""
        manager = VOQueueManager()

        result = manager.remove_queue("missing")

        assert result is False

    def test_clear_all(self):
        """Test clear_all clears all queues."""
        manager = VOQueueManager()
        q1 = manager.create_queue("dialogue")
        q2 = manager.create_queue("barks")

        q1.enqueue(VOLine())
        q2.enqueue(VOLine())

        manager.clear_all()

        assert q1.size == 0
        assert q2.size == 0

    def test_pause_all(self):
        """Test pause_all pauses all queues."""
        manager = VOQueueManager()
        q1 = manager.create_queue("dialogue")
        q2 = manager.create_queue("barks")

        manager.pause_all()

        assert q1.is_paused is True
        assert q2.is_paused is True

    def test_resume_all(self):
        """Test resume_all resumes all queues."""
        manager = VOQueueManager()
        q1 = manager.create_queue("dialogue")
        q2 = manager.create_queue("barks")
        manager.pause_all()

        manager.resume_all()

        assert q1.is_paused is False
        assert q2.is_paused is False

    def test_update_all(self):
        """Test update_all updates all queues."""
        manager = VOQueueManager()
        q1 = manager.create_queue("dialogue")
        q2 = manager.create_queue("barks")

        line1 = VOLine(duration_ms=100.0)
        line2 = VOLine(duration_ms=1000.0)

        q1.start_line(line1, 0.0)
        q2.start_line(line2, 0.0)

        completed = manager.update_all(200.0)

        assert "dialogue" in completed
        assert len(completed["dialogue"]) == 1
        assert "barks" not in completed  # Still playing

    def test_total_stats(self):
        """Test total_stats aggregates all queue stats."""
        manager = VOQueueManager()
        q1 = manager.create_queue("dialogue")
        q2 = manager.create_queue("barks")

        q1.enqueue(VOLine())
        q1.enqueue(VOLine())
        q2.enqueue(VOLine())

        stats = manager.total_stats

        assert stats["queue_count"] == 2
        assert stats["total_size"] == 3
        assert stats["total_enqueued"] == 3


# =============================================================================
# VOQueueManager Thread Safety Tests
# =============================================================================


class TestVOQueueManagerThreadSafety:
    """Thread safety tests for VOQueueManager."""

    def test_concurrent_create_queue(self):
        """Test concurrent queue creation."""
        manager = VOQueueManager()
        results = []

        def create_queues(prefix):
            for i in range(10):
                try:
                    manager.create_queue(f"{prefix}_{i}")
                    results.append(True)
                except ValueError:
                    results.append(False)
                time.sleep(0.001)

        threads = [threading.Thread(target=create_queues, args=(f"t{i}",)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have created 30 queues
        assert len(manager.queue_names) == 30

    def test_concurrent_update_all(self):
        """Test concurrent update_all operations."""
        manager = VOQueueManager()
        queue = manager.create_queue("test")

        # Start some lines
        for _ in range(10):
            queue.start_line(VOLine(duration_ms=1000.0), 0.0)

        def update_loop():
            for _ in range(20):
                manager.update_all(10.0)
                time.sleep(0.001)

        threads = [threading.Thread(target=update_loop) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should complete without deadlock
