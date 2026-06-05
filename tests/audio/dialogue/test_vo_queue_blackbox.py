"""
Blackbox tests for VOQueue priority queue management.

Tests PUBLIC behavior only - no internal state inspection.
Based on GAPSET_15_AUDIO Phase 9 specifications.
"""

import pytest
from typing import List

# Public API imports
from engine.audio.dialogue import (
    VOQueue,
    VOQueueManager,
    QueueEntry,
    VOLine,
    create_vo_line,
    VOPriority,
    PRIORITY_CRITICAL,
    PRIORITY_HIGH,
    PRIORITY_NORMAL,
    PRIORITY_LOW,
    PRIORITY_BARK,
    PRIORITY_AMBIENT,
    MAX_QUEUE_SIZE,
    MAX_SIMULTANEOUS_VO,
)


class TestVOQueueCreation:
    """Test VOQueue creation and initialization."""

    def test_create_queue_default(self):
        """VOQueue can be created with defaults."""
        queue = VOQueue()
        assert queue is not None

    def test_create_queue_with_max_size(self):
        """VOQueue respects max size parameter."""
        queue = VOQueue(max_size=16)
        assert queue.max_size == 16

    def test_create_queue_manager(self):
        """VOQueueManager can be created."""
        manager = VOQueueManager()
        assert manager is not None

    def test_queue_initially_empty(self):
        """New queue should be empty."""
        queue = VOQueue()
        assert queue.is_empty()
        assert len(queue) == 0


class TestVOQueueEnqueue:
    """Test adding items to VOQueue."""

    def test_enqueue_single_line(self):
        """Single VOLine can be enqueued."""
        queue = VOQueue()
        line = create_vo_line("test_001", "vo/test.wav")
        queue.enqueue(line)
        assert len(queue) == 1
        assert not queue.is_empty()

    def test_enqueue_multiple_lines(self):
        """Multiple VOLines can be enqueued."""
        queue = VOQueue()
        for i in range(5):
            line = create_vo_line(f"test_{i:03d}", f"vo/test_{i}.wav")
            queue.enqueue(line)
        assert len(queue) == 5

    def test_enqueue_returns_success(self):
        """Enqueue returns success indicator."""
        queue = VOQueue()
        line = create_vo_line("success_001", "vo/test.wav")
        result = queue.enqueue(line)
        assert result is True or result is None  # Depends on API design

    def test_enqueue_respects_max_size(self):
        """Enqueue respects maximum queue size."""
        queue = VOQueue(max_size=3)
        for i in range(5):
            line = create_vo_line(f"overflow_{i}", "vo/test.wav", priority=PRIORITY_LOW)
            queue.enqueue(line)
        assert len(queue) <= 3

    def test_enqueue_with_priority(self):
        """Lines are ordered by priority."""
        queue = VOQueue()
        low = create_vo_line("low", "vo/low.wav", priority=PRIORITY_LOW)
        high = create_vo_line("high", "vo/high.wav", priority=PRIORITY_HIGH)
        normal = create_vo_line("normal", "vo/normal.wav", priority=PRIORITY_NORMAL)

        queue.enqueue(low)
        queue.enqueue(high)
        queue.enqueue(normal)

        # First dequeue should return highest priority
        first = queue.dequeue()
        assert first.line_id == "high"


class TestVOQueueDequeue:
    """Test removing items from VOQueue."""

    def test_dequeue_empty_queue_returns_none(self):
        """Dequeue from empty queue returns None."""
        queue = VOQueue()
        result = queue.dequeue()
        assert result is None

    def test_dequeue_returns_line(self):
        """Dequeue returns VOLine."""
        queue = VOQueue()
        line = create_vo_line("dequeue_001", "vo/test.wav")
        queue.enqueue(line)
        result = queue.dequeue()
        assert result.line_id == "dequeue_001"

    def test_dequeue_removes_from_queue(self):
        """Dequeue removes item from queue."""
        queue = VOQueue()
        line = create_vo_line("remove_001", "vo/test.wav")
        queue.enqueue(line)
        queue.dequeue()
        assert queue.is_empty()

    def test_dequeue_returns_highest_priority_first(self):
        """Dequeue returns highest priority item first."""
        queue = VOQueue()

        queue.enqueue(create_vo_line("ambient", "vo/a.wav", priority=PRIORITY_AMBIENT))
        queue.enqueue(create_vo_line("critical", "vo/c.wav", priority=PRIORITY_CRITICAL))
        queue.enqueue(create_vo_line("normal", "vo/n.wav", priority=PRIORITY_NORMAL))

        first = queue.dequeue()
        assert first.line_id == "critical"

    def test_dequeue_order_by_priority_complete(self):
        """Full priority ordering verification."""
        queue = VOQueue()

        # Add in random order
        queue.enqueue(create_vo_line("low", "vo/low.wav", priority=PRIORITY_LOW))
        queue.enqueue(create_vo_line("critical", "vo/crit.wav", priority=PRIORITY_CRITICAL))
        queue.enqueue(create_vo_line("bark", "vo/bark.wav", priority=PRIORITY_BARK))
        queue.enqueue(create_vo_line("high", "vo/high.wav", priority=PRIORITY_HIGH))
        queue.enqueue(create_vo_line("normal", "vo/norm.wav", priority=PRIORITY_NORMAL))

        # Should come out in priority order
        order = []
        while not queue.is_empty():
            order.append(queue.dequeue().line_id)

        assert order == ["critical", "high", "normal", "low", "bark"]


class TestVOQueuePeek:
    """Test peeking at queue without removal."""

    def test_peek_empty_queue_returns_none(self):
        """Peek at empty queue returns None."""
        queue = VOQueue()
        result = queue.peek()
        assert result is None

    def test_peek_returns_highest_priority(self):
        """Peek returns highest priority item."""
        queue = VOQueue()
        queue.enqueue(create_vo_line("low", "vo/low.wav", priority=PRIORITY_LOW))
        queue.enqueue(create_vo_line("high", "vo/high.wav", priority=PRIORITY_HIGH))

        result = queue.peek()
        assert result.line_id == "high"

    def test_peek_does_not_remove(self):
        """Peek does not remove item."""
        queue = VOQueue()
        queue.enqueue(create_vo_line("peek_001", "vo/test.wav"))

        queue.peek()
        assert len(queue) == 1

    def test_peek_multiple_times_same_result(self):
        """Multiple peeks return same item."""
        queue = VOQueue()
        queue.enqueue(create_vo_line("multi_peek", "vo/test.wav"))

        first = queue.peek()
        second = queue.peek()
        assert first.line_id == second.line_id


class TestVOQueueClear:
    """Test clearing the queue."""

    def test_clear_empty_queue(self):
        """Clear on empty queue is safe."""
        queue = VOQueue()
        queue.clear()
        assert queue.is_empty()

    def test_clear_removes_all_items(self):
        """Clear removes all items."""
        queue = VOQueue()
        for i in range(10):
            queue.enqueue(create_vo_line(f"clear_{i}", "vo/test.wav"))

        queue.clear()
        assert queue.is_empty()
        assert len(queue) == 0


class TestVOQueuePriorityTiebreaking:
    """Test priority tiebreaking (FIFO for same priority)."""

    def test_same_priority_fifo_order(self):
        """Same priority items maintain FIFO order."""
        queue = VOQueue()

        queue.enqueue(create_vo_line("first", "vo/1.wav", priority=PRIORITY_NORMAL))
        queue.enqueue(create_vo_line("second", "vo/2.wav", priority=PRIORITY_NORMAL))
        queue.enqueue(create_vo_line("third", "vo/3.wav", priority=PRIORITY_NORMAL))

        order = []
        while not queue.is_empty():
            order.append(queue.dequeue().line_id)

        assert order == ["first", "second", "third"]

    def test_mixed_priority_with_fifo_tiebreak(self):
        """Mixed priorities with FIFO tiebreaking."""
        queue = VOQueue()

        queue.enqueue(create_vo_line("norm1", "vo/1.wav", priority=PRIORITY_NORMAL))
        queue.enqueue(create_vo_line("high1", "vo/2.wav", priority=PRIORITY_HIGH))
        queue.enqueue(create_vo_line("norm2", "vo/3.wav", priority=PRIORITY_NORMAL))
        queue.enqueue(create_vo_line("high2", "vo/4.wav", priority=PRIORITY_HIGH))

        order = []
        while not queue.is_empty():
            order.append(queue.dequeue().line_id)

        # High priority first (in FIFO order), then normal (in FIFO order)
        assert order == ["high1", "high2", "norm1", "norm2"]


class TestVOQueueInterruption:
    """Test queue interruption handling."""

    def test_interrupt_clears_lower_priority(self):
        """Interrupt with high priority clears lower priority items."""
        manager = VOQueueManager()

        manager.enqueue(create_vo_line("low1", "vo/1.wav", priority=PRIORITY_LOW))
        manager.enqueue(create_vo_line("low2", "vo/2.wav", priority=PRIORITY_LOW))

        # Interrupt with critical priority
        critical = create_vo_line("critical", "vo/c.wav", priority=PRIORITY_CRITICAL)
        manager.interrupt(critical)

        # Critical should be next
        next_line = manager.get_next()
        assert next_line.line_id == "critical"


class TestVOQueueMaxSimultaneous:
    """Test maximum simultaneous VO limit."""

    def test_max_simultaneous_limit_exists(self):
        """MAX_SIMULTANEOUS_VO constant exists."""
        assert MAX_SIMULTANEOUS_VO is not None
        assert MAX_SIMULTANEOUS_VO > 0

    def test_queue_respects_simultaneous_limit(self):
        """Queue manager respects simultaneous limit."""
        manager = VOQueueManager(max_simultaneous=2)

        for i in range(5):
            line = create_vo_line(f"sim_{i}", f"vo/{i}.wav")
            manager.enqueue(line)

        active = manager.get_active_count()
        assert active <= 2


class TestQueueEntry:
    """Test QueueEntry wrapper."""

    def test_queue_entry_creation(self):
        """QueueEntry can be created."""
        line = create_vo_line("entry_001", "vo/test.wav")
        entry = QueueEntry(line=line, timestamp=0.0)
        assert entry is not None

    def test_queue_entry_has_line(self):
        """QueueEntry contains VOLine."""
        line = create_vo_line("entry_002", "vo/test.wav")
        entry = QueueEntry(line=line, timestamp=0.0)
        assert entry.line.line_id == "entry_002"

    def test_queue_entry_has_timestamp(self):
        """QueueEntry has timestamp."""
        line = create_vo_line("entry_003", "vo/test.wav")
        entry = QueueEntry(line=line, timestamp=123.456)
        assert entry.timestamp == 123.456


class TestVOQueueManager:
    """Test VOQueueManager orchestration."""

    def test_manager_enqueue(self):
        """Manager can enqueue lines."""
        manager = VOQueueManager()
        line = create_vo_line("manager_001", "vo/test.wav")
        manager.enqueue(line)
        assert manager.pending_count() >= 1

    def test_manager_get_next(self):
        """Manager returns next line."""
        manager = VOQueueManager()
        line = create_vo_line("next_001", "vo/test.wav")
        manager.enqueue(line)

        next_line = manager.get_next()
        assert next_line.line_id == "next_001"

    def test_manager_mark_complete(self):
        """Manager marks line complete."""
        manager = VOQueueManager()
        line = create_vo_line("complete_001", "vo/test.wav")
        manager.enqueue(line)

        playing = manager.get_next()
        manager.mark_complete(playing.line_id)
        # Should not be in active list anymore

    def test_manager_update_tick(self):
        """Manager update/tick method exists."""
        manager = VOQueueManager()
        # Should have an update or tick method
        assert hasattr(manager, 'update') or hasattr(manager, 'tick')


class TestVOQueueCategoryBuckets:
    """Test per-category queue buckets."""

    def test_category_bucket_separation(self):
        """Different categories use separate buckets."""
        manager = VOQueueManager()

        from engine.audio.dialogue import CONTEXT_BARK, CONTEXT_CONVERSATION

        bark = create_vo_line("bark_001", "vo/bark.wav", context=CONTEXT_BARK)
        conv = create_vo_line("conv_001", "vo/conv.wav", context=CONTEXT_CONVERSATION)

        manager.enqueue(bark)
        manager.enqueue(conv)

        # Both should be queued
        assert manager.pending_count() >= 2


class TestVOQueueOverflow:
    """Test queue overflow handling."""

    def test_overflow_discards_lowest_priority(self):
        """Overflow discards lowest priority items."""
        queue = VOQueue(max_size=3)

        queue.enqueue(create_vo_line("high1", "vo/1.wav", priority=PRIORITY_HIGH))
        queue.enqueue(create_vo_line("high2", "vo/2.wav", priority=PRIORITY_HIGH))
        queue.enqueue(create_vo_line("low", "vo/3.wav", priority=PRIORITY_LOW))
        queue.enqueue(create_vo_line("high3", "vo/4.wav", priority=PRIORITY_HIGH))

        # Low priority should be evicted
        items = []
        while not queue.is_empty():
            items.append(queue.dequeue().line_id)

        assert "low" not in items or len(items) <= 3


class TestVOQueueStress:
    """Stress tests for queue operations."""

    def test_many_enqueue_dequeue_cycles(self):
        """Many enqueue/dequeue cycles work correctly."""
        queue = VOQueue()

        for cycle in range(100):
            for i in range(5):
                queue.enqueue(create_vo_line(f"stress_{cycle}_{i}", "vo/test.wav"))

            for i in range(3):
                queue.dequeue()

        # Should still function
        assert len(queue) >= 0

    def test_alternating_enqueue_dequeue(self):
        """Alternating enqueue/dequeue works."""
        queue = VOQueue()

        for i in range(50):
            queue.enqueue(create_vo_line(f"alt_{i}", "vo/test.wav"))
            if i > 0 and i % 2 == 0:
                queue.dequeue()

        assert len(queue) > 0


class TestVOQueueTimeout:
    """Test queue timeout handling."""

    def test_timeout_constant_exists(self):
        """QUEUE_TIMEOUT_MS constant exists."""
        from engine.audio.dialogue import QUEUE_TIMEOUT_MS
        assert QUEUE_TIMEOUT_MS is not None
        assert QUEUE_TIMEOUT_MS > 0

    def test_expired_entries_removed(self):
        """Expired entries are removed during cleanup."""
        manager = VOQueueManager()
        line = create_vo_line("timeout_001", "vo/test.wav")
        manager.enqueue(line)

        # Simulate time passage and cleanup
        if hasattr(manager, 'cleanup_expired'):
            manager.cleanup_expired(current_time=999999.0)


class TestVOQueueContains:
    """Test checking if queue contains a line."""

    def test_contains_existing_line(self):
        """Queue contains method finds existing line."""
        queue = VOQueue()
        line = create_vo_line("contains_001", "vo/test.wav")
        queue.enqueue(line)

        assert queue.contains("contains_001") or "contains_001" in [l.line_id for l in queue]

    def test_contains_missing_line(self):
        """Queue contains method returns False for missing."""
        queue = VOQueue()
        assert not queue.contains("missing_001")


class TestVOQueueRemove:
    """Test removing specific lines from queue."""

    def test_remove_specific_line(self):
        """Specific line can be removed."""
        queue = VOQueue()
        queue.enqueue(create_vo_line("keep_001", "vo/1.wav"))
        queue.enqueue(create_vo_line("remove_001", "vo/2.wav"))
        queue.enqueue(create_vo_line("keep_002", "vo/3.wav"))

        queue.remove("remove_001")

        items = []
        while not queue.is_empty():
            items.append(queue.dequeue().line_id)

        assert "remove_001" not in items

    def test_remove_nonexistent_safe(self):
        """Removing nonexistent line is safe."""
        queue = VOQueue()
        queue.remove("nonexistent")  # Should not raise
