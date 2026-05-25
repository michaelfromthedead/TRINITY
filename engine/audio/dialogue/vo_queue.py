"""
Voice-Over Queue Module.

Priority-based queue system for managing voice-over playback order.
Supports priority levels, interrupt handling, and queue management.
"""

from __future__ import annotations

import heapq
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

from .config import (
    DEFAULT_INTERRUPT_PRIORITY,
    MAX_QUEUE_SIZE,
    MAX_SIMULTANEOUS_VO,
    OVERLAP_DUCK_DB,
    QUEUE_TIMEOUT_MS,
    VOPriority,
)
from .vo_line import VOLine, VOLineState


@dataclass(order=True)
class QueueEntry:
    """
    Entry in the VO queue with priority ordering.

    Lower priority value = higher priority in heap (inverted for heapq).
    Entries with same priority are ordered by insertion time.
    """
    sort_key: tuple[int, float] = field(compare=True)
    line: VOLine = field(compare=False)
    enqueue_time: float = field(compare=False)
    timeout_ms: float = field(compare=False, default=QUEUE_TIMEOUT_MS)

    @classmethod
    def create(cls, line: VOLine, timeout_ms: float = QUEUE_TIMEOUT_MS) -> QueueEntry:
        """Create a queue entry with proper sort key."""
        now = time.time()
        # Negate priority so higher priority values come first
        sort_key = (-line.priority, now)
        return cls(sort_key=sort_key, line=line, enqueue_time=now, timeout_ms=timeout_ms)

    @property
    def is_expired(self) -> bool:
        """Check if this entry has timed out."""
        elapsed = (time.time() - self.enqueue_time) * 1000
        return elapsed > self.timeout_ms

    @property
    def age_ms(self) -> float:
        """Get time since enqueue in milliseconds."""
        return (time.time() - self.enqueue_time) * 1000


class VOQueue:
    """
    Priority queue for voice-over lines.

    Features:
    - Priority-based ordering (higher priority plays first)
    - Support for multiple simultaneous VO
    - Interrupt handling
    - Timeout for stale entries
    - Thread-safe operations
    """

    def __init__(
        self,
        max_size: int = MAX_QUEUE_SIZE,
        max_simultaneous: int = MAX_SIMULTANEOUS_VO,
        on_line_started: Optional[Callable[[VOLine], None]] = None,
        on_line_ended: Optional[Callable[[VOLine, bool], None]] = None,
    ) -> None:
        """
        Initialize the VO queue.

        Args:
            max_size: Maximum queue size
            max_simultaneous: Maximum concurrent VO playback
            on_line_started: Callback when a line starts playing
            on_line_ended: Callback when a line ends (bool indicates if interrupted)
        """
        self._queue: list[QueueEntry] = []
        self._lock = threading.RLock()
        self._max_size = max_size
        self._max_simultaneous = max_simultaneous
        self._active_lines: list[VOLine] = []
        self._on_line_started = on_line_started
        self._on_line_ended = on_line_ended
        self._interrupt_priority = DEFAULT_INTERRUPT_PRIORITY
        self._paused = False
        self._total_enqueued = 0
        self._total_played = 0
        self._total_dropped = 0
        self._total_interrupted = 0

    @property
    def size(self) -> int:
        """Get current queue size."""
        with self._lock:
            return len(self._queue)

    @property
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        with self._lock:
            return len(self._queue) == 0

    @property
    def is_full(self) -> bool:
        """Check if queue is at capacity."""
        with self._lock:
            return len(self._queue) >= self._max_size

    @property
    def active_count(self) -> int:
        """Get number of currently playing lines."""
        with self._lock:
            return len(self._active_lines)

    @property
    def can_play_more(self) -> bool:
        """Check if more lines can be played simultaneously."""
        with self._lock:
            return len(self._active_lines) < self._max_simultaneous

    @property
    def is_playing(self) -> bool:
        """Check if any line is currently playing."""
        with self._lock:
            return len(self._active_lines) > 0

    @property
    def is_paused(self) -> bool:
        """Check if queue processing is paused."""
        return self._paused

    @property
    def stats(self) -> dict[str, Any]:
        """Get queue statistics."""
        with self._lock:
            return {
                "queue_size": len(self._queue),
                "active_count": len(self._active_lines),
                "total_enqueued": self._total_enqueued,
                "total_played": self._total_played,
                "total_dropped": self._total_dropped,
                "total_interrupted": self._total_interrupted,
                "is_paused": self._paused,
            }

    def enqueue(
        self,
        line: VOLine,
        timeout_ms: float = QUEUE_TIMEOUT_MS,
        force: bool = False,
    ) -> bool:
        """
        Add a line to the queue.

        Args:
            line: The VO line to enqueue
            timeout_ms: Timeout before line is dropped
            force: Force add even if queue is full (drops lowest priority)

        Returns:
            True if line was added, False if rejected
        """
        with self._lock:
            # Clean expired entries first
            self._clean_expired()

            if len(self._queue) >= self._max_size:
                if force:
                    # Remove lowest priority item
                    if self._queue:
                        # Find lowest priority (highest sort key value due to negation)
                        lowest_idx = 0
                        for i, entry in enumerate(self._queue):
                            if entry.sort_key > self._queue[lowest_idx].sort_key:
                                lowest_idx = i
                        removed = self._queue.pop(lowest_idx)
                        heapq.heapify(self._queue)
                        self._total_dropped += 1
                else:
                    return False

            entry = QueueEntry.create(line, timeout_ms)
            heapq.heappush(self._queue, entry)
            self._total_enqueued += 1
            return True

    def dequeue(self) -> Optional[VOLine]:
        """
        Remove and return the highest priority line.

        Returns:
            The next line to play, or None if queue is empty
        """
        with self._lock:
            self._clean_expired()

            while self._queue:
                entry = heapq.heappop(self._queue)
                if not entry.is_expired:
                    return entry.line
                self._total_dropped += 1

            return None

    def peek(self) -> Optional[VOLine]:
        """
        Get the highest priority line without removing it.

        Returns:
            The next line to play, or None if queue is empty
        """
        with self._lock:
            self._clean_expired()
            if self._queue:
                return self._queue[0].line
            return None

    def clear(self) -> int:
        """
        Clear all pending entries from the queue.

        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = len(self._queue)
            self._queue.clear()
            self._total_dropped += count
            return count

    def remove_by_speaker(self, speaker_id: str) -> int:
        """
        Remove all lines from a specific speaker.

        Returns:
            Number of entries removed
        """
        with self._lock:
            original_count = len(self._queue)
            self._queue = [e for e in self._queue if e.line.speaker_id != speaker_id]
            heapq.heapify(self._queue)
            removed = original_count - len(self._queue)
            self._total_dropped += removed
            return removed

    def remove_by_tag(self, tag: str) -> int:
        """
        Remove all lines with a specific tag.

        Returns:
            Number of entries removed
        """
        with self._lock:
            original_count = len(self._queue)
            self._queue = [e for e in self._queue if tag not in e.line.tags]
            heapq.heapify(self._queue)
            removed = original_count - len(self._queue)
            self._total_dropped += removed
            return removed

    def remove_below_priority(self, priority: int) -> int:
        """
        Remove all lines below a certain priority.

        Returns:
            Number of entries removed
        """
        with self._lock:
            original_count = len(self._queue)
            self._queue = [e for e in self._queue if e.line.priority >= priority]
            heapq.heapify(self._queue)
            removed = original_count - len(self._queue)
            self._total_dropped += removed
            return removed

    def start_line(self, line: VOLine, current_time: float) -> bool:
        """
        Mark a line as actively playing.

        Args:
            line: The line that started playing
            current_time: Current game time

        Returns:
            True if line was started, False if max simultaneous reached
        """
        with self._lock:
            if len(self._active_lines) >= self._max_simultaneous:
                return False

            line.start_playback(current_time)
            self._active_lines.append(line)
            self._total_played += 1

            if self._on_line_started:
                self._on_line_started(line)

            return True

    def end_line(self, line: VOLine, interrupted: bool = False) -> bool:
        """
        Mark a line as finished playing.

        Args:
            line: The line that finished
            interrupted: Whether playback was interrupted

        Returns:
            True if line was found and removed from active list
        """
        with self._lock:
            if line in self._active_lines:
                self._active_lines.remove(line)
                line.complete_playback(interrupted)

                if interrupted:
                    self._total_interrupted += 1

                if self._on_line_ended:
                    self._on_line_ended(line, interrupted)

                return True
            return False

    def interrupt_for(self, incoming_priority: int) -> list[VOLine]:
        """
        Interrupt active lines for an incoming higher priority line.

        Args:
            incoming_priority: Priority of the incoming line

        Returns:
            List of interrupted lines
        """
        with self._lock:
            interrupted = []

            for line in list(self._active_lines):
                if line.can_be_interrupted_by(incoming_priority):
                    self._active_lines.remove(line)
                    line.complete_playback(interrupted=True)
                    interrupted.append(line)
                    self._total_interrupted += 1

                    if self._on_line_ended:
                        self._on_line_ended(line, True)

            return interrupted

    def get_active_lines(self) -> list[VOLine]:
        """Get a copy of currently active lines."""
        with self._lock:
            return list(self._active_lines)

    def update(self, delta_ms: float) -> list[VOLine]:
        """
        Update all active lines and return completed ones.

        Args:
            delta_ms: Time elapsed since last update

        Returns:
            List of lines that completed this update
        """
        completed = []

        with self._lock:
            for line in list(self._active_lines):
                line.update_playback(delta_ms)

                if line.is_completed:
                    self._active_lines.remove(line)
                    completed.append(line)

                    if self._on_line_ended:
                        self._on_line_ended(line, line.state == VOLineState.INTERRUPTED)

        return completed

    def pause(self) -> None:
        """Pause queue processing and all active playback."""
        with self._lock:
            self._paused = True
            for line in self._active_lines:
                line.pause()

    def resume(self) -> None:
        """Resume queue processing and playback."""
        with self._lock:
            self._paused = False
            for line in self._active_lines:
                line.resume()

    def pause_speaker(self, speaker_id: str) -> None:
        """Pause all lines from a specific speaker."""
        with self._lock:
            for line in self._active_lines:
                if line.speaker_id == speaker_id:
                    line.pause()

    def resume_speaker(self, speaker_id: str) -> None:
        """Resume all lines from a specific speaker."""
        with self._lock:
            for line in self._active_lines:
                if line.speaker_id == speaker_id:
                    line.resume()

    def get_ducking_level(self) -> float:
        """
        Calculate ducking level based on active VO.

        Returns:
            Ducking amount in dB (0 if no ducking needed)
        """
        with self._lock:
            if len(self._active_lines) <= 1:
                return 0.0
            return OVERLAP_DUCK_DB

    def _clean_expired(self) -> None:
        """Remove expired entries from the queue."""
        if not self._queue:
            return

        # Filter out expired entries
        original_count = len(self._queue)
        self._queue = [e for e in self._queue if not e.is_expired]

        if len(self._queue) != original_count:
            heapq.heapify(self._queue)
            self._total_dropped += original_count - len(self._queue)

    def __iter__(self) -> Iterator[VOLine]:
        """Iterate over queued lines in priority order."""
        with self._lock:
            # Create sorted copy
            sorted_entries = sorted(self._queue)
            for entry in sorted_entries:
                yield entry.line

    def __len__(self) -> int:
        """Get queue size."""
        return self.size

    def __bool__(self) -> bool:
        """Check if queue has entries."""
        return not self.is_empty


class VOQueueManager:
    """
    Manager for multiple named VO queues.

    Allows organizing VO into separate queues (e.g., dialogue, barks, ambient).
    """

    def __init__(self) -> None:
        """Initialize the queue manager."""
        self._queues: dict[str, VOQueue] = {}
        self._lock = threading.RLock()
        self._default_queue_name = "default"

    def create_queue(
        self,
        name: str,
        max_size: int = MAX_QUEUE_SIZE,
        max_simultaneous: int = MAX_SIMULTANEOUS_VO,
    ) -> VOQueue:
        """Create a new named queue."""
        with self._lock:
            if name in self._queues:
                raise ValueError(f"Queue '{name}' already exists")

            queue = VOQueue(max_size=max_size, max_simultaneous=max_simultaneous)
            self._queues[name] = queue
            return queue

    def get_queue(self, name: str) -> Optional[VOQueue]:
        """Get a queue by name."""
        with self._lock:
            return self._queues.get(name)

    def get_or_create_queue(
        self,
        name: str,
        max_size: int = MAX_QUEUE_SIZE,
        max_simultaneous: int = MAX_SIMULTANEOUS_VO,
    ) -> VOQueue:
        """Get existing queue or create new one."""
        with self._lock:
            if name not in self._queues:
                return self.create_queue(name, max_size, max_simultaneous)
            return self._queues[name]

    def remove_queue(self, name: str) -> bool:
        """Remove a queue."""
        with self._lock:
            if name in self._queues:
                del self._queues[name]
                return True
            return False

    def clear_all(self) -> None:
        """Clear all queues."""
        with self._lock:
            for queue in self._queues.values():
                queue.clear()

    def pause_all(self) -> None:
        """Pause all queues."""
        with self._lock:
            for queue in self._queues.values():
                queue.pause()

    def resume_all(self) -> None:
        """Resume all queues."""
        with self._lock:
            for queue in self._queues.values():
                queue.resume()

    def update_all(self, delta_ms: float) -> dict[str, list[VOLine]]:
        """Update all queues and return completed lines by queue name."""
        completed: dict[str, list[VOLine]] = {}

        with self._lock:
            for name, queue in self._queues.items():
                queue_completed = queue.update(delta_ms)
                if queue_completed:
                    completed[name] = queue_completed

        return completed

    @property
    def queue_names(self) -> list[str]:
        """Get list of queue names."""
        with self._lock:
            return list(self._queues.keys())

    @property
    def total_stats(self) -> dict[str, Any]:
        """Get combined statistics for all queues."""
        with self._lock:
            total = {
                "queue_count": len(self._queues),
                "total_size": 0,
                "total_active": 0,
                "total_enqueued": 0,
                "total_played": 0,
                "total_dropped": 0,
                "total_interrupted": 0,
            }

            for queue in self._queues.values():
                stats = queue.stats
                total["total_size"] += stats["queue_size"]
                total["total_active"] += stats["active_count"]
                total["total_enqueued"] += stats["total_enqueued"]
                total["total_played"] += stats["total_played"]
                total["total_dropped"] += stats["total_dropped"]
                total["total_interrupted"] += stats["total_interrupted"]

            return total
