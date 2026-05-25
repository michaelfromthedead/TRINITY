"""Bandwidth management for network replication.

Handles bandwidth allocation, priority-based entity scheduling, and
anti-starvation mechanisms for fair update distribution.
"""

from __future__ import annotations

import heapq
import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Optional

from ..config import get_config

_logger = logging.getLogger(__name__)

# Get config instance
_config = get_config()


class EntityPriority(IntEnum):
    """Priority levels for replicated entities."""
    CRITICAL = _config.PRIORITY_CRITICAL
    HIGH = _config.PRIORITY_HIGH
    NORMAL = _config.PRIORITY_NORMAL
    LOW = _config.PRIORITY_LOW
    MINIMAL = _config.PRIORITY_MINIMAL


# Default bandwidth limits - from config
DEFAULT_MAX_BPS = _config.DEFAULT_MAX_BANDWIDTH_BPS
DEFAULT_BURST_BPS = _config.DEFAULT_BURST_BANDWIDTH_BPS
DEFAULT_UPDATE_INTERVAL = _config.DEFAULT_UPDATE_INTERVAL

# Anti-starvation settings - from config
MAX_STARVATION_TIME = _config.MAX_STARVATION_TIME_SECONDS
STARVATION_PRIORITY_BOOST = _config.STARVATION_PRIORITY_BOOST


@dataclass(slots=True, order=True)
class PrioritizedEntity:
    """Entity wrapper for priority queue ordering.

    Negative priority for max-heap behavior with heapq (min-heap).

    Attributes:
        priority: Effective priority (negative for heap ordering)
        entity: The actual entity
        guid: Network GUID
        estimated_size: Estimated serialization size in bytes
        last_sent_time: Last time this entity was replicated
        starvation_time: Time since last successful replication
    """
    priority: float = field(compare=True)
    entity: Any = field(compare=False)
    guid: int = field(compare=False)
    estimated_size: int = field(compare=False, default=64)
    last_sent_time: float = field(compare=False, default=0.0)
    starvation_time: float = field(compare=False, default=0.0)

    def __post_init__(self):
        # Negate priority for max-heap behavior
        object.__setattr__(self, 'priority', -self.priority)


@dataclass
class BandwidthBudget:
    """Bandwidth budget for a connection or channel.

    Uses a token bucket algorithm for rate limiting with burst support.

    Attributes:
        max_bps: Maximum sustained bits per second
        burst_bps: Maximum burst bits per second
        current_tokens: Available bandwidth tokens (bits)
        last_update: Last token refill time
    """
    max_bps: int = DEFAULT_MAX_BPS * 8  # Convert to bits
    burst_bps: int = DEFAULT_BURST_BPS * 8
    current_tokens: float = field(default_factory=lambda: DEFAULT_BURST_BPS * 8)
    last_update: float = field(default_factory=time.time)

    def __post_init__(self):
        # Initialize tokens to burst capacity
        self.current_tokens = float(self.burst_bps)

    @property
    def current_usage_bps(self) -> float:
        """Estimate current bandwidth usage in bits per second."""
        return self.max_bps - self.current_tokens

    @property
    def available_bytes(self) -> int:
        """Get available bandwidth in bytes."""
        return int(self.current_tokens / 8)

    def refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_update
        self.last_update = now

        # Add tokens based on max rate
        self.current_tokens += elapsed * self.max_bps

        # Cap at burst limit
        if self.current_tokens > self.burst_bps:
            self.current_tokens = float(self.burst_bps)

    def can_send(self, size_bytes: int) -> bool:
        """Check if we have bandwidth to send data.

        Args:
            size_bytes: Size of data to send

        Returns:
            True if bandwidth available
        """
        self.refill()
        size_bits = size_bytes * 8
        return self.current_tokens >= size_bits

    def consume(self, size_bytes: int) -> bool:
        """Consume bandwidth tokens for sent data.

        Args:
            size_bytes: Size of data sent

        Returns:
            True if tokens consumed, False if insufficient
        """
        self.refill()
        size_bits = size_bytes * 8

        if self.current_tokens >= size_bits:
            self.current_tokens -= size_bits
            return True
        return False

    def reserve(self, size_bytes: int) -> bool:
        """Reserve bandwidth without consuming (for planning).

        Args:
            size_bytes: Size to reserve

        Returns:
            True if reservation possible
        """
        return self.can_send(size_bytes)

    def reset(self) -> None:
        """Reset budget to full capacity."""
        self.current_tokens = float(self.burst_bps)
        self.last_update = time.time()


class PriorityQueue:
    """Priority queue for bandwidth allocation.

    Manages entities sorted by priority with anti-starvation support.
    """
    __slots__ = ('_heap', '_entity_map', '_starvation_threshold')

    def __init__(self, starvation_threshold: float = MAX_STARVATION_TIME):
        """Initialize the priority queue.

        Args:
            starvation_threshold: Time before priority boost
        """
        self._heap: list[PrioritizedEntity] = []
        self._entity_map: dict[int, PrioritizedEntity] = {}  # guid -> entry
        self._starvation_threshold = starvation_threshold

    def add(
        self,
        entity: Any,
        guid: int,
        priority: float,
        estimated_size: int = _config.DEFAULT_ENTITY_SERIALIZATION_SIZE
    ) -> None:
        """Add or update an entity in the queue.

        Args:
            entity: The entity
            guid: Network GUID
            priority: Base priority
            estimated_size: Estimated serialization size
        """
        now = time.time()

        # Check for existing entry
        existing = self._entity_map.get(guid)
        if existing:
            # Update starvation time
            starvation = now - existing.last_sent_time
        else:
            starvation = 0.0

        # Apply anti-starvation boost
        effective_priority = priority
        if starvation > self._starvation_threshold:
            boost = (starvation / self._starvation_threshold) * STARVATION_PRIORITY_BOOST
            effective_priority = min(priority + boost, _config.MAX_PRIORITY_WITH_BOOST)

        entry = PrioritizedEntity(
            priority=effective_priority,
            entity=entity,
            guid=guid,
            estimated_size=estimated_size,
            last_sent_time=existing.last_sent_time if existing else now,
            starvation_time=starvation
        )

        self._entity_map[guid] = entry
        heapq.heappush(self._heap, entry)

    def remove(self, guid: int) -> Optional[PrioritizedEntity]:
        """Remove an entity from the queue.

        Args:
            guid: Network GUID

        Returns:
            Removed entry or None
        """
        entry = self._entity_map.pop(guid, None)
        # Note: We don't remove from heap for efficiency
        # Invalid entries are filtered during pop
        return entry

    def pop(self) -> Optional[PrioritizedEntity]:
        """Pop highest priority entity.

        Returns:
            Highest priority entity or None if empty
        """
        while self._heap:
            entry = heapq.heappop(self._heap)
            # Check if entry is still valid
            if entry.guid in self._entity_map:
                current = self._entity_map[entry.guid]
                if current.priority == entry.priority:
                    return entry
        return None

    def peek(self) -> Optional[PrioritizedEntity]:
        """Peek at highest priority entity without removing.

        Returns:
            Highest priority entity or None
        """
        while self._heap:
            entry = self._heap[0]
            if entry.guid in self._entity_map:
                current = self._entity_map[entry.guid]
                if current.priority == entry.priority:
                    return entry
            heapq.heappop(self._heap)
        return None

    def mark_sent(self, guid: int) -> None:
        """Mark an entity as successfully sent.

        Args:
            guid: Network GUID
        """
        entry = self._entity_map.get(guid)
        if entry:
            # Create updated entry with new send time
            new_entry = PrioritizedEntity(
                priority=-entry.priority,  # Restore original priority
                entity=entry.entity,
                guid=entry.guid,
                estimated_size=entry.estimated_size,
                last_sent_time=time.time(),
                starvation_time=0.0
            )
            self._entity_map[guid] = new_entry

    def update_priority(self, guid: int, new_priority: float) -> None:
        """Update an entity's priority.

        Args:
            guid: Network GUID
            new_priority: New priority value
        """
        entry = self._entity_map.get(guid)
        if entry:
            self.add(entry.entity, guid, new_priority, entry.estimated_size)

    def clear(self) -> None:
        """Clear the queue."""
        self._heap.clear()
        self._entity_map.clear()

    def __len__(self) -> int:
        return len(self._entity_map)

    def __bool__(self) -> bool:
        return bool(self._entity_map)


def allocate_bandwidth(
    entities: list[tuple[Any, int, float, int]],  # (entity, guid, priority, size)
    budget: BandwidthBudget
) -> list[tuple[Any, int]]:
    """Allocate bandwidth to entities based on priority.

    Uses a greedy algorithm to maximize priority within budget.
    Includes anti-starvation to ensure all entities eventually update.

    Args:
        entities: List of (entity, guid, priority, estimated_size) tuples
        budget: Available bandwidth budget

    Returns:
        List of (entity, guid) tuples that fit within budget
    """
    if not entities:
        return []

    # Refresh budget
    budget.refill()
    available = budget.available_bytes

    # Sort by priority (highest first)
    sorted_entities = sorted(entities, key=lambda x: x[2], reverse=True)

    result = []
    remaining = available

    for entity, guid, priority, size in sorted_entities:
        if size <= remaining:
            result.append((entity, guid))
            remaining -= size
        elif remaining < _config.MIN_UPDATE_PACKET_SIZE:
            # Not enough space for even small updates
            break

    return result


def allocate_bandwidth_fair(
    entities: list[tuple[Any, int, float, int]],
    budget: BandwidthBudget,
    last_sent_times: dict[int, float]  # guid -> timestamp
) -> list[tuple[Any, int]]:
    """Allocate bandwidth with fair scheduling and anti-starvation.

    Balances priority with time since last update to prevent starvation.

    Args:
        entities: List of (entity, guid, priority, estimated_size) tuples
        budget: Available bandwidth budget
        last_sent_times: Map of GUID to last sent timestamp

    Returns:
        List of (entity, guid) tuples that fit within budget
    """
    if not entities:
        return []

    now = time.time()
    budget.refill()
    available = budget.available_bytes

    # Calculate effective priority with starvation adjustment
    adjusted_entities = []
    for entity, guid, priority, size in entities:
        last_sent = last_sent_times.get(guid, 0.0)
        starvation = now - last_sent

        # Apply starvation boost
        effective_priority = priority
        if starvation > MAX_STARVATION_TIME:
            boost = (starvation / MAX_STARVATION_TIME) * STARVATION_PRIORITY_BOOST
            effective_priority = min(priority + boost, _config.MAX_PRIORITY_WITH_BOOST)

        adjusted_entities.append((entity, guid, effective_priority, size))

    # Sort by effective priority
    adjusted_entities.sort(key=lambda x: x[2], reverse=True)

    result = []
    remaining = available

    for entity, guid, _, size in adjusted_entities:
        if size <= remaining:
            result.append((entity, guid))
            remaining -= size
            # Update last sent time
            last_sent_times[guid] = now
        elif remaining < _config.MIN_UPDATE_PACKET_SIZE:
            break

    return result


@dataclass
class BandwidthManager:
    """Central manager for connection bandwidth allocation.

    Coordinates bandwidth budgets across multiple connections and channels.
    """
    default_max_bps: int = DEFAULT_MAX_BPS
    default_burst_bps: int = DEFAULT_BURST_BPS

    # Per-connection budgets
    _connection_budgets: dict[int, BandwidthBudget] = field(default_factory=dict)

    # Per-entity last sent times (for anti-starvation)
    _last_sent_times: dict[int, dict[int, float]] = field(default_factory=dict)

    # Priority queues per connection
    _priority_queues: dict[int, PriorityQueue] = field(default_factory=dict)

    def get_budget(self, connection_id: int) -> BandwidthBudget:
        """Get or create bandwidth budget for a connection.

        Args:
            connection_id: Connection identifier

        Returns:
            BandwidthBudget for the connection
        """
        if connection_id not in self._connection_budgets:
            self._connection_budgets[connection_id] = BandwidthBudget(
                max_bps=self.default_max_bps * 8,
                burst_bps=self.default_burst_bps * 8
            )
        return self._connection_budgets[connection_id]

    def get_queue(self, connection_id: int) -> PriorityQueue:
        """Get or create priority queue for a connection.

        Args:
            connection_id: Connection identifier

        Returns:
            PriorityQueue for the connection
        """
        if connection_id not in self._priority_queues:
            self._priority_queues[connection_id] = PriorityQueue()
        return self._priority_queues[connection_id]

    def queue_entity(
        self,
        connection_id: int,
        entity: Any,
        guid: int,
        priority: float,
        estimated_size: int = _config.DEFAULT_ENTITY_SERIALIZATION_SIZE
    ) -> None:
        """Queue an entity for replication to a connection.

        Args:
            connection_id: Target connection
            entity: Entity to replicate
            guid: Network GUID
            priority: Replication priority
            estimated_size: Estimated serialization size
        """
        queue = self.get_queue(connection_id)
        queue.add(entity, guid, priority, estimated_size)

    def allocate(self, connection_id: int) -> list[tuple[Any, int]]:
        """Allocate bandwidth and return entities to send.

        Args:
            connection_id: Connection to allocate for

        Returns:
            List of (entity, guid) tuples to send
        """
        budget = self.get_budget(connection_id)
        queue = self.get_queue(connection_id)

        # Get last sent times for this connection
        if connection_id not in self._last_sent_times:
            self._last_sent_times[connection_id] = {}
        last_sent = self._last_sent_times[connection_id]

        # Collect entities from queue
        entities = []
        temp_entries = []

        while queue:
            entry = queue.pop()
            if entry:
                entities.append((
                    entry.entity,
                    entry.guid,
                    -entry.priority,  # Restore positive priority
                    entry.estimated_size
                ))
                temp_entries.append(entry)
            else:
                break

        # Re-add entries to queue (they'll be updated when marked sent)
        for entry in temp_entries:
            queue.add(entry.entity, entry.guid, -entry.priority, entry.estimated_size)

        # Allocate bandwidth
        result = allocate_bandwidth_fair(entities, budget, last_sent)

        # Mark allocated entities as sent
        for _, guid in result:
            queue.mark_sent(guid)
            budget.consume(_config.DEFAULT_ENTITY_SERIALIZATION_SIZE)

        return result

    def mark_sent(
        self,
        connection_id: int,
        guid: int,
        size_bytes: int
    ) -> None:
        """Mark an entity as sent and consume bandwidth.

        Args:
            connection_id: Connection ID
            guid: Entity GUID
            size_bytes: Actual bytes sent
        """
        budget = self.get_budget(connection_id)
        budget.consume(size_bytes)

        queue = self.get_queue(connection_id)
        queue.mark_sent(guid)

        if connection_id not in self._last_sent_times:
            self._last_sent_times[connection_id] = {}
        self._last_sent_times[connection_id][guid] = time.time()

    def remove_connection(self, connection_id: int) -> None:
        """Remove a connection from tracking.

        Args:
            connection_id: Connection to remove
        """
        self._connection_budgets.pop(connection_id, None)
        self._priority_queues.pop(connection_id, None)
        self._last_sent_times.pop(connection_id, None)

    def remove_entity(self, guid: int) -> None:
        """Remove an entity from all connection queues.

        Args:
            guid: Entity GUID to remove
        """
        for queue in self._priority_queues.values():
            queue.remove(guid)

        for last_sent in self._last_sent_times.values():
            last_sent.pop(guid, None)

    def get_stats(self, connection_id: int) -> dict[str, Any]:
        """Get bandwidth statistics for a connection.

        Args:
            connection_id: Connection ID

        Returns:
            Dictionary of statistics
        """
        budget = self.get_budget(connection_id)
        queue = self.get_queue(connection_id)

        return {
            'max_bps': budget.max_bps // 8,
            'available_bytes': budget.available_bytes,
            'queued_entities': len(queue),
            'utilization': 1.0 - (budget.current_tokens / budget.burst_bps)
        }
