"""
Server-side rewind manager for lag compensation.

Maintains a history of world states that can be rewound to for
server-side hit detection. This allows the server to validate
shots at the positions where entities appeared to the client.

The rewind process:
1. Client fires at time T_client
2. Server receives at T_server = T_client + latency
3. Server rewinds world to T_client
4. Performs hit detection at historical positions
5. Restores current state and applies results
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import deque
import copy

from engine.networking.config import (
    DEFAULT_MAX_REWIND_TIME_MS,
    DEFAULT_TICK_RATE,
    calculate_max_history_frames,
)


# Type aliases
Vector3 = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]
EntityId = int


@dataclass
class EntityState:
    """State of a single entity at a point in time."""

    entity_id: EntityId
    """Unique identifier for this entity."""

    position: Vector3
    """Entity position."""

    rotation: Optional[Quaternion] = None
    """Entity rotation (optional)."""

    velocity: Vector3 = field(default_factory=lambda: (0.0, 0.0, 0.0))
    """Entity velocity."""

    custom_data: Dict[str, Any] = field(default_factory=dict)
    """Additional entity state for game-specific data."""

    def copy(self) -> EntityState:
        """Create a deep copy of this state."""
        return EntityState(
            entity_id=self.entity_id,
            position=self.position,
            rotation=self.rotation,
            velocity=self.velocity,
            custom_data=copy.deepcopy(self.custom_data),
        )


@dataclass
class WorldState:
    """
    Complete snapshot of relevant world state at a point in time.

    Contains all entities that may need to be rewound for
    lag compensation (typically all players and projectiles).
    """

    entities: Dict[EntityId, EntityState] = field(default_factory=dict)
    """Map of entity ID to entity state."""

    timestamp: float = 0.0
    """Server timestamp when this state was recorded."""

    tick: int = 0
    """Server tick number."""

    def add_entity(self, state: EntityState) -> None:
        """Add or update an entity in this world state."""
        self.entities[state.entity_id] = state

    def get_entity(self, entity_id: EntityId) -> Optional[EntityState]:
        """Get entity state by ID."""
        return self.entities.get(entity_id)

    def remove_entity(self, entity_id: EntityId) -> bool:
        """Remove an entity from the world state."""
        if entity_id in self.entities:
            del self.entities[entity_id]
            return True
        return False

    def copy(self) -> WorldState:
        """Create a deep copy of this world state."""
        new_state = WorldState(
            timestamp=self.timestamp,
            tick=self.tick,
        )
        for entity_id, entity_state in self.entities.items():
            new_state.entities[entity_id] = entity_state.copy()
        return new_state


@dataclass
class HistoryFrame:
    """
    A single frame in the rewind history.

    Contains the complete world state at a specific tick/timestamp.
    """

    tick: int
    """Server tick number for this frame."""

    timestamp: float
    """Server timestamp for this frame."""

    world_state: WorldState
    """Complete world state at this frame."""

    def get_entity_position(self, entity_id: EntityId) -> Optional[Vector3]:
        """Get an entity's position at this frame."""
        entity = self.world_state.get_entity(entity_id)
        if entity:
            return entity.position
        return None


class RewindManager:
    """
    Manages world state history for lag compensation.

    Records world states each tick and provides methods to rewind
    to historical states for server-side hit validation.

    Example:
        manager = RewindManager(max_history_ms=200)

        # Each tick:
        manager.record_frame(current_tick, world_state)

        # When validating a shot:
        client_time = calculate_client_view_time(server_time, client_rtt)
        historical_state = manager.rewind_to(client_time)

        # Perform hit detection with historical_state
        hit = raycast(origin, direction, historical_state)

        # Restore current state
        manager.restore_to_current()

        # Apply hit result
        if hit:
            apply_damage(hit.entity)
    """

    def __init__(
        self,
        max_history_ms: float = DEFAULT_MAX_REWIND_TIME_MS,
        tick_rate: float = DEFAULT_TICK_RATE,
    ) -> None:
        """
        Initialize the rewind manager.

        Args:
            max_history_ms: Maximum history to keep in milliseconds.
            tick_rate: Server tick rate for time calculations.
        """
        self._max_history_ms = max_history_ms
        self._tick_rate = tick_rate
        # Prevent division by zero
        self._ms_per_tick = 1000.0 / tick_rate if tick_rate > 0 else 1000.0

        # Calculate max frames based on tick rate and history time
        self._max_frames = calculate_max_history_frames(max_history_ms, tick_rate)

        self._history: deque[HistoryFrame] = deque(maxlen=self._max_frames)
        self._current_state: Optional[WorldState] = None
        self._is_rewound: bool = False
        self._rewind_frame: Optional[HistoryFrame] = None

    @property
    def max_history_ms(self) -> float:
        """Maximum history duration in milliseconds."""
        return self._max_history_ms

    @property
    def frame_count(self) -> int:
        """Number of frames in history."""
        return len(self._history)

    @property
    def is_rewound(self) -> bool:
        """Whether currently in rewound state."""
        return self._is_rewound

    @property
    def oldest_timestamp(self) -> Optional[float]:
        """Get the oldest recorded timestamp."""
        if self._history:
            return self._history[0].timestamp
        return None

    @property
    def newest_timestamp(self) -> Optional[float]:
        """Get the newest recorded timestamp."""
        if self._history:
            return self._history[-1].timestamp
        return None

    def record_frame(
        self,
        tick: int,
        world_state: WorldState,
    ) -> None:
        """
        Record a world state frame for history.

        Should be called each server tick after physics/simulation.

        Args:
            tick: Current server tick number.
            world_state: Current world state to record.
        """
        # Store current state reference
        self._current_state = world_state

        # Create history frame with deep copy
        frame = HistoryFrame(
            tick=tick,
            timestamp=world_state.timestamp,
            world_state=world_state.copy(),
        )

        self._history.append(frame)

    def get_frame_at_time(
        self,
        timestamp: float,
    ) -> Optional[HistoryFrame]:
        """
        Find the history frame closest to the given timestamp.

        Args:
            timestamp: The timestamp to look up.

        Returns:
            The closest history frame, or None if no history.
        """
        if not self._history:
            return None

        # Check bounds
        if timestamp <= self._history[0].timestamp:
            return self._history[0]
        if timestamp >= self._history[-1].timestamp:
            return self._history[-1]

        # Binary search for closest frame
        best_frame: Optional[HistoryFrame] = None
        best_diff = float('inf')

        for frame in self._history:
            diff = abs(frame.timestamp - timestamp)
            if diff < best_diff:
                best_diff = diff
                best_frame = frame
            elif frame.timestamp > timestamp:
                # Passed the target, stop searching
                break

        return best_frame

    def get_frame_at_tick(self, tick: int) -> Optional[HistoryFrame]:
        """
        Find the history frame for a specific tick.

        Args:
            tick: The tick number to look up.

        Returns:
            The history frame if found, None otherwise.
        """
        for frame in self._history:
            if frame.tick == tick:
                return frame
        return None

    def get_interpolated_frame(
        self,
        timestamp: float,
    ) -> Optional[HistoryFrame]:
        """
        Get interpolated frame at exact timestamp.

        If timestamp falls between two frames, creates an interpolated
        frame for more accurate lag compensation.

        Args:
            timestamp: Exact timestamp to interpolate to.

        Returns:
            Interpolated history frame.
        """
        if not self._history or len(self._history) < 2:
            return self.get_frame_at_time(timestamp)

        # Find surrounding frames
        before: Optional[HistoryFrame] = None
        after: Optional[HistoryFrame] = None

        for i, frame in enumerate(self._history):
            if frame.timestamp > timestamp:
                after = frame
                if i > 0:
                    before = self._history[i - 1]
                break
            before = frame

        if before is None or after is None:
            return self.get_frame_at_time(timestamp)

        # Interpolate between frames
        duration = after.timestamp - before.timestamp
        if duration <= 0:
            return before

        t = (timestamp - before.timestamp) / duration

        # Create interpolated world state
        interpolated_state = WorldState(
            timestamp=timestamp,
            tick=before.tick,  # Use earlier tick
        )

        # Interpolate each entity
        for entity_id, before_entity in before.world_state.entities.items():
            after_entity = after.world_state.get_entity(entity_id)

            if after_entity:
                # Interpolate position
                position = self._lerp_vector(
                    before_entity.position,
                    after_entity.position,
                    t,
                )

                # Interpolate velocity
                velocity = self._lerp_vector(
                    before_entity.velocity,
                    after_entity.velocity,
                    t,
                )

                interpolated_state.add_entity(EntityState(
                    entity_id=entity_id,
                    position=position,
                    velocity=velocity,
                    rotation=before_entity.rotation,  # Skip rotation interp for simplicity
                    custom_data=before_entity.custom_data,
                ))
            else:
                # Entity not in after frame, use before
                interpolated_state.add_entity(before_entity.copy())

        return HistoryFrame(
            tick=before.tick,
            timestamp=timestamp,
            world_state=interpolated_state,
        )

    def _lerp_vector(
        self,
        a: Vector3,
        b: Vector3,
        t: float,
    ) -> Vector3:
        """Linear interpolation between vectors."""
        return (
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t,
        )

    def rewind_to(
        self,
        timestamp: float,
        interpolate: bool = True,
    ) -> Optional[WorldState]:
        """
        Rewind to a specific timestamp for lag compensation.

        Args:
            timestamp: The timestamp to rewind to.
            interpolate: Whether to interpolate between frames.

        Returns:
            The historical world state at that time.
        """
        if self._is_rewound:
            raise RuntimeError("Already rewound - call restore_to_current() first")

        if interpolate:
            frame = self.get_interpolated_frame(timestamp)
        else:
            frame = self.get_frame_at_time(timestamp)

        if frame is None:
            return None

        self._is_rewound = True
        self._rewind_frame = frame

        return frame.world_state

    def restore_to_current(self) -> Optional[WorldState]:
        """
        Restore to the current world state after rewind.

        Returns:
            The current world state.
        """
        self._is_rewound = False
        self._rewind_frame = None
        return self._current_state

    def get_entity_at_time(
        self,
        entity_id: EntityId,
        timestamp: float,
    ) -> Optional[EntityState]:
        """
        Get a specific entity's state at a timestamp.

        Convenience method for single-entity lookups.

        Args:
            entity_id: The entity to look up.
            timestamp: The timestamp to look up.

        Returns:
            The entity's historical state.
        """
        frame = self.get_frame_at_time(timestamp)
        if frame:
            return frame.world_state.get_entity(entity_id)
        return None

    def clear_history(self) -> None:
        """Clear all recorded history."""
        self._history.clear()
        self._rewind_frame = None

    def get_history_time_range(self) -> Tuple[float, float]:
        """
        Get the time range covered by history.

        Returns:
            Tuple of (oldest_time, newest_time).
        """
        if not self._history:
            return (0.0, 0.0)
        return (self._history[0].timestamp, self._history[-1].timestamp)

    def can_rewind_to(self, timestamp: float) -> bool:
        """
        Check if we have history for the given timestamp.

        Args:
            timestamp: The timestamp to check.

        Returns:
            True if rewind is possible.
        """
        if not self._history:
            return False

        oldest = self._history[0].timestamp
        newest = self._history[-1].timestamp

        # Allow small buffer beyond recorded time
        buffer = self._ms_per_tick / 1000.0
        return oldest - buffer <= timestamp <= newest + buffer
