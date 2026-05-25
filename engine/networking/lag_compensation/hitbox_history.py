"""
Historical hitbox tracking for lag compensation.

Maintains a history of entity hitbox positions for server-side
hit validation. This allows accurate hit detection at the positions
where entities appeared to the shooting client.

The hitbox history is separate from full world state history for
efficiency - we only need position and bounds, not full entity state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque

from engine.networking.config import (
    DEFAULT_HITBOX_HISTORY_FRAMES,
    DEFAULT_TICK_RATE,
    HITBOX_CACHE_MULTIPLIER,
)


# Type aliases
Vector3 = Tuple[float, float, float]
EntityId = int


@dataclass
class Bounds:
    """
    Axis-aligned bounding box for hitbox calculations.

    Represents a box with min/max corners in 3D space.
    """

    min_point: Vector3 = field(default_factory=lambda: (0.0, 0.0, 0.0))
    """Minimum corner (x, y, z)."""

    max_point: Vector3 = field(default_factory=lambda: (1.0, 1.0, 1.0))
    """Maximum corner (x, y, z)."""

    @property
    def center(self) -> Vector3:
        """Get the center point of the bounds."""
        return (
            (self.min_point[0] + self.max_point[0]) / 2,
            (self.min_point[1] + self.max_point[1]) / 2,
            (self.min_point[2] + self.max_point[2]) / 2,
        )

    @property
    def size(self) -> Vector3:
        """Get the size of the bounds."""
        return (
            self.max_point[0] - self.min_point[0],
            self.max_point[1] - self.min_point[1],
            self.max_point[2] - self.min_point[2],
        )

    @property
    def extents(self) -> Vector3:
        """Get the half-size (extents) of the bounds."""
        return (
            (self.max_point[0] - self.min_point[0]) / 2,
            (self.max_point[1] - self.min_point[1]) / 2,
            (self.max_point[2] - self.min_point[2]) / 2,
        )

    def contains_point(self, point: Vector3) -> bool:
        """Check if a point is inside the bounds."""
        return (
            self.min_point[0] <= point[0] <= self.max_point[0] and
            self.min_point[1] <= point[1] <= self.max_point[1] and
            self.min_point[2] <= point[2] <= self.max_point[2]
        )

    def intersects(self, other: Bounds) -> bool:
        """Check if this bounds intersects another."""
        return (
            self.min_point[0] <= other.max_point[0] and
            self.max_point[0] >= other.min_point[0] and
            self.min_point[1] <= other.max_point[1] and
            self.max_point[1] >= other.min_point[1] and
            self.min_point[2] <= other.max_point[2] and
            self.max_point[2] >= other.min_point[2]
        )

    def translated(self, offset: Vector3) -> Bounds:
        """Create a new bounds translated by offset."""
        return Bounds(
            min_point=(
                self.min_point[0] + offset[0],
                self.min_point[1] + offset[1],
                self.min_point[2] + offset[2],
            ),
            max_point=(
                self.max_point[0] + offset[0],
                self.max_point[1] + offset[1],
                self.max_point[2] + offset[2],
            ),
        )

    @staticmethod
    def from_center_extents(center: Vector3, extents: Vector3) -> Bounds:
        """Create bounds from center point and half-sizes."""
        return Bounds(
            min_point=(
                center[0] - extents[0],
                center[1] - extents[1],
                center[2] - extents[2],
            ),
            max_point=(
                center[0] + extents[0],
                center[1] + extents[1],
                center[2] + extents[2],
            ),
        )

    def copy(self) -> Bounds:
        """Create a copy of this bounds."""
        return Bounds(
            min_point=self.min_point,
            max_point=self.max_point,
        )


@dataclass
class HitboxSnapshot:
    """
    A snapshot of an entity's hitbox at a point in time.

    Contains position and bounds for hit detection.
    """

    entity_id: EntityId
    """Unique identifier for the entity."""

    position: Vector3
    """World position of the entity."""

    bounds: Bounds
    """Local-space hitbox bounds."""

    timestamp: float
    """Server timestamp when recorded."""

    tick: int = 0
    """Server tick number."""

    is_active: bool = True
    """Whether this hitbox is active (can be hit)."""

    def get_world_bounds(self) -> Bounds:
        """Get hitbox bounds in world space."""
        return self.bounds.translated(self.position)

    def copy(self) -> HitboxSnapshot:
        """Create a copy of this snapshot."""
        return HitboxSnapshot(
            entity_id=self.entity_id,
            position=self.position,
            bounds=self.bounds.copy(),
            timestamp=self.timestamp,
            tick=self.tick,
            is_active=self.is_active,
        )


@dataclass
class EntityHitboxHistory:
    """History of hitbox snapshots for a single entity."""

    entity_id: EntityId
    """Entity this history belongs to."""

    snapshots: deque = field(default_factory=lambda: deque(maxlen=DEFAULT_HITBOX_HISTORY_FRAMES))
    """Circular buffer of snapshots."""

    default_bounds: Bounds = field(default_factory=Bounds)
    """Default bounds for this entity type."""

    def add_snapshot(self, snapshot: HitboxSnapshot) -> None:
        """Add a snapshot to history."""
        self.snapshots.append(snapshot)

    def get_at_time(self, timestamp: float) -> Optional[HitboxSnapshot]:
        """Get snapshot closest to timestamp."""
        if not self.snapshots:
            return None

        best: Optional[HitboxSnapshot] = None
        best_diff = float('inf')

        for snapshot in self.snapshots:
            diff = abs(snapshot.timestamp - timestamp)
            if diff < best_diff:
                best_diff = diff
                best = snapshot

        return best


class HitboxHistory:
    """
    Manages historical hitbox data for all tracked entities.

    Provides efficient lookup of hitbox positions at any point
    in the recorded history for lag-compensated hit detection.

    Example:
        history = HitboxHistory(max_frames=60)

        # Each tick:
        for entity in entities:
            history.record(entity.id, entity.position, entity.hitbox)

        # When checking a shot:
        client_time = calculate_view_time(...)
        hitbox = history.get_hitbox_at_time(target_id, client_time)
        if hitbox and ray_intersects_bounds(ray, hitbox.get_world_bounds()):
            # Hit!
    """

    def __init__(
        self,
        max_frames: int = DEFAULT_HITBOX_HISTORY_FRAMES,
        tick_rate: float = DEFAULT_TICK_RATE,
    ) -> None:
        """
        Initialize the hitbox history.

        Args:
            max_frames: Maximum frames of history to keep per entity.
            tick_rate: Server tick rate for time calculations.
        """
        self._max_frames = max_frames
        self._tick_rate = tick_rate
        self._current_tick: int = 0
        self._current_timestamp: float = 0.0

        # Entity ID -> history
        self._histories: Dict[EntityId, EntityHitboxHistory] = {}

        # Cached frame lookup for efficiency
        self._frame_cache: Dict[int, List[HitboxSnapshot]] = {}
        self._cache_max_size = max_frames * HITBOX_CACHE_MULTIPLIER

    @property
    def max_frames(self) -> int:
        """Maximum frames of history per entity."""
        return self._max_frames

    @property
    def entity_count(self) -> int:
        """Number of tracked entities."""
        return len(self._histories)

    @property
    def current_tick(self) -> int:
        """Current server tick."""
        return self._current_tick

    def record(
        self,
        entity_id: EntityId,
        position: Vector3,
        bounds: Bounds,
        timestamp: Optional[float] = None,
        tick: Optional[int] = None,
        is_active: bool = True,
    ) -> None:
        """
        Record a hitbox snapshot for an entity.

        Args:
            entity_id: The entity's unique identifier.
            position: Current world position.
            bounds: Current hitbox bounds (local space).
            timestamp: Optional override timestamp.
            tick: Optional override tick number.
            is_active: Whether the hitbox can be hit.
        """
        timestamp = timestamp if timestamp is not None else self._current_timestamp
        tick = tick if tick is not None else self._current_tick

        snapshot = HitboxSnapshot(
            entity_id=entity_id,
            position=position,
            bounds=bounds,
            timestamp=timestamp,
            tick=tick,
            is_active=is_active,
        )

        # Get or create entity history
        if entity_id not in self._histories:
            self._histories[entity_id] = EntityHitboxHistory(
                entity_id=entity_id,
                snapshots=deque(maxlen=self._max_frames),
                default_bounds=bounds.copy(),
            )

        self._histories[entity_id].add_snapshot(snapshot)

        # Update frame cache
        if tick not in self._frame_cache:
            self._frame_cache[tick] = []
            # Cleanup old cache entries
            if len(self._frame_cache) > self._cache_max_size:
                oldest_tick = min(self._frame_cache.keys())
                del self._frame_cache[oldest_tick]

        self._frame_cache[tick].append(snapshot)

    def set_tick(self, tick: int, timestamp: float) -> None:
        """
        Set the current tick and timestamp.

        Call this at the start of each server tick.

        Args:
            tick: Current server tick number.
            timestamp: Current server timestamp.
        """
        self._current_tick = tick
        self._current_timestamp = timestamp

    def get_hitbox_at_time(
        self,
        entity_id: EntityId,
        timestamp: float,
    ) -> Optional[HitboxSnapshot]:
        """
        Get an entity's hitbox at a specific timestamp.

        Args:
            entity_id: The entity to look up.
            timestamp: The timestamp to query.

        Returns:
            The hitbox snapshot closest to the timestamp.
        """
        history = self._histories.get(entity_id)
        if history is None:
            return None

        return history.get_at_time(timestamp)

    def get_hitbox_at_tick(
        self,
        entity_id: EntityId,
        tick: int,
    ) -> Optional[HitboxSnapshot]:
        """
        Get an entity's hitbox at a specific tick.

        Args:
            entity_id: The entity to look up.
            tick: The tick number to query.

        Returns:
            The hitbox snapshot at that tick.
        """
        history = self._histories.get(entity_id)
        if history is None:
            return None

        for snapshot in history.snapshots:
            if snapshot.tick == tick:
                return snapshot

        return None

    def get_all_hitboxes_at_time(
        self,
        timestamp: float,
        active_only: bool = True,
    ) -> List[HitboxSnapshot]:
        """
        Get all entity hitboxes at a specific timestamp.

        Args:
            timestamp: The timestamp to query.
            active_only: If True, only return active hitboxes.

        Returns:
            List of hitbox snapshots for all tracked entities.
        """
        results: List[HitboxSnapshot] = []

        for entity_id, history in self._histories.items():
            snapshot = history.get_at_time(timestamp)
            if snapshot is not None:
                if not active_only or snapshot.is_active:
                    results.append(snapshot)

        return results

    def get_all_hitboxes_at_tick(
        self,
        tick: int,
        active_only: bool = True,
    ) -> List[HitboxSnapshot]:
        """
        Get all entity hitboxes at a specific tick.

        Uses cached lookup for efficiency.

        Args:
            tick: The tick number to query.
            active_only: If True, only return active hitboxes.

        Returns:
            List of hitbox snapshots at that tick.
        """
        if tick in self._frame_cache:
            snapshots = self._frame_cache[tick]
            if active_only:
                return [s for s in snapshots if s.is_active]
            return list(snapshots)

        # Fall back to individual lookups
        results: List[HitboxSnapshot] = []
        for entity_id, history in self._histories.items():
            for snapshot in history.snapshots:
                if snapshot.tick == tick:
                    if not active_only or snapshot.is_active:
                        results.append(snapshot)
                    break

        return results

    def get_interpolated_hitbox(
        self,
        entity_id: EntityId,
        timestamp: float,
    ) -> Optional[HitboxSnapshot]:
        """
        Get interpolated hitbox at exact timestamp.

        Interpolates between surrounding snapshots for more
        accurate hit detection.

        Args:
            entity_id: The entity to look up.
            timestamp: Exact timestamp to interpolate to.

        Returns:
            Interpolated hitbox snapshot.
        """
        history = self._histories.get(entity_id)
        if history is None or len(history.snapshots) < 2:
            return self.get_hitbox_at_time(entity_id, timestamp)

        # Find surrounding snapshots
        before: Optional[HitboxSnapshot] = None
        after: Optional[HitboxSnapshot] = None

        for snapshot in history.snapshots:
            if snapshot.timestamp > timestamp:
                after = snapshot
                break
            before = snapshot

        if before is None or after is None:
            return self.get_hitbox_at_time(entity_id, timestamp)

        # Calculate interpolation factor
        duration = after.timestamp - before.timestamp
        if duration <= 0:
            return before

        t = (timestamp - before.timestamp) / duration

        # Interpolate position
        position = (
            before.position[0] + (after.position[0] - before.position[0]) * t,
            before.position[1] + (after.position[1] - before.position[1]) * t,
            before.position[2] + (after.position[2] - before.position[2]) * t,
        )

        return HitboxSnapshot(
            entity_id=entity_id,
            position=position,
            bounds=before.bounds,  # Don't interpolate bounds
            timestamp=timestamp,
            tick=before.tick,
            is_active=before.is_active and after.is_active,
        )

    def remove_entity(self, entity_id: EntityId) -> bool:
        """
        Remove an entity from tracking.

        Args:
            entity_id: The entity to remove.

        Returns:
            True if entity was being tracked.
        """
        if entity_id in self._histories:
            del self._histories[entity_id]
            return True
        return False

    def clear(self) -> None:
        """Clear all hitbox history."""
        self._histories.clear()
        self._frame_cache.clear()

    def get_entity_ids(self) -> List[EntityId]:
        """Get list of all tracked entity IDs."""
        return list(self._histories.keys())

    def has_entity(self, entity_id: EntityId) -> bool:
        """Check if an entity is being tracked."""
        return entity_id in self._histories
