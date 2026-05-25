"""
Debris Management System.

Handles spawning, lifetime management, pooling, and cleanup of debris
pieces created during destruction events.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set, Callable, Any
from collections import deque
from enum import IntEnum

from .config import (
    DEBRIS_LIFETIME,
    DEBRIS_MIN_LIFETIME,
    DEBRIS_MAX_LIFETIME,
    MAX_ACTIVE_DEBRIS,
    DEBRIS_POOL_INITIAL_SIZE,
    DEBRIS_POOL_GROWTH_FACTOR,
    DEBRIS_MERGE_DISTANCE,
    DEBRIS_MIN_VELOCITY,
    DEBRIS_SLEEP_TIME,
    DEBRIS_UPDATE_BATCH_SIZE,
    DEBRIS_ANGULAR_VELOCITY_MIN,
    DEBRIS_ANGULAR_VELOCITY_MAX,
    DEBRIS_IMPORTANCE_VOLUME_MULTIPLIER,
    DEBRIS_LOD_DISTANCE_FULL,
    DEBRIS_LOD_DISTANCE_REDUCED,
    DEBRIS_LOD_DISTANCE_SIMPLE,
    DebrisState,
)
from .fracture_voronoi import (
    Vec3,
    BoundingBox,
    Chunk,
    vec3_add,
    vec3_sub,
    vec3_mul,
    vec3_length,
    vec3_distance,
)


class DebrisLOD(IntEnum):
    """Level of detail for debris rendering."""
    FULL = 0       # Full mesh, full physics
    REDUCED = 1    # Simplified mesh, full physics
    SIMPLE = 2     # Box proxy, simplified physics
    PARTICLE = 3   # Point sprite, no physics


@dataclass(slots=True)
class Debris:
    """
    Represents a single debris piece.

    Attributes:
        id: Unique identifier.
        body_id: Physics body ID (external reference).
        chunk: Reference to the mesh chunk.
        lifetime: Total lifetime in seconds.
        spawn_time: Time when debris was spawned.
        position: Current world position.
        velocity: Current velocity.
        angular_velocity: Current angular velocity.
        state: Current debris state.
        lod: Current level of detail.
        importance: Importance score for cleanup priority.
        sleep_timer: Time spent below velocity threshold.
        parent_id: ID of the original destructible object.
        generation: Fracture generation (0 = first fracture, 1 = re-fracture, etc.).
    """
    id: int
    body_id: Optional[int] = None
    chunk: Optional[Chunk] = None
    lifetime: float = DEBRIS_LIFETIME
    spawn_time: float = 0.0
    position: Vec3 = (0.0, 0.0, 0.0)
    velocity: Vec3 = (0.0, 0.0, 0.0)
    angular_velocity: Vec3 = (0.0, 0.0, 0.0)
    state: DebrisState = DebrisState.ACTIVE
    lod: DebrisLOD = DebrisLOD.FULL
    importance: float = 1.0
    sleep_timer: float = 0.0
    parent_id: Optional[int] = None
    generation: int = 0

    @property
    def age(self) -> float:
        """Current age of the debris."""
        return time.time() - self.spawn_time

    @property
    def remaining_lifetime(self) -> float:
        """Remaining lifetime before cleanup."""
        return max(0.0, self.lifetime - self.age)

    @property
    def is_expired(self) -> bool:
        """Whether debris has exceeded its lifetime."""
        return self.age >= self.lifetime

    @property
    def speed(self) -> float:
        """Current speed (velocity magnitude)."""
        return vec3_length(self.velocity)

    def reset(self) -> None:
        """Reset debris to initial state for pooling."""
        self.body_id = None
        self.chunk = None
        self.lifetime = DEBRIS_LIFETIME
        self.spawn_time = 0.0
        self.position = (0.0, 0.0, 0.0)
        self.velocity = (0.0, 0.0, 0.0)
        self.angular_velocity = (0.0, 0.0, 0.0)
        self.state = DebrisState.POOLED
        self.lod = DebrisLOD.FULL
        self.importance = 1.0
        self.sleep_timer = 0.0
        self.parent_id = None
        self.generation = 0


@dataclass
class DebrisSpawnParams:
    """
    Parameters for spawning debris.

    Attributes:
        chunk: Mesh chunk for the debris.
        position: Spawn position.
        velocity: Initial velocity.
        angular_velocity: Initial angular velocity.
        lifetime: Debris lifetime.
        importance: Importance score.
        parent_id: ID of source object.
        generation: Fracture generation.
    """
    chunk: Chunk
    position: Vec3
    velocity: Vec3 = (0.0, 0.0, 0.0)
    angular_velocity: Vec3 = (0.0, 0.0, 0.0)
    lifetime: float = DEBRIS_LIFETIME
    importance: float = 1.0
    parent_id: Optional[int] = None
    generation: int = 0


class DebrisPool:
    """
    Object pool for debris instances.

    Provides efficient allocation and recycling of debris objects
    to minimize garbage collection pressure.
    """

    __slots__ = ('_pool', '_all_debris', '_next_id', '_max_size')

    def __init__(self, initial_size: int = DEBRIS_POOL_INITIAL_SIZE) -> None:
        """
        Initialize debris pool.

        Args:
            initial_size: Initial pool size.
        """
        self._pool: deque[Debris] = deque()
        self._all_debris: Dict[int, Debris] = {}
        self._next_id = 0
        self._max_size = initial_size

        # Pre-allocate pool
        for _ in range(initial_size):
            debris = Debris(id=self._next_id)
            debris.state = DebrisState.POOLED
            self._pool.append(debris)
            self._all_debris[self._next_id] = debris
            self._next_id += 1

    @property
    def pool_size(self) -> int:
        """Number of debris in the pool (available for use)."""
        return len(self._pool)

    @property
    def total_count(self) -> int:
        """Total number of debris instances."""
        return len(self._all_debris)

    @property
    def active_count(self) -> int:
        """Number of active debris."""
        return sum(
            1 for d in self._all_debris.values()
            if d.state not in (DebrisState.POOLED, DebrisState.PENDING_CLEANUP)
        )

    def acquire(self) -> Debris:
        """
        Acquire a debris instance from the pool.

        Returns:
            A debris instance ready for use.
        """
        if self._pool:
            debris = self._pool.popleft()
            debris.state = DebrisState.ACTIVE
            return debris

        # Pool exhausted - create new
        debris = Debris(id=self._next_id)
        self._all_debris[self._next_id] = debris
        self._next_id += 1

        return debris

    def release(self, debris: Debris) -> None:
        """
        Return a debris instance to the pool.

        Args:
            debris: Debris to return.
        """
        debris.reset()
        self._pool.append(debris)

    def get(self, debris_id: int) -> Optional[Debris]:
        """Get debris by ID."""
        return self._all_debris.get(debris_id)

    def clear(self) -> None:
        """Clear all debris and reset pool."""
        self._pool.clear()
        self._all_debris.clear()
        self._next_id = 0


class DebrisManager:
    """
    Manages debris lifecycle, updates, and cleanup.

    Handles spawning, aging, merging, and removal of debris pieces
    with configurable limits and LOD support.
    """

    __slots__ = (
        '_pool', '_active_debris', '_pending_spawn', '_pending_cleanup',
        '_max_active', '_merge_distance', '_sleep_velocity', '_sleep_time',
        '_lod_distances', '_on_debris_spawn', '_on_debris_destroy',
        '_current_time', '_batch_size'
    )

    def __init__(
        self,
        max_active: int = MAX_ACTIVE_DEBRIS,
        pool_size: int = DEBRIS_POOL_INITIAL_SIZE,
        merge_distance: float = DEBRIS_MERGE_DISTANCE,
        sleep_velocity: float = DEBRIS_MIN_VELOCITY,
        sleep_time: float = DEBRIS_SLEEP_TIME
    ) -> None:
        """
        Initialize debris manager.

        Args:
            max_active: Maximum number of active debris.
            pool_size: Initial pool size.
            merge_distance: Distance threshold for merging.
            sleep_velocity: Velocity threshold for sleep state.
            sleep_time: Time below threshold before cleanup.
        """
        self._pool = DebrisPool(pool_size)
        self._active_debris: Dict[int, Debris] = {}
        self._pending_spawn: List[DebrisSpawnParams] = []
        self._pending_cleanup: Set[int] = set()
        self._max_active = max_active
        self._merge_distance = merge_distance
        self._sleep_velocity = sleep_velocity
        self._sleep_time = sleep_time
        self._batch_size = DEBRIS_UPDATE_BATCH_SIZE
        self._current_time = 0.0

        # LOD distances (squared for efficiency)
        self._lod_distances: Dict[DebrisLOD, float] = {
            DebrisLOD.FULL: DEBRIS_LOD_DISTANCE_FULL ** 2,
            DebrisLOD.REDUCED: DEBRIS_LOD_DISTANCE_REDUCED ** 2,
            DebrisLOD.SIMPLE: DEBRIS_LOD_DISTANCE_SIMPLE ** 2,
            DebrisLOD.PARTICLE: float('inf')
        }

        # Callbacks
        self._on_debris_spawn: Optional[Callable[[Debris], None]] = None
        self._on_debris_destroy: Optional[Callable[[Debris], None]] = None

    @property
    def active_count(self) -> int:
        """Number of active debris pieces."""
        return len(self._active_debris)

    @property
    def max_active(self) -> int:
        """Maximum active debris limit."""
        return self._max_active

    @property
    def debris(self) -> Dict[int, Debris]:
        """All active debris."""
        return self._active_debris

    def set_callbacks(
        self,
        on_spawn: Optional[Callable[[Debris], None]] = None,
        on_destroy: Optional[Callable[[Debris], None]] = None
    ) -> None:
        """
        Set debris lifecycle callbacks.

        Args:
            on_spawn: Called when debris is spawned.
            on_destroy: Called when debris is destroyed.
        """
        self._on_debris_spawn = on_spawn
        self._on_debris_destroy = on_destroy

    def set_lod_distances(
        self,
        full: float = 10.0,
        reduced: float = 25.0,
        simple: float = 50.0
    ) -> None:
        """
        Set LOD transition distances.

        Args:
            full: Distance for full detail.
            reduced: Distance for reduced detail.
            simple: Distance for simple detail.
        """
        self._lod_distances[DebrisLOD.FULL] = full ** 2
        self._lod_distances[DebrisLOD.REDUCED] = reduced ** 2
        self._lod_distances[DebrisLOD.SIMPLE] = simple ** 2

    def spawn_debris(
        self,
        chunk: Chunk,
        velocity: Vec3,
        angular_velocity: Optional[Vec3] = None,
        lifetime: Optional[float] = None,
        importance: float = 1.0,
        parent_id: Optional[int] = None,
        generation: int = 0
    ) -> Optional[Debris]:
        """
        Spawn a new debris piece.

        Args:
            chunk: Mesh chunk for the debris.
            velocity: Initial velocity.
            angular_velocity: Initial angular velocity.
            lifetime: Debris lifetime.
            importance: Importance score (affects cleanup priority).
            parent_id: ID of source object.
            generation: Fracture generation.

        Returns:
            Spawned debris or None if limit reached.
        """
        # Check capacity
        if len(self._active_debris) >= self._max_active:
            # Try to make room by cleaning up low-importance debris
            if not self._force_cleanup(importance):
                return None

        # Acquire from pool
        debris = self._pool.acquire()

        # Initialize
        debris.chunk = chunk
        debris.position = chunk.centroid
        debris.velocity = velocity
        debris.angular_velocity = angular_velocity or (0.0, 0.0, 0.0)
        debris.lifetime = lifetime if lifetime else DEBRIS_LIFETIME
        debris.spawn_time = self._current_time or time.time()
        debris.importance = importance
        debris.parent_id = parent_id
        debris.generation = generation
        debris.state = DebrisState.ACTIVE
        debris.sleep_timer = 0.0
        debris.lod = DebrisLOD.FULL

        self._active_debris[debris.id] = debris

        if self._on_debris_spawn:
            self._on_debris_spawn(debris)

        return debris

    def spawn_debris_batch(
        self,
        params_list: List[DebrisSpawnParams]
    ) -> List[Debris]:
        """
        Spawn multiple debris pieces.

        Args:
            params_list: List of spawn parameters.

        Returns:
            List of spawned debris.
        """
        spawned = []

        for params in params_list:
            debris = self.spawn_debris(
                chunk=params.chunk,
                velocity=params.velocity,
                angular_velocity=params.angular_velocity,
                lifetime=params.lifetime,
                importance=params.importance,
                parent_id=params.parent_id,
                generation=params.generation
            )

            if debris:
                spawned.append(debris)
            else:
                break  # Hit capacity limit

        return spawned

    def update(
        self,
        dt: float,
        camera_position: Optional[Vec3] = None
    ) -> List[int]:
        """
        Update all debris pieces.

        Args:
            dt: Delta time since last update.
            camera_position: Camera position for LOD calculations.

        Returns:
            List of debris IDs that were cleaned up.
        """
        self._current_time += dt
        cleaned_up = []

        # Process in batches
        debris_list = list(self._active_debris.values())

        for i in range(0, len(debris_list), self._batch_size):
            batch = debris_list[i:i + self._batch_size]

            for debris in batch:
                # Skip if pending cleanup
                if debris.id in self._pending_cleanup:
                    continue

                # Update age and check expiration
                if debris.is_expired:
                    self._pending_cleanup.add(debris.id)
                    continue

                # Update sleep timer
                if debris.speed < self._sleep_velocity:
                    debris.sleep_timer += dt

                    if debris.sleep_timer >= self._sleep_time:
                        debris.state = DebrisState.SLEEPING
                else:
                    debris.sleep_timer = 0.0
                    if debris.state == DebrisState.SLEEPING:
                        debris.state = DebrisState.ACTIVE

                # Update LOD based on camera distance
                if camera_position:
                    debris.lod = self._calculate_lod(debris.position, camera_position)

        # Process cleanup
        for debris_id in self._pending_cleanup:
            if debris_id in self._active_debris:
                debris = self._active_debris[debris_id]

                if self._on_debris_destroy:
                    self._on_debris_destroy(debris)

                del self._active_debris[debris_id]
                self._pool.release(debris)
                cleaned_up.append(debris_id)

        self._pending_cleanup.clear()

        return cleaned_up

    def _calculate_lod(
        self,
        debris_position: Vec3,
        camera_position: Vec3
    ) -> DebrisLOD:
        """Calculate appropriate LOD level based on distance."""
        dx = debris_position[0] - camera_position[0]
        dy = debris_position[1] - camera_position[1]
        dz = debris_position[2] - camera_position[2]
        dist_sq = dx * dx + dy * dy + dz * dz

        if dist_sq < self._lod_distances[DebrisLOD.FULL]:
            return DebrisLOD.FULL
        elif dist_sq < self._lod_distances[DebrisLOD.REDUCED]:
            return DebrisLOD.REDUCED
        elif dist_sq < self._lod_distances[DebrisLOD.SIMPLE]:
            return DebrisLOD.SIMPLE
        else:
            return DebrisLOD.PARTICLE

    def _force_cleanup(self, min_importance: float) -> bool:
        """
        Force cleanup of low-importance debris to make room.

        Args:
            min_importance: Minimum importance to keep.

        Returns:
            True if room was made, False otherwise.
        """
        # Find candidates for cleanup
        candidates = [
            d for d in self._active_debris.values()
            if d.importance < min_importance or d.state == DebrisState.SLEEPING
        ]

        if not candidates:
            return False

        # Sort by importance and age (clean up oldest, least important first)
        candidates.sort(key=lambda d: (d.importance, -d.age))

        # Clean up enough to make room
        num_to_clean = max(1, len(self._active_debris) - self._max_active + 1)

        for debris in candidates[:num_to_clean]:
            self._pending_cleanup.add(debris.id)

        return True

    def merge_small_debris(
        self,
        min_volume: float = 0.01,
        merge_distance: Optional[float] = None
    ) -> int:
        """
        Merge small nearby debris pieces.

        Args:
            min_volume: Minimum volume to consider for merging.
            merge_distance: Distance threshold for merging.

        Returns:
            Number of debris pieces merged.
        """
        if merge_distance is None:
            merge_distance = self._merge_distance

        merge_distance_sq = merge_distance ** 2
        merged_count = 0

        # Find small debris
        small_debris = [
            d for d in self._active_debris.values()
            if d.chunk and d.chunk.volume < min_volume
        ]

        # Simple O(n^2) proximity check - could be optimized with spatial hash
        merged_ids: Set[int] = set()

        for i, debris_a in enumerate(small_debris):
            if debris_a.id in merged_ids:
                continue

            for debris_b in small_debris[i + 1:]:
                if debris_b.id in merged_ids:
                    continue

                # Check distance
                dist_sq = (
                    (debris_a.position[0] - debris_b.position[0]) ** 2 +
                    (debris_a.position[1] - debris_b.position[1]) ** 2 +
                    (debris_a.position[2] - debris_b.position[2]) ** 2
                )

                if dist_sq < merge_distance_sq:
                    # Merge B into A
                    merged_ids.add(debris_b.id)
                    self._pending_cleanup.add(debris_b.id)

                    # Update A with combined properties
                    if debris_a.chunk and debris_b.chunk:
                        debris_a.chunk.volume += debris_b.chunk.volume
                        debris_a.importance = max(debris_a.importance, debris_b.importance)

                    merged_count += 1

        return merged_count

    def get_debris_in_radius(
        self,
        center: Vec3,
        radius: float
    ) -> List[Debris]:
        """
        Get all debris within a radius of a point.

        Args:
            center: Center point.
            radius: Search radius.

        Returns:
            List of debris within radius.
        """
        radius_sq = radius ** 2
        result = []

        for debris in self._active_debris.values():
            dist_sq = (
                (debris.position[0] - center[0]) ** 2 +
                (debris.position[1] - center[1]) ** 2 +
                (debris.position[2] - center[2]) ** 2
            )

            if dist_sq <= radius_sq:
                result.append(debris)

        return result

    def get_debris_by_parent(self, parent_id: int) -> List[Debris]:
        """Get all debris from a specific parent object."""
        return [
            d for d in self._active_debris.values()
            if d.parent_id == parent_id
        ]

    def destroy_debris(self, debris_id: int) -> bool:
        """
        Immediately destroy a debris piece.

        Args:
            debris_id: ID of debris to destroy.

        Returns:
            True if debris was found and destroyed.
        """
        if debris_id in self._active_debris:
            self._pending_cleanup.add(debris_id)
            return True
        return False

    def destroy_all(self) -> int:
        """
        Destroy all active debris.

        Returns:
            Number of debris destroyed.
        """
        count = len(self._active_debris)
        self._pending_cleanup.update(self._active_debris.keys())
        return count

    def destroy_by_parent(self, parent_id: int) -> int:
        """
        Destroy all debris from a specific parent.

        Args:
            parent_id: Parent object ID.

        Returns:
            Number of debris destroyed.
        """
        count = 0
        for debris in self._active_debris.values():
            if debris.parent_id == parent_id:
                self._pending_cleanup.add(debris.id)
                count += 1
        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get debris manager statistics."""
        lod_counts = {lod: 0 for lod in DebrisLOD}
        state_counts = {state: 0 for state in DebrisState}
        total_volume = 0.0

        for debris in self._active_debris.values():
            lod_counts[debris.lod] += 1
            state_counts[debris.state] += 1
            if debris.chunk:
                total_volume += debris.chunk.volume

        return {
            'active_count': len(self._active_debris),
            'max_active': self._max_active,
            'pool_size': self._pool.pool_size,
            'pending_cleanup': len(self._pending_cleanup),
            'lod_counts': {lod.name: count for lod, count in lod_counts.items()},
            'state_counts': {state.name: count for state, count in state_counts.items()},
            'total_volume': total_volume
        }


def spawn_debris_from_fracture(
    manager: DebrisManager,
    chunks: List[Chunk],
    center_velocity: Vec3,
    spread_factor: float = 1.0,
    parent_id: Optional[int] = None,
    generation: int = 0,
    lifetime: float = DEBRIS_LIFETIME
) -> List[Debris]:
    """
    Spawn debris from fracture results with outward velocity.

    Args:
        manager: Debris manager.
        chunks: Fractured chunks.
        center_velocity: Base velocity for all debris.
        spread_factor: Multiplier for outward spread velocity.
        parent_id: ID of source object.
        generation: Fracture generation.
        lifetime: Debris lifetime.

    Returns:
        List of spawned debris.
    """
    if not chunks:
        return []

    # Calculate center point
    cx = sum(c.centroid[0] for c in chunks) / len(chunks)
    cy = sum(c.centroid[1] for c in chunks) / len(chunks)
    cz = sum(c.centroid[2] for c in chunks) / len(chunks)
    center = (cx, cy, cz)

    params_list = []

    for chunk in chunks:
        # Calculate outward velocity
        outward = vec3_sub(chunk.centroid, center)
        outward_length = vec3_length(outward)

        if outward_length > 0:
            outward = vec3_mul(outward, spread_factor / outward_length)
        else:
            outward = (0.0, 0.0, 0.0)

        velocity = vec3_add(center_velocity, outward)

        # Random angular velocity
        import random
        angular = (
            random.uniform(DEBRIS_ANGULAR_VELOCITY_MIN, DEBRIS_ANGULAR_VELOCITY_MAX),
            random.uniform(DEBRIS_ANGULAR_VELOCITY_MIN, DEBRIS_ANGULAR_VELOCITY_MAX),
            random.uniform(DEBRIS_ANGULAR_VELOCITY_MIN, DEBRIS_ANGULAR_VELOCITY_MAX)
        )

        # Importance based on volume
        importance = min(1.0, chunk.volume * DEBRIS_IMPORTANCE_VOLUME_MULTIPLIER)

        params_list.append(DebrisSpawnParams(
            chunk=chunk,
            position=chunk.centroid,
            velocity=velocity,
            angular_velocity=angular,
            lifetime=lifetime,
            importance=importance,
            parent_id=parent_id,
            generation=generation
        ))

    return manager.spawn_debris_batch(params_list)
