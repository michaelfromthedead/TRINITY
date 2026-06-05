"""
Priority computation system for world partition streaming.

Computes streaming priorities based on:
- Distance to streaming sources
- Player velocity prediction
- LOD level bonuses
- Visibility factors

Also handles cell activation triggers like clipmap updates and foliage merging.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Union,
)
import time


class ActivationType(Enum):
    """Types of cell activation events."""

    CLIPMAP_UPDATE = auto()      # Update clipmap LOD structure
    FOLIAGE_MERGE = auto()       # Merge foliage instances
    PHYSICS_ENABLE = auto()      # Enable physics simulation
    AI_ENABLE = auto()           # Enable AI processing
    AUDIO_ENABLE = auto()        # Enable audio sources
    LIGHTING_UPDATE = auto()     # Update lighting data
    NAVMESH_LINK = auto()        # Link to navmesh
    FULL_ACTIVATION = auto()     # Complete activation


@dataclass
class ActivationEvent:
    """Event triggered during cell activation."""

    activation_type: ActivationType
    cell_id: Tuple[int, int]
    timestamp: float = field(default_factory=time.time)
    priority: int = 0
    data: Optional[Any] = None

    def __lt__(self, other: "ActivationEvent") -> bool:
        """Compare by priority for sorting."""
        return self.priority > other.priority  # Higher priority first


@dataclass
class PriorityFactors:
    """
    Factors used in priority computation.

    Each factor contributes to the final priority score.
    """

    # Distance-based factors
    distance: float = 0.0              # Distance to nearest streaming source
    distance_normalized: float = 0.0   # Distance normalized to [0, 1]

    # Velocity-based factors
    velocity_dot: float = 0.0          # Dot product with velocity direction
    predicted_distance: float = 0.0    # Predicted distance after time delta

    # Visibility factors
    is_visible: bool = False           # Currently visible
    visibility_score: float = 0.0      # Visibility contribution

    # LOD factors
    lod_level: int = 0                 # Current LOD level
    lod_bonus: float = 0.0             # LOD-based priority bonus

    # Base priority
    base_priority: float = 0.0         # Base priority from chunk config

    @property
    def total_distance_factor(self) -> float:
        """Combined distance factor."""
        return 1.0 - self.distance_normalized

    @property
    def total_velocity_factor(self) -> float:
        """Combined velocity factor."""
        return max(0.0, self.velocity_dot)


@dataclass
class CellPriority:
    """Computed priority for a cell."""

    cell_id: Tuple[int, int]
    priority: float = 0.0
    factors: PriorityFactors = field(default_factory=PriorityFactors)
    compute_time: float = 0.0

    def __lt__(self, other: "CellPriority") -> bool:
        """Compare by priority for sorting."""
        return self.priority > other.priority  # Higher priority first

    def __le__(self, other: "CellPriority") -> bool:
        return self.priority >= other.priority

    def __gt__(self, other: "CellPriority") -> bool:
        return self.priority < other.priority

    def __ge__(self, other: "CellPriority") -> bool:
        return self.priority <= other.priority


@dataclass
class StreamingSource:
    """A source that drives streaming decisions."""

    position: Tuple[float, float] = (0.0, 0.0)
    velocity: Tuple[float, float] = (0.0, 0.0)
    load_radius: float = 1000.0
    priority_multiplier: float = 1.0
    is_active: bool = True
    source_id: str = ""

    @property
    def speed(self) -> float:
        """Get the current speed."""
        return math.sqrt(self.velocity[0] ** 2 + self.velocity[1] ** 2)

    def predicted_position(self, delta_time: float) -> Tuple[float, float]:
        """Get predicted position after delta_time."""
        return (
            self.position[0] + self.velocity[0] * delta_time,
            self.position[1] + self.velocity[1] * delta_time,
        )


@dataclass
class PriorityConfig:
    """Configuration for priority computation."""

    # Weight factors
    distance_weight: float = 1.0
    velocity_weight: float = 0.5
    visibility_weight: float = 0.3
    lod_weight: float = 0.2
    base_priority_weight: float = 1.0

    # Distance settings
    max_priority_distance: float = 100.0    # Distance at which priority = 1.0
    min_priority_distance: float = 2000.0   # Distance at which priority = 0.0

    # Velocity prediction
    prediction_time: float = 2.0            # Seconds to predict ahead
    velocity_threshold: float = 1.0         # Min velocity for prediction

    # LOD settings
    lod_bonus_per_level: float = 0.1        # Priority bonus per LOD level

    # Clamping
    min_priority: float = 0.0
    max_priority: float = 100.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.max_priority_distance <= 0:
            raise ValueError("max_priority_distance must be positive")
        if self.min_priority_distance <= self.max_priority_distance:
            raise ValueError(
                "min_priority_distance must be greater than max_priority_distance"
            )
        if self.prediction_time < 0:
            raise ValueError("prediction_time must be non-negative")


class PriorityComputer:
    """
    Computes streaming priorities for cells.

    Uses configurable weights to combine distance, velocity, visibility,
    and LOD factors into a final priority score.
    """

    def __init__(self, config: Optional[PriorityConfig] = None) -> None:
        self.config = config or PriorityConfig()
        self._sources: List[StreamingSource] = []
        self._cache: Dict[Tuple[int, int], CellPriority] = {}
        self._cache_valid = False

    def add_source(self, source: StreamingSource) -> None:
        """Add a streaming source."""
        self._sources.append(source)
        self._cache_valid = False

    def remove_source(self, source_id: str) -> bool:
        """Remove a streaming source by ID."""
        for i, source in enumerate(self._sources):
            if source.source_id == source_id:
                self._sources.pop(i)
                self._cache_valid = False
                return True
        return False

    def clear_sources(self) -> None:
        """Remove all streaming sources."""
        self._sources.clear()
        self._cache_valid = False

    def update_source(
        self,
        source_id: str,
        position: Optional[Tuple[float, float]] = None,
        velocity: Optional[Tuple[float, float]] = None,
    ) -> bool:
        """Update a streaming source."""
        for source in self._sources:
            if source.source_id == source_id:
                if position is not None:
                    source.position = position
                if velocity is not None:
                    source.velocity = velocity
                self._cache_valid = False
                return True
        return False

    def compute_priority(
        self,
        cell_x: int,
        cell_y: int,
        cell_center: Tuple[float, float],
        cell_size: float = 256.0,
        lod_level: int = 0,
        base_priority: float = 50.0,
        is_visible: bool = False,
    ) -> CellPriority:
        """
        Compute priority for a cell.

        Args:
            cell_x: Cell X coordinate
            cell_y: Cell Y coordinate
            cell_center: World-space center of the cell
            cell_size: Size of the cell in world units
            lod_level: Current LOD level
            base_priority: Base priority from chunk configuration
            is_visible: Whether the cell is currently visible

        Returns:
            CellPriority with computed priority and factors
        """
        start_time = time.time()
        cell_id = (cell_x, cell_y)

        factors = PriorityFactors(base_priority=base_priority)

        if not self._sources:
            # No sources, use only base priority
            priority = base_priority * self.config.base_priority_weight
            return CellPriority(
                cell_id=cell_id,
                priority=self._clamp_priority(priority),
                factors=factors,
                compute_time=(time.time() - start_time) * 1000,
            )

        # Find nearest active source
        min_distance = float("inf")
        best_source: Optional[StreamingSource] = None
        best_velocity_dot = 0.0

        for source in self._sources:
            if not source.is_active:
                continue

            # Calculate distance
            dx = cell_center[0] - source.position[0]
            dy = cell_center[1] - source.position[1]
            distance = math.sqrt(dx * dx + dy * dy)

            if distance < min_distance:
                min_distance = distance
                best_source = source

                # Calculate velocity dot product
                if source.speed > self.config.velocity_threshold:
                    # Normalize direction to cell
                    dir_len = distance if distance > 0.001 else 0.001
                    dir_x = dx / dir_len
                    dir_y = dy / dir_len

                    # Normalize velocity
                    vel_len = source.speed
                    vel_x = source.velocity[0] / vel_len
                    vel_y = source.velocity[1] / vel_len

                    best_velocity_dot = dir_x * vel_x + dir_y * vel_y

        factors.distance = min_distance

        # Normalize distance
        distance_range = (
            self.config.min_priority_distance - self.config.max_priority_distance
        )
        if distance_range > 0:
            normalized = (min_distance - self.config.max_priority_distance) / distance_range
            factors.distance_normalized = max(0.0, min(1.0, normalized))
        else:
            factors.distance_normalized = 0.0

        # Velocity prediction
        factors.velocity_dot = best_velocity_dot
        if best_source and best_velocity_dot > 0:
            predicted_pos = best_source.predicted_position(self.config.prediction_time)
            dx = cell_center[0] - predicted_pos[0]
            dy = cell_center[1] - predicted_pos[1]
            factors.predicted_distance = math.sqrt(dx * dx + dy * dy)

        # Visibility
        factors.is_visible = is_visible
        factors.visibility_score = 1.0 if is_visible else 0.0

        # LOD
        factors.lod_level = lod_level
        factors.lod_bonus = lod_level * self.config.lod_bonus_per_level

        # Compute final priority
        priority = self._compute_weighted_priority(factors)

        # Apply source multiplier
        if best_source:
            priority *= best_source.priority_multiplier

        return CellPriority(
            cell_id=cell_id,
            priority=self._clamp_priority(priority),
            factors=factors,
            compute_time=(time.time() - start_time) * 1000,
        )

    def _compute_weighted_priority(self, factors: PriorityFactors) -> float:
        """Compute weighted priority from factors."""
        config = self.config

        # Distance contribution (inverted: closer = higher priority)
        distance_contrib = (
            factors.total_distance_factor * config.distance_weight
        )

        # Velocity contribution (moving toward cell = higher priority)
        velocity_contrib = (
            factors.total_velocity_factor * config.velocity_weight
        )

        # Visibility contribution
        visibility_contrib = (
            factors.visibility_score * config.visibility_weight
        )

        # LOD contribution
        lod_contrib = factors.lod_bonus * config.lod_weight

        # Base priority contribution
        base_contrib = (
            factors.base_priority / 100.0 * config.base_priority_weight
        )

        # Combine all factors
        total_weight = (
            config.distance_weight
            + config.velocity_weight
            + config.visibility_weight
            + config.lod_weight
            + config.base_priority_weight
        )

        if total_weight > 0:
            normalized_priority = (
                distance_contrib
                + velocity_contrib
                + visibility_contrib
                + lod_contrib
                + base_contrib
            ) / total_weight
        else:
            normalized_priority = 0.5

        # Scale to priority range
        return (
            config.min_priority
            + normalized_priority * (config.max_priority - config.min_priority)
        )

    def _clamp_priority(self, priority: float) -> float:
        """Clamp priority to configured range."""
        return max(
            self.config.min_priority,
            min(self.config.max_priority, priority),
        )

    def compute_priorities_batch(
        self,
        cells: List[Tuple[int, int, Tuple[float, float]]],
        cell_size: float = 256.0,
        lod_levels: Optional[Dict[Tuple[int, int], int]] = None,
        base_priorities: Optional[Dict[Tuple[int, int], float]] = None,
        visible_cells: Optional[Set[Tuple[int, int]]] = None,
    ) -> List[CellPriority]:
        """
        Compute priorities for multiple cells.

        Args:
            cells: List of (cell_x, cell_y, center) tuples
            cell_size: Size of cells
            lod_levels: Optional dict mapping cell_id to LOD level
            base_priorities: Optional dict mapping cell_id to base priority
            visible_cells: Optional set of visible cell IDs

        Returns:
            List of CellPriority objects, sorted by priority (highest first)
        """
        lod_levels = lod_levels or {}
        base_priorities = base_priorities or {}
        visible_cells = visible_cells or set()

        priorities = []
        for cell_x, cell_y, center in cells:
            cell_id = (cell_x, cell_y)
            priority = self.compute_priority(
                cell_x=cell_x,
                cell_y=cell_y,
                cell_center=center,
                cell_size=cell_size,
                lod_level=lod_levels.get(cell_id, 0),
                base_priority=base_priorities.get(cell_id, 50.0),
                is_visible=cell_id in visible_cells,
            )
            priorities.append(priority)

        # Sort by priority (highest first)
        priorities.sort()
        return priorities

    def invalidate_cache(self) -> None:
        """Invalidate the priority cache."""
        self._cache.clear()
        self._cache_valid = False

    def get_sources(self) -> List[StreamingSource]:
        """Get all streaming sources."""
        return list(self._sources)


# =============================================================================
# CELL ACTIVATION TRIGGERS
# =============================================================================


# Type alias for activation callbacks
ActivationCallback = Callable[[ActivationEvent], None]


class CellActivationTrigger:
    """
    Manages cell activation triggers.

    Generates activation events when cells transition to ACTIVATED state,
    such as clipmap updates and foliage instance merging.
    """

    def __init__(self) -> None:
        self._callbacks: Dict[ActivationType, List[ActivationCallback]] = {
            t: [] for t in ActivationType
        }
        self._pending_events: List[ActivationEvent] = []
        self._processed_count = 0

    def register_callback(
        self,
        activation_type: ActivationType,
        callback: ActivationCallback,
    ) -> None:
        """Register a callback for an activation type."""
        self._callbacks[activation_type].append(callback)

    def unregister_callback(
        self,
        activation_type: ActivationType,
        callback: ActivationCallback,
    ) -> bool:
        """Unregister a callback."""
        if callback in self._callbacks[activation_type]:
            self._callbacks[activation_type].remove(callback)
            return True
        return False

    def trigger_activation(
        self,
        cell_id: Tuple[int, int],
        activation_types: Optional[List[ActivationType]] = None,
        priority: int = 0,
        data: Optional[Any] = None,
    ) -> List[ActivationEvent]:
        """
        Trigger activation events for a cell.

        Args:
            cell_id: Cell being activated
            activation_types: Types of activation to trigger (default: all)
            priority: Event priority
            data: Optional data to attach to events

        Returns:
            List of created activation events
        """
        if activation_types is None:
            activation_types = list(ActivationType)

        events = []
        timestamp = time.time()

        for act_type in activation_types:
            event = ActivationEvent(
                activation_type=act_type,
                cell_id=cell_id,
                timestamp=timestamp,
                priority=priority,
                data=data,
            )
            events.append(event)
            self._pending_events.append(event)

        # Sort pending events by priority
        self._pending_events.sort()

        return events

    def process_events(self, max_events: Optional[int] = None) -> int:
        """
        Process pending activation events.

        Args:
            max_events: Maximum events to process (None = all)

        Returns:
            Number of events processed
        """
        processed = 0
        events_to_process = (
            self._pending_events[:max_events]
            if max_events
            else list(self._pending_events)
        )

        for event in events_to_process:
            for callback in self._callbacks[event.activation_type]:
                callback(event)
            processed += 1

        # Remove processed events
        if max_events:
            self._pending_events = self._pending_events[max_events:]
        else:
            self._pending_events.clear()

        self._processed_count += processed
        return processed

    def trigger_clipmap_update(
        self,
        cell_id: Tuple[int, int],
        clipmap_data: Optional[Any] = None,
        priority: int = 100,
    ) -> ActivationEvent:
        """Trigger a clipmap update event."""
        event = ActivationEvent(
            activation_type=ActivationType.CLIPMAP_UPDATE,
            cell_id=cell_id,
            priority=priority,
            data=clipmap_data,
        )
        self._pending_events.append(event)
        self._pending_events.sort()
        return event

    def trigger_foliage_merge(
        self,
        cell_id: Tuple[int, int],
        foliage_data: Optional[Any] = None,
        priority: int = 50,
    ) -> ActivationEvent:
        """Trigger a foliage instance merge event."""
        event = ActivationEvent(
            activation_type=ActivationType.FOLIAGE_MERGE,
            cell_id=cell_id,
            priority=priority,
            data=foliage_data,
        )
        self._pending_events.append(event)
        self._pending_events.sort()
        return event

    def trigger_full_activation(
        self,
        cell_id: Tuple[int, int],
        data: Optional[Any] = None,
    ) -> List[ActivationEvent]:
        """Trigger all activation events for a cell."""
        return self.trigger_activation(
            cell_id=cell_id,
            activation_types=[
                ActivationType.CLIPMAP_UPDATE,
                ActivationType.FOLIAGE_MERGE,
                ActivationType.PHYSICS_ENABLE,
                ActivationType.AI_ENABLE,
                ActivationType.AUDIO_ENABLE,
                ActivationType.LIGHTING_UPDATE,
                ActivationType.NAVMESH_LINK,
                ActivationType.FULL_ACTIVATION,
            ],
            priority=100,
            data=data,
        )

    @property
    def pending_count(self) -> int:
        """Get number of pending events."""
        return len(self._pending_events)

    @property
    def processed_count(self) -> int:
        """Get total number of processed events."""
        return self._processed_count

    def clear_pending(self) -> int:
        """Clear all pending events."""
        count = len(self._pending_events)
        self._pending_events.clear()
        return count

    def get_pending_by_type(
        self,
        activation_type: ActivationType,
    ) -> List[ActivationEvent]:
        """Get pending events of a specific type."""
        return [
            e for e in self._pending_events
            if e.activation_type == activation_type
        ]

    def get_pending_for_cell(
        self,
        cell_id: Tuple[int, int],
    ) -> List[ActivationEvent]:
        """Get pending events for a specific cell."""
        return [e for e in self._pending_events if e.cell_id == cell_id]


__all__ = [
    # Priority computation
    "PriorityConfig",
    "PriorityFactors",
    "CellPriority",
    "StreamingSource",
    "PriorityComputer",
    # Activation triggers
    "ActivationType",
    "ActivationEvent",
    "ActivationCallback",
    "CellActivationTrigger",
]
