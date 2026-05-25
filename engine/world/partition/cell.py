"""
Streaming cell implementation for world partition.

Provides the StreamingCell class that represents a unit of world
content that can be independently loaded and unloaded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from engine.core.math.vec import Vec3
from engine.core.math.geometry import AABB
from engine.world.partition.constants import (
    DEFAULT_CELL_BOUNDS_MAX,
    CELL_BASE_MEMORY_BYTES,
    ACTOR_MEMORY_ESTIMATE_BYTES,
    FOLIAGE_MEMORY_ESTIMATE_BYTES,
    HALF_MULTIPLIER,
)


class CellState(Enum):
    """Loading state of a streaming cell."""
    UNLOADED = auto()   # No data in memory
    LOADING = auto()    # Async load in progress
    LOADED = auto()     # Data loaded but not active
    ACTIVATED = auto()  # Fully active and ticking
    UNLOADING = auto()  # Async unload in progress


@dataclass(frozen=True)
class CellCoord:
    """Immutable 2D cell coordinate."""
    x: int
    y: int

    def __add__(self, other: CellCoord) -> CellCoord:
        """Add two coordinates."""
        return CellCoord(self.x + other.x, self.y + other.y)

    def __sub__(self, other: CellCoord) -> CellCoord:
        """Subtract two coordinates."""
        return CellCoord(self.x - other.x, self.y - other.y)

    def distance_manhattan(self, other: CellCoord) -> int:
        """Calculate Manhattan distance to another coordinate."""
        return abs(self.x - other.x) + abs(self.y - other.y)

    def distance_chebyshev(self, other: CellCoord) -> int:
        """Calculate Chebyshev (chessboard) distance to another coordinate."""
        return max(abs(self.x - other.x), abs(self.y - other.y))

    def as_tuple(self) -> Tuple[int, int]:
        """Convert to tuple."""
        return (self.x, self.y)

    @classmethod
    def from_tuple(cls, t: Tuple[int, int]) -> CellCoord:
        """Create from tuple."""
        return cls(t[0], t[1])


@dataclass
class CellActor:
    """
    Reference to an actor within a cell.

    Stores minimal data needed for streaming decisions.
    """
    id: str = ""
    name: str = ""
    position: Vec3 = field(default_factory=lambda: Vec3(0, 0, 0))
    priority: int = 0
    persistent: bool = False
    tags: Set[str] = field(default_factory=set)
    data: Any = None  # Reference to actual actor data


@dataclass
class StreamingCell:
    """
    A streaming cell in the world partition grid.

    Cells are the fundamental unit of streaming - they can be
    independently loaded, activated, deactivated, and unloaded
    based on distance from streaming sources.
    """
    coord: CellCoord = field(default_factory=lambda: CellCoord(0, 0))

    # Bounds in world space
    bounds_min: Vec3 = field(default_factory=lambda: Vec3(0, 0, 0))
    bounds_max: Vec3 = field(default_factory=lambda: Vec3(
        DEFAULT_CELL_BOUNDS_MAX, DEFAULT_CELL_BOUNDS_MAX, DEFAULT_CELL_BOUNDS_MAX
    ))

    # Current state
    state: CellState = CellState.UNLOADED
    load_progress: float = 0.0  # 0.0 to 1.0 during loading

    # Content
    actors: List[CellActor] = field(default_factory=list)
    foliage: List[Any] = field(default_factory=list)  # Foliage instance data

    # HLOD (Hierarchical Level of Detail) proxy for this cell
    hlod_proxy: Any = None

    # Loading priority (higher = loads first)
    priority: int = 0

    # Timestamps
    last_load_time: float = 0.0
    last_activate_time: float = 0.0

    # Callbacks
    _on_load_callbacks: List[Callable[["StreamingCell"], None]] = field(
        default_factory=list, repr=False
    )
    _on_unload_callbacks: List[Callable[["StreamingCell"], None]] = field(
        default_factory=list, repr=False
    )
    _on_activate_callbacks: List[Callable[["StreamingCell"], None]] = field(
        default_factory=list, repr=False
    )
    _on_deactivate_callbacks: List[Callable[["StreamingCell"], None]] = field(
        default_factory=list, repr=False
    )

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def bounds(self) -> AABB:
        """Get the cell bounds as an AABB."""
        return AABB(self.bounds_min, self.bounds_max)

    @property
    def center(self) -> Vec3:
        """Get the center of the cell."""
        return (self.bounds_min + self.bounds_max) * HALF_MULTIPLIER

    @property
    def size(self) -> Vec3:
        """Get the size of the cell."""
        return self.bounds_max - self.bounds_min

    @property
    def is_loaded(self) -> bool:
        """Check if the cell is loaded or activated."""
        return self.state in (CellState.LOADED, CellState.ACTIVATED)

    @property
    def is_active(self) -> bool:
        """Check if the cell is activated."""
        return self.state == CellState.ACTIVATED

    @property
    def is_loading(self) -> bool:
        """Check if the cell is currently loading."""
        return self.state == CellState.LOADING

    @property
    def is_unloaded(self) -> bool:
        """Check if the cell is unloaded."""
        return self.state == CellState.UNLOADED

    def load(self) -> bool:
        """
        Begin loading the cell.

        Returns:
            True if load started successfully, False if already loading/loaded.
        """
        if self.state != CellState.UNLOADED:
            return False

        self.state = CellState.LOADING
        self.load_progress = 0.0

        return True

    def complete_load(self, timestamp: float = 0.0) -> bool:
        """
        Complete the cell loading process.

        Args:
            timestamp: Current time for tracking.

        Returns:
            True if load completed, False if not in loading state.
        """
        if self.state != CellState.LOADING:
            return False

        self.state = CellState.LOADED
        self.load_progress = 1.0
        self.last_load_time = timestamp

        for callback in self._on_load_callbacks:
            callback(self)

        return True

    def unload(self) -> bool:
        """
        Begin unloading the cell.

        Returns:
            True if unload started, False if not in loadable state.
        """
        if self.state not in (CellState.LOADED, CellState.ACTIVATED):
            return False

        if self.state == CellState.ACTIVATED:
            self.deactivate()

        self.state = CellState.UNLOADING

        return True

    def complete_unload(self) -> bool:
        """
        Complete the cell unloading process.

        Returns:
            True if unload completed, False if not in unloading state.
        """
        if self.state != CellState.UNLOADING:
            return False

        for callback in self._on_unload_callbacks:
            callback(self)

        # Clear content
        self.actors.clear()
        self.foliage.clear()
        self.hlod_proxy = None

        self.state = CellState.UNLOADED
        self.load_progress = 0.0

        return True

    def activate(self, timestamp: float = 0.0) -> bool:
        """
        Activate the cell for gameplay.

        Args:
            timestamp: Current time for tracking.

        Returns:
            True if activation succeeded.
        """
        if self.state != CellState.LOADED:
            return False

        self.state = CellState.ACTIVATED
        self.last_activate_time = timestamp

        for callback in self._on_activate_callbacks:
            callback(self)

        return True

    def deactivate(self) -> bool:
        """
        Deactivate the cell (but keep it loaded).

        Returns:
            True if deactivation succeeded.
        """
        if self.state != CellState.ACTIVATED:
            return False

        for callback in self._on_deactivate_callbacks:
            callback(self)

        self.state = CellState.LOADED

        return True

    def update_load_progress(self, progress: float) -> None:
        """
        Update the loading progress.

        Args:
            progress: Progress value from 0.0 to 1.0.
        """
        self.load_progress = max(0.0, min(1.0, progress))

    def add_actor(self, actor: CellActor) -> None:
        """
        Add an actor to the cell.

        Args:
            actor: Actor to add.
        """
        self.actors.append(actor)

    def remove_actor(self, actor_id: str) -> bool:
        """
        Remove an actor from the cell by ID.

        Args:
            actor_id: ID of the actor to remove.

        Returns:
            True if found and removed, False otherwise.
        """
        for i, actor in enumerate(self.actors):
            if actor.id == actor_id:
                self.actors.pop(i)
                return True
        return False

    def get_actor(self, actor_id: str) -> Optional[CellActor]:
        """
        Get an actor by ID.

        Args:
            actor_id: ID to search for.

        Returns:
            The actor if found, None otherwise.
        """
        for actor in self.actors:
            if actor.id == actor_id:
                return actor
        return None

    def get_actors_by_tag(self, tag: str) -> List[CellActor]:
        """
        Get all actors with a specific tag.

        Args:
            tag: Tag to search for.

        Returns:
            List of actors with the tag.
        """
        return [a for a in self.actors if tag in a.tags]

    def get_persistent_actors(self) -> List[CellActor]:
        """Get all persistent actors in the cell."""
        return [a for a in self.actors if a.persistent]

    def contains_point(self, point: Vec3) -> bool:
        """
        Check if a point is within the cell bounds.

        Args:
            point: Point to check.

        Returns:
            True if the point is within bounds.
        """
        return (
            self.bounds_min.x <= point.x <= self.bounds_max.x
            and self.bounds_min.y <= point.y <= self.bounds_max.y
            and self.bounds_min.z <= point.z <= self.bounds_max.z
        )

    def overlaps(self, other_min: Vec3, other_max: Vec3) -> bool:
        """
        Check if this cell overlaps with another AABB.

        Args:
            other_min: Minimum corner of other AABB.
            other_max: Maximum corner of other AABB.

        Returns:
            True if the bounds overlap.
        """
        return (
            self.bounds_min.x <= other_max.x and self.bounds_max.x >= other_min.x
            and self.bounds_min.y <= other_max.y and self.bounds_max.y >= other_min.y
            and self.bounds_min.z <= other_max.z and self.bounds_max.z >= other_min.z
        )

    def distance_to_point(self, point: Vec3) -> float:
        """
        Calculate the distance from the cell center to a point.

        Args:
            point: Point to measure to.

        Returns:
            Distance to the point.
        """
        return self.center.distance(point)

    def distance_squared_to_point(self, point: Vec3) -> float:
        """
        Calculate the squared distance from the cell center to a point.

        Args:
            point: Point to measure to.

        Returns:
            Squared distance to the point.
        """
        delta = self.center - point
        return delta.length_squared()

    def get_neighbors(self, grid: Any = None) -> List[CellCoord]:
        """
        Get coordinates of neighboring cells (8-connected).

        Args:
            grid: Optional grid reference for bounds checking.

        Returns:
            List of neighbor coordinates.
        """
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue

                neighbor = CellCoord(self.coord.x + dx, self.coord.y + dy)

                if grid is not None:
                    # Bounds check against grid
                    if grid.is_valid_coord(neighbor.x, neighbor.y):
                        neighbors.append(neighbor)
                else:
                    neighbors.append(neighbor)

        return neighbors

    def get_cardinal_neighbors(self) -> List[CellCoord]:
        """
        Get coordinates of cardinal (4-connected) neighbor cells.

        Returns:
            List of cardinal neighbor coordinates.
        """
        return [
            CellCoord(self.coord.x, self.coord.y - 1),  # North
            CellCoord(self.coord.x + 1, self.coord.y),  # East
            CellCoord(self.coord.x, self.coord.y + 1),  # South
            CellCoord(self.coord.x - 1, self.coord.y),  # West
        ]

    def on_load(self, callback: Callable[["StreamingCell"], None]) -> None:
        """Register a callback for when the cell is loaded."""
        self._on_load_callbacks.append(callback)

    def on_unload(self, callback: Callable[["StreamingCell"], None]) -> None:
        """Register a callback for when the cell is unloaded."""
        self._on_unload_callbacks.append(callback)

    def on_activate(self, callback: Callable[["StreamingCell"], None]) -> None:
        """Register a callback for when the cell is activated."""
        self._on_activate_callbacks.append(callback)

    def on_deactivate(self, callback: Callable[["StreamingCell"], None]) -> None:
        """Register a callback for when the cell is deactivated."""
        self._on_deactivate_callbacks.append(callback)

    def clear_callbacks(self) -> None:
        """Clear all registered callbacks."""
        self._on_load_callbacks.clear()
        self._on_unload_callbacks.clear()
        self._on_activate_callbacks.clear()
        self._on_deactivate_callbacks.clear()

    def get_memory_estimate(self) -> int:
        """
        Estimate memory usage of cell content in bytes.

        Returns:
            Estimated memory usage.
        """
        # Base estimate
        estimate = CELL_BASE_MEMORY_BYTES

        # Actors
        estimate += len(self.actors) * ACTOR_MEMORY_ESTIMATE_BYTES

        # Foliage
        estimate += len(self.foliage) * FOLIAGE_MEMORY_ESTIMATE_BYTES

        return estimate

    def serialize(self) -> Dict[str, Any]:
        """
        Serialize cell data for persistence.

        Returns:
            Dictionary of serialized data.
        """
        return {
            "coord": [self.coord.x, self.coord.y],
            "bounds_min": [self.bounds_min.x, self.bounds_min.y, self.bounds_min.z],
            "bounds_max": [self.bounds_max.x, self.bounds_max.y, self.bounds_max.z],
            "priority": self.priority,
            "actors": [
                {
                    "id": a.id,
                    "name": a.name,
                    "position": [a.position.x, a.position.y, a.position.z],
                    "priority": a.priority,
                    "persistent": a.persistent,
                    "tags": list(a.tags),
                }
                for a in self.actors
            ],
            "metadata": self.metadata,
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "StreamingCell":
        """
        Deserialize cell data.

        Args:
            data: Dictionary of serialized data.

        Returns:
            Deserialized StreamingCell.
        """
        actors = [
            CellActor(
                id=a["id"],
                name=a["name"],
                position=Vec3(*a["position"]),
                priority=a.get("priority", 0),
                persistent=a.get("persistent", False),
                tags=set(a.get("tags", [])),
            )
            for a in data.get("actors", [])
        ]

        return cls(
            coord=CellCoord(*data["coord"]),
            bounds_min=Vec3(*data["bounds_min"]),
            bounds_max=Vec3(*data["bounds_max"]),
            priority=data.get("priority", 0),
            actors=actors,
            metadata=data.get("metadata", {}),
        )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"StreamingCell(coord={self.coord}, state={self.state.name}, "
            f"actors={len(self.actors)})"
        )
